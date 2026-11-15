#version 460
// One camera-facing quad per live particle, in sorted order (far first).
// Colour, opacity and size follow the effect over the particle's life;
// media (smoke, dust) are lit by the sun (past the hills) and the sky and
// fogged, emissive particles add their colour (alpha 0) and fade in fog.
#include "particles_common.glsl"

layout(set = 0, binding = 7) uniform sampler2D sunShadowTex;

layout(location = 0) out vec2 outCorner;
layout(location = 1) out vec4 outColor;   // premultiplied
layout(location = 2) out vec2 outDepth;   // centre's view depth, radius

const vec2 CORNERS[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

float sunAt(vec3 p) {
    if ((u.flags.x & 1u) == 0u) {
        return 1.0;
    }
    float s = textureLod(sunShadowTex, (p.xz / u.map.x + 0.5) * u.map.yz, 0.0).r;
    return 1.0 - smoothstep(0.0, max(1.0, 0.8 * u.map.x), s - p.y);
}

void main() {
    uint slot = keys[gl_InstanceIndex] & 0xffffu;
    Particle p = particles[slot];
    uint e = p.info.x * 5u;
    float t = clamp(p.posAge.w / max(p.velLife.w, 1e-6), 0.0, 1.0);
    vec4 shape = u.effects[e + 2u];
    vec4 physics = u.effects[e + 4u];
    float radius = 0.5 * mix(shape.x, shape.y, t) * uintBitsToFloat(p.info.z);
    vec4 color = mix(u.effects[e], u.effects[e + 1u], t);
    color.a *= physics.w > 0.0 ? clamp(t / physics.w, 0.0, 1.0) : 1.0;
    // Each particle a little brighter or darker than its neighbours.
    color.rgb *= 0.8 + 0.4 * float((p.info.y >> 16) & 0xffu) / 255.0;

    // A random rotation per particle, so overlapping puffs do not line up.
    float angle = float(p.info.y & 0xffffu) * (6.2831853 / 65536.0);
    vec2 c = CORNERS[gl_VertexIndex];
    vec2 r = vec2(c.x * cos(angle) - c.y * sin(angle), c.x * sin(angle) + c.y * cos(angle));
    vec3 world = p.posAge.xyz + (u.camRight.xyz * r.x + u.camUp.xyz * r.y) * radius;
    gl_Position = u.viewProj * vec4(world, 1.0);

    float dist = distance(p.posAge.xyz, u.camPos.xyz);
    float fog = u.fog.w > 0.0 ? 1.0 - exp(-pow(dist * u.fog.w, 1.5)) : 0.0;
    if (physics.z > 0.5) {
        outColor = vec4(color.rgb * color.a * (1.0 - fog), 0.0);
    } else {
        vec3 sky = vec3(0.45, 0.55, 0.7);
        vec3 light = vec3(1.0, 0.95, 0.85) * 1.1 * sunAt(p.posAge.xyz) + sky * 0.55;
        vec3 lit = mix(color.rgb * light, u.fog.rgb, fog);
        outColor = vec4(lit * color.a, color.a);
    }
    outCorner = c;
    outDepth = vec2(dot(p.posAge.xyz - u.camPos.xyz, u.camForward.xyz), radius);
}
