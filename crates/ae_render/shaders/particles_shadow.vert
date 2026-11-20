#version 460
// The particles' coloured shadows (src/shadow.rs): every live particle
// that is not emissive (smoke, dust; not fire, sparks or flashes) as a
// quad facing the sun, into one cascade of `shadows.color`. The instances
// are the simulation's list of casters (the keys from u.pool.x), not the
// camera's sorted list: smoke off screen still shades the view.
#include "particles_common.glsl"

layout(push_constant) uniform Push { uint cascade; } pc;

layout(location = 0) out vec2 outCorner;
layout(location = 1) flat out vec4 outColor; // colour, opacity
layout(location = 2) flat out vec2 outDepth; // centre's depth toward the sun, radius in depth units

// A triangle strip per instance: four vertices, not six.
const vec2 CORNERS[4] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1));

void main() {
    Particle p = particles[keys[u.pool.x + gl_InstanceIndex]];
    uint e = p.info.x * 6u;
    vec4 physics = u.effects[e + 4u];
    float t = clamp(p.posAge.w / max(p.velLife.w, 1e-6), 0.0, 1.0);
    vec4 shape = u.effects[e + 2u];
    float radius = 0.5 * mix(shape.x, shape.y, t) * uintBitsToFloat(p.info.z);
    vec4 color = mix(u.effects[e], u.effects[e + 1u], t);
    color.a *= physics.w > 0.0 ? clamp(t / physics.w, 0.0, 1.0) : 1.0;

    // The same random rotation as the camera's quads.
    float angle = float(p.info.y & 0xffffu) * (6.2831853 / 65536.0);
    vec2 c = CORNERS[gl_VertexIndex];
    vec2 r = vec2(c.x * cos(angle) - c.y * sin(angle), c.x * sin(angle) + c.y * cos(angle));
    uint k = pc.cascade;
    vec3 l = vec3(dot(u.shadowAxes[0].xyz, p.posAge.xyz), dot(u.shadowAxes[1].xyz, p.posAge.xyz), dot(u.shadowAxes[2].xyz, p.posAge.xyz));
    vec3 clip = (l - u.shadowOrigin[k].xyz) * u.shadowScale[k].xyz;
    clip.xy += r * radius * u.shadowScale[k].xy;
    // Depth clamped into the clip volume: a particle beyond the cascade's
    // reach toward the sun still shades what is in it.
    gl_Position = vec4(clip.xy, clamp(clip.z, 0.0, 1.0), 1.0);
    outCorner = c;
    outColor = color;
    outDepth = vec2(clip.z, radius * u.shadowScale[k].z);
}
