// A live particle as drawn: centre, radius, rotation and premultiplied
// colour (alpha 0: emissive). Colour, opacity and size follow the effect
// over the particle's life; media (smoke, dust) are lit by the sun (past
// the hills) and the sky and fogged, emissive particles add their colour
// and fade in fog. Shared by the draw's vertex shader and the froxels'
// binning, so both see the same puff. After particles_common.glsl.
#include "photometry.glsl"

layout(set = 0, binding = 7) uniform sampler2D sunShadowTex;

float sunAt(vec3 p) {
    if ((u.flags.x & 1u) == 0u) {
        return 1.0;
    }
    float s = textureLod(sunShadowTex, (p.xz / u.map.x + 0.5) * u.map.yz, 0.0).r;
    return 1.0 - smoothstep(0.0, max(1.0, 0.8 * u.map.x), s - p.y);
}

struct Puff {
    vec3 centre;
    float radius;
    // The quad's rotation (a random one per particle, so overlapping
    // puffs do not line up).
    float angle;
    vec4 color;
};

Puff puffOf(uint slot) {
    Particle p = particles[slot];
    uint e = p.info.x * 5u;
    float t = clamp(p.posAge.w / max(p.velLife.w, 1e-6), 0.0, 1.0);
    vec4 shape = u.effects[e + 2u];
    vec4 physics = u.effects[e + 4u];
    Puff f;
    f.centre = p.posAge.xyz;
    f.radius = 0.5 * mix(shape.x, shape.y, t) * uintBitsToFloat(p.info.z);
    f.angle = float(p.info.y & 0xffffu) * (6.2831853 / 65536.0);
    vec4 color = mix(u.effects[e], u.effects[e + 1u], t);
    color.a *= physics.w > 0.0 ? clamp(t / physics.w, 0.0, 1.0) : 1.0;
    // Each particle a little brighter or darker than its neighbours.
    color.rgb *= 0.8 + 0.4 * float((p.info.y >> 16) & 0xffu) / 255.0;
    float dist = distance(p.posAge.xyz, u.camPos.xyz);
    float fog = u.fog.w > 0.0 ? 1.0 - exp(-pow(dist * u.fog.w, 1.5)) : 0.0;
    if (physics.z > 0.5) {
        // The bridge: old additive linear value to exposed physical
        // radiance of a unit luminance? No: emission is radiance directly.
        f.color = vec4(color.rgb * color.a * (1.0 - fog) * u.photometry.y, 0.0);
    } else {
        // The media's lighting is still the old expression (re-authored with
        // the emitters): its sun/sky coefficients through the bridge, the
        // clustered lights (physical illuminance) back into those units.
        vec3 sky = vec3(0.45, 0.55, 0.7);
        vec3 light = vec3(1.0, 0.95, 0.85) * 1.1 * sunAt(p.posAge.xyz) + sky * 0.55;
        if (p.info.w != 0u) {
            light += unpackParticleLight(p.info.w) / AE_PI;
        }
        vec3 lit = mix(color.rgb * light * u.photometry.y, u.fog.rgb, fog);
        f.color = vec4(lit * color.a, color.a);
    }
    return f;
}
