// A live particle as drawn: centre, radius, rotation, premultiplied
// colour and whether it is emissive. Colour, opacity and size follow the
// effect over the particle's life; media (smoke, dust) are lit by the sun
// (past the hills) and the sky and fogged, emissive particles (fire,
// sparks, flashes) add their colour, cover what is behind them by the
// effect's cover (alpha; 0: they only add light), and fade in fog. Shared by the draw's vertex shader and the froxels'
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
    // Its light is a bloom source (the glow targets), whatever its alpha.
    bool emissive;
};

Puff puffOf(uint slot) {
    Particle p = particles[slot];
    uint e = p.info.x * 6u;
    float t = clamp(p.posAge.w / max(p.velLife.w, 1e-6), 0.0, 1.0);
    vec4 shape = u.effects[e + 2u];
    vec4 physics = u.effects[e + 4u];
    vec4 emission = u.effects[e + 5u];
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
    float emis = mix(emission.x, emission.y, t);
    f.emissive = physics.z > 0.5;
    if (f.emissive) {
        // Emissive: colour is the tint, emission its luminance (cd/m²);
        // it hides `cover` of its opacity's worth of what is behind it.
        // Both fade in fog (a far fire leaves the fogged ground, not a
        // dark blot).
        f.color = vec4(color.rgb * color.a * emis, color.a * emission.z) * (1.0 - fog);
    } else {
        // Media: albedo times the illuminance it scatters, as radiance; the
        // sun past the hills, half the sky (a volume has no normal), and the
        // clustered lights.
        bool wide = u.photometry.y > 0.5;
        vec3 sunE = aeUnitTint(AE_SUN_TINT, wide) * (u.photometry.z * sunAt(p.posAge.xyz));
        vec3 skyE = aeUnitTint(AE_SKY_TINT, wide) * (u.photometry.w * 0.5);
        vec3 light = (sunE + skyE) / AE_PI;
        if (p.info.w != 0u) {
            // Packed as illuminance times exposure: back to lux.
            light += unpackParticleLight(p.info.w) / max(u.photometry.x, 1e-9) / AE_PI;
        }
        vec3 lit = mix(color.rgb * light, u.fog.rgb, fog);
        f.color = vec4(lit * color.a, color.a);
    }
    return f;
}
