// The units' shadow map lookup (src/shadow.rs), shared by the terrain, the
// water surface and the units: UNIT_CASCADES square cascades side by side,
// depth growing toward the sun, read with a GREATER_OR_EQUAL comparison
// sampler (lit where the receiver is at least as near the sun as the
// nearest caster; linear filtering gives 2x2 PCF). The cascades share the
// light's axes, so a point is transformed once and scaled per cascade.
//
// The particles' coloured shadows (src/shadow.rs) share the cascades at
// a quarter of the resolution, read with a linear sampler where axes[0].w is 1:
// rgb the optical depth of the particles between the texel and the sun,
// a the depth of the particle nearest the sun (0 where none).

#define UNIT_CASCADES 3

// Specialization constant 0: false compiles the lookup out (the terrain
// pipeline when there is no shadow map; the lookup's mere presence costs
// the terrain pass about 18 % on an RTX 3060 Ti, even when it returns early).
layout(constant_id = 0) const bool UNIT_SHADOWS = true;

// Sunlight reaching world point p (shading normal n) past the units: 1 lit,
// 0 in a unit's shadow; 1 everywhere when off (scale[0].w 0). The receiver
// is pushed off its surface by a texel or so along its normal and toward
// the sun (no acne on the units' faces). The first cascade that holds it
// answers; the last fades out at its edge. `tint` is what the particles
// between it and the sun let through, per channel (1: none); a receiver
// nearer the sun than the nearest particle (a unit above the smoke) is
// not tinted, over a fade of two metres.
float unitShadowAt(sampler2DShadow tex, sampler2D tintTex, vec4 axes[3], vec4 origin[UNIT_CASCADES], vec4 scale[UNIT_CASCADES], vec3 sun, vec3 p, vec3 n, out vec3 tint) {
    tint = vec3(1.0);
    if (!UNIT_SHADOWS || scale[0].w < 0.5) {
        return 1.0;
    }
    vec3 push = 1.5 * n + sun;
    vec3 lp = vec3(dot(axes[0].xyz, p), dot(axes[1].xyz, p), dot(axes[2].xyz, p));
    vec3 ld = vec3(dot(axes[0].xyz, push), dot(axes[1].xyz, push), dot(axes[2].xyz, push));
    float margin = 1.5 / float(textureSize(tex, 0).y);
    for (int k = 0; k < UNIT_CASCADES; ++k) {
        vec3 c = (lp + ld * origin[k].w - origin[k].xyz) * scale[k].xyz;
        vec2 uv = c.xy * 0.5 + 0.5;
        if (any(lessThan(uv, vec2(margin))) || any(greaterThan(uv, vec2(1.0 - margin))) || c.z < 0.0 || c.z > 1.0) {
            continue;
        }
        vec2 at = vec2((uv.x + float(k)) / float(UNIT_CASCADES), uv.y);
        float lit = textureLod(tex, vec3(at, c.z), 0.0);
        if (axes[0].w > 0.5) {
            // Kept half a texel inside the cascade (its texels are four
            // of the map's), so the filter does not reach the next one.
            float h = 0.5 / float(textureSize(tintTex, 0).y);
            vec4 s = textureLod(tintTex, vec2((clamp(uv.x, h, 1.0 - h) + float(k)) / float(UNIT_CASCADES), clamp(uv.y, h, 1.0 - h)), 0.0);
            float below = clamp((s.a - c.z) / (2.0 * scale[k].z), 0.0, 1.0);
            tint = exp(-s.rgb * below);
        }
        if (k == UNIT_CASCADES - 1) {
            float fade = smoothstep(0.75, 0.97, max(abs(c.x), abs(c.y)));
            lit = mix(lit, 1.0, fade);
            tint = mix(tint, vec3(1.0), fade);
        }
        return lit;
    }
    return 1.0;
}
