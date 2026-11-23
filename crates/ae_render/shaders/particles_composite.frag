#version 460
// The half-resolution particles over the full-resolution scene
// (premultiplied "over"). Depth-aware upsampling: where the four nearest
// half-resolution texels were softened against depths close to this
// pixel's, bilinear; across a depth edge, the texel whose depth is
// nearest this pixel's, so smoke behind a silhouette does not bleed onto
// it.
//
// The particles' light is a separate half-resolution target
// (particles_half.frag): it was blended per particle in sorted order, so
// smoke in front of a fire has already taken out its share and smoke
// behind it has not. This pass only upsamples it; the accumulated alpha
// (the particles' coverage) masks the scene's glow behind them. Deciding
// the glow from the accumulated colour here would lose the emission
// wherever smoke shared the pixel.
#include "particles_common.glsl"

#define halfColor AE_S2D(0, 0)
#define halfDepth AE_S2D(1, 0)
#define depthTex AE_S2D(2, 0)
// The particles' glow (B10G11R11), sorted and premultiplied like the
// colour. The test-only OIT blends leave it black.
#define halfGlow AE_S2D(3, 0)

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outGlow;

// View depth (infinite reverse-Z), the sky far but finite.
float viewDepth(float d) {
    return u.camPos.w / max(d, 1e-7);
}

void main() {
    ivec2 size = textureSize(halfColor, 0) - 1;
    float z = viewDepth(texelFetch(depthTex, ivec2(gl_FragCoord.xy), 0).r);
    vec2 lp = gl_FragCoord.xy * 0.5 - 0.5;
    ivec2 i0 = ivec2(floor(lp));
    vec2 f = lp - vec2(i0);
    ivec2 offsets[4] = ivec2[](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1));
    float weights[4] = float[]((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    // Relative: a slope seen at a grazing angle changes depth fast.
    float tolerance = 0.03 * z;
    vec4 bilinear = vec4(0.0);
    vec3 bilinearGlow = vec3(0.0);
    float worst = 0.0;
    float best = 1e38;
    vec4 nearest = vec4(0.0);
    vec3 nearestGlow = vec3(0.0);
    for (int k = 0; k < 4; ++k) {
        ivec2 t = clamp(i0 + offsets[k], ivec2(0), size);
        vec4 c = texelFetch(halfColor, t, 0);
        vec3 g = texelFetch(halfGlow, t, 0).rgb;
        float gap = abs(viewDepth(texelFetch(halfDepth, t, 0).r) - z);
        bilinear += c * weights[k];
        bilinearGlow += g * weights[k];
        worst = max(worst, gap);
        if (gap < best) {
            best = gap;
            nearest = c;
            nearestGlow = g;
        }
    }
    outColor = worst <= tolerance ? bilinear : nearest;
    vec3 particleGlow = worst <= tolerance ? bilinearGlow : nearestGlow;
    // Untouched pixels stay exactly as they were (no blend round trip).
    if (outColor == vec4(0.0)) {
        discard;
    }
    // The target holds physical radiance; the scene is exposed radiance.
    outColor.rgb *= u.photometry.x;
    // The particles' light joins the scene's glow (the water's glints, a
    // decal's glow), masked where the particles cover it. The rgb is
    // already the sorted per-particle result; the alpha must not be
    // applied to it again, only to what lies behind.
    outGlow = vec4(particleGlow * u.photometry.x, outColor.a);
}
