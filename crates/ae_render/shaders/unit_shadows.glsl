// Shared by opaque shadow receivers: terrain, units and water. The water
// reflection's receivers (MIRROR) are lit: the page pass requests pages for
// the camera's view only (shadow_reset.comp).
#include "shadow_pages.glsl"

// Specialization constant 0 compiles the lookup out on shadowless terrain.
layout(constant_id = 0) const bool UNIT_SHADOWS = true;

#define vsmTexture(image, samplerIndex) usampler2D(ae_images_uint[nonuniformEXT(image)], ae_samplers[nonuniformEXT(samplerIndex)])

float shadowCompare(uint image, uint samplerIndex, uint entry, ivec2 texel, float depth) {
    if (entry == SHADOW_PAGE_EMPTY) return 1.0;
    uint stored = texelFetch(vsmTexture(image, samplerIndex), shadowPhysicalTexel(entry & SHADOW_SLOT_MASK, texel), 0).r;
    return float(depth >= uintBitsToFloat(stored));
}

// Resolve each tap independently when the bilinear footprint crosses a
// virtual page boundary. Physical neighbors can back unrelated pages.
float shadowPcf(uint image, uint samplerIndex, int level, vec2 uv, float depth) {
    vec2 texel = uv * float(SHADOW_LEVEL_SIZE) - 0.5;
    ivec2 a = ivec2(floor(texel));
    ivec2 b = a + 1;
    ivec2 pageA = a / SHADOW_PAGE_SIZE;
    ivec2 pageB = b / SHADOW_PAGE_SIZE;
    uint entry00 = texelFetch(vsmTexture(image, samplerIndex), shadowTableAddress(pageA, level), 0).r;
    uint entry10 = entry00;
    uint entry01 = entry00;
    uint entry11 = entry00;
    if (pageA.x != pageB.x) {
        entry10 = texelFetch(vsmTexture(image, samplerIndex), shadowTableAddress(ivec2(pageB.x, pageA.y), level), 0).r;
        entry11 = entry10;
    }
    if (pageA.y != pageB.y) {
        entry01 = texelFetch(vsmTexture(image, samplerIndex), shadowTableAddress(ivec2(pageA.x, pageB.y), level), 0).r;
        entry11 = pageA.x == pageB.x ? entry01 :
            texelFetch(vsmTexture(image, samplerIndex), shadowTableAddress(pageB, level), 0).r;
    }
    if (max(max(entry00, entry10), max(entry01, entry11)) == SHADOW_PAGE_UNBACKED) return -1.0;
    if ((entry00 | entry10 | entry01 | entry11) == SHADOW_PAGE_EMPTY) return 1.0;
    vec2 weight = fract(texel);
    float top = mix(shadowCompare(image, samplerIndex, entry00, a, depth),
                    shadowCompare(image, samplerIndex, entry10, ivec2(b.x, a.y), depth), weight.x);
    float bottom = mix(shadowCompare(image, samplerIndex, entry01, ivec2(a.x, b.y), depth),
                       shadowCompare(image, samplerIndex, entry11, b, depth), weight.x);
    return mix(top, bottom, weight.y);
}
#undef vsmTexture

// Return a negative value when the level cannot answer. When the page
// pool fills, the caller can continue to a coarser level.
float sampleShadowLevel(uint shadowImage, uint shadowSampler, uint tintImage, uint tintSampler, bool colored,
                        int level, vec3 c, float depthScale, out vec3 tint) {
    vec2 uv = c.xy * 0.5 + 0.5;
    tint = vec3(1.0);
    if (max(abs(c.x), abs(c.y)) > SHADOW_EDGE_END) return -1.0;

    float lit = shadowPcf(shadowImage, shadowSampler, level, uv, clamp(c.z, 0.0, 1.0));
    if (lit < 0.0) return lit;
    if (colored) {
#define tintTex AE_S2D_AT(tintImage, tintSampler)
        float h = 0.5 / float(textureSize(tintTex, 0).y);
        vec2 inside = clamp(uv, h, 1.0 - h);
        vec4 s = textureLod(tintTex, vec2((inside.x + float(level)) / float(UNIT_CASCADES), inside.y), 0.0);
        float below = clamp((s.a - c.z) / (2.0 * depthScale), 0.0, 1.0);
        tint = exp(-s.rgb * below);
#undef tintTex
    }
    if (level == UNIT_CASCADES - 1) {
        float fade = smoothstep(0.75, 0.97, max(abs(c.x), abs(c.y)));
        lit = mix(lit, 1.0, fade);
        tint = mix(tint, vec3(1.0), fade);
    }
    return lit;
}

float unitShadowAt(uint shadowImage, uint shadowSampler, uint tintImage, uint tintSampler, vec4 axes[3], vec4 origin[UNIT_CASCADES], vec4 scale[UNIT_CASCADES], vec3 sun, vec3 p, vec3 n, out vec3 tint) {
    tint = vec3(1.0);
#ifdef MIRROR
    return 1.0;
#endif
    if (!UNIT_SHADOWS || scale[0].w < 0.5) return 1.0;

    vec3 push = 1.5 * n + sun;
    vec3 lp = vec3(dot(axes[0].xyz, p), dot(axes[1].xyz, p), dot(axes[2].xyz, p));
    // All clipmaps use the same unbiased depth interval.
    float lightDepth = (lp.z - origin[0].z) * scale[0].z;
    if (lightDepth < 0.0 || lightDepth > 1.0) return 1.0;
    vec3 ld = vec3(dot(axes[0].xyz, push), dot(axes[1].xyz, push), dot(axes[2].xyz, push));
    bool colored = axes[0].w > 0.5;
    float fineLit = 1.0;
    float coarseWeight = 0.0;
    for (int level = 0; level < UNIT_CASCADES; ++level) {
        // Apply the small bias after subtracting the origin, in local
        // light space. Adding it to a large world coordinate loses bits.
        vec3 c = ((lp - origin[level].xyz) + ld * origin[level].w) * scale[level].xyz;
        vec3 levelTint;
        float lit = sampleShadowLevel(shadowImage, shadowSampler, tintImage, tintSampler, colored,
                                      level, c, scale[level].z, levelTint);
        if (lit < 0.0) continue;
        if (coarseWeight > 0.0) {
            tint = mix(tint, levelTint, coarseWeight);
            return mix(fineLit, lit, coarseWeight);
        }
        tint = levelTint;
        fineLit = lit;
        if (level == UNIT_CASCADES - 1) break;

        float edgeWeight = smoothstep(SHADOW_EDGE_START, SHADOW_EDGE_END, max(abs(c.x), abs(c.y)));
        coarseWeight = level == 0 ? max(edgeWeight, 1.0 - axes[1].w) : edgeWeight;
        if (coarseWeight == 0.0) break;
    }
    return fineLit;
}
