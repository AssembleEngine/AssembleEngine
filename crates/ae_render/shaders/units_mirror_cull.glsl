#ifndef AE_UNITS_MIRROR_CULL
#define AE_UNITS_MIRROR_CULL
// The units' mirror cull (units_cull.comp, units_models_cull.comp with
// MIRROR): whether a unit's bounding sphere can show in the planar
// reflection at all. Conservative: it never drops a unit whose reflection
// some water pixel samples (M16 reflection cost: a unit-dense view with
// little or no water drew every unit a second time).
//
// * The mirror draw clips at the water level (units*.vert,
//   gl_ClipDistance), so a sphere wholly below it draws nothing.
// * The water reads the reflection at its own screen position moved by at
//   most REFLECT_SHIFT (water.frag). So the part of the sphere above the
//   water, mirrored and projected, widened by that shift, is the only
//   screen box where water may show it. The water surface lies within
//   WAVE_BOUND of the level, and is drawn over the map only (water mesh
//   leaves, terrain_common.glsl nodeBounds) where the ground is below it:
//   the box's rays through that slab meet the map over a rectangle where
//   some ground (the terrain's min pyramid) must be under the slab's top.
//
// Needs water_patch.glsl (WAVE_BOUND) and the mirror frame's inverse
// view-projection (units_common.glsl). Image slot 5 is the terrain's
// min/max pyramid (src/terrain/heightmap.rs, world heights; level
// PYRAMID_BASE_LEVEL at mip 0).
#include "water_reflect.glsl"

#define mirrorPyramid AE_S2D(5, 1)
const int MIRROR_PYRAMID_BASE = 2;

// Whether any ground over cells [a, b] (cell coordinates, clamped to the
// map) lies below `top`, from the pyramid level whose blocks are at least a
// quarter of the box's longer side: at most 6 x 6 blocks, so a long thin
// footprint (a far, grazing view) is not rounded up to blocks that reach
// the shore past it.
bool mirrorGroundBelow(vec2 a, vec2 b, float top) {
    int root = textureSize(mirrorPyramid, 0).x << MIRROR_PYRAMID_BASE;
    int coarsest = findMSB(root);
    float extent = max(max(b.x - a.x, b.y - a.y) * 0.25, 1.0);
    int level = clamp(int(ceil(log2(extent))), MIRROR_PYRAMID_BASE, coarsest);
    int n = root >> level;
    ivec2 t0 = clamp(ivec2(floor(a / float(1 << level))), ivec2(0), ivec2(n - 1));
    ivec2 t1 = clamp(ivec2(floor(b / float(1 << level))), ivec2(0), ivec2(n - 1));
    int mip = level - MIRROR_PYRAMID_BASE;
    for (int y = t0.y; y <= t1.y; y++) {
        for (int x = t0.x; x <= t1.x; x++) {
            if (texelFetch(mirrorPyramid, ivec2(x, y), mip).r < top) {
                return true;
            }
        }
    }
    return false;
}

bool mirrorReachesWater(vec3 centre, float r) {
    float level = u.height.z;
    if (centre.y + r < level) {
        return false;
    }
    vec3 eye = u.camPos.xyz;
    float low = level - WAVE_BOUND, high = level + WAVE_BOUND;
    // An eye inside or under the slab: the rays below need not meet it.
    if (eye.y <= high) {
        return true;
    }
    // The part above the water, mirrored, on the screen (NDC).
    vec3 lo3 = vec3(centre.x - r, max(centre.y - r, level), centre.z - r);
    vec3 hi3 = centre + r;
    vec2 lo = vec2(1e30), hi = vec2(-1e30);
    for (int k = 0; k < 8; k++) {
        vec3 corner = mix(lo3, hi3, vec3(ivec3(k, k >> 1, k >> 2) & 1));
        vec4 clip = u.mirrorProj * vec4(corner, 1.0);
        if (clip.w <= 1e-4) {
            return true;
        }
        lo = min(lo, clip.xy / clip.w);
        hi = max(hi, clip.xy / clip.w);
    }
    // The water's sample shift (UV, so twice that in NDC), a texel of the
    // half-resolution target's bilinear tap, and the screen's edge (the
    // water clamps its samples to it).
    float shift = 2.0 * REFLECT_SHIFT + 0.01;
    lo = max(lo - shift, vec2(-1.0));
    hi = min(hi + shift, vec2(1.0));
    if (any(greaterThan(lo, hi))) {
        return false;
    }
    // The box's corner rays through the slab: their hits bound every water
    // point that shows in the box.
    vec2 a = vec2(1e30), b = vec2(-1e30);
    for (int k = 0; k < 4; k++) {
        vec2 ndc = vec2((k & 1) != 0 ? hi.x : lo.x, (k & 2) != 0 ? hi.y : lo.y);
        vec4 p = u.inverseViewProj * vec4(ndc, 0.5, 1.0);
        vec3 dir = p.xyz / p.w - eye;
        // Level or rising: it reaches the horizon, never the slab's far side.
        if (dir.y >= -1e-6 * length(dir)) {
            return true;
        }
        for (int s = 0; s < 2; s++) {
            float t = ((s == 0 ? high : low) - eye.y) / dir.y;
            vec2 hit = eye.xz + t * dir.xz;
            a = min(a, hit);
            b = max(b, hit);
        }
    }
    // To cells (a cell's bilinear surface spans its corner samples: one
    // more on each side), then the map (the water is drawn over it only).
    vec2 cells = 1.0 / u.map.yz - 1.0;
    a = max(a / u.map.x - 1.0, vec2(0.0));
    b = min(b / u.map.x + 1.0, cells);
    if (any(greaterThan(a, b))) {
        return false;
    }
    return mirrorGroundBelow(a, b, high);
}
#endif
