// Shading helpers shared by the terrain and the water surface (both
// include terrain_common.glsl first): the hills' and the units' shadows, fog of war,
// distance fog, noise.

// Shadow heights (src/terrain/sun.rs): per heightmap corner, the highest
// point above it that the terrain toward the sun shades (world units).
layout(set = 0, binding = 14) uniform sampler2D sunShadowTex;
// The units' shadow map (src/shadow.rs; a 1x1 stand-in when off).
layout(set = 0, binding = 15) uniform sampler2DShadow unitShadowTex;

#include "unit_shadows.glsl"

// Sunlight reaching world point p (normal n) past the units.
float unitSunAt(vec3 p, vec3 n) {
    return unitShadowAt(unitShadowTex, u.unitShadowAxes, u.unitShadowOrigin, u.unitShadowScale, normalize(u.sun.xyz), p, n);
}

// Sunlight reaching height `y` over world point `xz` past the hills: 1 lit,
// 0 in shadow, over a metre or so (a third of a cell on coarse maps) of
// depth into the shadow. With FLAG_SHADOWS off, 1.
float sunAt(vec2 xz, float y) {
    if ((u.cfg.z & FLAG_SHADOWS) == 0u) {
        return 1.0;
    }
    float s = textureLod(sunShadowTex, (xz / u.map.w + 0.5) * u.height.zw, 0.0).r;
    return 1.0 - smoothstep(0.0, max(1.0, 0.8 * u.map.w), s - y);
}

// Vision (fog of war): two bit planes over the vision grid, line of sight
// then explored, rows of u.vision.w words.
layout(std430, set = 0, binding = 13) readonly buffer VisionBuf { uint vision[]; };

// 1 in line of sight, 0.5 explored, 0.2 never seen; cells off the grid read
// as the nearest edge cell (the blend reaches half a cell past the map).
float visionCell(ivec2 c) {
    c = clamp(c, ivec2(0), ivec2(u.vision.xy) - 1);
    uint stride = uint(u.vision.w);
    uint i = uint(c.y) * stride + uint(c.x) / 32u;
    uint bit = 1u << (uint(c.x) % 32u);
    if ((vision[i] & bit) != 0u) {
        return 1.0;
    }
    return (vision[stride * uint(u.vision.y) + i] & bit) != 0u ? 0.5 : 0.2;
}

// Bilinear over the four nearest cells, so 32 m cells do not show as blocks.
float visionAt(vec2 world) {
    vec2 p = world / u.vision.z - 0.5;
    ivec2 c = ivec2(floor(p));
    vec2 f = fract(p);
    float a = mix(visionCell(c), visionCell(c + ivec2(1, 0)), f.x);
    float b = mix(visionCell(c + ivec2(0, 1)), visionCell(c + ivec2(1, 1)), f.x);
    return mix(a, b, f.y);
}

// Dimmed by the vision at `world`; out of sight also loses colour, so it
// reads as remembered, not dark. Exactly `color` in line of sight.
vec3 applyVision(vec3 color, vec2 world) {
    float v = visionAt(world);
    color *= v;
    return mix(vec3(dot(color, vec3(0.3, 0.59, 0.11))), color, 0.4 + 0.6 * v);
}

// Share of the fog colour at distance `dist`.
float fogAmount(float dist) {
    return 1.0 - exp(-pow(dist * u.fog.w, 1.5));
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth value noise in 0..1.
float valueNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 t = f * f * (3.0 - 2.0 * f);
    float a = hash12(i), b = hash12(i + vec2(1.0, 0.0)), c = hash12(i + vec2(0.0, 1.0)), d = hash12(i + vec2(1.0, 1.0));
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}
