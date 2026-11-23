// Shared by units_cull.comp, units.vert and units.frag (ae_render::units).
//
// The instance stream is attribute-major: attribute a of instance i is word
// a * capacity + i (ATTRIBUTES in units.rs gives the order). Positions are
// on the ground plane (nav x, y = world x, z); the height comes from the
// terrain's height texture, or the water surface for classes that float.

#include "bindless.glsl"

#define ATTR_X 0u
#define ATTR_Y 1u
#define ATTR_PREV_X 2u
#define ATTR_PREV_Y 3u
// The unit's rotation from its frame (forward +x, up +y, right +z) to the
// world's: a quaternion (x, y, z, w), zero for the identity.
#define ATTR_ROT_X 4u
#define ATTR_ROT_Y 5u
#define ATTR_ROT_Z 6u
#define ATTR_ROT_W 7u
#define ATTR_RADIUS 8u
#define ATTR_CLASS 9u
// The unit's first animation record cell (ae_render::units::NO_POSE: at rest),
// POSE_MANUAL set when either record shows a manual joint.
#define ATTR_POSE 10u
#define NO_POSE 0xffffffffu
#define POSE_MANUAL 0x80000000u
#define MAX_CLASSES 16u

// Buffers: the handle table holds the frame block, the instances, the
// visible list, the indirect, the lights (AE_LIGHT_BASE 4), then the
// models' (units_models.glsl).
#define AE_LIGHT_BASE 4
layout(std140, set = AE_BINDLESS_SET, binding = AE_UNIFORM_BLOCKS) uniform Frame {
    mat4 viewProj;
    vec4 camPos;      // xyz eye; w = pixels per world unit at distance 1
    vec4 frustum[6];  // normalised planes, inside >= 0
    vec4 map;         // cell size, 1/(corners x), 1/(corners y), interpolation alpha
    vec4 height;      // scale, offset, water level, has water
    vec4 sun;         // xyz direction; w = minimum unit size in pixels
    vec4 fog;         // rgb, density (0 = off)
    uvec4 counts;     // instances, capacity, flags (1: hills' shadows), model commands
    vec4 classes[MAX_CLASSES]; // rgb; w: 1 = floats on the water surface, + 2 = has a model
    // The units' shadow cascades (src/shadow.rs, unit_shadows.glsl).
    vec4 unitShadowAxes[3];   // light axes: right, up, toward the sun
    vec4 unitShadowOrigin[3]; // per cascade: centre x, y, depth origin; w: world texel size
    vec4 unitShadowScale[3];  // per cascade: 1/half-size x, y, 1/depth range; w: 1 = on
    uvec4 anim;               // x: the tick drawn from (the stream's - 1); y: the fraction past it (float bits); z: animation table writes
    vec4 photometry;          // x: exposure, y: unused, z: sun lux, w: sky lux
} ae_Frame[];
#define u ae_Frame[AE_BUF(0)]

layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer Instances { uint words[]; } ae_Instances[];
#define inst ae_Instances[AE_BUF(1)]

#ifdef CULL
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) writeonly buffer Visible { uint visible[]; } ae_Visible[];
#define visible ae_Visible[AE_BUF(2)].visible
#ifdef MODELS
// The darts' VkDrawIndirectCommand (words 0..3), then one
// VkDrawIndexedIndirectCommand (5 words) per model primitive and class
// (counts.w of them), then the same commands again for each shadow
// cascade.
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) buffer Indirect { uint cmdWords[]; } ae_Indirect[];
#define cmdWords ae_Indirect[AE_BUF(3)].cmdWords
#else
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) buffer Indirect { uint vertexCount; uint instanceCount; uint firstVertex; uint firstInstance; } ae_Indirect[];
#define cmd ae_Indirect[AE_BUF(3)]
#endif
#else
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer Visible { uint visible[]; } ae_Visible[];
#define visible ae_Visible[AE_BUF(2)].visible
#endif

#define heightTex AE_S2D(0, 1)
// Shadow heights (src/terrain/sun.rs), world units; used when counts.z & 1.
#define sunShadowTex AE_S2D(1, 1)

#include "unit_shadows.glsl"

// Sunlight reaching world point p (normal n) past the units; `tint` past
// the particles. Handles: the units' shadow map 2 with the comparison
// sampler 4, the particles' coloured shadows 3 with sampler 3 (each a
// 1x1 stand-in when off).
float unitSunAt(vec3 p, vec3 n, out vec3 tint) {
    return unitShadowAt(AE_IMAGE(2), AE_SAMPLER(4), AE_IMAGE(3), AE_SAMPLER(3), u.unitShadowAxes, u.unitShadowOrigin, u.unitShadowScale, normalize(u.sun.xyz), p, n, tint);
}

// Sunlight reaching world point p past the hills (terrain_shading.glsl's
// sunAt with the units' uniforms).
float sunAt(vec3 p) {
    if ((u.counts.z & 1u) == 0u) {
        return 1.0;
    }
    float s = textureLod(sunShadowTex, (p.xz / u.map.x + 0.5) * u.map.yz, 0.0).r;
    return 1.0 - smoothstep(0.0, max(1.0, 0.8 * u.map.x), s - p.y);
}

float attrf(uint a, uint i) { return uintBitsToFloat(inst.words[a * u.counts.y + i]); }
uint attru(uint a, uint i) { return inst.words[a * u.counts.y + i]; }

struct Unit {
    vec3 pos;      // on the ground (or water surface)
    vec4 rot;      // unit frame (forward +x, up +y, right +z) to the world, normalised
    float size;    // drawn radius: the body radius, enlarged to the minimum pixel size
    uint cls;
};

// v turned by the unit quaternion q (x, y, z, w).
vec3 rotate(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

Unit loadUnit(uint i) {
    Unit unit;
    vec2 p = mix(vec2(attrf(ATTR_PREV_X, i), attrf(ATTR_PREV_Y, i)), vec2(attrf(ATTR_X, i), attrf(ATTR_Y, i)), u.map.w);
    vec2 uv = (p / u.map.x + 0.5) * u.map.yz;
    float h = textureLod(heightTex, uv, 0.0).r * u.height.x + u.height.y;
    unit.cls = min(attru(ATTR_CLASS, i), MAX_CLASSES - 1u);
    if (u.height.w > 0.5 && (uint(u.classes[unit.cls].w) & 1u) != 0u) {
        h = max(h, u.height.z);
    }
    unit.pos = vec3(p.x, h, p.y);
    vec4 rot = vec4(attrf(ATTR_ROT_X, i), attrf(ATTR_ROT_Y, i), attrf(ATTR_ROT_Z, i), attrf(ATTR_ROT_W, i));
    unit.rot = dot(rot, rot) > 1e-12 ? normalize(rot) : vec4(0.0, 0.0, 0.0, 1.0);
    float radius = attrf(ATTR_RADIUS, i);
    float pixels = 2.0 * radius * u.camPos.w / max(distance(unit.pos, u.camPos.xyz), 1e-3);
    unit.size = radius * max(1.0, u.sun.w / max(pixels, 1e-6));
    return unit;
}

// The class is drawn with a model (src/models.rs), not a dart.
bool hasModel(uint cls) {
    return (uint(u.classes[cls].w) & 2u) != 0u;
}
