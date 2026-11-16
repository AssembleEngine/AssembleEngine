// Shared by units_cull.comp, units.vert and units.frag (ae_render::units).
//
// The instance stream is attribute-major: attribute a of instance i is word
// a * capacity + i (ATTRIBUTES in units.rs gives the order). Positions are
// on the ground plane (nav x, y = world x, z); the height comes from the
// terrain's height texture, or the water surface for classes that float.

#define ATTR_X 0u
#define ATTR_Y 1u
#define ATTR_PREV_X 2u
#define ATTR_PREV_Y 3u
#define ATTR_HEADING_X 4u
#define ATTR_HEADING_Y 5u
#define ATTR_RADIUS 6u
#define ATTR_CLASS 7u
// The unit's first joint key (ae_render::units::NO_POSE: at rest).
#define ATTR_POSE 8u
#define MAX_CLASSES 16u

layout(std140, set = 0, binding = 0) uniform Frame {
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
    uvec4 anim;               // x: the tick drawn from (the stream's - 1); y: the fraction past it (float bits); z: joint key writes
} u;

layout(std430, set = 0, binding = 1) readonly buffer Instances { uint words[]; } inst;

#ifdef CULL
layout(std430, set = 0, binding = 2) writeonly buffer Visible { uint visible[]; };
#ifdef MODELS
// The darts' VkDrawIndirectCommand (words 0..3), then one
// VkDrawIndexedIndirectCommand (5 words) per model primitive and class
// (counts.w of them), then the same commands again for each shadow
// cascade.
layout(std430, set = 0, binding = 3) buffer Indirect { uint cmdWords[]; };
#else
layout(std430, set = 0, binding = 3) buffer Indirect { uint vertexCount; uint instanceCount; uint firstVertex; uint firstInstance; } cmd;
#endif
#else
layout(std430, set = 0, binding = 2) readonly buffer Visible { uint visible[]; };
#endif

layout(set = 0, binding = 4) uniform sampler2D heightTex;
// Shadow heights (src/terrain/sun.rs), world units; used when counts.z & 1.
layout(set = 0, binding = 5) uniform sampler2D sunShadowTex;
// The units' own shadow map (a 1x1 stand-in when off).
layout(set = 0, binding = 6) uniform sampler2DShadow unitShadowTex;
// The particles' coloured shadows (a 1x1 stand-in when off).
layout(set = 0, binding = 7) uniform sampler2D particleShadowTex;

#include "unit_shadows.glsl"

// Sunlight reaching world point p (normal n) past the units; `tint` past
// the particles.
float unitSunAt(vec3 p, vec3 n, out vec3 tint) {
    return unitShadowAt(unitShadowTex, particleShadowTex, u.unitShadowAxes, u.unitShadowOrigin, u.unitShadowScale, normalize(u.sun.xyz), p, n, tint);
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
    vec2 heading;  // unit vector on the ground plane
    float size;    // drawn radius: the body radius, enlarged to the minimum pixel size
    uint cls;
};

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
    vec2 heading = vec2(attrf(ATTR_HEADING_X, i), attrf(ATTR_HEADING_Y, i));
    unit.heading = dot(heading, heading) > 1e-12 ? normalize(heading) : vec2(1.0, 0.0);
    float radius = attrf(ATTR_RADIUS, i);
    float pixels = 2.0 * radius * u.camPos.w / max(distance(unit.pos, u.camPos.xyz), 1e-3);
    unit.size = radius * max(1.0, u.sun.w / max(pixels, 1e-6));
    return unit;
}

// The class is drawn with a model (src/models.rs), not a dart.
bool hasModel(uint cls) {
    return (uint(u.classes[cls].w) & 2u) != 0u;
}
