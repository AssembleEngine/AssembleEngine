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
#define MAX_CLASSES 16u

layout(std140, set = 0, binding = 0) uniform Frame {
    mat4 viewProj;
    vec4 camPos;      // xyz eye; w = pixels per world unit at distance 1
    vec4 frustum[6];  // normalised planes, inside >= 0
    vec4 map;         // cell size, 1/(corners x), 1/(corners y), interpolation alpha
    vec4 height;      // scale, offset, water level, has water
    vec4 sun;         // xyz direction; w = minimum unit size in pixels
    vec4 fog;         // rgb, density (0 = off)
    uvec4 counts;     // instances, capacity
    vec4 classes[MAX_CLASSES]; // rgb; w = 1: floats on the water surface
} u;

layout(std430, set = 0, binding = 1) readonly buffer Instances { uint words[]; } inst;

#ifdef CULL
layout(std430, set = 0, binding = 2) writeonly buffer Visible { uint visible[]; };
layout(std430, set = 0, binding = 3) buffer Indirect { uint vertexCount; uint instanceCount; uint firstVertex; uint firstInstance; } cmd;
#else
layout(std430, set = 0, binding = 2) readonly buffer Visible { uint visible[]; };
#endif

layout(set = 0, binding = 4) uniform sampler2D heightTex;

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
    if (u.height.w > 0.5 && u.classes[unit.cls].w > 0.5) {
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
