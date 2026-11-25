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
// world's: a quaternion (x, y, z, w), zero for the identity; the previous
// tick's, which loadUnit slerps to this one by the frame's alpha.
#define ATTR_ROT_X 4u
#define ATTR_ROT_Y 5u
#define ATTR_ROT_Z 6u
#define ATTR_ROT_W 7u
#define ATTR_PREV_ROT_X 8u
#define ATTR_PREV_ROT_Y 9u
#define ATTR_PREV_ROT_Z 10u
#define ATTR_PREV_ROT_W 11u
#define ATTR_RADIUS 12u
#define ATTR_CLASS 13u
// The unit's first animation record cell (ae_render::units::NO_POSE: at rest),
// POSE_MANUAL set when either record shows a manual joint.
#define ATTR_POSE 14u
#define ATTR_WATER 15u
#define ATTR_VIEW 16u
#define VIEW_SEEN 1u
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
    uvec4 lists;              // x: the placement records' first word in each visible list (placedUnit)
// The mirror frame (src/water/reflect.rs) and the camera's inverse
// view-projection: written whenever the scene has water (the cull reads
// them only while it fills the reflection's lists).
#if defined(MIRROR) || defined(CULL)
    mat4 mirrorProj;
    vec4 mirrorCamera;
    vec4 mirrorPlanes[6];
    mat4 inverseViewProj; // the camera's (units_mirror_cull.glsl)
#endif
} ae_Frame[];
#define u ae_Frame[AE_BUF(0)]

layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer Instances { uint words[]; } ae_Instances[];
#define inst ae_Instances[AE_BUF(1)]

#ifdef MIRROR
#define UNIT_VIEW_POS u.mirrorCamera.xyz
#define UNIT_VIEW_PLANES u.mirrorPlanes
#define UNIT_VISIBLE_INDEX ae.words.z
#define UNIT_INDIRECT_INDEX ae.words.w
#else
#define UNIT_VIEW_POS u.camPos.xyz
#define UNIT_VIEW_PLANES u.frustum
#define UNIT_VISIBLE_INDEX AE_BUF(2)
#define UNIT_INDIRECT_INDEX AE_BUF(3)
#endif

#ifdef CULL
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) writeonly buffer Visible { uint visible[]; } ae_Visible[];
#define visible ae_Visible[UNIT_VISIBLE_INDEX].visible
#ifdef MODELS
// The darts' VkDrawIndirectCommand (words 0..3), then one
// VkDrawIndexedIndirectCommand (5 words) per model primitive and class
// (counts.w of them), then the same commands again for each shadow
// cascade.
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) buffer Indirect { uint cmdWords[]; } ae_Indirect[];
#define cmdWords ae_Indirect[UNIT_INDIRECT_INDEX].cmdWords
#else
// The camera's command, then the water overlay's (overlayWord).
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) buffer Indirect { uint vertexCount; uint instanceCount; uint firstVertex; uint firstInstance; uint overlay[4]; } ae_Indirect[];
#define cmd ae_Indirect[UNIT_INDIRECT_INDEX]
#endif
#else
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer Visible { uint visible[]; } ae_Visible[];
#define visible ae_Visible[UNIT_VISIBLE_INDEX].visible
#endif

// Each visible list (the camera's and the reflection's) ends with the
// units' placement records: two vec4 per row from word u.lists.x (a
// multiple of 4), the cull's loadUnit once per unit (position, body
// radius; rotation), so the draws do not rebuild it per vertex
// (placedUnit). The camera's list has every row's; the reflection's, its
// eligible rows' as the reflection places them.
#ifdef CULL
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) writeonly buffer Placed { vec4 placed[]; } ae_Placed[];
#else
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer Placed { vec4 placed[]; } ae_Placed[];
#endif
uint placedAt(uint i) { return u.lists.x / 4u + 2u * i; }

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

// Own and Seen rows have current positions that auxiliary passes may use.
bool eligible(uint i) { return attru(ATTR_VIEW, i) <= VIEW_SEEN; }

#ifdef RIDE_WAVES
#define WAVE_SAMPLE 14
#include "water_patch.glsl"
#endif

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

vec4 quatAt(uint a, uint i) {
    return vec4(attrf(a, i), attrf(a + 1u, i), attrf(a + 2u, i), attrf(a + 3u, i));
}

vec4 unitQuat(vec4 q) {
    return dot(q, q) > 1e-12 ? normalize(q) : vec4(0.0, 0.0, 0.0, 1.0);
}

// Shortest-arc SLERP of two unit quaternions. Keep b when the dot is zero
// so a 180-degree turn chooses one of the two equally short arcs.
vec4 slerp(vec4 a, vec4 b, float t) {
    float cosHalfTheta = dot(a, b);
    if (cosHalfTheta < 0.0) {
        b = -b;
        cosHalfTheta = -cosHalfTheta;
    }
    if (cosHalfTheta >= 1.0 - 0.005) return normalize(mix(a, b, t));

    float halfTheta = acos(cosHalfTheta);
    // Normalisation supplies the common factor 1 / sin(halfTheta).
    return normalize(a * sin((1.0 - t) * halfTheta) + b * sin(t * halfTheta));
}

vec4 unitRotation(uint i, float alpha) {
    vec4 rot = quatAt(ATTR_ROT_X, i);
    if (alpha >= 1.0) return unitQuat(rot);
    vec4 prev = quatAt(ATTR_PREV_ROT_X, i);
    if (alpha <= 0.0 || all(equal(prev, rot))) return unitQuat(prev);

    return slerp(unitQuat(prev), unitQuat(rot), alpha);
}

#if defined(CULL) && !defined(MIRROR)
// The unit's height in the reflection (loadUnit sets it): a floating hull
// at the water level, a submerged row at its depth, as the MIRROR
// variant places it.
float unitMirrorY;
#endif

#ifdef RIDE_WAVES
// A floating hull (at p, over ground height `ground`, body radius `radius`)
// tilted and lifted onto the full tier's waves.
void rideWaves(inout Unit unit, uint i, vec2 p, float ground, float radius) {
    if (eligible(i) && u.height.w > 0.5 && ground < u.height.z && attrf(ATTR_WATER, i) <= 0.0
        && (uint(u.classes[unit.cls].w) & 1u) != 0u) {
        vec4 wave = waterWave(p);
        float weight = patchWeight(patchUV(p));
        vec2 slope = wave.xy * weight;
        vec3 normal = normalize(vec3(-slope.x, 1.0, -slope.y));
        vec4 tilt = normalize(vec4(normal.z, 0.0, -normal.x, 1.0 + normal.y));
        unit.rot = vec4(tilt.w * unit.rot.xyz + unit.rot.w * tilt.xyz + cross(tilt.xyz, unit.rot.xyz),
            tilt.w * unit.rot.w - dot(tilt.xyz, unit.rot.xyz));
        // Check the surface under the hull's ends. A local nonlinear
        // crest can rise above the tangent plane used to tilt the hull.
        float height = wave.z * weight;
        vec2 axis = rotate(unit.rot, vec3(1.0, 0.0, 0.0)).xz * radius;
        vec2 side = vec2(-axis.y, axis.x) * 0.8;
        float lift = 0.0;
        for (int k = 0; k < 4; ++k) {
            vec2 offset = (k < 2 ? axis : -axis) + ((k & 1) == 0 ? side : -side);
            lift = max(lift, waterDisplacement(p + offset) - (height + dot(slope, offset)));
        }
        unit.pos.y = max(ground, u.height.z + height + min(lift, 0.2) + 0.06 * weight);
    }
}
#endif

// The drawn radius: the body radius, enlarged to the minimum pixel size as
// seen from the view.
void sizeUnit(inout Unit unit, float radius) {
    float pixels = 2.0 * radius * u.camPos.w / max(distance(unit.pos, UNIT_VIEW_POS), 1e-3);
    unit.size = radius * max(1.0, u.sun.w / max(pixels, 1e-6));
}

float groundAt(vec2 p) {
    vec2 uv = (p / u.map.x + 0.5) * u.map.yz;
    return textureLod(heightTex, uv, 0.0).r * u.height.x + u.height.y;
}

Unit loadUnit(uint i) {
    Unit unit;
    vec2 p = mix(vec2(attrf(ATTR_PREV_X, i), attrf(ATTR_PREV_Y, i)), vec2(attrf(ATTR_X, i), attrf(ATTR_Y, i)), u.map.w);
    float h = groundAt(p);
#if defined(SUB) || defined(RIDE_WAVES)
    float ground = h;
#endif
    unit.cls = min(attru(ATTR_CLASS, i), MAX_CLASSES - 1u);
    if (u.height.w > 0.5 && (uint(u.classes[unit.cls].w) & 1u) != 0u) {
        h = max(h, u.height.z);
    }
#if defined(CULL) && !defined(MIRROR)
    unitMirrorY = h - max(attrf(ATTR_WATER, i), 0.0);
#endif
#ifdef MIRROR
    h -= max(attrf(ATTR_WATER, i), 0.0);
#elif defined(SUB)
    if (u.height.w > 0.5 && attrf(ATTR_WATER, i) > 0.0) {
        h = max(ground, u.height.z - attrf(ATTR_WATER, i));
    }
#endif
    unit.pos = vec3(p.x, h, p.y);
    unit.rot = unitRotation(i, u.map.w);
    float radius = attrf(ATTR_RADIUS, i);
#ifdef RIDE_WAVES
    rideWaves(unit, i, p, ground, radius);
#endif
    sizeUnit(unit, radius);
    return unit;
}

#ifdef CULL
// Row i's placement record in list `list` (a buffer handle): `unit` from
// loadUnit(i) (or mirrorUnit), before any wave riding.
void placeUnit(uint list, uint i, Unit unit) {
    uint at = placedAt(i);
    ae_Placed[list].placed[at] = vec4(unit.pos, attrf(ATTR_RADIUS, i));
    ae_Placed[list].placed[at + 1u] = unit.rot;
}
#else
// Row i as loadUnit places it, from the cull's placement record in this
// pass's visible list (the reflection's for MIRROR); floating hulls then
// ride the waves (RIDE_WAVES) and the size follows the view.
Unit placedUnit(uint i) {
    uint at = placedAt(i);
    vec4 a = ae_Placed[UNIT_VISIBLE_INDEX].placed[at];
    Unit unit;
    unit.pos = a.xyz;
    unit.rot = ae_Placed[UNIT_VISIBLE_INDEX].placed[at + 1u];
    unit.cls = min(attru(ATTR_CLASS, i), MAX_CLASSES - 1u);
#ifdef RIDE_WAVES
    rideWaves(unit, i, unit.pos.xz, groundAt(unit.pos.xz), a.w);
#endif
    sizeUnit(unit, a.w);
    return unit;
}
#endif

#if defined(CULL) && !defined(MIRROR)
// `unit` (from loadUnit(i)) as the MIRROR variant's loadUnit places and
// sizes it: at unitMirrorY, its minimum pixel size seen from the mirrored
// camera.
Unit mirrorUnit(Unit unit, uint i) {
    unit.pos.y = unitMirrorY;
    float radius = attrf(ATTR_RADIUS, i);
    float pixels = 2.0 * radius * u.camPos.w / max(distance(unit.pos, u.mirrorCamera.xyz), 1e-3);
    unit.size = radius * max(1.0, u.sun.w / max(pixels, 1e-6));
    return unit;
}

// The reflection's lists, filled by the same dispatch (`units.cull` with
// the reflection's buffers): words.y = 1, words.z the visible list,
// words.w the indirect arguments (water/reflect.rs push_buffers), in the
// main lists' layout (the darts' count at word 1, then the commands).
// Declared apart so the main lists' macros do not rename them.
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) writeonly buffer MirrorVisible { uint mirrorVisible[]; } ae_MirrorVisible[];
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) buffer MirrorIndirect { uint mirrorWords[]; } ae_MirrorIndirect[];
#define mirrorVisible ae_MirrorVisible[ae.words.z].mirrorVisible
#define mirrorWords ae_MirrorIndirect[ae.words.w].mirrorWords
bool mirrorListsBound() { return ae.words.y != 0u; }

bool inMirrorView(vec3 centre, float r) {
    for (int k = 0; k < 6; k++) {
        if (dot(u.mirrorPlanes[k].xyz, centre) + u.mirrorPlanes[k].w < -r) {
            return false;
        }
    }
    return true;
}
#endif

// The water overlay's commands in the indirect buffer (models.rs
// overlay_offset, in words): after the darts' command, the camera's model
// commands and each shadow cascade's copy; laid out as the camera's. In
// refraction frames the SUB cull splits the camera's lists by eligible():
// Own and Seen rows from the front of each region under the camera's
// commands (drawn before the water copy), blips and ghosts from the back
// under the overlay's (drawn over the water, units.water_overlay).
uint overlayWord() { return 4u + 5u * u.counts.w * (1u + uint(UNIT_CASCADES)); }

#ifndef CULL
// The row of instance `instance` in the list region at `base`: from the
// back for the water overlay's draw (SUB, ae.words.x = 1).
uint drawnRow(uint base, uint instance) {
#ifdef SUB
    if (ae.words.x != 0u) {
        return visible[base + u.counts.y - 1u - instance];
    }
#endif
    return visible[base + instance];
}
#endif

// The class is drawn with a model (src/models.rs), not a dart.
bool hasModel(uint cls) {
    return (uint(u.classes[cls].w) & 2u) != 0u;
}
