// Set 1 of a units renderer whose classes have models (src/models.rs):
// shared by units_models_cull.comp, units_pose.comp (which defines POSE),
// units_models.vert/.frag and units_models_shadow.vert. Include after
// units_common.glsl.

// Vertex words of every model, packed per primitive in its layout
// (ae_model::Layout): position 0..2, normal 3..5, tangent 6..9 (w: the
// bitangent's sign), uv0 10..11, uv1 12..13, colour 14..15 (RGBA unorm16);
// rigid (17 words): joint 16; skinned (20): joints 16..17 (u16), weights
// 18..19 (unorm16). The indices follow (bound as the index buffer).
layout(std430, set = 1, binding = 0) readonly buffer Geometry { uint vwords[]; };

struct Material {
    vec4 baseColor;
    vec4 emissive;  // rgb; w: alpha cutoff (negative: opaque)
    vec4 params;    // normal scale, occlusion strength, double-sided
    uvec4 tex;      // texture slots: base colour, normal, occlusion, emissive (0: none)
    uvec4 uvSet;
    vec4 xform[8];  // per texture: 2x2 matrix (columns), offset
};
layout(std430, set = 1, binding = 1) readonly buffer Materials { Material materials[]; };
// Skinning matrices, 3x4: each column holds a row of the affine transform
// (p' = vec4(p, 1) * m). The models' rest palettes, then a range per unit
// with animation records (rows' pose offsets point into either), then the pose
// pass's scratch.
#ifdef POSE
layout(std430, set = 1, binding = 2) buffer Palette { mat3x4 palette[]; };
#else
layout(std430, set = 1, binding = 2) readonly buffer Palette { mat3x4 palette[]; };
#endif

struct ModelClass {
    vec4 sphere; // bounding sphere (at rest, or over its clips), fitted units
    uvec4 cmd;   // first entry in classCommands, commands (0: a dart), rest palette offset, joints
    vec4 fit;    // x: 1 / footprint radius
    uvec4 pose;  // in anim[]: first bone, bones, first joint, the clips' head
};
layout(std430, set = 1, binding = 3) readonly buffer Classes { ModelClass modelClasses[MAX_CLASSES]; };

// One per indirect command (found by gl_DrawID + pc.drawBase).
struct DrawRecord {
    uint material;
    uint cls;
    uint visibleBase; // the class's region of the visible list (then one per cascade)
    uint vertexBase;  // first word of the primitive's vertices
    uint words;       // per vertex: 17 rigid, 20 skinned
    uint pad0;
    uint pad1;
    uint pad2;
};
layout(std430, set = 1, binding = 4) readonly buffer Draws { DrawRecord draws[]; };
// Per class its commands' indices (the cull counts each).
layout(std430, set = 1, binding = 5) readonly buffer ClassCommands { uint classCommands[]; };
// Per frame: [capacity] each row's palette offset, [capacity] the rows
// with animation records (the pose jobs), [capacity] each job's scratch offset.
layout(std430, set = 1, binding = 6) readonly buffer Rows { uint rows[]; };
// Partially bound (models::MAX_TEXTURES); indexed per draw, so uniformly.
layout(set = 1, binding = 7) uniform sampler2D textures[1024];
#ifdef POSE
// The models' animation table (models.rs, animation()): per model its bones
// (rest rows 0..2, parent in 3.x), joints (inverse bind rows 0..2, bone in
// 3.x), and from ModelClass.pose.w its clips (the sim's bake), masks and
// rest transforms. Words (uint) are indexed over the whole table, four a
// vec4.
layout(std430, set = 1, binding = 8) readonly buffer Anim { vec4 anim[]; };
uint animWord(uint w) { return floatBitsToUint(anim[w >> 2u][w & 3u]); }
// The units' animation table (units.rs RECORD_CELLS): two records per
// posed unit from its pose attribute. units_anim_writes.comp writes it.
#define RECORD_CELLS 15u
#ifdef ANIM_WRITES
layout(std430, set = 1, binding = 9) writeonly buffer Records { uvec4 records[]; };
#else
layout(std430, set = 1, binding = 9) readonly buffer Records { uvec4 records[]; };
#endif
// The frame's table writes (u.anim.z of them): cell i goes to table cell
// stagedAt[i].
layout(std430, set = 1, binding = 10) readonly buffer Staged { uvec4 staged[]; };
layout(std430, set = 1, binding = 11) readonly buffer StagedAt { uint stagedAt[]; };
#endif

// cascade: the shadow caster's (units_pose.comp: the number of jobs);
// drawBase: the multi-draw's first command.
layout(push_constant) uniform Push { uint cascade; uint drawBase; } pc;

uint poseOffset(uint row) { return rows[row]; }

struct Skinned {
    vec3 pos;
    vec3 normal;
    vec4 tangent;
    vec4 uv;    // uv0, uv1
    vec4 color;
};

float vword(uint b, uint k) { return uintBitsToFloat(vwords[b + k]); }

// Vertex v of draw d skinned by the palette from `paletteBase`, in model
// space.
Skinned skinVertex(DrawRecord d, uint v, uint paletteBase) {
    uint b = d.vertexBase + v * d.words;
    vec3 p = vec3(vword(b, 0u), vword(b, 1u), vword(b, 2u));
    vec3 n = vec3(vword(b, 3u), vword(b, 4u), vword(b, 5u));
    vec4 t = vec4(vword(b, 6u), vword(b, 7u), vword(b, 8u), vword(b, 9u));
    mat3x4 m;
    if (d.words == 17u) {
        m = palette[paletteBase + vwords[b + 16u]];
    } else {
        uint j01 = vwords[b + 16u];
        uint j23 = vwords[b + 17u];
        vec4 w = vec4(unpackUnorm2x16(vwords[b + 18u]), unpackUnorm2x16(vwords[b + 19u]));
        m = palette[paletteBase + (j01 & 0xffffu)] * w.x
          + palette[paletteBase + (j01 >> 16u)] * w.y
          + palette[paletteBase + (j23 & 0xffffu)] * w.z
          + palette[paletteBase + (j23 >> 16u)] * w.w;
    }
    Skinned s;
    s.pos = vec4(p, 1.0) * m;
    // The linear part (columns).
    mat3 a = transpose(mat3(m[0].xyz, m[1].xyz, m[2].xyz));
    // Normals by the cofactor matrix (the inverse transpose up to scale):
    // right under non-uniform scale.
    mat3 cof = mat3(cross(a[1], a[2]), cross(a[2], a[0]), cross(a[0], a[1]));
    s.normal = cof * n;
    s.tangent = vec4(a * t.xyz, t.w);
    s.uv = vec4(vword(b, 10u), vword(b, 11u), vword(b, 12u), vword(b, 13u));
    s.color = vec4(unpackUnorm2x16(vwords[b + 14u]), unpackUnorm2x16(vwords[b + 15u]));
    return s;
}

// A model-space direction in the unit's frame (place() without the move
// and the scale).
vec3 orient(Unit unit, vec3 d) {
    vec3 forward = vec3(unit.heading.x, 0.0, unit.heading.y);
    vec3 side = vec3(-unit.heading.y, 0.0, unit.heading.x);
    return forward * d.x + vec3(0.0, d.y, 0.0) + side * d.z;
}
