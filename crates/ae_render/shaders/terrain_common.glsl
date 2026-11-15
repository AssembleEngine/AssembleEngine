// Shared declarations + CBT/LEB routines. Mirrors src/terrain/leb.rs (which is unit-tested).
// Lifted from ~/map-rendering/shaders/common.glsl.

// Graphics shaders define GRAPHICS: buffers become read-only and CBT writers are omitted.
#ifdef GRAPHICS
#define BUF_QUAL readonly
#else
#define BUF_QUAL
#endif

#define LATTICE_BITS 20
#define SUBMESH_DENOM 4   // 2^(SUBMESH_LEVELS/2), SUBMESH_LEVELS = 4 (must stay even)
#define PYRAMID_BASE_LEVEL 2

#define FLAG_LOD_COLORS 1u
#define FLAG_WIRE_TRIS 2u
#define FLAG_WIRE_QUADS 4u
#define FLAG_MATERIALS 8u
#define FLAG_TILESET 16u
#define FLAG_WEIGHTS 32u
#define FLAG_VISION 64u
#define FLAG_WATER 128u   // the water surface is drawn (terrain: caustics under it)
#define FLAG_SHADOWS 256u // shade by the hills' shadows (sunShadowTex)

layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 viewProj;
    vec4 camPos;         // render camera
    vec4 lodCamPos;      // camera driving the subdivision (differs when frozen)
    vec4 lodFrustum[6];  // planes of the LOD camera, inside: dot(n,p)+w >= 0
    vec4 drawFrustum[6]; // planes of the render camera
    vec4 lod;            // x: target error (px), y: projection scale (px per unit at dist 1),
                         // z: error tolerance multiplier outside the frustum, w: merge hysteresis
    vec4 map;            // x,y: map size in cells, z: root square size in cells, w: cell size
    vec4 height;         // x: height scale, y: height offset, z,w: 1/samples (texel size)
    vec4 sun;            // xyz: direction to the sun
    uvec4 cfg;           // x: CBT max depth D, y: T = D-5, z: flags, w: draw-list capacity
    vec4 fog;            // rgb: fog colour, w: density
    vec4 water;          // x: water surface height (world), y: 1 if the map has water, z: extinction per unit depth,
                         // w: animation time (s; also without water)
    vec4 vision;         // vision grid: x,y size in cells, z: cell size, w: u32 words per row
    // The units' shadow cascades (src/shadow.rs, unit_shadows.glsl).
    vec4 unitShadowAxes[3];   // light axes: right, up, toward the sun
    vec4 unitShadowOrigin[3]; // per cascade: centre x, y, depth origin; w: world texel size
    vec4 unitShadowScale[3];  // per cascade: 1/half-size x, y, 1/depth range; w: 1 = on
} u;

layout(std430, set = 0, binding = 1) BUF_QUAL buffer TreeBuf { uint tree[]; };    // sum tree (depth 0..T)
layout(std430, set = 0, binding = 2) BUF_QUAL buffer BitBuf { uint bits[]; };     // CBT bitfield
layout(std430, set = 0, binding = 3) readonly buffer SnapBuf { uint snap[]; };    // bitfield snapshot
struct Leaf { ivec4 a; ivec4 b; };  // a = (v0, v1), b = (v2, heap id, 0)
layout(std430, set = 0, binding = 4) BUF_QUAL buffer LeafBuf { Leaf leaves[]; };  // draw list
// [0..3) update dispatch, [3..8) DrawIndexedIndirectCommand, [8] leaf count
layout(std430, set = 0, binding = 5) BUF_QUAL buffer IndirectBuf { uint ind[]; };
layout(set = 0, binding = 6) uniform sampler2D pyramid;   // (min, max, E_L, max E_{L-2}) per block
layout(set = 0, binding = 7) uniform sampler2D heightTex; // normalised heights, linear filtering
layout(set = 0, binding = 8) uniform sampler2D errorTex;  // mip m: E_(m+1), exact error at spacing 2^(m+1)
layout(set = 0, binding = 9) uniform usampler2D materialTex; // material id per cell (FLAG_MATERIALS), else 1x1

#define IND_DRAW_INSTANCES 4
#define IND_LEAF_COUNT 8

// --------------------------------------------------------------------------------------
// CBT (Dupuy 2020). Leaves at max depth D are bits; the sum tree is explicit for depth <= T,
// the 5 levels below are popcounts of one 32-bit word.
// --------------------------------------------------------------------------------------

uint depthOf(uint id) { return uint(findMSB(id)); }

// Number of leaves below a node, evaluated on the snapshot (consistent with `tree`).
uint nodeSum(uint id) {
    uint d = depthOf(id), t = u.cfg.y;
    if (d <= t) return tree[id];
    uint s = d - t;
    uint word = snap[(id >> s) - (1u << t)];
    uint width = 32u >> s;
    uint offset = (id & ((1u << s) - 1u)) * width;
    return bitCount(bitfieldExtract(word, int(offset), int(width)));
}

bool isLeaf(uint id) { return nodeSum(id) == 1u; }

uint decodeLeaf(uint handle) {
    uint id = 1u;
    while (nodeSum(id) > 1u) {
        uint c = nodeSum(2u * id);
        if (handle < c) {
            id = 2u * id;
        } else {
            handle -= c;
            id = 2u * id + 1u;
        }
    }
    return id;
}

uint bitIndex(uint id, uint depth) {
    uint D = u.cfg.x;
    return (id << (D - depth)) - (1u << D);
}

#ifndef GRAPHICS
void splitNode(uint id) {
    uint d = depthOf(id);
    if (d < u.cfg.x) {
        uint b = bitIndex(2u * id + 1u, d + 1u);
        atomicOr(bits[b >> 5], 1u << (b & 31u));
    }
}

void mergeNode(uint id) {
    uint b = bitIndex(id | 1u, depthOf(id));
    atomicAnd(bits[b >> 5], ~(1u << (b & 31u)));
}
#endif

// --------------------------------------------------------------------------------------
// LEB on a square made of two root triangles (heap ids 2, 3), integer lattice vertices.
// Triangle = (v0, v1, v2), v1 = right-angle apex, v0-v2 = hypotenuse.
// --------------------------------------------------------------------------------------

uint getBit(uint id, int b) { return b < 0 ? 0u : (id >> uint(b)) & 1u; }

void decodeTriangle(uint id, out ivec2 v0, out ivec2 v1, out ivec2 v2) {
    int depth = int(depthOf(id));
    const int R = 1 << LATTICE_BITS;
    v0 = ivec2(0, R); v1 = ivec2(0, 0); v2 = ivec2(R, 0);
    if (getBit(id, depth - 1) == 1u) {
        ivec2 a = v2, b = v0 + v2 - v1, c = v0;
        v0 = a; v1 = b; v2 = c;
    }
    for (int b = depth - 2; b >= 0; --b) {
        ivec2 m = (v0 + v2) / 2;
        if (getBit(id, b) == 0u) {
            v2 = v1; v1 = m;
        } else {
            v0 = v1; v1 = m;
        }
    }
}

// Same-depth neighbours (0 = none): x = left (across v1-v2), y = right (across v0-v1),
// z = edge (across the hypotenuse v0-v2), w = the node itself.
uvec4 neighbors(uint id) {
    int depth = int(depthOf(id));
    uint b = getBit(id, max(0, depth - 1));
    uvec4 n = uvec4(0u, 0u, 3u - b, 2u + b); // left, right, edge, self
    for (int bb = depth - 2; bb >= 0; --bb) {
        uint b2 = n.y == 0u ? 0u : 1u;
        uint b3 = n.z == 0u ? 0u : 1u;
        if (getBit(id, bb) == 0u) {
            n = uvec4(n.w << 1 | 1u, n.z << 1 | b3, n.y << 1 | b2, n.w << 1);
        } else {
            n = uvec4(n.z << 1, n.w << 1, n.x << 1, n.w << 1 | 1u);
        }
    }
    return n;
}

uint edgeNeighbor(uint id) { return neighbors(id).z; }

#ifndef GRAPHICS
void splitConforming(uint id) {
    if (depthOf(id) >= u.cfg.x) return;
    splitNode(id);
    uint n = edgeNeighbor(id);
    while (n >= 2u) {
        splitNode(n);
        n >>= 1;
        if (n >= 2u) {
            splitNode(n);
            n = edgeNeighbor(n);
        }
    }
}
#endif

// --------------------------------------------------------------------------------------
// Terrain-space helpers
// --------------------------------------------------------------------------------------

vec2 latticeToCell(ivec2 p) { return vec2(p) * (u.map.z / float(1 << LATTICE_BITS)); }

float sampleHeight(vec2 cell) {
    vec2 uv = (cell + 0.5) * u.height.zw;
    return textureLod(heightTex, uv, 0.0).r * u.height.x + u.height.y;
}

// --------------------------------------------------------------------------------------
// LOD error
// --------------------------------------------------------------------------------------

// Height bounds of the cell-space box [a, b] (clamped to the map), from the 2x2 pyramid
// texels covering it. Used for culling and for the camera distance.
void boxHeights(vec2 a, vec2 b, out float hmin, out float hmax) {
    int K = findMSB(int(u.map.z));
    int L = clamp(int(ceil(log2(max(max(b.x - a.x, b.y - a.y), 1.0)))), PYRAMID_BASE_LEVEL, K);
    int n = int(u.map.z) >> L;
    ivec2 t0 = clamp(ivec2(floor(a / float(1 << L))), ivec2(0), ivec2(n - 1));
    ivec2 t1 = clamp(ivec2(floor(b / float(1 << L))), ivec2(0), ivec2(n - 1));
    int mip = L - PYRAMID_BASE_LEVEL;
    vec4 s00 = texelFetch(pyramid, t0, mip);
    vec4 s10 = texelFetch(pyramid, ivec2(t1.x, t0.y), mip);
    vec4 s01 = texelFetch(pyramid, ivec2(t0.x, t1.y), mip);
    vec4 s11 = texelFetch(pyramid, t1, mip);
    hmin = min(min(s00.r, s10.r), min(s01.r, s11.r));
    hmax = max(max(s00.g, s10.g), max(s01.g, s11.g));
}

// Signed distance from p to the line p0->p1, positive on the side of `inner`.
float edgeDist(vec2 p, vec2 p0, vec2 p1, vec2 inner) {
    vec2 e = p1 - p0;
    vec2 nrm = normalize(vec2(-e.y, e.x));
    return dot(p - p0, nrm) * sign(dot(inner - p0, nrm));
}

// Max of E_l (error of a mesh with spacing 2^l) over the level-l blocks that overlap the
// triangle (c0, c1, c2); with `useTri` false, over all blocks overlapping its box [a, b].
float levelError(int l, vec2 c0, vec2 c1, vec2 c2, vec2 a, vec2 b, bool useTri) {
    if (l <= 0) return 0.0; // E_0 = 0: the mesh is the samples
    l = min(l, findMSB(int(u.map.z)));
    float ts = float(1 << l);
    int n = int(u.map.z) >> l;
    ivec2 t0 = clamp(ivec2(floor(a / ts)), ivec2(0), ivec2(n - 1));
    ivec2 t1 = clamp(ivec2(floor(b / ts)), ivec2(0), ivec2(n - 1));
    float r = ts * 0.7072; // block half-diagonal: conservative overlap test
    float e = 0.0;
    for (int y = t0.y; y <= t1.y; ++y) {
        for (int x = t0.x; x <= t1.x; ++x) {
            vec2 c = (vec2(x, y) + 0.5) * ts;
            if (useTri && (edgeDist(c, c0, c1, c2) < -r || edgeDist(c, c1, c2, c0) < -r
                    || edgeDist(c, c2, c0, c1) < -r)) continue;
            e = max(e, texelFetch(errorTex, ivec2(x, y), l - 1).r);
        }
    }
    return e;
}

// Geometric error of drawing a triangle with legs `leg` cells long as its sub-mesh (vertex
// spacing leg / SUBMESH_DENOM): E at that spacing, interpolated between the two power-of-two
// levels around it. Legs shrink by sqrt(2) per LEB depth, so every depth gets a distinct
// error (bounding boxes can't tell an axis-aligned triangle from its diagonal children).
float meshError(float leg, vec2 c0, vec2 c1, vec2 c2, vec2 a, vec2 b, bool useTri) {
    float ls = log2(max(leg / float(SUBMESH_DENOM), 1e-3));
    int l0 = int(floor(ls));
    float e0 = levelError(l0, c0, c1, c2, a, b, useTri);
    float e1 = levelError(l0 + 1, c0, c1, c2, a, b, useTri);
    return mix(e0, e1, ls - float(l0));
}

// Screen-space size (pixels) of a vertical world-space error `err` anywhere in the box
// [bmin, bmax], seen from the LOD camera. A vertical offset projects to the screen in
// proportion to the sine of the angle between the view ray and the vertical (Lindstrom et
// al. 1996): height errors seen from above barely move on screen (shading comes from
// per-pixel normals anyway), while at grazing angles - distant terrain, silhouettes - they
// are fully visible. We take the nearest distance and the largest sine over the box, an
// upper bound that keeps parents >= children (stable split/merge).
float projectError(float err, vec3 bmin, vec3 bmax) {
    vec3 cam = u.lodCamPos.xyz;
    vec3 c = clamp(cam, bmin, bmax);
    float dist = max(distance(c, cam), 1.0);
    float horiz = length(max(abs(bmin.xz - cam.xz), abs(bmax.xz - cam.xz)));
    float vert = abs(c.y - cam.y);
    float sine = horiz / max(sqrt(horiz * horiz + vert * vert), 1e-3);
    return err * u.lod.y * max(sine, 0.15) / dist;
}

vec3 cellToWorldBox(vec2 cell, float h) { return vec3(cell.x * u.map.w, h, cell.y * u.map.w); }
