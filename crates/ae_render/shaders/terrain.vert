#version 460
// One instance per visible CBT leaf; the vertex buffer is the shared 16-triangle LEB
// sub-mesh given as integer barycentric weights. All positions are computed on the
// integer lattice, so vertices shared by neighbouring leaves are bit-identical.
#define GRAPHICS
#include "terrain_common.glsl"

layout(location = 0) in ivec4 inWeights; // (w0, w1, w2, 0), sum = SUBMESH_DENOM

layout(location = 0) out vec3 outWorld;
layout(location = 1) flat out uint outDepth;

out gl_PerVertex {
    vec4 gl_Position;
#ifdef MIRROR
    float gl_ClipDistance[5];
#else
    float gl_ClipDistance[4];
#endif
};

void main() {
    Leaf leaf = leaves[gl_InstanceIndex];
    ivec2 v0 = leaf.a.xy, v1 = leaf.a.zw, v2 = leaf.b.xy;
    ivec2 p = (inWeights.x * v0 + inWeights.y * v1 + inWeights.z * v2) / SUBMESH_DENOM;

    vec2 cell = latticeToCell(p);
    float h = sampleHeight(cell);
    vec3 world = vec3(cell.x * u.map.w, h, cell.y * u.map.w);

#ifdef MIRROR
    gl_Position = u.mirrorProj * vec4(world, 1.0);
    gl_ClipDistance[4] = h - u.water.x;
#else
    gl_Position = u.viewProj * vec4(world, 1.0);
#endif
    // The LEB square is padded to a power of two: clip against the real map rectangle.
    gl_ClipDistance[0] = cell.x;
    gl_ClipDistance[1] = u.map.x - cell.x;
    gl_ClipDistance[2] = cell.y;
    gl_ClipDistance[3] = u.map.y - cell.y;
    outWorld = world;
    outDepth = depthOf(uint(leaf.b.z));
}
