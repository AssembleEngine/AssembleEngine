#version 460
// The water surface: one quad at the water level over the whole map (two
// triangles from gl_VertexIndex). The depth test against the terrain and
// units decides where it shows, so it needs no shoreline geometry, and it
// stops at the map's edges like the terrain.
#define GRAPHICS
#include "terrain_common.glsl"

layout(location = 0) out vec3 outWorld;

#ifdef WATER_MESH
#define WAVE_SAMPLE 14
#include "water_patch.glsl"
#else
const vec2 CORNERS[6] = vec2[](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));

#endif

void main() {
    vec2 extent = u.map.xy * u.map.w;
#ifdef WATER_MESH
    Leaf leaf = leaves[gl_InstanceIndex];
    ivec2 a = leaf.a.xy, b = leaf.a.zw, c0 = leaf.b.xy;
    uint tri = uint(gl_VertexIndex) / 3u;
    // Four bisections, the same 16-triangle submesh as terrain.
    for (int i = 3; i >= 0; --i) {
        ivec2 midpoint = (a + c0) / 2;
        if ((tri & (1u << i)) == 0u) {
            c0 = b;
            b = midpoint;
        } else {
            a = b;
            b = midpoint;
        }
    }
    ivec2 lattice = ivec2[](a, b, c0)[gl_VertexIndex % 3];
    vec2 c = extent * (vec2(lattice) / float(1 << LATTICE_BITS));
#else
    vec2 c = CORNERS[gl_VertexIndex] * extent;
#endif
    outWorld = vec3(c.x, u.water.x, c.y);
#ifdef WATER_MESH
    outWorld.y += waterDisplacement(c);
#endif
    gl_Position = u.viewProj * vec4(outWorld, 1.0);
}
