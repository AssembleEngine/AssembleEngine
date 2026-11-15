#version 460
// The water surface: one quad at the water level over the whole map (two
// triangles from gl_VertexIndex). The depth test against the terrain and
// units decides where it shows, so it needs no shoreline geometry, and it
// stops at the map's edges like the terrain.
#define GRAPHICS
#include "terrain_common.glsl"

layout(location = 0) out vec3 outWorld;

const vec2 CORNERS[6] = vec2[](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));

void main() {
    vec2 extent = u.map.xy * u.map.w;
    vec2 c = CORNERS[gl_VertexIndex] * extent;
    outWorld = vec3(c.x, u.water.x, c.y);
    gl_Position = u.viewProj * vec4(outWorld, 1.0);
}
