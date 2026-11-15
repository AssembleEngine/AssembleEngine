#version 460
// Slices, pass 1: the pixel's depth range of particles, max-blended: the
// reciprocal of the nearest front, the farthest back (0, 0: none).
#include "particles_common.glsl"
#include "particles_frag.glsl"

layout(location = 0) out vec2 outRange;

void main() {
    float front, back, len;
    outRange = chord(front, back, len) ? vec2(1.0 / front, back) : vec2(0.0);
}
