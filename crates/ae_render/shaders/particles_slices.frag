#version 460
// Slices, pass 2: the particle as a medium (particles_frag.glsl medium)
// added into the pixel's slices, equal parts of its depth range, by how
// much of each its chord covers: light and optical depth (additive).
#include "particles_common.glsl"
#include "particles_frag.glsl"

#define sliceRange AE_S2D(2, 0)

layout(location = 0) out vec4 outSlice0;
layout(location = 1) out vec4 outSlice1;
layout(location = 2) out vec4 outSlice2;
layout(location = 3) out vec4 outSlice3;

void main() {
    float front, back, len;
    bool covered = chord(front, back, len);
    float sigma;
    vec3 source;
    medium(inColor * texture(puffTex, inCorner * 0.5 + 0.5).r, len, sigma, source);
    vec2 r = texelFetch(sliceRange, ivec2(gl_FragCoord.xy), 0).xy;
    float lo = 1.0 / max(r.x, 1e-30);
    float w = (r.y - lo) * 0.25;
    // The chord's length in each slice.
    vec4 z0 = lo + vec4(0.0, 1.0, 2.0, 3.0) * w;
    vec4 o = covered ? max(min(vec4(back), z0 + w) - max(vec4(front), z0), 0.0) : vec4(0.0);
    outSlice0 = vec4(source, sigma) * o.x;
    outSlice1 = vec4(source, sigma) * o.y;
    outSlice2 = vec4(source, sigma) * o.z;
    outSlice3 = vec4(source, sigma) * o.w;
}
