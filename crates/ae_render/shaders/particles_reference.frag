#version 460
// Reference, pass 1: append this fragment to its pixel's list.
#include "particles_common.glsl"
#include "particles_frag.glsl"
#include "particles_reference.glsl"

void main() {
    float front;
    float k = coverage(front);
    if (k <= 0.0) {
        return;
    }
    vec4 c = inColor * k;
    uint pixel = uint(gl_FragCoord.y) * uint(textureSize(depthTex, 0).x) + uint(gl_FragCoord.x);
    uint n = atomicAdd(refCounts[pixel], 1u);
    if (n < REFERENCE_LAYERS) {
        refFragments[pixel * REFERENCE_LAYERS + n] = uvec4(floatBitsToUint(front), packHalf2x16(c.rg), packHalf2x16(c.ba), 0u);
    }
}
