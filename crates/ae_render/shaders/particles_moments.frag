#version 460
// Moment-based OIT (Muenstermann et al. 2018), pass 1: each medium
// fragment adds its absorbance and the absorbance-weighted powers of its
// warped depth (four power moments). Additive, so in any order.
#include "particles_common.glsl"
#include "particles_frag.glsl"

layout(location = 0) out float outB0;
layout(location = 1) out vec4 outMoments;

void main() {
    float front;
    float a = inColor.a * coverage(front);
    if (a <= 0.0) {
        discard;
    }
    float absorbance = -log(1.0 - min(a, 0.9999));
    float z = warpDepth(front);
    float z2 = z * z;
    outB0 = absorbance;
    outMoments = absorbance * vec4(z, z2, z2 * z, z2 * z2);
}
