#version 460
// Sorted particles: premultiplied "over", drawn back to front.
#include "particles_common.glsl"
#include "particles_frag.glsl"

layout(location = 0) out vec4 outColor;

void main() {
    float front;
    float k = coverage(front);
    if (k <= 0.0) {
        discard;
    }
    vec4 c = inColor * k;
    // Full resolution draws straight into the scene (downscale 1): the
    // half-resolution path is exposed by the composite.
    if ((u.flags.w & 1u) != 0u) {
        c.rgb *= u.photometry.x;
    }
    outColor = c;
}
