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
    outColor = inColor * k;
}
