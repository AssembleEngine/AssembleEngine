#version 460
// bloom.blur_x: an 11-tap Gaussian (sigma 2 quarter-resolution texels;
// +-5 is the 4% tail) along the image's x, as RA3-graphics' second pass.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
#include "bindless.glsl"
// Bindless handle: bloom 0; sampler 0.
#define bloom AE_S2D(0, 0)

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 last = textureSize(bloom, 0) - 1;
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    for (int k = -5; k <= 5; ++k) {
        float w = exp(-float(k * k) / 8.0);
        acc += w * texelFetch(bloom, clamp(p + ivec2(k, 0), ivec2(0), last), 0).rgb;
        wsum += w;
    }
    color = vec4(acc / wsum, 1.0);
}
