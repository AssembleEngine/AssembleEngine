#version 460
// Slices, pass 3: front to back, each slice a homogeneous medium of its
// summed optical depth and light.
#ifndef AE_BINDLESS_SET
#define AE_BINDLESS_SET 3
#endif
#include "bindless.glsl"
#define slice0 AE_S2D(3, 0)
#define slice1 AE_S2D(4, 0)
#define slice2 AE_S2D(5, 0)
#define slice3 AE_S2D(6, 0)

layout(location = 0) out vec4 outColor;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 s[4] = vec4[](texelFetch(slice0, p, 0), texelFetch(slice1, p, 0), texelFetch(slice2, p, 0), texelFetch(slice3, p, 0));
    vec3 c = vec3(0.0);
    float t = 1.0;
    for (int j = 0; j < 4; ++j) {
        float tau = s[j].a;
        if (tau > 1e-4) {
            float tr = exp(-tau);
            c += t * s[j].rgb / tau * (1.0 - tr);
            t *= tr;
        } else {
            c += t * s[j].rgb;
        }
    }
    outColor = vec4(c, 1.0 - t);
}
