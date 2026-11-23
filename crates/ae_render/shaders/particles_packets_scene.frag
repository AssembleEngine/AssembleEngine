#version 460
// The packets' full-resolution colour and glow over the scene, one to one:
// the targets are the scene's size, so there is no depth-aware upsample.
// The targets hold physical radiance and the scene exposed radiance.
#include "particles_common.glsl"

#define particleColor AE_S2D(0, 0)
#define unusedDepth AE_S2D(1, 0)
#define unusedDepth2 AE_S2D(2, 0)
#define particleGlow AE_S2D(3, 0)

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outGlow;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 c = texelFetch(particleColor, p, 0);
    if (c == vec4(0.0)) {
        discard;
    }
    vec3 glow = texelFetch(particleGlow, p, 0).rgb;
    c.rgb *= u.photometry.x;
    outColor = c;
    outGlow = vec4(glow * u.photometry.x, c.a);
}
