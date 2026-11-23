#version 460
// bloom.downsample: the glow target to one texel per 4x4 of the scene, at
// a quarter of the size (RA3-graphics' first bloom pass). Pass 0 keeps the
// brightest of four 2x2 means: their mean would thin a one- or two-pixel
// glint to nothing by the time it is blurred, and the single brightest
// texel would flicker with the water's glints.
//
// The glow holds only light sources (a glowing decal, a water glint, an
// emissive model or particle), stored at the camera's reference exposure.
// Auto-exposure's display scale joins here, before the clamp, so the clamp
// is exposure-relative: 4.0 of display-referred light whatever the EV does
// (as RA3 clamps): a sun glint many times the display's white would
// otherwise bloom into a large blob.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform Bloom {
    vec4 params; // x: the EV100 the scene was stored at, y: 1 when the graph has the exposure image
} b;
#include "bindless.glsl"
// Bindless handles: glow 0, exposure 1; sampler 0.
#define glow AE_S2D(0, 0)
#define exposureTex AE_S2D(1, 0)

const float BLOOM_CLAMP = 4.0;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 inv = 1.0 / vec2(textureSize(glow, 0));
    float scale = 1.0;
    if (b.params.y > 0.5) {
        float ev = texelFetch(exposureTex, ivec2(0), 0).x;
        scale = clamp(exp2(b.params.x - ev), 1.0 / 256.0, 256.0);
    }
    vec3 acc = vec3(0.0);
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            vec3 g = textureLod(glow, (vec2(p * 4) + vec2(1 + 2 * i, 1 + 2 * j)) * inv, 0.0).rgb * scale;
            acc = max(acc, min(g, vec3(BLOOM_CLAMP)));
        }
    }
    color = vec4(acc, 1.0);
}
