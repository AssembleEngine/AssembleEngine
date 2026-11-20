#version 460
// bloom.downsample: the glow target to one texel per 4x4 of the scene, at
// a quarter of the size (RA3-graphics' first bloom pass). Pass 0 keeps the
// brightest of four 2x2 means: their mean would thin a one- or two-pixel
// glint to nothing by the time it is blurred, and the single brightest
// texel would flicker with the water's glints.
//
// The glow holds only light sources (a glowing decal, a water glint, an
// emissive model or particle), already exposed like the scene. The clamp is
// on exposed light (as RA3 clamps the same way): a sun glint many times the
// display's white would otherwise bloom into a large blob.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 1) uniform sampler2D glow;

const float BLOOM_CLAMP = 4.0;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 inv = 1.0 / vec2(textureSize(glow, 0));
    vec3 acc = vec3(0.0);
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            acc = max(acc, min(textureLod(glow, (vec2(p * 4) + vec2(1 + 2 * i, 1 + 2 * j)) * inv, 0.0).rgb, vec3(BLOOM_CLAMP)));
        }
    }
    color = vec4(acc, 1.0);
}
