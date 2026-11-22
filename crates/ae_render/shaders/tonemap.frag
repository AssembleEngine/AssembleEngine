#version 460
// post.tonemap (src/tonemap.rs): the HDR scene (linear light, stored at the
// camera's reference exposure) to the output. There is no tone curve. With
// auto-exposure the stored scene is scaled by the metered, eased EV read from
// the 1x1 exposure image before the bloom joins it (the bloom is already
// display-referred: bloom.downsample applies the same scale). The output is
// the only limit: in SDR the output's format encodes (an sRGB attachment,
// which saturates at 1). In HDR the target is the display intermediate
// (RGBA16F, paper-white units) and post.encode encodes it, clipping at the
// display's peak.
#include "gamut.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform Tonemap {
    vec4 params;   // x: bloom strength, y: 1 when the graph has a bloom,
                   // z: 1 when it has the exposure image, w: the EV100 the
                   // scene was stored at
    mat3 toOutput; // the working space to the gamut written: SDR's BT.709,
                   // or in HDR the working space itself (the identity)
    vec4 outLuma;  // xyz: the written gamut's luminance row (the compression);
                   // w: 1 when it is narrower than the working space
} t;
layout(set = 0, binding = 1) uniform sampler2D scene;
layout(set = 0, binding = 2) uniform sampler2D bloom;
layout(set = 0, binding = 3) uniform sampler2D exposureTex;

void main() {
    vec3 c = max(texelFetch(scene, ivec2(gl_FragCoord.xy), 0).rgb, vec3(0.0));
    if (t.params.z > 0.5) {
        // EV100 is logarithmic: the display's exposure over the scene's is
        // 2^(scene EV - eased EV). Clamped so a garbage texel cannot blow
        // the image up or out.
        float ev = texelFetch(exposureTex, ivec2(0), 0).x;
        c *= clamp(exp2(t.params.w - ev), 1.0 / 256.0, 256.0);
    }
    // The bloom joins the scene's light (RA3's post.frag).
    if (t.params.y > 0.5) {
        c += texture(bloom, uv).rgb * t.params.x;
    }
    // SDR's output is BT.709; a BT.2020 working space is mapped into it
    // (and out-of-gamut colour compressed) here, since SDR has no
    // post.encode. In HDR the display stays in the working space (the
    // identity) and post.encode maps it. At the default working space the
    // matrix is the identity and nothing is compressed.
    c = t.toOutput * c;
    if (t.outLuma.w > 0.5) {
        c = aeGamutCompress(c, t.outLuma.xyz);
    }
    color = vec4(c, 1.0);
}
