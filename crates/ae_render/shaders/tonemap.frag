#version 460
// post.tonemap (src/tonemap.rs): the HDR scene (linear light, stored at the
// camera's reference exposure) to the output by the tone curve: 0 is none
// (the colour passes as is: an SDR target saturates at 1, and in HDR
// post.encode clips at the display's peak), 1 is Lottes 2016. With auto-exposure the
// stored scene is scaled by the metered, eased EV read from the 1x1 exposure
// image before the bloom joins it (the bloom is already display-referred:
// bloom.downsample applies the same scale). The curve maps the max channel
// and scales the others with it (hue kept); above paper white a share of the
// luminance-mapped grey joins in, so a fireball's core whitens while its edge
// stays orange. In SDR the output's format encodes (an sRGB attachment). In
// HDR the target is the display intermediate (RGBA16F, paper-white units,
// up to the peak) and post.encode encodes that.
#include "gamut.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform Tonemap {
    vec4 params;   // x: bloom strength, y: curve (0 none, 1 Lottes), z: 1 when the graph has a bloom,
                   // w: 1 when it has the exposure image
    vec4 exposure; // x: the EV100 the scene was stored at
    vec4 curve;    // x: b, y: c, z: contrast a, w: shoulder d
    mat3 toOutput; // the working space to the gamut written: SDR's BT.709,
                   // or in HDR the working space itself (the identity)
    vec4 luma;     // xyz: the working space's luminance row (the crosstalk)
    vec4 outLuma;  // xyz: the written gamut's luminance row (the compression);
                   // w: 1 when it is narrower than the working space
} t;
layout(set = 0, binding = 1) uniform sampler2D scene;
layout(set = 0, binding = 2) uniform sampler2D bloom;
layout(set = 0, binding = 3) uniform sampler2D exposureTex;

// Crosstalk: off at this scene value, complete at the peak (tonemap.rs).
const float CROSSTALK = 0.7;
const float CROSSTALK_START = 2.0;
const float CROSSTALK_PEAK = 16.0;

// x^a / ((x^a)^d * b + c). The shipped look has a = d = 1, so both powers
// fold away (uniform branch: no divergence) and the curve is one fma and a
// divide; the general form stays for tuning the toe and shoulder.
float curve1(float x) {
    float z = t.curve.z == 1.0 ? max(x, 0.0) : pow(max(x, 0.0), t.curve.z);
    float zd = t.curve.w == 1.0 ? z : pow(z, t.curve.w);
    return z / (zd * t.curve.x + t.curve.y);
}

vec3 lottes(vec3 c) {
    float mx = max(c.r, max(c.g, c.b));
    // The max channel decides the curve, the rest keep their ratio: no hue
    // shift from per-channel saturation.
    vec3 mapped = mx > 1e-6 ? c * (curve1(mx) / mx) : vec3(0.0);
    // Above paper white the core mixes toward a grey of the tonemapped
    // luminance, so it reads white-hot instead of a flat saturated orange.
    // Ordinary pixels take the branch: the second curve and the dot only
    // run where the crosstalk is non-zero.
    if (mx > CROSSTALK_START) {
        float y = dot(c, t.luma.xyz);
        float w = CROSSTALK * smoothstep(CROSSTALK_START, CROSSTALK_PEAK, mx);
        mapped = mix(mapped, vec3(curve1(y)), w);
    }
    return mapped;
}

void main() {
    vec3 c = max(texelFetch(scene, ivec2(gl_FragCoord.xy), 0).rgb, vec3(0.0));
    float scale = 1.0;
    if (t.params.w > 0.5) {
        // EV100 is logarithmic: the display's exposure over the scene's is
        // 2^(scene EV - eased EV). Clamped so a garbage texel cannot blow
        // the image up or out.
        float ev = texelFetch(exposureTex, ivec2(0), 0).x;
        scale = clamp(exp2(t.exposure.x - ev), 1.0 / 256.0, 256.0);
    }
    c *= scale;
    // The bloom joins the scene's light before the curve (RA3's post.frag),
    // so the curve rolls it off instead of it piling up into white.
    if (t.params.z > 0.5) {
        c += texture(bloom, uv).rgb * t.params.x;
    }
    if (t.params.y > 0.5) {
        c = lottes(c);
    }
    // SDR's output is BT.709; a BT.2020 working space is mapped into it
    // (and out-of-gamut colour compressed) here, since SDR has no
    // post.encode. In HDR the display stays in the working space (the
    // identity) and post.encode maps it. At the default working space the
    // matrix is the identity and nothing is compressed, so the frame is
    // bit-identical.
    c = t.toOutput * c;
    if (t.outLuma.w > 0.5) {
        c = aeGamutCompress(c, t.outLuma.xyz);
    }
    color = vec4(c, 1.0);
}
