#version 460
// post.tonemap (src/tonemap.rs): the HDR scene (linear light, stored at the
// camera's reference exposure) to the output by the tone curve: 0 clamps to
// 1 (the frames the 8-bit scene gave before the HDR target), 1 is ACES
// (Narkowicz's fit). With auto-exposure the stored scene is scaled by the
// metered, eased EV read from the 1x1 exposure image before the bloom joins
// it (the bloom is already display-referred: bloom.downsample applies the
// same scale). The output's format encodes (an sRGB swapchain or readback
// image).
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform Tonemap {
    vec4 params;   // x: bloom strength, y: curve, z: 1 when the graph has a bloom,
                   // w: 1 when it has the exposure image
    vec4 exposure; // x: the EV100 the scene was stored at
} t;
layout(set = 0, binding = 1) uniform sampler2D scene;
layout(set = 0, binding = 2) uniform sampler2D bloom;
layout(set = 0, binding = 3) uniform sampler2D exposureTex;

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
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
    // so ACES rolls it off instead of it piling up into white.
    if (t.params.z > 0.5) {
        c += texture(bloom, uv).rgb * t.params.x;
    }
    c = t.params.y > 0.5 ? aces(c) : min(c, vec3(1.0));
    color = vec4(c, 1.0);
}
