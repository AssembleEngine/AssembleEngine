#version 460
// post.encode (src/encode.rs): the display intermediate (linear light in
// paper-white units, in the working space) to the output. The display is
// mapped to the output gamut (the display's primaries: P3 by default for
// HDR10), compressed into it when the output gamut is narrower than the
// working space (out-of-gamut chroma loses saturation, luminance and hue
// kept), then re-expressed in the container's primaries: BT.2020 for HDR10
// and BT.709-relative for scRGB. HDR10 writes ST.2084 (PQ) at
// v * paper white nits into A2B10G10R10_UNORM; scRGB writes BT.709-relative
// linear light at v * paper white / 80 into RGBA16F, negatives kept (out of
// gamut, not black). Both clamp at the display's peak, so nothing encodes
// above what the display shows.
#include "gamut.glsl"
#define AE_DECLARE_PUSH 0
#include "bindless.glsl"
#include "program.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(std140, set = AE_BINDLESS_SET, binding = AE_UNIFORM_BLOCKS) uniform Encode {
    vec4 levels;     // x: paper white nits, y: peak nits, z: 1 scRGB / 0 HDR10,
                     // w: 1 when the output gamut is narrower than the working space
    mat3 toOutput;   // the working space to the output gamut
    mat3 toContainer; // the output gamut to BT.2020 (HDR10) or BT.709 (scRGB)
    vec4 luma;       // xyz: the output gamut's luminance row (the compression's grey)
} ae_uniforms[];
// Sampled image: display 0.
#define e AE_UNIFORMS
#define display AE_PROGRAM_IMAGE(0)

// ST.2084's OETF (SMPTE ST 2084), nits normalised to 10000.
vec3 pq(vec3 nits) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 n = max(nits, vec3(0.0)) / 10000.0;
    vec3 n1 = pow(n, vec3(m1));
    return pow((c1 + c2 * n1) / (1.0 + c3 * n1), vec3(m2));
}

void main() {
    // The display is sampled 1:1: the texel under this pixel, unfiltered.
    vec3 v = texelFetch(display, ivec2(gl_FragCoord.xy), 0).rgb;
    // The tone curve's shoulder ends at the display's peak; values at or past
    // it are the peak (the curve's asymptote is a few percent above its
    // peak anchor).
    v = min(v, vec3(e.levels.y / e.levels.x));
    vec3 o = e.toOutput * v;
    // Wide content on a narrower display: compress toward the achromatic
    // axis (not clip). In gamut it is returned unchanged.
    if (e.levels.w > 0.5) {
        // The compression keeps the output gamut's luminance.
        o = aeGamutCompress(o, e.luma.xyz);
    }
    vec3 w = e.toContainer * o;
    // The display's peak is a limit in the container's channels too: a gamut
    // mapping can move light between them, so clamp once more here.
    w = min(w, vec3(e.levels.y / e.levels.x));
    if (e.levels.z > 0.5) {
        // scRGB: 1.0 = 80 nits, BT.709-relative, negatives allowed.
        color = vec4(w * (e.levels.x / 80.0), 1.0);
    } else {
        // HDR10: BT.2020 primaries, PQ at the paper white's nits.
        color = vec4(pq(w * e.levels.x), 1.0);
    }
}
