#version 460
// Colour grade of the display image: contrast and saturation around mid
// grey, times a tint. It runs on the display intermediate (linear light in
// units of paper white, in the working space) before `post.encode`, so its
// values are free to exceed 1: contrast pivots at 0.18 of paper white and
// nothing is clamped at paper white here (only `post.encode` maps the
// display to the output and the peak).
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
// A bindless fullscreen program (contract version 3): its block and its
// sampled image are in the engine's global set, reached through the
// program's handle table.
#define AE_DECLARE_PUSH 0
#include "bindless.glsl"
#include "program.glsl"
layout(std140, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer Grade {
    vec4 tint;    // rgb multiplier
    vec4 params;  // contrast, saturation, unused, unused
} ae_uniforms[];
#define g AE_UNIFORMS
#define scene AE_PROGRAM_IMAGE(0)

void main() {
    vec3 c = texture(scene, uv).rgb;
    // Saturation around Rec. 709's luminance: mix() is linear in c, so in
    // the default BT.709 working space the result keeps its luminance. (In
    // a BT.2020 working space this row is not the space's luminance and the
    // grade shifts it slightly; the module does not read the working space
    // yet.)
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(luma), c, g.params.y);
    // Contrast around mid grey (0.18 of paper white), then the tint.
    c = (c - 0.18) * g.params.x + 0.18;
    // No clamp. Above 1 is the display's headroom. Below 0 is either wide
    // colour in a BT.709 working space (a P3 effect's negative blue, which
    // must reach `post.encode` whole) or the contrast pushing the darkest
    // values (under 0.18 * (1 - 1 / contrast), 0.019 at 1.12) below black;
    // HDR10's PQ and the SDR target clamp those per channel to black, as
    // the old max(c, 0) did, and scRGB passes them to the compositor.
    color = vec4(c * g.tint.rgb, 1.0);
}
