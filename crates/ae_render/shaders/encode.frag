#version 460
// post.encode (src/encode.rs): the display intermediate (linear light in
// paper-white units, in the working space) to the output. HDR10: ST.2084
// (PQ) in BT.2020 primaries at v * paper white nits, written into
// A2B10G10R10_UNORM. scRGB: BT.709-relative linear light at
// v * paper white / 80 (negatives kept: out of gamut, not black), written
// into RGBA16F. Both clamp at the display's peak, so nothing encodes
// above what the display shows. The image's format encodes (the target's
// own conversion, or the float's exact bits).
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform Encode {
    vec4 levels; // x: paper white nits, y: peak nits, z: 1 scRGB / 0 HDR10, w: unused
} e;
layout(set = 0, binding = 1) uniform sampler2D display;

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
    if (e.levels.z > 0.5) {
        // scRGB: 1.0 = 80 nits, BT.709 primaries, negatives allowed.
        color = vec4(v * (e.levels.x / 80.0), 1.0);
    } else {
        // HDR10: BT.2020 primaries (a pure re-expression of the working
        // space's BT.709 at this stage), PQ at the paper white's nits.
        vec3 w = vec3(
            dot(v, vec3(0.6274039, 0.3292830, 0.0433131)),
            dot(v, vec3(0.0690973, 0.9195404, 0.0113623)),
            dot(v, vec3(0.0163914, 0.0880133, 0.8955953)));
        color = vec4(pq(w * e.levels.x), 1.0);
    }
}
