// Gamut mapping (src/colorspace.rs, mirrored by `colorspace::compress`).
// A linear colour in an output gamut's RGB, moved toward its own achromatic
// (grey) value as far as its most negative channel needs and no further:
// luminance is kept, hue is kept (every channel's distance from the grey
// scales by one factor), and a colour already in gamut is returned
// unchanged. Not a clip: wide-gamut light loses chroma, not brightness.
// Values above white are brightness (an HDR emitter), not out of gamut; only
// negative channels are.
#ifndef GAMUT_GLSL
#define GAMUT_GLSL

vec3 aeGamutCompress(vec3 c, vec3 luma) {
    float lo = min(c.r, min(c.g, c.b));
    if (lo >= 0.0) {
        return c;
    }
    float y = dot(c, luma);
    if (y <= 0.0) {
        return vec3(0.0);
    }
    float t = y / (y - lo); // in (0, 1): at t the most negative channel is 0
    return mix(vec3(y), c, t);
}

#endif
