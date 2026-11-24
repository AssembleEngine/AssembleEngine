#version 460
// The raw displacement: a wave particle's bump (RA3 writes the same
// amplitude everywhere; the blur turns the scalar bump into a wave train),
// or a blast's splash stamp.
#include "water_splat.glsl"

// The prototype's splash flip-book (makeSplashFlipbook), evaluated: a
// cavity that opens with a raised rim in ~0.4 T, collapses by ~1.8 T and
// rebounds as a central jet at ~3 T, then oscillates down. Normalised to a
// unit-deep cavity of radius 1 (the stamp's corners are at radius 2).
float splash(vec2 uv, float tau) {
    float open = smoothstep(0.0, 0.4, tau), after = max(tau - 0.4, 0.0);
    float centre = tau < 0.4 ? -open : -exp(-after / 2.5) * cos(3.14159265 * after / 2.8);
    float cavity = 1.0 - 0.3 * smoothstep(0.4, 1.8, tau);
    float rimH = 0.6 * open * (1.0 - smoothstep(0.4, 1.2, tau));
    float r = 2.0 * length(uv), a = atan(uv.y, uv.x);
    float wobble = 1.0 + 0.04 * sin(5.0 * a + 1.3) + 0.03 * sin(9.0 * a + 4.1);
    float h = centre < 0.0 ? centre * exp(-r * r / (cavity * cavity)) : 1.8 * centre * exp(-r * r / 0.3);
    float rimR = (1.1 + 0.76 * after) * wobble;
    h += rimH * exp(-(r - rimR) * (r - rimR) / 0.09);
    return h * (1.0 - smoothstep(1.7, 2.0, r));
}

void main() {
    float h;
    if (stamp.x > 0.5) {
        // Fade over the last T, so the stamp does not pop out.
        h = splash(corner, stamp.y) * state.x * (1.0 - smoothstep(5.0, 6.0, stamp.y));
    } else {
        float r = length(corner);
        h = state.x * exp(-r * r * 3.2) * (1.0 - smoothstep(0.70, 1.0, r));
    }
    color = vec4(h, 0.0, 0.0, 0.0);
}
