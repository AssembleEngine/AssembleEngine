#ifndef AE_WATER_SWELL
#define AE_WATER_SWELL
// The full tier's analytic swell (the prototype's ocean_waves.glsl): three
// trains, each A ((1 + sin) / 2)^2 -- crests as sharp as a sine, flat
// troughs, smooth everywhere. The prototype's directions are turned so its
// first train runs with the map's wind, its frequencies scaled by `scale`
// (the map's swell length against the prototype's 105 m) and their speeds
// by sqrt(scale) (deep water), and the heights by `strength` (the map's
// swell amplitude over 0.4 m; 1 is the prototype's default sea).
//
// The wave blur adds it to the camera patch's height (where floating units
// read it), and the surface takes its gradient beyond the patch; the shore
// waves time their breakers by its phase.

const vec2 SWELL_DIR[3] = vec2[3](vec2(0.27472113, -0.96152395), vec2(-0.81923192, -0.57346234),
                                  vec2(-0.98058068, 0.19611614));
const float SWELL_FREQ[3] = float[3](0.060, 0.125, 0.085);
const float SWELL_AMP[3] = float[3](0.55, 0.22, 0.16);
const float SWELL_SPEED[3] = float[3](0.55, 0.90, 0.68);

struct Swell {
    vec2 wind;      // unit direction of the first train
    float scale;    // frequency scale
    float strength; // height scale
};

vec2 swellDir(Swell s, int i) {
    // The rotation taking SWELL_DIR[0] onto the wind.
    vec2 a = SWELL_DIR[0];
    float c = dot(a, s.wind), sn = a.x * s.wind.y - a.y * s.wind.x;
    vec2 d = SWELL_DIR[i];
    return vec2(c * d.x - sn * d.y, sn * d.x + c * d.y);
}

float swellPhase(Swell s, int i, vec2 p, float t) {
    return dot(swellDir(s, i), p) * SWELL_FREQ[i] * s.scale + t * SWELL_SPEED[i] * sqrt(s.scale);
}

// Height above the rest level (0 .. sum of the amplitudes) and its gradient.
float swellHeight(Swell s, vec2 p, float t, out vec2 grad) {
    float height = 0.0;
    grad = vec2(0.0);
    for (int i = 0; i < 3; ++i) {
        float phase = swellPhase(s, i, p, t);
        float h = 0.5 + 0.5 * sin(phase);
        height += SWELL_AMP[i] * h * h;
        grad += swellDir(s, i) * (SWELL_AMP[i] * h * cos(phase) * SWELL_FREQ[i] * s.scale);
    }
    grad *= s.strength;
    return height * s.strength;
}

// The swell is depth-limited: in shallow water a wave higher than ~0.8 of
// the depth has broken, so its height falls off toward the shore, leaving a
// wash of half a metre to a metre to run up the beach. A factor on the
// height over water `depth` metres deep (negative over the beach above the
// rest level).
//
// Unlike the prototype's island, an RTS map's beaches are nearly flat at
// their 8 m cells, so the prototype's run-up (up to a metre above the rest
// level) floods tens of metres of sand in tongues where the trains' crests
// meet: above the rest level the allowance runs out within 0.15 m.
float swellShoreFactor(Swell s, float depth) {
    float full = (SWELL_AMP[0] + SWELL_AMP[1] + SWELL_AMP[2]) * s.strength;
    float allowed = (0.6 * max(depth, 0.0) + 0.45 + 0.15 * s.strength) * smoothstep(-0.15, 0.25, depth);
    return clamp(allowed / max(full, 1e-3), 0.0, 1.0);
}
#endif
