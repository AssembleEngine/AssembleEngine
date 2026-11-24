// Surface refraction grows with depth until the surface's maximum bend.
vec2 refractionOffset(vec2 normal, float depth) {
    return normal * (0.035 * (1.0 - exp(-max(depth, 0.0) * 0.12)));
}

// Reject a mixed wet/dry filtering footprint and fade the bend across
// steep colour differences, including silhouettes of bright foregrounds.
float refractionGuard(vec4 center, vec4 bent) {
    float delta = length(bent.rgb - center.rgb) / max(length(center.rgb), 1e-4);
    return float(bent.a >= 0.999) * (1.0 - smoothstep(0.18, 0.45, delta));
}
