// Midpoint error for the submesh edges, supplemented by a bound for
// short waves that may fall between all the samples of a coarse leaf.
float waveEdgeError(vec2 a, vec2 b) {
    return abs(waterDisplacement((a + b) * 0.5)
        - (waterDisplacement(a) + waterDisplacement(b)) * 0.5);
}

float wavePixelError(vec2 a, vec2 b, vec2 c, vec3 lo, vec3 hi) {
    vec2 patchLo = patchOrigin();
    vec2 patchHi = patchLo + patchSpan();
    if (any(lessThan(hi.xz, patchLo)) || any(greaterThan(lo.xz, patchHi))) return 0.0;
    float edge = max(max(length(a - b), length(b - c)), length(c - a)) / float(SUBMESH_DENOM);
    float measured = max(max(waveEdgeError(a, b), waveEdgeError(b, c)), waveEdgeError(a, c))
        / float(SUBMESH_DENOM);
    float distanceToNode = max(length(u.camPos.xyz - clamp(u.camPos.xyz, lo, hi)), 1.0);
    float pixels = u.lod.y / distanceToNode;
    // Bound patch-intersecting leaves to a spacing that resolves the swell.
    // Close to the camera, tighten the spacing for the shorter wake ribbons.
    float spacing = clamp(distanceToNode * 0.065, 2.5, min(24.0, patchSpan() / 16.0));
    return max(measured * pixels, edge / spacing);
}
