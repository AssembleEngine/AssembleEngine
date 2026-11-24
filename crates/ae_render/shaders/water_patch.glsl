#ifndef AE_WATER_PATCH
#define AE_WATER_PATCH

// The patch's heights stay within this many metres of the water level (the
// blur clamps there; the water LOD's leaf bounds use it).
const float WAVE_BOUND = 4.0;

vec2 patchOrigin() { return uintBitsToFloat(ae.words2.xy); }
float patchSpan() { return uintBitsToFloat(ae.words2.z); }
vec2 patchUV(vec2 world) { return (world - patchOrigin()) / max(patchSpan(), 0.001); }
float patchWeight(vec2 uv) {
    vec2 edge = min(uv, 1.0 - uv);
    return smoothstep(0.0, 0.09, min(edge.x, edge.y));
}

#ifdef WAVE_SAMPLE
#ifndef WAVE_SAMPLER
#define WAVE_SAMPLER 1
#endif
vec4 waterWave(vec2 world) {
    if (patchSpan() <= 0.0) return vec4(0.0);
    return textureLod(AE_S2D(WAVE_SAMPLE, WAVE_SAMPLER), patchUV(world), 0.0);
}

float waterDisplacement(vec2 world) {
    return waterWave(world).z * patchWeight(patchUV(world));
}
#endif
#endif
