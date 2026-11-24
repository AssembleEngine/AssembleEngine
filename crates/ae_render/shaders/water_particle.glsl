#include "bindless.glsl"
#include "water_patch.glsl"

// One wave particle's spawn record (the prototype's WaveParticle, in world
// metres and seconds of the water clock; `waves.rs` writes them). Its state
// is a closed-form function of the record and the time (water_splat.glsl).
struct WaveSource {
    vec4 posVel;             // position at birth (x, z), velocity (m/s)
    vec4 dirBirthLife;       // orientation (x, z), birth, life
    vec4 ampRadiusChurnFoam; // wave amplitude, bump radius (m), churn, whitewater
    vec4 growArcDecaySpread; // ring growth, widening along a ring, e-folding time, churn spread
    vec4 kindScale;          // x: 0 a wave particle, 1 a blast's splash stamp; y: its time scale T (s)
};
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) WAVE_QUAL buffer WaveSources {
    WaveSource entries[];
} waveSources[];
