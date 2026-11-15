// Shared by the particle shaders (ae_render::particles).

#define MAX_EFFECTS 16

layout(std140, set = 0, binding = 0) uniform Frame {
    mat4 viewProj;
    vec4 camPos;      // xyz eye; w near plane (infinite reverse-Z: depth = near / z)
    vec4 camRight;
    vec4 camUp;
    vec4 camForward;
    vec4 frustum[6];  // normalised planes, inside >= 0
    vec4 step;        // dt, time, 1/width, 1/height
    uvec4 pool;       // capacity, spawn records, spawn total, ring head
    uvec4 flags;      // x & 1: hills' shadows; y: find the visible depth range (moments)
    vec4 map;         // cell size, 1/(corners x), 1/(corners y)
    vec4 height;      // scale, offset, water level, has water
    vec4 sun;         // xyz direction
    vec4 fog;         // rgb, density (0 = off)
    // Per effect, five vec4s (EffectDef::words): colour at birth, colour at
    // death, (size birth, size death, life min, life max), (speed min,
    // speed max, up, spread), (gravity, drag, emissive, fade in).
    vec4 effects[MAX_EFFECTS * 5];
} u;

// One particle: position and age; velocity and life (dead when age >=
// life); effect, seed, scale bits.
struct Particle {
    vec4 posAge;
    vec4 velLife;
    uvec4 info;
};

// The compute shaders define PARTICLES_WRITE; the draw reads only.
#ifdef PARTICLES_WRITE
#define PARTICLES_ACCESS
#else
#define PARTICLES_ACCESS readonly
#endif
layout(std430, set = 0, binding = 1) PARTICLES_ACCESS buffer State { Particle particles[]; };
layout(std430, set = 0, binding = 2) PARTICLES_ACCESS buffer Keys { uint keys[]; };
// The draw's indirect arguments, then this frame's range of the visible
// particles' view depths (float bits: positive floats order as uints).
layout(std430, set = 0, binding = 3) PARTICLES_ACCESS buffer Indirect { uint vertexCount; uint instanceCount; uint firstVertex; uint firstInstance; uint depthMin; uint depthMax; } cmd;

struct Spawn {
    vec4 posScale;
    uvec4 info; // effect (bit 16: on the ground), count, first index in this frame's spawns, seed
};
layout(std430, set = 0, binding = 4) readonly buffer Spawns { Spawn spawns[]; };
#ifdef PARTICLES_WRITE
layout(std430, set = 0, binding = 5) buffer KeysTmp { uint keysTmp[]; };
layout(std430, set = 0, binding = 6) buffer Histograms { uint histograms[]; };
#endif

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float rand01(inout uint state) {
    state = hash(state + 0x9e3779b9u);
    return float(state >> 8) * (1.0 / 16777216.0);
}
