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
    uvec4 flags;      // x & 1: hills' shadows; y: find the visible depth range (moments, froxels); z: list the shadow casters
    vec4 map;         // cell size, 1/(corners x), 1/(corners y)
    vec4 height;      // scale, offset, water level, has water
    vec4 sun;         // xyz direction
    vec4 fog;         // rgb, density (0 = off)
    // The units' shadow cascades (src/shadow.rs CascadeUniforms), for the
    // particles' coloured shadows: the light's axes; per cascade centre
    // x, y, depth origin; per cascade 1/half-size x, y, 1/depth range.
    vec4 shadowAxes[3];
    vec4 shadowOrigin[3];
    vec4 shadowScale[3];
    // Per effect, six vec4s (EffectDef::words): colour at birth, colour at
    // death, (size birth, size death, life min, life max), (speed min,
    // speed max, up, spread), (gravity, drag, emissive, fade in), and
    // (emission birth, emission death) in cd/m² for emissive effects.
    vec4 effects[MAX_EFFECTS * 6];
    vec4 photometry;  // x: exposure, y: unused, z: sun lux, w: sky lux
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
// The visible particles' sort keys, then from u.pool.x the slots of the
// coloured shadows' casters (live, not emissive; unsorted).
layout(std430, set = 0, binding = 2) PARTICLES_ACCESS buffer Keys { uint keys[]; };
// The draw's indirect arguments, then this frame's range of the visible
// particles' view depths (float bits: positive floats order as uints),
// then (32 bytes in) the coloured shadows' indirect arguments.
layout(std430, set = 0, binding = 3) PARTICLES_ACCESS buffer Indirect {
    uint vertexCount; uint instanceCount; uint firstVertex; uint firstInstance;
    uint depthMin; uint depthMax; uint pad0; uint pad1;
    uint casterVertexCount; uint casterCount; uint casterFirstVertex; uint casterFirstInstance;
} cmd;

struct Spawn {
    vec4 posScale;
    uvec4 info; // effect (bit 16: on the ground), count, first index in this frame's spawns, seed
};
layout(std430, set = 0, binding = 4) readonly buffer Spawns { Spawn spawns[]; };
#ifdef PARTICLES_WRITE
layout(std430, set = 0, binding = 5) buffer KeysTmp { uint keysTmp[]; };
layout(std430, set = 0, binding = 6) buffer Histograms { uint histograms[]; };
#endif

// A particle's light from the clustered lights (particles_sim.comp, in
// Particle.info.w): the illuminance times the frame exposure (the value the
// scene stores, so the 10-bit x / (1 + x) range covers the displayed one),
// per channel in 10 bits.
uint packParticleLight(vec3 c) {
    uvec3 q = uvec3(round(clamp(c / (1.0 + c), 0.0, 1.0) * 1023.0));
    return q.r | (q.g << 10) | (q.b << 20);
}

vec3 unpackParticleLight(uint w) {
    vec3 x = min(vec3(uvec3(w, w >> 10, w >> 20) & 1023u) / 1023.0, vec3(1022.0 / 1023.0));
    return x / (1.0 - x);
}

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

// A particle of premultiplied colour c (alpha 0: emissive) as a medium
// over a chord of length len: its extinction, and its light per metre
// (scattered, rgb / a times the extinction; or emitted, rgb / len). Over
// the whole chord it gives c, as blended.
void medium(vec4 c, float len, out float sigma, out vec3 source) {
    float a = min(c.a, 0.999);
    float k = a > 1e-6 ? -log(1.0 - a) / a : 1.0;
    sigma = a * k / len;
    source = c.rgb * k / len;
}
