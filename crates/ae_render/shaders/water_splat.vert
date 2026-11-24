#version 460
// The wave particles (the prototype's wave_particle.glsl and
// particle_churn.vert): each instance is one spawn record, its state a
// closed-form function of the record and the time, drawn as a quad into the
// camera patch -- additively into `water.raw` (the wave bump) or, with
// `ae.words.w` set, into `water.churn` (the churned water, whose footprint
// spreads as it ages). Dead particles collapse outside the clip volume.
//
// Push words: y the particle buffer, z the water clock (float bits), w the
// churn pass.
#define WAVE_QUAL readonly
#include "water_particle.glsl"

layout(location = 0) out vec2 corner;
layout(location = 1) flat out vec4 state;
layout(location = 2) flat out vec2 stamp;

const vec2 CORNERS[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
// The patch's texels (waves.rs SIZE).
const float PATCH_TEXELS = 1024.0;
const float CHURN_WIDTH = 2.5;  // metres added to the wave bump's radius
const float CHURN_SPREAD = 0.5; // metres of radius per second

void main() {
    WaveSource p = waveSources[ae.words.y].entries[gl_InstanceIndex];
    float time = uintBitsToFloat(ae.words.z);
    bool churnPass = ae.words.w != 0u;
    vec2 c = CORNERS[gl_VertexIndex];
    float age = time - p.dirBirthLife.z;
    float life = p.dirBirthLife.w;
    bool live = age >= 0.0 && age <= life;
    age = max(age, 0.0);
    float texel = patchSpan() / PATCH_TEXELS;
    vec2 dir = p.dirBirthLife.xy;
    vec2 center = p.posVel.xy + p.posVel.zw * age;
    // Nothing a particle still carries pops out at the end of its life.
    float lifeFade = 1.0 - smoothstep(0.8 * life, life, age);
    vec2 world;
    corner = c;
    if (p.kindScale.x > 0.5) {
        // A blast's splash: the crater flip-book over 6 T, the crater filling
        // half the stamp (half size 2 R).
        float tau = age / max(p.kindScale.y, 1e-3);
        live = live && !churnPass && tau <= 6.0;
        float halfSize = max(p.ampRadiusChurnFoam.y, 2.0 * texel);
        state = vec4(p.ampRadiusChurnFoam.x, 0.0, 0.0, age);
        stamp = vec2(1.0, tau);
        world = center + c * halfSize;
    } else {
        float radius = p.ampRadiusChurnFoam.y;
        float grow = p.growArcDecaySpread.x, arc = p.growArcDecaySpread.y;
        float decay = p.growArcDecaySpread.z, spread = p.growArcDecaySpread.w;
        // RA3's exponential fade and min(amplitude, 1); a ring's particles
        // grow as sqrt(1 + grow age) and widen along the ring as it spreads,
        // their amplitude scaled down to match.
        float halfWidth = max(radius, arc * length(p.posVel.zw) * age);
        float amp = p.ampRadiusChurnFoam.x * sqrt(1.0 + grow * age) * (radius / halfWidth)
            * exp(-age / decay) * clamp(age / 0.3, 0.0, 1.0);
        amp = min(amp, 1.0) * lifeFade;
        // The churned water calms over ~9 s, or over half its life if that
        // is shorter (a fast hull's trail is a fixed length, so its records
        // live briefly: the trail then thins along its whole length instead
        // of ending as a cut); whitewater is gone in a few.
        float churn = p.ampRadiusChurnFoam.z * exp(-age / min(9.0, 0.5 * life)) * clamp(age / 0.1, 0.0, 1.0)
            * lifeFade;
        float foam = p.ampRadiusChurnFoam.w * exp(-age / 5.0) * clamp(age / 0.05, 0.0, 1.0) * lifeFade;
        // The patch's texel grows with the camera distance: a bump keeps at
        // least 1.5 texels, and its volume.
        float minRadius = 1.5 * texel;
        float r = max(radius, minRadius), w = max(halfWidth, minRadius);
        vec2 halfSize;
        if (churnPass) {
            float radius0 = r + spread * CHURN_WIDTH;
            float grown = radius0 + spread * CHURN_SPREAD * age;
            float thin = sqrt(radius0 / grown);
            churn *= thin;
            foam *= thin;
            live = live && (churn > 0.0002 || foam > 0.0002);
            halfSize = vec2(grown * w / r, grown);
        } else {
            amp *= (radius / r) * (halfWidth / w);
            live = live && amp > 0.001;
            halfSize = vec2(w, r);
        }
        state = vec4(amp, churn, foam, age);
        stamp = vec2(0.0);
        world = center + c.x * vec2(-dir.y, dir.x) * halfSize.x + c.y * dir * halfSize.y;
    }
    gl_Position = live ? vec4(patchUV(world) * 2.0 - 1.0, 0.0, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
}
