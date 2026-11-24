#ifndef AE_WATER_OCEAN
#define AE_WATER_OCEAN
// The full tier's surface detail, ported from the prototype's ocean.frag
// (RA3's Ocean.fx PS_ULTRAHIGH as rebuilt in ~/RedAlert-graphics):
//
// * the wave normal: the camera patch's waves (wakes, rings, the static
//   displacement and the swell, water_blur.comp) blended into the analytic
//   swell's gradient beyond the patch;
// * the detail octaves: two +-55 degree copies of a looping animation of
//   wind ripples (after 0 A.D.), flattened in the lee and the shallows;
// * churned water (wakes, blasts, surf) drawn with normals, not a foam
//   colour: octaves of the static map at metre scales, each rotated and
//   drifting its own way, the finer pushed around by the coarser so the
//   pattern boils in place, growing and easing off as the churn ages; and
//   fine ripples over all the water;
// * foam: the swell's crests breaking on the shore, timed by the shore
//   field's travel time; the simulated waves breaking in the shallows; an
//   impact's whitewater, breaking into lace as it thins.
//
// The water draw binds (renderer.rs record_wave_surface): image 2 the shore
// field, 3 the churn, 4 the ripple loop, 5 the static map, 9 the foam; the
// push words carry the swell (waves.rs Swell).
#include "water_swell.glsl"

#define shoreTex AE_S2D(2, 1)
#define churnTex AE_S2D(3, 1)
#define rippleTex AE_S2D_ARRAY(4, 2)
#define staticTex AE_S2D(5, 2)
#define foamTex AE_S2D(9, 2)

Swell surfaceSwell() {
    return Swell(uintBitsToFloat(ae.words.xy), uintBitsToFloat(ae.words.z), uintBitsToFloat(ae.words.w));
}

// The ripple loop's length (detail.rs RIPPLE_LOOP) and the octaves' scale.
const float RIPPLE_LOOP = 12.0;
const float OCTAVE_SCALE = 0.021;

vec3 animatedNormal(vec2 uv, float t) {
    float frames = float(textureSize(rippleTex, 0).z);
    float f = fract(t / RIPPLE_LOOP) * frames;
    float f0 = floor(f);
    vec3 a = texture(rippleTex, vec3(uv, f0)).xyz;
    vec3 b = texture(rippleTex, vec3(uv, mod(f0 + 1.0, frames))).xyz;
    return mix(a, b, f - f0) * 2.0 - 1.0;
}

// The static map without its tiling (Heitz and Neyret's stochastic tiling):
// a triangle lattice over the texture, each vertex its own random offset
// and rotation; a point blends its triangle's three samples, normalised to
// keep the slope variance. The slope in the uv frame.
vec2 untiledNormal(vec2 uv, float lod) {
    const mat2 TO_SKEWED = mat2(1.0, -0.57735027, 0.0, 1.15470054);
    vec2 skewed = TO_SKEWED * (uv * 1.7);
    ivec2 base = ivec2(floor(skewed));
    vec3 b = vec3(fract(skewed), 0.0);
    b.z = 1.0 - b.x - b.y;
    ivec2 v[3];
    vec3 w;
    if (b.z > 0.0) {
        w = vec3(b.z, b.y, b.x);
        v[0] = base; v[1] = base + ivec2(0, 1); v[2] = base + ivec2(1, 0);
    } else {
        w = vec3(-b.z, 1.0 - b.y, 1.0 - b.x);
        v[0] = base + ivec2(1, 1); v[1] = base + ivec2(1, 0); v[2] = base + ivec2(0, 1);
    }
    w = w * w;
    vec2 n = vec2(0.0);
    for (int k = 0; k < 3; ++k) {
        uint h = uint(v[k].x) * 0x8da6b343u ^ uint(v[k].y) * 0xd8163841u;
        h ^= h >> 13; h *= 0x85ebca6bu; h ^= h >> 16;
        float a = float(h & 0xffu) * (6.2831853 / 256.0);
        vec2 offset = vec2(float((h >> 8) & 0xfffu), float(h >> 20)) / 4096.0;
        mat2 r = mat2(cos(a), sin(a), -sin(a), cos(a));
        vec2 nk = textureLod(staticTex, r * uv + offset, lod).xy * 2.0 - 1.0;
        n += w[k] * (transpose(r) * nk);
    }
    return n / sqrt(dot(w, w));
}

const int CHURN_OCTAVES = 3;
const float CHURN_FREQ[3] = float[](0.2, 0.39, 0.79);  // cycles per metre, fresh
const float CHURN_AGE_STEP = 4.0;
const float CHURN_MAX_LEVEL = 2.0;
const float CHURN_MIN_LOD = 4.0;
const float CHURN_AMP[3] = float[](0.6, 0.8, 1.0);
const float CHURN_ANGLE[3] = float[](0.6, 2.3, 4.1);
const vec2 CHURN_FLOW[3] = vec2[](vec2(0.0084, -0.0049), vec2(-0.0119, 0.0084), vec2(0.0168, 0.014));
const float CHURN_FINE_FREQ = 0.1;

// The churn's slope at p, its frequencies times `scale`, the map's mip at
// least `minLod`; `lost` the amplitude in octaves finer than the pixel.
vec2 churnSlope(vec2 p, vec2 dpx, vec2 dpy, float t, float pix, float scale, float minLod, out float lost) {
    float texSize = float(textureSize(staticTex, 0).x);
    vec2 slope = vec2(0.0), warp = vec2(0.0);
    lost = 0.0;
    for (int i = 0; i < CHURN_OCTAVES; ++i) {
        float freq = CHURN_FREQ[i] * scale;
        float c = cos(CHURN_ANGLE[i]), s = sin(CHURN_ANGLE[i]);
        mat2 rot = mat2(c, s, -s, c);
        vec2 uv = rot * (p + warp) * freq + CHURN_FLOW[i] * scale * t;
        // Coarser mips are flatter: their slopes are scaled back up, but
        // only to level 5 (blurrier shapes at full slope read as blobs).
        float footprint = max(length(dpx), length(dpy)) * freq * texSize;
        float lod = max(log2(max(footprint, 1e-6)) + 0.5, minLod);
        vec2 n = untiledNormal(uv, lod);
        float resolved = 1.0 - smoothstep(0.12, 0.3, freq * pix);
        slope += n * exp2(0.45 * min(lod, 5.0)) * 0.9 * CHURN_AMP[i] * resolved;
        lost += CHURN_AMP[i] * (1.0 - resolved);
        warp += n * (0.12 / freq);
    }
    return slope;
}

// As the churn ages its pattern doubles every CHURN_AGE_STEP seconds and
// eases off: two fixed scales cross-faded (scaling the position by the age
// would stretch the pattern wherever the age changes).
vec2 agedChurnSlope(vec2 p, vec2 dpx, vec2 dpy, float t, float pix, float age, out float lost) {
    float level = clamp(age / CHURN_AGE_STEP, 0.0, CHURN_MAX_LEVEL);
    float l0 = floor(level), f = level - l0;
    float lost0, lost1;
    vec2 s0 = churnSlope(p, dpx, dpy, t, pix, exp2(-l0), CHURN_MIN_LOD + l0, lost0);
    vec2 s1 = s0;
    lost1 = lost0;
    if (f > 0.0) {
        s1 = churnSlope(p, dpx, dpy, t, pix, exp2(-l0 - 1.0), CHURN_MIN_LOD + l0 + 1.0, lost1);
    }
    lost = mix(lost0, lost1, f);
    return mix(s0, s1, f) * pow(0.6, level);
}

// Fine ripples (0.3-0.45 m) over all the water, stronger in patches.
vec2 fineSlope(vec2 p, vec2 dpx, vec2 dpy, float t, float pix) {
    float resolved = 1.0 - smoothstep(0.12, 0.3, CHURN_FINE_FREQ * pix);
    if (resolved <= 0.0) return vec2(0.0);
    float m = valueNoise(p * 0.15 + vec2(0.0, t * 0.03)) * 0.65 + valueNoise(p * 0.4 + vec2(17.0, -t * 0.05)) * 0.35;
    float patches = mix(0.5, 1.0, smoothstep(0.45, 0.75, m));
    float texSize = float(textureSize(staticTex, 0).x);
    float footprint = max(length(dpx), length(dpy)) * CHURN_FINE_FREQ * texSize;
    float lod = max(log2(max(footprint, 1e-6)) + 0.5, 3.5);
    vec2 uv = p * CHURN_FINE_FREQ + vec2(-0.021, 0.035) * t;
    return untiledNormal(uv, lod) * exp2(0.45 * min(lod, 5.0)) * 0.8 * patches * resolved;
}

// The swell's own crests breaking on the shore (after Recoil's shore waves
// and 0 A.D.'s look), keeping the swell's period. A crest's phase is the
// swell's plus its travel time to the shore (the shore field), so further
// out it runs ahead and turns to meet the beach. It foams where the water
// is shallower than ~1.3 wave heights, densest on its face, trailing a
// wash; the waterline runs up and back with the swell, foaming as it
// rises. `still` is the depth under the rest level, `water` under the
// displaced surface; the foam's strength.
//
// An RTS map's beaches are steep against the prototype's island (8 m
// cells), so the depth rule alone leaves a strip a few metres wide: the
// crests also break through the last SURF_SECONDS of their run in, some
// 30 m, wherever the water is shallower than SURF_DEPTH.
const float SURF_SECONDS = 10.0;
const float SURF_DEPTH = 4.0;
float shoreWaves(Swell sw, vec2 p, float t, float still, float water, float wind) {
    if (still > SURF_DEPTH || water <= 0.0) return 0.0;
    vec2 uv = (p / u.map.w + 0.5) * u.height.zw;
    float tau = min(textureLod(shoreTex, uv, 0.0).r, 20.0);
    const int TRAINS[2] = int[2](0, 2);
    const float WEIGHT[2] = float[2](1.0, 0.35);
    float crests = 0.0, rise = 0.0;
    for (int k = 0; k < 2; ++k) {
        int i = TRAINS[k];
        float speed = SWELL_SPEED[i] * sqrt(sw.scale);
        float swellPhase = swellPhase(sw, i, p, t);
        float phase = swellPhase + tau * speed;
        // 0 as a crest passes, rising to 1 just before the next.
        float s = fract((phase - 1.5707963) / 6.2831853);
        // Wider than the prototype's (0.9, 9): a crest must read from an
        // RTS camera.
        crests += WEIGHT[k] * max(smoothstep(0.75, 1.0, s), exp(-s * 5.0));
        float h = 0.5 + 0.5 * sin(swellPhase);
        rise += SWELL_AMP[i] * h * cos(swellPhase) * speed;
    }
    float height = (SWELL_AMP[0] + SWELL_AMP[2]) * sw.strength;
    float breakDepth = clamp(height / 0.78, 0.3, 1.8);
    float surf = smoothstep(SURF_SECONDS, SURF_SECONDS * 0.3, tau) * (1.0 - smoothstep(SURF_DEPTH * 0.6, SURF_DEPTH, still));
    float breaking = max(smoothstep(breakDepth * 1.25, breakDepth * 0.55, still), surf) * crests;
    float runUp = max(1.0 - smoothstep(0.0, 0.3, water), 0.6 * smoothstep(3.0, 0.0, tau));
    float edge = runUp * mix(0.3, 1.0, smoothstep(-0.05, 0.15, rise * sw.strength));
    vec2 q = p * 0.025 + vec2(0.010, 0.006) * t;
    float along = smoothstep(0.2, 0.7, valueNoise(q) * 0.7 + valueNoise(q * 2.3 + 7.1) * 0.3);
    float seaState = smoothstep(0.0, 0.5, sw.strength) * mix(0.8, 1.4, smoothstep(1.0, 4.0, sw.strength));
    return (breaking * mix(0.5, 1.0, along) * 1.4 + edge) * seaState * mix(0.8, 1.0, wind);
}

struct Ocean {
    vec3 N;        // world normal
    float rough;   // 0..1: unresolved and churned detail widening the highlight
    vec2 detail;   // the ripple octaves' slope (breaks up the reflection)
    vec2 churnN;   // the churn's and fine ripples' slope
    float churn;   // 0..1
    float bubbles; // 0..1: the young churn's bubbles, lightening its facets
    float foam;    // 0..0.95
};

// The surface at P (displaced), with `water` metres of water under it.
Ocean oceanSurface(vec3 P, float t, float water) {
    Ocean o;
    Swell sw = surfaceSwell();
    float seabed = water - (P.y - u.water.x);  // depth under the rest level, signed
    float still = max(seabed, 0.0);
    vec2 uvPatch = patchUV(P.xz);
    float patchMix = patchWeight(uvPatch);
    vec4 wave = waterWave(P.xz);
    vec2 swellGrad;
    float shore = swellShoreFactor(sw, seabed);
    float swellH = swellHeight(sw, P.xz, t, swellGrad) * shore;
    swellGrad *= shore;
    // The prototype's normal takes the simulated waves' slope about 2.2
    // times (RA3's nDisp: 30 per raw unit per texel) and the swell's as it
    // is: wakes and rings read from above.
    vec2 slope = swellGrad + (wave.xy - swellGrad) * (patchMix * 2.2);
    // The wind ripples flatten in the lee of land and in the shallows.
    float wind = smoothstep(0.5, 6.0, still);

    float ca = cos(radians(55.0)), sa = sin(radians(55.0));
    mat2 rot = mat2(ca, -sa, sa, ca);
    vec2 nUV = P.xz * OCTAVE_SCALE;
    vec3 detail = animatedNormal(rot * nUV, t) + animatedNormal(transpose(rot) * nUV, t);
    detail.xy *= mix(0.25, 1.0, wind);

    // Churn: the particles' (with their mean age) and a little at the edge.
    vec3 churnSample = texture(churnTex, uvPatch).rgb * patchMix;
    float wakeChurn = smoothstep(0.01, 0.3, churnSample.r);
    float surf = 0.12 * (1.0 - smoothstep(0.05, 0.6, water));
    float churn = max(wakeChurn, surf);
    const float SURF_AGE = 3.0;
    float churnAge = (wakeChurn * churnSample.g / max(churnSample.r, 1e-5) + surf * SURF_AGE) / max(wakeChurn + surf, 1e-5);
    vec2 dpx = dFdx(P.xz), dpy = dFdy(P.xz);
    float pix = max(length(abs(dpx) + abs(dpy)), 1e-4);
    float rough = 0.4 * smoothstep(0.03, 0.15, CHURN_FINE_FREQ * pix);
    float calmFine = 0.5 * mix(0.2, 1.0, wind);
    vec2 churnN = fineSlope(P.xz, dpx, dpy, t, pix) * mix(calmFine, mix(1.0, 0.5, smoothstep(0.0, 10.0, churnAge)), churn);
    if (churn > 0.001) {
        float lost;
        churnN += agedChurnSlope(P.xz, dpx, dpy, t, pix, churnAge, lost) * churn;
        // Churned water is rough at every scale: a wide highlight, never a
        // pinpoint glint.
        rough = max(rough, clamp(churn * (0.5 + 0.6 * lost), 0.0, 1.0));
    }
    // Tangent space (x, z, up), as the prototype composes it: the waves,
    // the churn, then + static * 0.2.
    vec3 n = normalize(vec3(-slope + churnN * 1.2, 1.0));
    n = normalize(n + detail * 0.2);
    o.N = vec3(n.x, n.z, n.y);
    o.rough = rough;
    o.detail = detail.xy;
    o.churnN = churnN;
    o.churn = churn;
    // Fresh churn is full of bubbles; they rise out within seconds, and
    // older churn shows only its broken surface. Lightening all of it laid
    // a flat pale sheet over a convoy's overlapping wakes.
    o.bubbles = churn * exp(-churnAge / 2.5);

    // Foam: the shore, the simulated waves breaking, whitewater.
    float ca2 = cos(1.1), sa2 = sin(1.1);
    vec2 nudge = detail.xy * 0.03;
    float foamTexel = texture(foamTex, P.xz / 9.0 + vec2(0.011, 0.004) * t + nudge).r
        + texture(foamTex, mat2(ca2, sa2, -sa2, ca2) * P.xz / 14.3 + vec2(-0.006, 0.009) * t - nudge).r;
    float simCrest = max(wave.z - swellH, 0.0) * patchMix;  // metres
    float simBreak = smoothstep(0.55, 0.95, simCrest / max(still, 0.05)) * smoothstep(0.15, 0.5, simCrest)
        * (1.0 - smoothstep(2.5, 4.0, still));
    float simEdge = (1.0 - smoothstep(0.0, 0.3, water)) * smoothstep(0.1, 0.4, simCrest);
    float shoreFoam = clamp(max(shoreWaves(sw, P.xz, t, still, water, wind), max(simBreak, simEdge))
        * (0.3 + foamTexel * 0.6), 0.0, 0.92);
    float white = smoothstep(0.0, 1.0, churnSample.b);
    // Lacier than the prototype's dense sheet (0.45 + 0.5 foam): against the
    // darker water of an RTS sea a solid disc reads as a blob.
    float impactFoam = white * mix(foamTexel * 0.4, 0.25 + foamTexel * 0.45, white);
    o.foam = clamp(max(shoreFoam, impactFoam), 0.0, 0.95);
    return o;
}
#endif
