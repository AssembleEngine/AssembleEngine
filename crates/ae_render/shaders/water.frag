#version 460
// The water surface (M7), after RA3's Ocean.fx as rebuilt in
// ~/RedAlert-graphics (shaders/ocean.frag), cut down to what reads at RTS
// distances and drawn over the terrain instead of from copies of it:
//
// * RA3 composes refraction + reflection + environment × fresnel. Here the
//   refraction is what the target already holds: the terrain under the
//   water is tinted by its depth and lit by caustics (terrain.frag, the
//   part of RA3's underwater pass that applies), and the surface blends
//   over it, premultiplied: the sky it reflects weighted by fresnel, plus
//   the sun's glints (added light). No planar reflection, no refraction
//   copy, no wave particles: the surface has no geometry of its own to
//   reflect at these distances, and units are a few pixels across.
// * The waves are analytic and shade only (the plane is not displaced):
//   eight trains of sines from 48 m to 0.7 m, each fading out as it nears
//   the pixel size, its slope then counted as roughness that widens the
//   sun's highlight (RA3's `rough`), so the glitter neither aliases nor
//   vanishes with distance.
// * Shore foam (RA3's shoreWaves, by the water's depth instead of a baked
//   travel time): crests running in to the waterline and a foam line
//   along it, broken up by noise.
// * The waterline fades over the first half metre of depth, where the
//   terrain mesh and the plane meet (RA3's soft waterline).
#define GRAPHICS
#include "terrain_common.glsl"
#include "photometry.glsl"
#include "terrain_shading.glsl"
#define LIGHT_SET 1
#include "lights.glsl"

layout(location = 0) in vec3 inWorld;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outGlow;

const int TRAINS = 8;
// Wavelength (m), direction (radians), steepness (amplitude × wavenumber:
// the slope each train adds).
const float LENGTH[TRAINS] = float[](48.0, 27.0, 15.0, 8.3, 4.6, 2.5, 1.4, 0.75);
const float ANGLE[TRAINS] = float[](0.55, 1.15, 0.2, 0.9, -0.1, 0.65, 1.4, 0.35);
const float STEEP[TRAINS] = float[](0.05, 0.06, 0.07, 0.08, 0.08, 0.07, 0.06, 0.05);

// The surface's speculars (sun and clustered lights) use the shared GGX
// BRDF at the water's F0 (0.06, raised from water's 0.02 so the ripples
// read from above). This gain keeps the highlight's brightness where the
// step's reference captures put it: 0.062 was tuned without the receiver
// cosine, and the reference sun (42 degrees up) has N.L = sin(42) = 0.669
// at the mirror, so applying the cosine needs 0.062 / 0.669. The gain is an
// art calibration; the lights use the same response, so a source's glint
// does not depend on whether it is the sun.
const float WATER_SPECULAR_GAIN = 0.0926;

// Slope (d height / dx, d height / dz) of the waves at p, with trains finer
// than about three pixels faded out (`pix`: world units per pixel); `lost`
// gets the squared slope of what was faded.
vec2 waveSlope(vec2 p, float t, float pix, out float lost) {
    vec2 slope = vec2(0.0);
    lost = 0.0;
    for (int i = 0; i < TRAINS; ++i) {
        float k = 6.2831853 / LENGTH[i];
        float omega = sqrt(9.81 * k); // deep water
        vec2 d = vec2(cos(ANGLE[i]), sin(ANGLE[i]));
        float resolved = 1.0 - smoothstep(0.15, 0.4, pix / LENGTH[i]);
        float s = STEEP[i] * cos(dot(d, p) * k - omega * t);
        slope += d * s * resolved;
        lost += STEEP[i] * STEEP[i] * (1.0 - resolved);
    }
    return slope;
}

vec3 skyRadiance(vec3 dir, bool wide) {
    // A uniform hemisphere at the sky's radiance; the zenith a little
    // darker than the horizon (the fog colour is the horizon).
    vec3 zenith = aeUnitTint(AE_SKY_TINT, wide) * (u.photometry.w / AE_PI) * 0.6;
    return mix(u.fog.rgb, zenith, sqrt(clamp(dir.y, 0.0, 1.0)));
}

void main() {
    // Uniform per draw: no surface seen from below.
    if (u.camPos.y < u.water.x) {
        discard;
    }
    bool wide = u.photometry.y > 0.5;
    vec3 P = inWorld;
    float t = u.water.w;
    vec3 toEye = u.camPos.xyz - P;
    float dist = length(toEye);
    vec3 E = toEye / dist;

    // The ground under the surface (bilinear corners, as the terrain mesh
    // converges to) and the water's depth there.
    float depth = u.water.x - sampleHeight(P.xz / u.map.w);
    float waterline = smoothstep(0.0, 0.5, depth);

    float pix = max(length(dFdx(P.xz)), length(dFdy(P.xz)));
    float lost;
    vec2 slope = waveSlope(P.xz, t, pix, lost);
    // Calmer in the shallows (the waves feel the bottom).
    slope *= mix(0.35, 1.0, smoothstep(0.0, 4.0, depth));
    vec3 N = normalize(vec3(-slope.x, 1.0, -slope.y));
    float rough = clamp(sqrt(lost) * 4.0, 0.0, 1.0);
    // A narrow lobe on smooth water, wider where the unresolved waves make
    // the surface rough (the old Phong exponents 900..60 at the same lobe
    // widths).
    float roughness = mix(0.22, 0.43, rough);

    // Fresnel: Schlick with F0 0.06, raised from water's 0.02 so the
    // ripples still read from straight above (0 A.D. and RA3 floor theirs).
    float cosE = max(dot(E, N), 0.0);
    float f = clamp(1.0 - cosE, 0.0, 1.0);
    float f2 = f * f;
    float fresnel = 0.06 + 0.94 * (f2 * f2 * f);

    vec3 R = reflect(-E, N);
    vec3 L = normalize(u.sun.xyz);
    vec3 reflection = skyRadiance(R, wide);
    // The glints: the sun's and the clustered lights', through the shared
    // GGX lobe (specular only: the water's diffuse is the foam).
    vec3 tint;
    float units = unitSunAt(P, vec3(0.0, 1.0, 0.0), tint);
    vec3 sunE = aeUnitTint(AE_SUN_TINT, wide) * u.photometry.z * min(sunAt(P.xz, P.y), units) * tint;
    SurfaceBRDF s = SurfaceBRDF(vec3(0.0), vec3(0.06), max(roughness, 0.04), 0.0);
    // `brdfSpecular` is before the receiver cosine; the lights below apply
    // theirs in `lightsSpecular`.
    vec3 spec = brdfSpecular(s, N, E, L) * (max(dot(N, L), 0.0) * sunE);
    if (lightsOn()) {
        spec += lightsSpecular(lightClusterFrag(gl_FragCoord), P, N, E, s);
    }
    vec3 glint = spec * WATER_SPECULAR_GAIN;
    // The bloom's light sources on the water: only the crisp glints (RA3's
    // `glint * (1 - soft)`). Unresolved waves widen the lobe into a sheen:
    // many glints averaged over a pixel are not points of light, and
    // blooming that sheen lit a whole sun path as one white blob from afar.
    // Foam, the waterline fade and the fog take the glints, as RA3 takes
    // them; the bloom shader clamps what is left.
    vec3 surfaceGlow = glint * (1.0 - rough);

    // Premultiplied: alpha is how much of what lies under the surface it
    // hides; the glints are added light.
    vec3 rgb = reflection * fresnel + glint;
    float alpha = fresnel;

    // Shore foam: crests running in through the last metre and a half, and
    // a line along the waterline, laid on by a drifting noise.
    float run = fract(depth * 0.45 - t * 0.09 + valueNoise(P.xz * 0.02) * 0.6);
    float crest = smoothstep(0.82, 1.0, run) * (1.0 - smoothstep(0.4, 1.6, depth));
    float edge = 1.0 - smoothstep(0.05, 0.45, depth);
    float texture_ = valueNoise(P.xz * 0.45 + vec2(t * 0.05, -t * 0.03)) * 0.6 + valueNoise(P.xz * 1.1 + vec2(-t * 0.07, t * 0.04) + 7.1) * 0.4;
    float foam = clamp((crest * 0.8 + edge * 0.75) * smoothstep(0.3, 0.7, texture_ + 0.2 * edge), 0.0, 0.9);
    vec3 skyR = aeUnitTint(AE_SKY_TINT, wide) * (u.photometry.w / AE_PI);
    vec3 foamTint = aeToWorking(vec3(0.86, 0.90, 0.90), wide);
    vec3 foamColor = foamTint * (sunE * max(L.y, 0.0) * 0.45 / AE_PI + skyR * 0.7);
    // The clustered lights on the foam (their glints on the water come
    // with the speculars).
    if (lightsOn()) {
        foamColor += foamTint * lightsDiffuse(lightClusterFrag(gl_FragCoord), P, N) / AE_PI;
    }
    rgb = rgb * (1.0 - foam) + foamColor * foam;
    alpha = alpha * (1.0 - foam) + foam;

    // The soft waterline: nothing where the ground reaches the surface.
    rgb *= waterline;
    alpha *= waterline;
    surfaceGlow *= (1.0 - foam) * waterline;

    if ((u.cfg.z & FLAG_VISION) != 0u) {
        rgb = applyVision(rgb, P.xz);
        surfaceGlow = applyVision(surfaceGlow, P.xz);
    }
    // Distance fog over the surface's share (what lies under it is fogged
    // already): the blend then gives mix(surface over ground, fog, f).
    float fogged = fogAmount(dist);
    rgb = mix(rgb, u.fog.rgb * alpha, fogged);
    surfaceGlow *= 1.0 - fogged;
    // Stored exposed, like the rest of the scene.
    outColor = vec4(rgb * u.photometry.x, alpha);
    // The water replaces the glow under it: its glints are its own light.
    outGlow = vec4(surfaceGlow * u.photometry.x, 1.0);
}
