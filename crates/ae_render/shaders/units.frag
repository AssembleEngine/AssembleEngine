#version 460
// Flat-shaded unit faces, lit like the terrain (sun + sky), with its fog.
#include "units_common.glsl"
#include "photometry.glsl"
#define AE_LIGHT_BASE 4
#include "lights.glsl"

layout(location = 0) flat in vec3 inNormal;
layout(location = 1) flat in vec3 inColor;
layout(location = 2) in vec3 inWorld;

#ifdef SUB
layout(location = 3) flat in float inSceneAlpha;
#endif
layout(location = 0) out vec4 outColor;
#ifndef MIRROR
layout(location = 1) out vec4 outGlow;
#endif

void main() {
    vec3 n = normalize(inNormal);
    vec3 v = normalize(UNIT_VIEW_POS - inWorld);
    bool wide = u.photometry.y > 0.5;
    vec3 tint;
    float units = unitSunAt(inWorld, n, tint);
    // Illuminance (lux) before the receiver cosine; the particles' tint
    // multiplies the sunlight they colour.
    vec3 sunE = aeUnitTint(AE_SUN_TINT, wide) * (u.photometry.z * min(sunAt(inWorld), units)) * tint;
    vec3 skyR = aeUnitTint(AE_SKY_TINT, wide) * (u.photometry.w / AE_PI);
    // The darts are flat-shaded blocks: a rough dielectric with a slight
    // sheen, so they sit with the terrain rather than looking like metal.
    SurfaceBRDF s = SurfaceBRDF(inColor, vec3(0.04), 0.7, 0.0);
    vec3 color = brdfLight(s, n, v, normalize(u.sun.xyz), sunE);
    color += brdfSky(s, n, v) * skyR;
#ifndef MIRROR
    if (lightsOn()) {
        color += lightsSurface(lightClusterFrag(gl_FragCoord), inWorld, n, v, s);
    }
#endif
    float dist = distance(inWorld, UNIT_VIEW_POS);
    float fog = 1.0 - exp(-pow(dist * u.fog.w, 1.5));
    // Fog is physical radiance; exposure applies after the mix.
    outColor = vec4(mix(color, u.fog.rgb, fog) * u.photometry.x, 1.0);
#ifdef SUB
    outColor.a = inSceneAlpha;
#endif
    // A dart is no light source.
#ifndef MIRROR
    outGlow = vec4(0.0);
#endif
}
