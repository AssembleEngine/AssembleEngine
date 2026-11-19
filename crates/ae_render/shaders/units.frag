#version 460
// Flat-shaded unit faces, lit like the terrain (sun + sky), with its fog.
#include "units_common.glsl"
#include "photometry.glsl"
#define LIGHT_SET 2
#include "lights.glsl"

layout(location = 0) flat in vec3 inNormal;
layout(location = 1) flat in vec3 inColor;
layout(location = 2) in vec3 inWorld;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 n = normalize(inNormal);
    vec3 tint;
    float units = unitSunAt(inWorld, n, tint);
    vec3 diff = max(dot(n, normalize(u.sun.xyz)), 0.0) * min(sunAt(inWorld), units) * tint;
    vec3 sunE = aeUnitTint(AE_SUN_TINT) * (u.photometry.z * diff);
    vec3 skyE = aeUnitTint(AE_SKY_TINT) * (u.photometry.w * (0.5 + 0.5 * n.y));
    vec3 color = inColor * (sunE + skyE) / AE_PI;
    if (lightsOn()) {
        color += inColor * lightsDiffuse(lightClusterFrag(gl_FragCoord), inWorld, n) / AE_PI;
    }
    float dist = distance(inWorld, u.camPos.xyz);
    float fog = 1.0 - exp(-pow(dist * u.fog.w, 1.5));
    // Fog is physical radiance; exposure applies after the mix.
    outColor = vec4(mix(color, u.fog.rgb, fog) * u.photometry.x, 1.0);
}
