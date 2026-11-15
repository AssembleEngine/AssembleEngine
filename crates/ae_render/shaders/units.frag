#version 460
// Flat-shaded unit faces, lit like the terrain (sun + sky), with its fog.
#include "units_common.glsl"

layout(location = 0) flat in vec3 inNormal;
layout(location = 1) flat in vec3 inColor;
layout(location = 2) in vec3 inWorld;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 n = normalize(inNormal);
    float diff = max(dot(n, normalize(u.sun.xyz)), 0.0) * sunAt(inWorld);
    vec3 sky = vec3(0.45, 0.55, 0.7);
    vec3 color = inColor * (vec3(1.0, 0.95, 0.85) * 1.4 * diff + sky * (0.35 + 0.25 * n.y));
    float dist = distance(inWorld, u.camPos.xyz);
    float fog = 1.0 - exp(-pow(dist * u.fog.w, 1.5));
    outColor = vec4(mix(color, u.fog.rgb, fog), 1.0);
}
