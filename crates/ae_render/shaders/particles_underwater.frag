#version 460
#include "particles_common.glsl"
#include "particles_frag.glsl"
#include "particles_water.glsl"

layout(location = 3) in vec3 inWorld;
layout(location = 0) out vec4 outColor;

void main() {
    float front;
    float k = coverage(front);
    float surface = waterSurfaceAt(inWorld.xz);
    float depth = surface - inWorld.y;
    if (k <= 0.0 || depth <= 0.0 || !particleOverWater(inWorld.xz, surface)) discard;
    // Bright submerged puffs roll off at the waterline instead of ending
    // on a bright edge. Use a wider transition for a larger flash.
    if (emissive()) k *= smoothstep(0.0, clamp(inDepth.y * 0.3, 0.1, 3.0), depth);
    vec4 c = inColor * k;
    // The refraction attachment stores exposed HDR at either particle resolution.
    c.rgb *= u.photometry.x;
    c.rgb *= exp(-depth * vec3(0.12, 0.07, 0.04));
    outColor = c;
}
