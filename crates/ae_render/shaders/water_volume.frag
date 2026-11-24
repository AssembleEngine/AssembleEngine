#version 460
#define GRAPHICS
#define UNDERWATER
#include "terrain_common.glsl"
#include "photometry.glsl"
#include "terrain_shading.glsl"

#ifdef WATER_MESH
#define WAVE_SAMPLE 15
#include "water_patch.glsl"
#endif
#include "water_volume.glsl"

layout(location = 0) in vec2 uv;

// The share of daylight left at `depth` metres: the deep water stays dim, not black.
float downwelling(float depth) {
    return mix(0.25, 1.0, exp(-depth * 0.05));
}
layout(location = 0) out vec4 outColor;

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 scene = texelFetch(AE_S2D(12, 0), pixel, 0);
    outColor = scene;
    if (ae.words.x == 0u) return;

    vec3 nearPoint = waterWorldAt(uv, 1.0);
    float entry = waterEntry(nearPoint);
    if (entry == 0.0) return;

    vec3 ray = normalize(nearPoint - u.camPos.xyz);
    float z = texelFetch(AE_S2D(13, 0), pixel, 0).r;
    float hit = z > 0.0 ? length(waterWorldAt(uv, z) - nearPoint) : 128.0;
    float exitDistance = 128.0;
    bool exits = false;
    if (ray.y > 0.001) {
        float surface = max(waterHeight(nearPoint.xz) - nearPoint.y, 0.0);
        float d = surface / ray.y;
#ifdef WATER_MESH
        // Refine the plane intersection against the same displacement as the mesh.
        for (int i = 0; i < 2; ++i) {
            d = max(waterHeight((nearPoint + ray * d).xz) - nearPoint.y, 0.0) / ray.y;
        }
#endif
        vec2 exitXZ = (nearPoint + ray * d).xz;
        exits = insideWaterMap(exitXZ) && sampleHeight(exitXZ / u.map.w) < waterHeight(exitXZ) && d < hit;
        if (exits) exitDistance = d;
    }

    float distanceInWater = min(hit, exitDistance);
    float metres = clamp(distanceInWater, 0.0, 128.0);
    int lo = min(int(metres), 127);
    vec4 transmission = mix(u.waterTransmission[lo], u.waterTransmission[lo + 1], metres - float(lo));
    vec3 scatter = volumeLight(mix(u.shallowWater.rgb, u.deepWater.rgb, transmission.a), nearPoint.xz);
    // Light comes from above: the water is brighter looking up and darker
    // looking down, and darkens with depth, so the view keeps a sense of
    // up and down where nothing else is in sight.
    float eyeDepth = max(waterHeight(nearPoint.xz) - nearPoint.y, 0.0);
    float farDepth = max(eyeDepth - ray.y * distanceInWater, 0.0);
    float facing = mix(0.45, 1.3, smoothstep(-0.9, 0.9, ray.y));
    scatter *= downwelling(0.5 * (eyeDepth + farDepth)) * facing;
    vec4 wet = texelFetch(AE_S2D(14, 0), pixel, 0);

    // Keep the rendered underside on rays that reach the interface. For a
    // submerged hit, inherit the caustics and bubbles from the wet copy.
    vec3 incoming = scene.rgb;
    if (!exits) {
        if (z <= 0.0) incoming = scatter;
        // Geometry under water is lit by the light that reaches its depth.
        if (wet.a > 0.0) incoming = wet.rgb * downwelling(farDepth);
    }
    vec3 submerged = mix(scatter, incoming, transmission.rgb);
    // The bubble draw adds RGB to the refraction image without changing its
    // wet-geometry mask. On a clear ray those samples are just the particles.
    if (wet.a == 0.0) {
        submerged += wet.rgb * exp(-min(distanceInWater, 6.0) * 0.12);
    }
    outColor.rgb = mix(scene.rgb, submerged, entry);
}
