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

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

vec3 worldAt(ivec2 pixel, ivec2 size, float depth) {
    vec2 ndc = (vec2(pixel) + 0.5) / vec2(size) * 2.0 - 1.0;
    vec4 world = u.inverseViewProj * vec4(ndc, depth, 1.0);
    return world.xyz / world.w;
}

vec4 transmissionAt(float depth) {
    float metres = clamp(depth, 0.0, 128.0);
    int lo = min(int(metres), 127);
    return mix(u.waterTransmission[lo], u.waterTransmission[lo + 1], metres - float(lo));
}

void main() {
    ivec2 size = textureSize(AE_S2D(13, 0), 0);
    ivec2 pixel = min(ivec2(uv * vec2(size)), size - 1);
    float z = texelFetch(AE_S2D(13, 0), pixel, 0).r;
    // The reverse-Z clear value is the sky. A ray blocked on land
    // likewise contributes no samples to the refraction image.
    if (z <= 0.0) {
        outColor = vec4(0.0);
        return;
    }
    vec3 world = worldAt(pixel, size, z);
    float depth = u.water.x - world.y;
#ifdef WATER_MESH
    depth += waterDisplacement(world.xz);
#endif
    if (depth <= 0.0) {
        outColor = vec4(0.0);
        return;
    }

    vec4 scene = texelFetch(AE_S2D(12, 0), pixel, 0);
    bool sub = scene.a == 0.5;
    float path = sub ? min(depth * u.subWater.x, u.subWater.z) : depth;
    vec4 t = transmissionAt(path);
    vec3 keep = sub ? max(t.rgb, vec3(u.subWater.y)) : t.rgb;
    bool wide = u.photometry.y > 0.5;
    vec3 water = aeToWorking(mix(u.shallowWater.rgb, u.deepWater.rgb, t.a), wide);
    if ((u.cfg.z & FLAG_VISION) != 0u) {
        water = applyVision(water, world.xz);
    }
    // The scene under the water is fogged already (terrain.frag, after its
    // vision); the light the water scatters toward the eye fogs the same
    // way, or from afar the sea would keep its colour through a haze that
    // greys the land.
    water = mix(water, u.fog.rgb, fogAmount(distance(world, u.camPos.xyz)));
    vec3 color = scene.rgb * keep + water * (1.0 - keep) * u.photometry.x;

    if (u.shallowWater.w > 0.0) {
        vec3 shadowTint;
        float shadow = min(sunAt(world.xz, world.y), unitSunAt(world, vec3(0, 1, 0), shadowTint));
        float time = u.water.w;
        vec2 surface = world.xz + depth * u.sun.xz / max(u.sun.y, 0.08);
        vec2 q = fract(surface * u.shallowWater.w + time * vec2(0.031, -0.023));
        vec2 q2 = fract(surface * (u.shallowWater.w * 1.37) + time * vec2(-0.019, 0.037) + 0.5);
        float pixelWorld = distance(world, u.camPos.xyz) / max(u.lod.y, 1.0);
        float lod = max(log2(pixelWorld * u.shallowWater.w * float(textureSize(AE_S2D(14, 1), 0).x)), 0.0);
        // The two drifting copies' mean, as the prototype keeps the web
        // (their product leaves only the points where threads cross).
        float web = 0.5 * (textureLod(AE_S2D(14, 1), q, lod).r + textureLod(AE_S2D(14, 1), q2, lod).r);
        // The web's threads are about a tenth of its tile apart. From RTS
        // heights they fall a few pixels apart, where the two drifting
        // copies beat into a regular weave over the whole shelf: they fade
        // out from 12 pixels a thread to 5, before they get there.
        float threadsPerPixel = pixelWorld * u.shallowWater.w * 10.0;
        float resolved = 1.0 - smoothstep(0.08, 0.2, threadsPerPixel);
        float caustic = web * smoothstep(0.0, 0.5, depth) * exp(-depth * 0.18) * shadow * resolved;
        color += scene.rgb * shadowTint * caustic * 0.5;
    }

    if (sub && u.subWater.w > 0.5) {
        float edge = 0.0;
        for (int axis = 0; axis < 2; ++axis) {
            ivec2 d = axis == 0 ? ivec2(1, 0) : ivec2(0, 1);
            for (int step = -1; step <= 1; step += 2) {
                ivec2 at = clamp(pixel + step * d, ivec2(0), size - 1);
                edge += float(texelFetch(AE_S2D(12, 0), at, 0).a != 0.5);
            }
        }
        color += water * u.photometry.x * smoothstep(0.0, 3.0, edge) * 0.45;
    }
    outColor = vec4(color, 1.0);
}
