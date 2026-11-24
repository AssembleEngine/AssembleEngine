#ifndef AE_WATER_VOLUME
#define AE_WATER_VOLUME

vec3 waterWorldAt(vec2 uv, float z) {
    vec4 world = u.inverseViewProj * vec4(uv * 2.0 - 1.0, z, 1.0);
    return world.xyz / world.w;
}

float waterHeight(vec2 xz) {
    float height = u.water.x;
#ifdef WATER_MESH
    height += waterDisplacement(xz);
#endif
    return height;
}

bool insideWaterMap(vec2 xz) {
    return all(greaterThanEqual(xz, vec2(0.0))) && all(lessThanEqual(xz, u.map.xy * u.map.w));
}

float waterEntry(vec3 nearPoint) {
    if (!insideWaterMap(nearPoint.xz)) return 0.0;
    // The ray starts on the near plane. A soft entry around the displaced
    // surface makes a camera crossing and a wave crossing the camera continuous.
    float surface = waterHeight(nearPoint.xz);
    float depth = surface - nearPoint.y;
    if (depth <= 0.0 || sampleHeight(nearPoint.xz / u.map.w) >= surface) return 0.0;
    return smoothstep(0.0, 0.85, depth);
}

vec3 volumeLight(vec3 color, vec2 xz) {
    vec3 light = aeToWorking(color, u.photometry.y > 0.5);
    if ((u.cfg.z & FLAG_VISION) != 0u) light = applyVision(light, xz);
    return light * u.photometry.x;
}

#endif
