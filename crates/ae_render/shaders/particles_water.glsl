#define WAVE_SAMPLE 14
#define WAVE_SAMPLER 2
#include "water_patch.glsl"

float waterSurfaceAt(vec2 p) {
    return u.height.z + waterDisplacement(p);
}

bool particleOverWater(vec2 p, float surface) {
    vec2 cells = 1.0 / u.map.yz - 1.0;
    vec2 cell = p / u.map.x;
    if (any(lessThan(cell, vec2(0.0))) || any(greaterThan(cell, cells))) return false;
    float ground = textureLod(AE_S2D(8, 2), (cell + 0.5) * u.map.yz, 0.0).r * u.height.x + u.height.y;
    return ground < surface;
}
