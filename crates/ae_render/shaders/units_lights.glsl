// The models' lights (M12; ae_model::ModelLight, ae_render::models):
// per unit class, a range in the table of its model's lights, and the
// table itself. Included by units_lights.comp after units_common.glsl
// (MAX_CLASSES) and units_models.glsl (the palette, the model classes,
// the rows).

#ifndef UNITS_LIGHTS_GLSL
#define UNITS_LIGHTS_GLSL

// Table slots 18..20 (units_common.glsl).

// Three vec4 each: the colour and range, the spot's cosines / source radius
// / joint and kind, and the intensity in candela.
struct ModelLight {
    vec4 color;   // rgb: linear tint; w: the authored range (0: none)
    vec3 params;  // x, y: a spot's inner and outer cosines; z: source radius
    uint meta;    // joint (low 16 bits) | kind << 16
    vec4 energy;  // x: intensity (cd)
};

// Per class: the first model light of its model and how many.
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer ClassLights { uvec2 classLights[MAX_CLASSES]; } ae_ClassLights[];
#define classLights ae_ClassLights[AE_BUF(18)].classLights
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer ModelLightTable { ModelLight modelLights[]; } ae_ModelLightTable[];
#define modelLights ae_ModelLightTable[AE_BUF(19)].modelLights
// Per row: where its model's lights start after the host's (the host
// writes it with the rows).
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) buffer UnitLightAt { uint unitLightAt[]; } ae_UnitLightAt[];
#define unitLightAt ae_UnitLightAt[AE_BUF(20)].unitLightAt

uint modelLightCount(uint cls) {
    return classLights[cls].y;
}

#endif
