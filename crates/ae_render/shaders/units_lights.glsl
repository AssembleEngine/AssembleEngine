// The models' lights (M12; ae_model::ModelLight, ae_render::models):
// per unit class, a range in the table of its model's lights, and the
// table itself. Included by units_lights.comp after units_common.glsl
// (MAX_CLASSES) and units_models.glsl (the palette, the model classes,
// the rows).

#ifndef UNITS_LIGHTS_GLSL
#define UNITS_LIGHTS_GLSL

#ifndef MODEL_SET
#define MODEL_SET 1
#endif

// Three vec4 each: the colour and range, the spot's cosines / source radius
// / joint and kind, and the intensity in candela.
struct ModelLight {
    vec4 color;   // rgb: linear tint; w: the authored range (0: none)
    vec3 params;  // x, y: a spot's inner and outer cosines; z: source radius
    uint meta;    // joint (low 16 bits) | kind << 16
    vec4 energy;  // x: intensity (cd)
};

// Per class: the first model light of its model and how many.
layout(std430, set = MODEL_SET, binding = 12) readonly buffer ClassLights { uvec2 classLights[MAX_CLASSES]; };
layout(std430, set = MODEL_SET, binding = 13) readonly buffer ModelLightTable { ModelLight modelLights[]; };
// Per row: where its model's lights start after the host's (the host
// writes it with the rows).
layout(std430, set = MODEL_SET, binding = 14) buffer UnitLightAt { uint unitLightAt[]; };

uint modelLightCount(uint cls) {
    return classLights[cls].y;
}

#endif
