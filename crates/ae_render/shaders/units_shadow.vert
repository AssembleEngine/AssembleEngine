#version 460
// The units' shadow map (src/shadow.rs): every unit (instance = row of the
// stream, not the camera's visible list, so units off screen cast into
// the view) as the dart units.vert draws, into one cascade.
#include "units_common.glsl"
#include "units_dart.glsl"

// `ae.words.x`: the shadow caster's cascade (bindless.glsl's block).

void main() {
    // Radar blips and ghosts cast no shadow (eligible: Own and Seen only).
    if (!eligible(uint(gl_InstanceIndex))) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    Unit unit = placedUnit(uint(gl_InstanceIndex));
    if (hasModel(unit.cls)) {
        // Cast by its model (units_models_shadow.vert): outside the clip
        // volume.
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    vec3 world = place(unit, CORNERS[FACES[gl_VertexIndex]]);
    uint k = ae.words.x;
    vec3 l = vec3(dot(u.unitShadowAxes[0].xyz, world), dot(u.unitShadowAxes[1].xyz, world), dot(u.unitShadowAxes[2].xyz, world));
    gl_Position = vec4((l - u.unitShadowOrigin[k].xyz) * u.unitShadowScale[k].xyz, 1.0);
}
