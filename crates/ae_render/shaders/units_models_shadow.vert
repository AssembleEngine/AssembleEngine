#version 460
// The units' shadow map (src/shadow.rs) for the classes with models: every
// model command in one multi-draw per cascade (pc.cascade), one instance
// per unit of the command's class (every row, not the visible list: the
// rows grouped by class), skinned as units_models.vert.
#define MODELS
#include "units_common.glsl"
#include "units_models.glsl"
#include "units_dart.glsl"

void main() {
    DrawRecord d = draws[pc.drawBase + uint(gl_DrawID)];
    uint row = rows[MAX_CLASSES + rows[d.cls] + uint(gl_InstanceIndex)];
    Unit unit = loadUnit(row);
    Skinned s = skinVertex(d, uint(gl_VertexIndex), poseOffset(row));
    vec3 world = place(unit, s.pos * modelClasses[d.cls].fit.x);
    uint k = pc.cascade;
    vec3 l = vec3(dot(u.unitShadowAxes[0].xyz, world), dot(u.unitShadowAxes[1].xyz, world), dot(u.unitShadowAxes[2].xyz, world));
    gl_Position = vec4((l - u.unitShadowOrigin[k].xyz) * u.unitShadowScale[k].xyz, 1.0);
}
