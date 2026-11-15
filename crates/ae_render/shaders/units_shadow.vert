#version 460
// The units' shadow map (src/shadow.rs): every unit (instance = row of the
// stream, not the camera's visible list, so units off screen cast into
// the view) as the dart units.vert draws, into one cascade.
#include "units_common.glsl"
#include "units_dart.glsl"

layout(push_constant) uniform Push { uint cascade; } pc;

void main() {
    Unit unit = loadUnit(uint(gl_InstanceIndex));
    vec3 world = place(unit, CORNERS[FACES[gl_VertexIndex]]);
    uint k = pc.cascade;
    vec3 l = vec3(dot(u.unitShadowAxes[0].xyz, world), dot(u.unitShadowAxes[1].xyz, world), dot(u.unitShadowAxes[2].xyz, world));
    gl_Position = vec4((l - u.unitShadowOrigin[k].xyz) * u.unitShadowScale[k].xyz, 1.0);
}
