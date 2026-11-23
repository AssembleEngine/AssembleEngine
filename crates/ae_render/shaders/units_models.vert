#version 460
// Every model command of one pipeline in one multi-draw: the command's
// record by gl_DrawID; one instance per visible unit of its class (its
// region of the visible list), skinned by the unit's palette range and
// placed like the dart (units_dart.glsl).
#define MODELS
#include "units_common.glsl"
#include "units_models.glsl"
#include "units_dart.glsl"

layout(location = 0) out vec3 outWorld;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outTangent;
layout(location = 3) out vec4 outUv;
layout(location = 4) out vec4 outColor;
layout(location = 5) flat out uint outMaterial;

void main() {
    DrawRecord d = draws[ae.words.y + uint(gl_DrawID)];
    uint row = visible[d.visibleBase + uint(gl_InstanceIndex)];
    Unit unit = loadUnit(row);
    Skinned s = skinVertex(d, uint(gl_VertexIndex), poseOffset(row));
    vec3 world = place(unit, s.pos * modelClasses[d.cls].fit.x);
    outWorld = world;
    outNormal = orient(unit, s.normal);
    outTangent = vec4(orient(unit, s.tangent.xyz), s.tangent.w);
    outUv = s.uv;
    outColor = s.color;
    outMaterial = d.material;
    gl_Position = u.viewProj * vec4(world, 1.0);
}
