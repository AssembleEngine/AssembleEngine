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

#ifdef SUB
layout(location = 6) flat out float outSceneAlpha;
#endif

void main() {
    DrawRecord d = draws[ae.words.y + uint(gl_DrawID)];
    // With SUB, the cull's split list: the scene's rows, or the water
    // overlay's (ae.words.x).
    uint row = drawnRow(d.visibleBase, uint(gl_InstanceIndex));
#ifdef SUB
    outSceneAlpha = attrf(ATTR_WATER, row) > 0.0 ? 0.5 : 1.0;
#endif
    Unit unit = placedUnit(row);
    Skinned s = skinVertex(d, uint(gl_VertexIndex), poseOffset(row));
    vec3 world = place(unit, s.pos * modelClasses[d.cls].fit.x);
    outWorld = world;
    outNormal = orient(unit, s.normal);
    outTangent = vec4(orient(unit, s.tangent.xyz), s.tangent.w);
    outUv = s.uv;
    outColor = s.color;
    outMaterial = d.material;
#ifdef MIRROR
    gl_Position = u.mirrorProj * vec4(world, 1.0);
    gl_ClipDistance[0] = world.y - u.height.z;
#else
    gl_Position = u.viewProj * vec4(world, 1.0);
#endif
}
