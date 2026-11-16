// The units' shadow map (src/shadow.rs) for the classes with models: every
// model command in one multi-draw per cascade (pc.cascade), one instance
// per unit of the command's class the cull found in the cascade (the
// class's region of the visible list for that cascade), skinned as
// units_models.vert. Opaque commands
// with units_models_shadow.vert + units_shadow.frag, masked ones with
// units_models_shadow_masked.vert (MASKED) + units_models_shadow_masked.frag.
#define MODELS
#include "units_common.glsl"
#include "units_models.glsl"
#include "units_dart.glsl"

#ifdef MASKED
// For the masked caster's alpha test (units_models_shadow_masked.frag).
layout(location = 0) out vec4 outUv;
layout(location = 1) out float outAlpha;
layout(location = 2) flat out uint outMaterial;
#endif

void main() {
    DrawRecord d = draws[pc.drawBase + uint(gl_DrawID)];
    uint row = visible[d.visibleBase + (pc.cascade + 1u) * u.counts.y + uint(gl_InstanceIndex)];
    Unit unit = loadUnit(row);
    Skinned s = skinVertex(d, uint(gl_VertexIndex), poseOffset(row));
    vec3 world = place(unit, s.pos * modelClasses[d.cls].fit.x);
#ifdef MASKED
    outUv = s.uv;
    outAlpha = s.color.a;
    outMaterial = d.material;
#endif
    uint k = pc.cascade;
    vec3 l = vec3(dot(u.unitShadowAxes[0].xyz, world), dot(u.unitShadowAxes[1].xyz, world), dot(u.unitShadowAxes[2].xyz, world));
    gl_Position = vec4((l - u.unitShadowOrigin[k].xyz) * u.unitShadowScale[k].xyz, 1.0);
}
