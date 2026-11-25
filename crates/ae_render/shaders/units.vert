#version 460
// One instance per visible unit (index into the instance stream from the
// cull pass); 12 vertices: a dart (units_dart.glsl), flat-shaded per face.
#include "units_common.glsl"
#include "units_dart.glsl"

layout(location = 0) flat out vec3 outNormal;
layout(location = 1) flat out vec3 outColor;
layout(location = 2) out vec3 outWorld;

#ifdef SUB
layout(location = 3) flat out float outSceneAlpha;
#endif

void main() {
    // With SUB, the cull's split list: the scene's rows, or the water
    // overlay's (ae.words.x).
    uint row = drawnRow(0u, uint(gl_InstanceIndex));
#ifdef SUB
    outSceneAlpha = attrf(ATTR_WATER, row) > 0.0 ? 0.5 : 1.0;
#endif
    Unit unit = placedUnit(row);
    uint face = uint(gl_VertexIndex) / 3u;
    vec3 a = place(unit, CORNERS[FACES[face * 3u]]);
    vec3 b = place(unit, CORNERS[FACES[face * 3u + 1u]]);
    vec3 c = place(unit, CORNERS[FACES[face * 3u + 2u]]);
    vec3 world = place(unit, CORNERS[FACES[gl_VertexIndex]]);
    vec3 n = normalize(cross(b - a, c - a));
    // Outward: away from the dart's centre.
    vec3 centre = place(unit, vec3(-0.25, 0.3, 0.0));
    outNormal = dot(n, (a + b + c) / 3.0 - centre) < 0.0 ? -n : n;
    outColor = u.classes[unit.cls].rgb;
    outWorld = world;
#ifdef MIRROR
    gl_Position = u.mirrorProj * vec4(world, 1.0);
    gl_ClipDistance[0] = world.y - u.height.z;
#else
    gl_Position = u.viewProj * vec4(world, 1.0);
#endif
}
