#version 460
// One instance per visible unit (index into the instance stream from the
// cull pass); 12 vertices: a dart (tetrahedron) pointing along the
// heading, built from gl_VertexIndex, flat-shaded per face.
#include "units_common.glsl"

layout(location = 0) flat out vec3 outNormal;
layout(location = 1) flat out vec3 outColor;
layout(location = 2) out vec3 outWorld;

// Local frame: x forward, y up, z to the side; in body radii.
const vec3 CORNERS[4] = vec3[](
    vec3(1.3, 0.25, 0.0),  // nose
    vec3(-0.9, 0.0, -0.8), // left tail
    vec3(-0.9, 0.0, 0.8),  // right tail
    vec3(-0.5, 0.9, 0.0)   // top
);
const uint FACES[12] = uint[](0u, 1u, 3u, 0u, 3u, 2u, 1u, 2u, 3u, 0u, 2u, 1u);

vec3 place(Unit unit, vec3 local) {
    vec3 forward = vec3(unit.heading.x, 0.0, unit.heading.y);
    vec3 side = vec3(-unit.heading.y, 0.0, unit.heading.x);
    return unit.pos + (forward * local.x + vec3(0.0, local.y, 0.0) + side * local.z) * unit.size;
}

void main() {
    Unit unit = loadUnit(visible[gl_InstanceIndex]);
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
    gl_Position = u.viewProj * vec4(world, 1.0);
}
