// A unit's shape (units.vert and units_shadow.vert): a dart (tetrahedron)
// pointing along the heading, 12 vertices from gl_VertexIndex.

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
