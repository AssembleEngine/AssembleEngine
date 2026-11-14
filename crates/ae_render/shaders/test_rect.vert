#version 450
// Test shader: one axis-aligned rectangle (two triangles, 6 vertices) at a
// constant depth, from push constants. Clip-space x/y in [-1, 1].
layout(push_constant) uniform Rect {
    vec4 rect;   // x0, y0, x1, y1
    vec4 color;
    vec4 depth;  // x: depth
} pc;

void main() {
    const vec2 corners[6] = vec2[](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 0), vec2(1, 1), vec2(0, 1));
    vec2 c = corners[gl_VertexIndex];
    gl_Position = vec4(mix(pc.rect.xy, pc.rect.zw, c), pc.depth.x, 1.0);
}
