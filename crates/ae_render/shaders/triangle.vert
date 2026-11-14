#version 450
// Demo scene: one RGB triangle from gl_VertexIndex (no vertex input).
layout(location = 0) out vec3 color;

void main() {
    const vec2 positions[3] = vec2[](vec2(0.0, -0.7), vec2(0.7, 0.6), vec2(-0.7, 0.6));
    const vec3 colors[3] = vec3[](vec3(1.0, 0.2, 0.2), vec3(0.2, 1.0, 0.2), vec3(0.2, 0.3, 1.0));
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    color = colors[gl_VertexIndex];
}
