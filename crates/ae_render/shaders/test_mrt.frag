#version 450
// Test shader: the colour to target 0, half of it to target 1.
layout(push_constant) uniform Rect {
    vec4 rect;
    vec4 color;
    vec4 depth;
} pc;
layout(location = 0) out vec4 out0;
layout(location = 1) out vec4 out1;

void main() {
    out0 = pc.color;
    out1 = pc.color * 0.5;
}
