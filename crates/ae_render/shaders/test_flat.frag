#version 450
// Test shader: the colour to the single target.
layout(push_constant) uniform Rect {
    vec4 rect;
    vec4 color;
    vec4 depth;
} pc;
layout(location = 0) out vec4 out0;

void main() {
    out0 = pc.color;
}
