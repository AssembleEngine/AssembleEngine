#version 460
// Reflection test fixture: push constants instead of a uniform block.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(push_constant) uniform Push { vec4 tint; } push;
void main() {
    color = push.tint * vec4(uv, 0.0, 1.0);
}
