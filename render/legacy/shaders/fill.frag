#version 460
// The contract-2 per-pass form: set 0 binding 0 is the uniform block.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform Fill {
    vec4 tint;
} f;
void main() {
    color = f.tint;
}
