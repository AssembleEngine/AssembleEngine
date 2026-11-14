#version 450
// Demo composite: left half from one chain, right half from the other.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;
layout(set = 0, binding = 0) uniform sampler2D left_image;
layout(set = 0, binding = 1) uniform sampler2D right_image;

void main() {
    out_color = uv.x < 0.5 ? texture(left_image, uv) : texture(right_image, uv);
}
