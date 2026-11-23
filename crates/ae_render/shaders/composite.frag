#version 450
// Demo composite: left half from one chain, right half from the other.
// Push words x and y are the images' global indices; sampler slot 0 is the
// demo's nearest sampler (M15 step E).
#include "bindless.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

#define left_image AE_S2D_AT(ae.words.x, AE_SAMPLER(0))
#define right_image AE_S2D_AT(ae.words.y, AE_SAMPLER(0))

void main() {
    out_color = uv.x < 0.5 ? texture(left_image, uv) : texture(right_image, uv);
}
