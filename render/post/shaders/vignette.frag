#version 460
// Darkens the corners of the SDR target: black with alpha rising from
// `radius` outward, blended over what the target holds.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
#define AE_DECLARE_PUSH 0
#include "bindless.glsl"
#include "program.glsl"
layout(std140, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer Vignette {
    vec4 params; // strength, radius, unused, unused
} ae_uniforms[];
#define v AE_UNIFORMS

void main() {
    float d = length(uv - 0.5) * 1.41421356;
    color = vec4(0.0, 0.0, 0.0, v.params.x * smoothstep(v.params.y, 1.0, d));
}
