#version 460
// Reflection test fixture: the bindless program form (contract version 3).
// Two sampled images through the program's handle table, a std140 block of
// 32 bytes and a 16-byte push block.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
#define AE_DECLARE_PUSH 0
#include "bindless.glsl"
#include "program.glsl"
layout(std140, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer Params {
    vec4 tint;
    vec4 offset;
} ae_uniforms[];
#define scene AE_PROGRAM_IMAGE(0)
#define overlay AE_PROGRAM_IMAGE(1)
void main() {
    color = texture(scene, uv + AE_UNIFORMS.offset.xy) * AE_UNIFORMS.tint + texture(overlay, uv);
}
