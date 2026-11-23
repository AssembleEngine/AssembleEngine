#version 460
// Reflection test fixture: the two program forms mixed. The global set's
// arrays are here (binding 0 and 3, from `program.glsl`) and a per-pass
// combined image sampler at binding 1, which only the per-pass form has.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
#define AE_DECLARE_PUSH 0
#include "bindless.glsl"
#include "program.glsl"
layout(std140, set = AE_BINDLESS_SET, binding = AE_UNIFORM_BLOCKS) uniform Params {
    vec4 tint;
} ae_uniforms[];
layout(set = 0, binding = 1) uniform sampler2D legacy;
void main() {
    color = AE_UNIFORMS.tint * texture(legacy, uv);
}
