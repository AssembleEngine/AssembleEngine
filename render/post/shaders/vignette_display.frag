#version 460
// Darkens the corners of the display intermediate: it multiplies by
// (1 - strength * smoothstep(radius, 1, d)), so it is range-safe. Paper
// white and the peak above it scale together, nothing is clamped or
// encoded here, and `post.encode` still maps the result once, at the end.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
#define AE_DECLARE_PUSH 0
#include "bindless.glsl"
#include "program.glsl"
layout(std140, set = AE_BINDLESS_SET, binding = AE_UNIFORM_BLOCKS) uniform Vignette {
    vec4 params; // strength, radius, unused, unused
} ae_uniforms[];
#define v AE_UNIFORMS
#define display AE_PROGRAM_IMAGE(0)

void main() {
    float d = length(uv - 0.5) * 1.41421356;
    float k = 1.0 - v.params.x * smoothstep(v.params.y, 1.0, d);
    vec3 c = texture(display, uv).rgb;
    color = vec4(c * k, 1.0);
}
