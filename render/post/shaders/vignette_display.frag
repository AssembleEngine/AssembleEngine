#version 460
// Darkens the corners of the display intermediate: it multiplies by
// (1 - strength * smoothstep(radius, 1, d)), so it is range-safe. Paper
// white and the peak above it scale together, nothing is clamped or
// encoded here, and `post.encode` still maps the result once, at the end.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform Vignette {
    vec4 params; // strength, radius, unused, unused
} v;
layout(set = 0, binding = 1) uniform sampler2D display;

void main() {
    float d = length(uv - 0.5) * 1.41421356;
    float k = 1.0 - v.params.x * smoothstep(v.params.y, 1.0, d);
    vec3 c = texture(display, uv).rgb;
    color = vec4(c * k, 1.0);
}
