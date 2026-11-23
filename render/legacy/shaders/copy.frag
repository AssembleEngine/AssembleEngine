#version 460
// The contract-2 per-pass form: the uniform block at binding 0, then one
// combined image sampler per sampled intent (here one) at binding 1.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform Copy {
    vec4 gain;
} c;
layout(set = 0, binding = 1) uniform sampler2D src;
void main() {
    color = vec4(texture(src, uv).rgb * c.gain.rgb, 1.0);
}
