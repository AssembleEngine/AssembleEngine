#version 460
// Reflection test fixture: a uniform block ending at byte 28 (vec4 + vec2 +
// float at std140 offsets 0, 16, 24; 32 padded), two sampled inputs, one output.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform Params { vec4 tint; vec2 offset; float strength; } p;
layout(set = 0, binding = 1) uniform sampler2D scene;
layout(set = 0, binding = 2) uniform sampler2D overlay;
void main() {
    color = texture(scene, uv + p.offset) * p.tint + texture(overlay, uv) * p.strength;
}
