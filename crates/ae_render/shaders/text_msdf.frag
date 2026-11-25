#version 460

layout(set = 0, binding = 1) uniform sampler2D atlas;
layout(location = 0) in vec2 uv;
layout(location = 1) flat in vec4 color;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 field = texture(atlas, uv / vec2(textureSize(atlas, 0))).rgb;
    float edge = max(min(field.r, field.g), min(max(field.r, field.g), field.b));
    // The range covers four atlas texels. Derivatives convert it to
    // screen pixels, including when the quad is projected in perspective.
    vec2 screen_scale = 1.0 / max(fwidth(uv), vec2(1e-8));
    float screen_range = max(2.0 * (screen_scale.x + screen_scale.y), 1.0);
    float coverage = clamp(screen_range * (edge - 0.5) + 0.5, 0.0, 1.0);
    float alpha = color.a * coverage;
    out_color = vec4(color.rgb * alpha, alpha);
}
