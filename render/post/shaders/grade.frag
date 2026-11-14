#version 460
// Colour grade of the scene image: contrast and saturation around mid
// grey, times a tint. Writes alpha 1 (replaces the target).
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform Grade {
    vec4 tint;    // rgb multiplier
    vec4 params;  // contrast, saturation, unused, unused
} g;
layout(set = 0, binding = 1) uniform sampler2D scene;

void main() {
    vec3 c = texture(scene, uv).rgb;
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(luma), c, g.params.y);
    c = (c - 0.18) * g.params.x + 0.18;
    color = vec4(max(c, 0.0) * g.tint.rgb, 1.0);
}
