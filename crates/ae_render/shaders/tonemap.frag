#version 460
// post.tonemap (src/tonemap.rs): the HDR scene (linear light) to the
// output. Exposure, then the tone curve: 0 clamps to 1 (the frames the
// 8-bit scene gave before the HDR target), 1 is ACES (Narkowicz's fit).
// The output's format encodes (an sRGB swapchain or readback image).
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform Tonemap {
    vec4 params; // x: exposure, y: curve
} t;
layout(set = 0, binding = 1) uniform sampler2D scene;

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec3 c = max(texelFetch(scene, ivec2(gl_FragCoord.xy), 0).rgb, vec3(0.0)) * t.params.x;
    c = t.params.y > 0.5 ? aces(c) : min(c, vec3(1.0));
    color = vec4(c, 1.0);
}
