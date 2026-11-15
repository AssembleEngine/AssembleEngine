#version 460
// A soft puff. The particle is treated as a sphere: where its front
// surface would pass behind the scene it fades out (soft edges; this is
// also the depth test, the pass has no depth attachment).
#include "particles_common.glsl"

layout(set = 1, binding = 0) uniform sampler2D depthTex;
// The shape (src/particles.rs puff_mips): a ball broken up by noise.
layout(set = 1, binding = 1) uniform sampler2D puffTex;

layout(location = 0) in vec2 inCorner;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inDepth;

layout(location = 0) out vec4 outColor;

void main() {
    float r2 = dot(inCorner, inCorner);
    if (r2 >= 1.0) {
        discard;
    }
    float m = 1.0 - r2;
    float shape = texture(puffTex, inCorner * 0.5 + 0.5).r;
    // The scene's view depth (infinite reverse-Z; 0 is the sky).
    float d = texelFetch(depthTex, ivec2(gl_FragCoord.xy), 0).r;
    float scene = d > 0.0 ? u.camPos.w / d : 1e30;
    float front = inDepth.x - sqrt(m) * inDepth.y;
    float soft = clamp((scene - front) / max(inDepth.y, 1e-3), 0.0, 1.0);
    float k = shape * soft;
    if (k <= 0.0) {
        discard;
    }
    outColor = inColor * k;
}
