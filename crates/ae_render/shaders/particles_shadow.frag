#version 460
// The particles' coloured shadows (particles_shadow.vert): per channel
// the optical depth of what the particle lets through, -ln(1 - a(1 - rgb))
// (Recoil's multiplicative `1 - a + rgb a`, summed as a logarithm so it
// blends additively, in any order, into a target cleared to 0), and in
// alpha the depth toward the sun of its sphere's sunward surface over this
// texel (blended with max: the particle nearest the sun). A receiver
// beyond that surface is tinted; one nearer the sun is not, even when it
// is above a particle's centre and within its disc.
#include "particles_common.glsl"

#define puffTex AE_S2D(9, 1)

layout(location = 0) in vec2 inCorner;
layout(location = 1) flat in vec4 inColor;
layout(location = 2) flat in vec2 inDepth;

layout(location = 0) out vec4 outColor;

void main() {
    float r2 = dot(inCorner, inCorner);
    float a = r2 < 1.0 ? inColor.a * texture(puffTex, inCorner * 0.5 + 0.5).r : 0.0;
    // Transparent corners would raise the max depth without shading.
    if (a < 1.0 / 255.0) {
        discard;
    }
    vec3 through = 1.0 - min(a, 0.999) * (1.0 - clamp(inColor.rgb, 0.0, 1.0));
    outColor = vec4(-log(through), inDepth.x + sqrt(max(1.0 - r2, 0.0)) * inDepth.y);
}
