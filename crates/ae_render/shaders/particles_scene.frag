#version 460
// Sorted particles drawn straight into the scene (downscale 1): the scene
// colour (premultiplied "over") and the glow target. Only emissive
// particles give the bloom light (a covering fire also masks what is behind); smoke writes black and keeps
// its coverage in the glow's alpha, so a puff hides the glints behind it.
#include "particles_common.glsl"
#include "particles_frag.glsl"

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outGlow;

void main() {
    float front;
    float k = coverage(front);
    if (k <= 0.0) {
        discard;
    }
    vec4 c = inColor * k;
    // Full resolution draws straight into the scene (downscale 1): the
    // half-resolution path is exposed by the composite.
    if ((u.flags.w & 1u) != 0u) {
        c.rgb *= u.photometry.x;
    }
    outColor = c;
    outGlow = vec4(emissive() ? c.rgb : vec3(0.0), c.a);
}
