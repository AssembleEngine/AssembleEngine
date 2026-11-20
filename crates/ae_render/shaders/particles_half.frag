#version 460
// Sorted half-resolution particles: premultiplied "over" into the colour
// target, and their light into a second half-resolution glow target with
// the same sorted blend. The glow is decided per particle, as the
// full-resolution path decides it: an emissive puff (alpha 0) adds its
// light; a medium writes black and masks the light behind it by its
// coverage. The composite upsamples both, so smoke in front of a fire
// masks it by its coverage and smoke behind it does not, instead of the
// accumulated alpha hiding all of a pixel's emission. The exposure is
// applied by the composite (this target holds physical radiance).
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
    outColor = c;
    // The bloom's light sources: only emissive particles (alpha 0). Smoke
    // writes black and keeps its coverage, so a puff hides the glints
    // behind it; the blend is premultiplied, so its coverage also masks
    // the light already written behind it.
    outGlow = vec4(c.a > (0.5 / 255.0) ? vec3(0.0) : c.rgb, c.a);
}
