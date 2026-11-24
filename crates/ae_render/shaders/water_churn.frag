#version 460
// The churn stamp (the prototype's particle_churn.frag): r churn, g churn x
// age (additive, so the surface reads the churned water's mean age as
// g / r), b whitewater.
#include "water_splat.glsl"
void main() {
    float r = length(corner);
    float w = exp(-r * r * 2.5) * (1.0 - smoothstep(0.8, 1.0, r));
    float c = state.y * w;
    color = vec4(c, c * state.w, state.z * w, 0.0);
}
