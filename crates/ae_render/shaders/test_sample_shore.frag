#version 460
// Test-only (M16.5): read the shore field's travel time at one fixed texel
// and write it scaled into the target's red. The pass takes one sampled
// intent (the shore field), so AE_PROGRAM_IMAGE(0) is that image; a pruned
// producer reads zero while a live bake reads the field's tens of seconds.
layout(location = 0) out vec4 color;
#define AE_DECLARE_PUSH 0
#include "bindless.glsl"
#include "program.glsl"
#define shore AE_PROGRAM_IMAGE(0)
void main() {
    float seconds = texelFetch(shore, ivec2(119, 112), 0).r;
    color = vec4(clamp(seconds / 100.0, 0.0, 1.0), 0.0, 0.0, 1.0);
}
