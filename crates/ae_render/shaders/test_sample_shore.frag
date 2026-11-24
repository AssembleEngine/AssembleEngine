#version 460
// Test-only (M16.5): read the shore field's travel time at one fixed texel
// and write it scaled into the target's red. The pass takes one sampled
// intent (the shore field), so binding 1 is that image; a pruned producer
// reads zero while a live bake reads the field's tens of seconds.
layout(set = 0, binding = 1) uniform sampler2D shore;
layout(location = 0) out vec4 color;
void main() {
    float seconds = texelFetch(shore, ivec2(119, 112), 0).r;
    color = vec4(clamp(seconds / 100.0, 0.0, 1.0), 0.0, 0.0, 1.0);
}
