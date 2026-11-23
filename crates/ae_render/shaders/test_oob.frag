#version 460
// GPU-assisted validation fixture (M15 F): samples whatever index its handle
// table names, which the test sets past the descriptors the set binds.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
#define AE_DECLARE_PUSH 0
#include "bindless.glsl"
#include "program.glsl"
#define src AE_PROGRAM_IMAGE(0)
void main() {
    color = texture(src, uv);
}
