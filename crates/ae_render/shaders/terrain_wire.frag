#version 460
layout(location = 0) in vec3 inWorld;
layout(location = 1) flat in uint inDepth;
layout(location = 0) out vec4 outColor;

void main() {
    // slightly lighter lines on deeper (smaller) leaves keeps dense areas readable
    outColor = vec4(vec3(0.02 + 0.004 * float(inDepth) + 0.00002 * inWorld.y), 1.0);
}
