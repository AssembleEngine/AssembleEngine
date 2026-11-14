#version 460
// Reflection test fixture: writes a storage buffer its pass does not declare.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 1) uniform sampler2D scene;
layout(std430, set = 0, binding = 2) buffer Log { uint count; } log;
void main() {
    atomicAdd(log.count, 1u);
    color = texture(scene, uv);
}
