#version 460
// One camera-facing quad per live particle, in sorted order (far first),
// shaded as particles_puff.glsl says.
#include "particles_common.glsl"
#include "particles_puff.glsl"

layout(location = 0) out vec2 outCorner;
layout(location = 1) out vec4 outColor;   // premultiplied
layout(location = 2) out vec2 outDepth;   // centre's view depth, radius

const vec2 CORNERS[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

void main() {
    Puff f = puffOf(keys[gl_InstanceIndex] & 0xffffu);
    vec2 c = CORNERS[gl_VertexIndex];
    vec2 r = vec2(c.x * cos(f.angle) - c.y * sin(f.angle), c.x * sin(f.angle) + c.y * cos(f.angle));
    vec3 world = f.centre + (u.camRight.xyz * r.x + u.camUp.xyz * r.y) * f.radius;
    gl_Position = u.viewProj * vec4(world, 1.0);
    outColor = f.color;
    outCorner = c;
    outDepth = vec2(dot(f.centre - u.camPos.xyz, u.camForward.xyz), f.radius);
}
