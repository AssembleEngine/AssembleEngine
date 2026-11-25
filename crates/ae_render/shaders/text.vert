#version 460

struct Quad {
    vec4 bounds;
    vec4 uv;
    vec4 color;
};

layout(std430, set = 0, binding = 0) readonly buffer Quads {
    Quad quads[];
};

layout(push_constant) uniform Transform {
    mat4 local_to_clip;
};

layout(location = 0) out vec2 uv;
layout(location = 1) flat out vec4 color;

void main() {
    const vec2 corners[6] = vec2[6](
        vec2(0, 0), vec2(1, 0), vec2(0, 1),
        vec2(0, 1), vec2(1, 0), vec2(1, 1)
    );
    Quad quad = quads[gl_InstanceIndex];
    vec2 corner = corners[gl_VertexIndex];
    uv = mix(quad.uv.xy, quad.uv.zw, corner);
    color = quad.color;
    vec2 position = mix(quad.bounds.xy, quad.bounds.zw, corner);
    gl_Position = local_to_clip * vec4(position, 0.0, 1.0);
}
