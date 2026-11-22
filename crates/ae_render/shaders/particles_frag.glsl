// Shared by the particle fragment shaders (after particles_common.glsl).
// A soft puff. The particle is treated as a sphere: where its front
// surface would pass behind the scene it fades out (soft edges; this is
// also the depth test, the pass has no depth attachment).

// The scene depth at the target's resolution (infinite reverse-Z).
layout(set = 1, binding = 0) uniform sampler2D depthTex;
// The shape (src/particles.rs puff_mips): a ball broken up by noise.
layout(set = 1, binding = 1) uniform sampler2D puffTex;

layout(location = 0) in vec2 inCorner;
layout(location = 1) in vec4 inColor;   // premultiplied (alpha: what it covers)
layout(location = 2) in vec3 inDepth;   // centre's view depth, radius, 1: emissive

// Whether its light is a bloom source (an emissive particle, whatever it
// covers).
bool emissive() {
    return inDepth.z > 0.5;
}

// How much of the particle covers this pixel (0: none), and its sphere's
// front view depth here.
float coverage(out float front) {
    float shape = texture(puffTex, inCorner * 0.5 + 0.5).r;
    float r2 = dot(inCorner, inCorner);
    float m = max(1.0 - r2, 0.0);
    float d = texelFetch(depthTex, ivec2(gl_FragCoord.xy), 0).r;
    float scene = d > 0.0 ? u.camPos.w / d : 1e30;
    front = inDepth.x - sqrt(m) * inDepth.y;
    float soft = clamp((scene - front) / max(inDepth.y, 1e-3), 0.0, 1.0);
    return r2 < 1.0 ? shape * soft : 0.0;
}

// The moments' depth: the log of the view depth over this frame's range of
// visible particles, to [-1, 1].
float warpDepth(float z) {
    float lo = uintBitsToFloat(cmd.depthMin);
    float hi = max(uintBitsToFloat(cmd.depthMax), lo * 1.0001);
    return clamp(log(max(z, lo) / lo) / log(hi / lo), 0.0, 1.0) * 2.0 - 1.0;
}

// The particle's sphere along this pixel's view depth: its chord from the
// near plane to the scene (false: empty), and the whole chord's length
// (positive even when empty: it divides).
bool chord(out float front, out float back, out float len) {
    float r2 = dot(inCorner, inCorner);
    if (r2 >= 1.0) {
        front = back = 0.0;
        len = 1.0;
        return false;
    }
    float h = sqrt(1.0 - r2) * inDepth.y;
    float d = texelFetch(depthTex, ivec2(gl_FragCoord.xy), 0).r;
    len = max(2.0 * h, 1e-3);
    front = max(inDepth.x - h, u.camPos.w);
    back = min(inDepth.x + h, d > 0.0 ? u.camPos.w / d : 1e30);
    return front < back;
}
