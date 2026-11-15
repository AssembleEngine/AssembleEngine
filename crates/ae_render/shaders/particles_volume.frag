#version 460
// Volume reference, pass 1: append this fragment's span to its pixel's
// list. The particle is a sphere: the span is its chord along this pixel's
// view depth, front to back; its colour is the puff's (premultiplied) over
// the whole chord. The scene caps the span in the resolve.
#include "particles_common.glsl"
#include "particles_frag.glsl"
#include "particles_reference.glsl"

void main() {
    float r2 = dot(inCorner, inCorner);
    if (r2 >= 1.0) {
        return;
    }
    float half_ = sqrt(1.0 - r2) * inDepth.y;
    float front = inDepth.x - half_, back = inDepth.x + half_;
    float d = texelFetch(depthTex, ivec2(gl_FragCoord.xy), 0).r;
    float scene = d > 0.0 ? u.camPos.w / d : 1e30;
    vec4 c = inColor * texture(puffTex, inCorner * 0.5 + 0.5).r;
    if (front >= scene || (c.a <= 0.0 && max(c.r, max(c.g, c.b)) <= 0.0)) {
        return;
    }
    uint pixel = uint(gl_FragCoord.y) * uint(textureSize(depthTex, 0).x) + uint(gl_FragCoord.x);
    uint n = atomicAdd(refCounts[pixel], 1u);
    if (n < REFERENCE_LAYERS) {
        refFragments[pixel * REFERENCE_LAYERS + n] = uvec4(floatBitsToUint(front), floatBitsToUint(back), packHalf2x16(c.rg), packHalf2x16(c.ba));
    }
}
