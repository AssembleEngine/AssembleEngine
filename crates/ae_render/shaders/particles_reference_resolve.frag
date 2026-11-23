#version 460
// Reference, pass 2: sort the pixel's fragments far to near and blend
// them premultiplied. More fragments than the list holds: magenta.
#include "bindless.glsl"
#define depthTex AE_S2D(0, 0)
#include "particles_reference.glsl"

layout(location = 0) out vec4 outColor;

void main() {
    uint pixel = uint(gl_FragCoord.y) * uint(textureSize(depthTex, 0).x) + uint(gl_FragCoord.x);
    uint n = refCounts[pixel];
    if (n > REFERENCE_LAYERS) {
        outColor = vec4(1.0, 0.0, 1.0, 1.0);
        return;
    }
    uvec4 list[REFERENCE_LAYERS];
    for (uint i = 0u; i < n; ++i) {
        uvec4 f = refFragments[pixel * REFERENCE_LAYERS + i];
        // Insertion sort, far (larger depth) first; ties by colour bits
        // so equal depths blend the same whatever order they arrived in.
        uint j = i;
        while (j > 0u && (uintBitsToFloat(list[j - 1u].x) < uintBitsToFloat(f.x) || (list[j - 1u].x == f.x && list[j - 1u].y < f.y))) {
            list[j] = list[j - 1u];
            --j;
        }
        list[j] = f;
    }
    vec4 c = vec4(0.0);
    for (uint i = 0u; i < n; ++i) {
        vec4 f = vec4(unpackHalf2x16(list[i].y), unpackHalf2x16(list[i].z));
        c = f + c * (1.0 - f.a);
    }
    outColor = c;
}
