#version 460
// Volume reference, pass 2: the pixel's spans as homogeneous media along
// the view depth. A particle of opacity a over a chord of length L has
// extinction -ln(1 - a) / L and scatters its colour (rgb / a) by it; an
// additive one (a 0: emissive, covering nothing) emits rgb / L per metre;
// an emissive one that covers emits by the same rule as scattering. Its
// back is capped at the scene. Every span's ends split the depth into
// intervals; in each, the spans covering it sum their extinction and
// emission, integrated exactly front to back. A lone particle gives its premultiplied colour, as
// blended; overlapping ones mix by how much of their thickness overlaps,
// so their mix changes continuously as they pass through each other. More
// fragments than the list holds: magenta.
#include "bindless.glsl"
#define depthTex AE_S2D(0, 0)
layout(std140, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer Frame {
    mat4 viewProj;
    vec4 camPos;
} ae_Frame[];
#define u ae_Frame[AE_BUF(0)]
#include "particles_reference.glsl"

layout(location = 0) out vec4 outColor;

void main() {
    uint pixel = uint(gl_FragCoord.y) * uint(textureSize(depthTex, 0).x) + uint(gl_FragCoord.x);
    uint n = refCounts[pixel];
    if (n > REFERENCE_LAYERS) {
        outColor = vec4(1.0, 0.0, 1.0, 1.0);
        return;
    }
    float d = texelFetch(depthTex, ivec2(gl_FragCoord.xy), 0).r;
    float scene = d > 0.0 ? u.camPos.w / d : 1e30;
    // Per span: front, back (capped), extinction; scattered plus emitted
    // per metre.
    vec3 span[REFERENCE_LAYERS];
    vec3 source[REFERENCE_LAYERS];
    float cuts[2u * REFERENCE_LAYERS];
    uint m = 0u;
    for (uint i = 0u; i < n; ++i) {
        uvec4 f = refFragments[pixel * REFERENCE_LAYERS + i];
        float front = uintBitsToFloat(f.x), back = uintBitsToFloat(f.y);
        vec4 c = vec4(unpackHalf2x16(f.z), unpackHalf2x16(f.w));
        float len = max(back - front, 1e-3);
        float a = min(c.a, 0.999);
        float sigma = -log(1.0 - a) / len;
        // Scattered: colour (rgb / a) times extinction; emitted: rgb / len.
        source[m] = a > 0.0 ? c.rgb / a * sigma : c.rgb / len;
        span[m] = vec3(front, min(back, scene), sigma);
        cuts[2u * m] = span[m].x;
        cuts[2u * m + 1u] = span[m].y;
        ++m;
    }
    // The cuts, near to far (insertion sort).
    uint k = 2u * m;
    for (uint i = 1u; i < k; ++i) {
        float v = cuts[i];
        uint j = i;
        while (j > 0u && cuts[j - 1u] > v) {
            cuts[j] = cuts[j - 1u];
            --j;
        }
        cuts[j] = v;
    }
    vec3 c = vec3(0.0);
    float t = 1.0;
    for (uint i = 0u; i + 1u < k; ++i) {
        float z0 = cuts[i], z1 = cuts[i + 1u];
        float len = z1 - z0;
        if (len <= 0.0) {
            continue;
        }
        float sigma = 0.0;
        vec3 s = vec3(0.0);
        for (uint j = 0u; j < m; ++j) {
            if (span[j].x <= z0 && span[j].y >= z1) {
                sigma += span[j].z;
                s += source[j];
            }
        }
        if (sigma > 0.0) {
            float tr = exp(-sigma * len);
            c += t * s / sigma * (1.0 - tr);
            t *= tr;
        } else {
            c += t * s * len;
        }
    }
    outColor = vec4(c, 1.0 - t);
}
