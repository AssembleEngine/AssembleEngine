#version 460
// Froxels, pass 3: the light and transmittance in front of this pixel's
// scene depth, trilinear among the columns' boundaries (bilinear between
// the four nearest columns' centres, linear in slices).
#include "particles_common.glsl"
#include "particles_froxels.glsl"

#define depthTex AE_S2D(0, 0)

layout(location = 0) out vec4 outColor;

vec4 boundary(ivec2 col, ivec2 columns, float b) {
    col = clamp(col, ivec2(0), columns - 1);
    uint base = uint(col.y * columns.x + col.x), stride = uint(columns.x * columns.y);
    uint b0 = min(uint(b), FROXEL_SLICES - 1u);
    return mix(uintBitsToFloat(froxels[base + b0 * stride]), uintBitsToFloat(froxels[base + (b0 + 1u) * stride]), b - float(b0));
}

void main() {
    if (cmd.depthMax == 0u) {
        outColor = vec4(0.0);
        return;
    }
    float d = texelFetch(depthTex, ivec2(gl_FragCoord.xy), 0).r;
    float scene = d > 0.0 ? u.camPos.w / d : 1e30;
    float b = froxelSlice(scene, froxelRange());
    ivec2 columns = ivec2(froxelColumns());
    vec2 p = gl_FragCoord.xy / float(FROXEL) - 0.5;
    ivec2 c = ivec2(floor(p));
    vec2 w = p - vec2(c);
    vec4 v = mix(mix(boundary(c, columns, b), boundary(c + ivec2(1, 0), columns, b), w.x), mix(boundary(c + ivec2(0, 1), columns, b), boundary(c + ivec2(1, 1), columns, b), w.x), w.y);
    outColor = vec4(v.rgb, 1.0 - v.a);
}
