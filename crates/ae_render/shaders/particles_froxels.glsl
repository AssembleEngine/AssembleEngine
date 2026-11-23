// Froxels (ParticleBlend::Froxels), after particles_common.glsl: the half-
// resolution target in columns of FROXEL x FROXEL pixels, each column cut
// into FROXEL_SLICES slices of view depth, spaced exponentially over this
// frame's range of visible particles (the simulation's, as for the moments).

#define FROXEL 4u
#define FROXEL_SLICES 32u
// The sums' fixed point (uint atomics, core): 1/4096 of a unit of light or
// optical depth, up to a million.
#define FROXEL_FIXED 4096.0

// The scatter's indirect dispatch: one workgroup per visible particle.
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) buffer FroxelDispatch { uvec4 froxelDispatch; } ae_FroxelDispatch[];
#define froxelDispatch ae_FroxelDispatch[AE_BUF(7)].froxelDispatch
// Per boundary (FROXEL_SLICES + 1, near to far) the columns, row by row
// (neighbouring columns side by side): first each slice's light and
// optical depth (fixed point, from 1), then the light and transmittance in
// front of each boundary (float bits, from 0).
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) buffer Froxels { uvec4 froxels[]; } ae_Froxels[];
#define froxels ae_Froxels[AE_BUF(8)].froxels
// Per visible particle: centre and radius; premultiplied colour; view
// depth, angle, its columns' bounds (x | y << 16, first and last; float
// bits); empty when the bounds are reversed.
struct FroxelPuff {
    vec4 centreRadius;
    vec4 color;
    vec4 depthAngle;
};
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) buffer FroxelPuffs { FroxelPuff puffs[]; } ae_FroxelPuffs[];
#define puffs ae_FroxelPuffs[AE_BUF(9)].puffs

uvec2 halfSize() {
    return uvec2((uvec2(round(1.0 / u.step.zw)) + 1u) / 2u);
}

uvec2 froxelColumns() {
    return (halfSize() + FROXEL - 1u) / FROXEL;
}

// This frame's depth range: near, log(far / near).
vec2 froxelRange() {
    float lo = uintBitsToFloat(cmd.depthMin);
    float hi = max(uintBitsToFloat(cmd.depthMax), lo * 1.0001);
    return vec2(lo, log(hi / lo));
}

// A view depth's place among the slices' boundaries (0 to FROXEL_SLICES).
float froxelSlice(float z, vec2 range) {
    return clamp(log(max(z, range.x) / range.x) / range.y, 0.0, 1.0) * float(FROXEL_SLICES);
}

float froxelBoundary(float b, vec2 range) {
    return range.x * exp(b / float(FROXEL_SLICES) * range.y);
}
