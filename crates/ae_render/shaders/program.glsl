// A fullscreen program's interface (M15 step F): the push block and the
// accessors into the global set. Include after `bindless.glsl` (which
// declares the arrays), then declare the program's std140 block as
// `ae_uniforms[]` at binding 4 (the uniform blocks; M15 G):
//
//   layout(std140, set = AE_BINDLESS_SET, binding = AE_UNIFORM_BLOCKS)
//   uniform MyBlock { vec4 tint; } ae_uniforms[];
//
// `AE_UNIFORMS` is this pass's block and `AE_PROGRAM_IMAGE(k)` its k-th
// sampled image, in the pass's sampled-intent order; both go through the
// program's handle table, which the host writes before the pass records.
// The sampler is the pass's own (slot 0 of its table, in the engine's
// passes).
#ifndef AE_PROGRAM_GLSL
#define AE_PROGRAM_GLSL
layout(push_constant) uniform AeProgram {
    uint uniforms;      // the std140 block's global index
    uint samplerIndex;  // the sampler's global index (`sampler` is reserved)
    uint handles;       // the handle table's global index
    uint pad;
} ae_program;
#define AE_TABLE ae_program.handles
#define AE_UNIFORMS ae_uniforms[ae_program.uniforms]
#define AE_PROGRAM_IMAGE(k) AE_S2D_AT(AE_BUF(k), ae_program.samplerIndex)
#endif
