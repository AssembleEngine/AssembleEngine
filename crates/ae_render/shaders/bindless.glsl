// The global bindless set's shader side (M15). Include after `#version` and
// before any declaration that samples a texture.
//
// `AE_BINDLESS_SET` is 0: the per-pass sets are gone (M15 step E). A
// fullscreen program (a module's or the engine's tonemap, bloom and encode)
// includes `program.glsl` after this file instead of declaring its own set:
// its block and sampled images are reached through its handle table.
//
// The arrays are one binding each: sampled images at 0 (all their types
// alias it), samplers at 1, storage images at 2, storage buffers at 3 and
// uniform blocks at 4. Sampled images and separate samplers are combined at
// the use site, so a handle cannot carry a sampler and a sampler cannot
// carry an image. The CPU handle is typed, and only an index reaches the
// GPU.
//
// `ae.images` and `ae.samplers` are the pass's sampled and sampler slots in
// a 128-byte push block: `words` carries the pass's own push words (the
// terrain's LOD mode, the units' cascade, ...) and `words2`, then 16 image
// indices and 8 sampler indices.
//
// `AE_IMAGE(slot)` / `AE_SAMPLER(slot)` read a slot of that table and give
// the global index; the push block is dynamically uniform, so they carry no
// nonuniformEXT. The `_AT` forms take a global index already resolved (a
// material record's, or one passed to a function) and mark it non-uniform.

#ifndef AE_BINDLESS_GLSL
#define AE_BINDLESS_GLSL
#ifndef AE_BINDLESS_SET
#define AE_BINDLESS_SET 0
#endif
#extension GL_EXT_nonuniform_qualifier : require

layout(set = AE_BINDLESS_SET, binding = 0) uniform texture2D ae_images[];
layout(set = AE_BINDLESS_SET, binding = 0) uniform texture2DArray ae_images_array[];
layout(set = AE_BINDLESS_SET, binding = 0) uniform utexture2D ae_images_uint[];
layout(set = AE_BINDLESS_SET, binding = 1) uniform sampler ae_samplers[];
// A comparison sampler is the same descriptor; declaring it `samplerShadow`
// lets `sampler2DShadow(texture2D, samplerShadow)` combine it with a depth
// image from binding 0 (GL_KHR_vulkan_glsl: the shadow-ness comes from the
// sampler).
layout(set = AE_BINDLESS_SET, binding = 1) uniform samplerShadow ae_samplers_shadow[];

#ifndef AE_DECLARE_PUSH
#define AE_DECLARE_PUSH 1
#endif
#if AE_DECLARE_PUSH
// A pass that names its eight words defines AE_PUSH_WORDS (32 bytes of
// members) before the include; the table's offsets stay the same.
#ifndef AE_PUSH_WORDS
#define AE_PUSH_WORDS \
    uvec4 words;  /* the pass's own push words (see each shader) */ \
    uvec4 words2; /* four more */
#endif
layout(push_constant) uniform AePush {
    AE_PUSH_WORDS
    uvec4 images[4];   // 16 sampled-image indices
    uvec4 samplers[2]; // 8 sampler indices
} ae;
#endif

// Buffers (binding 3): a pass declares each storage block as an array
// aliasing binding 3 and reaches it through its handle table: `AE_BUF(k)`
// is the k-th global index in the pass's table, whose own index is push
// word 8 (`AE_TABLE`). A pass that names its push words names the eighth
// `aeTable` and defines AE_TABLE as `ae.aeTable`; a fullscreen program gets
// it from `program.glsl`.
//
// `std140` blocks that were `uniform` buffers keep binding 4 (M15 G): a
// storage buffer costs noticeably more per access on some drivers. A pass
// declares them as `layout(std140, set = AE_BINDLESS_SET, binding =
// AE_UNIFORM_BLOCKS) uniform Name { ... } ae_Name[];` and indexes with the
// same `AE_BUF(k)` handle.
#define AE_BUFFERS 3
#define AE_UNIFORM_BLOCKS 4
#define AE_STORAGE_IMAGES 2
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer AeTable { uint aeHandle[]; } ae_tables[];
#if AE_DECLARE_PUSH
#ifndef AE_TABLE
#define AE_TABLE ae.words2.w
#endif
#endif
#define AE_BUF(k) ae_tables[AE_TABLE].aeHandle[k]

#define AE_IMAGE(slot) (ae.images[(slot) >> 2][(slot) & 3])
#define AE_SAMPLER(slot) (ae.samplers[(slot) >> 2][(slot) & 3])
#define AE_S2D(image, sampler) sampler2D(ae_images[AE_IMAGE(image)], ae_samplers[AE_SAMPLER(sampler)])
#define AE_S2D_ARRAY(image, sampler) sampler2DArray(ae_images_array[AE_IMAGE(image)], ae_samplers[AE_SAMPLER(sampler)])
#define AE_US2D(image, sampler) usampler2D(ae_images_uint[AE_IMAGE(image)], ae_samplers[AE_SAMPLER(sampler)])
#define AE_SHADOW(image, sampler) sampler2DShadow(ae_images[AE_IMAGE(image)], ae_samplers_shadow[AE_SAMPLER(sampler)])
#define AE_S2D_AT(index, sampler) sampler2D(ae_images[nonuniformEXT(index)], ae_samplers[nonuniformEXT(sampler)])
#define AE_SHADOW_AT(index, sampler) sampler2DShadow(ae_images[nonuniformEXT(index)], ae_samplers_shadow[nonuniformEXT(sampler)])
#endif
