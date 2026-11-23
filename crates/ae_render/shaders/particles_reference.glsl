// The exact reference (tests only): every fragment of every particle is
// stored per pixel, then sorted by its sphere's front depth and blended.
#define REFERENCE_LAYERS 64u
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) buffer RefCounts { uint refCounts[]; } ae_RefCounts[];
#define refCounts ae_RefCounts[AE_BUF(7)].refCounts
// Front depth bits, colour (premultiplied, half floats), unused.
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) buffer RefFragments { uvec4 refFragments[]; } ae_RefFragments[];
#define refFragments ae_RefFragments[AE_BUF(8)].refFragments
