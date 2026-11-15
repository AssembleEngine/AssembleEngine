// The exact reference (tests only): every fragment of every particle is
// stored per pixel, then sorted by its sphere's front depth and blended.
#define REFERENCE_LAYERS 64u
layout(std430, set = 1, binding = 7) buffer RefCounts { uint refCounts[]; };
// Front depth bits, colour (premultiplied, half floats), unused.
layout(std430, set = 1, binding = 8) buffer RefFragments { uvec4 refFragments[]; };
