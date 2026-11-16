#version 460
// units_models.frag with the alpha mask (its own pipeline: discarding
// loses early depth, so opaque models do not pay for it).
#define MASKED
#include "units_models.frag.glsl"
