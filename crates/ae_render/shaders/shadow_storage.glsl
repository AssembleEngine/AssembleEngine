#include "shadow_pages.glsl"
layout(r32ui, set = AE_BINDLESS_SET, binding = AE_STORAGE_IMAGES) uniform uimage2D ae_shadow_images[];
#define shadowStorage ae_shadow_images[ae.words.z]
