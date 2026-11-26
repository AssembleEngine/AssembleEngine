#include "shadow_storage.glsl"

void storeShadowDepth() {
    ivec2 texel = ivec2(gl_FragCoord.xy);
    uint entry = imageLoad(shadowStorage, shadowTableAddress(texel / SHADOW_PAGE_SIZE, int(ae.words.x))).r;
    if ((entry & SHADOW_PAGE_DIRTY) != 0u && entry != SHADOW_PAGE_UNBACKED) {
        imageAtomicMax(shadowStorage, shadowPhysicalTexel(entry & SHADOW_SLOT_MASK, texel), floatBitsToUint(gl_FragCoord.z));
    }
}
