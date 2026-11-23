// The particles' LSD radix sort, 8 bits a pass, over the live keys
// (cmd.instanceCount of them). Ported from ~/particles (itself from
// VkRadixSort, https://github.com/MircoWerner/VkRadixSort): keys only (the
// key carries the slot), the count read from the indirect buffer, and the
// buffers chosen by direction (0: keys -> keysTmp, 1: back).
#define PARTICLES_WRITE
#define AE_PUSH_WORDS uint shift; uint direction; uint pushPad0; uint pushPad1; uvec4 pushPad2;
#include "particles_common.glsl"

#define WORKGROUP_SIZE 256
#define RADIX_SORT_BINS 256
#define BLOCKS_PER_WORKGROUP 4u

layout(local_size_x = WORKGROUP_SIZE) in;


uint numWorkgroups() {
    return (u.pool.x + BLOCKS_PER_WORKGROUP * WORKGROUP_SIZE - 1u) / (BLOCKS_PER_WORKGROUP * WORKGROUP_SIZE);
}

uint keyIn(uint i) {
    return ae.direction == 0u ? keys[i] : keysTmp[i];
}

void keyOut(uint i, uint key) {
    if (ae.direction == 0u) {
        keysTmp[i] = key;
    } else {
        keys[i] = key;
    }
}
