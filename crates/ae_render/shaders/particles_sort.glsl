// The particles' LSD radix sort, 8 bits a pass, over the live keys
// (cmd.instanceCount of them). Ported from ~/particles (itself from
// VkRadixSort, https://github.com/MircoWerner/VkRadixSort): keys only (the
// key carries the slot), the count read from the indirect buffer, and the
// buffers chosen by direction (0: keys -> keysTmp, 1: back).
#define PARTICLES_WRITE
#include "particles_common.glsl"

#define WORKGROUP_SIZE 256
#define RADIX_SORT_BINS 256
#define BLOCKS_PER_WORKGROUP 4u

layout(local_size_x = WORKGROUP_SIZE) in;

layout(push_constant) uniform Push {
    uint shift;
    uint direction;
} push;

uint numWorkgroups() {
    return (u.pool.x + BLOCKS_PER_WORKGROUP * WORKGROUP_SIZE - 1u) / (BLOCKS_PER_WORKGROUP * WORKGROUP_SIZE);
}

uint keyIn(uint i) {
    return push.direction == 0u ? keys[i] : keysTmp[i];
}

void keyOut(uint i, uint key) {
    if (push.direction == 0u) {
        keysTmp[i] = key;
    } else {
        keys[i] = key;
    }
}
