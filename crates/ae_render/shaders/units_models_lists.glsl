// The instance count of model command k.
uint countWord(uint k) { return 4u + 5u * k + 1u; }

// Unit i into its class's region of list l (0: the camera's, k + 1:
// clipmap k's), counted in each of the class's commands of that list.
void append(uint i, ModelClass mc, uint l) {
    uint shift = l * u.counts.w;
    uint first = classCommands[mc.cmd.x];
    uint slot = atomicAdd(cmdWords[countWord(first + shift)], 1u);
    for (uint k = 1u; k < mc.cmd.y; k++) {
        atomicAdd(cmdWords[countWord(classCommands[mc.cmd.x + k] + shift)], 1u);
    }
    visible[draws[first].visibleBase + l * u.counts.y + slot] = i;
}
