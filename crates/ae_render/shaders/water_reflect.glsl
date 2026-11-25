#ifndef AE_WATER_REFLECT
#define AE_WATER_REFLECT

// The water reads the planar reflection at its own screen position moved by
// at most this much on each axis (UV units; water.frag clamps its ripple and
// churn offsets to it). The units' mirror cull widens each unit's mirrored
// screen box by the same amount, so the two must change together.
const float REFLECT_SHIFT = 0.08;

#endif
