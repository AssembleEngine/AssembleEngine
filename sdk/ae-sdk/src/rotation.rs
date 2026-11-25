//! Explicit discontinuities in a unit's rendered orientation.

use crate::{Component, abi};

/// Skip interpolation into a unit's current rotation when the sequence
/// changes. Applies to both explicit orientation and velocity-derived facing.
/// Position and skeletal animation have their own interpolation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Component)]
#[component(id = 0xAE00_0000_0000_0016)]
pub struct RotationSnap {
    /// Zero means no snap has been requested.
    #[field(view = "los")]
    pub sequence: u32,
}

const _: () = assert!(RotationSnap::ID == abi::engine::ROTATION_SNAP);

impl RotationSnap {
    /// Call on the tick that changes the unit's facing.
    pub fn snap(&mut self) {
        self.sequence = self.sequence.wrapping_add(1).max(1);
    }
}
