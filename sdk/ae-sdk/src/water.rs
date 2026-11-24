//! Opt-in components for submerged movement and underwater detection.

use crate::Component;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Component)]
pub struct Dive {
    /// Metres below the surface. Zero is surfaced.
    #[field(view = "los")]
    pub depth: f32,
    /// Requested depth in metres.
    #[field(view = "los")]
    pub target: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Component)]
pub struct Sonar {
    /// Detection radius in world units.
    pub radius: f32,
}

/// Move toward the requested depth by at most `metres_per_tick`.
pub fn step_dive(dive: &mut Dive, metres_per_tick: f32) {
    let step = metres_per_tick.max(0.0);
    let delta = (dive.target.max(0.0) - dive.depth).clamp(-step, step);
    dive.depth = (dive.depth + delta).max(0.0);
}
