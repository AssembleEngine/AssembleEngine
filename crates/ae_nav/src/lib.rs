//! Navigation services lifted from the `pathsim` prototype: terrain grids,
//! the sector/portal hierarchy, destination contexts, flow fields, crowd
//! costs, ORCA avoidance and collision. Everything here is a pure function
//! over arrays or an immutable structure behind `Arc`; unit state, steering
//! and command handling live in the engine, not in this crate.

pub mod avoidance;
pub mod collision;
pub mod math;
pub mod nav;
pub mod par;
pub mod rng;
pub mod terrain;
pub mod thread_load;
pub mod workers;
