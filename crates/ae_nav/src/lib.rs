//! Navigation services lifted from the `pathsim` prototype: terrain grids,
//! the sector/portal hierarchy, destination contexts, flow fields, crowd
//! costs, ORCA avoidance and collision. Everything outside [`reference`] is a
//! pure function over arrays or an immutable structure behind `Arc`.
//! [`reference`] is pathsim's own simulation, kept to check the engine's
//! columnar navigation service against.

pub mod avoidance;
pub mod collision;
pub mod math;
pub mod nav;
pub mod par;
pub mod reference;
pub mod rng;
pub mod scenario;
pub mod service;
pub mod terrain;
pub mod thread_load;
pub mod workers;
