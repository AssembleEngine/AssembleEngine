//! The pathsim prototype's whole simulation (`sim.rs`, `commands.rs`,
//! `replay.rs`), lifted unchanged.
//!
//! This is a reference, not the engine's navigation service: `Simulation`
//! owns its own units and positions, which in the engine belong to the ECS.
//! It exists so the columnar service in [`crate::service`] can be checked
//! against it (same scenario, same digest) and so navbench numbers can be
//! compared with the prototype on the same machine.

pub mod commands;
pub mod replay;
pub mod sim;
