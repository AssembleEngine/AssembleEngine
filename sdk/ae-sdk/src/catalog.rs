//! Spawning catalogue defs (feature `catalog`): a game emits
//! [`SpawnRequest`]s and the engine's catalogue service (`ae_services::
//! catalog`) spawns the def's components for that allyteam at that
//! position, with a `Team`, the position in `Body` and a [`Def`] naming
//! the def. A def id is [`def_id`] of its name.

use crate::Component;

/// Spawn catalogue def `def` for allyteam `ally` at `(x, y)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_000B, event)]
pub struct SpawnRequest {
    pub def: u64,
    pub ally: u32,
    pub reserved: u32,
    pub x: f32,
    pub y: f32,
}

/// The def an entity was spawned from.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_000C)]
pub struct Def {
    #[field(view = "los")]
    pub id: u64,
}

/// A def's id: FNV-1a of its name (as `ae_catalog::def_id`).
pub const fn def_id(name: &str) -> u64 {
    crate::abi::name_id(name)
}

/// Declare the catalogue service's components and events (and the nav
/// ones it needs), see [`crate::use_nav!`].
#[macro_export]
macro_rules! use_catalog {
    () => {
        $crate::use_nav!();
        $crate::declare!($crate::catalog::SpawnRequest, $crate::catalog::Def);
    };
}
