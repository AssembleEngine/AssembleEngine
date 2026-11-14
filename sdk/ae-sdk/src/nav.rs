//! Components the engine's navigation and spatial services read and write.
//!
//! Their ids are fixed (`ae_abi::engine`) and the host checks a module's
//! layout of each against its own copy, so a game uses these types instead
//! of declaring look-alikes.
//!
//! * [`Body`]: position, velocity and radius. The nav service writes
//!   position and velocity of every unit with a [`NavAgent`] each tick; the
//!   spatial index is built from every body's position.
//! * [`NavAgent`]: a unit the nav service moves. `mode` is written back each
//!   tick ([`mode`] values).
//! * [`Goal`]: where the map or scenario wants the unit to go. The engine
//!   never acts on it; game systems decide when to order the unit there.
//! * [`MoveOrder`]: an event that orders one unit to a point. Orders with the
//!   same `group` emitted in one tick are one group command (a shared flow
//!   field and formation slots); the group id is the nav command id, so a
//!   game never reuses one.
//! * [`TerrainEdit`] / [`TerrainChanged`]: events that edit the terrain, and
//!   that report when an edit took effect.

use crate::{Component, Entity};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0001)]
pub struct Body {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0002)]
pub struct NavAgent {
    /// Top speed, world units per second. Read when the unit joins the
    /// service.
    pub max_speed: f32,
    /// Movement profile (class) index of the map's profiles.
    pub profile: u32,
    /// [`mode`] after the last nav tick.
    pub mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0003)]
pub struct Goal {
    pub x: f32,
    pub y: f32,
    /// Group to order the unit with (0: none).
    pub group: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0004, event)]
pub struct MoveOrder {
    pub entity: Entity,
    pub x: f32,
    pub y: f32,
    pub group: u64,
}

/// Side of a map cell in world units. Terrain is edited in cells.
pub const CELL_SIZE: f32 = 8.0;

/// An event that edits the terrain in the inclusive cell rectangle
/// `(x0, y0)..=(x1, y1)` (clipped to the map; empty if `x1 < x0` or
/// `y1 < y0`). `set` selects the fields that apply ([`terrain`] flags).
///
/// The nav service takes every edit emitted since its last run as one patch,
/// in event order (a later edit of a cell wins). A patch becomes
/// authoritative together with its navigation repair, a fixed sim-time delay
/// later; one repair runs at a time, and patches made meanwhile land together
/// with the next one. When a repair lands, units the edits buried are moved
/// to the nearest open cell and the service emits one [`TerrainChanged`].
/// Values out of range or unknown `set` flags stop the tick with an error.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0005, event)]
pub struct TerrainEdit {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
    pub set: u32,
    /// 1: solid (also makes the cell rock, cost 255); 0: open.
    pub blocked: u32,
    /// Ground height in decimetres (0..=2400); the water surface stays.
    pub height: i32,
    /// Traversal cost in tenths (10 = ordinary ground, 1..=255).
    pub cost: u32,
}

/// An event the nav service emits when a terrain repair lands, making its
/// patches authoritative (visible to navigation and collision): the inclusive
/// cell rectangle covering every cell they edited, and the terrain version
/// after them. Versions count patches, so they skip when one repair carries
/// several. What render and the copy layer refresh.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0006, event)]
pub struct TerrainChanged {
    pub version: u32,
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

/// [`TerrainEdit::set`] flags.
pub mod terrain {
    pub const BLOCKED: u32 = 1;
    pub const HEIGHT: u32 = 2;
    pub const COST: u32 = 4;
}

/// [`NavAgent::mode`] values.
pub mod mode {
    pub const IDLE: u32 = 0;
    pub const WAITING: u32 = 1;
    pub const MOVING: u32 = 2;
    pub const ARRIVED: u32 = 3;
    pub const BLOCKED: u32 = 4;
    pub const UNREACHABLE: u32 = 5;
}

const _: () = {
    assert!(<Body as Component>::ID == crate::abi::engine::BODY);
    assert!(<NavAgent as Component>::ID == crate::abi::engine::NAV_AGENT);
    assert!(<Goal as Component>::ID == crate::abi::engine::GOAL);
    assert!(<MoveOrder as crate::Event>::ID == crate::abi::engine::MOVE_ORDER);
    assert!(<TerrainEdit as crate::Event>::ID == crate::abi::engine::TERRAIN_EDIT);
    assert!(<TerrainChanged as crate::Event>::ID == crate::abi::engine::TERRAIN_CHANGED);
};
