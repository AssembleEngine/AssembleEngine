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
};
