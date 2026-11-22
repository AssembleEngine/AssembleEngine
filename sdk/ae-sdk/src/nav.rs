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
//! * [`Team`] and [`Sensor`]: who owns a unit and what it sees (line of
//!   sight and radar radii), read by the copy layer (`ae_view`) to build
//!   each observer's view. `nav_spawn` gives them to the map's units when
//!   the nav plugin has teams.
//!
//! Fields carry their view class (`#[field(view = ...)]`): positions show
//! on radar (wobbled), motion and kind in line of sight, orders and goals
//! to the owner only.
//!
//! * [`Goal`]: where the map or scenario wants the unit to go. The engine
//!   never acts on it; game systems decide when to order the unit there.
//! * [`MoveOrder`]: an event that orders one unit (move, stop, route or
//!   formation). Orders with the same `group` emitted in one tick are one
//!   group command (a shared flow field and formation slots); the group id is
//!   the nav command id, so a game never reuses one. [`OrderPoint`] events
//!   give a route's or formation's guide points.
//! * [`TerrainEdit`] / [`TerrainChanged`]: events that edit the terrain, and
//!   that report when an edit took effect. [`CostOverlay`]: an event that
//!   adds or removes extra traversal cost on a rectangle.

use crate::{Component, Entity};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0001)]
pub struct Body {
    #[field(view = "radar")]
    pub x: f32,
    #[field(view = "radar")]
    pub y: f32,
    #[field(view = "los")]
    pub vx: f32,
    #[field(view = "los")]
    pub vy: f32,
    #[field(view = "los")]
    pub radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0002)]
pub struct NavAgent {
    /// Top speed, world units per second. Read when the unit joins the
    /// service.
    #[field(view = "los")]
    pub max_speed: f32,
    /// Movement profile (class) index of the map's profiles.
    #[field(view = "los")]
    pub profile: u32,
    /// [`mode`] after the last nav tick.
    #[field(view = "owner")]
    pub mode: u32,
}

/// The allyteam that owns an entity (observers of it see `Own` rows).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0009)]
pub struct Team {
    #[field(view = "always")]
    pub ally: u32,
}

/// What an entity sees: line-of-sight and radar radii, world units (0:
/// none).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_000A)]
pub struct Sensor {
    #[field(view = "los")]
    pub los: f32,
    #[field(view = "los")]
    pub radar: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0003)]
pub struct Goal {
    #[field(view = "owner")]
    pub x: f32,
    #[field(view = "owner")]
    pub y: f32,
    /// Group to order the unit with (0: none).
    pub group: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0004, event, command)]
pub struct MoveOrder {
    pub entity: Entity,
    /// Where the group goes (a route's end, the formation's anchor when it
    /// has no guide points); the group's first order sets it. Ignored by
    /// [`order::STOP`].
    pub x: f32,
    pub y: f32,
    pub group: u64,
    /// [`order`] kind; every order of a group has the same.
    pub kind: u32,
    pub reserved: u32,
}

impl MoveOrder {
    /// Go to `(x, y)`; the group shares one flow field and arrival slots.
    pub fn move_to(entity: Entity, x: f32, y: f32, group: u64) -> Self {
        Self {
            entity,
            x,
            y,
            group,
            kind: order::MOVE,
            reserved: 0,
        }
    }

    /// Leave the current command and stand where the unit comes to rest.
    pub fn stop(entity: Entity, group: u64) -> Self {
        Self {
            entity,
            x: 0.0,
            y: 0.0,
            group,
            kind: order::STOP,
            reserved: 0,
        }
    }

    /// Go to `(x, y)` along the group's guide points ([`OrderPoint`]), in
    /// order, gathering at the end.
    pub fn route(entity: Entity, x: f32, y: f32, group: u64) -> Self {
        Self {
            entity,
            x,
            y,
            group,
            kind: order::ROUTE,
            reserved: 0,
        }
    }

    /// Take a slot spread evenly along the polyline of the group's guide
    /// points (or around `(x, y)` without any).
    pub fn formation(entity: Entity, x: f32, y: f32, group: u64) -> Self {
        Self {
            entity,
            x,
            y,
            group,
            kind: order::FORMATION,
            reserved: 0,
        }
    }
}

/// [`MoveOrder::kind`] values.
pub mod order {
    pub const MOVE: u32 = 0;
    pub const STOP: u32 = 1;
    pub const ROUTE: u32 = 2;
    pub const FORMATION: u32 = 3;
    /// Most guide points a group may have.
    pub const MAX_POINTS: u32 = 64;
    /// [`OrderPoint::index`] of a formation's facing: the point its front
    /// faces. Rows then stack behind the line, away from it, instead of
    /// alternating on both sides.
    pub const FACING: u32 = u32::MAX;
}

/// Guide point `index` (0, 1, .. without gaps, at most
/// [`order::MAX_POINTS`]) of the route or formation of `group`, emitted in
/// the same tick as the group's orders; or, with index [`order::FACING`],
/// the point a formation faces. A point may be emitted more than
/// once (say, by every unit of the group) if the copies agree.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0007, event, command)]
pub struct OrderPoint {
    pub group: u64,
    pub index: u32,
    pub x: f32,
    pub y: f32,
    pub reserved: u32,
}

/// Start point `index` of `count` of the loaded map, with the map's side
/// (`extent`, world units). The nav plugin's `map_starts` host system emits
/// them once, at tick 0: a game that schedules `host(map_starts)` before
/// its own start system reads them the same tick. Generated maps have no
/// start points (nothing is emitted).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_000D, event)]
pub struct MapStart {
    pub index: u32,
    pub count: u32,
    pub x: f32,
    pub y: f32,
    pub extent: f32,
    pub reserved: u32,
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
    /// [`changed`] flags.
    pub flags: u32,
}

/// [`TerrainChanged::flags`].
pub mod changed {
    /// The ground changed (blocked, height or material), so it looks
    /// different; without it only traversal costs did (cost edits,
    /// overlays).
    pub const GROUND: u32 = 1;
}

/// An event that adds ([`overlay::ADD`]) cost overlay `id`, `cost` tenths
/// (1..=254) of extra traversal cost on every open cell of the inclusive
/// cell rectangle `(x0, y0)..=(x1, y1)` (clipped to the map), or removes
/// ([`overlay::REMOVE`], other fields ignored) it. Overlays on a cell add
/// up (an open cell costs at most 254); unlike a [`TerrainEdit`] of the cost,
/// removing one restores what was under it. Danger zones, slow fields,
/// player-placed waypoints. An overlay lands like a terrain edit, with its
/// navigation repair, and is reported by a [`TerrainChanged`] without
/// [`changed::GROUND`]. It applies to every movement class. Adding an id in
/// use, or removing one not in use, stops the tick with an error.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0008, event)]
pub struct CostOverlay {
    pub id: u64,
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
    pub cost: u32,
    pub op: u32,
}

/// [`CostOverlay::op`] values.
pub mod overlay {
    pub const ADD: u32 = 0;
    pub const REMOVE: u32 = 1;
}

/// [`TerrainEdit::set`] flags.
pub mod terrain {
    pub const BLOCKED: u32 = 1;
    pub const HEIGHT: u32 = 2;
    pub const COST: u32 = 4;
}

/// A movement class: where units of the class may go. A game declares its
/// classes once with [`movement_classes!`](crate::movement_classes); a
/// unit's [`NavAgent::profile`] is an index into that list. Without the
/// declaration the map's own classes apply.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovementClass {
    /// At most [`CLASS_NAME_MAX`] bytes, unique.
    pub name: &'static str,
    /// Clearance radius in cells: a path is valid only where the whole
    /// `(2c + 1)²` square around each cell is open. At least
    /// `ceil(radius / CELL_SIZE - 0.5)`, so the nominal body fits.
    pub clearance_cells: u32,
    /// Steepest climb, in decimetres of height per cell.
    pub max_slope: u32,
    /// Nominal body radius and speed (world units, per second): what map
    /// spawners give the class's units. The service moves each unit with
    /// its own `Body::radius` and `NavAgent::max_speed`.
    pub radius: f32,
    pub speed: f32,
    /// Water depths (decimetres) the class may enter: land units
    /// `0..=ford`, ships `draft..=ANY_DEPTH`.
    pub min_depth: u32,
    pub max_depth: u32,
    /// Floats over water of any depth (and still crosses land).
    pub hover: bool,
}

/// [`MovementClass::max_depth`]: no limit.
pub const ANY_DEPTH: u32 = u32::MAX;
pub const CLASS_NAME_MAX: usize = 32;
/// Name of the data record [`movement_classes!`](crate::movement_classes)
/// writes.
pub const CLASSES_DATA: &str = "nav.classes";
/// Bytes per class in that record: name (32, zero-padded), then
/// `clearance_cells, max_slope: u32, radius, speed: f32, min_depth,
/// max_depth, hover, 0: u32`.
pub const CLASS_BYTES: usize = 64;

/// Length of the record of `n` classes.
pub const fn classes_record_len(n: usize) -> usize {
    crate::abi::data_header_len(CLASSES_DATA) + n * CLASS_BYTES
}

/// The record of `classes` (`B` must be [`classes_record_len`]).
pub const fn classes_record<const B: usize>(classes: &[MovementClass]) -> [u8; B] {
    use crate::abi::{put_bytes, put_u32};
    assert!(B == classes_record_len(classes.len()), "record length");
    let mut b = [0u8; B];
    let mut at = crate::abi::put_data_header(&mut b, CLASSES_DATA);
    let mut i = 0;
    while i < classes.len() {
        let c = &classes[i];
        assert!(
            !c.name.is_empty() && c.name.len() <= CLASS_NAME_MAX,
            "a movement class name is 1..=32 bytes"
        );
        put_bytes(&mut b, at, c.name.as_bytes());
        put_u32(&mut b, at + 32, c.clearance_cells);
        put_u32(&mut b, at + 36, c.max_slope);
        put_u32(&mut b, at + 40, c.radius.to_bits());
        put_u32(&mut b, at + 44, c.speed.to_bits());
        put_u32(&mut b, at + 48, c.min_depth);
        put_u32(&mut b, at + 52, c.max_depth);
        put_u32(&mut b, at + 56, c.hover as u32);
        at += CLASS_BYTES;
        i += 1;
    }
    b
}

/// Declare the game's movement classes (at most one per module), index 0
/// first:
///
/// ```ignore
/// ae_sdk::movement_classes![
///     MovementClass { name: "tank", clearance_cells: 2, max_slope: 95, radius: 4.2, speed: 80.0, min_depth: 0, max_depth: 20, hover: false },
///     MovementClass { name: "boat", clearance_cells: 2, max_slope: 60, radius: 5.0, speed: 72.0, min_depth: 30, max_depth: ANY_DEPTH, hover: false },
/// ];
/// ```
#[macro_export]
macro_rules! movement_classes {
    ($($class:expr),+ $(,)?) => {
        const _: () = {
            const CLASSES: &[$crate::nav::MovementClass] = &[$($class),+];
            #[used]
            #[cfg_attr(target_family = "wasm", unsafe(link_section = "ae.schema"))]
            static RECORD: [u8; $crate::nav::classes_record_len(CLASSES.len())] = $crate::nav::classes_record(CLASSES);
        };
    };
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
    assert!(<OrderPoint as crate::Event>::ID == crate::abi::engine::ORDER_POINT);
    assert!(<TerrainEdit as crate::Event>::ID == crate::abi::engine::TERRAIN_EDIT);
    assert!(<TerrainChanged as crate::Event>::ID == crate::abi::engine::TERRAIN_CHANGED);
    assert!(<CostOverlay as crate::Event>::ID == crate::abi::engine::COST_OVERLAY);
    assert!(<MapStart as crate::Event>::ID == crate::abi::engine::MAP_START);
};

/// Declare every component and event the engine's nav, spatial and view
/// services read or write (see [`crate::declare!`]), for a module that
/// schedules those services without naming all of them in its systems.
#[macro_export]
macro_rules! use_nav {
    () => {
        $crate::declare!(
            $crate::nav::Body,
            $crate::nav::NavAgent,
            $crate::nav::Goal,
            $crate::nav::Team,
            $crate::nav::Sensor,
            $crate::nav::MoveOrder,
            $crate::nav::OrderPoint,
            $crate::nav::TerrainEdit,
            $crate::nav::TerrainChanged,
            $crate::nav::CostOverlay,
        );
    };
}
