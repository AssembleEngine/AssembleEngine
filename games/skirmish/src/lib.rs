//! Skirmish: two allyteams on a nav map, defined by content.
//!
//! Game parameter: bits 0..24 = 0 (the map spawns no units of its own),
//! bits 24..40 = the map's side in metres (bases sit 15% in from opposite
//! corners), bits 40..48 = tanks each allyteam starts with.
//!
//! * `start` (tick 0): for each allyteam a `factory` at its base and its
//!   starting `tank`s in a ring around it, as `SpawnRequest`s; the engine's
//!   catalogue service spawns them from the defs in `content/defs/`.

use ae_sdk::catalog::{SpawnRequest, def_id};
use ae_sdk::nav::{ANY_DEPTH, CELL_SIZE, MovementClass};
use ae_sdk::{Component, Run, emit, schedule, system};

ae_sdk::movement_classes![
    MovementClass { name: "Scout", clearance_cells: 1, max_slope: 135, radius: CELL_SIZE * 0.34, speed: CELL_SIZE * 14.0, min_depth: 0, max_depth: 20, hover: false },
    MovementClass { name: "Heavy", clearance_cells: 2, max_slope: 95, radius: CELL_SIZE * 0.52, speed: CELL_SIZE * 10.0, min_depth: 0, max_depth: 20, hover: false },
    MovementClass { name: "Ship", clearance_cells: 2, max_slope: 60, radius: CELL_SIZE * 0.62, speed: CELL_SIZE * 9.0, min_depth: 30, max_depth: ANY_DEPTH, hover: false },
    MovementClass { name: "Hover", clearance_cells: 1, max_slope: 110, radius: CELL_SIZE * 0.40, speed: CELL_SIZE * 13.0, min_depth: 0, max_depth: ANY_DEPTH, hover: true },
    MovementClass { name: "Amphibious", clearance_cells: 1, max_slope: 110, radius: CELL_SIZE * 0.48, speed: CELL_SIZE * 9.0, min_depth: 0, max_depth: ANY_DEPTH, hover: false },
];

/// Hit points (a game component; defs set them).
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Health {
    #[field(view = "los")]
    pub hp: u32,
    #[field(view = "los")]
    pub max: u32,
}

// The engine services' components (Def, Team and Body are only ever
// created by the catalogue service).
ae_sdk::use_catalog!();

pub const ALLIES: u32 = 2;

/// The map's side, metres.
pub fn extent(run: &Run) -> f32 {
    ((run.param() >> 24) & 0xFFFF) as f32
}

/// Allyteam `ally`'s base.
pub fn base(run: &Run, ally: u32) -> (f32, f32) {
    let e = extent(run);
    if ally == 0 { (e * 0.15, e * 0.15) } else { (e * 0.85, e * 0.85) }
}

fn spawn(def: &str, ally: u32, x: f32, y: f32) {
    emit(&SpawnRequest { def: def_id(def), ally, reserved: 0, x, y });
}

#[system(emits(SpawnRequest))]
fn start(run: &Run) {
    if run.tick() != 0 {
        return;
    }
    let tanks = (run.param() >> 40) & 0xFF;
    for ally in 0..ALLIES {
        let (bx, by) = base(run, ally);
        spawn("factory", ally, bx, by);
        for i in 0..tanks {
            let a = i as f32 / tanks.max(1) as f32 * core::f32::consts::TAU;
            // A fixed-order polynomial for cos/sin keeps the guest free of
            // libm differences (f32 + - * only).
            let (c, s) = cos_sin(a);
            spawn("tank", ally, bx + 90.0 * c, by + 90.0 * s);
        }
    }
}

/// cos and sin of `a` in [0, 2π) by Taylor series around the nearest
/// multiple of π/2 (deterministic: only + - *).
fn cos_sin(a: f32) -> (f32, f32) {
    let q = (a / core::f32::consts::FRAC_PI_2 + 0.5) as i32;
    let x = a - q as f32 * core::f32::consts::FRAC_PI_2;
    let x2 = x * x;
    let s = x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0)));
    let c = 1.0 - x2 / 2.0 * (1.0 - x2 / 12.0 * (1.0 - x2 / 30.0));
    match q.rem_euclid(4) {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    }
}

schedule!(start, host(catalog_spawn), host(nav), host(spatial));
