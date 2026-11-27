//! A nav unit killed into `Dying` (parameter bit 0), or despawned (bit 0
//! clear), and two neighbours moving past where it stood.
//!
//! Systems: `start` (three units in a column with goals), `orders` (tick 0,
//! a `MoveOrder` each), `kill` (tick 5, the middle unit), `host(nav)`.

use ae_sdk::life::{self, Dying};
use ae_sdk::nav::{ANY_DEPTH, Body, Goal, MoveOrder, MovementClass, NavAgent};
use ae_sdk::script::Callins;
use ae_sdk::{Component, Run, Spawn, despawn, emit, movement_classes, schedule, system};

movement_classes![
    MovementClass {
        name: "Tank",
        clearance_cells: 1,
        max_slope: 60,
        radius: 1.0,
        speed: 4.0,
        min_depth: 0,
        max_depth: ANY_DEPTH,
        hover: false
    },
    MovementClass {
        name: "Fast",
        clearance_cells: 1,
        max_slope: 60,
        radius: 0.8,
        speed: 6.0,
        min_depth: 0,
        max_depth: ANY_DEPTH,
        hover: false
    }
];

/// The unit's place in the test (rows move between archetypes, so a
/// system cannot key on its row index).
#[repr(C)]
#[derive(Clone, Copy, Default, Component)]
pub struct Role {
    pub n: u32,
    pub reserved: u32,
}

/// The kill happens at this tick.
pub const KILL_AT: u64 = 5;

#[system()]
fn start(run: &Run) {
    if run.tick() != 0 {
        return;
    }
    for n in 0..3 {
        Spawn::new()
            .with(&Body {
                x: 100.0,
                y: 100.0 + n as f32 * 12.0,
                vx: 0.0,
                vy: 0.0,
                radius: 1.0,
            })
            .with(&NavAgent {
                max_speed: 4.0,
                profile: 0,
                mode: 0,
            })
            .with(&Goal {
                x: 300.0,
                y: 100.0 + n as f32 * 12.0,
                group: n as u64 + 1,
            })
            .with(&Callins::default())
            .with(&Role { n, reserved: 0 })
            .submit();
    }
}

#[system(read(Goal), emits(MoveOrder))]
fn orders(run: &Run) {
    if run.tick() != 0 {
        return;
    }
    let (goals, ids) = (run.read::<Goal>(), run.entities());
    for i in run.rows() {
        emit(&MoveOrder::move_to(ids[i], goals[i].x, goals[i].y, goals[i].group));
    }
}

#[system(read(Goal), read(Role), write(Callins), get(Dying))]
fn kill(run: &Run) {
    let (roles, posts, ids) = (run.read::<Role>(), run.write::<Callins>(), run.entities());
    for i in run.rows() {
        if roles[i].n != 1 || run.tick() != KILL_AT {
            continue;
        }
        if run.param() & 1 == 1 {
            life::kill(ids[i], &mut posts[i], run.tick(), 0);
        } else {
            despawn(ids[i]);
        }
    }
}

schedule!(start, orders, kill, host(nav));
