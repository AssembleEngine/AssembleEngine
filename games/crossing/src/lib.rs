//! Crossing game: the M2 exit workload.
//!
//! The map (the host's `nav_spawn`) places units of every movement class
//! and gives each a `Goal`. The game decides when units move: `orders`
//! sends every unit with a goal there at tick `delay` (game parameter bits
//! 32..48; 0 by default), one `MoveOrder` per unit, grouped as the map
//! grouped them. The engine's nav service (`host(nav)`) moves the units
//! every tick; the spatial index (`host(spatial)`) is rebuilt from their
//! bodies, and `crowding` counts each unit's neighbours with it, a quarter
//! of the units per tick.
//!
//! `churn` (tests only) replaces units at one tick: every unit whose entity
//! index is a multiple of 7 is despawned and a new one spawned with the same
//! body and agent, tagged with the entity it replaces. The nav service sees
//! units leave and join (with recycled entity ids) in one tick.
//!
//! `quake` (tests only) edits the terrain on two consecutive ticks. First,
//! around every body whose entity index is a multiple of 97, a square of
//! rock in a wider square of slow ground: the units there are buried when
//! the edit lands and the nav service moves them out. Then, around every
//! body whose index is 48 more than a multiple of 97, a raised square of
//! ground (land in a lake, a hill on land). The second edit arrives while
//! the first one's repair is still running, so it lands with the next
//! repair. `landed` records each `TerrainChanged` the nav service reports.
//!
//! `reorder` (tests only) gives every grouped unit a new order at one tick,
//! by its group: stop, a route through two guide points near the goal, or a
//! formation on a line through the goal.
//!
//! Game parameter: low 24 bits = unit count (read by the host's map), bits
//! 24..32 = the quake tick (0: none), bits 32..40 = the tick orders are
//! given, bits 40..48 = the reorder tick (0: none), bits 48..64 = the churn
//! tick (0: no churn).

use ae_sdk::nav::{Body, CELL_SIZE, Goal, MoveOrder, NavAgent, OrderPoint, TerrainChanged, TerrainEdit, terrain};
use ae_sdk::{Component, Entity, Run, Spawn, add, despawn, emit, schedule, spatial_radius, system};

/// Neighbours within four body radii, as of tick `tick`.
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Crowd {
    pub neighbours: u32,
    pub tick: u32,
}

/// On a unit `churn` spawned: the entity it replaced.
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Replaces {
    pub old: Entity,
}

/// A terrain version the nav service reported at tick `tick`.
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Landed {
    pub version: u32,
    pub tick: u32,
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

const NEIGHBOUR_RADII: f32 = 4.0;

// `quake`, mirrored in `ae_services/tests/crossing.rs`.
/// Half-sides (cells) of the rock square and of the slow ground around it,
/// and that ground's cost (tenths; 10 is ordinary ground).
const ROCK_HALF: i32 = 2;
const SLOW_HALF: i32 = 8;
const SLOW_COST: u32 = 40;
/// Half-side (cells) and ground height (decimetres) of the raised square.
const HILL_HALF: i32 = 4;
const HILL_HEIGHT: i32 = 900;
/// `quake` edits around units whose entity index is a multiple of this
/// (first tick), or 48 more than one (second tick).
const QUAKE_EVERY: u32 = 97;
const HILL_OFFSET: u32 = 48;

fn delay(run: &Run) -> u64 {
    (run.param() >> 32) & 0xFF
}

#[system(read(Goal), emits(MoveOrder))]
fn orders(run: &Run) {
    if run.tick() != delay(run) {
        return;
    }
    let (goals, ids) = (run.read::<Goal>(), run.entities());
    for i in run.rows() {
        let g = goals[i];
        if g.group != 0 {
            emit(&MoveOrder::move_to(ids[i], g.x, g.y, g.group));
        }
        add(ids[i], &Crowd { neighbours: 0, tick: 0 });
    }
}

// `reorder`, mirrored in `ae_services/tests/crossing.rs`.
/// Added to a unit's group id to make its new group's.
const REGROUP: u64 = 1 << 40;

/// Guide points of `reorder`'s route (`kind` 1) or formation (2) around a
/// goal at `(x, y)`.
fn reorder_points(kind: u64, x: f32, y: f32) -> [(f32, f32); 2] {
    if kind == 1 { [(x - 160.0, y + 160.0), (x - 80.0, y + 80.0)] } else { [(x - 96.0, y - 96.0), (x + 96.0, y + 96.0)] }
}

#[system(read(Goal), emits(MoveOrder, OrderPoint))]
fn reorder(run: &Run) {
    if run.tick() != (run.param() >> 40) & 0xFF || run.tick() == 0 {
        return;
    }
    let (goals, ids) = (run.read::<Goal>(), run.entities());
    for i in run.rows() {
        let g = goals[i];
        if g.group == 0 {
            continue;
        }
        let group = g.group + REGROUP;
        let kind = g.group % 3;
        if kind == 0 {
            emit(&MoveOrder::stop(ids[i], group));
            continue;
        }
        // Every unit emits its group's points; the copies agree.
        for (index, (x, y)) in reorder_points(kind, g.x, g.y).into_iter().enumerate() {
            emit(&OrderPoint { group, index: index as u32, x, y, reserved: 0 });
        }
        emit(&if kind == 1 { MoveOrder::route(ids[i], g.x, g.y, group) } else { MoveOrder::formation(ids[i], g.x, g.y, group) });
    }
}

/// The edits `quake` makes around world `(x, y)`: slow ground,
/// then rock (rock also sets cost 255, whatever the order).
fn quake_edits(x: f32, y: f32) -> [TerrainEdit; 2] {
    let (x0, y0, x1, y1) = square(x, y, SLOW_HALF);
    let slow = TerrainEdit { x0, y0, x1, y1, set: terrain::COST, blocked: 0, height: 0, cost: SLOW_COST };
    let (x0, y0, x1, y1) = square(x, y, ROCK_HALF);
    let rock = TerrainEdit { x0, y0, x1, y1, set: terrain::BLOCKED, blocked: 1, height: 0, cost: 0 };
    [slow, rock]
}

/// The edit of `quake`'s second tick around world `(x, y)`.
fn hill_edit(x: f32, y: f32) -> TerrainEdit {
    let (x0, y0, x1, y1) = square(x, y, HILL_HALF);
    TerrainEdit { x0, y0, x1, y1, set: terrain::HEIGHT, blocked: 0, height: HILL_HEIGHT, cost: 0 }
}

/// Cells within `half` cells of the cell holding world `(x, y)`.
fn square(x: f32, y: f32, half: i32) -> (i32, i32, i32, i32) {
    let (cx, cy) = ((x / CELL_SIZE) as i32, (y / CELL_SIZE) as i32);
    (cx - half, cy - half, cx + half, cy + half)
}

#[system(read(Body), emits(TerrainEdit))]
fn quake(run: &Run) {
    let at = (run.param() >> 24) & 0xFF;
    if at == 0 || !(at..=at + 1).contains(&run.tick()) {
        return;
    }
    let (bodies, ids) = (run.read::<Body>(), run.entities());
    for i in run.rows() {
        let (b, k) = (bodies[i], ids[i].index() % QUAKE_EVERY);
        if run.tick() == at && k == 0 {
            for e in quake_edits(b.x, b.y) {
                emit(&e);
            }
        } else if run.tick() == at + 1 && k == HILL_OFFSET {
            emit(&hill_edit(b.x, b.y));
        }
    }
}

#[system(reads_events(TerrainChanged))]
fn landed(run: &Run) {
    for c in run.events::<TerrainChanged>() {
        Spawn::new().with(&Landed { version: c.version, tick: run.tick() as u32, x0: c.x0, y0: c.y0, x1: c.x1, y1: c.y1 }).submit();
    }
}

#[system(staggered(4), after(spatial), read(Body), write(Crowd))]
fn crowding(run: &Run) {
    let (bodies, crowd) = (run.read::<Body>(), run.write::<Crowd>());
    let mut near = [Entity(0); 64];
    for i in run.rows() {
        let b = bodies[i];
        let n = spatial_radius(b.x, b.y, b.radius * NEIGHBOUR_RADII, &mut near);
        // The unit itself is always a hit.
        crowd[i] = Crowd { neighbours: n.saturating_sub(1) as u32, tick: run.tick() as u32 };
    }
}

#[system(read(Body, NavAgent))]
fn churn(run: &Run) {
    if run.tick() != run.param() >> 48 || run.tick() == 0 {
        return;
    }
    let (bodies, agents, ids) = (run.read::<Body>(), run.read::<NavAgent>(), run.entities());
    for i in run.rows() {
        if ids[i].index() % 7 == 0 {
            despawn(ids[i]);
            Spawn::new().with(&bodies[i]).with(&agents[i]).with(&Replaces { old: ids[i] }).submit();
        }
    }
}

schedule!(host(nav_spawn), orders, quake, reorder, host(nav), host(spatial), crowding, churn, landed);
