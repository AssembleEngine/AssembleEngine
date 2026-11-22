//! The `get` test module (M14 step 6, C): positions advanced by one system,
//! another reads other units' positions through `get(Position)`.
//!
//! Game parameter: bits 0..16 the unit count (a row of 1 m steps).
//!
//! Systems, in schedule order:
//! * `start` (tick 0): the units with a `Position`, a `Link` and an `Out`.
//! * `link` (tick 1, after the spawns applied): each unit points at the
//!   next in its work range.
//! * `advance` (tick 1 on): `x += 1`.
//! * `vanish` (tick [`VANISH`]): despawns the first unit of each range, so a
//!   reader in a later layer finds no entity there.
//! * `read`: for each unit, its target's `x` through `get_of`, or -1.
//!
//! `advance` writes `Position`, which `read` gets: the scheduler puts them in
//! different layers, so `read` sees this tick's writes.

use ae_sdk::{Component, Entity, Run, Spawn, despawn, schedule, system};

/// The tick a target is despawned at.
pub const VANISH: u64 = 5;

#[repr(C)]
#[derive(Clone, Copy, Default, Component)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Component)]
pub struct Link {
    pub target: Entity,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Component)]
pub struct Out {
    pub x: f32,
    pub none: u32,
}

#[system()]
fn start(run: &Run) {
    if run.tick() != 0 {
        return;
    }
    for i in 0..run.param() as u32 {
        Spawn::new()
            .with(&Position { x: i as f32, y: 0.0 })
            .with(&Link::default())
            .with(&Out::default())
            .submit();
    }
}

#[system(write(Link))]
fn link(run: &Run) {
    if run.tick() != 1 {
        return;
    }
    let (links, ids) = (run.write::<Link>(), run.entities());
    for i in run.rows() {
        // The next unit of this range (wrap).
        let n = ids.len();
        links[i].target = ids[(i + 1) % n];
    }
}

#[system(write(Position))]
fn advance(run: &Run) {
    if run.tick() == 0 {
        return;
    }
    let positions = run.write::<Position>();
    for i in run.rows() {
        positions[i].x += 1.0;
    }
}

#[system(write(Position))]
fn vanish(run: &Run) {
    if run.tick() != VANISH {
        return;
    }
    let ids = run.entities();
    if !ids.is_empty() {
        despawn(ids[0]);
    }
}

#[system(read(Link), get(Position), write(Out))]
fn read(run: &Run) {
    let (links, outs) = (run.read::<Link>(), run.write::<Out>());
    for i in run.rows() {
        match run.get_of::<Position>(links[i].target) {
            Some(p) => {
                outs[i].x = p.x;
                outs[i].none = 0;
            }
            None => {
                outs[i].x = -1.0;
                outs[i].none = 1;
            }
        }
    }
}

schedule!(start, link, advance, vanish, read);
