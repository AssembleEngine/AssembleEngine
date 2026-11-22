//! A module `get` must reject: a system gets a component it writes. The
//! schema parser rejects it when the module loads.

use ae_sdk::{Component, Run, schedule, system};

#[repr(C)]
#[derive(Clone, Copy, Default, Component)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

#[system(write(Position), get(Position))]
fn bad(run: &Run) {
    let positions = run.write::<Position>();
    for i in run.rows() {
        positions[i].x += 1.0;
    }
}

schedule!(bad);
