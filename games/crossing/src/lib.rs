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
//! Game parameter: low 32 bits = unit count (read by the host's map), bits
//! 32..48 = the tick orders are given, bits 48..64 = the churn tick (0: no
//! churn).

use ae_sdk::nav::{Body, Goal, MoveOrder, NavAgent};
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

const NEIGHBOUR_RADII: f32 = 4.0;

fn delay(run: &Run) -> u64 {
    (run.param() >> 32) & 0xFFFF
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
            emit(&MoveOrder { entity: ids[i], x: g.x, y: g.y, group: g.group });
        }
        add(ids[i], &Crowd { neighbours: 0, tick: 0 });
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

schedule!(host(nav_spawn), orders, host(nav), host(spatial), crowding, churn);
