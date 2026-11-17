//! Scripted: units with an animated model (M14), the engine's animation
//! test game. Every unit wears model [`MODEL`] (the content's, or the one a
//! test gives the `anim` service).
//!
//! Game parameter: bits 0..32 the unit count; 32..40 the grid step in
//! metres (0: 4), units of radius a quarter of it, walking at
//! [`WALK_SPEED`] per metre of radius; 40..52 and 52..64 the grid's corner
//! (x, y) in 16 m (the client's `-fox` benches put it under a camera).
//!
//! Systems, in schedule order:
//! * `start` (tick 0): the units in a grid, each with a `Walker`, a `Body`
//!   and an `Anim` on [`MODEL`].
//! * `steer`: a unit walks for [`WALK_TICKS`], then stands for
//!   [`STAND_TICKS`] (staggered by entity), turning back each time.
//! * `animate`: the clips and manual joints from what the unit does, with
//!   no script: the standing clip always on layer 0, the walking clip on
//!   layer 1 by the metre while it walks (blended in and out), the head
//!   turned to look to the front left while it stands (taken from the
//!   clips; `dir_in` and `turn_toward`), turned back to the front
//!   and given back to them when it walks.
//! * `host(anim)`: the engine advances the channels.
//! * `measure`: a unit's `Bounds` (made with it from its first pose)
//!   measured again once a cycle, [`MEASURE_TICK`] ticks into it (the head
//!   turned while it stands).

use ae_sdk::anim::{self, Anim, Axis, Bounds, model_id};
use ae_sdk::nav::Body;
use ae_sdk::{Component, Run, Spawn, schedule, system};

/// The units' model.
pub const MODEL: &str = "fox";
/// Ticks a unit walks, then stands.
pub const WALK_TICKS: u64 = 90;
pub const STAND_TICKS: u64 = 60;
/// Metres a second, per metre of radius.
pub const WALK_SPEED: f32 = 1.5;
/// Where in its cycle a unit's bounds are measured.
pub const MEASURE_TICK: u64 = WALK_TICKS + 30;
/// Walking clip seconds per metre, times the radius.
pub const WALK_PER_METRE: f32 = 0.6;

const DT: f32 = 1.0 / 30.0;

#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Walker {
    /// +1 or -1: along x.
    pub dir: f32,
    pub reserved: u32,
}

/// Where unit `entity` is in its walk-stand cycle at `tick`.
pub fn cycle(tick: u64, entity: ae_sdk::Entity) -> u64 {
    (tick + entity.index() as u64 * 7) % (WALK_TICKS + STAND_TICKS)
}

fn walking(tick: u64, entity: ae_sdk::Entity) -> bool {
    cycle(tick, entity) < WALK_TICKS
}

#[system()]
fn start(run: &Run) {
    if run.tick() != 0 {
        return;
    }
    let p = run.param();
    let n = p as u32;
    let step = match (p >> 32) & 0xff {
        0 => 4.0,
        s => s as f32,
    };
    let (x0, y0) = (((p >> 40) & 0xfff) as f32 * 16.0, ((p >> 52) & 0xfff) as f32 * 16.0);
    let anim = Anim::new(model_id(MODEL));
    let bounds = anim.bounds();
    for i in 0..n {
        let (x, y) = (x0 + (i % 100) as f32 * step, y0 + (i / 100) as f32 * step);
        Spawn::new()
            .with(&Body { x, y, vx: 0.0, vy: 0.0, radius: step / 4.0 })
            .with(&Walker { dir: if i % 2 == 0 { 1.0 } else { -1.0 }, reserved: 0 })
            .with(&anim)
            .with(&bounds)
            .submit();
    }
}

#[system(write(Body, Walker))]
fn steer(run: &Run) {
    let (bodies, walkers, ids) = (run.write::<Body>(), run.write::<Walker>(), run.entities());
    for i in run.rows() {
        let (b, w) = (&mut bodies[i], &mut walkers[i]);
        if walking(run.tick(), ids[i]) {
            b.vx = w.dir * WALK_SPEED * b.radius;
        } else {
            if b.vx != 0.0 {
                w.dir = -w.dir;
            }
            b.vx = 0.0;
        }
        b.x += b.vx * DT;
    }
}

/// Where a standing unit looks, in its frame (forward and to the left).
const LOOK: [f32; 3] = [0.6, 0.0, -0.8];

#[system(read(Body), write(Anim))]
fn animate(run: &Run) {
    let (bodies, anims) = (run.read::<Body>(), run.write::<Anim>());
    let model = model_id(MODEL);
    let (Some(survey), Some(walk), Some(head)) = (anim::clip(model, "Survey"), anim::clip(model, "Walk"), anim::bone(model, "b_Head_05")) else {
        return;
    };
    for i in run.rows() {
        let a = &mut anims[i];
        if a.playing(0) != survey {
            a.play(0, survey);
        }
        if bodies[i].vx != 0.0 {
            if a.playing(1) != walk {
                a.play(1, walk).per_metre(WALK_PER_METRE / bodies[i].radius).blend_in(200);
            } else if a.target[1] != 1.0 {
                a.layer(1).fade_to(1.0, 200);
            }
            // The head back to the front, then back to the clips.
            if a.is_manual(head) {
                if a.turned(head) {
                    a.release(head);
                } else {
                    a.turn(head, Axis::UP, 0.0, 2.0).ramp(8.0);
                }
            }
        } else {
            if a.playing(1) != 0 && a.target[1] != 0.0 {
                a.stop(1, 300);
            }
            // Look once to the front left: the direction as the head sees
            // it (its parents where the clips have them), then the
            // shortest turn there.
            if !a.is_manual(head) {
                let d = a.dir_in(head, LOOK);
                a.turn_toward(head, Axis::FORWARD, d, 2.0).ramp(8.0);
            }
        }
    }
}

#[system(read(Anim), write(Bounds))]
fn measure(run: &Run) {
    let (anims, bounds, ids) = (run.read::<Anim>(), run.write::<Bounds>(), run.entities());
    for i in run.rows() {
        if cycle(run.tick(), ids[i]) == MEASURE_TICK {
            anims[i].update_bounds(&mut bounds[i]);
        }
    }
}

schedule!(start, steer, animate, host(anim), measure);
