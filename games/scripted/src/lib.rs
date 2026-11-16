//! Scripted: units with an animated model (M14), the engine's animation
//! test game. Every unit wears model [`MODEL`] (the content's, or the one a
//! test gives the `anim` service).
//!
//! Game parameter: the low 32 bits are the unit count.
//!
//! Systems, in schedule order:
//! * `start` (tick 0): the units in a grid, each with a `Walker`, a `Body`
//!   and an `Anim` on [`MODEL`].
//! * `steer`: a unit walks for [`WALK_TICKS`], then stands for
//!   [`STAND_TICKS`] (staggered by entity), turning back each time.
//! * `animate`: the clips and manual joints from what the unit does, with
//!   no script: the standing clip always on layer 0, the walking clip on
//!   layer 1 by the metre while it walks (blended in and out), the head
//!   yawed while it stands (taken from the clips), turned back to the front
//!   and given back to them when it walks.
//! * `host(anim)`: the engine advances the channels.

use ae_sdk::anim::{self, Anim, model_id};
use ae_sdk::nav::Body;
use ae_sdk::{Component, Run, Spawn, schedule, system};

/// The units' model.
pub const MODEL: &str = "fox";
/// Ticks a unit walks, then stands.
pub const WALK_TICKS: u64 = 90;
pub const STAND_TICKS: u64 = 60;
/// Metres a second.
pub const WALK_SPEED: f32 = 1.5;
/// Walking clip seconds per metre.
pub const WALK_PER_METRE: f32 = 0.6;

const DT: f32 = 1.0 / 30.0;

#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Walker {
    /// +1 or -1: along x.
    pub dir: f32,
    pub reserved: u32,
}

fn walking(tick: u64, entity: ae_sdk::Entity) -> bool {
    (tick + entity.index() as u64 * 7) % (WALK_TICKS + STAND_TICKS) < WALK_TICKS
}

#[system()]
fn start(run: &Run) {
    if run.tick() != 0 {
        return;
    }
    let n = run.param() as u32;
    for i in 0..n {
        let (x, y) = ((i % 100) as f32 * 4.0, (i / 100) as f32 * 4.0);
        Spawn::new()
            .with(&Body { x, y, vx: 0.0, vy: 0.0, radius: 1.0 })
            .with(&Walker { dir: if i % 2 == 0 { 1.0 } else { -1.0 }, reserved: 0 })
            .with(&Anim::new(model_id(MODEL)))
            .submit();
    }
}

#[system(write(Body, Walker))]
fn steer(run: &Run) {
    let (bodies, walkers, ids) = (run.write::<Body>(), run.write::<Walker>(), run.entities());
    for i in run.rows() {
        let (b, w) = (&mut bodies[i], &mut walkers[i]);
        if walking(run.tick(), ids[i]) {
            b.vx = w.dir * WALK_SPEED;
        } else {
            if b.vx != 0.0 {
                w.dir = -w.dir;
            }
            b.vx = 0.0;
        }
        b.x += b.vx * DT;
    }
}

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
                a.play(1, walk).per_metre(WALK_PER_METRE).blend_in(200);
            } else if a.target[1] != 1.0 {
                a.layer(1).fade_to(1.0, 200);
            }
            // The head back to the front, then back to the clips.
            if a.is_manual(head) {
                if a.turned(head) {
                    a.release(head);
                } else {
                    a.yaw(head, 0.0, 2.0).ramp(8.0);
                }
            }
        } else {
            if a.playing(1) != 0 && a.target[1] != 0.0 {
                a.stop(1, 300);
            }
            a.yaw(head, 0.7, 2.0).ramp(8.0);
        }
    }
}

schedule!(start, steer, animate, host(anim));
