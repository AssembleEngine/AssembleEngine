//! The effects-at-joints test module (M14 step 6, B): units in a grid, each
//! turned its own way (every third tilted), emitting two effects at bones
//! every tick after the animation service has advanced them.
//!
//! Game parameter: bits 0..16 the unit count (a grid of 4 m steps); 16..24
//! the grid's corner x in 16 m; 24..32 the corner y.
//!
//! Systems, in schedule order:
//! * `start` (tick 0): the units with a `Body`, an `Anim` on the Fox and a
//!   mixed `Orientation`.
//! * `host(anim)`: the engine advances the channels.
//! * `emit`: a `"flash"` 0.3 m along `FORWARD` from the head, then a
//!   `"smoke"` 0.2 m up from the spine, both `Effect` events.
//! * `host(fx)`: the engine drains them (last, as designed).

use ae_sdk::abi::name_id;
use ae_sdk::aim::{Orientation, rotation};
use ae_sdk::anim::{self, Anim, Axis, deg, model_id};
use ae_sdk::fx::Effect;
use ae_sdk::nav::Body;
use ae_sdk::{Run, Spawn, schedule, system};

/// The units' model.
pub const MODEL: &str = "fox";
/// The effects' names (the client maps them to the catalogue's indices).
pub const FLASH: &str = "flash";
pub const SMOKE: &str = "smoke";
/// The bones they start at, by name (the animation service resolves them).
pub const HEAD: &str = "b_Head_05";
pub const SPINE: &str = "b_Spine01_02";

#[system()]
fn start(run: &Run) {
    if run.tick() != 0 {
        return;
    }
    let p = run.param();
    let n = (p & 0xffff) as u32;
    let (x0, y0) = (((p >> 16) & 0xff) as f32 * 16.0, ((p >> 24) & 0xff) as f32 * 16.0);
    let (step, radius) = (4.0f32, 1.0f32);
    for i in 0..n {
        let (x, y) = (x0 + (i % 16) as f32 * step, y0 + (i / 16) as f32 * step);
        let way = deg((i * 37 % 360) as f32);
        let facing = Orientation::along(libm::cosf(way), libm::sinf(way));
        let facing = match i % 3 {
            0 => facing.then(rotation(Axis::RIGHT.v, deg(12.0))),
            1 => facing.then(rotation(Axis::FORWARD.v, deg(-9.0))),
            _ => facing,
        };
        Spawn::new()
            .with(&Body {
                x,
                y,
                vx: 0.0,
                vy: 0.0,
                radius,
            })
            .with(&Anim::new(model_id(MODEL)))
            .with(&facing)
            .submit();
    }
}

#[system(read(Body, Anim, Orientation), emits(Effect))]
fn emit(run: &Run) {
    let (bodies, anims, orientations, ids) = (
        run.read::<Body>(),
        run.read::<Anim>(),
        run.read::<Orientation>(),
        run.entities(),
    );
    let model = model_id(MODEL);
    let (head, spine) = (anim::bone(model, HEAD), anim::bone(model, SPINE));
    let (Some(head), Some(spine)) = (head, spine) else {
        return;
    };
    let (flash, smoke) = (name_id(FLASH), name_id(SMOKE));
    let forward = Axis::FORWARD.v.map(|v| 0.3 * v);
    for i in run.rows() {
        let (a, b, o) = (&anims[i], &bodies[i], &orientations[i]);
        a.effect_at(ids[i], run.tick(), head, forward, b, o, flash, 2, 1.0);
        a.effect_at(ids[i], run.tick(), spine, [0.0, 0.2, 0.0], b, o, smoke, 1, 0.5);
    }
}

schedule!(start, host(anim), emit, host(fx));
