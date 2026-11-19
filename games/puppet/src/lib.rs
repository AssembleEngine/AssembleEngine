//! Puppet: units with an animated model run by a script (M14 step 5), the
//! script API's test game. Every unit wears model [`MODEL`] and walks and
//! stands as the `scripted` game's do, but what it does with its model is
//! its script's ([`unit`]): no system picks clips.
//!
//! The script waits for callins, which the system moving the unit posts
//! when it starts and stops. When it stops, it looks left and right,
//! pausing a while at each side ([`pause_ms`]), unless it walks again
//! first (a race: the look is dropped where it waits); when it walks, it
//! turns its head back and gives it back to the clips, unless it stops
//! first. What happened goes to its `Log`.
//!
//! Game parameter: bits 0..32 the unit count; bit 32 [`CHURN`]: at tick
//! [`CHURN_AT`] every fourth unit is despawned mid-script, and as many new
//! ones spawned the tick after (their entity indices, so their script
//! slots, reused).
//!
//! Systems, in schedule order:
//! * `start`: the units in a grid (tick 0, and the replacements).
//! * `steer`: a unit walks for [`WALK_TICKS`], then stands for
//!   [`STAND_TICKS`] (staggered by entity), turning back each time; it
//!   posts `STARTED_MOVING` / `STOPPED_MOVING`.
//! * `scripts`: `ae_sdk::script::resume`.
//! * `host(anim)`: the engine advances the channels.
//! * `churn`: the despawns of [`CHURN`].

use ae_sdk::anim::{self, Anim, Axis, deg, model_id};
use ae_sdk::nav::Body;
use ae_sdk::script::{self, Callins, Either, Script, ScriptAlloc, Unit, callin, join, race};
use ae_sdk::{Component, Run, Spawn, schedule, system};

pub const SLOTS: u32 = 16384;
ae_sdk::script_slots!(SLOTS, unit);

#[global_allocator]
static ALLOC: ScriptAlloc = ScriptAlloc;

/// The units' model.
pub const MODEL: &str = "fox";
pub const WALK_TICKS: u64 = 90;
pub const STAND_TICKS: u64 = 60;
pub const WALK_SPEED: f32 = 1.5;
pub const WALK_PER_METRE: f32 = 0.6;
pub const CHURN: u64 = 1 << 32;
pub const CHURN_AT: u64 = 100;
/// Grid step, metres.
pub const STEP: f32 = 4.0;
/// How far a standing unit looks to each side.
pub const LOOK_DEG: f32 = 50.0;

const DT: f32 = 1.0 / 30.0;

/// What a unit's script did.
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Log {
    /// Callins handled.
    pub starts: u32,
    pub stops: u32,
    /// Looks around finished, and cut short by walking.
    pub looks: u32,
    pub cut: u32,
    /// Head turns back finished, and cut short by stopping.
    pub backs: u32,
    pub backs_cut: u32,
    /// Ticks of the last pause slept ([`pause_ms`]; 0: none yet).
    pub slept: u32,
    /// The fewest ticks a head turn was waited for (`u32::MAX`: none).
    pub turn_min: u32,
}

impl Log {
    pub const NEW: Log = Log { starts: 0, stops: 0, looks: 0, cut: 0, backs: 0, backs_cut: 0, slept: 0, turn_min: u32::MAX };
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Walker {
    pub dir: f32,
    pub reserved: u32,
}

/// Where unit `entity` is in its walk-stand cycle at `tick`.
pub fn cycle(tick: u64, entity: ae_sdk::Entity) -> u64 {
    (tick + entity.index() as u64 * 7) % (WALK_TICKS + STAND_TICKS)
}

/// The pause at each side of a look, by entity index: 100 to 1150 ms, so
/// some looks fit a stand (2 s) and some do not.
pub fn pause_ms(index: u32) -> u32 {
    100 + (index % 8) * 150
}

fn spawn(i: u32) {
    let (x, y) = ((i % 100) as f32 * STEP, (i / 100) as f32 * STEP);
    Spawn::new()
        .with(&Body { x, y, vx: 0.0, vy: 0.0, radius: STEP / 4.0 })
        .with(&Walker { dir: if i % 2 == 0 { 1.0 } else { -1.0 }, reserved: 0 })
        .with(&Anim::new(model_id(MODEL)))
        .with(&Script::default())
        .with(&Callins::default())
        .with(&Log::NEW)
        .submit();
}

#[system()]
fn start(run: &Run) {
    let n = run.param() as u32;
    match run.tick() {
        0 => (0..n).for_each(spawn),
        t if t == CHURN_AT + 1 && run.param() & CHURN != 0 => (0..n.div_ceil(4)).for_each(|i| spawn(n + i)),
        _ => {}
    }
}

#[system(write(Body, Walker, Callins))]
fn steer(run: &Run) {
    let (bodies, walkers, posts, ids) = (run.write::<Body>(), run.write::<Walker>(), run.write::<Callins>(), run.entities());
    for i in run.rows() {
        let (b, w) = (&mut bodies[i], &mut walkers[i]);
        let walking = cycle(run.tick(), ids[i]) < WALK_TICKS;
        if walking && b.vx == 0.0 {
            b.vx = w.dir * WALK_SPEED * b.radius;
            posts[i].post(callin::STARTED_MOVING);
        } else if !walking && b.vx != 0.0 {
            w.dir = -w.dir;
            b.vx = 0.0;
            posts[i].post(callin::STOPPED_MOVING);
        }
        b.x += b.vx * DT;
    }
}

/// The model's names the script uses.
#[derive(Clone, Copy)]
struct Names {
    survey: u16,
    walk: u16,
    head: u16,
}

fn names() -> Names {
    let model = model_id(MODEL);
    let (Some(survey), Some(walk), Some(head)) = (anim::clip(model, "Survey"), anim::clip(model, "Walk"), anim::bone(model, "b_Head_05")) else {
        panic!("the {MODEL} model lacks a clip or bone the script uses");
    };
    Names { survey, walk, head }
}

/// Turn the head to `angle` and wait for it, noting how long that took.
async fn turn_head(u: Unit, head: u16, angle: f32) {
    u.update(|a: &mut Anim| {
        a.turn(head, Axis::UP, angle, 4.0).ramp(16.0);
    });
    let t0 = u.tick();
    u.turned(head).await;
    let took = (u.tick() - t0) as u32;
    u.update(|l: &mut Log| l.turn_min = l.turn_min.min(took));
}

/// Look left, pause, look right, pause.
async fn look_around(u: Unit, head: u16) {
    let pause = pause_ms(u.entity().index());
    for side in [-1.0, 1.0] {
        turn_head(u, head, side * deg(LOOK_DEG)).await;
        let t0 = u.tick();
        u.sleep(pause).await;
        let slept = (u.tick() - t0) as u32;
        u.update(|l: &mut Log| l.slept = slept);
    }
}

/// A unit's script.
async fn unit(u: Unit) {
    let n = names();
    u.update(|a: &mut Anim| {
        a.play(0, n.survey);
    });
    // A callin a race took, still to handle.
    let mut next = 0;
    loop {
        let c = match core::mem::take(&mut next) {
            0 => u.callin(callin::STARTED_MOVING | callin::STOPPED_MOVING).await,
            c => c,
        };
        if c & callin::STARTED_MOVING != 0 {
            u.update(|l: &mut Log| l.starts += 1);
            let radius = u.get::<Body>().radius;
            u.update(|a: &mut Anim| {
                a.play(1, n.walk).per_metre(WALK_PER_METRE / radius).blend_in(200);
            });
            if u.get::<Anim>().is_manual(n.head) {
                // The head back to the front (and at least a short while,
                // so it is not given back the tick it arrives), then back
                // to the clips; stopping first leaves it turned.
                match race(join(turn_head(u, n.head, 0.0), u.sleep(300)), u.callin(callin::STOPPED_MOVING)).await {
                    Either::Left(_) => u.update(|a: &mut Anim| {
                        a.release(n.head);
                        u.update(|l: &mut Log| l.backs += 1);
                    }),
                    Either::Right(k) => {
                        u.update(|l: &mut Log| l.backs_cut += 1);
                        next = k;
                    }
                }
            }
        } else if c & callin::STOPPED_MOVING != 0 {
            u.update(|l: &mut Log| l.stops += 1);
            u.update(|a: &mut Anim| a.stop(1, 300));
            match race(look_around(u, n.head), u.callin(callin::STARTED_MOVING)).await {
                Either::Left(()) => u.update(|l: &mut Log| l.looks += 1),
                Either::Right(k) => {
                    u.update(|l: &mut Log| l.cut += 1);
                    next = k;
                }
            }
        }
    }
}

#[system(read(Body), write(Script, Callins, Anim, Log))]
fn scripts(run: &Run) {
    script::resume(run, unit);
}

#[system(read(Body))]
fn churn(run: &Run) {
    if run.tick() != CHURN_AT || run.param() & CHURN == 0 {
        return;
    }
    for &e in run.entities() {
        if e.index() % 4 == 0 {
            ae_sdk::despawn(e);
        }
    }
}

schedule!(start, steer, scripts, host(anim), churn);
