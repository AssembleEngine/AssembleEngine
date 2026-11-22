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
//! With [`AIM`], units stand instead (no clips: the rest pose), each
//! facing its own way, every third one tilted; a weapon system (`aimer`)
//! asks weapon 0 to point at a target circling the grid, measured from the
//! head (`Anim::bone_world`), every [`AIM_EVERY`] ticks, and posts
//! `AIM`. The script turns the head there (`to_unit`, `dir_in`,
//! `turn_toward`; a new request cuts a turn short) and answers; `check`
//! measures how far the head's forward is from the direction it answered.
//!
//! Game parameter: bits 0..32 the unit count; bit 32 [`CHURN`]: at tick
//! [`CHURN_AT`] every fourth unit is despawned mid-script, and as many new
//! ones spawned the tick after (their entity indices, so their script
//! slots, reused); bit 33 [`AIM`].
//!
//! Systems, in schedule order:
//! * `start`: the units in a grid (tick 0, and the replacements).
//! * `steer`: a unit walks for [`WALK_TICKS`], then stands for
//!   [`STAND_TICKS`] (staggered by entity), turning back each time; it
//!   posts `STARTED_MOVING` / `STOPPED_MOVING`.
//! * `aimer` ([`AIM`]): the aim requests.
//! * `scripts`: `ae_sdk::script::resume`.
//! * `host(anim)`: the engine advances the channels.
//! * `check` ([`AIM`]): answered aims measured.
//! * `churn`: the despawns of [`CHURN`].

use ae_sdk::aim::{Orientation, Weapons, rotation};
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
pub const AIM: u64 = 1 << 33;
/// Ticks between aim requests.
pub const AIM_EVERY: u64 = 6;
/// The aim target: circling the grid's centre at this radius and height
/// (metres), at this rate (radians a tick).
pub const TARGET_RADIUS: f32 = 30.0;
pub const TARGET_HEIGHT: f32 = 3.0;
pub const TARGET_RATE: f32 = 0.01;
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
    /// [`AIM`]: answered requests measured, the last one measured, and the
    /// largest angle (degrees) between the head's forward and an answered
    /// direction.
    pub aims: u32,
    pub checked: u32,
    pub aim_err: f32,
}

impl Log {
    pub const NEW: Log = Log {
        starts: 0,
        stops: 0,
        looks: 0,
        cut: 0,
        backs: 0,
        backs_cut: 0,
        slept: 0,
        turn_min: u32::MAX,
        aims: 0,
        checked: 0,
        aim_err: 0.0,
    };
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

fn spawn(i: u32, aim: bool) {
    let (x, y) = ((i % 100) as f32 * STEP, (i / 100) as f32 * STEP);
    let dir = if i % 2 == 0 { 1.0 } else { -1.0 };
    let orientation = if aim {
        // Its own way round, every third one tilted forward or back.
        let way = deg((i * 37 % 360) as f32);
        let f = Orientation::along(libm::cosf(way), libm::sinf(way));
        match i % 3 {
            0 => f.then(rotation(Axis::RIGHT.v, deg(15.0))),
            1 => f.then(rotation(Axis::FORWARD.v, deg(-10.0))),
            _ => f,
        }
    } else {
        Orientation::along(dir, 0.0)
    };
    Spawn::new()
        .with(&Body {
            x,
            y,
            vx: 0.0,
            vy: 0.0,
            radius: STEP / 4.0,
        })
        .with(&Walker { dir, reserved: 0 })
        .with(&orientation)
        .with(&Weapons::default())
        .with(&Anim::new(model_id(MODEL)))
        .with(&Script::default())
        .with(&Callins::default())
        .with(&Log::NEW)
        .submit();
}

#[system()]
fn start(run: &Run) {
    let n = run.param() as u32;
    let aim = run.param() & AIM != 0;
    match run.tick() {
        0 => (0..n).for_each(|i| spawn(i, aim)),
        t if t == CHURN_AT + 1 && run.param() & CHURN != 0 => (0..n.div_ceil(4)).for_each(|i| spawn(n + i, aim)),
        _ => {}
    }
}

#[system(write(Body, Walker, Orientation, Callins))]
fn steer(run: &Run) {
    if run.param() & AIM != 0 {
        return;
    }
    let (bodies, walkers, orientations, posts, ids) = (
        run.write::<Body>(),
        run.write::<Walker>(),
        run.write::<Orientation>(),
        run.write::<Callins>(),
        run.entities(),
    );
    for i in run.rows() {
        let (b, w) = (&mut bodies[i], &mut walkers[i]);
        let walking = cycle(run.tick(), ids[i]) < WALK_TICKS;
        if walking && b.vx == 0.0 {
            b.vx = w.dir * WALK_SPEED * b.radius;
            orientations[i] = Orientation::along(w.dir, 0.0);
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
    let (Some(survey), Some(walk), Some(head)) = (
        anim::clip(model, "Survey"),
        anim::clip(model, "Walk"),
        anim::bone(model, "b_Head_05"),
    ) else {
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

/// Where the aim target is at `tick`, in the world (nav x, up, nav y),
/// for `units` in the grid.
pub fn target(tick: u64, units: u32) -> [f32; 3] {
    let (cx, cy) = (50.0 * STEP, (units / 100) as f32 * STEP * 0.5);
    let a = tick as f32 * TARGET_RATE;
    [
        cx + TARGET_RADIUS * libm::cosf(a),
        TARGET_HEIGHT,
        cy + TARGET_RADIUS * libm::sinf(a),
    ]
}

#[system(read(Body, Anim, Orientation), write(Weapons, Callins))]
fn aimer(run: &Run) {
    if run.param() & AIM == 0 || run.tick() % AIM_EVERY != 0 {
        return;
    }
    let (bodies, anims, orientations, weapons, posts) = (
        run.read::<Body>(),
        run.read::<Anim>(),
        run.read::<Orientation>(),
        run.write::<Weapons>(),
        run.write::<Callins>(),
    );
    let t = target(run.tick(), run.param() as u32);
    for i in run.rows() {
        // From the bone the script named (the root before its first poll).
        let from = anims[i]
            .bone_world(weapons[i].aim_from[0], &bodies[i], &orientations[i])
            .pos;
        weapons[i].ask(0, [t[0] - from[0], t[1] - from[1], t[2] - from[2]]);
        posts[i].post(callin::AIM);
    }
}

/// [`AIM`]: point the head where weapon 0 is asked, answering each
/// request once its turn is done (a newer request cuts a turn short).
async fn aim(u: Unit, head: u16) {
    // The head aims and "fires" (where the aim is measured from).
    u.update(|w: &mut Weapons| {
        w.aim_from[0] = head;
        w.fire_from[0] = head;
    });
    loop {
        u.callin(callin::AIM).await;
        while u.get::<Weapons>().wants_aim(0) {
            let w = u.get::<Weapons>();
            let (request, dir) = (w.asked[0], w.dir(0));
            let d = u.get::<Anim>().dir_in(head, u.to_unit(dir));
            u.update(|a: &mut Anim| {
                a.turn_toward(head, Axis::FORWARD, d, 3.0).ramp(12.0);
            });
            if let Either::Left(()) = race(u.turned(head), u.callin(callin::AIM)).await {
                u.update(|w: &mut Weapons| w.set_aimed(0, request));
            }
        }
    }
}

/// A unit's script.
async fn unit(u: Unit) {
    let n = names();
    if u.param() & AIM != 0 {
        return aim(u, n.head).await;
    }
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
                match race(
                    join(turn_head(u, n.head, 0.0), u.sleep(300)),
                    u.callin(callin::STOPPED_MOVING),
                )
                .await
                {
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

#[system(read(Body, Orientation), write(Script, Callins, Anim, Weapons, Log))]
fn scripts(run: &Run) {
    script::resume(run, unit);
}

#[system(read(Body, Anim, Orientation, Weapons), write(Log))]
fn check(run: &Run) {
    if run.param() & AIM == 0 {
        return;
    }
    let (bodies, anims, orientations, weapons, logs) = (
        run.read::<Body>(),
        run.read::<Anim>(),
        run.read::<Orientation>(),
        run.read::<Weapons>(),
        run.write::<Log>(),
    );
    let head = names().head;
    for i in run.rows() {
        let (w, l) = (&weapons[i], &mut logs[i]);
        if !w.is_aimed(0) || l.checked == w.aimed[0] {
            continue;
        }
        let f = anims[i]
            .bone_world(head, &bodies[i], &orientations[i])
            .dir(Axis::FORWARD);
        let d = w.dir(0);
        // The angle between them by atan2 (acos is coarse near 0).
        let cos = f[0] * d[0] + f[1] * d[1] + f[2] * d[2];
        let c = [
            f[1] * d[2] - f[2] * d[1],
            f[2] * d[0] - f[0] * d[2],
            f[0] * d[1] - f[1] * d[0],
        ];
        let sin = libm::sqrtf(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
        l.aim_err = l.aim_err.max(libm::atan2f(sin, cos).to_degrees());
        l.aims += 1;
        l.checked = w.aimed[0];
    }
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

schedule!(start, steer, aimer, scripts, host(anim), check, churn);
