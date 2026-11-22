//! Duel: the M14 step 6 test game (E). Pairs of a scripted Fox (A) and a
//! target Fox (B): A walks, stops and runs while its script takes the aim in
//! steps (an upper-body Survey, the spine turned, then the head), and fires
//! three times with a muzzle flash at the head. Each shot hits B (its head
//! flinches); the third kills it: B's death script drops it on its side,
//! emits smoke and ends, and B is despawned. A stops when its target goes
//! down, and lives on.
//!
//! Game parameter: bits 0..16 the pair count (a 10-wide grid of
//! [`STEP`] metres); bit 16 [`STUCK`]: the death script never ends, to test
//! the SDK's cap; bit 17 [`BACKWARDS`]: A walks backwards, against its
//! `Orientation`; bit 18 [`STANDING`]: A does not move and aims from the
//! rest pose (the exact aim bound). Pair `i`'s timeline starts `i ×
//! [`STAGGER`]` ticks late (pair 0 immediately), so the determinism runs
//! spread their work over ticks.
//!
//! Systems, in schedule order:
//! * `start` (tick 0): the pairs.
//! * `steer`: A moves and posts `STARTED_MOVING` / `STOPPED_MOVING`; it
//!   stops (and posts the stop) once its target is dying.
//! * `asker` (from [`AIM_START`], every [`AIM_EVERY`] ticks, while the
//!   target is not dying): asks weapon 0 to point from A's aim-from bone at
//!   B's chest (through `get(Body)`).
//! * `scripts`: A's aim and fire script (`resume_with_death`; A is never
//!   killed, so its death never runs).
//! * `targets`: B's script: idle, a flinch on each `HIT`, and the death.
//! * `host(anim)`: the engine advances the channels.
//! * `firer` (after the service): when aimed, fires 3 shots
//!   [`SHOT_EVERY`] ticks apart, emitting a `"flash"` at the head, a [`Hit`]
//!   on the target, and posting `FIRED`.
//! * `damage`: each [`Hit`] takes one of B's [`HP`], sets off a small
//!   impact (fire and sparks) at B's chest and posts `HIT`; the last one
//!   kills it (`life::kill`).
//! * `host(spatial)`: the index, without dying units.
//! * `check` (from the tick after the kill, while the target is dying): a
//!   spatial query at the target's place finds A, and not the dying B.
//! * `host(fx)`: the effects sink, last.

use ae_sdk::abi::name_id;
use ae_sdk::aim::{Orientation, Weapons};
use ae_sdk::anim::{self, Anim, Axis, aim_angle, deg, model_id};
use ae_sdk::fx::Effect;
use ae_sdk::life::{self, Dying, Fate};
use ae_sdk::nav::Body;
use ae_sdk::script::{self, Callins, Either, Script, ScriptAlloc, Unit, callin, race};
use ae_sdk::{Component, Entity, Run, Spawn, emit, schedule, spatial_radius, system};

pub const SLOTS: u32 = 1 << 18;
// Two composites (A's and B's), so the general form with their closures.
ae_sdk::script_slots!(SLOTS, |u: Unit| script::with_death(u, unit, death, 0), |u: Unit| {
    script::with_death(u, target, death, 0)
},);

#[global_allocator]
static ALLOC: ScriptAlloc = ScriptAlloc;

/// The units' model.
pub const MODEL: &str = "fox";
pub const STUCK: u64 = 1 << 16;
pub const BACKWARDS: u64 = 1 << 17;
pub const STANDING: u64 = 1 << 18;
/// Grid step between pairs, metres.
pub const STEP: f32 = 24.0;
/// Pairs per grid row.
pub const COLUMNS: u32 = 10;
/// Grid corner, metres.
pub const ORIGIN: f32 = 400.0;
/// B stands this far ahead (+x of A) and this far to A's left (nav -y).
pub const AHEAD: f32 = 12.0;
pub const LEFT: f32 = 6.0;
/// B's chest above the ground, metres.
pub const CHEST: f32 = 1.2;
pub const WALK_SPEED: f32 = 1.5;
pub const RUN_SPEED: f32 = 3.0;
/// Timeline (ticks since a pair's start).
pub const WALK_END: u64 = 45;
pub const RUN_START: u64 = 75;
pub const AIM_START: u64 = 90;
pub const AIM_EVERY: u64 = 3;
/// B's hit points: one a shot, so the last of A's [`SHOTS`] kills it.
pub const HP: u32 = SHOTS;
/// The death script's cap, milliseconds.
pub const CAP_MS: u32 = 8000;
/// Clip seconds per metre at walking and running (times the radius).
pub const WALK_PER_METRE: f32 = 0.6;
pub const RUN_PER_METRE: f32 = 0.6;
pub const SHOTS: u32 = 3;
pub const SHOT_EVERY: u64 = 10;
/// Ticks of stagger per pair.
pub const STAGGER: u64 = 2;
/// The muzzle flash: [`FORWARD`] from the fire-from bone, metres.
pub const FLASH_OFFSET: [f32; 3] = [0.3, 0.0, 0.0];
pub const FLASH: &str = "flash";
pub const SMOKE: &str = "smoke";
pub const FIRE: &str = "fire";
pub const SPARKS: &str = "sparks";
/// Effect scales for a Fox-sized unit (about 2 m long). The catalogue's
/// effects are sized for a small vehicle's explosion at scale 1 (a flash 14
/// to 26 m across, smoke up to 30 m).
/// The muzzle flash: about 0.7 to 1.3 m across.
pub const FLASH_SCALE: f32 = 0.05;
/// The impact on B, at its chest: fire about 0.5 to 1 m, and sparks.
pub const IMPACT_FIRE_SCALE: f32 = 0.1;
pub const IMPACT_SPARKS_SCALE: f32 = 0.15;
/// B's death smoke: about 1 to 4.5 m.
pub const SMOKE_SCALE: f32 = 0.15;

const DT: f32 = 1.0 / 30.0;

/// An actor's target (its entity, found at tick 1 through the spatial
/// index), the chest to aim at, and its place in the stagger.
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Pair {
    pub index: u32,
    pub target: Entity,
    pub chest: [f32; 3],
}

/// A target (B): its pair's index, so an actor can confirm the entity the
/// spatial index found beside it.
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct B {
    pub index: u32,
    pub reserved: u32,
}

/// B's health: [`HP`] at the start, one less a [`Hit`].
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Health {
    pub hp: u32,
    /// Hits taken.
    pub hits: u32,
}

/// A shot that hit `target`, from `shooter`.
#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(event)]
pub struct Hit {
    pub target: Entity,
    pub shooter: Entity,
}

/// The firer's state.
#[repr(C)]
#[derive(Clone, Copy, Default, Component)]
pub struct Gun {
    pub shots: u32,
    pub last: u64,
}

/// What one actor's systems and script did, for the tests.
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Log {
    pub starts: u32,
    pub stops: u32,
    pub aims: u32,
    pub answered: u32,
    pub fired: u32,
    pub flashed: u32,
    /// Shots that killed the target.
    pub kills: u32,
    /// Callins the script took: starts, stops, fires.
    pub handled_starts: u32,
    pub handled_stops: u32,
    pub handled_fired: u32,
    /// Turns of the aim chain that finished (the spine, the head).
    pub turned_spine: u32,
    pub turned_head: u32,
    /// Ticks the firer found the weapon aimed (before the shot limit).
    pub aimed_seen: u32,
    /// Spatial checks at the target's place while it is dying, and of those
    /// that found A and the dying B.
    pub checks: u32,
    pub found_a: u32,
    pub found_b: u32,
}

impl Log {
    pub const NEW: Log = Log {
        starts: 0,
        stops: 0,
        aims: 0,
        answered: 0,
        fired: 0,
        flashed: 0,
        kills: 0,
        handled_starts: 0,
        handled_stops: 0,
        handled_fired: 0,
        turned_spine: 0,
        turned_head: 0,
        aimed_seen: 0,
        checks: 0,
        found_a: 0,
        found_b: 0,
    };
}

/// The tick of a pair's timeline at `tick` (never before 0).
pub fn timeline(tick: u64, pair: u32) -> u64 {
    tick.saturating_sub(pair as u64 * STAGGER)
}

#[system()]
fn start(run: &Run) {
    if run.tick() != 0 {
        return;
    }
    let n = (run.param() & 0xffff) as u32;
    for i in 0..n {
        let (x, y) = (
            ORIGIN + (i % COLUMNS) as f32 * STEP,
            ORIGIN + (i / COLUMNS) as f32 * STEP,
        );
        let (tx, ty) = (x + AHEAD, y - LEFT);
        // A: B's entity is found by `pair` at tick 1 (spawns have no ids
        // until they apply).
        Spawn::new()
            .with(&Body {
                x,
                y,
                vx: 0.0,
                vy: 0.0,
                radius: 1.0,
            })
            // It faces +x; [`BACKWARDS`] moves it the other way.
            .with(&Orientation::along(1.0, 0.0))
            .with(&Anim::new(model_id(MODEL)))
            .with(&Script::default())
            .with(&Callins::default())
            .with(&Weapons::default())
            .with(&Pair {
                index: i,
                target: Entity(0),
                chest: [tx, CHEST, ty],
            })
            .with(&Gun::default())
            .with(&Log::NEW)
            .submit();
        // B: facing A, idling; its script flinches at hits and dies.
        Spawn::new()
            .with(&Body {
                x: tx,
                y: ty,
                vx: 0.0,
                vy: 0.0,
                radius: 1.0,
            })
            .with(&Orientation::along(-1.0, 0.0))
            .with(&Anim::new(model_id(MODEL)))
            .with(&Script::default())
            .with(&Callins::default())
            .with(&Health { hp: HP, hits: 0 })
            .with(&B { index: i, reserved: 0 })
            .submit();
    }
}

/// At tick 1, find each actor's target beside its chest, through the
/// spatial index (`B` confirms the pair), and write it to `Pair`.
#[system(write(Pair), get(B), after(spatial))]
fn pair(run: &Run) {
    if run.tick() != 1 {
        return;
    }
    let pairs = run.write::<Pair>();
    let mut out = [Entity(0); 8];
    for i in run.rows() {
        let p = &mut pairs[i];
        let n = spatial_radius(p.chest[0], p.chest[2], 0.5, &mut out);
        p.target = out[..n]
            .iter()
            .copied()
            .find(|&e| run.get_of::<B>(e).is_some_and(|b| b.index == p.index))
            .unwrap_or(Entity(0));
    }
}

#[system(read(Pair), write(Body, Callins, Log), get(Dying), without(Dying))]
fn steer(run: &Run) {
    let standing = run.param() & STANDING != 0;
    let backwards = run.param() & BACKWARDS != 0;
    let dir = if backwards { -1.0 } else { 1.0 };
    let (pairs, bodies, posts, logs) = (
        run.read::<Pair>(),
        run.write::<Body>(),
        run.write::<Callins>(),
        run.write::<Log>(),
    );
    for i in run.rows() {
        let t = timeline(run.tick(), pairs[i].index);
        let b = &mut bodies[i];
        if standing {
            b.vx = 0.0;
            continue;
        }
        // The target is down: stop for good.
        if run.get_of::<Dying>(pairs[i].target).is_some() {
            if b.vx != 0.0 {
                b.vx = 0.0;
                posts[i].post(callin::STOPPED_MOVING);
                logs[i].stops += 1;
            }
            continue;
        }
        match t {
            1 => {
                b.vx = dir * WALK_SPEED;
                posts[i].post(callin::STARTED_MOVING);
                logs[i].starts += 1;
            }
            WALK_END => {
                b.vx = 0.0;
                posts[i].post(callin::STOPPED_MOVING);
                logs[i].stops += 1;
            }
            RUN_START => {
                b.vx = dir * RUN_SPEED;
                posts[i].post(callin::STARTED_MOVING);
                logs[i].starts += 1;
            }
            _ => {}
        }
        b.x += b.vx * DT;
    }
}

/// Weapon 0's aim requests: from A's aim-from bone to B's chest, while B
/// is not dying.
#[system(
    read(Pair, Body, Anim, Orientation),
    write(Weapons, Callins, Log),
    get(Body, Dying),
    without(Dying)
)]
fn asker(run: &Run) {
    let (pairs, bodies, anims, orientations, weapons, posts, logs) = (
        run.read::<Pair>(),
        run.read::<Body>(),
        run.read::<Anim>(),
        run.read::<Orientation>(),
        run.write::<Weapons>(),
        run.write::<Callins>(),
        run.write::<Log>(),
    );
    let mut i = 0;
    let n = run.len();
    while i < n {
        let t = timeline(run.tick(), pairs[i].index);
        let down = run.get_of::<Dying>(pairs[i].target).is_some();
        if t >= AIM_START && (t - AIM_START) % AIM_EVERY == 0 && !down {
            if let Some(b) = run.get_of::<Body>(pairs[i].target) {
                let from = anims[i]
                    .bone_world(weapons[i].aim_from[0], &bodies[i], &orientations[i])
                    .pos;
                weapons[i].ask(0, [b.x - from[0], CHEST - from[1], b.y - from[2]]);
                posts[i].post(callin::AIM);
                logs[i].aims += 1;
            }
        }
        i += 1;
    }
}

/// A's aim and fire script (never killed: its death does not run).
#[system(
    read(Body, Pair, Orientation),
    write(Script, Callins, Anim, Weapons, Log),
    emits(Effect)
)]
fn scripts(run: &Run) {
    script::resume_with_death(run, unit, death, CAP_MS);
}

/// B's script: idle, flinch at hits, and the death.
#[system(read(Body, B, Orientation), write(Script, Callins, Anim), emits(Effect))]
fn targets(run: &Run) {
    script::resume_with_death(run, target, death, CAP_MS);
}

/// The three shots, once the script has answered: their flashes, and a
/// [`Hit`] on the target each.
#[system(
    read(Pair, Body, Anim, Orientation, Weapons),
    write(Gun, Callins, Log),
    get(Health),
    emits(Effect, Hit),
    without(Dying)
)]
fn firer(run: &Run) {
    let (pairs, bodies, anims, orientations, weapons, guns, posts, logs, ids) = (
        run.read::<Pair>(),
        run.read::<Body>(),
        run.read::<Anim>(),
        run.read::<Orientation>(),
        run.read::<Weapons>(),
        run.write::<Gun>(),
        run.write::<Callins>(),
        run.write::<Log>(),
        run.entities(),
    );
    for i in run.rows() {
        let t = timeline(run.tick(), pairs[i].index);
        let gun = &mut guns[i];
        if t < AIM_START || gun.shots >= SHOTS || !weapons[i].is_aimed(0) {
            continue;
        }
        logs[i].aimed_seen += 1;
        if gun.shots > 0 && t < gun.last + SHOT_EVERY {
            continue;
        }
        let bone = weapons[i].fire_from[0];
        anims[i].effect_at(
            ids[i],
            run.tick(),
            bone,
            FLASH_OFFSET,
            &bodies[i],
            &orientations[i],
            name_id(FLASH),
            2,
            FLASH_SCALE,
        );
        let target = pairs[i].target;
        if let Some(h) = run.get_of::<Health>(target) {
            emit(&Hit {
                target,
                shooter: ids[i],
            });
            // `damage` takes one hit point a shot after this system.
            if h.hp == 1 {
                logs[i].kills += 1;
            }
        }
        posts[i].post(callin::FIRED);
        gun.shots += 1;
        gun.last = t;
        logs[i].fired += 1;
        logs[i].flashed += 1;
    }
}

/// Hits on B: one hit point each, an impact (fire and sparks) at its chest,
/// `HIT` posted; the last kills it.
#[system(
    serial,
    read(B, Body, Anim, Orientation),
    write(Health, Callins),
    get(Dying),
    reads_events(Hit),
    emits(Effect),
    without(Dying)
)]
fn damage(run: &Run) {
    let (bodies, anims, orientations, health, posts, ids) = (
        run.read::<Body>(),
        run.read::<Anim>(),
        run.read::<Orientation>(),
        run.write::<Health>(),
        run.write::<Callins>(),
        run.entities(),
    );
    let mut hits: Vec<u64> = run.events::<Hit>().iter().map(|h| h.target.0).collect();
    hits.sort_unstable();
    for i in run.rows() {
        let e = ids[i].0;
        let from = hits.partition_point(|&h| h < e);
        let n = hits[from..].iter().take_while(|&&h| h == e).count() as u32;
        if n == 0 {
            continue;
        }
        let chest = names().spine2;
        for (effect, count, scale) in [(FIRE, 6, IMPACT_FIRE_SCALE), (SPARKS, 12, IMPACT_SPARKS_SCALE)] {
            anims[i].effect_at(
                ids[i],
                run.tick(),
                chest,
                [0.0; 3],
                &bodies[i],
                &orientations[i],
                name_id(effect),
                count * n,
                scale,
            );
        }
        let h = &mut health[i];
        h.hits += n;
        h.hp = h.hp.saturating_sub(n);
        posts[i].post(callin::HIT);
        if h.hp == 0 {
            life::kill(ids[i], &mut posts[i], run.tick(), 0);
        }
    }
}

/// From the tick after the target's kill, while it is dying: a spatial
/// query at its place finds A, not the target (dying units are excluded
/// from the index).
#[system(read(Pair), write(Log), get(Body, Dying), after(spatial))]
fn check(run: &Run) {
    let (pairs, logs, ids) = (run.read::<Pair>(), run.write::<Log>(), run.entities());
    // Neighbouring pairs (STEP apart) are in range too; the import returns
    // every hit and writes at most the buffer's length.
    let mut out = [Entity(0); 64];
    for i in run.rows() {
        let target = pairs[i].target;
        let (Some(d), Some(b)) = (run.get_of::<Dying>(target), run.get_of::<Body>(target)) else {
            continue;
        };
        // `Dying` is added at `damage`'s sync point, and the index may be
        // built in that layer: it leaves the index from the next tick.
        if d.killed_at >= run.tick() {
            continue;
        }
        let total = spatial_radius(b.x, b.y, 40.0, &mut out);
        assert!(total <= out.len(), "{total} spatial hits, more than the buffer holds");
        let hits = &out[..total];
        logs[i].checks += 1;
        if hits.contains(&ids[i]) {
            logs[i].found_a += 1;
        }
        if hits.contains(&pairs[i].target) {
            logs[i].found_b += 1;
        }
    }
}

/// A's script: the aim sequence (an upper-body clip, the spine, the head),
/// the flash's kick, or the death.
async fn unit(u: Unit) {
    let n = names();
    // Aimed from the body (a stable origin), fired from the head (the
    // muzzle): the head's forward follows the spine-to-target line.
    u.update(|w: &mut Weapons| {
        w.aim_from[0] = n.spine2;
        w.fire_from[0] = n.head;
    });
    // Standing: the rest pose (the exact aim bound); otherwise Survey, and
    // the first aim takes the rifle off the back with the upper clip.
    let standing = u.param() & STANDING != 0;
    if !standing {
        u.update(|a: &mut Anim| {
            a.play(0, n.survey);
        });
    }
    let mut rifle = standing;
    let mut starts = 0;
    let mut next = 0;
    loop {
        let c = match core::mem::take(&mut next) {
            0 => {
                u.callin(callin::STARTED_MOVING | callin::STOPPED_MOVING | callin::AIM | callin::FIRED)
                    .await
            }
            c => c,
        };
        if c & callin::STARTED_MOVING != 0 {
            // Walk first, then Run.
            starts += 1;
            let (clip, rate) = if starts == 1 {
                (n.walk, WALK_PER_METRE)
            } else {
                (n.run, RUN_PER_METRE)
            };
            u.update(|a: &mut Anim| {
                a.play(0, clip).per_metre(rate).blend_in(200);
            });
            u.update(|l: &mut Log| l.handled_starts += 1);
        }
        if c & callin::STOPPED_MOVING != 0 {
            u.update(|a: &mut Anim| {
                a.play(0, n.survey).blend_in(300);
            });
            u.update(|l: &mut Log| l.handled_stops += 1);
        }
        if c & callin::FIRED != 0 {
            // The kick: the head up and back, then the aim again.
            u.update(|a: &mut Anim| {
                a.turn_by(n.head, Axis::RIGHT, deg(-6.0), deg(720.0)).ramp(deg(3600.0));
            });
            u.update(|l: &mut Log| l.handled_fired += 1);
        }
        while u.get::<Weapons>().wants_aim(0) {
            if !rifle {
                u.update(|a: &mut Anim| {
                    a.play(1, n.survey).mask(n.upper).blend_in(250);
                });
                rifle = true;
                // A newer request cuts the sleep short; the newest direction
                // is aimed below all the same.
                if let Either::Right(_) = race(u.sleep(250), u.callin(callin::AIM)).await {
                    // The request came; fall through to the turns.
                }
            }
            // The spine about up, then the neck, then the head. A newer
            // request retargets the turn in progress (from where it is)
            // rather than starting the sequence again.
            loop {
                let d = u.to_unit(u.get::<Weapons>().dir(0));
                let yaw = aim_angle(Axis::UP, Axis::FORWARD, u.get::<Anim>().dir_in(n.spine2, d));
                u.update(|a: &mut Anim| {
                    a.turn(n.spine2, Axis::UP, yaw, deg(180.0)).ramp(deg(720.0));
                });
                if let Either::Left(()) = race(u.turned(n.spine2), u.callin(callin::AIM)).await {
                    break;
                }
            }
            u.update(|l: &mut Log| l.turned_spine += 1);
            // The neck, so no clipped parent is left to move the head after
            // it is aimed.
            loop {
                let d = u.to_unit(u.get::<Weapons>().dir(0));
                let d = u.get::<Anim>().dir_in(n.neck, d);
                u.update(|a: &mut Anim| {
                    a.turn_toward(n.neck, Axis::FORWARD, d, deg(720.0)).ramp(deg(3600.0));
                });
                if let Either::Left(()) = race(u.turned(n.neck), u.callin(callin::AIM)).await {
                    break;
                }
            }
            // The head, the shortest way (one free joint).
            loop {
                let d = u.to_unit(u.get::<Weapons>().dir(0));
                let d = u.get::<Anim>().dir_in(n.head, d);
                u.update(|a: &mut Anim| {
                    a.turn_toward(n.head, Axis::FORWARD, d, deg(720.0)).ramp(deg(3600.0));
                });
                if let Either::Left(()) = race(u.turned(n.head), u.callin(callin::AIM)).await {
                    break;
                }
            }
            u.update(|l: &mut Log| l.turned_head += 1);
            // Answer the newest request: the head points along it to within
            // the tick's target motion.
            u.update(|w: &mut Weapons| w.set_aimed(0, w.asked[0]));
            u.update(|l: &mut Log| l.answered += 1);
        }
    }
}

/// B's life: Survey looping; each `HIT` flinches the head up and back.
async fn target(u: Unit) {
    let n = names();
    u.update(|a: &mut Anim| {
        a.play(0, n.survey);
    });
    loop {
        u.callin(callin::HIT).await;
        u.update(|a: &mut Anim| {
            a.turn_by(n.head, Axis::RIGHT, deg(-15.0), deg(720.0)).ramp(deg(3600.0));
        });
        u.turned(n.head).await;
        u.update(|a: &mut Anim| {
            a.turn_by(n.head, Axis::RIGHT, deg(15.0), deg(240.0)).ramp(deg(1200.0));
        });
        u.turned(n.head).await;
        u.update(|a: &mut Anim| a.release(n.head));
    }
}

/// The death (B's; A is never killed): stop the upper body, play Survey
/// once, fall on its side, smoke where the spine lies, then wait;
/// `Fate::Remove` despawns it. With [`STUCK`], it never returns, and the
/// SDK's cap ends it.
async fn death(u: Unit) -> Fate {
    let n = names();
    u.update(|a: &mut Anim| a.stop(1, 200));
    u.update(|a: &mut Anim| {
        a.play(0, n.survey).once();
    });
    u.update(|a: &mut Anim| {
        a.turn(n.root, Axis::FORWARD, deg(90.0), deg(360.0)).ramp(deg(720.0));
    });
    u.turned(n.root).await;
    u.effect_at(n.spine, [0.0, 0.2, 0.0], name_id(SMOKE), 8, SMOKE_SCALE);
    u.ended(0).await;
    u.sleep(1500).await;
    if u.param() & STUCK != 0 {
        // The cap (8 s, [`CAP_MS`]) cuts this.
        u.sleep(60_000).await;
    }
    Fate::Remove
}

/// The model's names the scripts use.
#[derive(Clone, Copy)]
struct Names {
    survey: u16,
    walk: u16,
    run: u16,
    upper: u16,
    head: u16,
    neck: u16,
    spine2: u16,
    spine: u16,
    root: u16,
}

fn names() -> Names {
    let model = model_id(MODEL);
    let (
        Some(survey),
        Some(walk),
        Some(run),
        Some(upper),
        Some(head),
        Some(neck),
        Some(spine2),
        Some(spine),
        Some(root),
    ) = (
        anim::clip(model, "Survey"),
        anim::clip(model, "Walk"),
        anim::clip(model, "Run"),
        anim::mask(model, "upper"),
        anim::bone(model, "b_Head_05"),
        anim::bone(model, "b_Neck_04"),
        anim::bone(model, "b_Spine02_03"),
        anim::bone(model, "b_Spine01_02"),
        anim::bone(model, "b_Root_00"),
    )
    else {
        panic!("the {MODEL} model or rig file lacks a clip, mask or bone the script uses");
    };
    Names {
        survey,
        walk,
        run,
        upper,
        head,
        neck,
        spine2,
        spine,
        root,
    }
}

schedule!(
    start,
    steer,
    asker,
    scripts,
    targets,
    host(anim),
    firer,
    damage,
    host(spatial),
    pair,
    check,
    host(fx)
);
