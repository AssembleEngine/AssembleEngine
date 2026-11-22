//! The duel game (M14 step 6, E/F): pairs of a scripted Fox (A) and a
//! target Fox (B). A walks, stops and runs while its script takes the aim
//! in steps, fires three flashes at its head, and is killed: the death
//! script drops it, smokes, and it is despawned. `STUCK` tests the SDK's
//! cap, `BACKWARDS` draws A against its velocity, `STANDING` aims from the
//! rest pose.
//!
//! Checked at 8, 1, 2 and 4 workers (the world and everything the systems
//! did the same):
//! * every callin handled: two starts and one stop a moving unit, none when
//!   standing; every `FIRED` kicked the head;
//! * each answered aim points along the line from the aim-from bone to B's
//!   chest, measured independently in glam (standing: exact; running: the
//!   measured bound);
//! * three flashes an actor, each at the head's `bone_world` plus the
//!   offset of its tick, with A facing where its `Orientation` says;
//! * A is not in the spatial results from the tick after the kill, and B
//!   is; A is despawned at its death's end, and with `STUCK` exactly at
//!   `killed_at` + the cap;
//! * the digests, logs, effects and aim errors are equal.
//!
//! One runtime at a time: the column arena is process-wide.

use ae_model::anim::{Rig, qrot};
use ae_sdk::abi::name_id;
use ae_sdk::aim::Orientation;
use ae_sdk::anim::Anim;
use ae_sdk::fx::Effect;
use ae_sdk::life::Dying;
use ae_sdk::nav::Body;
use ae_services::anim::{AnimPlugin, channels, place};
use ae_services::fx::{FxFeed, FxPlugin};
use ae_services::spatial::SpatialPlugin;
use ae_sim::runtime::{Runtime, RuntimeConfig, World};
use anyhow::{Result, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

const HZ: u32 = 30;
/// Pairs and ticks of the determinism run (the last pair starts 398 ticks
/// late and dies 96 or so later).
const PAIRS: u64 = 200;
const TICKS: u64 = 760;
/// Pairs of one variant run.
const VARIANT_PAIRS: u64 = 8;
/// Long enough for the last staggered `STUCK` pair to reach its cap.
const VARIANT_TICKS: u64 = 460;
/// The duel game's cap, ticks (`CAP_MS`, 8 s at 30 Hz): a normal death (about
/// 4.9 s: the fall, Survey once, 1.5 s still) ends well inside it.
const CAP: u64 = 240;
/// `standings`, `backwards`, `stuck`: the duel game's parameter bits.
const STUCK: u64 = 1 << 16;
const BACKWARDS: u64 = 1 << 17;
const STANDING: u64 = 1 << 18;
/// Measured on the running variant (the clips move the aim chain's parents
/// between the solve and the answer); the standing one is exact.
const RUNNING_AIM_BOUND: f32 = 8.0;
const STANDING_AIM_BOUND: f32 = 1e-3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
struct Pair {
    index: u32,
    target: u64,
    chest: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Log {
    starts: u32,
    stops: u32,
    aims: u32,
    answered: u32,
    fired: u32,
    flashed: u32,
    kills: u32,
    handled_starts: u32,
    handled_stops: u32,
    handled_fired: u32,
    turned_spine: u32,
    turned_head: u32,
    aimed_seen: u32,
    checks: u32,
    found_a: u32,
    found_b: u32,
}

fn fox() -> Rig {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/khronos/Fox/Fox.glb");
    let mut rig = Rig::new(&ae_gltf::convert_file(&path).unwrap().model);
    rig.apply_rig_file(&std::fs::read(path.with_extension("rig.json")).unwrap())
        .unwrap();
    rig
}

fn decode<T: Copy>(bytes: &[u8]) -> T {
    assert_eq!(bytes.len(), size_of::<T>());
    unsafe { bytes.as_ptr().cast::<T>().read_unaligned() }
}

fn rows<T: Copy>(rt: &Runtime, world: &World, component: &str) -> Result<BTreeMap<u64, T>> {
    Ok(rt
        .rows(world, component)?
        .into_iter()
        .map(|(e, b)| (e, decode(&b)))
        .collect())
}

/// How far the head points from the line from the aim-from bone to its
/// target, degrees.
fn aim_error(rig: &Rig, head: u16, from: u16, anim: &Anim, body: &Body, o: &Orientation, chest: [f32; 3]) -> f32 {
    let (p, q) = rig.bone_frame(&channels(anim), from);
    let (origin, _) = place(o.rot, [body.x, body.y, body.radius / rig.footprint()], p, q);
    let (p, q) = rig.bone_frame(&channels(anim), head);
    let (_, rot) = place(o.rot, [body.x, body.y, body.radius / rig.footprint()], p, q);
    let f = qrot(rot, [1.0, 0.0, 0.0]);
    let d = [chest[0] - origin[0], chest[1] - origin[1], chest[2] - origin[2]];
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let c = ((f[0] * d[0] + f[1] * d[1] + f[2] * d[2]) / n).clamp(-1.0, 1.0);
    c.acos().to_degrees()
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Kind {
    role: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Counters {
    deaths: u32,
    polls: u32,
}

/// An actor's log when it was last seen, the tick it vanished at, and the
/// tick it was killed at.
#[derive(Clone, Debug, PartialEq)]
struct Actor {
    log: Log,
    gone: Option<u64>,
    killed: Option<u64>,
}

#[derive(Default)]
struct Outcome {
    digest: u128,
    actors: BTreeMap<u64, Actor>,
    /// The flash effects of the whole run, in tick then key order.
    effects: Vec<Effect>,
    /// Per actor, the largest aim error of an answered request (degrees).
    aim_errors: BTreeMap<u64, f32>,
    answered: u64,
}

fn run(rt: &mut Runtime, feed: &FxFeed, workers: usize, param: u64, ticks: u64) -> Result<Outcome> {
    let rig = fox();
    let (head, aim_from, flash) = (
        rig.bone("b_Head_05").unwrap(),
        rig.bone("b_Spine02_03").unwrap(),
        name_id("flash"),
    );
    let mut world = rt.world_for(param, workers)?;
    let mut out = Outcome::default();
    let mut previous: BTreeSet<u64> = BTreeSet::new();
    let mut answered: BTreeMap<u64, u32> = BTreeMap::new();
    let mut kills: BTreeMap<u64, u32> = BTreeMap::new();
    for _ in 0..ticks {
        rt.tick(&mut world, workers)?;
        let tick = world.tick;
        let pairs: BTreeMap<u64, Pair> = rows(rt, &world, "Pair")?;
        let bodies: BTreeMap<u64, Body> = rows(rt, &world, "Body")?;
        let anims: BTreeMap<u64, Anim> = rows(rt, &world, "Anim")?;
        let orientations: BTreeMap<u64, Orientation> = rows(rt, &world, "Orientation")?;
        let logs: BTreeMap<u64, Log> = rows(rt, &world, "Log")?;
        // The effects of this tick: three flashes an actor, at the head,
        // and A facing its `Orientation` (the unit's frame's forward).
        for e in feed.take() {
            if e.effect != flash {
                continue;
            }
            let (a, b, o) = (&anims[&e.entity.0], &bodies[&e.entity.0], &orientations[&e.entity.0]);
            ensure!(pairs.contains_key(&e.entity.0), "tick {tick}: flash for a non-actor");
            ensure!(e.bone == head, "tick {tick}: flash at bone {}, not the head", e.bone);
            let facing = qrot(o.rot, [1.0, 0.0, 0.0]);
            ensure!(facing[0] > 0.9, "tick {tick}: the actor's forward is {facing:?}");
            if param & BACKWARDS != 0 {
                // Drawn by its `Orientation` (+x) while it walks the other
                // way: the muzzle is behind where the pair started.
                let start = pairs[&e.entity.0].chest[0] - 12.0;
                ensure!(
                    e.pos[0] < start + 1.0 && b.vx <= 0.0,
                    "tick {tick}: backwards flash at {} (started at {start}), vx {}",
                    e.pos[0],
                    b.vx
                );
            }
            let (p, q) = rig.bone_frame(&channels(a), e.bone);
            let (pos, rot) = place(o.rot, [b.x, b.y, b.radius / rig.footprint()], p, q);
            let d = qrot(rot, [0.3, 0.0, 0.0]);
            let want = [pos[0] + d[0], pos[1] + d[1], pos[2] + d[2]];
            ensure!(e.pos == want, "tick {tick}: flash at {:?}, head at {want:?}", e.pos);
            out.effects.push(e);
        }
        // Answered aims: the head's direction along the aim-from-to-chest
        // line, checked in glam.
        for (&a, log) in &logs {
            let before = answered.get(&a).copied().unwrap_or(0);
            if log.answered > before {
                let err = aim_error(
                    &rig,
                    head,
                    aim_from,
                    &anims[&a],
                    &bodies[&a],
                    &orientations[&a],
                    pairs[&a].chest,
                );
                let e = out.aim_errors.entry(a).or_insert(0.0);
                *e = e.max(err);
                out.answered += (log.answered - before) as u64;
                answered.insert(a, log.answered);
            }
            let killed = kills.entry(a).or_insert(0);
            if log.kills > *killed {
                *killed = log.kills;
                out.actors
                    .entry(a)
                    .or_insert(Actor {
                        log: *log,
                        gone: None,
                        killed: None,
                    })
                    .killed = Some(tick);
            }
        }
        let alive: BTreeSet<u64> = pairs.keys().copied().collect();
        for &a in previous.difference(&alive) {
            out.actors
                .entry(a)
                .or_insert(Actor {
                    log: Log::default(),
                    gone: None,
                    killed: None,
                })
                .gone = Some(tick);
        }
        for (&a, log) in &logs {
            out.actors
                .entry(a)
                .or_insert(Actor {
                    log: *log,
                    gone: None,
                    killed: None,
                })
                .log = *log;
        }
        previous = alive;
    }
    out.digest = rt.digest(&world)?.total;
    Ok(out)
}

fn runtime(wasm: &[u8], feed: &Arc<FxFeed>, workers: usize) -> Result<Runtime> {
    let plugin = AnimPlugin::new([("fox".to_string(), fox())], 1.0 / HZ as f32);
    let cfg = RuntimeConfig {
        max_workers: workers,
        range_rows: 64,
        parallel_min_rows: 1,
        tick_hz: HZ,
        ..RuntimeConfig::default()
    };
    Runtime::with_hosts(
        wasm,
        cfg,
        vec![
            Arc::new(plugin),
            Arc::new(SpatialPlugin::square(2048.0, 64.0)),
            Arc::new(FxPlugin::new(Some(feed.clone()))),
        ],
    )
}

/// What every variant must show, given its parameter and how many pairs.
fn check(o: &Outcome, workers: usize, param: u64, pairs: u64) -> Result<()> {
    let standing = param & STANDING != 0;
    let stuck = param & STUCK != 0;
    ensure!(
        o.actors.len() as u64 == pairs,
        "{workers} workers: {} actors",
        o.actors.len()
    );
    ensure!(
        o.effects.len() as u64 == 3 * pairs,
        "{workers} workers: {} flashes, expected {}",
        o.effects.len(),
        3 * pairs
    );
    ensure!(o.answered > 0, "{workers} workers: no answered aim");
    let bound = if standing {
        STANDING_AIM_BOUND
    } else {
        RUNNING_AIM_BOUND
    };
    let worst = o.aim_errors.values().copied().fold(0.0f32, f32::max);
    ensure!(worst <= bound, "{workers} workers: aim error {worst}° > {bound}°");
    for (&a, actor) in &o.actors {
        let log = &actor.log;
        let (want_starts, want_stops) = if standing { (0, 0) } else { (2, 1) };
        ensure!(
            (log.starts, log.stops, log.handled_starts, log.handled_stops)
                == (want_starts, want_stops, want_starts, want_stops),
            "{workers} workers: actor {a} callins {log:?}"
        );
        ensure!(
            (log.fired, log.flashed, log.handled_fired) == (3, 3, 3),
            "{workers} workers: actor {a} fires {log:?}"
        );
        ensure!(log.kills == 1, "{workers} workers: actor {a} kills {log:?}");
        ensure!(
            log.checks > 0 && log.found_a == 0 && log.found_b == log.checks,
            "{workers} workers: actor {a} spatial {log:?}"
        );
        let (killed, gone) = (actor.killed.unwrap_or(0), actor.gone.unwrap_or(0));
        ensure!(
            killed > 0 && gone > killed,
            "{workers} workers: actor {a} killed {killed}, gone {gone}"
        );
        if stuck {
            ensure!(
                gone == killed + CAP,
                "{workers} workers: actor {a} despawned at {gone}, killed {killed}, cap {CAP}"
            );
        } else {
            ensure!(
                gone < killed + CAP,
                "{workers} workers: actor {a} died in time: {gone} / {killed}"
            );
        }
    }
    Ok(())
}

/// Everything the run did, which must be the same at any worker count.
fn same(a: &Outcome, b: &Outcome, workers: usize, base: usize) -> Result<()> {
    ensure!(
        a.digest == b.digest,
        "{workers} workers: digest {:016x}, {base} workers {:016x}",
        a.digest,
        b.digest
    );
    ensure!(
        a.actors == b.actors,
        "{workers} workers: logs differ from {base} workers"
    );
    ensure!(
        a.effects == b.effects,
        "{workers} workers: effects differ from {base} workers"
    );
    ensure!(
        a.aim_errors == b.aim_errors,
        "{workers} workers: aim errors differ from {base} workers"
    );
    Ok(())
}

#[test]
fn duel_runs_the_aim_fire_and_death_sequence_at_any_worker_count() -> Result<()> {
    let wasm = ae_sim::build::game("duel")?;
    let feed = Arc::new(FxFeed::default());
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let max = cores.clamp(1, 8);
    let mut rt = runtime(&wasm, &feed, max)?;
    let mut order = vec![max];
    order.extend([1, 2, 4, 8].into_iter().filter(|&w| w < max));
    let mut base: Option<(usize, Outcome)> = None;
    for &w in &order {
        let o = run(&mut rt, &feed, w, PAIRS, TICKS)?;
        check(&o, w, PAIRS, PAIRS)?;
        let worst = o.aim_errors.values().copied().fold(0.0f32, f32::max);
        eprintln!(
            "{w} worker(s): digest {:016x}; {} actors, {} flashes, {} aims answered (worst {worst:.4}°)",
            o.digest,
            o.actors.len(),
            o.effects.len(),
            o.answered
        );
        match &base {
            None => base = Some((w, o)),
            Some((base_workers, b)) => same(&o, b, w, *base_workers)?,
        }
    }
    // The variants need a fresh runtime (the column arena is process-wide).
    drop(rt);
    let wasm = ae_sim::build::game("duel")?;
    for (name, bit) in [("standing", STANDING), ("backwards", BACKWARDS), ("stuck", STUCK)] {
        let feed = Arc::new(FxFeed::default());
        let mut rt = runtime(&wasm, &feed, 4)?;
        let param = VARIANT_PAIRS | bit;
        let o = run(&mut rt, &feed, 4, param, VARIANT_TICKS)?;
        check(&o, 4, param, VARIANT_PAIRS)?;
        eprintln!(
            "{name}: {} actors, {} flashes, {} aims answered (worst {:.4}°)",
            o.actors.len(),
            o.effects.len(),
            o.answered,
            o.aim_errors.values().copied().fold(0.0f32, f32::max)
        );
        drop(rt);
    }

    // Death's edge cases, on their own module (a fresh runtime again).
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/death_guest/Cargo.toml");
    let wasm = ae_sim::build::build_guest(&manifest, "ae_services_death_guest")?;
    let cfg = RuntimeConfig {
        max_workers: 4,
        range_rows: 64,
        parallel_min_rows: 1,
        tick_hz: HZ,
        ..RuntimeConfig::default()
    };
    let mut rt = Runtime::with_hosts(&wasm, cfg, Vec::new())?;
    let mut base: Option<(usize, u128)> = None;
    for &w in &[4usize, 1] {
        let mut world = rt.world_for(3, w)?;
        for _ in 0..10 {
            rt.tick(&mut world, w)?;
        }
        let counters: BTreeMap<u64, Counters> = rows(&rt, &world, "Counters")?;
        let dying: BTreeMap<u64, Dying> = rows(&rt, &world, "Dying")?;
        let kinds: BTreeMap<u64, Kind> = rows(&rt, &world, "Kind")?;
        ensure!(
            counters.len() == 3 && dying.len() == 3,
            "{w} workers: {} corpses, {} counters: {:?} / {:?}",
            dying.len(),
            counters.len(),
            counters,
            dying
        );
        let mut deaths = [0; 3];
        let mut killed = [u64::MAX; 3];
        for (&e, kind) in &kinds {
            let role = kind.role as usize;
            deaths[role] = counters[&e].deaths;
            killed[role] = dying[&e].killed_at;
        }
        ensure!(deaths == [1, 1, 1], "{w} workers: deaths {deaths:?}");
        ensure!(killed == [0, 1, 2], "{w} workers: killed_at {killed:?}");
        let digest = rt.digest(&world)?.total;
        match &base {
            None => base = Some((w, digest)),
            Some((w0, d0)) => ensure!(
                digest == *d0,
                "{w} workers: digest {digest:016x}, {w0} workers {d0:016x}"
            ),
        }
    }
    Ok(())
}
