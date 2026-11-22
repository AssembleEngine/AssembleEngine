//! Unit scripts with callins (M14 step 5) on the `puppet` game: Foxes walk
//! and stand, the system moving them posts `STARTED_MOVING` /
//! `STOPPED_MOVING`, and each unit's `async` script plays its clips and
//! turns its head, racing a look around against the next callin.
//!
//! Checked at 1, 2, 4 and 8 workers (most first), the world the same:
//! * every unit handled its callins (starts and stops counted against its
//!   walk-stand cycle); looks both finished and were cut short by walking,
//!   head turns back both finished and were cut by stopping (`race`);
//! * `sleep(ms)` slept `ceil(ms × 30 / 1000)` ticks; a head turn was never
//!   over in the poll that set it (`turned` waits for the ticker);
//! * a unit waiting only for callins is not polled (its `due` is
//!   `u64::MAX`, its wait mask the two kinds), so far fewer polls than
//!   units times ticks;
//! * the clips are the script's: a walking unit has the walk on layer 1.
//!
//! Then units despawned mid-script and new ones spawned on their indices:
//! the new scripts start clean in the old slots, the same at any worker
//! count.
//!
//! Then aiming (`AIM`): standing units, each facing its own way (a third
//! tilted), asked every few ticks to point their head at a moving target
//! (from the head's world position, `anim:bone_world`); the script turns
//! it there (`to_unit`, `dir_in`, `turn_toward`) and answers. Every unit
//! answered many requests, and each answered direction is where the head's
//! forward points in the world, to a small fraction of a degree.
//!
//! One runtime at a time: the column arena is process-wide.

use ae_model::anim::Rig;
use ae_sdk::anim::Anim;
use ae_services::anim::AnimPlugin;
use ae_sim::runtime::{Runtime, RuntimeConfig};
use anyhow::{Result, ensure};
use std::path::Path;
use std::sync::Arc;

const UNITS: u64 = 1500;
const TICKS: u64 = 320;
const CHURN: u64 = 1 << 32;
const CHURN_AT: u64 = 100;
const AIM: u64 = 1 << 33;
const WALK_TICKS: u64 = 90;
const CYCLE: u64 = 150;
const HZ: u32 = 30;

/// `puppet::target`: the aim target at `tick` for `units`.
fn target(tick: u64, units: u32) -> glam::Vec3 {
    let (cx, cy) = (50.0 * 4.0, (units / 100) as f32 * 4.0 * 0.5);
    let a = tick as f32 * 0.01;
    glam::Vec3::new(cx + 30.0 * a.cos(), 3.0, cy + 30.0 * a.sin())
}

/// Where the head is in the world and where its rest forward points, by
/// the model's bone matrices (`Rig::globals`, not `bone_frame`'s chain),
/// the unit's orientation and fit, in glam.
fn head_in_world(
    rig: &Rig,
    head: u16,
    a: &Anim,
    b: &ae_sdk::nav::Body,
    orientation: [f32; 4],
) -> (glam::Vec3, glam::Vec3) {
    let affine = |ch: Option<&Anim>| {
        let mut pose = rig.rest_pose();
        if let Some(a) = ch {
            rig.pose(&ae_services::anim::channels(a), &mut pose);
        }
        let mut g = vec![ae_model::anim::IDENTITY; rig.bones()];
        rig.globals(&pose, &mut g);
        let m = g[head as usize];
        (
            glam::Mat3::from_cols_array(&[m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]]),
            glam::Vec3::new(m[3], m[7], m[11]),
        )
    };
    let ((now, at), (rest, _)) = (affine(Some(a)), affine(None));
    let carried = (now * rest.inverse() * glam::Vec3::X).normalize();
    let q = glam::Quat::from_array(orientation);
    let pos = glam::Vec3::new(b.x, 0.0, b.y) + q * (at * (b.radius / rig.footprint()));
    (pos, q * carried)
}

fn fox() -> Rig {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/khronos/Fox/Fox.glb");
    let mut rig = Rig::new(&ae_gltf::convert_file(&path).unwrap().model);
    // The masks from content, as the plugin and the renderer get them.
    rig.apply_rig_file(&std::fs::read(path.with_extension("rig.json")).unwrap())
        .unwrap();
    rig
}

fn decode<T: Copy>(bytes: &[u8]) -> T {
    assert_eq!(bytes.len(), size_of::<T>());
    unsafe { bytes.as_ptr().cast::<T>().read_unaligned() }
}

/// `puppet::Log`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Log {
    starts: u32,
    stops: u32,
    looks: u32,
    cut: u32,
    backs: u32,
    backs_cut: u32,
    slept: u32,
    turn_min: u32,
    aims: u32,
    checked: u32,
    aim_err: f32,
}

/// `ae_sdk::script::Script`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Script {
    state: u32,
    polls: u32,
    due: u64,
    size: u32,
    moved: u32,
    last: u32,
    waits: u32,
}

/// `ae_sdk::anim::Weapons`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Weapons {
    aim_from: [u16; 4],
    fire_from: [u16; 4],
    dir: [f32; 12],
    asked: [u32; 4],
    aimed: [u32; 4],
}

struct Outcome {
    digest: u128,
    bodies: Vec<(u64, ae_sdk::nav::Body)>,
    orientations: Vec<(u64, [f32; 4])>,
    weapons: Vec<(u64, Weapons)>,
    logs: Vec<(u64, Log)>,
    scripts: Vec<(u64, Script)>,
    anims: Vec<(u64, Anim)>,
}

fn run(rt: &mut Runtime, workers: usize, param: u64) -> Result<Outcome> {
    let mut world = rt.world_for(param, workers)?;
    for _ in 0..TICKS {
        rt.tick(&mut world, workers)?;
    }
    let rows = |name| -> Result<Vec<(u64, Vec<u8>)>> { rt.rows(&world, name) };
    Ok(Outcome {
        digest: rt.digest(&world)?.total,
        bodies: rows("Body")?.into_iter().map(|(e, b)| (e, decode(&b))).collect(),
        orientations: rows("Orientation")?.into_iter().map(|(e, b)| (e, decode(&b))).collect(),
        weapons: rows("Weapons")?.into_iter().map(|(e, b)| (e, decode(&b))).collect(),
        logs: rows("Log")?.into_iter().map(|(e, b)| (e, decode(&b))).collect(),
        scripts: rows("Script")?.into_iter().map(|(e, b)| (e, decode(&b))).collect(),
        anims: rows("Anim")?.into_iter().map(|(e, b)| (e, decode(&b))).collect(),
    })
}

/// Walk starts and stops `steer` posted to unit `index` in ticks `from..to`
/// (it walks for the first `WALK_TICKS` of its cycle).
fn transitions(index: u32, from: u64, to: u64) -> (u32, u32) {
    let walking = |t: u64| (t + index as u64 * 7) % CYCLE < WALK_TICKS;
    let (mut starts, mut stops) = (0, 0);
    for t in from..to {
        let before = t > from && walking(t - 1);
        match (before, walking(t)) {
            (false, true) => starts += 1,
            (true, false) => stops += 1,
            _ => {}
        }
    }
    (starts, stops)
}

#[test]
fn scripts_answer_callins_the_same_at_any_worker_count() -> Result<()> {
    let wasm = ae_sim::build::game("puppet")?;
    let plugin = AnimPlugin::new([("fox".to_string(), fox())], 1.0 / HZ as f32);
    let rig = plugin.rig(ae_sdk::anim::model_id("fox")).unwrap().clone();
    let (walk, head) = (rig.clip("Walk").unwrap(), rig.bone("b_Head_05").unwrap());
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let max = cores.clamp(1, 8);
    let cfg = RuntimeConfig {
        max_workers: max,
        range_rows: 128,
        parallel_min_rows: 256,
        tick_hz: HZ,
        ..RuntimeConfig::default()
    };
    let mut rt = Runtime::with_hosts(&wasm, cfg, vec![Arc::new(plugin)])?;

    let mut order = vec![max];
    order.extend([1, 2, 4, 8].into_iter().filter(|&w| w < max));
    let mut base: Option<(usize, u128)> = None;
    for &w in &order {
        let r = run(&mut rt, w, UNITS)?;
        ensure!(r.logs.len() == UNITS as usize && r.scripts.len() == UNITS as usize);
        let sum = |f: fn(&Log) -> u32| r.logs.iter().map(|(_, l)| f(l) as u64).sum::<u64>();
        let polls: u64 = r.scripts.iter().map(|(_, s)| s.polls as u64).sum();
        let idle = r.scripts.iter().filter(|(_, s)| s.due == u64::MAX).count();
        eprintln!(
            "{w} worker(s): digest {:016x}; {} starts, {} stops, looks {} done {} cut, turns back {} done {} cut; {polls} polls ({:.1}% of unit-ticks), {idle} waiting only for callins",
            r.digest as u64,
            sum(|l| l.starts),
            sum(|l| l.stops),
            sum(|l| l.looks),
            sum(|l| l.cut),
            sum(|l| l.backs),
            sum(|l| l.backs_cut),
            100.0 * polls as f64 / (UNITS * TICKS) as f64,
        );
        for (e, l) in &r.logs {
            let index = *e as u32;
            // Spawned at tick 0, moved from tick 1; callins posted at the
            // last tick are seen at the last tick's resume.
            let (starts, stops) = transitions(index, 1, TICKS);
            ensure!(
                (l.starts, l.stops) == (starts, stops),
                "unit {e}: handled {} starts, {} stops; posted {starts}, {stops}",
                l.starts,
                l.stops
            );
            if l.slept != 0 {
                let ms = 100 + (index % 8) * 150;
                let want = (ms * HZ).div_ceil(1000);
                ensure!(
                    l.slept == want,
                    "unit {e}: slept {} ticks for {ms} ms, not {want}",
                    l.slept
                );
            }
            ensure!(l.turn_min != 0, "unit {e}: a head turn was over the poll it was set");
        }
        for (e, s) in &r.scripts {
            ensure!(s.state == 1, "unit {e}: script not running: {s:?}");
            if s.due == u64::MAX {
                ensure!(
                    s.waits == 0b11,
                    "unit {e}: waits only for callins, but not the two moves: {s:?}"
                );
            }
        }
        for (e, a) in &r.anims {
            let k = (TICKS - 1 + (*e as u32 as u64) * 7) % CYCLE;
            if (10..WALK_TICKS).contains(&k) {
                ensure!(
                    a.clip[1] == walk,
                    "unit {e} at {k}: walking without the walk clip: {a:?}"
                );
            }
            if (WALK_TICKS + 2..CYCLE).contains(&k) {
                ensure!(
                    a.target[1] == 0.0 && a.is_manual(head),
                    "unit {e} at {k}: standing, the walk fading and the head looking: {a:?}"
                );
            }
        }
        ensure!(
            sum(|l| l.looks) > 0 && sum(|l| l.cut) > 0,
            "{w} workers: looks must both finish and be cut"
        );
        ensure!(sum(|l| l.backs) > 0, "{w} workers: no head turned back");
        ensure!(idle > 0, "{w} workers: no unit waits only for callins");
        ensure!(
            polls < UNITS * TICKS / 2,
            "{w} workers: {polls} polls: units waiting for callins are polled"
        );
        match base {
            None => base = Some((w, r.digest)),
            Some((bw, d)) => ensure!(
                r.digest == d,
                "{w} workers: digest {:032x}, {bw} workers {d:032x}",
                r.digest
            ),
        }
    }

    // Despawned mid-script, new units on their indices.
    let mut churned: Option<u128> = None;
    for &w in &[max, 1] {
        let r = run(&mut rt, w, UNITS | CHURN)?;
        let n = UNITS as usize;
        ensure!(
            r.scripts.len() == n - n.div_ceil(4) + n.div_ceil(4),
            "{w} workers: {} units after the churn",
            r.scripts.len()
        );
        let reused = r
            .scripts
            .iter()
            .filter(|(e, _)| (*e as u32).is_multiple_of(4) && (*e as u32) < UNITS as u32 && (e >> 32) != 0)
            .count();
        eprintln!(
            "churn, {w} worker(s): digest {:016x}; {reused} new units on old indices",
            r.digest as u64
        );
        ensure!(
            reused > 0,
            "no new unit reused a despawned one's index (the stale slot path is not covered)"
        );
        for (e, s) in &r.scripts {
            ensure!(
                s.state == 1 && s.polls > 0,
                "unit {e}: script not running after the churn: {s:?}"
            );
        }
        for (e, l) in &r.logs {
            if (e >> 32) != 0 {
                // Spawned at tick CHURN_AT + 1, moved from the tick after.
                let (starts, stops) = transitions(*e as u32, CHURN_AT + 2, TICKS);
                ensure!(
                    (l.starts, l.stops) == (starts, stops),
                    "new unit {e}: handled {} starts, {} stops; posted {starts}, {stops}",
                    l.starts,
                    l.stops
                );
            }
        }
        match churned {
            None => churned = Some(r.digest),
            Some(d) => ensure!(r.digest == d, "churn: {w} workers' digest differs"),
        }
    }

    // Aiming.
    let mut aimed: Option<u128> = None;
    for &w in &[max, 1] {
        let r = run(&mut rt, w, UNITS | AIM)?;
        let aims: u64 = r.logs.iter().map(|(_, l)| l.aims as u64).sum();
        let fewest = r.logs.iter().map(|(_, l)| l.aims).min().unwrap_or(0);
        let worst = r.logs.iter().map(|(_, l)| l.aim_err).fold(0.0f32, f32::max);
        eprintln!(
            "aim, {w} worker(s): digest {:016x}; {aims} answered aims measured (fewest {fewest} a unit), worst {worst:.5}°",
            r.digest as u64
        );
        ensure!(fewest >= 10, "{w} workers: a unit answered only {fewest} aims");
        ensure!(
            worst < 0.01,
            "{w} workers: an answered aim is {worst}° off the head's forward"
        );
        for (e, a) in &r.anims {
            ensure!(a.is_manual(head), "unit {e}: the head not turned: {a:?}");
        }
        // Checked on their own: units that answered the last request look
        // at where the target was then, from where their head is.
        let last = (TICKS - 1) / 6 * 6;
        let at = target(last, UNITS as u32);
        let (mut looked, mut worst) = (0, 0.0f32);
        ensure!([r.bodies.len(), r.orientations.len(), r.weapons.len()] == [r.anims.len(); 3]);
        for (((e, a), (eb, b)), ((ef, f), (ew, wp))) in
            r.anims.iter().zip(&r.bodies).zip(r.orientations.iter().zip(&r.weapons))
        {
            ensure!([eb, ef, ew] == [e; 3], "rows out of step: {e} {eb} {ef} {ew}");
            ensure!(
                (wp.aim_from[0], wp.fire_from[0]) == (head, head),
                "unit {e}: the script did not name its aim-from and fire-from bones: {wp:?}"
            );
            if wp.asked[0] == 0 || wp.aimed[0] != wp.asked[0] {
                continue;
            }
            let (pos, fwd) = head_in_world(&rig, head, a, b, *f);
            let want = (at - pos).normalize();
            let angle = fwd.angle_between(want).to_degrees();
            ensure!(angle < 0.05, "unit {e}: the head looks {angle}° off the target");
            looked += 1;
            worst = worst.max(angle);
        }
        eprintln!("aim, {w} worker(s): {looked} units looking at the target, worst {worst:.5}° (glam)");
        // The last request is a tick before the end: those whose turn was short.
        ensure!(
            looked > UNITS as usize / 5,
            "{w} workers: only {looked} units answered the last request"
        );
        match aimed {
            None => aimed = Some(r.digest),
            Some(d) => ensure!(r.digest == d, "aim: {w} workers' digest differs"),
        }
    }
    Ok(())
}
