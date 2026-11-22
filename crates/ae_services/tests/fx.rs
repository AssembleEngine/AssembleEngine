//! Effects at joints (M14 step 6, B) on the `fx_guest` test module: units
//! turned their own way (a third tilted) emit a `"flash"` at the head and a
//! `"smoke"` at the spine every tick, after the animation service. The
//! `fx` host system drains the stream into `FxFeed`.
//!
//! Checked at 8 and 1 workers (most first), the world and the feed the
//! same:
//! * two effects a unit a tick, in key order (entity ascending, flash then
//!   smoke);
//! * each `pos` is the bone's world position plus its offset, as
//!   `Rig::bone_frame` and `place` compute it; `dir` is the bone's world
//!   forward; the seed is the hash of entity, tick, effect and bone;
//! * the world digests are equal.
//!
//! One runtime at a time: the column arena is process-wide.

use ae_model::anim::{Rig, qrot};
use ae_sdk::aim::Orientation;
use ae_sdk::anim::{Anim, model_id};
use ae_sdk::fx::{Effect, seed};
use ae_sdk::nav::Body;
use ae_services::anim::{AnimPlugin, channels, place};
use ae_services::fx::{FxFeed, FxPlugin};
use ae_sim::runtime::{Runtime, RuntimeConfig};
use anyhow::{Result, ensure};
use std::path::Path;
use std::sync::Arc;

const UNITS: u64 = 200;
const TICKS: u64 = 60;
const HZ: u32 = 30;
const HEAD: &str = "b_Head_05";
const SPINE: &str = "b_Spine01_02";
const FORWARD: [f32; 3] = [0.3, 0.0, 0.0];
const UP: [f32; 3] = [0.0, 0.2, 0.0];

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

/// The rows of `component`, by entity (ascending).
fn rows<T: Copy>(rt: &Runtime, world: &ae_sim::runtime::World, component: &str) -> Result<Vec<(u64, T)>> {
    Ok(rt
        .rows(world, component)?
        .into_iter()
        .map(|(e, b)| (e, decode(&b)))
        .collect())
}

#[test]
fn effects_at_joints_follow_the_pose_the_same_at_any_worker_count() -> Result<()> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fx_guest/Cargo.toml");
    let wasm = ae_sim::build::build_guest(&manifest, "ae_services_fx_guest")?;
    let anim = AnimPlugin::new([("fox".to_string(), fox())], 1.0 / HZ as f32);
    let rig = anim.rig(model_id("fox")).unwrap().clone();
    let (head, spine) = (rig.bone(HEAD).unwrap(), rig.bone(SPINE).unwrap());
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let max = cores.clamp(1, 8);
    // Small ranges, so the rows of one tick are split across workers.
    let cfg = RuntimeConfig {
        max_workers: max,
        range_rows: 16,
        parallel_min_rows: 1,
        tick_hz: HZ,
        ..RuntimeConfig::default()
    };
    let feed = Arc::new(FxFeed::default());
    let mut rt = Runtime::with_hosts(
        &wasm,
        cfg,
        vec![Arc::new(anim), Arc::new(FxPlugin::new(Some(feed.clone())))],
    )?;
    let param = UNITS | (30u64 << 16) | (30u64 << 24);

    let mut order = vec![max];
    order.extend([1, 2, 4, 8].into_iter().filter(|&w| w < max));
    let mut base: Option<(usize, u128, Vec<Effect>)> = None;
    for &w in &order {
        let mut world = rt.world_for(param, w)?;
        let mut got: Vec<Effect> = Vec::new();
        for t in 1..=TICKS {
            rt.tick(&mut world, w)?;
            let tick = world.tick;
            ensure!(tick == t, "tick {tick}, expected {t}");
            let effects = feed.take();
            ensure!(
                effects.len() == 2 * UNITS as usize,
                "tick {t}: {} effects",
                effects.len()
            );
            let bodies: Vec<(u64, Body)> = rows(&rt, &world, "Body")?;
            let anims: Vec<(u64, Anim)> = rows(&rt, &world, "Anim")?;
            let orientations: Vec<(u64, Orientation)> = rows(&rt, &world, "Orientation")?;
            ensure!(
                bodies.len() == UNITS as usize && anims.len() == bodies.len() && orientations.len() == bodies.len()
            );
            for (k, e) in effects.iter().enumerate() {
                let unit = k / 2;
                let (entity, body) = &bodies[unit];
                let (bone, offset) = match k % 2 {
                    0 => (head, FORWARD),
                    _ => (spine, UP),
                };
                ensure!(
                    e.entity.0 == *entity,
                    "tick {t}: effect {k} for {} expected {entity}",
                    e.entity.0
                );
                // `run.tick()` is the tick being computed (0 for the first); the
                // world's and the stream's is one more.
                ensure!(
                    e.tick == t - 1 && e.bone == bone,
                    "tick {t}: effect {k} tick {} bone {} expected {} {bone}",
                    e.tick,
                    e.bone,
                    t - 1
                );
                ensure!(e.count == if k % 2 == 0 { 2 } else { 1 }, "tick {t}: effect {k} count");
                let a = &anims[unit].1;
                let o = &orientations[unit].1;
                let (p, q) = rig.bone_frame(&channels(a), bone);
                let (pos, rot) = place(o.rot, [body.x, body.y, body.radius / rig.footprint()], p, q);
                let d = qrot(rot, offset);
                let want = [pos[0] + d[0], pos[1] + d[1], pos[2] + d[2]];
                ensure!(e.pos == want, "tick {t}: effect {k} at {:?}, expected {want:?}", e.pos);
                let dir = qrot(rot, [1.0, 0.0, 0.0]);
                ensure!(
                    e.dir == dir,
                    "tick {t}: effect {k} direction {:?}, expected {dir:?}",
                    e.dir
                );
                ensure!(
                    e.seed == seed(e.entity, t - 1, e.effect, bone),
                    "tick {t}: effect {k} seed {}",
                    e.seed
                );
                ensure!(
                    e.effect == ae_sdk::abi::name_id(if k % 2 == 0 { "flash" } else { "smoke" }),
                    "tick {t}: effect {k} name"
                );
            }
            got.extend(effects);
        }
        let digest = rt.digest(&world)?.total;
        match &base {
            None => base = Some((w, digest, got.clone())),
            Some((w0, d0, g0)) => {
                ensure!(
                    digest == *d0,
                    "{w} workers: digest {digest:016x}, {w0} workers {d0:016x}"
                );
                ensure!(got == *g0, "{w} workers: the feed differs from {w0} workers");
            }
        }
        eprintln!(
            "{w} worker(s): digest {digest:016x}, {} effects, all at their bones",
            got.len()
        );
    }
    Ok(())
}
