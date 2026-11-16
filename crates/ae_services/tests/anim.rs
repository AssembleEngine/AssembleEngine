//! The animation service on the `scripted` game: units wearing the Khronos
//! Fox walk and stand; a plain game system picks clips and tweens by name
//! (`ae_sdk::anim` lookups) and the `anim` service advances them.
//! * The standing clip plays on layer 0; the walking clip blends in on
//!   layer 1 while a unit walks (its phase by the metre), and fades out and
//!   is cleared when it stands; the head is taken from the clips and yawed
//!   while it stands, turned back and given back when it walks.
//! * The world is the same at 1, 2, 4 and 8 workers.
//!
//! One runtime at a time: the column arena is process-wide.

use ae_model::anim::Rig;
use ae_sdk::anim::Anim;
use ae_services::anim::AnimPlugin;
use ae_sim::runtime::{Runtime, RuntimeConfig};
use anyhow::{Result, ensure};
use std::path::Path;
use std::sync::Arc;

const UNITS: u64 = 2000;
const TICK: f32 = 1.0 / 30.0;

fn fox() -> Rig {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/khronos/Fox/Fox.glb");
    Rig::new(&ae_gltf::convert_file(&path).unwrap().model)
}

fn decode<T: Copy>(bytes: &[u8]) -> T {
    assert_eq!(bytes.len(), size_of::<T>());
    unsafe { bytes.as_ptr().cast::<T>().read_unaligned() }
}

#[test]
fn units_walk_and_stand_animated() -> Result<()> {
    let wasm = ae_sim::build::game("scripted")?;
    let plugin = AnimPlugin::new([("fox".to_string(), fox())], TICK);
    let rig = plugin.rig(ae_sdk::anim::model_id("fox")).unwrap().clone();
    let (survey, walk, head) = (rig.clip("Survey").unwrap(), rig.clip("Walk").unwrap(), rig.bone("b_Head_05").unwrap());
    let run = |workers: usize, ticks: u64| -> Result<(u128, Vec<(u64, Anim)>)> {
        let cfg = RuntimeConfig { max_workers: 8, range_rows: 256, ..RuntimeConfig::default() };
        let mut rt = Runtime::with_hosts(&wasm, cfg, vec![Arc::new(plugin.clone())])?;
        let mut world = rt.world_for(UNITS, workers)?;
        for _ in 0..ticks {
            rt.tick(&mut world, workers)?;
        }
        let rows = rt.rows(&world, "Anim")?.into_iter().map(|(e, b)| (e, decode::<Anim>(&b))).collect();
        Ok((rt.digest(&world)?.total, rows))
    };
    // Tick 120: every unit is some way into its walk-stand cycle of 150.
    let (digest, rows) = run(1, 120)?;
    ensure!(rows.len() == UNITS as usize);
    let (mut walking, mut standing, mut fading) = (0, 0, 0);
    for (e, a) in &rows {
        ensure!(a.clip[0] == survey && a.weight[0] == 1.0, "unit {e}: the standing clip on layer 0: {a:?}");
        ensure!(a.phase[0] > 0.0 && a.phase[0] < rig.duration(survey), "unit {e}: layer 0 plays and loops");
        // Where the unit is in its cycle at the last tick the game ran.
        let k = (119 + (*e as u32 as u64) * 7) % 150;
        if k < 90 && k >= 8 {
            walking += 1;
            ensure!(a.clip[1] == walk && a.weight[1] == 1.0 && a.flags[1] & ae_sdk::anim::layer_flags::GROUND_SPEED != 0, "unit {e} at {k}: walking at full weight: {a:?}");
            ensure!(a.phase[1] > 0.0 && a.phase[1] < rig.duration(walk), "unit {e}: walking by the metre: {a:?}");
            if k >= 8 + 30 {
                ensure!(!a.is_manual(head), "unit {e} at {k}: the head turned back and given back to the clips: {a:?}");
            }
        } else if k >= 90 + 12 {
            standing += 1;
            ensure!(a.clip[1] == 0, "unit {e} at {k}: the walking clip cleared once faded: {a:?}");
            ensure!(a.is_manual(head), "unit {e} at {k}: the head taken from the clips: {a:?}");
            if k >= 90 + 35 {
                ensure!(a.turned(head) && a.bone[0] == head && a.value[0] == a.goal[0], "unit {e} at {k}: the head turned: {a:?}");
            }
        } else if k >= 90 {
            fading += 1;
            ensure!(a.target[1] == 0.0 || a.clip[1] == 0, "unit {e} at {k}: fading out: {a:?}");
        }
    }
    ensure!(walking > 500 && standing > 300 && fading > 50, "{walking} walking, {standing} standing, {fading} fading");
    for workers in [2, 4, 8] {
        let (d, _) = run(workers, 120)?;
        ensure!(d == digest, "{workers} workers: digest {d:x} vs {digest:x}");
    }
    Ok(())
}
