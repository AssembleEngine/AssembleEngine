//! The animation service on the `scripted` game: units wearing the Khronos
//! Fox walk and stand; a plain game system picks clips and tweens by name
//! (`ae_sdk::anim` lookups) and the `anim` service advances them.
//! * The standing clip plays on layer 0; the walking clip blends in on
//!   layer 1 while a unit walks (its phase by the metre), and fades out and
//!   is cleared when it stands; the head is taken from the clips and turned
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
        if (8..90).contains(&k) {
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

/// Aiming helpers: `aim_angle` and `turn_toward` (pure SDK math), and the
/// `anim:dir_in` query, whose bytes give `Rig::dir_in`.
#[test]
fn aiming_helpers() -> Result<()> {
    use ae_sdk::anim::{Axis, aim_angle};
    use ae_sim::host::HostPlugin;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};
    ensure!((aim_angle(Axis::UP, Axis::FORWARD, [0.0, 0.0, -1.0]) - FRAC_PI_2).abs() < 1e-6, "left is a positive turn about up");
    ensure!((aim_angle(Axis::RIGHT, Axis::FORWARD, [1.0, 1.0, 0.3]) - FRAC_PI_4).abs() < 1e-6, "up 45°, sideways ignored");
    ensure!(aim_angle(Axis::UP, Axis::FORWARD, [0.0, 1.0, 0.0]) == 0.0, "along the axis");
    // turn_toward: the goal's axis and angle take forward onto the direction.
    let d = [0.36, 0.48, -0.8];
    let mut a = Anim::new(0);
    a.turn_toward(3, Axis::FORWARD, d, 1.0);
    let [x, y, z, angle] = a.goal[0];
    let (s, c) = angle.sin_cos();
    // Rodrigues on (1, 0, 0).
    let p = [1.0f32, 0.0, 0.0];
    let k = [x, y, z];
    let kxp = [k[1] * p[2] - k[2] * p[1], k[2] * p[0] - k[0] * p[2], k[0] * p[1] - k[1] * p[0]];
    let kp = k[0];
    let r: Vec<f32> = (0..3).map(|i| p[i] * c + kxp[i] * s + k[i] * kp * (1.0 - c)).collect();
    ensure!(r.iter().zip(d).all(|(a, b)| (a - b).abs() < 1e-5), "forward turned onto {d:?}: {r:?}");
    ensure!(a.manual_flags[0] & ae_sdk::anim::manual_flags::SHORT != 0, "the short way");
    // The query's bytes against the rig, on a Fox mid-clip with a turned neck.
    let plugin = AnimPlugin::new([("fox".to_string(), fox())], TICK);
    let rig = plugin.rig(ae_sdk::anim::model_id("fox")).unwrap().clone();
    let (head, neck) = (rig.bone("b_Head_05").unwrap(), rig.bone("b_Neck_04").unwrap());
    let mut a = Anim::new(ae_sdk::anim::model_id("fox"));
    a.play(0, rig.clip("Survey").unwrap()).at(1.3);
    a.turn(neck, Axis::UP, 0.5, 1.0);
    let mut input = vec![0u8; size_of::<Anim>()];
    input.copy_from_slice(unsafe { std::slice::from_raw_parts((&a as *const Anim).cast::<u8>(), size_of::<Anim>()) });
    input.extend_from_slice(&(head as u32).to_le_bytes());
    for v in [0.6f32, 0.0, -0.8] {
        input.extend_from_slice(&v.to_le_bytes());
    }
    let op = ae_abi::name_id(ae_abi::query::ANIM_DIR_IN);
    ensure!(plugin.queries() == vec![op]);
    let mut out = [0u8; 12];
    ensure!(plugin.query(op, &input, &mut out)? == 12);
    let got: Vec<f32> = out.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let want = rig.dir_in(&ae_services::anim::channels(&a), head, [0.6, 0.0, -0.8]);
    ensure!(got == want, "{got:?} vs {want:?}");
    ensure!((want[0] - 0.6).abs() > 1e-3 || (want[2] + 0.8).abs() > 1e-3, "the clip and the neck turn move it: {want:?}");
    ensure!(plugin.query(op, &input[..10], &mut out).is_err(), "short input is an error, not a panic");
    Ok(())
}
