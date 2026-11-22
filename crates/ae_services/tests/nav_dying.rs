//! A dying unit leaves nav's rows like a despawn (M14 step 6, D/F) on the
//! `nav_dying_guest`: three units in a column move to +x; the middle one is
//! either despawned or killed into `Dying` at tick 5. From then on the
//! other two move exactly as with the despawn, and the dying one is not
//! moved by its `MoveOrder`.
//!
//! One runtime at a time: the column arena is process-wide.

use ae_sdk::nav::{Body, NavAgent};
use ae_services::nav::{Map, NavPlugin};
use ae_sim::runtime::{Runtime, RuntimeConfig, World};
use anyhow::{Result, ensure};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

const TICKS: u64 = 40;
const KILL_AT: u64 = 5;

fn decode<T: Copy>(bytes: &[u8]) -> T {
    assert_eq!(bytes.len(), size_of::<T>());
    unsafe { bytes.as_ptr().cast::<T>().read_unaligned() }
}

fn rows<T: Copy>(rt: &Runtime, world: &World, name: &str) -> Result<BTreeMap<u64, T>> {
    Ok(rt
        .rows(world, name)?
        .into_iter()
        .map(|(e, b)| (e, decode(&b)))
        .collect())
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Role {
    n: u32,
    reserved: u32,
}

struct Outcome {
    digest: u128,
    /// Per role: its position and mode at each tick after the kill.
    by_role: BTreeMap<u32, Vec<([f32; 2], u32)>>,
    /// `Dying` rows at the end.
    dying: usize,
}

fn run(rt: &mut Runtime, workers: usize, param: u64) -> Result<Outcome> {
    let mut world = rt.world_for(param, workers)?;
    let mut out = Outcome {
        digest: 0,
        by_role: BTreeMap::new(),
        dying: 0,
    };
    for _ in 0..TICKS {
        rt.tick(&mut world, workers)?;
        if world.tick <= KILL_AT {
            continue;
        }
        let bodies: BTreeMap<u64, Body> = rows(rt, &world, "Body")?;
        let agents: BTreeMap<u64, NavAgent> = rows(rt, &world, "NavAgent")?;
        let roles: BTreeMap<u64, Role> = rows(rt, &world, "Role")?;
        for (&e, role) in &roles {
            let n = role.n;
            let Some(b) = bodies.get(&e) else { continue };
            out.by_role.entry(n).or_default().push(([b.x, b.y], agents[&e].mode));
        }
    }
    out.digest = rt.digest(&world)?.total;
    out.dying = rt.rows(&world, "Dying")?.len();
    Ok(out)
}

#[test]
fn a_dying_agent_leaves_nav_like_a_despawn() -> Result<()> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/nav_dying_guest/Cargo.toml");
    let wasm = ae_sim::build::build_guest(&manifest, "ae_services_nav_dying_guest")?;
    let mut rt = Runtime::with_hosts(
        &wasm,
        RuntimeConfig {
            max_workers: 4,
            range_rows: 16,
            parallel_min_rows: 1,
            ..RuntimeConfig::default()
        },
        vec![Arc::new(NavPlugin::new(Map::LandCrossing, 256))],
    )?;
    let gone = run(&mut rt, 4, 0)?;
    let dying = run(&mut rt, 4, 1)?;
    ensure!(!gone.by_role[&0].is_empty(), "no samples after the kill");
    for n in [0u32, 2] {
        ensure!(
            gone.by_role[&n] == dying.by_role[&n],
            "unit {n} moves differently with a dying neighbour: {:?} vs {:?}",
            gone.by_role[&n],
            dying.by_role[&n]
        );
    }
    // The dying unit stayed (its rows are the same) and did not move after
    // its kill.
    let samples = &dying.by_role[&1];
    ensure!(samples.len() as u64 == TICKS - KILL_AT, "{} samples", samples.len());
    ensure!(
        samples.iter().all(|s| *s == samples[0]),
        "the dying unit moved: {:?}",
        samples
    );
    // And its `Dying` stays (it is the entity kept, not despawned).
    ensure!(
        dying.dying == 1 && gone.dying == 0,
        "Dying rows: {} / {}",
        dying.dying,
        gone.dying
    );
    eprintln!(
        "gone: digest {:016x}, dying: digest {:016x}; neighbours equal, the dying one still",
        gone.digest, dying.digest
    );
    Ok(())
}
