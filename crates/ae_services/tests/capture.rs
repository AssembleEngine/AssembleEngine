//! End-of-tick captures (`Runtime::capture`) on the crossing game.
//!
//! * Capturing `Body` and `NavAgent` fields after every tick leaves the
//!   world exactly as without it: the same final digest through orders,
//!   reorders, a terrain quake and churn (units despawned and replaced, so
//!   entity ids are recycled). A cached flecs query would have created an
//!   entity per capture and shifted the ids churn hands out.
//! * The captured words equal the components' bytes, matched by entity.
//!
//! The column arena is process-wide, so this file holds one test.

use ae_sdk::nav::{Body, NavAgent};
use ae_services::nav::{Map, NavPlugin};
use ae_services::spatial::SpatialPlugin;
use ae_sim::capture::{Capture, CaptureSpec};
use ae_sim::runtime::{Runtime, RuntimeConfig};
use anyhow::{Result, ensure};
use std::collections::BTreeMap;
use std::sync::Arc;

const UNITS: u64 = 600;
const MAP_CELLS: i32 = 512;
const TICKS: u64 = 120;
/// Quake at 40, orders at 5, reorder at 30, churn at 60.
const PARAM: u64 = UNITS | 40 << 24 | 5 << 32 | 30 << 40 | 60 << 48;

const FIELDS: [(&str, &str); 7] = [("Body", "x"), ("Body", "y"), ("Body", "vx"), ("Body", "vy"), ("Body", "radius"), ("NavAgent", "profile"), ("NavAgent", "mode")];

fn decode<T: Copy>(bytes: &[u8]) -> T {
    assert_eq!(bytes.len(), size_of::<T>());
    unsafe { bytes.as_ptr().cast::<T>().read_unaligned() }
}

#[test]
fn capturing_every_tick_leaves_the_world_unchanged() -> Result<()> {
    let wasm = ae_sim::build::game("crossing")?;
    let nav = NavPlugin { map: Map::WaterCrossing, map_cells: MAP_CELLS };
    let spatial = SpatialPlugin::square(nav.extent(), 64.0);
    let cfg = RuntimeConfig { max_workers: 2, range_rows: 128, parallel_min_rows: 256, ..RuntimeConfig::default() };
    let mut rt = Runtime::with_hosts(&wasm, cfg, vec![Arc::new(nav), Arc::new(spatial)])?;
    let spec = CaptureSpec::new(&rt.schema, &FIELDS)?;
    ensure!(CaptureSpec::new(&rt.schema, &[("Body", "nope")]).is_err(), "unknown fields are rejected");

    let mut plain = rt.world_for(PARAM, 2)?;
    for _ in 0..TICKS {
        rt.tick(&mut plain, 2)?;
    }
    let plain_digest = rt.digest(&plain)?;
    drop(plain);

    let mut world = rt.world_for(PARAM, 2)?;
    let mut capture = Capture::default();
    let mut sizes = Vec::new();
    for _ in 0..TICKS {
        rt.tick(&mut world, 2)?;
        rt.capture(&world, &spec, &mut capture)?;
        ensure!(capture.tick == world.tick);
        sizes.push(capture.len());
    }
    let captured_digest = rt.digest(&world)?;
    ensure!(captured_digest == plain_digest, "capturing changed the world: {:032x} vs {:032x}", captured_digest.total, plain_digest.total);
    ensure!(sizes.iter().all(|&n| n == UNITS as usize), "every unit is captured every tick: {:?}", &sizes[..4]);

    // The last capture equals the components, by entity.
    let bodies: BTreeMap<u64, Body> = rt.rows(&world, "Body")?.into_iter().map(|(e, b)| (e, decode(&b))).collect();
    let agents: BTreeMap<u64, NavAgent> = rt.rows(&world, "NavAgent")?.into_iter().map(|(e, a)| (e, decode(&a))).collect();
    ensure!(capture.len() == bodies.len());
    let mut moving = 0;
    for (r, e) in capture.entities.iter().enumerate() {
        let (b, a) = (&bodies[e], &agents[e]);
        let want = [b.x.to_bits(), b.y.to_bits(), b.vx.to_bits(), b.vy.to_bits(), b.radius.to_bits(), a.profile, a.mode];
        let got: Vec<u32> = capture.columns.iter().map(|c| c[r]).collect();
        ensure!(got == want, "entity {e:#x}: captured {got:?}, components {want:?}");
        moving += (b.vx != 0.0 || b.vy != 0.0) as usize;
    }
    ensure!(moving > 0, "units are moving at the end");
    println!("{} rows captured per tick, {moving} moving at tick {TICKS}; digest {:016x} with and without", capture.len(), captured_digest.short());
    Ok(())
}
