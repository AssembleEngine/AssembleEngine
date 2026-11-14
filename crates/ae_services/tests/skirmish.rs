//! The skirmish game spawning from its content: the catalogue
//! (`games/skirmish/content/defs/`) is loaded through the VFS, checked
//! against the game's schema, and the catalogue service spawns the
//! `SpawnRequest`s of the game's `start` system.
//! * Each allyteam gets one factory at its base and its tanks around it,
//!   with the def's values (health, radius, sensors, nav agent) and its
//!   `Team`.
//! * They are views' `Own` rows for their allyteam (the gap M4 left for
//!   game-spawned units is closed); the other base is far out of sight.
//! * The world is the same at 1 and 2 workers.
//! * A match with a handicap (30 more tanks for allyteam 0) ends with
//!   allyteam 0 the winner, at the same tick and digest at 1 and 2 workers.
//!   (An even match stalls: the armies meet in a corridor and only their
//!   fronts fight, while both factories keep building.)
//!
//! One runtime at a time: the column arena is process-wide.

use ae_catalog::{Catalog, def_id};
use ae_services::catalog::CatalogPlugin;
use ae_services::nav::{Map, NavPlugin};
use ae_services::spatial::SpatialPlugin;
use ae_services::view::ViewPipeline;
use ae_sim::runtime::{Runtime, RuntimeConfig};
use ae_view::{Observer, RowState};
use anyhow::{Result, ensure};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

const TANKS: u64 = 6;
const EXTENT: u64 = 2048;

/// Rows of a component: `(entity, bytes)`.
type Rows = Vec<(u64, Vec<u8>)>;
/// Digest, `Def`, `Team` and `Body` rows, both observers' views, the pipeline.
type Outcome = (u128, Rows, Rows, Rows, Vec<ae_view::ViewTable>, ViewPipeline);

fn decode<T: Copy>(bytes: &[u8]) -> T {
    unsafe { bytes.as_ptr().cast::<T>().read_unaligned() }
}

#[test]
fn defs_spawn_with_their_values_and_team() -> Result<()> {
    let wasm = ae_sim::build::game("skirmish")?;
    let schema = ae_sim::schema::Schema::from_module(&wasm)?;
    let mut vfs = ae_vfs::Vfs::new();
    vfs.mount_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../games/skirmish/content"))?;
    let catalog = Arc::new(Catalog::load(&vfs, "defs", &schema)?);
    let text = vfs.read_string("defs/units.json")?;
    ensure!(ae_catalog::CatalogFile::parse(&text)?.write() == text, "the content file is in canonical form");
    let factory_def = catalog.get("factory").unwrap().component("Factory").unwrap();
    ensure!(factory_def.bytes[..8] == def_id("tank").to_le_bytes(), "`\"builds\": \"tank\"` is the tank def's id");
    ensure!(catalog.defs.len() == 2, "factory and tank (the rest are abstract): {:?}", catalog.defs.values().map(|d| &d.name).collect::<Vec<_>>());
    let nav = NavPlugin::new(Map::LandCrossing, (EXTENT / 8) as i32);
    let param = EXTENT << 24 | TANKS << 40;
    let run = |workers: usize| -> Result<Outcome> {
        let cfg = RuntimeConfig { max_workers: 2, ..RuntimeConfig::default() };
        let mut rt = Runtime::with_hosts(&wasm, cfg, vec![Arc::new(nav.clone()), Arc::new(SpatialPlugin::square(nav.extent(), 64.0)), Arc::new(CatalogPlugin::new(catalog.clone()))])?;
        let mut world = rt.world_for(param, workers)?;
        for _ in 0..10 {
            rt.tick(&mut world, workers)?;
        }
        let mut pipeline = ViewPipeline::new(&rt.schema, &["Def", "Health"], 2, nav.extent())?;
        pipeline.capture(&rt, &world)?;
        let tables = pipeline.observe(&mut [Observer::new(0, 1), Observer::new(1, 2)]);
        Ok((rt.digest(&world)?.total, rt.rows(&world, "Def")?, rt.rows(&world, "Team")?, rt.rows(&world, "Body")?, tables, pipeline))
    };
    let (d2, defs, teams, bodies, tables, _) = run(2)?;
    let teams: BTreeMap<u64, u32> = teams.into_iter().map(|(e, b)| (e, decode::<u32>(&b))).collect();
    let bodies: BTreeMap<u64, [f32; 5]> = bodies.into_iter().map(|(e, b)| (e, decode(&b))).collect();
    let mut count: BTreeMap<(u64, u32), usize> = BTreeMap::new();
    for (e, b) in &defs {
        let def: u64 = decode(b);
        let ally = teams[e];
        *count.entry((def, ally)).or_default() += 1;
        let [x, y, _, _, radius] = bodies[e];
        let base = if ally == 0 { 0.15 * EXTENT as f32 } else { 0.85 * EXTENT as f32 };
        let d = (x - base).hypot(y - base);
        if def == def_id("factory") {
            ensure!(d < 1.0 && radius == 20.0, "factory of {ally} at ({x}, {y}), radius {radius}");
        } else {
            ensure!(d < 130.0 && radius == 4.0, "tank of {ally} {d:.0} m from base, radius {radius}");
        }
    }
    for ally in 0..2 {
        ensure!(count.get(&(def_id("factory"), ally)) == Some(&1) && count.get(&(def_id("tank"), ally)) == Some(&(TANKS as usize)), "allyteam {ally}: {count:?}");
    }
    for (o, t) in tables.iter().enumerate() {
        let own = t.state.iter().filter(|s| **s == RowState::Own).count();
        ensure!(own == 1 + TANKS as usize && t.len() == own, "observer {o}: {own} own rows of {} (the other base is out of sight)", t.len());
    }
    let (d1, ..) = run(1)?;
    ensure!(d1 == d2, "1 and 2 workers differ");

    // A whole match with a handicap (allyteam 0 starts with 30 more tanks):
    // its army breaks through and destroys the other factory; the judge
    // names it the winner. The same at 1 and 2 workers.
    let decisive = |workers: usize| -> Result<(u32, u64, u128)> {
        let cfg = RuntimeConfig { max_workers: 2, ..RuntimeConfig::default() };
        let mut rt = Runtime::with_hosts(&wasm, cfg, vec![Arc::new(nav.clone()), Arc::new(SpatialPlugin::square(nav.extent(), 64.0)), Arc::new(CatalogPlugin::new(catalog.clone()))])?;
        let mut world = rt.world_for(param | 30 << 48, workers)?;
        for _ in 0..6000 {
            rt.tick(&mut world, workers)?;
            let m: [u32; 4] = decode(&rt.rows(&world, "Match")?[0].1);
            if m[0] != u32::MAX && world.tick > m[2] as u64 + 30 {
                return Ok((m[0], m[2] as u64, rt.digest(&world)?.total));
            }
        }
        Ok((u32::MAX, 0, rt.digest(&world)?.total))
    };
    let (winner, at, d) = decisive(2)?;
    ensure!(winner == 0, "allyteam 0 (with 30 more tanks) wins: winner {winner:#x} at tick {at}");
    ensure!(decisive(1)? == (winner, at, d), "the match differs at 1 worker");
    println!("handicap match: allyteam {winner} won at tick {at}");
    println!("{} spawned per allyteam, own rows in both views; digest {:016x}", 1 + TANKS, d2 as u64);
    Ok(())
}
