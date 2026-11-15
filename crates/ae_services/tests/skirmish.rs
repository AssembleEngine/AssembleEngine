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
//! * Two bots play a whole match (`ae_bot::SkirmishBot`): units gather at
//!   their rally points, the bots attack-move waves at the enemy base, and
//!   the match ends with a winner at a pinned tick and digest (the golden),
//!   the same at 1 and 2 workers.
//!
//! One runtime at a time: the column arena is process-wide.

use ae_bot::SkirmishBot;
use ae_catalog::{Catalog, def_id};
use ae_services::catalog::CatalogPlugin;
use ae_services::nav::{Map, NavPlugin};
use ae_services::spatial::SpatialPlugin;
use ae_services::view::ViewPipeline;
use ae_sim::input::TickBundle;
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

/// The tests take turns (one runtime at a time).
static ONE_RUNTIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn decode<T: Copy>(bytes: &[u8]) -> T {
    unsafe { bytes.as_ptr().cast::<T>().read_unaligned() }
}

#[test]
fn defs_spawn_with_their_values_and_team() -> Result<()> {
    let _one = ONE_RUNTIME.lock().unwrap_or_else(|p| p.into_inner());
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

    // A whole match, bot against bot (`SkirmishBot::pair`): allyteam 0's
    // bot attacks in waves of 8, allyteam 1's in waves of 16. Same winner,
    // tick and digest at 1 and 2 workers, and the pinned golden (debug and
    // release builds agree). A gameplay change moves it: re-pin it on
    // purpose and say so in the commit.
    let decisive = |workers: usize| -> Result<(u32, u64, u128)> {
        let cfg = RuntimeConfig { max_workers: 2, ..RuntimeConfig::default() };
        let mut rt = Runtime::with_hosts(&wasm, cfg, vec![Arc::new(nav.clone()), Arc::new(SpatialPlugin::square(nav.extent(), 64.0)), Arc::new(CatalogPlugin::new(catalog.clone()))])?;
        let mut world = rt.world_for(param, workers)?;
        let bots = SkirmishBot::pair(&rt)?;
        // Attack-moves engage: unit-ticks chasing and holding, and returns
        // from either to heading on.
        let (mut engaged, mut states) = ([0u64; 3], BTreeMap::<u64, u32>::new());
        for _ in 0..12000 {
            let mut bundle = TickBundle::new(world.tick);
            for b in &bots {
                b.commands(&rt, &world, &mut bundle.commands)?;
            }
            rt.tick_with(&mut world, workers, &bundle)?;
            for (e, q) in rt.rows(&world, "Queue")? {
                let state: u32 = decode(&q[52..]);
                match state {
                    3 => engaged[0] += 1,
                    4 => engaged[1] += 1,
                    2 if matches!(states.get(&e), Some(3 | 4)) => engaged[2] += 1,
                    _ => {}
                }
                states.insert(e, state);
            }
            let m: [u32; 4] = decode(&rt.rows(&world, "Match")?[0].1);
            if m[0] != u32::MAX && world.tick > m[2] as u64 + 30 {
                ensure!(engaged.iter().all(|&n| n > 0), "chase, hold and resume unit-ticks: {engaged:?}");
                println!("engaged: {} unit-ticks chasing, {} holding, {} resumed", engaged[0], engaged[1], engaged[2]);
                return Ok((m[0], m[2] as u64, rt.digest(&world)?.total));
            }
        }
        Ok((u32::MAX, 0, rt.digest(&world)?.total))
    };
    let (winner, at, d) = decisive(2)?;
    println!("bot match: winner {winner:#x} at tick {at}, digest {:032x}", d);
    ensure!((winner, at, d) == (1, 6363, 0x40c9_e1b9_faa8_9d3b_33ab_7d8a_ee5e_b74e), "the golden bot match: winner {winner:#x} at tick {at}, digest {d:032x}");
    ensure!(decisive(1)? == (winner, at, d), "the match differs at 1 worker");
    println!("{} spawned per allyteam, own rows in both views; digest {:016x}", 1 + TANKS, d2 as u64);
    Ok(())
}

/// `UnitCommand`s into a tank's `Queue`: a move replaces the rally order,
/// appended commands queue behind it, the head is popped on arrival, and a
/// stop clears the queue.
#[test]
fn unit_commands_queue_and_advance() -> Result<()> {
    let _one = ONE_RUNTIME.lock().unwrap_or_else(|p| p.into_inner());
    let wasm = ae_sim::build::game("skirmish")?;
    let schema = ae_sim::schema::Schema::from_module(&wasm)?;
    let mut vfs = ae_vfs::Vfs::new();
    vfs.mount_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../games/skirmish/content"))?;
    let catalog = Arc::new(Catalog::load(&vfs, "defs", &schema)?);
    let nav = NavPlugin::new(Map::LandCrossing, (EXTENT / 8) as i32);
    let mut rt = Runtime::with_hosts(&wasm, RuntimeConfig::default(), vec![Arc::new(nav.clone()), Arc::new(SpatialPlugin::square(nav.extent(), 64.0)), Arc::new(CatalogPlugin::new(catalog))])?;
    let mut world = rt.world_for(EXTENT << 24 | TANKS << 40, 1)?;
    let stream = rt.schema.components.iter().find(|c| c.name == "UnitCommand").unwrap().id;
    for _ in 0..5 {
        rt.tick(&mut world, 1)?;
    }
    // Queue: kind[4], x[4], y[4], len, state, ...
    let queue = |rt: &Runtime, w: &ae_sim::runtime::World, e: u64| -> Result<(u32, u32, [f32; 4])> {
        let b = rt.rows(w, "Queue")?.into_iter().find(|r| r.0 == e).map(|r| r.1).unwrap();
        Ok((decode(&b[48..]), decode(&b[52..]), decode(&b[16..])))
    };
    let tank = rt.rows(&world, "Queue")?[0].0;
    let (len, state, _) = queue(&rt, &world, tank)?;
    ensure!(len == 1 && state == 2, "a new tank heads for its rally point: len {len}, state {state}");
    let command = |kind: u32, append: bool, x: f32, y: f32| ae_bot::unit_command(0, stream, tank, kind, append, x, y);
    let (bx, by) = ae_bot::base(world.param, 0);
    let mut bundle = TickBundle::new(world.tick);
    bundle.commands = vec![command(ae_bot::MOVE, false, bx + 40.0, by), command(ae_bot::MOVE, true, bx + 600.0, by + 600.0), command(ae_bot::ATTACK, true, bx, by)];
    rt.tick_with(&mut world, 1, &bundle)?;
    let (len, state, xs) = queue(&rt, &world, tank)?;
    ensure!(len == 3 && state == 2 && xs[..3] == [bx + 40.0, bx + 600.0, bx], "replaced, then two appended: len {len}, state {state}, x {xs:?}");
    // 40 m away: arrives, and the queue moves on to the far point.
    let mut advanced = false;
    for _ in 0..300 {
        rt.tick(&mut world, 1)?;
        let (len, _, xs) = queue(&rt, &world, tank)?;
        if len == 2 {
            ensure!(xs[0] == bx + 600.0, "the next head is the far point: {xs:?}");
            advanced = true;
            break;
        }
    }
    ensure!(advanced, "the tank never arrived at its first point");
    let mut bundle = TickBundle::new(world.tick);
    bundle.commands = vec![command(ae_bot::STOP, false, 0.0, 0.0)];
    rt.tick_with(&mut world, 1, &bundle)?;
    let (len, state, _) = queue(&rt, &world, tank)?;
    ensure!(len == 0 && state == 1, "stopped: len {len}, state {state}");
    Ok(())
}
