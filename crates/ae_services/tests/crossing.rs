//! The nav service inside the sim (`games/crossing` on the water map).
//!
//! * The game's schedule places the host systems: layers are
//!   `nav_spawn | orders | nav | spatial churn | crowding`.
//! * Equal world digests (which include the nav service's own state) at
//!   1, 2, 4 and 8 sim workers, with the nav pool sized to match.
//! * The strongest check: the same plan driven through the standalone
//!   service (`ae_nav::scenario::water_crossing`) reaches the same nav state
//!   hash and bit-identical bodies. That catches gather/scatter, ordering and
//!   grouping mistakes that equal digests cannot.
//! * The guest's spatial queries match brute force on the final bodies.
//! * Units changing under the service, against a standalone service driven
//!   through the same changes: orders given at tick 20 (when every row also
//!   moves to another table, as `orders` adds `Crowd`), and at tick 150 a
//!   seventh of the units despawned and replaced in the same tick by new
//!   entities (recycled ids, which sort after every fresh one). The same
//!   world edits the terrain at ticks 100 (`quake`: rock and slow ground
//!   around some units, burying them) and 101 (raised ground around
//!   others, while the first edit's repair still runs); each edit lands with its nav repair a fixed delay
//!   later, the service reports it (`TerrainChanged`, recorded by `landed`),
//!   and the reference gets the same patch at the same step.
//!
//! The column arena is process-wide, so this file holds one test.

use ae_nav::math::WorldPos;
use ae_nav::scenario::{Plan, Scenario, water_crossing};
use ae_nav::service::{CommandKind, CommandPayload, UnitSpec};
use ae_sdk::nav::{Body, CELL_SIZE, NavAgent, TerrainEdit, mode, terrain};
use ae_services::nav::{Map, NavPlugin, terrain_patch};
use ae_services::spatial::SpatialPlugin;
use ae_sim::runtime::{Runtime, RuntimeConfig, World};
use anyhow::{Result, ensure};
use std::collections::BTreeMap;
use std::sync::Arc;

const UNITS: u32 = 1000;
const MAP_CELLS: i32 = 768;
const TICKS: u64 = 300;
/// The churned world: orders at tick `DELAY`, units replaced at tick `CHURN`.
const DELAY: u64 = 20;
const CHURN: u64 = 150;
/// The churned world's terrain edit tick.
const QUAKE: u64 = 100;
/// The churned world's second orders (stop, route or formation by group).
const REORDER: u64 = 60;
/// `games/crossing`'s `reorder`: new group ids, and guide points by kind.
const REGROUP: u64 = 1 << 40;
fn reorder_points(kind: u64, x: f32, y: f32) -> [(f32, f32); 2] {
    if kind == 1 { [(x - 160.0, y + 160.0), (x - 80.0, y + 80.0)] } else { [(x - 96.0, y - 96.0), (x + 96.0, y + 96.0)] }
}

/// Final world digest (low 64 bits), identical in dev and release builds and
/// at every worker count. Regenerate deliberately: run with `--nocapture`.
const EXPECTED: u64 = 0xf3b9_6c8f_5582_9aec;

fn decode<T: Copy>(bytes: &[u8]) -> T {
    assert_eq!(bytes.len(), size_of::<T>());
    unsafe { bytes.as_ptr().cast::<T>().read_unaligned() }
}

struct Outcome {
    digest: u128,
    nav: u64,
    /// Bodies in entity order (= spawn order = the plan's unit order).
    bodies: Vec<Body>,
    modes: [usize; 6],
}

fn run(rt: &mut Runtime, workers: usize) -> Result<(Outcome, World)> {
    let mut world = rt.world_for(UNITS as u64, workers)?;
    let mut host_ms = 0.0;
    let started = std::time::Instant::now();
    for _ in 0..TICKS {
        host_ms += rt.tick(&mut world, workers)?.host_ns as f64 / 1e6;
    }
    let wall = started.elapsed().as_secs_f64() * 1e3;
    let bodies: Vec<Body> = rt.rows(&world, "Body")?.iter().map(|(_, b)| decode(b)).collect();
    let mut modes = [0; 6];
    for (_, a) in rt.rows(&world, "NavAgent")? {
        modes[decode::<NavAgent>(&a).mode as usize] += 1;
    }
    let digest = rt.digest(&world)?.total;
    let nav = rt.host_digest(&world, "nav").expect("nav has a digest") as u64;
    eprintln!(
        "{workers} worker(s): digest {:016x}, nav {nav:016x}, {} bodies, modes idle/waiting/moving/arrived/blocked/unreachable {modes:?}; {:.2} ms/tick, host systems {:.2}",
        digest as u64,
        bodies.len(),
        wall / TICKS as f64,
        host_ms / TICKS as f64
    );
    Ok((Outcome { digest, nav, bodies, modes }, world))
}

#[test]
fn nav_runs_in_the_sim_like_the_standalone_service() -> Result<()> {
    let wasm = ae_sim::build::game("crossing")?;
    let nav = NavPlugin { map: Map::WaterCrossing, map_cells: MAP_CELLS };
    let spatial = SpatialPlugin::square(nav.extent(), 64.0);
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 8);
    let cfg = RuntimeConfig { max_workers: workers, range_rows: 128, parallel_min_rows: 256, ..RuntimeConfig::default() };
    let mut rt = Runtime::with_hosts(&wasm, cfg, vec![Arc::new(nav), Arc::new(spatial)])?;

    let layers: Vec<Vec<&str>> = rt.schedule.layers.iter().map(|l| l.iter().map(|&i| rt.system_name(i)).collect()).collect();
    assert_eq!(layers, [vec!["nav_spawn"], vec!["orders", "quake"], vec!["reorder"], vec!["nav"], vec!["spatial", "churn", "landed"], vec!["crowding"]]);

    // The standalone service on the same plan.
    let mut reference = water_crossing(UNITS, MAP_CELLS, 1);
    reference.run(TICKS, |_, _, _| {});
    let ref_hash = reference.nav.state_hash();

    let (base, world) = run(&mut rt, 1)?;
    ensure!(base.bodies.len() == reference.bodies.len(), "{} bodies in the sim, {} in the reference", base.bodies.len(), reference.bodies.len());
    for (i, b) in base.bodies.iter().enumerate() {
        let r = &reference.bodies;
        let same = [b.x, b.y, b.vx, b.vy].map(f32::to_bits) == [r.pos_x[i], r.pos_y[i], r.vel_x[i], r.vel_y[i]].map(f32::to_bits);
        ensure!(same, "unit {i}: sim body {b:?}, reference ({}, {}) v ({}, {})", r.pos_x[i], r.pos_y[i], r.vel_x[i], r.vel_y[i]);
    }
    ensure!(base.digest as u64 == EXPECTED, "digest {:016x} differs from the pinned {EXPECTED:016x}", base.digest as u64);
    ensure!(base.nav == ref_hash, "nav state hash {:016x} in the sim, {ref_hash:016x} standalone", base.nav);
    ensure!(base.modes[mode::MOVING as usize] + base.modes[mode::ARRIVED as usize] > UNITS as usize / 2, "most units should be moving or there: {:?}", base.modes);

    // Guest spatial queries against brute force, for the rows `crowding`
    // updated on the last tick (after nav moved the bodies; nothing moves
    // them later in the tick).
    let by_entity: BTreeMap<u64, Body> = rt.rows(&world, "Body")?.iter().map(|(e, b)| (*e, decode(b))).collect();
    let mut checked = 0;
    for (e, c) in rt.rows(&world, "Crowd")? {
        let (neighbours, tick) = (u32::from_le_bytes(c[0..4].try_into()?), u32::from_le_bytes(c[4..8].try_into()?));
        if tick as u64 != TICKS - 1 {
            continue;
        }
        let me = by_entity[&e];
        let r = me.radius * 4.0;
        let brute = by_entity.values().filter(|o| {
            let (dx, dy) = (o.x - me.x, o.y - me.y);
            dx * dx + dy * dy <= r * r
        });
        ensure!(neighbours as usize == brute.count() - 1, "entity {e:#x}: {neighbours} neighbours by the index");
        checked += 1;
    }
    ensure!(checked >= UNITS as usize / 5, "only {checked} crowd rows from the last tick");
    eprintln!("bodies and nav state equal to the standalone service; {checked} spatial queries equal to brute force");
    drop(world);

    for w in [2, 4, 8].into_iter().filter(|&w| w <= workers) {
        let (r, _) = run(&mut rt, w)?;
        ensure!(r.digest == base.digest, "{w} workers: digest {:032x}, 1 worker {:032x}", r.digest, base.digest);
        ensure!(r.nav == base.nav && r.modes == base.modes, "{w} workers: nav state differs");
    }

    let one = churned(&mut rt, 1)?;
    let many = churned(&mut rt, workers)?;
    ensure!(one.digest == many.digest && one.nav == many.nav, "churned world: 1 and {workers} workers differ");
    let mut reference = churned_reference(&nav, &one)?;
    ensure!(one.bodies.len() == reference.bodies.len(), "{} bodies in the sim, {} in the reference", one.bodies.len(), reference.bodies.len());
    for (i, b) in one.bodies.iter().enumerate() {
        let r = &reference.bodies;
        let same = [b.x, b.y, b.vx, b.vy].map(f32::to_bits) == [r.pos_x[i], r.pos_y[i], r.vel_x[i], r.vel_y[i]].map(f32::to_bits);
        ensure!(same, "churned world, unit {i}: sim body {b:?}, reference ({}, {})", r.pos_x[i], r.pos_y[i]);
    }
    let ref_hash = reference.nav.state_hash();
    ensure!(one.nav == ref_hash, "churned world: nav state hash {:016x} in the sim, {ref_hash:016x} standalone", one.nav);
    // The reorder reached the service: route and formation commands with
    // units, and stopped units standing idle.
    reference.nav.publish_snapshot();
    let mut kinds: BTreeMap<String, (usize, u32)> = BTreeMap::new();
    for c in &reference.nav.snapshot().unwrap().commands {
        let k = kinds.entry(c.kind.clone()).or_default();
        k.0 += 1;
        k.1 += c.total_units;
    }
    for kind in ["route", "formation"] {
        ensure!(kinds.get(kind).is_some_and(|k| k.1 > 0), "no {kind} command with units: {kinds:?}");
    }
    ensure!(one.modes[mode::IDLE as usize] > 0, "no stopped unit: modes {:?}", one.modes);
    eprintln!("churned world: commands (count, units) by kind {kinds:?}; final modes {:?}", one.modes);

    // One repair runs at a time: the second patch, queued while the first
    // was being repaired, lands with the next repair. One `TerrainChanged`
    // per repair, covering its patch, versions in order.
    let patches = [quake_patch(&reference, &one.quaked[0], false)?, quake_patch(&reference, &one.quaked[1], true)?];
    let rects: Vec<Cells> = patches.iter().map(|p| p.dirty_cells().unwrap()).collect();
    let landed: Vec<Cells> = one.landed.iter().map(|l| l.2).collect();
    ensure!(landed == rects, "reported rectangles {landed:?}, the patches cover {rects:?}");
    let last = reference.nav.terrain.version;
    let versions: Vec<u32> = one.landed.iter().map(|l| l.0).collect();
    ensure!(versions == [last - 1, last], "reported versions {versions:?}, the terrain is at {last}");
    ensure!(one.landed[0].1 > QUAKE && one.landed[1].1 > one.landed[0].1, "reported at ticks {:?}", one.landed.iter().map(|l| l.1).collect::<Vec<_>>());
    let stats = reference.nav.stats();
    ensure!(stats.nav.terrain_rebuilds == 2, "{} terrain repairs", stats.nav.terrain_rebuilds);
    ensure!(stats.nav.terrain_relocated_units > 0, "the quake buried no unit");
    eprintln!(
        "churned world: {} units replaced at tick {CHURN}, orders at tick {DELAY}, terrain edited at ticks {QUAKE} ({} cells) and {} ({} cells), landed as versions {versions:?} at ticks {} and {}, {} units moved out; equal to the standalone service, and at 1 and {workers} workers",
        one.replaced.len(),
        patches[0].cells.len(),
        QUAKE + 1,
        patches[1].cells.len(),
        one.landed[0].1,
        one.landed[1].1,
        stats.nav.terrain_relocated_units
    );
    Ok(())
}

/// Inclusive cell rectangle `(x0, y0, x1, y1)`.
type Cells = (i32, i32, i32, i32);

struct Churned {
    digest: u128,
    nav: u64,
    /// Final bodies by entity (= service order: fresh ids, then recycled).
    bodies: Vec<Body>,
    /// Sim entity of each plan unit.
    plan_entities: Vec<u64>,
    /// `(new entity, the entity it replaced)`, by new entity.
    replaced: Vec<(u64, u64)>,
    /// Units per `NavAgent::mode` at the end.
    modes: [usize; 6],
    /// Every body `quake` saw on its two ticks (as of the end of ticks
    /// `QUAKE - 1` and `QUAKE`), by entity.
    quaked: [Vec<(u64, Body)>; 2],
    /// `(version, tick, cells)` of each terrain version reported.
    landed: Vec<(u32, u64, Cells)>,
}

/// `games/crossing`'s `quake` edits (the game is a wasm-only crate) around
/// a point. First tick: slow ground (cost 40) 8 cells each way, rock 2 each
/// way. Second tick: ground raised to 90 m, 4 cells each way.
fn quake_edits(x: f32, y: f32, hill: bool) -> Vec<TerrainEdit> {
    let (cx, cy) = ((x / CELL_SIZE) as i32, (y / CELL_SIZE) as i32);
    let edit = |h: i32, set, blocked, height, cost| TerrainEdit { x0: cx - h, y0: cy - h, x1: cx + h, y1: cy + h, set, blocked, height, cost };
    if hill { vec![edit(4, terrain::HEIGHT, 0, 900, 0)] } else { vec![edit(8, terrain::COST, 0, 0, 40), edit(2, terrain::BLOCKED, 1, 0, 0)] }
}

/// `quake` edits around units whose entity index is a multiple of this
/// (first tick), or `HILL_OFFSET` more than one (second tick).
const QUAKE_EVERY: u32 = 97;
const HILL_OFFSET: u32 = 48;

/// The patch of one `quake` tick from the bodies it saw (edits from several
/// units commute: each tick's edits set one value per field everywhere, and
/// rock wins over cost whatever the order).
fn quake_patch(s: &Scenario, bodies: &[(u64, Body)], hill: bool) -> Result<ae_nav::terrain::TerrainPatch> {
    let k = if hill { HILL_OFFSET } else { 0 };
    let edits: Vec<_> = bodies.iter().filter(|(e, _)| *e as u32 % QUAKE_EVERY == k).flat_map(|(_, b)| quake_edits(b.x, b.y, hill)).collect();
    ensure!(edits.len() >= 4, "quake edits around too few units ({} edits)", edits.len());
    let geom = &s.nav.terrain.geom;
    terrain_patch(&edits, geom.cells_x, geom.cells_y)
}

fn churned(rt: &mut Runtime, workers: usize) -> Result<Churned> {
    let mut world = rt.world_for(UNITS as u64 | QUAKE << 24 | DELAY << 32 | REORDER << 40 | CHURN << 48, workers)?;
    let (mut plan_entities, mut replaced, mut quaked) = (Vec::new(), Vec::new(), [Vec::new(), Vec::new()]);
    for t in 0..TICKS {
        rt.tick(&mut world, workers)?;
        if t == 0 {
            plan_entities = rt.rows(&world, "Body")?.into_iter().map(|(e, _)| e).collect();
        }
        if t == QUAKE - 1 || t == QUAKE {
            quaked[(t + 1 - QUAKE) as usize] = rt.rows(&world, "Body")?.iter().map(|(e, b)| (*e, decode(b))).collect();
        }
        if t == CHURN {
            replaced = rt.rows(&world, "Replaces")?.into_iter().map(|(e, r)| (e, u64::from_le_bytes(r[..8].try_into().unwrap()))).collect();
        }
    }
    ensure!(replaced.len() > UNITS as usize / 10, "churn replaced only {} units", replaced.len());
    // By version (rows come in entity order, and ids are recycled).
    let mut landed: Vec<(u32, u64, Cells)> = rt
        .rows(&world, "Landed")?
        .iter()
        .map(|(_, l)| {
            let w: [i32; 6] = decode(l);
            (w[0] as u32, w[1] as u64, (w[2], w[3], w[4], w[5]))
        })
        .collect();
    landed.sort_unstable_by_key(|l| l.0);
    ensure!(replaced.iter().all(|(new, _)| *new > *plan_entities.last().unwrap()), "replacements should have recycled ids above every fresh one");
    Ok(Churned {
        digest: rt.digest(&world)?.total,
        nav: rt.host_digest(&world, "nav").unwrap() as u64,
        bodies: rt.rows(&world, "Body")?.iter().map(|(_, b)| decode(b)).collect(),
        plan_entities,
        replaced,
        quaked,
        modes: {
            let mut modes = [0; 6];
            for (_, a) in rt.rows(&world, "NavAgent")? {
                modes[decode::<NavAgent>(&a).mode as usize] += 1;
            }
            modes
        },
        landed,
    })
}

/// The churned world's plan through the standalone service: no orders at
/// first, the plan's orders before step `DELAY + 1` (the sim's tick
/// `DELAY`), and the sim's replacements before step `CHURN + 2` (the sim's
/// nav sees them on tick `CHURN + 1`): removals in one batch, then new units
/// in ascending sim-entity order, each with the body of the unit it replaces;
/// the quake's patches before steps `QUAKE + 1` and `QUAKE + 2` (the sim's
/// nav runs after `quake`, in the same tick).
fn churned_reference(plugin: &NavPlugin, sim: &Churned) -> Result<Scenario> {
    let (nav, plan) = plugin.plan(UNITS, 1);
    let mut s = Scenario::from_plan(nav, &Plan { units: plan.units.clone(), orders: Vec::new() });
    // Scenario entity of plan unit k is k + 1.
    let unit_of = |sim_entity: u64| sim.plan_entities.binary_search(&sim_entity).map(|k| k as u64 + 1);
    for t in 0..TICKS {
        if t == DELAY {
            for (id, units, to) in &plan.orders {
                let mut unit_ids: Vec<u32> = units.iter().map(|&k| s.nav.local_id(k as u64 + 1).unwrap()).collect();
                unit_ids.sort_unstable();
                s.nav.queue_command_at(CommandPayload { id: *id, unit_ids, kind: CommandKind::MoveTo, requested_point: *to, corridor_tolerance_world: 48.0 });
            }
        }
        if t == REORDER {
            // A group's first order is its first unit's, so groups queue in
            // plan order (rows are in spawn order).
            for (id, units, to) in &plan.orders {
                let mut unit_ids: Vec<u32> = units.iter().map(|&k| s.nav.local_id(k as u64 + 1).unwrap()).collect();
                unit_ids.sort_unstable();
                let guides = |k| reorder_points(k, to.x, to.y).map(|(x, y)| WorldPos::new(x, y)).to_vec();
                let kind = match id % 3 {
                    0 => CommandKind::Stop,
                    1 => CommandKind::FollowCurve { guide_points: guides(1) },
                    _ => CommandKind::CurveFormation { guide_points: guides(2) },
                };
                s.nav.queue_command_at(CommandPayload { id: id + REGROUP, unit_ids, kind, requested_point: *to, corridor_tolerance_world: 48.0 });
            }
        }
        if t == QUAKE || t == QUAKE + 1 {
            let hill = t == QUAKE + 1;
            let patch = quake_patch(&s, &sim.quaked[hill as usize], hill)?;
            s.nav.queue_terrain_patch(patch);
        }
        if t == CHURN + 1 {
            let order: Vec<u64> = s.nav.entities().collect();
            let mut olds = Vec::new();
            for &(_, old) in &sim.replaced {
                olds.push(unit_of(old).map_err(|_| anyhow::anyhow!("replaced entity {old:#x} is not a plan unit"))?);
            }
            // Bodies of the replaced units, in service order, before removal.
            let row = |e: u64| order.iter().position(|&x| x == e).unwrap();
            let b = &s.bodies;
            let born: Vec<(UnitSpec, f32, f32)> = olds
                .iter()
                .map(|&e| {
                    let (i, p) = (row(e), plan.units[e as usize - 1]);
                    (UnitSpec { pos: WorldPos::new(b.pos_x[i], b.pos_y[i]), ..p }, b.vel_x[i], b.vel_y[i])
                })
                .collect();
            let keep: Vec<bool> = order.iter().map(|e| !olds.contains(e)).collect();
            let b = &mut s.bodies;
            for col in [&mut b.pos_x, &mut b.pos_y, &mut b.vel_x, &mut b.vel_y, &mut b.radius] {
                let mut k = keep.iter();
                col.retain(|_| *k.next().unwrap());
            }
            let mut k = keep.iter();
            b.profile.retain(|_| *k.next().unwrap());
            s.nav.remove_units(&olds);
            for (j, (spec, vx, vy)) in born.into_iter().enumerate() {
                s.nav.add_unit(1_000_000 + j as u64, spec);
                let b = &mut s.bodies;
                b.pos_x.push(spec.pos.x);
                b.pos_y.push(spec.pos.y);
                b.vel_x.push(vx);
                b.vel_y.push(vy);
                b.radius.push(spec.radius);
                b.profile.push(spec.profile_id);
            }
        }
        s.step();
    }
    Ok(s)
}
