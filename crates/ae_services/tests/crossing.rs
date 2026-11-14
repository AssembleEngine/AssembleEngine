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
//!   entities (recycled ids, which sort after every fresh one).
//!
//! The column arena is process-wide, so this file holds one test.

use ae_nav::math::WorldPos;
use ae_nav::scenario::{Plan, Scenario, water_crossing};
use ae_nav::service::{CommandKind, CommandPayload, UnitSpec};
use ae_sdk::nav::{Body, NavAgent, mode};
use ae_services::nav::{Map, NavPlugin};
use ae_services::spatial::SpatialPlugin;
use ae_sim::runtime::{Runtime, RuntimeConfig, World};
use anyhow::{Result, ensure};
use std::sync::Arc;

const UNITS: u32 = 1000;
const MAP_CELLS: i32 = 768;
const TICKS: u64 = 300;
/// The churned world: orders at tick `DELAY`, units replaced at tick `CHURN`.
const DELAY: u64 = 20;
const CHURN: u64 = 150;

/// Final world digest (low 64 bits), identical in dev and release builds and
/// at every worker count. Regenerate deliberately: run with `--nocapture`.
const EXPECTED: u64 = 0x45c0_5804_23ad_9cec;

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
    assert_eq!(layers, [vec!["nav_spawn"], vec!["orders"], vec!["nav"], vec!["spatial", "churn"], vec!["crowding"]]);

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
    let by_entity: std::collections::BTreeMap<u64, Body> = rt.rows(&world, "Body")?.iter().map(|(e, b)| (*e, decode(b))).collect();
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
    let reference = churned_reference(&nav, &one)?;
    ensure!(one.bodies.len() == reference.bodies.len(), "{} bodies in the sim, {} in the reference", one.bodies.len(), reference.bodies.len());
    for (i, b) in one.bodies.iter().enumerate() {
        let r = &reference.bodies;
        let same = [b.x, b.y, b.vx, b.vy].map(f32::to_bits) == [r.pos_x[i], r.pos_y[i], r.vel_x[i], r.vel_y[i]].map(f32::to_bits);
        ensure!(same, "churned world, unit {i}: sim body {b:?}, reference ({}, {})", r.pos_x[i], r.pos_y[i]);
    }
    let ref_hash = reference.nav.state_hash();
    ensure!(one.nav == ref_hash, "churned world: nav state hash {:016x} in the sim, {ref_hash:016x} standalone", one.nav);
    eprintln!("churned world: {} units replaced at tick {CHURN}, orders at tick {DELAY}; equal to the standalone service, and at 1 and {workers} workers", one.replaced.len());
    Ok(())
}

struct Churned {
    digest: u128,
    nav: u64,
    /// Final bodies by entity (= service order: fresh ids, then recycled).
    bodies: Vec<Body>,
    /// Sim entity of each plan unit.
    plan_entities: Vec<u64>,
    /// `(new entity, the entity it replaced)`, by new entity.
    replaced: Vec<(u64, u64)>,
}

fn churned(rt: &mut Runtime, workers: usize) -> Result<Churned> {
    let mut world = rt.world_for(UNITS as u64 | DELAY << 32 | CHURN << 48, workers)?;
    let (mut plan_entities, mut replaced) = (Vec::new(), Vec::new());
    for t in 0..TICKS {
        rt.tick(&mut world, workers)?;
        if t == 0 {
            plan_entities = rt.rows(&world, "Body")?.into_iter().map(|(e, _)| e).collect();
        }
        if t == CHURN {
            replaced = rt.rows(&world, "Replaces")?.into_iter().map(|(e, r)| (e, u64::from_le_bytes(r[..8].try_into().unwrap()))).collect();
        }
    }
    ensure!(replaced.len() > UNITS as usize / 10, "churn replaced only {} units", replaced.len());
    ensure!(replaced.iter().all(|(new, _)| *new > *plan_entities.last().unwrap()), "replacements should have recycled ids above every fresh one");
    Ok(Churned {
        digest: rt.digest(&world)?.total,
        nav: rt.host_digest(&world, "nav").unwrap() as u64,
        bodies: rt.rows(&world, "Body")?.iter().map(|(_, b)| decode(b)).collect(),
        plan_entities,
        replaced,
    })
}

/// The churned world's plan through the standalone service: no orders at
/// first, the plan's orders before step `DELAY + 1` (the sim's tick
/// `DELAY`), and the sim's replacements before step `CHURN + 2` (the sim's
/// nav sees them on tick `CHURN + 1`): removals in one batch, then new units
/// in ascending sim-entity order, each with the body of the unit it replaces.
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
