//! Headless, reproducible navigation benchmarks.
//!
//! Every scenario uses the fixed-seed default map (2560 x 2560 cells) unless
//! stated otherwise, issues commands at fixed ticks and runs the authoritative
//! 30 Hz simulation as fast as possible. Wall-clock numbers are measurements on
//! the machine running the benchmark; tick counts are deterministic.
//!
//! Usage:
//!   navbench [--quick] [--workers N] [--nav-workers N] [--sector N] [--only NAME[,NAME]] [--json PATH]
//!   navbench --scaling [--only ...]   # worker-count scaling + determinism check
//!   navbench --build-scaling          # hierarchy build / local repair vs threads
//!   navbench --replay FILE.json       # replay a recorded session at 1..24 workers
use ae_nav::reference::commands::{CommandKind, CommandPayload};
use ae_nav::math::{CellPos, WorldPos};
use ae_nav::reference::sim::{brush_patch, SimConfig, SimStats, Simulation, TickPhases, UnitMode};
use serde::Serialize;
use std::time::Instant;

const CELL: f32 = 8.0;

fn wp(cx: f32, cy: f32) -> WorldPos { WorldPos::new((cx + 0.5) * CELL, (cy + 0.5) * CELL) }

fn payload(id: u64, ids: Vec<u32>, kind: CommandKind, to: WorldPos) -> CommandPayload {
    CommandPayload { id, unit_ids: ids, kind, requested_point: to, corridor_tolerance_world: 48.0 }
}

#[derive(Clone, Debug)]
struct Opts {
    quick: bool,
    workers: usize,
    nav_workers: usize,
    sector: i32,
    only: Vec<String>,
    json: Option<String>,
    scaling: bool,
    build_scaling: bool,
    context_timing: bool,
    replay: Option<String>,
    no_avoidance: bool,
    yield_rule: bool,
    tau: Option<f32>,
    export_detour: Option<String>,
    /// Pace ticks at the real tick rate (as the app does) instead of
    /// running them back to back.
    realtime: bool,
    /// Print the phases of every tick slower than this (ms).
    slow_ticks: Option<f64>,
}

#[derive(Serialize, Default, Clone)]
struct Pct { median: f64, p95: f64, max: f64 }

fn pct(v: &mut [f64]) -> Pct {
    if v.is_empty() { return Pct::default(); }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |p: f64| v[(((v.len() - 1) as f64) * p).round() as usize];
    Pct { median: at(0.5), p95: at(0.95), max: *v.last().unwrap() }
}

#[derive(Serialize, Clone)]
struct ScenarioResult {
    name: String,
    units: usize,
    workers: usize,
    sector_cells: i32,
    ticks: u64,
    setup_ms: f64,
    wall_ms: f64,
    tick_ms: Pct,
    arrived: usize,
    /// Arrived via the settle policy (inside the arrival region, not at the
    /// exact slot). Included in `arrived`.
    settled: usize,
    unreachable: usize,
    moving_at_end: usize,
    /// Fraction of units that were moving (nav-driven) averaged over the run.
    mean_moving_fraction: f64,
    /// Tick at which 50% / 90% of commanded units had finished (arrived or
    /// reported unreachable), if reached.
    t50_ticks: Option<u64>,
    t90_ticks: Option<u64>,
    latency_ticks_median: f64,
    latency_ticks_p95: f64,
    latency_ticks_worst: u64,
    latency_ms_median: f64,
    latency_ms_p95: f64,
    latency_ms_worst: f64,
    jobs_completed: u64,
    plans: u64,
    fields_built: u64,
    field_expansions: u64,
    abstract_expansions: u64,
    nav_busy_ms: f64,
    cache_hit_rate: f64,
    cache_fields: usize,
    nav_memory_mb: f64,
    peak_destination_groups: usize,
    collision_candidates_per_tick: f64,
    contacts_per_tick: f64,
    solver_ms_per_tick: f64,
    avoidance_ms_per_tick: f64,
    /// Mean wall time per tick of each `advance_tick` phase.
    #[serde(default)]
    phases_ms_per_tick: TickPhases,
    /// Longest wait for due worker results in one tick (the job phase).
    #[serde(default)]
    jobs_ms_max: f64,
    /// Ticks whose `advance_tick` took longer than one tick period.
    #[serde(default)]
    ticks_over_budget: u64,
    #[serde(default)]
    realtime: bool,
    terrain_repairs: u64,
    repair_latency_ticks_max: u64,
    repair_ms_max: f64,
    stuck_settled: u64,
    skipped_stages: u64,
    worker_utilization: f64,
    hierarchy_build_ms: f64,
    stuck_gave_up: u64,
    final_hash: String,
    /// Commanded units that finished (arrived or unreachable) by the end.
    #[serde(default)]
    commanded_finished: usize,
    #[serde(default)]
    commanded: usize,
    /// Distance movers travelled before finishing over the straight
    /// distance from their start to the requested point.
    #[serde(default)]
    path_ratio: f64,
    /// How far the standing crowd (`Script::watch`) was pushed, in cells.
    #[serde(default)]
    watch_displacement_cells: WatchDisplacement,
    /// Commanded units blocked (no progress, `UnitMode::Blocked`) for more
    /// than 30 s in total.
    #[serde(default)]
    blocked_over_30s: usize,
    /// On an arena with walls: commanded units that crossed the arena's
    /// middle column more than 32 cells from its middle row (through a
    /// detour gap rather than the direct one).
    #[serde(default)]
    crossed_off_line: usize,
    /// Of those, the ones that crossed above the middle row (with two gaps
    /// either side of the line: how the army split).
    #[serde(default)]
    crossed_above: usize,
    /// Crowd-map changes that re-keyed a cohort's routes.
    #[serde(default)]
    crowd_rekeys: u64,
}

#[derive(Serialize, Default, Clone)]
struct WatchDisplacement { mean: f64, cells: Pct, over_2_cells: usize }

struct Scenario {
    name: &'static str,
    units: u32,
    ticks: u64,
    setup: fn(&mut Simulation, &mut Script),
    /// Flat open arena of this many cells (8-unit cells, cost 10, no
    /// height) instead of the default map.
    arena: Option<(i32, i32)>,
    /// Rock rectangles (x0, y0, x1, y1, inclusive cells) on the arena.
    walls: &'static [(i32, i32, i32, i32)],
    /// Run only when named with `--only`.
    opt_in: bool,
    /// Spawn as the app does (`SimConfig` default radius) instead of
    /// widening the spawn disc with the unit count.
    app_spawn: bool,
}

/// Inputs scheduled for future ticks (applied before that tick's step).
#[derive(Default)]
struct Script {
    commands: Vec<(u64, CommandPayload)>,
    edits: Vec<(u64, ae_nav::terrain::TerrainPatch)>,
    /// Uncommanded units whose displacement is reported (a standing crowd
    /// that movers push through or around).
    watch: Vec<u32>,
}

fn all_ids(sim: &Simulation) -> Vec<u32> { sim.units.iter().map(|u| u.id).collect() }

/// Place units on a lattice around a cell before tick 1. This is scenario
/// setup only; the simulation never teleports units.
fn place_block(sim: &mut Simulation, ids: &[u32], center: (i32, i32), spacing_cells: f32) {
    let geom = sim.hierarchy.terrain.geom.clone();
    let side = (ids.len() as f32).sqrt().ceil() as i32;
    let mut k = 0usize;
    let mut ring = 0i32;
    let mut used = std::collections::BTreeSet::new();
    while k < ids.len() && ring < 400 {
        for gy in -ring..=ring {
            for gx in -ring..=ring {
                if k >= ids.len() { break; }
                if gx.abs() != ring && gy.abs() != ring { continue; }
                if !used.insert((gx, gy)) { continue; }
                let p = WorldPos::new(
                    (center.0 as f32 + 0.5 + gx as f32 * spacing_cells) * geom.cell_size,
                    (center.1 as f32 + 0.5 + gy as f32 * spacing_cells) * geom.cell_size,
                );
                let c = geom.world_to_cell(p);
                let idx = (ids[k] - 1) as usize;
                let pid = sim.units[idx].profile_id;
                if !geom.in_bounds(c) || !sim.hierarchy.grid(pid).is_passable(&sim.terrain, c) { continue; }
                sim.place_unit(idx, p);
                k += 1;
            }
        }
        ring += 1;
    }
    let _ = side;
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario { name: "large_group_cross_map", units: 1000, ticks: 6000, setup: |sim, s| {
            // ~1,300 cells (10,400 world units) from the spawn plaza.
            s.commands.push((1, payload(1, all_ids(sim), CommandKind::MoveTo, wp(1400.0, 1560.0))));
        }, arena: None, walls: &[], opt_in: false, app_spawn: false },
        Scenario { name: "many_groups_unrelated", units: 1000, ticks: 4500, setup: |sim, s| {
            let ids = all_ids(sim);
            let dests = [(700, 2100), (1020, 1900), (400, 1700), (1400, 2300), (900, 1500), (300, 1200),
                (1200, 1250), (1600, 1900), (600, 2450), (1100, 2450), (1900, 2400), (200, 900),
                (800, 1100), (1500, 1450), (500, 2000), (1300, 1750)];
            for (g, chunk) in ids.chunks(ids.len().div_ceil(16)).enumerate() {
                let d = dests[g % dests.len()];
                s.commands.push((1, payload(g as u64 + 1, chunk.to_vec(), CommandKind::MoveTo, wp(d.0 as f32, d.1 as f32))));
            }
        }, arena: None, walls: &[], opt_in: false, app_spawn: false },
        Scenario { name: "frequent_replacement", units: 1000, ticks: 1800, setup: |sim, s| {
            let ids = all_ids(sim);
            let dests = [(700, 2100), (400, 1800), (900, 2350), (600, 1900), (1020, 1900), (500, 2200)];
            for k in 0..12u64 {
                let d = dests[(k as usize) % dests.len()];
                s.commands.push((1 + k * 15, payload(k + 1, ids.clone(), CommandKind::MoveTo, wp(d.0 as f32, d.1 as f32))));
            }
        }, arena: None, walls: &[], opt_in: false, app_spawn: false },
        Scenario { name: "slight_destination_changes", units: 1000, ticks: 2400, setup: |sim, s| {
            let ids = all_ids(sim);
            for k in 0..10u64 {
                s.commands.push((1 + k * 20, payload(k + 1, ids.clone(), CommandKind::MoveTo, wp(700.0 + k as f32 * 2.0, 2100.0 - k as f32))));
            }
        }, arena: None, walls: &[], opt_in: false, app_spawn: false },
        Scenario { name: "follow_curve_around_obstacles", units: 1000, ticks: 4500, setup: |sim, s| {
            // The shortest route east crosses the x=820 wall through the road
            // gap near y=1975. The drawn curve instead swings south around the
            // wall's end (y=2360), so following it is observable.
            let guides = vec![wp(260.0, 2290.0), wp(450.0, 2420.0), wp(820.0, 2450.0), wp(950.0, 2350.0), wp(1000.0, 2100.0)];
            s.commands.push((1, payload(1, all_ids(sim), CommandKind::FollowCurve { guide_points: guides }, wp(1000.0, 2100.0))));
        }, arena: None, walls: &[], opt_in: false, app_spawn: false },
        Scenario { name: "curve_formation", units: 1000, ticks: 3000, setup: |sim, s| {
            // The same curve as a formation: every unit's destination is a
            // slot on it, spread over its whole length (it crosses the x=820
            // wall's end, so some slots are displaced around the rock).
            let guides = vec![wp(260.0, 2290.0), wp(450.0, 2420.0), wp(820.0, 2450.0), wp(950.0, 2350.0), wp(1000.0, 2100.0)];
            s.commands.push((1, payload(1, all_ids(sim), CommandKind::CurveFormation { guide_points: guides }, wp(1000.0, 2100.0))));
        }, arena: None, walls: &[], opt_in: false, app_spawn: false },
        Scenario { name: "chokepoint_opposing_traffic", units: 1000, ticks: 2400, setup: |sim, s| {
            // Two groups on either side of the 29-cell wall gap at x=820,
            // y=1420..1450 swap sides.
            let ids = all_ids(sim);
            let (west, east) = ids.split_at(ids.len() / 2);
            place_block(sim, west, (700, 1435), 1.6);
            place_block(sim, east, (940, 1435), 1.6);
            s.commands.push((1, payload(1, west.to_vec(), CommandKind::MoveTo, wp(960.0, 1435.0))));
            s.commands.push((1, payload(2, east.to_vec(), CommandKind::MoveTo, wp(680.0, 1435.0))));
        }, arena: None, walls: &[], opt_in: false, app_spawn: false },
        Scenario { name: "terrain_edits_during_movement", units: 1000, ticks: 2400, setup: |sim, s| {
            s.commands.push((1, payload(1, all_ids(sim), CommandKind::MoveTo, wp(1020.0, 1900.0))));
            // Drop rock blobs on the road ahead of the group, then clear one.
            for (k, c) in [(560, 2165), (700, 2100), (860, 2000), (940, 1950)].iter().enumerate() {
                s.edits.push((90 + k as u64 * 150, brush_patch(CellPos::new(c.0, c.1), 9, Some(true), None, None)));
            }
            s.edits.push((800, brush_patch(CellPos::new(700, 2100), 9, Some(false), None, Some(10))));
        }, arena: None, walls: &[], opt_in: false, app_spawn: false },
        Scenario { name: "unreachable_destination", units: 1000, ticks: 300, setup: |sim, s| {
            // Interior of the sealed rock chamber at (2050, 1190).
            s.commands.push((1, payload(1, all_ids(sim), CommandKind::MoveTo, wp(2050.0, 1190.0))));
        }, arena: None, walls: &[], opt_in: false, app_spawn: false },
        Scenario { name: "scale_5000_one_group", units: 5000, ticks: 4500, setup: |sim, s| {
            s.commands.push((1, payload(1, all_ids(sim), CommandKind::MoveTo, wp(1020.0, 1900.0))));
        }, arena: None, walls: &[], opt_in: false, app_spawn: false },
        Scenario { name: "scale_10000_one_point", units: 10000, ticks: 4500, setup: |sim, s| {
            s.commands.push((1, payload(1, all_ids(sim), CommandKind::MoveTo, wp(1020.0, 1900.0))));
        }, arena: None, walls: &[], opt_in: false, app_spawn: false },
        Scenario { name: "sweep_out_perimeter", units: 10000, ticks: 6000, setup: |sim, s| {
            // A sweep out: 10,000 units packed at the map centre spread to a
            // formation on a loop 160 cells inside the map edge, so the crowd
            // fans out in every direction and crosses most sectors.
            let ids = all_ids(sim);
            place_block(sim, &ids, (1280, 1280), 1.6);
            let (lo, hi) = (160.0, 2400.0);
            let guides = vec![wp(lo, lo), wp(hi, lo), wp(hi, hi), wp(lo, hi), wp(lo, lo + 40.0)];
            s.commands.push((1, payload(1, ids, CommandKind::CurveFormation { guide_points: guides }, wp(lo, lo + 40.0))));
        }, arena: None, walls: &[], opt_in: false, app_spawn: false },
        Scenario { name: "scale_10000_many_groups", units: 10000, ticks: 4500, setup: |sim, s| {
            let ids = all_ids(sim);
            let dests = [(1400, 1560), (1020, 1900), (700, 1300), (1300, 2300), (1700, 1100), (400, 1500), (1100, 2450), (1900, 2400)];
            for (g, chunk) in ids.chunks(ids.len().div_ceil(8)).enumerate() {
                let d = dests[g];
                s.commands.push((1, payload(g as u64 + 1, chunk.to_vec(), CommandKind::MoveTo, wp(d.0 as f32, d.1 as f32))));
            }
        }, arena: None, walls: &[], opt_in: false, app_spawn: false },
        // Crowd through a standing crowd, on a flat 1280 x 768-cell arena
        // (16 x 12 sectors). Movers start at x=200 and are sent to x=1080
        // along y=384; a standing blob is centred on that line at x=640, or
        // off it (y=640) for the free-path baseline. The blob is packed
        // (1.15 cells apart: heavies almost touch) or loose (2.5 cells
        // apart: room to slip between). Opt-in: `--only crowd`.
        Scenario { name: "crowd_free_path", units: 12000, ticks: 9000, setup: |sim, s| {
            crowd(sim, s, 10000, 1.15, (640, 640), CrowdGoal::Across);
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_through_packed", units: 12000, ticks: 9000, setup: |sim, s| {
            crowd(sim, s, 10000, 1.15, (640, 384), CrowdGoal::Across);
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_around_packed_by_curve", units: 12000, ticks: 9000, setup: |sim, s| {
            // The detour drawn by hand: what going around can achieve.
            crowd(sim, s, 10000, 1.15, (640, 384), CrowdGoal::Curve);
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_into_packed_centre", units: 12000, ticks: 9000, setup: |sim, s| {
            crowd(sim, s, 10000, 1.15, (640, 384), CrowdGoal::BlobCentre);
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_through_loose", units: 4500, ticks: 9000, setup: |sim, s| {
            crowd(sim, s, 2500, 2.5, (640, 384), CrowdGoal::Across);
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_squad_through_packed", units: 10020, ticks: 9000, setup: |sim, s| {
            crowd(sim, s, 10000, 1.15, (640, 384), CrowdGoal::Across);
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_squad_through_loose", units: 2520, ticks: 9000, setup: |sim, s| {
            crowd(sim, s, 2500, 2.5, (640, 384), CrowdGoal::Across);
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        // Guards: cases a crowd-aware route must not make worse.
        Scenario { name: "crowd_chokepoint_filled", units: 12000, ticks: 9000, setup: |sim, s| {
            // A wall across x=630..650 with a 40-cell gap at y=364..404, and
            // the blob standing in and around the gap: no way around.
            crowd(sim, s, 10000, 1.15, (640, 384), CrowdGoal::Across);
        }, arena: Some(ARENA), walls: &[(630, 0, 650, 363), (630, 405, 650, 767)], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_blob_near_goal", units: 12000, ticks: 9000, setup: |sim, s| {
            crowd(sim, s, 10000, 1.15, (930, 384), CrowdGoal::Across);
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_second_group_follows", units: 4000, ticks: 9000, setup: |sim, s| {
            // Group A goes to the middle and stands there; at tick 1500
            // group B is sent across, past A's crowd.
            let ids = all_ids(sim);
            let (a, b) = ids.split_at(2000);
            place_block(sim, a, (200, 250), 1.6);
            place_block(sim, b, (200, 520), 1.6);
            s.commands.push((1, payload(1, a.to_vec(), CommandKind::MoveTo, wp(640.0, 384.0))));
            s.commands.push((1500, payload(2, b.to_vec(), CommandKind::MoveTo, wp(1080.0, 384.0))));
            s.watch = a.to_vec();
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_formation_line", units: 2000, ticks: 9000, setup: |sim, s| {
            // No standing crowd: a formation whose first arrivals stand in
            // the way of the rest.
            let ids = all_ids(sim);
            place_block(sim, &ids, (200, 384), 1.6);
            let guides = vec![wp(1000.0, 150.0), wp(1000.0, 618.0)];
            s.commands.push((1, payload(1, ids, CommandKind::CurveFormation { guide_points: guides }, wp(1000.0, 618.0))));
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_real_map_road", units: 12000, ticks: 9000, setup: |sim, s| {
            // Default map: a blob on the road junction at (700, 2100) between
            // the spawn plaza and (1020, 1900).
            let ids = all_ids(sim);
            let (standing, movers) = ids.split_at(10000);
            place_block(sim, standing, (700, 2100), 1.15);
            place_block(sim, movers, (260, 2290), 1.6);
            s.commands.push((1, payload(1, movers.to_vec(), CommandKind::MoveTo, wp(1020.0, 1900.0))));
            s.watch = standing.to_vec();
        }, arena: None, walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_arrived_army", units: 12000, ticks: 9000, setup: |sim, s| {
            // As in the app: an army is sent to the middle and stands on its
            // arrival slots; 60 s later another group is sent across it.
            arrived_army(sim, s, 10000, 2000);
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_50k_arrived_army", units: 50000, ticks: 9000, setup: |sim, s| {
            arrived_army(sim, s, 45000, 5000);
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_edge_start_corner", units: 12000, ticks: 4500, setup: |sim, s| {
            // As in a recorded app session: the movers stand along the
            // crowd's bottom edge and are sent to a point just past its west
            // side, so the straight line cuts the crowd's corner.
            let ids = all_ids(sim);
            let (standing, movers) = ids.split_at(10000);
            place_block(sim, standing, (640, 384), 1.15);
            place_block(sim, movers, (660, 484), 1.6);
            // Sent at tick 60, once the crowd map holds the crowd.
            s.commands.push((60, payload(1, movers.to_vec(), CommandKind::MoveTo, wp(555.0, 400.0))));
            s.watch = standing.to_vec();
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_50k_free_path", units: 50000, ticks: 9000, setup: |sim, s| {
            crowd(sim, s, 45000, 1.15, (640, 640), CrowdGoal::Across);
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_50k_through_packed", units: 50000, ticks: 9000, setup: |sim, s| {
            crowd(sim, s, 45000, 1.15, (640, 384), CrowdGoal::Across);
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        Scenario { name: "crowd_50k_around_packed_by_curve", units: 50000, ticks: 9000, setup: |sim, s| {
            crowd(sim, s, 45000, 1.15, (640, 384), CrowdGoal::WideCurve);
        }, arena: Some(ARENA), walls: &[], opt_in: true, app_spawn: false },
        // A queue at a gap with a free way around: a wall across a
        // 1280 x 1280-cell arena at x=630..650, a narrow gap on the direct
        // line (y=640) and a 40-cell gap off it, placed so that the route
        // through it is 5%, 20% or 50% longer. Movers start at (200, 640)
        // and are sent to (1080, 640). Opt-in: `--only jam`.
        Scenario { name: "jam_gap8_detour5", units: 5000, ticks: 12000, setup: jam,
            arena: Some(JAM_ARENA), walls: &[(630, 0, 650, 478), (630, 519, 650, 635), (630, 644, 650, 1279)], opt_in: true, app_spawn: false },
        Scenario { name: "jam_gap8_detour20", units: 5000, ticks: 12000, setup: jam,
            arena: Some(JAM_ARENA), walls: &[(630, 0, 650, 327), (630, 368, 650, 635), (630, 644, 650, 1279)], opt_in: true, app_spawn: false },
        Scenario { name: "jam_gap8_detour50", units: 5000, ticks: 12000, setup: jam,
            arena: Some(JAM_ARENA), walls: &[(630, 0, 650, 127), (630, 168, 650, 635), (630, 644, 650, 1279)], opt_in: true, app_spawn: false },
        Scenario { name: "jam_gap24_detour5", units: 5000, ticks: 12000, setup: jam,
            arena: Some(JAM_ARENA), walls: &[(630, 0, 650, 478), (630, 519, 650, 627), (630, 652, 650, 1279)], opt_in: true, app_spawn: false },
        Scenario { name: "jam_gap24_detour20", units: 5000, ticks: 12000, setup: jam,
            arena: Some(JAM_ARENA), walls: &[(630, 0, 650, 327), (630, 368, 650, 627), (630, 652, 650, 1279)], opt_in: true, app_spawn: false },
        Scenario { name: "jam_gap24_detour50", units: 5000, ticks: 12000, setup: jam,
            arena: Some(JAM_ARENA), walls: &[(630, 0, 650, 127), (630, 168, 650, 627), (630, 652, 650, 1279)], opt_in: true, app_spawn: false },
        Scenario { name: "jam_gap8_detour20_1k", units: 1000, ticks: 6000, setup: jam,
            arena: Some(JAM_ARENA), walls: &[(630, 0, 650, 327), (630, 368, 650, 635), (630, 644, 650, 1279)], opt_in: true, app_spawn: false },
        Scenario { name: "jam_gap8_detour20_20k", units: 20000, ticks: 18000, setup: jam,
            arena: Some(JAM_ARENA), walls: &[(630, 0, 650, 327), (630, 368, 650, 635), (630, 644, 650, 1279)], opt_in: true, app_spawn: false },
        // Two equal 16-cell gaps 150 cells either side of the line: a
        // route that flips between them shows up as re-keys and a slow t90.
        Scenario { name: "jam_equal_gaps", units: 5000, ticks: 12000, setup: jam,
            arena: Some(JAM_ARENA), walls: &[(630, 0, 650, 481), (630, 498, 650, 781), (630, 798, 650, 1279)], opt_in: true, app_spawn: false },
        // Guard: the 8-cell gap and no way around; must not get slower.
        Scenario { name: "jam_single_gap8", units: 5000, ticks: 12000, setup: jam,
            arena: Some(JAM_ARENA), walls: &[(630, 0, 650, 635), (630, 644, 650, 1279)], opt_in: true, app_spawn: false },
        // A recorded app session: 50,000 units spawned at the plaza, sent
        // to the road junction, then (tick 91) far north-east through two
        // wall gaps and a crack in a hill, each with a way around.
        Scenario { name: "jam_real_map_50k", units: 50000, ticks: 12000, setup: |sim, s| {
            let ids = all_ids(sim);
            s.commands.push((1, payload(1, ids.clone(), CommandKind::MoveTo, WorldPos::new(5604.0, 16804.0))));
            s.commands.push((91, payload(2, ids, CommandKind::MoveTo, WorldPos::new(16284.317, 7323.234))));
        }, arena: None, walls: &[], opt_in: true, app_spawn: true },
    ]
}

const ARENA: (i32, i32) = (1280, 768);
const JAM_ARENA: (i32, i32) = (1280, 1280);

/// All units start in a block at (200, 640), 1.6 cells apart, and are sent
/// to (1080, 640) at tick 1.
fn jam(sim: &mut Simulation, s: &mut Script) {
    let ids = all_ids(sim);
    place_block(sim, &ids, (200, 640), 1.6);
    s.commands.push((1, payload(1, ids, CommandKind::MoveTo, wp(1080.0, 640.0))));
}

enum CrowdGoal {
    /// MoveTo (1080, 384), across the blob's line.
    Across,
    /// FollowCurve to the same point, swinging north around a blob at
    /// (640, 384) with about 60 cells to spare.
    Curve,
    /// The same, around the 45,000-unit blob (245 cells across).
    WideCurve,
    /// MoveTo the blob's centre.
    BlobCentre,
}

/// `army` units start packed at (640, 384) and are sent there (they spread
/// to their arrival slots); at tick 1800 the other units, in a block at
/// (200, 384), are sent to (1080, 384) across them.
fn arrived_army(sim: &mut Simulation, s: &mut Script, army: usize, movers: usize) {
    let ids = all_ids(sim);
    let (standing, rest) = ids.split_at(army);
    let movers = &rest[..movers];
    place_block(sim, standing, (640, 384), 1.15);
    place_block(sim, movers, (200, 384), 1.6);
    s.commands.push((1, payload(1, standing.to_vec(), CommandKind::MoveTo, wp(640.0, 384.0))));
    s.commands.push((1800, payload(2, movers.to_vec(), CommandKind::MoveTo, wp(1080.0, 384.0))));
    s.watch = standing.to_vec();
}

/// The first `blob` units stand in a square block at `centre`, `spacing`
/// cells apart; the rest start in a block at (200, 384), 1.6 cells apart,
/// and get one command at tick 1.
fn crowd(sim: &mut Simulation, s: &mut Script, blob: usize, spacing: f32, centre: (i32, i32), goal: CrowdGoal) {
    let ids = all_ids(sim);
    let (standing, movers) = ids.split_at(blob);
    place_block(sim, standing, centre, spacing);
    place_block(sim, movers, (200, 384), 1.6);
    let to = match goal { CrowdGoal::BlobCentre => wp(centre.0 as f32, centre.1 as f32), _ => wp(1080.0, 384.0) };
    let kind = match goal {
        CrowdGoal::Curve => CommandKind::FollowCurve { guide_points: vec![wp(200.0, 384.0), wp(470.0, 230.0), wp(810.0, 230.0), wp(1080.0, 384.0)] },
        CrowdGoal::WideCurve => CommandKind::FollowCurve { guide_points: vec![wp(200.0, 384.0), wp(450.0, 160.0), wp(830.0, 160.0), wp(1080.0, 384.0)] },
        _ => CommandKind::MoveTo,
    };
    s.commands.push((1, payload(1, movers.to_vec(), kind, to)));
    s.watch = standing.to_vec();
}

fn sim_config(sc: &Scenario, opts: &Opts, workers: usize) -> SimConfig {
    SimConfig {
        unit_count: sc.units,
        worker_count: workers,
        nav_worker_count: opts.nav_workers,
        sector_cells: opts.sector,
        spawn_radius_cells: if sc.app_spawn { SimConfig::default().spawn_radius_cells } else { 96 + (sc.units as i32 / 100) },
        avoidance: !opts.no_avoidance,
        avoidance_time_horizon: opts.tau.unwrap_or(SimConfig::default().avoidance_time_horizon),
        yield_after_ticks: if opts.yield_rule { 30 } else { SimConfig::default().yield_after_ticks },
        ..SimConfig::default()
    }
}

fn run(sc: &Scenario, opts: &Opts, workers: usize) -> ScenarioResult {
    let ticks = if opts.quick { sc.ticks.min(600) } else { sc.ticks };
    let mut cfg = sim_config(sc, opts, workers);
    let t0 = Instant::now();
    let mut sim = match sc.arena {
        Some((w, h)) => {
            let mut tt = ae_nav::terrain::TestTerrain::new(w, h);
            tt.geom = ae_nav::math::WorldGeometry::new(w, h, cfg.cell_size, cfg.sector_cells);
            for &(x0, y0, x1, y1) in sc.walls { tt.wall(x0, y0, x1, y1); }
            cfg.map_cells = w.max(h);
            cfg.spawn_center_cell = (w / 2, h / 2);
            Simulation::from_generated_terrain(cfg, tt.into_terrain(cfg_seed()))
        }
        None => Simulation::new(cfg),
    };
    let mut script = Script::default();
    (sc.setup)(&mut sim, &mut script);
    let setup_ms = t0.elapsed().as_secs_f64() * 1000.0;
    // Finishing times, path length and the standing crowd's displacement
    // concern the units of the last commands (issued on the last command
    // tick), from that tick on.
    let last_cmd_tick = script.commands.iter().map(|(t, _)| *t).max().unwrap_or(1);
    let commanded: std::collections::BTreeSet<u32> = script.commands.iter().filter(|(t, _)| *t == last_cmd_tick)
        .flat_map(|(_, c)| c.unit_ids.iter().copied()).collect();
    let mut watch_start: Vec<WorldPos> = Vec::new();
    // Movers' path length over the straight distance to their command's
    // requested point, counted until each finishes.
    let goal_of: std::collections::BTreeMap<u32, WorldPos> = script.commands.iter().filter(|(t, _)| *t == last_cmd_tick)
        .flat_map(|(_, c)| c.unit_ids.iter().map(move |id| (*id, c.requested_point))).collect();
    let mut straight = 0.0f64;
    let mut travelled = 0.0f64;
    let mut tick_ms = Vec::with_capacity(ticks as usize);
    let mut moving_sum = 0.0;
    let mut cand = 0u64; let mut contacts = 0u64; let mut solver = 0.0; let mut avoid = 0.0;
    let mut peak_groups = 0usize;
    let mut ph = TickPhases::default();
    let (mut jobs_max, mut over_budget) = (0.0f64, 0u64);
    let period = std::time::Duration::from_secs_f64(1.0 / ae_nav::math::TICK_HZ as f64);
    let mut next_tick = Instant::now();
    let mut t50 = None; let mut t90 = None;
    let mut repair_lat_max = 0u64; let mut repair_ms_max = 0.0f64;
    let wall = Instant::now();
    let mut last: SimStats = sim.stats();
    let mut blocked_ticks = vec![0u32; sim.units.len()];
    // Where each unit first crossed the arena's middle column (y, cells).
    let mid_x = sc.arena.filter(|_| !sc.walls.is_empty()).map(|(w, _)| (w / 2) as f32 * sim.terrain.geom.cell_size);
    let mut cross_y: Vec<Option<f32>> = vec![None; sim.units.len()];
    // Ticks run on a pool thread, as in the app.
    sim.in_pool(|sim| {
        for t in 1..=ticks {
            if t == last_cmd_tick {
                watch_start = script.watch.iter().map(|id| sim.units[(*id - 1) as usize].body.pos).collect();
                straight = goal_of.iter().map(|(id, g)| sim.units[(*id - 1) as usize].body.pos.dist(*g) as f64).sum();
            }
            for (at, c) in &script.commands { if *at == t { sim.queue_command_at(c.clone()); } }
            for (at, e) in &script.edits { if *at == t { sim.queue_terrain_patch(e.clone()); } }
            if opts.realtime {
                let now = Instant::now();
                if next_tick > now { std::thread::sleep(next_tick - now); }
                next_tick += period;
            }
            let st = Instant::now();
            sim.advance_tick();
            let ms = st.elapsed().as_secs_f64() * 1000.0;
            tick_ms.push(ms);
            if ms > period.as_secs_f64() * 1000.0 { over_budget += 1; }
            let s = sim.stats();
            if opts.slow_ticks.is_some_and(|limit| ms > limit) { eprintln!("{} tick {t}: {ms:.1} ms {:?}", sc.name, s.phases); }
            jobs_max = jobs_max.max(s.phases.jobs_ms);
            moving_sum += s.moving as f64 / s.units.max(1) as f64;
            cand += s.collision.broadphase_candidates as u64;
            contacts += s.collision.contacts as u64;
            solver += s.collision.solver_ms;
            avoid += s.collision.avoidance_ms;
            let p = &s.phases;
            for (acc, v) in [(&mut ph.jobs_ms, p.jobs_ms), (&mut ph.commands_ms, p.commands_ms), (&mut ph.contexts_ms, p.contexts_ms),
                (&mut ph.steer_ms, p.steer_ms), (&mut ph.avoidance_ms, p.avoidance_ms), (&mut ph.physics_ms, p.physics_ms),
                (&mut ph.progress_ms, p.progress_ms), (&mut ph.statuses_ms, p.statuses_ms), (&mut ph.submit_ms, p.submit_ms),
                (&mut ph.snapshot_ms, p.snapshot_ms)] { *acc += v / ticks as f64; }
            peak_groups = peak_groups.max(s.nav.unique_destination_groups);
            repair_lat_max = repair_lat_max.max(s.nav.last_repair_latency_ticks);
            repair_ms_max = repair_ms_max.max(s.nav.last_repair_ms);
            let done = sim.units.iter().filter(|u| commanded.contains(&u.id) && (u.arrived || u.unreachable)).count();
            for (i, u) in sim.units.iter().enumerate() {
                if u.mode == UnitMode::Blocked { blocked_ticks[i] += 1; }
                if let Some(mx) = mid_x {
                    if cross_y[i].is_none() && u.body.pos.x >= mx && u.last_pos.x < mx { cross_y[i] = Some(u.body.pos.y / sim.terrain.geom.cell_size); }
                }
            }
            if !goal_of.is_empty() && t >= last_cmd_tick {
                travelled += sim.units.iter().filter(|u| goal_of.contains_key(&u.id) && !u.arrived && !u.unreachable)
                    .map(|u| u.body.pos.dist(u.last_pos) as f64).sum::<f64>();
            }
            let n = commanded.len().max(1);
            if t >= last_cmd_tick {
                if t50.is_none() && done * 2 >= n { t50 = Some(t); }
                if t90.is_none() && done * 10 >= n * 9 { t90 = Some(t); }
            }
            last = s;
        }
    });
    let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let nav = &last.nav;
    let hits = nav.cache_hits as f64; let misses = nav.cache_misses as f64;
    ScenarioResult {
        name: sc.name.to_string(), units: sim.units.len(), workers, sector_cells: opts.sector, ticks,
        setup_ms, wall_ms, tick_ms: pct(&mut tick_ms),
        arrived: last.arrived, settled: last.settled, unreachable: last.unreachable, moving_at_end: last.moving,
        mean_moving_fraction: moving_sum / ticks as f64,
        t50_ticks: t50, t90_ticks: t90,
        latency_ticks_median: nav.command_latency.median_ticks,
        latency_ticks_p95: nav.command_latency.p95_ticks,
        latency_ticks_worst: nav.command_latency.worst_ticks,
        latency_ms_median: nav.command_latency.median_ms,
        latency_ms_p95: nav.command_latency.p95_ms,
        latency_ms_worst: nav.command_latency.worst_ms,
        jobs_completed: nav.jobs_completed, plans: nav.plans_completed,
        fields_built: nav.fields_built, field_expansions: nav.field_expansions,
        abstract_expansions: nav.abstract_expansions,
        nav_busy_ms: nav.nav_busy_ms_total,
        cache_hit_rate: if hits + misses > 0.0 { hits / (hits + misses) } else { 0.0 },
        cache_fields: nav.cache_fields,
        nav_memory_mb: (nav.cache_bytes + nav.hierarchy_bytes) as f64 / 1_048_576.0,
        peak_destination_groups: peak_groups,
        collision_candidates_per_tick: cand as f64 / ticks as f64,
        contacts_per_tick: contacts as f64 / ticks as f64,
        solver_ms_per_tick: solver / ticks as f64,
        avoidance_ms_per_tick: avoid / ticks as f64,
        phases_ms_per_tick: ph,
        jobs_ms_max: jobs_max,
        ticks_over_budget: over_budget,
        realtime: opts.realtime,
        terrain_repairs: nav.terrain_rebuilds,
        repair_latency_ticks_max: repair_lat_max,
        repair_ms_max,
        stuck_settled: nav.stuck_settled, skipped_stages: nav.skipped_stages, stuck_gave_up: nav.stuck_gave_up,
        worker_utilization: nav.worker_utilization,
        hierarchy_build_ms: sim.hierarchy.build_ms,
        final_hash: format!("{:016x}", sim.state_hash()),
        commanded_finished: sim.units.iter().filter(|u| commanded.contains(&u.id) && (u.arrived || u.unreachable)).count(),
        commanded: commanded.len(),
        path_ratio: if straight > 0.0 { travelled / straight } else { 0.0 },
        watch_displacement_cells: {
            let mut d: Vec<f64> = script.watch.iter().zip(&watch_start)
                .map(|(id, p)| (sim.units[(*id - 1) as usize].body.pos.dist(*p) / sim.terrain.geom.cell_size) as f64).collect();
            let over = d.iter().filter(|&&x| x > 2.0).count();
            let mean = if d.is_empty() { 0.0 } else { d.iter().sum::<f64>() / d.len() as f64 };
            WatchDisplacement { mean, cells: pct(&mut d), over_2_cells: over }
        },
        blocked_over_30s: sim.units.iter().zip(&blocked_ticks)
            .filter(|(u, &b)| commanded.contains(&u.id) && b > 30 * ae_nav::math::TICK_HZ as u32).count(),
        crossed_off_line: match sc.arena {
            Some((_, h)) => sim.units.iter().zip(&cross_y)
                .filter(|(u, y)| commanded.contains(&u.id) && y.is_some_and(|y| (y - (h / 2) as f32).abs() > 32.0)).count(),
            None => 0,
        },
        crossed_above: match sc.arena {
            Some((_, h)) => sim.units.iter().zip(&cross_y)
                .filter(|(u, y)| commanded.contains(&u.id) && y.is_some_and(|y| y < (h / 2) as f32 - 32.0)).count(),
            None => 0,
        },
        crowd_rekeys: nav.crowd_rekeys,
    }
}

fn cfg_seed() -> u64 { SimConfig::default().seed }

fn print_table(rows: &[ScenarioResult]) {
    println!("| scenario | units | workers | sector | ticks | tick ms med/p95/max | moving frac | arrived exact/settled | unreach | t50/t90 ticks | first-move ticks med/p95/worst | first-move ms med/p95 | fields | cache hit | nav MB | gave up | repair max ticks/ms |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for r in rows {
        println!("| {} | {} | {} | {} | {} | {:.2}/{:.2}/{:.1} | {:.2} | {}/{} | {} | {}/{} | {:.0}/{:.0}/{} | {:.1}/{:.1} | {} | {:.0}% | {:.1} | {} | {}/{:.1} |",
            r.name, r.units, r.workers, r.sector_cells, r.ticks, r.tick_ms.median, r.tick_ms.p95, r.tick_ms.max,
            r.mean_moving_fraction, r.arrived - r.settled, r.settled, r.unreachable,
            r.t50_ticks.map(|t| t.to_string()).unwrap_or("-".into()), r.t90_ticks.map(|t| t.to_string()).unwrap_or("-".into()),
            r.latency_ticks_median, r.latency_ticks_p95, r.latency_ticks_worst,
            r.latency_ms_median, r.latency_ms_p95,
            r.fields_built, r.cache_hit_rate * 100.0, r.nav_memory_mb, r.stuck_gave_up,
            r.repair_latency_ticks_max, r.repair_ms_max);
    }
    println!();
    println!("| scenario | abstract expansions | field expansions | contexts | nav busy ms | worker util | collision cand/tick | contacts/tick | collision ms/tick | avoidance ms/tick |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for r in rows {
        println!("| {} | {} | {} | {} | {:.0} | {:.2}% | {:.0} | {:.0} | {:.2} | {:.2} |", r.name, r.abstract_expansions, r.field_expansions, r.plans,
            r.nav_busy_ms, r.worker_utilization * 100.0, r.collision_candidates_per_tick, r.contacts_per_tick, r.solver_ms_per_tick, r.avoidance_ms_per_tick);
    }
    if rows.iter().any(|r| r.name.starts_with("crowd")) {
        println!();
        println!("| scenario | movers finished | t50/t90 s | path / straight | standing crowd pushed: mean / p95 / max cells, units > 2 cells | contacts/tick | tick ms med/p95 |");
        println!("|---|---|---|---|---|---|---|");
        for r in rows.iter().filter(|r| r.name.starts_with("crowd")) {
            let secs = |t: Option<u64>| t.map(|t| format!("{:.0}", t as f64 / ae_nav::math::TICK_HZ as f64)).unwrap_or("-".into());
            let w = &r.watch_displacement_cells;
            println!("| {} | {}/{} | {}/{} | {:.2} | {:.2} / {:.1} / {:.1}, {} | {:.0} | {:.2}/{:.2} |", r.name, r.commanded_finished, r.commanded,
                secs(r.t50_ticks), secs(r.t90_ticks), r.path_ratio, w.mean, w.cells.p95, w.cells.max, w.over_2_cells,
                r.contacts_per_tick, r.tick_ms.median, r.tick_ms.p95);
        }
    }
    if rows.iter().any(|r| r.name.starts_with("jam")) {
        println!();
        println!("| scenario | movers finished | t50/t90 s | through the detour (above) | blocked > 30 s | path / straight | crowd re-keys |");
        println!("|---|---|---|---|---|---|---|");
        for r in rows.iter().filter(|r| r.name.starts_with("jam")) {
            let secs = |t: Option<u64>| t.map(|t| format!("{:.0}", t as f64 / ae_nav::math::TICK_HZ as f64)).unwrap_or("-".into());
            println!("| {} | {}/{} | {}/{} | {} ({}) | {} | {:.2} | {} |", r.name, r.commanded_finished, r.commanded,
                secs(r.t50_ticks), secs(r.t90_ticks), r.crossed_off_line, r.crossed_above, r.blocked_over_30s, r.path_ratio, r.crowd_rekeys);
        }
    }
    println!();
    println!("| scenario | worst wait for due results (ms) | ticks over one tick period |");
    println!("|---|---|---|");
    for r in rows { println!("| {} | {:.1} | {} |", r.name, r.jobs_ms_max, r.ticks_over_budget); }
    println!();
    println!("Mean ms per tick by `advance_tick` phase:");
    println!();
    println!("| scenario | jobs (incl. barrier wait) | commands | contexts | steering | avoidance | physics | progress | statuses | submit | snapshot | sum |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for r in rows {
        let p = &r.phases_ms_per_tick;
        let sum = p.jobs_ms + p.commands_ms + p.contexts_ms + p.steer_ms + p.avoidance_ms + p.physics_ms + p.progress_ms + p.statuses_ms + p.submit_ms + p.snapshot_ms;
        println!("| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |", r.name, p.jobs_ms, p.commands_ms, p.contexts_ms,
            p.steer_ms, p.avoidance_ms, p.physics_ms, p.progress_ms, p.statuses_ms, p.submit_ms, p.snapshot_ms, sum);
    }
}

/// Hierarchy full build and a local repair at several thread counts. This is
/// where navigation parallelism pays off; outputs are checked identical.
fn build_scaling(opts: &Opts) {
    use ae_nav::nav::{profiles_for, NavHierarchy};
    use std::sync::Arc;
    let cfg = SimConfig { sector_cells: opts.sector, ..SimConfig::default() };
    let terrain = Arc::new(ae_nav::terrain::Terrain::generate(&cfg.terrain_config()));
    let profiles = profiles_for(cfg.cell_size, cfg.radius_scale);
    let patch = brush_patch(CellPos::new(700, 2100), 9, Some(true), None, None);
    let (edited, applied) = terrain.with_patch(patch);
    let edited = Arc::new(edited);
    let rect = applied.dirty_cells().unwrap();
    println!("| sector cells | threads | full build ms | local repair ms | sectors recomputed (repair) | nodes | edges | identical |");
    println!("|---|---|---|---|---|---|---|---|");
    let mut reference: Option<Vec<Vec<(u32, u32)>>> = None;
    for threads in [1usize, 2, 4, 8, 16, 24] {
        let mut best_build = f64::MAX;
        let mut best_repair = f64::MAX;
        let mut h = None;
        let mut r = None;
        for _ in 0..3 {
            let t = Instant::now();
            let built = NavHierarchy::build_parallel(terrain.clone(), &profiles, threads);
            best_build = best_build.min(t.elapsed().as_secs_f64() * 1000.0);
            let t = Instant::now();
            let repaired = built.rebuild_local(edited.clone(), rect, threads);
            best_repair = best_repair.min(t.elapsed().as_secs_f64() * 1000.0);
            h = Some(built);
            r = Some(repaired);
        }
        let h = h.unwrap();
        let r = r.unwrap();
        let edges: Vec<Vec<(u32, u32)>> = r.hierarchies[0].out_edges.clone();
        let same = match &reference { None => { reference = Some(edges); true } Some(e) => *e == edges };
        let st = h.stats();
        println!("| {} | {} | {:.0} | {:.1} | {} | {} | {} | {} |", opts.sector, threads, best_build, best_repair, r.sectors_recomputed, st.nodes, st.edges, if same { "yes" } else { "NO" });
    }
}

/// Cost of one shared destination context (abstract backward Dijkstra) and
/// of one sector field, on the default map, single-threaded.
fn context_timing(opts: &Opts) {
    use ae_nav::nav::{build_context, build_flow_field, context_key, field_recipe, profiles_for, NavHierarchy};
    use std::sync::Arc;
    let cfg = SimConfig { sector_cells: opts.sector, ..SimConfig::default() };
    let terrain = Arc::new(ae_nav::terrain::Terrain::generate(&cfg.terrain_config()));
    let h = NavHierarchy::build_parallel(terrain, &profiles_for(cfg.cell_size, cfg.radius_scale), 8);
    let goals = [(1400, 1560), (700, 2100), (2290, 300), (1300, 700), (500, 1200), (1900, 2400), (1020, 1900), (300, 300)];
    let (mut ctx_ms, mut ctx_exp, mut field_ms, mut field_exp, mut n_ctx, mut n_field) = (Vec::new(), 0u64, Vec::new(), 0u64, 0, 0);
    for pid in 0..2u8 {
        for &(x, y) in &goals {
            let Some(key) = context_key(&h, pid, CellPos::new(x, y)) else { continue };
            let t = Instant::now();
            let Some(ctx) = build_context(&h, &key, None) else { continue };
            ctx_ms.push(t.elapsed().as_secs_f64() * 1000.0);
            ctx_exp += ctx.expansions as u64; n_ctx += 1;
            // Fields for 16 regions along the route from the spawn plaza.
            let ph = h.hierarchy(pid);
            let start = h.region_at(pid, CellPos::new(260, 2290)).unwrap();
            let Some(rec) = field_recipe(&h, &ctx, start.0, start.1, CellPos::new(x, y)) else { continue };
            let mut regions = vec![start];
            if let Some(far) = rec.best_exit_far_node { regions.extend(ae_nav::nav::regions_ahead(ph, &ctx, far, 15)); }
            for (s, r) in regions {
                let Some(rc) = field_recipe(&h, &ctx, s, r, CellPos::new(x, y)) else { continue };
                let t = Instant::now();
                if let Some(f) = build_flow_field(&h.terrain, h.grid(pid), rc.key) {
                    field_ms.push(t.elapsed().as_secs_f64() * 1000.0);
                    field_exp += f.expansions as u64; n_field += 1;
                }
            }
        }
    }
    let st = h.stats();
    println!("| sector cells | nodes | edges | contexts | context ms med/p95/max | expansions per context | fields | field ms med/p95/max | expansions per field |");
    println!("|---|---|---|---|---|---|---|---|---|");
    let c = pct(&mut ctx_ms);
    let f = pct(&mut field_ms);
    println!("| {} | {} | {} | {} | {:.2}/{:.2}/{:.2} | {} | {} | {:.3}/{:.3}/{:.3} | {} |", opts.sector, st.nodes, st.edges, n_ctx,
        c.median, c.p95, c.max, ctx_exp / n_ctx.max(1) as u64, n_field, f.median, f.p95, f.max, field_exp / n_field.max(1) as u64);
}

/// Replay a recorded session at several worker counts and report the first
/// divergent tick with the differing state parts.
fn replay(path: &str) {
    use ae_nav::reference::replay::{first_divergence, ReplayLog};
    let mut log = ReplayLog::load(path).expect("read replay file");
    log.sample_every = 1;
    println!("replay {}: {} events, {} ticks, map {} cells", path, log.events.len(), log.ticks, log.config.map_cells);
    let mut base_cfg = log.config.clone();
    base_cfg.worker_count = 1;
    base_cfg.nav_worker_count = 1;
    let t = Instant::now();
    let base = log.run_with_config(&base_cfg);
    println!("  1 worker: {:.0} ms, final hash {:016x}", t.elapsed().as_secs_f64() * 1000.0, base.last().map(|d| d.state_hash).unwrap_or(0));
    for w in [2usize, 4, 8, 16, 24] {
        let mut cfg = log.config.clone();
        cfg.worker_count = w;
        cfg.nav_worker_count = w;
        let t = Instant::now();
        let other = log.run_with_config(&cfg);
        match first_divergence(&base, &other) {
            None => println!("  {w} workers: identical at all {} ticks ({:.0} ms)", other.len(), t.elapsed().as_secs_f64() * 1000.0),
            Some(d) => println!("  {w} workers: {}", d.describe()),
        }
    }
}

fn parse() -> Opts {
    let mut o = Opts { quick: false, workers: 8, nav_workers: SimConfig::default().nav_worker_count, sector: 64, only: vec![], json: None, scaling: false, build_scaling: false, context_timing: false, replay: None, no_avoidance: false, yield_rule: false, tau: None, export_detour: None, realtime: false, slow_ticks: None };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--quick" => o.quick = true,
            "--no-avoidance" => o.no_avoidance = true,
            "--yield" => o.yield_rule = true,
            "--tau" => { o.tau = Some(args[i + 1].parse().expect("--tau SECONDS")); i += 1; }
            "--scaling" => o.scaling = true,
            "--build-scaling" => o.build_scaling = true,
            "--context-timing" => o.context_timing = true,
            "--replay" => { o.replay = Some(args[i + 1].clone()); i += 1; }
            "--workers" => { o.workers = args[i + 1].parse().expect("--workers N"); i += 1; }
            "--nav-workers" => { o.nav_workers = args[i + 1].parse().expect("--nav-workers N"); i += 1; }
            "--sector" => { o.sector = args[i + 1].parse().expect("--sector N"); i += 1; }
            "--only" => { o.only = args[i + 1].split(',').map(|s| s.to_string()).collect(); i += 1; }
            "--realtime" => o.realtime = true,
            "--export-detour" => { o.export_detour = Some(args[i + 1].clone()); i += 1; }
            "--slow-ticks" => { o.slow_ticks = Some(args[i + 1].parse().expect("--slow-ticks MS")); i += 1; }
            "--json" => { o.json = Some(args[i + 1].clone()); i += 1; }
            "-h" | "--help" => {
                println!("navbench [--quick] [--workers N] [--nav-workers N] [--sector N] [--only a,b] [--json PATH] [--scaling] [--build-scaling] [--context-timing] [--replay FILE] [--no-avoidance] [--yield] [--tau SECONDS] [--export-detour DIR] [--realtime] [--slow-ticks MS]");
                for s in scenarios() { println!("  {}", s.name); }
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    o
}

fn main() {
    let opts = parse();
    if let Some(path) = &opts.replay { replay(path); return; }
    if opts.build_scaling { build_scaling(&opts); return; }
    if opts.context_timing { context_timing(&opts); return; }
    if let Some(dir) = &opts.export_detour { export_detour(&opts, dir); return; }
    let all = scenarios();
    let selected: Vec<&Scenario> = all.iter()
        .filter(|s| if opts.only.is_empty() { !s.opt_in } else { opts.only.iter().any(|o| s.name.contains(o.as_str())) })
        .collect();
    let mut rows = Vec::new();
    if opts.scaling {
        // Same scenario at several worker counts (both pools): wall time
        // and determinism.
        for sc in &selected {
            let mut hashes = Vec::new();
            for w in [1usize, 2, 4, 8, 16, 24] {
                let r = run(sc, &Opts { nav_workers: w, ..opts.clone() }, w);
                eprintln!("{} workers={} wall={:.0}ms hash={}", r.name, w, r.wall_ms, r.final_hash);
                hashes.push(r.final_hash.clone());
                rows.push(r);
            }
            let same = hashes.windows(2).all(|w| w[0] == w[1]);
            println!("determinism {}: {}", sc.name, if same { "identical final hash at 1/2/4/8/16/24 workers" } else { "DIVERGED" });
        }
    } else {
        for sc in &selected {
            let r = run(sc, &opts, opts.workers);
            eprintln!("{}: wall {:.0} ms, arrived {}, unreachable {}", r.name, r.wall_ms, r.arrived, r.unreachable);
            rows.push(r);
        }
    }
    print_table(&rows);
    if let Some(path) = &opts.json {
        std::fs::write(path, serde_json::to_string_pretty(&rows).unwrap()).expect("write json");
    }
}

/// Scenarios for the Detour comparison: one MoveTo per unit, no terrain
/// edits, 1,000 units. Stock DetourCrowd does not reach the 5k/10k scale
/// (see BENCHMARKS), so those are not part of it.
const DETOUR_SCENARIOS: [&str; 3] = ["large_group_cross_map", "many_groups_unrelated", "chokepoint_opposing_traffic"];

/// Overlapping pairs: centres closer than 95% of the summed radii.
fn overlaps(units: &[ae_nav::reference::sim::Unit]) -> u64 {
    let cell = units.iter().map(|u| u.body.radius).fold(0.0f32, f32::max) * 2.0;
    let mut grid: std::collections::HashMap<(i32, i32), Vec<usize>> = std::collections::HashMap::new();
    for (i, u) in units.iter().enumerate() {
        grid.entry(((u.body.pos.x / cell).floor() as i32, (u.body.pos.y / cell).floor() as i32)).or_default().push(i);
    }
    let mut n = 0;
    for (i, u) in units.iter().enumerate() {
        let (cx, cy) = ((u.body.pos.x / cell).floor() as i32, (u.body.pos.y / cell).floor() as i32);
        for dy in -1..=1 { for dx in -1..=1 {
            let Some(v) = grid.get(&(cx + dx, cy + dy)) else { continue };
            for &j in v {
                if j <= i { continue; }
                if u.body.pos.dist(units[j].body.pos) < 0.95 * (u.body.radius + units[j].body.radius) { n += 1; }
            }
        }}
    }
    n
}

/// Export inputs for the Recast/Detour harness (`bench/detour/`) and the
/// same metrics measured on pathsim. Each unit's target is where it ended in
/// pathsim (its slot, or where it settled), so both sides move the same
/// units to the same points.
fn export_detour(opts: &Opts, dir: &str) {
    use std::io::Write;
    std::fs::create_dir_all(dir).expect("create export dir");
    let all = scenarios();
    let selected: Vec<&Scenario> = all.iter()
        .filter(|s| DETOUR_SCENARIOS.contains(&s.name) && (opts.only.is_empty() || opts.only.iter().any(|o| s.name.contains(o.as_str()))))
        .collect();
    let mut grid_written = false;
    for sc in selected {
        let prepare = |sim: &mut Simulation| { let mut script = Script::default(); (sc.setup)(sim, &mut script); script };
        // Pass 1: where every unit ends.
        let mut sim = Simulation::new(sim_config(sc, opts, opts.workers));
        let script = prepare(&mut sim);
        let starts: Vec<WorldPos> = sim.units.iter().map(|u| u.body.pos).collect();
        let mut cmd_tick = vec![0u64; sim.units.len()];
        for (t, c) in &script.commands { for id in &c.unit_ids { cmd_tick[(*id - 1) as usize] = *t; } }
        for t in 1..=sc.ticks {
            for (at, c) in &script.commands { if *at == t { sim.queue_command_at(c.clone()); } }
            sim.advance_tick();
        }
        let targets: Vec<WorldPos> = sim.units.iter().map(|u| u.body.pos).collect();
        if !grid_written {
            let h = &sim.hierarchy;
            let g = &h.terrain.geom;
            let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{dir}/grid.bin")).unwrap());
            f.write_all(&g.cells_x.to_le_bytes()).unwrap();
            f.write_all(&g.cells_y.to_le_bytes()).unwrap();
            f.write_all(&g.cell_size.to_le_bytes()).unwrap();
            f.write_all(&h.terrain.move_cost).unwrap();
            let flags: Vec<u8> = (0..h.terrain.len()).map(|i| h.grid(0).passable[i].min(1) | (h.grid(1).passable[i].min(1) << 1)).collect();
            f.write_all(&flags).unwrap();
            grid_written = true;
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{dir}/{}.units.txt", sc.name)).unwrap());
        writeln!(f, "{} {}", sc.ticks, sim.units.len()).unwrap();
        for (i, u) in sim.units.iter().enumerate() {
            writeln!(f, "{} {} {} {} {} {} {} {} {}", u.profile_id, u.body.radius, u.max_speed, u.accel,
                starts[i].x, starts[i].y, targets[i].x, targets[i].y, cmd_tick[i]).unwrap();
        }
        // Pass 2 (identical run): the metrics the harness also reports.
        let mut sim = Simulation::new(sim_config(sc, opts, opts.workers));
        let script = prepare(&mut sim);
        let n = sim.units.len();
        let mut reached: Vec<Option<u64>> = vec![None; n];
        let mut travelled = vec![0.0f64; n];
        let mut prev: Vec<WorldPos> = sim.units.iter().map(|u| u.body.pos).collect();
        let (mut contacts, mut tick_ms) = (0u64, Vec::new());
        for t in 1..=sc.ticks {
            for (at, c) in &script.commands { if *at == t { sim.queue_command_at(c.clone()); } }
            let st = Instant::now();
            sim.advance_tick();
            tick_ms.push(st.elapsed().as_secs_f64() * 1000.0);
            contacts += overlaps(&sim.units);
            for (i, u) in sim.units.iter().enumerate() {
                if reached[i].is_none() { travelled[i] += u.body.pos.dist(prev[i]) as f64; }
                prev[i] = u.body.pos;
                if reached[i].is_none() && u.body.pos.dist(targets[i]) <= 2.0 * u.body.radius { reached[i] = Some(t); }
            }
        }
        let report = detour_metrics(&reached, &travelled, &starts, &targets, contacts, sc.ticks, &mut tick_ms);
        std::fs::write(format!("{dir}/{}.pathsim.json", sc.name), serde_json::to_string_pretty(&report).unwrap()).unwrap();
        eprintln!("{}: {}", sc.name, serde_json::to_string(&report).unwrap());
    }
}

#[derive(Serialize)]
struct DetourMetrics { units: usize, reached: usize, t50: Option<u64>, t90: Option<u64>, contacts_per_tick: f64, path_ratio_mean: f64, tick_ms: Pct }

/// Shared definitions (the harness computes the same): a unit has reached
/// its target when within two radii of it; path ratio is distance walked
/// until then over the straight start-target distance.
fn detour_metrics(reached: &[Option<u64>], travelled: &[f64], starts: &[WorldPos], targets: &[WorldPos], contacts: u64, ticks: u64, tick_ms: &mut [f64]) -> DetourMetrics {
    let mut times: Vec<u64> = reached.iter().flatten().copied().collect();
    times.sort_unstable();
    let n = reached.len();
    let at = |k: usize| if k >= 1 && k <= times.len() { Some(times[k - 1]) } else { None };
    let ratios: Vec<f64> = (0..n).filter(|&i| reached[i].is_some() && starts[i].dist(targets[i]) > 80.0)
        .map(|i| travelled[i] / starts[i].dist(targets[i]) as f64).collect();
    DetourMetrics {
        units: n, reached: times.len(), t50: at(n.div_ceil(2)), t90: at((n * 9).div_ceil(10)),
        contacts_per_tick: contacts as f64 / ticks as f64,
        path_ratio_mean: ratios.iter().sum::<f64>() / ratios.len().max(1) as f64,
        tick_ms: pct(tick_ms),
    }
}
