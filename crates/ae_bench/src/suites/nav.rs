//! Navigation bench: generated terrain, no simulation, no window.

use super::Options;
use crate::measure::time;
use crate::record::Report;
use ae_nav::avoidance::{Agent, AvoidanceConfig, compute_velocities};
use ae_nav::math::{CellPos, WorldPos};
use ae_nav::nav::{NavHierarchy, build_context, build_flow_field, context_key, default_profiles, field_recipe};
use ae_nav::par::Par;
use ae_nav::terrain::{Terrain, TerrainConfig};
use anyhow::{Context, Result};
use std::sync::Arc;

const SEED: u64 = 0xA55E_4B1E;
const CELL_SIZE: f32 = 8.0;
const SECTOR_CELLS: i32 = 64;

/// A dense grid of agents with mixed headings: the worst case for neighbour search.
pub fn crowd(count: usize) -> Vec<Agent> {
    let side = (count as f32).sqrt().ceil() as usize;
    (0..count)
        .map(|k| {
            let (x, y) = ((k % side) as f32 * 11.0 + 100.0, (k / side) as f32 * 11.0 + 100.0);
            let heading = WorldPos::new(((k * 7) % 5) as f32 * 40.0 - 80.0, ((k * 3) % 7) as f32 * 30.0 - 90.0);
            Agent { pos: WorldPos::new(x, y), vel: heading.scaled(0.8), radius: 5.0, max_speed: 112.0, pref: heading, avoids: true, moving: k % 5 != 0, horizon: 0.0 }
        })
        .collect()
}

pub fn run(options: &Options) -> Result<Report> {
    let mut report = Report::new("nav");
    let cells = if options.small { 640 } else { 2560 };
    let tag = format!("map{cells}");
    let config = TerrainConfig::new(SEED, cells, CELL_SIZE, SECTOR_CELLS);

    report.time(&format!("{tag}/terrain_generate"), time(options.budget, || Terrain::generate(&config)));
    let terrain = Arc::new(Terrain::generate(&config));
    report.info(&format!("{tag}/terrain_digest_low32"), terrain.digest() & 0xFFFF_FFFF, "");

    let profiles = default_profiles(CELL_SIZE);
    let profile = profiles[0].id;
    report.time(&format!("{tag}/hierarchy_build_1t"), time(options.budget, || NavHierarchy::build(terrain.clone(), &profiles)));
    if options.threads > 1 {
        let metric = format!("{tag}/hierarchy_build_{}t", options.threads);
        report.time(&metric, time(options.budget, || NavHierarchy::build_parallel(terrain.clone(), &profiles, options.threads)));
    }
    let hierarchy = NavHierarchy::build_parallel(terrain.clone(), &profiles, options.threads);
    let stats = hierarchy.stats();
    report.info(&format!("{tag}/abstract_nodes"), stats.nodes as u64, "");
    report.info(&format!("{tag}/abstract_edges"), stats.edges as u64, "");
    report.counter(&format!("{tag}/hierarchy_bytes"), stats.bytes as u64, "B");

    // Goal: the passable cell nearest the map centre.
    let centre = CellPos::new(cells / 2, cells / 2);
    let goal = terrain
        .nearest_open_cell(centre, cells / 4, |c| context_key(&hierarchy, profile, c).is_some())
        .context("no passable cell near the map centre")?;
    let key = context_key(&hierarchy, profile, goal).context("goal lost its context key")?;
    let context = build_context(&hierarchy, &key, None).context("context build failed")?;
    report.counter(&format!("{tag}/context_expansions"), context.expansions as u64, "");
    report.time(&format!("{tag}/context_build"), time(options.budget, || build_context(&hierarchy, &key, None)));

    // The goal sector's own field: a full-sector Dijkstra.
    let recipe = field_recipe(&hierarchy, &context, key.goal_sector, key.goal_region, goal).context("no field recipe")?;
    let grid = hierarchy.grid(profile);
    let field = build_flow_field(&terrain, grid, recipe.key.clone()).context("flow field build failed")?;
    report.counter(&format!("{tag}/field_expansions"), field.expansions as u64, "");
    report.info(&format!("{tag}/field_reachable_cells"), field.reachable_count() as u64, "");
    report.time(&format!("{tag}/flow_field_build"), time(options.budget, || build_flow_field(&terrain, grid, recipe.key.clone())));

    let agents = crowd(if options.small { 2_000 } else { 10_000 });
    let avoidance = AvoidanceConfig::default();
    let agents_tag = format!("orca{}", agents.len());
    let serial = Par::new(1);
    report.time(&format!("{agents_tag}/step_1t"), time(options.budget, || compute_velocities(&agents, &avoidance, 1.0 / 30.0, &serial)));
    if options.threads > 1 {
        let pool = Par::new(options.threads);
        let metric = format!("{agents_tag}/step_{}t", options.threads);
        report.time(&metric, time(options.budget, || compute_velocities(&agents, &avoidance, 1.0 / 30.0, &pool)));
    }
    Ok(report)
}
