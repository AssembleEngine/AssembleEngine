//! Micro benches for navigation kernels: `cargo bench -p ae_nav`.

use ae_nav::avoidance::{compute_velocities, Agent, AvoidanceConfig};
use ae_nav::math::{CellPos, WorldPos};
use ae_nav::nav::{build_context, build_flow_field, context_key, default_profiles, field_recipe, NavHierarchy};
use ae_nav::par::Par;
use ae_nav::terrain::{Terrain, TerrainConfig};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use std::sync::Arc;

const CELLS: i32 = 640;
const CELL_SIZE: f32 = 8.0;

fn benches(c: &mut Criterion) {
    let terrain = Arc::new(Terrain::generate(&TerrainConfig::new(7, CELLS, CELL_SIZE, 64)));
    let profiles = default_profiles(CELL_SIZE);
    let profile = profiles[0].id;
    let hierarchy = NavHierarchy::build_parallel(terrain.clone(), &profiles, 4);
    let centre = CellPos::new(CELLS / 2, CELLS / 2);
    let goal = terrain
        .nearest_open_cell(centre, CELLS / 4, |cell| {
            context_key(&hierarchy, profile, cell).is_some()
        })
        .unwrap();
    let key = context_key(&hierarchy, profile, goal).unwrap();
    let context = build_context(&hierarchy, &key, None).unwrap();
    let recipe = field_recipe(&hierarchy, &context, key.goal_sector, key.goal_region, goal).unwrap();
    let grid = hierarchy.grid(profile);

    c.bench_function("context_build", |b| {
        b.iter(|| build_context(black_box(&hierarchy), &key, None))
    });
    c.bench_function("flow_field_build", |b| {
        b.iter(|| build_flow_field(&terrain, grid, black_box(recipe.key.clone())))
    });

    let agents: Vec<Agent> = (0..4000usize)
        .map(|k| {
            let heading = WorldPos::new(((k * 7) % 5) as f32 * 40.0 - 80.0, ((k * 3) % 7) as f32 * 30.0 - 90.0);
            Agent {
                pos: WorldPos::new((k % 64) as f32 * 11.0, (k / 64) as f32 * 11.0),
                vel: heading.scaled(0.8),
                radius: 5.0,
                max_speed: 112.0,
                pref: heading,
                avoids: true,
                moving: k % 5 != 0,
                horizon: 0.0,
            }
        })
        .collect();
    let config = AvoidanceConfig::default();
    for threads in [1usize, 4] {
        let pool = Par::new(threads);
        c.bench_with_input(BenchmarkId::new("orca_4000", threads), &pool, |b, pool| {
            b.iter(|| compute_velocities(black_box(&agents), &config, 1.0 / 30.0, pool))
        });
    }
}

criterion_group!(group, benches);
criterion_main!(group);
