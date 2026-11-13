// Shared body of the navigation golden-digest check.
//
// Included by `tests/golden.rs` (with `lib` = this crate) and, for the
// lift-fidelity check, by a scratch crate built against the original
// `pathsim` prototype (with `lib` = pathsim). It must only use APIs that
// exist unchanged in both.

use lib::avoidance::{compute_velocities, Agent, AvoidanceConfig};
use lib::collision::{Body, Collider, CollisionConfig};
use lib::math::{CellPos, WorldPos};
use lib::nav::{build_context, build_flow_field, context_key, default_profiles, field_recipe, NavHierarchy};
use lib::par::Par;
use lib::terrain::{Terrain, TerrainConfig};
use std::sync::Arc;

pub const GOLDEN_SEED: u64 = 0x601D_E2D1;
pub const GOLDEN_CELLS: i32 = 640;
pub const GOLDEN_AGENTS: usize = 2000;
pub const GOLDEN_TICKS: usize = 60;

/// FNV-1a over 64-bit words.
pub struct Fnv(pub u64);

impl Fnv {
    pub fn new() -> Self { Fnv(0xCBF2_9CE4_8422_2325) }
    pub fn word(&mut self, w: u64) {
        for byte in w.to_le_bytes() {
            self.0 = (self.0 ^ byte as u64).wrapping_mul(0x100_0000_01B3);
        }
    }
    pub fn pos(&mut self, p: WorldPos) {
        self.word(p.x.to_bits() as u64);
        self.word(p.y.to_bits() as u64);
    }
}

/// Every stage's digest, so a mismatch names the first stage that differs.
pub fn golden(threads: usize) -> Vec<(&'static str, u64)> {
    let mut out = Vec::new();
    let cell_size = 8.0;
    let terrain = Arc::new(Terrain::generate(&TerrainConfig::new(GOLDEN_SEED, GOLDEN_CELLS, cell_size, 64)));
    out.push(("terrain", terrain.digest()));

    let profiles = default_profiles(cell_size);
    let profile = profiles[0].id;
    let hierarchy = NavHierarchy::build_parallel(terrain.clone(), &profiles, threads);
    let stats = hierarchy.stats();
    let mut h = Fnv::new();
    for v in [stats.portals, stats.nodes, stats.edges, stats.regions, stats.sectors, stats.pass_cells] {
        h.word(v as u64);
    }
    out.push(("hierarchy_stats", h.0));

    let centre = CellPos::new(GOLDEN_CELLS / 2, GOLDEN_CELLS / 2);
    let goal = terrain.nearest_open_cell(centre, GOLDEN_CELLS / 4, |c| context_key(&hierarchy, profile, c).is_some()).expect("goal");
    let key = context_key(&hierarchy, profile, goal).expect("context key");
    let context = build_context(&hierarchy, &key, None).expect("context");
    let mut h = Fnv::new();
    for (label, hop) in context.labels.iter().zip(&context.next_hop) {
        h.word(*label);
        h.word(*hop as u64);
    }
    out.push(("context", h.0));

    let recipe = field_recipe(&hierarchy, &context, key.goal_sector, key.goal_region, goal).expect("recipe");
    let field = build_flow_field(&terrain, hierarchy.grid(profile), recipe.key.clone()).expect("field");
    let mut h = Fnv::new();
    for (cost, dir) in field.cost_to_go.iter().zip(&field.exit_dir) {
        h.word(*cost);
        h.word(*dir as u64);
    }
    out.push(("flow_field", h.0));

    // A crowd on open cells around the goal, all heading for it: dense
    // enough that avoidance and collision both do real work every tick.
    let target = terrain.world_of_cell(goal);
    let mut agents = Vec::with_capacity(GOLDEN_AGENTS);
    let side = (GOLDEN_AGENTS as f32).sqrt().ceil() as i32;
    let mut k = 0i32;
    while agents.len() < GOLDEN_AGENTS {
        let cell = CellPos::new(GOLDEN_CELLS / 2 - side + 2 * (k % side), GOLDEN_CELLS / 2 - side + 2 * (k / side));
        k += 1;
        if !terrain.geom.in_bounds(cell) || terrain.is_blocked(cell) {
            continue;
        }
        let p = terrain.world_of_cell(cell);
        agents.push(Agent { pos: p, vel: WorldPos::new(0.0, 0.0), radius: 5.0, max_speed: 60.0, pref: WorldPos::new(0.0, 0.0),
            avoids: true, moving: true, horizon: 0.0 });
    }
    let ids: Vec<u32> = (0..agents.len() as u32).collect();
    let par = Par::new(threads);
    let avoidance = AvoidanceConfig::default();
    let collision = CollisionConfig::default();
    let mut collider = Collider::new();
    let mut contacts = Vec::new();
    let dt = 1.0 / 30.0;
    for _ in 0..GOLDEN_TICKS {
        for a in agents.iter_mut() {
            a.pref = target.sub(a.pos).normalized_or(WorldPos::new(0.0, 0.0)).scaled(a.max_speed);
        }
        let velocities = compute_velocities(&agents, &avoidance, dt, &par);
        let mut bodies: Vec<Body> =
            agents.iter().zip(&velocities).map(|(a, v)| Body { pos: a.pos, vel: *v, radius: a.radius }).collect();
        collider.step(&ids, &mut bodies, &terrain, &collision, dt, 2, &par, &mut contacts, 0);
        for (a, b) in agents.iter_mut().zip(&bodies) {
            a.pos = b.pos;
            a.vel = b.vel;
        }
    }
    let mut h = Fnv::new();
    for a in &agents {
        h.pos(a.pos);
        h.pos(a.vel);
    }
    out.push(("crowd", h.0));
    out
}
