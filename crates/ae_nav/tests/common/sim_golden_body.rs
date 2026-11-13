// Shared by `tests/reference_sim.rs` and the scratch crate that runs the same
// scenario against the original pathsim prototype. The includer defines a
// module `p` exposing `sim`, `commands` and `math`.

/// A small whole-simulation scenario: a 768² map, 600 units, two commands
/// (a move across the map and, later, a curve formation for half the
/// units), and a terrain edit in the way. Returns `(tick, state_hash)` at
/// fixed ticks.
pub fn sim_golden(threads: usize) -> Vec<(u64, u64)> {
    use p::commands::{CommandKind, CommandPayload};
    use p::math::WorldPos;
    use p::sim::{brush_patch, SimConfig, Simulation};

    let cfg = SimConfig {
        map_cells: 768,
        unit_count: 600,
        worker_count: threads,
        nav_worker_count: 2,
        spawn_center_cell: (78, 687),
        spawn_radius_cells: 40,
        ..SimConfig::default()
    };
    let cell = cfg.cell_size;
    let at = |x: f32, y: f32| WorldPos::new((x + 0.5) * cell, (y + 0.5) * cell);
    let mut sim = Simulation::new(cfg);
    let ids: Vec<u32> = sim.units.iter().map(|u| u.id).collect();
    let half: Vec<u32> = ids.iter().copied().filter(|id| id % 2 == 0).collect();
    let mut out = Vec::new();
    sim.in_pool(|sim| {
        for t in 1..=420u64 {
            if t == 1 {
                sim.queue_command_at(CommandPayload {
                    id: 1, unit_ids: ids.clone(), kind: CommandKind::MoveTo,
                    requested_point: at(420.0, 468.0), corridor_tolerance_world: 48.0,
                });
            }
            if t == 90 {
                sim.queue_terrain_patch(brush_patch(p::math::CellPos::new(250, 560), 6, Some(true), None, None));
            }
            if t == 200 {
                sim.queue_command_at(CommandPayload {
                    id: 2, unit_ids: half.clone(),
                    kind: CommandKind::CurveFormation { guide_points: vec![at(300.0, 600.0), at(360.0, 560.0), at(420.0, 600.0)] },
                    requested_point: at(420.0, 600.0), corridor_tolerance_world: 48.0,
                });
            }
            sim.advance_tick();
            if t % 60 == 0 {
                out.push((t, sim.state_hash()));
            }
        }
    });
    out
}
