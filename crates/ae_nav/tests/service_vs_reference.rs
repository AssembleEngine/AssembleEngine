//! The columnar navigation service against the frozen pathsim reference.
//!
//! `same_hashes_as_reference` drives `NavService` through caller-owned body
//! columns with the scenario of `reference_sim.rs` and must reproduce the
//! reference's pinned state hashes exactly. The other tests cover what the
//! reference cannot do: units added and removed while others move, and
//! bodies changed by the caller between ticks.

use ae_nav::math::{CellPos, WorldPos};
use ae_nav::reference::sim::brush_patch;
use ae_nav::service::{BodyColumns, CommandKind, CommandPayload, NavConfig, NavService, UnitMode, UnitSpec};

/// Same values as `reference_sim.rs` (pathsim's own output).
const REFERENCE: &[(u64, u64)] = &[
    (60, 0x3f6cb446c7800fe9),
    (120, 0x6358d8701137daa9),
    (180, 0xef07809764f1bf56),
    (240, 0xd1d30baf014e0c33),
    (300, 0x8c488805c5659242),
    (360, 0x11f3d7b2c255ce81),
    (420, 0x298b2b04d2acd085),
];

const ENTITY_BASE: u64 = 0x0000_0001_0000_0000;

/// Bodies held the way the ECS would: SoA columns, rows in service order.
#[derive(Default)]
struct Columns {
    entity: Vec<u64>,
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    vel_x: Vec<f32>,
    vel_y: Vec<f32>,
    radius: Vec<f32>,
}

impl Columns {
    fn push(&mut self, entity: u64, spec: &UnitSpec) {
        self.entity.push(entity);
        self.pos_x.push(spec.pos.x);
        self.pos_y.push(spec.pos.y);
        self.vel_x.push(0.0);
        self.vel_y.push(0.0);
        self.radius.push(spec.radius);
    }
    fn remove(&mut self, gone: &[u64]) {
        let keep: Vec<bool> = self.entity.iter().map(|e| !gone.contains(e)).collect();
        let filter = |v: &mut Vec<f32>| {
            let mut k = keep.iter();
            v.retain(|_| *k.next().unwrap());
        };
        filter(&mut self.pos_x);
        filter(&mut self.pos_y);
        filter(&mut self.vel_x);
        filter(&mut self.vel_y);
        filter(&mut self.radius);
        let mut k = keep.iter();
        self.entity.retain(|_| *k.next().unwrap());
    }
    fn step(&mut self, nav: &mut NavService) {
        assert!(
            nav.entities().eq(self.entity.iter().copied()),
            "rows must follow service order"
        );
        nav.step(&mut BodyColumns {
            pos_x: &mut self.pos_x,
            pos_y: &mut self.pos_y,
            vel_x: &mut self.vel_x,
            vel_y: &mut self.vel_y,
            radius: &self.radius,
        });
    }
}

fn config(threads: usize) -> NavConfig {
    NavConfig {
        map_cells: 768,
        unit_count: 600,
        worker_count: threads,
        nav_worker_count: 2,
        spawn_center_cell: (78, 687),
        spawn_radius_cells: 40,
        ..NavConfig::default()
    }
}

fn at(cell: f32, x: f32, y: f32) -> WorldPos {
    WorldPos::new((x + 0.5) * cell, (y + 0.5) * cell)
}

fn setup(threads: usize) -> (NavService, Columns) {
    let mut nav = NavService::new(config(threads));
    let mut cols = Columns::default();
    for (k, spec) in nav.spawn_plan().into_iter().enumerate() {
        let entity = ENTITY_BASE + k as u64 * 7;
        nav.add_unit(entity, spec);
        cols.push(entity, &spec);
    }
    (nav, cols)
}

fn reference_scenario(threads: usize) -> Vec<(u64, u64)> {
    let (mut nav, mut cols) = setup(threads);
    let cell = nav.cfg.cell_size;
    let ids: Vec<u32> = cols.entity.iter().map(|e| nav.local_id(*e).unwrap()).collect();
    let half: Vec<u32> = ids.iter().copied().filter(|id| id % 2 == 0).collect();
    let mut out = Vec::new();
    nav.in_pool(|nav| {
        for t in 1..=420u64 {
            if t == 1 {
                nav.queue_command_at(CommandPayload {
                    id: 1,
                    unit_ids: ids.clone(),
                    kind: CommandKind::MoveTo,
                    requested_point: at(cell, 420.0, 468.0),
                    corridor_tolerance_world: 48.0,
                });
            }
            if t == 90 {
                nav.queue_terrain_patch(brush_patch(CellPos::new(250, 560), 6, Some(true), None, None));
            }
            if t == 200 {
                nav.queue_command_at(CommandPayload {
                    id: 2,
                    unit_ids: half.clone(),
                    kind: CommandKind::CurveFormation {
                        guide_points: vec![at(cell, 300.0, 600.0), at(cell, 360.0, 560.0), at(cell, 420.0, 600.0)],
                    },
                    requested_point: at(cell, 420.0, 600.0),
                    corridor_tolerance_world: 48.0,
                });
            }
            cols.step(nav);
            if t % 60 == 0 {
                out.push((t, nav.state_hash()));
            }
        }
    });
    // The service's one behaviour change from pathsim never fires here.
    assert_eq!(nav.stats().nav.fallback_slots, 0);
    out
}

#[test]
fn same_hashes_as_reference() {
    for threads in [1, 4] {
        let got = reference_scenario(threads);
        for (&(tick, g), &(_, want)) in got.iter().zip(REFERENCE) {
            assert_eq!(
                g, want,
                "tick {tick} at {threads} thread(s): service {g:#018x}, reference {want:#018x}"
            );
        }
        assert_eq!(got.len(), REFERENCE.len());
    }
}

/// Units are removed and added while a command runs; the result must not
/// depend on the thread count, and nothing may panic on stale ids.
fn churn_scenario(threads: usize) -> (u64, usize, Vec<u8>) {
    let (mut nav, mut cols) = setup(threads);
    let cell = nav.cfg.cell_size;
    let mut next_entity = ENTITY_BASE + 1_000_000;
    nav.in_pool(|nav| {
        for t in 1..=300u64 {
            if t == 1 || t == 150 {
                let ids: Vec<u32> = cols.entity.iter().map(|e| nav.local_id(*e).unwrap()).collect();
                let target = if t == 1 {
                    at(cell, 420.0, 468.0)
                } else {
                    at(cell, 200.0, 600.0)
                };
                nav.queue_command_at(CommandPayload {
                    id: t,
                    unit_ids: ids,
                    kind: CommandKind::MoveTo,
                    requested_point: target,
                    corridor_tolerance_world: 48.0,
                });
            }
            if t % 25 == 0 {
                // Remove every 9th unit (by row), add a few fresh ones at the spawn.
                let gone: Vec<u64> = cols
                    .entity
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(i, _)| (i + t as usize) % 9 == 0)
                    .map(|(_, e)| e)
                    .collect();
                assert_eq!(nav.remove_units(&gone), gone.len());
                cols.remove(&gone);
                for k in 0..5 {
                    let spec = UnitSpec {
                        pos: at(cell, 70.0 + k as f32 * 3.0, 690.0),
                        radius: 2.7,
                        max_speed: 100.0,
                        profile_id: (k % 2) as u8,
                    };
                    nav.add_unit(next_entity, spec);
                    cols.push(next_entity, &spec);
                    next_entity += 1;
                }
            }
            cols.step(nav);
        }
    });
    let mut modes = vec![0u8; nav.len()];
    nav.write_modes(&mut modes);
    (nav.state_hash(), nav.len(), modes)
}

#[test]
fn units_come_and_go_deterministically() {
    let one = churn_scenario(1);
    let four = churn_scenario(4);
    assert_eq!(one, four);
    let moving = one
        .2
        .iter()
        .filter(|&&m| m == UnitMode::Moving as u8 || m == UnitMode::Arrived as u8)
        .count();
    assert!(
        moving > one.1 / 2,
        "most units should be moving or arrived, got {moving} of {}",
        one.1
    );
}

#[test]
fn caller_owns_bodies_between_ticks() {
    let (mut nav, mut cols) = setup(1);
    let cell = nav.cfg.cell_size;
    nav.in_pool(|nav| {
        for _ in 0..5 {
            cols.step(nav);
        }
        // Teleport row 0 far away and stop it; the service must start from there.
        let target = at(cell, 300.0, 300.0);
        cols.pos_x[0] = target.x;
        cols.pos_y[0] = target.y;
        cols.vel_x[0] = 0.0;
        cols.vel_y[0] = 0.0;
        for _ in 0..60 {
            cols.step(nav);
        }
        let moved = WorldPos::new(cols.pos_x[0], cols.pos_y[0]).dist(target);
        assert!(
            moved < cell * 0.25,
            "an idle unit should stand where the caller put it, moved {moved}"
        );
        assert_eq!(nav.units[0].body.pos, WorldPos::new(cols.pos_x[0], cols.pos_y[0]));
    });
}

#[test]
fn terrain_edits_are_reported_when_they_land() {
    let (mut nav, mut cols) = setup(1);
    let mut changes = Vec::new();
    nav.in_pool(|nav| {
        for t in 1..=12u64 {
            if t == 2 {
                nav.queue_terrain_patch(brush_patch(CellPos::new(250, 560), 6, Some(true), None, None));
            }
            if t == 3 {
                nav.queue_terrain_patch(brush_patch(CellPos::new(100, 100), 2, None, Some(900), None));
            }
            cols.step(nav);
            changes.extend(nav.take_terrain_changes().into_iter().map(|c| (t, c)));
        }
    });
    // Each edit lands with its repair, a fixed delay after it is queued; the
    // second waits for the first repair to finish, so both are reported in
    // order with growing versions and their own cell rectangles.
    assert_eq!(changes.len(), 2, "{changes:?}");
    let ((t1, a), (t2, b)) = (changes[0], changes[1]);
    assert_eq!((a.tick, b.tick), (t1, t2));
    assert!(t1 > 2 && t2 > t1);
    assert!(b.version > a.version);
    assert_eq!(a.cells, (244, 554, 256, 566));
    assert_eq!(b.cells, (98, 98, 102, 102));
    assert_eq!(nav.terrain.version, b.version);
    assert!(nav.take_terrain_changes().is_empty());
}
