//! Curve formations with a facing: rows stack behind the line, on the side
//! away from the point the formation faces, instead of alternating on both
//! sides (pathsim's layout, kept without a facing).

use ae_nav::scenario::{land_crossing_plan, Scenario};

const UNITS: u32 = 300;
const MAP_CELLS: i32 = 384;

/// Slots of a curve formation of every unit along a straight line south of
/// the plaza, with or without a facing: `(slots ahead of the line, slots
/// behind it)` (more than a cell off it), ahead meaning toward `toward`.
fn formation_sides(facing: bool) -> (usize, usize) {
    use ae_nav::math::WorldPos;
    use ae_nav::service::{CommandKind, CommandPayload};
    let (nav, mut plan) = land_crossing_plan(UNITS, MAP_CELLS, 1);
    plan.orders.clear();
    let mut s = Scenario::from_plan(nav, &plan);
    let y = 2300.0;
    let line = vec![WorldPos::new(300.0, y), WorldPos::new(700.0, y)];
    let toward = WorldPos::new(500.0, 2000.0);
    let unit_ids: Vec<u32> = (1..=plan.units.len() as u64)
        .map(|e| s.nav.local_id(e).unwrap())
        .collect();
    let payload = CommandPayload {
        id: 1,
        unit_ids,
        kind: CommandKind::CurveFormation { guide_points: line },
        requested_point: WorldPos::new(700.0, y),
        corridor_tolerance_world: 48.0,
    };
    if facing {
        s.nav.queue_formation_facing(payload, toward)
    } else {
        s.nav.queue_command_at(payload)
    }
    s.step();
    s.nav.publish_snapshot();
    let cmd = &s.nav.snapshot().unwrap().commands[0];
    assert!(cmd.slots.len() > UNITS as usize * 9 / 10, "{} slots", cmd.slots.len());
    let cell = s.nav.terrain.geom.cell_size;
    let ahead = cmd.slots.iter().filter(|p| p.y < y - cell).count();
    let behind = cmd.slots.iter().filter(|p| p.y > y + cell).count();
    (ahead, behind)
}

#[test]
fn a_facing_formation_stacks_rows_behind_its_line() {
    let (ahead, behind) = formation_sides(false);
    eprintln!("without a facing: {ahead} slots ahead of the line, {behind} behind");
    assert!(ahead > 20 && behind > 20, "pathsim's rows alternate on both sides");
    let (ahead, behind) = formation_sides(true);
    eprintln!("facing north: {ahead} slots ahead of the line, {behind} behind");
    assert!(ahead == 0 && behind > 100, "rows stack behind the line");
}
