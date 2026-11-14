//! Cost overlays: extra traversal cost on a rectangle, removable.
//!
//! * Adding and then removing an overlay returns every cell's cost exactly
//!   to what it was (and the overlay layer to zeros), each change landing
//!   as one terrain version reported as a cost-only change.
//! * Units route around an expensive overlay: on the land map, an overlay
//!   across the middle of the straight line from the spawn plaza to the
//!   goal is entered far less than the same rectangle without it.

use ae_nav::scenario::{Scenario, land_crossing_plan};

const UNITS: u32 = 300;
const MAP_CELLS: i32 = 384;

/// Step until a terrain change lands (at most 30 ticks); its version.
fn land(s: &mut Scenario) -> u32 {
    for _ in 0..30 {
        s.step();
        if let Some(c) = s.nav.take_terrain_changes().pop() {
            assert!(!c.ground, "an overlay changes costs only");
            return c.version;
        }
    }
    panic!("no terrain change landed");
}

#[test]
fn removing_an_overlay_restores_costs() {
    let (nav, plan) = land_crossing_plan(UNITS, MAP_CELLS, 1);
    let mut s = Scenario::from_plan(nav, &plan);
    let before = s.nav.terrain.move_cost.clone();
    assert!(s.nav.terrain.overlay.is_empty(), "no overlay layer until the first overlay");
    let rect = (100, 100, 140, 120);
    assert!(s.nav.add_cost_overlay(7, rect, 50));
    assert!(!s.nav.add_cost_overlay(7, rect, 50), "ids are unique");
    assert!(s.nav.add_cost_overlay(8, (130, 110, 150, 130), 220));
    // One repair at a time: the second overlay lands with the next one.
    land(&mut s);
    let v1 = land(&mut s);
    let t = &s.nav.terrain;
    let w = t.geom.cells_x;
    for y in 95..135 {
        for x in 95..155 {
            let i = (y * w + x) as usize;
            let extra = [((100, 100, 140, 120), 50u16), ((130, 110, 150, 130), 220)].iter().filter(|((x0, y0, x1, y1), _)| (*x0..=*x1).contains(&x) && (*y0..=*y1).contains(&y)).map(|(_, c)| c).sum::<u16>();
            let want = if before[i] == 255 { 255 } else { (before[i] as u16 + extra).min(254) as u8 };
            assert_eq!(t.move_cost[i], want, "cell ({x}, {y})");
        }
    }
    assert!(s.nav.remove_cost_overlay(7));
    assert!(s.nav.remove_cost_overlay(8));
    assert!(!s.nav.remove_cost_overlay(8));
    land(&mut s);
    let v2 = land(&mut s);
    assert!(v2 > v1);
    let t = &s.nav.terrain;
    assert_eq!(t.move_cost, before, "costs back to the original");
    assert_eq!(t.base_cost, before);
    assert!(t.overlay.iter().all(|&o| o == 0));
}

/// Unit-ticks spent inside `rect` over `ticks` ticks, with or without an
/// overlay of `cost` on it.
fn time_inside(rect: (i32, i32, i32, i32), cost: Option<u8>, ticks: u64) -> u64 {
    let (nav, plan) = land_crossing_plan(UNITS, MAP_CELLS, 4);
    let mut s = Scenario::from_plan(nav, &plan);
    if let Some(cost) = cost {
        assert!(s.nav.add_cost_overlay(1, rect, cost));
    }
    let cell = s.nav.terrain.geom.cell_size;
    let mut inside = 0;
    for _ in 0..ticks {
        s.step();
        let b = &s.bodies;
        inside += b.pos_x.iter().zip(&b.pos_y).filter(|&(&x, &y)| {
            let (cx, cy) = ((x / cell) as i32, (y / cell) as i32);
            (rect.0..=rect.2).contains(&cx) && (rect.1..=rect.3).contains(&cy)
        }).count() as u64;
    }
    inside
}

#[test]
fn units_route_around_an_expensive_overlay() {
    let (nav, plan) = land_crossing_plan(UNITS, MAP_CELLS, 1);
    let cell = nav.terrain.geom.cell_size;
    let n = plan.units.len() as f32;
    let (px, py) = plan.units.iter().fold((0.0, 0.0), |(x, y), u| (x + u.pos.x / n, y + u.pos.y / n));
    let goal = plan.orders[0].2;
    let (mx, my) = (((px + goal.x) / 2.0 / cell) as i32, ((py + goal.y) / 2.0 / cell) as i32);
    let rect = (mx - 12, my - 12, mx + 12, my + 12);
    let ticks = 400;
    let control = time_inside(rect, None, ticks);
    let overlaid = time_inside(rect, Some(200), ticks);
    eprintln!("plaza ({px:.0}, {py:.0}), goal ({:.0}, {:.0}), overlay cells {rect:?}: unit-ticks inside {control} without, {overlaid} with", goal.x, goal.y);
    assert!(control > 1000, "the control run should cross the rectangle: {control}");
    assert!(overlaid * 4 < control, "units should avoid the overlay: {overlaid} vs {control}");
}
