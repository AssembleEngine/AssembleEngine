//! The coast fixture map (M16.2, criterion 4): a coastline, a 0-4 m shelf
//! with sand, rock and weed patches, a basin about 30 m deep, a river mouth
//! with one ford, and start points on land. Deterministic: the terrain hash
//! and the plan's state hash are pinned and identical at every worker count.

use ae_nav::scenario::{class, coast_plan, coast_starts, coast_terrain, COAST_WATER_LEVEL};
use ae_nav::service::NavConfig;
use ae_nav::terrain::Terrain;

const CELLS: i32 = 768;

/// Pinned state hashes (identical in dev and release builds). A change means
/// the coast generator, its plan or the service changed behaviour; on
/// another platform, a float determinism failure.
const COAST_TERRAIN_HASH: u64 = 0x7c54_f08f_c077_ebad;
const COAST_PLAN_HASH: u64 = 0xeffd_cac6_f659_0b8b;

fn depth_at(t: &Terrain, x: i32, y: i32) -> i16 {
    t.depth(t.geom.cell_index_xy(x, y).expect("in bounds"))
}

#[test]
fn coast_has_a_shelf_deep_water_a_mouth_and_land_starts() {
    let t = coast_terrain(CELLS);
    assert_eq!(t.water_level, COAST_WATER_LEVEL, "the coast map is wet");

    // The shelf (0.1-4 m) and the basin (at least 30 m) both span the map,
    // and the deepest water is about 30 m.
    let mut shelf = 0;
    let mut deep = 0;
    let mut deepest = 0;
    let mut materials = [0usize; 7];
    for i in 0..t.len() {
        let d = t.depth(i);
        if (1..=40).contains(&d) {
            shelf += 1;
            materials[t.material[i] as usize] += 1;
        }
        if d >= 300 {
            deep += 1;
        }
        deepest = deepest.max(d);
    }
    assert!(shelf > 10_000, "expected a shelf, got {shelf} cells 0.1-4 m deep");
    assert!(deep > 10_000, "expected a basin, got {deep} cells at least 30 m deep");
    assert!((300..=340).contains(&deepest), "deepest water is {deepest} dm");
    // The shelf's bed shows sand, rock and weed (forest), not one material.
    assert!(materials[2] > 1000, "shelf sand: {}", materials[2]);
    assert!(materials[3] > 100, "shelf rock: {}", materials[3]);
    assert!(materials[1] > 100, "shelf weed: {}", materials[1]);

    // The river runs north to its mouth as a channel: water at its centre,
    // land either side, and a ford shallow enough for a land unit.
    let row = |v: f32| (v * CELLS as f32) as i32;
    let channel = |v: f32| -> Vec<i32> {
        (0..(0.35 * CELLS as f32) as i32)
            .filter(|&x| depth_at(&t, x, row(v)) > 0)
            .collect()
    };
    let mid = channel(0.40);
    assert!(!mid.is_empty(), "the river at v 0.40");
    let (first, last) = (mid[0], *mid.last().unwrap());
    let centre = (first + last) / 2;
    assert!(
        last - first < (CELLS as f32 * 0.05) as i32,
        "the river is a channel: {first}..{last} of {CELLS}"
    );
    assert!(
        depth_at(&t, centre, row(0.40)) > 20,
        "the river's bed is {} dm deep",
        depth_at(&t, centre, row(0.40))
    );
    assert_eq!(
        depth_at(&t, first - 45.min(first), row(0.40)),
        0,
        "land west of the river"
    );
    assert_eq!(depth_at(&t, last + 45, row(0.40)), 0, "land east of the river");
    let ford = channel(0.30)
        .into_iter()
        .map(|x| depth_at(&t, x, row(0.30)))
        .min()
        .expect("the river is wet at the ford row");
    assert!(ford <= 20, "the ford is {ford} dm deep (a land unit fords 20)");

    // The river reaches the north edge and the sea.
    assert!(!channel(0.002).is_empty(), "the river enters at the north edge");
    assert!(
        depth_at(&t, (0.44 * CELLS as f32) as i32, row(0.82)) > 0,
        "the mouth is water"
    );

    // Start points are on dry, open land.
    for start in coast_starts(CELLS) {
        let c = t.geom.world_to_cell(ae_nav::math::WorldPos::new(start[0], start[1]));
        let i = t.geom.cell_index(c);
        assert_eq!(t.depth(i), 0, "start {start:?} is under water");
        assert_eq!(t.blocked[i], 0, "start {start:?} is blocked");
    }

    assert_eq!(
        t.digest(),
        COAST_TERRAIN_HASH,
        "coast terrain: got {:#018x}",
        t.digest()
    );
    assert_eq!(t.digest(), coast_terrain(CELLS).digest(), "deterministic");
}

/// 600 units of all five classes: every class holds its medium on the coast
/// map and the state hash does not depend on the worker count.
#[test]
fn coast_plan_is_deterministic_across_worker_counts() {
    let mut hashes = Vec::new();
    for workers in [1, 2, 4, 8] {
        let mut s = coast_plan(600, CELLS, workers);
        assert!(s.bodies.profile.contains(&class::SHIP), "ships on the coast");
        assert!(s.bodies.profile.contains(&class::HOVER));
        let terrain = s.nav.terrain.clone();
        s.run(300, |tick, _, b| {
            for i in 0..b.len() {
                let c = terrain.geom.world_to_cell(b.pos(i));
                let d = depth_at(&terrain, c.x, c.y);
                match b.profile[i] {
                    class::SCOUT | class::HEAVY => {
                        assert!(d <= 20, "tick {tick}: land unit {i} in {d} dm of water")
                    }
                    class::SHIP => assert!(d >= 30, "tick {tick}: ship {i} in {d} dm of water"),
                    _ => {}
                }
            }
        });
        assert!(
            s.nav.state_hash() != 0,
            "the plan runs at {workers} worker(s): {:#018x}",
            s.nav.state_hash()
        );
        hashes.push(s.nav.state_hash());
    }
    let _ = NavConfig::default();
    assert!(
        hashes.iter().all(|&h| h == hashes[0]),
        "state hashes differ by worker count: {hashes:?}"
    );
    assert_eq!(hashes[0], COAST_PLAN_HASH, "coast plan: got {:#018x}", hashes[0]);
}
