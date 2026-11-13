//! Movement classes on a map with water: land units ford only shallow water,
//! ships stay in deep water, hover and amphibious units cross anywhere. Both
//! paths (planner) and bodies (collision) must respect the class.

use ae_nav::math::{CellPos, WorldGeometry, WorldPos};
use ae_nav::nav::{astar, water_profiles, MovementProfile, ProfileGrid, FORD_DEPTH, SHIP_DRAFT};
use ae_nav::service::{BodyColumns, CommandKind, CommandPayload, NavConfig, NavService, UnitMode, UnitSpec};
use ae_nav::terrain::{Terrain, TerrainConfig, TestTerrain, WaterMode, SURFACE_WATER_LEVEL};

const W: i32 = 128;
const H: i32 = 64;
const CELL: f32 = 8.0;
const SECTOR: i32 = 16;
const SCOUT: u8 = 0;
const HEAVY: u8 = 1;
const SHIP: u8 = 2;
const HOVER: u8 = 3;
const AMPHIBIOUS: u8 = 4;

/// Pinned state hashes of the two water runs (identical in dev and release
/// builds). A change means the water rules, the generator or the service
/// changed behaviour; on another platform, a float determinism failure.
const LAKE_CROSSING_HASH: u64 = 0x4a57_65ec_d44a_9671;
const GENERATED_CROSSING_HASH: u64 = 0x3a31_1bd7_f1a1_2fc9;

/// Dry banks west and east of a deep lake (x 40..88, 6.8 m deep) with one
/// shallow ford across it (y 50..58, 0.8 m deep).
fn lake_map() -> Terrain {
    let mut tt = TestTerrain::new(W, H);
    tt.geom = WorldGeometry::new(W, H, CELL, SECTOR);
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            tt.height[i] = if (40..88).contains(&x) {
                if (50..58).contains(&y) { 160 } else { 100 }
            } else {
                180
            };
        }
    }
    let mut t = tt.into_terrain(1);
    t.flood(SURFACE_WATER_LEVEL);
    t
}

fn depth_at(t: &Terrain, c: CellPos) -> i16 {
    t.depth(t.geom.cell_index(c))
}

fn path(t: &Terrain, p: &MovementProfile, from: (i32, i32), to: (i32, i32)) -> Option<Vec<CellPos>> {
    let grid = ProfileGrid::build(t, p);
    astar(t, &grid, CellPos::new(from.0, from.1), CellPos::new(to.0, to.1), false).map(|p| p.cells)
}

#[test]
fn paths_follow_each_medium() {
    let t = lake_map();
    let profiles = water_profiles(CELL);
    let (west, east) = ((10, 20), (118, 20));

    for id in [SCOUT, HEAVY] {
        let cells = path(&t, &profiles[id as usize], west, east).expect("land units cross by the ford");
        assert!(cells.iter().all(|&c| depth_at(&t, c) <= FORD_DEPTH), "profile {id} entered deep water");
        assert!(cells.iter().any(|c| c.y >= 50), "profile {id} should detour through the ford");
    }
    for id in [HOVER, AMPHIBIOUS] {
        let cells = path(&t, &profiles[id as usize], west, east).expect("crosses the lake");
        assert!(cells.iter().all(|c| c.y < 50), "profile {id} should go straight across, not via the ford");
        assert!(cells.iter().any(|&c| depth_at(&t, c) > FORD_DEPTH));
    }
    let ship = &profiles[SHIP as usize];
    assert!(path(&t, ship, west, east).is_none(), "ships cannot start on land");
    let cells = path(&t, ship, (48, 10), (80, 30)).expect("ships sail the lake");
    assert!(cells.iter().all(|&c| depth_at(&t, c) >= SHIP_DRAFT));
    assert!(path(&t, ship, (48, 10), (48, 54)).is_none(), "the ford is too shallow for ships");
}

#[test]
fn dry_profiles_are_unchanged_on_dry_maps() {
    // With no water level, a ford-limited land profile sees exactly what
    // pathsim's profile sees.
    let t = Terrain::generate(&TerrainConfig::new(7, 256, CELL, 32));
    let mut wet = water_profiles(CELL)[SCOUT as usize].clone();
    let dry = ae_nav::nav::profiles_for(CELL, 1.0)[SCOUT as usize].clone();
    wet.medium = ae_nav::terrain::Medium::land(FORD_DEPTH);
    assert_eq!(ProfileGrid::build(&t, &wet).passable, ProfileGrid::build(&t, &dry).passable);
}

#[test]
fn surface_generator_makes_water_and_keeps_land_maps() {
    let mut cfg = TerrainConfig::new(0x5EED_CAFE_F00D_1234, 768, CELL, 64);
    let dry = Terrain::generate(&cfg);
    cfg.water = WaterMode::Surface;
    let wet = Terrain::generate(&cfg);
    let water_cells = (0..wet.len()).filter(|&i| wet.depth(i) > 0 && wet.blocked[i] == 0).count();
    let deep_cells = (0..wet.len()).filter(|&i| wet.depth(i) >= SHIP_DRAFT && wet.blocked[i] == 0).count();
    // About 1.8% of the map (two lakes and the river; 10,665 cells today).
    assert!(water_cells > 8_000, "expected real water, got {water_cells} cells");
    assert!(deep_cells > 6_000, "expected water ships can use, got {deep_cells} cells");
    assert!((0..dry.len()).all(|i| dry.depth(i) == 0));
    assert_ne!(dry.digest(), wet.digest());
}

struct Run {
    hash: u64,
    modes: Vec<u8>,
    profile: Vec<u8>,
    pos: Vec<WorldPos>,
}

/// 60 units of all five classes, ordered to the east bank; every tick every
/// body is checked against its class.
fn crossing(threads: usize) -> Run {
    let profiles = water_profiles(CELL);
    let cfg = NavConfig {
        map_cells: W, cell_size: CELL, sector_cells: SECTOR, worker_count: threads, unit_count: 0,
        ..NavConfig::default()
    };
    let mut nav = NavService::with_terrain(cfg, lake_map(), profiles.clone());
    let terrain = lake_map();
    let mut specs = Vec::new();
    let mut put = |profile: u8, x: i32, y: i32| {
        let p = &profiles[profile as usize];
        specs.push(UnitSpec { pos: WorldPos::new((x as f32 + 0.5) * CELL, (y as f32 + 0.5) * CELL), radius: p.unit_radius, max_speed: p.base_speed, profile_id: profile });
    };
    for k in 0..12 {
        put(SCOUT, 4 + (k % 4) * 3, 8 + (k / 4) * 4);
        put(HEAVY, 18 + (k % 4) * 4, 8 + (k / 4) * 5);
        put(SHIP, 48 + (k % 4) * 6, 8 + (k / 4) * 9);
        put(HOVER, 4 + (k % 4) * 3, 26 + (k / 4) * 4);
        put(AMPHIBIOUS, 18 + (k % 4) * 4, 26 + (k / 4) * 4);
    }
    let (mut px, mut py, mut vx, mut vy, mut radius) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (k, s) in specs.iter().enumerate() {
        nav.add_unit(500 + k as u64, *s);
        px.push(s.pos.x);
        py.push(s.pos.y);
        vx.push(0.0);
        vy.push(0.0);
        radius.push(s.radius);
    }
    let ids: Vec<u32> = (0..specs.len()).map(|k| nav.local_id(500 + k as u64).unwrap()).collect();
    nav.in_pool(|nav| {
        nav.queue_command_at(CommandPayload {
            id: 1, unit_ids: ids, kind: CommandKind::MoveTo,
            requested_point: WorldPos::new(118.5 * CELL, 20.5 * CELL), corridor_tolerance_world: 48.0,
        });
        for tick in 1..=900 {
            nav.step(&mut BodyColumns { pos_x: &mut px, pos_y: &mut py, vel_x: &mut vx, vel_y: &mut vy, radius: &radius });
            for (i, s) in specs.iter().enumerate() {
                let c = terrain.geom.world_to_cell(WorldPos::new(px[i], py[i]));
                let d = depth_at(&terrain, c);
                match s.profile_id {
                    SCOUT | HEAVY => assert!(d <= FORD_DEPTH, "tick {tick}: land unit {i} at {c:?} in {d} dm of water"),
                    SHIP => assert!(d >= SHIP_DRAFT, "tick {tick}: ship {i} at {c:?} in {d} dm of water"),
                    _ => {}
                }
            }
        }
    });
    let mut modes = vec![0; nav.len()];
    nav.write_modes(&mut modes);
    Run {
        hash: nav.state_hash(),
        modes,
        profile: specs.iter().map(|s| s.profile_id).collect(),
        pos: px.iter().zip(&py).map(|(&x, &y)| WorldPos::new(x, y)).collect(),
    }
}

#[test]
fn every_class_reaches_what_it_can() {
    let run = crossing(1);
    for (i, &p) in run.profile.iter().enumerate() {
        let cell = run.pos[i].x / CELL;
        if p == SHIP {
            // The goal is on land; ships end at the nearest deep water to it.
            assert!(cell > 78.0, "ship {i} should have sailed east, is at x cell {cell}");
        } else {
            assert_eq!(run.modes[i], UnitMode::Arrived as u8, "unit {i} (profile {p}) did not arrive, at x cell {cell}");
        }
    }
}

#[test]
fn water_crossing_is_deterministic() {
    let one = crossing(1);
    let four = crossing(4);
    assert_eq!(one.hash, LAKE_CROSSING_HASH, "lake crossing: got {:#018x}", one.hash);
    assert_eq!(one.hash, four.hash);
    assert_eq!(one.modes, four.modes);
}

/// The generated water map at 768², 1,000 units of all classes: classes hold
/// on every tick, amphibious units and hovercraft really use the water,
/// ships really sail, and the result does not depend on the thread count.
fn generated_crossing(threads: usize) -> (u64, Vec<u8>, [usize; 3]) {
    use ae_nav::scenario::{class, water_crossing};
    let mut s = water_crossing(1000, 768, threads);
    let terrain = s.nav.terrain.clone();
    let start: Vec<WorldPos> = (0..s.bodies.len()).map(|i| s.bodies.pos(i)).collect();
    let mut wet = vec![false; s.bodies.len()];
    s.run(1500, |tick, _, b| {
        for i in 0..b.len() {
            let c = terrain.geom.world_to_cell(b.pos(i));
            let d = depth_at(&terrain, c);
            match b.profile[i] {
                class::SCOUT | class::HEAVY => assert!(d <= FORD_DEPTH, "tick {tick}: land unit {i} at {c:?} in {d} dm of water"),
                class::SHIP => assert!(d >= SHIP_DRAFT, "tick {tick}: ship {i} at {c:?} in {d} dm"),
                _ => wet[i] |= d > FORD_DEPTH,
            }
        }
    });
    let b = &s.bodies;
    let crossed_wet = (0..b.len()).filter(|&i| wet[i]).count();
    let ships_moved = (0..b.len()).filter(|&i| b.profile[i] == class::SHIP && b.pos(i).dist(start[i]) > 20.0 * CELL).count();
    let mut modes = vec![0; s.nav.len()];
    s.nav.write_modes(&mut modes);
    if std::env::var_os("AE_WATER_DEBUG").is_some() {
        let mut by = std::collections::BTreeMap::new();
        for i in 0..b.len() { *by.entry((b.profile[i], modes[i])).or_insert(0) += 1; }
        eprintln!("(profile, mode) -> count: {by:?}");
        for i in (0..b.len()).filter(|&i| b.profile[i] == class::SHIP).take(50) {
            let c = terrain.geom.world_to_cell(b.pos(i));
            let c0 = terrain.geom.world_to_cell(start[i]);
            eprint!("{:?}->{:?} m{} ", (c0.x, c0.y), (c.x, c.y), modes[i]);
        }
        eprintln!();
    }
    let arrived = modes.iter().filter(|&&m| m == UnitMode::Arrived as u8).count();
    (s.nav.state_hash(), modes, [crossed_wet, ships_moved, arrived])
}

#[test]
fn generated_water_map_crossing() {
    let (hash, modes, [crossed_wet, ships_moved, arrived]) = generated_crossing(1);
    eprintln!("units {} arrived {arrived}, through deep water {crossed_wet}, ships moved {ships_moved}", modes.len());
    assert!(crossed_wet > 20, "hovercraft and amphibious units should take the water, {crossed_wet} did");
    assert!(ships_moved > 25, "most of the 50 ships should sail up the river, {ships_moved} did");
    assert!(arrived > modes.len() / 2, "most units should arrive in 50 s, {arrived} did");
    assert_eq!(hash, GENERATED_CROSSING_HASH, "generated crossing: got {hash:#018x}");
    assert_eq!(generated_crossing(4).0, hash, "thread count changed the result");
}
