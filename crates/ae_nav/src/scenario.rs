//! Standard navigation scenarios for benches, tests and headless runs.
//!
//! Each builds a [`NavService`] with its units and owns their bodies as SoA
//! columns ([`Bodies`]), the way the engine's ECS holds them. Commands are
//! queued for tick 1; call [`Scenario::step`] once per tick.

use crate::math::{CellPos, WorldGeometry, WorldPos};
use crate::nav::{profiles_for, water_profiles, MovementProfile};
use crate::service::{BodyColumns, CommandKind, CommandPayload, NavConfig, NavService, UnitSpec};
use crate::terrain::{reference_cell, Terrain, TerrainConfig, WaterMode};
use std::collections::BTreeSet;

/// Profile ids of [`water_profiles`].
pub mod class {
    pub const SCOUT: u8 = 0;
    pub const HEAVY: u8 = 1;
    pub const SHIP: u8 = 2;
    pub const HOVER: u8 = 3;
    pub const AMPHIBIOUS: u8 = 4;
}

/// Unit bodies as columns, rows in service order.
#[derive(Clone, Debug, Default)]
pub struct Bodies {
    pub pos_x: Vec<f32>,
    pub pos_y: Vec<f32>,
    pub vel_x: Vec<f32>,
    pub vel_y: Vec<f32>,
    pub radius: Vec<f32>,
    pub profile: Vec<u8>,
}

impl Bodies {
    pub fn columns(&mut self) -> BodyColumns<'_> {
        BodyColumns {
            pos_x: &mut self.pos_x,
            pos_y: &mut self.pos_y,
            vel_x: &mut self.vel_x,
            vel_y: &mut self.vel_y,
            radius: &self.radius,
        }
    }
    pub fn pos(&self, i: usize) -> WorldPos {
        WorldPos::new(self.pos_x[i], self.pos_y[i])
    }
    pub fn len(&self) -> usize {
        self.pos_x.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pos_x.is_empty()
    }
}

pub struct Scenario {
    pub nav: NavService,
    pub bodies: Bodies,
}

/// A scenario as data: the units to add, in order (unit `k` is entity
/// `k + 1` in [`Scenario`]), and the move commands to queue before the
/// first tick, in order: `(command id, unit indices, point)`.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub units: Vec<UnitSpec>,
    pub orders: Vec<(u64, Vec<usize>, WorldPos)>,
}

impl Plan {
    /// The order each unit belongs to, if any: `(command id, point)`.
    pub fn order_of(&self) -> Vec<Option<(u64, WorldPos)>> {
        let mut out = vec![None; self.units.len()];
        for (id, units, to) in &self.orders {
            for &u in units {
                out[u] = Some((*id, *to));
            }
        }
        out
    }
}

impl Scenario {
    /// Add the plan's units to `nav` (entity `k + 1` for unit `k`) and
    /// queue its orders.
    pub fn from_plan(nav: NavService, plan: &Plan) -> Self {
        let mut s = Scenario {
            nav,
            bodies: Bodies::default(),
        };
        for (k, spec) in plan.units.iter().enumerate() {
            s.add(k as u64 + 1, *spec);
        }
        for (id, units, to) in &plan.orders {
            s.command(*id, units.iter().map(|&u| u as u64 + 1), *to);
        }
        s
    }

    /// One tick. Run inside [`NavService::in_pool`] for the parallel phases
    /// to use the service's own pool (results are the same either way).
    pub fn step(&mut self) {
        let mut cols = self.bodies.columns();
        self.nav.step(&mut cols);
    }

    /// Run `ticks` ticks on the service's pool, calling `each` after every one.
    pub fn run(&mut self, ticks: u64, mut each: impl FnMut(u64, &NavService, &Bodies) + Send) {
        let Scenario { nav, bodies } = self;
        nav.in_pool(|nav| {
            for t in 1..=ticks {
                let mut cols = bodies.columns();
                nav.step(&mut cols);
                each(t, nav, bodies);
            }
        });
    }

    fn add(&mut self, entity: u64, spec: UnitSpec) {
        self.nav.add_unit(entity, spec);
        let b = &mut self.bodies;
        b.pos_x.push(spec.pos.x);
        b.pos_y.push(spec.pos.y);
        b.vel_x.push(0.0);
        b.vel_y.push(0.0);
        b.radius.push(spec.radius);
        b.profile.push(spec.profile_id);
    }

    fn command(&mut self, id: u64, entities: impl Iterator<Item = u64>, to: WorldPos) {
        let unit_ids = entities
            .map(|e| self.nav.local_id(e).expect("unit of the scenario"))
            .collect();
        self.nav.queue_command_at(CommandPayload {
            id,
            unit_ids,
            kind: CommandKind::MoveTo,
            requested_point: to,
            corridor_tolerance_world: 48.0,
        });
    }
}

fn cell_point(nav: &NavService, c: CellPos) -> WorldPos {
    nav.terrain.geom.cell_center(c)
}

fn config(units: u32, map_cells: i32, workers: usize) -> NavConfig {
    let scale = |v: i32| v * map_cells / 2560;
    NavConfig {
        unit_count: units,
        map_cells,
        worker_count: workers,
        spawn_center_cell: (scale(260), scale(2290)),
        spawn_radius_cells: if map_cells == 2560 {
            96 + units as i32 / 100
        } else {
            (96 + units as i32 / 100) * map_cells / 2560 + 8
        },
        ..NavConfig::default()
    }
}

/// navbench's `scale_10000_one_point` generalised: `units` pathsim units
/// (82% scouts, 18% heavies) from the spawn plaza to reference cell
/// (1020, 1900), on a dry map of `map_cells`². At 2560 cells and 10,000
/// units this is exactly navbench's scenario (same state hash).
pub fn land_crossing(units: u32, map_cells: i32, workers: usize) -> Scenario {
    let (nav, plan) = land_crossing_plan(units, map_cells, workers);
    Scenario::from_plan(nav, &plan)
}

/// [`land_crossing`] as a service without units and its plan.
pub fn land_crossing_plan(units: u32, map_cells: i32, workers: usize) -> (NavService, Plan) {
    let cfg = config(units, map_cells, workers);
    land_crossing_plan_with(units, map_cells, workers, profiles_for(cfg.cell_size, cfg.radius_scale))
}

/// [`land_crossing_plan`] with other movement profiles (classes 0 and 1
/// take the light and heavy units' places).
pub fn land_crossing_plan_with(
    units: u32,
    map_cells: i32,
    workers: usize,
    profiles: Vec<MovementProfile>,
) -> (NavService, Plan) {
    assert!(profiles.len() >= 2, "the land map's units use classes 0 and 1");
    let cfg = config(units, map_cells, workers);
    let terrain = Terrain::generate(&cfg.terrain_config());
    let mut nav = NavService::with_terrain(cfg, terrain, profiles);
    let specs = nav.spawn_plan();
    let goal = cell_point(&nav, reference_cell(&nav.terrain.geom, 1020, 1900));
    let all = (0..specs.len()).collect();
    (
        nav,
        Plan {
            units: specs,
            orders: vec![(1, all, goal)],
        },
    )
}

/// A plan on a given terrain (a loaded map): `units` land units on the
/// land map's spawn plaza (none for a game that spawns its own), no orders.
pub fn terrain_plan_with(
    units: u32,
    terrain: Terrain,
    workers: usize,
    profiles: Vec<MovementProfile>,
) -> (NavService, Plan) {
    let cfg = NavConfig {
        seed: terrain.seed,
        ..config(units, terrain.geom.cells_x, workers)
    };
    #[allow(unused_mut)]
    let mut nav = NavService::with_terrain(cfg, terrain, profiles);
    let specs = if units == 0 { Vec::new() } else { nav.spawn_plan() };
    (
        nav,
        Plan {
            units: specs,
            orders: Vec::new(),
        },
    )
}

/// The terrain of [`land_crossing_plan`] alone (no hierarchy, no units),
/// for viewers. Equal to the service's terrain for any unit count.
pub fn land_crossing_terrain(map_cells: i32) -> Terrain {
    Terrain::generate(&config(0, map_cells, 1).terrain_config())
}

/// The terrain of [`water_crossing_plan`] alone (no hierarchy, no units).
pub fn water_crossing_terrain(map_cells: i32) -> Terrain {
    let mut tc: TerrainConfig = config(0, map_cells, 1).terrain_config();
    tc.water = WaterMode::Surface;
    Terrain::generate(&tc)
}

/// The water map (`WaterMode::Surface`: lakes and a river between the spawn
/// plaza and the goal) with all five classes of [`water_profiles`]: of the
/// `units`, 5% are ships in the river, the rest spawn on the plaza as
/// scouts, heavies, hovercraft (one in ten) and amphibious units (one in
/// ten). Scouts and heavies move to reference cell (1400, 1560) across the
/// river (by the road's causeway); hovercraft and amphibious units to the
/// big lake (1357, 1650, beside the causeway), deep water no land unit may enter;
/// ships sail up the river to (1050, 900).
pub fn water_crossing(units: u32, map_cells: i32, workers: usize) -> Scenario {
    let (nav, plan) = water_crossing_plan(units, map_cells, workers);
    Scenario::from_plan(nav, &plan)
}

/// [`water_crossing`] as a service without units and its plan.
pub fn water_crossing_plan(units: u32, map_cells: i32, workers: usize) -> (NavService, Plan) {
    let cell_size = config(units, map_cells, workers).cell_size;
    water_crossing_plan_with(units, map_cells, workers, water_profiles(cell_size))
}

/// [`water_crossing_plan`] with other movement profiles (classes 0..=4 take
/// the places of scout, heavy, ship, hovercraft and amphibious units).
pub fn water_crossing_plan_with(
    units: u32,
    map_cells: i32,
    workers: usize,
    profiles: Vec<MovementProfile>,
) -> (NavService, Plan) {
    use class::*;
    assert!(
        profiles.len() > AMPHIBIOUS as usize,
        "the water map's units use classes 0..=4"
    );
    let ships = units / 20;
    let cfg = config(units - ships, map_cells, workers);
    let mut tc: TerrainConfig = cfg.terrain_config();
    tc.water = WaterMode::Surface;
    let terrain = Terrain::generate(&tc);
    let mut nav = NavService::with_terrain(cfg, terrain, profiles.clone());
    let scout_radius = profiles[SCOUT as usize].unit_radius;
    let scout_speed = profiles[SCOUT as usize].base_speed;
    let mut specs = Vec::new();
    for (k, mut spec) in nav.spawn_plan().into_iter().enumerate() {
        // Scouts' slots also fit hovercraft and amphibious units (clearance 1).
        if spec.profile_id == SCOUT && matches!(k % 10, 7 | 8) {
            let p = &profiles[if k % 10 == 7 { HOVER } else { AMPHIBIOUS } as usize];
            spec.radius *= p.unit_radius / scout_radius;
            spec.max_speed *= p.base_speed / scout_speed;
            spec.profile_id = p.id;
        }
        specs.push(spec);
    }
    let land_units = specs.len();

    // Ships: ship-passable cells nearest the river point, two cells apart.
    let geom = nav.terrain.geom.clone();
    let grid = nav.hierarchy.grid(SHIP).clone();
    // North of the road's causeway, which cuts the river near (1290, 1650).
    let anchor = reference_cell(&geom, 1150, 1250);
    let ship_goal = nav
        .terrain
        .nearest_open_cell(reference_cell(&geom, 1050, 900), 128, |c| {
            grid.is_passable(&nav.terrain, c)
        });
    let mut cells: Vec<CellPos> = (0..geom.cells_y)
        .flat_map(|y| (0..geom.cells_x).map(move |x| CellPos::new(x, y)))
        .filter(|&c| grid.is_passable(&nav.terrain, c))
        .collect();
    cells.sort_by_key(|c| ((c.x - anchor.x).pow(2) + (c.y - anchor.y).pow(2), c.y, c.x));
    let ship = &profiles[SHIP as usize];
    let mut taken = BTreeSet::new();
    let mut placed = 0u32;
    for c in cells {
        if placed >= ships {
            break;
        }
        if (-1..=1).any(|dy| (-1..=1).any(|dx| taken.contains(&(c.x + dx, c.y + dy)))) {
            continue;
        }
        taken.insert((c.x, c.y));
        specs.push(UnitSpec {
            pos: geom.cell_center(c),
            radius: ship.unit_radius,
            max_speed: ship.base_speed,
            profile_id: SHIP,
        });
        placed += 1;
    }

    // Land units cross on the road's causeway; hovercraft and amphibious
    // units gather in the middle of the big lake, where only they can go.
    let ground: Vec<usize> = (0..land_units)
        .filter(|&u| matches!(specs[u].profile_id, SCOUT | HEAVY))
        .collect();
    let crossers: Vec<usize> = (0..land_units)
        .filter(|&u| matches!(specs[u].profile_id, HOVER | AMPHIBIOUS))
        .collect();
    let mut orders = vec![(1, ground, cell_point(&nav, reference_cell(&geom, 1400, 1560)))];
    // In the big lake, clear of the road's causeway through its middle.
    orders.push((3, crossers, cell_point(&nav, reference_cell(&geom, 1357, 1650))));
    if let (Some(goal), true) = (ship_goal, placed > 0) {
        orders.push((
            2,
            (land_units..land_units + placed as usize).collect(),
            cell_point(&nav, goal),
        ));
    }
    (nav, Plan { units: specs, orders })
}

/// Water level of the coast fixture map, decimetres (42 m): the shelf sits
/// 0-4 m under it and the deep basin about 30 m.
pub const COAST_WATER_LEVEL: i16 = 420;

/// The coast fixture map's generation seed.
pub const COAST_SEED: u64 = 0xC0A5_7E4A_1D00_5EED;

/// The coast fixture map (M16.2, criterion 4): a north-south coastline with
/// a beach, a 0-4 m shelf (sand, rock and weed patches), a drop to a basin
/// about 30 m deep, and a river that enters at the north edge, bends east
/// and opens into the sea at a mouth. One shallow ford crosses the river.
///
/// Deterministic: [`COAST_SEED`], value noise only. Start points
/// ([`coast_starts`]) are on land. The GPU-side render fixture is the same
/// layout built from this terrain (the client's `--map coast`).
pub fn coast_terrain(map_cells: i32) -> Terrain {
    use crate::rng::fbm;
    use crate::terrain::{Material, NO_WATER, TERRAIN_VERSION_INITIAL};

    let geom = WorldGeometry::new(map_cells, map_cells, 8.0, 64);
    let n = (map_cells as usize) * (map_cells as usize);
    let mut t = Terrain {
        geom,
        seed: COAST_SEED,
        version: TERRAIN_VERSION_INITIAL,
        blocked: vec![0; n],
        height: vec![0; n],
        move_cost: vec![10; n],
        material: vec![Material::Grass as u8; n],
        water_level: NO_WATER,
        water_depth: Vec::new(),
        overlay: Vec::new(),
        base_cost: Vec::new(),
    };
    let land_noise = COAST_SEED ^ 0x11;
    let shelf_noise = COAST_SEED ^ 0x22;
    let patch_noise = COAST_SEED ^ 0x33;
    let deep_noise = COAST_SEED ^ 0x44;
    let shore_at = |v: f32| {
        0.42 + 0.045 * (v * std::f32::consts::TAU + 0.7).sin()
            + 0.04 * (fbm(land_noise, v * 5.0, 0.5, 2, 2.0, 0.5) - 0.5)
    };
    // The river enters at the north edge near u 0.20 and bends east to the
    // coast at `MOUTH_V`; its bed is 2.8-4.2 m deep, with a ford at `FORD_V`.
    const MOUTH_V: f32 = 0.82;
    const FORD_V: f32 = 0.30;
    const RIVER_W: f32 = 0.016;
    let mouth_shore = shore_at(MOUTH_V);
    for y in 0..map_cells {
        let v = y as f32 / map_cells as f32;
        let mouth = shore_at(v);
        let river_u = 0.20 + (v.min(MOUTH_V) / MOUTH_V).powi(2) * (mouth_shore - 0.20);
        for x in 0..map_cells {
            let i = t.geom.cell_index_xy(x, y).unwrap();
            let u = x as f32 / map_cells as f32;
            let d = u - mouth;
            let n = fbm(land_noise, u * 9.0, v * 9.0, 3, 2.0, 0.5) - 0.5;
            let (ground, material, cost) = if d < -0.02 {
                // Land: 42.5 m at the beach, rising inland; grass, forest
                // and mud patches.
                let rise = ((-d - 0.02) / 0.12).min(1.0);
                let h = 425.0 + rise * 90.0 + n * 90.0 * rise;
                let p = fbm(patch_noise, u * 24.0, v * 24.0, 2, 2.0, 0.5);
                let (m, c) = if rise < 0.12 {
                    (Material::Sand, 13)
                } else if p > 0.62 {
                    (Material::Forest, 15)
                } else if p < 0.36 {
                    (Material::Mud, 17)
                } else {
                    (Material::Grass, 10)
                };
                (h, m, c)
            } else if d < 0.015 {
                // Beach: 0-0.8 m above the water.
                (428.0 - (d + 0.02) / 0.035 * 16.0, Material::Sand, 13)
            } else if d < 0.13 {
                // The shelf: 0.8-4 m deep; sand with rock and weed patches.
                let k = (d - 0.015) / 0.115;
                let p = fbm(shelf_noise, u * 40.0, v * 40.0, 2, 2.0, 0.55);
                let m = if p > 0.68 {
                    Material::Rock
                } else if p < 0.32 {
                    // Weed beds: forest's dark green reads as weed under water.
                    Material::Forest
                } else {
                    Material::Sand
                };
                (412.0 - k * 32.0 + n * 6.0, m, 12)
            } else if d < 0.20 {
                // The drop: 4 m to 31 m deep over about 1.4 km.
                let k = (d - 0.13) / 0.07;
                let h = 380.0 - k * k * 270.0 + n * 10.0;
                (h, if n > 0.28 { Material::Rock } else { Material::Sand }, 11)
            } else {
                // The basin: 28-34 m deep.
                let k = ((d - 0.20) / 0.25).min(1.0);
                let h = 200.0 - k * 90.0 + (fbm(deep_noise, u * 18.0, v * 18.0, 2, 2.0, 0.5) - 0.5) * 40.0;
                let p = fbm(patch_noise, u * 14.0 + 7.0, v * 14.0 - 3.0, 2, 2.0, 0.5);
                let m = if p > 0.7 {
                    Material::Rock
                } else if p < 0.3 {
                    Material::Mud
                } else {
                    Material::Sand
                };
                (h.max(40.0), m, 10)
            };
            // The river's bed, carved over its channel; the ford is shallow
            // enough for a land unit (< 2 m) where the coast road crosses.
            if v <= MOUTH_V + RIVER_W {
                let rd = (u - river_u).abs();
                if rd < RIVER_W {
                    let across = 1.0 - rd / RIVER_W;
                    let ford = (v - FORD_V).abs() < 0.008;
                    let bed = if ford { 405.0 } else { 392.0 - across * 14.0 };
                    t.height[i] = bed.round().clamp(0.0, 2400.0) as i16;
                    t.material[i] = Material::Sand as u8;
                    t.move_cost[i] = 10;
                    continue;
                }
            }
            t.height[i] = ground.round().clamp(0.0, 2400.0) as i16;
            t.material[i] = material as u8;
            t.move_cost[i] = cost;
        }
    }
    t.flood(COAST_WATER_LEVEL);
    t
}

/// The coast map's start points (world units), both on land.
pub fn coast_starts(map_cells: i32) -> Vec<[f32; 2]> {
    let side = map_cells as f32 * NavConfig::default().cell_size;
    vec![[0.12 * side, 0.62 * side], [0.30 * side, 0.20 * side]]
}

fn coast_config(units: u32, map_cells: i32, workers: usize) -> NavConfig {
    let scale = |v: i32| v * map_cells / 2560;
    NavConfig {
        seed: COAST_SEED,
        map_cells,
        worker_count: workers,
        unit_count: units,
        spawn_center_cell: (scale(307), scale(1587)),
        spawn_radius_cells: if map_cells == 2560 {
            96 + units as i32 / 100
        } else {
            (96 + units as i32 / 100) * map_cells / 2560 + 8
        },
        ..NavConfig::default()
    }
}

/// The coast's shipping lane (x in the 2560-cell reference grid): past the
/// shelf's edge, in 5-8 m of water about a kilometre off the beach.
const LANE_X: i32 = 1450;

/// [`coast_plan`] as a service without units and its plan.
pub fn coast_plan(units: u32, map_cells: i32, workers: usize) -> Scenario {
    let (nav, plan) = coast_plan_with(
        units,
        map_cells,
        workers,
        water_profiles(NavConfig::default().cell_size),
    );
    Scenario::from_plan(nav, &plan)
}

/// The coast fixture map with the water map's five classes: land units spawn
/// on the western land and move to the beach, hovercraft and amphibious
/// units work the shelf, and a shipping lane runs the length of the coast
/// just off the shelf: one hull in ten, a third of them submarines, gather
/// at its southern end and sail north past the beach, so their wakes cross
/// the surf.
pub fn coast_plan_with(
    units: u32,
    map_cells: i32,
    workers: usize,
    profiles: Vec<MovementProfile>,
) -> (NavService, Plan) {
    use class::*;
    assert!(
        profiles.len() > AMPHIBIOUS as usize,
        "the coast map's units use classes 0..=4"
    );
    let ships = units / 10;
    let cfg = coast_config(units - ships, map_cells, workers);
    let terrain = coast_terrain(map_cells);
    let mut nav = NavService::with_terrain(cfg, terrain, profiles.clone());
    let scout_radius = profiles[SCOUT as usize].unit_radius;
    let scout_speed = profiles[SCOUT as usize].base_speed;
    let mut specs = Vec::new();
    for (k, mut spec) in nav.spawn_plan().into_iter().enumerate() {
        // Scouts' slots also fit hovercraft and amphibious units.
        if spec.profile_id == SCOUT && matches!(k % 10, 7 | 8) {
            let p = &profiles[if k % 10 == 7 { HOVER } else { AMPHIBIOUS } as usize];
            spec.radius *= p.unit_radius / scout_radius;
            spec.max_speed *= p.base_speed / scout_speed;
            spec.profile_id = p.id;
        }
        specs.push(spec);
    }
    let land_units = specs.len();

    // Ships at the lane's southern end, off the shelf, two cells apart.
    let geom = nav.terrain.geom.clone();
    let grid = nav.hierarchy.grid(SHIP).clone();
    let anchor = reference_cell(&geom, LANE_X, 2350);
    let mut cells: Vec<CellPos> = (0..geom.cells_y)
        .flat_map(|y| (0..geom.cells_x).map(move |x| CellPos::new(x, y)))
        .filter(|&c| grid.is_passable(&nav.terrain, c))
        .collect();
    cells.sort_by_key(|c| ((c.x - anchor.x).pow(2) + (c.y - anchor.y).pow(2), c.y, c.x));
    let ship = &profiles[SHIP as usize];
    let sub = profiles.iter().find(|p| {
        p.name == "Sub"
            && p.medium == ship.medium
            && p.clearance_cells == ship.clearance_cells
            && p.max_slope == ship.max_slope
    });
    let mut taken = BTreeSet::new();
    let mut placed = 0u32;
    for c in cells {
        if placed >= ships {
            break;
        }
        if (-1..=1).any(|dy| (-1..=1).any(|dx| taken.contains(&(c.x + dx, c.y + dy)))) {
            continue;
        }
        taken.insert((c.x, c.y));
        let hull = if placed % 3 == 2 { sub.unwrap_or(ship) } else { ship };
        specs.push(UnitSpec {
            pos: geom.cell_center(c),
            radius: hull.unit_radius,
            max_speed: hull.base_speed,
            profile_id: hull.id,
        });
        placed += 1;
    }

    // Land units down to the beach; hovercraft and amphibious units onto the
    // shelf; ships north up the lane.
    let ground: Vec<usize> = (0..land_units)
        .filter(|&u| matches!(specs[u].profile_id, SCOUT | HEAVY))
        .collect();
    let crossers: Vec<usize> = (0..land_units)
        .filter(|&u| matches!(specs[u].profile_id, HOVER | AMPHIBIOUS))
        .collect();
    let mut orders = vec![(1, ground, cell_point(&nav, reference_cell(&geom, 870, 1590)))];
    orders.push((3, crossers, cell_point(&nav, reference_cell(&geom, 1280, 1590))));
    if placed > 0 {
        orders.push((
            2,
            (land_units..land_units + placed as usize).collect(),
            cell_point(&nav, reference_cell(&geom, LANE_X, 250)),
        ));
    }
    (nav, Plan { units: specs, orders })
}
