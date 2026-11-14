//! Standard navigation scenarios for benches, tests and headless runs.
//!
//! Each builds a [`NavService`] with its units and owns their bodies as SoA
//! columns ([`Bodies`]), the way the engine's ECS holds them. Commands are
//! queued for tick 1; call [`Scenario::step`] once per tick.

use crate::math::{CellPos, WorldPos};
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
        BodyColumns { pos_x: &mut self.pos_x, pos_y: &mut self.pos_y, vel_x: &mut self.vel_x, vel_y: &mut self.vel_y, radius: &self.radius }
    }
    pub fn pos(&self, i: usize) -> WorldPos { WorldPos::new(self.pos_x[i], self.pos_y[i]) }
    pub fn len(&self) -> usize { self.pos_x.len() }
    pub fn is_empty(&self) -> bool { self.pos_x.is_empty() }
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
        let mut s = Scenario { nav, bodies: Bodies::default() };
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
        let unit_ids = entities.map(|e| self.nav.local_id(e).expect("unit of the scenario")).collect();
        self.nav.queue_command_at(CommandPayload { id, unit_ids, kind: CommandKind::MoveTo, requested_point: to, corridor_tolerance_world: 48.0 });
    }
}

fn cell_point(nav: &NavService, c: CellPos) -> WorldPos { nav.terrain.geom.cell_center(c) }

fn config(units: u32, map_cells: i32, workers: usize) -> NavConfig {
    let scale = |v: i32| v * map_cells / 2560;
    NavConfig {
        unit_count: units,
        map_cells,
        worker_count: workers,
        spawn_center_cell: (scale(260), scale(2290)),
        spawn_radius_cells: if map_cells == 2560 { 96 + units as i32 / 100 } else { (96 + units as i32 / 100) * map_cells / 2560 + 8 },
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
pub fn land_crossing_plan_with(units: u32, map_cells: i32, workers: usize, profiles: Vec<MovementProfile>) -> (NavService, Plan) {
    assert!(profiles.len() >= 2, "the land map's units use classes 0 and 1");
    let cfg = config(units, map_cells, workers);
    let terrain = Terrain::generate(&cfg.terrain_config());
    let mut nav = NavService::with_terrain(cfg, terrain, profiles);
    let specs = nav.spawn_plan();
    let goal = cell_point(&nav, reference_cell(&nav.terrain.geom, 1020, 1900));
    let all = (0..specs.len()).collect();
    (nav, Plan { units: specs, orders: vec![(1, all, goal)] })
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
pub fn water_crossing_plan_with(units: u32, map_cells: i32, workers: usize, profiles: Vec<MovementProfile>) -> (NavService, Plan) {
    use class::*;
    assert!(profiles.len() > AMPHIBIOUS as usize, "the water map's units use classes 0..=4");
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
    let ship_goal = nav.terrain.nearest_open_cell(reference_cell(&geom, 1050, 900), 128, |c| grid.is_passable(&nav.terrain, c));
    let mut cells: Vec<CellPos> = (0..geom.cells_y)
        .flat_map(|y| (0..geom.cells_x).map(move |x| CellPos::new(x, y)))
        .filter(|&c| grid.is_passable(&nav.terrain, c))
        .collect();
    cells.sort_by_key(|c| ((c.x - anchor.x).pow(2) + (c.y - anchor.y).pow(2), c.y, c.x));
    let ship = &profiles[SHIP as usize];
    let mut taken = BTreeSet::new();
    let mut placed = 0u32;
    for c in cells {
        if placed >= ships { break; }
        if (-1..=1).any(|dy| (-1..=1).any(|dx| taken.contains(&(c.x + dx, c.y + dy)))) { continue; }
        taken.insert((c.x, c.y));
        specs.push(UnitSpec { pos: geom.cell_center(c), radius: ship.unit_radius, max_speed: ship.base_speed, profile_id: SHIP });
        placed += 1;
    }

    // Land units cross on the road's causeway; hovercraft and amphibious
    // units gather in the middle of the big lake, where only they can go.
    let ground: Vec<usize> = (0..land_units).filter(|&u| matches!(specs[u].profile_id, SCOUT | HEAVY)).collect();
    let crossers: Vec<usize> = (0..land_units).filter(|&u| matches!(specs[u].profile_id, HOVER | AMPHIBIOUS)).collect();
    let mut orders = vec![(1, ground, cell_point(&nav, reference_cell(&geom, 1400, 1560)))];
    // In the big lake, clear of the road's causeway through its middle.
    orders.push((3, crossers, cell_point(&nav, reference_cell(&geom, 1357, 1650))));
    if let (Some(goal), true) = (ship_goal, placed > 0) {
        orders.push((2, (land_units..land_units + placed as usize).collect(), cell_point(&nav, goal)));
    }
    (nav, Plan { units: specs, orders })
}
