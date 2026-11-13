//! Standard navigation scenarios for benches, tests and headless runs.
//!
//! Each builds a [`NavService`] with its units and owns their bodies as SoA
//! columns ([`Bodies`]), the way the engine's ECS holds them. Commands are
//! queued for tick 1; call [`Scenario::step`] once per tick.

use crate::math::{CellPos, WorldPos};
use crate::nav::{profiles_for, water_profiles};
use crate::service::{BodyColumns, CommandKind, CommandPayload, NavConfig, NavService};
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

impl Scenario {
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

    fn add(&mut self, entity: u64, spec: crate::service::UnitSpec) {
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
    let cfg = config(units, map_cells, workers);
    let terrain = Terrain::generate(&cfg.terrain_config());
    let profiles = profiles_for(cfg.cell_size, cfg.radius_scale);
    let mut s = Scenario { nav: NavService::with_terrain(cfg, terrain, profiles), bodies: Bodies::default() };
    for (k, spec) in s.nav.spawn_plan().into_iter().enumerate() {
        s.add(k as u64 + 1, spec);
    }
    let goal = cell_point(&s.nav, reference_cell(&s.nav.terrain.geom, 1020, 1900));
    let n = s.bodies.len() as u64;
    s.command(1, 1..=n, goal);
    s
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
    use class::*;
    let ships = units / 20;
    let cfg = config(units - ships, map_cells, workers);
    let mut tc: TerrainConfig = cfg.terrain_config();
    tc.water = WaterMode::Surface;
    let terrain = Terrain::generate(&tc);
    let profiles = water_profiles(cfg.cell_size);
    let mut s = Scenario { nav: NavService::with_terrain(cfg, terrain, profiles.clone()), bodies: Bodies::default() };
    let scout_radius = profiles[SCOUT as usize].unit_radius;
    let scout_speed = profiles[SCOUT as usize].base_speed;
    for (k, mut spec) in s.nav.spawn_plan().into_iter().enumerate() {
        // Scouts' slots also fit hovercraft and amphibious units (clearance 1).
        if spec.profile_id == SCOUT && matches!(k % 10, 7 | 8) {
            let p = &profiles[if k % 10 == 7 { HOVER } else { AMPHIBIOUS } as usize];
            spec.radius *= p.unit_radius / scout_radius;
            spec.max_speed *= p.base_speed / scout_speed;
            spec.profile_id = p.id;
        }
        s.add(k as u64 + 1, spec);
    }
    let land_units = s.bodies.len() as u64;

    // Ships: ship-passable cells nearest the river point, two cells apart.
    let geom = s.nav.terrain.geom.clone();
    let grid = s.nav.hierarchy.grid(SHIP).clone();
    // North of the road's causeway, which cuts the river near (1290, 1650).
    let anchor = reference_cell(&geom, 1150, 1250);
    let ship_goal = s.nav.terrain.nearest_open_cell(reference_cell(&geom, 1050, 900), 128, |c| grid.is_passable(&s.nav.terrain, c));
    let mut cells: Vec<CellPos> = (0..geom.cells_y)
        .flat_map(|y| (0..geom.cells_x).map(move |x| CellPos::new(x, y)))
        .filter(|&c| grid.is_passable(&s.nav.terrain, c))
        .collect();
    cells.sort_by_key(|c| ((c.x - anchor.x).pow(2) + (c.y - anchor.y).pow(2), c.y, c.x));
    let ship = &profiles[SHIP as usize];
    let mut taken = BTreeSet::new();
    let mut placed = 0u32;
    for c in cells {
        if placed >= ships { break; }
        if (-1..=1).any(|dy| (-1..=1).any(|dx| taken.contains(&(c.x + dx, c.y + dy)))) { continue; }
        taken.insert((c.x, c.y));
        let spec = crate::service::UnitSpec { pos: geom.cell_center(c), radius: ship.unit_radius, max_speed: ship.base_speed, profile_id: SHIP };
        s.add(land_units + 1 + placed as u64, spec);
        placed += 1;
    }

    // Land units cross on the road's causeway; hovercraft and amphibious
    // units gather in the middle of the big lake, where only they can go.
    let profile_of = s.bodies.profile.clone();
    let ground: Vec<u64> = (1..=land_units).filter(|&e| matches!(profile_of[e as usize - 1], SCOUT | HEAVY)).collect();
    let crossers: Vec<u64> = (1..=land_units).filter(|&e| matches!(profile_of[e as usize - 1], HOVER | AMPHIBIOUS)).collect();
    let goal = cell_point(&s.nav, reference_cell(&geom, 1400, 1560));
    s.command(1, ground.into_iter(), goal);
    // In the big lake, clear of the road's causeway through its middle.
    let goal = cell_point(&s.nav, reference_cell(&geom, 1357, 1650));
    s.command(3, crossers.into_iter(), goal);
    if let (Some(goal), true) = (ship_goal, placed > 0) {
        let to = cell_point(&s.nav, goal);
        s.command(2, land_units + 1..=land_units + placed as u64, to);
    }
    s
}
