//! The navigation service as host systems.
//!
//! * `nav_spawn` (tick 0): spawns the map's units, each with a `Body`, a
//!   `NavAgent` and a `Goal` (the map's destination for it; the engine does
//!   not act on goals, the game's systems decide when to order units), and
//!   with [`NavPlugin::teams`] set, a `Team` and a `Sensor`.
//! * `nav` (every tick): one `NavService` step over every entity with a
//!   `Body` and a `NavAgent`.
//!
//! Each `nav` run:
//! 1. gathers the query's rows and sorts them by entity id;
//! 2. reconciles the service with them (a merge against the units' entities,
//!    kept sorted; skipped when nothing joined or left): units whose entity
//!    is gone are removed (one batch), new entities join in ascending entity
//!    order, taking their start position, radius, speed and profile from the
//!    row;
//! 3. turns the `MoveOrder` and `OrderPoint` events since its last run into
//!    group commands ([`group_orders`]);
//! 4. turns the `TerrainEdit` events since its last run into one terrain
//!    patch (event order; a later edit of a cell wins), see
//!    [`terrain_patch`], then adds and removes the `CostOverlay`s in event
//!    order (each its own patch);
//! 5. steps the service on the bodies in service order (positions,
//!    velocities and radii from the rows) on the service's own thread pool,
//!    then writes positions, velocities and each agent's mode back;
//! 6. emits a `TerrainChanged` for every terrain repair that landed in the
//!    step (a patch lands with its navigation repair a fixed sim-time delay
//!    after it was queued; patches queued while a repair runs land with the
//!    next one, under one event).
//!
//! The service's own state (cohorts, slots, fields, jobs) is the system's
//! digest part. Driven by the same plan, it reaches the same state hash as
//! the standalone [`ae_nav::scenario`] run (`tests/crossing.rs`).

use ae_abi::engine::{
    BODY, COST_OVERLAY, GOAL, MAP_START, MOVE_ORDER, NAV_AGENT, ORDER_POINT, SENSOR, TEAM, TERRAIN_CHANGED,
    TERRAIN_EDIT,
};
use ae_ecs::Access;
use ae_nav::math::WorldPos;
use ae_nav::nav::{MovementProfile, min_clearance_cells, profiles_for, water_profiles};
use ae_nav::scenario::{self, Plan};
use ae_nav::service::{BodyColumns, CommandKind, CommandPayload, NavConfig, NavService, UnitSpec};
use ae_nav::terrain::Medium;
use ae_nav::terrain::{CellEdit, SharedTerrain, TerrainPatch};
use ae_sdk::nav::{
    ANY_DEPTH, Body, CELL_SIZE, CLASS_BYTES, CLASS_NAME_MAX, CLASSES_DATA, CostOverlay, Goal, MapStart, MoveOrder,
    NavAgent, OrderPoint, Sensor, Team, TerrainChanged, TerrainEdit, changed, order, overlay, terrain,
};
use ae_sdk::{Component, Event};
use ae_sim::host::{HostCtx, HostDecl, HostInit, HostPlugin, HostSystem, bytes_of, expect_component};
use ae_sim::schedule::Cadence;
use ae_sim::schema::Schema;
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

/// Which map and units a world starts with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Map {
    /// [`scenario::water_crossing_plan`]: lakes and a river, all five
    /// movement classes.
    WaterCrossing,
    /// [`scenario::land_crossing_plan`]: pathsim's dry map.
    LandCrossing,
    /// [`scenario::coast_plan`]: the M16 water fixture (a coastline, a
    /// shelf, deep water and a river mouth), all five movement classes.
    Coast,
    /// [`NavPlugin::terrain`], a loaded map (`ae_map`), with the land map's
    /// movement classes; units of the map's own only when asked for.
    Loaded,
}

#[derive(Clone, Debug)]
pub struct NavPlugin {
    pub map: Map,
    /// Map side in cells (8 world units each); 2560 is pathsim's full map.
    pub map_cells: i32,
    /// Where landed ground changes are published (render, tools); `None`
    /// publishes nothing. Read-only sharing of the service's terrain, so
    /// it never affects the sim.
    pub ground: Option<Arc<GroundFeed>>,
    /// Allyteams the map's units are split into (0: none, and no `Team` or
    /// `Sensor` components): see [`NavPlugin::allies`] and
    /// [`NavPlugin::sensor_of`].
    pub teams: u32,
    /// The terrain of [`Map::Loaded`].
    pub terrain: Option<Arc<ae_nav::terrain::Terrain>>,
    /// The map's start points (world units), emitted by `map_starts`.
    pub starts: Vec<[f32; 2]>,
}

/// Line-of-sight radius per movement class of the crossing maps (scout,
/// heavy, ship, hovercraft, amphibious), world units.
pub const CLASS_LOS: [f32; 5] = [650.0, 300.0, 500.0, 450.0, 350.0];
/// One unit in this many carries a radar of [`RADAR_RANGE`].
pub const RADAR_EVERY: usize = 25;
pub const RADAR_RANGE: f32 = 2200.0;

/// The terrain as of the latest landed ground change, and the cells that
/// changed since the reader last took it.
#[derive(Clone, Debug)]
pub struct GroundUpdate {
    /// Terrain version (versions count patches, so they skip).
    pub version: u32,
    /// Inclusive cell rectangle `(x0, y0, x1, y1)` covering every cell
    /// whose ground (height, water depth, blocked, material) changed since
    /// the previous [`GroundFeed::take`].
    pub cells: (i32, i32, i32, i32),
    /// The whole authoritative terrain after the change (shared, immutable).
    pub terrain: SharedTerrain,
}

/// Landed ground changes, newest wins: the nav system publishes after each
/// step that landed one; a reader takes the latest terrain with the union
/// of the rectangles it has not seen, so a reader that falls behind still
/// refreshes everything that changed. Cost-only changes (cost edits,
/// overlays) are not published: they do not change how the ground looks.
#[derive(Debug, Default)]
pub struct GroundFeed {
    pending: Mutex<Option<GroundUpdate>>,
}

impl GroundFeed {
    pub fn publish(&self, version: u32, cells: (i32, i32, i32, i32), terrain: SharedTerrain) {
        let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        let cells = match pending.as_ref() {
            Some(old) => (
                old.cells.0.min(cells.0),
                old.cells.1.min(cells.1),
                old.cells.2.max(cells.2),
                old.cells.3.max(cells.3),
            ),
            None => cells,
        };
        *pending = Some(GroundUpdate {
            version,
            cells,
            terrain,
        });
    }

    /// The latest change and everything that changed before it since the
    /// last call, or `None` if nothing did.
    pub fn take(&self) -> Option<GroundUpdate> {
        self.pending.lock().unwrap_or_else(|p| p.into_inner()).take()
    }
}

impl NavPlugin {
    pub fn new(map: Map, map_cells: i32) -> Self {
        Self {
            map,
            map_cells,
            ground: None,
            teams: 0,
            terrain: None,
            // The coast fixture's own start points (M16.2): on dry land, so
            // `map_starts` and the skirmish game's bases use them instead
            // of its opposite-corner fallback.
            starts: match map {
                Map::Coast => scenario::coast_starts(map_cells),
                _ => Vec::new(),
            },
        }
    }

    /// A loaded map's terrain and start points ([`Map::Loaded`]).
    pub fn loaded(terrain: Arc<ae_nav::terrain::Terrain>, starts: Vec<[f32; 2]>) -> Self {
        Self {
            map: Map::Loaded,
            map_cells: terrain.geom.cells_x,
            ground: None,
            teams: 0,
            terrain: Some(terrain),
            starts,
        }
    }

    /// The allyteam of each of the map's units from their spawn x: units
    /// ranked by x (ties by plan order) and cut into `teams` equal shares,
    /// so the armies start side by side with a front between them.
    pub fn allies(&self, xs: &[f32]) -> Vec<u32> {
        let teams = self.teams.max(1) as usize;
        let mut order: Vec<usize> = (0..xs.len()).collect();
        order.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]).then(a.cmp(&b)));
        let mut allies = vec![0; xs.len()];
        for (rank, &i) in order.iter().enumerate() {
            allies[i] = (rank * teams / xs.len().max(1)) as u32;
        }
        allies
    }

    /// The sensors of the map's unit `index` (plan order) of movement class
    /// `profile`: line of sight by class ([`CLASS_LOS`]), and every
    /// [`RADAR_EVERY`]th unit a radar.
    pub fn sensor_of(&self, index: usize, profile: u8) -> Sensor {
        let los = CLASS_LOS.get(profile as usize).copied().unwrap_or(400.0);
        let radar = if index.is_multiple_of(RADAR_EVERY) {
            RADAR_RANGE
        } else {
            0.0
        };
        Sensor { los, radar }
    }

    /// The unit count in a game parameter: its low 24 bits (the game owns
    /// the rest).
    pub fn units(param: u64) -> u32 {
        (param & 0xFF_FFFF) as u32
    }

    /// The service (without units) and the plan for `units` units, with
    /// these movement classes (see [`Self::profiles`]).
    pub fn plan(&self, units: u32, workers: usize, profiles: Vec<MovementProfile>) -> (NavService, Plan) {
        match self.map {
            Map::WaterCrossing => scenario::water_crossing_plan_with(units, self.map_cells, workers, profiles),
            Map::LandCrossing => scenario::land_crossing_plan_with(units, self.map_cells, workers, profiles),
            Map::Coast => scenario::coast_plan_with(units, self.map_cells, workers, profiles),
            Map::Loaded => scenario::terrain_plan_with(
                units,
                (**self.terrain.as_ref().expect("Map::Loaded without a terrain")).clone(),
                workers,
                profiles,
            ),
        }
    }

    /// The map's own movement classes.
    pub fn map_profiles(&self) -> Vec<MovementProfile> {
        let cfg = NavConfig::default();
        match self.map {
            Map::WaterCrossing | Map::Coast => water_profiles(cfg.cell_size),
            Map::LandCrossing | Map::Loaded => profiles_for(cfg.cell_size, cfg.radius_scale),
        }
    }

    /// The classes a world uses: the module's ([`classes`]) if it declares
    /// any, else the map's. The map's plan spawns units of its own class
    /// indices, so the module must declare at least as many.
    pub fn profiles(&self, schema: &Schema) -> Result<Vec<MovementProfile>> {
        let map = self.map_profiles();
        let Some(game) = classes(schema)? else { return Ok(map) };
        ensure!(
            game.len() >= map.len(),
            "the module declares {} movement classes; the map's units use {}",
            game.len(),
            map.len()
        );
        Ok(game)
    }

    /// World extent in world units (the map is square, from the origin).
    pub fn extent(&self) -> f32 {
        self.map_cells as f32 * ae_nav::service::NavConfig::default().cell_size
    }
}

impl HostPlugin for NavPlugin {
    fn systems(&self) -> Vec<HostDecl> {
        let mut spawn = HostDecl::new("nav_spawn");
        spawn.creates = vec![BODY, NAV_AGENT, GOAL];
        if self.teams > 0 {
            spawn.creates.extend([TEAM, SENSOR]);
        }
        spawn.cadence = Cadence::Once;
        let mut nav = HostDecl::new("nav");
        // A dying unit leaves the service's rows: not moved, not colliding.
        nav.query = vec![
            (BODY, Access::Write),
            (NAV_AGENT, Access::Write),
            (ae_abi::engine::DYING, Access::Without),
        ];
        nav.reads_events = vec![MOVE_ORDER, ORDER_POINT, TERRAIN_EDIT, COST_OVERLAY];
        nav.emits = vec![TERRAIN_CHANGED];
        let mut starts = HostDecl::new("map_starts");
        starts.emits = vec![MAP_START];
        starts.cadence = Cadence::Once;
        vec![spawn, nav, starts]
    }

    fn check(&self, name: &str, schema: &Schema) -> Result<()> {
        expect_component(schema, Body::SCHEMA)?;
        expect_component(schema, NavAgent::SCHEMA)?;
        match name {
            "map_starts" => expect_component(schema, <MapStart as Event>::SCHEMA).map(drop),
            "nav_spawn" => {
                expect_component(schema, Goal::SCHEMA)?;
                if self.teams > 0 {
                    expect_component(schema, Team::SCHEMA)?;
                    expect_component(schema, Sensor::SCHEMA)?;
                }
                Ok(())
            }
            _ => {
                ensure!(
                    CELL_SIZE == ae_nav::service::NavConfig::default().cell_size,
                    "ae_sdk::nav::CELL_SIZE differs from the nav service's cell size"
                );
                for record in [
                    <MoveOrder as Event>::SCHEMA,
                    <OrderPoint as Event>::SCHEMA,
                    <TerrainEdit as Event>::SCHEMA,
                    <TerrainChanged as Event>::SCHEMA,
                    <CostOverlay as Event>::SCHEMA,
                ] {
                    expect_component(schema, record)?;
                }
                self.profiles(schema).map(drop)
            }
        }
    }

    fn instantiate(&self, init: &HostInit, schema: &Schema, names: &[&str]) -> Result<Vec<Box<dyn HostSystem>>> {
        let (service, plan) = self.plan(Self::units(init.param), init.workers, self.profiles(schema)?);
        // The spawner and the stepper are separate systems over one plan.
        let mut service = Some(service);
        let mut plan = Some(plan);
        names
            .iter()
            .map(|&name| -> Result<Box<dyn HostSystem>> {
                Ok(match name {
                    "nav_spawn" => Box::new(Spawner {
                        plan: plan.take().context("nav_spawn listed twice")?,
                        plugin: self.clone(),
                    }),
                    "nav" => Box::new(Nav::new(
                        service.take().context("nav listed twice")?,
                        self.ground.clone(),
                    )),
                    "map_starts" => Box::new(Starts {
                        starts: self.starts.clone(),
                        extent: self.extent(),
                    }),
                    other => bail!("no host system `{other}` in the nav plugin"),
                })
            })
            .collect()
    }
}

/// `map_starts`: one [`MapStart`] per start point, at tick 0.
struct Starts {
    starts: Vec<[f32; 2]>,
    extent: f32,
}

impl HostSystem for Starts {
    fn run(&mut self, ctx: &mut HostCtx<'_>) -> Result<()> {
        for (index, &[x, y]) in self.starts.iter().enumerate() {
            ctx.emit(
                MAP_START,
                bytes_of(&MapStart {
                    index: index as u32,
                    count: self.starts.len() as u32,
                    x,
                    y,
                    extent: self.extent,
                    reserved: 0,
                }),
            )?;
        }
        Ok(())
    }
}

struct Spawner {
    plan: Plan,
    plugin: NavPlugin,
}

impl HostSystem for Spawner {
    fn run(&mut self, ctx: &mut HostCtx<'_>) -> Result<()> {
        let orders = self.plan.order_of();
        let allies = if self.plugin.teams > 0 {
            self.plugin
                .allies(&self.plan.units.iter().map(|u| u.pos.x).collect::<Vec<_>>())
        } else {
            Vec::new()
        };
        for (index, (spec, order)) in self.plan.units.iter().zip(orders).enumerate() {
            let body = Body {
                x: spec.pos.x,
                y: spec.pos.y,
                vx: 0.0,
                vy: 0.0,
                radius: spec.radius,
            };
            let agent = NavAgent {
                max_speed: spec.max_speed,
                profile: spec.profile_id as u32,
                mode: 0,
            };
            let (group, to) = order.unwrap_or((0, spec.pos));
            let goal = Goal {
                x: to.x,
                y: to.y,
                group,
            };
            if self.plugin.teams > 0 {
                let (team, sensor) = (
                    Team { ally: allies[index] },
                    self.plugin.sensor_of(index, spec.profile_id),
                );
                ctx.spawn(&[
                    (BODY, bytes_of(&body)),
                    (NAV_AGENT, bytes_of(&agent)),
                    (GOAL, bytes_of(&goal)),
                    (TEAM, bytes_of(&team)),
                    (SENSOR, bytes_of(&sensor)),
                ])?;
            } else {
                ctx.spawn(&[
                    (BODY, bytes_of(&body)),
                    (NAV_AGENT, bytes_of(&agent)),
                    (GOAL, bytes_of(&goal)),
                ])?;
            }
        }
        Ok(())
    }
}

/// Row pointers of one entity, valid for one run.
#[derive(Clone, Copy)]
struct Row {
    entity: u64,
    body: *mut Body,
    agent: *mut NavAgent,
}

/// Per-tick buffers, kept to reuse their allocations.
#[derive(Default)]
struct Scratch {
    rows: Vec<Row>,
    /// `(entity, index in service order)` of every unit, by entity.
    units: Vec<(u64, u32)>,
    /// Row (in `rows`) of each unit, in service order.
    order: Vec<usize>,
    px: Vec<f32>,
    py: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    radius: Vec<f32>,
    modes: Vec<u8>,
}

struct Nav {
    service: NavService,
    used_groups: BTreeSet<u64>,
    scratch: Scratch,
    ground: Option<Arc<GroundFeed>>,
}

// Raw row pointers live only inside one `run`.
unsafe impl Send for Scratch {}

impl Nav {
    fn new(service: NavService, ground: Option<Arc<GroundFeed>>) -> Self {
        Self {
            service,
            used_groups: BTreeSet::new(),
            scratch: Scratch::default(),
            ground,
        }
    }
}

impl HostSystem for Nav {
    fn run(&mut self, ctx: &mut HostCtx<'_>) -> Result<()> {
        let nav = &mut self.service;
        let s = &mut self.scratch;

        // 1. Rows, by entity.
        s.rows.clear();
        for chunk in ctx.chunks() {
            let (bodies, agents) = unsafe { (chunk.column::<Body>(0), chunk.column::<NavAgent>(1)) };
            for (k, &entity) in chunk.entities.iter().enumerate() {
                s.rows.push(Row {
                    entity,
                    body: &mut bodies[k],
                    agent: &mut agents[k],
                });
            }
        }
        s.rows.sort_unstable_by_key(|r| r.entity);

        // 2. Reconcile units with rows: a merge of two entity-sorted lists,
        // skipped when the sets are equal (no unit joined or left).
        let same = s.rows.len() == s.units.len() && s.rows.iter().zip(&s.units).all(|(r, u)| r.entity == u.0);
        if !same {
            let (mut gone, mut new) = (Vec::new(), Vec::new());
            let (mut a, mut b) = (0, 0);
            while a < s.rows.len() || b < s.units.len() {
                let row = s.rows.get(a).map(|r| r.entity);
                let unit = s.units.get(b).map(|u| u.0);
                match (row, unit) {
                    (Some(r), Some(u)) if r == u => (a, b) = (a + 1, b + 1),
                    // In the rows only: a new unit.
                    (Some(r), u) if u.is_none_or(|u| r < u) => {
                        new.push(a);
                        a += 1;
                    }
                    // In the service only: its entity is gone.
                    (_, Some(u)) => {
                        gone.push(u);
                        b += 1;
                    }
                    _ => unreachable!("the loop runs while either list has items"),
                }
            }
            if !gone.is_empty() {
                nav.remove_units(&gone);
            }
            // New units join in ascending entity order.
            for &k in &new {
                let r = s.rows[k];
                let (b, a) = unsafe { (*r.body, *r.agent) };
                ensure!(
                    (a.profile as usize) < nav.profiles.len(),
                    "entity {:#x}: movement profile {} of {}",
                    r.entity,
                    a.profile,
                    nav.profiles.len()
                );
                ensure!(
                    b.radius > 0.0 && a.max_speed >= 0.0,
                    "entity {:#x}: bad body radius {} or speed {}",
                    r.entity,
                    b.radius,
                    a.max_speed
                );
                // Paths are only as wide as the class's clearance.
                let class = &nav.profiles[a.profile as usize];
                ensure!(
                    min_clearance_cells(b.radius, nav.cfg.cell_size) <= class.clearance_cells,
                    "entity {:#x}: body radius {} needs more than movement class `{}`'s clearance of {} cells",
                    r.entity,
                    b.radius,
                    class.name,
                    class.clearance_cells
                );
                nav.add_unit(
                    r.entity,
                    UnitSpec {
                        pos: WorldPos::new(b.x, b.y),
                        radius: b.radius,
                        max_speed: a.max_speed,
                        profile_id: a.profile as u8,
                    },
                );
            }
            s.units.clear();
            s.units.extend(nav.entities().enumerate().map(|(i, e)| (e, i as u32)));
            s.units.sort_unstable_by_key(|u| u.0);
        }
        debug_assert_eq!(s.rows.len(), s.units.len());

        // 3. Orders, grouped in order of each group's first event.
        let orders: Vec<MoveOrder> = ctx.events(MOVE_ORDER)?;
        let points: Vec<OrderPoint> = ctx.events(ORDER_POINT)?;
        for (command, facing) in group_orders(&orders, &points, |e| nav.local_id(e))? {
            ensure!(
                self.used_groups.insert(command.id),
                "move order group {} was used before: a group id is a nav command id and must be new",
                command.id
            );
            if command.unit_ids.is_empty() {
                continue;
            }
            match facing {
                Some(toward) => nav.queue_formation_facing(command, toward),
                None => nav.queue_command_at(command),
            }
        }

        // 4. Terrain edits, as one patch.
        let edits: Vec<TerrainEdit> = ctx.events(TERRAIN_EDIT)?;
        if !edits.is_empty() {
            let geom = &nav.terrain.geom;
            let patch = terrain_patch(&edits, geom.cells_x, geom.cells_y)?;
            if !patch.cells.is_empty() {
                nav.queue_terrain_patch(patch);
            }
        }
        // Cost overlays, in event order (each one patch).
        for o in ctx.events::<CostOverlay>(COST_OVERLAY)? {
            match o.op {
                overlay::ADD => {
                    ensure!(
                        (1..=254).contains(&o.cost),
                        "cost overlay {}: cost {} outside 1..=254",
                        o.id,
                        o.cost
                    );
                    ensure!(o.x0 <= o.x1 && o.y0 <= o.y1, "cost overlay {}: empty rectangle", o.id);
                    ensure!(
                        nav.add_cost_overlay(o.id, (o.x0, o.y0, o.x1, o.y1), o.cost as u8),
                        "cost overlay {} added twice",
                        o.id
                    );
                }
                overlay::REMOVE => ensure!(
                    nav.remove_cost_overlay(o.id),
                    "cost overlay {} removed but not in use",
                    o.id
                ),
                op => bail!("cost overlay {}: unknown op {op}", o.id),
            }
        }

        // 5. Step on the bodies, in service order. Rows and units are now
        // the same entities, both sorted: row k is unit `units[k].1`.
        let n = s.rows.len();
        s.order.resize(n, 0);
        for (k, u) in s.units.iter().enumerate() {
            s.order[u.1 as usize] = k;
        }
        for v in [&mut s.px, &mut s.py, &mut s.vx, &mut s.vy, &mut s.radius] {
            v.clear();
        }
        for &k in &s.order {
            let b = unsafe { *s.rows[k].body };
            s.px.push(b.x);
            s.py.push(b.y);
            s.vx.push(b.vx);
            s.vy.push(b.vy);
            s.radius.push(b.radius);
        }
        {
            ae_profile::zone!("nav_step");
            let mut cols = BodyColumns {
                pos_x: &mut s.px,
                pos_y: &mut s.py,
                vel_x: &mut s.vx,
                vel_y: &mut s.vy,
                radius: &s.radius,
            };
            nav.in_pool(|nav| nav.step(&mut cols));
        }
        s.modes.resize(n, 0);
        nav.write_modes(&mut s.modes);
        for (i, &k) in s.order.iter().enumerate() {
            let r = s.rows[k];
            unsafe {
                let b = &mut *r.body;
                (b.x, b.y, b.vx, b.vy) = (s.px[i], s.py[i], s.vx[i], s.vy[i]);
                (*r.agent).mode = s.modes[i] as u32;
            }
        }
        s.rows.clear();

        // 6. Terrain versions that took effect.
        for c in nav.take_terrain_changes() {
            let (x0, y0, x1, y1) = c.cells;
            let flags = if c.ground { changed::GROUND } else { 0 };
            ctx.emit(
                TERRAIN_CHANGED,
                bytes_of(&TerrainChanged {
                    version: c.version,
                    x0,
                    y0,
                    x1,
                    y1,
                    flags,
                }),
            )?;
            if let (true, Some(feed)) = (c.ground, &self.ground) {
                // The service's terrain is the newest landed version (this
                // change's, or a later one in the same step: the feed
                // keeps the union of the rectangles either way).
                feed.publish(nav.terrain.version, c.cells, Arc::clone(&nav.terrain));
            }
        }
        Ok(())
    }

    fn digest(&self) -> Option<u128> {
        Some(self.service.state_hash() as u128)
    }
}

/// One terrain patch from `edits`, in order, on a map of `cells_x` by
/// `cells_y` cells: each edit's rectangle clipped to the map, a later edit of
/// a cell winning field by field. Edits with unknown flags or values out of
/// range are errors (a game bug, not something to clamp silently).
pub fn terrain_patch(edits: &[TerrainEdit], cells_x: i32, cells_y: i32) -> Result<TerrainPatch> {
    let mut patch = TerrainPatch::new();
    for e in edits {
        ensure!(
            e.set & !(terrain::BLOCKED | terrain::HEIGHT | terrain::COST) == 0,
            "terrain edit with unknown flags {:#x}",
            e.set
        );
        let blocked = (e.set & terrain::BLOCKED != 0).then_some(e.blocked);
        let height = (e.set & terrain::HEIGHT != 0).then_some(e.height);
        let cost = (e.set & terrain::COST != 0).then_some(e.cost);
        ensure!(
            blocked.is_none_or(|b| b <= 1),
            "terrain edit: blocked must be 0 or 1, not {}",
            e.blocked
        );
        ensure!(
            height.is_none_or(|h| (0..=2400).contains(&h)),
            "terrain edit: height {} outside 0..=2400 decimetres",
            e.height
        );
        ensure!(
            cost.is_none_or(|c| (1..=255).contains(&c)),
            "terrain edit: cost {} outside 1..=255",
            e.cost
        );
        let (x0, y0, x1, y1) = (e.x0.max(0), e.y0.max(0), e.x1.min(cells_x - 1), e.y1.min(cells_y - 1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                patch.push(CellEdit {
                    x,
                    y,
                    blocked: blocked.map(|b| b == 1),
                    height: height.map(|h| h as i16),
                    cost: cost.map(|c| c as u8),
                    material: None,
                });
            }
        }
    }
    patch.finish();
    Ok(patch)
}

/// The movement classes a module declares (`ae_sdk::movement_classes!`, the
/// `nav.classes` data record), as nav profiles with ids in declaration
/// order; `None` without the declaration. Checked: 1..=255 classes, unique
/// names, a clearance that fits the nominal radius (at most 16 cells),
/// depths in range and `min_depth <= max_depth`.
pub fn classes(schema: &Schema) -> Result<Option<Vec<MovementProfile>>> {
    let Some(data) = schema.data.get(CLASSES_DATA) else {
        return Ok(None);
    };
    ensure!(
        !data.is_empty() && data.len().is_multiple_of(CLASS_BYTES) && data.len() / CLASS_BYTES <= 255,
        "`{CLASSES_DATA}`: {} bytes is not 1..=255 classes of {CLASS_BYTES}",
        data.len()
    );
    let word = |c: &[u8], at: usize| u32::from_le_bytes(c[at..at + 4].try_into().unwrap());
    let cell_size = NavConfig::default().cell_size;
    let mut out: Vec<MovementProfile> = Vec::new();
    for (id, c) in data.as_chunks::<CLASS_BYTES>().0.iter().enumerate() {
        let name_len = c[..CLASS_NAME_MAX]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(CLASS_NAME_MAX);
        let name = std::str::from_utf8(&c[..name_len]).context("movement class name is not UTF-8")?;
        ensure!(
            !name.is_empty() && c[name_len..CLASS_NAME_MAX].iter().all(|&b| b == 0),
            "movement class {id}: bad name"
        );
        ensure!(
            !out.iter().any(|p| p.name == name),
            "two movement classes are named `{name}`"
        );
        let [clearance, slope, radius, speed, min_depth, max_depth, hover, reserved] =
            [32, 36, 40, 44, 48, 52, 56, 60].map(|at| word(c, at));
        let (radius, speed) = (f32::from_bits(radius), f32::from_bits(speed));
        let depth = |d: u32| -> Result<i16> {
            if d == ANY_DEPTH {
                return Ok(i16::MAX);
            }
            i16::try_from(d)
                .ok()
                .filter(|&d| d < i16::MAX)
                .with_context(|| format!("movement class `{name}`: depth {d} decimetres out of range"))
        };
        let medium = Medium {
            min_depth: depth(min_depth)?,
            max_depth: depth(max_depth)?,
            hover: hover == 1,
        };
        ensure!(hover <= 1 && reserved == 0, "movement class `{name}`: bad flags");
        ensure!(
            medium.min_depth <= medium.max_depth,
            "movement class `{name}`: min depth above max depth"
        );
        ensure!(
            radius.is_finite() && radius > 0.0 && speed.is_finite() && speed >= 0.0,
            "movement class `{name}`: radius {radius}, speed {speed}"
        );
        let clearance = clearance as i32;
        ensure!(
            (min_clearance_cells(radius, cell_size)..=16).contains(&clearance),
            "movement class `{name}`: clearance {clearance} cells does not fit radius {radius} (or exceeds 16)"
        );
        ensure!(
            (1..i16::MAX as u32).contains(&slope),
            "movement class `{name}`: max slope {slope}"
        );
        out.push(MovementProfile {
            id: id as u8,
            name: name.into(),
            clearance_cells: clearance,
            max_slope: slope as i16,
            unit_radius: radius,
            base_speed: speed,
            medium,
        });
    }
    Ok(Some(out))
}

/// Group commands from one run's orders and guide points: orders with the
/// same `group` form one command whose id is the group, in order of each
/// group's first order (events arrive in key order), with the kind and
/// point of that first order and its units' ids ascending (orders for
/// entities without a unit, gone or not navigable, are dropped). A group's
/// guide points are its `OrderPoint`s by index; a formation's facing (the
/// point its front faces) is its point of index `FACING`. Errors (game
/// bugs): a group 0, an unknown kind, two kinds in one group, points for a
/// group without orders in the same run, for a move or stop, beyond
/// `MAX_POINTS`, with a gap, two copies of one point that differ, or a
/// facing for a route.
pub fn group_orders(
    orders: &[MoveOrder],
    points: &[OrderPoint],
    local_id: impl Fn(u64) -> Option<u32>,
) -> Result<Vec<(CommandPayload, Option<WorldPos>)>> {
    struct Group {
        id: u64,
        kind: u32,
        to: WorldPos,
        units: Vec<u32>,
        points: Vec<Option<WorldPos>>,
        facing: Option<WorldPos>,
    }
    let mut groups: Vec<Group> = Vec::new();
    let mut slot: BTreeMap<u64, usize> = BTreeMap::new();
    for o in orders {
        ensure!(o.group != 0, "move order for {:#x} without a group", o.entity.0);
        ensure!(
            o.kind <= order::FORMATION,
            "move order for {:#x}: unknown kind {}",
            o.entity.0,
            o.kind
        );
        let g = *slot.entry(o.group).or_insert_with(|| {
            groups.push(Group {
                id: o.group,
                kind: o.kind,
                to: WorldPos::new(o.x, o.y),
                units: Vec::new(),
                points: Vec::new(),
                facing: None,
            });
            groups.len() - 1
        });
        let group = &mut groups[g];
        ensure!(
            group.kind == o.kind,
            "move order group {}: kinds {} and {}",
            o.group,
            group.kind,
            o.kind
        );
        if let Some(id) = local_id(o.entity.0) {
            group.units.push(id);
        }
    }
    for p in points {
        let Some(&g) = slot.get(&p.group) else {
            bail!("guide point for group {} without orders in the same tick", p.group)
        };
        let group = &mut groups[g];
        ensure!(
            matches!(group.kind, order::ROUTE | order::FORMATION),
            "guide point for group {}, whose orders are not a route or formation",
            p.group
        );
        let at = WorldPos::new(p.x, p.y);
        let same = |q: WorldPos| [q.x, q.y].map(f32::to_bits) == [at.x, at.y].map(f32::to_bits);
        if p.index == order::FACING {
            ensure!(
                group.kind == order::FORMATION,
                "facing for group {}, which is not a formation",
                p.group
            );
            ensure!(
                group.facing.is_none_or(same),
                "facing of group {}: two different points",
                p.group
            );
            group.facing = Some(at);
            continue;
        }
        ensure!(
            p.index < order::MAX_POINTS,
            "guide point {} of group {}: at most {} points",
            p.index,
            p.group,
            order::MAX_POINTS
        );
        let i = p.index as usize;
        if group.points.len() <= i {
            group.points.resize(i + 1, None);
        }
        match group.points[i] {
            None => group.points[i] = Some(at),
            Some(q) => ensure!(
                [q.x, q.y].map(f32::to_bits) == [at.x, at.y].map(f32::to_bits),
                "guide point {i} of group {}: ({}, {}) and ({}, {})",
                p.group,
                q.x,
                q.y,
                at.x,
                at.y
            ),
        }
    }
    groups
        .into_iter()
        .map(|mut g| {
            let guide_points = g
                .points
                .iter()
                .enumerate()
                .map(|(i, p)| p.with_context(|| format!("group {}: guide point {i} missing", g.id)))
                .collect::<Result<Vec<_>>>()?;
            let kind = match g.kind {
                order::MOVE => CommandKind::MoveTo,
                order::STOP => CommandKind::Stop,
                order::ROUTE => CommandKind::FollowCurve { guide_points },
                _ => CommandKind::CurveFormation { guide_points },
            };
            g.units.sort_unstable();
            g.units.dedup();
            Ok((
                CommandPayload {
                    id: g.id,
                    unit_ids: g.units,
                    kind,
                    requested_point: g.to,
                    corridor_tolerance_world: CORRIDOR_TOLERANCE,
                },
                g.facing,
            ))
        })
        .collect()
}

/// How far (world units) a route may stray from its guide points.
pub const CORRIDOR_TOLERANCE: f32 = 48.0;

#[cfg(test)]
mod tests {
    use super::*;
    use ae_sdk::Entity;

    fn point(group: u64, index: u32, x: f32) -> OrderPoint {
        OrderPoint {
            group,
            index,
            x,
            y: 0.0,
            reserved: 0,
        }
    }

    /// Entity `e` is unit `e` (entities above 100 have no unit).
    fn run(orders: &[MoveOrder], points: &[OrderPoint]) -> Result<Vec<CommandPayload>> {
        Ok(group_orders(orders, points, |e| (e <= 100).then_some(e as u32))?
            .into_iter()
            .map(|(c, _)| c)
            .collect())
    }

    #[test]
    fn groups_keep_first_order_kind_point_and_sorted_units() -> Result<()> {
        let orders = [
            MoveOrder::route(Entity(5), 10.0, 20.0, 7),
            MoveOrder::stop(Entity(3), 2),
            MoveOrder::route(Entity(1), 99.0, 99.0, 7),
            MoveOrder::route(Entity(500), 0.0, 0.0, 7),
            MoveOrder::route(Entity(5), 10.0, 20.0, 7),
        ];
        // Every unit sends the points; copies agree, any order.
        let points = [point(7, 1, 2.0), point(7, 0, 1.0), point(7, 1, 2.0), point(7, 0, 1.0)];
        let out = run(&orders, &points)?;
        assert_eq!(out.iter().map(|c| c.id).collect::<Vec<_>>(), [7, 2]);
        assert_eq!(out[0].unit_ids, [1, 5]);
        assert_eq!(out[0].requested_point, WorldPos::new(10.0, 20.0));
        let CommandKind::FollowCurve { guide_points } = &out[0].kind else {
            panic!("{:?}", out[0].kind)
        };
        assert_eq!(guide_points, &[WorldPos::new(1.0, 0.0), WorldPos::new(2.0, 0.0)]);
        assert!(matches!(out[1].kind, CommandKind::Stop));
        let formation = run(&[MoveOrder::formation(Entity(1), 0.0, 0.0, 9)], &[])?;
        assert!(matches!(&formation[0].kind, CommandKind::CurveFormation { guide_points } if guide_points.is_empty()));
        // A facing, sent by every unit.
        let orders = [
            MoveOrder::formation(Entity(1), 0.0, 0.0, 9),
            MoveOrder::formation(Entity(2), 0.0, 0.0, 9),
        ];
        let facing = [
            point(9, order::FACING, 5.0),
            point(9, 0, 1.0),
            point(9, order::FACING, 5.0),
        ];
        let out = group_orders(&orders, &facing, |e| Some(e as u32))?;
        assert_eq!(out[0].1, Some(WorldPos::new(5.0, 0.0)));
        assert!(matches!(&out[0].0.kind, CommandKind::CurveFormation { guide_points } if guide_points.len() == 1));
        Ok(())
    }

    /// A module schema holding the record `movement_classes!` writes for
    /// two classes (and an empty `schedule!`), parsed as the host does.
    fn schema_with(classes: &[ae_sdk::nav::MovementClass; 2]) -> Schema {
        let mut section = ae_sdk::nav::classes_record::<{ ae_sdk::nav::classes_record_len(2) }>(classes).to_vec();
        for word in [ae_abi::KIND_ORDER, 16, 0, 0] {
            section.extend_from_slice(&word.to_le_bytes());
        }
        Schema::parse(&section).unwrap()
    }

    const TANK: ae_sdk::nav::MovementClass = ae_sdk::nav::MovementClass {
        name: "tank",
        clearance_cells: 1,
        max_slope: 95,
        radius: 4.0,
        speed: 80.0,
        min_depth: 0,
        max_depth: 20,
        hover: false,
    };

    #[test]
    fn declared_classes_become_profiles() -> Result<()> {
        let boat = ae_sdk::nav::MovementClass {
            name: "boat",
            clearance_cells: 2,
            min_depth: 30,
            max_depth: ANY_DEPTH,
            ..TANK
        };
        let got = classes(&schema_with(&[TANK, boat]))?.unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(
            (
                got[1].id,
                got[1].name.as_str(),
                got[1].clearance_cells,
                got[1].max_slope
            ),
            (1, "boat", 2, 95)
        );
        assert_eq!(got[0].medium, Medium::land(20));
        assert_eq!(got[1].medium, Medium::ship(30));
        assert_eq!((got[0].unit_radius, got[0].base_speed), (4.0, 80.0));
        assert!(classes(&Schema::default())?.is_none());
        Ok(())
    }

    #[test]
    fn bad_classes_are_errors() {
        let cases = [
            ([TANK, TANK], "two movement classes"),
            (
                [
                    TANK,
                    ae_sdk::nav::MovementClass {
                        name: "x",
                        clearance_cells: 0,
                        radius: 9.0,
                        ..TANK
                    },
                ],
                "does not fit",
            ),
            (
                [
                    TANK,
                    ae_sdk::nav::MovementClass {
                        name: "x",
                        min_depth: 30,
                        max_depth: 20,
                        ..TANK
                    },
                ],
                "min depth",
            ),
            (
                [
                    TANK,
                    ae_sdk::nav::MovementClass {
                        name: "x",
                        max_depth: 40_000,
                        ..TANK
                    },
                ],
                "out of range",
            ),
            (
                [
                    TANK,
                    ae_sdk::nav::MovementClass {
                        name: "x",
                        max_slope: 0,
                        ..TANK
                    },
                ],
                "max slope",
            ),
            (
                [
                    TANK,
                    ae_sdk::nav::MovementClass {
                        name: "x",
                        radius: f32::NAN,
                        ..TANK
                    },
                ],
                "radius",
            ),
        ];
        for (c, error) in cases {
            let e = classes(&schema_with(&c)).unwrap_err().to_string();
            assert!(e.contains(error), "expected `{error}`, got `{e}`");
        }
    }

    #[test]
    fn game_bugs_are_errors() {
        let route = MoveOrder::route(Entity(1), 0.0, 0.0, 7);
        let formation = MoveOrder::formation(Entity(1), 0.0, 0.0, 7);
        let cases: [(&[MoveOrder], &[OrderPoint], &str); 10] = [
            (&[route], &[point(7, order::FACING, 0.0)], "not a formation"),
            (
                &[formation],
                &[point(7, order::FACING, 0.0), point(7, order::FACING, 1.0)],
                "two different points",
            ),
            (&[MoveOrder::move_to(Entity(1), 0.0, 0.0, 0)], &[], "without a group"),
            (&[MoveOrder { kind: 4, ..route }], &[], "unknown kind"),
            (&[route, MoveOrder::stop(Entity(2), 7)], &[], "kinds"),
            (&[route], &[point(8, 0, 0.0)], "without orders"),
            (
                &[MoveOrder::move_to(Entity(1), 0.0, 0.0, 7)],
                &[point(7, 0, 0.0)],
                "not a route or formation",
            ),
            (&[route], &[point(7, order::MAX_POINTS, 0.0)], "at most"),
            (&[route], &[point(7, 1, 0.0)], "missing"),
            (&[route], &[point(7, 0, 0.0), point(7, 0, 1.0)], "guide point 0"),
        ];
        for (orders, points, error) in cases {
            let e = run(orders, points).unwrap_err().to_string();
            assert!(e.contains(error), "expected `{error}`, got `{e}`");
        }
    }
}
