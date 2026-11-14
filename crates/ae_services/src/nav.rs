//! The navigation service as host systems.
//!
//! * `nav_spawn` (tick 0): spawns the map's units, each with a `Body`, a
//!   `NavAgent` and a `Goal` (the map's destination for it; the engine does
//!   not act on goals, the game's systems decide when to order units).
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
//! 3. turns the `MoveOrder` events since its last run into group commands:
//!    orders with the same `group` form one command whose id is the group,
//!    queued in order of each group's first event (events arrive in key
//!    order), unit ids ascending;
//! 4. turns the `TerrainEdit` events since its last run into one terrain
//!    patch (event order; a later edit of a cell wins), see
//!    [`terrain_patch`];
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

use ae_abi::engine::{BODY, GOAL, MOVE_ORDER, NAV_AGENT, TERRAIN_CHANGED, TERRAIN_EDIT};
use ae_ecs::Access;
use ae_nav::math::WorldPos;
use ae_nav::scenario::{self, Plan};
use ae_nav::service::{BodyColumns, CommandKind, CommandPayload, NavService, UnitSpec};
use ae_nav::terrain::{CellEdit, TerrainPatch};
use ae_sdk::nav::{Body, CELL_SIZE, Goal, MoveOrder, NavAgent, TerrainChanged, TerrainEdit, terrain};
use ae_sdk::{Component, Event};
use ae_sim::host::{HostCtx, HostDecl, HostInit, HostPlugin, HostSystem, bytes_of, expect_component};
use ae_sim::schedule::Cadence;
use ae_sim::schema::Schema;
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};

/// Which map and units a world starts with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Map {
    /// [`scenario::water_crossing_plan`]: lakes and a river, all five
    /// movement classes.
    WaterCrossing,
    /// [`scenario::land_crossing_plan`]: pathsim's dry map.
    LandCrossing,
}

#[derive(Clone, Copy, Debug)]
pub struct NavPlugin {
    pub map: Map,
    /// Map side in cells (8 world units each); 2560 is pathsim's full map.
    pub map_cells: i32,
}

impl NavPlugin {
    /// The unit count in a game parameter: its low 24 bits (the game owns
    /// the rest).
    pub fn units(param: u64) -> u32 {
        (param & 0xFF_FFFF) as u32
    }

    /// The service (without units) and the plan for `units` units.
    pub fn plan(&self, units: u32, workers: usize) -> (NavService, Plan) {
        match self.map {
            Map::WaterCrossing => scenario::water_crossing_plan(units, self.map_cells, workers),
            Map::LandCrossing => scenario::land_crossing_plan(units, self.map_cells, workers),
        }
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
        spawn.cadence = Cadence::Once;
        let mut nav = HostDecl::new("nav");
        nav.query = vec![(BODY, Access::Write), (NAV_AGENT, Access::Write)];
        nav.reads_events = vec![MOVE_ORDER, TERRAIN_EDIT];
        nav.emits = vec![TERRAIN_CHANGED];
        vec![spawn, nav]
    }

    fn check(&self, name: &str, schema: &Schema) -> Result<()> {
        expect_component(schema, Body::SCHEMA)?;
        expect_component(schema, NavAgent::SCHEMA)?;
        match name {
            "nav_spawn" => expect_component(schema, Goal::SCHEMA).map(drop),
            _ => {
                ensure!(CELL_SIZE == ae_nav::service::NavConfig::default().cell_size, "ae_sdk::nav::CELL_SIZE differs from the nav service's cell size");
                for record in [<MoveOrder as Event>::SCHEMA, <TerrainEdit as Event>::SCHEMA, <TerrainChanged as Event>::SCHEMA] {
                    expect_component(schema, record)?;
                }
                Ok(())
            }
        }
    }

    fn instantiate(&self, init: &HostInit, names: &[&str]) -> Result<Vec<Box<dyn HostSystem>>> {
        let (service, plan) = self.plan(Self::units(init.param), init.workers);
        // The spawner and the stepper are separate systems over one plan.
        let mut service = Some(service);
        let mut plan = Some(plan);
        names
            .iter()
            .map(|&name| -> Result<Box<dyn HostSystem>> {
                Ok(match name {
                    "nav_spawn" => Box::new(Spawner { plan: plan.take().context("nav_spawn listed twice")? }),
                    "nav" => Box::new(Nav::new(service.take().context("nav listed twice")?)),
                    other => bail!("no host system `{other}` in the nav plugin"),
                })
            })
            .collect()
    }
}

struct Spawner {
    plan: Plan,
}

impl HostSystem for Spawner {
    fn run(&mut self, ctx: &mut HostCtx<'_>) -> Result<()> {
        let orders = self.plan.order_of();
        for (spec, order) in self.plan.units.iter().zip(orders) {
            let body = Body { x: spec.pos.x, y: spec.pos.y, vx: 0.0, vy: 0.0, radius: spec.radius };
            let agent = NavAgent { max_speed: spec.max_speed, profile: spec.profile_id as u32, mode: 0 };
            let (group, to) = order.unwrap_or((0, spec.pos));
            let goal = Goal { x: to.x, y: to.y, group };
            ctx.spawn(&[(BODY, bytes_of(&body)), (NAV_AGENT, bytes_of(&agent)), (GOAL, bytes_of(&goal))])?;
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
}

// Raw row pointers live only inside one `run`.
unsafe impl Send for Scratch {}

impl Nav {
    fn new(service: NavService) -> Self {
        Self { service, used_groups: BTreeSet::new(), scratch: Scratch::default() }
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
                s.rows.push(Row { entity, body: &mut bodies[k], agent: &mut agents[k] });
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
                ensure!((a.profile as usize) < nav.profiles.len(), "entity {:#x}: movement profile {} of {}", r.entity, a.profile, nav.profiles.len());
                ensure!(b.radius > 0.0 && a.max_speed >= 0.0, "entity {:#x}: bad body radius {} or speed {}", r.entity, b.radius, a.max_speed);
                nav.add_unit(r.entity, UnitSpec { pos: WorldPos::new(b.x, b.y), radius: b.radius, max_speed: a.max_speed, profile_id: a.profile as u8 });
            }
            s.units.clear();
            s.units.extend(nav.entities().enumerate().map(|(i, e)| (e, i as u32)));
            s.units.sort_unstable_by_key(|u| u.0);
        }
        debug_assert_eq!(s.rows.len(), s.units.len());

        // 3. Orders, grouped in order of each group's first event.
        let orders: Vec<MoveOrder> = ctx.events(MOVE_ORDER)?;
        let mut groups: Vec<(u64, WorldPos, Vec<u32>)> = Vec::new();
        let mut slot: BTreeMap<u64, usize> = BTreeMap::new();
        for o in &orders {
            ensure!(o.group != 0, "move order for {:#x} without a group", o.entity.0);
            let g = *slot.entry(o.group).or_insert_with(|| {
                groups.push((o.group, WorldPos::new(o.x, o.y), Vec::new()));
                groups.len() - 1
            });
            // Orders for entities without a unit (gone, or not navigable) are dropped.
            if let Some(id) = nav.local_id(o.entity.0) {
                groups[g].2.push(id);
            }
        }
        for (group, to, mut ids) in groups {
            ensure!(self.used_groups.insert(group), "move order group {group} was used before: a group id is a nav command id and must be new");
            ids.sort_unstable();
            ids.dedup();
            if !ids.is_empty() {
                nav.queue_command_at(CommandPayload { id: group, unit_ids: ids, kind: CommandKind::MoveTo, requested_point: to, corridor_tolerance_world: 48.0 });
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
            let mut cols = BodyColumns { pos_x: &mut s.px, pos_y: &mut s.py, vel_x: &mut s.vx, vel_y: &mut s.vy, radius: &s.radius };
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
            ctx.emit(TERRAIN_CHANGED, bytes_of(&TerrainChanged { version: c.version, x0, y0, x1, y1 }))?;
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
        ensure!(e.set & !(terrain::BLOCKED | terrain::HEIGHT | terrain::COST) == 0, "terrain edit with unknown flags {:#x}", e.set);
        let blocked = (e.set & terrain::BLOCKED != 0).then_some(e.blocked);
        let height = (e.set & terrain::HEIGHT != 0).then_some(e.height);
        let cost = (e.set & terrain::COST != 0).then_some(e.cost);
        ensure!(blocked.is_none_or(|b| b <= 1), "terrain edit: blocked must be 0 or 1, not {}", e.blocked);
        ensure!(height.is_none_or(|h| (0..=2400).contains(&h)), "terrain edit: height {} outside 0..=2400 decimetres", e.height);
        ensure!(cost.is_none_or(|c| (1..=255).contains(&c)), "terrain edit: cost {} outside 1..=255", e.cost);
        let (x0, y0, x1, y1) = (e.x0.max(0), e.y0.max(0), e.x1.min(cells_x - 1), e.y1.min(cells_y - 1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                patch.push(CellEdit { x, y, blocked: blocked.map(|b| b == 1), height: height.map(|h| h as i16), cost: cost.map(|c| c as u8), material: None });
            }
        }
    }
    patch.finish();
    Ok(patch)
}
