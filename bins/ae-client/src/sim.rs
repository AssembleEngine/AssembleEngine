//! The sim beside the renderer: the crossing game (`games/crossing`) with
//! the nav and spatial host systems on the same map the terrain shows,
//! ticked at the nav service's rate on its own thread. After every tick the
//! units' `Body` and `NavAgent` fields are captured
//! ([`ae_sim::runtime::Runtime::capture`]) and turned into the renderer's
//! [`InstanceStream`]: rows matched by entity against the previous capture
//! for the interpolation start and the heading. Stand-in for M4's copy
//! layer, which will publish filtered view tables instead.

use ae_nav::math::TICK_SECONDS;
use ae_services::nav::{GroundFeed, Map, NavPlugin};
use ae_services::spatial::SpatialPlugin;
use ae_render::terrain::app::{VisionFeed, VisionFrame};
use ae_render::terrain::renderer::VisionGrid;
use ae_render::decals::{Blend, Decal, DecalFeed, layer};
use ae_render::particles::{ParticleFeed, explosion};
use ae_render::units::{ATTRIBUTE_COUNT, InstanceStream, UnitClass, UnitFeed};
use ae_sensors::{Bits, SensorGrid};
use ae_services::view::{SENSOR_CELL, ViewPipeline};
use ae_view::{Observer, RowState, ViewTable};
use ae_sim::capture::{Capture, CaptureSpec};
use ae_sim::input::{Command, TickBundle};
use ae_sim::runtime::{Runtime, RuntimeConfig, World};
use ae_sdk::Entity;
use ae_sdk::nav::MoveOrder;
use anyhow::{Context, Result};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// What the renderer needs of a unit, by field.
const FIELDS: [(&str, &str); 6] = [("Body", "x"), ("Body", "y"), ("Body", "vx"), ("Body", "vy"), ("Body", "radius"), ("NavAgent", "profile")];
const F_X: usize = 0;
const F_Y: usize = 1;
const F_VX: usize = 2;
const F_VY: usize = 3;
const F_RADIUS: usize = 4;
const F_PROFILE: usize = 5;

#[derive(Clone, Debug)]
pub struct SimOptions {
    pub map: Map,
    pub map_cells: i32,
    pub units: u32,
    pub workers: usize,
    /// The crossing game's `quake` tick (1..=255; 0: none): rock squares,
    /// then raised squares a tick later, landing with the nav repairs.
    pub quake: u8,
    /// Allyteams (`NavPlugin::teams`; 0: none).
    pub teams: u32,
    /// Draw the world as this allyteam sees it (needs teams): its view
    /// table and its vision on the terrain. `None`: everything.
    pub observer: Option<u32>,
    /// `games/<game>`: `crossing` or `skirmish`.
    pub game: String,
    /// The game's content: package sources and content directories, in
    /// mount order (its catalogue spawns its units).
    pub mounts: Vec<std::path::PathBuf>,
    /// The game parameter, if not the game's default.
    pub param: Option<u64>,
    /// The sim module from a package (else `games/<game>` is built).
    pub wasm: Option<Arc<Vec<u8>>>,
    /// A map package's terrain and start points ([`Map::Loaded`]).
    pub loaded_map: Option<(Arc<ae_nav::terrain::Terrain>, Vec<[f32; 2]>)>,
}

/// A player's action from the window, in world coordinates.
#[derive(Clone, Copy, Debug)]
pub enum UiInput {
    /// Select the observer's own units within `radius` of `(x, y)`.
    Select { x: f32, y: f32, radius: f32 },
    /// Order the selection to `(x, y)`: in the skirmish game an
    /// attack-move with `attack`, added to each unit's queue with `append`
    /// (other games: a plain nav move).
    Move { x: f32, y: f32, attack: bool, append: bool },
}

/// What the window hands the sim thread.
pub type UiQueue = Arc<Mutex<Vec<UiInput>>>;

/// The slowest the client goes. There is no upper limit: `+` doubles the
/// multiplier, and once the requested tick length is below what a tick
/// costs, the sim thread simply runs flat out.
pub const MIN_SPEED: f32 = 0.25;

/// The sim's tick-rate multiplier, shared by the window (which changes it)
/// and the sim thread (which schedules ticks by it).
#[derive(Clone, Debug)]
pub struct SimSpeed(Arc<AtomicU32>);

impl Default for SimSpeed {
    fn default() -> Self {
        Self(Arc::new(AtomicU32::new(1.0f32.to_bits())))
    }
}

impl SimSpeed {
    pub fn current(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }

    fn set(&self, speed: f32) -> f32 {
        self.0.store(speed.to_bits(), Ordering::Relaxed);
        speed
    }

    /// Twice as fast, with no upper limit (the sim becomes compute-bound).
    pub fn faster(&self) -> f32 {
        self.set((self.current() * 2.0).min(f32::MAX))
    }

    /// Half as fast (clamped at [`MIN_SPEED`]).
    pub fn slower(&self) -> f32 {
        self.set((self.current() * 0.5).max(MIN_SPEED))
    }

    /// For the title bar: `0.5x`, `1x`, `16x`, `32x`, `64x`.
    pub fn label(&self) -> String {
        let speed = self.current();
        if speed >= 1e9 {
            format!("{speed:.0e}x")
        } else {
            format!("{speed}x")
        }
    }
}

/// Looks of the skirmish game's `Look.class` (own; enemies tinted).
const SKIRMISH_LOOKS: [[f32; 3]; 4] = [[0.55, 0.62, 0.75], [0.95, 0.75, 0.10], [0.20, 0.75, 0.30], [0.75, 0.20, 0.85]];

/// Radius drawn for a blip (its real one is line-of-sight only).
const BLIP_RADIUS: f32 = 5.0;
const ENEMY_TINT: [f32; 3] = [0.95, 0.15, 0.1];

impl SimOptions {
    fn plugin(&self) -> NavPlugin {
        let base = match &self.loaded_map {
            Some((terrain, starts)) => NavPlugin::loaded(terrain.clone(), starts.clone()),
            None => NavPlugin::new(self.map, self.map_cells),
        };
        NavPlugin { teams: if self.skirmish() { 0 } else { self.teams }, ..base }
    }

    pub fn skirmish(&self) -> bool {
        self.game == "skirmish"
    }

    /// Allyteams the views cover (the skirmish always has two).
    pub fn allies(&self) -> u32 {
        if self.skirmish() { 2 } else { self.teams }
    }

    /// The vision grid of an observer (the copy layer's sensor grid).
    pub fn vision_grid(&self) -> VisionGrid {
        let g = SensorGrid::new(self.plugin().extent(), self.plugin().extent(), SENSOR_CELL);
        VisionGrid { width: g.width, height: g.height, cell_size: g.cell_size }
    }

    /// The game parameter: `--param`, or the game's default (crossing:
    /// unit count and quake tick; skirmish: the map side and 8 tanks each).
    pub fn param(&self) -> u64 {
        self.param.unwrap_or(if self.skirmish() { (self.map_cells as u64 * 8) << 24 | 8 << 40 } else { self.units as u64 | (self.quake as u64) << 24 })
    }

    /// One look per movement class of the map (the crossing game declares
    /// the same classes): ships and hovercraft float. With an observer, the
    /// same classes again for enemies (tinted red), then one for blips
    /// (grey; their class is not known).
    /// The own looks, then enemies tinted red, a grey blip and a white
    /// selected look.
    pub fn classes(&self) -> Vec<UnitClass> {
        let own = self.own_classes();
        if self.observer.is_none() {
            return own;
        }
        let tint = |c: [f32; 3]| std::array::from_fn(|k| c[k] * 0.35 + ENEMY_TINT[k] * 0.65);
        let enemy = own.iter().map(|c| UnitClass { color: tint(c.color), floats: c.floats });
        own.iter().copied().chain(enemy).chain([UnitClass { color: [0.6, 0.6, 0.6], floats: false }, UnitClass { color: [1.0, 1.0, 1.0], floats: false }]).collect()
    }

    fn own_classes(&self) -> Vec<UnitClass> {
        const COLORS: [[f32; 3]; 5] = [[0.95, 0.75, 0.10], [0.80, 0.12, 0.08], [0.92, 0.92, 0.96], [0.10, 0.80, 0.90], [0.75, 0.20, 0.85]];
        if self.skirmish() {
            return SKIRMISH_LOOKS.iter().map(|&color| UnitClass { color, floats: false }).collect();
        }
        self.plugin().map_profiles().iter().enumerate().map(|(i, p)| UnitClass { color: COLORS[i % COLORS.len()], floats: p.medium.on_surface() }).collect()
    }
}

/// The sim and its world, on the thread that made them.
pub struct Sim {
    /// Before `rt`: fields drop in order, and the world's columns live in
    /// the runtime's memory.
    world: World,
    rt: Runtime,
    workers: usize,
    spec: CaptureSpec,
    capture: Capture,
    streamer: Streamer,
    view: Option<ViewState>,
    classes: u32,
    /// Skirmish: the bots of the allyteams no player plays, and the game's
    /// unit command stream.
    bots: Vec<ae_bot::SkirmishBot>,
    unit_command: Option<u64>,
    /// Explosions for units that die where the observer sees.
    effects: Option<Arc<Effects>>,
    /// The last capture's units, by entity: `(entity, x, y, radius)`.
    last_units: Vec<(u64, f32, f32, f32)>,
    /// The observer's line of sight at the last tick.
    last_los: Option<Bits>,
}

/// One observer's copy layer: the pipeline, the observer, what it has ever
/// seen, and where its vision goes.
struct ViewState {
    pipeline: ViewPipeline,
    observer: [Observer; 1],
    explored: Bits,
    vision: Option<Arc<VisionFeed>>,
    table: ViewTable,
    /// Spec fields the stream reads.
    fields: [usize; 6],
    /// Milliseconds of the last tick: capture + coverage, view build, rows
    /// + stream.
    ms: [f64; 3],
    /// The player's selection (own units, by view id).
    selected: std::collections::BTreeSet<u32>,
}

impl Sim {
    /// The world; landed ground changes go to `ground`, if given.
    pub fn new(o: &SimOptions, ground: Option<Arc<GroundFeed>>) -> Result<Self> {
        let wasm = match &o.wasm {
            Some(w) => w.to_vec(),
            None => ae_sim::build::game(&o.game).with_context(|| format!("building games/{}", o.game))?,
        };
        let nav = NavPlugin { ground, ..o.plugin() };
        let spatial = SpatialPlugin::square(nav.extent(), 64.0);
        let cfg = RuntimeConfig { max_workers: o.workers, ..RuntimeConfig::default() };
        let mut plugins: Vec<Arc<dyn ae_sim::host::HostPlugin>> = vec![Arc::new(nav), Arc::new(spatial)];
        if !o.mounts.is_empty() {
            let mut vfs = ae_vfs::Vfs::new();
            for m in &o.mounts {
                vfs.mount(m)?;
            }
            let schema = ae_sim::schema::Schema::from_module(&wasm)?;
            let catalog = ae_catalog::Catalog::load(&vfs, "defs", &schema).context("the content's catalogue")?;
            plugins.push(Arc::new(ae_services::catalog::CatalogPlugin::new(Arc::new(catalog))));
        }
        let rt = Runtime::with_hosts(&wasm, cfg, plugins)?;
        let world = rt.world_for(o.param(), o.workers)?;
        // The raw capture (no observer) draws units with a nav agent.
        let spec = CaptureSpec::new(&rt.schema, &FIELDS)?;
        // Skirmish units draw by their `Look`, crossing units by class.
        let (extra, class_field) = if o.skirmish() { ("Look", ("Look", "class")) } else { ("NavAgent", ("NavAgent", "profile")) };
        let view = match o.observer {
            Some(ally) => {
                anyhow::ensure!(ally < o.allies(), "observer {ally} of {} allyteams", o.allies());
                let pipeline = ViewPipeline::new(&rt.schema, &[extra], o.allies() as usize, o.plugin().extent())?;
                let field = |c: &str, f: &str| pipeline.field(c, f).with_context(|| format!("`{c}.{f}` is not viewed"));
                let fields = [field("Body", "x")?, field("Body", "y")?, field("Body", "vx")?, field("Body", "vy")?, field("Body", "radius")?, field(class_field.0, class_field.1)?];
                let grid = pipeline.coverage.grid;
                Some(ViewState { pipeline, observer: [Observer::new(ally, 0x0B5E_4E00 ^ ally as u64)], explored: Bits::new(grid.width, grid.height), vision: None, table: ViewTable::default(), fields, ms: [0.0; 3], selected: Default::default() })
            }
            None => None,
        };
        let classes = o.own_classes().len() as u32;
        let (bots, unit_command) = if o.skirmish() {
            let bots = ae_bot::SkirmishBot::pair(&rt)?.into_iter().filter(|b| o.observer != Some(b.ally)).collect();
            (bots, rt.schema.components.iter().find(|c| c.name == "UnitCommand").map(|c| c.id))
        } else {
            (Vec::new(), None)
        };
        Ok(Self { world, rt, workers: o.workers, spec, capture: Capture::default(), streamer: Streamer::default(), view, classes, bots, unit_command, effects: None, last_units: Vec::new(), last_los: None })
    }

    /// Publish the observer's vision here (with an observer).
    pub fn with_vision(mut self, feed: Arc<VisionFeed>) -> Self {
        if let Some(v) = self.view.as_mut() {
            v.vision = Some(feed);
        }
        self
    }

    /// Set off an explosion in `effects` for every unit that dies where
    /// the observer saw at the tick before (every unit without an observer).
    pub fn with_effects(mut self, effects: Arc<Effects>) -> Self {
        self.effects = Some(effects);
        self
    }

    /// Units gone from the capture since the last tick (despawned: dead)
    /// that stood in the observer's line of sight of that tick explode.
    /// Only what the observer saw: a unit dying in the fog leaves nothing.
    fn emit_deaths(&mut self) {
        let Some(effects) = &self.effects else { return };
        let c = &self.capture;
        let mut now: Vec<(u64, f32, f32, f32)> = (0..c.len()).map(|r| (c.entities[r], c.f32(F_X, r), c.f32(F_Y, r), c.f32(F_RADIUS, r))).collect();
        now.sort_unstable_by_key(|u| u.0);
        let grid = self.view.as_ref().map(|v| v.pipeline.coverage.grid);
        let sites = deaths(&self.last_units, &now, |x, y| match (&self.last_los, grid) {
            (Some(los), Some(grid)) => {
                let (cx, cy) = grid.cell(x, y);
                los.get(cx, cy)
            }
            _ => true,
        });
        for &(x, y, scale, seed) in &sites {
            effects.explode(x, y, scale, seed);
        }
        self.last_units = now;
        if let Some(v) = &self.view {
            self.last_los = Some(v.pipeline.coverage.los[v.observer[0].ally as usize].clone());
        }
    }

    /// Rows of each state in the last view: own, seen, blip, ghost.
    pub fn view_counts(&self) -> Option<[usize; 4]> {
        let v = self.view.as_ref()?;
        let mut n = [0; 4];
        for s in &v.table.state {
            n[*s as usize] += 1;
        }
        Some(n)
    }

    pub fn tick(&mut self) -> Result<()> {
        self.tick_with(&[])
    }

    /// One tick with the player's actions since the last: a selection
    /// changes what is selected; a move becomes one `MoveOrder` command per
    /// selected unit still in view (view ids resolved to entities by the
    /// observer, here in trusted code), all in one group.
    pub fn tick_with(&mut self, inputs: &[UiInput]) -> Result<()> {
        let mut bundle = TickBundle::new(self.world.tick);
        if let Some(v) = self.view.as_mut() {
            let ally = v.observer[0].ally;
            for input in inputs {
                match *input {
                    UiInput::Select { x, y, radius } => {
                        let t = &v.table;
                        let [fx, fy] = [v.fields[0], v.fields[1]].map(|f| t.fields.iter().position(|&c| c == f).unwrap());
                        v.selected = (0..t.len())
                            .filter(|&r| t.state[r] == RowState::Own && (f32::from_bits(t.columns[fx][r]) - x).hypot(f32::from_bits(t.columns[fy][r]) - y) <= radius)
                            .map(|r| t.ids[r])
                            .collect();
                        eprintln!("selected {} units", v.selected.len());
                    }
                    UiInput::Move { x, y, attack, append } => {
                        let group = 1u64 << 43 | self.world.tick << 8 | ally as u64;
                        for &id in &v.selected {
                            let Some(e) = v.observer[0].entity_of(id) else { continue };
                            bundle.commands.push(match self.unit_command {
                                Some(stream) => ae_bot::unit_command(ally, stream, e, if attack { ae_bot::ATTACK } else { ae_bot::MOVE }, append, x, y),
                                None => {
                                    let order = MoveOrder::move_to(Entity(e), x, y, group);
                                    let bytes = unsafe { std::slice::from_raw_parts((&order as *const MoveOrder).cast::<u8>(), size_of::<MoveOrder>()).to_vec() };
                                    Command { player: ally, stream: ae_abi::engine::MOVE_ORDER, bytes }
                                }
                            });
                        }
                    }
                }
            }
        }
        for b in &self.bots {
            b.commands(&self.rt, &self.world, &mut bundle.commands)?;
        }
        self.rt.tick_with(&mut self.world, self.workers, &bundle)?;
        Ok(())
    }

    /// Units of movement class `class` with a non-zero velocity in the last
    /// capture.
    pub fn moving(&self, class: u32) -> usize {
        let c = &self.capture;
        (0..c.len()).filter(|&r| c.columns[F_PROFILE][r] == class && (c.f32(F_VX, r) != 0.0 || c.f32(F_VY, r) != 0.0)).count()
    }

    /// Capture the world and build the stream (`tick_seconds` 0: no
    /// interpolation): every unit, or what the observer sees.
    pub fn stream(&mut self, tick_seconds: f32) -> Result<InstanceStream> {
        self.rt.capture(&self.world, &self.spec, &mut self.capture)?;
        if self.view.is_none() {
            self.emit_deaths();
        }
        let Some(v) = self.view.as_mut() else {
            return Ok(self.streamer.stream(&self.capture, Instant::now(), tick_seconds));
        };
        let t0 = Instant::now();
        v.pipeline.capture(&self.rt, &self.world)?;
        let t1 = Instant::now();
        v.table = v.pipeline.observe(&mut v.observer).pop().unwrap();
        let t2 = Instant::now();
        let ally = v.observer[0].ally as usize;
        let coverage = &v.pipeline.coverage;
        v.explored.or(&coverage.los[ally]);
        if let Some(feed) = &v.vision {
            feed.publish(Arc::new(VisionFrame { tick: self.world.tick, los: coverage.los[ally].words.clone(), explored: v.explored.words.clone() }));
        }
        let t = &v.table;
        let [fx, fy, fvx, fvy, fr, fp] = v.fields.map(|f| t.fields.iter().position(|&c| c == f).expect("viewed field"));
        let rows: Vec<UnitRow> = (0..t.len())
            .map(|r| {
                let shown = |c: usize| t.mask[r] & (1 << c) != 0;
                let word = |c: usize| t.columns[c][r];
                let class = match t.state[r] {
                    RowState::Own if v.selected.contains(&t.ids[r]) => 2 * self.classes + 1,
                    RowState::Own => word(fp),
                    _ if shown(fp) => self.classes + word(fp),
                    _ => 2 * self.classes,
                };
                UnitRow {
                    key: t.ids[r] as u64,
                    x: f32::from_bits(word(fx)),
                    y: f32::from_bits(word(fy)),
                    vx: f32::from_bits(word(fvx)),
                    vy: f32::from_bits(word(fvy)),
                    radius: if shown(fr) { word(fr) } else { BLIP_RADIUS.to_bits() },
                    class,
                    state: t.state[r] as u8,
                }
            })
            .collect();
        let stream = self.streamer.stream_rows(self.capture.tick, &rows, Instant::now(), tick_seconds);
        // After the rows: the last tick's line of sight decides, then
        // this one's is kept.
        self.emit_deaths();
        let v = self.view.as_mut().expect("observer");
        v.ms = [(t1 - t0).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3, t2.elapsed().as_secs_f64() * 1e3];
        Ok(stream)
    }

    /// The last view's milliseconds: capture + coverage, build, rows + stream.
    pub fn view_ms(&self) -> Option<[f64; 3]> {
        self.view.as_ref().map(|v| v.ms)
    }
}

/// Explosions (`(x, y, scale, seed)`) for the units of `last` missing from
/// `now` (both sorted by entity: `(entity, x, y, radius)`) that stood where
/// `seen(x, y)`.
fn deaths(last: &[(u64, f32, f32, f32)], now: &[(u64, f32, f32, f32)], seen: impl Fn(f32, f32) -> bool) -> Vec<(f32, f32, f32, u32)> {
    let mut sites = Vec::new();
    for &(id, x, y, radius) in last {
        if now.binary_search_by_key(&id, |u| u.0).is_err() && seen(x, y) {
            sites.push((x, y, (radius / 6.0).clamp(0.6, 3.0), (id ^ (id >> 32)) as u32));
        }
    }
    sites
}

/// Where explosions go: the particles (flash, fire, sparks, smoke) and a
/// scar on the ground, each when the client draws them.
#[derive(Debug, Default)]
pub struct Effects {
    pub particles: Option<Arc<ParticleFeed>>,
    pub decals: Option<Arc<DecalFeed>>,
    /// The atlas layers scars are picked from (by the explosion's seed) and
    /// how each blends; empty: the generated scorch mark.
    pub scars: Vec<(u32, Blend)>,
}

/// The damage an explosion of size `scale` stands for in Recoil's terms,
/// which set a scar's opacity and glow (`Decal::scar`): about 200 for a
/// unit of radius 12 (scale 2), so the bigger deaths glow fully.
pub const DAMAGE_PER_SCALE: f32 = 100.0;

impl Effects {
    /// An explosion of size `scale` on the ground at world `(x, z)`.
    pub fn explode(&self, x: f32, z: f32, scale: f32, seed: u32) {
        if let Some(feed) = &self.particles {
            feed.emit_all(&explosion([x, 1.0, z], true, scale, seed));
        }
        if let Some(feed) = &self.decals {
            let (layer, blend) = match self.scars.len() {
                0 => (layer::SCORCH, Blend::Replace),
                n => self.scars[(seed >> 7) as usize % n],
            };
            feed.place(Decal::scar([x, z], 3.5 * scale, seed, DAMAGE_PER_SCALE * scale, layer, blend));
        }
    }
}

/// One unit for the stream: `key` orders rows and matches them across
/// streams (an entity, or an observer's view id).
#[derive(Clone, Copy, Debug)]
pub struct UnitRow {
    pub key: u64,
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    /// `f32` bits.
    pub radius: u32,
    pub class: u32,
    /// Changing state (a blip seen, a unit lost to radar) snaps instead of
    /// interpolating.
    pub state: u8,
}

/// Turns captures into streams, remembering the last one by key.
#[derive(Default)]
pub struct Streamer {
    /// Previous stream, sorted by key: `(key, x, y, heading x, heading y, state)`.
    previous: Vec<(u64, f32, f32, f32, f32, u8)>,
    order: Vec<u32>,
}

impl Streamer {
    /// The stream of `capture` (rows in entity order). A unit's previous
    /// position is its position in the previous capture (its current one
    /// if it is new); its heading is its velocity, or its last heading
    /// while it stands still.
    pub fn stream(&mut self, capture: &Capture, published: Instant, tick_seconds: f32) -> InstanceStream {
        let n = capture.len();
        self.order.clear();
        self.order.extend(0..n as u32);
        self.order.sort_unstable_by_key(|&r| capture.entities[r as usize]);
        let rows: Vec<UnitRow> = self
            .order
            .iter()
            .map(|&r| {
                let r = r as usize;
                UnitRow {
                    key: capture.entities[r],
                    x: capture.f32(F_X, r),
                    y: capture.f32(F_Y, r),
                    vx: capture.f32(F_VX, r),
                    vy: capture.f32(F_VY, r),
                    radius: capture.columns[F_RADIUS][r],
                    class: capture.columns[F_PROFILE][r],
                    state: 0,
                }
            })
            .collect();
        self.stream_rows(capture.tick, &rows, published, tick_seconds)
    }

    /// The stream of `rows` (ascending by key), matched to the previous
    /// stream by key; a row whose state changed starts where it is.
    pub fn stream_rows(&mut self, tick: u64, rows: &[UnitRow], published: Instant, tick_seconds: f32) -> InstanceStream {
        let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = std::array::from_fn(|_| Vec::with_capacity(rows.len()));
        let mut next = Vec::with_capacity(rows.len());
        let mut p = 0;
        for row in rows {
            while p < self.previous.len() && self.previous[p].0 < row.key {
                p += 1;
            }
            let before = self.previous.get(p).filter(|prev| prev.0 == row.key);
            let moved = before.filter(|b| b.5 == row.state);
            let (x, y, vx, vy) = (row.x, row.y, row.vx, row.vy);
            let (px, py) = moved.map_or((x, y), |b| (b.1, b.2));
            let (hx, hy) = if vx * vx + vy * vy > 1e-6 { (vx, vy) } else { before.map_or((1.0, 0.0), |b| (b.3, b.4)) };
            for (column, word) in columns.iter_mut().zip([x.to_bits(), y.to_bits(), px.to_bits(), py.to_bits(), hx.to_bits(), hy.to_bits(), row.radius, row.class]) {
                column.push(word);
            }
            next.push((row.key, x, y, hx, hy, row.state));
        }
        self.previous = next;
        InstanceStream { tick, published, tick_seconds, columns }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SimStats {
    pub tick: u64,
    pub units: usize,
    /// Mean over the last report interval.
    pub tick_ms: f64,
    pub capture_ms: f64,
    /// Achieved tick rate over the last report interval, as a multiple of
    /// the nav rate (what the machine can actually run).
    pub actual_speed: f32,
    /// Ticks run late (the sim could not keep the tick rate).
    pub late: u64,
    pub ready: bool,
    /// The sim stopped with an error (printed when it happened).
    pub failed: bool,
    /// With an observer: its last view's rows (own, seen, blip, ghost).
    pub view: Option<[usize; 4]>,
}

/// The sim thread.
pub struct SimThread {
    stop: Arc<AtomicBool>,
    stats: Arc<Mutex<SimStats>>,
    handle: Option<JoinHandle<Result<()>>>,
}

impl SimThread {
    /// Build the world and tick it at the nav rate times `speed`,
    /// publishing units to `feed` and landed ground changes to `ground`.
    pub fn start(o: SimOptions, feed: Arc<UnitFeed>, ground: Arc<GroundFeed>, vision: Option<Arc<VisionFeed>>, effects: Option<Arc<Effects>>, ui: UiQueue, speed: SimSpeed) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(SimStats::default()));
        let (stop2, stats2) = (stop.clone(), stats.clone());
        let handle = std::thread::Builder::new()
            .name("sim".into())
            .spawn(move || {
                let result = run(o, feed, ground, vision, effects, ui, speed, &stop2, &stats2);
                if let Err(error) = &result {
                    eprintln!("sim failed: {error:#} (--units 0 runs without the sim)");
                    stats2.lock().unwrap_or_else(|p| p.into_inner()).failed = true;
                }
                result
            })
            .expect("spawn the sim thread");
        Self { stop, stats, handle: Some(handle) }
    }

    pub fn stats(&self) -> SimStats {
        *self.stats.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn stop(mut self) -> Result<SimStats> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.join().map_err(|_| anyhow::anyhow!("the sim thread panicked"))??;
        }
        Ok(self.stats())
    }
}

#[allow(clippy::too_many_arguments)]
fn run(o: SimOptions, feed: Arc<UnitFeed>, ground: Arc<GroundFeed>, vision: Option<Arc<VisionFeed>>, effects: Option<Arc<Effects>>, ui: UiQueue, speed: SimSpeed, stop: &AtomicBool, stats: &Mutex<SimStats>) -> Result<()> {
    let started = Instant::now();
    let mut sim = Sim::new(&o, Some(ground))?;
    if let Some(feed) = vision {
        sim = sim.with_vision(feed);
    }
    if let Some(effects) = effects {
        sim = sim.with_effects(effects);
    }
    eprintln!("sim: {} game, {} workers, world built in {:.2} s", o.game, o.workers, started.elapsed().as_secs_f32());
    let mut rate = speed.current();
    let mut period = Duration::from_secs_f32(TICK_SECONDS / rate);
    let mut next = Instant::now();
    let (mut tick_sum, mut capture_sum, mut count) = (0.0, 0.0, 0u32);
    let mut report_started = Instant::now();
    let mut late = 0;
    while !stop.load(Ordering::Relaxed) {
        let new_rate = speed.current();
        if new_rate != rate {
            // Schedule from now: the old period no longer applies.
            rate = new_rate;
            period = Duration::from_secs_f32(TICK_SECONDS / rate);
            next = Instant::now();
        }
        let t0 = Instant::now();
        let inputs = std::mem::take(&mut *ui.lock().unwrap_or_else(|p| p.into_inner()));
        sim.tick_with(&inputs)?;
        let t1 = Instant::now();
        // Interpolate positions over the tick length the viewer actually sees.
        let stream = sim.stream(TICK_SECONDS / rate)?;
        let (tick, units) = (stream.tick, stream.len());
        let view = sim.view_counts();
        feed.publish(Arc::new(stream));
        tick_sum += (t1 - t0).as_secs_f64() * 1e3;
        capture_sum += t1.elapsed().as_secs_f64() * 1e3;
        count += 1;
        let elapsed = report_started.elapsed().as_secs_f32();
        if count == 30 || (count > 0 && elapsed >= 0.5) {
            let actual_speed = if elapsed > 0.0 { count as f32 * TICK_SECONDS / elapsed } else { 0.0 };
            let mut s = stats.lock().unwrap_or_else(|p| p.into_inner());
            *s = SimStats { tick, units, tick_ms: tick_sum / count as f64, capture_ms: capture_sum / count as f64, actual_speed, late, ready: true, failed: false, view };
            (tick_sum, capture_sum, count) = (0.0, 0.0, 0);
            report_started = Instant::now();
        }
        if !period.is_zero() {
            next += period;
            let now = Instant::now();
            if now < next {
                std::thread::sleep(next - now);
            } else {
                late += 1;
                if now - next > period * 8 {
                    // Too far behind to catch up: drop the backlog.
                    next = now;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_render::units::ATTRIBUTES;

    fn capture(tick: u64, rows: &[(u64, f32, f32, f32, f32)]) -> Capture {
        let mut c = Capture { tick, entities: Vec::new(), columns: vec![Vec::new(); FIELDS.len()] };
        for &(e, x, y, vx, vy) in rows {
            c.entities.push(e);
            for (f, word) in [x.to_bits(), y.to_bits(), vx.to_bits(), vy.to_bits(), 2.5f32.to_bits(), (e % 3) as u32].into_iter().enumerate() {
                c.columns[f].push(word);
            }
        }
        c
    }

    fn column(s: &InstanceStream, name: &str) -> Vec<f32> {
        let a = ATTRIBUTES.iter().position(|&n| n == name).unwrap();
        s.columns[a].iter().map(|&w| f32::from_bits(w)).collect()
    }

    /// Rows come out in entity order; previous positions and headings
    /// follow entities across reordered captures; new units start where
    /// they are; standing units keep their heading.
    #[test]
    fn streams_match_rows_by_entity() {
        let mut s = Streamer::default();
        let now = Instant::now();
        let first = s.stream(&capture(1, &[(9, 90.0, 0.0, 0.0, 1.0), (4, 40.0, 0.0, 1.0, 0.0)]), now, 0.1);
        assert_eq!(column(&first, "x"), [40.0, 90.0]);
        assert_eq!(column(&first, "prev_x"), [40.0, 90.0], "first sight: no motion");
        // Table order changed, 4 stopped, 7 joined, 9 moved.
        let second = s.stream(&capture(2, &[(7, 70.0, 5.0, 0.0, 0.0), (9, 91.0, 0.5, 0.0, 1.0), (4, 40.0, 0.0, 0.0, 0.0)]), now, 0.1);
        assert_eq!(column(&second, "x"), [40.0, 70.0, 91.0]);
        assert_eq!(column(&second, "prev_x"), [40.0, 70.0, 90.0]);
        assert_eq!(column(&second, "prev_y"), [0.0, 5.0, 0.0]);
        assert_eq!(column(&second, "heading_x"), [1.0, 1.0, 0.0], "4 keeps its heading, 7 gets the default");
        assert_eq!(column(&second, "heading_y"), [0.0, 0.0, 1.0]);
        assert_eq!(second.columns[7], [1, 1, 0], "class = profile");
        assert_eq!(second.tick, 2);
        // 9 left.
        let third = s.stream(&capture(3, &[(4, 41.0, 0.0, 1.0, 0.0), (7, 70.0, 5.0, 0.0, 0.0)]), now, 0.1);
        assert_eq!(column(&third, "prev_x"), [40.0, 70.0]);
    }

    /// A unit gone since the last capture explodes where it stood, if the
    /// observer saw there; units that stay or join do not.
    #[test]
    fn units_that_vanish_where_seen_explode() {
        let last = [(1, 10.0, 10.0, 6.0), (2, 20.0, 20.0, 6.0), (3, -30.0, 30.0, 6.0), (5, 50.0, 50.0, 12.0)];
        let now = [(1, 11.0, 10.0, 6.0), (4, 40.0, 40.0, 6.0), (5, 51.0, 50.0, 12.0)];
        let sites = deaths(&last, &now, |x, _| x >= 0.0);
        assert_eq!(sites.len(), 1, "only 2 died where seen (3 died in the fog): {sites:?}");
        assert_eq!((sites[0].0, sites[0].1, sites[0].2), (20.0, 20.0, 1.0));
        // One explosion: flash, fire, sparks, smoke, and a scorch mark.
        let effects = Effects { particles: Some(Arc::default()), decals: Some(Arc::default()), scars: Vec::new() };
        effects.explode(20.0, 20.0, 1.0, sites[0].3);
        let bursts = effects.particles.as_ref().unwrap().take_all();
        assert_eq!(bursts.len(), 4, "flash, fire, sparks, smoke: {bursts:?}");
        assert!(bursts.iter().all(|b| b.pos[0] == 20.0 && b.pos[2] == 20.0 && b.ground));
        let decals = effects.decals.as_ref().unwrap().take();
        assert_eq!(decals.len(), 1);
        assert_eq!(decals[0].pos, [20.0, 20.0]);
        assert!(decals[0].glow > 0.5 && decals[0].glow_life > 0.0 && decals[0].layer == layer::SCORCH, "{:?}", decals[0]);
        assert!(deaths(&last, &last, |_, _| true).is_empty(), "nobody died");
    }

    #[test]
    fn speed_doubles_without_a_ceiling() {
        let speed = SimSpeed::default();
        assert_eq!(speed.current(), 1.0);
        assert_eq!(speed.faster(), 2.0);
        assert_eq!(speed.faster(), 4.0);
        assert_eq!(speed.slower(), 2.0);
        assert_eq!(speed.faster(), 4.0);
        assert_eq!(speed.faster(), 8.0);
        assert_eq!(speed.faster(), 16.0);
        assert_eq!(speed.faster(), 32.0);
        assert_eq!(speed.faster(), 64.0);
        assert_eq!(speed.label(), "64x");
        for _ in 0..20 {
            speed.faster();
        }
        assert_eq!(speed.current(), 64.0 * 2f32.powi(20), "no top step");
        for _ in 0..60 {
            speed.slower();
        }
        assert_eq!(speed.current(), MIN_SPEED, "clamped at the bottom");
        assert_eq!(speed.slower(), MIN_SPEED);
    }
}
