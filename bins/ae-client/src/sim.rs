//! The sim beside the renderer: the crossing game (`games/crossing`) with
//! the nav and spatial host systems on the same map the terrain shows,
//! ticked at the nav service's rate on its own thread. After every tick the
//! units' `Body` and `NavAgent` fields are captured
//! ([`ae_sim::runtime::Runtime::capture`]) and turned into the renderer's
//! [`InstanceStream`]: rows matched by entity against the previous capture
//! for the interpolation start, and drawn by their `Orientation` (read
//! apart from the capture, by entity) or their velocity's turn. Stand-in
//! for M4's copy layer, which will publish filtered view tables instead.
//!
//! The animation service (`ae_services::anim`) runs with the `--model`
//! models (and the content's); units with an `Anim` drawn with its model
//! are posed from it (`poser.rs`; the raw capture only, not yet an
//! observer's view).

use crate::poser::Poser;
use ae_model::anim::Rig;
use ae_nav::math::TICK_SECONDS;
use ae_render::decals::{Blend, Decal, DecalFeed, layer};
use ae_render::particles::{Burst, EFFECT_NAMES, ParticleFeed, effect, explosion};
use ae_render::terrain::app::{TerrainSource, VisionFeed, VisionFrame};
use ae_render::terrain::renderer::VisionGrid;
use ae_render::units::{ATTR_POSE, ATTRIBUTE_COUNT, AnimWrites, InstanceStream, NO_POSE, UnitClass, UnitFeed};
use ae_render::water::{WaterFeed, WaterImpact};
use ae_sdk::Entity;
use ae_sdk::aim::Orientation;
use ae_sdk::anim::{Anim, model_id};
use ae_sdk::fx::Effect;
use ae_sdk::nav::MoveOrder;
use ae_sensors::{Bits, SensorGrid};
use ae_services::anim::AnimPlugin;
use ae_services::fx::{FxFeed, FxPlugin};
use ae_services::nav::{GroundFeed, Map, NavPlugin};
use ae_services::spatial::SpatialPlugin;
use ae_services::view::{SENSOR_CELL, ViewPipeline, WaterColumn};
use ae_sim::capture::{Capture, CaptureSpec};
use ae_sim::input::{Command, TickBundle};
use ae_sim::runtime::{Runtime, RuntimeConfig, World};
use ae_view::{Observer, RowState, ViewTable};
use anyhow::{Context, Result};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// What the renderer needs of a unit, by field (`NavAgent.profile` but in
/// the scripted game, whose units are all class 0).
const FIELDS: [(&str, &str); 6] = [
    ("Body", "x"),
    ("Body", "y"),
    ("Body", "vx"),
    ("Body", "vy"),
    ("Body", "radius"),
    ("NavAgent", "profile"),
];
const F_X: usize = 0;
const F_Y: usize = 1;
const F_VX: usize = 2;
const F_VY: usize = 3;
const F_RADIUS: usize = 4;
const F_PROFILE: usize = 5;
const F_WATER: usize = 6;

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
    /// `games/<game>`: `crossing`, `skirmish` or `scripted`.
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
    /// `--model`: classes drawn with a model (and the animation service's
    /// models, by name).
    pub models: Vec<crate::models::ClassModel>,
}

/// A player's action from the window, in world coordinates.
#[derive(Clone, Copy, Debug)]
pub enum UiInput {
    /// Cycle through the available allyteam observers.
    NextObserver,
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
/// and the sim thread (which schedules ticks by it), and whether it is
/// paused.
#[derive(Clone, Debug)]
pub struct SimSpeed {
    speed: Arc<AtomicU32>,
    paused: Arc<AtomicBool>,
}

impl Default for SimSpeed {
    fn default() -> Self {
        Self {
            speed: Arc::new(AtomicU32::new(1.0f32.to_bits())),
            paused: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl SimSpeed {
    pub fn current(&self) -> f32 {
        f32::from_bits(self.speed.load(Ordering::Relaxed))
    }

    fn set(&self, speed: f32) -> f32 {
        self.speed.store(speed.to_bits(), Ordering::Relaxed);
        speed
    }

    pub fn paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    /// Pause if running, resume if paused; returns the new state.
    pub fn toggle_pause(&self) -> bool {
        let paused = !self.paused();
        self.paused.store(paused, Ordering::Relaxed);
        paused
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
const SKIRMISH_LOOKS: [[f32; 3]; 4] = [
    [0.55, 0.62, 0.75],
    [0.95, 0.75, 0.10],
    [0.20, 0.75, 0.30],
    [0.75, 0.20, 0.85],
];

/// Radius drawn for a blip (its real one is line-of-sight only).
const BLIP_RADIUS: f32 = 5.0;
const ENEMY_TINT: [f32; 3] = [0.95, 0.15, 0.1];

impl SimOptions {
    /// The sim's animation models by name: the `--model` ones, then the
    /// content's (`models/*.aemodel`) not named so. The renderer poses
    /// with the same rigs (`UnitsSetup::rigs`), so it samples the sim's
    /// clips and masks; both sides make them here.
    pub fn rigs(&self) -> Result<Vec<(String, Rig)>> {
        let mut rigs: Vec<(String, Rig)> = self
            .models
            .iter()
            .map(|m| {
                let mut rig = Rig::new(&m.model);
                if let Some(json) = &m.rig {
                    rig.apply_rig_file(json)
                        .with_context(|| format!("model {}'s rig file", m.name))?;
                }
                Ok((m.name.clone(), rig))
            })
            .collect::<Result<_>>()?;
        if !self.mounts.is_empty() {
            let mut vfs = ae_vfs::Vfs::new();
            for m in &self.mounts {
                vfs.mount(m)?;
            }
            for f in vfs.list("models", "aemodel")? {
                let model = ae_model::Model::from_bytes(&vfs.read(&f)?).with_context(|| format!("model {f}"))?;
                let name = f.trim_start_matches("models/").trim_end_matches(".aemodel").to_string();
                if !rigs.iter().any(|(n, _)| *n == name) {
                    let mut rig = Rig::new(&model);
                    let rig_file = format!("models/{name}.rig.json");
                    if vfs.exists(&rig_file) {
                        rig.apply_rig_file(&vfs.read(&rig_file)?)
                            .with_context(|| format!("model {name}'s rig file"))?;
                    }
                    rigs.push((name, rig));
                }
            }
        }
        Ok(rigs)
    }

    fn plugin(&self) -> NavPlugin {
        let base = match &self.loaded_map {
            Some((terrain, starts)) => NavPlugin::loaded(terrain.clone(), starts.clone()),
            None => NavPlugin::new(self.map, self.map_cells),
        };
        NavPlugin {
            teams: if self.skirmish() { 0 } else { self.teams },
            ..base
        }
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
        VisionGrid {
            width: g.width,
            height: g.height,
            cell_size: g.cell_size,
        }
    }

    /// The game parameter: `--param`, or the game's default (crossing:
    /// unit count and quake tick; skirmish: the map side and 8 tanks each).
    pub fn param(&self) -> u64 {
        self.param.unwrap_or(if self.skirmish() {
            (self.map_cells as u64 * 8) << 24 | 8 << 40
        } else {
            self.units as u64 | (self.quake as u64) << 24
        })
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
        let enemy = own.iter().map(|c| UnitClass {
            color: tint(c.color),
            gamut: ae_render::colorspace::Gamut::Bt709,
            floats: c.floats,
        });
        own.iter()
            .copied()
            .chain(enemy)
            .chain([
                UnitClass {
                    gamut: ae_render::colorspace::Gamut::Bt709,
                    color: [0.6, 0.6, 0.6],
                    floats: false,
                },
                UnitClass {
                    gamut: ae_render::colorspace::Gamut::Bt709,
                    color: [1.0, 1.0, 1.0],
                    floats: false,
                },
            ])
            .collect()
    }

    fn own_classes(&self) -> Vec<UnitClass> {
        const COLORS: [[f32; 3]; 5] = [
            [0.95, 0.75, 0.10],
            [0.80, 0.12, 0.08],
            [0.92, 0.92, 0.96],
            [0.10, 0.80, 0.90],
            [0.75, 0.20, 0.85],
        ];
        if self.skirmish() {
            return SKIRMISH_LOOKS
                .iter()
                .map(|&color| UnitClass {
                    gamut: ae_render::colorspace::Gamut::Bt709,
                    color,
                    floats: false,
                })
                .collect();
        }
        let mut classes: Vec<_> = self
            .plugin()
            .map_profiles()
            .iter()
            .enumerate()
            .map(|(i, p)| UnitClass {
                color: COLORS[i % COLORS.len()],
                gamut: ae_render::colorspace::Gamut::Bt709,
                floats: p.medium.on_surface(),
            })
            .collect();
        if self.game == "crossing" && self.map == Map::Coast {
            classes.push(UnitClass {
                color: [0.12, 0.48, 0.52],
                gamut: ae_render::colorspace::Gamut::Bt709,
                floats: true,
            });
        }
        classes
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
    dive: Option<WaterColumn>,
    streamer: Streamer,
    view: Option<ViewState>,
    classes: u32,
    class_floats: Vec<bool>,
    water_motion_tick: Option<u64>,
    /// Skirmish: the bots of the allyteams no player plays, and the game's
    /// unit command stream.
    bots: Vec<ae_bot::SkirmishBot>,
    unit_command: Option<u64>,
    /// Explosions for units that die where the observer sees.
    effects: Option<Arc<Effects>>,
    /// The last capture's units, sorted by entity.
    last_units: Vec<DeathRow>,
    /// The module declares `Orientation` (its units may be drawn by it).
    has_orientation: bool,
    /// The module declares `Dying` (a killed unit's script plays its death).
    has_dying: bool,
    /// Entities with `Dying` in the last capture, ascending.
    last_dying: Vec<u64>,
    /// The effects the sim's `fx` system published since the last stream.
    fx: Option<Arc<FxFeed>>,
    /// Effect names already warned about (not in the effects catalogue).
    warned_fx: Vec<u64>,
    /// The last `emit_fx`'s effects (tests only).
    #[cfg(test)]
    last_fx: Vec<Effect>,
    /// The units' channels after the last tick, by entity (with a poser).
    /// Milliseconds of the last stream's posing (reading the channels
    /// included).
    pose_ms: f64,
    read_ms: f64,
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
        let cfg = RuntimeConfig {
            max_workers: o.workers,
            ..RuntimeConfig::default()
        };
        // The effects sink, with a feed: the refactor's flashes and the
        // death smoke become particles where the observer sees them.
        let fx = Arc::new(FxFeed::default());
        let mut plugins: Vec<Arc<dyn ae_sim::host::HostPlugin>> = vec![
            Arc::new(nav),
            Arc::new(spatial),
            Arc::new(FxPlugin::new(Some(fx.clone()))),
        ];
        if !o.mounts.is_empty() {
            let mut vfs = ae_vfs::Vfs::new();
            for m in &o.mounts {
                vfs.mount(m)?;
            }
            let schema = ae_sim::schema::Schema::from_module(&wasm)?;
            let catalog = ae_catalog::Catalog::load(&vfs, "defs", &schema).context("the content's catalogue")?;
            plugins.push(Arc::new(ae_services::catalog::CatalogPlugin::new(Arc::new(catalog))));
        }
        let anim = AnimPlugin::new(o.rigs()?, TICK_SECONDS);
        plugins.push(Arc::new(anim.clone()));
        let rt = Runtime::with_hosts(&wasm, cfg, plugins)?;
        let world = rt.world_for(o.param(), o.workers)?;
        // The raw capture (no observer) draws units with a body; their
        // class is their nav agent's profile when the game runs a nav
        // service. Its units carry a `NavAgent`; `scripted` and `duel` do
        // not schedule nav and their units do not (though the SDK declares
        // `NavAgent` in their schema, so the schema alone cannot say).
        let has_nav = rt
            .schema
            .order
            .iter()
            .any(|slot| matches!(slot, ae_sim::schema::Slot::Host(name) if name == "nav" || name == "nav_spawn"));
        let spec = CaptureSpec::new(&rt.schema, &FIELDS[..if has_nav { FIELDS.len() } else { F_PROFILE }])?;
        // Units drawn with a model whose `Anim` names it are posed.
        // (the layout checked: the service's own check runs only when the
        // game schedules it).
        let has_anim = ae_sim::host::expect_component(&rt.schema, <Anim as ae_sdk::Component>::SCHEMA).is_ok();
        let has_orientation =
            ae_sim::host::expect_component(&rt.schema, <Orientation as ae_sdk::Component>::SCHEMA).is_ok();
        let has_dying =
            ae_sim::host::expect_component(&rt.schema, <ae_sdk::life::Dying as ae_sdk::Component>::SCHEMA).is_ok();
        let poser = (has_anim && !o.models.is_empty()).then(|| {
            let names =
                crate::models::names_per_class(&o.models, o.classes().len(), o.observer.map(|_| o.classes().len() - 2));
            Poser::new(
                names
                    .iter()
                    .map(|n| n.as_deref().map(model_id).filter(|&id| anim.rig(id).is_some()))
                    .collect(),
            )
        });
        // Skirmish units draw by their `Look`, crossing units by class.
        let (extra, class_field) = if o.skirmish() {
            ("Look", ("Look", "class"))
        } else {
            ("NavAgent", ("NavAgent", "profile"))
        };
        let view = match o.observer {
            Some(ally) => {
                anyhow::ensure!(ally < o.allies(), "observer {ally} of {} allyteams", o.allies());
                let pipeline = ViewPipeline::new(&rt.schema, &[extra], o.allies() as usize, o.plugin().extent())?;
                let field = |c: &str, f: &str| pipeline.field(c, f).with_context(|| format!("`{c}.{f}` is not viewed"));
                let fields = [
                    field("Body", "x")?,
                    field("Body", "y")?,
                    field("Body", "vx")?,
                    field("Body", "vy")?,
                    field("Body", "radius")?,
                    field(class_field.0, class_field.1)?,
                ];
                let grid = pipeline.coverage.grid;
                Some(ViewState {
                    pipeline,
                    observer: [Observer::new(ally, 0x0B5E_4E00 ^ ally as u64)],
                    explored: Bits::new(grid.width, grid.height),
                    vision: None,
                    table: ViewTable::default(),
                    fields,
                    ms: [0.0; 3],
                    selected: Default::default(),
                })
            }
            None => None,
        };
        let classes = o.own_classes().len() as u32;
        let dive = WaterColumn::new(&rt.schema, "Dive", "depth")?;
        let (bots, unit_command) = if o.skirmish() {
            let bots = ae_bot::SkirmishBot::pair(&rt)?
                .into_iter()
                .filter(|b| o.observer != Some(b.ally))
                .collect();
            (
                bots,
                rt.schema
                    .components
                    .iter()
                    .find(|c| c.name == "UnitCommand")
                    .map(|c| c.id),
            )
        } else {
            (Vec::new(), None)
        };
        Ok(Self {
            world,
            rt,
            workers: o.workers,
            spec,
            capture: Capture::default(),
            dive,
            streamer: Streamer {
                poser,
                ..Streamer::default()
            },
            view,
            classes,
            class_floats: o.classes().iter().map(|class| class.floats).collect(),
            water_motion_tick: None,
            bots,
            unit_command,
            effects: None,
            last_units: Vec::new(),
            has_orientation,
            has_dying,
            last_dying: Vec::new(),
            fx: Some(fx),
            warned_fx: Vec::new(),
            #[cfg(test)]
            last_fx: Vec::new(),
            pose_ms: 0.0,
            read_ms: 0.0,
        })
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
        if let Some(view) = &self.view {
            effects.set_coverage(view);
        }
        self.effects = Some(effects);
        self
    }

    /// Units gone from the capture since the last tick (despawned: dead)
    /// that stood in the observer's line of sight of that tick explode.
    /// Only what the observer saw: a unit dying in the fog leaves nothing.
    /// A unit that died by script (it had `Dying` in the last capture) is
    /// skipped: its effects are its death script's.
    fn emit_effects(&mut self, dying: &[u64]) {
        let Some(effects) = &self.effects else {
            self.emit_fx();
            return;
        };
        let c = &self.capture;
        let mut now: Vec<DeathRow> = (0..c.len())
            .map(|r| {
                (
                    c.entities[r],
                    c.f32(F_X, r),
                    c.f32(F_Y, r),
                    c.f32(F_RADIUS, r),
                    water_depth(c, r),
                )
            })
            .collect();
        now.sort_unstable_by_key(|u| u.0);
        let sites = deaths(&self.last_units, &now, &self.last_dying, |x, y, sub| {
            effects.eligible(x, y, sub)
        });
        for &(x, y, depth, scale, seed) in &sites {
            effects.explode_at_depth(x, y, depth, scale, seed);
        }
        if let Some(v) = &self.view {
            effects.set_coverage(v);
        }
        self.emit_fx();
        self.last_units = now;
        self.last_dying = dying.to_vec();
    }

    /// The observer's line of sight in the last view, and its grid.
    #[cfg(test)]
    pub fn sight(&self) -> Option<(Bits, SensorGrid)> {
        let v = self.view.as_ref()?;
        Some((
            v.pipeline.coverage.los[v.observer[0].ally as usize].clone(),
            v.pipeline.coverage.grid,
        ))
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
        if inputs.iter().any(|input| matches!(input, UiInput::NextObserver))
            && let Some(view) = &self.view
        {
            self.switch_observer((view.observer[0].ally + 1) % view.pipeline.coverage.allies() as u32)?;
        }
        let mut bundle = TickBundle::new(self.world.tick);
        if let Some(v) = self.view.as_mut() {
            let ally = v.observer[0].ally;
            for input in inputs {
                match *input {
                    UiInput::NextObserver => {}
                    UiInput::Select { x, y, radius } => {
                        let t = &v.table;
                        let [fx, fy] =
                            [v.fields[0], v.fields[1]].map(|f| t.fields.iter().position(|&c| c == f).unwrap());
                        v.selected = (0..t.len())
                            .filter(|&r| {
                                t.state[r] == RowState::Own
                                    && (f32::from_bits(t.columns[fx][r]) - x)
                                        .hypot(f32::from_bits(t.columns[fy][r]) - y)
                                        <= radius
                            })
                            .map(|r| t.ids[r])
                            .collect();
                        eprintln!("selected {} units", v.selected.len());
                    }
                    UiInput::Move { x, y, attack, append } => {
                        let group = 1u64 << 43 | self.world.tick << 8 | ally as u64;
                        for &id in &v.selected {
                            let Some(e) = v.observer[0].entity_of(id) else { continue };
                            bundle.commands.push(match self.unit_command {
                                Some(stream) => ae_bot::unit_command(
                                    ally,
                                    stream,
                                    e,
                                    if attack { ae_bot::ATTACK } else { ae_bot::MOVE },
                                    append,
                                    x,
                                    y,
                                ),
                                None => {
                                    let order = MoveOrder::move_to(Entity(e), x, y, group);
                                    let bytes = unsafe {
                                        std::slice::from_raw_parts(
                                            (&order as *const MoveOrder).cast::<u8>(),
                                            size_of::<MoveOrder>(),
                                        )
                                        .to_vec()
                                    };
                                    Command {
                                        player: ally,
                                        stream: ae_abi::engine::MOVE_ORDER,
                                        bytes,
                                    }
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

    pub fn switch_observer(&mut self, ally: u32) -> Result<()> {
        let Some(view) = &mut self.view else { return Ok(()) };
        anyhow::ensure!(
            (ally as usize) < view.pipeline.coverage.allies(),
            "unknown observer {ally}"
        );
        if view.observer[0].ally == ally {
            return Ok(());
        }
        view.observer[0] = Observer::new(ally, 0x0B5E_4E00 ^ ally as u64);
        view.explored.clear();
        view.table = ViewTable::default();
        view.selected.clear();
        self.streamer.reset_view();
        self.last_units.clear();
        self.last_dying.clear();
        self.water_motion_tick = None;
        if let Some(fx) = &self.fx {
            fx.take();
        }
        if let Some(effects) = &self.effects {
            effects.clear(view);
        }
        Ok(())
    }

    /// Units of movement class `class` with a non-zero velocity in the last
    /// capture.
    pub fn moving(&self, class: u32) -> usize {
        let c = &self.capture;
        (0..c.len())
            .filter(|&r| profile(c, r) == class && (c.f32(F_VX, r) != 0.0 || c.f32(F_VY, r) != 0.0))
            .count()
    }

    /// Capture the world and build the stream (`tick_seconds` 0: no
    /// interpolation): every unit, or what the observer sees.
    pub fn stream(&mut self, tick_seconds: f32) -> Result<InstanceStream> {
        self.rt.capture(&self.world, &self.spec, &mut self.capture)?;
        if let Some(dive) = &mut self.dive {
            dive.capture(&self.rt, &self.world)?;
            self.capture.columns.resize_with(F_WATER, Vec::new);
            if self.capture.columns[F_PROFILE].is_empty() {
                self.capture.columns[F_PROFILE].resize(self.capture.entities.len(), 0);
            }
            self.capture.columns.push(
                self.capture
                    .entities
                    .iter()
                    .map(|&e| dive.get(e).unwrap_or_default())
                    .collect(),
            );
        }
        let mut dying = Vec::new();
        if self.has_dying {
            for (ids, _) in self.rt.chunks(&self.world, "Dying")? {
                dying.extend_from_slice(ids);
            }
            dying.sort_unstable();
        }
        if self.view.is_none() {
            self.emit_effects(&dying);
        }
        let Some(v) = self.view.as_mut() else {
            let t0 = Instant::now();
            // The `Anim` rows borrowed from the world, by entity.
            let mut anims: Vec<(u64, &Anim)> = Vec::new();
            if self.streamer.poser.is_some() {
                for (ids, bytes) in self.rt.chunks(&self.world, "Anim")? {
                    anyhow::ensure!(
                        bytes.len() == ids.len() * size_of::<Anim>()
                            && bytes.as_ptr().align_offset(align_of::<Anim>()) == 0,
                        "the `Anim` column is not `Anim`s"
                    );
                    // SAFETY: plain data, aligned, a row per id.
                    let rows = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<Anim>(), ids.len()) };
                    anims.extend(ids.iter().copied().zip(rows));
                }
                // Tables hold rows in entity order while nothing has moved
                // between them.
                if !anims.is_sorted_by_key(|a| a.0) {
                    anims.sort_unstable_by_key(|a| a.0);
                }
            }
            // The `Orientation` rows borrowed from the world, by entity.
            let mut orientations: Vec<(u64, &Orientation)> = Vec::new();
            if self.has_orientation {
                for (ids, bytes) in self.rt.chunks(&self.world, "Orientation")? {
                    anyhow::ensure!(
                        bytes.len() == ids.len() * size_of::<Orientation>()
                            && bytes.as_ptr().align_offset(align_of::<Orientation>()) == 0,
                        "the `Orientation` column is not `Orientation`s"
                    );
                    // SAFETY: plain data, aligned, a row per id.
                    let rows = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<Orientation>(), ids.len()) };
                    orientations.extend(ids.iter().copied().zip(rows));
                }
                if !orientations.is_sorted_by_key(|a| a.0) {
                    orientations.sort_unstable_by_key(|a| a.0);
                }
            }
            self.read_ms = t0.elapsed().as_secs_f64() * 1e3;
            let stream = self
                .streamer
                .stream(&self.capture, &anims, &orientations, Instant::now(), tick_seconds);
            self.pose_ms = t0.elapsed().as_secs_f64() * 1e3;
            self.emit_water_motion(&stream);
            return Ok(stream);
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
            feed.publish(Arc::new(VisionFrame {
                tick: self.world.tick,
                los: coverage.los[ally].words.clone(),
                explored: v.explored.words.clone(),
            }));
        }
        let t = &v.table;
        let [fx, fy, fvx, fvy, fr, fp] = v
            .fields
            .map(|f| t.fields.iter().position(|&c| c == f).expect("viewed field"));
        let depth = v.pipeline.spec.submerged.and_then(|f| t.column(f));
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
                    water_depth: depth.map_or(0.0, |col| f32::from_bits(col[r])),
                    state: t.state[r],
                }
            })
            .collect();
        // The observer path keeps the velocity's turn: `Orientation` is not
        // a viewed field.
        let stream = self
            .streamer
            .stream_rows(self.capture.tick, &rows, &[], &[], Instant::now(), tick_seconds);
        // After the rows: the last tick's line of sight decides, then
        // this one's is kept.
        self.emit_effects(&dying);
        let v = self.view.as_mut().expect("observer");
        v.ms = [
            (t1 - t0).as_secs_f64() * 1e3,
            (t2 - t1).as_secs_f64() * 1e3,
            t2.elapsed().as_secs_f64() * 1e3,
        ];
        self.emit_water_motion(&stream);
        Ok(stream)
    }

    fn emit_water_motion(&mut self, stream: &InstanceStream) {
        let Some(fx) = &self.effects else { return };
        let (true, Some((source, _)), Some(particles)) = (fx.surface_dynamics, &fx.water, &fx.particles) else {
            return;
        };
        let period = (0.16 / stream.sim_seconds.max(0.01)).ceil() as u64;
        if self
            .water_motion_tick
            .is_some_and(|last| stream.tick.saturating_sub(last) < period)
        {
            return;
        }
        self.water_motion_tick = Some(stream.tick);
        use ae_render::units::{
            ATTR_CLASS, ATTR_PREV_X, ATTR_PREV_Y, ATTR_RADIUS, ATTR_VIEW, ATTR_WATER, ATTR_X, ATTR_Y,
        };
        let c = &stream.columns;
        let (x, z, prev_x, prev_z) = (&c[ATTR_X], &c[ATTR_Y], &c[ATTR_PREV_X], &c[ATTR_PREV_Y]);
        let (radius, class, water, view) = (&c[ATTR_RADIUS], &c[ATTR_CLASS], &c[ATTR_WATER], &c[ATTR_VIEW]);
        let f = |column: &[u32], row| f32::from_bits(column[row]);
        let level = fx.water_level();
        let mut bursts = Vec::new();
        const MIN_TRAVEL_SQ: f32 = 0.01;
        for r in 0..stream.len() {
            let state = view[r];
            if state > RowState::Seen as u32 {
                continue;
            }
            let p = glam::Vec2::new(f(x, r), f(z, r));
            let travel = p - glam::Vec2::new(f(prev_x, r), f(prev_z, r));
            let depth = f(water, r);
            if travel.length_squared() <= MIN_TRAVEL_SQ
                || source.height_at(p) >= level
                || (state != RowState::Own as u32 && !fx.eligible(p.x, p.y, depth > 0.0))
            {
                continue;
            }
            let radius = f(radius, r).clamp(1.0, 12.0);
            let bow = travel.normalize() * radius;
            let (at, y, kind) = if depth > 0.0 {
                (
                    p - bow * 0.75,
                    (level - depth).max(source.height_at(p) + 0.5),
                    effect::BUBBLE,
                )
            } else if self.class_floats.get(class[r] as usize).copied().unwrap_or(false) {
                (p + bow * 0.7, 0.65, effect::BOW_SPRAY)
            } else {
                continue;
            };
            bursts.push(Burst {
                pos: [at.x, y, at.y],
                ground: kind == effect::BOW_SPRAY,
                effect: kind,
                count: 3,
                scale: (radius / 3.0).clamp(0.7, 1.8),
                seed: stream.tick as u32 ^ at.x.to_bits().rotate_left(13) ^ at.y.to_bits(),
            });
        }
        particles.emit_all(&bursts);
    }

    /// The effects the sim's `fx` system has published since the last
    /// stream, as particles where the observer sees them. Effects of tick
    /// N start when the stream of tick N is published: the unit is then
    /// drawn between N-1 and N, so a burst leads the drawn bone by up to a
    /// tick of the unit's motion.
    fn emit_fx(&mut self) {
        let Some(feed) = &self.fx else { return };
        let effects = feed.take();
        #[cfg(test)]
        self.last_fx.clone_from(&effects);
        if effects.is_empty() {
            return;
        }
        let Some(fx) = self.effects.clone() else { return };
        let depth = |entity| {
            self.dive.as_ref().map_or(0.0, |dive| {
                dive.get(entity).map_or_else(
                    || {
                        self.last_units
                            .binary_search_by_key(&entity, |row| row.0)
                            .map_or(0.0, |i| self.last_units[i].4)
                    },
                    f32::from_bits,
                )
            })
        };
        let bursts = fx_bursts(&effects, &fx, &mut self.warned_fx, depth);
        if let Some(particles) = &fx.particles {
            particles.emit_all(&bursts);
        }
    }

    /// The effects the last `stream` took from the `fx` feed (tests only).
    #[cfg(test)]
    pub fn last_effects(&self) -> &[Effect] {
        &self.last_fx
    }

    /// The next stream writes the whole animation table (its reader missed
    /// writes: `UnitFeed::wants_whole`).
    pub fn write_whole_anim(&mut self) {
        if let Some(poser) = &mut self.streamer.poser {
            poser.write_whole();
        }
    }

    /// The table writes of a published stream, back from
    /// `UnitFeed::publish_reusing`: the next stream writes into their
    /// memory.
    pub fn recycle(&mut self, writes: AnimWrites) {
        self.streamer.spare = writes;
    }

    /// Milliseconds of the last stream's rows, posing included (without an
    /// observer).
    pub fn pose_ms(&self) -> f64 {
        self.pose_ms
    }

    /// Of [`Self::pose_ms`]: reading the `Anim` column into records, and
    /// the poser's comparing and copying (the rest is the stream's rows).
    pub fn pose_parts_ms(&self) -> (f64, f64) {
        (self.read_ms, self.streamer.poser_ms)
    }

    /// The last view's milliseconds: capture + coverage, build, rows + stream.
    pub fn view_ms(&self) -> Option<[f64; 3]> {
        self.view.as_ref().map(|v| v.ms)
    }
}

/// Row `r`'s nav profile (0 without the field).
fn profile(c: &Capture, r: usize) -> u32 {
    c.columns.get(F_PROFILE).map_or(0, |col| col[r])
}

fn water_depth(c: &Capture, row: usize) -> f32 {
    c.columns.get(F_WATER).map_or(0.0, |col| f32::from_bits(col[row]))
}

/// Entity, world x/z, radius and submerged depth.
type DeathRow = (u64, f32, f32, f32, f32);

/// Explosions for units missing from the new capture. Both inputs are sorted by entity.
fn deaths(
    last: &[DeathRow],
    now: &[DeathRow],
    dying: &[u64],
    eligible: impl Fn(f32, f32, bool) -> bool,
) -> Vec<(f32, f32, f32, f32, u32)> {
    let mut sites = Vec::new();
    for &(id, x, y, radius, depth) in last {
        if now.binary_search_by_key(&id, |u| u.0).is_err()
            && dying.binary_search(&id).is_err()
            && eligible(x, y, depth > 0.0)
        {
            sites.push((x, y, depth, (radius / 6.0).clamp(0.6, 3.0), (id ^ (id >> 32)) as u32));
        }
    }
    sites
}

/// The sim's effects as bursts: one per effect the observer sees (`fx`'s
/// `sees`, at the effect's nav `(x, y)`), named by the effects catalogue
/// (`EFFECT_NAMES`); an unknown name is warned about once and skipped.
/// `dir` is unused: the catalogue's effects are spherical for now.
fn fx_bursts(effects: &[Effect], fx: &Effects, warned: &mut Vec<u64>, depth: impl Fn(u64) -> f32) -> Vec<Burst> {
    let mut bursts = Vec::new();
    for e in effects {
        let Some(effect) = EFFECT_NAMES
            .iter()
            .position(|name| ae_sdk::abi::name_id(name) == e.effect)
        else {
            if !warned.contains(&e.effect) {
                warned.push(e.effect);
                eprintln!(
                    "effect {:#018x} is not one of {}: skipped",
                    e.effect,
                    EFFECT_NAMES.join(", ")
                );
            }
            continue;
        };
        let depth = depth(e.entity.0);
        if !fx.eligible(e.pos[0], e.pos[2], depth > 0.0) {
            continue;
        }
        bursts.push(Burst {
            pos: if depth > 0.0 {
                [e.pos[0], fx.water_level() - depth + e.pos[1], e.pos[2]]
            } else {
                e.pos
            },
            ground: depth <= 0.0,
            effect: effect as u32,
            count: e.count,
            scale: e.scale,
            seed: e.seed,
        });
    }
    bursts
}

/// Where explosions go: the particles (flash, fire, sparks, smoke) and a
/// scar on the ground, each when the client draws them.
#[derive(Default)]
pub struct Effects {
    pub particles: Option<Arc<ParticleFeed>>,
    pub decals: Option<Arc<DecalFeed>>,
    pub water: Option<(Arc<TerrainSource>, Arc<WaterFeed>)>,
    pub surface_dynamics: bool,
    /// The atlas layers scars are picked from (by the explosion's seed) and
    /// how each blends; empty: the generated scorch mark.
    pub scars: Vec<(u32, Blend)>,
    /// The observer's line of sight as of the last tick (the sim sets it);
    /// `None`: everything is seen (no observer, or no sim).
    sight: Mutex<Option<EffectSight>>,
}

#[derive(Debug)]
struct EffectSight {
    los: Bits,
    sonar: Bits,
    grid: SensorGrid,
}

impl EffectSight {
    fn eligible(&self, x: f32, z: f32, submerged: bool) -> bool {
        let (cx, cy) = self.grid.cell(x, z);
        (if submerged { &self.sonar } else { &self.los }).get(cx, cy)
    }
}

/// The damage an explosion of size `scale` stands for in Recoil's terms,
/// which set a scar's opacity and glow (`Decal::scar`; full from 128): 150
/// for the smallest death (scale 0.6), so every unit's death glows fully,
/// as a unit's death explosion does in Recoil.
pub const DAMAGE_PER_SCALE: f32 = 250.0;

impl Effects {
    /// Surface effects require line of sight. Submerged effects require sonar.
    pub fn eligible(&self, x: f32, z: f32, submerged: bool) -> bool {
        self.sight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .is_none_or(|sight| sight.eligible(x, z, submerged))
    }

    fn set_coverage(&self, view: &ViewState) {
        let cov = &view.pipeline.coverage;
        let ally = view.observer[0].ally as usize;
        self.set_sight(cov.los[ally].clone(), cov.sonar[ally].clone(), cov.grid);
    }

    fn set_sight(&self, los: Bits, sonar: Bits, grid: SensorGrid) {
        *self.sight.lock().unwrap_or_else(|p| p.into_inner()) = Some(EffectSight { los, sonar, grid });
    }

    fn clear(&self, view: &ViewState) {
        let cov = &view.pipeline.coverage;
        let empty = Bits::new(cov.grid.width, cov.grid.height);
        let mut sight = self.sight.lock().unwrap_or_else(|p| p.into_inner());
        *sight = Some(EffectSight {
            los: empty.clone(),
            sonar: empty,
            grid: cov.grid,
        });
        if let Some(feed) = &self.water {
            feed.1.clear();
        }
        if let Some(feed) = &self.particles {
            feed.clear();
        }
    }

    fn water_level(&self) -> f32 {
        self.water
            .as_ref()
            .and_then(|(source, _)| source.info.water_level)
            .unwrap_or(0.0)
    }

    /// [`Self::explode`] if the observer sees `(x, z)` (explosions set off
    /// by hand: X, `--explode`); whether it went off.
    pub fn explode_if_seen(&self, x: f32, z: f32, scale: f32, seed: u32) -> bool {
        // Hold the observer snapshot through publication. A simultaneous
        // observer change either clears this impact or filters it first.
        let sight = self.sight.lock().unwrap_or_else(|p| p.into_inner());
        let seen = sight.as_ref().is_none_or(|s| s.eligible(x, z, false));
        if seen {
            self.explode(x, z, scale, seed);
        }
        seen
    }

    /// An explosion of size `scale` at world `(x, z)`: on the ground, or a
    /// shell landing in the water ([`Self::hit_water`]).
    pub fn explode(&self, x: f32, z: f32, scale: f32, seed: u32) {
        if self.over_water(x, z) {
            self.hit_water(x, z, scale, seed);
        } else {
            self.explode_at_depth(x, z, 0.0, scale, seed);
        }
    }

    fn over_water(&self, x: f32, z: f32) -> bool {
        self.water
            .as_ref()
            .is_some_and(|(source, _)| source.height_at(glam::Vec2::new(x, z)) < self.water_level())
    }

    /// A shell or bomb landing in the water: no fireball and no scorch (a
    /// unit dying on the water keeps both, [`Self::explode_at_depth`]). The
    /// prototype's blast instead: a column of spray thrown up and a crown,
    /// and on the surface a crater, ring waves, a churned boil and
    /// whitewater (the renderer's wave particles, from the impact).
    fn hit_water(&self, x: f32, z: f32, scale: f32, seed: u32) {
        if let Some((_, feed)) = &self.water {
            feed.emit(WaterImpact {
                position: [x, z],
                radius: 3.5 * scale,
                seed,
            });
        }
        if let Some(particles) = &self.particles {
            let n = |base: f32| (base * scale.clamp(0.5, 4.0)).round() as u32;
            particles.emit_all(&[
                Burst {
                    pos: [x, 0.5, z],
                    ground: true,
                    effect: effect::WATER_COLUMN,
                    count: n(36.0),
                    scale,
                    seed,
                },
                Burst {
                    pos: [x, 0.7, z],
                    ground: true,
                    effect: effect::SPLASH,
                    count: n(24.0),
                    scale,
                    seed: seed ^ 0x2dc8_ee67,
                },
            ]);
        }
    }

    fn explode_at_depth(&self, x: f32, z: f32, depth: f32, scale: f32, seed: u32) {
        if let Some(feed) = &self.particles {
            let y = if depth > 0.0 { self.water_level() - depth } else { 1.0 };
            feed.emit_all(&explosion([x, y, z], depth <= 0.0, scale, seed));
        }
        if let Some((source, feed)) = &self.water
            && source.height_at(glam::Vec2::new(x, z)) < self.water_level()
        {
            if depth <= 0.0 {
                feed.emit(WaterImpact {
                    position: [x, z],
                    radius: 3.5 * scale,
                    seed,
                });
            }
            if self.surface_dynamics
                && let Some(particles) = &self.particles
            {
                particles.emit(Burst {
                    pos: [x, if depth <= 0.0 { 0.7 } else { self.water_level() - depth }, z],
                    ground: depth <= 0.0,
                    effect: if depth > 0.0 { effect::BUBBLE } else { effect::SPLASH },
                    count: (12.0 * scale.clamp(0.5, 4.0)) as u32,
                    scale,
                    seed: seed ^ 0x2dc8_ee67,
                });
            }
        }
        if depth > 0.0 {
            return;
        }
        if let Some(feed) = &self.decals {
            let (layer, blend) = match self.scars.len() {
                0 => (layer::SCORCH, Blend::Replace),
                n => self.scars[(seed >> 7) as usize % n],
            };
            feed.place(Decal::scar(
                [x, z],
                3.5 * scale,
                seed,
                DAMAGE_PER_SCALE * scale,
                layer,
                blend,
            ));
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
    /// Metres below the water surface. Zero for surfaced and land units.
    pub water_depth: f32,
    /// Changing state (a blip seen, a unit lost to radar) snaps instead of
    /// interpolating.
    pub state: RowState,
}

/// A unit as the last stream showed it.
#[derive(Clone, Copy, Debug)]
struct Shown {
    key: u64,
    x: f32,
    y: f32,
    hx: f32,
    hy: f32,
    state: RowState,
}

/// Turns captures into streams, remembering the last one by key.
#[derive(Default)]
pub struct Streamer {
    /// Previous stream, sorted by key.
    previous: Vec<Shown>,
    order: Vec<u32>,
    /// Packs the channels of units with a model and an `Anim` for the
    /// GPU's poses (none: every unit at rest).
    pub poser: Option<Poser>,
    /// The poser's `pose` attributes (scratch).
    poses: Vec<u32>,
    /// Table writes' memory back from the feed, for the next stream.
    pub spare: AnimWrites,
    /// Milliseconds the poser took in the last stream.
    pub poser_ms: f64,
}

impl Streamer {
    fn reset_view(&mut self) {
        self.previous.clear();
    }

    /// The stream of `capture` (rows in entity order), units posed by their
    /// channels in `anims` and turned by their `orientations` (both by
    /// entity). A unit's previous position is its position in the previous
    /// capture (its current one if it is new); a unit with an
    /// `Orientation` is drawn by it, any other by its velocity's turn or
    /// its last one while it stands still.
    pub fn stream(
        &mut self,
        capture: &Capture,
        anims: &[(u64, &Anim)],
        orientations: &[(u64, &Orientation)],
        published: Instant,
        tick_seconds: f32,
    ) -> InstanceStream {
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
                    class: profile(capture, r),
                    water_depth: water_depth(capture, r),
                    state: RowState::Own,
                }
            })
            .collect();
        self.stream_rows(capture.tick, &rows, anims, orientations, published, tick_seconds)
    }

    /// The stream of `rows` (ascending by key), matched to the previous
    /// stream by key; a row whose state changed starts where it is. Rows
    /// whose key has channels in `anims` (ascending by key) are posed;
    /// rows whose key has an `Orientation` (ascending too) are drawn by it.
    pub fn stream_rows(
        &mut self,
        tick: u64,
        rows: &[UnitRow],
        anims: &[(u64, &Anim)],
        orientations: &[(u64, &Orientation)],
        published: Instant,
        tick_seconds: f32,
    ) -> InstanceStream {
        let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = std::array::from_fn(|_| Vec::with_capacity(rows.len()));
        let mut anim = std::mem::take(&mut self.spare);
        let mut next = Vec::with_capacity(rows.len());
        let (mut p, mut a, mut o) = (0, 0, 0);
        // Rows with channels: (row, then key, class, channels).
        let mut posed = Vec::new();
        let mut units = Vec::new();
        for row in rows {
            while p < self.previous.len() && self.previous[p].key < row.key {
                p += 1;
            }
            let before = self.previous.get(p).filter(|prev| prev.key == row.key);
            let moved = before.filter(|b| b.state == row.state);
            let (x, y, vx, vy) = (row.x, row.y, row.vx, row.vy);
            let (px, py) = moved.map_or((x, y), |b| (b.x, b.y));
            let (hx, hy) = if vx * vx + vy * vy > 1e-6 {
                (vx, vy)
            } else {
                before.map_or((1.0, 0.0), |b| (b.hx, b.hy))
            };
            while a < anims.len() && anims[a].0 < row.key {
                a += 1;
            }
            if let (Some(_), Some((_, anim))) = (&self.poser, anims.get(a).filter(|x| x.0 == row.key)) {
                posed.push(next.len());
                units.push((row.key, row.class, *anim));
            }
            while o < orientations.len() && orientations[o].0 < row.key {
                o += 1;
            }
            let rot = orientations
                .get(o)
                .filter(|x| x.0 == row.key)
                .map_or_else(|| Orientation::along(hx, hy).rot, |x| x.1.rot);
            let words: [u32; ATTRIBUTE_COUNT] = [
                x.to_bits(),
                y.to_bits(),
                px.to_bits(),
                py.to_bits(),
                rot[0].to_bits(),
                rot[1].to_bits(),
                rot[2].to_bits(),
                rot[3].to_bits(),
                row.radius,
                row.class,
                NO_POSE,
                row.water_depth.to_bits(),
                row.state as u32,
            ];
            for (column, word) in columns.iter_mut().zip(words) {
                column.push(word);
            }
            next.push(Shown {
                key: row.key,
                x,
                y,
                hx,
                hy,
                state: row.state,
            });
        }
        if let Some(poser) = &mut self.poser {
            let t0 = Instant::now();
            poser.pose(tick, &units, &mut self.poses, &mut anim);
            self.poser_ms = t0.elapsed().as_secs_f64() * 1e3;
            for (&r, &pose) in posed.iter().zip(&self.poses) {
                columns[ATTR_POSE][r] = pose;
            }
        }
        self.previous = next;
        InstanceStream {
            tick,
            published,
            tick_seconds,
            sim_seconds: TICK_SECONDS,
            columns,
            anim,
        }
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
    pub fn start(
        o: SimOptions,
        feed: Arc<UnitFeed>,
        ground: Arc<GroundFeed>,
        vision: Option<Arc<VisionFeed>>,
        effects: Option<Arc<Effects>>,
        ui: UiQueue,
        speed: SimSpeed,
    ) -> Self {
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
        Self {
            stop,
            stats,
            handle: Some(handle),
        }
    }

    pub fn stats(&self) -> SimStats {
        *self.stats.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn stop(mut self) -> Result<SimStats> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("the sim thread panicked"))??;
        }
        Ok(self.stats())
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    o: SimOptions,
    feed: Arc<UnitFeed>,
    ground: Arc<GroundFeed>,
    vision: Option<Arc<VisionFeed>>,
    effects: Option<Arc<Effects>>,
    ui: UiQueue,
    speed: SimSpeed,
    stop: &AtomicBool,
    stats: &Mutex<SimStats>,
) -> Result<()> {
    let started = Instant::now();
    let mut sim = Sim::new(&o, Some(ground))?;
    if let Some(feed) = vision {
        sim = sim.with_vision(feed);
    }
    if let Some(effects) = effects {
        sim = sim.with_effects(effects);
    }
    eprintln!(
        "sim: {} game, {} workers, world built in {:.2} s",
        o.game,
        o.workers,
        started.elapsed().as_secs_f32()
    );
    let mut rate = speed.current();
    let mut period = Duration::from_secs_f32(TICK_SECONDS / rate);
    let mut next = Instant::now();
    let (mut tick_sum, mut capture_sum, mut count) = (0.0, 0.0, 0u32);
    let mut report_started = Instant::now();
    let mut late = 0;
    while !stop.load(Ordering::Relaxed) {
        if speed.paused() {
            // No ticks while paused; on resume, schedule from now and do
            // not let the idle time count against the measured tick rate.
            next = Instant::now();
            report_started = Instant::now();
            (tick_sum, capture_sum, count) = (0.0, 0.0, 0);
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }
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
        if feed.wants_whole() {
            sim.write_whole_anim();
        }
        // Interpolate positions over the tick length the viewer actually sees.
        let stream = sim.stream(TICK_SECONDS / rate)?;
        let (tick, units) = (stream.tick, stream.len());
        let view = sim.view_counts();
        sim.recycle(feed.publish_reusing(stream));
        tick_sum += (t1 - t0).as_secs_f64() * 1e3;
        capture_sum += t1.elapsed().as_secs_f64() * 1e3;
        count += 1;
        let elapsed = report_started.elapsed().as_secs_f32();
        if count == 30 || (count > 0 && elapsed >= 0.5) {
            let actual_speed = if elapsed > 0.0 {
                count as f32 * TICK_SECONDS / elapsed
            } else {
                0.0
            };
            let mut s = stats.lock().unwrap_or_else(|p| p.into_inner());
            *s = SimStats {
                tick,
                units,
                tick_ms: tick_sum / count as f64,
                capture_ms: capture_sum / count as f64,
                actual_speed,
                late,
                ready: true,
                failed: false,
                view,
            };
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
    use ae_render::units::{ATTR_VIEW, ATTRIBUTES};

    fn capture(tick: u64, rows: &[(u64, f32, f32, f32, f32)]) -> Capture {
        let mut c = Capture {
            tick,
            entities: Vec::new(),
            columns: vec![Vec::new(); FIELDS.len()],
        };
        for &(e, x, y, vx, vy) in rows {
            c.entities.push(e);
            for (f, word) in [
                x.to_bits(),
                y.to_bits(),
                vx.to_bits(),
                vy.to_bits(),
                2.5f32.to_bits(),
                (e % 3) as u32,
            ]
            .into_iter()
            .enumerate()
            {
                c.columns[f].push(word);
            }
        }
        c
    }

    fn column(s: &InstanceStream, name: &str) -> Vec<f32> {
        let a = ATTRIBUTES.iter().position(|&n| n == name).unwrap();
        s.columns[a].iter().map(|&w| f32::from_bits(w)).collect()
    }

    /// Rows come out in entity order; previous positions and turns follow
    /// entities across reordered captures; new units start where they are;
    /// standing units keep their turn; a unit without an `Orientation` is
    /// turned by its velocity.
    #[test]
    fn streams_match_rows_by_entity() {
        let mut s = Streamer::default();
        let now = Instant::now();
        let first = s.stream(
            &capture(1, &[(9, 90.0, 0.0, 0.0, 1.0), (4, 40.0, 0.0, 1.0, 0.0)]),
            &[],
            &[],
            now,
            0.1,
        );
        assert_eq!(column(&first, "x"), [40.0, 90.0]);
        assert_eq!(column(&first, "prev_x"), [40.0, 90.0], "first sight: no motion");
        // Table order changed, 4 stopped, 7 joined, 9 moved.
        let second = s.stream(
            &capture(
                2,
                &[
                    (7, 70.0, 5.0, 0.0, 0.0),
                    (9, 91.0, 0.5, 0.0, 1.0),
                    (4, 40.0, 0.0, 0.0, 0.0),
                ],
            ),
            &[],
            &[],
            now,
            0.1,
        );
        assert_eq!(column(&second, "x"), [40.0, 70.0, 91.0]);
        assert_eq!(column(&second, "prev_x"), [40.0, 70.0, 90.0]);
        assert_eq!(column(&second, "prev_y"), [0.0, 5.0, 0.0]);
        let half = std::f32::consts::FRAC_1_SQRT_2;
        assert_eq!(
            (
                column(&second, "rot_x"),
                column(&second, "rot_y"),
                column(&second, "rot_z"),
                column(&second, "rot_w"),
            ),
            (
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, -half],
                vec![0.0, 0.0, 0.0],
                vec![1.0, 1.0, half]
            ),
            "4 keeps its turn (+x), 7 gets the default (+x), 9 keeps its velocity's (+y)"
        );
        assert_eq!(second.columns[9], [1, 1, 0], "class = profile");
        assert_eq!(second.tick, 2);
        assert_eq!(second.columns[10], [NO_POSE; 3], "no poses without models");
        assert_eq!(column(&second, "water"), [0.0; 3]);
        assert_eq!(second.columns[ATTR_VIEW], [0; 3], "unfiltered stream");
        // 9 left.
        let third = s.stream(
            &capture(3, &[(4, 41.0, 0.0, 1.0, 0.0), (7, 70.0, 5.0, 0.0, 0.0)]),
            &[],
            &[],
            now,
            0.1,
        );
        assert_eq!(column(&third, "prev_x"), [40.0, 70.0]);
    }

    #[test]
    fn streams_pack_water_depth_and_observer_state() {
        let rows = [RowState::Own, RowState::Seen, RowState::Blip, RowState::Ghost].map(|state| UnitRow {
            key: state as u64,
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            radius: 1.0f32.to_bits(),
            class: 0,
            water_depth: if state == RowState::Seen { 4.5 } else { 0.0 },
            state,
        });
        let stream = Streamer::default().stream_rows(1, &rows, &[], &[], Instant::now(), 0.0);
        assert_eq!(column(&stream, "water"), [0.0, 4.5, 0.0, 0.0]);
        assert_eq!(stream.columns[ATTR_VIEW], [0, 1, 2, 3]);
        // The shaders' `eligible` (`view <= VIEW_SEEN`) takes Own and Seen,
        // never Blip or Ghost.
        let eligible = |state: RowState| state as u32 <= ae_render::units::VIEW_SEEN;
        assert!(eligible(RowState::Own) && eligible(RowState::Seen));
        assert!(!eligible(RowState::Blip) && !eligible(RowState::Ghost));
    }

    /// A unit with an `Orientation` is drawn by it, not by its velocity:
    /// one walking backwards is drawn backwards (A's regression), and a
    /// tilted rotation reaches the stream bit-exact. A unit without one
    /// keeps the velocity's turn.
    #[test]
    fn streams_draw_facing() {
        use ae_sdk::aim::rotation;
        let mut s = Streamer::default();
        let now = Instant::now();
        let back = Orientation::along(-1.0, 0.0);
        let tilt = Orientation::along(0.0, 1.0).then(rotation([1.0, 0.0, 0.0], 0.3));
        // 1 walks +x backwards, 2 walks +x tilted, 3 walks +x with no
        // `Orientation`, 4 stands with none (the default turn).
        let c = capture(
            1,
            &[
                (1, 0.0, 0.0, 1.0, 0.0),
                (2, 10.0, 0.0, 1.0, 0.0),
                (3, 20.0, 0.0, 1.0, 0.0),
                (4, 30.0, 0.0, 0.0, 0.0),
            ],
        );
        let stream = s.stream(&c, &[], &[(1, &back), (2, &tilt)], now, 0.1);
        let forward = Orientation::along(1.0, 0.0).rot;
        let expected = [back.rot, tilt.rot, forward, forward];
        for k in 0..4 {
            let words: Vec<u32> = stream.columns[4 + k].clone();
            assert_eq!(
                words,
                expected.map(|r| r[k].to_bits()),
                "row rotation {k}: 1 and 2 drawn by their `Orientation` bit-exact, not by their velocity; 3 and 4 by theirs"
            );
        }
        assert_eq!(back.to_world([1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0], "walking backwards");
    }

    /// With a poser, a unit drawn with a model whose `Anim` names it gets
    /// a slot of two animation records; streams write only the cells that
    /// changed, one record a tick.
    #[test]
    fn streams_carry_poses() {
        use ae_render::units::{Cell, POSE_CELLS, POSE_MANUAL, RECORD_CELLS};
        use ae_sdk::anim::{manual_flags, manual_kind};
        let arm = model_id("arm");
        // Class = entity % 3: class 0 is drawn with the arm, class 1 not.
        let mut s = Streamer {
            poser: Some(Poser::new(vec![Some(arm), None])),
            ..Streamer::default()
        };
        let turned = |angle: f32, model: u64| {
            let mut a = Anim::new(model);
            (a.bone[0], a.manual_kind[0]) = (1, manual_kind::TURN);
            let (sn, c) = (angle * 0.5).sin_cos();
            a.value[0] = [0.0, sn, 0.0, c];
            // Not shown yet: not in the record.
            (a.bone[1], a.manual_kind[1], a.manual_flags[1]) =
                (0, manual_kind::MOVE, manual_flags::FRESH | manual_flags::PENDING);
            (a.clip[2], a.mask[2], a.mode[2], a.kind[2], a.weight[2], a.phase[2]) = (3, 1, 2, 1, 0.5, 0.25);
            a
        };
        let record = crate::poser::record(&turned(0.1, arm), 7);
        assert_eq!(record[0], [7, 1, 0, 0], "the tick, one entry shown");
        assert_eq!(
            record[3],
            [3 | 1 << 16, 2 | 1 << 8, 0.5f32.to_bits(), 0.25f32.to_bits()],
            "layer 2"
        );
        assert_eq!(
            (record[1], record[5], record[7][1]),
            (
                [0; 4],
                [1 | (manual_kind::TURN as u32) << 16, 0, 0, 0],
                (0.05f32).sin().to_bits()
            )
        );
        let rows = [
            (3, 30.0, 0.0, 0.0, 0.0),
            (4, 40.0, 0.0, 0.0, 0.0),
            (6, 60.0, 0.0, 0.0, 0.0),
        ];
        let now = Instant::now();
        let anims = |angle: f32| {
            vec![
                (3, turned(angle, arm)),
                (4, turned(angle, arm)),
                (6, turned(angle, model_id("other"))),
            ]
        };
        fn refs(v: &[(u64, Anim)]) -> Vec<(u64, &Anim)> {
            v.iter().map(|(e, a)| (*e, a)).collect()
        }
        let mut table: Vec<Cell> = Vec::new();
        let ticks = |table: &[Cell]| (table[0][0], table[RECORD_CELLS][0]);
        let first = s.stream(&capture(10, &rows), &refs(&anims(0.1)), &[], now, 0.1);
        assert_eq!(
            first.columns[ATTR_POSE],
            [POSE_MANUAL, NO_POSE, NO_POSE],
            "4 is drawn as a dart, 6 is another model; 3 shows a manual joint"
        );
        assert!(
            first.anim.reset && first.anim.at == (0..POSE_CELLS as u32).collect::<Vec<_>>(),
            "a new table, both records: {:?}",
            first.anim.at
        );
        first.anim.apply(&mut table);
        assert_eq!(ticks(&table), (9, 10));
        assert_eq!(
            table[1..RECORD_CELLS],
            table[RECORD_CELLS + 1..],
            "first posed: no motion"
        );
        // The next tick's turn: the newer record is the last tick's, so
        // the new one goes over the older: its tick and the entry's value.
        let second = s.stream(&capture(11, &rows), &refs(&anims(0.2)), &[], now, 0.1);
        assert!(!second.anim.reset);
        assert_eq!(second.anim.at, [0, 7], "the older record's changed cells");
        second.anim.apply(&mut table);
        assert_eq!(ticks(&table), (11, 10));
        // Held there: nothing written.
        assert!(
            s.stream(&capture(12, &rows), &refs(&anims(0.2)), &[], now, 0.1)
                .anim
                .is_empty()
        );
        // A tick later it turns back: the newer (11) stamped the last tick
        // (13, it held so), the new one over the older (which held that
        // turn already: only its tick).
        let fourth = s.stream(&capture(14, &rows), &refs(&anims(0.1)), &[], now, 0.1);
        assert_eq!(fourth.anim.at, [0, RECORD_CELLS as u32]);
        fourth.anim.apply(&mut table);
        assert_eq!(ticks(&table), (13, 14));
        assert_eq!(table[RECORD_CELLS + 7][1], (0.05f32).sin().to_bits());
        assert!(
            s.stream(&capture(15, &rows), &refs(&anims(0.1)), &[], now, 0.1)
                .anim
                .is_empty()
        );
        // The reader missed writes: the whole table, as it stands.
        s.poser.as_mut().unwrap().write_whole();
        let whole = s.stream(&capture(16, &rows), &refs(&anims(0.1)), &[], now, 0.1);
        assert!(whole.anim.reset && whole.anim.at.len() == POSE_CELLS);
        assert_eq!(whole.anim.cells, table, "the cells as written");
        assert!(
            s.stream(&capture(17, &rows), &refs(&anims(0.1)), &[], now, 0.1)
                .anim
                .is_empty(),
            "then only what changes"
        );
        // Unit 3 gone, unit 6 now an arm: it takes the freed slot, fresh.
        let rows2 = [(4, 40.0, 0.0, 0.0, 0.0), (6, 60.0, 0.0, 0.0, 0.0)];
        let mut moved = anims(0.1);
        moved[2].1.model = arm;
        let sixth = s.stream(&capture(18, &rows2), &refs(&moved[1..]), &[], now, 0.1);
        assert_eq!(sixth.columns[ATTR_POSE], [NO_POSE, POSE_MANUAL]);
        assert_eq!(sixth.anim.at.len(), POSE_CELLS);
        assert_eq!(sixth.sim_seconds, TICK_SECONDS, "sim time per tick, whatever the speed");
        // Its manual joint gone: marked while the older record shows it.
        moved[2].1.manual_kind = [manual_kind::NONE; _];
        let seventh = s.stream(&capture(19, &rows2), &refs(&moved[1..]), &[], now, 0.1);
        assert_eq!(
            seventh.columns[ATTR_POSE],
            [NO_POSE, POSE_MANUAL],
            "the older record still shows it"
        );
        // (Held, nothing is written and the older keeps it: as the table.)
        let eighth = s.stream(&capture(20, &rows2), &refs(&moved[1..]), &[], now, 0.1);
        assert_eq!(
            eighth.columns[ATTR_POSE],
            [NO_POSE, POSE_MANUAL],
            "held: the records as they were"
        );
        moved[2].1.phase[2] = 0.5;
        let ninth = s.stream(&capture(21, &rows2), &refs(&moved[1..]), &[], now, 0.1);
        assert_eq!(ninth.columns[ATTR_POSE], [NO_POSE, 0], "neither does");
    }

    /// A unit gone since the last capture explodes where it stood, if the
    /// observer saw there; units that stay or join do not.
    #[test]
    fn units_that_vanish_where_seen_explode() {
        let last = [
            (1, 10.0, 10.0, 6.0, 0.0),
            (2, 20.0, 20.0, 6.0, 0.0),
            (3, -30.0, 30.0, 6.0, 0.0),
            (5, 50.0, 50.0, 12.0, 0.0),
        ];
        let now = [
            (1, 11.0, 10.0, 6.0, 0.0),
            (4, 40.0, 40.0, 6.0, 0.0),
            (5, 51.0, 50.0, 12.0, 0.0),
        ];
        let sites = deaths(&last, &now, &[], |x, _, _| x >= 0.0);
        assert_eq!(sites.len(), 1, "only 2 died where seen (3 died in the fog): {sites:?}");
        assert!(
            deaths(&last, &now, &[2], |x, _, _| x >= 0.0).is_empty(),
            "2 died by script (it had Dying): its death effects are its script's"
        );
        assert_eq!((sites[0].0, sites[0].1, sites[0].3), (20.0, 20.0, 1.0));
        // One explosion: flash, fire, sparks, smoke, and a scorch mark.
        let effects = Effects {
            particles: Some(Arc::default()),
            decals: Some(Arc::default()),
            ..Effects::default()
        };
        effects.explode(20.0, 20.0, 1.0, sites[0].4);
        let bursts = effects.particles.as_ref().unwrap().take_all();
        assert_eq!(bursts.len(), 4, "flash, fire, sparks, smoke: {bursts:?}");
        assert!(bursts.iter().all(|b| b.pos[0] == 20.0 && b.pos[2] == 20.0 && b.ground));
        let decals = effects.decals.as_ref().unwrap().take();
        assert_eq!(decals.len(), 1);
        assert_eq!(decals[0].pos, [20.0, 20.0]);
        assert!(
            decals[0].glow > 0.5 && decals[0].glow_life > 0.0 && decals[0].layer == layer::SCORCH,
            "{:?}",
            decals[0]
        );
        assert!(deaths(&last, &last, &[], |_, _, _| true).is_empty(), "nobody died");
    }

    #[test]
    fn submerged_deaths_require_sonar_even_in_line_of_sight() {
        let mut effects = Effects::default();
        let particles = Arc::new(ParticleFeed::default());
        effects.particles = Some(particles.clone());
        let last = [(7, 35.0, 35.0, 6.0, 4.0)];
        let grid = SensorGrid::new(128.0, 128.0, 32.0);
        let mut los = Bits::new(grid.width, grid.height);
        let mut sonar = los.clone();
        los.set(1, 1);
        let emit = || {
            let sites = deaths(&last, &[], &[], |x, z, sub| effects.eligible(x, z, sub));
            for (x, z, depth, scale, seed) in &sites {
                effects.explode_at_depth(*x, *z, *depth, *scale, *seed);
            }
            sites.len()
        };
        effects.set_sight(los.clone(), sonar.clone(), grid);
        assert_eq!(emit(), 0);
        assert!(particles.take_all().is_empty());
        sonar.set(1, 1);
        effects.set_sight(los, sonar, grid);
        assert_eq!(emit(), 1);
        let bursts = particles.take_all();
        assert_eq!(bursts.len(), 4);
        assert!(bursts.iter().all(|b| !b.ground && b.pos[1] == -4.0));
    }

    /// A sim effect becomes a burst only where the observer sees it, with
    /// the catalogue's index for its name; an unknown name is skipped and
    /// warned about once.
    #[test]
    fn effects_become_bursts_only_where_seen() {
        let fx = Effects {
            particles: Some(Arc::default()),
            ..Effects::default()
        };
        let flash = ae_sdk::abi::name_id("flash");
        let effect = |x: f32, y: f32, effect: u64| Effect {
            entity: Entity(1),
            tick: 5,
            pos: [x, 3.0, y],
            dir: [1.0, 0.0, 0.0],
            effect,
            count: 2,
            scale: 1.5,
            seed: 9,
            bone: 3,
            flags: 0,
        };
        // No observer: everything is seen.
        let bursts = fx_bursts(&[effect(5.0, 6.0, flash)], &fx, &mut Vec::new(), |_| 0.0);
        assert_eq!(bursts.len(), 1);
        assert_eq!(
            (
                bursts[0].pos,
                bursts[0].ground,
                bursts[0].effect,
                bursts[0].count,
                bursts[0].scale
            ),
            ([5.0, 3.0, 6.0], true, 0, 2, 1.5),
            "flash is the catalogue's first effect"
        );
        // One seen, one in the fog.
        let grid = SensorGrid::new(100.0, 100.0, 10.0);
        let mut los = Bits::new(grid.width, grid.height);
        los.set(0, 0);
        fx.set_sight(los.clone(), Bits::new(grid.width, grid.height), grid);
        let seen = effect(5.0, 6.0, flash);
        let unseen = effect(55.0, 55.0, flash);
        assert_eq!(fx_bursts(&[seen, unseen], &fx, &mut Vec::new(), |_| 0.0).len(), 1);
        // An unknown name: skipped, and only warned about once.
        let mut warned = Vec::new();
        let unknown = effect(5.0, 6.0, ae_sdk::abi::name_id("nope"));
        assert!(fx_bursts(&[unknown], &fx, &mut warned, |_| 0.0).is_empty());
        assert_eq!(warned, [ae_sdk::abi::name_id("nope")]);
        assert!(fx_bursts(&[unknown], &fx, &mut warned, |_| 0.0).is_empty());
        assert_eq!(warned.len(), 1);
    }

    /// An explosion set off by hand is seen whole or not at all: where the
    /// observer sees, particles and a scar; elsewhere neither.
    #[test]
    fn explosions_by_hand_go_off_only_in_sight() {
        let effects = Effects {
            particles: Some(Arc::default()),
            decals: Some(Arc::default()),
            ..Effects::default()
        };
        assert!(effects.eligible(500.0, 500.0, false), "no observer: everything is seen");
        let grid = SensorGrid::new(100.0, 100.0, 10.0);
        let mut los = Bits::new(grid.width, grid.height);
        los.set(2, 3);
        effects.set_sight(los, Bits::new(grid.width, grid.height), grid);
        assert!(!effects.explode_if_seen(55.0, 55.0, 1.0, 7));
        assert!(
            effects.particles.as_ref().unwrap().take_all().is_empty()
                && effects.decals.as_ref().unwrap().take().is_empty()
        );
        assert!(effects.explode_if_seen(25.0, 35.0, 1.0, 7));
        assert_eq!(effects.particles.as_ref().unwrap().take_all().len(), 4);
        assert_eq!(effects.decals.as_ref().unwrap().take().len(), 1);
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

    /// Pausing holds the rate (speed changes still apply on resume) and
    /// reports the new state, so the window can mirror it on the renderer.
    #[test]
    fn pause_toggles_without_changing_the_rate() {
        let speed = SimSpeed::default();
        assert!(!speed.paused());
        assert!(speed.toggle_pause());
        assert!(speed.paused());
        assert_eq!(speed.current(), 1.0);
        assert_eq!(speed.faster(), 2.0);
        assert!(speed.paused(), "a speed change does not resume");
        assert!(!speed.toggle_pause());
        assert!(!speed.paused());
        assert_eq!(speed.current(), 2.0);
    }
}
