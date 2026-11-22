//! Windowed client.
//!
//! ```text
//! ae-client [--map water|land|generated|demo] [--map-cells N] [--relief X]
//!           [--camera X,Z,DIST,YAW,PITCH] [--seconds S] [--exercise] [--mailbox]
//!           [--frames-in-flight N] [--units N] [--workers N] [--ticks N]
//!           [--render-module NAME|FILE.wasm|FILE.json] [--capture FILE.ppm]
//! ```
//!
//! `--render-module` takes the frame graph from a render module: an
//! in-repo one (`render/<NAME>`: `default`, the engine's own graph from
//! WASM; `post`, with a colour grade and a vignette N toggles), a `.wasm`
//! file, or a graph file (`.json`, e.g. `render/post.graph.json`: the same
//! graph as data, `ae_render::graph_file`). Its sources (or the file) are watched: an edit rebuilds it
//! and the running client swaps it in at the next frame, or keeps the old
//! one and prints why.
//!
//! `--probe-display` prints SDL's HDR state for the window and the (format,
//! colour space) pairs the Vulkan surface offers, with the device's HDR
//! extensions, then exits; it renders nothing. The HDR output path is chosen
//! from its report.
//!
//! Shows the terrain of the M2 crossing maps (`water`, the default, or
//! `land`; `--map-cells` as in `ae-headless`, default 2560 = pathsim's full
//! map), map-rendering's generated test map (`generated`), or the frame
//! graph demo (`demo`). The nav heightfield is taken as is: decimetres to
//! world units (1 unit = 1 m) times `--relief` (default 1). On a map with
//! water, a water surface (waves, sky reflection, sun glints, shore foam)
//! is drawn over the ground under it, which is tinted by its depth and lit
//! by caustics; `--no-water` leaves only the tint. The hills cast the
//! sun's shadows over the ground, the units and the water; `--no-shadows`
//! turns them off. Units that die where the observer sees explode
//! (particles: flash, fire, sparks, smoke; X sets one off at the camera
//! target, `--explode N` sets off N around it before a capture, each only
//! where the observer sees: an explosion is seen whole or not at all;
//! `--no-particles` turns them off; they draw at half resolution,
//! `--particles-full` at full) and leave a scar on the ground (a decal,
//! relit by its normal map, its centre glowing like hot metal for up to
//! two seconds; `--no-decals`). The scars are Recoil's
//! (`assets/decals/scars`; `--scars DIR` for another set, `--scars
//! generated` for the engine's generated scorch mark).
//!
//! On the crossing maps the sim runs beside the renderer: the crossing game
//! (`games/crossing`, built on first use) with `--units` units (default
//! 10000, 0 for none) on the same map, `--workers` sim workers, ticked at
//! 30 Hz on its own thread. Units are drawn as darts coloured by movement
//! class (scout yellow, heavy red, ship white, hovercraft cyan, amphibious
//! purple), interpolated between ticks; ships and hovercraft sit on the
//! water surface. The terrain shows at once, the units once the world is
//! built.
//!
//! `--model [CLASS=]FILE` draws class CLASS (every class without it, but
//! the observer's blips) with a model: an `.aemodel`, or a glTF converted
//! once by `ae-tool` into `target/models` (`models.rs`). The model is also
//! the sim's animation model named by the file's lower-case stem: units
//! whose `Anim` names it are posed by their channels (`--game scripted`:
//! Foxes walking and standing, `--model Fox.glb`); others stand at the
//! rest pose. A model replaces the class colour (no enemy tint).
//!
//! Controls: WASD / arrows pan, wheel zooms, right- or middle-drag orbits
//! and tilts, Q/E orbit, R/F tilt, shift faster; `+`/`-` sim speed (doubles
//! with no upper limit; the sim becomes compute-bound), Pause pauses and
//! resumes the sim and every animation clock (water, decals, particles),
//! `[`/`]` more or less terrain detail, T wireframe, L colour by LOD depth,
//! G fog, C freeze the LOD, P print the camera, V v-sync (on by default;
//! `--mailbox` starts it off), N toggle render module flag bit 0, M tileset,
//! B blend weights, O water surface, H shadows, Z bloom, X explosion,
//! U HDR/SDR, Esc quits.
//!
//! `--hdr off|auto|hdr10|scrgb` picks the swapchain's encoding (M13 stage 5;
//! auto by default: HDR10 when the surface offers the pair and SDL reports
//! the window HDR-capable, else scRGB, else SDR). An explicit `hdr10` or
//! `scrgb` the surface does not offer is an error naming what it does; it
//! never falls back silently. `--paper-white` and `--peak` (nits) override
//! the levels, both in the signal's nits; the defaults are the reference the
//! Wayland WSI's image description carries (203 nits for PQ, and
//! `set_luminances(0, 80, 203)` for scRGB) and that reference times SDL's
//! headroom (203 × 1.778 = 361 here). The compositor maps the signal's
//! reference onto its own (248 here), so 361 shows at the panel's 441. The swapchain is
//! recreated on `SDL_EVENT_WINDOW_HDR_STATE_CHANGED` and display changes,
//! and `vkSetHdrMetadataEXT` is called after every recreate. `AE_HDR_TOGGLE_SECONDS=N`
//! flips HDR and SDR every N seconds through the `U` key's path (a test
//! hook: a Wayland session has no synthetic keyboard). A render module
//! has no HDR support until M13 stage 6: with `--render-module` the client
//! falls back to SDR and says so once.
//!
//! `--capture` renders 1280x720 offscreen (no window; 60 frames so the
//! subdivision settles) and writes a binary PPM; with units it first runs
//! `--ticks` sim ticks (default 300) and shows that tick. It always uses the
//! SDR readback: a capture shows neither the HDR encoding nor the display's
//! levels.
//!
//! `--bench` measures fixed scenes offscreen and writes `ae_bench` reports
//! (see `bench.rs`; `--bench list` names them).
//!
//! `--game skirmish` plays the skirmish game (`games/skirmish`, its content
//! from `--content`, default `games/skirmish/content`; `--param` overrides
//! the game parameter) as allyteam 0 against allyteam 1's bot
//! (`ae_bot::SkirmishBot`; with `--observer all` the two bots play each
//! other): left click selects your units near the clicked ground (white),
//! right click moves them there, ctrl+right click attack-moves, and with
//! shift either is queued after their current commands. New units gather
//! at your rally point in front of the base.
//!
//! `--package SPEC|PATH` (repeatable, with `--packages DIR` libraries)
//! plays a game from packages alone: the game package's `module.wasm` and
//! content, a map package's terrain and start points (`ae_map`), e.g.
//! `--package skirmish.aegame --package crossing.aemap`.
//!
//! `--teams N` splits the crossing maps' units into N allyteams (with
//! sensors); `--observer K` (default 0 with teams; `all` for everything)
//! then draws the world as allyteam K sees it through the copy layer: its
//! own units in their class colours, enemies in line of sight tinted red,
//! radar blips grey, and the terrain lit where K sees now, dimmed where it
//! has looked before, dark elsewhere.
//!
//! `--record-threads N` records the frame's passes on N threads, one
//! command buffer per pass (default 1: all on the render thread).
//!
//! `--seconds` closes the window after S seconds; `--exercise` sets off
//! explosions every second (with particles) and resizes,
//! minimizes and restores on a schedule. Validation follows
//! `AE_VK_VALIDATION` (default on in debug builds); the exit status is 1 if
//! the layer reported anything.

mod bench;
mod ground;
mod models;
mod poser;
mod sim;
mod watch;

use ae_nav::terrain::{NO_WATER, Terrain};
use ae_services::nav::{GroundFeed, Map};
use ae_render::demo::DemoApp;
use ae_render::particles::{ParticleFeed, ParticlesSetup};
use ae_render::swapchain::{HdrRequest, PresentPolicy};
use ae_render::terrain::app::{HdrSettings, TerrainApp, TerrainOutput, TerrainSource, TerrainView, VisionFeed, lod_view};
use ae_render::terrain::edit::TerrainEditor;
use ae_render::terrain::tileset::{self, Tileset};
use ae_render::terrain::heightmap::{Heightmap, build_pyramid, generate};
use ae_render::terrain::renderer::{FrameStats, TerrainInfo, Wireframe};
use ae_render::thread::{GpuRenderConfig, GpuRenderThread, HdrControl};
use ae_render::units::{UnitFeed, UnitsSetup};
use ae_render::window::{SdlWindow, keys};
use anyhow::{Context, Result, bail};
use glam::Vec2;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq)]
enum MapKind {
    Water,
    Land,
    Generated,
    Demo,
    /// A map package's (`--package`).
    Package,
}

struct Options {
    map: MapKind,
    map_cells: i32,
    relief: f32,
    camera: Option<[f32; 5]>,
    seconds: Option<f32>,
    exercise: bool,
    present: PresentPolicy,
    frames_in_flight: usize,
    /// Render offscreen without a window and write this PPM file.
    capture: Option<String>,
    units: u32,
    workers: usize,
    /// Sim ticks before a capture.
    ticks: u64,
    render_module: Option<String>,
    quake: u8,
    /// Start in the tileset's blend-weight view.
    weights: bool,
    /// Draw the water surface on maps with water.
    water: bool,
    /// The sun's shadows.
    shadows: bool,
    /// The tonemap's exposure and curve.
    tone: ae_render::tonemap::ToneParams,
    /// Particles (explosions).
    particles: bool,
    /// Scorch marks where explosions go off (`--no-decals`).
    decals: bool,
    /// The scars' directory (`--scars DIR`; `generated`: the generated
    /// scorch mark); [`BUNDLED_SCARS`] when `None`.
    scars: Option<String>,
    /// Particles at full resolution (`--particles-full`), not half.
    particles_full: bool,
    /// `--particle-blend sorted|moments|slices|froxels|packets` (comparison; sorted by default).
    particle_blend: ae_render::particles::ParticleBlend,
    /// `--explode`: explosions around the camera target before a capture.
    explode: u32,
    /// `--capture-frames`: frames rendered before the capture is read.
    capture_frames: u32,
    /// `--lights N`: N fixed point lights scattered around the camera
    /// target (a test of the clustered lights).
    lights: u32,
    /// Material tileset name, `None` for the palette.
    tileset: Option<String>,
    tileset_size: u32,
    /// `--probe-display`: print the display's HDR capabilities and the
    /// surface's colour spaces, render nothing, and exit.
    probe_display: bool,
    /// The scene's working space (M13 stage 3), BT.709 by default.
    working_space: ae_render::colorspace::WorkingSpace,
    /// The output gamut (M13 stage 3); `None` is the encoding's default
    /// (P3 for HDR10). Only an HDR output maps a gamut.
    display_gamut: Option<ae_render::colorspace::Gamut>,
    /// The swapchain's encoding (`--hdr off|auto|hdr10|scrgb`, M13 stage 5).
    hdr: HdrRequest,
    /// `--paper-white` and `--peak` (nits); `None` takes the encoding's
    /// measured reference and SDL's headroom.
    paper_white: Option<f32>,
    peak: Option<f32>,
    /// `--bench`: scenes to measure instead of opening a window.
    bench: Option<bench::BenchOptions>,
    /// Threads recording render passes (1: the render thread alone).
    record_threads: usize,
    /// Allyteams of the crossing maps' units (0: none).
    teams: u32,
    /// The allyteam whose view is drawn (`None`: everything).
    observer: Option<u32>,
    /// `crossing` (default) or `skirmish`.
    game: String,
    content: Option<String>,
    param: Option<u64>,
    /// `--package SPEC|PATH` and `--packages DIR`, and what they gave: the
    /// mounts in order, the game's module and the map (terrain, starts).
    packages: Vec<String>,
    package_dirs: Vec<String>,
    mounts: Vec<std::path::PathBuf>,
    wasm: Option<Arc<Vec<u8>>>,
    loaded_map: Option<(Arc<Terrain>, Vec<[f32; 2]>)>,
    /// `--model [CLASS=]FILE`, in order.
    models: Vec<models::ClassModel>,
}

impl Default for Options {
    fn default() -> Self {
        Self { map: MapKind::Water, map_cells: 2560, relief: 1.0, camera: None, seconds: None, exercise: false, present: PresentPolicy::Fifo, frames_in_flight: 2, capture: None, units: 10_000, workers: default_workers(), ticks: 300, render_module: None, quake: 0, tileset: Some("temperate".into()), tileset_size: 512, weights: false, water: true, shadows: true, tone: ae_render::tonemap::ToneParams { auto: true, ..Default::default() }, particles: true, decals: true, scars: None, particles_full: false, particle_blend: ae_render::particles::ParticleBlend::Sorted, explode: 0, capture_frames: 60, lights: 0, bench: None, record_threads: 1, teams: 0, observer: None, game: "crossing".into(), content: None, param: None, packages: Vec::new(), package_dirs: Vec::new(), mounts: Vec::new(), wasm: None, loaded_map: None, models: Vec::new(), probe_display: false, working_space: Default::default(), display_gamut: None, hdr: HdrRequest::Auto, paper_white: None, peak: None }
    }
}

const USAGE: &str = "ae-client [--map water|land|generated|demo] [--map-cells N] [--relief X] [--camera X,Z,DIST,YAW,PITCH] [--seconds S] [--exercise] [--probe-display] [--mailbox] [--frames-in-flight N] [--units N] [--workers N] [--ticks N] [--render-module NAME|FILE.wasm|FILE.json] [--quake TICK] [--tileset temperate|desert|none] [--tileset-size N] [--weights] [--no-water] [--no-shadows] [--tonemap clamp|lottes] [--ev100 X] [--auto-exposure] [--sun-lux X] [--no-particles] [--no-decals] [--scars DIR|generated] [--particles-full] [--particle-blend sorted|moments|slices|froxels|packets] [--working-space bt709|bt2020] [--display-gamut bt709|p3|bt2020] [--hdr off|auto|hdr10|scrgb] [--paper-white N] [--peak N] [--explode N] [--capture-frames N] [--lights N] [--record-threads N] [--teams N] [--observer K|all] [--game crossing|skirmish|scripted] [--content DIR] [--package SPEC|PATH ... [--packages DIR]] [--param N] [--model [CLASS=]FILE.aemodel|.glb|.gltf ...] [--capture FILE.ppm]
       ae-client --bench SCENE[,SCENE...]|all|list [--frames N] [--warmup N] [--label NAME] [--out DIR] [--no-save]";

fn parse() -> Result<Options> {
    let mut o = Options::default();
    let mut observer_given = false;
    // `--ev100` alone is a manual exposure; with `--auto-exposure` (either
    // order) it is the reference the scene is stored at.
    let (mut ev100_given, mut auto_given) = (false, false);
    let mut args = std::env::args().skip(1);
    let value = |name: &str, args: &mut dyn Iterator<Item = String>| args.next().with_context(|| format!("{name} needs a value"));
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--map" => {
                o.map = match value("--map", &mut args)?.as_str() {
                    "water" => MapKind::Water,
                    "land" => MapKind::Land,
                    "generated" => MapKind::Generated,
                    "demo" => MapKind::Demo,
                    other => bail!("--map {other:?}: expected water, land, generated or demo"),
                }
            }
            "--map-cells" => o.map_cells = value("--map-cells", &mut args)?.parse().context("--map-cells")?,
            "--relief" => o.relief = value("--relief", &mut args)?.parse().context("--relief")?,
            "--camera" => {
                let text = value("--camera", &mut args)?;
                let parts: Vec<f32> = text.split(',').map(str::parse).collect::<Result<_, _>>().context("--camera X,Z,DIST,YAW,PITCH")?;
                let parts: [f32; 5] = parts.try_into().map_err(|_| anyhow::anyhow!("--camera takes five numbers"))?;
                o.camera = Some(parts);
            }
            "--seconds" => o.seconds = Some(value("--seconds", &mut args)?.parse().context("--seconds")?),
            "--exercise" => o.exercise = true,
            "--probe-display" => o.probe_display = true,
            "--working-space" => o.working_space = value("--working-space", &mut args)?.parse().context("--working-space")?,
            "--display-gamut" => o.display_gamut = Some(value("--display-gamut", &mut args)?.parse().context("--display-gamut")?),
            "--hdr" => {
                o.hdr = match value("--hdr", &mut args)?.as_str() {
                    "off" => HdrRequest::Off,
                    "auto" => HdrRequest::Auto,
                    "hdr10" => HdrRequest::Hdr10,
                    "scrgb" => HdrRequest::ScRgb,
                    other => bail!("--hdr {other:?}: expected off, auto, hdr10 or scrgb"),
                }
            }
            "--paper-white" => o.paper_white = Some(value("--paper-white", &mut args)?.parse().context("--paper-white")?),
            "--peak" => o.peak = Some(value("--peak", &mut args)?.parse().context("--peak")?),
            "--capture" => o.capture = Some(value("--capture", &mut args)?),
            "--units" => o.units = value("--units", &mut args)?.parse().context("--units")?,
            "--workers" => o.workers = value("--workers", &mut args)?.parse::<usize>().context("--workers")?.max(1),
            "--ticks" => o.ticks = value("--ticks", &mut args)?.parse().context("--ticks")?,
            "--render-module" => o.render_module = Some(value("--render-module", &mut args)?),
            "--tileset" => {
                let name = value("--tileset", &mut args)?;
                o.tileset = match name.as_str() {
                    "none" => None,
                    n if ae_render::terrain::tileset::NAMES.contains(&n) => Some(name),
                    n => bail!("--tileset {n:?}: expected one of {:?} or none", ae_render::terrain::tileset::NAMES),
                };
            }
            "--tileset-size" => o.tileset_size = value("--tileset-size", &mut args)?.parse().context("--tileset-size")?,
            "--weights" => o.weights = true,
            "--no-water" => o.water = false,
            "--no-shadows" => o.shadows = false,
            "--tonemap" => {
                o.tone.curve = match value("--tonemap", &mut args)?.as_str() {
                    "clamp" => ae_render::tonemap::Curve::Clamp,
                    "lottes" => ae_render::tonemap::Curve::Lottes,
                    other => bail!("--tonemap {other:?}: clamp or lottes"),
                }
            }
            "--ev100" => {
                o.tone.ev100 = value("--ev100", &mut args)?.parse().context("--ev100")?;
                ev100_given = true;
            }
            "--auto-exposure" => auto_given = true,
            "--sun-lux" => o.tone.sun_illuminance = value("--sun-lux", &mut args)?.parse().context("--sun-lux")?,
            "--no-particles" => o.particles = false,
            "--no-decals" => o.decals = false,
            "--scars" => o.scars = Some(value("--scars", &mut args)?),
            "--particles-full" => o.particles_full = true,
            "--particle-blend" => {
                o.particle_blend = match value("--particle-blend", &mut args)?.as_str() {
                    "sorted" => ae_render::particles::ParticleBlend::Sorted,
                    "moments" => ae_render::particles::ParticleBlend::Moments,
                    "slices" => ae_render::particles::ParticleBlend::Slices,
                    "froxels" => ae_render::particles::ParticleBlend::Froxels,
                    "packets" => ae_render::particles::ParticleBlend::Packets,
                    other => anyhow::bail!("--particle-blend {other}: sorted, moments, slices, froxels or packets"),
                }
            }
            "--explode" => o.explode = value("--explode", &mut args)?.parse().context("--explode N")?,
            "--capture-frames" => o.capture_frames = value("--capture-frames", &mut args)?.parse::<u32>().context("--capture-frames N")?.max(1),
            "--lights" => o.lights = value("--lights", &mut args)?.parse().context("--lights N")?,
            "--quake" => o.quake = value("--quake", &mut args)?.parse().context("--quake TICK (1..=255)")?,
            "--bench" => o.bench.get_or_insert_with(Default::default).target = value("--bench", &mut args)?,
            "--frames" => o.bench.get_or_insert_with(Default::default).frames = value("--frames", &mut args)?.parse().context("--frames")?,
            "--warmup" => o.bench.get_or_insert_with(Default::default).warmup = value("--warmup", &mut args)?.parse().context("--warmup")?,
            "--label" => o.bench.get_or_insert_with(Default::default).label = Some(value("--label", &mut args)?),
            "--out" => o.bench.get_or_insert_with(Default::default).out = value("--out", &mut args)?.into(),
            "--no-save" => o.bench.get_or_insert_with(Default::default).save = false,
            "--teams" => o.teams = value("--teams", &mut args)?.parse().context("--teams")?,
            "--game" => o.game = value("--game", &mut args)?,
            "--package" => o.packages.push(value("--package", &mut args)?),
            "--packages" => o.package_dirs.push(value("--packages", &mut args)?),
            "--content" => o.content = Some(value("--content", &mut args)?),
            "--param" => o.param = Some(value("--param", &mut args)?.parse().context("--param")?),
            "--model" => o.models.push(models::parse_arg(&value("--model", &mut args)?)?),
            "--observer" => {
                observer_given = true;
                o.observer = match value("--observer", &mut args)?.as_str() {
                    "all" => None,
                    n => Some(n.parse().context("--observer N|all")?),
                }
            }
            "--record-threads" => o.record_threads = value("--record-threads", &mut args)?.parse::<usize>().context("--record-threads")?.max(1),
            "--fifo" => o.present = PresentPolicy::Fifo,
            "--mailbox" => o.present = PresentPolicy::Mailbox,
            "--frames-in-flight" => o.frames_in_flight = value("--frames-in-flight", &mut args)?.parse().context("--frames-in-flight")?,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}\n{USAGE}"),
        }
    }
    if !o.packages.is_empty() {
        let packages = ae_vfs::resolve_specs(&o.package_dirs, &o.packages)?;
        let mut vfs = ae_vfs::Vfs::new();
        vfs.mount_packages(&packages)?;
        if let Some(game) = packages.iter().rev().find(|p| p.manifest.kind == "game") {
            o.game = game.manifest.name.clone();
        }
        if vfs.exists("module.wasm") {
            o.wasm = Some(Arc::new(vfs.read("module.wasm")?));
        }
        if vfs.exists(ae_map::MAP_JSON) {
            let map = ae_map::GameMap::load(&vfs)?;
            o.map = MapKind::Package;
            o.map_cells = map.info.cells;
            o.loaded_map = Some((Arc::new(map.to_terrain()), map.info.starts.clone()));
        }
        o.mounts = packages.into_iter().map(|p| p.source).collect();
        // The packages' defaults (start points, the game's own) rule, not
        // the client's: parameter 0 unless `--param` says otherwise.
        o.param.get_or_insert(0);
        eprintln!("packages: {} mount(s), game {}, {}", o.mounts.len(), o.game, if o.loaded_map.is_some() { "map from a package" } else { "no map package" });
    }
    if o.game == "skirmish" {
        o.teams = o.teams.max(2);
        if o.content.is_none() && o.mounts.is_empty() {
            o.content = Some(concat!(env!("CARGO_MANIFEST_DIR"), "/../../games/skirmish/content").into());
        }
    }
    if o.teams > 0 && !observer_given {
        o.observer = Some(0);
    }
    if let Some(k) = o.observer {
        anyhow::ensure!(k < o.teams, "--observer {k} needs --teams of at least {}", k + 1);
    }
    if o.bench.as_ref().is_some_and(|b| b.target.is_empty()) {
        bail!("--frames, --warmup, --label, --out and --no-save go with --bench\n{USAGE}");
    }
    if auto_given {
        o.tone.auto = true;
    } else if ev100_given {
        // A fixed EV is the manual exposure: auto-exposure would cancel it
        // out.
        o.tone.auto = false;
    }
    Ok(o)
}

/// Half the cores, 1 to 8: the nav service's pool and the sim workers
/// share them with the render and main threads.
fn default_workers() -> usize {
    (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2) / 2).clamp(1, 8)
}

/// The sim for this map, if it has units.
fn sim_options(o: &Options) -> Option<sim::SimOptions> {
    let map = match o.map {
        MapKind::Water => Map::WaterCrossing,
        MapKind::Land => Map::LandCrossing,
        MapKind::Package => Map::Loaded,
        MapKind::Generated | MapKind::Demo => return None,
    };
    let mounts = o.mounts.iter().cloned().chain(o.content.iter().map(Into::into)).collect();
    (o.units > 0).then_some(sim::SimOptions { map, map_cells: o.map_cells, units: o.units, workers: o.workers, quake: o.quake, teams: o.teams, observer: o.observer, game: o.game.clone(), mounts, param: o.param, wasm: o.wasm.clone(), loaded_map: o.loaded_map.clone(), models: o.models.clone() })
}

fn units_setup(s: &sim::SimOptions, feed: Arc<UnitFeed>) -> Result<UnitsSetup> {
    let classes = s.classes();
    if let Some(c) = s.models.iter().filter_map(|m| m.class).find(|&c| c as usize >= classes.len()) {
        bail!("--model {c}=...: there are {} unit classes (0 to {})", classes.len(), classes.len() - 1);
    }
    // With an observer the classes end with the blips' and the selection's.
    let blip = s.observer.map(|_| classes.len() - 2);
    let models = models::per_class(&s.models, classes.len(), blip);
    // Each class's model posed by the sim's rig of that name (its clips
    // and masks), made as the sim makes it.
    let rigs: Vec<(String, Arc<ae_model::anim::Rig>)> = s.rigs()?.into_iter().map(|(n, r)| (n, Arc::new(r))).collect();
    let rigs = models::names_per_class(&s.models, classes.len(), blip)
        .into_iter()
        .map(|name| rigs.iter().find(|(n, _)| Some(n) == name.as_ref()).map(|(_, r)| r.clone()))
        .collect();
    Ok(UnitsSetup { feed, capacity: s.units, classes, min_pixels: 7.0, models, rigs })
}

/// Corner heightmap of the nav terrain's ground (bed under water), with
/// world heights in metres × `relief`; and the water surface height.
fn nav_heightmap(terrain: &Terrain, relief: f32) -> (Heightmap, f32, f32, Option<f32>) {
    let cells = (terrain.geom.cells_x as usize, terrain.geom.cells_y as usize);
    let (heightmap, scale, offset) = Heightmap::from_cells(cells, ground::cell_ground(terrain, relief), None);
    let water = (terrain.water_level != NO_WATER).then_some(terrain.water_level as f32 * 0.1 * relief);
    (heightmap, scale, offset, water)
}

/// The map's terrain for render, and the nav terrain it shows (the
/// crossing maps).
fn terrain_source(o: &Options) -> Result<(TerrainSource, Option<Arc<Terrain>>)> {
    let started = Instant::now();
    let mut materials = None;
    let mut nav = None;
    let (heightmap, scale, offset, water, cell_size) = match o.map {
        MapKind::Water | MapKind::Land | MapKind::Package => {
            let terrain = match (o.map, &o.loaded_map) {
                (MapKind::Package, Some((t, _))) => (**t).clone(),
                (MapKind::Water, _) => ae_nav::scenario::water_crossing_terrain(o.map_cells),
                _ => ae_nav::scenario::land_crossing_terrain(o.map_cells),
            };
            let cell = terrain.geom.cell_size;
            materials = Some(terrain.material.clone());
            let (heightmap, scale, offset, water) = nav_heightmap(&terrain, o.relief);
            nav = Some(Arc::new(terrain));
            (heightmap, scale, offset, water, cell)
        }
        MapKind::Generated => {
            let side = (o.map_cells.max(2) as usize) | 1;
            (generate(side, side, 1), 3000.0 * o.relief, 0.0, None, 8.0)
        }
        MapKind::Demo => unreachable!(),
    };
    let generated = started.elapsed();
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let info = TerrainInfo::new(&heightmap, &pyramid, cell_size, scale, offset, water);
    eprintln!(
        "terrain {}x{} cells ({:.2} s), heights {:.0}..{:.0} m{}, error pyramid {:.2} s",
        info.cells.0,
        info.cells.1,
        generated.as_secs_f32(),
        offset,
        offset + scale,
        water.map(|w| format!(", water at {w:.0} m")).unwrap_or_default(),
        (started.elapsed() - generated).as_secs_f32()
    );
    Ok((TerrainSource { heightmap, pyramid, info, materials }, nav))
}

/// The material tileset of `--tileset`, generated (procedural layers).
fn tileset(o: &Options) -> Result<Option<Arc<Tileset>>> {
    let Some(name) = &o.tileset else { return Ok(None) };
    anyhow::ensure!(o.tileset_size.is_power_of_two() && o.tileset_size >= 4, "--tileset-size must be a power of two of at least 4");
    let started = Instant::now();
    let t = tileset::by_name(name, o.tileset_size).context("unknown tileset")?;
    eprintln!(
        "tileset {}: {} layers of {}², {:.1} MB with mips, generated in {:.2} s",
        t.name,
        t.layers.len(),
        t.size,
        t.bytes() as f64 / (1 << 20) as f64,
        started.elapsed().as_secs_f32()
    );
    Ok(Some(Arc::new(t)))
}

/// A particle feed and setup with the engine's effects (`--no-particles`: none;
/// `--particles-full`: drawn at full resolution).
fn particles(o: &Options) -> Option<(Arc<ParticleFeed>, ParticlesSetup)> {
    o.particles.then(|| {
        let feed = Arc::new(ParticleFeed::default());
        let mut setup = ParticlesSetup::new(feed.clone(), ae_render::particles::MAX_CAPACITY);
        if o.particles_full {
            setup.downscale = 1;
        }
        setup.blend = o.particle_blend;
        (feed, setup)
    })
}

/// Most scorch marks at once.
const DECALS: u32 = 1024;

/// Where explosions go (particles and scorch marks, as the options turn
/// them on), with `app` set up to draw them; `None` when both are off.
fn effects(app: TerrainApp, o: &Options) -> (TerrainApp, Option<Arc<sim::Effects>>) {
    let mut app = app;
    let mut effects = sim::Effects::default();
    if let Some((feed, setup)) = particles(o) {
        app = app.with_particles(setup);
        effects.particles = Some(feed);
    }
    if o.decals {
        let feed = Arc::new(ae_render::decals::DecalFeed::default());
        let mut setup = ae_render::decals::DecalsSetup::new(feed.clone(), DECALS);
        let dir = match o.scars.as_deref() {
            Some("generated") => None,
            Some(dir) => Some(dir),
            None => Some(BUNDLED_SCARS),
        };
        if let Some(dir) = dir {
            match recoil_scars(std::path::Path::new(dir)) {
                Ok((atlas, scars)) => {
                    setup.atlas = Some(Arc::new(atlas));
                    effects.scars = scars;
                }
                Err(e) => eprintln!("scars from {dir}: {e:#}; using the generated scorch mark"),
            }
        }
        app = app.with_decals(setup);
        effects.decals = Some(feed);
    }
    let any = effects.particles.is_some() || effects.decals.is_some();
    (app, any.then(|| Arc::new(effects)))
}

/// Recoil's scars, in the repository (`assets/decals/scars`).
const BUNDLED_SCARS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/decals/scars");

/// The generated atlas at 512² with the scars in `dir` added, and the
/// scars' layers, blended as Recoil blends explosion scars (modulate).
/// Every `NAME.tga` with a matching `NAME_normal.tga` is one scar; the
/// normal map's green handedness is detected (`normal_green_up`), so both
/// DirectX-style and OpenGL-style maps load.
fn recoil_scars(dir: &std::path::Path) -> Result<(ae_render::decals::DecalAtlas, Vec<(u32, ae_render::decals::Blend)>)> {
    use ae_render::decals::{Blend, default_atlas, normal_green_up, read_tga};
    let read = |name: &str| -> Result<ae_render::decals::Image> {
        let path = dir.join(name);
        read_tga(&std::fs::read(&path).with_context(|| path.display().to_string())?).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
    };
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tga") && !n.ends_with("_normal.tga"))
        .collect();
    names.sort();
    if names.is_empty() {
        bail!("no scar colour maps (*.tga with a matching *_normal.tga) in {}", dir.display());
    }
    let mut atlas = default_atlas(512);
    let mut scars = Vec::new();
    for name in &names {
        let stem = name.strip_suffix(".tga").unwrap_or(name);
        let colour = read(name)?;
        let normal = read(&format!("{stem}_normal.tga"))?;
        let up = normal_green_up(&colour, &normal);
        let layer = atlas.add_layer(&colour, Some((&normal, up))).map_err(anyhow::Error::msg)?;
        scars.push((layer, Blend::Modulate));
    }
    Ok((atlas, scars))
}

/// `count` explosions scattered around `target` (seeded; `--explode`, X),
/// each only where the observer sees (as deaths): how many went off.
fn explode_around(effects: &sim::Effects, target: Vec2, count: u32, seed: u32) -> u32 {
    let mut seen = 0;
    for k in 0..count {
        let h = (seed ^ k.wrapping_mul(0x9e37_79b9)).wrapping_mul(0x85eb_ca6b);
        let spread = if count > 1 { 120.0 } else { 0.0 };
        let (dx, dz) = (((h & 0xffff) as f32 / 65535.0 - 0.5) * spread, ((h >> 16) as f32 / 65535.0 - 0.5) * spread);
        let scale = 0.8 + (h % 7) as f32 * 0.2;
        seen += u32::from(effects.explode_if_seen(target.x + dx, target.y + dz, scale, h));
    }
    seen
}

/// Apply the tileset options to a new app.
fn with_look(mut app: TerrainApp, tileset: Option<Arc<Tileset>>, o: &Options) -> TerrainApp {
    if let Some(t) = tileset {
        app = app.with_tileset(t);
    }
    if o.water {
        app = app.with_water();
    }
    if o.shadows {
        app = app.with_shadows();
    }
    app = app.with_tone(o.tone).with_working_space(o.working_space);
    if let Some(gamut) = o.display_gamut {
        app = app.with_display_gamut(gamut);
    }
    app.view().lock().unwrap().weights = o.weights;
    app
}

/// Resize, minimize, restore on a fixed schedule (seconds after start).
fn exercise(window: &mut SdlWindow, started: Instant, step: &mut usize) {
    const STEPS: [f32; 6] = [0.8, 1.6, 2.4, 3.0, 3.6, 4.2];
    let at = STEPS[*step % STEPS.len()];
    let cycle = (*step / STEPS.len()) as f32 * 4.8;
    if started.elapsed().as_secs_f32() < cycle + at {
        return;
    }
    match *step % STEPS.len() {
        0 => window.set_size(800, 600),
        1 => window.set_size(1600, 900),
        2 => window.minimize(),
        3 => window.restore(),
        4 => window.set_size(640, 360),
        _ => window.set_size(1280, 720),
    }
    *step += 1;
}

/// Camera and switches from this frame's input.
fn steer(window: &SdlWindow, events: &ae_render::window::WindowEvents, view: &mut TerrainView, source: &TerrainSource, dt: f32, sim_speed: &sim::SimSpeed) {
    let held = window.held();
    let camera = &mut view.camera;
    let speed = camera.distance * if held.fast { 2.5 } else { 0.9 } * dt;
    let (forward, right) = camera.ground_axes();
    let mut pan = Vec2::ZERO;
    pan += forward * (held.forward as i32 - held.back as i32) as f32;
    pan += right * (held.right as i32 - held.left as i32) as f32;
    let (w, h) = source.info.extent();
    camera.target = (camera.target + pan * speed).clamp(Vec2::ZERO, Vec2::new(w, h));
    camera.yaw += (held.orbit_right as i32 - held.orbit_left as i32) as f32 * 1.5 * dt - events.drag.0 * 0.005;
    camera.pitch = (camera.pitch + (held.tilt_up as i32 - held.tilt_down as i32) as f32 * 1.0 * dt + events.drag.1 * 0.004).clamp(0.08, 1.5);
    camera.distance = (camera.distance * 0.88f32.powf(events.wheel)).clamp(20.0, w.max(h) * 2.0);
    if events.pressed & keys::PLUS != 0 {
        sim_speed.faster();
        println!("sim speed {}", sim_speed.label());
    }
    if events.pressed & keys::MINUS != 0 {
        sim_speed.slower();
        println!("sim speed {}", sim_speed.label());
    }
    if events.pressed & keys::LBRACKET != 0 {
        view.tau = (view.tau / 1.5).max(0.25);
    }
    if events.pressed & keys::RBRACKET != 0 {
        view.tau = (view.tau * 1.5).min(64.0);
    }
    if events.pressed & keys::T != 0 {
        view.wireframe = if view.wireframe == Wireframe::Off { Wireframe::Triangles } else { Wireframe::Off };
    }
    if events.pressed & keys::L != 0 {
        view.lod_colors = !view.lod_colors;
    }
    if events.pressed & keys::G != 0 {
        view.fog = !view.fog;
    }
    if events.pressed & keys::C != 0 {
        view.frozen = match view.frozen {
            Some(_) => None,
            None => {
                let (pw, ph) = window.pixel_size();
                Some(lod_view(source, &view.camera, pw.max(1), ph.max(1)))
            }
        };
    }
    if events.pressed & keys::N != 0 {
        view.module_flags ^= 1;
    }
    if events.pressed & keys::M != 0 {
        view.tileset = !view.tileset;
    }
    if events.pressed & keys::B != 0 {
        view.weights = !view.weights;
    }
    if events.pressed & keys::O != 0 {
        view.water = !view.water;
    }
    if events.pressed & keys::H != 0 {
        view.shadows = !view.shadows;
    }
    if events.pressed & keys::Z != 0 {
        view.bloom = !view.bloom;
        println!("bloom {}", if view.bloom { "on" } else { "off" });
    }
    if events.pressed & keys::P != 0 {
        let c = &view.camera;
        println!("--camera {:.0},{:.0},{:.0},{:.3},{:.3}  (error target {:.2} px)", c.target.x, c.target.y, c.distance, c.yaw, c.pitch, view.tau);
    }
}

/// Apply `--camera`.
fn place_camera(app: &TerrainApp, camera: Option<[f32; 5]>) {
    if let Some([x, z, distance, yaw, pitch]) = camera {
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera.target = Vec2::new(x, z);
        view.camera.distance = distance;
        view.camera.yaw = yaw;
        view.camera.pitch = pitch;
    }
}

/// `--lights N`: N fixed test lights around the camera's target (placed
/// first), in a capture and in the window alike.
fn scatter_lights(app: TerrainApp, n: u32) -> TerrainApp {
    if n == 0 {
        return app;
    }
    let target = app.view().lock().unwrap().camera.target;
    let source = app.source().clone();
    app.with_lights(ae_render::lights::scatter(n, target, 400.0, 4.0, 7, &|p| source.height_at(p)))
}

fn capture(source: Arc<TerrainSource>, nav: Option<Arc<Terrain>>, o: &Options, path: &str) -> Result<ExitCode> {
    use ae_render::device::{GpuDevice, GpuInstance};
    use ae_render::executor::ExecutorConfig;
    use ae_render::thread::{RenderApp, TargetInfo, run_headless};
    let instance = Arc::new(GpuInstance::new(&[], None)?);
    let gpu = Arc::new(GpuDevice::new(instance.clone(), None)?);
    let app = with_look(TerrainApp::new(source.clone(), TerrainOutput::Readback), tileset(o)?, o);
    let (mut app, effects_feed) = effects(app, o);
    if let Some(name) = &o.render_module {
        app = app.with_module(ae_render::module::build(name)?);
    }
    let mut sim_note = String::new();
    if let Some(s) = sim_options(o) {
        let started = Instant::now();
        let ground = Arc::new(GroundFeed::default());
        let mut sim = sim::Sim::new(&s, Some(ground.clone()))?;
        if s.observer.is_some() {
            let vision = Arc::new(VisionFeed::new(s.vision_grid()));
            sim = sim.with_vision(vision.clone());
            app = app.with_vision(vision);
        }
        let built = started.elapsed();
        for _ in 0..o.ticks {
            sim.tick()?;
        }
        // `--explode` only where the observer sees, as deaths.
        if let (Some(feed), Some((los, grid))) = (&effects_feed, sim.sight()) {
            feed.set_sight(los, grid);
        }
        let ticked = started.elapsed() - built;
        if let (Some(update), Some(mut shown)) = (ground.take(), nav) {
            let t = Instant::now();
            let mut editor = TerrainEditor::new(&source);
            let made = Instant::now();
            if let Some((u, tiles)) = ground::edit(&mut editor, &mut shown, &update, o.relief) {
                eprintln!(
                    "terrain version {}: {tiles} changed tiles in cells {:?}, {} bytes to upload, made in {:.1} ms (+ {:.1} ms copying the source)",
                    u.version,
                    update.cells,
                    u.bytes(),
                    made.elapsed().as_secs_f32() * 1e3,
                    (made - t).as_secs_f32() * 1e3
                );
                app.terrain_edits().lock().unwrap().pending.push(u);
            }
        }
        let t = Instant::now();
        let stream = sim.stream(0.0)?;
        let streamed = t.elapsed();
        if let (Some([own, seen, blip, ghost]), Some([c, b, r])) = (sim.view_counts(), sim.view_ms()) {
            eprintln!("observer {}: {own} own, {seen} seen, {blip} blips, {ghost} ghosts; capture + coverage {c:.3} ms, view {b:.3} ms, rows + stream {r:.3} ms", s.observer.unwrap_or(0));
        }
        // Where each class is (to aim `--camera` at).
        let mut classes: Vec<(u32, f64, f64)> = vec![(0, 0.0, 0.0); s.classes().len()];
        for r in 0..stream.len() {
            if let Some(c) = classes.get_mut(stream.columns[7][r] as usize) {
                *c = (c.0 + 1, c.1 + f32::from_bits(stream.columns[0][r]) as f64, c.2 + f32::from_bits(stream.columns[1][r]) as f64);
            }
        }
        for (i, (n, x, y)) in classes.iter().enumerate().filter(|c| c.1.0 > 0) {
            eprintln!("class {i}: {n} units, centre {:.0},{:.0}, {} moving", x / *n as f64, y / *n as f64, sim.moving(i as u32));
        }
        sim_note = format!(
            "; sim: {} units at tick {}, world {:.2} s, {:.2} ms/tick, capture + stream {:.3} ms",
            stream.len(),
            stream.tick,
            built.as_secs_f32(),
            ticked.as_secs_f64() * 1e3 / o.ticks.max(1) as f64,
            streamed.as_secs_f64() * 1e3
        );
        let feed = Arc::new(UnitFeed::default());
        feed.publish(Arc::new(stream));
        app = app.with_units(units_setup(&s, feed)?);
    }
    place_camera(&app, o.camera);
    app = scatter_lights(app, o.lights);
    if let Some(feed) = &effects_feed {
        let target = app.view().lock().unwrap().camera.target;
        let seen = explode_around(feed, target, o.explode, 1);
        if seen < o.explode {
            eprintln!("--explode: {} of {} out of sight", o.explode - seen, o.explode);
        }
    }
    let extent = ash::vk::Extent2D { width: 1280, height: 720 };
    app.create_offscreen(&gpu, extent)?;
    let target = TargetInfo { extent, format: ash::vk::Format::R8G8B8A8_SRGB, color_space: ash::vk::ColorSpaceKHR::SRGB_NONLINEAR };
    let executor = run_headless(&gpu, &mut app, target, o.capture_frames as usize, ExecutorConfig { frames_in_flight: o.frames_in_flight, profile: true, record_threads: o.record_threads, ..ExecutorConfig::default() })?;
    let pixels = app.readback().context("no readback")?;
    let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
    ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
    std::fs::write(path, ppm).with_context(|| format!("writing {path}"))?;
    let stats = *app.stats().lock().unwrap();
    let counts = app.last_unit_counts();
    {
        let e = app.terrain_edits();
        let e = e.lock().unwrap();
        if e.applied > 0 {
            eprintln!("terrain edits: {} applied (version {}), {} bytes uploaded in {:.2} ms", e.applied, e.version, e.last_bytes, e.last_ms);
        }
    }
    RenderApp::destroy(&mut app, &gpu);
    drop(gpu);
    let (errors, warnings) = (instance.counters.errors(), instance.counters.warnings());
    let gpu_ms: Vec<String> = executor.pass_names.iter().zip(&executor.pass_gpu_ms).map(|(name, ms)| format!("{name} {ms:.3}")).collect();
    println!(
        "wrote {path}: {} leaves, {} drawn{}; GPU ms per pass: {}{sim_note}{}; {}",
        stats.leaves,
        stats.drawn,
        counts.map(|(rows, visible)| format!(", {visible} of {rows} units visible")).unwrap_or_default(),
        gpu_ms.join(", "),
        if o.record_threads > 1 { format!("; {} of {} frames recorded on {} threads", executor.parallel_frames, executor.frames, o.record_threads) } else { String::new() },
        validation_text(instance.validation, errors, warnings)
    );
    Ok(if errors + warnings > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS })
}

fn run(o: Options) -> Result<ExitCode> {
    if o.probe_display {
        // A window is needed for a surface, and on Wayland the compositor
        // describes it only once it is mapped; nothing is rendered.
        let mut window = SdlWindow::new("AssembleEngine display probe", 320, 200)?;
        ae_render::display::probe(&mut window)?;
        return Ok(ExitCode::SUCCESS);
    }
    if let Some(b) = &o.bench {
        return bench::run(&bench::BenchOptions { record_threads: o.record_threads, particles_full: o.particles_full, particle_blend: o.particle_blend, ..b.clone() });
    }
    let (source, nav) = match o.map {
        MapKind::Demo => (None, None),
        _ => {
            let (source, nav) = terrain_source(&o)?;
            (Some(Arc::new(source)), nav)
        }
    };
    if let Some(path) = &o.capture {
        return capture(source.context("--capture needs a terrain map")?, nav, &o, path);
    }
    let mut window = SdlWindow::new("AssembleEngine", 1280, 720)?;
    // SDL owns the window's HDR state and delivers it once the window is
    // mapped; read it before the swapchain is created (M13 stage 5).
    let display_state = ae_render::display::settled_hdr_state(&mut window);
    let mut hdr_capable = display_state.enabled && display_state.headroom > 1.0 && display_state.sdr_white_level > 0.0;
    let mut headroom = if hdr_capable { display_state.headroom } else { 1.0 };
    let mut display_peak = if hdr_capable { display_state.peak_nits() } else { 0.0 };
    let mut hdr_request = o.hdr;
    // A module graph has no HDR support until stage 6: HDR stays off for
    // the whole run, including the toggle key.
    let module_blocks_hdr = o.render_module.is_some();
    if module_blocks_hdr && hdr_request != HdrRequest::Off {
        // M13 stage 6 adds HDR to the module ABI; until then a module graph
        // samples and writes an SDR target (and check_graph would reject an
        // HDR one), so fall back once, clearly.
        eprintln!("--render-module {} with --hdr {}: render modules have no HDR support until M13 stage 6; falling back to SDR", o.render_module.as_deref().unwrap_or(""), match hdr_request { HdrRequest::Auto => "auto", HdrRequest::Hdr10 => "hdr10", HdrRequest::ScRgb => "scrgb", HdrRequest::Off => "off" });
        hdr_request = HdrRequest::Off;
    }
    let hdr_settings = Arc::new(Mutex::new(HdrSettings { paper_white: o.paper_white, peak: o.peak, headroom, display_peak }));
    println!(
        "[hdr] --hdr {} | SDL HDR {} | SDR white {:.1} nits | headroom {:.3}x | paper white {} | peak {}",
        match hdr_request { HdrRequest::Off => "off", HdrRequest::Auto => "auto", HdrRequest::Hdr10 => "hdr10", HdrRequest::ScRgb => "scrgb" },
        if display_state.enabled { "enabled" } else { "off" },
        display_state.paper_white_nits(),
        headroom,
        o.paper_white.map(|n| format!("{n:.1} nits (--paper-white)")).unwrap_or_else(|| "the WSI's reference".into()),
        o.peak.map(|n| format!("{n:.1} nits (--peak)")).unwrap_or_else(|| format!("paper white × headroom (the panel's {display_peak:.1} nits)")),
    );
    let config = GpuRenderConfig { frames_in_flight: o.frames_in_flight, present: o.present, max_fps: 0, profile: false, verbose: false, record_threads: o.record_threads, hdr: HdrControl { request: hdr_request, capable: hdr_capable, headroom } };
    type Shared = (Arc<Mutex<TerrainView>>, Arc<Mutex<FrameStats>>);
    let mut terrain: Option<Shared> = None;
    let mut sim_thread = None;
    let mut ground_thread = None;
    let mut effects_feed: Option<Arc<sim::Effects>> = None;
    let mut edits = None;
    let mut module: Option<(watch::ModuleWatch, Arc<Mutex<ae_render::terrain::app::ModuleStatus>>)> = None;
    let ui: sim::UiQueue = Arc::default();
    let sim_speed = sim::SimSpeed::default();
    let mut vsync = o.present == PresentPolicy::Fifo;
    let mut thread = match &source {
        None => GpuRenderThread::start(&window, DemoApp::default(), config)?,
        Some(source) => {
            let (mut app, feed) = effects(with_look(TerrainApp::new(source.clone(), TerrainOutput::Present), tileset(&o)?, &o), &o);
            app = app.with_hdr(hdr_settings.clone());
            effects_feed = feed;
            if let Some(s) = sim_options(&o) {
                let feed = Arc::new(UnitFeed::default());
                app = app.with_units(units_setup(&s, feed.clone())?);
                let vision = s.observer.map(|_| Arc::new(VisionFeed::new(s.vision_grid())));
                if let Some(v) = &vision {
                    app = app.with_vision(v.clone());
                }
                let ground = Arc::new(GroundFeed::default());
                let shown = nav.clone().context("units need a crossing map")?;
                ground_thread = Some(ground::GroundThread::start(ground.clone(), source.clone(), shown, o.relief, app.terrain_edits()));
                edits = Some(app.terrain_edits());
                sim_thread = Some(sim::SimThread::start(s, feed, ground, vision, effects_feed.clone(), ui.clone(), sim_speed.clone()));
            }
            if let Some(name) = &o.render_module {
                let wasm = ae_render::module::build(name)?;
                app = app.with_module(wasm.clone());
                module = Some((watch::ModuleWatch::start(name.clone(), wasm, app.module_reload()), app.module_status()));
            }
            place_camera(&app, o.camera);
            app = scatter_lights(app, o.lights);
            terrain = Some((app.view(), app.stats()));
            GpuRenderThread::start(&window, app, config)?
        }
    };
    let label = match o.map {
        MapKind::Water => "water crossing",
        MapKind::Land => "land crossing",
        MapKind::Generated => "generated",
        MapKind::Demo => "demo",
        MapKind::Package => "package map",
    };
    let started = Instant::now();
    let mut last = Instant::now();
    let mut sample = (Instant::now(), 0u64);
    let mut step = 0;
    let mut last_boom = Instant::now();
    let mut module_errors = 0;
    // A test hook for the runtime HDR toggle (the U key path), since a
    // Wayland session has no synthetic keyboard here: flip every N seconds.
    let hdr_toggle_seconds: Option<f32> = std::env::var("AE_HDR_TOGGLE_SECONDS").ok().and_then(|s| s.parse().ok());
    let mut next_hdr_toggle = 0.0f32;
    // SDL's HDR properties arrive with the compositor's surface description
    // after the event: read them until the SDR white level is there.
    let mut hdr_settle: Option<Instant> = None;
    loop {
        let events = window.pump();
        if events.pressed & keys::PAUSE != 0 {
            let paused = sim_speed.toggle_pause();
            if let Some((view, _)) = &terrain {
                view.lock().unwrap_or_else(|p| p.into_inner()).paused = paused;
            }
            println!("simulation {}", if paused { "paused" } else { "running" });
        }
        let hdr_toggle_due = hdr_toggle_seconds.is_some_and(|_| started.elapsed().as_secs_f32() >= next_hdr_toggle);
        if hdr_toggle_due {
            next_hdr_toggle += hdr_toggle_seconds.unwrap_or(1.0);
        }
        if (events.pressed & keys::U != 0 || hdr_toggle_due) && module_blocks_hdr {
            println!("HDR stays off: render modules have no HDR support until M13 stage 6");
        } else if events.pressed & keys::U != 0 || hdr_toggle_due {
            hdr_request = if hdr_request == HdrRequest::Off { o.hdr.toggled_on() } else { HdrRequest::Off };
            println!("HDR {}", if hdr_request == HdrRequest::Off { "off (SDR)" } else { match hdr_request { HdrRequest::Hdr10 => "hdr10", HdrRequest::ScRgb => "scrgb", _ => "auto" } });
            thread.set_hdr(HdrControl { request: hdr_request, capable: hdr_capable, headroom });
        }
        if events.hdr_changed || events.display_changed {
            // The compositor's HDR switch or a display change: re-read SDL
            // and recreate the swapchain for the new state.
            hdr_settle = Some(Instant::now());
        }
        if let Some(since) = hdr_settle {
            let state = ae_render::display::hdr_state(&window);
            if state.sdr_white_level > 0.0 || since.elapsed().as_millis() > 800 {
                hdr_capable = state.enabled && state.headroom > 1.0 && state.sdr_white_level > 0.0;
                headroom = if hdr_capable { state.headroom } else { 1.0 };
                display_peak = if hdr_capable { state.peak_nits() } else { 0.0 };
                if let Ok(mut settings) = hdr_settings.lock() {
                    settings.headroom = headroom;
                    settings.display_peak = display_peak;
                }
                println!(
                    "[hdr] window state changed: HDR {} | SDR white {:.1} nits | headroom {:.3}x | peak {:.1} nits | {}",
                    if state.enabled { "enabled" } else { "off" },
                    state.paper_white_nits(),
                    headroom,
                    display_peak,
                    if hdr_capable { "capable" } else { "SDR only" }
                );
                thread.set_hdr(HdrControl { request: hdr_request, capable: hdr_capable, headroom });
                hdr_settle = None;
            }
        }
        if events.pressed & keys::V != 0 {
            vsync = !vsync;
            thread.set_present(if vsync { PresentPolicy::Fifo } else { PresentPolicy::Mailbox });
            println!("vsync {}", if vsync { "on" } else { "off" });
        }
        // Clicks: left selects own units near the ground point, right
        // orders the selection there (ctrl: attack-move, shift: queued).
        if let (Some((view, _)), Some(source)) = (&terrain, &source) {
            let view = *view.lock().unwrap_or_else(|p| p.into_inner());
            for click in events.clicks.iter().flatten() {
                let Some(at) = ground_at(&view, source, window.pixel_size(), click.x, click.y) else { continue };
                let input = if click.button == ae_render::window::Click::LEFT { sim::UiInput::Select { x: at.x, y: at.y, radius: 120.0 } } else { sim::UiInput::Move { x: at.x, y: at.y, attack: click.ctrl, append: click.shift } };
                ui.lock().unwrap_or_else(|p| p.into_inner()).push(input);
            }
        }
        thread.handle_events(&events);
        if events.quit || !thread.running() || o.seconds.is_some_and(|s| started.elapsed().as_secs_f32() >= s) {
            break;
        }
        let dt = last.elapsed().as_secs_f32().min(0.1);
        last = Instant::now();
        if let (Some((view, _)), Some(source)) = (&terrain, &source) {
            let mut view = view.lock().unwrap_or_else(|p| p.into_inner());
            steer(&window, &events, &mut view, source, dt, &sim_speed);
            if let Some(feed) = effects_feed.as_ref().filter(|_| events.pressed & keys::X != 0) {
                if explode_around(feed, view.camera.target, 1, started.elapsed().as_nanos() as u32) == 0 {
                    println!("X: the camera target is out of sight; no explosion");
                }
            }
            // The exercise keeps particles alive through its resizes.
            if let Some(feed) = effects_feed.as_ref().filter(|_| o.exercise && !sim_speed.paused() && last_boom.elapsed().as_secs_f32() >= 1.0) {
                explode_around(feed, view.camera.target, 3, started.elapsed().as_nanos() as u32);
                last_boom = Instant::now();
            }
        }
        if o.exercise {
            exercise(&mut window, started, &mut step);
        }
        let elapsed = sample.0.elapsed().as_secs_f64();
        if elapsed >= 0.5 {
            let render = thread.stats();
            let frames = render.frames;
            let fps = frames.saturating_sub(sample.1) as f64 / elapsed;
            let detail = match &terrain {
                Some((view, stats)) => {
                    let s = *stats.lock().unwrap_or_else(|p| p.into_inner());
                    let tau = view.lock().unwrap_or_else(|p| p.into_inner()).tau;
                    format!(" | {} leaves, {} drawn, {tau:.2} px", s.leaves, s.drawn)
                }
                None => String::new(),
            };
            let sim = match &sim_thread {
                Some(t) => {
                    let s = t.stats();
                    if s.failed {
                        " | sim failed (see the terminal)".into()
                    } else if s.ready {
                        let view = s.view.map(|[own, seen, blip, _]| format!(", observer {}: {own} own, {seen} seen, {blip} blips", o.observer.unwrap_or(0))).unwrap_or_default();
                        format!(" | {} units, tick {}, {:.1} ms/tick, {:.1}x({}) speed{view}", s.units, s.tick, s.tick_ms, s.actual_speed, sim_speed.label())
                    } else {
                        " | building the world".into()
                    }
                }
                None => String::new(),
            };
            let terrain_version = edits.as_ref().map(|e| e.lock().unwrap_or_else(|p| p.into_inner()).version).filter(|&v| v > 0);
            let detail = detail + &sim + &terrain_version.map(|v| format!(" | terrain v{v}")).unwrap_or_default();
            let detail = match &module {
                Some((_, status)) => {
                    let s = status.lock().unwrap_or_else(|p| p.into_inner()).clone();
                    for error in &s.errors[module_errors.min(s.errors.len())..] {
                        eprintln!("{error}");
                    }
                    module_errors = s.errors.len();
                    format!("{detail} | module {} (loads {})", o.render_module.as_deref().unwrap_or(""), s.loads)
                }
                None => detail,
            };
            let detail = detail + if sim_speed.paused() { " | paused" } else { "" };
            let detail = detail + if render.present_mode == "FIFO" { " | vsync on" } else { " | vsync off" };
            window.set_title(&format!("AssembleEngine | {label} | {fps:.0} fps{detail}"));
            sample = (Instant::now(), frames);
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let stats = thread.stop();
    drop(ground_thread);
    if let Some(e) = &edits {
        let e = e.lock().unwrap_or_else(|p| p.into_inner());
        if e.applied > 0 {
            println!("terrain edits: {} applied (version {}), last {} bytes uploaded in {:.2} ms", e.applied, e.version, e.last_bytes, e.last_ms);
        }
    }
    if let Some((_, status)) = &module {
        let s = status.lock().unwrap_or_else(|p| p.into_inner()).clone();
        for error in &s.errors[module_errors.min(s.errors.len())..] {
            eprintln!("{error}");
        }
        println!("render module: {} loads, {} errors", s.loads, s.errors.len());
    }
    if let Some(t) = sim_thread {
        let s = t.stop().context("sim thread")?;
        if s.ready {
            println!(
                "sim: {} units, tick {}, last second: {:.2} ms/tick, capture + stream {:.3} ms/tick; {} ticks ran late",
                s.units, s.tick, s.tick_ms, s.capture_ms, s.late
            );
        }
    }
    if o.record_threads > 1 {
        println!("{} of {} frames recorded on {} threads", stats.executor.parallel_frames, stats.executor.frames, o.record_threads);
    }
    println!(
        "{} frames in {:.1} s on {} ({}), {} swapchain recreates, {} graph bakes, {} acquire timeouts, {}",
        stats.frames,
        started.elapsed().as_secs_f32(),
        stats.device,
        stats.present_mode,
        stats.swapchain_recreates,
        stats.executor.bakes,
        stats.acquire_timeouts,
        validation_text(stats.validation, stats.validation_errors, stats.validation_warnings)
    );
    if let Some(error) = stats.error {
        bail!("render thread: {error}");
    }
    Ok(if stats.validation_errors + stats.validation_warnings > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS })
}

/// The terrain point under a click (`x`, `y` as fractions of the window):
/// the camera ray through it, marched over the heightmap, the crossing
/// bisected. `None` if the ray misses the map.
fn ground_at(view: &TerrainView, source: &TerrainSource, (w, h): (u32, u32), x: f32, y: f32) -> Option<Vec2> {
    let mut camera = view.camera;
    camera.ground = source.height_at(camera.target);
    let eye = camera.eye(|p| source.height_at(p));
    let inverse = (camera.proj(w as f32 / h.max(1) as f32) * camera.view(eye)).inverse();
    let at = |z: f32| {
        let p = inverse * glam::Vec4::new(x * 2.0 - 1.0, y * 2.0 - 1.0, z, 1.0);
        p.truncate() / p.w
    };
    // Reverse-Z: depth 1 is the near plane.
    let (near, mid) = (at(1.0), at(0.5));
    let dir = (mid - near).normalize();
    let (ew, eh) = source.info.extent();
    let above = |p: glam::Vec3| p.y > source.height_at(Vec2::new(p.x, p.z));
    let (mut t, mut prev) = (0.0f32, 0.0f32);
    while t < 4.0 * ew.max(eh) {
        let p = near + dir * t;
        if !above(p) {
            let (mut lo, mut hi) = (prev, t);
            for _ in 0..24 {
                let m = 0.5 * (lo + hi);
                if above(near + dir * m) { lo = m } else { hi = m }
            }
            let hit = near + dir * hi;
            return (hit.x >= 0.0 && hit.z >= 0.0 && hit.x <= ew && hit.z <= eh).then_some(Vec2::new(hit.x, hit.z));
        }
        prev = t;
        t += (t * 0.01).max(2.0);
    }
    None
}

fn validation_text(on: bool, errors: u64, warnings: u64) -> String {
    if on { format!("validation: {errors} errors, {warnings} warnings") } else { "validation off (AE_VK_VALIDATION=1 turns it on)".into() }
}

fn main() -> ExitCode {
    match parse().and_then(run) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Corners average the ground (surface minus water depth) of the cells
    /// around them; heights are decimetres → metres; the water level too.
    #[test]
    fn nav_terrain_becomes_a_corner_heightmap() {
        let terrain = ae_nav::scenario::water_crossing_terrain(256);
        let (cx, cy) = (terrain.geom.cells_x as usize, terrain.geom.cells_y as usize);
        let (hm, scale, offset, water) = nav_heightmap(&terrain, 2.0);
        assert_eq!((hm.width, hm.height), (cx + 1, cy + 1));
        let ground = |x: usize, y: usize| {
            let i = y * cx + x;
            (terrain.height[i] - terrain.water_depth.get(i).copied().unwrap_or(0)) as f32 * 0.2
        };
        let world = |x: usize, y: usize| hm.data[y * hm.width + x] * scale + offset;
        for (x, y) in [(1, 1), (100, 57), (cx - 1, cy - 1)] {
            let want = (ground(x - 1, y - 1) + ground(x, y - 1) + ground(x - 1, y) + ground(x, y)) / 4.0;
            assert!((world(x, y) - want).abs() < 1e-3 * scale.max(1.0), "corner ({x}, {y})");
        }
        assert!((world(0, 0) - ground(0, 0)).abs() < 1e-3 * scale.max(1.0), "a map corner has one cell");
        assert_eq!(water, Some(terrain.water_level as f32 * 0.2));
        assert!(terrain.water_depth.iter().any(|&d| d > 0), "the map has water to subtract");
    }
}
