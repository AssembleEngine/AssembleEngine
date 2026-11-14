//! Windowed client.
//!
//! ```text
//! ae-client [--map water|land|generated|demo] [--map-cells N] [--relief X]
//!           [--camera X,Z,DIST,YAW,PITCH] [--seconds S] [--exercise] [--fifo]
//!           [--frames-in-flight N] [--units N] [--workers N] [--ticks N]
//!           [--render-module NAME|FILE.wasm|FILE.json] [--capture FILE.ppm]
//! ```
//!
//! `--render-module` takes the frame graph from a render module: an
//! in-repo one (`render/<NAME>`: `default`, the engine's own graph from
//! WASM; `post`, with a colour grade and a vignette V toggles), a `.wasm`
//! file, or a graph file (`.json`, e.g. `render/post.graph.json`: the same
//! graph as data, `ae_render::graph_file`). Its sources (or the file) are watched: an edit rebuilds it
//! and the running client swaps it in at the next frame, or keeps the old
//! one and prints why.
//!
//! Shows the terrain of the M2 crossing maps (`water`, the default, or
//! `land`; `--map-cells` as in `ae-headless`, default 2560 = pathsim's full
//! map), map-rendering's generated test map (`generated`), or the frame
//! graph demo (`demo`). The nav heightfield is taken as is: decimetres to
//! world units (1 unit = 1 m) times `--relief` (default 1); ground under
//! water is drawn with the water's depth as a tint.
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
//! Controls: WASD / arrows pan, wheel zooms, right- or middle-drag orbits
//! and tilts, Q/E orbit, R/F tilt, shift faster; `+`/`-` more or less
//! detail, T wireframe, L colour by LOD depth, G fog, C freeze the LOD,
//! P print the camera, V toggle render module flag bit 0, Esc quits.
//!
//! `--capture` renders 1280x720 offscreen (no window; 60 frames so the
//! subdivision settles) and writes a binary PPM; with units it first runs
//! `--ticks` sim ticks (default 300) and shows that tick.
//!
//! `--bench` measures fixed scenes offscreen and writes `ae_bench` reports
//! (see `bench.rs`; `--bench list` names them).
//!
//! `--record-threads N` records the frame's passes on N threads, one
//! command buffer per pass (default 1: all on the render thread).
//!
//! `--seconds` closes the window after S seconds; `--exercise` resizes,
//! minimizes and restores on a schedule. Validation follows
//! `AE_VK_VALIDATION` (default on in debug builds); the exit status is 1 if
//! the layer reported anything.

mod bench;
mod ground;
mod sim;
mod watch;

use ae_nav::terrain::{NO_WATER, Terrain};
use ae_services::nav::{GroundFeed, Map};
use ae_render::demo::DemoApp;
use ae_render::swapchain::PresentPolicy;
use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, TerrainView, lod_view};
use ae_render::terrain::edit::TerrainEditor;
use ae_render::terrain::tileset::{self, Tileset};
use ae_render::terrain::heightmap::{Heightmap, build_pyramid, generate};
use ae_render::terrain::renderer::{FrameStats, TerrainInfo, Wireframe};
use ae_render::thread::{GpuRenderConfig, GpuRenderThread};
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
    /// Material tileset name, `None` for the palette.
    tileset: Option<String>,
    tileset_size: u32,
    /// `--bench`: scenes to measure instead of opening a window.
    bench: Option<bench::BenchOptions>,
    /// Threads recording render passes (1: the render thread alone).
    record_threads: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { map: MapKind::Water, map_cells: 2560, relief: 1.0, camera: None, seconds: None, exercise: false, present: PresentPolicy::Mailbox, frames_in_flight: 2, capture: None, units: 10_000, workers: default_workers(), ticks: 300, render_module: None, quake: 0, tileset: Some("temperate".into()), tileset_size: 512, weights: false, bench: None, record_threads: 1 }
    }
}

const USAGE: &str = "ae-client [--map water|land|generated|demo] [--map-cells N] [--relief X] [--camera X,Z,DIST,YAW,PITCH] [--seconds S] [--exercise] [--fifo] [--frames-in-flight N] [--units N] [--workers N] [--ticks N] [--render-module NAME|FILE.wasm|FILE.json] [--quake TICK] [--tileset temperate|desert|none] [--tileset-size N] [--weights] [--record-threads N] [--capture FILE.ppm]
       ae-client --bench SCENE[,SCENE...]|all|list [--frames N] [--warmup N] [--label NAME] [--out DIR] [--no-save]";

fn parse() -> Result<Options> {
    let mut o = Options::default();
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
            "--quake" => o.quake = value("--quake", &mut args)?.parse().context("--quake TICK (1..=255)")?,
            "--bench" => o.bench.get_or_insert_with(Default::default).target = value("--bench", &mut args)?,
            "--frames" => o.bench.get_or_insert_with(Default::default).frames = value("--frames", &mut args)?.parse().context("--frames")?,
            "--warmup" => o.bench.get_or_insert_with(Default::default).warmup = value("--warmup", &mut args)?.parse().context("--warmup")?,
            "--label" => o.bench.get_or_insert_with(Default::default).label = Some(value("--label", &mut args)?),
            "--out" => o.bench.get_or_insert_with(Default::default).out = value("--out", &mut args)?.into(),
            "--no-save" => o.bench.get_or_insert_with(Default::default).save = false,
            "--record-threads" => o.record_threads = value("--record-threads", &mut args)?.parse::<usize>().context("--record-threads")?.max(1),
            "--fifo" => o.present = PresentPolicy::Fifo,
            "--frames-in-flight" => o.frames_in_flight = value("--frames-in-flight", &mut args)?.parse().context("--frames-in-flight")?,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}\n{USAGE}"),
        }
    }
    if o.bench.as_ref().is_some_and(|b| b.target.is_empty()) {
        bail!("--frames, --warmup, --label, --out and --no-save go with --bench\n{USAGE}");
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
        MapKind::Generated | MapKind::Demo => return None,
    };
    (o.units > 0).then_some(sim::SimOptions { map, map_cells: o.map_cells, units: o.units, workers: o.workers, quake: o.quake })
}

fn units_setup(s: &sim::SimOptions, feed: Arc<UnitFeed>) -> UnitsSetup {
    UnitsSetup { feed, capacity: s.units, classes: s.classes(), min_pixels: 7.0 }
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
        MapKind::Water | MapKind::Land => {
            let terrain = if o.map == MapKind::Water {
                ae_nav::scenario::water_crossing_terrain(o.map_cells)
            } else {
                ae_nav::scenario::land_crossing_terrain(o.map_cells)
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

/// Apply the tileset options to a new app.
fn with_look(mut app: TerrainApp, tileset: Option<Arc<Tileset>>, o: &Options) -> TerrainApp {
    if let Some(t) = tileset {
        app = app.with_tileset(t);
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
fn steer(window: &SdlWindow, events: &ae_render::window::WindowEvents, view: &mut TerrainView, source: &TerrainSource, dt: f32) {
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
        view.tau = (view.tau / 1.5).max(0.25);
    }
    if events.pressed & keys::MINUS != 0 {
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
    if events.pressed & keys::V != 0 {
        view.module_flags ^= 1;
    }
    if events.pressed & keys::M != 0 {
        view.tileset = !view.tileset;
    }
    if events.pressed & keys::B != 0 {
        view.weights = !view.weights;
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

fn capture(source: Arc<TerrainSource>, nav: Option<Arc<Terrain>>, o: &Options, path: &str) -> Result<ExitCode> {
    use ae_render::device::{GpuDevice, GpuInstance};
    use ae_render::executor::ExecutorConfig;
    use ae_render::thread::{RenderApp, TargetInfo, run_headless};
    let instance = Arc::new(GpuInstance::new(&[], None)?);
    let gpu = Arc::new(GpuDevice::new(instance.clone(), None)?);
    let mut app = with_look(TerrainApp::new(source.clone(), TerrainOutput::Readback), tileset(o)?, o);
    if let Some(name) = &o.render_module {
        app = app.with_module(ae_render::module::build(name)?);
    }
    let mut sim_note = String::new();
    if let Some(s) = sim_options(o) {
        let started = Instant::now();
        let ground = Arc::new(GroundFeed::default());
        let mut sim = sim::Sim::new(&s, Some(ground.clone()))?;
        let built = started.elapsed();
        for _ in 0..o.ticks {
            sim.tick()?;
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
        app = app.with_units(units_setup(&s, feed));
    }
    place_camera(&app, o.camera);
    let extent = ash::vk::Extent2D { width: 1280, height: 720 };
    app.create_offscreen(&gpu, extent)?;
    let target = TargetInfo { extent, format: ash::vk::Format::R8G8B8A8_SRGB };
    let executor = run_headless(&gpu, &mut app, target, 60, ExecutorConfig { frames_in_flight: o.frames_in_flight, profile: true, record_threads: o.record_threads, ..ExecutorConfig::default() })?;
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
    if let Some(b) = &o.bench {
        return bench::run(&bench::BenchOptions { record_threads: o.record_threads, ..b.clone() });
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
    let mut window = SdlWindow::new("AssemblyEngine", 1280, 720)?;
    let config = GpuRenderConfig { frames_in_flight: o.frames_in_flight, present: o.present, max_fps: 0, profile: false, verbose: false, record_threads: o.record_threads };
    type Shared = (Arc<Mutex<TerrainView>>, Arc<Mutex<FrameStats>>);
    let mut terrain: Option<Shared> = None;
    let mut sim_thread = None;
    let mut ground_thread = None;
    let mut edits = None;
    let mut module: Option<(watch::ModuleWatch, Arc<Mutex<ae_render::terrain::app::ModuleStatus>>)> = None;
    let mut thread = match &source {
        None => GpuRenderThread::start(&window, DemoApp::default(), config)?,
        Some(source) => {
            let mut app = with_look(TerrainApp::new(source.clone(), TerrainOutput::Present), tileset(&o)?, &o);
            if let Some(s) = sim_options(&o) {
                let feed = Arc::new(UnitFeed::default());
                app = app.with_units(units_setup(&s, feed.clone()));
                let ground = Arc::new(GroundFeed::default());
                let shown = nav.clone().context("units need a crossing map")?;
                ground_thread = Some(ground::GroundThread::start(ground.clone(), source.clone(), shown, o.relief, app.terrain_edits()));
                edits = Some(app.terrain_edits());
                sim_thread = Some(sim::SimThread::start(s, feed, ground));
            }
            if let Some(name) = &o.render_module {
                let wasm = ae_render::module::build(name)?;
                app = app.with_module(wasm.clone());
                module = Some((watch::ModuleWatch::start(name.clone(), wasm, app.module_reload()), app.module_status()));
            }
            place_camera(&app, o.camera);
            terrain = Some((app.view(), app.stats()));
            GpuRenderThread::start(&window, app, config)?
        }
    };
    let label = match o.map {
        MapKind::Water => "water crossing",
        MapKind::Land => "land crossing",
        MapKind::Generated => "generated",
        MapKind::Demo => "demo",
    };
    let started = Instant::now();
    let mut last = Instant::now();
    let mut sample = (Instant::now(), 0u64);
    let mut step = 0;
    let mut module_errors = 0;
    loop {
        let events = window.pump();
        thread.handle_events(&events);
        if events.quit || !thread.running() || o.seconds.is_some_and(|s| started.elapsed().as_secs_f32() >= s) {
            break;
        }
        let dt = last.elapsed().as_secs_f32().min(0.1);
        last = Instant::now();
        if let (Some((view, _)), Some(source)) = (&terrain, &source) {
            let mut view = view.lock().unwrap_or_else(|p| p.into_inner());
            steer(&window, &events, &mut view, source, dt);
        }
        if o.exercise {
            exercise(&mut window, started, &mut step);
        }
        let elapsed = sample.0.elapsed().as_secs_f64();
        if elapsed >= 0.5 {
            let frames = thread.stats().frames;
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
                        format!(" | {} units, tick {}, {:.1} ms/tick", s.units, s.tick, s.tick_ms)
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
            window.set_title(&format!("AssemblyEngine | {label} | {fps:.0} fps{detail}"));
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
        "{} frames in {:.1} s on {} ({}), {} swapchain recreates, {} acquire timeouts, {}",
        stats.frames,
        started.elapsed().as_secs_f32(),
        stats.device,
        stats.present_mode,
        stats.swapchain_recreates,
        stats.acquire_timeouts,
        validation_text(stats.validation, stats.validation_errors, stats.validation_warnings)
    );
    if let Some(error) = stats.error {
        bail!("render thread: {error}");
    }
    Ok(if stats.validation_errors + stats.validation_warnings > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS })
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
