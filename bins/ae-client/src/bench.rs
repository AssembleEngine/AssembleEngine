//! `ae-client --bench <scene>|all|list`: GPU and render-CPU times of fixed
//! scenes, offscreen at 1280x720, as `ae_bench` reports (one suite per
//! scene, `gpu-<scene>`) for `ae-tool bench-diff`.
//!
//! Each scene is a map, a look (tileset), units at a fixed sim tick and a
//! fixed camera or camera path. After `--warmup` frames (the subdivision
//! settles; default 60) and at least a second (GPU clocks) it renders `--frames` frames (default 300) with
//! per-pass GPU timestamps and reports:
//! * time `pass/<name>` and `gpu_total`: median GPU time per frame, without
//!   the readback copy only offscreen rendering has;
//! * time `cpu_frame`: the render thread's CPU time per frame (declare,
//!   prepare, record, submit; the wait for a free frame slot excluded), and
//!   `cpu_record`, the executor's recording and barrier part of it (both
//!   with a 50% `bench-diff` tolerance: they are noisy);
//! * counters `leaves` and `drawn`: CBT leaves and leaves drawn, summed
//!   over the measured frames (deterministic: more is a regression);
//! * info `units_visible` (in the last frame; with models also
//!   `units_cascade0..2`, the model units in each shadow cascade),
//!   `frames`, `width`, `height`: the workload; a change means times are
//!   not comparable.
//!
//! Units come from the crossing game run headless to tick 300 (once per
//! map, shared by its scenes), as `--capture` does. A scene with
//! explosions emits that many every frame (warm-up included, so the
//! particle pool is in its steady state when measuring) at seeded places
//! around the camera target; time advances 1/60 s a frame.
//!
//! The `-fox` scenes draw every unit with the Khronos Fox (converted on
//! first use, `models.rs`), every unit playing a clip (walking while it
//! moves, surveying while it stands): the `units pose` pass poses all of
//! them each frame. They also report `model_setup`, the CPU (and upload)
//! time of building the renderer's model set (`ModelSet::new`: geometry,
//! textures, baked clips, cull spheres), median of 5.

use super::models::{self, ClassModel, ClipChoice, ClipPlay, Model};
use super::{MapKind, Options, effects, place_camera, sim_options, terrain_source, tileset, units_setup, with_look};
use ae_bench::Report;
use ae_bench::diff::Thresholds;
use ae_bench::measure::{Sample, summarise};
use ae_nav::terrain::Terrain;
use ae_render::device::{GpuDevice, GpuInstance};
use ae_render::executor::ExecutorConfig;
use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
use ae_render::thread::{Headless, RenderApp, TargetInfo};
use ae_render::units::{InstanceStream, UnitFeed};
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct BenchOptions {
    pub target: String,
    pub frames: usize,
    pub warmup: usize,
    pub label: Option<String>,
    pub out: PathBuf,
    pub save: bool,
    /// `--record-threads`.
    pub record_threads: usize,
    /// `--particles-full`.
    pub particles_full: bool,
    /// `--particle-blend`.
    pub particle_blend: ae_render::particles::ParticleBlend,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self { target: String::new(), frames: 300, warmup: 60, label: None, out: PathBuf::from("bench/results"), save: true, record_threads: 1, particles_full: false, particle_blend: ae_render::particles::ParticleBlend::Sorted }
    }
}

/// How the camera moves while measuring.
#[derive(Clone, Copy)]
enum Path {
    Fixed,
    /// One full turn of yaw over the measured frames, around the target.
    Orbit,
}

struct Scene {
    name: &'static str,
    about: &'static str,
    map: MapKind,
    map_cells: i32,
    units: u32,
    tileset: Option<&'static str>,
    camera: [f32; 5],
    path: Path,
    /// Explosions emitted per frame (0: no particle renderer).
    explosions: u32,
    /// Every unit the Fox, playing a clip.
    fox: bool,
}

/// The passes the offscreen graph adds to copy the image to the host
/// (`TerrainOutput::Readback`): left out of the report, since their time
/// follows the host bus (±25% between runs).
const READBACK_PASSES: [&str; 2] = ["copy", "host read"];

/// `bench-diff` tolerance of the CPU times: tens of microseconds, whose
/// medians vary by up to 30% between runs on a desktop (thread migration,
/// clocks), occasionally 60%. They catch large regressions only.
const CPU_TOLERANCE: f64 = 0.5;

/// Absolute GPU time change a pass may show without being a regression
/// (two ticks of a 1 µs timer); only widens the tolerance of short passes.
const SMALL_PASS_NS: f64 = 2_048.0;

/// Minimum warm-up per scene, after the `--warmup` frames.
const WARMUP_SECONDS: f32 = 1.0;

const ARMY: [f32; 5] = [2900.0, 17900.0, 1400.0, 0.4, 0.85];
const CLOSE: [f32; 5] = [2900.0, 17900.0, 500.0, 0.4, 0.6];

/// The Fox's clips (1 Survey, 2 Walk, 3 Run): walk while moving, survey
/// standing still, so every unit plays one.
const FOX_CLIPS: ClipChoice = ClipChoice { moving: ClipPlay::looping(2), still: ClipPlay { clip: 1, mode: models::PlayMode::PingPong } };

const SCENES: [Scene; 9] = [
    Scene { name: "overview", about: "water crossing map, the middle from 3 km (500 of 10k units visible), temperate", map: MapKind::Water, map_cells: 2560, units: 10_000, tileset: Some("temperate"), camera: [10240.0, 10240.0, 3000.0, 0.0, 0.9], path: Path::Fixed, explosions: 0, fox: false },
    Scene { name: "army", about: "water crossing map, the army from above (8.6k units visible), temperate", map: MapKind::Water, map_cells: 2560, units: 10_000, tileset: Some("temperate"), camera: ARMY, path: Path::Fixed, explosions: 0, fox: false },
    Scene { name: "orbit", about: "water crossing map, one orbit around the river at a low angle, 10k units, temperate", map: MapKind::Water, map_cells: 2560, units: 10_000, tileset: Some("temperate"), camera: [10320.0, 13200.0, 1500.0, 0.0, 0.45], path: Path::Orbit, explosions: 0, fox: false },
    Scene { name: "close", about: "water crossing map, close-up of the ground, no units, temperate", map: MapKind::Water, map_cells: 2560, units: 0, tileset: Some("temperate"), camera: CLOSE, path: Path::Fixed, explosions: 0, fox: false },
    Scene { name: "palette", about: "as close, with the per-material palette instead of a tileset", map: MapKind::Water, map_cells: 2560, units: 0, tileset: None, camera: CLOSE, path: Path::Fixed, explosions: 0, fox: false },
    Scene { name: "battle", about: "as army, closer, with 2 explosions a frame among the units (about 40k live particles; each leaves a scorch mark, so about 700 decals by the end)", map: MapKind::Water, map_cells: 2560, units: 10_000, tileset: Some("temperate"), camera: [2900.0, 17900.0, 900.0, 0.4, 0.7], path: Path::Fixed, explosions: 2, fox: false },
    Scene { name: "army-fox", about: "as army, every unit the Fox (1,728 vertices, 24 joints) playing a clip", map: MapKind::Water, map_cells: 2560, units: 10_000, tileset: Some("temperate"), camera: ARMY, path: Path::Fixed, explosions: 0, fox: true },
    Scene { name: "near-fox", about: "as army-fox, closer (fewer units, larger on screen)", map: MapKind::Water, map_cells: 2560, units: 10_000, tileset: Some("temperate"), camera: [2900.0, 17900.0, 600.0, 0.4, 0.7], path: Path::Fixed, explosions: 0, fox: true },
    Scene { name: "cliffs", about: "generated 1024 map, steep cliffs at a low angle, temperate", map: MapKind::Generated, map_cells: 1024, units: 0, tileset: Some("temperate"), camera: [4100.0, 4100.0, 900.0, 0.6, 0.45], path: Path::Fixed, explosions: 0, fox: false },
];

fn options_for(scene: &Scene, b: &BenchOptions, fox: Option<Arc<Model>>) -> Options {
    let mut o = Options::default();
    if let Some(model) = fox.filter(|_| scene.fox) {
        o.models = vec![ClassModel { class: None, model }];
        o.clip = FOX_CLIPS;
    }
    o.particles_full = b.particles_full;
    o.particle_blend = b.particle_blend;
    o.map = scene.map;
    o.map_cells = scene.map_cells;
    o.units = scene.units;
    o.tileset = scene.tileset.map(str::to_string);
    o.camera = Some(scene.camera);
    o.quake = 0;
    o.ticks = 300;
    o
}

/// Map sources and unit streams, built once per map.
#[derive(Default)]
struct Cache {
    sources: BTreeMap<(u8, i32), (Arc<TerrainSource>, Option<Arc<Terrain>>)>,
    streams: BTreeMap<(u8, i32, u32, bool), Arc<InstanceStream>>,
    fox: Option<Arc<Model>>,
}

impl Cache {
    fn source(&mut self, o: &Options) -> Result<Arc<TerrainSource>> {
        let key = (o.map as u8, o.map_cells);
        if !self.sources.contains_key(&key) {
            let (source, nav) = terrain_source(o)?;
            self.sources.insert(key, (Arc::new(source), nav));
        }
        Ok(self.sources[&key].0.clone())
    }

    fn stream(&mut self, o: &Options) -> Result<Option<Arc<InstanceStream>>> {
        let Some(s) = sim_options(o) else { return Ok(None) };
        let key = (o.map as u8, o.map_cells, o.units, !o.models.is_empty());
        if let Some(stream) = self.streams.get(&key) {
            return Ok(Some(stream.clone()));
        }
        let started = Instant::now();
        let mut sim = super::sim::Sim::new(&s, None)?;
        for _ in 0..o.ticks {
            sim.tick()?;
        }
        let stream = Arc::new(sim.stream(0.0)?);
        eprintln!("sim: {} units at tick {} in {:.1} s", stream.len(), stream.tick, started.elapsed().as_secs_f32());
        self.streams.insert(key, stream.clone());
        Ok(Some(stream))
    }
}

pub fn run(b: &BenchOptions) -> Result<ExitCode> {
    if b.target == "list" {
        for s in &SCENES {
            println!("{:<10} {}", s.name, s.about);
        }
        return Ok(ExitCode::SUCCESS);
    }
    let scenes: Vec<&Scene> = if b.target == "all" {
        SCENES.iter().collect()
    } else {
        let names: Vec<&str> = b.target.split(',').collect();
        let mut picked = Vec::new();
        for name in names {
            picked.push(SCENES.iter().find(|s| s.name == name).with_context(|| format!("unknown scene `{name}` (try --bench list)"))?);
        }
        picked
    };
    anyhow::ensure!(b.frames >= 10, "--frames must be at least 10");

    let instance = Arc::new(GpuInstance::new(&[], None)?);
    let gpu = Arc::new(GpuDevice::new(instance.clone(), None)?);
    if gpu.timestamp_period <= 0.0 {
        bail!("{} has no graphics timestamps", gpu.device_name);
    }
    eprintln!("device {}", gpu.device_name);
    let mut cache = Cache::default();
    let mut reports = Vec::new();
    for scene in scenes {
        let report = measure(&gpu, &mut cache, scene, b)?;
        report.print();
        if b.save {
            let label = b.label.as_deref().unwrap_or(&report.commit);
            println!("saved {}", report.save(&b.out, label)?.display());
        }
        reports.push(report);
    }
    drop(gpu);
    let (errors, warnings) = (instance.counters.errors(), instance.counters.warnings());
    println!("{} scenes; {}", reports.len(), super::validation_text(instance.validation, errors, warnings));
    Ok(if errors + warnings > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS })
}

fn ns(sample_ms: &mut [f64]) -> Sample {
    for v in sample_ms.iter_mut() {
        *v *= 1e6;
    }
    summarise(sample_ms)
}

fn measure(gpu: &Arc<GpuDevice>, cache: &mut Cache, scene: &Scene, b: &BenchOptions) -> Result<Report> {
    if scene.fox && cache.fox.is_none() {
        cache.fox = Some(models::fox()?);
    }
    let o = options_for(scene, b, cache.fox.clone());
    let source = cache.source(&o)?;
    let mut app = with_look(TerrainApp::new(source, TerrainOutput::Readback), tileset(&o)?, &o);
    if let (Some(s), Some(stream)) = (sim_options(&o), cache.stream(&o)?) {
        let feed = Arc::new(UnitFeed::default());
        feed.publish(stream);
        app = app.with_units(units_setup(&s, feed)?);
    }
    // Explosions (particles and scorch marks) only in scenes that set them off.
    let feed = if scene.explosions > 0 {
        let (with, feed) = effects(app, &o);
        app = with;
        feed
    } else {
        None
    };
    place_camera(&app, o.camera);
    let extent = ash::vk::Extent2D { width: 1280, height: 720 };
    app.create_offscreen(gpu, extent)?;
    let target = TargetInfo { extent, format: ash::vk::Format::R8G8B8A8_SRGB };
    let config = ExecutorConfig { frames_in_flight: 2, profile: true, record_threads: b.record_threads, ..ExecutorConfig::default() };
    let mut headless = Headless::new(gpu, &mut app, target, config)?;
    let view = app.view();
    let stats = app.stats();
    let yaw0 = scene.camera[3];
    let set_frame = |i: usize, total: usize| {
        if let Path::Orbit = scene.path {
            view.lock().unwrap().camera.yaw = yaw0 + std::f32::consts::TAU * i as f32 / total as f32;
        }
    };
    // This frame's explosions, scattered over 600 m around the target.
    let mut emitted = 0u32;
    let mut emit = || {
        let Some(feed) = &feed else { return };
        for _ in 0..scene.explosions {
            let h = emitted.wrapping_mul(0x9e37_79b9).wrapping_add(0x7f4a_7c15).wrapping_mul(0x85eb_ca6b);
            let (dx, dz) = (((h & 0xffff) as f32 / 65535.0 - 0.5) * 600.0, ((h >> 16) as f32 / 65535.0 - 0.5) * 600.0);
            let scale = 0.8 + (h % 7) as f32 * 0.2;
            feed.explode(scene.camera[0] + dx, scene.camera[1] + dz, scale, h);
            emitted += 1;
        }
    };

    set_frame(0, b.frames);
    // Frames for the subdivision to settle, and time for the GPU's clocks
    // to rise (the first scene otherwise measures 15-25% slow).
    let warm = Instant::now();
    for _ in 0..b.warmup {
        emit();
        headless.frames(&mut app, 1)?;
    }
    while warm.elapsed().as_secs_f32() < WARMUP_SECONDS {
        emit();
        headless.frames(&mut app, 1)?;
    }
    headless.take_gpu_frames();
    let (mut cpu_frame, mut cpu_record) = (Vec::with_capacity(b.frames), Vec::with_capacity(b.frames));
    let (mut leaves, mut drawn) = (0u64, 0u64);
    let mut gpu_frames = Vec::with_capacity(b.frames);
    for i in 0..b.frames {
        set_frame(i, b.frames);
        emit();
        let started = Instant::now();
        headless.frames(&mut app, 1)?;
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        let e = headless.stats();
        cpu_frame.push((elapsed - e.wait_ms as f64).max(0.0));
        cpu_record.push(e.record_ms as f64);
        let s = *stats.lock().unwrap();
        leaves += s.leaves as u64;
        drawn += s.drawn as u64;
        gpu_frames.extend(headless.take_gpu_frames());
    }
    headless.wait()?;
    // The GPU cull's count of the last frame: read once everything has
    // finished, since during the run it can come from an earlier frame.
    let visible = app.last_unit_counts().map_or(0, |(_, v)| v as u64);
    // Model units drawn into each shadow cascade (summed over classes).
    let cascades = app.last_unit_cascade_counts().iter().fold([0u64; 3], |acc, c| std::array::from_fn(|k| acc[k] + c[k] as u64));
    let names = headless.stats().pass_names;
    gpu_frames.retain(|f| f.len() == names.len());
    anyhow::ensure!(gpu_frames.len() >= b.frames / 2, "only {} of {} frames have GPU timestamps", gpu_frames.len(), b.frames);
    drop(headless);
    RenderApp::destroy(&mut app, gpu);

    let mut report = Report::new(&format!("gpu-{}", scene.name));
    if let Some(s) = sim_options(&o).filter(|_| scene.fox) {
        report.time_with_tolerance("model_setup", model_setup(gpu, &s)?, CPU_TOLERANCE);
    }
    let counted: Vec<bool> = names.iter().map(|n| !READBACK_PASSES.contains(&n.as_str())).collect();
    let mut total: Vec<f64> = gpu_frames.iter().map(|f| f.iter().zip(&counted).filter(|(_, c)| **c).map(|(&ms, _)| ms as f64).sum()).collect();
    report.time("gpu_total", ns(&mut total));
    for (p, name) in names.iter().enumerate().filter(|(p, _)| counted[*p]) {
        let mut times: Vec<f64> = gpu_frames.iter().map(|f| f[p] as f64).collect();
        let sample = ns(&mut times);
        // GPU timers tick in about a microsecond: allow a couple of ticks.
        let tolerance = SMALL_PASS_NS / sample.median_ns.max(1.0);
        if tolerance > Thresholds::default().time {
            report.time_with_tolerance(&format!("pass/{name}"), sample, tolerance);
        } else {
            report.time(&format!("pass/{name}"), sample);
        }
    }
    report.time_with_tolerance("cpu_frame", ns(&mut cpu_frame), CPU_TOLERANCE);
    report.time_with_tolerance("cpu_record", ns(&mut cpu_record), CPU_TOLERANCE);
    report.counter("leaves", leaves, "");
    report.counter("drawn", drawn, "");
    report.info("units_visible", visible, "");
    if scene.fox {
        for (k, n) in cascades.iter().enumerate() {
            report.info(&format!("units_cascade{k}"), *n, "");
        }
    }
    report.info("frames", b.frames as u64, "");
    report.info("width", extent.width as u64, "px");
    report.info("height", extent.height as u64, "px");
    Ok(report)
}

/// `ModelSet::new` for the scene's classes and units, 5 times.
fn model_setup(gpu: &GpuDevice, s: &super::sim::SimOptions) -> Result<Sample> {
    let classes = s.classes().len();
    let per_class = models::per_class(&s.models, classes, s.observer.map(|_| classes - 2));
    let mut times = Vec::new();
    for _ in 0..5 {
        let started = Instant::now();
        let mut set = ae_render::models::ModelSet::new(gpu, &per_class, s.units, ae_render::units::MAX_CLASSES)?;
        times.push(started.elapsed().as_secs_f64() * 1e3);
        set.destroy(gpu);
    }
    Ok(ns(&mut times))
}
