//! `ae-client --bench <scene>|all|list`: GPU and render-CPU times of fixed
//! scenes, offscreen at 1280x720, as `ae_bench` reports (one suite per
//! scene, `gpu-<scene>`, or `gpu-<scene>-hdr10`/`-scrgb` for an HDR run)
//! for `ae-tool bench-diff`.
//!
//! Each scene is a map, a look (tileset), units at a fixed sim tick and a
//! fixed camera or camera path. After `--warmup` frames (the subdivision
//! settles; default 60) and at least a second (GPU clocks) it renders `--frames` frames (default 300) with
//! per-pass GPU timestamps and reports:
//! * time `pass/<name>`: median GPU time per frame; `gpu_total`: the sum of
//!   the passes' medians (the per-frame sums' spread), both without
//!   the readback copy only offscreen rendering has;
//! * time `cpu_frame`: the render thread's CPU time per frame (declare,
//!   prepare, record, submit; the wait for a free frame slot excluded), and
//!   `cpu_record`, the executor's recording and barrier part of it (both
//!   with a 50% `bench-diff` tolerance: they are noisy);
//! * counters `leaves` and `drawn`: CBT leaves and leaves drawn, summed
//!   over the measured frames (deterministic: more is a regression);
//! * info `units_visible` (in the last frame; with models also
//!   `units_cascade0..2`, the model units in each shadow cascade),
//!   `lights` (the clustered lights in view, in the last frame with any),
//!   `frames`, `width`, `height`: the workload; a change means times are
//!   not comparable.
//!
//! `--bench-output sdr|hdr10|scrgb` renders the readback in that encoding
//! (M13: the HDR graph adds `post.encode` and a 10-bit or 16F target); the
//! levels are fixed at the client's 203/361 signal nits, so runs are
//! reproducible whatever the display. `--bench-size WxH` sets the extent
//! (default 1280x720). A non-SDR run's records are named
//! `gpu-<scene>-hdr10` or `gpu-<scene>-scrgb`, so `bench-diff` never pairs
//! them with SDR's. The readback copy's passes (`copy`, `host read`) are
//! reported as `pass/copy` and `pass/host read` with a 50% tolerance: their
//! time follows the host bus, and they are still left out of `gpu_total`,
//! so the total stays comparable with earlier records.
//!
//! Units come from the crossing game run headless to tick 300 (once per
//! map, shared by its scenes), as `--capture` does. A scene with
//! explosions emits that many every frame (warm-up included, so the
//! particle pool is in its steady state when measuring) at seeded places
//! around the camera target; time advances 1/60 s a frame.
//!
//! The `-fox` scenes run the `scripted` game instead (10 000 Foxes in a
//! grid under the army camera, walking and standing, animated by the sim:
//! clips blended by the animation service, heads turned), drawn with the
//! Khronos Fox (converted on first use, `models.rs`): the client packs
//! every unit's channels into animation records and the `units pose` pass
//! evaluates their clips and builds every palette each frame. They also
//! report `model_setup`, the CPU (and upload) time of building the
//! renderer's model set (`ModelSet::new`: geometry, textures, clip tables,
//! cull spheres), median of 5, and `cpu_anim_pack`, the client's time to
//! read the channels and pack every unit for a stream (median over the
//! last 10 ticks), with info `anim_cells` (the animation table, 16-byte
//! cells) and `anim_writes` (cells written a tick). `live-fox` keeps the
//! sim ticking (every second frame, in the `--warmup` frames and the
//! measured ones) and reports `cpu_anim_pack` and `anim_writes` over the
//! measured ticks (with two parts of the pack: `cpu_anim_pack_read`,
//! reading the `Anim` column into records, and `cpu_anim_pack_poser`, the
//! poser's comparing and copying; the rest is the stream's rows), `cpu_anim_publish` (the feed taking a tick's writes, on
//! the sim thread), `cpu_anim_upload` (the render thread taking and
//! staging them) and `anim_cells_uploaded`.

use super::models::{self, ClassModel};
use super::{MapKind, Options, effects, place_camera, sim_options, terrain_source, tileset, units_setup, with_look};
use ae_bench::Report;
use ae_bench::diff::Thresholds;
use ae_bench::measure::{Sample, summarise};
use ae_nav::terrain::Terrain;
use ae_render::device::{GpuDevice, GpuInstance};
use ae_render::encode::{DisplayLevels, OutputEncoding};
use ae_render::executor::ExecutorConfig;
use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
use ae_render::thread::{Headless, RenderApp, TargetInfo};
use ae_render::units::{AnimWrites, InstanceStream, UnitFeed};
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
    /// `--bench-output`: what the readback target encodes (M13).
    pub output: OutputEncoding,
    /// `--bench-size`: the offscreen extent.
    pub size: (u32, u32),
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            target: String::new(),
            frames: 300,
            warmup: 60,
            label: None,
            out: PathBuf::from("bench/results"),
            save: true,
            record_threads: 1,
            particles_full: false,
            particle_blend: ae_render::particles::ParticleBlend::Sorted,
            output: OutputEncoding::Sdr,
            size: (1280, 720),
        }
    }
}

/// The display levels every `--bench-output` run uses (M13 stage 5's
/// reference: 203 nit paper white, 1.778 headroom). Fixed so the numbers do
/// not move with the display the bench runs on.
const BENCH_LEVELS: DisplayLevels = DisplayLevels {
    paper_white: 203.0,
    peak: 361.0,
};

/// Set in the child processes [`run_isolated`] starts.
const CHILD_ENV: &str = "AE_BENCH_SCENE_PROCESS";

/// Several scenes run one process each, with this process's arguments and
/// `--bench <scene>`. In one process the scenes after the heavy ones (the
/// fox and model scenes) measured up to 3x slow in some runs and not in
/// others (`lights-200` 0.75-2.3 ms, its terrain 0.30-0.62 ms), at full
/// clocks and with VRAM to spare; the same scene in a fresh process right
/// after the heavy ones measures 0.76 ms. Two whole runs of one binary
/// flagged 11-13 metrics against each other in one process, 1 as one
/// process per scene. The cause inside the process is not known yet.
fn run_isolated(scenes: &[&Scene]) -> Result<ExitCode> {
    let exe = std::env::current_exe().context("the client's own path")?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let at = args
        .iter()
        .position(|a| a == "--bench")
        .context("--bench in the arguments")?
        + 1;
    let mut failed = Vec::new();
    for scene in scenes {
        let mut child_args = args.clone();
        child_args[at] = scene.name.to_string();
        let status = std::process::Command::new(&exe)
            .args(&child_args)
            .env(CHILD_ENV, "1")
            .status()
            .with_context(|| format!("run scene {}", scene.name))?;
        if !status.success() {
            failed.push(scene.name);
        }
    }
    println!(
        "{} scenes, one process each; {}",
        scenes.len(),
        if failed.is_empty() {
            "all succeeded".to_string()
        } else {
            format!("failed: {}", failed.join(", "))
        }
    );
    Ok(if failed.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
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
    /// The side of the square around the target they scatter over (metres).
    spread: f32,
    /// Every unit the Fox, playing a clip.
    fox: bool,
    /// The sim keeps ticking: a fresh tick every [`LIVE_FRAMES`] frames,
    /// posed and uploaded as when playing.
    live: bool,
    /// Fixed point lights scattered around the target (the clusters'
    /// measure).
    lights: u32,
    /// The Fox models carry a point light each: the pass appends one light
    /// per unit (the models' lights' measure).
    model_lights: bool,
}

/// Frames a tick in the `live` scenes (a 30 Hz sim at 60 frames a second).
const LIVE_FRAMES: usize = 2;

/// The passes the offscreen graph adds to copy the image to the host
/// (`TerrainOutput::Readback`): left out of `gpu_total`, since their time
/// follows the host bus (±25% between runs), but reported on their own with
/// that tolerance (the copy is the target's bandwidth measured, M13 stage
/// 7).
const READBACK_PASSES: [&str; 2] = ["copy", "host read"];

/// `bench-diff` tolerance of the readback passes.
const READBACK_TOLERANCE: f64 = 0.5;

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

/// The `scripted` game's parameter for the `-fox` scenes: 10 000 units on
/// a 12 m grid (radius 3) from (2304, 17296), the army camera's view.
const FOX_PARAM: u64 = 10_000 | 12 << 32 | 144 << 40 | 1081 << 52;

const SCENES: [Scene; 17] = [
    Scene {
        name: "overview",
        about: "water crossing map, the middle from 3 km (500 of 10k units visible), temperate",
        map: MapKind::Water,
        map_cells: 2560,
        units: 10_000,
        tileset: Some("temperate"),
        camera: [10240.0, 10240.0, 3000.0, 0.0, 0.9],
        path: Path::Fixed,
        explosions: 0,
        spread: 600.0,
        fox: false,
        live: false,
        lights: 0,
        model_lights: false,
    },
    Scene {
        name: "army",
        about: "water crossing map, the army from above (8.6k units visible), temperate",
        map: MapKind::Water,
        map_cells: 2560,
        units: 10_000,
        tileset: Some("temperate"),
        camera: ARMY,
        path: Path::Fixed,
        explosions: 0,
        spread: 600.0,
        fox: false,
        live: false,
        lights: 0,
        model_lights: false,
    },
    Scene {
        name: "orbit",
        about: "water crossing map, one orbit around the river at a low angle, 10k units, temperate",
        map: MapKind::Water,
        map_cells: 2560,
        units: 10_000,
        tileset: Some("temperate"),
        camera: [10320.0, 13200.0, 1500.0, 0.0, 0.45],
        path: Path::Orbit,
        explosions: 0,
        spread: 600.0,
        fox: false,
        live: false,
        lights: 0,
        model_lights: false,
    },
    Scene {
        name: "close",
        about: "water crossing map, close-up of the ground, no units, temperate",
        map: MapKind::Water,
        map_cells: 2560,
        units: 0,
        tileset: Some("temperate"),
        camera: CLOSE,
        path: Path::Fixed,
        explosions: 0,
        spread: 600.0,
        fox: false,
        live: false,
        lights: 0,
        model_lights: false,
    },
    Scene {
        name: "palette",
        about: "as close, with the per-material palette instead of a tileset",
        map: MapKind::Water,
        map_cells: 2560,
        units: 0,
        tileset: None,
        camera: CLOSE,
        path: Path::Fixed,
        explosions: 0,
        spread: 600.0,
        fox: false,
        live: false,
        lights: 0,
        model_lights: false,
    },
    Scene {
        name: "battle",
        about: "as army, closer, with 2 explosions a frame among the units (about 40k live particles; each leaves a scorch mark, so about 700 decals by the end)",
        map: MapKind::Water,
        map_cells: 2560,
        units: 10_000,
        tileset: Some("temperate"),
        camera: [2900.0, 17900.0, 900.0, 0.4, 0.7],
        path: Path::Fixed,
        explosions: 2,
        spread: 600.0,
        fox: false,
        live: false,
        lights: 0,
        model_lights: false,
    },
    Scene {
        name: "scorched",
        about: "as battle without units, close (120 m), its 2 explosions a frame over 100 m: overlapping scorch marks fill most of the view (the decals' measure)",
        map: MapKind::Water,
        map_cells: 2560,
        units: 0,
        tileset: Some("temperate"),
        camera: [2900.0, 17900.0, 120.0, 0.4, 0.7],
        path: Path::Fixed,
        explosions: 2,
        spread: 100.0,
        fox: false,
        live: false,
        lights: 0,
        model_lights: false,
    },
    Scene {
        name: "army-fox",
        about: "as army, 10k Foxes (1,728 vertices, 24 joints) animated by the sim (the scripted game)",
        map: MapKind::Water,
        map_cells: 2560,
        units: 10_000,
        tileset: Some("temperate"),
        camera: ARMY,
        path: Path::Fixed,
        explosions: 0,
        spread: 600.0,
        fox: true,
        live: false,
        lights: 0,
        model_lights: false,
    },
    Scene {
        name: "army-model-lights",
        about: "as army-fox, each Fox carrying a point light: the pass appends one light per unit (10k, the list's 8192 cap drops the rest)",
        map: MapKind::Water,
        map_cells: 2560,
        units: 10_000,
        tileset: Some("temperate"),
        camera: ARMY,
        path: Path::Fixed,
        explosions: 0,
        spread: 600.0,
        fox: true,
        live: false,
        lights: 0,
        model_lights: true,
    },
    Scene {
        name: "live-fox",
        about: "as army-fox, the sim running: a tick every second frame, its animation records packed and uploaded",
        map: MapKind::Water,
        map_cells: 2560,
        units: 10_000,
        tileset: Some("temperate"),
        camera: ARMY,
        path: Path::Fixed,
        explosions: 0,
        spread: 600.0,
        fox: true,
        live: true,
        lights: 0,
        model_lights: false,
    },
    Scene {
        name: "live-puppet",
        about: "as live-fox, the 10k Foxes of the puppet game (a script per unit: clips, callins, head turns), on its 4 m grid",
        map: MapKind::Water,
        map_cells: 2560,
        units: 10_000,
        tileset: Some("temperate"),
        camera: [200.0, 200.0, 1400.0, 0.4, 0.85],
        path: Path::Fixed,
        explosions: 0,
        spread: 0.0,
        fox: true,
        live: true,
        lights: 0,
        model_lights: false,
    },
    Scene {
        name: "near-fox",
        about: "as army-fox, closer (fewer units, larger on screen)",
        map: MapKind::Water,
        map_cells: 2560,
        units: 10_000,
        tileset: Some("temperate"),
        camera: [2900.0, 17900.0, 600.0, 0.4, 0.7],
        path: Path::Fixed,
        explosions: 0,
        spread: 600.0,
        fox: true,
        live: false,
        lights: 0,
        model_lights: false,
    },
    Scene {
        name: "lights-200",
        about: "as battle without explosions, 200 fixed point lights over 800 m around the target (ranges 15-45 m)",
        map: MapKind::Water,
        map_cells: 2560,
        units: 10_000,
        tileset: Some("temperate"),
        camera: [2900.0, 17900.0, 900.0, 0.4, 0.7],
        path: Path::Fixed,
        explosions: 0,
        spread: 600.0,
        fox: false,
        live: false,
        lights: 200,
        model_lights: false,
    },
    Scene {
        name: "lights-1000",
        about: "as lights-200 with 1000 lights",
        map: MapKind::Water,
        map_cells: 2560,
        units: 10_000,
        tileset: Some("temperate"),
        camera: [2900.0, 17900.0, 900.0, 0.4, 0.7],
        path: Path::Fixed,
        explosions: 0,
        spread: 600.0,
        fox: false,
        live: false,
        lights: 1000,
        model_lights: false,
    },
    Scene {
        name: "lights-4000",
        about: "as lights-200 with 4000 lights",
        map: MapKind::Water,
        map_cells: 2560,
        units: 10_000,
        tileset: Some("temperate"),
        camera: [2900.0, 17900.0, 900.0, 0.4, 0.7],
        path: Path::Fixed,
        explosions: 0,
        spread: 600.0,
        fox: false,
        live: false,
        lights: 4000,
        model_lights: false,
    },
    Scene {
        name: "lights-low",
        about: "1000 lights as lights-1000, the camera low (600 m, pitch 0.3: the view reaches far into the distance)",
        map: MapKind::Water,
        map_cells: 2560,
        units: 10_000,
        tileset: Some("temperate"),
        camera: [2900.0, 17900.0, 600.0, 0.4, 0.3],
        path: Path::Fixed,
        explosions: 0,
        spread: 600.0,
        fox: false,
        live: false,
        lights: 1000,
        model_lights: false,
    },
    Scene {
        name: "cliffs",
        about: "generated 1024 map, steep cliffs at a low angle, temperate",
        map: MapKind::Generated,
        map_cells: 1024,
        units: 0,
        tileset: Some("temperate"),
        camera: [4100.0, 4100.0, 900.0, 0.6, 0.45],
        path: Path::Fixed,
        explosions: 0,
        spread: 600.0,
        fox: false,
        live: false,
        lights: 0,
        model_lights: false,
    },
];

fn options_for(scene: &Scene, b: &BenchOptions, fox: Option<ClassModel>) -> Options {
    let mut o = Options::default();
    if let Some(model) = fox.filter(|_| scene.fox) {
        o.models = vec![model];
        // The puppet game's 10k, each with a script; the rest the scripted
        // game's grid parameter.
        if scene.name == "live-puppet" {
            o.game = "puppet".into();
            o.param = Some(10_000);
        } else {
            o.game = "scripted".into();
            o.param = Some(FOX_PARAM);
        }
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

/// The terrain of a map and its nav terrain, built once per map.
type Sources = BTreeMap<(u8, i32), (Arc<TerrainSource>, Option<Arc<Terrain>>)>;

/// A scene's unit stream and its poser, built once per map and unit set.
type Streams = BTreeMap<(u8, i32, u32, bool), (Arc<InstanceStream>, Option<Posing>)>;

/// Map sources and unit streams, built once per map.
#[derive(Default)]
struct Cache {
    sources: Sources,
    streams: Streams,
    fox: Option<ClassModel>,
    fox_light: Option<ClassModel>,
}

impl Cache {
    fn source(&mut self, o: &Options) -> Result<Arc<TerrainSource>> {
        let key = (o.map as u8, o.map_cells);
        if let std::collections::btree_map::Entry::Vacant(e) = self.sources.entry(key) {
            let (source, nav) = terrain_source(o)?;
            e.insert((Arc::new(source), nav));
        }
        Ok(self.sources[&key].0.clone())
    }

    /// The units at tick `o.ticks`, and with models how posing them went
    /// (the last 10 ticks are streamed, so records span real motion); the
    /// stream carries the whole animation table.
    fn stream(&mut self, o: &Options) -> Result<Option<(Arc<InstanceStream>, Option<Posing>)>> {
        let Some(s) = sim_options(o) else { return Ok(None) };
        let key = (o.map as u8, o.map_cells, o.units, !o.models.is_empty());
        if let Some(stream) = self.streams.get(&key) {
            return Ok(Some(stream.clone()));
        }
        let started = Instant::now();
        let mut sim = super::sim::Sim::new(&s, None)?;
        let posed = !o.models.is_empty();
        let (mut pose_ms, mut writes, mut table) = (Vec::new(), Vec::new(), Vec::new());
        let mut stream = None;
        for t in 0..o.ticks {
            sim.tick()?;
            if (posed && t + 10 >= o.ticks) || t + 1 == o.ticks {
                let mut s = sim.stream(0.0)?;
                s.anim.apply(&mut table);
                pose_ms.push(sim.pose_ms());
                writes.push(s.anim.len() as f64);
                // Their memory for the next stream, as publishing does.
                sim.recycle(std::mem::take(&mut s.anim));
                stream = Some(s);
            }
        }
        let mut stream = stream.context("no ticks")?;
        let posing = posed.then(|| Posing {
            pack: ns(&mut pose_ms[1..].to_vec()),
            writes: median(&mut writes[1..].to_vec()),
            cells: table.len() as u64,
        });
        stream.anim = AnimWrites::whole(&table);
        eprintln!(
            "sim: {} units at tick {} in {:.1} s{}",
            stream.len(),
            stream.tick,
            started.elapsed().as_secs_f32(),
            if posed {
                format!(
                    ", {} animation cells, {} written in the last tick, packed in {:.2} ms",
                    table.len(),
                    writes.last().copied().unwrap_or(0.0),
                    sim.pose_ms()
                )
            } else {
                String::new()
            }
        );
        let stream = Arc::new(stream);
        self.streams.insert(key, (stream.clone(), posing));
        Ok(Some((stream, posing)))
    }
}

/// How packing went: the client's time per stream (reading the channels,
/// packing them, finding the cells that changed), cells written per tick
/// (medians; the first stream, which writes every cell, left out), and
/// cells in the table.
#[derive(Clone, Copy)]
struct Posing {
    pack: Sample,
    writes: u64,
    cells: u64,
}

fn median(v: &mut [f64]) -> u64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).map_or(0, |&m| m as u64)
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
            picked.push(
                SCENES
                    .iter()
                    .find(|s| s.name == name)
                    .with_context(|| format!("unknown scene `{name}` (try --bench list)"))?,
            );
        }
        picked
    };
    anyhow::ensure!(b.frames >= 10, "--frames must be at least 10");
    if scenes.len() > 1 && std::env::var_os(CHILD_ENV).is_none() {
        return run_isolated(&scenes);
    }

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
    println!(
        "{} scenes; {}",
        reports.len(),
        super::validation_text(instance.validation, errors, warnings)
    );
    Ok(if errors + warnings > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    })
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
    if scene.model_lights && cache.fox_light.is_none() {
        cache.fox_light = Some(models::fox_with_light()?);
    }
    let fox = if scene.model_lights {
        cache.fox_light.clone()
    } else {
        cache.fox.clone()
    };
    let o = options_for(scene, b, fox);
    let source = cache.source(&o)?;
    let mut app = with_look(TerrainApp::new(source, TerrainOutput::Readback), tileset(&o)?, &o)
        .with_display(b.output, BENCH_LEVELS);
    let mut posing = None;
    // The live scenes' sim and feed: every stream is published (each
    // writes only the cells that changed).
    let mut live = None;
    match sim_options(&o) {
        Some(s) if scene.live => {
            let mut sim = super::sim::Sim::new(&s, None)?;
            let feed = Arc::new(UnitFeed::default());
            for t in 0..o.ticks {
                sim.tick()?;
                if t + 10 >= o.ticks {
                    let stream = sim.stream(0.0)?;
                    sim.recycle(feed.publish_reusing(stream));
                }
            }
            app = app.with_units(units_setup(&s, feed.clone())?);
            live = Some((sim, feed));
        }
        Some(s) => {
            if let Some((stream, pose)) = cache.stream(&o)? {
                posing = pose;
                let feed = Arc::new(UnitFeed::default());
                feed.publish(stream);
                app = app.with_units(units_setup(&s, feed)?);
            }
        }
        None => {}
    }
    // Live: a tick every `LIVE_FRAMES` frames, its packing time and
    // table writes.
    let (mut live_pose, mut live_writes, mut live_publish) = (Vec::new(), Vec::new(), Vec::new());
    let (mut live_read, mut live_poser) = (Vec::new(), Vec::new());
    let mut tick_live = |frame: usize, measured: bool| -> Result<()> {
        let Some((sim, feed)) = live.as_mut().filter(|_| frame.is_multiple_of(LIVE_FRAMES)) else {
            return Ok(());
        };
        sim.tick()?;
        if feed.wants_whole() {
            sim.write_whole_anim();
        }
        let stream = sim.stream(0.0)?;
        let writes = stream.anim.len();
        let started = Instant::now();
        let spare = feed.publish_reusing(stream);
        if measured {
            live_publish.push(started.elapsed().as_secs_f64() * 1e3);
            live_pose.push(sim.pose_ms());
            let (read, poser) = sim.pose_parts_ms();
            live_read.push(read);
            live_poser.push(poser);
            live_writes.push(writes as f64);
        }
        sim.recycle(spare);
        Ok(())
    };
    // Explosions (particles and scorch marks) only in scenes that set them off.
    let feed = if scene.explosions > 0 {
        let (with, feed) = effects(app, &o);
        app = with;
        feed
    } else {
        None
    };
    place_camera(&app, o.camera);
    if scene.lights > 0 {
        let target = app.view().lock().unwrap().camera.target;
        let source = app.source().clone();
        app = app.with_lights(ae_render::lights::scatter(scene.lights, target, 400.0, 4.0, 7, &|p| {
            source.height_at(p)
        }));
    }
    let extent = ash::vk::Extent2D {
        width: b.size.0,
        height: b.size.1,
    };
    app.create_offscreen(gpu, extent)?;
    let target = TargetInfo {
        extent,
        format: ash::vk::Format::R8G8B8A8_SRGB,
        color_space: ash::vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: true,
        record_threads: b.record_threads,
        ..ExecutorConfig::default()
    };
    let mut headless = Headless::new(gpu, &mut app, target, config)?;
    let view = app.view();
    let stats = app.stats();
    let yaw0 = scene.camera[3];
    let set_frame = |i: usize, total: usize| {
        if let Path::Orbit = scene.path {
            view.lock().unwrap().camera.yaw = yaw0 + std::f32::consts::TAU * i as f32 / total as f32;
        }
    };
    // This frame's explosions, scattered over `spread` around the target.
    let mut emitted = 0u32;
    let mut emit = || {
        let Some(feed) = &feed else { return };
        for _ in 0..scene.explosions {
            let h = emitted
                .wrapping_mul(0x9e37_79b9)
                .wrapping_add(0x7f4a_7c15)
                .wrapping_mul(0x85eb_ca6b);
            let (dx, dz) = (
                ((h & 0xffff) as f32 / 65535.0 - 0.5) * scene.spread,
                ((h >> 16) as f32 / 65535.0 - 0.5) * scene.spread,
            );
            let scale = 0.8 + (h % 7) as f32 * 0.2;
            feed.explode(scene.camera[0] + dx, scene.camera[1] + dz, scale, h);
            emitted += 1;
        }
    };

    set_frame(0, b.frames);
    // Time for the GPU's clocks to rise (the first scene otherwise measures
    // 15-25% slow): frames with no explosions and no ticks, so how many
    // there are cannot change what is measured.
    let warm = Instant::now();
    while warm.elapsed().as_secs_f32() < WARMUP_SECONDS {
        headless.frames(&mut app, 1)?;
    }
    // Then exactly `--warmup` frames for the subdivision to settle, with
    // the explosions and ticks: the scene's state when measuring starts is
    // the same whatever the frame rate (a time-based count here moved
    // `battle`'s lights and its total between runs).
    for warm_frame in 0..b.warmup {
        emit();
        tick_live(warm_frame, false)?;
        headless.frames(&mut app, 1)?;
    }
    headless.take_gpu_frames();
    let (mut cpu_frame, mut cpu_record) = (Vec::with_capacity(b.frames), Vec::with_capacity(b.frames));
    let (mut leaves, mut drawn) = (0u64, 0u64);
    let mut gpu_frames = Vec::with_capacity(b.frames);
    let (mut upload_ms, mut uploaded) = (Vec::new(), Vec::new());
    for i in 0..b.frames {
        set_frame(i, b.frames);
        emit();
        tick_live(i, true)?;
        let started = Instant::now();
        headless.frames(&mut app, 1)?;
        let (cells, ms) = app.last_anim_upload();
        if cells > 0 && scene.live {
            upload_ms.push(ms);
            uploaded.push(cells as f64);
        }
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
    let cascades = app
        .last_unit_cascade_counts()
        .iter()
        .fold([0u64; 3], |acc, c| std::array::from_fn(|k| acc[k] + c[k] as u64));
    let names = headless.stats().pass_names;
    gpu_frames.retain(|f| f.len() == names.len());
    anyhow::ensure!(
        gpu_frames.len() >= b.frames / 2,
        "only {} of {} frames have GPU timestamps",
        gpu_frames.len(),
        b.frames
    );
    drop(headless);
    RenderApp::destroy(&mut app, gpu);

    // An HDR run is its own record (`gpu-<scene>-hdr10`, `-scrgb`): HDR10
    // has SDR's 4 B/px, so the workload info alone would not keep
    // `bench-diff` from comparing it with an SDR record.
    let suffix = match b.output {
        OutputEncoding::Sdr => "",
        OutputEncoding::Hdr10Pq => "-hdr10",
        OutputEncoding::ScRgb => "-scrgb",
    };
    let mut report = Report::new(&format!("gpu-{}{suffix}", scene.name));
    if let Some(s) = sim_options(&o).filter(|_| scene.fox) {
        report.time_with_tolerance("model_setup", model_setup(gpu, &s)?, CPU_TOLERANCE);
    }
    if let Some(p) = posing {
        report.time_with_tolerance("cpu_anim_pack", p.pack, CPU_TOLERANCE);
        report.info("anim_cells", p.cells, "");
        report.info("anim_writes", p.writes, "");
    }
    if scene.live {
        anyhow::ensure!(!live_pose.is_empty() && !upload_ms.is_empty(), "no live ticks measured");
        report.time_with_tolerance("cpu_anim_pack", ns(&mut live_pose), CPU_TOLERANCE);
        report.time_with_tolerance("cpu_anim_pack_read", ns(&mut live_read), CPU_TOLERANCE);
        report.time_with_tolerance("cpu_anim_pack_poser", ns(&mut live_poser), CPU_TOLERANCE);
        report.time_with_tolerance("cpu_anim_publish", ns(&mut live_publish), CPU_TOLERANCE);
        report.time_with_tolerance("cpu_anim_upload", ns(&mut upload_ms), CPU_TOLERANCE);
        report.info("anim_writes", median(&mut live_writes), "");
        report.info("anim_cells_uploaded", median(&mut uploaded), "");
    }
    let counted: Vec<bool> = names.iter().map(|n| !READBACK_PASSES.contains(&n.as_str())).collect();
    let passes: Vec<(&String, Sample)> = names
        .iter()
        .enumerate()
        .filter(|(p, _)| counted[*p])
        .map(|(p, name)| {
            let mut times: Vec<f64> = gpu_frames.iter().map(|f| f[p] as f64).collect();
            (name, ns(&mut times))
        })
        .collect();
    // `gpu_total` is the sum of the passes' medians. The median of each
    // frame's sum moved 10-13% between runs while every pass agreed within
    // 2%: single passes spike in some frames (`terrain lod` ±79%), so the
    // sum's median falls between modes. The per-frame sums still give the
    // spread.
    let mut total: Vec<f64> = gpu_frames
        .iter()
        .map(|f| {
            f.iter()
                .zip(&counted)
                .filter(|(_, c)| **c)
                .map(|(&ms, _)| ms as f64)
                .sum()
        })
        .collect();
    let per_frame = ns(&mut total);
    let summed = Sample {
        median_ns: passes.iter().map(|(_, s)| s.median_ns).sum(),
        min_ns: passes.iter().map(|(_, s)| s.min_ns).sum(),
        ..per_frame
    };
    report.time("gpu_total", summed);
    for (name, sample) in passes {
        // GPU timers tick in about a microsecond: allow a couple of ticks.
        let tolerance = SMALL_PASS_NS / sample.median_ns.max(1.0);
        if tolerance > Thresholds::default().time {
            report.time_with_tolerance(&format!("pass/{name}"), sample, tolerance);
        } else {
            report.time(&format!("pass/{name}"), sample);
        }
    }
    // The readback copy, on its own: outside `gpu_total` (host bus), but
    // measured for the target's bandwidth (M13 stage 7).
    for (p, name) in names.iter().enumerate().filter(|(p, _)| !counted[*p]) {
        let mut times: Vec<f64> = gpu_frames.iter().map(|f| f[p] as f64).collect();
        report.time_with_tolerance(&format!("pass/{name}"), ns(&mut times), READBACK_TOLERANCE);
    }
    report.info("bytes_per_pixel", b.output.bytes_per_pixel(), "B/px");
    report.time_with_tolerance("cpu_frame", ns(&mut cpu_frame), CPU_TOLERANCE);
    report.time_with_tolerance("cpu_record", ns(&mut cpu_record), CPU_TOLERANCE);
    report.counter("leaves", leaves, "");
    report.counter("drawn", drawn, "");
    report.info("units_visible", visible, "");
    // The clustered lights listed (in view) in the last frame with any.
    report.info("lights", app.light_stats().lock().unwrap().lights as u64, "");
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
fn model_setup(gpu: &Arc<GpuDevice>, s: &super::sim::SimOptions) -> Result<Sample> {
    let classes = s.classes().len();
    let per_class = models::per_class(&s.models, classes, s.observer.map(|_| classes - 2));
    // The textures register in a registry of their own, made outside the
    // timing (its queued writes are never flushed: nothing draws).
    let mut bindless = ae_render::bindless::Bindless::new(gpu, 1)?;
    let mut times = Vec::new();
    for _ in 0..5 {
        let started = Instant::now();
        let mut set = ae_render::models::ModelSet::new(
            gpu,
            &mut bindless,
            &per_class,
            &[],
            s.units,
            ae_render::units::MAX_CLASSES,
        )?;
        times.push(started.elapsed().as_secs_f64() * 1e3);
        set.destroy(gpu);
    }
    bindless.destroy(gpu);
    Ok(ns(&mut times))
}
