//! The death sequence drawn (M14 step 6, F): one duel pair on a land map,
//! offscreen 1280×720, a fixed camera on A.
//!
//! * Each flash (the sim's `Effect` at the head) is drawn where the effect
//!   is, projected to pixels: the centroid of the flash's bright pixels is
//!   within a tick of the unit's motion of it. The same with `BACKWARDS`:
//!   the flash is behind the actor's velocity, where its `Orientation` puts
//!   it.
//! * Two frames between the death clip's end and the despawn are identical
//!   in A's box (particles off).
//! * After the despawn A's box shows no unit and no explosion (the script's
//!   death does not set off the vanish explosion).
//! * Validation reports nothing.
//!
//! Ignored by default: needs a Vulkan 1.3 GPU. Run with
//! `cargo test -p ae-client -- --ignored --test-threads=1`.

use super::*;
use ae_render::particles::{EFFECT_NAMES, ParticleFeed, ParticlesSetup};
use ae_render::terrain::app::{TerrainSource, lod_view};
use ae_render::terrain::heightmap::build_pyramid;
use ae_render::terrain::renderer::TerrainInfo;
use ae_render::thread::{Headless, TargetInfo};
use ae_render::units::InstanceStream;
use glam::{Vec2, Vec3};
use std::collections::BTreeMap;

const TICKS: u64 = 400;
const EXTENT: (u32, u32) = (1280, 720);
/// Pixels: one tick of the unit's motion (0.1 m at a 60 m camera is about
/// 1.5), the flash's size and the centroid's rounding.
const FLASH_TOLERANCE: f32 = 12.0;

struct Scene {
    /// Frame per tick (the map's frames in order).
    frames: BTreeMap<u64, Vec<u8>>,
    /// The terrain alone.
    baseline: Vec<u8>,
    /// The terrain heights (the effect's ground).
    source: Arc<TerrainSource>,
    /// The flash effects, by tick.
    flashes: BTreeMap<u64, Vec<ae_sdk::fx::Effect>>,
    /// The tick the smoke was seen at (the fall's end), if any.
    smoke: Option<u64>,
    /// A's position at the last tick it was alive (world x, nav y).
    last: [f32; 2],
    /// Tick A vanished at.
    gone: Option<u64>,
    validation: (u64, u64),
    view_proj: glam::Mat4,
    extent: (u32, u32),
}

fn scene(particles: bool, backwards: bool) -> Result<Scene> {
    use ae_render::device::{GpuDevice, GpuInstance};
    use ae_render::executor::ExecutorConfig;
    use ae_render::terrain::app::TerrainApp;

    let instance = Arc::new(GpuInstance::new(&[], Some(true))?);
    anyhow::ensure!(instance.validation, "these tests need VK_LAYER_KHRONOS_validation");
    let gpu = Arc::new(GpuDevice::new(instance.clone(), None)?);

    let terrain = ae_nav::scenario::land_crossing_terrain(512);
    let (heightmap, scale, offset, water) = nav_heightmap(&terrain, 1.0);
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let info = TerrainInfo::new(&heightmap, &pyramid, terrain.geom.cell_size, scale, offset, water);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });

    let fox = models::fox()?;
    let param = 1 | if backwards { 1 << 17 } else { 0 };
    let s = sim::SimOptions {
        map: ae_services::nav::Map::LandCrossing,
        map_cells: 512,
        units: 2,
        workers: 1,
        quake: 0,
        teams: 0,
        observer: None,
        game: "duel".into(),
        mounts: Vec::new(),
        param: Some(param),
        wasm: None,
        loaded_map: None,
        models: vec![fox],
    };
    let particle_feed = Arc::new(ParticleFeed::default());
    let mut effects = sim::Effects::default();
    effects.particles = particles.then(|| particle_feed.clone());
    let effects = Arc::new(effects);
    let mut sim = sim::Sim::new(&s, None)?.with_effects(effects);
    let unit_feed = Arc::new(UnitFeed::default());

    let mut app =
        TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(super::units_setup(&s, unit_feed.clone())?);
    if particles {
        let mut setup = ParticlesSetup::new(particle_feed, ae_render::particles::MAX_CAPACITY);
        setup.downscale = 1;
        app = app.with_particles(setup);
    }
    let camera = {
        let view = app.view();
        let mut view = view.lock().unwrap_or_else(|p| p.into_inner());
        view.camera.target = Vec2::new(406.0, 400.0);
        view.camera.distance = 100.0;
        view.camera.pitch = 0.9;
        view.camera.yaw = 0.6;
        view.fog = false;
        view.camera
    };
    let (w, h) = EXTENT;
    let lod = lod_view(&source, &camera, w, h);
    let extent = ash::vk::Extent2D { width: w, height: h };
    let target = TargetInfo {
        extent,
        format: ash::vk::Format::R8G8B8A8_SRGB,
        color_space: ash::vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    app.create_offscreen(&gpu, extent)?;
    let mut headless = Headless::new(&gpu, &mut app, target, ExecutorConfig::default())?;

    // The terrain alone: the baseline for the box after the despawn.
    unit_feed.publish(Arc::new(InstanceStream {
        tick: 0,
        published: std::time::Instant::now(),
        tick_seconds: 0.0,
        sim_seconds: 0.0,
        columns: Default::default(),
        anim: Default::default(),
    }));
    headless.frames(&mut app, 1)?;
    headless.wait()?;
    let baseline = app.readback().context("no readback")?;

    let mut frames = BTreeMap::new();
    let baseline_frame = baseline;
    let mut flashes: BTreeMap<u64, Vec<ae_sdk::fx::Effect>> = BTreeMap::new();
    let mut smoke = None;
    let mut last = [400.0, 400.0];
    let mut gone = None;
    let mut seen = false;
    for tick in 0..=TICKS {
        sim.tick()?;
        let stream = sim.stream(0.0)?;
        // A is the first row while both are drawn; its despawn is the
        // stream's two rows becoming one.
        let bodies = stream.columns[0].len();
        if bodies == 2 {
            last = [
                f32::from_bits(stream.columns[0][0]),
                f32::from_bits(stream.columns[1][0]),
            ];
            seen = true;
        } else if seen && bodies == 1 && gone.is_none() {
            gone = Some(tick);
        }
        unit_feed.publish(Arc::new(stream));
        headless.frames(&mut app, 1)?;
        headless.wait()?;
        let pixels = app.readback().context("no readback")?;
        let fx = sim.last_effects().to_vec();
        // A's position, while it is drawn.
        let flash = ae_sdk::abi::name_id(EFFECT_NAMES[0]);
        let smoke_id = ae_sdk::abi::name_id("smoke");
        if fx.iter().any(|e| e.effect == flash) {
            flashes.insert(tick, fx.iter().copied().filter(|e| e.effect == flash).collect());
            frames.insert(tick, pixels.clone());
        }
        if fx.iter().any(|e| e.effect == smoke_id) {
            smoke = Some(tick);
            frames.insert(tick, pixels.clone());
        }
        if let Some(s) = smoke
            && (tick == s + 10 || tick == s + 11)
        {
            frames.insert(tick, pixels.clone());
        }
        if let Some(g) = gone {
            // One frame after the despawn: the box with nothing in it.
            if tick <= g + 1 {
                frames.insert(tick, pixels.clone());
            }
            if tick == g + 1 {
                break;
            }
        }
    }
    // Frames after the smoke (the clip's end) and after the despawn: the
    // last few rendered ones are enough.
    headless.finish()?;
    ae_render::thread::RenderApp::destroy(&mut app, &gpu);
    eprintln!(
        "frames {:?}..{:?}, smoke {smoke:?}, gone {gone:?}",
        frames.keys().next(),
        frames.keys().last()
    );
    let counters = &instance.counters;
    Ok(Scene {
        frames,
        baseline: baseline_frame,
        source,
        flashes,
        smoke,
        last,
        gone,
        validation: (counters.errors(), counters.warnings()),
        view_proj: lod.view_proj,
        extent: EXTENT,
    })
}

fn project(scene: &Scene, p: Vec3) -> (f32, f32) {
    let clip = scene.view_proj * p.extend(1.0);
    (
        (clip.x / clip.w * 0.5 + 0.5) * scene.extent.0 as f32,
        (clip.y / clip.w * 0.5 + 0.5) * scene.extent.1 as f32,
    )
}

fn pixel(scene: &Scene, (u, v): (f32, f32)) -> usize {
    (v as usize * scene.extent.0 as usize + u as usize) * 4
}

/// The brightest pixel and the centroid of the bright ones in a box.
fn centroid(scene: &Scene, frame: &[u8], at: (f32, f32), half: i32) -> Option<(f32, f32)> {
    let (cx, cy) = (at.0 as i32, at.1 as i32);
    let mut best = 0u32;
    for dy in -half..=half {
        for dx in -half..=half {
            let (x, y) = (cx + dx, cy + dy);
            if x < 0 || y < 0 || x >= scene.extent.0 as i32 || y >= scene.extent.1 as i32 {
                continue;
            }
            let i = pixel(scene, (x as f32, y as f32));
            let l = frame[i] as u32 + frame[i + 1] as u32 + frame[i + 2] as u32;
            best = best.max(l);
        }
    }
    if best < 200 {
        return None;
    }
    let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0u32);
    for dy in -half..=half {
        for dx in -half..=half {
            let (x, y) = (cx + dx, cy + dy);
            if x < 0 || y < 0 || x >= scene.extent.0 as i32 || y >= scene.extent.1 as i32 {
                continue;
            }
            let i = pixel(scene, (x as f32, y as f32));
            let l = frame[i] as u32 + frame[i + 1] as u32 + frame[i + 2] as u32;
            if l * 10 >= best * 8 {
                sx += x as f64;
                sy += y as f64;
                n += 1;
            }
        }
    }
    (n > 0).then(|| ((sx / n as f64) as f32, (sy / n as f64) as f32))
}

#[test]
#[ignore = "needs a Vulkan GPU"]
fn the_duel_death_sequence_is_drawn_where_the_sim_puts_it() -> Result<()> {
    // Flashes with particles, on a unit that walks forwards and one that
    // walks backwards.
    for backwards in [false, true] {
        let scene = scene(true, backwards)?;
        assert_eq!(scene.validation, (0, 0), "validation messages");
        assert!(!scene.flashes.is_empty(), "no flash was drawn");
        let mut checked = 0;
        for (tick, effects) in &scene.flashes {
            let frame = &scene.frames[tick];
            for e in effects {
                // The effect's world point: its height is above the
                // terrain there.
                let ground = scene.source.height_at(Vec2::new(e.pos[0], e.pos[2]));
                let p = Vec3::new(e.pos[0], ground + e.pos[1], e.pos[2]);
                let at = project(&scene, p);
                let got =
                    centroid(&scene, frame, at, 40).with_context(|| format!("tick {tick}: no flash at {at:?}"))?;
                let d = ((got.0 - at.0).powi(2) + (got.1 - at.1).powi(2)).sqrt();
                assert!(
                    d <= FLASH_TOLERANCE,
                    "tick {tick}: flash centroid {got:?} is {d:.1} px from the effect at {at:?}"
                );
                if backwards {
                    assert!(e.pos[0] < 400.0, "backwards: flash at {:?}", e.pos);
                }
                checked += 1;
            }
        }
        eprintln!("{} flash(es) drawn where the sim puts them", checked);
    }

    // The corpse holds still, and its despawn leaves nothing: particles off.
    let scene = scene(false, false)?;
    assert_eq!(scene.validation, (0, 0), "validation messages");
    let smoke = scene.smoke.context("no smoke effect")?;
    let w = scene.extent.0 as usize;
    let last = project(&scene, Vec3::new(scene.last[0], 0.0, scene.last[1]));

    let (box_x, box_y) = (last.0 as i32 - 40, last.1 as i32 - 40);
    let box_pixels: Vec<usize> = (-40..40)
        .flat_map(|dy| (-40..40).map(move |dx| (dx, dy)))
        .filter_map(|(dx, dy)| {
            let (x, y) = (box_x + dx, box_y + dy);
            (x >= 0 && y >= 0 && (x as usize) < w && (y as usize) * w < w * scene.extent.1 as usize)
                .then(|| (y as usize * w + x as usize) * 4)
        })
        .collect();
    // Two frames while the death clip has ended and A is still there.
    let a = &scene.frames[&(smoke + 10)];
    let b = &scene.frames[&(smoke + 11)];
    assert!(
        box_pixels.iter().all(|&i| a[i..i + 4] == b[i..i + 4]),
        "the corpse's frames differ in its box"
    );
    // After the despawn, A's box is the terrain (no unit, no explosion).
    let gone = scene.gone.context("A never vanished")?;
    let after = &scene.frames[&(gone + 1).min(TICKS)];
    let changed: Vec<usize> = box_pixels
        .iter()
        .copied()
        .filter(|&i| after[i..i + 4] != scene.baseline[i..i + 4])
        .collect();
    assert!(
        changed.len() * 100 < box_pixels.len(),
        "{} of {} pixels in A's box changed after the despawn (a unit or an explosion)",
        changed.len(),
        box_pixels.len()
    );
    eprintln!(
        "corpse still at ticks {smoke}+10/11; A vanished at {gone}; box {} px",
        box_pixels.len()
    );
    Ok(())
}
