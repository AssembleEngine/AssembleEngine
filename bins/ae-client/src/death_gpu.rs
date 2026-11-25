//! The death sequence drawn (M14 step 6, F): one duel pair on a land map,
//! offscreen 1280×720, a fixed camera on the pair. A shoots three times;
//! the third shot kills B, which falls, lies still and is despawned.
//!
//! * Each flash (the sim's `Effect` at the head) is drawn where the effect
//!   is, projected to pixels: the centroid of the flash's bright pixels is
//!   within a tick of the unit's motion of it. The same with `BACKWARDS`:
//!   the flash is behind the actor's velocity, where its `Orientation` puts
//!   it.
//! * Two frames in the still second and a half between the death clip's
//!   end and the despawn are identical in B's box (particles off; A, which
//!   idles, stands outside it).
//! * At the despawn B leaves the picture (its box changes from the corpse's
//!   frame), and no burst goes out: the script's death does not set off the
//!   client's vanish explosion.
//! * Validation reports nothing.
//!
//! Ignored by default: needs a Vulkan 1.3 GPU. Run with
//! `cargo test -p ae-client -- --ignored --test-threads=1`; with
//! `AE_TEST_CAPTURE_DIR=<dir>` the frames it keeps are saved there as PPM.

use super::*;
use ae_render::particles::{EFFECT_NAMES, ParticleFeed, ParticlesSetup};
use ae_render::terrain::app::{TerrainSource, lod_view};
use ae_render::terrain::heightmap::build_pyramid;
use ae_render::terrain::renderer::TerrainInfo;
use ae_render::thread::{Headless, TargetInfo};
use glam::{Vec2, Vec3};
use std::collections::BTreeMap;

const TICKS: u64 = 400;
const EXTENT: (u32, u32) = (1280, 720);
/// Pixels: one tick of the unit's motion (0.1 m at the 100 m camera is
/// about 1), the flash's size and the centroid's rounding.
const FLASH_TOLERANCE: f32 = 12.0;
/// Frames kept before the current one, so the corpse's frames shortly
/// before the despawn are at hand once it happens. The corpse lies still
/// for 1.5 s (45 ticks) after its clip ends; [`STILL`] ticks before the
/// despawn are inside that.
const RECENT: usize = 12;
/// Half of B's box, pixels: the corpse (about 10–15 px at this camera) fits,
/// and A, which stops about 50 px away, stays clear by [`CLEAR`].
const HALF: i32 = 24;
const CLEAR: f32 = 20.0;
const STILL: u64 = 10;

struct Scene {
    /// Frame per tick (the map's frames in order).
    frames: BTreeMap<u64, Vec<u8>>,
    /// Bursts sent to the particle feed at the despawn's tick and the one
    /// after (particles off: the feed is drained each tick, not drawn).
    vanish_bursts: usize,
    /// The terrain heights (the effect's ground).
    source: Arc<TerrainSource>,
    /// The flash effects, by tick.
    flashes: BTreeMap<u64, Vec<ae_sdk::fx::Effect>>,
    /// The tick the smoke was seen at (the fall's end, not the clip's), if
    /// any.
    smoke: Option<u64>,
    /// B's position at the last tick it was drawn (world x, nav y), and A's
    /// then.
    last: [f32; 2],
    actor: [f32; 2],
    /// Tick B vanished at.
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
    // Particles off: the feed still takes the client's bursts (drained each
    // tick, not drawn), so a vanish explosion is seen.
    effects.particles = Some(particle_feed.clone());
    let effects = Arc::new(effects);
    let mut sim = sim::Sim::new(&s, None)?.with_effects(effects);
    let unit_feed = Arc::new(UnitFeed::default());

    let mut app =
        TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(super::units_setup(&s, unit_feed.clone())?);
    if particles {
        let mut setup = ParticlesSetup::new(particle_feed.clone(), ae_render::particles::MAX_CAPACITY);
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

    let mut frames = BTreeMap::new();
    let mut vanish_bursts = 0;
    let mut flashes: BTreeMap<u64, Vec<ae_sdk::fx::Effect>> = BTreeMap::new();
    let mut smoke = None;
    let mut last = [412.0, 394.0];
    let mut actor = [400.0, 400.0];
    let mut gone = None;
    let mut seen = false;
    let mut recent: std::collections::VecDeque<(u64, Vec<u8>)> = std::collections::VecDeque::new();
    for tick in 0..=TICKS {
        sim.tick()?;
        let stream = sim.stream(0.0)?;
        // A then B (rows in entity order, A spawned first) while both are
        // drawn; B's despawn is the stream's two rows becoming one.
        let bodies = stream.columns[0].len();
        if bodies == 2 {
            actor = [
                f32::from_bits(stream.columns[0][0]),
                f32::from_bits(stream.columns[1][0]),
            ];
            last = [
                f32::from_bits(stream.columns[0][1]),
                f32::from_bits(stream.columns[1][1]),
            ];
            seen = true;
        } else if seen && bodies == 1 && gone.is_none() {
            gone = Some(tick);
        }
        unit_feed.publish(Arc::new(stream));
        if !particles {
            let bursts = particle_feed.take_all().len();
            if gone.is_some() {
                vanish_bursts += bursts;
            }
        }
        headless.frames(&mut app, 1)?;
        headless.wait()?;
        let pixels = app.readback().context("no readback")?;
        let fx = sim.last_effects().to_vec();
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
        if gone == Some(tick) {
            // The corpse's last frames, inside its still time.
            frames.extend(recent.drain(..));
        }
        if let Some(g) = gone {
            // One frame after the despawn: the box with nothing in it.
            if tick <= g + 1 {
                frames.insert(tick, pixels.clone());
            }
            if tick == g + 1 {
                break;
            }
        } else {
            recent.push_back((tick, pixels));
            if recent.len() > RECENT {
                recent.pop_front();
            }
        }
    }
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
        vanish_bursts,
        source,
        flashes,
        smoke,
        last,
        actor,
        gone,
        validation: (counters.errors(), counters.warnings()),
        view_proj: lod.view_proj,
        extent: EXTENT,
    })
}

/// With `AE_TEST_CAPTURE_DIR` set, save the scene's kept frames there as
/// PPM (`duel-<name>-<tick>.ppm`), as the renderer's GPU tests do.
fn save_frames(scene: &Scene, name: &str) -> Result<()> {
    let Some(dir) = std::env::var_os("AE_TEST_CAPTURE_DIR") else {
        return Ok(());
    };
    std::fs::create_dir_all(&dir)?;
    let (w, h) = scene.extent;
    for (tick, rgba) in &scene.frames {
        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
        ppm.extend(rgba.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]));
        std::fs::write(std::path::Path::new(&dir).join(format!("duel-{name}-{tick}.ppm")), ppm)?;
    }
    Ok(())
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
    let _one = ONE_RUNTIME.lock().unwrap_or_else(|p| p.into_inner());
    // Flashes with particles, on a unit that walks forwards and one that
    // walks backwards.
    for backwards in [false, true] {
        let scene = scene(true, backwards)?;
        save_frames(&scene, if backwards { "backwards" } else { "forwards" })?;
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
    save_frames(&scene, "death")?;
    assert_eq!(scene.validation, (0, 0), "validation messages");
    let smoke = scene.smoke.context("no smoke effect")?;
    let w = scene.extent.0 as usize;
    // B's box: 2 × HALF px square, centred on B's ground point (the
    // terrain's height there, not 0).
    let ground = scene.source.height_at(Vec2::new(scene.last[0], scene.last[1]));
    let last = project(&scene, Vec3::new(scene.last[0], ground, scene.last[1]));
    let (cx, cy) = (last.0 as i32, last.1 as i32);
    let h = scene.extent.1 as usize;
    let box_pixels: Vec<usize> = (-HALF..HALF)
        .flat_map(|dy| (-HALF..HALF).map(move |dx| (dx, dy)))
        .filter_map(|(dx, dy)| {
            let (x, y) = (cx + dx, cy + dy);
            (x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h).then(|| (y as usize * w + x as usize) * 4)
        })
        .collect();
    assert!(
        box_pixels.len() * 2 > (4 * HALF * HALF) as usize,
        "B's box at {last:?} is mostly off screen"
    );
    // A idles (Survey) near B: it must stand clear of B's box, or the
    // still-frame check sees A move.
    let ground_a = scene.source.height_at(Vec2::new(scene.actor[0], scene.actor[1]));
    let a_at = project(&scene, Vec3::new(scene.actor[0], ground_a, scene.actor[1]));
    let apart = (a_at.0 - last.0).abs().max((a_at.1 - last.1).abs());
    assert!(
        apart > HALF as f32 + CLEAR,
        "A at {a_at:?} is {apart:.0} px from B's box centre {last:?}: inside or at its edge"
    );
    let gone = scene.gone.context("B never vanished")?;
    let (still_a, still_b) = (gone - STILL - 1, gone - STILL);
    anyhow::ensure!(
        smoke + 1 < still_a,
        "B vanished at {gone}, too soon after its fall ended ({smoke}) for a still corpse"
    );
    // Two frames after the death clip has ended, while B lies still.
    let a = &scene.frames[&still_a];
    let b = &scene.frames[&still_b];
    assert!(
        box_pixels.iter().all(|&i| a[i..i + 4] == b[i..i + 4]),
        "the corpse's frames differ in its box"
    );
    // After the despawn B has left its box: the frame changed where the
    // corpse was (the box centre is on it), and no burst went out.
    let after = &scene.frames[&(gone + 1).min(TICKS)];
    let removed = box_pixels.iter().filter(|&&i| b[i..i + 4] != after[i..i + 4]).count();
    assert!(
        removed >= 16,
        "B's box barely changed at the despawn ({removed} pixels): the corpse is still drawn"
    );
    assert_eq!(
        scene.vanish_bursts, 0,
        "bursts at the despawn: the client set off a vanish explosion for a scripted death"
    );
    eprintln!(
        "corpse still at ticks {still_a}/{still_b}, fall ended {smoke}; B vanished at {gone}, {removed} of {} px changed; A {apart:.0} px away",
        box_pixels.len()
    );
    Ok(())
}
