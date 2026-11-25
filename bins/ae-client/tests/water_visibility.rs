//! The actual sim, observer, client stream and GPU graph exercised together.
#[allow(dead_code)]
#[path = "../src/models.rs"]
mod models;
#[allow(dead_code)]
#[path = "../src/poser.rs"]
mod poser;
#[allow(dead_code)]
#[path = "../src/sim.rs"]
mod sim;

use ae_framegraph::{BufferDesc, PassKind, ResourceClass, ResourceId, Role};
use ae_render::device::{GpuDevice, GpuInstance};
use ae_render::executor::{Binding, BufferBinding, Executor, ExecutorConfig, PassContext, Recorder};
use ae_render::library::{LibraryPass, WaterTiers};
use ae_render::memory::OwnedBuffer;
use ae_render::particles::{Burst, ParticleBlend, ParticleFeed, ParticlesSetup, effect};
use ae_render::terrain::app::{
    NATIVE, PassWork, ScenePass, TerrainApp, TerrainOutput, TerrainSource, VisionFeed, lod_view,
};
use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
use ae_render::terrain::renderer::TerrainInfo;
use ae_render::thread::{RenderApp, TargetInfo};
use ae_render::units::{ATTR_VIEW, ATTR_WATER, InstanceStream, UnitFeed, UnitsSetup};
use ae_render::water::{WaterFeed, WaterImpact};
use ae_services::nav::Map;
use ae_view::RowState;
use anyhow::Result;
use ash::vk;
use glam::Vec2;
use gpu_allocator::MemoryLocation;
use std::path::Path;
use std::sync::{Arc, Mutex};

const EXTENT: vk::Extent2D = vk::Extent2D {
    width: 256,
    height: 192,
};
const REFRACTION: &str = "water.refraction";
const REFLECTION: &str = "water.reflection";

// The sim's Flecs column allocator is process-wide.
static SIM_TEST: Mutex<()> = Mutex::new(());

fn options(wasm: Arc<Vec<u8>>, param: u64) -> sim::SimOptions {
    sim::SimOptions {
        map: Map::WaterCrossing,
        map_cells: 64,
        units: 0,
        workers: 1,
        quake: 0,
        teams: 2,
        observer: Some(0),
        game: "water-visibility".into(),
        mounts: vec![],
        param: Some(param),
        wasm: Some(wasm),
        loaded_map: None,
        models: vec![],
    }
}

struct Probe {
    copies: Vec<(ResourceId, OwnedBuffer)>,
    particle_counts: Option<(ResourceId, ResourceId)>,
    gpu: Arc<GpuDevice>,
    executor: Executor<ScenePass>,
    app: TerrainApp,
    units: Arc<UnitFeed>,
    vision: Arc<VisionFeed>,
    tick: u64,
}

impl Probe {
    fn new(
        o: &sim::SimOptions,
        source: &Arc<TerrainSource>,
        vision: Arc<VisionFeed>,
        effects: Option<(&Arc<WaterFeed>, &Arc<ParticleFeed>, ParticleBlend, u32)>,
    ) -> Result<Self> {
        let instance = Arc::new(GpuInstance::new(&[], Some(true))?);
        assert!(instance.validation);
        let gpu = Arc::new(GpuDevice::new(instance, None)?);
        let units = Arc::new(UnitFeed::default());
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_water()
            .with_water_tiers(WaterTiers {
                reflect: true,
                refract: true,
                waves: effects.is_some(),
                ..Default::default()
            })
            .with_units(UnitsSetup {
                feed: units.clone(),
                capacity: 16,
                classes: o.classes(),
                min_pixels: 1.0,
                models: Vec::new(),
                rigs: Vec::new(),
            })
            .with_vision(vision.clone());
        if let Some((water, particles, blend, downscale)) = effects {
            let mut setup = ParticlesSetup::new(particles.clone(), 1024);
            setup.blend = blend;
            setup.downscale = downscale;
            // Isolate the flash sprite from the burst light, which also
            // illuminates the seabed and therefore changes refraction.
            setup.effects[effect::FLASH as usize].light = None;
            app = app.with_water_feed(water.clone()).with_particles(setup);
        } else {
            app = app.with_shadows();
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = Vec2::splat(256.0);
            view.camera.distance = 115.0;
            view.camera.pitch = 0.64;
            view.camera.yaw = 0.0;
            view.time = Some(0.0);
            view.fog = false;
        }
        let target = TargetInfo {
            extent: EXTENT,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        app.create_offscreen(&gpu, EXTENT)?;
        let mut executor = Executor::new(
            gpu.clone(),
            ExecutorConfig {
                frames_in_flight: 2,
                profile: false,
                record_threads: 2,
                ..Default::default()
            },
        )?;
        app.create(&gpu, executor.bindless_mut(), target, 2)?;
        let (mut graph, _) = app.declare(target, executor.bindless_mut())?;
        let mut add = |name: &str, label: &str, before: Option<LibraryPass>| -> Result<_> {
            let index = graph.resources.iter().position(|r| r.name == name).unwrap();
            let image = ResourceId(index as u32);
            let ae_framegraph::ResourceDesc::Image(ref mut image_desc) = graph.resources[index].desc else {
                panic!("expected an image: {name}");
            };
            image_desc.usage |= vk::ImageUsageFlags::TRANSFER_SRC;
            let desc = BufferDesc {
                size: u64::from(image_desc.width) * u64::from(image_desc.height) * 8,
                usage: vk::BufferUsageFlags::TRANSFER_DST,
            };
            let buffer = graph.buffer(format!("{label} probe"), desc, ResourceClass::External);
            graph
                .pass(
                    format!("{label} copy"),
                    PassKind::Transfer,
                    ScenePass {
                        program: NATIVE,
                        work: PassWork::Readback { image, buffer },
                    },
                )
                .access(image, Role::TransferSrc)
                .discard(buffer, Role::TransferDst)
                .id();
            if let Some(before) = before {
                let index = graph
                    .passes
                    .iter()
                    .position(|pass| match before {
                        LibraryPass::WaterDraw => matches!(pass.record.work, PassWork::WaterSurface { .. }),
                        LibraryPass::ParticlesDraw => matches!(pass.record.work, PassWork::Particles { .. }),
                        _ => false,
                    })
                    .expect("copy insertion point");
                let copy = graph.passes.pop().unwrap();
                graph.passes.insert(index, copy);
                for root in &mut graph.roots {
                    if root.index() >= index {
                        root.0 += 1;
                    }
                }
            }
            graph
                .pass(
                    format!("{label} host read"),
                    PassKind::Transfer,
                    ScenePass {
                        program: NATIVE,
                        work: PassWork::Library(LibraryPass::HostRead),
                    },
                )
                .access(buffer, Role::HostRead)
                .root();
            Ok((buffer, OwnedBuffer::new(&gpu, desc, MemoryLocation::GpuToCpu, label)?))
        };
        let mut copies = vec![add(REFLECTION, REFLECTION, None)?, add(REFRACTION, REFRACTION, None)?];
        if effects.is_some() {
            for name in ["water.raw", "water.churn", "water.waves"] {
                copies.push(add(name, name, None)?);
            }
            copies.push(add("scene", "before water", Some(LibraryPass::WaterDraw))?);
            copies.push(add("scene", "after particles", None)?);
            copies.push(add("scene", "before particles", Some(LibraryPass::ParticlesDraw))?);
        }
        let particle_counts = if effects.is_some() {
            let source = ResourceId(
                graph
                    .resources
                    .iter()
                    .position(|r| r.name == "particles.indirect")
                    .unwrap() as u32,
            );
            let desc = BufferDesc {
                size: 64,
                usage: vk::BufferUsageFlags::TRANSFER_DST,
            };
            let buffer = graph.buffer("particle count probe", desc, ResourceClass::External);
            graph
                .pass(
                    "particle count copy",
                    PassKind::Transfer,
                    ScenePass {
                        program: COUNTS_COPY,
                        work: PassWork::Library(LibraryPass::HostRead),
                    },
                )
                .access(source, Role::TransferSrc)
                .discard(buffer, Role::TransferDst)
                .id();
            graph
                .pass(
                    "particle count host read",
                    PassKind::Transfer,
                    ScenePass {
                        program: NATIVE,
                        work: PassWork::Library(LibraryPass::HostRead),
                    },
                )
                .access(buffer, Role::HostRead)
                .root();
            copies.push((
                buffer,
                OwnedBuffer::new(&gpu, desc, MemoryLocation::GpuToCpu, "particle counts")?,
            ));
            Some((source, buffer))
        } else {
            None
        };
        executor.set_graph(graph)?;
        let mut probe = Self {
            copies,
            particle_counts,
            gpu,
            executor,
            app,
            units,
            vision,
            tick: 0,
        };
        for _ in 0..8 {
            probe.draw()?;
        }
        {
            let view = probe.app.view();
            let mut view = view.lock().unwrap();
            view.frozen = Some(lod_view(source, &view.camera, EXTENT.width, EXTENT.height));
        }
        Ok(probe)
    }

    fn draw(&mut self) -> Result<Vec<Vec<u8>>> {
        let token = self.executor.begin_frame()?;
        self.app.prepare(self.executor.bindless_mut(), token.slot)?;
        let externals = (0..self.executor.graph().resources.len() as u32)
            .map(ResourceId)
            .map(|id| {
                if let Some((i, (_, b))) = self.copies.iter().enumerate().find(|(_, (probe, _))| *probe == id) {
                    Some(Binding::Buffer(BufferBinding {
                        buffer: b.buffer,
                        size: b.size,
                        offset: 0,
                        identity: 9967 + i as u64,
                        storage: None,
                    }))
                } else {
                    self.app.external(id, token.slot)
                }
            })
            .collect::<Vec<_>>();
        let mut recorder = ProbeRecorder {
            inner: self.app.recorder(),
            particle_counts: self.particle_counts,
        };
        self.executor
            .execute(token, &mut recorder, &|id| externals[id.index()], &[], &[])?;
        self.executor.wait(token)?;
        Ok(self
            .copies
            .iter_mut()
            .map(|(_, b)| b.mapped().unwrap().to_vec())
            .collect())
    }

    fn render(&mut self, mut stream: InstanceStream) -> Result<Vec<Vec<u8>>> {
        self.tick += 1;
        stream.tick = self.tick;
        self.units.publish(Arc::new(stream));
        if let Some(frame) = self.vision.latest() {
            self.vision.publish(Arc::new(ae_render::terrain::app::VisionFrame {
                tick: self.tick,
                los: frame.los.clone(),
                explored: frame.explored.clone(),
            }));
        }
        self.draw()
    }

    fn finish(mut self) -> Result<()> {
        self.gpu.wait_idle();
        drop(self.executor);
        self.app.destroy(&self.gpu);
        for (_, buffer) in &mut self.copies {
            buffer.destroy(&self.gpu);
        }
        let instance = self.gpu.instance.clone();
        drop(self.gpu);
        assert_eq!(instance.counters.errors(), 0);
        assert_eq!(instance.counters.warnings(), 0);
        Ok(())
    }
}

const COUNTS_COPY: u32 = NATIVE - 1;

struct ProbeRecorder<'a> {
    inner: &'a mut dyn Recorder<ScenePass>,
    particle_counts: Option<(ResourceId, ResourceId)>,
}

impl Recorder<ScenePass> for ProbeRecorder<'_> {
    fn skip(&self, pass: &ScenePass) -> bool {
        self.inner.skip(pass)
    }

    fn clear_color(&self, pass: &ScenePass, resource: ResourceId) -> [f32; 4] {
        self.inner.clear_color(pass, resource)
    }

    fn clear_depth(&self, pass: &ScenePass, resource: ResourceId) -> (f32, u32) {
        self.inner.clear_depth(pass, resource)
    }

    fn record(&mut self, pass: &ScenePass, ctx: &PassContext<'_>) -> Result<()> {
        if pass.program == COUNTS_COPY {
            let (src, dst) = self.particle_counts.expect("particle count copy");
            let (src, dst) = (ctx.buffer(src)?, ctx.buffer(dst)?);
            unsafe {
                ctx.device.cmd_copy_buffer(
                    ctx.cmd,
                    src.buffer,
                    dst.buffer,
                    &[vk::BufferCopy::default()
                        .src_offset(src.offset)
                        .dst_offset(dst.offset)
                        .size(dst.size)],
                );
            }
            Ok(())
        } else {
            self.inner.record(pass, ctx)
        }
    }
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn water_copies_follow_the_real_clients_observer_stream() -> Result<()> {
    let _sim = SIM_TEST.lock().unwrap();
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/guest/water_visibility/Cargo.toml");
    let wasm = Arc::new(ae_sim::build::build_guest(&manifest, "ae_water_visibility_fixture")?);
    let fixture = options(wasm.clone(), 0);
    let vision = Arc::new(VisionFeed::new(fixture.vision_grid()));
    let (heightmap, scale, offset) = Heightmap::from_cells((64, 64), |_, _| -10.0, None);
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, scale, offset, Some(0.0));
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let mut probe = Probe::new(&fixture, &source, vision.clone(), None)?;
    let water = Arc::new(WaterFeed::default());
    let particles = Arc::new(ae_render::particles::ParticleFeed::default());
    let mut effects = sim::Effects::default();
    effects.water = Some((source, water.clone()));
    effects.particles = Some(particles.clone());
    let effects = Arc::new(effects);

    for (mode, initial, gained) in [
        (0, None, false), // hidden ship
        (1, None, false), // submerged in LOS, beyond sonar
        (2, Some(RowState::Blip), false),
        (3, Some(RowState::Ghost), false),
        (4, None, false),                // explored-only
        (5, Some(RowState::Seen), true), // sonar gain, then loss
    ] {
        let mut traces = [Vec::new(), Vec::new()];
        for (run, present) in traces.iter_mut().zip([false, true]) {
            let mut sim = sim::Sim::new(&options(wasm.clone(), mode | (u64::from(present) << 4)), None)?
                .with_vision(vision.clone())
                .with_effects(effects.clone());
            for t in 1..=4 {
                sim.tick()?;
                let stream = sim.stream(0.0)?;
                if t >= 3 {
                    let target = stream.columns[ATTR_VIEW]
                        .iter()
                        .zip(&stream.columns[ATTR_WATER])
                        .find(|(state, _)| **state != RowState::Own as u32);
                    let expected = if present && (t == 3 || !gained) { initial } else { None };
                    assert_eq!(
                        target.map(|(&state, _)| state),
                        expected.map(|s| s as u32),
                        "mode {mode}, tick {t}"
                    );
                    if gained && expected.is_some() {
                        assert_eq!(f32::from_bits(*target.unwrap().1), 4.0);
                    }
                    run.push(probe.render(stream)?);
                }
            }
            if mode == 1 {
                water.emit(WaterImpact {
                    position: [256.0; 2],
                    radius: 3.5,
                    seed: 17,
                });
                particles.emit_all(&ae_render::particles::explosion([256.0, 0.0, 256.0], false, 1.0, 17));
                sim.switch_observer(1)?;
                assert!(water.take().is_empty());
                assert!(particles.take_all().is_empty());
                let stream = sim.stream(0.0)?;
                assert_eq!(stream.columns[ATTR_WATER].contains(&4.0f32.to_bits()), present);
                run.push(probe.render(stream)?);
                sim.switch_observer(0)?;
                run.push(probe.render(sim.stream(0.0)?)?);
            }
        }
        for (tick, (absent, present)) in traces[0].iter().zip(&traces[1]).enumerate() {
            let sub_visible = (gained && tick == 0) || (mode == 1 && tick == 2);
            assert!(
                absent[0] == present[0],
                "mode {mode}, sample {tick}: reflection changed"
            );
            assert_eq!(
                absent[1] != present[1],
                sub_visible,
                "mode {mode}, sample {tick}: refraction"
            );
        }
    }
    probe.finish()
}

#[test]
fn coast_submarines_dive_and_surface_in_the_client_stream() -> Result<()> {
    use ae_render::units::{ATTR_CLASS, ATTR_WATER};

    let _sim = SIM_TEST.lock().unwrap();
    let opts = sim::SimOptions {
        map: Map::Coast,
        map_cells: 256,
        units: 80,
        workers: 1,
        quake: 0,
        teams: 0,
        observer: None,
        game: "crossing".into(),
        mounts: vec![],
        param: None,
        wasm: None,
        loaded_map: None,
        models: vec![],
    };
    let mut sim = sim::Sim::new(&opts, None)?;
    let mut dived = false;
    let mut surfaced = false;
    for tick in 1..=340 {
        sim.tick()?;
        if tick % 10 != 0 {
            continue;
        }
        let stream = sim.stream(0.0)?;
        let depths: Vec<_> = stream.columns[ATTR_CLASS]
            .iter()
            .zip(&stream.columns[ATTR_WATER])
            .filter_map(|(&class, &depth)| (class == 5).then_some(f32::from_bits(depth)))
            .collect();
        assert!(!depths.is_empty(), "the coast has submarines");
        if dived && depths.contains(&0.0) {
            surfaced = true;
            break;
        }
        dived |= depths.iter().any(|&d| d >= 3.9);
    }
    assert!(dived && surfaced, "dived={dived}, surfaced={surfaced}");
    Ok(())
}

fn flat_water() -> Arc<TerrainSource> {
    let (heightmap, scale, offset) = Heightmap::from_cells((64, 64), |_, _| -10.0, None);
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, scale, offset, Some(0.0));
    Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    })
}

const BED: usize = 1;
const RAW: usize = 2;
const CHURN: usize = 3;
const WAVES: usize = 4;
const PRE_WATER: usize = 5;
const FINAL_SCENE: usize = 6;
const PRE_PARTICLES: usize = 7;
const COUNTS: usize = 8;

fn particle_counts(images: &[Vec<u8>]) -> (u32, u32) {
    let word = |offset| u32::from_ne_bytes(images[COUNTS][offset..offset + 4].try_into().unwrap());
    (word(4), word(52))
}

fn has_wave(raw: &[u8]) -> bool {
    raw.as_chunks::<8>()
        .0
        .iter()
        .any(|texel| texel[0] != 0 || texel[1] & 0x7f != 0)
}

fn wave_width(raw: &[u8]) -> usize {
    let mut lo = 1024;
    let mut hi = 0;
    for (i, texel) in raw.as_chunks::<8>().0.iter().enumerate() {
        if has_wave(texel) {
            lo = lo.min(i % 1024);
            hi = hi.max(i % 1024);
        }
    }
    hi.saturating_sub(lo)
}

fn two_arms(raw: &[u8]) -> bool {
    (0..1024).any(|x| {
        let mut rows = (0..1024).map(|z| has_wave(&raw[(z * 1024 + x) * 8..][..8]));
        rows.by_ref().any(|filled| filled) && rows.by_ref().any(|filled| !filled) && rows.any(|filled| filled)
    })
}

impl Probe {
    fn time(&self, time: f32) {
        self.app.view().lock().unwrap().time = Some(time);
    }

    fn reset_effects(&mut self, water: &WaterFeed, particles: &ParticleFeed) -> Result<()> {
        water.clear();
        particles.clear();
        self.units.publish(Arc::new(InstanceStream {
            tick: self.tick,
            published: std::time::Instant::now(),
            tick_seconds: 0.0,
            sim_seconds: 0.05,
            columns: std::array::from_fn(|_| Vec::new()),
            anim: Default::default(),
        }));
        self.draw()?;
        Ok(())
    }
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn surface_dynamics_follow_the_clients_observer() -> Result<()> {
    let _sim = SIM_TEST.lock().unwrap();
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/guest/water_visibility/Cargo.toml");
    let wasm = Arc::new(ae_sim::build::build_guest(&manifest, "ae_water_visibility_fixture")?);
    let fixture = options(wasm.clone(), 0);
    let vision = Arc::new(VisionFeed::new(fixture.vision_grid()));
    let source = flat_water();
    let water = Arc::new(WaterFeed::default());
    let particles = Arc::new(ParticleFeed::default());
    let mut effects = sim::Effects::default();
    effects.water = Some((source.clone(), water.clone()));
    effects.particles = Some(particles.clone());
    effects.surface_dynamics = true;
    let effects = Arc::new(effects);

    let mut probe = Probe::new(
        &fixture,
        &source,
        vision.clone(),
        Some((&water, &particles, ParticleBlend::Sorted, 1)),
    )?;
    for mode in 0..=4 {
        let mut run = |present: bool| -> Result<_> {
            probe.time(0.0);
            probe.reset_effects(&water, &particles)?;
            let mut sim = sim::Sim::new(&options(wasm.clone(), mode | 32 | (u64::from(present) << 4)), None)?
                .with_vision(vision.clone())
                .with_effects(effects.clone());
            for tick in 1..=5 {
                sim.tick()?;
                let stream = sim.stream(0.0)?;
                if tick == 5 {
                    probe.time(0.2);
                    let images = probe.render(stream)?;
                    assert!(
                        images[FINAL_SCENE] == images[PRE_PARTICLES],
                        "mode {mode}: contact emitted above-water particles"
                    );
                    // A contact never enters the pre-surface water targets.
                    return Ok(images.into_iter().take(WAVES + 1).collect::<Vec<_>>());
                }
            }
            unreachable!()
        };
        assert!(
            run(false)? == run(true)?,
            "hidden/contact mode {mode} changed a water target"
        );
    }

    for (mode, moving, label) in [
        (6, false, "parked ship"),
        (6, true, "moving ship"),
        (7, true, "moving submarine"),
    ] {
        probe.time(0.0);
        probe.reset_effects(&water, &particles)?;
        let param = mode | 16 | if moving { 32 } else { 0 };
        let mut sim = sim::Sim::new(&options(wasm.clone(), param), None)?
            .with_vision(vision.clone())
            .with_effects(effects.clone());
        let mut motion = false;
        let mut eligible_stream = None;
        for tick in 1..=8 {
            sim.tick()?;
            let stream = sim.stream(0.0)?;
            let pending = particles.take_all();
            let wanted = if mode == 7 { effect::BUBBLE } else { effect::BOW_SPRAY };
            if !pending.is_empty() {
                assert!(
                    pending.iter().all(|burst| burst.effect == wanted && tick <= 6),
                    "{label}, tick {tick}: {pending:?}"
                );
                motion = true;
            }
            particles.emit_all(&pending);
            if mode == 6 && moving && tick == 6 {
                eligible_stream = Some(stream.clone());
            }
            probe.time(tick as f32 * 0.1);
            let images = probe.render(stream)?;
            if tick == 6 || tick == 8 {
                assert_eq!(
                    has_wave(&images[RAW]),
                    mode == 6 && moving,
                    "{label}, tick {tick}: wake"
                );
                assert_eq!(
                    has_wave(&images[CHURN]),
                    mode == 6 && moving,
                    "{label}, tick {tick}: churn"
                );
                if mode == 6 && moving {
                    assert!(two_arms(&images[RAW]), "{label}: expected a Kelvin V");
                }
                if mode == 7 && tick == 8 {
                    // Sonar was lost on tick 7. The already emitted bubbles
                    // still contribute while the submerged hull is hidden.
                    assert!(particle_counts(&images).1 > 0, "bubbles vanished on loss of sonar");
                }
            }
        }
        assert_eq!(motion, moving, "{label}: motion effect publication");

        // Whatever was spawned remains short-lived. A perspective switch
        // retires it immediately and prevents the old stream re-emitting.
        sim.switch_observer(1)?;
        let switched = probe.render(sim.stream(0.0)?)?;
        assert!(
            !has_wave(&switched[RAW]) && !has_wave(&switched[CHURN]),
            "{label}: stale wake after switch"
        );
        assert_eq!(
            particle_counts(&switched),
            (0, 0),
            "{label}: old particles after switching"
        );
        sim.switch_observer(0)?;
        if let Some(stream) = eligible_stream {
            probe.time(1.1);
            probe.reset_effects(&water, &particles)?;
            probe.app.view().lock().unwrap().camera.target = Vec2::splat(48.0);
            assert!(
                !has_wave(&probe.render(stream)?[RAW]),
                "a ship outside the patch emitted a wake"
            );
            probe.app.view().lock().unwrap().camera.target = Vec2::splat(256.0);
        }
        // The churned trail lives 24 s (the prototype's), the arms 10 s.
        probe.time(26.0);
        let expired = probe.render(sim.stream(0.0)?)?;
        assert!(!has_wave(&expired[RAW]), "{label}: stale wake after expiry");
    }

    // A blast's rings start as its cavity opens and run out at the speed
    // of their wavelength: measured 2 s on, a larger crater's are wider.
    let measure_ring = |probe: &mut Probe, radius, start: f32| -> Result<_> {
        probe.time(start);
        probe.reset_effects(&water, &particles)?;
        water.emit(WaterImpact {
            position: [256.0; 2],
            radius,
            seed: 17,
        });
        probe.draw()?;
        probe.time(start + 2.0);
        Ok(wave_width(&probe.draw()?[RAW]))
    };
    let small = measure_ring(&mut probe, 5.0, 30.0)?;
    let large = measure_ring(&mut probe, 10.0, 40.0)?;
    assert!(large >= small * 3 / 2 && small > 0, "ring widths {small} and {large}");
    // The rings and the boil last up to 42 s at this size.
    probe.time(40.0 + 45.0);
    assert!(!has_wave(&probe.draw()?[RAW]), "impact ring survived its lifetime");
    probe.finish()
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn full_quality_particles_draw_on_the_correct_side_of_the_surface() -> Result<()> {
    let fixture = options(Arc::new(Vec::new()), 0);
    let source = flat_water();
    let mut bubble_refraction = None;
    for (blend, downscale) in [
        (ParticleBlend::Sorted, 1),
        (ParticleBlend::Sorted, 2),
        (ParticleBlend::Packets, 1),
        (ParticleBlend::Packets, 2),
    ] {
        let vision = Arc::new(VisionFeed::new(fixture.vision_grid()));
        let water = Arc::new(WaterFeed::default());
        let particles = Arc::new(ParticleFeed::default());
        let mut probe = Probe::new(&fixture, &source, vision, Some((&water, &particles, blend, downscale)))?;
        let mut layer = |burst: Burst, underwater: bool| -> Result<_> {
            probe.time(0.0);
            probe.reset_effects(&water, &particles)?;
            particles.emit(burst);
            probe.time(0.1);
            let mut live = probe.draw()?;
            let (above, below) = particle_counts(&live);
            assert_eq!(
                above + below,
                burst.count,
                "{blend:?}: every particle should be drawn once"
            );
            let expected_below = if underwater { burst.count } else { 0 };
            assert_eq!(below, expected_below, "{blend:?}: incorrect layer count for {burst:?}");
            particles.clear();
            let clean = probe.draw()?;
            anyhow::ensure!(
                live[PRE_WATER] == clean[PRE_WATER],
                "{blend:?}: particles wrote the pre-water scene"
            );
            assert_eq!(
                live[FINAL_SCENE] != live[PRE_PARTICLES],
                !underwater,
                "{blend:?}: duplicate or absent draw after water"
            );
            assert_eq!(
                live[BED] != clean[BED],
                underwater,
                "{blend:?}: refraction for {burst:?}"
            );
            assert!(
                live[FINAL_SCENE] != clean[FINAL_SCENE],
                "{blend:?}: invisible burst {burst:?}"
            );
            Ok(std::mem::take(&mut live[BED]))
        };
        let puff = |height, kind| Burst {
            pos: [256.0, height, 256.0],
            ground: false,
            effect: kind,
            count: 8,
            scale: 1.4,
            seed: 42,
        };
        let refraction = layer(puff(-2.0, effect::BUBBLE), true)?;
        if let Some(want) = &bubble_refraction {
            assert!(
                want == &refraction,
                "{blend:?} {downscale}x: refraction must use the same exposure"
            );
        } else {
            bubble_refraction = Some(refraction);
        }
        layer(puff(0.05, effect::BUBBLE), true)?;
        layer(puff(4.0, effect::BOW_SPRAY), false)?;

        // The flash from X has a centre one metre above the surface and
        // extends several metres across it. It must retain its full shape.
        // At this seed and scale, downward spawn jitter would also drop
        // those centres below the waterline unless the burst is anchored.
        let [surface_flash, ..] = ae_render::particles::explosion([256.0, 1.0, 256.0], true, 2.0, 1);
        layer(surface_flash, false)?;
        let [submerged_flash, ..] = ae_render::particles::explosion([256.0, -4.0, 256.0], false, 1.0, 42);
        layer(submerged_flash, true)?;
        probe.finish()?;
    }
    Ok(())
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn floating_hull_rides_the_swell_without_cutting_through_a_crest() -> Result<()> {
    let fixture = options(Arc::new(Vec::new()), 0);
    let source = flat_water();
    let vision = Arc::new(VisionFeed::new(fixture.vision_grid()));
    let water = Arc::new(WaterFeed::default());
    let particles = Arc::new(ParticleFeed::default());
    let mut probe = Probe::new(
        &fixture,
        &source,
        vision,
        Some((&water, &particles, ParticleBlend::Sorted, 1)),
    )?;
    let point = Vec2::splat(256.0);
    let camera = {
        let view = probe.app.view();
        let mut view = view.lock().unwrap();
        view.camera.distance = 35.0;
        view.frozen = None;
        view.camera
    };
    let stream = InstanceStream {
        tick: 1,
        published: std::time::Instant::now(),
        tick_seconds: 0.0,
        sim_seconds: 1.0 / 30.0,
        columns: [
            256.0f32.to_bits(),
            256.0f32.to_bits(),
            256.0f32.to_bits(),
            256.0f32.to_bits(),
            0,
            0,
            0,
            1.0f32.to_bits(),
            0,
            0,
            0,
            1.0f32.to_bits(),
            2.3f32.to_bits(),
            2,
            ae_render::units::NO_POSE,
            0,
            RowState::Own as u32,
        ]
        .map(|word| vec![word]),
        anim: Default::default(),
    };
    let params = ae_render::water::WaterParams::default();
    // The swell's highest and lowest moments at the hull in its first
    // 30 s (three trains: no single period).
    let swell = |t: f32| ae_render::water::swell_height(&params, point.to_array(), t);
    let moments = (0..3000).map(|i| i as f32 * 0.01);
    let crest = moments.clone().max_by(|a, b| swell(*a).total_cmp(&swell(*b))).unwrap();
    let trough = moments.min_by(|a, b| swell(*a).total_cmp(&swell(*b))).unwrap();
    let mut projected_y = Vec::new();
    for time in [crest, trough] {
        probe.time(time);
        probe.reset_effects(&water, &particles)?;
        let clean = probe.draw()?;
        let raised = probe.render(stream.clone())?;
        let mut rows = Vec::new();
        for (i, (before, empty)) in raised[PRE_WATER]
            .as_chunks::<8>()
            .0
            .iter()
            .zip(clean[PRE_WATER].as_chunks::<8>().0)
            .enumerate()
        {
            if before != empty {
                assert_eq!(
                    before,
                    &raised[FINAL_SCENE][i * 8..][..8],
                    "surface intersected the visible hull at {i}"
                );
                rows.push((i / EXTENT.width as usize) as f32);
            }
        }
        assert!(rows.len() > 30, "the hull must be on screen");
        projected_y.push(rows.iter().sum::<f32>() / rows.len() as f32);
    }
    let projection = lod_view(&source, &camera, EXTENT.width, EXTENT.height).view_proj;
    let screen_y = |height| {
        let clip = projection * glam::Vec4::new(point.x, height + 1.0, point.y, 1.0);
        (clip.y / clip.w * 0.5 + 0.5) * EXTENT.height as f32
    };
    let expected = screen_y(swell(crest)) - screen_y(swell(trough));
    let actual = projected_y[0] - projected_y[1];
    // The patch also carries the static displacement (two scrolling octaves
    // of up to about 0.3 m together), which the swell alone does not model.
    assert!(
        expected.abs() > 3.0 && (expected - actual).abs() <= 2.5,
        "swell height in the image: measured {actual:.2} px, expected {expected:.2} px"
    );
    probe.finish()
}
