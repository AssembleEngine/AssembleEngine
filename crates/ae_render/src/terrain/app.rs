//! The terrain as a [`RenderApp`]: a LOD compute pass, a draw pass into the
//! backbuffer (or an offscreen target read back to the host) with a
//! transient reverse-Z depth buffer, and present. With units
//! ([`TerrainApp::with_units`]), three more passes after the terrain draw:
//! reset, cull and draw the units into the same target and depth buffer
//! (see [`crate::units`]).
//!
//! With water ([`TerrainApp::with_water`], on a map with a water level), a
//! water surface pass follows, blending over the terrain and units. With
//! shadows ([`TerrainApp::with_shadows`]) the hills shade the terrain, the
//! units and the water.
//!
//! The graph is the engine's own ([`native_graph`]) or a render module's
//! ([`TerrainApp::with_module`]): the module declares resources and passes
//! from the pass library ([`crate::library`]); the host sizes the library
//! resources, checks the wiring and records the passes. A new module can be
//! handed over while running ([`TerrainApp::module_reload`]): it replaces
//! the old one at the next frame if it loads, declares and wires; otherwise
//! the old one keeps rendering and the error is reported
//! ([`TerrainApp::module_status`]). A trap in the module's per-frame call
//! stops calling it; its last graph keeps rendering.
//!
//! The view (camera, error target, debug switches) is shared with the main
//! thread through [`TerrainApp::view`]; the render thread reads it once per
//! frame and never waits for anyone else.

use super::camera::Camera;
use super::heightmap::{Heightmap, Pyramid};
use super::sun;
use super::tileset::Tileset;
use super::renderer::{FrameParams, FrameStats, LodView, TerrainInfo, TerrainRenderer, TerrainUpdate, VisionGrid, Wireframe};
use crate::device::GpuDevice;
use crate::executor::{Binding, BufferBinding, ImageBinding, PassContext, Recorder, SharedRecorder};
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::thread::{RenderApp, TargetInfo};
use crate::library::{self, LibraryEnv, LibraryPass, LibraryResource, ResourceMap};
use crate::fullscreen::FullscreenProgram;
use crate::module::{ModuleFrame, RenderModule};
use ae_render_abi::module::{DECLARE_READBACK, DECLARE_UNITS, DECLARE_WATER, FrameInput, Program, ProgramTable};
use crate::units::{UnitFrame, UnitRenderer, UnitsSetup};
use ae_framegraph::{BufferDesc, FrameGraph, ImageDesc, LoadOp, PassKind, ResourceClass, ResourceId, Role, ShaderStage};
use anyhow::{Context as _, Result};
use ash::vk;
use glam::Vec2;
use gpu_allocator::MemoryLocation;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
/// Sky / fog colour (linear).
pub const SKY: [f32; 4] = [0.62, 0.70, 0.80, 1.0];
/// sRGB like the swapchain, so a capture shows what the window shows.
const READBACK_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

/// The map: heights, error pyramid and how they map to the world. Built
/// once on the CPU, shared by the main thread (camera ground height) and the
/// render thread (uploads).
pub struct TerrainSource {
    pub heightmap: Heightmap,
    pub pyramid: Pyramid,
    pub info: TerrainInfo,
    /// Material id per cell (see [`TerrainInfo::has_materials`]).
    pub materials: Option<Vec<u8>>,
}

impl TerrainSource {
    /// World height of the terrain under world point (x, z).
    pub fn height_at(&self, world: Vec2) -> f32 {
        self.heightmap.sample(world / self.info.cell_size) * self.info.height_scale + self.info.height_offset
    }

    /// A camera looking at the middle of the map from the south.
    pub fn overview_camera(&self) -> Camera {
        let (w, h) = self.info.extent();
        let target = Vec2::new(w * 0.5, h * 0.5);
        Camera { target, ground: self.height_at(target), distance: w.max(h) * 0.6, yaw: 0.0, pitch: 0.75, fov_y: 0.8 }
    }
}

/// Terrain edits handed to the render thread ([`TerrainApp::terrain_edits`]),
/// applied in order at the start of the next frame, and what has been
/// applied so far.
#[derive(Debug, Default)]
pub struct TerrainEdits {
    pub pending: Vec<TerrainUpdate>,
    /// Updates applied, and the terrain version of the last one.
    pub applied: u32,
    pub version: u32,
    /// Bytes uploaded by the last frame that applied any.
    pub last_bytes: usize,
    pub last_ms: f32,
}

/// What the main thread controls.
#[derive(Clone, Copy, Debug)]
pub struct TerrainView {
    pub camera: Camera,
    pub tau: f32,
    pub wireframe: Wireframe,
    pub lod_colors: bool,
    pub fog: bool,
    /// Keep the subdivision of this view (fly around a frozen mesh).
    pub frozen: Option<LodView>,
    /// Passed to the render module every frame ([`FrameInput::flags`]).
    pub module_flags: u32,
    /// Shade with the tileset if the app has one ([`TerrainApp::with_tileset`]).
    pub tileset: bool,
    /// The tileset's blend-weight debug view.
    pub weights: bool,
    /// Shade by the vision feed, if the app has one ([`TerrainApp::with_vision`]).
    pub vision: bool,
    /// Draw the water surface, if the app has one ([`TerrainApp::with_water`]).
    pub water: bool,
    /// Animation time (waves, foam, caustics) in seconds; `None`: wall
    /// clock since the app was made when presenting, the frame index / 60
    /// when reading back (deterministic captures and tests).
    pub time: Option<f32>,
    /// Shade by the hills' shadows, if the app has them ([`TerrainApp::with_shadows`]).
    pub shadows: bool,
}

/// One observer's vision for the terrain: line of sight now and cells ever
/// seen, one bit per cell of `grid` (rows of [`VisionGrid::stride`] words).
#[derive(Clone, Debug)]
pub struct VisionFrame {
    pub tick: u64,
    pub los: Vec<u64>,
    pub explored: Vec<u64>,
}

/// Where the sim side publishes vision frames (newest wins) for a grid
/// fixed when the app is made.
#[derive(Debug)]
pub struct VisionFeed {
    pub grid: VisionGrid,
    latest: Mutex<Option<Arc<VisionFrame>>>,
}

impl VisionFeed {
    pub fn new(grid: VisionGrid) -> Self {
        Self { grid, latest: Mutex::new(None) }
    }

    pub fn publish(&self, frame: Arc<VisionFrame>) {
        *self.latest.lock().unwrap_or_else(|p| p.into_inner()) = Some(frame);
    }

    pub fn latest(&self) -> Option<Arc<VisionFrame>> {
        self.latest.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
}

/// One pass of the scene's graph: the program it came from (its payload;
/// [`NATIVE`] for the engine's own graph) and what it records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenePass {
    pub program: u32,
    pub work: PassWork,
}

/// The program of the native graph's passes (never disabled).
pub const NATIVE: u32 = u32::MAX;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PassWork {
    Library(LibraryPass),
    /// [`LibraryPass::ReadbackCopy`] with its source image and buffer.
    Readback { image: ResourceId, buffer: ResourceId },
    /// A module's fullscreen program (by payload), sampling `sampled` in
    /// order.
    Fullscreen { sampled: Vec<ResourceId> },
}

/// The render module driving the graph, with its wired declaration.
struct ModuleState {
    module: RenderModule,
    /// Target the declaration is for.
    declared_for: (vk::Extent2D, vk::Format),
    graph: FrameGraph<ScenePass>,
    resources: ResourceMap,
    /// By payload: the fullscreen programs and where their uniform blocks
    /// start in the module's per-frame uniform bytes.
    programs: Vec<Option<(FullscreenProgram, usize)>>,
    uniform_bytes: usize,
    frame: ModuleFrame,
    /// `frame` trapped: no more calls; the graph keeps rendering.
    failed: bool,
    frames: u32,
}

/// A module's declaration, wired and with its programs built.
struct ModuleGraph {
    graph: FrameGraph<ScenePass>,
    resources: ResourceMap,
    programs: Vec<Option<(FullscreenProgram, usize)>>,
    uniform_bytes: usize,
}

/// Render module loads and errors, for the main thread.
#[derive(Clone, Debug, Default)]
pub struct ModuleStatus {
    /// Modules loaded and installed (the first one included).
    pub loads: u32,
    /// A module is driving the graph.
    pub active: bool,
    /// Every error so far, oldest first (failed reloads, traps).
    pub errors: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainOutput {
    Present,
    /// Render into an app-owned image and copy it to a host buffer
    /// ([`TerrainApp::readback`]); for tests and captures.
    Readback,
}

/// The installed graph's library resources (by resource index) and its
/// readback image, if any.
struct Wiring {
    resources: ResourceMap,
}

struct Offscreen {
    image: OwnedImage,
    buffer: OwnedBuffer,
    extent: vk::Extent2D,
}

pub struct TerrainApp {
    source: Arc<TerrainSource>,
    view: Arc<Mutex<TerrainView>>,
    stats: Arc<Mutex<FrameStats>>,
    output: TerrainOutput,
    renderer: Option<TerrainRenderer>,
    units: Option<UnitsSetup>,
    tileset: Option<Arc<Tileset>>,
    vision: Option<Arc<VisionFeed>>,
    /// Per slot: the tick of the vision frame written into it.
    vision_written: Vec<Option<u64>>,
    unit_renderer: Option<UnitRenderer>,
    /// `(rows, visible)` of the last completed frame with units.
    unit_counts: Arc<Mutex<(u32, u32)>>,
    /// The slot `prepare` last filled.
    last_slot: usize,
    offscreen: Option<Offscreen>,
    wiring: Option<Wiring>,
    engine: Option<wasmtime::Engine>,
    module: Option<ModuleState>,
    gpu: Option<Arc<GpuDevice>>,
    frames_in_flight: usize,
    /// Linear, clamped: what fullscreen programs sample with.
    sampler: vk::Sampler,
    pending_module: Arc<Mutex<Option<Vec<u8>>>>,
    module_status: Arc<Mutex<ModuleStatus>>,
    edits: Arc<Mutex<TerrainEdits>>,
    extent: vk::Extent2D,
    update_lod: bool,
    wireframe: Wireframe,
    /// Draw water (opt-in; only on a map with a water level).
    water: bool,
    /// The water is drawn this frame (`view.water`).
    water_on: bool,
    /// Shadows (opt-in): the hills' ([`super::sun`]).
    shadows: bool,
    /// Frames prepared, and when the app was made (animation time).
    frames: u64,
    started: Instant,
}

impl TerrainApp {
    pub fn new(source: Arc<TerrainSource>, output: TerrainOutput) -> Self {
        let view = TerrainView { camera: source.overview_camera(), tau: 2.0, wireframe: Wireframe::Off, lod_colors: false, fog: true, frozen: None, module_flags: 0, tileset: true, weights: false, vision: true, water: true, time: None, shadows: true };
        Self {
            source,
            view: Arc::new(Mutex::new(view)),
            stats: Arc::default(),
            output,
            renderer: None,
            units: None,
            tileset: None,
            vision: None,
            vision_written: Vec::new(),
            unit_renderer: None,
            unit_counts: Arc::default(),
            last_slot: 0,
            offscreen: None,
            wiring: None,
            engine: None,
            module: None,
            gpu: None,
            frames_in_flight: 1,
            sampler: vk::Sampler::null(),
            pending_module: Arc::default(),
            module_status: Arc::default(),
            edits: Arc::default(),
            extent: vk::Extent2D { width: 1, height: 1 },
            update_lod: true,
            wireframe: Wireframe::Off,
            water: false,
            water_on: false,
            shadows: false,
            frames: 0,
            started: Instant::now(),
        }
    }

    /// Draw the water surface (a `water.draw` pass after the units) on a map
    /// with a water level; on a map without one this does nothing.
    pub fn with_water(mut self) -> Self {
        self.water = self.source.info.water_level.is_some();
        self
    }

    /// Shade by the sun's shadows: the hills' over the terrain, the units
    /// and the water (a heightfield computed when the renderer is made and
    /// kept up to date by terrain edits).
    pub fn with_shadows(mut self) -> Self {
        self.shadows = true;
        self
    }

    /// Shade the terrain with `tileset` (its layers are uploaded when the
    /// renderer is made; the map supplies only heights and material ids,
    /// so one tileset serves any map). Without one, the material palette.
    pub fn with_tileset(mut self, tileset: Arc<Tileset>) -> Self {
        self.tileset = Some(tileset);
        self
    }

    /// Shade the terrain by an observer's vision (fog of war) from `feed`;
    /// until its first frame everything reads as unexplored.
    pub fn with_vision(mut self, feed: Arc<VisionFeed>) -> Self {
        self.vision = Some(feed);
        self
    }

    /// Draw the units of `setup.feed` on the terrain.
    pub fn with_units(mut self, setup: UnitsSetup) -> Self {
        self.units = Some(setup);
        self
    }

    /// Take the graph from the render module `wasm` (loaded at the first
    /// frame; failing to load it is an error of that frame).
    pub fn with_module(self, wasm: Vec<u8>) -> Self {
        *self.pending_module.lock().unwrap_or_else(|p| p.into_inner()) = Some(wasm);
        self
    }

    /// Put a module here to replace the running one at the next frame.
    pub fn module_reload(&self) -> Arc<Mutex<Option<Vec<u8>>>> {
        self.pending_module.clone()
    }

    /// Push [`TerrainUpdate`]s here (made by a
    /// [`super::edit::TerrainEditor`] over the same source) to apply them
    /// at the next frame. Applying one waits for the device to go idle, as
    /// the textures are shared by every frame in flight: fine for edits
    /// that land a few times a second at most.
    pub fn terrain_edits(&self) -> Arc<Mutex<TerrainEdits>> {
        self.edits.clone()
    }

    /// Apply pending terrain edits (device idle first: every slot's frame
    /// may sample the textures).
    fn apply_edits(&mut self) -> Result<()> {
        let pending = std::mem::take(&mut self.edits.lock().unwrap_or_else(|p| p.into_inner()).pending);
        if pending.is_empty() {
            return Ok(());
        }
        let started = Instant::now();
        let gpu = self.gpu.clone().context("no device")?;
        unsafe { gpu.device.device_wait_idle() }.context("wait for the device before a terrain edit")?;
        let renderer = self.renderer.as_mut().context("terrain renderer not created")?;
        for update in &pending {
            renderer.apply(&gpu, update).context("terrain edit upload")?;
        }
        let mut edits = self.edits.lock().unwrap_or_else(|p| p.into_inner());
        edits.applied += pending.len() as u32;
        edits.version = pending.last().map_or(edits.version, |u| u.version);
        edits.last_bytes = pending.iter().map(TerrainUpdate::bytes).sum();
        edits.last_ms = started.elapsed().as_secs_f32() * 1e3;
        Ok(())
    }

    pub fn module_status(&self) -> Arc<Mutex<ModuleStatus>> {
        self.module_status.clone()
    }

    /// Units in the stream and units the cull passed, of the most recently
    /// completed frame.
    pub fn unit_counts(&self) -> Arc<Mutex<(u32, u32)>> {
        self.unit_counts.clone()
    }

    /// `(rows, visible)` of the last frame `prepare` filled, once the device
    /// is idle (after `run_headless`).
    pub fn last_unit_counts(&mut self) -> Option<(u32, u32)> {
        let units = self.unit_renderer.as_mut()?;
        Some((units.rows(self.last_slot), units.read_visible(self.last_slot)))
    }

    /// The shared view; the main thread writes it, the render thread reads
    /// it once per frame.
    pub fn view(&self) -> Arc<Mutex<TerrainView>> {
        self.view.clone()
    }

    /// Leaf counts of the most recently completed frame.
    pub fn stats(&self) -> Arc<Mutex<FrameStats>> {
        self.stats.clone()
    }

    /// The last offscreen frame (RGBA8, sRGB-encoded, row-major), after
    /// `run_headless`.
    pub fn readback(&mut self) -> Option<Vec<u8>> {
        let offscreen = self.offscreen.as_mut()?;
        let bytes = (offscreen.extent.width * offscreen.extent.height * 4) as usize;
        offscreen.buffer.mapped().map(|data| data[..bytes].to_vec())
    }

    /// The camera matrices of `view` for the current target.
    fn params(&self, view: &TerrainView) -> FrameParams {
        let time = view.time.unwrap_or_else(|| match self.output {
            TerrainOutput::Readback => self.frames as f32 / 60.0,
            TerrainOutput::Present => self.started.elapsed().as_secs_f32(),
        });
        let aspect = self.extent.width as f32 / self.extent.height.max(1) as f32;
        let mut camera = view.camera;
        camera.ground = self.source.height_at(camera.target);
        let eye = camera.eye(|p| self.source.height_at(p));
        let view_proj = camera.proj(aspect) * camera.view(eye);
        let live = LodView { pos: eye, view_proj, proj_scale: camera.proj_scale(self.extent.height as f32) };
        FrameParams {
            view_proj,
            cam_pos: eye,
            lod_view: view.frozen.unwrap_or(live),
            tau: view.tau,
            update_lod: view.frozen.is_none(),
            wireframe: view.wireframe,
            lod_colors: view.lod_colors,
            fog: view.fog,
            tileset: view.tileset,
            weights: view.weights,
            vision: view.vision && self.vision.is_some(),
            water: view.water && self.water,
            time,
            shadows: view.shadows && self.shadows,
        }
    }
}

/// The terrain's fog (colour and density) for other passes, as the terrain
/// renderer sets it.
fn fog(info: &TerrainInfo, on: bool) -> [f32; 4] {
    let density = if on { 1.0 / (info.cells.0.max(info.cells.1) as f32 * info.cell_size * 1.2) } else { 0.0 };
    [0.62, 0.70, 0.80, density]
}

/// The view a frozen LOD would keep (call on the main thread with the
/// target's size).
pub fn lod_view(source: &TerrainSource, camera: &Camera, width: u32, height: u32) -> LodView {
    let mut camera = *camera;
    camera.ground = source.height_at(camera.target);
    let eye = camera.eye(|p| source.height_at(p));
    let view_proj = camera.proj(width as f32 / height.max(1) as f32) * camera.view(eye);
    LodView { pos: eye, view_proj, proj_scale: camera.proj_scale(height as f32) }
}

impl RenderApp for TerrainApp {
    type Pass = ScenePass;

    fn create(&mut self, gpu: &Arc<GpuDevice>, format: vk::Format, frames_in_flight: usize) -> Result<()> {
        if let Some(mut old) = self.unit_renderer.take() {
            old.destroy(gpu);
        }
        if let Some(mut old) = self.renderer.take() {
            old.destroy(gpu);
        }
        self.gpu = Some(gpu.clone());
        self.frames_in_flight = frames_in_flight;
        if self.sampler == vk::Sampler::null() {
            let info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            self.sampler = unsafe { gpu.device.create_sampler(&info, None)? };
        }
        let color = if self.output == TerrainOutput::Readback { READBACK_FORMAT } else { format };
        let s = &self.source;
        let shadows = self.shadows.then(|| sun::shadow_heights(&s.heightmap, s.info.cell_size, s.info.height_scale, s.info.height_offset, sun::direction()));
        let terrain = TerrainRenderer::new(gpu, &s.heightmap, &s.pyramid, s.info, s.materials.as_deref(), self.tileset.as_deref(), self.vision.as_ref().map(|f| f.grid), shadows.as_deref(), color, DEPTH_FORMAT, frames_in_flight)
            .context("terrain renderer")?;
        if let Some(setup) = &self.units {
            self.unit_renderer = Some(UnitRenderer::new(gpu, &terrain, setup, color, DEPTH_FORMAT, frames_in_flight).context("unit renderer")?);
        }
        self.renderer = Some(terrain);
        Ok(())
    }

    fn declare(&mut self, target: TargetInfo) -> Result<(FrameGraph<ScenePass>, Option<ResourceId>)> {
        self.extent = target.extent;
        let env = self.env(target)?;
        let pending = self.pending_module.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(wasm) = pending {
            match self.load_module(&wasm, &env) {
                Ok(state) => {
                    if let Some(old) = self.module.replace(state) {
                        self.retire(old.programs);
                    }
                    let mut status = self.module_status.lock().unwrap_or_else(|p| p.into_inner());
                    status.loads += 1;
                    status.active = true;
                }
                Err(error) if self.module.is_none() => return Err(error.context("render module")),
                Err(error) => self.module_status.lock().unwrap_or_else(|p| p.into_inner()).errors.push(format!("reload rejected, the running module stays: {error:#}")),
            }
        }
        let (graph, resources) = match self.module.take() {
            Some(mut state) => {
                if state.declared_for != (env.extent, env.color_format) {
                    let declared = self.declare_module(&mut state.module, &env);
                    match declared {
                        Ok(d) => {
                            let old = std::mem::replace(&mut state.programs, d.programs);
                            self.retire(old);
                            (state.graph, state.resources, state.uniform_bytes, state.declared_for) = (d.graph, d.resources, d.uniform_bytes, (env.extent, env.color_format));
                        }
                        Err(error) => {
                            self.module = Some(state);
                            return Err(error.context("render module declaring a new target"));
                        }
                    }
                }
                let out = (state.graph.clone(), state.resources.clone());
                self.module = Some(state);
                out
            }
            None => {
                let (graph, wiring) = self.wire(native_graph(&env, self.output), &env, |p| Ok((NATIVE, ScenePassSource::Library(*p))))?;
                (graph, wiring.resources)
            }
        };
        let backbuffers: Vec<ResourceId> = graph.resources.iter().enumerate().filter(|(_, r)| r.class == ResourceClass::Backbuffer).map(|(i, _)| ResourceId(i as u32)).collect();
        match self.output {
            TerrainOutput::Present => anyhow::ensure!(backbuffers.len() == 1, "the graph declares {} Backbuffer resources; presenting needs one", backbuffers.len()),
            TerrainOutput::Readback => anyhow::ensure!(backbuffers.is_empty(), "an offscreen graph declares no Backbuffer"),
        }
        self.wiring = Some(Wiring { resources });
        Ok((graph, backbuffers.first().copied()))
    }

    fn prepare(&mut self, slot: usize) -> Result<()> {
        self.apply_edits()?;
        let view = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        let params = self.params(&view);
        self.update_lod = params.update_lod;
        self.wireframe = params.wireframe;
        self.water_on = params.water;
        self.frames += 1;
        let renderer = self.renderer.as_mut().context("terrain renderer not created")?;
        let stats = renderer.read_stats(slot);
        *self.stats.lock().unwrap_or_else(|p| p.into_inner()) = stats;
        renderer.write_frame(slot, &params);
        if let Some(feed) = &self.vision {
            if self.vision_written.len() <= slot {
                self.vision_written.resize(slot + 1, None);
            }
            match feed.latest() {
                Some(frame) if self.vision_written[slot] != Some(frame.tick) => {
                    renderer.write_vision(slot, &frame.los, &frame.explored)?;
                    self.vision_written[slot] = Some(frame.tick);
                }
                Some(_) => {}
                None if self.vision_written[slot].is_none() => {
                    let empty = vec![0u64; feed.grid.words()];
                    renderer.write_vision(slot, &empty, &empty)?;
                    self.vision_written[slot] = Some(u64::MAX);
                }
                None => {}
            }
        }
        if let (Some(units), Some(setup)) = (self.unit_renderer.as_mut(), self.units.as_ref()) {
            // This slot's previous frame has completed: its count is final.
            let previous = (units.rows(slot), units.read_visible(slot));
            *self.unit_counts.lock().unwrap_or_else(|p| p.into_inner()) = previous;
            let stream = setup.feed.latest();
            let frame = UnitFrame {
                view_proj: params.view_proj,
                cam_pos: params.cam_pos,
                proj_scale: view.camera.proj_scale(self.extent.height as f32),
                fog: fog(&self.source.info, params.fog),
                shadows: params.shadows,
            };
            units.write_frame(slot, &frame, stream.as_deref(), Instant::now());
        }
        if let Some(state) = self.module.as_mut().filter(|m| !m.failed) {
            let mut input = FrameInput {
                width: self.extent.width,
                height: self.extent.height,
                frame: state.frames,
                units: self.unit_renderer.as_ref().map_or(0, |u| u.rows(slot)),
                time: state.module.loaded().elapsed().as_secs_f32(),
                flags: view.module_flags,
                view_proj: params.view_proj.to_cols_array(),
                ..FrameInput::default()
            };
            input.camera[..3].copy_from_slice(&params.cam_pos.to_array());
            let result = state.module.frame(&input).and_then(|frame| {
                anyhow::ensure!(frame.uniforms.len() == state.uniform_bytes, "`frame` returned {} uniform bytes; the fullscreen programs declare {}", frame.uniforms.len(), state.uniform_bytes);
                Ok(frame)
            });
            match result {
                Ok(frame) => {
                    for (program, offset) in state.programs.iter_mut().flatten() {
                        let n = program.uniform_bytes as usize;
                        program.write_uniforms(slot, &frame.uniforms[*offset..*offset + n]);
                    }
                    state.frame = frame;
                }
                Err(error) => {
                    state.failed = true;
                    state.frame = ModuleFrame::default();
                    self.module_status.lock().unwrap_or_else(|p| p.into_inner()).errors.push(format!("render module stopped (its last graph keeps rendering): {error:#}"));
                }
            }
            state.frames += 1;
        }
        self.last_slot = slot;
        Ok(())
    }

    fn external(&self, resource: ResourceId, slot: usize) -> Option<Binding> {
        let lib = (*self.wiring.as_ref()?.resources.get(resource.index())?)?;
        let units = || self.unit_renderer.as_ref();
        let offscreen = || self.offscreen.as_ref();
        Some(match lib {
            LibraryResource::TerrainLeaves => Binding::Buffer(self.renderer.as_ref()?.leaves()),
            LibraryResource::TerrainIndirect => Binding::Buffer(self.renderer.as_ref()?.indirect()),
            LibraryResource::UnitInstances => Binding::Buffer(units()?.instances(slot)),
            LibraryResource::UnitVisible => Binding::Buffer(units()?.visible()),
            LibraryResource::UnitIndirect => Binding::Buffer(units()?.indirect()),
            LibraryResource::UnitCounts => Binding::Buffer(units()?.counts(slot)),
            LibraryResource::ReadbackTarget => {
                let o = offscreen()?;
                Binding::Image(ImageBinding { image: o.image.image, view: o.image.view, format: READBACK_FORMAT, extent: o.extent, range: o.image.range(), identity: (0xAE7 << 40) | 10 })
            }
            LibraryResource::ReadbackBuffer => {
                let o = offscreen()?;
                Binding::Buffer(BufferBinding { buffer: o.buffer.buffer, offset: 0, size: o.buffer.size, identity: (0xAE7 << 40) | 11 })
            }
        })
    }

    fn recorder(&mut self) -> &mut dyn Recorder<ScenePass> {
        self
    }

    fn status(&self) -> String {
        let stats = *self.stats.lock().unwrap_or_else(|p| p.into_inner());
        let units = match &self.unit_renderer {
            Some(_) => {
                let (rows, visible) = *self.unit_counts.lock().unwrap_or_else(|p| p.into_inner());
                format!(", {rows} units, {visible} visible")
            }
            None => String::new(),
        };
        format!("terrain: {} leaves, {} drawn{units}", stats.leaves, stats.drawn)
    }

    fn destroy(&mut self, gpu: &Arc<GpuDevice>) {
        if let Some(state) = self.module.take() {
            for (mut program, _) in state.programs.into_iter().flatten() {
                program.destroy(gpu);
            }
        }
        if self.sampler != vk::Sampler::null() {
            unsafe { gpu.device.destroy_sampler(self.sampler, None) };
            self.sampler = vk::Sampler::null();
        }
        if let Some(mut units) = self.unit_renderer.take() {
            units.destroy(gpu);
        }
        if let Some(mut renderer) = self.renderer.take() {
            renderer.destroy(gpu);
        }
        if let Some(mut offscreen) = self.offscreen.take() {
            offscreen.image.destroy(gpu);
            offscreen.buffer.destroy(gpu);
        }
    }
}

impl TerrainApp {
    /// Create the offscreen target and readback buffer for `extent`
    /// (readback output only; call before `run_headless`).
    pub fn create_offscreen(&mut self, gpu: &GpuDevice, extent: vk::Extent2D) -> Result<()> {
        let desc = ImageDesc::color_2d(READBACK_FORMAT, extent.width, extent.height, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC);
        let image = OwnedImage::new(gpu, desc, "terrain offscreen")?;
        let size = extent.width as u64 * extent.height as u64 * 4;
        let buffer = OwnedBuffer::new(gpu, BufferDesc { size, usage: vk::BufferUsageFlags::TRANSFER_DST }, MemoryLocation::GpuToCpu, "terrain readback")?;
        self.offscreen = Some(Offscreen { image, buffer, extent });
        Ok(())
    }
}

impl Recorder<ScenePass> for TerrainApp {
    fn skip(&self, pass: &ScenePass) -> bool {
        self.module.as_ref().is_some_and(|m| m.frame.is_disabled(pass.program))
    }

    fn clear_color(&self, _pass: &ScenePass, _resource: ResourceId) -> [f32; 4] {
        SKY
    }

    fn clear_depth(&self, _pass: &ScenePass, _resource: ResourceId) -> (f32, u32) {
        (0.0, 0) // reverse-Z
    }

    fn record(&mut self, pass: &ScenePass, ctx: &PassContext<'_>) -> Result<()> {
        self.scene_recorder().record(pass, ctx)
    }

    fn shared(&self) -> Option<Box<dyn SharedRecorder<ScenePass> + '_>> {
        Some(Box::new(self.scene_recorder()))
    }
}

impl TerrainApp {
    /// What recording reads, borrowed for one frame: the same from one
    /// thread or several.
    fn scene_recorder(&self) -> SceneRecorder<'_> {
        SceneRecorder {
            renderer: self.renderer.as_ref(),
            units: self.unit_renderer.as_ref(),
            programs: self.module.as_ref().map(|m| m.programs.as_slice()),
            sampler: self.sampler,
            update_lod: self.update_lod,
            wireframe: self.wireframe,
            water: self.water_on,
        }
    }
}

/// [`TerrainApp`]'s recording state, without the module's WASM instance.
struct SceneRecorder<'a> {
    renderer: Option<&'a TerrainRenderer>,
    units: Option<&'a UnitRenderer>,
    programs: Option<&'a [Option<(FullscreenProgram, usize)>]>,
    sampler: vk::Sampler,
    update_lod: bool,
    wireframe: Wireframe,
    water: bool,
}

impl SharedRecorder<ScenePass> for SceneRecorder<'_> {
    fn record(&self, pass: &ScenePass, ctx: &PassContext<'_>) -> Result<()> {
        let units = || self.units.context("unit renderer not created");
        let terrain = || self.renderer.context("terrain renderer not created");
        match &pass.work {
            PassWork::Library(lib) => match lib {
                LibraryPass::TerrainLod => terrain()?.record_lod(ctx, self.update_lod),
                LibraryPass::TerrainDraw => terrain()?.record_draw(ctx, self.wireframe),
                LibraryPass::UnitsReset => units()?.record_reset(ctx),
                LibraryPass::UnitsCull => units()?.record_cull(ctx),
                LibraryPass::UnitsDraw => units()?.record_draw(ctx),
                LibraryPass::UnitsCount => units()?.record_count(ctx),
                LibraryPass::WaterDraw => {
                    if self.water {
                        terrain()?.record_water(ctx);
                    }
                }
                LibraryPass::ReadbackCopy => anyhow::bail!("readback.copy is wired as PassWork::Readback"),
                LibraryPass::HostRead | LibraryPass::Present => {}
            },
            PassWork::Fullscreen { sampled } => {
                let programs = self.programs.context("fullscreen pass without a module")?;
                let (program, _) = programs.get(pass.program as usize).and_then(Option::as_ref).context("fullscreen program missing")?;
                let images = sampled.iter().map(|&r| ctx.image(r)).collect::<Result<Vec<_>>>()?;
                program.record(ctx, &images, self.sampler)?;
            }
            PassWork::Readback { image, buffer } => {
                let (image, buffer) = (ctx.image(*image)?, ctx.buffer(*buffer)?);
                crate::demo::record_readback(ctx, image, buffer);
            }
        }
        Ok(())
    }
}

/// How a graph's pass payload is wired, before checks.
#[derive(Clone, Copy, Debug)]
pub enum ScenePassSource {
    Library(LibraryPass),
    /// A fullscreen program; checked and built by the caller.
    Fullscreen,
}

/// The scene's graph as the engine declares it natively: the terrain, the
/// units when the scene has them, the water surface when it draws water,
/// and present or readback. Library buffers
/// are declared with size 0 ([`crate::library::resolve`] sizes them). A
/// render module that declares the same graph gets the same bake.
pub fn native_graph(env: &LibraryEnv, output: TerrainOutput) -> FrameGraph<LibraryPass> {
    let (width, height) = (env.extent.width, env.extent.height);
    let mut graph = FrameGraph::new();
    let leaves = library_buffer(&mut graph, LibraryResource::TerrainLeaves);
    let indirect = library_buffer(&mut graph, LibraryResource::TerrainIndirect);
    let depth = graph.image("depth", ImageDesc::depth_2d(env.depth_format, width, height, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT), ResourceClass::Transient);
    let target = match output {
        TerrainOutput::Present => graph.image("target", ImageDesc::color_2d(env.color_format, width, height, vk::ImageUsageFlags::COLOR_ATTACHMENT), ResourceClass::Backbuffer),
        TerrainOutput::Readback => graph.image(
            LibraryResource::ReadbackTarget.name(),
            ImageDesc::color_2d(env.color_format, width, height, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC),
            ResourceClass::External,
        ),
    };
    graph
        .pass("terrain lod", PassKind::Compute, LibraryPass::TerrainLod)
        .access(leaves, Role::StorageBufferWrite(ShaderStage::Compute))
        .access(indirect, Role::StorageBufferWrite(ShaderStage::Compute))
        .id();
    graph
        .pass("terrain", PassKind::Graphics, LibraryPass::TerrainDraw)
        .access(leaves, Role::StorageBufferRead(ShaderStage::Vertex))
        .access(indirect, Role::IndirectBuffer)
        .color(target, LoadOp::Clear)
        .depth(depth, LoadOp::Clear)
        .id();
    if env.units {
        let instances = library_buffer(&mut graph, LibraryResource::UnitInstances);
        let visible = library_buffer(&mut graph, LibraryResource::UnitVisible);
        let draws = library_buffer(&mut graph, LibraryResource::UnitIndirect);
        graph.pass("units reset", PassKind::Transfer, LibraryPass::UnitsReset).access(draws, Role::TransferDst).id();
        graph
            .pass("units cull", PassKind::Compute, LibraryPass::UnitsCull)
            .access(instances, Role::StorageBufferRead(ShaderStage::Compute))
            .access(visible, Role::StorageBufferWrite(ShaderStage::Compute))
            .access(draws, Role::StorageBufferWrite(ShaderStage::Compute))
            .id();
        // Declared after the terrain draw: a second writer of the target
        // and the depth buffer, ordered after it.
        graph
            .pass("units", PassKind::Graphics, LibraryPass::UnitsDraw)
            .access(instances, Role::StorageBufferRead(ShaderStage::Vertex))
            .access(visible, Role::StorageBufferRead(ShaderStage::Vertex))
            .access(draws, Role::IndirectBuffer)
            .color(target, LoadOp::Load)
            .depth(depth, LoadOp::Load)
            .id();
        let counts = library_buffer(&mut graph, LibraryResource::UnitCounts);
        graph.pass("units count", PassKind::Transfer, LibraryPass::UnitsCount).access(draws, Role::TransferSrc).discard(counts, Role::TransferDst).id();
        graph.pass("units count read", PassKind::Transfer, LibraryPass::HostRead).access(counts, Role::HostRead).root().id();
    }
    if env.water {
        // After the units (a third writer of the target), testing against
        // the depth they leave without writing it.
        graph.pass("water", PassKind::Graphics, LibraryPass::WaterDraw).color(target, LoadOp::Load).depth_read(depth).id();
    }
    match output {
        TerrainOutput::Present => {
            graph.pass("present", PassKind::Present, LibraryPass::Present).present(target).root().id();
        }
        TerrainOutput::Readback => {
            let readback = library_buffer(&mut graph, LibraryResource::ReadbackBuffer);
            graph.pass("copy", PassKind::Transfer, LibraryPass::ReadbackCopy).access(target, Role::TransferSrc).discard(readback, Role::TransferDst).id();
            graph.pass("host read", PassKind::Transfer, LibraryPass::HostRead).access(readback, Role::HostRead).root().id();
        }
    }
    graph
}

/// A library buffer as a graph declares it: `External`, size 0.
pub fn library_buffer<R>(graph: &mut FrameGraph<R>, r: LibraryResource) -> ResourceId {
    graph.buffer(r.name(), BufferDesc { size: 0, usage: vk::BufferUsageFlags::empty() }, ResourceClass::External)
}

impl TerrainApp {
    /// Load `wasm` and wire its declaration for `env`.
    fn load_module(&mut self, wasm: &[u8], env: &LibraryEnv) -> Result<ModuleState> {
        let engine = self.engine.get_or_insert_with(crate::module::engine).clone();
        let mut module = RenderModule::load(&engine, wasm)?;
        let d = self.declare_module(&mut module, env)?;
        Ok(ModuleState {
            module,
            declared_for: (env.extent, env.color_format),
            graph: d.graph,
            resources: d.resources,
            programs: d.programs,
            uniform_bytes: d.uniform_bytes,
            frame: ModuleFrame::default(),
            failed: false,
            frames: 0,
        })
    }

    /// Destroy programs a replaced declaration used, once the GPU is done
    /// with them (a reload or resize: waiting for idle is fine there).
    fn retire(&self, programs: Vec<Option<(FullscreenProgram, usize)>>) {
        let programs: Vec<_> = programs.into_iter().flatten().collect();
        if programs.is_empty() {
            return;
        }
        if let Some(gpu) = &self.gpu {
            gpu.wait_idle();
            for (mut program, _) in programs {
                program.destroy(gpu);
            }
        }
    }

    /// The module's graph for `env`, decoded and wired, with its fullscreen
    /// programs reflected, checked and built.
    fn declare_module(&self, module: &mut RenderModule, env: &LibraryEnv) -> Result<ModuleGraph> {
        let flags = if env.units { DECLARE_UNITS } else { 0 } | if self.output == TerrainOutput::Readback { DECLARE_READBACK } else { 0 } | if env.water { DECLARE_WATER } else { 0 };
        let declared = module
            .declare(env.extent.width, env.extent.height, env.color_format.as_raw() as u32, flags)?
            .with_context(|| format!("the module declines a {}x{} {:?} target", env.extent.width, env.extent.height, env.color_format))?;
        let graph = ae_framegraph::descriptor::decode(&declared.graph).map_err(|e| anyhow::anyhow!("graph declaration: {e}"))?;
        let programs = ProgramTable::decode(&declared.programs).map_err(|e| anyhow::anyhow!("program table: {e}"))?;
        let (graph, wiring) = self.wire(graph, env, |&payload| {
            let program = programs.get(payload as usize).with_context(|| format!("payload {payload} names no program (the table has {})", programs.len()))?;
            match *program {
                Program::Library(name) => LibraryPass::from_name(name).map(|lib| (payload, ScenePassSource::Library(lib))).with_context(|| format!("no library pass `{name}`")),
                Program::Fullscreen { .. } => Ok((payload, ScenePassSource::Fullscreen)),
            }
        })?;
        // Build each fullscreen program for the one pass that runs it (its
        // descriptor sets are per frame slot, rewritten when the pass
        // records, so two passes cannot share them in one frame).
        let gpu = self.gpu.as_ref().context("not created")?;
        let mut built: Vec<Option<(FullscreenProgram, usize)>> = (0..programs.len()).map(|_| None).collect();
        let mut shapes: Vec<Option<(Vec<vk::Format>, usize)>> = vec![None; programs.len()];
        let mut runner: Vec<Option<&str>> = vec![None; programs.len()];
        let result = (|| -> Result<usize> {
            for pass in &graph.passes {
                let PassWork::Fullscreen { sampled } = &pass.record.work else { continue };
                let formats: Vec<vk::Format> = pass
                    .color_attachments()
                    .map(|i| match graph.resources[i.resource.index()].desc {
                        ae_framegraph::ResourceDesc::Image(image) => image.format,
                        ae_framegraph::ResourceDesc::Buffer(_) => vk::Format::UNDEFINED,
                    })
                    .collect();
                let slot = pass.record.program as usize;
                if let Some(first) = runner[slot] {
                    let label = match programs[slot] {
                        Program::Fullscreen { label, .. } => label,
                        Program::Library(name) => name,
                    };
                    anyhow::bail!("passes `{first}` and `{}` both run fullscreen program {slot} (`{label}`); a fullscreen program runs in one pass (push it once per pass)", pass.name);
                }
                runner[slot] = Some(&pass.name);
                shapes[slot] = Some((formats, sampled.len()));
            }
            let mut offset = 0;
            for (slot, program) in programs.iter().enumerate() {
                let Program::Fullscreen { label, spirv, uniform_bytes } = *program else { continue };
                let Some((formats, sampled)) = &shapes[slot] else { continue };
                built[slot] = Some((FullscreenProgram::new(gpu, label, spirv, uniform_bytes, *sampled, formats, self.frames_in_flight)?, offset));
                offset += uniform_bytes as usize;
            }
            Ok(offset)
        })();
        match result {
            Ok(uniform_bytes) => Ok(ModuleGraph { graph, resources: wiring.resources, programs: built, uniform_bytes }),
            Err(error) => {
                for (mut program, _) in built.into_iter().flatten() {
                    program.destroy(gpu);
                }
                Err(error)
            }
        }
    }

    /// What the library needs to agree with for `target`.
    fn env(&self, target: TargetInfo) -> Result<LibraryEnv> {
        let renderer = self.renderer.as_ref().context("terrain renderer not created")?;
        Ok(LibraryEnv { color_format: renderer.formats().0, depth_format: DEPTH_FORMAT, extent: target.extent, units: self.unit_renderer.is_some(), water: self.water && renderer.has_water() })
    }

    /// Size the library resources of `graph`, check its library passes and
    /// turn its payloads into [`ScenePass`]es.
    fn wire<R>(&self, mut graph: FrameGraph<R>, env: &LibraryEnv, source: impl Fn(&R) -> Result<(u32, ScenePassSource)>) -> Result<(FrameGraph<ScenePass>, Wiring)> {
        let units = self.unit_renderer.as_ref();
        let terrain = self.renderer.as_ref();
        let offscreen = self.offscreen.as_ref();
        let resources = library::resolve(&mut graph, env, |r| match r {
            LibraryResource::TerrainLeaves => terrain.map(|t| t.leaves_desc()),
            LibraryResource::TerrainIndirect => terrain.map(|t| t.indirect_desc()),
            LibraryResource::UnitInstances => units.map(|u| u.instances_desc()),
            LibraryResource::UnitVisible => units.map(|u| u.visible_desc()),
            LibraryResource::UnitIndirect => units.map(|u| u.indirect_desc()),
            LibraryResource::UnitCounts => units.map(|u| u.counts_desc()),
            LibraryResource::ReadbackBuffer => offscreen.map(|o| BufferDesc { size: o.buffer.size, usage: vk::BufferUsageFlags::TRANSFER_DST }),
            LibraryResource::ReadbackTarget => None,
        })?;
        if resources.contains(&Some(LibraryResource::ReadbackTarget)) && self.offscreen.as_ref().is_none_or(|o| o.extent != env.extent) {
            anyhow::bail!("`readback.target` needs an offscreen target of the graph's extent (create_offscreen)");
        }
        let mut passes = Vec::with_capacity(graph.passes.len());
        for (index, pass) in graph.passes.iter().enumerate() {
            let (program, source) = source(&pass.record).with_context(|| format!("pass `{}`", pass.name))?;
            let work = match source {
                ScenePassSource::Library(lib) => {
                    library::check(&graph, index, lib, &resources, env)?;
                    match lib {
                        LibraryPass::ReadbackCopy => {
                            let image = pass.intents.iter().find(|i| i.role == Role::TransferSrc).map(|i| i.resource).context("readback image")?;
                            let buffer = pass.intents.iter().find(|i| resources[i.resource.index()] == Some(LibraryResource::ReadbackBuffer)).map(|i| i.resource).context("readback buffer")?;
                            PassWork::Readback { image, buffer }
                        }
                        lib => PassWork::Library(lib),
                    }
                }
                ScenePassSource::Fullscreen => {
                    let name = &pass.name;
                    anyhow::ensure!(pass.kind == PassKind::Graphics, "fullscreen pass `{name}` must be a Graphics pass");
                    let mut sampled = Vec::new();
                    for intent in &pass.intents {
                        let r = &graph.resources[intent.resource.index()];
                        match intent.role {
                            Role::ColorAttachment(_) => {}
                            Role::Sampled(ShaderStage::Fragment) if r.desc.is_image() => sampled.push(intent.resource),
                            role => anyhow::bail!("fullscreen pass `{name}` uses `{}` as {role:?}; it takes colour attachments and fragment-sampled images", r.name),
                        }
                    }
                    anyhow::ensure!(pass.color_attachments().count() > 0, "fullscreen pass `{name}` has no colour attachment");
                    PassWork::Fullscreen { sampled }
                }
            };
            passes.push(ScenePass { program, work });
        }
        let graph = FrameGraph {
            resources: graph.resources,
            passes: graph.passes.into_iter().zip(passes).map(|(p, record)| ae_framegraph::Pass { name: p.name, kind: p.kind, scope: p.scope, intents: p.intents, conditional: p.conditional, record }).collect(),
            roots: graph.roots,
            plan_version: graph.plan_version,
        };
        Ok((graph, Wiring { resources }))
    }
}
