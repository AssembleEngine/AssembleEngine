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
use super::renderer::{
    FrameParams, FrameStats, LodView, TerrainInfo, TerrainRenderer, TerrainUpdate, VisionGrid, Wireframe,
};
use super::sun;
use super::tileset::Tileset;
use crate::bloom::{Bloom, BloomPass};
use crate::colorspace::{Gamut, WorkingSpace};
use crate::decals::{DecalAtlas, DecalSet, DecalsSetup};
use crate::device::GpuDevice;
use crate::encode::{DisplayLevels, Encode, OutputEncoding};
use crate::executor::{Binding, BufferBinding, ImageBinding, PassContext, Recorder, SharedRecorder};
use crate::exposure::{Exposure, ExposureFrame};
use crate::fullscreen::FullscreenProgram;
use crate::library::{self, LibraryEnv, LibraryPass, LibraryResource, ResourceMap};
use crate::lights::{Light, LightFrame, LightRenderer, LightStats};
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::module::DeclareTarget;
use crate::module::{ModuleFrame, RenderModule};
use crate::particles::{ParticleFrame, ParticleRenderer, ParticlesSetup};
use crate::thread::{RenderApp, TargetInfo};
use crate::tonemap::{ToneParams, Tonemap};
use crate::units::{UnitFrame, UnitRenderer, UnitsSetup};
use ae_framegraph::{
    BufferDesc, FrameGraph, ImageDesc, LoadOp, PassKind, ResourceClass, ResourceId, Role, ShaderStage,
};
use ae_render_abi::module::{
    DECLARE_PARTICLES, DECLARE_READBACK, DECLARE_SHADOWS, DECLARE_UNITS, DECLARE_WATER, FrameInput, Program,
    ProgramTable,
};
use anyhow::{Context as _, Result};
use ash::vk;
use glam::Vec2;
use gpu_allocator::MemoryLocation;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
/// The HDR scene's format ([`ae_render_abi::module::SCENE_FORMAT`]).
pub const SCENE_FORMAT: vk::Format = vk::Format::from_raw(ae_render_abi::module::SCENE_FORMAT as i32);
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
        Camera {
            target,
            ground: self.height_at(target),
            distance: w.max(h) * 0.6,
            yaw: 0.0,
            pitch: 0.75,
            fov_y: 0.8,
        }
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
    /// Hold every animation clock: water, decals, particles and (with the
    /// sim thread stopped) unit animation. The app freezes the clocks at
    /// the frame the pause begins and skips the paused interval on resume.
    pub paused: bool,
    /// Add the bloom to the scene before the tone curve (the app's
    /// [`ToneParams::bloom`] when on; 0 when off).
    pub bloom: bool,
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
        Self {
            grid,
            latest: Mutex::new(None),
        }
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
    Readback {
        image: ResourceId,
        buffer: ResourceId,
    },
    /// A module's fullscreen program (by payload), sampling `sampled` in
    /// order.
    Fullscreen {
        sampled: Vec<ResourceId>,
    },
    /// [`LibraryPass::ParticlesDraw`] with its colour attachment, the
    /// depth buffer it samples, and the glow target it writes.
    Particles {
        color: ResourceId,
        depth: ResourceId,
        glow: ResourceId,
    },
    /// [`LibraryPass::Tonemap`] with the scene image it samples, the bloom
    /// (the last `bloom.blur_y` output), if the graph has one, and the
    /// exposure image (the eased EV), if the graph has one.
    Tonemap {
        scene: ResourceId,
        bloom: Option<ResourceId>,
        exposure: Option<ResourceId>,
    },
    /// A `post.bloom` pass: its sampled input and colour attachment, and
    /// the exposure image the downsample scales by.
    Bloom {
        pass: BloomPass,
        input: ResourceId,
        output: ResourceId,
        exposure: Option<ResourceId>,
    },
    /// [`LibraryPass::ExposureMeter`] with the scene image it meters, the
    /// depth it rebuilds world positions from (to undo the fog of war's
    /// dimming), and the glow (the light the bloom shows), each if the graph
    /// has the meter sample it.
    ExposureMeter {
        scene: ResourceId,
        depth: Option<ResourceId>,
        glow: Option<ResourceId>,
    },
    /// [`LibraryPass::PostEncode`] with the display image it encodes.
    Encode {
        display: ResourceId,
    },
}

/// What a module's declaration is for: the target's size and (format,
/// colour space) pair, and the display spec `declare` was given (a pair
/// change or a levels change redeclares it, exactly as the native graph is
/// rebuilt for a new target).
#[derive(Clone, Copy, Debug, PartialEq)]
struct DeclareKey {
    extent: vk::Extent2D,
    format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    encoding: OutputEncoding,
    /// Paper white and peak as bits (an exact comparison).
    paper_white: u32,
    peak: u32,
    /// The working space and output gamut `declare` was given (constant
    /// within a run today, but part of what the module saw).
    working: WorkingSpace,
    gamut: Gamut,
}

/// The render module driving the graph, with its wired declaration.
struct ModuleState {
    module: RenderModule,
    /// What the declaration is for; `None` after a rebuild's destroy (the
    /// module stays loaded and redeclares on the next frame).
    declared_for: Option<DeclareKey>,
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

/// The window's HDR settings, kept up to date by the main thread (M13 stage
/// 5): the user's overrides and SDL's headroom. The reference and the
/// derived peak come from the encoding the swapchain's colour space selects.
#[derive(Clone, Copy, Debug)]
pub struct HdrSettings {
    /// `--paper-white` (nits); `None` is the encoding's reference.
    pub paper_white: Option<f32>,
    /// `--peak` (nits of the signal, the same frame as the paper white);
    /// `None` is the paper white times SDL's headroom.
    pub peak: Option<f32>,
    /// SDL's headroom above SDR white (1.0 when not HDR).
    pub headroom: f32,
    /// SDL's peak for the display in the panel's nits (`SDR white ×
    /// headroom`, the compositor's advertised maximum; 0 when SDL reports
    /// none). For the log only: the signal's peak is relative to the
    /// signal's reference, see [`Self::levels`].
    pub display_peak: f32,
}

impl Default for HdrSettings {
    fn default() -> Self {
        Self {
            paper_white: None,
            peak: None,
            headroom: 1.0,
            display_peak: 0.0,
        }
    }
}

impl HdrSettings {
    /// The levels for `encoding` (the swapchain's), in the signal's nits.
    /// The paper white is the WSI's measured reference (203 nits for both PQ
    /// and scRGB here) unless overridden; the peak is the paper white times
    /// SDL's headroom.
    pub fn levels(&self, encoding: OutputEncoding) -> DisplayLevels {
        let reference = match encoding {
            OutputEncoding::Hdr10Pq => crate::encode::HDR10_REFERENCE_NITS,
            OutputEncoding::ScRgb => crate::encode::SCRGB_REFERENCE_NITS,
            OutputEncoding::Sdr => crate::encode::SCRGB_REFERENCE_NITS,
        };
        let paper_white = self.paper_white.unwrap_or(reference).max(1.0);
        // The compositor maps the signal's reference white onto its own
        // (KWin 6.8's `ColorDescription::toOther` scales by
        // `other.referenceLuminance() / referenceLuminance()`): signal 203
        // shows at the desktop's 248 here. So the signal's peak is the
        // headroom over the signal's paper white (203 × 1.778 = 361, which
        // shows at 441), not SDL's absolute 441: that would land at 539 and
        // KWin's perceptual tonemapper would compress it again on top of
        // the curve's shoulder.
        let peak = self.peak.unwrap_or(paper_white * self.headroom).max(paper_white);
        DisplayLevels { paper_white, peak }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainOutput {
    Present,
    /// Render into an app-owned image and copy it to a host buffer
    /// ([`TerrainApp::readback`]); for tests and captures.
    Readback,
}

/// The installed graph's library resources (by resource index), its readback
/// image, if any, and its glow (the bloom's input).
struct Wiring {
    resources: ResourceMap,
    glow: Option<ResourceId>,
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
    particles: Option<ParticlesSetup>,
    particle_renderer: Option<ParticleRenderer>,
    /// `post.tonemap`, for the output's format.
    tonemap: Option<Tonemap>,
    /// The glow's three bloom passes (quarter size), when the graph has a
    /// glow.
    bloom: Option<Bloom>,
    /// The metering and adaptation passes, and the eased EV they hand the
    /// tonemap.
    exposure: Option<Exposure>,
    /// When the previous frame adapted the exposure (wall clock, pauses
    /// included: the eye keeps adapting while the game is paused).
    last_adapt_at: Option<Instant>,
    tone: ToneParams,
    /// How the output image encodes the display's light (SDR keeps the
    /// tonemap writing the target).
    encoding: OutputEncoding,
    /// The display's paper white and peak, in nits (HDR only).
    levels: DisplayLevels,
    /// The window's HDR settings (the client keeps them current); `None`
    /// keeps [`Self::with_display`]'s levels.
    hdr: Option<Arc<Mutex<HdrSettings>>>,
    /// The scene's working space (M13 stage 3): BT.709 unless set.
    working: WorkingSpace,
    /// The output gamut; `None` is the encoding's default (P3 for HDR10).
    display_gamut: Option<Gamut>,
    /// `post.encode`, for the output's format (HDR only).
    encode: Option<Encode>,
    /// The configured bloom strength, kept while [`TerrainView::bloom`] is
    /// switched off.
    tone_bloom: f32,
    /// The clustered lights (made before the renderers that read them).
    lights: Option<LightRenderer>,
    /// Lights the app was given (`with_lights`), listed every frame.
    fixed_lights: Vec<Light>,
    /// This frame's list (kept to reuse its allocation).
    frame_lights: Vec<Light>,
    light_stats: Arc<Mutex<LightStats>>,
    /// Every light at every pixel (`with_unculled_lights`).
    unculled_lights: bool,
    /// The output's format (the swapchain's or the readback image's).
    output_format: vk::Format,
    /// The backbuffer's colour space (the graph's redeclare key; an
    /// offscreen app has none, so `SRGB_NONLINEAR`).
    output_space: vk::ColorSpaceKHR,
    decals: Option<DecalsSetup>,
    /// The live decals (kept across renderer re-creation).
    decal_set: Option<DecalSet>,
    /// Per slot: the decal version written into it.
    decals_written: Vec<Option<u64>>,
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
    /// Its global sampler-array index once registered (the post passes use
    /// it instead of the object).
    sampler_handle: Option<crate::bindless::SamplerHandle>,
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
    /// The readback animation's frame index (held while paused), and when
    /// the app was made (the presenting wall clock).
    frames: u64,
    started: Instant,
    /// When the view was paused, and how long it has been paused before
    /// this pause (skipped by the wall clock on resume).
    paused_at: Option<Instant>,
    paused_for: Duration,
}

impl TerrainApp {
    pub fn new(source: Arc<TerrainSource>, output: TerrainOutput) -> Self {
        let view = TerrainView {
            camera: source.overview_camera(),
            tau: 2.0,
            wireframe: Wireframe::Off,
            lod_colors: false,
            fog: true,
            frozen: None,
            module_flags: 0,
            tileset: true,
            weights: false,
            vision: true,
            water: true,
            time: None,
            shadows: true,
            paused: false,
            bloom: true,
        };
        let tone = ToneParams::default();
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
            particles: None,
            particle_renderer: None,
            tonemap: None,
            bloom: None,
            exposure: None,
            last_adapt_at: None,
            tone_bloom: tone.bloom,
            tone,
            encoding: OutputEncoding::Sdr,
            levels: DisplayLevels::default(),
            hdr: None,
            working: WorkingSpace::Bt709,
            display_gamut: None,
            encode: None,
            lights: None,
            fixed_lights: Vec::new(),
            frame_lights: Vec::new(),
            light_stats: Arc::default(),
            unculled_lights: false,
            output_format: vk::Format::UNDEFINED,
            output_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            decals: None,
            decal_set: None,
            decals_written: Vec::new(),
            last_slot: 0,
            offscreen: None,
            wiring: None,
            engine: None,
            module: None,
            gpu: None,
            frames_in_flight: 1,
            sampler: vk::Sampler::null(),
            sampler_handle: None,
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
            paused_at: None,
            paused_for: Duration::ZERO,
        }
    }

    /// Draw the water surface (a `water.draw` pass after the units) on a map
    /// with a water level; on a map without one this does nothing.
    pub fn with_water(mut self) -> Self {
        self.water = self.source.info.water_level.is_some();
        self
    }

    /// The map the app draws.
    pub fn source(&self) -> &Arc<TerrainSource> {
        &self.source
    }

    /// Lights listed every frame besides the bursts' (fixed in the world).
    /// Their colours are BT.709 (the `Light` constructors normalise them
    /// there); the app converts them into the working space each frame.
    pub fn with_lights(mut self, lights: Vec<Light>) -> Self {
        self.fixed_lights = lights;
        self
    }

    /// Light by every light everywhere, without the clusters: the
    /// reference they must match (tests, measurements).
    pub fn with_unculled_lights(mut self) -> Self {
        self.unculled_lights = true;
        self
    }

    /// What the clustered lights cut, from the last completed frame.
    pub fn light_stats(&self) -> Arc<Mutex<LightStats>> {
        self.light_stats.clone()
    }

    /// The camera exposure (EV100), the bloom's strength and the sun.
    pub fn with_tone(mut self, tone: ToneParams) -> Self {
        self.tone = tone;
        self.tone_bloom = tone.bloom;
        self
    }

    /// The output's encoding and the display's levels (M13): SDR (the
    /// default) keeps the tonemap writing the sRGB target; HDR adds the
    /// display intermediate and `post.encode`. An offscreen app uses this
    /// encoding directly; a present-path app derives it from the swapchain's
    /// colour space and takes levels from [`Self::with_hdr`] (M13 stage 5).
    pub fn with_display(mut self, encoding: OutputEncoding, levels: DisplayLevels) -> Self {
        assert!(
            levels.paper_white > 0.0 && levels.peak >= levels.paper_white,
            "display levels: paper white {:.3} nits, peak {:.3}",
            levels.paper_white,
            levels.peak
        );
        self.encoding = encoding;
        self.levels = levels;
        self
    }

    /// Take the display's levels and HDR mode from the client (M13 stage
    /// 5): a present-path app derives its encoding from the swapchain's
    /// colour space and its levels from these settings.
    pub fn with_hdr(mut self, settings: Arc<Mutex<HdrSettings>>) -> Self {
        self.hdr = Some(settings);
        self
    }

    /// The output gamut (M13 stage 3): the display's primaries. Default:
    /// [`OutputEncoding::default_gamut`] (BT.709 for SDR, P3 for HDR10,
    /// BT.2020 for scRGB).
    pub fn with_display_gamut(mut self, gamut: Gamut) -> Self {
        self.display_gamut = Some(gamut);
        self
    }

    /// The space the scene is lit in (M13 stage 3): BT.709 (the default) or
    /// BT.2020. Authored colours are converted into it, and the output
    /// converts back; only the default keeps SDR captures bit-identical.
    pub fn with_working_space(mut self, working: WorkingSpace) -> Self {
        self.working = working;
        self
    }

    /// The output gamut in effect.
    pub fn gamut(&self) -> Gamut {
        self.display_gamut.unwrap_or_else(|| self.encoding.default_gamut())
    }

    /// The backbuffer's colour space (the module redeclare key).
    fn output_format_space(&self) -> vk::ColorSpaceKHR {
        self.output_space
    }

    /// What a module's declaration is for (`declare`'s key).
    fn declare_key(&self, env: &LibraryEnv) -> DeclareKey {
        DeclareKey {
            extent: env.extent,
            format: env.color_format,
            color_space: self.output_format_space(),
            encoding: env.encoding,
            paper_white: env.levels.paper_white.to_bits(),
            peak: env.levels.peak.to_bits(),
            working: env.working,
            gamut: env.gamut,
        }
    }

    /// The window's HDR settings (defaults when the client gave none).
    fn hdr_settings(&self) -> HdrSettings {
        self.hdr
            .as_ref()
            .map(|h| *h.lock().unwrap_or_else(|p| p.into_inner()))
            .unwrap_or_default()
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

    /// Draw the particles of `setup.feed` (`particles.sim` and
    /// `particles.draw`, after the water).
    pub fn with_particles(mut self, setup: ParticlesSetup) -> Self {
        self.particles = Some(setup);
        self
    }

    /// Paint the decals of `setup.feed` on the terrain (inside the terrain
    /// draw: no pass of their own).
    pub fn with_decals(mut self, setup: DecalsSetup) -> Self {
        self.decal_set = Some(DecalSet::new(setup.capacity));
        self.decals = Some(setup);
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

    /// Cells the last frame uploaded into the units' animation table, and
    /// the render thread's milliseconds for it.
    pub fn last_anim_upload(&self) -> (u32, f64) {
        self.unit_renderer.as_ref().map_or((0, 0.0), |u| u.anim_upload())
    }

    /// Per model class, the units the cull put in each shadow cascade in
    /// the last frame (after `run_headless`); for tests and tools.
    pub fn last_unit_cascade_counts(&mut self) -> Vec<[u32; crate::shadow::CASCADES]> {
        let slot = self.last_slot;
        self.unit_renderer
            .as_mut()
            .map_or(Vec::new(), |units| units.read_cascade_counts(slot))
    }

    /// Each row's skinning palette in the last frame (empty for a dart),
    /// once the device is idle (after `run_headless`); for tests and tools.
    pub fn last_unit_poses(&mut self, gpu: &GpuDevice) -> Result<Vec<Vec<[f32; 12]>>> {
        let slot = self.last_slot;
        match self.unit_renderer.as_mut() {
            Some(units) => units.read_poses(gpu, slot),
            None => Ok(Vec::new()),
        }
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

    /// The last offscreen frame (row-major, after `run_headless`): sRGB 8-bit
    /// for an SDR output, 10-10-10-2 packed for HDR10 and 16-bit float for
    /// scRGB ([`OutputEncoding::readback_format`]).
    pub fn readback(&mut self) -> Option<Vec<u8>> {
        let offscreen = self.offscreen.as_mut()?;
        let bytes =
            (offscreen.extent.width as u64 * offscreen.extent.height as u64 * self.encoding.bytes_per_pixel()) as usize;
        offscreen.buffer.mapped().map(|data| data[..bytes].to_vec())
    }

    /// The camera matrices of `view` for the current target, at `now` (the
    /// frame the pause froze, when paused).
    fn params(&self, view: &TerrainView, now: Instant) -> FrameParams {
        let time = view.time.unwrap_or_else(|| match self.output {
            TerrainOutput::Readback => self.frames as f32 / 60.0,
            TerrainOutput::Present => now
                .saturating_duration_since(self.started)
                .saturating_sub(self.paused_for)
                .as_secs_f32(),
        });
        // Effects follow the sim when presenting a live unit stream
        // (readback keeps `time`: deterministic captures and tests).
        let stream = match (self.output, view.time) {
            (TerrainOutput::Present, None) => self.units.as_ref().and_then(|u| u.feed.latest()),
            _ => None,
        };
        let effect_time = stream.map_or(time, |s| s.sim_time(s.alpha(now)) as f32);
        let aspect = self.extent.width as f32 / self.extent.height.max(1) as f32;
        let mut camera = view.camera;
        camera.ground = self.source.height_at(camera.target);
        let eye = camera.eye(|p| self.source.height_at(p));
        let view_proj = camera.proj(aspect) * camera.view(eye);
        let live = LodView {
            pos: eye,
            view_proj,
            proj_scale: camera.proj_scale(self.extent.height as f32),
        };
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
            effect_time,
            shadows: view.shadows && self.shadows,
            // y: the working space is BT.2020 (the shaders' `wide` flag).
            photometry: [
                self.tone.exposure(),
                if self.working.is_wide() { 1.0 } else { 0.0 },
                self.tone.sun_illuminance,
                crate::lighting::SKY_ILLUMINANCE,
            ],
            working: self.working,
            // The units' shadows when the graph draws them.
            // (and the particles' coloured shadows when it draws those).
            unit_shadows: (view.shadows && self.shadows && self.graph_has(LibraryResource::ShadowMap)).then(|| {
                crate::shadow::Cascades {
                    tinted: self.graph_has(LibraryResource::ParticleShadows),
                    ..crate::shadow::cascades(
                        eye,
                        (camera.target3() - eye).normalize(),
                        ((camera.fov_y * 0.5).tan() * aspect, (camera.fov_y * 0.5).tan()),
                        camera.distance,
                        sun::direction(),
                    )
                }
            }),
        }
    }

    /// Whether the graph in use declares library image `r` (the units'
    /// shadow map, the particles' coloured shadows: it then draws and
    /// samples it).
    fn graph_has(&self, r: LibraryResource) -> bool {
        self.wiring.as_ref().is_some_and(|w| w.resources.contains(&Some(r)))
    }
}

/// The terrain's fog (colour and density) for other passes, as the terrain
/// renderer sets it.
/// Distance fog as the sky's radiance and its density.
fn fog(info: &TerrainInfo, on: bool, working: WorkingSpace) -> [f32; 4] {
    let density = if on {
        1.0 / (info.cells.0.max(info.cells.1) as f32 * info.cell_size * 1.2)
    } else {
        0.0
    };
    let r = crate::lighting::sky_radiance(working);
    [r[0], r[1], r[2], density]
}

/// The view a frozen LOD would keep (call on the main thread with the
/// target's size).
pub fn lod_view(source: &TerrainSource, camera: &Camera, width: u32, height: u32) -> LodView {
    let mut camera = *camera;
    camera.ground = source.height_at(camera.target);
    let eye = camera.eye(|p| source.height_at(p));
    let view_proj = camera.proj(width as f32 / height.max(1) as f32) * camera.view(eye);
    LodView {
        pos: eye,
        view_proj,
        proj_scale: camera.proj_scale(height as f32),
    }
}

impl RenderApp for TerrainApp {
    type Pass = ScenePass;

    fn create(
        &mut self,
        gpu: &Arc<GpuDevice>,
        bindless: &mut crate::bindless::Bindless,
        target: TargetInfo,
        frames_in_flight: usize,
    ) -> Result<()> {
        // A rebuild gives the previous build's global-set handles back
        // before their device objects go (M15).
        self.retire_handles(bindless);
        if let Some(mut old) = self.tonemap.take() {
            old.retire(bindless);
            old.destroy(gpu);
        }
        if let Some(mut old) = self.encode.take() {
            old.retire(bindless);
            old.destroy(gpu);
        }
        if let Some(mut old) = self.bloom.take() {
            old.retire(bindless);
            old.destroy(gpu);
        }
        if let Some(mut old) = self.exposure.take() {
            old.destroy(gpu);
        }
        if let Some(mut old) = self.particle_renderer.take() {
            old.destroy(gpu);
        }
        if let Some(mut old) = self.unit_renderer.take() {
            old.destroy(gpu);
        }
        if let Some(mut old) = self.renderer.take() {
            old.destroy(gpu);
        }
        if let Some(mut old) = self.lights.take() {
            old.destroy(gpu);
        }
        self.gpu = Some(gpu.clone());
        self.frames_in_flight = frames_in_flight;
        // The module programs' own sets still bind a combined image
        // sampler; the registry owns the object and deduplicates it.
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let handle = bindless.sampler(&info)?;
        self.sampler = bindless.sampler_object(handle);
        self.sampler_handle = Some(handle);
        self.create_output(gpu, bindless, target, frames_in_flight)?;
        self.bloom = Some(Bloom::new(gpu, bindless, frames_in_flight).context("bloom")?);
        self.exposure = Some(Exposure::new(gpu, bindless, frames_in_flight).context("exposure")?);
        self.last_adapt_at = None;
        let mut lights = LightRenderer::new(gpu, bindless, frames_in_flight).context("lights")?;
        lights.set_unculled(self.unculled_lights);
        // The scene's draws write the HDR scene image.
        let color = SCENE_FORMAT;
        let s = &self.source;
        let shadows = self.shadows.then(|| {
            sun::shadow_heights(
                &s.heightmap,
                s.info.cell_size,
                s.info.height_scale,
                s.info.height_offset,
                sun::direction(),
            )
        });
        let default_atlas;
        let decals = match &self.decals {
            Some(setup) => {
                let atlas: &DecalAtlas = match &setup.atlas {
                    Some(a) => a,
                    None => {
                        default_atlas = crate::decals::default_atlas(128);
                        &default_atlas
                    }
                };
                Some((atlas, setup.capacity))
            }
            None => None,
        };
        self.decals_written.clear();
        let terrain = TerrainRenderer::new(
            gpu,
            bindless,
            &s.heightmap,
            &s.pyramid,
            s.info,
            s.materials.as_deref(),
            self.tileset.as_deref(),
            self.vision.as_ref().map(|f| f.grid),
            shadows.as_deref(),
            self.shadows && self.units.is_some(),
            decals,
            &lights,
            color,
            DEPTH_FORMAT,
            frames_in_flight,
        )
        .context("terrain renderer")?;
        if let Some(setup) = &self.units {
            // The looks' colours carry their own gamut tags; the renderer
            // converts them into the working space at upload (stage 4).
            self.unit_renderer = Some(
                UnitRenderer::new(
                    gpu,
                    bindless,
                    &terrain,
                    &lights,
                    setup,
                    self.working,
                    color,
                    DEPTH_FORMAT,
                    frames_in_flight,
                )
                .context("unit renderer")?,
            );
        }
        if let Some(setup) = &self.particles {
            // So do the effects' colours and their bursts' light tints.
            self.particle_renderer = Some(
                ParticleRenderer::new(
                    gpu,
                    bindless,
                    &terrain,
                    &lights,
                    setup,
                    self.working,
                    color,
                    frames_in_flight,
                )
                .context("particle renderer")?,
            );
        }
        self.renderer = Some(terrain);
        self.lights = Some(lights);
        Ok(())
    }

    fn hdr_metadata(&self) -> Option<(DisplayLevels, Gamut)> {
        self.encoding.uses_display().then(|| (self.levels, self.gamut()))
    }

    fn declare(
        &mut self,
        target: TargetInfo,
        bindless: &mut crate::bindless::Bindless,
    ) -> Result<(FrameGraph<ScenePass>, Option<ResourceId>)> {
        self.extent = target.extent;
        let env = self.env(target)?;
        let pending = self.pending_module.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(wasm) = pending {
            match self.load_module(&wasm, &env, bindless) {
                Ok(state) => {
                    if let Some(old) = self.module.replace(state) {
                        self.retire(old.programs, bindless);
                    }
                    let mut status = self.module_status.lock().unwrap_or_else(|p| p.into_inner());
                    status.loads += 1;
                    status.active = true;
                }
                Err(error) if self.module.is_none() => return Err(error.context("render module")),
                Err(error) => self
                    .module_status
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .errors
                    .push(format!("reload rejected, the running module stays: {error:#}")),
            }
        }
        let (graph, resources, glow) = match self.module.take() {
            Some(mut state) => {
                if state.declared_for != Some(self.declare_key(&env)) {
                    let declared = self.declare_module(&mut state.module, &env, bindless);
                    match declared {
                        Ok(d) => {
                            let old = std::mem::replace(&mut state.programs, d.programs);
                            self.retire(old, bindless);
                            (state.graph, state.resources, state.uniform_bytes, state.declared_for) =
                                (d.graph, d.resources, d.uniform_bytes, Some(self.declare_key(&env)));
                        }
                        Err(error) => {
                            self.module = Some(state);
                            return Err(error.context("render module declaring a new target"));
                        }
                    }
                }
                let glow = library::glow_resource(&state.graph);
                let out = (state.graph.clone(), state.resources.clone(), glow);
                self.module = Some(state);
                out
            }
            None => {
                let (graph, wiring) = self.wire(native_graph(&env, self.output), &env, |p| {
                    Ok((NATIVE, ScenePassSource::Library(*p)))
                })?;
                (graph, wiring.resources, wiring.glow)
            }
        };
        let backbuffers: Vec<ResourceId> = graph
            .resources
            .iter()
            .enumerate()
            .filter(|(_, r)| r.class == ResourceClass::Backbuffer)
            .map(|(i, _)| ResourceId(i as u32))
            .collect();
        match self.output {
            TerrainOutput::Present => anyhow::ensure!(
                backbuffers.len() == 1,
                "the graph declares {} Backbuffer resources; presenting needs one",
                backbuffers.len()
            ),
            TerrainOutput::Readback => {
                anyhow::ensure!(backbuffers.is_empty(), "an offscreen graph declares no Backbuffer")
            }
        }
        self.wiring = Some(Wiring { resources, glow });
        Ok((graph, backbuffers.first().copied()))
    }

    fn prepare(&mut self, bindless: &mut crate::bindless::Bindless, slot: usize) -> Result<()> {
        self.apply_edits()?;

        if self.hdr.is_some() && self.encoding.uses_display() {
            self.levels = self.hdr_settings().levels(self.encoding);
        }
        let view = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        // Freeze the clocks at the frame the pause begins; afterwards the
        // wall clock skips the paused interval, so resuming neither jumps
        // nor replays it.
        let wall_now = Instant::now();
        if view.paused {
            self.paused_at.get_or_insert(wall_now);
        } else if let Some(at) = self.paused_at.take() {
            self.paused_for += wall_now.saturating_duration_since(at);
        }
        let frame_now = self.paused_at.unwrap_or(wall_now);
        let params = self.params(&view, frame_now);
        self.update_lod = params.update_lod;
        self.wireframe = params.wireframe;
        self.water_on = params.water;
        // The readback clock is frame-based: hold it with the rest.
        if !view.paused {
            self.frames += 1;
        }
        // The view's switch keeps the configured strength for resuming,
        // and turns the bloom off without changing the graph.
        self.tone.bloom = if view.bloom { self.tone_bloom } else { 0.0 };
        let has_bloom = self.wiring.as_ref().is_some_and(|w| w.glow.is_some());
        let has_exposure = self.graph_has(LibraryResource::Exposure);
        // Auto-exposure's step: real time, not the effects' clock, so the
        // exposure keeps adapting while paused (an explosion inspected on a
        // paused frame still darkens the image). Readback steps a fixed
        // 1/60 s a frame, keeping captures deterministic. Capped so a stall
        // cannot jump the exposure.
        let dt = match self.output {
            TerrainOutput::Readback => 1.0 / 60.0,
            TerrainOutput::Present => self.last_adapt_at.map_or(0.0, |last| {
                wall_now.saturating_duration_since(last).as_secs_f32().min(0.1)
            }),
        };
        self.last_adapt_at = Some(wall_now);
        // The meter undoes the fog of war's dimming where the terrain shades
        // by it this frame.
        let vision = if params.vision {
            self.renderer.as_ref().and_then(|r| r.vision_uniform())
        } else {
            None
        };
        if let Some(exposure) = &mut self.exposure {
            exposure.write_frame(
                slot,
                &ExposureFrame {
                    auto: self.tone.auto,
                    scene_ev100: self.tone.ev100,
                    dt,
                    view_proj: params.view_proj,
                    vision,
                    bloom: if has_bloom { self.tone.bloom } else { 0.0 },
                    working: self.working,
                },
            );
        }
        let tone = self.tone;
        if let Some(tonemap) = &mut self.tonemap {
            // SDR writes BT.709; HDR writes the display intermediate in the
            // working space, which post.encode maps to the output gamut.
            let written = if self.encoding.uses_display() {
                self.working.gamut()
            } else {
                Gamut::Bt709
            };
            tonemap.write(slot, tone, has_bloom, has_exposure, self.working, written);
        }
        let gamut = self.gamut();
        if let Some(encode) = &mut self.encode {
            encode.write(slot, self.encoding, self.levels, self.working, gamut);
        }
        if let Some(bloom) = &mut self.bloom {
            bloom.write(slot, self.tone.ev100, has_exposure);
        }
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
        if let (Some(set), Some(setup)) = (self.decal_set.as_mut(), self.decals.as_ref()) {
            for decal in setup.feed.take() {
                set.insert(decal, params.effect_time);
            }
            set.expire(params.effect_time);
            if self.decals_written.len() <= slot {
                self.decals_written.resize(slot + 1, None);
            }
            if self.decals_written[slot] != Some(set.version) {
                renderer.write_decals(slot, set.encode())?;
                self.decals_written[slot] = Some(set.version);
            }
            // The pipeline with decals only while one may be on screen
            // (decals never expire: otherwise it would stay on for good).
            let i = &self.source.info;
            let heights = (i.height_offset, i.height_offset + i.height_scale);
            renderer.set_decals_live(
                slot,
                set.any_in_view(&super::renderer::frustum_planes(params.view_proj), heights),
            );
        }
        if let (Some(units), Some(setup)) = (self.unit_renderer.as_mut(), self.units.as_ref()) {
            // This slot's previous frame has completed: its count is final.
            let previous = (units.rows(slot), units.read_visible(slot));
            *self.unit_counts.lock().unwrap_or_else(|p| p.into_inner()) = previous;
            let frame = UnitFrame {
                view_proj: params.view_proj,
                cam_pos: params.cam_pos,
                proj_scale: view.camera.proj_scale(self.extent.height as f32),
                fog: fog(&self.source.info, params.fog, self.working),
                shadows: params.shadows,
                unit_shadows: params.unit_shadows,
                photometry: params.photometry,
            };
            units.write_frame(slot, &frame, &setup.feed, frame_now);
        }
        let mut camera = view.camera;
        camera.ground = self.source.height_at(camera.target);
        let v = camera.view(params.cam_pos);
        let axes = [v.row(0).truncate(), v.row(1).truncate(), -v.row(2).truncate()];
        if let Some(particles) = self.particle_renderer.as_mut() {
            let frame = ParticleFrame {
                view_proj: params.view_proj,
                cam_pos: params.cam_pos,
                axes,
                near: camera.near(),
                extent: self.extent,
                time: params.effect_time,
                fog: fog(&self.source.info, params.fog, self.working),
                shadows: params.shadows,
                unit_shadows: params.unit_shadows.filter(|c| c.tinted),
                photometry: params.photometry,
            };
            let gpu = self.gpu.clone().context("no device")?;
            particles.write_frame(&gpu, bindless, slot, &frame)?;
        }
        if let Some(lights) = self.lights.as_mut() {
            // The fixed lights and the bursts', those that can reach the
            // view (by their bounding spheres).
            let mut list = std::mem::take(&mut self.frame_lights);
            list.clear();
            // The app's lights are authored in BT.709 and normalised to
            // unit luminance there (the `Light` constructors); the cluster
            // lives in the working space.
            list.extend(self.fixed_lights.iter().map(|light| {
                let mut light = *light;
                light.color = self.working.from_709(light.color);
                light
            }));
            if let Some(particles) = self.particle_renderer.as_mut() {
                let source = &self.source;
                // The bursts' lights are already in the working space: their
                // effect's gamut was converted and normalised at upload.
                particles.lights(&|p| source.height_at(p), &mut list);
            }
            let planes = super::renderer::frustum_planes(params.view_proj);
            // Auto-exposure may reach MIN_EV100 (the largest exposure, the
            // widest cull radius): bound the lights there so adaptation
            // never hides one it would show.
            // Only a graph with the exposure image adapts.
            let ev100 = if self.tone.auto && has_exposure {
                crate::lighting::MIN_EV100
            } else {
                self.tone.ev100
            };
            let cutoff = crate::lighting::cull_cutoff(crate::lighting::exposure(ev100));
            list.retain(|l| {
                let (c, r) = l.bounds(cutoff);
                planes.iter().all(|p| p[0] * c.x + p[1] * c.y + p[2] * c.z + p[3] >= -r)
            });
            let clustered = self
                .wiring
                .as_ref()
                .is_some_and(|w| w.resources.contains(&Some(LibraryResource::LightClusters)));
            // The units' models may carry lights, which the pass appends
            // even with no host light.
            let unit_rows = self.unit_renderer.as_ref().map_or(0, |u| u.rows(slot));
            // The append needs the rows; without them the slot's total is
            // stale (no stream this frame) and must not reach the header.
            let unit_lights = if unit_rows > 0 {
                self.unit_renderer.as_ref().map_or(0, |u| u.model_light_total(slot))
            } else {
                0
            };
            let model_lights = unit_lights > 0 && self.unit_renderer.as_ref().is_some_and(|u| u.has_model_lights());
            // The map's heights, with room above for the units, the
            // particles and the lights over them.
            let i = &self.source.info;
            let (y_lo, y_hi) = (i.height_offset, i.height_offset + i.height_scale + 80.0);
            let depths =
                crate::lights::depth_range(params.view_proj, params.cam_pos, axes[2], camera.near(), y_lo, y_hi);
            let on = clustered && (!list.is_empty() || model_lights);
            let frame = LightFrame {
                view_proj: params.view_proj,
                cam_pos: params.cam_pos,
                axes,
                near: camera.near(),
                extent: self.extent,
                on,
                depths,
                unit_rows,
                unit_lights,
                cutoff,
            };
            lights.write_frame(slot, &frame, &list);
            *self.light_stats.lock().unwrap_or_else(|p| p.into_inner()) = lights.stats();
            self.frame_lights = list;
        }
        if let Some(state) = self.module.as_mut().filter(|m| !m.failed) {
            let mut input = FrameInput {
                width: self.extent.width,
                height: self.extent.height,
                frame: state.frames,
                units: self.unit_renderer.as_ref().map_or(0, |u| u.rows(slot)),
                time: frame_now
                    .saturating_duration_since(state.module.loaded())
                    .saturating_sub(self.paused_for)
                    .as_secs_f32(),
                flags: view.module_flags,
                view_proj: params.view_proj.to_cols_array(),
                ..FrameInput::default()
            };
            input.camera[..3].copy_from_slice(&params.cam_pos.to_array());
            let result = state.module.frame(&input).and_then(|frame| {
                anyhow::ensure!(
                    frame.uniforms.len() == state.uniform_bytes,
                    "`frame` returned {} uniform bytes; the fullscreen programs declare {}",
                    frame.uniforms.len(),
                    state.uniform_bytes
                );
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
                    self.module_status
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .errors
                        .push(format!(
                            "render module stopped (its last graph keeps rendering): {error:#}"
                        ));
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
            LibraryResource::UnitPalette => Binding::Buffer(units()?.palette()),
            LibraryResource::ReadbackTarget => {
                let o = offscreen()?;
                Binding::Image(ImageBinding {
                    image: o.image.image,
                    view: o.image.view,
                    format: self.output_format,
                    extent: o.extent,
                    range: o.image.range(),
                    identity: (0xAE7 << 40) | 10,
                    sampled: None,
                    storage: None,
                })
            }
            LibraryResource::ReadbackBuffer => {
                let o = offscreen()?;
                Binding::Buffer(BufferBinding {
                    buffer: o.buffer.buffer,
                    offset: 0,
                    size: o.buffer.size,
                    identity: (0xAE7 << 40) | 11,
                    storage: None,
                })
            }
            LibraryResource::ShadowMap => Binding::Image(self.renderer.as_ref()?.unit_shadow_map().binding()),
            LibraryResource::ParticleShadows => {
                Binding::Image(self.renderer.as_ref()?.unit_shadow_map().color_binding())
            }
            LibraryResource::ParticleState => Binding::Buffer(self.particle_renderer.as_ref()?.state()),
            LibraryResource::ParticleKeys => Binding::Buffer(self.particle_renderer.as_ref()?.keys()),
            LibraryResource::ParticleIndirect => Binding::Buffer(self.particle_renderer.as_ref()?.indirect()),
            LibraryResource::LightList => Binding::Buffer(self.lights.as_ref()?.list(slot)),
            LibraryResource::LightClusters => Binding::Buffer(self.lights.as_ref()?.clusters(slot)),
            LibraryResource::Exposure => Binding::Image(self.exposure.as_ref()?.image()),
            LibraryResource::ExposureHistogram => Binding::Buffer(self.exposure.as_ref()?.histogram()),
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

    fn retarget(
        &mut self,
        gpu: &Arc<GpuDevice>,
        bindless: &mut crate::bindless::Bindless,
        target: TargetInfo,
        frames_in_flight: usize,
    ) -> Result<bool> {
        if self.renderer.is_none() || self.output != TerrainOutput::Present || frames_in_flight != self.frames_in_flight
        {
            self.retire_handles(bindless);
            self.destroy(gpu);
            self.create(gpu, bindless, target, frames_in_flight)?;
            return Ok(true);
        }
        // Only the output side depends on the pair. The terrain's tree
        // refines a level per frame from its root and holds the sim's
        // ground edits, the particles and the exposure carry state too:
        // rebuilding them made the ground settle anew on every HDR toggle.
        self.retire_module_programs(gpu, bindless);
        self.create_output(gpu, bindless, target, frames_in_flight)?;
        Ok(false)
    }

    fn destroy(&mut self, gpu: &Arc<GpuDevice>) {
        self.destroy_module_programs(gpu);
        // The registry owns the sampler object; this is only the handle
        // recording passes use.
        self.sampler = vk::Sampler::null();
        if let Some(mut tonemap) = self.tonemap.take() {
            tonemap.destroy(gpu);
        }
        if let Some(mut encode) = self.encode.take() {
            encode.destroy(gpu);
        }
        if let Some(mut bloom) = self.bloom.take() {
            bloom.destroy(gpu);
        }
        if let Some(mut exposure) = self.exposure.take() {
            exposure.destroy(gpu);
        }
        if let Some(mut particles) = self.particle_renderer.take() {
            particles.destroy(gpu);
        }
        if let Some(mut units) = self.unit_renderer.take() {
            units.destroy(gpu);
        }
        if let Some(mut renderer) = self.renderer.take() {
            renderer.destroy(gpu);
        }
        if let Some(mut lights) = self.lights.take() {
            lights.destroy(gpu);
        }
        if let Some(mut offscreen) = self.offscreen.take() {
            offscreen.image.destroy(gpu);
            offscreen.buffer.destroy(gpu);
        }
    }
}

impl TerrainApp {
    /// Give every module's global-set handles back (the previous build's).
    /// The device objects are destroyed right after; the registry outlives
    /// them.
    fn retire_handles(&mut self, bindless: &mut crate::bindless::Bindless) {
        if let Some(handle) = self.sampler_handle.take() {
            bindless.retire_sampler(handle);
        }
        // The post chain and the module's programs too: retarget's full
        // path destroys them without a retire of its own. A program's
        // retire is idempotent, so create's own retires are harmless.
        if let Some(old) = self.tonemap.as_mut() {
            old.retire(bindless);
        }
        if let Some(old) = self.encode.as_mut() {
            old.retire(bindless);
        }
        if let Some(old) = self.bloom.as_mut() {
            old.retire(bindless);
        }
        if let Some(state) = self.module.as_mut() {
            for (program, _) in state.programs.iter_mut().flatten() {
                program.retire(bindless);
            }
        }
        if let Some(old) = self.exposure.as_mut() {
            old.retire(bindless);
        }
        if let Some(old) = self.particle_renderer.as_mut() {
            old.retire(bindless);
        }
        if let Some(old) = self.unit_renderer.as_mut() {
            old.retire(bindless);
        }
        if let Some(old) = self.renderer.as_mut() {
            old.retire(bindless);
        }
        if let Some(old) = self.lights.as_mut() {
            old.retire(bindless);
        }
    }

    /// The objects that depend on the output (format, colour space) pair:
    /// the encoding, its levels, the tonemap and `post.encode`.
    fn create_output(
        &mut self,
        gpu: &Arc<GpuDevice>,
        bindless: &mut crate::bindless::Bindless,
        target: TargetInfo,
        frames_in_flight: usize,
    ) -> Result<()> {
        if let Some(mut old) = self.tonemap.take() {
            old.retire(bindless);
            old.destroy(gpu);
        }
        if let Some(mut old) = self.encode.take() {
            old.retire(bindless);
            old.destroy(gpu);
        }
        // The client's present path follows the swapchain's colour space
        // (M13 stage 5): HDR10, scRGB or SDR. An offscreen app keeps the
        // encoding it was given (the capture path stays SDR).
        if self.output == TerrainOutput::Present {
            self.encoding = target.target().encoding();
        }
        if self.hdr.is_some() && self.encoding.uses_display() {
            self.levels = self.hdr_settings().levels(self.encoding);
        }
        let output = if self.output == TerrainOutput::Readback {
            self.encoding.readback_format()
        } else {
            target.format
        };
        self.output_format = output;
        self.output_space = if self.output == TerrainOutput::Present {
            target.color_space
        } else {
            vk::ColorSpaceKHR::SRGB_NONLINEAR
        };
        // In HDR the tonemap writes the display intermediate, paper-white
        // units; `post.encode` maps it to the output.
        let tonemap_out = if self.encoding.uses_display() {
            crate::encode::DISPLAY_FORMAT
        } else {
            output
        };
        self.tonemap = Some(Tonemap::new(gpu, bindless, tonemap_out, frames_in_flight).context("tonemap")?);
        self.encode = if self.encoding.uses_display() {
            Some(Encode::new(gpu, bindless, self.encoding, output, frames_in_flight).context("encode")?)
        } else {
            None
        };
        Ok(())
    }

    /// The running module stays loaded across a rebuild or a retarget: its
    /// format-dependent programs go, its declaration is made again for the
    /// new target.
    fn destroy_module_programs(&mut self, gpu: &Arc<GpuDevice>) {
        self.take_module_programs(gpu);
    }

    /// The same, giving the programs' global-set handles back first (the
    /// registry is still alive).
    fn retire_module_programs(&mut self, gpu: &Arc<GpuDevice>, bindless: &mut crate::bindless::Bindless) {
        if let Some(state) = self.module.as_mut() {
            gpu.wait_idle();
            for (mut program, _) in std::mem::take(&mut state.programs).into_iter().flatten() {
                program.retire(bindless);
                program.destroy(gpu);
            }
            state.declared_for = None;
        }
    }

    fn take_module_programs(&mut self, gpu: &Arc<GpuDevice>) {
        if let Some(state) = self.module.as_mut() {
            gpu.wait_idle();
            for (mut program, _) in std::mem::take(&mut state.programs).into_iter().flatten() {
                program.destroy(gpu);
            }
            state.declared_for = None;
        }
    }

    /// Create the offscreen target and readback buffer for `extent`
    /// (readback output only; call before `run_headless`).
    pub fn create_offscreen(&mut self, gpu: &GpuDevice, extent: vk::Extent2D) -> Result<()> {
        let desc = ImageDesc::color_2d(
            self.encoding.readback_format(),
            extent.width,
            extent.height,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        );
        let image = OwnedImage::new(gpu, desc, "terrain offscreen")?;
        let size = extent.width as u64 * extent.height as u64 * self.encoding.bytes_per_pixel();
        let buffer = OwnedBuffer::new(
            gpu,
            BufferDesc {
                size,
                usage: vk::BufferUsageFlags::TRANSFER_DST,
            },
            MemoryLocation::GpuToCpu,
            "terrain readback",
        )?;
        self.offscreen = Some(Offscreen { image, buffer, extent });
        Ok(())
    }
}

impl Recorder<ScenePass> for TerrainApp {
    fn skip(&self, pass: &ScenePass) -> bool {
        self.module.as_ref().is_some_and(|m| m.frame.is_disabled(pass.program))
    }

    fn clear_color(&self, pass: &ScenePass, resource: ResourceId) -> [f32; 4] {
        match pass.work {
            // No optical depth, no particle toward the sun.
            PassWork::Library(LibraryPass::ShadowsParticles) => [0.0; 4],
            // The glow target holds only light sources: black where none is
            // (the sky is not one).
            _ if self.wiring.as_ref().is_some_and(|w| w.glow == Some(resource)) => [0.0, 0.0, 0.0, 1.0],
            // The sky behind the terrain: its radiance, stored exposed like
            // the scene.
            _ => {
                let r = crate::lighting::sky_radiance(self.working);
                let e = self.tone.exposure();
                [r[0] * e, r[1] * e, r[2] * e, 1.0]
            }
        }
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
            particles: self.particle_renderer.as_ref(),
            tonemap: self.tonemap.as_ref(),
            encode: self.encode.as_ref(),
            bloom: self.bloom.as_ref(),
            exposure: self.exposure.as_ref(),
            lights: self.lights.as_ref(),
            programs: self.module.as_ref().map(|m| m.programs.as_slice()),
            sampler: self.sampler,
            sampler_index: self.sampler_handle.map_or(0, |h| h.index),
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
    particles: Option<&'a ParticleRenderer>,
    tonemap: Option<&'a Tonemap>,
    encode: Option<&'a Encode>,
    bloom: Option<&'a Bloom>,
    exposure: Option<&'a Exposure>,
    lights: Option<&'a LightRenderer>,
    programs: Option<&'a [Option<(FullscreenProgram, usize)>]>,
    sampler: vk::Sampler,
    sampler_index: u32,
    update_lod: bool,
    wireframe: Wireframe,
    water: bool,
}

impl SharedRecorder<ScenePass> for SceneRecorder<'_> {
    fn record(&self, pass: &ScenePass, ctx: &PassContext<'_>) -> Result<()> {
        let units = || self.units.context("unit renderer not created");
        let terrain = || self.renderer.context("terrain renderer not created");
        let particles = || self.particles.context("particle renderer not created");
        let exposure = || self.exposure.context("exposure not created");
        match &pass.work {
            PassWork::Library(lib) => match lib {
                LibraryPass::TerrainLod => terrain()?.record_lod(ctx, self.update_lod),
                LibraryPass::TerrainDraw => terrain()?.record_draw(ctx, self.wireframe),
                LibraryPass::UnitsReset => units()?.record_reset(ctx),
                LibraryPass::UnitsCull => units()?.record_cull(ctx),
                LibraryPass::UnitsPose => units()?.record_pose(ctx),
                LibraryPass::UnitsDraw => units()?.record_draw(ctx),
                LibraryPass::UnitsCount => units()?.record_count(ctx),
                LibraryPass::ShadowsDraw => units()?.record_shadows(ctx),
                LibraryPass::ShadowsParticles => particles()?.record_shadows(ctx),
                LibraryPass::WaterDraw => {
                    if self.water {
                        terrain()?.record_water(ctx);
                    }
                }
                LibraryPass::ParticlesSim => particles()?.record_sim(ctx),
                LibraryPass::LightsCluster => self
                    .lights
                    .context("lights not created")?
                    .record_cluster(ctx, self.units),
                LibraryPass::ExposureClear => exposure()?.record_clear(ctx),
                LibraryPass::ExposureAdapt => exposure()?.record_adapt(ctx),
                LibraryPass::ParticlesDraw => anyhow::bail!("particles.draw is wired as PassWork::Particles"),
                LibraryPass::ReadbackCopy => anyhow::bail!("readback.copy is wired as PassWork::Readback"),
                LibraryPass::Tonemap => anyhow::bail!("post.tonemap is wired as PassWork::Tonemap"),
                LibraryPass::BloomDownsample | LibraryPass::BloomBlurX | LibraryPass::BloomBlurY => {
                    anyhow::bail!("bloom passes are wired as PassWork::Bloom")
                }
                LibraryPass::ExposureMeter => anyhow::bail!("exposure.meter is wired as PassWork::ExposureMeter"),
                LibraryPass::PostEncode => anyhow::bail!("post.encode is wired as PassWork::Encode"),
                LibraryPass::HostRead | LibraryPass::Present => {}
            },
            PassWork::Fullscreen { sampled } => {
                let programs = self.programs.context("fullscreen pass without a module")?;
                let (program, _) = programs
                    .get(pass.program as usize)
                    .and_then(Option::as_ref)
                    .context("fullscreen program missing")?;
                let images = sampled.iter().map(|&r| ctx.image(r)).collect::<Result<Vec<_>>>()?;
                program.record(
                    ctx,
                    &images,
                    crate::fullscreen::Sampler {
                        object: self.sampler,
                        index: self.sampler_index,
                    },
                )?;
            }
            PassWork::Particles { color, depth, glow } => {
                particles()?.record_draw(ctx, ctx.image(*color)?, ctx.image(*depth)?, ctx.image(*glow)?)?
            }
            PassWork::Tonemap { scene, bloom, exposure } => {
                let bloom = bloom.map(|r| ctx.image(r)).transpose()?;
                let exposure = exposure.map(|r| ctx.image(r)).transpose()?;
                self.tonemap.context("tonemap not created")?.record(
                    ctx,
                    ctx.image(*scene)?,
                    bloom,
                    exposure,
                    self.sampler_index,
                )?
            }
            PassWork::Encode { display } => {
                self.encode
                    .context("encode not created")?
                    .record(ctx, ctx.image(*display)?, self.sampler_index)?
            }
            PassWork::Bloom {
                pass: bloom_pass,
                input,
                output,
                exposure,
            } => {
                let exposure = exposure.map(|r| ctx.image(r)).transpose()?;
                self.bloom.context("bloom not created")?.record(
                    ctx,
                    *bloom_pass,
                    ctx.image(*input)?,
                    ctx.image(*output)?,
                    exposure,
                    self.sampler_index,
                )?
            }
            PassWork::ExposureMeter { scene, depth, glow } => {
                let depth = depth.map(|r| ctx.image(r)).transpose()?;
                let glow = glow.map(|r| ctx.image(r)).transpose()?;
                let vision = self.renderer.map(|t| t.vision_handle());
                exposure()?.record_meter(ctx, ctx.image(*scene)?, depth, glow, vision)?
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

/// The scene's graph as the engine declares it natively: the units'
/// shadow map when they cast shadows, the terrain, the units when the
/// scene has them, the water surface when it draws water, and present or
/// readback. Library buffers
/// are declared with size 0 ([`crate::library::resolve`] sizes them). A
/// render module that declares the same graph gets the same bake.
pub fn native_graph(env: &LibraryEnv, output: TerrainOutput) -> FrameGraph<LibraryPass> {
    let (width, height) = (env.extent.width, env.extent.height);
    let mut graph = FrameGraph::new();
    let leaves = library_buffer(&mut graph, LibraryResource::TerrainLeaves);
    let indirect = library_buffer(&mut graph, LibraryResource::TerrainIndirect);
    // Sampled usage for the particles (always, so a graph is the same
    // with and without them up to their passes).
    let depth = graph.image(
        "depth",
        ImageDesc::depth_2d(
            env.depth_format,
            width,
            height,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        ),
        ResourceClass::Transient,
    );
    let target = match output {
        TerrainOutput::Present => graph.image(
            "target",
            ImageDesc::color_2d(env.color_format, width, height, vk::ImageUsageFlags::COLOR_ATTACHMENT),
            ResourceClass::Backbuffer,
        ),
        TerrainOutput::Readback => graph.image(
            LibraryResource::ReadbackTarget.name(),
            ImageDesc::color_2d(
                env.color_format,
                width,
                height,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            ),
            ResourceClass::External,
        ),
    };
    // The HDR scene the draws write; `post.tonemap` turns it into the target
    // (SDR) or into the display, which `post.encode` maps to the target (HDR).
    let scene = graph.image(
        ae_render_abi::module::SCENE_NAME,
        ImageDesc::color_2d(
            env.scene_format,
            width,
            height,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        ),
        ResourceClass::Transient,
    );
    let display = env.encoding.uses_display().then(|| {
        graph.image(
            crate::encode::DISPLAY_NAME,
            ImageDesc::color_2d(
                crate::encode::DISPLAY_FORMAT,
                width,
                height,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ),
            ResourceClass::Transient,
        )
    });
    // The glow (the bloom's input) is the second colour attachment of every
    // scene draw; the bloom reads it at a quarter of the size.
    let glow = graph.image(
        ae_render_abi::module::GLOW_NAME,
        ImageDesc::color_2d(
            vk::Format::from_raw(ae_render_abi::module::GLOW_FORMAT as i32),
            width,
            height,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        ),
        ResourceClass::Transient,
    );
    let (bw, bh) = ae_render_abi::module::bloom_size(width, height);
    let bloom_image = |graph: &mut FrameGraph<LibraryPass>, name: &str| {
        graph.image(
            name,
            ImageDesc::color_2d(
                vk::Format::from_raw(ae_render_abi::module::BLOOM_FORMAT as i32),
                bw,
                bh,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ),
            ResourceClass::Transient,
        )
    };
    let bloom0 = bloom_image(&mut graph, ae_render_abi::module::BLOOM_NAME);
    let bloom1 = bloom_image(&mut graph, ae_render_abi::module::BLOOM_SCRATCH_NAME);
    // The unit buffers, then the units' shadow map.
    let units = env.units.then(|| {
        [
            LibraryResource::UnitInstances,
            LibraryResource::UnitVisible,
            LibraryResource::UnitIndirect,
            LibraryResource::UnitCounts,
            LibraryResource::UnitPalette,
        ]
        .map(|r| library_buffer(&mut graph, r))
    });
    let map = (env.units && env.shadows).then(|| library_shadow_map(&mut graph));
    let particles = env.particles.then(|| {
        [
            LibraryResource::ParticleState,
            LibraryResource::ParticleKeys,
            LibraryResource::ParticleIndirect,
        ]
        .map(|r| library_buffer(&mut graph, r))
    });
    // The particles' coloured shadows, with the map.
    let tint = (map.is_some() && particles.is_some()).then(|| library_particle_shadows(&mut graph));
    // The clustered lights: the list the host writes and each cluster's.
    let lights = [LibraryResource::LightList, LibraryResource::LightClusters].map(|r| library_buffer(&mut graph, r));
    // The eased EV and its metering histogram (host-owned, shared by the
    // slots: the executor derives the cross-frame barriers).
    let exposure = graph.image(
        ae_render_abi::module::EXPOSURE_NAME,
        ImageDesc::color_2d(
            vk::Format::from_raw(ae_render_abi::module::EXPOSURE_FORMAT as i32),
            1,
            1,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
        ),
        ResourceClass::External,
    );
    let exposure_histogram = library_buffer(&mut graph, LibraryResource::ExposureHistogram);
    // A receiver samples the maps the graph has, and reads the lights.
    let receivers = [map, tint];
    fn lit(
        pass: ae_framegraph::PassBuilder<'_, LibraryPass>,
        lights: [ResourceId; 2],
        stage: ShaderStage,
    ) -> ae_framegraph::PassBuilder<'_, LibraryPass> {
        pass.access(lights[0], Role::StorageBufferRead(stage))
            .access(lights[1], Role::StorageBufferRead(stage))
    }
    fn receive(
        mut pass: ae_framegraph::PassBuilder<'_, LibraryPass>,
        maps: [Option<ResourceId>; 2],
        lights: [ResourceId; 2],
    ) -> ae_framegraph::PassBuilder<'_, LibraryPass> {
        for map in maps.into_iter().flatten() {
            pass = pass.sampled(map, ShaderStage::Fragment);
        }
        lit(pass, lights, ShaderStage::Fragment)
    }
    graph
        .pass("terrain lod", PassKind::Compute, LibraryPass::TerrainLod)
        .access(leaves, Role::StorageBufferWrite(ShaderStage::Compute))
        .access(indirect, Role::StorageBufferWrite(ShaderStage::Compute))
        .id();
    // The units' poses and cull, before anything draws them (the shadows
    // read the cull's lists per cascade).
    if let Some([instances, visible, draws, _, palette]) = units {
        graph
            .pass("units pose", PassKind::Compute, LibraryPass::UnitsPose)
            .access(instances, Role::StorageBufferRead(ShaderStage::Compute))
            .access(palette, Role::StorageBufferWrite(ShaderStage::Compute))
            .id();
        graph
            .pass("units reset", PassKind::Transfer, LibraryPass::UnitsReset)
            .access(draws, Role::TransferDst)
            .id();
        graph
            .pass("units cull", PassKind::Compute, LibraryPass::UnitsCull)
            .access(instances, Role::StorageBufferRead(ShaderStage::Compute))
            .access(visible, Role::StorageBufferWrite(ShaderStage::Compute))
            .access(draws, Role::StorageBufferWrite(ShaderStage::Compute))
            .id();
    }
    // The lights' clusters, before anything lit.
    let write = Role::StorageBufferWrite(ShaderStage::Compute);
    graph
        .pass("lights", PassKind::Compute, LibraryPass::LightsCluster)
        .access(lights[0], write)
        .access(lights[1], write)
        .id();
    // The units' shadows: the units drawn from the sun, before anything
    // that samples the map.
    if let (Some([instances, visible, draws, _, palette]), Some(map)) = (units, map) {
        graph
            .pass("shadows", PassKind::Graphics, LibraryPass::ShadowsDraw)
            .access(instances, Role::StorageBufferRead(ShaderStage::Vertex))
            .access(visible, Role::StorageBufferRead(ShaderStage::Vertex))
            .access(draws, Role::IndirectBuffer)
            .access(palette, Role::StorageBufferRead(ShaderStage::Vertex))
            .depth(map, LoadOp::Clear)
            .id();
    }
    // The particles' coloured shadows: this frame's particles (the
    // simulation, declared later, is their state's only writer, so this
    // reads after it).
    if let (Some([state, keys, draws]), Some(tint)) = (particles, tint) {
        graph
            .pass("particle shadows", PassKind::Graphics, LibraryPass::ShadowsParticles)
            .access(state, Role::StorageBufferRead(ShaderStage::Vertex))
            .access(keys, Role::StorageBufferRead(ShaderStage::Vertex))
            .access(draws, Role::IndirectBuffer)
            .color(tint, LoadOp::Clear)
            .id();
    }
    receive(
        graph
            .pass("terrain", PassKind::Graphics, LibraryPass::TerrainDraw)
            .access(leaves, Role::StorageBufferRead(ShaderStage::Vertex))
            .access(indirect, Role::IndirectBuffer)
            .color(scene, LoadOp::Clear)
            .color(glow, LoadOp::Clear)
            .depth(depth, LoadOp::Clear),
        receivers,
        lights,
    )
    .id();
    if let Some([instances, visible, draws, counts, palette]) = units {
        // Declared after the terrain draw: a second writer of the target
        // and the depth buffer, ordered after it.
        receive(
            graph
                .pass("units", PassKind::Graphics, LibraryPass::UnitsDraw)
                .access(instances, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(visible, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(draws, Role::IndirectBuffer)
                .access(palette, Role::StorageBufferRead(ShaderStage::Vertex))
                .color(scene, LoadOp::Load)
                .color(glow, LoadOp::Load)
                .depth(depth, LoadOp::Load),
            receivers,
            lights,
        )
        .id();
        graph
            .pass("units count", PassKind::Transfer, LibraryPass::UnitsCount)
            .access(draws, Role::TransferSrc)
            .discard(counts, Role::TransferDst)
            .id();
        graph
            .pass("units count read", PassKind::Transfer, LibraryPass::HostRead)
            .access(counts, Role::HostRead)
            .root()
            .id();
    }
    if env.water {
        // After the units (a third writer of the target), testing against
        // the depth they leave without writing it.
        receive(
            graph
                .pass("water", PassKind::Graphics, LibraryPass::WaterDraw)
                .color(scene, LoadOp::Load)
                .color(glow, LoadOp::Load)
                .depth_read(depth),
            receivers,
            lights,
        )
        .id();
    }
    if let Some([state, keys, draws]) = particles {
        lit(
            graph
                .pass("particles sim", PassKind::Compute, LibraryPass::ParticlesSim)
                .access(state, write)
                .access(keys, write)
                .access(draws, write),
            lights,
            ShaderStage::Compute,
        )
        .id();
        // Over everything else, sampling the depth the scene left.
        graph
            .pass("particles", PassKind::Graphics, LibraryPass::ParticlesDraw)
            .access(state, Role::StorageBufferRead(ShaderStage::Vertex))
            .access(keys, Role::StorageBufferRead(ShaderStage::Vertex))
            .access(draws, Role::IndirectBuffer)
            .color(scene, LoadOp::Load)
            .color(glow, LoadOp::Load)
            .sampled(depth, ShaderStage::Fragment)
            .id();
    }
    // Automatic exposure: the histogram zeroed, the scene metered (with the
    // depth, to undo the fog of war's dimming, and the glow the bloom
    // shows), the EV adapted. It runs after every scene writer (the
    // particles) and before the bloom and the tonemap, so the frame's eased
    // EV is the one shown.
    graph
        .pass("exposure clear", PassKind::Transfer, LibraryPass::ExposureClear)
        .access(exposure_histogram, Role::TransferDst)
        .id();
    graph
        .pass("exposure meter", PassKind::Compute, LibraryPass::ExposureMeter)
        .sampled(scene, ShaderStage::Compute)
        .sampled(depth, ShaderStage::Compute)
        .sampled(glow, ShaderStage::Compute)
        .access(exposure_histogram, Role::StorageBufferReadWrite(ShaderStage::Compute))
        .id();
    graph
        .pass("exposure adapt", PassKind::Compute, LibraryPass::ExposureAdapt)
        .access(exposure_histogram, Role::StorageBufferRead(ShaderStage::Compute))
        .access(exposure, Role::StorageImageReadWrite(ShaderStage::Compute))
        .id();
    // The bloom: the glow at a quarter size, brightest 2x2 mean per 4x4,
    // then an 11-tap Gaussian along x and y (RA3-graphics' three passes).
    graph
        .pass("bloom downsample", PassKind::Graphics, LibraryPass::BloomDownsample)
        .sampled(glow, ShaderStage::Fragment)
        .sampled(exposure, ShaderStage::Fragment)
        .color(bloom0, LoadOp::DontCare)
        .id();
    graph
        .pass("bloom blur x", PassKind::Graphics, LibraryPass::BloomBlurX)
        .sampled(bloom0, ShaderStage::Fragment)
        .color(bloom1, LoadOp::DontCare)
        .id();
    graph
        .pass("bloom blur y", PassKind::Graphics, LibraryPass::BloomBlurY)
        .sampled(bloom1, ShaderStage::Fragment)
        .color(bloom0, LoadOp::DontCare)
        .id();
    graph
        .pass("tonemap", PassKind::Graphics, LibraryPass::Tonemap)
        .sampled(scene, ShaderStage::Fragment)
        .sampled(bloom0, ShaderStage::Fragment)
        .sampled(exposure, ShaderStage::Fragment)
        .color(display.unwrap_or(target), LoadOp::DontCare)
        .id();
    if let Some(display) = display {
        // The display's light, in the output's encoding (PQ or scRGB).
        graph
            .pass("encode", PassKind::Graphics, LibraryPass::PostEncode)
            .sampled(display, ShaderStage::Fragment)
            .color(target, LoadOp::DontCare)
            .id();
    }
    match output {
        TerrainOutput::Present => {
            graph
                .pass("present", PassKind::Present, LibraryPass::Present)
                .present(target)
                .root()
                .id();
        }
        TerrainOutput::Readback => {
            let readback = library_buffer(&mut graph, LibraryResource::ReadbackBuffer);
            graph
                .pass("copy", PassKind::Transfer, LibraryPass::ReadbackCopy)
                .access(target, Role::TransferSrc)
                .discard(readback, Role::TransferDst)
                .id();
            graph
                .pass("host read", PassKind::Transfer, LibraryPass::HostRead)
                .access(readback, Role::HostRead)
                .root()
                .id();
        }
    }
    graph
}

/// A library buffer as a graph declares it: `External`, size 0.
pub fn library_buffer<R>(graph: &mut FrameGraph<R>, r: LibraryResource) -> ResourceId {
    graph.buffer(
        r.name(),
        BufferDesc {
            size: 0,
            usage: vk::BufferUsageFlags::empty(),
        },
        ResourceClass::External,
    )
}

/// The units' shadow map as a graph declares it (`External`).
pub fn library_shadow_map<R>(graph: &mut FrameGraph<R>) -> ResourceId {
    use ae_render_abi::module as m;
    let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
    let desc = ImageDesc::depth_2d(
        vk::Format::from_raw(m::SHADOW_MAP_FORMAT as i32),
        m::SHADOW_MAP_WIDTH,
        m::SHADOW_MAP_HEIGHT,
        usage,
    );
    graph.image(LibraryResource::ShadowMap.name(), desc, ResourceClass::External)
}

/// The particles' coloured shadows as a graph declares them (`External`).
pub fn library_particle_shadows<R>(graph: &mut FrameGraph<R>) -> ResourceId {
    use ae_render_abi::module as m;
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
    let desc = ImageDesc::color_2d(
        vk::Format::from_raw(m::PARTICLE_SHADOW_FORMAT as i32),
        m::PARTICLE_SHADOW_WIDTH,
        m::PARTICLE_SHADOW_HEIGHT,
        usage,
    );
    graph.image(LibraryResource::ParticleShadows.name(), desc, ResourceClass::External)
}

impl TerrainApp {
    /// Load `wasm` and wire its declaration for `env`.
    fn load_module(
        &mut self,
        wasm: &[u8],
        env: &LibraryEnv,
        bindless: &mut crate::bindless::Bindless,
    ) -> Result<ModuleState> {
        let engine = self.engine.get_or_insert_with(crate::module::engine).clone();
        let mut module = RenderModule::load(&engine, wasm)?;
        let d = self.declare_module(&mut module, env, bindless)?;
        Ok(ModuleState {
            module,
            declared_for: Some(self.declare_key(env)),
            graph: d.graph,
            resources: d.resources,
            programs: d.programs,
            uniform_bytes: d.uniform_bytes,
            frame: ModuleFrame::default(),
            failed: false,
            frames: 0,
        })
    }

    /// Give back and destroy the programs a replaced declaration used, once
    /// the GPU is done with them (a reload or resize: waiting for idle is
    /// fine there).
    fn retire(&self, programs: Vec<Option<(FullscreenProgram, usize)>>, bindless: &mut crate::bindless::Bindless) {
        let programs: Vec<_> = programs.into_iter().flatten().collect();
        if programs.is_empty() {
            return;
        }
        if let Some(gpu) = &self.gpu {
            gpu.wait_idle();
            for (mut program, _) in programs {
                program.retire(bindless);
                program.destroy(gpu);
            }
        }
    }

    /// The module's graph for `env`, decoded and wired, with its fullscreen
    /// programs reflected, checked and built.
    fn declare_module(
        &self,
        module: &mut RenderModule,
        env: &LibraryEnv,
        bindless: &mut crate::bindless::Bindless,
    ) -> Result<ModuleGraph> {
        // Contract 2 modules keep the per-pass form; 3 may use either.
        let legacy_only = module.version() < 3;
        let flags = if env.units { DECLARE_UNITS } else { 0 }
            | if self.output == TerrainOutput::Readback {
                DECLARE_READBACK
            } else {
                0
            }
            | if env.water { DECLARE_WATER } else { 0 }
            | if env.shadows { DECLARE_SHADOWS } else { 0 }
            | if env.particles { DECLARE_PARTICLES } else { 0 };
        let target = DeclareTarget {
            width: env.extent.width,
            height: env.extent.height,
            color_format: env.color_format,
            flags,
            encoding: env.encoding,
            levels: env.levels,
            working: env.working,
            gamut: env.gamut,
        };
        let declared = module.declare(&target)?.with_context(|| {
            format!(
                "the module declines a {}x{} {:?} {:?} target",
                env.extent.width, env.extent.height, env.color_format, env.encoding
            )
        })?;
        let graph = ae_framegraph::descriptor::decode(&declared.graph)
            .map_err(|e| anyhow::anyhow!("graph declaration: {e}"))?;
        let programs = ProgramTable::decode(&declared.programs).map_err(|e| anyhow::anyhow!("program table: {e}"))?;
        let (graph, wiring) = self.wire(graph, env, |&payload| {
            let program = programs
                .get(payload as usize)
                .with_context(|| format!("payload {payload} names no program (the table has {})", programs.len()))?;
            match *program {
                Program::Library(name) => LibraryPass::from_name(name)
                    .map(|lib| (payload, ScenePassSource::Library(lib)))
                    .with_context(|| format!("no library pass `{name}`")),
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
                let PassWork::Fullscreen { sampled } = &pass.record.work else {
                    continue;
                };
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
                    anyhow::bail!(
                        "passes `{first}` and `{}` both run fullscreen program {slot} (`{label}`); a fullscreen program runs in one pass (push it once per pass)",
                        pass.name
                    );
                }
                runner[slot] = Some(&pass.name);
                shapes[slot] = Some((formats, sampled.len()));
            }
            let mut offset = 0;
            for (slot, program) in programs.iter().enumerate() {
                let Program::Fullscreen {
                    label,
                    spirv,
                    uniform_bytes,
                } = *program
                else {
                    continue;
                };
                let Some((formats, sampled)) = &shapes[slot] else {
                    continue;
                };
                built[slot] = Some((
                    FullscreenProgram::new(
                        gpu,
                        bindless,
                        label,
                        spirv,
                        uniform_bytes,
                        *sampled,
                        formats,
                        self.frames_in_flight,
                        legacy_only,
                    )?,
                    offset,
                ));
                offset += uniform_bytes as usize;
            }
            Ok(offset)
        })();
        match result {
            Ok(uniform_bytes) => Ok(ModuleGraph {
                graph,
                resources: wiring.resources,
                programs: built,
                uniform_bytes,
            }),
            Err(error) => {
                for (mut program, _) in built.into_iter().flatten() {
                    program.retire(bindless);
                    program.destroy(gpu);
                }
                Err(error)
            }
        }
    }

    /// What the library needs to agree with for `target`.
    fn env(&self, target: TargetInfo) -> Result<LibraryEnv> {
        let renderer = self.renderer.as_ref().context("terrain renderer not created")?;
        Ok(LibraryEnv {
            color_format: self.output_format,
            scene_format: renderer.formats().0,
            depth_format: DEPTH_FORMAT,
            extent: target.extent,
            units: self.unit_renderer.is_some(),
            water: self.water && renderer.has_water(),
            shadows: self.unit_renderer.is_some() && renderer.unit_shadow_map().is_real(),
            particles: self.particle_renderer.is_some(),
            encoding: self.encoding,
            levels: self.levels,
            working: self.working,
            gamut: self.gamut(),
        })
    }

    /// Size the library resources of `graph`, check its library passes and
    /// turn its payloads into [`ScenePass`]es.
    fn wire<R>(
        &self,
        mut graph: FrameGraph<R>,
        env: &LibraryEnv,
        source: impl Fn(&R) -> Result<(u32, ScenePassSource)>,
    ) -> Result<(FrameGraph<ScenePass>, Wiring)> {
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
            LibraryResource::UnitPalette => units.map(|u| u.palette_desc()),
            LibraryResource::ReadbackBuffer => offscreen.map(|o| BufferDesc {
                size: o.buffer.size,
                usage: vk::BufferUsageFlags::TRANSFER_DST,
            }),
            LibraryResource::ParticleState => self.particle_renderer.as_ref().map(|p| p.state_desc()),
            LibraryResource::ParticleKeys => self.particle_renderer.as_ref().map(|p| p.keys_desc()),
            LibraryResource::ParticleIndirect => self.particle_renderer.as_ref().map(|p| p.indirect_desc()),
            LibraryResource::LightList => self.lights.as_ref().map(|l| l.list_desc()),
            LibraryResource::LightClusters => self.lights.as_ref().map(|l| l.clusters_desc()),
            LibraryResource::ExposureHistogram => self.exposure.as_ref().map(|e| e.histogram_desc()),
            LibraryResource::ReadbackTarget
            | LibraryResource::ShadowMap
            | LibraryResource::ParticleShadows
            | LibraryResource::Exposure => None,
        })?;
        if resources.contains(&Some(LibraryResource::ReadbackTarget))
            && self.offscreen.as_ref().is_none_or(|o| o.extent != env.extent)
        {
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
                            let image = pass
                                .intents
                                .iter()
                                .find(|i| i.role == Role::TransferSrc)
                                .map(|i| i.resource)
                                .context("readback image")?;
                            let buffer = pass
                                .intents
                                .iter()
                                .find(|i| resources[i.resource.index()] == Some(LibraryResource::ReadbackBuffer))
                                .map(|i| i.resource)
                                .context("readback buffer")?;
                            PassWork::Readback { image, buffer }
                        }
                        LibraryPass::ParticlesDraw => {
                            let depth = pass
                                .intents
                                .iter()
                                .find(|i| {
                                    i.role == Role::Sampled(ShaderStage::Fragment)
                                        && resources[i.resource.index()].is_none()
                                })
                                .map(|i| i.resource)
                                .context("particles depth")?;
                            let mut color = None;
                            let mut glow = None;
                            for intent in pass
                                .intents
                                .iter()
                                .filter(|i| matches!(i.role, Role::ColorAttachment(_)))
                            {
                                let ae_framegraph::ResourceDesc::Image(image) =
                                    graph.resources[intent.resource.index()].desc
                                else {
                                    continue;
                                };
                                if image.format == env.scene_format {
                                    color = Some(intent.resource);
                                } else {
                                    glow = Some(intent.resource);
                                }
                            }
                            let color = color.context("particles colour")?;
                            let glow = glow.context("particles glow")?;
                            PassWork::Particles { color, depth, glow }
                        }
                        LibraryPass::Tonemap => {
                            let sampled: Vec<ResourceId> = pass
                                .intents
                                .iter()
                                .filter(|i| i.role == Role::Sampled(ShaderStage::Fragment))
                                .map(|i| i.resource)
                                .collect();
                            let image = |r: ResourceId| match graph.resources[r.index()].desc {
                                ae_framegraph::ResourceDesc::Image(image) => image,
                                ae_framegraph::ResourceDesc::Buffer(_) => unreachable!("post.tonemap samples an image"),
                            };
                            let is_exposure = |r: ResourceId| {
                                let i = image(r);
                                i.width == 1
                                    && i.height == 1
                                    && i.format == vk::Format::from_raw(ae_render_abi::module::EXPOSURE_FORMAT as i32)
                            };
                            let scene = sampled
                                .iter()
                                .copied()
                                .find(|&r| {
                                    image(r).width == env.extent.width
                                        && image(r).height == env.extent.height
                                        && image(r).format == env.scene_format
                                })
                                .context("tonemap scene")?;
                            let exposure = sampled.iter().copied().find(|&r| is_exposure(r));
                            let bloom = sampled.iter().copied().find(|&r| r != scene && Some(r) != exposure);
                            PassWork::Tonemap { scene, bloom, exposure }
                        }
                        LibraryPass::BloomDownsample | LibraryPass::BloomBlurX | LibraryPass::BloomBlurY => {
                            let kind = match lib {
                                LibraryPass::BloomDownsample => BloomPass::Downsample,
                                LibraryPass::BloomBlurX => BloomPass::BlurX,
                                _ => BloomPass::BlurY,
                            };
                            let sampled: Vec<ResourceId> = pass
                                .intents
                                .iter()
                                .filter(|i| i.role == Role::Sampled(ShaderStage::Fragment))
                                .map(|i| i.resource)
                                .collect();
                            let size = |r: ResourceId| match graph.resources[r.index()].desc {
                                ae_framegraph::ResourceDesc::Image(image) => (image.width, image.height),
                                ae_framegraph::ResourceDesc::Buffer(_) => (0, 0),
                            };
                            let input = sampled
                                .iter()
                                .copied()
                                .find(|&r| size(r) != (1, 1))
                                .context("bloom input")?;
                            let exposure = sampled.iter().copied().find(|&r| size(r) == (1, 1));
                            let output = pass
                                .intents
                                .iter()
                                .find(|i| matches!(i.role, Role::ColorAttachment(_)))
                                .map(|i| i.resource)
                                .context("bloom output")?;
                            PassWork::Bloom {
                                pass: kind,
                                input,
                                output,
                                exposure,
                            }
                        }
                        LibraryPass::ExposureMeter => {
                            // Format and size, as the library checks them.
                            let spec = |r: ResourceId| match graph.resources[r.index()].desc {
                                ae_framegraph::ResourceDesc::Image(image) => {
                                    Some((image.format, image.width, image.height))
                                }
                                ae_framegraph::ResourceDesc::Buffer(_) => None,
                            };
                            let (w, h) = (env.extent.width, env.extent.height);
                            let format = |raw: u32| vk::Format::from_raw(raw as i32);
                            let sampled: Vec<ResourceId> = pass
                                .intents
                                .iter()
                                .filter(|i| i.role == Role::Sampled(ShaderStage::Compute))
                                .map(|i| i.resource)
                                .collect();
                            let find =
                                |want: (vk::Format, u32, u32)| sampled.iter().copied().find(|&r| spec(r) == Some(want));
                            let scene = find((env.scene_format, w, h)).context("exposure meter scene")?;
                            let depth = find((env.depth_format, w, h));
                            let glow = find((format(ae_render_abi::module::GLOW_FORMAT), w, h));
                            PassWork::ExposureMeter { scene, depth, glow }
                        }
                        LibraryPass::PostEncode => {
                            let display = pass
                                .intents
                                .iter()
                                .find(|i| i.role == Role::Sampled(ShaderStage::Fragment))
                                .map(|i| i.resource)
                                .context("post.encode display")?;
                            PassWork::Encode { display }
                        }
                        lib => PassWork::Library(lib),
                    }
                }
                ScenePassSource::Fullscreen => {
                    let name = &pass.name;
                    anyhow::ensure!(
                        pass.kind == PassKind::Graphics,
                        "fullscreen pass `{name}` must be a Graphics pass"
                    );
                    let mut sampled = Vec::new();
                    for intent in &pass.intents {
                        let r = &graph.resources[intent.resource.index()];
                        match intent.role {
                            Role::ColorAttachment(_) => {}
                            Role::Sampled(ShaderStage::Fragment) if r.desc.is_image() => sampled.push(intent.resource),
                            role => anyhow::bail!(
                                "fullscreen pass `{name}` uses `{}` as {role:?}; it takes colour attachments and fragment-sampled images",
                                r.name
                            ),
                        }
                    }
                    anyhow::ensure!(
                        pass.color_attachments().count() > 0,
                        "fullscreen pass `{name}` has no colour attachment"
                    );
                    PassWork::Fullscreen { sampled }
                }
            };
            passes.push(ScenePass { program, work });
        }
        library::check_graph(&graph, &resources, env, |i| match &passes[i].work {
            PassWork::Library(lib) => Some(*lib),
            PassWork::ExposureMeter { .. } => Some(LibraryPass::ExposureMeter),
            PassWork::Encode { .. } => Some(LibraryPass::PostEncode),
            PassWork::Bloom { pass, .. } => Some(match pass {
                BloomPass::Downsample => LibraryPass::BloomDownsample,
                BloomPass::BlurX => LibraryPass::BloomBlurX,
                BloomPass::BlurY => LibraryPass::BloomBlurY,
            }),
            _ => None,
        })?;
        let glow = library::glow_resource(&graph);
        let graph = FrameGraph {
            resources: graph.resources,
            passes: graph
                .passes
                .into_iter()
                .zip(passes)
                .map(|(p, record)| ae_framegraph::Pass {
                    name: p.name,
                    kind: p.kind,
                    scope: p.scope,
                    intents: p.intents,
                    conditional: p.conditional,
                    record,
                })
                .collect(),
            roots: graph.roots,
            plan_version: graph.plan_version,
        };
        Ok((graph, Wiring { resources, glow }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_peak_is_the_headroom_over_the_signal_reference() {
        // The compositor maps the signal's 203-nit reference onto its own
        // 248, so the signal's peak is 203 × SDL's headroom (it shows at the
        // panel's 441), not the panel's absolute 441 (which would show at
        // 539 and be tone-mapped again).
        let hdr = HdrSettings {
            paper_white: None,
            peak: None,
            headroom: 441.0 / 248.0,
            display_peak: 441.0,
        };
        for encoding in [OutputEncoding::Hdr10Pq, OutputEncoding::ScRgb] {
            let levels = hdr.levels(encoding);
            assert_eq!(levels.paper_white, 203.0, "{encoding:?}");
            assert!(
                (levels.peak - 203.0 * 441.0 / 248.0).abs() < 1e-3,
                "{encoding:?}: {levels:?}"
            );
            assert!((levels.headroom() - 441.0 / 248.0).abs() < 1e-5);
        }
        // KDE's SDR mode (SDR white 155, peak 200): the headroom is above 1,
        // so no floor at the paper white is needed.
        let sdr_mode = HdrSettings {
            headroom: 200.0 / 155.0,
            display_peak: 200.0,
            ..hdr
        };
        assert!((sdr_mode.levels(OutputEncoding::Hdr10Pq).peak - 203.0 * 200.0 / 155.0).abs() < 1e-3);
        // Overrides are in the signal's nits and taken as given.
        let forced = HdrSettings {
            paper_white: Some(100.0),
            peak: Some(400.0),
            ..hdr
        };
        assert_eq!(
            forced.levels(OutputEncoding::Hdr10Pq),
            DisplayLevels {
                paper_white: 100.0,
                peak: 400.0
            }
        );
    }
}
