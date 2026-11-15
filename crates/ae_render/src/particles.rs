//! Particles: a GPU pool of billboards, spawned in bursts, moved, sorted
//! back to front by their centres and blended over the scene.
//!
//! * One blend for every particle: premultiplied "over". An *emissive*
//!   effect (fire, sparks, flashes) writes alpha 0, so it adds light, and
//!   it is still sorted with the rest: fire behind smoke is dimmed by the
//!   smoke in front of it. *Media* (smoke, dust) cover what is behind them.
//! * The sort key is `(far-to-near depth, 16 bits) << 16 | slot`. Keys are
//!   unique, so the radix sort orders them the same whatever order the
//!   simulation appended them in: frames are deterministic for a given
//!   sequence of bursts and times.
//! * Soft edges: the draw samples the depth buffer and fades a particle
//!   where it meets the scene, treating it as a sphere of its size (no hard
//!   line where smoke meets the ground); it tests depth itself that way.
//! * The pool is a ring of [`ParticlesSetup::capacity`] slots (at most
//!   [`MAX_CAPACITY`], the key's 16 slot bits): a burst takes the slots
//!   after the last one, overwriting the oldest particles when it is full.
//!
//! Bursts come from a [`ParticleFeed`] (the client fills it from what the
//! observer sees). Effects are data ([`EffectDef`]); [`default_effects`]
//! has the engine's set. In the frame graph, two passes: `particles sim`
//! (compute: spawn, move, cull, key, sort; it records its own barriers
//! between those steps, all inside its buffers) and `particles` (graphics:
//! one instanced draw from the indirect buffer).
//!
//! The draw is fill-bound, so by default it runs at half resolution
//! ([`ParticlesSetup::downscale`]): the scene depth is reduced to the
//! farthest of each 2x2 block, the particles soften against that into a
//! half-resolution RGBA16F target, and a composite puts it over the scene,
//! bilinear where the four nearest texels' depths match the pixel's and
//! the nearest-depth texel across an edge (no smoke bleeding onto a unit or
//! a cliff in front of it). All of it records inside the `particles` pass.
//!
//! With the units' shadow map, the particles also cast coloured shadows
//! ([`crate::shadow`]): the `shadows.particles` pass draws every live
//! particle that is not emissive (smoke, dust) from the sun into
//! `shadows.color`, optical depth per channel summed and the nearest
//! particle's depth kept by a max, and the terrain, the units and the
//! water multiply their sunlight by what it lets through.

use crate::device::GpuDevice;
use crate::executor::{BufferBinding, ImageBinding, PassContext};
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::pipeline::{self, Blend, GraphicsDesc, spirv};
use crate::terrain::renderer::{TerrainInfo, TerrainRenderer, frustum_planes};
use crate::upload;
use ae_framegraph::{BufferDesc, ImageDesc};
use anyhow::{Context as _, Result, ensure};
use ash::vk;
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use std::sync::{Arc, Mutex};

/// Most slots a pool can have (the sort key's slot bits).
pub const MAX_CAPACITY: u32 = 1 << 16;
/// Most effects in a table.
pub const MAX_EFFECTS: usize = 16;
/// Most bursts spawned in one frame (more wait for the next).
pub const MAX_BURSTS: usize = 1024;

const STATE_BYTES: u64 = 48;
const SORT_BLOCKS: u32 = 4;
const SORT_WORKGROUP: u32 = 256;
const SORT_BINS: u64 = 256;
const VERTICES: u32 = 6;
/// Byte offset of the coloured shadows' draw in the indirect buffer.
const CASTER_DRAW: u64 = 32;
/// The puff texture's side (texels).
const PUFF: usize = 128;

const IDENTITY_STATE: u64 = (0xAE7 << 40) | (5 << 32);
const IDENTITY_KEYS: u64 = (0xAE7 << 40) | (5 << 32) | 1;
const IDENTITY_INDIRECT: u64 = (0xAE7 << 40) | (5 << 32) | 2;

/// One effect: how its particles start, move and look. Sizes in world
/// units (metres), times in seconds; a burst's `scale` multiplies sizes,
/// speeds and the spawn spread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectDef {
    /// Linear colour and opacity at birth and at death (interpolated over
    /// the life). For an emissive effect the colour is the light it adds
    /// (may exceed 1) and the opacity scales it.
    pub color_start: [f32; 4],
    pub color_end: [f32; 4],
    /// Diameter at birth and at death.
    pub size: [f32; 2],
    /// Life, chosen per particle in this range.
    pub life: [f32; 2],
    /// Initial speed range.
    pub speed: [f32; 2],
    /// 0: directions uniform over the sphere; 1: straight up.
    pub up: f32,
    /// Particles start in a ball of this radius around the burst.
    pub spread: f32,
    /// Downward acceleration (negative rises: hot smoke).
    pub gravity: f32,
    /// Velocity lost per second (exponential).
    pub drag: f32,
    /// Adds light (alpha 0) instead of covering. Emissive effects cast no
    /// coloured shadow; the others (smoke, dust) do.
    pub emissive: bool,
    /// Fades in over this fraction of the life (0: born at full opacity).
    pub fade_in: f32,
}

impl EffectDef {
    fn words(&self) -> [[f32; 4]; 5] {
        [
            self.color_start,
            self.color_end,
            [self.size[0], self.size[1], self.life[0], self.life[1].max(self.life[0])],
            [self.speed[0], self.speed[1], self.up, self.spread],
            [self.gravity, self.drag, self.emissive as u32 as f32, self.fade_in],
        ]
    }

    fn max_life(&self) -> f32 {
        self.life[0].max(self.life[1])
    }
}

/// Indices into [`default_effects`].
pub mod effect {
    pub const FLASH: u32 = 0;
    pub const FIRE: u32 = 1;
    pub const SPARKS: u32 = 2;
    pub const SMOKE: u32 = 3;
    pub const DUST: u32 = 4;
}

/// The engine's effects, indexed by [`effect`].
pub fn default_effects() -> Vec<EffectDef> {
    vec![
        // FLASH: a short bright ball.
        EffectDef {
            color_start: [6.0, 4.2, 2.2, 1.0],
            color_end: [1.2, 0.5, 0.15, 0.0],
            size: [14.0, 26.0],
            life: [0.12, 0.2],
            speed: [0.0, 1.0],
            up: 0.0,
            spread: 1.0,
            gravity: 0.0,
            drag: 0.0,
            emissive: true,
            fade_in: 0.0,
        },
        // FIRE: rising flames that cool from yellow to dark red.
        EffectDef {
            color_start: [1.9, 0.85, 0.22, 0.9],
            color_end: [0.5, 0.08, 0.02, 0.0],
            size: [5.0, 11.0],
            life: [0.35, 0.9],
            speed: [3.0, 14.0],
            up: 0.55,
            spread: 3.0,
            gravity: -9.0,
            drag: 2.0,
            emissive: true,
            fade_in: 0.08,
        },
        // SPARKS: small, fast, falling.
        EffectDef {
            color_start: [5.0, 3.6, 1.6, 1.0],
            color_end: [1.6, 0.4, 0.05, 0.0],
            size: [0.9, 0.5],
            life: [0.5, 1.3],
            speed: [25.0, 60.0],
            up: 0.35,
            spread: 0.5,
            gravity: 22.0,
            drag: 0.6,
            emissive: true,
            fade_in: 0.0,
        },
        // SMOKE: dark, rising, spreading, thinning.
        EffectDef {
            color_start: [0.09, 0.085, 0.08, 0.8],
            color_end: [0.36, 0.36, 0.36, 0.0],
            size: [7.0, 30.0],
            life: [2.5, 5.5],
            speed: [3.0, 10.0],
            up: 0.45,
            spread: 6.0,
            gravity: -2.5,
            drag: 0.7,
            emissive: false,
            fade_in: 0.1,
        },
        // DUST: low, brown, short.
        EffectDef {
            color_start: [0.42, 0.36, 0.27, 0.45],
            color_end: [0.55, 0.5, 0.42, 0.0],
            size: [4.0, 15.0],
            life: [1.2, 2.8],
            speed: [2.0, 7.0],
            up: 0.25,
            spread: 2.0,
            gravity: -0.4,
            drag: 1.2,
            emissive: false,
            fade_in: 0.05,
        },
    ]
}

/// `count` particles of effect `effect` from `pos` (world space).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Burst {
    /// World position; with `ground`, `pos[1]` is the height above the
    /// terrain (or the water surface where it is higher).
    pub pos: [f32; 3],
    pub ground: bool,
    pub effect: u32,
    pub count: u32,
    /// Multiplies the effect's sizes, speeds and spread.
    pub scale: f32,
    /// Varies the particles between bursts.
    pub seed: u32,
}

/// An explosion of size `scale` (1: a small vehicle's) with the default
/// effects: flash, fire, sparks, smoke.
pub fn explosion(pos: [f32; 3], ground: bool, scale: f32, seed: u32) -> [Burst; 4] {
    let n = |base: f32| (base * scale.clamp(0.5, 4.0)).round() as u32;
    [
        Burst { pos, ground, effect: effect::FLASH, count: 2, scale, seed },
        Burst { pos, ground, effect: effect::FIRE, count: n(24.0), scale, seed: seed ^ 0x9e37_79b9 },
        Burst { pos, ground, effect: effect::SPARKS, count: n(40.0), scale, seed: seed ^ 0x7f4a_7c15 },
        Burst { pos, ground, effect: effect::SMOKE, count: n(20.0), scale, seed: seed ^ 0x2545_f491 },
    ]
}

/// Where bursts are queued for the render thread (taken every frame).
#[derive(Debug, Default)]
pub struct ParticleFeed {
    queued: Mutex<Vec<Burst>>,
}

impl ParticleFeed {
    pub fn emit(&self, burst: Burst) {
        self.queued.lock().unwrap_or_else(|p| p.into_inner()).push(burst);
    }

    pub fn emit_all(&self, bursts: &[Burst]) {
        self.queued.lock().unwrap_or_else(|p| p.into_inner()).extend_from_slice(bursts);
    }

    /// Every queued burst, oldest first.
    pub fn take_all(&self) -> Vec<Burst> {
        std::mem::take(&mut *self.queued.lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// Up to `max` queued bursts, oldest first; the rest stay queued.
    fn take(&self, max: usize) -> Vec<Burst> {
        let mut queued = self.queued.lock().unwrap_or_else(|p| p.into_inner());
        let n = queued.len().min(max);
        queued.drain(..n).collect()
    }
}

pub struct ParticlesSetup {
    pub feed: Arc<ParticleFeed>,
    /// Pool slots (at most [`MAX_CAPACITY`]).
    pub capacity: u32,
    /// At most [`MAX_EFFECTS`]; a burst naming another effect spawns nothing.
    pub effects: Vec<EffectDef>,
    /// 1: draw at full resolution, straight into the scene. 2: draw into a
    /// half-resolution target and composite it over the scene (the draw is
    /// fill-bound; see [`ParticleRenderer::record_draw`]).
    pub downscale: u32,
    /// How overlapping particles blend ([`ParticleBlend::Sorted`] unless
    /// comparing; the others need `downscale` 2).
    pub blend: ParticleBlend,
}

impl ParticlesSetup {
    /// `capacity` slots, the default effects, half resolution, sorted.
    pub fn new(feed: Arc<ParticleFeed>, capacity: u32) -> Self {
        Self { feed, capacity, effects: default_effects(), downscale: 2, blend: ParticleBlend::Sorted }
    }
}

/// How the particles that cover a pixel are blended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleBlend {
    /// Sorted back to front by centre depth, premultiplied "over".
    Sorted,
    /// Moment-based order-independent transparency (four power moments,
    /// Muenstermann et al. 2018): unsorted, three passes.
    Moments,
    /// Exact: every fragment stored per pixel, sorted by its sphere's front
    /// depth and blended. For tests (needs `fragmentStoresAndAtomics`; a
    /// list of [`REFERENCE_LAYERS`] per pixel, magenta past that).
    Reference,
    /// Exact, as a volume: every fragment's span (its sphere's chord along
    /// the pixel's depth, capped at the scene) stored per pixel; the spans
    /// cut the depth into intervals, each integrated as the media covering
    /// it (extinction `-ln(1 - a) / length`, colour and emission per
    /// metre). Overlapping puffs mix by how much of their thickness
    /// overlaps. For tests, as [`ParticleBlend::Reference`].
    Volume,
    /// Thickness-aware, unsorted: a first pass finds each pixel's depth
    /// range of particles (nearest front, farthest back); a second adds
    /// each particle's optical depth and light into [`SLICES`] equal slices
    /// of that range, by how much of each its chord covers; a resolve
    /// integrates the slices front to back, each a homogeneous medium. A
    /// lone puff comes out as blended; overlapping ones mix by thickness,
    /// continuously.
    Slices,
}

/// Depth slices per pixel in [`ParticleBlend::Slices`].
pub const SLICES: usize = 4;

impl ParticleBlend {
    /// Stores every fragment per pixel ([`ParticleBlend::Reference`],
    /// [`ParticleBlend::Volume`]).
    pub fn lists(self) -> bool {
        matches!(self, Self::Reference | Self::Volume)
    }
}

/// Fragments per pixel in [`ParticleBlend::Reference`].
pub const REFERENCE_LAYERS: u64 = 64;

/// The reduced-resolution targets of one frame slot.
struct HalfTargets {
    extent: vk::Extent2D,
    /// Premultiplied particles (RGBA16F: emissive colours exceed 1).
    color: OwnedImage,
    /// The farthest scene depth of each 2x2 block (R32F, reverse-Z).
    depth: OwnedImage,
    /// [`ParticleBlend::Slices`]: the range (RG32F), the slices
    /// (RGBA16F each).
    /// [`ParticleBlend::Moments`]: b0 (R32F), b1..b4 (RGBA32F), the media
    /// and the emissive sums (RGBA16F).
    moments: Vec<OwnedImage>,
    /// [`ParticleBlend::Reference`]: per-pixel counts and fragments.
    reference: Vec<OwnedBuffer>,
}

impl HalfTargets {
    fn destroy(mut self, gpu: &GpuDevice) {
        self.color.destroy(gpu);
        self.depth.destroy(gpu);
        for image in &mut self.moments {
            image.destroy(gpu);
        }
        for buffer in &mut self.reference {
            buffer.destroy(gpu);
        }
    }
}

const SLICE_FORMATS: [vk::Format; 1 + SLICES] = [vk::Format::R32G32_SFLOAT, vk::Format::R16G16B16A16_SFLOAT, vk::Format::R16G16B16A16_SFLOAT, vk::Format::R16G16B16A16_SFLOAT, vk::Format::R16G16B16A16_SFLOAT];
const MOMENT_FORMATS: [vk::Format; 4] = [vk::Format::R32_SFLOAT, vk::Format::R32G32B32A32_SFLOAT, vk::Format::R16G16B16A16_SFLOAT, vk::Format::R16G16B16A16_SFLOAT];

const HALF_COLOR: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const HALF_DEPTH: vk::Format = vk::Format::R32_SFLOAT;

/// Per-frame inputs.
#[derive(Clone, Copy, Debug)]
pub struct ParticleFrame {
    pub view_proj: Mat4,
    pub cam_pos: Vec3,
    /// The camera's right, up and forward axes.
    pub axes: [Vec3; 3],
    /// The projection's near plane (infinite reverse-Z: depth = near / z).
    pub near: f32,
    pub extent: vk::Extent2D,
    /// Simulation time in seconds (it steps by the difference, at most 0.1 s).
    pub time: f32,
    /// Fog colour (linear) and density; density 0 turns fog off.
    pub fog: [f32; 4],
    /// Shade smoke by the hills' shadows (when the terrain has them).
    pub shadows: bool,
    /// The units' shadow cascades when the graph draws the particles'
    /// coloured shadows this frame (else the pass draws nothing).
    pub unit_shadows: Option<crate::shadow::Cascades>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FrameUbo {
    view_proj: [f32; 16],
    cam_pos: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    forward: [f32; 4],
    frustum: [[f32; 4]; 6],
    /// dt, time, 1/width, 1/height.
    step: [f32; 4],
    /// capacity, spawn records, spawn total, ring head.
    pool: [u32; 4],
    /// flags (1: hills' shadows), find the depth range (moments), list
    /// the shadow casters, 0.
    flags: [u32; 4],
    map: [f32; 4],
    height: [f32; 4],
    sun: [f32; 4],
    fog: [f32; 4],
    shadow: crate::shadow::CascadeUniforms,
    effects: [[[f32; 4]; 5]; MAX_EFFECTS],
}

/// A burst as the spawn shader reads it.
#[repr(C)]
#[derive(Clone, Copy)]
struct SpawnRecord {
    /// xyz, scale.
    pos: [f32; 4],
    /// effect (bit 16: on the ground), count, first index in this
    /// frame's spawns, seed.
    info: [u32; 4],
}

pub struct ParticleRenderer {
    capacity: u32,
    info: TerrainInfo,
    effects: Vec<EffectDef>,
    feed: Arc<ParticleFeed>,
    shadows: bool,
    set_layout: vk::DescriptorSetLayout,
    /// Set 1: the depth buffer, for the draw alone (written when it
    /// records; set 0 is bound by the simulation earlier in the frame).
    depth_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    desc_pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    depth_sets: Vec<vk::DescriptorSet>,
    depth_sampler: vk::Sampler,
    /// Every particle's shape: a soft ball broken up by noise (mipmapped).
    puff: OwnedImage,
    puff_sampler: vk::Sampler,
    spawn: vk::Pipeline,
    sim: vk::Pipeline,
    histogram: vk::Pipeline,
    scatter: vk::Pipeline,
    draw: vk::Pipeline,
    state: OwnedBuffer,
    keys: OwnedBuffer,
    keys_tmp: OwnedBuffer,
    histograms: OwnedBuffer,
    indirect: OwnedBuffer,
    ubos: Vec<OwnedBuffer>,
    spawns: Vec<OwnedBuffer>,
    /// Per slot: `(records, total)` spawned by that frame.
    spawned: Vec<(u32, u32)>,
    /// Per slot: the frame has live particles (it simulates and draws).
    active: Vec<bool>,
    head: u32,
    last_time: Option<f32>,
    /// Simulation time when the last particle spawned so far dies.
    alive_until: f32,
    sim_time: f32,
    downscale: u32,
    /// Per slot, with `downscale` 2: made when the slot first draws, and
    /// again when the extent changes.
    half: Vec<Option<HalfTargets>>,
    /// Per slot: set 1 of the half-resolution draw (the half depth, the
    /// puff).
    half_sets: Vec<vk::DescriptorSet>,
    /// Set 1 of the composite: half colour, half depth, full depth.
    composite_layout: vk::DescriptorSetLayout,
    composite_pipeline_layout: vk::PipelineLayout,
    composite_sets: Vec<vk::DescriptorSet>,
    depth_down: vk::Pipeline,
    composite: vk::Pipeline,
    blend: ParticleBlend,
    /// [`ParticleBlend::Moments`]: generate, shade, resolve.
    /// [`ParticleBlend::Reference`]: store, resolve. Else empty.
    oit: Vec<vk::Pipeline>,
    /// The coloured shadows: set 0 and the cascade as a push constant.
    caster_layout: vk::PipelineLayout,
    caster: vk::Pipeline,
    /// Per slot: the frame draws the coloured shadows.
    casting: Vec<bool>,
}

impl ParticleRenderer {
    pub fn new(gpu: &GpuDevice, terrain: &TerrainRenderer, setup: &ParticlesSetup, color_format: vk::Format, frames_in_flight: usize) -> Result<Self> {
        ensure!((1..=MAX_CAPACITY).contains(&setup.capacity), "particle capacity {} (1..={MAX_CAPACITY})", setup.capacity);
        ensure!(setup.effects.len() <= MAX_EFFECTS, "{} particle effects (at most {MAX_EFFECTS})", setup.effects.len());
        ensure!(matches!(setup.downscale, 1 | 2), "particle downscale {} (1 or 2)", setup.downscale);
        ensure!(setup.blend == ParticleBlend::Sorted || setup.downscale == 2, "{:?} particles need downscale 2", setup.blend);
        ensure!(!setup.blend.lists() || gpu.fragment_stores, "the particles' reference needs fragmentStoresAndAtomics");
        let formats: &[vk::Format] = match setup.blend {
            ParticleBlend::Moments => &MOMENT_FORMATS,
            ParticleBlend::Slices => &SLICE_FORMATS,
            _ => &[],
        };
        for &format in formats {
            let features = unsafe { gpu.instance.instance.get_physical_device_format_properties(gpu.physical, format) }.optimal_tiling_features;
            ensure!(features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND), "{format:?} cannot be blended into ({:?} particles)", setup.blend);
        }
        let device = &gpu.device;
        let capacity = setup.capacity;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        // Zeroed: every slot starts dead (life 0).
        let state = upload::buffer(gpu, &vec![0u8; (capacity as u64 * STATE_BYTES) as usize], storage, "particle state")?;
        // The visible keys, then the coloured shadows' casters.
        let keys = OwnedBuffer::new(gpu, BufferDesc { size: capacity as u64 * 8, usage: storage }, MemoryLocation::GpuOnly, "particle keys")?;
        let keys_tmp = OwnedBuffer::new(gpu, BufferDesc { size: capacity as u64 * 4, usage: storage }, MemoryLocation::GpuOnly, "particle keys (sort)")?;
        let groups = Self::sort_groups(capacity);
        let histograms = OwnedBuffer::new(gpu, BufferDesc { size: groups as u64 * SORT_BINS * 4, usage: storage }, MemoryLocation::GpuOnly, "particle sort histograms")?;
        let indirect = upload::buffer(gpu, pipeline::push_bytes(&[VERTICES, 0, 0, 0, 0x7f7f_ffff, 0, 0, 0, 4, 0, 0, 0]), Self::indirect_usage(), "particle indirect")?;
        let ubos = (0..frames_in_flight)
            .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: size_of::<FrameUbo>() as u64, usage: vk::BufferUsageFlags::UNIFORM_BUFFER }, MemoryLocation::CpuToGpu, "particle frame"))
            .collect::<Result<Vec<_>>>()?;
        let spawns = (0..frames_in_flight)
            .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: (MAX_BURSTS * size_of::<SpawnRecord>()) as u64, usage: storage }, MemoryLocation::CpuToGpu, "particle spawns"))
            .collect::<Result<Vec<_>>>()?;
        let mips = puff_mips();
        let puff = upload::image(gpu, vk::Format::R8_UNORM, PUFF as u32, PUFF as u32, &mips.iter().map(Vec::as_slice).collect::<Vec<_>>(), "particle puff")?;
        let puff_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .max_lod(vk::LOD_CLAMP_NONE)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };
        let depth_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };

        let all = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let (shadow_view, shadow_sampler) = terrain.shadow_texture();
        let (height_view, height_sampler) = terrain.height_texture();
        let (set_layout, depth_layout, pipeline_layout, desc_pool, sets, depth_sets, half_sets, composite_layout, composite_pipeline_layout, composite_sets) = unsafe {
            let ssbo = vk::DescriptorType::STORAGE_BUFFER;
            let image = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
            let set_layout = pipeline::set_layout(
                device,
                &[(vk::DescriptorType::UNIFORM_BUFFER, all), (ssbo, all), (ssbo, all), (ssbo, all), (ssbo, all), (ssbo, all), (ssbo, all), (image, all), (image, all), (image, all)],
            )?;
            // Set 1 of the draws: the depth, the puff; the moments' four
            // targets or the slices' five; the reference's counts and
            // fragments.
            let f = vk::ShaderStageFlags::FRAGMENT;
            let depth_layout = pipeline::set_layout(device, &[(image, f), (image, f), (image, f), (image, f), (image, f), (image, f), (image, f), (ssbo, f), (ssbo, f)])?;
            // The sort's push constant: shift, direction.
            let pipeline_layout = pipeline::pipeline_layout(device, &[set_layout, depth_layout], 8, vk::ShaderStageFlags::COMPUTE)?;
            let fragment = vk::ShaderStageFlags::FRAGMENT;
            let composite_layout = pipeline::set_layout(device, &[(image, fragment), (image, fragment), (image, fragment)])?;
            let composite_pipeline_layout = pipeline::pipeline_layout(device, &[set_layout, composite_layout], 0, vk::ShaderStageFlags::FRAGMENT)?;
            let n = frames_in_flight as u32;
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: n },
                vk::DescriptorPoolSize { ty: ssbo, descriptor_count: 10 * n },
                vk::DescriptorPoolSize { ty: image, descriptor_count: 20 * n },
            ];
            let desc_pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(4 * n).pool_sizes(&sizes), None)?;
            let allocate = |layout: vk::DescriptorSetLayout| {
                let layouts = vec![layout; frames_in_flight];
                device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&layouts))
            };
            let sets = allocate(set_layout)?;
            let depth_sets = allocate(depth_layout)?;
            let half_sets = allocate(depth_layout)?;
            let composite_sets = allocate(composite_layout)?;
            let shape = [vk::DescriptorImageInfo { sampler: puff_sampler, image_view: puff.view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
            // The puff, in both the full- and the half-resolution draw's set
            // (the downsample, bound with the full one, does not read it).
            let writes: Vec<_> = depth_sets.iter().chain(&half_sets).map(|&set| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(image).image_info(&shape)).collect();
            device.update_descriptor_sets(&writes, &[]);
            for (slot, &set) in sets.iter().enumerate() {
                let info = |buffer: &OwnedBuffer| [vk::DescriptorBufferInfo { buffer: buffer.buffer, offset: 0, range: vk::WHOLE_SIZE }];
                let buffers = [info(&ubos[slot]), info(&state), info(&keys), info(&indirect), info(&spawns[slot]), info(&keys_tmp), info(&histograms)];
                let mut writes: Vec<vk::WriteDescriptorSet> = buffers
                    .iter()
                    .enumerate()
                    .map(|(i, info)| {
                        let ty = if i == 0 { vk::DescriptorType::UNIFORM_BUFFER } else { ssbo };
                        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(ty).buffer_info(info)
                    })
                    .collect();
                let shadow = [vk::DescriptorImageInfo { sampler: shadow_sampler, image_view: shadow_view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(7).descriptor_type(image).image_info(&shadow));
                let height = [vk::DescriptorImageInfo { sampler: height_sampler, image_view: height_view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(8).descriptor_type(image).image_info(&height));
                // The puff again, for the coloured shadows (bound with set 0
                // alone: set 1 is rewritten when the draw records).
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(9).descriptor_type(image).image_info(&shape));
                device.update_descriptor_sets(&writes, &[]);
            }
            (set_layout, depth_layout, pipeline_layout, desc_pool, sets, depth_sets, half_sets, composite_layout, composite_pipeline_layout, composite_sets)
        };
        let compute = |code: &[u8], what: &str| pipeline::compute_pipeline(device, code, pipeline_layout).with_context(|| format!("particle {what} pipeline"));
        let spawn = compute(spirv!("particles_spawn.comp"), "spawn")?;
        let sim = compute(spirv!("particles_sim.comp"), "sim")?;
        let histogram = compute(spirv!("particles_sort_histogram.comp"), "sort histogram")?;
        let scatter = compute(spirv!("particles_sort_scatter.comp"), "sort scatter")?;
        let draw = pipeline::graphics_pipeline(
            device,
            &GraphicsDesc {
                vertex: spirv!("particles.vert"),
                fragment: spirv!("particles.frag"),
                layout: pipeline_layout,
                color_formats: &[if setup.downscale > 1 { HALF_COLOR } else { color_format }],
                // Depth is sampled: the shader tests (and fades) by itself.
                depth: None,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                blend: Blend::Premultiplied,
            },
        )
        .context("particle pipeline")?;
        let fullscreen = |fragment: &[u8], layout, format, blend| {
            pipeline::graphics_pipeline(device, &GraphicsDesc { vertex: spirv!("fullscreen.vert"), fragment, layout, color_formats: &[format], depth: None, topology: vk::PrimitiveTopology::TRIANGLE_LIST, blend })
        };
        let depth_down = fullscreen(spirv!("particles_depth_down.frag"), pipeline_layout, HALF_DEPTH, Blend::Off).context("particle depth downsample pipeline")?;
        let composite = fullscreen(spirv!("particles_composite.frag"), composite_pipeline_layout, color_format, Blend::Premultiplied).context("particle composite pipeline")?;
        let particle_pipeline = |fragment: &[u8], formats: &[vk::Format], blend| {
            pipeline::graphics_pipeline(device, &GraphicsDesc { vertex: spirv!("particles.vert"), fragment, layout: pipeline_layout, color_formats: formats, depth: None, topology: vk::PrimitiveTopology::TRIANGLE_LIST, blend })
        };
        let oit = match setup.blend {
            ParticleBlend::Sorted => Vec::new(),
            ParticleBlend::Moments => vec![
                particle_pipeline(spirv!("particles_moments.frag"), &MOMENT_FORMATS[..2], Blend::Additive).context("particle moments pipeline")?,
                particle_pipeline(spirv!("particles_moments_shade.frag"), &MOMENT_FORMATS[2..], Blend::Additive).context("particle moments shade pipeline")?,
                fullscreen(spirv!("particles_moments_resolve.frag"), pipeline_layout, HALF_COLOR, Blend::Off).context("particle moments resolve pipeline")?,
            ],
            ParticleBlend::Reference => vec![
                particle_pipeline(spirv!("particles_reference.frag"), &[], Blend::Off).context("particle reference pipeline")?,
                fullscreen(spirv!("particles_reference_resolve.frag"), pipeline_layout, HALF_COLOR, Blend::Off).context("particle reference resolve pipeline")?,
            ],
            ParticleBlend::Slices => vec![
                particle_pipeline(spirv!("particles_slices_range.frag"), &SLICE_FORMATS[..1], Blend::Max).context("particle slices range pipeline")?,
                particle_pipeline(spirv!("particles_slices.frag"), &SLICE_FORMATS[1..], Blend::Additive).context("particle slices pipeline")?,
                fullscreen(spirv!("particles_slices_resolve.frag"), pipeline_layout, HALF_COLOR, Blend::Off).context("particle slices resolve pipeline")?,
            ],
            ParticleBlend::Volume => vec![
                particle_pipeline(spirv!("particles_volume.frag"), &[], Blend::Off).context("particle volume pipeline")?,
                fullscreen(spirv!("particles_volume_resolve.frag"), pipeline_layout, HALF_COLOR, Blend::Off).context("particle volume resolve pipeline")?,
            ],
        };
        let caster_layout = pipeline::pipeline_layout(device, &[set_layout], 4, vk::ShaderStageFlags::VERTEX)?;
        let caster = pipeline::graphics_pipeline(
            device,
            &GraphicsDesc {
                vertex: spirv!("particles_shadow.vert"),
                fragment: spirv!("particles_shadow.frag"),
                layout: caster_layout,
                color_formats: &[crate::shadow::color_format()],
                depth: None,
                topology: vk::PrimitiveTopology::TRIANGLE_STRIP,
                blend: Blend::AdditiveMaxAlpha,
            },
        )
        .context("particle shadow pipeline")?;
        Ok(Self {
            capacity,
            info: *terrain.info(),
            effects: setup.effects.clone(),
            feed: setup.feed.clone(),
            shadows: terrain.has_shadows(),
            set_layout,
            depth_layout,
            pipeline_layout,
            desc_pool,
            sets,
            depth_sets,
            depth_sampler,
            puff,
            puff_sampler,
            spawn,
            sim,
            histogram,
            scatter,
            draw,
            state,
            keys,
            keys_tmp,
            histograms,
            indirect,
            ubos,
            spawns,
            spawned: vec![(0, 0); frames_in_flight],
            active: vec![false; frames_in_flight],
            head: 0,
            last_time: None,
            alive_until: f32::NEG_INFINITY,
            sim_time: 0.0,
            downscale: setup.downscale,
            half: (0..frames_in_flight).map(|_| None).collect(),
            half_sets,
            composite_layout,
            composite_pipeline_layout,
            composite_sets,
            depth_down,
            composite,
            blend: setup.blend,
            oit,
            caster_layout,
            caster,
            casting: vec![false; frames_in_flight],
        })
    }

    /// `slot`'s half-resolution targets for a frame of `extent`, made or
    /// remade (the slot's previous submission has completed), and the sets
    /// that read them.
    fn ensure_half(&mut self, gpu: &GpuDevice, slot: usize, extent: vk::Extent2D) -> Result<()> {
        let half = vk::Extent2D { width: extent.width.div_ceil(2).max(1), height: extent.height.div_ceil(2).max(1) };
        if self.half[slot].as_ref().is_some_and(|h| h.extent == half) {
            return Ok(());
        }
        if let Some(old) = self.half[slot].take() {
            old.destroy(gpu);
        }
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
        let image = |format, name: &str| OwnedImage::new(gpu, ImageDesc::color_2d(format, half.width, half.height, usage), name);
        // Built up so that a failure part way frees what was made.
        let mut targets = HalfTargets { extent: half, color: image(HALF_COLOR, "particles (half resolution)")?, depth: image(HALF_DEPTH, "particle depth (half resolution)")?, moments: Vec::new(), reference: Vec::new() };
        let made = (|| -> Result<()> {
            if self.blend == ParticleBlend::Moments {
                for (format, name) in MOMENT_FORMATS.iter().zip(["particle moment b0", "particle moments", "particle media", "particle emissive"]) {
                    targets.moments.push(image(*format, name)?);
                }
            }
            if self.blend == ParticleBlend::Slices {
                for (format, name) in SLICE_FORMATS.iter().zip(["particle slice range", "particle slice 0", "particle slice 1", "particle slice 2", "particle slice 3"]) {
                    targets.moments.push(image(*format, name)?);
                }
            }
            if self.blend.lists() {
                let pixels = half.width as u64 * half.height as u64;
                let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
                targets.reference.push(OwnedBuffer::new(gpu, BufferDesc { size: pixels * 4, usage }, MemoryLocation::GpuOnly, "particle reference counts")?);
                targets.reference.push(OwnedBuffer::new(gpu, BufferDesc { size: pixels * REFERENCE_LAYERS * 16, usage }, MemoryLocation::GpuOnly, "particle reference fragments")?);
            }
            Ok(())
        })();
        if let Err(err) = made {
            targets.destroy(gpu);
            return Err(err);
        }
        let info = |view| [vk::DescriptorImageInfo { sampler: self.depth_sampler, image_view: view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
        let (color_info, depth_info) = (info(targets.color.view), info(targets.depth.view));
        let moment_infos: Vec<_> = targets.moments.iter().map(|m| info(m.view)).collect();
        let buffer_infos: Vec<_> = targets.reference.iter().map(|b| [vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: vk::WHOLE_SIZE }]).collect();
        let image = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
        let mut writes = vec![
            vk::WriteDescriptorSet::default().dst_set(self.half_sets[slot]).dst_binding(0).descriptor_type(image).image_info(&depth_info),
            vk::WriteDescriptorSet::default().dst_set(self.composite_sets[slot]).dst_binding(0).descriptor_type(image).image_info(&color_info),
            vk::WriteDescriptorSet::default().dst_set(self.composite_sets[slot]).dst_binding(1).descriptor_type(image).image_info(&depth_info),
        ];
        for (i, info) in moment_infos.iter().enumerate() {
            writes.push(vk::WriteDescriptorSet::default().dst_set(self.half_sets[slot]).dst_binding(2 + i as u32).descriptor_type(image).image_info(info));
        }
        for (i, info) in buffer_infos.iter().enumerate() {
            writes.push(vk::WriteDescriptorSet::default().dst_set(self.half_sets[slot]).dst_binding(7 + i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info));
        }
        unsafe { gpu.device.update_descriptor_sets(&writes, &[]) };
        self.half[slot] = Some(targets);
        Ok(())
    }

    fn sort_groups(capacity: u32) -> u32 {
        capacity.div_ceil(SORT_BLOCKS * SORT_WORKGROUP)
    }

    fn indirect_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Whether the frame that last used `slot` had live particles.
    pub fn active(&self, slot: usize) -> bool {
        self.active[slot]
    }

    pub fn state(&self) -> BufferBinding {
        BufferBinding { buffer: self.state.buffer, offset: 0, size: self.state.size, identity: IDENTITY_STATE }
    }

    pub fn state_desc(&self) -> BufferDesc {
        BufferDesc { size: self.state.size, usage: vk::BufferUsageFlags::STORAGE_BUFFER }
    }

    pub fn keys(&self) -> BufferBinding {
        BufferBinding { buffer: self.keys.buffer, offset: 0, size: self.keys.size, identity: IDENTITY_KEYS }
    }

    pub fn keys_desc(&self) -> BufferDesc {
        BufferDesc { size: self.keys.size, usage: vk::BufferUsageFlags::STORAGE_BUFFER }
    }

    pub fn indirect(&self) -> BufferBinding {
        BufferBinding { buffer: self.indirect.buffer, offset: 0, size: self.indirect.size, identity: IDENTITY_INDIRECT }
    }

    pub fn indirect_desc(&self) -> BufferDesc {
        BufferDesc { size: self.indirect.size, usage: Self::indirect_usage() }
    }

    /// Take this frame's bursts from the feed and write `slot`'s uniforms
    /// and spawn list (the slot's previous submission has completed); with
    /// a reduced resolution, make the slot's targets when it draws.
    pub fn write_frame(&mut self, gpu: &GpuDevice, slot: usize, frame: &ParticleFrame) -> Result<()> {
        let dt = self.last_time.map_or(0.0, |last| (frame.time - last).clamp(0.0, 0.1));
        self.last_time = Some(frame.time);
        self.sim_time += dt;
        let bursts = self.feed.take(MAX_BURSTS);
        let mut records = Vec::with_capacity(bursts.len());
        let mut total = 0u32;
        for b in &bursts {
            let Some(def) = self.effects.get(b.effect as usize) else { continue };
            let count = b.count.min(self.capacity - total);
            if count == 0 {
                continue;
            }
            records.push(SpawnRecord { pos: [b.pos[0], b.pos[1], b.pos[2], b.scale], info: [b.effect | (b.ground as u32) << 16, count, total, b.seed] });
            total += count;
            self.alive_until = self.alive_until.max(self.sim_time + def.max_life());
        }
        let head = self.head;
        self.head = (self.head + total) % self.capacity;
        if let Some(bytes) = self.spawns[slot].mapped() {
            let raw = pipeline::slice_bytes(&records);
            bytes[..raw.len()].copy_from_slice(raw);
        }
        self.spawned[slot] = (records.len() as u32, total);
        // Simulate while anything spawned may live (and one frame after,
        // so the last dead particles leave the list).
        self.active[slot] = total > 0 || self.sim_time <= self.alive_until + 0.2;
        let i = &self.info;
        let mut effects = [[[0.0; 4]; 5]; MAX_EFFECTS];
        for (slot, def) in effects.iter_mut().zip(&self.effects) {
            *slot = def.words();
        }
        let [right, up, forward] = frame.axes;
        let ubo = FrameUbo {
            view_proj: frame.view_proj.to_cols_array(),
            cam_pos: frame.cam_pos.extend(frame.near).to_array(),
            right: right.extend(0.0).to_array(),
            up: up.extend(0.0).to_array(),
            forward: forward.extend(0.0).to_array(),
            frustum: frustum_planes(frame.view_proj),
            step: [dt, self.sim_time, 1.0 / frame.extent.width.max(1) as f32, 1.0 / frame.extent.height.max(1) as f32],
            pool: [self.capacity, records.len() as u32, total, head],
            flags: [(frame.shadows && self.shadows) as u32, (self.blend == ParticleBlend::Moments) as u32, frame.unit_shadows.is_some() as u32, 0],
            map: [i.cell_size, 1.0 / (i.cells.0 + 1) as f32, 1.0 / (i.cells.1 + 1) as f32, 0.0],
            height: [i.height_scale, i.height_offset, i.water_level.unwrap_or(0.0), i.water_level.is_some() as u32 as f32],
            sun: crate::terrain::sun::direction().extend(0.0).to_array(),
            fog: frame.fog,
            shadow: crate::shadow::Cascades::uniforms(frame.unit_shadows.as_ref()),
            effects,
        };
        self.casting[slot] = frame.unit_shadows.is_some();
        if let Some(bytes) = self.ubos[slot].mapped() {
            bytes[..size_of::<FrameUbo>()].copy_from_slice(pipeline::push_bytes(&ubo));
        }
        if self.downscale > 1 && self.active[slot] {
            self.ensure_half(gpu, slot, frame.extent).context("particle targets")?;
        }
        Ok(())
    }

    /// Spawn, move, cull, key and sort. Records inside a compute pass that
    /// writes [`Self::state`], [`Self::keys`] and [`Self::indirect`]; the
    /// barriers between its dispatches are its own.
    pub fn record_sim(&self, ctx: &PassContext<'_>) {
        if !self.active[ctx.slot] {
            return;
        }
        let (device, cmd) = (ctx.device, ctx.cmd);
        let (_, total) = self.spawned[ctx.slot];
        let barrier = || {
            let memory = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)];
            unsafe { device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&memory)) };
        };
        unsafe {
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.sets[ctx.slot]], &[]);
            // Spawn (its first invocation also zeroes the instance count).
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.spawn);
            device.cmd_dispatch(cmd, total.div_ceil(64).max(1), 1, 1);
            barrier();
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.sim);
            device.cmd_dispatch(cmd, self.capacity.div_ceil(64), 1, 1);
            barrier();
            if self.blend != ParticleBlend::Sorted {
                // Order independent: no sort.
                return;
            }
            // LSD radix sort of the live keys, 8 bits a pass; four passes
            // end back in `keys`.
            let groups = Self::sort_groups(self.capacity);
            for pass in 0..4u32 {
                let push = [pass * 8, pass & 1];
                device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, pipeline::push_bytes(&push));
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.histogram);
                device.cmd_dispatch(cmd, groups, 1, 1);
                barrier();
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.scatter);
                device.cmd_dispatch(cmd, groups, 1, 1);
                barrier();
            }
        }
    }

    /// Every live particle that is not emissive into each cascade of the
    /// coloured shadows (nothing when idle or when the frame draws none:
    /// the target stays cleared), from the casters the simulation listed.
    /// Records inside a graphics pass that reads [`Self::state`] and
    /// [`Self::keys`] in the vertex stage, [`Self::indirect`] as an
    /// indirect buffer, and has `shadows.color` as its one colour
    /// attachment, cleared to 0.
    pub fn record_shadows(&self, ctx: &PassContext<'_>) {
        if !self.active[ctx.slot] || !self.casting[ctx.slot] {
            return;
        }
        let (device, cmd) = (ctx.device, ctx.cmd);
        let side = ae_render_abi::module::PARTICLE_SHADOW_HEIGHT;
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.caster);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.caster_layout, 0, &[self.sets[ctx.slot]], &[]);
            for k in 0..crate::shadow::CASCADES as u32 {
                let viewport = vk::Viewport { x: (k * side) as f32, y: 0.0, width: side as f32, height: side as f32, min_depth: 0.0, max_depth: 1.0 };
                device.cmd_set_viewport(cmd, 0, &[viewport]);
                device.cmd_set_scissor(cmd, 0, &[vk::Rect2D { offset: vk::Offset2D { x: (k * side) as i32, y: 0 }, extent: vk::Extent2D { width: side, height: side } }]);
                device.cmd_push_constants(cmd, self.caster_layout, vk::ShaderStageFlags::VERTEX, 0, &k.to_le_bytes());
                device.cmd_draw_indirect(cmd, self.indirect.buffer, CASTER_DRAW, 1, 0);
            }
        }
    }

    /// The live particles over the scene. Records inside a graphics pass
    /// with one colour attachment that reads [`Self::state`] and
    /// [`Self::keys`] in the vertex stage, [`Self::indirect`] as an
    /// indirect buffer and samples `depth` in the fragment stage; `color`
    /// is that colour attachment (in `COLOR_ATTACHMENT_OPTIMAL`).
    ///
    /// At half resolution the work happens inside the pass, outside the
    /// graph: it ends the pass's rendering scope, renders the depth
    /// downsample and the particles into the slot's own targets (with its
    /// own barriers), then begins a scope on `color` again (load, store)
    /// for the composite, which the graph ends. The pass's contract stays
    /// the same at either resolution.
    pub fn record_draw(&self, ctx: &PassContext<'_>, color: ImageBinding, depth: ImageBinding) -> Result<()> {
        if !self.active[ctx.slot] {
            return Ok(());
        }
        let (device, cmd, slot) = (ctx.device, ctx.cmd, ctx.slot);
        let depth_set = self.depth_sets[slot];
        let image = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
        let full = [vk::DescriptorImageInfo { sampler: self.depth_sampler, image_view: depth.view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
        // The slot's previous frame has completed, and nothing in this one
        // has bound these sets yet: they are idle.
        let mut writes = vec![vk::WriteDescriptorSet::default().dst_set(depth_set).dst_binding(0).descriptor_type(image).image_info(&full)];
        if self.downscale > 1 {
            writes.push(vk::WriteDescriptorSet::default().dst_set(self.composite_sets[slot]).dst_binding(2).descriptor_type(image).image_info(&full));
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        if self.downscale == 1 {
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.draw);
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[self.sets[slot], depth_set], &[]);
                device.cmd_draw_indirect(cmd, self.indirect.buffer, 0, 1, 0);
            }
            return Ok(());
        }
        let half = self.half[slot].as_ref().context("particle targets not made (write_frame)")?;
        let color_range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let to_attachment = |image: vk::Image| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(image)
                .subresource_range(color_range)
        };
        let to_sampled = |image: vk::Image| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image)
                .subresource_range(color_range)
        };
        let begin = |views: &[vk::ImageView], extent: vk::Extent2D, load: vk::AttachmentLoadOp| {
            let attachments: Vec<_> = views
                .iter()
                .map(|&view| {
                    vk::RenderingAttachmentInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(load)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })
                })
                .collect();
            let area = vk::Rect2D { offset: vk::Offset2D::default(), extent };
            unsafe {
                device.cmd_begin_rendering(cmd, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&attachments));
                device.cmd_set_viewport(cmd, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: extent.width as f32, height: extent.height as f32, min_depth: 0.0, max_depth: 1.0 }]);
                device.cmd_set_scissor(cmd, 0, &[area]);
            }
        };
        let barrier = |barriers: &[vk::ImageMemoryBarrier2]| unsafe { device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(barriers)) };
        let buffer_barrier = |buffer: vk::Buffer, src: (vk::PipelineStageFlags2, vk::AccessFlags2), dst: (vk::PipelineStageFlags2, vk::AccessFlags2)| {
            let barriers = [vk::BufferMemoryBarrier2::default().src_stage_mask(src.0).src_access_mask(src.1).dst_stage_mask(dst.0).dst_access_mask(dst.1).buffer(buffer).offset(0).size(vk::WHOLE_SIZE)];
            unsafe { device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().buffer_memory_barriers(&barriers)) };
        };
        let fragment_read = (vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ);
        let half_sets = [self.half_sets[slot]];
        let draw_particles = |pipeline: vk::Pipeline| unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 1, &half_sets, &[]);
            device.cmd_draw_indirect(cmd, self.indirect.buffer, 0, 1, 0);
        };
        let fullscreen = |pipeline: vk::Pipeline| unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 1, &half_sets, &[]);
            device.cmd_draw(cmd, 3, 1, 0, 0);
        };
        unsafe {
            device.cmd_end_rendering(cmd);
            if self.blend.lists() {
                device.cmd_fill_buffer(cmd, half.reference[0].buffer, 0, vk::WHOLE_SIZE, 0);
            }
            // The scene depth's farthest per 2x2 block.
            barrier(&[to_attachment(half.depth.image), to_attachment(half.color.image)]);
            begin(&[half.depth.view], half.extent, vk::AttachmentLoadOp::DONT_CARE);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.depth_down);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[self.sets[slot], depth_set], &[]);
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_rendering(cmd);
            match self.blend {
                ParticleBlend::Sorted => {
                    // The particles, softened against it, into a cleared target.
                    barrier(&[to_sampled(half.depth.image)]);
                    begin(&[half.color.view], half.extent, vk::AttachmentLoadOp::CLEAR);
                    draw_particles(self.draw);
                    device.cmd_end_rendering(cmd);
                }
                ParticleBlend::Moments => {
                    // The depth range the simulation found, for the warp.
                    buffer_barrier(self.indirect.buffer, (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE), fragment_read);
                    let m = &half.moments;
                    barrier(&[to_sampled(half.depth.image), to_attachment(m[0].image), to_attachment(m[1].image)]);
                    begin(&[m[0].view, m[1].view], half.extent, vk::AttachmentLoadOp::CLEAR);
                    draw_particles(self.oit[0]);
                    device.cmd_end_rendering(cmd);
                    barrier(&[to_sampled(m[0].image), to_sampled(m[1].image), to_attachment(m[2].image), to_attachment(m[3].image)]);
                    begin(&[m[2].view, m[3].view], half.extent, vk::AttachmentLoadOp::CLEAR);
                    draw_particles(self.oit[1]);
                    device.cmd_end_rendering(cmd);
                    barrier(&[to_sampled(m[2].image), to_sampled(m[3].image)]);
                    begin(&[half.color.view], half.extent, vk::AttachmentLoadOp::DONT_CARE);
                    fullscreen(self.oit[2]);
                    device.cmd_end_rendering(cmd);
                }
                ParticleBlend::Slices => {
                    let m = &half.moments;
                    barrier(&[to_sampled(half.depth.image), to_attachment(m[0].image)]);
                    begin(&[m[0].view], half.extent, vk::AttachmentLoadOp::CLEAR);
                    draw_particles(self.oit[0]);
                    device.cmd_end_rendering(cmd);
                    let slices: Vec<_> = m[1..].iter().map(|s| to_attachment(s.image)).chain([to_sampled(m[0].image)]).collect();
                    barrier(&slices);
                    let views: Vec<_> = m[1..].iter().map(|s| s.view).collect();
                    begin(&views, half.extent, vk::AttachmentLoadOp::CLEAR);
                    draw_particles(self.oit[1]);
                    device.cmd_end_rendering(cmd);
                    let sampled: Vec<_> = m[1..].iter().map(|s| to_sampled(s.image)).collect();
                    barrier(&sampled);
                    begin(&[half.color.view], half.extent, vk::AttachmentLoadOp::DONT_CARE);
                    fullscreen(self.oit[2]);
                    device.cmd_end_rendering(cmd);
                }
                ParticleBlend::Reference | ParticleBlend::Volume => {
                    let (counts, fragments) = (half.reference[0].buffer, half.reference[1].buffer);
                    let stores = (vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE);
                    buffer_barrier(counts, (vk::PipelineStageFlags2::CLEAR, vk::AccessFlags2::TRANSFER_WRITE), stores);
                    barrier(&[to_sampled(half.depth.image)]);
                    begin(&[], half.extent, vk::AttachmentLoadOp::DONT_CARE);
                    draw_particles(self.oit[0]);
                    device.cmd_end_rendering(cmd);
                    buffer_barrier(counts, (vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE), fragment_read);
                    buffer_barrier(fragments, (vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE), fragment_read);
                    begin(&[half.color.view], half.extent, vk::AttachmentLoadOp::DONT_CARE);
                    fullscreen(self.oit[1]);
                    device.cmd_end_rendering(cmd);
                }
            }
            // Over the scene.
            // (and the scene's colour: written before the scope was ended).
            let scene = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(color.image)
                .subresource_range(color.range);
            barrier(&[to_sampled(half.color.image), scene]);
            begin(&[color.view], color.extent, vk::AttachmentLoadOp::LOAD);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.composite);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.composite_pipeline_layout, 0, &[self.sets[slot], self.composite_sets[slot]], &[]);
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
        Ok(())
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        let device = &gpu.device;
        unsafe {
            for &p in [self.spawn, self.sim, self.histogram, self.scatter, self.draw, self.depth_down, self.composite, self.caster].iter().chain(&self.oit) {
                device.destroy_pipeline(p, None);
            }
            device.destroy_sampler(self.depth_sampler, None);
            device.destroy_sampler(self.puff_sampler, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_pipeline_layout(self.composite_pipeline_layout, None);
            device.destroy_pipeline_layout(self.caster_layout, None);
            device.destroy_descriptor_pool(self.desc_pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_descriptor_set_layout(self.depth_layout, None);
            device.destroy_descriptor_set_layout(self.composite_layout, None);
        }
        for half in self.half.iter_mut().filter_map(Option::take) {
            half.destroy(gpu);
        }
        self.puff.destroy(gpu);
        for b in self.ubos.iter_mut().chain(self.spawns.iter_mut()) {
            b.destroy(gpu);
        }
        for b in [&mut self.state, &mut self.keys, &mut self.keys_tmp, &mut self.histograms, &mut self.indirect] {
            b.destroy(gpu);
        }
    }
}

/// The puff texture and its mips: a ball fading to its edge (`(1 - r²)²`)
/// whose density four octaves of value noise break into billows; mip `i`
/// halves mip `i - 1` (2x2 box).
fn puff_mips() -> Vec<Vec<u8>> {
    fn hash(x: i32, y: i32, octave: i32) -> f32 {
        let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ (octave as u32).wrapping_mul(0xcb1a_b31f);
        h ^= h >> 13;
        h = h.wrapping_mul(0x5bd1_e995);
        h ^= h >> 15;
        (h & 0xffff) as f32 / 65535.0
    }
    fn value_noise(x: f32, y: f32, octave: i32) -> f32 {
        let (ix, iy) = (x.floor() as i32, y.floor() as i32);
        let (fx, fy) = (x - x.floor(), y - y.floor());
        let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
        let top = hash(ix, iy, octave) + (hash(ix + 1, iy, octave) - hash(ix, iy, octave)) * sx;
        let bottom = hash(ix, iy + 1, octave) + (hash(ix + 1, iy + 1, octave) - hash(ix, iy + 1, octave)) * sx;
        top + (bottom - top) * sy
    }
    let mut level: Vec<f32> = (0..PUFF * PUFF)
        .map(|i| {
            let (u, v) = ((i % PUFF) as f32 + 0.5, (i / PUFF) as f32 + 0.5);
            let (x, y) = (u / PUFF as f32 * 2.0 - 1.0, v / PUFF as f32 * 2.0 - 1.0);
            let r2 = x * x + y * y;
            if r2 >= 1.0 {
                return 0.0;
            }
            let ball = (1.0 - r2) * (1.0 - r2);
            let mut noise = 0.0;
            let mut amplitude = 0.5;
            for octave in 0..4 {
                let f = 4.0 * (1 << octave) as f32;
                noise += amplitude * value_noise(u / PUFF as f32 * f, v / PUFF as f32 * f, octave);
                amplitude *= 0.5;
            }
            (ball * (0.35 + 1.3 * noise) - 0.04).clamp(0.0, 1.0)
        })
        .collect();
    let mut mips = Vec::new();
    let mut side = PUFF;
    loop {
        mips.push(level.iter().map(|&d| (d * 255.0).round() as u8).collect());
        if side == 1 {
            break;
        }
        let half = side / 2;
        level = (0..half * half)
            .map(|i| {
                let (x, y) = (i % half * 2, i / half * 2);
                0.25 * (level[y * side + x] + level[y * side + x + 1] + level[(y + 1) * side + x] + level[(y + 1) * side + x + 1])
            })
            .collect();
        side = half;
    }
    mips
}
