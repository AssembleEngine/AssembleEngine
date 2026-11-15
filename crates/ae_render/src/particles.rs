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

use crate::device::GpuDevice;
use crate::executor::{BufferBinding, ImageBinding, PassContext};
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::pipeline::{self, Blend, GraphicsDesc, spirv};
use crate::terrain::renderer::{TerrainInfo, TerrainRenderer, frustum_planes};
use crate::upload;
use ae_framegraph::BufferDesc;
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
    /// Adds light (alpha 0) instead of covering.
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
}

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
    /// flags (1: hills' shadows), 0, 0, 0.
    flags: [u32; 4],
    map: [f32; 4],
    height: [f32; 4],
    sun: [f32; 4],
    fog: [f32; 4],
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
}

impl ParticleRenderer {
    pub fn new(gpu: &GpuDevice, terrain: &TerrainRenderer, setup: &ParticlesSetup, color_format: vk::Format, frames_in_flight: usize) -> Result<Self> {
        ensure!((1..=MAX_CAPACITY).contains(&setup.capacity), "particle capacity {} (1..={MAX_CAPACITY})", setup.capacity);
        ensure!(setup.effects.len() <= MAX_EFFECTS, "{} particle effects (at most {MAX_EFFECTS})", setup.effects.len());
        let device = &gpu.device;
        let capacity = setup.capacity;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        // Zeroed: every slot starts dead (life 0).
        let state = upload::buffer(gpu, &vec![0u8; (capacity as u64 * STATE_BYTES) as usize], storage, "particle state")?;
        let keys = OwnedBuffer::new(gpu, BufferDesc { size: capacity as u64 * 4, usage: storage }, MemoryLocation::GpuOnly, "particle keys")?;
        let keys_tmp = OwnedBuffer::new(gpu, BufferDesc { size: capacity as u64 * 4, usage: storage }, MemoryLocation::GpuOnly, "particle keys (sort)")?;
        let groups = Self::sort_groups(capacity);
        let histograms = OwnedBuffer::new(gpu, BufferDesc { size: groups as u64 * SORT_BINS * 4, usage: storage }, MemoryLocation::GpuOnly, "particle sort histograms")?;
        let indirect = upload::buffer(gpu, pipeline::push_bytes(&[VERTICES, 0, 0, 0]), Self::indirect_usage(), "particle indirect")?;
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
        let (set_layout, depth_layout, pipeline_layout, desc_pool, sets, depth_sets) = unsafe {
            let ssbo = vk::DescriptorType::STORAGE_BUFFER;
            let image = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
            let set_layout = pipeline::set_layout(
                device,
                &[(vk::DescriptorType::UNIFORM_BUFFER, all), (ssbo, all), (ssbo, all), (ssbo, all), (ssbo, all), (ssbo, all), (ssbo, all), (image, all), (image, all)],
            )?;
            let depth_layout = pipeline::set_layout(device, &[(image, vk::ShaderStageFlags::FRAGMENT), (image, vk::ShaderStageFlags::FRAGMENT)])?;
            // The sort's push constant: shift, direction.
            let pipeline_layout = pipeline::pipeline_layout(device, &[set_layout, depth_layout], 8, vk::ShaderStageFlags::COMPUTE)?;
            let n = frames_in_flight as u32;
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: n },
                vk::DescriptorPoolSize { ty: ssbo, descriptor_count: 6 * n },
                vk::DescriptorPoolSize { ty: image, descriptor_count: 5 * n },
            ];
            let desc_pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(2 * n).pool_sizes(&sizes), None)?;
            let layouts = vec![set_layout; frames_in_flight];
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&layouts))?;
            let layouts = vec![depth_layout; frames_in_flight];
            let depth_sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&layouts))?;
            let shape = [vk::DescriptorImageInfo { sampler: puff_sampler, image_view: puff.view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
            let writes: Vec<_> = depth_sets.iter().map(|&set| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(image).image_info(&shape)).collect();
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
                device.update_descriptor_sets(&writes, &[]);
            }
            (set_layout, depth_layout, pipeline_layout, desc_pool, sets, depth_sets)
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
                color_formats: &[color_format],
                // Depth is sampled: the shader tests (and fades) by itself.
                depth: None,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                blend: Blend::Premultiplied,
            },
        )
        .context("particle pipeline")?;
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
        })
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
    /// and spawn list (the slot's previous submission has completed).
    pub fn write_frame(&mut self, slot: usize, frame: &ParticleFrame) {
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
            flags: [(frame.shadows && self.shadows) as u32, 0, 0, 0],
            map: [i.cell_size, 1.0 / (i.cells.0 + 1) as f32, 1.0 / (i.cells.1 + 1) as f32, 0.0],
            height: [i.height_scale, i.height_offset, i.water_level.unwrap_or(0.0), i.water_level.is_some() as u32 as f32],
            sun: crate::terrain::sun::direction().extend(0.0).to_array(),
            fog: frame.fog,
            effects,
        };
        if let Some(bytes) = self.ubos[slot].mapped() {
            bytes[..size_of::<FrameUbo>()].copy_from_slice(pipeline::push_bytes(&ubo));
        }
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

    /// The live particles over the scene. Records inside a graphics pass
    /// with one colour attachment that reads [`Self::state`] and
    /// [`Self::keys`] in the vertex stage, [`Self::indirect`] as an
    /// indirect buffer and samples `depth` in the fragment stage.
    pub fn record_draw(&self, ctx: &PassContext<'_>, depth: ImageBinding) {
        if !self.active[ctx.slot] {
            return;
        }
        let (device, cmd) = (ctx.device, ctx.cmd);
        let depth_set = self.depth_sets[ctx.slot];
        let image = [vk::DescriptorImageInfo { sampler: self.depth_sampler, image_view: depth.view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
        unsafe {
            // The slot's previous frame has completed, and nothing in this
            // one has bound the set yet: it is idle.
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(depth_set).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&image)], &[]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.draw);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[self.sets[ctx.slot], depth_set], &[]);
            device.cmd_draw_indirect(cmd, self.indirect.buffer, 0, 1, 0);
        }
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        let device = &gpu.device;
        unsafe {
            for p in [self.spawn, self.sim, self.histogram, self.scatter, self.draw] {
                device.destroy_pipeline(p, None);
            }
            device.destroy_sampler(self.depth_sampler, None);
            device.destroy_sampler(self.puff_sampler, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.desc_pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_descriptor_set_layout(self.depth_layout, None);
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
