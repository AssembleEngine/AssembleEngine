//! The global descriptor set (M15 step A): one fixed-capacity array per
//! resource class, index 0 (or 0..3 for sampled images) a fallback, and a
//! handle registry that issues indices in creation order.
//!
//! The set is written only in [`Bindless::begin_frame`], after the frame
//! slot's fence has signalled, so no update-after-bind feature is needed.
//! A handle registered later in the frame resolves to the fallback until
//! its slot has flushed (design decision 2).
//!
//! Only the index reaches the GPU. The generation stays on the CPU and is
//! checked by [`Bindless::index_2d`] and its siblings, so a stale handle
//! draws the fallback and logs once instead of reading another resource's
//! descriptor.
//!
//! Handles are owned: a module registers the ones it makes inside a group
//! ([`Bindless::group`]) and gives them back when it is destroyed
//! ([`Bindless::retire_group`]). Samplers are deduplicated by their create
//! info and reference-counted, so passes that want the same filtering share
//! one index and one object.

use crate::device::GpuDevice;
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::upload;
use ae_framegraph::{BakedGraph, FrameGraph, ImageDesc, ResourceDesc, ResourceId};
use anyhow::{Result, bail};
use ash::vk;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub const SAMPLED_IMAGE_CAPACITY: u32 = 4096;
pub const SAMPLER_CAPACITY: u32 = 64;
pub const STORAGE_IMAGE_CAPACITY: u32 = 256;
pub const STORAGE_BUFFER_CAPACITY: u32 = 1024;

pub const BINDING_SAMPLED_IMAGE: u32 = 0;
pub const BINDING_SAMPLER: u32 = 1;
pub const BINDING_STORAGE_IMAGE: u32 = 2;
pub const BINDING_STORAGE_BUFFER: u32 = 3;

/// The stages of the shared pipeline layout's push range ([`Push`], 128
/// bytes). Every `cmd_push_constants` on [`Bindless::pipeline_layout`] passes
/// all of them.
pub const PUSH_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(
    vk::ShaderStageFlags::VERTEX.as_raw()
        | vk::ShaderStageFlags::FRAGMENT.as_raw()
        | vk::ShaderStageFlags::COMPUTE.as_raw(),
);

/// The view type of a sampled handle, which selects its fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampledKind {
    Image2D,
    Image2DArray,
    Uint2D,
}

impl SampledKind {
    fn fallback(self) -> u32 {
        match self {
            Self::Image2D => 0,
            Self::Image2DArray => 1,
            Self::Uint2D => 2,
        }
    }

    fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Image2D),
            1 => Some(Self::Image2DArray),
            2 => Some(Self::Uint2D),
            _ => None,
        }
    }
}

/// A sampled-image handle. The index is the only part a shader sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageHandle {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sampled2D(pub ImageHandle);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sampled2DArray(pub ImageHandle);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampledU2D(pub ImageHandle);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StorageImageHandle {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplerHandle {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferHandle {
    pub index: u32,
    pub generation: u32,
}

/// A set of handles registered together and given back together: a module
/// opens one around the handles it makes ([`Bindless::group`]) and retires
/// it in its teardown ([`Bindless::retire_group`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupId(u32);

/// The create info that makes two samplers the same. `p_next` is not part
/// of it: every sampler the engine makes is a plain
/// [`vk::SamplerCreateInfo`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SamplerKey {
    flags: vk::SamplerCreateFlags,
    mag_filter: vk::Filter,
    min_filter: vk::Filter,
    mipmap_mode: vk::SamplerMipmapMode,
    address_u: vk::SamplerAddressMode,
    address_v: vk::SamplerAddressMode,
    address_w: vk::SamplerAddressMode,
    mip_lod_bias: u32,
    anisotropy: vk::Bool32,
    max_anisotropy: u32,
    compare: vk::Bool32,
    compare_op: vk::CompareOp,
    min_lod: u32,
    max_lod: u32,
    border_color: vk::BorderColor,
    unnormalized: vk::Bool32,
}

impl SamplerKey {
    fn new(info: &vk::SamplerCreateInfo<'_>) -> Self {
        Self {
            flags: info.flags,
            mag_filter: info.mag_filter,
            min_filter: info.min_filter,
            mipmap_mode: info.mipmap_mode,
            address_u: info.address_mode_u,
            address_v: info.address_mode_v,
            address_w: info.address_mode_w,
            mip_lod_bias: info.mip_lod_bias.to_bits(),
            anisotropy: info.anisotropy_enable,
            max_anisotropy: info.max_anisotropy.to_bits(),
            compare: info.compare_enable,
            compare_op: info.compare_op,
            min_lod: info.min_lod.to_bits(),
            max_lod: info.max_lod.to_bits(),
            border_color: info.border_color,
            unnormalized: info.unnormalized_coordinates,
        }
    }
}

/// One created sampler: the settings that deduplicate it, the object (owned
/// by the registry) and how many handles hold it.
#[derive(Clone, Copy, Debug)]
struct SamplerEntry {
    key: SamplerKey,
    sampler: vk::Sampler,
    refs: u32,
}

/// Live counts, binding order, for `ExecutorStats`.
pub const CLASS_COUNT: usize = 4;

/// The 128-byte push block `bindless.glsl` declares: eight of the pass's own
/// words, 16 sampled-image indices and 8 sampler indices.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Push {
    pub words: [u32; 4],
    pub words2: [u32; 4],
    pub images: [[u32; 4]; 4],
    pub samplers: [[u32; 4]; 2],
}

const _: () = assert!(size_of::<Push>() == 128);

impl Push {
    pub fn image(&mut self, at: usize, index: u32) {
        self.images[at / 4][at % 4] = index;
    }

    pub fn sampler(&mut self, at: usize, index: u32) {
        self.samplers[at / 4][at % 4] = index;
    }

    /// The pass's handle table (`AE_TABLE`, push word 8).
    pub fn table(&mut self, index: u32) {
        self.words2[3] = index;
    }

    pub fn sampler_at(&self, at: usize) -> u32 {
        self.samplers[at / 4][at % 4]
    }
}

/// The limits a device must meet for the fixed capacities. Without
/// update-after-bind these are the ordinary per-stage and per-set limits.
pub fn check_limits(limits: &vk::PhysicalDeviceLimits) -> Result<(), String> {
    let checks = [
        (
            "sampled images",
            SAMPLED_IMAGE_CAPACITY,
            limits.max_per_stage_descriptor_sampled_images,
            limits.max_descriptor_set_sampled_images,
        ),
        (
            "samplers",
            SAMPLER_CAPACITY,
            limits.max_per_stage_descriptor_samplers,
            limits.max_descriptor_set_samplers,
        ),
        (
            "storage images",
            STORAGE_IMAGE_CAPACITY,
            limits.max_per_stage_descriptor_storage_images,
            limits.max_descriptor_set_storage_images,
        ),
        (
            "storage buffers",
            STORAGE_BUFFER_CAPACITY,
            limits.max_per_stage_descriptor_storage_buffers,
            limits.max_descriptor_set_storage_buffers,
        ),
    ];
    for (class, want, per_stage, per_set) in checks {
        if per_stage < want || per_set < want {
            return Err(format!(
                "the bindless {class} capacity {want} exceeds the device's ordinary descriptor \
                 limits (maxPerStageDescriptor {per_stage}, maxDescriptorSet {per_set})"
            ));
        }
    }
    let total = SAMPLED_IMAGE_CAPACITY + SAMPLER_CAPACITY + STORAGE_IMAGE_CAPACITY + STORAGE_BUFFER_CAPACITY;
    if limits.max_per_stage_resources < total {
        return Err(format!(
            "the bindless set's {total} descriptors exceed the device's maxPerStageResources {}",
            limits.max_per_stage_resources
        ));
    }
    Ok(())
}

/// The indices the executor's registry would issue for a graph's first
/// bake, in declaration order: `[sampled, storage image, storage buffer]`
/// per resource name. The `.dot` goldens use it; it must agree with
/// `Executor::sync_graph_handles`.
pub fn graph_indices<R>(graph: &FrameGraph<R>, baked: &BakedGraph) -> BTreeMap<String, [Option<u32>; 3]> {
    let mut next = [3u32, 1, 1];
    let mut out = BTreeMap::new();
    for (index, resource) in baked.resources.iter().enumerate() {
        if !resource.reachable || resource.physical.is_none() {
            continue;
        }
        let name = graph.resource_name(ResourceId(index as u32)).to_string();
        let mut indices = [None; 3];
        match baked.physical[resource.physical.unwrap() as usize].desc {
            ResourceDesc::Image(image) => {
                if image.usage.contains(vk::ImageUsageFlags::SAMPLED) {
                    indices[0] = Some(next[0]);
                    next[0] += 1;
                }
                if image.usage.contains(vk::ImageUsageFlags::STORAGE) {
                    indices[1] = Some(next[1]);
                    next[1] += 1;
                }
            }
            ResourceDesc::Buffer(buffer) => {
                if buffer.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
                    indices[2] = Some(next[2]);
                    next[2] += 1;
                }
            }
        }
        out.insert(name, indices);
    }
    out
}

/// One slab index's CPU state.
#[derive(Clone, Copy, Debug)]
struct Entry {
    generation: u32,
    live: bool,
    kind: u32,
}

/// A generation-checked index slab. `reserved` indices at the start are the
/// fallbacks and are never freed.
#[derive(Debug)]
struct Slab {
    class: &'static str,
    capacity: u32,
    next: u32,
    free: VecDeque<u32>,
    entries: Vec<Entry>,
    live: u32,
}

impl Slab {
    fn new(class: &'static str, reserved: u32, capacity: u32, kind: u32) -> Self {
        let entries = (0..reserved)
            .map(|_| Entry {
                generation: 0,
                live: true,
                kind,
            })
            .collect();
        Self {
            class,
            capacity,
            next: reserved,
            free: VecDeque::new(),
            entries,
            live: reserved,
        }
    }

    fn alloc(&mut self, kind: u32) -> Result<(u32, u32)> {
        let index = if let Some(index) = self.free.pop_front() {
            index
        } else if self.next < self.capacity {
            let index = self.next;
            self.next += 1;
            self.entries.push(Entry {
                generation: 0,
                live: false,
                kind,
            });
            index
        } else {
            bail!(
                "the bindless {} capacity {} is exhausted; the registry has a leak or the scene \
                 is past the design limit",
                self.class,
                self.capacity
            );
        };
        let entry = &mut self.entries[index as usize];
        entry.generation = entry.generation.wrapping_add(1).max(1);
        entry.live = true;
        entry.kind = kind;
        self.live += 1;
        Ok((index, entry.generation))
    }

    /// Stop resolving `index` (it is retiring). False if it was not live,
    /// so a handle cannot be retired, and later freed, twice.
    fn retire(&mut self, index: u32) -> bool {
        match self.entries.get_mut(index as usize) {
            Some(entry) if entry.live => {
                entry.live = false;
                self.live -= 1;
                true
            }
            _ => false,
        }
    }

    /// Return a retired index to the free list once every slot has
    /// flushed its fallback.
    fn free(&mut self, index: u32) {
        debug_assert!(!self.entries[index as usize].live, "freeing a live index");
        debug_assert!(!self.free.contains(&index), "index freed twice");
        self.free.push_back(index);
    }

    fn resolve(&self, index: u32, generation: u32) -> bool {
        self.entries
            .get(index as usize)
            .is_some_and(|entry| entry.live && entry.generation == generation)
    }

    fn is_live(&self, index: u32) -> bool {
        self.entries.get(index as usize).is_some_and(|entry| entry.live)
    }

    fn generation(&self, index: u32) -> u32 {
        self.entries[index as usize].generation
    }

    /// Make a retiring index live again for a new owner (a sampler whose
    /// settings are already in every slot's set). Its generation advances,
    /// so handles of the previous owner stay stale.
    fn revive(&mut self, index: u32) -> u32 {
        let entry = &mut self.entries[index as usize];
        debug_assert!(!entry.live, "reviving a live index");
        entry.generation = entry.generation.wrapping_add(1).max(1);
        entry.live = true;
        self.live += 1;
        entry.generation
    }

    fn kind(&self, index: u32) -> u32 {
        self.entries.get(index as usize).map_or(u32::MAX, |entry| entry.kind)
    }
}

/// Which frame slots have applied a retirement's fallback write.
#[derive(Clone, Debug)]
struct Retirements {
    slots: Vec<bool>,
}

impl Retirements {
    fn new(frames: usize) -> Self {
        Self {
            slots: vec![false; frames],
        }
    }

    /// Mark `slot` flushed; true when every slot has.
    fn flush(&mut self, slot: usize) -> bool {
        if let Some(done) = self.slots.get_mut(slot) {
            *done = true;
        }
        self.slots.iter().all(|done| *done)
    }

    /// Whether any slot has applied the fallback yet. Until one has, the
    /// retiring index still names its old resource in every slot's set.
    fn any_flushed(&self) -> bool {
        self.slots.iter().any(|done| *done)
    }
}

#[derive(Clone, Copy, Debug)]
enum Write {
    Sampled {
        index: u32,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    },
    Sampler {
        index: u32,
        sampler: vk::Sampler,
    },
    StorageImage {
        index: u32,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    },
    StorageBuffer {
        index: u32,
        buffer: vk::Buffer,
        offset: u64,
        size: u64,
    },
}

/// The global set, its fallbacks and the handle slabs.
pub struct Bindless {
    gpu: Arc<GpuDevice>,
    frames: usize,
    set_layout: vk::DescriptorSetLayout,
    /// The one pipeline layout every engine pass uses: the global set at 0
    /// and [`Push`] in [`PUSH_STAGES`].
    pipeline_layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    sampled: Slab,
    samplers: Slab,
    storage_images: Slab,
    storage_buffers: Slab,
    pending: Vec<Vec<Write>>,
    retiring: Vec<(usize, u32, Retirements)>,
    /// The handles each open or finished [`GroupId`] owns.
    groups: Vec<Vec<(u8, u32, u32)>>,
    /// The groups currently being built; allocations go to the last one.
    open_groups: Vec<u32>,
    /// The created samplers by index (index 0 the fallback), and the
    /// settings that deduplicate them.
    sampler_objects: Vec<Option<SamplerEntry>>,
    sampler_of: HashMap<SamplerKey, u32>,
    warned: HashSet<(u32, u32)>,
    fallback_images: Vec<OwnedImage>,
    fallback_storage: OwnedImage,
    fallback_buffer: OwnedBuffer,
    sampler: vk::Sampler,
}

impl Bindless {
    pub fn new(gpu: &Arc<GpuDevice>, frames: usize) -> Result<Self> {
        anyhow::ensure!(frames > 0, "frames_in_flight must be positive");
        let properties = unsafe { gpu.instance.instance.get_physical_device_properties(gpu.physical) };
        check_limits(&properties.limits).map_err(|err| anyhow::anyhow!("{err}"))?;
        let device = &gpu.device;
        let stages = vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE;
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_SAMPLED_IMAGE)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(SAMPLED_IMAGE_CAPACITY)
                .stage_flags(stages),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_SAMPLER)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(SAMPLER_CAPACITY)
                .stage_flags(stages),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_STORAGE_IMAGE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(STORAGE_IMAGE_CAPACITY)
                .stage_flags(stages),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_STORAGE_BUFFER)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(STORAGE_BUFFER_CAPACITY)
                .stage_flags(stages),
        ];
        let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND; 4];
        let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .push_next(&mut flags_info);
        let set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;
        let pipeline_layout =
            crate::pipeline::pipeline_layout(device, &[set_layout], size_of::<Push>() as u32, PUSH_STAGES)?;

        let n = frames as u32;
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: SAMPLED_IMAGE_CAPACITY * n,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: SAMPLER_CAPACITY * n,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: STORAGE_IMAGE_CAPACITY * n,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: STORAGE_BUFFER_CAPACITY * n,
            },
        ];
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&sizes),
                None,
            )
        }?;
        let layouts = vec![set_layout; frames];
        let sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }?;

        // 1x1 fallbacks: white 2D and 2D array (the same texel), a uint
        // twin for `utexture2D`, and a zero RGBA32F storage image.
        let white = [255u8; 4];
        let white_layer = vec![&white[..]];
        let fallback_images = vec![
            upload::image(gpu, vk::Format::R8G8B8A8_UNORM, 1, 1, &[&white], "bindless fallback 2D")?,
            upload::image_layers(
                gpu,
                vk::Format::R8G8B8A8_UNORM,
                1,
                1,
                &[white_layer],
                vk::ImageViewType::TYPE_2D_ARRAY,
                "bindless fallback 2D array",
            )?,
            upload::image(gpu, vk::Format::R8_UINT, 1, 1, &[&[1u8][..]], "bindless fallback uint")?,
        ];
        let storage_desc = ImageDesc::color_2d(
            vk::Format::R32G32B32A32_SFLOAT,
            1,
            1,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
        );
        let fallback_storage = OwnedImage::new(gpu, storage_desc, "bindless fallback storage image")?;
        let storage_range = fallback_storage.range();
        upload::one_shot(gpu, |cmd| unsafe {
            let to_general = [vk::ImageMemoryBarrier2::default()
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(fallback_storage.image)
                .subresource_range(storage_range)];
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_general));
            let clear = vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            };
            device.cmd_clear_color_image(
                cmd,
                fallback_storage.image,
                vk::ImageLayout::GENERAL,
                &clear,
                &[storage_range],
            );
            let cleared = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(fallback_storage.image)
                .subresource_range(storage_range)];
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&cleared));
        })?;
        let fallback_buffer = upload::buffer(
            gpu,
            &[0u8; 256],
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "bindless fallback buffer",
        )?;
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler = unsafe { device.create_sampler(&sampler_info, None) }?;
        let mut this = Self {
            gpu: gpu.clone(),
            frames,
            set_layout,
            pipeline_layout,
            pool,
            sets,
            sampled: Slab::new("sampled image", 3, SAMPLED_IMAGE_CAPACITY, SampledKind::Image2D as u32),
            samplers: Slab::new("sampler", 1, SAMPLER_CAPACITY, 0),
            storage_images: Slab::new("storage image", 1, STORAGE_IMAGE_CAPACITY, 0),
            storage_buffers: Slab::new("storage buffer", 1, STORAGE_BUFFER_CAPACITY, 0),
            pending: (0..frames).map(|_| Vec::new()).collect(),
            retiring: Vec::new(),
            groups: Vec::new(),
            open_groups: Vec::new(),
            sampler_objects: vec![Some(SamplerEntry {
                key: SamplerKey::new(&sampler_info),
                sampler,
                refs: 1,
            })],
            sampler_of: HashMap::from([(SamplerKey::new(&sampler_info), 0)]),
            warned: HashSet::new(),
            fallback_images,
            fallback_storage,
            fallback_buffer,
            sampler,
        };
        // The reserved sampled entries stand for three different
        // declarations: 2D, 2D array and uint.
        this.sampled.entries[0].kind = SampledKind::Image2D as u32;
        this.sampled.entries[1].kind = SampledKind::Image2DArray as u32;
        this.sampled.entries[2].kind = SampledKind::Uint2D as u32;
        // The fallbacks are written directly: no frame can be recording yet.
        let fallbacks = [
            Write::Sampled {
                index: 0,
                view: this.fallback_images[0].view,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Write::Sampled {
                index: 1,
                view: this.fallback_images[1].view,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Write::Sampled {
                index: 2,
                view: this.fallback_images[2].view,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            Write::Sampler { index: 0, sampler },
            Write::StorageImage {
                index: 0,
                view: this.fallback_storage.view,
                layout: vk::ImageLayout::GENERAL,
            },
            Write::StorageBuffer {
                index: 0,
                buffer: this.fallback_buffer.buffer,
                offset: 0,
                size: this.fallback_buffer.size,
            },
        ];
        for slot in 0..frames {
            this.flush_writes(device, slot, fallbacks.iter().copied());
        }
        Ok(this)
    }

    /// The shared pipeline layout (M15 step E): the global set at 0 and the
    /// 128-byte [`Push`] block in every stage. Passes bind with it and never
    /// destroy it; [`Self::destroy`] does. Since every pass's layout is this
    /// one, a pipeline switch keeps set 0 and the push block valid.
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn set_layout(&self) -> vk::DescriptorSetLayout {
        self.set_layout
    }

    /// The set for a frame slot.
    pub fn set(&self, slot: usize) -> vk::DescriptorSet {
        self.sets[slot]
    }

    /// Build `make`'s handles as one group: every handle it registers
    /// (directly or through a module it builds) is retired together by
    /// [`Self::retire_group`]. Groups nest; a failure retires what was
    /// made before the error, since nothing else can be using it.
    pub fn group<R>(&mut self, make: impl FnOnce(&mut Self) -> Result<R>) -> Result<(R, GroupId)> {
        let id = GroupId(self.groups.len() as u32);
        self.groups.push(Vec::new());
        self.open_groups.push(id.0);
        let made = make(self);
        debug_assert_eq!(self.open_groups.pop(), Some(id.0), "groups close in order");
        match made {
            Ok(value) => Ok((value, id)),
            Err(err) => {
                self.retire_group(id);
                Err(err)
            }
        }
    }

    /// Retire every handle `group` registered. Handles retired already (or
    /// reused by another owner since) are left alone.
    pub fn retire_group(&mut self, group: GroupId) {
        for (slab, index, generation) in std::mem::take(&mut self.groups[group.0 as usize]) {
            self.retire_checked(slab as usize, index, generation);
        }
    }

    /// Record a handle in the group being built, if any.
    fn track(&mut self, slab: u8, index: u32, generation: u32) {
        if let Some(&group) = self.open_groups.last() {
            self.groups[group as usize].push((slab, index, generation));
        }
    }

    pub fn sampled_2d(&mut self, view: vk::ImageView, layout: vk::ImageLayout) -> Result<Sampled2D> {
        Ok(Sampled2D(self.register_sampled(view, layout, SampledKind::Image2D)?))
    }

    pub fn sampled_2d_array(&mut self, view: vk::ImageView, layout: vk::ImageLayout) -> Result<Sampled2DArray> {
        Ok(Sampled2DArray(self.register_sampled(
            view,
            layout,
            SampledKind::Image2DArray,
        )?))
    }

    pub fn sampled_u2d(&mut self, view: vk::ImageView, layout: vk::ImageLayout) -> Result<SampledU2D> {
        Ok(SampledU2D(self.register_sampled(view, layout, SampledKind::Uint2D)?))
    }

    fn register_sampled(
        &mut self,
        view: vk::ImageView,
        layout: vk::ImageLayout,
        kind: SampledKind,
    ) -> Result<ImageHandle> {
        let (index, generation) = self.sampled.alloc(kind as u32)?;
        self.queue_all(Write::Sampled { index, view, layout });
        self.track(0, index, generation);
        Ok(ImageHandle { index, generation })
    }

    /// A sampler for `info`, created once and shared: passes that ask for
    /// the same settings get the same index, and the registry owns the
    /// object. A retired index whose fallback no slot has applied yet is
    /// reused for the same settings: every slot's descriptor still names
    /// that sampler, so the new owner needs no write.
    pub fn sampler(&mut self, info: &vk::SamplerCreateInfo<'_>) -> Result<SamplerHandle> {
        let key = SamplerKey::new(info);
        if let Some(&index) = self.sampler_of.get(&key)
            && self.sampler_objects.get(index as usize).is_some_and(Option::is_some)
        {
            if self.samplers.is_live(index) {
                if let Some(entry) = self.sampler_objects[index as usize].as_mut() {
                    entry.refs += 1;
                }
                let generation = self.samplers.generation(index);
                self.track(1, index, generation);
                return Ok(SamplerHandle { index, generation });
            }
            let retiring = self
                .retiring
                .iter()
                .position(|(slab, retiring, state)| *slab == 1 && *retiring == index && !state.any_flushed());
            if let Some(at) = retiring {
                self.retiring.remove(at);
                for pending in &mut self.pending {
                    pending.retain(|write| !matches!(write, Write::Sampler { index: i, .. } if *i == index));
                }
                if let Some(entry) = self.sampler_objects[index as usize].as_mut() {
                    entry.refs = 1;
                }
                let generation = self.samplers.revive(index);
                self.track(1, index, generation);
                return Ok(SamplerHandle { index, generation });
            }
        }
        let sampler = unsafe { self.gpu.device.create_sampler(info, None) }?;
        let (index, generation) = self.samplers.alloc(0)?;
        if index as usize >= self.sampler_objects.len() {
            self.sampler_objects.resize_with(index as usize + 1, || None);
        }
        self.sampler_objects[index as usize] = Some(SamplerEntry { key, sampler, refs: 1 });
        self.sampler_of.insert(key, index);
        self.queue_all(Write::Sampler { index, sampler });
        self.track(1, index, generation);
        Ok(SamplerHandle { index, generation })
    }

    /// The sampler object behind a handle (the fallback's if it is stale),
    /// for a pass whose own layout still binds a combined image sampler.
    pub fn sampler_object(&self, handle: SamplerHandle) -> vk::Sampler {
        if self.samplers.resolve(handle.index, handle.generation)
            && let Some(entry) = self.sampler_objects.get(handle.index as usize).and_then(|e| *e)
        {
            return entry.sampler;
        }
        self.sampler
    }

    pub fn storage_image(&mut self, view: vk::ImageView, layout: vk::ImageLayout) -> Result<StorageImageHandle> {
        let (index, generation) = self.storage_images.alloc(0)?;
        self.queue_all(Write::StorageImage { index, view, layout });
        self.track(2, index, generation);
        Ok(StorageImageHandle { index, generation })
    }

    pub fn buffer(&mut self, buffer: vk::Buffer, offset: u64, size: u64) -> Result<BufferHandle> {
        let (index, generation) = self.storage_buffers.alloc(0)?;
        self.queue_all(Write::StorageBuffer {
            index,
            buffer,
            offset,
            size,
        });
        self.track(3, index, generation);
        Ok(BufferHandle { index, generation })
    }

    /// One sampled index whose descriptor differs per frame slot (a
    /// per-slot target). Every slot starts at the fallback;
    /// [`Self::update_slotted_2d`] writes one slot's set only, so a slot's
    /// set never references another slot's resource (which that slot may
    /// destroy while this one is in flight).
    pub fn slotted_2d(&mut self) -> Result<Sampled2D> {
        let (index, generation) = self.sampled.alloc(SampledKind::Image2D as u32)?;
        let view = self.fallback_images[SampledKind::Image2D.fallback() as usize].view;
        self.queue_all(Write::Sampled {
            index,
            view,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        });
        self.track(0, index, generation);
        Ok(Sampled2D(ImageHandle { index, generation }))
    }

    /// Point `slot`'s descriptor of a slotted handle at `view`. Applied at
    /// that slot's next flush (`begin_frame`, or [`Self::flush_open_slot`]).
    pub fn update_slotted_2d(&mut self, handle: Sampled2D, slot: usize, view: vk::ImageView, layout: vk::ImageLayout) {
        if self.sampled.resolve(handle.0.index, handle.0.generation) {
            self.pending[slot].push(Write::Sampled {
                index: handle.0.index,
                view,
                layout,
            });
        }
    }

    /// One storage-image index per frame slot, as [`Self::slotted_2d`].
    pub fn slotted_storage_image(&mut self) -> Result<StorageImageHandle> {
        let (index, generation) = self.storage_images.alloc(0)?;
        self.queue_all(Write::StorageImage {
            index,
            view: self.fallback_storage.view,
            layout: vk::ImageLayout::GENERAL,
        });
        self.track(2, index, generation);
        Ok(StorageImageHandle { index, generation })
    }

    pub fn update_slotted_storage_image(
        &mut self,
        handle: StorageImageHandle,
        slot: usize,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) {
        if self.storage_images.resolve(handle.index, handle.generation) {
            self.pending[slot].push(Write::StorageImage {
                index: handle.index,
                view,
                layout,
            });
        }
    }

    /// One storage-buffer index whose slot `k` descriptor is `buffers[k]`
    /// (whole), for a buffer the pass keeps one of per frame slot.
    pub fn per_slot_buffer(&mut self, buffers: &[crate::memory::OwnedBuffer]) -> Result<BufferHandle> {
        anyhow::ensure!(buffers.len() == self.frames, "one buffer per frame slot");
        let handle = self.slotted_buffer()?;
        for (slot, b) in buffers.iter().enumerate() {
            self.update_slotted_buffer(handle, slot, b.buffer, 0, b.size);
        }
        Ok(handle)
    }

    /// A whole buffer, the same in every slot.
    pub fn whole_buffer(&mut self, buffer: &crate::memory::OwnedBuffer) -> Result<BufferHandle> {
        self.buffer(buffer.buffer, 0, buffer.size)
    }

    /// One storage-buffer index per frame slot, as [`Self::slotted_2d`].
    pub fn slotted_buffer(&mut self) -> Result<BufferHandle> {
        let (index, generation) = self.storage_buffers.alloc(0)?;
        self.queue_all(Write::StorageBuffer {
            index,
            buffer: self.fallback_buffer.buffer,
            offset: 0,
            size: self.fallback_buffer.size,
        });
        self.track(3, index, generation);
        Ok(BufferHandle { index, generation })
    }

    pub fn update_slotted_buffer(
        &mut self,
        handle: BufferHandle,
        slot: usize,
        buffer: vk::Buffer,
        offset: u64,
        size: u64,
    ) {
        if self.storage_buffers.resolve(handle.index, handle.generation) {
            self.pending[slot].push(Write::StorageBuffer {
                index: handle.index,
                buffer,
                offset,
                size,
            });
        }
    }

    /// A graph resource's sampled index. `views` is one view per frame
    /// slot; the caller has waited for idle (`set_graph`), so every slot can
    /// be written directly.
    pub fn register_graph_sampled(
        &mut self,
        device: &ash::Device,
        kind: SampledKind,
        views: &[vk::ImageView],
        layout: vk::ImageLayout,
    ) -> Result<ImageHandle> {
        anyhow::ensure!(views.len() == self.frames, "one view per frame slot");
        let (index, generation) = self.sampled.alloc(kind as u32)?;
        self.write_all_slots(device, |slot| Write::Sampled {
            index,
            view: views[slot],
            layout,
        });
        Ok(ImageHandle { index, generation })
    }

    pub fn update_graph_sampled(
        &self,
        device: &ash::Device,
        handle: ImageHandle,
        views: &[vk::ImageView],
        layout: vk::ImageLayout,
    ) {
        self.write_all_slots(device, |slot| Write::Sampled {
            index: handle.index,
            view: views[slot],
            layout,
        });
    }

    pub fn register_graph_storage_image(
        &mut self,
        device: &ash::Device,
        views: &[vk::ImageView],
        layout: vk::ImageLayout,
    ) -> Result<StorageImageHandle> {
        anyhow::ensure!(views.len() == self.frames, "one view per frame slot");
        let (index, generation) = self.storage_images.alloc(0)?;
        self.write_all_slots(device, |slot| Write::StorageImage {
            index,
            view: views[slot],
            layout,
        });
        Ok(StorageImageHandle { index, generation })
    }

    pub fn update_graph_storage_image(
        &self,
        device: &ash::Device,
        handle: StorageImageHandle,
        views: &[vk::ImageView],
        layout: vk::ImageLayout,
    ) {
        self.write_all_slots(device, |slot| Write::StorageImage {
            index: handle.index,
            view: views[slot],
            layout,
        });
    }

    pub fn register_graph_buffer(
        &mut self,
        device: &ash::Device,
        buffers: &[(vk::Buffer, u64, u64)],
    ) -> Result<BufferHandle> {
        anyhow::ensure!(buffers.len() == self.frames, "one buffer per frame slot");
        let (index, generation) = self.storage_buffers.alloc(0)?;
        self.write_all_slots(device, |slot| {
            let (buffer, offset, size) = buffers[slot];
            Write::StorageBuffer {
                index,
                buffer,
                offset,
                size,
            }
        });
        Ok(BufferHandle { index, generation })
    }

    pub fn update_graph_buffer(&self, device: &ash::Device, handle: BufferHandle, buffers: &[(vk::Buffer, u64, u64)]) {
        self.write_all_slots(device, |slot| {
            let (buffer, offset, size) = buffers[slot];
            Write::StorageBuffer {
                index: handle.index,
                buffer,
                offset,
                size,
            }
        });
    }

    fn write_all_slots(&self, device: &ash::Device, write: impl Fn(usize) -> Write) {
        for slot in 0..self.frames {
            self.flush_writes(device, slot, std::iter::once(write(slot)));
        }
    }

    fn queue_all(&mut self, write: Write) {
        for slot in 0..self.frames {
            self.pending[slot].push(write);
        }
    }

    /// This handle's index, or its fallback if the handle has been retired.
    pub fn index_2d(&mut self, handle: Sampled2D) -> u32 {
        self.resolve_sampled(handle.0, SampledKind::Image2D)
    }

    pub fn index_2d_array(&mut self, handle: Sampled2DArray) -> u32 {
        self.resolve_sampled(handle.0, SampledKind::Image2DArray)
    }

    pub fn index_u2d(&mut self, handle: SampledU2D) -> u32 {
        self.resolve_sampled(handle.0, SampledKind::Uint2D)
    }

    pub fn index_sampler(&mut self, handle: SamplerHandle) -> u32 {
        if self.samplers.resolve(handle.index, handle.generation) {
            handle.index
        } else {
            self.warn(handle.index, handle.generation);
            0
        }
    }

    pub fn index_storage_image(&mut self, handle: StorageImageHandle) -> u32 {
        if self.storage_images.resolve(handle.index, handle.generation) {
            handle.index
        } else {
            self.warn(handle.index, handle.generation);
            0
        }
    }

    pub fn index_buffer(&mut self, handle: BufferHandle) -> u32 {
        if self.storage_buffers.resolve(handle.index, handle.generation) {
            handle.index
        } else {
            self.warn(handle.index, handle.generation);
            0
        }
    }

    fn resolve_sampled(&mut self, handle: ImageHandle, kind: SampledKind) -> u32 {
        if self.sampled.resolve(handle.index, handle.generation) {
            return handle.index;
        }
        self.warn(handle.index, handle.generation);
        // Use the retired entry's own type, so the fallback matches the
        // declaration the handle was registered for.
        SampledKind::from_u32(self.sampled.kind(handle.index))
            .unwrap_or(kind)
            .fallback()
    }

    fn warn(&mut self, index: u32, generation: u32) {
        if self.warned.insert((index, generation)) {
            eprintln!("bindless: stale handle {generation} at index {index}; drawing the fallback");
        }
    }

    /// Whether a live sampled handle was registered as `kind`.
    pub fn is_kind(&self, handle: ImageHandle, kind: SampledKind) -> bool {
        self.sampled.kind(handle.index) == kind as u32
    }

    /// Retire a sampled handle.
    pub fn retire_sampled(&mut self, handle: ImageHandle) {
        self.retire_checked(0, handle.index, handle.generation);
    }

    pub fn retire_sampler(&mut self, handle: SamplerHandle) {
        self.retire_checked(1, handle.index, handle.generation);
    }

    pub fn retire_storage_image(&mut self, handle: StorageImageHandle) {
        self.retire_checked(2, handle.index, handle.generation);
    }

    pub fn retire_buffer(&mut self, handle: BufferHandle) {
        self.retire_checked(3, handle.index, handle.generation);
    }

    /// Retire `index` if `generation` still owns it. A sampler index goes
    /// only when its last holder gives it back; the others go at once.
    fn retire_checked(&mut self, slab: usize, index: u32, generation: u32) {
        let live = match slab {
            0 => self.sampled.resolve(index, generation),
            1 => self.samplers.resolve(index, generation),
            2 => self.storage_images.resolve(index, generation),
            _ => self.storage_buffers.resolve(index, generation),
        };
        if !live {
            return;
        }
        if slab == 1 {
            let last = match self.sampler_objects.get_mut(index as usize).and_then(Option::as_mut) {
                Some(entry) if entry.refs > 1 => {
                    entry.refs -= 1;
                    false
                }
                _ => true,
            };
            if !last {
                return;
            }
        }
        self.retire(slab, index);
    }

    fn retire(&mut self, slab: usize, index: u32) {
        let retiring = match slab {
            0 => self.sampled.retire(index),
            1 => self.samplers.retire(index),
            2 => self.storage_images.retire(index),
            _ => self.storage_buffers.retire(index),
        };
        if !retiring {
            return;
        }
        let fallback = match slab {
            0 => SampledKind::from_u32(self.sampled.kind(index))
                .unwrap_or(SampledKind::Image2D)
                .fallback(),
            _ => 0,
        };
        // The retired index itself points at the fallback's contents, so a
        // command buffer recorded before the retire still reads white (or
        // the zero buffer), not another resource.
        let write = match slab {
            0 => Write::Sampled {
                index,
                view: self.fallback_images[fallback as usize].view,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            1 => Write::Sampler {
                index,
                sampler: self.sampler,
            },
            2 => Write::StorageImage {
                index,
                view: self.fallback_storage.view,
                layout: vk::ImageLayout::GENERAL,
            },
            _ => Write::StorageBuffer {
                index,
                buffer: self.fallback_buffer.buffer,
                offset: 0,
                size: self.fallback_buffer.size,
            },
        };
        for slot in 0..self.frames {
            self.pending[slot].push(write);
        }
        self.retiring.push((slab, index, Retirements::new(self.frames)));
    }

    /// Apply this slot's queued writes. The caller has waited the slot's
    /// fence (the set is idle), so no update-after-bind is needed.
    pub fn begin_frame(&mut self, device: &ash::Device, slot: usize) -> u64 {
        let pending = std::mem::take(&mut self.pending[slot]);
        let writes = self.flush_writes(device, slot, pending.into_iter());
        let mut freed = Vec::new();
        self.retiring.retain_mut(|(slab, index, retirements)| {
            if retirements.flush(slot) {
                freed.push((*slab, *index));
                false
            } else {
                true
            }
        });
        for (slab, index) in freed {
            match slab {
                0 => self.sampled.free(index),
                1 => {
                    // Every slot now names the fallback: the object can go.
                    if let Some(entry) = self.sampler_objects.get_mut(index as usize).and_then(Option::take) {
                        if self.sampler_of.get(&entry.key) == Some(&index) {
                            self.sampler_of.remove(&entry.key);
                        }
                        unsafe { device.destroy_sampler(entry.sampler, None) };
                    }
                    self.samplers.free(index);
                }
                2 => self.storage_images.free(index),
                _ => self.storage_buffers.free(index),
            }
        }
        writes
    }

    /// Apply `slot`'s writes queued since its `begin_frame`, before the
    /// frame records (`Executor::execute`). The slot's fence has been
    /// waited and none of its commands are recorded yet, so its set is
    /// still idle; a handle registered or updated in `prepare` is therefore
    /// usable in the same frame. Retirement bookkeeping stays with
    /// `begin_frame`.
    pub fn flush_open_slot(&mut self, device: &ash::Device, slot: usize) -> u64 {
        if self.pending[slot].is_empty() {
            return 0;
        }
        let pending = std::mem::take(&mut self.pending[slot]);
        self.flush_writes(device, slot, pending.into_iter())
    }

    fn flush_writes(&self, device: &ash::Device, slot: usize, pending: impl Iterator<Item = Write>) -> u64 {
        let pending: Vec<Write> = pending.collect();
        let mut images: Vec<[vk::DescriptorImageInfo; 1]> = Vec::new();
        let mut buffers: Vec<[vk::DescriptorBufferInfo; 1]> = Vec::new();
        // (binding, array element, type, index into `images` or `buffers`).
        let mut dsts: Vec<(u32, u32, vk::DescriptorType, usize)> = Vec::new();
        for item in pending {
            match item {
                Write::Sampled { index, view, layout } => {
                    images.push([vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: view,
                        image_layout: layout,
                    }]);
                    dsts.push((
                        BINDING_SAMPLED_IMAGE,
                        index,
                        vk::DescriptorType::SAMPLED_IMAGE,
                        images.len() - 1,
                    ));
                }
                Write::Sampler { index, sampler } => {
                    images.push([vk::DescriptorImageInfo {
                        sampler,
                        image_view: vk::ImageView::null(),
                        image_layout: vk::ImageLayout::UNDEFINED,
                    }]);
                    dsts.push((BINDING_SAMPLER, index, vk::DescriptorType::SAMPLER, images.len() - 1));
                }
                Write::StorageImage { index, view, layout } => {
                    images.push([vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: view,
                        image_layout: layout,
                    }]);
                    dsts.push((
                        BINDING_STORAGE_IMAGE,
                        index,
                        vk::DescriptorType::STORAGE_IMAGE,
                        images.len() - 1,
                    ));
                }
                Write::StorageBuffer {
                    index,
                    buffer,
                    offset,
                    size,
                } => {
                    buffers.push([vk::DescriptorBufferInfo {
                        buffer,
                        offset,
                        range: size,
                    }]);
                    dsts.push((
                        BINDING_STORAGE_BUFFER,
                        index,
                        vk::DescriptorType::STORAGE_BUFFER,
                        buffers.len() - 1,
                    ));
                }
            }
        }
        let writes: Vec<vk::WriteDescriptorSet<'_>> = dsts
            .iter()
            .map(|&(binding, index, ty, at)| {
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(self.sets[slot])
                    .dst_binding(binding)
                    .dst_array_element(index)
                    .descriptor_type(ty);
                if ty == vk::DescriptorType::STORAGE_BUFFER {
                    write.buffer_info(&buffers[at])
                } else {
                    write.image_info(&images[at])
                }
            })
            .collect();
        if !writes.is_empty() {
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        writes.len() as u64
    }

    /// How many distinct stale handles have been reported (each logs
    /// once).
    pub fn stale_warnings(&self) -> usize {
        self.warned.len()
    }

    /// Live entries per class, binding order.
    pub fn live(&self) -> [u32; CLASS_COUNT] {
        [
            self.sampled.live,
            self.samplers.live,
            self.storage_images.live,
            self.storage_buffers.live,
        ]
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        let device = &gpu.device;
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            for entry in self.sampler_objects.drain(..).flatten() {
                device.destroy_sampler(entry.sampler, None);
            }
        }
        for image in &mut self.fallback_images {
            image.destroy(gpu);
        }
        self.fallback_storage.destroy(gpu);
        self.fallback_buffer.destroy(gpu);
    }
}

/// A pass's handle table (bindless.glsl `AE_BUF(k)`): a storage buffer of
/// global buffer and storage-image indices, written once at creation. Every
/// per-pass buffer is fixed or slotted, so its index never changes and the
/// table needs no per-frame write. Its own index goes in push word 8
/// ([`Push::table`]).
pub struct HandleTable {
    buffer: OwnedBuffer,
    pub index: u32,
}

impl HandleTable {
    pub fn new(gpu: &GpuDevice, bindless: &mut Bindless, indices: &[u32], name: &str) -> Result<Self> {
        anyhow::ensure!(!indices.is_empty(), "{name}: an empty handle table");
        let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let buffer = upload::buffer(gpu, &bytes, vk::BufferUsageFlags::STORAGE_BUFFER, name)?;
        let index = bindless.whole_buffer(&buffer)?.index;
        Ok(Self { buffer, index })
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        self.buffer.destroy(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_follow_creation_order_and_reuse_the_oldest_free() {
        let mut slab = Slab::new("sampled image", 3, 8, 0);
        let (a, _) = slab.alloc(0).unwrap();
        let (b, _) = slab.alloc(0).unwrap();
        let (c, _) = slab.alloc(0).unwrap();
        assert_eq!((a, b, c), (3, 4, 5));
        assert!(slab.retire(4));
        slab.free(4);
        let (d, _) = slab.alloc(0).unwrap();
        assert_eq!(d, 4, "a freed index comes back before a new one");
        let (e, _) = slab.alloc(0).unwrap();
        assert_eq!(e, 6, "the high-water mark continues in creation order");
    }

    #[test]
    fn a_retired_index_waits_for_every_slot() {
        let mut slab = Slab::new("sampled image", 3, 8, 0);
        let (index, generation) = slab.alloc(0).unwrap();
        assert!(slab.retire(index));
        assert!(!slab.resolve(index, generation), "a retiring index no longer resolves");
        assert!(
            !slab.retire(index),
            "a second retire is refused, so the index is freed once"
        );
        let mut retirements = Retirements::new(2);
        assert!(!retirements.flush(0), "slot 1 has not applied the fallback");
        // The real registry does not free the index until this returns true.
        let (next, _) = slab.alloc(0).unwrap();
        assert_ne!(next, index, "the index is not free while slot 1 has not flushed");
        assert!(retirements.flush(1), "both slots have now flushed");
        slab.free(index);
        let (again, _) = slab.alloc(0).unwrap();
        assert_eq!(again, index, "the retired index is reused after the last slot");
    }

    #[test]
    fn a_stale_generation_does_not_resolve() {
        let mut slab = Slab::new("sampled image", 3, 8, SampledKind::Image2D as u32);
        let (index, first) = slab.alloc(SampledKind::Image2D as u32).unwrap();
        assert!(slab.retire(index));
        slab.free(index);
        let (again, second) = slab.alloc(SampledKind::Image2DArray as u32).unwrap();
        assert_eq!(again, index);
        assert_ne!(first, second);
        assert!(!slab.resolve(index, first), "the old generation is stale");
        assert!(slab.resolve(index, second));
        assert_eq!(slab.kind(index), SampledKind::Image2DArray as u32);
    }

    #[test]
    fn a_revived_index_serves_the_new_owner_and_stales_the_old_one() {
        let mut slab = Slab::new("sampler", 1, 8, 0);
        let (index, first) = slab.alloc(0).unwrap();
        assert!(slab.retire(index));
        assert_eq!(slab.live, 1, "the fallback alone is live");
        let second = slab.revive(index);
        assert_ne!(second, first, "the new owner gets a new generation");
        assert!(slab.resolve(index, second));
        assert!(!slab.resolve(index, first), "the old owner's handle is stale");
        assert_eq!(slab.live, 2);
    }

    #[test]
    fn a_retirement_flushes_every_slot_once() {
        let mut retirements = Retirements::new(3);
        assert!(!retirements.any_flushed(), "nothing applied the fallback yet");
        assert!(!retirements.flush(0));
        assert!(retirements.any_flushed());
        assert!(!retirements.flush(0), "a repeated flush of a slot is harmless");
        assert!(!retirements.flush(1));
        assert!(retirements.flush(2));
    }

    #[test]
    fn sampler_keys_follow_the_create_info() {
        let base = vk::SamplerCreateInfo::default().mag_filter(vk::Filter::LINEAR);
        assert_eq!(SamplerKey::new(&base), SamplerKey::new(&base));
        assert_ne!(
            SamplerKey::new(&base),
            SamplerKey::new(&base.min_filter(vk::Filter::LINEAR))
        );
        // The app's linear sampler (max_lod 0) and the terrain's (all mips).
        assert_ne!(
            SamplerKey::new(&base),
            SamplerKey::new(&base.max_lod(vk::LOD_CLAMP_NONE))
        );
        // A comparison sampler is not the same object as the plain one.
        assert_ne!(
            SamplerKey::new(&base),
            SamplerKey::new(&base.compare_enable(true).compare_op(vk::CompareOp::GREATER_OR_EQUAL))
        );
    }

    #[test]
    fn a_full_slab_names_its_class() {
        let mut slab = Slab::new("storage buffer", 1, 2, 0);
        slab.alloc(0).unwrap();
        let err = slab.alloc(0).unwrap_err().to_string();
        assert!(err.contains("storage buffer") && err.contains("capacity 2"), "{err}");
    }

    #[test]
    fn the_capacity_error_names_the_class_and_the_limit() {
        let mut limits = vk::PhysicalDeviceLimits {
            max_per_stage_descriptor_sampled_images: SAMPLED_IMAGE_CAPACITY,
            max_descriptor_set_sampled_images: SAMPLED_IMAGE_CAPACITY,
            max_per_stage_descriptor_samplers: SAMPLER_CAPACITY,
            max_descriptor_set_samplers: SAMPLER_CAPACITY,
            max_per_stage_descriptor_storage_images: STORAGE_IMAGE_CAPACITY,
            max_descriptor_set_storage_images: STORAGE_IMAGE_CAPACITY,
            max_per_stage_descriptor_storage_buffers: STORAGE_BUFFER_CAPACITY,
            max_descriptor_set_storage_buffers: STORAGE_BUFFER_CAPACITY,
            max_per_stage_resources: u32::MAX,
            ..Default::default()
        };
        assert!(check_limits(&limits).is_ok());
        limits.max_descriptor_set_storage_buffers = STORAGE_BUFFER_CAPACITY - 1;
        let err = check_limits(&limits).unwrap_err();
        assert!(err.contains("storage buffers"), "{err}");
    }
}
