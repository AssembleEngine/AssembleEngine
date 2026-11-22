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

use crate::device::GpuDevice;
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::upload;
use ae_framegraph::{BakedGraph, FrameGraph, ImageDesc, ResourceDesc, ResourceId};
use anyhow::{Result, bail};
use ash::vk;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;

/// The set number of the global set while the per-pass layouts still use
/// sets 0 and 1. Step E renumbers it to 0.
pub const BINDLESS_SET: u32 = 2;

pub const SAMPLED_IMAGE_CAPACITY: u32 = 4096;
pub const SAMPLER_CAPACITY: u32 = 64;
pub const STORAGE_IMAGE_CAPACITY: u32 = 256;
pub const STORAGE_BUFFER_CAPACITY: u32 = 1024;

pub const BINDING_SAMPLED_IMAGE: u32 = 0;
pub const BINDING_SAMPLER: u32 = 1;
pub const BINDING_STORAGE_IMAGE: u32 = 2;
pub const BINDING_STORAGE_BUFFER: u32 = 3;

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

/// Live counts, binding order, for `ExecutorStats`.
pub const CLASS_COUNT: usize = 4;

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

    fn free(&mut self, index: u32) {
        if let Some(entry) = self.entries.get_mut(index as usize)
            && entry.live
        {
            entry.live = false;
            self.live -= 1;
        }
        self.free.push_back(index);
    }

    fn resolve(&self, index: u32, generation: u32) -> bool {
        self.entries
            .get(index as usize)
            .is_some_and(|entry| entry.live && entry.generation == generation)
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
    frames: usize,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    sampled: Slab,
    samplers: Slab,
    storage_images: Slab,
    storage_buffers: Slab,
    pending: Vec<Vec<Write>>,
    retiring: Vec<(usize, u32, Retirements)>,
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
            frames,
            set_layout,
            pool,
            sets,
            sampled: Slab::new("sampled image", 3, SAMPLED_IMAGE_CAPACITY, SampledKind::Image2D as u32),
            samplers: Slab::new("sampler", 1, SAMPLER_CAPACITY, 0),
            storage_images: Slab::new("storage image", 1, STORAGE_IMAGE_CAPACITY, 0),
            storage_buffers: Slab::new("storage buffer", 1, STORAGE_BUFFER_CAPACITY, 0),
            pending: (0..frames).map(|_| Vec::new()).collect(),
            retiring: Vec::new(),
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

    pub fn set_layout(&self) -> vk::DescriptorSetLayout {
        self.set_layout
    }

    /// The set for a frame slot.
    pub fn set(&self, slot: usize) -> vk::DescriptorSet {
        self.sets[slot]
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
        Ok(ImageHandle { index, generation })
    }

    pub fn sampler(&mut self, sampler: vk::Sampler) -> Result<SamplerHandle> {
        let (index, generation) = self.samplers.alloc(0)?;
        self.queue_all(Write::Sampler { index, sampler });
        Ok(SamplerHandle { index, generation })
    }

    pub fn storage_image(&mut self, view: vk::ImageView, layout: vk::ImageLayout) -> Result<StorageImageHandle> {
        let (index, generation) = self.storage_images.alloc(0)?;
        self.queue_all(Write::StorageImage { index, view, layout });
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
        Ok(BufferHandle { index, generation })
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

    /// Retire a sampled handle.
    pub fn retire_sampled(&mut self, handle: ImageHandle) {
        if self.sampled.resolve(handle.index, handle.generation) {
            self.retire(0, handle.index);
        }
    }

    pub fn retire_sampler(&mut self, handle: SamplerHandle) {
        if self.samplers.resolve(handle.index, handle.generation) {
            self.retire(1, handle.index);
        }
    }

    pub fn retire_storage_image(&mut self, handle: StorageImageHandle) {
        if self.storage_images.resolve(handle.index, handle.generation) {
            self.retire(2, handle.index);
        }
    }

    pub fn retire_buffer(&mut self, handle: BufferHandle) {
        if self.storage_buffers.resolve(handle.index, handle.generation) {
            self.retire(3, handle.index);
        }
    }

    fn retire(&mut self, slab: usize, index: u32) {
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
                1 => self.samplers.free(index),
                2 => self.storage_images.free(index),
                _ => self.storage_buffers.free(index),
            }
        }
        writes
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
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
        }
        for image in &mut self.fallback_images {
            image.destroy(gpu);
        }
        self.fallback_storage.destroy(gpu);
        self.fallback_buffer.destroy(gpu);
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
        slab.free(4);
        let (d, _) = slab.alloc(0).unwrap();
        assert_eq!(d, 4, "a freed index comes back before a new one");
        let (e, _) = slab.alloc(0).unwrap();
        assert_eq!(e, 6, "the high-water mark continues in creation order");
    }

    #[test]
    fn a_retired_index_waits_for_every_slot() {
        let mut slab = Slab::new("sampled image", 3, 8, 0);
        let (index, _) = slab.alloc(0).unwrap();
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
        slab.free(index);
        let (again, second) = slab.alloc(SampledKind::Image2DArray as u32).unwrap();
        assert_eq!(again, index);
        assert_ne!(first, second);
        assert!(!slab.resolve(index, first), "the old generation is stale");
        assert!(slab.resolve(index, second));
        assert_eq!(slab.kind(index), SampledKind::Image2DArray as u32);
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
