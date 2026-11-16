//! One-shot uploads of static data (terrain textures, lookup tables, meshes)
//! through a staging buffer, outside the frame graph: the call waits for
//! the copy, so the data is ready (images in `SHADER_READ_ONLY_OPTIMAL`)
//! before any frame uses it. For data that never changes after creation;
//! per-frame data goes through graph resources.

use crate::device::{GpuDevice, fatal};
use crate::memory::{OwnedBuffer, OwnedImage};
use ae_framegraph::{BufferDesc, ImageDesc};
use anyhow::{Context as _, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;

/// Record `record` into a fresh command buffer, submit it and wait.
pub fn one_shot(gpu: &GpuDevice, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
    let device = &gpu.device;
    unsafe {
        let pool = device
            .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(gpu.queue_family).flags(vk::CommandPoolCreateFlags::TRANSIENT), None)
            .context("upload command pool")?;
        let result = (|| -> Result<()> {
            let cmd = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1))?[0];
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            record(cmd);
            device.end_command_buffer(cmd)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let submit = vk::SubmitInfo2::default().command_buffer_infos(&cmds);
            let submitted = device.queue_submit2(gpu.queue, &[submit], fence).map_err(fatal).context("upload submit");
            let waited = submitted.and_then(|()| device.wait_for_fences(&[fence], true, u64::MAX).map_err(fatal).context("upload wait"));
            device.destroy_fence(fence, None);
            waited
        })();
        device.destroy_command_pool(pool, None);
        result
    }
}

fn staging(gpu: &GpuDevice, bytes: &[u8], name: &str) -> Result<OwnedBuffer> {
    let desc = BufferDesc { size: bytes.len().max(4) as u64, usage: vk::BufferUsageFlags::TRANSFER_SRC };
    let mut buffer = OwnedBuffer::new(gpu, desc, MemoryLocation::CpuToGpu, name)?;
    buffer.mapped().context("staging buffer is not host-visible")?[..bytes.len()].copy_from_slice(bytes);
    Ok(buffer)
}

/// A device-local buffer holding `bytes` (`usage` gets `TRANSFER_DST` added).
pub fn buffer(gpu: &GpuDevice, bytes: &[u8], usage: vk::BufferUsageFlags, name: &str) -> Result<OwnedBuffer> {
    buffer_sized(gpu, bytes.len() as u64, bytes, usage, name)
}

/// A device-local buffer of `size` bytes starting with `bytes` (the rest
/// undefined; `usage` gets `TRANSFER_DST` added).
pub fn buffer_sized(gpu: &GpuDevice, size: u64, bytes: &[u8], usage: vk::BufferUsageFlags, name: &str) -> Result<OwnedBuffer> {
    anyhow::ensure!(bytes.len() as u64 <= size, "{name}: {} bytes into a buffer of {size}", bytes.len());
    let desc = BufferDesc { size: size.max(4), usage: usage | vk::BufferUsageFlags::TRANSFER_DST };
    let target = OwnedBuffer::new(gpu, desc, MemoryLocation::GpuOnly, name)?;
    let mut source = staging(gpu, bytes, "staging")?;
    let size = bytes.len() as u64;
    let copied = one_shot(gpu, |cmd| unsafe {
        if size > 0 {
            gpu.device.cmd_copy_buffer(cmd, source.buffer, target.buffer, &[vk::BufferCopy::default().size(size)]);
        }
        // Visible to every later use on this queue.
        let barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
        gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&barrier));
    });
    source.destroy(gpu);
    if let Err(err) = copied {
        let mut target = target;
        target.destroy(gpu);
        return Err(err);
    }
    Ok(target)
}

/// `size` bytes of `buffer` from `offset`, copied to the host (for tests and
/// tools: it waits for the device; `buffer` needs `TRANSFER_SRC`).
pub fn read_buffer(gpu: &GpuDevice, buffer: vk::Buffer, offset: u64, size: u64) -> Result<Vec<u8>> {
    let desc = BufferDesc { size: size.max(4), usage: vk::BufferUsageFlags::TRANSFER_DST };
    let mut host = OwnedBuffer::new(gpu, desc, MemoryLocation::GpuToCpu, "readback")?;
    let copied = one_shot(gpu, |cmd| unsafe {
        // After every earlier use on this queue.
        let before = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
        gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&before));
        if size > 0 {
            gpu.device.cmd_copy_buffer(cmd, buffer, host.buffer, &[vk::BufferCopy::default().src_offset(offset).size(size)]);
        }
        let after = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&after));
    });
    let bytes = copied.and_then(|()| Ok(host.mapped().context("readback buffer is not host-visible")?[..size as usize].to_vec()));
    host.destroy(gpu);
    bytes
}

/// A sampled 2D colour image with one mip per entry of `mips` (mip `i` is
/// `width >> i` by `height >> i`, tightly packed), left in
/// `SHADER_READ_ONLY_OPTIMAL`.
pub fn image(gpu: &GpuDevice, format: vk::Format, width: u32, height: u32, mips: &[&[u8]], name: &str) -> Result<OwnedImage> {
    image_layers(gpu, format, width, height, &[mips.to_vec()], vk::ImageViewType::TYPE_2D, name)
}

/// A sampled 2D (array) colour image: `layers[l][m]` is mip `m` of layer
/// `l` (every layer the same mip count), with a view of `view_type`, left
/// in `SHADER_READ_ONLY_OPTIMAL`.
pub fn image_layers(gpu: &GpuDevice, format: vk::Format, width: u32, height: u32, layers: &[Vec<&[u8]>], view_type: vk::ImageViewType, name: &str) -> Result<OwnedImage> {
    let mip_levels = layers.first().map_or(1, Vec::len) as u32;
    anyhow::ensure!(!layers.is_empty() && layers.iter().all(|l| l.len() as u32 == mip_levels), "{name}: every layer needs the same mips");
    let desc = ImageDesc { mip_levels, layers: layers.len() as u32, ..ImageDesc::color_2d(format, width, height, vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST) };
    let target = OwnedImage::with_view(gpu, desc, view_type, name)?;
    // Regions at 16-byte offsets (a multiple of any texel size here).
    let mut bytes = Vec::new();
    let mut regions = Vec::new();
    for (layer, mips) in layers.iter().enumerate() {
        for (level, mip) in mips.iter().enumerate() {
            bytes.resize(bytes.len().next_multiple_of(16), 0);
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(bytes.len() as u64)
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: level as u32, base_array_layer: layer as u32, layer_count: 1 })
                    .image_extent(vk::Extent3D { width: (width >> level).max(1), height: (height >> level).max(1), depth: 1 }),
            );
            bytes.extend_from_slice(mip);
        }
    }
    let mut source = staging(gpu, &bytes, "staging")?;
    let range = target.range();
    let copied = one_shot(gpu, |cmd| unsafe {
        let to_dst = [vk::ImageMemoryBarrier2::default()
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target.image)
            .subresource_range(range)];
        gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_dst));
        gpu.device.cmd_copy_buffer_to_image(cmd, source.buffer, target.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions);
        let to_read = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target.image)
            .subresource_range(range)];
        gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_read));
    });
    source.destroy(gpu);
    if let Err(err) = copied {
        let mut target = target;
        target.destroy(gpu);
        return Err(err);
    }
    Ok(target)
}

/// A rectangle of one mip of an image to rewrite: `bytes` holds `width` x
/// `height` texels, tightly packed.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub mip: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}

/// Rewrite `regions` of sampled images made by [`image`] (in
/// `SHADER_READ_ONLY_OPTIMAL`, which they are left in again), keeping every
/// texel outside them. The caller makes sure no submitted work still uses
/// the images (a device wait-idle); the copy itself is waited for.
pub fn update_images(gpu: &GpuDevice, updates: &[(&OwnedImage, &[Region])]) -> Result<()> {
    // Each region starts at a multiple of 16 bytes: a copy's buffer offset
    // must be a multiple of its image's texel size (1 to 16 bytes here).
    const ALIGN: usize = 16;
    let mut bytes = Vec::new();
    let mut offsets = Vec::new();
    for r in updates.iter().flat_map(|(_, regions)| regions.iter()) {
        bytes.resize(bytes.len().next_multiple_of(ALIGN), 0);
        offsets.push(bytes.len() as u64);
        bytes.extend_from_slice(&r.bytes);
    }
    if bytes.is_empty() {
        return Ok(());
    }
    let mut source = staging(gpu, &bytes, "staging")?;
    let barrier = |image: &OwnedImage, old: vk::ImageLayout, new: vk::ImageLayout, src: (vk::PipelineStageFlags2, vk::AccessFlags2), dst: (vk::PipelineStageFlags2, vk::AccessFlags2)| {
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src.0)
            .src_access_mask(src.1)
            .dst_stage_mask(dst.0)
            .dst_access_mask(dst.1)
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image.image)
            .subresource_range(image.range())
    };
    let read = (vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::SHADER_SAMPLED_READ);
    let copy = (vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_WRITE);
    let copied = one_shot(gpu, |cmd| unsafe {
        let to_dst: Vec<_> = updates.iter().filter(|(_, r)| !r.is_empty()).map(|(image, _)| barrier(image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL, read, copy)).collect();
        gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_dst));
        let mut offsets = offsets.iter();
        for (image, regions) in updates {
            let copies: Vec<vk::BufferImageCopy> = regions
                .iter()
                .map(|r| {
                    vk::BufferImageCopy::default()
                        .buffer_offset(*offsets.next().expect("an offset per region"))
                        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: r.mip, base_array_layer: 0, layer_count: 1 })
                        .image_offset(vk::Offset3D { x: r.x as i32, y: r.y as i32, z: 0 })
                        .image_extent(vk::Extent3D { width: r.width, height: r.height, depth: 1 })
                })
                .collect();
            if !copies.is_empty() {
                gpu.device.cmd_copy_buffer_to_image(cmd, source.buffer, image.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copies);
            }
        }
        let to_read: Vec<_> = updates.iter().filter(|(_, r)| !r.is_empty()).map(|(image, _)| barrier(image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, copy, read)).collect();
        gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_read));
    });
    source.destroy(gpu);
    copied
}
