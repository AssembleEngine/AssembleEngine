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
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(gpu.queue_family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )
            .context("upload command pool")?;
        let result = (|| -> Result<()> {
            let cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(1),
            )?[0];
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            record(cmd);
            device.end_command_buffer(cmd)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let submit = vk::SubmitInfo2::default().command_buffer_infos(&cmds);
            let submitted = device
                .queue_submit2(gpu.queue, &[submit], fence)
                .map_err(fatal)
                .context("upload submit");
            let waited = submitted.and_then(|()| {
                device
                    .wait_for_fences(&[fence], true, u64::MAX)
                    .map_err(fatal)
                    .context("upload wait")
            });
            device.destroy_fence(fence, None);
            waited
        })();
        device.destroy_command_pool(pool, None);
        result
    }
}

fn staging(gpu: &GpuDevice, bytes: &[u8], name: &str) -> Result<OwnedBuffer> {
    let desc = BufferDesc {
        size: bytes.len().max(4) as u64,
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
    };
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
pub fn buffer_sized(
    gpu: &GpuDevice,
    size: u64,
    bytes: &[u8],
    usage: vk::BufferUsageFlags,
    name: &str,
) -> Result<OwnedBuffer> {
    anyhow::ensure!(
        bytes.len() as u64 <= size,
        "{name}: {} bytes into a buffer of {size}",
        bytes.len()
    );
    let desc = BufferDesc {
        size: size.max(4),
        usage: usage | vk::BufferUsageFlags::TRANSFER_DST,
    };
    let target = OwnedBuffer::new(gpu, desc, MemoryLocation::GpuOnly, name)?;
    let mut source = staging(gpu, bytes, "staging")?;
    let size = bytes.len() as u64;
    let copied = one_shot(gpu, |cmd| unsafe {
        if size > 0 {
            gpu.device.cmd_copy_buffer(
                cmd,
                source.buffer,
                target.buffer,
                &[vk::BufferCopy::default().size(size)],
            );
        }
        // Visible to every later use on this queue.
        let barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
        gpu.device
            .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&barrier));
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
    let desc = BufferDesc {
        size: size.max(4),
        usage: vk::BufferUsageFlags::TRANSFER_DST,
    };
    let mut host = OwnedBuffer::new(gpu, desc, MemoryLocation::GpuToCpu, "readback")?;
    let copied = one_shot(gpu, |cmd| unsafe {
        // After every earlier use on this queue.
        let before = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
        gpu.device
            .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&before));
        if size > 0 {
            gpu.device.cmd_copy_buffer(
                cmd,
                buffer,
                host.buffer,
                &[vk::BufferCopy::default().src_offset(offset).size(size)],
            );
        }
        let after = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        gpu.device
            .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&after));
    });
    let bytes = copied
        .and_then(|()| Ok(host.mapped().context("readback buffer is not host-visible")?[..size as usize].to_vec()));
    host.destroy(gpu);
    bytes
}

/// A sampled 2D colour image with one mip per entry of `mips` (mip `i` is
/// `width >> i` by `height >> i`, tightly packed), left in
/// `SHADER_READ_ONLY_OPTIMAL`.
pub fn image(
    gpu: &GpuDevice,
    format: vk::Format,
    width: u32,
    height: u32,
    mips: &[&[u8]],
    name: &str,
) -> Result<OwnedImage> {
    image_layers(
        gpu,
        format,
        width,
        height,
        &[mips.to_vec()],
        vk::ImageViewType::TYPE_2D,
        name,
    )
}

pub(crate) fn mipped_image_spec(
    format: vk::Format,
    width: u32,
    height: u32,
    layers: u32,
) -> (ImageDesc, vk::ImageCreateFlags, vk::Format) {
    let twin = storage_twin(format);
    let flags = if twin != format {
        vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE
    } else {
        vk::ImageCreateFlags::empty()
    };
    let usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::STORAGE;
    let desc = ImageDesc {
        mip_levels: mip_levels(width, height),
        layers,
        ..ImageDesc::color_2d(format, width, height, usage)
    };
    (desc, flags, twin)
}

pub(crate) const MIP_STORAGE_FEATURES: vk::FormatFeatureFlags2 = vk::FormatFeatureFlags2::from_raw(
    vk::FormatFeatureFlags2::STORAGE_IMAGE.as_raw() | vk::FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT.as_raw(),
);

/// A sampled 2D (array) colour image: `layers[l][m]` is mip `m` of layer
/// `l` (every layer the same mip count), with a view of `view_type`, left
/// in `SHADER_READ_ONLY_OPTIMAL`.
pub fn image_layers(
    gpu: &GpuDevice,
    format: vk::Format,
    width: u32,
    height: u32,
    layers: &[Vec<&[u8]>],
    view_type: vk::ImageViewType,
    name: &str,
) -> Result<OwnedImage> {
    let mip_levels = layers.first().map_or(1, Vec::len) as u32;
    anyhow::ensure!(
        !layers.is_empty() && layers.iter().all(|l| l.len() as u32 == mip_levels),
        "{name}: every layer needs the same mips"
    );
    let desc = ImageDesc {
        mip_levels,
        layers: layers.len() as u32,
        ..ImageDesc::color_2d(
            format,
            width,
            height,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )
    };
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
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: level as u32,
                        base_array_layer: layer as u32,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: (width >> level).max(1),
                        height: (height >> level).max(1),
                        depth: 1,
                    }),
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
        gpu.device
            .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_dst));
        gpu.device.cmd_copy_buffer_to_image(
            cmd,
            source.buffer,
            target.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
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
        gpu.device
            .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_read));
    });
    source.destroy(gpu);
    if let Err(err) = copied {
        let mut target = target;
        target.destroy(gpu);
        return Err(err);
    }
    Ok(target)
}

/// Every mip of a sampled colour image (in `SHADER_READ_ONLY_OPTIMAL`,
/// made with `TRANSFER_SRC`, as [`image_mipped`]'s), copied to the host:
/// per level, its layers one after another, `texel_bytes` a texel. For tests
/// and tools: it waits for the device.
pub fn read_image(gpu: &GpuDevice, image: &OwnedImage, texel_bytes: u64) -> Result<Vec<Vec<u8>>> {
    read_image_from(gpu, image, texel_bytes, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
}

/// [`read_image`] for an image in `layout` (e.g. a storage image left in
/// GENERAL): it is copied from that layout and left in it again.
pub fn read_image_from(
    gpu: &GpuDevice,
    image: &OwnedImage,
    texel_bytes: u64,
    from_layout: vk::ImageLayout,
) -> Result<Vec<Vec<u8>>> {
    let desc = image.desc;
    let sizes: Vec<u64> = (0..desc.mip_levels)
        .map(|i| ((desc.width >> i).max(1) * (desc.height >> i).max(1)) as u64 * desc.layers as u64 * texel_bytes)
        .collect();
    let offsets: Vec<u64> = sizes
        .iter()
        .scan(0u64, |at, &s| {
            let o = *at;
            *at = (o + s).next_multiple_of(16);
            Some(o)
        })
        .collect();
    let total = offsets.last().copied().unwrap_or(0) + sizes.last().copied().unwrap_or(0);
    let mut host = OwnedBuffer::new(
        gpu,
        BufferDesc {
            size: total.max(4),
            usage: vk::BufferUsageFlags::TRANSFER_DST,
        },
        MemoryLocation::GpuToCpu,
        "image readback",
    )?;
    let range = image.range();
    let copied = one_shot(gpu, |cmd| unsafe {
        let layout = |from,
                      to,
                      src: (vk::PipelineStageFlags2, vk::AccessFlags2),
                      dst: (vk::PipelineStageFlags2, vk::AccessFlags2)| {
            [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src.0)
                .src_access_mask(src.1)
                .dst_stage_mask(dst.0)
                .dst_access_mask(dst.1)
                .old_layout(from)
                .new_layout(to)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.image)
                .subresource_range(range)]
        };
        let any = (
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
        );
        let copy = (vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_READ);
        let before = layout(from_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, any, copy);
        gpu.device
            .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&before));
        let regions: Vec<vk::BufferImageCopy> = (0..desc.mip_levels)
            .map(|i| {
                vk::BufferImageCopy::default()
                    .buffer_offset(offsets[i as usize])
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: i,
                        base_array_layer: 0,
                        layer_count: desc.layers,
                    })
                    .image_extent(vk::Extent3D {
                        width: (desc.width >> i).max(1),
                        height: (desc.height >> i).max(1),
                        depth: 1,
                    })
            })
            .collect();
        gpu.device.cmd_copy_image_to_buffer(
            cmd,
            image.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            host.buffer,
            &regions,
        );
        let after = layout(
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            from_layout,
            copy,
            (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
        );
        gpu.device
            .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&after));
        let to_host = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        gpu.device
            .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&to_host));
    });
    let levels = copied.and_then(|()| {
        let bytes = host.mapped().context("readback buffer is not host-visible")?;
        Ok(offsets
            .iter()
            .zip(&sizes)
            .map(|(&o, &s)| bytes[o as usize..(o + s) as usize].to_vec())
            .collect())
    });
    host.destroy(gpu);
    levels
}

/// How [`image_mipped`] makes each mip from the one above (a 2x2 box,
/// `shaders/mipgen.comp`). sRGB images are averaged in linear light.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MipFilter {
    /// Every channel averaged.
    Average = 0,
    /// Colour with coverage in alpha: rgb averaged weighted by alpha, so
    /// transparent texels leave no colour on the edges of smaller mips;
    /// alpha averaged.
    Coverage = 1,
    /// A tangent normal's x, y in rg (`n * 0.5 + 0.5`, z out of the surface
    /// rebuilt): the vectors summed and renormalised; b, a averaged.
    Normal = 2,
    /// Signed tangent XY with positive Z reconstructed before averaging.
    NormalSigned = 3,
}

/// Mips of a full chain down to 1x1.
pub fn mip_levels(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

/// The UNORM format an sRGB format's texels are stored as (the mip
/// generator's storage view; sRGB formats are not storage formats).
fn storage_twin(format: vk::Format) -> vk::Format {
    match format {
        vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_SRGB => vk::Format::B8G8R8A8_UNORM,
        vk::Format::R8G8_SRGB => vk::Format::R8G8_UNORM,
        vk::Format::R8_SRGB => vk::Format::R8_UNORM,
        f => f,
    }
}

/// A sampled 2D (array) colour image from mip 0 of each layer (`layers[l]`,
/// tightly packed), its full mip chain ([`mip_levels`]) made on the GPU by
/// `filter`, with a view of `view_type`, left in `SHADER_READ_ONLY_OPTIMAL`.
/// The engine's one way to mip a texture: a level's texel is the 2x2 box of
/// the level above, clamped at its edge (so power-of-two sides tile exactly;
/// an odd side drops its last row or column).
#[allow(clippy::too_many_arguments)]
pub fn image_mipped(
    gpu: &GpuDevice,
    format: vk::Format,
    width: u32,
    height: u32,
    layers: &[&[u8]],
    filter: MipFilter,
    view_type: vk::ImageViewType,
    name: &str,
) -> Result<OwnedImage> {
    anyhow::ensure!(!layers.is_empty() && width > 0 && height > 0, "{name}: no texels");
    let (desc, flags, twin) = mipped_image_spec(format, width, height, layers.len() as u32);
    let srgb = twin != format;
    anyhow::ensure!(
        gpu.optimal_format_features(twin).contains(MIP_STORAGE_FEATURES),
        "{name}: {twin:?} takes no formatless storage writes here, so its mips cannot be made"
    );
    let levels = desc.mip_levels;
    let mut target = OwnedImage::with_view_flags(gpu, desc, view_type, flags, name)?;
    let mut bytes = Vec::new();
    let mut regions = Vec::new();
    for (layer, texels) in layers.iter().enumerate() {
        bytes.resize(bytes.len().next_multiple_of(16), 0);
        regions.push(
            vk::BufferImageCopy::default()
                .buffer_offset(bytes.len() as u64)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: layer as u32,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                }),
        );
        bytes.extend_from_slice(texels);
    }
    let mut source = staging(gpu, &bytes, "staging")?;
    let made = MipGen::new(gpu, &target, twin, levels).and_then(|mut mipgen| {
        let range = target.range();
        let image = target.image;
        let barrier = |src: (vk::PipelineStageFlags2, vk::AccessFlags2),
                       dst: (vk::PipelineStageFlags2, vk::AccessFlags2),
                       layouts: (vk::ImageLayout, vk::ImageLayout),
                       range: vk::ImageSubresourceRange| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src.0)
                .src_access_mask(src.1)
                .dst_stage_mask(dst.0)
                .dst_access_mask(dst.1)
                .old_layout(layouts.0)
                .new_layout(layouts.1)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
        };
        let none = (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE);
        let copy_write = (vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_WRITE);
        let compute_write = (
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
        let compute_rw = (
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
        let general = vk::ImageLayout::GENERAL;
        let level = |i: u32| vk::ImageSubresourceRange {
            base_mip_level: i,
            level_count: 1,
            ..range
        };
        let push = [filter as u32, u32::from(srgb)];
        let done = one_shot(gpu, |cmd| unsafe {
            let d = &gpu.device;
            let to_dst = [barrier(
                none,
                copy_write,
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                range,
            )];
            d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_dst));
            d.cmd_copy_buffer_to_image(
                cmd,
                source.buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
            // The whole chain in GENERAL while it is made: level i - 1 is
            // read sampled while level i is written as storage.
            let to_general = [barrier(
                copy_write,
                compute_rw,
                (vk::ImageLayout::TRANSFER_DST_OPTIMAL, general),
                range,
            )];
            d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_general));
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, mipgen.pipeline);
            d.cmd_push_constants(
                cmd,
                mipgen.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                crate::pipeline::push_bytes(&push),
            );
            for i in 1..levels {
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    mipgen.layout,
                    0,
                    &[mipgen.sets[i as usize - 1]],
                    &[],
                );
                let (w, h) = ((width >> i).max(1), (height >> i).max(1));
                d.cmd_dispatch(cmd, w.div_ceil(8), h.div_ceil(8), layers.len() as u32);
                let written = [barrier(compute_write, compute_rw, (general, general), level(i))];
                d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&written));
            }
            let all = (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
            );
            let to_read = [barrier(
                compute_rw,
                all,
                (general, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                range,
            )];
            d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_read));
        });
        mipgen.destroy(gpu);
        done
    });
    source.destroy(gpu);
    if let Err(err) = made {
        target.destroy(gpu);
        return Err(err.context(format!("making the mips of {name}")));
    }
    Ok(target)
}

/// The mip generator's objects for one image (made and destroyed per call:
/// it runs a few times at start-up).
struct MipGen {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    /// Per level from 1: (sampled view of the level above, storage view).
    views: Vec<vk::ImageView>,
    sets: Vec<vk::DescriptorSet>,
}

impl MipGen {
    fn new(gpu: &GpuDevice, target: &OwnedImage, twin: vk::Format, levels: u32) -> Result<Self> {
        let d = &gpu.device;
        let mut m = MipGen {
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            pool: vk::DescriptorPool::null(),
            sampler: vk::Sampler::null(),
            views: Vec::new(),
            sets: Vec::new(),
        };
        let built = (|| -> Result<()> {
            let compute = vk::ShaderStageFlags::COMPUTE;
            m.set_layout = crate::pipeline::set_layout(
                d,
                &[
                    (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, compute),
                    (vk::DescriptorType::STORAGE_IMAGE, compute),
                ],
            )?;
            m.layout = crate::pipeline::pipeline_layout(d, &[m.set_layout], 8, compute)?;
            m.pipeline = crate::pipeline::compute_pipeline(d, crate::pipeline::spirv!("mipgen.comp"), m.layout)?;
            if levels < 2 {
                return Ok(());
            }
            let count = levels - 1;
            unsafe {
                m.sampler = d.create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST)
                        .max_lod(vk::LOD_CLAMP_NONE),
                    None,
                )?;
                let sizes = [
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: count,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: count,
                    },
                ];
                m.pool = d.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(count)
                        .pool_sizes(&sizes),
                    None,
                )?;
                let layouts = vec![m.set_layout; count as usize];
                m.sets = d.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(m.pool)
                        .set_layouts(&layouts),
                )?;
            }
            let range = target.range();
            let view =
                |m: &mut MipGen, level: u32, format: vk::Format, usage: vk::ImageUsageFlags| -> Result<vk::ImageView> {
                    let mut usage = vk::ImageViewUsageCreateInfo::default().usage(usage);
                    let info = vk::ImageViewCreateInfo::default()
                        .image(target.image)
                        .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                        .format(format)
                        .subresource_range(vk::ImageSubresourceRange {
                            base_mip_level: level,
                            level_count: 1,
                            ..range
                        })
                        .push_next(&mut usage);
                    let v = unsafe { d.create_image_view(&info, None) }?;
                    m.views.push(v);
                    Ok(v)
                };
            for i in 1..levels {
                let above = view(&mut m, i - 1, target.desc.format, vk::ImageUsageFlags::SAMPLED)?;
                let here = view(&mut m, i, twin, vk::ImageUsageFlags::STORAGE)?;
                let image_info = [vk::DescriptorImageInfo {
                    sampler: m.sampler,
                    image_view: above,
                    image_layout: vk::ImageLayout::GENERAL,
                }];
                let storage_info = [vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: here,
                    image_layout: vk::ImageLayout::GENERAL,
                }];
                let set = m.sets[i as usize - 1];
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&image_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&storage_info),
                ];
                unsafe { d.update_descriptor_sets(&writes, &[]) };
            }
            Ok(())
        })();
        match built {
            Ok(()) => Ok(m),
            Err(err) => {
                m.destroy(gpu);
                Err(err)
            }
        }
    }

    fn destroy(&mut self, gpu: &GpuDevice) {
        let d = &gpu.device;
        unsafe {
            for v in self.views.drain(..) {
                d.destroy_image_view(v, None);
            }
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_sampler(self.sampler, None);
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
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
    let barrier = |image: &OwnedImage,
                   old: vk::ImageLayout,
                   new: vk::ImageLayout,
                   src: (vk::PipelineStageFlags2, vk::AccessFlags2),
                   dst: (vk::PipelineStageFlags2, vk::AccessFlags2)| {
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
    let read = (
        vk::PipelineStageFlags2::ALL_COMMANDS,
        vk::AccessFlags2::SHADER_SAMPLED_READ,
    );
    let copy = (vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_WRITE);
    let copied = one_shot(gpu, |cmd| unsafe {
        let to_dst: Vec<_> = updates
            .iter()
            .filter(|(_, r)| !r.is_empty())
            .map(|(image, _)| {
                barrier(
                    image,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    read,
                    copy,
                )
            })
            .collect();
        gpu.device
            .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_dst));
        let mut offsets = offsets.iter();
        for (image, regions) in updates {
            let copies: Vec<vk::BufferImageCopy> = regions
                .iter()
                .map(|r| {
                    vk::BufferImageCopy::default()
                        .buffer_offset(*offsets.next().expect("an offset per region"))
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: r.mip,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D {
                            x: r.x as i32,
                            y: r.y as i32,
                            z: 0,
                        })
                        .image_extent(vk::Extent3D {
                            width: r.width,
                            height: r.height,
                            depth: 1,
                        })
                })
                .collect();
            if !copies.is_empty() {
                gpu.device.cmd_copy_buffer_to_image(
                    cmd,
                    source.buffer,
                    image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copies,
                );
            }
        }
        let to_read: Vec<_> = updates
            .iter()
            .filter(|(_, r)| !r.is_empty())
            .map(|(image, _)| {
                barrier(
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    copy,
                    read,
                )
            })
            .collect();
        gpu.device
            .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_read));
    });
    source.destroy(gpu);
    copied
}
