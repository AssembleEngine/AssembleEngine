//! Owned images and buffers backed by `gpu-allocator` (one allocation each;
//! no suballocation of aliased resources, §1.6).

use crate::device::GpuDevice;
use crate::swapchain::color_range;
use ae_framegraph::{BufferDesc, ImageDesc};
use anyhow::{Result, anyhow};
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

pub struct OwnedImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub desc: ImageDesc,
    allocation: Option<Allocation>,
}

pub struct OwnedBuffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    allocation: Option<Allocation>,
}

fn subresource(desc: &ImageDesc) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: desc.aspect,
        level_count: desc.mip_levels,
        layer_count: desc.layers,
        ..color_range()
    }
}

impl OwnedImage {
    pub fn new(gpu: &GpuDevice, desc: ImageDesc, name: &str) -> Result<Self> {
        Self::with_view(gpu, desc, vk::ImageViewType::TYPE_2D, name)
    }

    /// With a view of `view_type` over every mip and layer (`TYPE_2D_ARRAY`
    /// for a `sampler2DArray`, even with one layer).
    pub fn with_view(gpu: &GpuDevice, desc: ImageDesc, view_type: vk::ImageViewType, name: &str) -> Result<Self> {
        Self::with_view_flags(gpu, desc, view_type, vk::ImageCreateFlags::empty(), name)
    }

    /// [`Self::with_view`] with image create `flags` (`MUTABLE_FORMAT` for
    /// a view of another format, as the mip generator's storage view of an
    /// sRGB image).
    pub fn with_view_flags(
        gpu: &GpuDevice,
        desc: ImageDesc,
        view_type: vk::ImageViewType,
        flags: vk::ImageCreateFlags,
        name: &str,
    ) -> Result<Self> {
        let info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(vk::ImageType::TYPE_2D)
            .format(desc.format)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: 1,
            })
            .mip_levels(desc.mip_levels)
            .array_layers(desc.layers)
            .samples(desc.samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { gpu.device.create_image(&info, None) }?;
        let requirements = unsafe { gpu.device.get_image_memory_requirements(image) };
        let allocation = gpu
            .allocator()
            .as_mut()
            .ok_or_else(|| anyhow!("allocator is gone"))?
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            });
        let allocation = match allocation {
            Ok(allocation) => allocation,
            Err(err) => {
                unsafe { gpu.device.destroy_image(image, None) };
                return Err(anyhow!("allocating {name}: {err}"));
            }
        };
        unsafe {
            gpu.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }?;
        // With EXTENDED_USAGE the image's usage may name one its own format
        // lacks (storage on sRGB, for a view of another format); the
        // whole-image view drops storage.
        let mut view_usage = vk::ImageViewUsageCreateInfo::default().usage(desc.usage & !vk::ImageUsageFlags::STORAGE);
        let mut view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(desc.format)
            .subresource_range(subresource(&desc));
        if flags.contains(vk::ImageCreateFlags::EXTENDED_USAGE) {
            view_info = view_info.push_next(&mut view_usage);
        }
        let view = unsafe { gpu.device.create_image_view(&view_info, None) }?;
        gpu.set_name(image, name);
        gpu.set_name(view, name);
        Ok(Self {
            image,
            view,
            desc,
            allocation: Some(allocation),
        })
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        unsafe {
            gpu.device.destroy_image_view(self.view, None);
            gpu.device.destroy_image(self.image, None);
        }
        if let (Some(allocation), Some(allocator)) = (self.allocation.take(), gpu.allocator().as_mut()) {
            let _ = allocator.free(allocation);
        }
    }

    pub fn range(&self) -> vk::ImageSubresourceRange {
        subresource(&self.desc)
    }
}

impl OwnedBuffer {
    /// `location`: `GpuOnly` for graph buffers, `CpuToGpu` for staging and
    /// instance rings, `GpuToCpu` for readback.
    pub fn new(gpu: &GpuDevice, desc: BufferDesc, location: MemoryLocation, name: &str) -> Result<Self> {
        let info = vk::BufferCreateInfo::default()
            .size(desc.size.max(4))
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { gpu.device.create_buffer(&info, None) }?;
        let requirements = unsafe { gpu.device.get_buffer_memory_requirements(buffer) };
        let allocation = gpu
            .allocator()
            .as_mut()
            .ok_or_else(|| anyhow!("allocator is gone"))?
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            });
        let allocation = match allocation {
            Ok(allocation) => allocation,
            Err(err) => {
                unsafe { gpu.device.destroy_buffer(buffer, None) };
                return Err(anyhow!("allocating {name}: {err}"));
            }
        };
        unsafe {
            gpu.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }?;
        gpu.set_name(buffer, name);
        Ok(Self {
            buffer,
            size: desc.size,
            allocation: Some(allocation),
        })
    }

    /// Host-visible bytes (`None` for device-local memory).
    pub fn mapped(&mut self) -> Option<&mut [u8]> {
        let size = self.size as usize;
        self.allocation
            .as_mut()
            .and_then(|allocation| allocation.mapped_slice_mut())
            .map(|slice| &mut slice[..size])
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        unsafe { gpu.device.destroy_buffer(self.buffer, None) };
        if let (Some(allocation), Some(allocator)) = (self.allocation.take(), gpu.allocator().as_mut()) {
            let _ = allocator.free(allocation);
        }
    }
}
