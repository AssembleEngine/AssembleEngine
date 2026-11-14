//! Swapchain: format/present-mode selection, acquire, present, recreate.
//! Owned by the render thread. Resize is a dirty flag (set by the window
//! event forwarded from the main thread, or by `OUT_OF_DATE`/`SUBOPTIMAL`);
//! a 0×0 extent (minimized) skips frames without recreating.

use crate::device::GpuDevice;
use anyhow::{Context as _, Result};
use ash::vk;

/// How long `acquire` waits for an image before giving the frame up.
const ACQUIRE_TIMEOUT_NS: u64 = 100_000_000;

/// Result of `Swapchain::acquire`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Acquired {
    Image { index: u32, suboptimal: bool },
    /// The swapchain must be recreated.
    OutOfDate,
    /// No image within the timeout (nothing was signalled): skip the frame.
    TimedOut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentPolicy {
    /// MAILBOX when available, else FIFO.
    Mailbox,
    Fifo,
}

pub struct Swapchain {
    pub loader: ash::khr::swapchain::Device,
    pub surface: vk::SurfaceKHR,
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
    /// Signalled by the present-ready submit, one per image.
    pub render_finished: Vec<vk::Semaphore>,
    /// Bumped on every (re)creation.
    pub generation: u64,
    policy: PresentPolicy,
}

/// Prefer an sRGB BGRA/RGBA format with the sRGB non-linear colour space,
/// then UNORM, then whatever comes first (the baseline took `[0]` blindly).
fn choose_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    let preferred = [
        vk::Format::B8G8R8A8_SRGB,
        vk::Format::R8G8B8A8_SRGB,
        vk::Format::B8G8R8A8_UNORM,
        vk::Format::R8G8B8A8_UNORM,
    ];
    for format in preferred {
        if let Some(found) = formats
            .iter()
            .find(|f| f.format == format && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        {
            return *found;
        }
    }
    formats[0]
}

impl Swapchain {
    pub fn new(gpu: &GpuDevice, surface: vk::SurfaceKHR, size: (u32, u32), policy: PresentPolicy) -> Result<Self> {
        let loader = ash::khr::swapchain::Device::new(&gpu.instance.instance, &gpu.device);
        let mut swapchain = Self {
            loader,
            surface,
            handle: vk::SwapchainKHR::null(),
            images: Vec::new(),
            views: Vec::new(),
            format: vk::SurfaceFormatKHR::default(),
            extent: vk::Extent2D::default(),
            present_mode: vk::PresentModeKHR::FIFO,
            render_finished: Vec::new(),
            generation: 0,
            policy,
        };
        swapchain.recreate(gpu, size)?;
        Ok(swapchain)
    }

    /// (Re)create for the window's pixel size. Returns `false` (and keeps no
    /// swapchain images) while the extent is zero.
    pub fn recreate(&mut self, gpu: &GpuDevice, size: (u32, u32)) -> Result<bool> {
        gpu.wait_idle();
        let surface_loader = &gpu.instance.surface_loader;
        let capabilities =
            unsafe { surface_loader.get_physical_device_surface_capabilities(gpu.physical, self.surface) }?;
        let mut extent = capabilities.current_extent;
        if extent.width == u32::MAX {
            extent = vk::Extent2D {
                width: size.0.clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
                height: size.1.clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height),
            };
        }
        if extent.width == 0 || extent.height == 0 {
            return Ok(false);
        }
        let formats = unsafe { surface_loader.get_physical_device_surface_formats(gpu.physical, self.surface) }?;
        let format = choose_format(&formats);
        let modes = unsafe { surface_loader.get_physical_device_surface_present_modes(gpu.physical, self.surface) }?;
        let present_mode = match self.policy {
            PresentPolicy::Mailbox if modes.contains(&vk::PresentModeKHR::MAILBOX) => vk::PresentModeKHR::MAILBOX,
            _ => vk::PresentModeKHR::FIFO,
        };
        let mut image_count = capabilities.min_image_count.max(3);
        if capabilities.max_image_count > 0 {
            image_count = image_count.min(capabilities.max_image_count);
        }
        let old = self.handle;
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old);
        let handle = unsafe { self.loader.create_swapchain(&info, None) }.context("vkCreateSwapchainKHR")?;
        self.destroy_images(gpu);
        if old != vk::SwapchainKHR::null() {
            unsafe { self.loader.destroy_swapchain(old, None) };
        }
        self.handle = handle;
        self.images = unsafe { self.loader.get_swapchain_images(handle) }?;
        for (index, &image) in self.images.iter().enumerate() {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(color_range());
            let view = unsafe { gpu.device.create_image_view(&view_info, None) }?;
            gpu.set_name(image, &format!("swapchain image {index}"));
            self.views.push(view);
            let semaphore = unsafe { gpu.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
            self.render_finished.push(semaphore);
        }
        self.format = format;
        self.extent = extent;
        self.present_mode = present_mode;
        self.generation += 1;
        Ok(true)
    }

    fn destroy_images(&mut self, gpu: &GpuDevice) {
        unsafe {
            for view in self.views.drain(..) {
                gpu.device.destroy_image_view(view, None);
            }
            for semaphore in self.render_finished.drain(..) {
                gpu.device.destroy_semaphore(semaphore, None);
            }
        }
        self.images.clear();
    }

    pub fn ready(&self) -> bool {
        self.handle != vk::SwapchainKHR::null() && !self.images.is_empty()
    }

    /// Acquire the next image, waiting at most `ACQUIRE_TIMEOUT` so a stuck
    /// presentation engine cannot keep the render thread from stopping.
    pub fn acquire(&self, signal: vk::Semaphore) -> Result<Acquired> {
        match unsafe { self.loader.acquire_next_image(self.handle, ACQUIRE_TIMEOUT_NS, signal, vk::Fence::null()) } {
            Ok((index, suboptimal)) => Ok(Acquired::Image { index, suboptimal }),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(Acquired::OutOfDate),
            Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => Ok(Acquired::TimedOut),
            Err(err) => Err(crate::device::fatal(err)).context("vkAcquireNextImageKHR"),
        }
    }

    /// `Ok(false)` when the swapchain must be recreated.
    pub fn present(&self, queue: vk::Queue, image: u32) -> Result<bool> {
        let wait = [self.render_finished[image as usize]];
        let swapchains = [self.handle];
        let indices = [image];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait)
            .swapchains(&swapchains)
            .image_indices(&indices);
        match unsafe { self.loader.queue_present(queue, &info) } {
            Ok(suboptimal) => Ok(!suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(false),
            Err(err) => Err(crate::device::fatal(err)).context("vkQueuePresentKHR"),
        }
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        gpu.wait_idle();
        self.destroy_images(gpu);
        if self.handle != vk::SwapchainKHR::null() {
            unsafe { self.loader.destroy_swapchain(self.handle, None) };
            self.handle = vk::SwapchainKHR::null();
        }
    }
}

pub fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}
