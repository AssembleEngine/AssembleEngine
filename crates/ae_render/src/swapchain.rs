//! Swapchain: format/present-mode selection, acquire, present, recreate.
//! Owned by the render thread. Resize is a dirty flag (set by the window
//! event forwarded from the main thread, or by `OUT_OF_DATE`/`SUBOPTIMAL`);
//! a 0×0 extent (minimized) skips frames without recreating.

use crate::colorspace::Gamut;
use crate::device::GpuDevice;
use crate::encode::{DisplayLevels, OutputEncoding};
use anyhow::{Context as _, Result, bail};
use ash::vk;

/// How long `acquire` waits for an image before giving the frame up.
const ACQUIRE_TIMEOUT_NS: u64 = 100_000_000;

/// Result of `Swapchain::acquire`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Acquired {
    Image {
        index: u32,
        suboptimal: bool,
    },
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

/// What the client asks the swapchain to encode (`--hdr`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HdrRequest {
    /// An sRGB 8-bit swapchain: the SDR path, unchanged.
    #[default]
    Off,
    /// HDR whenever the surface offers it: HDR10 first, else scRGB, else
    /// SDR. SDL's window HDR state does not gate it (it only supplies the
    /// levels); a compositor with HDR off maps the signal to SDR itself.
    Auto,
    /// HDR10 (PQ) only: an error when the pair is not offered.
    Hdr10,
    /// scRGB only: an error when the pair is not offered.
    ScRgb,
}

impl HdrRequest {
    /// The request's command-line spelling.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::Hdr10 => "hdr10",
            Self::ScRgb => "scrgb",
        }
    }

    /// The HDR mode a toggle turns on (an explicit mode stays explicit; Off
    /// and Auto both turn on Auto).
    pub fn toggled_on(self) -> Self {
        match self {
            Self::Hdr10 => Self::Hdr10,
            Self::ScRgb => Self::ScRgb,
            Self::Off | Self::Auto => Self::Auto,
        }
    }
}

/// What the surface is asked for: a format and the colour space it stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetFormat {
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
}

impl TargetFormat {
    /// The encoding this pair carries.
    pub fn encoding(self) -> OutputEncoding {
        match self.color_space {
            vk::ColorSpaceKHR::HDR10_ST2084_EXT => OutputEncoding::Hdr10Pq,
            vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT => OutputEncoding::ScRgb,
            _ => OutputEncoding::Sdr,
        }
    }

    /// The pair as one line (the swapchain report and errors).
    pub fn describe(self) -> String {
        format!("{:?} + {:?}", self.format, self.color_space)
    }
}

/// The HDR10 pair (M13 stage 5): 10-bit, ST.2084 in BT.2020.
const HDR10: TargetFormat = TargetFormat {
    format: vk::Format::A2B10G10R10_UNORM_PACK32,
    color_space: vk::ColorSpaceKHR::HDR10_ST2084_EXT,
};
/// The scRGB pair: extended linear sRGB in half floats.
const SCRGB: TargetFormat = TargetFormat {
    format: vk::Format::R16G16B16A16_SFLOAT,
    color_space: vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
};

fn offered(formats: &[vk::SurfaceFormatKHR], want: TargetFormat) -> bool {
    formats
        .iter()
        .any(|f| f.format == want.format && f.color_space == want.color_space)
}

/// The surface's pairs, compactly, for an error that names what it does
/// offer.
fn offered_summary(formats: &[vk::SurfaceFormatKHR]) -> String {
    let mut spaces: Vec<String> = Vec::new();
    for f in formats {
        let name = format!("{:?}", f.color_space);
        if !spaces.contains(&name) {
            spaces.push(name);
        }
    }
    let mut format_names: Vec<String> = Vec::new();
    for f in formats {
        let name = format!("{:?}", f.format);
        if !format_names.contains(&name) {
            format_names.push(name);
        }
    }
    format!(
        "{} pair(s): colour spaces [{}] over formats [{}]",
        formats.len(),
        spaces.join(", "),
        format_names.join(", ")
    )
}

/// The static HDR10 metadata for an output gamut and display levels (what
/// `VK_EXT_hdr_metadata` carries): the primaries with D65 white, the
/// luminance range from 0 to the peak, and the content light levels. We do
/// not measure a frame average, so `maxFALL` is the conservative peak (no
/// frame can average above it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrMetadataValues {
    /// R, G, B primaries (`x`, `y`).
    pub primaries: [[f32; 2]; 3],
    pub white: [f32; 2],
    pub max_luminance: f32,
    pub min_luminance: f32,
    pub max_cll: f32,
    pub max_fall: f32,
}

/// [`HdrMetadataValues`] for `levels` and the output `gamut`.
pub fn hdr_metadata_values(levels: DisplayLevels, gamut: Gamut) -> HdrMetadataValues {
    HdrMetadataValues {
        primaries: gamut.primaries(),
        white: [0.3127, 0.3290],
        max_luminance: levels.peak,
        min_luminance: 0.0,
        max_cll: levels.peak,
        max_fall: levels.peak,
    }
}

/// Resolve a request against the surface's pairs. An explicit HDR mode that
/// is not offered is an error naming what is; Auto takes HDR10, then scRGB
/// (HDR10 first: half scRGB's bytes per pixel and the signal HDR displays
/// take natively), then SDR; Off is always SDR.
pub fn resolve_target(formats: &[vk::SurfaceFormatKHR], request: HdrRequest) -> Result<TargetFormat> {
    let sdr = || {
        let chosen = choose_format(formats);
        TargetFormat {
            format: chosen.format,
            color_space: chosen.color_space,
        }
    };
    match request {
        HdrRequest::Off => Ok(sdr()),
        HdrRequest::Auto if offered(formats, HDR10) => Ok(HDR10),
        HdrRequest::Auto if offered(formats, SCRGB) => Ok(SCRGB),
        HdrRequest::Auto => Ok(sdr()),
        other @ (HdrRequest::Hdr10 | HdrRequest::ScRgb) => {
            let want = if other == HdrRequest::Hdr10 { HDR10 } else { SCRGB };
            if offered(formats, want) {
                Ok(want)
            } else {
                bail!(
                    "--hdr {}: the surface does not offer {}; it offers {}",
                    if other == HdrRequest::Hdr10 { "hdr10" } else { "scrgb" },
                    want.describe(),
                    offered_summary(formats)
                )
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SwapchainOptions {
    pub present: PresentPolicy,
    pub hdr: HdrRequest,
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
    options: SwapchainOptions,
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
    pub fn new(gpu: &GpuDevice, surface: vk::SurfaceKHR, size: (u32, u32), options: SwapchainOptions) -> Result<Self> {
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
            policy: options.present,
            options,
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
                width: size
                    .0
                    .clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
                height: size.1.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            };
        }
        if extent.width == 0 || extent.height == 0 {
            return Ok(false);
        }
        let formats = unsafe { surface_loader.get_physical_device_surface_formats(gpu.physical, self.surface) }?;
        let target = resolve_target(&formats, self.options.hdr)?;
        let format = vk::SurfaceFormatKHR {
            format: target.format,
            color_space: target.color_space,
        };
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
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
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

    /// The pair this swapchain stores (its encoding).
    pub fn target(&self) -> TargetFormat {
        TargetFormat {
            format: self.format.format,
            color_space: self.format.color_space,
        }
    }

    /// Ask for a pair next recreate. The swapchain keeps presenting in the
    /// old one until the caller recreates it.
    pub fn set_options(&mut self, options: SwapchainOptions) {
        self.policy = options.present;
        self.options = options;
    }

    /// The requested present policy (not the mode the surface gave us).
    pub fn policy(&self) -> PresentPolicy {
        self.policy
    }

    /// Change the policy; the caller recreates the swapchain.
    pub fn set_policy(&mut self, policy: PresentPolicy) {
        self.policy = policy;
        self.options.present = policy;
    }

    /// `VK_EXT_hdr_metadata`: tell the compositor what the swapchain's
    /// signal means (M13 stage 5). The primaries are the output gamut's
    /// (D65 white), the luminance range is from 0 to the peak, and the
    /// content light levels are bounded by the peak (we do not measure a
    /// frame average, so the conservative `maxFALL = peak` is stated).
    pub fn set_hdr_metadata(&self, gpu: &GpuDevice, levels: DisplayLevels, gamut: Gamut) -> Result<()> {
        let Some(loader) = &gpu.hdr_metadata else {
            return Ok(());
        };
        if self.handle == vk::SwapchainKHR::null() {
            return Ok(());
        }
        let values = hdr_metadata_values(levels, gamut);
        let [red, green, blue] = values.primaries;
        let xy = |p: [f32; 2]| vk::XYColorEXT { x: p[0], y: p[1] };
        let metadata = vk::HdrMetadataEXT::default()
            .display_primary_red(xy(red))
            .display_primary_green(xy(green))
            .display_primary_blue(xy(blue))
            .white_point(xy(values.white))
            .max_luminance(values.max_luminance)
            .min_luminance(values.min_luminance)
            .max_content_light_level(values.max_cll)
            .max_frame_average_light_level(values.max_fall);
        unsafe { loader.set_hdr_metadata(&[self.handle], &[metadata]) };
        println!(
            "[render] vkSetHdrMetadataEXT: {} | {} primaries R {:.5}/{:.5} G {:.5}/{:.5} B {:.5}/{:.5}, white 0.31270/0.32900 | max {:.1} nits, min 0, maxCLL {:.1}, maxFALL {:.1}",
            self.target().describe(),
            gamut.tag(),
            red[0],
            red[1],
            green[0],
            green[1],
            blue[0],
            blue[1],
            values.max_luminance,
            values.max_cll,
            values.max_fall
        );
        Ok(())
    }

    /// Acquire the next image, waiting at most `ACQUIRE_TIMEOUT` so a stuck
    /// presentation engine cannot keep the render thread from stopping.
    pub fn acquire(&self, signal: vk::Semaphore) -> Result<Acquired> {
        match unsafe {
            self.loader
                .acquire_next_image(self.handle, ACQUIRE_TIMEOUT_NS, signal, vk::Fence::null())
        } {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs() -> Vec<vk::SurfaceFormatKHR> {
        vec![
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_SRGB,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::A2B10G10R10_UNORM_PACK32,
                color_space: vk::ColorSpaceKHR::HDR10_ST2084_EXT,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::R16G16B16A16_SFLOAT,
                color_space: vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
            },
        ]
    }

    #[test]
    fn the_metadata_states_the_output_gamut_and_the_peak() {
        let values = hdr_metadata_values(
            DisplayLevels {
                paper_white: 203.0,
                peak: 441.0,
            },
            Gamut::P3,
        );
        // Display P3's primaries and D65 white.
        assert_eq!(values.primaries, [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]]);
        assert_eq!(values.white, [0.3127, 0.3290]);
        assert_eq!(values.max_luminance, 441.0);
        assert_eq!(values.min_luminance, 0.0);
        // A conservative bound: no frame average can exceed the peak.
        assert_eq!(values.max_cll, 441.0);
        assert_eq!(values.max_fall, 441.0);
        // BT.2020 for scRGB's default output gamut.
        assert_eq!(
            hdr_metadata_values(DisplayLevels::default(), Gamut::Bt2020).primaries[0],
            [0.708, 0.292]
        );
    }

    #[test]
    fn the_three_targets_are_the_three_encodings() {
        let sdr = TargetFormat {
            format: vk::Format::B8G8R8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let hdr10 = TargetFormat {
            format: vk::Format::A2B10G10R10_UNORM_PACK32,
            color_space: vk::ColorSpaceKHR::HDR10_ST2084_EXT,
        };
        let scrgb = TargetFormat {
            format: vk::Format::R16G16B16A16_SFLOAT,
            color_space: vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
        };
        assert_eq!(sdr.encoding(), OutputEncoding::Sdr);
        assert_eq!(hdr10.encoding(), OutputEncoding::Hdr10Pq);
        assert_eq!(scrgb.encoding(), OutputEncoding::ScRgb);
        // The HDR target formats are the readback's (the same pass writes
        // both); SDR's capture is RGBA where the swapchain may be BGRA.
        assert_eq!(hdr10.format, OutputEncoding::Hdr10Pq.readback_format());
        assert_eq!(scrgb.format, OutputEncoding::ScRgb.readback_format());
        assert_eq!(sdr.format, vk::Format::B8G8R8A8_SRGB);
    }

    #[test]
    fn auto_prefers_hdr10_then_scrgb_then_sdr() {
        let all = pairs();
        // Both offered: HDR10.
        assert_eq!(resolve_target(&all, HdrRequest::Auto).unwrap(), HDR10);
        // No HDR10 pair: scRGB.
        let no_hdr10: Vec<_> = all
            .iter()
            .copied()
            .filter(|p| p.color_space != vk::ColorSpaceKHR::HDR10_ST2084_EXT)
            .collect();
        assert_eq!(resolve_target(&no_hdr10, HdrRequest::Auto).unwrap(), SCRGB);
        // Neither: SDR.
        let sdr_only: Vec<_> = all
            .iter()
            .copied()
            .filter(|p| p.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .collect();
        assert_eq!(
            resolve_target(&sdr_only, HdrRequest::Auto).unwrap().encoding(),
            OutputEncoding::Sdr
        );
        // Off is always SDR.
        assert_eq!(
            resolve_target(&all, HdrRequest::Off).unwrap().encoding(),
            OutputEncoding::Sdr
        );
    }

    #[test]
    fn an_explicit_mode_is_offered_or_an_error() {
        let all = pairs();
        assert_eq!(resolve_target(&all, HdrRequest::Hdr10).unwrap(), HDR10);
        assert_eq!(resolve_target(&all, HdrRequest::ScRgb).unwrap(), SCRGB);
        // An explicit mode the surface does not offer names what it does,
        // and never falls back silently.
        let sdr_only: Vec<_> = all
            .iter()
            .copied()
            .filter(|p| p.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .collect();
        let error = resolve_target(&sdr_only, HdrRequest::Hdr10).unwrap_err().to_string();
        assert!(error.contains("A2B10G10R10_UNORM_PACK32 + HDR10_ST2084_EXT"), "{error}");
        assert!(
            error.contains("SRGB_NONLINEAR") && error.contains("B8G8R8A8_SRGB"),
            "{error}"
        );
        let no_scrgb: Vec<_> = all
            .iter()
            .copied()
            .filter(|p| p.color_space != vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT)
            .collect();
        let error = resolve_target(&no_scrgb, HdrRequest::ScRgb).unwrap_err();
        assert!(error.to_string().contains("EXTENDED_SRGB_LINEAR_EXT"), "{error}");
    }
}
