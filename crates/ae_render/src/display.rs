//! What the display can do: SDL's HDR state for the window, and the
//! (format, colour space) pairs the Vulkan surface offers, with the device's
//! HDR extensions. Backs `ae-client --probe-display`, which renders nothing:
//! the HDR output path is chosen from its report rather than guessed, because
//! neither Vulkan nor SDL reports the display's primaries.
//!
//! The numbers here are the ones the output pass needs later: SDL's SDR white
//! level is in scRGB units (1.0 = 80 nits) and its headroom is a multiple of
//! SDR white, so paper white is `sdr_white_level × 80` nits and peak is
//! `paper_white × headroom`.

use crate::device::GpuInstance;
use crate::window::{SdlWindow, SurfaceSource as _};
use anyhow::Result;
use ash::vk;
use sdl3_sys::everything as sdl;
use std::ffi::CStr;
use std::time::{Duration, Instant};

/// How long the probe pumps the window's events before reading SDL's HDR
/// properties. On Wayland they arrive with the surface's preferred image
/// description after the window is mapped, not with `SDL_CreateWindow`.
const HDR_STATE_TIMEOUT: Duration = Duration::from_millis(700);

/// SDL's view of the window's display.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrState {
    /// HDR headroom above the SDR white point is available to this window.
    pub enabled: bool,
    /// SDR white in scRGB units (1.0 = 80 nits; 0 = not reported).
    pub sdr_white_level: f32,
    /// Extra range above SDR white, in SDR whites (1.0 = none).
    pub headroom: f32,
}

impl Default for HdrState {
    fn default() -> Self {
        Self {
            enabled: false,
            sdr_white_level: 1.0,
            headroom: 1.0,
        }
    }
}

impl HdrState {
    /// Paper white (diffuse white) in nits.
    pub fn paper_white_nits(&self) -> f32 {
        self.sdr_white_level * 80.0
    }

    /// The display's peak in nits.
    pub fn peak_nits(&self) -> f32 {
        self.paper_white_nits() * self.headroom
    }
}

/// Read SDL's HDR properties for the window (no event pumping).
pub fn hdr_state(window: &SdlWindow) -> HdrState {
    unsafe {
        let props = sdl::SDL_GetWindowProperties(window.raw());
        if props.0 == 0 {
            return HdrState::default();
        }
        HdrState {
            enabled: sdl::SDL_GetBooleanProperty(
                props,
                sdl::SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN,
                false,
            ),
            sdr_white_level: sdl::SDL_GetFloatProperty(
                props,
                sdl::SDL_PROP_WINDOW_SDR_WHITE_LEVEL_FLOAT,
                0.0,
            ),
            headroom: sdl::SDL_GetFloatProperty(
                props,
                sdl::SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT,
                1.0,
            ),
        }
    }
}

/// Pump events until the window's HDR state looks settled (or the timeout
/// passes) and return it. A window that is not HDR still reports SDR white
/// once the compositor has described its surface, hence the wait either way.
fn settled_hdr_state(window: &mut SdlWindow) -> HdrState {
    let started = Instant::now();
    let mut state = hdr_state(window);
    while started.elapsed() < HDR_STATE_TIMEOUT && state.sdr_white_level == 0.0 {
        window.pump();
        std::thread::sleep(Duration::from_millis(20));
        state = hdr_state(window);
    }
    state
}

/// What an HDR colour space means, for the report.
fn hdr_space(space: vk::ColorSpaceKHR) -> Option<&'static str> {
    match space {
        vk::ColorSpaceKHR::HDR10_ST2084_EXT => Some("HDR10, ST.2084 (PQ) in BT.2020"),
        vk::ColorSpaceKHR::HDR10_HLG_EXT => Some("HDR10, HLG in BT.2020"),
        vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT => Some("scRGB, extended linear sRGB"),
        vk::ColorSpaceKHR::BT2020_LINEAR_EXT => Some("linear BT.2020"),
        vk::ColorSpaceKHR::DISPLAY_P3_LINEAR_EXT => Some("linear Display-P3"),
        _ => None,
    }
}

fn describe_display(window: &SdlWindow) -> String {
    unsafe {
        let display = sdl::SDL_GetDisplayForWindow(window.raw());
        let name = sdl::SDL_GetDisplayName(display);
        let name = if name.is_null() {
            "?".into()
        } else {
            CStr::from_ptr(name).to_string_lossy()
        };
        let props = sdl::SDL_GetDisplayProperties(display);
        let hdr =
            sdl::SDL_GetBooleanProperty(props, sdl::SDL_PROP_DISPLAY_HDR_ENABLED_BOOLEAN, false);
        format!(
            "{name:?} (display {}): HDR {}",
            display.0,
            if hdr { "enabled" } else { "off" }
        )
    }
}

/// Print the window's display capabilities and the surface's format/colour
/// space pairs. Renders nothing and creates no device.
pub fn probe(window: &mut SdlWindow) -> Result<()> {
    let state = settled_hdr_state(window);
    println!("[display] {}", describe_display(window));
    println!(
        "[window] HDR {} | SDR white {} scRGB | headroom {}x",
        if state.enabled { "enabled" } else { "off" },
        state.sdr_white_level,
        state.headroom
    );
    if state.enabled && state.sdr_white_level > 0.0 {
        println!(
            "[window] paper white {:.1} nits, peak {:.1} nits",
            state.paper_white_nits(),
            state.peak_nits()
        );
    } else if state.enabled {
        println!("[window] SDL reported no SDR white level");
    }

    // The surface's own extensions, plus the colour-space one when the
    // loader has it (it only widens the list of pairs; nothing is created).
    let entry = unsafe { ash::Entry::load() }?;
    let available = unsafe { entry.enumerate_instance_extension_properties(None) }?;
    let colorspace = available.iter().any(|ext| {
        ext.extension_name_as_c_str().ok() == Some(ash::ext::swapchain_colorspace::NAME)
    });
    drop(entry);
    let mut extensions = window.instance_extensions()?;
    if colorspace {
        extensions.push(ash::ext::swapchain_colorspace::NAME.as_ptr());
    }
    println!(
        "[vulkan] VK_EXT_swapchain_colorspace: {}",
        if colorspace {
            "available, enabled"
        } else {
            "missing: only the baseline colour space is offered"
        }
    );

    let instance = GpuInstance::new(&extensions, None)?;
    let surface = window.create_surface(&instance.entry, &instance.instance)?;
    for physical in unsafe { instance.instance.enumerate_physical_devices() }? {
        let properties = unsafe { instance.instance.get_physical_device_properties(physical) };
        let name = properties
            .device_name_as_c_str()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let families = unsafe {
            instance
                .instance
                .get_physical_device_queue_family_properties(physical)
        };
        let presents = (0..families.len() as u32).any(|family| {
            unsafe {
                instance
                    .surface_loader
                    .get_physical_device_surface_support(physical, family, surface)
            }
            .unwrap_or(false)
        });
        if !presents {
            println!("[device] {name}: skipped, no queue family can present to the surface");
            continue;
        }
        let version = properties.api_version;
        let device_extensions = unsafe {
            instance
                .instance
                .enumerate_device_extension_properties(physical)
        }?;
        let hdr_metadata = device_extensions
            .iter()
            .any(|ext| ext.extension_name_as_c_str().ok() == Some(ash::ext::hdr_metadata::NAME));
        println!(
            "[device] {name} (Vulkan {}.{}.{}): VK_EXT_hdr_metadata {}",
            vk::api_version_major(version),
            vk::api_version_minor(version),
            vk::api_version_patch(version),
            if hdr_metadata { "present" } else { "missing" }
        );
        let formats = unsafe {
            instance
                .surface_loader
                .get_physical_device_surface_formats(physical, surface)
        }?;
        println!("[surface] {} (format, colour space) pair(s)", formats.len());
        for pair in &formats {
            let note = hdr_space(pair.color_space)
                .map(|note| format!("  <- {note}"))
                .unwrap_or_default();
            println!(
                "  {:<28} {:<32}{note}",
                format!("{:?}", pair.format),
                format!("{:?}", pair.color_space)
            );
        }
        // Per device, not merged: two devices need not offer the same pairs.
        let offered = |space: vk::ColorSpaceKHR| -> String {
            let mut names: Vec<String> = Vec::new();
            for pair in formats.iter().filter(|pair| pair.color_space == space) {
                let name = format!("{:?}", pair.format);
                if !names.contains(&name) {
                    names.push(name);
                }
            }
            if names.is_empty() {
                "not offered".to_string()
            } else {
                names.join(", ")
            }
        };
        println!(
            "[hdr] HDR10 (PQ): {}",
            offered(vk::ColorSpaceKHR::HDR10_ST2084_EXT)
        );
        println!(
            "[hdr] scRGB: {}",
            offered(vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT)
        );
    }
    instance.destroy_surface(surface);

    Ok(())
}
