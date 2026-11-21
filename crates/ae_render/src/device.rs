//! Vulkan instance and device (§3.7). Vulkan 1.3 core with
//! `dynamicRendering` and `synchronization2` required, `timelineSemaphore`
//! enabled for later. One graphics queue (that can present, when a surface
//! is given) serves graphics, compute and transfer passes.
//!
//! Validation is enumerated, never assumed: it is on by default in debug
//! builds when the Khronos layer is installed, opt-out/in with
//! `AE_VK_VALIDATION=0|1`, with synchronization validation
//! (`AE_VK_SYNC_VALIDATION`, default on with validation). Every
//! validation error and warning is counted (`ValidationCounters`) so smoke
//! runs can fail on them.

use anyhow::{Context as _, Result, anyhow, bail};
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::ffi::{CStr, c_char, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Validation messages seen by the debug messenger.
#[derive(Debug, Default)]
pub struct ValidationCounters {
    pub errors: AtomicU64,
    pub warnings: AtomicU64,
    printed: AtomicU64,
}

impl ValidationCounters {
    pub fn errors(&self) -> u64 {
        self.errors.load(Ordering::Acquire)
    }

    pub fn warnings(&self) -> u64 {
        self.warnings.load(Ordering::Acquire)
    }
}

const MAX_PRINTED_MESSAGES: u64 = 32;

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    // Safety: `user` is the `ValidationCounters` owned by `GpuInstance`,
    // which outlives the messenger.
    let counters = unsafe { &*(user as *const ValidationCounters) };
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        counters.errors.fetch_add(1, Ordering::AcqRel);
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        counters.warnings.fetch_add(1, Ordering::AcqRel);
    }
    let printed = counters.printed.fetch_add(1, Ordering::AcqRel);
    if printed < MAX_PRINTED_MESSAGES {
        let message = unsafe {
            data.as_ref()
                .filter(|data| !data.p_message.is_null())
                .map(|data| CStr::from_ptr(data.p_message).to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        eprintln!("[vulkan] {severity:?} {kind:?}: {message}");
    } else if printed == MAX_PRINTED_MESSAGES {
        eprintln!("[vulkan] further validation messages suppressed (counted)");
    }
    vk::FALSE
}

fn env_flag(key: &str, default: bool) -> bool {
    match std::env::var(key).as_deref().map(str::trim) {
        Ok("1") | Ok("true") | Ok("on") | Ok("yes") => true,
        Ok("0") | Ok("false") | Ok("off") | Ok("no") => false,
        _ => default,
    }
}

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Instance-level state; may be created on the main thread (surface
/// creation) and shared with the render thread.
pub struct GpuInstance {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub validation: bool,
    /// `VK_EXT_debug_utils` is enabled (object names, labels).
    pub debug_utils_enabled: bool,
    pub counters: Arc<ValidationCounters>,
    debug_utils: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub surface_loader: ash::khr::surface::Instance,
}

impl GpuInstance {
    /// `extensions`: what the surface needs (empty for headless).
    /// `validation`: `None` reads `AE_VK_VALIDATION` (default: on in
    /// debug builds).
    pub fn new(extensions: &[*const c_char], validation: Option<bool>) -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.context("loading the Vulkan library")?;
        let version = unsafe { entry.try_enumerate_instance_version() }?.unwrap_or(vk::API_VERSION_1_0);
        if version < vk::API_VERSION_1_3 {
            bail!(
                "Vulkan 1.3 instance required, loader reports {}.{}",
                vk::api_version_major(version),
                vk::api_version_minor(version)
            );
        }
        let layers = unsafe { entry.enumerate_instance_layer_properties() }?;
        let has_validation = layers
            .iter()
            .any(|layer| layer.layer_name_as_c_str().ok() == Some(VALIDATION_LAYER));
        let want_validation =
            validation.unwrap_or_else(|| env_flag("AE_VK_VALIDATION", cfg!(debug_assertions)));
        if want_validation && !has_validation {
            eprintln!("[vulkan] validation requested but VK_LAYER_KHRONOS_validation is not installed");
        }
        let validation = want_validation && has_validation;
        // Synchronization validation is enabled through VK_EXT_validation_features,
        // which the validation layer provides; enable it explicitly.
        let has_validation_features = validation
            && unsafe { entry.enumerate_instance_extension_properties(Some(VALIDATION_LAYER)) }
                .unwrap_or_default()
                .iter()
                .any(|ext| ext.extension_name_as_c_str().ok() == Some(ash::ext::validation_features::NAME));
        let want_sync = validation && env_flag("AE_VK_SYNC_VALIDATION", true);
        if want_sync && !has_validation_features {
            eprintln!("[vulkan] synchronization validation unavailable: the layer lacks VK_EXT_validation_features");
        }
        let sync_validation = want_sync && has_validation_features;

        let available = unsafe { entry.enumerate_instance_extension_properties(None) }?;
        let has_debug_utils = available
            .iter()
            .any(|ext| ext.extension_name_as_c_str().ok() == Some(ash::ext::debug_utils::NAME));

        let mut extension_names: Vec<*const c_char> = extensions.to_vec();
        if has_debug_utils {
            extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        if sync_validation {
            extension_names.push(ash::ext::validation_features::NAME.as_ptr());
        }
        let layer_names: Vec<*const c_char> = if validation {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        };
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"AssembleEngine")
            .engine_name(c"AssembleEngine")
            .api_version(vk::API_VERSION_1_3);
        let counters = Arc::new(ValidationCounters::default());
        let mut messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback))
            .user_data(Arc::as_ptr(&counters) as *mut c_void);
        let enabled_features = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
        let mut validation_features = vk::ValidationFeaturesEXT::default().enabled_validation_features(&enabled_features);
        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&extension_names);
        if validation && has_debug_utils {
            create_info = create_info.push_next(&mut messenger_info);
        }
        if sync_validation {
            create_info = create_info.push_next(&mut validation_features);
        }
        let instance = unsafe { entry.create_instance(&create_info, None) }.context("vkCreateInstance")?;
        let debug_utils = if validation && has_debug_utils {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT {
                p_next: std::ptr::null(),
                ..messenger_info
            };
            let messenger = unsafe { loader.create_debug_utils_messenger(&messenger_info, None) }?;
            Some((loader, messenger))
        } else {
            None
        };
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        Ok(Self {
            entry,
            instance,
            validation,
            debug_utils_enabled: has_debug_utils,
            counters,
            debug_utils,
            surface_loader,
        })
    }

    pub fn destroy_surface(&self, surface: vk::SurfaceKHR) {
        if surface != vk::SurfaceKHR::null() {
            unsafe { self.surface_loader.destroy_surface(surface, None) };
        }
    }
}

impl Drop for GpuInstance {
    fn drop(&mut self) {
        unsafe {
            if let Some((loader, messenger)) = self.debug_utils.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// Device-level state, owned by the render thread.
pub struct GpuDevice {
    pub instance: Arc<GpuInstance>,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family: u32,
    pub queue: vk::Queue,
    pub allocator: Mutex<Option<Allocator>>,
    pub debug_utils: Option<ash::ext::debug_utils::Device>,
    /// Nanoseconds per timestamp tick (0 if timestamps are unsupported).
    pub timestamp_period: f32,
    pub device_name: String,
    /// `fillModeNonSolid` is enabled (line-mode wireframe pipelines).
    pub fill_mode_non_solid: bool,
    /// `fragmentStoresAndAtomics` is enabled (the particles' exact
    /// per-pixel reference, tests only).
    pub fragment_stores: bool,
    /// `textureCompressionBC` is enabled (BC1–BC7 model textures).
    pub texture_compression_bc: bool,
    /// A compute pipeline can pin 32-lane, full subgroups
    /// (`subgroupSizeControl` and `computeFullSubgroups` are enabled, 32 is in
    /// the size range for compute, and eight subgroups fit a workgroup), as
    /// the tile-binned particle blend needs
    /// ([`crate::pipeline::compute_pipeline_subgroup32`]).
    pub compute_subgroups_32: bool,
    /// `multiDrawIndirect`, `shaderDrawParameters` and
    /// `descriptorBindingPartiallyBound` are enabled: the units' models
    /// (one multi-draw per pipeline, per-draw records by `gl_DrawID`, a
    /// partially bound texture array).
    pub gpu_driven: bool,
    /// A frame fence timed out: the GPU may still be executing, so waiting
    /// for idle could block forever and destroying objects is unsafe.
    hung: AtomicBool,
}

impl GpuDevice {
    /// Pick a Vulkan 1.3 device with a graphics queue (that presents to
    /// `surface`, if given; compute support preferred).
    pub fn new(instance: Arc<GpuInstance>, surface: Option<vk::SurfaceKHR>) -> Result<Self> {
        let vk_instance = &instance.instance;
        let physicals = unsafe { vk_instance.enumerate_physical_devices() }?;
        let mut candidates = Vec::new();
        for physical in physicals {
            let properties = unsafe { vk_instance.get_physical_device_properties(physical) };
            if properties.api_version < vk::API_VERSION_1_3 {
                continue;
            }
            let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut features12)
                .push_next(&mut features13);
            unsafe { vk_instance.get_physical_device_features2(physical, &mut features) };
            let base = features.features;
            if features13.dynamic_rendering == vk::FALSE
                || base.shader_clip_distance == vk::FALSE
                || base.independent_blend == vk::FALSE
                || base.shader_storage_image_extended_formats == vk::FALSE
                || base.shader_storage_image_write_without_format == vk::FALSE
                || features13.synchronization2 == vk::FALSE
                || features13.shader_demote_to_helper_invocation == vk::FALSE
                || features12.timeline_semaphore == vk::FALSE
            {
                continue;
            }
            let families = unsafe { vk_instance.get_physical_device_queue_family_properties(physical) };
            let mut chosen: Option<u32> = None;
            // One queue runs every pass: it must do graphics and compute.
            for (index, family) in families.iter().enumerate() {
                if !family.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE) {
                    continue;
                }
                if let Some(surface) = surface {
                    let present = unsafe {
                        instance
                            .surface_loader
                            .get_physical_device_surface_support(physical, index as u32, surface)
                    }
                    .unwrap_or(false);
                    if !present {
                        continue;
                    }
                }
                chosen = Some(index as u32);
                break;
            }
            let Some(family) = chosen else { continue };
            let rank = match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 2,
            };
            let mut features11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut features12b = vk::PhysicalDeviceVulkan12Features::default();
            let mut more = vk::PhysicalDeviceFeatures2::default().push_next(&mut features11).push_next(&mut features12b);
            unsafe { vk_instance.get_physical_device_features2(physical, &mut more) };
            let gpu_driven = base.multi_draw_indirect == vk::TRUE && features11.shader_draw_parameters == vk::TRUE && features12b.descriptor_binding_partially_bound == vk::TRUE;
            let optional = (base.fill_mode_non_solid == vk::TRUE, base.fragment_stores_and_atomics == vk::TRUE, base.texture_compression_bc == vk::TRUE, gpu_driven);
            candidates.push((rank, physical, family, properties, optional));
        }
        candidates.sort_by_key(|(rank, ..)| *rank);
        let (_, physical, queue_family, properties, (fill_mode_non_solid, fragment_stores, texture_compression_bc, gpu_driven)) = candidates
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no Vulkan 1.3 device with dynamic rendering, synchronization2, clip distances, storage images of any format and a graphics+compute queue"))?;
        let device_name = properties
            .device_name_as_c_str()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        // Pinned 32-lane subgroups (the packets' particle blend); the
        // features are core in 1.3 but optional.
        let compute_subgroups_32 = {
            let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut features13);
            unsafe { vk_instance.get_physical_device_features2(physical, &mut features) };
            let mut control = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut control).push_next(&mut subgroup);
            unsafe { vk_instance.get_physical_device_properties2(physical, &mut properties2) };
            features13.subgroup_size_control == vk::TRUE
                && features13.compute_full_subgroups == vk::TRUE
                && (control.min_subgroup_size..=control.max_subgroup_size).contains(&32)
                && control.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE)
                && control.max_compute_workgroup_subgroups >= 8
                && subgroup.supported_stages.contains(vk::ShaderStageFlags::COMPUTE)
                && subgroup.supported_operations.contains(vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::BALLOT)
        };
        let extensions: Vec<*const c_char> = if surface.is_some() {
            vec![ash::khr::swapchain::NAME.as_ptr()]
        } else {
            Vec::new()
        };
        // `discard` compiles to OpDemoteToHelperInvocation for Vulkan 1.3
        // targets (a required 1.3 feature, but it must be enabled).
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .shader_demote_to_helper_invocation(true)
            .subgroup_size_control(compute_subgroups_32)
            .compute_full_subgroups(compute_subgroups_32);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true).descriptor_binding_partially_bound(gpu_driven);
        let mut features11 = vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(gpu_driven);
        // Clip distances: the terrain clips its power-of-two LEB square to the map.
        let base_features = vk::PhysicalDeviceFeatures::default()
            .shader_clip_distance(true)
            .independent_blend(true)
            // The mip generator (shaders/mipgen.comp) writes R8, RG8 and
            // RGBA8 images through one storage image without a format.
            .shader_storage_image_extended_formats(true)
            .shader_storage_image_write_without_format(true)
            .fill_mode_non_solid(fill_mode_non_solid)
            .fragment_stores_and_atomics(fragment_stores)
            .texture_compression_bc(texture_compression_bc)
            .multi_draw_indirect(gpu_driven);
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&extensions)
            .enabled_features(&base_features)
            .push_next(&mut features11)
            .push_next(&mut features12)
            .push_next(&mut features13);
        let device = unsafe { vk_instance.create_device(physical, &create_info, None) }.context("vkCreateDevice")?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: vk_instance.clone(),
            device: device.clone(),
            physical_device: physical,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(|err| anyhow!("gpu-allocator: {err}"));
        let allocator = match allocator {
            Ok(allocator) => allocator,
            Err(err) => {
                unsafe { device.destroy_device(None) };
                return Err(err);
            }
        };
        let debug_utils = instance
            .debug_utils_enabled
            .then(|| ash::ext::debug_utils::Device::new(vk_instance, &device));
        let timestamp_period = if properties.limits.timestamp_compute_and_graphics == vk::TRUE {
            properties.limits.timestamp_period
        } else {
            0.0
        };
        Ok(Self {
            instance,
            physical,
            device,
            queue_family,
            queue,
            allocator: Mutex::new(Some(allocator)),
            debug_utils,
            timestamp_period,
            device_name,
            fill_mode_non_solid,
            fragment_stores,
            texture_compression_bc,
            compute_subgroups_32,
            gpu_driven,
            hung: AtomicBool::new(false),
        })
    }

    pub fn allocator(&self) -> std::sync::MutexGuard<'_, Option<Allocator>> {
        self.allocator.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Name a Vulkan object for validation messages and captures.
    pub fn set_name<H: vk::Handle>(&self, handle: H, name: &str) {
        let Some(debug_utils) = &self.debug_utils else { return };
        let Ok(name) = std::ffi::CString::new(name) else { return };
        let info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(handle)
            .object_name(&name);
        unsafe {
            let _ = debug_utils.set_debug_utils_object_name(&info);
        }
    }

    pub fn begin_label(&self, cmd: vk::CommandBuffer, name: &str) {
        let Some(debug_utils) = &self.debug_utils else { return };
        let Ok(name) = std::ffi::CString::new(name) else { return };
        let label = vk::DebugUtilsLabelEXT::default()
            .label_name(&name)
            .color([0.4, 0.6, 0.9, 1.0]);
        unsafe { debug_utils.cmd_begin_debug_utils_label(cmd, &label) };
    }

    pub fn end_label(&self, cmd: vk::CommandBuffer) {
        let Some(debug_utils) = &self.debug_utils else { return };
        unsafe { debug_utils.cmd_end_debug_utils_label(cmd) };
    }

    /// Wait for the device to go idle; a no-op once the device is marked
    /// hung (that wait has no timeout).
    pub fn wait_idle(&self) {
        if self.is_hung() {
            return;
        }
        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }

    pub fn mark_hung(&self) {
        self.hung.store(true, Ordering::Release);
    }

    pub fn is_hung(&self) -> bool {
        self.hung.load(Ordering::Acquire)
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        self.wait_idle();
        // The allocator must free its memory blocks before the device goes.
        drop(self.allocator().take());
        unsafe { self.device.destroy_device(None) };
    }
}

/// Turn a Vulkan error into an error report. Device loss is fatal: there is
/// no recovery path (the render thread exits and the run reports it).
pub fn fatal(err: vk::Result) -> anyhow::Error {
    if err == vk::Result::ERROR_DEVICE_LOST {
        anyhow!("Vulkan device lost (fatal: device-loss recovery is not implemented)")
    } else {
        anyhow!("{err:?}")
    }
}
