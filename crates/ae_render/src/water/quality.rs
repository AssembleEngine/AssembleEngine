//! The rich tiers' Vulkan image requirements, checked before declaring a graph.

use crate::device::GpuDevice;
use crate::library::{LibraryEnv, LibraryResource as R, WaterTiers, image_desc};
use crate::upload::{MIP_STORAGE_FEATURES, mipped_image_spec};
use ae_framegraph::ImageDesc;
use ash::vk;

struct Requirement {
    name: &'static str,
    image: ImageDesc,
    flags: vk::ImageCreateFlags,
    view_features: vk::FormatFeatureFlags2,
    mip_storage: Option<vk::Format>,
}

fn unsupported(request: WaterTiers, env: &LibraryEnv, supports: impl Fn(&Requirement) -> bool) -> Option<Requirement> {
    let mut requirements = Vec::new();
    let linear = vk::FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR;
    let blend = vk::FormatFeatureFlags2::COLOR_ATTACHMENT_BLEND;
    let mut add = |resource: R, view_features| {
        requirements.push(Requirement {
            name: resource.name(),
            image: image_desc(resource, env).expect("rich image in a wet scene"),
            flags: vk::ImageCreateFlags::empty(),
            view_features,
            mip_storage: None,
        });
    };
    if request.reflect {
        add(R::WaterReflection, linear);
        add(R::WaterReflectionDepth, vk::FormatFeatureFlags2::empty());
    }
    if request.refract || request.volume {
        add(R::WaterRefraction, linear | blend);
    }
    if request.waves {
        add(R::WaterRaw, blend);
        add(R::WaterChurn, blend);
        add(R::WaterWaves, linear);
    }
    if request.volume {
        add(R::WaterVolumeOut, linear);
    }
    if request.refract || request.volume {
        for (name, format, size, layers) in [
            (
                R::WaterCaustics.name(),
                super::caustics::FORMAT,
                super::caustics::SIZE,
                1,
            ),
            (
                "seabed layers",
                vk::Format::R8G8B8A8_SRGB,
                crate::terrain::tileset::SEABED_SIZE,
                3,
            ),
        ] {
            let (image, flags, storage) = mipped_image_spec(format, size, size, layers);
            requirements.push(Requirement {
                name,
                image,
                flags,
                view_features: linear,
                mip_storage: Some(storage),
            });
        }
    }
    requirements.into_iter().find(|r| !supports(r))
}

pub(crate) fn unsupported_image(request: WaterTiers, env: &LibraryEnv, gpu: &GpuDevice) -> Option<String> {
    unsupported(request, env, |r| {
        let d = r.image;
        let properties = unsafe {
            gpu.instance.instance.get_physical_device_image_format_properties(
                gpu.physical,
                d.format,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::OPTIMAL,
                d.usage,
                r.flags,
            )
        };
        properties.is_ok_and(|p| {
            d.width <= p.max_extent.width
                && d.height <= p.max_extent.height
                && d.mip_levels <= p.max_mip_levels
                && d.layers <= p.max_array_layers
                && p.sample_counts.contains(d.samples)
        }) && gpu.optimal_format_features(d.format).contains(r.view_features)
            && r.mip_storage
                .is_none_or(|format| gpu.optimal_format_features(format).contains(MIP_STORAGE_FEATURES))
    })
    .map(|r| format!("{} ({:?}, {:?})", r.name, r.image.format, r.image.usage))
}
