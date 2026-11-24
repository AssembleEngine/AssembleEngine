//! The full-resolution water volume reads the completed HDR scene.

use super::underwater;
use crate::pipeline::spirv;
use anyhow::Result;
use ash::vk;

pub(crate) fn pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    format: vk::Format,
    waves: bool,
) -> Result<vk::Pipeline> {
    let fragment = if waves {
        spirv!("water_volume_waves.frag").as_slice()
    } else {
        spirv!("water_volume.frag").as_slice()
    };
    underwater::pipeline_with_shader(device, layout, format, fragment)
}
