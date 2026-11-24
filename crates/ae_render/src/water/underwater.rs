//! Frame data and pipeline for the depth-reconstructed refraction image.

use super::WaterParams;
use crate::pipeline::{self, Blend, GraphicsDesc, spirv};
use anyhow::Result;
use ash::vk;
use glam::Mat4;

const METRES: usize = 128;

#[repr(C)]
#[derive(Clone, Copy)]
struct UnderwaterFrame {
    inverse_view_proj: [f32; 16],
    shallow: [f32; 4],
    deep: [f32; 4],
    submerged: [f32; 4],
    transmission: [[f32; 4]; METRES + 1],
}

pub(crate) const FRAME_BYTES: usize = size_of::<UnderwaterFrame>();

pub(crate) fn write(bytes: &mut [u8], view_proj: Mat4, p: WaterParams) {
    let frame = UnderwaterFrame {
        inverse_view_proj: view_proj.inverse().to_cols_array(),
        shallow: [p.shallow[0], p.shallow[1], p.shallow[2], p.caustics],
        deep: [p.deep[0], p.deep[1], p.deep[2], 0.0],
        submerged: [p.sub_tint, p.sub_keep, p.sub_tint_max, f32::from(p.outline)],
        transmission: std::array::from_fn(|metres| {
            let depth = metres as f32;
            let [r, g, b] = p.absorption.map(|k| (-depth * k).exp());
            let mean = p.absorption.iter().sum::<f32>() / 3.0;
            [r, g, b, 1.0 - (-depth * mean * 0.3).exp()]
        }),
    };
    bytes[..FRAME_BYTES].copy_from_slice(pipeline::push_bytes(&frame));
}

pub(crate) fn pipeline(device: &ash::Device, layout: vk::PipelineLayout, format: vk::Format) -> Result<vk::Pipeline> {
    pipeline_with_shader(device, layout, format, spirv!("water_underwater.frag"))
}

pub(crate) fn pipeline_with_shader(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    format: vk::Format,
    fragment: &[u8],
) -> Result<vk::Pipeline> {
    pipeline::graphics_pipeline(
        device,
        &GraphicsDesc {
            vertex: spirv!("fullscreen.vert"),
            fragment,
            layout,
            color_formats: &[format],
            depth: None,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            blend: Blend::Off,
            glow_blend: Blend::Off,
        },
    )
}
