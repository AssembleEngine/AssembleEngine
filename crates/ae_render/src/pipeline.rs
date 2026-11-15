//! Pipeline helpers for the host passes: SPIR-V modules (compiled from
//! `shaders/` by `build.rs`), dynamic-rendering graphics pipelines without
//! vertex input (any number of colour attachments, optional depth), and compute pipelines.

use anyhow::{Context as _, Result};
use ash::vk;

/// SPIR-V compiled at build time (`OUT_DIR/<file>.spv`).
macro_rules! spirv {
    ($file:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $file, ".spv"))
    };
}
pub(crate) use spirv;

pub fn shader_module(device: &ash::Device, bytes: &[u8]) -> Result<vk::ShaderModule> {
    let code = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).context("reading SPIR-V")?;
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    Ok(unsafe { device.create_shader_module(&info, None) }?)
}

/// Depth test and write of a pipeline that renders with a depth attachment.
#[derive(Clone, Copy, Debug)]
pub struct DepthState {
    pub format: vk::Format,
    /// Write depth (`false` for a pass using the attachment read-only).
    pub write: bool,
    pub compare: vk::CompareOp,
}

pub struct GraphicsDesc<'a> {
    pub vertex: &'a [u8],
    pub fragment: &'a [u8],
    pub layout: vk::PipelineLayout,
    /// One format per colour attachment, in the pass's declaration order.
    pub color_formats: &'a [vk::Format],
    pub depth: Option<DepthState>,
    pub topology: vk::PrimitiveTopology,
    /// Blending on every colour attachment.
    pub blend: Blend,
}

/// How a pipeline's output blends with the attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blend {
    /// Overwrite.
    Off,
    /// `src × a + dst × (1 − a)`.
    Straight,
    /// `src + dst × (1 − a)`: the shader multiplies by alpha itself, so it
    /// can also add light (`a` 0 with a colour adds it), which straight
    /// alpha cannot do in a UNORM target (sources clamp to 1).
    Premultiplied,
}

pub fn graphics_pipeline(device: &ash::Device, desc: &GraphicsDesc<'_>) -> Result<vk::Pipeline> {
    let vertex = shader_module(device, desc.vertex)?;
    let fragment = shader_module(device, desc.fragment)?;
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(desc.topology);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let attachment = vec![vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(desc.blend != Blend::Off)
        .src_color_blend_factor(if desc.blend == Blend::Premultiplied { vk::BlendFactor::ONE } else { vk::BlendFactor::SRC_ALPHA })
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA);
        desc.color_formats.len()];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachment);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(desc.color_formats);
    let depth_stencil = match desc.depth {
        Some(depth) => {
            rendering = rendering.depth_attachment_format(depth.format);
            if ae_framegraph::has_stencil(depth.format) {
                rendering = rendering.stencil_attachment_format(depth.format);
            }
            vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(depth.write)
                .depth_compare_op(depth.compare)
        }
        None => vk::PipelineDepthStencilStateCreateInfo::default(),
    };
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(desc.layout)
        .push_next(&mut rendering);
    let result = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
    unsafe {
        device.destroy_shader_module(vertex, None);
        device.destroy_shader_module(fragment, None);
    }
    let pipelines = result.map_err(|(_, err)| err).context("vkCreateGraphicsPipelines")?;
    Ok(pipelines[0])
}

pub fn compute_pipeline(device: &ash::Device, code: &[u8], layout: vk::PipelineLayout) -> Result<vk::Pipeline> {
    let module = shader_module(device, code)?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
    let result = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
    unsafe { device.destroy_shader_module(module, None) };
    let pipelines = result.map_err(|(_, err)| err).context("vkCreateComputePipelines")?;
    Ok(pipelines[0])
}

pub fn pipeline_layout(
    device: &ash::Device,
    sets: &[vk::DescriptorSetLayout],
    push_bytes: u32,
    push_stages: vk::ShaderStageFlags,
) -> Result<vk::PipelineLayout> {
    let ranges = [vk::PushConstantRange::default()
        .stage_flags(push_stages)
        .offset(0)
        .size(push_bytes)];
    let mut info = vk::PipelineLayoutCreateInfo::default().set_layouts(sets);
    if push_bytes > 0 {
        info = info.push_constant_ranges(&ranges);
    }
    Ok(unsafe { device.create_pipeline_layout(&info, None) }?)
}

/// A descriptor set layout of `bindings` (type, stages), binding numbers in
/// order.
pub fn set_layout(
    device: &ash::Device,
    bindings: &[(vk::DescriptorType, vk::ShaderStageFlags)],
) -> Result<vk::DescriptorSetLayout> {
    let bindings: Vec<_> = bindings
        .iter()
        .enumerate()
        .map(|(index, (kind, stages))| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(index as u32)
                .descriptor_type(*kind)
                .descriptor_count(1)
                .stage_flags(*stages)
        })
        .collect();
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    Ok(unsafe { device.create_descriptor_set_layout(&info, None) }?)
}

pub fn push_bytes<T: Copy>(value: &T) -> &[u8] {
    // Safety: `T: Copy` plain-old-data push-constant blocks.
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>()) }
}

pub fn slice_bytes<T: Copy>(values: &[T]) -> &[u8] {
    // Safety: plain-old-data (integers and floats) only.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}
