//! The water-plane reflection uses real world positions and a mirrored camera.
use crate::executor::{BufferBinding, PassContext};
use crate::terrain::renderer::frustum_planes;
use anyhow::{Context, Result};
use ash::vk;
use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct MirrorFrame {
    view_proj: [f32; 16],
    eye: [f32; 4],
    planes: [[f32; 4]; 6],
}

impl MirrorFrame {
    pub fn new(view_proj: Mat4, eye: Vec3, level: f32) -> Self {
        let reflect = Mat4::from_scale_rotation_translation(
            Vec3::new(1.0, -1.0, 1.0),
            glam::Quat::IDENTITY,
            Vec3::Y * (2.0 * level),
        );
        let view_proj = view_proj * reflect;
        Self {
            view_proj: view_proj.to_cols_array(),
            eye: reflect.transform_point3(eye).extend(1.0).to_array(),
            planes: frustum_planes(view_proj),
        }
    }

    pub fn write(self, bytes: &mut [u8], offset: usize) {
        bytes[offset..offset + size_of::<Self>()].copy_from_slice(crate::pipeline::push_bytes(&self));
    }
}

/// The reflection targets' extent for `divisor` (2 is the default half
/// resolution, 1 a full-resolution comparison; M16.21).
pub(crate) fn scaled_extent(extent: vk::Extent2D, divisor: u32) -> vk::Extent2D {
    let divisor = divisor.max(1);
    vk::Extent2D {
        width: extent.width.div_ceil(divisor),
        height: extent.height.div_ceil(divisor),
    }
}

pub(crate) fn push_buffers(
    push: &mut crate::bindless::Push,
    list: BufferBinding,
    indirect: BufferBinding,
) -> Result<()> {
    push.words[2] = list.storage.context("reflection list storage handle")?;
    push.words[3] = indirect.storage.context("reflection indirect storage handle")?;
    Ok(())
}

/// Order the steps within the mirrored CBT cull. The graph orders the
/// passes around the cull, including the main CBT update it reads.
pub(crate) fn cull_barrier(ctx: &PassContext<'_>) {
    let memory = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::DRAW_INDIRECT)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::INDIRECT_COMMAND_READ)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::DRAW_INDIRECT)
        .dst_access_mask(
            vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_STORAGE_WRITE
                | vk::AccessFlags2::INDIRECT_COMMAND_READ,
        )];
    unsafe {
        ctx.device
            .cmd_pipeline_barrier2(ctx.cmd, &vk::DependencyInfo::default().memory_barriers(&memory));
    }
}

/// On error destroy the pipelines compiled before the failing one.
pub(crate) fn pipelines<const N: usize>(
    device: &ash::Device,
    mut create: impl FnMut(usize) -> Result<vk::Pipeline>,
) -> Result<[vk::Pipeline; N]> {
    let mut pipelines = [vk::Pipeline::null(); N];
    for i in 0..N {
        match create(i) {
            Ok(pipeline) => pipelines[i] = pipeline,
            Err(error) => {
                for pipeline in &pipelines[..i] {
                    unsafe { device.destroy_pipeline(*pipeline, None) };
                }
                return Err(error);
            }
        }
    }
    Ok(pipelines)
}
