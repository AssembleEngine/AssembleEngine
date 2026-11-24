//! Water uses the terrain's conforming CBT and longest-edge bisection.
//! Its tree is independent: wave error has a different distribution than
//! ground error, and water tracks the live camera even with terrain LOD frozen.
use super::waves::WavePatch;
use crate::bindless::{Bindless, GroupId, HandleTable, PUSH_STAGES, Push};
use crate::device::GpuDevice;
use crate::executor::{BufferBinding, PassContext};
use crate::memory::OwnedBuffer;
use crate::pipeline::{compute_pipeline, push_bytes, slice_bytes, spirv};
use crate::terrain::leb::Cbt;
use crate::upload;
use ae_framegraph::BufferDesc;
use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;

const DEPTH: u32 = 16;
const TREE_DEPTH: u32 = DEPTH - 5;
const LEAVES: u32 = 1 << DEPTH;

pub(crate) fn leaves_desc() -> BufferDesc {
    BufferDesc {
        size: u64::from(LEAVES) * 32,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
    }
}

pub(crate) fn indirect_desc() -> BufferDesc {
    BufferDesc {
        size: 9 * 4,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
    }
}

pub(crate) struct WaterMesh {
    scratch: [OwnedBuffer; 3],
    table: HandleTable,
    group: GroupId,
    prepare: vk::Pipeline,
    update: vk::Pipeline,
    reduce: vk::Pipeline,
    layout: vk::PipelineLayout,
}

impl WaterMesh {
    pub fn new(gpu: &GpuDevice, bindless: &mut Bindless, frame: u32) -> Result<Self> {
        let layout = bindless.pipeline_layout();
        let [prepare, update, reduce] = super::reflect::pipelines(&gpu.device, |i| {
            let code = match i {
                0 => spirv!("cbt_prepare_water.comp").as_slice(),
                1 => spirv!("water_lod_error.comp").as_slice(),
                _ => spirv!("cbt_reduce_water.comp").as_slice(),
            };
            compute_pipeline(&gpu.device, code, layout)
        })?;
        let mut scratch = Vec::with_capacity(3);
        let result = (|| {
            let cbt = Cbt::new(DEPTH);
            let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
            scratch.push(upload::buffer(gpu, slice_bytes(&cbt.tree), storage, "water CBT tree")?);
            scratch.push(upload::buffer(
                gpu,
                slice_bytes(&cbt.bits),
                storage | vk::BufferUsageFlags::TRANSFER_SRC,
                "water CBT bits",
            )?);
            scratch.push(OwnedBuffer::new(
                gpu,
                BufferDesc {
                    size: scratch[1].size,
                    usage: storage | vk::BufferUsageFlags::TRANSFER_DST,
                },
                MemoryLocation::GpuOnly,
                "water CBT snapshot",
            )?);
            bindless.group(|b| {
                let mut handles = vec![frame];
                for buffer in &scratch {
                    handles.push(b.whole_buffer(buffer)?.index);
                }
                HandleTable::new(gpu, b, &handles, "water CBT handles")
            })
        })();
        let (table, group) = match result {
            Ok(made) => made,
            Err(error) => {
                for pipeline in [prepare, update, reduce] {
                    unsafe { gpu.device.destroy_pipeline(pipeline, None) };
                }
                for mut buffer in scratch {
                    buffer.destroy(gpu);
                }
                return Err(error);
            }
        };
        Ok(Self {
            scratch: scratch.try_into().ok().expect("three CBT buffers"),
            table,
            group,
            prepare,
            update,
            reduce,
            layout,
        })
    }

    pub fn record(
        &self,
        ctx: &PassContext<'_>,
        mut push: Push,
        patch: WavePatch,
        waves: u32,
        leaves: BufferBinding,
        indirect: BufferBinding,
    ) -> Result<()> {
        push.table(self.table.index);
        patch.push(&mut push);
        push.image(13, indirect.storage.context("water indirect storage")?);
        push.image(14, waves);
        push.image(15, leaves.storage.context("water mesh leaf storage")?);
        let (device, cmd) = (ctx.device, ctx.cmd);
        let barrier = || unsafe {
            let memory = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&memory));
        };
        let dispatch = |pipeline, words, groups| unsafe {
            let mut block = push;
            block.words = words;
            device.cmd_push_constants(cmd, self.layout, PUSH_STAGES, 0, push_bytes(&block));
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            if groups > 0 {
                device.cmd_dispatch(cmd, groups, 1, 1);
            } else {
                device.cmd_dispatch_indirect(cmd, indirect.buffer, indirect.offset);
            }
            barrier();
        };
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[ctx.global_set],
                &[],
            );
        }
        // The private CBT persists across submissions. Snapshot its bitfield
        // so concurrent splits and merges decode the same original leaves.
        barrier();
        for (mode, draw) in [(0, 0), (0, 0), (1, 0), (2, 1)] {
            unsafe {
                device.cmd_copy_buffer(
                    cmd,
                    self.scratch[1].buffer,
                    self.scratch[2].buffer,
                    &[vk::BufferCopy::default().size(self.scratch[1].size)],
                );
            }
            barrier();
            dispatch(self.prepare, [0, draw, 0, 0], 1);
            dispatch(self.update, [mode, draw, 0, 0], 0);
            if mode != 2 {
                dispatch(self.reduce, [0, 0, 0, 0], (1 << TREE_DEPTH) / 256);
                for level in (8..TREE_DEPTH).rev() {
                    dispatch(self.reduce, [1, 0, level, 0], (1 << level) / 256);
                }
                dispatch(self.reduce, [2, 0, 0, 0], 1);
            }
        }
        dispatch(self.prepare, [1, 0, 0, 0], 1);
        Ok(())
    }

    pub fn retire(&self, bindless: &mut Bindless) {
        bindless.retire_group(self.group);
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        for pipeline in [self.prepare, self.update, self.reduce] {
            unsafe { gpu.device.destroy_pipeline(pipeline, None) };
        }
        self.table.destroy(gpu);
        for buffer in &mut self.scratch {
            buffer.destroy(gpu);
        }
    }
}
