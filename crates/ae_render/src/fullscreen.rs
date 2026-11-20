//! Fullscreen programs: a render module's own fragment shader (content
//! SPIR-V from its program table) drawn as one fullscreen triangle into
//! its pass's colour attachments, with the engine's `fullscreen.vert`.
//!
//! The interface is fixed by the pass declaration, and the shader is
//! reflected ([`reflect`]) and checked against it ([`check`]) before a
//! pipeline is made:
//! * set 0, binding 0: the uniform block when the program declares
//!   `uniform_bytes > 0`: from the end of its last member up to that
//!   rounded to 16 bytes (std140), so a module may send the bare struct or
//!   it padded;
//! * set 0, bindings 1..=n: a combined image sampler (2D) per `Sampled`
//!   intent of the pass, in intent order (sampled linearly, clamped);
//! * nothing else: no storage buffers or images (the pass declares no
//!   storage access), no push constants;
//! * one fragment output per colour attachment, locations 0..n.
//!
//! Colour attachments blend with straight alpha (`src_alpha,
//! one_minus_src_alpha`), so a shader writing alpha 1 replaces and an
//! overlay writing less blends over what the target holds.

use crate::device::GpuDevice;
use crate::executor::{ImageBinding, PassContext};
use crate::memory::OwnedBuffer;
use crate::pipeline::{self, Blend, GraphicsDesc, spirv};
use ae_framegraph::BufferDesc;
use anyhow::{Context as _, Result, bail};
use ash::vk;
use gpu_allocator::MemoryLocation;
use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::{Decoration, Op, StorageClass};
use std::collections::HashMap;

/// A descriptor binding the shader declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingKind {
    /// A `uniform` block ending at this byte (its last member's end).
    UniformBuffer(u32),
    CombinedImageSampler,
    StorageBuffer,
    StorageImage,
    /// Anything else (separate samplers or images, texel buffers, ...).
    Other(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reflection {
    /// `(set, binding, kind)`, sorted.
    pub bindings: Vec<(u32, u32, BindingKind)>,
    /// Locations of the fragment outputs, sorted.
    pub outputs: Vec<u32>,
    pub push_constants: bool,
    /// Has a `Fragment` entry point named `main`.
    pub fragment_main: bool,
}

fn word(op: &Operand) -> Option<u32> {
    match op {
        Operand::LiteralBit32(v) | Operand::IdRef(v) => Some(*v),
        _ => None,
    }
}

/// Reflect a fragment shader's interface.
pub fn reflect(spirv: &[u8]) -> Result<Reflection> {
    let module = rspirv::dr::load_bytes(spirv).map_err(|e| anyhow::anyhow!("not SPIR-V: {e}"))?;
    let mut decorations: HashMap<u32, Vec<&Instruction>> = HashMap::new();
    let mut member_offsets: HashMap<(u32, u32), u32> = HashMap::new();
    let mut member_matrix_stride: HashMap<(u32, u32), u32> = HashMap::new();
    for inst in &module.annotations {
        match inst.class.opcode {
            Op::Decorate => {
                if let Some(target) = inst.operands.first().and_then(word) {
                    decorations.entry(target).or_default().push(inst);
                }
            }
            Op::MemberDecorate => {
                let (Some(target), Some(member)) = (inst.operands.first().and_then(word), inst.operands.get(1).and_then(word)) else { continue };
                match (inst.operands.get(2), inst.operands.get(3).and_then(word)) {
                    (Some(Operand::Decoration(Decoration::Offset)), Some(offset)) => {
                        member_offsets.insert((target, member), offset);
                    }
                    (Some(Operand::Decoration(Decoration::MatrixStride)), Some(stride)) => {
                        member_matrix_stride.insert((target, member), stride);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    let decoration = |id: u32, which: Decoration| -> Option<Option<u32>> {
        decorations.get(&id)?.iter().find(|i| i.operands.get(1) == Some(&Operand::Decoration(which))).map(|i| i.operands.get(2).and_then(word))
    };
    let types: HashMap<u32, &Instruction> = module.types_global_values.iter().filter_map(|i| i.result_id.map(|id| (id, i))).collect();
    let constant = |id: u32| -> Option<u32> {
        let i = types.get(&id)?;
        (i.class.opcode == Op::Constant).then(|| i.operands.first().and_then(word)).flatten()
    };
    // Byte size of a type in a block (std140/std430 offsets as decorated).
    fn size(id: u32, types: &HashMap<u32, &Instruction>, offsets: &HashMap<(u32, u32), u32>, strides: &HashMap<(u32, u32), u32>, decoration: &dyn Fn(u32, Decoration) -> Option<Option<u32>>, constant: &dyn Fn(u32) -> Option<u32>) -> Result<u32> {
        let t = types.get(&id).context("unknown type")?;
        let operand = |k: usize| t.operands.get(k).and_then(word).context("malformed type");
        Ok(match t.class.opcode {
            Op::TypeFloat | Op::TypeInt => operand(0)? / 8,
            Op::TypeBool => 4,
            Op::TypeVector => size(operand(0)?, types, offsets, strides, decoration, constant)? * operand(1)?,
            Op::TypeMatrix => bail!("a bare matrix outside a struct member"),
            Op::TypeArray => {
                let stride = decoration(id, Decoration::ArrayStride).flatten().context("array without ArrayStride")?;
                stride * constant(operand(1)?).context("array length is not a constant")?
            }
            Op::TypeStruct => {
                let mut end = 0;
                for (m, member) in t.operands.iter().enumerate() {
                    let member = word(member).context("malformed struct")?;
                    let offset = *offsets.get(&(id, m as u32)).context("struct member without Offset")?;
                    let mt = types.get(&member).context("unknown member type")?;
                    let bytes = if mt.class.opcode == Op::TypeMatrix {
                        let columns = mt.operands.get(1).and_then(word).context("malformed matrix")?;
                        columns * strides.get(&(id, m as u32)).copied().context("matrix without MatrixStride")?
                    } else {
                        size(member, types, offsets, strides, decoration, constant)?
                    };
                    end = end.max(offset + bytes);
                }
                end
            }
            other => bail!("unsupported type {other:?} in a block"),
        })
    }
    let mut out = Reflection {
        fragment_main: module.entry_points.iter().any(|e| {
            matches!(e.operands.first(), Some(Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Fragment))) && matches!(e.operands.get(2), Some(Operand::LiteralString(name)) if name == "main")
        }),
        ..Reflection::default()
    };
    for var in module.types_global_values.iter().filter(|i| i.class.opcode == Op::Variable) {
        let id = var.result_id.context("variable without id")?;
        let Some(Operand::StorageClass(class)) = var.operands.first() else { continue };
        let pointer = types.get(&var.result_type.context("variable without type")?).context("unknown pointer type")?;
        let pointee = pointer.operands.get(1).and_then(word).context("malformed pointer")?;
        let pointee_op = types.get(&pointee).map(|t| t.class.opcode);
        match class {
            StorageClass::Output => {
                if let Some(Some(location)) = decoration(id, Decoration::Location) {
                    out.outputs.push(location);
                }
            }
            StorageClass::PushConstant => out.push_constants = true,
            StorageClass::Uniform | StorageClass::UniformConstant | StorageClass::StorageBuffer => {
                let set = decoration(id, Decoration::DescriptorSet).flatten().unwrap_or(0);
                let binding = decoration(id, Decoration::Binding).flatten().context("a resource without a binding")?;
                let kind = match (class, pointee_op) {
                    (StorageClass::StorageBuffer, _) => BindingKind::StorageBuffer,
                    (StorageClass::Uniform, Some(Op::TypeStruct)) if decoration(pointee, Decoration::BufferBlock).is_some() => BindingKind::StorageBuffer,
                    (StorageClass::Uniform, Some(Op::TypeStruct)) => BindingKind::UniformBuffer(size(pointee, &types, &member_offsets, &member_matrix_stride, &decoration, &constant)?),
                    (StorageClass::UniformConstant, Some(Op::TypeSampledImage)) => BindingKind::CombinedImageSampler,
                    (StorageClass::UniformConstant, Some(Op::TypeImage)) => {
                        // Operand 6: 1 = sampled, 2 = storage.
                        let sampled = types[&pointee].operands.get(6).and_then(word);
                        if sampled == Some(2) { BindingKind::StorageImage } else { BindingKind::Other("separate image".into()) }
                    }
                    (_, op) => BindingKind::Other(format!("{op:?}")),
                };
                out.bindings.push((set, binding, kind));
            }
            _ => {}
        }
    }
    out.bindings.sort_by_key(|b| (b.0, b.1));
    out.outputs.sort_unstable();
    Ok(out)
}

/// Check a reflected shader against its pass: `uniform_bytes` declared,
/// `sampled` inputs, `colors` attachments. The error says what differs.
pub fn check(r: &Reflection, uniform_bytes: u32, sampled: usize, colors: usize) -> Result<(), String> {
    if !r.fragment_main {
        return Err("no fragment entry point `main`".into());
    }
    if r.push_constants {
        return Err("uses push constants; fullscreen programs take a uniform block at binding 0".into());
    }
    let mut want: Vec<(u32, u32, BindingKind)> = Vec::new();
    if uniform_bytes > 0 {
        want.push((0, 0, BindingKind::UniformBuffer(uniform_bytes)));
    }
    want.extend((1..=sampled as u32).map(|b| (0, b, BindingKind::CombinedImageSampler)));
    for b in &r.bindings {
        match want.iter().find(|w| (w.0, w.1) == (b.0, b.1)) {
            Some(w) if w.2 == b.2 => {}
            Some((_, _, BindingKind::UniformBuffer(declared))) if matches!(b.2, BindingKind::UniformBuffer(end) if (end..=end.next_multiple_of(16)).contains(declared)) => {}
            Some((_, _, BindingKind::UniformBuffer(declared))) => {
                let BindingKind::UniformBuffer(end) = b.2 else { return Err(format!("set {} binding {} is {:?}; the pass declares a uniform block", b.0, b.1, b.2)) };
                return Err(format!("the uniform block is {end} bytes ({} padded); the program declares {declared}", end.next_multiple_of(16)));
            }
            Some(w) => return Err(format!("set {} binding {} is {:?}; the pass declares {:?}", b.0, b.1, b.2, w.2)),
            None => {
                let declared = match b.2 {
                    BindingKind::StorageBuffer | BindingKind::StorageImage => " (the pass declares no storage access)",
                    _ => "",
                };
                return Err(format!("set {} binding {} ({:?}) is not declared by the pass{declared}", b.0, b.1, b.2));
            }
        }
    }
    let expected: Vec<u32> = (0..colors as u32).collect();
    if r.outputs != expected {
        return Err(format!("writes outputs at locations {:?}; the pass has {colors} colour attachment(s)", r.outputs));
    }
    Ok(())
}

/// A checked fullscreen program: pipeline, per-slot uniform buffers and
/// descriptor sets.
pub struct FullscreenProgram {
    pub label: String,
    pub uniform_bytes: u32,
    sampled: u32,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    ubos: Vec<OwnedBuffer>,
    pipeline: vk::Pipeline,
}

impl FullscreenProgram {
    /// Reflect and check `spirv`, then build the program.
    #[allow(clippy::too_many_arguments)]
    pub fn new(gpu: &GpuDevice, label: &str, spirv: &[u8], uniform_bytes: u32, sampled: usize, color_formats: &[vk::Format], frames_in_flight: usize) -> Result<Self> {
        let reflection = reflect(spirv).with_context(|| format!("program `{label}`"))?;
        check(&reflection, uniform_bytes, sampled, color_formats.len()).map_err(|e| anyhow::anyhow!("program `{label}`: {e}"))?;
        let device = &gpu.device;
        let stages = vk::ShaderStageFlags::FRAGMENT;
        let mut kinds = Vec::new();
        if uniform_bytes > 0 {
            kinds.push((0u32, vk::DescriptorType::UNIFORM_BUFFER));
        }
        kinds.extend((1..=sampled as u32).map(|b| (b, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)));
        let bindings: Vec<_> = kinds.iter().map(|&(b, ty)| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(ty).descriptor_count(1).stage_flags(stages)).collect();
        let n = frames_in_flight as u32;
        unsafe {
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let layout = pipeline::pipeline_layout(device, &[set_layout], 0, vk::ShaderStageFlags::empty())?;
            let mut sizes = vec![vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: n * (sampled as u32).max(1) }];
            if uniform_bytes > 0 {
                sizes.push(vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: n });
            }
            let pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&sizes), None)?;
            let layouts = vec![set_layout; frames_in_flight];
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))?;
            let ubos = if uniform_bytes > 0 {
                (0..frames_in_flight)
                    .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: uniform_bytes as u64, usage: vk::BufferUsageFlags::UNIFORM_BUFFER }, MemoryLocation::CpuToGpu, label))
                    .collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
            for (slot, &set) in sets.iter().enumerate() {
                if let Some(ubo) = ubos.get(slot) {
                    let info = [vk::DescriptorBufferInfo { buffer: ubo.buffer, offset: 0, range: vk::WHOLE_SIZE }];
                    device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&info)], &[]);
                }
            }
            let pipeline = pipeline::graphics_pipeline(
                device,
                &GraphicsDesc {
                    vertex: spirv!("fullscreen.vert"),
                    fragment: spirv,
                    layout,
                    color_formats,
                    depth: None,
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    blend: Blend::Straight,
                    glow_blend: Blend::Off,
                },
            )
            .with_context(|| format!("program `{label}` pipeline"))?;
            Ok(Self { label: label.into(), uniform_bytes, sampled: sampled as u32, set_layout, layout, pool, sets, ubos, pipeline })
        }
    }

    /// This frame's uniform block for `slot` (its previous use completed).
    pub fn write_uniforms(&mut self, slot: usize, bytes: &[u8]) {
        if let Some(mapped) = self.ubos.get_mut(slot).and_then(|u| u.mapped()) {
            let n = bytes.len().min(self.uniform_bytes as usize);
            mapped[..n].copy_from_slice(&bytes[..n]);
        }
    }

    /// Draw, sampling `images` (the pass's sampled inputs in order) with
    /// `sampler`. Records inside the pass's graphics scope.
    pub fn record(&self, ctx: &PassContext<'_>, images: &[ImageBinding], sampler: vk::Sampler) -> Result<()> {
        anyhow::ensure!(images.len() == self.sampled as usize, "program `{}` samples {} images, given {}", self.label, self.sampled, images.len());
        let set = self.sets[ctx.slot];
        let infos: Vec<[vk::DescriptorImageInfo; 1]> = images.iter().map(|i| [vk::DescriptorImageInfo { sampler, image_view: i.view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }]).collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(k, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1 + k as u32).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(info))
            .collect();
        unsafe {
            if !writes.is_empty() {
                // The slot's previous frame has completed: the set is idle.
                ctx.device.update_descriptor_sets(&writes, &[]);
            }
            ctx.device.cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            if self.uniform_bytes > 0 || self.sampled > 0 {
                ctx.device.cmd_bind_descriptor_sets(ctx.cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[set], &[]);
            }
            ctx.device.cmd_draw(ctx.cmd, 3, 1, 0, 0);
        }
        Ok(())
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        unsafe {
            gpu.device.destroy_pipeline(self.pipeline, None);
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device.destroy_descriptor_pool(self.pool, None);
            gpu.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
        for u in &mut self.ubos {
            u.destroy(gpu);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interface of `test_post.frag` (a uniform block of 28 bytes,
    /// two sampled inputs, one output) passes; every difference from what
    /// a pass declares is named.
    #[test]
    fn shaders_are_checked_against_their_pass() {
        let ok = reflect(spirv!("test_post.frag")).unwrap();
        assert_eq!(ok.bindings, [(0, 0, BindingKind::UniformBuffer(28)), (0, 1, BindingKind::CombinedImageSampler), (0, 2, BindingKind::CombinedImageSampler)]);
        assert_eq!(ok.outputs, [0]);
        assert!(ok.fragment_main && !ok.push_constants);
        check(&ok, 32, 2, 1).unwrap();
        check(&ok, 28, 2, 1).unwrap();
        let e = check(&ok, 48, 2, 1).unwrap_err();
        assert!(e.contains("28 bytes (32 padded)") && e.contains("declares 48"), "{e}");
        assert!(check(&ok, 24, 2, 1).is_err());
        assert!(check(&ok, 32, 1, 1).unwrap_err().contains("binding 2"));
        assert!(check(&ok, 32, 3, 1).is_ok(), "an unused declared input is allowed");
        assert!(check(&ok, 32, 2, 2).unwrap_err().contains("colour attachment"));
        assert!(check(&ok, 0, 2, 1).unwrap_err().contains("binding 0"));

        let storage = reflect(spirv!("test_post_storage.frag")).unwrap();
        let e = check(&storage, 0, 1, 1).unwrap_err();
        assert!(e.contains("no storage access"), "{e}");

        let pushes = reflect(spirv!("test_post_push.frag")).unwrap();
        assert!(check(&pushes, 0, 0, 1).unwrap_err().contains("push constants"));

        assert!(reflect(&[1, 2, 3, 4]).is_err());
        let vertex = reflect(spirv!("fullscreen.vert")).unwrap();
        assert!(check(&vertex, 0, 0, 0).unwrap_err().contains("fragment"));
    }
}
