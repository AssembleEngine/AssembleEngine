//! Fullscreen programs: a render module's own fragment shader (content
//! SPIR-V from its program table) drawn as one fullscreen triangle into
//! its pass's colour attachments, with the engine's `fullscreen.vert`.
//!
//! The shader is reflected ([`reflect`]) and checked against the pass
//! ([`check`]) before a pipeline is made. Contract version 4 has one form
//! ([`Form::Bindless`]); the per-pass form that version 3 still accepted was
//! dropped, and a shader that declares it is refused by binding.
//!
//! * **Bindless** ([`Form::Bindless`]): only the global set (M15 F).
//!   Binding 0 is `texture2D images[]`, binding 1 `sampler samplers[]`, and
//!   binding 4 the `std140` blocks: the program's block and its handle
//!   table (`uvec4 handles[]`), both uniform buffers (M15 G measured a
//!   storage buffer at 20-30% more per access). The push block is at most
//!   16 bytes: `uniforms` (the block's index), `sampler` (the sampler's
//!   index), `handles` (the table's index) and a pad word. The table holds
//!   the pass's sampled-image indices, in intent order.
//!
//! The program writes one fragment output per colour attachment, at
//! locations 0..n. Colour attachments blend with straight alpha
//! (`src_alpha, one_minus_src_alpha`), so a shader writing alpha 1 replaces
//! and an overlay writing less blends over what the target holds.

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

/// Which interface a checked program's shader uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// Only the global set: sampled images at 0, samplers at 1, the std140
    /// blocks and the handle table at 4, and a push block (contract
    /// version 4; version 3 also used it).
    Bindless,
}

/// A descriptor binding the shader declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingKind {
    /// A `uniform` block ending at this byte (its last member's end).
    UniformBuffer(u32),
    CombinedImageSampler,
    StorageBuffer,
    StorageImage,
    /// A runtime array of sampled images (`texture2D x[]`), binding 0 of
    /// the bindless form.
    SampledImageArray,
    /// A runtime array of separate samplers (`sampler x[]`), binding 1.
    SamplerArray,
    /// A runtime array of storage buffers, binding 3. A program refuses
    /// it: its block and handle table are uniform blocks at binding 4.
    StorageBufferArray,
    /// A runtime array of `uniform` blocks, binding 4: the program's std140
    /// block (M15 G: a former uniform buffer stays one).
    UniformBufferArray,
    /// Anything else (separate samplers or images, texel buffers, ...).
    Other(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reflection {
    /// `(set, binding, kind)`, sorted.
    pub bindings: Vec<(u32, u32, BindingKind)>,
    /// Locations of the fragment outputs, sorted.
    pub outputs: Vec<u32>,
    /// The push block's byte size; 0 when there is none.
    pub push_bytes: u32,
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
                let (Some(target), Some(member)) = (
                    inst.operands.first().and_then(word),
                    inst.operands.get(1).and_then(word),
                ) else {
                    continue;
                };
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
        decorations
            .get(&id)?
            .iter()
            .find(|i| i.operands.get(1) == Some(&Operand::Decoration(which)))
            .map(|i| i.operands.get(2).and_then(word))
    };
    let types: HashMap<u32, &Instruction> = module
        .types_global_values
        .iter()
        .filter_map(|i| i.result_id.map(|id| (id, i)))
        .collect();
    let constant = |id: u32| -> Option<u32> {
        let i = types.get(&id)?;
        (i.class.opcode == Op::Constant)
            .then(|| i.operands.first().and_then(word))
            .flatten()
    };
    // Byte size of a type in a block (std140/std430 offsets as decorated).
    fn size(
        id: u32,
        types: &HashMap<u32, &Instruction>,
        offsets: &HashMap<(u32, u32), u32>,
        strides: &HashMap<(u32, u32), u32>,
        decoration: &dyn Fn(u32, Decoration) -> Option<Option<u32>>,
        constant: &dyn Fn(u32) -> Option<u32>,
    ) -> Result<u32> {
        let t = types.get(&id).context("unknown type")?;
        let operand = |k: usize| t.operands.get(k).and_then(word).context("malformed type");
        Ok(match t.class.opcode {
            Op::TypeFloat | Op::TypeInt => operand(0)? / 8,
            Op::TypeBool => 4,
            Op::TypeVector => size(operand(0)?, types, offsets, strides, decoration, constant)? * operand(1)?,
            Op::TypeMatrix => bail!("a bare matrix outside a struct member"),
            Op::TypeArray => {
                let stride = decoration(id, Decoration::ArrayStride)
                    .flatten()
                    .context("array without ArrayStride")?;
                stride * constant(operand(1)?).context("array length is not a constant")?
            }
            Op::TypeRuntimeArray => bail!("a runtime array inside a block has no size"),
            Op::TypeStruct => {
                let mut end = 0;
                for (m, member) in t.operands.iter().enumerate() {
                    let member = word(member).context("malformed struct")?;
                    let offset = *offsets.get(&(id, m as u32)).context("struct member without Offset")?;
                    let mt = types.get(&member).context("unknown member type")?;
                    let bytes = if mt.class.opcode == Op::TypeMatrix {
                        let columns = mt.operands.get(1).and_then(word).context("malformed matrix")?;
                        columns
                            * strides
                                .get(&(id, m as u32))
                                .copied()
                                .context("matrix without MatrixStride")?
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
            matches!(
                e.operands.first(),
                Some(Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Fragment))
            ) && matches!(e.operands.get(2), Some(Operand::LiteralString(name)) if name == "main")
        }),
        ..Reflection::default()
    };
    for var in module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Variable)
    {
        let id = var.result_id.context("variable without id")?;
        let Some(Operand::StorageClass(class)) = var.operands.first() else {
            continue;
        };
        let pointer = types
            .get(&var.result_type.context("variable without type")?)
            .context("unknown pointer type")?;
        let pointee = pointer.operands.get(1).and_then(word).context("malformed pointer")?;
        let pointee_type = types.get(&pointee).context("unknown pointee type")?;
        let pointee_op = Some(pointee_type.class.opcode);
        match class {
            StorageClass::Output => {
                if let Some(Some(location)) = decoration(id, Decoration::Location) {
                    out.outputs.push(location);
                }
            }
            StorageClass::PushConstant => {
                out.push_bytes = size(
                    pointee,
                    &types,
                    &member_offsets,
                    &member_matrix_stride,
                    &decoration,
                    &constant,
                )?;
            }
            StorageClass::Uniform | StorageClass::UniformConstant | StorageClass::StorageBuffer => {
                let set = decoration(id, Decoration::DescriptorSet).flatten().unwrap_or(0);
                let binding = decoration(id, Decoration::Binding)
                    .flatten()
                    .context("a resource without a binding")?;
                // The element type of a runtime array (a bindless array).
                let element = (pointee_op == Some(Op::TypeRuntimeArray))
                    .then(|| pointee_type.operands.first().and_then(word))
                    .flatten();
                let element_op = element.and_then(|e| types.get(&e)).map(|t| t.class.opcode);
                let kind = match (*class, pointee_op, element_op) {
                    (_, Some(Op::TypeRuntimeArray), Some(Op::TypeImage)) => BindingKind::SampledImageArray,
                    (_, Some(Op::TypeRuntimeArray), Some(Op::TypeSampler)) => BindingKind::SamplerArray,
                    (StorageClass::Uniform, Some(Op::TypeRuntimeArray), Some(Op::TypeStruct)) => {
                        BindingKind::UniformBufferArray
                    }
                    (_, Some(Op::TypeRuntimeArray), _) => BindingKind::StorageBufferArray,
                    (StorageClass::StorageBuffer, _, _) => BindingKind::StorageBuffer,
                    (StorageClass::Uniform, Some(Op::TypeStruct), _)
                        if decoration(pointee, Decoration::BufferBlock).is_some() =>
                    {
                        BindingKind::StorageBuffer
                    }
                    (StorageClass::Uniform, Some(Op::TypeStruct), _) => BindingKind::UniformBuffer(size(
                        pointee,
                        &types,
                        &member_offsets,
                        &member_matrix_stride,
                        &decoration,
                        &constant,
                    )?),
                    (StorageClass::UniformConstant, Some(Op::TypeSampledImage), _) => BindingKind::CombinedImageSampler,
                    (StorageClass::UniformConstant, Some(Op::TypeImage), _) => {
                        // Operand 6: 1 = sampled, 2 = storage.
                        let sampled = types[&pointee].operands.get(6).and_then(word);
                        if sampled == Some(2) {
                            BindingKind::StorageImage
                        } else {
                            BindingKind::Other("separate image".into())
                        }
                    }
                    (_, op, _) => BindingKind::Other(format!("{op:?}")),
                };
                out.bindings.push((set, binding, kind));
            }
            _ => {}
        }
    }
    out.bindings.sort_by_key(|b| (b.0, b.1));
    // Blocks alias one binding (the handle table and the std140 blocks all
    // live at binding 3), so the same declaration appears more than once.
    out.bindings.dedup();
    out.outputs.sort_unstable();
    Ok(out)
}

/// The push block the bindless form takes: `uniforms`, `sampler`, `handles`,
/// pad.
pub const PUSH_BYTES: u32 = 16;

/// Check a reflected shader against its pass: `uniform_bytes` declared,
/// `sampled` inputs, `colors` attachments. Contract version 4 has only the
/// bindless form; the per-pass form that version 3 still accepted was
/// dropped, so a shader that declares it is refused by binding.
pub fn check(r: &Reflection, uniform_bytes: u32, sampled: usize, colors: usize) -> Result<Form, String> {
    if !r.fragment_main {
        return Err("no fragment entry point `main`".into());
    }
    let expected: Vec<u32> = (0..colors as u32).collect();
    if r.outputs != expected {
        return Err(format!(
            "writes outputs at locations {:?}; the pass has {colors} colour attachment(s)",
            r.outputs
        ));
    }
    // The per-pass form (set 0: the uniform block, then a combined image
    // sampler per sampled intent) ended at contract version 3.
    if let Some((set, binding, kind)) = r
        .bindings
        .iter()
        .find(|(_, _, kind)| matches!(kind, BindingKind::UniformBuffer(_) | BindingKind::CombinedImageSampler))
    {
        return Err(format!(
            "set {set} binding {binding} ({kind:?}) is the per-pass fullscreen form, which contract version 4 dropped; use the global set's bindless arrays"
        ));
    }
    // The bindless form: the global set's arrays and a small push block.
    let bindless = |set: &u32, binding: &u32, kind: &BindingKind| {
        *set == 0
            && matches!(
                (binding, kind),
                (0, BindingKind::SampledImageArray)
                    | (1, BindingKind::SamplerArray)
                    | (3, BindingKind::StorageBufferArray)
                    | (4, BindingKind::UniformBufferArray)
            )
    };
    if let Some((set, binding, kind)) = r
        .bindings
        .iter()
        .find(|(set, binding, kind)| !bindless(set, binding, kind))
    {
        return Err(format!(
            "set {set} binding {binding} ({kind:?}) is not part of the bindless fullscreen form (set 0 bindings 0, 1, 3 and 4, and a 16-byte push block)"
        ));
    }
    if r.push_bytes > PUSH_BYTES {
        return Err(format!(
            "the push block is {} bytes; the bindless form takes at most {PUSH_BYTES}",
            r.push_bytes
        ));
    }
    let has = |binding: u32, want: &BindingKind| {
        r.bindings
            .iter()
            .any(|(set, b, kind)| *set == 0 && *b == binding && kind == want)
    };
    if sampled > 0 {
        for (binding, want) in [(0, BindingKind::SampledImageArray), (1, BindingKind::SamplerArray)] {
            if !has(binding, &want) {
                return Err(format!(
                    "the pass samples {sampled} image(s); the bindless form declares set 0 binding {binding} ({want:?})"
                ));
            }
        }
    }
    // The block and the handle table are both blocks in the uniform array
    // (binding 4). A std140 block written as a storage buffer at binding 3
    // reflects as one; name it, because that is the mistake this catches
    // (the table moved there in F and to binding 4 in G).
    if has(3, &BindingKind::StorageBufferArray) {
        return Err(
            "set 0 binding 3 is a storage buffer; a program's std140 block and its handle table are uniform blocks at binding 4"
                .into(),
        );
    }
    if (sampled > 0 || uniform_bytes > 0) && !has(4, &BindingKind::UniformBufferArray) {
        return Err(format!(
            "the program declares a {uniform_bytes}-byte std140 block and {sampled} sampled image(s); the bindless form reaches both through set 0 binding 4 (UniformBufferArray)"
        ));
    }
    if r.push_bytes == 0 && (sampled > 0 || uniform_bytes > 0) {
        return Err(format!(
            "declares no push block; the bindless form takes up to {PUSH_BYTES} bytes (uniforms, sampler, handles)"
        ));
    }
    Ok(Form::Bindless)
}

/// The pass's sampler as a bindless program needs it: its index in the
/// global sampler array.
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    pub index: u32,
}

/// A checked fullscreen program: its pipeline and its global-set handles.
pub struct FullscreenProgram {
    pub label: String,
    pub uniform_bytes: u32,
    form: Form,
    /// The pass's sampled intents (the bindless form's table size).
    sampled: usize,
    pipeline: vk::Pipeline,
    bindless: Option<BindlessState>,
}

/// The bindless form's state, shared by a module's program and the engine's
/// ([`BindlessProgram`]): a std140 block and a handle table per frame slot
/// (one slotted index each), and the group that owns both.
struct BindlessState {
    /// The shared pipeline layout (the global set at 0, the push block).
    layout: vk::PipelineLayout,
    ubos: Vec<OwnedBuffer>,
    tables: Vec<OwnedBuffer>,
    /// The std140 block's slotted index and the handle table's.
    uniforms: crate::bindless::UniformBufferHandle,
    handles: crate::bindless::UniformBufferHandle,
    group: Option<crate::bindless::GroupId>,
}

impl FullscreenProgram {
    /// Reflect and check `spirv`, then build the program.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &GpuDevice,
        bindless: &mut crate::bindless::Bindless,
        label: &str,
        spirv: &[u8],
        uniform_bytes: u32,
        sampled: usize,
        color_formats: &[vk::Format],
        frames_in_flight: usize,
    ) -> Result<Self> {
        let reflection = reflect(spirv).with_context(|| format!("program `{label}`"))?;
        let form = check(&reflection, uniform_bytes, sampled, color_formats.len())
            .map_err(|e| anyhow::anyhow!("program `{label}`: {e}"))?;
        let device = &gpu.device;
        let (mut this, group) = bindless.group(|bindless| -> Result<Self> {
            let bindless_state =
                BindlessState::new(gpu, bindless, label, uniform_bytes, sampled.max(1), frames_in_flight)?;
            let pipeline = pipeline::graphics_pipeline(
                device,
                &GraphicsDesc {
                    vertex: spirv!("fullscreen.vert"),
                    fragment: spirv,
                    layout: bindless.pipeline_layout(),
                    color_formats,
                    depth: None,
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    blend: Blend::Straight,
                    glow_blend: Blend::Off,
                },
            )
            .with_context(|| format!("program `{label}` pipeline"))?;
            Ok(Self {
                label: label.into(),
                uniform_bytes,
                form,
                sampled,
                pipeline,
                bindless: Some(bindless_state),
            })
        })?;
        this.bindless.as_mut().expect("a checked program has a form").group = Some(group);
        Ok(this)
    }

    /// The form the shader was checked against.
    pub fn form(&self) -> Form {
        self.form
    }

    /// This frame's uniform block for `slot` (its previous use completed).
    pub fn write_uniforms(&mut self, slot: usize, bytes: &[u8]) {
        let Some(state) = &mut self.bindless else { return };
        if let Some(mapped) = state.ubos.get_mut(slot).and_then(|u| u.mapped()) {
            let n = bytes.len().min(self.uniform_bytes as usize);
            mapped[..n].copy_from_slice(&bytes[..n]);
        }
    }

    /// Draw, sampling `images` (the pass's sampled inputs in order) with
    /// `sampler`. Records inside the pass's graphics scope.
    pub fn record(&self, ctx: &PassContext<'_>, images: &[ImageBinding], sampler: Sampler) -> Result<()> {
        anyhow::ensure!(
            images.len() == self.sampled,
            "program `{}` samples {} images, given {}",
            self.label,
            self.sampled,
            images.len()
        );
        let state = self.bindless.as_ref().expect("a checked program has a form");
        let indices: Vec<u32> = images.iter().map(|image| image.sampled.unwrap_or(0)).collect();
        state.record(ctx, self.pipeline, &indices, sampler.index);
        Ok(())
    }

    /// Give the bindless form's handles back; the device objects stay until
    /// [`Self::destroy`].
    pub fn retire(&mut self, bindless: &mut crate::bindless::Bindless) {
        if let Some(state) = self.bindless.as_mut() {
            state.retire(bindless);
        }
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        unsafe {
            gpu.device.destroy_pipeline(self.pipeline, None);
        }
        if let Some(mut state) = self.bindless.take() {
            state.destroy(gpu);
        }
    }
}

/// An engine-owned fullscreen program on the global set (M15 F): tonemap,
/// bloom and encode. It takes its sampled images as global indices when it
/// records, writes its handle table for the slot and pushes `uniforms`,
/// `sampler` and `handles`.
pub struct BindlessProgram {
    pub label: String,
    pub uniform_bytes: u32,
    state: BindlessState,
    pipeline: vk::Pipeline,
}

impl BindlessProgram {
    /// `sampled`: how many images the shader samples (its table's size).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &GpuDevice,
        bindless: &mut crate::bindless::Bindless,
        label: &str,
        spirv: &[u8],
        uniform_bytes: u32,
        sampled: usize,
        color_formats: &[vk::Format],
        frames_in_flight: usize,
    ) -> Result<Self> {
        let (mut this, group) = bindless.group(|bindless| -> Result<Self> {
            let state = BindlessState::new(gpu, bindless, label, uniform_bytes, sampled, frames_in_flight)?;
            let pipeline = pipeline::graphics_pipeline(
                &gpu.device,
                &GraphicsDesc {
                    vertex: spirv!("fullscreen.vert"),
                    fragment: spirv,
                    layout: state.layout,
                    color_formats,
                    depth: None,
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    blend: Blend::Straight,
                    glow_blend: Blend::Off,
                },
            )
            .with_context(|| format!("program `{label}` pipeline"))?;
            Ok(Self {
                label: label.into(),
                uniform_bytes,
                state,
                pipeline,
            })
        })?;
        this.state.group = Some(group);
        Ok(this)
    }

    pub fn write_uniforms(&mut self, slot: usize, bytes: &[u8]) {
        if let Some(mapped) = self.state.ubos.get_mut(slot).and_then(|u| u.mapped()) {
            let n = bytes.len().min(self.uniform_bytes as usize);
            mapped[..n].copy_from_slice(&bytes[..n]);
        }
    }

    /// Draw, sampling the images at `images` with `samplers` (indices into
    /// the global arrays; one sampler serves the shader). Records inside the
    /// pass's graphics scope.
    pub fn record(&self, ctx: &PassContext<'_>, images: &[u32], samplers: &[u32]) -> Result<()> {
        anyhow::ensure!(
            images.len() <= 16,
            "program `{}` needs more than 16 sampled images",
            self.label
        );
        let sampler = samplers.first().copied().unwrap_or(0);
        self.state.record(ctx, self.pipeline, images, sampler);
        Ok(())
    }

    /// Give the global-set handles back; the device objects stay until
    /// [`Self::destroy`].
    pub fn retire(&mut self, bindless: &mut crate::bindless::Bindless) {
        self.state.retire(bindless);
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        unsafe {
            gpu.device.destroy_pipeline(self.pipeline, None);
        }
        self.state.destroy(gpu);
    }
}

impl BindlessState {
    fn new(
        gpu: &GpuDevice,
        bindless: &mut crate::bindless::Bindless,
        label: &str,
        uniform_bytes: u32,
        sampled: usize,
        frames_in_flight: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            sampled <= crate::bindless::TABLE_SLOTS,
            "program `{label}` samples {sampled} images; the handle table holds {}",
            crate::bindless::TABLE_SLOTS
        );
        let ubos = (0..frames_in_flight)
            .map(|_| {
                OwnedBuffer::new(
                    gpu,
                    BufferDesc {
                        size: uniform_bytes.max(16) as u64,
                        usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                    },
                    MemoryLocation::CpuToGpu,
                    label,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // The handle table is a `uvec4` block in the uniform array (M15 G),
        // so every slot holds the whole declared block.
        let tables = (0..frames_in_flight)
            .map(|_| {
                OwnedBuffer::new(
                    gpu,
                    BufferDesc {
                        size: (crate::bindless::TABLE_SLOTS * 4) as u64,
                        usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                    },
                    MemoryLocation::CpuToGpu,
                    "program handles",
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let uniforms = bindless.slotted_uniform_buffer()?;
        for (slot, ubo) in ubos.iter().enumerate() {
            bindless.update_slotted_uniform_buffer(uniforms, slot, ubo.buffer, 0, ubo.size);
        }
        let handles = bindless.slotted_uniform_buffer()?;
        for (slot, table) in tables.iter().enumerate() {
            bindless.update_slotted_uniform_buffer(handles, slot, table.buffer, 0, table.size);
        }
        Ok(Self {
            layout: bindless.pipeline_layout(),
            ubos,
            tables,
            uniforms,
            handles,
            group: None,
        })
    }

    /// Record the pass: the slot's handle table gets `images`, then the
    /// push block names the block, the sampler and the table.
    fn record(&self, ctx: &PassContext<'_>, pipeline: vk::Pipeline, images: &[u32], sampler: u32) {
        let table = &self.tables[ctx.slot];
        let bytes: Vec<u8> = images.iter().flat_map(|index| index.to_le_bytes()).collect();
        // The indices are the same every frame, so this writes once per
        // bake; the compare is a few bytes. The slot's previous frame has
        // completed: only this slot reads the buffer.
        if table.mapped_read(0, bytes.len()) != Some(&bytes[..]) {
            table.write_mapped(0, &bytes);
        }
        let push = [self.uniforms.index, sampler, self.handles.index, 0];
        unsafe {
            ctx.device
                .cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            // The shared layout's push range covers every stage; a push
            // must name all of them.
            ctx.device.cmd_push_constants(
                ctx.cmd,
                self.layout,
                crate::bindless::PUSH_STAGES,
                0,
                pipeline::push_bytes(&push),
            );
            ctx.device.cmd_bind_descriptor_sets(
                ctx.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[ctx.global_set],
                &[],
            );
            ctx.device.cmd_draw(ctx.cmd, 3, 1, 0, 0);
        }
    }

    fn retire(&mut self, bindless: &mut crate::bindless::Bindless) {
        if let Some(group) = self.group.take() {
            bindless.retire_group(group);
        }
    }

    fn destroy(&mut self, gpu: &GpuDevice) {
        for ubo in &mut self.ubos {
            ubo.destroy(gpu);
        }
        for table in &mut self.tables {
            table.destroy(gpu);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bindless form: the global set's arrays, the std140 block and the
    /// handle table at 4 and a 16-byte push block. A per-pass binding (with
    /// or without the global set's arrays), a block or table at binding 3
    /// and a push block over its size are refused with the binding named.
    #[test]
    fn the_bindless_form_is_checked_and_the_per_pass_form_is_refused() {
        let program = reflect(spirv!("test_program.frag")).unwrap();
        assert_eq!(program.push_bytes, PUSH_BYTES);
        assert_eq!(
            program.bindings,
            [
                (0, 0, BindingKind::SampledImageArray),
                (0, 1, BindingKind::SamplerArray),
                (0, 4, BindingKind::UniformBufferArray),
            ]
        );
        assert_eq!(check(&program, 32, 2, 1).unwrap(), Form::Bindless);
        assert!(check(&program, 32, 2, 2).unwrap_err().contains("colour attachment"));

        // The per-pass form ended at contract version 3, whether it comes
        // alone or beside the global set's arrays; the binding is named.
        let per_pass = Reflection {
            bindings: vec![
                (0, 0, BindingKind::UniformBuffer(28)),
                (0, 1, BindingKind::CombinedImageSampler),
            ],
            outputs: vec![0],
            fragment_main: true,
            ..Default::default()
        };
        let e = check(&per_pass, 28, 1, 1).unwrap_err();
        assert!(
            e.contains("per-pass") && e.contains("version 4") && e.contains("binding 0"),
            "{e}"
        );
        let mut mixed = program.clone();
        mixed.bindings.push((0, 2, BindingKind::CombinedImageSampler));
        mixed.bindings.sort_by_key(|b| (b.0, b.1));
        let e = check(&mixed, 32, 2, 1).unwrap_err();
        assert!(
            e.contains("per-pass") && e.contains("version 4") && e.contains("binding 2"),
            "{e}"
        );

        // A program that declares nothing needs no global-set binding.
        let empty = Reflection {
            outputs: vec![0],
            fragment_main: true,
            ..Default::default()
        };
        assert_eq!(check(&empty, 0, 0, 1).unwrap(), Form::Bindless);

        // A program without its uniform array cannot reach its block or its
        // images.
        let mut missing = program.clone();
        missing.bindings.retain(|(_, binding, _)| *binding != 4);
        let e = check(&missing, 32, 2, 1).unwrap_err();
        assert!(e.contains("binding 4"), "{e}");

        // A std140 block at binding 3 is a storage buffer: refused, and the
        // message says where the block belongs.
        let mut storage_block = program.clone();
        for b in &mut storage_block.bindings {
            if b.1 == 4 {
                b.1 = 3;
                b.2 = BindingKind::StorageBufferArray;
            }
        }
        storage_block.bindings.sort_by_key(|b| (b.0, b.1));
        let e = check(&storage_block, 32, 2, 1).unwrap_err();
        assert!(e.contains("binding 3") && e.contains("binding 4"), "{e}");

        // The push block is at most 16 bytes.
        let mut wide = program.clone();
        wide.push_bytes = 32;
        assert!(check(&wide, 32, 2, 1).unwrap_err().contains("at most 16"));

        assert!(reflect(&[1, 2, 3, 4]).is_err());
        let vertex = reflect(spirv!("fullscreen.vert")).unwrap();
        assert!(check(&vertex, 0, 0, 0).unwrap_err().contains("fragment"));
    }
}
