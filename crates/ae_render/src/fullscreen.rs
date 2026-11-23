//! Fullscreen programs: a render module's own fragment shader (content
//! SPIR-V from its program table) drawn as one fullscreen triangle into
//! its pass's colour attachments, with the engine's `fullscreen.vert`.
//!
//! The shader is reflected ([`reflect`]) and checked against the pass
//! ([`check`]) before a pipeline is made. Two forms exist, and a shader
//! uses exactly one (contract version 3 accepts both; version 2 modules
//! may only use the first, and the next version drops it):
//!
//! * **Per-pass** ([`Form::Legacy`]): set 0, binding 0 is the uniform block
//!   when the program declares `uniform_bytes > 0`: from the end of its
//!   last member up to that rounded to 16 bytes (std140), so a module may
//!   send the bare struct or it padded; set 0, bindings 1..=n are one
//!   combined image sampler (2D) per `Sampled` intent of the pass, in
//!   intent order (sampled linearly, clamped). Nothing else: no storage
//!   buffers or images, no push constants.
//! * **Bindless** ([`Form::Bindless`]): only the global set (M15 F).
//!   Binding 0 is `texture2D images[]`, binding 1 `sampler samplers[]`,
//!   binding 3 the program's `uint handles[]` table and binding 4 its
//!   `std140` block (`uniform` blocks stay uniform buffers: M15 G measured
//!   a storage buffer at 20-30% more per access). The push block is at most
//!   16 bytes: `uniforms` (the block's index), `sampler` (the sampler's
//!   index), `handles` (the table's index) and a pad word. The table holds
//!   the pass's sampled-image indices, in intent order.
//!
//! Either form writes one fragment output per colour attachment, at
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
    /// Set 0: the uniform block at binding 0 and a combined image sampler
    /// per sampled intent at 1.. (contract version 2).
    Legacy,
    /// Only the global set: sampled images at 0, samplers at 1, the std140
    /// blocks and the handle table at 3, and a push block (contract
    /// version 3).
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
    /// A runtime array of storage buffers, binding 3: the handle table.
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
/// `sampled` inputs, `colors` attachments. The form is detected and the
/// two cannot be mixed. `legacy_only` is set for a contract-2 module, whose
/// programs must use the per-pass form.
pub fn check(
    r: &Reflection,
    uniform_bytes: u32,
    sampled: usize,
    colors: usize,
    legacy_only: bool,
) -> Result<Form, String> {
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
    // A mix of the two forms names the binding that starts it.
    let bindless_kind = |kind: &BindingKind| {
        matches!(
            kind,
            BindingKind::SampledImageArray
                | BindingKind::SamplerArray
                | BindingKind::StorageBufferArray
                | BindingKind::UniformBufferArray
        )
    };
    let any_bindless = r.bindings.iter().any(|(_, _, kind)| bindless_kind(kind));
    let any_per_pass = r
        .bindings
        .iter()
        .any(|(_, _, kind)| matches!(kind, BindingKind::UniformBuffer(_) | BindingKind::CombinedImageSampler));
    if any_bindless && any_per_pass {
        let (set, binding, kind) = r
            .bindings
            .iter()
            .find(|(_, _, kind)| bindless_kind(kind))
            .expect("a bindless binding");
        return Err(format!(
            "set {set} binding {binding} ({kind:?}) mixes the bindless arrays with the per-pass form"
        ));
    }
    // The bindless form: the global set's arrays and a small push block.
    let bindless = r.bindings.iter().all(|(set, binding, kind)| {
        *set == 0
            && matches!(
                (binding, kind),
                (0, BindingKind::SampledImageArray)
                    | (1, BindingKind::SamplerArray)
                    | (3, BindingKind::StorageBufferArray)
                    | (4, BindingKind::UniformBufferArray)
            )
    });
    if bindless && !r.bindings.is_empty() {
        if legacy_only {
            return Err("uses the bindless form; a contract-2 module's programs take the per-pass one".into());
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
        if sampled > 0 && !has(3, &BindingKind::StorageBufferArray) {
            return Err("the bindless form declares its handle table at set 0 binding 3".into());
        }
        if uniform_bytes > 0 && !has(4, &BindingKind::UniformBufferArray) {
            // A std140 block at binding 3 reflects as a storage buffer; name
            // it, because that is the mistake this catches.
            if has(3, &BindingKind::StorageBufferArray) {
                return Err(
                    "the program's std140 block is at set 0 binding 3 (a storage buffer); a program block is at binding 4"
                        .into(),
                );
            }
            return Err(format!(
                "the program declares a {uniform_bytes}-byte std140 block; the bindless form declares it at set 0 binding 4 (UniformBufferArray)"
            ));
        }
        if r.push_bytes == 0 && (sampled > 0 || uniform_bytes > 0) {
            return Err(format!(
                "declares no push block; the bindless form takes up to {PUSH_BYTES} bytes (uniforms, sampler, handles)"
            ));
        }
        return Ok(Form::Bindless);
    }
    // The per-pass form (contract version 2): set 0, the block then the
    // samplers.
    if r.push_bytes > 0 {
        return Err(format!(
            "uses a {}-byte push block; the per-pass form takes a uniform block at binding 0",
            r.push_bytes
        ));
    }
    let mut want: Vec<(u32, u32, BindingKind)> = Vec::new();
    if uniform_bytes > 0 {
        want.push((0, 0, BindingKind::UniformBuffer(uniform_bytes)));
    }
    want.extend((1..=sampled as u32).map(|b| (0, b, BindingKind::CombinedImageSampler)));
    for b in &r.bindings {
        match want.iter().find(|w| (w.0, w.1) == (b.0, b.1)) {
            Some(w) if w.2 == b.2 => {}
            Some((_, _, BindingKind::UniformBuffer(declared))) if matches!(b.2, BindingKind::UniformBuffer(end) if (end..=end.next_multiple_of(16)).contains(declared)) =>
                {}
            Some((_, _, BindingKind::UniformBuffer(declared))) => {
                let BindingKind::UniformBuffer(end) = b.2 else {
                    return Err(format!(
                        "set {} binding {} is {:?}; the pass declares a uniform block",
                        b.0, b.1, b.2
                    ));
                };
                return Err(format!(
                    "the uniform block is {end} bytes ({} padded); the program declares {declared}",
                    end.next_multiple_of(16)
                ));
            }
            Some(w) => {
                return Err(format!(
                    "set {} binding {} is {:?}; the pass declares {:?}",
                    b.0, b.1, b.2, w.2
                ));
            }
            None => {
                let declared = match b.2 {
                    BindingKind::StorageBuffer | BindingKind::StorageImage => " (the pass declares no storage access)",
                    BindingKind::SampledImageArray
                    | BindingKind::SamplerArray
                    | BindingKind::StorageBufferArray
                    | BindingKind::UniformBufferArray => {
                        " (the bindless arrays cannot be mixed with the per-pass form)"
                    }
                    _ => "",
                };
                return Err(format!(
                    "set {} binding {} ({:?}) is not declared by the pass{declared}",
                    b.0, b.1, b.2
                ));
            }
        }
    }
    Ok(Form::Legacy)
}

/// The pass's sampler as each form needs it: the object for a per-pass
/// program's combined image sampler, the global index for a bindless one.
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    pub object: vk::Sampler,
    pub index: u32,
}

/// A checked fullscreen program: its pipeline and, by form, the per-pass
/// sets or the global-set handles.
pub struct FullscreenProgram {
    pub label: String,
    pub uniform_bytes: u32,
    form: Form,
    /// The pass's sampled intents (the bindless form's table size).
    sampled: usize,
    pipeline: vk::Pipeline,
    legacy: Option<Legacy>,
    bindless: Option<BindlessState>,
}

/// The per-pass form: a layout, a set per frame slot and a uniform buffer
/// per slot.
struct Legacy {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    ubos: Vec<OwnedBuffer>,
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
    handles: crate::bindless::BufferHandle,
    group: Option<crate::bindless::GroupId>,
}

impl FullscreenProgram {
    /// Reflect and check `spirv`, then build the program for its form.
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
        legacy_only: bool,
    ) -> Result<Self> {
        let reflection = reflect(spirv).with_context(|| format!("program `{label}`"))?;
        let form = check(&reflection, uniform_bytes, sampled, color_formats.len(), legacy_only)
            .map_err(|e| anyhow::anyhow!("program `{label}`: {e}"))?;
        let device = &gpu.device;
        let (mut this, group) = bindless.group(|bindless| -> Result<Self> {
            let (legacy, bindless_state) = match form {
                Form::Legacy => (Some(Legacy::new(gpu, uniform_bytes, sampled, frames_in_flight)?), None),
                Form::Bindless => (
                    None,
                    Some(BindlessState::new(
                        gpu,
                        bindless,
                        label,
                        uniform_bytes,
                        sampled.max(1),
                        frames_in_flight,
                    )?),
                ),
            };
            let layout = match (&legacy, &bindless_state) {
                (Some(legacy), _) => legacy.layout,
                _ => bindless.pipeline_layout(),
            };
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
            Ok(Self {
                label: label.into(),
                uniform_bytes,
                form,
                sampled,
                pipeline,
                legacy,
                bindless: bindless_state,
            })
        })?;
        match this.bindless.as_mut() {
            Some(state) => state.group = Some(group),
            // Nothing to retire: the group is empty.
            None => bindless.retire_group(group),
        }
        Ok(this)
    }

    /// The form the shader was checked against.
    pub fn form(&self) -> Form {
        self.form
    }

    /// This frame's uniform block for `slot` (its previous use completed).
    pub fn write_uniforms(&mut self, slot: usize, bytes: &[u8]) {
        let ubos = match (&mut self.legacy, &mut self.bindless) {
            (Some(legacy), _) => &mut legacy.ubos,
            (_, Some(state)) => &mut state.ubos,
            _ => return,
        };
        if let Some(mapped) = ubos.get_mut(slot).and_then(|u| u.mapped()) {
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
        match &self.legacy {
            Some(legacy) => {
                legacy.record(ctx, self.pipeline, images, sampler.object);
                Ok(())
            }
            None => {
                let state = self.bindless.as_ref().expect("a checked program has a form");
                let indices: Vec<u32> = images.iter().map(|image| image.sampled.unwrap_or(0)).collect();
                state.record(ctx, self.pipeline, &indices, sampler.index);
                Ok(())
            }
        }
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
        if let Some(mut legacy) = self.legacy.take() {
            legacy.destroy(gpu);
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

impl Legacy {
    fn new(gpu: &GpuDevice, uniform_bytes: u32, sampled: usize, frames_in_flight: usize) -> Result<Self> {
        let device = &gpu.device;
        let stages = vk::ShaderStageFlags::FRAGMENT;
        let mut kinds = Vec::new();
        if uniform_bytes > 0 {
            kinds.push((0u32, vk::DescriptorType::UNIFORM_BUFFER));
        }
        kinds.extend((1..=sampled as u32).map(|b| (b, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)));
        let bindings: Vec<_> = kinds
            .iter()
            .map(|&(b, ty)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(ty)
                    .descriptor_count(1)
                    .stage_flags(stages)
            })
            .collect();
        let n = frames_in_flight as u32;
        unsafe {
            let set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let layout = pipeline::pipeline_layout(device, &[set_layout], 0, vk::ShaderStageFlags::empty())?;
            let mut sizes = vec![vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: n * (sampled as u32).max(1),
            }];
            if uniform_bytes > 0 {
                sizes.push(vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: n,
                });
            }
            let pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&sizes),
                None,
            )?;
            let layouts = vec![set_layout; frames_in_flight];
            let sets = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )?;
            let ubos = if uniform_bytes > 0 {
                (0..frames_in_flight)
                    .map(|_| {
                        OwnedBuffer::new(
                            gpu,
                            BufferDesc {
                                size: uniform_bytes as u64,
                                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                            },
                            MemoryLocation::CpuToGpu,
                            "fullscreen program",
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
            for (slot, &set) in sets.iter().enumerate() {
                if let Some(ubo) = ubos.get(slot) {
                    let info = [vk::DescriptorBufferInfo {
                        buffer: ubo.buffer,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    }];
                    device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&info)],
                        &[],
                    );
                }
            }
            Ok(Self {
                set_layout,
                layout,
                pool,
                sets,
                ubos,
            })
        }
    }

    fn record(&self, ctx: &PassContext<'_>, pipeline: vk::Pipeline, images: &[ImageBinding], sampler: vk::Sampler) {
        let set = self.sets[ctx.slot];
        let infos: Vec<[vk::DescriptorImageInfo; 1]> = images
            .iter()
            .map(|i| {
                [vk::DescriptorImageInfo {
                    sampler,
                    image_view: i.view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }]
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(k, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1 + k as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(info)
            })
            .collect();
        unsafe {
            if !writes.is_empty() {
                // The slot's previous frame has completed: the set is idle.
                ctx.device.update_descriptor_sets(&writes, &[]);
            }
            ctx.device
                .cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            if !self.sets.is_empty() {
                ctx.device.cmd_bind_descriptor_sets(
                    ctx.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    &[set],
                    &[],
                );
            }
            ctx.device.cmd_draw(ctx.cmd, 3, 1, 0, 0);
        }
    }

    fn destroy(&mut self, gpu: &GpuDevice) {
        unsafe {
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device.destroy_descriptor_pool(self.pool, None);
            gpu.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
        for ubo in &mut self.ubos {
            ubo.destroy(gpu);
        }
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
        let tables = (0..frames_in_flight)
            .map(|_| {
                OwnedBuffer::new(
                    gpu,
                    BufferDesc {
                        size: (sampled * 4) as u64,
                        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
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
        let handles = bindless.slotted_buffer()?;
        for (slot, table) in tables.iter().enumerate() {
            bindless.update_slotted_buffer(handles, slot, table.buffer, 0, table.size);
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

    /// The per-pass form of `test_post.frag` (a uniform block of 28 bytes,
    /// two sampled inputs, one output) passes; every difference from what
    /// a pass declares is named.
    #[test]
    fn the_per_pass_form_is_checked_against_its_pass() {
        let ok = reflect(spirv!("test_post.frag")).unwrap();
        assert_eq!(
            ok.bindings,
            [
                (0, 0, BindingKind::UniformBuffer(28)),
                (0, 1, BindingKind::CombinedImageSampler),
                (0, 2, BindingKind::CombinedImageSampler)
            ]
        );
        assert_eq!(ok.outputs, [0]);
        assert!(ok.fragment_main && ok.push_bytes == 0);
        assert_eq!(check(&ok, 32, 2, 1, false).unwrap(), Form::Legacy);
        check(&ok, 28, 2, 1, false).unwrap();
        let e = check(&ok, 48, 2, 1, false).unwrap_err();
        assert!(e.contains("28 bytes (32 padded)") && e.contains("declares 48"), "{e}");
        assert!(check(&ok, 24, 2, 1, false).is_err());
        assert!(check(&ok, 32, 1, 1, false).unwrap_err().contains("binding 2"));
        assert!(
            check(&ok, 32, 3, 1, false).is_ok(),
            "an unused declared input is allowed"
        );
        assert!(check(&ok, 32, 2, 2, false).unwrap_err().contains("colour attachment"));
        assert!(check(&ok, 0, 2, 1, false).unwrap_err().contains("binding 0"));

        let storage = reflect(spirv!("test_post_storage.frag")).unwrap();
        let e = check(&storage, 0, 1, 1, false).unwrap_err();
        assert!(e.contains("no storage access"), "{e}");

        let pushes = reflect(spirv!("test_post_push.frag")).unwrap();
        assert_eq!(pushes.push_bytes, 16);
        assert!(check(&pushes, 0, 0, 1, false).unwrap_err().contains("push block"));

        assert!(reflect(&[1, 2, 3, 4]).is_err());
        let vertex = reflect(spirv!("fullscreen.vert")).unwrap();
        assert!(check(&vertex, 0, 0, 0, false).unwrap_err().contains("fragment"));
    }

    /// The bindless form: the global set's arrays, the handle table at 3,
    /// the std140 block at 4 and a 16-byte push block. A mix of the two
    /// forms, a block at binding 3 or a bindless shader in a contract-2
    /// module is refused with the binding named.
    #[test]
    fn the_bindless_form_is_checked_and_forms_do_not_mix() {
        let program = reflect(spirv!("test_program.frag")).unwrap();
        assert_eq!(program.push_bytes, PUSH_BYTES);
        assert_eq!(
            program.bindings,
            [
                (0, 0, BindingKind::SampledImageArray),
                (0, 1, BindingKind::SamplerArray),
                (0, 3, BindingKind::StorageBufferArray),
                (0, 4, BindingKind::UniformBufferArray),
            ]
        );
        assert_eq!(check(&program, 32, 2, 1, false).unwrap(), Form::Bindless);
        assert!(check(&program, 32, 2, 1, true).unwrap_err().contains("contract-2"));

        // A contract-2 module keeps the per-pass form; a version-3 module
        // may still use it.
        let legacy = reflect(spirv!("test_post.frag")).unwrap();
        assert_eq!(check(&legacy, 32, 2, 1, false).unwrap(), Form::Legacy);
        assert_eq!(check(&legacy, 32, 2, 1, true).unwrap(), Form::Legacy);

        // A shader that mixes the two forms names the bindless binding.
        let mut mixed = program.clone();
        mixed.bindings.push((0, 2, BindingKind::CombinedImageSampler));
        mixed.bindings.sort_by_key(|b| (b.0, b.1));
        let e = check(&mixed, 32, 2, 1, false).unwrap_err();
        assert!(e.contains("mixes") && e.contains("binding 0"), "{e}");

        // A bindless shader that declares no table cannot reach its images.
        let mut missing = program.clone();
        missing.bindings.retain(|(_, binding, _)| *binding != 3);
        let e = check(&missing, 32, 2, 1, false).unwrap_err();
        assert!(e.contains("binding 3"), "{e}");

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
        let e = check(&storage_block, 32, 2, 1, false).unwrap_err();
        assert!(e.contains("binding 3") && e.contains("binding 4"), "{e}");

        // A program with no block needs no binding 4.
        let mut no_block = program.clone();
        no_block.bindings.retain(|(_, binding, _)| *binding != 4);
        assert_eq!(check(&no_block, 0, 2, 1, false).unwrap(), Form::Bindless);
        assert!(check(&no_block, 32, 2, 1, false).is_err());

        // The push block is at most 16 bytes.
        let mut wide = program.clone();
        wide.push_bytes = 32;
        assert!(check(&wide, 32, 2, 1, false).unwrap_err().contains("at most 16"));
    }
}
