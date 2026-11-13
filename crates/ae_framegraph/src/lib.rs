//! Single-queue, bake-then-execute frame graph (docs/TASK-frame-graph.md §3).
//!
//! Declarations are typed and arena-backed: `ResourceId`/`PassId` index into
//! `Vec`s, there is no string lookup, no `RefCell` and no boxed component.
//! A pass declares *what* it does with each resource (`Role`); the Vulkan
//! layout, stage and access masks are derived from the role in one match
//! (`Role::usage`). Baking (`bake`) is a pure function of the declarations:
//! order, lifetimes, alias groups, rendering-scope groups and the declarative
//! accesses that the barrier state machine (`barriers`) turns into concrete
//! barriers at record time.
//!
//! This module depends on `ash` for Vulkan *types* only. It never calls
//! Vulkan, and it knows nothing about SDL, the simulation, the ECS, or the
//! presentation slots, so all of it is testable without a GPU.

pub mod bake;
pub mod barriers;
pub mod dot;

pub use bake::{
    AliasHandoff, BakeError, BakeOptions, BakedAccess, BakedGraph, BakedPass, BakedResource, Edge,
    PhysicalResource, bake,
};
pub use barriers::{AccessState, BarrierDesc, StateKey, StateStore, pass_barriers};

use ash::vk;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PassId(pub u32);

impl ResourceId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl PassId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Leak-rule namespace (§4.2). Resources and passes belong to one view
/// scope; `Scope::GLOBAL` is host-shared, read-only data (atlases, constant
/// tables). A pass may write only resources of its own scope and may read its
/// own scope or global resources that no scoped pass writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Scope(pub u32);

impl Scope {
    pub const GLOBAL: Scope = Scope(0);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageDesc {
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub usage: vk::ImageUsageFlags,
    pub mip_levels: u32,
    pub layers: u32,
    pub samples: vk::SampleCountFlags,
    pub aspect: vk::ImageAspectFlags,
}

impl ImageDesc {
    /// A single-mip, single-layer, single-sample colour image.
    pub fn color_2d(format: vk::Format, width: u32, height: u32, usage: vk::ImageUsageFlags) -> Self {
        Self {
            format,
            width,
            height,
            usage,
            mip_levels: 1,
            layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            aspect: vk::ImageAspectFlags::COLOR,
        }
    }

    pub fn depth_2d(format: vk::Format, width: u32, height: u32, usage: vk::ImageUsageFlags) -> Self {
        Self {
            aspect: vk::ImageAspectFlags::DEPTH,
            ..Self::color_2d(format, width, height, usage)
        }
    }

    pub fn extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceDesc {
    Image(ImageDesc),
    Buffer(BufferDesc),
}

impl ResourceDesc {
    pub fn is_image(&self) -> bool {
        matches!(self, ResourceDesc::Image(_))
    }
}

/// Who owns the contents and the allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceClass {
    /// Graph-owned, contents live within one frame; eligible for aliasing.
    /// The first access must be a discarding write.
    Transient,
    /// Graph-owned, contents survive across frames (per frame slot).
    Persistent,
    /// Owned outside the graph and bound per frame (offscreen targets,
    /// readback buffers, uploaded instance data).
    External,
    /// The presented swapchain image, bound per frame.
    Backbuffer,
}

#[derive(Clone, Debug)]
pub struct Resource {
    pub name: String,
    pub desc: ResourceDesc,
    pub class: ResourceClass,
    pub scope: Scope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PassKind {
    /// Recorded inside one dynamic-rendering scope.
    Graphics,
    Compute,
    Transfer,
    /// Transition of the backbuffer to `PRESENT_SRC_KHR`; records nothing.
    Present,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    /// Vertex and fragment.
    Graphics,
    Compute,
}

impl ShaderStage {
    fn stages(self) -> vk::PipelineStageFlags2 {
        match self {
            ShaderStage::Vertex => vk::PipelineStageFlags2::VERTEX_SHADER,
            ShaderStage::Fragment => vk::PipelineStageFlags2::FRAGMENT_SHADER,
            ShaderStage::Graphics => {
                vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER
            }
            ShaderStage::Compute => vk::PipelineStageFlags2::COMPUTE_SHADER,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoadOp {
    Clear,
    Load,
    DontCare,
}

impl LoadOp {
    pub fn vk(self) -> vk::AttachmentLoadOp {
        match self {
            LoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
            LoadOp::Load => vk::AttachmentLoadOp::LOAD,
            LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
        }
    }
}

/// What a pass does with a resource. The role alone decides layout, stage and
/// access masks, whether the access writes, and which usage flags the
/// resource must have been created with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// No depth role yet: the executor and pipelines build colour
    /// attachments only, so a depth intent would bake and render nothing.
    ColorAttachment(LoadOp),
    Sampled(ShaderStage),
    StorageImageRead(ShaderStage),
    StorageImageWrite(ShaderStage),
    StorageImageReadWrite(ShaderStage),
    StorageBufferRead(ShaderStage),
    StorageBufferWrite(ShaderStage),
    UniformBuffer(ShaderStage),
    VertexBuffer,
    IndexBuffer,
    IndirectBuffer,
    TransferSrc,
    TransferDst,
    /// The host reads the buffer after the frame's fence (readback). Only
    /// in a transfer pass whose every intent is a host read (it records
    /// nothing; its barrier makes the earlier transfer write host-visible).
    HostRead,
    Present,
}

/// Derived synchronization usage of one access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Usage {
    /// `UNDEFINED` for buffers.
    pub layout: vk::ImageLayout,
    pub stages: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub write: bool,
}

impl Role {
    pub fn usage(self) -> Usage {
        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        let make = |layout, stages, access, write| Usage {
            layout,
            stages,
            access,
            write,
        };
        match self {
            Role::ColorAttachment(load) => make(
                L::COLOR_ATTACHMENT_OPTIMAL,
                S::COLOR_ATTACHMENT_OUTPUT,
                if load == LoadOp::Load {
                    A::COLOR_ATTACHMENT_READ | A::COLOR_ATTACHMENT_WRITE
                } else {
                    A::COLOR_ATTACHMENT_WRITE
                },
                true,
            ),
            Role::Sampled(stage) => make(L::SHADER_READ_ONLY_OPTIMAL, stage.stages(), A::SHADER_SAMPLED_READ, false),
            Role::StorageImageRead(stage) => make(L::GENERAL, stage.stages(), A::SHADER_STORAGE_READ, false),
            Role::StorageImageWrite(stage) => make(L::GENERAL, stage.stages(), A::SHADER_STORAGE_WRITE, true),
            Role::StorageImageReadWrite(stage) => make(
                L::GENERAL,
                stage.stages(),
                A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
                true,
            ),
            Role::StorageBufferRead(stage) => make(L::UNDEFINED, stage.stages(), A::SHADER_STORAGE_READ, false),
            Role::StorageBufferWrite(stage) => make(L::UNDEFINED, stage.stages(), A::SHADER_STORAGE_WRITE, true),
            Role::UniformBuffer(stage) => make(L::UNDEFINED, stage.stages(), A::UNIFORM_READ, false),
            Role::VertexBuffer => make(L::UNDEFINED, S::VERTEX_ATTRIBUTE_INPUT, A::VERTEX_ATTRIBUTE_READ, false),
            Role::IndexBuffer => make(L::UNDEFINED, S::INDEX_INPUT, A::INDEX_READ, false),
            Role::IndirectBuffer => make(L::UNDEFINED, S::DRAW_INDIRECT, A::INDIRECT_COMMAND_READ, false),
            Role::TransferSrc => make(L::TRANSFER_SRC_OPTIMAL, S::ALL_TRANSFER, A::TRANSFER_READ, false),
            Role::TransferDst => make(L::TRANSFER_DST_OPTIMAL, S::ALL_TRANSFER, A::TRANSFER_WRITE, true),
            Role::HostRead => make(L::UNDEFINED, S::HOST, A::HOST_READ, false),
            Role::Present => make(L::PRESENT_SRC_KHR, S::NONE, A::NONE, false),
        }
    }

    /// Image roles take image layouts; transfer roles apply to both kinds.
    pub fn applies_to(self, desc: &ResourceDesc) -> bool {
        match self {
            Role::ColorAttachment(_)
            | Role::Sampled(_)
            | Role::StorageImageRead(_)
            | Role::StorageImageWrite(_)
            | Role::StorageImageReadWrite(_)
            | Role::Present => desc.is_image(),
            Role::StorageBufferRead(_)
            | Role::StorageBufferWrite(_)
            | Role::UniformBuffer(_)
            | Role::VertexBuffer
            | Role::IndexBuffer
            | Role::IndirectBuffer
            | Role::HostRead => !desc.is_image(),
            Role::TransferSrc | Role::TransferDst => true,
        }
    }

    /// Usage flags the resource must have been created with (`None`: no
    /// requirement, e.g. presenting).
    pub fn missing_usage(self, desc: &ResourceDesc) -> Option<String> {
        match *desc {
            ResourceDesc::Image(image) => {
                use vk::ImageUsageFlags as U;
                let required = match self {
                    Role::ColorAttachment(_) => U::COLOR_ATTACHMENT,
                    Role::Sampled(_) => U::SAMPLED,
                    Role::StorageImageRead(_) | Role::StorageImageWrite(_) | Role::StorageImageReadWrite(_) => {
                        U::STORAGE
                    }
                    Role::TransferSrc => U::TRANSFER_SRC,
                    Role::TransferDst => U::TRANSFER_DST,
                    _ => U::empty(),
                };
                (!image.usage.contains(required)).then(|| format!("{required:?}"))
            }
            ResourceDesc::Buffer(buffer) => {
                use vk::BufferUsageFlags as U;
                let required = match self {
                    Role::StorageBufferRead(_) | Role::StorageBufferWrite(_) => U::STORAGE_BUFFER,
                    Role::UniformBuffer(_) => U::UNIFORM_BUFFER,
                    Role::VertexBuffer => U::VERTEX_BUFFER,
                    Role::IndexBuffer => U::INDEX_BUFFER,
                    Role::IndirectBuffer => U::INDIRECT_BUFFER,
                    Role::TransferSrc => U::TRANSFER_SRC,
                    Role::TransferDst => U::TRANSFER_DST,
                    _ => U::empty(),
                };
                (!buffer.usage.contains(required)).then(|| format!("{required:?}"))
            }
        }
    }

    pub fn is_attachment(self) -> bool {
        matches!(self, Role::ColorAttachment(_))
    }

    /// The access replaces the previous contents entirely.
    pub fn implies_discard(self) -> bool {
        matches!(self, Role::ColorAttachment(LoadOp::Clear | LoadOp::DontCare))
    }

    pub fn load_op(self) -> Option<LoadOp> {
        match self {
            Role::ColorAttachment(load) => Some(load),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Intent {
    pub resource: ResourceId,
    pub role: Role,
    /// Previous contents are not needed (old layout `UNDEFINED`). Implied by
    /// a clearing or don't-care attachment load.
    pub discard: bool,
}

#[derive(Clone, Debug)]
pub struct Pass<R> {
    pub name: String,
    pub kind: PassKind,
    pub scope: Scope,
    pub intents: Vec<Intent>,
    /// May be skipped at record time. A skipped pass keeps its edges in the
    /// bake; only its recording and barriers are omitted that frame.
    pub conditional: bool,
    /// Executor payload (what to record); ignored by the bake.
    pub record: R,
}

impl<R> Pass<R> {
    /// Colour attachments in declaration order.
    pub fn color_attachments(&self) -> impl Iterator<Item = &Intent> {
        self.intents
            .iter()
            .filter(|intent| matches!(intent.role, Role::ColorAttachment(_)))
    }
}

/// The declarations of one frame topology. Cleared and refilled, not
/// reallocated, when the topology changes (§3.4).
#[derive(Clone, Debug)]
pub struct FrameGraph<R = ()> {
    pub resources: Vec<Resource>,
    pub passes: Vec<Pass<R>>,
    /// Output roots (present or store passes). Reachability from the roots
    /// decides what runs.
    pub roots: Vec<PassId>,
    /// Caller-provided version folded into the topology signature (e.g.
    /// per-view render plan versions).
    pub plan_version: u64,
}

impl<R> Default for FrameGraph<R> {
    fn default() -> Self {
        Self {
            resources: Vec::new(),
            passes: Vec::new(),
            roots: Vec::new(),
            plan_version: 0,
        }
    }
}

impl<R> FrameGraph<R> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop all declarations, keeping the allocations.
    pub fn clear(&mut self) {
        self.resources.clear();
        self.passes.clear();
        self.roots.clear();
        self.plan_version = 0;
    }

    pub fn image(&mut self, name: impl Into<String>, desc: ImageDesc, class: ResourceClass) -> ResourceId {
        self.resource(name, ResourceDesc::Image(desc), class, Scope::GLOBAL)
    }

    pub fn buffer(&mut self, name: impl Into<String>, desc: BufferDesc, class: ResourceClass) -> ResourceId {
        self.resource(name, ResourceDesc::Buffer(desc), class, Scope::GLOBAL)
    }

    pub fn resource(
        &mut self,
        name: impl Into<String>,
        desc: ResourceDesc,
        class: ResourceClass,
        scope: Scope,
    ) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(Resource {
            name: name.into(),
            desc,
            class,
            scope,
        });
        id
    }

    pub fn set_scope(&mut self, resource: ResourceId, scope: Scope) {
        self.resources[resource.index()].scope = scope;
    }

    pub fn pass(&mut self, name: impl Into<String>, kind: PassKind, record: R) -> PassBuilder<'_, R> {
        let id = PassId(self.passes.len() as u32);
        self.passes.push(Pass {
            name: name.into(),
            kind,
            scope: Scope::GLOBAL,
            intents: Vec::new(),
            conditional: false,
            record,
        });
        PassBuilder { graph: self, id }
    }

    pub fn add_root(&mut self, pass: PassId) {
        self.roots.push(pass);
    }

    pub fn resource_name(&self, resource: ResourceId) -> &str {
        self.resources
            .get(resource.index())
            .map(|resource| resource.name.as_str())
            .unwrap_or("<unknown resource>")
    }

    pub fn pass_name(&self, pass: PassId) -> &str {
        self.passes
            .get(pass.index())
            .map(|pass| pass.name.as_str())
            .unwrap_or("<unknown pass>")
    }

    /// Topology signature: everything that shapes the bake (kinds, intents,
    /// resource descriptions, classes, scopes, roots, conditions, plan
    /// version), and nothing else. Names are excluded.
    pub fn signature(&self) -> u64 {
        let mut hash = Fnv64::new();
        hash.u64(self.plan_version);
        hash.u64(self.resources.len() as u64);
        for resource in &self.resources {
            match resource.desc {
                ResourceDesc::Image(image) => {
                    hash.u64(1);
                    hash.u64(image.format.as_raw() as u64);
                    hash.u64(image.width as u64);
                    hash.u64(image.height as u64);
                    hash.u64(image.usage.as_raw() as u64);
                    hash.u64(image.mip_levels as u64);
                    hash.u64(image.layers as u64);
                    hash.u64(image.samples.as_raw() as u64);
                    hash.u64(image.aspect.as_raw() as u64);
                }
                ResourceDesc::Buffer(buffer) => {
                    hash.u64(2);
                    hash.u64(buffer.size);
                    hash.u64(buffer.usage.as_raw() as u64);
                }
            }
            hash.u64(resource.class as u64);
            hash.u64(resource.scope.0 as u64);
        }
        hash.u64(self.passes.len() as u64);
        for pass in &self.passes {
            hash.u64(pass.kind as u64);
            hash.u64(pass.scope.0 as u64);
            hash.u64(pass.conditional as u64);
            hash.u64(pass.intents.len() as u64);
            for intent in &pass.intents {
                hash.u64(intent.resource.0 as u64);
                hash.bytes(format!("{:?}", intent.role).as_bytes());
                hash.u64(intent.discard as u64);
            }
        }
        for root in &self.roots {
            hash.u64(root.0 as u64);
        }
        hash.finish()
    }
}

/// Fluent pass declaration. Every method appends one intent.
pub struct PassBuilder<'a, R> {
    graph: &'a mut FrameGraph<R>,
    id: PassId,
}

impl<R> PassBuilder<'_, R> {
    fn intent(self, resource: ResourceId, role: Role, discard: bool) -> Self {
        let pass = &mut self.graph.passes[self.id.index()];
        pass.intents.push(Intent {
            resource,
            role,
            discard: discard || role.implies_discard(),
        });
        self
    }

    /// Access without discarding previous contents.
    pub fn access(self, resource: ResourceId, role: Role) -> Self {
        self.intent(resource, role, false)
    }

    /// A write that replaces the previous contents (old layout `UNDEFINED`).
    pub fn discard(self, resource: ResourceId, role: Role) -> Self {
        self.intent(resource, role, true)
    }

    pub fn color(self, resource: ResourceId, load: LoadOp) -> Self {
        self.access(resource, Role::ColorAttachment(load))
    }

    pub fn sampled(self, resource: ResourceId, stage: ShaderStage) -> Self {
        self.access(resource, Role::Sampled(stage))
    }

    pub fn present(self, resource: ResourceId) -> Self {
        self.access(resource, Role::Present)
    }

    pub fn scope(self, scope: Scope) -> Self {
        self.graph.passes[self.id.index()].scope = scope;
        self
    }

    pub fn conditional(self) -> Self {
        self.conditional_if(true)
    }

    pub fn conditional_if(self, conditional: bool) -> Self {
        self.graph.passes[self.id.index()].conditional |= conditional;
        self
    }

    pub fn root(self) -> Self {
        self.graph.roots.push(self.id);
        self
    }

    pub fn id(self) -> PassId {
        self.id
    }
}

/// FNV-1a 64 (deterministic across runs and platforms).
pub(crate) struct Fnv64(u64);

impl Fnv64 {
    pub(crate) fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    pub(crate) fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= *byte as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    pub(crate) fn finish(&self) -> u64 {
        self.0
    }
}
