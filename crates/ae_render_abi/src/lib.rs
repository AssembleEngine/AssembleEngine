//! Frame graph declaration ABI between a render module (WASM guest) and the
//! host.
//!
//! The render module describes resources and passes with [`GraphBuilder`]
//! and hands the host one byte buffer. It does this at start-up and whenever
//! it wants a different topology; per-frame data travels separately. The
//! host validates and decodes the buffer (`ae_framegraph::descriptor`), then
//! bakes it and rebakes only when the topology signature changes.
//!
//! # Wire format (version 1)
//!
//! All words are little-endian `u32`.
//!
//! | Section | Words | Content |
//! |---|---|---|
//! | header | 8 | magic, version, plan version (lo, hi), resource count, pass count, intent count, name bytes |
//! | resources | 13 each | kind, class, scope, name offset, name length, 8 parameters |
//! | passes | 8 each | kind, flags, scope, name offset, name length, first intent, intent count, payload |
//! | intents | 4 each | resource, role, argument, discard |
//! | names | bytes | UTF-8, referenced by offset and length |
//!
//! Image parameters: format, width, height, usage, mip levels, layers,
//! samples, aspect (raw Vulkan values). Buffer parameters: size (lo, hi),
//! usage, then zeros. A pass's intents are contiguous and passes list them
//! in order. `payload` is opaque to the graph: the render module's own pass
//! handle, returned to it when the pass is recorded.

pub const MAGIC: u32 = u32::from_le_bytes(*b"AERG");
pub const VERSION: u32 = 1;

pub const HEADER_WORDS: usize = 8;
pub const RESOURCE_WORDS: usize = 13;
pub const PASS_WORDS: usize = 8;
pub const INTENT_WORDS: usize = 4;

pub const MAX_RESOURCES: usize = 8192;
pub const MAX_PASSES: usize = 8192;
pub const MAX_INTENTS: usize = 1 << 16;
pub const MAX_NAME_BYTES: usize = 1 << 20;

pub const RESOURCE_IMAGE: u32 = 0;
pub const RESOURCE_BUFFER: u32 = 1;

pub const PASS_FLAG_ROOT: u32 = 1 << 0;
pub const PASS_FLAG_CONDITIONAL: u32 = 1 << 1;

macro_rules! wire_enum {
    ($(#[$doc:meta])* $name:ident { $($variant:ident = $value:literal),* $(,)? }) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u32)]
        pub enum $name { $($variant = $value),* }

        impl $name {
            pub fn from_wire(value: u32) -> Option<Self> {
                match value {
                    $($value => Some(Self::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

wire_enum!(
    /// Who owns a resource's contents and allocation.
    Class { Transient = 0, Persistent = 1, External = 2, Backbuffer = 3 }
);
wire_enum!(PassKind { Graphics = 0, Compute = 1, Transfer = 2, Present = 3 });
wire_enum!(ShaderStage { Vertex = 0, Fragment = 1, Graphics = 2, Compute = 3 });
wire_enum!(LoadOp { Clear = 0, Load = 1, DontCare = 2 });
wire_enum!(
    /// What a pass does with a resource. Roles marked `(stage)` take a
    /// [`ShaderStage`] argument, `ColorAttachment` takes a [`LoadOp`]; the
    /// rest take 0.
    RoleKind {
        ColorAttachment = 0,
        Sampled = 1,
        StorageImageRead = 2,
        StorageImageWrite = 3,
        StorageImageReadWrite = 4,
        StorageBufferRead = 5,
        StorageBufferWrite = 6,
        UniformBuffer = 7,
        VertexBuffer = 8,
        IndexBuffer = 9,
        IndirectBuffer = 10,
        TransferSrc = 11,
        TransferDst = 12,
        HostRead = 13,
        Present = 14,
    }
);

impl RoleKind {
    pub fn takes_stage(self) -> bool {
        use RoleKind::*;
        matches!(self, Sampled | StorageImageRead | StorageImageWrite | StorageImageReadWrite | StorageBufferRead | StorageBufferWrite | UniformBuffer)
    }
}

/// A role with its argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
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
    HostRead,
    Present,
}

impl Role {
    pub fn to_wire(self) -> (u32, u32) {
        use Role::*;
        match self {
            ColorAttachment(load) => (RoleKind::ColorAttachment as u32, load as u32),
            Sampled(s) => (RoleKind::Sampled as u32, s as u32),
            StorageImageRead(s) => (RoleKind::StorageImageRead as u32, s as u32),
            StorageImageWrite(s) => (RoleKind::StorageImageWrite as u32, s as u32),
            StorageImageReadWrite(s) => (RoleKind::StorageImageReadWrite as u32, s as u32),
            StorageBufferRead(s) => (RoleKind::StorageBufferRead as u32, s as u32),
            StorageBufferWrite(s) => (RoleKind::StorageBufferWrite as u32, s as u32),
            UniformBuffer(s) => (RoleKind::UniformBuffer as u32, s as u32),
            VertexBuffer => (RoleKind::VertexBuffer as u32, 0),
            IndexBuffer => (RoleKind::IndexBuffer as u32, 0),
            IndirectBuffer => (RoleKind::IndirectBuffer as u32, 0),
            TransferSrc => (RoleKind::TransferSrc as u32, 0),
            TransferDst => (RoleKind::TransferDst as u32, 0),
            HostRead => (RoleKind::HostRead as u32, 0),
            Present => (RoleKind::Present as u32, 0),
        }
    }

    /// The access replaces previous contents (mirrors the host rule).
    pub fn implies_discard(self) -> bool {
        matches!(self, Role::ColorAttachment(LoadOp::Clear | LoadOp::DontCare))
    }
}

/// Image description in raw Vulkan values (`VkFormat`, `VkImageUsageFlags`,
/// `VkSampleCountFlags`, `VkImageAspectFlags`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSpec {
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub usage: u32,
    pub mip_levels: u32,
    pub layers: u32,
    pub samples: u32,
    pub aspect: u32,
}

impl ImageSpec {
    /// Single mip, single layer, single sample, colour aspect.
    pub fn color_2d(format: u32, width: u32, height: u32, usage: u32) -> Self {
        Self { format, width, height, usage, mip_levels: 1, layers: 1, samples: 1, aspect: 1 }
    }
}

/// Handle to a declared resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceRef(pub u32);

/// Handle to a declared pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassRef(pub u32);

/// Writes a frame graph declaration.
#[derive(Default)]
pub struct GraphBuilder {
    plan_version: u64,
    resources: Vec<[u32; RESOURCE_WORDS]>,
    passes: Vec<[u32; PASS_WORDS]>,
    intents: Vec<[u32; INTENT_WORDS]>,
    names: Vec<u8>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Folded into the topology signature: bump it to force a rebake.
    pub fn plan_version(&mut self, version: u64) -> &mut Self {
        self.plan_version = version;
        self
    }

    fn name(&mut self, name: &str) -> (u32, u32) {
        let offset = self.names.len() as u32;
        self.names.extend_from_slice(name.as_bytes());
        (offset, name.len() as u32)
    }

    pub fn image(&mut self, name: &str, spec: ImageSpec, class: Class) -> ResourceRef {
        self.image_scoped(name, spec, class, 0)
    }

    pub fn image_scoped(&mut self, name: &str, spec: ImageSpec, class: Class, scope: u32) -> ResourceRef {
        let (offset, len) = self.name(name);
        let s = spec;
        self.resources.push([RESOURCE_IMAGE, class as u32, scope, offset, len, s.format, s.width, s.height, s.usage, s.mip_levels, s.layers, s.samples, s.aspect]);
        ResourceRef(self.resources.len() as u32 - 1)
    }

    pub fn buffer(&mut self, name: &str, size: u64, usage: u32, class: Class) -> ResourceRef {
        self.buffer_scoped(name, size, usage, class, 0)
    }

    pub fn buffer_scoped(&mut self, name: &str, size: u64, usage: u32, class: Class, scope: u32) -> ResourceRef {
        let (offset, len) = self.name(name);
        self.resources.push([RESOURCE_BUFFER, class as u32, scope, offset, len, size as u32, (size >> 32) as u32, usage, 0, 0, 0, 0, 0]);
        ResourceRef(self.resources.len() as u32 - 1)
    }

    /// Start a pass. `payload` is returned to the module when the pass is recorded.
    pub fn pass(&mut self, name: &str, kind: PassKind, payload: u32) -> PassWriter<'_> {
        let (offset, len) = self.name(name);
        let first = self.intents.len() as u32;
        self.passes.push([kind as u32, 0, 0, offset, len, first, 0, payload]);
        let index = self.passes.len() - 1;
        PassWriter { builder: self, index }
    }

    /// The finished declaration.
    pub fn finish(&self) -> Vec<u8> {
        let header = [
            MAGIC,
            VERSION,
            self.plan_version as u32,
            (self.plan_version >> 32) as u32,
            self.resources.len() as u32,
            self.passes.len() as u32,
            self.intents.len() as u32,
            self.names.len() as u32,
        ];
        let words = header.len() + self.resources.len() * RESOURCE_WORDS + self.passes.len() * PASS_WORDS + self.intents.len() * INTENT_WORDS;
        let mut out = Vec::with_capacity(words * 4 + self.names.len());
        let mut put = |w: &u32| out.extend_from_slice(&w.to_le_bytes());
        header.iter().for_each(&mut put);
        self.resources.iter().flatten().for_each(&mut put);
        self.passes.iter().flatten().for_each(&mut put);
        self.intents.iter().flatten().for_each(&mut put);
        out.extend_from_slice(&self.names);
        out
    }
}

/// Declares one pass's intents. Each method appends one intent.
pub struct PassWriter<'a> {
    builder: &'a mut GraphBuilder,
    index: usize,
}

impl PassWriter<'_> {
    fn intent(self, resource: ResourceRef, role: Role, discard: bool) -> Self {
        let (role, arg) = role.to_wire();
        self.builder.intents.push([resource.0, role, arg, discard as u32]);
        self.builder.passes[self.index][6] += 1;
        self
    }

    /// Use a resource, keeping its contents (unless the role implies otherwise).
    pub fn access(self, resource: ResourceRef, role: Role) -> Self {
        let discard = role.implies_discard();
        self.intent(resource, role, discard)
    }

    /// Use a resource whose previous contents are not needed.
    pub fn discard(self, resource: ResourceRef, role: Role) -> Self {
        self.intent(resource, role, true)
    }

    pub fn color(self, resource: ResourceRef, load: LoadOp) -> Self {
        self.access(resource, Role::ColorAttachment(load))
    }

    pub fn sampled(self, resource: ResourceRef, stage: ShaderStage) -> Self {
        self.access(resource, Role::Sampled(stage))
    }

    pub fn present(self, resource: ResourceRef) -> Self {
        self.access(resource, Role::Present)
    }

    pub fn scope(self, scope: u32) -> Self {
        self.builder.passes[self.index][2] = scope;
        self
    }

    pub fn conditional(self) -> Self {
        self.builder.passes[self.index][1] |= PASS_FLAG_CONDITIONAL;
        self
    }

    pub fn root(self) -> Self {
        self.builder.passes[self.index][1] |= PASS_FLAG_ROOT;
        self
    }

    pub fn id(self) -> PassRef {
        PassRef(self.index as u32)
    }
}
