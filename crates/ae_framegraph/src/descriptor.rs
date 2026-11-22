//! Host side of the frame graph declaration ABI (`ae_render_abi`): validate
//! and decode a render module's declaration into a [`FrameGraph`], and the
//! reverse (used by tests and benches).
//!
//! Decoding checks the wire format only: bounds, counts, enum values,
//! names, resource references. Whether the graph makes sense (roles against
//! usage flags, single writers, cycles, scopes) is checked by [`crate::bake`],
//! exactly as for a natively declared graph.

use crate::{
    BufferDesc, FrameGraph, ImageDesc, Intent, LoadOp, Pass, PassId, PassKind, ResourceClass, ResourceDesc, ResourceId,
    Role, Scope, ShaderStage,
};
use ae_render_abi as abi;
use ash::vk;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DescriptorError {
    Truncated {
        needed: usize,
        got: usize,
    },
    TrailingBytes(usize),
    BadMagic(u32),
    UnsupportedVersion(u32),
    TooMany {
        what: &'static str,
        count: usize,
        max: usize,
    },
    BadEnum {
        what: &'static str,
        index: usize,
        value: u32,
    },
    BadName {
        what: &'static str,
        index: usize,
    },
    BadResource {
        pass: usize,
        resource: u32,
    },
    BadIntentRange {
        pass: usize,
    },
    BadImage {
        index: usize,
        reason: &'static str,
    },
}

impl fmt::Display for DescriptorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { needed, got } => write!(f, "declaration truncated: needs {needed} bytes, got {got}"),
            Self::TrailingBytes(n) => write!(f, "{n} unexpected bytes after the declaration"),
            Self::BadMagic(m) => write!(f, "bad magic {m:#010x}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported declaration version {v}"),
            Self::TooMany { what, count, max } => write!(f, "{count} {what} (at most {max})"),
            Self::BadEnum { what, index, value } => write!(f, "{what} {index}: invalid value {value}"),
            Self::BadName { what, index } => write!(f, "{what} {index}: name out of range or not UTF-8"),
            Self::BadResource { pass, resource } => write!(f, "pass {pass}: no resource {resource}"),
            Self::BadIntentRange { pass } => write!(f, "pass {pass}: intents are not contiguous and in order"),
            Self::BadImage { index, reason } => write!(f, "resource {index}: {reason}"),
        }
    }
}

impl std::error::Error for DescriptorError {}

const MAX_IMAGE_EXTENT: u32 = 1 << 15;

struct Words<'a>(&'a [u8]);

impl Words<'_> {
    fn at(&self, word: usize) -> u32 {
        u32::from_le_bytes(self.0[word * 4..word * 4 + 4].try_into().unwrap())
    }
}

fn stage(value: u32, what: &'static str, index: usize) -> Result<ShaderStage, DescriptorError> {
    Ok(
        match abi::ShaderStage::from_wire(value).ok_or(DescriptorError::BadEnum { what, index, value })? {
            abi::ShaderStage::Vertex => ShaderStage::Vertex,
            abi::ShaderStage::Fragment => ShaderStage::Fragment,
            abi::ShaderStage::Graphics => ShaderStage::Graphics,
            abi::ShaderStage::Compute => ShaderStage::Compute,
        },
    )
}

fn role(kind: u32, arg: u32, index: usize) -> Result<Role, DescriptorError> {
    use abi::RoleKind as K;
    let kind = K::from_wire(kind).ok_or(DescriptorError::BadEnum {
        what: "intent role",
        index,
        value: kind,
    })?;
    let s = |arg| stage(arg, "intent shader stage", index);
    let no_arg = |role: Role| {
        if arg == 0 {
            Ok(role)
        } else {
            Err(DescriptorError::BadEnum {
                what: "intent argument",
                index,
                value: arg,
            })
        }
    };
    let load = || {
        Ok(
            match abi::LoadOp::from_wire(arg).ok_or(DescriptorError::BadEnum {
                what: "intent load op",
                index,
                value: arg,
            })? {
                abi::LoadOp::Clear => LoadOp::Clear,
                abi::LoadOp::Load => LoadOp::Load,
                abi::LoadOp::DontCare => LoadOp::DontCare,
            },
        )
    };
    match kind {
        K::ColorAttachment => Ok(Role::ColorAttachment(load()?)),
        K::DepthAttachment => Ok(Role::DepthAttachment(load()?)),
        K::DepthRead => no_arg(Role::DepthRead),
        K::Sampled => Ok(Role::Sampled(s(arg)?)),
        K::StorageImageRead => Ok(Role::StorageImageRead(s(arg)?)),
        K::StorageImageWrite => Ok(Role::StorageImageWrite(s(arg)?)),
        K::StorageImageReadWrite => Ok(Role::StorageImageReadWrite(s(arg)?)),
        K::StorageBufferRead => Ok(Role::StorageBufferRead(s(arg)?)),
        K::StorageBufferWrite => Ok(Role::StorageBufferWrite(s(arg)?)),
        K::StorageBufferReadWrite => Ok(Role::StorageBufferReadWrite(s(arg)?)),
        K::UniformBuffer => Ok(Role::UniformBuffer(s(arg)?)),
        K::VertexBuffer => no_arg(Role::VertexBuffer),
        K::IndexBuffer => no_arg(Role::IndexBuffer),
        K::IndirectBuffer => no_arg(Role::IndirectBuffer),
        K::TransferSrc => no_arg(Role::TransferSrc),
        K::TransferDst => no_arg(Role::TransferDst),
        K::HostRead => no_arg(Role::HostRead),
        K::Present => no_arg(Role::Present),
    }
}

/// Validate and decode a declaration. Pass payloads become the graph's
/// per-pass record type.
pub fn decode(bytes: &[u8]) -> Result<FrameGraph<u32>, DescriptorError> {
    use DescriptorError as E;
    let header_bytes = abi::HEADER_WORDS * 4;
    if bytes.len() < header_bytes {
        return Err(E::Truncated {
            needed: header_bytes,
            got: bytes.len(),
        });
    }
    let w = Words(bytes);
    if w.at(0) != abi::MAGIC {
        return Err(E::BadMagic(w.at(0)));
    }
    if w.at(1) != abi::VERSION {
        return Err(E::UnsupportedVersion(w.at(1)));
    }
    let plan_version = w.at(2) as u64 | (w.at(3) as u64) << 32;
    let counts = [
        ("resources", w.at(4) as usize, abi::MAX_RESOURCES),
        ("passes", w.at(5) as usize, abi::MAX_PASSES),
        ("intents", w.at(6) as usize, abi::MAX_INTENTS),
        ("name bytes", w.at(7) as usize, abi::MAX_NAME_BYTES),
    ];
    for (what, count, max) in counts {
        if count > max {
            return Err(E::TooMany { what, count, max });
        }
    }
    let [(_, resources, _), (_, passes, _), (_, intents, _), (_, name_bytes, _)] = counts;
    let resource_base = abi::HEADER_WORDS;
    let pass_base = resource_base + resources * abi::RESOURCE_WORDS;
    let intent_base = pass_base + passes * abi::PASS_WORDS;
    let names_start = (intent_base + intents * abi::INTENT_WORDS) * 4;
    let total = names_start + name_bytes;
    if bytes.len() < total {
        return Err(E::Truncated {
            needed: total,
            got: bytes.len(),
        });
    }
    if bytes.len() > total {
        return Err(E::TrailingBytes(bytes.len() - total));
    }
    let names = &bytes[names_start..];
    let name = |offset: u32, len: u32, what: &'static str, index: usize| -> Result<String, DescriptorError> {
        let (offset, len) = (offset as usize, len as usize);
        names
            .get(offset..offset.checked_add(len).ok_or(E::BadName { what, index })?)
            .and_then(|slice| std::str::from_utf8(slice).ok())
            .map(str::to_string)
            .ok_or(E::BadName { what, index })
    };

    let mut graph = FrameGraph::<u32>::new();
    graph.plan_version = plan_version;
    for index in 0..resources {
        let r = |k: usize| w.at(resource_base + index * abi::RESOURCE_WORDS + k);
        let class = match abi::Class::from_wire(r(1)).ok_or(E::BadEnum {
            what: "resource class",
            index,
            value: r(1),
        })? {
            abi::Class::Transient => ResourceClass::Transient,
            abi::Class::Persistent => ResourceClass::Persistent,
            abi::Class::External => ResourceClass::External,
            abi::Class::Backbuffer => ResourceClass::Backbuffer,
        };
        let desc = match r(0) {
            abi::RESOURCE_IMAGE => {
                let bad = |reason| E::BadImage { index, reason };
                let (format, width, height) = (r(5), r(6), r(7));
                let (mip_levels, layers, samples, aspect) = (r(9), r(10), r(11), r(12));
                if format == 0 || format > i32::MAX as u32 {
                    return Err(bad("invalid format"));
                }
                if width == 0 || height == 0 || width > MAX_IMAGE_EXTENT || height > MAX_IMAGE_EXTENT {
                    return Err(bad("extent out of range"));
                }
                if mip_levels == 0 || mip_levels > 16 || layers == 0 || layers > 2048 {
                    return Err(bad("mip or layer count out of range"));
                }
                if !samples.is_power_of_two() || samples > 64 {
                    return Err(bad("sample count is not 1..64 and a power of two"));
                }
                if aspect == 0 {
                    return Err(bad("empty aspect"));
                }
                ResourceDesc::Image(ImageDesc {
                    format: vk::Format::from_raw(format as i32),
                    width,
                    height,
                    usage: vk::ImageUsageFlags::from_raw(r(8)),
                    mip_levels,
                    layers,
                    samples: vk::SampleCountFlags::from_raw(samples),
                    aspect: vk::ImageAspectFlags::from_raw(aspect),
                })
            }
            abi::RESOURCE_BUFFER => {
                if (8..13).any(|k| r(k) != 0) {
                    return Err(E::BadEnum {
                        what: "buffer parameter",
                        index,
                        value: 1,
                    });
                }
                ResourceDesc::Buffer(BufferDesc {
                    size: r(5) as u64 | (r(6) as u64) << 32,
                    usage: vk::BufferUsageFlags::from_raw(r(7)),
                })
            }
            value => {
                return Err(E::BadEnum {
                    what: "resource kind",
                    index,
                    value,
                });
            }
        };
        let name = name(r(3), r(4), "resource", index)?;
        graph.resource(name, desc, class, Scope(r(2)));
    }

    let mut next_intent = 0usize;
    for index in 0..passes {
        let p = |k: usize| w.at(pass_base + index * abi::PASS_WORDS + k);
        let kind = match abi::PassKind::from_wire(p(0)).ok_or(E::BadEnum {
            what: "pass kind",
            index,
            value: p(0),
        })? {
            abi::PassKind::Graphics => PassKind::Graphics,
            abi::PassKind::Compute => PassKind::Compute,
            abi::PassKind::Transfer => PassKind::Transfer,
            abi::PassKind::Present => PassKind::Present,
        };
        let flags = p(1);
        if flags & !(abi::PASS_FLAG_ROOT | abi::PASS_FLAG_CONDITIONAL) != 0 {
            return Err(E::BadEnum {
                what: "pass flags",
                index,
                value: flags,
            });
        }
        let (first, count) = (p(5) as usize, p(6) as usize);
        if first != next_intent || first + count > intents {
            return Err(E::BadIntentRange { pass: index });
        }
        next_intent += count;
        let mut pass_intents = Vec::with_capacity(count);
        for i in first..first + count {
            let it = |k: usize| w.at(intent_base + i * abi::INTENT_WORDS + k);
            if it(0) as usize >= resources {
                return Err(E::BadResource {
                    pass: index,
                    resource: it(0),
                });
            }
            let role = role(it(1), it(2), i)?;
            let discard = match it(3) {
                0 => false,
                1 => true,
                value => {
                    return Err(E::BadEnum {
                        what: "intent discard flag",
                        index: i,
                        value,
                    });
                }
            };
            pass_intents.push(Intent {
                resource: ResourceId(it(0)),
                role,
                discard: discard || role.implies_discard(),
            });
        }
        graph.passes.push(Pass {
            name: name(p(3), p(4), "pass", index)?,
            kind,
            scope: Scope(p(2)),
            intents: pass_intents,
            conditional: flags & abi::PASS_FLAG_CONDITIONAL != 0,
            record: p(7),
        });
        if flags & abi::PASS_FLAG_ROOT != 0 {
            graph.roots.push(PassId(index as u32));
        }
    }
    if next_intent != intents {
        return Err(E::BadIntentRange { pass: passes });
    }
    Ok(graph)
}

fn stage_to_wire(stage: ShaderStage) -> abi::ShaderStage {
    match stage {
        ShaderStage::Vertex => abi::ShaderStage::Vertex,
        ShaderStage::Fragment => abi::ShaderStage::Fragment,
        ShaderStage::Graphics => abi::ShaderStage::Graphics,
        ShaderStage::Compute => abi::ShaderStage::Compute,
    }
}

fn role_to_wire(role: Role) -> abi::Role {
    use abi::Role as W;
    let load = |load| match load {
        LoadOp::Clear => abi::LoadOp::Clear,
        LoadOp::Load => abi::LoadOp::Load,
        LoadOp::DontCare => abi::LoadOp::DontCare,
    };
    match role {
        Role::ColorAttachment(l) => W::ColorAttachment(load(l)),
        Role::DepthAttachment(l) => W::DepthAttachment(load(l)),
        Role::DepthRead => W::DepthRead,
        Role::Sampled(s) => W::Sampled(stage_to_wire(s)),
        Role::StorageImageRead(s) => W::StorageImageRead(stage_to_wire(s)),
        Role::StorageImageWrite(s) => W::StorageImageWrite(stage_to_wire(s)),
        Role::StorageImageReadWrite(s) => W::StorageImageReadWrite(stage_to_wire(s)),
        Role::StorageBufferRead(s) => W::StorageBufferRead(stage_to_wire(s)),
        Role::StorageBufferWrite(s) => W::StorageBufferWrite(stage_to_wire(s)),
        Role::StorageBufferReadWrite(s) => W::StorageBufferReadWrite(stage_to_wire(s)),
        Role::UniformBuffer(s) => W::UniformBuffer(stage_to_wire(s)),
        Role::VertexBuffer => W::VertexBuffer,
        Role::IndexBuffer => W::IndexBuffer,
        Role::IndirectBuffer => W::IndirectBuffer,
        Role::TransferSrc => W::TransferSrc,
        Role::TransferDst => W::TransferDst,
        Role::HostRead => W::HostRead,
        Role::Present => W::Present,
    }
}

/// Encode a natively declared graph. `payload` maps each pass's record to
/// the wire payload. Roots must be declared in pass order (as `.root()` on
/// each pass does), since the wire format stores them as pass flags.
pub fn encode<R>(graph: &FrameGraph<R>, payload: impl Fn(&R) -> u32) -> Vec<u8> {
    let mut builder = abi::GraphBuilder::new();
    builder.plan_version(graph.plan_version);
    for resource in &graph.resources {
        let class = match resource.class {
            ResourceClass::Transient => abi::Class::Transient,
            ResourceClass::Persistent => abi::Class::Persistent,
            ResourceClass::External => abi::Class::External,
            ResourceClass::Backbuffer => abi::Class::Backbuffer,
        };
        match resource.desc {
            ResourceDesc::Image(i) => {
                let spec = abi::ImageSpec {
                    format: i.format.as_raw() as u32,
                    width: i.width,
                    height: i.height,
                    usage: i.usage.as_raw(),
                    mip_levels: i.mip_levels,
                    layers: i.layers,
                    samples: i.samples.as_raw(),
                    aspect: i.aspect.as_raw(),
                };
                builder.image_scoped(&resource.name, spec, class, resource.scope.0);
            }
            ResourceDesc::Buffer(b) => {
                builder.buffer_scoped(&resource.name, b.size, b.usage.as_raw(), class, resource.scope.0);
            }
        }
    }
    for (index, pass) in graph.passes.iter().enumerate() {
        let kind = match pass.kind {
            PassKind::Graphics => abi::PassKind::Graphics,
            PassKind::Compute => abi::PassKind::Compute,
            PassKind::Transfer => abi::PassKind::Transfer,
            PassKind::Present => abi::PassKind::Present,
        };
        let mut writer = builder
            .pass(&pass.name, kind, payload(&pass.record))
            .scope(pass.scope.0);
        for intent in &pass.intents {
            let resource = abi::ResourceRef(intent.resource.0);
            let role = role_to_wire(intent.role);
            writer = if intent.discard {
                writer.discard(resource, role)
            } else {
                writer.access(resource, role)
            };
        }
        if pass.conditional {
            writer = writer.conditional();
        }
        if graph.roots.contains(&PassId(index as u32)) {
            writer = writer.root();
        }
        writer.id();
    }
    builder.finish()
}
