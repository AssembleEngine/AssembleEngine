//! Bake: validation → edges → topological order from the roots → reorder →
//! lifetimes → alias groups → baked accesses (§3.3). Pure and deterministic:
//! no hash-map iteration, and every tie is broken by pass or resource index.
//!
//! There is no pass merging. The baseline merged graphics passes with
//! identical attachment sets into one rendering scope, but under the
//! single-writer rule two passes can never share a written attachment, so a
//! merge group could never form. Layered content (world, overlay, UI) is one
//! graphics pass with ordered draw lists instead; each graphics pass is one
//! dynamic-rendering scope.

use super::{FrameGraph, Intent, PassId, PassKind, ResourceClass, ResourceDesc, ResourceId, Role, Scope, Usage};
use ash::vk;
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub struct BakeOptions {
    /// Merge transient resources with equal descriptions and disjoint
    /// lifetimes into one physical allocation.
    pub aliasing: bool,
}

impl Default for BakeOptions {
    fn default() -> Self {
        Self { aliasing: true }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BakeError {
    NoRoot,
    UnknownRoot {
        index: u32,
    },
    DuplicateRoot {
        pass: String,
    },
    UnknownResource {
        pass: String,
        index: u32,
    },
    WrongResourceKind {
        pass: String,
        resource: String,
        role: Role,
    },
    MissingUsage {
        pass: String,
        resource: String,
        role: Role,
        required: String,
    },
    DuplicateIntent {
        pass: String,
        resource: String,
    },
    RoleNotAllowed {
        pass: String,
        kind: PassKind,
        role: Role,
    },
    PresentTarget {
        pass: String,
        resource: String,
    },
    PresentShape {
        pass: String,
    },
    NoAttachment {
        pass: String,
    },
    HostReadShape {
        pass: String,
    },
    AttachmentExtent {
        pass: String,
    },
    DepthShape {
        pass: String,
    },
    UninitializedTransient {
        resource: String,
        pass: String,
    },
    Cycle {
        passes: Vec<String>,
    },
    CrossScope {
        pass: String,
        resource: String,
    },
}

impl fmt::Display for BakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BakeError::NoRoot => write!(f, "frame graph has no root pass"),
            BakeError::UnknownRoot { index } => write!(f, "root pass #{index} does not exist"),
            BakeError::DuplicateRoot { pass } => write!(f, "pass {pass:?} is marked as a root twice"),
            BakeError::UnknownResource { pass, index } => {
                write!(f, "pass {pass:?} references unknown resource #{index}")
            }
            BakeError::WrongResourceKind { pass, resource, role } => {
                write!(
                    f,
                    "pass {pass:?} uses {resource:?} as {role:?}, which does not apply to its kind"
                )
            }
            BakeError::MissingUsage {
                pass,
                resource,
                role,
                required,
            } => write!(
                f,
                "pass {pass:?} uses {resource:?} as {role:?} but it lacks usage {required}"
            ),
            BakeError::DuplicateIntent { pass, resource } => {
                write!(f, "pass {pass:?} declares {resource:?} more than once")
            }
            BakeError::RoleNotAllowed { pass, kind, role } => {
                write!(f, "pass {pass:?} ({kind:?}) may not use role {role:?}")
            }
            BakeError::PresentTarget { pass, resource } => {
                write!(f, "pass {pass:?} presents {resource:?}, which is not a backbuffer")
            }
            BakeError::PresentShape { pass } => {
                write!(f, "present pass {pass:?} must have exactly one Present intent")
            }
            BakeError::NoAttachment { pass } => write!(
                f,
                "graphics pass {pass:?} has no attachment or fragment storage image output"
            ),
            BakeError::HostReadShape { pass } => {
                write!(f, "transfer pass {pass:?} mixes host reads with other accesses")
            }
            BakeError::AttachmentExtent { pass } => {
                write!(f, "graphics pass {pass:?} has attachments of different extents")
            }
            BakeError::DepthShape { pass } => {
                write!(f, "graphics pass {pass:?} has more than one depth attachment")
            }
            BakeError::UninitializedTransient { resource, pass } => write!(
                f,
                "transient {resource:?} is first accessed by {pass:?} without a discarding write"
            ),
            BakeError::Cycle { passes } => write!(f, "dependency cycle through {}", passes.join(" -> ")),
            BakeError::CrossScope { pass, resource } => {
                write!(f, "pass {pass:?} may not access {resource:?} (view scope violation)")
            }
        }
    }
}

impl std::error::Error for BakeError {}

/// One declared access of a baked pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BakedAccess {
    pub resource: ResourceId,
    pub role: Role,
    pub usage: Usage,
    /// Old layout `UNDEFINED` (declared discard, or first use of a
    /// transient / alias member).
    pub discard: bool,
    /// Attachment store op: `DONT_CARE` when a transient's last use is this
    /// pass.
    pub store: Option<vk::AttachmentStoreOp>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BakedPass {
    pub pass: PassId,
    pub kind: PassKind,
    pub conditional: bool,
    pub accesses: Vec<BakedAccess>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BakedResource {
    pub reachable: bool,
    pub image: bool,
    /// First and last index into `BakedGraph::passes`.
    pub lifetime: Option<(u32, u32)>,
    /// Physical allocation; `None` for external/backbuffer or unreachable.
    pub physical: Option<u32>,
    /// A transient used by exactly one pass (never stored).
    pub transient_single_pass: bool,
}

/// One allocation unit (per frame slot at realization).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalResource {
    pub desc: ResourceDesc,
    pub class: ResourceClass,
    pub scope: Scope,
    /// Logical members in lifetime order.
    pub members: Vec<ResourceId>,
}

/// A physical allocation switches from one logical resource to the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AliasHandoff {
    pub physical: u32,
    pub from: ResourceId,
    pub to: ResourceId,
    /// Index into `BakedGraph::passes` of `to`'s first use.
    pub at_pass: u32,
}

/// A dependency edge (for diagnostics and dot output).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    pub from: PassId,
    pub to: PassId,
    pub resource: ResourceId,
}

/// The baked plan. Contains no names, so equal declarations bake to equal
/// plans (`PartialEq`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BakedGraph {
    pub signature: u64,
    pub passes: Vec<BakedPass>,
    /// Indexed by `ResourceId`.
    pub resources: Vec<BakedResource>,
    pub physical: Vec<PhysicalResource>,
    pub handoffs: Vec<AliasHandoff>,
    pub edges: Vec<Edge>,
}

impl BakedGraph {
    pub fn order(&self) -> impl Iterator<Item = PassId> + '_ {
        self.passes.iter().map(|pass| pass.pass)
    }

    /// Allocated bytes per frame slot (images estimated as width × height ×
    /// 4 bytes per texel; enough to compare aliasing on/off).
    pub fn estimated_bytes(&self) -> u64 {
        self.physical
            .iter()
            .map(|physical| match physical.desc {
                ResourceDesc::Image(image) => image.width as u64 * image.height as u64 * 4 * image.layers as u64,
                ResourceDesc::Buffer(buffer) => buffer.size,
            })
            .sum()
    }
}

struct Analysis {
    /// Sorted, deduplicated dependency passes per pass.
    deps: Vec<Vec<PassId>>,
    edges: Vec<Edge>,
}

pub fn bake<R>(graph: &FrameGraph<R>, options: BakeOptions) -> Result<BakedGraph, BakeError> {
    validate(graph)?;
    let analysis = analyze(graph)?;
    let topo = topological_from_roots(graph, &analysis)?;
    let order = reorder(graph, &analysis, &topo);
    validate_transient_first_use(graph, &order)?;
    Ok(assemble(graph, &analysis, order, options))
}

fn validate<R>(graph: &FrameGraph<R>) -> Result<(), BakeError> {
    if graph.roots.is_empty() {
        return Err(BakeError::NoRoot);
    }
    let mut seen_roots = Vec::with_capacity(graph.roots.len());
    for &root in &graph.roots {
        if root.index() >= graph.passes.len() {
            return Err(BakeError::UnknownRoot { index: root.0 });
        }
        if seen_roots.contains(&root) {
            return Err(BakeError::DuplicateRoot {
                pass: graph.pass_name(root).to_string(),
            });
        }
        seen_roots.push(root);
    }
    for pass in &graph.passes {
        let name = || pass.name.clone();
        let mut seen: Vec<ResourceId> = Vec::with_capacity(pass.intents.len());
        let mut extent: Option<vk::Extent2D> = None;
        let mut attachments = 0;
        let mut storage_target = false;
        for intent in &pass.intents {
            let Some(resource) = graph.resources.get(intent.resource.index()) else {
                return Err(BakeError::UnknownResource {
                    pass: name(),
                    index: intent.resource.0,
                });
            };
            let resource_name = || resource.name.clone();
            if seen.contains(&intent.resource) {
                return Err(BakeError::DuplicateIntent {
                    pass: name(),
                    resource: resource_name(),
                });
            }
            seen.push(intent.resource);
            if !intent.role.applies_to(&resource.desc) {
                return Err(BakeError::WrongResourceKind {
                    pass: name(),
                    resource: resource_name(),
                    role: intent.role,
                });
            }
            if let Some(required) = intent.role.missing_usage(&resource.desc) {
                return Err(BakeError::MissingUsage {
                    pass: name(),
                    resource: resource_name(),
                    role: intent.role,
                    required,
                });
            }
            let allowed = match pass.kind {
                PassKind::Graphics => !matches!(
                    intent.role,
                    Role::Present | Role::TransferSrc | Role::TransferDst | Role::HostRead
                ),
                PassKind::Compute => {
                    !intent.role.is_attachment()
                        && !matches!(
                            intent.role,
                            Role::Present
                                | Role::TransferSrc
                                | Role::TransferDst
                                | Role::VertexBuffer
                                | Role::IndexBuffer
                                | Role::HostRead
                        )
                }
                PassKind::Transfer => matches!(intent.role, Role::TransferSrc | Role::TransferDst | Role::HostRead),
                PassKind::Present => intent.role == Role::Present,
            };
            if !allowed {
                return Err(BakeError::RoleNotAllowed {
                    pass: name(),
                    kind: pass.kind,
                    role: intent.role,
                });
            }
            if intent.role == Role::Present && resource.class != ResourceClass::Backbuffer {
                return Err(BakeError::PresentTarget {
                    pass: name(),
                    resource: resource_name(),
                });
            }
            let scope_ok =
                resource.scope == pass.scope || (resource.scope == Scope::GLOBAL && !intent.role.usage().write);
            if !scope_ok {
                return Err(BakeError::CrossScope {
                    pass: name(),
                    resource: resource_name(),
                });
            }
            storage_target |= matches!(
                intent.role,
                Role::StorageImageWrite(crate::ShaderStage::Fragment | crate::ShaderStage::Graphics)
                    | Role::StorageImageReadWrite(crate::ShaderStage::Fragment | crate::ShaderStage::Graphics)
            );
            if intent.role.is_attachment() {
                attachments += 1;
                if let ResourceDesc::Image(image) = resource.desc {
                    match extent {
                        Some(previous) if previous != image.extent() => {
                            return Err(BakeError::AttachmentExtent { pass: name() });
                        }
                        _ => extent = Some(image.extent()),
                    }
                }
            }
        }
        if pass.intents.iter().filter(|intent| intent.role.is_depth()).count() > 1 {
            return Err(BakeError::DepthShape { pass: name() });
        }
        if pass.kind == PassKind::Graphics && attachments == 0 && !storage_target {
            return Err(BakeError::NoAttachment { pass: name() });
        }
        // Barriers precede a pass's recording, so a host read in the same
        // pass as a copy would make the buffer host-visible before the copy.
        let host_reads = pass
            .intents
            .iter()
            .filter(|intent| intent.role == Role::HostRead)
            .count();
        if host_reads > 0 && host_reads != pass.intents.len() {
            return Err(BakeError::HostReadShape { pass: name() });
        }
        if pass.kind == PassKind::Present && pass.intents.len() != 1 {
            return Err(BakeError::PresentShape { pass: name() });
        }
    }
    Ok(())
}

/// Dependencies from resource versions, in pass declaration order. Each
/// write starts a new version of its resource:
///
/// * a read depends on the latest write declared before it, or, when no
///   write precedes it, on the first write (so a lone writer is always read,
///   whatever the declaration order);
/// * a write depends on the previous write (write after write) and on every
///   read of the previous version (write after read).
///
/// With one writer per resource this is plain producer → consumer.
fn analyze<R>(graph: &FrameGraph<R>) -> Result<Analysis, BakeError> {
    let mut first_writer: Vec<Option<PassId>> = vec![None; graph.resources.len()];
    for (index, pass) in graph.passes.iter().enumerate() {
        for intent in &pass.intents {
            if intent.role.usage().write {
                first_writer[intent.resource.index()].get_or_insert(PassId(index as u32));
            }
        }
    }
    let mut last_writer: Vec<Option<PassId>> = vec![None; graph.resources.len()];
    // Reads of the current version (only once a write has been declared).
    let mut readers: Vec<Vec<PassId>> = vec![Vec::new(); graph.resources.len()];
    // Reads declared before the first write: they read the first version,
    // so they join its readers when the first write is reached.
    let mut early: Vec<Vec<PassId>> = vec![Vec::new(); graph.resources.len()];
    let mut deps: Vec<Vec<PassId>> = vec![Vec::new(); graph.passes.len()];
    let mut edges = Vec::new();
    for (index, pass) in graph.passes.iter().enumerate() {
        let id = PassId(index as u32);
        let mut depend = |from: PassId, resource: ResourceId| {
            if from != id {
                deps[index].push(from);
                edges.push(Edge { from, to: id, resource });
            }
        };
        for intent in &pass.intents {
            let r = intent.resource.index();
            if intent.role.usage().write {
                if let Some(previous) = last_writer[r] {
                    depend(previous, intent.resource);
                }
                for reader in std::mem::take(&mut readers[r]) {
                    depend(reader, intent.resource);
                }
                if last_writer[r].is_none() {
                    readers[r] = std::mem::take(&mut early[r]);
                }
                last_writer[r] = Some(id);
            } else if let Some(writer) = last_writer[r] {
                depend(writer, intent.resource);
                readers[r].push(id);
            } else if let Some(writer) = first_writer[r] {
                depend(writer, intent.resource);
                early[r].push(id);
            }
        }
        deps[index].sort_unstable();
        deps[index].dedup();
    }
    edges.sort_by_key(|edge| (edge.from, edge.to, edge.resource));
    edges.dedup();
    Ok(Analysis { deps, edges })
}

/// DFS post-order over dependencies from the roots, in root order; passes
/// the roots do not reach are pruned.
fn topological_from_roots<R>(graph: &FrameGraph<R>, analysis: &Analysis) -> Result<Vec<PassId>, BakeError> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        New,
        Active,
        Done,
    }
    let mut marks = vec![Mark::New; graph.passes.len()];
    let mut order = Vec::with_capacity(graph.passes.len());
    // Explicit stack: (pass, next dependency index).
    let mut stack: Vec<(PassId, usize)> = Vec::new();
    for &root in &graph.roots {
        if marks[root.index()] != Mark::New {
            continue;
        }
        marks[root.index()] = Mark::Active;
        stack.push((root, 0));
        while let Some(&mut (pass, ref mut next)) = stack.last_mut() {
            let deps = &analysis.deps[pass.index()];
            if *next < deps.len() {
                let dep = deps[*next];
                *next += 1;
                match marks[dep.index()] {
                    Mark::New => {
                        marks[dep.index()] = Mark::Active;
                        stack.push((dep, 0));
                    }
                    Mark::Active => {
                        let start = stack.iter().position(|(entry, _)| *entry == dep).unwrap_or(0);
                        let mut passes: Vec<String> = stack[start..]
                            .iter()
                            .map(|(entry, _)| graph.pass_name(*entry).to_string())
                            .collect();
                        passes.push(graph.pass_name(dep).to_string());
                        return Err(BakeError::Cycle { passes });
                    }
                    Mark::Done => {}
                }
            } else {
                marks[pass.index()] = Mark::Done;
                order.push(pass);
                stack.pop();
            }
        }
    }
    Ok(order)
}

/// Greedy list scheduling over the topological set (§1.3 step 6, without the
/// merge bonus): prefer the pass whose latest dependency was scheduled most
/// recently (consumers follow producers, which shortens transient
/// lifetimes), then fewer pending readers; ties by topological index.
fn reorder<R>(graph: &FrameGraph<R>, analysis: &Analysis, topo: &[PassId]) -> Vec<PassId> {
    let n = graph.passes.len();
    let mut in_set = vec![false; n];
    let mut topo_index = vec![usize::MAX; n];
    for (index, &pass) in topo.iter().enumerate() {
        in_set[pass.index()] = true;
        topo_index[pass.index()] = index;
    }
    // Readers (within the set) of each pass's outputs.
    let mut readers = vec![0i64; n];
    for edge in &analysis.edges {
        if in_set[edge.from.index()] && in_set[edge.to.index()] {
            readers[edge.from.index()] += 1;
        }
    }
    let mut position = vec![usize::MAX; n];
    let mut order: Vec<PassId> = Vec::with_capacity(topo.len());
    let mut remaining: Vec<PassId> = topo.to_vec();
    while !remaining.is_empty() {
        let mut best: Option<(i64, usize, usize)> = None; // (score, topo index, remaining index)
        for (slot, &pass) in remaining.iter().enumerate() {
            let deps = &analysis.deps[pass.index()];
            if deps
                .iter()
                .any(|dep| in_set[dep.index()] && position[dep.index()] == usize::MAX)
            {
                continue;
            }
            let mut score = 0i64;
            let latest_dep = deps
                .iter()
                .filter(|dep| in_set[dep.index()])
                .map(|dep| position[dep.index()] as i64 + 1)
                .max()
                .unwrap_or(0);
            score += latest_dep;
            score -= readers[pass.index()];
            let candidate = (score, topo_index[pass.index()], slot);
            best = match best {
                None => Some(candidate),
                Some(current) if candidate.0 > current.0 || (candidate.0 == current.0 && candidate.1 < current.1) => {
                    Some(candidate)
                }
                keep => keep,
            };
        }
        let (_, _, slot) = best.expect("the topological set always has a ready pass");
        let pass = remaining.remove(slot);
        position[pass.index()] = order.len();
        order.push(pass);
    }
    order
}

/// A transient's contents exist only after a discarding write in this frame.
fn validate_transient_first_use<R>(graph: &FrameGraph<R>, order: &[PassId]) -> Result<(), BakeError> {
    let mut seen = vec![false; graph.resources.len()];
    for &pass_id in order {
        let pass = &graph.passes[pass_id.index()];
        for intent in &pass.intents {
            let index = intent.resource.index();
            if seen[index] {
                continue;
            }
            seen[index] = true;
            let resource = &graph.resources[index];
            if resource.class == ResourceClass::Transient && !(intent.role.usage().write && intent.discard) {
                return Err(BakeError::UninitializedTransient {
                    resource: resource.name.clone(),
                    pass: pass.name.clone(),
                });
            }
        }
    }
    Ok(())
}

fn assemble<R>(graph: &FrameGraph<R>, analysis: &Analysis, order: Vec<PassId>, options: BakeOptions) -> BakedGraph {
    let resource_count = graph.resources.len();

    // Lifetimes (pass indices) over reachable resources.
    let mut resources = vec![BakedResource::default(); resource_count];
    for (index, &pass) in order.iter().enumerate() {
        for intent in &graph.passes[pass.index()].intents {
            let baked = &mut resources[intent.resource.index()];
            baked.reachable = true;
            baked.image = graph.resources[intent.resource.index()].desc.is_image();
            baked.lifetime = Some(match baked.lifetime {
                None => (index as u32, index as u32),
                Some((first, _)) => (first, index as u32),
            });
        }
    }

    // Physical assignment. Persistent resources get their own allocation;
    // transients alias by exact description (buffers: equal size and usage),
    // same scope (no memory reuse across views), and disjoint lifetimes.
    let mut by_first_use: Vec<ResourceId> = (0..resource_count as u32)
        .map(ResourceId)
        .filter(|id| resources[id.index()].reachable)
        .collect();
    by_first_use.sort_by_key(|id| (resources[id.index()].lifetime.map(|l| l.0), *id));
    let mut physical: Vec<PhysicalResource> = Vec::new();
    let mut physical_end: Vec<u32> = Vec::new();
    let mut handoffs = Vec::new();
    for id in by_first_use {
        let resource = &graph.resources[id.index()];
        if matches!(resource.class, ResourceClass::External | ResourceClass::Backbuffer) {
            continue;
        }
        let (first, last) = resources[id.index()].lifetime.expect("reachable");
        let reuse = (options.aliasing && resource.class == ResourceClass::Transient)
            .then(|| {
                physical.iter().enumerate().position(|(index, candidate)| {
                    candidate.class == ResourceClass::Transient
                        && candidate.desc == resource.desc
                        && candidate.scope == resource.scope
                        && physical_end[index] < first
                })
            })
            .flatten();
        let index = match reuse {
            Some(index) => {
                let from = *physical[index].members.last().expect("non-empty group");
                handoffs.push(AliasHandoff {
                    physical: index as u32,
                    from,
                    to: id,
                    at_pass: first,
                });
                physical[index].members.push(id);
                physical_end[index] = last;
                index
            }
            None => {
                physical.push(PhysicalResource {
                    desc: resource.desc,
                    class: resource.class,
                    scope: resource.scope,
                    members: vec![id],
                });
                physical_end.push(last);
                physical.len() - 1
            }
        };
        resources[id.index()].physical = Some(index as u32);
    }

    for (index, resource) in graph.resources.iter().enumerate() {
        if let Some((first, last)) = resources[index].lifetime {
            resources[index].transient_single_pass = resource.class == ResourceClass::Transient && first == last;
        }
    }

    // Baked accesses, with discard and store ops.
    let mut first_access_seen = vec![false; resource_count];
    let mut passes = Vec::with_capacity(order.len());
    for (index, &pass_id) in order.iter().enumerate() {
        let pass = &graph.passes[pass_id.index()];
        let accesses = pass
            .intents
            .iter()
            .map(|intent: &Intent| {
                let resource = &graph.resources[intent.resource.index()];
                let first_use = !std::mem::replace(&mut first_access_seen[intent.resource.index()], true);
                let discard = intent.discard || (first_use && resource.class == ResourceClass::Transient);
                let store = intent.role.is_attachment().then(|| {
                    let (_, last) = resources[intent.resource.index()].lifetime.expect("reachable");
                    if resource.class == ResourceClass::Transient && last as usize == index {
                        vk::AttachmentStoreOp::DONT_CARE
                    } else {
                        vk::AttachmentStoreOp::STORE
                    }
                });
                BakedAccess {
                    resource: intent.resource,
                    role: intent.role,
                    usage: intent.role.usage(),
                    discard,
                    store,
                }
            })
            .collect();
        passes.push(BakedPass {
            pass: pass_id,
            kind: pass.kind,
            conditional: pass.conditional,
            accesses,
        });
    }

    let mut edges: Vec<_> = analysis
        .edges
        .iter()
        .copied()
        .filter(|edge| order.contains(&edge.from) && order.contains(&edge.to))
        .collect();
    edges.sort_by_key(|edge| (edge.from, edge.to, edge.resource));

    BakedGraph {
        signature: graph.signature(),
        passes,
        resources,
        physical,
        handoffs,
        edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferDesc, FrameGraph, ImageDesc, LoadOp, PassKind, ResourceClass, Role, Scope, ShaderStage};

    const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

    fn color_usage() -> vk::ImageUsageFlags {
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE
    }

    fn image(graph: &mut FrameGraph, name: &str, class: ResourceClass) -> ResourceId {
        graph.image(name, ImageDesc::color_2d(FORMAT, 64, 64, color_usage()), class)
    }

    fn backbuffer(graph: &mut FrameGraph) -> ResourceId {
        graph.image(
            "backbuffer",
            ImageDesc::color_2d(
                vk::Format::B8G8R8A8_UNORM,
                64,
                64,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            ),
            ResourceClass::Backbuffer,
        )
    }

    /// scene -> blur -> composite(backbuffer) -> present
    fn chain() -> FrameGraph {
        let mut graph = FrameGraph::new();
        let scene = image(&mut graph, "scene", ResourceClass::Transient);
        let blurred = image(&mut graph, "blurred", ResourceClass::Transient);
        let back = backbuffer(&mut graph);
        graph
            .pass("scene", PassKind::Graphics, ())
            .color(scene, LoadOp::Clear)
            .id();
        graph
            .pass("blur", PassKind::Compute, ())
            .access(scene, Role::StorageImageRead(ShaderStage::Compute))
            .discard(blurred, Role::StorageImageWrite(ShaderStage::Compute))
            .id();
        graph
            .pass("composite", PassKind::Graphics, ())
            .sampled(blurred, ShaderStage::Fragment)
            .color(back, LoadOp::Clear)
            .id();
        graph.pass("present", PassKind::Present, ()).present(back).root().id();
        graph
    }

    fn names(graph: &FrameGraph, baked: &BakedGraph) -> Vec<String> {
        baked.order().map(|pass| graph.pass_name(pass).to_string()).collect()
    }

    #[test]
    fn missing_root_is_an_error() {
        let mut graph = chain();
        graph.roots.clear();
        assert_eq!(bake(&graph, BakeOptions::default()), Err(BakeError::NoRoot));
    }

    #[test]
    fn duplicate_root_marker_is_an_error() {
        let mut graph = chain();
        graph.add_root(PassId(3));
        assert!(matches!(
            bake(&graph, BakeOptions::default()),
            Err(BakeError::DuplicateRoot { pass }) if pass == "present"
        ));
    }

    #[test]
    fn cycle_is_an_error() {
        let mut graph = FrameGraph::new();
        let a = image(&mut graph, "a", ResourceClass::Persistent);
        let b = image(&mut graph, "b", ResourceClass::Persistent);
        let back = backbuffer(&mut graph);
        graph
            .pass("p", PassKind::Compute, ())
            .access(a, Role::StorageImageRead(ShaderStage::Compute))
            .access(b, Role::StorageImageWrite(ShaderStage::Compute))
            .id();
        graph
            .pass("q", PassKind::Compute, ())
            .access(b, Role::StorageImageRead(ShaderStage::Compute))
            .access(a, Role::StorageImageWrite(ShaderStage::Compute))
            .id();
        graph
            .pass("draw", PassKind::Graphics, ())
            .sampled(a, ShaderStage::Fragment)
            .color(back, LoadOp::Clear)
            .id();
        graph.pass("present", PassKind::Present, ()).present(back).root().id();
        let err = bake(&graph, BakeOptions::default()).unwrap_err();
        let BakeError::Cycle { passes } = err else {
            panic!("expected a cycle, got {err:?}")
        };
        assert!(
            passes.contains(&"p".to_string()) && passes.contains(&"q".to_string()),
            "{passes:?}"
        );
    }

    /// A second writer declared before the present draws over the first;
    /// one declared after it writes a version nobody reads and is pruned.
    #[test]
    fn writers_are_ordered_by_declaration() {
        let mut graph = chain();
        let back = ResourceId(2);
        let present = graph.passes.pop().unwrap();
        graph.roots.clear();
        graph
            .pass("overlay", PassKind::Graphics, ())
            .color(back, LoadOp::Load)
            .id();
        graph.passes.push(present);
        graph.roots.push(PassId(graph.passes.len() as u32 - 1));
        let baked = bake(&graph, BakeOptions::default()).unwrap();
        assert_eq!(
            names(&graph, &baked),
            ["scene", "blur", "composite", "overlay", "present"]
        );

        let mut late = chain();
        late.pass("overlay", PassKind::Graphics, ())
            .color(back, LoadOp::Load)
            .id();
        let baked = bake(&late, BakeOptions::default()).unwrap();
        assert_eq!(names(&late, &baked), ["scene", "blur", "composite", "present"]);
    }

    /// A read of one version runs before the next write (write after read),
    /// and readers see the version declared before them.
    #[test]
    fn versions_order_reads_between_writes() {
        let mut graph = FrameGraph::new();
        let a = image(&mut graph, "a", ResourceClass::Persistent);
        let b = image(&mut graph, "b", ResourceClass::Transient);
        let c = image(&mut graph, "c", ResourceClass::Transient);
        let back = backbuffer(&mut graph);
        graph
            .pass("write a", PassKind::Graphics, ())
            .color(a, LoadOp::Clear)
            .id();
        graph
            .pass("read a v1", PassKind::Compute, ())
            .access(a, Role::StorageImageRead(ShaderStage::Compute))
            .discard(b, Role::StorageImageWrite(ShaderStage::Compute))
            .id();
        graph
            .pass("rewrite a", PassKind::Graphics, ())
            .color(a, LoadOp::Load)
            .id();
        graph
            .pass("read a v2", PassKind::Compute, ())
            .access(a, Role::StorageImageRead(ShaderStage::Compute))
            .discard(c, Role::StorageImageWrite(ShaderStage::Compute))
            .id();
        graph
            .pass("compose", PassKind::Graphics, ())
            .sampled(b, ShaderStage::Fragment)
            .sampled(c, ShaderStage::Fragment)
            .color(back, LoadOp::Clear)
            .id();
        graph.pass("present", PassKind::Present, ()).present(back).root().id();
        let baked = bake(&graph, BakeOptions::default()).unwrap();
        let order = names(&graph, &baked);
        let at = |name: &str| order.iter().position(|n| n == name).unwrap();
        assert!(at("write a") < at("read a v1"));
        assert!(at("read a v1") < at("rewrite a"), "{order:?}");
        assert!(at("rewrite a") < at("read a v2"), "{order:?}");
        let edge = |from: u32, to: u32| {
            baked.edges.contains(&Edge {
                from: PassId(from),
                to: PassId(to),
                resource: a,
            })
        };
        assert!(edge(0, 1) && edge(0, 2) && edge(1, 2) && edge(2, 3));
        assert!(!edge(0, 3), "v2's reader depends on the rewrite, not the first write");
    }

    /// A read declared before the first write reads the first version, so
    /// it runs before a second write. (`read` draws into the backbuffer
    /// first; `final` loads it and adds what `use` made from the second
    /// version.)
    #[test]
    fn early_reads_precede_the_second_write() {
        let mut graph = FrameGraph::new();
        let a = image(&mut graph, "a", ResourceClass::Persistent);
        let other = image(&mut graph, "other", ResourceClass::Transient);
        let back = backbuffer(&mut graph);
        graph
            .pass("read", PassKind::Graphics, ())
            .sampled(a, ShaderStage::Fragment)
            .color(back, LoadOp::Clear)
            .id();
        graph.pass("w1", PassKind::Graphics, ()).color(a, LoadOp::Clear).id();
        graph.pass("w2", PassKind::Graphics, ()).color(a, LoadOp::Load).id();
        graph
            .pass("use", PassKind::Graphics, ())
            .sampled(a, ShaderStage::Fragment)
            .color(other, LoadOp::Clear)
            .id();
        graph
            .pass("final", PassKind::Graphics, ())
            .sampled(other, ShaderStage::Fragment)
            .color(back, LoadOp::Load)
            .id();
        graph.pass("present", PassKind::Present, ()).present(back).root().id();
        let baked = bake(&graph, BakeOptions::default()).unwrap();
        let order = names(&graph, &baked);
        let at = |name: &str| order.iter().position(|n| n == name).unwrap();
        assert!(
            at("w1") < at("read") && at("read") < at("w2") && at("w2") < at("use"),
            "{order:?}"
        );
        assert!(
            baked.edges.contains(&Edge {
                from: PassId(0),
                to: PassId(2),
                resource: a
            }),
            "write after read"
        );
    }

    fn depth_image(graph: &mut FrameGraph, name: &str, class: ResourceClass) -> ResourceId {
        let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
        graph.image(name, ImageDesc::depth_2d(vk::Format::D32_SFLOAT, 64, 64, usage), class)
    }

    /// Opaque clears depth, units load and extend it, decals test against
    /// it read-only, and post samples it: one depth buffer, two writers.
    #[test]
    fn depth_buffer_shared_by_passes() {
        let mut graph = FrameGraph::new();
        let depth = depth_image(&mut graph, "depth", ResourceClass::Transient);
        let gbuf0 = image(&mut graph, "albedo", ResourceClass::Transient);
        let gbuf1 = image(&mut graph, "normal", ResourceClass::Transient);
        let back = backbuffer(&mut graph);
        graph
            .pass("terrain", PassKind::Graphics, ())
            .color(gbuf0, LoadOp::Clear)
            .color(gbuf1, LoadOp::Clear)
            .depth(depth, LoadOp::Clear)
            .id();
        graph
            .pass("units", PassKind::Graphics, ())
            .color(gbuf0, LoadOp::Load)
            .color(gbuf1, LoadOp::Load)
            .depth(depth, LoadOp::Load)
            .id();
        graph
            .pass("decals", PassKind::Graphics, ())
            .color(gbuf0, LoadOp::Load)
            .depth_read(depth)
            .id();
        graph
            .pass("light", PassKind::Graphics, ())
            .sampled(gbuf0, ShaderStage::Fragment)
            .sampled(gbuf1, ShaderStage::Fragment)
            .sampled(depth, ShaderStage::Fragment)
            .color(back, LoadOp::Clear)
            .id();
        graph.pass("present", PassKind::Present, ()).present(back).root().id();
        let baked = bake(&graph, BakeOptions::default()).unwrap();
        assert_eq!(
            names(&graph, &baked),
            ["terrain", "units", "decals", "light", "present"]
        );
        let access = |pass: usize| {
            baked.passes[pass]
                .accesses
                .iter()
                .find(|a| a.resource == depth)
                .copied()
                .unwrap()
        };
        assert!(access(0).discard && !access(1).discard);
        assert_eq!(
            access(0).usage.layout,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );
        assert_eq!(access(2).usage.layout, vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        assert!(!access(2).usage.write);
        assert_eq!(access(1).store, Some(vk::AttachmentStoreOp::STORE));
        assert_eq!(graph.passes[0].color_attachments().count(), 2);
        assert_eq!(graph.passes[0].depth_attachment().map(|i| i.resource), Some(depth));
        // Decals' read-only depth sees the units' version, so it runs after
        // units even though it also writes albedo after them.
        assert!(baked.edges.contains(&Edge {
            from: PassId(1),
            to: PassId(2),
            resource: depth
        }));
    }

    #[test]
    fn depth_roles_are_checked() {
        let mut graph = FrameGraph::new();
        let depth = depth_image(&mut graph, "depth", ResourceClass::Transient);
        let other = depth_image(&mut graph, "other", ResourceClass::Transient);
        let color = image(&mut graph, "color", ResourceClass::Transient);
        let back = backbuffer(&mut graph);
        graph
            .pass("two depths", PassKind::Graphics, ())
            .depth(depth, LoadOp::Clear)
            .depth(other, LoadOp::Clear)
            .color(back, LoadOp::Clear)
            .id();
        graph.pass("present", PassKind::Present, ()).present(back).root().id();
        assert_eq!(
            bake(&graph, BakeOptions::default()),
            Err(BakeError::DepthShape {
                pass: "two depths".into()
            })
        );

        graph.passes[0].intents.remove(1);
        graph.passes[0].intents[0].resource = color;
        assert!(
            matches!(
                bake(&graph, BakeOptions::default()),
                Err(BakeError::WrongResourceKind { .. })
            ),
            "a colour image is not a depth attachment"
        );

        graph.passes[0].intents[0] = Intent {
            resource: depth,
            role: Role::ColorAttachment(LoadOp::Clear),
            discard: true,
        };
        assert!(
            matches!(
                bake(&graph, BakeOptions::default()),
                Err(BakeError::WrongResourceKind { .. })
            ),
            "a depth image is not a colour attachment"
        );

        // A depth-only pass (a shadow map) is a valid graphics pass.
        let mut shadow = FrameGraph::new();
        let map = depth_image(&mut shadow, "shadow", ResourceClass::Transient);
        let back = backbuffer(&mut shadow);
        shadow
            .pass("shadow", PassKind::Graphics, ())
            .depth(map, LoadOp::Clear)
            .id();
        shadow
            .pass("lit", PassKind::Graphics, ())
            .sampled(map, ShaderStage::Fragment)
            .color(back, LoadOp::Clear)
            .id();
        shadow.pass("present", PassKind::Present, ()).present(back).root().id();
        let baked = bake(&shadow, BakeOptions::default()).unwrap();
        assert_eq!(names(&shadow, &baked), ["shadow", "lit", "present"]);
        assert_eq!(baked.passes[0].accesses[0].store, Some(vk::AttachmentStoreOp::STORE));
    }

    #[test]
    fn unreachable_pass_is_pruned() {
        let mut graph = chain();
        let debug = image(&mut graph, "debug", ResourceClass::Transient);
        graph
            .pass("debug overlay", PassKind::Graphics, ())
            .color(debug, LoadOp::Clear)
            .id();
        let baked = bake(&graph, BakeOptions::default()).unwrap();
        assert_eq!(names(&graph, &baked), ["scene", "blur", "composite", "present"]);
        assert!(!baked.resources[debug.index()].reachable);
        assert_eq!(baked.resources[debug.index()].physical, None);
    }

    #[test]
    fn bake_is_deterministic_and_name_independent() {
        let graph = chain();
        let a = bake(&graph, BakeOptions::default()).unwrap();
        let b = bake(&graph, BakeOptions::default()).unwrap();
        assert_eq!(a, b);
        let mut renamed = chain();
        for (index, pass) in renamed.passes.iter_mut().enumerate() {
            pass.name = format!("pass{index}");
        }
        assert_eq!(
            bake(&renamed, BakeOptions::default()).unwrap(),
            a,
            "names do not shape the plan"
        );
        assert_eq!(crate::dot::to_dot(&graph, &a), crate::dot::to_dot(&graph, &b));
    }

    #[test]
    fn lifetimes_cover_first_and_last_use() {
        let graph = chain();
        let baked = bake(&graph, BakeOptions::default()).unwrap();
        assert_eq!(baked.resources[0].lifetime, Some((0, 1)), "scene");
        assert_eq!(baked.resources[1].lifetime, Some((1, 2)), "blurred");
        assert_eq!(baked.resources[2].lifetime, Some((2, 3)), "backbuffer");
        assert!(baked.passes[1].accesses[1].discard);
        assert_eq!(
            baked.passes[0].accesses[0].store,
            Some(vk::AttachmentStoreOp::STORE),
            "read later"
        );
    }

    #[test]
    fn transient_reads_need_a_discarding_writer() {
        let mut graph = FrameGraph::new();
        let scratch = image(&mut graph, "scratch", ResourceClass::Transient);
        let back = backbuffer(&mut graph);
        graph
            .pass("draw", PassKind::Graphics, ())
            .sampled(scratch, ShaderStage::Fragment)
            .color(back, LoadOp::Clear)
            .id();
        graph.pass("present", PassKind::Present, ()).present(back).root().id();
        assert!(matches!(
            bake(&graph, BakeOptions::default()),
            Err(BakeError::UninitializedTransient { resource, .. }) if resource == "scratch"
        ));
    }

    #[test]
    fn validation_rejects_bad_declarations() {
        let mut graph = chain();
        let buffer = graph.buffer(
            "buf",
            BufferDesc {
                size: 64,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            },
            ResourceClass::Persistent,
        );
        graph.passes[1].intents.push(Intent {
            resource: buffer,
            role: Role::StorageBufferRead(ShaderStage::Compute),
            discard: false,
        });
        assert!(matches!(
            bake(&graph, BakeOptions::default()),
            Err(BakeError::MissingUsage { .. })
        ));

        let mut graph = chain();
        graph.passes[1].intents[0].role = Role::ColorAttachment(LoadOp::Load);
        assert!(matches!(
            bake(&graph, BakeOptions::default()),
            Err(BakeError::RoleNotAllowed { .. })
        ));

        let mut graph = chain();
        let not_back = image(&mut graph, "offscreen", ResourceClass::External);
        graph.passes[3].intents[0].resource = not_back;
        assert!(matches!(
            bake(&graph, BakeOptions::default()),
            Err(BakeError::PresentTarget { .. })
        ));

        let mut graph = chain();
        let duplicate = graph.passes[1].intents[0];
        graph.passes[1].intents.push(duplicate);
        assert!(matches!(
            bake(&graph, BakeOptions::default()),
            Err(BakeError::DuplicateIntent { .. })
        ));

        // Host reads: never in a graphics or compute pass, and never mixed
        // with a copy in one transfer pass.
        let host_graph = |mixed: bool, kind: PassKind| {
            let mut graph = FrameGraph::<()>::new();
            let desc = BufferDesc {
                size: 64,
                usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            };
            let source = graph.buffer("source", desc, ResourceClass::Persistent);
            let readback = graph.buffer("readback", desc, ResourceClass::Persistent);
            let pass = graph.pass("read", kind, ()).access(readback, Role::HostRead);
            if mixed {
                pass.access(source, Role::TransferSrc).root().id();
            } else {
                pass.root().id();
            }
            graph
        };
        assert!(bake(&host_graph(false, PassKind::Transfer), BakeOptions::default()).is_ok());
        assert!(matches!(
            bake(&host_graph(true, PassKind::Transfer), BakeOptions::default()),
            Err(BakeError::HostReadShape { .. })
        ));
        assert!(matches!(
            bake(&host_graph(false, PassKind::Compute), BakeOptions::default()),
            Err(BakeError::RoleNotAllowed { .. })
        ));
    }

    /// The layered-pass case the baseline merged: a second graphics pass
    /// loading the first one's attachment is a second writer, ordered after
    /// the first and read by the present.
    #[test]
    fn attachment_continuation_is_a_second_writer() {
        let mut graph = FrameGraph::new();
        let back = backbuffer(&mut graph);
        graph
            .pass("world", PassKind::Graphics, ())
            .color(back, LoadOp::Clear)
            .id();
        graph
            .pass("overlay", PassKind::Graphics, ())
            .color(back, LoadOp::Load)
            .id();
        graph.pass("present", PassKind::Present, ()).present(back).root().id();
        let baked = bake(&graph, BakeOptions::default()).unwrap();
        assert_eq!(names(&graph, &baked), ["world", "overlay", "present"]);
        assert!(baked.passes[0].accesses[0].discard && !baked.passes[1].accesses[0].discard);
    }

    #[test]
    fn aliasing_reuses_disjoint_transients_with_equal_descriptions() {
        let mut graph = FrameGraph::new();
        let a = image(&mut graph, "a", ResourceClass::Transient);
        let b = image(&mut graph, "b", ResourceClass::Transient);
        let c = image(&mut graph, "c", ResourceClass::Transient);
        let back = backbuffer(&mut graph);
        let storage = |stage| Role::StorageImageWrite(stage);
        graph
            .pass("make a", PassKind::Compute, ())
            .discard(a, storage(ShaderStage::Compute))
            .id();
        graph
            .pass("a to b", PassKind::Compute, ())
            .access(a, Role::StorageImageRead(ShaderStage::Compute))
            .discard(b, storage(ShaderStage::Compute))
            .id();
        graph
            .pass("b to c", PassKind::Compute, ())
            .access(b, Role::StorageImageRead(ShaderStage::Compute))
            .discard(c, storage(ShaderStage::Compute))
            .id();
        graph
            .pass("show c", PassKind::Graphics, ())
            .sampled(c, ShaderStage::Fragment)
            .color(back, LoadOp::Clear)
            .id();
        graph.pass("present", PassKind::Present, ()).present(back).root().id();

        let plain = bake(&graph, BakeOptions { aliasing: false }).unwrap();
        let aliased = bake(&graph, BakeOptions::default()).unwrap();
        assert_eq!(plain.physical.len(), 3);
        assert_eq!(aliased.physical.len(), 2, "a and c share one allocation");
        assert!(aliased.estimated_bytes() < plain.estimated_bytes());
        assert_eq!(
            aliased.resources[a.index()].physical,
            aliased.resources[c.index()].physical
        );
        assert_eq!(
            aliased.handoffs,
            vec![AliasHandoff {
                physical: 0,
                from: a,
                to: c,
                at_pass: 2,
            }]
        );
        assert!(aliased.passes[2].accesses[1].discard, "the new member starts undefined");
    }

    #[test]
    fn buffers_alias_only_with_equal_size_and_usage() {
        let mut graph = FrameGraph::new();
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let small = graph.buffer("small", BufferDesc { size: 64, usage }, ResourceClass::Transient);
        let large = graph.buffer("large", BufferDesc { size: 128, usage }, ResourceClass::Transient);
        let out = image(&mut graph, "out", ResourceClass::Transient);
        let back = backbuffer(&mut graph);
        let write = Role::StorageBufferWrite(ShaderStage::Compute);
        let read = Role::StorageBufferRead(ShaderStage::Compute);
        graph
            .pass("fill small", PassKind::Compute, ())
            .discard(small, write)
            .id();
        graph
            .pass("small to large", PassKind::Compute, ())
            .access(small, read)
            .discard(large, write)
            .id();
        graph
            .pass("large to image", PassKind::Compute, ())
            .access(large, read)
            .discard(out, Role::StorageImageWrite(ShaderStage::Compute))
            .id();
        graph
            .pass("show", PassKind::Graphics, ())
            .sampled(out, ShaderStage::Fragment)
            .color(back, LoadOp::Clear)
            .id();
        graph.pass("present", PassKind::Present, ()).present(back).root().id();
        let baked = bake(&graph, BakeOptions::default()).unwrap();
        assert_ne!(
            baked.resources[small.index()].physical,
            baked.resources[large.index()].physical,
            "different sizes never alias (the baseline's tautology)"
        );
    }

    #[test]
    fn scoped_passes_cannot_cross_views() {
        let player = Scope(1);
        let full = Scope(2);
        let mut graph = FrameGraph::new();
        let full_image = image(&mut graph, "full target", ResourceClass::External);
        graph.set_scope(full_image, full);
        let back = backbuffer(&mut graph);
        graph.set_scope(back, player);
        graph
            .pass("full draw", PassKind::Graphics, ())
            .scope(full)
            .color(full_image, LoadOp::Clear)
            .root()
            .id();
        graph
            .pass("player draw", PassKind::Graphics, ())
            .scope(player)
            .sampled(full_image, ShaderStage::Fragment)
            .color(back, LoadOp::Clear)
            .id();
        graph
            .pass("present", PassKind::Present, ())
            .scope(player)
            .present(back)
            .root()
            .id();
        assert!(matches!(
            bake(&graph, BakeOptions::default()),
            Err(BakeError::CrossScope { pass, resource }) if pass == "player draw" && resource == "full target"
        ));
    }

    #[test]
    fn conditional_pass_does_not_change_the_plan() {
        let graph = chain();
        let mut conditional = chain();
        conditional.passes[1].conditional = true;
        let a = bake(&graph, BakeOptions::default()).unwrap();
        let b = bake(&conditional, BakeOptions::default()).unwrap();
        assert_eq!(a.order().collect::<Vec<_>>(), b.order().collect::<Vec<_>>());
        for (x, y) in a.passes.iter().zip(&b.passes) {
            assert_eq!(x.accesses, y.accesses);
        }
    }

    #[test]
    fn signature_tracks_topology_only() {
        let a = chain();
        let mut renamed = chain();
        renamed.passes[0].name = "other".into();
        assert_eq!(a.signature(), renamed.signature());
        let mut changed = chain();
        changed.passes[2].intents[1].role = Role::ColorAttachment(LoadOp::DontCare);
        assert_ne!(a.signature(), changed.signature());
    }
}
