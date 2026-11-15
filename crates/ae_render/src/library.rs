//! Engine pass library: the passes a scene's graph wires up by name (the
//! terrain, the units and their shadows, the water surface, readback,
//! present), with the resources each needs and the roles it uses them in.
//!
//! A graph, native or declared by a render module, uses a library pass by
//! naming it ([`LibraryPass::from_name`]) and giving it its resources in
//! the roles of [`LibraryPass::requires`]. Library resources (the terrain's
//! draw list, the unit buffers, the readback buffer) belong to the host,
//! which alone knows their sizes: a graph declares them as `External`
//! buffers of size 0 under their library name ([`LibraryResource`]), and
//! the host fills in size and usage before baking ([`resolve`]).
//! [`check`] rejects a graph that wires a library pass without its
//! resources and roles, gives it other resources, or declares a library
//! resource as something else; errors name the pass and the resource.
//!
//! The units' shadow map (`shadows.map`, written by `shadows.draw`) is
//! the graph's choice: a graph that declares it must have every receiver
//! (`terrain.draw`, `units.draw`, `water.draw`) sample it
//! ([`LibraryPass::receives`], [`check_shadows`]), and the host then
//! shades by it; a graph without it renders without the units' shadows.
//! The particles' coloured shadows (`shadows.color`, written by
//! `shadows.particles`) come with it: a graph that declares them also
//! declares `shadows.map`, and every receiver samples both.
//!
//! Particles (`particles.sim`, `particles.draw`) are drawn after
//! everything opaque and the water: the draw blends into the colour target
//! and samples the depth buffer (it tests depth itself, fading where a
//! particle meets the scene), so it takes the depth as
//! `Sampled(Fragment)` rather than as an attachment.

use ae_framegraph::{BufferDesc, FrameGraph, PassKind, ResourceClass, ResourceDesc, ResourceId, Role, ShaderStage};
use ash::vk;
use std::fmt;

/// Host-owned resources a graph can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryResource {
    TerrainLeaves,
    TerrainIndirect,
    UnitInstances,
    UnitVisible,
    UnitIndirect,
    UnitCounts,
    /// Offscreen output: the image (sized and formatted by the declaration,
    /// checked against the target) and the host buffer it is copied to.
    ReadbackTarget,
    ReadbackBuffer,
    /// The units' shadow map: a depth image of fixed size
    /// ([`ae_render_abi::module::shadow_map_spec`]).
    ShadowMap,
    /// The particles' pool (`particles.sim` spawns and moves them).
    ParticleState,
    /// The live particles' sort keys, back to front (`particles.sim`
    /// writes, `particles.draw` reads).
    ParticleKeys,
    /// The particles' indirect draw (its instance count is the live count).
    ParticleIndirect,
    /// The particles' coloured shadows: optical depth per channel and the
    /// sun-nearest particle's depth, laid out like [`Self::ShadowMap`]
    /// ([`ae_render_abi::module::particle_shadow_spec`]).
    ParticleShadows,
}

impl LibraryResource {
    pub const ALL: [LibraryResource; 13] = [
        Self::TerrainLeaves,
        Self::TerrainIndirect,
        Self::UnitInstances,
        Self::UnitVisible,
        Self::UnitIndirect,
        Self::UnitCounts,
        Self::ReadbackTarget,
        Self::ReadbackBuffer,
        Self::ShadowMap,
        Self::ParticleState,
        Self::ParticleKeys,
        Self::ParticleIndirect,
        Self::ParticleShadows,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::TerrainLeaves => "terrain.leaves",
            Self::TerrainIndirect => "terrain.indirect",
            Self::UnitInstances => "units.instances",
            Self::UnitVisible => "units.visible",
            Self::UnitIndirect => "units.indirect",
            Self::UnitCounts => "units.counts",
            Self::ReadbackTarget => "readback.target",
            Self::ReadbackBuffer => "readback.buffer",
            Self::ShadowMap => ae_render_abi::module::SHADOW_MAP_NAME,
            Self::ParticleState => "particles.state",
            Self::ParticleKeys => "particles.keys",
            Self::ParticleIndirect => "particles.indirect",
            Self::ParticleShadows => ae_render_abi::module::PARTICLE_SHADOW_NAME,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|r| r.name() == name)
    }

    pub fn is_image(self) -> bool {
        matches!(self, Self::ReadbackTarget | Self::ShadowMap | Self::ParticleShadows)
    }
}

/// Passes a graph can wire up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryPass {
    TerrainLod,
    TerrainDraw,
    UnitsReset,
    UnitsCull,
    UnitsDraw,
    UnitsCount,
    /// The units' shadow map: every unit (not only the visible ones) drawn
    /// from the sun into [`LibraryResource::ShadowMap`], depth only.
    ShadowsDraw,
    /// The particles' coloured shadows: every live particle that is not
    /// emissive (the simulation lists them after the visible keys, with
    /// their own indirect draw) drawn from the sun into
    /// [`LibraryResource::ParticleShadows`].
    ShadowsParticles,
    /// The water surface over the terrain and units: tests against their
    /// depth read-only and blends into the colour target.
    WaterDraw,
    /// Spawn, move and sort the particles (compute).
    ParticlesSim,
    /// The particles, back to front, blended over the scene; samples the
    /// depth buffer.
    ParticlesDraw,
    /// Copy an image to [`LibraryResource::ReadbackBuffer`].
    ReadbackCopy,
    /// Records nothing: the pass that declares host reads.
    HostRead,
    /// Records nothing: the present pass.
    Present,
}

/// What a graphics library pass draws into besides its resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attachments {
    pub color: usize,
    pub depth: DepthUse,
}

/// How a graphics library pass uses the depth buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthUse {
    /// `DepthAttachment` (any load op): tests and writes.
    Write,
    /// `DepthRead`: tests only.
    Read,
    /// `Sampled(Fragment)`: the shader reads it (and tests by itself).
    Sampled,
}

impl LibraryPass {
    pub const ALL: [LibraryPass; 14] = [
        Self::TerrainLod,
        Self::TerrainDraw,
        Self::UnitsReset,
        Self::UnitsCull,
        Self::UnitsDraw,
        Self::UnitsCount,
        Self::ShadowsDraw,
        Self::ShadowsParticles,
        Self::WaterDraw,
        Self::ParticlesSim,
        Self::ParticlesDraw,
        Self::ReadbackCopy,
        Self::HostRead,
        Self::Present,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::TerrainLod => "terrain.lod",
            Self::TerrainDraw => "terrain.draw",
            Self::UnitsReset => "units.reset",
            Self::UnitsCull => "units.cull",
            Self::UnitsDraw => "units.draw",
            Self::UnitsCount => "units.count",
            Self::ShadowsDraw => "shadows.draw",
            Self::ShadowsParticles => "shadows.particles",
            Self::WaterDraw => "water.draw",
            Self::ParticlesSim => "particles.sim",
            Self::ParticlesDraw => "particles.draw",
            Self::ReadbackCopy => "readback.copy",
            Self::HostRead => "host.read",
            Self::Present => "present",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name)
    }

    pub fn kind(self) -> PassKind {
        match self {
            Self::TerrainLod | Self::UnitsCull | Self::ParticlesSim => PassKind::Compute,
            Self::TerrainDraw | Self::UnitsDraw | Self::ShadowsDraw | Self::ShadowsParticles | Self::WaterDraw | Self::ParticlesDraw => PassKind::Graphics,
            Self::UnitsReset | Self::UnitsCount | Self::ReadbackCopy | Self::HostRead => PassKind::Transfer,
            Self::Present => PassKind::Present,
        }
    }

    /// The library resources the pass uses, each in exactly this role.
    pub fn requires(self) -> &'static [(LibraryResource, Role)] {
        use LibraryResource as R;
        match self {
            Self::TerrainLod => &[(R::TerrainLeaves, Role::StorageBufferWrite(ShaderStage::Compute)), (R::TerrainIndirect, Role::StorageBufferWrite(ShaderStage::Compute))],
            Self::TerrainDraw => &[(R::TerrainLeaves, Role::StorageBufferRead(ShaderStage::Vertex)), (R::TerrainIndirect, Role::IndirectBuffer)],
            Self::UnitsReset => &[(R::UnitIndirect, Role::TransferDst)],
            Self::UnitsCull => &[
                (R::UnitInstances, Role::StorageBufferRead(ShaderStage::Compute)),
                (R::UnitVisible, Role::StorageBufferWrite(ShaderStage::Compute)),
                (R::UnitIndirect, Role::StorageBufferWrite(ShaderStage::Compute)),
            ],
            Self::UnitsDraw => &[
                (R::UnitInstances, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::UnitVisible, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::UnitIndirect, Role::IndirectBuffer),
            ],
            Self::UnitsCount => &[(R::UnitIndirect, Role::TransferSrc), (R::UnitCounts, Role::TransferDst)],
            Self::ShadowsDraw => &[(R::UnitInstances, Role::StorageBufferRead(ShaderStage::Vertex)), (R::ShadowMap, Role::DepthAttachment(ae_framegraph::LoadOp::Clear))],
            Self::ShadowsParticles => &[
                (R::ParticleState, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::ParticleKeys, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::ParticleIndirect, Role::IndirectBuffer),
                (R::ParticleShadows, Role::ColorAttachment(ae_framegraph::LoadOp::Clear)),
            ],
            Self::ParticlesSim => &[
                (R::ParticleState, Role::StorageBufferWrite(ShaderStage::Compute)),
                (R::ParticleKeys, Role::StorageBufferWrite(ShaderStage::Compute)),
                (R::ParticleIndirect, Role::StorageBufferWrite(ShaderStage::Compute)),
            ],
            Self::ParticlesDraw => &[
                (R::ParticleState, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::ParticleKeys, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::ParticleIndirect, Role::IndirectBuffer),
            ],
            Self::ReadbackCopy => &[(R::ReadbackBuffer, Role::TransferDst)],
            Self::WaterDraw | Self::HostRead | Self::Present => &[],
        }
    }

    /// Attachments a graphics pass needs (one colour target and the depth
    /// buffer, whose formats the host checks).
    pub fn attachments(self) -> Option<Attachments> {
        match self {
            Self::TerrainDraw | Self::UnitsDraw => Some(Attachments { color: 1, depth: DepthUse::Write }),
            Self::WaterDraw => Some(Attachments { color: 1, depth: DepthUse::Read }),
            Self::ParticlesDraw => Some(Attachments { color: 1, depth: DepthUse::Sampled }),
            _ => None,
        }
    }

    /// Whether it needs the unit renderer.
    pub fn uses_units(self) -> bool {
        matches!(self, Self::UnitsReset | Self::UnitsCull | Self::UnitsDraw | Self::UnitsCount | Self::ShadowsDraw)
    }

    /// Whether it needs the particle renderer.
    pub fn uses_particles(self) -> bool {
        matches!(self, Self::ParticlesSim | Self::ParticlesDraw | Self::ShadowsParticles)
    }

    /// Whether it shades by the shadow maps: in a graph that declares
    /// [`LibraryResource::ShadowMap`] (and [`LibraryResource::ParticleShadows`])
    /// it must sample each in this role.
    pub fn receives(self) -> Option<Role> {
        matches!(self, Self::TerrainDraw | Self::UnitsDraw | Self::WaterDraw).then_some(Role::Sampled(ShaderStage::Fragment))
    }
}

/// A graph's use of the library broken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryError(pub String);

impl fmt::Display for LibraryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LibraryError {}

fn error(text: String) -> LibraryError {
    LibraryError(text)
}

/// The host's formats and sizes the graph must agree with.
#[derive(Clone, Copy, Debug)]
pub struct LibraryEnv {
    pub color_format: vk::Format,
    pub depth_format: vk::Format,
    pub extent: vk::Extent2D,
    /// Units exist in this scene.
    pub units: bool,
    /// The scene draws water (the renderer has its pipeline).
    pub water: bool,
    /// The units cast shadows (the renderer has the units' shadow map, and
    /// with particles the particles' coloured shadows).
    pub shadows: bool,
    /// The scene has particles (the particle renderer exists).
    pub particles: bool,
}

/// Library resource of each graph resource (by `ResourceId` index).
pub type ResourceMap = Vec<Option<LibraryResource>>;

/// Fill in library buffer sizes (declared as 0) from `desc` and map every
/// resource to its library resource. A library resource must be declared
/// `External`, as a buffer of size 0 (or exactly its size), for
/// [`LibraryResource::ReadbackTarget`] as an image of the target's extent
/// and colour format, and for [`LibraryResource::ShadowMap`] as its spec
/// (in a scene whose units cast shadows).
pub fn resolve<R>(graph: &mut FrameGraph<R>, env: &LibraryEnv, desc: impl Fn(LibraryResource) -> Option<BufferDesc>) -> Result<ResourceMap, LibraryError> {
    let mut map = Vec::with_capacity(graph.resources.len());
    for resource in &mut graph.resources {
        let Some(lib) = LibraryResource::from_name(&resource.name) else {
            map.push(None);
            continue;
        };
        if resource.class != ResourceClass::External {
            return Err(error(format!("library resource `{}` must be External, not {:?}", resource.name, resource.class)));
        }
        match (&mut resource.desc, lib.is_image()) {
            (ResourceDesc::Image(image), true) if lib == LibraryResource::ShadowMap => {
                use ae_render_abi::module as m;
                if !env.shadows {
                    return Err(error(format!("library resource `{}` does not exist in this scene (its units cast no shadows)", resource.name)));
                }
                let format = vk::Format::from_raw(m::SHADOW_MAP_FORMAT as i32);
                if image.format != format || image.width != m::SHADOW_MAP_WIDTH || image.height != m::SHADOW_MAP_HEIGHT || !image.aspect.contains(vk::ImageAspectFlags::DEPTH) {
                    return Err(error(format!(
                        "`{}` is {:?} {}x{}; the library's is {format:?} {}x{} with a depth aspect (shadow_map_spec)",
                        resource.name, image.format, image.width, image.height, m::SHADOW_MAP_WIDTH, m::SHADOW_MAP_HEIGHT
                    )));
                }
            }
            (ResourceDesc::Image(image), true) if lib == LibraryResource::ParticleShadows => {
                use ae_render_abi::module as m;
                if !(env.shadows && env.particles) {
                    return Err(error(format!("library resource `{}` does not exist in this scene (it needs units that cast shadows and particles)", resource.name)));
                }
                let format = vk::Format::from_raw(m::PARTICLE_SHADOW_FORMAT as i32);
                if image.format != format || image.width != m::PARTICLE_SHADOW_WIDTH || image.height != m::PARTICLE_SHADOW_HEIGHT || !image.aspect.contains(vk::ImageAspectFlags::COLOR) {
                    return Err(error(format!(
                        "`{}` is {:?} {}x{}; the library's is {format:?} {}x{} with a colour aspect (particle_shadow_spec)",
                        resource.name, image.format, image.width, image.height, m::PARTICLE_SHADOW_WIDTH, m::PARTICLE_SHADOW_HEIGHT
                    )));
                }
            }
            (ResourceDesc::Image(image), true) => {
                if image.format != env.color_format || image.width != env.extent.width || image.height != env.extent.height {
                    return Err(error(format!(
                        "`{}` is {:?} {}x{}; the target is {:?} {}x{}",
                        resource.name, image.format, image.width, image.height, env.color_format, env.extent.width, env.extent.height
                    )));
                }
            }
            (ResourceDesc::Buffer(buffer), false) => {
                let Some(want) = desc(lib) else {
                    return Err(error(format!("library resource `{}` does not exist in this scene", resource.name)));
                };
                if buffer.size != 0 && buffer.size != want.size {
                    return Err(error(format!("`{}` is declared with {} bytes; the library's is {} (declare 0)", resource.name, buffer.size, want.size)));
                }
                *buffer = want;
            }
            _ => return Err(error(format!("library resource `{}` declared as the wrong kind", resource.name))),
        }
        map.push(Some(lib));
    }
    Ok(map)
}

/// Check that pass `pass` of `graph`, wired as library pass `lib`, uses
/// what the library needs and nothing else.
pub fn check<R>(graph: &FrameGraph<R>, pass: usize, lib: LibraryPass, map: &ResourceMap, env: &LibraryEnv) -> Result<(), LibraryError> {
    let p = &graph.passes[pass];
    let name = &p.name;
    if p.kind != lib.kind() {
        return Err(error(format!("pass `{name}` ({}) must be a {:?} pass, not {:?}", lib.name(), lib.kind(), p.kind)));
    }
    if lib.uses_units() && !env.units {
        return Err(error(format!("pass `{name}` ({}) needs units; this scene has none", lib.name())));
    }
    if lib.uses_particles() && !env.particles {
        return Err(error(format!("pass `{name}` ({}) needs particles; this scene has none", lib.name())));
    }
    if lib == LibraryPass::WaterDraw && !env.water {
        return Err(error(format!("pass `{name}` (water.draw) needs water; this scene draws none")));
    }
    if matches!(lib, LibraryPass::ShadowsDraw | LibraryPass::ShadowsParticles) && !env.shadows {
        return Err(error(format!("pass `{name}` ({}) needs units that cast shadows; this scene's cast none", lib.name())));
    }
    let resource = |id: ResourceId| &graph.resources[id.index()];
    // The shadow maps, for a receiver in a graph that declares them.
    let receives: Vec<(LibraryResource, Role)> = [LibraryResource::ShadowMap, LibraryResource::ParticleShadows]
        .into_iter()
        .filter(|r| map.contains(&Some(*r)))
        .filter_map(|r| lib.receives().map(|role| (r, role)))
        .collect();
    for &(needed, role) in lib.requires().iter().chain(&receives) {
        if !p.intents.iter().any(|i| map[i.resource.index()] == Some(needed) && i.role == role) {
            let why = if receives.iter().any(|r| r.0 == needed) { format!(" (the graph declares `{}`: every receiver samples it)", needed.name()) } else { String::new() };
            return Err(error(format!("pass `{name}` ({}) must use `{}` as {role:?}{why}", lib.name(), needed.name())));
        }
    }
    let (mut color, mut depth) = (0, 0);
    for intent in &p.intents {
        let r = resource(intent.resource);
        if lib.requires().iter().chain(&receives).any(|&(needed, role)| map[intent.resource.index()] == Some(needed) && intent.role == role) {
            continue;
        }
        match (lib, intent.role) {
            (_, Role::ColorAttachment(_)) if lib.attachments().is_some() => {
                color += 1;
                if let ResourceDesc::Image(image) = r.desc
                    && image.format != env.color_format
                {
                    return Err(error(format!("pass `{name}` draws into `{}` ({:?}); the library's pipelines target {:?}", r.name, image.format, env.color_format)));
                }
            }
            (_, Role::DepthAttachment(_)) | (_, Role::DepthRead) | (_, Role::Sampled(ShaderStage::Fragment))
                if lib.attachments().is_some_and(|a| match a.depth {
                    DepthUse::Write => matches!(intent.role, Role::DepthAttachment(_)),
                    DepthUse::Read => intent.role == Role::DepthRead,
                    DepthUse::Sampled => intent.role == Role::Sampled(ShaderStage::Fragment) && r.desc.is_image(),
                }) =>
            {
                depth += 1;
                if let ResourceDesc::Image(image) = r.desc
                    && image.format != env.depth_format
                {
                    return Err(error(format!("pass `{name}` uses depth `{}` ({:?}); the library's pipelines use {:?}", r.name, image.format, env.depth_format)));
                }
            }
            (LibraryPass::ReadbackCopy, Role::TransferSrc) if r.desc.is_image() => {}
            (LibraryPass::HostRead, Role::HostRead) => {}
            (LibraryPass::Present, Role::Present) => {}
            _ => return Err(error(format!("pass `{name}` ({}) does not use `{}` as {:?}", lib.name(), r.name, intent.role))),
        }
    }
    if let Some(a) = lib.attachments()
        && (color != a.color || depth != 1)
    {
        let role = match a.depth {
            DepthUse::Write => "a depth attachment",
            DepthUse::Read => "a read-only depth attachment (depth_read)",
            DepthUse::Sampled => "the depth buffer sampled in the fragment stage (sampled:fragment)",
        };
        return Err(error(format!("pass `{name}` ({}) needs {} colour attachment(s) and {role}; it has {color} and {depth}", lib.name(), a.color)));
    }
    if lib == LibraryPass::ReadbackCopy && p.intents.iter().filter(|i| i.role == Role::TransferSrc).count() != 1 {
        return Err(error(format!("pass `{name}` (readback.copy) copies one image")));
    }
    if lib == LibraryPass::Present && p.intents.len() != 1 {
        return Err(error(format!("pass `{name}` (present) presents one image")));
    }
    Ok(())
}

/// Graph-wide: a graph that declares the units' shadow map has a
/// `shadows.draw` pass write it, and one that declares the particles'
/// coloured shadows declares the map too and has a `shadows.particles`
/// pass write them (`libs`: each pass's library pass).
pub fn check_shadows<R>(graph: &FrameGraph<R>, map: &ResourceMap, libs: impl Fn(usize) -> Option<LibraryPass>) -> Result<(), LibraryError> {
    let written = |lib: LibraryPass| (0..graph.passes.len()).any(|i| libs(i) == Some(lib));
    let (units, particles) = (map.contains(&Some(LibraryResource::ShadowMap)), map.contains(&Some(LibraryResource::ParticleShadows)));
    if units && !written(LibraryPass::ShadowsDraw) {
        return Err(error("the graph declares `shadows.map` but no `shadows.draw` pass writes it".into()));
    }
    if particles && !units {
        return Err(error("the graph declares `shadows.color` without `shadows.map` (the coloured shadows use its cascades)".into()));
    }
    if particles && !written(LibraryPass::ShadowsParticles) {
        return Err(error("the graph declares `shadows.color` but no `shadows.particles` pass writes it".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::app::{TerrainOutput, native_graph};

    fn env() -> LibraryEnv {
        LibraryEnv { color_format: vk::Format::R8G8B8A8_SRGB, depth_format: vk::Format::D32_SFLOAT, extent: vk::Extent2D { width: 64, height: 32 }, units: true, water: true, shadows: true, particles: true }
    }

    fn size(r: LibraryResource) -> Option<BufferDesc> {
        Some(BufferDesc { size: 64 + r as u64, usage: vk::BufferUsageFlags::STORAGE_BUFFER })
    }

    /// Resolve and check every pass; the first error.
    fn wire(mut graph: FrameGraph<LibraryPass>, env: &LibraryEnv) -> Result<FrameGraph<LibraryPass>, LibraryError> {
        let map = resolve(&mut graph, env, size)?;
        for (i, p) in graph.passes.iter().enumerate() {
            check(&graph, i, p.record, &map, env)?;
        }
        Ok(graph)
    }

    fn pass(graph: &FrameGraph<LibraryPass>, lib: LibraryPass) -> usize {
        graph.passes.iter().position(|p| p.record == lib).unwrap()
    }

    #[test]
    fn the_native_graph_wires_and_gets_library_sizes() {
        for output in [TerrainOutput::Present, TerrainOutput::Readback] {
            let graph = wire(native_graph(&env(), output), &env()).unwrap();
            let leaves = graph.resources.iter().find(|r| r.name == "terrain.leaves").unwrap();
            assert_eq!(leaves.desc, ae_framegraph::ResourceDesc::Buffer(size(LibraryResource::TerrainLeaves).unwrap()));
        }
        let no_units = LibraryEnv { units: false, ..env() };
        assert!(wire(native_graph(&no_units, TerrainOutput::Present), &no_units).is_ok());
        let dry = LibraryEnv { water: false, ..env() };
        let graph = wire(native_graph(&dry, TerrainOutput::Present), &dry).unwrap();
        assert!(!graph.passes.iter().any(|p| p.record == LibraryPass::WaterDraw), "no water pass without water");
        let graph = native_graph(&env(), TerrainOutput::Readback);
        let (units, water) = (pass(&graph, LibraryPass::UnitsDraw), pass(&graph, LibraryPass::WaterDraw));
        assert!(water > units, "the water is declared after the units");
    }

    #[test]
    fn broken_wiring_is_rejected_by_name() {
        let err = |graph: FrameGraph<LibraryPass>, env: &LibraryEnv| wire(graph, env).unwrap_err().0;
        let base = || native_graph(&env(), TerrainOutput::Present);

        // A required role missing.
        let mut g = base();
        let cull = pass(&g, LibraryPass::UnitsCull);
        g.passes[cull].intents.retain(|i| !matches!(i.role, Role::StorageBufferWrite(_)) || g.resources[i.resource.index()].name != "units.visible");
        let e = err(g, &env());
        assert!(e.contains("`units cull`") && e.contains("`units.visible`"), "{e}");

        // The wrong pass kind.
        let mut g = base();
        let lod = pass(&g, LibraryPass::TerrainLod);
        g.passes[lod].kind = PassKind::Graphics;
        assert!(err(g, &env()).contains("must be a Compute pass"));

        // An extra resource the library pass does not use.
        let mut g = base();
        let lod = pass(&g, LibraryPass::TerrainLod);
        let extra = g.buffer("mine", BufferDesc { size: 16, usage: vk::BufferUsageFlags::STORAGE_BUFFER }, ResourceClass::Transient);
        g.passes[lod].intents.push(ae_framegraph::Intent { resource: extra, role: Role::StorageBufferRead(ShaderStage::Compute), discard: false });
        let e = err(g, &env());
        assert!(e.contains("does not use `mine`"), "{e}");

        // Attachments in formats the library's pipelines do not target.
        let other = LibraryEnv { color_format: vk::Format::B8G8R8A8_SRGB, ..env() };
        let e = err(native_graph(&other, TerrainOutput::Present), &env());
        assert!(e.contains("draws into `target`"), "{e}");
        let other = LibraryEnv { depth_format: vk::Format::D32_SFLOAT_S8_UINT, ..env() };
        let e = err(native_graph(&other, TerrainOutput::Present), &env());
        assert!(e.contains("uses depth `depth`"), "{e}");

        // A library resource declared as something else.
        let mut g = base();
        let i = g.resources.iter().position(|r| r.name == "terrain.leaves").unwrap();
        g.resources[i].class = ResourceClass::Transient;
        assert!(err(g, &env()).contains("`terrain.leaves` must be External"));
        let mut g = base();
        g.resources[i].desc = ae_framegraph::ResourceDesc::Buffer(BufferDesc { size: 12, usage: vk::BufferUsageFlags::STORAGE_BUFFER });
        assert!(err(g, &env()).contains("declare 0"));

        // Unit passes in a scene without units.
        let no_units = LibraryEnv { units: false, ..env() };
        let e = err(base(), &no_units);
        assert!(e.contains("does not exist in this scene") || e.contains("needs units"), "{e}");

        // The water pass writing depth instead of testing it read-only, or
        // without depth.
        let mut g = base();
        let water = pass(&g, LibraryPass::WaterDraw);
        for intent in &mut g.passes[water].intents {
            if intent.role == Role::DepthRead {
                intent.role = Role::DepthAttachment(ae_framegraph::LoadOp::Load);
            }
        }
        let e = err(g, &env());
        assert!(e.contains("`water`") && e.contains("does not use `depth`"), "{e}");
        let mut g = base();
        g.passes[water].intents.retain(|i| i.role != Role::DepthRead);
        let e = err(g, &env());
        assert!(e.contains("read-only depth attachment"), "{e}");
        // The terrain reading depth only.
        let mut g = base();
        let draw = pass(&g, LibraryPass::TerrainDraw);
        for intent in &mut g.passes[draw].intents {
            if intent.role.is_depth() {
                intent.role = Role::DepthRead;
            }
        }
        assert!(err(g, &env()).contains("does not use `depth` as DepthRead"));
        // A water pass in a scene without water.
        let dry = LibraryEnv { water: false, ..env() };
        let e = err(base(), &dry);
        assert!(e.contains("needs water"), "{e}");

        // A readback target of the wrong extent.
        let other = LibraryEnv { extent: vk::Extent2D { width: 65, height: 32 }, ..env() };
        let e = err(native_graph(&other, TerrainOutput::Readback), &env());
        assert!(e.contains("`readback.target`"), "{e}");

        // Particles: the depth as a read-only attachment instead of
        // sampled, and particle passes in a scene without particles.
        let mut g = base();
        let draw = pass(&g, LibraryPass::ParticlesDraw);
        let depth = g.passes[draw].intents.iter().position(|i| i.role == Role::Sampled(ShaderStage::Fragment)).unwrap();
        g.passes[draw].intents[depth].role = Role::DepthRead;
        let e = err(g, &env());
        assert!(e.contains("`particles`") && e.contains("DepthRead"), "{e}");
        let still = LibraryEnv { particles: false, ..env() };
        let e = err(base(), &still);
        assert!(e.contains("needs particles") || e.contains("`shadows.color` does not exist"), "{e}");

        // The particles' coloured shadows: a receiver that does not sample
        // them, their pass loading instead of clearing, a graph without
        // their pass or without the units' map.
        let mut g = base();
        let water = pass(&g, LibraryPass::WaterDraw);
        let tint = g.resources.iter().position(|r| r.name == "shadows.color").unwrap();
        g.passes[water].intents.retain(|i| i.resource.index() != tint);
        let e = err(g, &env());
        assert!(e.contains("`water`") && e.contains("`shadows.color`") && e.contains("every receiver"), "{e}");
        let mut g = base();
        let cast = pass(&g, LibraryPass::ShadowsParticles);
        for intent in &mut g.passes[cast].intents {
            if matches!(intent.role, Role::ColorAttachment(_)) {
                intent.role = Role::ColorAttachment(ae_framegraph::LoadOp::Load);
            }
        }
        let e = err(g, &env());
        assert!(e.contains("`particle shadows`") && e.contains("`shadows.color`"), "{e}");
        let shadows = |g: &FrameGraph<LibraryPass>| {
            let mut g = FrameGraph { resources: g.resources.clone(), passes: g.passes.clone(), roots: g.roots.clone(), plan_version: g.plan_version };
            let map = resolve(&mut g, &env(), size).unwrap();
            check_shadows(&g, &map, |i| Some(g.passes[i].record)).map_err(|e| e.0)
        };
        let g = base();
        assert_eq!(shadows(&g), Ok(()));
        let mut g = base();
        g.passes.retain(|p| p.record != LibraryPass::ShadowsParticles);
        let e = shadows(&g).unwrap_err();
        assert!(e.contains("no `shadows.particles` pass"), "{e}");
        let mut g = base();
        let i = g.resources.iter().position(|r| r.name == "shadows.map").unwrap();
        g.resources[i].name = "not the map".into();
        let e = shadows(&g).unwrap_err();
        assert!(e.contains("without `shadows.map`"), "{e}");
    }
}
