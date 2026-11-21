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
//! ([`LibraryPass::receives`], [`check_graph`]), and the host then
//! shades by it; a graph without it renders without the units' shadows.
//! The particles' coloured shadows (`shadows.color`, written by
//! `shadows.particles`) come with it: a graph that declares them also
//! declares `shadows.map`, and every receiver samples both.
//!
//! The clustered lights (`lights.list`, `lights.clusters`, written by
//! `lights.cluster`; [`crate::lights`]) are the graph's choice the same
//! way: a graph that declares them has every lit pass (the three receivers
//! in the fragment stage, `particles.sim` in compute) read both
//! ([`LibraryPass::lit`]); a graph without them renders unlit by them.
//!
//! Particles (`particles.sim`, `particles.draw`) are drawn after
//! everything opaque and the water: the draw blends into the colour target
//! and samples the depth buffer (it tests depth itself, fading where a
//! particle meets the scene), so it takes the depth as
//! `Sampled(Fragment)` rather than as an attachment.
//!
//! The glow (`glow`, [`ae_render_abi::module::GLOW_NAME`]) is the graph's
//! choice the same way: a graph that declares it has every scene draw write
//! it as a second colour attachment (the light sources only; the rest
//! black) and the three `post.bloom` passes read it at a quarter of the
//! size. A graph without it draws the same scene with no bloom.
//!
//! Automatic exposure ([`crate::exposure`], [`LibraryResource::Exposure`]):
//! a graph that declares the 1×1 exposure image has the `exposure.clear`,
//! `exposure.meter` and `exposure.adapt` passes, and `post.bloom`'s
//! downsample and `post.tonemap` sample the image (the scenes are stored at
//! the camera's reference EV and the tonemap applies the eased one). All or
//! none: the checks reject a partial declaration or the wrong order.

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
    /// The skinning palettes of the units with models (`units.pose`
    /// writes, `units.draw` and `shadows.draw` read).
    UnitPalette,
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
    /// The frame's light list (the host writes it; `lights.cluster` counts
    /// into its header): optional, like the shadow maps. A graph that
    /// declares it and [`Self::LightClusters`] has every lit pass read
    /// both ([`LibraryPass::lit`]).
    LightList,
    /// The lights of each view cluster (`lights.cluster` writes).
    LightClusters,
    /// The 1×1 exposure image (a small RGBA32F buffer): the metered, eased
    /// EV100. `exposure.adapt` writes it; `post.bloom`'s downsample and
    /// `post.tonemap` sample it ([`ae_render_abi::module::EXPOSURE_NAME`]).
    Exposure,
    /// The metering histogram ([`ae_render_abi::module::EXPOSURE_HISTOGRAM_NAME`]):
    /// `exposure.clear` zeroes it, `exposure.meter` adds counts,
    /// `exposure.adapt` reads it.
    ExposureHistogram,
}

impl LibraryResource {
    pub const ALL: [LibraryResource; 18] = [
        Self::TerrainLeaves,
        Self::TerrainIndirect,
        Self::UnitInstances,
        Self::UnitVisible,
        Self::UnitIndirect,
        Self::UnitCounts,
        Self::UnitPalette,
        Self::ReadbackTarget,
        Self::ReadbackBuffer,
        Self::ShadowMap,
        Self::ParticleState,
        Self::ParticleKeys,
        Self::ParticleIndirect,
        Self::ParticleShadows,
        Self::LightList,
        Self::LightClusters,
        Self::Exposure,
        Self::ExposureHistogram,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::TerrainLeaves => "terrain.leaves",
            Self::TerrainIndirect => "terrain.indirect",
            Self::UnitInstances => "units.instances",
            Self::UnitVisible => "units.visible",
            Self::UnitIndirect => "units.indirect",
            Self::UnitCounts => "units.counts",
            Self::UnitPalette => "units.palette",
            Self::ReadbackTarget => "readback.target",
            Self::ReadbackBuffer => "readback.buffer",
            Self::ShadowMap => ae_render_abi::module::SHADOW_MAP_NAME,
            Self::ParticleState => "particles.state",
            Self::ParticleKeys => "particles.keys",
            Self::ParticleIndirect => "particles.indirect",
            Self::ParticleShadows => ae_render_abi::module::PARTICLE_SHADOW_NAME,
            Self::LightList => "lights.list",
            Self::LightClusters => "lights.clusters",
            Self::Exposure => ae_render_abi::module::EXPOSURE_NAME,
            Self::ExposureHistogram => ae_render_abi::module::EXPOSURE_HISTOGRAM_NAME,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|r| r.name() == name)
    }

    pub fn is_image(self) -> bool {
        matches!(self, Self::ReadbackTarget | Self::ShadowMap | Self::ParticleShadows | Self::Exposure)
    }
}

/// Passes a graph can wire up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryPass {
    TerrainLod,
    TerrainDraw,
    UnitsReset,
    UnitsCull,
    /// The units' poses: each animated unit's skinning palette, once per
    /// bone per frame, into [`LibraryResource::UnitPalette`] (compute).
    UnitsPose,
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
    /// Each view cluster's lights, from [`LibraryResource::LightList`]
    /// into [`LibraryResource::LightClusters`] (compute).
    LightsCluster,
    /// The HDR scene image (sampled in the fragment stage, of the scene's
    /// format) to the target in an SDR graph (where the tonemap encodes), or
    /// to the display ([`crate::encode::DISPLAY_FORMAT`], paper-white units)
    /// in an HDR one: the tone curve (the scene is stored exposed), plus the
    /// bloom's light when the graph has a `glow` and `post.bloom` passes and
    /// the display exposure when it declares the exposure image.
    Tonemap,
    /// The display intermediate (sampled in the fragment stage) to the
    /// output's encoding (one colour attachment of the target's format), in
    /// an HDR graph only: ST.2084 or scRGB, clamped at the display's peak
    /// ([`crate::encode`]).
    PostEncode,
    /// The metering histogram's counts zeroed for the frame (`transfer`;
    /// `cmd_fill_buffer`).
    ExposureClear,
    /// The scene's log luminance into the histogram (compute): the scene
    /// sampled in the compute stage at a quarter of the size, one count per
    /// 4×4 block.
    ExposureMeter,
    /// The histogram into the eased EV100, the 1×1 exposure image (compute,
    /// one invocation).
    ExposureAdapt,
    /// The glow target's brightest mean per 4×4 block, at a quarter of the
    /// size: `bloom.downsample` (the glow `Sampled(Fragment)`, the first
    /// bloom image a colour attachment).
    BloomDownsample,
    /// The 11-tap Gaussian along x, then along y: `bloom.blur_x` and
    /// `bloom.blur_y` (one bloom image sampled, the other a colour
    /// attachment).
    BloomBlurX,
    BloomBlurY,
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
    /// The second colour attachment is the graph's `glow` image (when it
    /// declares one): every scene draw writes it.
    pub glow: bool,
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
    pub const ALL: [LibraryPass; 24] = [
        Self::TerrainLod,
        Self::TerrainDraw,
        Self::UnitsReset,
        Self::UnitsCull,
        Self::UnitsPose,
        Self::UnitsDraw,
        Self::UnitsCount,
        Self::ShadowsDraw,
        Self::ShadowsParticles,
        Self::WaterDraw,
        Self::ParticlesSim,
        Self::ParticlesDraw,
        Self::LightsCluster,
        Self::Tonemap,
        Self::PostEncode,
        Self::BloomDownsample,
        Self::BloomBlurX,
        Self::BloomBlurY,
        Self::ExposureClear,
        Self::ExposureMeter,
        Self::ExposureAdapt,
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
            Self::UnitsPose => "units.pose",
            Self::UnitsDraw => "units.draw",
            Self::UnitsCount => "units.count",
            Self::ShadowsDraw => "shadows.draw",
            Self::ShadowsParticles => "shadows.particles",
            Self::WaterDraw => "water.draw",
            Self::ParticlesSim => "particles.sim",
            Self::ParticlesDraw => "particles.draw",
            Self::LightsCluster => "lights.cluster",
            Self::Tonemap => "post.tonemap",
            Self::PostEncode => "post.encode",
            Self::BloomDownsample => "bloom.downsample",
            Self::BloomBlurX => "bloom.blur_x",
            Self::BloomBlurY => "bloom.blur_y",
            Self::ExposureClear => "exposure.clear",
            Self::ExposureMeter => "exposure.meter",
            Self::ExposureAdapt => "exposure.adapt",
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
            Self::TerrainLod | Self::UnitsCull | Self::UnitsPose | Self::ParticlesSim | Self::LightsCluster | Self::ExposureMeter | Self::ExposureAdapt => PassKind::Compute,
            Self::TerrainDraw | Self::UnitsDraw | Self::ShadowsDraw | Self::ShadowsParticles | Self::WaterDraw | Self::ParticlesDraw | Self::Tonemap | Self::PostEncode | Self::BloomDownsample | Self::BloomBlurX | Self::BloomBlurY => PassKind::Graphics,
            Self::UnitsReset | Self::UnitsCount | Self::ReadbackCopy | Self::HostRead | Self::ExposureClear => PassKind::Transfer,
            Self::Present => PassKind::Present,
        }
    }

    /// The library resources the pass uses, each in exactly this role.
    pub fn requires(self) -> &'static [(LibraryResource, Role)] {
        use LibraryResource as R;
        match self {
            Self::ExposureClear => &[(R::ExposureHistogram, Role::TransferDst)],
            Self::ExposureMeter => &[(R::ExposureHistogram, Role::StorageBufferReadWrite(ShaderStage::Compute))],
            // Read and write: the previous frame's EV is the easing's start.
            Self::ExposureAdapt => &[
                (R::ExposureHistogram, Role::StorageBufferRead(ShaderStage::Compute)),
                (R::Exposure, Role::StorageImageReadWrite(ShaderStage::Compute)),
            ],
            Self::TerrainLod => &[(R::TerrainLeaves, Role::StorageBufferWrite(ShaderStage::Compute)), (R::TerrainIndirect, Role::StorageBufferWrite(ShaderStage::Compute))],
            Self::TerrainDraw => &[(R::TerrainLeaves, Role::StorageBufferRead(ShaderStage::Vertex)), (R::TerrainIndirect, Role::IndirectBuffer)],
            Self::UnitsReset => &[(R::UnitIndirect, Role::TransferDst)],
            Self::UnitsCull => &[
                (R::UnitInstances, Role::StorageBufferRead(ShaderStage::Compute)),
                (R::UnitVisible, Role::StorageBufferWrite(ShaderStage::Compute)),
                (R::UnitIndirect, Role::StorageBufferWrite(ShaderStage::Compute)),
            ],
            Self::UnitsPose => &[(R::UnitInstances, Role::StorageBufferRead(ShaderStage::Compute)), (R::UnitPalette, Role::StorageBufferWrite(ShaderStage::Compute))],
            Self::UnitsDraw => &[
                (R::UnitInstances, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::UnitVisible, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::UnitIndirect, Role::IndirectBuffer),
                (R::UnitPalette, Role::StorageBufferRead(ShaderStage::Vertex)),
            ],
            Self::UnitsCount => &[(R::UnitIndirect, Role::TransferSrc), (R::UnitCounts, Role::TransferDst)],
            Self::ShadowsDraw => &[
                (R::UnitInstances, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::UnitVisible, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::UnitIndirect, Role::IndirectBuffer),
                (R::UnitPalette, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::ShadowMap, Role::DepthAttachment(ae_framegraph::LoadOp::Clear)),
            ],
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
            Self::LightsCluster => &[(R::LightList, Role::StorageBufferWrite(ShaderStage::Compute)), (R::LightClusters, Role::StorageBufferWrite(ShaderStage::Compute))],
            Self::ReadbackCopy => &[(R::ReadbackBuffer, Role::TransferDst)],
            Self::WaterDraw | Self::Tonemap | Self::PostEncode | Self::BloomDownsample | Self::BloomBlurX | Self::BloomBlurY | Self::HostRead | Self::Present => &[],
        }
    }

    /// Attachments a graphics pass needs (one colour target and the depth
    /// buffer, whose formats the host checks).
    pub fn attachments(self) -> Option<Attachments> {
        match self {
            Self::TerrainDraw | Self::UnitsDraw => Some(Attachments { color: 1, glow: true, depth: DepthUse::Write }),
            Self::WaterDraw => Some(Attachments { color: 1, glow: true, depth: DepthUse::Read }),
            Self::ParticlesDraw => Some(Attachments { color: 1, glow: true, depth: DepthUse::Sampled }),
            _ => None,
        }
    }

    /// Whether it needs the unit renderer.
    pub fn uses_units(self) -> bool {
        matches!(self, Self::UnitsReset | Self::UnitsCull | Self::UnitsPose | Self::UnitsDraw | Self::UnitsCount | Self::ShadowsDraw)
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

    /// Whether it is lit by the clustered lights: in a graph that declares
    /// [`LibraryResource::LightList`] and [`LibraryResource::LightClusters`]
    /// it must read both in this role.
    pub fn lit(self) -> Option<Role> {
        match self {
            Self::TerrainDraw | Self::UnitsDraw | Self::WaterDraw => Some(Role::StorageBufferRead(ShaderStage::Fragment)),
            Self::ParticlesSim => Some(Role::StorageBufferRead(ShaderStage::Compute)),
            _ => None,
        }
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
    /// The output's format (the swapchain's or the readback image's):
    /// what `post.tonemap` writes.
    pub color_format: vk::Format,
    /// The HDR scene's (`ae_render_abi::module::SCENE_FORMAT`): what the
    /// scene's draws write and `post.tonemap` samples.
    pub scene_format: vk::Format,
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
    /// How the output image encodes the display's light (SDR has no display
    /// intermediate; the tonemap writes the target, as before).
    pub encoding: crate::encode::OutputEncoding,
    /// The display's paper white and peak, in nits (unused in SDR).
    pub levels: crate::encode::DisplayLevels,
}

impl LibraryEnv {
    /// The image `post.tonemap` writes: the display ([`crate::encode::DISPLAY_FORMAT`],
    /// paper-white units) in an HDR graph, the output in an SDR one.
    pub fn display_format(&self) -> vk::Format {
        if self.encoding.uses_display() {
            crate::encode::DISPLAY_FORMAT
        } else {
            self.color_format
        }
    }
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
            (ResourceDesc::Image(image), true) if lib == LibraryResource::Exposure => {
                use ae_render_abi::module as m;
                let format = vk::Format::from_raw(m::EXPOSURE_FORMAT as i32);
                if image.format != format || image.width != 1 || image.height != 1 || !image.aspect.contains(vk::ImageAspectFlags::COLOR) {
                    return Err(error(format!(
                        "`{}` is {:?} {}x{}; the exposure image is {format:?} 1x1 with a colour aspect (exposure_spec)",
                        resource.name, image.format, image.width, image.height
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
        .chain([LibraryResource::LightList, LibraryResource::LightClusters].into_iter().filter(|r| map.contains(&Some(*r))).filter_map(|r| lib.lit().map(|role| (r, role))))
        .collect();
    for &(needed, role) in lib.requires().iter().chain(&receives) {
        if !p.intents.iter().any(|i| map[i.resource.index()] == Some(needed) && i.role == role) {
            let why = match needed {
                LibraryResource::LightList | LibraryResource::LightClusters if receives.iter().any(|r| r.0 == needed) => format!(" (the graph declares `{}`: every lit pass reads it)", needed.name()),
                _ if receives.iter().any(|r| r.0 == needed) => format!(" (the graph declares `{}`: every receiver samples it)", needed.name()),
                _ => String::new(),
            };
            return Err(error(format!("pass `{name}` ({}) must use `{}` as {role:?}{why}", lib.name(), needed.name())));
        }
    }
    if lib == LibraryPass::Tonemap {
        return check_tonemap(graph, pass, env);
    }
    if lib == LibraryPass::PostEncode {
        return check_encode(graph, pass, env);
    }
    if lib == LibraryPass::ExposureMeter {
        return check_exposure_meter(graph, pass, env);
    }
    if matches!(lib, LibraryPass::BloomDownsample | LibraryPass::BloomBlurX | LibraryPass::BloomBlurY) {
        return check_bloom(graph, pass, lib, env);
    }
    let glow = glow_resource(graph).is_some();
    let glow_format = vk::Format::from_raw(ae_render_abi::module::GLOW_FORMAT as i32);
    let (mut scene, mut glow_color, mut depth) = (0, 0, 0);
    for intent in &p.intents {
        let r = resource(intent.resource);
        if lib.requires().iter().chain(&receives).any(|&(needed, role)| map[intent.resource.index()] == Some(needed) && intent.role == role) {
            continue;
        }
        match (lib, intent.role) {
            (_, Role::ColorAttachment(_)) if lib.attachments().is_some() => {
                let ResourceDesc::Image(image) = r.desc else {
                    return Err(error(format!("pass `{name}` draws `{}`, which is not an image", r.name)));
                };
                let is_scene = r.name == ae_render_abi::module::SCENE_NAME || (image.format == env.scene_format && image.format != glow_format);
                if is_scene && image.format != env.scene_format {
                    return Err(error(format!("pass `{name}` draws into `{}` ({:?}); the library's pipelines target the scene's {:?}", r.name, image.format, env.scene_format)));
                } else if is_scene {
                    scene += 1;
                    if (image.width, image.height) != (env.extent.width, env.extent.height) {
                        return Err(error(format!("pass `{name}` draws into `{}` at {}x{}; the scene is {}x{}", r.name, image.width, image.height, env.extent.width, env.extent.height)));
                    }
                } else if image.format == glow_format && lib.attachments().is_some_and(|a| a.glow) {
                    glow_color += 1;
                    if (image.width, image.height) != (env.extent.width, env.extent.height) {
                        return Err(error(format!("pass `{name}` writes the glow `{}` at {}x{}; it is {}x{}", r.name, image.width, image.height, env.extent.width, env.extent.height)));
                    }
                } else {
                    return Err(error(format!("pass `{name}` draws into `{}` ({:?}); the library's pipelines target the scene's {:?} and its glow {:?}", r.name, image.format, env.scene_format, glow_format)));
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
    if let Some(a) = lib.attachments() {
        let need_glow = a.glow && glow;
        if scene != a.color || glow_color != usize::from(need_glow) || depth != 1 {
            let role = match a.depth {
                DepthUse::Write => "a depth attachment",
                DepthUse::Read => "a read-only depth attachment (depth_read)",
                DepthUse::Sampled => "the depth buffer sampled in the fragment stage (sampled:fragment)",
            };
            let glow_text = if need_glow { ", the glow as a second colour attachment," } else { "" };
            return Err(error(format!(
                "pass `{name}` ({}) needs {} colour attachment(s){glow_text} and {role}; it has {scene} scene, {glow_color} glow and {depth} depth",
                lib.name(),
                a.color
            )));
        }
    }
    if lib == LibraryPass::ReadbackCopy && p.intents.iter().filter(|i| i.role == Role::TransferSrc).count() != 1 {
        return Err(error(format!("pass `{name}` (readback.copy) copies one image")));
    }
    if lib == LibraryPass::Present && p.intents.len() != 1 {
        return Err(error(format!("pass `{name}` (present) presents one image")));
    }
    Ok(())
}

/// The graph's `glow` image, if it declares one (by name: any image may
/// use the name, but it must be of [`ae_render_abi::module::GLOW_FORMAT`]).
pub fn glow_resource<R>(graph: &FrameGraph<R>) -> Option<ResourceId> {
    graph.resources.iter().position(|r| r.name == ae_render_abi::module::GLOW_NAME).map(|i| ResourceId(i as u32))
}

/// `post.tonemap`: the scene `sampled:fragment` (and the bloom, a quarter
/// of its size, when the graph declares a glow) and one colour attachment
/// of the output's, nothing else. With the exposure image it samples it
/// third (`sampled:fragment`, the graph's 1×1 exposure spec).
fn check_tonemap<R>(graph: &FrameGraph<R>, pass: usize, env: &LibraryEnv) -> Result<(), LibraryError> {
    let p = &graph.passes[pass];
    let name = &p.name;
    let bloom_format = vk::Format::from_raw(ae_render_abi::module::BLOOM_FORMAT as i32);
    let exposure_format = vk::Format::from_raw(ae_render_abi::module::EXPOSURE_FORMAT as i32);
    let (bw, bh) = ae_render_abi::module::bloom_size(env.extent.width, env.extent.height);
    let (mut scene, mut bloom, mut exposure, mut color) = (0, 0, 0, 0);
    let mut sampled = 0;
    for intent in &p.intents {
        let r = &graph.resources[intent.resource.index()];
        let ResourceDesc::Image(image) = r.desc else {
            return Err(error(format!("pass `{name}` (post.tonemap) does not use buffer `{}`", r.name)));
        };
        match intent.role {
            Role::Sampled(ShaderStage::Fragment) if sampled == 0 => {
                if image.format != env.scene_format || (image.width, image.height) != (env.extent.width, env.extent.height) {
                    return Err(error(format!(
                        "pass `{name}` (post.tonemap) samples `{}` ({:?} {}x{}) first; the scene is {:?} {}x{}",
                        r.name, image.format, image.width, image.height, env.scene_format, env.extent.width, env.extent.height
                    )));
                }
                scene += 1;
                sampled += 1;
            }
            Role::Sampled(ShaderStage::Fragment) if image.format == exposure_format && (image.width, image.height) == (1, 1) => {
                exposure += 1;
                sampled += 1;
            }
            Role::Sampled(ShaderStage::Fragment) => {
                if image.format != bloom_format || (image.width, image.height) != (bw, bh) {
                    return Err(error(format!(
                        "pass `{name}` (post.tonemap) samples `{}` ({:?} {}x{}); the bloom is {:?} {bw}x{bh} or the exposure is {exposure_format:?} 1x1",
                        r.name, image.format, image.width, image.height, bloom_format
                    )));
                }
                bloom += 1;
                sampled += 1;
            }
            Role::ColorAttachment(_) if image.format == env.display_format() => color += 1,
            Role::ColorAttachment(_) => {
                return Err(error(format!(
                    "pass `{name}` (post.tonemap) writes `{}` ({:?}); it writes the {} ({:?})",
                    r.name,
                    image.format,
                    if env.encoding.uses_display() { "display" } else { "output" },
                    env.display_format()
                )))
            }
            role => return Err(error(format!("pass `{name}` (post.tonemap) does not use `{}` as {role:?}", r.name))),
        }
    }
    let need_bloom = glow_resource(graph).is_some();
    let need_exposure = graph.resources.iter().any(|r| r.name == ae_render_abi::module::EXPOSURE_NAME);
    if scene != 1 || color != 1 || bloom != usize::from(need_bloom) {
        return Err(error(format!(
            "pass `{name}` (post.tonemap) samples one scene image{}and writes one colour attachment; it has {scene}, {bloom}, {exposure} and {color}",
            if need_bloom { " and the bloom " } else { " " }
        )));
    }
    if need_exposure && exposure != 1 {
        return Err(error(format!(
            "pass `{name}` (post.tonemap) must sample `{}` (the graph declares it: the tonemap applies the eased EV)",
            ae_render_abi::module::EXPOSURE_NAME
        )));
    }
    Ok(())
}

/// `post.encode`: the display (sampled in the fragment stage, full size,
/// [`crate::encode::DISPLAY_FORMAT`]) and one colour attachment of the
/// target's, nothing else. Only an HDR graph has it.
fn check_encode<R>(graph: &FrameGraph<R>, pass: usize, env: &LibraryEnv) -> Result<(), LibraryError> {
    let p = &graph.passes[pass];
    let name = &p.name;
    if !env.encoding.uses_display() {
        return Err(error(format!("pass `{name}` (post.encode) does not exist in an SDR graph; post.tonemap writes the output")));
    }
    let (w, h) = (env.extent.width, env.extent.height);
    let (mut display, mut target) = (0, 0);
    for intent in &p.intents {
        let r = &graph.resources[intent.resource.index()];
        let ResourceDesc::Image(image) = r.desc else {
            return Err(error(format!("pass `{name}` (post.encode) does not use buffer `{}`", r.name)));
        };
        match intent.role {
            Role::Sampled(ShaderStage::Fragment) if image.format == crate::encode::DISPLAY_FORMAT && (image.width, image.height) == (w, h) => display += 1,
            Role::ColorAttachment(_) if image.format == env.color_format && (image.width, image.height) == (w, h) => target += 1,
            Role::Sampled(ShaderStage::Fragment) => {
                return Err(error(format!(
                    "pass `{name}` (post.encode) samples `{}` ({:?} {}x{}); the display is {:?} {w}x{h}",
                    r.name, image.format, image.width, image.height, crate::encode::DISPLAY_FORMAT
                )))
            }
            Role::ColorAttachment(_) => return Err(error(format!("pass `{name}` (post.encode) writes `{}` ({:?}); the output is {:?}", r.name, image.format, env.color_format))),
            role => return Err(error(format!("pass `{name}` (post.encode) does not use `{}` as {role:?}", r.name))),
        }
    }
    if display != 1 || target != 1 {
        return Err(error(format!("pass `{name}` (post.encode) samples one display image and writes one colour attachment; it has {display} and {target}")));
    }
    Ok(())
}

/// `exposure.meter`: the scene sampled in the compute stage, full size, and
/// optionally the depth (to undo the fog of war's dimming) and the glow
/// (the light the bloom shows), both full size, nothing else (its histogram
/// is in the library's requirements).
fn check_exposure_meter<R>(graph: &FrameGraph<R>, pass: usize, env: &LibraryEnv) -> Result<(), LibraryError> {
    let p = &graph.passes[pass];
    let name = &p.name;
    let histogram = LibraryResource::ExposureHistogram.name();
    let glow_format = vk::Format::from_raw(ae_render_abi::module::GLOW_FORMAT as i32);
    let (w, h) = (env.extent.width, env.extent.height);
    let (mut scene, mut depth, mut glow, mut other) = (0, 0, 0, 0);
    for intent in &p.intents {
        let r = &graph.resources[intent.resource.index()];
        if r.name == histogram && intent.role == Role::StorageBufferReadWrite(ShaderStage::Compute) {
            continue;
        }
        let ResourceDesc::Image(image) = r.desc else {
            other += 1;
            continue;
        };
        if intent.role != Role::Sampled(ShaderStage::Compute) {
            other += 1;
            continue;
        }
        match (image.format, image.width, image.height) {
            (f, iw, ih) if f == env.scene_format && (iw, ih) == (w, h) => scene += 1,
            (f, iw, ih) if f == env.depth_format && (iw, ih) == (w, h) => depth += 1,
            (f, iw, ih) if f == glow_format && (iw, ih) == (w, h) => glow += 1,
            _ => other += 1,
        }
    }
    if scene != 1 || depth > 1 || glow > 1 || other != 0 {
        return Err(error(format!(
            "pass `{name}` (exposure.meter) samples one scene image ({:?} {w}x{h}), at most one depth ({:?}) and at most one glow ({glow_format:?}), full size, in the compute stage, and adds to `{histogram}`; it has {scene} scene, {depth} depth, {glow} glow and {other} other intents",
            env.scene_format, env.depth_format
        )));
    }
    Ok(())
}

/// The bloom's passes: `bloom.downsample` reads the full-size glow (and the
/// exposure image, when the graph declares it) and writes the first
/// quarter-size bloom image; the two blurs read one and write the other.
/// One sampled input and one colour attachment, no depth.
fn check_bloom<R>(graph: &FrameGraph<R>, pass: usize, lib: LibraryPass, env: &LibraryEnv) -> Result<(), LibraryError> {
    let p = &graph.passes[pass];
    let name = &p.name;
    let glow_format = vk::Format::from_raw(ae_render_abi::module::GLOW_FORMAT as i32);
    let bloom_format = vk::Format::from_raw(ae_render_abi::module::BLOOM_FORMAT as i32);
    let exposure_format = vk::Format::from_raw(ae_render_abi::module::EXPOSURE_FORMAT as i32);
    let (bw, bh) = ae_render_abi::module::bloom_size(env.extent.width, env.extent.height);
    let (want_in, want_size) = if lib == LibraryPass::BloomDownsample { (glow_format, (env.extent.width, env.extent.height)) } else { (bloom_format, (bw, bh)) };
    let (mut input, mut output, mut exposure) = (0, 0, 0);
    for intent in &p.intents {
        let r = &graph.resources[intent.resource.index()];
        let ResourceDesc::Image(image) = r.desc else {
            return Err(error(format!("pass `{name}` ({}) does not use buffer `{}`", lib.name(), r.name)));
        };
        match intent.role {
            Role::Sampled(ShaderStage::Fragment) if image.format == want_in && (image.width, image.height) == want_size => input += 1,
            Role::Sampled(ShaderStage::Fragment) if lib == LibraryPass::BloomDownsample && image.format == exposure_format && (image.width, image.height) == (1, 1) => exposure += 1,
            Role::ColorAttachment(_) if image.format == bloom_format && (image.width, image.height) == (bw, bh) => output += 1,
            Role::Sampled(ShaderStage::Fragment) => {
                return Err(error(format!(
                    "pass `{name}` ({}) samples `{}` ({:?} {}x{}); it reads {:?} {}x{}",
                    lib.name(), r.name, image.format, image.width, image.height, want_in, want_size.0, want_size.1
                )));
            }
            Role::ColorAttachment(_) => {
                return Err(error(format!(
                    "pass `{name}` ({}) writes `{}` ({:?} {}x{}); it writes the bloom {:?} {bw}x{bh}",
                    lib.name(), r.name, image.format, image.width, image.height, bloom_format
                )));
            }
            role => return Err(error(format!("pass `{name}` ({}) does not use `{}` as {role:?}", lib.name(), r.name))),
        }
    }
    let need_exposure = lib == LibraryPass::BloomDownsample && graph.resources.iter().any(|r| r.name == ae_render_abi::module::EXPOSURE_NAME);
    if need_exposure && exposure != 1 {
        return Err(error(format!(
            "pass `{name}` (bloom.downsample) must sample `{}` (the graph declares it: the clamp follows the eased EV)",
            ae_render_abi::module::EXPOSURE_NAME
        )));
    }
    if input != 1 || output != 1 || exposure != usize::from(need_exposure) || p.intents.len() != 2 + usize::from(need_exposure) {
        return Err(error(format!(
            "pass `{name}` ({}) samples one image{} and writes one colour attachment; it has {input}, {exposure} and {output} ({} intents)",
            lib.name(),
            if need_exposure { " and the exposure" } else { "" },
            p.intents.len()
        )));
    }
    Ok(())
}

/// Graph-wide: a graph that draws the units (or their shadows) poses them
/// (`units.pose`, which alone writes their palettes); a graph that declares
/// the units' shadow map has a `shadows.draw` pass write it, and one that
/// declares the particles' coloured shadows declares the map too and has a
/// `shadows.particles` pass write them (`libs`: each pass's library pass).
/// An HDR output has a `post.encode` pass and an SDR one has none (the
/// tonemap writes the target in SDR and the display in HDR).
pub fn check_graph<R>(graph: &FrameGraph<R>, map: &ResourceMap, env: &LibraryEnv, libs: impl Fn(usize) -> Option<LibraryPass>) -> Result<(), LibraryError> {
    let written = |lib: LibraryPass| (0..graph.passes.len()).any(|i| libs(i) == Some(lib));
    if (written(LibraryPass::UnitsDraw) || written(LibraryPass::ShadowsDraw)) && !written(LibraryPass::UnitsPose) {
        return Err(error("the graph draws the units but no `units.pose` pass writes their palettes (`units.palette`)".into()));
    }
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
    let (list, clusters) = (map.contains(&Some(LibraryResource::LightList)), map.contains(&Some(LibraryResource::LightClusters)));
    if list != clusters {
        return Err(error("the graph declares one of `lights.list` and `lights.clusters`; the lights need both".into()));
    }
    if clusters && !written(LibraryPass::LightsCluster) {
        return Err(error("the graph declares `lights.clusters` but no `lights.cluster` pass writes it".into()));
    }
    // The pass appends the units' model lights from the palettes
    // `units.pose` writes: it must run after it.
    if clusters && written(LibraryPass::UnitsPose) {
        let first = |lib: LibraryPass| (0..graph.passes.len()).find(|&i| libs(i) == Some(lib));
        if let (Some(cluster), Some(pose)) = (first(LibraryPass::LightsCluster), first(LibraryPass::UnitsPose))
            && cluster < pose
        {
            return Err(error("the graph clusters the lights before `units.pose`; the pass appends the units' model lights from the palettes it writes".into()));
        }
    }
    // The glow (the bloom's input) and the bloom's three passes go
    // together: a graph either declares the glow image and all of them, or
    // neither.
    if let Some(glow) = glow_resource(graph) {
        let r = &graph.resources[glow.index()];
        let want = vk::Format::from_raw(ae_render_abi::module::GLOW_FORMAT as i32);
        match r.desc {
            ResourceDesc::Image(image) if image.format == want => {}
            ResourceDesc::Image(image) => return Err(error(format!("`{}` is {:?}; the glow is {want:?} (glow_spec)", r.name, image.format))),
            ResourceDesc::Buffer(_) => return Err(error(format!("`{}` is a buffer; the glow is an image", r.name))),
        }
        let missing: Vec<&str> = [LibraryPass::BloomDownsample, LibraryPass::BloomBlurX, LibraryPass::BloomBlurY]
            .into_iter()
            .filter(|lib| !written(*lib))
            .map(LibraryPass::name)
            .collect();
        if !missing.is_empty() {
            return Err(error(format!("the graph declares `{}` but no `{}` pass (the bloom reads it)", r.name, missing.join("` or `"))));
        }
    } else if written(LibraryPass::BloomDownsample) || written(LibraryPass::BloomBlurX) || written(LibraryPass::BloomBlurY) {
        return Err(error("the graph has `post.bloom` passes but no `glow` image (bloom reads the light sources' glow)".into()));
    }
    // Automatic exposure: the image and the histogram go with the three
    // passes, all or none.
    let (exposure, histogram) = (map.contains(&Some(LibraryResource::Exposure)), map.contains(&Some(LibraryResource::ExposureHistogram)));
    if exposure != histogram {
        return Err(error(format!(
            "the graph declares one of `{}` and `{}`; auto-exposure needs both",
            LibraryResource::Exposure.name(),
            LibraryResource::ExposureHistogram.name()
        )));
    }
    let exposure_passes = [LibraryPass::ExposureClear, LibraryPass::ExposureMeter, LibraryPass::ExposureAdapt];
    let missing: Vec<&str> = exposure_passes.into_iter().filter(|lib| !written(*lib)).map(LibraryPass::name).collect();
    if exposure && !missing.is_empty() {
        return Err(error(format!(
            "the graph declares `{}` but no `{}` pass",
            LibraryResource::Exposure.name(),
            missing.join("`, `")
        )));
    }
    if exposure {
        // Write-write on the histogram orders nothing: a clear declared
        // after the meter would wipe the frame's counts.
        let first = |lib: LibraryPass| (0..graph.passes.len()).find(|&i| libs(i) == Some(lib));
        if let (Some(clear), Some(meter), Some(adapt)) = (first(LibraryPass::ExposureClear), first(LibraryPass::ExposureMeter), first(LibraryPass::ExposureAdapt))
            && !(clear < meter && meter < adapt)
        {
            return Err(error("the graph's auto-exposure passes must run `exposure.clear`, then `exposure.meter`, then `exposure.adapt`".into()));
        }
    }
    if !exposure && exposure_passes.into_iter().any(written) {
        return Err(error(format!(
            "the graph has auto-exposure passes but no `{}` image (the eased EV has nowhere to live)",
            LibraryResource::Exposure.name()
        )));
    }
    // The display intermediate and `post.encode` go with an HDR output: the
    // tonemap writes the display there, so only the encode writes the target.
    if env.encoding.uses_display() != written(LibraryPass::PostEncode) {
        return Err(error(if env.encoding.uses_display() {
            "the output is HDR but the graph has no `post.encode` pass (the tonemap writes the display, not the target)".into()
        } else {
            "the graph has `post.encode` but the output is SDR (the tonemap writes the target)".into()
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::app::{TerrainOutput, native_graph};
    use ae_framegraph::{ImageDesc, LoadOp, PassKind};

    fn env() -> LibraryEnv {
        LibraryEnv { color_format: vk::Format::R8G8B8A8_SRGB, scene_format: vk::Format::R16G16B16A16_SFLOAT, depth_format: vk::Format::D32_SFLOAT, extent: vk::Extent2D { width: 64, height: 32 }, units: true, water: true, shadows: true, particles: true, encoding: crate::encode::OutputEncoding::Sdr, levels: crate::encode::DisplayLevels::default() }
    }

    fn size(r: LibraryResource) -> Option<BufferDesc> {
        Some(BufferDesc { size: 64 + r as u64, usage: vk::BufferUsageFlags::STORAGE_BUFFER })
    }

    /// Resolve and check every pass, then the graph-wide rules; the first
    /// error.
    fn wire(mut graph: FrameGraph<LibraryPass>, env: &LibraryEnv) -> Result<FrameGraph<LibraryPass>, LibraryError> {
        let map = resolve(&mut graph, env, size)?;
        for (i, p) in graph.passes.iter().enumerate() {
            check(&graph, i, p.record, &map, env)?;
        }
        check_graph(&graph, &map, env, |i| Some(graph.passes[i].record))?;
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
            for lib in [LibraryPass::ExposureClear, LibraryPass::ExposureMeter, LibraryPass::ExposureAdapt] {
                assert!(graph.passes.iter().any(|p| p.record == lib), "the scene meters and adapts the exposure");
            }
            assert!(graph.resources.iter().any(|r| r.name == ae_render_abi::module::EXPOSURE_NAME && r.desc.is_image()));
            let histogram = graph.resources.iter().find(|r| r.name == ae_render_abi::module::EXPOSURE_HISTOGRAM_NAME).unwrap();
            assert_eq!(histogram.desc, ae_framegraph::ResourceDesc::Buffer(size(LibraryResource::ExposureHistogram).unwrap()));
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
    fn an_hdr_graph_adds_the_display_and_the_encode() {
        use crate::encode::{DISPLAY_FORMAT, DISPLAY_NAME, DisplayLevels, OutputEncoding};
        let hdr = LibraryEnv { color_format: vk::Format::A2B10G10R10_UNORM_PACK32, encoding: OutputEncoding::Hdr10Pq, levels: DisplayLevels::default(), ..env() };
        let graph = wire(native_graph(&hdr, TerrainOutput::Readback), &hdr).unwrap();
        // The tonemap writes the display (paper-white units), and the encode
        // reads it and writes the 10-bit target.
        let display = graph.resources.iter().find(|r| r.name == DISPLAY_NAME).expect("the display exists");
        assert_eq!(display.desc, ae_framegraph::ResourceDesc::Image(ImageDesc::color_2d(DISPLAY_FORMAT, 64, 32, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)));
        let tonemap = &graph.passes[pass(&graph, LibraryPass::Tonemap)];
        let colour = tonemap.intents.iter().find(|i| matches!(i.role, Role::ColorAttachment(_))).unwrap();
        assert_eq!(graph.resources[colour.resource.index()].name, DISPLAY_NAME);
        let encode = &graph.passes[pass(&graph, LibraryPass::PostEncode)];
        assert!(encode.intents.iter().any(|i| i.role == Role::Sampled(ShaderStage::Fragment) && graph.resources[i.resource.index()].name == DISPLAY_NAME));
        assert!(encode.intents.iter().any(|i| matches!(i.role, Role::ColorAttachment(_)) && graph.resources[i.resource.index()].name == LibraryResource::ReadbackTarget.name()));
        // An SDR graph has neither: the tonemap writes the target.
        let sdr = wire(native_graph(&env(), TerrainOutput::Readback), &env()).unwrap();
        assert!(!sdr.passes.iter().any(|p| p.record == LibraryPass::PostEncode));
        assert!(!sdr.resources.iter().any(|r| r.name == DISPLAY_NAME));
    }

    #[test]
    fn the_encode_needs_an_hdr_graph_and_the_display() {
        use crate::encode::{DISPLAY_FORMAT, DISPLAY_NAME};
        let sdr = env();
        let mut g = FrameGraph::<LibraryPass>::new();
        let display = g.image(DISPLAY_NAME, ImageDesc::color_2d(DISPLAY_FORMAT, 64, 32, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED), ResourceClass::Transient);
        let target = g.image("target", ImageDesc::color_2d(sdr.color_format, 64, 32, vk::ImageUsageFlags::COLOR_ATTACHMENT), ResourceClass::Transient);
        g.pass("encode", PassKind::Graphics, LibraryPass::PostEncode).sampled(display, ShaderStage::Fragment).color(target, LoadOp::DontCare).id();
        // An SDR output has no display and no encode.
        let message = wire(g, &sdr).unwrap_err().0;
        assert!(message.contains("does not exist in an SDR graph"), "{message}");
        // In HDR the tonemap must write the display, not the target.
        let hdr = LibraryEnv { color_format: vk::Format::A2B10G10R10_UNORM_PACK32, encoding: crate::encode::OutputEncoding::Hdr10Pq, ..env() };
        let mut g = FrameGraph::<LibraryPass>::new();
        let scene = g.image(ae_render_abi::module::SCENE_NAME, ImageDesc::color_2d(hdr.scene_format, 64, 32, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED), ResourceClass::Transient);
        let out = g.image("target", ImageDesc::color_2d(hdr.color_format, 64, 32, vk::ImageUsageFlags::COLOR_ATTACHMENT), ResourceClass::Transient);
        g.pass("tonemap", PassKind::Graphics, LibraryPass::Tonemap).sampled(scene, ShaderStage::Fragment).color(out, LoadOp::DontCare).id();
        let message = wire(g, &hdr).unwrap_err().0;
        assert!(message.contains("it writes the display"), "{message}");
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
        let other = LibraryEnv { scene_format: vk::Format::B10G11R11_UFLOAT_PACK32, ..env() };
        let e = err(native_graph(&other, TerrainOutput::Present), &env());
        assert!(e.contains("draws into `scene`"), "{e}");
        let other = LibraryEnv { color_format: vk::Format::B8G8R8A8_SRGB, ..env() };
        let e = err(native_graph(&other, TerrainOutput::Present), &env());
        assert!(e.contains("(post.tonemap) writes `target`"), "{e}");
        // The tonemap sampling no scene.
        let mut g = base();
        let t = pass(&g, LibraryPass::Tonemap);
        g.passes[t].intents.retain(|i| !matches!(i.role, Role::Sampled(_)));
        let e = err(g, &env());
        assert!(e.contains("samples one scene image"), "{e}");
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
            check_graph(&g, &map, &env(), |i| Some(g.passes[i].record)).map_err(|e| e.0)
        };
        let g = base();
        assert_eq!(shadows(&g), Ok(()));
        let mut g = base();
        let (cluster, pose) = (pass(&g, LibraryPass::LightsCluster), pass(&g, LibraryPass::UnitsPose));
        g.passes.swap(cluster, pose);
        let e = shadows(&g).unwrap_err();
        assert!(e.contains("clusters the lights before `units.pose`"), "{e}");
        let mut g = base();
        g.passes.retain(|p| p.record != LibraryPass::ShadowsParticles);
        let e = shadows(&g).unwrap_err();
        assert!(e.contains("no `shadows.particles` pass"), "{e}");
        let mut g = base();
        let i = g.resources.iter().position(|r| r.name == "shadows.map").unwrap();
        g.resources[i].name = "not the map".into();
        let e = shadows(&g).unwrap_err();
        assert!(e.contains("without `shadows.map`"), "{e}");
        // Units drawn but never posed.
        let mut g = base();
        g.passes.retain(|p| p.record != LibraryPass::UnitsPose);
        let e = shadows(&g).unwrap_err();
        assert!(e.contains("no `units.pose` pass"), "{e}");

        // Automatic exposure: the three passes all run in order, the image
        // and the histogram go together, and the tonemap and the bloom read
        // the eased EV.
        let mut g = base();
        g.passes.retain(|p| p.record != LibraryPass::ExposureAdapt);
        let e = err(g, &env());
        assert!(e.contains("declares `exposure`") && e.contains("exposure.adapt"), "{e}");
        let mut g = base();
        let (clear, meter) = (pass(&g, LibraryPass::ExposureClear), pass(&g, LibraryPass::ExposureMeter));
        g.passes.swap(clear, meter);
        let e = err(g, &env());
        assert!(e.contains("must run `exposure.clear`, then"), "{e}");
        let mut g = base();
        let t = pass(&g, LibraryPass::Tonemap);
        let exposure = g.resources.iter().position(|r| r.name == ae_render_abi::module::EXPOSURE_NAME).unwrap();
        g.passes[t].intents.retain(|i| i.resource.index() != exposure);
        let e = err(g, &env());
        assert!(e.contains("must sample `exposure`"), "{e}");
        let mut g = base();
        let d = pass(&g, LibraryPass::BloomDownsample);
        g.passes[d].intents.retain(|i| i.resource.index() != exposure);
        let e = err(g, &env());
        assert!(e.contains("bloom.downsample") && e.contains("`exposure`"), "{e}");
        // The meter samples the scene, not an arbitrary image.
        let mut g = base();
        let m = pass(&g, LibraryPass::ExposureMeter);
        let scene = g.passes[m].intents.iter().position(|i| i.role == Role::Sampled(ShaderStage::Compute)).unwrap();
        let blooms = g.resources.iter().position(|r| r.name == "bloom.0").unwrap();
        g.passes[m].intents[scene].resource = ResourceId(blooms as u32);
        let e = err(g, &env());
        assert!(e.contains("exposure.meter") && e.contains("one scene image"), "{e}");
    }
}
