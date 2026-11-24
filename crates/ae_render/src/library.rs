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

use ae_framegraph::{
    BufferDesc, FrameGraph, ImageDesc, PassKind, ResourceClass, ResourceDesc, ResourceId, Role, ShaderStage,
};
use ash::vk;
use std::fmt;

mod water;
pub use water::WaterTiers;

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
    /// The shore travel-time field (M16.5): R16F, one texel per ground
    /// height sample, `water.shore` writes it after a ground change. Zero at
    /// the waterline, rising seaward; the rich tiers time their shore foam
    /// with it.
    WaterShore,
    /// The tileable caustic web (M16.5): RGBA8 with mips, made once when the
    /// water renderer is built; the refract tier's underwater pass projects
    /// it.
    WaterCaustics,
    TerrainLeavesReflect,
    TerrainIndirectReflect,
    UnitVisibleReflect,
    UnitIndirectReflect,
    WaterReflection,
    WaterReflectionDepth,
    WaterRefraction,
    WaterParticles,
    WaterRaw,
    WaterChurn,
    WaterWaves,
    WaterMeshLeaves,
    WaterMeshIndirect,
    WaterVolumeOut,
}

impl LibraryResource {
    pub const ALL: [LibraryResource; 34] = [
        Self::TerrainLeaves,
        Self::TerrainIndirect,
        Self::TerrainLeavesReflect,
        Self::TerrainIndirectReflect,
        Self::UnitVisibleReflect,
        Self::UnitIndirectReflect,
        Self::WaterReflection,
        Self::WaterReflectionDepth,
        Self::WaterRefraction,
        Self::WaterVolumeOut,
        Self::WaterParticles,
        Self::WaterRaw,
        Self::WaterChurn,
        Self::WaterWaves,
        Self::WaterMeshLeaves,
        Self::WaterMeshIndirect,
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
        Self::WaterShore,
        Self::WaterCaustics,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::TerrainLeaves => "terrain.leaves",
            Self::TerrainIndirect => "terrain.indirect",
            Self::TerrainLeavesReflect => ae_render_abi::module::TERRAIN_LEAVES_REFLECT_NAME,
            Self::TerrainIndirectReflect => ae_render_abi::module::TERRAIN_INDIRECT_REFLECT_NAME,
            Self::UnitVisibleReflect => ae_render_abi::module::UNITS_VISIBLE_REFLECT_NAME,
            Self::UnitIndirectReflect => ae_render_abi::module::UNITS_INDIRECT_REFLECT_NAME,
            Self::WaterReflection => ae_render_abi::module::WATER_REFLECTION_NAME,
            Self::WaterReflectionDepth => ae_render_abi::module::WATER_REFLECTION_DEPTH_NAME,
            Self::WaterRefraction => ae_render_abi::module::WATER_REFRACTION_NAME,
            Self::WaterVolumeOut => ae_render_abi::module::WATER_VOLUME_OUT_NAME,
            Self::WaterParticles => ae_render_abi::module::WATER_PARTICLES_NAME,
            Self::WaterRaw => ae_render_abi::module::WATER_RAW_NAME,
            Self::WaterChurn => ae_render_abi::module::WATER_CHURN_NAME,
            Self::WaterWaves => ae_render_abi::module::WATER_WAVES_NAME,
            Self::WaterMeshLeaves => ae_render_abi::module::WATER_MESH_LEAVES_NAME,
            Self::WaterMeshIndirect => ae_render_abi::module::WATER_MESH_INDIRECT_NAME,
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
            Self::WaterShore => "water.shore",
            Self::WaterCaustics => "water.caustics",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|r| r.name() == name)
    }

    pub fn is_transient(self) -> bool {
        matches!(
            self,
            Self::TerrainLeavesReflect
                | Self::TerrainIndirectReflect
                | Self::UnitVisibleReflect
                | Self::UnitIndirectReflect
                | Self::WaterReflection
                | Self::WaterReflectionDepth
                | Self::WaterRefraction
                | Self::WaterVolumeOut
                | Self::WaterParticles
                | Self::WaterRaw
                | Self::WaterChurn
                | Self::WaterWaves
                | Self::WaterMeshLeaves
                | Self::WaterMeshIndirect
        )
    }

    pub fn is_image(self) -> bool {
        matches!(
            self,
            Self::ReadbackTarget
                | Self::ShadowMap
                | Self::ParticleShadows
                | Self::Exposure
                | Self::WaterShore
                | Self::WaterCaustics
                | Self::WaterReflection
                | Self::WaterReflectionDepth
                | Self::WaterRefraction
                | Self::WaterVolumeOut
                | Self::WaterRaw
                | Self::WaterChurn
                | Self::WaterWaves
        )
    }
}

/// Passes a graph can wire up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryPass {
    TerrainLod,
    TerrainDraw,
    TerrainCullReflect,
    TerrainReflect,
    UnitsResetReflect,
    UnitsCullReflect,
    UnitsReflect,
    UnitsReset,
    UnitsCull,
    /// The units' poses: each animated unit's skinning palette, once per
    /// bone per frame, into [`LibraryResource::UnitPalette`] (compute).
    UnitsPose,
    UnitsDraw,
    UnitsWaterOverlay,
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
    WaterUnderwater,
    WaterVolume,
    WaterEmit,
    WaterSplat,
    WaterChurn,
    WaterBlur,
    WaterLod,
    /// The shore travel-time field (M16.5): a jump flood over the ground
    /// height texture into [`LibraryResource::WaterShore`], recorded only
    /// after a ground change (the host skips it otherwise).
    WaterShore,
    /// Spawn, move and sort the particles (compute).
    ParticlesSim,
    /// The particles, back to front, blended over the scene; samples the
    /// depth buffer.
    ParticlesDraw,
    ParticlesUnderwater,
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
    pub const ALL: [LibraryPass; 39] = [
        Self::TerrainLod,
        Self::TerrainDraw,
        Self::TerrainCullReflect,
        Self::TerrainReflect,
        Self::UnitsResetReflect,
        Self::UnitsCullReflect,
        Self::UnitsReflect,
        Self::UnitsReset,
        Self::UnitsCull,
        Self::UnitsPose,
        Self::UnitsDraw,
        Self::UnitsWaterOverlay,
        Self::UnitsCount,
        Self::ShadowsDraw,
        Self::ShadowsParticles,
        Self::WaterDraw,
        Self::WaterUnderwater,
        Self::WaterVolume,
        Self::WaterEmit,
        Self::WaterSplat,
        Self::WaterChurn,
        Self::WaterBlur,
        Self::WaterLod,
        Self::WaterShore,
        Self::ParticlesSim,
        Self::ParticlesDraw,
        Self::ParticlesUnderwater,
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
            Self::TerrainCullReflect => "terrain.cull_reflect",
            Self::TerrainReflect => "terrain.reflect",
            Self::UnitsResetReflect => "units.reset_reflect",
            Self::UnitsCullReflect => "units.cull_reflect",
            Self::UnitsReflect => "units.reflect",
            Self::UnitsReset => "units.reset",
            Self::UnitsCull => "units.cull",
            Self::UnitsPose => "units.pose",
            Self::UnitsDraw => "units.draw",
            Self::UnitsWaterOverlay => "units.water_overlay",
            Self::UnitsCount => "units.count",
            Self::ShadowsDraw => "shadows.draw",
            Self::ShadowsParticles => "shadows.particles",
            Self::WaterDraw => "water.draw",
            Self::WaterUnderwater => "water.underwater",
            Self::WaterVolume => "water.volume",
            Self::WaterEmit => "water.emit",
            Self::WaterSplat => "water.splat",
            Self::WaterChurn => "water.churn",
            Self::WaterBlur => "water.blur",
            Self::WaterLod => "water.lod",
            Self::WaterShore => "water.shore",
            Self::ParticlesSim => "particles.sim",
            Self::ParticlesDraw => "particles.draw",
            Self::ParticlesUnderwater => "particles.underwater",
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
            Self::TerrainLod
            | Self::TerrainCullReflect
            | Self::UnitsCullReflect
            | Self::UnitsCull
            | Self::UnitsPose
            | Self::WaterShore
            | Self::WaterEmit
            | Self::WaterBlur
            | Self::WaterLod
            | Self::ParticlesSim
            | Self::LightsCluster
            | Self::ExposureMeter
            | Self::ExposureAdapt => PassKind::Compute,
            Self::TerrainDraw
            | Self::TerrainReflect
            | Self::UnitsReflect
            | Self::UnitsDraw
            | Self::UnitsWaterOverlay
            | Self::ShadowsDraw
            | Self::ShadowsParticles
            | Self::WaterDraw
            | Self::WaterUnderwater
            | Self::WaterVolume
            | Self::WaterSplat
            | Self::WaterChurn
            | Self::ParticlesDraw
            | Self::ParticlesUnderwater
            | Self::Tonemap
            | Self::PostEncode
            | Self::BloomDownsample
            | Self::BloomBlurX
            | Self::BloomBlurY => PassKind::Graphics,
            Self::UnitsReset
            | Self::UnitsResetReflect
            | Self::UnitsCount
            | Self::ReadbackCopy
            | Self::HostRead
            | Self::ExposureClear => PassKind::Transfer,
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
            Self::TerrainLod => &[
                (R::TerrainLeaves, Role::StorageBufferWrite(ShaderStage::Compute)),
                (R::TerrainIndirect, Role::StorageBufferWrite(ShaderStage::Compute)),
            ],
            Self::TerrainDraw => &[
                (R::TerrainLeaves, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::TerrainIndirect, Role::IndirectBuffer),
            ],
            Self::TerrainCullReflect => &[
                (R::TerrainIndirect, Role::StorageBufferRead(ShaderStage::Compute)),
                (R::TerrainLeavesReflect, Role::StorageBufferWrite(ShaderStage::Compute)),
                (
                    R::TerrainIndirectReflect,
                    Role::StorageBufferWrite(ShaderStage::Compute),
                ),
            ],
            Self::TerrainReflect => &[
                (R::TerrainLeavesReflect, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::TerrainIndirectReflect, Role::IndirectBuffer),
                (R::WaterReflection, Role::ColorAttachment(ae_framegraph::LoadOp::Clear)),
                (
                    R::WaterReflectionDepth,
                    Role::DepthAttachment(ae_framegraph::LoadOp::Clear),
                ),
            ],
            Self::UnitsResetReflect => &[(R::UnitIndirectReflect, Role::TransferDst)],
            Self::UnitsCullReflect => &[
                (R::UnitInstances, Role::StorageBufferRead(ShaderStage::Compute)),
                (R::UnitVisibleReflect, Role::StorageBufferWrite(ShaderStage::Compute)),
                (R::UnitIndirectReflect, Role::StorageBufferWrite(ShaderStage::Compute)),
            ],
            Self::UnitsReflect => &[
                (R::UnitInstances, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::UnitVisibleReflect, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::UnitIndirectReflect, Role::IndirectBuffer),
                (R::UnitPalette, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::WaterReflection, Role::ColorAttachment(ae_framegraph::LoadOp::Load)),
                (
                    R::WaterReflectionDepth,
                    Role::DepthAttachment(ae_framegraph::LoadOp::Load),
                ),
            ],
            Self::UnitsReset => &[(R::UnitIndirect, Role::TransferDst)],
            Self::UnitsCull => &[
                (R::UnitInstances, Role::StorageBufferRead(ShaderStage::Compute)),
                (R::UnitVisible, Role::StorageBufferWrite(ShaderStage::Compute)),
                (R::UnitIndirect, Role::StorageBufferWrite(ShaderStage::Compute)),
            ],
            Self::UnitsPose => &[
                (R::UnitInstances, Role::StorageBufferRead(ShaderStage::Compute)),
                (R::UnitPalette, Role::StorageBufferWrite(ShaderStage::Compute)),
            ],
            Self::UnitsDraw | Self::UnitsWaterOverlay => &[
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
            Self::ParticlesDraw | Self::ParticlesUnderwater => &[
                (R::ParticleState, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::ParticleKeys, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::ParticleIndirect, Role::IndirectBuffer),
            ],
            Self::LightsCluster => &[
                (R::LightList, Role::StorageBufferWrite(ShaderStage::Compute)),
                (R::LightClusters, Role::StorageBufferWrite(ShaderStage::Compute)),
            ],
            Self::WaterUnderwater => &[(R::WaterCaustics, Role::Sampled(ShaderStage::Fragment))],
            Self::WaterVolume => &[(R::WaterRefraction, Role::Sampled(ShaderStage::Fragment))],
            Self::WaterEmit => &[(R::WaterParticles, Role::StorageBufferWrite(ShaderStage::Compute))],
            Self::WaterSplat => &[
                (R::WaterParticles, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::WaterRaw, Role::ColorAttachment(ae_framegraph::LoadOp::Clear)),
            ],
            Self::WaterChurn => &[
                (R::WaterParticles, Role::StorageBufferRead(ShaderStage::Vertex)),
                (R::WaterChurn, Role::ColorAttachment(ae_framegraph::LoadOp::Clear)),
            ],
            Self::WaterBlur => &[
                (R::WaterRaw, Role::Sampled(ShaderStage::Compute)),
                (R::WaterChurn, Role::Sampled(ShaderStage::Compute)),
                (R::WaterWaves, Role::StorageImageWrite(ShaderStage::Compute)),
            ],
            Self::WaterLod => &[
                (R::WaterWaves, Role::Sampled(ShaderStage::Compute)),
                (R::WaterMeshLeaves, Role::StorageBufferWrite(ShaderStage::Compute)),
                (R::WaterMeshIndirect, Role::StorageBufferWrite(ShaderStage::Compute)),
            ],
            Self::WaterShore => &[(R::WaterShore, Role::StorageImageWrite(ShaderStage::Compute))],
            Self::ReadbackCopy => &[(R::ReadbackBuffer, Role::TransferDst)],
            Self::WaterDraw
            | Self::Tonemap
            | Self::PostEncode
            | Self::BloomDownsample
            | Self::BloomBlurX
            | Self::BloomBlurY
            | Self::HostRead
            | Self::Present => &[],
        }
    }

    /// Attachments a graphics pass needs (one colour target and the depth
    /// buffer, whose formats the host checks).
    pub fn attachments(self) -> Option<Attachments> {
        match self {
            Self::TerrainDraw | Self::UnitsDraw | Self::UnitsWaterOverlay => Some(Attachments {
                color: 1,
                glow: true,
                depth: DepthUse::Write,
            }),
            Self::WaterDraw => Some(Attachments {
                color: 1,
                glow: true,
                depth: DepthUse::Read,
            }),
            Self::ParticlesDraw | Self::ParticlesUnderwater | Self::WaterUnderwater | Self::WaterVolume => {
                Some(Attachments {
                    color: 1,
                    glow: self == Self::ParticlesDraw,
                    depth: DepthUse::Sampled,
                })
            }
            _ => None,
        }
    }

    /// Whether it needs the unit renderer.
    pub fn uses_units(self) -> bool {
        matches!(
            self,
            Self::UnitsReset
                | Self::UnitsResetReflect
                | Self::UnitsCullReflect
                | Self::UnitsReflect
                | Self::UnitsCull
                | Self::UnitsPose
                | Self::UnitsDraw
                | Self::UnitsWaterOverlay
                | Self::UnitsCount
                | Self::ShadowsDraw
        )
    }

    /// Whether it needs the particle renderer.
    pub fn uses_particles(self) -> bool {
        matches!(
            self,
            Self::ParticlesSim | Self::ParticlesDraw | Self::ParticlesUnderwater | Self::ShadowsParticles
        )
    }

    /// Whether it shades by the shadow maps: in a graph that declares
    /// [`LibraryResource::ShadowMap`] (and [`LibraryResource::ParticleShadows`])
    /// it must sample each in this role.
    pub fn receives(self) -> Option<Role> {
        matches!(
            self,
            Self::TerrainDraw
                | Self::UnitsDraw
                | Self::UnitsWaterOverlay
                | Self::TerrainReflect
                | Self::UnitsReflect
                | Self::WaterDraw
                | Self::WaterUnderwater
        )
        .then_some(Role::Sampled(ShaderStage::Fragment))
    }

    /// Whether it is lit by the clustered lights: in a graph that declares
    /// [`LibraryResource::LightList`] and [`LibraryResource::LightClusters`]
    /// it must read both in this role.
    pub fn lit(self) -> Option<Role> {
        match self {
            Self::TerrainDraw | Self::UnitsDraw | Self::UnitsWaterOverlay | Self::WaterDraw => {
                Some(Role::StorageBufferRead(ShaderStage::Fragment))
            }
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
    /// The ground height texture's size, when the map has water (the shore
    /// field's size; [`LibraryResource::WaterShore`] and
    /// [`LibraryResource::WaterCaustics`] exist only then).
    pub ground: Option<(u32, u32)>,
    /// The engine's own graph declares the water data: the `water.shore`
    /// bake pass and the two library images (M16.5). Off for the M7 graph,
    /// so an ordinary scene allocates no water data.
    pub bake_water: bool,
    /// How the output image encodes the display's light (SDR has no display
    /// intermediate; the tonemap writes the target, as before).
    pub encoding: crate::encode::OutputEncoding,
    /// The display's paper white and peak, in nits (unused in SDR).
    pub levels: crate::encode::DisplayLevels,
    /// The scene's working space ([`crate::colorspace::WorkingSpace`]): what
    /// the scene's colours and the meter's luminance are in.
    pub working: crate::colorspace::WorkingSpace,
    /// The output gamut the display intermediate is mapped to (unused in
    /// SDR, whose tonemap maps to BT.709 itself).
    pub gamut: crate::colorspace::Gamut,
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
/// resource to its library resource. Reflection targets and lists are
/// `Transient`. Host bindings are `External`. Declare a buffer with size 0
/// (or exactly its size), for
/// [`LibraryResource::ReadbackTarget`] as an image of the target's extent
/// and colour format, and for a host-filled image as a shape the bound image
/// supports: a module's explicit descriptor may ask for less usage than the
/// allocation has, never more (see [`check_bound_image`]).
pub fn resolve<R>(
    graph: &mut FrameGraph<R>,
    env: &LibraryEnv,
    desc: impl Fn(LibraryResource) -> Option<BufferDesc>,
) -> Result<ResourceMap, LibraryError> {
    let mut map = Vec::with_capacity(graph.resources.len());
    let mut seen = std::collections::HashSet::new();
    for resource in &mut graph.resources {
        let Some(lib) = LibraryResource::from_name(&resource.name) else {
            map.push(None);
            continue;
        };
        // A library resource is one host object, but the graph tracks
        // producers and readers by resource id: a second declaration of the
        // same name would bind the same image under another id, so a writer
        // for the first is not a dependency of a reader of the second (and
        // the unrooted writer is pruned). The JSON loader refuses duplicate
        // names; the module path must too.
        if !seen.insert(lib) {
            return Err(error(format!(
                "library resource `{}` is declared more than once; the host binds one object and the graph's producers are tracked by resource id",
                resource.name
            )));
        }
        let class = if lib.is_transient() {
            ResourceClass::Transient
        } else {
            ResourceClass::External
        };
        if resource.class != class {
            return Err(error(format!(
                "library resource `{}` must be {class:?}, not {:?}",
                resource.name, resource.class
            )));
        }
        match (&mut resource.desc, lib.is_image()) {
            (ResourceDesc::Image(image), true) if lib == LibraryResource::ShadowMap => {
                use ae_render_abi::module as m;
                if !env.shadows {
                    return Err(error(format!(
                        "library resource `{}` does not exist in this scene (its units cast no shadows)",
                        resource.name
                    )));
                }
                let format = vk::Format::from_raw(m::SHADOW_MAP_FORMAT as i32);
                if image.format != format
                    || image.width != m::SHADOW_MAP_WIDTH
                    || image.height != m::SHADOW_MAP_HEIGHT
                    || !image.aspect.contains(vk::ImageAspectFlags::DEPTH)
                {
                    return Err(error(format!(
                        "`{}` is {:?} {}x{}; the library's is {format:?} {}x{} with a depth aspect (shadow_map_spec)",
                        resource.name,
                        image.format,
                        image.width,
                        image.height,
                        m::SHADOW_MAP_WIDTH,
                        m::SHADOW_MAP_HEIGHT
                    )));
                }
            }
            (ResourceDesc::Image(image), true) if lib == LibraryResource::ParticleShadows => {
                use ae_render_abi::module as m;
                if !(env.shadows && env.particles) {
                    return Err(error(format!(
                        "library resource `{}` does not exist in this scene (it needs units that cast shadows and particles)",
                        resource.name
                    )));
                }
                let format = vk::Format::from_raw(m::PARTICLE_SHADOW_FORMAT as i32);
                if image.format != format
                    || image.width != m::PARTICLE_SHADOW_WIDTH
                    || image.height != m::PARTICLE_SHADOW_HEIGHT
                    || !image.aspect.contains(vk::ImageAspectFlags::COLOR)
                {
                    return Err(error(format!(
                        "`{}` is {:?} {}x{}; the library's is {format:?} {}x{} with a colour aspect (particle_shadow_spec)",
                        resource.name,
                        image.format,
                        image.width,
                        image.height,
                        m::PARTICLE_SHADOW_WIDTH,
                        m::PARTICLE_SHADOW_HEIGHT
                    )));
                }
            }
            (ResourceDesc::Image(image), true) if lib == LibraryResource::Exposure => {
                use ae_render_abi::module as m;
                let format = vk::Format::from_raw(m::EXPOSURE_FORMAT as i32);
                if image.format != format
                    || image.width != 1
                    || image.height != 1
                    || !image.aspect.contains(vk::ImageAspectFlags::COLOR)
                {
                    return Err(error(format!(
                        "`{}` is {:?} {}x{}; the exposure image is {format:?} 1x1 with a colour aspect (exposure_spec)",
                        resource.name, image.format, image.width, image.height
                    )));
                }
            }
            (ResourceDesc::Image(image), true)
                if matches!(lib, LibraryResource::WaterRefraction | LibraryResource::WaterVolumeOut) =>
            {
                let want = image_desc(lib, env)?;
                if (image.format, image.width, image.height) != (want.format, want.width, want.height) {
                    return Err(error(format!(
                        "`{}` must match the HDR scene's format and size",
                        lib.name()
                    )));
                }
                check_bound_image(lib.name(), image, &want)?;
            }
            (ResourceDesc::Image(image), true) if lib == LibraryResource::WaterShore => {
                let Some((w, h)) = env.ground else {
                    return Err(error(format!(
                        "library resource `{}` does not exist in this scene (the map has no water)",
                        resource.name
                    )));
                };
                let format = crate::water::shore::FIELD_FORMAT;
                if image.format != format
                    || image.width != w
                    || image.height != h
                    || !image.aspect.contains(vk::ImageAspectFlags::COLOR)
                {
                    return Err(error(format!(
                        "`{}` is {:?} {}x{}; the shore field is {format:?} {w}x{h} with a colour aspect (one texel per ground height sample)",
                        resource.name, image.format, image.width, image.height
                    )));
                }
            }
            (ResourceDesc::Image(image), true) if lib == LibraryResource::WaterCaustics => {
                if env.ground.is_none() {
                    return Err(error(format!(
                        "library resource `{}` does not exist in this scene (the map has no water)",
                        resource.name
                    )));
                }
                let (format, size) = (crate::water::caustics::FORMAT, crate::water::caustics::SIZE);
                let mips = crate::upload::mip_levels(size, size);
                if image.format != format
                    || image.width != size
                    || image.height != size
                    || image.mip_levels != mips
                    || !image.aspect.contains(vk::ImageAspectFlags::COLOR)
                {
                    return Err(error(format!(
                        "`{}` is {:?} {}x{} with {} mips; the caustic web is {format:?} {size}x{size} with {mips} (caustics::SIZE)",
                        resource.name, image.format, image.width, image.height, image.mip_levels
                    )));
                }
            }
            (ResourceDesc::Image(image), true) if lib.is_transient() => {
                let canonical = image_desc(lib, env)?;
                if (image.format, image.width, image.height) != (canonical.format, canonical.width, canonical.height) {
                    return Err(error(format!(
                        "`{}` must be a {:?} attachment of {}x{}",
                        resource.name, canonical.format, canonical.width, canonical.height
                    )));
                }
            }
            (ResourceDesc::Buffer(buffer), true) => {
                // A library resource named without its descriptor: a graph
                // file's or module's `kind = library` entry. The host supplies
                // the image's format and size (the graph cannot know them).
                if buffer.size != 0 {
                    return Err(error(format!(
                        "`{}` is declared with {} bytes; a library image takes no size",
                        resource.name, buffer.size
                    )));
                }
                resource.desc = ResourceDesc::Image(image_desc(lib, env)?);
            }
            (ResourceDesc::Image(image), true) => {
                if image.format != env.color_format
                    || image.width != env.extent.width
                    || image.height != env.extent.height
                {
                    return Err(error(format!(
                        "`{}` is {:?} {}x{}; the target is {:?} {}x{}",
                        resource.name,
                        image.format,
                        image.width,
                        image.height,
                        env.color_format,
                        env.extent.width,
                        env.extent.height
                    )));
                }
            }
            (ResourceDesc::Buffer(buffer), false) => {
                let Some(want) = desc(lib) else {
                    return Err(error(format!(
                        "library resource `{}` does not exist in this scene",
                        resource.name
                    )));
                };
                if buffer.size != 0 && buffer.size != want.size {
                    return Err(error(format!(
                        "`{}` is declared with {} bytes; the library's is {} (declare 0)",
                        resource.name, buffer.size, want.size
                    )));
                }
                *buffer = want;
            }
            _ => {
                return Err(error(format!(
                    "library resource `{}` declared as the wrong kind",
                    resource.name
                )));
            }
        }
        // The host binds the image it allocated (the water images are built
        // by the water renderer, the shadow maps by the terrain). The graph
        // bakes the descriptor's usage into its access states and bindless
        // handles, so a module's explicit descriptor may promise less than
        // the allocation supports but never more: otherwise the graph is
        // accepted and every draw reports a Vulkan usage/layout error.
        if let ResourceDesc::Image(image) = &resource.desc
            && let Ok(canonical) = image_desc(lib, env)
        {
            check_bound_image(&resource.name, image, &canonical)?;
        }
        map.push(Some(lib));
    }
    Ok(map)
}

/// The descriptor of a library image (`None` for a library buffer): what a
/// graph file's or module's `kind = library` entry is filled in with, and
/// what a declared image has to match. The water data exist only on a map
/// with water; a `None` [`LibraryEnv::ground`] refuses them.
pub fn image_desc(lib: LibraryResource, env: &LibraryEnv) -> Result<ImageDesc, LibraryError> {
    use LibraryResource as R;
    use ae_render_abi::module as m;
    let missing = |why: &str| {
        error(format!(
            "library image `{}` does not exist in this scene ({why})",
            lib.name()
        ))
    };
    Ok(match lib {
        R::ShadowMap => {
            if !env.shadows {
                return Err(missing("its units cast no shadows"));
            }
            let mut desc = ImageDesc::color_2d(
                vk::Format::from_raw(m::SHADOW_MAP_FORMAT as i32),
                m::SHADOW_MAP_WIDTH,
                m::SHADOW_MAP_HEIGHT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            );
            desc.aspect = vk::ImageAspectFlags::DEPTH;
            desc
        }
        R::ParticleShadows => {
            if !(env.shadows && env.particles) {
                return Err(missing("it needs units that cast shadows and particles"));
            }
            ImageDesc::color_2d(
                vk::Format::from_raw(m::PARTICLE_SHADOW_FORMAT as i32),
                m::PARTICLE_SHADOW_WIDTH,
                m::PARTICLE_SHADOW_HEIGHT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            )
        }
        R::Exposure => ImageDesc::color_2d(
            vk::Format::from_raw(m::EXPOSURE_FORMAT as i32),
            1,
            1,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
        ),
        R::WaterShore => {
            let Some((w, h)) = env.ground else {
                return Err(missing("the map has no water"));
            };
            ImageDesc::color_2d(
                crate::water::shore::FIELD_FORMAT,
                w,
                h,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            )
        }
        R::WaterCaustics => {
            if env.ground.is_none() {
                return Err(missing("the map has no water"));
            }
            let size = crate::water::caustics::SIZE;
            ImageDesc {
                mip_levels: crate::upload::mip_levels(size, size),
                ..ImageDesc::color_2d(crate::water::caustics::FORMAT, size, size, vk::ImageUsageFlags::SAMPLED)
            }
        }

        R::WaterRaw | R::WaterChurn | R::WaterWaves => {
            if !env.water {
                return Err(missing("the scene draws no water"));
            }
            crate::water::waves::patch_desc(lib == R::WaterWaves)
        }
        R::WaterRefraction | R::WaterVolumeOut => ImageDesc::color_2d(
            env.scene_format,
            env.extent.width,
            env.extent.height,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
        ),
        R::WaterReflection | R::WaterReflectionDepth => {
            if !env.water {
                return Err(missing("the scene draws no water"));
            }
            let extent = crate::water::reflect::half_extent(env.extent);
            if lib == R::WaterReflection {
                ImageDesc::color_2d(
                    env.scene_format,
                    extent.width,
                    extent.height,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                )
            } else {
                ImageDesc::depth_2d(
                    env.depth_format,
                    extent.width,
                    extent.height,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                )
            }
        }
        R::ReadbackTarget
        | R::TerrainLeavesReflect
        | R::TerrainIndirectReflect
        | R::UnitVisibleReflect
        | R::UnitIndirectReflect
        | R::TerrainLeaves
        | R::TerrainIndirect
        | R::UnitInstances
        | R::UnitVisible
        | R::UnitIndirect
        | R::UnitCounts
        | R::UnitPalette
        | R::ReadbackBuffer
        | R::ParticleState
        | R::ParticleKeys
        | R::ParticleIndirect
        | R::LightList
        | R::LightClusters
        | R::WaterParticles
        | R::WaterMeshLeaves
        | R::WaterMeshIndirect
        | R::ExposureHistogram => {
            return Err(error(format!(
                "library resource `{}` is not an image this scene fills in (declare its descriptor)",
                lib.name()
            )));
        }
    })
}

/// An explicit descriptor may ask for no more than the image the host binds:
/// unsupported usage, extra samples, layers, mips or aspects describe a
/// different image than the one the draws actually use. Usage is a subset
/// (a declaration may sample an image it never attaches); the rest must
/// match the allocation.
fn check_bound_image(name: &str, image: &ImageDesc, canonical: &ImageDesc) -> Result<(), LibraryError> {
    let unsupported = image.usage & !canonical.usage;
    if !unsupported.is_empty() {
        return Err(error(format!(
            "`{name}` declares image usage {unsupported:?}, which the bound image does not support (its usage is {:?})",
            canonical.usage
        )));
    }
    if image.samples != canonical.samples {
        return Err(error(format!(
            "`{name}` is {:?} sampled; the bound image is {:?}",
            image.samples, canonical.samples
        )));
    }
    if image.layers != canonical.layers {
        return Err(error(format!(
            "`{name}` has {} layers; the bound image has {}",
            image.layers, canonical.layers
        )));
    }
    if image.mip_levels != canonical.mip_levels {
        return Err(error(format!(
            "`{name}` has {} mip levels; the bound image has {}",
            image.mip_levels, canonical.mip_levels
        )));
    }
    if image.aspect != canonical.aspect {
        return Err(error(format!(
            "`{name}` has aspect {:?}; the bound image's is {:?}",
            image.aspect, canonical.aspect
        )));
    }
    Ok(())
}

/// Check that pass `pass` of `graph`, wired as library pass `lib`, uses
/// what the library needs and nothing else.
pub fn check<R>(
    graph: &FrameGraph<R>,
    pass: usize,
    lib: LibraryPass,
    map: &ResourceMap,
    env: &LibraryEnv,
) -> Result<(), LibraryError> {
    let p = &graph.passes[pass];
    let name = &p.name;
    if p.kind != lib.kind() {
        return Err(error(format!(
            "pass `{name}` ({}) must be a {:?} pass, not {:?}",
            lib.name(),
            lib.kind(),
            p.kind
        )));
    }
    if lib.uses_units() && !env.units {
        return Err(error(format!(
            "pass `{name}` ({}) needs units; this scene has none",
            lib.name()
        )));
    }
    if lib.uses_particles() && !env.particles {
        return Err(error(format!(
            "pass `{name}` ({}) needs particles; this scene has none",
            lib.name()
        )));
    }
    if lib == LibraryPass::WaterDraw && !env.water {
        return Err(error(format!(
            "pass `{name}` (water.draw) needs water; this scene draws none"
        )));
    }
    if matches!(lib, LibraryPass::ShadowsDraw | LibraryPass::ShadowsParticles) && !env.shadows {
        return Err(error(format!(
            "pass `{name}` ({}) needs units that cast shadows; this scene's cast none",
            lib.name()
        )));
    }
    let resource = |id: ResourceId| &graph.resources[id.index()];
    // The shadow maps, for a receiver in a graph that declares them.
    let receives: Vec<(LibraryResource, Role)> = [LibraryResource::ShadowMap, LibraryResource::ParticleShadows]
        .into_iter()
        .filter(|r| map.contains(&Some(*r)))
        .filter_map(|r| lib.receives().map(|role| (r, role)))
        .chain(
            [LibraryResource::LightList, LibraryResource::LightClusters]
                .into_iter()
                .filter(|r| map.contains(&Some(*r)))
                .filter_map(|r| lib.lit().map(|role| (r, role))),
        )
        .collect();
    for &(needed, role) in lib.requires().iter().chain(&receives) {
        if !p
            .intents
            .iter()
            .any(|i| map[i.resource.index()] == Some(needed) && i.role == role)
        {
            let why = match needed {
                LibraryResource::LightList | LibraryResource::LightClusters
                    if receives.iter().any(|r| r.0 == needed) =>
                {
                    format!(" (the graph declares `{}`: every lit pass reads it)", needed.name())
                }
                _ if receives.iter().any(|r| r.0 == needed) => {
                    format!(" (the graph declares `{}`: every receiver samples it)", needed.name())
                }
                _ => String::new(),
            };
            return Err(error(format!(
                "pass `{name}` ({}) must use `{}` as {role:?}{why}",
                lib.name(),
                needed.name()
            )));
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
    if matches!(
        lib,
        LibraryPass::BloomDownsample | LibraryPass::BloomBlurX | LibraryPass::BloomBlurY
    ) {
        return check_bloom(graph, pass, lib, env);
    }
    // The water data a surface may sample when the graph declares them: M7's
    // shader ignores them; the rich tiers read them (stage 3).
    let mut allowed: Vec<(LibraryResource, Role)> = if lib == LibraryPass::WaterDraw {
        [
            LibraryResource::WaterShore,
            LibraryResource::WaterCaustics,
            LibraryResource::WaterReflection,
            LibraryResource::WaterRefraction,
        ]
        .into_iter()
        .filter(|r| map.contains(&Some(*r)))
        .map(|r| (r, Role::Sampled(ShaderStage::Fragment)))
        .collect()
    } else {
        Vec::new()
    };

    if map.contains(&Some(LibraryResource::WaterWaves)) {
        if lib == LibraryPass::WaterDraw {
            allowed.extend([
                (LibraryResource::WaterWaves, Role::Sampled(ShaderStage::Graphics)),
                (
                    LibraryResource::WaterMeshLeaves,
                    Role::StorageBufferRead(ShaderStage::Vertex),
                ),
                (LibraryResource::WaterMeshIndirect, Role::IndirectBuffer),
                (LibraryResource::WaterChurn, Role::Sampled(ShaderStage::Fragment)),
            ]);
        }
        if matches!(
            lib,
            LibraryPass::ParticlesUnderwater | LibraryPass::WaterUnderwater | LibraryPass::WaterVolume
        ) {
            allowed.push((LibraryResource::WaterWaves, Role::Sampled(ShaderStage::Fragment)));
        }
        if lib == LibraryPass::ParticlesSim {
            allowed.push((LibraryResource::WaterWaves, Role::Sampled(ShaderStage::Compute)));
        }
        if matches!(
            lib,
            LibraryPass::UnitsDraw | LibraryPass::ShadowsDraw | LibraryPass::UnitsReflect
        ) {
            allowed.push((LibraryResource::WaterWaves, Role::Sampled(ShaderStage::Vertex)));
        }
    }
    let glow = glow_resource(graph).is_some();
    let glow_format = vk::Format::from_raw(ae_render_abi::module::GLOW_FORMAT as i32);
    let (mut scene, mut glow_color, mut depth, mut source) = (0, 0, 0, 0);
    for intent in &p.intents {
        let r = resource(intent.resource);
        if lib
            .requires()
            .iter()
            .chain(&receives)
            .chain(&allowed)
            .any(|&(needed, role)| map[intent.resource.index()] == Some(needed) && intent.role == role)
        {
            continue;
        }
        match (lib, intent.role) {
            (LibraryPass::WaterUnderwater | LibraryPass::WaterVolume, Role::Sampled(ShaderStage::Fragment))
                if r.name == ae_render_abi::module::SCENE_NAME =>
            {
                source += 1;
                if !matches!(r.desc, ResourceDesc::Image(image) if image.format == env.scene_format
                    && (image.width, image.height) == (env.extent.width, env.extent.height))
                {
                    return Err(error(format!("pass `{name}` needs a full-size HDR scene")));
                }
            }
            (_, Role::ColorAttachment(_)) if lib.attachments().is_some() => {
                let ResourceDesc::Image(image) = r.desc else {
                    return Err(error(format!(
                        "pass `{name}` draws `{}`, which is not an image",
                        r.name
                    )));
                };
                let is_scene = r.name == ae_render_abi::module::SCENE_NAME
                    || (image.format == env.scene_format && image.format != glow_format);
                if is_scene && image.format != env.scene_format {
                    return Err(error(format!(
                        "pass `{name}` draws into `{}` ({:?}); the library's pipelines target the scene's {:?}",
                        r.name, image.format, env.scene_format
                    )));
                } else if is_scene {
                    scene += 1;
                    if (image.width, image.height) != (env.extent.width, env.extent.height) {
                        return Err(error(format!(
                            "pass `{name}` draws into `{}` at {}x{}; the scene is {}x{}",
                            r.name, image.width, image.height, env.extent.width, env.extent.height
                        )));
                    }
                } else if image.format == glow_format && lib.attachments().is_some_and(|a| a.glow) {
                    glow_color += 1;
                    if (image.width, image.height) != (env.extent.width, env.extent.height) {
                        return Err(error(format!(
                            "pass `{name}` writes the glow `{}` at {}x{}; it is {}x{}",
                            r.name, image.width, image.height, env.extent.width, env.extent.height
                        )));
                    }
                } else {
                    return Err(error(format!(
                        "pass `{name}` draws into `{}` ({:?}); the library's pipelines target the scene's {:?} and its glow {:?}",
                        r.name, image.format, env.scene_format, glow_format
                    )));
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
                    return Err(error(format!(
                        "pass `{name}` uses depth `{}` ({:?}); the library's pipelines use {:?}",
                        r.name, image.format, env.depth_format
                    )));
                }
            }
            (LibraryPass::ReadbackCopy, Role::TransferSrc) if r.desc.is_image() => {}
            (LibraryPass::HostRead, Role::HostRead) => {}
            (LibraryPass::Present, Role::Present) => {}
            _ => {
                return Err(error(format!(
                    "pass `{name}` ({}) does not use `{}` as {:?}",
                    lib.name(),
                    r.name,
                    intent.role
                )));
            }
        }
    }
    if matches!(lib, LibraryPass::WaterUnderwater | LibraryPass::WaterVolume) && source != 1 {
        return Err(error(format!("pass `{name}` must sample `scene` once")));
    }
    if let Some(a) = lib.attachments() {
        let need_glow = a.glow && glow;
        if scene != a.color || glow_color != usize::from(need_glow) || depth != 1 {
            let role = match a.depth {
                DepthUse::Write => "a depth attachment",
                DepthUse::Read => "a read-only depth attachment (depth_read)",
                DepthUse::Sampled => "the depth buffer sampled in the fragment stage (sampled:fragment)",
            };
            let glow_text = if need_glow {
                ", the glow as a second colour attachment,"
            } else {
                ""
            };
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
    graph
        .resources
        .iter()
        .position(|r| r.name == ae_render_abi::module::GLOW_NAME)
        .map(|i| ResourceId(i as u32))
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
            return Err(error(format!(
                "pass `{name}` (post.tonemap) does not use buffer `{}`",
                r.name
            )));
        };
        match intent.role {
            Role::Sampled(ShaderStage::Fragment) if sampled == 0 => {
                if image.format != env.scene_format
                    || (image.width, image.height) != (env.extent.width, env.extent.height)
                {
                    return Err(error(format!(
                        "pass `{name}` (post.tonemap) samples `{}` ({:?} {}x{}) first; the scene is {:?} {}x{}",
                        r.name,
                        image.format,
                        image.width,
                        image.height,
                        env.scene_format,
                        env.extent.width,
                        env.extent.height
                    )));
                }
                scene += 1;
                sampled += 1;
            }
            Role::Sampled(ShaderStage::Fragment)
                if image.format == exposure_format && (image.width, image.height) == (1, 1) =>
            {
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
                    if env.encoding.uses_display() {
                        "display"
                    } else {
                        "output"
                    },
                    env.display_format()
                )));
            }
            role => {
                return Err(error(format!(
                    "pass `{name}` (post.tonemap) does not use `{}` as {role:?}",
                    r.name
                )));
            }
        }
    }
    let need_bloom = glow_resource(graph).is_some();
    let need_exposure = graph
        .resources
        .iter()
        .any(|r| r.name == ae_render_abi::module::EXPOSURE_NAME);
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
        return Err(error(format!(
            "pass `{name}` (post.encode) does not exist in an SDR graph; post.tonemap writes the output"
        )));
    }
    let (w, h) = (env.extent.width, env.extent.height);
    let (mut display, mut target) = (0, 0);
    for intent in &p.intents {
        let r = &graph.resources[intent.resource.index()];
        let ResourceDesc::Image(image) = r.desc else {
            return Err(error(format!(
                "pass `{name}` (post.encode) does not use buffer `{}`",
                r.name
            )));
        };
        match intent.role {
            Role::Sampled(ShaderStage::Fragment)
                if image.format == crate::encode::DISPLAY_FORMAT && (image.width, image.height) == (w, h) =>
            {
                display += 1
            }
            Role::ColorAttachment(_) if image.format == env.color_format && (image.width, image.height) == (w, h) => {
                target += 1
            }
            Role::Sampled(ShaderStage::Fragment) => {
                return Err(error(format!(
                    "pass `{name}` (post.encode) samples `{}` ({:?} {}x{}); the display is {:?} {w}x{h}",
                    r.name,
                    image.format,
                    image.width,
                    image.height,
                    crate::encode::DISPLAY_FORMAT
                )));
            }
            Role::ColorAttachment(_) => {
                return Err(error(format!(
                    "pass `{name}` (post.encode) writes `{}` ({:?}); the output is {:?}",
                    r.name, image.format, env.color_format
                )));
            }
            role => {
                return Err(error(format!(
                    "pass `{name}` (post.encode) does not use `{}` as {role:?}",
                    r.name
                )));
            }
        }
    }
    if display != 1 || target != 1 {
        return Err(error(format!(
            "pass `{name}` (post.encode) samples one display image and writes one colour attachment; it has {display} and {target}"
        )));
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
    let (want_in, want_size) = if lib == LibraryPass::BloomDownsample {
        (glow_format, (env.extent.width, env.extent.height))
    } else {
        (bloom_format, (bw, bh))
    };
    let (mut input, mut output, mut exposure) = (0, 0, 0);
    for intent in &p.intents {
        let r = &graph.resources[intent.resource.index()];
        let ResourceDesc::Image(image) = r.desc else {
            return Err(error(format!(
                "pass `{name}` ({}) does not use buffer `{}`",
                lib.name(),
                r.name
            )));
        };
        match intent.role {
            Role::Sampled(ShaderStage::Fragment)
                if image.format == want_in && (image.width, image.height) == want_size =>
            {
                input += 1
            }
            Role::Sampled(ShaderStage::Fragment)
                if lib == LibraryPass::BloomDownsample
                    && image.format == exposure_format
                    && (image.width, image.height) == (1, 1) =>
            {
                exposure += 1
            }
            Role::ColorAttachment(_) if image.format == bloom_format && (image.width, image.height) == (bw, bh) => {
                output += 1
            }
            Role::Sampled(ShaderStage::Fragment) => {
                return Err(error(format!(
                    "pass `{name}` ({}) samples `{}` ({:?} {}x{}); it reads {:?} {}x{}",
                    lib.name(),
                    r.name,
                    image.format,
                    image.width,
                    image.height,
                    want_in,
                    want_size.0,
                    want_size.1
                )));
            }
            Role::ColorAttachment(_) => {
                return Err(error(format!(
                    "pass `{name}` ({}) writes `{}` ({:?} {}x{}); it writes the bloom {:?} {bw}x{bh}",
                    lib.name(),
                    r.name,
                    image.format,
                    image.width,
                    image.height,
                    bloom_format
                )));
            }
            role => {
                return Err(error(format!(
                    "pass `{name}` ({}) does not use `{}` as {role:?}",
                    lib.name(),
                    r.name
                )));
            }
        }
    }
    let need_exposure = lib == LibraryPass::BloomDownsample
        && graph
            .resources
            .iter()
            .any(|r| r.name == ae_render_abi::module::EXPOSURE_NAME);
    if need_exposure && exposure != 1 {
        return Err(error(format!(
            "pass `{name}` (bloom.downsample) must sample `{}` (the graph declares it: the clamp follows the eased EV)",
            ae_render_abi::module::EXPOSURE_NAME
        )));
    }
    if input != 1
        || output != 1
        || exposure != usize::from(need_exposure)
        || p.intents.len() != 2 + usize::from(need_exposure)
    {
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
pub fn check_graph<R>(
    graph: &FrameGraph<R>,
    map: &ResourceMap,
    env: &LibraryEnv,
    libs: impl Fn(usize) -> Option<LibraryPass>,
) -> Result<WaterTiers, LibraryError> {
    check_graph_named(graph, map, env, |i| libs(i).map(LibraryPass::name))
}

/// Check the tier contracts before resolving library names to recorders.
/// This also diagnoses partial declarations of tiers whose recorders are
/// still under development.
pub(crate) fn check_graph_named<'a, R>(
    graph: &FrameGraph<R>,
    map: &ResourceMap,
    env: &LibraryEnv,
    libs: impl Fn(usize) -> Option<&'a str>,
) -> Result<WaterTiers, LibraryError> {
    let written = |lib: LibraryPass| (0..graph.passes.len()).any(|i| libs(i) == Some(lib.name()));
    let draws_units = [
        LibraryPass::UnitsDraw,
        LibraryPass::UnitsWaterOverlay,
        LibraryPass::ShadowsDraw,
        LibraryPass::UnitsReflect,
    ];
    if draws_units.into_iter().any(&written) && !written(LibraryPass::UnitsPose) {
        return Err(error(
            "the graph draws the units but no `units.pose` pass writes their palettes (`units.palette`)".into(),
        ));
    }
    let (units, particles) = (
        map.contains(&Some(LibraryResource::ShadowMap)),
        map.contains(&Some(LibraryResource::ParticleShadows)),
    );
    if units && !written(LibraryPass::ShadowsDraw) {
        return Err(error(
            "the graph declares `shadows.map` but no `shadows.draw` pass writes it".into(),
        ));
    }
    if particles && !units {
        return Err(error(
            "the graph declares `shadows.color` without `shadows.map` (the coloured shadows use its cascades)".into(),
        ));
    }
    if particles && !written(LibraryPass::ShadowsParticles) {
        return Err(error(
            "the graph declares `shadows.color` but no `shadows.particles` pass writes it".into(),
        ));
    }
    // The water data: the shore field has one host-scheduled producer, and a
    // graph that declares it needs that producer (a reader without one would
    // sample an undefined image).
    let shore_passes = (0..graph.passes.len())
        .filter(|&i| libs(i) == Some(LibraryPass::WaterShore.name()))
        .count();
    if shore_passes > 1 {
        return Err(error(format!(
            "the graph has {shore_passes} `water.shore` passes; the field has one producer (they share its scratch images)"
        )));
    }
    if map.contains(&Some(LibraryResource::WaterShore)) && shore_passes == 0 {
        return Err(error(
            "the graph declares `water.shore` but no `water.shore` pass bakes it".to_string(),
        ));
    }
    let (list, clusters) = (
        map.contains(&Some(LibraryResource::LightList)),
        map.contains(&Some(LibraryResource::LightClusters)),
    );
    if list != clusters {
        return Err(error(
            "the graph declares one of `lights.list` and `lights.clusters`; the lights need both".into(),
        ));
    }
    if clusters && !written(LibraryPass::LightsCluster) {
        return Err(error(
            "the graph declares `lights.clusters` but no `lights.cluster` pass writes it".into(),
        ));
    }
    // The pass appends the units' model lights from the palettes
    // `units.pose` writes: it must run after it.
    if clusters && written(LibraryPass::UnitsPose) {
        let first = |lib: LibraryPass| (0..graph.passes.len()).find(|&i| libs(i) == Some(lib.name()));
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
            ResourceDesc::Image(image) => {
                return Err(error(format!(
                    "`{}` is {:?}; the glow is {want:?} (glow_spec)",
                    r.name, image.format
                )));
            }
            ResourceDesc::Buffer(_) => return Err(error(format!("`{}` is a buffer; the glow is an image", r.name))),
        }
        let missing: Vec<&str> = [
            LibraryPass::BloomDownsample,
            LibraryPass::BloomBlurX,
            LibraryPass::BloomBlurY,
        ]
        .into_iter()
        .filter(|lib| !written(*lib))
        .map(LibraryPass::name)
        .collect();
        if !missing.is_empty() {
            return Err(error(format!(
                "the graph declares `{}` but no `{}` pass (the bloom reads it)",
                r.name,
                missing.join("` or `")
            )));
        }
    } else if written(LibraryPass::BloomDownsample)
        || written(LibraryPass::BloomBlurX)
        || written(LibraryPass::BloomBlurY)
    {
        return Err(error(
            "the graph has `post.bloom` passes but no `glow` image (bloom reads the light sources' glow)".into(),
        ));
    }
    // Automatic exposure: the image and the histogram go with the three
    // passes, all or none.
    let (exposure, histogram) = (
        map.contains(&Some(LibraryResource::Exposure)),
        map.contains(&Some(LibraryResource::ExposureHistogram)),
    );
    if exposure != histogram {
        return Err(error(format!(
            "the graph declares one of `{}` and `{}`; auto-exposure needs both",
            LibraryResource::Exposure.name(),
            LibraryResource::ExposureHistogram.name()
        )));
    }
    let exposure_passes = [
        LibraryPass::ExposureClear,
        LibraryPass::ExposureMeter,
        LibraryPass::ExposureAdapt,
    ];
    let missing: Vec<&str> = exposure_passes
        .into_iter()
        .filter(|lib| !written(*lib))
        .map(LibraryPass::name)
        .collect();
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
        let first = |lib: LibraryPass| (0..graph.passes.len()).find(|&i| libs(i) == Some(lib.name()));
        if let (Some(clear), Some(meter), Some(adapt)) = (
            first(LibraryPass::ExposureClear),
            first(LibraryPass::ExposureMeter),
            first(LibraryPass::ExposureAdapt),
        ) && !(clear < meter && meter < adapt)
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
            format!(
                "the output is {} but the graph has no `post.encode` pass: the tonemap writes the display, not the target",
                env.encoding.label()
            )
        } else {
            format!(
                "the output is {} but the graph has an extra `post.encode` pass: the tonemap writes the target",
                env.encoding.label()
            )
        }));
    }
    water::check(graph, env, libs)
}

/// Check the water contract against the passes the bake kept. The tier's
/// resources remain in the declaration, so a pruned producer or consumer
/// is reported the same way as a missing one.
pub(crate) fn check_baked_water_tiers<'a, R>(
    graph: &FrameGraph<R>,
    baked: &ae_framegraph::BakedGraph,
    env: &LibraryEnv,
    libs: impl Fn(usize) -> Option<&'a str>,
) -> Result<WaterTiers, LibraryError> {
    let mut live = vec![false; graph.passes.len()];
    for pass in baked.order() {
        live[pass.index()] = true;
    }
    water::check(graph, env, |i| if live[i] { libs(i) } else { None })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::app::{TerrainOutput, native_graph};
    use ae_framegraph::{ImageDesc, LoadOp, PassKind};

    fn env() -> LibraryEnv {
        LibraryEnv {
            color_format: vk::Format::R8G8B8A8_SRGB,
            scene_format: vk::Format::R16G16B16A16_SFLOAT,
            depth_format: vk::Format::D32_SFLOAT,
            extent: vk::Extent2D { width: 64, height: 32 },
            units: true,
            water: true,
            shadows: true,
            particles: true,
            ground: Some((64, 64)),
            bake_water: false,
            encoding: crate::encode::OutputEncoding::Sdr,
            levels: crate::encode::DisplayLevels::default(),
            working: Default::default(),
            gamut: Default::default(),
        }
    }

    fn size(r: LibraryResource) -> Option<BufferDesc> {
        Some(BufferDesc {
            size: 64 + r as u64,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        })
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
            assert_eq!(
                leaves.desc,
                ae_framegraph::ResourceDesc::Buffer(size(LibraryResource::TerrainLeaves).unwrap())
            );
            for lib in [
                LibraryPass::ExposureClear,
                LibraryPass::ExposureMeter,
                LibraryPass::ExposureAdapt,
            ] {
                assert!(
                    graph.passes.iter().any(|p| p.record == lib),
                    "the scene meters and adapts the exposure"
                );
            }
            assert!(
                graph
                    .resources
                    .iter()
                    .any(|r| r.name == ae_render_abi::module::EXPOSURE_NAME && r.desc.is_image())
            );
            let histogram = graph
                .resources
                .iter()
                .find(|r| r.name == ae_render_abi::module::EXPOSURE_HISTOGRAM_NAME)
                .unwrap();
            assert_eq!(
                histogram.desc,
                ae_framegraph::ResourceDesc::Buffer(size(LibraryResource::ExposureHistogram).unwrap())
            );
        }
        let no_units = LibraryEnv { units: false, ..env() };
        assert!(wire(native_graph(&no_units, TerrainOutput::Present), &no_units).is_ok());
        let dry = LibraryEnv { water: false, ..env() };
        let graph = wire(native_graph(&dry, TerrainOutput::Present), &dry).unwrap();
        assert!(
            !graph.passes.iter().any(|p| p.record == LibraryPass::WaterDraw),
            "no water pass without water"
        );
        let graph = native_graph(&env(), TerrainOutput::Readback);
        let (units, water) = (
            pass(&graph, LibraryPass::UnitsDraw),
            pass(&graph, LibraryPass::WaterDraw),
        );
        assert!(water > units, "the water is declared after the units");
    }

    #[test]
    fn reflection_declares_clear_and_load_on_the_half_resolution_target() {
        use crate::terrain::app::native_graph_with_water_tiers;
        let env = env();
        let tiers = WaterTiers {
            reflect: true,
            ..Default::default()
        };
        let mut graph = native_graph_with_water_tiers(&env, TerrainOutput::Present, tiers);
        assert!(wire(graph.clone(), &env).is_ok());
        let pass = pass(&graph, LibraryPass::UnitsReflect);
        let color = graph.passes[pass]
            .intents
            .iter_mut()
            .find(|i| matches!(i.role, Role::ColorAttachment(_)))
            .unwrap();
        color.role = Role::ColorAttachment(LoadOp::Clear);
        let error = wire(graph, &env).unwrap_err().0;
        assert!(
            error.contains("units.reflect") && error.contains("water.reflection"),
            "{error}"
        );
    }

    #[test]
    fn reflected_and_overlaid_units_need_their_palettes() {
        for lib in ["units.reflect", "units.water_overlay"] {
            let mut graph = FrameGraph::new();
            graph.pass(lib, PassKind::Graphics, Some(lib)).id();
            let error = check_graph_named(&graph, &Vec::new(), &env(), |i| graph.passes[i].record)
                .unwrap_err()
                .0;
            assert!(error.contains("no `units.pose` pass"), "{lib}: {error}");
        }
    }

    #[test]
    fn refraction_reserves_scene_alpha_after_units() {
        use crate::terrain::app::native_graph_with_water_tiers;
        let env = env();
        let mut graph = native_graph_with_water_tiers(
            &env,
            TerrainOutput::Present,
            WaterTiers {
                refract: true,
                ..Default::default()
            },
        );
        let resources = resolve(&mut graph, &env, size).unwrap();
        let scene = ResourceId(graph.resources.iter().position(|r| r.name == "scene").unwrap() as u32);
        let underwater = pass(&graph, LibraryPass::WaterUnderwater);
        graph
            .pass("custom scene alpha reader", PassKind::Graphics, LibraryPass::HostRead)
            .sampled(scene, ShaderStage::Fragment)
            .id();
        let injected = graph.passes.pop().unwrap();
        graph.passes.insert(underwater, injected);
        let error = check_graph_named(&graph, &resources, &env, |i| Some(graph.passes[i].record.name()))
            .unwrap_err()
            .0;
        assert!(error.contains("alpha is reserved"), "{error}");
    }

    #[test]
    fn water_tiers_require_real_producers_and_consumers() {
        use ae_render_abi::module::{
            WATER_REFLECTION_NAME, WATER_REFRACTION_NAME, WATER_VOLUME_OUT_NAME, WATER_WAVES_NAME,
        };

        for (producer, output) in [
            ("terrain.reflect", WATER_REFLECTION_NAME),
            ("water.underwater", WATER_REFRACTION_NAME),
            ("water.blur", WATER_WAVES_NAME),
            ("water.volume", WATER_VOLUME_OUT_NAME),
        ] {
            let mut graph = FrameGraph::new();
            graph.pass("display label", PassKind::Graphics, producer).id();
            let error = check_graph_named(&graph, &Vec::new(), &env(), |i| Some(graph.passes[i].record))
                .unwrap_err()
                .0;
            assert!(error.contains(producer) && error.contains(output), "{error}");
        }

        let desc = ImageDesc::color_2d(
            env().scene_format,
            64,
            64,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        );
        let mut graph = FrameGraph::new();
        let output = graph.image(WATER_REFRACTION_NAME, desc, ResourceClass::Transient);
        let scene = graph.image("scene", desc, ResourceClass::Transient);
        let depth = graph.image(
            "depth",
            ImageDesc::depth_2d(
                env().depth_format,
                64,
                64,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ),
            ResourceClass::Transient,
        );
        graph
            .pass("terrain", PassKind::Graphics, Some("terrain.draw"))
            .color(scene, LoadOp::Clear)
            .depth(depth, LoadOp::Clear)
            .id();
        let env = LibraryEnv {
            units: false,
            shadows: false,
            particles: false,
            ..env()
        };
        // A matching label on a fullscreen program cannot own refraction.
        graph
            .pass("water.underwater", PassKind::Graphics, None)
            .color(output, LoadOp::Clear)
            .sampled(scene, ShaderStage::Fragment)
            .sampled(depth, ShaderStage::Fragment)
            .id();
        graph
            .pass("water surface", PassKind::Graphics, Some("water.draw"))
            .color(scene, LoadOp::Load)
            .depth_read(depth)
            .sampled(output, ShaderStage::Fragment)
            .root()
            .id();
        let check = |graph: &FrameGraph<Option<&str>>| {
            let libs = |i: usize| graph.passes[i].record;
            check_graph_named(graph, &vec![None; graph.resources.len()], &env, libs)
        };
        let error = check(&graph).unwrap_err().0;
        assert!(
            error.contains("water.underwater") && error.contains(WATER_REFRACTION_NAME),
            "{error}"
        );
        graph.passes[1].record = Some("water.underwater");
        assert_eq!(
            check(&graph),
            Ok(WaterTiers {
                refract: true,
                ..WaterTiers::default()
            })
        );

        let check_baked = |graph: &FrameGraph<Option<&str>>| {
            let baked = ae_framegraph::bake(graph, Default::default()).unwrap();
            check_baked_water_tiers(graph, &baked, &env, |i| graph.passes[i].record)
        };
        assert!(check_baked(&graph).unwrap().refract);
        graph.roots.clear();
        let unrelated = graph.image("unrelated", desc, ResourceClass::Transient);
        graph
            .pass("unrelated", PassKind::Graphics, None)
            .color(unrelated, LoadOp::Clear)
            .root()
            .id();
        let error = check_baked(&graph).unwrap_err().0;
        assert!(
            error.contains("water.draw") && error.contains(WATER_REFRACTION_NAME),
            "{error}"
        );

        graph.passes.swap(1, 2);
        let error = check(&graph).unwrap_err().0;
        assert!(
            error.contains("must run before") && error.contains(WATER_REFRACTION_NAME),
            "{error}"
        );
        graph.passes.swap(1, 2);
        graph.passes[2].intents.retain(|i| i.resource != output);
        let error = check(&graph).unwrap_err().0;
        assert!(
            error.contains("must sample") && error.contains(WATER_REFRACTION_NAME),
            "{error}"
        );

        let mut graph = FrameGraph::new();
        let output = graph.image(WATER_VOLUME_OUT_NAME, desc, ResourceClass::Transient);
        graph
            .pass("volume", PassKind::Graphics, Some("water.volume"))
            .color(output, LoadOp::Clear)
            .id();
        graph
            .pass("tonemap", PassKind::Graphics, Some("post.tonemap"))
            .sampled(output, ShaderStage::Fragment)
            .id();
        let error = check(&graph).unwrap_err().0;
        assert!(
            error.contains("water.volume") && error.contains(WATER_REFRACTION_NAME),
            "{error}"
        );
    }

    /// M16.5: the water data are graph resources of the library's fixed
    /// size and format — the shore field one texel per ground sample, the
    /// caustic web its 256² with the full mip chain — and they exist only on
    /// a map with water (a dry scene is refused, naming the resource).
    #[test]
    fn the_water_data_is_a_graph_resource_of_the_librarys_size() {
        use crate::water::{caustics, shore};
        let shore_desc = |w: u32, h: u32| {
            ImageDesc::color_2d(
                shore::FIELD_FORMAT,
                w,
                h,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
            )
        };
        let shore_graph = |desc: ImageDesc| {
            let mut graph = FrameGraph::<LibraryPass>::new();
            let resource = graph.image(LibraryResource::WaterShore.name(), desc, ResourceClass::External);
            graph
                .pass(
                    LibraryPass::WaterShore.name(),
                    PassKind::Compute,
                    LibraryPass::WaterShore,
                )
                .access(resource, Role::StorageImageWrite(ShaderStage::Compute))
                .root();
            graph
        };
        let env = env();
        assert!(wire(shore_graph(shore_desc(64, 64)), &env).is_ok());
        let error = wire(shore_graph(shore_desc(32, 64)), &env).unwrap_err().0;
        assert!(error.contains("water.shore") && error.contains("64x64"), "{error}");
        let dry = LibraryEnv { ground: None, ..env };
        let error = wire(shore_graph(shore_desc(64, 64)), &dry).unwrap_err().0;
        assert!(error.contains("water.shore") && error.contains("no water"), "{error}");

        let caustics_desc = ImageDesc {
            mip_levels: crate::upload::mip_levels(caustics::SIZE, caustics::SIZE),
            ..ImageDesc::color_2d(
                caustics::FORMAT,
                caustics::SIZE,
                caustics::SIZE,
                vk::ImageUsageFlags::SAMPLED,
            )
        };
        // The web has no pass of its own (it is made when the water renderer
        // is built, like the terrain's textures): resolve checks its
        // descriptor.
        let caustics_graph = |desc: ImageDesc| {
            let mut graph = FrameGraph::<LibraryPass>::new();
            graph.image(LibraryResource::WaterCaustics.name(), desc, ResourceClass::External);
            graph
        };
        assert!(resolve(&mut caustics_graph(caustics_desc), &env, |_| None).is_ok());
        let short = ImageDesc {
            mip_levels: 1,
            ..caustics_desc
        };
        let mut graph = caustics_graph(short);
        let error = resolve(&mut graph, &env, |_| None).unwrap_err().0;
        assert!(error.contains("water.caustics") && error.contains("256x256"), "{error}");

        // M16.5 fourth review: an explicit descriptor may promise no more
        // usage or shape than the image the host binds. The shore field is
        // storage+sampled (never an attachment), the web is sampled only,
        // and both are single-sample, single-layer 2D images.
        let shore_err = |desc: ImageDesc| wire(shore_graph(desc), &env).unwrap_err().0;
        let shore_base = shore_desc(64, 64);
        let mut attachment = shore_base;
        attachment.usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        let error = shore_err(attachment);
        assert!(
            error.contains("water.shore") && error.contains("COLOR_ATTACHMENT") && error.contains("usage"),
            "{error}"
        );
        let mut samples = shore_base;
        samples.samples = vk::SampleCountFlags::TYPE_2;
        let error = shore_err(samples);
        assert!(error.contains("water.shore") && error.contains("TYPE_2"), "{error}");
        let mut layers = shore_base;
        layers.layers = 2;
        let error = shore_err(layers);
        assert!(error.contains("water.shore") && error.contains("layers"), "{error}");
        let mut mips = shore_base;
        mips.mip_levels = 2;
        let error = shore_err(mips);
        assert!(error.contains("water.shore") && error.contains("mip levels"), "{error}");
        let mut aspect = shore_base;
        aspect.aspect |= vk::ImageAspectFlags::DEPTH;
        let error = shore_err(aspect);
        assert!(error.contains("water.shore") && error.contains("aspect"), "{error}");

        let caustics_err = |desc: ImageDesc| resolve(&mut caustics_graph(desc), &env, |_| None).unwrap_err().0;
        let mut storage = caustics_desc;
        storage.usage |= vk::ImageUsageFlags::STORAGE;
        let error = caustics_err(storage);
        assert!(
            error.contains("water.caustics") && error.contains("STORAGE") && error.contains("usage"),
            "{error}"
        );
        let mut samples = caustics_desc;
        samples.samples = vk::SampleCountFlags::TYPE_2;
        assert!(caustics_err(samples).contains("TYPE_2"));
        let mut layers = caustics_desc;
        layers.layers = 2;
        assert!(caustics_err(layers).contains("layers"));
        let mut aspect = caustics_desc;
        aspect.aspect |= vk::ImageAspectFlags::DEPTH;
        assert!(caustics_err(aspect).contains("aspect"));
    }

    /// M16.5 fifth review: a library resource is one host object, so a graph
    /// cannot declare it twice under different ids: the graph tracks producers
    /// and readers by resource id, and the second reader would not depend on
    /// the first writer (whose unrooted bake is pruned). The JSON loader
    /// refuses duplicate names; the module path must too.
    #[test]
    fn a_library_resource_is_declared_once() {
        use crate::water::shore;
        let env = env();
        let shore_desc = ImageDesc::color_2d(
            shore::FIELD_FORMAT,
            64,
            64,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        );
        let mut graph = FrameGraph::<LibraryPass>::new();
        graph.image(LibraryResource::WaterShore.name(), shore_desc, ResourceClass::External);
        graph.image(LibraryResource::WaterShore.name(), shore_desc, ResourceClass::External);
        let error = resolve(&mut graph, &env, |_| None).unwrap_err().0;
        assert!(
            error.contains("water.shore") && error.contains("more than once"),
            "{error}"
        );

        let leaves = BufferDesc {
            size: 0,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        };
        let mut graph = FrameGraph::<LibraryPass>::new();
        graph.buffer(LibraryResource::TerrainLeaves.name(), leaves, ResourceClass::External);
        graph.buffer(LibraryResource::TerrainLeaves.name(), leaves, ResourceClass::External);
        let size = |lib: LibraryResource| {
            (lib == LibraryResource::TerrainLeaves).then_some(BufferDesc {
                size: 16,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            })
        };
        let error = resolve(&mut graph, &env, size).unwrap_err().0;
        assert!(
            error.contains("terrain.leaves") && error.contains("more than once"),
            "{error}"
        );
    }

    #[test]
    fn an_hdr_graph_adds_the_display_and_the_encode() {
        use crate::encode::{DISPLAY_FORMAT, DISPLAY_NAME, DisplayLevels, OutputEncoding};
        let hdr = LibraryEnv {
            color_format: vk::Format::A2B10G10R10_UNORM_PACK32,
            encoding: OutputEncoding::Hdr10Pq,
            levels: DisplayLevels::default(),
            ..env()
        };
        let graph = wire(native_graph(&hdr, TerrainOutput::Readback), &hdr).unwrap();
        // The tonemap writes the display (paper-white units), and the encode
        // reads it and writes the 10-bit target.
        let display = graph
            .resources
            .iter()
            .find(|r| r.name == DISPLAY_NAME)
            .expect("the display exists");
        assert_eq!(
            display.desc,
            ae_framegraph::ResourceDesc::Image(ImageDesc::color_2d(
                DISPLAY_FORMAT,
                64,
                32,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
            ))
        );
        let tonemap = &graph.passes[pass(&graph, LibraryPass::Tonemap)];
        let colour = tonemap
            .intents
            .iter()
            .find(|i| matches!(i.role, Role::ColorAttachment(_)))
            .unwrap();
        assert_eq!(graph.resources[colour.resource.index()].name, DISPLAY_NAME);
        let encode = &graph.passes[pass(&graph, LibraryPass::PostEncode)];
        assert!(
            encode
                .intents
                .iter()
                .any(|i| i.role == Role::Sampled(ShaderStage::Fragment)
                    && graph.resources[i.resource.index()].name == DISPLAY_NAME)
        );
        assert!(encode.intents.iter().any(|i| matches!(i.role, Role::ColorAttachment(_))
            && graph.resources[i.resource.index()].name == LibraryResource::ReadbackTarget.name()));
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
        let display = g.image(
            DISPLAY_NAME,
            ImageDesc::color_2d(
                DISPLAY_FORMAT,
                64,
                32,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ),
            ResourceClass::Transient,
        );
        let target = g.image(
            "target",
            ImageDesc::color_2d(sdr.color_format, 64, 32, vk::ImageUsageFlags::COLOR_ATTACHMENT),
            ResourceClass::Transient,
        );
        g.pass("encode", PassKind::Graphics, LibraryPass::PostEncode)
            .sampled(display, ShaderStage::Fragment)
            .color(target, LoadOp::DontCare)
            .id();
        // An SDR output has no display and no encode.
        let message = wire(g, &sdr).unwrap_err().0;
        assert!(message.contains("does not exist in an SDR graph"), "{message}");
        // In HDR the tonemap must write the display, not the target.
        let hdr = LibraryEnv {
            color_format: vk::Format::A2B10G10R10_UNORM_PACK32,
            encoding: crate::encode::OutputEncoding::Hdr10Pq,
            ..env()
        };
        let mut g = FrameGraph::<LibraryPass>::new();
        let scene = g.image(
            ae_render_abi::module::SCENE_NAME,
            ImageDesc::color_2d(
                hdr.scene_format,
                64,
                32,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ),
            ResourceClass::Transient,
        );
        let out = g.image(
            "target",
            ImageDesc::color_2d(hdr.color_format, 64, 32, vk::ImageUsageFlags::COLOR_ATTACHMENT),
            ResourceClass::Transient,
        );
        g.pass("tonemap", PassKind::Graphics, LibraryPass::Tonemap)
            .sampled(scene, ShaderStage::Fragment)
            .color(out, LoadOp::DontCare)
            .id();
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
        g.passes[cull].intents.retain(|i| {
            !matches!(i.role, Role::StorageBufferWrite(_)) || g.resources[i.resource.index()].name != "units.visible"
        });
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
        let extra = g.buffer(
            "mine",
            BufferDesc {
                size: 16,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            },
            ResourceClass::Transient,
        );
        g.passes[lod].intents.push(ae_framegraph::Intent {
            resource: extra,
            role: Role::StorageBufferRead(ShaderStage::Compute),
            discard: false,
        });
        let e = err(g, &env());
        assert!(e.contains("does not use `mine`"), "{e}");

        // Attachments in formats the library's pipelines do not target.
        let other = LibraryEnv {
            scene_format: vk::Format::B10G11R11_UFLOAT_PACK32,
            ..env()
        };
        let e = err(native_graph(&other, TerrainOutput::Present), &env());
        assert!(e.contains("draws into `scene`"), "{e}");
        let other = LibraryEnv {
            color_format: vk::Format::B8G8R8A8_SRGB,
            ..env()
        };
        let e = err(native_graph(&other, TerrainOutput::Present), &env());
        assert!(e.contains("(post.tonemap) writes `target`"), "{e}");
        // The tonemap sampling no scene.
        let mut g = base();
        let t = pass(&g, LibraryPass::Tonemap);
        g.passes[t].intents.retain(|i| !matches!(i.role, Role::Sampled(_)));
        let e = err(g, &env());
        assert!(e.contains("samples one scene image"), "{e}");
        let other = LibraryEnv {
            depth_format: vk::Format::D32_SFLOAT_S8_UINT,
            ..env()
        };
        let e = err(native_graph(&other, TerrainOutput::Present), &env());
        assert!(e.contains("uses depth `depth`"), "{e}");

        // A library resource declared as something else.
        let mut g = base();
        let i = g.resources.iter().position(|r| r.name == "terrain.leaves").unwrap();
        g.resources[i].class = ResourceClass::Transient;
        assert!(err(g, &env()).contains("`terrain.leaves` must be External"));
        let mut g = base();
        g.resources[i].desc = ae_framegraph::ResourceDesc::Buffer(BufferDesc {
            size: 12,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        });
        assert!(err(g, &env()).contains("declare 0"));

        // Unit passes in a scene without units.
        let no_units = LibraryEnv { units: false, ..env() };
        let e = err(base(), &no_units);
        assert!(
            e.contains("does not exist in this scene") || e.contains("needs units"),
            "{e}"
        );

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
        let other = LibraryEnv {
            extent: vk::Extent2D { width: 65, height: 32 },
            ..env()
        };
        let e = err(native_graph(&other, TerrainOutput::Readback), &env());
        assert!(e.contains("`readback.target`"), "{e}");

        // Particles: the depth as a read-only attachment instead of
        // sampled, and particle passes in a scene without particles.
        let mut g = base();
        let draw = pass(&g, LibraryPass::ParticlesDraw);
        let depth = g.passes[draw]
            .intents
            .iter()
            .position(|i| i.role == Role::Sampled(ShaderStage::Fragment))
            .unwrap();
        g.passes[draw].intents[depth].role = Role::DepthRead;
        let e = err(g, &env());
        assert!(e.contains("`particles`") && e.contains("DepthRead"), "{e}");
        let still = LibraryEnv {
            particles: false,
            ..env()
        };
        let e = err(base(), &still);
        assert!(
            e.contains("needs particles") || e.contains("`shadows.color` does not exist"),
            "{e}"
        );

        // The particles' coloured shadows: a receiver that does not sample
        // them, their pass loading instead of clearing, a graph without
        // their pass or without the units' map.
        let mut g = base();
        let water = pass(&g, LibraryPass::WaterDraw);
        let tint = g.resources.iter().position(|r| r.name == "shadows.color").unwrap();
        g.passes[water].intents.retain(|i| i.resource.index() != tint);
        let e = err(g, &env());
        assert!(
            e.contains("`water`") && e.contains("`shadows.color`") && e.contains("every receiver"),
            "{e}"
        );
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
            let mut g = FrameGraph {
                resources: g.resources.clone(),
                passes: g.passes.clone(),
                roots: g.roots.clone(),
                plan_version: g.plan_version,
            };
            let map = resolve(&mut g, &env(), size).unwrap();
            check_graph(&g, &map, &env(), |i| Some(g.passes[i].record)).map_err(|e| e.0)
        };
        let g = base();
        assert_eq!(shadows(&g), Ok(WaterTiers::default()));
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
        let (clear, meter) = (
            pass(&g, LibraryPass::ExposureClear),
            pass(&g, LibraryPass::ExposureMeter),
        );
        g.passes.swap(clear, meter);
        let e = err(g, &env());
        assert!(e.contains("must run `exposure.clear`, then"), "{e}");
        let mut g = base();
        let t = pass(&g, LibraryPass::Tonemap);
        let exposure = g
            .resources
            .iter()
            .position(|r| r.name == ae_render_abi::module::EXPOSURE_NAME)
            .unwrap();
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
        let scene = g.passes[m]
            .intents
            .iter()
            .position(|i| i.role == Role::Sampled(ShaderStage::Compute))
            .unwrap();
        let blooms = g.resources.iter().position(|r| r.name == "bloom.0").unwrap();
        g.passes[m].intents[scene].resource = ResourceId(blooms as u32);
        let e = err(g, &env());
        assert!(e.contains("exposure.meter") && e.contains("one scene image"), "{e}");
    }
}
