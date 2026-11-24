//! Frame graphs from a data file: the same declaration a render module's
//! `declare` returns (graph descriptor + program table), written as JSON
//! instead of code. The host loads one wherever it takes a render module
//! ([`crate::module::RenderModule::load`] tells them apart by their first
//! byte), so it gets the same validation, wiring and hot reload.
//!
//! ```json
//! {
//!   "version": 1,
//!   "resources": [
//!     { "name": "terrain.leaves", "kind": "library" },
//!     { "name": "depth", "kind": "image", "format": "D32_SFLOAT", "usage": ["depth_stencil_attachment", "sampled"] },
//!     { "name": "scene", "kind": "image", "format": "target", "usage": ["color_attachment", "sampled"] },
//!     { "name": "target", "kind": "output" }
//!   ],
//!   "programs": [
//!     { "name": "grade", "fragment": "post/shaders/grade.frag", "uniforms": [1.04, 1.0, 0.94, 1.0] }
//!   ],
//!   "passes": [
//!     { "name": "terrain lod", "kind": "compute", "library": "terrain.lod",
//!       "access": ["terrain.leaves storage_buffer_write:compute"] },
//!     { "name": "grade", "kind": "graphics", "program": "grade",
//!       "access": ["scene sampled:fragment", "target color:dont_care"] },
//!     { "name": "vignette", "kind": "graphics", "program": "vignette", "flag": 0,
//!       "access": ["target color:load"] }
//!   ]
//! }
//! ```
//!
//! * **Resources** are declared in file order. `library`: an engine library
//!   resource by its name, without a descriptor: the
//!   host fills in a buffer's size or an image's format and size, including
//!   the units' shadow map `shadows.map`, the particles' coloured shadows
//!   `shadows.color`, the exposure image `exposure` and the water data
//!   (`water.shore`, `water.caustics`). Reflection draw buffers are
//!   transient. Other library resources are external.
//!   `image`:
//!   the target's size times `scale` (default 1), `format` a Vulkan format
//!   name, `target` (the target's colour format) or `display` (the HDR
//!   display intermediate's), `usage` flag names, `class` `transient`
//!   (default) or `persistent`. `output`: the image the
//!   frame ends in, exactly one; it becomes `target` (`Backbuffer`) when
//!   presenting and `readback.target` (`External`, also a transfer source)
//!   when reading back, and the host appends the present pass or the
//!   readback copy and host read, as modules write them.
//! * **Programs** are fullscreen fragment shaders: `fragment` is a `.spv`
//!   file or a `.frag` compiled with `glslc` (both relative to the file;
//!   [`inline`] replaces it by `fragment_hex` so the bytes stand alone),
//!   `uniforms` the floats of its uniform block (binding 0). With
//!   `uniforms_flag` (a bit 0..32) and `uniforms_off` (the same count), the
//!   block switches: `uniforms` while the bit is set, `uniforms_off` while
//!   it is clear, and the pass itself always runs (the HDR vignette
//!   multiplies by zero when off, so the encode always reads a written
//!   image).
//! * **Passes** run in file order. `kind` `graphics`, `compute` or
//!   `transfer`; a payload, either `library` (a library pass name) or
//!   `program`; `access` entries `"<resource> <role>[ discard]"`, roles
//!   `color:<load>`, `depth:<load>` (`clear`, `load`, `dont_care`),
//!   `depth_read`, `sampled:<stage>`, `storage_image_read:<stage>`,
//!   `storage_image_write:<stage>`, `storage_image_read_write:<stage>`,
//!   `storage_buffer_read:<stage>`, `storage_buffer_write:<stage>`,
//!   `storage_buffer_read_write:<stage>`,
//!   `uniform:<stage>` (`vertex`, `fragment`, `graphics`, `compute`),
//!   `vertex_buffer`, `index_buffer`, `indirect`, `transfer_src`,
//!   `transfer_dst`, `host_read`; `flag` n makes the pass conditional, run
//!   only while bit n of the application's flags is set; `root` keeps a
//!   pass whose results nothing reads. An access to a resource the scene
//!   leaves out (its `when` does not hold) is left out with it, so a pass
//!   can sample `shadows.map` only where the map exists. Use
//!   `water.volume.out|scene` to select the first active resource.
//! * Resources and passes may carry `when`: `units`, `no_units`,
//!   `readback`, `present`, `water`, `no_water`, `shadows` or `no_shadows`
//!   (the units cast shadows), `water_reflect`, `water_refract`,
//!   `water_waves` or `water_volume` (the requested wet-scene tiers),
//!   `water_reflect_units` (reflection and units),
//!   `particles` or `no_particles`,
//!   `particle_shadows` (both shadows and particles), `hdr` or `sdr` (the
//!   output's encoding: HDR keeps the display intermediate and ends in
//!   `post.encode`), to exist only in such scenes. A graph is checked for
//!   every combination at load, SDR and both HDR encodings included.

use crate::colorspace::{Gamut, WorkingSpace};
use crate::encode::{DisplayLevels, OutputEncoding};
use crate::module::{DeclareTarget, Declared, ModuleFrame};
use ae_render_abi::module::{
    DECLARE_PARTICLES, DECLARE_READBACK, DECLARE_SHADOWS, DECLARE_UNITS, DECLARE_WATER, DECLARE_WATER_REFLECT,
    DECLARE_WATER_REFRACT, DECLARE_WATER_VOLUME, DECLARE_WATER_WAVES, FrameInput, Program, ProgramTable,
};
use ae_render_abi::{Class, GraphBuilder, ImageSpec, LoadOp, PassKind, ResourceRef, Role, ShaderStage};
use anyhow::{Context as _, Result, bail, ensure};
use ash::vk;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

pub const FILE_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSpec {
    version: u32,
    #[serde(default)]
    resources: Vec<ResourceSpec>,
    #[serde(default)]
    programs: Vec<ProgramSpec>,
    passes: Vec<PassSpec>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResourceSpec {
    name: String,
    kind: ResourceKind,
    #[serde(default)]
    when: Option<When>,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    usage: Vec<String>,
    #[serde(default)]
    class: Option<ImageClass>,
    #[serde(default)]
    scale: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResourceKind {
    Library,
    Image,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ImageClass {
    Transient,
    Persistent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum When {
    Units,
    NoUnits,
    Readback,
    Present,
    Water,
    NoWater,
    WaterReflect,
    WaterReflectUnits,
    WaterRefract,
    WaterRefractUnits,
    ParticlesRefract,
    ParticlesWaterSplit,
    ParticlesNoRefract,
    WaterWaves,
    WaterVolume,
    Shadows,
    NoShadows,
    Particles,
    NoParticles,
    /// The particles cast coloured shadows: units that cast shadows and
    /// particles.
    ParticleShadows,
    /// The output is HDR (HDR10 or scRGB): the graph keeps the display
    /// intermediate and ends in `post.encode`.
    Hdr,
    /// The output is SDR: `post.tonemap` writes the target, no display and
    /// no encode.
    Sdr,
}

impl When {
    fn holds(when: Option<When>, flags: u32, hdr: bool) -> bool {
        let (units, readback, water, shadows) = (
            flags & DECLARE_UNITS != 0,
            flags & DECLARE_READBACK != 0,
            flags & DECLARE_WATER != 0,
            flags & DECLARE_SHADOWS != 0,
        );
        let refract = water && flags & DECLARE_WATER_REFRACT != 0;
        let particles = flags & DECLARE_PARTICLES != 0;
        match when {
            None => true,
            Some(When::Units) => units,
            Some(When::NoUnits) => !units,
            Some(When::Readback) => readback,
            Some(When::Present) => !readback,
            Some(When::Water) => water,
            Some(When::NoWater) => !water,
            Some(When::WaterReflect) => water && flags & DECLARE_WATER_REFLECT != 0,
            Some(When::WaterReflectUnits) => units && Self::holds(Some(When::WaterReflect), flags, hdr),
            Some(When::WaterRefract) => refract,
            Some(When::WaterRefractUnits) => refract && units,
            Some(When::ParticlesRefract) => particles && refract,
            Some(When::ParticlesWaterSplit) => particles && refract && flags & DECLARE_WATER_WAVES != 0,
            Some(When::ParticlesNoRefract) => particles && !refract,
            Some(When::WaterWaves) => water && flags & DECLARE_WATER_WAVES != 0,
            Some(When::WaterVolume) => {
                water
                    && flags & (DECLARE_WATER_REFRACT | DECLARE_WATER_VOLUME)
                        == DECLARE_WATER_REFRACT | DECLARE_WATER_VOLUME
            }
            Some(When::Shadows) => shadows,
            Some(When::NoShadows) => !shadows,
            Some(When::Particles) => flags & DECLARE_PARTICLES != 0,
            Some(When::NoParticles) => flags & DECLARE_PARTICLES == 0,
            Some(When::ParticleShadows) => shadows && flags & DECLARE_PARTICLES != 0,
            Some(When::Hdr) => hdr,
            Some(When::Sdr) => !hdr,
        }
    }
}

/// Every scene the host can request: shadows require units, water tiers
/// require a wet scene, and underwater volume requires refraction.
pub const SCENE_FLAGS: [u32; 156] = {
    // Twelve dry combinations and twelve wet ones, with twelve valid tier
    // combinations for each wet scene. The four ABI tier bits are contiguous.
    let mut flags = [0; 156];
    let mut index = 0;
    let mut base = 0;
    while base < SCENE_FLAGS_BASE.len() {
        let mut particles = 0;
        while particles <= DECLARE_PARTICLES {
            let scene = SCENE_FLAGS_BASE[base] | particles;
            let mut tiers = 0;
            while tiers < DECLARE_WATER_VOLUME * 2 {
                if (tiers == 0 || scene & DECLARE_WATER != 0)
                    && (tiers & DECLARE_WATER_VOLUME == 0 || tiers & DECLARE_WATER_REFRACT != 0)
                {
                    flags[index] = scene | tiers;
                    index += 1;
                }
                tiers += DECLARE_WATER_REFLECT;
            }
            particles += DECLARE_PARTICLES;
        }
        base += 1;
    }
    assert!(index == flags.len());
    flags
};

const SCENE_FLAGS_BASE: [u32; 12] = [
    0,
    DECLARE_UNITS,
    DECLARE_READBACK,
    DECLARE_UNITS | DECLARE_READBACK,
    DECLARE_WATER,
    DECLARE_UNITS | DECLARE_WATER,
    DECLARE_READBACK | DECLARE_WATER,
    DECLARE_UNITS | DECLARE_READBACK | DECLARE_WATER,
    DECLARE_UNITS | DECLARE_SHADOWS,
    DECLARE_UNITS | DECLARE_READBACK | DECLARE_SHADOWS,
    DECLARE_UNITS | DECLARE_WATER | DECLARE_SHADOWS,
    DECLARE_UNITS | DECLARE_READBACK | DECLARE_WATER | DECLARE_SHADOWS,
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProgramSpec {
    name: String,
    #[serde(default)]
    fragment: Option<String>,
    #[serde(default)]
    fragment_hex: Option<String>,
    #[serde(default)]
    uniforms: Vec<f32>,
    /// When set, the program's uniform block switches with this flag bit:
    /// `uniforms` while it is set, `uniforms_off` while it is clear. A pass
    /// that always runs but whose effect is switched on may need this (the
    /// HDR vignette multiplies by zero when off, the pass itself staying in
    /// the graph so the encode always reads a written image).
    #[serde(default)]
    uniforms_flag: Option<u32>,
    #[serde(default)]
    uniforms_off: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PassSpec {
    name: String,
    kind: PassKindSpec,
    #[serde(default)]
    library: Option<String>,
    #[serde(default)]
    program: Option<String>,
    #[serde(default)]
    access: Vec<String>,
    #[serde(default)]
    when: Option<When>,
    #[serde(default)]
    flag: Option<u32>,
    #[serde(default)]
    root: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PassKindSpec {
    Graphics,
    Compute,
    Transfer,
}

/// Per program-table entry: what `frame` writes and disables.
#[derive(Debug, Default)]
struct Entry {
    /// The pass's conditional bit: disabled while it is clear.
    flag: Option<u32>,
    /// The uniform block while the bit is set (or no bit is involved).
    uniforms: Vec<u8>,
    /// When the block switches with a bit: `(bit, block while clear)`.
    uniforms_off: Option<(u32, Vec<u8>)>,
}

/// A parsed, checked graph file (its programs' SPIR-V in memory).
pub struct GraphFile {
    spec: FileSpec,
    /// By program name: SPIR-V.
    spirv: BTreeMap<String, Vec<u8>>,
    /// From the last `declare`: per program-table entry, its uniform bytes
    /// and flag bits.
    entries: Vec<Entry>,
    pub loaded: std::time::Instant,
}

fn hex_decode(text: &str) -> Result<Vec<u8>> {
    ensure!(text.len().is_multiple_of(2), "odd number of hex digits");
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).with_context(|| format!("hex digits {:?}", &text[i..i + 2])))
        .collect()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Vulkan format by name (the ones render graphs use), `target` (the
/// output's), `display` (the HDR display intermediate's,
/// [`ae_render_abi::module::DISPLAY_FORMAT`]), `scene` (the HDR scene's,
/// [`ae_render_abi::module::SCENE_FORMAT`]), `glow` or `bloom`.
fn format(name: &str, target: u32) -> Result<u32> {
    Ok(match name {
        "target" => target,
        "display" => ae_render_abi::module::DISPLAY_FORMAT,
        "scene" => ae_render_abi::module::SCENE_FORMAT,
        "glow" => ae_render_abi::module::GLOW_FORMAT,
        "bloom" => ae_render_abi::module::BLOOM_FORMAT,
        "R8G8B8A8_UNORM" => 37,
        "R8G8B8A8_SRGB" => 43,
        "B8G8R8A8_UNORM" => 44,
        "B8G8R8A8_SRGB" => 50,
        "R16G16B16A16_SFLOAT" => 97,
        "B10G11R11_UFLOAT_PACK32" => 122,
        "R32_SFLOAT" => 100,
        "R32G32B32A32_SFLOAT" => 109,
        "D32_SFLOAT" => 126,
        "D24_UNORM_S8_UINT" => 129,
        "D32_SFLOAT_S8_UINT" => 130,
        other => bail!("unknown format {other:?}"),
    })
}

/// Image aspect bits of a format: depth, depth + stencil, or colour.
fn aspect(format: u32) -> u32 {
    match format {
        126 => 2,
        129 | 130 => 2 | 4,
        _ => 1,
    }
}

fn usage(names: &[String]) -> Result<u32> {
    names.iter().try_fold(0, |bits, name| {
        Ok(bits
            | match name.as_str() {
                "transfer_src" => 0x1,
                "transfer_dst" => 0x2,
                "sampled" => 0x4,
                "storage" => 0x8,
                "color_attachment" => 0x10,
                "depth_stencil_attachment" => 0x20,
                other => bail!("unknown usage {other:?}"),
            })
    })
}

fn stage(name: &str) -> Result<ShaderStage> {
    Ok(match name {
        "vertex" => ShaderStage::Vertex,
        "fragment" => ShaderStage::Fragment,
        "graphics" => ShaderStage::Graphics,
        "compute" => ShaderStage::Compute,
        other => bail!("unknown shader stage {other:?}"),
    })
}

fn load(name: &str) -> Result<LoadOp> {
    Ok(match name {
        "clear" => LoadOp::Clear,
        "load" => LoadOp::Load,
        "dont_care" => LoadOp::DontCare,
        other => bail!("unknown load op {other:?}"),
    })
}

fn role(text: &str) -> Result<Role> {
    let (name, arg) = match text.split_once(':') {
        Some((n, a)) => (n, Some(a)),
        None => (text, None),
    };
    let needs = |what: &str| arg.with_context(|| format!("role `{name}` needs `:{what}`"));
    let none = || -> Result<()> {
        ensure!(arg.is_none(), "role `{name}` takes no argument");
        Ok(())
    };
    Ok(match name {
        "color" => Role::ColorAttachment(load(needs("load op")?)?),
        "depth" => Role::DepthAttachment(load(needs("load op")?)?),
        "depth_read" => none().map(|()| Role::DepthRead)?,
        "sampled" => Role::Sampled(stage(needs("stage")?)?),
        "storage_image_read" => Role::StorageImageRead(stage(needs("stage")?)?),
        "storage_image_write" => Role::StorageImageWrite(stage(needs("stage")?)?),
        "storage_image_read_write" => Role::StorageImageReadWrite(stage(needs("stage")?)?),
        "storage_buffer_read" => Role::StorageBufferRead(stage(needs("stage")?)?),
        "storage_buffer_write" => Role::StorageBufferWrite(stage(needs("stage")?)?),
        "storage_buffer_read_write" => Role::StorageBufferReadWrite(stage(needs("stage")?)?),
        "uniform" => Role::UniformBuffer(stage(needs("stage")?)?),
        "vertex_buffer" => none().map(|()| Role::VertexBuffer)?,
        "index_buffer" => none().map(|()| Role::IndexBuffer)?,
        "indirect" => none().map(|()| Role::IndirectBuffer)?,
        "transfer_src" => none().map(|()| Role::TransferSrc)?,
        "transfer_dst" => none().map(|()| Role::TransferDst)?,
        "host_read" => none().map(|()| Role::HostRead)?,
        "present" => bail!("`present` is added by the host after the last pass (declare an `output` resource)"),
        other => bail!("unknown role {other:?}"),
    })
}

/// `"<resource> <role>[ discard]"`. Names separated by `|` select the first active resource.
fn access(text: &str) -> Result<(&str, Role, bool)> {
    let words: Vec<&str> = text.split_whitespace().collect();
    match words.as_slice() {
        [resource, r] => Ok((resource, role(r)?, false)),
        [resource, r, "discard"] => Ok((resource, role(r)?, true)),
        _ => bail!("access {text:?}: expected \"<resource> <role>\" or \"<resource> <role> discard\""),
    }
}

impl GraphFile {
    /// Parse and check a graph file whose programs are inline
    /// (`fragment_hex`), as [`inline`] writes it.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let spec: FileSpec = serde_json::from_slice(bytes).context("parsing the graph file")?;
        ensure!(
            spec.version == FILE_VERSION,
            "graph file version {}; this host reads {FILE_VERSION}",
            spec.version
        );
        let mut spirv = BTreeMap::new();
        for p in &spec.programs {
            if let Some(path) = &p.fragment {
                bail!(
                    "program `{}`: `fragment` {path:?} is not inlined (load the file through `graph_file::inline` or `module::build`)",
                    p.name
                );
            }
            let hex = p
                .fragment_hex
                .as_deref()
                .with_context(|| format!("program `{}` has no `fragment`", p.name))?;
            let bytes = hex_decode(hex).with_context(|| format!("program `{}`", p.name))?;
            ensure!(
                !bytes.is_empty() && bytes.len().is_multiple_of(4),
                "program `{}`: SPIR-V of {} bytes",
                p.name,
                bytes.len()
            );
            ensure!(
                spirv.insert(p.name.clone(), bytes).is_none(),
                "program `{}` is declared twice",
                p.name
            );
        }
        let mut names = std::collections::BTreeSet::new();
        for r in &spec.resources {
            ensure!(names.insert(r.name.as_str()), "resource `{}` is declared twice", r.name);
        }
        ensure!(
            spec.resources.iter().filter(|r| r.kind == ResourceKind::Output).count() == 1,
            "a graph file declares exactly one `output` resource"
        );
        let file = Self {
            spec,
            spirv,
            entries: Vec::new(),
            loaded: std::time::Instant::now(),
        };
        // Every scene and output encoding the file can be asked for must
        // declare (a graph file is checked before its first frame, not
        // discovered per frame).
        for encoding in [OutputEncoding::Sdr, OutputEncoding::Hdr10Pq, OutputEncoding::ScRgb] {
            for flags in SCENE_FLAGS {
                let target = DeclareTarget {
                    width: 64,
                    height: 64,
                    color_format: vk::Format::B8G8R8A8_SRGB,
                    flags,
                    encoding,
                    levels: DisplayLevels::default(),
                    working: WorkingSpace::Bt709,
                    gamut: Gamut::Bt709,
                };
                file.build(&target).with_context(|| {
                    format!(
                        "declaring for {encoding:?}, {}units, {}water, {}shadows, {}particles, {}, flags {flags:#x}",
                        if flags & DECLARE_UNITS != 0 { "" } else { "no " },
                        if flags & DECLARE_WATER != 0 { "" } else { "no " },
                        if flags & DECLARE_SHADOWS != 0 { "" } else { "no " },
                        if flags & DECLARE_PARTICLES != 0 { "" } else { "no " },
                        if flags & DECLARE_READBACK != 0 {
                            "readback"
                        } else {
                            "present"
                        }
                    )
                })?;
            }
        }
        Ok(file)
    }

    fn build(&self, target: &DeclareTarget) -> Result<(Declared, Vec<Entry>)> {
        let (width, height, color_format, flags) = (
            target.width,
            target.height,
            target.color_format.as_raw() as u32,
            target.flags,
        );
        let readback = flags & DECLARE_READBACK != 0;
        let hdr = target.encoding.uses_display();
        let mut g = GraphBuilder::new();
        let mut table = ProgramTable::new();
        let mut entries = Vec::new();
        let mut refs: BTreeMap<&str, ResourceRef> = BTreeMap::new();
        let mut output = None;
        for r in self.spec.resources.iter().filter(|r| When::holds(r.when, flags, hdr)) {
            let at = || format!("resource `{}`", r.name);
            let reference = match r.kind {
                ResourceKind::Library => {
                    ensure!(
                        r.format.is_none() && r.usage.is_empty() && r.class.is_none() && r.scale.is_none(),
                        "{}: a library resource takes no format, usage, class or scale",
                        at()
                    );
                    if r.name == ae_render_abi::module::SHADOW_MAP_NAME {
                        g.image(&r.name, ae_render_abi::module::shadow_map_spec(), Class::External)
                    } else if r.name == ae_render_abi::module::PARTICLE_SHADOW_NAME {
                        g.image(&r.name, ae_render_abi::module::particle_shadow_spec(), Class::External)
                    } else if r.name == ae_render_abi::module::EXPOSURE_NAME {
                        g.image(&r.name, ae_render_abi::module::exposure_spec(), Class::External)
                    } else {
                        let lib = crate::library::LibraryResource::from_name(&r.name);
                        let class = if lib.is_some_and(|r| r.is_transient()) {
                            Class::Transient
                        } else {
                            Class::External
                        };
                        g.buffer(&r.name, 0, 0, class)
                    }
                }
                ResourceKind::Image => {
                    let fmt = format(
                        r.format
                            .as_deref()
                            .with_context(|| format!("{} needs a format", at()))?,
                        color_format,
                    )
                    .with_context(at)?;
                    let scale = r.scale.unwrap_or(1.0);
                    ensure!(scale > 0.0 && scale <= 4.0, "{}: scale {scale} (0 < scale <= 4)", at());
                    let size = |v: u32| ((v as f32 * scale).round() as u32).max(1);
                    let class = match r.class.unwrap_or(ImageClass::Transient) {
                        ImageClass::Transient => Class::Transient,
                        ImageClass::Persistent => Class::Persistent,
                    };
                    let spec = ImageSpec {
                        aspect: aspect(fmt),
                        ..ImageSpec::color_2d(fmt, size(width), size(height), usage(&r.usage).with_context(at)?)
                    };
                    g.image(&r.name, spec, class)
                }
                ResourceKind::Output => {
                    ensure!(
                        r.format.is_none() && r.class.is_none() && r.scale.is_none(),
                        "{}: the output has the target's format and size",
                        at()
                    );
                    let extra = usage(&r.usage).with_context(at)?;
                    let reference = if readback {
                        g.image(
                            "readback.target",
                            ImageSpec::color_2d(color_format, width, height, 0x10 | 0x1 | extra),
                            Class::External,
                        )
                    } else {
                        g.image(
                            "target",
                            ImageSpec::color_2d(color_format, width, height, 0x10 | extra),
                            Class::Backbuffer,
                        )
                    };
                    output = Some(reference);
                    reference
                }
            };
            refs.insert(&r.name, reference);
        }
        let output = output.context("the `output` resource has a `when` that excludes this scene")?;
        for p in self.spec.passes.iter().filter(|p| When::holds(p.when, flags, hdr)) {
            let at = || format!("pass `{}`", p.name);
            let payload = match (&p.library, &p.program) {
                (Some(name), None) => {
                    entries.push(Entry {
                        flag: p.flag,
                        ..Entry::default()
                    });
                    table.library(name)
                }
                (None, Some(name)) => {
                    let spec = self
                        .spec
                        .programs
                        .iter()
                        .find(|q| &q.name == name)
                        .with_context(|| format!("{}: no program `{name}`", at()))?;
                    let block = |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
                    let uniforms = block(&spec.uniforms);
                    let uniforms_off = match spec.uniforms_flag {
                        None => None,
                        Some(bit) => {
                            ensure!(bit < 32, "{}: uniforms_flag bit {bit} (0..32)", at());
                            ensure!(
                                spec.uniforms_off.len() == spec.uniforms.len(),
                                "{}: uniforms_off has {} floats; uniforms has {}",
                                at(),
                                spec.uniforms_off.len(),
                                spec.uniforms.len()
                            );
                            Some((bit, block(&spec.uniforms_off)))
                        }
                    };
                    let index = table.push(Program::Fullscreen {
                        label: &spec.name,
                        spirv: &self.spirv[&spec.name],
                        uniform_bytes: uniforms.len() as u32,
                    });
                    entries.push(Entry {
                        flag: p.flag,
                        uniforms,
                        uniforms_off,
                    });
                    index
                }
                _ => bail!("{}: give exactly one of `library` and `program`", at()),
            };
            let kind = match p.kind {
                PassKindSpec::Graphics => PassKind::Graphics,
                PassKindSpec::Compute => PassKind::Compute,
                PassKindSpec::Transfer => PassKind::Transfer,
            };
            let mut w = g.pass(&p.name, kind, payload);
            for a in &p.access {
                let (resource, role, discard) = access(a).with_context(at)?;
                let declared = |name: &str| self.spec.resources.iter().any(|r| r.name == name);
                let reference = if declared(resource) {
                    refs.get(resource).copied()
                } else {
                    let mut first = None;
                    for name in resource.split('|') {
                        ensure!(declared(name), "{}: no resource `{name}`", at());
                        first = first.or_else(|| refs.get(name).copied());
                    }
                    first
                };
                // Declared, but not in this scene: the access goes with it.
                let Some(reference) = reference else { continue };

                w = if discard {
                    w.discard(reference, role)
                } else {
                    w.access(reference, role)
                };
            }
            if let Some(bit) = p.flag {
                ensure!(bit < 32, "{}: flag bit {bit} (0..32)", at());
                w = w.conditional();
            }
            if p.root {
                w = w.root();
            }
            w.id();
        }
        // The tail every module writes: present, or copy out and read.
        if readback {
            let buffer = g.buffer("readback.buffer", 0, 0, Class::External);
            entries.push(Entry::default());
            let copy = table.library("readback.copy");
            g.pass("copy", PassKind::Transfer, copy)
                .access(output, Role::TransferSrc)
                .discard(buffer, Role::TransferDst)
                .id();
            entries.push(Entry::default());
            let read = table.library("host.read");
            g.pass("host read", PassKind::Transfer, read)
                .access(buffer, Role::HostRead)
                .root()
                .id();
        } else {
            entries.push(Entry::default());
            let present = table.library("present");
            g.pass("present", PassKind::Present, present)
                .present(output)
                .root()
                .id();
        }
        Ok((
            Declared {
                graph: g.finish(),
                programs: table.finish(),
            },
            entries,
        ))
    }

    /// The graph for a target (as a module's `declare`).
    pub fn declare(&mut self, target: &DeclareTarget) -> Result<Option<Declared>> {
        let (declared, entries) = self.build(target)?;
        self.entries = entries;
        Ok(Some(declared))
    }

    /// The uniforms (switched by their `uniforms_flag` bit where they have
    /// one); conditional passes disabled while their flag bit is clear.
    pub fn frame(&mut self, input: &FrameInput) -> ModuleFrame {
        let mut frame = ModuleFrame::default();
        for (index, entry) in self.entries.iter().enumerate() {
            let uniforms = match &entry.uniforms_off {
                Some((bit, off)) if input.flags & (1 << bit) == 0 => off,
                _ => &entry.uniforms,
            };
            frame.uniforms.extend_from_slice(uniforms);
            if entry.flag.is_some_and(|bit| input.flags & (1 << bit) == 0) {
                let word = index / 32;
                if frame.disabled.len() <= word {
                    frame.disabled.resize(word + 1, 0);
                }
                frame.disabled[word] |= 1 << (index % 32);
            }
        }
        frame
    }
}

/// Read a graph file and make it self-contained: each program's `fragment`
/// (a `.spv` file, or a `.frag` compiled with `glslc`, `GLSLC` overriding;
/// paths relative to the file) becomes `fragment_hex`. The result is what
/// [`GraphFile::parse`] (and so `RenderModule::load`) takes.
pub fn inline(path: &Path) -> Result<Vec<u8>> {
    let text = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut value: serde_json::Value =
        serde_json::from_slice(&text).with_context(|| format!("parse {}", path.display()))?;
    let dir = path.parent().unwrap_or(Path::new("."));
    if let Some(programs) = value.get_mut("programs").and_then(|p| p.as_array_mut()) {
        for program in programs {
            let Some(object) = program.as_object_mut() else {
                continue;
            };
            let Some(fragment) = object.remove("fragment") else {
                continue;
            };
            let fragment = fragment.as_str().context("`fragment` is a path")?.to_string();
            let file = dir.join(&fragment);
            let spirv = if fragment.ends_with(".spv") {
                std::fs::read(&file).with_context(|| format!("read {}", file.display()))?
            } else {
                compile(&file)?
            };
            object.insert("fragment_hex".into(), serde_json::Value::String(hex_encode(&spirv)));
        }
    }
    Ok(serde_json::to_vec(&value)?)
}

/// `glslc` a fragment shader as the in-repo modules' build scripts do, with
/// the engine's program interface headers (`bindless.glsl`,
/// `program.glsl`) on the include path.
fn compile(file: &Path) -> Result<Vec<u8>> {
    let glslc = std::env::var("GLSLC").unwrap_or_else(|_| "glslc".into());
    let include = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let output = std::process::Command::new(&glslc)
        .args(["--target-env=vulkan1.3", "-O", "-Werror", "-I"])
        .arg(&include)
        .arg(file)
        .args(["-o", "-"])
        .output()
        .with_context(|| format!("running {glslc}"))?;
    ensure!(
        output.status.success(),
        "{glslc} failed for {}:\n{}",
        file.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(passes: &str) -> String {
        format!(
            r#"{{"version": 1, "resources": [{{"name": "terrain.leaves", "kind": "library"}}, {{"name": "terrain.indirect", "kind": "library"}}, {{"name": "target", "kind": "output"}}], "passes": [{passes}]}}"#
        )
    }

    const LOD: &str = r#"{"name": "terrain lod", "kind": "compute", "library": "terrain.lod", "access": ["terrain.leaves storage_buffer_write:compute", "terrain.indirect storage_buffer_write:compute"]}"#;

    #[test]
    fn roles_and_accesses_parse() {
        assert_eq!(role("color:clear").unwrap(), Role::ColorAttachment(LoadOp::Clear));
        assert_eq!(
            role("storage_buffer_read:vertex").unwrap(),
            Role::StorageBufferRead(ShaderStage::Vertex)
        );
        assert_eq!(
            access("units.counts transfer_dst discard").unwrap(),
            ("units.counts", Role::TransferDst, true)
        );
        for bad in ["color", "indirect:vertex", "sampled:pixel", "present", "blend:add"] {
            assert!(role(bad).is_err(), "{bad}");
        }
        assert!(access("one").is_err() && access("a indirect maybe").is_err());
    }

    #[test]
    fn water_tier_keys_select_only_the_requested_wet_graph_resources_and_passes() {
        let tiers = ["water_reflect", "water_refract", "water_waves", "water_volume"];
        let text = serde_json::json!({
            "version": 1,
            "resources": std::iter::once(serde_json::json!({"name": "target", "kind": "output"}))
                .chain(tiers.map(|key| serde_json::json!({"name": key, "kind": "library", "when": key})))
                .collect::<Vec<_>>(),
            "passes": tiers.map(|key| serde_json::json!({
                "name": key, "kind": "transfer", "library": "test", "when": key,
                "access": [format!("{key} transfer_dst")],
            })),
        });
        let mut file = GraphFile::parse(&serde_json::to_vec(&text).unwrap()).unwrap();
        for (flags, want) in [
            (DECLARE_WATER, vec![]),
            (DECLARE_WATER | DECLARE_WATER_REFLECT, vec!["water_reflect"]),
            (DECLARE_WATER | DECLARE_WATER_REFRACT, vec!["water_refract"]),
            (DECLARE_WATER | DECLARE_WATER_WAVES, vec!["water_waves"]),
            (DECLARE_WATER | DECLARE_WATER_VOLUME, vec![]),
            (
                DECLARE_WATER | DECLARE_WATER_REFRACT | DECLARE_WATER_VOLUME,
                vec!["water_refract", "water_volume"],
            ),
            (
                DECLARE_WATER_REFLECT | DECLARE_WATER_REFRACT | DECLARE_WATER_WAVES | DECLARE_WATER_VOLUME,
                vec![],
            ),
        ] {
            let target = DeclareTarget {
                width: 64,
                height: 64,
                color_format: vk::Format::B8G8R8A8_SRGB,
                flags,
                encoding: OutputEncoding::Sdr,
                levels: DisplayLevels::default(),
                working: WorkingSpace::Bt709,
                gamut: Gamut::Bt709,
            };
            let declared = file.declare(&target).unwrap().unwrap();
            let graph = ae_framegraph::descriptor::decode(&declared.graph).unwrap();
            let resources: Vec<_> = graph
                .resources
                .iter()
                .filter(|r| r.name != "target")
                .map(|r| r.name.as_str())
                .collect();
            let passes: Vec<_> = graph
                .passes
                .iter()
                .filter(|p| p.name != "present")
                .map(|p| p.name.as_str())
                .collect();
            assert_eq!(resources, want, "{flags:#x}");
            assert_eq!(passes, want, "{flags:#x}");
        }
    }

    /// M16.5: a graph file names the water data as library resources (it
    /// cannot know their format or size); `resolve` fills in the host's
    /// images — the shore field's and the caustic web's descriptors — and
    /// the shore bake pass passes the library's checks.
    #[test]
    fn library_images_get_their_host_descriptors() {
        use crate::library::{LibraryEnv, LibraryPass, check, resolve};
        let text = r#"{"version": 1, "resources": [
            {"name": "water.shore", "kind": "library"},
            {"name": "water.caustics", "kind": "library"},
            {"name": "target", "kind": "output"}],
          "passes": [
            {"name": "shore", "kind": "compute", "library": "water.shore",
             "access": ["water.shore storage_image_write:compute"], "root": true}]}"#;
        let mut file = GraphFile::parse(text.as_bytes()).expect("the graph file parses");
        let target = DeclareTarget {
            width: 64,
            height: 64,
            color_format: vk::Format::B8G8R8A8_SRGB,
            flags: ae_render_abi::module::DECLARE_READBACK | ae_render_abi::module::DECLARE_WATER,
            encoding: OutputEncoding::Sdr,
            levels: DisplayLevels::default(),
            working: WorkingSpace::Bt709,
            gamut: Gamut::Bt709,
        };
        let declared = file.declare(&target).unwrap().expect("the file declares");
        let mut graph = ae_framegraph::descriptor::decode(&declared.graph).expect("the descriptor decodes");
        let env = LibraryEnv {
            color_format: vk::Format::B8G8R8A8_SRGB,
            scene_format: crate::terrain::app::SCENE_FORMAT,
            depth_format: vk::Format::D32_SFLOAT,
            extent: vk::Extent2D { width: 64, height: 64 },
            units: false,
            water: true,
            shadows: false,
            particles: false,
            ground: Some((257, 257)),
            bake_water: false,
            encoding: OutputEncoding::Sdr,
            levels: DisplayLevels::default(),
            working: WorkingSpace::Bt709,
            gamut: Gamut::Bt709,
        };
        let map = resolve(&mut graph, &env, |_| {
            Some(ae_framegraph::BufferDesc {
                size: 64,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            })
        })
        .expect("the library resolves");
        let desc = |name: &str| {
            graph
                .resources
                .iter()
                .find(|r| r.name == name)
                .unwrap_or_else(|| panic!("no resource {name}"))
                .desc
        };
        let shore = desc("water.shore");
        let ae_framegraph::ResourceDesc::Image(shore) = shore else {
            panic!("water.shore is a buffer: {shore:?}");
        };
        assert_eq!(
            (shore.format, shore.width, shore.height),
            (crate::water::shore::FIELD_FORMAT, 257, 257),
            "the shore field's host descriptor"
        );
        let caustics = desc("water.caustics");
        let ae_framegraph::ResourceDesc::Image(caustics) = caustics else {
            panic!("water.caustics is a buffer: {caustics:?}");
        };
        assert_eq!(
            (caustics.format, caustics.width, caustics.height, caustics.mip_levels),
            (crate::water::caustics::FORMAT, 256, 256, 9),
            "the caustic web's host descriptor"
        );
        let index = graph
            .passes
            .iter()
            .position(|p| p.name == "shore")
            .expect("the shore pass");
        check(&graph, index, LibraryPass::WaterShore, &map, &env).expect("the shore pass checks");
    }

    #[test]
    fn errors_name_what_is_wrong() {
        let err = |text: String| {
            GraphFile::parse(text.as_bytes())
                .err()
                .map(|e| format!("{e:#}"))
                .unwrap_or_default()
        };
        assert!(err(file(LOD)).is_empty(), "{}", err(file(LOD)));
        let e = err(file(
            &LOD.replace("terrain.indirect storage", "terrain.indirekt storage"),
        ));
        assert!(
            e.contains("pass `terrain lod`") && e.contains("no resource `terrain.indirekt`"),
            "{e}"
        );
        let e = err(file(
            &LOD.replace(r#""library": "terrain.lod""#, r#""program": "nope""#),
        ));
        assert!(e.contains("no program `nope`"), "{e}");
        let e = err(file(&LOD.replace("compute\", \"library", "computer\", \"library")));
        assert!(e.contains("parsing"), "{e}");
        let e = err(file(LOD).replace("\"version\": 1", "\"version\": 2"));
        assert!(e.contains("version 2"), "{e}");
        let e = err(file(LOD).replace(
            r#"{"name": "target", "kind": "output"}"#,
            r#"{"name": "target", "kind": "library"}"#,
        ));
        assert!(e.contains("exactly one `output`"), "{e}");
    }

    #[test]
    fn conditional_passes_follow_their_flag() {
        let text = file(&format!(
            r#"{LOD}, {{"name": "extra", "kind": "compute", "library": "terrain.lod", "flag": 3, "access": ["terrain.leaves storage_buffer_write:compute"], "root": true}}"#
        ));
        let mut f = GraphFile::parse(text.as_bytes()).unwrap();
        let target = DeclareTarget {
            width: 64,
            height: 64,
            color_format: vk::Format::B8G8R8A8_SRGB,
            flags: 0,
            encoding: OutputEncoding::Sdr,
            levels: DisplayLevels::default(),
            working: WorkingSpace::Bt709,
            gamut: Gamut::Bt709,
        };
        f.declare(&target).unwrap();
        let input = |flags| FrameInput {
            flags,
            ..FrameInput::default()
        };
        assert!(f.frame(&input(0)).is_disabled(1) && !f.frame(&input(0)).is_disabled(0));
        assert!(!f.frame(&input(1 << 3)).is_disabled(1));
    }
}
