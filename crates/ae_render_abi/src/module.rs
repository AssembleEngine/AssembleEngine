//! The render module contract: what a render module (a WASM module with
//! its own memory, no imports) exports, and the program table that tells
//! the host what records each pass.
//!
//! # Exports
//!
//! | Export | Signature | Called |
//! |---|---|---|
//! | `memory` | memory | |
//! | [`EXPORT_VERSION`] | `() -> u32` ([`MODULE_VERSION`]) | at load |
//! | [`EXPORT_DECLARE_INPUT`] | `() -> u32` | at load: where the host writes [`DeclareInput`] |
//! | [`EXPORT_DECLARE`] | `() -> u32` | at load, on reload, when the target changes |
//! | [`EXPORT_INPUT`] | `() -> u32` | at load: where the host writes [`FrameInput`] |
//! | [`EXPORT_FRAME`] | `() -> u32` | once per frame, after the host wrote the input |
//!
//! The host writes [`DeclareInput`] first: the target's size and colour
//! format, the scene's flags, and the output's display spec (the encoding,
//! the levels in signal nits, the working space and the output gamut).
//! `declare` then returns the address of four `u32`s: graph pointer and
//! length (the frame graph declaration, [`crate::GraphBuilder`]), program
//! table pointer and length ([`ProgramTable`]); 0 if the module cannot
//! render that target. [`DeclareInput::flags`] holds [`DECLARE_UNITS`],
//! [`DECLARE_READBACK`], [`DECLARE_WATER`], [`DECLARE_SHADOWS`] and
//! [`DECLARE_PARTICLES`] (the last three added later; a module that ignores
//! them simply draws no water, no unit shadows, or no particles).
//!
//! `frame` returns the address of [`FrameOutput`] as four `u32`s: pointer
//! and word count of a bitset of *disabled* programs (a conditional pass
//! whose payload's bit is set is skipped), pointer and length of the
//! uniform bytes: every fullscreen program's block, in program order, each
//! [`Program::Fullscreen::uniform_bytes`] long.
//!
//! # The display rule (contract version 2)
//!
//! When [`DeclareInput::encoding`] is not [`OutputEncoding::Sdr`], the
//! graph keeps a display intermediate and ends in a `post.encode` pass:
//! `post.tonemap` writes the display ([`display_spec`], RGBA16F linear
//! light in units of paper white, in the working space), every pass that
//! changes colour (a module's grade, a vignette) runs on it, and only
//! `post.encode` writes the output. In SDR there is no display image and no
//! encode: `post.tonemap` writes the output itself. `declare` must honour
//! exactly the encoding it is given; the host checks the rule
//! (`ae_render::library::check_graph`) and refuses a graph with a missing
//! or extra `post.encode`.
//!
//! # Program table (version 1)
//!
//! Little-endian `u32` words: header `magic, version, count, blob bytes`,
//! then per program `kind, name offset, name length, data offset, data
//! length, uniform bytes`, then the blob. A pass's payload indexes it.
//! `kind` 0 is a library pass (name = its library name, no data); 1 is a
//! fullscreen pass (name = a label, data = fragment shader SPIR-V;
//! `uniform bytes` = the size of its std140 block, 0 for none).
//!
//! A fullscreen shader has one of two forms, and the host reflects which
//! (they cannot be mixed):
//!
//! * **Per-pass** (contract version 2 only): set 0 binding 0 is the
//!   uniform block, bindings 1.. are one combined image sampler per
//!   `Sampled` intent, in intent order.
//! * **Bindless** (contract version 3): only the global set. Binding 0 is
//!   `texture2D images[]`, binding 1 `sampler samplers[]`, binding 3 the
//!   program's handle table (`uint handles[]`) and binding 4 its `std140`
//!   block (`uniform` blocks are uniform buffers, not storage buffers: the
//!   driver's uniform path is cheaper). A push block of at most 16 bytes
//!   carries `uniforms` (the block's index in the array), `sampler` (the
//!   sampler's index), `handles` (the table's index) and a pad word. The
//!   table holds the pass's sampled-image indices in intent order, written
//!   by the host once per bake: the indices are the same in every frame
//!   slot.
//!
//! The next contract version drops the per-pass form.

pub const EXPORT_VERSION: &str = "ae_render_version";
pub const EXPORT_DECLARE_INPUT: &str = "ae_render_declare_input";
pub const EXPORT_DECLARE: &str = "ae_render_declare";
pub const EXPORT_INPUT: &str = "ae_render_input";
pub const EXPORT_FRAME: &str = "ae_render_frame";
/// The contract version: 2 adds [`DeclareInput`] (the display spec) and the
/// display rule; 3 moves a fullscreen program's resources into the global
/// bindless set (the program table is unchanged; the shader's form is
/// reflected, and either form is accepted for version 3 only). A version 1
/// module (its `declare` took the four scalars) is refused at load, never
/// called with the new signature.
pub const MODULE_VERSION: u32 = 3;

/// `declare` flag: the scene has units (`units.*` resources and passes exist).
pub const DECLARE_UNITS: u32 = 1 << 0;
/// `declare` flag: render to `readback.target` and copy it out, instead of
/// presenting a `Backbuffer`.
pub const DECLARE_READBACK: u32 = 1 << 1;
/// `declare` flag: the scene draws water (the `water.draw` library pass
/// exists).
pub const DECLARE_WATER: u32 = 1 << 2;
/// `declare` flag: the units cast shadows (set only with
/// [`DECLARE_UNITS`]): the `shadows.draw` library pass and the
/// `shadows.map` library image exist. A graph that declares `shadows.map`
/// has `shadows.draw` write it and every `terrain.draw`, `units.draw` and
/// `water.draw` pass sample it (`sampled:fragment`); a graph that does not
/// simply renders without the units' shadows.
pub const DECLARE_SHADOWS: u32 = 1 << 3;
/// `declare` flag: the scene has particles: the `particles.sim` (compute)
/// and `particles.draw` (graphics) library passes and the
/// `particles.state`, `particles.keys` and `particles.indirect` library
/// buffers exist. `particles.draw` blends into one colour attachment and
/// samples the depth buffer in the fragment stage (soft edges; it tests
/// depth itself), so a graph that draws particles declares its depth
/// buffer with sampled usage. A graph without them draws no particles.
pub const DECLARE_PARTICLES: u32 = 1 << 4;

/// How the output image encodes the display's light ([`DeclareInput::encoding`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum OutputEncoding {
    /// An sRGB 8-bit target: no display intermediate, no `post.encode`.
    Sdr = 0,
    /// HDR10: ST.2084 (PQ) in BT.2020, at `v * paper_white` nits.
    Hdr10Pq = 1,
    /// scRGB: BT.709-relative linear light, `v * paper_white / 80`.
    ScRgb = 2,
}

impl OutputEncoding {
    pub fn from_wire(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Sdr),
            1 => Some(Self::Hdr10Pq),
            2 => Some(Self::ScRgb),
            _ => None,
        }
    }

    /// Whether the graph keeps a display intermediate and ends in
    /// `post.encode` (the display rule).
    pub fn uses_display(self) -> bool {
        !matches!(self, Self::Sdr)
    }
}

/// The space the scene is lit in and the display intermediate stores
/// ([`DeclareInput::working_space`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum WorkingSpace {
    /// BT.709 / sRGB primaries: the default.
    Bt709 = 0,
    /// BT.2020 primaries, D65 white.
    Bt2020 = 1,
}

impl WorkingSpace {
    pub fn from_wire(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Bt709),
            1 => Some(Self::Bt2020),
            _ => None,
        }
    }
}

/// The primaries the display intermediate is mapped to before encoding
/// ([`DeclareInput::output_gamut`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Gamut {
    Bt709 = 0,
    P3 = 1,
    Bt2020 = 2,
}

impl Gamut {
    pub fn from_wire(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Bt709),
            1 => Some(Self::P3),
            2 => Some(Self::Bt2020),
            _ => None,
        }
    }

    /// The R, G and B primaries' chromaticities (`x`, `y`), D65 white.
    pub fn primaries(self) -> [[f32; 2]; 3] {
        match self {
            Self::Bt709 => [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]],
            Self::P3 => [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
            Self::Bt2020 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        }
    }
}

/// What the host writes before calling `declare` (`#[repr(C)]`, 60 bytes):
/// the target and the output's display spec. Fixed size, so an ABI
/// extension appends reserved words rather than moving fields.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DeclareInput {
    pub width: u32,
    pub height: u32,
    /// `VkFormat` of the target (the swapchain's, or the readback image's).
    pub color_format: u32,
    /// [`DECLARE_UNITS`], [`DECLARE_READBACK`], [`DECLARE_WATER`],
    /// [`DECLARE_SHADOWS`], [`DECLARE_PARTICLES`].
    pub flags: u32,
    /// [`OutputEncoding`] as a wire value.
    pub encoding: u32,
    /// [`WorkingSpace`] as a wire value.
    pub working_space: u32,
    /// [`Gamut`] as a wire value.
    pub output_gamut: u32,
    /// Paper white in nits: signal nits, relative to the output's reference
    /// (the display's 1.0, and the display intermediate's unit).
    pub paper_white: f32,
    /// The display's peak in the same signal nits (paper white × headroom).
    pub peak: f32,
    pub reserved: [u32; 6],
}

pub const DECLARE_INPUT_BYTES: usize = 60;
const _: () = assert!(size_of::<DeclareInput>() == DECLARE_INPUT_BYTES);

impl DeclareInput {
    pub fn to_bytes(&self) -> [u8; DECLARE_INPUT_BYTES] {
        // Safety: repr(C), only u32 and f32 fields, no padding.
        unsafe { core::mem::transmute_copy(self) }
    }

    pub fn from_bytes(bytes: &[u8; DECLARE_INPUT_BYTES]) -> Self {
        // Safety: as above; every bit pattern is valid.
        unsafe { core::mem::transmute_copy(bytes) }
    }

    /// The encoding, as the module should read it (`None` for a value this
    /// ABI does not know).
    pub fn output_encoding(&self) -> Option<OutputEncoding> {
        OutputEncoding::from_wire(self.encoding)
    }

    pub fn working(&self) -> Option<WorkingSpace> {
        WorkingSpace::from_wire(self.working_space)
    }

    pub fn gamut(&self) -> Option<Gamut> {
        Gamut::from_wire(self.output_gamut)
    }
}

/// The `shadows.map` library image as a graph declares it (`External`):
/// [`SHADOW_MAP_FORMAT`], [`SHADOW_MAP_WIDTH`] × [`SHADOW_MAP_HEIGHT`],
/// usage depth attachment and sampled, depth aspect.
pub const SHADOW_MAP_FORMAT: u32 = 126; // VK_FORMAT_D32_SFLOAT
/// The map's cascades side by side, each [`SHADOW_MAP_HEIGHT`] square.
pub const SHADOW_CASCADES: u32 = 3;
pub const SHADOW_MAP_HEIGHT: u32 = 1536;
pub const SHADOW_MAP_WIDTH: u32 = SHADOW_CASCADES * SHADOW_MAP_HEIGHT;

/// The name and spec a graph declares `shadows.map` with (class `External`).
pub const SHADOW_MAP_NAME: &str = "shadows.map";
pub fn shadow_map_spec() -> crate::ImageSpec {
    // Usage: sampled (0x4) | depth-stencil attachment (0x20).
    crate::ImageSpec::depth_2d(SHADOW_MAP_FORMAT, SHADOW_MAP_WIDTH, SHADOW_MAP_HEIGHT, 0x4 | 0x20)
}

/// The particles' coloured shadows (`shadows.color`), in a scene whose
/// units cast shadows ([`DECLARE_SHADOWS`]) and that has particles
/// ([`DECLARE_PARTICLES`]): laid out like `shadows.map` (the same cascades
/// side by side) at a quarter of its resolution, smoke being soft. RGB: the
/// optical depth of the particles between the sun and each texel, per
/// channel (cleared to 0, summed); A: the depth toward the sun of the
/// particle nearest the sun (kept by a max). Written by the
/// `shadows.particles` library pass (`color:clear`; `particles.state` and
/// `particles.keys` read in the vertex stage, `particles.indirect` as an
/// indirect buffer). A graph that declares it also
/// declares `shadows.map`, and every receiver samples both.
pub const PARTICLE_SHADOW_FORMAT: u32 = 97; // VK_FORMAT_R16G16B16A16_SFLOAT
pub const PARTICLE_SHADOW_WIDTH: u32 = SHADOW_MAP_WIDTH / 4;
pub const PARTICLE_SHADOW_HEIGHT: u32 = SHADOW_MAP_HEIGHT / 4;
pub const PARTICLE_SHADOW_NAME: &str = "shadows.color";
pub fn particle_shadow_spec() -> crate::ImageSpec {
    // Usage: sampled (0x4) | colour attachment (0x10).
    crate::ImageSpec::color_2d(
        PARTICLE_SHADOW_FORMAT,
        PARTICLE_SHADOW_WIDTH,
        PARTICLE_SHADOW_HEIGHT,
        0x4 | 0x10,
    )
}

/// The scene's colour image (HDR): what `terrain.draw`, `units.draw`,
/// `water.draw` and `particles.draw` draw into (linear light, may exceed
/// 1). A graph declares it itself (any name, usually [`SCENE_NAME`];
/// `Transient`) with this format and with sampled usage, and a
/// `post.tonemap` pass (graphics; the scene `sampled:fragment`, one colour
/// attachment of the target's format) turns it into the output.
pub const SCENE_FORMAT: u32 = 97; // VK_FORMAT_R16G16B16A16_SFLOAT
pub const SCENE_NAME: &str = "scene";
pub fn scene_spec(width: u32, height: u32) -> crate::ImageSpec {
    // Usage: sampled (0x4) | colour attachment (0x10).
    crate::ImageSpec::color_2d(SCENE_FORMAT, width, height, 0x4 | 0x10)
}

/// The display intermediate (`DISPLAY_NAME`): the working space's linear
/// light in units of paper white, which `post.encode` maps to the output.
/// An HDR graph declares it (`Transient`, sampled, full size) and has
/// `post.tonemap` and any colour-changing pass (a module's grade, a
/// vignette) write it; only `post.encode` writes the target. An SDR graph
/// has no display and no encode.
pub const DISPLAY_FORMAT: u32 = 97; // VK_FORMAT_R16G16B16A16_SFLOAT
pub const DISPLAY_NAME: &str = "display";
pub fn display_spec(width: u32, height: u32) -> crate::ImageSpec {
    // Usage: sampled (0x4) | colour attachment (0x10).
    crate::ImageSpec::color_2d(DISPLAY_FORMAT, width, height, 0x4 | 0x10)
}

/// The scene's glow image: the second colour attachment of every scene
/// draw, holding only light sources (a glowing decal, a water glint, an
/// emissive model or particle) and black everywhere else. A graph that
/// declares [`GLOW_NAME`] has every `terrain.draw`, `units.draw`,
/// `water.draw` and `particles.draw` write it and its `post.bloom` passes
/// read it; everything that is not a light source writes black (covering
/// what is behind it). The bloom reads it at a quarter of the size.
/// `B10G11R11` as RA3-graphics' glow target: it has no alpha, so its
/// attachment write mask is RGB.
pub const GLOW_FORMAT: u32 = 122; // VK_FORMAT_B10G11R11_UFLOAT_PACK32
pub const GLOW_NAME: &str = "glow";
pub fn glow_spec(width: u32, height: u32) -> crate::ImageSpec {
    // Usage: sampled (0x4) | colour attachment (0x10).
    crate::ImageSpec::color_2d(GLOW_FORMAT, width, height, 0x4 | 0x10)
}

/// A bloom buffer (`bloom.downsample`, `bloom.blur_x`, `bloom.blur_y`):
/// the glow at a quarter of the target's size, blurred with an 11-tap
/// Gaussian (RA3-graphics' three passes) and added to the scene before the
/// tonemap. `RGBA16F`, like the scene, so the blur does not band.
pub const BLOOM_FORMAT: u32 = 97; // VK_FORMAT_R16G16B16A16_SFLOAT
pub const BLOOM_NAME: &str = "bloom.0";
pub const BLOOM_SCRATCH_NAME: &str = "bloom.1";
/// The bloom images' size for a target `width` × `height` (a quarter, at
/// least 1).
pub fn bloom_size(width: u32, height: u32) -> (u32, u32) {
    (
        ((width as f32 * 0.25).round() as u32).max(1),
        ((height as f32 * 0.25).round() as u32).max(1),
    )
}
pub fn bloom_spec(width: u32, height: u32) -> crate::ImageSpec {
    let (w, h) = bloom_size(width, height);
    // Usage: sampled (0x4) | colour attachment (0x10).
    crate::ImageSpec::color_2d(BLOOM_FORMAT, w, h, 0x4 | 0x10)
}

/// The exposure image: a 1×1 `RGBA32F` the `exposure` library passes read
/// and write, sampled by `post.bloom` and `post.tonemap`. It is the frame's
/// metered, eased EV100, a small buffer so the exposure reaches the shading
/// passes on the GPU with no CPU readback:
///
/// | texel | meaning |
/// |---|---|
/// | `x` | the EV100 the tonemap and the bloom apply this frame |
/// | `y` | the metered mean log2 luminance of the stored scene |
/// | `z` | the target EV100 the metering asked for, before easing |
/// | `w` | 1 when the frame metered (auto-exposure on), 0 when manual |
///
/// A graph that declares it also declares [`EXPOSURE_HISTOGRAM_NAME`] and
/// the `exposure.clear`, `exposure.meter` and `exposure.adapt` passes, and
/// `post.bloom`'s downsample and `post.tonemap` sample it (the scene is
/// stored at the camera's reference EV and the tonemap applies the eased
/// one). `RGBA32F`: 16 bytes, and the diagnostics cost nothing to carry.
pub const EXPOSURE_FORMAT: u32 = 109; // VK_FORMAT_R32G32B32A32_SFLOAT
pub const EXPOSURE_NAME: &str = "exposure";
/// The metering histogram: [`EXPOSURE_BINS`] `u32` counts of the scene's
/// log2 luminance in [`EXPOSURE_LOG2_MIN`]..[`EXPOSURE_LOG2_MAX`]
/// (`exposure.meter` adds, `exposure.adapt` reads and the next frame's
/// `exposure.clear` zeroes), a host-owned `External` buffer.
pub const EXPOSURE_HISTOGRAM_NAME: &str = "exposure.histogram";
pub const EXPOSURE_BINS: u32 = 128;
pub const EXPOSURE_LOG2_MIN: f32 = -8.0;
pub const EXPOSURE_LOG2_MAX: f32 = 8.0;
pub const EXPOSURE_HISTOGRAM_BYTES: u64 = EXPOSURE_BINS as u64 * 4;

/// The exposure image as a graph declares it (`External`, 1×1): sampled and
/// storage (the `exposure.adapt` pass writes it).
pub fn exposure_spec() -> crate::ImageSpec {
    // Usage: sampled (0x4) | storage (0x8).
    crate::ImageSpec::color_2d(EXPOSURE_FORMAT, 1, 1, 0x4 | 0x8)
}

pub const PROGRAM_MAGIC: u32 = u32::from_le_bytes(*b"AEPT");

pub const PROGRAM_VERSION: u32 = 1;
const PROGRAM_HEADER_WORDS: usize = 4;
const PROGRAM_WORDS: usize = 6;
pub const MAX_PROGRAMS: usize = 4096;
pub const MAX_PROGRAM_BLOB: usize = 16 << 20;

pub const PROGRAM_LIBRARY: u32 = 0;
pub const PROGRAM_FULLSCREEN: u32 = 1;

/// What the host writes before each `frame` call (`#[repr(C)]`, 256 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameInput {
    pub width: u32,
    pub height: u32,
    pub frame: u32,
    pub units: u32,
    /// Seconds since the module was loaded.
    pub time: f32,
    /// Switches the host application passes through (e.g. debug views
    /// toggled by keys); their meaning is the module's.
    pub flags: u32,
    pub pad: [u32; 2],
    /// Column-major view-projection (Vulkan clip space, reverse-Z).
    pub view_proj: [f32; 16],
    /// Camera position (xyz).
    pub camera: [f32; 4],
    pub reserved: [[u32; 4]; 9],
}

pub const FRAME_INPUT_BYTES: usize = 256;
const _: () = assert!(size_of::<FrameInput>() == FRAME_INPUT_BYTES);

impl FrameInput {
    pub fn to_bytes(&self) -> [u8; FRAME_INPUT_BYTES] {
        // Safety: repr(C), only u32 and f32 fields, no padding.
        unsafe { core::mem::transmute_copy(self) }
    }

    pub fn from_bytes(bytes: &[u8; FRAME_INPUT_BYTES]) -> Self {
        // Safety: as above; every bit pattern is valid.
        unsafe { core::mem::transmute_copy(bytes) }
    }
}

/// One entry of a program table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Program<'a> {
    /// An engine library pass, by name (`terrain.draw`, `units.cull`, ...).
    Library(&'a str),
    /// A fullscreen triangle with this fragment shader into the pass's
    /// colour attachments.
    Fullscreen {
        label: &'a str,
        spirv: &'a [u8],
        uniform_bytes: u32,
    },
}

/// Writes a program table; each `push` returns the payload to give passes.
#[derive(Default)]
pub struct ProgramTable {
    entries: Vec<[u32; PROGRAM_WORDS]>,
    blob: Vec<u8>,
}

impl ProgramTable {
    pub fn new() -> Self {
        Self::default()
    }

    fn put(&mut self, bytes: &[u8]) -> (u32, u32) {
        // 4-byte aligned, so SPIR-V words stay aligned in the blob.
        while !self.blob.len().is_multiple_of(4) {
            self.blob.push(0);
        }
        let offset = self.blob.len() as u32;
        self.blob.extend_from_slice(bytes);
        (offset, bytes.len() as u32)
    }

    pub fn push(&mut self, program: Program<'_>) -> u32 {
        let entry = match program {
            Program::Library(name) => {
                let (no, nl) = self.put(name.as_bytes());
                [PROGRAM_LIBRARY, no, nl, 0, 0, 0]
            }
            Program::Fullscreen {
                label,
                spirv,
                uniform_bytes,
            } => {
                let (no, nl) = self.put(label.as_bytes());
                let (d, dl) = self.put(spirv);
                [PROGRAM_FULLSCREEN, no, nl, d, dl, uniform_bytes]
            }
        };
        self.entries.push(entry);
        self.entries.len() as u32 - 1
    }

    pub fn library(&mut self, name: &str) -> u32 {
        self.push(Program::Library(name))
    }

    pub fn finish(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut put = |w: u32| out.extend_from_slice(&w.to_le_bytes());
        for w in [
            PROGRAM_MAGIC,
            PROGRAM_VERSION,
            self.entries.len() as u32,
            self.blob.len() as u32,
        ] {
            put(w);
        }
        for w in self.entries.iter().flatten() {
            put(*w);
        }
        out.extend_from_slice(&self.blob);
        out
    }

    /// Decode a table: its programs in payload order.
    pub fn decode(bytes: &[u8]) -> Result<Vec<Program<'_>>, String> {
        let word = |i: usize| -> Result<u32, String> {
            bytes
                .get(i * 4..i * 4 + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .ok_or_else(|| format!("program table truncated at word {i}"))
        };
        if word(0)? != PROGRAM_MAGIC {
            return Err("not a program table (bad magic)".into());
        }
        if word(1)? != PROGRAM_VERSION {
            return Err(format!(
                "program table version {} (this host reads {PROGRAM_VERSION})",
                word(1)?
            ));
        }
        let (count, blob_len) = (word(2)? as usize, word(3)? as usize);
        if count > MAX_PROGRAMS || blob_len > MAX_PROGRAM_BLOB {
            return Err(format!(
                "program table too large: {count} programs, {blob_len} blob bytes"
            ));
        }
        let blob_start = (PROGRAM_HEADER_WORDS + count * PROGRAM_WORDS) * 4;
        if bytes.len() != blob_start + blob_len {
            return Err(format!(
                "program table is {} bytes; its header says {}",
                bytes.len(),
                blob_start + blob_len
            ));
        }
        let blob = &bytes[blob_start..];
        let slice = |offset: u32, len: u32, what: &str, index: usize| -> Result<&[u8], String> {
            blob.get(offset as usize..offset as usize + len as usize)
                .ok_or_else(|| format!("program {index}: {what} out of bounds"))
        };
        (0..count)
            .map(|index| {
                let w = |k: usize| word(PROGRAM_HEADER_WORDS + index * PROGRAM_WORDS + k);
                let name = core::str::from_utf8(slice(w(1)?, w(2)?, "name", index)?)
                    .map_err(|_| format!("program {index}: name is not UTF-8"))?;
                match w(0)? {
                    PROGRAM_LIBRARY => Ok(Program::Library(name)),
                    PROGRAM_FULLSCREEN => Ok(Program::Fullscreen {
                        label: name,
                        spirv: slice(w(3)?, w(4)?, "shader", index)?,
                        uniform_bytes: w(5)?,
                    }),
                    kind => Err(format!("program {index}: unknown kind {kind}")),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_tables_round_trip_and_reject_damage() {
        let mut t = ProgramTable::new();
        assert_eq!(t.library("terrain.draw"), 0);
        let spirv = [0x0723_0203u32.to_le_bytes(), [1, 2, 3, 4]].concat();
        assert_eq!(
            t.push(Program::Fullscreen {
                label: "post",
                spirv: &spirv,
                uniform_bytes: 32
            }),
            1
        );
        let bytes = t.finish();
        let programs = ProgramTable::decode(&bytes).unwrap();
        assert_eq!(
            programs,
            [
                Program::Library("terrain.draw"),
                Program::Fullscreen {
                    label: "post",
                    spirv: &spirv,
                    uniform_bytes: 32
                }
            ]
        );
        assert!(ProgramTable::decode(&bytes[..bytes.len() - 1]).is_err());
        let mut bad = bytes.clone();
        bad[0] ^= 1;
        assert!(ProgramTable::decode(&bad).is_err());
        let mut bad = bytes.clone();
        bad[16] = 9; // kind of program 0
        assert!(ProgramTable::decode(&bad).unwrap_err().contains("unknown kind"));
        let mut bad = bytes;
        bad[20] = 200; // name offset of program 0
        assert!(ProgramTable::decode(&bad).unwrap_err().contains("out of bounds"));
        // Every truncation fails cleanly.
        let bytes = t.finish();
        for n in 0..bytes.len() {
            assert!(ProgramTable::decode(&bytes[..n]).is_err());
        }
    }

    #[test]
    fn frame_input_round_trips() {
        let mut input = FrameInput {
            width: 3,
            height: 4,
            time: 1.5,
            ..FrameInput::default()
        };
        input.view_proj[5] = -2.0;
        assert_eq!(FrameInput::from_bytes(&input.to_bytes()), input);
    }

    #[test]
    fn declare_input_round_trips_the_display_spec() {
        let input = DeclareInput {
            width: 3840,
            height: 1600,
            color_format: 48,
            flags: DECLARE_UNITS | DECLARE_READBACK,
            encoding: OutputEncoding::Hdr10Pq as u32,
            working_space: WorkingSpace::Bt2020 as u32,
            output_gamut: Gamut::P3 as u32,
            paper_white: 203.0,
            peak: 361.0,
            reserved: [0; 6],
        };
        let back = DeclareInput::from_bytes(&input.to_bytes());
        assert_eq!(back, input);
        assert_eq!(back.output_encoding(), Some(OutputEncoding::Hdr10Pq));
        assert_eq!(back.working(), Some(WorkingSpace::Bt2020));
        assert_eq!(back.gamut(), Some(Gamut::P3));
        assert_eq!(Gamut::P3.primaries(), [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]]);
        // The display rule: only SDR has no display intermediate.
        assert!(
            !OutputEncoding::Sdr.uses_display()
                && OutputEncoding::Hdr10Pq.uses_display()
                && OutputEncoding::ScRgb.uses_display()
        );
        // Unknown wire values are refusable rather than guessed.
        assert_eq!(OutputEncoding::from_wire(9), None);
        assert_eq!(WorkingSpace::from_wire(2), None);
        assert_eq!(Gamut::from_wire(3), None);
    }
}
