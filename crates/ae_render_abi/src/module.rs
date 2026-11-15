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
//! | [`EXPORT_DECLARE`] | `(width, height, color_format, flags) -> u32` | at load, on reload, when the target changes |
//! | [`EXPORT_INPUT`] | `() -> u32` | at load: where the host writes [`FrameInput`] |
//! | [`EXPORT_FRAME`] | `() -> u32` | once per frame, after the host wrote the input |
//!
//! `declare` returns the address of four `u32`s: graph pointer and length
//! (the frame graph declaration, [`crate::GraphBuilder`]), program table
//! pointer and length ([`ProgramTable`]); 0 if the module cannot render
//! that target. `flags` holds [`DECLARE_UNITS`], [`DECLARE_READBACK`],
//! [`DECLARE_WATER`], [`DECLARE_SHADOWS`] and [`DECLARE_PARTICLES`] (the
//! last three added later; a module that ignores them simply draws no
//! water, no unit shadows, or no particles).
//! `frame` returns the address of [`FrameOutput`] as four `u32`s: pointer
//! and word count of a bitset of *disabled* programs (a conditional pass
//! whose payload's bit is set is skipped), pointer and length of the
//! uniform bytes: every fullscreen program's block, in program order, each
//! [`Program::Fullscreen::uniform_bytes`] long.
//!
//! # Program table (version 1)
//!
//! Little-endian `u32` words: header `magic, version, count, blob bytes`,
//! then per program `kind, name offset, name length, data offset, data
//! length, uniform bytes`, then the blob. A pass's payload indexes it.
//! `kind` 0 is a library pass (name = its library name, no data); 1 is a
//! fullscreen pass (name = a label, data = fragment shader SPIR-V;
//! `uniform bytes` = size of its uniform block at binding 0, 0 for none;
//! its sampled inputs are bindings 1.. in intent order).

pub const EXPORT_VERSION: &str = "ae_render_version";
pub const EXPORT_DECLARE: &str = "ae_render_declare";
pub const EXPORT_INPUT: &str = "ae_render_input";
pub const EXPORT_FRAME: &str = "ae_render_frame";
pub const MODULE_VERSION: u32 = 1;

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
    crate::ImageSpec::color_2d(PARTICLE_SHADOW_FORMAT, PARTICLE_SHADOW_WIDTH, PARTICLE_SHADOW_HEIGHT, 0x4 | 0x10)
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
    Fullscreen { label: &'a str, spirv: &'a [u8], uniform_bytes: u32 },
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
            Program::Fullscreen { label, spirv, uniform_bytes } => {
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
        for w in [PROGRAM_MAGIC, PROGRAM_VERSION, self.entries.len() as u32, self.blob.len() as u32] {
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
            bytes.get(i * 4..i * 4 + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(|| format!("program table truncated at word {i}"))
        };
        if word(0)? != PROGRAM_MAGIC {
            return Err("not a program table (bad magic)".into());
        }
        if word(1)? != PROGRAM_VERSION {
            return Err(format!("program table version {} (this host reads {PROGRAM_VERSION})", word(1)?));
        }
        let (count, blob_len) = (word(2)? as usize, word(3)? as usize);
        if count > MAX_PROGRAMS || blob_len > MAX_PROGRAM_BLOB {
            return Err(format!("program table too large: {count} programs, {blob_len} blob bytes"));
        }
        let blob_start = (PROGRAM_HEADER_WORDS + count * PROGRAM_WORDS) * 4;
        if bytes.len() != blob_start + blob_len {
            return Err(format!("program table is {} bytes; its header says {}", bytes.len(), blob_start + blob_len));
        }
        let blob = &bytes[blob_start..];
        let slice = |offset: u32, len: u32, what: &str, index: usize| -> Result<&[u8], String> {
            blob.get(offset as usize..offset as usize + len as usize).ok_or_else(|| format!("program {index}: {what} out of bounds"))
        };
        (0..count)
            .map(|index| {
                let w = |k: usize| word(PROGRAM_HEADER_WORDS + index * PROGRAM_WORDS + k);
                let name = core::str::from_utf8(slice(w(1)?, w(2)?, "name", index)?).map_err(|_| format!("program {index}: name is not UTF-8"))?;
                match w(0)? {
                    PROGRAM_LIBRARY => Ok(Program::Library(name)),
                    PROGRAM_FULLSCREEN => Ok(Program::Fullscreen { label: name, spirv: slice(w(3)?, w(4)?, "shader", index)?, uniform_bytes: w(5)? }),
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
        assert_eq!(t.push(Program::Fullscreen { label: "post", spirv: &spirv, uniform_bytes: 32 }), 1);
        let bytes = t.finish();
        let programs = ProgramTable::decode(&bytes).unwrap();
        assert_eq!(programs, [Program::Library("terrain.draw"), Program::Fullscreen { label: "post", spirv: &spirv, uniform_bytes: 32 }]);
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
        let mut input = FrameInput { width: 3, height: 4, time: 1.5, ..FrameInput::default() };
        input.view_proj[5] = -2.0;
        assert_eq!(FrameInput::from_bytes(&input.to_bytes()), input);
    }
}
