//! The engine's model format: one skinned format for every model (M11).
//!
//! glTF 2.0 is converted offline (`ae_gltf`, `ae-tool model convert`) to a
//! [`Model`]; the runtime reads it with [`Model::from_bytes`] (from the VFS,
//! `ae_vfs::Vfs::read`).
//!
//! * **Bones** are the glTF nodes of the scene, parents before children,
//!   under one synthetic root (bone 0) that turns glTF's axes into the
//!   engine's ([`GLTF_TO_ENGINE`]). Each has its rest transform as a full
//!   matrix (non-uniform scale and all) and as TRS for clips to override.
//! * **Joints** are the skinning palette: a bone and an inverse bind matrix.
//!   A skinned glTF mesh references its skin's joints; a mesh hanging on a
//!   node (a "piece") references one joint, the node with the identity, at
//!   weight 1. So there is one vertex layout and one draw path, and moving a
//!   bone moves both kinds.
//! * **Primitives** are triangle lists, one material each, in the one shared
//!   vertex and index buffer.
//! * **Textures** are KTX2 files in a GPU format ([`ktx`]).
//! * **Clips** are glTF's animations as data: keyframes per bone and path,
//!   with their interpolation ([`pose`] samples them).
//!
//! Engine model space: +X forward, +Y up, +Z right (a unit's heading turns
//! +X onto it), metres.
//!
//! File layout (little-endian): magic `AEMODEL\0`, version (u32), header
//! length (u32), the header (JSON, padded to 16 bytes), then the blob; the
//! header's [`Range`]s are byte ranges in the blob, 16-byte aligned.

pub mod ktx;
pub mod pose;

use anyhow::{Context as _, Result, bail, ensure};
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

// Vertices are written and uploaded as their in-memory bytes.
#[cfg(target_endian = "big")]
compile_error!("ae_model assumes a little-endian host");

pub const MAGIC: [u8; 8] = *b"AEMODEL\0";
pub const VERSION: u32 = 1;
/// Influences per vertex.
pub const MAX_INFLUENCES: usize = 4;

/// glTF (+Y up, front +Z, +X left) to the engine (+X forward, +Y up, +Z
/// right): a quarter turn about +Y, so glTF +Z goes to +X and +X to -Z.
pub const GLTF_TO_ENGINE: Quat = Quat::from_xyzw(0.0, std::f32::consts::FRAC_1_SQRT_2, 0.0, std::f32::consts::FRAC_1_SQRT_2);

/// One vertex; 80 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// xyz tangent, w the bitangent's sign (glTF's convention); (1, 0, 0, 1)
    /// where the primitive has no normal map.
    pub tangent: [f32; 4],
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    /// Linear RGBA, unorm16; white where the glTF has no COLOR_0.
    pub color: [u16; 4],
    /// Indices into [`Model::joints`].
    pub joints: [u16; 4],
    /// Unorm16, summing to 65535.
    pub weights: [u16; 4],
}

pub const VERTEX_BYTES: usize = std::mem::size_of::<Vertex>();
const _: () = assert!(VERTEX_BYTES == 80);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Range {
    pub offset: u64,
    pub len: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bone {
    pub name: String,
    /// Always lower than the bone's own index; `None` for the root only.
    pub parent: Option<u32>,
    /// Rest transform relative to the parent, column-major.
    pub rest: [f32; 16],
    /// The same as translation, rotation (xyzw) and scale; clips override
    /// these. Approximate where `rest` is not a TRS (a sheared glTF matrix,
    /// which glTF forbids on animated nodes).
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Joint {
    pub bone: u32,
    /// Column-major.
    pub inverse_bind: [f32; 16],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Primitive {
    pub material: u32,
    /// Vertices of this primitive (indices are relative to `first_vertex`).
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Filter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Wrap {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sampler {
    pub mag: Filter,
    pub min: Filter,
    /// `None`: no mipmapping (glTF's NEAREST / LINEAR minification).
    pub mip: Option<Filter>,
    pub wrap_s: Wrap,
    pub wrap_t: Wrap,
}

impl Default for Sampler {
    fn default() -> Self {
        Sampler { mag: Filter::Linear, min: Filter::Linear, mip: Some(Filter::Linear), wrap_s: Wrap::Repeat, wrap_t: Wrap::Repeat }
    }
}

/// `KHR_texture_transform`: uv' = T(offset) R(rotation) S(scale) uv.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct UvTransform {
    pub offset: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
}

impl UvTransform {
    /// The 3x2 matrix (columns) the shader applies: `m[0] * u + m[1] * v + m[2]`.
    pub fn matrix(&self) -> [[f32; 2]; 3] {
        let (s, c) = self.rotation.sin_cos();
        [[c * self.scale[0], -s * self.scale[0]], [s * self.scale[1], c * self.scale[1]], self.offset]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureRef {
    pub texture: u32,
    /// 0 or 1: which UV set.
    pub uv: u32,
    pub sampler: Sampler,
    pub transform: Option<UvTransform>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum AlphaMode {
    Opaque,
    Mask { cutoff: f32 },
    Blend,
}

/// glTF's metallic-roughness material.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Material {
    pub name: String,
    /// Linear RGBA.
    pub base_color: [f32; 4],
    pub base_color_texture: Option<TextureRef>,
    pub metallic: f32,
    pub roughness: f32,
    /// Roughness in G, metalness in B.
    pub metallic_roughness_texture: Option<TextureRef>,
    pub normal_texture: Option<TextureRef>,
    pub normal_scale: f32,
    pub occlusion_texture: Option<TextureRef>,
    pub occlusion_strength: f32,
    /// Linear RGB, `KHR_materials_emissive_strength` multiplied in.
    pub emissive: [f32; 3],
    pub emissive_texture: Option<TextureRef>,
    pub alpha: AlphaMode,
    pub double_sided: bool,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            name: String::new(),
            base_color: [1.0; 4],
            base_color_texture: None,
            metallic: 1.0,
            roughness: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            emissive: [0.0; 3],
            emissive_texture: None,
            alpha: AlphaMode::Opaque,
            double_sided: false,
        }
    }
}

/// A KTX2 file in the blob ([`ktx`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Texture {
    pub name: String,
    pub ktx2: Range,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Path {
    Translation,
    Rotation,
    Scale,
}

impl Path {
    pub fn components(self) -> usize {
        match self {
            Path::Rotation => 4,
            _ => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interpolation {
    Step,
    Linear,
    /// Values are (in-tangent, value, out-tangent) per key.
    CubicSpline,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Channel {
    pub bone: u32,
    pub path: Path,
    pub interpolation: Interpolation,
    /// Key times (f32, seconds, increasing) in the blob.
    pub times: Range,
    /// Key values (f32) in the blob.
    pub values: Range,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    pub name: String,
    /// The last key time over all channels.
    pub duration: f32,
    pub channels: Vec<Channel>,
}

/// Axis-aligned box in model space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// The header: everything but the bulk data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Header {
    bones: Vec<Bone>,
    joints: Vec<Joint>,
    primitives: Vec<Primitive>,
    materials: Vec<Material>,
    textures: Vec<Texture>,
    clips: Vec<Clip>,
    bounds: Bounds,
    vertices: Range,
    indices: Range,
}

/// A model in memory.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Model {
    pub bones: Vec<Bone>,
    pub joints: Vec<Joint>,
    pub primitives: Vec<Primitive>,
    pub materials: Vec<Material>,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// KTX2 files, one per [`Model::textures`] entry.
    pub texture_names: Vec<String>,
    pub textures: Vec<Vec<u8>>,
    /// Channels' `times` and `values` are ranges in `clip_data`, in f32s
    /// (not bytes) while in memory.
    pub clips: Vec<Clip>,
    pub clip_data: Vec<f32>,
    /// At the rest pose.
    pub bounds: Bounds,
}

fn align16(n: usize) -> usize {
    n.div_ceil(16) * 16
}

impl Model {
    pub fn clip_floats(&self, r: Range) -> &[f32] {
        &self.clip_data[r.offset as usize..(r.offset + r.len) as usize]
    }

    /// The vertices as bytes (for upload).
    pub fn vertex_bytes(&self) -> &[u8] {
        // SAFETY: Vertex is repr(C), 80 bytes, no padding, plain numbers.
        unsafe { std::slice::from_raw_parts(self.vertices.as_ptr().cast(), self.vertices.len() * VERTEX_BYTES) }
    }

    /// Checks the references between the parts.
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.bones.is_empty() && self.bones[0].parent.is_none(), "bone 0 must be the root");
        for (i, b) in self.bones.iter().enumerate().skip(1) {
            let p = b.parent.with_context(|| format!("bone {i} ({}) has no parent", b.name))?;
            ensure!((p as usize) < i, "bone {i} ({}): parent {p} not before it", b.name);
        }
        for (i, j) in self.joints.iter().enumerate() {
            ensure!((j.bone as usize) < self.bones.len(), "joint {i}: bone {} out of range", j.bone);
        }
        ensure!(self.texture_names.len() == self.textures.len(), "texture names and data differ in count");
        let tex_ok = |t: &Option<TextureRef>| t.is_none_or(|t| (t.texture as usize) < self.textures.len() && t.uv < 2);
        for (i, m) in self.materials.iter().enumerate() {
            ensure!(
                tex_ok(&m.base_color_texture) && tex_ok(&m.metallic_roughness_texture) && tex_ok(&m.normal_texture) && tex_ok(&m.occlusion_texture) && tex_ok(&m.emissive_texture),
                "material {i} ({}): texture reference out of range",
                m.name
            );
        }
        for (i, p) in self.primitives.iter().enumerate() {
            ensure!((p.material as usize) < self.materials.len(), "primitive {i}: material out of range");
            ensure!(p.index_count % 3 == 0, "primitive {i}: {} indices, not triangles", p.index_count);
            let v_end = p.first_vertex as u64 + p.vertex_count as u64;
            let i_end = p.first_index as u64 + p.index_count as u64;
            ensure!(v_end <= self.vertices.len() as u64 && i_end <= self.indices.len() as u64, "primitive {i}: range out of bounds");
            ensure!(self.indices[p.first_index as usize..i_end as usize].iter().all(|&x| x < p.vertex_count), "primitive {i}: index out of range");
            for v in &self.vertices[p.first_vertex as usize..v_end as usize] {
                ensure!(v.joints.iter().all(|&j| (j as usize) < self.joints.len()), "primitive {i}: joint out of range");
            }
        }
        for c in &self.clips {
            for ch in &c.channels {
                ensure!((ch.bone as usize) < self.bones.len(), "clip {}: bone out of range", c.name);
                ensure!(ch.times.offset + ch.times.len <= self.clip_data.len() as u64 && ch.values.offset + ch.values.len <= self.clip_data.len() as u64, "clip {}: data out of range", c.name);
                let keys = ch.times.len as usize;
                let per = ch.path.components() * if ch.interpolation == Interpolation::CubicSpline { 3 } else { 1 };
                ensure!(keys > 0 && ch.values.len as usize == keys * per, "clip {}: {} keys but {} values", c.name, keys, ch.values.len);
                ensure!(self.clip_floats(ch.times).windows(2).all(|w| w[0] <= w[1]), "clip {}: key times not increasing", c.name);
            }
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut blob: Vec<u8> = Vec::new();
        let mut put = |bytes: &[u8]| -> Range {
            blob.resize(align16(blob.len()), 0);
            let r = Range { offset: blob.len() as u64, len: bytes.len() as u64 };
            blob.extend_from_slice(bytes);
            r
        };
        let vertices = put(self.vertex_bytes());
        let index_bytes: Vec<u8> = self.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let indices = put(&index_bytes);
        let textures = self.textures.iter().zip(&self.texture_names).map(|(t, name)| Texture { name: name.clone(), ktx2: put(t) }).collect();
        let clip_bytes: Vec<u8> = self.clip_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let clip_base = put(&clip_bytes).offset;
        let bytes_of = |r: Range| Range { offset: clip_base + r.offset * 4, len: r.len * 4 };
        let clips = self
            .clips
            .iter()
            .map(|c| Clip { channels: c.channels.iter().map(|ch| Channel { times: bytes_of(ch.times), values: bytes_of(ch.values), ..ch.clone() }).collect(), ..c.clone() })
            .collect();
        let header = Header {
            bones: self.bones.clone(),
            joints: self.joints.clone(),
            primitives: self.primitives.clone(),
            materials: self.materials.clone(),
            textures,
            clips,
            bounds: self.bounds,
            vertices,
            indices,
        };
        let mut json = serde_json::to_vec(&header).expect("model header serialises");
        json.resize(align16(json.len() + 16) - 16, b' ');
        let mut out = Vec::with_capacity(16 + json.len() + blob.len());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&json);
        out.extend_from_slice(&blob);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Model> {
        ensure!(bytes.len() >= 16 && bytes[..8] == MAGIC, "not an engine model (no AEMODEL magic)");
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != VERSION {
            bail!("model version {version}, this build reads {VERSION} (convert it again)");
        }
        let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        ensure!(16 + header_len <= bytes.len(), "model header truncated");
        let header: Header = serde_json::from_slice(&bytes[16..16 + header_len]).context("model header")?;
        let blob = &bytes[16 + header_len..];
        let slice = |r: Range, what: &str| -> Result<&[u8]> {
            let end = r.offset.checked_add(r.len).filter(|&e| e <= blob.len() as u64).with_context(|| format!("model {what} out of the file"))?;
            Ok(&blob[r.offset as usize..end as usize])
        };
        let vb = slice(header.vertices, "vertices")?;
        ensure!(vb.len() % VERTEX_BYTES == 0, "vertex data not whole vertices");
        let vertices = vb.as_chunks::<VERTEX_BYTES>().0.iter().map(|c| vertex_from_le(c)).collect();
        let ib = slice(header.indices, "indices")?;
        ensure!(ib.len() % 4 == 0, "index data not whole u32s");
        let indices = ib.as_chunks::<4>().0.iter().map(|&c| u32::from_le_bytes(c)).collect();
        let mut textures = Vec::new();
        let mut texture_names = Vec::new();
        for t in &header.textures {
            textures.push(slice(t.ktx2, "texture")?.to_vec());
            texture_names.push(t.name.clone());
        }
        // Clip data: every channel's ranges, rebased onto one f32 vector.
        let mut clip_data = Vec::new();
        let mut clips = Vec::new();
        for c in &header.clips {
            let mut channels = Vec::new();
            for ch in &c.channels {
                let mut floats = |r: Range| -> Result<Range> {
                    let b = slice(r, "clip")?;
                    ensure!(b.len() % 4 == 0, "clip data not whole f32s");
                    let start = clip_data.len() as u64;
                    clip_data.extend(b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)));
                    Ok(Range { offset: start, len: (b.len() / 4) as u64 })
                };
                let times = floats(ch.times)?;
                let values = floats(ch.values)?;
                channels.push(Channel { times, values, ..ch.clone() });
            }
            clips.push(Clip { channels, ..c.clone() });
        }
        let model = Model {
            bones: header.bones,
            joints: header.joints,
            primitives: header.primitives,
            materials: header.materials,
            vertices,
            indices,
            texture_names,
            textures,
            clips,
            clip_data,
            bounds: header.bounds,
        };
        model.validate()?;
        Ok(model)
    }
}

fn vertex_from_le(b: &[u8]) -> Vertex {
    let f = |i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
    let h = |i: usize| u16::from_le_bytes(b[i * 2..i * 2 + 2].try_into().unwrap());
    Vertex {
        position: [f(0), f(1), f(2)],
        normal: [f(3), f(4), f(5)],
        tangent: [f(6), f(7), f(8), f(9)],
        uv0: [f(10), f(11)],
        uv1: [f(12), f(13)],
        color: [h(28), h(29), h(30), h(31)],
        joints: [h(32), h(33), h(34), h(35)],
        weights: [h(36), h(37), h(38), h(39)],
    }
}

/// TRS to a matrix.
pub fn trs(t: [f32; 3], r: [f32; 4], s: [f32; 3]) -> Mat4 {
    Mat4::from_scale_rotation_translation(Vec3::from(s), Quat::from_array(r), Vec3::from(t))
}

#[cfg(test)]
mod tests;
