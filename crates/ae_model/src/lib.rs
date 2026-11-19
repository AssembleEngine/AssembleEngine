//! The engine's model format (M11).
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
//!   weight 1. Moving a bone moves both kinds.
//! * **Primitives** are triangle lists, one material each, in the one shared
//!   vertex and index buffer. Each has a GPU [`Layout`]: rigid (one joint
//!   per vertex: pieces, merged per material by
//!   [`Model::merge_primitives`]) or skinned (four joints and weights).
//! * **Textures** are KTX2 files in a GPU format ([`ktx`]).
//! * **Clips** are glTF's animations as data: keyframes per bone and path,
//!   with their interpolation ([`pose`] samples them, and bakes them at a
//!   fixed rate for the renderer).
//! * **Lights** are glTF's punctual lights (`KHR_lights_punctual`): each
//!   rides the joint of the node that carries it and shines along that
//!   node's local -Z (glTF's convention). Their colour is already in the
//!   engine's light units; the renderer appends one light per unit and
//!   model light to the frame's light list.
//!
//! Engine model space: +X forward, +Y up, +Z right (a unit's heading turns
//! +X onto it), metres.
//!
//! File layout (little-endian): magic `AEMODEL\0`, version (u32), header
//! length (u32), the header (JSON, padded to 16 bytes), then the blob; the
//! header's [`Range`]s are byte ranges in the blob, 16-byte aligned.

pub mod anim;
pub mod ktx;
pub mod pose;

use anyhow::{Context as _, Result, bail, ensure};
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

// Vertices are written and uploaded as their in-memory bytes.
#[cfg(target_endian = "big")]
compile_error!("ae_model assumes a little-endian host");

pub const MAGIC: [u8; 8] = *b"AEMODEL\0";
pub const VERSION: u32 = 3;
/// Influences per vertex.
pub const MAX_INFLUENCES: usize = 4;

/// glTF's candela (point and spot intensity) to the engine's light units
/// (the renderer's `Light::color`): 100 cd is 1.0, a light as bright as
/// the sun's diffuse term at the source. Engine light units are arbitrary;
/// this is the documented mapping the importer uses.
pub const CANDELA_TO_ENGINE: f32 = 0.01;
/// A glTF light without a `range` is infinite (the spec's default). The
/// engine's lights are all finite: the importer gives such a light this
/// range (the map's widest scale), so its cutoff is beyond anything the
/// camera sees and its cull keeps it in every cluster.
pub const UNBOUNDED_RANGE: f32 = 10_000.0;

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

/// A glTF punctual light (`KHR_lights_punctual`) that rides a bone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightKind {
    Point,
    Spot,
}

/// A light of the model, carried by a node: the renderer appends one light
/// per unit and model light to the frame's light list.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelLight {
    pub name: String,
    /// The joint of the node carrying it (index into [`Model::joints`]):
    /// its palette matrix, with the node's pose, gives the light's position
    /// and its direction (the node's local -Z, glTF's convention).
    pub joint: u32,
    pub kind: LightKind,
    /// Linear RGB in the engine's light units ([`CANDELA_TO_ENGINE`]).
    pub color: [f32; 3],
    /// The distance past which it lights nothing (metres,
    /// [`UNBOUNDED_RANGE`] when the glTF has none).
    pub range: f32,
    /// A spot's half-angles from its axis (radians). 0 for a point.
    pub inner_angle: f32,
    pub outer_angle: f32,
    /// The source's size (metres), which regularises the falloff near it.
    pub size: f32,
}

/// How a primitive's vertices are packed for the GPU (vertex pulling).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layout {
    /// One joint per vertex at weight 1 (pieces): [`RIGID_WORDS`] words.
    Rigid,
    /// Four joints and weights: [`SKINNED_WORDS`] words.
    Skinned,
}

/// 4-byte words of a rigid vertex: position 0..2, normal 3..5, tangent
/// 6..9, uv0 10..11, uv1 12..13, colour 14..15 (RGBA unorm16), joint 16.
pub const RIGID_WORDS: usize = 17;
/// 4-byte words of a skinned vertex: as rigid up to the colour, joints
/// 16..17 (u16), weights 18..19 (unorm16).
pub const SKINNED_WORDS: usize = 20;

impl Layout {
    pub fn words(self) -> usize {
        match self {
            Layout::Rigid => RIGID_WORDS,
            Layout::Skinned => SKINNED_WORDS,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Primitive {
    pub material: u32,
    /// Rigid: every vertex has weights `[65535, 0, 0, 0]`.
    pub layout: Layout,
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

/// A model's footprint radius: how far its rest box reaches across the
/// ground plane (x, z) from the origin. A unit of radius `r` draws its
/// model scaled by `r / footprint` (the renderer's fit, the sim's
/// `anim:bone_world`).
pub fn footprint(b: &Bounds) -> f32 {
    let f = [b.min[0], b.max[0], b.min[2], b.max[2]].iter().fold(0.0f32, |a, v| a.max(v.abs()));
    if f > 1e-6 {
        f
    } else {
        let d = [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-3)
    }
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
    lights: Vec<ModelLight>,
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
    pub lights: Vec<ModelLight>,
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

    /// The vertices of `p` packed for the GPU in its layout
    /// ([`Layout::words`] words each).
    pub fn gpu_words(&self, p: &Primitive) -> Vec<u32> {
        let vertices = &self.vertices[p.first_vertex as usize..(p.first_vertex + p.vertex_count) as usize];
        let mut out = Vec::with_capacity(vertices.len() * p.layout.words());
        for v in vertices {
            out.extend(v.position.iter().chain(&v.normal).chain(&v.tangent).chain(&v.uv0).chain(&v.uv1).map(|f| f.to_bits()));
            out.push(v.color[0] as u32 | (v.color[1] as u32) << 16);
            out.push(v.color[2] as u32 | (v.color[3] as u32) << 16);
            match p.layout {
                Layout::Rigid => out.push(v.joints[0] as u32),
                Layout::Skinned => {
                    out.push(v.joints[0] as u32 | (v.joints[1] as u32) << 16);
                    out.push(v.joints[2] as u32 | (v.joints[3] as u32) << 16);
                    out.push(v.weights[0] as u32 | (v.weights[1] as u32) << 16);
                    out.push(v.weights[2] as u32 | (v.weights[3] as u32) << 16);
                }
            }
        }
        out
    }

    /// Gives each primitive the cheapest layout its vertices allow (rigid
    /// when every vertex has one joint at weight 1), then merges the
    /// primitives of one material and layout into one (a piece model
    /// draws once per material: the per-vertex joint picks the piece).
    /// Order: by first appearance.
    pub fn merge_primitives(&mut self) {
        for p in &mut self.primitives {
            let vs = &self.vertices[p.first_vertex as usize..(p.first_vertex + p.vertex_count) as usize];
            p.layout = if vs.iter().all(|v| v.weights == [65535, 0, 0, 0]) { Layout::Rigid } else { Layout::Skinned };
        }
        let mut groups: Vec<((u32, Layout), Vec<usize>)> = Vec::new();
        for (i, p) in self.primitives.iter().enumerate() {
            match groups.iter_mut().find(|(k, _)| *k == (p.material, p.layout)) {
                Some((_, members)) => members.push(i),
                None => groups.push(((p.material, p.layout), vec![i])),
            }
        }
        let (mut vertices, mut indices, mut primitives) = (Vec::new(), Vec::new(), Vec::new());
        for ((material, layout), members) in groups {
            let (first_vertex, first_index) = (vertices.len() as u32, indices.len() as u32);
            for i in members {
                let p = &self.primitives[i];
                let base = vertices.len() as u32 - first_vertex;
                let mut vs = self.vertices[p.first_vertex as usize..(p.first_vertex + p.vertex_count) as usize].to_vec();
                if layout == Layout::Rigid {
                    for v in &mut vs {
                        v.joints = [v.joints[0], 0, 0, 0];
                    }
                }
                vertices.extend(vs);
                indices.extend(self.indices[p.first_index as usize..(p.first_index + p.index_count) as usize].iter().map(|x| x + base));
            }
            primitives.push(Primitive { material, layout, first_vertex, vertex_count: vertices.len() as u32 - first_vertex, first_index, index_count: indices.len() as u32 - first_index });
        }
        (self.vertices, self.indices, self.primitives) = (vertices, indices, primitives);
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
        for (i, l) in self.lights.iter().enumerate() {
            ensure!((l.joint as usize) < self.joints.len(), "light {i} ({}): joint {} out of range", l.name, l.joint);
            ensure!(l.range.is_finite() && l.range > 0.0, "light {i} ({}): range {}", l.name, l.range);
            ensure!(l.size.is_finite() && l.size > 0.0, "light {i} ({}): size {}", l.name, l.size);
            ensure!(
                l.inner_angle.is_finite() && l.outer_angle.is_finite() && 0.0 <= l.inner_angle && l.inner_angle <= l.outer_angle && l.outer_angle <= std::f32::consts::FRAC_PI_2,
                "light {i} ({}): angles {}..{} rad",
                l.name,
                l.inner_angle,
                l.outer_angle
            );
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
                ensure!(p.layout == Layout::Skinned || v.weights == [65535, 0, 0, 0], "primitive {i}: rigid with a vertex of several influences");
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
            lights: self.lights.clone(),
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
            lights: header.lights,
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
