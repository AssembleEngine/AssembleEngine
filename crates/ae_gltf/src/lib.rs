//! glTF 2.0 (`.gltf`, `.glb`) to the engine model ([`ae_model::Model`]).
//!
//! What is read and what is not (docs/PLAN.md, M11):
//! * the default scene's nodes become bones under a root that turns glTF's
//!   axes into the engine's ([`ae_model::GLTF_TO_ENGINE`]); a mesh hanging
//!   on a node is skinned to that node's bone at weight 1, a skinned mesh
//!   to its skin's joints;
//! * the metallic-roughness material with every texture, alpha modes,
//!   double-sided, samplers, `KHR_texture_transform` (kept per texture
//!   reference), `KHR_materials_emissive_strength` (multiplied in),
//!   `KHR_texture_basisu` ([`textures`]);
//! * POSITION, NORMAL (flat normals where absent), TANGENT (MikkTSpace
//!   where absent and a normal map needs it), TEXCOORD_0/1, COLOR_0,
//!   JOINTS_0/1 + WEIGHTS_0/1 (the four strongest kept, renormalised);
//!   sparse accessors;
//! * animations as clips (translation, rotation, scale; their keys and
//!   interpolation as they are);
//! * `KHR_lights_punctual`: point and spot lights become
//!   [`ae_model::ModelLight`]s on the joint of the node carrying them
//!   (their colour already in the engine's units). Directional lights are
//!   ignored with a warning naming them: the engine has one fixed sun.
//!
//! Warned and dropped: TEXCOORD_2+, COLOR_1+, more than four influences,
//! morph targets and their weight channels, points and lines, cameras,
//! directional lights, unknown optional extensions. Refused: any required
//! extension besides the four above (Draco and meshopt among them).

mod mesh;
pub mod textures;

use ae_model::{
    AlphaMode, Bone, Bounds, Channel, Clip, DEFAULT_SOURCE_RADIUS, Filter, GLTF_TO_ENGINE, Interpolation, Joint,
    LightKind, Material, Model, ModelLight, Path as BonePath, Primitive, Range, Sampler, TextureRef, UvTransform, Wrap,
    pose,
};
use anyhow::{Context as _, Result, bail, ensure};
use base64::Engine as _;
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use std::path::Path;

/// Required extensions the converter handles.
pub const ACCEPTED_EXTENSIONS: [&str; 4] = [
    "KHR_texture_transform",
    "KHR_materials_emissive_strength",
    "KHR_texture_basisu",
    "KHR_lights_punctual",
];

pub struct Converted {
    pub model: Model,
    /// What was dropped or approximated, one line each.
    pub warnings: Vec<String>,
}

/// Converts the file at `path` (external buffers and images relative to it).
pub fn convert_file(path: &Path) -> Result<Converted> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    convert(&bytes, path.parent())
}

/// Converts glTF or GLB bytes; `base` resolves relative URIs (`None`: only
/// embedded data allowed).
pub fn convert(bytes: &[u8], base: Option<&Path>) -> Result<Converted> {
    let gltf = gltf::Gltf::from_slice_without_validation(bytes).context("glTF parse")?;
    let blob = gltf.blob;
    let json = gltf.document.into_json();
    for ext in &json.extensions_required {
        if !ACCEPTED_EXTENSIONS.contains(&ext.as_str()) {
            bail!(
                "glTF requires extension {ext}, which the converter does not take (taken: {})",
                ACCEPTED_EXTENSIONS.join(", ")
            );
        }
    }
    let mut warnings = Vec::new();
    for ext in &json.extensions_used {
        if !ACCEPTED_EXTENSIONS.contains(&ext.as_str()) {
            warnings.push(format!("extension {ext} ignored"));
        }
    }
    let doc = gltf::Document::from_json(json).context("glTF validation")?;
    let buffers = load_buffers(&doc, blob, base)?;
    let mut c = Converter {
        doc: &doc,
        buffers: &buffers,
        base,
        warnings,
        model: Model::default(),
        node_bone: HashMap::new(),
        joint_of: HashMap::new(),
        texture_of: HashMap::new(),
        material_of: HashMap::new(),
    };
    c.run()?;
    let mut model = c.model;
    model.merge_primitives();
    model.validate().context("converted model")?;
    Ok(Converted {
        model,
        warnings: c.warnings,
    })
}

fn read_uri(uri: &str, base: Option<&Path>, what: &str) -> Result<Vec<u8>> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let (_, data) = rest
            .split_once(";base64,")
            .with_context(|| format!("{what}: data URI not base64"))?;
        return base64::engine::general_purpose::STANDARD
            .decode(data)
            .with_context(|| format!("{what}: base64"));
    }
    let base = base.with_context(|| format!("{what}: external file {uri} but no base directory"))?;
    let decoded = percent_decode(uri);
    std::fs::read(base.join(&decoded)).with_context(|| format!("{what}: read {decoded}"))
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16)
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn load_buffers(doc: &gltf::Document, mut blob: Option<Vec<u8>>, base: Option<&Path>) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::new();
    for b in doc.buffers() {
        let data = match b.source() {
            gltf::buffer::Source::Bin => blob.take().context("GLB buffer without a BIN chunk")?,
            gltf::buffer::Source::Uri(uri) => read_uri(uri, base, &format!("buffer {}", b.index()))?,
        };
        ensure!(
            data.len() >= b.length(),
            "buffer {}: {} bytes, declared {}",
            b.index(),
            data.len(),
            b.length()
        );
        out.push(data);
    }
    Ok(out)
}

struct Converter<'a> {
    doc: &'a gltf::Document,
    buffers: &'a [Vec<u8>],
    base: Option<&'a Path>,
    warnings: Vec<String>,
    model: Model,
    /// glTF node -> bone.
    node_bone: HashMap<usize, u32>,
    /// (bone, inverse bind bits) -> joint.
    joint_of: HashMap<(u32, [u32; 16]), u16>,
    /// (glTF image, sRGB) -> model texture.
    texture_of: HashMap<(usize, bool), u32>,
    /// glTF material (None: the default) -> model material.
    material_of: HashMap<Option<usize>, u32>,
}

impl<'a> Converter<'a> {
    fn buffer(&self) -> impl Clone + Fn(gltf::Buffer<'_>) -> Option<&'a [u8]> + 'a {
        let buffers = self.buffers;
        move |b: gltf::Buffer<'_>| buffers.get(b.index()).map(|v| v.as_slice())
    }

    fn warn(&mut self, w: String) {
        if !self.warnings.contains(&w) {
            self.warnings.push(w);
        }
    }

    fn run(&mut self) -> Result<()> {
        let root = Mat4::from_quat(GLTF_TO_ENGINE);
        let (_, r, _) = root.to_scale_rotation_translation();
        self.model.bones.push(Bone {
            name: "__gltf_root".into(),
            parent: None,
            rest: root.to_cols_array(),
            translation: [0.0; 3],
            rotation: r.to_array(),
            scale: [1.0; 3],
        });
        let roots: Vec<gltf::Node> = match self.doc.default_scene().or_else(|| self.doc.scenes().next()) {
            Some(s) => s.nodes().collect(),
            None => {
                // No scene: every node that is nobody's child.
                let mut child = vec![false; self.doc.nodes().len()];
                for n in self.doc.nodes() {
                    for c in n.children() {
                        child[c.index()] = true;
                    }
                }
                self.doc.nodes().filter(|n| !child[n.index()]).collect()
            }
        };
        let mut order = Vec::new();
        for n in roots {
            self.add_bones(n, 0, &mut order)?;
        }
        if self.doc.cameras().len() > 0 {
            self.warn("cameras ignored".into());
        }
        for node in order {
            if let Some(m) = node.mesh() {
                self.add_mesh(&node, &m)?;
            }
        }
        ensure!(!self.model.primitives.is_empty(), "no triangles in the scene");
        for a in self.doc.animations() {
            self.add_clip(&a)?;
        }
        self.model.bounds = self.rest_bounds();
        Ok(())
    }

    fn add_bones<'d>(&mut self, node: gltf::Node<'d>, parent: u32, order: &mut Vec<gltf::Node<'d>>) -> Result<()> {
        ensure!(
            !self.node_bone.contains_key(&node.index()),
            "node {} reached twice (not a tree)",
            node.index()
        );
        let bone = self.model.bones.len() as u32;
        self.node_bone.insert(node.index(), bone);
        let rest = Mat4::from_cols_array_2d(&node.transform().matrix());
        let (t, r, s) = node.transform().decomposed();
        let name = node
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("node{}", node.index()));
        if let Some(light) = node.light() {
            let what = format!("light {} on node {}", light.name().unwrap_or("(unnamed)"), name);
            let kind = light.kind();
            let intensity = light.intensity();
            let color = light.color();
            let range = light.range();
            match kind {
                gltf::khr_lights_punctual::Kind::Directional => {
                    // The engine has one sun; a directional light on a unit
                    // would swing it, and instancing would make one per unit.
                    self.warn(format!("{what}: directional light ignored (the engine's sun is fixed)"));
                }
                kind => {
                    let joint = self.joint(bone, Mat4::IDENTITY)?;
                    let (kind, inner_angle, outer_angle) = match kind {
                        gltf::khr_lights_punctual::Kind::Spot {
                            inner_cone_angle,
                            outer_cone_angle,
                        } => {
                            let outer = outer_cone_angle.clamp(1e-3, std::f32::consts::FRAC_PI_2);
                            (LightKind::Spot, inner_cone_angle.clamp(0.0, outer), outer)
                        }
                        _ => (LightKind::Point, 0.0, 0.0),
                    };
                    let range = match range {
                        Some(r) if r.is_finite() && r > 0.0 => r,
                        Some(r) => {
                            self.warn(format!("{what}: range {r} is not positive; treated as no cutoff"));
                            0.0
                        }
                        None => 0.0,
                    };
                    self.model.lights.push(ModelLight {
                        name: what,
                        joint: joint.into(),
                        kind,
                        // glTF's intensity is already candela (point and
                        // spot); the colour is the linear tint.
                        color,
                        intensity: intensity.max(0.0),
                        range,
                        inner_angle,
                        outer_angle,
                        // The source radius regularises the near field; glTF
                        // has none, so a range-derived size or a small
                        // default.
                        size: if range > 0.0 {
                            (range / 8.0).max(0.05)
                        } else {
                            DEFAULT_SOURCE_RADIUS
                        },
                    });
                }
            }
        }
        self.model.bones.push(Bone {
            name,
            parent: Some(parent),
            rest: rest.to_cols_array(),
            translation: t,
            rotation: r,
            scale: s,
        });
        order.push(node.clone());
        for c in node.children() {
            self.add_bones(c, bone, order)?;
        }
        Ok(())
    }

    fn joint(&mut self, bone: u32, inverse_bind: Mat4) -> Result<u16> {
        let bits = inverse_bind.to_cols_array().map(f32::to_bits);
        if let Some(&j) = self.joint_of.get(&(bone, bits)) {
            return Ok(j);
        }
        let j = u16::try_from(self.model.joints.len()).context("more than 65536 joints")?;
        self.model.joints.push(Joint {
            bone,
            inverse_bind: inverse_bind.to_cols_array(),
        });
        self.joint_of.insert((bone, bits), j);
        Ok(j)
    }

    /// Per skin joint, the model's joint.
    fn skin_joints(&mut self, skin: &gltf::Skin) -> Result<Vec<u16>> {
        let reader = skin.reader(self.buffer());
        let ibms: Vec<Mat4> = match reader.read_inverse_bind_matrices() {
            Some(it) => it.map(|m| Mat4::from_cols_array_2d(&m)).collect(),
            None => vec![Mat4::IDENTITY; skin.joints().len()],
        };
        ensure!(
            ibms.len() >= skin.joints().len(),
            "skin {}: {} inverse bind matrices for {} joints",
            skin.index(),
            ibms.len(),
            skin.joints().len()
        );
        let mut out = Vec::new();
        for (j, node) in skin.joints().enumerate() {
            let bone = *self
                .node_bone
                .get(&node.index())
                .with_context(|| format!("skin {}: joint node {} not in the scene", skin.index(), node.index()))?;
            out.push(self.joint(bone, ibms[j])?);
        }
        Ok(out)
    }

    fn add_mesh(&mut self, node: &gltf::Node, m: &gltf::Mesh) -> Result<()> {
        let bone = self.node_bone[&node.index()];
        let skin_joints = match node.skin() {
            Some(skin) => Some(self.skin_joints(&skin)?),
            None => None,
        };
        for (pi, prim) in m.primitives().enumerate() {
            let what = format!("mesh {} ({}) primitive {pi}", m.index(), m.name().unwrap_or(""));
            let material = self.material(prim.material())?;
            let tangent_uv = self.model.materials[material as usize].normal_texture.map(|t| t.uv);
            let source = mesh::read_primitive(&prim, self.buffer(), &what, &mut self.warnings)?;
            let Some(source) = source else { continue };
            let joints = match (&skin_joints, source.has_joints) {
                (Some(j), true) => mesh::Joints::Skin(j.clone()),
                (Some(_), false) => {
                    self.warn(format!(
                        "{what}: on a skinned node but without JOINTS_0/WEIGHTS_0; drawn as a piece"
                    ));
                    mesh::Joints::Piece(self.joint(bone, Mat4::IDENTITY)?)
                }
                (None, true) => {
                    self.warn(format!(
                        "{what}: has JOINTS_0 but its node has no skin; drawn as a piece"
                    ));
                    mesh::Joints::Piece(self.joint(bone, Mat4::IDENTITY)?)
                }
                (None, false) => mesh::Joints::Piece(self.joint(bone, Mat4::IDENTITY)?),
            };
            let (vertices, indices) = mesh::build(source, &joints, tangent_uv, &what, &mut self.warnings)?;
            if indices.is_empty() {
                continue;
            }
            self.model.primitives.push(Primitive {
                material,
                layout: ae_model::Layout::Skinned,
                first_vertex: self.model.vertices.len() as u32,
                vertex_count: vertices.len() as u32,
                first_index: self.model.indices.len() as u32,
                index_count: indices.len() as u32,
            });
            self.model.vertices.extend(vertices);
            self.model.indices.extend(indices);
        }
        Ok(())
    }

    fn material(&mut self, m: gltf::Material) -> Result<u32> {
        if let Some(&i) = self.material_of.get(&m.index()) {
            return Ok(i);
        }
        let pbr = m.pbr_metallic_roughness();
        let what = format!("material {}", m.name().unwrap_or("(default)"));
        let base_color_texture = match pbr.base_color_texture() {
            Some(t) => Some(self.texture_ref(t.texture(), t.tex_coord(), t.texture_transform(), true, &what)?),
            None => None,
        };
        let metallic_roughness_texture = match pbr.metallic_roughness_texture() {
            Some(t) => Some(self.texture_ref(t.texture(), t.tex_coord(), t.texture_transform(), false, &what)?),
            None => None,
        };
        let (normal_texture, normal_scale) = match m.normal_texture() {
            Some(t) => {
                let tr = transform_of(t.extension_value("KHR_texture_transform"));
                (
                    Some(self.texture_ref_raw(t.texture(), t.tex_coord(), tr, false, &what)?),
                    t.scale(),
                )
            }
            None => (None, 1.0),
        };
        let (occlusion_texture, occlusion_strength) = match m.occlusion_texture() {
            Some(t) => {
                let tr = transform_of(t.extension_value("KHR_texture_transform"));
                (
                    Some(self.texture_ref_raw(t.texture(), t.tex_coord(), tr, false, &what)?),
                    t.strength(),
                )
            }
            None => (None, 1.0),
        };
        let emissive_texture = match m.emissive_texture() {
            Some(t) => Some(self.texture_ref(t.texture(), t.tex_coord(), t.texture_transform(), true, &what)?),
            None => None,
        };
        let strength = m.emissive_strength().unwrap_or(1.0);
        let alpha = match m.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask {
                cutoff: m.alpha_cutoff().unwrap_or(0.5),
            },
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        };
        let material = Material {
            name: m.name().unwrap_or("").to_owned(),
            base_color: pbr.base_color_factor(),
            base_color_texture,
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            metallic_roughness_texture,
            normal_texture,
            normal_scale,
            occlusion_texture,
            occlusion_strength,
            emissive: m.emissive_factor().map(|c| c * strength),
            emissive_texture,
            alpha,
            double_sided: m.double_sided(),
        };
        let i = self.model.materials.len() as u32;
        self.model.materials.push(material);
        self.material_of.insert(m.index(), i);
        Ok(i)
    }

    fn texture_ref(
        &mut self,
        t: gltf::Texture,
        uv: u32,
        tr: Option<gltf::texture::TextureTransform>,
        srgb: bool,
        what: &str,
    ) -> Result<TextureRef> {
        let tr = tr.map(|tr| {
            (
                UvTransform {
                    offset: tr.offset(),
                    rotation: tr.rotation(),
                    scale: tr.scale(),
                },
                tr.tex_coord(),
            )
        });
        self.texture_ref_raw(t, uv, tr, srgb, what)
    }

    fn texture_ref_raw(
        &mut self,
        t: gltf::Texture,
        uv: u32,
        tr: Option<(UvTransform, Option<u32>)>,
        srgb: bool,
        what: &str,
    ) -> Result<TextureRef> {
        // KHR_texture_basisu's source first, the fallback otherwise.
        let basis = t
            .extension_value("KHR_texture_basisu")
            .and_then(|v| v.get("source"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let image = match (basis, t.source()) {
            (Some(i), _) => i,
            (None, Some(img)) => img.index(),
            (None, None) => bail!("{what}: texture {} has no source", t.index()),
        };
        let texture = self.image(image, srgb)?;
        let mut uv = tr.and_then(|(_, o)| o).unwrap_or(uv);
        if uv > 1 {
            self.warn(format!("{what}: TEXCOORD_{uv} not kept; uses TEXCOORD_0"));
            uv = 0;
        }
        let s = t.sampler();
        let filter = |f: Option<gltf::texture::MagFilter>| match f {
            Some(gltf::texture::MagFilter::Nearest) => Filter::Nearest,
            _ => Filter::Linear,
        };
        use gltf::texture::MinFilter as M;
        let (min, mip) = match s.min_filter() {
            None | Some(M::LinearMipmapLinear) => (Filter::Linear, Some(Filter::Linear)),
            Some(M::Nearest) => (Filter::Nearest, None),
            Some(M::Linear) => (Filter::Linear, None),
            Some(M::NearestMipmapNearest) => (Filter::Nearest, Some(Filter::Nearest)),
            Some(M::LinearMipmapNearest) => (Filter::Linear, Some(Filter::Nearest)),
            Some(M::NearestMipmapLinear) => (Filter::Nearest, Some(Filter::Linear)),
        };
        let wrap = |w: gltf::texture::WrappingMode| match w {
            gltf::texture::WrappingMode::Repeat => Wrap::Repeat,
            gltf::texture::WrappingMode::MirroredRepeat => Wrap::MirroredRepeat,
            gltf::texture::WrappingMode::ClampToEdge => Wrap::ClampToEdge,
        };
        let sampler = Sampler {
            mag: filter(s.mag_filter()),
            min,
            mip,
            wrap_s: wrap(s.wrap_s()),
            wrap_t: wrap(s.wrap_t()),
        };
        Ok(TextureRef {
            texture,
            uv,
            sampler,
            transform: tr.map(|(t, _)| t),
        })
    }

    fn image(&mut self, index: usize, srgb: bool) -> Result<u32> {
        if let Some(&t) = self.texture_of.get(&(index, srgb)) {
            return Ok(t);
        }
        let img = self
            .doc
            .images()
            .nth(index)
            .with_context(|| format!("image {index} out of range"))?;
        let name = img.name().map(str::to_owned).unwrap_or_else(|| format!("image{index}"));
        let what = format!("image {index} ({name})");
        let bytes = match img.source() {
            gltf::image::Source::View { view, .. } => {
                let buf = &self.buffers[view.buffer().index()];
                buf.get(view.offset()..view.offset() + view.length())
                    .with_context(|| format!("{what}: view out of its buffer"))?
                    .to_vec()
            }
            gltf::image::Source::Uri { uri, .. } => read_uri(uri, self.base, &what)?,
        };
        let ktx2 = textures::to_ktx2(&bytes, srgb, &mut self.warnings, &what)?;
        let t = self.model.textures.len() as u32;
        self.model.textures.push(ktx2);
        self.model
            .texture_names
            .push(if srgb { name } else { format!("{name} (linear)") });
        self.texture_of.insert((index, srgb), t);
        Ok(t)
    }

    fn add_clip(&mut self, a: &gltf::Animation) -> Result<()> {
        let name = a
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("clip{}", a.index()));
        let mut channels = Vec::new();
        let mut duration = 0.0f32;
        for ch in a.channels() {
            let target = ch.target();
            let Some(&bone) = self.node_bone.get(&target.node().index()) else {
                self.warn(format!(
                    "clip {name}: a channel on node {} outside the scene, dropped",
                    target.node().index()
                ));
                continue;
            };
            let reader = ch.reader(self.buffer());
            let times: Vec<f32> = reader
                .read_inputs()
                .with_context(|| format!("clip {name}: no key times"))?
                .collect();
            use gltf::animation::util::ReadOutputs as O;
            let (path, values): (BonePath, Vec<f32>) = match reader
                .read_outputs()
                .with_context(|| format!("clip {name}: no key values"))?
            {
                O::Translations(it) => (BonePath::Translation, it.flatten().collect()),
                O::Rotations(it) => (BonePath::Rotation, it.into_f32().flatten().collect()),
                O::Scales(it) => (BonePath::Scale, it.flatten().collect()),
                O::MorphTargetWeights(_) => {
                    self.warn(format!("clip {name}: morph target weights dropped"));
                    continue;
                }
            };
            let interpolation = match ch.sampler().interpolation() {
                gltf::animation::Interpolation::Step => Interpolation::Step,
                gltf::animation::Interpolation::Linear => Interpolation::Linear,
                gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
            };
            let per = path.components()
                * if interpolation == Interpolation::CubicSpline {
                    3
                } else {
                    1
                };
            ensure!(
                !times.is_empty() && values.len() == times.len() * per,
                "clip {name}: {} keys, {} values",
                times.len(),
                values.len()
            );
            ensure!(
                times.windows(2).all(|w| w[0] <= w[1]),
                "clip {name}: key times not increasing"
            );
            duration = duration.max(*times.last().unwrap());
            let data = &mut self.model.clip_data;
            let t = Range {
                offset: data.len() as u64,
                len: times.len() as u64,
            };
            data.extend(&times);
            let v = Range {
                offset: data.len() as u64,
                len: values.len() as u64,
            };
            data.extend(&values);
            channels.push(Channel {
                bone,
                path,
                interpolation,
                times: t,
                values: v,
            });
        }
        if !channels.is_empty() {
            self.model.clips.push(Clip {
                name,
                duration,
                channels,
            });
        }
        Ok(())
    }

    fn rest_bounds(&self) -> Bounds {
        let palette = pose::rest_palette(&self.model);
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for v in &self.model.vertices {
            let p = Vec3::from(v.position);
            let mut q = Vec3::ZERO;
            for k in 0..4 {
                let w = v.weights[k] as f32 / 65535.0;
                if w > 0.0 {
                    q += palette[v.joints[k] as usize].transform_point3(p) * w;
                }
            }
            min = min.min(q);
            max = max.max(q);
        }
        Bounds {
            min: min.to_array(),
            max: max.to_array(),
        }
    }
}

/// `KHR_texture_transform` from a raw extension value (for the normal and
/// occlusion infos, which the gltf crate gives no accessor for).
fn transform_of(v: Option<&gltf::json::Value>) -> Option<(UvTransform, Option<u32>)> {
    let v = v?;
    let pair = |key: &str, d: [f32; 2]| {
        v.get(key)
            .and_then(|a| a.as_array())
            .map(|a| {
                [
                    a[0].as_f64().unwrap_or(d[0] as f64) as f32,
                    a[1].as_f64().unwrap_or(d[1] as f64) as f32,
                ]
            })
            .unwrap_or(d)
    };
    let t = UvTransform {
        offset: pair("offset", [0.0, 0.0]),
        rotation: v.get("rotation").and_then(|r| r.as_f64()).unwrap_or(0.0) as f32,
        scale: pair("scale", [1.0, 1.0]),
    };
    Some((t, v.get("texCoord").and_then(|t| t.as_u64()).map(|t| t as u32)))
}

#[cfg(test)]
mod tests;
