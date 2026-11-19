//! Engine models (`ae_model`, M11) on the GPU, for the units: the static
//! part of a units renderer whose classes have models. GPU-driven and
//! bindless: one multi-draw per pipeline draws every model.
//!
//! * One geometry buffer holds every model's vertices, packed per
//!   primitive in its layout (rigid or skinned, `ae_model::Layout`, read by
//!   vertex pulling in `shaders/units_models.glsl`), then every index.
//! * Materials are one table; textures one partially bound descriptor
//!   array the materials index.
//! * Each (class, primitive) is one `VkDrawIndexedIndirectCommand` with a
//!   draw record (material, class, vertex base and layout, the class's
//!   region of the visible list) that the shaders find by `gl_DrawID`.
//!   Commands are ordered opaque first, then alpha-masked (a pipeline that
//!   discards loses early depth), so the camera draws with two multi-draws.
//! * Poses: the palette (`units.palette`) holds 3x4 matrices (rows). Its
//!   head is each model's rest palette, written once; a unit at rest
//!   points at it. A unit with animation records (its channels at two
//!   ticks, `crate::units::RECORD_CELLS`) gets a range of its own, which
//!   the pose pass (`shaders/units_pose.comp`) fills each frame: each bone
//!   evaluated from its model's clips as the sim's `Rig::pose` does, lerped
//!   between the records at the drawn time, the hierarchy walked, the
//!   inverse binds applied, once per unit and bone (vertices only blend
//!   matrices). The animation table ([`ModelSet::anim`], [`animation`])
//!   holds the bones, joints, and the rig's clips (the sim's bake), masks
//!   and rest transforms. A model with clips is culled with a sphere that
//!   covers every clip frame; a unit with manual joints shown with one
//!   that also covers any bone turned any way ([`turned_sphere`]).
//! * A model is fitted to the unit: its footprint radius (the largest
//!   distance of its rest bounds from the origin along x or z) is drawn at
//!   the unit's drawn size, as the dart's body radius is. Model space is
//!   the engine's (+X forward, +Y up, +Z right), the dart's local frame.
//!
//! Materials (step 2 of M11): base colour (factor × vertex colour ×
//! texture), alpha mask, normal map, occlusion and emissive; the units'
//! lighting (sun, sky, shadows, fog). Metallic-roughness is not used yet;
//! blended materials are drawn as masks at 0.5.

use crate::device::GpuDevice;
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::pipeline;
use crate::upload;
use ae_model::anim::Rig;
use ae_model::{AlphaMode, Filter, Model, Sampler, TextureRef, Wrap, ktx};
use anyhow::{Context as _, Result, ensure};
use ash::vk;
use glam::Mat4;
use std::sync::Arc;

/// Texture descriptors in the models' set (partially bound; slot 0: a
/// white texel).
pub const MAX_TEXTURES: u32 = 1024;
/// Bytes of one `VkDrawIndexedIndirectCommand`.
pub const COMMAND_BYTES: u64 = 20;
/// Bytes before the first model command in the indirect buffer: the darts'
/// `VkDrawIndirectCommand`.
pub const DART_COMMAND_BYTES: u64 = 16;
/// Bytes of a palette matrix: 3x4, the rows of an affine transform.
pub const PALETTE_ENTRY_BYTES: u64 = 48;
/// The cull sphere of a model with clips covers its clips' frames and this
/// much more (motion between frames).
const ANIMATED_MARGIN: f32 = 1.05;

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuMaterial {
    base_color: [f32; 4],
    /// rgb; w: alpha cutoff (negative: opaque).
    emissive: [f32; 4],
    /// Normal scale, occlusion strength, double-sided, unused.
    params: [f32; 4],
    /// Texture slots: base colour, normal, occlusion, emissive (0: none).
    tex: [u32; 4],
    /// UV set of each.
    uv: [u32; 4],
    /// Per texture two vec4: the 2x2 matrix (columns), the offset.
    xform: [[f32; 4]; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuClass {
    /// Bounding sphere (at rest, or over its clips), in fitted units
    /// (centre, radius).
    sphere: [f32; 4],
    /// First entry in the class-commands list, commands (0: no model, a
    /// dart), rest palette offset, joints.
    cmd: [u32; 4],
    /// x: 1 / footprint radius; y: the radius of the sphere a unit with
    /// manual joints is culled with (0: as the others), z: its centre as a
    /// multiple of `sphere`'s ([`turned_sphere`]).
    fit: [f32; 4],
    /// In the animation table (vec4s): the first bone, bones, the first
    /// joint, the clips' head.
    pose: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuDraw {
    material: u32,
    class: u32,
    visible_base: u32,
    /// First word of the primitive's vertices in the geometry buffer.
    vertex_base: u32,
    /// Words per vertex (its layout).
    words: u32,
    pad: [u32; 3],
}

/// One command: a primitive of a class's model.
#[derive(Clone, Copy, Debug)]
pub struct Draw {
    pub class: u32,
    pub index_count: u32,
    pub first_index: u32,
}

/// A model light as `units_lights.glsl` reads it (32 bytes): the colour and
/// range, the spot's cosines and source size, and the joint and kind.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuModelLight {
    /// rgb; w: the range (metres).
    color: [f32; 4],
    /// x, y: a spot's inner and outer cosines; z: the source size; w: the
    /// joint in the low 16 bits, the kind (1 point, 2 spot) above it
    /// (`f32` bits).
    params: [f32; 4],
}

/// A unit class drawn with a model.
#[derive(Clone, Debug)]
pub struct ModelClass {
    pub class: u32,
    /// Its model's rest palette (palette entries).
    pub palette_offset: u32,
    pub joints: u32,
    pub bones: u32,
    /// Its region of the visible list.
    pub visible_base: u32,
    /// Its commands (indices into [`ModelSet::draws`]).
    pub commands: Vec<u32>,
}

pub struct ModelSet {
    /// Vertex words, then the indices from [`Self::index_offset`].
    pub geometry: OwnedBuffer,
    pub index_offset: u64,
    pub materials: OwnedBuffer,
    pub class_table: OwnedBuffer,
    pub draw_table: OwnedBuffer,
    pub class_commands: OwnedBuffer,
    /// The animation table, vec4s: per model its bones, joints, clips,
    /// masks and rest transforms ([`animation`]).
    pub anim: OwnedBuffer,
    /// How many lights the classes have (`units_lights.comp` appends them).
    pub light_count: u32,
    /// Per unit class, the range of its model's lights in the table
    /// (first, count); the host's light offsets use it.
    pub class_lights: Vec<[u32; 2]>,
    /// `class_lights` and `model_lights` as the models' set binds them.
    pub class_light_table: OwnedBuffer,
    pub model_light_table: OwnedBuffer,
    images: Vec<OwnedImage>,
    samplers: Vec<vk::Sampler>,
    /// Texture descriptors (slot 0 first); the array's rest is unbound.
    pub slots: Vec<(vk::ImageView, vk::Sampler)>,
    pub classes: Vec<ModelClass>,
    /// Every command, opaque ones first.
    pub draws: Vec<Draw>,
    /// Commands drawn by the opaque pipeline (the rest: the masked one).
    pub opaque: u32,
    /// The models' rest palettes: palette entries before the units' ranges.
    pub rest_len: u32,
    /// The most joints and bones of a model with joints (each posed unit
    /// needs that many palette entries and as many scratch ones).
    pub max_joints: u32,
    pub max_bones: u32,
    /// The rest palettes, 12 floats per entry.
    rest: Vec<f32>,
    /// The indirect buffer's initial contents: the darts' command, then
    /// the models' (instance counts 0), then the models' again for each
    /// shadow cascade.
    pub template: Vec<u8>,
    /// Model class (index into `classes`) of each unit class.
    pub class_model: Vec<Option<usize>>,
}

/// The sphere a unit with manual joints shown is culled with: the one
/// enclosing the model's `sphere` (centre, radius) and the sphere about the
/// model's origin of radius `reach` (`ae_model::pose::turned_reach`: any
/// bone turned any way), 1% larger; as (radius, centre as a multiple of
/// `sphere`'s centre). (0, 0) without reach (a model without joints).
pub fn turned_sphere(sphere: [f32; 4], reach: f32) -> [f32; 2] {
    if reach <= 0.0 {
        return [0.0, 0.0];
    }
    let (d, r) = (glam::Vec3::new(sphere[0], sphere[1], sphere[2]).length(), sphere[3]);
    let (radius, along) = if reach >= d + r {
        (reach, 0.0)
    } else if r >= d + reach {
        (r, 1.0)
    } else {
        let radius = 0.5 * (d + r + reach);
        (radius, (radius - reach) / d)
    };
    [radius * 1.01, along]
}

/// glTF's sampler as Vulkan's.
fn sampler_info(s: &Sampler) -> vk::SamplerCreateInfo<'static> {
    let filter = |f: Filter| match f {
        Filter::Nearest => vk::Filter::NEAREST,
        Filter::Linear => vk::Filter::LINEAR,
    };
    let wrap = |w: Wrap| match w {
        Wrap::Repeat => vk::SamplerAddressMode::REPEAT,
        Wrap::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        Wrap::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
    };
    vk::SamplerCreateInfo::default()
        .mag_filter(filter(s.mag))
        .min_filter(filter(s.min))
        .mipmap_mode(match s.mip {
            Some(Filter::Linear) => vk::SamplerMipmapMode::LINEAR,
            _ => vk::SamplerMipmapMode::NEAREST,
        })
        .address_mode_u(wrap(s.wrap_s))
        .address_mode_v(wrap(s.wrap_t))
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        // No mipmapping: the base level only.
        .max_lod(if s.mip.is_some() { vk::LOD_CLAMP_NONE } else { 0.0 })
}

/// Footprint radius and bounding sphere (centre, radius) of `model` at
/// rest, in model units.
pub fn fit(model: &Model) -> (f32, [f32; 4]) {
    let (min, max) = (glam::Vec3::from(model.bounds.min), glam::Vec3::from(model.bounds.max));
    let footprint = ae_model::footprint(&model.bounds);
    let centre = (min + max) * 0.5;
    let radius = (max - min).length() * 0.5;
    (footprint, [centre.x, centre.y, centre.z, radius])
}

/// The rows of an affine matrix (its last row dropped): a palette entry.
pub fn rows(m: Mat4) -> [f32; 12] {
    let mut out = [0.0; 12];
    for i in 0..3 {
        out[i * 4..i * 4 + 4].copy_from_slice(&m.row(i).to_array());
    }
    out
}

/// Append `model`'s bones and joints, and `rig`'s clips, masks and rest
/// transforms, to the animation table; its `GpuClass::pose`. The layout
/// (vec4s; "words" are 32-bit, four a vec4, indexed from the table's
/// start; `shaders/units_models.glsl`):
/// * bones: rest matrix rows 0..2, then (parent, 0, 0, 0);
/// * joints: inverse bind rows 0..2, then (bone, 0, 0, 0);
/// * at `pose.w`, the clips' head: (clips, masks, first rest vec4, mask
///   word), then per clip two vec4s: (frames, rate, duration, tracks),
///   (first key vec4, track-of word, 0, 0);
/// * per clip a word per bone: its track, or `u32::MAX` (not animated);
/// * per mask a word per bone: its weight (`f32` bits);
/// * per bone its rest transform: (translation, 0), rotation, (scale, 0);
/// * per clip its keys, frame by frame, track by track: three vec4s as
///   the rest transforms.
///
/// The clips are `rig`'s bake ([`ae_model::anim::Rig::clip_table`]): the
/// sim's own keys, not baked again here.
pub fn animation(model: &Model, rig: &Rig, anim: &mut Vec<[f32; 4]>) -> [u32; 4] {
    let word = |u: u32| f32::from_bits(u);
    let bone_base = anim.len() as u32;
    for b in &model.bones {
        let r = rows(Mat4::from_cols_array(&b.rest));
        anim.extend([[r[0], r[1], r[2], r[3]], [r[4], r[5], r[6], r[7]], [r[8], r[9], r[10], r[11]], [word(b.parent.unwrap_or(u32::MAX)), 0.0, 0.0, 0.0]]);
    }
    let joint_base = anim.len() as u32;
    for j in &model.joints {
        let r = rows(Mat4::from_cols_array(&j.inverse_bind));
        anim.extend([[r[0], r[1], r[2], r[3]], [r[4], r[5], r[6], r[7]], [r[8], r[9], r[10], r[11]], [word(j.bone), 0.0, 0.0, 0.0]]);
    }
    let bones = rig.bones();
    let clips: Vec<_> = (1..=rig.clip_names().len() as u16).filter_map(|c| rig.clip_table(c)).collect();
    let masks = rig.masks();
    // Words after the head, in vec4s: track-of words, mask words.
    let head = anim.len();
    let words_at = head + 1 + 2 * clips.len();
    let word_count = (clips.len() + masks.len()) * bones;
    let rest_at = words_at + word_count.div_ceil(4);
    let mut keys_at = rest_at + 3 * bones;
    anim.push([word(clips.len() as u32), word(masks.len() as u32), word(rest_at as u32), word((4 * words_at + clips.len() * bones) as u32)]);
    for (c, clip) in clips.iter().enumerate() {
        anim.push([word(clip.frames), clip.rate, clip.duration, word(clip.tracks.len() as u32)]);
        anim.push([word(keys_at as u32), word((4 * words_at + c * bones) as u32), 0.0, 0.0]);
        keys_at += 3 * clip.keys.len();
    }
    let mut words = vec![u32::MAX; word_count.next_multiple_of(4)];
    for (c, clip) in clips.iter().enumerate() {
        for (i, &b) in clip.tracks.iter().enumerate() {
            words[c * bones + b as usize] = i as u32;
        }
    }
    for (k, m) in masks.iter().enumerate() {
        for (b, w) in m.weights.iter().enumerate() {
            words[(clips.len() + k) * bones + b] = w.to_bits();
        }
    }
    anim.extend(words.chunks(4).map(|w| [word(w[0]), word(w[1]), word(w[2]), word(w[3])]));
    let trs = |p: &ae_model::anim::Trs| [[p.t[0], p.t[1], p.t[2], 0.0], p.r, [p.s[0], p.s[1], p.s[2], 0.0]];
    anim.extend(rig.rest().iter().flat_map(trs));
    for clip in &clips {
        anim.extend(clip.keys.iter().flat_map(trs));
    }
    [bone_base, model.bones.len() as u32, joint_base, head as u32]
}

/// Images and samplers made so far, destroyed if building fails.
struct Textures<'a> {
    gpu: &'a GpuDevice,
    images: Vec<OwnedImage>,
    samplers: Vec<(Sampler, vk::Sampler)>,
    /// (unique model, texture) → image.
    image_of: Vec<(usize, u32, usize)>,
    /// (image, sampler) per descriptor slot from 1.
    slots: Vec<(usize, usize)>,
}

impl Textures<'_> {
    fn sampler(&mut self, s: &Sampler) -> Result<usize> {
        if let Some(i) = self.samplers.iter().position(|(k, _)| k == s) {
            return Ok(i);
        }
        self.samplers.push((*s, unsafe { self.gpu.device.create_sampler(&sampler_info(s), None) }?));
        Ok(self.samplers.len() - 1)
    }

    /// The descriptor slot (from 1) of `r`, the UV set and the transform.
    fn slot(&mut self, model: &Model, m: usize, r: &Option<TextureRef>) -> Result<(u32, u32, [[f32; 4]; 2])> {
        let Some(r) = r else { return Ok((0, 0, [[1.0, 0.0, 0.0, 1.0], [0.0; 4]])) };
        let image = match self.image_of.iter().find(|&&(mm, t, _)| mm == m && t == r.texture) {
            Some(&(_, _, i)) => i,
            None => {
                let bytes = model.textures.get(r.texture as usize).context("texture index")?;
                let tex = ktx::read(bytes).with_context(|| format!("texture {}", r.texture))?;
                let compressed = tex.format.block().is_some_and(|(w, _)| w > 1);
                ensure!(!compressed || self.gpu.texture_compression_bc, "texture {} is block-compressed and the device has no textureCompressionBC", r.texture);
                let levels: Vec<&[u8]> = tex.levels.iter().map(Vec::as_slice).collect();
                let format = vk::Format::from_raw(tex.format.0 as i32);
                self.images.push(upload::image(self.gpu, format, tex.width, tex.height, &levels, "model texture")?);
                self.image_of.push((m, r.texture, self.images.len() - 1));
                self.images.len() - 1
            }
        };
        let sampler = self.sampler(&r.sampler)?;
        let slot = match self.slots.iter().position(|&s| s == (image, sampler)) {
            Some(i) => i,
            None => {
                self.slots.push((image, sampler));
                self.slots.len() - 1
            }
        };
        ensure!(slot + 1 < MAX_TEXTURES as usize, "more than {} textures and samplers in the units' models", MAX_TEXTURES - 1);
        let t = r.transform.map_or([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], |t| t.matrix());
        Ok(((slot + 1) as u32, r.uv, [[t[0][0], t[0][1], t[1][0], t[1][1]], [t[2][0], t[2][1], 0.0, 0.0]]))
    }

    fn destroy(&mut self) {
        for image in &mut self.images {
            image.destroy(self.gpu);
        }
        for (_, s) in self.samplers.drain(..) {
            unsafe { self.gpu.device.destroy_sampler(s, None) };
        }
    }
}

fn gpu_material(mat: &ae_model::Material, tex: &mut Textures<'_>, model: &Model, m: usize) -> Result<GpuMaterial> {
    let mut g = GpuMaterial {
        base_color: mat.base_color,
        emissive: [mat.emissive[0], mat.emissive[1], mat.emissive[2], -1.0],
        params: [mat.normal_scale, mat.occlusion_strength, mat.double_sided as u32 as f32, 0.0],
        tex: [0; 4],
        uv: [0; 4],
        xform: [[0.0; 4]; 8],
    };
    let refs = [&mat.base_color_texture, &mat.normal_texture, &mat.occlusion_texture, &mat.emissive_texture];
    for (k, r) in refs.into_iter().enumerate() {
        let (slot, uv, x) = tex.slot(model, m, r)?;
        g.tex[k] = slot;
        g.uv[k] = uv;
        g.xform[2 * k] = x[0];
        g.xform[2 * k + 1] = x[1];
    }
    match mat.alpha {
        AlphaMode::Opaque => {}
        AlphaMode::Mask { cutoff } => g.emissive[3] = cutoff,
        AlphaMode::Blend => {
            eprintln!("[models] material `{}` blends: drawn as a mask at 0.5 (blending is not drawn yet)", mat.name);
            g.emissive[3] = 0.5;
        }
    }
    Ok(g)
}

impl ModelSet {
    /// Regions of the visible list per model class: the camera's, then
    /// one per shadow cascade, each `capacity` rows.
    pub const REGIONS: u32 = 1 + crate::shadow::CASCADES as u32;

    /// Entries of the visible list: the darts' region, then the model
    /// classes' regions.
    pub fn visible_len(&self, capacity: u32) -> u64 {
        capacity as u64 * (1 + Self::REGIONS as u64 * self.classes.len() as u64)
    }

    /// `models[class]` for each unit class (`None`: a dart), posed by
    /// `rigs[class]` (the sim's rig of that model: its clips and masks;
    /// none or beyond the list: one made from the model, without masks);
    /// `capacity` units per visible-list region.
    pub fn new(gpu: &GpuDevice, models: &[Option<Arc<Model>>], rigs: &[Option<Arc<Rig>>], capacity: u32, max_classes: usize) -> Result<Self> {
        ensure!(gpu.gpu_driven, "the units' models need multiDrawIndirect, shaderDrawParameters and descriptorBindingPartiallyBound");
        // The fragment stage sees the models' texture array and the units'
        // four samplers.
        let limits = unsafe { gpu.instance.instance.get_physical_device_properties(gpu.physical) }.limits;
        let need = MAX_TEXTURES + 4;
        ensure!(
            limits.max_per_stage_descriptor_samplers >= need && limits.max_per_stage_descriptor_sampled_images >= need,
            "the units' models need {need} samplers per shader stage; the device allows {} samplers and {} sampled images",
            limits.max_per_stage_descriptor_samplers,
            limits.max_per_stage_descriptor_sampled_images
        );
        let mut tex = Textures { gpu, images: Vec::new(), samplers: Vec::new(), image_of: Vec::new(), slots: Vec::new() };
        match Self::build(gpu, models, rigs, capacity, max_classes, &mut tex) {
            Ok(set) => Ok(set),
            Err(err) => {
                tex.destroy();
                Err(err)
            }
        }
    }

    fn build(gpu: &GpuDevice, models: &[Option<Arc<Model>>], rigs: &[Option<Arc<Rig>>], capacity: u32, max_classes: usize, tex: &mut Textures<'_>) -> Result<Self> {
        // Unique models: their vertex words, indices and materials.
        let mut unique: Vec<Arc<Model>> = Vec::new();
        let mut words: Vec<u32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut materials: Vec<GpuMaterial> = Vec::new();
        // Per unique model: per primitive (vertex base word, first index);
        // the base material.
        let mut placed: Vec<(Vec<(u32, u32)>, u32)> = Vec::new();
        // Per unique model: its rest palette offset, its animation table
        // entry (`GpuClass::pose`), its cull sphere, its lights.
        struct Posed {
            rest_offset: u32,
            pose: [u32; 4],
            sphere: [f32; 4],
            turned: [f32; 2],
            lights: Vec<GpuModelLight>,
        }
        let mut posed: Vec<Posed> = Vec::new();
        let mut anim: Vec<[f32; 4]> = Vec::new();
        let mut rest: Vec<f32> = Vec::new();
        let (mut max_joints, mut max_bones) = (0u32, 0u32);
        for (c, model) in models.iter().enumerate().take(max_classes) {
            let Some(model) = model else { continue };
            if unique.iter().any(|u| Arc::ptr_eq(u, model)) {
                continue;
            }
            let m = unique.len();
            model.validate().context("model")?;
            ensure!(!model.primitives.is_empty(), "a unit model without triangles");
            let mut prims = Vec::new();
            for p in &model.primitives {
                prims.push((words.len() as u32, indices.len() as u32));
                words.extend(model.gpu_words(p));
                indices.extend_from_slice(&model.indices[p.first_index as usize..(p.first_index + p.index_count) as usize]);
            }
            placed.push((prims, materials.len() as u32));
            for mat in &model.materials {
                materials.push(gpu_material(mat, tex, model, m)?);
            }
            let rest_offset = (rest.len() / 12) as u32;
            rest.extend(ae_model::pose::rest_palette(model).iter().flat_map(|p| rows(*p)));
            let rig = match rigs.get(c) {
                Some(Some(rig)) => rig.clone(),
                _ => Arc::new(Rig::new(model)),
            };
            ensure!(rig.bones() == model.bones.len(), "a rig of {} bones for a model of {}", rig.bones(), model.bones.len());
            let pose = animation(model, &rig, &mut anim);
            let (footprint, mut sphere) = fit(model);
            if !model.clips.is_empty() {
                let reach = ae_model::pose::animated_radius(model, glam::Vec3::new(sphere[0], sphere[1], sphere[2]));
                sphere[3] = sphere[3].max(reach * ANIMATED_MARGIN);
            }
            if !model.joints.is_empty() {
                max_joints = max_joints.max(model.joints.len() as u32);
                max_bones = max_bones.max(model.bones.len() as u32);
            }
            let sphere = sphere.map(|v| v / footprint);
            let turned = turned_sphere(sphere, ae_model::pose::turned_reach(model) / footprint);
            let lights = model
                .lights
                .iter()
                .map(|l| GpuModelLight {
                    color: [l.color[0], l.color[1], l.color[2], l.range],
                    params: [
                        if l.kind == ae_model::LightKind::Spot { l.inner_angle.cos() } else { 1.0 },
                        if l.kind == ae_model::LightKind::Spot { l.outer_angle.cos() } else { 1.0 },
                        l.size.max(1e-3),
                        f32::from_bits(l.joint | ((if l.kind == ae_model::LightKind::Spot { 2 } else { 1 }) << 16)),
                    ],
                })
                .collect();
            posed.push(Posed { rest_offset, pose, sphere, turned, lights });
            unique.push(model.clone());
        }

        // Classes, palette ranges, commands (opaque first) and records.
        let mut classes: Vec<ModelClass> = Vec::new();
        let mut table = vec![GpuClass::default(); max_classes];
        let mut class_model = vec![None; max_classes];
        let mut class_lights = vec![[0u32; 2]; max_classes];
        let mut model_lights: Vec<GpuModelLight> = Vec::new();
        // (masked, class, draw record, command)
        let mut pending: Vec<(bool, usize, GpuDraw, Draw)> = Vec::new();
        for (c, model) in models.iter().take(max_classes).enumerate() {
            let Some(model) = model else { continue };
            let m = unique.iter().position(|u| Arc::ptr_eq(u, model)).unwrap();
            let (prims, base_material) = &placed[m];
            // The darts' region, then per model class the camera's region
            // and one per shadow cascade.
            let visible_base = capacity * (1 + Self::REGIONS * classes.len() as u32);
            for (p, &(vertex_base, first_index)) in model.primitives.iter().zip(prims) {
                let masked = !matches!(model.materials[p.material as usize].alpha, AlphaMode::Opaque);
                let record = GpuDraw { material: base_material + p.material, class: c as u32, visible_base, vertex_base, words: p.layout.words() as u32, pad: [0; 3] };
                pending.push((masked, classes.len(), record, Draw { class: c as u32, index_count: p.index_count, first_index }));
            }
            let (footprint, _) = fit(model);
            let Posed { rest_offset, pose, sphere, turned, lights } = &posed[m];
            let joints = model.joints.len() as u32;
            table[c] = GpuClass { sphere: *sphere, cmd: [0, model.primitives.len() as u32, *rest_offset, joints], fit: [1.0 / footprint, turned[0], turned[1], 0.0], pose: *pose };
            if !lights.is_empty() {
                class_lights[c] = [model_lights.len() as u32, lights.len() as u32];
                model_lights.extend_from_slice(lights);
            }
            class_model[c] = Some(classes.len());
            classes.push(ModelClass {
                class: c as u32,
                palette_offset: *rest_offset,
                joints,
                bones: model.bones.len() as u32,
                visible_base,
                commands: Vec::new(),
            });
        }
        // Stable: within each pipeline, classes and primitives in order.
        pending.sort_by_key(|&(masked, ..)| masked);
        let opaque = pending.iter().filter(|(masked, ..)| !masked).count() as u32;
        let mut template: Vec<u8> = Vec::new();
        for w in [crate::units::UNIT_VERTICES, 0, 0, 0] {
            template.extend_from_slice(&w.to_le_bytes());
        }
        let (mut records, mut draws) = (Vec::new(), Vec::new());
        for (k, (_, mc, record, draw)) in pending.into_iter().enumerate() {
            classes[mc].commands.push(k as u32);
            for w in [draw.index_count, 0, draw.first_index, 0, 0] {
                template.extend_from_slice(&w.to_le_bytes());
            }
            records.push(record);
            draws.push(draw);
        }
        // Each cascade's copy of the commands (counted by the cull, drawn by
        // the shadow caster).
        let camera = template[DART_COMMAND_BYTES as usize..].to_vec();
        for _ in 0..crate::shadow::CASCADES {
            template.extend_from_slice(&camera);
        }
        let mut class_commands = Vec::new();
        for mc in &classes {
            table[mc.class as usize].cmd[0] = class_commands.len() as u32;
            class_commands.extend_from_slice(&mc.commands);
        }

        // Slot 0: a white texel.
        tex.images.push(upload::image(gpu, vk::Format::R8G8B8A8_UNORM, 1, 1, &[&[255; 4]], "model white")?);
        let white = (tex.images.len() - 1, tex.sampler(&Sampler::default())?);
        let slots: Vec<_> = std::iter::once(white).chain(tex.slots.iter().copied()).map(|(i, s)| (tex.images[i].view, tex.samplers[s].1)).collect();

        // The geometry: vertex words, then the indices (16-byte aligned).
        let mut geometry: Vec<u8> = pipeline::slice_bytes(&words).to_vec();
        geometry.resize(geometry.len().next_multiple_of(16), 0);
        let index_offset = geometry.len() as u64;
        geometry.extend_from_slice(pipeline::slice_bytes(&indices));
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let mut made: Vec<OwnedBuffer> = Vec::new();
        let result = (|| -> Result<()> {
            made.push(upload::buffer(gpu, &geometry, storage | vk::BufferUsageFlags::INDEX_BUFFER, "model geometry")?);
            made.push(upload::buffer(gpu, pipeline::slice_bytes(&materials), storage, "model materials")?);
            made.push(upload::buffer(gpu, pipeline::slice_bytes(&table), storage, "model classes")?);
            made.push(upload::buffer(gpu, pipeline::slice_bytes(&records), storage, "model draws")?);
            made.push(upload::buffer(gpu, pipeline::slice_bytes(&class_commands), storage, "model class commands")?);
            made.push(upload::buffer(gpu, pipeline::slice_bytes(&anim), storage, "model animation")?);
            made.push(upload::buffer(gpu, pipeline::slice_bytes(&class_lights), storage, "model class lights")?);
            made.push(upload::buffer(gpu, pipeline::slice_bytes(&model_lights), storage, "model lights")?);
            Ok(())
        })();
        if let Err(err) = result {
            for b in &mut made {
                b.destroy(gpu);
            }
            return Err(err);
        }
        let [geometry, materials, class_table, draw_table, class_commands, anim, class_light_table, model_light_table]: [OwnedBuffer; 8] = made.try_into().ok().unwrap();
        Ok(Self {
            geometry,
            index_offset,
            materials,
            class_table,
            draw_table,
            class_commands,
            anim,
            light_count: model_lights.len() as u32,
            class_lights,
            class_light_table,
            model_light_table,
            images: std::mem::take(&mut tex.images),
            samplers: tex.samplers.drain(..).map(|(_, s)| s).collect(),
            slots,
            classes,
            draws,
            opaque,
            rest_len: (rest.len() / 12) as u32,
            max_joints,
            max_bones,
            rest,
            template,
            class_model,
        })
    }

    /// Commands among the models' (after the darts').
    pub fn commands(&self) -> u32 {
        self.draws.len() as u32
    }

    /// The palette buffer's head: each model's rest palette (written once;
    /// units at rest point at it).
    pub fn rest_palettes(&self) -> &[u8] {
        pipeline::slice_bytes(&self.rest)
    }

    /// The palette's layout for `capacity` units: the first entry of the
    /// units' ranges, of the scratch ranges (the pose pass's bone
    /// matrices), and the entries in all.
    pub fn palette_layout(&self, capacity: u32) -> (u64, u64, u64) {
        let ranges = self.rest_len as u64;
        let scratch = ranges + capacity as u64 * self.max_joints as u64;
        (ranges, scratch, scratch + capacity as u64 * self.max_bones as u64)
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        for image in &mut self.images {
            image.destroy(gpu);
        }
        for &s in &self.samplers {
            unsafe { gpu.device.destroy_sampler(s, None) };
        }
        self.samplers.clear();
        for b in [
            &mut self.geometry,
            &mut self.materials,
            &mut self.class_table,
            &mut self.draw_table,
            &mut self.class_commands,
            &mut self.anim,
            &mut self.class_light_table,
            &mut self.model_light_table,
        ] {
            b.destroy(gpu);
        }
    }
}
