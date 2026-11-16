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
//! * Each class has a palette range of its own (its pose); each unit is
//!   given a palette offset per frame (`rows`), so per-unit poses need no
//!   change in the shaders. For now every unit of a class points at the
//!   class's rest palette.
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
use ae_framegraph::BufferDesc;
use ae_model::{AlphaMode, Filter, Model, Sampler, TextureRef, Wrap, ktx};
use anyhow::{Context as _, Result, ensure};
use ash::vk;
use glam::Mat4;
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

/// Texture descriptors in the models' set (partially bound; slot 0: a
/// white texel).
pub const MAX_TEXTURES: u32 = 1024;
/// Bytes of one `VkDrawIndexedIndirectCommand`.
pub const COMMAND_BYTES: u64 = 20;
/// Bytes before the first model command in the indirect buffer: the darts'
/// `VkDrawIndirectCommand`.
pub const DART_COMMAND_BYTES: u64 = 16;

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
    /// Bounding sphere at rest, in fitted units (centre, radius).
    sphere: [f32; 4],
    /// First entry in the class-commands list, commands (0: no model, a
    /// dart), palette offset, unused.
    cmd: [u32; 4],
    /// x: 1 / footprint radius.
    fit: [f32; 4],
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

/// A unit class drawn with a model.
#[derive(Clone, Debug)]
pub struct ModelClass {
    pub class: u32,
    pub palette_offset: u32,
    /// Its region of the visible list.
    pub visible_base: u32,
    /// Its commands (indices into [`ModelSet::draws`]).
    pub commands: Vec<u32>,
    /// Its rest palette (until poses are played).
    pub rest: Vec<Mat4>,
}

pub struct ModelSet {
    /// Vertex words, then the indices from [`Self::index_offset`].
    pub geometry: OwnedBuffer,
    pub index_offset: u64,
    pub materials: OwnedBuffer,
    pub class_table: OwnedBuffer,
    pub draw_table: OwnedBuffer,
    pub class_commands: OwnedBuffer,
    images: Vec<OwnedImage>,
    samplers: Vec<vk::Sampler>,
    /// Texture descriptors (slot 0 first); the array's rest is unbound.
    pub slots: Vec<(vk::ImageView, vk::Sampler)>,
    pub classes: Vec<ModelClass>,
    /// Every command, opaque ones first.
    pub draws: Vec<Draw>,
    /// Commands drawn by the opaque pipeline (the rest: the masked one).
    pub opaque: u32,
    /// Joints of all classes: the palette's length.
    pub palette_len: u32,
    /// The indirect buffer's initial contents: the darts' command, then
    /// the models' (instance counts 0).
    pub template: Vec<u8>,
    /// Model class (index into `classes`) of each unit class.
    pub class_model: Vec<Option<usize>>,
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
    let footprint = [min.x, max.x, min.z, max.z].iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let footprint = if footprint > 1e-6 { footprint } else { (max - min).length().max(1e-3) };
    let centre = (min + max) * 0.5;
    let radius = (max - min).length() * 0.5;
    (footprint, [centre.x, centre.y, centre.z, radius])
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
    /// `models[class]` for each unit class (`None`: a dart); `capacity`
    /// units per visible-list region.
    pub fn new(gpu: &GpuDevice, models: &[Option<Arc<Model>>], capacity: u32, max_classes: usize) -> Result<Self> {
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
        match Self::build(gpu, models, capacity, max_classes, &mut tex) {
            Ok(set) => Ok(set),
            Err(err) => {
                tex.destroy();
                Err(err)
            }
        }
    }

    fn build(gpu: &GpuDevice, models: &[Option<Arc<Model>>], capacity: u32, max_classes: usize, tex: &mut Textures<'_>) -> Result<Self> {
        // Unique models: their vertex words, indices and materials.
        let mut unique: Vec<Arc<Model>> = Vec::new();
        let mut words: Vec<u32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut materials: Vec<GpuMaterial> = Vec::new();
        // Per unique model: per primitive (vertex base word, first index);
        // the base material.
        let mut placed: Vec<(Vec<(u32, u32)>, u32)> = Vec::new();
        for model in models.iter().take(max_classes).flatten() {
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
            unique.push(model.clone());
        }

        // Classes, palette ranges, commands (opaque first) and records.
        let mut classes: Vec<ModelClass> = Vec::new();
        let mut table = vec![GpuClass::default(); max_classes];
        let mut class_model = vec![None; max_classes];
        // (masked, class, draw record, command)
        let mut pending: Vec<(bool, usize, GpuDraw, Draw)> = Vec::new();
        let mut palette_len = 0u32;
        for (c, model) in models.iter().take(max_classes).enumerate() {
            let Some(model) = model else { continue };
            let m = unique.iter().position(|u| Arc::ptr_eq(u, model)).unwrap();
            let (prims, base_material) = &placed[m];
            let visible_base = capacity * (1 + classes.len() as u32);
            for (p, &(vertex_base, first_index)) in model.primitives.iter().zip(prims) {
                let masked = !matches!(model.materials[p.material as usize].alpha, AlphaMode::Opaque);
                let record = GpuDraw { material: base_material + p.material, class: c as u32, visible_base, vertex_base, words: p.layout.words() as u32, pad: [0; 3] };
                pending.push((masked, classes.len(), record, Draw { class: c as u32, index_count: p.index_count, first_index }));
            }
            let (footprint, sphere) = fit(model);
            table[c] = GpuClass { sphere: sphere.map(|v| v / footprint), cmd: [0, model.primitives.len() as u32, palette_len, 0], fit: [1.0 / footprint, 0.0, 0.0, 0.0] };
            class_model[c] = Some(classes.len());
            classes.push(ModelClass { class: c as u32, palette_offset: palette_len, visible_base, commands: Vec::new(), rest: ae_model::pose::rest_palette(model) });
            palette_len += model.joints.len() as u32;
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
            Ok(())
        })();
        if let Err(err) = result {
            for b in &mut made {
                b.destroy(gpu);
            }
            return Err(err);
        }
        let [geometry, materials, class_table, draw_table, class_commands]: [OwnedBuffer; 5] = made.try_into().ok().unwrap();
        Ok(Self {
            geometry,
            index_offset,
            materials,
            class_table,
            draw_table,
            class_commands,
            images: std::mem::take(&mut tex.images),
            samplers: tex.samplers.drain(..).map(|(_, s)| s).collect(),
            slots,
            classes,
            draws,
            opaque,
            palette_len,
            template,
            class_model,
        })
    }

    /// Bytes of one slot's palette buffer.
    pub fn palette_bytes(&self) -> u64 {
        (self.palette_len.max(1) as u64) * 64
    }

    /// Commands among the models' (after the darts').
    pub fn commands(&self) -> u32 {
        self.draws.len() as u32
    }

    /// A host-visible per-slot buffer for the palettes.
    pub fn palette_buffer(&self, gpu: &GpuDevice) -> Result<OwnedBuffer> {
        OwnedBuffer::new(gpu, BufferDesc { size: self.palette_bytes(), usage: vk::BufferUsageFlags::STORAGE_BUFFER }, MemoryLocation::CpuToGpu, "model palette")
    }

    /// Write each class's palette (its rest pose for now) into `bytes`.
    pub fn write_palettes(&self, bytes: &mut [u8]) {
        for c in &self.classes {
            let at = c.palette_offset as usize * 64;
            let floats: Vec<f32> = c.rest.iter().flat_map(|m| m.to_cols_array()).collect();
            bytes[at..at + floats.len() * 4].copy_from_slice(pipeline::slice_bytes(&floats));
        }
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        for image in &mut self.images {
            image.destroy(gpu);
        }
        for &s in &self.samplers {
            unsafe { gpu.device.destroy_sampler(s, None) };
        }
        self.samplers.clear();
        for b in [&mut self.geometry, &mut self.materials, &mut self.class_table, &mut self.draw_table, &mut self.class_commands] {
            b.destroy(gpu);
        }
    }
}
