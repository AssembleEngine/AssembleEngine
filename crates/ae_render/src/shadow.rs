//! Virtual shadow maps for opaque unit casters. Overlapping light clipmaps
//! share an R32_UINT physical page pool. A compute pass requests pages from
//! conservative caster bounds and assigns physical slots coarse first.
//! Pages keep their depth while the projection, physical slot and casters
//! are unchanged. Rasterization updates dirty pages with atomic max.
//!
//! Particle optical depth uses the same projections in a small dense atlas.

use crate::bindless::{Bindless, Sampled2D};
use crate::device::GpuDevice;
use crate::executor::ImageBinding;
use crate::memory::OwnedImage;
use crate::upload;
use ae_framegraph::ImageDesc;
use ae_render_abi::module as m;
use anyhow::Result;
use ash::vk;
use glam::{Mat4, Vec3, Vec4};

pub const CASCADES: usize = m::SHADOW_CASCADES as usize;
/// One virtual clipmap's side in texels.
pub const CASCADE_TEXELS: u32 = m::SHADOW_LEVEL_TEXELS;
/// Additional light-space depth toward the sun for offscreen casters.
pub const CASTER_REACH: f32 = 400.0;

const IDENTITY: u64 = (0xAE7 << 40) | (3 << 32);
const IDENTITY_COLOR: u64 = (0xAE7 << 40) | (3 << 32) | 1;

pub fn format() -> vk::Format {
    vk::Format::from_raw(m::SHADOW_MAP_FORMAT as i32)
}

/// The particles' coloured shadows' format.
pub fn color_format() -> vk::Format {
    vk::Format::from_raw(m::PARTICLE_SHADOW_FORMAT as i32)
}

/// The shadow map image and the particles' coloured shadows (or 1x1
/// stand-ins, so receivers always have them to bind), the nearest
/// sampler receivers read the map with and the linear one for the colour.
pub struct ShadowMap {
    image: OwnedImage,
    color: OwnedImage,
    /// The global indices: both images, the depth sampler and the
    /// colour sampler.
    sampled: Sampled2D,
    color_sampled: Sampled2D,
    storage_index: u32,
    sampler_index: u32,
    color_sampler_index: u32,
    real: bool,
}

impl ShadowMap {
    /// The full map when `real`, else a 1x1 stand-in. Either is left in
    /// `SHADER_READ_ONLY_OPTIMAL`, the layout receivers bind it in when a
    /// graph does not draw it.
    pub fn new(gpu: &GpuDevice, bindless: &mut Bindless, real: bool) -> Result<Self> {
        let (w, h) = if real {
            (m::SHADOW_MAP_WIDTH, m::SHADOW_MAP_HEIGHT)
        } else {
            (1, 1)
        };
        if real
            && !(gpu.fragment_stores
                && gpu
                    .optimal_format_features(format())
                    .contains(vk::FormatFeatureFlags2::STORAGE_IMAGE_ATOMIC))
        {
            anyhow::bail!(
                "virtual shadow maps require Vulkan fragmentStoresAndAtomics and R32_UINT storage image atomics"
            );
        }
        let desc = ImageDesc::color_2d(
            format(),
            w,
            h,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        );
        let image = OwnedImage::new(gpu, desc, "unit shadow map")?;
        let (cw, ch) = if real {
            (m::PARTICLE_SHADOW_WIDTH, m::PARTICLE_SHADOW_HEIGHT)
        } else {
            (1, 1)
        };
        let color_desc = ImageDesc::color_2d(
            color_format(),
            cw,
            ch,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        );
        let color = OwnedImage::new(gpu, color_desc, "particle shadows")?;
        let to_sampled = |handle, range| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(handle)
                .subresource_range(range)
        };
        let barriers = [
            to_sampled(image.image, image.range()),
            to_sampled(color.image, color.range()),
        ];
        upload::one_shot(gpu, |cmd| unsafe {
            gpu.device
                .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&barriers))
        })?;
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0);
        let sampled = bindless.sampled_2d(image.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
        let color_sampled = bindless.sampled_2d(color.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
        let storage_index = if real {
            bindless.storage_image(image.view, vk::ImageLayout::GENERAL)?.index
        } else {
            0
        };
        let sampler_index = bindless
            .sampler(&info.mag_filter(vk::Filter::NEAREST).min_filter(vk::Filter::NEAREST))?
            .index;
        let color_sampler_index = bindless.sampler(&info)?.index;
        Ok(Self {
            image,
            color,
            sampled,
            storage_index,
            color_sampled,
            sampler_index,
            color_sampler_index,
            real,
        })
    }

    /// Whether this is the full map (else the 1x1 stand-in).
    pub fn is_real(&self) -> bool {
        self.real
    }

    pub fn sampled_index(&self) -> u32 {
        self.sampled.0.index
    }

    pub fn storage_index(&self) -> u32 {
        self.storage_index
    }

    pub fn color_sampled_index(&self) -> u32 {
        self.color_sampled.0.index
    }

    /// The nearest sampler for the unsigned depth pool.
    pub fn sampler_index(&self) -> u32 {
        self.sampler_index
    }

    pub fn color_sampler_index(&self) -> u32 {
        self.color_sampler_index
    }

    /// The `shadows.map` library image.
    pub fn binding(&self) -> ImageBinding {
        ImageBinding {
            initial: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Self::bind(&self.image, IDENTITY)
        }
    }

    /// The `shadows.color` library image.
    pub fn color_binding(&self) -> ImageBinding {
        Self::bind(&self.color, IDENTITY_COLOR)
    }

    fn bind(image: &OwnedImage, identity: u64) -> ImageBinding {
        let d = &image.desc;
        ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: image.image,
            view: image.view,
            format: d.format,
            extent: vk::Extent2D {
                width: d.width,
                height: d.height,
            },
            range: image.range(),
            identity,
            sampled: None,
            storage: None,
        }
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        self.image.destroy(gpu);
        self.color.destroy(gpu);
    }
}

/// The cascades for one frame. They share the light's axes and differ in
/// centre and size only, so a shader transforms a point once and scales
/// it per cascade.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cascades {
    /// World to cascade clip space: x and y in -1..1 over the cascade,
    /// depth 0..1 growing toward the sun.
    pub matrices: [Mat4; CASCADES],
    /// World size of one texel per cascade.
    pub texel: [f32; CASCADES],
    /// The light's axes: right, up, toward the sun.
    pub axes: [Vec3; 3],
    /// Per cascade, along the axes: centre x, y and the depth origin.
    pub origin: [Vec3; CASCADES],
    /// Per cascade: 1 / half-size (x, y), 1 / depth range.
    pub scale: [Vec3; CASCADES],
    /// The particles' coloured shadows are drawn this frame (receivers
    /// read them).
    pub tinted: bool,
    /// Weight of the finest clipmap after crossing a zoom octave.
    pub finest_weight: f32,
}

/// The cascades as the shaders' uniforms (`unit_shadows.glsl`), `std140`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CascadeUniforms {
    /// Right, up, toward the sun; `axes[0].w` 1 when the particles'
    /// coloured shadows are drawn. `axes[1].w` carries the finest clipmap weight.
    axes: [[f32; 4]; 3],
    /// Per cascade: centre x, y, depth origin; `w` the world texel size.
    origin: [[f32; 4]; CASCADES],
    /// Per cascade: 1 / half-size (x, y), 1 / depth range; `w` 1 when
    /// shading by them (0: off).
    scale: [[f32; 4]; CASCADES],
}

impl Cascades {
    pub fn uniforms(cascades: Option<&Cascades>) -> CascadeUniforms {
        let Some(c) = cascades else {
            return CascadeUniforms::default();
        };
        CascadeUniforms {
            axes: [
                c.axes[0].extend(if c.tinted { 1.0 } else { 0.0 }).to_array(),
                c.axes[1].extend(c.finest_weight).to_array(),
                c.axes[2].extend(0.0).to_array(),
            ],
            origin: std::array::from_fn(|k| c.origin[k].extend(c.texel[k]).to_array()),
            scale: std::array::from_fn(|k| c.scale[k].extend(1.0).to_array()),
        }
    }
}

/// An orthonormal basis `(right, up)` across the sun's direction.
fn light_basis(sun: Vec3) -> (Vec3, Vec3) {
    let helper = if sun.y.abs() < 0.99 { Vec3::Y } else { Vec3::Z };
    let right = helper.cross(sun).normalize();
    (right, sun.cross(right))
}

/// Nested light clipmaps centered on the camera focus and snapped to pages.
/// The coarsest clipmap covers the same range of camera depths as the old
/// shadow slices. All levels share the light-space depth interval.
pub fn cascades(eye: Vec3, forward: Vec3, tan_half: (f32, f32), focus: f32, sun: Vec3) -> Cascades {
    let sun = sun.normalize();
    let (right, up) = light_basis(sun);
    let diagonal = tan_half.0.hypot(tan_half.1);
    let focus_point = eye + forward * focus;
    let coverage = focus * (1.8f32.hypot(2.8 * diagonal));
    // Quantize size in octaves so small zoom changes leave the page grid fixed.
    let size_log = (coverage * 1.05).log2();
    let octave = size_log.ceil();
    let outer_radius = octave.exp2();
    // At the octave boundary the previous finest grid becomes the new
    // middle grid. Fade in the newly available resolution over the next
    // tenth of an octave, keeping the old grid at full weight at the switch.
    let t = ((octave - size_log) / 0.1).clamp(0.0, 1.0);
    let finest_weight = t * t * (3.0 - 2.0 * t);
    let cz = focus_point.dot(sun);
    let depth_snap = outer_radius * 2.0 / m::SHADOW_LEVEL_PAGES as f32;
    let zmin = (cz / depth_snap).round() * depth_snap - outer_radius;
    let depth = 2.0 * outer_radius + CASTER_REACH;
    let mut out = Cascades {
        matrices: [Mat4::IDENTITY; CASCADES],
        texel: [0.0; CASCADES],
        axes: [right, up, sun],
        origin: [Vec3::ZERO; CASCADES],
        scale: [Vec3::ZERO; CASCADES],
        tinted: false,
        finest_weight,
    };
    for k in 0..CASCADES {
        let radius = outer_radius / (1u32 << (CASCADES - 1 - k)) as f32;
        let texel = 2.0 * radius / CASCADE_TEXELS as f32;
        let page = texel * m::SHADOW_PAGE_TEXELS as f32;
        let snap = |v: f32| (v / page).round() * page;
        let (cx, cy) = (snap(focus_point.dot(right)), snap(focus_point.dot(up)));
        let rows = [
            (right / radius).extend(-cx / radius),
            (up / radius).extend(-cy / radius),
            (sun / depth).extend(-zmin / depth),
            Vec4::W,
        ];
        out.matrices[k] = Mat4::from_cols(rows[0], rows[1], rows[2], rows[3]).transpose();
        out.texel[k] = texel;
        out.origin[k] = Vec3::new(cx, cy, zmin);
        out.scale[k] = Vec3::new(1.0 / radius, 1.0 / radius, 1.0 / depth);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::camera::Camera;
    use glam::Vec2;

    /// A camera's view-projection and its cascades.
    fn camera(yaw: f32, target: Vec2) -> (Mat4, Vec3, Cascades) {
        let c = Camera {
            target,
            ground: 0.0,
            distance: 300.0,
            yaw,
            pitch: 0.75,
            fov_y: 0.8,
        };
        let eye = c.eye(|_| 0.0);
        let tan = (c.fov_y * 0.5).tan();
        let cascades = cascades(
            eye,
            (c.target3() - eye).normalize(),
            (tan * 16.0 / 9.0, tan),
            c.distance,
            crate::terrain::sun::direction(),
        );
        (c.proj(16.0 / 9.0) * c.view(eye), eye, cascades)
    }

    /// Ground the camera sees (flat, at height 0) falls in a cascade, in
    /// range, and depth grows toward the sun.
    #[test]
    fn the_view_lands_in_the_cascades() {
        let sun = crate::terrain::sun::direction();
        let (vp, eye, c) = camera(0.4, Vec2::new(500.0, 800.0));
        let inv = vp.inverse();
        for y in [-0.9f32, -0.5, 0.0, 0.5, 0.9] {
            for x in [-0.95f32, 0.0, 0.95] {
                // The ground point under this pixel.
                let p = inv * Vec4::new(x, y, 1.0, 1.0);
                let p = p.truncate() / p.w;
                let dir = (p - eye).normalize();
                let ground = eye + dir * (-eye.y / dir.y);
                let held = c.matrices.iter().any(|m| {
                    let q = *m * ground.extend(1.0);
                    q.x.abs() < 1.0 && q.y.abs() < 1.0 && (0.0..=1.0).contains(&q.z)
                });
                assert!(held, "pixel ({x}, {y}) at {ground:?} is in no cascade");
            }
        }
        let m = c.matrices[0];
        let below = (m * Vec4::new(500.0, 0.0, 800.0, 1.0)).z;
        let above = (m * (Vec3::new(500.0, 0.0, 800.0) + sun * 10.0).extend(1.0)).z;
        assert!(above > below, "depth grows toward the sun");
        assert!(c.texel[0] < c.texel[1] && c.texel[1] < c.texel[2], "{:?}", c.texel);
    }

    #[test]
    fn zoom_octave_retains_the_previous_grid_during_the_transition() {
        let tan_half = (0.75f32, 0.42f32);
        let boundary = 128.0 / (1.05 * 1.8f32.hypot(2.8 * tan_half.0.hypot(tan_half.1)));
        let target = Vec3::new(2_900.0, 50.0, 17_900.0);
        let forward = Vec3::new(0.0, -0.6, 0.8);
        let at = |focus| {
            cascades(
                target - forward * focus,
                forward,
                tan_half,
                focus,
                crate::terrain::sun::direction(),
            )
        };
        let before = at(boundary * 1.0001);
        let after = at(boundary * 0.9999);
        assert_eq!(before.finest_weight, 1.0);
        assert!(after.finest_weight < 0.001);
        assert_eq!(before.texel[0], after.texel[1]);
        assert_eq!(before.origin[0].truncate(), after.origin[1].truncate());
        assert_eq!(at(boundary * 0.9).finest_weight, 1.0);
    }

    /// Panning moves the clipmaps by whole pages, and rotating keeps their size.
    #[test]
    fn panning_snaps_to_pages_and_turning_keeps_the_size() {
        let (_, _, a) = camera(0.4, Vec2::new(500.0, 800.0));
        let (_, _, b) = camera(0.4, Vec2::new(503.7, 801.3));
        for k in 0..CASCADES {
            assert_eq!(a.texel[k], b.texel[k]);
            // The same world point moves by a whole number of pages.
            let p = Vec4::new(510.0, 3.0, 790.0, 1.0);
            let shift = ((a.matrices[k] * p).truncate() - (b.matrices[k] * p).truncate())
                * (m::SHADOW_LEVEL_PAGES as f32 / 2.0);
            for v in [shift.x, shift.y] {
                assert!((v - v.round()).abs() < 1e-2, "cascade {k}: shift {v} pages");
            }
        }
        let (_, _, c) = camera(1.9, Vec2::new(500.0, 800.0));
        assert_eq!(a.texel, c.texel);
    }
}
