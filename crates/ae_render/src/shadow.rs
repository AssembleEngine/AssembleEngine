//! The units' shadows (M7): a shadow map of the units only, in cascades.
//!
//! The hills shade by their shadow heights ([`crate::terrain::sun`]); what
//! moves, the units, is drawn from the sun into a depth map that the
//! terrain, the units and the water surface look up. The map is the
//! `shadows.map` library image, [`CASCADES`] square cascades side by side,
//! written by the `shadows.draw` library pass (every unit, not only the
//! ones in view: one just off screen casts into it).
//!
//! Each cascade covers a slice of the view by depth from the camera
//! ([`cascades`]): an orthographic box along the sun around the slice's
//! bounding sphere, its centre snapped to whole texels so the shadows do
//! not crawl as the camera pans, and stretched toward the sun so casters
//! outside the slice (and outside the view) still land in it. Depth grows
//! toward the sun (cleared to 0, the nearest kept with `GREATER_OR_EQUAL`,
//! as the main view's reverse-Z), so a receiver is lit where its depth is
//! at least the stored one. Receivers pick the first cascade that holds
//! them; past the last the shadows fade out.
//!
//! The particles' coloured shadows (`shadows.color`, written by
//! `shadows.particles`, [`crate::particles`]) share the cascades at a
//! quarter of the resolution (384 texels a side: rasterising big smoke
//! quads is the pass's cost, and smoke shadows are soft). Per texel: the
//! particles' optical depth per channel (summed, so in any order) and, in
//! alpha, the depth toward the sun of the sunward surface of the particle
//! nearest the sun (kept by a max). A receiver farther from the sun than
//! that surface multiplies its sunlight by `exp(-optical depth)`; one
//! nearer the sun (a unit above the smoke) is not tinted. Recoil (after
//! WickedEngine) multiplies a colour cleared to white instead, which is
//! the same product, but keeps no depth.

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
/// One cascade's side in texels.
pub const CASCADE_TEXELS: u32 = m::SHADOW_MAP_HEIGHT;
/// The view slices' far ends, in multiples of the camera's distance to
/// what it looks at (the RTS camera sees from about 0.7 to 2 of it on
/// flat ground).
pub const SLICES: [f32; CASCADES] = [0.95, 1.55, 2.8];
/// How far toward the sun past a slice's sphere a caster still lands in
/// its cascade (world units).
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
/// stand-ins, so receivers always have them to bind), the comparison
/// sampler receivers read the map with and the linear one for the colour.
pub struct ShadowMap {
    image: OwnedImage,
    sampler: vk::Sampler,
    color: OwnedImage,
    color_sampler: vk::Sampler,
    /// The global indices: both images, the comparison sampler and the
    /// colour sampler.
    sampled: Sampled2D,
    color_sampled: Sampled2D,
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
        let desc = ImageDesc::depth_2d(
            format(),
            w,
            h,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
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
            .compare_enable(true)
            .compare_op(vk::CompareOp::GREATER_OR_EQUAL)
            .max_lod(0.0);
        let sampler = unsafe { gpu.device.create_sampler(&info, None) }?;
        let color_sampler = unsafe { gpu.device.create_sampler(&info.compare_enable(false), None) }?;
        let sampled = bindless.sampled_2d(image.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
        let color_sampled = bindless.sampled_2d(color.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
        let sampler_index = bindless.sampler(sampler)?.index;
        let color_sampler_index = bindless.sampler(color_sampler)?.index;
        Ok(Self {
            image,
            sampler,
            color,
            color_sampler,
            sampled,
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

    /// View and comparison sampler, for `sampler2DShadow`.
    pub fn texture(&self) -> (vk::ImageView, vk::Sampler) {
        (self.image.view, self.sampler)
    }

    /// The coloured shadows' view and linear sampler (contents meaningful
    /// only in a frame whose cascades are [`Cascades::tinted`]).
    pub fn color_texture(&self) -> (vk::ImageView, vk::Sampler) {
        (self.color.view, self.color_sampler)
    }

    pub fn sampled_index(&self) -> u32 {
        self.sampled.0.index
    }

    pub fn color_sampled_index(&self) -> u32 {
        self.color_sampled.0.index
    }

    /// The comparison sampler, for `sampler2DShadow` over `sampled_index`.
    pub fn sampler_index(&self) -> u32 {
        self.sampler_index
    }

    pub fn color_sampler_index(&self) -> u32 {
        self.color_sampler_index
    }

    /// The `shadows.map` library image.
    pub fn binding(&self) -> ImageBinding {
        Self::bind(&self.image, IDENTITY)
    }

    /// The `shadows.color` library image.
    pub fn color_binding(&self) -> ImageBinding {
        Self::bind(&self.color, IDENTITY_COLOR)
    }

    fn bind(image: &OwnedImage, identity: u64) -> ImageBinding {
        let d = &image.desc;
        ImageBinding {
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
        unsafe {
            gpu.device.destroy_sampler(self.sampler, None);
            gpu.device.destroy_sampler(self.color_sampler, None);
        }
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
}

/// The cascades as the shaders' uniforms (`unit_shadows.glsl`), `std140`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CascadeUniforms {
    /// Right, up, toward the sun; `axes[0].w` 1 when the particles'
    /// coloured shadows are drawn ([`Cascades::tinted`]).
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
            axes: std::array::from_fn(|k| c.axes[k].extend(if k == 0 && c.tinted { 1.0 } else { 0.0 }).to_array()),
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

/// The cascades for a camera at `eye` looking along `forward` (unit
/// length, the up vector world +y), the tangents of its half field of view
/// across and up `tan_half`, `focus` world units from what it looks at,
/// and the sun in direction `sun`. A slice's sphere depends on the field
/// of view and the depths only, so it is the same wherever the camera is
/// and however it turns (no float noise resizing it).
pub fn cascades(eye: Vec3, forward: Vec3, tan_half: (f32, f32), focus: f32, sun: Vec3) -> Cascades {
    let sun = sun.normalize();
    let (right, up) = light_basis(sun);
    let diagonal = tan_half.0 * tan_half.0 + tan_half.1 * tan_half.1;
    let mut out = Cascades {
        matrices: [Mat4::IDENTITY; CASCADES],
        texel: [0.0; CASCADES],
        axes: [right, up, sun],
        origin: [Vec3::ZERO; CASCADES],
        scale: [Vec3::ZERO; CASCADES],
        tinted: false,
    };
    let mut near = 0.0f32;
    for (k, slice) in SLICES.iter().enumerate() {
        let far = slice * focus;
        // Centred on the axis halfway: its farthest points are the far corners.
        let centre = eye + forward * (0.5 * (near + far));
        let radius = (0.25 * (far - near) * (far - near) + diagonal * far * far).sqrt();
        let texel = 2.0 * radius / CASCADE_TEXELS as f32;
        let snap = |v: f32| (v / texel).round() * texel;
        let (cx, cy, cz) = (snap(centre.dot(right)), snap(centre.dot(up)), centre.dot(sun));
        let (zmin, zmax) = (cz - radius, cz + radius + CASTER_REACH);
        let depth = zmax - zmin;
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
        near = far;
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

    /// Panning moves the cascades by whole texels (no crawling edges), and
    /// turning the camera keeps their size.
    #[test]
    fn panning_snaps_to_texels_and_turning_keeps_the_size() {
        let (_, _, a) = camera(0.4, Vec2::new(500.0, 800.0));
        let (_, _, b) = camera(0.4, Vec2::new(503.7, 801.3));
        for k in 0..CASCADES {
            assert_eq!(a.texel[k], b.texel[k]);
            // The same world point moves by a whole number of texels.
            let p = Vec4::new(510.0, 3.0, 790.0, 1.0);
            let shift =
                ((a.matrices[k] * p).truncate() - (b.matrices[k] * p).truncate()) * (CASCADE_TEXELS as f32 / 2.0);
            for v in [shift.x, shift.y] {
                assert!((v - v.round()).abs() < 1e-2, "cascade {k}: shift {v} texels");
            }
        }
        let (_, _, c) = camera(1.9, Vec2::new(500.0, 800.0));
        assert_eq!(a.texel, c.texel);
    }
}
