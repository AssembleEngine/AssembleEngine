//! RTS-style orbit camera: pans over the map, zooms with the wheel.

use glam::{Mat4, Vec2, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Ground point looked at (world x, z).
    pub target: Vec2,
    pub ground: f32,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
}

impl Camera {
    pub fn target3(&self) -> Vec3 {
        Vec3::new(self.target.x, self.ground, self.target.y)
    }

    pub fn offset_dir(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * sy, sp, cp * cy)
    }

    pub fn eye(&self, terrain_height: impl Fn(Vec2) -> f32) -> Vec3 {
        let mut e = self.target3() + self.offset_dir() * self.distance;
        e.y = e.y.max(terrain_height(Vec2::new(e.x, e.z)) + 4.0);
        e
    }

    /// Horizontal unit vectors (forward = screen up, right = screen right) as world (x, z).
    pub fn ground_axes(&self) -> (Vec2, Vec2) {
        let (s, c) = self.yaw.sin_cos();
        (Vec2::new(-s, -c), Vec2::new(c, -s))
    }

    pub fn view(&self, eye: Vec3) -> Mat4 {
        Mat4::look_at_rh(eye, self.target3(), Vec3::Y)
    }

    /// Reverse-Z infinite projection for Vulkan (y down in clip space).
    pub fn proj(&self, aspect: f32) -> Mat4 {
        let near = (self.distance * 0.002).clamp(0.25, 20.0);
        let mut p = Mat4::perspective_infinite_reverse_rh(self.fov_y, aspect, near);
        p.y_axis.y = -p.y_axis.y;
        p
    }

    /// Pixels per world unit at unit distance.
    pub fn proj_scale(&self, viewport_height: f32) -> f32 {
        viewport_height / (2.0 * (self.fov_y * 0.5).tan())
    }
}
