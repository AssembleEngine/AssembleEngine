//! Poses: the bones' local transforms, from the rest pose or a clip, to
//! model-space bone matrices and the skinning palette.

use crate::{Interpolation, Model, Path, Vertex};
use glam::{Mat4, Quat, Vec3, Vec4};

/// Local transforms of every bone. A bone a clip touched is composed from
/// its TRS; any other keeps its rest matrix (which may not be a TRS).
#[derive(Clone, Debug, PartialEq)]
pub struct Pose {
    pub translation: Vec<Vec3>,
    pub rotation: Vec<Quat>,
    pub scale: Vec<Vec3>,
    pub animated: Vec<bool>,
}

impl Pose {
    pub fn rest(model: &Model) -> Pose {
        Pose {
            translation: model.bones.iter().map(|b| Vec3::from(b.translation)).collect(),
            rotation: model.bones.iter().map(|b| Quat::from_array(b.rotation)).collect(),
            scale: model.bones.iter().map(|b| Vec3::from(b.scale)).collect(),
            animated: vec![false; model.bones.len()],
        }
    }

    /// Clip `clip` at `time` seconds (held at its first and last keys
    /// outside them; the caller wraps for a loop) over this pose.
    pub fn sample(&mut self, model: &Model, clip: usize, time: f32) {
        for ch in &model.clips[clip].channels {
            let times = model.clip_floats(ch.times);
            let values = model.clip_floats(ch.values);
            let n = ch.path.components();
            let v = sample_channel(times, values, n, ch.interpolation, ch.path == Path::Rotation, time);
            let b = ch.bone as usize;
            match ch.path {
                Path::Translation => self.translation[b] = v.truncate(),
                Path::Rotation => self.rotation[b] = Quat::from_vec4(v).normalize(),
                Path::Scale => self.scale[b] = v.truncate(),
            }
            self.animated[b] = true;
        }
    }

    /// Local matrices.
    pub fn locals(&self, model: &Model) -> Vec<Mat4> {
        (0..model.bones.len())
            .map(|b| {
                if self.animated[b] {
                    Mat4::from_scale_rotation_translation(self.scale[b], self.rotation[b], self.translation[b])
                } else {
                    Mat4::from_cols_array(&model.bones[b].rest)
                }
            })
            .collect()
    }
}

/// Model-space bone matrices from local ones (parents come first).
pub fn globals(model: &Model, locals: &[Mat4]) -> Vec<Mat4> {
    let mut g: Vec<Mat4> = Vec::with_capacity(locals.len());
    for (b, bone) in model.bones.iter().enumerate() {
        let m = match bone.parent {
            Some(p) => g[p as usize] * locals[b],
            None => locals[b],
        };
        g.push(m);
    }
    g
}

/// The skinning palette: per joint, its bone's matrix times its inverse
/// bind matrix.
pub fn palette(model: &Model, globals: &[Mat4]) -> Vec<Mat4> {
    model.joints.iter().map(|j| globals[j.bone as usize] * Mat4::from_cols_array(&j.inverse_bind)).collect()
}

/// The rest pose's palette.
pub fn rest_palette(model: &Model) -> Vec<Mat4> {
    let pose = Pose::rest(model);
    palette(model, &globals(model, &pose.locals(model)))
}

/// Frames per second clips are baked at, at least: each clip's rate is
/// rounded up so its last frame falls on its duration.
pub const BAKE_RATE: f32 = 60.0;

/// A bone's local transform as translation, rotation and scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trs {
    pub t: Vec3,
    pub r: Quat,
    pub s: Vec3,
}

/// A clip sampled at a fixed rate ([`bake`]): per frame, the local TRS of
/// each bone the clip animates (a track). Played by blending two frames
/// (lerp, and nlerp for the rotation): what the renderer's pose pass does
/// on the GPU, and [`BakedClip::locals`] on the CPU. Clips loop.
#[derive(Clone, Debug, PartialEq)]
pub struct BakedClip {
    /// The bone of each track, increasing.
    pub tracks: Vec<u32>,
    pub frames: u32,
    /// Frames per second (0 with a single frame).
    pub rate: f32,
    pub duration: f32,
    /// `[frame][track]`; rotations of consecutive frames on one hemisphere
    /// (so a plain nlerp takes the short way).
    pub keys: Vec<Trs>,
}

/// Clip `clip` sampled with [`Pose::sample`] at [`BAKE_RATE`] or a little
/// more, from 0 to its duration inclusive.
pub fn bake(model: &Model, clip: usize) -> BakedClip {
    let c = &model.clips[clip];
    let mut tracks: Vec<u32> = c.channels.iter().map(|ch| ch.bone).collect();
    tracks.sort_unstable();
    tracks.dedup();
    let (frames, rate) = match c.duration > 0.0 {
        true => {
            let n = (c.duration * BAKE_RATE).ceil().max(1.0) as u32;
            (n + 1, n as f32 / c.duration)
        }
        false => (1, 0.0),
    };
    let mut keys: Vec<Trs> = Vec::with_capacity(frames as usize * tracks.len());
    for f in 0..frames {
        let t = if f + 1 == frames { c.duration } else { f as f32 / rate };
        let mut pose = Pose::rest(model);
        pose.sample(model, clip, t);
        for (i, &b) in tracks.iter().enumerate() {
            let b = b as usize;
            let mut r = pose.rotation[b];
            if f > 0 && keys[(f as usize - 1) * tracks.len() + i].r.dot(r) < 0.0 {
                r = -r;
            }
            keys.push(Trs { t: pose.translation[b], r, s: pose.scale[b] });
        }
    }
    BakedClip { tracks, frames, rate, duration: c.duration, keys }
}

impl BakedClip {
    /// The two frames around `time` seconds (wrapped into the clip: clips
    /// loop) and the second one's weight; as the pose pass computes them.
    pub fn frames_at(&self, time: f32) -> (usize, usize, f32) {
        let t = if self.duration > 0.0 { time - self.duration * (time / self.duration).floor() } else { 0.0 };
        let f = t * self.rate;
        let last = self.frames as usize - 1;
        let a = (f.max(0.0) as usize).min(last);
        (a, (a + 1).min(last), (f - a as f32).clamp(0.0, 1.0))
    }

    /// Local matrices at `time`: each bone's rest matrix, the tracks'
    /// bones blended from two frames.
    pub fn locals(&self, model: &Model, time: f32) -> Vec<Mat4> {
        let mut locals: Vec<Mat4> = model.bones.iter().map(|b| Mat4::from_cols_array(&b.rest)).collect();
        let (a, b, s) = self.frames_at(time);
        let n = self.tracks.len();
        for (i, &bone) in self.tracks.iter().enumerate() {
            let (ka, kb) = (self.keys[a * n + i], self.keys[b * n + i]);
            let r = Quat::from_vec4(Vec4::from(ka.r).lerp(Vec4::from(kb.r), s).normalize());
            locals[bone as usize] = Mat4::from_scale_rotation_translation(ka.s.lerp(kb.s, s), r, ka.t.lerp(kb.t, s));
        }
        locals
    }
}

/// Vertex `v`'s position skinned by `palette`.
pub fn skin(v: &Vertex, palette: &[Mat4]) -> Vec3 {
    let m = (0..4).fold(Mat4::ZERO, |m, k| m + palette[v.joints[k] as usize] * (v.weights[k] as f32 / 65535.0));
    m.transform_point3(Vec3::from(v.position))
}

/// The largest distance from `centre` of a vertex in any frame of any
/// baked clip (0 without clips): how far animation reaches.
pub fn animated_radius(model: &Model, centre: Vec3) -> f32 {
    let mut radius = 0.0f32;
    for clip in 0..model.clips.len() {
        let baked = bake(model, clip);
        for f in 0..baked.frames {
            let time = if f + 1 == baked.frames { baked.duration } else { f as f32 / baked.rate.max(1e-9) };
            // At the frame itself (the blend's weight 0).
            let time = time.min(baked.duration * (1.0 - 1e-6));
            let palette = palette(model, &globals(model, &baked.locals(model, time)));
            for v in &model.vertices {
                radius = radius.max((skin(v, &palette) - centre).length());
            }
        }
    }
    radius
}

/// One channel at time `t`; `n` components per value.
fn sample_channel(times: &[f32], values: &[f32], n: usize, interp: Interpolation, rotation: bool, t: f32) -> Vec4 {
    let cubic = interp == Interpolation::CubicSpline;
    // Value k (for a cubic spline, the middle of its three).
    let value = |k: usize| -> Vec4 {
        let base = if cubic { (k * 3 + 1) * n } else { k * n };
        let mut v = [0.0f32; 4];
        v[..n].copy_from_slice(&values[base..base + n]);
        Vec4::from_array(v)
    };
    let tangent = |k: usize, out: bool| -> Vec4 {
        let base = (k * 3 + if out { 2 } else { 0 }) * n;
        let mut v = [0.0f32; 4];
        v[..n].copy_from_slice(&values[base..base + n]);
        Vec4::from_array(v)
    };
    let last = times.len() - 1;
    if t <= times[0] || last == 0 {
        return value(0);
    }
    if t >= times[last] {
        return value(last);
    }
    // times[k] <= t < times[k + 1]
    let k = times.partition_point(|&x| x <= t) - 1;
    let (t0, t1) = (times[k], times[k + 1]);
    let dt = t1 - t0;
    let s = if dt > 0.0 { (t - t0) / dt } else { 0.0 };
    match interp {
        Interpolation::Step => value(k),
        Interpolation::Linear => {
            if rotation {
                let a = Quat::from_vec4(value(k)).normalize();
                let b = Quat::from_vec4(value(k + 1)).normalize();
                Vec4::from(a.slerp(b, s))
            } else {
                value(k).lerp(value(k + 1), s)
            }
        }
        Interpolation::CubicSpline => {
            let (s2, s3) = (s * s, s * s * s);
            let v = value(k) * (2.0 * s3 - 3.0 * s2 + 1.0)
                + tangent(k, true) * (dt * (s3 - 2.0 * s2 + s))
                + value(k + 1) * (-2.0 * s3 + 3.0 * s2)
                + tangent(k + 1, false) * (dt * (s3 - s2));
            if rotation { v.normalize() } else { v }
        }
    }
}
