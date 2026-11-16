//! Poses: the bones' local transforms, from the rest pose or a clip, to
//! model-space bone matrices and the skinning palette.

use crate::{Interpolation, Model, Path};
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
