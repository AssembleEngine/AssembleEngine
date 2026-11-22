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
    model
        .joints
        .iter()
        .map(|j| globals[j.bone as usize] * Mat4::from_cols_array(&j.inverse_bind))
        .collect()
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
            keys.push(Trs {
                t: pose.translation[b],
                r,
                s: pose.scale[b],
            });
        }
    }
    BakedClip {
        tracks,
        frames,
        rate,
        duration: c.duration,
        keys,
    }
}

/// How a clip plays past its end.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum PlayMode {
    /// From the start again.
    #[default]
    Loop = 0,
    /// Backwards to the start, then forwards again.
    PingPong = 1,
    /// Once, then holding the last frame.
    Once = 2,
}

impl PlayMode {
    pub const ALL: [PlayMode; 3] = [PlayMode::Loop, PlayMode::PingPong, PlayMode::Once];

    pub fn from_bits(bits: u32) -> Self {
        match bits {
            1 => PlayMode::PingPong,
            2 => PlayMode::Once,
            _ => PlayMode::Loop,
        }
    }

    /// `time` seconds into a clip `duration` long, as a time in the clip
    /// (`0..=duration`); as the pose pass computes it.
    pub fn clip_time(self, time: f32, duration: f32) -> f32 {
        if duration <= 0.0 {
            return 0.0;
        }
        let time = time.max(0.0);
        match self {
            PlayMode::Loop => time - duration * (time / duration).floor(),
            PlayMode::PingPong => {
                let t = time - 2.0 * duration * (time / (2.0 * duration)).floor();
                if t > duration { 2.0 * duration - t } else { t }
            }
            PlayMode::Once => time.min(duration),
        }
    }
}

impl BakedClip {
    /// The two frames around `time` seconds (into the clip as `mode` plays
    /// it) and the second one's weight; as the pose pass computes them.
    pub fn frames_at(&self, time: f32, mode: PlayMode) -> (usize, usize, f32) {
        let t = mode.clip_time(time, self.duration);
        let f = t * self.rate;
        let last = self.frames as usize - 1;
        let a = (f.max(0.0) as usize).min(last);
        (a, (a + 1).min(last), (f - a as f32).clamp(0.0, 1.0))
    }

    /// Local matrices at `time` played as `mode`: each bone's rest matrix,
    /// the tracks' bones blended from two frames.
    pub fn locals(&self, model: &Model, time: f32, mode: PlayMode) -> Vec<Mat4> {
        let mut locals: Vec<Mat4> = model.bones.iter().map(|b| Mat4::from_cols_array(&b.rest)).collect();
        let (a, b, s) = self.frames_at(time, mode);
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
    let m = (0..4).fold(Mat4::ZERO, |m, k| {
        m + palette[v.joints[k] as usize] * (v.weights[k] as f32 / 65535.0)
    });
    m.transform_point3(Vec3::from(v.position))
}

/// The largest distance from `centre` of a vertex in any frame of any
/// baked clip (0 without clips): how far animation reaches.
pub fn animated_radius(model: &Model, centre: Vec3) -> f32 {
    let mut radius = 0.0f32;
    for clip in 0..model.clips.len() {
        let baked = bake(model, clip);
        for f in 0..baked.frames {
            let time = if f + 1 == baked.frames {
                baked.duration
            } else {
                f as f32 / baked.rate.max(1e-9)
            };
            // At the frame itself (the blend's weight 0).
            let time = time.min(baked.duration * (1.0 - 1e-6));
            let palette = palette(model, &globals(model, &baked.locals(model, time, PlayMode::Loop)));
            for v in &model.vertices {
                radius = radius.max((skin(v, &palette) - centre).length());
            }
        }
    }
    radius
}

/// How far from the model's origin a vertex can be with any of its bones
/// turned any way (M14's manual turns and spins), each bone's translation
/// and scale as its rest pose or a clip frame has them, a clip's change
/// from its first frame added once (additive layers): a bound, not the
/// farthest pose. Each bone's origin is at most its parent's reach plus
/// its longest translation times the scales above it; a vertex at most its
/// bone's reach plus its distance from the bone times the scales down to
/// it (a skinned vertex: the most of its bones', the blend being a convex
/// combination). Manual moves and scales are not bounded (they go
/// anywhere). 0 without joints (nothing moves).
pub fn turned_reach(model: &Model) -> f32 {
    if model.joints.is_empty() {
        return 0.0;
    }
    let n = model.bones.len();
    // Per bone: its longest local translation and largest scale factor
    // (the largest stretch of its local matrix: its largest scale when the
    // rest matrix is its TRS, else bounded by the Frobenius norm).
    let mut t: Vec<f32> = model
        .bones
        .iter()
        .map(|b| Mat4::from_cols_array(&b.rest).w_axis.truncate().length())
        .collect();
    let mut s: Vec<f32> = model
        .bones
        .iter()
        .map(|b| {
            let m = Mat4::from_cols_array(&b.rest);
            let trs = Mat4::from_scale_rotation_translation(
                Vec3::from(b.scale),
                Quat::from_array(b.rotation).normalize(),
                Vec3::from(b.translation),
            );
            let scale = Vec3::from(b.scale).abs().max_element();
            match m.abs_diff_eq(trs, 1e-4 * scale.max(1.0)) {
                true => scale,
                false => (m.x_axis.truncate().length_squared()
                    + m.y_axis.truncate().length_squared()
                    + m.z_axis.truncate().length_squared())
                .sqrt(),
            }
        })
        .collect();
    let (mut t_add, mut s_add) = (vec![0.0f32; n], vec![1.0f32; n]);
    for clip in 0..model.clips.len() {
        let baked = bake(model, clip);
        let tracks = baked.tracks.len();
        let (mut dt, mut ds) = (vec![0.0f32; tracks], vec![1.0f32; tracks]);
        for (k, key) in baked.keys.iter().enumerate() {
            let (i, b) = (k % tracks, baked.tracks[k % tracks] as usize);
            let first = baked.keys[i];
            t[b] = t[b].max(key.t.length());
            s[b] = s[b].max(key.s.abs().max_element());
            dt[i] = dt[i].max((key.t - first.t).length());
            let ratio = Vec3::select(first.s.cmpne(Vec3::ZERO), key.s / first.s, Vec3::ONE);
            ds[i] = ds[i].max(ratio.abs().max_element());
        }
        for (i, &b) in baked.tracks.iter().enumerate() {
            t_add[b as usize] += dt[i];
            s_add[b as usize] *= ds[i];
        }
    }
    // Each bone's origin's reach, and the scale its children are under.
    let (mut reach, mut scale) = (vec![0.0f32; n], vec![1.0f32; n]);
    for (b, bone) in model.bones.iter().enumerate() {
        let (r, sc) = bone
            .parent
            .map_or((0.0, 1.0), |p| (reach[p as usize], scale[p as usize]));
        reach[b] = r + sc * (t[b] + t_add[b]);
        scale[b] = sc * s[b] * s_add[b];
    }
    let mut most = 0.0f32;
    for v in &model.vertices {
        for k in 0..4 {
            if v.weights[k] == 0 && k > 0 {
                continue;
            }
            let j = &model.joints[v.joints[k] as usize];
            let b = j.bone as usize;
            let local = Mat4::from_cols_array(&j.inverse_bind).transform_point3(Vec3::from(v.position));
            most = most.max(reach[b] + scale[b] * local.length());
        }
    }
    most
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
