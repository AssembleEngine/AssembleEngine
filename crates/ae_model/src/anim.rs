//! Animation channels (M14): what a unit's model is doing, as plain data
//! the sim keeps in its columns ([`Channels`]). [`Rig::advance`] moves them
//! on by a tick (clip phases, layer weights, tweens) without any script;
//! [`Rig::pose`] turns them into the bones' local transforms, and
//! [`Rig::globals`] and [`Rig::palette`] into 3x4 matrices.
//!
//! The pose is a pure function of the channels, in strict scalar f32: no
//! FMA, no SIMD (so no reordered sums), trig from `libm`, and clips
//! resampled here by the same rules, so every machine, the client and the
//! sim's on-demand queries get the same bits. glam is not used.
//!
//! The stack, per bone, in local space (translation and scale lerp,
//! rotation nlerp on one hemisphere), from the rest pose up:
//! * **Layers**, in index order: each a clip at a phase, played by its
//!   [`PlayMode`], weighted by its weight times its mask's weight for the
//!   bone ([`Rig::add_mask`]; mask 0 is the whole model).
//!   - A *blend* layer mixes its clip over what is below it
//!     (`lerp(below, clip, w)`). Over the whole model that is a cross-fade
//!     or a mix of two clips by a parameter; on a mask it is an override
//!     (the upper body aiming while the legs run). Bones its clip does not
//!     animate are left as they are.
//!   - An *additive* layer adds its clip's change from the clip's first
//!     frame (a recoil kick, breathing).
//! * **Tweens**, after the layers, in index order: turn about a local axis,
//!   move along one, or spin; on one bone, or on every bone of a mask by the
//!   mask's weight (a twist spread over a spine). Either *over the rest
//!   pose* (replacing what the layers did to that part of the bone, by the
//!   weight) or *additive* (on top of the result).
//!
//! A bone nothing touches keeps its rest matrix, which need not be a TRS.

use crate::pose::{BAKE_RATE, PlayMode};
use crate::{Interpolation, Model, Path};

/// Clip layers per unit.
pub const LAYERS: usize = 4;
/// Tweens per unit.
pub const TWEENS: usize = 8;
/// No clip ([`Layer::clip`]), no bone ([`Tween::bone`]).
pub const NONE: u16 = u16::MAX;

/// [`Layer::kind`].
pub mod kind {
    pub const BLEND: u8 = 0;
    pub const ADDITIVE: u8 = 1;
}

/// [`Layer::flags`].
pub mod layer_flags {
    /// `speed` is clip seconds per metre the unit moves (the ticker's
    /// ground speed), not per second: no foot sliding.
    pub const GROUND_SPEED: u8 = 1;
    /// Set by [`super::Rig::advance`] when a clip played once has reached
    /// its end (its start, played backwards); it holds that frame.
    pub const ENDED: u8 = 2;
}

/// [`Tween::kind`].
pub mod tween_kind {
    pub const NONE: u8 = 0;
    /// To `target` radians about the axis, at `speed` radians a second (the
    /// short way round; 0: at once).
    pub const TURN: u8 = 1;
    /// To `target` along the axis, at `speed` a second (0: at once).
    pub const MOVE: u8 = 2;
    /// About the axis at `speed` radians a second, for ever.
    pub const SPIN: u8 = 3;
}

/// [`Tween::flags`].
pub mod tween_flags {
    /// On top of the layers' result, not over the rest pose.
    pub const ADDITIVE: u8 = 1;
}

/// One clip layer. 28 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layer {
    /// The model's clip, or [`NONE`] (an empty layer).
    pub clip: u16,
    /// 0: the whole model; `k`: the rig's mask `k` ([`Rig::add_mask`]).
    pub mask: u16,
    /// [`PlayMode`] as its bits.
    pub mode: u8,
    /// [`kind`].
    pub kind: u8,
    /// [`layer_flags`].
    pub flags: u8,
    pub reserved: u8,
    pub weight: f32,
    /// Where `weight` is going, at `fade` a second; a layer whose weight and
    /// target are both 0 is cleared.
    pub target: f32,
    /// Weight per second (0: at once).
    pub fade: f32,
    /// Clip seconds per second (or per metre, [`layer_flags::GROUND_SPEED`]);
    /// negative plays backwards.
    pub speed: f32,
    /// Seconds into the clip (kept within one cycle by the ticker).
    pub phase: f32,
}

impl Default for Layer {
    fn default() -> Self {
        Self { clip: NONE, mask: 0, mode: 0, kind: kind::BLEND, flags: 0, reserved: 0, weight: 0.0, target: 0.0, fade: 0.0, speed: 0.0, phase: 0.0 }
    }
}

impl Layer {
    pub fn is_empty(&self) -> bool {
        self.clip == NONE
    }

    pub fn ended(&self) -> bool {
        self.flags & layer_flags::ENDED != 0
    }
}

/// One procedural channel. 20 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tween {
    /// The bone, or [`NONE`] with a mask.
    pub bone: u16,
    /// 0: just `bone`; `k`: every bone of the rig's mask `k`, by its weight.
    pub mask: u16,
    /// [`tween_kind`].
    pub kind: u8,
    /// 0 x, 1 y, 2 z: the bone's local axis.
    pub axis: u8,
    /// [`tween_flags`].
    pub flags: u8,
    pub reserved: u8,
    /// Radians or distance now.
    pub value: f32,
    pub target: f32,
    pub speed: f32,
}

impl Default for Tween {
    fn default() -> Self {
        Self { bone: NONE, mask: 0, kind: tween_kind::NONE, axis: 0, flags: 0, reserved: 0, value: 0.0, target: 0.0, speed: 0.0 }
    }
}

impl Tween {
    pub fn is_empty(&self) -> bool {
        self.kind == tween_kind::NONE
    }

    /// At its target (a spin never is).
    pub fn done(&self) -> bool {
        matches!(self.kind, tween_kind::TURN | tween_kind::MOVE) && self.value == self.target
    }
}

/// A unit's animation channels: what the sim keeps per unit. 272 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Channels {
    pub layers: [Layer; LAYERS],
    pub tweens: [Tween; TWEENS],
}

const _: () = assert!(size_of::<Layer>() == 28 && size_of::<Tween>() == 20 && size_of::<Channels>() == 272);

/// A bone's local transform: translation, rotation (x, y, z, w), scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trs {
    pub t: [f32; 3],
    pub r: [f32; 4],
    pub s: [f32; 3],
}

/// A 3x4 affine matrix, by rows (what the GPU palette holds).
pub type Affine = [f32; 12];

pub const IDENTITY: Affine = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

/// A named weight per bone.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub name: String,
    pub weights: Vec<f32>,
}

/// A clip resampled at a fixed rate: per frame, the local TRS of each bone
/// it animates (a track); consecutive rotations on one hemisphere.
#[derive(Clone, Debug, PartialEq)]
struct Baked {
    tracks: Vec<u32>,
    frames: u32,
    rate: f32,
    duration: f32,
    keys: Vec<Trs>,
}

impl Baked {
    fn key(&self, frame: usize, track: usize) -> Trs {
        self.keys[frame * self.tracks.len() + track]
    }

    fn frames_at(&self, phase: f32, mode: PlayMode) -> (usize, usize, f32) {
        let t = mode.clip_time(phase, self.duration);
        let f = t * self.rate;
        let last = self.frames as usize - 1;
        let a = (f.max(0.0) as usize).min(last);
        (a, (a + 1).min(last), (f - a as f32).clamp(0.0, 1.0))
    }
}

/// The bones' local transforms, and which of them anything posed.
#[derive(Clone, Debug, PartialEq)]
pub struct Pose {
    pub local: Vec<Trs>,
    pub posed: Vec<bool>,
}

/// A model's skeleton, clips and masks, as the channels need them.
#[derive(Clone, Debug, PartialEq)]
pub struct Rig {
    names: Vec<String>,
    parents: Vec<u32>,
    rest: Vec<Trs>,
    rest_affine: Vec<Affine>,
    joints: Vec<(u32, Affine)>,
    clips: Vec<Baked>,
    clip_names: Vec<String>,
    masks: Vec<Mask>,
}

impl Rig {
    pub fn new(model: &Model) -> Rig {
        let rest: Vec<Trs> = model.bones.iter().map(|b| Trs { t: b.translation, r: b.rotation, s: b.scale }).collect();
        let mut rig = Rig {
            names: model.bones.iter().map(|b| b.name.clone()).collect(),
            parents: model.bones.iter().map(|b| b.parent.unwrap_or(u32::MAX)).collect(),
            rest_affine: model.bones.iter().map(|b| affine_of_cols(&b.rest)).collect(),
            joints: model.joints.iter().map(|j| (j.bone, affine_of_cols(&j.inverse_bind))).collect(),
            clips: Vec::new(),
            clip_names: model.clips.iter().map(|c| c.name.clone()).collect(),
            masks: Vec::new(),
            rest,
        };
        rig.clips = (0..model.clips.len()).map(|c| bake(model, &rig.rest, c)).collect();
        rig
    }

    pub fn bones(&self) -> usize {
        self.parents.len()
    }

    pub fn joints(&self) -> usize {
        self.joints.len()
    }

    pub fn bone(&self, name: &str) -> Option<u16> {
        self.names.iter().position(|n| n == name).map(|b| b as u16)
    }

    pub fn clip(&self, name: &str) -> Option<u16> {
        self.clip_names.iter().position(|n| n == name).map(|c| c as u16)
    }

    pub fn clip_names(&self) -> &[String] {
        &self.clip_names
    }

    /// Clip `clip`'s length in seconds (0 for none).
    pub fn duration(&self, clip: u16) -> f32 {
        self.clips.get(clip as usize).map_or(0.0, |c| c.duration)
    }

    /// Weight 1 on `root` and every bone under it, 0 elsewhere.
    pub fn subtree(&self, root: u16) -> Vec<f32> {
        let mut w = vec![0.0f32; self.bones()];
        for b in 0..self.bones() {
            let p = self.parents[b];
            if b == root as usize || (p != u32::MAX && w[p as usize] > 0.0) {
                w[b] = 1.0;
            }
        }
        w
    }

    /// Adds a mask (a weight per bone) and returns its id for
    /// [`Layer::mask`] and [`Tween::mask`].
    pub fn add_mask(&mut self, name: &str, weights: Vec<f32>) -> u16 {
        assert_eq!(weights.len(), self.bones(), "a mask has a weight per bone");
        self.masks.push(Mask { name: name.into(), weights });
        self.masks.len() as u16
    }

    pub fn mask(&self, name: &str) -> Option<u16> {
        self.masks.iter().position(|m| m.name == name).map(|k| k as u16 + 1)
    }

    fn mask_weights(&self, mask: u16) -> Option<&[f32]> {
        match mask {
            0 => None,
            k => self.masks.get(k as usize - 1).map(|m| &m.weights[..]),
        }
    }

    pub fn rest_pose(&self) -> Pose {
        Pose { local: self.rest.clone(), posed: vec![false; self.bones()] }
    }

    /// One tick of `dt` seconds: layers' weights fade (a layer faded out is
    /// cleared), their phases move on (by the unit's `ground_speed` for
    /// [`layer_flags::GROUND_SPEED`]) and wrap, a clip played once stops at
    /// its end ([`layer_flags::ENDED`]); tweens move toward their targets.
    pub fn advance(&self, ch: &mut Channels, dt: f32, ground_speed: f32) {
        for l in ch.layers.iter_mut() {
            if l.is_empty() {
                continue;
            }
            l.weight = toward(l.weight, l.target, l.fade * dt, l.fade <= 0.0);
            if l.weight <= 0.0 && l.target <= 0.0 {
                *l = Layer::default();
                continue;
            }
            let rate = if l.flags & layer_flags::GROUND_SPEED != 0 { l.speed * ground_speed } else { l.speed };
            let d = self.duration(l.clip);
            l.phase += rate * dt;
            match PlayMode::from_bits(l.mode as u32) {
                PlayMode::Loop => l.phase = wrap(l.phase, d),
                PlayMode::PingPong => l.phase = wrap(l.phase, 2.0 * d),
                PlayMode::Once => {
                    if (rate > 0.0 && l.phase >= d) || (rate < 0.0 && l.phase <= 0.0) {
                        l.flags |= layer_flags::ENDED;
                    }
                    l.phase = l.phase.clamp(0.0, d);
                }
            }
        }
        for tw in ch.tweens.iter_mut() {
            let step = tw.speed * dt;
            match tw.kind {
                tween_kind::TURN => {
                    let diff = wrap_pi(tw.target - tw.value);
                    tw.value = if tw.speed <= 0.0 || diff.abs() <= step { tw.target } else { wrap_pi(tw.value + if diff > 0.0 { step } else { -step }) };
                }
                tween_kind::MOVE => {
                    let diff = tw.target - tw.value;
                    tw.value = if tw.speed <= 0.0 || diff.abs() <= step { tw.target } else { tw.value + if diff > 0.0 { step } else { -step } };
                }
                tween_kind::SPIN => tw.value = wrap_pi(tw.value + step),
                _ => {}
            }
        }
    }

    /// The local transforms for `ch`, into `out` (from [`Rig::rest_pose`]).
    pub fn pose(&self, ch: &Channels, out: &mut Pose) {
        out.local.copy_from_slice(&self.rest);
        out.posed.fill(false);
        for l in &ch.layers {
            if l.is_empty() || l.weight <= 0.0 {
                continue;
            }
            let Some(c) = self.clips.get(l.clip as usize) else { continue };
            let mask = self.mask_weights(l.mask);
            let (a, b, s) = c.frames_at(l.phase, PlayMode::from_bits(l.mode as u32));
            for (i, &bone) in c.tracks.iter().enumerate() {
                let w = (l.weight * mask.map_or(1.0, |m| m[bone as usize])).min(1.0);
                if w <= 0.0 {
                    continue;
                }
                let k = lerp_trs(c.key(a, i), c.key(b, i), s);
                let p = &mut out.local[bone as usize];
                if l.kind == kind::ADDITIVE {
                    let r0 = c.key(0, i);
                    for j in 0..3 {
                        p.t[j] += (k.t[j] - r0.t[j]) * w;
                        let ratio = if r0.s[j] != 0.0 { k.s[j] / r0.s[j] } else { 1.0 };
                        p.s[j] *= 1.0 + (ratio - 1.0) * w;
                    }
                    let d = qmul(qconj(r0.r), k.r);
                    p.r = qnorm(qmul(p.r, nlerp([0.0, 0.0, 0.0, 1.0], d, w)));
                } else {
                    *p = lerp_trs(*p, k, w);
                }
                out.posed[bone as usize] = true;
            }
        }
        for tw in &ch.tweens {
            if tw.is_empty() {
                continue;
            }
            match self.mask_weights(tw.mask) {
                Some(m) => {
                    for (b, &w) in m.iter().enumerate() {
                        if w > 0.0 {
                            self.apply_tween(tw, b, w.min(1.0), out);
                        }
                    }
                }
                None if (tw.bone as usize) < self.bones() => self.apply_tween(tw, tw.bone as usize, 1.0, out),
                None => {}
            }
        }
    }

    fn apply_tween(&self, tw: &Tween, b: usize, w: f32, out: &mut Pose) {
        let mut axis = [0.0f32; 3];
        axis[(tw.axis as usize).min(2)] = 1.0;
        let additive = tw.flags & tween_flags::ADDITIVE != 0;
        let p = &mut out.local[b];
        match tw.kind {
            tween_kind::TURN | tween_kind::SPIN => {
                if additive {
                    p.r = qnorm(qmul(p.r, axis_angle(axis, tw.value * w)));
                } else {
                    let to = qnorm(qmul(self.rest[b].r, axis_angle(axis, tw.value)));
                    p.r = nlerp(p.r, to, w);
                }
            }
            tween_kind::MOVE => {
                for (j, a) in axis.iter().enumerate() {
                    if additive {
                        p.t[j] += a * tw.value * w;
                    } else {
                        let to = self.rest[b].t[j] + a * tw.value;
                        p.t[j] += (to - p.t[j]) * w;
                    }
                }
            }
            _ => return,
        }
        out.posed[b] = true;
    }

    /// Model-space bone matrices (parents come first).
    pub fn globals(&self, pose: &Pose, out: &mut [Affine]) {
        for b in 0..self.bones() {
            let local = if pose.posed[b] { affine_of_trs(&pose.local[b]) } else { self.rest_affine[b] };
            out[b] = match self.parents[b] {
                u32::MAX => local,
                p => mul(&out[p as usize], &local),
            };
        }
    }

    /// The skinning palette: per joint, its bone's matrix times its inverse
    /// bind matrix.
    pub fn palette(&self, globals: &[Affine], out: &mut [Affine]) {
        for (o, (bone, inverse_bind)) in out.iter_mut().zip(&self.joints) {
            *o = mul(&globals[*bone as usize], inverse_bind);
        }
    }
}

/// `x` moved toward `to` by at most `step` (or all the way, `at_once`).
fn toward(x: f32, to: f32, step: f32, at_once: bool) -> f32 {
    let d = to - x;
    if at_once || d.abs() <= step { to } else if d > 0.0 { x + step } else { x - step }
}

/// `x` into `[0, period)` (0 for no period).
fn wrap(x: f32, period: f32) -> f32 {
    if period <= 0.0 {
        return 0.0;
    }
    let y = x - period * libm::floorf(x / period);
    if (0.0..period).contains(&y) { y } else { 0.0 }
}

/// An angle into `[-π, π)`.
fn wrap_pi(a: f32) -> f32 {
    use core::f32::consts::{PI, TAU};
    let y = a - TAU * libm::floorf((a + PI) / TAU);
    if (-PI..PI).contains(&y) { y } else { -PI }
}

fn qmul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

fn qconj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn qnorm(q: [f32; 4]) -> [f32; 4] {
    let len = libm::sqrtf(dot4(q, q));
    if len > 0.0 { [q[0] / len, q[1] / len, q[2] / len, q[3] / len] } else { [0.0, 0.0, 0.0, 1.0] }
}

/// From `a` to `b` by `s`, the short way, normalised.
fn nlerp(a: [f32; 4], b: [f32; 4], s: f32) -> [f32; 4] {
    let b = if dot4(a, b) < 0.0 { [-b[0], -b[1], -b[2], -b[3]] } else { b };
    qnorm([a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s, a[3] + (b[3] - a[3]) * s])
}

fn slerp(a: [f32; 4], b: [f32; 4], s: f32) -> [f32; 4] {
    let mut d = dot4(a, b);
    let b = if d < 0.0 {
        d = -d;
        [-b[0], -b[1], -b[2], -b[3]]
    } else {
        b
    };
    if d > 0.9995 {
        return nlerp(a, b, s);
    }
    let th = libm::acosf(d);
    let sn = libm::sinf(th);
    let (wa, wb) = (libm::sinf((1.0 - s) * th) / sn, libm::sinf(s * th) / sn);
    qnorm([a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb, a[2] * wa + b[2] * wb, a[3] * wa + b[3] * wb])
}

fn axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let (s, c) = (libm::sinf(angle * 0.5), libm::cosf(angle * 0.5));
    [axis[0] * s, axis[1] * s, axis[2] * s, c]
}

fn lerp_trs(a: Trs, b: Trs, s: f32) -> Trs {
    let l = |x: [f32; 3], y: [f32; 3]| [x[0] + (y[0] - x[0]) * s, x[1] + (y[1] - x[1]) * s, x[2] + (y[2] - x[2]) * s];
    Trs { t: l(a.t, b.t), r: nlerp(a.r, b.r, s), s: l(a.s, b.s) }
}

pub fn affine_of_trs(p: &Trs) -> Affine {
    let [x, y, z, w] = p.r;
    let [sx, sy, sz] = p.s;
    [
        (1.0 - 2.0 * (y * y + z * z)) * sx,
        2.0 * (x * y - w * z) * sy,
        2.0 * (x * z + w * y) * sz,
        p.t[0],
        2.0 * (x * y + w * z) * sx,
        (1.0 - 2.0 * (x * x + z * z)) * sy,
        2.0 * (y * z - w * x) * sz,
        p.t[1],
        2.0 * (x * z - w * y) * sx,
        2.0 * (y * z + w * x) * sy,
        (1.0 - 2.0 * (x * x + y * y)) * sz,
        p.t[2],
    ]
}

/// A column-major 4x4 (its affine part) by rows.
pub fn affine_of_cols(m: &[f32; 16]) -> Affine {
    [m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14]]
}

pub fn mul(a: &Affine, b: &Affine) -> Affine {
    let mut c = [0.0f32; 12];
    for i in 0..3 {
        for j in 0..4 {
            let mut v = a[i * 4] * b[j] + a[i * 4 + 1] * b[4 + j] + a[i * 4 + 2] * b[8 + j];
            if j == 3 {
                v += a[i * 4 + 3];
            }
            c[i * 4 + j] = v;
        }
    }
    c
}

/// Point `p` through `m`.
pub fn transform(m: &Affine, p: [f32; 3]) -> [f32; 3] {
    let row = |i: usize| m[i * 4] * p[0] + m[i * 4 + 1] * p[1] + m[i * 4 + 2] * p[2] + m[i * 4 + 3];
    [row(0), row(1), row(2)]
}

/// Clip `clip` resampled at [`BAKE_RATE`] or a little more, 0 to its
/// duration inclusive (as `pose::bake`, in this module's arithmetic).
fn bake(model: &Model, rest: &[Trs], clip: usize) -> Baked {
    let c = &model.clips[clip];
    let mut tracks: Vec<u32> = c.channels.iter().map(|ch| ch.bone).collect();
    tracks.sort_unstable();
    tracks.dedup();
    let (frames, rate) = match c.duration > 0.0 {
        true => {
            let n = libm::ceilf(c.duration * BAKE_RATE).max(1.0) as u32;
            (n + 1, n as f32 / c.duration)
        }
        false => (1, 0.0),
    };
    let mut keys: Vec<Trs> = Vec::with_capacity(frames as usize * tracks.len());
    for f in 0..frames {
        let t = if f + 1 == frames { c.duration } else { f as f32 / rate };
        let base = keys.len();
        keys.extend(tracks.iter().map(|&b| rest[b as usize]));
        for ch in &c.channels {
            let i = tracks.binary_search(&ch.bone).unwrap();
            let v = sample(model.clip_floats(ch.times), model.clip_floats(ch.values), ch.path, ch.interpolation, t);
            let k = &mut keys[base + i];
            match ch.path {
                Path::Translation => k.t = [v[0], v[1], v[2]],
                Path::Rotation => k.r = qnorm(v),
                Path::Scale => k.s = [v[0], v[1], v[2]],
            }
        }
        if f > 0 {
            for i in 0..tracks.len() {
                let prev = keys[base - tracks.len() + i].r;
                let k = &mut keys[base + i];
                if dot4(prev, k.r) < 0.0 {
                    k.r = [-k.r[0], -k.r[1], -k.r[2], -k.r[3]];
                }
            }
        }
    }
    Baked { tracks, frames, rate, duration: c.duration, keys }
}

/// One channel at time `t` (held at its first and last keys outside them).
fn sample(times: &[f32], values: &[f32], path: Path, interp: Interpolation, t: f32) -> [f32; 4] {
    let n = path.components();
    let cubic = interp == Interpolation::CubicSpline;
    let at = |base: usize| {
        let mut v = [0.0f32; 4];
        v[..n].copy_from_slice(&values[base..base + n]);
        v
    };
    // Value k (for a cubic spline the middle of its three), and tangents.
    let value = |k: usize| at(if cubic { (k * 3 + 1) * n } else { k * n });
    let tangent = |k: usize, out: bool| at((k * 3 + if out { 2 } else { 0 }) * n);
    let last = times.len() - 1;
    if t <= times[0] || last == 0 {
        return value(0);
    }
    if t >= times[last] {
        return value(last);
    }
    let k = times.partition_point(|&x| x <= t) - 1;
    let (t0, t1) = (times[k], times[k + 1]);
    let dt = t1 - t0;
    let s = if dt > 0.0 { (t - t0) / dt } else { 0.0 };
    let rotation = path == Path::Rotation;
    match interp {
        Interpolation::Step => value(k),
        Interpolation::Linear if rotation => slerp(qnorm(value(k)), qnorm(value(k + 1)), s),
        Interpolation::Linear => {
            let (a, b) = (value(k), value(k + 1));
            [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s, a[3] + (b[3] - a[3]) * s]
        }
        Interpolation::CubicSpline => {
            let (s2, s3) = (s * s, s * s * s);
            let (h00, h10, h01, h11) = (2.0 * s3 - 3.0 * s2 + 1.0, dt * (s3 - 2.0 * s2 + s), -2.0 * s3 + 3.0 * s2, dt * (s3 - s2));
            let (v0, m0, v1, m1) = (value(k), tangent(k, true), value(k + 1), tangent(k + 1, false));
            let v = [0, 1, 2, 3].map(|j| v0[j] * h00 + m0[j] * h10 + v1[j] * h01 + m1[j] * h11);
            if rotation { qnorm(v) } else { v }
        }
    }
}
