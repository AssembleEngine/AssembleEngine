//! Animation channels (M14): what a unit's model is doing, as plain data
//! the sim keeps in its columns ([`Channels`]). [`Rig::advance`] moves them
//! on by a tick (clip phases, layer weights, manual joints) without any
//! script; [`Rig::pose`] turns them into the bones' local transforms, and
//! [`Rig::globals`] and [`Rig::palette`] into 3x4 matrices.
//!
//! The pose is a pure function of the channels, in strict scalar f32: no
//! FMA, no SIMD (so no reordered sums), trig from `libm`, and clips
//! resampled here by the same rules, so every machine, the client and the
//! sim's on-demand queries get the same bits. glam is not used.
//!
//! Each bone is either **clipped** or **manual**:
//! * A clipped bone is built in local space (translation and scale lerp,
//!   rotation nlerp on one hemisphere) from the rest pose up by the
//!   **layers**, in index order: each a clip at a phase, played by its
//!   [`PlayMode`], weighted by its weight times its mask's weight for the
//!   bone ([`Rig::add_mask`]; mask 0 is the whole model).
//!   - A *blend* layer mixes its clip over what is below it
//!     (`lerp(below, clip, w)`). Over the whole model that is a cross-fade
//!     or a mix of two clips by a parameter; on a mask it is an override
//!     (the upper body aiming while the legs run). Bones its clip does not
//!     animate are left as they are.
//!   - An *additive* layer adds its clip's change from the clip's first
//!     frame (a recoil kick, breathing).
//! * A bone with a [`Manual`] entry (turn, spin, move, scale) takes no clip
//!   at all: its rest pose with its entries' values in place of their
//!   parts. Its children are still clipped. Playing a clip
//!   ([`layer_flags::NEW`]) takes back the bones it animates within its
//!   mask (entries given in the same tick stay); a bone a clip was driving
//!   starts its entry from where the clip had it.
//!
//! A bone nothing touches keeps its rest matrix, which need not be a TRS.

use crate::pose::{BAKE_RATE, PlayMode};
use crate::{Interpolation, Model, Path};

/// Clip layers per unit.
pub const LAYERS: usize = 4;
/// Manual entries per unit.
pub const MANUAL: usize = 8;

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
    /// Just played: the next [`super::Rig::advance`] hands this clip's
    /// bones back from manual entries (unless additive), then clears it.
    pub const NEW: u8 = 4;
}

/// [`Manual::kind`]. Turn and spin are a bone's rotation entry, move its
/// translation entry, scale its scale entry; a bone has at most one of each.
pub mod manual_kind {
    pub const NONE: u8 = 0;
    /// To a rotation at up to `speed` radians a second (0: at once).
    pub const TURN: u8 = 1;
    /// About an axis at `speed` radians a second (signed), for ever.
    pub const SPIN: u8 = 2;
    /// To a position at up to `speed` a second (0: at once).
    pub const MOVE: u8 = 3;
    /// To a scale at up to `speed` a second (0: at once).
    pub const SCALE: u8 = 4;
}

/// [`Manual::flags`].
pub mod manual_flags {
    /// `target` holds the command as given (see [`super::Manual`]); the
    /// next [`super::Rig::advance`] turns it into the bone's terms.
    pub const PENDING: u8 = 1;
    /// `value` is not set yet: the bone starts from where it is shown.
    pub const FRESH: u8 = 2;
    /// A turn by an angle from where the bone is, not from its rest pose.
    pub const RELATIVE: u8 = 4;
    /// The axis is in the bone's own frame, not the model's.
    pub const BONE_AXIS: u8 = 8;
    /// A turn the short way from where the bone is, whatever the angle's
    /// sign says (aiming: tracking round the back).
    pub const SHORT: u8 = 16;
}

/// One clip layer. 28 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layer {
    /// The model's clip `clip - 1`; 0: none (an empty layer; all zeros is
    /// empty channels).
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
        Self { clip: 0, mask: 0, mode: 0, kind: kind::BLEND, flags: 0, reserved: 0, weight: 0.0, target: 0.0, fade: 0.0, speed: 0.0, phase: 0.0 }
    }
}

impl Layer {
    pub fn is_empty(&self) -> bool {
        self.clip == 0
    }

    pub fn ended(&self) -> bool {
        self.flags & layer_flags::ENDED != 0
    }
}

/// One manual part of a bone (its rotation, translation or scale), moving
/// to a target with a trapezoid speed profile. 52 bytes.
///
/// While [`manual_flags::PENDING`], `target` is the command: a turn's axis
/// (xyz, in the model's frame or with [`manual_flags::BONE_AXIS`] the
/// bone's) and angle (w, radians from the rest pose, or from where the bone
/// is with [`manual_flags::RELATIVE`]); a spin's axis; a move's offset from
/// the rest position in the model's frame; a scale's factors on the rest
/// scale. After that it is in the bone's local terms: a rotation quaternion
/// (x, y, z, w; for a spin, the axis), a translation or a scale.
///
/// Rotations are never sign-normalised: the sign of `dot(value, target)`
/// is the way round (positive the short way, negative the long way), so a
/// turn goes from angle to angle as the numbers say (from -100° to 170° is
/// 270° the long way). A bone a clip or a spin left is taken on the rest
/// pose's side, an angle within ±180°.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Manual {
    pub bone: u16,
    /// [`manual_kind`].
    pub kind: u8,
    /// [`manual_flags`].
    pub flags: u8,
    /// Now: a quaternion, or xyz.
    pub value: [f32; 4],
    pub target: [f32; 4],
    /// The top speed, a second (a spin's rate, signed); 0: at once.
    pub speed: f32,
    /// Speed gained a second (0: at once).
    pub accel: f32,
    /// Speed lost a second when closing in or spinning down (0: stops dead).
    pub decel: f32,
    /// The speed now.
    pub rate: f32,
}

impl Manual {
    pub fn is_empty(&self) -> bool {
        self.kind == manual_kind::NONE
    }

    /// At its target (a spin: at its rate). An empty entry is done.
    pub fn done(&self) -> bool {
        match self.kind {
            manual_kind::NONE => true,
            _ if self.flags & manual_flags::PENDING != 0 => false,
            manual_kind::SPIN => self.rate == self.speed,
            _ => self.value == self.target,
        }
    }

    /// Its value shows (not waiting for its start).
    fn shows(&self) -> bool {
        !self.is_empty() && self.flags & manual_flags::FRESH == 0
    }
}

/// A unit's animation channels: what the sim keeps per unit. 528 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Channels {
    pub layers: [Layer; LAYERS],
    pub manual: [Manual; MANUAL],
}

const _: () = assert!(size_of::<Layer>() == 28 && size_of::<Manual>() == 52 && size_of::<Channels>() == 528);

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

/// A baked clip, read only ([`Rig::clip_table`]): the bones it animates
/// (its tracks, ascending), and per frame (`frames` of them, `rate` a
/// second, the last at `duration`) each track's local transform,
/// `keys[frame * tracks.len() + track]`. A frame between two is their lerp
/// (rotations nlerp the short way); a layer samples it at
/// `PlayMode::clip_time(phase, duration) * rate`.
#[derive(Clone, Copy, Debug)]
pub struct ClipTable<'a> {
    pub tracks: &'a [u32],
    pub frames: u32,
    pub rate: f32,
    pub duration: f32,
    pub keys: &'a [Trs],
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
    /// Each bone's rotation in the model's frame at rest.
    rest_frame: Vec<[f32; 4]>,
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
            rest_frame: Vec::new(),
            rest,
        };
        rig.clips = (0..model.clips.len()).map(|c| bake(model, &rig.rest, c)).collect();
        for b in 0..rig.bones() {
            let r = rig.rest[b].r;
            let q = match rig.parents[b] {
                u32::MAX => r,
                p => qnorm(qmul(rig.rest_frame[p as usize], r)),
            };
            rig.rest_frame.push(q);
        }
        rig
    }

    /// `v` (in the model's frame) in the local frame of `bone`'s children,
    /// or of `bone`'s parent (`parent`), by rest rotations only.
    fn in_rest_frame(&self, bone: usize, v: [f32; 3], parent: bool) -> [f32; 3] {
        let b = if parent { self.parents[bone] } else { bone as u32 };
        match b {
            u32::MAX => v,
            b => qrot(qconj(self.rest_frame[b as usize]), v),
        }
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

    /// The clip named `name`, numbered from 1 as [`Layer::clip`].
    pub fn clip(&self, name: &str) -> Option<u16> {
        self.clip_names.iter().position(|n| n == name).map(|c| c as u16 + 1)
    }

    /// The model's clip names (clip `k + 1` is `clip_names()[k]`).
    pub fn clip_names(&self) -> &[String] {
        &self.clip_names
    }

    pub fn bone_names(&self) -> &[String] {
        &self.names
    }

    pub fn masks(&self) -> &[Mask] {
        &self.masks
    }

    /// The bones' parents (`u32::MAX`: a root); parents come first.
    pub fn parents(&self) -> &[u32] {
        &self.parents
    }

    /// The bones' rest transforms (what an unposed part starts from).
    pub fn rest(&self) -> &[Trs] {
        &self.rest
    }

    /// Clip `clip` (numbered from 1) as baked: what [`Rig::pose`] samples,
    /// for the GPU to sample the same keys.
    pub fn clip_table(&self, clip: u16) -> Option<ClipTable<'_>> {
        self.baked(clip).map(|c| ClipTable { tracks: &c.tracks, frames: c.frames, rate: c.rate, duration: c.duration, keys: &c.keys })
    }

    fn baked(&self, clip: u16) -> Option<&Baked> {
        self.clips.get((clip as usize).wrapping_sub(1))
    }

    /// Clip `clip`'s length in seconds (0 for none).
    pub fn duration(&self, clip: u16) -> f32 {
        self.baked(clip).map_or(0.0, |c| c.duration)
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
    /// [`Layer::mask`].
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
    /// its end ([`layer_flags::ENDED`]); new clips take back their bones;
    /// manual entries take their commands and move toward their targets.
    pub fn advance(&self, ch: &mut Channels, dt: f32, ground_speed: f32) {
        self.take_back(ch);
        self.start_manual(ch);
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
        for m in ch.manual.iter_mut() {
            match m.kind {
                manual_kind::TURN => step_turn(m, dt),
                manual_kind::SPIN => {
                    let up = m.speed * m.rate >= 0.0 && m.speed.abs() > m.rate.abs();
                    let a = if up { m.accel } else { m.decel };
                    m.rate = toward(m.rate, m.speed, a * dt, a <= 0.0);
                    if m.rate != 0.0 {
                        m.value = qnorm(qmul(m.value, axis_angle([m.target[0], m.target[1], m.target[2]], m.rate * dt)));
                        // On the rest pose's side, so a turn after it goes
                        // the way its angle says.
                        if self.rest.get(m.bone as usize).is_some_and(|r| dot4(m.value, r.r) < 0.0) {
                            m.value = neg4(m.value);
                        }
                    }
                }
                manual_kind::MOVE | manual_kind::SCALE => step_line(m, dt),
                _ => {}
            }
        }
    }

    /// Clips just played take back the bones they animate within their mask
    /// from manual entries given before this tick.
    fn take_back(&self, ch: &mut Channels) {
        for i in 0..LAYERS {
            let l = ch.layers[i];
            if l.flags & layer_flags::NEW == 0 {
                continue;
            }
            ch.layers[i].flags &= !layer_flags::NEW;
            let Some(c) = self.baked(l.clip).filter(|_| l.kind != kind::ADDITIVE) else { continue };
            let mask = self.mask_weights(l.mask);
            for m in ch.manual.iter_mut() {
                let b = m.bone as usize;
                let covered = c.tracks.binary_search(&(b as u32)).is_ok() && mask.map_or(1.0, |w| w.get(b).copied().unwrap_or(0.0)) > 0.0;
                if !m.is_empty() && m.flags & manual_flags::PENDING == 0 && covered {
                    *m = Manual::default();
                }
            }
        }
    }

    /// Pending manual entries into their bones' terms; fresh ones start from
    /// where their bone is shown.
    fn start_manual(&self, ch: &mut Channels) {
        let before = *ch;
        for k in 0..MANUAL {
            let m = ch.manual[k];
            if !m.is_empty() && m.flags & manual_flags::PENDING != 0 {
                ch.manual[k] = self.resolve(&before, m).unwrap_or_default();
            }
        }
    }

    /// Pending entry `m` in its bone's terms, against channels `before`
    /// (`None`: no such bone).
    fn resolve(&self, before: &Channels, mut m: Manual) -> Option<Manual> {
        let b = m.bone as usize;
        let rest = *self.rest.get(b)?;
        if m.flags & manual_flags::FRESH != 0 {
            let manual = before.manual.iter().any(|o| o.shows() && o.bone == m.bone);
            let shown = if manual { rest } else { self.clip_local(before, b) };
            m.value = match m.kind {
                manual_kind::TURN | manual_kind::SPIN if dot4(shown.r, rest.r) < 0.0 => neg4(shown.r),
                manual_kind::TURN | manual_kind::SPIN => shown.r,
                manual_kind::MOVE => [shown.t[0], shown.t[1], shown.t[2], 0.0],
                _ => [shown.s[0], shown.s[1], shown.s[2], 0.0],
            };
            m.rate = 0.0;
        }
        let v = [m.target[0], m.target[1], m.target[2]];
        let axis = || norm3(if m.flags & manual_flags::BONE_AXIS != 0 { v } else { self.in_rest_frame(b, v, false) });
        m.target = match m.kind {
            manual_kind::TURN => {
                let q = axis_angle(axis(), m.target[3]);
                let t = qnorm(qmul(if m.flags & manual_flags::RELATIVE != 0 { m.value } else { rest.r }, q));
                if m.flags & manual_flags::SHORT != 0 && dot4(m.value, t) < 0.0 { neg4(t) } else { t }
            }
            manual_kind::SPIN => {
                let a = axis();
                [a[0], a[1], a[2], 0.0]
            }
            manual_kind::MOVE => {
                let o = self.in_rest_frame(b, v, true);
                [rest.t[0] + o[0], rest.t[1] + o[1], rest.t[2] + o[2], 0.0]
            }
            _ => [rest.s[0] * v[0], rest.s[1] * v[1], rest.s[2] * v[2], 0.0],
        };
        m.flags = 0;
        Some(m)
    }

    /// Direction `d` (in the unit's frame) as `bone` sees it, in the frame
    /// its turns are given in (the unit's axes as at rest), with the bone's
    /// parents **where they are going**: a manual rotation at its target
    /// (pending commands included), a clipped bone at its clips' value now.
    /// Turning the bone so that a direction given at rest points along the
    /// result points it along `d`. With the parents at rest it is `d`.
    pub fn dir_in(&self, ch: &Channels, bone: u16, d: [f32; 3]) -> [f32; 3] {
        let b = bone as usize;
        if b >= self.bones() {
            return d;
        }
        // The parents' chain, root first.
        let mut chain = [0u32; 64];
        let mut n = 0;
        let mut p = self.parents[b];
        while p != u32::MAX && n < chain.len() {
            chain[n] = p;
            n += 1;
            p = self.parents[p as usize];
        }
        let mut g = [0.0, 0.0, 0.0, 1.0];
        for &a in chain[..n].iter().rev() {
            g = qnorm(qmul(g, self.going(ch, a as usize)));
        }
        // The bone's frame at rest under its parents' pose, back to the
        // unit's axes as they are at rest.
        let frame = qnorm(qmul(g, self.rest[b].r));
        qrot(self.rest_frame[b], qrot(qconj(frame), d))
    }

    /// Bone `b`'s local rotation where it is going (see [`Rig::dir_in`]).
    fn going(&self, ch: &Channels, b: usize) -> [f32; 4] {
        let mut manual = false;
        for m in ch.manual.iter().filter(|m| !m.is_empty() && m.bone as usize == b) {
            manual |= m.shows();
            if matches!(m.kind, manual_kind::TURN | manual_kind::SPIN) {
                let m = if m.flags & manual_flags::PENDING != 0 { self.resolve(ch, *m).unwrap_or(*m) } else { *m };
                return if m.kind == manual_kind::TURN { m.target } else { m.value };
            }
        }
        if manual { self.rest[b].r } else { self.clip_local(ch, b).r }
    }

    /// Bone `b`'s local transform from the layers alone.
    fn clip_local(&self, ch: &Channels, b: usize) -> Trs {
        let mut p = self.rest[b];
        for l in &ch.layers {
            if l.is_empty() || l.weight <= 0.0 {
                continue;
            }
            let Some(c) = self.baked(l.clip) else { continue };
            let Ok(i) = c.tracks.binary_search(&(b as u32)) else { continue };
            let w = (l.weight * self.mask_weights(l.mask).map_or(1.0, |m| m[b])).min(1.0);
            if w > 0.0 {
                let (fa, fb, s) = c.frames_at(l.phase, PlayMode::from_bits(l.mode as u32));
                layer_onto(&mut p, l.kind, c, i, (fa, fb, s), w);
            }
        }
        p
    }

    /// The local transforms for `ch`, into `out` (from [`Rig::rest_pose`]).
    pub fn pose(&self, ch: &Channels, out: &mut Pose) {
        out.local.copy_from_slice(&self.rest);
        out.posed.fill(false);
        for l in &ch.layers {
            if l.is_empty() || l.weight <= 0.0 {
                continue;
            }
            let Some(c) = self.baked(l.clip) else { continue };
            let mask = self.mask_weights(l.mask);
            let frames = c.frames_at(l.phase, PlayMode::from_bits(l.mode as u32));
            for (i, &bone) in c.tracks.iter().enumerate() {
                let w = (l.weight * mask.map_or(1.0, |m| m[bone as usize])).min(1.0);
                if w > 0.0 {
                    layer_onto(&mut out.local[bone as usize], l.kind, c, i, frames, w);
                    out.posed[bone as usize] = true;
                }
            }
        }
        // Manual bones: their rest pose, with their entries' values.
        let shown = || ch.manual.iter().filter(|m| m.shows() && (m.bone as usize) < self.bones());
        for m in shown() {
            out.local[m.bone as usize] = self.rest[m.bone as usize];
        }
        for m in shown() {
            let (b, v) = (m.bone as usize, m.value);
            let p = &mut out.local[b];
            match m.kind {
                manual_kind::TURN | manual_kind::SPIN => p.r = v,
                manual_kind::MOVE => p.t = [v[0], v[1], v[2]],
                _ => p.s = [v[0], v[1], v[2]],
            }
            out.posed[b] = true;
        }
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

/// A joint key for the GPU (the pose pass, `units_pose.comp`): a bone's
/// local transform at a tick, 12 words: translation and the tick (the
/// `u32` itself, not a float: exact at any tick), rotation (x, y, z, w),
/// scale and a spare word. The GPU draws a bone on the line through its
/// two keys ([`on_line`]).
pub type Key = [u32; 12];

/// `p` at `tick` as a key.
pub fn key(p: &Trs, tick: u32) -> Key {
    let f = |x: f32| x.to_bits();
    [f(p.t[0]), f(p.t[1]), f(p.t[2]), tick, f(p.r[0]), f(p.r[1]), f(p.r[2]), f(p.r[3]), f(p.s[0]), f(p.s[1]), f(p.s[2]), 0]
}

/// A key's transform and tick.
pub fn key_trs(k: &Key) -> (Trs, u32) {
    let f = |i: usize| f32::from_bits(k[i]);
    (Trs { t: [f(0), f(1), f(2)], r: [f(4), f(5), f(6), f(7)], s: [f(8), f(9), f(10)] }, k[3])
}

/// The transform at `tick + frac` on the line through keys `a` and `b`
/// (older first; past `b` too): translation and scale lerped, rotation
/// nlerped as stored (the keys on one hemisphere). The tick difference is
/// taken in integers first, as the pose pass does. Equal ticks: `b`.
pub fn on_line(a: &Key, b: &Key, tick: u32, frac: f32) -> Trs {
    let ((pa, ta), (pb, tb)) = (key_trs(a), key_trs(b));
    let span = tb.wrapping_sub(ta) as i32;
    if span == 0 {
        return pb;
    }
    let s = (tick.wrapping_sub(ta) as i32 as f32 + frac) / span as f32;
    let l = |x: [f32; 3], y: [f32; 3]| [x[0] + (y[0] - x[0]) * s, x[1] + (y[1] - x[1]) * s, x[2] + (y[2] - x[2]) * s];
    let r = qnorm(add4(pa.r, [(pb.r[0] - pa.r[0]) * s, (pb.r[1] - pa.r[1]) * s, (pb.r[2] - pa.r[2]) * s, (pb.r[3] - pa.r[3]) * s]));
    Trs { t: l(pa.t, pb.t), r, s: l(pa.s, pb.s) }
}

/// How far a drawn transform may be from the posed one (a joint key line's
/// value against the pose: within it, no new key).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    translation_sq: f32,
    /// sin(angle / 2), squared.
    half_sin_sq: f32,
    scale: f32,
}

impl Tolerance {
    /// A `translation` distance in the model's units, an `angle` in
    /// radians, a `scale` difference per axis.
    pub fn new(translation: f32, angle: f32, scale: f32) -> Self {
        let half = libm::sinf(0.5 * angle.clamp(0.0, core::f32::consts::PI));
        Self { translation_sq: translation * translation, half_sin_sq: half * half, scale }
    }
}

/// `a` and `b` are within `tol` of each other (the rotations' angle by the
/// vector part of their difference, either sign: exact for small angles).
pub fn within(a: &Trs, b: &Trs, tol: &Tolerance) -> bool {
    let d = |x: [f32; 3], y: [f32; 3]| [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    let t = d(a.t, b.t);
    if t[0] * t[0] + t[1] * t[1] + t[2] * t[2] > tol.translation_sq {
        return false;
    }
    if d(a.s, b.s).iter().any(|x| x.abs() > tol.scale) {
        return false;
    }
    // sin(angle / 2) = |vector part of conj(a) b| for unit quaternions.
    let q = qmul(qconj(a.r), b.r);
    q[0] * q[0] + q[1] * q[1] + q[2] * q[2] <= tol.half_sin_sq
}

/// `b`'s rotation on `a`'s hemisphere (a key's rotation before it goes on
/// a line from `a`: manual rotations are never sign-normalised).
pub fn same_side(a: &Trs, mut b: Trs) -> Trs {
    if dot4(a.r, b.r) < 0.0 {
        b.r = neg4(b.r);
    }
    b
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

/// The speed for the next tick of a move with `remaining` to go: up by the
/// acceleration to the top speed, and down to what the deceleration can
/// still stop from.
fn profile(m: &Manual, remaining: f32, dt: f32) -> f32 {
    let mut r = if m.accel > 0.0 { (m.rate + m.accel * dt).min(m.speed) } else { m.speed };
    if m.decel > 0.0 {
        r = r.min(libm::sqrtf(2.0 * m.decel * remaining));
    }
    r
}

/// A turn's tick: along the great arc from `value` to `target` (as signed,
/// so the long way when their dot is negative).
fn step_turn(m: &mut Manual, dt: f32) {
    let (a, b) = (m.value, m.target);
    let diff = libm::sqrtf(dot4(sub4(b, a), sub4(b, a)));
    let sum = libm::sqrtf(dot4(add4(b, a), add4(b, a)));
    // The angle between the two as 4-vectors; the rotation is twice it.
    let arc = 2.0 * libm::atan2f(diff, sum);
    let remaining = 2.0 * arc;
    if m.speed <= 0.0 || remaining <= 0.0 {
        (m.value, m.rate) = (b, 0.0);
        return;
    }
    let r = profile(m, remaining, dt);
    let step = r * dt;
    let sn = libm::sinf(arc);
    if step >= remaining || (sn < 1e-4 && arc > 1.0) {
        (m.value, m.rate) = (b, 0.0);
        return;
    }
    let s = step / remaining;
    let (wa, wb) = if sn < 1e-4 { (1.0 - s, s) } else { (libm::sinf((1.0 - s) * arc) / sn, libm::sinf(s * arc) / sn) };
    m.value = qnorm([a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb, a[2] * wa + b[2] * wb, a[3] * wa + b[3] * wb]);
    m.rate = r;
}

/// A move's or scale's tick: along the straight line to `target`.
fn step_line(m: &mut Manual, dt: f32) {
    let d = sub4(m.target, m.value);
    let dist = libm::sqrtf(dot4(d, d));
    let r = if m.speed > 0.0 { profile(m, dist, dt) } else { 0.0 };
    let step = r * dt;
    if m.speed <= 0.0 || step >= dist {
        (m.value, m.rate) = (m.target, 0.0);
    } else {
        let s = step / dist;
        m.value = [m.value[0] + d[0] * s, m.value[1] + d[1] * s, m.value[2] + d[2] * s, 0.0];
        m.rate = r;
    }
}

/// Layer key `i` of `c` (between frames `a` and `b` by `s`) onto `p`, by
/// weight `w`: blended over it, or its change from the clip's first frame
/// added.
fn layer_onto(p: &mut Trs, kind: u8, c: &Baked, i: usize, (a, b, s): (usize, usize, f32), w: f32) {
    let k = lerp_trs(c.key(a, i), c.key(b, i), s);
    if kind == kind::ADDITIVE {
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
}

fn add4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

fn sub4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

fn neg4(a: [f32; 4]) -> [f32; 4] {
    [-a[0], -a[1], -a[2], -a[3]]
}

fn norm3(v: [f32; 3]) -> [f32; 3] {
    let len = libm::sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if len > 0.0 { [v[0] / len, v[1] / len, v[2] / len] } else { [0.0, 1.0, 0.0] }
}

/// `v` rotated by `q`.
fn qrot(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let r = qmul(qmul(q, [v[0], v[1], v[2], 0.0]), qconj(q));
    [r[0], r[1], r[2]]
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
