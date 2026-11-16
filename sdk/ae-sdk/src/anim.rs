//! Animation channels (feature `anim`, M14): what a unit's model is doing,
//! as synced columns. A system (or a unit script) sets targets with
//! [`Anim::play`], [`Anim::turn`] and friends; the engine's `anim` service
//! (`host(anim)`, `ae_services::anim`) advances them every tick without
//! calling the module: clip phases (by time, or by the unit's ground speed
//! for no foot sliding), layer fades, manual joints toward their targets.
//! The pose is a pure function of these columns (`ae_model::anim`),
//! evaluated by the client for drawing and by the engine for joint queries.
//!
//! Per unit: [`LAYERS`] clip layers, applied in index order (each a clip
//! with a play mode, a weight and a joint mask; blended over the layers
//! below, or added to them), and [`MANUAL`] manual entries.
//!
//! **A bone is either clipped or manual.** Turning, spinning, moving or
//! scaling a bone takes it away from the clips (its children keep theirs);
//! [`Anim::release`] gives it back, and so does playing a clip that
//! animates it (from the next tick; a command given in the same tick as the
//! `play` stays). A bone the clips were driving starts from where they had
//! it. A bone has at most one rotation entry (turn or spin), one move and
//! one scale: the next command replaces it.
//!
//! Axes are in the unit's frame ([`Axis::UP`], [`Axis::FORWARD`],
//! [`Axis::RIGHT`]), as the bone is at rest; the engine turns them into the
//! bone's own frame (glTF bone frames are arbitrary).
//!
//! Clips, bones and masks are the model's, found by name with [`clip`],
//! [`bone`] and [`mask`] (the service publishes them from the content's
//! models; look them up once and keep the numbers).

use crate::{Component, abi};

pub const LAYERS: usize = 4;
pub const MANUAL: usize = 8;

/// [`Anim::mode`].
pub mod mode {
    pub const LOOP: u8 = 0;
    pub const PING_PONG: u8 = 1;
    /// Once, then holding the last frame ([`super::Anim::ended`]).
    pub const ONCE: u8 = 2;
}

/// [`Anim::kind`].
pub mod kind {
    /// Mixed over the layers below by the weight (on a mask: an override).
    pub const BLEND: u8 = 0;
    /// The clip's change from its first frame added to the layers below.
    pub const ADDITIVE: u8 = 1;
}

/// [`Anim::flags`].
pub mod layer_flags {
    /// `speed` is clip seconds per metre moved.
    pub const GROUND_SPEED: u8 = 1;
    /// Set by the engine when a clip played once reached its end.
    pub const ENDED: u8 = 2;
    /// Just played: the engine hands the clip's bones back from manual
    /// entries, then clears it.
    pub const NEW: u8 = 4;
}

/// [`Anim::manual_kind`].
pub mod manual_kind {
    pub const NONE: u8 = 0;
    pub const TURN: u8 = 1;
    pub const SPIN: u8 = 2;
    pub const MOVE: u8 = 3;
    pub const SCALE: u8 = 4;
}

/// [`Anim::manual_flags`] (the engine's bookkeeping).
pub mod manual_flags {
    pub const PENDING: u8 = 1;
    pub const FRESH: u8 = 2;
    pub const RELATIVE: u8 = 4;
    pub const BONE_AXIS: u8 = 8;
    pub const SHORT: u8 = 16;
}

/// An axis for a turn or a spin: in the unit's frame (x forward, y up,
/// z right) or in the bone's own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axis {
    pub v: [f32; 3],
    pub bone: bool,
}

impl Axis {
    /// Turning about it by a positive angle turns forward to the left.
    pub const UP: Axis = Axis::model([0.0, 1.0, 0.0]);
    /// Turning about it by a positive angle rolls up to the right.
    pub const FORWARD: Axis = Axis::model([1.0, 0.0, 0.0]);
    /// Turning about it by a positive angle tips forward up.
    pub const RIGHT: Axis = Axis::model([0.0, 0.0, 1.0]);

    /// In the unit's (the model's) frame.
    pub const fn model(v: [f32; 3]) -> Axis {
        Axis { v, bone: false }
    }

    /// In the bone's own frame (as the model file has it).
    pub const fn bone(v: [f32; 3]) -> Axis {
        Axis { v, bone: true }
    }
}

/// Degrees in radians.
pub const fn deg(degrees: f32) -> f32 {
    degrees * (core::f32::consts::PI / 180.0)
}

/// A unit's animation channels. Layer `i` is the `[i]` of the layer
/// arrays, manual entry `k` the `[k]` of the manual arrays (the engine's
/// `ae_model::anim::Channels`, by field).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_000E)]
pub struct Anim {
    /// The model: [`model_id`] of its name (a def sets it by name).
    #[field(view = "los")]
    pub model: u64,
    /// The model's clip, from 1; 0: an empty layer.
    #[field(view = "los")]
    pub clip: [u16; LAYERS],
    /// 0: the whole model; else a mask ([`mask`]).
    #[field(view = "los")]
    pub mask: [u16; LAYERS],
    /// [`mode`].
    #[field(view = "los")]
    pub mode: [u8; LAYERS],
    /// [`kind`].
    #[field(view = "los")]
    pub kind: [u8; LAYERS],
    /// [`layer_flags`].
    #[field(view = "los")]
    pub flags: [u8; LAYERS],
    pub reserved: [u8; LAYERS],
    #[field(view = "los")]
    pub weight: [f32; LAYERS],
    /// Where the weight goes at `fade` a second; a layer at weight 0 going
    /// to 0 is cleared.
    #[field(view = "los")]
    pub target: [f32; LAYERS],
    #[field(view = "los")]
    pub fade: [f32; LAYERS],
    /// Clip seconds a second (or a metre); negative plays backwards.
    #[field(view = "los")]
    pub speed: [f32; LAYERS],
    /// Seconds into the clip.
    #[field(view = "los")]
    pub phase: [f32; LAYERS],
    #[field(view = "los")]
    pub bone: [u16; MANUAL],
    /// [`manual_kind`]; [`manual_kind::NONE`]: an empty entry.
    #[field(view = "los")]
    pub manual_kind: [u8; MANUAL],
    /// [`manual_flags`].
    #[field(view = "los")]
    pub manual_flags: [u8; MANUAL],
    /// Now, in the bone's terms (a quaternion, or xyz).
    #[field(view = "los")]
    pub value: [[f32; 4]; MANUAL],
    /// The command until the engine takes it, then the goal in the bone's
    /// terms (`ae_model::anim::Manual`).
    #[field(view = "los")]
    pub goal: [[f32; 4]; MANUAL],
    /// Top speed a second (a spin's rate, signed); 0: at once.
    #[field(view = "los")]
    pub top_speed: [f32; MANUAL],
    /// Speed gained a second; 0: at once.
    #[field(view = "los")]
    pub accel: [f32; MANUAL],
    /// Speed lost a second closing in or spinning down; 0: stops dead.
    #[field(view = "los")]
    pub decel: [f32; MANUAL],
    /// The speed now.
    #[field(view = "los")]
    pub rate: [f32; MANUAL],
}

impl Anim {
    /// Nothing playing on model `model` ([`model_id`]).
    pub const fn new(model: u64) -> Self {
        Self {
            model,
            clip: [0; LAYERS],
            mask: [0; LAYERS],
            mode: [0; LAYERS],
            kind: [0; LAYERS],
            flags: [0; LAYERS],
            reserved: [0; LAYERS],
            weight: [0.0; LAYERS],
            target: [0.0; LAYERS],
            fade: [0.0; LAYERS],
            speed: [0.0; LAYERS],
            phase: [0.0; LAYERS],
            bone: [0; MANUAL],
            manual_kind: [0; MANUAL],
            manual_flags: [0; MANUAL],
            value: [[0.0; 4]; MANUAL],
            goal: [[0.0; 4]; MANUAL],
            top_speed: [0.0; MANUAL],
            accel: [0.0; MANUAL],
            decel: [0.0; MANUAL],
            rate: [0.0; MANUAL],
        }
    }

    /// Play `clip` on `layer` from its start, looping, at full weight over
    /// the whole model; adjust with the returned [`LayerSet`]. Bones the
    /// clip animates (within its mask) leave their manual entries, unless
    /// the layer is made additive.
    pub fn play(&mut self, layer: usize, clip: u16) -> LayerSet<'_> {
        self.clip[layer] = clip;
        self.mask[layer] = 0;
        self.mode[layer] = mode::LOOP;
        self.kind[layer] = kind::BLEND;
        self.flags[layer] = layer_flags::NEW;
        self.weight[layer] = 1.0;
        self.target[layer] = 1.0;
        self.fade[layer] = 0.0;
        self.speed[layer] = 1.0;
        self.phase[layer] = 0.0;
        LayerSet { anim: self, layer }
    }

    /// Change `layer` in place (its clip keeps playing).
    pub fn layer(&mut self, layer: usize) -> LayerSet<'_> {
        LayerSet { anim: self, layer }
    }

    /// Fade `layer` out over `ms` and clear it (0: clear it now).
    pub fn stop(&mut self, layer: usize, ms: u32) {
        if ms == 0 {
            self.clear(layer);
        } else {
            self.target[layer] = 0.0;
            self.fade[layer] = 1000.0 / ms as f32;
        }
    }

    fn clear(&mut self, layer: usize) {
        self.clip[layer] = 0;
        self.mask[layer] = 0;
        self.mode[layer] = 0;
        self.kind[layer] = 0;
        self.flags[layer] = 0;
        self.weight[layer] = 0.0;
        self.target[layer] = 0.0;
        self.fade[layer] = 0.0;
        self.speed[layer] = 0.0;
        self.phase[layer] = 0.0;
    }

    /// The clip on `layer` (0: none).
    pub fn playing(&self, layer: usize) -> u16 {
        self.clip[layer]
    }

    /// A clip played once on `layer` has reached its end.
    pub fn ended(&self, layer: usize) -> bool {
        self.flags[layer] & layer_flags::ENDED != 0
    }

    /// Turn `bone` about `axis` to `angle` radians from its rest pose, at
    /// up to `speed` radians a second (0: at once). From where the bone is
    /// to `angle` as the numbers say: from -100° to 170° is 270° the long
    /// way. A bone a clip or a spin left is taken as within ±180°. Less
    /// than a full turn; more is a spin.
    pub fn turn(&mut self, bone: u16, axis: Axis, angle: f32, speed: f32) -> ManualSet<'_> {
        self.command(bone, manual_kind::TURN, axis.bone, [axis.v[0], axis.v[1], axis.v[2], angle], speed)
    }

    /// Turn `bone` by `angle` from where it is.
    pub fn turn_by(&mut self, bone: u16, axis: Axis, angle: f32, speed: f32) -> ManualSet<'_> {
        let set = self.turn(bone, axis, angle, speed);
        set.anim.manual_flags[set.slot] |= manual_flags::RELATIVE;
        set
    }

    /// Turn `bone` the shortest way so that `pointing` (where it points at
    /// rest, usually [`Axis::FORWARD`]) points along `dir`, a direction as
    /// the bone sees it ([`Anim::dir_in`]). For one free joint (a head
    /// looking); a joint that turns about one axis takes [`aim_angle`].
    pub fn turn_toward(&mut self, bone: u16, pointing: Axis, dir: [f32; 3], speed: f32) -> ManualSet<'_> {
        let (p, d) = (norm(pointing.v), norm(dir));
        let c = cross(p, d);
        let (s, cos) = (libm::sqrtf(dot(c, c)), dot(p, d));
        let axis = if s > 1e-6 {
            [c[0] / s, c[1] / s, c[2] / s]
        } else {
            // Along or against it: any axis across it.
            let o = if p[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            norm(cross(p, o))
        };
        self.turn(bone, Axis { v: axis, bone: pointing.bone }, libm::atan2f(s, cos), speed).short()
    }

    /// Direction `dir` (in the unit's frame) as `bone` sees it, with its
    /// parents where they are going (a turn given to a parent counts as
    /// done, a clipped parent is where its clips have it now): what
    /// [`aim_angle`] and [`Anim::turn_toward`] take. Give the parents'
    /// turns first: a barrel's direction is then against the turned turret.
    /// The engine evaluates the pose (`ae_model::anim::Rig::dir_in`).
    pub fn dir_in(&self, bone: u16, dir: [f32; 3]) -> [f32; 3] {
        const N: usize = size_of::<Anim>();
        let mut input = [0u8; N + 16];
        // SAFETY: `Anim` is plain data (`repr(C)`, no padding read back).
        input[..N].copy_from_slice(unsafe { core::slice::from_raw_parts((self as *const Anim).cast::<u8>(), N) });
        input[N..N + 4].copy_from_slice(&(bone as u32).to_le_bytes());
        for (k, x) in dir.iter().enumerate() {
            input[N + 4 + 4 * k..N + 8 + 4 * k].copy_from_slice(&x.to_le_bytes());
        }
        let mut out = [0f32; 3];
        let op = abi::name_id(abi::query::ANIM_DIR_IN);
        let n = unsafe { crate::host::ae_query(op, input.as_ptr() as usize as u32, input.len() as u32, out.as_mut_ptr() as usize as u32, 12) };
        assert!(n == 12, "the engine has no animation service (`ae_query` {})", n);
        out
    }

    /// Spin `bone` about `axis` at `rate` radians a second (signed), for
    /// ever; reached at once unless [`ManualSet::accel`].
    pub fn spin(&mut self, bone: u16, axis: Axis, rate: f32) -> ManualSet<'_> {
        self.command(bone, manual_kind::SPIN, axis.bone, [axis.v[0], axis.v[1], axis.v[2], 0.0], rate)
    }

    /// Slow `bone`'s spin to a stop at `decel` radians a second squared
    /// (0: at once); it then holds where it stopped.
    pub fn stop_spin(&mut self, bone: u16, decel: f32) {
        if let Some(k) = self.find(bone, manual_kind::SPIN) {
            self.top_speed[k] = 0.0;
            self.decel[k] = decel;
        }
    }

    /// Move `bone` to `offset` from its rest position, in the unit's frame,
    /// at up to `speed` a second (0: at once).
    pub fn move_to(&mut self, bone: u16, offset: [f32; 3], speed: f32) -> ManualSet<'_> {
        self.command(bone, manual_kind::MOVE, false, [offset[0], offset[1], offset[2], 0.0], speed)
    }

    /// Scale `bone` to `factor` times its rest scale (on its own axes), at
    /// up to `speed` a second (0: at once).
    pub fn scale_to(&mut self, bone: u16, factor: [f32; 3], speed: f32) -> ManualSet<'_> {
        self.command(bone, manual_kind::SCALE, false, [factor[0], factor[1], factor[2], 0.0], speed)
    }

    /// Give `bone` back to the clips (its manual entries cleared).
    pub fn release(&mut self, bone: u16) {
        for k in 0..MANUAL {
            if self.manual_kind[k] != manual_kind::NONE && self.bone[k] == bone {
                self.clear_manual(k);
            }
        }
    }

    /// `bone` is not turning (its turn reached, its spin at its rate, or
    /// neither).
    pub fn turned(&self, bone: u16) -> bool {
        self.find(bone, manual_kind::TURN).is_none_or(|k| self.manual_done(k))
    }

    /// `bone`'s move has arrived (or it has none).
    pub fn moved(&self, bone: u16) -> bool {
        self.find(bone, manual_kind::MOVE).is_none_or(|k| self.manual_done(k))
    }

    /// `bone`'s scale has arrived (or it has none).
    pub fn scaled(&self, bone: u16) -> bool {
        self.find(bone, manual_kind::SCALE).is_none_or(|k| self.manual_done(k))
    }

    /// `bone` is manual: the clips do not move it.
    pub fn is_manual(&self, bone: u16) -> bool {
        (0..MANUAL).any(|k| self.manual_kind[k] != manual_kind::NONE && self.bone[k] == bone)
    }

    fn manual_done(&self, k: usize) -> bool {
        match self.manual_kind[k] {
            manual_kind::NONE => true,
            _ if self.manual_flags[k] & manual_flags::PENDING != 0 => false,
            manual_kind::SPIN => self.rate[k] == self.top_speed[k],
            _ => self.value[k] == self.goal[k],
        }
    }

    /// The entry for `bone`'s part that `kind` drives (turn and spin are
    /// one: the rotation).
    fn find(&self, bone: u16, kind: u8) -> Option<usize> {
        let part = |k: u8| if k == manual_kind::SPIN { manual_kind::TURN } else { k };
        (0..MANUAL).find(|&k| self.manual_kind[k] != manual_kind::NONE && self.bone[k] == bone && part(self.manual_kind[k]) == part(kind))
    }

    fn command(&mut self, bone: u16, kind: u8, bone_axis: bool, goal: [f32; 4], speed: f32) -> ManualSet<'_> {
        // The bone's entry for this part, else a free one, else the last.
        let found = self.find(bone, kind);
        let k = found.or_else(|| (0..MANUAL).find(|&k| self.manual_kind[k] == manual_kind::NONE)).unwrap_or(MANUAL - 1);
        let mut flags = manual_flags::PENDING;
        if found.is_none() {
            self.clear_manual(k);
            flags |= manual_flags::FRESH;
        } else {
            // Still waiting for its start, or now a different motion.
            flags |= self.manual_flags[k] & manual_flags::FRESH;
            if self.manual_kind[k] != kind {
                self.rate[k] = 0.0;
            }
        }
        if bone_axis {
            flags |= manual_flags::BONE_AXIS;
        }
        self.bone[k] = bone;
        self.manual_kind[k] = kind;
        self.manual_flags[k] = flags;
        self.goal[k] = goal;
        self.top_speed[k] = speed;
        self.accel[k] = 0.0;
        self.decel[k] = 0.0;
        ManualSet { anim: self, slot: k }
    }

    fn clear_manual(&mut self, k: usize) {
        self.bone[k] = 0;
        self.manual_kind[k] = 0;
        self.manual_flags[k] = 0;
        self.value[k] = [0.0; 4];
        self.goal[k] = [0.0; 4];
        self.top_speed[k] = 0.0;
        self.accel[k] = 0.0;
        self.decel[k] = 0.0;
        self.rate[k] = 0.0;
    }
}

/// Settings of one layer ([`Anim::play`]).
pub struct LayerSet<'a> {
    anim: &'a mut Anim,
    layer: usize,
}

impl LayerSet<'_> {
    pub fn looping(self) -> Self {
        self.anim.mode[self.layer] = mode::LOOP;
        self
    }

    pub fn ping_pong(self) -> Self {
        self.anim.mode[self.layer] = mode::PING_PONG;
        self
    }

    pub fn once(self) -> Self {
        self.anim.mode[self.layer] = mode::ONCE;
        self.anim.flags[self.layer] &= !layer_flags::ENDED;
        self
    }

    /// Clip seconds a second (1: as authored; negative: backwards).
    pub fn speed(self, speed: f32) -> Self {
        self.anim.speed[self.layer] = speed;
        self.anim.flags[self.layer] &= !layer_flags::GROUND_SPEED;
        self
    }

    /// Clip seconds per metre the unit moves (no foot sliding).
    pub fn per_metre(self, seconds: f32) -> Self {
        self.anim.speed[self.layer] = seconds;
        self.anim.flags[self.layer] |= layer_flags::GROUND_SPEED;
        self
    }

    /// Added to the layers below (its change from the clip's first frame);
    /// it leaves manual bones alone.
    pub fn additive(self) -> Self {
        self.anim.kind[self.layer] = kind::ADDITIVE;
        self
    }

    /// Only on mask `mask` ([`mask`]): over the layers below, an override.
    pub fn mask(self, mask: u16) -> Self {
        self.anim.mask[self.layer] = mask;
        self
    }

    /// The weight to hold (1 by default).
    pub fn weight(self, weight: f32) -> Self {
        self.anim.weight[self.layer] = weight;
        self.anim.target[self.layer] = weight;
        self
    }

    /// From weight 0 to the target over `ms`.
    pub fn blend_in(self, ms: u32) -> Self {
        if ms > 0 {
            self.anim.weight[self.layer] = 0.0;
            self.anim.fade[self.layer] = self.anim.target[self.layer] / (ms as f32 / 1000.0);
        }
        self
    }

    /// To weight `weight` over `ms` from where it is.
    pub fn fade_to(self, weight: f32, ms: u32) -> Self {
        self.anim.target[self.layer] = weight;
        let d = weight - self.anim.weight[self.layer];
        self.anim.fade[self.layer] = if ms == 0 { 0.0 } else { (if d < 0.0 { -d } else { d }) / (ms as f32 / 1000.0) };
        self
    }

    /// Start `seconds` into the clip.
    pub fn at(self, seconds: f32) -> Self {
        self.anim.phase[self.layer] = seconds;
        self
    }
}

/// Settings of one manual entry ([`Anim::turn`] and friends).
pub struct ManualSet<'a> {
    anim: &'a mut Anim,
    slot: usize,
}

impl ManualSet<'_> {
    /// Speed up at `accel` a second squared (0: at once).
    pub fn accel(self, accel: f32) -> Self {
        self.anim.accel[self.slot] = accel;
        self
    }

    /// Slow down to stop at the target at `decel` a second squared (0:
    /// stop dead).
    pub fn decel(self, decel: f32) -> Self {
        self.anim.decel[self.slot] = decel;
        self
    }

    /// The same both ways.
    pub fn ramp(self, accel: f32) -> Self {
        self.accel(accel).decel(accel)
    }

    /// A turn the short way from where the bone is, whatever the angle
    /// says (aiming: tracking a target round the back).
    pub fn short(self) -> Self {
        self.anim.manual_flags[self.slot] |= manual_flags::SHORT;
        self
    }
}

/// The angle to turn a bone about `axis` (from its rest pose) so that
/// `pointing` (where it points at rest) comes closest to `dir`, a direction
/// as the bone sees it ([`Anim::dir_in`]): both across the axis, one
/// `atan2`. A turret about [`Axis::UP`], a barrel about [`Axis::RIGHT`],
/// both pointing [`Axis::FORWARD`]. 0 when `dir` is along the axis.
pub fn aim_angle(axis: Axis, pointing: Axis, dir: [f32; 3]) -> f32 {
    let a = norm(axis.v);
    let across = |v: [f32; 3]| {
        let k = dot(a, v);
        [v[0] - a[0] * k, v[1] - a[1] * k, v[2] - a[2] * k]
    };
    let (p, d) = (across(pointing.v), across(dir));
    libm::atan2f(dot(a, cross(p, d)), dot(p, d))
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(v: [f32; 3]) -> [f32; 3] {
    let len = libm::sqrtf(dot(v, v));
    if len > 0.0 { [v[0] / len, v[1] / len, v[2] / len] } else { v }
}

/// A model's id: [`abi::name_id`] of its name (`models/<name>.*` in the
/// content).
pub const fn model_id(name: &str) -> u64 {
    abi::name_id(name)
}

fn lookup(model: u64, prefix: &str, name: &str) -> Option<u32> {
    let v = unsafe { crate::host::ae_lookup(model, abi::name_id_from(abi::name_id(prefix), name)) };
    (v != abi::lookup::MISSING).then_some(v)
}

/// Clip `name` of model `model`, numbered from 1 as [`Anim::clip`].
pub fn clip(model: u64, name: &str) -> Option<u16> {
    lookup(model, abi::lookup::CLIP, name).map(|v| v as u16)
}

/// Bone `name` of model `model`.
pub fn bone(model: u64, name: &str) -> Option<u16> {
    lookup(model, abi::lookup::BONE, name).map(|v| v as u16)
}

/// Joint mask `name` of model `model`, numbered from 1 as [`Anim::mask`].
pub fn mask(model: u64, name: &str) -> Option<u16> {
    lookup(model, abi::lookup::MASK, name).map(|v| v as u16)
}
