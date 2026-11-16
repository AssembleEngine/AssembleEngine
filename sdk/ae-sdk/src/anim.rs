//! Animation channels (feature `anim`, M14): what a unit's model is doing,
//! as synced columns. A system (or a unit script) sets targets with
//! [`Anim::play`], [`Anim::turn`] and friends; the engine's `anim` service
//! (`host(anim)`, `ae_services::anim`) advances them every tick without
//! calling the module: clip phases (by time, or by the unit's ground speed
//! for no foot sliding), layer fades, tweens toward their targets. The pose
//! is a pure function of these columns (`ae_model::anim`), evaluated by the
//! client for drawing and by the engine for joint queries.
//!
//! Per unit: [`LAYERS`] clip layers, applied in index order (each a clip
//! with a play mode, a weight and a joint mask; blended over the layers
//! below, or added to them), then [`TWEENS`] procedural channels (turn, move
//! or spin a bone, or every bone of a mask). All zeros is nothing playing.
//!
//! Clips, bones and masks are the model's, found by name with [`clip`],
//! [`bone`] and [`mask`] (the service publishes them from the content's
//! models; look them up once and keep the numbers).

use crate::{Component, abi};

pub const LAYERS: usize = 4;
pub const TWEENS: usize = 8;
/// No bone: a tween over a mask ([`TweenSet::mask`]).
pub const NO_BONE: u16 = u16::MAX;

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
}

/// [`Anim::tween_kind`].
pub mod tween_kind {
    pub const NONE: u8 = 0;
    pub const TURN: u8 = 1;
    pub const MOVE: u8 = 2;
    pub const SPIN: u8 = 3;
}

/// [`Anim::tween_flags`].
pub mod tween_flags {
    /// On top of the layers' result, not over the rest pose.
    pub const ADDITIVE: u8 = 1;
}

/// A bone's local axis.
pub mod axis {
    pub const X: u8 = 0;
    pub const Y: u8 = 1;
    pub const Z: u8 = 2;
}

/// A unit's animation channels. Layer `i` is the `[i]` of the layer
/// arrays, tween `k` the `[k]` of the tween arrays (the engine's
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
    /// The bone, or [`NO_BONE`] with a mask.
    #[field(view = "los")]
    pub bone: [u16; TWEENS],
    #[field(view = "los")]
    pub tween_mask: [u16; TWEENS],
    /// [`tween_kind`].
    #[field(view = "los")]
    pub tween_kind: [u8; TWEENS],
    /// [`axis`].
    #[field(view = "los")]
    pub axis: [u8; TWEENS],
    /// [`tween_flags`].
    #[field(view = "los")]
    pub tween_flags: [u8; TWEENS],
    pub tween_reserved: [u8; TWEENS],
    /// Radians (turn, spin) or distance (move) now.
    #[field(view = "los")]
    pub value: [f32; TWEENS],
    #[field(view = "los")]
    pub tween_target: [f32; TWEENS],
    /// A second (0: at once).
    #[field(view = "los")]
    pub tween_speed: [f32; TWEENS],
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
            bone: [0; TWEENS],
            tween_mask: [0; TWEENS],
            tween_kind: [0; TWEENS],
            axis: [0; TWEENS],
            tween_flags: [0; TWEENS],
            tween_reserved: [0; TWEENS],
            value: [0.0; TWEENS],
            tween_target: [0.0; TWEENS],
            tween_speed: [0.0; TWEENS],
        }
    }

    /// Play `clip` on `layer` from its start, looping, at full weight over
    /// the whole model; adjust with the returned [`LayerSet`].
    pub fn play(&mut self, layer: usize, clip: u16) -> LayerSet<'_> {
        self.clip[layer] = clip;
        self.mask[layer] = 0;
        self.mode[layer] = mode::LOOP;
        self.kind[layer] = kind::BLEND;
        self.flags[layer] = 0;
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

    /// Turn `bone` about its local `axis` to `angle` radians at `speed`
    /// radians a second (the short way; 0: at once), over its rest pose.
    pub fn turn(&mut self, slot: usize, bone: u16, axis: u8, angle: f32, speed: f32) -> TweenSet<'_> {
        self.set_tween(slot, bone, tween_kind::TURN, axis, angle, speed)
    }

    /// Move `bone` along its local `axis` to `distance` from its rest
    /// position at `speed` a second (0: at once).
    pub fn move_to(&mut self, slot: usize, bone: u16, axis: u8, distance: f32, speed: f32) -> TweenSet<'_> {
        self.set_tween(slot, bone, tween_kind::MOVE, axis, distance, speed)
    }

    /// Spin `bone` about its local `axis` at `speed` radians a second.
    pub fn spin(&mut self, slot: usize, bone: u16, axis: u8, speed: f32) -> TweenSet<'_> {
        let value = if self.tween_kind[slot] == tween_kind::SPIN && self.bone[slot] == bone { self.value[slot] } else { 0.0 };
        let t = self.set_tween(slot, bone, tween_kind::SPIN, axis, 0.0, speed);
        t.anim.value[slot] = value;
        t
    }

    fn set_tween(&mut self, slot: usize, bone: u16, kind: u8, axis: u8, target: f32, speed: f32) -> TweenSet<'_> {
        // A tween that was already turning this bone this way goes on from
        // where it is.
        if self.tween_kind[slot] != kind || self.bone[slot] != bone || self.axis[slot] != axis {
            self.value[slot] = 0.0;
        }
        self.bone[slot] = bone;
        self.tween_mask[slot] = 0;
        self.tween_kind[slot] = kind;
        self.axis[slot] = axis;
        self.tween_flags[slot] = 0;
        self.tween_target[slot] = target;
        self.tween_speed[slot] = speed;
        TweenSet { anim: self, slot }
    }

    /// Clear tween `slot` (the bone goes back to what the layers give).
    pub fn stop_tween(&mut self, slot: usize) {
        self.bone[slot] = 0;
        self.tween_mask[slot] = 0;
        self.tween_kind[slot] = 0;
        self.axis[slot] = 0;
        self.tween_flags[slot] = 0;
        self.value[slot] = 0.0;
        self.tween_target[slot] = 0.0;
        self.tween_speed[slot] = 0.0;
    }

    /// Tween `slot` has reached its target (a spin never does).
    pub fn tween_done(&self, slot: usize) -> bool {
        matches!(self.tween_kind[slot], tween_kind::TURN | tween_kind::MOVE) && self.value[slot] == self.tween_target[slot]
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

    /// Added to the layers below (its change from the clip's first frame).
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

/// Settings of one tween ([`Anim::turn`]).
pub struct TweenSet<'a> {
    anim: &'a mut Anim,
    slot: usize,
}

impl TweenSet<'_> {
    /// On top of the layers' result instead of over the rest pose.
    pub fn additive(self) -> Self {
        self.anim.tween_flags[self.slot] |= tween_flags::ADDITIVE;
        self
    }

    /// Every bone of mask `mask` by its weight, instead of one bone.
    pub fn mask(self, mask: u16) -> Self {
        self.anim.tween_mask[self.slot] = mask;
        self.anim.bone[self.slot] = NO_BONE;
        self
    }

    /// Start from `value` instead of where the tween is.
    pub fn from(self, value: f32) -> Self {
        self.anim.value[self.slot] = value;
        self
    }
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
