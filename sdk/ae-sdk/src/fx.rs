//! Effects at joints (feature `fx`, M14): an [`Effect`] event emitted where
//! a bone is, with its direction and a seed, for the view to turn into
//! particles. The sim knows no renderer indices: `effect` is the
//! [`name_id`](crate::abi::name_id) of the effect's name.
//!
//! Opt-in, as `aim` (every event a module declares takes an entity id in
//! it), and it pulls `aim` in: an effect is placed by the unit's
//! [`Orientation`](crate::aim::Orientation) and [`Anim::bone_world`].

use crate::Entity;
use crate::abi;
use crate::aim::{BoneWorld, Orientation};
use crate::anim::{Anim, Axis};
use crate::nav::Body;
use crate::{Component, Event};

/// A particle effect at a bone, in the world: the event a game's system
/// emits (`emits(Effect)`) at the tick it happens. The view starts it when
/// it draws that tick.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0014, event)]
pub struct Effect {
    /// The unit the effect belongs to (a flash's light rides it).
    pub entity: Entity,
    /// The tick it was emitted at.
    pub tick: u64,
    /// World nav x, the height above the unit's ground point, nav y
    /// (`Anim::bone_world`'s frame).
    pub pos: [f32; 3],
    /// The bone's world forward (a unit vector); zero for none.
    pub dir: [f32; 3],
    /// [`name_id`](crate::abi::name_id) of the effect's name (`"flash"`,
    /// `"fire"`, `"smoke"`, …).
    pub effect: u64,
    pub count: u32,
    /// Multiplies the effect's sizes, speeds and spread.
    pub scale: f32,
    /// Varies the particles between bursts: hashed from the entity, tick,
    /// effect and bone.
    pub seed: u32,
    /// The bone the effect is at.
    pub bone: u16,
    /// Reserved for directional effects; zero.
    pub flags: u16,
}

const _: () = assert!(<Effect as Event>::ID == abi::engine::EFFECT);

/// The seed of an effect: a deterministic hash of `(entity, tick, effect,
/// bone)`, so two effects of one kind from different bones in one tick
/// differ (splitmix64's finaliser over the four).
pub fn seed(entity: Entity, tick: u64, effect: u64, bone: u16) -> u32 {
    let mix = |mut h: u64| {
        h ^= h >> 30;
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^ (h >> 31)
    };
    (mix(mix(mix(mix(entity.0) ^ tick) ^ effect) ^ bone as u64) >> 32) as u32
}

impl Anim {
    /// Emit an [`Effect`] at `bone` for the unit `entity` at `tick`: its
    /// origin, plus `offset` in the unit-frame axes as the bone carries
    /// them (`BoneWorld::dir`, so `0.3 * Axis::FORWARD.v` puts it at a
    /// muzzle ahead of the bone). `effect` is the name's
    /// [`name_id`](crate::abi::name_id); the emitting system declares
    /// `emits(Effect)`. A script's poll uses [`Unit::effect_at`]
    /// ([`crate::script::Unit`]).
    #[allow(clippy::too_many_arguments)]
    pub fn effect_at(
        &self,
        entity: Entity,
        tick: u64,
        bone: u16,
        offset: [f32; 3],
        body: &Body,
        orientation: &Orientation,
        effect: u64,
        count: u32,
        scale: f32,
    ) {
        let at = self.bone_world(bone, body, orientation);
        let o = at.point(offset);
        let dir = at.dir(Axis::FORWARD);
        crate::emit(&Effect {
            entity,
            tick,
            pos: [at.pos[0] + o[0], at.pos[1] + o[1], at.pos[2] + o[2]],
            dir,
            effect,
            count,
            scale,
            seed: seed(entity, tick, effect, bone),
            bone,
            flags: 0,
        });
    }
}

#[cfg(feature = "script")]
impl crate::script::Unit {
    /// Emit an [`Effect`] at `bone` of the polled unit: its entity and
    /// tick, and its `Body` and `Orientation` (terms of the resuming
    /// system). See [`Anim::effect_at`].
    pub fn effect_at(&self, bone: u16, offset: [f32; 3], effect: u64, count: u32, scale: f32) {
        let anim = self.get::<Anim>();
        anim.effect_at(
            self.entity(),
            self.tick(),
            bone,
            offset,
            &self.get::<Body>(),
            &self.get::<Orientation>(),
            effect,
            count,
            scale,
        );
    }
}

impl BoneWorld {
    /// `offset` in the unit-frame axes, as the bone turns them (see
    /// [`Anim::effect_at`]).
    pub(crate) fn point(&self, offset: [f32; 3]) -> [f32; 3] {
        crate::aim::qrot(self.rot, offset)
    }
}
