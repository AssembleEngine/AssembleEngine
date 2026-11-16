//! The animation service (M14): `anim` (every tick) advances every unit's
//! animation channels (`ae_sdk::anim::Anim`) with its model's rig
//! (`ae_model::anim::Rig::advance`): clip phases (by the unit's ground
//! speed where a layer asks), layer fades, tweens. No module code runs for
//! it: a unit that only plays clips and tweens costs no script call.
//!
//! The plugin holds the content's models by name; it publishes their clip,
//! bone and mask names for `ae_sdk::anim::{clip, bone, mask}`
//! (`ae_abi::lookup`). A unit whose model the plugin does not have still
//! advances, as a model without clips (phases stay at 0).

use ae_abi::engine::{ANIM, BODY};
use ae_abi::{lookup, name_id, name_id_from};
use ae_ecs::Access;
use ae_model::anim::{Channels, LAYERS, Layer, Rig, TWEENS, Tween};
use ae_sdk::Component;
use ae_sdk::anim::Anim;
use ae_sdk::nav::Body;
use ae_sim::host::{HostCtx, HostDecl, HostInit, HostPlugin, HostSystem, expect_component};
use ae_sim::schema::Schema;
use anyhow::{Context as _, Result, bail};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct AnimPlugin {
    /// By model id (`ae_sdk::anim::model_id` of the name).
    rigs: Arc<BTreeMap<u64, (String, Arc<Rig>)>>,
    tick_seconds: f32,
}

impl AnimPlugin {
    /// The content's models by name, and the sim's tick length.
    pub fn new(models: impl IntoIterator<Item = (String, Rig)>, tick_seconds: f32) -> Self {
        let rigs = models.into_iter().map(|(name, rig)| (name_id(&name), (name, Arc::new(rig)))).collect();
        Self { rigs: Arc::new(rigs), tick_seconds }
    }

    /// The content's models: every `models/*.aemodel` (`models/fox.aemodel`
    /// is model `fox`), or `None` without any.
    pub fn from_content(vfs: &ae_vfs::Vfs, tick_seconds: f32) -> Result<Option<Self>> {
        let files = vfs.list("models", "aemodel")?;
        if files.is_empty() {
            return Ok(None);
        }
        let mut models = Vec::new();
        for f in files {
            let model = ae_model::Model::from_bytes(&vfs.read(&f)?).with_context(|| format!("model {f}"))?;
            let name = f.trim_start_matches("models/").trim_end_matches(".aemodel").to_string();
            models.push((name, Rig::new(&model)));
        }
        Ok(Some(Self::new(models, tick_seconds)))
    }

    pub fn rig(&self, model: u64) -> Option<&Arc<Rig>> {
        self.rigs.get(&model).map(|(_, r)| r)
    }
}

impl HostPlugin for AnimPlugin {
    fn systems(&self) -> Vec<HostDecl> {
        let mut anim = HostDecl::new("anim");
        anim.query = vec![(ANIM, Access::Write), (BODY, Access::Read)];
        vec![anim]
    }

    fn check(&self, _name: &str, schema: &Schema) -> Result<()> {
        expect_component(schema, Anim::SCHEMA)?;
        expect_component(schema, Body::SCHEMA)?;
        Ok(())
    }

    fn instantiate(&self, _init: &HostInit, _schema: &Schema, names: &[&str]) -> Result<Vec<Box<dyn HostSystem>>> {
        names
            .iter()
            .map(|&name| -> Result<Box<dyn HostSystem>> {
                match name {
                    "anim" => Ok(Box::new(Ticker { rigs: self.rigs.clone(), none: Rig::new(&Default::default()), dt: self.tick_seconds })),
                    other => bail!("no host system `{other}` in the animation plugin"),
                }
            })
            .collect()
    }

    fn names(&self) -> Vec<(u64, u64, u32)> {
        let mut out = Vec::new();
        for (&model, (_, rig)) in self.rigs.iter() {
            let key = |prefix: &str, name: &str| name_id_from(name_id(prefix), name);
            out.extend(rig.clip_names().iter().enumerate().map(|(k, n)| (model, key(lookup::CLIP, n), k as u32 + 1)));
            out.extend(rig.bone_names().iter().enumerate().map(|(k, n)| (model, key(lookup::BONE, n), k as u32)));
            out.extend(rig.masks().iter().enumerate().map(|(k, m)| (model, key(lookup::MASK, &m.name), k as u32 + 1)));
        }
        out
    }
}

struct Ticker {
    rigs: Arc<BTreeMap<u64, (String, Arc<Rig>)>>,
    none: Rig,
    dt: f32,
}

impl HostSystem for Ticker {
    fn run(&mut self, ctx: &mut HostCtx<'_>) -> Result<()> {
        for chunk in ctx.chunks() {
            let anims = unsafe { chunk.column::<Anim>(0) };
            let bodies = unsafe { chunk.column::<Body>(1) };
            for (a, b) in anims.iter_mut().zip(bodies.iter()) {
                let rig = self.rigs.get(&a.model).map_or(&self.none, |(_, r)| r);
                let mut ch = channels(a);
                rig.advance(&mut ch, self.dt, (b.vx * b.vx + b.vy * b.vy).sqrt());
                store(&ch, a);
            }
        }
        Ok(())
    }
}

/// An `Anim` column value as the evaluator's channels.
pub fn channels(a: &Anim) -> Channels {
    let mut ch = Channels::default();
    for (i, l) in ch.layers.iter_mut().enumerate() {
        *l = Layer {
            clip: a.clip[i],
            mask: a.mask[i],
            mode: a.mode[i],
            kind: a.kind[i],
            flags: a.flags[i],
            reserved: 0,
            weight: a.weight[i],
            target: a.target[i],
            fade: a.fade[i],
            speed: a.speed[i],
            phase: a.phase[i],
        };
    }
    for (k, t) in ch.tweens.iter_mut().enumerate() {
        *t = Tween {
            bone: a.bone[k],
            mask: a.tween_mask[k],
            kind: a.tween_kind[k],
            axis: a.axis[k],
            flags: a.tween_flags[k],
            reserved: 0,
            value: a.value[k],
            target: a.tween_target[k],
            speed: a.tween_speed[k],
        };
    }
    ch
}

/// Channels back into an `Anim` (its model unchanged).
pub fn store(ch: &Channels, a: &mut Anim) {
    for (i, l) in ch.layers.iter().enumerate().take(LAYERS) {
        (a.clip[i], a.mask[i], a.mode[i], a.kind[i], a.flags[i]) = (l.clip, l.mask, l.mode, l.kind, l.flags);
        (a.weight[i], a.target[i], a.fade[i], a.speed[i], a.phase[i]) = (l.weight, l.target, l.fade, l.speed, l.phase);
    }
    for (k, t) in ch.tweens.iter().enumerate().take(TWEENS) {
        (a.bone[k], a.tween_mask[k], a.tween_kind[k], a.axis[k], a.tween_flags[k]) = (t.bone, t.mask, t.kind, t.axis, t.flags);
        (a.value[k], a.tween_target[k], a.tween_speed[k]) = (t.value, t.target, t.speed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_agree() {
        assert_eq!((ae_sdk::anim::LAYERS, ae_sdk::anim::TWEENS), (LAYERS, TWEENS));
        assert_eq!(ae_sdk::anim::NO_BONE, ae_model::anim::NONE);
        let mut a = Anim::new(7);
        a.play(1, 3).once().per_metre(0.25).mask(2).blend_in(100).at(0.5);
        a.turn(5, 4, ae_sdk::anim::axis::Z, 1.0, 2.0).additive();
        a.spin(6, 9, ae_sdk::anim::axis::Y, 3.0).mask(1);
        let ch = channels(&a);
        assert_eq!(ch.layers[1], Layer { clip: 3, mask: 2, mode: 2, kind: 0, flags: 1, reserved: 0, weight: 0.0, target: 1.0, fade: 10.0, speed: 0.25, phase: 0.5 });
        assert!(ch.layers[0].is_empty() && ch.tweens[0].is_empty());
        assert_eq!(ch.tweens[5], Tween { bone: 4, mask: 0, kind: 1, axis: 2, flags: 1, reserved: 0, value: 0.0, target: 1.0, speed: 2.0 });
        assert_eq!(ch.tweens[6].bone, ae_model::anim::NONE);
        let mut back = Anim::new(7);
        store(&ch, &mut back);
        assert_eq!(back, a);
        // The SDK's constants are the evaluator's.
        use ae_model::anim::{kind, layer_flags, tween_flags, tween_kind};
        assert_eq!((ae_sdk::anim::kind::ADDITIVE, ae_sdk::anim::layer_flags::GROUND_SPEED, ae_sdk::anim::layer_flags::ENDED), (kind::ADDITIVE, layer_flags::GROUND_SPEED, layer_flags::ENDED));
        assert_eq!((ae_sdk::anim::tween_kind::TURN, ae_sdk::anim::tween_kind::MOVE, ae_sdk::anim::tween_kind::SPIN, ae_sdk::anim::tween_flags::ADDITIVE), (tween_kind::TURN, tween_kind::MOVE, tween_kind::SPIN, tween_flags::ADDITIVE));
        assert_eq!(ae_sdk::anim::mode::ONCE as u32, ae_model::pose::PlayMode::Once as u32);
        assert_eq!(ae_sdk::anim::mode::PING_PONG as u32, ae_model::pose::PlayMode::PingPong as u32);
    }
}
