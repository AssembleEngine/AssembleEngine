//! The animation service (M14): `anim` (every tick) advances every unit's
//! animation channels (`ae_sdk::anim::Anim`) with its model's rig
//! (`ae_model::anim::Rig::advance`): clip phases (by the unit's ground
//! speed where a layer asks), layer fades, tweens. No module code runs for
//! it: a unit that only plays clips and tweens costs no script call.
//!
//! The plugin holds the content's models by name; it publishes their clip,
//! bone and mask names for `ae_sdk::anim::{clip, bone, mask}`
//! (`ae_abi::lookup`). A unit whose model the plugin does not have still
//! advances, as a model without clips (phases stay at 0). Rows are
//! independent: the ticker splits them across the world's workers (its own
//! pool), with the same results for any split; a unit with nothing playing
//! and no manual entry is skipped (a tick would not change it).
//!
//! Queries (`ae_abi::query`), pure functions of the content and the
//! unit's `Anim`: `anim:dir_in` ([`Rig::dir_in`]) and `anim:bounds`
//! ([`Rig::bounds`]: the box of the mesh posed as the `Anim` has it, for
//! `ae_sdk::anim::Anim::update_bounds`).

use ae_abi::engine::{ANIM, BODY};
use ae_abi::{lookup, name_id, name_id_from, query};
use ae_ecs::Access;
use ae_model::anim::{Channels, LAYERS, Layer, MANUAL, Manual, Rig};
use ae_sdk::Component;
use ae_sdk::anim::Anim;
use ae_sdk::nav::Body;
use ae_sim::host::{HostCtx, HostDecl, HostInit, HostPlugin, HostSystem, expect_component};
use ae_sim::schema::Schema;
use anyhow::{Context as _, Result, bail, ensure};
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

    /// The ticker's work on rows, for benches and tools.
    pub fn advance(&self) -> Advance {
        Advance { rigs: self.rigs.clone(), none: Arc::new(Rig::new(&Default::default())), dt: self.tick_seconds }
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

    fn instantiate(&self, init: &HostInit, _schema: &Schema, names: &[&str]) -> Result<Vec<Box<dyn HostSystem>>> {
        // One source of truth: the plugin's tick length is the runtime's.
        let hz = 1.0 / self.tick_seconds;
        ensure!((hz - init.tick_hz as f32).abs() < 1e-3 * hz, "the animation service ticks at {hz} Hz, the sim at {} Hz", init.tick_hz);
        names
            .iter()
            .map(|&name| -> Result<Box<dyn HostSystem>> {
                match name {
                    "anim" => Ok(Box::new(Ticker { advance: self.advance(), par: ae_nav::par::Par::new(init.workers) })),
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

    fn queries(&self) -> Vec<u64> {
        vec![name_id(query::ANIM_DIR_IN), name_id(query::ANIM_BOUNDS), name_id(query::ANIM_BONE_WORLD)]
    }

    fn query(&self, op: u64, input: &[u8], out: &mut [u8]) -> Result<usize> {
        let n = size_of::<Anim>();
        if op == name_id(query::ANIM_BOUNDS) {
            ensure!(input.len() == n && out.len() >= 24, "bounds: {} bytes in, {} out", input.len(), out.len());
            let a: Anim = unsafe { input.as_ptr().cast::<Anim>().read_unaligned() };
            let b = self.rigs.get(&a.model).map_or_else(ae_model::Bounds::default, |(_, r)| r.bounds(&channels(&a)));
            for (k, x) in b.min.iter().chain(&b.max).enumerate() {
                out[4 * k..4 * k + 4].copy_from_slice(&x.to_le_bytes());
            }
            return Ok(24);
        }
        if op == name_id(query::ANIM_BONE_WORLD) {
            ensure!(input.len() == n + 32 && out.len() >= 28, "bone_world: {} bytes in, {} out", input.len(), out.len());
            let a: Anim = unsafe { input.as_ptr().cast::<Anim>().read_unaligned() };
            let word = |k: usize| u32::from_le_bytes(input[n + 4 * k..n + 4 * k + 4].try_into().unwrap());
            let f = |k: usize| f32::from_bits(word(k));
            let (bone, x, y, radius, facing) = (word(0), f(1), f(2), f(3), [f(4), f(5), f(6), f(7)]);
            let rig = self.rigs.get(&a.model).map(|(_, r)| r).with_context(|| format!("bone_world: no model {:#x}", a.model))?;
            ensure!((bone as usize) < rig.bones(), "bone_world: no bone {bone}");
            let (p, q) = rig.bone_frame(&channels(&a), bone as u16);
            let (pos, rot) = place(facing, [x, y, radius / rig.footprint()], p, q);
            for (k, v) in pos.iter().chain(&rot).enumerate() {
                out[4 * k..4 * k + 4].copy_from_slice(&v.to_le_bytes());
            }
            return Ok(28);
        }
        ensure!(op == name_id(query::ANIM_DIR_IN), "query {op:#x} is not the animation plugin's");
        ensure!(input.len() == n + 16 && out.len() >= 12, "dir_in: {} bytes in, {} out", input.len(), out.len());
        let a: Anim = unsafe { input.as_ptr().cast::<Anim>().read_unaligned() };
        let word = |k: usize| u32::from_le_bytes(input[n + 4 * k..n + 4 * k + 4].try_into().unwrap());
        let (bone, d) = (word(0), [1, 2, 3].map(|k| f32::from_bits(word(k))));
        let rig = self.rigs.get(&a.model).map(|(_, r)| r).with_context(|| format!("dir_in: no model {:#x}", a.model))?;
        ensure!((bone as usize) < rig.bones(), "dir_in: no bone {bone}");
        let v = rig.dir_in(&channels(&a), bone as u16, d);
        for (k, x) in v.iter().enumerate() {
            out[4 * k..4 * k + 4].copy_from_slice(&x.to_le_bytes());
        }
        Ok(12)
    }
}

struct Ticker {
    advance: Advance,
    /// The world's workers (rows split across them; serial at 1).
    par: ae_nav::par::Par,
}

impl HostSystem for Ticker {
    fn run(&mut self, ctx: &mut HostCtx<'_>) -> Result<()> {
        for chunk in ctx.chunks() {
            let anims = unsafe { chunk.column::<Anim>(0) };
            let bodies = unsafe { chunk.column::<Body>(1) };
            self.advance.rows_par(&self.par, anims, bodies);
        }
        Ok(())
    }
}

/// What the ticker does to rows: the content's rigs and the tick length.
#[derive(Clone)]
pub struct Advance {
    rigs: Arc<BTreeMap<u64, (String, Arc<Rig>)>>,
    none: Arc<Rig>,
    dt: f32,
}

impl Advance {
    /// One tick of every row: `anims[i]` with the ground speed of
    /// `bodies[i]`.
    pub fn rows(&self, anims: &mut [Anim], bodies: &[Body]) {
        for (a, b) in anims.iter_mut().zip(bodies.iter()) {
            // Nothing playing, fading or manual: a tick changes nothing.
            if a.clip == [0; LAYERS] && a.flags == [0; LAYERS] && a.manual_kind == [0; MANUAL] {
                continue;
            }
            let rig = self.rigs.get(&a.model).map_or(&self.none, |(_, r)| r);
            let mut ch = channels(a);
            rig.advance(&mut ch, self.dt, (b.vx * b.vx + b.vy * b.vy).sqrt());
            store(&ch, a);
        }
    }

    /// [`Advance::rows`] split across `par`'s threads (rows are
    /// independent: the same results for any split).
    pub fn rows_par(&self, par: &ae_nav::par::Par, anims: &mut [Anim], bodies: &[Body]) {
        par.chunks_mut_at(anims, |start, part| self.rows(part, &bodies[start..start + part.len()]));
    }
}

/// An `Anim` column value as the evaluator's channels.
pub fn channels(a: &Anim) -> Channels {
    Channels {
        layers: std::array::from_fn(|i| Layer {
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
        }),
        manual: std::array::from_fn(|k| Manual {
            bone: a.bone[k],
            kind: a.manual_kind[k],
            flags: a.manual_flags[k],
            value: a.value[k],
            target: a.goal[k],
            speed: a.top_speed[k],
            accel: a.accel[k],
            decel: a.decel[k],
            rate: a.rate[k],
        }),
    }
}

/// Channels back into an `Anim` (its model unchanged).
pub fn store(ch: &Channels, a: &mut Anim) {
    for (i, l) in ch.layers.iter().enumerate().take(LAYERS) {
        (a.clip[i], a.mask[i], a.mode[i], a.kind[i], a.flags[i]) = (l.clip, l.mask, l.mode, l.kind, l.flags);
        (a.weight[i], a.target[i], a.fade[i], a.speed[i], a.phase[i]) = (l.weight, l.target, l.fade, l.speed, l.phase);
    }
    for (k, m) in ch.manual.iter().enumerate().take(MANUAL) {
        (a.bone[k], a.manual_kind[k], a.manual_flags[k], a.value[k], a.goal[k]) = (m.bone, m.kind, m.flags, m.value, m.target);
        (a.top_speed[k], a.accel[k], a.decel[k], a.rate[k]) = (m.speed, m.accel, m.decel, m.rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_agree() {
        use ae_sdk::anim::{Axis, deg};
        assert_eq!((ae_sdk::anim::LAYERS, ae_sdk::anim::MANUAL), (LAYERS, MANUAL));
        assert_eq!(size_of::<Anim>(), 8 + size_of::<Channels>(), "the model id and the channels: 536 bytes");
        let mut a = Anim::new(7);
        a.play(1, 3).once().per_metre(0.25).mask(2).blend_in(100).at(0.5);
        a.turn(4, Axis::RIGHT, 1.0, 2.0).ramp(3.0);
        a.spin(9, Axis::bone([0.0, 1.0, 0.0]), 3.0);
        a.turn(4, Axis::UP, deg(90.0), 1.0);
        let ch = channels(&a);
        assert_eq!(ch.layers[1], Layer { clip: 3, mask: 2, mode: 2, kind: 0, flags: 5, reserved: 0, weight: 0.0, target: 1.0, fade: 10.0, speed: 0.25, phase: 0.5 });
        assert!(ch.layers[0].is_empty());
        let fresh = manual_flags::PENDING | manual_flags::FRESH;
        let pi = std::f32::consts::PI;
        assert_eq!(ch.manual[0], Manual { bone: 4, kind: 1, flags: fresh, target: [0.0, 1.0, 0.0, deg(90.0)], speed: 1.0, ..Default::default() }, "the second turn replaced the first");
        assert!((deg(90.0) - pi / 2.0).abs() < 1e-6);
        assert_eq!(ch.manual[1], Manual { bone: 9, kind: 2, flags: fresh | manual_flags::BONE_AXIS, target: [0.0, 1.0, 0.0, 0.0], speed: 3.0, ..Default::default() });
        assert!(ch.manual[2].is_empty());
        let mut back = Anim::new(7);
        store(&ch, &mut back);
        assert_eq!(back, a);
        // The SDK's constants are the evaluator's.
        use ae_model::anim::{kind, layer_flags, manual_flags, manual_kind};
        use ae_sdk::anim as sdk;
        assert_eq!((sdk::kind::ADDITIVE, sdk::layer_flags::GROUND_SPEED, sdk::layer_flags::ENDED, sdk::layer_flags::NEW), (kind::ADDITIVE, layer_flags::GROUND_SPEED, layer_flags::ENDED, layer_flags::NEW));
        assert_eq!((sdk::manual_kind::TURN, sdk::manual_kind::SPIN, sdk::manual_kind::MOVE, sdk::manual_kind::SCALE), (manual_kind::TURN, manual_kind::SPIN, manual_kind::MOVE, manual_kind::SCALE));
        assert_eq!((sdk::manual_flags::PENDING, sdk::manual_flags::FRESH, sdk::manual_flags::RELATIVE, sdk::manual_flags::BONE_AXIS), (manual_flags::PENDING, manual_flags::FRESH, manual_flags::RELATIVE, manual_flags::BONE_AXIS));
        assert_eq!(ae_sdk::anim::mode::ONCE as u32, ae_model::pose::PlayMode::Once as u32);
        assert_eq!(ae_sdk::anim::mode::PING_PONG as u32, ae_model::pose::PlayMode::PingPong as u32);
    }
}

/// A model-space point `p` and rotation `q` of a unit at ground point
/// `(x, y)` (nav), drawn at `scale` world units a model unit, facing
/// `facing`: in the world (nav x, height above the ground point, nav y).
/// Strict scalar f32, the same everywhere.
pub fn place(facing: [f32; 4], [x, y, scale]: [f32; 3], p: [f32; 3], q: [f32; 4]) -> ([f32; 3], [f32; 4]) {
    use ae_model::anim::{qmul, qrot};
    let v = qrot(facing, [p[0] * scale, p[1] * scale, p[2] * scale]);
    ([x + v[0], v[1], y + v[2]], qmul(facing, q))
}
