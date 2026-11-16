//! Poses for the renderer (M14 step 3): each unit's `Anim` channels, as the
//! sim left them after a tick, evaluated on the CPU by the sim's own pose
//! function (`ae_model::anim::Rig::pose`) into joint keys for the GPU's
//! pose pass (`ae_render::units::InstanceStream::keys`).
//!
//! A posed unit gets two keys per bone of its model: its local transform
//! at the previous tick and at this one (the same twice when it was not
//! posed at the previous tick). The GPU draws each bone on the line
//! through them at the drawn time, a tick behind as positions are. A unit
//! is posed when its class is drawn with a model and its `Anim` names that
//! model (`ClassModel::name`).
//!
//! Not yet (step 3's second half, docs/PLAN.md "Joint keys"): uploading
//! only the bones that leave their line, stable key slots per unit,
//! skipping units whose channels did not change, spins by their rate.

use ae_model::anim::{Key, Pose, Rig, Trs, key, same_side};
use ae_render::units::NO_POSE;
use ae_sdk::anim::Anim;
use std::collections::BTreeMap;
use std::sync::Arc;

/// One unit as posed at a tick: its key, the tick, its bones in
/// `Poser::locals`.
#[derive(Clone, Copy)]
struct Posed {
    key: u64,
    tick: u32,
    first: u32,
    bones: u32,
}

pub struct Poser {
    /// The rigs, by model id (`ae_sdk::anim::model_id`).
    rigs: BTreeMap<u64, Arc<Rig>>,
    /// Per stream class: the model id of the model it is drawn with.
    classes: Vec<Option<u64>>,
    /// Scratch pose per model id.
    poses: BTreeMap<u64, Pose>,
    /// Units posed at the last stream (by key) and their bones; the ones
    /// being posed now.
    last: Vec<Posed>,
    locals: Vec<Trs>,
    next: Vec<Posed>,
    next_locals: Vec<Trs>,
    /// Merge position in `last`.
    at: usize,
}

impl Poser {
    /// `rigs` by model id; `classes`: each stream class's model id.
    pub fn new(rigs: BTreeMap<u64, Arc<Rig>>, classes: Vec<Option<u64>>) -> Self {
        Self { rigs, classes, poses: BTreeMap::new(), last: Vec::new(), locals: Vec::new(), next: Vec::new(), next_locals: Vec::new(), at: 0 }
    }

    /// Start a stream.
    pub fn begin(&mut self) {
        self.next.clear();
        self.next_locals.clear();
        self.at = 0;
    }

    /// The `pose` attribute of the unit `key` of class `class` with
    /// channels `anim` at `tick` (keys ascending within a stream), its keys
    /// appended to `keys`; [`NO_POSE`] when it is not posed.
    pub fn row(&mut self, tick: u64, key_of: u64, class: u32, anim: &Anim, keys: &mut Vec<Key>) -> u32 {
        let Some(&Some(model)) = self.classes.get(class as usize) else { return NO_POSE };
        if anim.model != model {
            return NO_POSE;
        }
        let Some(rig) = self.rigs.get(&model) else { return NO_POSE };
        let pose = self.poses.entry(model).or_insert_with(|| rig.rest_pose());
        rig.pose(&ae_services::anim::channels(anim), pose);
        let (now, bones) = (tick as u32, rig.bones() as u32);
        while self.at < self.last.len() && self.last[self.at].key < key_of {
            self.at += 1;
        }
        let before = self.last.get(self.at).filter(|p| p.key == key_of && p.tick.wrapping_add(1) == now && p.bones == bones);
        let first = keys.len() as u32;
        let start = self.next_locals.len() as u32;
        for (b, local) in pose.local.iter().enumerate() {
            let old = before.map_or(*local, |p| self.locals[p.first as usize + b]);
            let new = same_side(&old, *local);
            keys.extend([key(&old, now.wrapping_sub(1)), key(&new, now)]);
            self.next_locals.push(new);
        }
        self.next.push(Posed { key: key_of, tick: now, first: start, bones });
        first
    }

    /// End a stream: its poses are the next one's previous ones.
    pub fn end(&mut self) {
        std::mem::swap(&mut self.last, &mut self.next);
        std::mem::swap(&mut self.locals, &mut self.next_locals);
    }
}
