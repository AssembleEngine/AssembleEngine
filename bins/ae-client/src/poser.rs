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
//! model (`ClassModel::name`). Units are posed in parallel (rayon), each
//! into its own range of the keys.
//!
//! Not yet (step 3's second half, docs/PLAN.md "Joint keys"): uploading
//! only the bones that leave their line, stable key slots per unit,
//! skipping units whose channels did not change, spins by their rate.

use ae_model::anim::{Key, Pose, Rig, Trs, key, same_side};
use ae_render::units::NO_POSE;
use ae_sdk::anim::Anim;
use rayon::prelude::*;
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

/// One unit to pose in this stream.
struct Job<'a> {
    rig: &'a Rig,
    anim: &'a Anim,
    /// Its bones at the last tick in `Poser::locals`, if posed then.
    before: Option<u32>,
}

pub struct Poser {
    /// The rigs, by model id (`ae_sdk::anim::model_id`).
    rigs: BTreeMap<u64, Arc<Rig>>,
    /// Per stream class: the model id of the model it is drawn with.
    classes: Vec<Option<u64>>,
    /// Units posed at the last stream (by key) and their bones; the ones
    /// being posed now.
    last: Vec<Posed>,
    locals: Vec<Trs>,
    next: Vec<Posed>,
    next_locals: Vec<Trs>,
}

impl Poser {
    /// `rigs` by model id; `classes`: each stream class's model id.
    pub fn new(rigs: BTreeMap<u64, Arc<Rig>>, classes: Vec<Option<u64>>) -> Self {
        Self { rigs, classes, last: Vec::new(), locals: Vec::new(), next: Vec::new(), next_locals: Vec::new() }
    }

    /// Pose `units` (key, class, channels; ascending by key) at `tick`:
    /// each one's `pose` attribute into `poses` ([`NO_POSE`] when it is
    /// not posed), its keys into `keys` (both cleared first).
    pub fn pose(&mut self, tick: u64, units: &[(u64, u32, &Anim)], poses: &mut Vec<u32>, keys: &mut Vec<Key>) {
        let now = tick as u32;
        poses.clear();
        self.next.clear();
        let mut jobs = Vec::with_capacity(units.len());
        let (mut at, mut total) = (0, 0usize);
        for &(key_of, class, anim) in units {
            let rig = match self.classes.get(class as usize) {
                Some(&Some(model)) if anim.model == model => self.rigs.get(&model),
                _ => None,
            };
            let Some(rig) = rig else {
                poses.push(NO_POSE);
                continue;
            };
            let bones = rig.bones() as u32;
            while at < self.last.len() && self.last[at].key < key_of {
                at += 1;
            }
            let before = self.last.get(at).filter(|p| p.key == key_of && p.tick.wrapping_add(1) == now && p.bones == bones).map(|p| p.first);
            poses.push(2 * total as u32);
            self.next.push(Posed { key: key_of, tick: now, first: total as u32, bones });
            jobs.push(Job { rig, anim, before });
            total += bones as usize;
        }
        keys.clear();
        keys.resize(2 * total, [0; 12]);
        self.next_locals.clear();
        self.next_locals.resize(total, Trs { t: [0.0; 3], r: [0.0, 0.0, 0.0, 1.0], s: [1.0; 3] });
        // Each job's ranges of the keys and of the locals.
        let (mut key_rest, mut local_rest) = (&mut keys[..], &mut self.next_locals[..]);
        let mut work = Vec::with_capacity(jobs.len());
        for job in jobs {
            let bones = job.rig.bones();
            let (k, kr) = std::mem::take(&mut key_rest).split_at_mut(2 * bones);
            let (l, lr) = std::mem::take(&mut local_rest).split_at_mut(bones);
            (key_rest, local_rest) = (kr, lr);
            work.push((job, k, l));
        }
        let last = &self.locals;
        work.into_par_iter().with_min_len(256).for_each_init(BTreeMap::<usize, Pose>::new, |poses, (job, keys, locals)| {
            let pose = poses.entry(job.rig as *const Rig as usize).or_insert_with(|| job.rig.rest_pose());
            job.rig.pose(&ae_services::anim::channels(job.anim), pose);
            for (b, local) in pose.local.iter().enumerate() {
                let old = job.before.map_or(*local, |first| last[first as usize + b]);
                let new = same_side(&old, *local);
                keys[2 * b] = key(&old, now.wrapping_sub(1));
                keys[2 * b + 1] = key(&new, now);
                locals[b] = new;
            }
        });
        std::mem::swap(&mut self.last, &mut self.next);
        std::mem::swap(&mut self.locals, &mut self.next_locals);
    }
}
