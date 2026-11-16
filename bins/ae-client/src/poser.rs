//! Poses for the renderer (M14 step 3): each unit's `Anim` channels, as the
//! sim left them after a tick, evaluated on the CPU by the sim's own pose
//! function (`ae_model::anim::Rig::pose`) into writes to the joint key
//! table (`ae_render::units::KeyWrites`, kept by the `UnitFeed`).
//!
//! A posed unit has a slot in the table, the same while it stays posed:
//! two keys per bone (`stride` keys a slot, for the largest model). The
//! GPU draws each bone on the line through its two keys at the drawn time,
//! a tick behind as positions are. Each tick a bone's pose at the tick is
//! checked against its line ([`LINE_TOLERANCE`]); only a bone that left it
//! gets new keys: its pose at the last tick and at this one (two writes),
//! or only this one over the older key when the newer is the last tick's
//! (one write). A unit new to the table gets both keys at its pose (no
//! motion). A unit whose channels are the same as at its last pose is not
//! posed again (the pose is a function of the channels); its bones are
//! still checked against their lines. A unit is posed when its class is
//! drawn with a model and its `Anim` names that model
//! (`ClassModel::name`). Units are posed in parallel (rayon), by slot.
//!
//! Not yet (docs/PLAN.md "Joint keys"): spins by their rate.

use ae_model::anim::{Key, Pose, Rig, Tolerance, Trs, key, key_trs, on_line, same_side, within};
use ae_render::units::{KeyWrites, NO_POSE};
use ae_sdk::anim::Anim;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

/// A drawn bone may be this far from its pose before it gets new keys:
/// radians of rotation, scale, and translation as a fraction of the
/// model's reach (its farthest bone at rest).
pub const LINE_TOLERANCE: f32 = 1e-3;

struct Rigged {
    rig: Arc<Rig>,
    tolerance: Tolerance,
}

/// A unit posed at the last stream.
#[derive(Clone, Copy)]
struct Unit {
    key: u64,
    model: u64,
    slot: u32,
}

/// A slot's work in a stream.
#[derive(Clone, Copy)]
struct Job<'a> {
    rigged: &'a Rigged,
    anim: &'a Anim,
    /// New to its slot: both keys at its pose.
    fresh: bool,
}

pub struct Poser {
    /// The rigs, by model id (`ae_sdk::anim::model_id`).
    rigs: BTreeMap<u64, Rigged>,
    /// Per stream class: the model id of the model it is drawn with.
    classes: Vec<Option<u64>>,
    /// Bones of the largest rig; keys a slot (two per bone).
    max_bones: usize,
    stride: usize,
    /// Units posed at the last stream, ascending by key.
    units: Vec<Unit>,
    /// Slots in use or free; free ones.
    slots: usize,
    free: Vec<u32>,
    /// Per slot: its keys as written (`stride`), its bones' local
    /// transforms when last posed (`max_bones`), the channels and the
    /// tick then.
    table: Vec<Key>,
    locals: Vec<Trs>,
    anims: Vec<Anim>,
    ticks: Vec<u32>,
    /// Per slot: a bit per key written this stream.
    written: Vec<u64>,
    /// A stream was made (the first one resets the table).
    started: bool,
}

impl Poser {
    /// `rigs` by model id; `classes`: each stream class's model id.
    pub fn new(rigs: BTreeMap<u64, Arc<Rig>>, classes: Vec<Option<u64>>) -> Self {
        let max_bones = rigs.values().map(|r| r.bones()).max().unwrap_or(0).max(1);
        let rigs = rigs.into_iter().map(|(id, rig)| (id, Rigged { tolerance: tolerance(&rig), rig })).collect();
        Self {
            rigs,
            classes,
            max_bones,
            stride: 2 * max_bones,
            units: Vec::new(),
            slots: 0,
            free: Vec::new(),
            table: Vec::new(),
            locals: Vec::new(),
            anims: Vec::new(),
            ticks: Vec::new(),
            written: Vec::new(),
            started: false,
        }
    }

    /// Pose `units` (key, class, channels; ascending by key) at `tick`:
    /// each one's `pose` attribute into `poses` (cleared first;
    /// [`NO_POSE`] when it is not posed), the keys that changed into
    /// `writes` (replacing what it held; its memory is reused).
    pub fn pose(&mut self, tick: u64, units: &[(u64, u32, &Anim)], poses: &mut Vec<u32>, writes: &mut KeyWrites) {
        let now = tick as u32;
        writes.reset = !self.started;
        self.started = true;
        // Each unit's model, if posed.
        let models: Vec<Option<u64>> = units
            .iter()
            .map(|&(_, class, anim)| match self.classes.get(class as usize) {
                Some(&Some(model)) if anim.model == model && self.rigs.contains_key(&model) => Some(model),
                _ => None,
            })
            .collect();
        // Units that stay keep their slots; the others' slots are freed
        // before new units take slots.
        let mut kept = vec![None; units.len()];
        let mut at = 0;
        for unit in std::mem::take(&mut self.units) {
            while at < units.len() && units[at].0 < unit.key {
                at += 1;
            }
            match units.get(at) {
                Some(u) if u.0 == unit.key && models[at] == Some(unit.model) => kept[at] = Some(unit.slot),
                _ => self.free.push(unit.slot),
            }
        }
        let mut wanted = Vec::new();
        poses.clear();
        for (i, &(key_of, _, anim)) in units.iter().enumerate() {
            let Some(model) = models[i] else {
                poses.push(NO_POSE);
                continue;
            };
            let (slot, fresh) = match kept[i] {
                Some(slot) => (slot, false),
                None => (self.take_slot(), true),
            };
            wanted.push((slot, model, anim, fresh));
            self.units.push(Unit { key: key_of, model, slot });
            poses.push((slot as usize * self.stride) as u32);
        }
        let mut jobs: Vec<Option<Job>> = vec![None; self.slots];
        for (slot, model, anim, fresh) in wanted {
            jobs[slot as usize] = Some(Job { rigged: &self.rigs[&model], anim, fresh });
        }
        let stride = self.stride;
        let words = stride.div_ceil(64);
        let counts: Vec<u32> = self
            .table
            .par_chunks_mut(stride)
            .zip(self.locals.par_chunks_mut(self.max_bones))
            .zip(self.anims.par_iter_mut())
            .zip(self.ticks.par_iter_mut())
            .zip(self.written.par_chunks_mut(words))
            .zip(jobs.par_iter())
            .with_min_len(64)
            .map_init(BTreeMap::<usize, Pose>::new, |scratch, (((((keys, locals), last_anim), last_tick), written), job)| {
                written.fill(0);
                if let Some(job) = job {
                    pose_slot(job, now, keys, locals, last_anim, last_tick, scratch, |k| written[k / 64] |= 1 << (k % 64));
                }
                written.iter().map(|w| w.count_ones()).sum()
            })
            .collect();
        // The written keys in slot order, each slot's from its offset
        // (filled in parallel; `writes` keeps its memory from stream to
        // stream, see `UnitFeed::publish_reusing`).
        let n = counts.iter().map(|&c| c as usize).sum();
        // Every entry below `n` is written: only new length is cleared.
        writes.at.truncate(n);
        writes.at.resize(n, 0);
        writes.keys.truncate(n);
        writes.keys.resize(n, [0; 12]);
        let (mut at_rest, mut keys_rest) = (&mut writes.at[..], &mut writes.keys[..]);
        let mut outs = Vec::with_capacity(counts.len());
        for &c in &counts {
            let (a, ar) = std::mem::take(&mut at_rest).split_at_mut(c as usize);
            let (k, kr) = std::mem::take(&mut keys_rest).split_at_mut(c as usize);
            (at_rest, keys_rest) = (ar, kr);
            outs.push((a, k));
        }
        let table = &self.table;
        outs.into_par_iter().zip(self.written.par_chunks(words)).enumerate().with_min_len(64).for_each(|(slot, ((at, keys), written))| {
            let mut i = 0;
            for (w, &bits) in written.iter().enumerate() {
                let mut bits = bits;
                while bits != 0 {
                    let k = slot * stride + w * 64 + bits.trailing_zeros() as usize;
                    (at[i], keys[i]) = (k as u32, table[k]);
                    i += 1;
                    bits &= bits - 1;
                }
            }
        });
    }

    /// A free slot, or a new one.
    fn take_slot(&mut self) -> u32 {
        if let Some(slot) = self.free.pop() {
            return slot;
        }
        let rest = Trs { t: [0.0; 3], r: [0.0, 0.0, 0.0, 1.0], s: [1.0; 3] };
        self.slots += 1;
        self.table.resize(self.slots * self.stride, [0; 12]);
        self.locals.resize(self.slots * self.max_bones, rest);
        self.anims.resize(self.slots, Anim::new(0));
        self.ticks.resize(self.slots, 0);
        self.written.resize(self.slots * self.stride.div_ceil(64), 0);
        (self.slots - 1) as u32
    }
}

/// One unit's tick: its pose (unless its channels are as last time), each
/// bone checked against its keys' line, new keys into `keys` and their
/// places in the slot through `write`.
#[allow(clippy::too_many_arguments)]
fn pose_slot(job: &Job, now: u32, keys: &mut [Key], locals: &mut [Trs], last_anim: &mut Anim, last_tick: &mut u32, scratch: &mut BTreeMap<usize, Pose>, mut write: impl FnMut(usize)) {
    let rig = &job.rigged.rig;
    let then = now.wrapping_sub(1);
    // `locals` is the pose at the last tick.
    let behind = !job.fresh && last_tick.wrapping_add(1) == now;
    let posed = match job.fresh || *last_anim != *job.anim {
        true => {
            let pose = scratch.entry(Arc::as_ptr(rig) as usize).or_insert_with(|| rig.rest_pose());
            rig.pose(&ae_services::anim::channels(job.anim), pose);
            Some(&pose.local)
        }
        false => None,
    };
    for b in 0..rig.bones() {
        let p = posed.map_or(locals[b], |l| l[b]);
        let (older, newer) = (2 * b, 2 * b + 1);
        if job.fresh {
            for (k, t) in [(older, then), (newer, now)] {
                keys[k] = key(&p, t);
                write(k);
            }
        } else if !within(&on_line(&keys[older], &keys[newer], now, 0.0), &p, &job.rigged.tolerance) {
            // Which of the two is the newer key (ticks wrap).
            let (o, n) = match keys[newer][3].wrapping_sub(keys[older][3]) as i32 >= 0 {
                true => (older, newer),
                false => (newer, older),
            };
            if keys[n][3] == then {
                // The newer key is the last tick's pose: the new line
                // starts there.
                keys[o] = key(&same_side(&key_trs(&keys[n]).0, p), now);
                write(o);
            } else {
                let from = if behind { locals[b] } else { p };
                keys[older] = key(&from, then);
                keys[newer] = key(&same_side(&from, p), now);
                write(older);
                write(newer);
            }
        }
        locals[b] = p;
    }
    *last_anim = *job.anim;
    *last_tick = now;
}

/// [`LINE_TOLERANCE`] for `rig`.
fn tolerance(rig: &Rig) -> Tolerance {
    let mut globals = vec![[0.0; 12]; rig.bones()];
    rig.globals(&rig.rest_pose(), &mut globals);
    let reach = globals.iter().map(|g| (g[3] * g[3] + g[7] * g[7] + g[11] * g[11]).sqrt()).fold(0.0f32, f32::max);
    Tolerance::new(LINE_TOLERANCE * reach.max(1e-3), LINE_TOLERANCE, LINE_TOLERANCE)
}
