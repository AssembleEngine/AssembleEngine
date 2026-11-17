//! The animation ticker (`ae_services::anim`, the `anim` host system's
//! row work) on seeded channels, without the rest of a sim tick: 10 000
//! and 50 000 units of a synthetic 27-bone model with three clips and an
//! upper-body mask. A unit in ten is idle (nothing playing); the others
//! play a looping clip, half of them a second one by ground speed fading
//! in and out on the mask, a fifth an additive clip; a third have a manual
//! joint turning, and each tick one unit in a hundred is given a new turn
//! (a pending entry, which the ticker resolves against the pose). Timed:
//! one tick at 1 thread and at `--threads`. Counter `digest`: the channels
//! after [`DIGEST_TICKS`] ticks (48 bits of FNV-1a over the rows), the same
//! at both thread counts or the suite fails; a changed digest means the
//! ticker's results changed.

use super::Options;
use crate::measure::time;
use crate::record::Report;
use ae_model::anim::Rig;
use ae_model::{Bone, Channel, Clip, Interpolation, Model, Path, Range};
use ae_nav::par::Par;
use ae_sdk::anim::{Anim, Axis, model_id};
use ae_sdk::nav::Body;
use ae_services::anim::{Advance, AnimPlugin};
use anyhow::{Result, bail};

const BONES: usize = 27;
const DIGEST_TICKS: u64 = 90;

/// Bone `b`'s parent is `(b - 1) / 2`; every bone has a rotation track in
/// every clip, the root a translation one too.
fn model() -> Model {
    let quat = |axis: usize, a: f32| {
        let mut q = [0.0, 0.0, 0.0, (a / 2.0).cos()];
        q[axis] = (a / 2.0).sin();
        q
    };
    let cols = |t: [f32; 3]| {
        let mut m = [0.0f32; 16];
        (m[0], m[5], m[10], m[15]) = (1.0, 1.0, 1.0, 1.0);
        (m[12], m[13], m[14]) = (t[0], t[1], t[2]);
        m
    };
    let bones = (0..BONES)
        .map(|b| {
            let t = if b == 0 { [0.0; 3] } else { [0.1 * (b % 3) as f32, 0.3, 0.05 * (b % 5) as f32] };
            Bone { name: format!("b{b}"), parent: (b > 0).then(|| (b as u32 - 1) / 2), rest: cols(t), translation: t, rotation: [0.0, 0.0, 0.0, 1.0], scale: [1.0; 3] }
        })
        .collect();
    let mut data = Vec::new();
    let mut clips = Vec::new();
    for (c, duration) in [1.0f32, 0.8, 1.5].into_iter().enumerate() {
        let keys = 9;
        let times: Vec<f32> = (0..keys).map(|k| duration * k as f32 / (keys - 1) as f32).collect();
        let mut channels = Vec::new();
        let push = |data: &mut Vec<f32>, values: Vec<f32>| {
            let t = Range { offset: data.len() as u64, len: times.len() as u64 };
            data.extend(&times);
            let v = Range { offset: data.len() as u64, len: values.len() as u64 };
            data.extend(values);
            (t, v)
        };
        for b in 0..BONES {
            let values = (0..keys).flat_map(|k| quat((b + c) % 3, 0.4 * ((k + b) as f32 * 0.7).sin())).collect();
            let (t, v) = push(&mut data, values);
            channels.push(Channel { bone: b as u32, path: Path::Rotation, interpolation: Interpolation::Linear, times: t, values: v });
        }
        let values = (0..keys).flat_map(|k| [0.0, 0.05 * (k as f32).sin(), 0.0]).collect();
        let (t, v) = push(&mut data, values);
        channels.push(Channel { bone: 0, path: Path::Translation, interpolation: Interpolation::Linear, times: t, values: v });
        clips.push(Clip { name: format!("clip{c}"), duration, channels });
    }
    Model { bones, clips, clip_data: data, ..Default::default() }
}

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// `n` units' channels and bodies.
fn seed(n: usize, model: u64, upper: u16) -> (Vec<Anim>, Vec<Body>) {
    let mut f = rng(0xA417_0005);
    let (mut anims, mut bodies) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let mut a = Anim::new(model);
        let speed = if i % 2 == 0 { 1.0 + f() * 3.0 } else { 0.0 };
        if i % 10 != 0 {
            a.play(0, 1 + (i % 3) as u16).at(f() * 0.8);
            if i % 2 == 0 {
                a.play(1, 2).per_metre(0.3).mask(upper).blend_in(300);
            }
            if i % 5 == 1 {
                a.play(2, 3).additive().weight(0.5);
            }
            if i % 3 == 0 {
                a.turn(5 + (i % 7) as u16, Axis::UP, f() * 3.0 - 1.5, 2.0).ramp(6.0);
            }
        }
        anims.push(a);
        bodies.push(Body { x: 0.0, y: 0.0, vx: speed, vy: 0.0, radius: 1.0 });
    }
    (anims, bodies)
}

/// What a game script would do at `tick`: a new turn for one unit in a
/// hundred (pending until the ticker takes it).
fn commands(anims: &mut [Anim], tick: u64) {
    for (i, a) in anims.iter_mut().enumerate() {
        if (i as u64 + tick) % 100 == 0 && i % 10 != 0 {
            let angle = ((i as u64 * 31 + tick) % 300) as f32 / 100.0 - 1.5;
            a.turn(5 + (i % 7) as u16, Axis::UP, angle, 2.0).ramp(6.0);
        }
    }
}

fn tick(advance: &Advance, par: &Par, anims: &mut [Anim], bodies: &[Body], t: u64) {
    commands(anims, t);
    advance.rows_par(par, anims, bodies);
}

fn digest(anims: &[Anim]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for a in anims {
        // SAFETY: plain data, `repr(C)` with no padding.
        let bytes = unsafe { std::slice::from_raw_parts((a as *const Anim).cast::<u8>(), size_of::<Anim>()) };
        for &b in bytes {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h & ((1 << 48) - 1)
}

pub fn run(options: &Options) -> Result<Report> {
    let mut report = Report::new("anim");
    let mut rig = Rig::new(&model());
    let upper = rig.add_mask("upper", rig.subtree(1));
    let plugin = AnimPlugin::new([("synthetic".to_string(), rig)], 1.0 / 30.0);
    let advance = plugin.advance();
    let model = model_id("synthetic");
    let threads = options.threads.max(1);
    let (serial, parallel) = (Par::new(1), Par::new(threads));
    for n in [10_000, 50_000] {
        if options.small && n > 10_000 {
            continue;
        }
        let tag = format!("units{n}");
        let run_digest = |par: &Par| {
            let (mut anims, bodies) = seed(n, model, upper);
            for t in 0..DIGEST_TICKS {
                tick(&advance, par, &mut anims, &bodies, t);
            }
            digest(&anims)
        };
        let d = run_digest(&serial);
        if run_digest(&parallel) != d {
            bail!("{tag}: the channels differ at 1 and {threads} threads");
        }
        report.counter(&format!("{tag}/digest"), d, "");
        for (par, label) in [(&serial, "1t".to_string()), (&parallel, format!("{threads}t"))] {
            let (mut anims, bodies) = seed(n, model, upper);
            let mut t = 0;
            report.time(
                &format!("{tag}/tick_{label}"),
                time(options.budget, || {
                    tick(&advance, par, &mut anims, &bodies, t);
                    t += 1;
                }),
            );
        }
    }
    report.info("bones", BONES as u64, "");
    Ok(report)
}
