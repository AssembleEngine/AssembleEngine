//! The full tier's surface dynamics: the prototype's wave particles
//! (`~/RedAlert-graphics`: `src/scene.cpp` `updateScene`, `emitBomb`;
//! `shaders/wave_particle.glsl`) in a camera-centred patch.
//!
//! The CPU appends spawn records -- position, velocity, orientation, birth,
//! life, amplitude, radius, churn, whitewater -- and keeps them until they
//! die; every record's state is a closed-form function of the record and
//! the time, evaluated on the GPU. Floating hulls in the observer's stream
//! drop records at the stern and bow every tick: the side pairs drift
//! outward at the Kelvin half-angle so the trail opens into a V, and a
//! mound on the centre line leaves the churned trail. An impact is three
//! ring waves, a churned boil and a splash stamp (the crater's flip-book).
//!
//! The GPU copies the records (`water.emit`), stamps them additively into
//! `water.raw` (wave bumps) and `water.churn` (churn, its age, whitewater),
//! and filters the raw field with the swell into `water.waves`
//! (`water_blur.comp`).
use super::detail::DetailTextures;
use super::{WaterFeed, WaterParams};
use crate::bindless::{Bindless, GroupId, PUSH_STAGES, Push};
use crate::device::GpuDevice;
use crate::executor::{BufferBinding, ImageBinding, PassContext};
use crate::memory::OwnedBuffer;
use crate::pipeline::{self, Blend, GraphicsDesc, slice_bytes, spirv};
use crate::terrain::app::TerrainSource;
use crate::units::{
    ATTR_CLASS, ATTR_PREV_X, ATTR_PREV_Y, ATTR_RADIUS, ATTR_VIEW, ATTR_WATER, ATTR_X, ATTR_Y, InstanceStream,
    UnitsSetup, VIEW_SEEN,
};
use ae_framegraph::{BufferDesc, ImageDesc};
use anyhow::{Context, Result};
use ash::vk;
use glam::{Vec2, Vec3};
use gpu_allocator::MemoryLocation;
use std::collections::VecDeque;
use std::f32::consts::TAU;

pub(crate) const SIZE: u32 = 1024;
pub(crate) const FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// The live records' cap: at most about 2,000 per hull at steady state
/// (the wake's length is fixed, so a faster hull's records live shorter),
/// 537 per blast. The oldest go first when it is reached.
pub(crate) const MAX_SOURCES: usize = 65_536;
const SOURCE_WORDS: usize = 20;
/// The blur's output tiles (`water_blur.comp` TILE).
const BLUR_TILE: u32 = 32;

/// The prototype's simulation texel (600 m over 2048 texels): its
/// amplitudes and blur gains are tuned for it, and the patch matches it at
/// its closest.
const PROTOTYPE_TEXEL: f32 = 600.0 / 2048.0;
/// Metres per raw unit (`water_blur.comp` HEIGHT_SCALE).
const HEIGHT_SCALE: f32 = 8.0;
const G: f32 = 9.81;
/// Wake particles: amplitude per 0.05 s of emission, life; the churned
/// centre trail's life.
const WAKE_AMPLITUDE: f32 = 0.16;
const WAKE_INTERVAL: f32 = 0.05;
const WAKE_LIFE: f32 = 10.0;
/// The prototype's boats' speed (m/s), which its per-emission amplitude is
/// tuned for; a wake record's furthest spacing along the path (m) and the
/// most records per hull and tick.
const WAKE_REFERENCE_SPEED: f32 = 6.0;
const WAKE_SPACING: f32 = 1.0;
const WAKE_MAX_STEPS: f32 = 6.0;
const TRAIL_LIFE: f32 = 24.0;
/// How far behind the hull its wake reaches, in hull radii: the
/// prototype's boats (about 3 m, 6 m/s) keep their arms 10 s and their
/// trail 24 s, some 20 and 50 radii. A wake is a length on the water, not a
/// time: an RTS hull doing 100 m/s with the prototype's lives would leave a
/// lane kilometres long. The lives are these lengths over the speed, never
/// longer than the prototype's.
const ARM_RADII: f32 = 20.0;
const TRAIL_RADII: f32 = 50.0;
/// The trail's churn per record, below the prototype's 0.05: its records
/// overlap a dozen deep, and at 0.05 the sum stays saturated until the
/// trail's end, which then reads as a uniform band with a cut end. At this
/// value it saturates only behind the hull and thins along the trail.
const TRAIL_CHURN: f32 = 0.015;
/// A blast's ring waves and churned boil, at the default 5 m crater.
const BLAST_RING_LIFE: f32 = 30.0;
const BLAST_BOIL_LIFE: f32 = 30.0;
const BLAST_RADIUS_DEFAULT: f32 = 5.0;

pub(crate) fn patch_desc(filtered: bool) -> ImageDesc {
    let write = if filtered {
        vk::ImageUsageFlags::STORAGE
    } else {
        vk::ImageUsageFlags::COLOR_ATTACHMENT
    };
    ImageDesc::color_2d(
        FORMAT,
        SIZE,
        SIZE,
        write | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
    )
}

pub(crate) fn particle_desc() -> BufferDesc {
    BufferDesc {
        size: (MAX_SOURCES * SOURCE_WORDS * 4) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
    }
}

#[derive(Clone, Copy)]
pub(crate) struct WavePatch {
    pub origin: Vec2,
    pub span: f32,
}

impl WavePatch {
    /// The patch around the camera's target: 1.5 times the camera distance
    /// across, from the prototype's resolution (300 m, 0.29 m per texel) up
    /// to 1.6 km.
    pub fn new(target: Vec2, camera: Vec3, distance: f32) -> Self {
        let span = (distance * 1.5).clamp(SIZE as f32 * PROTOTYPE_TEXEL, 1600.0);
        let ahead = target - Vec2::new(camera.x, camera.z);
        let center = target - ahead * 0.2;
        let texel = span / SIZE as f32;
        Self {
            origin: (center / texel).floor() * texel - span * 0.5,
            span,
        }
    }

    pub fn push(self, push: &mut Push) {
        push.words2[..3].copy_from_slice(&[self.origin.x.to_bits(), self.origin.y.to_bits(), self.span.to_bits()]);
    }

    /// Whether `p` is within `margin` (a share of the span) of the patch.
    fn reaches(self, p: Vec2, margin: f32) -> bool {
        let uv = (p - self.origin) / self.span;
        uv.cmpge(Vec2::splat(-margin)).all() && uv.cmple(Vec2::splat(1.0 + margin)).all()
    }
}

/// One spawn record (`water_particle.glsl` WaveSource).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct WaveParticle {
    pub position: Vec2,
    pub velocity: Vec2,
    pub direction: Vec2,
    pub birth: f32,
    pub life: f32,
    pub amplitude: f32,
    pub radius: f32,
    pub churn: f32,
    pub foam: f32,
    pub grow: f32,
    pub arc: f32,
    pub decay: f32,
    pub spread: f32,
    /// 0 a wave particle, 1 a blast's splash stamp.
    pub kind: f32,
    /// The splash's time scale T (s).
    pub scale: f32,
}

impl WaveParticle {
    fn wave(position: Vec2, velocity: Vec2, direction: Vec2, birth: f32, amplitude: f32, radius: f32) -> Self {
        Self {
            position,
            velocity,
            direction,
            birth,
            life: WAKE_LIFE,
            amplitude,
            radius,
            decay: 4.0,
            spread: 1.0,
            ..Self::default()
        }
    }

    fn words(&self) -> [f32; SOURCE_WORDS] {
        [
            self.position.x,
            self.position.y,
            self.velocity.x,
            self.velocity.y,
            self.direction.x,
            self.direction.y,
            self.birth,
            self.life,
            self.amplitude,
            self.radius,
            self.churn,
            self.foam,
            self.grow,
            self.arc,
            self.decay,
            self.spread,
            self.kind,
            self.scale,
            0.0,
            0.0,
        ]
    }
}

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

fn hash01(i: u32, salt: u32) -> f32 {
    (hash(i.wrapping_mul(2_654_435_761) ^ salt.wrapping_mul(2_246_822_519).wrapping_add(1)) & 0xffff) as f32 / 65535.0
}

/// Peak of a Gaussian bump (standard deviation `sigma`, metres) after one
/// pass of RA3's blur (21 taps of a raised cosine, normalised by 21) at the
/// prototype's texel: a flat field keeps 10/21 of its height per pass.
fn blur_gain(sigma: f32) -> f32 {
    (-10..=10)
        .map(|i| {
            let x = i as f32 * PROTOTYPE_TEXEL;
            (-x * x / (2.0 * sigma * sigma)).exp() * 0.5 * ((std::f32::consts::PI * i as f32 / 10.0).cos() + 1.0)
        })
        .sum::<f32>()
        / 21.0
}

/// A ship's wake records for the path it covered in one tick (the
/// prototype's `emitBoat`). The prototype drops its records every 0.05 s from
/// boats doing about 6 m/s, some 0.3 m apart; an RTS hull covers metres per
/// tick, so the records are laid along the path at most [`WAKE_SPACING`]
/// apart (closer than the blur's reach, so the trail stays continuous) with
/// the prototype's amplitude per metre of path. `from` and `to` are the
/// drawn hull's positions at the tick's start and end, `radius` its size;
/// the records are born over the tick, ending at `birth`.
pub(crate) fn wake(from: Vec2, to: Vec2, radius: f32, birth: f32, dt: f32, step: u32, out: &mut Vec<WaveParticle>) {
    let path = to - from;
    let length = path.length();
    if length < 0.05 * dt.max(1e-3) {
        return;
    }
    let fwd = path / length;
    let speed = length / dt.max(1e-3);
    let right = Vec2::new(fwd.y, -fwd.x);
    let steps = (length / WAKE_SPACING).ceil().clamp(1.0, WAKE_MAX_STEPS) as u32;
    let spacing = length / steps as f32;
    // Amplitude and churn per metre of path, as the prototype lays them.
    let per_metre = spacing / (WAKE_REFERENCE_SPEED * WAKE_INTERVAL);
    // The particle footprint is ~1 texel (sigma); the blur gives the wave
    // its shape.
    let footprint = 3.0 * PROTOTYPE_TEXEL;
    // The arms drift out at the Kelvin half-angle (tan 19.47 deg = 0.354).
    let side = 0.354 * speed;
    let arm_life = (ARM_RADII * radius / speed).min(WAKE_LIFE);
    let trail_life = (TRAIL_RADII * radius / speed).min(TRAIL_LIFE);
    for k in 0..steps {
        let along = (k + 1) as f32 / steps as f32;
        let at = from + path * along;
        let born = birth - dt * (1.0 - along);
        let sub = step.wrapping_mul(WAKE_MAX_STEPS as u32).wrapping_add(k);
        // Small emission-to-emission variation keeps the wake from
        // accumulating into a perfectly periodic pattern.
        let jitter = 0.9 + 0.2 * hash01(sub, 11);
        let stern = at - fwd * (0.88 * radius);
        let bow = at + fwd * (1.32 * radius);
        let amp = WAKE_AMPLITUDE * per_metre;
        let mut add = |p: Vec2, vel: Vec2, amplitude: f32, r: f32, churn: f32, life: f32| {
            out.push(WaveParticle {
                churn: churn * per_metre,
                life,
                ..WaveParticle::wave(p, vel, fwd, born, amplitude, r)
            });
        };
        // The stern pair; churn strongest on the centre line (the
        // propeller wash), weaker on the arms.
        add(
            stern + right * (0.22 * radius),
            right * side * jitter,
            amp,
            footprint * 1.5,
            0.012,
            arm_life,
        );
        add(
            stern - right * (0.22 * radius),
            -right * side * jitter,
            amp,
            footprint * 1.5,
            0.012,
            arm_life,
        );
        // The bow wave, pushed out faster than the stern pair.
        add(
            bow + right * (0.16 * radius),
            right * (side * 1.6 * jitter),
            amp * 0.7,
            footprint * 1.5,
            0.012,
            arm_life,
        );
        add(
            bow - right * (0.16 * radius),
            -right * (side * 1.6 * jitter),
            amp * 0.7,
            footprint * 1.5,
            0.012,
            arm_life,
        );
        // The propeller wash on the centre line, about as wide as the beam:
        // a turbulent trail, not a ribbon. Each record drifts a little to
        // one side, starts a different size, spreads at its own rate and
        // lives its own share of the trail, so the trail wanders, widens
        // behind the hull and frays out at its end.
        let drift = (hash01(sub, 23) * 2.0 - 1.0) * 0.18 * radius / trail_life.max(0.5);
        let size = 0.26 * radius * (0.7 + 0.6 * hash01(sub, 29));
        out.push(WaveParticle {
            churn: TRAIL_CHURN * per_metre,
            life: trail_life * (0.55 + 0.45 * hash01(sub, 31)),
            spread: 1.0 + 2.0 * hash01(sub, 37),
            ..WaveParticle::wave(stern, right * drift, fwd, born, amp * 0.1, size)
        });
    }
}

/// A detonation at `at` with crater radius `radius` (the prototype's
/// `emitBomb`, scaled by Froude: lengths as R, times and speeds as
/// sqrt(R / g)): the splash stamp (cavity, collapse, jet), three ring waves
/// (the rim, the collapse, the jet's fall) and the churned boil, whose
/// whitewater comes as the thrown water falls back. The spray in the air
/// is the client's particles.
pub(crate) fn blast(at: Vec2, radius: f32, birth: f32, seed: u32, out: &mut Vec<WaveParticle>) {
    let r = radius.clamp(0.15, 10.0);
    let t = (r / G).sqrt();
    let k = r / BLAST_RADIUS_DEFAULT;
    let phase = hash01(seed, 7) * TAU;
    let flat_gain = (10.0f32 / 21.0).powi(2);
    // The crater is half as deep as it is wide; the stamp's half size is
    // twice its radius.
    out.push(WaveParticle {
        position: at,
        birth,
        life: 6.0 * t,
        amplitude: 0.5 * r / HEIGHT_SCALE / flat_gain,
        radius: 2.0 * r,
        decay: 1.0,
        kind: 1.0,
        scale: t,
        ..WaveParticle::default()
    });
    struct Ring {
        count: u32,
        start: f32,
        r0: f32,
        b: f32,
        height: f32,
        foam: f32,
        churn: f32,
    }
    let rings = [
        Ring {
            count: 200,
            start: 0.4 * t,
            r0: 1.1 * r,
            b: 1.3 * r,
            height: 0.25 * r,
            foam: 0.5,
            churn: 0.3,
        },
        Ring {
            count: 140,
            start: 1.8 * t,
            r0: 0.8 * r,
            b: 0.8 * r,
            height: 0.15 * r,
            foam: 0.3,
            churn: 0.2,
        },
        Ring {
            count: 100,
            start: 4.6 * t,
            r0: 0.6 * r,
            b: 0.5 * r,
            height: 0.08 * r,
            foam: 0.0,
            churn: 0.1,
        },
    ];
    let around = |a: f32, m: f32| 0.5 + 0.3 * (3.0 * a + phase * m).sin() + 0.2 * (7.0 * a + phase * (m + 1.3)).sin();
    let ring_life = BLAST_RING_LIFE * k.sqrt().clamp(0.5, 1.3);
    let ring_count = if r < 1.0 { 0 } else { rings.len() };
    for (n, ring) in rings.iter().enumerate().take(ring_count) {
        let speed = (G * 2.8 * ring.b / TAU).sqrt();
        // The summed crest of N bumps on the circle, per unit particle
        // amplitude, through both blur passes (along the ring it is flat).
        let overlap = ring.count as f32 * ring.b * 0.98 / (TAU * ring.r0);
        let gain = 10.0 / 21.0 * blur_gain(0.395 * ring.b.max(4.0));
        let amp = ring.height / HEIGHT_SCALE / gain / overlap;
        let nf = n as f32;
        for i in 0..ring.count {
            let a = TAU * (i as f32 + 0.5) / ring.count as f32;
            let h = around(a, 1.0 + 1.1 * nf) - 0.5;
            let j = hash01(i, 3 + 4 * n as u32) - 0.5;
            let rr = ring.r0 * (1.0 + 0.035 * (a * (5.0 - nf) + phase * (2.0 - 0.3 * nf)).sin());
            let d = Vec2::new(a.cos(), a.sin());
            let tangent = Vec2::new(-d.y, d.x);
            let vel = d * speed * (1.0 + 0.1 * h + 0.08 * j) + tangent * speed * (0.14 + 0.12 * j);
            out.push(WaveParticle {
                position: at + d * rr,
                velocity: vel,
                direction: d,
                birth: birth + ring.start,
                life: ring_life - ring.start,
                amplitude: amp,
                radius: ring.b,
                churn: ring.churn / overlap,
                foam: ring.foam / overlap,
                grow: vel.length() / rr,
                arc: 1.4 * TAU / ring.count as f32,
                decay: 15.0,
                spread: k.min(1.0),
                ..WaveParticle::default()
            });
        }
    }
    // The boil: no wave, only churn and whitewater, spreading and swirling
    // as one patch. It comes as the thrown water falls back, around the rim
    // from ~1 T, then inward.
    let count = if r < 1.0 { 12 } else { 96 };
    let v_scale = k.sqrt();
    for i in 0..count {
        let (h1, h2) = (hash01(i, 17), hash01(i, 29));
        let rr = 1.6 * r * h2.sqrt();
        let a = TAU * h1 + phase;
        let d = Vec2::new(a.cos(), a.sin());
        let vel = (d * (0.4 + 0.15 * rr / k) + Vec2::new(-d.y, d.x) * (0.3 + 0.6 * around(a, 3.3)) * (rr / (1.6 * r)))
            * v_scale;
        out.push(WaveParticle {
            position: at + d * rr,
            velocity: vel,
            direction: d,
            birth: birth + t * (1.0 + 2.0 * (1.0 - h2.sqrt())),
            life: BLAST_BOIL_LIFE * k.sqrt().clamp(0.2, 1.4),
            amplitude: 0.0,
            radius: ((0.8 + 0.4 * h1) * r).max(0.5),
            churn: 0.35 * 32.0 / count as f32,
            foam: 0.3 * 32.0 / count as f32,
            decay: 4.0,
            spread: k.min(1.0),
            ..WaveParticle::default()
        });
    }
}

#[derive(Default)]
struct WaveHistory {
    live: VecDeque<WaveParticle>,
    generation: u64,
    last_tick: Option<u64>,
    scratch: Vec<WaveParticle>,
}

impl WaveHistory {
    fn update(
        &mut self,
        patch: WavePatch,
        time: f32,
        terrain: &TerrainSource,
        feed: Option<&WaterFeed>,
        units: Option<(&UnitsSetup, &InstanceStream)>,
    ) {
        self.scratch.clear();
        if let Some(feed) = feed {
            let (generation, impacts) = feed.drain();
            if self.generation != generation {
                self.generation = generation;
                self.live.clear();
                // Wait for a new stream so the replaced observer's last
                // frame cannot immediately reconstruct its old wakes.
                self.last_tick = units.map(|(_, stream)| stream.tick);
            }
            for impact in impacts {
                blast(
                    Vec2::from_array(impact.position),
                    impact.radius,
                    time,
                    impact.seed,
                    &mut self.scratch,
                );
            }
        }
        if let Some((setup, stream)) = units
            && self.last_tick != Some(stream.tick)
            && let Some(level) = terrain.info.water_level
        {
            let first = self.last_tick.is_none();
            self.last_tick = Some(stream.tick);
            let dt = stream.sim_seconds.max(0.001);
            let c = &stream.columns;
            let f = |attr: usize, row: usize| f32::from_bits(c[attr][row]);
            for row in 0..stream.len().min(setup.capacity as usize) {
                if first || c[ATTR_VIEW][row] > VIEW_SEEN || f(ATTR_WATER, row) > 0.0 {
                    continue;
                }
                let class = (c[ATTR_CLASS][row] as usize).min(crate::units::MAX_CLASSES - 1);
                if !setup
                    .classes
                    .get(class)
                    .or_else(|| setup.classes.last())
                    .is_some_and(|c| c.floats)
                {
                    continue;
                }
                // The drawn hull is at the previous tick's position when
                // this stream arrives (the renderer interpolates towards
                // the new one), so the wake starts behind it.
                let now = Vec2::new(f(ATTR_X, row), f(ATTR_Y, row));
                let drawn = Vec2::new(f(ATTR_PREV_X, row), f(ATTR_PREV_Y, row));
                let radius = f(ATTR_RADIUS, row).clamp(1.0, 35.0);
                // Hulls just outside the patch emit too: the camera moves
                // the patch, and a wake that was never laid would show as a
                // gap in the trail when it comes into view.
                if !patch.reaches(drawn, 0.5) || terrain.height_at(drawn) >= level {
                    continue;
                }
                let step = (stream.tick as u32) ^ (row as u32).wrapping_mul(0x9e37_79b9);
                wake(drawn - (now - drawn), drawn, radius, time, dt, step, &mut self.scratch);
            }
        }
        self.live.retain(|p| time - p.birth < p.life);
        self.live.extend(self.scratch.drain(..));
        let overflow = self.live.len().saturating_sub(MAX_SOURCES);
        self.live.drain(..overflow);
    }
}

pub(crate) struct Waves {
    history: WaveHistory,
    input: Vec<OwnedBuffer>,
    input_index: u32,
    detail: DetailTextures,
    group: GroupId,
    pipelines: [vk::Pipeline; 4],
    layout: vk::PipelineLayout,
    push: Push,
    count: u32,
    /// The water clock the records were written against.
    time: f32,
    /// The swell this frame's blur used (the surface reads the same).
    swell: Swell,
}

/// What the full tier's water draw binds besides the terrain's own data.
#[derive(Clone, Copy)]
pub(crate) struct WaveSurface {
    pub patch: WavePatch,
    /// Sampled indices: the filtered waves, the churn, the shore field.
    pub waves: u32,
    pub churn: u32,
    pub shore: u32,
    /// [`Waves::detail`].
    pub detail: [u32; 3],
    /// [`Waves::swell_words`].
    pub swell: [u32; 4],
}

impl Waves {
    /// `terrain` is the terrain's push block: its linear clamp and repeat
    /// samplers (1, 2) and its height texture (image 1).
    pub fn new(gpu: &GpuDevice, bindless: &mut Bindless, slots: usize, terrain: &Push) -> Result<Self> {
        let mut input = Vec::with_capacity(slots);
        let mut detail = None;
        let upload = (|| {
            for _ in 0..slots {
                input.push(OwnedBuffer::new(
                    gpu,
                    particle_desc(),
                    MemoryLocation::CpuToGpu,
                    "water particle upload",
                )?);
            }
            bindless.group(|b| {
                let index = b.per_slot_buffer(&input)?.index;
                detail = Some(DetailTextures::new(gpu, b)?);
                Ok(index)
            })
        })();
        let fail = |input: &mut Vec<OwnedBuffer>, detail: &mut Option<DetailTextures>| {
            for buffer in input {
                buffer.destroy(gpu);
            }
            if let Some(detail) = detail {
                detail.destroy(gpu);
            }
        };
        let (input_index, group) = match upload {
            Ok(handles) => handles,
            Err(error) => {
                fail(&mut input, &mut detail);
                return Err(error);
            }
        };
        let device = &gpu.device;
        let layout = bindless.pipeline_layout();
        let compute = |shader| pipeline::compute_pipeline(device, shader, layout);
        let draw = |fragment| {
            pipeline::graphics_pipeline(
                device,
                &GraphicsDesc {
                    vertex: spirv!("water_splat.vert"),
                    fragment,
                    layout,
                    color_formats: &[FORMAT],
                    depth: None,
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    blend: Blend::Additive,
                    glow_blend: Blend::Off,
                },
            )
        };
        let built = super::reflect::pipelines(device, |i| match i {
            0 => compute(spirv!("water_emit.comp")),
            1 => draw(spirv!("water_splat.frag")),
            2 => draw(spirv!("water_churn.frag")),
            _ => compute(spirv!("water_blur.comp")),
        });
        let pipelines = match built {
            Ok(pipelines) => pipelines,
            Err(err) => {
                bindless.retire_group(group);
                fail(&mut input, &mut detail);
                return Err(err);
            }
        };
        let detail = detail.expect("made with the upload");
        let mut push = Push::default();
        push.sampler(1, terrain.sampler_at(1));
        push.sampler(2, terrain.sampler_at(2));
        push.image(7, detail.static_map);
        push.image(8, terrain.images[0][1]);
        Ok(Self {
            history: WaveHistory::default(),
            input,
            input_index,
            detail,
            group,
            pipelines,
            layout,
            push,
            count: 0,
            time: 0.0,
            swell: Swell::of(&WaterParams::default()),
        })
    }

    /// The detail textures' sampled indices: the static map, the ripple
    /// loop, the foam.
    pub fn detail(&self) -> [u32; 3] {
        [self.detail.static_map, self.detail.ripples, self.detail.foam]
    }

    /// The water draw's push words for the swell (`water.frag`).
    pub fn swell_words(&self) -> [u32; 4] {
        self.swell.words()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        slot: usize,
        patch: WavePatch,
        time: f32,
        swell_time: f32,
        params: WaterParams,
        terrain: &TerrainSource,
        feed: Option<&WaterFeed>,
        setup: Option<&UnitsSetup>,
    ) {
        let stream = setup.and_then(|s| s.feed.latest());
        self.history
            .update(patch, time, terrain, feed, setup.zip(stream.as_deref()));
        let target = self.input[slot].mapped().expect("wave input upload");
        self.count = self.history.live.len() as u32;
        for (dst, src) in target
            .as_chunks_mut::<{ SOURCE_WORDS * 4 }>()
            .0
            .iter_mut()
            .zip(&self.history.live)
        {
            dst.copy_from_slice(slice_bytes(&src.words()));
        }
        patch.push(&mut self.push);
        let info = &terrain.info;
        let swell = Swell::of(&params);
        self.swell = swell;
        for (at, value) in [
            (2, swell.wind[0]),
            (3, swell.wind[1]),
            (4, swell.scale),
            (5, swell.strength),
            (6, swell_time),
            (9, info.height_scale),
            (10, info.height_offset),
            (11, info.cell_size),
            (12, 1.0 / (info.cells.0 + 1) as f32),
            (13, 1.0 / (info.cells.1 + 1) as f32),
            (14, info.water_level.unwrap_or(0.0)),
        ] {
            self.push.image(at, f32::to_bits(value));
        }
        self.time = time;
    }

    pub fn record_emit(&self, ctx: &PassContext<'_>, output: BufferBinding) -> Result<()> {
        let mut push = self.push;
        push.words = [
            self.input_index,
            output.storage.context("wave particle storage")?,
            self.count,
            0,
        ];
        self.bind(ctx, vk::PipelineBindPoint::COMPUTE, self.pipelines[0], &push);
        if self.count > 0 {
            unsafe { ctx.device.cmd_dispatch(ctx.cmd, self.count.div_ceil(64), 1, 1) };
        }
        Ok(())
    }

    pub fn record_splat(&self, ctx: &PassContext<'_>, input: BufferBinding, churn: bool) -> Result<()> {
        let mut push = self.push;
        push.words[1] = input.storage.context("wave particle storage")?;
        push.words[2] = self.time.to_bits();
        push.words[3] = u32::from(churn);
        self.bind(
            ctx,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipelines[1 + usize::from(churn)],
            &push,
        );
        unsafe { ctx.device.cmd_draw(ctx.cmd, 6, self.count, 0, 0) };
        Ok(())
    }

    pub fn record_blur(
        &self,
        ctx: &PassContext<'_>,
        raw: ImageBinding,
        _churn: ImageBinding,
        output: ImageBinding,
    ) -> Result<()> {
        let mut push = self.push;
        push.image(0, raw.sampled.context("raw wave sample")?);
        push.words[0] = output.storage.context("filtered wave storage")?;
        self.bind(ctx, vk::PipelineBindPoint::COMPUTE, self.pipelines[3], &push);
        unsafe { ctx.device.cmd_dispatch(ctx.cmd, SIZE / BLUR_TILE, SIZE / BLUR_TILE, 1) };
        Ok(())
    }

    fn bind(&self, ctx: &PassContext<'_>, point: vk::PipelineBindPoint, pipeline: vk::Pipeline, push: &Push) {
        unsafe {
            ctx.device.cmd_bind_pipeline(ctx.cmd, point, pipeline);
            ctx.device
                .cmd_bind_descriptor_sets(ctx.cmd, point, self.layout, 0, &[ctx.global_set], &[]);
            ctx.device
                .cmd_push_constants(ctx.cmd, self.layout, PUSH_STAGES, 0, pipeline::push_bytes(push));
        }
    }

    pub fn retire(&self, bindless: &mut Bindless) {
        bindless.retire_group(self.group);
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        for pipeline in self.pipelines {
            unsafe { gpu.device.destroy_pipeline(pipeline, None) };
        }
        for buffer in &mut self.input {
            buffer.destroy(gpu);
        }
        self.detail.destroy(gpu);
    }
}

/// The swell's parameters as `water_swell.glsl` takes them, from a map's
/// look: the first train runs with the wind, the frequencies are scaled
/// from the prototype's 105 m to the map's swell length, the heights from
/// its default sea (0.4 m) to the map's amplitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Swell {
    pub wind: [f32; 2],
    pub scale: f32,
    pub strength: f32,
}

impl Swell {
    pub fn of(params: &WaterParams) -> Self {
        let wind = Vec2::from_array(params.wind).try_normalize().unwrap_or(Vec2::X);
        Self {
            wind: wind.to_array(),
            scale: (TAU / 0.060 / params.swell_length.max(1.0)).clamp(0.5, 2.5),
            strength: (params.swell_amplitude / 0.4).clamp(0.0, 4.0),
        }
    }

    /// The water draw's push words (`water.frag`).
    pub fn words(&self) -> [u32; 4] {
        [self.wind[0], self.wind[1], self.scale, self.strength].map(f32::to_bits)
    }

    /// `water_swell.glsl`'s `swellHeight` in deep water: the height above
    /// the rest level at `p` (x, z) and time `t`.
    pub fn height(&self, p: Vec2, t: f32) -> f32 {
        const DIR: [[f32; 2]; 3] = [
            [0.274_721_13, -0.961_523_95],
            [-0.819_231_9, -0.573_462_3],
            [-0.980_580_7, 0.196_116_14],
        ];
        const FREQ: [f32; 3] = [0.060, 0.125, 0.085];
        const AMP: [f32; 3] = [0.55, 0.22, 0.16];
        const SPEED: [f32; 3] = [0.55, 0.90, 0.68];
        let wind = Vec2::from_array(self.wind);
        let a = Vec2::from_array(DIR[0]);
        let (c, sn) = (a.dot(wind), a.x * wind.y - a.y * wind.x);
        (0..3)
            .map(|i| {
                let d = Vec2::from_array(DIR[i]);
                let dir = Vec2::new(c * d.x - sn * d.y, sn * d.x + c * d.y);
                let phase = dir.dot(p) * FREQ[i] * self.scale + t * SPEED[i] * self.scale.sqrt();
                let h = 0.5 + 0.5 * phase.sin();
                AMP[i] * h * h
            })
            .sum::<f32>()
            * self.strength
    }
}

/// The full tier's swell height above the rest level at `p` (x, z) and
/// water clock `t`, in deep water, for a map's look (what a floating hull
/// rides, besides the patch's static displacement, wakes and rings).
pub fn swell_height(params: &WaterParams, p: [f32; 2], t: f32) -> f32 {
    Swell::of(params).height(Vec2::from_array(p), t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_moving_hull_drops_kelvin_arms_and_a_trail() {
        let mut out = Vec::new();
        // 5 m/s for a tick: 1/6 m of path, one emission.
        wake(
            Vec2::ZERO,
            Vec2::new(5.0 / 30.0, 0.0),
            5.0,
            1.0,
            1.0 / 30.0,
            0,
            &mut out,
        );
        assert_eq!(out.len(), 5);
        // The arms drift out sideways at the Kelvin half-angle, the mound
        // stays and lives longest.
        let arms: Vec<_> = out.iter().filter(|p| p.velocity.y.abs() > 1.0).collect();
        assert_eq!(arms.len(), 4);
        for arm in &arms {
            assert!(arm.velocity.x.abs() < 1e-6, "arms drift across the course");
            assert!(arm.velocity.y.abs() >= 0.354 * 5.0 * 0.9 - 1e-4);
        }
        assert!(
            out.iter()
                .any(|p| p.velocity.y.abs() < 0.1 && p.life > 10.0 && p.life <= TRAIL_LIFE)
        );
        // A hull that does not move leaves nothing.
        let mut still = Vec::new();
        wake(Vec2::ZERO, Vec2::ZERO, 5.0, 1.0, 1.0 / 30.0, 0, &mut still);
        assert!(still.is_empty());
        // A fast hull lays its records along the path, closer than the
        // blur's reach, with the same amplitude per metre.
        let mut fast = Vec::new();
        wake(Vec2::ZERO, Vec2::new(3.0, 0.0), 5.0, 1.0, 1.0 / 30.0, 0, &mut fast);
        assert_eq!(fast.len(), 15);
        let per_metre = |v: &[WaveParticle], length: f32| v.iter().map(|p| p.amplitude).sum::<f32>() / length;
        assert!((per_metre(&fast, 3.0) - per_metre(&out, 5.0 / 30.0)).abs() < 1e-4);
    }

    #[test]
    fn a_fast_hulls_wake_is_a_length_not_a_time() {
        // A coast ship: radius 5 m at 100 m/s, one tick.
        let (radius, speed) = (5.0, 100.0);
        let mut out = Vec::new();
        wake(
            Vec2::ZERO,
            Vec2::new(speed / 30.0, 0.0),
            radius,
            1.0,
            1.0 / 30.0,
            7,
            &mut out,
        );
        let arm = ARM_RADII * radius / speed;
        let trail = TRAIL_RADII * radius / speed;
        assert!(
            out.iter().all(|p| p.life <= trail + 1e-4),
            "nothing outlives the trail's length"
        );
        let arms: Vec<_> = out.iter().filter(|p| p.velocity.y.abs() > 1.0).collect();
        assert!(!arms.is_empty() && arms.iter().all(|p| (p.life - arm).abs() < 1e-4));
        // The trail frays: its records' lives differ.
        let lives: Vec<f32> = out
            .iter()
            .filter(|p| p.velocity.y.abs() <= 1.0)
            .map(|p| p.life)
            .collect();
        assert!(lives.iter().any(|&l| (l - lives[0]).abs() > 0.05), "{lives:?}");
        // Steady state: a tick's records times their lives, 30 ticks a
        // second. A dozen such hulls stay far under the cap.
        let live: f32 = out.iter().map(|p| p.life).sum::<f32>() * 30.0;
        assert!(12.0 * live < MAX_SOURCES as f32 / 4.0, "{live} records per hull");
    }

    #[test]
    fn a_blast_sends_three_rings_a_boil_and_a_splash() {
        let mut out = Vec::new();
        blast(Vec2::new(10.0, 20.0), 5.0, 2.0, 1, &mut out);
        assert_eq!(out.len(), 1 + 200 + 140 + 100 + 96);
        assert_eq!(out.iter().filter(|p| p.kind == 1.0).count(), 1);
        // The rings start after the cavity opens and run outward.
        let rings: Vec<_> = out.iter().filter(|p| p.amplitude > 0.0 && p.kind == 0.0).collect();
        assert_eq!(rings.len(), 440);
        for p in &rings {
            assert!(p.birth > 2.0);
            assert!((p.position - Vec2::new(10.0, 20.0)).dot(p.velocity) > 0.0);
        }
        // A bullet makes no rings, only a fleck of boil and its splash.
        let mut bullet = Vec::new();
        blast(Vec2::ZERO, 0.3, 0.0, 1, &mut bullet);
        assert_eq!(bullet.len(), 1 + 12);
    }

    #[test]
    fn the_swell_follows_the_maps_look() {
        let s = Swell::of(&WaterParams::default());
        assert!((Vec2::from_array(s.wind).length() - 1.0).abs() < 1e-6);
        assert!(
            s.strength > 0.9 && s.strength < 1.0,
            "the default look is about the prototype's sea"
        );
        assert!(
            s.scale > 1.0,
            "the default 48 m swell is shorter than the prototype's 105 m"
        );
    }
}
