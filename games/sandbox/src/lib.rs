//! Sandbox game: the M1 exit workload.
//!
//! Units wander between goals, steered by a small potential field (a few
//! square roots and divisions per row, so system time dominates dispatch
//! cost). Each wears down; when its health runs out it emits a `Death`
//! event, despawns, and a replacement spawns elsewhere. A serial census
//! system counts deaths from the event stream into a singleton.
//!
//! Game parameter: low 32 bits = unit count; bits 32..48 = wear in percent
//! (0 means 100). At 100% a unit lives about 30 s (at 30 Hz); 500% is the
//! stress setting, about 6 s, roughly 240 deaths per tick at 50k units.

use ae_sdk::{Component, Entity, Run, Spawn, emit, schedule, system};

#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Velocity {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Brain {
    pub goal: [f32; 2],
    pub rng: u32,
    pub speed: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Health {
    pub hp: f32,
    pub armour: f32,
}

/// Singleton: one entity carries it.
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Census {
    pub deaths: u64,
    /// Sum of death positions' bits, so the digest sees which units died where.
    pub where_sum: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(event)]
pub struct Death {
    pub entity: Entity,
    pub x: f32,
    pub y: f32,
}

const MAP: f32 = 4096.0;
const DT: f32 = 1.0 / 30.0;
const ATTRACTORS: [[f32; 3]; 8] = [
    [512.0, 512.0, 900.0],
    [3584.0, 512.0, 700.0],
    [2048.0, 2048.0, 1500.0],
    [512.0, 3584.0, 800.0],
    [3584.0, 3584.0, 1100.0],
    [1024.0, 2560.0, -600.0],
    [3072.0, 1536.0, -700.0],
    [2048.0, 800.0, 500.0],
];

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

fn unit(s: &mut u32) -> f32 {
    (xorshift(s) >> 8) as f32 / (1u32 << 24) as f32
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn spawn_unit(seed: u32) {
    let mut s = seed | 1;
    let p = Position {
        x: unit(&mut s) * MAP,
        y: unit(&mut s) * MAP,
    };
    let goal = [unit(&mut s) * MAP, unit(&mut s) * MAP];
    let brain = Brain {
        goal,
        rng: s,
        speed: 40.0 + unit(&mut s) * 60.0,
    };
    let health = Health {
        hp: 60.0 + unit(&mut s) * 140.0,
        armour: unit(&mut s) * 0.5,
    };
    Spawn::new()
        .with(&p)
        .with(&Velocity { x: 0.0, y: 0.0 })
        .with(&brain)
        .with(&health)
        .submit();
}

fn wear_percent(run: &Run) -> f32 {
    match (run.param() >> 32) as u16 {
        0 => 100.0,
        p => p as f32,
    }
}

/// Tick 0: the units and the census singleton.
#[system(once, serial)]
fn setup(run: &Run) {
    for i in 0..run.param() as u32 {
        spawn_unit(i.wrapping_mul(0x9E37_79B9) ^ 0x5EED);
    }
    Spawn::new()
        .with(&Census {
            deaths: 0,
            where_sum: 0,
        })
        .submit();
}

/// Steering, a quarter of the units per tick.
#[system(staggered(4), write(Velocity, Brain), read(Position))]
fn think(run: &Run) {
    let (vel, brain, pos) = (run.write::<Velocity>(), run.write::<Brain>(), run.read::<Position>());
    for i in run.rows() {
        let (p, b) = (pos[i], &mut brain[i]);
        let (mut dx, mut dy) = (b.goal[0] - p.x, b.goal[1] - p.y);
        let d = (dx * dx + dy * dy).sqrt();
        if d < 32.0 {
            b.goal = [unit(&mut b.rng) * MAP, unit(&mut b.rng) * MAP];
            continue;
        }
        dx /= d;
        dy /= d;
        // Potential field: attractors pull (or push) with 1/r falloff.
        for [ax, ay, w] in ATTRACTORS {
            let (ex, ey) = (ax - p.x, ay - p.y);
            let r2 = ex * ex + ey * ey + 1.0e4;
            let r = r2.sqrt();
            dx += w * ex / (r2 * r) * 40.0;
            dy += w * ey / (r2 * r) * 40.0;
        }
        let n = (dx * dx + dy * dy).sqrt().max(1.0e-3);
        let jitter = unit(&mut b.rng) * 0.2 + 0.9;
        vel[i] = Velocity {
            x: dx / n * b.speed * jitter,
            y: dy / n * b.speed * jitter,
        };
    }
}

#[system(write(Position), read(Velocity))]
fn integrate(run: &Run) {
    let (pos, vel) = (run.write::<Position>(), run.read::<Velocity>());
    for i in run.rows() {
        pos[i].x = (pos[i].x + vel[i].x * DT).clamp(0.0, MAP);
        pos[i].y = (pos[i].y + vel[i].y * DT).clamp(0.0, MAP);
    }
}

/// Every other tick: random wear; the dead are replaced.
#[system(every(2), write(Health), read(Position), emits(Death))]
fn wear(run: &Run) {
    let (hp, pos, ids) = (run.write::<Health>(), run.read::<Position>(), run.entities());
    let scale = 0.0008 * wear_percent(run) / 100.0;
    for i in run.rows() {
        let roll = mix(ids[i].0 ^ run.tick() << 32);
        let hit = (roll % 1000) as f32 * scale * (1.0 - hp[i].armour);
        hp[i].hp -= hit;
        if hp[i].hp <= 0.0 {
            emit(&Death {
                entity: ids[i],
                x: pos[i].x,
                y: pos[i].y,
            });
            ae_sdk::despawn(ids[i]);
            spawn_unit((roll >> 32) as u32);
        }
    }
}

#[system(serial, write(Census), reads_events(Death))]
fn census(run: &Run) {
    let deaths = run.events::<Death>();
    for c in run.write::<Census>() {
        c.deaths += deaths.len() as u64;
        for d in deaths {
            c.where_sum = c
                .where_sum
                .wrapping_add(mix(d.entity.0 ^ (d.x.to_bits() as u64) << 32 ^ d.y.to_bits() as u64));
        }
    }
}

schedule!(setup, think, integrate, wear, census);
