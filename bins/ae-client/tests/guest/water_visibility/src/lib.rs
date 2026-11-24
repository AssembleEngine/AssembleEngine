//! Controlled sensor changes for the client's real observer-stream tests.
use ae_sdk::water::{Dive, Sonar};
use ae_sdk::{Component, Run, Spawn, schedule, system};

#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(id = 0xAE00_0000_0000_0001)]
struct Body {
    #[field(view = "radar")]
    x: f32,
    #[field(view = "radar")]
    y: f32,
    #[field(view = "los")]
    vx: f32,
    #[field(view = "los")]
    vy: f32,
    #[field(view = "los")]
    radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(id = 0xAE00_0000_0000_0009)]
struct Team {
    #[field(view = "always")]
    ally: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(id = 0xAE00_0000_0000_000A)]
struct Sensor {
    los: f32,
    radar: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(id = 0xAE00_0000_0000_0002)]
struct NavAgent {
    // Opt this fixture into ghosts without changing any game's schema.
    #[field(view = "ghost")]
    profile: u32,
}

const HIDDEN_SUB: u64 = 1;
const BLIP: u64 = 2;
const GHOST: u64 = 3;
const EXPLORED: u64 = 4;
const SONAR: u64 = 5;
const MOVING_SHIP: u64 = 6;
const MOVING_SUB: u64 = 7;
const MOVE: u64 = 32;

#[system(write(Sensor, Sonar))]
fn coverage(run: &Run) {
    let mode = run.param() & 15;
    for (sensor, sonar) in run.write::<Sensor>().iter_mut().zip(run.write::<Sonar>()) {
        sensor.los = if matches!(mode, HIDDEN_SUB | SONAR)
            || (run.tick() < 6 && matches!(mode, MOVING_SHIP | MOVING_SUB))
            || (run.tick() < 2 && matches!(mode, GHOST | EXPLORED))
        {
            640.0
        } else {
            0.0
        };
        sensor.radar = if mode == BLIP { 640.0 } else { 0.0 };
        sonar.radius = if (mode == SONAR && run.tick() == 2) || (mode == MOVING_SUB && run.tick() < 6) {
            640.0
        } else {
            0.0
        };
    }
}

#[system]
fn setup(run: &Run) {
    let mode = run.param() & 15;
    if run.tick() == 0 {
        Spawn::new()
            .with(&Body {
                x: 40.0,
                y: 48.0,
                vx: 0.0,
                vy: 0.0,
                radius: 1.2,
            })
            .with(&Team { ally: 0 })
            .with(&Sensor { los: 0.0, radar: 0.0 })
            .with(&Sonar { radius: 0.0 })
            .with(&NavAgent { profile: 0 })
            .submit();
    }
    let present = run.param() & 16 != 0;
    if present && run.tick() == if mode == EXPLORED { 2 } else { 0 } {
        let target = Spawn::new()
            .with(&Body {
                x: 256.0,
                y: 256.0,
                vx: 0.0,
                vy: 0.0,
                radius: 2.3,
            })
            .with(&Team { ally: 1 })
            .with(&Sensor { los: 0.0, radar: 0.0 })
            .with(&NavAgent { profile: 2 });
        if matches!(mode, HIDDEN_SUB | SONAR | MOVING_SUB) {
            target
                .with(&Dive {
                    depth: 4.0,
                    target: 4.0,
                })
                .submit();
        } else {
            target.submit();
        }
    }
}

#[system(write(Body))]
fn advance(run: &Run) {
    if run.param() & MOVE != 0 {
        for body in run.write::<Body>().iter_mut().filter(|b| b.radius > 2.0) {
            body.x += 2.5;
            body.vx = 30.0;
        }
    }
}

schedule!(setup, coverage, advance);
