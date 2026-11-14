//! Skirmish: two allyteams on a nav map, defined by content.
//!
//! Game parameter: bits 0..24 = 0 (the map spawns no units of its own),
//! bits 24..40 = the map's side in metres (bases sit 15% in from opposite
//! corners), bits 40..48 = tanks each allyteam starts with, bits 48..56 =
//! extra tanks allyteam 0 starts with (a handicap; 0 for an even match).
//!
//! Systems, in schedule order:
//! * `start` (tick 0): each allyteam's `factory` at its base and its
//!   starting `tank`s in a ring around it, as `SpawnRequest`s, and the
//!   `Match` entity; the engine's catalogue service spawns the defs
//!   (`content/defs/`).
//! * `produce`: a factory earns `income` metal a tick; when it has `cost`
//!   it starts a unit of def `builds` and spawns it `build_ticks` later
//!   beside itself.
//! * `rally`: a unit with `Orders.given == 0` is sent to the enemy base.
//! * `scan` → `fire` → `armor` → `apply`: every damageable unit publishes
//!   where it is (`Presence`); a weapon off cooldown targets the nearest
//!   enemy in range (`DamageRequest`); armor reduces each request
//!   (`Damage`, at least 1); damage lowers health, and a unit at 0 is
//!   despawned. Each step is a stream the next reads, in event order, so
//!   the result never depends on how many workers emitted.
//! * `census` → `judge`: factories counted per allyteam; when one has none
//!   left, the other wins (`Match.winner`, `Match.ended_at`); when both lose
//!   their last in the same tick it is a draw (`DRAW`).

use ae_sdk::catalog::{SpawnRequest, def_id};
use ae_sdk::nav::{ANY_DEPTH, Body, CELL_SIZE, MoveOrder, MovementClass, Team};
use ae_sdk::{Component, Entity, Run, Spawn, despawn, emit, schedule, system};

ae_sdk::movement_classes![
    MovementClass { name: "Scout", clearance_cells: 1, max_slope: 135, radius: CELL_SIZE * 0.34, speed: CELL_SIZE * 14.0, min_depth: 0, max_depth: 20, hover: false },
    MovementClass { name: "Heavy", clearance_cells: 2, max_slope: 95, radius: CELL_SIZE * 0.52, speed: CELL_SIZE * 10.0, min_depth: 0, max_depth: 20, hover: false },
    MovementClass { name: "Ship", clearance_cells: 2, max_slope: 60, radius: CELL_SIZE * 0.62, speed: CELL_SIZE * 9.0, min_depth: 30, max_depth: ANY_DEPTH, hover: false },
    MovementClass { name: "Hover", clearance_cells: 1, max_slope: 110, radius: CELL_SIZE * 0.40, speed: CELL_SIZE * 13.0, min_depth: 0, max_depth: ANY_DEPTH, hover: true },
    MovementClass { name: "Amphibious", clearance_cells: 1, max_slope: 110, radius: CELL_SIZE * 0.48, speed: CELL_SIZE * 9.0, min_depth: 0, max_depth: ANY_DEPTH, hover: false },
];

// The engine services' components (Def, Team and Body are only ever
// created by the catalogue service).
ae_sdk::use_catalog!();

/// Hit points.
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Health {
    #[field(view = "los")]
    pub hp: u32,
    #[field(view = "los")]
    pub max: u32,
}

/// Damage taken off every hit (a hit does at least 1).
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Armor {
    #[field(view = "los")]
    pub value: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Weapon {
    #[field(view = "los")]
    pub range: f32,
    #[field(view = "los")]
    pub damage: u32,
    /// Ticks between shots.
    #[field(view = "los")]
    pub reload: u32,
    #[field(view = "owner")]
    pub cooldown: u32,
}

/// A factory with its own economy.
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Factory {
    /// The def it builds.
    #[field(view = "los")]
    pub builds: u64,
    #[field(view = "owner")]
    pub metal: u32,
    #[field(view = "owner")]
    pub income: u32,
    #[field(view = "owner")]
    pub cost: u32,
    #[field(view = "owner")]
    pub build_ticks: u32,
    /// Ticks into the unit being built (0: idle).
    #[field(view = "owner")]
    pub progress: u32,
    pub reserved: u32,
}

/// Whether the unit was given its rally order.
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Orders {
    #[field(view = "owner")]
    pub given: u32,
}

/// The match: the winning allyteam (`NO_WINNER` until decided) and when.
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Match {
    #[field(view = "always")]
    pub winner: u32,
    pub reserved: u32,
    #[field(view = "always")]
    pub ended_at: u64,
}

pub const NO_WINNER: u32 = u32::MAX;
pub const DRAW: u32 = u32::MAX - 1;

#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(event)]
pub struct Presence {
    pub entity: Entity,
    pub ally: u32,
    pub reserved: u32,
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(event)]
pub struct DamageRequest {
    pub target: Entity,
    pub amount: u32,
    /// The shooter's allyteam.
    pub ally: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(event)]
pub struct Damage {
    pub target: Entity,
    pub amount: u32,
    pub ally: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(event)]
pub struct FactoryCount {
    pub ally: u32,
    pub count: u32,
}

pub const ALLIES: u32 = 2;

/// The map's side, metres.
pub fn extent(run: &Run) -> f32 {
    ((run.param() >> 24) & 0xFFFF) as f32
}

/// Allyteam `ally`'s base.
pub fn base(run: &Run, ally: u32) -> (f32, f32) {
    let e = extent(run);
    if ally == 0 { (e * 0.15, e * 0.15) } else { (e * 0.85, e * 0.85) }
}

fn spawn(def: &str, ally: u32, x: f32, y: f32) {
    emit(&SpawnRequest { def: def_id(def), ally, reserved: 0, x, y });
}

#[system(emits(SpawnRequest))]
fn start(run: &Run) {
    if run.tick() != 0 {
        return;
    }
    Spawn::new().with(&Match { winner: NO_WINNER, reserved: 0, ended_at: 0 }).submit();
    for ally in 0..ALLIES {
        let tanks = ((run.param() >> 40) & 0xFF) + if ally == 0 { (run.param() >> 48) & 0xFF } else { 0 };
        let (bx, by) = base(run, ally);
        spawn("factory", ally, bx, by);
        for i in 0..tanks {
            let a = i as f32 / tanks.max(1) as f32 * core::f32::consts::TAU;
            let (c, s) = cos_sin(a);
            spawn("tank", ally, bx + 90.0 * c, by + 90.0 * s);
        }
    }
}

/// cos and sin of `a` in [0, 2π) by Taylor series around the nearest
/// multiple of π/2 (only + - *: the same on every machine).
fn cos_sin(a: f32) -> (f32, f32) {
    let q = (a / core::f32::consts::FRAC_PI_2 + 0.5) as i32;
    let x = a - q as f32 * core::f32::consts::FRAC_PI_2;
    let x2 = x * x;
    let s = x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0)));
    let c = 1.0 - x2 / 2.0 * (1.0 - x2 / 12.0 * (1.0 - x2 / 30.0));
    match q.rem_euclid(4) {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    }
}

#[system(serial, read(Team, Body), write(Factory), emits(SpawnRequest))]
fn produce(run: &Run) {
    let (teams, bodies, factories) = (run.read::<Team>(), run.read::<Body>(), run.write::<Factory>());
    for i in run.rows() {
        let f = &mut factories[i];
        f.metal = f.metal.saturating_add(f.income);
        if f.progress == 0 {
            if f.metal >= f.cost {
                f.metal -= f.cost;
                f.progress = 1;
            }
        } else if f.progress >= f.build_ticks {
            f.progress = 0;
            let b = bodies[i];
            emit(&SpawnRequest { def: f.builds, ally: teams[i].ally, reserved: 0, x: b.x + b.radius + 12.0, y: b.y + b.radius + 12.0 });
        } else {
            f.progress += 1;
        }
    }
}

#[system(serial, read(Team), write(Orders), emits(MoveOrder))]
fn rally(run: &Run) {
    let (teams, orders, ids) = (run.read::<Team>(), run.write::<Orders>(), run.entities());
    for i in run.rows() {
        if orders[i].given != 0 {
            continue;
        }
        orders[i].given = 1;
        let ally = teams[i].ally;
        let (x, y) = base(run, (ally + 1) % ALLIES);
        // One group per allyteam and tick: units sent together share a flow field.
        let group = 1u64 << 41 | run.tick() << 8 | ally as u64;
        emit(&MoveOrder::move_to(ids[i], x, y, group));
    }
}

#[system(serial, read(Team, Body, Health), emits(Presence))]
fn scan(run: &Run) {
    let (teams, bodies, ids) = (run.read::<Team>(), run.read::<Body>(), run.entities());
    for i in run.rows() {
        emit(&Presence { entity: ids[i], ally: teams[i].ally, reserved: 0, x: bodies[i].x, y: bodies[i].y });
    }
}

#[system(serial, read(Team, Body), write(Weapon), reads_events(Presence), emits(DamageRequest))]
fn fire(run: &Run) {
    let (teams, bodies, weapons) = (run.read::<Team>(), run.read::<Body>(), run.write::<Weapon>());
    let seen = run.events::<Presence>();
    for i in run.rows() {
        let w = &mut weapons[i];
        if w.cooldown > 0 {
            w.cooldown -= 1;
            continue;
        }
        let (me, b) = (teams[i].ally, bodies[i]);
        let r2 = w.range * w.range;
        // The nearest enemy in range; ties keep the first seen.
        let mut best: Option<(f32, Entity)> = None;
        for p in seen {
            if p.ally == me {
                continue;
            }
            let (dx, dy) = (p.x - b.x, p.y - b.y);
            let d2 = dx * dx + dy * dy;
            if d2 <= r2 && best.is_none_or(|(bd, _)| d2 < bd) {
                best = Some((d2, p.entity));
            }
        }
        if let Some((_, target)) = best {
            emit(&DamageRequest { target, amount: w.damage, ally: me });
            w.cooldown = w.reload;
        }
    }
}

#[system(serial, read(Armor), reads_events(DamageRequest), emits(Damage))]
fn armor(run: &Run) {
    let (armors, ids) = (run.read::<Armor>(), run.entities());
    let mut by_entity: Vec<(u64, u32)> = run.rows().map(|i| (ids[i].0, armors[i].value)).collect();
    by_entity.sort_unstable();
    for r in run.events::<DamageRequest>() {
        let a = by_entity.binary_search_by_key(&r.target.0, |e| e.0).map_or(0, |k| by_entity[k].1);
        emit(&Damage { target: r.target, amount: r.amount.saturating_sub(a).max(1), ally: r.ally });
    }
}

#[system(serial, read(Team), write(Health), reads_events(Damage))]
fn apply(run: &Run) {
    let (teams, health, ids) = (run.read::<Team>(), run.write::<Health>(), run.entities());
    let mut hits: Vec<(u64, u32, u32)> = run.events::<Damage>().iter().map(|d| (d.target.0, d.ally, d.amount)).collect();
    hits.sort_unstable();
    for i in run.rows() {
        let e = ids[i].0;
        let from = hits.partition_point(|h| h.0 < e);
        let total: u32 = hits[from..].iter().take_while(|h| h.0 == e).filter(|h| h.1 != teams[i].ally).map(|h| h.2).fold(0, u32::saturating_add);
        if total == 0 {
            continue;
        }
        health[i].hp = health[i].hp.saturating_sub(total);
        if health[i].hp == 0 {
            despawn(ids[i]);
        }
    }
}

#[system(serial, read(Team, Factory), emits(FactoryCount))]
fn census(run: &Run) {
    let teams = run.read::<Team>();
    let mut count = [0u32; ALLIES as usize];
    for i in run.rows() {
        if let Some(c) = count.get_mut(teams[i].ally as usize) {
            *c += 1;
        }
    }
    for (ally, count) in count.into_iter().enumerate() {
        emit(&FactoryCount { ally: ally as u32, count });
    }
}

#[system(serial, write(Match), reads_events(FactoryCount))]
fn judge(run: &Run) {
    let matches = run.write::<Match>();
    let counts = run.events::<FactoryCount>();
    for i in run.rows() {
        let m = &mut matches[i];
        if m.winner != NO_WINNER || run.tick() == 0 {
            continue;
        }
        let lost: Vec<u32> = counts.iter().filter(|c| c.count == 0).map(|c| c.ally).collect();
        match lost.len() {
            0 => {}
            1 => m.winner = (lost[0] + 1) % ALLIES,
            _ => m.winner = DRAW,
        }
        if m.winner != NO_WINNER {
            m.ended_at = run.tick();
        }
    }
}

schedule!(start, produce, host(catalog_spawn), rally, host(nav), host(spatial), scan, fire, armor, apply, census, judge);
