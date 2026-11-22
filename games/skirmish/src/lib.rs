//! Skirmish: two allyteams on a nav map, defined by content.
//!
//! The bases are the map's first two start points (`MapStart`, from a map
//! package). Game parameter: bits 0..24 = 0 (the map spawns no units of
//! its own); bits 24..40 = the map's side in metres, used only on a map
//! without start points (bases then sit 15% in from opposite corners);
//! bits 40..48 = tanks each allyteam starts with (0: [`START_TANKS`]);
//! bits 48..56 = extra tanks allyteam 0 starts with (a handicap). A map
//! package with start points needs no parameter at all.
//!
//! Systems, in schedule order:
//! * `map_starts` (host, tick 0): the map's start points.
//! * `start` (tick 0): each allyteam's `factory` at its base and its
//!   starting `tank`s in a ring around it, as `SpawnRequest`s, and the
//!   `Match` entity (with the bases); the engine's catalogue service spawns
//!   the defs (`content/defs/`).
//! * `publish_bases`: the bases as `Base` events, every tick.
//! * `produce`: a factory earns `income` metal a tick; when it has `cost`
//!   it starts a unit of def `builds` and spawns it `build_ticks` later
//!   beside itself.
//! * `scan`: every damageable unit publishes where it is (`Presence`).
//! * `orders`: players' `UnitCommand`s (move, attack-move, stop; replace
//!   or append) into each unit's `Queue`, and the queue's head into nav
//!   orders (nav takes them next tick). A new unit first gathers at its
//!   allyteam's rally point; an attack-move diverts to enemies within
//!   acquisition range, then goes on. Nothing attacks on its own: players
//!   and bots (`ae_bot`) send the attacks.
//! * `fire` → `armor` → `apply`: a weapon off cooldown targets the nearest
//!   enemy in range (`DamageRequest`); armor reduces each request
//!   (`Damage`, at least 1); damage lowers health, and a unit at 0 is
//!   despawned. Each step is a stream the next reads, in event order, so
//!   the result never depends on how many workers emitted.
//! * `census` → `judge`: factories counted per allyteam; when one has none
//!   left, the other wins (`Match.winner`, `Match.ended_at`); when both lose
//!   their last in the same tick it is a draw (`DRAW`).

use ae_sdk::catalog::{SpawnRequest, def_id};
use ae_sdk::nav::{ANY_DEPTH, Body, CELL_SIZE, MapStart, MoveOrder, MovementClass, NavAgent, Team, mode};
use ae_sdk::{Component, Entity, Run, Spawn, despawn, emit, schedule, system};

ae_sdk::movement_classes![
    MovementClass {
        name: "Scout",
        clearance_cells: 1,
        max_slope: 135,
        radius: CELL_SIZE * 0.34,
        speed: CELL_SIZE * 14.0,
        min_depth: 0,
        max_depth: 20,
        hover: false
    },
    MovementClass {
        name: "Heavy",
        clearance_cells: 2,
        max_slope: 95,
        radius: CELL_SIZE * 0.52,
        speed: CELL_SIZE * 10.0,
        min_depth: 0,
        max_depth: 20,
        hover: false
    },
    MovementClass {
        name: "Ship",
        clearance_cells: 2,
        max_slope: 60,
        radius: CELL_SIZE * 0.62,
        speed: CELL_SIZE * 9.0,
        min_depth: 30,
        max_depth: ANY_DEPTH,
        hover: false
    },
    MovementClass {
        name: "Hover",
        clearance_cells: 1,
        max_slope: 110,
        radius: CELL_SIZE * 0.40,
        speed: CELL_SIZE * 13.0,
        min_depth: 0,
        max_depth: ANY_DEPTH,
        hover: true
    },
    MovementClass {
        name: "Amphibious",
        clearance_cells: 1,
        max_slope: 110,
        radius: CELL_SIZE * 0.48,
        speed: CELL_SIZE * 9.0,
        min_depth: 0,
        max_depth: ANY_DEPTH,
        hover: false
    },
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

/// How clients draw the unit (an index into their looks; content data).
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Look {
    #[field(view = "los")]
    pub class: u32,
}

// Only content sets it; no system names it.
ae_sdk::declare!(Look);

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

/// A unit's command queue: up to [`QUEUE`] commands (`kind`, `x`, `y`;
/// the head first) and what it is doing about the head (`state`). All
/// zeros is a unit fresh from the factory ([`state::NEW`]).
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Queue {
    pub kind: [u32; QUEUE],
    pub x: [f32; QUEUE],
    pub y: [f32; QUEUE],
    #[field(view = "owner")]
    pub len: u32,
    #[field(view = "owner")]
    pub state: u32,
    /// The enemy being chased or held against ([`state::CHASE`], [`state::HOLD`]).
    pub target: Entity,
    /// Tick of the last nav order.
    pub since: u64,
    /// Where the last nav order sent it.
    pub ox: f32,
    pub oy: f32,
}

pub const QUEUE: usize = 4;

/// [`Queue::state`].
pub mod state {
    /// Fresh from the factory: gets its rally order.
    pub const NEW: u32 = 0;
    /// Nothing ordered to nav for the head (an empty queue: idle).
    pub const IDLE: u32 = 1;
    /// Ordered to the head's point.
    pub const HEADING: u32 = 2;
    /// Attack-moving, diverted toward an enemy out of range.
    pub const CHASE: u32 = 3;
    /// Attack-moving, stopped with an enemy in range.
    pub const HOLD: u32 = 4;
}

/// A player's command for one of their units ([`UnitCommand::kind`]).
pub mod command {
    /// Go to `(x, y)`, ignoring enemies on the way.
    pub const MOVE: u32 = 1;
    /// Go to `(x, y)`, engaging every enemy that comes within acquisition
    /// range ([`super::ACQUIRE`] × weapon range) on the way, then going on.
    pub const ATTACK: u32 = 2;
    /// Clear the queue and stop.
    pub const STOP: u32 = 3;
}

/// A command for a unit (sent by players and bots as tick commands):
/// `append` 0 replaces the queue, 1 adds to its end (dropped when full).
/// Commands for entities that are gone or have no queue are ignored.
#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(event, command)]
pub struct UnitCommand {
    pub entity: Entity,
    pub kind: u32,
    pub append: u32,
    pub x: f32,
    pub y: f32,
}

/// Attack-move acquisition range as a multiple of the weapon's range.
pub const ACQUIRE: f32 = 1.5;
/// How far in front of its base (toward the enemy) a new unit gathers, metres.
pub const RALLY: f32 = 220.0;

/// The match: the winning allyteam (`NO_WINNER` until decided) and when,
/// and where each allyteam's base is (every player knows).
#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Match {
    #[field(view = "always")]
    pub winner: u32,
    pub reserved: u32,
    #[field(view = "always")]
    pub ended_at: u64,
    #[field(view = "always")]
    pub base_x: [f32; ALLIES as usize],
    #[field(view = "always")]
    pub base_y: [f32; ALLIES as usize],
}

/// Allyteam `ally`'s base, published every tick by `bases` (from `Match`).
#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(event)]
pub struct Base {
    pub ally: u32,
    pub x: f32,
    pub y: f32,
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

/// Tanks each allyteam starts with when the parameter does not say.
pub const START_TANKS: u64 = 6;

/// The bases: the map's first two start points; on a map without them
/// (the generated maps), 15% in from opposite corners of a map whose side
/// in metres is in the parameter.
fn bases(run: &Run, starts: &[MapStart]) -> [(f32, f32); ALLIES as usize] {
    let at = |i: u32| starts.iter().find(|s| s.index == i).map(|s| (s.x, s.y));
    if let (Some(a), Some(b)) = (at(0), at(1)) {
        return [a, b];
    }
    let e = ((run.param() >> 24) & 0xFFFF) as f32;
    [(e * 0.15, e * 0.15), (e * 0.85, e * 0.85)]
}

fn spawn(def: &str, ally: u32, x: f32, y: f32) {
    emit(&SpawnRequest {
        def: def_id(def),
        ally,
        reserved: 0,
        x,
        y,
    });
}

#[system(reads_events(MapStart), emits(SpawnRequest))]
fn start(run: &Run) {
    if run.tick() != 0 {
        return;
    }
    let at = bases(run, run.events::<MapStart>());
    Spawn::new()
        .with(&Match {
            winner: NO_WINNER,
            reserved: 0,
            ended_at: 0,
            base_x: at.map(|b| b.0),
            base_y: at.map(|b| b.1),
        })
        .submit();
    let tanks = match (run.param() >> 40) & 0xFF {
        0 => START_TANKS,
        n => n,
    };
    for ally in 0..ALLIES {
        let tanks = tanks + if ally == 0 { (run.param() >> 48) & 0xFF } else { 0 };
        let (bx, by) = at[ally as usize];
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
            emit(&SpawnRequest {
                def: f.builds,
                ally: teams[i].ally,
                reserved: 0,
                x: b.x + b.radius + 12.0,
                y: b.y + b.radius + 12.0,
            });
        } else {
            f.progress += 1;
        }
    }
}

/// Every allyteam's base as an event, for systems over other rows.
#[system(serial, read(Match), emits(Base))]
fn publish_bases(run: &Run) {
    let matches = run.read::<Match>();
    for i in run.rows() {
        for ally in 0..ALLIES {
            emit(&Base {
                ally,
                x: matches[i].base_x[ally as usize],
                y: matches[i].base_y[ally as usize],
            });
        }
    }
}

/// Where allyteam `ally`'s new units gather: [`RALLY`] metres from its
/// base toward the enemy's (`None` before the bases are known).
fn rally_point(bases: &[Base], ally: u32) -> Option<(f32, f32)> {
    let at = |a: u32| bases.iter().find(|b| b.ally == a).map(|b| (b.x, b.y));
    let ((bx, by), (ex, ey)) = (at(ally)?, at((ally + 1) % ALLIES)?);
    let (dx, dy) = (ex - bx, ey - by);
    let d = (dx * dx + dy * dy).sqrt().max(1.0);
    Some((bx + dx / d * RALLY, by + dy / d * RALLY))
}

/// Nav order groups of one tick: orders to the same point share a group
/// (one flow field); a group id is never reused (tick in the high bits).
struct Groups {
    tick: u64,
    points: Vec<(u32, u32)>,
}

impl Groups {
    fn of(&mut self, x: f32, y: f32) -> u64 {
        let key = (x.to_bits(), y.to_bits());
        let k = self.points.iter().position(|&p| p == key).unwrap_or_else(|| {
            self.points.push(key);
            self.points.len() - 1
        });
        1u64 << 62 | self.tick << 16 | k as u64
    }
}

fn pop(q: &mut Queue) {
    for k in 1..QUEUE {
        q.kind[k - 1] = q.kind[k];
        q.x[k - 1] = q.x[k];
        q.y[k - 1] = q.y[k];
    }
    q.len = q.len.saturating_sub(1);
    q.kind[QUEUE - 1] = 0;
    q.state = state::IDLE;
    q.target = Entity(0);
}

fn push(q: &mut Queue, kind: u32, x: f32, y: f32) {
    let n = q.len as usize;
    if n < QUEUE {
        (q.kind[n], q.x[n], q.y[n]) = (kind, x, y);
        q.len += 1;
    }
}

/// Players' commands into queues, then each unit's head into nav orders:
/// * a new unit queues a move to its allyteam's rally point;
/// * a move goes to its point; on arrival (or when it cannot get there)
///   the queue moves on;
/// * an attack-move also looks for the nearest enemy within acquisition
///   range every 8 ticks (staggered by entity): out of weapon range it
///   chases it, in range it stops and holds (the weapon fires on its
///   own); with none left it goes on to its point.
///
/// Runs after `scan` (this tick's `Presence`); nav takes its orders next
/// tick.
#[system(
    serial,
    read(Team, Body, NavAgent, Weapon),
    write(Queue),
    reads_events(UnitCommand, Presence, Base),
    emits(MoveOrder)
)]
fn orders(run: &Run) {
    let (teams, bodies, agents, weapons, queues, ids) = (
        run.read::<Team>(),
        run.read::<Body>(),
        run.read::<NavAgent>(),
        run.read::<Weapon>(),
        run.write::<Queue>(),
        run.entities(),
    );
    let tick = run.tick();
    let mut cmds: Vec<(u64, usize)> = run
        .events::<UnitCommand>()
        .iter()
        .enumerate()
        .map(|(k, c)| (c.entity.0, k))
        .collect();
    cmds.sort_unstable();
    let all = run.events::<UnitCommand>();
    let seen = run.events::<Presence>();
    let bases = run.events::<Base>();
    let mut groups = Groups {
        tick,
        points: Vec::new(),
    };
    for i in run.rows() {
        let (q, e, ally, b) = (&mut queues[i], ids[i], teams[i].ally, bodies[i]);
        if q.state == state::NEW
            && let Some((x, y)) = rally_point(bases, ally)
        {
            push(q, command::MOVE, x, y);
            q.state = state::IDLE;
        }
        let from = cmds.partition_point(|c| c.0 < e.0);
        for &(_, k) in cmds[from..].iter().take_while(|c| c.0 == e.0) {
            let c = all[k];
            if c.kind == command::STOP {
                q.len = 0;
                q.state = state::IDLE;
                q.target = Entity(0);
                q.since = tick;
                emit(&MoveOrder::stop(e, groups.of(f32::NAN, 0.0)));
                continue;
            }
            if c.kind != command::MOVE && c.kind != command::ATTACK {
                continue;
            }
            if c.append == 0 {
                q.len = 0;
                q.state = state::IDLE;
                q.target = Entity(0);
            }
            push(q, c.kind, c.x, c.y);
        }
        if q.len == 0 {
            continue;
        }
        // The head reached: next command.
        let settled = matches!(agents[i].mode, mode::ARRIVED | mode::UNREACHABLE) && tick > q.since + 2;
        if q.state == state::HEADING && settled {
            pop(q);
            if q.len == 0 {
                continue;
            }
        }
        let (kind, hx, hy) = (q.kind[0], q.x[0], q.y[0]);
        if kind == command::ATTACK && (e.0 + tick).is_multiple_of(8) {
            let w = weapons[i];
            let reach = w.range * ACQUIRE;
            let mut best: Option<(f32, Presence)> = None;
            for p in seen {
                if p.ally == ally {
                    continue;
                }
                let (dx, dy) = (p.x - b.x, p.y - b.y);
                let d2 = dx * dx + dy * dy;
                if d2 <= reach * reach && best.is_none_or(|(bd, _)| d2 < bd) {
                    best = Some((d2, *p));
                }
            }
            match best {
                Some((d2, p)) if d2 <= (w.range * 0.9) * (w.range * 0.9) => {
                    if q.state != state::HOLD {
                        q.state = state::HOLD;
                        q.target = p.entity;
                        q.since = tick;
                        emit(&MoveOrder::stop(e, groups.of(f32::NAN, 0.0)));
                    }
                }
                Some((_, p)) => {
                    let moved = (p.x - q.ox) * (p.x - q.ox) + (p.y - q.oy) * (p.y - q.oy) > 32.0 * 32.0;
                    if q.state != state::CHASE || q.target != p.entity || moved {
                        (q.state, q.target, q.since, q.ox, q.oy) = (state::CHASE, p.entity, tick, p.x, p.y);
                        emit(&MoveOrder::move_to(e, p.x, p.y, groups.of(p.x, p.y)));
                    }
                }
                None => {
                    if q.state != state::HEADING {
                        q.state = state::IDLE;
                    }
                }
            }
        }
        if q.state == state::IDLE {
            (q.state, q.target, q.since, q.ox, q.oy) = (state::HEADING, Entity(0), tick, hx, hy);
            emit(&MoveOrder::move_to(e, hx, hy, groups.of(hx, hy)));
        }
    }
}

#[system(serial, read(Team, Body, Health), emits(Presence))]
fn scan(run: &Run) {
    let (teams, bodies, ids) = (run.read::<Team>(), run.read::<Body>(), run.entities());
    for i in run.rows() {
        emit(&Presence {
            entity: ids[i],
            ally: teams[i].ally,
            reserved: 0,
            x: bodies[i].x,
            y: bodies[i].y,
        });
    }
}

#[system(
    serial,
    read(Team, Body),
    write(Weapon),
    reads_events(Presence),
    emits(DamageRequest)
)]
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
            emit(&DamageRequest {
                target,
                amount: w.damage,
                ally: me,
            });
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
        let a = by_entity
            .binary_search_by_key(&r.target.0, |e| e.0)
            .map_or(0, |k| by_entity[k].1);
        emit(&Damage {
            target: r.target,
            amount: r.amount.saturating_sub(a).max(1),
            ally: r.ally,
        });
    }
}

#[system(serial, read(Team), write(Health), reads_events(Damage))]
fn apply(run: &Run) {
    let (teams, health, ids) = (run.read::<Team>(), run.write::<Health>(), run.entities());
    let mut hits: Vec<(u64, u32, u32)> = run
        .events::<Damage>()
        .iter()
        .map(|d| (d.target.0, d.ally, d.amount))
        .collect();
    hits.sort_unstable();
    for i in run.rows() {
        let e = ids[i].0;
        let from = hits.partition_point(|h| h.0 < e);
        let total: u32 = hits[from..]
            .iter()
            .take_while(|h| h.0 == e)
            .filter(|h| h.1 != teams[i].ally)
            .map(|h| h.2)
            .fold(0, u32::saturating_add);
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
        emit(&FactoryCount {
            ally: ally as u32,
            count,
        });
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

schedule!(
    host(map_starts),
    start,
    publish_bases,
    produce,
    host(catalog_spawn),
    host(nav),
    host(spatial),
    scan,
    orders,
    fire,
    armor,
    apply,
    census,
    judge
);
