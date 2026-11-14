//! Player commands (`Runtime::tick_with`) on the crossing game, with the
//! game's own orders held back (order tick 255, after the run):
//! * `MoveOrder` commands for 40 units, sent at tick 10: only those travel
//!   far (others are at most pushed aside), nearly all of them;
//! * the same bundles in a fresh run give the same digest;
//! * invalid commands (unknown stream, a stream players may not send on,
//!   a wrong size) are dropped and counted, and the world is as without
//!   them;
//! * an order for an entity that does not exist is ignored, the same way
//!   every run.
//!
//! One runtime at a time: the column arena is process-wide.

use ae_abi::engine::{MOVE_ORDER, TERRAIN_EDIT};
use ae_sdk::Entity;
use ae_sdk::nav::{Body, MoveOrder};
use ae_services::nav::{Map, NavPlugin};
use ae_services::spatial::SpatialPlugin;
use ae_sim::input::{Command, TickBundle};
use ae_sim::runtime::{Runtime, RuntimeConfig};
use anyhow::{Result, ensure};
use std::sync::Arc;

const UNITS: u64 = 600;
const TICKS: u64 = 60;
/// The game's own orders at tick 255: never, in this run.
const PARAM: u64 = UNITS | 255 << 32;

fn bytes_of<T: Copy>(v: &T) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts((v as *const T).cast::<u8>(), size_of::<T>()).to_vec() }
}

fn decode<T: Copy>(bytes: &[u8]) -> T {
    unsafe { bytes.as_ptr().cast::<T>().read_unaligned() }
}

/// Run the crossing game with `script(tick, units)` giving each tick's
/// bundle; the final digest, the units that travelled far, and commands
/// in/dropped.
/// Final digest, the units that travelled far, commands accepted and dropped.
type Outcome = (u128, Vec<u64>, u64, u64);

/// Each tick's bundle from the tick and the units' bodies.
type Script<'a> = &'a dyn Fn(u64, &[(u64, Body)]) -> TickBundle;

fn run(script: Script<'_>) -> Result<Outcome> {
    let wasm = ae_sim::build::game("crossing")?;
    let nav = NavPlugin::new(Map::WaterCrossing, 256);
    let spatial = SpatialPlugin::square(nav.extent(), 64.0);
    let cfg = RuntimeConfig { max_workers: 2, ..RuntimeConfig::default() };
    let mut rt = Runtime::with_hosts(&wasm, cfg, vec![Arc::new(nav), Arc::new(spatial)])?;
    let mut world = rt.world_for(PARAM, 2)?;
    let (mut accepted, mut dropped) = (0, 0);
    let mut start: Vec<(u64, Body)> = Vec::new();
    for _ in 0..TICKS {
        let units: Vec<(u64, Body)> = rt.rows(&world, "Body")?.into_iter().map(|(e, b)| (e, decode(&b))).collect();
        if world.tick == 1 {
            start = units.clone();
        }
        let bundle = script(world.tick, &units);
        let s = rt.tick_with(&mut world, 2, &bundle)?;
        accepted += s.commands_in;
        dropped += s.commands_dropped;
    }
    let end: Vec<(u64, Body)> = rt.rows(&world, "Body")?.into_iter().map(|(e, b)| (e, decode(&b))).collect();
    // Units that travelled over 60 m (ordered ones walk 50 ticks; units
    // they push aside move a few metres).
    let moved = end.iter().zip(&start).filter(|((_, a), (_, b))| (a.x - b.x).hypot(a.y - b.y) > 60.0).map(|((e, _), _)| *e).collect();
    Ok((rt.digest(&world)?.total, moved, accepted, dropped))
}

fn orders(tick: u64, units: &[(u64, Body)]) -> TickBundle {
    let mut b = TickBundle::new(tick);
    if tick == 10 {
        for (e, body) in units.iter().take(40) {
            let order = MoveOrder::move_to(Entity(*e), body.x + 300.0, body.y + 200.0, 1 << 40 | 7);
            b.commands.push(Command { player: 0, stream: MOVE_ORDER, bytes: bytes_of(&order) });
        }
    }
    b
}

#[test]
fn commands_reach_the_sim_identically() -> Result<()> {
    // No commands: nothing moves.
    let (idle, moved, ..) = run(&|t, _| TickBundle::new(t))?;
    ensure!(moved.is_empty(), "{} units moved without orders", moved.len());

    // Orders for 40 units: those move.
    let sent = std::cell::RefCell::new(Vec::new());
    let (ordered, moved, accepted, dropped) = run(&|t, units| {
        if t == 10 {
            sent.borrow_mut().extend(units.iter().take(40).map(|(e, _)| *e));
        }
        orders(t, units)
    })?;
    ensure!(accepted == 40 && dropped == 0, "accepted {accepted}, dropped {dropped}");
    ensure!(ordered != idle, "the orders changed the world");
    let mut sent = sent.into_inner();
    sent.sort_unstable();
    let mut far = moved.clone();
    far.sort_unstable();
    // Only ordered units travel far, and nearly all of them do (one may be
    // blocked or its goal unreachable for its class).
    ensure!(far.iter().all(|e| sent.binary_search(e).is_ok()), "a unit nobody ordered travelled far");
    ensure!(far.len() >= 36, "{} of the 40 ordered units travelled far", far.len());

    // The same bundles again: the same world.
    let (again, ..) = run(&orders)?;
    ensure!(again == ordered, "replaying the bundles gave another digest");

    // Invalid commands next to the valid ones: dropped, world unchanged.
    let with_junk = |t: u64, units: &[(u64, Body)]| {
        let mut b = orders(t, units);
        if t == 10 {
            b.commands.insert(0, Command { player: 0, stream: 0xDEAD_BEEF, bytes: vec![0; 32] });
            b.commands.push(Command { player: 1, stream: TERRAIN_EDIT, bytes: vec![0; 32] });
            b.commands.push(Command { player: 1, stream: MOVE_ORDER, bytes: vec![0; 5] });
        }
        b
    };
    let (junk, _, accepted, dropped) = run(&with_junk)?;
    ensure!(accepted == 40 && dropped == 3, "accepted {accepted}, dropped {dropped}");
    ensure!(junk == ordered, "dropped commands changed the world");

    // An order for an entity that does not exist: no error, same each run.
    let ghost = |t: u64, units: &[(u64, Body)]| {
        let mut b = orders(t, units);
        if t == 10 {
            let order = MoveOrder::move_to(Entity(0x7777_0000_0123_4567), 100.0, 100.0, 1 << 40 | 8);
            b.commands.push(Command { player: 0, stream: MOVE_ORDER, bytes: bytes_of(&order) });
        }
        b
    };
    let (g1, moved, ..) = run(&ghost)?;
    let (g2, ..) = run(&ghost)?;
    ensure!(g1 == g2 && moved.len() >= 36, "an order for a missing entity: digests {g1:x} / {g2:x}, {} moved", moved.len());
    println!("idle {:016x}, ordered {:016x}, with a missing entity {:016x}", idle as u64, ordered as u64, g1 as u64);
    Ok(())
}
