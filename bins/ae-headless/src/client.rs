//! Lockstep client (`--connect ADDR --player P`): runs the world the
//! command line names, tick `t` only with the relay's bundle `t`.
//!
//! With `--script orders` it plays player P: at the script's order ticks
//! it sends up to 32 of the orders the replay script would give player P,
//! for a tick `delay` ahead, where `delay` follows the measured round trip
//! (half of it in ticks, plus 2). Inputs are resent until the relay
//! reports them taken. Digests go to the relay at every checkpoint tick
//! and at the end. `--latency/--jitter/--loss` emulate a bad network on
//! what this client sends; `--inject TICK:Component.field:ROW` makes this
//! client diverge (the relay must notice). With `--script bot` (the
//! skirmish game) player P is allyteam P's bot of `ae_bot::SkirmishBot::pair`
//! instead, sending what it decides the same way.

use super::{Opts, Script};
use super::replay::{content, header, orders, runtime};
use ae_net::{Emulator, Packet, micros};
use ae_sim::input::{Command, TickBundle};
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

/// Orders a player sends per order tick, at most (one datagram's worth).
const MAX_ORDERS: usize = 32;

/// The intended tick of a new batch of inputs: `delay` ticks ahead, but
/// never a tick a batch was already sent for. The relay keeps the first
/// version it receives of each `(player, intended tick)` (resends are
/// identical), so adding to a sent batch would be lost.
fn schedule(tick: u64, delay: u64, last: Option<u64>) -> u64 {
    let at = tick + delay;
    match last {
        Some(l) if at <= l => l + 1,
        _ => at,
    }
}

pub fn run(o: &Opts, relay: &str, player: u32) -> Result<bool> {
    let relay: SocketAddr = relay.parse().with_context(|| format!("relay address {relay}"))?;
    let wasm = ae_sim::build::game(&o.game)?;
    let nav = o.nav.map(|m| (m, o.map_cells, o.teams));
    let param = o.units | (o.param_hi as u64) << 32;
    let content = content(o, &wasm)?;
    let h = header(&wasm, param, nav, content.as_ref(), o.checkpoint_every.max(1))?;
    let workers = o.workers[0];
    let mut rt = runtime(&wasm, nav, content.as_ref(), workers, o.range_rows)?;
    let mut world = rt.world_for(param, workers)?;
    let bot = if o.script == Script::Bot { Some(ae_bot::SkirmishBot::pair(&rt)?.into_iter().nth(player as usize).context("--script bot plays player 0 or 1")?) } else { None };

    let socket = UdpSocket::bind("127.0.0.1:0")?;
    socket.set_read_timeout(Some(Duration::from_millis(1)))?;
    let mut net = Emulator::new(o.latency, o.jitter, o.loss, o.seed ^ ((player as u64 + 1) * 0x9E37));
    let epoch = Instant::now();
    let mut buf = vec![0u8; 65536];
    let hello = Packet::Hello { player, header: h.clone() }.encode();

    let mut bundles: BTreeMap<u64, TickBundle> = BTreeMap::new();
    let mut pending: BTreeMap<u64, Vec<Command>> = BTreeMap::new();
    let mut reported: Vec<(u64, u128)> = Vec::new();
    let (mut end, mut desync, mut joined) = (0u64, 0u64, false);
    let mut rtt_ms = 100.0f64;
    let mut sent_orders = 0usize;
    let mut last_intended: Option<u64> = None;
    // Most bundles waiting at once (how far ahead the relay ran).
    let mut most_queued = 0usize;
    let mut last_send = Instant::now() - Duration::from_secs(1);
    let mut finished: Option<Instant> = None;
    let period = Duration::from_secs_f64(1.0 / 30.0);
    loop {
        // Receive.
        while let Ok((n, _)) = socket.recv_from(&mut buf) {
            let Ok(Packet::Relay { bundles: bs, taken, echo_us, end: e, desync: d }) = Packet::decode(&buf[..n]) else { continue };
            joined = true;
            for b in bs {
                if b.tick >= world.tick {
                    bundles.entry(b.tick).or_insert(b);
                }
            }
            pending.retain(|t, _| *t > taken);
            if echo_us > 0 {
                let sample = (micros(epoch).saturating_sub(echo_us)) as f64 / 1000.0;
                rtt_ms = rtt_ms * 0.9 + sample * 0.1;
            }
            end = end.max(e);
            desync = desync.max(d);
        }
        // Run every tick whose bundle is here.
        most_queued = most_queued.max(bundles.len());
        while let Some(bundle) = bundles.remove(&world.tick) {
            if o.script != Script::None {
                let mine: Vec<Command> = if let Some(bot) = &bot {
                    let mut c = Vec::new();
                    bot.commands(&rt, &world, &mut c)?;
                    c.into_iter().take(MAX_ORDERS).collect()
                } else {
                    orders(&rt, &world, 16)?.commands.into_iter().filter(|c| c.player == player).take(MAX_ORDERS).collect()
                };
                if !mine.is_empty() {
                    let delay = ((rtt_ms / 2.0) / (period.as_secs_f64() * 1000.0)).ceil() as u64 + 2;
                    let at = schedule(world.tick, delay, last_intended);
                    last_intended = Some(at);
                    sent_orders += mine.len();
                    pending.insert(at, mine);
                }
            }
            if let Some(fault) = &o.inject
                && let Some(e) = fault.apply(&rt, &mut world)?
            {
                eprintln!("player {player}: injected a fault in {}.{} of entity {e:#x} after tick {}", fault.component, fault.field, fault.tick);
            }
            rt.tick_with(&mut world, workers, &bundle)?;
            if world.tick % h.checkpoint_every == 0 || world.tick == end {
                reported.push((world.tick, rt.digest(&world)?.total));
                if reported.len() > 8 {
                    reported.remove(0);
                }
            }
        }
        // Send (every 10 ms).
        if last_send.elapsed() >= Duration::from_millis(10) {
            last_send = Instant::now();
            let packet = if joined {
                Packet::Client { player, inputs: pending.iter().map(|(t, c)| (*t, c.clone())).collect(), want: world.tick, digests: reported.clone(), sent_us: micros(epoch) }.encode()
            } else {
                hello.clone()
            };
            net.send(relay, packet);
        }
        net.flush(&socket)?;
        if end > 0 && world.tick >= end {
            let since = *finished.get_or_insert_with(Instant::now);
            // Keep reporting the final digest a while (packets get lost).
            if since.elapsed() > Duration::from_millis(1500) {
                break;
            }
        }
        std::thread::sleep(Duration::from_micros(500));
    }
    let digest = rt.digest(&world)?;
    println!(
        "player {player}: {} ticks, {sent_orders} orders sent, at most {most_queued} bundles queued, rtt {rtt_ms:.0} ms, sent {} datagrams (dropped {}); digest {:016x}{}",
        world.tick,
        net.sent,
        net.dropped,
        digest.short(),
        if desync > 0 { format!("; the relay saw a DESYNC at tick {desync}") } else { String::new() }
    );
    Ok(desync == 0)
}

#[cfg(test)]
mod tests {
    use super::schedule;

    /// Two batches that would land on one tick (the delay shrank as the
    /// round trip improved) get distinct ticks, in order.
    #[test]
    fn batches_never_share_an_intended_tick() {
        let first = schedule(100, 6, None);
        let second = schedule(102, 3, Some(first));
        assert_eq!((first, second), (106, 107));
        let third = schedule(110, 2, Some(second));
        assert_eq!(third, 112, "a later tick is used as is");
        let mut last = None;
        let mut seen = std::collections::BTreeSet::new();
        for (tick, delay) in [(5, 9), (6, 8), (7, 7), (8, 2), (9, 1), (10, 0), (30, 4)] {
            let at = schedule(tick, delay, last);
            assert!(seen.insert(at) && last.is_none_or(|l| at > l) && at >= tick + delay);
            last = Some(at);
        }
    }
}
