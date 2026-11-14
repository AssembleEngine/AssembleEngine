//! Lockstep relay (no sim).
//!
//! ```text
//! ae-relay --players N --ticks T [--bind ADDR] [--hz 30] [--record FILE.aerep]
//!          [--latency MS] [--jitter MS] [--loss PCT] [--seed S]
//! ```
//! Waits for N players (`ae-headless --connect`), checks they all build the
//! same world (the replay header they send), then closes one bundle per
//! tick at `--hz`: every input a player sent for this tick or earlier and
//! not yet taken, by player, then intended tick, then the player's order.
//! Clients run bundle `t` as tick `t`; they report digests, which must
//! agree. After tick T it tells everyone the game ended, waits for every
//! final digest, and exits 0 if no digest ever disagreed. `--record`
//! writes the game as a replay (bundles, and the clients' digests as
//! checkpoints) that `ae-headless --replay --verify` can check alone.
//! `--latency/--jitter/--loss` emulate a bad network on what the relay
//! sends (clients have the same options for theirs).

use ae_net::{Emulator, Packet, micros};
use ae_replay::{Checkpoint, Header, Recorder};
use ae_sim::input::{Command, TickBundle};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::net::{SocketAddr, UdpSocket};
use std::process::ExitCode;
use std::time::{Duration, Instant};

struct Opts {
    players: u32,
    ticks: u64,
    bind: String,
    hz: f64,
    record: Option<String>,
    latency: u64,
    jitter: u64,
    loss: f64,
    seed: u64,
}

fn parse() -> Result<Opts> {
    let mut o = Opts { players: 2, ticks: 300, bind: "127.0.0.1:7600".into(), hz: 30.0, record: None, latency: 0, jitter: 0, loss: 0.0, seed: 1 };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--players" => o.players = value()?.parse()?,
            "--ticks" => o.ticks = value()?.parse()?,
            "--bind" => o.bind = value()?,
            "--hz" => o.hz = value()?.parse()?,
            "--record" => o.record = Some(value()?),
            "--latency" => o.latency = value()?.parse()?,
            "--jitter" => o.jitter = value()?.parse()?,
            "--loss" => o.loss = value()?.parse::<f64>()? / 100.0,
            "--seed" => o.seed = value()?.parse()?,
            _ => bail!("unknown argument {a}"),
        }
    }
    ensure!(o.players > 0 && o.hz > 0.0, "--players and --hz must be positive");
    Ok(o)
}

struct Peer {
    addr: SocketAddr,
    /// Next bundle this peer needs.
    want: u64,
    /// Inputs received, not yet taken: intended tick -> commands.
    pending: BTreeMap<u64, Vec<Command>>,
    /// Newest intended tick taken into a bundle.
    taken: u64,
    echo_us: u64,
    final_digest: Option<u128>,
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool> {
    let o = parse()?;
    let socket = UdpSocket::bind(&o.bind).with_context(|| format!("bind {}", o.bind))?;
    socket.set_read_timeout(Some(Duration::from_millis(1)))?;
    let mut net = Emulator::new(o.latency, o.jitter, o.loss, o.seed);
    let epoch = Instant::now();
    let mut buf = vec![0u8; 65536];
    let mut peers: BTreeMap<u32, Peer> = BTreeMap::new();
    let mut header: Option<Header> = None;
    eprintln!("relay on {}: waiting for {} players", socket.local_addr()?, o.players);

    // Lobby: every player says hello with the same world.
    while peers.len() < o.players as usize {
        if let Ok((n, from)) = socket.recv_from(&mut buf)
            && let Ok(Packet::Hello { player, header: h }) = Packet::decode(&buf[..n])
        {
            match &header {
                None => header = Some(h),
                Some(first) => first.check(&h).with_context(|| format!("player {player} builds another world"))?,
            }
            ensure!(player < o.players, "player {player} of {}", o.players);
            peers.entry(player).or_insert(Peer { addr: from, want: 0, pending: BTreeMap::new(), taken: 0, echo_us: 0, final_digest: None });
        }
    }
    let header = header.unwrap();
    eprintln!("relay: {} players joined; game of {} ticks at {} Hz", o.players, o.ticks, o.hz);
    let mut recorder = match &o.record {
        Some(path) => Some(Recorder::new(std::io::BufWriter::new(std::fs::File::create(path)?), &Header { checkpoint_every: header.checkpoint_every, ..header.clone() })?),
        None => None,
    };

    let period = Duration::from_secs_f64(1.0 / o.hz);
    let mut bundles: Vec<TickBundle> = Vec::new();
    let mut digests: BTreeMap<u64, BTreeMap<u32, u128>> = BTreeMap::new();
    let (mut desync, mut checked, mut commands, mut late) = (0u64, 0u64, 0u64, 0u64);
    // Clients resend their recent digests: ticks up to here are done.
    let mut compared_through: Option<u64> = None;
    let mut next_close = Instant::now();
    let started = Instant::now();
    let mut ended_at: Option<Instant> = None;
    loop {
        // Receive everything waiting.
        while let Ok((n, from)) = socket.recv_from(&mut buf) {
            let Ok(Packet::Client { player, inputs, want, digests: ds, sent_us }) = Packet::decode(&buf[..n]) else { continue };
            let Some(peer) = peers.get_mut(&player) else { continue };
            peer.addr = from;
            peer.want = peer.want.max(want);
            peer.echo_us = sent_us;
            for (tick, cmds) in inputs {
                if tick > peer.taken || (peer.taken == 0 && tick == 0) {
                    peer.pending.entry(tick).or_insert(cmds);
                }
            }
            for (tick, d) in ds {
                if tick == o.ticks {
                    peer.final_digest = Some(d);
                }
                if compared_through.is_none_or(|c| tick > c) {
                    digests.entry(tick).or_default().insert(player, d);
                }
            }
        }
        // Compare digests every player has reported.
        while let Some((&tick, by_player)) = digests.first_key_value() {
            if by_player.len() < o.players as usize {
                break;
            }
            let first = *by_player.values().next().unwrap();
            if by_player.values().any(|d| *d != first) {
                if desync == 0 {
                    eprintln!("DESYNC at tick {tick}: {:?}", by_player.iter().map(|(p, d)| format!("player {p}: {:016x}", *d as u64)).collect::<Vec<_>>());
                }
                desync = if desync == 0 { tick } else { desync };
            } else if let Some(r) = recorder.as_mut() {
                r.checkpoint(&Checkpoint { tick, total: first, parts: Vec::new() })?;
            }
            checked += 1;
            compared_through = Some(tick);
            digests.pop_first();
        }
        // Close the next tick's bundle on time.
        let now = Instant::now();
        if (bundles.len() as u64) < o.ticks && now >= next_close {
            let tick = bundles.len() as u64;
            let mut bundle = TickBundle::new(tick);
            for (player, peer) in peers.iter_mut() {
                let due: Vec<u64> = peer.pending.range(..=tick).map(|(t, _)| *t).collect();
                for t in due {
                    let cmds = peer.pending.remove(&t).unwrap();
                    late += (t < tick) as u64;
                    bundle.commands.extend(cmds.into_iter().map(|c| Command { player: *player, ..c }));
                    peer.taken = peer.taken.max(t);
                }
            }
            commands += bundle.commands.len() as u64;
            if let Some(r) = recorder.as_mut() {
                r.bundle(&bundle)?;
            }
            bundles.push(bundle);
            next_close += period;
            if now > next_close + period * 4 {
                next_close = now;
            }
        }
        let end = if bundles.len() as u64 >= o.ticks { o.ticks } else { 0 };
        if end > 0 && ended_at.is_none() {
            ended_at = Some(Instant::now());
        }
        // Send each peer the bundles it needs.
        for peer in peers.values() {
            let mut out = Vec::new();
            let mut bytes = 0;
            for b in bundles.iter().skip(peer.want as usize) {
                let size = 12 + b.commands.iter().map(|c| 16 + c.bytes.len()).sum::<usize>();
                if bytes + size > ae_net::MAX_BUNDLE_BYTES && !out.is_empty() {
                    break;
                }
                bytes += size;
                out.push(b.clone());
            }
            net.send(peer.addr, Packet::Relay { bundles: out, taken: peer.taken, echo_us: peer.echo_us, end, desync }.encode());
        }
        net.flush(&socket)?;
        let finals = peers.values().filter(|p| p.final_digest.is_some()).count();
        if end > 0 && (finals == peers.len() || ended_at.is_some_and(|t| t.elapsed() > Duration::from_secs(20))) {
            // Keep telling stragglers the game ended for a moment.
            if finals == peers.len() {
                for _ in 0..20 {
                    for peer in peers.values() {
                        net.send(peer.addr, Packet::Relay { bundles: Vec::new(), taken: peer.taken, echo_us: peer.echo_us, end, desync }.encode());
                    }
                    net.flush(&socket)?;
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            break;
        }
        std::thread::sleep(Duration::from_micros(500));
    }
    let finals: Vec<Option<u128>> = peers.values().map(|p| p.final_digest).collect();
    let agree = finals.iter().all(|d| d.is_some() && *d == finals[0]);
    if let Some(r) = recorder {
        r.finish(o.ticks)?;
    }
    let wall = started.elapsed().as_secs_f64();
    println!(
        "relay: {} ticks in {wall:.1} s ({:.1} Hz), {} players, {commands} commands ({late} taken late), {checked} digest rounds compared, {}; sent {} datagrams, dropped {}; final digests {}",
        bundles.len(),
        bundles.len() as f64 / wall,
        peers.len(),
        if desync == 0 { "in sync".to_string() } else { format!("DESYNC first at tick {desync}") },
        net.sent,
        net.dropped,
        if agree { format!("agree ({:016x})", finals[0].unwrap() as u64) } else { format!("DIFFER {finals:x?}") },
    );
    let _ = micros(epoch);
    Ok(desync == 0 && agree)
}
