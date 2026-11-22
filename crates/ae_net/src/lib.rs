//! Relay-paced lockstep (M5), over UDP.
//!
//! The relay owns time: every tick it closes one [`TickBundle`] from the
//! player inputs it has, and every client runs tick `t` only with bundle
//! `t`. A client never waits for another client, only for the relay; an
//! input that reaches the relay late goes into a later bundle. Everyone
//! runs the same bundles in the same order, so the worlds stay equal; the
//! clients report digests and the relay compares them.
//!
//! Packets are idempotent so loss needs no handshakes: a client resends
//! every input the relay has not taken yet and says which bundle it needs
//! next; the relay resends the bundles from there. Both sides send every
//! frame.
//!
//! * [`Packet::Hello`]: a client joins as `player` with its world's header
//!   (every client must build the same world).
//! * [`Packet::Client`]: inputs `(intended tick, commands)` not yet taken,
//!   the next bundle wanted, recent `(tick, digest)` pairs, a timestamp for
//!   the round trip.
//! * [`Packet::Relay`]: bundles from the one wanted (as many as fit), the
//!   newest input tick taken from this player, the echoed timestamp, the
//!   tick the game ends at (0 while running), and the first desynced tick
//!   seen (0: none).
//!
//! [`Emulator`] delays, jitters and drops outgoing datagrams (seeded), to
//! test lockstep under bad networks on loopback.

use ae_replay::Header;
use ae_sim::input::{Command, TickBundle};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

/// Bundle bytes per relay datagram, at most (UDP on loopback takes 64 KiB).
pub const MAX_BUNDLE_BYTES: usize = 48 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub enum Packet {
    Hello {
        player: u32,
        header: Header,
    },
    Client {
        player: u32,
        inputs: Vec<(u64, Vec<Command>)>,
        want: u64,
        digests: Vec<(u64, u128)>,
        sent_us: u64,
    },
    Relay {
        bundles: Vec<TickBundle>,
        taken: u64,
        echo_us: u64,
        end: u64,
        desync: u64,
    },
}

struct W(Vec<u8>);
impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u128(&mut self, v: u128) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.0.extend_from_slice(s.as_bytes());
    }
}

struct R<'a>(&'a [u8], usize);
impl R<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let s = self.0.get(self.1..self.1 + n).context("packet truncated")?;
        self.1 += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u128(&mut self) -> Result<u128> {
        Ok(u128::from_le_bytes(self.take(16)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        ensure!(n <= 4096, "a string of {n} bytes");
        Ok(String::from_utf8(self.take(n)?.to_vec())?)
    }
    fn count(&mut self, max: usize) -> Result<usize> {
        let n = self.u32()? as usize;
        ensure!(n <= max, "{n} items (at most {max})");
        Ok(n)
    }
}

const HELLO: u8 = 1;
const CLIENT: u8 = 2;
const RELAY: u8 = 3;

impl Packet {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        match self {
            Packet::Hello { player, header } => {
                w.u8(HELLO);
                w.u32(*player);
                w.str(&header.engine);
                w.u128(header.module_hash);
                w.u128(header.schema_hash);
                w.u64(header.param);
                w.u32(header.config.len() as u32);
                for (k, v) in &header.config {
                    w.str(k);
                    w.str(v);
                }
                w.u64(header.checkpoint_every);
            }
            Packet::Client {
                player,
                inputs,
                want,
                digests,
                sent_us,
            } => {
                w.u8(CLIENT);
                w.u32(*player);
                w.u64(*want);
                w.u64(*sent_us);
                w.u32(inputs.len() as u32);
                for (tick, commands) in inputs {
                    let mut b = Vec::new();
                    TickBundle {
                        tick: *tick,
                        commands: commands.clone(),
                    }
                    .encode(&mut b);
                    w.0.extend_from_slice(&b);
                }
                w.u32(digests.len() as u32);
                for (t, d) in digests {
                    w.u64(*t);
                    w.u128(*d);
                }
            }
            Packet::Relay {
                bundles,
                taken,
                echo_us,
                end,
                desync,
            } => {
                w.u8(RELAY);
                w.u64(*taken);
                w.u64(*echo_us);
                w.u64(*end);
                w.u64(*desync);
                w.u32(bundles.len() as u32);
                for b in bundles {
                    b.encode(&mut w.0);
                }
            }
        }
        w.0
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = R(bytes, 0);
        let p = match r.u8()? {
            HELLO => {
                let player = r.u32()?;
                let engine = r.str()?;
                let (module_hash, schema_hash, param) = (r.u128()?, r.u128()?, r.u64()?);
                let n = r.count(64)?;
                let mut config = BTreeMap::new();
                for _ in 0..n {
                    config.insert(r.str()?, r.str()?);
                }
                let checkpoint_every = r.u64()?;
                Packet::Hello {
                    player,
                    header: Header {
                        engine,
                        module_hash,
                        schema_hash,
                        param,
                        config,
                        checkpoint_every,
                    },
                }
            }
            CLIENT => {
                let (player, want, sent_us) = (r.u32()?, r.u64()?, r.u64()?);
                let n = r.count(4096)?;
                let mut inputs = Vec::with_capacity(n);
                for _ in 0..n {
                    let (b, used) = TickBundle::decode(&r.0[r.1..])?;
                    r.1 += used;
                    inputs.push((b.tick, b.commands));
                }
                let n = r.count(4096)?;
                let mut digests = Vec::with_capacity(n);
                for _ in 0..n {
                    digests.push((r.u64()?, r.u128()?));
                }
                Packet::Client {
                    player,
                    inputs,
                    want,
                    digests,
                    sent_us,
                }
            }
            RELAY => {
                let (taken, echo_us, end, desync) = (r.u64()?, r.u64()?, r.u64()?, r.u64()?);
                let n = r.count(1 << 16)?;
                let mut bundles = Vec::with_capacity(n.min(1024));
                for _ in 0..n {
                    let (b, used) = TickBundle::decode(&r.0[r.1..])?;
                    r.1 += used;
                    bundles.push(b);
                }
                Packet::Relay {
                    bundles,
                    taken,
                    echo_us,
                    end,
                    desync,
                }
            }
            k => bail!("unknown packet kind {k}"),
        };
        ensure!(r.1 == bytes.len(), "{} spare bytes after a packet", bytes.len() - r.1);
        Ok(p)
    }
}

/// A seeded bad network on the sending side: every datagram is delayed by
/// `latency` plus up to `jitter` (so they may arrive out of order), and
/// dropped with probability `loss`.
pub struct Emulator {
    pub latency: Duration,
    pub jitter: Duration,
    pub loss: f64,
    rng: u64,
    queue: Vec<(Instant, SocketAddr, Vec<u8>)>,
    pub sent: u64,
    pub dropped: u64,
}

impl Emulator {
    pub fn new(latency_ms: u64, jitter_ms: u64, loss: f64, seed: u64) -> Self {
        Self {
            latency: Duration::from_millis(latency_ms),
            jitter: Duration::from_millis(jitter_ms),
            loss,
            rng: seed | 1,
            queue: Vec::new(),
            sent: 0,
            dropped: 0,
        }
    }

    fn next(&mut self) -> f64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Queue a datagram (or drop it).
    pub fn send(&mut self, to: SocketAddr, bytes: Vec<u8>) {
        self.sent += 1;
        if self.next() < self.loss {
            self.dropped += 1;
            return;
        }
        let delay = self.latency + self.jitter.mul_f64(self.next());
        self.queue.push((Instant::now() + delay, to, bytes));
    }

    /// Send what is due.
    pub fn flush(&mut self, socket: &UdpSocket) -> Result<()> {
        let now = Instant::now();
        let mut i = 0;
        while i < self.queue.len() {
            if self.queue[i].0 <= now {
                let (_, to, bytes) = self.queue.swap_remove(i);
                // A full buffer drops the datagram, like a real network.
                let _ = socket.send_to(&bytes, to);
            } else {
                i += 1;
            }
        }
        Ok(())
    }
}

/// Microseconds since `epoch` (packet timestamps).
pub fn micros(epoch: Instant) -> u64 {
    epoch.elapsed().as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packets_round_trip() {
        let header = Header {
            engine: "0.0.1".into(),
            module_hash: 5,
            schema_hash: 6,
            param: 7,
            config: [("map".to_string(), "water".to_string())].into(),
            checkpoint_every: 30,
        };
        let cmd = Command {
            player: 2,
            stream: 4,
            bytes: vec![1, 2, 3],
        };
        for p in [
            Packet::Hello { player: 3, header },
            Packet::Client {
                player: 1,
                inputs: vec![(9, vec![cmd.clone()]), (10, vec![])],
                want: 8,
                digests: vec![(30, 0xABC)],
                sent_us: 123,
            },
            Packet::Relay {
                bundles: vec![
                    TickBundle {
                        tick: 8,
                        commands: vec![cmd],
                    },
                    TickBundle::new(9),
                ],
                taken: 7,
                echo_us: 123,
                end: 0,
                desync: 0,
            },
        ] {
            let bytes = p.encode();
            assert_eq!(Packet::decode(&bytes).unwrap(), p);
            assert!(Packet::decode(&bytes[..bytes.len() - 1]).is_err());
        }
    }

    #[test]
    fn the_emulator_drops_about_its_loss_rate() {
        let mut e = Emulator::new(0, 0, 0.1, 42);
        let to: SocketAddr = "127.0.0.1:9".parse().unwrap();
        for _ in 0..10_000 {
            e.send(to, vec![0]);
        }
        assert!((800..1200).contains(&e.dropped), "{} dropped", e.dropped);
    }
}
