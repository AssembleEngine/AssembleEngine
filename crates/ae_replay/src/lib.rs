//! Replays (`.aerep`): what a game needs to be played again exactly.
//!
//! * A [`Header`]: engine version, hashes of the game module and of its
//!   schema, the game parameter, the host plugins' configuration (map,
//!   size, teams, as `key=value` text), and the checkpoint interval. A
//!   replay only plays on a world built from the same; [`Header::check`]
//!   names the first difference.
//! * The command bundles ([`ae_sim::input::TickBundle`], the same type the
//!   network carries), only for ticks that have commands.
//! * [`Checkpoint`]s: the world digest every `checkpoint_every` ticks, with
//!   its parts (per component and per host service), so a mismatch points
//!   at what diverged.
//! * The tick the recording ended at.
//!
//! Layout: magic `AEREP\0\0\x01`, then records `kind: u8, length: u32,
//! payload`; kind 1 header, 2 bundle, 3 checkpoint, 4 end. Integers are
//! little-endian; strings are `u32` length + UTF-8.

use ae_sim::input::TickBundle;
use ae_sim::runtime::Digest;
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"AEREP\0\0\x01";
const HEADER: u8 = 1;
const BUNDLE: u8 = 2;
const CHECKPOINT: u8 = 3;
const END: u8 = 4;

/// Everything that shapes the world a replay plays on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub engine: String,
    /// xxh3-128 of the game module's bytes.
    pub module_hash: u128,
    /// xxh3-128 of the module's schema section.
    pub schema_hash: u128,
    pub param: u64,
    /// Host plugins' configuration, e.g. `map=water`, `map_cells=2560`.
    pub config: BTreeMap<String, String>,
    pub checkpoint_every: u64,
}

impl Header {
    /// The difference that keeps this replay from playing on `world`'s
    /// header, if any.
    pub fn check(&self, world: &Header) -> Result<()> {
        ensure!(self.engine == world.engine, "the replay was recorded with engine {}, this is {}", self.engine, world.engine);
        ensure!(self.module_hash == world.module_hash, "the game module differs (replay {:032x}, here {:032x})", self.module_hash, world.module_hash);
        ensure!(self.schema_hash == world.schema_hash, "the game schema differs");
        ensure!(self.param == world.param, "game parameter {:#x} in the replay, {:#x} here", self.param, world.param);
        for (k, v) in &self.config {
            match world.config.get(k) {
                Some(here) if here == v => {}
                Some(here) => bail!("{k} = {v} in the replay, {here} here"),
                None => bail!("{k} = {v} in the replay, unset here"),
            }
        }
        if let Some(k) = world.config.keys().find(|k| !self.config.contains_key(*k)) {
            bail!("{k} is set here but not in the replay");
        }
        Ok(())
    }
}

pub fn hash(bytes: &[u8]) -> u128 {
    xxhash_rust::xxh3::xxh3_128(bytes)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    /// The world's tick when the digest was taken (after `tick` ticks).
    pub tick: u64,
    pub total: u128,
    /// `(part, hash, rows)`, as [`Digest::parts`].
    pub parts: Vec<(String, u128, u64)>,
}

impl Checkpoint {
    pub fn of(tick: u64, d: &Digest) -> Self {
        Self { tick, total: d.total, parts: d.parts.clone() }
    }

    /// Parts whose hash or row count differ from `other`'s (by name; a part
    /// on one side only counts too).
    pub fn differing(&self, other: &Checkpoint) -> Vec<String> {
        let theirs: BTreeMap<&str, (u128, u64)> = other.parts.iter().map(|(n, h, r)| (n.as_str(), (*h, *r))).collect();
        let ours: BTreeMap<&str, (u128, u64)> = self.parts.iter().map(|(n, h, r)| (n.as_str(), (*h, *r))).collect();
        let mut out: Vec<String> = ours.iter().filter(|(n, v)| theirs.get(*n) != Some(v)).map(|(n, _)| n.to_string()).collect();
        out.extend(theirs.keys().filter(|n| !ours.contains_key(*n)).map(|n| n.to_string()));
        out
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Replay {
    pub header: Option<Header>,
    /// By tick.
    pub bundles: BTreeMap<u64, TickBundle>,
    pub checkpoints: Vec<Checkpoint>,
    /// The world's tick when recording ended.
    pub end: u64,
}

impl Replay {
    /// The bundle for `tick` (empty when the replay has none).
    pub fn bundle(&self, tick: u64) -> TickBundle {
        self.bundles.get(&tick).cloned().unwrap_or_else(|| TickBundle::new(tick))
    }

    pub fn header(&self) -> Result<&Header> {
        self.header.as_ref().context("the replay has no header")
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::decode(&bytes).with_context(|| format!("decode {}", path.display()))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.starts_with(MAGIC), "not a replay (bad magic)");
        let mut r = Replay::default();
        let mut at = MAGIC.len();
        let mut ended = false;
        while at < bytes.len() {
            ensure!(!ended, "records after the end record");
            ensure!(at + 5 <= bytes.len(), "truncated record header at byte {at}");
            let kind = bytes[at];
            let len = u32::from_le_bytes(bytes[at + 1..at + 5].try_into().unwrap()) as usize;
            let payload = bytes.get(at + 5..at + 5 + len).with_context(|| format!("truncated record at byte {at}"))?;
            at += 5 + len;
            let mut p = Reader { bytes: payload, at: 0 };
            match kind {
                HEADER => {
                    ensure!(r.header.is_none(), "two headers");
                    let engine = p.string()?;
                    let (module_hash, schema_hash, param) = (p.u128()?, p.u128()?, p.u64()?);
                    let n = p.u32()?;
                    let mut config = BTreeMap::new();
                    for _ in 0..n {
                        config.insert(p.string()?, p.string()?);
                    }
                    let checkpoint_every = p.u64()?;
                    r.header = Some(Header { engine, module_hash, schema_hash, param, config, checkpoint_every });
                }
                BUNDLE => {
                    let (b, used) = TickBundle::decode(payload)?;
                    ensure!(used == payload.len(), "bundle record with {} spare bytes", payload.len() - used);
                    ensure!(r.bundles.last_key_value().is_none_or(|(t, _)| *t < b.tick), "bundles out of tick order at tick {}", b.tick);
                    r.bundles.insert(b.tick, b);
                }
                CHECKPOINT => {
                    let (tick, total) = (p.u64()?, p.u128()?);
                    let n = p.u32()?;
                    let mut parts = Vec::with_capacity(n as usize);
                    for _ in 0..n {
                        parts.push((p.string()?, p.u128()?, p.u64()?));
                    }
                    r.checkpoints.push(Checkpoint { tick, total, parts });
                }
                END => {
                    r.end = p.u64()?;
                    ended = true;
                }
                other => bail!("unknown record kind {other}"),
            }
            ensure!(p.at == payload.len() || kind == BUNDLE, "record kind {kind} has {} spare bytes", payload.len() - p.at);
        }
        ensure!(r.header.is_some(), "the replay has no header");
        ensure!(ended, "the replay has no end record (recording stopped early?)");
        Ok(r)
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let s = self.bytes.get(self.at..self.at + n).context("record truncated")?;
        self.at += n;
        Ok(s)
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
    fn string(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        ensure!(n <= 1 << 16, "a string of {n} bytes");
        Ok(String::from_utf8(self.take(n)?.to_vec())?)
    }
}

/// Writes a replay as the game runs: header first, then bundles and
/// checkpoints in tick order, then [`Recorder::finish`].
pub struct Recorder<W: Write> {
    out: W,
}

fn string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

impl<W: Write> Recorder<W> {
    pub fn new(mut out: W, h: &Header) -> Result<Self> {
        out.write_all(MAGIC)?;
        let mut p = Vec::new();
        string(&mut p, &h.engine);
        p.extend_from_slice(&h.module_hash.to_le_bytes());
        p.extend_from_slice(&h.schema_hash.to_le_bytes());
        p.extend_from_slice(&h.param.to_le_bytes());
        p.extend_from_slice(&(h.config.len() as u32).to_le_bytes());
        for (k, v) in &h.config {
            string(&mut p, k);
            string(&mut p, v);
        }
        p.extend_from_slice(&h.checkpoint_every.to_le_bytes());
        let mut r = Self { out };
        r.record(HEADER, &p)?;
        Ok(r)
    }

    fn record(&mut self, kind: u8, payload: &[u8]) -> Result<()> {
        self.out.write_all(&[kind])?;
        self.out.write_all(&(payload.len() as u32).to_le_bytes())?;
        self.out.write_all(payload)?;
        Ok(())
    }

    /// The bundle of one tick (skipped when it has no commands).
    pub fn bundle(&mut self, b: &TickBundle) -> Result<()> {
        if b.commands.is_empty() {
            return Ok(());
        }
        let mut p = Vec::new();
        b.encode(&mut p);
        self.record(BUNDLE, &p)
    }

    pub fn checkpoint(&mut self, c: &Checkpoint) -> Result<()> {
        let mut p = Vec::new();
        p.extend_from_slice(&c.tick.to_le_bytes());
        p.extend_from_slice(&c.total.to_le_bytes());
        p.extend_from_slice(&(c.parts.len() as u32).to_le_bytes());
        for (n, h, rows) in &c.parts {
            string(&mut p, n);
            p.extend_from_slice(&h.to_le_bytes());
            p.extend_from_slice(&rows.to_le_bytes());
        }
        self.record(CHECKPOINT, &p)
    }

    /// End at the world's tick `end`; the writer back.
    pub fn finish(mut self, end: u64) -> Result<W> {
        self.record(END, &end.to_le_bytes())?;
        self.out.flush()?;
        Ok(self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_sim::input::Command;

    fn header() -> Header {
        let mut config = BTreeMap::new();
        config.insert("map".into(), "water".into());
        config.insert("map_cells".into(), "2560".into());
        Header { engine: "0.0.1".into(), module_hash: 1, schema_hash: 2, param: 10_000, config, checkpoint_every: 30 }
    }

    #[test]
    fn replays_round_trip_and_headers_name_differences() {
        let mut rec = Recorder::new(Vec::new(), &header()).unwrap();
        rec.bundle(&TickBundle::new(3)).unwrap(); // empty: not written
        let b = TickBundle { tick: 5, commands: vec![Command { player: 1, stream: 4, bytes: vec![9; 8] }] };
        rec.bundle(&b).unwrap();
        let c = Checkpoint { tick: 30, total: 77, parts: vec![("Body".into(), 5, 100), ("nav".into(), 6, 0)] };
        rec.checkpoint(&c).unwrap();
        let bytes = rec.finish(60).unwrap();
        let r = Replay::decode(&bytes).unwrap();
        assert_eq!(r.header.as_ref(), Some(&header()));
        assert_eq!((r.bundles.len(), r.bundle(5), r.bundle(3).commands.len(), r.end), (1, b, 0, 60));
        assert_eq!(r.checkpoints, vec![c.clone()]);
        assert!(Replay::decode(&bytes[..bytes.len() - 3]).is_err(), "truncated");
        assert!(Replay::decode(&bytes[..bytes.len() - 13]).unwrap_err().to_string().contains("end record"));

        let mut other = header();
        other.config.insert("map_cells".into(), "512".into());
        assert!(header().check(&other).unwrap_err().to_string().contains("map_cells = 2560 in the replay, 512 here"));
        other = header();
        other.param = 3;
        assert!(header().check(&other).unwrap_err().to_string().contains("parameter"));
        assert!(header().check(&header()).is_ok());

        let mut moved = c.clone();
        moved.parts[1].1 = 99;
        moved.parts.push(("Goal".into(), 1, 1));
        assert_eq!(c.differing(&moved), vec!["nav".to_string(), "Goal".to_string()]);
    }
}
