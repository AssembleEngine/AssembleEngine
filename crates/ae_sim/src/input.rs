//! Player commands: the only way anything outside the sim changes it.
//!
//! A [`TickBundle`] holds every command for one tick, in the order every
//! peer applies them (whoever builds the bundle, the relay or a replay,
//! fixes the order; peers never re-sort). A command is one event of a
//! stream the game marks as a command stream (`#[component(event,
//! command)]`), sent by a player: [`crate::runtime::Runtime::tick_with`]
//! appends the bundle's valid commands to their streams before the tick's
//! first layer, so readers see them that tick, and drops invalid ones
//! (unknown or non-command stream, wrong size) the same way on every peer.
//! Commands name entities by their 64-bit sim ids, never by an observer's
//! view ids; a command for an entity that no longer exists is the game's
//! (or service's) to ignore.
//!
//! Bundles encode to bytes ([`TickBundle::encode`]) for replays and the
//! network.

use anyhow::{Context, Result, ensure};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Command {
    pub player: u32,
    /// Event stream (component id).
    pub stream: u64,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TickBundle {
    pub tick: u64,
    pub commands: Vec<Command>,
}

/// Largest command payload accepted when decoding.
pub const MAX_COMMAND_BYTES: usize = 4096;

impl TickBundle {
    pub fn new(tick: u64) -> Self {
        Self {
            tick,
            commands: Vec::new(),
        }
    }

    /// `tick: u64, count: u32`, then per command `player: u32, stream: u64,
    /// len: u32, bytes` (little-endian).
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.tick.to_le_bytes());
        out.extend_from_slice(&(self.commands.len() as u32).to_le_bytes());
        for c in &self.commands {
            out.extend_from_slice(&c.player.to_le_bytes());
            out.extend_from_slice(&c.stream.to_le_bytes());
            out.extend_from_slice(&(c.bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&c.bytes);
        }
    }

    /// Decode one bundle from the front of `bytes`; the bundle and the bytes
    /// it took.
    pub fn decode(bytes: &[u8]) -> Result<(Self, usize)> {
        let mut at = 0;
        let mut take = |n: usize| -> Result<&[u8]> {
            let s = bytes.get(at..at + n).context("bundle truncated")?;
            at += n;
            Ok(s)
        };
        let tick = u64::from_le_bytes(take(8)?.try_into().unwrap());
        let count = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        ensure!(count <= 1 << 20, "{count} commands in one bundle");
        let mut commands = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            let player = u32::from_le_bytes(take(4)?.try_into().unwrap());
            let stream = u64::from_le_bytes(take(8)?.try_into().unwrap());
            let len = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
            ensure!(len <= MAX_COMMAND_BYTES, "a command of {len} bytes");
            commands.push(Command {
                player,
                stream,
                bytes: take(len)?.to_vec(),
            });
        }
        Ok((Self { tick, commands }, at))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundles_round_trip_and_truncation_is_an_error() {
        let b = TickBundle {
            tick: 42,
            commands: vec![
                Command {
                    player: 1,
                    stream: 7,
                    bytes: vec![1, 2, 3],
                },
                Command {
                    player: 0,
                    stream: 9,
                    bytes: vec![],
                },
            ],
        };
        let mut bytes = Vec::new();
        b.encode(&mut bytes);
        bytes.extend_from_slice(&[0xFF; 3]);
        let (back, used) = TickBundle::decode(&bytes).unwrap();
        assert_eq!((back, used), (b, bytes.len() - 3));
        assert!(TickBundle::decode(&bytes[..used - 1]).is_err());
    }
}
