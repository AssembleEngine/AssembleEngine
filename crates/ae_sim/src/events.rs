//! Event streams: typed, fixed-size rows emitted by systems.
//!
//! Emitted events are recorded under the emitting range's command key and
//! merged in key order when the layer ends, like structural commands, so a
//! stream's contents never depend on the worker count.
//!
//! Each reader of a stream has a cursor. When a reader runs it gets every
//! event appended since it last ran, then its cursor moves to the end. With
//! the scheduler's ordering (an emitter and a reader of one stream never
//! share a layer) a reader placed after the emitter sees the same tick's
//! events; one placed before it sees them on its next run. Either way every
//! reader sees every event exactly once, also when it runs less often than
//! every tick. Events all live readers have seen are dropped at tick end.

use crate::command::Emitted;
use anyhow::{Result, ensure};
use std::collections::BTreeMap;

struct Stream {
    size: usize,
    /// Sequence number of the first event still held.
    first: u64,
    /// Packed event bytes from `first` on.
    bytes: Vec<u8>,
    /// Reader (system index) -> sequence number of its next unseen event.
    cursors: BTreeMap<usize, u64>,
}

impl Stream {
    fn end(&self) -> u64 {
        self.first + (self.bytes.len() / self.size) as u64
    }
}

#[derive(Default)]
pub struct EventStore {
    streams: BTreeMap<u64, Stream>,
    /// Readers that will never run again (cadence `Once` after tick 0).
    retired: Vec<usize>,
}

impl EventStore {
    pub fn declare(&mut self, stream: u64, size: usize) {
        self.streams.entry(stream).or_insert(Stream { size, first: 0, bytes: Vec::new(), cursors: BTreeMap::new() });
    }

    pub fn add_reader(&mut self, stream: u64, reader: usize) -> Result<()> {
        let s = self.streams.get_mut(&stream).ok_or_else(|| anyhow::anyhow!("unknown event stream {stream:#x}"))?;
        s.cursors.insert(reader, 0);
        Ok(())
    }

    pub fn retire(&mut self, reader: usize) {
        if !self.retired.contains(&reader) {
            self.retired.push(reader);
        }
    }

    /// Append a layer's events, already merged in key order.
    pub fn append(&mut self, events: &[Emitted]) -> Result<()> {
        for (_, stream, bytes) in events {
            let s = self.streams.get_mut(stream).ok_or_else(|| anyhow::anyhow!("unknown event stream {stream:#x}"))?;
            ensure!(bytes.len() == s.size, "event of {} bytes in a stream of {}", bytes.len(), s.size);
            if !s.cursors.is_empty() {
                s.bytes.extend_from_slice(bytes);
            }
        }
        Ok(())
    }

    /// Events `reader` has not seen yet, packed, and their count.
    pub fn unseen(&self, stream: u64, reader: usize) -> (&[u8], u32) {
        let s = &self.streams[&stream];
        let from = s.cursors[&reader];
        let skip = ((from - s.first) as usize) * s.size;
        let bytes = &s.bytes[skip..];
        (bytes, (bytes.len() / s.size) as u32)
    }

    pub fn mark_seen(&mut self, stream: u64, reader: usize) {
        let s = self.streams.get_mut(&stream).unwrap();
        let end = s.end();
        s.cursors.insert(reader, end);
    }

    /// Drop events every reader that may still run has seen.
    pub fn prune(&mut self) {
        for s in self.streams.values_mut() {
            let live = s.cursors.iter().filter(|(r, _)| !self.retired.contains(r)).map(|(_, &c)| c);
            let keep_from = live.min().unwrap_or(s.end());
            let drop = ((keep_from - s.first) as usize) * s.size;
            s.bytes.drain(..drop);
            s.first = keep_from;
        }
    }

    /// Hash of everything held: per stream, the reader cursors and the
    /// unpruned events. Part of the world digest (events cross ticks).
    pub fn digest(&self) -> crate::digest::StateHash {
        let mut h = crate::digest::StateHash::new();
        for (&id, s) in &self.streams {
            let mut row = Vec::with_capacity(16 + s.cursors.len() * 16 + s.bytes.len());
            row.extend_from_slice(&s.first.to_le_bytes());
            for (&reader, &cursor) in &s.cursors {
                row.extend_from_slice(&(reader as u64).to_le_bytes());
                row.extend_from_slice(&cursor.to_le_bytes());
            }
            row.extend_from_slice(&s.bytes);
            h.add_row(u64::MAX, id, &row);
        }
        h
    }

    /// Events held, all streams.
    pub fn held(&self) -> usize {
        self.streams.values().map(|s| s.bytes.len() / s.size).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CommandKey;

    fn ev(v: u8) -> Emitted {
        (CommandKey { system: 0, range: 0, seq: v as u32 }, 7, vec![v, 0])
    }

    #[test]
    fn every_reader_sees_each_event_once() {
        let mut store = EventStore::default();
        store.declare(7, 2);
        store.add_reader(7, 1).unwrap();
        store.add_reader(7, 2).unwrap();
        store.append(&[ev(1), ev(2)]).unwrap();
        assert_eq!(store.unseen(7, 1), (&[1, 0, 2, 0][..], 2));
        store.mark_seen(7, 1);
        store.prune();
        assert_eq!(store.held(), 2, "reader 2 has not seen them yet");
        store.append(&[ev(3)]).unwrap();
        assert_eq!(store.unseen(7, 1), (&[3, 0][..], 1));
        assert_eq!(store.unseen(7, 2).1, 3);
        store.mark_seen(7, 2);
        store.prune();
        assert_eq!(store.held(), 1);
        store.retire(1);
        store.prune();
        assert_eq!(store.held(), 0, "a retired reader does not hold events");
    }

    #[test]
    fn unread_streams_hold_nothing() {
        let mut store = EventStore::default();
        store.declare(7, 2);
        store.append(&[ev(1)]).unwrap();
        assert_eq!(store.held(), 0);
        assert!(store.append(&[(CommandKey { system: 0, range: 0, seq: 0 }, 7, vec![1])]).is_err(), "wrong size");
    }
}
