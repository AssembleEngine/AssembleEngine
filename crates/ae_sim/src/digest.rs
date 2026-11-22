//! Order-independent world digest (lifted from `ecs_sim/src/engine/hash.rs`).
//!
//! Every `(entity, component, hashed fields)` row is hashed with xxh3-128 and
//! the row hashes are summed mod 2^128, so storage order (tables, rows,
//! worker count) cannot influence the result and nothing needs sorting.
//! Only fields flagged `hashed` in the schema are read, field by field in
//! stable-id order, so padding bytes never reach the digest.

/// Wrapping sum of row hashes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StateHash {
    sum: u128,
    rows: u64,
}

/// Rows up to this size are hashed from a stack buffer.
const ROW_INLINE: usize = 112;

impl StateHash {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_row(&mut self, entity: u64, component: u64, bytes: &[u8]) {
        let row = if bytes.len() <= ROW_INLINE {
            let mut buffer = [0u8; 16 + ROW_INLINE];
            buffer[..8].copy_from_slice(&entity.to_le_bytes());
            buffer[8..16].copy_from_slice(&component.to_le_bytes());
            buffer[16..16 + bytes.len()].copy_from_slice(bytes);
            xxhash_rust::xxh3::xxh3_128(&buffer[..16 + bytes.len()])
        } else {
            let mut hasher = xxhash_rust::xxh3::Xxh3::new();
            hasher.update(&entity.to_le_bytes());
            hasher.update(&component.to_le_bytes());
            hasher.update(bytes);
            hasher.digest128()
        };
        self.sum = self.sum.wrapping_add(row);
        self.rows += 1;
    }

    pub fn merge(&mut self, other: &StateHash) {
        self.sum = self.sum.wrapping_add(other.sum);
        self.rows += other.rows;
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    pub fn finish(&self) -> u128 {
        self.sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(rows: &[(u64, u64, &[u8])]) -> u128 {
        let mut state = StateHash::new();
        for &(entity, component, bytes) in rows {
            state.add_row(entity, component, bytes);
        }
        state.finish()
    }

    #[test]
    fn order_independent_and_sees_every_field() {
        let rows: [(u64, u64, &[u8]); 3] = [(1, 10, &[1, 2, 3, 4]), (2, 10, &[5, 6, 7, 8]), (1, 11, &[9; 8])];
        let base = hash(&rows);
        assert_eq!(hash(&[rows[2], rows[0], rows[1]]), base, "order independent");
        assert_ne!(
            hash(&[(1, 10, &[5, 6, 7, 8]), (2, 10, &[1, 2, 3, 4]), rows[2]]),
            base,
            "swapped data between entities"
        );
        assert_ne!(hash(&[(1, 12, &[1, 2, 3, 4]), rows[1], rows[2]]), base, "component id");
        assert_ne!(hash(&[(1, 10, &[1, 2, 3, 5]), rows[1], rows[2]]), base, "one data bit");
        assert_ne!(hash(&rows[..2]), base, "a missing row");
        assert_ne!(
            hash(&[rows[0], rows[0], rows[1], rows[2]]),
            base,
            "a duplicated row does not cancel"
        );
        let big = [7u8; 200];
        let mut streamed = StateHash::new();
        streamed.add_row(3, 4, &big);
        let mut buffer = Vec::new();
        buffer.extend_from_slice(&3u64.to_le_bytes());
        buffer.extend_from_slice(&4u64.to_le_bytes());
        buffer.extend_from_slice(&big);
        assert_eq!(streamed.finish(), xxhash_rust::xxh3::xxh3_128(&buffer));
    }
}
