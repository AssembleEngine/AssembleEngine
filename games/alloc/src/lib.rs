//! Allocation probe: guest heap use from parallel systems.
//!
//! Every sandbox system works on the stack. Real game code uses `Vec`,
//! `BTreeMap`, `String` and boxes, from systems that run on several worker
//! instances at once over one shared memory and one heap (wasi-libc's
//! malloc, behind its lock). This game exercises that:
//!
//! * `lists` (parallel): per row, a `Vec` grown by push (realloc), sorted,
//!   bucketed into a `BTreeMap`, formatted into a `String` and parsed back.
//!   Each result is also computed on the stack; a mismatch counts as bad.
//! * `blocks` (parallel, same layer as `lists`): per row, a heap block that
//!   lives until the row's next run, so it is usually freed by a different
//!   worker instance than the one that allocated it. One row in 128 takes
//!   64-256 KiB, so the heap grows (`memory.grow`) from several instances at
//!   once. Contents are checked before every free.
//!
//! The test reads the `bad` counts and the held block sizes from the host.
//!
//! Game parameter: the number of probe rows.

use ae_sdk::{Component, Run, Spawn, schedule, system};
use std::collections::BTreeMap;

#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Lists {
    pub checksum: u64,
    pub len: u32,
    pub bad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Blocks {
    pub checksum: u64,
    /// Guest address of the block held until the next run (0: none). An
    /// address depends on allocation order, so it is not hashed.
    #[field(unhashed)]
    pub held: u32,
    #[field(unhashed)]
    pub held_len: u32,
    pub held_seed: u32,
    pub bad: u32,
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn pattern(seed: u32, j: usize) -> u8 {
    (seed.wrapping_mul(j as u32 | 1).wrapping_add(j as u32) >> 11) as u8
}

const BIG_MIN: usize = 64 << 10;

#[system(once, serial)]
fn setup(run: &Run) {
    for _ in 0..run.param() {
        Spawn::new()
            .with(&Lists { checksum: 0, len: 0, bad: 0 })
            .with(&Blocks { checksum: 0, held: 0, held_len: 0, held_seed: 0, bad: 0 })
            .submit();
    }
}

#[system(write(Lists))]
fn lists(run: &Run) {
    let (rows, ids) = (run.write::<Lists>(), run.entities());
    for i in run.rows() {
        let k = mix(ids[i].0 ^ run.tick() << 40);
        let len = 1 + (k % 400) as usize;
        // Stack results.
        let (mut sum, mut xor) = (0u64, 0u64);
        // Heap: grown one push at a time, so it reallocates repeatedly.
        let mut v: Vec<u64> = Vec::new();
        for j in 0..len as u64 {
            let x = mix(k.wrapping_add(j));
            v.push(x);
            sum = sum.wrapping_add(x);
            xor ^= x;
        }
        v.sort_unstable();
        let mut ok = v.len() == len && v.windows(2).all(|w| w[0] <= w[1]);
        ok &= v.iter().fold(0u64, |a, &x| a.wrapping_add(x)) == sum;
        ok &= v.iter().fold(0u64, |a, &x| a ^ x) == xor;
        let mut buckets: BTreeMap<u32, u32> = BTreeMap::new();
        for &x in &v {
            *buckets.entry((x % 97) as u32).or_insert(0) += 1;
        }
        ok &= buckets.values().sum::<u32>() == len as u32;
        let text = format!("{sum:x}-{len}");
        let (hex, n) = text.split_once('-').unwrap();
        ok &= u64::from_str_radix(hex, 16) == Ok(sum) && n.parse::<usize>() == Ok(len);
        // Order-sensitive: depends on the sort and the map's iteration.
        let weighted = v.iter().enumerate().fold(0u64, |a, (j, &x)| a.wrapping_add(x.wrapping_mul(j as u64 + 1)));
        let spread = buckets.iter().fold(0u64, |a, (&b, &c)| mix(a ^ (b as u64) << 32 ^ c as u64));
        let r = &mut rows[i];
        r.checksum = mix(r.checksum ^ weighted ^ spread);
        r.len = len as u32;
        r.bad += !ok as u32;
    }
}

#[system(write(Blocks))]
fn blocks(run: &Run) {
    let (rows, ids) = (run.write::<Blocks>(), run.entities());
    for i in run.rows() {
        let r = &mut rows[i];
        if r.held != 0 {
            // Allocated on an earlier tick, usually by another instance.
            let block = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.held as usize as *mut u8, r.held_len as usize)) };
            let mut ok = true;
            let mut sum = 0u64;
            for (j, &b) in block.iter().enumerate() {
                ok &= b == pattern(r.held_seed, j);
                sum = sum.wrapping_add(b as u64);
            }
            r.bad += !ok as u32;
            r.checksum = mix(r.checksum ^ sum ^ (block.len() as u64) << 40);
            drop(block);
        }
        let k = mix(ids[i].0 ^ run.tick() << 40 ^ 0xB10C);
        let len = if k % 128 == 0 { BIG_MIN + (k >> 8) as usize % (192 << 10) } else { 16 + (k >> 8) as usize % 4000 };
        let seed = (k >> 32) as u32;
        let block: Box<[u8]> = (0..len).map(|j| pattern(seed, j)).collect();
        r.held_len = len as u32;
        r.held_seed = seed;
        r.held = Box::into_raw(block) as *mut u8 as usize as u32;
    }
}

schedule!(setup, lists, blocks);
