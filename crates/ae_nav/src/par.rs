//! Data-parallel helpers on a persistent thread pool, for per-tick work on
//! the simulation thread (avoidance, collision). Starting scoped threads cost
//! about 130 µs per parallel section with 8 threads, several times per tick;
//! a pool dispatch costs a few microseconds.
//!
//! The simulation loop itself runs on a pool thread (`install`), so these
//! sections start inside the pool instead of being handed over by a thread
//! outside it, which then sleeps and must be woken when the section ends.
//!
//! Results never depend on how the work is split: callers use these only for
//! independent per-item work and combine the outputs in index order.
use crate::thread_load;
use rayon::prelude::*;
use std::sync::Arc;

/// Inputs below this many items run on the caller.
const MIN_PARALLEL_ITEMS: usize = 1024;
/// Smallest piece handed to the pool.
const MIN_PIECE_ITEMS: usize = 256;
/// Pieces per thread, so the pool can balance uneven work (units are stored
/// group by group, and idle units cost far less than moving ones).
const PIECES_PER_THREAD: usize = 4;

#[derive(Clone)]
pub struct Par {
    pool: Option<Arc<rayon::ThreadPool>>,
    threads: usize,
}

impl Par {
    /// A pool of `threads` threads; 0 or 1 runs everything on the caller.
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let pool = (threads > 1)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|i| format!("sim-par-{i}"))
                    .start_handler(|i| thread_load::register(&format!("sim-par-{i}")))
                    .build()
                    .expect("build simulation thread pool")
            })
            .map(Arc::new);
        Self { pool, threads }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Pieces to split `n` items into.
    fn pieces(&self, n: usize) -> usize {
        if self.pool.is_none() || n < MIN_PARALLEL_ITEMS {
            return 1;
        }
        (self.threads * PIECES_PER_THREAD).min(n / MIN_PIECE_ITEMS).max(1)
    }

    /// Run `f` on a pool thread, which then takes the parallel sections it
    /// starts. The caller blocks meanwhile; its thread-load slot goes with
    /// `f`.
    pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(pool) if rayon::current_thread_index().is_none() => {
                let slot = thread_load::current();
                thread_load::idle(|| pool.install(|| thread_load::lend(slot, f)))
            }
            _ => f(),
        }
    }

    /// Run `a` and `b`, in parallel when there is a pool.
    pub fn join<A: Send, B: Send>(&self, a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B + Send) -> (A, B) {
        match &self.pool {
            Some(pool) => run(pool, || {
                rayon::join(
                    || {
                        let _b = thread_load::busy();
                        a()
                    },
                    || {
                        let _b = thread_load::busy();
                        b()
                    },
                )
            }),
            None => (a(), b()),
        }
    }

    /// Sort `items` (unstable). Callers sort by unique keys, so the result
    /// does not depend on the algorithm or the thread count.
    pub fn sort_unstable<T: Ord + Send>(&self, items: &mut [T]) {
        match &self.pool {
            Some(pool) if items.len() >= MIN_PARALLEL_ITEMS => run(pool, || {
                let _b = thread_load::busy();
                items.par_sort_unstable()
            }),
            _ => items.sort_unstable(),
        }
    }

    /// `f` over every item, results in item order.
    pub fn map<T: Sync, R: Send>(&self, items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
        match &self.pool {
            // One busy span per rayon job, not per item.
            Some(pool) if items.len() > 1 => run(pool, || {
                items.par_iter().map_init(thread_load::busy, |_, x| f(x)).collect()
            }),
            _ => items.iter().map(f).collect(),
        }
    }

    /// `f` over contiguous index ranges covering `0..n`, results in range
    /// order.
    pub fn ranges<R: Send>(&self, n: usize, f: impl Fn(usize, usize) -> R + Sync + Send) -> Vec<R> {
        let k = self.pieces(n);
        match &self.pool {
            Some(pool) if k > 1 => {
                let chunk = n.div_ceil(k);
                run(pool, || {
                    (0..k)
                        .into_par_iter()
                        .map(|c| {
                            let _b = thread_load::busy();
                            f((c * chunk).min(n), ((c + 1) * chunk).min(n))
                        })
                        .collect()
                })
            }
            _ => vec![f(0, n)],
        }
    }

    /// `f` over contiguous mutable chunks of `items`, results in chunk order.
    pub fn chunks_mut<T: Send, R: Send>(&self, items: &mut [T], f: impl Fn(&mut [T]) -> R + Sync + Send) -> Vec<R> {
        self.chunks_mut_at(items, |_, c| f(c))
    }

    /// `f(start, chunk)` over contiguous mutable chunks of `items`, where
    /// `start` is the index of the chunk's first item; results in chunk order.
    pub fn chunks_mut_at<T: Send, R: Send>(
        &self,
        items: &mut [T],
        f: impl Fn(usize, &mut [T]) -> R + Sync + Send,
    ) -> Vec<R> {
        let k = self.pieces(items.len());
        match &self.pool {
            Some(pool) if k > 1 => {
                let chunk = items.len().div_ceil(k);
                run(pool, || {
                    items
                        .par_chunks_mut(chunk)
                        .enumerate()
                        .map(|(c, s)| {
                            let _b = thread_load::busy();
                            f(c * chunk, s)
                        })
                        .collect()
                })
            }
            _ => vec![f(0, items)],
        }
    }
}

/// `pool.install(f)`. A caller outside the pool blocks until `f` is done, so
/// it counts as idle meanwhile; a pool thread runs `f` inline.
fn run<R: Send>(pool: &rayon::ThreadPool, f: impl FnOnce() -> R + Send) -> R {
    if rayon::current_thread_index().is_some() {
        pool.install(f)
    } else {
        thread_load::idle(|| pool.install(f))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_preserve_order_for_any_thread_count() {
        let expected: Vec<usize> = (0..5000).collect();
        for t in [1, 2, 3, 8] {
            let par = Par::new(t);
            let got: Vec<usize> = par
                .ranges(5000, |lo, hi| (lo..hi).collect::<Vec<_>>())
                .into_iter()
                .flatten()
                .collect();
            assert_eq!(got, expected);
            let mut items = vec![1u32; 5000];
            let sums = par.chunks_mut(&mut items, |c| {
                for x in c.iter_mut() {
                    *x += 1;
                }
                c.len()
            });
            assert_eq!(sums.iter().sum::<usize>(), 5000);
            assert!(items.iter().all(|&x| x == 2));
        }
    }
}
