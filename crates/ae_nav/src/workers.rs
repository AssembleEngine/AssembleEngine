//! Simple deterministic compute pool. Jobs have stable IDs; results are
//! collected by ID and applied by the simulation at fixed tick boundaries.
//! Urgent jobs (hierarchy repairs) are taken before queued field builds;
//! that changes only when a result is ready, never what it is.
use crate::nav::fields::{build_flow_field, FieldKey, FlowField};
use crate::nav::hierarchy::NavHierarchy;
use crate::terrain::{Terrain, TerrainPatch};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Independent navigation jobs. Each is a pure function of immutable inputs,
/// so the result does not depend on which worker runs it or when.
#[derive(Clone)]
pub enum WorkerJob {
    /// Terrain edits and the local hierarchy repair they need: `patches`
    /// are applied in order to a copy of `previous`'s terrain (copying a
    /// large map costs a few ms of memory bandwidth, so it happens here, not
    /// on the simulation thread); `rect` (inclusive) covers their cells.
    RepairHierarchy {
        previous: Arc<NavHierarchy>,
        patches: Vec<TerrainPatch>,
        rect: (i32, i32, i32, i32),
        threads: usize,
    },
    BuildField {
        hierarchy: Arc<NavHierarchy>,
        key: FieldKey,
    },
}

impl WorkerJob {
    /// A repair holds back every terrain edit queued after it, so it skips
    /// ahead of field builds.
    fn urgent(&self) -> bool {
        matches!(self, WorkerJob::RepairHierarchy { .. })
    }
}

pub enum WorkerResult {
    Hierarchy {
        hierarchy: Arc<NavHierarchy>,
    },
    Field {
        key: FieldKey,
        field: Option<Box<FlowField>>,
    },
}

struct Sink {
    results: Mutex<BTreeMap<u64, WorkerResult>>,
    /// Obsolete job IDs whose results are dropped on arrival.
    discarded: Mutex<std::collections::BTreeSet<u64>>,
    cv: Condvar,
}

#[derive(Default)]
struct Queues {
    urgent: VecDeque<(u64, WorkerJob)>,
    normal: VecDeque<(u64, WorkerJob)>,
    stop: bool,
}

impl Queues {
    fn pop(&mut self) -> Option<(u64, WorkerJob)> {
        self.urgent.pop_front().or_else(|| self.normal.pop_front())
    }
}

struct Inbox {
    queues: Mutex<Queues>,
    cv: Condvar,
}

pub struct ComputePool {
    inbox: Arc<Inbox>,
    sink: Arc<Sink>,
    workers: Vec<JoinHandle<()>>,
    busy_ns: Arc<AtomicU64>,
    pub worker_count: usize,
    next_id: u64,
}

impl ComputePool {
    pub fn new(worker_count: usize) -> Self {
        let n = worker_count.max(1);
        let inbox = Arc::new(Inbox {
            queues: Mutex::new(Queues::default()),
            cv: Condvar::new(),
        });
        let sink = Arc::new(Sink {
            results: Mutex::new(BTreeMap::new()),
            discarded: Mutex::new(Default::default()),
            cv: Condvar::new(),
        });
        let busy_ns = Arc::new(AtomicU64::new(0));
        let mut workers = Vec::with_capacity(n);
        for wi in 0..n {
            let inbox = Arc::clone(&inbox);
            let sink = Arc::clone(&sink);
            let busy_ns = Arc::clone(&busy_ns);
            workers.push(
                std::thread::Builder::new()
                    .name(format!("nav-worker-{wi}"))
                    .spawn(move || {
                        crate::thread_load::register(&format!("nav-worker-{wi}"));
                        loop {
                            let packet = {
                                let Ok(mut q) = inbox.queues.lock() else { break };
                                loop {
                                    if q.stop {
                                        break None;
                                    }
                                    if let Some(p) = q.pop() {
                                        break Some(p);
                                    }
                                    q = match inbox.cv.wait(q) {
                                        Ok(q) => q,
                                        Err(_) => break None,
                                    };
                                }
                            };
                            let Some((id, job)) = packet else { break };
                            let started = std::time::Instant::now();
                            let busy = crate::thread_load::busy();
                            let result = match job {
                                WorkerJob::RepairHierarchy {
                                    previous,
                                    patches,
                                    rect,
                                    threads,
                                } => {
                                    let mut terrain: Terrain = (*previous.terrain).clone();
                                    for p in patches {
                                        terrain.apply_patch(p);
                                    }
                                    WorkerResult::Hierarchy {
                                        hierarchy: Arc::new(previous.rebuild_local(Arc::new(terrain), rect, threads)),
                                    }
                                }
                                WorkerJob::BuildField { hierarchy, key } => {
                                    let grid = hierarchy.grid(key.profile_id);
                                    let field = build_flow_field(&hierarchy.terrain, grid, key.clone());
                                    WorkerResult::Field {
                                        key,
                                        field: field.map(Box::new),
                                    }
                                }
                            };
                            drop(busy);
                            busy_ns.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            // Hold `discarded` while publishing so a concurrent
                            // `discard(id)` either sees the result or marks it.
                            let mut discarded = match sink.discarded.lock() {
                                Ok(d) => d,
                                Err(_) => break,
                            };
                            if discarded.remove(&id) {
                                continue;
                            }
                            if let Ok(mut map) = sink.results.lock() {
                                map.insert(id, result);
                                sink.cv.notify_all();
                            }
                            drop(discarded);
                        }
                    })
                    .expect("spawn nav worker"),
            );
        }
        Self {
            inbox,
            sink,
            workers,
            busy_ns,
            worker_count: n,
            next_id: 1,
        }
    }

    pub fn busy_ns(&self) -> u64 {
        self.busy_ns.load(Ordering::Relaxed)
    }

    pub fn submit(&mut self, job: WorkerJob) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        if let Ok(mut q) = self.inbox.queues.lock() {
            if job.urgent() {
                q.urgent.push_back((id, job))
            } else {
                q.normal.push_back((id, job))
            }
            self.inbox.cv.notify_one();
        }
        id
    }

    pub fn wait_for(&self, ids: &[u64]) {
        if ids.is_empty() {
            return;
        }
        crate::thread_load::idle(|| self.wait_for_inner(ids));
    }

    fn wait_for_inner(&self, ids: &[u64]) {
        let mut guard = match self.sink.results.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        loop {
            if ids.iter().all(|id| guard.contains_key(id)) {
                break;
            }
            guard = match self.sink.cv.wait(guard) {
                Ok(g) => g,
                Err(e) => e.into_inner(),
            };
        }
    }

    pub fn take(&self, id: u64) -> Option<WorkerResult> {
        self.sink.results.lock().ok()?.remove(&id)
    }

    /// Drop a result without blocking (obsolete job); late results are
    /// discarded when they arrive.
    pub fn discard(&self, id: u64) {
        // Lock order (discarded, then results) matches the worker side.
        let mut d = match self.sink.discarded.lock() {
            Ok(d) => d,
            Err(e) => e.into_inner(),
        };
        let removed = self
            .sink
            .results
            .lock()
            .map(|mut g| g.remove(&id).is_some())
            .unwrap_or(false);
        if !removed {
            d.insert(id);
        }
    }

    pub fn completed_count(&self) -> usize {
        self.sink.results.lock().map(|g| g.len()).unwrap_or(0)
    }
}

impl Drop for ComputePool {
    fn drop(&mut self) {
        if let Ok(mut q) = self.inbox.queues.lock() {
            q.stop = true;
        }
        self.inbox.cv.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}
