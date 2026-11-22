//! Persistent sim workers. Worker 0 is the calling thread; workers 1..N are
//! threads that each own one guest instance for the life of the runtime.
//!
//! A batch is one schedule layer: a list of jobs (one work range, or all of
//! a serial system's ranges) that workers claim from a shared counter. Which
//! worker runs which job does not matter for the result: every command and
//! event is keyed by `{system position, range, seq}` and merged by key.

use crate::command::{Command, CommandKey, Emitted};
use crate::host::SpatialIndex;
use crate::imports::{IdMap, WorkerCtx, WorldPtr};
use crate::work::WorkRange;
use ae_abi::{EventSpan, MAX_EVENT_READS, MAX_TERMS, RunDesc};
use ae_wasm::{GuestMemory, Instance};
use anyhow::{Result, anyhow};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use wasmtime::TypedFunc;

/// A system as one layer runs it.
pub struct BatchSystem {
    /// Profiler zone name.
    pub name: Arc<str>,
    /// Index into the runtime's systems (module and host).
    pub index: usize,
    /// Index into the module's systems (and each worker's export table).
    pub module: usize,
    pub position: u32,
    pub term_count: u32,
    pub term_ids: [u64; MAX_TERMS],
    pub write_mask: u32,
    pub stagger: Option<(u32, u32)>,
    pub events: [EventSpan; MAX_EVENT_READS],
    pub event_count: u32,
    /// Stable component ids the system may read with `ae_get`.
    pub gets: [u64; ae_abi::MAX_GETS],
    pub get_count: u32,
    pub ranges: Vec<WorkRange>,
}

/// Ranges `first..end` of one system.
pub struct Job {
    pub system: usize,
    pub first: usize,
    pub end: usize,
}

pub struct Batch {
    pub systems: Vec<BatchSystem>,
    pub jobs: Vec<Job>,
    pub tick: u64,
    pub param: u64,
    pub ids: Arc<IdMap>,
    pub memory: GuestMemory,
    pub spatial: Option<Arc<SpatialIndex>>,
    /// The world the layer reads (`ae_get`); structural changes wait for the
    /// layer's sync point, so it is valid for the whole batch.
    pub world: WorldPtr,
    next: AtomicUsize,
    failed: AtomicBool,
}

impl Batch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        systems: Vec<BatchSystem>,
        jobs: Vec<Job>,
        tick: u64,
        param: u64,
        ids: Arc<IdMap>,
        memory: GuestMemory,
        spatial: Option<Arc<SpatialIndex>>,
        world: WorldPtr,
    ) -> Self {
        Self {
            systems,
            jobs,
            tick,
            param,
            ids,
            memory,
            spatial,
            world,
            next: AtomicUsize::new(0),
            failed: AtomicBool::new(false),
        }
    }
}

/// What one worker produced in a batch.
#[derive(Default)]
pub struct WorkerOut {
    pub commands: Vec<(CommandKey, Command)>,
    pub events: Vec<Emitted>,
    /// Fuel per batch system (zero without metering).
    pub fuel: Vec<u64>,
    pub calls: u64,
    pub error: Option<anyhow::Error>,
}

pub struct Worker {
    pub instance: Instance<WorkerCtx>,
    /// System exports, by schema index.
    pub funcs: Vec<TypedFunc<u32, ()>>,
    /// Guest offsets of this worker's run descriptor and entity-id buffer.
    pub desc: u32,
    pub ids_buf: u32,
    pub ids_cap: u32,
    pub metered: bool,
}

impl Worker {
    pub fn run(&mut self, batch: &Batch) -> WorkerOut {
        let mut out = WorkerOut {
            fuel: vec![0; batch.systems.len()],
            ..Default::default()
        };
        let ctx = self.instance.store.data_mut();
        ctx.ids = Some(batch.ids.clone());
        ctx.spatial = batch.spatial.clone();
        ctx.world = Some(batch.world);
        if let Err(e) = self.run_jobs(batch, &mut out) {
            batch.failed.store(true, Ordering::Relaxed);
            out.error = Some(e);
        }
        let ctx = self.instance.store.data_mut();
        ctx.spatial = None;
        ctx.world = None;
        out.commands = ctx.commands.take();
        out.events = ctx.commands.take_events();
        out
    }

    fn run_jobs(&mut self, batch: &Batch, out: &mut WorkerOut) -> Result<()> {
        loop {
            if batch.failed.load(Ordering::Relaxed) {
                return Ok(());
            }
            let Some(job) = batch.jobs.get(batch.next.fetch_add(1, Ordering::Relaxed)) else {
                return Ok(());
            };
            let sys = &batch.systems[job.system];
            let _zone = ae_profile::zone_dyn(&sys.name);
            for range in &sys.ranges[job.first..job.end] {
                if range.count > self.ids_cap {
                    return Err(anyhow!("work range of {} rows exceeds the worker buffer", range.count));
                }
                let mut desc = RunDesc::zeroed();
                desc.tick = batch.tick;
                desc.param = batch.param;
                desc.system = sys.position;
                desc.range = range.index;
                desc.count = range.count;
                desc.entities = self.ids_buf;
                if let Some((n, phase)) = sys.stagger {
                    desc.stagger_n = n;
                    desc.stagger_phase = phase;
                }
                desc.term_count = sys.term_count;
                desc.event_count = sys.event_count;
                desc.write_mask = sys.write_mask;
                desc.columns = range.columns;
                desc.term_ids = sys.term_ids;
                desc.events = sys.events;
                {
                    let ctx = self.instance.store.data_mut();
                    ctx.gets = sys.gets;
                    ctx.get_count = sys.get_count;
                }
                unsafe {
                    if range.count > 0 {
                        // flecs keeps entity ids on the host heap; the guest gets a copy.
                        std::ptr::copy_nonoverlapping(
                            range.entities,
                            batch.memory.ptr(self.ids_buf).cast::<u64>(),
                            range.count as usize,
                        );
                    }
                    batch.memory.ptr(self.desc).cast::<RunDesc>().write(desc);
                }
                self.instance.store.data_mut().commands.begin(sys.position, range.index);
                let before = if self.metered {
                    self.instance.store.get_fuel()?
                } else {
                    0
                };
                self.funcs[sys.module]
                    .call(&mut self.instance.store, self.desc)
                    .map_err(|e| anyhow!("system trapped: {e:?}"))?;
                if self.metered {
                    out.fuel[job.system] += before - self.instance.store.get_fuel()?;
                }
                out.calls += 1;
            }
        }
    }
}

enum Slot {
    Idle,
    Work(Arc<Batch>),
    Running,
    Done(WorkerOut),
    Shutdown,
}

struct Mailbox {
    slot: Mutex<Slot>,
    cv: Condvar,
    /// Set with `Slot::Work` / `Slot::Done`, so the other side can spin on it
    /// before sleeping on the condvar.
    posted: AtomicBool,
    finished: AtomicBool,
}

/// Spin iterations before sleeping (a few tens of microseconds): layers
/// follow each other faster than a condvar wake-up.
const SPIN: u32 = 20_000;

fn spin_until(flag: &AtomicBool) {
    for _ in 0..SPIN {
        if flag.load(Ordering::Acquire) {
            return;
        }
        std::hint::spin_loop();
    }
}

pub struct Crew {
    main: Worker,
    mailboxes: Vec<Arc<Mailbox>>,
    threads: Vec<JoinHandle<()>>,
}

impl Crew {
    pub fn new(mut workers: Vec<Worker>) -> Result<Self> {
        anyhow::ensure!(!workers.is_empty(), "a crew needs at least one worker");
        let rest = workers.split_off(1);
        let main = workers.pop().unwrap();
        let mut mailboxes = Vec::new();
        let mut threads = Vec::new();
        for (i, mut worker) in rest.into_iter().enumerate() {
            let mailbox = Arc::new(Mailbox {
                slot: Mutex::new(Slot::Idle),
                cv: Condvar::new(),
                posted: AtomicBool::new(false),
                finished: AtomicBool::new(false),
            });
            let mb = mailbox.clone();
            let handle = std::thread::Builder::new()
                .name(format!("sim-worker-{}", i + 1))
                .spawn(move || {
                    ae_profile::set_thread_name(&format!("sim-worker-{}", i + 1));
                    loop {
                        spin_until(&mb.posted);
                        let batch = {
                            let mut slot = mb.slot.lock().unwrap();
                            loop {
                                match std::mem::replace(&mut *slot, Slot::Idle) {
                                    Slot::Work(batch) => {
                                        *slot = Slot::Running;
                                        mb.posted.store(false, Ordering::Release);
                                        break batch;
                                    }
                                    Slot::Shutdown => return,
                                    other => {
                                        *slot = other;
                                        slot = mb.cv.wait(slot).unwrap();
                                    }
                                }
                            }
                        };
                        let out = worker.run(&batch);
                        drop(batch);
                        *mb.slot.lock().unwrap() = Slot::Done(out);
                        mb.finished.store(true, Ordering::Release);
                        mb.cv.notify_all();
                    }
                })?;
            mailboxes.push(mailbox);
            threads.push(handle);
        }
        Ok(Self {
            main,
            mailboxes,
            threads,
        })
    }

    /// Most workers a batch can use.
    pub fn size(&self) -> usize {
        1 + self.threads.len()
    }

    /// Run a batch on up to `active` workers (the caller is worker 0; no
    /// more helpers are woken than there are jobs). Outputs in worker order;
    /// their order does not matter to the merge.
    pub fn run(&mut self, batch: Arc<Batch>, active: usize) -> Vec<WorkerOut> {
        self.run_beside(batch, active, || ()).0
    }

    /// [`Self::run`], but the caller first runs `first` while the helpers
    /// start on the batch, then joins them on the jobs left.
    pub fn run_beside<R>(
        &mut self,
        batch: Arc<Batch>,
        active: usize,
        first: impl FnOnce() -> R,
    ) -> (Vec<WorkerOut>, R) {
        let helpers = active.clamp(1, self.size()).min(batch.jobs.len().max(1)) - 1;
        for mb in &self.mailboxes[..helpers] {
            *mb.slot.lock().unwrap() = Slot::Work(batch.clone());
            mb.posted.store(true, Ordering::Release);
            mb.cv.notify_all();
        }
        let r = first();
        let mut outs = vec![self.main.run(&batch)];
        for mb in &self.mailboxes[..helpers] {
            spin_until(&mb.finished);
            let mut slot = mb.slot.lock().unwrap();
            loop {
                match std::mem::replace(&mut *slot, Slot::Idle) {
                    Slot::Done(out) => {
                        mb.finished.store(false, Ordering::Release);
                        outs.push(out);
                        break;
                    }
                    other => {
                        *slot = other;
                        slot = mb.cv.wait(slot).unwrap();
                    }
                }
            }
        }
        (outs, r)
    }

    pub fn main_worker(&mut self) -> &mut Worker {
        &mut self.main
    }
}

impl Drop for Crew {
    fn drop(&mut self) {
        for mb in &self.mailboxes {
            *mb.slot.lock().unwrap() = Slot::Shutdown;
            mb.cv.notify_all();
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}
