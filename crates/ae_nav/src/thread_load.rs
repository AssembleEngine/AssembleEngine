//! Per-thread busy time, for the debug UI's thread graph. Each pathsim
//! thread registers a named slot; work is bracketed by `busy()` guards and
//! blocking waits by `idle()`. Unregistered threads record nothing. This is
//! wall-clock instrumentation only and never feeds back into the simulation.
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

struct Slot {
    name: String,
    /// Finished busy time.
    busy_ns: AtomicU64,
    /// Start of the open busy span (ns since `epoch()` + 1), 0 when idle.
    since: AtomicU64,
    alive: AtomicBool,
    /// The thread is recording under another slot (`lend`), so this one
    /// stands for nothing meanwhile.
    hosting: AtomicBool,
}

static SLOTS: Mutex<Vec<Arc<Slot>>> = Mutex::new(Vec::new());

fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

fn now_ns() -> u64 {
    epoch().elapsed().as_nanos() as u64 + 1
}

/// Marks the slot dead when its thread exits, so it can be reused.
struct Registration(Arc<Slot>);

impl Drop for Registration {
    fn drop(&mut self) {
        close_span(&self.0);
        self.0.alive.store(false, Ordering::Release);
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Registration>> = const { RefCell::new(None) };
    /// A slot borrowed from another thread (`lend`), used instead of CURRENT.
    static LENT: RefCell<Option<Arc<Slot>>> = const { RefCell::new(None) };
}

/// Register the calling thread under `name`. A dead slot of the same name is
/// reused, so short-lived threads (pools in tests, hierarchy helpers) do
/// not grow the list.
pub fn register(name: &str) {
    let mut slots = SLOTS.lock().unwrap_or_else(|e| e.into_inner());
    let slot = match slots
        .iter()
        .find(|s| s.name == name && !s.alive.load(Ordering::Acquire))
    {
        Some(s) => {
            s.alive.store(true, Ordering::Release);
            Arc::clone(s)
        }
        None => {
            let s = Arc::new(Slot {
                name: name.to_owned(),
                busy_ns: AtomicU64::new(0),
                since: AtomicU64::new(0),
                alive: AtomicBool::new(true),
                hosting: AtomicBool::new(false),
            });
            slots.push(Arc::clone(&s));
            s
        }
    };
    drop(slots);
    CURRENT.with(|c| *c.borrow_mut() = Some(Registration(slot)));
}

fn with_slot<R>(f: impl FnOnce(&Slot) -> R) -> Option<R> {
    let lent = LENT.try_with(|l| l.borrow().clone()).ok().flatten();
    if let Some(s) = lent {
        return Some(f(&s));
    }
    CURRENT
        .try_with(|c| c.borrow().as_ref().map(|r| f(&r.0)))
        .ok()
        .flatten()
}

/// The calling thread's slot, to hand to `lend` on another thread.
pub struct SlotRef(Option<Arc<Slot>>);

pub fn current() -> SlotRef {
    SlotRef(LENT.try_with(|l| l.borrow().clone()).ok().flatten().or_else(|| {
        CURRENT
            .try_with(|c| c.borrow().as_ref().map(|r| Arc::clone(&r.0)))
            .ok()
            .flatten()
    }))
}

/// Run `f` with the calling thread recorded under `slot`: work a thread
/// hands to another while it waits (`Par::install`) still counts as the
/// first thread's. The owner must not record anything meanwhile.
pub fn lend<R>(slot: SlotRef, f: impl FnOnce() -> R) -> R {
    struct Restore {
        prev: Option<Arc<Slot>>,
        own: Option<Arc<Slot>>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            let prev = self.prev.take();
            let _ = LENT.try_with(|l| *l.borrow_mut() = prev);
            if let Some(own) = &self.own {
                own.hosting.store(false, Ordering::Release);
            }
        }
    }
    let Some(s) = slot.0 else { return f() };
    let own = CURRENT
        .try_with(|c| c.borrow().as_ref().map(|r| Arc::clone(&r.0)))
        .ok()
        .flatten();
    if let Some(own) = &own {
        own.hosting.store(true, Ordering::Release);
    }
    let _restore = Restore {
        prev: LENT.with(|l| l.borrow_mut().replace(s)),
        own,
    };
    f()
}

fn open_span(s: &Slot) -> bool {
    if s.since.load(Ordering::Relaxed) != 0 {
        return false;
    }
    s.since.store(now_ns(), Ordering::Release);
    true
}

fn close_span(s: &Slot) {
    let since = s.since.swap(0, Ordering::AcqRel);
    if since != 0 {
        s.busy_ns.fetch_add(now_ns().saturating_sub(since), Ordering::Relaxed);
    }
}

/// Counts the calling thread as busy until dropped. Nested guards are no-ops.
pub struct Busy(bool);

impl Drop for Busy {
    fn drop(&mut self) {
        if self.0 {
            with_slot(close_span);
        }
    }
}

pub fn busy() -> Busy {
    Busy(with_slot(open_span).unwrap_or(false))
}

/// Run `f` (a blocking wait) with the calling thread counted as idle.
pub fn idle<R>(f: impl FnOnce() -> R) -> R {
    let was_busy = with_slot(|s| {
        let b = s.since.load(Ordering::Relaxed) != 0;
        close_span(s);
        b
    })
    .unwrap_or(false);
    let r = f();
    if was_busy {
        with_slot(open_span);
    }
    r
}

/// One registered thread: total busy ns so far (including an open span).
pub struct Sample {
    pub name: String,
    pub busy_ns: u64,
    pub alive: bool,
    /// Recording under another thread's slot (see `lend`).
    pub hosting: bool,
}

/// Every slot in registration order (indices are stable).
pub fn sample() -> Vec<Sample> {
    let now = now_ns();
    let slots = SLOTS.lock().unwrap_or_else(|e| e.into_inner());
    slots
        .iter()
        .map(|s| {
            let since = s.since.load(Ordering::Acquire);
            let open = if since != 0 { now.saturating_sub(since) } else { 0 };
            Sample {
                name: s.name.clone(),
                busy_ns: s.busy_ns.load(Ordering::Relaxed) + open,
                alive: s.alive.load(Ordering::Acquire),
                hosting: s.hosting.load(Ordering::Acquire),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn busy_and_idle_spans_are_counted() {
        std::thread::spawn(|| {
            register("thread-load-test");
            let total = || {
                sample()
                    .into_iter()
                    .find(|s| s.name == "thread-load-test" && s.alive)
                    .unwrap()
                    .busy_ns
            };
            {
                let _b = busy();
                let _nested = busy();
                std::thread::sleep(std::time::Duration::from_millis(5));
                idle(|| std::thread::sleep(std::time::Duration::from_millis(50)));
            }
            let t = total();
            assert!(t >= 5_000_000 && t < 50_000_000, "busy {t} ns");
        })
        .join()
        .unwrap();
        assert!(sample().iter().any(|s| s.name == "thread-load-test" && !s.alive));
    }
}
