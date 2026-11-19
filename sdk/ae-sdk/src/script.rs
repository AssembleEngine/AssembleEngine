//! Unit coroutines (M14): each unit's script is a plain Rust `async fn`,
//! its future kept in a slot of a host-owned arena so it can wait across
//! ticks and be resumed on any worker instance.
//!
//! ```ignore
//! ae_sdk::script_slots!(1024, 16384);
//! #[global_allocator]
//! static ALLOC: ae_sdk::script::ScriptAlloc = ae_sdk::script::ScriptAlloc;
//!
//! async fn unit(u: Unit) {
//!     loop {
//!         u.update(|c: &mut Counter| c.n += 1);
//!         u.sleep_ticks(3).await;
//!     }
//! }
//!
//! #[system(write(Script, Counter))]
//! fn scripts(run: &Run) {
//!     ae_sdk::script::resume(run, unit);
//! }
//! ```
//!
//! How it holds together:
//! * **Slots.** The module declares the arena ([`script_slots!`]): the
//!   host claims it in the shared memory, zeroes it for each new world and
//!   tells the module where it is (`ae_abi::script`). A unit's slot is its
//!   entity index, so its address never depends on the worker count, and
//!   the future never moves (`Pin` holds). A save would store the arena as
//!   a blob; it loads only into the same module build.
//! * **Resuming.** [`resume`] runs inside an ordinary parallel system over
//!   the units' [`Script`] rows (fixed work ranges, so the result is the
//!   same at any worker count): a new row's future is made, and every row
//!   whose wake tick has come is polled once. A row the script has
//!   finished is dropped from its slot.
//! * **Its unit.** A script reaches its unit only through [`Unit`], which
//!   reads the row being polled (set for each poll, per instance), never
//!   through a pointer kept across an `await`: columns move between ticks.
//!   It writes only its own unit's components (the system's `write` terms);
//!   spawns and events go through the usual commands.
//! * **Waiting.** The engine's futures ([`Unit::sleep_ticks`],
//!   [`Unit::next_tick`]) set the row's wake tick. A poll that returns
//!   `Pending` without setting one awaited something that is not the
//!   engine's and traps.
//! * **No heap across an `await`.** While a poll runs, the module's
//!   allocator ([`ScriptAlloc`], which the module installs) hands out
//!   memory from its instance's bump arena, reset after every poll. Freeing
//!   a block from an earlier poll traps (`heap_survived_an_await`); with
//!   the `script-checks` feature, the arena is also poisoned on reset, so
//!   such a block reads as garbage. Outside polls it is the system
//!   allocator.
//!
//! Not yet (M14 step 5): a global wake queue (here every `Script` row is
//! visited for its wake tick), callins, waits on the animation channels,
//! `sleep` in milliseconds, slots freed when a unit is despawned mid-script.

use crate::abi::script as abi_script;
use crate::{Component, Entity, Run};
use core::cell::Cell;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A unit's coroutine. Spawn a unit with `Script::default()`; [`resume`]
/// does the rest.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0010)]
pub struct Script {
    /// [`state`] values.
    pub state: u32,
    /// Times polled.
    pub polls: u32,
    /// The tick it is next polled at.
    pub due: u64,
    /// `size_of` its future (bytes of its slot in use).
    pub size: u32,
    /// Polls on another instance than the poll before (a diagnostic: which
    /// instance runs a range is not sim state).
    #[field(unhashed)]
    pub moved: u32,
    /// The instance of the last poll ([`instance_tag`]).
    #[field(unhashed)]
    pub last: u32,
    pub reserved: u32,
}

/// [`Script::state`] values.
pub mod state {
    /// Not started: its future is made at the next [`resume`](super::resume).
    pub const NEW: u32 = 0;
    pub const RUNNING: u32 = 1;
    /// Returned; its slot is empty.
    pub const DONE: u32 = 2;
}

/// Bytes of the [`script_slots!`] record.
pub const RECORD_LEN: usize = crate::abi::data_header_len(abi_script::DATA) + 8;

/// The arena record: `slots` slots of `slot_bytes`.
pub const fn slots_record(slot_bytes: u32, slots: u32) -> [u8; RECORD_LEN] {
    assert!(slot_bytes > 0 && slot_bytes.is_multiple_of(abi_script::SLOT_ALIGN), "slot bytes must be a positive multiple of 16");
    let mut b = [0u8; RECORD_LEN];
    let at = crate::abi::put_data_header(&mut b, abi_script::DATA);
    crate::abi::put_u32(&mut b, at, slot_bytes);
    crate::abi::put_u32(&mut b, at + 4, slots);
    b
}

/// Declare the module's slot arena: `slots` slots (entity indices below
/// it) of `slot_bytes` each (at least the script future's size).
#[macro_export]
macro_rules! script_slots {
    ($slot_bytes:expr, $slots:expr $(,)?) => {
        const _: () = {
            #[used]
            #[cfg_attr(target_family = "wasm", unsafe(link_section = "ae.schema"))]
            static RECORD: [u8; $crate::script::RECORD_LEN] = $crate::script::slots_record($slot_bytes, $slots);
        };
    };
}

/// What the polled row is, for [`Unit`] and the engine's futures.
struct PollCx {
    run: *const Run,
    row: usize,
    tick: u64,
    entity: Entity,
    /// The earliest wake a future asked for (`u64::MAX`: none).
    due: Cell<u64>,
}

thread_local! {
    static CX: Cell<*const PollCx> = const { Cell::new(core::ptr::null()) };
    static TAG: u8 = const { 0 };
}

fn cx() -> &'static PollCx {
    let p = CX.with(|c| c.get());
    assert!(!p.is_null(), "a script's unit used outside its poll");
    unsafe { &*p }
}

/// This instance's tag (the address of a thread-local: one per instance).
pub fn instance_tag() -> u32 {
    TAG.with(|t| t as *const u8 as usize as u32)
}

/// The unit a script runs for: its components in the row being polled.
/// Copy it freely; it holds nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unit {
    _private: (),
}

impl Unit {
    pub fn entity(&self) -> Entity {
        cx().entity
    }

    pub fn tick(&self) -> u64 {
        cx().tick
    }

    fn at<T: Component>(&self, write: bool) -> *mut T {
        let c = cx();
        unsafe { (*c.run).column_ptr::<T>(write).add(c.row) }
    }

    /// Its `T` (a term of the resuming system).
    pub fn get<T: Component>(&self) -> T {
        unsafe { *self.at::<T>(false) }
    }

    /// Set its `T` (a `write` term of the resuming system).
    pub fn set<T: Component>(&self, value: T) {
        unsafe { *self.at::<T>(true) = value }
    }

    /// Change its `T` in place (a `write` term). The borrow ends with the
    /// closure, so it cannot be held across an `await`.
    pub fn update<T: Component, R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        f(unsafe { &mut *self.at::<T>(true) })
    }

    /// Wait `ticks` ticks (at least one).
    pub fn sleep_ticks(&self, ticks: u64) -> Sleep {
        Sleep { ticks: ticks.max(1), until: 0 }
    }

    /// Wait for the next tick.
    pub fn next_tick(&self) -> Sleep {
        self.sleep_ticks(1)
    }
}

/// [`Unit::sleep_ticks`].
#[must_use = "a sleep does nothing unless awaited"]
pub struct Sleep {
    ticks: u64,
    /// Set at the first poll (0: not polled).
    until: u64,
}

impl Future for Sleep {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        let c = cx();
        if self.until == 0 {
            self.until = c.tick + self.ticks;
        }
        if c.tick >= self.until {
            return Poll::Ready(());
        }
        c.due.set(c.due.get().min(self.until));
        Poll::Pending
    }
}

struct Slots {
    base: usize,
    slot_bytes: usize,
    slots: u32,
}

fn slots() -> Slots {
    let look = |name| unsafe { crate::host::ae_lookup(abi_script::SCOPE, name) };
    let (base, slot_bytes, slots) = (look(abi_script::BASE), look(abi_script::SLOT_BYTES), look(abi_script::SLOTS));
    assert!(base != crate::abi::lookup::MISSING, "the module declares no script slots (script_slots!)");
    Slots { base: base as usize, slot_bytes: slot_bytes as usize, slots }
}

/// Run the scripts of this range's units: make the futures of new ones
/// (`script(unit)`), and poll each one whose wake tick has come. Call it
/// from a system with `write(Script)` and the terms the script uses, with
/// the same `script` every time.
pub fn resume<Fut: Future<Output = ()> + 'static>(run: &Run, script: impl Fn(Unit) -> Fut) {
    let arena = slots();
    assert!(size_of::<Fut>() <= arena.slot_bytes && align_of::<Fut>() <= abi_script::SLOT_ALIGN as usize, "the script future does not fit a slot");
    let rows: *mut Script = run.column_ptr::<Script>(true);
    let (ids, tick) = (run.entities(), run.tick());
    let waker = Waker::noop();
    for i in run.rows() {
        let row = unsafe { rows.add(i) };
        let mut s = unsafe { row.read() };
        if s.state == state::DONE {
            continue;
        }
        let index = ids[i].index();
        assert!(index < arena.slots, "entity index past the script slots");
        let slot = (arena.base + index as usize * arena.slot_bytes) as *mut Fut;
        if s.state == state::NEW {
            unsafe { slot.write(script(Unit::default())) };
            s = Script { state: state::RUNNING, due: tick, size: size_of::<Fut>() as u32, ..s };
        }
        if s.due > tick {
            continue;
        }
        let poll_cx = PollCx { run, row: i, tick, entity: ids[i], due: Cell::new(u64::MAX) };
        CX.with(|c| c.set(&poll_cx));
        bump_enter();
        let done = unsafe { Pin::new_unchecked(&mut *slot) }.poll(&mut Context::from_waker(waker)).is_ready();
        bump_leave();
        CX.with(|c| c.set(core::ptr::null()));
        // The script may have written its own row; this one is the resumer's.
        let tag = instance_tag();
        s.polls += 1;
        s.moved += (s.polls > 1 && s.last != tag) as u32;
        s.last = tag;
        if done {
            unsafe {
                core::ptr::drop_in_place(slot);
                core::ptr::write_bytes(slot.cast::<u8>(), 0, arena.slot_bytes);
            }
            s.state = state::DONE;
        } else {
            let due = poll_cx.due.get();
            assert!(due > tick, "a script waits for something that is not the engine's");
            s.due = due;
        }
        unsafe { row.write(s) };
    }
}

// ---- The poll-scoped heap -------------------------------------------------

/// Bytes of each instance's bump arena (one poll's allocations), and its
/// alignment: a block's arena is found by masking its address.
const BUMP_BYTES: usize = 1 << 20;
const MAX_ARENAS: usize = 64;
#[cfg(feature = "script-checks")]
const POISON: u8 = 0xA5;

#[derive(Clone, Copy)]
struct Bump {
    base: usize,
    top: usize,
    /// Polls this instance has run; a block's header holds its poll's.
    epoch: u32,
    on: bool,
}

thread_local! {
    static BUMP: Cell<Bump> = const { Cell::new(Bump { base: 0, top: 0, epoch: 0, on: false }) };
}

/// Every instance's arena (allocator bookkeeping, not sim state).
static ARENAS: [AtomicUsize; MAX_ARENAS] = [const { AtomicUsize::new(0) }; MAX_ARENAS];
static ARENA_COUNT: AtomicUsize = AtomicUsize::new(0);

fn arena_of(p: usize) -> Option<usize> {
    let base = p & !(BUMP_BYTES - 1);
    let n = ARENA_COUNT.load(Ordering::Acquire).min(MAX_ARENAS);
    ARENAS[..n].iter().any(|a| a.load(Ordering::Relaxed) == base).then_some(base)
}

fn bump_enter() {
    BUMP.with(|b| {
        let mut v = b.get();
        if v.base == 0 {
            let p = unsafe { System.alloc(Layout::from_size_align(BUMP_BYTES, BUMP_BYTES).unwrap()) };
            assert!(!p.is_null(), "no memory for a script arena");
            let k = ARENA_COUNT.fetch_add(1, Ordering::AcqRel);
            assert!(k < MAX_ARENAS, "too many script arenas");
            ARENAS[k].store(p as usize, Ordering::Release);
            v.base = p as usize;
        }
        v.top = v.base;
        v.on = true;
        b.set(v);
    });
}

fn bump_leave() {
    BUMP.with(|b| {
        let mut v = b.get();
        #[cfg(feature = "script-checks")]
        unsafe {
            core::ptr::write_bytes(v.base as *mut u8, POISON, v.top - v.base)
        };
        v.top = v.base;
        v.epoch = v.epoch.wrapping_add(1);
        v.on = false;
        b.set(v);
    });
}

/// A block from an earlier poll was freed or grown: something allocated in
/// a script was kept across an `await`.
#[inline(never)]
#[cold]
fn heap_survived_an_await() -> ! {
    // Distinct bodies, so the linker does not fold the two (the trap's
    // backtrace names the check).
    die(1)
}

#[inline(never)]
#[cold]
fn script_arena_full() -> ! {
    die(2)
}

fn die(why: u32) -> ! {
    // No panics inside the allocator (they allocate).
    core::hint::black_box(why);
    #[cfg(target_family = "wasm")]
    core::arch::wasm32::unreachable();
    #[cfg(not(target_family = "wasm"))]
    std::process::abort();
}

/// The block at `p` (in arena `arena`) is from this instance's current poll.
fn check_live(p: usize, arena: usize) {
    let v = BUMP.with(|b| b.get());
    let epoch = unsafe { ((p - 8) as *const u32).read() };
    if !v.on || arena != v.base || epoch != v.epoch {
        heap_survived_an_await();
    }
}

/// The module's allocator: poll-scoped bump memory while a script runs,
/// the system allocator otherwise. A module that runs scripts installs it:
/// `#[global_allocator] static ALLOC: ScriptAlloc = ScriptAlloc;`.
pub struct ScriptAlloc;

unsafe impl GlobalAlloc for ScriptAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut v = BUMP.with(|b| b.get());
        if !v.on {
            return unsafe { System.alloc(layout) };
        }
        // An 8-byte header (the poll's epoch) before every block.
        let start = (v.top + 8).next_multiple_of(layout.align().max(8));
        if start + layout.size() > v.base + BUMP_BYTES {
            script_arena_full();
        }
        unsafe { ((start - 8) as *mut u32).write(v.epoch) };
        v.top = start + layout.size();
        BUMP.with(|b| b.set(v));
        start as *mut u8
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        match arena_of(ptr as usize) {
            Some(arena) => check_live(ptr as usize, arena),
            None => unsafe { System.dealloc(ptr, layout) },
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        match arena_of(ptr as usize) {
            Some(arena) => {
                check_live(ptr as usize, arena);
                let new = unsafe { self.alloc(Layout::from_size_align_unchecked(new_size, layout.align())) };
                if !new.is_null() {
                    unsafe { core::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size)) };
                }
                new
            }
            None => unsafe { System.realloc(ptr, layout, new_size) },
        }
    }
}
