//! Unit coroutines (M14): each unit's script is a plain Rust `async fn`,
//! its future kept in a slot of a host-owned arena so it can wait across
//! ticks and be resumed on any worker instance.
//!
//! ```ignore
//! use ae_sdk::script::{self, Callins, Either, Script, Unit, callin, race};
//!
//! ae_sdk::script_slots!(16384, unit);
//! #[global_allocator]
//! static ALLOC: ae_sdk::script::ScriptAlloc = ae_sdk::script::ScriptAlloc;
//!
//! async fn unit(u: Unit) {
//!     loop {
//!         if u.callin(callin::STOPPED_MOVING).await != 0 {
//!             // Look around until it walks again (the race drops the look).
//!             if let Either::Right(_) = race(look_around(u), u.callin(callin::STARTED_MOVING)).await {
//!                 u.update(|c: &mut Counter| c.interrupted += 1);
//!             }
//!         }
//!     }
//! }
//!
//! #[system(write(Script, Callins, Counter))]
//! fn scripts(run: &Run) {
//!     script::resume(run, unit);
//! }
//! ```
//!
//! How it holds together:
//! * **Slots.** The module declares the arena ([`script_slots!`]) with its
//!   script functions, so a slot is as large as their largest future: the
//!   host claims it in the shared memory, zeroes it for each new world and
//!   tells the module where it is (`ae_abi::script`). A unit's slot is its
//!   entity index, so its address never depends on the worker count, and
//!   the future never moves (`Pin` holds). A slot starts with a header (the
//!   owner, the future's type, how to drop it), so a script left by a unit
//!   despawned mid-script is dropped when the index is next scripted, and a
//!   unit resumed with another script than it started with traps. A save
//!   would store the arena as a blob; it loads only into the same module
//!   build.
//! * **Resuming.** [`resume`] runs inside an ordinary parallel system over
//!   the units' [`Script`] rows (fixed work ranges, so the result is the
//!   same at any worker count): a new row's future is made, and every row
//!   whose wake tick has come, or that has a callin it waits for, is polled
//!   once. A row the script has finished is dropped from its slot.
//! * **Its unit.** A script reaches its unit only through [`Unit`], which
//!   reads the row being polled (set for each poll, per instance), never
//!   through a pointer kept across an `await`: columns move between ticks.
//!   It writes only its own unit's components (the system's `write` terms);
//!   spawns and events go through the usual commands.
//! * **Waiting.** The engine's futures set the row's wake tick
//!   ([`Unit::sleep`], [`Unit::sleep_ticks`], [`Unit::next_tick`], and the
//!   waits that check every tick: [`Unit::until`] and the animation ones)
//!   or its wait mask ([`Unit::callin`]). A poll that returns `Pending`
//!   having set neither awaited something that is not the engine's and
//!   traps. A unit that waits only for callins is not polled until one
//!   comes.
//! * **Callins** are bits in the unit's [`Callins`] row (the [`callin`]
//!   kinds), posted by any system that writes that row: the one that moves
//!   it, or one applying events (a hit) to their target's row. A callin
//!   reaches the script if it waits for it when [`resume`] sees it, or is
//!   polled that tick anyway; either way it is cleared then, so callins
//!   nobody waits for are dropped (as Recoil's are when no thread listens).
//!   State the script checks later belongs in components, not callins.
//! * **Several things at once** without more futures per unit: [`join`]
//!   (both) and [`race`] (the first; the other is dropped, which is
//!   Recoil's `Signal` killing a thread).
//! * **No heap across an `await`.** While a poll runs, the module's
//!   allocator ([`ScriptAlloc`], which the module installs) hands out
//!   memory from its instance's bump arena, reset after every poll. Freeing
//!   a block from an earlier poll traps (`heap_survived_an_await`); with
//!   the `script-checks` feature, the arena is also poisoned on reset, so
//!   such a block reads as garbage. Outside polls it is the system
//!   allocator.
//!
//! Not yet: a global wake queue (every `Script` row is visited for its
//! wake); the aim callin's direction (needs a synced facing); queries of
//! other units; effects at joints.

use crate::abi::script as abi_script;
use crate::{Component, Entity, Run};
use core::cell::Cell;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// A unit's coroutine. Spawn a unit with `Script::default()` (and
/// `Callins::default()` if anything posts it callins); [`resume`] does the
/// rest.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0010)]
pub struct Script {
    /// [`state`] values.
    pub state: u32,
    /// Times polled.
    pub polls: u32,
    /// The tick it is next polled at (`u64::MAX`: only a callin wakes it).
    pub due: u64,
    /// `size_of` its future (bytes of its slot in use, past the header).
    pub size: u32,
    /// Polls on another instance than the poll before (a diagnostic: which
    /// instance runs a range is not sim state, so neither hashed, synced
    /// nor saved).
    #[field(unhashed, unsynced, unsaved)]
    pub moved: u32,
    /// The instance of the last poll ([`instance_tag`]).
    #[field(unhashed, unsynced, unsaved)]
    pub last: u32,
    /// The [`callin`] kinds it waits for (any of them wakes it).
    pub waits: u32,
}

const _: () = assert!(<Script as Component>::ID == crate::abi::engine::SCRIPT);

/// [`Script::state`] values.
pub mod state {
    /// Not started: its future is made at the next [`resume`](super::resume).
    pub const NEW: u32 = 0;
    pub const RUNNING: u32 = 1;
    /// Returned; its slot is empty.
    pub const DONE: u32 = 2;
}

/// Callins posted to a unit's script and not yet seen by [`resume`]
/// (which clears them; the resuming system declares `write(Callins)`).
/// Spawn a unit with `Callins::default()`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0011)]
pub struct Callins {
    /// [`callin`] bits.
    pub pending: u32,
    pub reserved: u32,
}

const _: () = assert!(<Callins as Component>::ID == crate::abi::engine::CALLINS);

impl Callins {
    /// Post `kinds` ([`callin`] bits).
    pub fn post(&mut self, kinds: u32) {
        self.pending |= kinds;
    }
}

/// Callin kinds: bits of [`Callins::pending`] and [`Script::waits`]. The
/// engine's (after Recoil's unit script callins) are posted by whatever
/// system does the thing; the bits from [`GAME`] up are the game's own.
pub mod callin {
    pub const STARTED_MOVING: u32 = 1 << 0;
    pub const STOPPED_MOVING: u32 = 1 << 1;
    /// A weapon is asked to aim.
    pub const AIM: u32 = 1 << 2;
    /// A weapon fired.
    pub const FIRED: u32 = 1 << 3;
    /// The unit was hit.
    pub const HIT: u32 = 1 << 4;
    /// The unit was killed.
    pub const KILLED: u32 = 1 << 5;
    /// The first of the game's own kinds.
    pub const GAME: u32 = 1 << 16;
    pub const ALL: u32 = u32::MAX;
}

/// Bytes of a slot's header, before the future.
pub const HEADER_BYTES: u32 = 16;

/// What a slot holds: whose future, of which type, and how to drop it.
#[repr(C)]
struct Header {
    owner: u64,
    /// [`type_tag`] of the future (0: an empty slot).
    tag: u32,
    /// `drop_future::<Fut>` as a function table index.
    drop: u32,
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

/// `size_of` the future `script` returns (for [`script_slots!`]).
pub const fn future_bytes<F: Fn(Unit) -> Fut, Fut: Future<Output = ()>>(_script: &F) -> u32 {
    assert!(align_of::<Fut>() <= abi_script::SLOT_ALIGN as usize, "a script future aligned past 16");
    size_of::<Fut>() as u32
}

/// A slot's bytes for futures of `sizes`: the header and the largest.
pub const fn slot_bytes(sizes: &[u32]) -> u32 {
    let (mut k, mut max) = (0, 0);
    while k < sizes.len() {
        if sizes[k] > max {
            max = sizes[k];
        }
        k += 1;
    }
    (HEADER_BYTES + max).next_multiple_of(abi_script::SLOT_ALIGN)
}

/// Declare the module's slot arena: `slots` slots (entity indices below
/// it), each large enough for the futures of the script functions listed
/// (every function the module passes to [`resume`]).
#[macro_export]
macro_rules! script_slots {
    ($slots:expr, $($script:expr),+ $(,)?) => {
        const _: () = {
            #[used]
            #[cfg_attr(target_family = "wasm", unsafe(link_section = "ae.schema"))]
            static RECORD: [u8; $crate::script::RECORD_LEN] =
                $crate::script::slots_record($crate::script::slot_bytes(&[$($crate::script::future_bytes(&$script)),+]), $slots);
        };
    };
}

/// What the polled row is, for [`Unit`] and the engine's futures.
struct PollCx {
    run: *const Run,
    row: usize,
    tick: u64,
    entity: Entity,
    /// The resuming system writes `Callins`.
    callins: bool,
    /// Callins posted and not yet taken by a [`Unit::callin`] this poll.
    seen: Cell<u32>,
    /// The earliest wake a future asked for (`u64::MAX`: none).
    due: Cell<u64>,
    /// The callins futures wait for.
    waits: Cell<u32>,
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

/// The sim's ticks a second (`ae_abi::sim::TICK_HZ`).
pub fn tick_hz() -> u32 {
    static HZ: AtomicU32 = AtomicU32::new(0);
    match HZ.load(Ordering::Relaxed) {
        0 => {
            let hz = unsafe { crate::host::ae_lookup(crate::abi::sim::SCOPE, crate::abi::sim::TICK_HZ) };
            assert!(hz != crate::abi::lookup::MISSING && hz > 0, "the host gives no tick rate");
            HZ.store(hz, Ordering::Relaxed);
            hz
        }
        hz => hz,
    }
}

/// Ticks of `ms` milliseconds, rounded up: `ceil(ms × rate / 1000)`.
pub fn ms_ticks(ms: u32) -> u64 {
    (ms as u64 * tick_hz() as u64).div_ceil(1000)
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

    /// The host's game parameter.
    pub fn param(&self) -> u64 {
        unsafe { (*cx().run).param() }
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

    /// Wait `ms` milliseconds (at least one tick): [`ms_ticks`].
    pub fn sleep(&self, ms: u32) -> Sleep {
        self.sleep_ticks(ms_ticks(ms))
    }

    /// Wait `ticks` ticks (at least one).
    pub fn sleep_ticks(&self, ticks: u64) -> Sleep {
        Sleep { ticks: ticks.max(1), until: 0 }
    }

    /// Wait for the next tick.
    pub fn next_tick(&self) -> Sleep {
        self.sleep_ticks(1)
    }

    /// Wait until `done()` (checked now, then once a tick).
    pub fn until<F: FnMut() -> bool>(&self, done: F) -> Until<F> {
        Until { done }
    }

    /// Wait for a callin of `kinds` ([`callin`] bits); its output is the
    /// kinds that came (one poll's), which it takes, so the next wait does
    /// not see them again. Needs `write(Callins)` in the resuming system.
    pub fn callin(&self, kinds: u32) -> CallinWait {
        CallinWait { kinds }
    }
}

#[cfg(feature = "aim")]
impl Unit {
    /// A world direction in its frame, by its `Facing` (a term of the
    /// resuming system): what an aim request gives, as `Anim::dir_in`
    /// takes it.
    pub fn to_unit(&self, dir: [f32; 3]) -> [f32; 3] {
        self.get::<crate::aim::Facing>().to_unit(dir)
    }
}

#[cfg(feature = "anim")]
impl Unit {
    /// Wait for `bone`'s turn (or spin up) to finish (`Anim::turned`; the
    /// resuming system reads `Anim`). A turn set this poll has not started.
    pub fn turned(&self, bone: u16) -> Until<impl FnMut() -> bool> {
        let u = *self;
        self.until(move || u.get::<crate::anim::Anim>().turned(bone))
    }

    /// Wait for `bone`'s move to arrive (`Anim::moved`).
    pub fn moved(&self, bone: u16) -> Until<impl FnMut() -> bool> {
        let u = *self;
        self.until(move || u.get::<crate::anim::Anim>().moved(bone))
    }

    /// Wait for `layer`'s clip to end (`Anim::ended`: a clip played once,
    /// at its end). A clip played this poll has not.
    pub fn ended(&self, layer: usize) -> Until<impl FnMut() -> bool> {
        let u = *self;
        self.until(move || u.get::<crate::anim::Anim>().ended(layer))
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

/// [`Unit::until`].
#[must_use = "a wait does nothing unless awaited"]
pub struct Until<F> {
    done: F,
}

impl<F: FnMut() -> bool> Future for Until<F> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        // `F` is never pinned (the closure is only called).
        let this = unsafe { self.get_unchecked_mut() };
        if (this.done)() {
            return Poll::Ready(());
        }
        let c = cx();
        c.due.set(c.due.get().min(c.tick + 1));
        Poll::Pending
    }
}

/// [`Unit::callin`].
#[must_use = "a wait does nothing unless awaited"]
pub struct CallinWait {
    kinds: u32,
}

impl Future for CallinWait {
    type Output = u32;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<u32> {
        let c = cx();
        assert!(c.callins, "a script waits for a callin, and the resuming system does not write Callins");
        let got = c.seen.get() & self.kinds;
        if got != 0 {
            c.seen.set(c.seen.get() & !got);
            return Poll::Ready(got);
        }
        c.waits.set(c.waits.get() | self.kinds);
        Poll::Pending
    }
}

/// A future run inside [`join`] or [`race`]: running, then its output.
enum Branch<F: Future> {
    Running(F),
    Done(F::Output),
    Taken,
}

impl<F: Future> Branch<F> {
    /// Poll it if it runs; true once it is done.
    ///
    /// # Safety
    /// `self` is pinned (the future inside is never moved while running).
    unsafe fn poll(&mut self, cx: &mut Context<'_>) -> bool {
        match self {
            Branch::Running(f) => match unsafe { Pin::new_unchecked(f) }.poll(cx) {
                Poll::Ready(out) => {
                    // Drops the finished future in place.
                    *self = Branch::Done(out);
                    true
                }
                Poll::Pending => false,
            },
            Branch::Done(_) => true,
            Branch::Taken => unreachable!("a finished branch polled"),
        }
    }

    fn take(&mut self) -> F::Output {
        match core::mem::replace(self, Branch::Taken) {
            Branch::Done(out) => out,
            _ => unreachable!("a branch taken before it was done"),
        }
    }
}

/// Run `a` and `b` together; done when both are.
pub fn join<A: Future, B: Future>(a: A, b: B) -> Join<A, B> {
    Join { a: Branch::Running(a), b: Branch::Running(b) }
}

/// [`join`].
#[must_use = "a join does nothing unless awaited"]
pub struct Join<A: Future, B: Future> {
    a: Branch<A>,
    b: Branch<B>,
}

impl<A: Future, B: Future> Future for Join<A, B> {
    type Output = (A::Output, B::Output);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        let (a, b) = unsafe { (this.a.poll(cx), this.b.poll(cx)) };
        if a && b { Poll::Ready((this.a.take(), this.b.take())) } else { Poll::Pending }
    }
}

/// Which of a [`race`]'s futures finished first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Either<A, B> {
    Left(A),
    Right(B),
}

/// Run `a` and `b` together until one is done (`a` first if both finish
/// in one poll); the other is dropped where it waits.
pub fn race<A: Future, B: Future>(a: A, b: B) -> Race<A, B> {
    Race { a: Branch::Running(a), b: Branch::Running(b) }
}

/// [`race`].
#[must_use = "a race does nothing unless awaited"]
pub struct Race<A: Future, B: Future> {
    a: Branch<A>,
    b: Branch<B>,
}

impl<A: Future, B: Future> Future for Race<A, B> {
    type Output = Either<A::Output, B::Output>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        if unsafe { this.a.poll(cx) } {
            return Poll::Ready(Either::Left(this.a.take()));
        }
        if unsafe { this.b.poll(cx) } {
            return Poll::Ready(Either::Right(this.b.take()));
        }
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

/// A tag for the future type `Fut` (never 0): its name, size and alignment.
fn type_tag<Fut>() -> u32 {
    let mut h: u32 = 0x811C_9DC5;
    let size = (size_of::<Fut>() as u32).to_le_bytes();
    let align = (align_of::<Fut>() as u32).to_le_bytes();
    for &b in core::any::type_name::<Fut>().as_bytes().iter().chain(&size).chain(&align) {
        h = (h ^ b as u32).wrapping_mul(0x0100_0193);
    }
    h | 1
}

/// Drop the future at `p` (a slot's, past the header).
unsafe fn drop_future<Fut>(p: *mut u8) {
    unsafe { core::ptr::drop_in_place(p.cast::<Fut>()) }
}

fn drop_index<Fut>() -> u32 {
    drop_future::<Fut> as unsafe fn(*mut u8) as usize as u32
}

/// Drop whatever future the slot at `slot` holds, by its header, and empty it.
unsafe fn clear_slot(slot: usize, slot_bytes: usize) {
    let header = unsafe { &*(slot as *const Header) };
    if header.tag != 0 {
        #[cfg(target_family = "wasm")]
        unsafe {
            let drop: unsafe fn(*mut u8) = core::mem::transmute(header.drop as usize);
            drop((slot + HEADER_BYTES as usize) as *mut u8);
        }
        #[cfg(not(target_family = "wasm"))]
        unreachable!("script slots exist only inside the engine");
    }
    unsafe { core::ptr::write_bytes(slot as *mut u8, 0, slot_bytes) };
}

/// Run the scripts of this range's units: make the futures of new ones
/// (`script(unit)`), and poll each one whose wake tick has come or that has
/// a callin it waits for. Call it from a system with `write(Script)`,
/// `write(Callins)` if anything posts callins, and the terms the script
/// uses; `script` is one of the functions given to [`script_slots!`].
pub fn resume<Fut: Future<Output = ()> + 'static>(run: &Run, script: impl Fn(Unit) -> Fut) {
    let arena = slots();
    assert!(
        HEADER_BYTES as usize + size_of::<Fut>() <= arena.slot_bytes && align_of::<Fut>() <= abi_script::SLOT_ALIGN as usize,
        "the script future does not fit a slot (is its function in script_slots!?)"
    );
    let (tag, drop) = (type_tag::<Fut>(), drop_index::<Fut>());
    let rows: *mut Script = run.column_ptr::<Script>(true);
    let posts: Option<*mut Callins> = run.has::<Callins>().then(|| run.column_ptr::<Callins>(true));
    let (ids, tick) = (run.entities(), run.tick());
    let waker = Waker::noop();
    for i in run.rows() {
        let row = unsafe { rows.add(i) };
        let mut s = unsafe { row.read() };
        // Callins are seen by this visit, whether they wake the script or not.
        let pending = match posts {
            Some(p) => unsafe { core::mem::take(&mut (*p.add(i)).pending) },
            None => 0,
        };
        if s.state == state::DONE {
            continue;
        }
        let index = ids[i].index();
        assert!(index < arena.slots, "entity index past the script slots");
        let slot = arena.base + index as usize * arena.slot_bytes;
        let header = slot as *mut Header;
        let fut = (slot + HEADER_BYTES as usize) as *mut Fut;
        if s.state == state::NEW {
            // A unit despawned mid-script left its future here, or this one
            // restarts its script.
            unsafe { clear_slot(slot, arena.slot_bytes) };
            unsafe {
                fut.write(script(Unit::default()));
                header.write(Header { owner: ids[i].0, tag, drop });
            }
            s = Script { state: state::RUNNING, due: tick, size: size_of::<Fut>() as u32, waits: 0, ..s };
        } else {
            let h = unsafe { &*header };
            assert!(h.owner == ids[i].0 && h.tag == tag, "a unit resumed with another script than its slot holds");
        }
        if s.due > tick && pending & s.waits == 0 {
            continue;
        }
        let poll_cx = PollCx {
            run,
            row: i,
            tick,
            entity: ids[i],
            callins: posts.is_some(),
            seen: Cell::new(pending),
            due: Cell::new(u64::MAX),
            waits: Cell::new(0),
        };
        CX.with(|c| c.set(&poll_cx));
        bump_enter();
        let done = unsafe { Pin::new_unchecked(&mut *fut) }.poll(&mut Context::from_waker(waker)).is_ready();
        bump_leave();
        CX.with(|c| c.set(core::ptr::null()));
        // The script may have written its own row; this one is the resumer's.
        let me = instance_tag();
        s.polls += 1;
        s.moved += (s.polls > 1 && s.last != me) as u32;
        s.last = me;
        if done {
            unsafe { clear_slot(slot, arena.slot_bytes) };
            s = Script { state: state::DONE, due: u64::MAX, waits: 0, ..s };
        } else {
            let (due, waits) = (poll_cx.due.get(), poll_cx.waits.get());
            assert!(due != u64::MAX || waits != 0, "a script waits for something that is not the engine's");
            assert!(due > tick, "a script asked to wake in the past");
            s.due = due;
            s.waits = waits;
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
