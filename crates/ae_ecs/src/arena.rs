//! Column arena: where flecs puts component columns.
//!
//! The host claims a range of the sim guest's shared linear memory and
//! installs this allocator as flecs' column allocator, so every component
//! column lives where guest code can address it, while flecs' own state
//! (world, tables, entity index) stays on the host heap.
//!
//! The allocator is deterministic: given the same sequence of calls it hands
//! out the same offsets. It uses power-of-two size classes with per-class
//! free lists; a block's class sits in a 16-byte header in front of it.
//!
//! The flecs column-allocator hook is process-wide, so there is one arena
//! per process. Install it before creating any world and keep it for the
//! life of the process (`install` refuses a second, different range).

use std::ffi::c_void;
use std::sync::Mutex;

const HEADER: usize = 16;
const MIN_CLASS: u32 = 6; // 64-byte blocks, so every block is 16-byte aligned
const CLASSES: usize = 48;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaStats {
    /// Bytes from the arena start to the highest block ever handed out.
    pub high_water: usize,
    /// Bytes in live blocks (rounded to their class).
    pub live: usize,
    pub peak_live: usize,
    pub allocations: u64,
    pub reallocations: u64,
    /// Reallocations that fit the existing block (no copy).
    pub in_place_reallocations: u64,
    pub frees: u64,
}

struct Arena {
    base: usize,
    len: usize,
    top: usize,
    free: [Vec<usize>; CLASSES],
    stats: ArenaStats,
}

static ARENA: Mutex<Option<Arena>> = Mutex::new(None);

fn class_of(size: usize) -> u32 {
    (size.max(1) + HEADER)
        .next_power_of_two()
        .trailing_zeros()
        .max(MIN_CLASS)
}

impl Arena {
    fn alloc(&mut self, size: usize) -> *mut c_void {
        let class = class_of(size);
        let block = match self.free[class as usize].pop() {
            Some(block) => block,
            None => {
                let block = self.top;
                let end = block + (1usize << class);
                if end > self.len {
                    return std::ptr::null_mut();
                }
                self.top = end;
                self.stats.high_water = end;
                block
            }
        };
        self.stats.allocations += 1;
        self.stats.live += 1 << class;
        self.stats.peak_live = self.stats.peak_live.max(self.stats.live);
        let address = self.base + block;
        unsafe {
            (address as *mut u32).write(class);
            (address + HEADER) as *mut c_void
        }
    }

    fn class_of_block(&self, ptr: *mut c_void) -> u32 {
        let address = ptr as usize;
        assert!(
            address >= self.base + HEADER && address < self.base + self.len,
            "pointer {address:#x} is not from the column arena"
        );
        unsafe { ((address - HEADER) as *const u32).read() }
    }

    fn free(&mut self, ptr: *mut c_void) {
        let class = self.class_of_block(ptr);
        self.free[class as usize].push(ptr as usize - HEADER - self.base);
        self.stats.frees += 1;
        self.stats.live -= 1 << class;
    }

    fn realloc(&mut self, ptr: *mut c_void, size: usize) -> *mut c_void {
        if ptr.is_null() {
            return self.alloc(size);
        }
        self.stats.reallocations += 1;
        let old_class = self.class_of_block(ptr);
        if class_of(size) == old_class {
            self.stats.in_place_reallocations += 1;
            return ptr;
        }
        let new = self.alloc(size);
        if new.is_null() {
            return new;
        }
        let old_capacity = (1usize << old_class) - HEADER;
        unsafe { std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), new.cast::<u8>(), old_capacity.min(size)) };
        self.free(ptr);
        new
    }
}

fn with_arena<R>(f: impl FnOnce(&mut Arena) -> R) -> R {
    let mut guard = ARENA.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    f(guard.as_mut().expect("column arena used before install"))
}

unsafe extern "C-unwind" fn column_malloc(size: i32) -> *mut c_void {
    let ptr = with_arena(|a| a.alloc(size as usize));
    assert!(!ptr.is_null(), "column arena exhausted ({size} bytes requested)");
    ptr
}

unsafe extern "C-unwind" fn column_realloc(ptr: *mut c_void, size: i32) -> *mut c_void {
    let new = with_arena(|a| a.realloc(ptr, size as usize));
    assert!(!new.is_null(), "column arena exhausted ({size} bytes requested)");
    new
}

unsafe extern "C-unwind" fn column_free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    // After `release` (its memory is gone with its runtime) frees are moot.
    if let Some(a) = ARENA.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
        a.free(ptr);
    }
}

/// Use `[base, base + len)` for every flecs column from now on.
///
/// # Safety
/// The range must be valid, writable memory that stays mapped for the rest
/// of the process (the sim's shared linear memory never moves). Call before
/// creating any flecs world.
pub unsafe fn install(base: *mut u8, len: usize) -> Result<(), String> {
    let mut guard = ARENA.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    match guard.as_ref() {
        Some(arena) if arena.base == base as usize && arena.len == len => return Ok(()),
        Some(arena) => {
            return Err(format!(
                "column arena already installed at {:#x}+{:#x}; flecs' column allocator is process-wide",
                arena.base, arena.len
            ));
        }
        None => {}
    }
    *guard = Some(Arena {
        base: base as usize,
        len,
        top: 0,
        free: std::array::from_fn(|_| Vec::new()),
        stats: ArenaStats::default(),
    });
    drop(guard);
    unsafe { crate::flecs_backend::install_column_allocator(column_malloc, column_realloc, column_free) };
    Ok(())
}

/// Forget the arena installed at `base` (its owner, the sim runtime, is
/// going away with the memory), so another runtime can install its own.
/// Every flecs world using it must be gone. A different base is left alone.
pub fn release(base: *mut u8) {
    let mut guard = ARENA.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if guard.as_ref().is_some_and(|a| a.base == base as usize) {
        *guard = None;
    }
}

/// The installed range, if any.
pub fn range() -> Option<(usize, usize)> {
    ARENA
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .map(|a| (a.base, a.base + a.len))
}

pub fn contains(ptr: *const u8) -> bool {
    range().is_some_and(|(lo, hi)| (lo..hi).contains(&(ptr as usize)))
}

pub fn stats() -> ArenaStats {
    ARENA
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .map(|a| a.stats)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arena(len: usize) -> (Vec<u64>, Arena) {
        let mut backing = vec![0u64; len / 8];
        let base = backing.as_mut_ptr() as usize;
        (
            backing,
            Arena {
                base,
                len,
                top: 0,
                free: std::array::from_fn(|_| Vec::new()),
                stats: ArenaStats::default(),
            },
        )
    }

    #[test]
    fn same_calls_give_same_offsets() {
        let run = || {
            let (_backing, mut a) = arena(1 << 20);
            let mut offsets = Vec::new();
            let mut live = Vec::new();
            for i in 0..500usize {
                let p = a.alloc((i * 37) % 3000 + 1);
                offsets.push(p as usize - a.base);
                live.push(p);
                if i % 3 == 0 {
                    let q = live.swap_remove((i * 7) % live.len());
                    a.free(q);
                }
                if i % 5 == 0 && !live.is_empty() {
                    let k = (i * 11) % live.len();
                    live[k] = a.realloc(live[k], (i * 53) % 5000 + 1);
                    offsets.push(live[k] as usize - a.base);
                }
            }
            offsets
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn realloc_keeps_contents_and_reuses_freed_blocks() {
        let (_backing, mut a) = arena(1 << 20);
        let p = a.alloc(40);
        unsafe { std::ptr::copy_nonoverlapping(b"0123456789".as_ptr(), p.cast::<u8>(), 10) };
        assert_eq!(a.realloc(p, 45), p, "same class: in place");
        let q = a.realloc(p, 400);
        assert_ne!(q, p);
        assert_eq!(unsafe { std::slice::from_raw_parts(q.cast::<u8>(), 10) }, b"0123456789");
        let r = a.alloc(30);
        assert_eq!(r, p, "the freed 64-byte block is reused");
        assert_eq!(a.stats.live, 64 + 512);
    }

    #[test]
    fn exhaustion_returns_null() {
        let (_backing, mut a) = arena(4096);
        // 3000 bytes plus the header round up to the whole 4 KiB.
        assert!(!a.alloc(3000).is_null());
        assert!(a.alloc(1).is_null());
    }
}
