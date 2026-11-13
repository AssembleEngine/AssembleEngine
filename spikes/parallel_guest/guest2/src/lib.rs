//! A second sim module sharing linear memory with another, separately
//! linked module. Its heap comes from the host (`host_alloc`), never from
//! its own copy of the libc allocator, which would claim pages the other
//! module's allocator also thinks it owns.

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::hint::black_box;

#[link(wasm_import_module = "env")]
extern "C" {
    fn host_alloc(size: u32, align: u32) -> u32;
    fn host_free(ptr: u32);
    fn host_block_size(ptr: u32) -> u32;
}

struct HostAllocator;

unsafe impl GlobalAlloc for HostAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        host_alloc(layout.size() as u32, layout.align() as u32) as *mut u8
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = layout;
        host_free(ptr as u32)
    }
}

#[global_allocator]
static ALLOCATOR: HostAllocator = HostAllocator;

// The C allocator too: defining these keeps wasi-libc's dlmalloc out of the
// link, so nothing in this module can ever run a second heap.
#[no_mangle]
pub unsafe extern "C" fn malloc(size: usize) -> *mut u8 {
    host_alloc(size as u32, 16) as *mut u8
}

#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut u8) {
    if !ptr.is_null() {
        host_free(ptr as u32)
    }
}

#[no_mangle]
pub unsafe extern "C" fn calloc(count: usize, size: usize) -> *mut u8 {
    let bytes = count.saturating_mul(size);
    let ptr = malloc(bytes);
    if !ptr.is_null() {
        ptr.write_bytes(0, bytes);
    }
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn realloc(ptr: *mut u8, size: usize) -> *mut u8 {
    let new = malloc(size);
    if !ptr.is_null() && !new.is_null() {
        let old = host_block_size(ptr as u32) as usize;
        std::ptr::copy_nonoverlapping(ptr, new, old.min(size));
        free(ptr);
    }
    new
}

#[no_mangle]
pub unsafe extern "C" fn aligned_alloc(align: usize, size: usize) -> *mut u8 {
    host_alloc(size as u32, align as u32) as *mut u8
}

#[no_mangle]
pub unsafe extern "C" fn posix_memalign(out: *mut *mut u8, align: usize, size: usize) -> i32 {
    let ptr = aligned_alloc(align, size);
    *out = ptr;
    if ptr.is_null() { 12 } else { 0 }
}

/// Static data: must land in this module's region.
static WEIGHTS: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = 200 + (i as u32 * 37) % 56; // 200..255
        i += 1;
    }
    table
};

thread_local! {
    static CALLS: Cell<u32> = const { Cell::new(0) };
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

/// Scale positions by a per-row table weight, going through a heap scratch
/// buffer so every call exercises the host allocator.
#[no_mangle]
pub unsafe extern "C" fn damp(pos: *mut Vec2, count: u32, first_row: u32) {
    let pos = std::slice::from_raw_parts_mut(pos, count as usize);
    let mut scratch: Vec<Vec2> = Vec::with_capacity(pos.len());
    for (i, p) in pos.iter().enumerate() {
        let w = WEIGHTS[(first_row as usize + i) & 255] as f32 * (1.0 / 256.0);
        scratch.push(Vec2 { x: p.x * w, y: p.y * w });
    }
    pos.copy_from_slice(black_box(&scratch));
}

#[no_mangle]
pub extern "C" fn tls_bump() -> u32 {
    CALLS.with(|c| {
        c.set(c.get() + 1);
        c.get()
    })
}

/// Where this module's static data and a fresh heap allocation live.
#[no_mangle]
pub extern "C" fn static_address() -> u32 {
    WEIGHTS.as_ptr() as u32
}

#[no_mangle]
pub extern "C" fn heap_probe() -> u32 {
    let b = Box::new(black_box(7u64));
    let address = &*b as *const u64 as u32;
    drop(b);
    address
}
