//! Guest side of the parallel-instances spike: system-like kernels that work
//! in place on columns the host passes as offsets into shared linear memory.

use std::cell::Cell;
use std::hint::black_box;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

/// Empty call: measures the host -> guest call cost alone.
#[no_mangle]
pub extern "C" fn noop() {}

/// pos += vel * dt, then a little per-row float work so a row costs roughly
/// what a small gameplay system costs.
#[no_mangle]
pub unsafe extern "C" fn integrate(pos: *mut Vec2, vel: *const Vec2, count: u32, dt: f32) {
    let pos = std::slice::from_raw_parts_mut(pos, count as usize);
    let vel = std::slice::from_raw_parts(vel, count as usize);
    for (p, v) in pos.iter_mut().zip(vel) {
        p.x += v.x * dt;
        p.y += v.y * dt;
        let r = (p.x * p.x + p.y * p.y).sqrt();
        if r > 1000.0 {
            let k = 1000.0 / r;
            p.x *= k;
            p.y *= k;
        }
    }
}

thread_local! {
    static CALLS: Cell<u32> = const { Cell::new(0) };
}

/// Per-instance thread-local counter. Instances that share a TLS block would
/// see each other's increments.
#[no_mangle]
pub extern "C" fn tls_bump() -> u32 {
    CALLS.with(|c| {
        c.set(c.get() + 1);
        c.get()
    })
}

/// Fills a buffer on the shadow stack with `tag`, spins, then counts the
/// bytes that changed. Instances that share a shadow stack corrupt each
/// other's buffer when run concurrently. Returns the number of bad bytes.
#[no_mangle]
#[inline(never)]
pub extern "C" fn stack_probe(tag: u32, spins: u32) -> u32 {
    let mut buffer = [0u8; 4096];
    let byte = tag as u8;
    for b in black_box(&mut buffer).iter_mut() {
        *b = byte;
    }
    let mut acc = 0u32;
    for i in 0..spins {
        acc = black_box(acc.wrapping_mul(31).wrapping_add(i));
    }
    black_box(acc);
    black_box(&buffer).iter().filter(|b| **b != byte).count() as u32
}

/// Address of a shadow-stack local, so the host can check where the stack is.
#[no_mangle]
#[inline(never)]
pub extern "C" fn stack_address() -> u32 {
    let local = 0u32;
    black_box(&local) as *const u32 as u32
}
