# M0 risk spikes

Results on a Ryzen 9 5900X (12 cores / 24 threads), Linux, rustc 1.98.1, wasmtime 45.0.2, release builds.
Re-run a spike to refresh its numbers.

## parallel_guest — N wasmtime instances over one shared memory

`cargo run -p spike_parallel_guest --release`

Guest: Rust `cdylib` for `wasm32-wasip1-threads` (imports a shared `env.memory`). One `Store` + `Instance`
per worker thread, all linked to the same `SharedMemory`.

**Result: the model works.** What it takes:

- The first instance runs `__wasm_call_ctors` and owns the guest heap. Its start function initialises
  memory once; later instances skip that (flag in shared memory), so data segments are not re-applied.
- Every further instance needs its own shadow stack and TLS block before its first call:
  set the exported `__stack_pointer` global, and call `__wasm_init_tls(block)` with a block of
  `__tls_size` bytes aligned to `__tls_align`. The guest must export those symbols (see `guest/build.rs`).
- Without that setup a second instance shares the first one's shadow stack and has `__tls_base = 0`,
  so its thread-locals land in low memory. The spike shows this as a negative control.
- With it, 8 instances ran 2000 concurrent calls each with 0 corrupted stack bytes and independent
  thread-local counters.

| Measurement | Value |
|---|---|
| Column bytes after 30 ticks, 1/2/4/8/16 workers vs native Rust | identical digest in every case |
| Native Rust, 1 thread | 3.2 ns/row |
| WASM, 1 worker | 6.1–6.8 ns/row |
| Speedup at 2 / 4 / 8 / 16 workers | 1.99× / 3.66× / 6.92× / 10.88× |
| Empty host→guest call | 16–17 ns |
| ns/row at 64 / 256 / 1024 / 4096 rows per call | 6.99 / 6.99 / 6.65 / 6.74 (range size does not matter from 64 rows up) |
| Fuel metering | about +10% per row |
| Epoch interruption | about +10% per row (no cheaper than fuel on this kernel) |
| NaN canonicalization off | 3.7 ns/row, i.e. canonical NaNs cost about 45% on this float-heavy kernel |

Consequences for M1:

- Batching several chunks per call is not needed; a 1024-row range is fine and smaller ones cost nothing extra.
- Epoch interruption is not a cheaper watchdog than fuel here. Decision (docs/PLAN.md): neither in release
  for the trusted sim; a host watchdog thread aborts on a tick overrun. Fuel only in dev and bench builds.
- NaN canonicalization is the largest single cost of running this kernel in WASM. Without it the guest
  is within about 15% of native. Whether the sim can avoid it (for example by canonicalizing only where
  NaN bits can become observable) is an open design question; it stays on until that is answered.

## ECS structural changes — `ae-tool bench ecs-structural`

50k entities, 5k structural commands per tick (20% despawn, 20% spawn, 30% add, 30% remove component),
applied serially in a fixed order through the patched flecs backend.

| Measurement | Value |
|---|---|
| Apply one tick of 5k commands | 0.42 ms |
| Populate 50k entities | 12.7 ms |
| Two worlds fed the same commands | same digest (entity ids repeat) |
| Digest across separate processes | same (recorded in the bench report) |

## flecs_columns — flecs tables in the shared memory, processed by N guests

`cargo run -p spike_parallel_guest --release --bin flecs_columns`

The M1 core shape, end to end. The host claims 512 MiB of the guest's shared memory by growing it
itself (after the first instance's constructors), runs a deterministic power-of-two allocator over that
range, and points the flecs column allocator hook at it. 1M entities in Position/Velocity tables (plus
Health and a toggled Stunned component) are integrated in place by 1–16 guest instances over 1024-row
ranges; between ticks the host applies 10k structural commands (spawn, despawn, add, remove) serially.

| Check | Result |
|---|---|
| Every column flecs allocated | inside the arena |
| Guest shadow stacks and a 1 MiB guest `malloc` made after the claim | outside the arena (dlmalloc grows around it) |
| World digest after 30 ticks: native Rust vs 1/2/4/8/16 guest workers | identical |
| System phase, native / 1 / 8 / 16 workers | 3.3 / 6.9 / 1.2 / 0.75 ms per tick (5.8× at 8, 9.2× at 16) |
| Structural phase per tick | about 7,000 real table changes, 2.6 ms |
| Spawn directly into the final table (`ecs_new_w_table`, then in-place sets) | 107 ns each |
| Despawn a random entity | 456 ns each |
| Add Stunned to a random entity (table move) | 460 ns each |
| Remove Stunned (table move; rare in this run, 121 per tick) | about 1.7 µs each |

Toggles that would not change an entity's table are skipped and not counted, so every operation above
is a real table change. A first version spawned through 2–3 intermediate tables; switching to direct
spawns made spawning cheap but barely moved the total, because despawns and adds on random rows of
million-row tables dominate (cache misses). M1 therefore needs enable-bit columns for high-frequency
toggles (no table change at all) and structural commands applied in batches grouped by source and
destination table. The kernel here is deliberately small, so the system-vs-structural ratio says little
about real games; the per-change cost is the number to carry forward.

## multi_module — two independently linked sim modules over one memory

`cargo run -p spike_parallel_guest --release --bin multi_module`

Module 1 is the normal guest. Module 2 is a separate crate linked with `--global-base=64 MiB`.

**Result: works, with three rules.**

1. **Reserve the ranges before module 1 grows into them.** The host grows the memory to 96 MiB right
   after module 1's constructors; module 1's dlmalloc later grows past 96 MiB (its columns landed at
   0x6040020).
2. **The second module must not have its own heap.** wasi-libc links dlmalloc even when Rust's
   `#[global_allocator]` is replaced (environment setup references `malloc`). Module 2 therefore defines
   `malloc`/`free`/`calloc`/`realloc`/`aligned_alloc`/`posix_memalign` itself, routed to host imports;
   with that, `dlmalloc` and `sbrk` are no longer in the binary.
3. **Set the shadow stack before the first call, constructors included.** `--global-base` moves data
   but not the stack: the linker still puts module 2's default stack pointer at 1 MiB, on top of module
   1's stack. The host assigns each instance a stack and TLS block from module 2's heap range.

| Check | Result |
|---|---|
| Module 2 statics / heap / stacks | inside 64–80 MiB / 80–96 MiB / 80–96 MiB |
| Both modules concurrently (2 workers each) on two halves of one column, 30 ticks | equals native Rust |
| Module 1 sentinel buffer, module 1 and module 2 thread-locals afterwards | intact, independent |
| Module 2 host heap | 7686 allocations, high-water 536 KiB |

For runtime-loadable modules the SDK has to provide rules 2 and 3 automatically (allocator shim and
an exported stack/TLS contract), and the engine has to hand out link-time base addresses or require
relocatable (PIC) modules. That stays M10 work; the mechanism is proven.

## render ABI — `cargo test -p ae_framegraph --test descriptor`

`ae_render_abi` (no dependencies; builds for `wasm32-unknown-unknown`) holds the declaration wire format
and the `GraphBuilder` a render module uses. `ae_framegraph::descriptor` validates and decodes it.

| Check | Result |
|---|---|
| ecs_sim demo graph written through `GraphBuilder` vs built natively (present and readback variants) | same `signature()`, same bake, same dot export |
| Encode → decode round trip of native graphs | same signature and bake |
| 14 hand-made malformed declarations | each rejected with the expected error |
| 20,000 randomly corrupted declarations | no panic in decode or bake (2,088 decoded, 1,347 also baked) |
| Decode cost, 34 / 258 / 1026 passes | 3.5 / 25 / 96 µs (about 5% of bake time; only on topology change) |

The demo graph fits the format with nothing missing. What M3 still has to add to both sides: depth
roles, pipelines and draw-source handles per pass, and the per-frame parameter block.

## nav golden digests — `cargo test -p ae_nav --test golden`

`crates/ae_nav/tests/common/golden_body.rs` builds terrain, hierarchy, one destination context and one flow
field for a fixed seed, then runs 60 ticks of ORCA plus collision on a 2000-agent crowd, and digests
every stage's bits. The expected values come from running the same file against the original
`~/pathfinding` crate.

| Check | Result |
|---|---|
| `ae_nav` vs original pathsim, x86-64 | identical at every stage (lift is bit-faithful) |
| 1 vs 4 threads; dev (opt-level 1) vs release | identical |
| Compiled to `wasm32-wasip1`, run under wasmtime 50 with NaN canonicalization | identical |

The WASM run goes through a different compiler backend (Cranelift) on a 32-bit target, so it is real
evidence that the f32 rules hold. It does not replace aarch64: that needs `qemu-user` and an aarch64
cross linker (`sudo pacman -S qemu-user aarch64-linux-gnu-gcc`), or a CI runner. Run:
`CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run -W nan-canonicalization=y --" cargo test -p ae_nav --test golden --target wasm32-wasip1 -- golden_digests_single_thread`

## Not done yet

- aarch64 and Windows runs of the nav golden test (tools above, or CI).
- Flecs entity-id arrays in the column arena (second patch).
