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
- Epoch interruption is not a cheaper watchdog than fuel here, so "fuel off, epoch on" needs a second look.
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

## Not done yet

- f32 determinism of the nav services on Windows and aarch64 (needs CI runners on those platforms).
- Several independently built sim modules over one memory.
- Render ABI expressiveness (re-express the ecs_sim demo graph through a declaration descriptor).
- Flecs entity-id arrays in the column arena (second patch).
