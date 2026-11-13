# AssemblyEngine

An RTS-first game engine: a deterministic lockstep simulation on an ECS, with game logic in WASM modules
that run on many cores at once, native services (navigation, spatial queries, sensors) behind batch
APIs, and a frame-graph renderer whose topology a WASM render module declares.

The architecture and roadmap are in [docs/PLAN.md](docs/PLAN.md). Spike results and measured numbers
are in [spikes/README.md](spikes/README.md).

## Status

M0 (workspace, lifted prototypes, benchmarks, risk spikes) is done except one check: running the nav
golden test on aarch64, which the f32 body-math decision depends on. It needs
`sudo pacman -S qemu-user aarch64-linux-gnu-gcc` or an aarch64 CI runner.

M1 (parallel deterministic sim core) has started: the column arena, the multi-instance runtime and
deferred structural commands with a deterministic merge are in place and tested. Workers are still
spawned per tick and merge+apply is serial, so the timings printed by the tests are not scaling numbers.

| Crate | What it is | Origin |
|---|---|---|
| `crates/ae_ecs` | Storage and query layer; flecs behind `flecs_backend`; the column arena | ecs_sim (`flecs.rs`, vendored flecs 4.1.6 + patch) |
| `crates/ae_wasm` | wasmtime runtime: deterministic config, shared memory, N instances with private stacks and TLS | new, from the M0 spikes |
| `crates/ae_sim` | Work ranges, deferred commands keyed `{system, range, seq}`, guest command imports | new |
| `crates/ae_nav` | Flow-field navigation, ORCA avoidance, collision | pathsim, lifted unchanged |
| `crates/ae_framegraph` | Frame graph declare/bake/barriers, plus the declaration decoder | ecs_sim `frame_graph` |
| `crates/ae_render_abi` | Frame graph declaration wire format and builder for render modules (no deps, builds for WASM) | new |
| `crates/ae_profile` | Tracy zones, plots and frame marks; nothing without the `tracy` feature | new |
| `crates/ae_bench` | Benchmark records, regression diff, standalone subsystem suites | new |
| `bins/ae-tool` | `bench` and `bench-diff` | new |
| `spikes/parallel_guest` | M0 risk spikes (three binaries) | new |
| `vendor/` | flecs and its build crate | ecs_sim `vendor/` |

## Requirements

- Rust 1.98 with the `wasm32-wasip1-threads` target (`rustup target add wasm32-wasip1-threads`).
- A C compiler (flecs is built from source).
- Optional: `wasmtime` CLI to run the nav golden test as WASM; the Tracy profiler GUI for `--features tracy`.

## Commands

```sh
cargo test --workspace                     # everything; builds test guests on the fly

# Benchmarks (no window, GPU or network needed)
cargo build --release -p ae-tool
./target/release/ae-tool bench list
./target/release/ae-tool bench all --label mychange            # writes bench/results/mychange/
./target/release/ae-tool bench-diff bench/results/baseline bench/results/mychange   # exit 1 on regression
cargo bench -p ae_framegraph -p ae_nav     # criterion micro benches

# Profiling
cargo run --release -p ae-tool --features tracy -- bench nav --quick --no-save   # attach Tracy

# Spikes
cargo run --release -p spike_parallel_guest                      # N instances over one shared memory
cargo run --release -p spike_parallel_guest --bin flecs_columns  # flecs columns in shared memory, N guests
cargo run --release -p spike_parallel_guest --bin multi_module   # two separately linked modules

# Determinism
cargo test --release -p ae_sim --test parallel_commands -- --nocapture   # 1..8 workers, same world
cargo test -p ae_nav --test golden                                       # bit-equal to the pathsim prototype
```

Set `AE_BENCH_MACHINE` on a dedicated benchmark machine so wall times are only ever compared against
results from the same machine; deterministic counters are compared everywhere.
