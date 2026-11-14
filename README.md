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

M1 (parallel deterministic sim core) runs end to end: a game is a WASM module built with `ae-sdk`
(components, events and systems declared with macros into a schema section); the runtime derives the
schedule layers from the declared reads and writes, runs each layer's systems on persistent workers,
merges commands and events by key, and digests the world. The sandbox game at 50k units gives the same
digest at 1, 2, 4, 8 and 16 workers and in dev and release builds, and runs about 5× faster on 8
workers than on 1 (mean tick, 4.97-5.18× over four runs). Guest heap allocation (`Vec`, `BTreeMap`,
`String`, boxes freed by another worker) from parallel systems is correct, but wasi-libc's one heap
lock serialises it: an allocation-heavy probe runs slower at 8 workers than at 4. Not done yet:
per-worker bump arenas, enable-bit columns, batched structural apply, editor hints in the schema,
cross-platform digest CI. Details: [docs/PLAN.md](docs/PLAN.md) "M1 progress".

M2 (native services) runs inside the sim: the columnar navigation service carved from pathsim
(bit-equal to it), movement classes with water (ships, hovercraft, amphibious, fording land units) and
the spatial index are host systems in the game's schedule (`schedule!(host(nav_spawn), orders, quake,
reorder, zones, host(nav), host(spatial), crowding, churn, landed)`), ordered by the same read/write rules as WASM systems.
The game decides when units move by emitting `MoveOrder` events (move, stop, route or formation, with
`OrderPoint` guide points) and edits the terrain with `TerrainEdit`
events and removable `CostOverlay`s (the nav service repairs its paths, moves buried units out and
reports `TerrainChanged`); it declares its movement classes with `movement_classes!`; its systems query
the spatial index. Engine systems run beside the game systems of their layer. 10k units cross the 2560² water map with the same digest at 1, 2, 4 and
8 workers and the same nav state as the standalone service, also when units are despawned and replaced
under it and when the terrain changes under them; a tick takes 4.6-4.75 ms
(median) on 8 workers, of which the nav system is 4.2-4.3 ms against 3.9-4.0 ms standalone. Details: [docs/PLAN.md](docs/PLAN.md) "M2 progress".

M3 (rendering) has started: the frame graph has depth attachments and ordered writers (several passes
drawing into one target), `ae_render` holds the lifted Vulkan executor and render thread, GPU tests run
offscreen with zero validation messages, and `ae-client` shows the crossing map's terrain (GPU-refined
CBT/LEB from map-rendering, coloured by the nav materials, water as a depth tint) with RTS camera
controls, and 10k units of the crossing game on it: the sim runs at 30 Hz on its own thread, an
end-of-tick capture of `Body`/`NavAgent` feeds an instance stream, and the GPU culls and draws them
(interpolated between ticks, ships on the water) while the render thread runs at ~2,500 fps without
waiting for the sim. The render module ABI is next. Details: [docs/PLAN.md](docs/PLAN.md) "M3 progress".

| Crate | What it is | Origin |
|---|---|---|
| `crates/ae_ecs` | Storage and query layer; flecs behind `flecs_backend`; the column arena | ecs_sim (`flecs.rs`, vendored flecs 4.1.6 + patch) |
| `crates/ae_wasm` | wasmtime runtime: deterministic config, shared memory, N instances with private stacks and TLS | new, from the M0 spikes |
| `crates/ae_abi` | Sim module ABI: run descriptor, schema section records, import names (no deps, builds for WASM) | new |
| `crates/ae_sim` | Runtime: schema, layered schedule, worker crew, host systems, deferred commands and events keyed `{system, range, seq}`, digest, end-of-tick field captures | new; schedule and digest from ecs_sim |
| `crates/ae_services` | Engine services as host systems: `nav_spawn` + `nav` (the nav service over `Body`/`NavAgent` rows, `MoveOrder`/`OrderPoint` events as group commands, `TerrainEdit` events as terrain patches), `spatial` (index for guest queries) | new |
| `sdk/ae-sdk`, `sdk/ae-sdk-macros` | Guest SDK: `#[derive(Component)]`, `#[system(...)]`, `schedule!` (with `host(..)` entries), typed columns, commands, events, spatial queries; engine components and `movement_classes!` in `ae_sdk::nav` (feature `nav`) | new |
| `games/sandbox` | The M1 exit workload (built for wasm32 by the tools; not a workspace member) | new |
| `games/crossing` | The M2 exit workload: orders of every kind, neighbour counts and terrain edits in WASM, the units moved by the nav host system | new |
| `games/alloc` | Guest heap allocation probe for `tests/guest_alloc.rs` | new |
| `bins/ae-headless` | Runs a game module at several worker counts; prints digests, timings, speedups | new |
| `crates/ae_nav` | Flow-field navigation, ORCA avoidance, collision; `NavService` over body columns; movement classes and water; `reference` = pathsim's own sim | pathsim (`reference` unchanged; service carved from it) |
| `crates/ae_spatial` | Per-tick grid with batch radius / box / nearest-k / sweep queries as hit streams | new |
| `crates/ae_framegraph` | Frame graph declare/bake/barriers (colour, depth, several writers per resource), plus the declaration decoder | ecs_sim `frame_graph` |
| `crates/ae_render` | Vulkan side: device with validation counters, memory, uploads, swapchain, pipelines, frame graph executor, render thread, SDL3 window; adaptive terrain (`terrain`); GPU-culled unit instances (`units`); shaders compiled by `build.rs` with `glslc` | ecs_sim `render/`; terrain from map-rendering |
| `bins/ae-client` | Window plus render thread and sim thread: the crossing map's terrain and units with camera controls, or offscreen captures | new |
| `crates/ae_render_abi` | Frame graph declaration wire format and builder for render modules (no deps, builds for WASM) | new |
| `crates/ae_profile` | Tracy zones, plots and frame marks; nothing without the `tracy` feature | new |
| `crates/ae_bench` | Benchmark records, regression diff, standalone subsystem suites | new |
| `bins/ae-tool` | `bench`, `bench-diff`, `navbench` (runs `ae_nav`'s binary) | new |
| `spikes/parallel_guest` | M0 risk spikes (three binaries) | new |
| `vendor/` | flecs and its build crate | ecs_sim `vendor/` |

## Requirements

- Rust 1.98 with the `wasm32-wasip1-threads` target (`rustup target add wasm32-wasip1-threads`).
- A C compiler (flecs is built from source).
- `glslc` (shaderc) to build `ae_render`; SDL3 and a Vulkan 1.3 driver to run `ae-client` and the GPU
  tests, which also need `VK_LAYER_KHRONOS_validation`. `ae-client` builds the crossing game on first
  use, like `ae-headless` (the wasm target above); `--units 0` runs it without the sim.
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

# Sim (M1)
cargo build --release -p ae-headless
./target/release/ae-headless --game sandbox --units 50000 --ticks 3000 --workers 1,2,4,8,16 --digest
./target/release/ae-headless --game sandbox --units 50000 --ticks 600 --workers 1,8 --param-hi 500   # stress churn
./target/release/ae-headless --game sandbox --units 10000 --ticks 300 --workers 1 --fuel            # guest fuel per system
# Nav inside the sim (M2): 10k units on the water map; --compare also runs the standalone service
./target/release/ae-headless --game crossing --nav water --units 10000 --ticks 600 --workers 1,2,4,8 --compare
./target/release/ae-tool bench sim-dispatch --threads 16 --no-save

# Determinism
cargo test --release -p ae_sim --test parallel_commands -- --nocapture   # 1..8 workers, same world
cargo test -p ae_sim --test sandbox                                      # whole runtime, pinned digest
cargo test --release -p ae_sim --test guest_alloc -- --nocapture         # guest heap from parallel systems
cargo test --release -p ae_services --test crossing -- --nocapture       # nav in the sim = standalone nav
cargo test -p ae_nav --test golden                                       # bit-equal to the pathsim prototype
cargo test -p ae_nav --test reference_sim --test service_vs_reference    # whole sim and service equal pathsim
cargo test --release -p ae_nav --test water                              # movement classes on water maps

# Navigation at scale
./target/release/ae-tool bench nav-sim --threads 8 --no-save            # 10k units, land + water, 1..8 workers
cargo run --release -p ae_nav --bin navbench -- --quick                  # pathsim's navbench on the lifted reference
./target/release/ae-tool navbench --scaling                              # same binary, worker scaling (build navbench first)

# Rendering (needs a Vulkan 1.3 GPU)
cargo test -p ae_render --test gpu -- --ignored --test-threads=1         # offscreen, validation must stay silent
cargo run --release -p ae-client                                         # water crossing map + 10k units (WASD, wheel, right-drag)
cargo run --release -p ae-client -- --map land --capture land.ppm        # 300 sim ticks, then an offscreen frame + GPU ms per pass
cargo run --release -p ae-client -- --units 0                            # terrain only
cargo test --release -p ae_services --test capture                       # capturing every tick leaves the digest unchanged
AE_VK_VALIDATION=1 cargo run --release -p ae-client -- --seconds 8 --exercise   # resize/minimize cycle
```

Set `AE_BENCH_MACHINE` on a dedicated benchmark machine so wall times are only ever compared against
results from the same machine; deterministic counters are compared everywhere.
