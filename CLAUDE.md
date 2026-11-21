# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Reading rules

- Never read `~/ecs_sim/src/engine/engine.rs` whole.
  Search it (grep), or read only the lines you need
  with an offset and limit. This applies to subagents
  too: pass this rule along in their prompts.
- `docs/PLAN.md` is about 3,500 lines (roadmap, per-milestone "Mx progress" sections, TODO).
  Grep for headings and read the slice you need.

## What this is

AssembleEngine is an RTS-first engine. It has three parts:

- a deterministic lockstep sim on an ECS (flecs, vendored),
- game logic in WASM modules that run on many workers over one shared memory,
- a Vulkan 1.3 frame-graph renderer whose topology a WASM render module declares.

Where to look:

- `README.md` has the status of each milestone, the crate table and the full command list.
- `docs/architecture.md` draws the structure level by level. Its "Reading the code" table maps a
  kind of change to the files where it starts.
- `docs/PLAN.md` holds the design and the progress notes.
- `~/ecs_sim`, pathsim and map-rendering are the prototypes much of the code was lifted from
  (`PLAN.md` § Workspace has the lift map).

## Build and test

Requirements:

- Rust 1.98 with the `wasm32-wasip1-threads` target (sim guests) and `wasm32-unknown-unknown`
  (render modules).
- A C compiler, for flecs.
- `glslc`, to build `ae_render`. Its `build.rs` compiles the shaders.
- To run the client or the GPU tests: SDL3, a Vulkan 1.3 driver and `VK_LAYER_KHRONOS_validation`.

```sh
cargo test --workspace                                   # everything; builds test guests on the fly
cargo test -p ae_sim --test sandbox                      # one integration test file
cargo test -p ae_sim --test sandbox -- some_test_name    # one test
cargo test -p ae_render --test gpu -- --ignored --test-threads=1   # GPU tests (see below)
AE_UPDATE_GOLDEN=1 cargo test -p ae_render --test golden # regenerate frame-graph .dot goldens
cargo run --release -p ae-client                         # window: crossing map + 10k units
cargo build --release -p ae-headless && ./target/release/ae-headless --game sandbox --units 50000 --ticks 3000 --workers 1,2,4,8,16 --digest
```

- The dev profile is `opt-level = 1`. Determinism and nav tests that the README runs with `--release`
  are much slower without it.
- **GPU tests** are `#[ignore]`d. Without `--ignored` they are skipped. Without
  `--test-threads=1` many of them fail spuriously. Set `AE_TEST_CAPTURE_DIR=<dir>` to save their
  frames as PPM. The validation layer must report zero messages.
- **Guests are not workspace members.** `games/*`, `render/default`, `render/post`, the spike guests
  and `crates/ae_sim/tests/guest` each have their own manifest. Tools and tests build them through
  `ae_sim::build::game(name)` / `build_guest`. The output goes to `target/game-guests`, and
  `RUSTFLAGS` is cleared for that build. `--game X` means `games/X` (crate `ae_game_X`), or a path
  ending in `.wasm`.
- **Benchmarks:**
  - `ae-tool bench …` and `ae-client --bench …` write records under `bench/results/<label>/`.
  - Only `bench/results/baseline/` is committed.
  - `ae-client --bench` never compares runs. Judge regressions with
    `ae-tool bench-diff <base> <head>`, which exits 1 on a regression.
  - Set `AE_BENCH_MACHINE` so wall times are compared only within one machine.
- No rustfmt/clippy config and no CI exist yet; use default `cargo fmt` and `cargo clippy`.

## Architecture rules that constrain changes

- **Only the sim's columns are authoritative.** Views, render state, UI and bots are derived from
  them and may be stale or filtered. Nothing on the render or client side may feed back into the sim
  except through `TickBundle` commands.
- **Determinism is the core invariant.**
  - A digest must be identical at any worker count and in dev and release builds.
  - Row ranges are fixed and do not depend on the worker count.
  - Commands and events merge in key order `{system, range, seq}`. A duplicate key is a hard error.
  - Systems get no clock, random or thread-count input.
  - f32 follows strict rules: no FMA, transcendentals via `libm`, NaN canonicalised.
  - flecs is built with `force_build_release` in every profile, because a debug flecs shifts entity
    ids and therefore digests.
  - Tests pin digests, such as `ae_sim --test sandbox`, and the `ae_nav` golden and reference tests
    are bit-equal to pathsim. Treat a changed digest as a bug unless the change is meant to alter
    sim behaviour.
- **The schema drives everything.** A guest declares its components, events, systems and
  `schedule!` with `ae-sdk` macros into a schema section of the module. From that section the engine
  derives:
  - the schedule layers, from each system's declared reads and writes (emitting a stream counts as
    a write),
  - the digest layout and saves,
  - catalogue validation,
  - view classes (`#[field(view = ...)]`).

  The engine hard-codes no component names.
- **Native services are host systems.** `host(nav)`, `host(spatial)`, `host(nav_spawn)` and
  `catalog_spawn` live in `ae_services`. They sit in the same schedule and follow the same
  read/write rules as WASM systems, and they plug in through `ae_sim::host::HostPlugin`.
- **Dependency rules.**
  - `ae_sim` must not depend on `ae_services`.
  - `ae_render` must not depend on `ae_sim` or `ae_view`. `ae-client` wires them together, so the
    renderer runs with no sim (`--units 0`, GPU tests).
- **Guests never see flecs.** They talk to the engine over `ae_abi` (sim) and `ae_render_abi`
  (render modules). Both crates have no dependencies and build for WASM.
- **flecs edits go through patches.** Never edit `vendor/flecs-upstream/`. Changes to flecs are
  unified diffs in `vendor/patches/`, applied into `vendor/flecs_ecs_sys/src/` (see
  `vendor/README.md`). The column-allocator patch is why component columns live in guest memory.
- **Threads.** The sim runs at a fixed 30 Hz on its own thread. The render thread runs free and
  interpolates between the last two end-of-tick captures. Nothing in the render path may stall a
  tick. Terrain edits reach the renderer through the ground feed.

## Conventions

- **Commit subjects** take the form `area: lowercase summary`, for example `render:`, `client:`,
  `docs:` or `client, tool:`. A code commit is usually followed by its own `docs:` commit, which
  updates `docs/PLAN.md` (progress, TODO) and the README or architecture doc where they describe
  the change.
- **Docs prose** describes what exists and states measured numbers with their conditions. Lists
  under "Not built yet" or TODO keep future work apart from the description of what exists.
