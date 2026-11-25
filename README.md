# AssembleEngine

An RTS-first game engine: a deterministic lockstep simulation on an ECS, with game logic in WASM modules
that run on many cores at once, native services (navigation, spatial queries, sensors) behind batch
APIs, and a frame-graph renderer whose topology a WASM render module declares.

The architecture and roadmap are in [docs/PLAN.md](docs/PLAN.md); [docs/architecture.md](docs/architecture.md)
draws the same structure level by level. Spike results and measured numbers
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
waiting for the sim. The frame graph can come from a render module (WASM): it wires engine library
passes (terrain, units) and its own fullscreen shaders (reflected and checked against the declared
passes), and is swapped while running when its sources change (`ae-client --render-module post`). The
render ABI (contract version 2) carries the output's display spec, so a module declares the same HDR
tail as the native graph (`--render-module post --hdr auto`).
Terrain edits the sim makes reach the render heightfield (`ae-client --quake 60`): the nav service
publishes landed ground changes, and the renderer patches its textures and error pyramid in place.
Terrain is dressed by material tilesets (reusable surface layers, projected by orientation and
height-blended; procedural `temperate` and `desert` sets for now, [docs/tilesets.md](docs/tilesets.md)).
`ae-client --bench all` measures fixed scenes on the GPU for `bench-diff`; `--record-threads N` records
render passes on several threads (one command buffer per pass); a frame graph can also come from a
data file (`--render-module render/post.graph.json`). M3 is complete.

M4 (copy layer and fog): `ae_sensors` stamps per-allyteam line-of-sight and radar grids, `ae_view`
builds each observer's view table by the fields' view classes (`#[field(view = ...)]`), and
`ae-client --teams 2 --observer 0` draws the world as that allyteam sees it, with fog of war on the
terrain. Details: [docs/PLAN.md](docs/PLAN.md) "M4 progress".

Water on wet maps defaults to `full`, with planar reflection, refraction, a displaced surface,
riding hulls, wakes, impact rings, spray, bubbles and an underwater view. `full` is built from the
RedAlert-graphics prototype's ocean, adapted to RTS camera heights and hull speeds: its three-train swell and scrolling static displacement, wave
particles shaped by RA3's blur into Kelvin wakes and blast rings, churned trails drawn with normals,
breakers timed by the baked shore field, and a shell landing in the water throws up a column of spray
instead of a fireball. Use
`--water-quality off|m7|reflect|refract|full` to select a tier. `reflect` adds planar reflection to
M7's analytic surface and `refract` adds the shaded seabed and submerged hulls. A device that
lacks a required rich image format or usage falls back to `m7`. O toggles water in the client,
and `--no-water` starts without it. The native graph, `render/default` and
`render/default.graph.json` support these tiers. On the coast, Crossing's submarines dive and
surface as they sail. With `--teams 2 --observer 0`, Y cycles the observer.
A map package's optional `water` object sets
absorption, the shallow and deep colours, wind, swell, foam, caustics and the submerged-unit aids,
with defaults that reproduce today's frame. The M16 port of the prototype's
water stack — survey, settings, the passes, what each delivery ported and paired captures — is
[docs/water.md](docs/water.md); `--map coast` is its fixture map (a coastline, a shelf, deep water,
a river mouth and a shipping lane), which takes a sea adapted from the prototype's unless a map
package sets its own. `--capture-seconds S` keeps the sim running for S seconds before a capture is read, so it shows
live wakes and rings. The hills cast the
sun's shadows over the ground, the units and the water (a heightfield, exact at any distance; H
toggles, `--no-shadows`), and the units cast theirs through a cascaded shadow map (to become a virtual
shadow map later). Units that die where the player sees explode: GPU particles (flash, fire, sparks,
smoke) sorted back to front and faded softly into the ground (X sets one off under the mouse cursor, `--no-particles`; drawn at half resolution and composited with a depth-aware upsample, `--particles-full`; comparison blends with `--particle-blend moments|slices|froxels|packets`); smoke and
dust cast coloured shadows on the ground, the units and the water (with the units' shadows). Each
explosion leaves a scar (Recoil's, in `assets/decals/scars`), a decal the terrain's shader paints
and wraps over cliff edges, relit by its normal map, its centre glowing like hot metal for up to two
seconds (`--no-decals`; `--scars generated` for the engine's own scorch mark). Details: [docs/PLAN.md](docs/PLAN.md) "M7 progress".

M13 (HDR output) is done. `--hdr auto|hdr10|scrgb|off` picks the swapchain pair the surface offers
(auto: HDR10, then scRGB, then SDR; an explicit mode the surface lacks is an error naming what it
offers), recreating on SDL's HDR state and the `U` key, with `vkSetHdrMetadataEXT` after every
recreate; `--probe-display` prints the display's state and the surface's pairs and exits. The levels
are signal nits, measured from what the WSI's own image description carries: 203 nit paper white and
203 × SDL's headroom as the peak (361 here, shown at the panel's 441), overridable by
`--paper-white`/`--peak`. The scene is lit in BT.709 or BT.2020 (`--working-space`), exposed with no
tone curve (SDR saturates at white, HDR clips at the display's peak), and encoded by `post.encode` into HDR10's
10-bit PQ in BT.2020 or scRGB's 16F, mapped to the output gamut (`--display-gamut bt709|p3|bt2020`;
P3 by default for HDR10). Render modules declare the same tail (the render ABI's display spec).
SDR at the defaults is unchanged. Details: [docs/PLAN.md](docs/PLAN.md) "M13 progress";
[docs/lighting.md](docs/lighting.md) for the colour and exposure numbers.

M14 (unit scripts) is done. A unit's model is driven by a script: a plain Rust `async fn` per
unit, its future in a deterministic slot of the shared memory and resumed on any worker (callins,
`race`/`join`, waits on turns and clips), so a walk, an aim in steps, a shot with a muzzle flash at
a joint and a death fall are all synced and replay bit-identically. The pose is a pure function of
synced channels (clips by layer and mask, manual joint turns/moves/scales); the animation service
advances it on the CPU and the client's `units pose` pass builds the palettes. Masks come from a
rig file beside the model (`Fox.rig.json`: upper and lower bodies). Units have a synced
`Orientation` (a quaternion, drawn with tilt), systems can read other units' components (`get`),
effects at a bone go through the `Effect` stream and an `fx` sink to the client's particles, and a
killed unit (`Dying`, excluded by `without(Dying)` from spatial queries and nav) runs its death
script under a time cap. `games/duel` is the test game (walk, stop, run, a two-step aim while
running, three shots that hit and kill the target, its death). Details: [docs/PLAN.md](docs/PLAN.md) "M14" and "M14 step 6".

M15 (bindless resources) is done. The library passes read their images,
samplers, buffers and storage images from one global descriptor set per frame slot
(`crates/ae_render/src/bindless.rs`, `shaders/bindless.glsl`):
- Images and samplers come from a 128-byte push table.
- Buffers come from a per-pass handle table.
- Per-slot resources use slotted handles, so no index changes between frames.
- Models of every type draw in two multi-draws without rebinding textures.
- Samplers are deduplicated by their settings and shared; a module's handles are retired when
  it is destroyed, so a renderer rebuild does not use up the registry's indices.
- `std140` blocks and the handle tables are uniform buffers at binding 4, not storage buffers:
  measuring found a storage buffer costs 20–30% more per access (step G).

Frames are byte-identical to before (all 61 GPU-test captures). Every engine pipeline and every
fullscreen program (the engine's tonemap, bloom and encode, and a render module's own shaders)
reads the global set at 0: a program's std140 block, its sampled images and its sampler are
handles in a small per-program table, reached through a 16-byte push block. A contract-2 module
may still use the old per-pass form for one version. `bench-diff` against the pre-M15 baseline
(`335e770`): `gpu_total` is within +0.8% to +5.7%, and 2 of 749 metrics regress (a noisy water
pass and the lights list at +10% on one scene), accepted as the cost of the global set. Details: [docs/PLAN.md](docs/PLAN.md)
"M15 — design".

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
| `crates/ae_vfs` | Content file system: layered directory and zip mounts, package manifests and dependency resolution (`Library`), the content hash | new |
| `crates/ae_map` | Map packages (`.aemap`): `map.json` (size, seed, water, terrain-type table, starts) plus `height.png` (16-bit) and `types.png`, to and from nav terrain | new |
| `crates/ae_catalog` | Defs as data: JSON catalogue files resolved against the game's schema to component bytes (`extends`, `abstract`, canonical writer) | new |
| `games/skirmish` | The M6 workload: two allyteams with factories, weapons, armour and command queues (move, attack-move, stop), units and buildings defined by `content/defs/*.json` | new |
| `crates/ae_bot` | Host-side bots that play an allyteam through tick commands, like a player (`SkirmishBot`; `ae-headless --script bot`, the client's opponent) | new |
| `games/alloc` | Guest heap allocation probe for `tests/guest_alloc.rs` | new |
| `bins/ae-headless` | Runs a game module at several worker counts; prints digests, timings, speedups | new |
| `crates/ae_nav` | Flow-field navigation, ORCA avoidance, collision; `NavService` over body columns; movement classes and water; `reference` = pathsim's own sim | pathsim (`reference` unchanged; service carved from it) |
| `crates/ae_spatial` | Per-tick grid with batch radius / box / nearest-k / sweep queries as hit streams | new |
| `crates/ae_framegraph` | Frame graph declare/bake/barriers (colour, depth, several writers per resource), plus the declaration decoder | ecs_sim `frame_graph` |
| `crates/ae_render` | Vulkan side: device with validation counters, memory, uploads, swapchain, pipelines, frame graph executor, render thread, SDL3 window and the display/colour-space probe (`display`); adaptive terrain (`terrain`); GPU-culled unit instances (`units`); the pass library, render module host and fullscreen programs with SPIR-V reflection (`library`, `module`, `fullscreen`); shaders compiled by `build.rs` with `glslc` | ecs_sim `render/`; terrain from map-rendering |
| `bins/ae-client` | Window plus render thread and sim thread: the crossing map's terrain and units with camera controls, or offscreen captures | new |
| `crates/ae_render_abi` | Frame graph declaration wire format and builder; the render module contract and program table (no deps, builds for WASM) | new |
| `sdk/ae-render-sdk` | Guest SDK for render modules: `RenderModule` trait, `render_module!` exports | new |
| `render/default`, `render/post` | Render modules: the engine's scene graph from WASM; the same with a colour grade and a vignette (built for `wasm32-unknown-unknown`; not workspace members) | new |
| `crates/ae_profile` | Tracy zones, plots and frame marks; nothing without the `tracy` feature | new |
| `crates/ae_bench` | Benchmark records, regression diff, standalone subsystem suites | new |
| `bins/ae-tool` | `bench`, `bench-diff`, `navbench` (runs `ae_nav`'s binary), `catalog-fmt`, `pack [--module GAME]`, `unpack-list`, `map-export`, `capture-diff`, `hooks install` | new |
| `spikes/parallel_guest` | M0 risk spikes (three binaries) | new |
| `vendor/` | flecs and its build crate | ecs_sim `vendor/` |

## Requirements

- Rust 1.98.1, pinned by `rust-toolchain.toml` (rustup installs the channel, `rustfmt`, `clippy` and
  both wasm targets on first use: `wasm32-wasip1-threads` for sim guests and `wasm32-unknown-unknown`
  for render modules).
- A C compiler (flecs is built from source).
- `glslc` (shaderc) to build `ae_render`; SDL3 and a Vulkan 1.3 driver to run `ae-client` and the GPU
  tests, which also need `VK_LAYER_KHRONOS_validation`. The SPIR-V gate pins the compiler it was
  recorded with (2026.4 here); another supported glslc builds, but its binary hashes are compared
  advisorily unless `AE_STRICT_SPIRV=1`. `ae-client` builds the crossing game on first
  use, like `ae-headless` (the wasm target above); `--units 0` runs it without the sim.
- Optional: `wasmtime` CLI to run the nav golden test as WASM; the Tracy profiler GUI for `--features tracy`.

## Formatting, lints and the commit hook

One format, `rustfmt` at `max_width = 120` (`rustfmt.toml`), on the toolchain `rust-toolchain.toml`
pins, so the output does not drift between releases. `tools/fmt.sh` formats the whole tree —
the workspace and the guest crates, which have their own manifests and are not workspace members —
and `tools/fmt.sh --check` is its read-only form. `cargo fmt --all` alone is not the command here:
it follows the `flecs_ecs_sys` path dependency into `vendor/`, the patched flecs copy, which stays
unformatted.

Lints are `cargo clippy --workspace --all-targets -- -D warnings`, clippy's defaults only. A fix
must not change bits: where a suggestion would move float results (`approx_constant`,
`suboptimal_flops` and friends), the code keeps its value and carries a scoped `#[allow(...)]` with
the reason.

The hook is tracked (`.githooks/pre-commit`) and installed once per clone:

```sh
./target/release/ae-tool hooks install   # or: git config core.hooksPath .githooks
tools/fmt.sh                             # format, then commit
```

It refuses a commit whose staged `.rs` files the formatter rejects, and one where
`cargo clippy --workspace --all-targets -- -D warnings` fails once any `.rs`, `Cargo.toml`,
`Cargo.lock`, `build.rs` or `.cargo/` file is staged; a docs-only commit returns at once. Measured
warm on one machine: 5 ms for a docs-only commit, 1.6 s for the formatter check, and clippy's own
run for a code commit, about 2 s when nothing changed since the last clippy run and 15 s after code
changes (`docs/PLAN.md` § M17 progress). It is a local guard, not a security boundary:
`git commit --no-verify` skips it, and such a commit must be re-run through the hook before it is
shared. It reads the working tree, not the index, so with a partially staged file, stage everything
first.

## Commands

```sh
cargo test --workspace                     # everything; builds test guests on the fly
AE_TEST_CAPTURE_DIR=/tmp/a cargo test -p ae_render --test gpu -- --ignored --test-threads=1
./target/release/ae-tool capture-diff /tmp/a /tmp/b   # compare two capture runs byte for byte

# Benchmarks (no window, GPU or network needed)
cargo build --release -p ae-tool
./target/release/ae-tool bench list
./target/release/ae-tool bench all --label mychange            # writes bench/results/mychange/
./target/release/ae-tool bench-diff bench/results/baseline bench/results/mychange   # exit 1 on regression
cargo bench -p ae_framegraph -p ae_nav     # criterion micro benches
./target/release/ae-client --bench all --label mychange          # GPU scenes (needs a GPU), one process each, ~1.5 min: bench/results/mychange/gpu-*.json
./target/release/ae-client --bench battle --bench-output hdr10 --bench-size 2560x1440 --label hdr-check   # the readback in HDR10 at 1440p (sdr|hdr10|scrgb)

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
cargo test -p ae_nav --test coast                                       # the M16 coast fixture's digests

# Navigation at scale
./target/release/ae-tool bench nav-sim --threads 8 --no-save            # 10k units, land + water, 1..8 workers
cargo run --release -p ae_nav --bin navbench -- --quick                  # pathsim's navbench on the lifted reference
./target/release/ae-tool navbench --scaling                              # same binary, worker scaling (build navbench first)

# Rendering (needs a Vulkan 1.3 GPU)
cargo test -p ae_render --test gpu -- --ignored --test-threads=1         # offscreen, validation must stay silent
cargo run --release -p ae-client                                         # water crossing map + 10k units (WASD, wheel, right-drag)
cargo run --release -p ae-client -- --map land --capture land.ppm        # 300 sim ticks, then an offscreen frame + GPU ms per pass
cargo run --release -p ae-client -- --map coast --map-cells 768 --units 120 # coastal water
cargo run --release -p ae-client -- --map coast --map-cells 768 --units 120 --camera 4816,2800,22,0.1,-1,57.5 # look up from underwater
cargo test -p ae-client --test water_visibility -- --ignored --test-threads=1 # observer, waves, hulls and particle layers
cargo run --release -p ae-client -- --units 0                            # terrain only
cargo run --release -p ae-client -- --render-module post                 # graph from render/post; edit it while running (N: vignette)
cargo run --release -p ae-client -- --tileset desert                     # another surface library on the same map (M: palette, B: blend weights)
cargo run --release -p ae-client -- --units 0 --camera 10700,12800,350,2.5,0.45   # the lake by the causeway: waves, glints, foam (O: water)
cargo run --release -p ae-client -- --map generated --units 0          # mountains and mesas casting the sun's shadows (H: shadows)
cargo run --release -p ae-client -- --game skirmish                     # units fighting; those that die in view explode (X: an explosion under the mouse cursor)
cargo run --release -p ae-client -- --map land --units 0 --camera 6000,6000,350,0.6,0.45 --explode 12 --capture boom.ppm   # explosions, a second in
cargo run --release -p ae-client -- --quake 60                           # the crossing game's quake at tick 60: rock, then raised ground, on screen
cargo run --release -p ae-client -- --probe-display                      # the display's HDR state and the surface's colour spaces, then exit
cargo run --release -p ae-client -- --working-space bt2020               # light the scene in BT.2020 (SDR maps it back to BT.709; --display-gamut bt709|p3|bt2020 for HDR output)
cargo run --release -p ae-client -- --hdr auto                           # HDR10 when the surface offers it, else scRGB, else SDR (U toggles HDR/SDR)
cargo run --release -p ae-client -- --hdr hdr10 --paper-white 203 --peak 361   # force HDR10 with explicit levels (signal nits; KWin shows 203 at its 248)
cargo run --release -p ae-client -- --render-module post --hdr auto      # a module declares the HDR tail too (N: vignette)
cargo test -p ae_render --test module                                    # render modules without a GPU
cargo test --release -p ae_services --test capture                       # capturing every tick (and the ground feed) leaves the digest unchanged
AE_VK_VALIDATION=1 cargo run --release -p ae-client -- --seconds 8 --exercise   # resize/minimize cycle
```

Set `AE_BENCH_MACHINE` on a dedicated benchmark machine so wall times are only ever compared against
results from the same machine; deterministic counters are compared everywhere.

## Client controls

WASD / arrows pan, the wheel zooms, right- or middle-drag orbits and tilts, Q/E orbit, R/F tilt,
Page Up/Page Down raise and lower the orbit point (Home returns it to the ground), shift moves faster; `+`/`-` sim speed, Pause pauses the sim and the animation clocks, `[`/`]` terrain detail,
T wireframe, L colour by LOD depth, G fog, C freeze the LOD, P print the camera, V v-sync, N render
module flag 0, M tileset, B blend weights, O water, H shadows, Z bloom, X explosion under the mouse, **U HDR/SDR**,
Esc quits.

`--capture` always writes the SDR readback (the offscreen path), so a capture shows neither the HDR
encoding nor the display's levels.
