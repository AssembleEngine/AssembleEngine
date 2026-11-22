# AssembleEngine — architecture and roadmap

## Context

AssembleEngine is a new RTS-focused (but flexible) engine, a spiritual successor to Recoil: same extensibility
(a whole game defined by scripts + content), none of the old constraints (serial sim, per-call Lua marshalling,
GL-era renderer, 16-bit ids, hard-coded economy). `/home/ak/AssembleEngine` is empty; this plan sets the
architecture and the milestone order. Four prototypes supply most of the hard parts:

- `~/ecs_sim` (Rust): native flecs with column storage in guest linear memory, wasmtime guest ABI, per-tick
  copy/filter stages, and the tested frame graph. Sim is serial and was only run at ~100–350 entities.
- `~/pathfinding` (Rust): deterministic sector/portal + flow-field navigation, ORCA, collision; 10k units ≈ 4.8 ms tick.
- `~/map-rendering` (Rust) and `~/RedAlert-graphics` (C++): CBT/LEB adaptive terrain; RA3-style water stack.
- `~/vkflecs-baseline`: older frame graph, reference only (async compute, depth roles, shader reflection, bindless).

**Decisions fixed with the user**

| Topic | Decision |
|---|---|
| Netcode | Deterministic lockstep. A copy layer feeds each local context a filtered/degraded subset (fog, radar wobble); filtering runs on every client. |
| ECS | Patched flecs behind a narrow seam; guests and ABI never see flecs. Custom ECS stays a later swap. |
| Host language | Rust. Guests: any WASM language, Rust SDK first. |
| Trust | Game sim + render modules trusted (speed first). UI widgets and AIs untrusted (strict sandbox, filtered data only). |
| Frame graph | The render WASM module declares topology (overturns `ecs_sim/docs/TASK-frame-graph.md` §4.5). Topology at init/on change, data per frame. |
| Sim modules | One linked sim module per game in v1; M0 spike tests several modules over one memory. |
| UI | RmlUi over FFI (RML/RCSS documents + data models), driven by untrusted widget modules. |
| Body math | f32 under strict rules; cross-platform golden-hash check in M0, fixed-point only if it fails. |

Assumed defaults (change if wrong): tick rate is per-game data, default 30 Hz; aarch64 as a determinism
CI target.

Decided (2026-11-14): Vulkan 1.3 is the rendering baseline on Linux (and Windows, if targeted); macOS is
left for later.

Decided from the M0 spike: the trusted sim runs with **no fuel and no epoch interruption** in release
(both cost about 10% per row). A host watchdog thread catches a tick overrun, logs the system and row
range that was running, and aborts: a fatal local error, never a sim-visible outcome. Fuel is on only in
dev and bench builds, as the deterministic per-system cost counter. Untrusted modules always run with
fuel and memory caps.

NaN canonicalization stays on (cost up to about 45% on pure float kernels). M1 re-measures it on a mixed
kernel and evaluates the alternative: canonicalize float fields only where bits become observable (digest,
save, net encode; the schema knows which fields are floats) and forbid reading float bits in sim code
(`to_bits`, `total_cmp`, `copysign`, sign tests).

## Architecture

### Worlds and modules

| World | Storage | Written by | Purpose |
|---|---|---|---|
| Sim (authoritative) | flecs; columns in the sim guest's shared linear memory | sim module, native services | The only hashed, saved, replayed state |
| View (per local observer) | native SoA tables, previous + latest generation | copy layer only | Filtered replica for render, UI, AI |
| Render state | render module's private memory + host GPU objects | render module | Graph topology, materials, per-frame params |

| Module kind | Trust | Sees | Synced |
|---|---|---|---|
| Sim game module | trusted, N instances over one shared memory | sim columns in place + native services | yes |
| View-rule module (optional) | trusted | captured rows | no (deterministic) |
| Render module | trusted, private memory | view streams, camera | no |
| UI widget / AI | untrusted, capped, allowlisted imports | view tables only | no |

Untrusted modules affect the game only by emitting player commands through the normal input path, so they
cannot desync it.

Threads: main (SDL3 window/input), sim coordinator, one fork-join job pool shared by sim / copy layer / render
recording, render thread (never waits on sim), net IO, asset IO, unsynced-script thread.
Binaries: `ae-client`, `ae-headless` (sim only: CI, replay verify, AI host), `ae-relay` (no sim), `ae-tool`.

### Sim core

- **Seam (`ae_ecs`)**: `declare_component`, `declare_query`, `chunks`, `apply_commands`, `snapshot/restore`;
  engine `EntityId(u64)` / `ComponentId(u32)`. Flecs stays inside `ae_ecs::flecs_backend`.
- **Column arena**: keep the `ecs_column_allocator_set` hook, but point it at a host-owned deterministic arena
  in a reserved range of the shared memory instead of guest `malloc` (today's design ties fuel and heap
  layout to flecs growth). Second small flecs patch routes entity-id arrays there too, removing the per-chunk id copy.
- **Parallel sim callins (core requirement)**: several worker threads run guest system callbacks at the same
  time. One wasmtime Store/Instance per worker over the same `SharedMemory`, host-assigned stack + TLS per
  instance. Two levels of parallelism: (a) systems in the same schedule layer (no read/write conflict) run
  concurrently; (b) one system's tables are split into fixed-size row ranges, independent of worker count,
  and fanned out. `schedule.rs` layers become authoritative (conflict edges from declared read/write sets).
  "Where it makes sense": a system declares `Parallel` or `Serial`; the host also runs a system serially
  when its row count is below a threshold (fixed constant, so the choice never depends on machine or
  timing). Results are identical either way, so this is purely a cost decision. Parallel systems use
  per-worker bump arenas, no global heap.
- **Commands**: spawn/despawn/add/remove deferred into per-worker buffers keyed
  `{system, table, chunk, row_seq}`, merged by key and applied serially at layer sync points (extends `preentity.rs`).
- **Structure**: add/remove supported (deferred); enable-bit columns for high-frequency toggles; no flecs pairs
  in ABI v1 — relations are `EntityId` fields + a native reverse-index service. Query terms: AND, NOT, OPTIONAL, tags, singletons.
- **Schema**: one descriptor in a wasm custom section, generated by `#[derive(Component)]`. Per field: stable id,
  type, offset, default, flags (`synced`, `hashed`, `saved`, view class). Host derives digest layout,
  save/net codecs, catalogue validation. No hard-coded component names. Saves map by field id.
- **Cadences**: `EveryTick`, `Every{n,phase}`, `Staggered{n}` (row filter by `entity_index % n`).
- **Events and hooks**: typed columnar streams, double-buffered per tick, concatenated in command-key order.
  Recoil's veto/override/claim callins become ordered reductions over request streams
  (e.g. `DamageRequest` rows → modifier systems in priority order → one apply system).
- **Guest state rule**: no persistent sim state outside components/singletons; CI checks save→load→continue digests.

### Determinism

- Guests: WASM float + Cranelift NaN canonicalization, relaxed SIMD off, no clock/random imports (ecs_sim config).
- Native services: integers where possible; f32 limited to `+ - * /` and `sqrt` in fixed order, no FMA,
  transcendentals via pure-Rust `libm`; no hash-map iteration; parallel results merged in index order.
- Async native jobs install at a deterministic due tick in job-id order (pathsim rule, engine-wide).
- Digest: order-independent xxh3-128 row hash (`hash.rs`), schema-driven, plus per-service part hashes.
- Desync: per-component/per-service digests dumped; `ae-headless --bisect` reports first divergent tick/component/entity.
- Replay = header (engine, schema, content hashes, start script) + per-tick commands + periodic digests.
  Save = schema-driven columnar dump + service blobs (not a raw memory image).

### Native services (batch: columns in, columns out)

| Crate | From | Notes |
|---|---|---|
| `ae_nav` | pathsim `nav/*`, `avoidance.rs`, `collision.rs` | `MoveGoal` stream + `Body` in → `PreferredVel`, `NavStatus` out. Add data-driven movement classes (depth range, land/water mask, hover → ships, subs, amphibious), cost overlays, formation facing; second hierarchy level later. Steering carved out of `sim.rs`. |
| `ae_spatial` | new | per-tick grid; batch radius/box/nearest-k; segment sweeps as hit streams |
| `ae_sensors` | new | per-allyteam LOS/air/radar/sonar/seismic/jammer grids → `VisibilityMask` |
| `ae_terrain` | pathsim `terrain.rs`, map-rendering `heightmap.rs` | authoritative heightfield + water level; patch stream → nav repair + render dirty rect |

Engine-native: the above plus command-queue container, relation index, RNG. Game WASM: economy, weapons and
damage, unit AI/state machines, build/factory logic, animation drivers, command semantics, victory rules.

### Copy / filter layer (`ae_view`)

Generalises ecs_sim's capture → filter → validate → publish (`presentation.rs`, `publish.rs`, `stage.rs`):
end-of-tick memcpy of flagged columns → native rule executor with a view class per field
(`Always`, `OwnerOnly`, `Los`, `RadarDegraded`, `Ghost`, `Never`) evaluated against sensor masks →
optional trusted WASM kernels for game-specific degradation. Radar wobble is a hash of (tick, entity, observer).
View rows use stable `ViewId`s that hide sim ids. `ViewSlot` latest/previous publish; the tick never waits.
Interpolation: previous + latest transforms uploaded per instance, lerped in the shader.

### Renderer

- Lift `ecs_sim/src/render/frame_graph/*` unchanged with its tests, then add in order: depth-stencil, MRT,
  multi-writer versions, indirect/compute buffers, fully bindless resources (M15), async compute
  (vkflecs-baseline as reference).
- Render ABI: `rg_declare(descriptor)` lists resources, passes (roles, attachments, pipeline, draw source), roots;
  host validates and rebakes only when `signature()` changes. One per-frame call returns pass enables,
  uniform blocks, draw-stream params. Shaders are content SPIR-V, reflected with rspirv
  (`vkflecs-baseline/src/shader_reflection.rs`) and checked against declared access.
- Bindless resources (M15): shaders reach a resource by a renderer-issued 32-bit handle into a
  few global descriptor arrays (sampled images, samplers, storage images, storage buffers,
  uniform buffers) instead of a binding slot. The frame graph still owns resources, transitions
  and lifetimes, and handles stay valid across rebakes; the per-binding layouts above remain
  until every library pass and module has moved.
- Draw sources are data: `InstanceStream{view table, field→attribute map, mesh/material set}`; compute pass
  does culling + LOD → indirect draws. Multi-threaded recording per pass group.
- Engine pass libraries the module instantiates, wires, or replaces: `terrain_cbt` (map-rendering `leb.rs` +
  RedAlert `cbt.glsl`; surface appearance from the material tilesets below), `water_ra` (RedAlert
  `src/draw.cpp:189-610` pass order + shaders; M16), cascaded shadows, fog of war (sensor grid → texture),
  decals, GPU particles, post, UI.
- Models: glTF converted offline to one bone-based format (M11: bones and joints; pieces are joints at weight 1), drawn in two GPU layouts (rigid: one joint per vertex; skinned: four), GPU-driven; piece animation = sim WASM writing `PieceTransform` columns. Superseded in part (2026-11-16): glTF loaded directly, no `.aemodel` (Later improvements, "glTF loaded at run time"); animation by unit scripts and a CPU pose fast path (M14), not `PieceTransform` columns.
- UI: RmlUi behind a C FFI crate, rendered in a frame graph pass; untrusted widget modules drive documents
  and data models through a validated API and read only view tables.

#### Terrain appearance — reusable material tilesets (MrBob's proposal)

Terrain layout and terrain appearance are separate inputs. A map supplies only its heightmap and
gameplay data (resource locations, features, …); artists do not author a unique colour and normal
texture set for every map.

- **Material tilesets** are reusable per environment — a desert set might contain sand, rock and other
  surface layers. Each layer has a tileable albedo texture and a matching normal map. Maps reuse the
  same collection; a custom look replaces the textures, not the shader.
- **Projection follows surface orientation**: planar tiling on upward-facing ground, duoplanar
  projection on steep slopes and cliffs, and duoplanar or triplanar where the material setup calls
  for it.
- **Two-level blending**: geometry normals give the broad transitions between flatter ground and
  cliffs; the tileset normal maps' pixel normals break those transitions up with fine detail. This is
  the central point: boundaries between materials become irregular and detailed instead of a smooth
  line determined only by terrain slope.
- **The shader assembles the appearance**: material masks are generated procedurally from surface
  orientation and normal-map detail, then used to blend the tileable layers. The same setup therefore
  works across different heightmaps without repainting terrain appearance per map.
- **Map data**: `.aemap` keeps gameplay type/material masks (passability, terrain type, resource
  placement); per-map colour and normal textures are not required.
- **M3 validation**: demonstrate the shader on an existing heightmap and document the shader source or
  the blending logic, the tileset structure with texture counts/sizes/formats, the engine inputs and
  controls, and how tilesets are shared and replaced. The intended result is less duplicated artwork
  and texture data, more consistent environments, and a split workflow: map creators own layout,
  material artists own reusable surface libraries.

### Content, session, networking

- `ae_vfs`: layered directories + archives, manifest (`name`, `version`, `depends`, `kind`), per-file hashes →
  synced content root (widgets/skins excluded).
- `ae_catalog`: data files → schema-validated prefabs (`DefId`); explicit `extends`; economy/armour are just game components.
- `.aemap`: one archive (height, gameplay type/material masks, water field, start positions, metadata).
  Terrain colour and normal detail come from shared material tilesets, not per-map textures. `ae_map`
  bakes nav grids, render error pyramid, shore travel-time field from it.
- Hot reload: render module, shaders, widgets always; catalogue at tick boundary (dev/single-player);
  sim module when schema-compatible.
- `ae_net`: relay-paced lockstep (`TickBundle{tick, commands}`), adaptive input delay, QUIC (`quinn`),
  schema-versioned commands with 64-bit ids; handshake checks engine, module, schema, content hashes.
  Late join v1 = fast-forward from command log. Teams/allyteams/neutral are engine-level.

## Benchmarking and profiling (in place from M0)

Goal: catch regressions and find bottlenecks per subsystem, without launching the whole engine.

**Benchmark layers**

| Layer | What it measures | How |
|---|---|---|
| Micro (`cargo bench -p <crate>`) | One function or kernel: chunk dispatch call cost, command merge, digest, column capture, view rules, flow-field build, ORCA step, grid queries, frame graph bake, barrier planning | `criterion` benches in each crate |
| Subsystem (`ae-tool bench <name>`) | One crate driven by a synthetic world, no window, no GPU, no network: `sim-dispatch`, `nav`, `spatial`, `sensors`, `view`, `framegraph-bake`, `net-loopback` | `ae_bench` crate with seeded world generators (N entities, archetype mix, map size) so every service runs standalone. Ports `pathfinding/src/bin/navbench.rs` and `ecs_sim/examples/presentation_bench.rs`. |
| Scenario (`ae-headless --bench <scenario>`) | Whole sim tick on recorded replays and scripted scenarios at 1k/10k/50k units, per worker count | Per-system and per-service time, tick median / p99 / max, scaling curve, memory |
| GPU (`ae-client --bench <scene> --frames N`) | Per-pass GPU time on fixed camera paths | Vulkan timestamp queries per frame graph pass; runs only where a GPU runner exists |

**Regression detection**
- Every bench writes one JSON record (commit, machine id, metric, value, spread) into `bench/results/`;
  `ae-tool bench-diff <base> <head>` prints a table and fails above a per-metric threshold.
- Deterministic counters alongside wall time, because they are noise-free and gate CI reliably: guest fuel
  per system (wasmtime fuel in dev builds), flow-field expansions, node counts, command counts, bytes
  copied, allocations. A fuel increase on an unchanged scenario is a real regression, on any machine.
- Wall-time benches run on one pinned machine (nightly); counter benches run on every PR.
- Each milestone's exit numbers (10k/50k digests and timings) become permanent scenario benches.

**Tracy (optional, cargo feature `tracy`; zero cost when off)**
- `ae_profile` crate: thin macros over `tracy-client`, compiled to nothing without the feature.
- Zones: tick phases, each schedule layer, each system, each row-range job per worker, command merge, each
  native service stage, copy-layer stages, frame graph bake, per-pass recording, submit/present.
- Guest zones: `host_profile_begin/end` imports, linked as no-ops when profiling is off, so game modules can
  mark their own hot spots; module function names resolved for the sim module.
- GPU zones: per-pass timestamp queries reported as Tracy GPU zones.
- Plots: entity and archetype counts, fuel per tick, queue depths, view lag, memory per arena.
  Frame marks for sim tick and render frame separately. Lock contention on the job pool.
- Also enable wasmtime's `perfmap`/`jitdump` profiling option so `perf` can see into guest code.
- The same per-system timings feed the editor's schedule graph and a built-in overlay, so basic numbers are
  available even without Tracy attached.

## Editor and visual tooling (exploratory, optional)

What Godot and Unreal do, and what fits this architecture. Sourced from my knowledge of both engines, not from
a fresh review of their code.

| Idea | Where it comes from | Fit for AssembleEngine |
|---|---|---|
| Inspector generated from reflected properties + hints (range, enum, asset type) | Godot `@export`, Unreal `UPROPERTY` meta → Details panel | **Take, nearly free.** The component schema already exists; add editor hints per field (label, range, unit, enum, reference kind, tooltip). |
| Editor attaches to a running game in another process; live tree, live property edits | Godot remote scene tree / debugger | **Take.** `ae_inspect` protocol over a local socket to `ae-client`/`ae-headless`. Edits travel as synced dev commands, so they stay deterministic and land in replays. |
| ECS debugger: archetypes, processors, fragment access, timings | Unreal Mass Entity debugger | **Take.** Archetype/table browser plus the schedule graph (layers, read/write conflicts, per-system cost from dev-mode fuel). |
| Render graph visualiser | Unreal RDG Insights | **Take.** Already have `.dot`; make it live with pass timings and resource lifetimes. |
| Spreadsheet editing of definition rows | Unreal Data Tables / Data Assets, Godot Resources | **Take.** RTS balance is table work: grid view over catalogue defs, with `extends` shown as inherited/overridden cells (Unreal "data-only Blueprint", Godot inherited scenes). |
| Hierarchical tags | Unreal Gameplay Tags | **Take.** Replaces Recoil's 32 category bits. |
| Import pipeline with per-asset settings and a hash-keyed derived cache | Godot `.import` sidecars, Unreal Derived Data Cache | **Take.** Same mechanism as the map bake cache. |
| Play-in-editor with several clients and network emulation | Unreal multiplayer PIE | **Take.** Editor launches N clients + relay with injected latency/loss; determinism adds replay scrubbing and step-by-tick for free. |
| Terrain sculpt/paint, splines, foliage painting | Unreal Landscape, Godot terrain plugins | **Take, later.** Map editor on top of the terrain patch stream, with live nav and sensor overlays. Recoil games already do this in-engine (SpringBoard). |
| Material node graph → shader; instances override parameters only | Unreal Material Editor / instances, Godot VisualShader | **Take, later.** Graph compiles to SPIR-V; fits "pipelines are validated content"; instance = parameter block. |
| Particle systems as data-oriented module stacks compiled to CPU VM or compute | Unreal Niagara | **Take, later.** Effect graph → compute shader; the modern replacement for Recoil's CEG. |
| Behaviour as data-oriented state machines | Unreal StateTree (built for Mass), Behavior Trees, Godot AnimationTree | **Take, later.** Graph compiles to tables run by one batch WASM system; also covers unit piece animation. |
| Frame graph as an editable node graph | (follows from our design) | **Possible.** Topology is already a descriptor, so it can be authored as data instead of code. |
| Data-driven modifiers with duration and stacking | Unreal Gameplay Ability System effects | **Possible.** Natural front end for the damage/economy reduction streams. |
| Editor extended in the game's own script language | Godot `@tool` / EditorPlugin, Unreal Editor Utility Widgets | **Take.** Game archives ship editor panels as WASM modules. |
| General-purpose visual scripting | Unreal Blueprints, Godot VisualScript | **Skip.** Godot removed VisualScript in 4.0 for lack of use; Blueprint's per-node VM is the per-call cost this engine exists to avoid. Domain graphs above cover the designer use cases. |
| Scene tree / actor object model; editor and game in one process | Godot nodes, Unreal actors | **Skip.** Conflicts with ECS-as-truth; a separate editor process also survives game crashes. |

**Proposed shape:** `ae-editor` is a separate executable that attaches to a running client or headless sim.
Shell and dev panels in `egui` (fast to build, suits inspectors); in-game UI stays RmlUi. Everything the
editor shows is derived from the schema, the catalogue, the schedule and the frame graph descriptor.

**What this costs the core plan now** (cheap early, expensive to retrofit):
- M1: editor-hint metadata in the schema descriptor; hierarchical tags; dev commands as a synced command class.
- M3: frame graph descriptor loadable from a data file as well as built in code.
- M6: catalogue and map files in a stable, round-trippable text form; import settings + hash-keyed cache.

Editor milestones (after the core they depend on):
E1 (after M1) inspector, archetype browser, schedule graph, tick step/pause ·
E2 (after M3) live frame graph view, profiler via Tracy ·
E3 (after M6) catalogue grid editor, multi-client play-in-editor, replay scrub ·
E4 (after M7) map editor · E5 material, effect and state-machine graphs.

## Workspace (`/home/ak/AssembleEngine`)

`crates/`: `ae_abi`, `ae_ecs`, `ae_wasm`, `ae_sim`, `ae_jobs`, `ae_nav`, `ae_terrain`, `ae_map`, `ae_spatial`,
`ae_sensors`, `ae_view`, `ae_framegraph`, `ae_render`, `ae_render_abi`, `ae_vfs`, `ae_catalog`, `ae_net`,
`ae_replay`, `ae_save`, `ae_ui` (RmlUi FFI), `ae_inspect`, `ae_profile`, `ae_bench` · `sdk/`: `ae-sdk`, `ae-sdk-macros`, `ae-render-sdk` ·
`bins/`: `ae-client`, `ae-headless`, `ae-relay`, `ae-tool`, `ae-editor` (optional) · `games/sandbox` · `spikes/`.

Lift map (key sources): `ecs_sim/src/engine/{abi,flecs,schema,schedule,preentity,hash,publish,presentation,stage}.rs`,
`ecs_sim/vendor/*`, `ecs_sim/guest/src/lib.rs`, `ecs_sim/src/render/{frame_graph/*,executor,thread,device,swapchain,memory,pipeline}.rs`,
`pathfinding/src/{nav/*,avoidance,collision,terrain,par,workers,math}.rs`, `map-rendering/src/{leb,heightmap}.rs` + shaders,
`RedAlert-graphics/shaders/*`. `guest.rs` is rewritten (thread-local bound). Not lifted: pathsim `sim.rs`/`app.rs`
wholesale, `vkflecs-baseline/src/render_graph.rs`, RedAlert C++ host code.

## Roadmap

| M | Scope | Exit criteria / verification |
|---|---|---|
| **M0** Workspace + spikes | `git init`, workspace skeleton, flecs vendor + patch building, frame graph crate, `ae_nav` without macroquad, `ae_profile` (Tracy feature) + `ae_bench` + `bench-diff`, risk spikes (below) | `cargo test --workspace` green incl. lifted 22 frame-graph tests; spike reports with numbers; first benches (frame graph bake, nav) recorded and diffable |
| **M1** Parallel sim core | seam, column arena, worker instances, layered scheduler, command merge, events, cadences, schema section, SDK derive, digest, `ae-headless` | sandbox game at 10k/50k entities: identical digests at 1/2/4/8/16 workers and debug vs release; ≥5× on 8 cores. `ae-headless --game sandbox --units 50000 --ticks 3000 --workers N --digest` |
| **M2** Native services | `ae_nav` via columns, movement classes incl. water/hover/amphibious, `ae_spatial`, terrain patches | 10k units cross a map with water near pathsim's tick budget; golden-hash tests; `ae-tool navbench --scaling` |
| **M3** Render module + terrain | depth/MRT, declaration ABI, reflection, instance streams + GPU culling, `terrain_cbt` with reusable material tilesets (orientation + pixel-normal blending), threaded recording | `ae-client` shows terrain + 10k units from M2; terrain tileset demonstrated on an existing heightmap and documented; topology hot-reload without restart; validation layers clean |
| **M4** Copy layer + fog | `ae_sensors`, rule executor, ghosts, radar degradation, interpolation, fog pass | two observers see different worlds; property test: no hidden field reaches a view; view build < 2 ms at 10k |
| **M5** Lockstep + replay | `ae_net`, `ae-relay`, replay verify, desync bisect, cross-platform CI | 4 clients with injected latency/loss in sync 30 min; replay digests equal on Linux + Windows; injected desync localised |
| **M6** Content + mini-game | VFS, catalogue, `.aemap`, command queues, factories, economy as components, damage reductions | playable 2-faction skirmish defined only by an archive; bot-vs-bot headless golden digest |
| **M17** Formatting + lints (next stage) | decided: rustfmt with `max_width = 120`, a pinned toolchain, one mechanical reformat, clippy warnings fixed, a tracked pre-commit hook that gates both | `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` clean; a commit that breaks either is refused; pinned digests unchanged |
| Later | **M11** glTF models + skinning · **M12** lights + HDR scene target (medium) · **M13** HDR output (lower) · **M14** unit scripts (in design) · **M15** fully bindless resources (medium) · **M16** RedAlert-graphics survey and port (water first) — see "Planned milestones M11–M16" · M7 water + shadows + particles + decals · M8 RmlUi + widget sandbox · M9 AI hosting + save/load + snapshot late-join · M10 tooling, second nav level, async compute, runtime multi-module sim | |

### M17 — Formatting, lints and the commit gate (next stage)

The workspace has neither a formatter config nor a linter config, and no CI. This stage adds both
gates locally, before more code lands: one canonical format, clippy clean, and a pre-commit hook
that refuses a commit which breaks either. It is the next stage, ahead of M8–M10 and the remaining
M11–M16 work.

**Format: rustfmt, one config, no alternatives.** Decided:
- `cargo fmt` / rustfmt only (no dprint, no custom formatter, no nightly-only options), so the
  command is the one a Rust developer already runs and its output is defined by the toolchain.
- A workspace `rustfmt.toml` with `max_width = 120`; every other option stays at its default.
  Rationale: the existing hand-wrapped code is written mostly at 120 columns or less (69,481 lines
  of Rust; 4,542 lines over 100 columns, 2,348 over 120), so 120 makes the one-time diff the
  smallest and matches the style already in the tree. Defaults elsewhere mean no configuration
  drift and no bikeshedding. rustfmt at its default width proposes 3,765 hunks in the current
  tree; re-measure with 120 before the mechanical commit.
- `edition = "2024"` comes from the workspace, not a duplicated setting.
- A pinned toolchain, `rust-toolchain.toml` (channel 1.98.1, components `rustfmt`, `clippy`,
  targets `wasm32-wasip1-threads` and `wasm32-unknown-unknown`), because rustfmt's output changes
  between releases. The hook, a developer's shell and CI then use the same binary.
- Not formatted: `vendor/` (the patched flecs copy; formatting would fight the patch files),
  `target/`, and generated `.dot`/SPIR-V. Formatting does include the guest crates that are not
  workspace members (`games/*`, `render/*`, `spikes/*/guest*`, `crates/ae_sim/tests/guest`),
  through their own manifests.

**Lints: clippy clean, with determinism as the veto.** The target is
`cargo clippy --workspace --all-targets -- -D warnings`. Measured starting point (this machine,
2026-11-21): the run fails on clippy's deny-by-default correctness lints, `approx_constant` in
`ae_render/src/terrain/heightmap.rs` (eight occurrences) and `ae_spatial/tests/brute_force.rs`
(one); because it stops there, the warning count is not yet known for the whole workspace (the
crates that built report 17 warnings in `ae_ecs` and 23 in `ae_nav`, with their test targets).
Warnings are fixed in reviewable batches, grouped by lint rather than by file, and every batch is
checked by `cargo test --workspace` and the pinned digests.

- **A clippy fix must not change bits.** `approx_constant` is the clear example: replacing
  `0.7071` with `FRAC_1_SQRT_2` changes the generated heightmap and therefore captures and goldens.
  Where the suggested fix would change a digest, a golden or deterministic content, keep the value
  and put a scoped `#[allow(clippy::…)]` with the reason next to it. `suboptimal_flops` (it
  suggests `mul_add`, which is FMA) and similar float suggestions are reviewed by hand in the
  sim and nav code, where the determinism rules already forbid FMA and transcendentals outside
  `libm`. An `allow` is always scoped to the item and carries a comment; crate-level allows need a
  written reason in the commit.
- No lint is enabled beyond clippy's defaults in this stage, except `-D warnings`; pedantic and
  nursery lints are a later decision, after the tree is clean.
- Guest crates are formatted but not clippy-checked here (they build for wasm and some have their
  own manifests); their clippy check joins the guest build step when CI exists. That gap is
  accepted for now.

**The hook: a tracked `.githooks/pre-commit`, installed once.** Git does not clone hooks, so the
repo carries them in `.githooks/` and the setup is one idempotent command,
`git config core.hooksPath .githooks`, documented in the README's build section and in CLAUDE.md.
`ae-tool hooks install` prints and sets it (and is safe to run twice). The hook:
1. Collects the staged paths. A commit touching no Rust, manifest or build input returns at once,
   so a docs-only commit stays instant.
2. With any staged `.rs` file, runs the formatter check over the workspace and the guest crates.
   It fails with the file names and `cargo fmt --all` as the fix.
3. With any staged `.rs`, `Cargo.toml`, `Cargo.lock`, `build.rs` or `.cargo/` file, runs
   `cargo clippy --workspace --all-targets -- -D warnings` and prints the first diagnostics.
   Warm, incremental runs reuse `target/`; the first cold run after a dependency change is the
   expensive one, and the hook's time is measured in step 3, not guessed. If a warm run is too
   slow to live with, the hook narrows to the touched packages (`cargo clippy -p …`) and CI, when
   it exists, runs the full one.
4. Never edits the index or the working tree, and has no skip environment variable: `git commit
   --no-verify` is git's own escape hatch for a work-in-progress commit, and the stage documents
   that such a commit must be re-run through the hook before it is shared. The hook is a local
   guard, not a security boundary.

Partial staging is the known rough edge: the formatter and clippy read the working tree, not the
index, so an unstaged edit elsewhere can fail the check. The message says which command to run
(`cargo fmt --all`, or stage all with `git add -A`); no attempt is made to format a temporary
index copy, because the simple check is the one people can reason about.

**Steps** (each its own commit):
1. `rust-toolchain.toml`, `rustfmt.toml` (`max_width = 120`) and the guest-format wrapper; record
   the hunk count at 120.
2. The mechanical reformat only, no behavior change; add the commit to `.git-blame-ignore-revs`
   and say so in the commit body.
3. Fix clippy: first the two `approx_constant` sites (with scoped allows where the bits are
   pinned), then the remaining warnings; record the counts per crate before and after and measure
   a warm clippy run.
4. `.githooks/pre-commit`, `ae-tool hooks install`, README and CLAUDE.md, plus `docs/PLAN.md`
   progress.
5. Verify: a deliberately misformatted staged change is refused; a warning-producing staged change
   is refused; a clean commit passes; a fresh clone with the hook installed passes on docs-only and
   on code commits.

**Exit criteria:**
- `cargo fmt --all -- --check` exits 0, guests included, from a fresh checkout.
- `cargo clippy --workspace --all-targets -- -D warnings` exits 0.
- The hook refuses both a misformatted and a warning-producing commit, and does not fire on a
  docs-only one.
- `cargo test --workspace` is green, and the pinned digests (`ae_sim --test sandbox`, the crossing
  and skirmish goldens, the replay tests) are unchanged: formatting cannot move bits, and any
  clippy fix that would is not taken.
- README and CLAUDE.md state the toolchain pin, the format, the two commands and the hook setup.

**Not in M17:** CI (still none; the hook is local), pedantic/nursery lints, formatting of
`vendor/`, and clippy on the wasm guest crates. These are revisited when CI exists.

### M0 findings that shape M1 (details and numbers: `spikes/README.md`)

- Proven: N guest instances over one shared memory, each with a host-assigned shadow stack and TLS
  block, running in place over real flecs columns allocated from a host arena inside that memory;
  bit-identical to native at 1–16 workers, 6.4× at 8 workers.
- The serial structural phase is the next bottleneck. On 1M entities a real table change (despawn,
  or adding a component to a random entity) costs about 0.46 µs; spawning straight into the final table
  costs about 0.1 µs. About 7,000 changes per tick take 2.6 ms. M1 adds enable-bit columns for
  high-frequency toggles (no table change at all), spawns into final tables, and applies structural
  commands batched by source and destination table.
- Every sim module instance must get its stack pointer set before its first call; a module that is not
  the heap owner must route `malloc` and friends to the host. Both become SDK and loader rules.
- Nav lift is bit-faithful to pathsim (golden digests), also when compiled to WASM.
- Release watchdog is a host thread, not fuel or epochs (see Assumed defaults).

### M1 progress

Done and tested (`cargo test -p ae_sim`, `cargo test --release -p ae_sim --test parallel_commands`):
- `ae_ecs::arena`: deterministic column allocator over a host-claimed range of the shared memory.
- `ae_wasm`: deterministic wasmtime config, shared memory, page claims, N instances with private
  stacks and TLS (instance 0 owns the heap).
- `ae_sim::command`: work ranges cut in table order, per-worker command buffers keyed `{system, range,
  seq}`, merge by key, serial apply with direct-to-table spawns. 100k entities, 30 ticks of a system
  that despawns, spawns, adds and removes components from parallel ranges: identical digest at 1, 2, 4
  and 8 workers, equal to a native reference; merging by worker instead of key gives a different digest
  on every run (negative control).
- `ae_abi` + `sdk/ae-sdk` + `sdk/ae-sdk-macros`: `#[derive(Component)]` (also `#[component(event)]`),
  `#[system(write(..), read(..), emits(..), reads_events(..), after(..), serial, every(n, phase) |
  staggered(n) | once)]` and `schedule!(..)` write self-delimiting records into the `ae.schema` custom
  section. Stable ids are explicit or FNV-1a of the name; field records carry offset, size, kind and
  `hashed`/`synced`/`saved` flags. The host never relies on record order (the linker's choice): it keys
  components by id, systems by name, and takes system order from the one `schedule!` record. Each
  system is the export `ae_sys_<name>(desc)` with one uniform signature; the host writes a `RunDesc`
  (columns, entity ids, tick, stagger filter, event spans, write mask) into per-worker scratch.
  Guest imports use stable ids; the host maps them, checks value sizes, and rejects events emitted
  into streams the system did not declare.
- `ae_sim::schedule`: registration order refined by `after`, then an edge for every write/write,
  write/read and read/write overlap on a component or event stream, pointing along that order; layer =
  longest path. Layers are a function of the declarations only, so which layer's commands a system sees
  is game semantics, not timing.
- `ae_sim::events`: events are recorded under the emitting range's key and merged by key at the layer
  end. Each reader has a cursor: when it runs it gets every event appended since its last run, so every
  reader sees every event exactly once, also at `every(n)` cadence. Held events and cursors are in the
  digest. Deviation from "double-buffered per tick" above: events live until every reader has seen
  them, so a reader scheduled before its emitter gets them on its next run, and a slow-cadence reader
  misses nothing.
- Cadences: `every(n, phase)`, `staggered(n)` (rows whose entity index `% n == tick % n`, filtered by
  the SDK's `run.rows()`), `once` (tick 0, setup).
- `ae_sim::crew`: persistent workers, each owning one instance; the caller is worker 0. A layer is one
  batch of jobs (a range, or all ranges of a serial or small system) claimed from a shared counter;
  workers spin briefly before sleeping, and no more are woken than there are jobs.
- `ae_sim::digest`: order-independent xxh3-128 row sum (from `hash.rs`) over hashed fields only, field
  by field, so padding never reaches it; per-component parts for desync reports.
- `ae-headless`: `--game sandbox --units N --ticks N --workers 1,2,4,8,16 [--digest] [--fuel]
  [--param-hi N]`; fails on digest mismatch. `ae-tool bench sim-dispatch`; baseline recorded.
- Tracy zones per tick, layer, job (system name) and merge/apply; plots of commands and events.

Exit runs, sandbox game (5 systems in 4 layers; per row a small potential-field steering step on a
quarter of the units, integration, wear with death events, despawn and respawn; a serial census reads
the death events into a singleton). Ryzen 9 5900X, 3000 ticks, release:

| Units | Churn | Digest equal 1/2/4/8/16 w, dev = release | 1 w mean tick | 8 w mean tick (speedup) | 8 w median | 8 w system phase | 16 w mean |
|---|---|---|---|---|---|---|---|
| 50k | default (units live ~30 s) | yes | 1.35-1.40 ms | 0.27 ms (4.97-5.18×) | 5.6-5.9× | 5.7-5.9× | 6.1-6.3× |
| 50k | stress (~6 s, ~240 deaths/tick) | yes | 1.54 ms | 0.40 ms (3.8×) | 3.5× | 5.5× | 3.5× |
| 10k | default | yes | 0.28 ms | 0.08 ms (3.6×) | 4.1× | 4.0× | 4.4× |
| 10k | stress | yes | 0.31 ms | 0.11 ms (2.8×) | 4.1× | 3.8× | 3.1× |

The 50k default row spans four runs in both worker orders (the first world in a process pays first-touch
page faults, so 1 worker run first is slightly slower). Fuel metering (`--fuel`) gives the same digest
as unmetered runs, and so do range sizes of 100, 256 and 1024 rows. Tick 0 (spawning every unit,
serial) takes 25-37 ms at 50k and is excluded from the mean.

The churn setting was chosen after the first measurement: the sandbox originally turned every unit over
in about 6 s and reached 3.8× at 8 workers. That rate is now the `stress` row; the default (about 30 s)
is what an RTS with heavy fighting looks like, and is the row that meets the target.

What the numbers say: the ≥5× target is met, just, at 50k with realistic churn (mean tick after setup;
whole run including setup 4.8-5.1×), and only there. At 10k the
whole tick is under 0.3 ms on one worker, so per-layer dispatch (four wake-ups and merges per tick)
dominates. Under stress churn the serial structural apply (about 0.25 ms on a wear tick, ~1100
commands) is the limit, as the M0 spike predicted; system-phase scaling stays at 5.5×.

Guest heap from parallel systems (`cargo test --release -p ae_sim --test guest_alloc -- --nocapture`,
`games/alloc`): two parallel systems in one layer allocate on every row (vectors grown by push,
sorts, B-tree maps, strings, and blocks held until the row's next run, so most are freed by another
instance; one block in 128 is 64-256 KiB, so the heap grows from several instances at once). Every
result is checked against a stack computation in the guest and every held block's contents before its
free. 4000 rows, 90 ticks, 128-row ranges: no failed check, identical digests at 8, 1, 2 and 4 workers,
and the heap grew under 8-way load (263 -> 280 MiB) and then stopped.
- Found on the first run: under contention wasi-libc's malloc lock calls
  `wasi_snapshot_preview1::sched_yield`, which the host did not define, so the guest trapped. The host
  now provides it (a thread yield; counted, `ae_sim::imports::sched_yields`).
- Correct, but not fast: the one heap lock serialises allocation. 23 million lock waits at 8 workers;
  wall time 5.75 s at 1 worker, 3.25 s at 2, 2.01 s at 4 and 3.78 s at 8 (8 is slower than 4). The
  probe is far more allocation-heavy than typical system code, but per-worker bump arenas (planned
  above) are now a measured need, not a precaution, for any game that allocates per row.
- The guest heap is not reset between worlds of one runtime (worlds share the module instance);
  blocks a world leaves allocated stay allocated.

Not done: per-worker bump arenas for parallel systems; enable-bit columns; structural apply batched by
source/destination table (needs a design that keeps the key order's result); singletons and
NOT/OPTIONAL terms in the SDK; editor hints in the schema; save→load→continue; cross-platform digest
CI; per-system wall time inside a parallel layer.

### M2 progress

Started before M1 finished, on request: the services were built and tested standalone, with columnar
APIs, then wired into the sim schedule as host systems (see "In the sim" below).

Done and tested (`cargo test -p ae_nav -p ae_spatial`; benches `ae-tool bench nav-sim`, `bench spatial`):
- `ae_nav::reference`: pathsim's whole simulation (`sim.rs`, `commands.rs`, `replay.rs`) lifted
  unchanged, plus navbench as `ae_nav`'s `navbench` binary. State hashes pinned against the original
  crate at 1 and 4 threads; navbench at 10k units gives the original's hashes and counters exactly.
- `ae_nav::service::NavService`: the columnar service carved from the reference. Units are added and
  removed by entity id; `step(BodyColumns)` reads position, velocity and radius from caller columns and
  writes position and velocity back, so the ECS owns bodies between ticks and the service keeps only
  navigation bookkeeping (cohorts, slots, fields, jobs). A body moved by the caller is taken as placed
  there. Driven with the reference's scenario it gives the reference's state hashes; with units added
  and removed while moving it is equal at 1 and 4 threads. `take_terrain_changes()` reports each terrain
  version when it becomes authoritative, with its dirty cell rectangle (for render and the copy layer).
- Movement classes and water: a water level per map, `Terrain::height` as the standing surface (the
  water surface over water) plus a `water_depth` layer that exists only on wet maps, and a
  `Medium {min_depth, max_depth, hover}` per movement profile, used by passability, slope (surface
  movers on the surface, ground movers on the bed), climb cost and collision walls (`Collider::step_in_media`: a ship collides with the shore, a tank with deep
  water). `WaterMode::Surface` generates lakes with banks and a river with two fords and a causeway
  between the spawn plaza and the goals. `water_profiles()`: scout, heavy (ford 2 m), ship (draft 3 m),
  hovercraft, amphibious. Dry maps and default profiles are bit-identical to pathsim (all goldens pass
  unchanged).
- `ae_nav::scenario`: the land and water scenarios as library code, shared by tests and benches.
- `ae_spatial`: uniform grid by counting sort; batch radius, box, nearest-k and swept-circle queries
  answered as CSR hit streams in query order; hits in a fixed order, equal at any thread count; checked
  against brute force including items outside the bounds and on cell edges.

Measured (this machine, 24 threads, release; `bench/results/baseline/{nav-sim,spatial}.json`, recorded
on the uncommitted M2 tree on top of `349fe80`):
- 10k units, 8 workers, 600 ticks: land 4.0 ms median tick (pathsim's navbench on the same machine:
  4.4 ms, of which 0.66 ms is its render snapshot, which the service does not make); water map with
  all five classes 3.9–4.1 ms median, 4.8–5.8 ms p95 over two runs (tick spread ±25–35%). Equal
  hashes at 1/2/4/8 workers. 1 worker: 16.7 ms. State hashes of a lake crossing and of the 768²
  generated water map are pinned in `tests/water.rs` (identical in dev and release).
- Dry-map cost of the water support: none measurable (hierarchy build 3.76 s vs 3.73 s, flow field
  244 µs vs 244 µs, against the pre-M2 commit on the same machine). A first version that computed
  surface heights on the fly in the edge cost made hierarchy and field builds 10–16% slower, hence the
  surface-plus-depth layout.
- 10k water crossing run to 4,500 ticks (150 s): 7,823 arrived (all 500 ships, 6,059 of 6,241 scouts,
  659 of 773 hovercraft; heavies still queuing at the causeway), no unit outside its medium on any tick,
  1,384 hover/amphibious units through deep water. Median tick 3.4–4.0 ms throughout, p95 up to 6.6 ms
  as crowds pack at the goal.
- Spatial, 50k units in squads, 10k queries: rebuild 0.45 ms; radius-160 queries (83 hits each)
  12.5 ms at 1 thread, 2.0 ms at 8; nearest-8 9.7 / 1.55 ms; 360-unit sweeps 6.0 / 0.96 ms.

Deviations from the plan, deliberate:
- The service integrates and collides bodies itself ("Body in → Body out") instead of returning only
  `PreferredVel`: pathsim's steering, ORCA and collision substeps are one coupled loop, and splitting
  them would lose the bit-equality with the reference that makes the carve trustworthy. The ECS is
  still authoritative between ticks. Splitting out `PreferredVel` stays possible later.
- `NavService::state_hash` still hashes bodies (as the reference does, for the equality check); in the
  engine digest the body part is redundant with the ECS digest and the nav part is the service's part.
- One behaviour change from pathsim, in the service only: when arrival slots run out, a last pass drops
  the line-of-sight rule (same component still required), so ships in a winding river get slots
  instead of giving up. Counted in `NavRuntimeStats::fallback_slots`: 0 in the two scenarios checked
  against pathsim (asserted in `service_vs_reference`; 0 in `nav-sim` land), 55 in the 10k water run.
  pathsim's crowd and chokepoint navbench scenarios have not been driven through the service yet.
- Terrain stays in `ae_nav::terrain`; the planned `ae_terrain` crate split waits for map files (M6) and
  the render heightfield (M3; done, see "Terrain edits on screen").

In the sim (`cargo test --release -p ae_services --test crossing -- --nocapture`; `ae-headless --game
crossing --nav water --compare`):
- Host systems (`ae_sim::host`): native code in the sim schedule. A module places them with
  `schedule!(.., host(nav), ..)`; a `HostPlugin` given to `Runtime::with_hosts` provides them, checks the
  module's layout of every engine component they touch against the engine's own record, and creates
  fresh per-world state. A host system declares query terms, created components, emitted and read
  streams, `after` and a cadence, and is ordered by the same conflict rules as module systems; its
  commands and events carry its schedule position and merge with everyone else's. It runs on the
  coordinating thread while the crew's helpers start on its layer's module systems (which cannot
  conflict with it; its query is read before they start), then that thread joins the jobs left
  (`Crew::run_beside`; `RuntimeConfig::overlap_hosts = false` runs it after them, same results). It sees
  its whole query at once and may use its own threads. State it keeps outside components is a digest
  part. `HostPlugin::instantiate` gets the module's schema, including its data records.
- Engine components (`ae_sdk::nav`, opt-in cargo feature `nav`, fixed ids in `ae_abi::engine`): `Body
  {x, y, vx, vy, radius}`, `NavAgent {max_speed, profile, mode}`, `Goal {x, y, group}` and the events
  `MoveOrder {entity, x, y, group, kind}` (`order::MOVE`, `STOP`, `ROUTE`, `FORMATION`; constructors
  `MoveOrder::move_to`, `stop`, `route`, `formation`), `OrderPoint {group, index, x, y}` (a route's or
  formation's guide points, at most 64; a point may be sent more than once, by every unit of the group
  say, if the copies agree), `TerrainEdit {x0, y0, x1, y1, set, blocked, height, cost}` (a cell
  rectangle; `set` flags pick the fields) and `TerrainChanged {version, x0, y0, x1, y1}`. Opt-in because the derive writes a schema record for every type it
  compiles: unconditionally linked, they were registered in the sandbox too and shifted its entity ids
  (and its pinned digest).
- `ae_services::nav`: `nav_spawn` (tick 0) spawns a map plan's units; `nav` (every tick) gathers the
  `Body`+`NavAgent` rows, sorts them by entity, reconciles the service with them (a merge against the
  units' sorted entities, skipped when nothing joined or left: removals in one batch, new units in
  entity order), turns the `MoveOrder` and `OrderPoint` events since its last run into group
  commands (`group_orders`: one per `group`, id = group, in order of first event, kind and point of the
  first order, unit ids ascending; mixed kinds in a group, points without orders in the same tick, gaps
  and disagreeing copies are errors), steps the service on its own pool and writes
  positions, velocities and modes back. The `TerrainEdit` events since its last run become one terrain
  patch (event order, a later edit of a cell wins; rectangles clipped to the map; out-of-range values are
  errors and stop the tick), queued before the step: it lands with its nav repair a fixed sim-time delay
  later (2 ticks at 30 Hz), buried units are moved to the nearest open cell, and `nav` emits one
  `TerrainChanged` per landed repair (what render and the copy layer will refresh). One repair runs at a
  time; patches queued meanwhile land together with the next, under one event whose rectangle covers
  them all and whose version is the terrain's after them (versions count patches, so they can skip). The game decides when units move (`games/crossing`: orders at
  the tick in its parameter). `ae_nav::scenario` plans are now data (`Plan`), so a map's units and
  orders can be spawned by the sim or added to a standalone service.
- `ae_services::spatial`: rebuilds the grid from every `Body` in its layer and publishes it to module
  systems, which call `ae_sdk::spatial_radius` (import `ae_spatial_radius`, answered on the calling
  worker by `Grid::radius_one`, no thread pool). The index is only replaced in its own layer, so a
  query's answer is deterministic wherever the querying system sits; `after(spatial)` gets this tick's.
- Checks: 1,000 units on the 768² water map for 300 ticks reach the same nav state hash and
  bit-identical bodies as the same plan driven through the standalone service, and the same pinned world
  digest (nav state included) in dev and release and at 1/2/4/8 workers; 250 guest spatial queries equal
  brute force. A second world exercises units changing under the service: orders at tick 20 (when every
  row also moves to another table, as the game adds a component), and at tick 150 a game system
  despawns a seventh of the units and spawns replacements in the same tick (143 removals and 143 joins
  with recycled entity ids). The same world edits the terrain on two ticks (`quake`): at tick 100 rock
  5×5 cells inside 17×17 cells of slow ground around every unit whose entity index is a multiple of 97
  (2,431 cells), at tick 101, while that repair runs, ground raised to 90 m in 9×9 cells around others
  (718 cells; land in lakes, hills on land). They land as versions 2 and 3 at ticks 102 and 104, 109
  units are moved out, and `landed` records two `TerrainChanged` whose rectangles are the two patches'.
  At tick 60 `reorder` gives every unit a new order by its group: a stop (157 units), a route
  through two guide points (793) and a formation on a line (50), each unit sending its group's points.
  Its bodies and nav state equal a standalone service driven through the same commands, removals,
  additions and patches at the same steps, at 1 and 8 workers. (347 units end unreachable: the terrain
  edits cut them off from their goals; without the reorder the count is the same.) Not covered yet: un-blocking cells.
- Host systems run whole tables, so a `Staggered` host cadence is rejected.

Exit run, 10k units on the 2560² water map, 600 ticks (`--compare`: the same plan through the
standalone service at the same thread count; nav state equal at every count; digests equal). Medians,
ranges over two runs:

| Workers | Sim tick | Nav host system | Standalone service | Spatial | Module systems (orders, crowding, churn) |
|---|---|---|---|---|---|
| 1 | 17.3-17.7 ms | 16.6-16.9 ms | 16.4-16.6 ms | 0.19-0.20 ms | 0.51-0.53 ms |
| 2 | 10.1-10.5 ms | 9.6-9.9 ms | 9.2-9.4 ms | 0.20-0.23 ms | 0.33-0.37 ms |
| 4 | 6.4-6.5 ms | 6.0 ms | 5.4-5.7 ms | 0.20-0.22 ms | 0.23-0.25 ms |
| 8 | 4.62-4.75 ms | 4.20-4.28 ms | 3.89-3.96 ms | 0.21-0.22 ms | 0.23-0.25 ms |

Run to 3,000 ticks (100 s) at 1 and 8 workers: same digest, nav state equal to the standalone service;
8 workers 4.64 ms median (nav system 4.19 ms vs 3.86 ms standalone), p99 13.0 ms.

The nav step itself costs what it does standalone (median 4.0-4.1 ms at 8 workers vs 3.84-3.96 ms over
two runs); gather, reconcile, orders, columns and scatter add 0.13 ms. A first version looked units up
in the service's entity map and binary-searched rows per unit: 0.6 ms more. A 32-unit spatial cell
made the rebuild 0.4 ms (it clears and prefix-sums one counter per cell, 410k cells on this map); 64
gives 0.2 ms and identical hits. Building a world (terrain, hierarchy) takes 2.5 s at 8 threads.

Re-measured after these changes (host systems touch the tick loop, the crew and the scenarios): M1's
sandbox at 50k units, 3000 ticks, 8 workers: 5.35-5.45× mean tick after setup in both run orders (was
4.97-5.18×), same digest. `bench-diff` against the baselines: `spatial` 10/10 unchanged, `nav-sim` 37
unchanged and 3 improved, `sim-dispatch` 43 unchanged, 2 improved, 1 flagged (`setup_tick_8t` under
stress churn 22.6 -> 25.0 ms: a single-sample metric, the tick that spawns 50k units, which varies
25-37 ms between headless runs) and the 16-thread rows absent (the baseline was recorded with
`--threads 16`).

Terrain edits added the `quake` and `landed` systems and the `Landed` component to the crossing game,
which shifted its entity ids: the pinned digest was re-pinned (bodies and nav state still equal the
standalone service). With no edit they cost nothing measurable: 10k units, 600 ticks, 8 workers, 4.70 ms
median (nav system 4.22 ms vs 3.91 ms standalone), 1 worker 17.46 ms; digests and nav state equal. The
unit count in the game parameter is now its low 24 bits (`NavPlugin::units`); crossing uses bits 24..32
for the edit tick. Order kinds changed `MoveOrder`'s layout (32 bytes) and added `OrderPoint`, so
the digest was re-pinned again; crossing's order tick is now bits 32..40 and the reorder tick bits
40..48. The exit run after both: 8 workers 4.62 ms median (nav system 4.16 ms vs 3.89 ms standalone),
1 worker 17.65 ms, digests and nav state equal.

Movement classes from game data: a module declares named data for engine services in `KIND_DATA`
schema records (`u32 name_len, u32 0`, name, payload; `Schema::data`). `ae_sdk::movement_classes![..]`
writes `nav.classes`: per class (64 bytes, built at compile time) a name, clearance in cells, max slope,
nominal radius and speed (what map spawners use), a water depth range (`ANY_DEPTH`: no limit) and hover.
`NavPlugin` builds the service's profiles from it (`ae_services::nav::classes`, checked: unique names,
clearance at least the nominal radius needs and at most 16 cells, depths in range and ordered, slope
1..32766), or from the map's when a module declares none; the map's plan needs at least as many classes
as it uses (the scenario plans now take profiles: `water_crossing_plan_with`, `land_crossing_plan_with`).
`games/crossing` declares the water map's five classes; the test checks they parse to exactly
`water_profiles()` and that patching one (scout slope 1) changes the world's nav state within 30 ticks.
The record changes no digest.

Host systems beside module systems: a first version ran them on a scoped thread during the crew batch;
on the exit run it cost 0.1-0.3 ms a tick (a thread spawn per layer, and `spatial` slower on a cold
thread: 0.29 vs 0.22 ms), with nothing in crossing's layers to overlap. Running them on the coordinating
thread while the helpers start, then joining, costs nothing measurable there (8 workers, three runs each:
4.77-4.84 ms with, 4.72-4.85 ms without) and, with `crowding` moved into `spatial`'s layer (an experiment,
not kept: the game keeps `after(spatial)`), saves 0.24-0.34 ms a tick (4.37-4.38 vs 4.61-4.72 ms), same
digest. The crossing test checks equal digests with overlap off. M1's sandbox after the change: 50k
units, 8 workers, 5.70x mean tick after setup, digests equal. On 8 cores (`taskset -c 0-7`, 8 workers),
where nav's own pool and the crew's helpers compete, with a busy module system placed in nav's layer (an
experiment, not kept): overlap still wins, 8.73-8.93 vs 9.17-9.73 ms with a heavy one (nav itself
slows from 4.6 to 7.3 ms) and 6.37-6.44 vs 6.73-6.98 ms with a light one. So it stays on by default.
A unit joining the service must fit its class: `min_clearance_cells(Body::radius) <= clearance`.

Cost overlays (`NavService::add_cost_overlay(id, cells, cost)`, `remove_cost_overlay(id)`; in the sim
the `CostOverlay` event, `op` `ADD` or `REMOVE`): extra traversal cost on a cell rectangle, summed where
overlays meet, an open cell at most 254; blocked cells stay blocked. Unlike a cost `TerrainEdit` an
overlay is removed without knowing what was under it: the terrain keeps `base_cost` and `overlay` layers
(empty until the first overlay, like `water_depth`) and `move_cost` stays the effective cost, so the
edge-cost hot path is unchanged and maps without overlays keep every golden and pinned hash (`nav` and
`nav-sim` bench-diff: 51 unchanged, 0 regressed). An overlay change is a terrain patch (an overlay layer
on `TerrainPatch`) and lands with its nav repair; `TerrainChange::ground` and
`TerrainChanged::flags` (`changed::GROUND`) say whether the ground changed or only costs, so render can
skip cost-only changes. Overlays apply to every movement class: flow fields are shared between classes,
so a per-class overlay would need per-class costs (not done). Checked (`ae_nav/tests/overlay.rs`): add
then remove restores every cost exactly; 300 units on the 384-cell land map spend 2,169 unit-ticks in a
cost-200 overlay across their way against 19,203 in the same rectangle without it. In the sim, `zones`
adds a slow zone at tick 100 and removes it at 130; the churned world's terrain changes equal the
standalone service's one for one: (version 2, tick 102, ground), (4, 104, ground: the overlay and the
raised ground in one repair), (5, 132, costs only).

Formation facing: pathsim's curve formations put extra rows on both sides of the line, alternating, with
no front ("no notion of a front or a facing", pathsim's LIMITATIONS). `NavService::queue_formation_facing
(payload, toward)` gives one a front: along each part of the curve the first row stands on it and
further rows stack on the side away from `toward`. Bodies have no heading, so facing is about where
slots are, not which way units point. Without a facing pathsim's layout stays (the reference equality
and goldens are unchanged). In the sim: an `OrderPoint` with index `order::FACING`. Checked
(`ae_nav/tests/formation.rs`): 300 units on a 400-unit line, 140 slots ahead and 134 behind without a
facing, 0 and 274 with one; `reorder`'s formation faces north-east, and the churned world fails against a
reference that drops the facing.

These changes (`CostOverlay`, `TerrainChanged::flags`, `Landed::flags`, the `zones` system) re-pinned the
crossing digest again (nav state unchanged). The exit run after them: 8 workers 4.69 ms median (nav
system 4.22 ms vs 4.03 ms standalone), 1 worker 17.59 ms, digests and nav state equal.

M2's listed work is done. Left for later: per-class cost overlays, units turning to a facing (needs a
heading in `Body`), and the second hierarchy level (M10).

### M3 progress

Frame graph (`cargo test -p ae_framegraph`):
- Depth: `Role::DepthAttachment(LoadOp)` (tested and written, `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`) and
  `Role::DepthRead` (tested only, `DEPTH_STENCIL_READ_ONLY_OPTIMAL`); a sampled depth image uses
  `Sampled`. At most one depth role per graphics pass (`BakeError::DepthShape`); a depth-only graphics
  pass (a shadow map) is valid. Attachment roles now also check the image aspect (a colour image is not
  a depth attachment and the reverse).
- Several colour attachments per pass already worked (bound in declaration order); now tested on the GPU.
- Several writers per resource ("versions"), replacing the lifted single-writer rule
  (`MultipleWriters` is gone): in pass declaration order each write starts a version; a read depends on
  the latest write declared before it (or, with none before it, on the first write, so single-writer
  graphs bake exactly as before: all 22 lifted tests pass unchanged except the one that asserted the
  old error, which now asserts the order); a write depends on the previous write and on every read of
  the previous version. Reads declared before the first write join the first version's readers, so a
  second write waits for them too (`early_reads_precede_the_second_write`, which fails without that
  rule). A write declared after the last read of a resource is pruned like any unreachable pass. This is what lets terrain clear depth, units load and extend it, and decals test
  against it read-only.
- Wire format: `RoleKind::DepthAttachment = 15`, `DepthRead = 16`, appended, so older descriptors still
  decode and `VERSION` stays 1; `ImageSpec::depth_2d`, `PassBuilder::depth`, `depth_read`. A module's
  depth + continued-attachment graph decodes to the native one (`tests/descriptor.rs`).
- `ae-tool bench framegraph-bake` against `bench/results/baseline`: 36 metrics, 34 unchanged, 0
  regressed (bake +2 to +8%, under the threshold; barrier counts, physical resources and descriptor
  bytes identical).

`ae_render` (new; `cargo test -p ae_render --test gpu -- --ignored --test-threads=1`, needs a Vulkan
1.3 GPU and `VK_LAYER_KHRONOS_validation`): ecs_sim's `device`, `memory`, `swapchain`, `pipeline`,
`executor`, `thread`, `window` (SDL3 through `sdl3-sys`, as the plan says) and the demo graph as a test
fixture, lifted with module paths changed only. Not lifted: `view_renderer`, `packet`, `view` (they
draw ecs_sim's presentation views; M4's copy layer replaces them). Shaders in `crates/ae_render/shaders`
are compiled by `build.rs` with `glslc` (`GLSLC` overrides; a missing `glslc` fails the build with
that message). Additions: the executor binds a pass's depth attachment (and stencil when the image has a
stencil aspect), clears it from `Recorder::clear_depth` (default 1.0) and turns a first-use depth load
into a clear like colour; `GraphicsDesc` takes `color_formats` (one per attachment) and an optional
`DepthState {format, write, compare}`. GPU tests (RTX 3060 Ti, validation including synchronization
validation, any message fails the test): the four lifted ones, plus `depth_and_multiple_targets`: three
passes share one transient depth buffer over two colour targets (clear, load-and-extend, read-only
decal), and every expected pixel is checked in both targets over three frames, once with `D32_SFLOAT`
and once with `D32_SFLOAT_S8_UINT` (`ImageDesc::depth_2d` gives combined formats both aspects, the
pipeline declares the stencil format, the executor binds the stencil attachment). Zero validation
messages.

Open: `ae_render` and `ae-client` are workspace members with no feature gate, so `cargo test
--workspace` (and the headless CI targets, including the pending aarch64 check) now needs `glslc` and
SDL3 installed. Fix when CI is set up: `default-members` without them for headless runners, or
checked-in SPIR-V.

`ae-client` (new): SDL3 window plus `ae_render`'s render thread, for now showing the demo graph (triangle,
compute blur/invert, composite to the swapchain). `--seconds S` exits after S seconds; `--exercise`
resizes, minimizes and restores on a schedule; the exit status is 1 if validation reported anything.
`AE_VK_VALIDATION=1 ae-client --seconds 6 --exercise` (debug): 14,937 frames, 6 swapchain recreates,
0 validation errors or warnings. Environment variables renamed from `ECS_SIM_*` to `AE_*`
(`AE_VK_VALIDATION`, `AE_VK_SYNC_VALIDATION`, `AE_RENDER_TRACE`, `AE_RENDER_TRACE_EVENTS`).

Terrain (`ae_render::terrain`, lifted from `~/map-rendering`): the CBT/LEB subdivision refined on the GPU
against a screen-space error target (Dupuy's concurrent binary tree, conforming longest-edge bisection,
an error pyramid for the metric; see map-rendering's README for the method). `leb`, `heightmap` (PGM,
test-terrain generator, error pyramid) and `camera` are lifted unchanged with their CPU tests (the
camera gained `Copy`); the shaders are lifted with an include renamed. The renderer is rewritten around
the frame graph:
- Two passes: `terrain lod` (compute) runs the prototype's split/merge/reduce sequence with its own
  internal full barriers and declares the draw list and indirect arguments as storage writes;
  `terrain` (graphics) reads the draw list in the vertex stage and the arguments as an indirect buffer,
  with a colour target and a transient reverse-Z depth buffer (`D32_SFLOAT`, cleared to 0 through
  `Recorder::clear_depth`). The CBT buffers are owned by the renderer and shared by every frame slot
  (the subdivision evolves frame to frame); uniforms are per slot.
- Static data goes up through `ae_render::upload` (staging buffer, one-shot command buffer, waited).
- The device now requires `shaderClipDistance` (the LEB square is a power of two, clipped to the map)
  and enables `fillModeNonSolid` where present (wireframe).
- Additions to the prototype's shaders: a water level in the uniforms tints ground below it by depth
  (no water surface until M7), and an optional per-cell material texture colours the ground
  (bilinear between cell centres, slopes still turn to rock), a placeholder for the tilesets.
- The heightfield is the M2 crossing map's nav terrain (`ae_nav::scenario::water_crossing_terrain`,
  `land_crossing_terrain`: the terrain without hierarchy or units, equal to the service's, tested in
  `tests/water.rs`): ground = surface − water depth, corners averaged from the cells around them
  (unit test in `ae-client`); materials from `Terrain::material`. Heights: nav stores decimetres and
  never relates them to world units (its slope limits are raw decimetres per 8-unit cell: scout 135,
  heavy 95, ship 60), so render assumes 1 world unit = 1 m (factor 0.1, `--relief` on top). Under that
  assumption a scout's limit is a 59° step, a heavy's 50°. To settle when units get real scale.
- `ae-client` shows it: `--map water|land|generated|demo`, `--map-cells`, `--relief`, `--camera`,
  RTS controls (WASD/arrows, wheel, right-drag, Q/E, R/F, `+`/`-` sim speed, doubling with no
  upper limit (compute-bound), Pause pauses the sim and every animation clock, `[`/`]` detail,
  T wireframe, L LOD colours, G fog, C freeze LOD, P print camera, V v-sync, N render module
  flag, Z bloom); `--capture FILE.ppm` renders
  1280x720 offscreen without a
  window and prints GPU time per pass. The SDL window now reports wheel, drag, toggle keys and held
  movement keys (ecs_sim's F "full frame" key is gone).
- Pause (2026-11-20): the Pause key stops the sim thread and freezes the renderer's animation
  clocks: water time, decals and particles (`effect_time`), unit interpolation and the render
  module's time. The paused interval is skipped on resume (and readback holds its frame index),
  so nothing jumps or replays. GPU `pause_holds_the_water_clock`: 30 frames while paused are
  byte-identical to the pause frame, while an unpaused run of the same length has moved.
- Captures are `R8G8B8A8_SRGB` like the swapchain, so they show what the window shows; the
  prototype's fog and lighting make the 20 km overview look pale (G toggles fog), left as is until the
  tilesets redo the look. `ae-client` prints "validation off" instead of zero counts when the layer is
  not loaded (`GpuRenderStats::validation`).
- Checked: GPU test `terrain_renders_offscreen` (generated 513² map, 24 frames: refined beyond 64
  leaves, sky above terrain, zero validation messages). The water map (2560² cells, heights 1..150 m,
  water at 17 m): terrain 1.0 s and error pyramid 0.19 s to build; with validation, 0 messages in
  `--capture` and in an 8 s windowed `--exercise` run (7 swapchain recreates). GPU per frame at
  1280x720 (RTX 3060 Ti, release, profiling on): LOD pass 0.25–0.33 ms, draw 0.05–0.07 ms, at 700 to
  3,500 leaves for overview, close-up and grazing views; about 2,450 fps windowed over the whole map.
  These were printed only; the GPU bench below (`ae-client --bench`) now records them.

Units (the 10k-unit instance stream):
- End-of-tick capture in the sim core: `ae_sim::capture::CaptureSpec` names 4-byte fields by
  component and field (resolved through the schema, so `ae_sim` names no components);
  `Runtime::capture` copies them, from every entity with all those components, into reused
  field-major columns plus entity ids. It runs between ticks through an uncached flecs query:
  flecs gives every cached query an entity, so a cached capture query (and the old `count`/`rows`,
  now uncached too) created mid-run shifted the ids later spawns get. `ae_services/tests/capture.rs`
  runs the crossing game through orders, a reorder, a quake and churn, capturing after every tick,
  and gets the same final digest as without capturing (with a cached query it differs: checked);
  the captured words equal the components' bytes by entity.
- Render side (`ae_render::units`): an `InstanceStream` is a fixed set of 4-byte attributes per row
  (`x, y, prev_x, prev_y, heading_x, heading_y, radius, class`; M11 adds `clip, clip_time`), column by column; whoever feeds it
  maps its own fields onto them, so the renderer never sees the sim, and M4's view tables replace
  the feeder, not the renderer. `UnitFeed` holds the latest stream behind a mutex held only to swap an
  `Arc`: the producer publishes after each tick, the render thread takes the latest at `prepare`;
  neither waits for the other. Positions interpolate from `prev` to latest over the tick after
  publication.
- Five passes after the terrain LOD: `units reset` (transfer: instance count 0), `units cull`
  (compute: frustum test per unit, appends visible indices and counts them atomically), `units`
  (graphics, declared after the terrain draw: one indirect instanced draw of a 12-vertex dart built from `gl_VertexIndex`, sized by
  radius, turned by heading, coloured by class, lit and fogged like the terrain). The draw loads the
  terrain's colour target and depth buffer: the ordered-writer case the frame graph change was for.
  `units count` copies the indirect arguments to a host buffer per slot and `units count read`
  (a host-read root) makes it visible, so `prepare` reads the previous frame's visible count (status
  line, `--capture`).
  Instance data is one host-visible buffer per frame slot (fixed capacity, so the graph never
  rebakes on unit count), rewritten only when a newer tick arrives. Heights come from the terrain's
  height texture; classes whose medium is on the surface (ships, hovercraft) sit at
  max(ground, water level), the rest on the ground (amphibious on the bed). Units smaller than 7 px
  are drawn 7 px across, so armies stay visible from the overview.
- `ae-client` runs the crossing game (`--units`, default 10000; `--workers`, default half the
  cores up to 8) on a sim thread at the nav rate (30 Hz) on the same map and map size the terrain
  shows; after each tick it captures `Body.{x,y,vx,vy,radius}` and `NavAgent.profile` and builds the
  stream, matching rows to the previous capture by entity (sorted merge; new units start where they
  are; a standing unit keeps its last heading; unit-tested). `--capture` first runs `--ticks` ticks
  (default 300) synchronously and prints where each class is.
- Executor stats now carry the baked pass names beside the GPU times (the times were printed
  against a guessed order before; the bake puts the unit reset and cull before the terrain LOD).
- Checked: GPU test `units_render_over_terrain` (four units, one behind the camera: the visible
  count read back is 3, and 4 when the shader's frustum test is disabled, checked; each visible
  unit shows its class colour at its projected position, a floating one at the water level,
  nothing at a moved unit's previous position; zero validation messages). Water map, 10k units,
  release, RTX 3060 Ti: world built in 2.5–2.6 s; capture + stream 0.14–0.30 ms per tick; sim tick
  4.6–5.0 ms (capture runs) to 5.2–6.0 ms (beside a 2,500 fps render thread, 8 workers), so 30 Hz
  holds with one late tick (the first); windowed about 2,500–2,600 fps with or without units.
  With validation (synchronization validation included, the default): 0 messages in `--capture`
  on both maps and in 12 s windowed `--exercise` runs (11 swapchain recreates) with the sim
  running, so the per-slot instance rewrite every tick is covered.
- GPU per pass, 1280x720, profiling on, 60-frame offscreen captures over the 6k-unit army
  (`--camera 2900,17900,1400,0.4,0.85`, 8,606 of 10,000 units visible): unit cull 0.005–0.017 ms,
  unit draw 0.03–0.14 ms. These captures are noisy: terrain LOD measured 0.27–0.34 ms without units
  (last turn's 0.25–0.33 ms holds; the old labels happened to be right, as that graph bakes in
  declaration order) but 0.27–0.56 ms with them. The spread is not the units' drawing: with
  `--ticks 1` (5,168 visible) LOD is 0.27 ms, and the unrelated readback copy swings with it
  (0.17–0.94 ms); it follows the 1.5 s, 8-worker sim run right before rendering (GPU clocks, at a
  guess). The GPU bench (warm-up, more frames, recorded spread) has to settle such numbers.
- Found on the way: a struct holding both a `Runtime` and its `World` must declare the world first
  (fields drop in order, and the world's columns live in the runtime's memory); `World::drop` after
  the runtime crashed in flecs.
- Not done: the cull only tests the frustum (no occlusion, no LOD meshes); one draw for all
  classes. (Terrain edits now reach the render heightfield: see "Terrain edits on screen" below.)
  `ae-client` now builds `games/crossing` on first use
  (cargo and the `wasm32-wasip1-threads` target, as `ae-headless`); a sim failure is printed when
  it happens and shown in the title, and `--units 0` runs without the sim.

Unit culling tests (`cargo test -p ae_render --test gpu -- --ignored`): `ae_render::units::cull_margins`
is the cull shader's test on the CPU (same sphere, minimum-size enlargement and planes).
`units_cull_matches_the_cpu_reference` scatters 3,000 units (some 6 km away and under a unit across)
around an oblique camera, leaving out those within half a world unit of a plane, and the GPU keeps
exactly the units the CPU keeps (416). The set includes units culled by each of the four side planes
(620+ each), 65 kept with their centre outside the view, and 23 kept only because they are drawn at the
minimum pixel size. A stream longer than the capacity uploads and culls `capacity` rows. Checked by
breaking the shader: testing the centre only (351 kept) or ignoring the minimum size (387) both fail it.

Render modules (the M3 exit item "topology hot-reload without restart"):
- Pass library (`ae_render::library`): the terrain and unit passes, readback copy, host read and
  present are library passes by name (`terrain.lod`, `units.draw`, ...), each with the library
  resources it needs and the roles it uses them in (`LibraryPass::requires`), and for the draws one
  colour and one depth attachment in the formats the library's pipelines target. Library resources
  (`terrain.leaves`, `units.instances`, `readback.buffer`, ...) belong to the host, which alone knows
  their sizes: a graph declares them `External` with size 0 and the host fills in size and usage
  before baking (`resolve`). `check` rejects a library pass of the wrong kind, missing a resource or
  role, using anything else, or drawing into the wrong formats, and a library resource of the wrong
  class or kind; errors name the pass and the resource (unit tests break the native graph in each of
  these ways). The native scene (`TerrainApp`) now declares its graph through the library
  (`native_graph`); all GPU tests passed unchanged across the refactor.
- Module contract (`ae_render_abi::module`, wire format documented there): a module is a WASM module
  with its own memory and no imports, exporting `ae_render_version`, `ae_render_declare(width,
  height, color_format, flags)` (the graph declaration plus a program table: per payload, a library
  pass by name or a fullscreen program with its SPIR-V and uniform size), `ae_render_input` (where
  the host writes `FrameInput`: extent, frame, time, view-projection, camera, unit count, host
  flags) and `ae_render_frame` (a bitset of disabled programs, whose conditional passes are skipped,
  and every fullscreen program's uniform block). `declare` runs at load, on reload and when the
  target changes, `frame` every frame. Topology changes rebake (the signature); a skipped pass does
  not. The program table is its own record, so the graph descriptor and its fuzz test are
  unchanged. Rust modules use `sdk/ae-render-sdk` (`RenderModule` trait, `render_module!` writes the
  exports); they build for `wasm32-unknown-unknown` (`ae_render::module::build`, output in
  `target/render-modules`).
- Host (`ae_render::module`, `TerrainApp::with_module`): wasmtime 45 (the workspace's), default
  config; what the module returns is copied out with bounds checks, decoded, validated, and wired
  through the library. `tests/module.rs`: the in-repo `render/default` module declares the native
  graph (same signature, same bake, same library pass at every position, for both outputs with and
  without units); modules that import something, have the wrong version, lack exports, return
  pointers outside their memory or trap are rejected with messages saying so.
- Fullscreen programs (`ae_render::fullscreen`): a module's own fragment shader over the engine's
  fullscreen triangle into the pass's colour attachments (straight-alpha blending, so overlays
  blend). The SPIR-V is reflected with `rspirv` and checked against the pass before a pipeline is
  made: set 0 binding 0 the uniform block (declared size between the block's end and its std140
  16-byte rounding), bindings 1.. a combined image sampler per `Sampled` intent in order, nothing
  else (a storage buffer fails with "the pass declares no storage access"), no push constants, one
  output per colour attachment. Unit test on fixture shaders checks every rule.
- Reload: the client (or a test) puts new module bytes in `TerrainApp::module_reload`; at the next
  frame the scene loads, declares and wires it and swaps it in (old fullscreen pipelines destroyed
  after a device idle), or keeps the running module and records the error (`module_status`). A trap in
  `frame` stops calling the module; its last graph keeps rendering. `render/post` is a second
  module: the scene into an intermediate image, a colour grade (fullscreen program) into the target,
  and a vignette (conditional fullscreen program) enabled by host flag bit 0.
- Checked on the GPU (validation on, 0 messages): `the_default_module_renders_like_the_native_graph`
  (terrain and 400 units, 16 frames: byte-identical pixels and unit counts);
  `render_modules_reload_while_running`, one executor: default, then post (bakes 1 → 2, the grade
  changes most of the picture), the vignette flag (no rebake; corners 201.6 → 97.6, centre 186.2 →
  186.2), garbage bytes (rejected, post keeps rendering the same picture), then a module serving
  the default graph that traps in `frame` (installed, bakes → 3, stopped, its graph renders like the
  default module's). Between the last two, a graph running one fullscreen program in two passes is
  rejected naming both passes (a program's descriptor sets are per frame slot and rewritten when its
  pass records, so a second pass in the same frame would update a set already bound).
- `ae-client --render-module NAME|FILE.wasm` watches the module's sources (or file), rebuilds on
  change and reloads; N toggles flag bit 0. Topology swap through the present path, windowed, with
  validation: a `.wasm` file replaced (by `mv`) with the post module and back while `--exercise`
  resized and minimized (24 s, 21 swapchain recreates, 3,000 units): 3 loads, 0 errors, 0
  validation messages. Live run with validation (`--render-module post
  --exercise`, 26 s, 23 swapchain recreates, 2,000 units): an edit to the grade reloaded, a syntax
  error printed the compiler error while the running module stayed, the revert reloaded; 0
  validation messages. GPU over the army view: grade 0.14 ms at 1280x720 (profiling with
  validation; noisy as noted above).
- `cargo test --workspace` now also needs the `wasm32-unknown-unknown` target (the tests build
  `render/default` and `render/post`): part of the open CI gating issue with `glslc` and SDL3.
- Fullscreen pipelines always blend (straight alpha). A first write with `DontCare` into a float
  target could blend with undefined contents (NaN × 0); today's targets are 8-bit sRGB. Give programs
  a blend switch before float targets.
- `FrameInput`'s padding became `flags` with the contract still at version 1 (no modules outside the
  repository exist yet).
- Not done: shader reflection covers fullscreen programs only (library passes are engine code);
  multi-threaded recording; module draw sources (`InstanceStream` bindings for module shaders); a
  module cannot yet declare compute programs; no epoch/fuel watchdog on the module (an infinite loop
  in `frame` hangs the render thread; trusted, like the sim module).

Terrain edits on screen (`TerrainChanged` reaching the render heightfield):
- Sim side (`ae_services::nav`): `NavPlugin::ground` (new field, so `NavPlugin` is `Clone` instead
  of `Copy`; `NavPlugin::new(map, cells)` leaves it `None`) is an optional `GroundFeed`. When a
  repair that changed the ground lands, the `nav` system publishes the service's terrain (the
  immutable `Arc<Terrain>` it already holds, nothing copied) with the change's cell rectangle;
  the feed keeps the newest terrain and the union of rectangles not yet taken, so a reader that
  falls behind still refreshes everything. Cost-only changes are not published.
  `tests/capture.rs` now runs the plain world without the feed and the captured one with it: same
  digest; the published terrain (version 4) differs from the start map in 551 cells, all inside the
  published rectangle.
- That rectangle is one box around every patch in a repair: the crossing game's quake scatters ~100
  small squares, so the box is ~1,040 x 1,890 cells, which uploaded whole is 15 MB and a 25–32 ms
  render-thread stall. So `ae-client` compares the new terrain with the one on screen inside the box,
  in 32-cell tiles (ground height and material; `ground::changed_rects`, unit-tested), and edits
  only the changed cells of each tile: 44–50 tiles, 0.2–0.3 MB.
- Render side: `Pyramid::update` recomputes every pyramid block that reads a changed corner (a
  block's error needs all its samples, so the top levels rescan most of the map; several
  rectangles share their blocks), bit-identical to `build_pyramid` (unit test over edge, corner,
  boundary-straddling and out-of-range edits, single and batched; it fails if the boundary block or
  the uncached E_0 is left out). `terrain::edit::TerrainEditor` keeps CPU copies of heights, pyramid
  and materials and turns changed cells into `TerrainUpdate` regions (corners average the cells
  around them as at load, `Heightmap::from_cells`, normalised with the load-time scale, so an edit
  may leave [0, 1]; the textures are float). Unit test: after edits the copy equals a map built from
  scratch from the edited cells. `TerrainApp::terrain_edits` takes updates; the render thread
  applies them at the start of the next frame: device wait-idle (every frame in flight samples the
  textures), then `upload::update_images` (`SHADER_READ_ONLY` → `TRANSFER_DST` → back, so texels
  outside the regions are kept; each region at a 16-byte staging offset). Heights, pyramid, errors
  and materials change in the same frame.
- `ae-client --quake TICK` sets the crossing game's quake tick; a thread runs the editor (the
  pyramid update is ~67 ms on the 2560² map, so neither the sim nor the render thread does it).
  `--capture` applies the landed edits before rendering. The title shows the terrain version.
- Checked: GPU test `terrain_edits_reach_the_heightfield` (flat map, a 13-cell square raised by
  40 m and turned to rock while rendering through one executor: the unit standing in it moves
  15 px up the screen to the new ground and nothing red stays at the old height, the square's
  colour changes, and 0 of 71,872 pixels outside the square's screen box (one cell of margin for
  normals) change; then three small rectangles of unaligned sizes in one update; 0 validation
  messages; that second update is pushed while frames are in flight, so the wait-idle path runs
  under synchronization validation). The unaligned case was found by validation in `--capture` (a copy's buffer offset must
  be a multiple of the texel size) and the test fails with the alignment removed; a region offset
  off by one fails it too. Not caught: uploading from `UNDEFINED` instead of `SHADER_READ_ONLY`
  passes on this driver (NVIDIA keeps the old texels; validation does not flag it), so that path
  rests on review. Water map, 10k units, `--quake 10 --ticks 120 --capture`: 50 tiles, 309 KB,
  edit 67 ms on the CPU, applied in 1.1–2.3 ms (9 ms with validation); with validation 0 messages.
  Windowed, `--quake 60 --exercise`, 14 s with validation: both landings (rock at version 2,
  raised ground at version 4) applied, 13 swapchain recreates, 0 validation messages. Late sim
  ticks over 12 s windowed runs: 1 without the quake, 4 with it (two runs each), and still 4 with
  the editor switched off (a temporary switch, removed): the 3 extra are the sim's own quake work
  (nav repairs, moving buried units), not the edit thread.
- Not done: the camera's ground height (`TerrainSource::height_at`) still reads the load-time
  heightmap, so the orbit target does not follow edited ground. Applying an edit idles the device
  (fine for edits a few times a second; per-frame terrain deformation would need the copies inside
  the graph). The CBT re-refines over the following frames; nothing forces the edited area to split
  at once.

Material tilesets (the proposal above; full write-up with captures, texture formats, settings and
numbers in [docs/tilesets.md](tilesets.md)):
- `ae_render::terrain::tileset`: a tileset is up to 8 tileable layers (albedo + height in an
  `R8G8B8A8_SRGB` array, normals in an `R8G8_UNORM` array, full mips made on the GPU from mip 0), a
  table from the 8 gameplay material ids to a ground and a cliff layer, and per-layer (tile size,
  normal strength, projection, directional) and per-set (anti-tiling degree, cliff slope range,
  blend depth, detail tilt, boundary warp, macro variation) settings. Two procedural placeholder
  sets, `temperate` and `desert`, 7 layers each, from periodic gradient and cellular noise (14 MB at
  512², generated in 0.06 s on 24 threads).
- Shader (`terrain.frag`, `tileset()`): ground layers planar, cliff layers duoplanar, `Triplanar`
  layers triplanar, with sign-matched uv/tangent flips and whiteout normal blending; every sample
  uses `textureGrad`. Two-level blending: broad masks (bilinear material cells looked up through a
  noise warp, so boundaries cross the 8 m grid; a cliff mask from the slope tilted by the ground's
  detail normals), then a height blend over the five candidates. A weights debug view shows the
  blend as flat layer colours.
- The tileset is not part of the map: `TerrainApp::with_tileset(Arc<Tileset>)`, optional; without
  it the shader's palette path is unchanged, and `--tileset none` captures are byte-identical to
  before (water and generated maps). `ae-client --tileset temperate|desert|none` (default
  temperate), `--tileset-size`, `--weights`; M toggles tileset/palette, B the weights view.
- Plumbing: `OwnedImage::with_view` (2D-array views), `upload::image_layers` (layers × mips, 16-byte
  staging offsets; the tileset now goes through `upload::image_mipped`, mips on the GPU), a trilinear REPEAT sampler, bindings 10–12 (pool sizes raised).
- Checked: tiling, unit normals, determinism and table validation (unit tests; the tiling measure
  fails on a broken edge); GPU test `tileset_boundaries_are_irregular_and_bounded` (a material
  boundary meanders 15 px and frays in 66 of 232 rows inside its allowed band, the palette's is
  straight; a steepening ramp shows rock where steep, grass on the flats, irregular edge; removing
  the warp or the height blend fails it); 0 validation messages in the tests and in windowed runs
  (with 10k units and a quake, the weights view on the generated map, and under the `post` render
  module). The M/B keys were not pressed in a run (no key injection), and no check has looked at a
  terrain edit under a tileset (it reads the same textures the edits update).
- Cost (terrain draw, 1280x720, without anti-tiling): 0.17–0.19 ms with a tileset vs 0.05–0.08 ms
  with the palette on the water map, 0.34–0.38 ms over cliffs. With the default anti-tiling degree 1
  (every layer, rock included): 0.23 ms close, 0.30 ms overview, 0.73 ms over cliffs. Sampling a
  corner's layer once when corners share it, and the cliff layer only where the mask is non-zero,
  halved it; doing the reuse by runtime index into the tap array had made it slower (0.43 ms
  overview), constant indices fixed that.
- Not done: detail-normal signs on cliffs checked by eye from opposite yaws only; procedural
  textures only (no import, no anisotropic filtering); maps without materials use slot 0 everywhere
  (the palette's snow and dry bands are gone with a tileset); no runtime tileset swap; one tile per
  layer (macro variation only, no stochastic tiling).

GPU bench (`ae-client --bench <scene>[,<scene>...]|all|list [--frames N] [--warmup N] [--label NAME]
[--out DIR] [--no-save]`):
- Six fixed scenes, offscreen at 1280x720 with per-pass timestamps: `overview` (water map middle
  from 3 km), `army` (8,606 of 10k units visible), `orbit` (one turn of yaw at a low angle, so LOD
  and culling change every frame), `close`, `palette` (as `close` without the tileset) and `cliffs`
  (generated map). Units come from the crossing game run headless to tick 300, once per map. Each
  scene warms up for 60 frames and at least a second (GPU clocks: without it the first scene read
  15–25 % slow), then renders 300.
- One report per scene, `bench/results/<label>/gpu-<scene>.json`, read by `ae-tool bench-diff`:
  median GPU time per pass and in total, render-thread CPU time per frame (`cpu_frame`, fence waits
  excluded) and its recording part (`cpu_record`); counters `leaves` and `drawn` (summed over the
  frames: deterministic, so more is a regression); info `units_visible` (last frame, read after the
  GPU finished), frames and size. The readback copy only offscreen rendering has is left out of
  `gpu_total`: it follows the host bus (±25 % between runs). Since M13 stage 7 it is reported on
  its own (`pass/copy`, `pass/host read`, 50 % tolerance), and a `--bench-output hdr10|scrgb`
  run writes `gpu-<scene>-hdr10.json` or `gpu-<scene>-scrgb.json`.
- **Stable as a gate since 2026-11-22.** Three changes, each against a measured cause:
  - Several scenes run one process each (`--bench all` starts itself per scene with the same
    arguments). In one process the scenes after the fox and model scenes measured up to 3x
    slow in some runs and not in others (`lights-200` 0.75–2.3 ms, its terrain 0.30–0.62 ms),
    at full clocks (1830–1950 MHz, 6801 MHz memory, the 200 W power cap reached from `scorched`
    on) and with VRAM to spare (5.5 of 8 GB); the same scene in a fresh process right after the
    heavy ones measured 0.76 ms. The cause inside the process is not known (TODO).
  - The warm-up spends its time-based second (GPU clocks) with no explosions or ticks, then runs
    exactly `--warmup` frames with them, so `battle`'s state when measuring starts no longer
    depends on the frame rate (its light count moved 212–215 and its total 1.24–1.41 ms).
  - `gpu_total` is the sum of the passes' medians, with the per-frame sums' spread. The median
    of the per-frame sums moved 10–13% between runs (`lights-1000`, `lights-low`) while every
    pass agreed within 2%: single passes spike in some frames (`terrain lod` ±79%).

  Not the cause: the clocks between scenes (the GPU drops to P5 for the ~3 s a scene sets up,
  and is back at full clocks within the warm-up), timestamp attribution (a full barrier before
  each end timestamp left `near-fox`'s terrain at 265/440/545 µs), and the CBT (309 leaves in
  every frame). Before, two whole runs of one binary flagged 11–13 metrics against each other;
  with one process per scene and the fixed warm-up, two pairs flagged 1 and 4 (only `gpu_total`
  in light scenes, and `cliffs`' exposure meter); with the summed total as well, the final pair
  flagged 2: `cliffs`' `pass/exposure meter`, which is 19 or 195 µs for a process's lifetime, and
  its `gpu_total`. A whole run takes 93 s (43 s in
  one process). The baseline's GPU records were re-recorded this way at 335e770.
- `ae_bench` gained a per-record `tolerance` that widens `--time-tolerance` for that metric:
  CPU times get 50 % (tens of microseconds; medians move by up to 30 % between runs, once 60 %),
  passes under ~20 µs get 2 µs absolute (the timer ticks in about 1 µs). `Executor::take_gpu_frames`
  gives the per-frame pass times (capped at 16,384 frames; reset by a rebake).
- Repeatability: three back-to-back `--bench all` runs, 81 metrics, 0 flagged between any two;
  GPU totals agree within 1 %. 0 validation messages (`AE_VK_VALIDATION=1`, 30 frames each). All six
  scenes take about 8 s, most of it building the map and the sim.
- Baseline (RTX 3060 Ti, `bench/results/baseline/gpu-*.json`), GPU ms per frame:

  | Scene | Total | `terrain lod` | `terrain` | Leaves drawn per frame |
  |---|---|---|---|---|
  | overview | 0.40 | 0.22 | 0.17 | 423 |
  | army | 0.41 | 0.21 | 0.15 | 46 (units 0.04) |
  | orbit | 0.48 | 0.23 | 0.23 | 2,619 |
  | close | 0.36 | 0.21 | 0.15 | 42 |
  | palette | 0.26 | 0.21 | 0.04 | 42 |
  | cliffs | 0.59 | 0.23 | 0.36 | 14,043 |

  The CBT update (`terrain lod`, a compute pass) is a fixed ~0.21 ms in every scene, half or more of
  the frame: the first thing to look at for GPU time.
- Not done: no GPU runner in CI, so nothing runs it automatically; scenes are built into the
  client (not data); one resolution; the CPU times are too noisy to gate small changes.

Threaded recording (`ExecutorConfig::record_threads`, `ae-client --record-threads N`, default 1):
- `Executor::execute` first works out every pass serially in baked order: skip, barriers (the access
  state machine is order-dependent), first-use load-to-clear, clear values and attachments. Then,
  with `record_threads > 1`, a recorder that offers `Recorder::shared` (a `SharedRecorder: Sync`
  view for one frame) and at least two passes to record, each pass is recorded into its own primary
  command buffer (one pool per slot and pass, made at bake time) on a rayon pool of that many
  threads. A head buffer holds the timestamp-query reset; one `vkQueueSubmit2` runs them in baked
  order. Otherwise everything goes into one buffer on the calling thread, as before. Each pass
  records the same commands either way (timestamps, label, barriers, rendering scope, viewport and
  scissor, payload); primaries instead of secondaries avoid rendering inheritance. The rule is
  fixed, never timing-based.
- `TerrainApp` records through a borrowed `SceneRecorder` (terrain and unit renderers, fullscreen
  programs, sampler), without the module's WASM instance. `RenderApp::Pass` must be `Sync`.
  `DemoRenderer` caches descriptor sets while recording, so it stays serial.
- Checked: `--capture` with 10k units and under the `post` module, 1 vs 4 threads: byte-identical,
  0 validation messages, 60 of 60 frames parallel. `depth_and_multiple_targets` (D32 and D32S8)
  run serial and on 3 threads. New GPU test `parallel_recording_matches_serial`: 24 graphics passes
  × 3,000 draws, later passes over earlier ones, identical pixels under synchronization validation.
  Windowed `--exercise --record-threads 4` under validation, 8–10 s (one run with the `post`
  module): 7 and 9 swapchain recreates (rebakes recreate the per-pass pools), every frame parallel,
  0 validation messages.
- Numbers (no validation): the stress graph records in 2.14 ms on 1 thread, 1.62 on 2, 0.78 on 4,
  0.58 on 8 (3.7×). The real scenes are too light to gain: their passes record in 40–60 µs (the
  terrain and units are GPU-driven, a few draws per pass), and 4 threads add 15–25 µs of fork-join
  and submit cost per frame (`--bench all --record-threads 4` vs 1: `cpu_frame` +20–40 %, GPU times
  unchanged). So the default stays at 1; threads pay off once passes issue thousands of draws.
- Not done: the parallel unit is a whole pass; one heavy pass cannot be split, and passes are not
  grouped (a pass per command buffer). The recording pool is the executor's own, not the engine-wide
  job pool the plan describes.

Golden `.dot` (Verification, "golden frame graph `.dot` output"): `crates/ae_render/tests/golden.rs`
bakes the native scene graph (with units presenting and reading back, without units) and the demo
graph (present, readback) and compares `dot::to_dot` with `tests/golden/*.dot`; it also checks the
export is the same when built twice. `AE_UPDATE_GOLDEN=1` rewrites the files after an intended change.
An edited golden line fails the test, naming the line.

Frame graphs from data files (the editor's M3 cost, "descriptor loadable from a data file"):
- `ae_render::graph_file`: a JSON file declares resources (`library` buffers, `image`s sized from the
  target with an optional `scale`, one `output`), fullscreen programs (`fragment`: a `.spv`, or a
  `.frag` compiled with `glslc`; constant `uniforms`) and passes (`library` or `program` payload,
  access strings such as `"depth depth:clear"` or `"units.counts transfer_dst discard"`, `flag` n for
  a conditional pass bit n of the app flags enables, `when` units / no_units / readback / present).
  The host appends present or readback copy + host read for the output, as the modules write them.
  It produces the same `Declared` (graph descriptor + program table) a module's `declare` returns.
- `RenderModule` is now WASM or a graph file (JSON when the bytes start with `{`), so a file goes
  through the same checks, library wiring and hot reload. `module::build("x.json")` inlines the
  shaders (`fragment_hex`) so the bytes stand alone; a raw file with paths is refused by name.
  `ae-client --render-module render/post.graph.json`; the watcher follows the file and its
  directory.
- `render/default.graph.json` and `render/post.graph.json` restate the two in-repo modules. Test
  `tests/graph_file.rs` (no GPU): same descriptors (decoded and re-encoded, since the builder's name
  blob follows call order) and byte-identical program tables, SPIR-V included, at 3 targets × 4 scene
  kinds, and the same uniforms and enabled passes for app flags 0–3. A changed uniform or flag in
  the file fails it. Unit tests cover role parsing and errors that name the pass and resource.
- `--capture` with `post` and with `render/post.graph.json` (2,000 units): byte-identical, 0 validation
  messages. Windowed, under validation: editing a uniform in the file reloaded it (2 loads); a
  misspelt program name was rejected ("pass `vignette`: no program `vignet`") and the running graph
  stayed. Removing the `vignette` pass from the file mid-run (a topology change): 2 loads, 2 graph
  bakes (1 in the same run without the edit), 0 validation messages. The client's exit line now
  prints the bake count.
- Not done: uniforms are constants (no time or camera inputs; that still needs a WASM module); no
  text form for compute programs (only library compute passes); no `ae-tool` command to convert a
  module's declaration into a file.

M3 is complete: every scope item has landed (depth/MRT, declaration ABI, reflection, instance streams
and GPU culling, `terrain_cbt` with material tilesets, threaded recording, the GPU bench, golden
`.dot`, graph files). The exit run (`ae-client` with 10k units, topology hot reload, validation
clean) is covered by the runs above (topology: the graph-file pass removal). What each part leaves open is in its "Not done" line. Next is
M4 (copy layer and fog).

### M4 progress

Built as libraries first, with synthetic-world tests and benches, before the schema, the game or the
client change.

`ae_sensors` (coverage):
- Per allyteam, one bit per cell for line of sight and for radar. A position becomes the cell
  `floor(x / cell)` (clamped to the grid), a radius `floor(r / cell)` cells, and a unit covers the
  cells with `dx² + dy² <= r²`, stamped per row as a span of half width `isqrt(r² - dy²)`. Integer
  math after the cell step and no hash maps, so it can move into the sim as a native service
  unchanged. Allyteams stamp in parallel, with the same bits as serial.
- Tests: three random worlds against a brute-force oracle of the same predicate (units off the map,
  zero and NaN radii, rows of unknown allyteams); parallel equals serial at 10k units; spans across
  word boundaries; `isqrt` exact up to 2⁶⁴.
- `ae-tool bench sensors` (baseline recorded): 10k units, 2 allyteams, 640² cells: 1.24 ms serial,
  0.76 ms parallel; 8 allyteams 0.36 ms parallel; 50k units 6.2 ms serial, 1.4 ms parallel.

`ae_view` (rule executor):
- Input: one tick's capture (entity ids, one column per field, allyteam per row) and the coverage;
  a `ViewSpec` gives each field a class (`Never`, `Always`, `OwnerOnly`, `Los`, `RadarDegraded`,
  `Ghost`) and names the position fields. An `Observer` (an allyteam and a local seed) builds its
  view table: rows `Own`, `Seen` (line of sight), `Blip` (radar only) or `Ghost`, a column per
  non-`Never` field, a per-row mask of the fields it may know (hidden values are 0), in view-id
  order.
- View ids hide sim ids: given on coming into view, in an order shuffled by the observer's seed;
  kept while the entity stays in view; new after a recycled id (new generation) or a return after
  absence. A ghosting table (one with a `Ghost` field) keeps a row last seen in line of sight as a
  `Ghost` with its last-seen `Ghost`, `RadarDegraded` (the exact position) and `Always` values, until
  the observer has line of sight to
  that cell again; a returning entity takes its ghost's id back. Blip positions drift by up to
  `wobble` world units, hashed from (seed, view id, tick bucket) and blended across `wobble_ticks`.
- Property test (`crates/ae_view/tests/property.rs`): 4 seeds × plain and ghosting tables × 60
  ticks, 300 entities, three allyteams, movement, churn with recycled ids, two observers, against an
  oracle written from the documented class table: layout without `Never` columns; row states equal
  the coverage; every entity in view has exactly one row; each field shown exactly where its class
  allows, hidden values 0, shown values exact (blips within the wobble, ghosts equal to the last
  sighting, never on a cell in line of sight); no word equals either half of a sim id; ids unique,
  ascending, kept in view, reused only by a returning entity taking its ghost's id; a ghost's own
  position equals the last sighting. It saw 105 ghost rows, 61,820 blip positions, and the two
  observers differed on 170 ticks. Five mutants of the executor fail it (`Los` shown in blips, an
  entity id in hidden values, ghosts kept in line of sight, ids reset each tick, ghosts without
  their position; the last was the executor's first version, found in review: ghosts over `Body`
  would have had x and y hidden). A unit test checks the wobble is bounded, smooth and differs by
  observer and id.
- `ae-tool bench view` (baseline recorded): 10k units, 2 allyteams, 7 fields shown, ~7,100 rows per
  observer of which 1,630 blips: one observer 0.52 ms, a tick for two observers (preparation and both
  builds) 1.08 ms; with a ghosting table 1.13 ms; the `_contact` case (allyteams mixed inside every
  squad, 10,000 rows per observer, as on the crossing map) 1.38 ms. 50k units and 8 allyteams: 3.3 ms
  for two observers.

View classes in the schema, teams and sensors in the sim:
- Field flags bits 8..12 carry the view class (`ae_abi::FIELD_VIEW_*`; 0 is never, so a field nobody
  classified stays hidden; the host decoder already accepted unknown flag bits).
  `#[field(view = "los")]` (`never`, `always`, `owner`, `los`, `radar`, `ghost`) sets it;
  `FieldDecl::view()` reads it. The SDK's engine components are classified: `Body.x/y` radar,
  `vx/vy/radius` line of sight; `NavAgent.max_speed/profile` line of sight, `mode` owner;
  `Goal.x/y` owner, `group` never.
- New engine components `ae_sdk::nav::Team { ally }` (always) and `Sensor { los, radar }` (line of
  sight), ids `0xAE00…09` and `…0A`. They reach a game's schema without the game naming them (as
  `ae-sim`'s new `schema_dump` example shows for the crossing game).
- `NavPlugin::teams` (default 0: unchanged worlds): `nav_spawn` adds a `Team` and a `Sensor` to every
  map unit. Allyteams are equal shares of the units ranked by spawn x (the crossing plan spawns one
  dense block; a split by map stripes gave 150 vs 2,850), line of sight by class (`CLASS_LOS`, 300
  to 650 m), a 2,200 m radar on every 25th unit.
- The pinned crossing digest moved (`0x59aa…f2de` to `0x2869…d4be`; dev and release agree, all worker
  counts agree): declaring two more components shifts flecs entity ids. With the two components
  removed, the view-class bits alone left it unchanged. Every SDK component added later will move
  game digests the same way; giving game entities a fixed id range would end that.
- `ae_services::view::ViewPipeline`: captures every 4-byte field of `Body`, `Team`, `Sensor` (and
  named extra components) that has a view class, plus the sensor inputs, stamps coverage at 32 m
  cells and builds observers' tables.
- Test `ae_services/tests/view.rs`: crossing, 3,000 units on 512 cells, 2 teams, two observers
  every 15 ticks. Own rows are exactly the allyteam's units (all fields, `mode` included); `Seen`
  rows are enemy units in line of sight with no owner fields; blips show no line-of-sight or owner
  fields; no view word lies in the entity index range; the observers' worlds differ (observer 0:
  1,500 own, 1,350 seen, ~105 blips; observer 1 sees all 1,500 enemies). Viewing leaves the digest
  unchanged, and with teams 1 and 2 workers agree. Two views build in 0.4–0.9 ms here (3,000 rows).

The client draws an observer's view (`ae-client --teams N [--observer K|all]`):
- With `--teams`, the sim thread runs a `ViewPipeline` and one `Observer` (default allyteam 0;
  `all` draws everything as before). Units are streamed from its view table: own units in their
  class colours, enemies in line of sight in the same classes tinted red, blips grey (their class
  and radius are line-of-sight fields, so a blip gets a fixed 5 m radius). The streamer now matches
  rows by key (entity, or view id) and snaps a row whose state changed (a blip seen, a unit lost)
  instead of interpolating across it. `MAX_CLASSES` went from 8 to 16 (11 classes in use).
- Fog of war on the terrain: `TerrainApp::with_vision(VisionFeed)`. The sim publishes the
  observer's line-of-sight plane and the cells it has ever seen (OR of every tick's line of sight);
  the renderer copies the newest frame into the slot's host-visible storage buffer (binding 13,
  two bit planes) in `prepare`, only when it changed, and `FrameUbo` gained a `vision` vec4. Flag 64:
  the shader blends the four nearest cells (1 in sight, 0.5 explored, 0.2 never; lookups clamped
  to the grid) and desaturates what is out of sight. A host write before submit needs no graph pass.
- GPU test `vision_shades_the_terrain`: full line of sight is byte-identical to no vision; with sight
  on the left half the left matches no vision (luma 167.0 vs 167.0), explored ground reads 132.6 and
  never-seen ground 96.2 (sRGB bytes); a feed with no frame yet reads as never seen. All 15 GPU tests
  pass with validation.
- Captures (`--teams 2 --capture`, 10k units at tick 300): observer 0 has 5,000 own, 4,409 seen,
  91 blips; observer 1 5,000 own, 4,815 seen, 185 blips. The ships in the river, out of observer 0's
  sensors, are not drawn for it. The view path on the sim thread, first build (every id fresh):
  capture + coverage 0.87–0.90 ms, view 1.48–1.54 ms, rows + stream 0.19–0.20 ms.
- Windowed `--teams 2 --seconds 10 --exercise` under validation: 9 swapchain recreates, 0 messages,
  5.4 ms/tick sim, capture + view + stream 2.5 ms per tick on the sim thread, 1 late tick.
- `ae-client --bench all` against the baseline: all 81 GPU metrics unchanged (vision off costs
  nothing measurable).

Real workload: `ae-headless --game crossing --nav water --units 10000 --ticks 300 --workers 8 --teams 2
--observers 2` (new options: after every tick, capture, coverage and the observers' builds, medians
over the second half): 10,000 own + 9,224 seen + 276 blip rows across the two views; building both
1.55 ms (0.77 ms per observer); capture + coverage 0.84 ms more, so the whole view step is 2.4 ms.

M4 exit criteria: two observers see different worlds (the live test and the captures); the
property test holds against an independent oracle; building the views of two observers takes
1.55 ms at 10k units on the crossing map (under the 2 ms the exit asks), but capture and coverage
add 0.84 ms, so the full copy-layer step for two observers is 2.4 ms.
- Not done: the game's churn replacements get no `Team`/`Sensor` and leave every view (the live
  test checks it, and that flecs gives recycled indices a new generation, so their 64-bit and view
  ids are fresh); ghosts are only exercised on synthetic worlds (the crossing game has no static
  units); the optional trusted WASM view-rule kernels and the `ViewSlot` latest/previous publish of
  the architecture do not exist (the client's sim thread publishes a stream and a vision frame);
  explored cells are remembered by the client's sim thread, not by the copy layer; there is no key
  to switch observers while running; radar shows no coverage on the terrain; jammers, sonar and
  seismic sensors are not modelled; the sensors run on the view side, not as a sim service (the
  plan lists them under native services); decals do not exist yet. (The units' own shadows, M7,
  follow fog for free: casters are the observer's stream, so a hidden unit casts none.)

M4 is complete with the caveats above. Next is M5 (lockstep and replay).

### M5 progress

One sim per process: a second `Runtime` in the same process fails ("column arena already installed",
flecs' column allocator is process-wide), checked with two identical configs at once. So
multi-client lockstep runs one process per client from the start (loopback UDP); in-process tests
can cover the relay and protocol logic only.

Player commands (`ae_sim::input`):
- `TickBundle { tick, commands }`, a command being `{ player, stream, bytes }`: one event of a
  stream the game marks as a command stream (`#[component(event, command)]`, schema flag
  `COMPONENT_COMMAND`; the SDK marks `MoveOrder` and `OrderPoint`). The one input type: replays and
  the relay will carry the same bundles. Binary encoding with a round-trip test.
- `Runtime::tick_with(world, workers, &bundle)` appends the valid commands to their streams before
  the first layer, in bundle order (whoever builds the bundle fixes the order; peers never sort), so
  readers see them that tick; it drops commands on unknown or non-command streams or of the wrong
  size, identically everywhere, and counts both (`TickStats::commands_in/dropped`). `tick` is
  `tick_with` an empty bundle. Commands name 64-bit sim ids, never view ids.
- Test `ae_services/tests/input.rs` (crossing with the game's own orders held back): `MoveOrder`
  commands for 40 units at tick 10 make only ordered units travel far (39 of 40 did; others are
  pushed aside); the same bundles in a fresh run give the same digest; three invalid commands are
  dropped and the world is as without them; an order for a nonexistent entity is ignored the same
  way every run. Crossing's pinned digest is unchanged (the command flag is not hashed).
- Not done: ownership (a player ordering another's unit) is not checked; that belongs to the relay
  or to game systems reading `Team`.

Replays (`ae_replay`, `ae-headless --record / --replay --verify`):
- A replay is a header (engine version, xxh3-128 of the game module and of its schema section, the
  game parameter, the host configuration `map`, `map_cells`, `teams`, the checkpoint interval), the
  tick bundles that have commands, checkpoints (digest total and its parts per component and host
  service), and an end record. `Header::check` names the first difference; a file without its end
  record is refused.
- `--record FILE` runs one worker count, with `--script orders` (two players; every 40 ticks each
  orders about 1 unit in 16, chosen by a hash of entity and tick, to a point up to 800 m away; built
  from the world alone) and a checkpoint every `--checkpoint-every` ticks (default 30; a digest costs
  0.9 ms at 10k units against a 4.2 ms tick, `AE_DIGEST_COST=1` measures it). `--replay FILE
  --verify` builds the world the header names, feeds it the file's bundles only and compares every
  checkpoint; a mismatch prints the tick and the differing parts and exits 1.
- 4,000 units on 1,024 cells, 600 ticks, 7,424 commands: 364 KB; replayed at 1 and 8 workers, all 20
  checkpoints match; another module is refused ("the game module differs").
- Sequential runtimes in one process: the column arena was keyed by the address of the runtime's
  memory, so a second runtime only worked when its memory landed at the same address (tests had
  been lucky). A runtime now releases its arena when it drops (`ae_ecs::arena::release`), and the
  next one installs a fresh arena. Still one live sim per process.

Desync localisation (`--replay FILE --bisect --inject TICK:Component.field:ROW`):
- `Runtime::poke` overwrites a 4-byte field of one entity (fault injection only). The bisect runs
  three times from tick 0, since there are no snapshots: the faulty run recording every tick's
  digest; the clean run up to the first tick whose digest differs, dumping every component row
  there; the faulty run again to that tick, dumping and diffing rows by entity and field (values
  shown by field kind). A divergence that shows only in a host service's part is reported as inside
  that service.
- Water map, 4,000 units, a bit of `Body.x` flipped on one unit after tick 200: "first divergence
  after tick 201 ... parts that differ: (host nav), Body", one row, that entity's `Body.x`, in 4.6 s.
  Digests compare states after ticks, so the fault shows after the next tick, by when the nav service
  may have moved the unit and its neighbours by it.
- Test `bins/ae-headless/tests/replay.rs` drives the binary: record (600 units, 120 ticks, scripted
  orders), verify at 1 and 2 workers, refuse the sandbox module, and bisect a flipped `Body.y`
  (divergence after tick 61, at the injected entity, at most 3 entities).
- Not done: bisect compares two runs on one machine; between machines it would need each side's
  per-tick digests and rows (the relay's job); no Windows or aarch64 runs (no such machines here),
  so "replay digests equal on Linux and Windows" is not checked.

Lockstep over UDP (`ae_net`, `ae-relay`, `ae-headless --connect`):
- Relay-paced: `ae-relay --players N --ticks T [--hz 30]` waits for N clients whose world headers
  agree (the replay header), then closes one bundle per tick on its own clock from every input a
  player sent for that tick or earlier, by player, then intended tick, then the player's order; a
  late input goes into a later bundle, so no client waits for another, only for the relay. Clients
  run tick t only with bundle t.
- Loss needs no handshakes: every 10 ms a client resends all inputs the relay has not reported taken,
  the next bundle it needs and its recent digests; the relay sends each client the bundles from the
  one it needs (up to 48 KB per datagram). Inputs are scheduled `rtt/2` in ticks plus 2 ahead
  (adaptive input delay from the echoed timestamps).
- The relay compares the digests every client reports at each checkpoint and at the end (a desync
  is reported with each player's digest and makes it exit 1), and `--record` writes the game as a
  replay (bundles, plus the agreed digests as checkpoints) that `ae-headless --replay --verify` checks
  alone. `ae_net::Emulator` delays (latency + jitter, so reordering) and drops (seeded) outgoing
  datagrams on either side.
- Clients with `--script orders` send their own player's orders (at most 32 per order tick);
  `--inject` makes one client diverge.
- Run: relay + 4 clients (2,000 units, 512 cells, 2 workers each), 40 ms latency + 0..30 ms jitter
  + 5% loss on every sender, 300 ticks: 29.4 Hz, 1,024 commands (1 taken late), 10 digest rounds, in
  sync, the relay dropped 922 of 17,280 datagrams; the recorded replay verifies (10 checkpoints).
- Test `bins/ae-headless/tests/lockstep.rs` (builds `ae-relay`): 3 clients, 30 ms + 0..20 ms jitter +
  5% loss, 240 ticks: in sync, 8 digest rounds, every client at the agreed digest, the recording
  verifies; then a bit flipped in one client after tick 100 is reported as a desync at tick 120 (the
  next checkpoint) and the relay exits 1.
- Not done: QUIC (`quinn`) as planned; plain UDP with idempotent packets instead, no encryption or
  authentication; players cannot join late or reconnect; the relay does not validate ownership;
  checkpoints only carry totals over the network (parts stay local), so a network desync is
  localised by replaying the recording with `--bisect` on one machine.

### M6 progress

Content files and the catalogue:
- `ae_vfs`: a stack of directory mounts (a later mount overrides files of earlier ones); content
  paths are relative and `/`-separated (absolute, `..`, empty parts and backslashes refused);
  `content_hash` is xxh3-128 over every visible file's path and bytes, for replay and lockstep
  headers. Archives will be another kind of mount.
- `ae_catalog`: defs as JSON (the graph files' format family): components and field values,
  `extends` (field by field) and `abstract` defs. Resolved against the game's schema to component
  bytes (unset fields zero); a def id is FNV-1a of its name, like component ids, so adding a def
  renumbers nothing. Twelve errors are tested to name the def and the problem (unknown component or
  field, wrong kind, integer out of range, entity/byte fields, events, unknown parent, `extends`
  cycle, version, unknown keys, a def in two files). `CatalogFile::write` is canonical (sorted keys,
  two-space indent): parse, write, parse gives the same bytes, and messy input writes canonically.

Spawning defs (`ae_sdk` feature `catalog`, `ae_services::catalog`):
- `SpawnRequest { def, ally, x, y }` (event) and `Def { id }` (line of sight); the host system
  `catalog_spawn` spawns each request in event order: the def's components, `Body.x/y` at the
  request, a `Team`, a `Def`; an unknown def is skipped and counted in its digest part.
- Schema records in the game crate: a type's record lived in its defining crate's object, which the
  linker drops when nothing references it (the skirmish game only emitted `SpawnRequest`, and named
  no nav type at all). `#[system]` now also writes copies of the records of every type it names,
  and `ae_sdk::declare!(T, ...)`, `use_nav!()` and `use_catalog!()` declare types a game uses
  without naming them in a system. The parser already accepted identical duplicates; crossing's
  pinned digest is unchanged.
- `games/skirmish` (new): parameter = map side in metres << 24 | tanks << 40; `start` spawns a
  `factory` at each allyteam's base (15% in from opposite corners) and its tanks in a ring;
  `content/defs/units.json` holds `unit`/`structure` (abstract), `tank`, `factory`, and a game
  component `Health`. Test `ae_services/tests/skirmish.rs`: the catalogue loads through the VFS;
  each allyteam gets one factory at its base (radius 20) and 6 tanks within 130 m (radius 4); the
  view of each allyteam has exactly its 7 units as `Own` rows (the M4 gap for game-spawned units is
  closed); 1 and 2 workers agree.

Skirmish rules (all in the game crate; content in `games/skirmish/content/defs/units.json`):
- Components `Health`, `Armor`, `Weapon`, `Factory` (its own economy: income, cost, build time, the def
  it `builds`), `Orders`, `Match`; event streams `Presence`, `DamageRequest`, `Damage`, `FactoryCount`.
- Systems: `produce` (metal, then a unit of `builds` beside the factory), `rally` (new units ordered
  to the enemy base), `scan` → `fire` (nearest enemy in range off cooldown) → `armor` (each request
  less armor, at least 1) → `apply` (health; despawn at 0), `census` → `judge` (no factories left:
  the other allyteam wins; both in one tick: a draw). Damage is a chain of streams, each read in
  event order, so the outcome never depends on how many workers emitted.
- The catalogue takes a def name for a 64-bit field (`"builds": "tank"` is the tank's id);
  `ae-tool catalog-fmt FILE...` rewrites catalogue files canonically, and the test checks the
  game's content file is canonical.
- An even match stalls: the armies meet in a corridor of the land map 1.1–1.4 km from the enemy
  factories, only the fronts fight (first at 144 m, just outside the 140 m range I started with;
  200 m now), and both factories keep feeding them. With a handicap (parameter bits 48..56: 30 more
  starting tanks for allyteam 0) allyteam 0 breaks through and wins at tick 2,497, with the same
  tick and digest at 1 and 2 workers (test `skirmish.rs`).
- `ae-headless --content DIR` mounts a game's content, loads its catalogue against the module, adds
  the catalogue service, and puts `content = <hash>` in the replay and lockstep header: a replay of
  the skirmish refuses to play without `--content` ("content = … in the replay, unset here") or with
  one tank's hp changed (two hashes named). Recorded handicap match: 2,600 ticks, 87 checkpoints,
  48 KB, verifies at 1 worker (end-to-end test in `tests/replay.rs`). `Header::check` was being
  called from the wrong side, which reversed its messages; fixed.
- Since then: units take commands, the client plays allyteam 0 and bots play the rest (below).

Command queues, attack-move and bots:
- `UnitCommand { entity, kind, append, x, y }` is the game's own command stream (move, attack-move,
  stop; `append` 0 replaces the queue, 1 queues behind it); `Queue` holds up to 4 commands and what
  the unit is doing about the head. The `orders` system (after `scan`, so it reads this tick's
  `Presence`; nav takes its orders next tick) replaces `rally`/`Orders`: a new unit queues a move
  to its allyteam's rally point (220 m in front of its base); a move's head pops when nav reports the
  unit `ARRIVED` or `UNREACHABLE`; an attack-move looks for the nearest enemy within 1.5× weapon
  range every 8 ticks (staggered by entity): out of range it chases (re-ordered when the enemy moved
  32 m), in range it stops and holds, with none left it goes on. Orders to the same point in one tick
  share a nav group (one flow field). Nothing attacks on its own any more.
- `ae_bot::SkirmishBot` (new crate): a host-side command generator for one allyteam. Every 30 ticks
  (its own phase) it counts its idle units (empty queue, past the rally) and, with `wave` or more,
  attack-moves all of them at the enemy base. It reads only its own `Team`/`Queue` rows and the base
  positions every player knows; its commands go into the tick bundle like a player's, so replays
  record them and lockstep peers do not run each other's bots. `SkirmishBot::pair`: waves of 8 for
  allyteam 0, 16 for allyteam 1.
- Golden (`ae_services/tests/skirmish.rs`): the pair plays a 2 km land map from 6 tanks each;
  allyteam 1 wins at tick 6,363, digest `40c9e1b9faa89d3b33ab7d8aee5eb74e`, the same at 1 and 2
  workers and in dev and release builds. It replaces the handicap match (bits 48..56 still work).
  The test also checks that attack-moves engage (24,411 unit-ticks chasing, 2,070 holding, 106
  returns to heading on), since weapons fire regardless of orders and a wave that only walked would
  win too. A second test checks the queue: replace, append, pop on arrival, stop.
- `ae-headless --script bot` (record, and `--connect` where player P is bot P): a recorded bot match,
  6,400 ticks with 184 commands and 214 checkpoints, 133 KB, verifies at another worker count; the
  replay end-to-end test records 1,200 ticks of it. Over the relay (2 bot clients, 30 ms + 0..20 ms
  jitter + 5% loss, 1,500 ticks): 29.9 Hz, 40 commands, 50 of 50 digest rounds in sync, and the
  relay's recording verifies at 2 workers (run by hand, not a test). A lockstep client sends at most
  32 commands per order tick, so a larger idle pool goes over several bot decisions.
- `ae_bot::unit_command` encodes a `UnitCommand` for the bot, the client and the tests.
- `ae-client --game skirmish`: right click moves, ctrl+right click attack-moves, shift queues;
  allyteam 1 is the bot (with `--observer all` both are). Clicks carry shift/ctrl state.
- Not done: no ownership check on commands (the game cannot see who sent an event, and the relay does
  not check either; runtime-supplied sender metadata is the fix); idle units do not guard (they
  only shoot what comes in range); the bot does not scout, defend or build; archives and manifests
  in `ae_vfs`, and `.aemap`, are still to do for the exit criterion "defined only by an archive".

Archives and packages (`ae_vfs`):
- Zip archives mount like directories (`Vfs::mount` takes either). The reader takes stored and
  deflated entries (`miniz_oxide`), checks every entry's CRC-32 and size, and refuses encryption,
  zip64, multi-disk, duplicate paths and non-content paths. The writer is deterministic (sorted
  entries, a fixed 1980 timestamp, deflate level 6 unless stored is smaller), so a package's bytes
  follow from its files alone. Python's `zipfile` reads our archives and we read its own; an
  archive of a directory has the directory's content hash.
- A package is a directory or archive with `manifest.json` (`name`, `version`, `kind` = game, map,
  mod or library, `depends` = names or `name@version`; unknown keys refused). The manifest is
  metadata: not a content file, not in the content hash (so the skirmish content's hash, and its
  replays, are unchanged by its new manifest). `Library::scan` finds packages in directories (zip
  files by signature, any extension); `resolve` orders a package after its dependencies (each once,
  depth first), naming cycles, missing packages and who needed them, and two versions of one name.
- `ae-tool pack DIR OUT` writes a package directory as an archive and prints its content hash;
  `ae-tool unpack-list FILE` reads and checks every entry.

Map packages (`ae_map`) and games from archives:
- A map package (kind `map`) is `map.json` (version, side in cells, terrain seed, water level in
  dm or none, a terrain-type table of name / cost / solid / material, start points, description) and
  two PNGs an image editor opens: `height.png` (16-bit greyscale, ground height in dm, 0..=2400,
  before water) and `types.png` (8-bit, indices into the type table). Colour and normal detail stay
  in the shared tilesets. Loading checks sizes, bit depths, height range, type indices, materials and
  unknown keys, naming the file and cell. `GameMap::to_terrain` floods water last, as the generators
  do; `from_terrain` makes one type per distinct cost / solid / material. Both crossing maps go
  through export, a zip and loading bit-identical (terrain digests equal), and the files are
  deterministic.
- `Map::Loaded` + `NavPlugin::loaded(terrain)`: the nav service on a loaded terrain (land movement
  classes; `scenario::terrain_plan_with`, no units of the map's own unless asked for).
- `ae-tool map-export land|water CELLS DIR [NAME]` writes a generated map as a map package;
  `games/skirmish/maps/crossing` (package `skirmish-crossing`, 256 cells, 6 types, 20 KB) is the
  land map so exported. `ae-tool pack DIR OUT --module GAME` adds the built sim module as
  `module.wasm`.
- `ae-headless` builds every run in one place (`setup.rs`): a mount stack from `--package SPEC|PATH`
  (repeatable, each with its dependencies from `--packages DIR` libraries) and `--content DIR`; its
  `module.wasm` is the sim module (else `--game`), its `defs/` the catalogue, its `map.json` the map
  (else `--nav`). Headers add `packages = name@version,...`; a replay of a package run finds them
  in `--packages` by name (without one it says so). Headers of runs without packages are unchanged.
- Test `ae-headless/tests/packages.rs`: the skirmish's game package (content + `module.wasm`) and
  the map package, zipped into a library, play a 1,500-tick bot match with no `--game` and no
  `--nav` and end at the same digest as the source path (game crate, content directory, generated
  map); the recording replays from the library by name. It also checks the committed map package
  still equals the generated map. (By hand: the 6,400-tick match from archives ends at the same
  `ffc0d8ea4ea2f386` as from source.)
- Start points reach the game: `ae_sdk::nav::MapStart { index, count, x, y, extent }` (engine id
  `0xAE00_0000_0000_000D`), emitted once at tick 0 by the nav plugin's new `map_starts` host system
  from `NavPlugin::starts` (a loaded map's `starts`; generated maps have none). The skirmish
  schedules `host(map_starts)` first: its bases are the first two start points (else, on a generated
  map, the parameter's corners), kept in `Match.base_x/base_y` (view `always`) and published every
  tick as `Base` events for `orders` (rally points). Starting tanks default to 6 when the parameter
  does not say, so a map package with starts needs no parameter. The bot reads the bases from
  `Match` (`ae_bot::bases`) instead of recomputing them.
- The event store's digest leaves out streams with no readers that never held an event: adding
  `MapStart` to the SDK put its record into the crossing game's schema too (the linker keeps the
  object's records) and moved crossing's pinned digest; with the rule it is back to `286918c1…`.
- Golden re-pinned on purpose (larger `Match`; the rally now waits for the `Base` events): allyteam
  1 wins at tick 6,457, digest `b7159ef63b6848f4612f49b4dd2522b7`, 1 and 2 workers, dev and release.
- The archive test now runs the packages with no game parameter at all. Its state equals the
  source run's in every digest part but the event store's: the store hashes stream positions, and
  the archive run has a `MapStart` stream the generated map does not.
- `ae-client --package SPEC|PATH ... [--packages DIR]`: game, content and map from packages
  (`ae_vfs::resolve_specs`, shared with `ae-headless`); the map package is drawn and simulated
  (`--map` is ignored). Checked by hand: windowed with two bots, and `--capture` (12 of 12 units
  visible).
- Not done: no match file (teams, players, handicaps as data); no resource placement in maps; no
  import settings or hash-keyed cache; packages are not synced between lockstep peers (each needs
  the same files; the header's content hash refuses others).

One archive is the whole game: `ae-tool pack games/skirmish/content OUT --module skirmish --with
games/skirmish/maps/crossing` (59 KB: content, `module.wasm`, the map's files; the map's manifest left
out), run as `ae-headless --package OUT --units 0 --script bot` with no library, `--game`, `--nav` or
parameter. `packages.rs` plays it for 6,500 ticks (the match is decided at 6,457). Its state equals the
source run's in every part but the event store's, and its total digest `5bdc31cf5379b7760c4b88f1e82fe5e4`
is pinned, the same in dev and release: the bot-vs-bot golden of the archive-defined game. With
packages, `ae-client` passes parameter 0 unless `--param` is given, so the archive's defaults rule (6
tanks, the map's starts), as in headless.

Benches at the end of M6 (`ae-tool bench all --label m6`, CPU suites; `bench-diff` against
`bench/results/baseline`): 287 metrics, 0 regressed, none of them affected by M6's changes for the
worse (the event-digest rule, the new sim systems). The GPU suites were not rerun; nothing in render
changed. Not done: no benches of their own for `ae_vfs`, `ae_map`, `ae_catalog` or `ae_bot`, or a
skirmish scenario bench.

M6 is complete with the caveats above (no ownership check on commands; no match file; the bot only
attacks in waves; the client's mouse orders (right click, ctrl, shift) have only been smoke-run, not
played by hand). Next is M7 (water, shadows, particles, decals).

M2's exit command also runs as `ae-tool navbench --scaling` (forwards to `ae_nav`'s `navbench` binary in
the same target directory).

### M7 progress

Water (`cargo test -p ae_render --test gpu water -- --ignored`):
- Scope against the plan's `water_ra`: RA3's pass order (wave-particle simulation, planar reflection,
  refraction copy, underwater pass, the surface, spray) is not ported. What is ported is its
  surface's composition, cut to what reads at RTS distances: the ground under the water, tinted by
  its depth (as since M3) and now lit by caustics, is the refraction and underwater part, and the
  surface blends over it. A deliberate deviation, not `water_ra` done: no planar reflection (the
  surface reflects an analytic sky), no refraction distortion, no displaced geometry, no wakes, no
  bubbles or spray. Those would need a reflection pass of the terrain (its LOD is the main
  camera's) and copies of the scene, and are worth it close up, not at 1–3 km.
- Library pass `water.draw` (graphics; one colour attachment loaded, the depth buffer as
  `DepthRead`; no library resources). `Attachments::depth` is now `DepthUse::{Write, Read}`, so
  `library::check` takes `DepthRead` for the water pass only (and refuses it for the terrain and
  units, a depth write for the water, and the pass in a scene without water; unit tests).
  `LibraryEnv::water`; the native graph declares the pass after the units (the target's third
  writer, the depth buffer's reader). New goldens `scene_units_water_present` and
  `scene_water_readback`; the existing ones are unchanged.
- Opt-in like the tileset and vision: `TerrainApp::with_water()` (on a map with a water level),
  `TerrainView::water` switches it per frame (the pass records nothing; the terrain drops its
  caustics). `TerrainView::time` (new): animation time, by default the wall clock when presenting
  and the frame index / 60 when reading back, so captures and tests are deterministic.
- Render modules: `DECLARE_WATER` (flag bit 2; a module that ignores it draws no water, so the
  contract stays at version 1), `DeclareEnv::water` in the SDK, `when: water | no_water` in graph
  files (every file is checked for all eight flag combinations). `render/default` and `render/post`
  (code and `.graph.json`) declare the water pass; the default module still declares the native
  graph pass for pass in all eight scenes.
- The surface (`water.vert`, `water.frag`): one quad at the water level over the map, so the depth
  test against the terrain and units decides where water shows (keeping early-Z; no shoreline
  geometry). Eight analytic wave trains from 48 m to 0.75 m (deep-water speeds) shade it; each fades
  out as it nears three pixels and its slope becomes roughness that widens the sun's highlight
  (RA3's `rough`), so the glitter neither aliases nor disappears with distance; waves calm in the
  shallows. Fresnel (Schlick with F0 0.06, raised from water's 0.02 so the ripples read from straight
  above) weights an analytic sky; the sun's glints
  are added light, hence a new premultiplied blend (`pipeline::Blend::{Off, Straight,
  Premultiplied}` replaces the bool). Shore foam: crests running in through the last 1.5 m and a
  line along the waterline, laid on by drifting noise (RA3's `shoreWaves`, by depth instead of a
  baked travel time). The waterline fades over the first 0.5 m (the mesh and the plane meet there).
  Fog of war and distance fog apply to the surface's share, so the blend gives the fogged mix of
  surface and ground. Vision lookup, fog and noise moved to `terrain_shading.glsl`, shared with the
  terrain.
- GPU test `water_covers_the_low_ground_only` (flat map, land half 20 m above the water, water half
  20 m under it, an island 35 m high in it, a land unit and a ship; validation on): water added but
  switched off is byte-identical to no water; with water, 0 of the land and island points change
  and 1,661 of 1,678 water points do; the ship keeps its colour; never-seen water gets half of what
  the surface adds in sight (7.2 vs 14.3); half a second later only water points change. Turning the
  pipeline's depth test off fails it (the island's far edge: over land the waterline fade alone
  shows nothing, which the first version of the test, without the island, did not catch). All 16
  GPU tests pass with validation.
- `ae-client`: water on by default on wet maps, `--no-water`, O toggles it. With validation: a
  capture with two teams (fog of war on the lake) and an 8 s windowed `--exercise` (7 swapchain
  recreates) with 2,000 units, 0 messages. Through the render modules (`--render-module default`
  and `post`, the lake view of the README): the water pass runs in both graphs (in `post` into its
  intermediate image, then graded), 0 messages.
- Cost (`ae-client --bench all --label m7-water` against the baseline; the five water-map scenes now
  draw water): `water` 2–3 µs in the close and army views, 9 µs over the overview, 23 µs in the
  orbit; 0 of 81 earlier metrics regressed (terrain draw +2–5 %, inside the tolerance).
- Found on the way: `.gitignore`'s `*.dot` kept the golden plans out of git, so the golden test
  failed on a fresh checkout; they are now tracked.
- Not done: everything listed under scope above; no underwater view (the surface is not drawn
  with the camera below it); one water level per map (the nav terrain has one); foam is thin on the
  crossing map's steep banks; the colours are tuned by eye on a few captures.

The hills' shadows (`cargo test -p ae_render --lib sun`, `cargo test -p ae_render --test gpu hills -- --ignored`):
- The plan's "cascaded shadows" split in two. The terrain's shadows are not a shadow map: the sun
  is fixed and the terrain is a heightfield, so `terrain::sun` computes per heightmap corner the
  *shadow height*, the highest point above it that terrain toward the sun shades (`S(p) = max over
  t of H(p + t d) - t tan(e)`), and every receiver compares its height with it. Exact at every
  distance, no depth bias or acne, no cascades, and view-independent: hills off screen or behind
  the camera cast into the view (a shadow map drawn from `terrain.leaves` would not, the draw list
  being culled and refined for the main camera). The units' own shadows (casters that move) are
  the next step, a shadow map of the units only.
- On the CPU, exactly: each corner marches toward the sun one cell at a time over the bilinear
  heights and stops when even the map's highest point could not rise above what it found, rows
  split over threads; 51 ms for the crossing map's 2561² corners, on the render thread when the
  renderer is made. A first version swept the map once against the light
  (`S(p) = max(H(p), S(p + s d) - s tan(e))`, interpolating between rows): O(n), but the
  interpolation diffuses thin ridges' shadows away (a 3-corner tower's lost half its height over
  10 steps; the unit test caught it). Terrain edits keep it current: `TerrainEditor` holds the
  shadow heights and `sun::update` recomputes the edited rectangles and as far away from the sun
  as any shadow can reach (5 ms for a 10x10 edit on the crossing map, mostly scanning for the
  height range); `TerrainUpdate::shadows` carries the regions (unit test: equal to a recompute
  from scratch; the editor's test: equal to a map built from the edited cells).
- Receivers: an R32F texture (binding 14 of the terrain set, binding 5 of the units'), flag 256
  (`FLAG_SHADOWS`) and `counts.z` bit 0 for the units. The terrain compares against the bilinear
  height at the pixel, not the mesh's height (the LOD error would read as shadow on convex
  slopes); units and the water surface against their own height. Light fades over 0.8 cell of
  depth into the shadow (the edge follows the corners, so a sharp edge shows their staircase).
  Shadows dim the sun's term only (sky light stays); on the water they take the glints, and the
  caustics under it go with the ground's sun term. One sun for every pass: `sun::direction()`
  (the terrain and units had each their own copy of it).
- Opt-in like water: `TerrainApp::with_shadows()`, `TerrainView::shadows` per frame; `ae-client`
  on by default, `--no-shadows`, H toggles. No graph change (no new pass or resource; the goldens
  are unchanged).
- GPU test `hills_shade_away_from_the_sun` (flat map, two 90 m towers, one in view and one whose
  every corner is off screen, a unit in a shadow and one in the sun, validation on): shadows made
  but switched off are byte-identical to none; 0 of 7,132 sunlit ground points change; 23 of 24
  points in the in-view tower's shadow and 32 of 32 in the off-screen tower's lose a quarter of
  their brightness or more; the shaded unit darkens (184 to 97; its colour kept off white so the sunlit one is not clipped), the sunlit one is unchanged;
  after a terrain edit levels the first tower its shadow is gone (0 points differ from no shadows)
  and the other's remains. Making the shader's lookup return 1 fails it, as does dropping the
  edit's shadow regions in `TerrainRenderer::apply`. All 17 GPU tests pass with validation;
  `--quake 60` with shadows (an edit at run time) and the generated map's mountains, 0 messages.
- Cost (`--bench all --label m7-hill-shadows` against `m7-water`): the terrain draw +3–5 % in five
  scenes (5–13 µs: two more texture reads per pixel) and +20 % in `palette` (45 to 54 µs, the
  cheapest terrain shader, flagged by `bench-diff`); 85 of 86 metrics unchanged.
- Not done: the shadow edge is a cell soft whatever the distance
  to the caster (no contact hardening); the sun is fixed (a moving sun would recompute, 51 ms);
  `TerrainApp::create` builds the textures from the load-time map, so a renderer made again after
  edits loses them (as the heights did already).

The units' shadows (`cargo test -p ae_render --lib shadow`, `cargo test -p ae_render --test gpu units_cast -- --ignored`):
- A shadow map of the units only (the hills keep their shadow heights): library image
  `shadows.map` (D32, three 1536² cascades side by side, `shadow_map_spec()`), written by library
  pass `shadows.draw` (depth only, every row of the unit stream: casters off screen or behind the
  camera still cast into the view). Depth grows toward the sun, cleared to 0, GREATER_OR_EQUAL; the
  receivers read it with a comparison sampler (linear: 2x2 PCF), pushed off their surface by
  `(1.5 n + sun)` texels, and take `min(hills, units)`.
- Cascades (`src/shadow.rs`): slices at 0.95, 1.55 and 2.8 times the camera distance, each a
  sphere computed from the camera's intrinsics (not its corners: the radius moved with float noise
  as the camera panned, and the map shimmered), its centre snapped to whole texels in light space;
  casters reach 400 m toward the sun. Receivers take the first cascade that holds them; the last
  fades out at its edge. Unit tests: the view lands in the cascades; panning moves by whole texels
  and turning keeps the size.
- Graph contract: a graph that declares `shadows.map` must have every receiver (`terrain.draw`,
  `units.draw`, `water.draw`) sample it in the fragment stage and some `shadows.draw` write it
  (`library::check`, `check_shadows`); `DECLARE_SHADOWS` (set with units and shadows on) tells a
  module to declare them; graph files take `when: shadows`. A module that declares nothing keeps
  working, without unit shadows. The default and post modules and both `.graph.json` files do;
  two new goldens, the old ones unchanged. The renderer always binds an image (a 1x1 stand-in when
  off: terrain binding 15, units binding 6).
- Toggles with the hills' shadows (`--no-shadows`, H). Fog of war needs nothing: the casters are
  the observer's stream.
- GPU test `units_cast_shadows` (flat map, a unit pointing at the sun in view, one whose every
  corner is just off screen, one in the sun; a CPU ray cast over the darts splits the ground into
  deep shadow and lit; validation on): 75 of 75 and 110 of 110 deep-shadow points darken; the lit
  ground and the sunlit unit's lit faces are byte-identical; shadows made but switched off are
  byte-identical to none. Making the lookup return 1 fails it, as does drawing no casters. All 18
  GPU tests pass with validation (one at a time: in parallel they run out of memory on this 8 GB
  card with a desktop running, at HEAD too); the client with 2,000 units, 0 messages.
- Cost (`--bench all --label m7-unit-shadows-3` against the previous commit): in the three unit
  scenes the `shadows` pass is 36–43 µs and the terrain draw +17–28 µs (+9–11 %), totals
  +12–14 %; the scenes without units unchanged. The lookup's mere presence in the terrain shader
  cost 18 % even where it returned at once (`cliffs`, 377 to 448 µs: registers, not work), so it is
  a specialization constant (`UNIT_SHADOWS`) and the terrain pipeline is built without it when
  there is no map. Sampling once after choosing the cascade, or `textureLod`, changed nothing.
- Not done: the cascades are a stand-in. The intended shadow map is a virtual one
  (https://graphics-programming.org/blog/virtual-shadow-maps: one huge virtual map, pages
  allocated where the view needs them, cached while nothing moves); it will replace the cascades
  behind the same `shadows.map` / `shadows.draw` contract. No contact hardening; the water and
  units shaders keep the lookup compiled in (they exist only with units or water).

The rest of M7, in order: particles, then particles' coloured shadows, then decals (then the
stretch goal, volumetric smoke). Stop for a context compaction after each. All three are done
(below), and so is the goal of decals that look like Recoil's scars. Next is the stretch goal.

Particles (plan):
- Goal: a strong system, not a demo. It must handle tens of thousands of live particles and
  overlapping explosions over a battle, blend without visible popping, and cost nothing when
  nothing burns.
- Measured starting point. `~/particles`: 100k particles in one fire-and-smoke column, 1280x720,
  this machine.
  - Mode 1 (GPU radix sort by centre depth, instanced quads, premultiplied "over"): 24.5 ms.
  - Mode 3 (3DGS-style tile-binned compute blend, front to back with early-out): 67.9 ms.
  - Mode 2 (LBVH, exact front-to-back per pixel): 101.3 ms.
  - `~/wavelet_tr` (Haar-wavelet OIT, 16 coefficients): poor with many overlapping layers, and
    slow.
  - Under heavy overlap, exact per-pixel ordering costs 3–4x. Approximations with a fixed number
    of terms per pixel (wavelets, moments, weighted blended) break down exactly there. The cost
    is fill rate, not sorting.
- Two blend classes, separated by the effect's definition:
  - *Emissive* (fire, sparks, flashes, glows) is additive. It is order independent by
    construction, so it needs no sort.
  - *Media* (smoke, dust, steam) is premultiplied "over" and needs an order. That is where order
    independence matters.
- Media start as mode 1:
  - **Sort.** The GPU radix sort (`~/particles` `radix_*.comp`) orders particles by centre depth.
    The key breaks ties by particle index, so equal depths never swap between frames.
  - **Soft edges.** Each quad fades over a per-particle thickness against the scene depth. This
    hides the cut where a quad meets the ground or a unit.
  - **Fill rate.** Draw into a reduced-resolution target (half, then measured) and upsample it
    over the scene, with a depth-aware filter on a min/max-downsampled scene depth.
- Candidates for media, each measured against an exact per-pixel reference (mode 2 style, tests
  only) for error and cost before the design is fixed:
  - moment-based OIT (4 power moments);
  - mode 3 at reduced resolution;
  - a froxel volume for large smoke (Hillaire, Frostbite 2015). It is order independent,
    integrates with the distance fog, and can give the sun's transmittance through smoke.
  - Keep whichever wins on both.
- **Simulation.** On the GPU: a compute pass, a ring buffer, and emitters uploaded each frame.
  - Emitters come from the observer's view stream (shots, hits, deaths, burning units), so fog
    of war hides them.
  - Visual only: not deterministic, not in the digest.
  - Lighting: the sun times the hill and unit shadows, looked up once per particle at its centre;
    sky light; fog of war dims particles as it dims units.
- **Library contract**, like the units':
  - passes `particles.sim` (compute) and `particles.draw` (graphics, depth read);
  - library buffers `particles.*`;
  - `DECLARE_PARTICLES`, and `when: particles` in graph files;
  - new goldens, both modules, and both graph files.
- **Bench.** A new `--bench` scene with a battle's worth of explosions. The existing scenes must be
  unchanged: no particles means no pass.
- **GPU tests** (validation on):
  - Shuffling the emissive particles' submission order gives a byte-identical image.
  - A blue smoke particle in front of a red one shows blue, whatever the order they were emitted
    in.
  - The soft fade leaves no hard line on the ground.
  - Particles in hidden cells are not drawn.

Particles, first step (`cargo test -p ae_render --test gpu particles -- --ignored`, `cargo test -p ae-client vanish`):
- `ae_render::particles`: a GPU pool, a ring of up to 65,536 slots (the sort key's slot bits).
  - Bursts (`Burst`: position, effect, count, scale, seed; `ground` puts it above the terrain or
    the water) come through a `ParticleFeed`. Effects are data (`EffectDef`: colour and size over
    the life, speeds, `up`, spread, gravity, drag, emissive, fade-in); `default_effects` has flash,
    fire, sparks, smoke and dust, and `explosion` combines four of them.
- Passes:
  - **`particles.sim`** (compute) records its own barriers between five steps: spawn (it also
    resets the count), simulate, cull and key, then a 4-pass 8-bit LSD radix sort.
    - The radix sort is ported from `~/particles`: keys only, and the count is read from the
      indirect buffer.
  - **`particles.draw`** (graphics) is one indirect instanced draw of camera-facing quads.
- **One sorted list, one blend.**
  - All particles are sorted back to front by centre and drawn premultiplied. Emissive particles
    add light and cover what is behind them by their effect's `cover` (0: alpha 0, light only;
    fire 0.5 and flash 0.3 since 2026-11-22), and fire behind smoke is dimmed by it.
  - The key is `(far-to-near log depth, 16 bits) << 16 | slot`. Unique keys make the order
    independent of the order the simulation's invocations appended them in: the same bursts give
    the same frame. Particles in one depth bucket fall back to slot order, which follows the order
    of emission.
  - This is the plan's "media sorted by centre" applied to everything; the separate emissive
    class was dropped as unnecessary.
- **Soft edges, and no depth attachment.**
  - The draw samples the depth buffer (`sampled:fragment`, a new `DepthUse::Sampled`).
  - It treats each particle as a sphere of its size, fading where the sphere's front passes behind
    the scene; this is also its depth test.
  - The depth buffer is now declared with sampled usage in every graph (native, both modules,
    both graph files). The existing goldens and bench scenes are unchanged.
- **Look.** A 128² puff texture generated at start (mip 0; the GPU makes the mips): a radial
  falloff broken up by four octaves of value noise.
  - Each particle gets a random rotation and brightness.
  - Smoke and dust are lit by the sun (past the hills) and the sky, and fogged. Emissive
    particles fade in the fog.
- **Contract.**
  - Library resources `particles.state`, `particles.keys`, `particles.indirect`.
  - `DECLARE_PARTICLES` (`DeclareEnv::particles`); graph files take `when: particles` /
    `no_particles` (`SCENE_FLAGS` doubled to 24).
  - One new golden. The module test now also runs with particles.
- **Client.**
  - Particles are on by default (`--no-particles`).
  - On the sim thread, a unit whose entity is gone from the capture explodes where it stood, if
    the observer's line of sight of the tick before covered that cell. A unit dying in the fog
    leaves nothing, and its own sight dying with it does not hide it.
  - X sets off an explosion at the camera target; `--explode N` sets off N before a capture.
  - The skirmish game shows them. No other events reach the client yet: no shots, no impacts.
- **GPU test `particles_sort_blend_and_soften`** (validation on):
  - A blue puff in front of a red one shows blue, and red when swapped. The frame is
    byte-identical whichever was emitted first (their depths differ).
  - A puff under the ground draws nothing, and nor do dead particles or an empty pool (both
    byte-identical to no particle renderer).
  - A puff cut by the ground changes by at most 6.1 between neighbouring pixels, against a
    strongest contribution of 70.7.
  - An explosion of the default effects changes the frame, and is byte-identical when rendered
    again: about 90 particles at clustered depths, where the slot bits decide.
  - Skipping the sort fails it, and so does a hard depth test instead of the fade.
  - Unit test: deaths only where seen.
- **Cost.** New bench scene `battle`: the army view, closer, with 2 explosions a frame, about 40k
  live particles.
  - `particles sim` 58 µs, `particles` (draw) 173 µs. The other six scenes are unchanged against
    `m7-unit-shadows-3` (89 of 109 metrics compared, 0 regressed).
  - An idle renderer (the default client before anything dies) costs the empty draw pass,
    2.4–3 µs: the depth buffer's move to sampled and the target's load and store. The sim pass
    records nothing (`m7-particles-idle`: army, overview, orbit, cliffs; totals within noise).
  - The sort first cost 130 µs: the port's 32 blocks per workgroup suit millions of keys and gave
    a 64k pool only 8 workgroups. With 4 blocks it takes about 42 µs; 1 and 2 blocks were slower,
    because the scatter's prefix walks every workgroup.
- **Not done yet:**
  - Order independence: measured below ("order independence measured"); volumetric smoke and
    dust is the stretch goal at the end of M7.
  - Particles are not shaded by the units' shadow map (they cast coloured shadows since; see
    "Particles' coloured shadows").
  - Fog of war does not dim them (their emitters are only where the observer sees).
  - Effects are not in the content catalogue.
  - No continuous emitters (burning wrecks, dust behind moving units).

Particles, second step: half resolution (same tests, plus `bench battle`, `--particles-full`):
- **The `particles` pass draws at half resolution by default** (`ParticlesSetup::downscale` 2; 1
  keeps the full-resolution draw straight into the scene).
- **Inside the pass, outside the graph.** The graph opens the pass's rendering scope on the scene
  colour before the renderer records. The renderer:
  - ends that scope;
  - reduces the scene depth to the farthest of each 2x2 block (R32F; a fragment pass, so the
    graph's barrier for fragment-stage sampling covers it);
  - draws the particles into its own RGBA16F target, cleared, softened against that depth (the
    same shaders: they read the half-resolution depth);
  - begins a scope on the scene colour again (load, store) and composites over it, which the
    graph then ends.
  - Each frame slot has its own two targets, made when it first draws and remade on resize, with
    its own barriers. The pass's contract, the modules, the graph files and the goldens are
    unchanged.
- **Depth-aware composite.** Per pixel, the four nearest half-resolution texels:
  - bilinear when all four were softened against depths within 3 % of the pixel's view depth;
  - otherwise the texel whose depth is nearest, so smoke behind a silhouette stays off it.
  - Pixels with no particle are discarded, so untouched pixels stay byte-identical.
- **Tests.** The particle GPU test runs all its checks at both resolutions; the thresholds hold
  unchanged (half: blue/red still dominant, ground cut 5.6 of 70.7). New: a puff behind a
  plateau's rim. The plateau is untouched at full resolution, and at half resolution 0 of its
  pixels change; a plain bilinear upsample changes 55 of them, by up to 65 in luma.
  - The other direction: none of the 837 pixels the puff shows on above the rim loses more than
    half of it at half resolution. This only guards against a gross halo. No variant broke it:
    the nearest instead of the farthest depth per block gives the same image here, because the
    nearest-depth texel covers both.
- **Cost** (`bench battle`, about 40k live particles, 1280x720): the `particles` pass 85 µs at
  half resolution against 171 µs at full (`m7-particles-half`, `m7-particles-full`); the sim pass
  is unchanged at 58 µs. Idle should be unchanged (not re-measured): by the code, an inactive slot
  neither makes targets nor records anything inside either pass.
- Validation clean: the GPU tests, the bench, and captures with explosions natively, with the
  post module and with both graph files. Resizes: `--exercise` now also sets off explosions every
  second; 12 s windowed, 11 swapchain recreations with particles alive, 0 messages.

Particles, order independence measured (`cargo test -p ae_render --test gpu particles_order -- --ignored --nocapture`):
- **Three blends**, all at half resolution (`ParticleBlend`, `--particle-blend sorted|moments`):
  - *Sorted*: the centre sort, as before.
  - *Moments*: moment-based OIT (four power moments, Peters' reconstruction). Three passes: the
    moments (R32F + RGBA32F, additive), the colour weighted by the reconstructed transmittance
    (media and emissive summed apart), and a resolve. No sort. The depth warp is the log over the
    frame's range of visible particles, which the simulation finds (a subgroup min/max, only in
    this mode).
  - *Reference* (tests only, needs `fragmentStoresAndAtomics`, now enabled when present): every
    fragment stored per pixel (64 at most; magenta past that), sorted by its sphere's front
    depth, blended. Its depth is the same front depth the soft fade uses: sorting by the quads'
    own depth would equal the centre sort, since the quads face the camera.
- **A dense cluster** (48 smoke puffs of four colours, 8–30 m, and 10 fire puffs in a 40 m ball;
  11,673 pixels): error against the reference, per channel, sorted mean 1.57 (max 51), moments
  mean 1.43 (max 32). The centre sort is already close to exact order.
- **Two puffs crossing in depth** (red 16 m, blue 26 m, opacity 0.85, a metre a frame): the
  largest change between consecutive frames, per channel:
  - sorted 116 (208 pixel-frames over 40), reference 116 (179), moments 115 (372).
  - With the warp range pinned to constants, moments give 112 (178): their extra popping is the
    per-frame range moving, not the method. Their error against the reference is then lower
    (mean 0.41, max 34; sorted 1.14, max 116).
  - Luma first hid the swap: red and blue have similar luma, so the jump measured 14.
- **Conclusion.** The popping comes from blending particles as flat layers, not from the order:
  the exact per-pixel order pops as much as the sort. No layer method (moments, tile-binned,
  k-buffers) removes it.
  - What does is treating smoke as a volume: overlapping particles mixed by how much of their
    thickness overlaps, which changes continuously as they pass through each other.
- **Cost** (`bench battle`): moments 17 µs sim (no sort), 552 µs draw, against sorted 59 + 86 µs
  (`m7-oit-moments`, `m7-oit-sorted`). The sorted path is unchanged.
- **Kept:** the sort for everything. Moments and the reference stay as comparison modes.
- **Next:** volumetric smoke and dust, moved to a stretch goal at the end of M7 ("M7 stretch
  goal", after decals).

Particles' coloured shadows (plan):
- **Recoil's method.** It is WickedEngine's "easy transparent shadow maps";
  `ProjectileDrawer::DrawShadowTransparent` and `ProjFXFragShadowProg.glsl` implement it.
  - The shadow map gets a colour target, cleared to white.
  - After the opaque casters, the particles flagged `castShadow` are drawn depth-tested against the
    opaque depth, without writing it.
  - They blend multiplicatively: `dst *= 1 - a + rgb*a`.
  - Receivers multiply the sun by the depth test times that colour.
- **Here:**
  - library image `shadows.color` (RGBA16F, laid out like `shadows.map`);
  - written by a library pass `shadows.particles` after `shadows.draw`;
  - only for effects flagged to cast (smoke, dust; not sparks).
- **Recoil's colour has no depth.** Receivers above the smoke, nearer the sun, are tinted too, for
  example a unit flying over a smoke column.
  - Here alpha keeps the depth of the particle nearest the sun, with blend op MAX for alpha and
    multiply for colour.
  - A receiver is tinted only if it lies farther from the sun than that particle.
  - A receiver between two smoke layers is tinted fully. That is accepted.
- **Cost.** The colour read goes inside the lookup that `UNIT_SHADOWS` already gates. Check it
  with `bench-diff` on the terrain pass: the lookup's mere presence once cost that pass 18 %.
- **When the map exists.** Today `DECLARE_SHADOWS` comes only with units. With particles, the map
  and the colour target exist when shadows are on and there are units *or* particles.
  `shadows.draw` still needs units. `receives()` and `check_shadows` cover `shadows.color` too.
- **Test:**
  - A smoke column between the sun and the ground tints the ground in its shadow by the smoke's
    colour.
  - A unit above the smoke is not tinted.
  - Sparks cast nothing.
  - Shadows switched off are byte-identical to none.

Particles' coloured shadows (`cargo test -p ae_render --test gpu particles_cast -- --ignored --nocapture`):
- **Stored as optical depth, not a multiplied colour.** Recoil multiplies a target cleared to
  white by `1 - a + rgb a`. Here each caster adds `-ln(1 - a (1 - rgb))` per channel (blend
  ONE/ONE) into a target cleared to 0, and receivers take `exp(-sum)`: the same product, order
  independent, and the graph clears to 0 (`clear_color` is per pass; this one returns 0).
  - Alpha keeps, by blend op MAX (new `Blend::AdditiveMaxAlpha`), the depth toward the sun of the
    caster's sphere surface facing the sun **over that texel** (centre + `sqrt(1 - r²)` radii), not
    one value per particle. A receiver is tinted only beyond it, over a 2 m fade.
  - The plan's single "depth of the particle nearest the sun" failed the test: ground just up-sun
    of a low puff lies within its disc as the sun sees it and below its top, though no sun ray from
    it crosses the puff, and was tinted.
  - No depth test against `shadows.map`: receivers already multiply by the opaque test, and the
    alpha depth handles the order.
- **Contract.**
  - Library image `shadows.color` (`particle_shadow_spec`: RGBA16F, the cascades side by side at a
    quarter of `shadows.map`'s resolution, 1152x384).
  - Library pass `shadows.particles`: `color:clear`; it reads `particles.state` and `particles.keys`
    in the vertex stage and takes `particles.indirect` as an indirect buffer.
  - A graph that declares `shadows.color` also declares `shadows.map` (its cascades), has a
    `shadows.particles` pass write it, and every receiver samples it (`check_shadows`, `check`).
  - No new declare flag: modules declare it with `shadows` and `particles`; graph files take
    `when: particle_shadows`. Both modules, both graph files, the SDK and the ABI do.
  - The particles golden changes: the sim now runs before the shadow pass (it is the only writer
    of the state the pass reads, so the bake orders it first). The other goldens are unchanged.
- **Casters.** Every live particle that is not emissive, in view or not (off-screen smoke shades
  the view). `EffectDef::emissive` decides it; there is no separate cast flag.
  - The simulation lists their slots after the visible keys (the keys buffer is now twice the
    capacity). The second draw command in `particles.indirect` (32 bytes in, a 4-vertex strip per
    instance) draws them into each cascade.
  - Drawing every pool slot instead cost 148 µs in `battle`, nearly all vertex work.
  - The cascades reach the particles' uniforms through `ParticleFrame::unit_shadows`.
- **Receivers.**
  - `unitShadowAt` also returns the tint (terrain binding 16, units binding 7, water through the
    terrain's set). It reads inside the `UNIT_SHADOWS`-gated cascade loop, only while
    `axes[0].w` is set (`Cascades::tinted`: the graph draws the colour this frame).
  - The colour is clamped half a texel inside its cascade, since the depth map's 1.5-texel margin
    is under one colour texel.
  - The target (3.5 MB) exists whenever the units' map does, with or without particles; a 1x1
    stand-in otherwise.
- **Test** `particles_cast_coloured_shadows` (flat map, one unit, still effects, validation on):
  - Orange smoke 25 m up tints the 301 ground points deep in its shadow: red 1.00, green 0.86, blue
    0.77 of the unlit-by-smoke frame (sRGB bytes; the sky's light is not tinted).
  - The ground clear of the shadow is byte-identical (91,040 points).
  - The same puff off screen (no point within 1.6 radii in view) tints its 302 points in view.
  - The same puff emissive casts nothing.
  - A puff buried just under the unit (829 ground points within its disc as the sun sees it)
    leaves the frame byte-identical to no particles.
  - Shadows switched off are byte-identical to no shadows.
  - Each check fails when broken: no tint; no depth test; one depth per particle (the plan's);
    emissive particles casting; casters taken from the camera's list (0 of 302).
  - All 21 GPU tests pass with validation. So do the client captures with explosions (native,
    the post module, both graph files), the bench, and 12 s of `--exercise` (11 swapchain
    recreations): 0 messages.
- **Cost** (`--bench all --label m7-particle-shadows` against `m7-particles`; `bench-diff`: 0
  regressed):
  - `battle` (about 40k live particles): the new pass is 40 µs.
    - It was 80 µs at half resolution: the pass is bound by rasterising big smoke quads (56 µs
      with every fragment discarded). Strips instead of 6-vertex lists saved about 10 µs at
      quarter resolution and nothing at half.
  - Idle (particles on, nothing burning): 3 µs, the clear.
  - The terrain pass, within `bench-diff`'s tolerance but consistently, +0.7 to +3.5 % (2–6 µs) in
    the four scenes with unit shadows (army, orbit, overview; battle, where it samples). Scenes
    without unit shadows are unchanged.
- **Not done:**
  - Without units there is no shadow map, so smoke casts nothing (the plan wanted the map with
    units *or* particles). The client always has units.
  - Particles do not receive shadows (neither the units' nor each other's).
  - The cascades snap to whole texels of the depth map, a quarter of a colour texel, so as the
    camera pans the smoke's shadow is rasterised at another sub-texel offset (a slight shimmer;
    the smoke moves anyway).
  - A unit under smoke is tinted by the same lookup but not tested; the water's glints are tinted
    too, also untested.

Decals (plan):
- **Recoil projects decals through boxes** (`rts/Rendering/Env/Decals/GroundDecalHandler.cpp`,
  `GroundDecalsFragProg.glsl`). A volume per decal reads the scene depth and reconstructs the
  position. Its vertical extent has to be guessed:
  - too small, and the decal is cut on slopes;
  - too large, and it paints whatever else is in the box: units, or the far side of a cliff.
- **Here the receivers are known.** The terrain applies decals in its own shader (forward decals),
  so there is no volume.
  - The heightfield has one surface per xz, so a decal is a footprint: centre, half-size,
    rotation, a layer in a decal atlas, a fade, and a start time.
  - Units never apply decals, so they are never painted.
- **Binning.** A world-space 2D grid (cells of tens of metres) lists the decals touching each cell.
  It is updated when decals change; the terrain fragment loops over its cell's list.
  - A specialization constant compiles all of it out when there are no decals.
  - Check it with `bench-diff` on the terrain pass.
- **Corners and cliffs.** A top-down projection stretches a decal down a cliff face. Instead:
  - Project the decal three ways around its centre, as triplanar texturing does, weighted by
    the surface normal with constant-overlap weights (see `docs/antitiling.md`).
  - The side projections take v from the height relative to the decal's centre, so a decal across
    a cliff's edge or into its foot wraps onto both faces.
  - The footprint's mask fades with 3D distance.
- **Spawn.** Sim events (explosions leave scorch marks; deaths; building footprints) come through
  the observer's stream.
  - A fixed capacity: the oldest is evicted first.
  - A decal seen once stays as last seen under fog, like the explored terrain.
- **GPU test** (a step heightmap: a plateau with a cliff; one decal across the top edge, the convex
  corner, and one at the foot, the concave corner):
  - both faces of each corner are painted;
  - the texel stretch on the cliff face stays within 1.5x of the flat ground's;
  - the ground outside the footprints is byte-identical;
  - a unit standing in a footprint is not painted;
  - no decals is byte-identical to before.

Decals (`cargo test -p ae_render --test gpu decals -- --ignored --nocapture`, `cargo test -p ae_render --lib decal`):
- **Inside the terrain's draw** (`ae_render::decals`, `terrain_decals.glsl`): no pass, no library
  resource, no graph change; the goldens, the modules and the graph files are unchanged. Units and
  the water are never painted.
  - A decal (`Decal`: centre x, z, half-sizes, rotation, atlas layer, opacity, fade-in, life,
    fade-out) is placed through a `DecalFeed`; the app stamps its start time. `TerrainApp::with_decals` keeps a ring of
    `DecalsSetup::capacity` (the oldest goes first) across renderer re-creation.
  - The centre's height is read from the height texture on the GPU, so a decal follows terrain edits.
  - Atlas: two texture arrays (`DecalAtlas`: mip 0 of each layer as the GPU takes it, sRGB colour
    with straight alpha and the packed surface; the GPU makes the mips, the colour's weighted by
    alpha); `default_atlas(size)` generates a scorch mark and a crater. Each layer also carries its
    `LayerMaterial` (metallic; roughness comes from the layer's surface texture; M12 step 3),
    blended into the ground by the decal's coverage.
  - Specialization constant 1 (`DECALS`): the terrain pipeline with decals is built only with
    `with_decals`. Each frame it is used only while some decal's box (its square in plan, the map's
    height range) meets the view frustum. Decals never expire, so without this gate the first death
    would turn it on for good, at about 5 % of the terrain pass. With none in view, the frame is the
    one without decals (byte-identical, the same pipeline).
- **Binning: a spatial hash, not a grid over the map.** The world's 32 m cells hash into
  `buckets(capacity)` (four per decal, a power of two); each bucket lists its decals oldest first,
  so newer decals cover older ones. Another cell's decal in the same bucket fails the footprint
  test before anything else is read. A decal's bounding radius is clamped to 32 m (at most 3x3
  cells), so the buffers have a fixed size. The encode is a counting sort into buffers kept
  between frames.
  - The planned grid over the map was 320x320 cells on the 10 km map: in `battle`, which places
    two decals a frame, rebuilding it cost the CPU frame 1.7 ms (one `Vec` per cell), 73 µs as a
    counting sort with the hash. With each decal's words and buckets computed at placement, it is
    12 µs for 1024 decals.
- **Projection: unfolded over the ground** (replaced the first version's three projections, below).
  A point's offset from the centre is the plan ray from the centre to it, read every `stride`
  (one heightmap cell, or an eighth of the larger half-size if that is longer, so at most 12
  steps). Each segment is flattened along the ground's slope, isometrically: `d' = d + dh G /
  (1 + sqrt(1 + |G|^2))`, with `dh` the segment's own rise and `G` the slope at its start, middle
  and end averaged by steepness, so that across a fold the face's slope is used. The sum, turned
  into the decal's frame, is the uv. Each point's height and slope come from one gather of its
  heightmap cell.
  - Continuous in the point (a new step starts where the last segment has become that step), exact
    over a planar slope and over a fold at any angle (a cliff's edge or foot), and the plan offset
    on flat ground. There is one image, nothing to blend, so no seams.
  - Over a ridge or a valley within one segment, the rise and the steepest slope can disagree
    along the ray. The gain along the ray is held at 0 or more, so no point unfolds nearer the
    centre than it is in plan.
  - Not exact across a crease between two sloped facets. On the scripted map with seed 8, eight-pixel
    differences of the unfolded coordinate against the levelled tangent plane put the rate down
    the facet below a crease at roughly 0.85 to 0.9 of the ideal, about 1 on the plateau above it.
    Measured through lighting and the tonemap, so approximate.
  - The extent is the square in unfolded metres, faded over its last 15 %. Unfolding never
    shortens the distance from the centre, but on a slope across the decal's axes it can bring a
    point past the plan square's corner back inside. So the early-out is the bounding circle in
    plan, which is also what the binning lists.
  - Gradients and the normal map's frame come from the tangent plane at the point, levelled by the
    smallest rotation that takes the heightmap's normal to up.
  - Every atlas sample uses explicit gradients (the loop is non-uniform control flow).
- **Why not three projections.** The first version projected three ways around the centre (from
  above, and along the decal's x and z with `-sign(n) dy` in place of the hidden coordinate),
  blended by the heightmap's normal. The side projections agree with the top one only where the
  edge passes through the centre, and every decal in the first test was centred on an edge. On
  real ground a scar on a slope's edge was cut along a line (sharp blend) or smeared down the
  slope (soft blend). Two other attempts failed:
  - Unfolding u along the decal's rows and v along its columns: a fold diagonal to the axes adds
    its drop to both, so the map folds over itself (fingers down the slope), and T and B collapse
    on a shared drop (NaN normals).
  - A radial scale (uv = plan offset times arc length over plan length): it stretches a face
    beside the centre along the edge instead of unfolding across it.
- **GPU test** `decals_wrap_over_cliffs`. A step map with 2 m cells along x: a 30 m plateau ends
  in a 20 m cliff facing +x down to ground at 10 m, with a 6 m step up to 36 m behind it. The atlas
  has eight black and white stripes across the decal's x. Four decals of half-size 12 m: E on the
  cliff's top edge, F at its foot, L on the low step's edge (all three centred on corner samples),
  and G 4.5 m back from the top edge. That is a stripe and a half, so projections around the
  centre would put G's face stripes in opposite phase. A separate frame has only R, like G but
  turned by 0.6 rad, so the cliff crosses its axes diagonally.
  - R's expected coordinate: the ground unfolded in world (x by its length along the ground from
    R's centre), then turned into R's frame.
  - Sampled along x and then onto the face, by the unfolded coordinate, the stripes match the
    atlas everywhere at full opacity. Samples within 0.5 m of a stripe boundary, near the unit and
    in the plateau's last 2.5 m (where the heightmap normal already leans) are left out. Counts:
    - E: plateau 153 of 153, face 189 of 189.
    - F: ground 150 of 150, face 192 of 192.
    - G: plateau 180 of 180, face 150 of 150.
    - R: plateau 793 of 793, face 440 of 440.
  - The test checks phase, not only stripe count, so it also catches stretching: a face at 1.5x
    the flat ground's texel size would put its third stripe 3 m off.
  - Ground beyond every decal's unfolded square is byte-identical (33,087 points).
  - L no longer prints again at the foot of its step. The ground there is within the first
    version's 3D extent but past the unfolded footprint (6 m down the face, then along the
    ground), and is unchanged at all 78 points; the first version painted all 78.
  - A unit standing in F's footprint is unchanged, and the ground beside it is painted.
  - Decals made but none placed are byte-identical to no decals.
  - Mutations, each caught:
    - the first version's three projections (G face 78 of 150);
    - unfolding u along rows and v along columns (R plateau 753 of 793, just under the 95 % bar;
      the test stops at the first failing segment, so R's face was not reached);
    - a segment's slope across the ray taken at its middle only (G face 130 of 150);
    - the specialization forced off (no stripes).
  - All 38 GPU tests pass, run with `--test-threads=1`.
  - Cost: `battle`'s terrain pass 420 µs before, 423 µs after (`bench-diff`: 27 of 28 metrics
    unchanged, none regressed). From 900 m the scorch marks are a few pixels each, so this hides
    the cost.
  - The close measure is the bench scene `scorched` (no units, 120 m away, 2 explosions a frame
    over 100 m, so overlapping marks fill most of the view). Its terrain pass is about 2.55 ms and
    varies ±33 % within a run as marks pile up, so the unfold's own cost was measured by running it
    8 times per decal and dividing the extra time by 7:
    - slopes by forward differences (three bilinear taps a point, 9 + 6n a fragment): ~150 µs;
    - each point's height and slope from one `textureGather` of its cell (`groundAt`, 3 + 2n):
      ~93 µs. Kept.
    - Skipping the middle sample where the ends' slopes already predict a segment's rise: no faster
      (~100 µs); this map is rarely that planar, and the branch costs. Not kept.
  - `groundAt`'s slope is the bilinear surface's exact gradient, which steps at cell edges where the
    forward differences ramped over a quarter cell. On seeds 6 and 8 of the scripted map, 0.01 to
    0.02 % of pixels changed by more than 2 %, all in the cracks below the crease, with no seam.
- **Where it departs from the plan:**
  - The plan projected each decal three ways around its centre. It is unfolded over the ground
    instead (Projection, above), and its extent is the unfolded square, not a 3D distance.
  - The plan had a decal seen once stay as last seen under the fog. Decals now have a time to live
    (below), and they end under the fog too.
- **Client:** every explosion (deaths the observer sees, X, `--explode`) leaves a scorch mark of
  3.5 x scale metres through `sim::Effects`, which carries both feeds. Capacity is 1024;
  `--no-decals` turns them off.
- **Time to live** (added after the first commit):
  - `Decal::life` (0: until evicted) and `Decal::fade_out`; the shader fades the opacity over the
    last `fade_out` seconds, and the app drops a decal once its life ends (`DecalSet::expire`,
    each frame). A fade-out longer than the life starts the decal partly faded.
  - Scorch marks live 60 s (`SCORCH_LIFE`) and fade over their last 20 s. Recoil's scars live
    0.5 to 60 s, scaled by the explosion's damage and the decal level setting, with alpha
    decaying over the whole life.
  - Fixed on the way: the shader's time came from `u.water.w`, which was 0 on a map without
    water, so neither the fade-in nor the fade-out ran there. It is now the time on every map;
    the water code reads it only when the map has water.
  - Tests:
    - Decals placed at time 0 to live 0.05 s draw exactly as none by the last frame (7/60 s).
    - Decals halfway through a 1 s fade-out land a median 0.55 of the way from none to full
      (1092 samples, sRGB bytes; all between 0.33 and 0.70).
    - Before the time fix they were at 1.00.
    - A unit test covers the expiry and the encoded end times.
- **Cost** (`--bench all --label m7-decals` against `m7-particle-shadows`; `bench-diff`: 109 of
  110 metrics unchanged):
  - `battle` (two explosions a frame, so about 700 scorch marks live around the view): the terrain
    pass rises from 181 to 218 µs (+20 %).
    - Compiled in with an empty loop it is 190 µs, so the loop's presence costs 5 %.
    - Reading the centre's height from the CPU instead would save 6 µs; it would no longer follow
      terrain edits, so the GPU read stays.
    - The CPU frame is 114 µs against 117.
  - The other six scenes have no decals and are unchanged. In a game, the cost applies only while
    a scorch mark is in view.
- **Validation clean** (0 messages):
  - client captures with 8 explosions: natively, with the post module, and with both graph files;
  - a close capture without particles, which shows the twelve scorch marks;
  - 12 s of `--exercise` (11 swapchain recreations).
- **Not done:**
  - Deaths are the only events; there are no decals for impacts or building footprints.
  - Decals change the albedo only, not the normals.
  - A decal's height and normal come from the heightmap, not the LOD mesh. Where a coarse mesh
    sits off the heightmap, the decal follows the drawn surface (its world position) but blends
    its projections by the heightmap's normal.

M7 goal (after decals, before the stretch goal): decals that look like Recoil's scars.
- **Textures.** Recoil's scars, `~/RecoilEngine/cont/base/bitmaps/bitmaps/scars/`:
  - `scarN.tga`, each with a `scarN_normal.tga` (the bundled set is now ten 512² scars);
  - RGBA, 8-bit alpha. In the normal maps, rg is the tangent normal x/y (z rebuilt),
    b is roughness (linear, taken as the layer's roughness), and alpha is the glow
    (heat) mask (`normVal.w`).
  - Load them into the decal atlas, replacing or joining the generated scorch and crater. The
    licence must allow it, or use them only as a reference for our own; check before shipping.
  - The atlas gains a second array, the normals with the glow mask, sampled with the same
    gradients.
- **Shading**, from Recoil's fragment shader
  (`cont/base/springcontent/shaders/GLSL/GroundDecalsFragProg.glsl`):
  - **Relit by its normal map.** A TBN built from the ground's normal and the decal's rotation
    axis (Gram–Schmidt), the decal normal `TBN * (n * 2 - 1)`, and `NdotL` with the sun times
    the ground's shadow, plus ambient.
    - Here: blend the decal's normal into the terrain's shading normal by its alpha, so the hills'
      and units' shadows, fog and vision still apply in one place.
    - In our triplanar projection each axis needs its own tangent frame, as the tileset's
      `project()` already builds.
  - **Colour mixed with the ground.** For explosions Recoil uses `2 * decal * ground` (its
    minimap diffuse), so a scar darkens whatever is under it instead of painting one colour.
    Compare that with the current replace.
  - **Glow.** Each scar has a glow and a glow life (default 2 s, decaying linearly), with an
    "overglow" flicker near full (`+0.2 * |sin(0.02 frame)|`).
    - It is weighted toward the centre (`smoothstep(0.9, 0.1, d / r)^7`), killed under shallow
      water, and coloured by black-body temperature (`BlackBody(mask * mix(1, 3700, glow))`)
      added to the lit colour.
    - Here it is emissive: add it after lighting, untouched by the sun and shadows. It should
      also not be dimmed as much by fog of war, or shown at all in cells not in sight.
    - Optionally, a glow colour map instead of the black body (Recoil's `scarGlowColorMap`).
  - **Shaping.** Explosion scars are cut by an ellipsoid (`EllipsoidRedunction`) and by
    `dot(decal up, N)^exp` (`vDotElimExp`), which removes them from steep faces. Ours wrap over
    cliffs on purpose; keep that, but compare.
- **Life.** Recoil sets life, alpha and glow from the explosion's damage and radius:
  - alpha `clamp(2 * damage / 255, 0.8, 1)`;
  - life `clamp(3 * damage, 15, 1800)` frames at 30 fps, times the decal level setting;
  - glow `clamp(2 * damage / 255, 0, 1)`.
  Scale ours from the explosion's scale the same way.
- **Cost.** A second atlas sample and a TBN per projection, on top of the 37 µs `battle` already
  pays. Check `bench-diff` on the terrain pass; the glow could be its own specialization or skipped
  once every glow has ended.
- **Test:**
  - a glowing decal brightens its centre in shadow;
  - the glow is gone after its life, and the frame then equals the same decal without glow;
  - a normal-mapped decal lit from opposite sun directions differs where its normal map tilts;
  - the existing decal test is unchanged.

Recoil's scars (results):
- **Licence.** Recoil is GPL; the scars' rights holder (this repository's owner) allowed their
  use, so they are in `assets/decals/scars` (ten scars: the old `scar2` plus nine more 512²
  scars, `scar2`–`scar10`; the client discovers every `NAME.tga` with a matching
  `NAME_normal.tga`) and the client uses them by default (`--scars DIR` for another set,
  `--scars generated` for the engine's). The generated scorch and crater gained their own
  normal maps (from generated heights) and, for the scorch, a glow mask.
- **Textures.**
  - `read_tga` reads 32-bit TGA, raw or run-length, either row order.
  - `DecalAtlas::add_layer` stores a source of the atlas's size byte for byte; any other square
    is resampled (box when shrinking by a whole factor, bilinear otherwise; colour in linear
    light). Only the atlas needs power-of-two mips. The client's atlas with scars is 512²
    (`default_atlas(512)` generates its layers at that size, the relief scaled to keep its
    slopes).
  - The normal map is already the atlas's surface layout (rg the normal's x/y, b roughness, a
    glow): `add_layer` only flips green (`255 - g`) for a green-up map. Nothing rebuilds z on
    the CPU; the shader does. The surface is a second array (binding 19, UNORM; its mips
    renormalize the normal and average roughness and the mask).
  - Handedness differs per file: the old scar1's normal map had green down the image, the old
    scar2's up (its own curl, measured where alpha > 0.3: +0.80 and -0.41, -0.81 on a 4x
    downsample). `normal_green_up` measures it and `add_layer` flips to the atlas's convention
    (green toward the decal's +z, down the rows). A unit test loads every bundled scar pair:
    after loading, its curl is positive (0.02 to 0.73 across the bundled set; a baked map can
    carry little tangential signal), as the generated layers'.
- **Mips on the GPU** (2026-11-21): every texture the engine mips is uploaded as mip 0 and its
  chain made by one compute shader (`upload::image_mipped`, `shaders/mipgen.comp`). Each level
  is a 2x2 box of the one above, clamped at an odd edge, with every layer in one dispatch. sRGB
  images are read through their sRGB view (linear) and written through a UNORM storage view
  (`MUTABLE_FORMAT | EXTENDED_USAGE`), encoding sRGB in the shader. Filters: `Average`,
  `Coverage` (rgb weighted by alpha) and `Normal` (rg a tangent normal's x/y, z rebuilt, summed
  and renormalized; b, a averaged).
  - Users: the decal atlas (Coverage) and surface (Normal), the tileset albedo (Average) and
    normals (Normal, RG8), the anti-tiling noise and the particle puff (R8), and model textures
    that arrive with level 0 alone. `ae_gltf` now stores PNG/JPEG as level 0 only. BC and Basis
    textures keep the levels they bring (a compute shader cannot write them). The terrain
    pyramid and error levels are data reductions, not filtered mips, and stay as they are.
  - The device now requires `shaderStorageImageExtendedFormats` and
    `shaderStorageImageWriteWithoutFormat`: one formatless shader writes R8, RG8 and RGBA8.
  - The tileset's per-layer mean, which the anti-tiling blend needs, is now averaged over mip 0
    on the CPU instead of read from the last mip.
  - GPU `mips_are_made_on_the_gpu` (with `upload::read_image`): coverage leaves no colour from
    a transparent texel, black and white average to sRGB 188, a normal mip is the renormalized
    sum (185, not the plain 179), and RG8 and R8 work. All 40 GPU tests pass.
- **Do the scars' normal maps show?** GPU `scar_normal_maps_relight_the_ground`: `scar1`, loaded
  as the client loads it, with Modulate blend on flat sunlit ground, against the same atlas with
  that layer's normals made flat (roughness and glow kept). Over the scar's footprint:
  - close (50 m): mean |Δluma| 4.6, and 36 % of pixels change by more than 2;
  - at gameplay distance (250 m): mean 2.4, and 32 %.

  They relight the ground, but subtly. The scars are mostly gentle (median tilt 0.15, p90 0.63),
  dark under Modulate, and rough (blue about 0.93). The same numbers before and after the GPU
  mips.
- **Record**: 16 words, adding glow, glow end, 1 / glow life and the blend.
- **Shading** (`terrain_decals.glsl`; `applyDecals` returns albedo, normal and glow):
  - **Relit by its normal map.** Each projection's tangents are the world gradients of its u
    and v (top: the decal's x and z; the side projections: up, signed as the unfolding, and the
    other axis), Gram–Schmidt against the heightmap normal. The decal's normal replaces the
    shading normal by its alpha (a flat map leaves the heightmap's normal: the scar covers the
    tileset's detail). It changes N·L and the sky term only; the units' shadow lookup keeps the
    ground's normal.
  - **Blend::Modulate** (the client's Recoil scars): `ground * 2^2.2 * decal`, Recoil's
    `2 * decal * ground` in gamma space with our linear albedo. The generated scorch keeps
    `Replace`, as it was drawn for it.
  - **Glow**, as Recoil's: the glow mask as a black body at `mask * mix(1, 3700, heat)`, heat
    the glow less the time since placement over the glow life (a weaker glow ends sooner), the
    overglow flicker (`0.2 |sin(0.6 t)|` near full), `smoothstep(0.9, 0.1, d)^7` of the 3D
    distance from the centre over the half-diagonal (round). The fade-in does not dim it. Its gamma-space
    colour is made linear (`^2.2`) and added after lighting: untouched by the sun and shadows,
    put out by water within a metre, and dimmed by the fog of war as the scar under it is.
  - An explosion is seen whole or not at all. Deaths explode only where the observer saw them;
    X and `--explode` now follow the same rule (`Effects::explode_if_seen`, the sight the sim
    sets each tick). Before, X went off in the fog: a scar showed with no glow, since the glow
    was hidden out of sight.
    Brightness goes as temperature^4, so a glow is near its end by half its life, as Recoil's.
  - **Not done:** the ellipsoid and `dot(up, N)^exp` shaping (ours wrap over cliffs on purpose),
    the glow colour map, the water absorption of the scar's colour.
- **Life from size.** `Decal::scar` applies Recoil's damage rules: opacity
  `clamp(2 damage / 255, 0.8, 1)`, glow `clamp(2 damage / 255, 0, 1)` falling by 1 over 2 s
  (`SCAR_GLOW_LIFE`). The client takes damage as 250 x the explosion's scale (deaths: scale
  `radius / 6`, 0.6 to 3), so every death glows fully. Life stays `SCORCH_LIFE` (60 s; Recoil's `clamp(3 damage, 15, 1800)`
  frames would be 15 s at scale 1 without its decal level setting).
- **Tests:**
  - GPU `decals_relit_modulated_and_glowing`, on a plateau whose west cliff faces the sun:
    - normals leaning +u on the cliff's top edge: the plateau 167 → 131 (leaning away from
      the sun) and the face 167 → 183 (unfolded downward, leaning up), reversed when turned by
      pi (183 and 128);
    - normals leaning +v turned by pi / 2 light as +u turned by pi (184 and 129), which checks
      the green channel's sign and the other side projection;
    - modulate: sRGB 0.5 leaves 1.000 of the ground, 0.25 leaves 0.513;
    - a glow in the hill's shadow: the centre 80 → 172 luma, orange (255, 144, 92), held to
      the centre; 170 while the decal fades in (0.4 s, read at 0.12 s); past its life,
      byte-identical to the same decal without glow.
  - `decals_wrap_over_cliffs` unchanged; unit tests for the TGA reader, resampling, the
    conventions, and the client's scar.
- **Cost** (`--bench all --label m7-scars` against `m7-decals`; 109 of 110 metrics unchanged):
  `battle`'s terrain pass 216 → 225 µs (+4 %, within its tolerance) for the second sample per
  projection, the tangent frames and the glow. The one flagged metric is `battle`'s particle
  shadows (39 → 43 µs), a pass decals do not touch. With Recoil's scars as the default
  (512² atlas, modulate) and the glow fix, `m7-scars-2`: 110 of 110 unchanged, terrain 224 µs.
- **Fixed after the first commit:** no glow showed in a game. Deaths (scale 0.6 to 0.8) made
  damage 60 to 80, so glow 0.5 to 0.65, and brightness goes as about heat^5 (3 % at the start);
  the 0.4 s fade-in dimmed it further, and by the capture (1 s) it was gone. Now deaths glow
  fully, the fade-in leaves the glow alone, the decay and spread are Recoil's. The glow is at
  its brightest for the first half second; by 1 s it is about 3 %, as in Recoil.

M7 stretch goal (after decals): volumetric smoke and dust.
- Why: the order-independence measurement showed that particles blended as flat layers pop when
  they cross (116 per channel), whatever the order; treating smoke as a volume, overlapping puffs
  mixed by how much of their thickness overlaps, changes smoothly as they pass through each other.
- First a volume version of the reference: each fragment's front and back (the back capped at the
  scene depth), split into intervals, extinction `-ln(1 - a) / length` and emission per metre
  summed. Check that it pops well below 116, so the measure is proven.
- Then measure against it: a 3D volume grid in camera space (froxels, Hillaire, Frostbite 2015)
  and a per-pixel blend that accounts for thickness.
- Sparks, fire and flashes stay sorted, since they are small and additive.

Volumetric smoke, first two steps (`cargo test -p ae_render --test gpu particles_order -- --ignored
--nocapture`; the order test now runs five blends):
- **Volume reference** (`ParticleBlend::Volume`, tests only, the reference's lists):
  - Each fragment stores its sphere's chord along the pixel's view depth.
  - The resolve caps every back at the scene and cuts the depth at every span's ends. In each
    interval, the spans covering it sum extinction `-ln(1 - a) / L` and light per metre
    (scattered `rgb / a` times extinction; emissive `rgb / L`), integrated exactly front to back.
  - A lone puff comes out as blended.
- **It is right, and it does not pop:**
  - Crossing: the largest change between frames is 7 (0 pixel-frames over 40), against 116 for
    the sort and for the exact layers.
  - Apart (25 m in front and behind, chords disjoint): within 0.02–0.08 of the layers (max 3), and
    the two frames differ by 116. It keeps the depth order; it is not smooth by ignoring it.
  - Cluster: 1.75 (max 31) from the layers. Emitted in reverse: identical.
  - Scored against it now: the sort 1.66 (max 22) in the cluster, 1.62 (max 70) in the crossing.
- **Slices** (`ParticleBlend::Slices`, `--particle-blend slices`), the per-pixel blend that accounts
  for thickness. Three passes at half resolution, no sort:
  - Range: each pixel's nearest front (as a reciprocal) and farthest back, max-blended (RG16F; RG32F at first).
  - Slices: four equal slices of that range (RGBA16F each, additive). Each fragment adds its
    light and optical depth to each slice, by the length of its chord inside.
  - Resolve: each slice is a homogeneous medium, front to back.
  - A lone puff is exact whatever the slicing. Overlapping puffs mix within a slice, continuously.
  - Bug found on the way: corners outside a puff's circle had chord length 0, so the light per
    metre was infinite and multiplying by an overlap of 0 gave NaN in the sums. The chord helper now
    returns a length of 1 when empty.
- **Slices, measured against the volume:**
  - Cluster: 0.57 (max 10); 0.59 with the RG16F range.
  - Crossing: pops 6 (0 over 40), 0.19 (max 7); 0.27 with the RG16F range.
  - Emitted in reverse: identical.
- **Cost** (`bench battle`, about 40k particles; `m7-vol-sorted`, `m7-vol-slices`): slices 16 µs sim
  (no sort) + 494 µs draw, against sorted 58 + 82 µs.
  - That is about the moments' cost (552 µs). Against the volume the slices are much closer than
    the moments: in the crossing 0.19 (max 7) against 1.79 (max 70), popping 6 against 115; in the
    cluster 0.57 (max 10) against 1.16 (max 29).
  - Split (each pass skipped in turn): 32 µs fixed (depth, resolve, composite), the range pass
    about 210 µs, the slices pass about 250 µs. The slices pass falls to 75 µs when its outputs are
    all zero (framebuffer compression), so it is bound by the bandwidth of four RGBA16F blends.
  - The range as RG16F halves the range pass (about 96 µs): 378 µs in all (`m7-vol-slices-rg16`).
    The global range the simulation already has would save the rest of it, but one far puff
    would stretch every pixel's slices.
- **Froxels** (`ParticleBlend::Froxels`, `--particle-blend froxels`), the grid in camera space
  (Hillaire). Columns of 4x4 half-resolution pixels, 32 slices each, spaced exponentially over the
  visible particles' depth range (the simulation's, as for the moments). Four passes, no sort:
  - Puffs: each visible particle shaded as the draw shades it (the vertex shader's shading moved
    to `particles_puff.glsl`, shared), with the columns its quad covers; this also writes the
    scatter's indirect dispatch.
  - Scatter: one workgroup per puff, over its columns. Along each column's centre ray, the puff's
    chord as a medium (the volume's model, `medium()`) adds its light and optical depth to the
    slices it crosses. Fixed-point `uint` atomics (core; no float atomics).
  - Integrate: one invocation per column, front to back, each slice a homogeneous medium.
  - Resolve: the light and transmittance at the scene's depth, trilinear (four columns, two
    boundaries).
- **Froxels, measured against the volume:**
  - Cluster: 6.4 (max 55); over 8x8 blocks (the columns' size here), 4.5. Averaging does not take
    all of it out. Part is aliasing: a puff smaller than a column is caught or missed by its
    centre ray (the fire loses its small bright puffs). Part is the resolve's bilinear spreading
    each column over its neighbours' blocks: the crossing, whose puffs are 16 and 26 m, still
    shows 3.8 over blocks.
  - Crossing: pops 10 (0 over 40), 5.6 (max 42); over blocks 3.8.
  - Apart (25 m in front and behind): the swap is 66, against the volume's 116. The order holds;
    the blur lowers the peak.
  - Stretched (a dim near puff at a third of the distance, a range of about 3 to 1, slices about
    5.5 m thick at the pair): pops 10, 3.0 from the volume; apart still 66.
  - A puff half in the ground (the volume caps it exactly): 1.4 (max 16).
  - Emitted in reverse: identical.
  - Columns of 2 pixels: cluster 2.3, crossing 2.0, but 447 µs; worse than the slices on both.
- **Froxels, cost** (`bench battle`; `m7-vol-froxels`): 17 µs sim + 181 µs draw, against sorted
  58 + 82 µs and slices 16 + 378 µs.
  - First version, gathered (each column of a bin of 8x8 columns testing every puff binned
    there): about 610 µs. The tests alone took about 290 µs; scattering by puff does only the
    columns each puff covers.
  - Columns stored per column first: the integration's accesses did not coalesce (about 210 µs
    for empty bins); slice-major, about 130.
  - 64 slices: 266 µs. Clearing and integrating the grid is most of the fixed cost, so halving the
    slices took 85 µs off. The test's numbers did not change, stretched or not: at its scale the
    screen-plane blur dominates. Not measured: the battle's range (camera 900 m away) makes
    slices of tens of metres, where nearby puffs blend as one medium, out of order.
  - Of the 181 µs: about 32 the passes every half-resolution blend has (depth, composite), 65 the
    scatter, the rest clearing (7.5 MB), the puffs and the integration.
- **Choice:** the sort stays the default.
  - Both volume methods stop the popping (slices 6, froxels 10, against 116).
  - The slices are near the volume (0.6) but cost 2.8 times the sort (394 µs against 140, sim and
    draw).
  - The froxels cost 1.4 times the sort (198 µs) but are blurred and alias small puffs (6.4 from
    the volume in the cluster, the sort 1.7).
  - Neither is worth it while the popping is a short flicker as puffs cross. If it shows in play,
    froxels for smoke and dust (sparks, fire and flashes sorted, over them) is the cheaper
    route, with work first:
    - a few rays per column, or a filter over the columns, against the aliasing;
    - the slices' thickness at the battle's range checked (a test at that scale);
    - dust on the ground checked in play: the scene's depth falls inside a slice, so a buried
      puff fades by where the ground lies in it, and that shifts as the frame's range moves.

Packets, blend mode 8, and the half/full decision (`ParticleBlend::Packets`,
`--particle-blend packets`; the M12 TODO):
- **The port.** The radix sort's front-to-back ranks are marked in a bitset per tile (a
  word-of-words summary, under an 8 MiB budget; the 65,536-slot pool fits 16-pixel tiles at
  half resolution and 32-pixel tiles at 1280x720 full). A compute pass packs each rank's puff
  (centre in the camera's axes, depth,
  1/radius, premultiplied colour, angle, mip) and marks the tiles its screen bounds cover.
  A second compute pass walks a tile in packets of 8x4 pixels (one 32-lane subgroup per
  packet): for each 32-rank word one lane classifies one candidate against the packet's ray
  rectangle, the ballot collects the hits, and all lanes shade the same hits for their own
  pixels, front to back. Every lane classifies its candidate while any pixel of the packet
  is live (one whose own pixel is saturated or off the screen still does). The blend's
  pipeline pins 32-lane, full subgroups (`subgroupSizeControl`, `computeFullSubgroups`;
  `GpuDevice::compute_subgroups_32`), with a 256-wide workgroup per tile. It writes the colour and the per-particle glow into its own RGBA16F
  targets, then composites them over the scene: the depth-aware upsample at half resolution,
  one to one at full. A puff is the draw's sphere and soft edge (`particles_frag.glsl`), with
  an explicit mip from its screen diameter (the projection's vertical scale taken from a
  projected point, not `viewProj[1][1]`, which the camera's pitch scales). The scene depth
  (full) or the half depth is made visible to the compute stage before the blend samples it.
- **Correctness.** The order test runs seven blends with Packets: in the cluster it is 0.75
  (max 26) from the exact per-pixel reference (sorted 0.74, max 26), and 0.86 from the volume
  (sorted 0.86); crossing it pops 63 with 81 pixel-frames over 40 (sorted 64 and 80,
  reference 64 and 58) and is 0.94 from the volume (sorted 0.93); emitted in reverse the
  frame changes by 1 (float rounding, as the sort). The test asserts the packets' cluster
  error within 0.05 of the sorted draw's: before the lane fix above, a saturated lane stopped
  classifying its rank for the whole packet, which measured 0.84 and popped 57. The new GPU
  test `particles_packets_both_resolutions` checks blue in front, a buried puff, an
  explosion and the per-particle glow at both resolutions (a fire in front of smoke blooms
  9.2 luma at full and 7.3 at half).
- **Cost** (`bench battle`, about 40k live particles, 1280x720; three runs each, medians;
  the machine was shared with other workflows, so the packet draw's per-run spread was
  ±100-170% while its median stayed within 1%):
  - sorted: 69 µs sim + 152 µs draw at half, 69 + 383 at full.
  - packets: 69 µs sim (the same sort) + 274 µs draw at half, 69 + 681 at full. Rerun
    after the lane, mip and subgroup fixes (three runs each, same scene, the machine again
    shared): 70 + 274 at half and 70 + 689 at full, sorted 70 + 152 and 70 + 382; the
    ratios below are unchanged.
  - Packets cost 1.8 times the sorted draw at either resolution; full resolution costs
    2.5 times half for both. The 8 MiB tile masks and the bin/pack pass are fixed; the
    blend's per-pixel depth and texture work and the four times the tiles dominate full.
  - The earlier unloaded numbers for the sorted path were 58 + 85 µs (half) and 58 + 171
    (full); the 1.8 ratio is the comparable part.
- **Decision:** half resolution. The packet blend is 2.5 times cheaper there and carries the
  per-particle glow and the depth-aware composite. The sorted draw stays the default,
  though: the packets are 1.8 times its cost at either resolution for the same order, so
  `Packets` remains a comparison blend (like the moments, slices and froxels).
  `bench-diff` of the sorted half path before and after (`m12-packets-sorted` against
  `m12-pkt-sorted-half-r1`): 0 regressions; against `m12-pkt-packets-half`:
  `pass/particles` +81% (expected, the comparison mode).
- Validation clean (synchronization validation on): 41 GPU tests; client captures of the crossing map with explosions at
  both resolutions report 0 errors and 0 warnings.

### M0 risk spikes

| Risk | Experiment | Fallback |
|---|---|---|
| N wasmtime instances over one `SharedMemory` (stack/TLS, data segments) | 8 stores, one memory, disjoint column ranges; correctness + ns/call | one instance with wasi-threads inside the guest |
| Per-chunk call overhead | same spike at 64/256/1024/4096 rows | several chunks per call via descriptor array |
| Flecs determinism + cost of large deferred merges; id-array patch | 50k entities, 5k sorted structural commands/tick, digest + merge time | enable-bits by default; earlier custom ECS behind the seam |
| f32 host-service determinism across platforms | pathsim golden-hash tests on Windows + aarch64 | fixed-point bodies |
| Several sim modules over one memory | two independently built modules with separate heap/data regions sharing columns | copy-in/copy-out column windows for secondary modules |
| Render ABI expressiveness | re-express `ecs_sim/src/render/demo.rs` graph through the descriptor | larger engine pass libraries, smaller module surface |

## Verification (end to end)

- Every milestone: new subsystems ship with micro + subsystem benches; `ae-tool bench-diff` against the previous milestone shows no unexplained regression.
- Every milestone: `cargo test --workspace`; determinism suite (digest equality across worker counts, debug/release, save→load→continue).
- From M1: `ae-headless` digest runs in CI on Linux x86-64, Windows x86-64, Linux aarch64.
- From M3: GPU tests with Vulkan validation + sync validation, zero messages; golden frame graph `.dot` output.
- From M5: `ae-headless --replay x.aerep --verify` on all CI platforms.

## Planned milestones M11–M16 (M11, M12, M13 done; M14 in design)

Numbers are labels, not an order (M8–M10 keep theirs). Order by priority: M11 first (M12 needs
it), M12 medium, M13 lower and after M12. Whether they come before or after M8–M10 is open.
M14 (unit scripts) needs M11; its order against M12/M13 is open. M15 (bindless resources) is
independent of the others and can come after M3. M16 (the RedAlert-graphics port) is a survey
with water as its fixed scope; its parts can land one at a time after M7.

The plan so far had only one line on models (Renderer: "glTF converted offline; piece animation =
sim WASM writing `PieceTransform` columns"). No loader, format or skinning was planned. M11 fills
that in.

### M11 — glTF models and skinning

- Scope: an offline converter (glTF 2.0 `.gltf`/`.glb` to an engine model in the archive) and the
  runtime that draws it.
- One bone-based format: a basic model (meshes attached to nodes, the "pieces") is converted to
  bones and joints with one weight: each mesh's vertices get the one joint of the node it hangs
  on, weight 1. So piece animation (`PieceTransform` columns from the sim) drives bones the same
  way for pieces and skins. On the GPU two layouts (rigid, skinned; decided below, 2026-11-16).
- Supported:
  - meshes attached to pieces (node hierarchy); skinned models with one or several skins, their
    joints and inverse bind matrices;
  - the glTF 2.0 PBR material: base colour, metallic-roughness, normal, occlusion and emissive
    textures and factors; alpha modes (opaque, mask, blend); double-sided; texture samplers;
  - the standard attributes: POSITION, NORMAL, TANGENT (generated when absent, as glTF
    prescribes), TEXCOORD_0 and TEXCOORD_1 (UV1/UV2), COLOR_0 (vertex colours), JOINTS_n and
    WEIGHTS_n; `KHR_texture_transform` and `KHR_materials_emissive_strength` to consider.
- Skipped: morph targets (vertex morphs); the converter warns, and does not fail, when it meets
  them.
- Animation clips: out of scope here (the sim writes `PieceTransform`); decide whether glTF
  animations are imported at all (see questions).
- Ask the user about any attribute, extension or case the plan does not decide (for example
  TEXCOORD_2 and above, COLOR_1, more than four influences per vertex, sparse accessors, cameras,
  non-uniform node scale, extensions such as `KHR_materials_*`, `KHR_draco_mesh_compression`,
  `EXT_meshopt_compression`, KTX2/Basis textures). Do not choose silently.
- Verification: loader tests on the Khronos sample models (a basic piece model, a skinned one),
  goldens for the drawn result, validation layers clean; the model's size and load time measured.
- Decided with the user (2026-11-16), M11 next, before M8–M10:
  - more than four influences per vertex: keep the four strongest, renormalise, warn;
    TEXCOORD_2 and above, COLOR_1: dropped with a warning; non-uniform node scale: supported
    (bones are full matrices);
  - extensions accepted: `KHR_texture_transform`, `KHR_materials_emissive_strength`,
    `KHR_texture_basisu` (KTX2/Basis, transcoded by the converter); not accepted (the converter
    fails, naming it): `KHR_draco_mesh_compression`, `EXT_meshopt_compression`; sparse accessors
    are core glTF and read;
  - animation clips: imported as data (sampled tracks in the model), no runtime player yet; a sim
    system can play them into `PieceTransform` later.
  - textures stay in the format the glTF brings, in the KTX2 suite: each is stored in the model as
    a KTX2 in a GPU format. PNG/JPEG become RGBA8 with mips (zstd supercompression, lossless);
    KTX2 already in a GPU format is kept; Basis-compressed KTX2 is transcoded by the converter to
    BC7, so the runtime needs no transcoder.
  - Superseded (2026-11-16) by "glTF loaded at run time" under Later improvements: no `.aemodel`,
    KTX2 only as an optional cache; and clips evaluated on the CPU (M14).
- Not in the sim yet: no `PieceTransform` column exists. In M11 the renderer takes a pose per
  unit (the rest pose, or a clip at a time: the stream's `clip` and `clip_time`); the sim
  columns come later (their per-bone interpolation goes in the same pose pass).
- Decided with the user (2026-11-16, after reading RecoilEngine's GL4 model path: one global
  vertex and index buffer, per-unit transform ranges in one buffer, multi-draw per bin; its
  per-vertex quaternion interpolation and transform chains are the part not to copy):
  - two vertex layouts, chosen per primitive and read by vertex pulling: rigid (one joint per
    vertex, 68 bytes; pieces sharing a material merged into one primitive, so a model draws once
    per material) and skinned (four joints and weights, 80 bytes). Replaces "one uniform format";
  - per-unit poses: each unit's palette range in one big buffer, its offset per instance; a
    compute pass resolves bone transforms to matrices once per bone per frame (interpolating
    sim ticks there, not per vertex); vertices only blend matrices (3x4 rows, 48 bytes);
  - all models' vertices and indices in one GPU buffer.
- Progress (details: `docs/models.md`):
  - Step 1, format and converter — done (2026-11-16). `ae_model` is the format (JSON header +
    aligned blob, versioned), KTX2 textures, poses and clips. `ae_gltf` is the converter (the C++
    Basis transcoder stays out of the runtime). `ae-tool model convert|info`. Tests on glTFs built
    in the tests and on three vendored Khronos samples (SimpleSkin CC0, BoxAnimated CC-BY, Fox
    CC0/CC-BY; `assets/models/khronos`). Sizes and load times measured (Fox: 295 KB, 0.08 ms to
    load).
  - Step 2, drawing at the rest pose — done (2026-11-16). Format version 2 (rigid and skinned
    layouts per primitive, primitives merged by material and layout). `ae_render::models`: one
    geometry buffer, a material table, a partially bound texture array, per-(class, primitive)
    commands with draw records by `gl_DrawID`; the cull fills them, two multi-draws (opaque,
    masked) draw them, the units' shadow caster draws them too. Opt-in per class
    (`UnitsSetup::models`): with none, the darts' renderer unchanged (24 GPU tests pass as
    before). New GPU test `units_draw_models`, validation clean.
  - Step 3, poses — done (2026-11-16). Clips baked at load (`ae_model::pose::bake`, 60 fps or a
    little more; the GPU blends two frames, no key search). Stream attributes `clip`,
    `clip_time` (10 attributes now). Library pass `units.pose` and resource `units.palette` in
    the native graph, both render modules and both graph files (`check_graph` rejects a graph
    that draws units without posing them); the palette is one device-local buffer of 3x4
    matrices: rest palettes, per-unit ranges for units playing a clip, scratch. Cull spheres
    of animated models cover every baked frame (+5%). New GPU test `units_pose_models` (palettes
    read back against the CPU, drawn poses, the lag, the padded cull); 26 GPU tests pass,
    validation clean.
  - Step 4, client and benches — done (2026-11-16). `ae-client --model [CLASS=]FILE`
    (glTF converted by `ae-tool` on first use) and `--clip MOVING[,STILL]`; bench scenes
    `army-fox`, `near-fox`. Validation clean. Measured: pose pass 0.31 ms for 10 000 Foxes; model
    shadows 5 ms whatever the view (every unit drawn into every cascade). The cull now also
    culls model units per shadow cascade (sides only) into per-cascade lists and commands
    (GPU test `units_model_shadows_culled`, 27 GPU tests pass): 4.4 ms in `army-fox`, since
    nearly every Fox is in all three nested cascades. Baselines `gpu-army-fox`, `gpu-near-fox`.
  - After M11: model shadows are still the largest cost with many models (vertex-bound). Not
    worked on further (decided 2026-11-16): the cascades are to be replaced by virtual shadow
    maps (Later improvements). Blending and choosing clips belongs to unit scripts (M14).
  - Fixed after step 4 (2026-11-16): clips jumped at every tick at any sim speed but 1x (drawn a
    wall-clock tick behind, while clip time gains a sim tick); the stream now carries the sim
    time of a tick. Play modes loop, ping-pong, once (`--clip 2,1:pingpong`); a unit's clip starts
    when it switches to it, and the standing clip waits 6 still ticks.

### M12 — Lights and the HDR scene target (priority: medium)

- Depends on M11 for glTF lights.
- Lights: glTF lights (`KHR_lights_punctual`: point, spot, directional) are part of a model and
  follow their piece's transform. The engine has its own kinds for effects (projectiles, lasers,
  bullets): point, cone, beam. Effect lights come from the sim as view-table rows, culled and
  uploaded like instance streams. Beams are capsules or lines; cones have an angular falloff.
- Many lights through clustered forward or Forward+ (a compute pass bins the lights into
  view-space tiles or clusters; lit passes read the per-cluster lists). Choose by measuring at the
  RTS's light counts; terrain, units and water share the same light lists.
- HDR rendering target: the scene is lit and blended in a floating-point target (RGBA16F or
  R11G11B10F, to be measured) and tone-mapped by a post pass. Particles, water and fog keep their
  blends and are checked against it. SDR output stays as it is.
- Open: shadows for lights (none at first; the sun's cascades stay the only shadows), and how the
  sun and sky light share units with the cluster lights.

Decided with the user (2026-11-19): M12 now (after M14 step 5; the rest of M14 is in TODO).
Scope: lights, lights with speculars, a bloom buffer that holds only light sources (it can be
smaller than the screen), and the HDR scene target. References: `~/particles` (its clustered
lights, `src/lighting.rs` and `shaders/particle_light*`; and its blend mode 8, the fastest it
has) and `~/RedAlert-graphics` (speculars, the glow target and `bloom.comp`, the ACES tonemap
in `post.frag`).

My defaults, not yet confirmed:
- The first light sources are the client's: the particles' emissive effects (flash, fire,
  sparks) give lights from their bursts, and glTF `KHR_lights_punctual` (today dropped by
  `ae_gltf` with a warning). Effect lights from the sim come with effects at joints (M14 TODO).
- Light kinds as in `~/particles`: point, cone, beam (a capsule along a segment), its falloffs
  (`(1 − d²/r²)² / (1 + d²)`, a cone's smoothstep between two cosines).
- Not its grid: a 24×28×24 world grid with a 32-bit mask per cell caps a scene at 32 lights.
  Lists per cluster (an index list per view cluster or tile), measured clustered against
  Forward+ at hundreds and thousands of lights, as above.
- Particles are lit once per particle in compute (as `~/particles` does), not per pixel.
- The glow target is a second colour attachment of the scene's passes at full size (one render
  area per pass): every pipeline that draws into the scene writes it (light sources their light,
  everything else black, or covering it by its alpha). The bloom reads it at a quarter of the
  size: the brightest of four 2×2 means, then an 11-tap Gaussian along x and along y
  (RA3-graphics' three passes), added before the tonemap.

Steps (commit and stop for compaction after each):
1. HDR scene target and a tonemap pass: the scene drawn into a float target (RGBA16F against
   B10G11R11F, measured), the tonemap and grade writing the output. In the native graph and the
   default and post modules (the scene format apart from the output's, through the render ABI).
   First with a curve that leaves today's frames as they are (the GPU goldens), then exposure
   and the curve.
2. Lights: the light list, the clusters, diffuse light on the terrain, units and water, the
   particles lit per particle; the first sources (particle bursts, glTF lights).
3. Speculars: the sun's and the lights' on units, terrain and water.
4. The glow target and the bloom.
5. Automatic exposure (eye adaptation): meter the scene in a histogram, ease the EV, and hand
   the tonemap and the bloom the adapted exposure.
6. The particles' blend mode 8 from `~/particles` (tile bins walked in packets of pixels),
   measured against the sorted default. With Mode 8, decide whether the particles stay at half
   resolution or go to full resolution (see "Packets, blend mode 8, and the half/full
   decision").

- Step 1, the HDR scene target and the tonemap — done (2026-11-19).
  - The terrain, units, water and particles draw into `scene`
    (`ae_render_abi::module::SCENE_FORMAT`, RGBA16F; `scene_spec`, `SCENE_NAME`; format name
    `scene` in graph files), and the new library pass `post.tonemap` (graphics: the scene
    `sampled:fragment`, one colour attachment of the output's format;
    `ae_render::tonemap`, `shaders/tonemap.frag`) writes the target. `LibraryEnv` has the
    scene's format apart from the output's; the library checks the draws against the one and
    the tonemap against the other. The native graph, the default and post modules and both
    JSON graphs do it; the post module tonemaps into `ldr` (the target's format) and grades
    that.
  - The curve: clamped at 1 by default (the frames as before: the 27 GPU tests pass
    unchanged) or ACES (`ToneParams`, `TerrainApp::with_tone`; `ae-client --tonemap
    clamp|aces`). ACES at an exposure that matched the clamp's mean luma in the near-army
    view (129 against 129 of 255) and is more saturated (68 against 56). The default
    stayed the clamp until lights and the bloom gave the scene light above 1; the
    photometric step below moves exposure before the scene's storage and makes it EV100
    (`--ev100`), and the ACES switch is its own commit.
  - Cost (bench scenes, 1280×720, against 4c389da): the draws into RGBA16F within ±2% of the
    8-bit target; `pass/tonemap` 15–20 µs. B10G11R11F measured too: the tonemap 11–15 µs and
    the draws 1–2% faster, about 8 µs of `gpu_total` in all. RGBA16F kept: 10-bit mantissas
    against B10G11R11F's 5–6 (banding and hue shifts in dark gradients), and alpha. The glow
    target can be B10G11R11F (as RA3-graphics').
- Step 2, first part: the light list, the clusters, diffuse light, the particles lit, the
  bursts' lights — done (2026-11-19).
  - `ae_render::lights`, `shaders/lights*.{glsl,comp}`. A `Light` (64 bytes) is a point, a
    cone or a beam, with `~/particles`' falloff, distance measured in units of the light's
    source size (`profile.z`). This way one formula suits a muzzle flash and a burning tank.
  - The library pass `lights.cluster` (compute) writes `lights.list` and `lights.clusters`:
    - It copies the host's list into device memory. The list lives in host-visible memory
      otherwise: every workgroup and pixel read it over the bus, and the cull took 0.3 ms at
      200 lights.
    - It bounds each light by the tiles and slices its sphere reaches.
    - It lists each screen tile's lights, by those bounds and then the tile's four frustum
      planes.
    - It lists each cluster's lights from its tile's list, by slice and then the cluster's
      view-space box.
    - Every list is in light order, compacted by a workgroup prefix sum, so sums are the same
      every run. Cuts are counted (`LightStats`): at most 1024 lights a tile, 256 a cluster,
      8192 a frame.
  - The grid: 40×24 tiles (about 32 px at 1280×720) × 16 slices. The slices are spaced
    exponentially over the depths the map's height range (plus 80 m) can occupy in view
    (`lights::depth_range`, from the frustum's corner rays), not over a fixed near and far.
    `AE_LIGHT_SLICES` overrides the slice count, to measure.
  - Optional in a graph, like the shadow maps. A graph that declares the two buffers has every
    lit pass read them (`LibraryPass::lit`): the terrain, units and water draws in the fragment
    stage, `particles.sim` in compute. The native graph, both modules and both JSON graphs
    declare them; the reload test's hand-built graph doesn't.
  - Lit passes use the lights' own descriptor set: set 1 for the terrain and water, set 2 for
    the units (an empty set 1 without models) and the particles.
  - Diffuse light is added to each existing lighting term: terrain, darts, models, and the
    water's foam (the water's glints from the lights come with step 3).
  - Media particles in view are lit once per particle in `particles_sim.comp`, the light packed
    into `Particle.info.w` and added in `puffOf`. Every blend mode gets it.
  - No light and no lights pass leaves the frames unchanged: the 27 GPU tests passed unchanged
    before any light source existed.
  - Sources: `EffectDef::light` (`EffectLight`) gives a burst one point light over it. FLASH
    has range 70 m, source size 16 m and lasts 0.3 s; FIRE has range 40 m, source size 10 m,
    lasts 1.2 s and flickers. Lengths are × the burst's scale, life × √scale, and the light
    falls off as (1 − t)². So `explosion()` gives two lights. They follow sim time, sit over
    the ground or water for ground bursts, and are culled against the frustum on the CPU.
  - `TerrainApp::with_lights` takes fixed lights. `lights::scatter` gives seeded test lights;
    `ae-client --lights N` uses it, as do the bench scenes `lights-200`, `lights-1000`,
    `lights-4000` (the battle camera) and `lights-low` (1000 lights, camera pitch 0.3).
  - The reference: `with_unculled_lights` makes every pixel and particle sum every light.
    `lights_clusters_match_the_unculled_reference` (GPU) checks clustered frames are
    byte-identical to it, from above and low, with 600 points plus 40 cones and 40 beams over
    a generated map with water, units and smoke. It also checks nothing is cut. New flags:
    `ae-client --capture-frames N` and `--lights N`.
  - Measured (1280×720, `gpu_total` in µs), against Forward+ without its depth bounds. This
    engine has no depth prepass, so 1 slice means 2D tiles; a true Forward+ would need a depth
    prepass first.

    | Scene | 1 slice | 24 fixed slices | 16 fitted slices (chosen) |
    |---|---|---|---|
    | battle | 893 | 926 | 902 |
    | lights-200 | 623 | 664 | 644 |
    | lights-4000 | 1458 | 1490 | 1393 |
    | lights-low | 987 | 919 | 918 |

    - From the RTS's usual height depth slices hardly matter; low over the ground they save
      about 70 µs. Fitting the slices to the map's depths lowers the lit passes' cost; at
      4000 lights the terrain goes from 715 µs (1 slice) to 657.
    - The cull: 22 µs (1 slice) to 48 µs (16) at 200 lights, 151 µs at 4000. Most of the
      cost is the clusters' workgroups, one per cluster even when the cluster is empty
      (possible later: one workgroup per tile walking its slices).
    - Earlier designs, measured: 16×9 tiles × 24 slices testing every light per cluster cost
      970 µs for the cull and 1559 µs for the terrain at 4000 lights. Bounds alone (no plane
      or box test) made the cull cheap but the lists loose (terrain 1051 µs at 4000).
  - Against 5b282ed (`bench-diff`, army, battle, close, army-fox, overview): scenes without
    lights are unchanged (the pass records nothing, `pass/lights` 0). `battle` (210 burst
    lights) costs +180 µs: lights 57, terrain +77, units +32, particle sim +12. `army-fox`
    `pass/water` 4 → 174 µs is the water's two modes (TODO), not this change.
  - Fixed after the commit: unculled, the pass skipped the host list's copy too, and the
    reference test passed only because the reference ran after the clustered frame and got
    its freed memory. The pass now always copies. The test renders the reference first and
    checks that it, too, is lit.
- Step 2, second part: glTF `KHR_lights_punctual` lights, appended on the GPU from the posed
  units — done (2026-11-20).
  - **The model carries them.** `ae_model` version 4: `ModelLight { name, joint, kind
    (Point | Spot), color, intensity, range, inner_angle, outer_angle, size }`. The joint is
    the light node's (an identity inverse bind), so the palette matrix gives its position,
    and its direction is that node's local -Z through the pose (glTF's convention), with the
    node's scale ignored. The converter reads `KHR_lights_punctual` (`ae_gltf`), passes the
    intensity through as candela, clamps spot angles to [0, π/2] and inner ≤ outer, and takes
    an absent `range` as no authored cutoff (`range = 0`; the renderer derives a cull radius
    from the intensity and exposure). Its source radius is `range / 8`, or
    `DEFAULT_SOURCE_RADIUS` (0.25 m) without a range. A **directional** light is ignored with
    a warning naming it and its node: the engine has one fixed sun, and a per-unit
    directional light would swing the scene with the unit's facing. Beams remain engine-only.
  - **Appended where the rows are.** `UnitRenderer::write_rows` (which already walks a new
    stream's rows) also writes each row's light offset, in row order, and the slot's total;
    the append is one dispatch, `units_lights.comp`, in the `lights.cluster` pass between the
    copy and the bounds (a host-side prefix first cost 500 µs a frame walked serially by one
    workgroup; the rows are already being walked on the host). The dispatch walks every unit
    row, reads the light's palette joint (the `units.pose` pass, an earlier graph pass, writes
    it: the append's barrier makes it visible) and the class's light range, and places the
    light as the unit's model is drawn (fitted, turned, at the unit's height and drawn size).
    The list's count and dropped count are written by the host: host lights first, then model
    lights in row order, capped at `MAX_LIGHTS` (`LIGHT_MAX`); the bounds then walk the whole
    table (`MAX_LIGHTS / 64` groups) because the append made it longer than the host's list.
    The light's `range` is not scaled by the unit's model fit (metres stay metres). A graph
    that clusters the lights before `units.pose` is rejected: the append reads the palettes
    it writes.
  - **Tests.** `ae_gltf` converts a point light, a turned spot and a directional one: the
    palette gives the node's world position and the spot's direction, the point's colour and
    range are the candela mapping, the directional warns, and the model round-trips.
    GPU `model_lights_light_the_ground` (validation on): two units (one point, one spot) with
    no host light; the point raises the red channel of the ground beside it, the spot raises
    blue in front of its node (world -z) and leaves the ground behind it byte-identical, the
    clustered frame equals the unculled reference, a second render is byte-identical, a model
    without lights leaves the frame as before, and the stats count exactly two appended
    lights. All 29 GPU tests pass.
  - **Cost.** New bench scene `army-model-lights` (army camera, 10 000 Foxes each with a
    point light of range 15 m): `pass/lights` 557 µs (copy, append of 8 192 lights,
    bounds/tiles/cull; 8 192 listed, 1 808 dropped by the cap), against 0 in `army-fox`;
    the units' draw +493 µs from summing the lights that reach the view. Same-session pair:
    `gpu_total` 8 612 → 9 630 µs (the terrain and water differences are the known run-to-run
    spread). The append is skipped with no model lights: `army-fox` and every earlier scene
    are unchanged. Validation clean (`AE_VK_VALIDATION=1`, 0 errors/warnings).
  - Not done: a light's direction is only tested through a horizontal, wide cone; a model's
    light does not scale its range with the model's fit (deliberate); a glTF directional
    light cannot become the sun (each model would carry one); assets with gaps in the model
    light table (only per class), and no light shadows.
- Step 2.5: photometric lighting — done (2026-11-20). Full write-up with units, formulas,
  calibration, the emitters/water/fog migration and verification: [lighting.md](lighting.md).
  - **Units and constants** (`ae_render::lighting`): world units are metres; point/spot
    intensity is candela, the sun is lux, emission is cd/m²; Lambertian radiance is
    `albedo/pi * E`. Sun 100 000 lx and sky 20 000 lx on an upward surface, with unit-luminance
    tints (mirrored in `shaders/photometry.glsl`). Exposure is EV100
    (`1 / (1.2 * 2^EV100)`, default 14, which keeps the old `1.6` terrain look); the scene
    passes apply it before storing into the RGBA16F scene, and `post.tonemap` only curves.
    `ToneParams { ev100, curve, sun_illuminance }`; the client takes `--ev100` and `--sun-lux`
    (replacing `--exposure`).
  - **One direct-light contract** (`lights.glsl`): `lightIlluminance` returns the linear RGB
    illuminance before the receiver cosine, with the true distance, the source-radius
    denominator clamp, the glTF quartic range window (`range = 0`: none), a squared cosine
    ramp for spots, and the beam's segment distance. Payload unchanged at 64 bytes: `profile`
    is (inner cos, outer cos, source radius, intensity cd); `range` is the authored cutoff
    only. The terrain, darts and models, water's foam and the particle simulation all consume
    it; receivers apply `albedo/pi` and the cosine once. `lights_bounds`/`lights_tiles`/
    `lights_cluster` bound a light at
    `sqrt(max channel(tint * intensity) / (0.002 / exposure))` when it has no authored range
    (about 5 m for 1 000 cd at EV14, 797 m for a 25 000 000 cd flash), capping at the authored
    range; the shading window never sees the cull radius.
  - **Emitters, water and fog re-authored** (same step, 2026-11-20): particles carry a tint
    and emission in cd/m² (FLASH ≈87 000, FIRE ≈20 000, SPARKS ≈74 000 at birth; media carry
    albedo and are lit by physical sun/sky over π and the clustered lights over π); burst
    lights store tint and candela (FLASH 32.5M, FIRE 3.17M); glTF emissive is read as cd/m²
    at 19 660.8 per unit; decal glow at 19 660.8; water's sky reflection is physical sky
    radiance, its glint scales with the sun, its foam is lit over π, its deep/shallow body
    radiances are 786/3 146 etc. cd/m²; fog and the scene clear colour are the sky's radiance
    (exposed when stored). No reference-exposure factor remains in the shaders. The media's
    clustered light is packed as illuminance × exposure (10-bit `x/(1+x)` per channel), so a
    bright light is not clipped at the old 1 022 lx ceiling; a spot's direction takes the
    node's scale-free columns (glTF's ignore-scale rule), correct for non-uniform scale.
  - **Verification**: `lighting` and `lights::bounds` unit tests; `ae_gltf` candela
    passthrough/omitted range/direction/warning; GPU
    `photometric_sun_exposure_and_inverse_square` (sun scales with illuminance; one EV doubles
    the *stored* scene, so exposure is before storage; 100 000 cd adds 4:1 at 2 m vs 4 m; a
    source radius larger than the range does not extinguish the light) and
    `particles_light_by_illuminance` (1e6 and 4e6 cd at 10 m add 4:1 to a smoke puff, above
    the old packing's ceiling). All 31 GPU tests pass
    with validation; the image tests needed no re-baselining — a generated-map capture with 8
    explosions differs from the old build by a mean 1.8/255 (0.4% of pixels above 8, the
    burst lights' retuning). `ae-client --bench battle` is unchanged within the run spread.
  - **ACES default** (its own commit, 2026-11-20): `ToneParams::default` is now
    `Curve::Aces`; `Clamp` stays for migration captures. The scene clear colour is the sky's
    radiance times exposure. Five GPU tests with absolute thresholds were re-baselined for
    the curve (the generated-map capture with eight explosions shifts by a mean 24/255; blur
    and pop measures that read display bytes now see roughly 70% of the clamp values).
  - Not done: authored sources have no glTF radius (the
    range-derived size is a convention); the ACES/sky/fog calibration is a look decision
    (capture numbers in lighting.md).
- Step 3, speculars — done (2026-11-20). Full write-up in [lighting.md](lighting.md)
  ("Speculars" and "Colour"), the per-layer setting in [tilesets.md](tilesets.md).
  - **One BRDF** (`shaders/lights.glsl`): GGX D, height-correlated Smith visibility,
    Schlick F; `kD = (1 - F) * (1 - metallic)`; the receiver cosine applied once. The sun and
    the clustered list use the same functions (`brdfLight`, `lightsSurface`,
    `lightsSpecular`); the sun carries the hills'/units' shadows and the particles' tint, the
    lights none.
  - **Materials**: glTF's metallic and roughness factors and the metallic-roughness texture
    (linear, its own UV set and `KHR_texture_transform`) in the model material table (288
    bytes, the std430 offsets pinned by a unit test); `F0 = mix(0.04, base, metallic)`. The
    terrain is a dielectric (`F0` 0.04) with a per-tileset-layer roughness in the UBO (grass
    and scrub 0.92-0.95, sand 0.75-0.78, rock 0.50-0.52, wet beds and mud 0.45-0.48; the
    palette path 0.85). Darts 0.04/0.7.
  - **Ambient**: the sky is a uniform hemisphere, so its specular has a closed form,
    `F(N.V) * skyRadiance * (0.5 + 0.5 R.y) * ao`. There is no IBL, split-sum LUT,
    multi-scatter compensation, area light or per-light shadow; a metal's diffuse is 0, so
    this term keeps it visible away from a highlight. That boundary is documented, not a
    placeholder that silently differs.
  - **Water**: the sun's glint is the shared GGX lobe (F0 0.06, roughness 0.22..0.43 from the
    unresolved wave slope, the old Phong exponents 900..60) and the clustered lights get the
    same lobe. The water's first code was the one caller that skipped the receiver cosine; it
    now applies it, and `WATER_SPECULAR_GAIN` went from 0.062 to 0.0926 (= 0.062 / sin 42°, the
    reference sun's cosine at the mirror) so the reference-angle brightness is unchanged and
    the glint follows the sun's elevation (a lake capture before/after differs by a mean
    0.14/255). A first version of the shared BRDF itself multiplied the specular by the
    cosine twice (the terrain test found it).
  - **Decals paint a material, not only a colour**: the channel allocation is albedo RGBA (RGB
    colour, A coverage) and surface RGBA (R/G the tangent normal x and y, B roughness, A glow);
    the normal's z is reconstructed in the shader and metallic stays a per-layer factor, but
    roughness comes strictly from the texture, with no layer factor. Each `DecalAtlas` layer
    carries a `LayerMaterial` (metallic; one `vec4` per layer, binding 21), blended into the
    ground by the decal's coverage. The default atlas generates per-texel roughness maps: the
    scorch 0.855 to 0.95 (char and cinders rougher than the burnt ring), the crater 0.81 to
    0.9 (the pit rougher than its rim). A scar takes its roughness from the normal map's blue;
    a layer added without a normal map writes 1. Layers default to metallic 0 until
    `DecalAtlas::set_layer_material` says otherwise. The terrain's `F0` becomes
    `mix(0.04, albedo, metallic)`, so a painted metal decal works. GPU
    `decals_carry_their_roughness_and_metallic`: three identical grey layers under a light at
    the view's mirror read 199 (roughness texture 0.15), 136 (0.95) and 249 (metallic 1.0);
    the albedo and normals are the same, so only the layers' roughness textures and metallic
    factor explain the difference. CPU tests cover the surface packing, the defaults and the
    range checks.
  - **Tests**: GPU `speculars_follow_material_and_lights` (a smooth metal's white sun
    highlight is wider than a dielectric's and a rough metal has none; with the sun off a
    clustered light at the mirror lights the metal and the same light at the opposite azimuth
    does not), `lights_glint_on_water` (181 to 254 luma) and `lights_specular_on_the_ground`
    (163 sky to 232 diffuse, 250 at the mirror). 35 GPU tests pass with validation;
    `cargo test --workspace` green. The decals' modulate threshold was re-baselined (a
    quarter-grey decal leaves 0.63 of the ground instead of 0.51: the specular's ambient does
    not scale with the albedo).
  - **Cost** (`--bench`, `m12-speculars-before` against `m12-speculars-2`): battle's total
    988 to 1 059 µs (+7.2%; terrain +14%, units +34%), cliffs +9.3%, lights-1000 +13.1%,
    army-fox +2.0%. The first version used `pow` for Schlick's fifth power and cost
    lights-200's terrain +125% (265 to 594 µs); five multiplies give +4.3% (289 µs). The
    terrain pays the sun and ambient BRDF on every pixel and one lobe per clustered light.
  - Not done: no environment map/IBL; no multi-scatter energy compensation; the water's gain
    is an art calibration, not a measurement; blended glTF materials are still a mask at 0.5.
- Step 4, the glow target and the bloom — done (2026-11-20). Write-up in
  [lighting.md](lighting.md) ("Glow and bloom").
  - **The glow** (`GLOW_NAME`, `B10G11R11`, the scene's second colour attachment): every
    `terrain.draw`, `units.draw`, `water.draw` and `particles.draw` writes it. The terrain writes
    a decal's black-body glow after vision and fog; a model material writes its emissive; the
    water writes only its crisp glints; an emissive particle writes its emission (masking by its cover) and smoke
    writes black with its coverage, so it hides the light behind it. Darts, the sky and every
    lit surface write 0. The buffer is exposed radiance, like the scene.
  - **The bloom** (`bloom.downsample`, `bloom.blur_x`, `bloom.blur_y`; `ae_render::bloom`,
    `shaders/bloom_*.frag`): quarter-size `RGBA16F` buffers; the brightest of four 2×2 means per
    4×4 block, then an 11-tap Gaussian along x and y (RA3-graphics' three passes, sigma 2).
    `post.tonemap` samples the scene and the result and adds it by `ToneParams::bloom` (0.3,
    RA3's `kBloomStrength`) before the curve. The glow is clamped at 4.0 exposed radiance
    (`BLOOM_CLAMP`). A graph without a glow binds the scene to the second sampler and switches
    the bloom off, so old module graphs still run.
  - **Water's excessive glow, as RA3 avoids it**: only the crisp glints are bloom light,
    `glint * (1 - rough)` (RA3's `glint * (1 - soft)` in `ocean.frag`). A rough or distant water
    pixel is many glints averaged into a sheen; blooming it lit the whole sun path as one white
    blob. Foam, the waterline fade and fog also remove the glints from the glow. The on-screen
    gain is unchanged; the gate only decides whether the light is a point source.
  - **Graphs**: the native graph, both modules and both `.graph.json` files declare `glow`,
    `bloom.0` and `bloom.1` and the three passes; the tonemap samples the scene and `bloom.0`.
    The goldens were regenerated for the added passes and resources. The sorted half-res
    particle draw writes a separate half-resolution `B10G11R11` glow target, per particle in
    sorted order; the composite upsamples it with the same depth-aware weights as the colour,
    and only the accumulated coverage masks the scene's glow. This is the fix for the first
    version of the composite, which classified the accumulated colour by its alpha and lost a
    fire's glow wherever smoke shared the half texel (a fire in front of smoke is the normal
    explosion: `explosion()` emits both). The test-only OIT blends leave the per-particle glow
    black.
  - **Other changes**: `GraphicsDesc` blends the second attachment separately (the water
    replaces the glow while blending its colour), which needs `independentBlend` (now a device
    requirement); `ToneParams::bloom` turns the bloom off for comparisons without changing the
    graph.
  - **Verification**: GPU `bloom_spreads_a_glow_but_not_lit_ground` (a glowing decal lifts a
    ring 12–28 px out by 7.5 luma with the bloom on; sunlit ground 80 px away changes by 0.0),
    and GPU `half_resolution_fire_in_front_of_smoke_blooms` (a fire in front of smoke lifts a
    ring by 10.7 luma at half resolution and 12.5 at full; smoke in front takes its share, 4.1;
    the first composite gives 0.0 at half). 38 GPU tests pass with validation; `cargo test
    --workspace` green.
  - **Cost** (`m12-before-bloom` against `m12-bloom-2`): battle 1 057 → 1 078 µs (+2.0%; bloom
    13.3 µs, tonemap unchanged), cliffs 862 → 851 µs and army-fox 8 168 → 8 265 µs (+1.2%, within
    the run spread), lights-1000 917 → 947 µs (+3.3%; its terrain spread exceeds the bloom). The
    scene draws pay a second colour attachment. The half-res particle glow adds a second
    half-resolution attachment: battle's `pass/particles` 85.0 → 151.6 µs and the frame
    1 078 → 1 143 µs (+6.0%); the other scenes have no particles there and stay within their
    spread.
  - Not done: the sky has no sun disc, so the sun itself does not bloom; bloom is a two-pass
    separable Gaussian, not a multi-scale pyramid; water replaces the glow under its whole
    surface, so a glowing decal under shallow, nearly transparent water loses its bloom (RA3 does
    the same; TODO); the test-only OIT particle blends do not feed the bloom.
- Step 5, automatic exposure — done (2026-11-21). Full write-up in
  [lighting.md](lighting.md) ("Automatic exposure").
  - **GPU-resident loop** (`ae_render::exposure`): `exposure.clear` (transfer) zeroes a 128-bin
    histogram; `exposure.meter` (compute) samples the scene at a quarter resolution (four taps
    per 4×4 block), averages the block and adds its log2 luminance to the bin (shared-memory
    bins per workgroup, then one global add per non-zero bin); `exposure.adapt` (compute, one
    invocation) trims the darkest and brightest 2%, computes the target
    `scene_ev100 + (mean_log2 - log2(key))`, eases toward it exponentially in EV space (1.5/s
    toward a brighter scene, 0.5/s toward a darker one, wall-clock time with pauses included;
    readback steps 1/60 s a frame) and clamps it to 10..18 EV. It writes a
    1×1 `RGBA32F` exposure image (eased EV, metered mean, target, metered flag). No CPU
    readback.
  - **Fog of war is not metered**: the meter samples the depth too and, where the frame shades by
    vision, divides each tap by the vision at its world position (rebuilt from the depth), so an
    unexplored frame meters as if in sight and the fog keeps its manual-EV darkness.
  - **Roles**: the meter declares the histogram `StorageBufferReadWrite` (a new frame-graph role,
    ABI code 17, `storage_buffer_read_write` in graph files: atomics read and write) and the
    adapt pass the image `StorageImageReadWrite` (it reads the previous EV).
  - **Where a new EV takes effect**: the scene draws still store at the camera EV
    (`ToneParams::ev100`, the reference); `bloom.downsample` and `post.tonemap` sample the
    exposure image and scale by `2^(scene_ev - eased_ev)`, so the frame's tonemap and bloom use
    the EV adapted from that same frame's metering. With auto off, adaptation writes the camera
    EV and the factor is exactly 1, leaving today's frames unchanged (`--ev100` alone selects
    this; the client's default is auto, and `--auto-exposure` in either order with `--ev100`
    keeps auto with that EV as the reference).
  - **The bloom's clamp is exposure-relative**: the downsample applies the display scale before
    clamping each tap at 4.0, so the clamp stays 4.0 of display-referred light whatever the
    adaptation does; the tonemap adds the already-scaled bloom.
  - **The key** is `KEY_LUMINANCE` 0.5, pre-curve display-linear: what the reference scene
    (untextured terrain, 100 000 lx) meters at EV14, so adaptation settles on the calibrated
    look (1.0 settled 0.9 EV brighter: display mean 0.773 against 0.596; M13 stage 1's
    curve: 0.353 against 0.366). Culling uses
    `MIN_EV100` (the widest exposure adaptation may reach) when auto is on and the graph has the
    exposure image, so a light is never cut that the brighter image would show; a bench run
    with auto against `--ev100 14` shows no consistent change.
  - **Graphs**: the native graph, the default module and `render/default.graph.json` declare the
    `exposure` image (as a library image), the `exposure.histogram` buffer and the three passes
    between the particles and the bloom; the tonemap and the bloom downsample sample the exposure
    image. The library checks the image, the histogram and the three passes as all-or-none, their
    order, and the two reads. The graph goldens were regenerated.
  - **Verification**: CPU `lighting::target_ev100`, `lighting::adapt_ev100` and the `library`
    wiring tests; GPU `auto_exposure_holds_the_key_across_sunlight`: a flat ground at 100 000
    and 25 000 lx converges over 5 s to display means 0.575 and 0.567 (ratio 1.015), while the
    same change at a manual EV14 gives 1.88; auto at 100 000 lx is within 4% of manual EV14
    (0.575 vs 0.596); an all-unexplored frame keeps 0.212 of the clear frame under auto and
    0.220 at manual (1.022 under auto with the vision correction switched off); the run is 10 s
    for the slower rates. GPU `auto_exposure_meters_the_glow_and_the_centre`: a 5×5 patch of
    glowing scars darkens the ground away from it to 192.0 luma with the bloom on and 192.3
    off (equal with the glow unmetered), and to 192.3 at the centre against 200.2 near the edge
    (198.2 and 199.7 with the centre weight at 1). All 43 GPU tests pass with validation, and
    the auto-exposure tests with sync validation. M13 stage 1's curve moved them: 0.353 and
    0.353 (ratio 1.000), manual 1.73, 0.353 against 0.366, and the glow 154.6/154.8, the
    centre 154.8 against 161.4 (159.7 and 160.8 with the weight at 1).
  - **The glow and the centre**: the meter adds the glow at the bloom's strength (clamped at 4.0
    at the reference exposure), and weights each block 4× at the centre falling over 0.25
    screen heights (`METER_CENTER_WEIGHT`/`METER_CENTER_SIGMA`). The glow is unblurred: metering
    the blurred bloom would put the meter after the bloom, whose downsample reads this frame's
    exposure, and the frame graph ties a read before a resource's first write to that write
    (a cycle). The bloom's blur spreads about 20 px at 1280×720, so the difference is small.
  - Open: no spot metering on a selection or the cursor; the meter includes the sky; one
    adaptation curve (two rates), no per-scene look or min/max override beyond the constants.

### M13 — HDR output on a supporting display (priority: lower; done)

- After M12 (needs the HDR scene target). Split into stages; each ends at a review that
  commits its code and docs before the next starts.
- **Range and gamut are separate axes.** Range: SDR, HDR10 (PQ) or scRGB (HLG is not
  used). Gamut: the working space the scene is lit in (BT.709, the default, or BT.2020)
  and the output gamut it is mapped to (BT.709, P3 or BT.2020).
- **Output chain in HDR graphs**: scene → `post.tonemap` → display → (module grade,
  vignette, later UI) → `post.encode` → swapchain. `display` is RGBA16F linear in units
  of paper white, in the working space, up to peak / paper white. `post.encode` maps the
  gamut to the output gamut by compression toward the achromatic axis (not clipping),
  then encodes PQ in BT.2020 at `v × paper white` nits (HDR10) or `v × paper white / 80`
  BT.709-relative (scRGB, negatives allowed).
- **SDR at the defaults stays byte-identical.** The tone curve (stage 1) is the one
  deliberate change to the SDR captures.
- Output gamut is a setting (`--display-gamut p3|bt2020|bt709`, P3 by default for HDR10).
  On Wayland the compositor does report the panel's target primaries
  (`wp_color_management`: here R 0.688477/0.308594, G 0.257813/0.669922,
  B 0.147461/0.061523, about 96% of P3); SDL does not expose them, so they stay a setting
  until the client queries the compositor itself. Levels default from SDL (paper white =
  SDR white level × 80, peak = paper white × headroom) with `--paper-white`/`--peak`
  overrides, corrected in stage 5 by the reference the WSI actually sends (below).
- **Stage 0, the probe, exists** (`ae-client --probe-display`, `ae_render::display`): it
  opens a window, enables `VK_EXT_swapchain_colorspace`, and prints SDL's HDR state and
  every (format, colour space) pair the surface offers, then exits. It renders nothing.
  On the reference machine (LG ULTRAWIDE 3840×1600 over DP-3, KWin Wayland, RTX 3060 Ti,
  Vulkan 1.4.357) it reports: HDR enabled; SDL white level 3.1 scRGB (248 nits), headroom
  1.778 (peak 441 nits); `VK_EXT_hdr_metadata` present; 30 pairs, among them
  `A2B10G10R10_UNORM_PACK32` and `A2R10G10B10_UNORM_PACK32` with `HDR10_ST2084_EXT` and
  `HDR10_HLG_EXT`, and `R16G16B16A16_SFLOAT` with `EXTENDED_SRGB_LINEAR_EXT`. Under
  `WAYLAND_DEBUG=1` KWin sends the same luminances (0, 441, 248), target_max_cll 441,
  target_max_fall 402, `tf gamma22`, and target primaries that match the EDID. KDE's HDR
  page shows SDR brightness 320 nits and peak 570 (an override); the protocol advertises
  248 and 441.
- The output description's white point is 0.359478/0.362865, against the EDID's D65
  0.3134/0.3291 — most likely Night Light or another white-point adjustment in KWin.
  Stage 3 takes only the R/G/B primaries (or P3) and keeps D65 white: KWin adapts the
  white point itself.
- **Decided from the probe (2026-11-21): both encodings.** HDR10 (PQ) at
  `A2B10G10R10_UNORM_PACK32` first, scRGB at `R16G16B16A16_SFLOAT` as the alternative;
  no HLG.
- **Measured (2026-11-22, stage 5): what the WSI itself sends.** Under `WAYLAND_DEBUG=1`
  the NVIDIA Wayland WSI's `wp_image_description_creator_params_v1` sends, for HDR10,
  `set_primaries_named(6)` (BT.2020) and `set_tf_named(11)` (`st2084_pq`) with **no
  `set_luminances`**, so PQ's protocol default reference white, 203 nits, applies. For
  scRGB it sends `set_primaries_named(1)` (sRGB), `set_tf_named(5)` (`ext_linear`) and
  `set_luminances(0, 80, 203)`: 1.0 in the signal is 80 nits, and the reference white is
  203 nits (the protocol's own default would be 80). KWin's output description (SDL's
  `get_output`) says `luminances(0, 441, 248)`, `target_max_cll 441` and
  `target_max_fall 402`: the display peaks at 441 and its SDR white is 248. KWin 6.8.3
  maps a surface's reference white onto its output's (`ColorDescription::toOther` in
  `src/core/colorspace.cpp` scales by `other.referenceLuminance() / referenceLuminance()`,
  or affinely black-to-black and reference-to-reference for the perceptual intent the WSI
  requests with `set_image_description(..., 0)`), and a PQ description without
  `set_luminances` gets `defaultReferenceLuminanceFor(PQ) = 203`
  (`src/wayland/colormanagement_v1.cpp`). So signal 203 shows at the desktop's 248, and
  the signal's levels are relative to 203: paper white is 203 in both encodings and the
  peak is 203 × SDL's headroom 1.778 = 361 nits of signal, which shows at the panel's 441.
  Sending the panel's absolute 441 would land at 441 × 248/203 = 539, and KWin adds its
  own perceptual tonemapper when the mapped peak exceeds the output's maximum
  (`colorpipeline.cpp`), compressing the highlights a second time after the curve's
  shoulder. Nothing is encoded at 248. (Review fix: the first stage-5 version sent 441.)
- Stages:
  - **1. A parametric, hue-preserving tone curve** (replaces Narkowicz's fit, and with it
    the ACES TODO above). Lottes 2016: `x^a / ((x^a)^d · b + c)`, with `pow` for the
    contrast and the shoulder and peak / paper white as a parameter; the hue handling is
    the max-channel tonemap plus the ratio, then a crosstalk/saturation step. SDR is the
    same curve at ratio 1. (The no-FMA and `libm` rules bind the sim, not render shaders,
    so `pow` is fine.) `KEY_LUMINANCE` and the EV14 reference move with the curve, as one
    change. Gate: sunlit untextured ground at the default exposure displays at a mean of
    0.35–0.45 linear (from 0.596), and a fireball keeps an orange-to-yellow gradient with
    no clipped disc; before re-baselining, before/after captures (near army, explosion)
    are posted for a look review; every moved threshold is listed old → new, including
    the auto-exposure GPU tests, with bench-diff.
  - **2. The display intermediate and `post.encode`, headless** (native graph, no
    swapchain or ABI change). `OutputEncoding {Sdr, Hdr10Pq, ScRgb}` and `DisplayLevels`
    in `LibraryEnv`; readback into A2B10G10R10_UNORM and RGBA16F. Gate: SDR goldens
    unchanged; PQ of 100/203/1000 nits within 1 LSB of ST.2084; HDR readback at peak =
    paper white matches the SDR frame; an emitter decodes above paper white and nothing
    above peak; zero validation messages.
  - **3. Working space and gamut mapping.** `--working-space bt709|bt2020`, conversions at
    every sample site and on the host (light, particle, team, fog, sky colours), the
    meter's luminance row, and compression to the output gamut in `post.encode`
    (`--display-gamut p3` by default; from the primaries only, D65 white). Gate: BT.709
    keeps the SDR captures byte-identical; unlit BT.2020 content reaches SDR within 1 LSB
    of BT.709 and the lit difference is reported as numbers; synthetic patches decode
    inside P3 and outside BT.709, are compressed with hue kept, and in-gamut patches are
    untouched.
  - **4. Authoring wide-gamut colours.** A gamut tag (`srgb` default, `p3`, `bt2020`) on
    the types effects and lights are authored with, converted at upload; one effect
    re-authored in P3; catalogue validation rejects an unknown tag.
  - **5. The HDR swapchain in the client.** Choose the pair from the probe's data, gate
    HDR on the setting and the offered pair and SDL's window state, recreate on
    `SDL_EVENT_WINDOW_HDR_STATE_CHANGED` and monitor changes, key the swapchain rebuild
    and the graph redeclare on (format, colour space), call `vkSetHdrMetadataEXT` after
    every recreate. Defaults come from SDL, corrected by the measured WSI reference.
    Flags `--hdr off|auto|hdr10|scrgb`, `--paper-white`, `--peak`, `--display-gamut`, and
    a toggle key. Gate: a windowed HDR run under `WAYLAND_DEBUG=1`, grepping the client's
    own `wp_image_description_creator_params_v1` requests (`set_tf_named`,
    `set_luminances`, `set_mastering_*`) and including what `vkSetHdrMetadataEXT` sends,
    to derive the encode reference from what is actually sent; a toggle through KDE's HDR
    switch recreating the swapchain both ways; `--hdr off` identical to stage 4; zero
    validation messages.
  - **6. Render ABI and modules.** Bump the ABI; `declare` carries the encoding, levels,
    working space and output gamut; SDK `display_spec`/`DISPLAY_NAME`; both guest modules
    and both JSON graphs; the post module's grade works above 1 (contrast pivots at
    mid-grey, no clamp to 1); new HDR `.dot` goldens.
  - **7. Measure and document.** Bench at 1440p and 3840×1600: tonemap and encode passes
    and the bandwidth of a 10-bit against a 16F swapchain, compared with `bench-diff`.
- **The tone curve was removed on 2026-11-22** (the user's decision). In SDR Lottes fitted 16
  scene units into white by lifting the darks (slope 1.29 at zero: scene 0.02 showed at
  0.025) and flattening the midtones (1.0 at 0.58, 2.0 at 0.75): the default map's luma
  spread at EV14 fell from 11.5 to 7.3 (5th-95th percentile 123-153 against 124-142), so
  the frame looked washed out; at the reference panel's 1.78 headroom the HDR curve was
  nearly linear to 1 and changed little. `Clamp` (its alternative) capped HDR at paper white,
  so `U` showed no difference with it. `post.tonemap` keeps its name and now applies the
  auto exposure, adds the bloom and maps the gamut (uniform 80 bytes, from 128); the output
  is the only clip (SDR saturates at 1, `post.encode` clips at the peak). `--tonemap`,
  `Curve`, `Lottes` and the crosstalk are gone. Auto-exposure's key is 0.32: at 0.5 with no
  curve the default map adapted brighter until its road and the lights on it clipped (mean
  sRGB luma 150 at 0.5 s, 186 at 10 s, `--capture`, 1280×720, 10,000 units); at 0.32 it
  settles at its start's look (143.5, then 152.8), byte-identical before and after the
  removal. GPU tests: `the_curve_places_the_sunlit_ground` is gone; five are retuned with
  their measured reasons (the auto-exposure key test settles at the key, 0.297 against
  0.32; the ground specular's light is 4 EV dimmer; particle emission order allows 3 codes;
  decals allow 1 code of float noise outside their extent; the waves move 1 in 60 water
  pixels, 34 of 1678, every change one code). 57 pass with validation. The GPU bench's
  baseline was recorded with Lottes, so its `pass/tonemap` differs from it; not re-recorded.
- **Stage 1 is done** (2026-11-21). Lottes 2016 (`x^a / ((x^a)^d * b + c)`) was then the
  default curve (`Clamp` stayed for exact linear reads). `a` and `d` are 1 (with these
  anchors a larger shoulder exponent makes the curve fall again past the peak, so values
  above it would display darker than white), and `b`, `c` are solved on the CPU from two
  anchors: the automatic exposure's key (0.5 pre-curve) displays at 0.40 linear, and 16.0
  displays as white. The curve maps the max channel and scales the others with it (hue
  kept); from 2.0 to 16.0 scene units a `smoothstep`-weighted share (up to 0.7) of the
  luminance-mapped grey joins in, so a fireball's core whitens while its edge stays
  orange. At EV14 the sunlit untextured ground displays at a mean of 0.366 linear (0.596
  with the fit) while the stored scene is unchanged (0.492), and automatic exposure
  settles at 0.353 against 0.366 manual. `KEY_LUMINANCE` and EV14 did not move: the
  curve's mid anchor *is* the key, so the calibration and the curve are one definition.
  Before/after captures (the army at 1400 m; 12 explosions at 350 m) turn the fireball's
  flat white discs (253,253,252; mean chroma 26 above 200 luma) into a warm gradient with
  an orange rim (core 243,242,232; mean chroma 67 above 200 luma), and the ground reads
  darker with more colour. Three GPU thresholds were re-baselined, each one measuring a
  code delta the curve compresses: `decals_relit_modulated_and_glowing`'s glow rise 60 →
  25 codes (33 measured, 95 → 129), `scar_normal_maps_relight_the_ground`'s close-up mean
  |dluma| 3.0 → 2.5 (2.93 measured), and `water_covers_the_low_ground_only`'s wave motion
  a fifteenth → a twentieth (105 of 1678 pixels). All the others pass unchanged, including
  `auto_exposure_holds_the_key_across_sunlight`, whose manual ratio fell from 1.88 to 1.73
  against its 1.6 floor. New `the_curve_places_the_sunlit_ground` pins the gate.
  Cost: `pass/tonemap` 16.4 → 17.4 µs in `--bench army` at 1280×720 (three interleaved
  rounds against the ACES commit on the same machine; one 1.024 µs timestamp step),
  `gpu_total` +0.2%. `bench-diff` against `bench/results/baseline` is not usable here: that
  baseline is from 785e1d3 (2026-11-14), has no exposure passes, and flags M4–M12's own
  growth (26 regressions unrelated to the curve); re-record it (TODO).
- **Stage 2 is done** (2026-11-21). `post.encode` and the display intermediate exist in the
  native graph, headless: `OutputEncoding {Sdr, Hdr10Pq, ScRgb}` and `DisplayLevels
  {paper_white, peak}` (nits) in `LibraryEnv`, `ae_render::encode` holding the pass and a
  CPU mirror of the shader's mapping. A `TerrainApp::with_display` renders to
  `A2B10G10R10_UNORM_PACK32` or `RGBA16F` and reads back the packed 10-bit or half-float
  pixels. `post.encode` samples the display 1:1 (`texelFetch`), clamps at `peak /
  paper_white`, and writes ST.2084 in BT.2020 (a pure re-expression of the working space's
  BT.709: the gamut *compression* is stage 3) or scRGB at `v * paper_white / 80` with
  negatives kept. The curve's peak anchor is `Lottes::peak_out`: 1 in SDR, `peak /
  paper_white` in HDR, solved with the mid anchor; the key still displays at 0.40 of paper
  white either way, but only the key: below it the HDR toe is flatter (slope at zero 0.95
  at 441/203 nits against SDR's 1.29), so darks display about 20% darker than in SDR
  (scene 0.1: 0.092 against 0.115), and from the key to paper white brighter (scene 1.0:
  0.691 against 0.579). The scene's peak stays at 16 units — the content's range, chosen in
  stage 1 — and only what it displays at rises to the display's peak, so the content's
  whole range uses the display's; moving the scene's peak to 16 × headroom would reserve
  the top of the display's range for scene light the content does not have, and scaling
  both anchors with the headroom would brighten every midtone. At peak = paper white the
  curve is the SDR one (the identity the readback test measures). SDR is untouched:
  `post.tonemap` writes the sRGB target directly, the native graph has no display and no
  encode, and the `.dot` goldens did not move.
  Measured: `post.encode` writes ST.2084's 10-bit codes at 100, 203 and 1000 nits exactly
  (0 LSB off, 520/594/769) and a value past the peak stops at its code; at peak = paper
  white the HDR10 frame's decoded luminance matches the SDR frame's to a mean of 0.0019 and
  a max of 0.0032 linear (one quantization step), the 80-nit scRGB readback matches it to
  0.0011/0.0028 per channel and the 203-nit one over 2.5375 the same, and an explosion
  (203-nit paper white, 441-nit peak) reaches 437.7 nits with no channel above 439.8. All
  49 GPU tests pass with validation (5 new), the lib tests 58, and the module and golden
  tests are unchanged. SDR is byte-identical: the same skirmish capture from the stage-1
  commit and this stage compares equal byte for byte (the solver's peak equation is the old
  expression when `peak_out` is 1). Cost and HDR speed are measured in stage 7 (benchmarks
  are deferred there). Not built: HDR swapchains (stage 5), the module ABI (stage 6), so an
  HDR app reads back and a module graph cannot be HDR yet.
- **Stage 3 is done** (2026-11-21). Range and gamut are separate:
  `ae_render::colorspace` holds `WorkingSpace {Bt709, Bt2020}` and
  `Gamut {Bt709, P3, Bt2020}` with matrices from the primaries only (D65 white; the
  panel's white-point adjustment stays the compositor's), luminance rows and a
  compression toward the achromatic axis. The app takes `with_working_space` and
  `with_display_gamut`; `LibraryEnv` carries both, and the client has
  `--working-space bt709|bt2020` and `--display-gamut bt709|p3|bt2020`. BT.709 is the
  default and SDR stays bit-identical: the release skirmish capture from stage 2 still
  compares equal with `cmp`, and the `.dot` goldens did not move.
  Every authored colour enters the lighting in the working space. On the host: the
  sun/sky tints (and the sky's radiance, clear colour, fog and water reflection), light
  colours, particle effect colours, team colours and model lights; in the shaders: the
  terrain's composed albedo, the models' base colour and emission, the water's foam and
  body colours. `photometry.y` stops being unused and is the shaders' wide flag. The
  exposure meter and the curve's crosstalk take the working space's luminance row.
  `post.encode` maps the display to the output gamut (P3 by default for HDR10), then
  compresses when the output gamut is narrower than the working space (luminance and
  hue kept, in-gamut colours untouched, only negative channels count), then re-expresses
  in the container (BT.2020 for HDR10; BT.709 for scRGB, whose default output gamut is
  BT.2020 so its extended negatives pass). SDR has no encode: `post.tonemap` maps the
  curve's result to BT.709 itself, with the identity matrix at the default. In HDR the
  tonemap's matrix is the identity and the display stays in the working space, so the
  gamut map runs once (review fix: the first version mapped to BT.709 in HDR too, and
  post.encode then read the display as BT.2020, 8.74% off on the sky; the SDR
  compression also took the working space's luminance row for a BT.709 colour).
  Measured: an unlit sky in BT.2020 reaches SDR within 1 code of BT.709 (worst 1, 2522
  of 57 600 channels differ); the same sunlit ground lit in the two spaces differs by a
  channel mean of 1.54 codes, p99 2 and max 2 (46.3% within 1 code), the expected residue of
  multiplying colours in different bases and curving the max channel. With a BT.2020
  working space and P3 output, P3 green decodes inside P3 (0.000, 1.003, -0.000) and
  outside BT.709 (-0.226 red), an in-P3 patch is untouched (0.301, 0.502, 0.201 against
  0.300, 0.500, 0.200), and BT.2020 green is compressed into P3 (0.001, 0.955, 0.186),
  still outside BT.709, luminance and hue kept. Automatic exposure meters the BT.2020
  scene with its own luminance row: the sunlit ground settles at 0.3607 against BT.709's
  0.3529 (ratio 1.022, the lit basis difference above, not an exposure change). The
  unlit sky through the whole HDR10 chain decodes to the same light in either working
  space, worst channel 1.04% apart. 54 GPU tests pass with validation (5 new), 64 lib tests (6 new), module 1, golden 2,
  graph_file 2; a `--working-space bt2020` client capture renders with 0 validation
  errors and warnings. Re-checked after the review fixes: the release explosion capture (`--map land
  --units 0 --camera 6000,6000,350,0.6,0.45 --explode 12`) is byte-identical between
  75d276d and stage 3, and repeats byte for byte.
  The levels stay at PQ's reference 203 nits in the tests and defaults: SDL reports 248
  nits of SDR white on the reference machine, but that is the compositor's reference,
  not necessarily the one the WSI attaches to the swapchain; stage 5 measures what is
  actually sent, and nothing here builds a gamut decision on 248. Not built then: a gamut
  tag on authored colours (stage 4, below), the HDR swapchain (stage 5), the module ABI
  (stage 6).
- **Stage 4 is done** (2026-11-22). Authored colours carry a gamut tag:
  `Gamut` serialises as `srgb` (the default), `p3` or `bt2020` (`bt709`,
  `display-p3` and `rec2020` are aliases), and an unknown tag is a parse error. Effects
  moved from a Rust table to `assets/effects.json` (version 1), a strict catalogue:
  unknown names, fields, versions and tags are rejected, and
  `particles::parse_effects` validates it and orders the table by `effect` index
  (`EFFECT_NAMES`: flash, fire, sparks, smoke, dust). Each definition's particle colours
  and burst light tint are in its gamut; `ParticleRenderer::new` converts them into the
  working space at upload and normalises the light tint there (`unit_tint_from`).
  `UnitClass` carries a gamut too (converted in `UnitRenderer::new`). The engine's fixed
  and scattered lights stay BT.709 (`Light` is the 64-byte GPU record) and glTF's model
  lights are linear sRGB by the spec. The fire is re-authored in P3 as unit-luminance
  tints like every emissive colour (the `emission` scalar carries the brightness; a
  catalogue test checks it): its death tint `[3.784, 0.189, 0.032]` uploads into the
  default BT.709 working space as `[4.593, 0.038, -0.054]`, and an HDR10 P3 output
  decodes it back to `[3.789, 0.187, 0.032]` (still outside BT.709). The emission adds
  linearly, so in BT.709 the negative blue takes blue from the light behind the fire
  (exact in a linear basis); only a pixel that ends negative is clipped, by the SDR
  target. Review fix: the first P3 fire had luminance 0.965 and 0.951, dimming it by
  3.5-4.9% along with the hue change. The display intermediate still
  stays in the working space: only SDR's tonemap or `post.encode` maps it to the output.
  55 GPU tests pass with validation (1 new), 67 lib tests (3 new), module 1, golden 2,
  graph_file 2 (after the review: 68 lib tests, 4 new). The default SDR frame without
  explosions is still byte-identical to stage 3 (release, `--map land --units 0 --camera
  6000,6000,350,0.6,0.45`, built from 0ae663c and from stage 4); the explosion capture
  (the same plus `--explode 12 --capture-frames 10`) changes as intended: 48 136 of
  921 600 pixels differ (5.22%), mean channel 0.249, max 63, and the brightest 200
  pixels' mean goes from (239.5, 233.8, 158.9) to (239.6, 228.2, 102.4), the fire's
  deeper red. Not built then: a light catalogue beyond
  effects (fixed lights are generated, not content), a gamut tag in glTF (`extras`), the
  HDR swapchain (stage 5), the module ABI (stage 6).
- **Stage 5 is done** (2026-11-22). The client's swapchain carries an HDR encoding:
  `--hdr off|auto|hdr10|scrgb` (auto by default) resolves against the surface's pairs
  (`ae_render::swapchain::{HdrRequest, TargetFormat, resolve_target}`). Auto takes HDR10
  (`A2B10G10R10_UNORM_PACK32` + `HDR10_ST2084_EXT`) when the surface offers it, else scRGB (`R16G16B16A16_SFLOAT` +
  `EXTENDED_SRGB_LINEAR_EXT`), else SDR. An explicit mode the surface does not offer is
  an error naming what it does; nothing falls back silently. `VK_EXT_swapchain_colorspace`
  is enabled on the instance (with a surface) and `VK_EXT_hdr_metadata` on the device, and
  `vkSetHdrMetadataEXT` runs after every recreate. The swapchain is recreated on
  `SDL_EVENT_WINDOW_HDR_STATE_CHANGED`, display/monitor events and the `U` toggle key;
  on a change of the (format, colour space) pair the app is retargeted and the graph
  redeclared. `RenderApp::retarget` destroys and creates the app by default; the terrain
  app rebuilds only the tonemap, `post.encode` and the render module's programs, and
  keeps the terrain renderer, units, particles, lights and exposure (fixed 2026-11-22:
  the full rebuild restarted the terrain's tree at 4 leaves, which took ~18 frames to
  refine back to ~700 on the default map, so the ground and water visibly settled on
  every `U` press, and the sim's ground edits went back to the loaded map; with the
  retarget the per-frame leaf count stays at 700 through the toggles, and a 11 s
  validated window toggling every 2 s reports 6 recreates and 0 errors/warnings). The
  window title shows the output and its levels beside the display's, e.g. `HDR10 (PQ),
  white 203, peak 361 nits -> display white 248, peak 441 nits`. The app takes its encoding from the swapchain's colour space and its levels from
  `HdrSettings` (paper white 203 from the WSI's description, peak 203 × SDL's headroom
  = 361 in the signal's nits, shown at the panel's 441);
  `--paper-white`/`--peak` override, `--display-gamut` keeps its stage-3 meaning (P3 by
  default for HDR10). `--capture` still writes the SDR readback and does not show the HDR
  swapchain. A render module has no HDR support until stage 6, so `--render-module` with
  HDR requested logs one line and falls back to SDR.
  Measured under `WAYLAND_DEBUG=1`, the client's own `wp_image_description_creator_params_v1`
  requests are:

  ```
  # --hdr hdr10
  -> wp_color_manager_v1#64.get_surface(new id wp_color_management_surface_v1#86, wl_surface#5)
  -> wp_color_manager_v1#64.create_parametric_creator(new id wp_image_description_creator_params_v1#87)
  -> wp_image_description_creator_params_v1#87.set_primaries_named(6)        # BT.2020
  -> wp_image_description_creator_params_v1#87.set_tf_named(11)              # st2084_pq
  -> wp_image_description_creator_params_v1#87.set_max_cll(361)
  -> wp_image_description_creator_params_v1#87.set_max_fall(361)
  -> wp_image_description_creator_params_v1#87.set_mastering_display_primaries(680000, 320000, 265000, 690000, 150000, 60000, 312700, 329000)
  -> wp_image_description_creator_params_v1#87.set_mastering_luminance(0, 361)
  -> wp_image_description_creator_params_v1#87.create(new id wp_image_description_v1#88)
  -> wp_color_management_surface_v1#86.set_image_description(wp_image_description_v1#88, 0)
  # no set_luminances: PQ's protocol default reference white is 203 nits

  # --hdr scrgb
  -> wp_image_description_creator_params_v1#87.set_primaries_named(1)        # sRGB
  -> wp_image_description_creator_params_v1#87.set_tf_named(5)               # ext_linear
  -> wp_image_description_creator_params_v1#87.set_max_cll(361)
  -> wp_image_description_creator_params_v1#87.set_max_fall(361)
  -> wp_image_description_creator_params_v1#87.set_mastering_display_primaries(708000, 292000, 170000, 797000, 131000, 46000, 312700, 329000)
  -> wp_image_description_creator_params_v1#87.set_mastering_luminance(0, 361)
  -> wp_image_description_creator_params_v1#87.set_luminances(0, 80, 203)     # 1.0 = 80 nits, reference white 203
  -> wp_image_description_creator_params_v1#87.create(new id wp_image_description_v1#88)
  ```

  The values `vkSetHdrMetadataEXT` sends (logged by the client) are the output gamut's
  primaries with D65 white, `max_luminance = peak` (361), `min_luminance = 0`,
  `max_content_light_level = 361` and `max_frame_average_light_level = 361` (a stated
  conservative bound: no frame average can exceed the peak). The WSI forwards them into
  the description (`set_max_cll`, `set_max_fall`, `set_mastering_*` above), so they are
  in the signal's nits, relative to its 203 reference, like the levels. KWin's output
  description is `luminances(0, 441, 248)`, `target_max_cll 441`. The excerpts above
  were re-recorded after the review fix (3 s windowed runs, 0 validation messages); the
  WSI sent a description in every run observed (HDR10 and scRGB).
  `--hdr off` is byte-identical to stage 4 (07e928a): release captures at `--map land
  --units 0 --camera 6000,6000,350,0.6,0.45`, with and without `--explode 12
  --capture-frames 10`, compare equal with `cmp`. Under `AE_VK_VALIDATION=1`: a 10 s
  windowed HDR10 run with `--exercise` (resize/minimize) and the same with scRGB each
  report 9 swapchain recreates and 0 validation errors/warnings; a 24 s run through
  KDE's HDR switch (off/on twice, `kscreen-doctor output.DP-3.hdr.disable|enable`)
  recreates 5 times and reports 0/0 (these runs used the first version's absolute
  peak: 441 in HDR mode, and 203 in SDR mode, floored at the paper white because KWin
  advertises 200 there; with the fix the peak is 203 × headroom, 361 and 262, and no
  floor is reached); a run
  with the runtime toggle (the `U` path, `AE_HDR_TOGGLE_SECONDS=3` for the test hook)
  recreates 5 times, rebakes 5 graphs and reports 0/0. SDL keeps reporting the window
  HDR-capable while KWin's HDR switch is off (SDR white 155, peak 200), so `auto` stays
  on HDR10 and the metadata follows the lower headroom; the compositor maps the
  signal's 203 onto its 155.
  Since 2026-11-22 `auto` does not consult SDL at all: it takes HDR10, then scRGB, whenever
  the surface offers them (HDR10 first: 4 B/px against scRGB's 8, and the signal HDR
  displays take natively), and SDL's state only supplies the headroom (1.0, peak = paper
  white, when SDL reports no HDR). That settles the KDE-switch-off question by keeping HDR.
  55 GPU tests pass with validation, 73 lib tests (5 new, one of them the default
  levels), module 1, golden 2, graph_file 2; `cargo test --workspace` is clean. Two bugs
  the recreate exposed are fixed: the instance now enables `VK_EXT_swapchain_colorspace`
  (and the device `VK_EXT_hdr_metadata`) only with a surface, and after an app rebuild the
  render thread has the executor forget every external access state
  (`Executor::forget_external_states`), so a new image under a reused identity (the
  exposure image first showed it) does not inherit the old one's layout. Re-run after the
  review fixes, under `AE_VK_VALIDATION=1`: 8 s `--exercise` with HDR10 and with scRGB
  (7 recreates each) and 9 s of `AE_HDR_TOGGLE_SECONDS=2` (5 recreates, 5 rebakes,
  metadata resent at 361 on each HDR entry), all 0 errors and 0 warnings; the `--hdr off`
  captures still compare equal with 07e928a's. Not built then: HDR
  captures, the module ABI and HDR render modules (stage 6, below), benchmarks
  (stage 7).
  With peak 361/203 the curve's HDR anchor is the headroom 1.778: `b` 0.500, `c` 1.000,
  toe slope 1.00 (0.95 with the first version's 441), so scene 0.1 displays at 0.095
  against SDR's 0.115, scene 1.0 at 0.667 against 0.579, paper white is reached at scene
  2.0 and a fireball core (scene 4.2) at 1.355 of paper white, 275 signal nits (about 336
  on the panel). Look checks for the display (not measured in a capture): HDR darks
  against SDR, the far ground's toe (stage 1's TODO), and whether the P3 fire reads
  redder in HDR than in SDR.
- **Stage 6 is done** (2026-11-22). The render ABI's contract version is 2
  (`ae_render_abi::module::MODULE_VERSION`; the graph descriptor's wire format stays 1:
  no codes changed and nothing moved, and the program table stays 1). Before `declare`,
  the host writes a [`DeclareInput`] (`ae_render_abi::module::DeclareInput`, 60 bytes)
  into the module's memory at the address its `ae_render_declare_input` export returns:
  the target's width, height, colour format and scene flags, and the output's display
  spec — the encoding (`Sdr`/`Hdr10Pq`/`ScRgb`), the levels in signal nits
  (`paper_white`, `peak`), the working space (BT.709 or BT.2020) and the output gamut
  (BT.709, P3, BT.2020). A module built for the old contract is refused before anything
  else is read: `render module ABI version 1; this host runs 2 (rebuild the module)`
  (checked with the real v1 `render/post` built from bb69261, and pinned by a WAT test
  with that exact text). The SDK exposes the spec as `DeclareEnv` (`encoding`,
  `paper_white`, `peak`, `working_space`, `output_gamut`, `uses_display()`,
  `headroom()`) and adds `display_spec()`, `DISPLAY_NAME` and `DISPLAY_FORMAT`; the
  display rule is written into both the ABI and SDK docs.
  Both guest modules and both JSON graphs now declare the HDR tail: `post.tonemap`
  writes the display intermediate (`RGBA16F`, paper-white units, working space), the
  module's colour passes run on it, and `post.encode` is the only pass that writes the
  target; SDR keeps `post.tonemap` writing the target with no display and no encode.
  `graph_file` gained `when: hdr|sdr` and the `display` format name, and a program's
  uniform block can switch with a flag bit (`uniforms_flag`/`uniforms_off`, for a pass
  that always runs but must be a no-op when off). `check_graph`'s rule (post.encode
  exactly when the output is HDR) now applies to module graphs, with messages naming
  the missing or extra pass and the encoding:
  `the output is HDR10 (PQ) but the graph has no post.encode pass: the tonemap writes
  the display, not the target`.
  The post module's grade runs on the display, before the encode: contrast pivots at
  0.18 of paper white, saturation mixes around Rec. 709's luma axis (luminance kept in the BT.709 working space),
  and nothing is clamped at 1 — only `post.encode` maps the display, and above-peak
  values stop at the peak there. Its vignette multiplies the display
  (`1 - strength * smoothstep(radius, 1, d)`), which is range-safe: paper white and the
  peak scale together. In SDR the grade writes the target and the vignette still blends
  black over it (there is no display image to multiply); the SDR frame is unchanged.
  The client's `--render-module` + HDR fallback and the toggle refusal are gone: a pair
  change destroys and creates the app as before, and the running module stays loaded
  (its format-dependent programs are rebuilt) and redeclares for the new display spec;
  `declared_for` includes the encoding and the levels, so a levels change redeclares too.
  Six HDR `.dot` goldens are new (native HDR10 and scRGB with the full scene, and
  `default`/`post` for each); the ten SDR goldens are unchanged.
  Measured: `--hdr off` release captures are byte-identical to bb69261 for the native
  graph, `--render-module default`, `--render-module post` and both JSON graphs
  (`--map land --units 0 --camera 6000,6000,350,0.6,0.45`, baseline built in a worktree
  with its own `CARGO_TARGET_DIR`; all five compare equal with `cmp`). The default
  module's HDR10 readback equals the native graph's exactly — worst channel 0 PQ codes,
  0 of 64,000 pixels differ. The post grade, extracted from the built module and run
  before a real `post.encode` (paper white 203, peak 1000 so the encode does not clip):
  at display 1.5 the native encode decodes to 1.503 and the grade to
  `[1.720, 1.660, 1.562]` (the maths says `[1.725, 1.658, 1.559]`); at 2.0, 1.996 to
  `[2.303, 2.222, 2.093]` (`[2.307, 2.218, 2.085]`); a coloured `[1.8, 1.2, 0.9]`
  goes to `[2.160, 1.304, 0.858]` and `[2.4, 1.6, 1.2]` to `[2.887, 1.750, 1.158]`
  (in paper-white units; every channel within 0.008 of the maths, and the luminance is
  the input's through contrast and the tint). With the client's 361-nit peak, display
  values above 1.778 are above the display's peak and `post.encode` stops them there,
  as for the native graph. Under `AE_VK_VALIDATION=1`, all 0 errors and 0 warnings:
  `--render-module post --hdr hdr10 --exercise` for 10 s (480 frames, 9 recreates,
  9 bakes), the same with scRGB (490 frames, 9 recreates, 9 bakes), the `U` toggle path
  with the module for 13 s (5 recreates, 5 bakes — the module redeclares for each
  encoding), a 30 s HDR10 run with two real hot reloads of the post module (two source
  edits that changed its bytes: 4273 frames, 3 loads, 0 module errors), and the post
  JSON graph in HDR10 for 5 s (675 frames). 58 GPU tests pass with validation (3 new:
  the module's HDR10 frame against native, the HDR reload/vignette, the grade above
  paper white), 73 lib tests, 3 ABI tests (1 new), module 3 (1 new), golden 1 (16
  cases), graph_file 2; `cargo test --workspace` is clean. Not built: HDR captures
  (stage 5's line stands), benchmarks (stage 7).
- **Stage 7 is done** (2026-11-22); M13 is complete. `ae-client --bench` gained
  `--bench-output sdr|hdr10|scrgb` (the readback rendered in that encoding through
  `TerrainApp::with_display`, at fixed 203/361 signal nits so a run does not depend on the
  display) and `--bench-size WxH`. A non-SDR run's records are `gpu-<scene>-hdr10` or
  `gpu-<scene>-scrgb` (HDR10 has SDR's 4 B/px, so the workload info alone would not keep
  `bench-diff` from pairing it with an SDR record). The readback's `copy` and `host read` passes are reported
  (`pass/copy`, `pass/host read`, with the host bus's 50% `bench-diff` tolerance) but stay
  out of `gpu_total`, which therefore remains comparable with older records.
  Measured on the reference machine (RTX 3060 Ti, KWin Wayland; release; 300 frames after
  60 warm-up frames and at least 1 s; two runs per configuration; scenes `overview`, `army`,
  `battle` (2 explosions a frame) and `close`; medians over the scenes, with the largest
  between-run change in brackets):

  | 2560×1440 | SDR | HDR10 | scRGB |
  |---|---|---|---|
  | `pass/tonemap` | 62 µs (2%) | 77 µs (1%) | 78 µs (0%) |
  | `pass/encode` | — | 68 µs (3%) | 84 µs (0%) |
  | `pass/copy` | 661 µs (0.2–7%) | 662 µs (2%) | 1 325 µs (6%) |
  | `gpu_total` (`army`) | 1.41 ms | 1.50 ms (+6%) | 1.52 ms (+7%) |

  | 3840×1600 | SDR | HDR10 | scRGB |
  |---|---|---|---|
  | `pass/tonemap` | 108 µs (1%) | 128 µs (1%) | 128 µs (1%) |
  | `pass/encode` | — | 112 µs (0%) | 136 µs (1%) |
  | `pass/copy` | 1 101 µs (2%) | 1 088 µs (0.3%) | 2 342 µs (1–2%) |
  | `gpu_total` (`army`) | 2.13 ms | 2.27 ms (+7%) | 2.27 ms (+6%) |

  The HDR tail adds 1.6–8% of `gpu_total` across these scenes (1.6–4.3% in `battle`, whose
  particles dominate); HDR10 and scRGB totals are within run noise of each other. Bytes per
  frame: 1440p is 3,686,400 px, 14.75 MB at 4 B/px (SDR and HDR10) against 29.49 MB at
  8 B/px (scRGB); 1600p is 6,144,000 px, 24.58 against 49.15 MB. The copy moves twice that
  (the image is read and the buffer written): 44.6 GB/s at 1440p for all three encodings and
  42.0–45.2 GB/s at 1600p, so doubling the target's bytes doubles the copy time
  (661 → 1 325 µs at 1440p, 1 101 → 2 342 µs at 1600p). The encode is fill-bound on the
  target and scales with pixels (68 → 112 µs for HDR10 and 84 → 136 µs for scRGB at 1.67×
  the pixels).
  `bench/results/baseline` was re-recorded at 06fa4c4 with `AE_BENCH_MACHINE=rtx3060ti` and
  the default settings (SDR, 1280×720; both `ae-tool bench all` and `ae-client --bench all`),
  closing stage 1's TODO. `ae-tool bench-diff` against 2804be3 (the last pre-M13 commit,
  built in a worktree with its own `CARGO_TARGET_DIR`) flags 5 of 504 metrics: `pass/tonemap`
  +13.3% (`lights-low`, `orbit`) and +23.1% (`palette`) from the Lottes curve, verified in
  isolation (e.g. `army` 16.4 → 17.4 µs, `army-model-lights` 17.4 → 19.5 µs at 1280×720);
  `pass/water` 5.1 → 178.2 µs, the known bimodality in the M14 TODO (it flips on either
  binary); and `pass/units` in `army-model-lights` +37% in that one run, while three isolated
  repeats give 3.18–3.21 ms on the parent and 3.18–3.23 ms here. A full-suite timing is not
  stable enough for the light scenes: two full runs of the *same* binary, diffed against
  each other, flag 11 metrics, all `pass/terrain`/`gpu_total` in them (up to +215% on
  `lights-200`'s terrain, a GPU power-state effect), which is why the stage tables above come
  from repeated fixed-subset runs. `--hdr off` remains byte-identical to ba9600f: the two
  standard land captures (`--map land --units 0 --camera 6000,6000,350,0.6,0.45`, with and
  without `--explode 12 --capture-frames 10`) compare equal with `cmp`. 58 GPU tests pass
  with validation (unchanged), 73 lib tests, 3 ABI, module 3, golden 1 (16 cases),
  graph_file 2; `cargo test --workspace` is clean.
- **Fire covers the ground** (2026-11-22). In HDR10 a fireball looked transparent: the scorch
  mark and a fox behind it showed through. Fire and flash were purely additive (alpha 0), so
  nothing behind them was ever hidden; SDR's clip at 1.0 turned the sum into a flat white disc,
  HDR's clip at the peak (361 / 203 = 1.78× paper white on the reference panel) kept the
  ground's variation, and the scorch mark had faded in fully after 0.4 s while the fire burns
  0.35–0.9 s. Two changes:
  - `EffectDef::cover` (0..1 of the opacity, `#[serde(default)]` 0, emissive effects only):
    an emissive particle's alpha. Fire 0.5, flash 0.3 (`assets/effects.json`). The glow
    classifier no longer reads alpha 0 as emissive: the flag is the third component of the
    particle's depth varying (a separate flat varying draws a performance warning from the
    layer in every pass that does not read it) and `PackedPuff.ext.w` for the packets; the
    moments blend counts an emissive particle's cover as black media.
  - `Decal::scorch` (and so every scar) fades in over `SCORCH_FADE_IN`, 1 s (was 0.4 s).

  Measured by `explosion_fire_covers_the_ground` (flat ground at 320×240, camera 80 m at pitch
  0.9, one scale-1 explosion's flash and 24 fire puffs, their emission and lights off so only
  what they cover shows, half-resolution sorted blend, bloom off): the share of the ground
  left in a 6 px disc at the burst is 0.030 at 0.10 s, 0.102 at 0.30 s and 0.731 at 0.60 s
  (the fire has risen and faded; the decal's fade-in takes over), within 20 px 0.619, 0.603
  and 0.860. With cover 0 it is 1.000. Fire 0.8 would give 0.004 / 0.026 / 0.603; 0.5 keeps
  the fireball's rim translucent. 58 GPU tests pass with validation (one new), 68 lib tests,
  golden 1; a 3 s HDR10 window with `--explode 6` reports 0 errors, 0 warnings. Not yet
  looked at on the panel.
- Not in M13: HDR screenshots and captures (they stay 8-bit sRGB); UI composition (M8,
  which should draw into `display` at paper white); HDR on X11; per-display ICC profiles;
  dynamic HDR metadata.

### M14 — Unit scripts (in design)

What a unit does with its model in a given situation: choose, blend and hold clips; turn joints;
aim (possibly a sequence: take the rifle off the back, turn the body, then aim); fire with a
muzzle flash, recoil, particles, sound; deploy; die. An FSM per unit with coroutines and a
scheduler, after Recoil's unit scripts (`RecoilEngine/rts/Sim/Units/Scripts`: callins
`Create`, `StartMoving`, `StopMoving`, `AimWeapon`, `FireWeapon`, `Killed`, …; threads with
`Sleep`, `WaitForTurn`, signal masks; engine-side `Move`/`Turn`/`Spin` of pieces at a speed).
Replaces the client's `--clip`.

Decided with the user (2026-11-16):
- **Scripts are sim (synced)**, as in Recoil. A sequence like the rifle one is gameplay (the
  shot waits for it), and a pose that only the client knows cannot be blended with script-driven
  joints, nor checked against bounding and collision volumes. Running scripts outside the sim
  (on the view) stays a maybe for later; the script API should not depend on where it runs.
- Scripts are written in Rust (or anything that compiles to WASM), in the game's sim module. **No
  custom bytecode.**
- **One global scheduler**, not one per script, so woken coroutines run in parallel.
- `Sleep` takes milliseconds, so scripts do not depend on the tick rate (chosen once per engine
  start): `ceil(ms × rate / 1000)` ticks.
- Clip play modes: loop, ping-pong, once (in the M11 path now, `ae_model::pose::PlayMode`;
  carried over).
- A death sequence is a must: a killed unit stays in the sim, not targetable and not colliding,
  until its `Killed` script ends (with a cap), then it is removed (or leaves a corpse).

Design:
- **Animation channels are synced columns; the engine animates.** `Turn(joint, axis, angle,
  speed)` and friends only set targets. Per unit: clip layers and joint tweens (turn, move, spin
  to a target at a speed), in fixed-size columns (a small, per-game maximum of layers and
  tweens). A native ticker advances them every tick without calling the script (the **fast
  path**: a unit that only plays clips or tweens never runs script code) and wakes coroutines
  waiting on them. A clip's speed can follow the unit's ground speed (no foot sliding).
- **Blending and partial override** (decided with the user, 2026-11-16): a unit runs and aims
  at the same time, the legs from a clip, the upper body from another clip plus procedural
  turns. The pose is built per joint, in local space (translation and scale lerp, rotation
  nlerp on one hemisphere), from an ordered stack:
  - **layers**, each a clip with a play mode, speed, start tick and a **weight**. A layer is
    one of:
    - *blend*: mixed with the layers below by its weight (cross-fades: walk to run, idle to
      walk; or two clips mixed by a parameter such as ground speed);
    - *override*: replaces the layers below on its **joint mask** (by its weight);
    - *additive*: adds its clip's change from the clip's reference pose (recoil kick,
      breathing, a flinch on a hit).
  - **joint masks**, named per model: a joint and its descendants (e.g. "upper body" from the
    spine up), optionally with a weight per joint (a spine twist spread over three spine joints).
    Built from the model's joint names at load; a layer or tween without a mask covers the whole
    model.
  - **weights fade**: `blend_in` / `blend_out` times move a layer's weight to its target in the
    ticker (no script call), and a layer at weight 0 after `blend_out` is removed.
  - **manual joint channels: a bone is either manual or clipped** (decided with the user,
    2026-11-16; replaces "tweens on top, overriding or additive"). Turn, spin, move and scale
    on a bone knock the clips out of that bone, and the other way round. Exclusion is per
    bone: its children still take their clips (a turret turn under a clipped hull; a spine turn
    with clipped arms). No additive turns and no turns spread over a mask; aim-while-running is
    a locomotion layer on a lower-body mask plus manual turns above it.
    - Entries are found by bone, not by slot: the next turn or spin on a bone replaces its
      rotation entry (Recoil did it per axis), a move its translation entry, a scale its
      scale entry. Channels a manual bone has no entry for hold the rest pose.
    - A turn is axis/angle with a speed, acceleration and deceleration (a trapezoid speed
      profile). Inside, the entry holds the current and target rotations as quaternions,
      never sign-normalised: the target is `(cos θ/2, sin θ/2 · axis)` and the sign of
      `dot(current, target)` picks the arc (positive the short way, negative the long way),
      so a turn by 270° goes the long way. One turn covers less than a full revolution; more
      is a spin (axis, rate, acceleration; `stop_spin` decelerates to rest). Move and scale go
      along a straight line at a speed with the same profile.
    - Axes are given in the unit's frame (`Axis::UP`, `FORWARD`, `RIGHT`) and converted to the
      bone's frame through its rest rotation at load (glTF bone frames are arbitrary, unlike
      Recoil's S3O pieces): the bone turns about that axis as seen in the rest pose. Escape
      hatches `Axis::model(v)`, `Axis::bone(v)`. Helpers: `deg`, `turn_by` (relative), `spin`,
      `stop_spin`, `turned` / `spun_up` (WaitForTurn). No heading/pitch helpers (decided with
      the user: on a per-bone basis, and for a tilted unit, they are hard to define sanely; it
      was a big mess in Recoil). A client debug view draws each bone's unit-frame axes.
    - My defaults, not yet confirmed: a turn on a bone a clip was driving starts from that
      bone's current clip value (no pop, as Recoil's Turn starts from the current angle);
      a layer's clip takes back the bones it animates within its mask when it is played, its
      fade-in starting from the manual pose being dropped is not attempted (a pop, for now);
      an additive layer (a recoil kick, breathing) does not evict manual bones and is not
      applied to them.
  - Sketch of the script side (names to settle):

    ```rust
    let anim = unit.anim();
    anim.play(RUN, Layer::BASE).looping().speed_from_ground().mask(LOWER_BODY);
    anim.play(AIM_RIFLE, Layer(1)).mask(UPPER_BODY).blend_in(ms(200));
    anim.turn(SPINE, Axis::UP, a, deg_per_s(180)).accel(deg_per_s2(720));  // SPINE leaves the clips
    anim.turn(GUN_ARM, Axis::RIGHT, b, deg_per_s(120));
    anim.turned(SPINE).await;                 // WaitForTurn
    anim.play(RECOIL, Layer(2)).additive().once();
    ```

  - The order is fixed (clip layers by index, then manual bones), so the result is the same on
    every machine and in the sim's on-demand queries.
- **The pose is a pure function of the channels and the tick**, in strict f32 (the determinism
  rules: nlerp, no FMA, `libm` for the trig of angle targets), evaluated on the CPU:
  - by the sim, **on demand**: a script asking where a joint is (a muzzle, a foot on the ground).
    Clips are kept on the CPU (no GPU baking),
    and the headless sim loads models without textures;
  - by the client, for every unit in view, from the channel columns in the view (same code, same
    result); the GPU gets each joint's local keys and builds the palette (see "Joint keys"
    below). The cull sphere comes from the pose (or the model's reach over its clips and tween
    limits).
- **Coroutines without bytecode: plain Rust `async` scripts** (decided with the user,
  2026-11-16). A suspended coroutine must survive save→load and be the same at any worker count:
  - each unit's future lives in a fixed slot of a host-owned deterministic arena in the shared
    memory (stable address, so `Pin` holds; the slot size is the script's `size_of` future);
  - no heap across an `await`: per-worker arenas are reset each tick and poisoned in dev builds,
    so a violation shows up at once;
  - a save stores the slot arena as a blob next to the columns; it loads only into the same
    module build (the save header already names the module hash). Late join stays a
    fast-forward from the command log.
  - Not chosen: a proc macro compiling scripts into plain-data state enums saved by the schema
    (saves would survive a rebuild, but only a subset of Rust and much more work).
- **Scheduler and parallelism.** A global wake queue keyed by (due tick, entity): timers,
  awaited tweens and clips, callins (events from the sim systems: started or stopped moving,
  aim request, fired, hit, killed). Each tick, woken coroutines run in fixed row ranges on the
  workers (the sim's usual rule, so the result never depends on worker count). A coroutine
  writes only its own unit (its components and channels); everything else goes through commands
  and events, applied in key order at the layer's sync point.
- **Queries.** Recoil let a script read any sim data. Here: its own unit's components directly;
  other entities through a read-only view of the state at the start of the script phase (by
  entity id), the spatial service, and joint positions through the pose function. Random access
  is slower than column scans, so this is for scripts' occasional questions, not per-tick loops
  over many units.
- **Effects** (particles, lights, sound) are events emitted at a joint, carried to the view and
  started at the drawn joint.
- Numbers to hold to (`army-fox`: 10 000 Foxes, 24 joints): 240 000 joint poses per tick on
  the client to evaluate, at worst 240 000 keys of 48 bytes to upload (11.5 MB a tick, about
  350 MB/s at 30 Hz; Fox clips are keyed at 24/s, so a running Fox breaks its lines often and
  saves little), against the M11 GPU pose pass (0.31 ms). To re-measure with step 3. If
  missed: no posing off screen and out of the shadows, distant units every n ticks.
- Aiming is by joint turns and clips only (decided with the user, 2026-11-16); look-at / IK on
  a joint chain is a pie-in-the-sky TODO under Later improvements.
- **Aiming, the script side** (requested by the user, 2026-11-16; the weapons system is designed
  later, so its side here is only what aiming needs). Recoil had `AimFromWeapon` and
  `QueryWeapon` (queries: the bone aim is measured from, and the bone projectiles leave from)
  and the callin `AimWeapon(heading, pitch)`. Heading and pitch go (they are ill-defined per
  bone and on a tilted unit); the rest maps as follows:
  - **Aim-from and fire-from bones are fields the script writes** (per weapon, in a weapon
    component), not callins: the engine reads them with no module call, and a script
    alternates barrels by writing the fire-from field after each shot. The engine finds their
    world positions on demand by the pose function (the unit's transform times the bone's
    model-space matrix at that tick): where the aim direction is measured from, and where
    projectiles and muzzle effects start.
  - **The aim callin gets a world-space unit vector**, from the aim-from bone towards the
    target (the way the gun must point), not angles. It comes again while the target moves
    (Recoil re-called `AimWeapon` every few frames). The script turns bones, awaits them, and
    says the weapon is aimed (Recoil's `AimWeapon` returning true): a weapon fires only then.
  - **Helpers** to get from that vector to a turn:
    - `unit.to_unit(dir)`: world to the unit's frame, by its full orientation (tilt
      included).
    - `anim.dir_in(bone, d)`: a unit-frame direction in `bone`'s own (rest) frame, with the
      bone's parents **where they are going**: a manual parent at its turn's target, a clipped
      one at the clips' value this tick. So after turning a turret, the barrel's direction is
      already against the turned turret, and both joints turn at once. This evaluates the
      pose function on the unit's channels: a host import (pure, so the same on every peer
      and worker).
    - `aim_angle(axis, pointing, d)`: the angle about one bone axis that brings `pointing`
      closest to `d` (both projected onto the plane across the axis; one `atan2`). For a
      turret about its up and a barrel about its right. `turn_toward(bone, pointing, d)`:
      the shortest rotation, for a single free joint (a head looking).
    - `pointing` is the bone's rest forward (the unit's `FORWARD` at rest) unless the model
      names another (the per-model content file, next to the masks).
    - Turns from aiming take the short way from where the bone is (a `short()` option on
      `turn`: the target's sign set by the current rotation), so tracking a target round the
      back does not swing the long way.
  - Sketch:

    ```rust
    async fn aim(unit: &mut Unit, weapon: usize, dir: Vec3) -> bool {
        let d = unit.to_unit(dir);
        let anim = unit.anim();
        let yaw = aim_angle(Axis::UP, Axis::FORWARD, anim.dir_in(TURRET, d));
        anim.turn(TURRET, Axis::UP, yaw, deg_per_s(90)).short().ramp(deg_per_s2(360));
        let up = aim_angle(Axis::RIGHT, Axis::FORWARD, anim.dir_in(BARREL, d));
        anim.turn(BARREL, Axis::RIGHT, up, deg_per_s(60)).short();
        anim.turned(TURRET).await;
        anim.turned(BARREL).await;
        true
    }
    ```

  - Parents that clips move (a spine while running) shift the aim after it is solved; the
    next aim call corrects it. Exact aim keeps the aim chain manual (a turret on a hull).
- No per-joint collision volumes in the sim (decided with the user, 2026-11-16): a skinned mesh
  does not split well into per-joint volumes. Units keep their one volume; the pose's reach
  stays within it (the cull sphere already covers every clip frame), and joint positions are
  for scripts' aiming and effects only.
- Verification: a scripted test unit (walk, stop, a multi-step aim while running: an upper-body
  override plus manual turns on bones the clips leave alone; fire with a flash at the muzzle; die holding the last frame) in a GPU test; save→load→continue with coroutines
  suspended; digests equal at 1/2/4/8 workers; `army-fox` with scripts against the M11 numbers.
  Save/load does not exist yet (M9): the save→load check waits for it.
- Steps: (1) the pose evaluator; (2) channel components and the native ticker, driven by a
  plain game system; (3) the client poses on the CPU from the view, `--clip` goes; (4) a
  coroutine spike (a future per unit, resumed on another worker, digests at 1/2/4/8 workers; done);
  (5) the script API and callins; (6) the death sequence and the test unit.
- **Joint keys** (step 3, decided with the user, 2026-11-16; replaces the per-unit palette
  slots of cf2acf2):
  - The GPU gets **local** joint poses, not matrices: per joint two keys of 48 bytes,
    `(pos.xyz, tick) (rotation quaternion) (scale.xyz, spare)`, each tagged with its tick. Local
    translation, rotation and scale move along lines between keys (nlerp on the rotation);
    globals do not, so the GPU builds the hierarchy and the palette (a compute pass in place of
    the M11 pose pass).
  - **Extrapolation along the line.** The GPU draws a joint at the drawn time on the line
    through its two keys, going past the newer key when needed. Each tick the CPU poses the
    joint and checks it against that line (a render tolerance); only a joint that leaves its
    line is uploaded. Units are drawn one tick behind (between N-1 and N), so the CPU knows N
    before it is drawn: no misprediction pop. On a break at N the keys become (N-1, N), two
    writes, unless the newer key is already N-1 (one write, over the older key): the drawn
    span must lie on the new line.
  - **Spins carry their rate** (axis and angular speed in the key's spare lanes, or a flag):
    two keys a tick apart cannot show more than half a turn a tick, and a constant spin then
    costs no uploads at all.
  - **Clips keep their original keys**: no resampling at 60 fps, so a linear clip between
    keys is on the line (hand-keyed and sparse clips gain most; Fox clips, at 24 keys/s,
    little).
  - **Unchanged units are not posed at all**: the pose is a pure function of the channels, so
    a unit whose `Anim` bytes are the same as at its last pose keeps its keys. Extrapolation
    saves uploads; this saves the CPU posing.
  - Measure first: the CPU cost of 240 000 joints a tick and the writes into host-visible
    memory are likelier limits than PCIe.
  - Step 3 poses every unit the view shows (line of sight). A coarse CPU frustum test comes only
    if the numbers ask for it.
- Progress:
  - Step 1, the pose evaluator — done (2026-11-16). `ae_model::anim`: `Channels` (4 layers of
    28 bytes, 8 tweens of 20; 272 bytes a unit), `Rig` (a model's bones, joints, clips resampled
    at 60 fps in its own arithmetic, named masks), `Rig::advance` (the ticker: fades, phases,
    play modes, ground speed, tweens the short way round) and `Rig::pose`/`globals`/`palette`
    (3x4 rows). Scalar f32 and `libm`, no glam. Blend and override are one kind (a blend on a
    mask is the override); additive adds the change from the clip's first frame. Tween over the
    rest pose or additive, on a bone or a mask by its weights. Tests: the glam path's palettes
    on the test model and on every Fox clip (1e-4 relative), masks, weights, additive, the
    ticker, tweens, repeatable bits. Clips are numbered from 1 (0: an empty layer), so all
    zeros is nothing playing.
  - Step 2, channel columns and the ticker — done (2026-11-16). Engine component `Anim`
    (`ae_abi::engine::ANIM`; `ae_sdk::anim`, feature `anim`): the model's id and the channels
    by field (280 bytes), viewed in line of sight, with a builder (`play(..).per_metre(..)
    .blend_in(ms)`, `stop(layer, ms)`, `turn(..).additive()`, `spin`, `move_to`,
    `tween_done`, `ended`). Name lookups: host plugins publish `(scope, name, value)`
    (`HostPlugin::names`), modules ask with the new import `ae_lookup`; `ae_sdk::anim::{clip,
    bone, mask}` by the model's name. Host system `anim` (`ae_services::anim::AnimPlugin`,
    from `models/*.aemodel` in the content, wired into ae-headless) advances every unit by
    `Rig::advance` with its ground speed. Test game `games/scripted`: Foxes walk and stand,
    a plain system picks the clips and turns the head; test `ae_services/tests/anim.rs`
    (layers, fades, clearing, tweens; digests equal at 1/2/4/8 workers). The ticker runs on
    the coordinating thread (rows are independent: a parallel split is easy if it shows).
    Not yet: masks from content (a mask file per model).
  - Manual joints — done (2026-11-16). Tweens replaced by `ae_model::anim::Manual` (52 bytes:
    bone, kind, flags, value and target as quaternion or xyz, top speed, accel, decel, rate);
    `Channels` 528 bytes, `Anim` 536. Kinds turn, spin, move, scale; one rotation, one move
    and one scale entry per bone. The SDK writes the command (`PENDING`; `FRESH` for a new
    entry); `Rig::advance` turns it into the bone's terms (model-frame axes through the
    bone's rest rotation; a fresh entry starts from the clips' value, on the rest pose's
    side), then steps it (great arc as signed, the trapezoid by accel/decel; a spin stays on
    the rest side). `play` sets `layer_flags::NEW`; the ticker then clears entries (given
    before that tick) on the bones the clip animates within its mask, additive clips
    excepted. SDK: `turn`, `turn_by`, `spin`, `stop_spin`, `move_to`,
    `scale_to`, `release`, `turned` / `moved` / `scaled`, `is_manual`; `.accel`, `.decel`,
    `.ramp`; `Axis::{UP, FORWARD, RIGHT, model, bone}`, `deg`. Test game: the head turned
    while standing, turned back and released when walking. Not yet: the client debug view
    of bone axes; keeping clips' original keys (with step 3).
  - Aiming helpers — done (2026-11-16). `Rig::dir_in` (the direction as a bone sees it, in
    its turns' frame, with the parents where they are going: a pending or moving turn at its
    target, a clipped parent at its clips' value); `manual_flags::SHORT` (`.short()`: the
    target's sign set by the current rotation). Generic pure host queries: `HostPlugin::
    queries`/`query` and the import `ae_query(op, in, in_len, out, out_len)` (`ae_abi::query`);
    the anim plugin answers `anim:dir_in`. SDK: `Anim::dir_in`, `turn_toward` (shortest, for
    one free joint; short), `aim_angle(axis, pointing, dir)`; `libm` under feature `anim`.
    Test game: a standing Fox looks once to its front left through the import. Not yet
    (with step 5 and the weapons): the aim callin with the world direction, `to_unit` (the
    unit's orientation), the aim-from / fire-from fields and their world positions, the
    pointing axis from the per-model content file.
  - Step 3, first cut — done (2026-11-16): correct first, fast next. The client runs the
    animation service with the `--model` models (named by the file's lower-case stem) and the
    content's; after every tick it reads the `Anim` column (`Runtime::each_row`) and poses each
    unit drawn with the model its `Anim` names (`bins/ae-client/src/poser.rs`, `Rig::pose`). The
    stream carries joint keys (`ae_model::anim::Key`: tick as a `u32`; `on_line`,
    `same_side`), two per bone, the unit's pose at the last tick and this one; attribute `pose`
    replaces `clip`/`clip_time`. `units.pose` evaluates the line at the drawn tick (integer
    tick difference), walks the hierarchy, applies the inverse binds; the GPU-baked clips are
    gone, `--clip` is gone. Pass and resource names unchanged (graphs untouched). Bench scenes
    `army-fox`/`near-fox` now run the `scripted` game (decided by me, not yet confirmed: the
    crossing game has no `Anim`, and giving it one would touch every crossing test and
    digest): 10 000 Foxes on a 12 m grid under the army camera (game parameter: grid step and
    corner). Tests: `units_pose_models` against `on_line` + `Rig::palette` (on the newer key
    and past it), `streams_carry_poses`; 27 GPU tests pass, validation clean, no golden
    changed. Measured: `units pose` 0.39 ms (27 bones; 0.31 ms baked), CPU posing 18.8 ms a
    tick on one thread, 540 000 keys (26 MB) a stream; the upload cost is not measured (the
    bench publishes one stream; live, every tick's keys are copied into each frame slot that
    draws it, about 1.5 GB/s of host-visible writes at 30 Hz: commit B's bench needs a fresh
    tick every frame or two). Next: posing in parallel, uploading
    only bones that leave their line (stable key slots per unit, writes carried over skipped
    streams), skipping unchanged units, spins by rate; then clips' original keys. Not yet:
    observer views (no `Anim` in view tables), cull spheres for manual joints.
  - Step 3, joint key lines — done (2026-11-17). Posing in parallel (rayon, by slot). A posed
    unit keeps a slot in a key table (two keys per bone of the largest model) while it stays
    posed; each bone is checked against its line (`ae_model::anim::within`, `Tolerance`: 1e-3
    rad, 1e-3 scale, 1e-3 of the model's reach); off it: keys at its last pose and this one, or
    one write over the older key when the newer is the last tick's; a new unit gets both keys
    at its pose; an unchanged `Anim` is not posed again (still checked). Streams carry writes
    (`units::KeyWrites`); `UnitFeed` keeps the batches until the renderer takes them, each key
    once at its newest value (`take`); a reader that missed writes gets the feed to ask for the
    whole table (`wants_whole`, the poser writes its mirror); `publish_reusing` hands taken
    batches' memory back to the poser. The renderer keeps one
    device-local table, stages each frame's writes per slot and scatters them at the head of
    `units.pose` (`units_keys.comp`, barriers against earlier frames' reads); a frame dropped
    after staging makes the next take ask for the whole table. Bench `live-fox` (a tick every
    second frame, in the warm-up and measured frames only). Tests: `feed_hands_out_key_writes`,
    `streams_carry_poses` (one and two writes, on the line, skipped tick, unchanged, the whole
    table on request, slot reuse), `units_pose_models` (a stale whole table, then writes).
    Measured (`live-fox`): CPU posing 18.8 → 6.5 ms, 154 000 keys a tick (28%, 7.4 MB), the feed
    keeping them 4 µs (sim thread), upload 1.1 ms (render thread), `units pose` 0.43 ms with the
    scatter. Validation (sync included) clean on `live-fox`. Posing is memory bound
    from about four threads (the 26 MB key mirror and last poses read each tick). Next: spins by
    their rate (with a GPU test of keys stored newer first, as one-write breaks leave them);
    clips' original keys (Fox clips resampled at 60 fps break most lines every tick); then
    regenerate the `gpu-*-fox` baselines (and a first `gpu-live-fox`). Not run yet: the
    windowed client with the delta keys (the offscreen `live-fox` only).
  - Hybrid animation, phase 3 (clips on the GPU) — done (2026-11-17;
    `docs/hybrid-animation.md`). It supersedes the joint keys: spins by rate and clips' original
    keys are dropped (the GPU samples the sim's uniform 60/s bake). The client packs each posed
    unit's `Anim` into an animation record (`units::RECORD_CELLS`, 15 16-byte cells: layers'
    clip, mask, mode, kind, weight, phase; shown manual entries' bone, kind, value), two per slot,
    each stamped with its tick; a tick writes one record over the older (its changed cells), or
    none when it matches the newer (stamping the newer the last tick first when it is older).
    `KeyWrites` became `AnimWrites` (16-byte cells), the feed and the scatter
    (`units_anim_writes.comp`) unchanged in kind. The models' animation table carries each rig's
    bake (`Rig::clip_table`), masks and rest transforms (`UnitsSetup::rigs`: the sim's rig per
    class; none: `Rig::new` of the model, which the sim's are while masks are not from content);
    `units_pose.comp` evaluates both records as `Rig::pose` does and lerps them at the drawn tick
    (held at either end). Tests: `units_pose_models` against `Rig::pose` (play modes, a wrap,
    blended, weighted, masked and additive layers, manual turns/moves/scales, a fresh entry,
    newer-first records, opposite hemispheres; checked to fail on two shader mutations),
    `streams_carry_poses`, `feed_hands_out_table_writes`. Measured (`live-fox`): CPU 6.5 →
    1.3–1.8 ms a tick (reading the `Anim` rows and packing; no bone evaluated), 154 000 keys
    (7.4 MB) → 28 500 cells (0.57 MB) a tick, upload 1.1 → 0.12 ms, `units pose` 0.43 → 0.37 ms;
    validation clean. Not done from the proposal (each needs a decision): phase 0's separate
    ticker bench (done 2026-11-18, below), phase 1 (instance columns once a tick), phase 2 (the
    ticker's copies, wrap, parallel rows: done 2026-11-18 but integer phases, which re-pin
    digests), phase 4 (a synced facing for aim and fire,
    a schema change), phase 5 (`Bounds` and the callout, a schema change; pose after the cull),
    phase 6 (analytic phases). Not run yet: the windowed client.
    Then (2026-11-17): the joint key helpers removed from `ae_model::anim`; the renderer poses
    with the sim's rigs (`SimOptions::rigs` makes them for both, so content masks reach the
    GPU); a unit with a manual joint shown is culled with a sphere covering any turn
    (`ae_model::pose::turned_reach`, `models::turned_sphere`; manual moves and scales not
    bounded). Phase 1 measured and not done: each tick's instance columns are copied once per
    frame slot that draws it (twice at 60 fps, 30 Hz), 0.055 ms a copy on the render thread
    (`live-fox`); copying once would save about 0.05 ms a tick, against a device-local table,
    a GPU copy and barrier, and changed graph declarations.
  - Bounds — done (2026-11-17). `Rig::bounds(ch)`: the box of the mesh posed by the channels
    (`Rig::pose`, `globals`, `palette`, then every distinct skinned vertex: its joints'
    matrices blended by weight in joint order, the position through the blend; strict scalar
    f32), in model space. The anim plugin answers `anim:bounds` (input the `Anim`; out min, max;
    zeros for a model it does not have). SDK: component `Bounds { min, max }` (engine id
    `…000F`), `Anim::bounds()`, `Anim::update_bounds(&mut Bounds)`: on demand, synced, no
    per-tick cost; pending commands are not applied (measure after `host(anim)` to see them).
    Test game: units spawn with their first pose's bounds and are measured once a cycle after
    the service. Fox: 2.8 µs a call (1728 vertices). Tests: the arm model's exact boxes (rest,
    a clip, a manual turn, a pending one ignored), the Fox's box against the glam skinning path
    (every clip at three times, a head turn; tight on every face), the query's bytes, the game's
    `Bounds` rows (this tick's pose, the first pose, the 1/2/4/8-worker digests). Not done: the
    catalogue spawner does not fill `Bounds` (no def has an `Anim` yet); the renderer does not
    use `Bounds`.
  - Ticker, pack and cull — done (2026-11-18).
    - Bench `anim` (`ae-tool bench anim`): the ticker's rows (`ae_services::anim::Advance`) on
      seeded channels of a 27-bone model, 10k and 50k units, 1 and N threads; a digest of the
      channels after 90 ticks, equal at both thread counts.
    - Ticker (phase 2 less integer phases): `start_manual` copies only with a command pending,
      `wrap` skips `floorf` in range, idle units skipped, `channels()` built in place, rows over
      the world's workers. Digests unchanged (bit-identical). 10k: 0.885 → 0.653 ms (1 thread),
      0.150 ms (8); 50k: 4.39 → 3.27 ms, 0.65 ms.
    - Pack (`live-fox` `cpu_anim_pack` 1.44 → 0.9–1.1 ms; the 0.3 ms target not met): the stream
      borrows the `Anim` column (`Runtime::chunks`: 0.38 → 0.013 ms); the poser packs blocks of
      512 slots in parallel and keeps its scratch (about 0.6 ms, bound by memory); the rest is
      the stream's other rows. New metrics `cpu_anim_pack_read`, `cpu_anim_pack_poser`. Next
      lever: a smaller `Anim` for the renderer to read (a schema change, not decided).
    - Cull: a unit's manual joints shown are its pose attribute's `POSE_MANUAL` bit (set by the
      poser from the slot's two records, masked off by the pose pass), so the cull no longer
      reads the animation table (it depended on `record_pose`'s barrier). `units cull` 9.7 →
      7.3 µs. `units_pose_models` fails without the bit.
    - The fox baselines re-taken at cdd4196 (`bench-diff` of two later runs: 82 metrics
      unchanged). eaa0fa7's `gpu_total` rise (0.3–0.4 ms) was not the cull's (about 3 µs):
      about half of `live-fox`'s and `near-fox`'s was `pass/water` going from 3–4 to about 180
      µs (the baseline before had `army-fox` at 174 µs already: two modes, about 180 µs in
      every run since); the rest, and all of `army-fox`'s, is not explained. The fox scenes
      include the game's bounds calls (about 67 a tick, 0.19 ms of sim time).
  - Step 4, coroutine spike — done (2026-11-19). The design holds: plain `async fn` scripts,
    resumed on any worker, give the same result at any worker count.
    - SDK `ae_sdk::script` (feature `script`): the module declares a slot arena
      (`script_slots!(slot_bytes, slots)`, data record `script.slots`); the runtime claims it in
      the shared memory, zeroes it for each world and publishes it (`ae_lookup` scope
      `script`). A unit's slot is its entity index (no allocation, the same at any worker
      count). Component `Script { state, polls, due, size, … }` (engine id `…0010`).
    - `script::resume(run, script)` runs in an ordinary parallel system over the `Script` rows
      (the fixed work ranges): a new row's future is written into its slot, a due row is polled
      once (noop waker), a finished one is dropped and its slot zeroed. `Unit` reads and writes
      the polled row (`get`, `set`, `update` with a closure, so no borrow lasts across an
      `await`) through a per-instance poll context; `sleep_ticks`, `next_tick` set the wake
      tick; a `Pending` poll that set none traps (it awaited something not the engine's).
    - No heap across an `await`: `ScriptAlloc`, the module's global allocator, gives a poll
      memory from its instance's bump arena (1 MiB, taken from the guest heap once per
      instance, not host-claimed), reset after every poll (stricter than per tick); freeing or
      growing a block from an earlier poll traps (`heap_survived_an_await`, named in the trap's
      backtrace); feature `script-checks` also poisons the arena on reset (guests always build
      release, so not `debug_assertions`). Outside polls it is the system allocator.
    - Test `ae_sim/tests/coroutines.rs` on `games/coro`: 3750 units (a quarter joining at tick
      40), 200 ticks, `range_rows` 128; each script sleeps 1–9 ticks per round, awaits a nested
      `async fn` tick by tick, uses the heap inside a poll, spawns from a resumed poll, keeps
      locals across awaits. Digests equal at 8/1/2/4 workers, and so is the whole slot arena's
      hash at two ticks mid-run (the suspended futures' bytes, what a save would store: no
      stray padding seen); of the polls, about 40% at 2 workers, 60% at 4 and 70% at 8 ran on
      another instance than the poll before; a future is 152 bytes. A `Vec` kept across an `await` reads back as poison, and
      dropping it traps.
    - Not done, for step 5: the global wake queue (every `Script` row is visited for its wake
      tick; visiting only woken units needs runs over rows across tables, an ABI change, or
      `Awake` tags by commands); callins; waits on the channels (`turned`, clips); `sleep` in
      milliseconds (the tick rate is only the anim service's `tick_seconds` today); one script
      per module (`resume` must get the same function every time; a type check is not done);
      the slot size checked at run time, not from the future's type; a unit despawned
      mid-script leaves its future in the slot (overwritten, never dropped, when the index is
      reused: harmless while futures own no heap); save→load of the arena waits for M9.
    - Slot = entity index makes the arena as large as the highest index times the slot size,
      counting entities with no script (sparks here, projectiles later): 100k entities at 512
      bytes are about 51 MB. A slot allocator (serial, or at the command apply) fixes it.
    - The slot arena's equal hashes are observed, not guaranteed: a new future is copied into
      its slot with whatever its padding held on the making instance's stack. If they ever
      differ, look there first, not at the sim's determinism.
  - Step 5, part 1: callins, races, waits — done (2026-11-19).
    - Several things at once in one unit without more futures: `script::join` (both) and
      `race` (the first; the other is dropped where it waits: Recoil's `Signal` killing a
      thread), no heap. `race(look_around(u), u.callin(STARTED_MOVING))`.
    - Callins: engine component `Callins { pending }` (`…0011`), bits of `script::callin`
      (`STARTED_MOVING`, `STOPPED_MOVING`, `AIM`, `FIRED`, `HIT`, `KILLED`; `GAME` and up the
      game's), posted by whatever system writes the unit's row (the one moving it; events
      such as a hit applied to their target's row by a system, as skirmish's damage is).
      `u.callin(kinds).await` returns the kinds that came and takes them; `Script::waits` (in
      place of `reserved`) holds what the script waits for. A row is polled if its wake tick
      has come or a pending callin is one it waits for; `resume` clears the pending bits it
      sees, so callins nobody waits for are dropped (decided by me, not yet confirmed: state
      to check later belongs in components). A unit waiting only for callins has `due`
      `u64::MAX` and is not polled. Not the global wake queue: rows are still visited.
    - Waits: `u.sleep(ms)` (`ceil(ms × rate / 1000)` ticks); the rate is the new
      `RuntimeConfig::tick_hz` (default 30), published as `ae_abi::sim::TICK_HZ` and in
      `HostInit::tick_hz`; the anim plugin checks its tick length against it. `u.until(f)`
      (checked each tick), and with `anim`: `u.turned(bone)`, `u.moved(bone)`,
      `u.ended(layer)` (a turn or play set in the poll is not done in it). `u.param()`.
    - Step 4 leftovers: a slot starts with a 16-byte header (owner, the future type's tag,
      its drop function), so a new script on a slot drops what a unit despawned mid-script
      left there, and resuming a unit with another future type traps (several script types
      per module are fine, one per unit). The slot size comes from the futures' types:
      `script_slots!(slots, unit, ...)` lists the script functions.
    - Test `ae_services/tests/scripts.rs` on the new game `games/puppet` (Foxes walk and
      stand as in `scripted`, but a script plays the clips and turns the head, posted
      start/stop callins): every posted callin handled; looks finished and cut by walking;
      `sleep(ms)` exact; no turn over in its poll; about 27% of unit-ticks polled; digests
      equal at 8/1/2/4 workers; units despawned mid-script and new ones on their indices
      (254 reused), equal at 8 and 1. Coro's digest changed (the `Script` and slot layouts).
    - Not done then: aiming (below); queries of other units; effects at joints; nothing
      engine-side posts callins yet (nav could post the moves).
  - Step 5, part 2: facing and aiming — done (2026-11-19).
    - **A synced facing** (decided with the user, 2026-11-19: an engine component, synced and
      saved; hybrid phase 4): `Facing { rot }` (`…0012`, `ae_sdk::aim`), a unit quaternion
      from the unit's frame (forward +x, up +y, right +z) to the world's (nav x, up, nav y:
      the renderer's frame, so the identity faces +x as the renderer's heading (1, 0) does).
      A quaternion, not heading and pitch (tilt allowed). `Facing::along(dx, dy)`, `then`
      (a further turn in the unit's frame), `to_unit`, `to_world`; `anim::rotation(axis,
      angle)`. Whoever places the unit writes it (the game; nav does not yet). The renderer
      still draws the heading from the velocity, not from `Facing`: the client's capture is
      one query over all its components, so `Facing` must be read apart and merged by
      entity (as the `Anim` rows are), not added to it. **A prerequisite of step 6** (the
      muzzle flash at the sim's fire-from position is off on a unit whose drawn heading is
      its velocity's when the two differ).
    - `Anim::bone_world(bone, &Body, &Facing)` (query `anim:bone_world`;
      `Rig::bone_frame`): the bone's origin in the world, the model fitted to the unit's
      radius as the renderer draws it (`ae_model::footprint`, now shared with the renderer's
      fit), and the rotation from the unit's frame at rest to the world as the bone carries
      it (`BoneWorld::dir(Axis::FORWARD)`: where it points). Heights are above the unit's
      ground point: the sim has no terrain height (my default, not yet confirmed; the aim
      between units at different heights is off by their height difference until it has).
      Feature `aim` (`["anim", "nav"]`, module `ae_sdk::aim`): opt-in as `nav` is, since
      every component compiled takes an entity id in the module (with them under `anim`,
      `scripted` declared both, shifting its entities and the fox scenes).
    - `Weapons` (`…0013`): per weapon (4), the aim-from and fire-from bones (the script
      writes them), the asked direction and request count (`ask`, with the `AIM` callin) and
      the request the script answered (`set_aimed`; `is_aimed`: it may fire). The weapons
      system stays the game's. `Unit::to_unit(dir)`. GPU tests (27) pass with the shared fit.
    - Test (`scripts.rs`, puppet's `AIM` mode): 1500 standing Foxes, each facing its own
      way, a third tilted, asked every 6 ticks to point the head at a target circling the
      grid (from the head's `bone_world`); the script aims (`to_unit`, `dir_in`,
      `turn_toward`, a newer request cutting the turn short) and answers; 72 929 answers,
      each within 0.00004° of the head's world forward (in game); independently (glam on
      `Rig::globals`, the facing, the fit), units that answered the last request look at the
      target from their head to 1e-5°; digests equal at 8 and 1 workers. Ignoring the facing
      in `bone_world` fails it (180° off).
    - Not done: queries of other units; effects at joints; the renderer drawing `Facing`;
      nav writing `Facing` and posting the move callins.

### M15 — Fully bindless resources (priority: medium)

Where the renderer is today: every pass's descriptor set layout is its shader's set/binding
contract (`pipeline::set_layout`), image views are written into per-pass sets per frame slot,
and the only array is the models' 1024-descriptor partially bound combined image sampler
array (`models::MAX_TEXTURES`, `descriptorBindingPartiallyBound`). Fullscreen programs declare
one uniform block and one combined image sampler per sampled intent. Until now, "bindless"
appeared only in the lift list above.

Target: a resource is addressed by a handle, not a binding slot.
- **Handles.** A renderer-owned slab (`ImageHandle`, `BufferHandle`, generational) issues an
  index into a global array per resource class; the frame graph assigns a handle when it
  creates the resource and keeps it across rebakes. Roles still select transitions and
  barriers; a handle never changes a resource's lifetime.
- **Descriptor layout.** A small fixed set of global sets with variable-count arrays: sampled
  images (2D, 2D array, cube, depth/comparison), samplers, storage images, storage buffers,
  uniform buffers. Requires `runtimeDescriptorArray`,
  `descriptorBindingVariableDescriptorCount`, `descriptorBindingPartiallyBound`, the
  `descriptorBinding*UpdateAfterBind` flags and the `shader*Array(Non)UniformIndexing` flags,
  with `nonuniformEXT` at non-uniform index sites.
- **Shaders.** `layout(set = 0, binding = 0) uniform sampler2D textures[];` with handles from
  push constants, uniforms, draw records and instance attributes. The library passes move
  first; fullscreen programs follow through the render ABI (a bindless program declares which
  arrays and how many entries it indexes), with the module contract version bumped and the
  old per-binding form kept for one version.
- **Descriptor strategy.** Descriptor sets with update-after-bind first;
  `VK_EXT_descriptor_buffer` (not in Vulkan 1.3) is measured as a follow-up only if descriptor
  writes still show on the CPU frame.
- **Limits and fallbacks.** A fixed capacity per class, counted and reported like the light
  cuts; a white/black fallback for retired handles; out-of-range indices are a shader bug,
  checked in tests with GPU-assisted validation.
- **Reflection.** rspirv gains the array declarations and dynamic-index flags; a module that
  indexes an array it did not declare, or with a count it was not given, fails at load.
- **Samplers.** Decide combined image samplers versus separate image and sampler arrays (the
  tileset needs a trilinear REPEAT sampler, shadows a comparison sampler, the models a large
  texture array).

Steps (commit and stop for compaction after each):
1. Device features and the handle table: enable the descriptor-indexing feature set and the
   dynamic-index flags for every stage; report a device that lacks them (today's baseline
   stays). No renderer change yet.
2. Sampled images: the global array and the terrain's textures (palette, tileset, decals,
   shadows, vision, noise) plus the post chain; byte-identical frames; cost measured.
3. Buffers and storage images: lights, particles, units, palettes and the model tables go
   bindless; model draw records carry texture handles; uniform blocks follow.
4. Module contract: bindless arrays in `ae_render_abi`, reflection, the SDK, both modules and
   both graph files; the contract version bumps and the compatibility path is tested.
5. Descriptor buffer evaluation; retire per-pass descriptor writes where it wins.

Verification:
- GPU tests: for every existing scene, the bindless path and the fixed-slot path render
  byte-identical frames (terrain, water, units, particles, decals, models, the post module);
  validation and synchronization validation clean, and GPU-assisted validation clean in a
  test-only run that deliberately indexes past the count.
- A module with more sampled images than any old per-pass layout held (for example 64)
  renders; a retired handle draws the fallback and logs once.
- Resize and topology hot reload keep handles valid (the graph bakes again, the arrays do not
  move); the frame-graph goldens gain handles but not passes or resources.
- `bench-diff`: GPU time within the scene tolerances; the CPU frame records descriptor writes
  before and after, and the shader indexing cost is measured (register pressure and
  non-uniform indexing can cost more than the writes saved).
- Handles are assigned in creation order, never by address or hash iteration, so the bench
  scenes and captures stay seeded.

Not decided: one set or one per resource class; combined versus separate samplers; whether to
require `VK_EXT_descriptor_buffer`; whether modules move to handles immediately or keep
per-sampler bindings for a version; how the editor and the frame-graph visualiser name
handles.

### M16 — RedAlert-graphics survey and port (water first, priority: medium)

`~/RedAlert-graphics` is the C++/Vulkan reconstruction of RA3's look: `src/draw.cpp` holds the
frame's pass order, and `README.md` documents each shader it rebuilds, its parameters and its
cost. The plan already lifted its CBT/LEB terrain and its decal look; M7 ported the water's
surface composition only (no reflection, refraction, underwater pass or displaced surface).
This milestone surveys the prototype and ports the pieces that read well at RTS distances.
**Only water is fixed scope now**; everything else is decided when it is implemented, against
the prototype and the frame budget.

**Committed: the water stack.**
- Pass order, from `src/draw.cpp` (~240-490): wave-particle simulation into a displacement
  texture → churn target → separable blur → adaptive ocean LOD (CBT/LEB, `src/water_lod.cpp`,
  `cbt_*`) → shadow map → planar reflection (geometry only, mirrored across the water plane,
  sky excluded) → opaque scene → refraction copy → underwater pass into the copy (depth tint,
  caustics, submerged-unit treatment) → bubbles into the copy → water surface
  (`ocean.frag`: `refraction + reflection × .5 + environment × fresnel × 2 × shadow`) →
  submarine overlay → spray over the water.
- In our renderer that becomes library passes over new resources: `water.displace` (compute),
  `water.churn`, `water.reflect` (a mirrored terrain + units pass), `water.refract` (a
  scene-colour + depth copy), `water.underwater` (full screen), `water.draw` (replacing
  today's flat quad with a displaced surface), `water.spray` and `water.bubbles`. The frame
  graph's ordered writers and depth roles already have the shape; the pass library, goldens,
  modules and graph files grow with them.
- The surface mesh: today water is one quad at the water level and the depth test decides
  coverage, so there is nothing to displace. Port the demo's adaptive ocean mesh, or start
  with a fixed grid over the map; the demo's error metric (analytic swell + measured
  displacement curvature, `cbt_error.comp`) is the reference. Measure at our cameras and map
  sizes (the demo's ocean square is 2.6 km; our maps are 10 km).
- Displacement: wakes and blast rings as wave particles (`particle.*`, `wave_particle.glsl`,
  `particle_churn.*`, `gpu_particle.*`, `blur.comp`), which produce the height/normal field
  the surface reads. Emitters come from the observer's view stream (hull positions and
  speeds, impact events), so fog of war hides them and nothing enters the sim digest.
- Data: absorption, wind, swell, foam, caustics and the RTS visibility aids become map or game
  settings rather than constants.
- Budget: the demo's own costs are in its README (blur 0.27-0.36 ms, ocean draw, reflection,
  refraction, underwater pass at 1280x720 on an RTX 3060 Ti). Today our whole frame is about
  0.4-0.6 ms for terrain + units at 720p; the port agrees a water budget when the first scene
  runs. `--bench` gains water scenes (overview, army, close, underwater) recorded like the
  others.

**Candidates, decided at implementation time** (the user's list; each entry says what the
prototype does and what our side needs):
- **Reflection, refraction and underwater passes.** Reflection is a real mirrored pass: flip
  the camera across the water plane, draw only terrain/units above it (sky and water excluded),
  break it up with the detail normals, weight it by Fresnel. Refraction is a scene copy
  distorted by the surface normal, with the depth growing the bend and a small blur; 0 A.D.'s
  colour-delta guard rejects samples that would pull above-water geometry down. The underwater
  pass runs after the opaque scene and before the water: it reconstructs world position from
  depth, tints by absorption (RA3's 1D depth LUT), adds baked caustics projected along the sun
  and scaled by the shadow map, and writes the refraction copy. Ours needs a scene copy, a
  depth-sampling full-screen pass and a policy for what the copies may contain (see
  visibility).
- **How to paint terrain and underwater units.** The sea floor gets weed/rock patches, sand
  ripples and fine grain so the refraction does not read as a blur; material tilesets already
  give us the layers, so add underwater layers and caustic lighting to them. Submerged units
  are lit by the sunlight that reaches their depth and tinted by the water above them
  (`SUB_TINT`, with `SUB_KEEP` so paint stays readable and a `SUB_TINT_MAX` cap); RA3 also
  offers a soft outline, a darkened surface silhouette and an after-water overlay
  (`submerged.frag`) as player aids. Decide which aids fit an RTS camera.
- **Underwater particle effects, including bubbles.** The demo's bubbles are spray records
  with negative gravity: clusters of eight few-centimetre bubbles, wobbling as they rise,
  clear in the middle, mirror-rimmed, sun highlight on top, a flat dot under two pixels, lit by
  the light that reaches their depth, drawn into the refraction copy after the underwater pass
  and so bent and softened by the water. Propellers emit them; ships' wakes do not. Blast
  splash and bow spray are separate ballistic clumps with soft-particle fades, a flip-book
  stamped into the displacement/churn targets and whitewater fading over seconds. Our particle
  pool would need a second, depth-aware class (in the copy, not over the water), emitters from
  the view stream, and a decision on how much of the flip-book simulation to port.
- **Water effects like wakes and sprays.** The prototype's wakes: wave particles at stern and
  bow drifting outward at the Kelvin half-angle, amplitude fading with age, churn in a second
  target (normals, not foam colour), a blur that shapes both, displacement limits that ease
  rather than clamp, and blast rings scaled by the crater radius (cavity, jet, rings, spray,
  whitewater) with shallow-water breaking. At RTS distances wakes are a few pixels: decide a
  distance at which they are drawn (near camera only) and a cheaper far path.
- **Units visibility questions for RTS.** Reflection, refraction and the underwater pass can
  each leak a unit the observer cannot see, and a reflection of a ghost would be a lie. Draw
  only the observer's view stream into every copy, and define what a blip, a ghost or an
  explored-only cell reflects (probably nothing). A submerged unit is invisible without
  sonar/line of sight; the visibility aids above are presentation, not simulation. The
  renderer receives submerged/visibility state from the view tables, never from the sim.
- **Underwater view.** The demo clamps its camera above the water, so this is new design: when
  the camera dips, a full-screen water fog/tint, the surface seen from below (Snell's window,
  underside shading, no sky), bubbles and particles in the volume, and a blend at the crossing.
  Decide whether the RTS camera is ever allowed below the surface, and what the game does when
  it is.

**Also in the prototype, not committed** (decide by what the first water scene needs): cloud
shadows, the sky, the decal merge/cloud clusters, its bloom technique (we have our own), the
flip-book particle systems, and the shore travel-time field and caustic bake (we would compute
these in `ae_map` or a bake step, not per frame).

Steps:
1. Survey: run the demo, capture reference frames at RTS-like cameras, and write
   `docs/water.md` with its passes, resources, parameters and costs; decide the candidate
   pieces above.
2. Data and bake: absorption/wind/swell/foam settings, a shore travel-time field and a caustic
   texture (map bake or runtime), so the shaders have the demo's inputs.
3. Reflection and refraction copies: the passes and graph resources, the water sampling them,
   an RTS camera scene and its cost.
4. Underwater pass and unit painting: depth-reconstructed tint and caustics, submerged-unit
   absorption and the chosen visibility aids.
5. Displacement, wakes, spray and bubbles: emitters from the view stream, the surface mesh,
   and the near/far LOD decision.
6. Underwater view and the visibility policy across every water pass, then tuning.

Verification:
- Captures against the demo at the same view (same map, sea floor and camera) for colour,
  reflection, refraction, foam, wakes, spray and bubbles, with the numbers in `docs/water.md`.
- GPU tests: reflection shows only above-water geometry; refraction bends the floor but never
  paints a unit above the water; the underwater tint matches the depth LUT; bubbles land in
  the refraction copy and not in the scene; a submerged unit is absent from an observer's
  reflection, refraction, underwater pass and overlay without line of sight; water switched
  off is byte-identical to today.
- Validation and synchronization validation clean; `bench-diff` with the new water scenes and
  no regressions in the existing ones; no sim digest changes.

Not decided: adaptive ocean mesh (CBT) versus a fixed grid at our map sizes; a real mirrored
reflection versus a cheaper screen-space approximation; which wakes/blasts are view-side
(visual only) and which need a synced water field (submarines, amphibious gameplay); the
visibility policy's details; whether the underwater pass runs always or only near/below the
surface; how the copy layer exposes submerged state to the renderer (a view field or a mask).

## TODO (open, not a priority now)

- **Establish GitLab CI/CD pipelines** (requested 2026-11-22). There is no CI yet: M17's
  pre-commit hook is local, and several notes above wait on it (the `glslc`/SDL3 gating of
  `cargo test --workspace`, clippy on the wasm guests, the cross-platform digest runs). Scope,
  as a `.gitlab-ci.yml`:
  - `fmt` (`cargo fmt --all -- --check`, guest crates included) and `clippy`
    (`cargo clippy --workspace --all-targets -- -D warnings`) on the pinned toolchain
    (`rust-toolchain.toml`), with `target/` cached;
  - `test` on a runner that has `glslc`, SDL3 and a Vulkan driver: `cargo test --workspace`
    and the `ae_sim` digest/sandbox tests;
  - digest and `ae-headless --replay --verify` jobs on Linux x86-64, Windows x86-64 and Linux
    aarch64 (the determinism targets);
  - a GPU runner for `ae_render --test gpu -- --ignored --test-threads=1` with the validation
    layers, which must report zero messages;
  - `ae-client --bench` on one pinned machine (nightly, `AE_BENCH_MACHINE` set) with
    `ae-tool bench-diff` against `bench/results/baseline`, while the deterministic counter
    benches run on every merge request.
  Conditions: a `default-members` split (or checked-in SPIR-V) so the headless runners need no
  `glslc`, SDL3 or GPU; a named machine for the wall-time bench, since those numbers compare
  only within one machine; and the guest crates built through `ae_sim::build::game` so their
  format and clippy checks join the same jobs.

From the M14 hybrid animation work (2026-11-18); each needs a decision before it is done.

- **Animation pack under 0.3 ms.** `live-fox` `cpu_anim_pack` is 0.9–1.1 ms; the poser's
  0.6 ms is bound by memory (2.8 MB of `Anim` fields, 4.8 MB of records a tick). Next lever: a
  smaller animation state for the renderer to read (a trimmed or split `Anim`: a schema change).
- **`Bounds` at creation.** Should the engine guarantee it (filled when a unit with an `Anim`
  is spawned, the catalogue spawner included), rather than each game measuring once?
- **`Bounds` units.** Model space today; metres (scaled by the unit's size) instead?
- **Hybrid animation phases not done** (`docs/hybrid-animation.md`): phase 1 (instance
  columns copied once a tick), the integer phases of phase 2 (re-pin digests), phase 4 (a
  synced facing for aim and fire, a schema change), phase 5's pose after the cull, phase 6
  (analytic phases).
- **`SimOptions::rigs()` bakes every rig twice** (once for the sim, once for the renderer);
  build once and share.
- **The eaa0fa7 `gpu_total` rise** in the fox scenes not explained by `pass/water` (all of
  `army-fox`'s, about half of the others'); `pass/water`'s two modes (3–4 vs about 180 µs)
  not investigated. `cliffs`' `pass/exposure meter` has the same kind of two modes (19 or
  195 µs, fixed for a process's lifetime; 2026-11-22), and so do the fox scenes' water: about
  175 µs that one pass carries or not. A guess to test first: a layout transition or depth
  decompression that the consuming pass's barriers pay (the begin timestamp is written before
  them), depending on where the image's memory landed. A flagged `pass/water` or `cliffs`
  exposure meter is checked against this before it is called a regression.
- **Why scenes slow down inside one bench process** (2026-11-22). After the fox and model
  scenes, later scenes in the same process measured up to 3x slow in some runs, at full clocks
  and with VRAM to spare; the bench now runs one process per scene instead. Conditions for
  going back to one process: find the state that carries over (allocations, descriptor pools,
  the model cache, threads left running) and show two whole one-process runs diffing clean.

From the M14 coroutine spike (2026-11-19):

- **A global wake queue for unit scripts.** `ae_sdk::script::resume` visits every `Script`
  row for its wake tick. Visiting only the woken units needs a run over rows across tables
  (an ABI change). A cheap version: a host-side ordered set of (due tick, entity) that adds an
  `Awake` tag through commands, the resume system querying `(Script, Awake)`; it costs tag
  churn every tick and is not needed while the scan is cheap.

From M14 step 5 (2026-11-19); the rest of M14, moved here, not a priority now:

- **The renderer drawing `Facing`.** The client still draws each unit's heading from its
  velocity. The client reads unit data with one query that requires every component it lists,
  so adding `Facing` to it would drop units that have none. `Facing` needs its own read, merged
  by entity, the way animation data is already read. Step 6 can't start without this: the muzzle
  flash would be drawn in the wrong place on any unit whose facing differs from its movement.
- **Queries of other units.** A script needs to read a component of another entity, such as a
  target's position or `Facing`, that its own system doesn't list. This needs a new host import,
  and aiming at another unit depends on it.
- **Effects at joints.** The sim would emit an event at a bone's world position, such as a
  muzzle flash or particles. The sim side can be done now; drawing it fits with step 6.
- Optional: **nav writing `Facing` and posting the start/stop moving callins.** Right now the
  game does both itself.
- **Step 6: the death sequence and the test unit.**
  - Death: a killed unit stays in the sim, can't be targeted and doesn't collide, until its
    `Killed` script ends (with a time cap). Then it's removed or leaves a corpse.
  - Test unit, in a GPU test: it walks, stops and aims in several steps while running (an
    upper-body clip on top of the running clip, plus script-driven turns on bones the clips
    don't move); it fires with a flash at the muzzle; it dies holding the last frame.
  - Checks: digests equal at 1, 2, 4 and 8 workers, and `army-fox` with scripts measured against
    the M11 numbers.
  - The save → load → continue check with scripts suspended has to wait for M9, since saving
    doesn't exist yet.

From M12 (2026-11-19):

- **Particle half resolution with blend mode 8 — done (2026-11-21).** Mode 8 is
  `ParticleBlend::Packets` (`--particle-blend packets`); it measured 2.5 times cheaper at half
  resolution than at full (274 against 681 µs draw in `bench battle`), but 1.8 times the
  sorted draw at either, so the sorted half-resolution draw stays the default and `Packets`
  stays a comparison blend. Write-up above ("Packets, blend mode 8, and the half/full
  decision").
- **Automatic exposure (eye adaptation), as AAA games do — done (2026-11-21).** A compute pass
  meters the HDR scene's log luminance into a 128-bin histogram (four taps per 4×4 block);
  `exposure.adapt` trims the darkest and brightest 2%, targets the key, eases exponentially in EV
  space (1.5/s toward a brighter scene, 0.5/s toward a darker one, wall-clock time), weights the
  centre of the frame 4× and counts the glow the bloom shows, clamps to 10..18 EV and writes a
  1×1 exposure image the bloom
  and the tonemap sample, with no CPU readback. The scene is stored at the camera's reference EV
  and the tonemap applies the eased EV's scale, so a new EV takes effect in the same frame's
  bloom and tonemap. The bloom's 4.0 clamp is exposure-relative (the scale is applied before the
  clamp). The fog of war's dimming is divided out of the metering. Write-up in
  [lighting.md](lighting.md) ("Automatic exposure"). Not done: spot metering on a selection or
  the cursor, excluding the sky, and per-scene tuning beyond the constants.
- **Roughness-aware sky specular.** `brdfAmbient` is Schlick Fresnel times a hemisphere
  factor, independent of roughness; at a low camera pitch the ground's ambient specular
  approaches the sky's colour (Fresnel rises at grazing) where a roughness-aware directional
  albedo would leave most of it diffuse. Fine at the usual RTS pitch (Fresnel about 0.07);
  fix with the split-sum environment term or an analytic roughness factor when low cameras
  matter.
- **glTF's default metallic 1.0.** A material that omits its pbr block renders as a fully
  rough metal (no diffuse, only the sky's and the lights' specular), as the spec says; content
  authored before M12 step 3 looked different. Decide whether the importer should warn on
  omitted pbr data, or whether generated materials get metallic 0 by convention.
- **glTF `KHR_lights_punctual` conformance** — decided and done with M12 step 2's second part
  (2026-11-20). What the importer does, and where it deliberately differs:
  - Types: point and spot map to our point and cone; **directional** lights are ignored with a
    warning that names the light and its node (one fixed sun; a per-unit directional light
    would swing the scene with the unit's facing, and instancing would give one per unit). This
    knowingly departs from the spec's "must render all three types".
  - Intensity: passed through as candela; the colour is the linear tint. (Superseded the
    earlier `CANDELA_TO_ENGINE` bridge with step 2.5's photometric units.)
  - Range: absent (infinite in the spec) is no authored cutoff (`range = 0`); the renderer
    bounds it by the exposure-derived cull radius. A non-positive range warns and takes the
    same.
  - Radial attenuation: physical inverse square with the spec's quartic window
    (`max(1 − (d/range)^4, 0) / max(d², r_s²)`), the source radius regularising only the near
    field (step 2.5). Departs from the spec only in that regularisation.
  - Spot angles: clamped to [0, π/2] and inner ≤ outer, then our inner/outer cosines.
  - Beam: engine-only (not expressible in the extension; no glTF asset produces one).
  - Check: an authored glTF with point, spot and directional lights in `ae_gltf`'s tests;
    position and direction through the rest palette, the candela mapping, the warning, and the
    model's round trip. The GPU side is `model_lights_light_the_ground` (above).
- **Lights seen in the air (participating media, maybe particles).** A light with nothing
  opaque near it (a flare or tracer in the air, a flash over water or a cliff edge) shows
  nothing today: only surfaces and particles take light. Let a scene define a medium (haze,
  dust, smoke, fog: a density and colour, global or in volumes) that the lights scatter in, so
  they show as glows or shafts in the air. Options to weigh: froxel in-scattering on the light
  clusters' grid (with the sun's shadow map for shafts), an analytic glow per light (the
  integral of its falloff along the view ray, cheap but without occlusion), or emitting
  particles (dust or sparks) for a light to light. Bloom (M12 step 4) covers only the sources
  themselves.
- **Glow for lights with a visible representation (volumetric or textured).** The glow is
  emitter-only, so a clustered light with no geometry (the `--lights` scenes, a glTF
  `KHR_lights_punctual` on a bone) writes nothing, and its ground pool — a lit surface — is
  deliberately not bloomed. Revisit this with "Lights seen in the air" and any textured light
  representation (a billboard or sprite for a lamp, flare, muzzle flash or light shaft):
  - a medium's in-scattered radiance: decide whether it writes the glow, in what units
    (exposed, like the scene) and under which clamp, and how geometry occludes it;
  - an emissive texture/billboard: write it like a model's emissive, depth-tested against the
    scene, with its fade/LOD and camera-facing size, so it cannot bloom through walls;
  - keep lit surfaces out (the contract "a visible source, not a lit surface"); re-check that
    sunlit snow and white sails stay unbloomed;
  - cost at `lights-*` (1 000–4 000 lights): whether many small sources need a sparse or
    quarter-size glow pass, and whether a light's representation blooms through water (the
    water replaces the glow today);
  - test: a lamp billboard or in-scattering adds a halo with bloom on and none with it off,
    while the ground it lights changes by 0.0; plus an occlusion case.

From M13 stage 1 (2026-11-21):

- **Look check: the curve's toe.** The solved Lottes curve has slope `1 / c` ≈ 1.29 at zero,
  which lifts the darks; with the lowered midtones the far ground reads greyer and flatter in
  the stage 1 captures. If it reads muddy on the display, try `contrast` (`a`) above 1, which
  deepens the toe without moving the anchors.
- **Look check in stage 5: HDR darks against SDR.** With `a = 1` the HDR curve's toe is flatter
  than SDR's (slope at zero 1.00 at the reference panel's headroom 1.778 against 1.29), so HDR
  shows the darks about 17% darker (scene 0.1: 0.095 against 0.115) and the key-to-paper-white
  range brighter than SDR. Compare on the display; if
  the darks read crushed, solve `a` per output so the HDR curve also meets the SDR one at a
  dark anchor (scene 0.1 → 0.115), keeping the key and peak anchors.

From M13 stage 3 (2026-11-21):

- **Gamut compression that also bounds the top channel.** The compression lifts negative
  channels only and keeps luminance, so a saturated wide colour can still exceed the output's
  top: BT.2020 red at 1.0 becomes (1.211, 0.000, 0.072) in BT.709, and the SDR target clips
  the red (as does the HDR peak clamp). Reduce chroma until the largest channel also fits, at
  the same hue, and measure the change on saturated emitters.

From the render side (2026-11-22):

- **A screen-space distortion effect** (heat haze over burning ground, the shockwave ring of
  an explosion, a cloaked unit's shimmer, the lensing above a water splash). It displaces where
  the already-rendered scene is sampled instead of drawing geometry, so it is a view-side
  effect like the particles: its targets come from the observer's effects, never enter the sim
  digest, and the fog of war hides them with everything else. Style it after the water's
  missing refraction (`water_ra`, M16, wants the same scene copy) rather than as a second
  mechanism.
  - **Collect the individual targets first, summarise second.** Each live distortion writes a
    target into a per-frame list: a world position and height, a radius (or box), a direction
    field (radial for a shockwave, an upward ripple for heat, along a normal for a cloak), a
    strength, a falloff and a life or noise phase. Then one summary step turns the list into a
    coarse screen-space field — a quarter-resolution texture, or a binning compute pass over
    8x8 or 16x16 pixel tiles — holding one offset vector per texel. The distortion pass reads
    that one sample (bilinear) per pixel, so N overlapping effects cost the same as one and the
    list stays a data structure the benches and the editor can show.
  - **Make overlapping distortions sane, not additive.** Summing the vectors gives a pixel
    twice the offset where two identical effects overlap and lets a crowd smear the image
    without bound. Combine instead as a weight-normalised average, `v = Σ w_i v_i /
    max(1, Σ w_i)`: two identical distortions of one pixel then give the offset of one, and a
    single effect is left untouched. Then clamp `|v|` to a fixed maximum (a few pixels, or a
    fraction of the screen), so no pile-up distorts more than the limit. Every candidate below
    must be order independent — commutative and associative, up to the clamp — so the frame
    does not depend on emitter order, on which worker appended a target, or on a sort:
    - **A. weight-normalised average, clamped** (the default above): smooth, exact for
      identical overlaps, but a strong and a weak effect meeting give a mean rather than the
      strong one;
    - **B. max magnitude, direction from the stronger target**: exactly one effect where they
      coincide, hard creases where two fields of different direction cross;
    - **C. coverage saturate `1 - Π (1 - c_i)`, times one direction** (as the particles'
      optical depth does with `exp(-Σ)`): only the strength saturates and the direction needs a
      separate choice;
    - also try drawing each target's quad into the coarse target with a saturating blend
      (`MAX`, or `1 - (1 - a)(1 - b)`), which gets an order-free combine for free and is
      cheaper than a binning pass.
    - Measure A against B and the quad variant on a dense overlap scene (a dozen shockwaves
      in one spot), for both the combine's sanity (two identical targets versus one) and the
      look at the crossing points; keep the one that reads best and is cheapest.
  - **Where it sits in the frame.** The warp is a uv offset into the scene colour, so it belongs
    between the scene's draws and the bloom and the tonemap (the same place the exposure
    meter sits), and the depth, the vision lookup and the HDR exposure stay unwarped. Decide:
    - whether the bloom is sampled at the same warped uv (the glow then bends with the scene)
      or at the pixel's own uv (the glow stays put);
    - whether the distortion samples the scene only, or the scene and the glow, so a fire
      behind heat haze keeps its bloom;
    - edge handling: clamp the uv and fade the offset to zero at the screen edge, since there
      is nothing outside the frame to pull in;
    - a screen-space field is rebuilt every frame (the camera moves), so do not try to cache
      it; a world-space field would need the depth buffer to reconstruct positions, and that is
      its own design.
  - **Graph and ABI.** Like water and the shadows: a library pass (`distortion.summarise`,
    `distortion.apply`), a library resource or two, `DECLARE_DISTORTION` (or the field declared
    as optional, with the pass recording nothing when no target is live), `when: distortion` in
    graph files, both modules and both `.graph.json` files, new goldens, and the `library`
    checks (a graph with a distortion target feed needs the pass). A module that knows nothing
    about it keeps rendering as today. The UI (M8) and the render module's grade and vignette
    draw after it, so they are never warped; a module's own post chain would need the pass
    inserted, which is why the field is a library pass and not baked into the tonemap.
  - **Cost and tests.** One quarter-resolution target plus the collect/summary pass and a
    fullscreen apply (the tonemap's own 17 µs at 1280x720 is the scale), measured in `battle`
    and a new `heat` bench scene with dozens of live fields against `bench-diff`. Gates:
    distortion off is byte-identical to today; two identical overlapping targets change the
    frame exactly as one does; the clamped maximum is never exceeded; a shockwave over the sky
    does nothing; emitters outside the observer's sight distort nothing; the field is
    deterministic for a given target list.
  - Not decided: whether the target list is filtered by the observer's vision at collect time
    or the field is masked by the vision texture in the apply; whether a target is a screen
    quad (cheap, camera-facing) or a world sphere projected at collect time; whether the water's
    refraction (M16) reuses the same combine and field.

From M13 stage 6 (2026-11-22):

- **The post module's grade in a BT.2020 working space.** Its saturation mixes around Rec. 709's
  luminance row (a shader constant), so it keeps luminance only in the default BT.709 working
  space. Pass the working space's row in the grade's uniform (the module has
  `DeclareEnv::working_space`), and measure a saturated patch's luminance before and after.

From M13 stage 7 (2026-11-22):

- **Bound the top channel in gamut compression.** `colorspace::compress` scales a colour toward
  its luminance until the lowest channel reaches 0 (`t = y / (y - lo)`), so compressing wide
  colour into a narrower output gamut raises the top channel — the further the negative, the
  larger the raise — and it can leave the output gamut's own maximum. Conditions for a fix: cap
  or soft-limit the top channel when the source's was inside the output gamut, and measure a
  saturated P3/BT.2020 patch's per-channel values, luminance and hue against stage 3's
  synthetic patches (`post.encode`'s out-of-gamut test).
- **Look checks waiting on the user's display** (nothing here changes the curve or the grade
  until they report):
  - the HDR curve's toe: with `a = 1`, HDR's slope at zero is 1.00 at headroom 1.778 against
    SDR's 1.29, so scene 0.1 displays at 0.095 instead of 0.115 and the far ground/foliage may
    read crushed or muddy; if so, solve `a` per output so HDR meets SDR at the anchors and
    deepen the toe without moving paper white.
  - the post module's grade above paper white: whether the +12% contrast and +15% saturation
    around 0.18 read right on the display where only the explosion core reaches it, and
    whether the HDR vignette (a multiply in linear light, not the SDR black blend) darkens the
    corners smoothly on a bright sky.
  - HDR darks against SDR and whether the P3 fire reads redder in HDR than in SDR.
  Record any result here as reported, with the display and settings used.
- **Query the compositor's target primaries instead of `--display-gamut`.** SDL does not expose
  them; KWin does over `wp_color_management` (the probe's reference machine reports about 96% of
  P3) or they can come from the EDID in sysfs. Conditions: the queried primaries drive the
  output gamut and the `vkSetHdrMetadataEXT` primaries, `--display-gamut` stays the override,
  and a run reports the queried values beside the chosen ones.
- **Reinspect `--working-space` and `--display-gamut`, and remove both if neither earns its
  keep** (added 2026-11-22). At the defaults neither changes a visible pixel:
  - a BT.709 scene lies inside P3, so `--display-gamut bt2020` compresses nothing, and its only
    other effect, the metadata primaries, is probably ignored by KWin (unchecked);
  - in SDR, `--working-space bt2020` moves sunlit ground by a channel mean of 1.54 codes, max 2
    (stage 3's measurement, taken under the since-removed Lottes curve), and its wider colours
    are compressed straight back into BT.709.

  The case for keeping them is HDR on a wide-gamut panel: saturated lit colour beyond BT.709
  that P3 output can show. Conditions:
  - re-measure without the curve, in SDR, HDR10 and scRGB: a saturated scene (fire, team
    colours, coloured lights on grass) lit in BT.709 against BT.2020, with the share of display
    pixels outside BT.709;
  - check whether KWin uses the HDR metadata primaries;
  - look at the result on the reference panel.

  If no case shows a difference worth the option, remove the switch and keep the fixed choice:
  - the working space stays BT.709;
  - the output gamut follows the encoding (BT.709 for SDR, P3 for HDR10, BT.2020 for scRGB)
    until the queried primaries above replace it.

  Remove the matching render-module ABI fields only if no module reads them. If one case does
  show a difference, keep the option for that case alone and record the numbers that justify
  it.
- **HDR captures.** `--capture` still writes the SDR readback (8-bit sRGB), so a capture shows
  neither the encoding nor the display's levels. Conditions: a capture option selects
  `sdr|hdr10|scrgb` (the readback already supports all three), writes the packed 10-bit or 16F
  pixels (or a PNG beside the raw), and a test decodes a known display value through the chosen
  encoding.

Pie in the sky, some time:

- **`KHR_physics_rigid_bodies` for collision volumes and IK joint limits.** Explore the glTF
  physics extension and decide what of it to take: its shapes (box, sphere, capsule, convex
  hull, compound) as the units' collision volumes, and its joints' limits (hinge ranges, swing
  and twist cones) as the limits a look-at/IK solver must respect ("Pie in the sky, some time:
  look-at / IK for unit scripts", under Later improvements). Today a unit has one volume and
  aiming is by joint turns and clips only. Survey first: what exporters write, how other
  engines read it, and whether its shapes suit the one volume per unit that the nav and spatial
  services read (a hitbox per body part would be a new sim concept, not just a loader change).

## Later improvements

- **Virtual shadow maps** (https://graphics-programming.org/blog/virtual-shadow-maps).
  - What: one very large virtual shadow map; pages are allocated only where the view needs them,
    and cached while nothing in them moves.
  - It replaces the units' cascades (M7) behind the same `shadows.map` / `shadows.draw` contract.
  - The particles' coloured shadows need a colour page for each depth page.
- **Anti-tiling of the terrain's material tilesets** — done (2026-11-16; a tileset degree since
  2026-11-21).
  - After Quílez's "texture repetition", technique 3: a smooth tiling noise texture (one fetch)
    indexes `degree × 8` offset (or rotated) versions of each layer's texture; two blend only near
    the steps, with a variance-preserving blend around the layer's mean. The degree (0..1) is a
    property of the tileset, not a specialization constant: 0 samples each layer once, 1 uses 8
    versions, and the shader always carries the path. Each layer's `directional` flag shifts
    instead of rotating, for textures with a direction (strata, ripples). There is no
    `--no-anti-tiling` switch; a degree-0 tileset is the comparison.
  - From above, the correlation at one repeat (above its baseline) falls 0.56 → 0.02 on mud and
    0.41 → 0.11 on grass at degree 1 (a tenth of the degree leaves 0.21 and 0.25: the knob is
    continuous), with mean and contrast within 1 %. Cost at degree 1, against degree 0: +45 µs
    close, +57 µs overview, +228 µs cliffs (720p; 186.0 → 231.3, 239.6 → 297.0, 503.0 → 730.7).
    The cliffs increase is mostly the rock layer, now anti-tiled as part of the tileset.
  - Not adopted: the first example's constant-overlap triplanar weights and RNM normals.
  - Details, the variants measured and not kept, are in `docs/antitiling.md`.
- **glTF loaded at run time; `.aemodel` removed** (decided 2026-11-16; after M11, before the
  pie-in-the-sky items, not immediately).
  - What: the engine loads `.gltf`/`.glb` from the content as it is; `ae_model`'s file format,
    `ae-tool model convert` and the client's conversion step go. The load does what the
    converter does now (layouts, merged primitives, tangents, joints), in memory.
  - Textures:
    - default: PNG/JPEG decoded at load into RGBA8 with mips;
    - option (a setting): KTX2. At engine start every PNG/JPEG texture not yet converted is
      encoded to a GPU format (BC7) and cached; later loads read the cache. Cache key: a fast
      hash of the source bytes (xxh3-128, already in the engine, rather than SHA-512), plus the
      encoder and its settings, so a new encoder version re-encodes. The cache is local, never
      synced content. The first start with many textures is slow (BC7 encoding); encode on the
      job pool and show progress.
    - KTX2 with Basis (`KHR_texture_basisu`) in a glTF: accepted, so the engine needs a Basis
      transcoder to BC7 (decided 2026-11-16). Candidate: the pure-Rust `basisu` crate (0.1.0,
      a port of the C++ transcoder, upstream v2_1_0, claims bit-exact; very new): compare its BC7
      output with the C++ transcoder's on the test textures. Fallback: the C++ transcoder the
      converter links now (`basisu_c_sys`, transcoder only). Writing our own (UASTC to BC7 is
      the small part, ETC1S the large one) only if both fail.
  - Axes: the synthetic root takes glTF's Y-up to the engine's axes (and any unit scale) at
    load, so nothing downstream converts.
  - Rigid and skinned layouts stay as they are.
  - `KHR_materials_emissive_strength`: folded into the material's emissive factor.
    `KHR_texture_transform`: folded into the UVs at load when every texture of a material that
    reads that UV set has the same transform; otherwise the material keeps the transform (a
    per-texture 2x3 in the material table) or the load warns. To decide when such a model shows
    up.
  - Clips: kept on the CPU only (M14), not baked for the GPU.
  - Costs to measure: load time per model (the Fox reads in 0.08 ms as `.aemodel` today; parsing,
    PNG decoding, mip generation and tangents now happen at every load), and the client's size
    (glTF parsing and image decoders linked in). If load time asks for it, the processed
    geometry can go into the same local cache; the cache is an implementation detail, not a
    content format.
- **Effects follow the sim's speed** (requested 2026-11-16) — done (2026-11-16): with a live
  unit stream, the frame's effect time is the sim time drawn
  (`InstanceStream::sim_time`, `FrameParams::effect_time`); particles step by it (at most 0.1 s
  a frame, so past 6x at 60 fps they lag the sim), decals fade and cool by it (`u.effects.x`);
  water and the render module stay on wall time; readback keeps the deterministic time. Seen
  in the client under validation, not yet checked by eye at other speeds.
  - Particles step by `effect_time` (`particles.rs`: `dt` from `frame.time`), and decals fade,
    cool and expire by the same time, so both speed up, slow down and stop with the sim.
  - Water ripples and the render module stay on wall time at other sim speeds; the Pause key
    holds their clocks too (and skips the paused interval on resume).
- **Pie in the sky, some time: look-at / IK for unit scripts** (TODO, not planned in detail;
  after M14). Aiming a joint chain at a point, feet on uneven ground.
- **Pie in the sky, some time: revisit the froxels** (`ParticleBlend::Froxels`, M7 stretch goal).
  - Seen in play, they look poor: too coarse and too discrete, as if the smoke updated at a lower
    frame rate than the rest of the scene.
  - Likely causes to look at:
    - one centre ray per 4x4 column, so small puffs pop in and out between columns;
    - the bilinear resolve over coarse columns;
    - the exponential slices following the frame's particle depth range, which moves every frame.
  - Directions:
    - several jittered rays per column, with temporal reprojection and accumulation (as Hillaire
      does);
    - a fixed depth distribution instead of one fitted to each frame;
    - a filter better than bilinear between columns (tricubic, or guided by depth).
  - The bar stays the same: no popping, close to the volume reference, and a cost near the sort's.
- **Pie in the sky, some time: zstd-backed archives instead of zip** (`ae_vfs`; the mount/read
  abstraction stays, the container changes).
  - Why: archives are now how a whole game ships, and deflate is the weak link: modest ratios, a
    slow decoder, and every read inflates again. zstd at a high level (packing is offline) should
    shrink `module.wasm` and the JSON defs, and at any level it decompresses several times faster
    than `miniz_oxide`, in a smaller archive held in memory. The KTX2/PNG payloads are already
    compressed and would be stored, so expect the gain in the small-file tail, not across the board.
  - Container: a small native format (`AEPKG` magic + version), a sorted path table (path, method,
    uncompressed size, xxh3-128, offset) in a footer, and one independent zstd frame per file (or
    stored when zstd does not shrink it), so a read decompresses one file and a corrupt entry fails
    only itself. Same path rules and duplicate checks as the zip reader; no zip64 or central-directory
    scan. Large files can be split into independent frames later if range reads are wanted.
  - Unchanged: mounts and override order, `manifest.json`, `content_hash` over the uncompressed
    visible files, `Library::scan`/`resolve`, replay and lockstep headers, `ae-tool pack`/
    `unpack-list`. Because the hash is over files, switching the container keeps every existing
    replay valid; only the committed packages need repacking. Zip stays a read-only mount (found by
    magic), so old packages and third-party tooling keep working, and `ae-tool pack --format zip`
    keeps interop.
  - Determinism: sorted paths, fixed metadata, one thread and a pinned zstd level and version make a
    package's bytes follow from its files again. A zstd upgrade that changes the output is a repack
    diff only, never a content or replay change.
  - Optional: a per-package zstd dictionary (trained on `defs/*.json` and other small files, shipped
    as package metadata, not content) for the small-file tail; long-distance matching for the wasm
    module.
  - Alternatives: ZIP method 93 (zstd inside zip) keeps the familiar container but not the interop
    (tool support varies, `miniz_oxide` has no zstd), so it buys little; tar.zst is trivial but has
    no per-file random access, so mounting decompresses all of it.
  - Gate: a pack/unpack bench (ratio, pack time, cold and warm read time, peak RSS) over the
    skirmish game package, the map package, the Fox model and a texture-heavy library against
    today's deflate; adopt zstd as the default only if the win is clear. "If possible" is the point
    of the item: the format work is small once the numbers ask for it.

## Not yet researched

Recoil's Skirmish AI interface (`RecoilEngine/rts/ExternalAI/`) is outside the functional docs; review before M9.
