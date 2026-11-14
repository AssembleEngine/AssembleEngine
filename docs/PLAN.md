# AssemblyEngine — architecture and roadmap

## Context

AssemblyEngine is a new RTS-focused (but flexible) engine, a spiritual successor to Recoil: same extensibility
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

Assumed defaults (change if wrong): tick rate is per-game data, default 30 Hz; Linux + Windows on Vulkan 1.3,
aarch64 as a determinism CI target.

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
  multi-writer versions, indirect/compute buffers, bindless, async compute (vkflecs-baseline as reference).
- Render ABI: `rg_declare(descriptor)` lists resources, passes (roles, attachments, pipeline, draw source), roots;
  host validates and rebakes only when `signature()` changes. One per-frame call returns pass enables,
  uniform blocks, draw-stream params. Shaders are content SPIR-V, reflected with rspirv
  (`vkflecs-baseline/src/shader_reflection.rs`) and checked against declared access.
- Draw sources are data: `InstanceStream{view table, field→attribute map, mesh/material set}`; compute pass
  does culling + LOD → indirect draws. Multi-threaded recording per pass group.
- Engine pass libraries the module instantiates, wires, or replaces: `terrain_cbt` (map-rendering `leb.rs` +
  RedAlert `cbt.glsl`; surface appearance from the material tilesets below), `water_ra` (RedAlert
  `src/draw.cpp:189-610` pass order + shaders), cascaded shadows, fog of war (sensor grid → texture),
  decals, GPU particles, post, UI.
- Models: glTF converted offline; piece animation = sim WASM writing `PieceTransform` columns.
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

| Idea | Where it comes from | Fit for AssemblyEngine |
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
| Later | M7 water + shadows + particles + decals · M8 RmlUi + widget sandbox · M9 AI hosting + save/load + snapshot late-join · M10 tooling, second nav level, async compute, runtime multi-module sim | |

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

Done and tested (`cargo test --release -p ae_sim --test parallel_commands`):
- `ae_ecs::arena`: deterministic column allocator over a host-claimed range of the shared memory.
- `ae_wasm`: deterministic wasmtime config, shared memory, page claims, N instances with private
  stacks and TLS (instance 0 owns the heap).
- `ae_sim`: work ranges cut in table order, per-worker command buffers keyed `{system, range, seq}`,
  merge by key, serial apply with direct-to-table spawns, and the `ae_spawn`/`ae_despawn`/`ae_add`/
  `ae_remove` imports. 100k entities, 30 ticks of a system that despawns, spawns, adds and removes
  components from parallel ranges: identical digest at 1, 2, 4 and 8 workers and equal to a native
  reference. Merging by worker instead of key gives a different digest on every run (negative control).

Next: scheduler with read/write sets and layers (from `ecs_sim/src/engine/schedule.rs`), schema custom
section and SDK derive, events as columnar streams, cadences, the order-independent digest from
`hash.rs`, `ae-headless`, a sim-dispatch bench suite, and the 10k/50k-entity exit runs.

### M2 progress

Started before M1 finished, on request: the services are built and tested standalone, with columnar
APIs, and get wired into the sim schedule once the M1 scheduler exists.

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
  the render heightfield (M3).

Next for M2: run the service from a sim system once the M1 scheduler lands (gather/scatter by entity
through work ranges, a `MoveGoal` event stream instead of `CommandPayload`), movement classes from game
data instead of `water_profiles()`, cost overlays, formation facing, and the spatial grid built from
sim columns each tick.

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

## Not yet researched

Recoil's Skirmish AI interface (`RecoilEngine/rts/ExternalAI/`) is outside the functional docs; review before M9.
