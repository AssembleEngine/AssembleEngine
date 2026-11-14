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
  the render heightfield (M3).

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
  RTS controls (WASD/arrows, wheel, right-drag, Q/E, R/F, `+`/`-` detail, T wireframe, L LOD colours,
  G fog, C freeze LOD, P print camera); `--capture FILE.ppm` renders 1280x720 offscreen without a
  window and prints GPU time per pass. The SDL window now reports wheel, drag, toggle keys and held
  movement keys (ecs_sim's F "full frame" key is gone).
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
  These are printed, not recorded: the GPU bench (`ae-client --bench`, JSON for `bench-diff`) is still
  to be written.

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
  (`x, y, prev_x, prev_y, heading_x, heading_y, radius, class`), column by column; whoever feeds it
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
  classes; terrain edits (`TerrainChanged`) still do not reach the render heightfield, so units can
  stand above or inside edited ground. `ae-client` now builds `games/crossing` on first use
  (cargo and the `wasm32-wasip1-threads` target, as `ae-headless`); a sim failure is printed when
  it happens and shown in the title, and `--units 0` runs without the sim.

Next in M3: the render module ABI, reflection and hot reload (the units and terrain passes become
passes a render module declares), then material tilesets (procedural placeholder textures until a
texture source is chosen) replacing the material palette, and `TerrainChanged` patches reaching the
render heightfield. The GPU bench (`ae-client --bench`, JSON for `bench-diff`) is still to be
written.

M2's exit command also runs as `ae-tool navbench --scaling` (forwards to `ae_nav`'s `navbench` binary in
the same target directory).

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
