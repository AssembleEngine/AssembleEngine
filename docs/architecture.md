# AssembleEngine — architecture, drawn

Five readings, each a level deeper. Start at the top, stop where the question is answered.

| Level | Answers | Section |
|---|---|---|
| L0 | What are the moving parts, and which of them own truth? | [The engine as five layers](#l0--the-engine-as-five-layers) |
| L1 | What runs on what thread, and when? | [Units of execution](#l1--units-of-execution) |
| L2 | What happens inside one tick / one frame? | [The tick](#l2--one-tick), [one frame](#l2--one-frame) |
| L2 | How does each system work? | [Per-system](#l2--per-system) |
| L3 | Which crate sits on which? | [Crate map](#l3--crate-map) |

The prose plan is in [PLAN.md](PLAN.md); measured numbers are in
[../spikes/README.md](../spikes/README.md) and the progress sections of PLAN.md. This file only
describes structure.

---

## L0 — the engine as five layers

```
┌──────────────────────────────────────────────────────────────────────────┐
│ CONTENT        .aemap · defs/*.json · game.wasm · render.wasm            │
│ (hashed)       material tilesets · shaders · decals · replays            │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ ae_vfs: mounts, dependency order, hash
┌───────────────────────────────────▾──────────────────────────────────────┐
│ SESSION        ae-client · ae-headless · ae-relay                        │
│                one world per game, N observers, N players                │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ TickBundle { tick, commands }
┌───────────────────────────────────▾──────────────────────────────────────┐
│ SIM            ae_sim::Runtime — the only authoritative state            │
│ (30 Hz)        layered schedule · WASM systems ∥ native host services    │
│                deferred commands · typed events · xxh3 digest            │
│                module and host systems read/write the same columns       │
└───────────────────────────────────┼──────────────────────────────────────┘
                                    │ end-of-tick capture:
                                    │ fields the schema marks
                                    │ with a view class
┌───────────────────────────────────▾──────────────────────────────────────┐
│ COPY           ae_services::ViewPipeline — per observer, filtered        │
│ (per tick)     ae_sensors coverage → ae_view tables → instance stream    │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ instance streams, camera, UI data
┌───────────────────────────────────▾──────────────────────────────────────┐
│ RENDER         ae_framegraph + ae_render — Vulkan 1.3, own thread        │
│ (~free-run)    render module declares topology; host bakes and executes  │
└──────────────────────────────────────────────────────────────────────────┘
```

Three rules explain most of the code:

1. **One world is authoritative.** Only the sim's columns are hashed, saved and replayed. Views,
   render state, UI and AI are derived and may be stale, filtered or absent.
2. **Guest code never sees flecs.** Game logic is WASM over `ae_abi`; the ECS lives behind
   `ae_ecs`. Native services are host systems in the same schedule, ordered by the same read/write
   declarations.
3. **Everything observable is derived from the schema.** The schema section in the module bytes
   drives the digest layout, saves, net codecs, catalogue validation, view classes and editor hints.
   There are no hard-coded component names in the engine.

---

## L1 — units of execution

| Thread | Owns | Never does |
|---|---|---|
| main (SDL3) | window, input, present timing, camera | touch world state |
| sim coordinator | one tick: prepare, dispatch, merge, apply, digest | wait for the renderer |
| crew (N, persistent) | one wasmtime store/instance each, over one shared memory | touch rows outside its work range |
| host service pools | nav field jobs, steering, avoidance, physics (behind `ae_nav`) | change results by thread count |
| render thread | frame graph bake, recording, submit | wait for a tick |
| sim thread (client) | `Runtime::tick_with` at game rate | block the window |
| ground feed | landed terrain changes → renderer patches | affect the digest |
| net IO | packets, resend, digest reports | pace the sim itself (the relay does) |
| watch (dev) | rebuild render module/shader on file change | swap a broken one in |

```
 sim thread     [ tick t ]   [ tick t+1 ]   [ tick t+2 ]   ──────▸ 30 Hz, fixed step
                     │            │              │
                     └────────────┴──────────────┴──▸ capture ─▸ view build ──┐
                                                                              │
 render thread  ◂─────────────────────────────────────────────────────────────┘  free-running frame loop
                     each frame: read the last stream, lerp previous → latest
```

Nothing in the render path can stall a tick: the renderer reads the last published capture and
interpolates between previous and latest positions when it runs ahead or behind.

---

## L2 — one tick

```
tick_with(bundle)
  │
  ├─ 1  bundle check: every command must be a declared command event with the
  │     exact size; key = { system: u32::MAX, range: player, seq }
  │     bad ones are dropped and counted — identically on every peer
  │
  ├─ 2  for each schedule layer ◂───────────────────────────────────────────┐
  │      │                                                                  │
  │      ├─ prepare()                                                       │
  │      │    ├─ query → WorkRange[]  (row chunks of `range_rows`)          │
  │      │    ├─ copy unseen events for this layer into guest staging       │
  │      │    └─ jobs: Parallel → one job per range                         │
  │      │            Serial   → one job for all ranges                     │
  │      │            rows < parallel_min_rows → one job anyway             │
  │      │                                                                  │
  │      ├─ run:   crew.run(batch)  ∥  run_hosts(jobs)                      │
  │      │         N instances, private stacks + TLS, one shared memory     │
  │      │         hosts touch only rows no module system of the layer      │
  │      │         writes — so the overlap is sound, not lucky              │
  │      │                                                                  │
  │      ├─ merge: commands sorted by { system, range, seq }                │
  │      │         events sorted by key, then stream; a duplicate key is    │
  │      │         a hard error (it would mean a range ran twice)           │
  │      │                                                                  │
  │      ├─ apply: structural commands, serially, in merged order           │
  │      │                                                                  │
  │      └─ events: append to the store; readers mark their seen offset ────┘
  │
  ├─ 3  events.prune(); tick += 1
  │
  └─ 4  stats: fuel per system, host ns, prepare/apply ns, command and event
        counts — the deterministic counters benches compare
```

**Why layers exist.** `ae_sim::schedule::compute` turns each system's declared reads and writes
into conflict edges, then greedily packs systems into layers. Consecutive systems in one layer
never conflict, so they can run at once. Order *inside* a layer is still fixed (`position`), because
that is what keys the commands and events; the parallel speedup never changes the bytes.

A real schedule, from `games/crossing`:

```
schedule!(host(nav_spawn), orders, quake, reorder, zones, host(nav),
          host(spatial), crowding, churn, landed)
```

`orders` and `reorder` both read `Goal` and both emit `MoveOrder` — and emitting a stream counts as a
write, so they are deliberately kept in separate layers; their output is still ordered by the merge
key. `quake` emits a `TerrainEdit` that `landed` consumes after the nav has repaired it. `crowding`
is `staggered(4)` and `after(spatial)`, so it is row-filtered by entity index and sees this tick's
index. `host(nav)` and `host(spatial)` sit in the list like any other system; only their body is Rust
instead of WASM.

**What makes it deterministic.** Fixed row ranges independent of worker count; no clock, random or
thread-count input to a system; merged command order by key; events in key order; native jobs
installed at a deterministic due tick in job-id order; f32 under strict rules (no FMA,
transcendentals via `libm`); NaN canonicalised.

### Lower: the state a tick touches

```
ae_sim::World
  ├ world      flecs world; component columns in the guest's shared linear memory
  ├ queries    per system: flecs query + stable term ids
  ├ applier    structural commands (spawn/despawn/add/remove), serial
  ├ events     EventStore: typed streams, per-reader seen marks, pruned per tick
  ├ ids        IdMap: sim EntityId(u64) ↔ flecs entity, guest-visible ids are stable
  ├ spatial    SpatialIndex, replaced only in the spatial system's own layer
  ├ hosts      Box<dyn HostSystem> per scheduled host system
  └ tick, param
```

```
column arena (ae_ecs::arena)
  guest linear memory
  ├─ host-claimed range: every component column flecs allocates
  │    power-of-two classes, 16-byte header, deterministic free lists
  ├─ instance 0: guest heap (malloc), __wasm_call_ctors
  └─ instances 1..N: private shadow stack + TLS block, allocated by instance 0
```

The arena is why guest systems can write columns in place: flecs keeps its own state on the host
heap, but its columns land in memory the guest can address. One arena per process, installed before
the first world.

---

## L2 — one frame

```
 render module (WASM)                       host (ae_render)
 ────────────────────                       ────────────────
 rg_declare(desc) ────▸ ae_render_abi ────▸ module.rs
                                            validate against the schema
                                            reflect SPIR-V, check bindings
                                            │
                                            ▼  once, or when the source changes
                                            ae_framegraph
                                             declare → bake → barriers
                                             rebake only when signature() changes
                                            │
                                            ▼  every frame
                                            executor
                                             module.frame(FrameInput) gives pass enables,
                                             uniforms and draw-stream params
                                             one command buffer per pass, recorded on N threads
                                            │
                                            ▼
                                            frames in flight → submit → present
                                            │
                                            ▼
                                              Vulkan 1.3 GPU
```

The split is the point: **topology is declared once** (or when it changes), **data is per frame**.
A graph can come from a render module (`--render-module post`), a data file
(`render/post.graph.json`), or engine code, and can be swapped while running.

### The pass library

Render modules wire these; the engine owns their implementations.

| Pass | Kind | Reads | Writes |
|---|---|---|---|
| `terrain.lod` | compute | heightfield, error pyramid, camera | `terrain.leaves`, `terrain.indirect` |
| `terrain.draw` | graphics | leaves, indirect, tilesets | colour 1, depth (write) |
| `units.reset` | transfer | — | `units.indirect` (reset) |
| `units.cull` | compute | `units.instances`, camera | `units.visible`, `units.indirect` |
| `units.draw` | graphics | instances, visible, indirect | colour 1, depth (write) |
| `units.count` | transfer | `units.indirect` | `units.counts` |
| `shadows.draw` | graphics | instances (all units) | `shadows.map` (depth) |
| `shadows.particles` | graphics | particle state, keys, indirect | `shadows.particles` (colour) |
| `water.draw` | graphics | depth (read), sky, fog | colour 1 |
| `particles.sim` | compute | — | state, keys, indirect |
| `particles.draw` | graphics | state, keys, indirect, depth | colour (half-res + upsample) |
| `post.tonemap` | graphics | the scene, the bloom, the exposure image | the target (SDR, mapped from the working space to BT.709) or the display (HDR, in the working space) |
| `post.encode` | graphics | the display (HDR only) | the target: the output gamut compressed if narrower, then PQ in BT.2020 or scRGB |
| `readback.copy` | transfer | an image | `readback.buffer` |
| `host.read` / `present` | transfer / present | — | — |

### The output chain

An SDR graph ends at the target: `post.tonemap` maps the scene (and bloom, exposure) from the
working space to BT.709 and writes the 8-bit sRGB image. An HDR graph adds the display
intermediate and the encode:

```
scene (RGBA16F, scene light) ─▸ post.tonemap ─▸ display (RGBA16F, linear light in units of
                                               paper white, in the working space, up to
                                               peak / paper white)
                                                  │
                                   module passes that change colour on the display
                                   (post's grade, its multiplying vignette; the UI will
                                   write it too)
                                                  │
                                                  ▼
                                            post.encode ─▸ target (A2B10G10R10 PQ in BT.2020,
                                                            or RGBA16F scRGB)
```

Only `post.encode` maps the display to the target: it compresses out-of-gamut colour toward the
achromatic axis when the output gamut is narrower than the working space, re-expresses it in the
container's primaries, and stops values at the display's peak before writing PQ or scRGB. The
levels (`DisplayLevels {paper_white, peak}`) are signal nits, relative to the reference the
swapchain's image description carries (203 on Wayland), and the client applies them to the
`vkSetHdrMetadataEXT` call after every recreate.

### The render module ABI (contract version 2)

`ae_render_declare` reads a `DeclareInput` the host writes at `ae_render_declare_input`: the
target's size/format and scene flags, and the output's display spec — the encoding (`Sdr`,
`Hdr10Pq`, `ScRgb`), the levels, the working space and the output gamut. A module built for
contract 1 is refused at load by number. The display rule follows from the encoding: an HDR
module declares `display` (`display_spec()`), has `post.tonemap` write it, runs its colour
passes on it and ends in `post.encode`; an SDR module has neither and lets `post.tonemap` write
the target. `library::check_graph` enforces that on module graphs and names the encoding and the
missing or extra pass. A module stays loaded across a swapchain pair change and redeclares for
the new display spec.

### Draw sources are data

```
 ViewTable column ──field→attribute map──▸ InstanceStream ──▸ upload
                                                │
                                    units.cull on the GPU
                                                │
                                    indirect draw, LOD by distance
```

No per-unit host work per frame: the capture publishes a column, the upload copies it, the compute
pass culls and writes the indirect arguments. Transforms arrive as previous + latest and are
interpolated in the shader.

### Render-side subsystems

```
terrain (ae_render::terrain)
  heightmap ─▸ CBT/LEB subdivision ─▸ leaf buffer ─▸ draw
  error pyramid ── patched in place on GroundUpdate (no rebake)
  material tilesets: layers × (albedo, normal), projection by orientation,
  two-level blend: geometry normals for the broad transitions,
  normal-map detail to break up the boundary

units (ae_render::units)
  instance stream · GPU cull · indirect draw · interpolation

shadows (ae_render::shadow)
  heightfield → exact terrain shadow at any distance
  units → cascaded shadow map (virtual shadow map later)
  particles → coloured optical depth

particles / decals
  GPU simulation + back-to-front sort, half-res composite
  decals sample the terrain height and wrap over cliff edges
```

---

## L2 — per-system

### Sim core (`ae_sim`)

| Module | Role |
|---|---|
| `runtime` | loads the module, builds the world, runs the tick loop |
| `schema` | parses the schema section; component and field records |
| `schedule` | conflict edges → layers; cadences; host declarations |
| `crew`, `work` | worker instances, work ranges, job batches |
| `command` | deferred structural commands, keyed and merged |
| `events` | typed columnar streams, a cursor per reader, seen events dropped at tick end |
| `host` | `HostPlugin` / `HostSystem` traits, `HostCtx` |
| `imports` | the guest's import surface (queries, columns, emit, spawn, spatial) |
| `digest` | xxh3-128 per row, merged order-independently |
| `capture` | end-of-tick field capture by name (view, render, ground feed) |
| `input` | `TickBundle`, the same type the net and replays carry |

Cadences: `EveryTick`, `Every { n, phase }`, `Staggered { n }` (row filter by
`entity_index % n`), `Once`. Cadence never changes results, only which ticks a system sees.

### ECS seam (`ae_ecs`)

```
declarations the engine uses          implementation
────────────────────────────          ──────────────
declare_component  ────────────────▸  flecs_backend (vendored 4.1.6 + 2 patches)
declare_query                         · columns via ae_ecs::arena
chunks / for_each_chunk               · entity ids in the arena too (no per-chunk copy)
apply_commands                        · flecs types never leave this module
snapshot / restore
EntityId(u64) · ComponentId(u64)      guest + ABI never see flecs
```

The seam is deliberately narrow so a custom ECS can replace flecs later without touching `ae_sim`,
the SDK or any game.

### WASM boundary (`ae_wasm`, `ae_abi`, `sdk/`)

```
 module.wasm
 ├─ code         wasm32-wasip1-threads
 ├─ memory       imported env.memory  ← one SharedMemory for all instances
 └─ custom section "ae_schema"
      KIND_COMPONENT  id, size, align, flags, fields{id, offset, type, flags}
      KIND_SYSTEM     mode, cadence, terms{id, access}, emits, reads, after
      KIND_ORDER      system order (the linker's order is not trusted)
      KIND_DATA       named data for services, e.g. nav.classes
```

Host side: one `Store`/instance per worker, deterministic wasmtime config (NaN canonicalisation on,
relaxed SIMD off, no clock/random imports), per-instance stack and TLS, trusted modules unmetered in
release (watchdog instead), untrusted modules always fuelled and capped.

Guest side (`ae-sdk`): `#[derive(Component)]`, `#[system(read(..), write(..), emits(..))]`,
`schedule!(.., host(..), ..)`, typed column access, `Spawn`/`despawn`/`add`/`remove`, `emit`,
`spatial_radius`, and engine components in `ae_sdk::nav` (feature `nav`).

### Native services

They are host systems: declared like a module system, ordered by the same conflict rules, merged
into the same command and event streams, and hashed as their own digest part.

**Navigation (`ae_nav` / `ae_services::nav`)**

```
 MoveOrder · OrderPoint · TerrainEdit · CostOverlay events
                     │
                     ▼
 ┌────────────────────────────────────────────────────────────────────┐
 │ ae_nav::NavService                                                 │
 │  commands ─▸ cohorts ─▸ stages   (abstract route: sectors/portals) │
 │      │                                                             │
 │      ├─ jobs      flow fields per (sector, region); prefetch ahead │
 │      ├─          terrain edits queued, repaired on the due tick    │
 │      ├─ contexts  destination contexts for stages                  │
 │      ├─ steer     field lookup + look-ahead walk + slot assignment │
 │      ├─ avoidance ORCA against k nearest                           │
 │      ├─ physics   integrate + collision response                   │
 │      ├─ progress  arrive / settle / give up → NavAgent.mode        │
 │      └─ statuses  CommandStatus updates                            │
 └──────┬─────────────────────────────────────────────────────────────┘
        │ Body / NavAgent rows, read and written in place
        ▼
   Body.x/y/vx/vy · NavAgent.mode · TerrainChanged · service digest part
```

Movement classes (declared by the game with `movement_classes!`) decide passability, speed and
depth handling: land, ship, hovercraft, amphibious, fording. Terrain edits bump a version; the
service repairs paths, moves buried units out, and publishes `TerrainChanged` plus landed ground
rectangles for the renderer.

**Spatial index (`ae_spatial` / `ae_services::spatial`)**

```
 every Body ──▸ uniform grid, rebuilt each run
                   │
                   ├─ radius(circle, n)    ─┐
                   ├─ boxes(rect, n)        ├─▸ hit streams to guest systems
                   ├─ nearest(x, y, k)      │   (table order, row order)
                   └─ sweep(segment)       ─┘
```

Items are in query order, so hits are a function of world state only. A system that wants this
tick's positions declares `after(spatial)`.

**Sensors and the copy layer (`ae_sensors`, `ae_view`, `ae_services::view`)**

```
 sim columns (end of tick)
        │ CaptureSpec: only fields with a view class in the schema
        ▼
 ┌──────────────┐  Body.x/y · Team.ally · Sensor.los/radar
 │ capture      │──────────────────▸ ae_sensors::Coverage
 └──────┬───────┘                    per-allyteam LOS / radar grids
        │ rows + coverage + ally
        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ ae_view::Observer::build                                         │
 │   each field: a ViewClass                                        │
 │     Never · Always · OwnerOnly · Los · RadarDegraded · Ghost     │
 │   each row: a RowState                                           │
 │     Own · Seen · Blip · Ghost                                    │
 │   shows(class, state) decides the field                          │
 │   RadarDegraded: wobble = hash(tick, view id, observer)          │
 │   rows keep a stable ViewId; sim ids stay hidden                 │
 └──────┬───────────────────────────────────────────────────────────┘
        │ ViewTable (typed columns)
        ▼
   instance stream: x, y, prev_x, prev_y  ──▸ lerped in the shader
```

**Catalogue spawning (`ae_services::catalog`)**

```
 SpawnRequest events ──▸ catalog_spawn (host, every tick)
                             │ ae_catalog::Catalog
                             │   def → components + field values
                             ▼
                        one entity per request, in event order
                        Body.x/y = request position, Team, Def
```

A missing def is skipped and counted, identically on every peer: the catalogue is part of the content
hash, so "missing" cannot differ between machines.

### Content (`ae_vfs`, `ae_catalog`, `ae_map`, `ae_model`)

```
 archive.zip / directory
        │
        ▼
 ae_vfs::Vfs — stack of mounts, later overrides earlier, file by file
        │
        ├─ manifest.json ─▸ Library: name, version, kind, depends
        │                   resolves a mount order, dependencies first
        ├─ content hash ──▸ session handshake, replay header, desync checks
        │
        ├─ defs/*.json ───▸ ae_catalog: extends / abstract, field by field
        │                   → component bytes, validated against the schema
        ├─ maps/*.aemap ──▸ ae_map: map.json (size, seed, water, types, starts)
        │                   height.png (16-bit) + types.png → nav terrain
        └─ models/*.aemodel ▸ ae_model: bones, joints, primitives, KTX2, clips
```

### Net, replay, bots (`ae_net`, `ae_replay`, `ae_bot`)

```
 player input ──▸ client ──Packet::Client──▸ relay ──Packet::Relay──▸ every client
                    │                        │
                    │                        ├─ closes one TickBundle{tick, commands}
                    │                        └─ compares reported digests
                    ▼
              tick_with(bundle)        idempotent packets: loss needs no handshake
                                       a late input lands in a later bundle
```

```
 .aerep replay
 ├─ Header      engine version · module hash · schema hash · param · host config
 ├─ bundles     only ticks that had commands (the same type the net carries)
 ├─ checkpoints digest every N ticks, with per-component and per-service parts
 └─ end tick
```

Bots live outside the sim and emit normal tick commands, so replays record them and peers do not run
them to stay in sync.

### Benchmarks and profiling (`ae_bench`, `ae_profile`)

```
 micro       cargo bench -p <crate>        one kernel
 subsystem   ae-tool bench <name>          one crate, synthetic world, no GPU
 scenario    ae-headless --bench           whole tick, recorded replays, worker curve
 GPU         ae-client --bench             per-pass GPU timestamps
                    │
                    ▼
 bench/results/<label>/*.json  ──▸ ae-tool bench-diff ──▸ exit 1 over threshold
 deterministic counters (fuel, fields built, bytes copied) gate every PR;
 wall time only on one pinned machine
```

`ae_profile` is thin macros over Tracy, compiled to nothing without the `tracy` feature: tick
phases, layers, systems, row-range jobs, service stages, capture, bake, per-pass record, submit.

---

## L3 — crate map

Each group is a stack: an entry depends on the entries after it in the same group, and on the
rows below.

```
 BINARIES
   ae-client            window + sim thread + render thread + ground feed
   ae-headless          sim only: digests, timings, replay verify, bots
   ae-relay             lockstep pacing and digest comparison, no world
   ae-tool              bench, bench-diff, navbench, pack, catalog-fmt, map-export

 BENCH
   ae_bench             drives each subsystem standalone; writes diffable records

 SESSION / SERVICES
   ae_services          nav_spawn · nav · spatial · catalog_spawn · view pipeline
   ae_bot               SkirmishBot, commands through the normal input path
   ae_net  ae_replay    relay-paced lockstep; the command log and checkpoints

 CONTENT
   ae_vfs               layered mounts, manifests, dependency order, content hash
   ae_catalog  ae_map    defs as data → component bytes; .aemap ↔ nav terrain
   ae_model  ae_gltf    the one skinned model format; the offline glTF converter

 WORLDS
   ae_sim               runtime, schedule, crew, commands, events, digest, capture
   ae_view              view classes, observer tables, radar wobble
   ae_sensors           per-allyteam coverage grids

 NATIVE SERVICES
   ae_nav               terrain, sectors/portals, flow fields, ORCA, collision
   ae_spatial           per-tick uniform grid, batch queries as hit streams

 RENDER
   ae_render            Vulkan executor, render thread, terrain, units, effects
   ae_framegraph        declare / bake / barriers / dot
   ae_render_abi        wire format, module contract, program table

 FOUNDATION
   ae_ecs               seam + column arena           (vendored flecs 4.1.6)
   ae_wasm              wasmtime instances, shared memory, determinism
   ae_abi               schema section, imports       (no deps, builds for WASM)
   ae_profile           Tracy zones and plots; nothing without the feature

 GUESTS (not workspace members; built for wasm by the tools)
   sdk/ae-sdk + ae-sdk-macros      sim modules
   sdk/ae-render-sdk               render modules
   games/sandbox  games/crossing  games/skirmish  games/alloc   render/default  render/post
```

Two rules keep this from tangling:

- `ae_sim` does not depend on `ae_services`. Services depend on `ae_sim` and plug in through
  `ae_sim::host::HostPlugin`.
- `ae_render` does not depend on `ae_sim` or `ae_view`. The client wires them together, so the
  renderer can run with no sim at all (`--units 0`, `--map demo`, GPU tests).

---

## Traces — follow three features down

**A move order, input to pixels**

```
 input / bot / net  ─▸ TickBundle{tick, [MoveOrder bytes]}
   ae_sim::runtime::tick_with   validate against the schema → event stream
   game system `orders`         reads Goal, emits MoveOrder (guest, WASM)
   layer merge                  key {system, range, seq}
   host(nav)                    ae_nav::NavService consumes MoveOrder as a group command
                                cohorts → stage → flow field → steer → ORCA → physics
   Body.x/y/vx/vy, NavAgent.mode   written in place, in the guest's columns
   end of tick                  capture Body.x/y/vx/vy + radius + profile
   ae_services::view            coverage → per-observer ViewTable + wobble
   ae-client                    upload instance stream (x/y, prev_x/prev_y)
   units.cull (GPU)             frustum cull, indirect args
   units.draw                   interpolate, draw
```

**A terrain edit, sim to shadows**

```
 game `quake` emits TerrainEdit (guest)
   host(nav)          queue edit; repair sectors on the due tick; move buried units out
   TerrainChanged     emitted, consumed by game `landed` next layer
   GroundFeed         publish { version, changed rect, shared terrain }
   ground thread      patch height texture + error pyramid in place
   render             terrain leaves, terrain shadows, water depth, decals re-sample
```

**A new unit from a definition**

```
 content/defs/tank.json ──extends──▸ ae_catalog ──▸ component bytes
   game emits SpawnRequest{def, x, y, ally}
   host(catalog_spawn)  one entity, in event order, Team + Def + def's components
   host(nav)            NavAgent gets a profile, steering, slots
   host(spatial)        appears in the index next build
   digest               new row hashed; replay and peers agree
```

---

## Reading the code

| You want to change… | Start at |
|---|---|
| the tick order or scheduler | `crates/ae_sim/src/schedule.rs`, `runtime.rs` |
| what a guest can call | `crates/ae_sim/src/imports.rs`, `crates/ae_abi/src/lib.rs` |
| what the SDK exposes | `sdk/ae-sdk/src/lib.rs`, `sdk/ae-sdk-macros/src/lib.rs` |
| a native service | `crates/ae_services/src/*.rs` + its crate |
| what is hashed / saved | `crates/ae_sim/src/digest.rs`, `schema.rs` |
| what an observer may see | `crates/ae_view/src/lib.rs`, `ae_services/src/view.rs` |
| a render pass | `crates/ae_render/src/library.rs`, `executor.rs` |
| how the output reaches the display (colour space, HDR) | `crates/ae_render/src/swapchain.rs`, `display.rs`, `library.rs` |
| the working space, the gamut map and its compression | `crates/ae_render/src/colorspace.rs`, `encode.rs` |
| the exposure and bloom tail (`post.tonemap`, no curve), the display intermediate and the encode | `crates/ae_render/src/tonemap.rs`, `encode.rs`, `library.rs` |
| HDR levels, the swapchain's colour space and its metadata | `crates/ae_render/src/swapchain.rs`, `display.rs`, `bins/ae-client/src/main.rs` |
| a particle effect's look, or the gamut an authored colour is in | `assets/effects.json` (parsed by `crates/ae_render/src/particles.rs`), `colorspace.rs` |
| graph validation | `crates/ae_framegraph/src/{descriptor,bake,barriers}.rs` |
| the render module contract | `crates/ae_render_abi/src/`, `ae_render/src/module.rs` |
| content packaging | `crates/ae_vfs/src/lib.rs`, `ae_catalog/src/lib.rs` |

## Not built yet

Named here so the diagrams are not read as promises: RmlUi UI and the widget sandbox (M8),
`ae-editor` with the inspector/archetype/schedule views, the second navigation hierarchy level,
async compute and runtime multi-module sims, querying the
compositor's colour-management target primaries (an own `wp_color_management` client on SDL's
`wl_display`, or the EDID from sysfs; the display gamut is a setting for now), and the per-worker
bump arenas that would remove wasi-libc's heap lock from allocation-heavy guest systems.

M11 (glTF models and skinning) is just starting: the model format (`ae_model`) and the offline
converter (`ae_gltf`) exist, and nothing draws one yet.
