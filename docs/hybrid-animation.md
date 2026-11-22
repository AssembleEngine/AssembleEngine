# Hybrid animation — GPU clip playback, CPU aim/fire, on-demand bounds

Status: design, 2026-11-17. Phase 3 (GPU clip evaluation, `182ecd4`) and the bounds callout
(`b612a7e`, §8; the render cull still uses its static sphere) are implemented; phases 0–2 and
4–6 are not (see `docs/PLAN.md`, M14 progress). Numbers marked **measured** come from
`docs/PLAN.md` / `docs/models.md` and the commits named below; numbers marked **estimate** are
arithmetic from those, to be replaced by a bench before the design is frozen.

## 1. Summary of the proposal

The pose is no longer computed for every unit every tick on the CPU. It becomes a **derived,
presentation-only quantity** that the GPU evaluates from a compact per-unit animation state, while
the **inputs and the gameplay-relevant outputs stay synced and CPU-side**:

| Kept on the CPU (synced, hashed, saved) | Moved to the GPU (derived, drawn) |
|---|---|
| Clip/layer selection, masks, weights, fades, phase bookkeeping | Per-bone clip sampling, layer blending, masks, additive layers, manual-bone application, hierarchy, skinning palette |
| Manual joint channels (aim, deploy, recoil commands) and their stepping | Applying those manual values to the pose |
| Aim solving, fire decisions, fire-from/muzzle world positions | Anything about the drawn pose |
| Bounding-volume updates, on request | Cull-sphere decisions, shadow/draw transforms |

Three properties make this safe:

- **H1 — Semantic sync.** Every value that can influence synced state is produced by CPU code
  from synced inputs with the engine's strict f32 rules. The GPU never produces sim-visible data.
- **H2 — One clip source.** CPU queries and the GPU sample the *same* baked clip table (the CPU
  bake that `Rig::new` already makes). The GPU does not bake or sample glTF keys itself, so a CPU
  joint query and the drawn pose agree before any optional quantization.
- **H3 — Per-unit, not per-bone.** No per-bone data crosses the bus. A compact per-unit record
  is uploaded per tick, and each tick's upload happens once regardless of frame count.

The result is not a loss of synchronization; it is **deferred pose evaluation**. The CPU can
reproduce any drawn pose on demand (chain queries, bounds), so gameplay remains exact and
deterministic, while the per-tick cost becomes O(units that changed), not O(units × bones).

## 2. Where the cost is today

### 2.1 Measured baseline

From `docs/models.md` (M14 step 3, 10 000 Foxes, 24 bones, RTX 3060 Ti, 24 threads):

| Item | Value | Notes |
|---|---|---|
| CPU posing | **6.5 ms per stream** (5.9–6.5 across runs) | `bins/ae-client/src/poser.rs`; the whole visible army, every tick |
| Joint keys written | **154 000 of 540 000** (7.4 MB) per tick | 48 B per key, 2 keys per bone |
| Render-thread upload | **1.1 ms** per tick | `UnitFeed::take` + staging into mapped memory |
| GPU `units pose` pass | **0.43 ms** per frame | includes the key scatter |
| Key table (GPU) | ~26 MB | 540 000 × 48 B |
| Key mirror (CPU) | ~26 MB | `Poser::table` |
| Staging per frame slot | ~28 MB × slots | host-visible |
| Palette (GPU) | ~23 MB | unchanged by this design |

The improvements in the three commits are real: parallel posing took 18.8 → ~10 ms; key lines,
stable slots and skipped unchanged units took it to 6.5 ms; the feed fix took the publish cost from
3.4 ms to 4 µs. But they do not change the shape: **every joint is still evaluated on the CPU, and
every changed bone still sends 48 bytes over the bus**.

### 2.2 The scaling arithmetic

Per unit per tick, at 10 000 units and 30 Hz:

| Cost | Today | Per unit | At 50 000 |
|---|---|---|---|
| CPU pose | 6.5 ms | ~650 ns | **~32.5 ms/tick** (infeasible at 30 Hz) |
| Joint keys | 7.4 MB | ~740 B | **37 MB/tick (1.1 GB/s)** |
| CPU mirror | ~26 MB resident | ~2.6 KB | ~130 MB |
| GPU pose | 0.43 ms | ~43 ns | ~2.2 ms |

This is the "doesn't scale" the request names. Tuning the key encoding (16-bit quats, bigger line
tolerance) might buy 2–3×; it cannot remove the O(bones) CPU evaluation or the per-bone traffic.
The structural fix is to stop sending bone transforms and stop computing them on the CPU.

### 2.3 Sim-side costs, and two things the profile does not show yet

The synced animation component is `ae_sdk::anim::Anim`: **536 bytes** per unit
(`model` + 4 layers × 28 B + 8 manual entries × 52 B). It is hashed, saved and synced by default.
`ae_services::anim::Ticker` runs as one serial host system and does a read-modify-write of that
whole component for every unit every tick:

- 536 B × 10 000 = **5.36 MB per tick of read-modify-write**, ~160 MB/s of memory traffic at 30 Hz
  before any arithmetic; 50 000 units is 26.8 MB/tick.
- `Rig::advance` copies the entire `Channels` once per unit per tick even when no manual command is
  pending (`crates/ae_model/src/anim.rs:458`, `let before = *ch;`): another **5.3 MB/tick** at
  10 000 units.
- The ticker loops all 4 layers and all 8 manual entries per unit, and `wrap()` calls
  `libm::floorf` per playing layer per tick, on one thread.
- The ticker has **not been benchmarked separately**; it is hidden inside the sim tick today. This
  design treats it as a first-class target and adds a bench for it.

`Anim` is also hashed in full: 5.36 MB of digest input per tick at 10 000 units. Trimming the
record is therefore worth doing, but it is secondary to removing the pose and key traffic.

## 3. The relaxation: what "synced animation" now means

The original plan mandated fully synced animation. The relaxed rule is a three-tier split, not a
blanket "animation is view-side":

| Tier | Examples | Where | Determinism |
|---|---|---|---|
| **Synced state** | active clips and layers, masks, weights, targets, fades, phases, manual joint values/targets, model id | sim columns | strict f32, hashed, saved, digest-checked |
| **Synced outputs** | aim solution, `aimed`/`ended`/turn-complete events, fire tick, fire-from and muzzle world positions, effective bounding volume | sim, computed on demand | strict f32, hashed where persisted |
| **Derived presentation** | per-bone local/global transforms, skinning palette, clip sampling, blend results, interpolation between ticks, cull sphere use | GPU | not hashed, not saved, may differ by a few ULP per GPU |

Rules that follow:

1. A script may read *any* joint of its unit at any tick on the CPU. That is the escape hatch that
   makes the GPU path an optimization rather than a semantic change.
2. The sim never reads a value that only the GPU has. If a future feature needs the pose, it calls
   the CPU query; if that becomes hot, the query is optimized (chain-only, cached), not moved to
   the GPU.
3. The GPU may hold a different pose from the CPU's exact one only within a stated visual
   tolerance (Section 10). With H2 that tolerance is zero until quantization is introduced.
4. Discrete animation state changes (clip switched, manual entry started/ended, `ended` flag) happen
   at integer tick boundaries on both sides. Smooth interpolation is only for continuous values
   (phase, weight, manual value).

## 4. Target architecture

```
 sim tick (30 Hz)                    client sim thread                     render thread / GPU
 ────────────────                    ──────────────────                    ───────────────────
 Anim / Manuals (synced)  ─capture─▶ packer ──record deltas──▶ UnitFeed ──▶ per-unit state upload
   │                                    │                                      (once per tick)
   │ on demand                          │                                         │
   ├─▶ aim / fire / joint queries       │                                         ▼
   ├─▶ bounds callout (full pose)       │                                  units.pose (every frame)
   └─▶ ticker: phases, fades, manuals   │                                  reads state + clip table
                                        │                                         │
                                        │                                         ▼
                                        └── view/LOS rows only ──────────▶ cull, shadows, draws
```

Responsibilities:

| Stage | Runs on | Per-tick cost | Per-frame cost |
|---|---|---|---|
| Ticker (channels) | sim, parallel rows | O(units with active entries), cheap integer phase | — |
| Packer | client sim thread | compare compact records, build deltas | — |
| Transport | render thread | one upload per tick, ring/scatter | — |
| Pose | GPU compute | — | O(posed units × bones) |
| Cull / shadows / draws | GPU | — | as today |

The packer replaces `Poser` (`bins/ae-client/src/poser.rs`) and the key table. `UnitFeed` keeps its
role (latest stream + unread deltas + "reader missed writes → ask for a whole state"), but the
payload is per-unit animation records, not per-bone keys.

## 5. Sim-side design

### 5.1 Channels remain authoritative

The ticker keeps doing exactly what `Rig::advance` does today semantically: fades, phase advance and
wrap, play modes, `ended`, manual turn/spin/move/scale stepping, clip-take-back. It writes the same
`Anim` fields, so save/load, replay, digests and the script API are unchanged in meaning.

What changes is cost and layout:

1. **No whole-`Channels` copy per unit.** `start_manual` copies 528 B before it knows whether any
   entry is pending. Check for pending entries first; copy only then. (One-line shape fix, big
   memory win.)
2. **No `libm::floorf` in the hot path.** A phase in `[0, period)` can be wrapped with a compare
   and one subtraction (or with integer sub-tick units); `wrap` is called for every playing layer
   every tick.
3. **Skip inactive work.** Track per unit whether any layer is active and whether any manual entry
   is active. A unit with neither costs one flag read. The common case is one looping locomotion
   layer.
4. **Parallel rows.** Make the ticker split row ranges over the job pool like other native
   services. The result is row-local, so it is worker-count independent.
5. **Integer phase (recommended).** Store phase in sub-tick units (e.g. 1/4096 clip cycle, or
   clip frames at the bake rate) and the rate in the same units per tick. Advancement is an
   integer add and wrap; no drift, no libm, and a cheap digest. The GPU receives the same integer
   units and converts once per frame.
6. **Trim the component (optional, second step).** Split the 8-manual-entry tail (416 B) into an
   optional `Manuals` component present only on units that use manual joints, and shrink the common
   part. This cuts the sim's RMW, digest and save cost for the majority of units. It changes the
   schema and thus re-pins game digests (Section 15), so it is a separate, deliberate step.

Target (estimate, to be confirmed by a new `ae-tool bench anim`): < 0.5 ms/tick at 10 000 units,
< 2 ms/tick at 50 000, single-digit MB/s of memory traffic.

### 5.2 What the ticker must *not* do

It must not compute a pose. In particular:

- It does not call `Rig::pose`, `Rig::globals` or `Rig::palette` for ordinary units.
- It does not compute bounds unless the script asked.
- It does not iterate bones of an inactive unit.

### 5.3 On-demand CPU queries (aiming, firing, any joint)

The existing `Rig::dir_in` is already chain-only (`crates/ae_model/src/anim.rs:511`): it walks the
bone's ancestors and evaluates only those bones' clip/manual values. That is the right shape. The
design adds:

- `Rig::bone_world(ch, bone) -> Affine` (or a position + direction form) using the same chain walk,
  returning a **model-space** transform. The unit's placement turns it into world space.
- `Rig::bone_position(ch, bone, placement) -> [f32; 3]`, the fire-from/muzzle query.
- A per-unit, per-tick **query cache** in the sim: the first query in a tick evaluates the needed
  chain(s); later queries for the same unit reuse the result. This bounds the cost when a unit has
  several weapons or several effects asking for the same joint.
- A documented cadence rule: aim solves and joint queries happen when the target changes or on the
  script's cadence, not unconditionally every tick for every unit. The engine cannot enforce it
  without changing semantics, so the bench measures it and a counter reports query count per tick.

Cost (estimate): 0.3–1 µs for a 3–8 bone chain. 10 000 units all querying every tick would be
3–10 ms/tick, so the design budget is "units with a reason to query", with the per-tick cache and
cadence; the benchmark gate covers a realistic 2 000 aiming units at a 4-tick cadence.

**Prerequisite: a synced unit orientation.** Aiming and firing need the unit's frame, not just its
position. `Body` has `x, y, vx, vy, radius`; the renderer derives heading from the stream, but the
sim has no heading when a unit stands still. The design needs one of:

- add `heading` (and, if tilt matters, `pitch`) to `Body`, or
- add a `Facing { heading, pitch }` engine component, synced and view-classified.

`to_unit(dir)` (the M14 aim API) then has a defined frame. Ground height for the muzzle comes from
the authoritative terrain service (`terrain:height(x, y)` query or an existing accessor), not from
the renderer. This is a prerequisite for the "firing stays on the CPU" requirement, independent of
the GPU pose question.

### 5.4 Firing

The weapon system asks for the fire-from bone's world transform at the fire tick, takes the
direction from the solved aim (deterministic CPU value), spawns the projectile there, and emits a
view event carrying that world position. The visual muzzle flash therefore does not need to read
the GPU pose; it is placed at the exact sim position, and any sub-frame pose difference is invisible.

## 6. GPU presentation path

### 6.1 Clip table: upload the CPU bake

`Rig::new` already bakes every clip into a fixed-rate table (`Baked`: tracks, frames, rate, keys)
using the strict scalar path. The renderer uploads a GPU form of that table once per model:

```
ModelClips {
    clip_base: u32,
    masks: [ { first, count } ],       // weight per bone, u16
    clips: [ { duration, frames, rate, first_track, track_count, key_base } ],
    tracks: [ bone id ],
    keys:   [ Trs ],                   // pos, quat, scale; optionally 16-bit quantized
}
```

This is a GPU *copy* of the CPU bake, not a second bake. It is built lazily for models actually
drawn, from the same `Arc<Rig>` the animation service uses. Fox: 3 clips × 24 bones × ~60 frames ×
48 B ≈ 0.2 MB (estimate); quantization halves that if memory matters. Total for a content set is
tens of MB worst case, uploaded once at load, not per tick.

The glTF/source clips stay on the CPU for the sim and queries; nothing about the engine format
changes.

### 6.2 Per-unit state record

One persistent record per posed unit, indexed by the existing stable slot concept (the `pose`
attribute already carries a slot offset; it becomes the animation-state slot). Sketch, ~64 B
before manuals:

```rust
struct AnimRecord {
    tick: u32,                  // when this record was written
    model: u16,                 // model index (validated against the class)
    flags: u8,                  // active layer mask, "whole reset", ...
    manual: u8,                 // active manual count
    manual_base: u32,           // into the manual array
    layers: [LayerRecord; 4],   // clip, mask, mode/kind/flags, weight, phase
    bounds: [f32; 4],           // optional per-unit bound (Section 8)
}

struct LayerRecord {            // 12 B
    clip: u16,                  // global clip id, 0 = inactive
    mask: u8,                   // mask index (0 = whole model)
    flags: u8,                  // play mode, kind, ended, active
    phase: u16,                 // cycle fraction, e.g. 1/65536
    weight: u16,                // 0..65535
    // prev values or analytic anchors, depending on the variant below
}
```

The record above is the *logical device state*. Transport is **column-major**, like the instance
stream: the packer emits only the columns that changed (phase, weight, discrete fields, manual
block), and the GPU state is updated in place. Uploading the whole AoS record per unit is the
simple fallback, but costs ~64 B/unit/tick (~0.64 MB at 10 000) and should be avoided.

Two transport variants, chosen by measurement:

**A. Per-tick values (first implementation, exact).** Upload phase and weight for every changed
unit every tick; the GPU interpolates between the previous record and the current one at the drawn
`tick - 1 + alpha` (the same rule positions use). Discrete fields are taken from the current
record; continuous ones are lerped. Manual values likewise carry `prev`/`current` with their ticks
(a unit-level key line, but only for manual bones).

- Traffic (columnar): 4 layers × (2 + 2) B of continuously changing values, plus discrete fields
  on change. With one active layer: **~4 B/unit/tick**; with all four: ~16 B/unit/tick. The AoS
  fallback is ~64 B/unit/tick.
- CPU pack: compare 40–64 B/unit against the last sent record; ~0.05–0.1 ms at 10 000 units.
- This is simple, exact, and already 10–50× less than the key path.

**B. Analytic anchors (large-count variant).** The record also carries, per layer, `start_tick`,
`phase0`, `rate` (cycle fraction per tick), and for fades `weight0`, `target`, `fade`, `fade_tick`.
The GPU computes phase/weight at the drawn tick directly. For `GROUND_SPEED` layers the GPU derives
the rate from the position delta already in the instance stream (world units per tick × clip
seconds per metre); the sim remains authoritative for `ended` and queries. Then a steady marching
army uploads **nothing per tick**; only clip switches, fades, manual values and bounds move.

Variant B is the end state if 50 000-unit animation traffic still matters after A. It adds GPU
duplication of the wrap/fade formulas, which are small and presentation-only. It does not change
the semantic split.

### 6.3 Transport: one upload per tick, not per frame

Today the instance columns and the key staging are per frame slot, so a 30 Hz tick drawn at 60 fps
can upload the same data twice. The design:

- Keep a **persistent device-local state table** (as the key table is today) and apply per-tick
  deltas with a small scatter at the head of `units.pose`, exactly like `units_keys.comp` does for
  keys. The table survives frames; only changed records are staged.
- Stage deltas in per-slot host-visible buffers (as today) but only the compact records; no 28 MB
  per-slot buffers. If the renderer misses deltas (dropped frame, first take), keep the existing
  `wants_whole`/`reset` mechanism and send the whole table once.
- Upload each tick once: if two frame slots present the same tick, the second uses the table the
  first already applied. This is a property of the persistent table, not of the staging.
- Alternatively, a ring of per-tick state buffers with per-slot descriptor binding. This removes
  the scatter pass and in-flight hazards, at the cost of a ring and dynamic binding. The scattered
  persistent table is closer to today's proven code and is the recommendation; the ring is the
  fallback if the scatter's barriers become a problem.
- Merge the animation delta copy with the instance-column copy where possible (one staging buffer,
  one copy command), so a tick is one transport operation rather than two.
- Do not upload the clip table per tick; it is static per model.
- Do not upload the palette; it is device-local and written by `units.pose` (unchanged).

### 6.4 The pose pass

`units_pose.comp` is extended from "two keys per bone" to "evaluate one unit's rig from the state
record and the clip table":

1. Load the `AnimRecord` for the row's slot (or use the rest palette when `NO_POSE`).
2. Start every bone at its rest TRS.
3. For each active layer (4): get the clip's frames at the drawn phase (loop / ping-pong / once),
   then for each track bone apply `layer_onto` semantics: blend lerp by `weight × mask`, or
   additive delta from frame 0. This mirrors `Rig::pose` exactly.
4. Apply manual entries: for each active entry, set the bone to rest and override rotation,
   translation or scale (and lerp the value between the last two ticks for smoothness).
5. Walk the hierarchy (parents before children, as today), then multiply each joint by its inverse
   bind into the palette range.

The pass still runs once per unit per frame (not once per shadow cascade), because the palette is
per row and shadows read the same palette. Workgroup layout: one invocation per unit, as today;
24–60 bones is a comfortable per-invocation loop.

**GPU cost estimate:** the current key-line pose is 43 ns/unit. Full clip evaluation adds clip key
fetches (2 frames × ~48 B per bone per layer) and blending. Estimate 60–120 ns/unit, i.e. **0.6–1.2
ms at 10 000 units per frame** for 24 bones and 1–2 layers. This is the number to measure first;
Section 6.5 lists the levers if it exceeds budget.

### 6.5 Ordering, culling and LOD

The graph currently declares `units pose` **before** `units reset`/`units cull`
(`crates/ae_render/src/terrain/app.rs:999–1013`), so every row is posed even if it is invisible and
outside every shadow cascade.

The design reorders to: reset → cull → pose → shadows → draws, and makes the pose dispatch cover
the union of:

- rows visible to the camera, and
- rows in any shadow cascade's caster list.

The cull already writes per-class regions for the camera and each cascade; it adds a per-row
"needed anywhere this frame" flag (or appends row indices once to a pose list). This removes posing
for off-screen units that cast no shadow. In a typical RTS most of the army is in a nested cascade,
so the saving is modest; it matters for wide maps and for units behind the camera.

Levers if the pose pass is still too expensive (in order):

1. **Distance cadence.** Far units (below a pixel-size threshold) re-pose every 2nd or 4th frame
   and keep their last palette; the parameter interpolation does not need to run. Deterministic per
   frame and purely visual. The cull already computes screen size for culling.
2. **Layer LOD.** Drop additive/cosmetic layers beyond a distance; the semantic layer stack is
   unchanged for near units.
3. **Pose on the tick, interpolate palettes in the vertex shader.** Two palettes per unit and a
   per-vertex blend; trades pose ALU for vertex bandwidth. Not recommended first: vertex fetch
   doubles and shadows would need the same interpolation.

The existing palette memory (23 MB at 10 000 Foxes) is unchanged.

## 7. Host→device traffic budget

At 10 000 units, 30 Hz ticks, 60 fps frames:

| Stream | Today | Proposed (A) | Proposed (B) |
|---|---|---|---|
| Unit position/heading columns (9 × 4 B) | 360 KB/tick, up to ×2 per slot | 360 KB/tick, once | 360 KB/tick, once |
| Joint keys | **7.4 MB/tick** | **0** | **0** |
| Animation records | — | ~40–160 KB/tick columnar (1–4 active layers); ~0.64 MB AoS fallback | ~0 steady; changes only |
| Manual entries | — | ~10–60 KB/tick (sparse) | ~10–60 KB/tick |
| Rows/index buffer | ~120 KB/tick | ~40 KB/tick | ~40 KB/tick |
| Clip tables | — | once at load | once at load |
| **Total per tick** | **~7.8 MB (233 MB/s; up to 466 with duplication)** | **~0.46–0.6 MB (14–18 MB/s)** | **~0.4 MB, mostly positions (12 MB/s)** |

Resident memory:

| Item | Today | Proposed |
|---|---|---|
| CPU key mirror | 23–26 MB | 0.4–1.5 MB (last records) |
| CPU key table (poser) | 26 MB | — |
| Per-slot staging | 28 MB × slots | < 1 MB × slots |
| GPU key table | 26 MB | — |
| GPU animation state | — | 0.4–1.5 MB |
| GPU clip table | — | ~0.2 MB per model |
| GPU palette | 23 MB | 23 MB |

The first implementation (A) already removes ~93% of the per-tick bytes and all per-bone traffic.
Variant B removes the rest for steady-state clips. Both are far below the 7.4 MB/tick of joint keys.

Additional traffic rules:

- Upload only for units in the observer's view/LOS set (the packer runs on the view path; hidden
  units animate in the sim but are not drawn or sent). This also means an enemy entering line of
  sight sends a whole record once.
- Quantize only where measured: phase and weight fit u16; manual quaternions can be i16 later.
  Do not quantize the CPU-side state (queries use exact floats).
- Use 16-byte-aligned staging offsets, as the terrain uploader already does.
- Keep a counter `anim_bytes_uploaded` and `anim_records_uploaded` in the renderer stats and the
  bench JSON; traffic is a first-class regression metric, not a side note.

## 8. Bounding volumes and the `update_bounds` callout

There are three different volumes in the engine and the design must not conflate them:

1. **Movement/collision radius** — `Body.radius`, used by nav and spatial. Gameplay, already synced.
2. **Pose/selection volume** — how large the unit currently is (a deployed artillery piece, a unit
   with legs extended, a raised arm). This is what the callout updates.
3. **Render cull sphere** — the model-space sphere the GPU tests against the frustum/cascades.

Today (3) is per model class: a sphere covering every baked clip frame, +5%
(`crates/ae_render/src/models.rs`, `fit` / `animated_radius`). It does not cover manual joints.

The design:

- **Default (3) is conservative and static.** Add a per-model manual reach allowance: the maximum
  distance from the root to any bone, with its subtree, over the model's declared scale range. The
  cull sphere is `max(clip bounds, manual reach)`. Correctness then never depends on a script call:
  a unit can never be culled because it forgot to update its bounds. No per-tick cost, no traffic.
- **`Bounds` is a synced component, computed on demand.** Sketch:

  ```rust
  pub struct Bounds {
      pub offset: [f32; 3],   // model-space centre
      pub radius: f32,        // model-space radius
      pub height: f32,        // vertical extent (selection/height tests)
      pub tick: u32,          // when computed (0 = none)
      pub flags: u8,          // valid, exact, grows-only
  }
  ```

- **The callout** is a flag in the synced channels, not a module call:

  ```rust
  // script, after a deployment or a pose change that alters the unit's volume
  anim.update_bounds();       // sets REQUEST_BOUNDS; the engine handles it this tick
  let b = anim.bounds();      // last Bounds (or the static model bound if none)
  ```

  A small `bounds` host system (or the ticker itself) sees the flag on the flagged rows only,
  evaluates that unit's current pose on the CPU (full `Rig::pose` + globals, strict f32),
  computes the sphere/box, writes `Bounds`, clears the flag, and bumps a version. Cost ~2–10 µs
  per request. A deterministic counter reports requests per tick; a debug warning fires if a game
  requests bounds for thousands of units every tick.

- **A cheaper variant** avoids the full pose: at load, compute each clip's model-space bounds and
  each manual bone's reach; the on-request bound is the union of the active clips' bounds plus the
  manual reach. O(layers), not O(bones). Recommended for the render/selection volume; the exact
  pose variant is kept for cases where the tighter bound is worth the cost.
- **Where the bound is used:**
  - Sim: selection/hit tests and any gameplay query that wants the current extent (not movement
    collision; `Body.radius` stays the movement volume).
  - Render: the cull may use the per-unit `Bounds` when it is tighter than the static sphere. It is
    uploaded in the animation record only when its tick changes; no per-tick traffic. If the design
    keeps per-class spheres only, the callout still serves gameplay and selection, and the renderer
    is unaffected.
- **Why on demand is the right rule:** the full pose is exactly the work this design removes from
  the per-tick path. Bounds change at state transitions (deploy, undeploy, a one-off large turn),
  not continuously, so a callout is both cheaper and more precise than a per-tick recompute.

## 9. API sketch (for discussion, not implementation)

SDK:

```rust
// animation (unchanged semantics, new transport underneath)
anim.play(RUN, Layer::BASE).looping().speed_from_ground().mask(LOWER_BODY);
anim.turn(TURRET, Axis::UP, yaw, deg_per_s(90)).short();
anim.turned(TURRET).await;

// on-demand queries, CPU and exact
let from = anim.bone_world(FIRE_FROM);          // world-space, via Facing + terrain height
let dir = anim.bone_direction(FIRE_FROM);       // or derived from two bones
let d = unit.to_unit(target - from);

// bounds
anim.update_bounds();
let b = anim.bounds();
```

Renderer/library:

- `units.pose` stays the library pass name; it now reads `units.anim` (state) and the model clip
  table instead of the key table.
- Remove `units.keys` / `KeyWrites`; add `units.anim` (state buffer) and `models.clips`.
- `units pose` still writes `units.palette`; graphs, modules and graph files keep declaring the
  pass by name. Their resource lists change, so the graph goldens and the default/post declarations
  change once.

## 10. Determinism, parity and tests

### 10.1 Determinism

- Synced state and outputs: unchanged strict-f32 rules; digests at 1/2/4/8 workers, dev/release,
  save→load→continue, replay verify.
- GPU pose: not hashed, not saved. It must not feed any synced value.
- The only new synced arithmetic is the bounds computation (strict f32, same as `Rig::pose`) and
  any trimmed phase representation (integer, exact).
- If phase is made integer, `ended` timing must be defined in that representation and pinned in
  tests; it is gameplay-visible through script waits.

### 10.2 CPU/GPU parity

- **Pose parity:** read back the palette for random channels and compare with `Rig::pose` +
  `globals` + `palette` at an integer tick (alpha 1). Existing `units_pose_models` already does
  this for the key path; extend it to layers, masks, additive layers, manual entries and all play
  modes. Tolerance: 1e-3 relative (the current test's tolerance), tighter before quantization.
- **Aim parity:** for a unit turning a turret and barrel while running a clip, compare the CPU
  fire-from position and direction with the drawn pose's corresponding bone. Gate: position error
  < 5 mm and direction error < 0.1° at typical model scales (Fox ~1 m, a 10 m mech larger). With
  H2 (same bake) the only source is interpolation/quantization; state the budget in the test.
- **Bounds parity:** after `update_bounds`, every vertex of the posed mesh at that tick is inside
  the reported bound (except deliberately excluded scale ranges); checked on the test models.
- **Delta transport:** a whole table, sparse deltas, a dropped frame, a reused slot, a hidden unit
  entering LOS, and a state reset all produce the same drawn pose as a whole-table reference.
- **Validation:** all GPU tests with Vulkan + synchronization validation, zero messages, including
  a live run with `--exercise` and a resize while animation state is in flight.

### 10.3 Bench gates

New/extended metrics, recorded in `bench/results/`:

| Metric | Today | Gate |
|---|---|---|
| `anim_tick` (sim, ms/tick, 10k/50k) | not measured | < 0.5 / < 2 ms |
| `cpu_anim_pack` (ms/tick) | 6.5 ms (pose) | < 0.3 ms at 10k |
| `anim_records` / `anim_bytes` per tick | 154k / 7.4 MB | < 10k / < 0.6 MB (A); near 0 steady (B) |
| `cpu_key_upload` | 1.1 ms | < 0.05 ms |
| `units pose` GPU | 0.43 ms | < 1.2 ms at 10k |
| `joint_keys` | 540k / 26 MB | 0 |
| GPU total (`army-fox`, `near-fox`) | 8.1 / 6.5 ms | no regression beyond tolerance (`live-fox` to be baselined) |

Also add a synthetic `ae-tool bench anim` (seeded 10k/50k channels, N layers/manuals) so the sim
ticker can be measured without the client and gated on every PR by deterministic counters.

## 11. Alternatives considered

| Option | Saving | Why not (yet) |
|---|---|---|
| Keep joint keys, compress to 16-bit, raise line tolerance | ~2–3× traffic | Still O(bones) CPU pose; still per-bone data; doesn't reach 50k |
| CPU pose only for visible units | proportional to visibility | RTS armies are mostly visible; saves little; doesn't fix O(bones) traffic |
| GPU-baked clips M11-style (one clip per unit) | removes pose CPU | No layers, masks, additive, manual joints; would have to be rebuilt anyway |
| Vertex-shader pose (evaluate per vertex) | no palette pass | Repeats bone math per vertex; hierarchy impossible; worse |
| GPU animation state machine, event-only upload (no per-tick state) | near-zero traffic | Persistent GPU state, tick/advance synchronization, drift risk; variant B gets most of it without the state machine |
| Move the whole animation service off the sim (view-side) | removes sim ticker | Breaks script queries, `ended` timing, save/load and determinism; rejected by M14 decisions |
| Per-joint collision volumes | more precise hits | Already rejected in M14; not needed for this problem |
| CPU evaluates aim chain, GPU everything else (chosen) | pose CPU ~0, traffic ~10–50× lower | — |

The chosen path is deliberately incremental: variant A is simple and exact and already solves the
scaling problem at 10k; variant B is a measured follow-up for 50k. Neither requires a GPU state
machine, and both keep the CPU able to reproduce the pose.

## 12. Risks and open questions

1. **`Anim` layout / component split.** Splitting manual entries into an optional component and
   trimming the record cuts sim traffic and digest cost, but changes the schema and re-pins every
   game digest (declaring a component shifts entity ids; see M4/M6 notes). Decide whether that is
   acceptable now or deferred until after the transport work.
2. **Facing/height for aiming.** The sim currently has no heading. The CPU aim/fire design needs a
   synced orientation (and an authoritative ground height). Add to `Body` or a `Facing` component?
   This is a prerequisite, not an optimization.
3. **Visual tolerance.** What divergence between the CPU aim direction and the drawn barrel is
   acceptable? Recommend < 0.1° and < 5 mm at the muzzle, enforced by a test. If a game needs
   tighter, use exact per-tick phase (variant A) and no quantization.
4. **Manual entries per unit.** Is 8 entries still right, or can it drop to 4 (turret, barrel,
   spine, recoil) in the common component, with an overflow component for the rare unit? Affects
   layout and script API.
5. **Bounds semantics.** Is the callout for the *selection/pose* volume only, or must it also change
   `Body.radius` (movement collision)? They should stay separate; confirm.
6. **Hidden units.** May animation freeze entirely for units outside the observer's view (no
   packing, no posing)? The sim still advances them; only their presentation is skipped. Recommended
   yes.
7. **Priority / timing.** This supersedes M14 step 3's key-line transport. Does it land before M14
   steps 5–6 (script API, death sequence) and M8–M10, or after? The design is compatible with either,
   but step 3's key path should not be extended further if this is next.
8. **GPU pose budget.** The 0.6–1.2 ms estimate may be optimistic on low-end GPUs. The fallbacks
   (visible-only pose, distance cadence, layer LOD) are listed; the bench decides.

## 13. Phased plan

**Phase 0 — baseline and instrumentation (no behavior change).**
Measure the sim ticker (`anim_tick`) separately; add byte/record counters to the renderer; record
`live-fox` with per-tick traffic. Confirm where the 6.5 ms and 7.4 MB actually go.

**Phase 1 — transport fixes independent of the pose.**
Upload a tick's instance columns once, not per frame slot (persistent table or tick ring); stage
only compact ranges; merge the animation and position copies. This is low-risk and already reduces
traffic at 60 fps.

**Phase 2 — sim-side cost.**
Remove the 528 B copy, the libm wrap and inactive work; parallelize the ticker; add the `anim`
bench. Optionally split `Manuals` and trim `Anim` (deliberate digest re-pin).

**Phase 3 — GPU clip evaluation (variant A).**
Add the model clip table (from `Rig`'s bake), the per-unit state record and the delta transport;
rewrite `units.pose` to evaluate layers/masks/manual entries; remove the key table, `Poser`,
`KeyWrites` and `units.keys`. Keep a debug switch to the old key path for one release of A/B
comparison. Add pose parity tests.

**Phase 4 — CPU aim/fire queries.**
Add `Orientation` (or `Body.heading`), the ground-height accessor, `bone_world`/`bone_position`, the
per-tick query cache and the aim parity test. Wire the test game's aiming to the new queries.

**Phase 5 — bounds callout and cull changes.**
Add `Bounds`, the request flag/handler, the static manual reach allowance, and (optionally) per-unit
cull bounds from the animation record. Reorder the graph to cull before pose and pose only visible
and shadow-casting rows.

**Phase 6 — large-count variant (if needed).**
Analytic phase/fade anchors (variant B) and ground-speed derivation on the GPU; measure 50k
traffic. Only then consider distance cadence/LOD.

Each phase ends with `cargo test --workspace`, the determinism suite, validation-clean GPU runs and
`ae-tool bench-diff` against the previous phase.

## 14. Expected outcome

At 10 000 animated units, per tick:

- CPU pose work: **6.5 ms → ~0** (GPU), plus a pack step of ~0.1–0.3 ms.
- Sim ticker: **unmeasured → < 0.5 ms** (and < 2 ms at 50k), parallel and allocation-free.
- Host→device: joint keys **7.4 MB → 0**; all per-tick animation traffic **~0.1–0.6 MB**, no per-bone
  data, once per tick (positions excluded).
- CPU resident: **~52 MB of key mirror/table → ~1–2 MB**.
- GPU pose: 0.43 ms → ~0.6–1.2 ms per frame (the price of moving the work; still a small fraction
  of the 8 ms army frame, whose shadows dominate).
- Aiming and firing remain CPU-exact and deterministic; bounds update only when a script asks.

At 50 000 units the current path is ~32 ms of CPU pose and ~37 MB/tick of keys; the proposed path is
a few ms of parallel ticking and (variant B) almost no animation traffic, with the GPU pose
budgeted and, if needed, distance-cadenced.

## Appendix A — budget arithmetic

All at 10 000 units, 24 bones, 30 Hz, 60 fps, 48-byte keys.

```
today
  CPU pose            6.5 ms
  keys/tick           154_000 * 48 B = 7.39 MB   -> 222 MB/s
  key table           10_000 * 24 * 2 * 48 B = 23.0 MB (26 MB with the largest model)
  key mirror          same order                  = ~26 MB
  staging/slot        same order                  = ~28 MB * slots
  Anim RMW            10_000 * 536 B              = 5.36 MB/tick
  start_manual copy   10_000 * 528 B              = 5.28 MB/tick
  GPU pose            0.43 ms/frame

proposed A (per-tick records)
  active-layer values 10_000 * (2 + 2) B = 40 KB/tick (1 layer)
                      10_000 * (8 + 8) B = 160 KB/tick (4 layers)
  manual entries      active units * entries * ~32 B; 10% * 2 = ~64 KB/tick
  positions           10_000 * 9 * 4 B = 360 KB/tick (unchanged, once per tick)
  total               ~0.46-0.6 MB/tick = 14-18 MB/s
  CPU pose            0 (queries and bounds only)
  pack                ~40-64 B/unit compared -> ~0.1 ms

proposed B (analytic anchors)
  steady marching army: no animation records; positions only (360 KB/tick)
  changes: clip switches, fades, manual values, bounds
```

## Appendix B — baseline references

- `docs/models.md`, "M14 step 3" table and benches: 6.5 ms posing, 154k keys / 7.4 MB, 1.1 ms
  upload, 0.43 ms `units pose`; key table 26 MB; palette 23 MB; per-slot staging 28 MB.
- `docs/PLAN.md`, M14 progress: steps 1–3, joint-key lines, feed batching, the aiming helpers and
  the open items ("spins by rate", "clips' original keys", "observer views", "cull spheres for
  manual joints").
- Commits `1d65949`, `0ccc283`, `d83baa4`: parallel posing, key lines/stable slots, feed write
  batches.
- Code sites: `sdk/ae-sdk/src/anim.rs` (`Anim`, 536 B), `crates/ae_model/src/anim.rs`
  (`Rig::advance/pose/dir_in`, `Baked`, `Key`), `crates/ae_services/src/anim.rs` (`Ticker`),
  `bins/ae-client/src/poser.rs` (`Poser`), `crates/ae_render/src/units.rs`
  (`InstanceStream`, `KeyWrites`, `UnitFeed`, `record_pose`), `crates/ae_render/src/models.rs`
  (`ModelSet::anim`, cull spheres), `crates/ae_render/shaders/units_pose.comp`,
  `units_keys.comp`, `crates/ae_render/src/terrain/app.rs` (graph order).
