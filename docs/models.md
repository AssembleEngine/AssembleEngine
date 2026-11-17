# Models (M11)

One model format for everything the engine draws besides terrain, water and particles: a skinned
model. glTF 2.0 is converted offline; the runtime reads only the engine format.

```text
ae-tool model convert IN.gltf|IN.glb OUT.aemodel   # prints warnings, size, convert time
ae-tool model info FILE.aemodel                    # parts, textures, clips, load time
```

- `crates/ae_model`: the format (read, write, validate), KTX2 textures (`ktx`), poses and clips
  (`pose`). Light dependencies only (serde_json, glam, `ktx2`, pure-Rust zstd decoding).
- `crates/ae_gltf`: the converter, used by `ae-tool`. It pulls the image decoders, MikkTSpace and the
  Basis Universal transcoder (C++), so the client and renderer never build them.

## The format

Little-endian: magic `AEMODEL\0`, version (u32, now 2: primitives have a layout), header length (u32), a JSON header padded
to 16 bytes, then a blob. The header's byte ranges point into the blob, 16-byte aligned. A model
of another version is refused ("convert it again").

- **Bones**: the glTF nodes of the scene, parents before children, under one synthetic root
  (bone 0, `__gltf_root`). That root turns glTF's axes (+Y up, front +Z, +X left) into the engine's
  (+X forward, +Y up, +Z right) with a quarter turn about +Y. Each bone has its rest transform as a
  full matrix (non-uniform scale kept) and as TRS for clips to override.
- **Joints**: the skinning palette, each a bone and an inverse bind matrix. A skinned mesh's vertices
  reference its skin's joints. A mesh hanging on a node (a piece) gets one joint, that node's bone
  with the identity, and every vertex on it at weight 1. So pieces and skins share one vertex
  layout and one draw path; moving a bone moves either kind. Joints are shared where bone and
  matrix agree.
- **Vertices** (80 bytes): position, normal, tangent (xyz + bitangent sign), UV0, UV1, colour
  (RGBA unorm16, linear), four joints (u16), four weights (unorm16, summing to 65535).
- **Primitives**: triangle lists, one material each, in one shared vertex and index buffer. Each
  has a GPU layout (decision 2026-11-16, after RecoilEngine's GL4 model path): **rigid** (every
  vertex on one joint at weight 1; 17 words on the GPU, the joint in the last) or **skinned** (four
  joints and weights; 20 words). The converter gives each primitive the cheapest layout its
  vertices allow and merges the primitives of one material and layout (`Model::merge_primitives`):
  a piece model draws once per material, its per-vertex joint picking the piece. In memory and in
  the file every vertex is the 80-byte one; `Model::gpu_words` packs a primitive for the GPU.
- **Materials**: glTF's metallic-roughness: base colour, metallic-roughness, normal (with scale),
  occlusion (with strength), emissive (`KHR_materials_emissive_strength` multiplied in); alpha
  opaque, mask (cutoff) or blend; double-sided. Each texture reference carries its UV set, its
  sampler and its `KHR_texture_transform` (kept per reference, not baked into UVs: two textures may
  share a UV set with different transforms).
- **Textures**: KTX2 files in a GPU format (decision 2026-11-16: "whatever format the glTF brings,
  preferably from the KTX2 suite"):
  - PNG / JPEG: RGBA8, sRGB for base colour and emissive, UNORM otherwise; a full mip chain (2x2
    box, in linear light for sRGB); zstd level 19 supercompression, so lossless.
  - KTX2 already in a GPU format (RGBA8, BC1–BC7): kept, its VkFormat switched to the sRGB or
    UNORM twin its use asks for.
  - Basis Universal KTX2 (`KHR_texture_basisu`, ETC1S or UASTC): transcoded to BC7, every level
    it has. The runtime needs no transcoder.
  - The same image used as colour and as data is stored twice (sRGB and UNORM).
- **Clips**: glTF animations as data: per channel a bone, a path (translation, rotation, scale), the
  interpolation (step, linear, cubic spline) and the keys as they are. `pose::Pose::sample` plays
  them (slerp for linear rotations; held at the first and last key outside them).
  `pose::bake` samples a clip at 60 frames per second or a little more (so the last frame falls on
  its duration) into per-frame TRS of the bones it animates; `BakedClip::locals` plays that the
  way the renderer's pose pass does (two frames blended: lerp, nlerp; clips loop). Baking shifts a
  step-interpolated key by up to a frame. The sim does not drive bones yet (`PieceTransform`).

## The renderer (`ae_render::models`, the units)

`UnitsSetup::models[class]` gives a unit class a model; the others stay darts, and with no model
at all the units renderer is exactly the darts' one (the same pipelines, buffers and goldens).

GPU-driven and bindless, after RecoilEngine's GL4 model path (one global geometry buffer,
per-unit transform ranges, multi-draw) without its per-vertex quaternion interpolation:

- **Resources**: one geometry buffer (every model's vertices packed per primitive in its layout,
  read by vertex pulling, then every index); one material table; one partially bound array of
  1024 textures the materials index; an animation table (bones, joints, clips, baked frames); one
  palette buffer (`units.palette`); per frame slot a rows buffer (pose offsets and jobs). The
  visible list holds per model class a region for the camera and one per shadow cascade.
- **Commands**: one `VkDrawIndexedIndirectCommand` per (class, primitive), with a draw record
  (material, class, vertex base and layout, the class's region of the visible list) the shaders
  read by `gl_DrawID`. Opaque commands first, then alpha-masked ones: two pipelines, because a
  shader that discards loses early depth. The camera draws every model in two multi-draws; the
  shadow map in one per cascade.
- **Cull** (`units_models_cull.comp`, in the units' cull pass): a model unit is tested with its
  model's bounding sphere, appended to its class's region of the visible list and counted in each
  of its class's commands. The reset copies the commands' template (instance counts 0).
- **Shadows**: with the units' shadow map on, the cull also tests each model unit's sphere against
  each cascade's sides in light space (not its depth: a caster nearer the sun than the cascade's
  box still shadows what is in it) and appends it to its class's region for that cascade, counted
  in that cascade's copy of the commands (after the camera's in the indirect buffer). The shadow
  caster draws each cascade's commands; the darts' caster draws every dart and skips model
  classes. Opaque and masked commands in a multi-draw each per cascade; the masked caster tests
  alpha as the camera does, so what is masked away casts no shadow. The graphs declare the reset
  and the cull before the shadows.
- **Poses** (M14, hybrid animation, `docs/hybrid-animation.md`; replaced the CPU-posed joint
  keys of step 3, which replaced the GPU-baked clips): the GPU evaluates each unit's pose from
  its sim animation channels; nothing is posed on the CPU (the pose is presentation; the sim
  answers its own joint questions, `Rig::dir_in`). The models' animation table (binding 8,
  `models::animation`) holds each model's clips as the **sim's own bake**
  (`ae_model::anim::Rig::clip_table`: per clip its frames at 60/s, a track word per bone, keys
  as translation, rotation, scale), its masks (a weight word per bone) and its rest transforms,
  from the rig the renderer is given per class (`UnitsSetup::rigs`: the client's are the sim's,
  made by `SimOptions::rigs` as the sim makes them, masks included; none: `Rig::new` of the
  model). The client
  (`poser.rs`) packs each posed unit's `Anim` into an **animation record** of 15 16-byte cells
  (`units::RECORD_CELLS`: the tick and the shown manual entries' count; per layer clip, mask,
  mode, kind, weight, phase; per shown manual entry its bone, kind and value). A posed unit
  has a slot of two records, each stamped with its tick, the same while it stays posed; each
  tick its record is compared with the newer one: the same (standing, a clip held) and nothing
  is written; otherwise it goes over the older one when the newer is the last tick's (only the
  cells that differ), or the newer is stamped the last tick first when it is older (it held so
  since). Units are packed in parallel (rayon). Streams carry the writes
  (`InstanceStream::anim`, `units::AnimWrites`); the `UnitFeed` keeps them until the renderer
  takes them, each cell once at its newest value (`UnitFeed::take`), so every stream must be
  published (`publish_reusing` hands taken batches' memory back). A renderer that missed
  writes (its first take with no whole table pending, a frame dropped after staging, too many
  batches untaken) makes the feed ask the producer for the whole table (`wants_whole`; the
  packer writes its mirror, `reset`). The stream's `pose` attribute is the unit's first cell
  (`units::NO_POSE`: the rest pose). The palette holds 3x4 matrices (rows, 48 bytes): each
  model's rest palette at its head (written once; units at rest point there), then a range per
  posed unit, then the pose pass's scratch. Each frame the render thread stages the cells it
  took (host visible, per slot, bindings 10 and 11); when a new tick arrives it gives each row
  its offset and lists the posed units (pose jobs). The `units pose` pass (`units.pose`,
  compute, before the shadows and the draws) first scatters the staged cells into the units'
  animation table (`units_anim_writes.comp`; device local, one for all frame slots, with
  barriers against the earlier frames' reads, binding 9), then runs one invocation per job
  (`units_pose.comp`): each bone evaluated from each record as `Rig::pose` does (rest, then
  the layers in order: the clip at its phase by its play mode, two frames lerped; blended by
  weight × mask weight, at most 1, or its change from the clip's first frame added; then a
  bone with a shown manual entry takes its rest transform with the entries' values), drawn
  between the two records' transforms at the drawn tick (the stream's tick − 1 plus the
  interpolation weight; tick differences in integers; lerp and nlerp the short way, held at
  the older before it and the newer past it; only one record evaluated then), a bone neither
  record poses at its rest matrix; the hierarchy walked in place, each joint's bone times its
  inverse bind. So a pose is the lerp of the two ticks' poses, as the joint keys drew it (a
  spin faster than half a turn a tick shows wrong, as it did). The vertex shader only blends
  up to four matrices (`palette[offset + joint]`), normals by the cofactor matrix (right under
  non-uniform scale). Memory: the palette is sized for every unit posed with the largest
  posable model (capacity × (joints + bones) × 48 bytes; Fox, 10 000 units: 23 MB), the units'
  animation table for two records per unit (480 bytes; 10 000 units: 4.8 MB, device local),
  and per frame slot staging for as many writes (4.8 MB + 4 bytes a cell, host visible); the
  CPU mirror is as large as the table. The Fox's clips take 0.6 MB of the models' table.
- **Cull sphere**: a model with clips is culled with a sphere covering every vertex in every baked
  frame of every clip, 5% larger (motion between frames), for all units of its class. Manual
  joints (M14) can reach further: not covered yet.
- **Fit and frame**: model space is the dart's local frame (+X along the heading, +Y up, +Z to the
  right). The model's footprint radius (its rest bounds' largest extent from the origin along x or
  z) is drawn at the unit's drawn size, so the minimum pixel size applies as for darts.
- **Materials**: base colour (factor × vertex colour × texture), alpha mask, normal map, occlusion,
  emissive, double-sided; lit like the darts (sun past the hills, units and particles, sky, fog).
  Not yet: metallic-roughness, blending (blended materials are drawn as masks at 0.5).
  Implemented but not yet covered by a GPU test: normal maps, occlusion, emissive, the UV
  transform, non-default samplers, BC7 textures on the GPU.
- **Device**: needs `multiDrawIndirect`, `shaderDrawParameters` and
  `descriptorBindingPartiallyBound` (enabled when present; a units renderer with models refuses to
  start without them), 1028 samplers per shader stage (checked), and `textureCompressionBC` for BC
  textures. A model without triangles is refused.

Tests (`crates/ae_render/tests/gpu.rs`, validation on): `units_draw_models`, a piece model built in
the test (opaque front, masked right side, a block masked away), the Fox and a dart in one frame,
without and with the units' shadow map; front and right land where the heading puts them (not
turned, not mirrored), the mask discards, the Fox shows its texture, the blocks cast shadows and
the masked-away block none. `units_pose_models`: a base-and-arm model built in the test (two
clips, one moving the base and so the arm; its rig given a mask), the Fox and BoxAnimated with
animation records (ticks 5 and 7, some stored newer first): one layer in every play mode, a loop
across its wrap, two layers blended and weighted, an additive layer on the mask, a weight-0
layer, a clip and a mask the model lacks, manual turns, moves and scales (one not shown yet),
a turn's records on opposite hemispheres; published as a stale whole table and then the records
as writes over it, units at rest and with records not in the table, drawn half way between the
records and on the newer; every row's palette read back equals the sim's (`Rig::pose` of each
record's channels, lerped, `Rig::globals`, `Rig::palette`) to 1e-3 relative (checked to fail
with the additive rotation or the manual rotations dropped from the shader); the arm is drawn
where its pose puts it; a unit whose rest sphere is off screen but whose arm reaches in is drawn. `units_model_shadows_culled`: a pillar far outside every
cascade is counted in none; a pillar off screen toward the sun is in a cascade and shadows the
ground in view (checked to fail when the cascades take only the units the camera sees). `ae_model` tests: baked clips against the sampler (linear
translation, rotation across hemispheres, a duration not a whole number of frames, looping) and
the animated reach.

## The client

```text
ae-client --model [CLASS=]FILE.aemodel|.glb|.gltf
ae-client --game scripted --model assets/models/khronos/Fox/Fox.glb
```

`--model` draws class CLASS with the model; without CLASS every class (but the observer's blips).
A glTF is converted once by `ae-tool model convert` into `target/models/<name>.aemodel` (again when
the glTF is newer); the client does not link the converter. The model is also an animation model
of the sim's (`ae_services::anim`), named by the file's lower-case stem (`Fox.glb`: `fox`), next
to the content's `models/*.aemodel`. Units whose `Anim` names their class's model are posed by
their channels after every tick (the raw capture; not yet an observer's view); the rest stand at
the rest pose. What units play is the game's: `--game scripted` (M14's test game) walks and
stands Foxes, blending the clips by the animation service and turning their heads. `--clip`
(M11's client-side clip choice) is gone.

## Benches

`ae-client --bench army-fox,live-fox,near-fox` (`bins/ae-client/src/bench.rs`): since M14 step 3
they run the `scripted` game (10 000 Foxes on a 12 m grid under the `army` camera, walking and
standing, animated by the sim) instead of the crossing army playing `--clip`, so the pose pass
poses all 10 000 each frame; `near-fox` with a closer camera. `army-fox` and `near-fox` draw one
tick (the whole animation table uploaded once); `live-fox` keeps the sim running, a tick every
second frame, packed and uploaded as when playing. They also report `model_setup`
(`ModelSet::new` for the Fox: geometry, texture upload, clip tables, cull spheres),
`cpu_anim_pack` (the client's time to read the channels and pack the units' records, per
stream), `anim_cells` (the table, 16-byte cells) and `anim_writes` (cells written a tick);
`live-fox` also `cpu_anim_publish` (the feed keeping a tick's writes, on the sim thread),
`cpu_anim_upload` (the render thread's time to take and stage them) and
`anim_cells_uploaded`. The sim ticks in the `--warmup` frames and the measured ones only, so the
workload does not follow the frame rate.

M14 (RTX 3060 Ti, 24 threads, 2026-11-17; `live-fox` for the last two rows):

| | CPU a tick | Written a tick | Upload (render thread) | `units pose` |
|---|---|---|---|---|
| Joint keys, first cut (every key, one thread) | 18.8 ms posing | 540 000 keys (26 MB) | not measured | 0.39 ms |
| Joint keys in parallel (every key) | ~10 ms posing | 540 000 keys | not measured | 0.39 ms |
| Joint keys: lines, slots, unchanged skipped | 6.5 ms posing | 154 000 keys (7.4 MB) | 1.1 ms | 0.43 ms (scatter included) |
| Hybrid: records, clips sampled on the GPU | 1.3–1.8 ms packing | 28 500 cells (0.57 MB) | 0.12 ms | 0.37 ms (scatter included) |

The CPU time is reading the units' `Anim` rows (536 bytes each), matching them to the stream and
packing; no bone is evaluated. A walking Fox writes its record's tick and its layers' phases a
tick (about three cells). The `units pose` pass samples the clips for both records only while
the drawn tick is between them. `gpu_total` (baselines, `bench/results/baseline/gpu-*-fox.json`):
8.3 ms `army-fox`, 8.2 ms `live-fox`, 6.9 ms `near-fox`, mostly the shadows (no earlier baseline
of the `scripted` scenes to compare: the old ones were of the crossing army). Validation clean
(`live-fox` and the GPU tests). A `--capture` of the scripted game shows the Foxes mid-stride
and standing, as the sim plays them.

The M11 numbers (the crossing army playing `--clip`):

RTX 3060 Ti, 1280x720, 300 frames, median GPU times (2026-11-16), before and after the shadow
cascades' cull (casters: model units drawn into the three cascades, summed):

| Scene | Visible | Pose | Cull | Shadows | Casters | Units | GPU total | Model setup |
|---|---|---|---|---|---|---|---|---|
| army (darts) | 8 606 | — | 0.005 ms | 0.044 ms | — | 0.036 ms | 0.53 ms | — |
| army-fox, before | 8 594 | 0.31 ms | 0.010 ms | 5.3 ms | 30 000 | 2.07 ms | 8.9 ms | 16 ms |
| army-fox, after | 8 594 | 0.31 ms | 0.024 ms | 4.4 ms | 28 313 | 2.08 ms | 8.1 ms | 17 ms |
| near-fox, before | 4 398 | 0.31 ms | 0.007 ms | 4.8 ms | 30 000 | 1.08 ms | 7.4 ms | 18 ms |
| near-fox, after | 4 398 | 0.31 ms | 0.020 ms | 3.9 ms | 25 077 | 1.07 ms | 6.5 ms | 16 ms |

Before the cull every unit went into every cascade, at any view (a `--capture` with 424 units
visible: 4.9 ms of shadows). The cull leaves out what no cascade holds, but the cascades are large
and nested, so in these views nearly every Fox is drawn into all three: the shadows stay
vertex-bound (about 48 million vertices). The pose pass, 31 ns a unit, stays as it is.
Validation runs (`AE_VK_VALIDATION=1`: army, army-fox, near-fox, battle): 0 errors, 0 warnings.

## The converter: what it reads, drops and refuses

Read: the default scene (or the first one; with no scene, every root node). It reads:

- POSITION, NORMAL, TANGENT, TEXCOORD_0/1, COLOR_0, JOINTS_0/1 + WEIGHTS_0/1;
- sparse accessors;
- triangle lists, strips and fans;
- the inverse bind matrices (identity when absent).

- No normals: flat normals.
- No tangents and a normal map: MikkTSpace tangents on the normal map's UV set (unwelded, then
  welded back).
- A skinned mesh ignores its node's transform (as glTF prescribes).

Decisions taken with the user (docs/PLAN.md, M11):

- More than four influences: the four strongest kept and renormalised, with a warning.
- TEXCOORD_2 and above, COLOR_1 and above: dropped, with a warning.
- Non-uniform node scale: kept (bones are full matrices).
- Taken extensions: `KHR_texture_transform`, `KHR_materials_emissive_strength`, `KHR_texture_basisu`.
  Any other required extension (Draco, meshopt, ...) fails the conversion, naming it; optional
  ones are ignored with a warning.
- Morph targets and their weight channels, points and lines, cameras: dropped, with a warning.
- Lights (`KHR_lights_punctual`): ignored until M12.

## Measured

`ae-tool model convert` on the Khronos samples (release build; load = parse and validate,
textures not decoded):

| Model | glTF | Model file | Convert | Load | Contents |
|---|---|---|---|---|---|
| SimpleSkin | 3.5 KB | 3.0 KB | 0.2 ms | 0.006 ms | 2 joints, 10 vertices, 1 clip |
| BoxAnimated | 12 KB | 31 KB | 0.3 ms | 0.022 ms | 2 pieces, 320 vertices (rigid: 68 B each on the GPU), 1 clip |
| Fox | 163 KB | 295 KB | 332 ms | 0.075 ms | 24 joints, 1,728 vertices, 1024² texture, 3 clips |
| CesiumMan | 438 KB | 906 KB | 596 ms | 0.094 ms | 20 joints, 3,273 vertices, 1024² texture |
| AnimatedCube | 568 KB | 739 KB | 236 ms | 0.028 ms | 512² texture, 1 clip |

The files are larger than the glTF where textures dominate: lossless RGBA8 with mips against PNG or
JPEG (the Fox's JPEG most of all). Converting is mostly zstd at level 19.

Tests: `cargo test -p ae_model -p ae_gltf`. They cover:

- glTFs built in the tests: axes, pieces, influences, extensions, textures and transforms,
  tangents, sparse accessors, strips, clips, and a Basis UASTC KTX2 made by the encoder and
  transcoded;
- the vendored Khronos samples (`assets/models/khronos`, with their licences).
