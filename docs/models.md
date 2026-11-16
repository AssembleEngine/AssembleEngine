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
  palette buffer (`units.palette`); per frame slot a rows buffer.
- **Commands**: one `VkDrawIndexedIndirectCommand` per (class, primitive), with a draw record
  (material, class, vertex base and layout, the class's region of the visible list) the shaders
  read by `gl_DrawID`. Opaque commands first, then alpha-masked ones: two pipelines, because a
  shader that discards loses early depth. The camera draws every model in two multi-draws; the
  shadow map in one per cascade.
- **Cull** (`units_models_cull.comp`, in the units' cull pass): a model unit is tested with its
  model's bounding sphere, appended to its class's region of the visible list and counted in each
  of its class's commands. The reset copies the commands' template (instance counts 0).
- **Shadows**: the shadow caster draws every row; the rows grouped by class (written on the CPU
  when a new tick arrives) give each class's instances. The darts' caster skips model classes.
  Opaque and masked commands in a multi-draw each per cascade; the masked caster tests alpha as
  the camera does, so what is masked away casts no shadow.
- **Poses**: the stream's `clip` and `clip_time` attributes pick a unit's pose: `clip` 0 (or past
  its model's clips) the rest pose, `k` the model's clip `k - 1` at `clip_time` seconds (at the
  stream's tick; drawn a tick behind as positions are, clamped at 0; clips loop). The palette
  holds 3x4 matrices (rows, 48 bytes): each model's rest palette at its head (written once; units
  at rest point there), then a range per unit playing a clip, then the pose pass's scratch. When a
  new tick arrives the CPU gives each row its offset and lists the units playing a clip (pose
  jobs). The `units pose` pass (`units.pose`, compute, before the shadows and the draws) runs one
  invocation per job: rest local matrices, the clip's animated bones blended from two baked
  frames, the hierarchy walked in place, each joint's bone times its inverse bind. So bones are
  interpolated once per unit and frame; the vertex shader only blends up to four matrices
  (`palette[offset + joint]`), normals by the cofactor matrix (right under non-uniform scale).
  Memory: the palette is sized for every unit playing a clip of the largest animated model
  (capacity × (joints + bones) × 48 bytes; Fox, 10 000 units: 23 MB).
- **Cull sphere**: a model with clips is culled with a sphere covering every vertex in every baked
  frame of every clip, 5% larger (motion between frames), for all units of its class.
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
clips, one moving the base and so the arm), the Fox and BoxAnimated playing clips, units at rest
and with a clip their model lacks, drawn on time and a tick behind; every row's palette read back
equals the CPU's (`ae_model::pose`) to 1e-3 relative; the arm is drawn where its pose puts it; a
unit whose rest sphere is off screen but whose arm reaches in is drawn. Checked to fail without
the lag and without the padded sphere. `ae_model` tests: baked clips against the sampler (linear
translation, rotation across hemispheres, a duration not a whole number of frames, looping) and
the animated reach.

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
