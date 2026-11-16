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

Little-endian: magic `AEMODEL\0`, version (u32, now 1), header length (u32), a JSON header padded
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
- **Primitives**: triangle lists, one material each, in one shared vertex and index buffer.
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
  them (slerp for linear rotations; held at the first and last key outside them). No runtime
  player yet: the sim will drive bones (`PieceTransform`, not yet in the sim).

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
| BoxAnimated | 12 KB | 31 KB | 0.3 ms | 0.022 ms | 2 pieces, 320 vertices, 1 clip |
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
