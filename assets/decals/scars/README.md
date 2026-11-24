# Scars

Explosion scars from Recoil (`cont/base/bitmaps/bitmaps/scars`), used with the permission of
their rights holder, who is also this repository's owner. The bundled set contains ten
512×512 pairs of 32-bit RLE TGAs.

- `scarN.tga` stores sRGB colour with straight, linear coverage in alpha.
- `scarN_mat.tga` stores linear micro-height, metalness, roughness and glow in RGBA.
  Height 128 is neutral. U increases across columns towards decal-local +X and V increases
  down rows towards decal-local +Z. The terrain heightmap supplies the actual crater shape.

The client loads colour files in name order. A matching `_mat.tga` takes precedence over a
legacy `_normal.tga` (tangent normal XY, roughness, glow). Legacy green handedness is detected
as before. Source images can be any square and are resampled to the atlas size.

The renderer builds signed tangent normals from the height slopes and averages
and renormalizes the vectors through the mip chain. The normal array uses Vulkan
`R16G16_SNORM`, for about 16 MiB at 512×512 with the ten bundled scars and two
generated layers. Colour, material and normals share the terrain-following UV
footprint. The shader estimates the projection's local tangent frame from the
heightfield on curved terrain.

Overlap defaults to ordered replacement of scar relief, oldest first, with terrain detail
retained beneath it. The renderer API sets each decal's height range through `Decal::relief`.
`Relief::Proportional` defaults to 24/512 of the full width. Explosion scars use 96/512
to make the relief easier to see. `Relief::World` uses a fixed range in metres.

Client options for inspection:

- `--scars DIR` loads another set. `--scars generated` uses the built-in layers.
- `--decal-debug height|coverage|normal|tangent-normal` selects an unlit view.
  Per-layer views show the newest contributing decal. The final normal uses
  world-space XYZ as RGB.

**Relief modes.** Use `--decal-relief off|normal|rcsm` or press **F6** in
the client. **F7** cycles debug views and **F9** switches the RCSM overlap policy.
See [Terrain decal relief](../../../docs/decal-relief.md) for render settings,
validation and benchmark commands.
