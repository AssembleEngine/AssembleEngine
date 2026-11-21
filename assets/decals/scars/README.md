# Scars

Explosion scars from Recoil (`cont/base/bitmaps/bitmaps/scars`), used here with the permission
of their rights holder, who is also this repository's owner.

- `scarN.tga`: colour, sRGB, straight alpha.
- `scarN_normal.tga`: the tangent normal x/y in rg (red toward the decal's +x, green
  toward its +z, down the image; z is rebuilt), roughness in b (linear, 0..1), glow mask
  in alpha. The loader detects the green handedness from the normal field, so DirectX-style
  (green down) and OpenGL-style (green up) maps both work. Scar layers take their roughness
  strictly from b; there is no roughness factor.
- The bundled set is ten 512² scars: `scar1` is Recoil's old `scar2`, kept as it is, and
  `scar2`–`scar10` are further scars in the same style. All are 32-bit RLE TGA.
- The bundled files' b still holds the old normal-z component until their roughness pass;
  the loader uses it as roughness meanwhile.
- A source may be any square; the client's atlas resamples it to 512². The client loads every
  `*.tga` with a matching `*_normal.tga` from the directory, in name order
  (`--scars DIR` for another set, `--scars generated` for the engine's generated scorch mark).
