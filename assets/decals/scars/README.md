# Scars

Explosion scars from Recoil (`cont/base/bitmaps/bitmaps/scars`), used here with the permission
of their rights holder, who is also this repository's owner.

- `scarN.tga`: colour, sRGB, straight alpha.
- `scarN_normal.tga`: tangent-space normal in rgb, glow mask in alpha. The loader detects the
  green handedness from the normal field, so DirectX-style (green down) and OpenGL-style
  (green up) maps both work.
- The bundled set is one scar, `scar1` (the old `scar2`, kept as in Recoil). The client loads
  every `*.tga` with a matching `*_normal.tga` from the directory, in name order.

The client uses them by default (`--scars DIR` for another set, `--scars generated` for the
engine's generated scorch mark).
