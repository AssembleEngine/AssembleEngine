# Scars

Explosion scars from Recoil (`cont/base/bitmaps/bitmaps/scars`), used here with the permission
of their rights holder, who is also this repository's owner.

- `scarN.tga`: colour, sRGB, straight alpha.
- `scarN_normal.tga`: tangent-space normal in rgb, glow mask in alpha. scar1's green points down
  the image (DirectX style), scar2's up (OpenGL style); the loader flips scar2's.
- scar1 is re-encoded run-length (lossless, the same pixels); scar2 is as in Recoil.

The client uses them by default (`--scars DIR` for another set, `--scars generated` for the
engine's generated scorch mark).
