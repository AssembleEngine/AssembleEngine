# Terrain decal relief

`ae-client --decal-relief off|normal|rcsm` selects the initial technique.
**F6** cycles the modes in a running client. Normal mapping is the default.
**F9** cycles the RCSM overlap policy. The window title shows the active relief mode and debug view.
Switching modes keeps the same decal instances, textures, lifetimes and lighting.

| Setting | Default | Purpose |
| --- | --- | --- |
| `--decal-relief-strength` | 1 | Multiplier on the instance's physical height range. RCSM applies an additional 2.5× multiplier to both parallax and normal strength. |
| `--rcsm-overlap safe\|dominant\|independent` | safe | See the overlap policies below. |
| `--rcsm-steps` | 48 | Maximum cone proposals, capped at 256. The exact cell audit gets four times this budget, capped at 256 cells. |
| `--rcsm-refine` | 6 | Maximum binary refinement steps in a signed round-off bracket, capped at 24. An analytical root is available even with zero bisections. |
| `--rcsm-min-pixels`, `--rcsm-full-pixels` | 0.75, 2 | Fade the physical parallax range according to projected displacement in screen pixels. |
| `--rcsm-min-view-dot` | 0.18 | Minimum geometric normal–view cosine. The parallax range fades over the next 0.25 in cosine, up to a cosine of 1. |
| `--rcsm-force` | off | Bypass the screen-footprint LOD factor. Domain, angle, overlap, and traversal checks still apply. |

`--decal-debug VIEW` or **F7** selects an unlit view. The names are `off`,
`height`, `coverage`, `normal`,
`tangent-normal`, `cone`, `cell-bounds`, `iterations`, `depth`, `uv`, `failed`,
`overlap`, and `lod`. In `uv`, blue indicates a validated hit. `failed` marks
attempted traces that used the bump fallback in red. `overlap` marks conservative
fallback in green. The `lod` view encodes requested quality, successful quality
and the amount of bump fallback in RGB.

## Source and shading

The ten in-tree `scarN_mat.tga` files are the pre-macro-crater micro-height sources
under their current filenames. All three modes use their same R, G, B and A
channels for height, metalness, roughness and glow. The colour alpha supplies
coverage. The shader samples every decal property at the hit UV in RCSM, then
composites and lights the result in the terrain pass. Height geometry, depth and
shadows remain tied to the real terrain.

The conventional path builds signed XY normals at the common 24/512 reference
range. It scales decoded slopes by each instance's world dimensions and strength,
blends normals in a common terrain tangent frame in decal order, and uses
reoriented normal mapping to compose with the terrain's detail. Mips reconstruct,
average and renormalize XYZ. The tangent frame uses the local slope of the
unfolded projection.

RCSM samples the same signed tangent normals at the traced UV. It samples the
original UV for legacy normal-only packs, generated normal-only layers, and
pixels where tracing falls back.

## Cone tracing and overlap

The implementation follows [Robust Cone Step Mapping, Bán, Valasek, Bálint and
Vad (2024)](https://doi.org/10.2312/sr.20241146), specifically corrected relaxed
generation (Algorithm 2), the 3×3 minimum aperture correction (Equation 2), and
cell-max traversal (Algorithm 1). The publication is CC BY. The
[authors' reference demo](https://github.com/Bundas102/robust-cone-map) was consulted.
The implementation is derived from the paper and uses the renderer's Rust/Vulkan
and GLSL interfaces.

The GPU generator prunes Algorithm 2's vertex search with a maximum-height
quadtree. The runtime uses the paper's cone step and next-cell step to propose an
interval. A conservative bound on the bilinear heightfield's slope certifies the
prefix. The rest is checked in cell order using maximum-height bounds and an
analytical first root for each potentially intersecting bilinear patch. This
additional check handles features entered and exited within a single cell.
Refinement narrows a tiny bracket around an analytical crossing when the bracket
has a sign change. Budget exhaustion, an unsupported angle or a ray outside the
decal domain uses ordinary bump.

The canonical height axis uses the original UNORM height values, with the
minimum and maximum expanded to include neutral 128/255. The world ray
starts at the maximum and descends to the minimum. Ratios account for physical
height and both horizontal dimensions. The ray amplitude eases down as either
endpoint approaches the atlas boundary or the projected ray approaches the
cell-traversal budget. Height sampling in traversal explicitly
interpolates the four source texels at level zero. Cone parameters stay at level
zero and have no averaged mip chain. Final material taps use the original screen
footprint at the displaced UV. Curved terrain uses the existing local planar
differential of the unfolded UVs.

The default overlap policy bounds each decal by a world-space disc. A sweep over
the live instances computes clearance to the other discs. The shader compares that
bound to the full possible planar displacement, fading within the safe interval
and using normal mapping when clearance is inadequate. This may fall back
across an entire pair of nearby decals even where individual texels are clear.
The experimental dominant policy selects the newest original-UV visible layer,
traces it once, and applies its world-space offset to the other layers where the
sample remains in bounds. The experimental independent policy traces each layer
separately. Both are approximations when virtual surfaces intersect.

## Cache and memory

Cone preprocessing is keyed by SHA-256 over the canonical height bytes, dimensions,
format and boundary/version constants, and the generator shader source. Files are
validated by size and payload checksum and published by atomic rename under
`$XDG_CACHE_HOME/assembleengine/decal-cones` or `~/.cache/assembleengine/decal-cones`.
`AE_DECAL_CONE_REPORT=1` reports cache-miss build time, including readback,
bilinear correction and cache I/O, separately from frame timing.

For the default 512² array of ten bundled scars plus two generated layers, colour,
material and signed normals each use about 16 MiB with full
mips. The level-zero RGBA32F acceleration array uses 48 MiB for corrected apertures,
certified apertures, and maximum/minimum cell heights. Total texture payload is
approximately **96 MiB**.
Eleven distinct heightfields, including the shared neutral fallback, consume
about **44 MiB** in the on-disk cache. Figures exclude Vulkan allocation padding
and startup staging buffers.

## Validation and comparisons

The Vulkan tests exercise the renderer paths and compare the production cone
tracer against an independent double-precision bilinear-cell oracle. They cover
constant and ramp heights, sinusoidal relief, narrow creases, raised plates, signed
height, axial and oblique rays, boundaries and exhausted iteration budgets.

```sh
cargo test -p ae_render --lib gpu_relaxed_cones_and_first_hits_match_the_bilinear_reference -- --ignored --nocapture
cargo test -p ae_render --test gpu decal -- --ignored --test-threads=1
cargo test -p ae_render --test gpu scar_height_relief_relights_the_ground -- --ignored
```

For GPU terrain-pass timings with identical placement, run each command with
`--decal-relief normal` and `--decal-relief rcsm`. Fixed scars use a deterministic
seed and the same lifetime in each run.

```sh
ae-client --bench scar-close,scar-orbit --decal-relief rcsm --no-save
ae-client --bench scar-far --bench-scars 36 --bench-scar-spread 240 --decal-relief rcsm --no-save
ae-client --bench scar-far --bench-scars 1024 --bench-scar-spread 450 --decal-relief rcsm --no-save
ae-client --bench scar-close --bench-scars 12 --bench-scar-spread 3 --decal-relief rcsm --no-save
```

### Initial implementation measurements

RTX 3060 Ti, 1280×720, 96 measured frames after 70 fixed warm-up frames.
These measurements used a 1.5× RCSM height scale, the 24-proposal budget
and the since-removed surface-gradient fallback.
The desktop compositor remained active. Values are median GPU time for
the terrain pass in milliseconds, including the base terrain draw.

| Scene | Tangent normals | RCSM |
| --- | ---: | ---: |
| One scar, 16 m | 0.121 | 0.685 |
| One scar, full orbit | 0.120 | 0.673 |
| 36 scattered scars, 600 m | 0.106 | 0.155 |
| 1,024 dense scars, 600 m | 0.213 | 0.306 |
| 12 overlapping scars, 16 m | 0.508 | 0.751 |

Cold preprocessing of the ten distinct non-flat bundled fields took 524 ms
including GPU readback, CPU correction and writing the cache. The cache directory
contained 46,137,696 bytes. Subsequent benchmark processes used the cache.

The exact-intersection test compared 1,801 valid rays with zero misses and
a maximum normalized-depth error below 0.0000005. A one-proposal cap produced
1,709 deliberate fallbacks.

In the 22 m comparison view, covered pixels that attempted RCSM produced 88,421
hits and 6,345 bump fallbacks. Successful pixels averaged 10.89 cone proposals
and reached a maximum of 24. Raising the cone cap to 128 recovered all 4,370
budget-related fallbacks in that view. The remaining 1,975 rays left the texture
domain. These counts use a coverage threshold of approximately 0.045.

The terrain tests switch an existing decal from normal mapping to RCSM over a
crater in the terrain mesh and exercise normal mapping fallback on overlapping
footprints.

## Implementation files

| Area | Files |
| --- | --- |
| Atlas preprocessing and overlap clearance | `crates/ae_render/src/decals.rs`, `decals_relief.rs`, `shaders/decal_cones.comp`, `shaders/mipgen.comp` |
| Drawing and live selection | `crates/ae_render/src/terrain/app.rs`, `terrain/renderer.rs`, `shaders/terrain_decals.glsl`, `shaders/decal_relief.glsl`, `shaders/terrain.frag` |
| Client controls and measurement | `bins/ae-client/src/main.rs`, `bench.rs`, `crates/ae_render/src/window.rs` |
| Validation | `crates/ae_render/tests/gpu.rs`, the oracle in `decals_relief.rs`, `shaders/decal_relief_probe.comp`, and the updated SPIR-V pins |
