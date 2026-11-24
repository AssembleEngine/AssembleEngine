# Water (M16)

M16 ports the water stack of `~/RedAlert-graphics` (the C++/Vulkan reconstruction of Red
Alert 3's look) into the frame graph. This document is the survey the port starts from:
the prototype's passes, resources, parameters and published costs, captures of the
prototype and this engine at RTS cameras, and the mapping from one to the other.

- Scope and milestones: `docs/PLAN.md` § M16 and the M16 plan.
- Engine baseline: M7 water at `8c5cbb9` (one flat quad; no reflection, refraction,
  underwater pass, displaced mesh, wakes, spray or bubbles).
- Prototype: `21c28f9`.
- Survey and captures: 2026-11-23, stage 1 (M16.3).

Status: stage 1 (survey, baseline gates, coast fixture). The engine's own pass costs are
**not** measured here; they need the user's consent and the bench scenes (M16.10, M16.21).

## Captures

All captures are 1280x720 and are stored at 800x450 under `docs/water/`. The machine is an
NVIDIA GeForce RTX 3060 Ti (device `driverVersion` `0x9b124100`, i.e. 620.73.04), Linux
7.3.5, Vulkan loader 1.4.363, device API 1.4.357, with `VK_LAYER_KHRONOS_validation`.

| Capture | Camera | Settings |
| --- | --- | --- |
| [prototype overview](water/prototype-overview.png) | eye (157.8, 390, 275.1) m, yaw −2.54, pitch −0.95, 48° fov, looking at the island | 180 frames from t = 9 s at 1/60 s; waves 1, bloom, cloud shadows and merge on |
| [prototype fleet](water/prototype-fleet.png) | eye (52.6, 130, 91.7) m, yaw −2.54, pitch −0.95 (the prototype's default) | as above |
| [prototype close](water/prototype-close.png) | eye (110, 35, 110) m, yaw −2.264, pitch −0.242 | as above |
| [prototype blast](water/prototype-blast.png) | fleet camera | bomb at world (0, 80) at t = 10.2 s, crater radius 10 m; 21 frames from t = 10.2 s (the capture is t = 10.53 s) |
| [engine overview](water/engine-overview.png) | `--camera 4300,4000,5200,0.35,0.70` (target x, z; distance; yaw; pitch) | `--map coast --map-cells 1024 --units 600 --ticks 30 --capture-frames 8`; M7 water on, shadows on, fog on, temperate tileset |
| [engine fleet](water/engine-fleet.png) | `--camera 6300,3700,1300,0.35,0.45` | as above |
| [engine close](water/engine-close.png) | `--camera 6300,3700,220,0.35,0.30` | as above |
| [engine below](water/engine-below.png) | `--camera 6300,3700,260,0.35,0.02`: the eye is about 30 m under the 42 m surface | as above |

The prototype has no camera command line and clamps its camera above the water (its
`moveCamera` keeps `cameraPos_.y >= 2.5`), so its stills come from a temporary copy of the
checkout with a `--camera X Y Z YAW PITCH` option added (option parsing plus a
`Renderer::setCamera` setter; no rendering change). The user's checkout is untouched.

The overview, fleet and close captures are the three matched pairs. The blast capture is a
standalone prototype reference: M7 has no blast ring to pair it with. The below-surface
capture is engine-only: the prototype has no such view, so its baseline stands alone and
stage 6 (M16.19) designs the look.

`engine-below.png` is the pre-M16 look from under the plane: the M7 quad has no back face,
so the surface is absent, the sky is visible, and the seabed and the ships' markers are
not refracted. That missing surface, the fog and Snell's window are exactly what M16.19
adds.

### What the captures show

- **The prototype reads at its scale.** At its default camera (130 m up, a 2.6 km ocean
  square around a small island) the shallow shelf turns from sand to turquoise to deep
  blue over some 100 m, the shore breakers are readable, hulls carry paint, and the blast
  is visible as a spray ring plus a whitewater centre. The submarines' propeller bubbles
  and their submerged hulls are only just visible at this camera; the large dark
  red-brown patch under the island is the sea bed at depth, tinted by the water.
- **The engine's M7 surface is a flat quad.** The coast's shelf colouring and the material
  patches are already there — `terrain.frag` owns the depth tint and caustics under
  `FLAG_WATER` — but the surface is one plane: no hull reflection, no bent refraction, no
  displaced crests, no wake and no ring. The close capture shows the surface's fine
  analytic ripples and the sun's glints; the fleet and close captures show how small the
  client's placeholder unit markers are at these distances, so hull readability must be
  judged after M16.8/M16.16 (models and riding hulls), not from these markers.
- **The scales differ by more than an order of magnitude.** The prototype's wave
  simulation covers 600 m and its mesh square 2.6 km; the coast fixture is 8.2 km (1024
  cells x 8 m), and the engine overview camera is 3.3 km up against the prototype's 390 m.
  Wave trains of 0.75-48 m and wakes a few metres wide are sub-pixel at the overview
  camera. This is why M16 clips wakes to a camera-centred patch with a distance cut-off
  and fades to the analytic trains outside it, rather than refining the whole map.
- **The prototype's mesh edge shows** as the square's rim in the overview capture (the
  sky beyond it). The engine's camera-centred patch avoids that only if the far water
  hands over to the M7 analytic surface; a hard patch edge would be worse than the
  prototype's.

## Prototype frame

`src/draw.cpp` (pass order ~240-490), with the resource sizes the prototype allocates for
a `w` x `h` swapchain (`src/renderer.cpp`, `src/scene_types.hpp`):

| # | Pass | Writes | Reads | Notes |
| --- | --- | --- | --- | --- |
| 1 | Wave particles | `simImage_[1]` 2048² RGBA16F, `churnImage_` 1024² RGBA16F | 65 536-slot particle ring | Additive quads; the vertex shader evaluates `pos = start + vel x age`; amplitude fades `exp(-age/4)`, clamped `min(amp, 1)`; wakes at the Kelvin half-angle (0.354 x hull speed) from bow and stern every 0.05 s. The same ring is drawn twice: first with `particle_churn.vert/frag` into the churn target (unblurred: the churn footprints are soft and wide already), then into the raw displacement (`src/draw.cpp:263-275`) |
| 2 | Effect particles | `simImage_[1]` 2048² RGBA16F | 4096-slot effect ring | Blast splash and bow spray stamp a 4x4 procedural flip-book (cavity/rim, collapse, jet) into the raw displacement only; they make no churn (`src/draw.cpp:263-275`) |
| 3 | Static displacement + blur | `simImage_[0]` 2048² RGBA16F | `simImage_[1]` | One 32x32-tile compute dispatch: separably blurs U then V in shared memory, adding two scrolling static octaves and RA3's transversal/longitudinal kernel; 600 m over 2048² = 0.29 m per texel |
| 4 | Adaptive ocean mesh | draw list and indirect args; ~38k-167k triangles | `simImage_`, error grid 256² | CBT/LEB over a 2.6 km square (`kOceanExtent` 1300); split above 1 px, merge below 0.7 px, spacing capped at 48 px; error = analytic swell `A (kL)²/8` plus measured displacement curvature |
| 5 | Shadow map | 2048² D32 | terrain, objects | One orthographic sun shadow: `lookAt` from the sun 300 m out, an orthographic box of ±235 m around the island, depth 1-750 m (`src/draw.cpp:63-69, 295-321`). The water receives it, hulls darken the water's environment term, and the caustics are scaled by it |
| 6 | Reflection | full-size RGBA16F colour + D24S8 depth | terrain and above-plane units | Mirrored camera, geometry only, sky and water excluded; detail normals break it up |
| 7 | Opaque scene | full-size RGBA16F + D24S8 | | Terrain, objects, sea floor |
| 8 | Underwater pass | `refractionImage_` full-size RGBA16F | opaque HDR colour + depth, caustics, shadow | Full screen: reconstructs world position from depth, leaves above-water/sky pixels; depth LUT tint (absorption + in-scatter), caustics projected along the sun and scaled by the shadow; submerged-unit tint and optional aids |
| 9 | Bubbles | `bubbleImage_` full-size RGBA16F | scene depth | Separate premultiplied layer of small rim-lit bubbles, drawn after the underwater pass; the ocean samples and softens it |
| 10 | Water surface | HDR scene | `simImage_[0]`, churn, reflection, refraction, bubbles, shadow | `ocean.vert` displaces by the simulation and analytic swell (`.13` clamp, eased). `ocean.frag` starts from the refracted, tinted water and adds `min(reflection, 1) * (0.3 * clamp(fresnel * 2, 0, 1))` plus `skyLum * ENV_COLOR * fresnel * 2 * shadow` (`shaders/ocean.frag:434-438`); RA3's flat `reflection * .5` is not what the prototype does, it weights the HDR reflection by Fresnel so looking straight down mostly shows the water. Churn normals (churn, not foam colour), shore foam and sun glints are added on top |
| 11 | Spray | HDR scene | | Ballistic clumps with soft-particle fades over the water |
| 12 | Post | LDR | glow | Bloom (0.3 strength, quarter res) and the tone curve |

The prototype's host also draws the sea floor's material (sand, weed, rock), a seabed
texture whose third channel is the shore travel time (Dijkstra to 8 m depth), a baked
caustic web (256²), a 64-frame wind-ripple normal array (256², 12 s loop), a foam texture
(512²) and a 512² cloud-shadow map. Cloud shadows, the sky, its bloom and the decal merge
are outside M16's committed scope.

### Parameters (prototype)

| Constant | Value | Meaning |
| --- | --- | --- |
| `kSimArea` | 600 m | Side of the wave simulation square; the camera-centred patch M16 builds on |
| `kSimTexSize` | 2048 | Simulation texture side (0.29 m/texel) |
| `kDispScale` | 5.0 | Horizontal displacement scale of the height field |
| `kHeightScale` | 8.0 | Vertical displacement scale |
| `WATER_ABSORB` | (0.22, 0.10, 0.09) x 0.5 per m | Channel absorption (inverse metres) |
| `SUB_ABSORB` | 0.6 | Submerged-unit absorption multiplier |
| `SUB_TINT` / `SUB_TINT_MAX` / `SUB_KEEP` | 4.0 / 24 m / 0.6 | Unit tint: as if through 4x the water above, at most 24 m, keeping at least 0.6 of its brightness |
| `CAUSTIC_TILE` / `CAUSTIC_STRENGTH` / `CAUSTIC_SPEED` / `CAUSTIC_FADE` | 14 m / 0.6 / 0.012 s^-1 / 9 m | Caustic web scale, strength, drift and depth fade |
| `kWaterLodTargetPx` / `kWaterLodMaxSpacingPx` | 1.0 / 48 px | Split threshold and spacing cap |
| `cbt_error` grid | 256² | Curvature/amplitude grid over the simulation area |
| `SPRAY_SOFT` | 0.2 | Spray soft-particle depth scale (m) |

### Published costs (prototype README, RTX 3060 Ti)

| Work | Cost |
| --- | --- |
| Tiled blur, both axes in one dispatch | 0.27 ms (two fragment passes: 0.60 ms) |
| Plain 2048² copy | 0.17 ms |
| Blur with the static displacement folded in | 0.36 ms (the separate pass it replaces: 0.165 ms) |
| Curvature pyramid rebuild / amortised | 0.24 / 0.12 ms per frame |
| Anisotropic caustic filtering (not used) | 0.20-0.25 ms |
| Bloom and cloud shadows, whole frame | about 0.1 ms |
| Terrain shading after the LOD change | 0.69 -> 0.15 ms |

Triangle counts at 1920x1080 (prototype README): 38k / 52k / 74k at default view with
`--waves` 0 / 1 / 2; 50k / 97k / 167k at its low oblique camera. The same capture prints
28k (overview), 43.8k (fleet) and 20k (close) at 1280x720.

## Engine baseline (M7) and where each pass goes

Today the engine's frame is: units reset/pose, terrain LOD, units cull/count, shadows,
lights, particles sim and shadows, terrain, units, **water**, particles, exposure, bloom,
tonemap, readback. The water is one pass:

| Today | Pass | Writes | Reads |
| --- | --- | --- | --- |
| Water surface | `water.draw` (graph name `water`) | `scene` (load), `glow` (load) | `depth` (read-only), `shadows`, lights tables |

`water.vert/frag` draw one map-sized quad at the water level. `water.frag` builds normals
from eight analytic sine trains (0.75-48 m, angles 0.2-1.4 rad, slopes 0.05-0.08) that
fade to a roughness term at distance and pixel size, adds Fresnel sky, sun glints (GGX
through the M12 light path), distance fog and shore foam, and blends over the opaque
scene with the depth test deciding coverage. Under `FLAG_WATER`, `terrain.frag` owns the
depth tint and caustics for the seabed (`terrain.frag:422-444`).

M16's pass map, from the architecture's §2.2 (all new names are graph resources, and all
barriers are inferred):

| Prototype pass | Engine pass | Tier flag | Writes |
| --- | --- | --- | --- |
| Wave and effect particles | `water.emit` | `water_waves` | `water.particles` (library ring buffer) |
| Particle displacement stamps | `water.splat`, `water.churn` | `water_waves` | `water.raw` (RGBA16F camera patch), `water.churn` (RG16F) |
| Static displacement + blur | `water.blur` | `water_waves` | `water.waves` (height + normal) |
| Adaptive mesh | `water.lod` | `water_waves` | `water.mesh.*` (indirect list) |
| - | `water.shore` | data | `water.shore` (R16F travel time), `water.caustics` |
| Reflection | `terrain.cull_reflect`, `units.reset_reflect`, `units.cull_reflect`, `terrain.reflect`, `units.reflect` | `water_reflect` | `water.reflection` (scene format, scale 0.5) + depth |
| Underwater pass | `water.underwater` | `water_refract` | `water.refraction` (scene format, full size) |
| Bubbles | `particles.underwater` (the split's below-water run) | `water_refract` | `water.refraction` (load) |
| Water surface | `water.draw` (RICH adds the underside) | tier-specific variants | `scene`, `glow`, depth read |
| Spray | `particles.draw` (above-water run) | `water_waves` | as today |
| Underwater view | `water.volume` | `water_volume` (requires `water_refract`) | `water.volume.out`, which the meter and tonemap read in place of `scene` |

Ownership is decided by the baked graph, not the flags: `terrain.frag` loses the tint and
caustics only in a graph with `water.underwater`; `water.draw` samples refraction only in
a graph with `water.refraction`; a v3 module, `render/post` or `render/legacy` therefore
keeps M7 water byte for byte. The M7 shaders never read `WaterLook`, so their SPIR-V and
frames stay identical (`tests/spirv.rs`, `tests/gpu.rs` `m16_baseline_*`).

## Contracts that shape the port

- **One owner per effect.** The tint and caustics belong to `terrain.frag` in the M7 path
  and to `water.underwater` in the refract path, never both. The surface's reflection and
  refraction sampling belongs to `water.draw`'s RICH variant.
- **Visibility.** Every copied target draws from the observer's view stream only: blips,
  ghosts and explored-only cells contribute nothing (strict everywhere; the shadow pass
  included). A reflection of a ghost would be a lie. Hidden-unit frames must be
  byte-identical to frames with the unit removed (criteria 12-14).
- **Determinism.** All water animation is view-side: emitters come from the view stream
  and impacts from the client's water feed. No water state enters a sim digest, and no
  render pass stalls a tick.
- **Single scene transform owner.** All passes consume the graph's depth and colour with
  declared roles; the copies are graph resources so barriers, lifetimes and resize follow
  the executor, not hand-written transitions.
- **Data.** Absorption, wind, swell, foam, caustics and the aids move to
  `MapInfo.water` (`WaterLook`) with defaults equal to today's constants; old archives are
  byte-identical (`map.json` gains no field). The M7 path ignores the settings on purpose.

## The coast fixture map (M16.2)

`ae_nav::scenario::coast_terrain(cells)` is deterministic (`COAST_SEED`) and 8 m per cell:

- a north-south coastline with a sand beach;
- a 0-4 m shelf with sand, rock and weed (forest) patches;
- a drop to a basin 30-34 m deep;
- a river entering at the north edge, bending east and opening into the sea at a mouth,
  with one ford (1.5 m) a land unit can cross;
- water at 42 m (`COAST_WATER_LEVEL`), start points on land (`coast_starts`).

`ae_services::nav::Map::Coast`, `ae-client --map coast`, and the GPU test
`coast_fixture_renders` use it. `ae_map` exports and loads it bit-identically (nav digest
included). The render-side GPU tests take the same generator's terrain through
`Heightmap::from_cells`, so the fixture the shaders see is the fixture the client loads.

## Water settings and the data bake (M16.4, M16.5)

A map package may carry a `water` object in `map.json` (`ae_map::WaterLook`): absorption per
channel, the shallow and deep in-scattered radiance, the wind's direction and speed, the
dominant swell's amplitude and length, foam, the caustic scale, the submerged-unit limits
(`sub_tint`, `sub_keep`, `sub_tint_max`) and the outline aid. Every field is optional and
defaults to the M7 frame's values (the table test in `crates/ae_render/src/water/params.rs`
ties each default to its source constant); an archive without the object loads and
re-exports byte for byte. The client converts it into `ae_render::water::WaterParams` field
by field. The default client tier is `m7`. Select `off`, `reflect`, `refract` or `full`
with `--water-quality`.

Two baked inputs exist for the rich tiers, made when a graph asks for them and neither
sampled by M7:

- **Shore travel time** (`water.shore`, R16F, one texel per ground height sample): a jump
  flood from the waterline, baked on the GPU. A texel holds the seconds a swell crest
  takes to reach the shore: zero at the waterline and rising seaward, its distance taken
  over the shallow-water speed at the 1 m break depth (`0.6 sqrt(g * 1 m)`). The seeds are
  the water texels with a land neighbour, stored as integer texel coordinates (RG16UI):
  exact at the 2560-cell default and the 8192-cell archive limit, where a half-float
  centre would already round. The `water.shore` pass is conditional: it records only after
  the ground changed (the app marks it from a terrain update's cells) and on the first
  frame. On an edit the field is rebuilt in full — a new waterline can be the nearest shore
  for a wedge of the sea, so a partial rebuild would be wrong beyond the edit — and the
  values outside the new shore's influence are unchanged; the GPU test raises an island in
  the river channel and checks that only its neighbourhood moved. There is no offline bake,
  and the field follows every live edit.
- **Caustic web** (`water.caustics`, RGBA8 256² with the full mip chain): a deterministic,
  tileable pattern with integer wave periods, computed on the CPU and uploaded once through
  `upload::image_mipped`, whose GPU mip generator makes its chain. Being uploaded already,
  it starts shader-read-only and its first graph read keeps it (the graph executor starts
  an external image at the layout its binding declares, so the upload is not discarded).

Both are `library.rs` resources. A graph declares them as External images with the library's
format and size; a graph file or module may instead name them as `library` resources
without a descriptor and the host fills it in. `water.shore` is a compute pass; the caustic
web has no pass (it is made with the renderer, like the terrain's textures). A scene
without water cannot declare them.

`TerrainApp::with_water_data` requests both images for the native graph. The default
refract graph requests the caustic web. A graph
that declares either image makes the app allocate the data before wiring, and a graph that
stops declaring them drops it. The M16.1 frame gate covers M7, water-off and dry scenes.

## Refraction and submerged units (M16.9–M16.14)

`water.underwater` samples the opaque HDR scene and reverse-Z depth. It reconstructs the
world point and writes a full-resolution `water.refraction` texture. Wet pixels carry
alpha 1. The sky and above-water geometry clear to zero. `water.draw` increases the
sample offset with seabed depth and blends back toward the centre sample at mixed wet/dry
footprints and sharp colour changes. `full` also samples the planar reflection.

The underwater pass linearly interpolates a per-channel absorption table at one-metre
intervals from 0 to 128 m. Sun-projected caustics fade with depth and use the terrain and
unit shadows. The refractive terrain variant supplies mipped sand ripples and grain, weed
and pebble layers. Submerged geometry marks its scene pixels with alpha 0.5. The pass
applies `sub_tint`, floors transmission at `sub_keep`, caps the tint path at `sub_tint_max`
and adds a soft outline when requested by the map.

Crossing's coast places submarines among its ships. Enemy submerged hulls require sonar
coverage. Radar-only, explored-only and ghost rows contribute no geometry, lights or
shadows to the water copies. Blips and ghosts draw over the surface in their own pass.
Submerged effects use sonar and surface effects use line of sight. Switching allyteams with
Y clears pending impacts and particle bursts, retires live particles and starts fresh
explored and ghost history for the new observer.

```sh
cargo run -p ae-client -- --map coast --map-cells 768 --units 200 \
  --water-quality full --camera 4820,2520,550,0.4,0.54
cargo test -p ae-client --test water_visibility --locked -- --ignored --test-threads=1
```

Add `--teams 2 --observer 0` to exercise sonar and observer switching. The client
visibility test compares both named water images byte-for-byte with the target unit
removed, then checks the positive sonar and ownership cases.

## Surface dynamics (M16.15–M16.18)

`full` adds a 1024×1024 world-space wave patch whose footprint grows with the
camera distance, from 300 m (the prototype's 0.29 m texel) to 1600 m across.
The patch follows the view and fades into the analytic swell at its edges.
Moving floating units in the current observer stream drop wave particles;
eligible impacts drop a blast's. GPU passes stamp them into `water.raw` and
`water.churn`, and a tiled blur adds the static displacement and the swell
and writes slopes and height to `water.waves`. Delivery 4 (below) replaced
the first version's ribbons and single sine with the prototype's model.

The surface has its own conforming concurrent binary tree (CBT) and uses
terrain's longest-edge bisection and reduction shaders. It refines the
patch using projected displacement error and a spacing bound. The
surface, refraction mask and floating hulls sample the same height field.
The close-camera probe checks hull motion between a swell crest and
trough and verifies that the crest leaves the visible hull clear.

Moving visible ships emit bow spray. Moving submarines emit propeller
bubbles when owned or visible to sonar. An impact can produce surface
splash or submerged bubbles. Sorted and packet blends split the visible
particle list at the displaced waterline. Surface bursts keep emissive
spawn centres above the water. Emissive particles use their centres to
choose a layer so a surface explosion keeps the flash intact. Submerged
flashes fade out toward the waterline. Other particles that intersect the
surface go into the submerged layer. The underwater draw writes exposed
refraction colour with the wet mask intact at either particle resolution.
Above-water particles draw over the surface.
The moments, slices, froxels, reference and volume modes each use a single
above-water layer.

Loss of sight stops new wakes and bubbles from the contact. Effects
already emitted decay over their own lifetimes. Switching observers clears
pending effects and retires the live ones. Client GPU probes exercise
hidden ships and submarines, radar and ghost states, the sonar transition,
observer switching and the two particle draws.

## Underwater view and quality (M16.19–M16.20)

The volume pass tests each near-plane pixel against the displaced waterline. Its entry
blend and the surface's near-plane fade smooth the crossing. The underside shows light
transmitted inside Snell's window and total internal reflection outside it. The reflected
colour uses nearby wet refraction samples and a colour between the deep and shallow
water, so the mirror beyond the window still reads lighter than a look down. The final pass
uses distance-based absorption and in-scatter to shade wet geometry, bubbles and the
surface. Light comes from above: the in-scatter is brighter looking up and darker looking
down, and it and the geometry under water darken with depth (to a quarter of daylight in
the deep), so the view keeps a sense of up and down. Rays that start in air copy the scene
through exactly. The volume image feeds both exposure and tonemap.

The client keeps the camera's eye out of a 2 m band around the water level, which the
displaced surface can reach (the swell lifts it up to about a metre, the patch is clamped
to ±4 m): an eye in it moves to the nearer edge, or above the
water when the seabed leaves no room below. Without it the swell carried an eye at the
surface in and out of the water every few seconds.

The client defaults to `full` on maps with water. `--water-quality m7` selects the analytic
surface, `reflect` adds planar reflection, `refract` adds the shaded seabed and submerged
hulls, and `off` removes the surface. Unsupported rich formats or usages trigger an M7
fallback. O toggles the installed surface. The camera orbits a point on the ground, which
under water is the seabed. With the volume on, F (or a right-drag up) tilts the camera past
horizontal, down to 4 m above the seabed. Over water the wheel zooms in to 5 m from the
orbit point instead of 20 m, so zooming in carries the camera down to the seabed. Page Up raises
the orbit point off the seabed and up through the water, Page Down lowers it, and Home
returns it to the ground; P prints the camera with that height. To look up at the surface
from below, raise the orbit point above the water and tilt down. The optional sixth
`--camera` value sets the same height from the command line:

```sh
cargo run --locked -p ae-client -- --map coast --map-cells 768 --units 120 \
  --camera 4816,2800,22,0.1,-1,57.5
```

## The prototype's pieces in `full` (M16 deliveries 4 and 5)

Delivery 3's `full` tier had the passes but cut-down contents: one sine of swell, wakes
stamped as ribbons that lived 1.8 s, a 3×3 blur, foam only where the water was under 1.6 m,
a sky reflection that turned the sea milky, and explosions on the water that used the land's
fireball. Delivery 4 ports the prototype's pieces themselves (`~/RedAlert-graphics`:
`ocean_waves.glsl`, `wave_particle.glsl`, `blur.comp`, `particle_churn.*`, `ocean.frag`,
`scene.cpp` `emitBoat` and `emitBomb`). The port is not a match for the prototype's look:
its constants assume a 640 m island seen from 100-300 m, and several of them had to change
for an RTS map (below). Delivery 5 changes the seabed, the underwater pass and the caustic
web, which the `refract` tier shares; the M7 and `reflect` frames are unchanged.

- **Swell** (`water_swell.glsl`): the prototype's three trains, each A((1 + sin)/2)²,
  turned so the first runs with the map's wind, scaled from its 105 m train to the map's
  swell length and from its default sea (0.4 m) to the map's amplitude, and depth-limited
  near the shore. The blur adds it to the patch, where floating hulls ride it; beyond the
  patch the surface takes its gradient.
- **Static displacement**: two ±75° scrolling copies of a tileable fBm height (RA3's
  `StaticDisplacement`), added as the blur loads its tile.
- **Wave particles** (`waves.rs`, `water_splat.*`): spawn records evaluated in closed form
  on the GPU. A hull drops a Kelvin pair at the stern drifting out at 0.354 of its speed,
  a faster bow pair and a centre trail of propeller wash; the amplitude fades as exp(−age/4). A blast drops a splash stamp (the prototype's crater
  flip-book, evaluated: cavity, collapse, jet), three ring waves of 200, 140 and 100
  particles that run out at their wavelength's speed for 15–39 s, and a churned boil whose
  whitewater comes as the thrown water falls back. Records live in a host list of up to
  65,536 (at most about 2,000 per hull at steady state, 537 per blast).
- **Blur** (`water_blur.comp`): RA3's 21-tap raised cosine along U then V on a 32×32 tile
  in shared memory; the derivative taps give the slopes from the same pass. RA3's soft
  limit (linear to 0.09, easing to 1.6 raw units) and 8 m per unit; the patch is clamped
  to ±4 m (`WAVE_BOUND`). The longitudinal (horizontal chop) term is not ported: the surface
  moves vertically only, so the other consumers of `water.waves` keep its layout.
- **Surface** (`water_ocean.glsl`, under `WATER_MESH` in `water.frag`): the waves' slope
  (2.2 times for the simulated part, as RA3's normal takes it), two ±55° octaves of a 64-frame
  12 s loop of wind ripples, fine ripples and the churned water's facets from the static map
  with stochastic tiling, growing and easing as the churn ages; churned water lightens where
  its facets face the sun while it is young (its bubbles rise out within seconds). The composition is the prototype's: the sky's luminance through a
  dark teal (`ENV_COLOR`) weighted by a linear Fresnel floored at 0.1 and by the shadow, plus
  the planar reflection at 0.3 of twice that, added over the refraction.
- **Foam**: the swell's crests breaking on the shore, timed by the shore field's travel time
  (the default graph's waves tier now bakes and samples `water.shore`); the simulated waves
  breaking in the shallows; an impact's whitewater, breaking into lace as it thins.
- **Explosions on the water**: a shell or bomb landing in the water (X, `--explode`) throws a
  column of spray (`water_column` in `assets/effects.json`) and a crown, with the blast's
  crater, rings, boil and whitewater on the surface; no fireball and no scorch. A unit dying
  on the water keeps its fireball and also rings the surface.
- **Detail textures** (`water/detail.rs`): the static map (512²), the ripple loop (256² × 64)
  and the foam (512²), generated on the CPU once per process as the prototype does, mipped by
  `upload::image_mipped` (averaged: the shaders expect coarser normal mips to be flatter).
- **Riding hulls**: the shadow casters and the mirrored draw ride the waves as the draw does
  (`RIDE_WAVES` variants; the shadows and units-reflect passes sample `water.waves`), and the
  culls widen a floating row's sphere by `WAVE_BOUND`.

Where an RTS map differs from the prototype's island, the port departs from it:

- Hulls cross metres per tick (Crossing's ships about 70-110 m/s against the prototype's
  boats' 6), so wake records are laid along the path at most 1 m apart with the prototype's
  amplitude per metre of path; emitting once per tick left gaps wider than the blur.
- A wake is a length, not a time: the arms reach 20 hull radii behind the hull and the trail
  50 (the prototype's boats' 10 s and 24 s), so a record lives that length over the speed,
  never longer than the prototype's. With the prototype's lives a convoy at 100 m/s left a
  lane kilometres long. Twelve coast ships (radius 3-5 m, 87 m/s on average) peak at 8,809
  live records over a 20 s capture, against 42,699 with the prototype's lives. Hulls within
  half a patch of the camera's patch emit too: before, a hull outside the patch emitted
  nothing, so a pan that let a convoy leave and re-enter it left a hole in the trail.
- The trail's records each drift a little sideways, start a different size, spread at their
  own rate and live a different share of the trail; their churn is 0.015 per record (the
  prototype's 0.05 stays saturated a dozen records deep, which read as a ruler-straight band
  with a cut end), and it calms over half the record's life when that is under 9 s. Only
  young churn lightens: a convoy's overlapping trails otherwise laid a flat pale sheet.
- The coast's beaches are nearly flat at 8 m cells: the breakers also run through the last
  10 s of the crests' run in (about 30 m) wherever the water is under 4 m, and the swell's
  run-up ends 0.15 m above the rest level (the prototype's metre flooded tens of metres of
  sand in tongues). The crest bands are wider and the shore foam stronger than the
  prototype's, so they read from an RTS camera.
- The blast's whitewater is lacier than the prototype's dense sheet, which reads as a blob
  against the darker water.
- The patch's texel grows with the camera distance (0.29 to 1.6 m): a particle keeps at least
  1.5 texels and its volume, so wakes widen rather than vanish from afar.
- The caustics fade out between 12 and 5 pixels a thread: from RTS heights the threads land
  a few pixels apart, where the two drifting copies read as a regular weave. The caustic web is jittered
  cells with warped edges (it was crossed integer sines, a visible lattice from any height),
  and the underwater pass takes the two copies' mean, as the prototype does.
- The coast's shelf is 0.8-4 m deep and 700 m wide, where the prototype's clear water hides
  nothing: the coast look absorbs about three times as fast (0.42, 0.15, 0.13 per metre) with
  a greener, brighter in-scatter. The sea floor (`terrain.frag` `WATER_REFRACT_OWNER`) is
  sand with weed beds tens of metres across in a turned domain, rock only on slopes, its
  textures blurred by about a mip per metre of water, and it keeps only the floor's own
  normal (the beach sand's 3.6 m ripple normals had striped the whole shelf).
- The light the water scatters toward the eye is fogged with the scene under it; it was
  added after the fog, so from afar the sea stayed saturated while the land greyed.
- The terrain refines where it crosses the water level, in every tier. Its LOD discounts a
  height error seen from above (the error projects to the screen with the sine of the view
  angle, floored at 0.15), but at the waterline the same error moves the shoreline sideways:
  the coast's river, 2.8-4.2 m deep between banks half a metre to several metres high,
  showed its bed as triangles of sand above the water and the water over its banks, and
  from the whole map it read as a dry road. Nodes whose height range comes within 1 m of
  the water level take their error undiscounted and doubled (`cbt_update.comp`). From afar
  a channel narrower than the sub-mesh's spacing still fell between its vertices, so where
  such a node is off by more than 0.25 m, half its spacing also counts as shoreline error.
  On the coast (768 cells) the terrain goes from 105 to 1,506 leaves seen from 2.6 km (41
  to 1,153 drawn), from 24 to 961 from 7 km, and from 280 to 1,577 from 700 m; not timed
  yet.

The coast fixture takes the prototype's sea adapted to its shallow shelf
(`ae_map::WaterLook::prototype`: the absorption and in-scatter above, the prototype's swell
and caustic tile) unless a map package sets its own; the look the M7 constants reproduce made the full tier's
water milky. Crossing's coast now runs a shipping lane: one hull in ten, a third of them
submarines, gather off the shelf at the southern end and sail north along the coast about a
kilometre off the beach (`ae_nav::scenario::coast_plan_with`; the coast plan's digest is
re-pinned).

| Live capture | Command |
| --- | --- |
| [shipping lane](water/engine-d4-lane.png) | `--map coast --map-cells 768 --units 120 --ticks 60 --capture-frames 30 --capture-seconds 6 --camera 3500,5300,380,0.35,0.85` |
| [shore](water/engine-d4-shore.png) | `--map coast --map-cells 768 --units 0 --capture-frames 30 --capture-seconds 2 --camera 2500,3400,700,-1.57,1.0` |
| [blasts, 1.2 s in](water/engine-d4-blast.png) | `--map coast --map-cells 768 --units 0 --capture-frames 10 --capture-seconds 1.2 --explode 3 --camera 4300,2400,160,0.35,0.85` |

The GPU costs of the new passes (the blur's larger kernel, up to 65,536 particles drawn
twice, the surface's extra texture reads) are not measured yet: they wait for the benchmark
run with the user's consent, like the rest of M16.21.

## Final coast captures (M16.21)

| View | Prototype | Engine M7 | Engine full |
| --- | --- | --- | --- |
| Overview | [overview](water/prototype-overview.png) | [overview](water/engine-overview.png) | [overview](water/engine-final-overview.png) |
| Fleet | [fleet](water/prototype-fleet.png) | [fleet](water/engine-fleet.png) | [fleet](water/engine-final-fleet.png) |
| Close | [close](water/prototype-close.png) | [close](water/engine-close.png) | [close](water/engine-final-close.png) |
| Basin, looking across from depth |  | [below](water/engine-below.png) | [below](water/engine-final-below.png) |
| Looking up from 3 m below |  |  | [Snell's window](water/engine-final-window.png) |

The first four engine comparisons reuse the M16.3 cameras, map settings and frame count. The
overview, fleet and close frames were captured again after delivery 4; the coast's ships now
sail the lane, so the fleet and close cameras show open water.
The last uses the `water-under` camera above, with 30 ticks and eight frames.

The engine's pass costs, the reflection scale comparison, and the water mesh's triangle
counts and pixel error are pending consent for the four coast benchmark scenes
(`water-overview`, `water-fleet`, `water-close`, `water-under`). Budget sign-off follows
those measurements. The current settings use a half-resolution reflection and an adaptive
CBT surface.

## Reproducing the captures

Engine (from the repository root; the client writes PPM):

```sh
cargo run -q -p ae-client -- --map coast --map-cells 1024 --units 600 --ticks 30 \
  --water-quality full --capture /tmp/overview.ppm --capture-frames 8 --camera 4300,4000,5200,0.35,0.70
```

A plain capture renders one frozen stream, so it shows no wakes and only the first frames
of a ring. `--capture-seconds S` keeps the sim running for S seconds of game time after the
settling frames (two frames per tick, the water clock pinned to the ticks). The coast's
three ships start near (4892, 2413):

```sh
cargo run --release -p ae-client -- --map coast --map-cells 768 --units 120 --ticks 150 \
  --capture-frames 30 --capture-seconds 1.5 --camera 4892,2413,200,0.35,0.95 --capture /tmp/ships.ppm
```

The prototype (a checkout with the temporary `--camera` option described above):

```sh
./build/redalert-graphics --headless --width 1280 --height 720 --frames 180 --time 9 \
  --camera 110 35 110 -2.264 -0.242 --out /tmp/close.png
```

The checked-in images are those captures resized to 800x450 with optimized PNG encoding.
The baseline gates that pin the engine's pre-M16 frames and shader binaries are
`crates/ae_render/tests/data/m16_baseline.txt` and `tests/data/m16_spirv.txt`; see
`tests/gpu.rs` (`m16_baseline_*`) and `tests/spirv.rs`. The shader gate pins the compiler's
identity with the binaries: on the pinned toolchain a hash difference is shader drift and
fails, on a different supported compiler it is advisory (with the files printed), and
`AE_STRICT_SPIRV=1` requires the pinned toolchain.
