# Photometric lighting (M12)

The renderer works in physical units: diffuse and GGX speculars consume the
same direct-light inputs. The scene is stored at the camera's reference
exposure; automatic exposure meters it on the GPU and the tonemap applies the
eased EV (Automatic exposure below).

## Units and defaults

- One world unit is one metre (`ae_render::lighting::WORLD_UNIT_METRES`).
- Point and spot lights carry a luminous intensity in candela (cd = lm/sr);
  the sun is a direct-normal illuminance in lux (lm/m²); emission is
  luminance in cd/m². Surfaces are glTF metallic-roughness: a Lambertian
  diffuse term plus a GGX specular, with `albedo / pi * E` for the
  illuminance `E` they receive (Speculars below).
- Scene defaults (`ae_render::lighting`): 100 000 lx sun, 20 000 lx sky on an
  upward-facing surface, sun tint `(1, 0.95, 0.85)` and sky tint
  `(0.45, 0.55, 0.70)`, both normalised to unit luminance so the scalars are
  the illuminances. The tints are mirrored in `shaders/photometry.glsl`.
- Camera exposure is EV100 with `exposure = 1 / (1.2 * 2^EV100)`; the default
  is EV14 (≈ 5.086e-5), which keeps the old terrain look: a fully lit
  Lambertian surface gets `100000 * exposure / pi ≈ 1.62`, the old `1.6` sun
  coefficient. The default tone curve is Lottes 2016 (Colour below);
  `Clamp` stays for migration captures and debugging.
- The exposure is applied by the scene passes before they store into the
  RGBA16F scene; `post.tonemap` only curves. Every pass that writes the
  scene (terrain, units, water, the particle modes' final writes) uses the
  same per-frame value (`FrameParams::photometry`), resolved before light
  culling. This is the manual path (`ToneParams::auto` off) and the
  reference the automatic one is relative to.

## Automatic exposure

Eye adaptation (`ae_render::exposure`, `ToneParams::auto`) closes the HDR
loop on the GPU: the CPU never reads a value back, and a new EV takes effect
in the same frame's bloom and tonemap.

- **Metering** (`exposure.meter`, compute): the scene is sampled at a
  quarter resolution (four taps per 4×4 block, averaged), and its Rec. 709
  luminance's log2 added to one of 128 histogram bins over −8..+8 EV
  (`EXPOSURE_BINS`/`EXPOSURE_LOG2_MIN`/`EXPOSURE_LOG2_MAX`). Each 8×8
  workgroup bins its blocks in shared memory and adds only its non-zero
  bins to the global histogram, so a uniform frame does not funnel every
  count into one global atomic. The histogram is one `u32` per bin in a
  host-owned buffer, zeroed by `exposure.clear` (transfer) earlier in the
  frame; the meter declares it `storage_buffer_read_write` (atomics read and
  write). With auto off the meter returns at once.
- **The fog of war is not metered**: the terrain and the water dim
  out-of-sight ground by the vision (0.5 explored, 0.2 never seen), a look
  rather than light. The meter also samples the depth: where the frame
  shades by vision, it rebuilds each tap's world position from the depth
  and the inverse view-projection, reads the vision planes there (the same
  bilinear lookup as `terrain_shading.glsl`) and divides the tap by it, so a
  mostly unexplored frame meters as if it were in sight and the fog stays
  as dark as at a manual EV. Texels with no geometry (depth 0, reverse-Z)
  count as in sight.
- **The glow counts**: the meter also samples the glow (the light sources
  the bloom spreads) and adds it to each tap at the bloom's strength (0.3,
  0 with the bloom off), clamped at 4.0 like `bloom.downsample` but at the
  reference exposure, so the light the bloom shows darkens the image as the
  scene's own light does. It is the unblurred glow: the meter runs before
  the bloom, whose downsample reads this frame's exposure, and the blur
  spreads the light only about 20 pixels at 1280×720. The effect is small,
  since the scene already holds the same emitters at full strength: in the
  GPU test a fireball-sized patch of glowing scars darkens the surrounding
  ground by 0.2 of 255 luma more with the bloom on (154.6 against 154.8).
  An explosion's fire and smoke particles are drawn into the scene and
  metered at full weight.
- **Centre-weighted**: each block's count is weighted by
  `1 + (METER_CENTER_WEIGHT - 1) * exp(-r² / (2 * METER_CENTER_SIGMA²))`,
  `r` its distance from the frame's centre in screen heights: 4× at the
  centre, half-way down at about 0.3 heights (`METER_CENTER_SIGMA` 0.25).
  Counts are integers (4 per unit weight), so the percentiles stay `u32`
  arithmetic. In the GPU test the same patch darkens the ground to 154.8
  luma at the centre and 161.4 near the edge (159.7 and 160.8 with the
  weight set to 1).
- **Adaptation** (`exposure.adapt`, compute, one invocation): the trimmed
  mean log2 luminance — the darkest and brightest 2% skipped
  (`METER_LOW_PERCENTILE`/`METER_HIGH_PERCENTILE`), so a small glint or a
  dark corner does not set the exposure — gives the target EV100
  `scene_ev100 + (mean_log2 - log2(KEY_LUMINANCE))`: a brighter scene needs
  less exposure. `KEY_LUMINANCE` is 0.5 (pre-curve display-linear), what the
  reference scene (untextured terrain under 100 000 lx) meters at EV14, so
  adaptation settles on the calibrated look: the GPU test's auto capture
  there has display mean 0.353 against 0.366 at manual EV14. The EV eases
  exponentially in EV space,
  `ev += (target - ev) * (1 - exp(-rate * dt))` (`dt` is wall-clock time
  in the window, pauses included, so the eye keeps adapting on a paused
  frame; a fixed 1/60 s per frame in readback; capped at 0.1 s), faster
  toward a brighter scene (`TOWARD_BRIGHTER_PER_SECOND`, 1.5/s: about two
  thirds of the way in 0.7 s) than toward a darker one
  (`TOWARD_DARKER_PER_SECOND`, 0.5/s: two thirds in 2 s), and is clamped to
  10..18 EV
  (`MIN_EV100`/`MAX_EV100`). Exponential rather than a capped step: a target
  flickering one bin (1/8 EV) each frame at 60 Hz ripples the exposure by
  under 0.01 EV a frame (the unit test), where a 2 EV/s cap followed it by
  0.03. The pass declares the image `storage_image_read_write` (it reads
  the previous EV). The result, the metered mean and the target are written
  to a 1×1 `RGBA32F` exposure image; a first frame or a manual frame writes
  the camera EV instead. `MIN_EV100` is four EV above the reference sun's,
  far above a real night's EV: night scenes are meant to stay dark.
- **Where it applies**: the scene's draws store it at the camera EV
  (`ToneParams::ev100`, the reference), then `bloom.downsample` and
  `post.tonemap` multiply by `2^(scene_ev - eased_ev)`, sampled from the
  1×1 image. Manual mode writes the camera EV to the image, so the factor is
  exactly 1 and today's frames are unchanged.
- **The bloom's clamp is exposure-relative**: `bloom.downsample` applies the
  display scale *before* clamping each tap at 4.0, so the clamp is 4.0 of
  display-referred light whatever the adaptation does (`BLOOM_CLAMP`); the
  tonemap adds the already-scaled bloom. A graph without the exposure image
  binds the scene to the tonemap's (and glow to the bloom's) stand-in
  sampler and scales by 1.
- **Culling** (`lights.rs`) uses the widest exposure the adaptation may
  reach, `MIN_EV100`, when auto is on and the graph has the exposure image,
  so a light that lightens as the image brightens is never cut; manual mode
  (or a graph without the image) uses the camera EV. `ae-client --bench all`
  with auto against `--ev100 14`, same build and machine, shows no
  consistent change in `gpu_total` or `pass/lights` (the flagged scenes
  flipped sign on a rerun).
- **Cost and limits**: the meter is four samples per 4×4 block (about 58k
  counts at 1280×720, plus a depth tap each with vision) and one invocation
  for the adaptation; its `pass/exposure meter` time in the bench is about
  10–30 µs whether it meters or returns at once, i.e. the dispatch's own
  overhead (gpu-cliffs reads about 0.4 ms in both modes, the wait for the
  scene passes landing in this pass). The histogram and the 1×1 image are
  shared by the frame slots, and the frame graph derives their cross-frame
  barriers; sync validation reports no hazard. It meters whatever the scene
  shows, sky included; units drawn on dimmed ground are divided by the
  ground's vision too; the glow is metered unblurred; the centre weighting
  is fixed (no spot metering on a selection or the cursor) and there is no
  eye-adaptation curve beyond the two rates.

## Direct light

One evaluator returns the linear RGB illuminance before the receiver
cosine; the material applies `albedo / pi` and the cosine once
(`shaders/lights.glsl`):

    Lo += albedo / pi * E * max(N.L, 0) * shadow

For a point or spot:

    w = authored range > 0 ? max(1 - (d/range)^4, 0) : 1
    E = tint * intensity * w * cone / max(d^2, source_radius^2)

The distance, the denominator clamp and the culling radius are separate: the
source radius only limits the near field, and never changes the range
window. A spot uses a squared cosine ramp between glTF's inner and outer
half-angles.

The sun's input is `sunTint * sunIlluminance` (one unit-luminance tint); the
shared path applies the cosine and the shadow. Sky diffuse is
`skyTint * skyIlluminance * (0.5 + 0.5 * N.y)` — the hemispherical
approximation — with no shadow. Water reflection and fog are the sky's
radiance (below).

## Speculars

A direct light is evaluated once per light. `shaders/lights.glsl` has the
shared BRDF; the sun and the clustered list use the same code
(`brdfLight`, `lightsSurface`), and the cosine is applied once:

    H = normalize(V + L)
    a = roughness^2
    D = a^2 / (pi * ((N.H)^2 * (a^2 - 1) + 1)^2)            (GGX)
    G / (4 N.V N.L) = 0.5 / (N.L * sqrt((N.V)^2 (1-a^2) + a^2)
                           + N.V * sqrt((N.L)^2 (1-a^2) + a^2))  (Smith)
    F = F0 + (1 - F0) * (1 - V.H)^5                         (Schlick)
    spec = D * G/(4 N.V N.L) * F * N.L * E
    kD = (1 - F(N.V)) * (1 - metallic)
    diffuse = kD * albedo / pi * N.L * E

`F0` is 0.04 for a dielectric and `mix(0.04, base colour, metallic)` for a
model. The sun's `E` carries the hills' and units' shadows and the
particles' tint; the clustered lights have no shadows. Roughness is clamped
to 0.04, and `a` to 1e-4.

Without an environment map the sky is the only specular environment. It is a
uniform hemisphere, so the integral has a closed form:

    L_sky = skyTint * skyIlluminance / pi
    ambient = F(N.V) * L_sky * (0.5 + 0.5 * R.y) * ao        (R = reflect(-V, N))

(the same hemispherical approximation the diffuse sky uses). This is what
keeps a metal visible where no light glints: `kD` is 0 for a metal, so its
only light is the sun's and the lights' lobes plus this term. There is
deliberately no prefiltered IBL, split-sum LUT, multi-scatter compensation,
area light, or per-light shadow yet; the sky's radiance is a constant.

Materials:

- **Models**: glTF's metallic-roughness. The metallic and roughness factors
  and the metallic-roughness texture (roughness in G, metallic in B) are in
  the material table (`models.rs`, 288 bytes); the texture keeps its own UV
  set and `KHR_texture_transform` and is sampled as linear data. Blended
  materials are still drawn as masks at 0.5. glTF's defaults are metallic 1
  and roughness 1: a material that omits its pbr block is a fully rough
  metal (no diffuse, only the sky's and the lights' specular), which is what
  the spec asks for and a visible change for content authored before this
  step.
- **Terrain**: the ground is a dielectric at `F0 = 0.04` by default.
  `LayerSettings::roughness` gives each tileset layer its specular response
  (grass and scrub 0.92-0.95, sand 0.75-0.78, rock 0.50-0.52, wet beds and
  mud 0.45-0.48, gravel 0.82), blended with the same weights as the albedo.
  The palette path (no tileset) uses 0.85. Decals paint their own material:
  each atlas layer has a `LayerMaterial` (roughness, metallic; default
  0.9/0), blended into the ground by the decal's coverage, so soot (the
  default scorch is 0.95) takes the gloss off whatever it lands on and a
  custom layer can be glossier or metallic
  (`DecalAtlas::set_layer_material`). The layer's surface texture is RGBA8:
  normal x and y, roughness, glow (normal z is reconstructed in the shader);
  the material's roughness factor multiplies that texture, as glTF's
  metallic-roughness does. The default atlas generates a per-texel
  roughness map (charred centre rougher than the burnt ring); custom layers
  write 1 and let their factor set it.
- **Darts**: 0.04 and roughness 0.7, so the flat blocks keep a slight sheen.
- **Water**: `F0 = 0.06` (raised from water's 0.02, as before) and roughness
  0.22 to 0.43 from the unresolved wave slope — the old hand-tuned Phong
  exponents 900..60 at the same lobe widths. The sun glint is the shared GGX
  lobe with the receiver cosine, and the clustered lights get the same lobe
  and cosine. `WATER_SPECULAR_GAIN = 0.0926` is the art calibration: it is
  `0.062` (the brightness of the step's reference captures, tuned before the
  cosine was in) divided by `sin(42°) = 0.669`, the reference sun's cosine at
  the mirror, so the reference-angle glint keeps its brightness and other sun
  elevations now follow `N·L`. The sky reflection and its Fresnel term are
  unchanged.

### Decal channels

Decals are forward blends inside the terrain's shader, so their two texture
arrays are packed for the surface they paint:

| Texture | Format | R | G | B | A |
|---|---|---|---|---|---|
| `decalAtlas` | `R8G8B8A8_SRGB` | albedo.r | albedo.g | albedo.b | coverage |
| `decalSurface` | `R8G8B8A8_UNORM` | normal.x | normal.y | roughness | glow |

Normal z is `sqrt(1 - x² - y²)` (it points out of the ground). Per layer,
`LayerMaterial` supplies the roughness and metallic factors in a storage
buffer (one `vec4`: x roughness, y metallic, z specular scale reserved).
Roughness is factor × texture, as glTF's metallic-roughness; metallic stays a
factor because it is uniform per material. A third array would be needed for
per-texel metallic.

## Culling

A light without an authored range is bounded by the illuminance where it
stops mattering:

    E_cut = 0.002 / exposure          (≈ 39.3 lx at EV14)
    r_cull = sqrt(max channel of (tint * intensity) / E_cut)
    bounds = authored range > 0 ? min(authored range, r_cull) : r_cull

The bounds generation, the per-tile test and the cluster box test all use
this reach; the shading window uses only the authored range. A 1 000 cd
light reaches ≈ 5 m at EV14, a 25 000 000 cd flash ≈ 797 m (or its authored
70 m). The bound is not clamped to the map: an unbounded, very bright light
must reach every cluster it can illuminate. With automatic exposure the
bound uses `MIN_EV100` (the largest exposure the adaptation can reach), so
a brighter image never culls a light it then shows.

## Emitters, water and fog

Everything is authored in physical units; there is no reference-exposure
bridge left in the shaders.

- **Particles**: an `EffectDef` has a linear tint plus an emission in
  cd/m² over its life (FLASH about 87 000 at birth, FIRE about 20 000,
  SPARKS about 74 000). Media (smoke, dust) carry albedo instead; their
  lighting is `(sunE * sunAt + skyE * 0.5) / pi` plus the clustered lights
  over pi. The clustered illuminance is packed per particle as illuminance
  × exposure in 10 bits per channel (`x/(1+x)`, up to about 1 022 after
  exposure), so a bright light keeps its value instead of clipping at the
  old 1 022 lx ceiling. The frozen test effects use 19 661 cd/m² per old
  unit.
- **Burst lights**: `EffectLight` has a tint and a luminous intensity at
  birth in candela (FLASH 32.5 million, FIRE 3.17 million, from the old
  brightness at a quarter of their range): tuning targets, not measured
  equivalents.
- **glTF emissive**: the linear product of the emissive texture/factor and
  `KHR_materials_emissive_strength` is dimensionless; the engine reads it
  as luminance in cd/m² at `19660.8` per unit (a convention documented for
  content authors).
- **Decals**: the scar's black-body glow (about 0..1 at full heat) is
  multiplied by 19 660.8 cd/m².
- **Water**:
  - the sky reflection is physical radiance for a uniform hemisphere,
    `skyTint * skyIlluminance / pi`, with a slightly darker zenith; the
    horizon is the fog colour;
  - the sun's glint and the clustered lights' are GGX lobes from the shared
    BRDF, scaled by the sun's illuminance or the light's, and by a constant
    calibrated to the old arbitrary-unit highlight at the reference
    exposure (Speculars above);
  - foam is lit physically: sun over pi, the sky, and the clustered lights
    over pi;
  - the water body's deep and shallow in-scattered radiances are 786 and
    3 146 cd/m² (red), 3 146 and 7 471 (green), 4 719 and 8 258 (blue).
- **Fog** is the sky's radiance, and so is the scene's clear colour (both
  multiplied by the frame exposure when stored): the horizon matches the
  sky by construction.

## Glow and bloom

The bloom is fed by a glow image, never by the scene: a lit white sail or
sunlit sand does not bloom however bright it is. Every scene draw writes the
glow as a second colour attachment (the graph's `glow` image,
`B10G11R11`), and the bloom reads it at a quarter of the size.

| Source | What it writes to the glow |
|---|---|
| terrain | a decal's black-body glow only (after vision and fog) |
| units | a model material's emission only; darts write 0 |
| water | the sun's and the clustered lights' **crisp** glints only |
| particles | the emission of emissive particles only; smoke writes 0 and its coverage hides the light behind it |

Everything else writes black. The sky, terrain, units and water cover what
is behind them (blend off), the water replaces the glow under it, and the
particles use premultiplied coverage so a puff hides a glint behind it.

**Half-resolution particles.** The default particle draw is at half
resolution, so its glow cannot be recovered from the composited colour:
that buffer holds every particle's premultiplied RGBA at once, and its
alpha is the accumulated smoke coverage, not a label for “emission”. The
sorted draw therefore writes a second half-resolution `B10G11R11` target
in the same sorted order: an emissive puff (alpha 0) adds its light, a
medium writes black and masks the light behind it by its coverage. The
composite upsamples that target with the same depth-aware weights as the
colour into the graph's glow; only the accumulated coverage masks the
scene's glow behind the particles. A fire in front of smoke keeps its
bloom, smoke in front of a fire takes its share, and smoke behind a fire
does not. Every particle is either emissive or a medium
(`EffectDef::emissive`), so “alpha 0” is a type, not a fade; the 0.5/255
test is that classifier. The test-only OIT blends (`Moments`, `Slices`,
`Reference`, `Volume`, `Froxels`) do not write the per-particle glow; the
composite clears it and adds no emission (their tests run with the bloom
off).

The bloom is three quarter-size passes (`bloom.downsample`,
`bloom.blur_x`, `bloom.blur_y`, RA3-graphics' `bloom.comp`): the brightest
of four 2×2 means per 4×4 block, then an 11-tap Gaussian (sigma 2
quarter-size texels, ±5 is the 4% tail) along x and y. The result is added
to the scene before the tone curve, scaled by `ToneParams::bloom` (0.3,
RA3's `kBloomStrength`; 0 turns it off, and the client's `Z` toggles it
without changing the graph), so the curve rolls it off with the rest. The buffer is
`RGBA16F` like the scene; a graph without a glow (and its three bloom
passes) binds the scene to the second tonemap sampler and switches bloom
off. The glow input is clamped at 4.0 exposed radiance (`BLOOM_CLAMP`), as
RA3 clamps it: a sun glint many times the display's white would otherwise
bloom into a large blob.

**Water's excessive glow, and how RA3 avoids it.** RA3-graphics only sends
the crisp glints to the bloom: `glint * (1 - soft)`, where `soft` is the
unresolved-wave roughness (`shaders/ocean.frag`, "Only the crisp ones"). A
rough or distant water pixel is many glints averaged together, a sheen
rather than points of light; blooming it lit the whole sun path as one
white blob. Our water uses the same gate (`surfaceGlow = glint * (1 -
rough)`), and also removes foam, the waterline fade and fog from the glow,
as RA3 does. The gain (`WATER_SPECULAR_GAIN`) still sets the on-screen
brightness; the gate is only about whether the light is a point source for
the bloom.

Water replaces the glow under its whole surface, so a glowing decal under
water loses its bloom even where the water's colour is nearly transparent
(low fresnel). The terrain already fades a decal's glow over the first
metre of depth, and RA3-graphics' ocean replaces the glow the same way
(`outGlow` alpha 1.0 with `GlowMode::Follow`); the asymmetry with the
alpha-blended colour remains. Tracked as a TODO.

The bloom buffer holds display-referred radiance: `bloom.downsample`
applies the eased EV's scale before its clamp, so the 4.0 clamp is
exposure-relative and stays 4.0 of display light under automatic exposure
(Automatic exposure above).

## glTF

`KHR_lights_punctual` point and spot intensities are candela, passed through
unchanged; the colour is the linear tint. A light with no `range` has no
authored cutoff (`range = 0` in the model; the renderer derives the cull
radius). glTF has no source radius: a range-derived one (`range / 8`) or
`DEFAULT_SOURCE_RADIUS` (0.25 m). Directional lights are ignored with a
warning that names them (one fixed sun). Orientation is the node's local -Z
through the pose, with node scale ignored (glTF's rule). The engine's
falloff is physical inverse-square with the spec's quartic window; it
departs from the spec only in the source-radius regularisation near the
source.

The model format is version 4 (`ae_model::VERSION`), which carries
`ModelLight { color, intensity, range, inner_angle, outer_angle, size }`;
cached models of older versions are converted again.

## Colour

- **Assets are decoded by format, not by shader maths.** Base colour,
  emissive, terrain albedo and decal colour are `*_SRGB` attachments and
  textures; the sampler decodes them, and mips were filtered in linear
  light. Normal, occlusion, metallic-roughness, height and terrain normal
  textures are linear (`*_UNORM`). The one deliberate gamma-space authoring
  path is the decal shader's black-body glow (`terrain_decals.glsl`
  linearises it), because the glow mask is authored in gamma space.
- **Shading is linear RGB in the working space**, BT.709's primaries by
  default (sRGB's) or BT.2020 (`--working-space bt2020`, M13 stage 3). White
  is D65 in both; conversions are linear 3x3 matrices that keep white and
  luminance (`ae_render::colorspace`). Nothing does
  `pow(2.2)`/`pow(1/2.2)` to convert colour space.
- **The scene is exposed linear RGBA16F** at the camera's reference EV;
  `post.tonemap` applies the automatic exposure's scale and the curve and
  writes display-referred linear values, and the output attachment is
  `*_SRGB`, so the hardware performs the sRGB OETF on store.
- **The tone curve is Lottes 2016** (`x^a / ((x^a)^d * b + c)`, GDC
  "Advanced Techniques and Optimization of HDR"; M13 stage 1 replaced
  Narkowicz's ACES fit). `a` (1.0) is the contrast, `d` (1.0) the shoulder,
  and `b`, `c` are solved on the CPU from two anchors: the automatic
  exposure's key (0.5 pre-curve) displays at 0.40 linear, and 16.0 displays
  as white. With these anchors `d > 1` makes the curve fall again past the
  peak (values above it would display darker than white), so 1 is the largest
  monotone shoulder. The curve maps the max channel and scales the others
  with it, so an emitter keeps its hue; from 2.0 to 16.0 scene units a
  smoothstep-weighted share (up to 0.7) of the luminance-mapped grey joins
  in, so a fireball's core whitens while its edge stays orange. At EV14 the
  sunlit untextured ground displays at a mean of 0.366 linear (0.596 with
  the fit), and the GPU test `the_curve_places_the_sunlit_ground` pins it.
  In SDR the output attachment is a 0..1 sRGB image and the hardware
  encodes it; HDR output keeps a display intermediate and encodes it
  (`post.encode`, HDR output below).
- Readback is a raw copy of the output attachment, so SDR captures are
  already display-encoded bytes (an HDR readback is PQ or half-float).

## Working space and gamut (M13 stage 3)

Range (SDR, HDR10 or scRGB) and gamut are separate axes. The gamut axis has
two ends: the space the scene is lit in (the *working space*) and the gamut
the output is mapped to (the *output gamut*, the display's primaries).

- **The working space** is `WorkingSpace::{Bt709, Bt2020}` in
  `ae_render::colorspace`, set on the app (`TerrainApp::with_working_space`)
  and carried in `LibraryEnv`. BT.709 is the default and what every capture
  before stage 3 used. White stays D65 in both, so a conversion is a linear
  matrix from the primaries: BT.709 to BT.2020 is
  `[0.6274, 0.3293, 0.0433; 0.0691, 0.9195, 0.0114; 0.0164, 0.0880, 0.8956]`,
  the matrix the HDR10 container already used.
- **Every authored colour enters the lighting in the working space.**
  Host-side: the sun and sky tints (and so the sky's radiance, the clear
  colour, the fog and the water's reflection), light colours, particle
  effect colours, team colours, and the model lights (`units_lights.comp`).
  In the shaders: the terrain's composed albedo (tileset, palette, decals),
  the models' base colour and emission, and the water's foam and body
  colours. All are authored in BT.709 except the effects and unit looks that
  carry a `p3` or `bt2020` tag (M13 stage 4, next section); the host converts
  each into the working space on entry. Colours the host passes as tints are
  normalised to unit luminance in the working space, so the illuminance
  scalars still are illuminances.
- **The exposure meter and the curve's crosstalk use the working space's
  luminance row** (Rec. 709's 0.2126/0.7152/0.0722 or BT.2020's
  0.2627/0.6780/0.0593): the scene's stored light is measured in the space
  it is stored in.
- **The output gamut** is `Gamut::{Bt709, P3, Bt2020}`. The default follows
  the range: BT.709 for SDR, Display P3 for HDR10 (the reference display's
  primaries), BT.2020 for scRGB (an extended BT.709-relative signal whose
  container can carry a wider gamut as negative components). P3 is D65
  Display P3: R 0.680/0.320, G 0.265/0.690, B 0.150/0.060. Only the
  primaries are taken; the white point stays D65, so the compositor's own
  white-point adaptation (KWin's Night Light on the reference machine)
  is left to it. On Wayland the compositor reports the panel's primaries
  (`wp_color_management`); SDL does not, so this stays a setting until the
  client queries the compositor itself (M13 stage 5).
- **Gamut compression** (`post.encode`, mirrored by
  `colorspace::compress`): when the output gamut is narrower than the
  working space (BT.2020 content on P3 or BT.709), the colour is moved
  toward its own achromatic value, by one scale of every channel's distance
  from the grey, until the most negative channel reaches zero. Luminance is
  kept exactly, hue is kept (the channel differences from the grey scale
  together), and a colour already in gamut is returned unchanged. It is not
  a clip: out-of-gamut primaries lose chroma, not brightness. Values above
  white are brightness (an HDR emitter), not a chromaticity, so only
  negative channels count as out of gamut. The compressed colour is then
  re-expressed in the container's primaries: BT.2020 for HDR10, BT.709 for
  scRGB (negatives kept: an scRGB signal is extended, and the display shows
  what it can). At the default BT.709 working space nothing is narrower than
  the default output gamuts, so no compression runs and the map is a pure
  re-expression.
- **SDR has no encode pass**: `post.tonemap` maps the curve's result from
  the working space to BT.709 itself (and compresses into it when the
  working space is wider, with BT.709's luminance row). At the default BT.709
  working space the matrix is the identity and SDR captures are
  bit-identical to before stage 3. In HDR the tonemap leaves the display
  intermediate in the working space (its matrix is the identity), so the
  gamut map runs once, in `post.encode`.
- **Compression lifts only the negative channels.** A colour it keeps the
  luminance of can still have a channel above the output's top: BT.2020 red
  at 1.0 compresses to (1.211, 0.000, 0.072) in BT.709, and the SDR target
  clips the red at 1 (likewise at the HDR peak clamp). Not built yet: a
  chroma reduction that also brings the top channel inside.
- **The levels stay 203 nits of paper white** in the tests and defaults:
  PQ's protocol reference. SDL reports 248 nits of SDR white on the
  reference machine, and KWin sends 248 too, but that is the compositor's
  own reference, not necessarily the one the Vulkan WSI attaches to the
  swapchain. If the NVIDIA WSI sends PQ without its luminances, the
  protocol's default for PQ is 203, and encoding at 248 would be about 22%
  bright and push the peak past the panel's. M13 stage 5 measures what the
  WSI actually sends before the client chooses; until then 203 is correct
  for the tests and the headless readback, and nothing builds a gamut
  decision on 248.

## Authored colour gamut (M13 stage 4)

- **The tag** is `Gamut`, written `srgb` (BT.709, the default), `p3`
  (Display P3) or `bt2020`; `bt709`, `display-p3` and `rec2020` are aliases.
  There is no fourth value: an unknown tag in a catalogue is a parse error,
  not a silent BT.709 default (`colorspace::Gamut`'s serde knows only these
  spellings, and `particles::parse_effects` reports the file's error).
- **Effects are authored in the catalogue** `assets/effects.json` (version
  1): the engine's five effects by name, in `effect` index order (`flash`,
  `fire`, `sparks`, `smoke`, `dust`). `particles::default_effects` parses it
  once. The file is strict: an unknown effect name, version, field or gamut
  tag is rejected, and every colour a definition carries — its particle
  colours and its burst light's tint — is in the definition's gamut.
- **The conversion happens once, at upload, into the working space.** The
  particle renderer converts `color_start`/`color_end` and normalises each
  burst light's tint in the effect's gamut before converting it
  (`WorkingSpace::unit_tint_from`), so the candela scalar still is the
  light's luminance. The unit renderer converts each look's colour
  (`UnitClass::gamut`). The scene's fixed and scattered lights are generated
  in BT.709 (`Light` is the 64-byte GPU record the cluster reads); glTF
  model lights are linear sRGB by the spec, and `units_lights.comp` converts
  them. Nothing converts again on the way to the display: the HDR display
  intermediate stays in the working space, and only `post.encode` (or SDR's
  `post.tonemap`) maps it to the output gamut.
- **The fire is re-authored in P3.** Like every emissive colour its tints
  are unit luminance in their gamut (the `emission` scalar is the
  brightness in cd/m²; `emissive_colours_are_unit_luminance_tints_in_their_gamut`
  checks the catalogue, the flash's death tint being tuned at 0.978 before
  it). Its death tint is `[3.784, 0.189, 0.032]` in Display P3; uploaded
  into the default BT.709 working space it is `[4.593, 0.038, -0.054]`
  (luminance kept), a red with no BT.709 blue. The emission adds linearly,
  so the negative blue takes blue from the light behind the fire, which is
  exact in a linear basis; only a pixel whose sum ends negative is clipped,
  by the SDR target, so the SDR fire is a deeper red. An HDR10 output at P3
  carries it whole, and `p3_authored_effect_survives_post_encode` decodes
  it back to `[3.789, 0.187, 0.032]` (still outside BT.709: `-0.054` blue). The other
  four effects stay `srgb`, and an SDR frame with no explosion is
  byte-identical to stage 3's.
- **A tag is not a second working space.** It is a fact about the author's
  numbers, not a licence to map at upload: the colour is converted into the
  working space and no further. A P3 colour outside BT.709 therefore has a
  negative component in the default working space, which is meaningful: HDR
  P3 shows it and SDR clips it, exactly as other wide content behaves.

## HDR output

The output's encoding is separate from the scene's light: the scene stays
exposed linear RGBA16F whatever the output, and only the end of the chain
changes.

- **The display intermediate** (`post.encode`, `ae_render::encode`) is
  RGBA16F linear light in units of paper white, in the working space, from 0
  to `peak / paper_white`. `post.tonemap` writes it instead of the target
  when the output is HDR, and `post.encode` maps it to the target;
  `DisplayLevels {paper_white, peak}` (nits) comes from
  `LibraryEnv`/`TerrainApp::with_display`. SDR graphs have neither: the
  tonemap writes the sRGB target directly, so SDR output is byte-identical
  to what it was before HDR existed.
- **The curve's white point stays at 16 scene units; what it displays at
  rises.** `Lottes::peak_out` is the output of the peak anchor in
  paper-white units: 1 in SDR and `peak / paper_white` in HDR. The mid anchor
  stays the metered key at 0.40 of paper white in both, so the exposure
  places the key where SDR does; above it the curve is deliberately
  different, or the extra display range would go unused. Below it the
  HDR curve is not the SDR one either: with `a = 1` the solved toe is
  flatter (slope at zero `1 / c`: 1.29 in SDR, 1.00 at the reference panel's
  headroom 1.778, which the client sends as 361/203 nits; 0.95 at 441/203),
  so the darks display about 17% darker than in SDR (scene 0.1: 0.115 SDR,
  0.095 HDR; 0.25: 0.247 against 0.222) and the range from the key to paper
  white brighter (scene 1.0: 0.579 against 0.667). An emitter between the two anchors (a
  fireball is several scene units) displays above paper white on an HDR
  display. The scene's peak (16) is a property of the content — stage 1 chose
  it as the top of the scene's range — so it maps to the display's peak. The
  alternative, moving the scene's peak to 16 × headroom, spends the top of
  the display's range on scene light the content does not have and compresses
  the emitters further; scaling both anchors with the headroom would brighten
  every midtone. At `peak = paper_white` the curve is the SDR one exactly.
- **HDR10**: ST.2084 (PQ) in BT.2020 at `v * paper_white` nits, 10 bits per
  channel (`A2B10G10R10_UNORM_PACK32`). The display is mapped to the output
  gamut (P3 by default), compressed into it when the working space is wider,
  then re-expressed in BT.2020 (the container's primaries). The display's
  values are clamped at `peak / paper_white` before encoding, so no channel
  encodes above the display's peak (the curve's asymptote is a few percent
  above its peak anchor).
- **scRGB**: BT.709-relative linear light at `v * paper_white / 80` (1.0 =
  80 nits) in `R16G16B16A16_SFLOAT`; values below 0 are kept as out of
  gamut. At an 80-nit paper white the frame's values are the display's own;
  at 203 nits they are 2.5375 times. The default output gamut is BT.2020, so
  the conversion from the default BT.709 working space is a no-op and a
  negative (imaginary) value stays negative: scRGB's extended range is the
  point of the encoding.
- **The defaults** are PQ's protocol reference 203 nits and a 1000-nit peak
  for a headless app. The client measures what the WSI's own image
  description carries (M13 stage 5, next section) and uses 203 nits of
  paper white and 203 × SDL's headroom as the peak (361 nits of signal,
  which the compositor shows at the panel's 441).

### The swapchain (M13 stage 5)

- **The pair is chosen, not guessed.** `--hdr off|auto|hdr10|scrgb` (auto
  by default) resolves against the surface's offered pairs: auto takes HDR10
  (`A2B10G10R10_UNORM_PACK32` + `HDR10_ST2084_EXT`) when the surface offers
  it and SDL reports the window HDR-capable, else scRGB
  (`R16G16B16A16_SFLOAT` + `EXTENDED_SRGB_LINEAR_EXT`), else the SDR pair.
  An explicit `hdr10` or `scrgb` the surface does not offer is an error
  naming what it does. `U` toggles HDR and SDR at runtime; the swapchain is
  recreated on that, on `SDL_EVENT_WINDOW_HDR_STATE_CHANGED` and on display
  or monitor changes, and the app/graph are rebuilt when the (format, colour
  space) pair changes.
- **The measured references.** Under `WAYLAND_DEBUG=1` the NVIDIA Wayland
  WSI sends `set_primaries_named(6)` (BT.2020) and `set_tf_named(11)`
  (`st2084_pq`) with **no `set_luminances`** for HDR10, so PQ's protocol
  default reference white, 203 nits, applies. For scRGB it sends
  `set_primaries_named(1)` (sRGB), `set_tf_named(5)` (`ext_linear`) and
  `set_luminances(0, 80, 203)`: 1.0 in the signal is 80 nits, and the
  reference white is 203 nits. Paper white is therefore 203 in both.
- **The levels are the signal's, relative to its reference.** KWin 6.8.3
  maps a surface's reference white onto its output's
  (`ColorDescription::toOther` scales by the ratio of the references), so
  the signal's 203 shows at the desktop's 248. The peak is therefore the
  headroom over the signal's paper white: 203 × 1.778 = 361 nits of signal,
  shown at the panel's 441. Sending the panel's absolute 441 would map to
  539, and KWin's perceptual tonemapper (added when the mapped peak exceeds
  the output's maximum) would compress the highlights a second time.
  Nothing is encoded at 248. `--paper-white` and `--peak` override both, in
  the signal's nits.
- **`vkSetHdrMetadataEXT` after every recreate** carries the output gamut's
  primaries and D65 white, `max_luminance = peak`, `min_luminance = 0`,
  `maxCLL = peak` and `maxFALL = peak` (361 on the reference panel; a
  conservative bound: we do not measure a frame average, and no frame can
  average above the peak). The WSI forwards them into the image description
  (`set_max_cll`, `set_mastering_luminance`), so they are signal nits like
  the levels. The values are logged (`[render] vkSetHdrMetadataEXT: ...`).
- **The display path is unchanged.** The display intermediate stays in the
  working space; `post.encode` writes the target in HDR10's 10-bit PQ or
  scRGB's 16F, and SDR's `post.tonemap` writes the 8-bit sRGB target. The
  offscreen capture path always uses the SDR readback, so `--capture` does
  not show the HDR encoding.
- **Modules fall back.** A render module has no HDR support until M13 stage
  6; `--render-module` with HDR requested logs once and uses SDR rather than
  failing `check_graph` every frame.
- **Verification**: `ae_render::swapchain` (CPU) pins that the three target
  pairs are the three encodings, auto's HDR10 -> scRGB -> SDR preference,
  that an explicit mode the surface lacks is an error naming the offered
  pairs, the metadata values (P3 primaries, peak/0/peak), and that the
  default peak is the headroom over the signal's paper white (361 at
  203 × 1.778, 262 in KDE's SDR mode). With
  `AE_VK_VALIDATION=1`, 10 s windowed HDR10 and scRGB runs with `--exercise`
  each recreate 9 times with zero messages, a run through KDE's HDR switch
  recreates 5 times (recorded with the first version's absolute peak; the
  peak is now 203 × headroom, 361 in KDE's HDR mode and 262 in its SDR
  mode), and the `U` toggle recreates 5 times and rebakes 5 graphs with
  zero messages. `--hdr off` release captures are byte-identical to stage
  4's.
- **Verification**: `post_encode_writes_st_2084_codes` (a constant display
  through the pass: 100, 203 and 1000 nits are ST.2084's codes 520, 594 and
  769 exactly, and a value past the peak stops at its code),
  `post_encode_writes_scrgb_relative_to_80_nits` (203/80 scale, negatives
  kept), `hdr10_readback_matches_the_sdr_frame_at_paper_white_peak` (the
  decoded luminance matches the SDR frame to a mean of 0.0019 and a max of
  0.0032), `scrgb_readback_scales_the_display_by_paper_white` (channel-wise
  0.0011/0.0028 at both 80 and 203 nits), and
  `hdr_emitters_reach_above_paper_white_and_stop_at_peak` (an explosion at a
  203-nit paper white and 441-nit peak reaches 437.7 nits, no channel above
  439.8). The lib test `pq_codes_match_st_2084` pins the OETF and its
  inverse.

## Verification

- CPU: `ae_render::lighting` (tints keep unit luminance, EV14 is the
  reference, cull radius follows the exposure, the metered target EV and the
  two adaptation rates with their bounds); `lights::bounds` (intensity
  versus the cutoff, authored range cap, beams, no light); `ae_gltf` (candela
  passthrough, omitted range, spot angles and direction, directional
  warning); `models` (the material table's std430 offsets); `terrain::tileset`
  (roughness in the UBO and its range check); `library` (the exposure image,
  histogram and three passes go together and run in order, and the tonemap
  and bloom downsample sample the eased EV).
- The SDR path is unchanged, byte for byte: a release capture of the same skirmish
  frame from the stage-1 commit and from stage 2 compares equal (`cmp`), and it still
  does after stage 3's colour-space plumbing (the default working space's matrices are
  the identity and nothing is compressed). The `.dot` goldens did not move. In HDR the
  solve's peak equation reduces to the old expression at `peak_out == 1`.
- `ae_render::colorspace` (CPU): every gamut pair keeps white and luminance and
  round-trips in a few 1e-6; BT.709 to BT.2020 is the container matrix; P3 contains
  BT.709 and BT.2020 contains both; compression leaves an in-gamut colour exactly
  alone, brings an out-of-gamut one to a boundary channel with luminance and hue kept,
  and turns imaginary all-negative light to black; a BT.2020 tint stays unit-luminance.
- `ae_render::colorspace` and `ae_render::particles` (CPU): the tag parses as
  `srgb`/`bt709`, `p3`/`display-p3` or `bt2020`/`rec2020`, and an unknown tag is an
  error; the bundled `assets/effects.json` is the engine's five effects in `effect`
  order, with the fire tagged P3, its BT.709 death tint negative in blue and its
  luminance kept, and its burst light unit-luminance in P3; an unknown effect name,
  field or version is rejected.
- GPU `p3_authored_effect_survives_post_encode` (validation on): the fire's P3 death
  tint `[3.784, 0.189, 0.032]`, uploaded into the default BT.709 working space as
  `[4.593, 0.038, -0.054]`, decodes from an HDR10 P3 frame back to
  `[3.789, 0.187, 0.032]` (still outside BT.709, `-0.054` blue): the display intermediate kept the working space and only
  `post.encode` mapped the gamut, once. The release explosion capture
  (`--map land --units 0 --camera 6000,6000,350,0.6,0.45 --explode 12
  --capture-frames 10`) changes accordingly against 0ae663c: 5.22% of pixels differ
  (mean channel 0.249, max 63) and the brightest 200 pixels' mean goes from
  (239.5, 233.8, 158.9) to (239.6, 228.2, 102.4). An SDR frame with no explosion is byte-identical to
  stage 3's.
- GPU `bt2020_unlit_content_reaches_sdr_within_one_code` (validation on): the sky
  (the unlit clear colour) rendered in a BT.2020 working space reaches SDR within 1
  code of the BT.709 frame (worst 1, 2522 of 57 600 channels differ).
- GPU `bt2020_lit_content_differs_from_bt709_by_measured_numbers` (validation on):
  the same sunlit ground in the working spaces differs by a channel mean of 1.54
  codes, p99 2 and max 2 (46.3% of channels within 1 code): lighting multiplies each
  colour in its own space and the curve maps the max channel, so lit content is close
  but not identical, as expected.
- GPU `auto_exposure_meters_bt2020_in_its_own_space` (validation on): the same sunlit
  ground, auto-exposed in BT.709 and BT.2020, settles at display means 0.3529 and
  0.3607 (ratio 1.022): the meter uses the space's own luminance row, and the residue
  is the lit-content basis difference, not an exposure shift.
- GPU `post_encode_compresses_wide_content_into_the_output_gamut` (validation on):
  with a BT.2020 working space and P3 output, P3 green decodes to P3 (0.000, 1.003,
  -0.000) and is outside BT.709 (-0.226 in red), the in-P3 patch comes back untouched
  (0.301, 0.502, 0.201 against 0.300, 0.500, 0.200), and BT.2020 green is compressed
  into P3 (0.001, 0.955, 0.186), still outside BT.709, with its luminance and hue
  kept.
- GPU `hdr10_bt2020_working_space_maps_to_the_output_once` (validation on): the
  unlit sky through the whole HDR10 chain (203/1000 nits, clamp curve) decodes to the
  same light in a BT.2020 working space as in BT.709, worst channel 1.04% apart (about
  one PQ code). Before the tonemap left the HDR display in the working space it was
  8.74%: the tonemap mapped to BT.709 and post.encode then read that as BT.2020.
- GPU `post_encode_writes_st_2084_codes` and `post_encode_writes_scrgb_relative_to_80_nits`
  (validation on): PQ's codes at 100/203/1000 nits are exact (0 LSB), and
  scRGB is display × paper white / 80 with negatives kept.
- GPU `hdr10_readback_matches_the_sdr_frame_at_paper_white_peak` and
  `scrgb_readback_scales_the_display_by_paper_white` (validation on): with
  peak = paper white the HDR frame's decoded light is the SDR frame's
  (mean 0.0019, max 0.0032 linear; scRGB 0.0011/0.0028 per channel), and at
  203 nits it is the same frame over 2.5375.
- GPU `hdr_emitters_reach_above_paper_white_and_stop_at_peak` (validation
  on): an explosion's core decodes to 437.7 nits with no channel above the
  441-nit peak.
- GPU `the_curve_places_the_sunlit_ground` (validation on): flat untextured
  ground under the full sun at the default exposure displays at a mean of
  0.366 linear (0.596 under Narkowicz's fit), while the clamp curve reads the
  same stored scene as before (0.492), so the curve moves the display and not
  the exposure.
- GPU `auto_exposure_holds_the_key_across_sunlight` (validation on): a flat
  ground lit by 100 000 and 25 000 lx, looked at straight down, converges
  over 5 s of the readback clock to display means within 0.3% of each other
  (0.353 and 0.353, ratio 1.000), while at a manual EV14 the same change
  gives 1.73. Both are in range, so it is the eased EV and not a clamp.
- GPU `photometric_sun_exposure_and_inverse_square` (validation on): the sun
  scales with its illuminance and one EV doubles the stored scene (so the
  exposure happens before storage, and the tonemap only curves); a white
  100 000 cd light adds a quarter at 4 m of what it adds at 2 m (inverse
  square, read through the clamp curve); a source radius larger than the
  range clamps the near field without extinguishing the light.
- GPU `particles_light_by_illuminance`: an opaque smoke puff lit 10 m away
  by 1 000 000 cd and by 4 000 000 cd adds 4:1 the light (at 10 000 and
  40 000 lx, both past the old packing's 1 022 lx ceiling).
- GPU `speculars_follow_material_and_lights`: at the sun's mirror on a flat
  box top a smooth metal's highlight is white over a wide area, a rough
  metal's is not white, and a dielectric's (F0 0.04) white core is much
  smaller; with the sun off, a clustered light at the view's mirror lights
  the smooth metal by 79 luma and the same light at the opposite azimuth
  (same distance and elevation) does not, while the rough metal and the
  dielectric get no highlight.
- GPU `lights_glint_on_water` and `lights_specular_on_the_ground`: the
  water's glint peak rises from 181 to 254 with a 2 000 000 000 cd light at
  its mirror; a grazing terrain pixel goes from 163 (sky only) to 232 (the
  light's diffuse) and 250 with the light at the mirror. (The waves put a
  glint wherever a normal meets a light's half vector, so the water test
  compares against no light; the flat model top pins the mirror geometry.)
- The decals' modulate threshold was re-baselined: a quarter-grey decal now
  leaves 0.63 of the ground instead of 0.51, because the specular's ambient
  floor does not scale with the albedo.
- GPU `decals_carry_their_roughness_and_metallic`: three identical grey
  decal layers under a light at the view's mirror — final roughness
  0.15 (factor 0.5 x texture 0.3) reads 199 luma, 0.95 (1.0 x 0.95) 136,
  and a metallic 0.15 layer (F0 = its colour) 249. The frames differ only
  in the layer's material, so the albedo and normals cannot explain it.
  CPU: the surface packs normal xy, roughness and glow; layer materials
  default to matte, are range-checked, survive `resized`, and `add_layer`
  appends a default.
- GPU `bloom_spreads_a_glow_but_not_lit_ground`: a glowing decal lifts the
  ground in a ring 12–28 px around it by 7.5 luma when the bloom is on, while
  sunlit ground 80 px away changes by 0.0 (no lit surface enters the glow);
  the view's bloom switch (the client's `Z`) draws exactly strength 0.
- GPU `half_resolution_fire_in_front_of_smoke_blooms`: a fire in front of
  smoke (half-resolution sorted particles, the default) lifts a ring
  10–30 px around it by 10.7 luma with the bloom on; the full-resolution
  path gives 12.5, and smoke in front of the fire takes its share (4.1).
  Classifying the accumulated half colour by its alpha (the first version
  of the composite) gives 0.0 at half resolution.
- The 38 GPU tests pass with validation. A generated-map capture with eight
  explosions: under `Clamp` the calibrated EV14 output differs from the old
  build by a mean 1.8/255 (0.4% of pixels above 8, from the retuned
  explosion lights); the ACES default shifts it by a mean 24/255; the
  physical sky, fog and water (step 3) darken the mean from 177 to 139
  under ACES while bringing it closer to the old frame (mean absolute
  difference 13.7 instead of 25.8). Thresholds in the GPU image tests that
  read display values were re-baselined for both changes; none lost its
  property (the water's vision dimming now checks the water's luma rather
  than the old bridge's larger surface share).
- Cost (`ae-client --bench`, before/after `m12-speculars-before` and
  `m12-speculars-2`; 1280x720, median GPU time):

  | Scene | Total | Terrain | Units |
  |---|---|---|---|
  | battle | 988 → 1 059 µs (+7.2%) | 362 → 414 µs | 69 → 92 µs |
  | cliffs | 801 → 876 µs (+9.3%) | 539 → 621 µs | — |
  | lights-1000 | 823 → 930 µs (+13.1%) | 352 → 428 µs | 93 → 127 µs |
  | army-fox | 8 147 → 8 306 µs (+2.0%) | 247 → 426 µs | 2 326 → 2 314 µs |

  The terrain pays the fixed BRDF (sun and ambient specular) on every pixel;
  with many lights it additionally pays one GGX lobe per light. The first
  version used `pow` for Schlick's fifth power: `lights-200`'s terrain rose
  125% (265 → 594 µs). Five multiplies cut it to +4.3% (289 µs). The
  near-fox terrain median has a 100% spread between runs, so its delta is
  not read as a regression; `gpu_total` there moves within 1%.
- Cost of the bloom (`m12-before-bloom` against `m12-bloom-2`, the same
  four scenes): battle 1 057 → 1 078 µs (+2.0%; the three bloom passes are
  13.3 µs and the tonemap stays 16.4), cliffs 862 → 851 µs and army-fox
  8 168 → 8 265 µs (+1.2%, within the run spread), lights-1000 917 → 947 µs
  (+3.3%; its terrain median varies by more than the bloom's 14 µs). The
  scene draws pay a second colour attachment.
- Cost of the half-resolution particle glow (`m12-bloom-2` against
  `m12-particle-glow`): battle's `pass/particles` 85.0 → 151.6 µs and the
  frame 1 078 → 1 143 µs (+6.0%), the draw now writing a second
  half-resolution attachment; cliffs (853 µs), lights-1000 (954 µs) and
  army-fox (8 167 µs) have no particles in these scenes and stay within
  their run spread. The test-only OIT blends skip the second attachment.

Open: an environment map if the metals' sky term is not enough. Two known
limitations are tracked in `PLAN.md`: the sky specular ignores roughness (a
low camera at grazing incidence gets a strong sky reflection on the ground,
where a roughness-aware directional albedo would dim it), and glTF's default
metallic 1 makes content without pbr data a rough metal.
