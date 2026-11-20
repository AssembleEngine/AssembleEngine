# Photometric lighting (M12)

The renderer works in physical units: diffuse and GGX speculars consume the
same direct-light inputs, and exposure happens before the HDR scene is
stored.

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
  coefficient. The default tone curve is ACES; `Clamp` stays for migration
  captures and debugging.
- The exposure is applied by the scene passes before they store into the
  RGBA16F scene; `post.tonemap` only curves. Every pass that writes the
  scene (terrain, units, water, the particle modes' final writes) uses the
  same per-frame value (`FrameParams::photometry`), resolved before light
  culling.

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
- **Terrain**: a dielectric at `F0 = 0.04`. `LayerSettings::roughness` gives
  each tileset layer its specular response (grass and scrub 0.92-0.95, sand
  0.75-0.78, rock 0.50-0.52, wet beds and mud 0.45-0.48, gravel 0.82), blended
  with the same weights as the albedo. The palette path (no tileset) uses
  0.85. Decals change the albedo and the normal, not the roughness.
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
must reach every cluster it can illuminate.

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
- **Shading is linear RGB** (Rec.709 primaries). Nothing does
  `pow(2.2)`/`pow(1/2.2)` to convert colour space.
- **The scene is exposed linear RGBA16F**; `post.tonemap` applies the curve
  and writes display-referred linear values, and the output attachment is
  `*_SRGB`, so the hardware performs the sRGB OETF on store.
- **The ACES curve is Narkowicz's fit**: linear to display-referred linear,
  per channel in the input's primaries (no AP0/AP1, no D60 adaptation), an
  art choice rather than a colorimetric ACES output transform. It is
  pre-exposed (`aces(1.0) = 0.803`, about 232 in sRGB), which the EV14
  calibration relies on. A 0..1 sRGB attachment is an SDR assumption; HDR
  output (M13) needs PQ or scRGB and another ODT.
- Readback is a raw copy of the sRGB attachment, so captures are already
  display-encoded bytes.

## Verification

- CPU: `ae_render::lighting` (tints keep unit luminance, EV14 is the
  reference, cull radius follows the exposure); `lights::bounds` (intensity
  versus the cutoff, authored range cap, beams, no light); `ae_gltf` (candela
  passthrough, omitted range, spot angles and direction, directional
  warning); `models` (the material table's std430 offsets); `terrain::tileset`
  (roughness in the UBO and its range check).
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
  water's glint peak rises from 181 to 252 with a 2 000 000 000 cd light at
  its mirror; a grazing terrain pixel goes from 163 (sky only) to 232 (the
  light's diffuse) and 250 with the light at the mirror. (The waves put a
  glint wherever a normal meets a light's half vector, so the water test
  compares against no light; the flat model top pins the mirror geometry.)
- The decals' modulate threshold was re-baselined: a quarter-grey decal now
  leaves 0.63 of the ground instead of 0.51, because the specular's ambient
  floor does not scale with the albedo.
- The 34 GPU tests pass with validation. A generated-map capture with eight
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

Open: auto-exposure after the bloom; an environment map if the metals' sky
term is not enough. Two known limitations are tracked in `PLAN.md`: the sky
specular ignores roughness (a low camera at grazing incidence gets a strong
sky reflection on the ground, where a roughness-aware directional albedo
would dim it), and glTF's default metallic 1 makes content without pbr data
a rough metal.
