# Photometric lighting (M12)

The renderer works in physical units before M12's GGX speculars, so diffuse
and specular consume the same direct-light inputs.

## Units and defaults

- One world unit is one metre (`ae_render::lighting::WORLD_UNIT_METRES`).
- Point and spot lights carry a luminous intensity in candela (cd = lm/sr);
  the sun is a direct-normal illuminance in lux (lm/m²); emission is
  luminance in cd/m². Surfaces are Lambertian for now: outgoing radiance is
  `albedo / pi * E` for the illuminance `E` they receive.
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
  - the sun's glint is scaled by the sun's illuminance (its strength is
    calibrated to read like the old arbitrary-unit highlight at the
    reference exposure);
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

## Verification

- CPU: `ae_render::lighting` (tints keep unit luminance, EV14 is the
  reference, cull radius follows the exposure); `lights::bounds` (intensity
  versus the cutoff, authored range cap, beams, no light); `ae_gltf` (candela
  passthrough, omitted range, spot angles and direction, directional
  warning).
- GPU `photometric_sun_exposure_and_inverse_square` (validation on): the sun
  scales with its illuminance and one EV doubles the stored scene (so the
  exposure happens before storage, and the tonemap only curves); a white
  100 000 cd light adds a quarter at 4 m of what it adds at 2 m (inverse
  square, read through the clamp curve); a source radius larger than the
  range clamps the near field without extinguishing the light.
- GPU `particles_light_by_illuminance`: an opaque smoke puff lit 10 m away
  by 1 000 000 cd and by 4 000 000 cd adds 4:1 the light (at 10 000 and
  40 000 lx, both past the old packing's 1 022 lx ceiling).
- The 31 GPU tests pass with validation. A generated-map capture with eight
  explosions: under `Clamp` the calibrated EV14 output differs from the old
  build by a mean 1.8/255 (0.4% of pixels above 8, from the retuned
  explosion lights); the ACES default shifts it by a mean 24/255; the
  physical sky, fog and water (step 3) darken the mean from 177 to 139
  under ACES while bringing it closer to the old frame (mean absolute
  difference 13.7 instead of 25.8). Thresholds in the GPU image tests that
  read display values were re-baselined for both changes; none lost its
  property (the water's vision dimming now checks the water's luma rather
  than the old bridge's larger surface share).
- `ae-client --bench battle`: unchanged pass structure and timings within
  run-to-run spread; the new bench label is `m12-photometric`.

Open: speculars (M12 step 3) consume the same `E`; auto-exposure after the
bloom.
