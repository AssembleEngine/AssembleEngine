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
  coefficient. The default curve stays `Clamp` until the ACES switch.
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
approximation — with no shadow. Water reflection and fog are still bridged
(see below).

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

## Migration bridge (temporary)

Everything still authored in the old arbitrary units is converted at the
fixed EV14 reference exposure, not the frame's:

- additive emission (particles' emissive colours, decal glow, glTF
  emissive, caustics' old constants): multiply by
  `legacy_emission_scale = 1 / 5.086e-5 ≈ 19 661`;
- old punctual lights (the particle bursts): matched through the old
  attenuation at a quarter of the light's range, so the receiver there keeps
  its old brightness (`legacy_light_scale = pi / 5.086e-5`);
- water and its fog colour: the surface is still authored in the old units
  and its final radiance is scaled by `reference exposure` before the
  frame's exposure;
- distance fog: the old fog colour as physical radiance.

These go away when the emitters, water and fog are re-authored in cd/m² and
radiance (the next commit of the M12 photometric step).

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
- The 30 GPU tests pass with validation. Existing image tests were not
  re-baselined: the calibrated EV14 output matches the old frames closely
  (a generated-map capture with eight explosions differs by a mean 1.8/255,
  0.4% of pixels above 8, from the explosion lights' retuning).
- `ae-client --bench battle`: unchanged pass structure and timings within
  run-to-run spread; the new bench label is `m12-photometric`.

Open: speculars (M12 step 3) consume the same `E`; the ACES default switch
(step 2 of the landing sequence); re-authoring the emitters/water/fog;
auto-exposure after the bloom.
