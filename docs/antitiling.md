# Anti-tiling

Implemented for the terrain's material tilesets: `layerTexel` in
`crates/ae_render/shaders/terrain.frag`, the per-layer `AntiTiling` setting in
`crates/ae_render/src/terrain/tileset.rs`, defaults and costs in `docs/tilesets.md`. The example
it started from is kept verbatim at the end, in an older GLSL dialect (`texture2D`).

What the example does:
- Hides a texture's repetition by sampling it twice per position: rotated and offset by two
  hashed seeds. A value noise picks the seeds and blends between them.
- Blending two samples flattens the contrast. To restore it, the blend is re-centred on the
  texture's constant mean colour and rescaled by `1/sqrt(w0² + w1²)`. This is the
  variance-preserving blend.
- Triplanar weights have a constant overlap zone (`BlendZone`).
- Triplanar normals are combined with reoriented normal mapping (RNM) against the surface normal.

How the implementation differs:
- Sampled with `textureGrad`, the gradients rotated with the uv (the offset has none), so the mip
  does not jump where the seed changes.
- The tangent normals rotate back into the uv frame (the example used them as sampled, which
  tilts rotated copies the wrong way).
- Per layer: `Off`, `Shifted` (offsets only, for textures with a direction: strata, ripples) or
  `Rotated`. The path is a specialization constant, compiled in only when a layer uses it; an
  untaken path measured free.
- Seeds come from a 16-entry table (golden-ratio angles, R2-sequence offsets), not a hash and
  `sin`/`cos` per pixel: the terrain pass is bound by its arithmetic, and this saved about a
  fifth of the overhead.
- The triangle weights are sharpened about 1/2 (×2, still summing to 1), and a copy of weight 0
  is not sampled.
- The means are the layer's last mip (albedo averaged in linear light, height linearly), computed
  on the CPU and kept in the tileset table. The normals' mean is taken as flat.
- The noise is 8 features per repeat in the layer's uv, as in the example.
- Not adopted: the constant-overlap triplanar weights and RNM. The tileset keeps its pow-4
  weights and whiteout blend; the anti-tiled path does not change them.

Measurements (terrain pass, `ae-client --bench close,cliffs,overview`), of variants that were
not kept:
- Every layer anti-tiled, rotated: +47 % close, +38 % overview, +65 % cliffs. Per-layer selection
  is the lever: the defaults cost +24 % close (half the view is an anti-tiled layer), +15 %
  overview, +8 % cliffs.
- The value noise is about a third of the overhead (replaced by a constant).
- Sharpening the crossfade so one copy is sampled over most of the ground saves little (×2: a
  few %, ×8: about as much): the cost is mostly arithmetic per call, not samples.
- A noise in world space, computed once per pixel and shared by the layers: much slower
  (+81 % close), presumably from holding it across the taps (register pressure).
- Straight two-sample code without the skip: slightly slower than the loop with it.

Next steps, if the cost matters:
- Read the noise from a small tiling texture (one fetch) in place of four hashes, as Íñigo
  Quílez's "texture repetition" article does in its cheapest technique. That article (from
  memory: its URL returned 404 when checked) is the origin of this kind of method. Its
  blend factor also subtracts a fraction of the two copies' colour difference before the
  smoothstep, so the transition follows the texture's features. Worth trying against the
  variance-preserving blend.

```glsl
// ============================================================================
// Anti-tiling sampling — shared by color and normals.
// Uniforms: `float randomness` (layer count) and, per texture, a precomputed
// constant mean color: `vec4 meanColor = texture2D(tex, vec2(0.5), 15.0)`.
// ============================================================================

#define HASHSCALE1 443.8975
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * HASHSCALE1);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

float noise(vec2 p) {
    vec2 ip = floor(p);
    vec2 u = fract(p);
    u = u * u * (3.0 - 2.0 * u);
    return mix(
        mix(hash12(ip), hash12(ip + vec2(1.0, 0.0)), u.x),
        mix(hash12(ip + vec2(0.0, 1.0)), hash12(ip + vec2(1.0, 1.0)), u.x),
        u.y);
}

#define HASHSCALE3 vec3(.1031, .1030, .0973)
vec3 hash31(float p) {
    vec3 p3 = fract(vec3(p) * HASHSCALE3);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}

const float TWO_PI = 6.2831853;

vec4 texture2D_NoTiling(sampler2D tex, vec2 uv, vec4 meanColor) {
    float t = noise(uv * 8.0) * randomness * 0.5;

    vec2 ph = fract(vec2(t + 0.5, t));
    vec2 sd = floor(vec2(t + 0.5, t)) * 2.0 + vec2(0.0, 1.0);
    vec2 wt = min(ph, 1.0 - ph) * 2.0;

    vec3 h0 = hash31(sd.x);
    float a0 = h0.x * TWO_PI;
    vec4 s0 = texture2D(tex, vec2(cos(a0) * uv.x - sin(a0) * uv.y,
                                  sin(a0) * uv.x + cos(a0) * uv.y) + h0.yz);

    vec3 h1 = hash31(sd.y);
    float a1 = h1.x * TWO_PI;
    vec4 s1 = texture2D(tex, vec2(cos(a1) * uv.x - sin(a1) * uv.y,
                                  sin(a1) * uv.x + cos(a1) * uv.y) + h1.yz);

    vec4 col = s0 * wt.x + s1 * wt.y;
    return (col - meanColor) * inversesqrt(dot(wt, wt)) + meanColor;
}

vec4 sampleTexture(sampler2D tex, vec2 uv, vec4 meanColor) {
    return texture2D_NoTiling(tex, uv, meanColor);
}

// ============================================================================
// Triplanar color + normals.
// ============================================================================

vec3 TriPlanarBlendWeightsConstantOverlap(vec3 n) {
    vec3 w = n * n;
    float maxBlend = max(w.x, max(w.y, w.z));
    const float BlendZone = 0.8;
    w = max(w - maxBlend * BlendZone, 0.0);
    return w / (w.x + w.y + w.z);
}

vec4 TriPlanar(sampler2D tex, vec3 p, vec3 n, vec4 meanColor) {
    vec3 w = TriPlanarBlendWeightsConstantOverlap(n);
    return w.z * sampleTexture(tex, p.xy, meanColor)
         + w.x * sampleTexture(tex, p.zy, meanColor)
         + w.y * sampleTexture(tex, p.xz, meanColor);
}

#define NORM2SNORM(v) (v * 2.0 - 1.0)

vec3 NormalBlend_UnpackedRNM(vec3 n1, vec3 n2) {
    n1 += vec3(0, 0, 1);
    n2 *= vec3(-1, -1, 1);
    return n1 * dot(n1, n2) / n1.z - n2;
}

vec3 TriPlanarNormal(sampler2D tex, vec3 p, vec3 n, vec4 meanColor) {
    vec3 w = TriPlanarBlendWeightsConstantOverlap(n);
    vec3 an = abs(n);
    vec3 s = sign(n);

    vec3 tx = NORM2SNORM(sampleTexture(tex, p.zy, meanColor).xyz);
    vec3 ty = NORM2SNORM(sampleTexture(tex, p.xz, meanColor).xyz);
    vec3 tz = NORM2SNORM(sampleTexture(tex, p.xy, meanColor).xyz);

    tx = NormalBlend_UnpackedRNM(vec3(n.zy, an.x), tx);
    ty = NormalBlend_UnpackedRNM(vec3(n.xz, an.y), ty);
    tz = NormalBlend_UnpackedRNM(vec3(n.xy, an.z), tz);

    tx.z *= s.x;
    ty.z *= s.y;
    tz.z *= s.z;

    return normalize(tx.zyx * w.x + ty.xzy * w.y + tz.xyz * w.z);
}
```

Usage (color `tex2`, normals `tex1`):

```glsl
vec4 color = TriPlanar(tex2, p, worldNormal, tex2MeanColor);
vec3 N     = TriPlanarNormal(tex1, p, worldNormal, tex1MeanColor);
```
