# Anti-tiling

Implemented for the terrain's material tilesets: `layerTexel` in
`crates/ae_render/shaders/terrain.frag`, the `Tileset::anti_tiling` degree and the
per-layer `directional` flag in `crates/ae_render/src/terrain/tileset.rs`, defaults and costs in
`docs/tilesets.md`. The example it started from is kept verbatim at the end, in an older GLSL
dialect (`texture2D`).

What the example does:
- Hides a texture's repetition by sampling it twice per position: rotated and offset by two
  hashed seeds. A value noise picks the seeds and blends between them.
- Blending two samples flattens the contrast. To restore it, the blend is re-centred on the
  texture's constant mean colour and rescaled by `1/sqrt(w0² + w1²)`. This is the
  variance-preserving blend.
- Triplanar weights have a constant overlap zone (`BlendZone`).
- Triplanar normals are combined with reoriented normal mapping (RNM) against the surface normal.

How the implementation works (Íñigo Quílez, "texture repetition", 2015, technique 3; the
example above is a variant of it):
- The tileset has one `anti_tiling` degree in 0..1 (the example's `randomness`): 0 samples each
  layer once, 1 uses 8 versions. The degree is continuous, not a switch, and it is the only
  control; the shader always carries the path, so no specialization constant or second pipeline
  exists.
- A smooth noise times the degree's number of versions is an index into versions of the texture,
  each moved by a seed: offset, and rotated unless the layer is `directional`. The degree also
  fades each version's transform in (rotation and offset scaled toward zero), so small degrees
  sample nearly the plain uv. Each pixel shows the two versions its index lies between, mostly
  just one.
- The noise is one fetch from a generated tiling texture (`tileset::anti_tiling_noise`: 256²
  `R8_UNORM`, gradient noise with octaves offset from one another so their lattice zeros do not
  line up into a grid, values ranked onto 0..1 so every version covers about the same share).
  It is read at 1/48 of the layer's uv: features span about 3 repeats.
- Two versions blend only in the middle fifth of each index step (0.4..0.6; Quílez: 0.2..0.8),
  the transition pushed by up to 0.05 by the versions' colour difference (Quílez's
  `f - 0.1*sum(cola-colb)`), so it follows the texture's features. Outside the band one version
  is sampled. The band is narrow because the second version is most of the cost.
- The blend is re-centred on the layer's mean (albedo averaged over mip 0 in linear light, height
  linearly; in the tileset table) and scaled by `1/|weights|`: the example's variance-preserving
  blend. The normals' mean is taken as flat.
- Sampled with `textureGrad`, the gradients rotated with the uv (the offset has none), so the mip
  does not jump where the version changes. The tangent normals rotate back into the uv frame
  (the example used them as sampled, which tilts rotated copies the wrong way).
- Seeds come from a 16-entry table (golden-ratio angles, R2-sequence offsets).
- A layer's `directional` flag shifts its copies instead of rotating them, for textures with a
  direction (strata, ripples). Others rotate.
- Not adopted: the example's constant-overlap triplanar weights and RNM. The tileset keeps its
  pow-4 weights and whiteout blend.

Measurements (terrain pass, `ae-client --bench close,cliffs,overview`, 1280×720, RTX 3060 Ti):
- Defaults (every layer at degree 1, so the cliffs include the shifted rock): close +45 µs
  (+24 %, 186.0 → 231.3), overview +57 µs (+24 %, 239.6 → 297.0), cliffs +228 µs (+45 %,
  503.0 → 730.7). Against the same tileset at degree 0.
- Earlier, with rocks off: close +49 µs (+32 %), overview +62 µs (+31 %), cliffs +136 µs
  (+36 %). Turning the rock layer on is most of the cliffs increase.
- With Quílez's band (0.2..0.8) on the ground layers: +46 %, +43 %, +51 %. One version only (no
  blend, an experiment): +17 %, +20 %, +17 %, so the noise fetch and a moved sample cost about
  half, the second version the rest. Branch-free (always two versions): no faster than with the
  early returns.
- The first implementation, the example as given (a value noise of 8 features per repeat
  computed per call, two seeds crossfaded everywhere): on grass and forest floor it cost too much
  to leave on, and with them off it did nothing where tiling shows most. Its noise was about a
  third of its cost. A world-space noise shared by the layers, computed once per pixel, was much
  slower (+81 % close), presumably from register pressure.
- Not measured yet: Quílez's techniques 1 (four samples per tile corner) and 2 (Voronoi, nine
  samples); both cost more samples than technique 3.

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
