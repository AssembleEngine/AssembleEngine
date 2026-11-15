# Anti-tiling (future work)

A planned improvement to the terrain's material tilesets (`crates/ae_render/shaders/terrain.frag`,
`docs/tilesets.md`). The example below was supplied as the reference to start from. It is kept
verbatim, in an older GLSL dialect (`texture2D`); it is not compiled anywhere yet.

What it does:
- Hides a texture's repetition by sampling it twice per position: rotated and offset by two
  hashed seeds. A value noise picks the seeds and blends between them.
- Blending two samples flattens the contrast. To restore it, the blend is re-centred on the
  texture's constant mean colour and rescaled by `1/sqrt(w0² + w1²)`. This is the
  variance-preserving blend.
- Triplanar weights have a constant overlap zone (`BlendZone`).
- Triplanar normals are combined with reoriented normal mapping (RNM) against the surface normal.

Notes for when it is implemented:
- Rotating the UV and jumping between hash seeds breaks the UV derivatives where the seed changes,
  so the mip level jumps and seams show. Sample with `textureGrad`, rotating `dFdx(uv)`/`dFdy(uv)`
  by the same angle (the derivative of the offset is zero).
- Sample cost: two samples per projection, three projections, colour and normals: about 12 per
  layer where the example is used in full. The terrain pass is very sensitive to shader size
  (M7: a never-taken branch cost it 18 %). Enable it per layer, behind a specialization constant
  or a pipeline variant, and run `bench-diff` on the terrain pass.
- Compute `meanColor` on the CPU when the tileset is loaded (the mean of the top mip), not with
  `texture2D(tex, vec2(0.5), 15.0)` per pixel.
- It replaces or extends the broad brightness variation the terrain shader already applies
  ("Anti-tiling" in `terrain.frag`). Compare the triplanar weights and the normal blend with the
  tileset's current orientation and pixel-normal blending; keep whichever reads better.

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
