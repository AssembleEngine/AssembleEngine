#version 460
// Per-pixel normals come straight from the heightmap, so lighting does not depend on the
// mesh resolution: coarse triangles in flat areas are indistinguishable from fine ones.
#define GRAPHICS
#include "terrain_common.glsl"
#include "photometry.glsl"
#include "terrain_shading.glsl"
#include "terrain_decals.glsl"
#define LIGHT_SET 1
#include "lights.glsl"
#ifdef BARY
#extension GL_EXT_fragment_shader_barycentric : require
#endif

layout(location = 0) in vec3 inWorld;
layout(location = 1) flat in uint inDepth;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outGlow;

vec3 hue(float h) {
    return clamp(abs(fract(h + vec3(0.0, 2.0, 1.0) / 3.0) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Material tilesets (src/terrain/tileset.rs, docs/tilesets.md)
// ---------------------------------------------------------------------------

#define tileAlbedo AE_S2D_ARRAY(4, 2) // rgb albedo, a height
#define tileNormal AE_S2D_ARRAY(5, 2) // rg tangent-space normal
layout(set = 0, binding = 12) uniform TilesetUBO {
    uvec4 materials[8]; // per material id: x ground layer, y cliff layer
    vec4 layers[8];     // x repeats per world unit, y normal strength, z flags (bit 0
                        // triplanar, bit 1 directional), w unused
    vec4 means[8];      // per layer: mean albedo (linear) and height
    vec4 roughness[8];  // per layer: x GGX roughness
    vec4 params;        // x, y cliff slope start/end, z blend depth, w detail tilt
    vec4 warp;          // x boundary warp (cells), y 1 / its feature size (world units),
                        // z macro variation, w 1 / its feature size
    vec4 antiTiling;    // x anti-tiling degree (0..1)
} tiles;

// One projected layer: albedo (a = height) and the world normal with the
// layer's detail applied.
struct Tap { vec4 albedo; vec3 normal; };

// One texel of a layer: albedo (a = height) and the tangent-space normal's
// x, y (-1..1, before the layer's strength).
struct Texel { vec4 albedo; vec2 normal; };

Texel texel(float l, vec2 uv, vec2 gx, vec2 gy) {
    return Texel(textureGrad(tileAlbedo, vec3(uv, l), gx, gy), textureGrad(tileNormal, vec3(uv, l), gx, gy).xy * 2.0 - 1.0);
}

// Per seed (anti-tiling): the rotation's cos and sin and the offset, a
// table rather than docs/antitiling.md's hash per pixel (the terrain pass
// is bound by its arithmetic): angles spaced by the golden ratio, offsets
// by the R2 sequence.
const vec4 ANTI_TILING_SEEDS[16] = vec4[](
    vec4(-0.362375, 0.932032, 0.254878, 0.069840),
    vec4(0.896783, -0.442471, 0.009755, 0.639681),
    vec4(-0.960145, -0.279504, 0.764633, 0.209521),
    vec4(0.519179, 0.854666, 0.519511, 0.779361),
    vec4(0.194492, -0.980904, 0.274388, 0.349201),
    vec4(-0.806004, 0.591911, 0.029266, 0.919042),
    vec4(0.994152, 0.107991, 0.784144, 0.488882),
    vec4(-0.660110, -0.751169, 0.539021, 0.058722),
    vec4(-0.020663, 0.999786, 0.293899, 0.628563),
    vec4(0.690583, -0.723254, 0.048777, 0.198403),
    vec4(-0.997765, 0.066823, 0.803654, 0.768243),
    vec4(0.780859, 0.624707, 0.558532, 0.338083),
    vec4(-0.153797, -0.988102, 0.313410, 0.907924),
    vec4(-0.554048, 0.832485, 0.068287, 0.477764),
    vec4(0.970873, -0.239594, 0.823165, 0.047604),
    vec4(-0.877735, -0.479146, 0.578043, 0.617445));

// The layer at `uv` (in repeats), anti-tiled by the tileset's degree,
// after Quilez's "texture repetition", technique 3: a smooth noise (one
// fetch, features of about 3 repeats, values spread evenly over 0..1)
// times the degree's number of versions is an index into versions of the
// texture, each offset (and rotated, unless the texture is directional) by
// a seed from the table. Each pixel shows the two versions its index lies
// between, mostly just one: they blend only in the middle of each step,
// where the transition is pushed by the versions' difference so it
// follows the texture's features. Blending flattens the contrast, so it is
// re-centred on the layer's mean and scaled by 1 / |weights|, which keeps
// the variance (the normals' mean is taken as flat: a tileable height
// field has no mean slope). The gradients rotate with the uv (the offset
// has none), so mips do not jump where the version changes; the normals
// rotate back into the uv's frame. The degree (0..1) scales both the
// number of versions and each version's transform, so it is a continuous
// control, not a switch.
#define antiTilingNoise AE_S2D(11, 2)
const float ANTI_TILING_NOISE_SCALE = 1.0 / 48.0; // 16 features per texture
const float ANTI_TILING_VERSIONS = 8.0;          // at degree 1
// The part of each index step where two versions blend (Quilez: 0.2..0.8;
// narrower, since the second version is most of the cost), and how far
// the versions' difference may push it.
const vec2 ANTI_TILING_BAND = vec2(0.4, 0.6);
const float ANTI_TILING_PUSH = 0.05;

Texel version(float l, uint seed, bool rotates, vec2 uv, vec2 gx, vec2 gy) {
    vec4 h = ANTI_TILING_SEEDS[seed & 15u];
    // The degree fades the transform in: at 0 a version is the plain uv, at
    // 1 the seed's full rotation and offset.
    float degree = tiles.antiTiling.x;
    vec2 c = rotates ? normalize(mix(vec2(1.0, 0.0), h.xy, degree)) : vec2(1.0, 0.0);
    mat2 r = mat2(c.x, c.y, -c.y, c.x);
    Texel t = texel(l, r * uv + h.zw * degree, r * gx, r * gy);
    t.normal = t.normal * r; // transpose(r) * normal
    return t;
}

Texel layerTexel(uint layer, vec2 uv, vec2 gx, vec2 gy) {
    float l = float(layer);
    float amount = tiles.antiTiling.x * ANTI_TILING_VERSIONS;
    if (amount <= 0.0) {
        return texel(l, uv, gx, gy);
    }
    float s = ANTI_TILING_NOISE_SCALE;
    float index = textureGrad(antiTilingNoise, uv * s, gx * s, gy * s).r * amount;
    uint i = uint(index);
    float f = fract(index);
    bool rotates = (uint(tiles.layers[layer].z) & 2u) == 0u;
    // Near either end of a step only one version shows.
    if (f < ANTI_TILING_BAND.x - ANTI_TILING_PUSH) {
        return version(l, i, rotates, uv, gx, gy);
    }
    if (f > ANTI_TILING_BAND.y + ANTI_TILING_PUSH) {
        return version(l, i + 1u, rotates, uv, gx, gy);
    }
    Texel a = version(l, i, rotates, uv, gx, gy), b = version(l, i + 1u, rotates, uv, gx, gy);
    // The difference term is clamped to the push, so the blend is 0 and 1
    // where the early returns assume.
    vec3 d = a.albedo.rgb - b.albedo.rgb;
    float w = smoothstep(ANTI_TILING_BAND.x, ANTI_TILING_BAND.y, f - clamp(0.1 * (d.r + d.g + d.b), -ANTI_TILING_PUSH, ANTI_TILING_PUSH));
    vec2 wt = vec2(1.0 - w, w);
    vec4 mean = tiles.means[layer];
    float k = inversesqrt(dot(wt, wt));
    vec4 albedo = (a.albedo * wt.x + b.albedo * wt.y - mean) * k + mean;
    vec2 normal = (a.normal * wt.x + b.normal * wt.y) * k;
    return Texel(clamp(albedo, 0.0, 1.0), normal);
}

// Sample `layer` projected along the axes weighted by `axes` (x, y, z
// projections; y is the planar one from above), with gradients from the
// world-position derivatives, so the result does not depend on control
// flow. Each projection flips its u by the side the surface faces and the
// tangent x with it, so the detail normal tilts the right way on faces
// pointing down either axis; the tangent normal is combined with the
// geometric one by a whiteout blend.
Tap project(uint layer, vec3 P, vec3 dx, vec3 dy, vec3 n, vec3 axes) {
    float k = tiles.layers[layer].x;
    float strength = tiles.layers[layer].y;
    vec3 sg = vec3(n.x < 0.0 ? -1.0 : 1.0, n.y < 0.0 ? -1.0 : 1.0, n.z < 0.0 ? -1.0 : 1.0);
    vec4 a = vec4(0.0);
    vec3 w = vec3(0.0);
    if (axes.x > 0.0) {
        vec2 uv = vec2(P.z * sg.x, P.y) * k;
        vec2 gx = vec2(dx.z * sg.x, dx.y) * k, gy = vec2(dy.z * sg.x, dy.y) * k;
        Texel x = layerTexel(layer, uv, gx, gy);
        vec2 t = x.normal * strength;
        float tz = sqrt(max(1.0 - dot(t, t), 0.0));
        t.x *= sg.x;
        vec3 tn = vec3(t + n.zy, tz * n.x);
        a += axes.x * x.albedo;
        w += axes.x * tn.zyx;
    }
    if (axes.y > 0.0) {
        vec2 uv = vec2(P.x * sg.y, P.z) * k;
        vec2 gx = vec2(dx.x * sg.y, dx.z) * k, gy = vec2(dy.x * sg.y, dy.z) * k;
        Texel x = layerTexel(layer, uv, gx, gy);
        vec2 t = x.normal * strength;
        float tz = sqrt(max(1.0 - dot(t, t), 0.0));
        t.x *= sg.y;
        vec3 tn = vec3(t + n.xz, tz * n.y);
        a += axes.y * x.albedo;
        w += axes.y * tn.xzy;
    }
    if (axes.z > 0.0) {
        vec2 uv = vec2(-P.x * sg.z, P.y) * k;
        vec2 gx = vec2(-dx.x * sg.z, dx.y) * k, gy = vec2(-dy.x * sg.z, dy.y) * k;
        Texel x = layerTexel(layer, uv, gx, gy);
        vec2 t = x.normal * strength;
        float tz = sqrt(max(1.0 - dot(t, t), 0.0));
        t.x *= -sg.z;
        vec3 tn = vec3(t + n.xy, tz * n.z);
        a += axes.z * x.albedo;
        w += axes.z * tn.xyz;
    }
    return Tap(a, normalize(w));
}

vec3 triplanarAxes(vec3 n) {
    vec3 b = pow(abs(n), vec3(4.0));
    return b / max(b.x + b.y + b.z, 1e-6);
}

bool triplanar(uint layer) {
    return (uint(tiles.layers[layer].z) & 1u) != 0u;
}

// Ground: planar unless the layer is triplanar.
Tap groundTap(uint layer, vec3 P, vec3 dx, vec3 dy, vec3 n) {
    vec3 axes = triplanar(layer) ? triplanarAxes(n) : vec3(0.0, 1.0, 0.0);
    return project(layer, P, dx, dy, n, axes);
}

// Cliffs: duoplanar (the two horizontal axes) unless the layer is triplanar.
Tap cliffTap(uint layer, vec3 P, vec3 dx, vec3 dy, vec3 n) {
    vec3 axes;
    if (triplanar(layer)) {
        axes = triplanarAxes(n);
    } else {
        vec2 b = pow(abs(n.xz) + 1e-3, vec2(4.0));
        b /= b.x + b.y;
        axes = vec3(b.x, 0.0, b.y);
    }
    return project(layer, P, dx, dy, n, axes);
}

const vec3 LAYER_COLORS[8] = vec3[](
    vec3(0.10, 0.75, 0.10), vec3(0.05, 0.35, 0.10), vec3(0.95, 0.85, 0.25), vec3(0.55, 0.55, 0.60),
    vec3(0.15, 0.35, 0.95), vec3(0.90, 0.45, 0.10), vec3(0.45, 0.25, 0.10), vec3(0.95, 0.10, 0.85));

struct Surface { vec3 albedo; vec3 normal; vec3 weights; float roughness; };

// Two-level blending. Broad masks: the material cells around the pixel
// (bilinear between cell centres) and the cliff mask from the slope,
// itself tilted by the ground layers' detail normals. Then a height blend
// over those candidates: each one's weight plus its layer height, keeping
// only what lies within `blend depth` of the highest, so boundaries follow
// the textures' relief (pebbles poke through sand, rock shows through
// grass in its cracks) instead of a smooth line.
Surface tileset(vec2 cell, vec3 P, vec3 dx, vec3 dy, vec3 n) {
    uint mats[4] = uint[](0u, 0u, 0u, 0u);
    // Meandering boundaries: displace the cell lookup by a smooth value
    // noise (features of `1 / warp.y` world units, two octaves), so
    // boundaries cut across cells instead of tracing their edges.
    vec2 q = P.xz * tiles.warp.y;
    float wx = valueNoise(q) * 0.65 + valueNoise(q * 2.03 + 17.1) * 0.35;
    float wy = valueNoise(q + vec2(41.7, 7.3)) * 0.65 + valueNoise(q * 2.03 + vec2(3.9, 29.4)) * 0.35;
    vec2 p = cell + (vec2(wx, wy) - 0.5) * 2.0 * tiles.warp.x - 0.5;
    ivec2 c0 = ivec2(floor(p));
    vec2 f = p - vec2(c0);
    if ((u.cfg.z & FLAG_MATERIALS) != 0u) {
        ivec2 hi = ivec2(u.map.xy) - 1;
        mats[0] = min(texelFetch(materialTex, clamp(c0, ivec2(0), hi), 0).r, 7u);
        mats[1] = min(texelFetch(materialTex, clamp(c0 + ivec2(1, 0), ivec2(0), hi), 0).r, 7u);
        mats[2] = min(texelFetch(materialTex, clamp(c0 + ivec2(0, 1), ivec2(0), hi), 0).r, 7u);
        mats[3] = min(texelFetch(materialTex, clamp(c0 + ivec2(1, 1), ivec2(0), hi), 0).r, 7u);
    }
    float wc[4] = float[]((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);

    Tap taps[5];
    uint layer[5];
    float w[5];
    vec3 groundN = vec3(0.0);
    int dominant = 0;
    // Explicit gradients make branching safe: a corner whose layer an
    // earlier corner has reuses its tap (inside a material all four do),
    // and the cliff layer is sampled only where the cliff mask is not 0.
    // Indices stay constant (no runtime indexing into the tap array).
    for (int i = 0; i < 4; i++) layer[i] = tiles.materials[mats[i]].x;
    taps[0] = groundTap(layer[0], P, dx, dy, n);
    taps[1] = layer[1] == layer[0] ? taps[0] : groundTap(layer[1], P, dx, dy, n);
    taps[2] = layer[2] == layer[0] ? taps[0] : layer[2] == layer[1] ? taps[1] : groundTap(layer[2], P, dx, dy, n);
    taps[3] = layer[3] == layer[0] ? taps[0] : layer[3] == layer[1] ? taps[1] : layer[3] == layer[2] ? taps[2] : groundTap(layer[3], P, dx, dy, n);
    for (int i = 0; i < 4; i++) {
        groundN += wc[i] * taps[i].normal;
        if (wc[i] > wc[dominant]) dominant = i;
    }
    groundN = normalize(groundN);
    float slope = 1.0 - mix(n.y, groundN.y, tiles.params.w);
    float cliff = smoothstep(tiles.params.x, tiles.params.y, slope);
    layer[4] = tiles.materials[mats[dominant]].y;
    taps[4] = cliff > 0.0 ? cliffTap(layer[4], P, dx, dy, n) : Tap(vec4(0.0), n);
    for (int i = 0; i < 4; i++) w[i] = wc[i] * (1.0 - cliff);
    w[4] = cliff;

    float top = -1e9;
    for (int i = 0; i < 5; i++) {
        if (w[i] > 0.0) top = max(top, taps[i].albedo.a + w[i]);
    }
    float floorH = top - tiles.params.z;
    Surface s = Surface(vec3(0.0), vec3(0.0), vec3(0.0), 0.0);
    float total = 0.0;
    for (int i = 0; i < 5; i++) {
        float b = w[i] > 0.0 ? max(taps[i].albedo.a + w[i] - floorH, 0.0) : 0.0;
        s.albedo += b * taps[i].albedo.rgb;
        s.normal += b * taps[i].normal;
        s.weights += b * LAYER_COLORS[layer[i] & 7u];
        s.roughness += b * tiles.roughness[layer[i]].x;
        total += b;
    }
    // Anti-tiling: a broad brightness variation over every layer.
    vec2 m = P.xz * tiles.warp.w;
    float macroNoise = valueNoise(m) * 0.6 + valueNoise(m * 2.7 + 11.3) * 0.4;
    s.albedo /= total;
    s.albedo *= 1.0 + (macroNoise - 0.5) * 2.0 * tiles.warp.z;
    s.weights /= total;
    s.roughness /= total;
    s.normal = normalize(s.normal);
    return s;
}

void main() {
    // No light of its own until a glowing decal says otherwise; every
    // return path below leaves the glow black.
    outGlow = vec4(0.0);
    // In uniform control flow: the tileset samples with explicit gradients.
    vec3 dPdx = dFdx(inWorld), dPdy = dFdy(inWorld);
    float cs = u.map.w;
    vec2 cell = inWorld.xz / cs;
    float hl = sampleHeight(cell - vec2(1, 0)), hr = sampleHeight(cell + vec2(1, 0));
    float hd = sampleHeight(cell - vec2(0, 1)), hu = sampleHeight(cell + vec2(0, 1));
    vec3 n = normalize(vec3(hl - hr, 2.0 * cs, hd - hu));
    vec3 heightNormal = n;

    float t = (inWorld.y - u.height.y) / max(u.height.x, 1e-3);
    float slope = 1.0 - n.y;

    vec3 grass = vec3(0.23, 0.33, 0.12);
    vec3 dry = vec3(0.42, 0.38, 0.22);
    vec3 rock = vec3(0.36, 0.33, 0.30);
    vec3 darkRock = vec3(0.22, 0.20, 0.19);
    vec3 snow = vec3(0.92, 0.94, 0.97);

    vec3 albedo = mix(grass, dry, smoothstep(0.08, 0.2, t));
    albedo = mix(albedo, mix(rock, darkRock, smoothstep(0.3, 0.8, t)), smoothstep(0.12, 0.3, slope));
    float snowLine = 0.55 + 0.1 * sin(inWorld.x * 0.0013) * cos(inWorld.z * 0.0011);
    albedo = mix(albedo, snow, smoothstep(snowLine, snowLine + 0.05, t) * (1.0 - smoothstep(0.35, 0.6, slope)));

    // Specular roughness: the tileset's per-layer value when one is bound,
    // else a matte dielectric (ground, mud, grass) for the palette path.
    // The palette path is a dielectric; decals can make patches metallic.
    float roughness = 0.85;
    float metallic = 0.0;
    if ((u.cfg.z & FLAG_MATERIALS) != 0u) {
        // Per-cell gameplay materials (AssembleEngine addition), blended
        // bilinearly between cell centres; slopes still turn to rock.
        // Placeholder until material tilesets.
        const vec3 palette[8] = vec3[](
            vec3(0.23, 0.33, 0.12), // grass
            vec3(0.10, 0.20, 0.07), // forest
            vec3(0.60, 0.54, 0.38), // sand
            vec3(0.36, 0.33, 0.30), // rock
            vec3(0.32, 0.30, 0.22), // water (bed)
            vec3(0.45, 0.41, 0.35), // road
            vec3(0.30, 0.23, 0.15), // mud
            vec3(1.0, 0.0, 1.0));
        vec2 p = cell - 0.5;
        ivec2 c0 = ivec2(floor(p));
        vec2 f = p - vec2(c0);
        ivec2 hi = ivec2(u.map.xy) - 1;
        vec3 m00 = palette[min(texelFetch(materialTex, clamp(c0, ivec2(0), hi), 0).r, 7u)];
        vec3 m10 = palette[min(texelFetch(materialTex, clamp(c0 + ivec2(1, 0), ivec2(0), hi), 0).r, 7u)];
        vec3 m01 = palette[min(texelFetch(materialTex, clamp(c0 + ivec2(0, 1), ivec2(0), hi), 0).r, 7u)];
        vec3 m11 = palette[min(texelFetch(materialTex, clamp(c0 + ivec2(1, 1), ivec2(0), hi), 0).r, 7u)];
        albedo = mix(mix(m00, m10, f.x), mix(m01, m11, f.x), f.y);
        albedo = mix(albedo, rock, smoothstep(0.2, 0.45, slope));
    }

    if ((u.cfg.z & FLAG_TILESET) != 0u) {
        Surface ts = tileset(cell, inWorld, dPdx, dPdy, n);
        if ((u.cfg.z & FLAG_WEIGHTS) != 0u) {
            // Debug view: blend weights as flat layer colours, unlit.
            outColor = vec4(ts.weights, 1.0);
            return;
        }
        albedo = ts.albedo;
        n = ts.normal;
        roughness = ts.roughness;
    }

    // The shading normal under the decals (shadows keep the ground's).
    vec3 shadeN = n;
    vec3 glow = vec3(0.0);
    if (DECALS) {
        Decals d = applyDecals(albedo, roughness, metallic, n, inWorld, dPdx, dPdy, heightNormal);
        albedo = d.albedo;
        roughness = d.roughness;
        metallic = d.metallic;
        shadeN = d.normal;
        glow = d.glow;
    }

    if ((u.cfg.z & FLAG_LOD_COLORS) != 0u) {
        albedo = mix(albedo, hue(float(inDepth) * 0.13), 0.65);
    }
    // Every colour above was authored in BT.709 (the tileset's sRGB albedo,
    // the palette, the decals): the lighting runs in the working space.
    bool wide = u.photometry.y > 0.5;
    albedo = aeToWorking(albedo, wide);
    glow = aeToWorking(glow, wide);

    vec3 L = normalize(u.sun.xyz);
    vec3 V = normalize(u.camPos.xyz - inWorld);
    // On the ground, against the ground as the heights texture has it (the
    // mesh's LOD error would read as shadow on convex slopes).
    // Both shadows at the bilinear ground under the pixel (what the hills'
    // shadow heights and the units' feet are measured against), not the
    // LOD mesh's height. The darker one wins (they are the same sun).
    vec3 ground = vec3(inWorld.x, sampleHeight(inWorld.xz / u.map.w), inWorld.z);
    vec3 tint;
    float units = unitSunAt(ground, n, tint);
    // Direct sun and sky as illuminance (lux), before the receiver cosine;
    // the sky is the hemispherical approximation (half on verticals). The
    // terrain is a rough dielectric (F0 0.04, the tileset's roughness);
    // a decal's material can raise the roughness or make the patch
    // metallic (F0 = its colour).
    vec3 sunE = aeUnitTint(AE_SUN_TINT, wide) * (u.photometry.z * min(sunAt(ground.xz, ground.y), units)) * tint;
    vec3 skyE = aeUnitTint(AE_SKY_TINT, wide) * (u.photometry.w * (0.5 + 0.5 * shadeN.y));
    vec3 skyR = aeUnitTint(AE_SKY_TINT, wide) * (u.photometry.w / AE_PI);
    SurfaceBRDF s = SurfaceBRDF(albedo, mix(vec3(0.04), albedo, metallic), max(roughness, 0.04), metallic);
    vec3 color = brdfLight(s, shadeN, V, L, sunE);
    color += brdfDiffuse(s, shadeN, V) * skyE + brdfAmbient(s, shadeN, V) * skyR;
    // The clustered lights (no shadows).
    if (lightsOn()) {
        color += lightsSurface(lightClusterFrag(gl_FragCoord), inWorld, shadeN, V, s);
    }

    // Ground under water (AssembleEngine addition): tint by the depth of the
    // water column. This is the part of RA3's underwater pass the water
    // surface (water.frag) builds on: the surface blends its reflection
    // over this.
    if (u.water.y > 0.5 && inWorld.y < u.water.x) {
        float depth = u.water.x - inWorld.y;
        if ((u.cfg.z & FLAG_WATER) != 0u) {
            // Caustics while the surface is drawn: the sunlight focused by
            // the waves, two drifting layers of ridged noise whose
            // product leaves bright threads, fading with depth.
            vec2 q = inWorld.xz * 0.12;
            float t = u.water.w;
            float r0 = 1.0 - abs(valueNoise(q + vec2(t * 0.11, t * 0.07)) * 2.0 - 1.0);
            float r1 = 1.0 - abs(valueNoise(q * 1.37 + vec2(-t * 0.09, t * 0.12) + 5.3) * 2.0 - 1.0);
            float caustic = pow(r0 * r1, 6.0) * exp(-depth * 0.18) * smoothstep(0.0, 0.5, depth);
            color += brdfDiffuse(s, shadeN, V) * sunE * max(dot(shadeN, L), 0.0) * caustic * 0.9;
        }
        // The water body's in-scattered radiance (cd/m²), deep and shallow.
        vec3 deep = aeToWorking(vec3(786.0, 3146.0, 4719.0), wide);
        vec3 shallow = aeToWorking(vec3(3146.0, 7471.0, 8258.0), wide);
        vec3 tint = mix(shallow, deep, 1.0 - exp(-depth * u.water.z * 0.3));
        color = mix(color * aeToWorking(vec3(0.6, 0.8, 0.85), wide), tint, 1.0 - exp(-depth * u.water.z));
        // Water puts a glow out within a metre.
        glow *= 1.0 - smoothstep(0.0, 1.0, depth);
    }

    // Emissive: after the light and the shadows. Explosions go off only in
    // sight (the client's rule); if the spot falls out of sight while it
    // glows, the fog of war dims the glow as it dims the scar under it.
    // A scar's glow, as luminance: the decal shader's black-body glow
    // (0..~1 at full heat) times its peak (cd/m²).
    vec3 emitted = glow * 19660.8;
    color += emitted;

    if ((u.cfg.z & FLAG_VISION) != 0u) {
        color = applyVision(color, inWorld.xz);
        emitted = applyVision(emitted, inWorld.xz);
    }

    float dist = distance(inWorld, u.camPos.xyz);
    float fogged = fogAmount(dist);
    color = mix(color, u.fog.rgb, fogged);
    // The scene buffer holds exposed radiance (the tonemap only curves).
    color *= u.photometry.x;
    // The bloom's light sources: only the emission, after the same fog and
    // vision the scene sees (fog is not a light source).
    outGlow = vec4(emitted * (1.0 - fogged) * u.photometry.x, 1.0);

#ifdef BARY
    // Wireframe overlay. Sub-mesh triangles are emitted as (v0, apex, v2), so the
    // hypotenuse is the edge where the apex weight is 0: hiding it shows LEB quads.
    if ((u.cfg.z & (FLAG_WIRE_TRIS | FLAG_WIRE_QUADS)) != 0u) {
        vec3 b = gl_BaryCoordEXT;
        vec3 edge = smoothstep(vec3(0.0), fwidth(b) * 1.2, b);
        float e = (u.cfg.z & FLAG_WIRE_QUADS) != 0u ? min(edge.x, edge.z) : min(min(edge.x, edge.y), edge.z);
        color = mix(vec3(0.02), color, 0.35 + 0.65 * e);
    }
#endif
    outColor = vec4(color, 1.0);
}
