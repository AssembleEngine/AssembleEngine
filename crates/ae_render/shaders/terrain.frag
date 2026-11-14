#version 460
// Per-pixel normals come straight from the heightmap, so lighting does not depend on the
// mesh resolution: coarse triangles in flat areas are indistinguishable from fine ones.
#define GRAPHICS
#include "terrain_common.glsl"
#ifdef BARY
#extension GL_EXT_fragment_shader_barycentric : require
#endif

layout(location = 0) in vec3 inWorld;
layout(location = 1) flat in uint inDepth;
layout(location = 0) out vec4 outColor;

vec3 hue(float h) {
    return clamp(abs(fract(h + vec3(0.0, 2.0, 1.0) / 3.0) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Material tilesets (src/terrain/tileset.rs, docs/tilesets.md)
// ---------------------------------------------------------------------------

layout(set = 0, binding = 10) uniform sampler2DArray tileAlbedo; // rgb albedo, a height
layout(set = 0, binding = 11) uniform sampler2DArray tileNormal; // rg tangent-space normal
layout(set = 0, binding = 12) uniform TilesetUBO {
    uvec4 materials[8]; // per material id: x ground layer, y cliff layer
    vec4 layers[8];     // x repeats per world unit, y normal strength, z 1 = triplanar
    vec4 params;        // x, y cliff slope start/end, z blend depth, w detail tilt
    vec4 warp;          // x boundary warp (cells), y 1 / its feature size (world units),
                        // z macro variation, w 1 / its feature size
} tiles;

// One projected layer: albedo (a = height) and the world normal with the
// layer's detail applied.
struct Tap { vec4 albedo; vec3 normal; };

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
    float l = float(layer);
    if (axes.x > 0.0) {
        vec2 uv = vec2(P.z * sg.x, P.y) * k;
        vec2 gx = vec2(dx.z * sg.x, dx.y) * k, gy = vec2(dy.z * sg.x, dy.y) * k;
        vec2 t = (textureGrad(tileNormal, vec3(uv, l), gx, gy).xy * 2.0 - 1.0) * strength;
        float tz = sqrt(max(1.0 - dot(t, t), 0.0));
        t.x *= sg.x;
        vec3 tn = vec3(t + n.zy, tz * n.x);
        a += axes.x * textureGrad(tileAlbedo, vec3(uv, l), gx, gy);
        w += axes.x * tn.zyx;
    }
    if (axes.y > 0.0) {
        vec2 uv = vec2(P.x * sg.y, P.z) * k;
        vec2 gx = vec2(dx.x * sg.y, dx.z) * k, gy = vec2(dy.x * sg.y, dy.z) * k;
        vec2 t = (textureGrad(tileNormal, vec3(uv, l), gx, gy).xy * 2.0 - 1.0) * strength;
        float tz = sqrt(max(1.0 - dot(t, t), 0.0));
        t.x *= sg.y;
        vec3 tn = vec3(t + n.xz, tz * n.y);
        a += axes.y * textureGrad(tileAlbedo, vec3(uv, l), gx, gy);
        w += axes.y * tn.xzy;
    }
    if (axes.z > 0.0) {
        vec2 uv = vec2(-P.x * sg.z, P.y) * k;
        vec2 gx = vec2(-dx.x * sg.z, dx.y) * k, gy = vec2(-dy.x * sg.z, dy.y) * k;
        vec2 t = (textureGrad(tileNormal, vec3(uv, l), gx, gy).xy * 2.0 - 1.0) * strength;
        float tz = sqrt(max(1.0 - dot(t, t), 0.0));
        t.x *= -sg.z;
        vec3 tn = vec3(t + n.xy, tz * n.z);
        a += axes.z * textureGrad(tileAlbedo, vec3(uv, l), gx, gy);
        w += axes.z * tn.xyz;
    }
    return Tap(a, normalize(w));
}

vec3 triplanarAxes(vec3 n) {
    vec3 b = pow(abs(n), vec3(4.0));
    return b / max(b.x + b.y + b.z, 1e-6);
}

// Ground: planar unless the layer is triplanar.
Tap groundTap(uint layer, vec3 P, vec3 dx, vec3 dy, vec3 n) {
    vec3 axes = tiles.layers[layer].z > 0.5 ? triplanarAxes(n) : vec3(0.0, 1.0, 0.0);
    return project(layer, P, dx, dy, n, axes);
}

// Cliffs: duoplanar (the two horizontal axes) unless the layer is triplanar.
Tap cliffTap(uint layer, vec3 P, vec3 dx, vec3 dy, vec3 n) {
    vec3 axes;
    if (tiles.layers[layer].z > 0.5) {
        axes = triplanarAxes(n);
    } else {
        vec2 b = pow(abs(n.xz) + 1e-3, vec2(4.0));
        b /= b.x + b.y;
        axes = vec3(b.x, 0.0, b.y);
    }
    return project(layer, P, dx, dy, n, axes);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth value noise in 0..1.
float valueNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 t = f * f * (3.0 - 2.0 * f);
    float a = hash12(i), b = hash12(i + vec2(1.0, 0.0)), c = hash12(i + vec2(0.0, 1.0)), d = hash12(i + vec2(1.0, 1.0));
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}

const vec3 LAYER_COLORS[8] = vec3[](
    vec3(0.10, 0.75, 0.10), vec3(0.05, 0.35, 0.10), vec3(0.95, 0.85, 0.25), vec3(0.55, 0.55, 0.60),
    vec3(0.15, 0.35, 0.95), vec3(0.90, 0.45, 0.10), vec3(0.45, 0.25, 0.10), vec3(0.95, 0.10, 0.85));

struct Surface { vec3 albedo; vec3 normal; vec3 weights; };

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
    Surface s = Surface(vec3(0.0), vec3(0.0), vec3(0.0));
    float total = 0.0;
    for (int i = 0; i < 5; i++) {
        float b = w[i] > 0.0 ? max(taps[i].albedo.a + w[i] - floorH, 0.0) : 0.0;
        s.albedo += b * taps[i].albedo.rgb;
        s.normal += b * taps[i].normal;
        s.weights += b * LAYER_COLORS[layer[i] & 7u];
        total += b;
    }
    // Anti-tiling: a broad brightness variation over every layer.
    vec2 m = P.xz * tiles.warp.w;
    float macroNoise = valueNoise(m) * 0.6 + valueNoise(m * 2.7 + 11.3) * 0.4;
    s.albedo /= total;
    s.albedo *= 1.0 + (macroNoise - 0.5) * 2.0 * tiles.warp.z;
    s.weights /= total;
    s.normal = normalize(s.normal);
    return s;
}

void main() {
    // In uniform control flow: the tileset samples with explicit gradients.
    vec3 dPdx = dFdx(inWorld), dPdy = dFdy(inWorld);
    float cs = u.map.w;
    vec2 cell = inWorld.xz / cs;
    float hl = sampleHeight(cell - vec2(1, 0)), hr = sampleHeight(cell + vec2(1, 0));
    float hd = sampleHeight(cell - vec2(0, 1)), hu = sampleHeight(cell + vec2(0, 1));
    vec3 n = normalize(vec3(hl - hr, 2.0 * cs, hd - hu));

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

    if ((u.cfg.z & FLAG_MATERIALS) != 0u) {
        // Per-cell gameplay materials (AssemblyEngine addition), blended
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
    }

    if ((u.cfg.z & FLAG_LOD_COLORS) != 0u) {
        albedo = mix(albedo, hue(float(inDepth) * 0.13), 0.65);
    }

    vec3 L = normalize(u.sun.xyz);
    float diff = max(dot(n, L), 0.0);
    vec3 sky = vec3(0.45, 0.55, 0.7);
    vec3 color = albedo * (vec3(1.0, 0.95, 0.85) * 1.6 * diff + sky * (0.25 + 0.25 * n.y));

    // Ground under water (AssemblyEngine addition): tint by the depth of the
    // water column until a water pass exists (M7).
    if (u.water.y > 0.5 && inWorld.y < u.water.x) {
        float depth = u.water.x - inWorld.y;
        vec3 deep = vec3(0.04, 0.16, 0.24);
        vec3 shallow = vec3(0.16, 0.38, 0.42);
        vec3 tint = mix(shallow, deep, 1.0 - exp(-depth * u.water.z * 0.3));
        color = mix(color * vec3(0.6, 0.8, 0.85), tint, 1.0 - exp(-depth * u.water.z));
    }

    float dist = distance(inWorld, u.camPos.xyz);
    float fog = 1.0 - exp(-pow(dist * u.fog.w, 1.5));
    color = mix(color, u.fog.rgb, fog);

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
