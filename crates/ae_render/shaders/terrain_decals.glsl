// Decals (src/decals.rs), applied to the terrain's albedo before lighting.
// Needs terrain_common.glsl (u, sampleHeight).
//
// The buffer, in 32-bit words: a header (bucket mask, 0, cell size as float
// bits, decal capacity), the decals (16 words each: centre x, z, half-sizes
// x, z, cos, sin, layer, opacity, start time, 1 / fade-in, end time (0:
// none), 1 / fade-out (0: none), glow, glow end time, 1 / glow life (0: no
// glow), blend (0 replace, 1 modulate)), one word
// per bucket of the spatial hash of world cells (first index << 8 | count),
// the indices.

// Specialization constant 1: false compiles the decals out (the terrain
// pipeline used while there are none).
layout(constant_id = 1) const bool DECALS = false;

layout(std430, set = 0, binding = 17) readonly buffer DecalBuf { uint decalWords[]; };
layout(set = 0, binding = 18) uniform sampler2DArray decalAtlas;
// Layer for layer: the tangent-space normal (n * 0.5 + 0.5) and the glow mask.
layout(set = 0, binding = 19) uniform sampler2DArray decalNormals;

vec4 decalFloats(uint at) {
    return uintBitsToFloat(uvec4(decalWords[at], decalWords[at + 1u], decalWords[at + 2u], decalWords[at + 3u]));
}

// One projection: the atlas at `uv` (decal space, -1..1) with its
// gradients, premultiplied, and the normal map's texel (`m`: the normal in
// the frame T, B, N, the world directions of +u, +v and out of the ground;
// w the glow mask). Explicit gradients: this runs in non-uniform control
// flow.
vec4 decalTap(vec2 uv, vec2 gx, vec2 gy, float layer, vec3 T, vec3 B, vec3 N, out vec4 m) {
    vec3 at = vec3(uv * 0.5 + 0.5, layer);
    vec4 t = textureGrad(decalAtlas, at, gx * 0.5, gy * 0.5);
    vec4 nm = textureGrad(decalNormals, at, gx * 0.5, gy * 0.5);
    // Gram-Schmidt against the ground's normal: T and B are the world
    // gradients of u and v, so the frame follows the projection.
    T = normalize(T - N * dot(T, N));
    B = normalize(B - N * dot(B, N) - T * dot(B, T));
    vec3 tn = nm.xyz * 2.0 - 1.0;
    m = vec4(tn.x * T + tn.y * B + tn.z * N, nm.w);
    return vec4(t.rgb * t.a, t.a);
}

// Recoil's black body (shadertoy MslSDl): the colour of a body at `t`
// kelvin, its brightness rising as t^4 (1 near 2500 K), in gamma space.
vec3 blackBody(float t) {
    float u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t * t) / (1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t * t);
    float v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t * t) / (1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t * t);
    float x = 3.0 * u / (2.0 * u - 8.0 * v + 4.0);
    float y = 2.0 * v / (2.0 * u - 8.0 * v + 4.0);
    vec3 xyz = vec3(x / y, 1.0, (1.0 - x - y) / y);
    vec3 rgb = xyz * mat3(3.2404542, -1.5371385, -0.4985314, -0.9692660, 1.8760108, 0.0415560, 0.0556434, -0.2040259, 1.0572252);
    return max(rgb, 0.0) * pow(0.0004 * t, 4.0);
}

// Gamma 2.2 to linear of Recoil's `2 * decal * ground` (Blend::Modulate),
// with the decal's colour sampled linear: 2^2.2.
const float MODULATE = 4.594794;

// What the decals leave: the albedo, the shading normal, and the glow
// (linear, added after lighting).
struct Decals { vec3 albedo; vec3 normal; vec3 glow; };

// The decals over world point P (derivatives dPdx, dPdy; heightmap normal
// hn), oldest first so newer ones cover older, on `albedo` and the shading
// normal `n`.
//
// Each decal is projected three ways around its centre (the ground's height
// there), weighted by hn in the decal's frame with constant-overlap
// triplanar weights: from above (u, v), and along its own x and z axes
// with the height below the centre in place of the coordinate the face
// hides (-sign(n) dy), so a decal across a cliff's edge or into its foot
// continues onto the face as if unfolded. Its extent fades with the 3D
// distance from the centre: square in plan, rounded in the vertical.
//
// Its normal map replaces the shading normal by its alpha (a flat map
// leaves the heightmap's normal: the scar covers the ground's detail). Its
// glow (Recoil's scars): the glow mask lit as a black body, hotter with the
// decal's glow, which fades linearly over its glow life, flickers near
// full, and is held to the centre.
Decals applyDecals(vec3 albedo, vec3 n, vec3 P, vec3 dPdx, vec3 dPdy, vec3 hn) {
    Decals o = Decals(albedo, n, vec3(0.0));
    uint mask = decalWords[0];
    float cellSize = uintBitsToFloat(decalWords[2]);
    ivec2 c = ivec2(floor(P.xz / cellSize));
    uint bucketsAt = 4u + 16u * decalWords[3];
    uint indexAt = bucketsAt + mask + 1u;
    uint entry = decalWords[bucketsAt + ((uint(c.x) * 73856093u ^ uint(c.y) * 19349663u) & mask)];
    uint first = entry >> 8u, count = entry & 255u;
    float time = u.water.w;
    float layers = float(textureSize(decalAtlas, 0).z);
    for (uint i = 0u; i < count; i++) {
        uint at = 4u + 16u * decalWords[indexAt + first + i];
        vec4 a = decalFloats(at);       // centre x, z, half x, half z
        vec4 b = decalFloats(at + 4u);  // cos, sin, layer, opacity
        // Decal frame, scaled to -1..1 over the half-sizes (x, z): outside
        // in plan (another cell's decal in this bucket, or past its edge)
        // before anything else is read.
        vec2 ax = b.xy, az = vec2(-b.y, b.x);
        vec2 inv = 1.0 / a.zw;
        vec2 dxz = P.xz - a.xy;
        vec2 qp = vec2(dot(dxz, ax), dot(dxz, az)) * inv;
        if (max(abs(qp.x), abs(qp.y)) >= 1.0 || b.z >= layers) continue;
        vec4 f = decalFloats(at + 8u);   // start, 1 / fade-in, end, 1 / fade-out
        vec4 g = decalFloats(at + 12u);  // glow, glow end, 1 / glow life, blend
        float dy = P.y - sampleHeight(a.xy / u.map.w);
        // Rounded in the vertical, the height scaled by the smaller half.
        float r = length(vec2(max(abs(qp.x), abs(qp.y)), dy * max(inv.x, inv.y)));
        if (r >= 1.0) continue;
        vec3 q = vec3(qp.x, dy, qp.y);
        vec3 qx = vec3(dot(dPdx.xz, ax) * inv.x, dPdx.y, dot(dPdx.xz, az) * inv.y);
        vec3 qy = vec3(dot(dPdy.xz, ax) * inv.x, dPdy.y, dot(dPdy.xz, az) * inv.y);
        vec3 nl = vec3(dot(hn.xz, ax), hn.y, dot(hn.xz, az));
        vec3 w = nl * nl;
        w = max(w - max(w.x, max(w.y, w.z)) * 0.8, 0.0);
        w /= w.x + w.y + w.z;
        float sx = nl.x < 0.0 ? 1.0 : -1.0, sz = nl.z < 0.0 ? 1.0 : -1.0;
        // The world directions of the decal's +x and +z, and up.
        vec3 X = vec3(ax.x, 0.0, ax.y), Z = vec3(az.x, 0.0, az.y), Y = vec3(0.0, 1.0, 0.0);
        vec4 s = vec4(0.0), sn = vec4(0.0), m;
        if (w.y > 0.0) {
            vec4 t = decalTap(q.xz, qx.xz, qy.xz, b.z, X, Z, hn, m);
            s += w.y * t;
            sn += w.y * vec4(m.xyz * t.a, m.w);
        }
        if (w.x > 0.0) {
            vec4 t = decalTap(vec2(sx * q.y * inv.x, q.z), vec2(sx * qx.y * inv.x, qx.z), vec2(sx * qy.y * inv.x, qy.z), b.z, sx * Y, Z, hn, m);
            s += w.x * t;
            sn += w.x * vec4(m.xyz * t.a, m.w);
        }
        if (w.z > 0.0) {
            vec4 t = decalTap(vec2(q.x, sz * q.y * inv.y), vec2(qx.x, sz * qx.y * inv.y), vec2(qy.x, sz * qy.y * inv.y), b.z, X, sz * Y, hn, m);
            s += w.z * t;
            sn += w.z * vec4(m.xyz * t.a, m.w);
        }
        float fade = f.y > 0.0 ? clamp((time - f.x) * f.y, 0.0, 1.0) : 1.0;
        if (f.w > 0.0) fade *= clamp((f.z - time) * f.w, 0.0, 1.0);
        float opacity = b.w * fade * (1.0 - smoothstep(0.85, 1.0, r));
        float keep = 1.0 - s.a * opacity;
        vec3 colour = g.w > 0.5 ? o.albedo * MODULATE * s.rgb : s.rgb;
        o.albedo = o.albedo * keep + colour * opacity;
        o.normal = o.normal * keep + sn.xyz * opacity;
        o.glow *= keep;
        float heat = g.z > 0.0 ? g.x * clamp((g.y - time) * g.z, 0.0, 1.0) : 0.0;
        if (heat > 0.0) {
            heat += smoothstep(0.75, 1.0, heat) * 0.2 * abs(sin(0.6 * time));
            // Held to the centre by the 3D distance (round, as Recoil's).
            heat *= pow(smoothstep(0.9, 0.1, length(vec3(qp, dy * max(inv.x, inv.y)))), 7.0);
            vec3 e = blackBody(sn.w * mix(1.0, 3700.0, heat)) * heat;
            o.glow += pow(min(e, vec3(16.0)), vec3(2.2)) * s.a * opacity;
        }
    }
    o.normal = normalize(o.normal);
    return o;
}
