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
// gradients, premultiplied. Explicit gradients: this runs in non-uniform
// control flow.
vec4 decalTap(vec2 uv, vec2 gx, vec2 gy, float layer) {
    vec4 t = textureGrad(decalAtlas, vec3(uv * 0.5 + 0.5, layer), gx * 0.5, gy * 0.5);
    return vec4(t.rgb * t.a, t.a);
}

// `albedo` under the decals over world point P (derivatives dPdx, dPdy;
// heightmap normal n), oldest first so newer ones cover older.
//
// Each decal is projected three ways around its centre (the ground's height
// there), weighted by n in the decal's frame with constant-overlap
// triplanar weights: from above (u, v), and along its own x and z axes
// with the height below the centre in place of the coordinate the face
// hides (-sign(n) dy), so a decal across a cliff's edge or into its foot
// continues onto the face as if unfolded. Its extent fades with the 3D
// distance from the centre: square in plan, rounded in the vertical.
vec3 applyDecals(vec3 albedo, vec3 P, vec3 dPdx, vec3 dPdy, vec3 n) {
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
        vec4 f = decalFloats(at + 8u);  // start, 1 / fade-in, end, 1 / fade-out
        float dy = P.y - sampleHeight(a.xy / u.map.w);
        // Rounded in the vertical, the height scaled by the smaller half.
        float r = length(vec2(max(abs(qp.x), abs(qp.y)), dy * max(inv.x, inv.y)));
        if (r >= 1.0) continue;
        vec3 q = vec3(qp.x, dy, qp.y);
        vec3 qx = vec3(dot(dPdx.xz, ax) * inv.x, dPdx.y, dot(dPdx.xz, az) * inv.y);
        vec3 qy = vec3(dot(dPdy.xz, ax) * inv.x, dPdy.y, dot(dPdy.xz, az) * inv.y);
        vec3 nl = vec3(dot(n.xz, ax), n.y, dot(n.xz, az));
        vec3 w = nl * nl;
        w = max(w - max(w.x, max(w.y, w.z)) * 0.8, 0.0);
        w /= w.x + w.y + w.z;
        float sx = nl.x < 0.0 ? 1.0 : -1.0, sz = nl.z < 0.0 ? 1.0 : -1.0;
        vec4 s = vec4(0.0);
        if (w.y > 0.0) s += w.y * decalTap(q.xz, qx.xz, qy.xz, b.z);
        if (w.x > 0.0) s += w.x * decalTap(vec2(sx * q.y * inv.x, q.z), vec2(sx * qx.y * inv.x, qx.z), vec2(sx * qy.y * inv.x, qy.z), b.z);
        if (w.z > 0.0) s += w.z * decalTap(vec2(q.x, sz * q.y * inv.y), vec2(qx.x, sz * qx.y * inv.y), vec2(qy.x, sz * qy.y * inv.y), b.z);
        float fade = f.y > 0.0 ? clamp((time - f.x) * f.y, 0.0, 1.0) : 1.0;
        if (f.w > 0.0) fade *= clamp((f.z - time) * f.w, 0.0, 1.0);
        float opacity = b.w * fade * (1.0 - smoothstep(0.85, 1.0, r));
        albedo = albedo * (1.0 - s.a * opacity) + s.rgb * opacity;
    }
    return albedo;
}
