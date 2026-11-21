// Decals (src/decals.rs), applied to the terrain's albedo, normal and
// surface material (roughness, metallic) before lighting.
// Needs terrain_common.glsl (u, groundAt).
//
// The buffer, in 32-bit words: a header (bucket mask, 0, cell size as float
// bits, decal capacity), the decals (16 words each: centre x, z, half-sizes
// x, z, cos, sin, layer, opacity, start time, 1 / fade-in, end time (0:
// none), 1 / fade-out (0: none), glow, 0, 1 / glow life (0: no glow),
// blend (0 replace, 1 modulate)), one word
// per bucket of the spatial hash of world cells (first index << 8 | count),
// the indices.

// Specialization constant 1: false compiles the decals out (the terrain
// pipeline used while there are none).
layout(constant_id = 1) const bool DECALS = false;

layout(std430, set = 0, binding = 17) readonly buffer DecalBuf { uint decalWords[]; };
layout(set = 0, binding = 18) uniform sampler2DArray decalAtlas;
// Layer for layer: the tangent normal (xy, n * 0.5 + 0.5; z is
// reconstructed), the roughness texture (times the layer's factor) and the
// glow mask.
layout(set = 0, binding = 19) uniform sampler2DArray decalSurface;
// Layer for layer: x roughness factor, y metallic factor (src/decals.rs,
// LayerMaterial).
layout(std430, set = 0, binding = 21) readonly buffer DecalMaterialBuf { vec4 decalMaterials[]; };

vec4 decalFloats(uint at) {
    return uintBitsToFloat(uvec4(decalWords[at], decalWords[at + 1u], decalWords[at + 2u], decalWords[at + 3u]));
}

// One projection: the atlas at `uv` (decal space, -1..1) with its
// gradients, premultiplied; the surface's normal in the frame T, B, N (the
// world directions of +u, +v and out of the ground; w the glow mask); and
// the roughness texture. The normal's z is rebuilt (it points out of the
// ground). Explicit gradients: this runs in non-uniform control flow.
vec4 decalTap(vec2 uv, vec2 gx, vec2 gy, float layer, vec3 T, vec3 B, vec3 N, out vec4 m, out float roughness) {
    vec3 at = vec3(uv * 0.5 + 0.5, layer);
    vec4 t = textureGrad(decalAtlas, at, gx * 0.5, gy * 0.5);
    vec4 sm = textureGrad(decalSurface, at, gx * 0.5, gy * 0.5);
    // Gram-Schmidt against the ground's normal: T and B are the world
    // gradients of u and v, so the frame follows the projection.
    T = normalize(T - N * dot(T, N));
    B = normalize(B - N * dot(B, N) - T * dot(B, T));
    vec2 nxy = sm.xy * 2.0 - 1.0;
    vec3 tn = vec3(nxy, sqrt(max(1.0 - dot(nxy, nxy), 0.0)));
    m = vec4(tn.x * T + tn.y * B + tn.z * N, sm.w);
    roughness = sm.z;
    return vec4(t.rgb * t.a, t.a);
}

// One segment of the ground flattened into the plane: plan vector `d`
// (along the unit ray `dir`), rise `dh`, the ground's slope at its start,
// middle and end. The rise is taken along the slope where it happens (the
// three averaged by their steepness: across a fold, the face's), flattened
// as that tangent plane is, isometrically: with G.d = dh, |d'|^2 = |d|^2 +
// dh^2, the direction across G kept. Over a ridge or a valley within the
// segment the rise and the steepest slope can disagree along the ray; the
// gain along it is held at 0 or more, so a segment never comes out shorter
// along the ray than in plan.
vec2 unfoldSegment(vec2 d, vec2 dir, float dh, vec2 g0, vec2 gm, vec2 g1) {
    vec3 w = vec3(dot(g0, g0), dot(gm, gm), dot(g1, g1)) + 1e-3;
    vec2 G = (w.x * g0 + w.y * gm + w.z * g1) / (w.x + w.y + w.z);
    vec2 gain = dh / (1.0 + sqrt(1.0 + dot(G, G))) * G;
    float along = dot(gain, dir);
    return d + gain + dir * (max(along, 0.0) - along);
}

// A point's offset from a decal's centre c (its ground gc, groundAt), unfolded
// over the ground (metres, world xz). The plan ray from c to the point
// (its ground gp; d = its plan offset) is read every `stride` metres,
// fixed per decal, and each segment flattened along its slope, so the
// result is continuous in the point (a new step starts where the last
// segment has become the step), exact over a planar slope and over a fold
// (a cliff's edge or foot) at any angle, the plan offset on flat ground,
// and never nearer the centre along the ray than in plan.
vec2 unfoldRay(vec2 c, vec3 gc, vec2 d, vec3 gp, float stride) {
    float rho = length(d);
    if (rho < 1e-4) return d;
    vec2 dir = d / rho;
    // At most 12: rho < the half-diagonal <= sqrt(2) 8 strides.
    int n = min(int(rho / stride), 12);
    vec2 sum = vec2(0.0), at = c;
    vec3 g = gc;
    for (int i = 1; i <= n; i++) {
        vec2 next = c + dir * (stride * float(i));
        vec3 gi = groundAt(next);
        sum += unfoldSegment(dir * stride, dir, gi.x - g.x, g.yz, groundAt((at + next) * 0.5).yz, gi.yz);
        at = next;
        g = gi;
    }
    vec2 p = c + d;
    return sum + unfoldSegment(p - at, dir, gp.x - g.x, g.yz, groundAt((at + p) * 0.5).yz, gp.yz);
}

// Rotates v by the smallest rotation taking the unit `from` to up, or
// (sign -1) up to `from`.
vec3 levelBy(vec3 v, vec3 from, float sign) {
    vec3 k = sign * cross(from, vec3(0.0, 1.0, 0.0));
    return v + cross(k, v) + cross(k, cross(k, v)) / (1.0 + from.y);
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

// What the decals leave: the albedo, the shading normal, the surface
// material (roughness, metallic) and the glow (linear, added after
// lighting).
struct Decals { vec3 albedo; vec3 normal; float roughness; float metallic; vec3 glow; };

// The decals over world point P (derivatives dPdx, dPdy; heightmap normal
// hn), oldest first so newer ones cover older, on `albedo`, the shading
// normal `n` and the surface's roughness and metallic.
//
// Each decal is unfolded over the ground: u is the length along the ground
// from its centre line along its x axis, v along its z axis (the ground
// read every heightmap cell), so a decal across a cliff's edge or into its
// foot continues onto the face, and the image stays continuous over any
// ground. Its extent is the square in unfolded metres, faded over its last
// 15 %.
//
// Its normal map replaces the shading normal by its alpha (a flat map
// leaves the heightmap's normal: the scar covers the ground's detail). Its
// glow (Recoil's scars): the glow mask lit as a black body, hotter with the
// decal's glow, which fades linearly over its glow life, flickers near
// full, and is held to the centre.
Decals applyDecals(vec3 albedo, float roughness, float metallic, vec3 n, vec3 P, vec3 dPdx, vec3 dPdy, vec3 hn) {
    Decals o = Decals(albedo, n, roughness, metallic, vec3(0.0));
    uint mask = decalWords[0];
    float cellSize = uintBitsToFloat(decalWords[2]);
    ivec2 c = ivec2(floor(P.xz / cellSize));
    uint bucketsAt = 4u + 16u * decalWords[3];
    uint indexAt = bucketsAt + mask + 1u;
    uint entry = decalWords[bucketsAt + ((uint(c.x) * 73856093u ^ uint(c.y) * 19349663u) & mask)];
    uint first = entry >> 8u, count = entry & 255u;
    float time = u.effects.x;
    float layers = float(textureSize(decalAtlas, 0).z);
    // The pixel's own ground, gathered on the first decal that reaches the
    // unfold: a pixel with no decals never pays for it, and a bucket with
    // many never pays more than once.
    vec3 groundP = vec3(0.0);
    bool haveGroundP = false;
    for (uint i = 0u; i < count; i++) {
        uint at = 4u + 16u * decalWords[indexAt + first + i];
        vec4 a = decalFloats(at);       // centre x, z, half x, half z
        vec4 b = decalFloats(at + 4u);  // cos, sin, layer, opacity
        // Decal frame, scaled to -1..1 over the half-sizes (x, z). Outside
        // its bounding circle in plan (another cell's decal in this bucket,
        // or past its edge) before anything else is read: unfolding never
        // shortens the distance from the centre, but on a slope across the
        // decal's axes it can turn a point past the plan square's corner
        // back inside.
        vec2 ax = b.xy, az = vec2(-b.y, b.x);
        vec2 inv = 1.0 / a.zw;
        vec2 dxz = P.xz - a.xy;
        if (dot(dxz, dxz) >= dot(a.zw, a.zw) || b.z >= layers) continue;
        vec4 f = decalFloats(at + 8u);   // start, 1 / fade-in, end, 1 / fade-out
        vec4 mat = decalMaterials[uint(b.z)];  // x roughness factor, y metallic factor
        vec4 g = decalFloats(at + 12u);  // glow, 0, 1 / glow life, blend
        // Unfolded over the ground from the centre, so a decal across a
        // cliff's edge or into its foot continues onto the face, and over
        // any ground it is one continuous image (no projections to blend,
        // so no seams).
        if (!haveGroundP) {
            groundP = groundAt(P.xz);
            haveGroundP = true;
        }
        float stride = max(u.map.w, max(a.z, a.w) * 0.125);
        vec2 un = unfoldRay(a.xy, groundAt(a.xy), dxz, groundP, stride);
        vec2 q = vec2(dot(un, ax), dot(un, az)) * inv;
        // Square, in unfolded metres.
        float r = max(abs(q.x), abs(q.y));
        if (r >= 1.0) continue;
        // The gradients and the frame from the ground's tangent plane at P
        // levelled (hn to up): its screen steps unfolded, and the world
        // directions of the decal's +x and +z.
        vec3 lx = levelBy(dPdx, hn, 1.0), ly = levelBy(dPdy, hn, 1.0);
        vec2 qx = vec2(dot(lx.xz, ax), dot(lx.xz, az)) * inv;
        vec2 qy = vec2(dot(ly.xz, ax), dot(ly.xz, az)) * inv;
        vec3 T = levelBy(vec3(ax.x, 0.0, ax.y), hn, -1.0);
        vec3 B = levelBy(vec3(az.x, 0.0, az.y), hn, -1.0);
        vec4 m;
        float sr;
        vec4 s = decalTap(q, qx, qy, b.z, T, B, hn, m, sr);
        vec4 sn = vec4(m.xyz * s.a, m.w);
        sr *= s.a;
        // The fade-in leaves the glow alone (the explosion's flash covers
        // it; Recoil's scars fade in alpha only, over 6 frames).
        float fadeIn = f.y > 0.0 ? clamp((time - f.x) * f.y, 0.0, 1.0) : 1.0;
        float shown = b.w * (1.0 - smoothstep(0.85, 1.0, r));
        if (f.w > 0.0) shown *= clamp((f.z - time) * f.w, 0.0, 1.0);
        float opacity = shown * fadeIn;
        float cover = s.a * opacity;
        float keep = 1.0 - cover;
        vec3 colour = g.w > 0.5 ? o.albedo * MODULATE * s.rgb : s.rgb;
        o.albedo = o.albedo * keep + colour * opacity;
        o.normal = o.normal * keep + sn.xyz * opacity;
        // The layer's material replaces the ground's by the same coverage:
        // soot (a rough dielectric) takes the gloss off what it lands on.
        // `sr` is the coverage-weighted texture roughness, `mat.x` the
        // layer's factor (glTF's factor x texture); metallic is uniform
        // across the layer.
        o.roughness = o.roughness * keep + (mat.x * sr) * opacity;
        o.metallic = o.metallic * keep + mat.y * cover;
        o.glow *= keep;
        // Recoil's decay: the glow less the time since placement over the
        // glow life (a weaker glow ends sooner).
        float heat = g.z > 0.0 ? g.x - (time - f.x) * g.z : 0.0;
        if (heat > 0.0) {
            heat += smoothstep(0.75, 1.0, heat) * 0.2 * abs(sin(0.6 * time));
            // Held to the centre by the unfolded distance (round), over
            // Recoil's half-diagonal sqrt(hx^2 + hz^2 + height^2), the
            // height there the smaller half.
            heat *= pow(smoothstep(0.9, 0.1, length(q) * 0.57735), 7.0);
            vec3 e = blackBody(sn.w * mix(1.0, 3700.0, heat)) * heat;
            o.glow += pow(min(e, vec3(16.0)), vec3(2.2)) * s.a * shown;
        }
    }
    o.normal = normalize(o.normal);
    return o;
}
