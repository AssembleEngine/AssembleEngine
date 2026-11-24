// Decals (src/decals.rs), applied to the terrain's albedo, normal and
// surface material (roughness, metallic) before lighting.
// Needs terrain_common.glsl (u, groundAt).
//
// The buffer, in 32-bit words: a header (bucket mask, debug view, cell size as float
// bits, decal capacity), the decals (17 words each: centre x, z, half-sizes
// x, z, cos, sin, layer, opacity, start time, 1 / fade-in, end time (0:
// none), 1 / fade-out (0: none), glow, height range, 1 / glow life (0: no glow),
// blend (0 replace, 1 modulate), conservative clearance), one word
// per bucket of the spatial hash of world cells (first index << 8 | count),
// the indices.

// Specialization constant 1: false compiles the decals out (the terrain
// pipeline used while there are none).
layout(constant_id = 1) const bool DECALS = false;
layout(constant_id = 2) const uint DECAL_RELIEF_MODE = 1u;

layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer DecalBuf { uint decalWords[]; } ae_DecalBuf[];
#define decalWords ae_DecalBuf[AE_BUF(8)].decalWords
#define decalAtlas AE_S2D_ARRAY(9, 5)
#define decalSurface AE_S2D_ARRAY(10, 5)
#define decalNormals AE_S2D_ARRAY(14, 5)
#define decalCones AE_S2D_ARRAY(15, 1)
#include "decal_relief.glsl"

const uint DECAL_WORDS = 17u;
const uint RELIEF_DISABLED = 0u;
const uint RELIEF_CONE = 2u;
const float DEFAULT_HEIGHT_RATIO = 24.0 / 512.0;
// Metallic factor, texture metalness weight, minimum and maximum height.
layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) readonly buffer DecalMaterialBuf { vec4 decalMaterials[]; } ae_DecalMaterialBuf[];
#define decalMaterials ae_DecalMaterialBuf[AE_BUF(9)].decalMaterials

vec4 decalFloats(uint at) {
    return uintBitsToFloat(uvec4(decalWords[at], decalWords[at + 1u], decalWords[at + 2u], decalWords[at + 3u]));
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

// Recoil's black body (shadertoy MslSDl): the colour of a body at `t`
// kelvin, its brightness rising as t^4 (1 near 2500 K), in gamma space.
vec3 blackBody(float t) {
    float cu = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t * t) / (1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t * t);
    float v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t * t) / (1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t * t);
    float x = 3.0 * cu / (2.0 * cu - 8.0 * v + 4.0);
    float y = 2.0 * v / (2.0 * cu - 8.0 * v + 4.0);
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

vec3 decalTangentNormal(vec2 xy) {
    return vec3(xy, sqrt(max(0.0, 1.0 - dot(xy, xy))));
}

vec3 reorientDecalNormal(vec3 base, vec3 detail) {
    vec3 t = base + vec3(0.0, 0.0, 1.0);
    vec3 reflected = detail * vec3(-1.0, -1.0, 1.0);
    return normalize(t * (dot(t, reflected) / max(t.z, 1e-4)) - reflected);
}

void decalGround(vec3 P, vec3 hn, out vec3 groundP, out vec3 groundX, out vec3 groundZ) {
    groundP = groundAt(P.xz);
    // The shortest rotation from up to hn, applied to the flat X and Z axes.
    vec2 k = hn.xz / (1.0 + hn.y);
    float xz = -hn.x * k.y;
    groundX = vec3(1.0 - hn.x * k.x, -hn.x, xz);
    groundZ = vec3(xz, -hn.z, 1.0 - hn.z * k.y);
}

vec2 decalPoint(vec4 a, vec4 b, vec3 P, vec3 groundP) {
    float stride = max(u.map.w, max(a.z, a.w) * 0.125);
    vec2 un = unfoldRay(a.xy, groundAt(a.xy), P.xz - a.xy, groundP, stride);
    return vec2(dot(un, b.xy), dot(un, vec2(-b.y, b.x))) / a.zw;
}

// Local differential of the unfolded projection. Exact for a plane;
// the local slope approximates the full unfolded ray on curved ground.
void decalUvAxes(vec4 a, vec4 b, vec3 groundX, vec3 groundZ, out vec3 gradU, out vec3 gradV) {
    gradU = (b.x * groundX + b.y * groundZ) * (0.5 / a.z);
    gradV = (-b.y * groundX + b.x * groundZ) * (0.5 / a.w);
}

vec2 decalUvStep(vec3 step, vec3 gradU, vec3 gradV) {
    return vec2(dot(step, gradU), dot(step, gradV));
}

// x is fade-in, y is fade-out. Glow keeps its existing fade-in exemption.
vec2 decalFades(vec4 f, float time) {
    return vec2(f.y > 0.0 ? clamp((time - f.x) * f.y, 0.0, 1.0) : 1.0,
                f.w > 0.0 ? clamp((f.z - time) * f.w, 0.0, 1.0) : 1.0);
}

struct DecalTrace {
    ReliefHit hit;
    float quality;
    float depth;
    bool overlap;
};

// Smooth minimum of 1 and the safe scale. The shoulder starts fading
// before the requested ray would hit the limit.
float reliefLimit(float safeScale) {
    float shoulder = max(0.25 - abs(1.0 - safeScale), 0.0);
    return max(0.0, min(1.0, safeScale) - shoulder * shoulder);
}

float reliefDomainScale(vec2 uv, vec2 topOffset, vec2 bottomOffset) {
    vec2 negative = max(-min(topOffset, bottomOffset), 0.0);
    vec2 positive = max(max(topOffset, bottomOffset), 0.0);
    vec2 down = mix(vec2(2.0), 0.98 * uv / max(negative, 1e-8), greaterThan(negative, vec2(0.0)));
    vec2 up = mix(vec2(2.0), 0.98 * (1.0 - uv) / max(positive, 1e-8), greaterThan(positive, vec2(0.0)));
    vec2 limit = min(down, up);
    return min(limit.x, limit.y);
}

DecalTrace traceDecal(vec3 uv, vec2 gx, vec2 gy, vec2 bounds, float range, float clearance,
                      bool safeOverlap, float noV, vec3 tangentPerHeight, vec3 gradU, vec3 gradV) {
    DecalTrace result = DecalTrace(ReliefHit(uv.xy, 0.0, false, 0u), 0.0, 0.0, false);
    if (safeOverlap && clearance <= 0.0 && range > 0.0) {
        result.overlap = true;
        return result;
    }
    if (range <= 0.0 || isinf(range) || isnan(range) || bounds.x >= bounds.y
        || noV <= uintBitsToFloat(u.decalRelief.y)) return result;
    vec2 signedBounds = bounds - NEUTRAL_HEIGHT;
    float heightMax = max(abs(signedBounds.x), abs(signedBounds.y)) * range;
    vec2 uvPerHeight = decalUvStep(tangentPerHeight, gradU, gradV);
    float minDot = uintBitsToFloat(u.decalRelief.y);
    float quality = smoothstep(minDot, min(1.0, minDot + 0.25), noV);
    if ((u.decalRelief.x & 4u) != 0u) {
        float area = abs(determinant(mat2(gx, gy)));
        if (area <= 1e-12) return result;
        float pixels = length(inverse(mat2(gx, gy)) * (uvPerHeight * heightMax));
        quality *= smoothstep(u.effects.z, u.effects.w, pixels);
    }
    vec2 topOffset = uvPerHeight * (signedBounds.y * range * quality);
    vec2 bottomOffset = uvPerHeight * (signedBounds.x * range * quality);
    // Shorten the physical ray before either end can leave the texture.
    // Keep its intersections on the original bilinear heightfield.
    float domainScale = reliefLimit(reliefDomainScale(uv.xy, topOffset, bottomOffset));
    // Each cone proposal reaches at least the next cell wall. Budget
    // both UV axes plus the start/end cells and repeat visits in audits.
    vec2 cells = abs(topOffset - bottomOffset) * vec2(textureSize(decalSurface, 0).xy);
    float cellBudget = max(0.0, min(float(u.decalRelief.z), float(min(4u * u.decalRelief.z, 256u)) / 3.0) - 3.0);
    float travelCells = (cells.x + cells.y) * domainScale;
    quality *= domainScale * reliefLimit(cellBudget / max(travelCells, 1e-8));
    if (safeOverlap) {
        float shift = length(tangentPerHeight.xz) * heightMax * quality;
        if (shift >= clearance) {
            result.overlap = true;
            return result;
        }
        quality *= 1.0 - smoothstep(clearance * 0.7, clearance, shift);
    }
    result.quality = quality;
    if (quality <= 0.0) return result;
    // Fade the uniform ray amplitude, then trace the original canonical
    // heightfield. No spatial mask is baked into the intersection.
    float rayRange = quality * range;
    vec2 start = uv.xy + uvPerHeight * (signedBounds.y * rayRange);
    vec2 travel = uvPerHeight * ((bounds.x - bounds.y) * rayRange);
    result.hit = reliefTrace(ReliefRay(start, travel, bounds, rayRange), int(uv.z), u.decalRelief.z, u.decalRelief.w);
    if (result.hit.valid) {
        result.depth = clamp((signedBounds.y - result.hit.heightWorld / rayRange) / (bounds.y - bounds.x), 0.0, 1.0);
    }
    return result;
}

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
// Decal normals are blended in the terrain tangent frame. Glow (Recoil's
// scars) is the glow mask lit as a black body, hotter with the
// decal's glow, which fades linearly over its glow life, flickers near
// full, and is held to the centre.
Decals applyDecals(vec3 albedo, float roughness, float metallic, vec3 n, vec3 P,
                   vec3 dPdx, vec3 dPdy, vec3 hn, vec3 eye, uint debug, out vec3 debugColour) {
    Decals o = Decals(albedo, n, roughness, metallic, vec3(0.0));
    const uint mode = DECAL_RELIEF_MODE;
    debugColour = vec3(0.0);
    vec3 scarNormal = vec3(0.0, 0.0, 1.0);
    uint mask = decalWords[0];
    float cellSize = uintBitsToFloat(decalWords[2]);
    ivec2 c = ivec2(floor(P.xz / cellSize));
    uint bucketsAt = 4u + DECAL_WORDS * decalWords[3];
    uint indexAt = bucketsAt + mask + 1u;
    uint entry = decalWords[bucketsAt + ((uint(c.x) * 73856093u ^ uint(c.y) * 19349663u) & mask)];
    uint first = entry >> 8u, count = entry & 255u;
    float time = u.effects.x;
    float layers = float(textureSize(decalAtlas, 0).z);
    // The pixel's own ground, gathered on the first decal that reaches the
    // unfold: a pixel with no decals never pays for it, and a bucket with
    // many never pays more than once.
    vec3 groundP = vec3(0.0);
    vec3 groundX = vec3(0.0), groundZ = vec3(0.0);
    bool haveGroundP = false;
    uint policy = u.decalRelief.x & 3u;
    float noV = 0.0;
    vec3 tangentPerHeight = vec3(0.0);
    if (mode == RELIEF_CONE && count > 0u) {
        vec3 V = normalize(eye - P);
        noV = dot(hn, V);
        if (noV > 0.0) tangentPerHeight = V / noV - hn;
    }
    DecalTrace dominant = DecalTrace(ReliefHit(vec2(0.0), 0.0, false, 0u), 0.0, 0.0, false);
    if (mode == RELIEF_CONE && policy == 1u && noV > uintBitsToFloat(u.decalRelief.y)) {
        // Experimental: the newest visible source at the original UV
        // supplies one common world displacement for every decal here.
        for (uint i = count; i > 0u; --i) {
            uint at = 4u + DECAL_WORDS * decalWords[indexAt + first + i - 1u];
            vec4 a = decalFloats(at), b = decalFloats(at + 4u);
            vec2 dxz = P.xz - a.xy;
            if (dot(dxz, dxz) >= dot(a.zw, a.zw) || b.z >= layers || b.w <= 0.0) continue;
            vec2 bounds = decalMaterials[uint(b.z)].zw;
            if (bounds.x >= bounds.y) continue;
            if (!haveGroundP) {
                decalGround(P, hn, groundP, groundX, groundZ);
                haveGroundP = true;
            }
            vec2 q = decalPoint(a, b, P, groundP);
            if (max(abs(q.x), abs(q.y)) >= 1.0) continue;
            vec3 uv = vec3(q * 0.5 + 0.5, b.z);
            vec2 fades = decalFades(decalFloats(at + 8u), time);
            // Fix the winner in world space even as the texture footprint
            // and its filtered alpha change with the camera.
            if (textureLod(decalAtlas, uv, 0.0).a * b.w * fades.x * fades.y <= 0.0) continue;
            vec3 gu, gv;
            decalUvAxes(a, b, groundX, groundZ, gu, gv);
            vec2 gx = decalUvStep(dPdx, gu, gv), gy = decalUvStep(dPdy, gu, gv);
            float range = decalFloats(at + 12u).y * u.effects.y * fades.x * fades.y;
            dominant = traceDecal(uv, gx, gy, bounds, range, 0.0, false, noV, tangentPerHeight, gu, gv);
            break;
        }
    }
    for (uint i = 0u; i < count; i++) {
        uint at = 4u + DECAL_WORDS * decalWords[indexAt + first + i];
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
        vec4 mat = decalMaterials[uint(b.z)];
        vec4 g = decalFloats(at + 12u);  // glow, height range, 1 / glow life, blend
        // Unfolded over the ground from the centre, so a decal across a
        // cliff's edge or into its foot continues onto the face, and over
        // any ground it is one continuous image (no projections to blend,
        // so no seams).
        if (!haveGroundP) {
            decalGround(P, hn, groundP, groundX, groundZ);
            haveGroundP = true;
        }
        vec2 q = decalPoint(a, b, P, groundP);
        // Square, in unfolded metres.
        float r = max(abs(q.x), abs(q.y));
        if (r >= 1.0) continue;
        vec3 gradU, gradV;
        decalUvAxes(a, b, groundX, groundZ, gradU, gradV);
        vec2 gx = decalUvStep(dPdx, gradU, gradV);
        vec2 gy = decalUvStep(dPdy, gradU, gradV);
        vec3 uv = vec3(q * 0.5 + 0.5, b.z);
        vec2 fades = decalFades(f, time);
        float heightRange = g.y * u.effects.y;
        if (isinf(heightRange) || isnan(heightRange)) heightRange = 0.0;
        if (mode == RELIEF_CONE) {
            DecalTrace trace = dominant;
            if (policy == 1u) {
                vec2 moved = uv.xy + decalUvStep(tangentPerHeight * trace.hit.heightWorld, gradU, gradV);
                if (trace.hit.valid) uv.xy += (moved - uv.xy) * reliefLimit(reliefDomainScale(uv.xy, moved - uv.xy, vec2(0.0)));
            } else {
                float clearance = uintBitsToFloat(decalWords[at + 16u]);
                trace = traceDecal(uv, gx, gy, mat.zw, heightRange * fades.x * fades.y,
                                   clearance, policy == 0u, noV, tangentPerHeight, gradU, gradV);
                if (trace.hit.valid) uv.xy = trace.hit.uv;
            }
            if (debug == 7u) debugColour = vec3(float(trace.hit.steps) / max(1.0, float(u.decalRelief.z)));
            if (debug == 8u) debugColour = vec3(trace.depth);
            if (debug == 9u) debugColour = vec3(uv.xy, trace.hit.valid ? 1.0 : 0.0);
            if (debug == 10u) debugColour = vec3(trace.quality > 0.0 && !trace.hit.valid ? 1.0 : 0.0, 0.0, 0.0);
            if (debug == 11u) debugColour = vec3(0.0, trace.overlap ? 1.0 : 0.0, 0.0);
            if (debug == 12u) debugColour = vec3(trace.quality, trace.hit.valid ? trace.quality : 0.0, 1.0 - trace.quality);
        }
        if (debug == 5u) debugColour = vec3(textureLod(decalCones, uv, 0.0).r);
        if (debug == 6u) {
            ivec2 dim = textureSize(decalCones, 0).xy;
            ivec2 cell = clamp(ivec2(floor(uv.xy * vec2(dim) - 0.5)), ivec2(0), dim - 1);
            vec2 cellBounds = texelFetch(decalCones, ivec3(cell, int(uv.z)), 0).ba;
            debugColour = vec3(cellBounds.y, cellBounds.x, cellBounds.x - cellBounds.y);
        }
        vec4 s = textureGrad(decalAtlas, uv, gx, gy);
        vec4 m = textureGrad(decalSurface, uv, gx, gy);
        // The fade-in leaves the glow alone (the explosion's flash covers
        // it; Recoil's scars fade in alpha only, over 6 frames).
        float shown = b.w * (1.0 - smoothstep(0.85, 1.0, r)) * fades.y;
        float opacity = shown * fades.x;
        float cover = s.a * opacity;
        vec3 colour = g.w > 0.5 ? o.albedo * MODULATE * s.rgb : s.rgb;
        o.albedo = mix(o.albedo, colour, cover);
        o.roughness = mix(o.roughness, m.b, cover);
        o.metallic = mix(o.metallic, mat.x + mat.y * m.g, cover);
        if (cover > 0.0) {
            if (mode != RELIEF_DISABLED || debug == 4u) {
                vec3 tn = decalTangentNormal(textureGrad(decalNormals, uv, gx, gy).rg);
                if (debug == 4u) debugColour = 0.5 + 0.5 * tn;
                if (mode != RELIEF_DISABLED) {
                    vec2 xy = tn.xy * (0.5 * inv * heightRange / (max(tn.z, 1e-4) * DEFAULT_HEIGHT_RATIO));
                    vec3 detail = normalize(vec3(xy.x * ax + xy.y * az, 1.0));
                    scarNormal = mix(scarNormal, detail, cover);
                }
            }
            if (debug == 1u) debugColour = vec3(clamp(0.5 + m.r - 128.0 / 255.0, 0.0, 1.0));
            if (debug == 2u) debugColour = vec3(cover);
        }
        o.glow *= 1.0 - cover;
        // Recoil's decay: the glow less the time since placement over the
        // glow life (a weaker glow ends sooner).
        float heat = g.z > 0.0 ? g.x - (time - f.x) * g.z : 0.0;
        if (heat > 0.0) {
            heat += smoothstep(0.75, 1.0, heat) * 0.2 * abs(sin(0.6 * time));
            // Held to the centre by the unfolded distance (round), over
            // Recoil's half-diagonal sqrt(hx^2 + hz^2 + height^2), the
            // height there the smaller half.
            heat *= pow(smoothstep(0.9, 0.1, length(q) * 0.57735), 7.0);
            vec3 e = blackBody(m.w * mix(1.0, 3700.0, heat)) * heat;
            o.glow += pow(min(e, vec3(16.0)), vec3(2.2)) * s.a * shown;
        }
    }
    if (mode != RELIEF_DISABLED && any(notEqual(scarNormal.xy, vec2(0.0)))) {
        vec3 base = vec3(dot(o.normal, groundX), dot(o.normal, groundZ), dot(o.normal, hn));
        vec3 detail = reorientDecalNormal(base, normalize(scarNormal));
        o.normal = normalize(detail.x * groundX + detail.y * groundZ + detail.z * hn);
    }
    if (debug == 3u) debugColour = 0.5 + 0.5 * o.normal;
    return o;
}
