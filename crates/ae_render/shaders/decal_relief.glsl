// Robust Cone Step Mapping, Bán et al. (2024), sections 3-5.
// https://doi.org/10.2312/sr.20241146
// The paper's cell-max step proposes a span. A Lipschitz cone certifies
// a prefix and exact bilinear-cell tests certify the remaining span,
// including crossings that enter and exit the surface within one cell.

const float NEUTRAL_HEIGHT = 128.0 / 255.0;

struct ReliefRay {
    vec2 uv;                // UV at maximum height
    vec2 travel;            // UV displacement over t=[0,1]
    vec2 bounds;            // min and max in canonical UNORM height units
    float range;            // world metres per canonical height unit
};

struct ReliefHit {
    vec2 uv;
    float heightWorld;
    bool valid;
    uint steps;
};

ivec2 reliefCell(vec2 p, vec2 v) {
    vec2 base = floor(p);
    bvec2 onBackwardFace = bvec2(
        p.x == base.x && v.x < 0.0,
        p.y == base.y && v.y < 0.0
    );
    return ivec2(base) - ivec2(onBackwardFace);
}

vec2 reliefCrossings(vec2 p, vec2 v, ivec2 cell) {
    vec2 face = vec2(cell) + vec2(greaterThan(v, vec2(0.0)));
    return vec2(v.x == 0.0 ? 2.0 : (face.x - p.x) / v.x,
                v.y == 0.0 ? 2.0 : (face.y - p.y) / v.y);
}

// First root of height(ray) - bilinearHeight(ray) inside one cell.
// Parameterize by a fraction of the cell span to keep the coefficients scaled.
float reliefPatchRoot(vec4 h, vec2 p, vec2 v, float z, float dz) {
    vec2 side = h.yz - h.xx;
    float crossTerm = h.w - h.y - h.z + h.x;
    float a = -crossTerm * v.x * v.y;
    float b = dz - dot(side, v) - crossTerm * (p.x * v.y + p.y * v.x);
    float c = z - h.x - dot(side, p) - crossTerm * p.x * p.y;
    if (c <= 0.0) return 0.0;
    float closest = a > 0.0 ? clamp(-b / (2.0 * a), 0.0, 1.0) : 1.0;
    if ((a * closest + b) * closest + c > 0.0) return 2.0;
    float towardRoot = sqrt(max(0.0, b * b - 4.0 * a * c)) - b;
    return towardRoot > 0.0 ? clamp(2.0 * c / towardRoot, 0.0, closest) : closest;
}

vec4 reliefCorners(ivec2 cell, ivec2 hi, int layer) {
    ivec2 a = clamp(cell, ivec2(0), hi);
    ivec2 b = clamp(cell + 1, ivec2(0), hi);
    return vec4(texelFetch(decalSurface, ivec3(a, layer), 0).r,
                texelFetch(decalSurface, ivec3(b.x, a.y, layer), 0).r,
                texelFetch(decalSurface, ivec3(a.x, b.y, layer), 0).r,
                texelFetch(decalSurface, ivec3(b, layer), 0).r);
}

float reliefHeight(vec2 p, ivec2 hi, int layer) {
    vec4 h = reliefCorners(ivec2(floor(p)), hi, layer);
    vec2 a = fract(p);
    return mix(mix(h.x, h.y, a.x), mix(h.z, h.w, a.x), a.y);
}

// Certify a proposed interval with bounded exact cell traversal. Return
// false if the audit budget runs out, leaving the pixel on ordinary bump.
bool reliefCheckSpan(vec2 origin, vec2 velocity, vec2 bounds, int layer, ivec2 hi,
                     float begin, float end, inout uint budget, out float hit) {
    ivec2 cell = reliefCell(origin + begin * velocity, velocity);
    float depth = bounds.y - bounds.x;
    hit = 2.0;
    for (; budget > 0u; --budget) {
        vec2 crossings = reliefCrossings(origin, velocity, cell);
        float crossAt = min(crossings.x, crossings.y);
        float stop = min(end, crossAt);
        if (stop >= begin) {
            float lowest = bounds.y - stop * depth;
            float cellMax = texelFetch(decalCones, ivec3(clamp(cell, ivec2(0), hi), layer), 0).b;
            if (lowest <= cellMax + 1e-6) {
                vec4 h = reliefCorners(cell, hi, layer);
                float span = stop - begin;
                float root = reliefPatchRoot(h, origin + begin * velocity - vec2(cell),
                                             span * velocity, bounds.y - begin * depth, -span * depth);
                if (root <= 1.0) {
                    hit = begin + span * root;
                    return true;
                }
            }
        }
        if (crossAt >= end) return true;
        if (crossings.x <= crossAt) cell.x += velocity.x < 0.0 ? -1 : 1;
        if (crossings.y <= crossAt) cell.y += velocity.y < 0.0 ? -1 : 1;
        begin = max(begin, crossAt);
    }
    return false;
}

ReliefHit reliefTrace(ReliefRay ray, int layer, uint coneBudget, uint refineBudget) {
    ReliefHit hit = ReliefHit(ray.uv, 0.0, false, 0u);
    if (any(lessThan(min(ray.uv, ray.uv + ray.travel), vec2(0.0))) ||
        any(greaterThan(max(ray.uv, ray.uv + ray.travel), vec2(1.0)))) return hit;
    float drop = ray.bounds.y - ray.bounds.x;
    if (drop <= 0.0 || coneBudget == 0u) return hit;
    float lateral = length(ray.travel);
    float t = 0.0;
    ivec2 size = textureSize(decalSurface, 0).xy;
    vec2 origin = ray.uv * vec2(size) - 0.5;
    vec2 velocity = ray.travel * vec2(size);
    uint auditBudget = min(4u * coneBudget, 256u);
    for (uint i = 0u; i < coneBudget; ++i) {
        hit.steps = i + 1u;
        vec2 uv = ray.uv + t * ray.travel;
        float h = reliefHeight(origin + t * velocity, size - 1, layer);
        float gap = ray.bounds.y - t * drop - h;
        if (gap <= 0.0 || lateral == 0.0) {
            hit.uv = uv;
            hit.heightWorld = (h - NEUTRAL_HEIGHT) * ray.range;
            hit.valid = true;
            return hit;
        }
        vec2 cones = textureLod(decalCones, vec3(uv, layer), 0.0).rg;
        vec2 dt = gap * cones / (vec2(lateral) + drop * cones);
        ivec2 cell = reliefCell(origin + t * velocity, velocity);
        vec2 crossings = reliefCrossings(origin, velocity, cell);
        // At an exact wall, reconstructing the position can round back
        // into the previous cell. Advance that axis to the next wall.
        if (any(lessThanEqual(crossings, vec2(t)))) {
            cell += ivec2(sign(velocity)) * ivec2(lessThanEqual(crossings, vec2(t)));
            crossings = reliefCrossings(origin, velocity, cell);
        }
        float end = min(1.0, max(t + dt.x, min(crossings.x, crossings.y)));
        // Reserve a numerical margin in the independently certified prefix.
        float safe = min(end, t + max(0.0, dt.y - 2e-6));
        float at;
        if (!reliefCheckSpan(origin, velocity, ray.bounds, layer, size - 1, safe, end, auditBudget, at)) return hit;
        if (at <= 1.0) {
            // The exact cell root is the refinement. Optionally narrow a
            // round-off-sized bracket on the same bilinear surface.
            float lo = max(safe, at - 2e-6);
            float hi = min(end, at + 2e-6);
            if (refineBudget > 0u && ray.bounds.y - lo * drop > reliefHeight(origin + lo * velocity, size - 1, layer)
                && ray.bounds.y - hi * drop <= reliefHeight(origin + hi * velocity, size - 1, layer)) {
                for (uint k = 0u; k < refineBudget && hi - lo > 1e-7; ++k) {
                    float mid = (lo + hi) * 0.5;
                    float surface = reliefHeight(origin + mid * velocity, size - 1, layer);
                    if (ray.bounds.y - mid * drop > surface) lo = mid;
                    else hi = mid;
                }
                at = (lo + hi) * 0.5;
            }
            hit.uv = ray.uv + at * ray.travel;
            hit.heightWorld = (ray.bounds.y - at * drop - NEUTRAL_HEIGHT) * ray.range;
            hit.valid = true;
            return hit;
        }
        if (end >= 1.0 || end <= t) return hit;
        t = end;
    }
    return hit;
}
