// The clustered lights (src/lights.rs): the frame's light list and its
// per-cluster index lists, in their own descriptor set. A shader defines
// LIGHT_SET (the set number its pipeline layout gives the lights) before
// including this; the cull (lights_cluster.comp) defines LIGHTS_WRITE.
//
// The view is cut into LIGHT_TILES_X x LIGHT_TILES_Y screen tiles and
// LIGHT_SLICES slices of view depth, spaced exponentially from
// lf.slicing.x; a tile lists the lights whose bounding spheres reach its
// frustum (lights_tiles.comp), and a cluster those of its tile whose
// bounds reach its box (lights_cluster.comp), both in light order. Kinds:
// point, cone (between two cosines about its axis), beam (a capsule along
// a segment). A light gives illuminance in lux (src/lighting.rs): intensity
// (profile.w, candela) times the glTF quartic range window over the squared
// distance, the near field regularised by the source radius (profile.z).
// The receiver masks it by its cosine once and multiplies by albedo/pi.

#ifndef LIGHTS_GLSL
#define LIGHTS_GLSL

#define LIGHT_POINT 1u
#define LIGHT_CONE 2u
#define LIGHT_BEAM 4u
#define LIGHT_TILES_X 40u
#define LIGHT_TILES_Y 24u
#define LIGHT_TILES (LIGHT_TILES_X * LIGHT_TILES_Y)
#define LIGHT_SLICES 16u
#define LIGHT_CLUSTERS (LIGHT_TILES * LIGHT_SLICES)
// A tile's list (lights_tiles.comp) and the clusters' index list
// (lights.rs PER_TILE, INDEX_CAPACITY).
#define LIGHT_PER_TILE 1024u
// Most lights a frame lists (lights.rs MAX_LIGHTS).
#define LIGHT_MAX 8192u
#define LIGHT_INDEX_CAPACITY (LIGHT_CLUSTERS * 32u)

struct Light {
    vec3 position;
    float range;      // the authored cutoff (metres); 0: none
    vec3 color;       // linear tint, unit luminance
    uint kind;        // LIGHT_POINT, LIGHT_CONE, LIGHT_BEAM
    vec3 direction;   // a cone's axis, a beam's segment (unit)
    float halfLength; // a beam's half length
    vec4 profile;     // x, y: a cone's inner and outer cosines; z: source radius (m); w: intensity (cd)
};

layout(std140, set = LIGHT_SET, binding = 0) uniform LightFrame {
    mat4 viewProj;    // the cull's light bounds
    mat4 invViewProj; // the cull's tile corners
    vec4 camPos;      // xyz eye; w: the near plane (infinite reverse-Z: depth = near / z)
    vec4 camRight;
    vec4 camUp;
    vec4 camForward;
    uvec4 grid;       // tiles x, y, slices; w: 1 when the lights are on this frame, 2 on and unculled (every light at every pixel: the reference)
    vec4 slicing;     // x: first slice's far depth; y: slices per log unit of depth; z, w: viewport size (px)
    uvec4 model;      // x: the host's lights (where the units' model lights append at); y: the units' rows; z, w: unused
    vec4 photometry;  // x: the cull cutoff (lux): bounds = sqrt(intensity / x)
} lf;

#ifdef LIGHTS_WRITE
#define LIGHTS_ACCESS
#else
#define LIGHTS_ACCESS readonly
#endif

// The host writes the count and the lights; the cull counts the indices it
// used and the clusters it cut short.
layout(std430, set = LIGHT_SET, binding = 1) LIGHTS_ACCESS buffer LightList {
    uint lightCount;
    uint indexCount;
    uint overflow;
    uint dropped;
    Light lights[];
};

// Per cluster (offset, count) into lightIndex; per light its tiles and
// slices (lights_bounds.comp: x the tile rectangle, x0 y0 x1 y1 a byte
// each, inclusive; y the slices, first and last a byte each; 0: off
// screen); per screen tile the lights that reach it (the clusters'
// candidates).
layout(std430, set = LIGHT_SET, binding = 2) LIGHTS_ACCESS buffer LightClusters {
    uvec2 clusterRange[LIGHT_CLUSTERS];
    uvec2 lightBounds[LIGHT_MAX];
    uint tileCount[LIGHT_TILES];
    uint tileLights[LIGHT_TILES * LIGHT_PER_TILE];
    uint lightIndex[];
};

bool lightsOn() {
    return lf.grid.w != 0u;
}

// The slice of view depth z.
uint lightSlice(float z) {
    float s = z > lf.slicing.x ? 1.0 + log(z / lf.slicing.x) * lf.slicing.y : 0.0;
    return min(uint(s), lf.grid.z - 1u);
}

// The tile of the pixel at `px` (in the viewport).
uvec2 lightTile(vec2 px) {
    return min(uvec2(max(px, vec2(0.0)) * vec2(lf.grid.xy) / lf.slicing.zw), lf.grid.xy - 1u);
}

// The cluster of the pixel at `px` (in the viewport) at view depth `z`.
uint lightCluster(vec2 px, float z) {
    uvec2 tile = lightTile(px);
    return (lightSlice(z) * lf.grid.y + tile.y) * lf.grid.x + tile.x;
}

// The cluster of a fragment (gl_FragCoord in a pass at the viewport's size).
uint lightClusterFrag(vec4 fragCoord) {
    return lightCluster(fragCoord.xy, lf.camPos.w / max(fragCoord.z, 1e-12));
}

// The cluster of world point p (any stage).
uint lightClusterAt(mat4 viewProj, vec3 p) {
    vec4 clip = viewProj * vec4(p, 1.0);
    vec2 ndc = clip.xy / max(clip.w, 1e-6);
    return lightCluster((ndc * 0.5 + 0.5) * lf.slicing.zw, clip.w);
}

// The radius at which a light is not worth drawing: its authored cutoff,
// or (without one) the distance its illuminance falls to the frame's cull
// cutoff. The bounds and the cluster cull use it; the shading window uses
// only the authored range.
float lightReach(Light l) {
    // Conservatively by the brightest channel: a saturated tint can exceed
    // the luminance intensity in one channel.
    float brightest = max(max(l.color.r, l.color.g), max(l.color.b, 0.0));
    float reach = l.profile.w > 0.0 ? sqrt(l.profile.w * brightest / max(lf.photometry.x, 1e-9)) : 0.0;
    return l.range > 0.0 ? min(l.range, reach) : reach;
}

// The directional illuminance (lux) of light l arriving at p, before the
// receiver cosine. `L` is the unit direction to the light (to the beam's
// nearest point). A range of 0 is no authored cutoff; the distance is
// regularised by the source radius, and a cone cuts between its cosines.
vec3 lightIlluminance(Light l, vec3 p, out vec3 L) {
    vec3 source = l.position;
    float ends = 1.0;
    if (l.kind == LIGHT_BEAM) {
        float along = dot(p - l.position, l.direction);
        source += l.direction * clamp(along, -l.halfLength, l.halfLength);
        ends = 1.0 - smoothstep(0.85, 1.0, abs(along) / max(l.halfLength, 1e-4));
    }
    vec3 toLight = source - p;
    float d2 = dot(toLight, toLight);
    if (d2 <= 0.0 || l.profile.w <= 0.0) {
        L = vec3(0.0, 1.0, 0.0);
        return vec3(0.0);
    }
    L = toLight * inversesqrt(d2);
    float window = 1.0;
    if (l.range > 0.0) {
        float q = d2 / (l.range * l.range);
        window = max(1.0 - q * q, 0.0);
    }
    float cone = 1.0;
    if (l.kind == LIGHT_CONE) {
        // glTF's two half-angles, clamped by the CPU: a squared cosine ramp
        // between the outer and inner cosines.
        float scale = 1.0 / max(l.profile.x - l.profile.y, 1e-5);
        float t = clamp(dot(l.direction, -L) * scale - l.profile.y * scale, 0.0, 1.0);
        cone = t * t;
    }
    float radius = l.profile.z;
    float denom = max(d2, radius * radius);
    return l.color * (l.profile.w * window * cone * ends / denom);
}

// The lights to sum at a cluster: (first, count), and whether they are
// the whole list (the reference) rather than the cluster's indices.
uvec2 lightRange(uint cluster, out bool unculled) {
    unculled = lf.grid.w == 2u;
    return unculled ? uvec2(0u, lightCount) : clusterRange[cluster];
}

// The illuminance at p (lux) from the lights that reach it, times the
// receiver cosine: what a Lambertian surface multiplies by albedo/pi.
vec3 lightsDiffuse(uint cluster, vec3 p, vec3 n) {
    if (!lightsOn()) {
        return vec3(0.0);
    }
    bool unculled;
    uvec2 range = lightRange(cluster, unculled);
    vec3 sum = vec3(0.0);
    for (uint k = 0u; k < range.y; ++k) {
        Light l = lights[unculled ? k : lightIndex[range.x + k]];
        vec3 L;
        vec3 E = lightIlluminance(l, p, L);
        sum += E * max(dot(n, L), 0.0);
    }
    return sum;
}

// The illuminance (lux) arriving at p from every direction (a medium:
// smoke, dust): no receiver cosine.
vec3 lightsAround(uint cluster, vec3 p) {
    if (!lightsOn()) {
        return vec3(0.0);
    }
    bool unculled;
    uvec2 range = lightRange(cluster, unculled);
    vec3 sum = vec3(0.0);
    for (uint k = 0u; k < range.y; ++k) {
        Light l = lights[unculled ? k : lightIndex[range.x + k]];
        vec3 L;
        sum += lightIlluminance(l, p, L);
    }
    return sum;
}

#endif
