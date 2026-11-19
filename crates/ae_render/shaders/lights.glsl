// The clustered lights (src/lights.rs): the frame's light list and its
// per-cluster index lists, in their own descriptor set. A shader defines
// LIGHT_SET (the set number its pipeline layout gives the lights) before
// including this; the cull (lights_cluster.comp) defines LIGHTS_WRITE.
//
// The view is cut into LIGHT_TILES_X x LIGHT_TILES_Y screen tiles and
// LIGHT_SLICES slices of view depth, spaced exponentially from
// lf.slicing.x; a tile lists the lights whose bounding spheres reach its
// frustum (lights_tiles.comp), and a cluster those of its tile whose
// bounds reach its box (lights_cluster.comp), both in light order. Kinds and falloffs after ~/particles (particle_lights.glsl):
// point, cone (between two cosines about its axis), beam (a capsule along
// a segment). Distances in the falloff are in units of the light's source
// size (profile.z), so one formula serves a muzzle flash and a burning
// tank alike.

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
    float range;     // nothing past it
    vec3 color;      // linear, may exceed 1
    uint kind;       // LIGHT_POINT, LIGHT_CONE, LIGHT_BEAM
    vec3 direction;  // a cone's axis, a beam's segment (unit)
    float halfLength; // a beam's half length
    vec4 profile;    // x, y: a cone's inner and outer cosines; z: source size (> 0)
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

// ~/particles' falloff, distances over the source size: (1 - d²/r²)² /
// (1 + d²/s²).
float lightFalloff(Light l, float d2) {
    float edge = max(1.0 - d2 / (l.range * l.range), 0.0);
    return edge * edge / (1.0 + d2 / (l.profile.z * l.profile.z));
}

// Light l at p: the vector to it (to the beam's nearest point) and its
// attenuation.
float lightAt(Light l, vec3 p, out vec3 toLight) {
    vec3 source = l.position;
    float ends = 1.0;
    if (l.kind == LIGHT_BEAM) {
        float along = dot(p - l.position, l.direction);
        source += l.direction * clamp(along, -l.halfLength, l.halfLength);
        ends = 1.0 - smoothstep(0.85, 1.0, abs(along) / max(l.halfLength, 1e-4));
    }
    toLight = source - p;
    float d2 = dot(toLight, toLight);
    float a = lightFalloff(l, d2) * ends;
    if (l.kind == LIGHT_CONE && d2 > 0.0) {
        a *= smoothstep(l.profile.y, l.profile.x, dot(-toLight, l.direction) * inversesqrt(d2));
    }
    return a;
}

// The lights to sum at a cluster: (first, count), and whether they are
// the whole list (the reference) rather than the cluster's indices.
uvec2 lightRange(uint cluster, out bool unculled) {
    unculled = lf.grid.w == 2u;
    return unculled ? uvec2(0u, lightCount) : clusterRange[cluster];
}

// Diffuse light (to multiply by the albedo) at p with normal n.
vec3 lightsDiffuse(uint cluster, vec3 p, vec3 n) {
    if (!lightsOn()) {
        return vec3(0.0);
    }
    bool unculled;
    uvec2 range = lightRange(cluster, unculled);
    vec3 sum = vec3(0.0);
    for (uint k = 0u; k < range.y; ++k) {
        Light l = lights[unculled ? k : lightIndex[range.x + k]];
        vec3 toLight;
        float a = lightAt(l, p, toLight);
        float d = length(toLight);
        float lambert = d > 1e-4 ? max(dot(n, toLight / d), 0.0) : 1.0;
        sum += l.color * (a * lambert);
    }
    return sum;
}

// Light arriving at p from every direction (a medium: smoke, dust).
vec3 lightsAround(uint cluster, vec3 p) {
    if (!lightsOn()) {
        return vec3(0.0);
    }
    bool unculled;
    uvec2 range = lightRange(cluster, unculled);
    vec3 sum = vec3(0.0);
    for (uint k = 0u; k < range.y; ++k) {
        Light l = lights[unculled ? k : lightIndex[range.x + k]];
        vec3 toLight;
        sum += l.color * lightAt(l, p, toLight);
    }
    return sum;
}

#endif
