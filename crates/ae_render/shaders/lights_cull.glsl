// Shared by the two culls of lights.cluster (src/lights.rs):
// lights_tiles.comp (each screen tile's lights) and lights_cluster.comp
// (each cluster's, from its tile's). One workgroup of 64 per tile or
// cluster: candidates are walked in order, 64 at a time, each lane testing
// one, and the hits compacted by a prefix sum over the workgroup into
// `picked`, so a list keeps light order whatever the scheduling (the sums
// over it are the same every run). Past `cap` entries a list is cut, and
// the cut is counted.
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require
#include "bindless.glsl"
#define AE_LIGHT_BASE 0
#define LIGHTS_WRITE
#include "lights.glsl"

layout(local_size_x = 64) in;

shared uint subgroupHits[64];

vec3 toView(vec3 world) {
    vec3 d = world - lf.camPos.xyz;
    return vec3(dot(d, lf.camRight.xyz), dot(d, lf.camUp.xyz), dot(d, lf.camForward.xyz));
}

vec3 toViewDir(vec3 d) {
    return vec3(dot(d, lf.camRight.xyz), dot(d, lf.camUp.xyz), dot(d, lf.camForward.xyz));
}

// The world point at viewport pixel px and view depth z.
vec3 unproject(vec2 px, float z) {
    vec2 ndc = px / lf.slicing.zw * 2.0 - 1.0;
    vec4 w = lf.invViewProj * vec4(ndc, lf.camPos.w / z, 1.0);
    return w.xyz / w.w;
}

// The pixel rectangle of tile t (x, y).
vec4 tileRect(uvec2 t) {
    vec2 size = lf.slicing.zw / vec2(lf.grid.xy);
    return vec4(vec2(t) * size, vec2(t + 1u) * size);
}

// One step of the ordered compaction: every lane with `hit` gets its
// place (from `count`); returns the hits of all lanes. Call in uniform
// control flow.
uint compact(bool hit, uint count, out uint place) {
    uint h = hit ? 1u : 0u;
    uint before = subgroupExclusiveAdd(h);
    uint total = subgroupAdd(h);
    if (subgroupElect()) {
        subgroupHits[gl_SubgroupID] = total;
    }
    barrier();
    place = count + before;
    uint sum = 0u;
    for (uint s = 0u; s < gl_NumSubgroups; ++s) {
        place += s < gl_SubgroupID ? subgroupHits[s] : 0u;
        sum += subgroupHits[s];
    }
    barrier();
    return sum;
}
