// A model's material (src/models.rs) lit like the darts (units.frag): sun
// past the hills, the units and the particles, sky, fog. Base colour,
// alpha mask (the masked pipeline, MASKED defined), normal map, occlusion
// and emissive; no metallic-roughness yet.
#define MODELS
#include "units_common.glsl"
#include "units_models.glsl"

layout(location = 0) in vec3 inWorld;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec4 inUv;
layout(location = 4) in vec4 inColor;
layout(location = 5) flat in uint inMaterial;

layout(location = 0) out vec4 outColor;

// Texture k of the material (0 base colour, 1 normal, 2 occlusion,
// 3 emissive) at its UV set, through its KHR_texture_transform.
vec4 sampleTex(Material m, uint k) {
    vec2 uv = m.uvSet[k] == 0u ? inUv.xy : inUv.zw;
    vec4 a = m.xform[2u * k];
    uv = mat2(a.xy, a.zw) * uv + m.xform[2u * k + 1u].xy;
    return texture(textures[m.tex[k]], uv);
}

void main() {
    Material m = materials[inMaterial];
    vec4 base = m.baseColor * inColor;
    if (m.tex.x != 0u) {
        base *= sampleTex(m, 0u);
    }
#ifdef MASKED
    if (base.a < m.emissive.w) {
        discard;
    }
#endif
    vec3 n = normalize(inNormal);
    vec3 t = inTangent.xyz;
    if (m.params.z > 0.5 && dot(n, u.camPos.xyz - inWorld) < 0.0) {
        n = -n;
        t = -t;
    }
    if (m.tex.y != 0u && dot(t, t) > 1e-12) {
        t = normalize(t - n * dot(n, t));
        vec3 b = cross(n, t) * inTangent.w;
        vec3 tn = sampleTex(m, 1u).xyz * 2.0 - 1.0;
        tn.xy *= m.params.x;
        n = normalize(mat3(t, b, n) * tn);
    }
    float ao = m.tex.z != 0u ? mix(1.0, sampleTex(m, 2u).r, m.params.y) : 1.0;
    vec3 emissive = m.emissive.rgb;
    if (m.tex.w != 0u) {
        emissive *= sampleTex(m, 3u).rgb;
    }
    vec3 tint;
    float units = unitSunAt(inWorld, n, tint);
    vec3 diff = max(dot(n, normalize(u.sun.xyz)), 0.0) * min(sunAt(inWorld), units) * tint;
    vec3 sky = vec3(0.45, 0.55, 0.7);
    vec3 color = base.rgb * (vec3(1.0, 0.95, 0.85) * 1.4 * diff + sky * (0.35 + 0.25 * n.y) * ao) + emissive;
    float dist = distance(inWorld, u.camPos.xyz);
    float fog = 1.0 - exp(-pow(dist * u.fog.w, 1.5));
    outColor = vec4(mix(color, u.fog.rgb, fog), 1.0);
}
