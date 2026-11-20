// A model's material (src/models.rs) lit like the darts (units.frag): sun
// past the hills, the units and the particles, sky, fog. glTF's
// metallic-roughness: base colour, metallic and roughness (factors and
// texture), normal map, occlusion and emissive.
#define MODELS
#include "photometry.glsl"
#include "units_common.glsl"
#include "units_models.glsl"
#define LIGHT_SET 2
#include "lights.glsl"

layout(location = 0) in vec3 inWorld;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec4 inUv;
layout(location = 4) in vec4 inColor;
layout(location = 5) flat in uint inMaterial;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outGlow;

// Texture k of the material (0 base colour, 1 normal, 2 occlusion,
// 3 emissive) at its UV set, through its KHR_texture_transform.
vec4 sampleTex(Material m, uint k) {
    vec2 uv = m.uvSet[k] == 0u ? inUv.xy : inUv.zw;
    vec4 a = m.xform[2u * k];
    uv = mat2(a.xy, a.zw) * uv + m.xform[2u * k + 1u].xy;
    return texture(textures[m.tex[k]], uv);
}

// The metallic-roughness texture (roughness in G, metallic in B), through
// its own UV set and transform.
vec4 sampleMetallicRoughness(Material m) {
    vec2 uv = m.uvSet2.x == 0u ? inUv.xy : inUv.zw;
    vec4 a = m.xform2[0u];
    uv = mat2(a.xy, a.zw) * uv + m.xform2[1u].xy;
    return texture(textures[m.tex2.x], uv);
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
    float metallic = clamp(m.pbr.x, 0.0, 1.0);
    float roughness = clamp(m.pbr.y, 0.0, 1.0);
    if (m.tex2.x != 0u) {
        vec4 mr = sampleMetallicRoughness(m);
        roughness *= mr.g;
        metallic *= mr.b;
    }
    roughness = max(roughness, 0.04);

    vec3 v = normalize(u.camPos.xyz - inWorld);
    vec3 tint;
    float units = unitSunAt(inWorld, n, tint);
    // Illuminance (lux) before the cosine; the particles' tint colours the
    // sunlight they shade.
    vec3 sunE = aeUnitTint(AE_SUN_TINT) * (u.photometry.z * min(sunAt(inWorld), units)) * tint;
    vec3 skyE = aeUnitTint(AE_SKY_TINT) * (u.photometry.w * (0.5 + 0.5 * n.y) * ao);
    vec3 skyR = aeUnitTint(AE_SKY_TINT) * (u.photometry.w / AE_PI);
    SurfaceBRDF s = SurfaceBRDF(base.rgb, mix(vec3(0.04), base.rgb, metallic), roughness, metallic);
    vec3 color = brdfLight(s, n, v, normalize(u.sun.xyz), sunE);
    color += brdfDiffuse(s, n, v) * skyE + brdfAmbient(s, n, v) * (skyR * ao);
    // glTF's emissive factor x strength is dimensionless; the engine reads
    // it as luminance in cd/m² at this scale (documented in lighting.md).
    vec3 emitted = emissive * 19660.8;
    color += emitted;
    if (lightsOn()) {
        color += lightsSurface(lightClusterFrag(gl_FragCoord), inWorld, n, v, s);
    }
    float dist = distance(inWorld, u.camPos.xyz);
    float fog = 1.0 - exp(-pow(dist * u.fog.w, 1.5));
    outColor = vec4(mix(color, u.fog.rgb, fog) * u.photometry.x, 1.0);
    // The bloom's light sources: the material's emission, after the same
    // fog the scene sees (a fogged model's emission is not a light source).
    outGlow = vec4(emitted * (1.0 - fog) * u.photometry.x, 1.0);
}
