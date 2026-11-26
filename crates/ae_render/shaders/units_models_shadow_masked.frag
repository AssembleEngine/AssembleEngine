#version 460
// The units' shadow map for alpha-masked model materials: the camera's
// alpha test (units_models.frag.glsl), so what is masked away casts no
// shadow.
#define MODELS
#include "units_common.glsl"
#include "units_models.glsl"
#include "shadow_write.glsl"

layout(location = 0) in vec4 inUv;
layout(location = 1) in float inAlpha;
layout(location = 2) flat in uint inMaterial;

void main() {
    Material m = materials[inMaterial];
    float alpha = m.baseColor.a * inAlpha;
    if (m.tex.x != 0u) {
        vec2 uv = m.uvSet.x == 0u ? inUv.xy : inUv.zw;
        vec4 a = m.xform[0];
        alpha *= texture(
            sampler2D(
                ae_images[nonuniformEXT(m.tex.x)],
                ae_samplers[nonuniformEXT(m.texSampler.x)]
            ),
            mat2(a.xy, a.zw) * uv + m.xform[1].xy
        ).a;
    }
    if (alpha < m.emissive.w) {
        discard;
    }
    storeShadowDepth();
}
