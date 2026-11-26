struct ShadowCaster {
    vec4 positionSize;
    vec4 rotation;
    vec4 bounds;
    uvec4 state; // class + 1 when eligible, and whether it can change without moving
};

layout(std430, set = AE_BINDLESS_SET, binding = AE_BUFFERS) buffer ShadowCache {
    uint valid;
    uint casterCount;
    uint previousRows;
    uint reuse;
    vec4 origin[UNIT_CASCADES];
    vec4 scale[UNIT_CASCADES];
    ShadowCaster casters[];
} ae_shadow_caches[];
#define shadowCache ae_shadow_caches[ae.words.y]
