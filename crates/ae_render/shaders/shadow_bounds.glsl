// Conservative page rectangle for a sphere in light space.
bool shadowPageBounds(vec4 bounds, int level, out ivec2 first, out ivec2 last) {
    vec3 clip = (bounds.xyz - u.unitShadowOrigin[level].xyz) * u.unitShadowScale[level].xyz;
    vec3 reach = bounds.w * u.unitShadowScale[level].xyz;
    if (clip.z + reach.z < 0.0 || clip.z - reach.z > 1.0) return false;
    // Half a texel for raster sample positions and floating point roundoff.
    vec2 uv = clip.xy * 0.5 + 0.5;
    vec2 extent = reach.xy * 0.5 + 0.5 / float(SHADOW_LEVEL_SIZE);
    vec2 low = (uv - extent) * float(SHADOW_LEVEL_PAGES);
    vec2 high = (uv + extent) * float(SHADOW_LEVEL_PAGES);
    if (any(lessThan(high, vec2(0))) || any(greaterThanEqual(low, vec2(SHADOW_LEVEL_PAGES)))) return false;
    first = ivec2(floor(max(low, vec2(0))));
    last = ivec2(floor(min(high, vec2(SHADOW_LEVEL_PAGES - 1))));
    return true;
}
