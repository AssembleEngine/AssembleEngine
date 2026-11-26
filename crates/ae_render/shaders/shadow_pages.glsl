#ifndef AE_SHADOW_PAGES_GLSL
#define AE_SHADOW_PAGES_GLSL

// Mirrors the shadow image spec in ae_render_abi::module.
#define UNIT_CASCADES 3
const int SHADOW_PAGE_SIZE = 64;
const int SHADOW_LEVEL_PAGES = 32;
const int SHADOW_LEVEL_SIZE = SHADOW_LEVEL_PAGES * SHADOW_PAGE_SIZE;
const float SHADOW_EDGE_START = 0.90;
const float SHADOW_EDGE_END = 1.0 - 5.0 / float(SHADOW_LEVEL_SIZE);
const float SHADOW_RECEIVER_PAD_TEXELS = 4.0;

// Terrain shades opaque shadows at its heightfield within this distance
// of the rasterized surface. The page request pass uses the same bound.
float shadowTerrainHeightMargin(float outerTexel) {
    return max(1.0, 8.0 * outerTexel);
}
const uint SHADOW_POOL_COLUMNS = 64u;
const uint SHADOW_POOL_PAGES = SHADOW_POOL_COLUMNS * uint(SHADOW_LEVEL_PAGES);
const int SHADOW_PAGE_ROWS = UNIT_CASCADES * SHADOW_LEVEL_PAGES;
const int SHADOW_CLEAR_ROW = SHADOW_LEVEL_SIZE + SHADOW_PAGE_ROWS;
const uint SHADOW_REQUEST_CASTER = 1u;
const uint SHADOW_REQUEST_DIRTY = 2u;
const uint SHADOW_REQUEST_VIEW = 4u;
const uint SHADOW_REQUEST_INNER = 8u;
const uint SHADOW_PAGE_DIRTY = 0x80000000u;
const uint SHADOW_SLOT_MASK = 0x7fffffffu;
const uint SHADOW_PAGE_EMPTY = 0u;
// A requested page with no backing. Receivers retry at a coarser level.
const uint SHADOW_PAGE_UNBACKED = 0xffffffffu;

ivec2 shadowTableAddress(ivec2 page, int level) {
    return ivec2(page.x, SHADOW_LEVEL_SIZE + (UNIT_CASCADES - 1 - level) * SHADOW_LEVEL_PAGES + page.y);
}

ivec2 shadowRequestAddress(ivec2 tableAddress) {
    return tableAddress + ivec2(SHADOW_LEVEL_PAGES, 0);
}

ivec2 shadowPhysicalPage(uint slot) {
    return ivec2(slot % SHADOW_POOL_COLUMNS, slot / SHADOW_POOL_COLUMNS) * SHADOW_PAGE_SIZE;
}

ivec2 shadowPhysicalTexel(uint entry, ivec2 texel) {
    return shadowPhysicalPage(entry - 1u) + (texel % SHADOW_PAGE_SIZE);
}

#endif
