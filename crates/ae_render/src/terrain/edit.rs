//! Terrain edits on the render side: a CPU copy of the heightmap, error
//! pyramid, materials and shadow heights that takes changed cells and produces the
//! [`TerrainUpdate`] the renderer applies (texture regions only, so a
//! small edit uploads a few kilobytes).
//!
//! The textures are built from per-cell ground heights the way
//! [`Heightmap::from_cells`] builds them at load: corners average the
//! cells around them, normalised with the map's load-time scale and offset
//! (an edit may leave [0, 1]; the textures are float), so an edited map
//! has the same heights, pyramid and shadows as one built from scratch with
//! the same scale and offset. The shadow regions reach past the edit, as
//! far away from the sun as its shadow (old or new) can fall.

use super::app::TerrainSource;
use super::heightmap::{Heightmap, PYRAMID_BASE_LEVEL, Pyramid, Rect};
use super::renderer::{TerrainInfo, TerrainUpdate};
use super::sun;
use crate::upload::Region;

pub struct TerrainEditor {
    heightmap: Heightmap,
    pyramid: Pyramid,
    materials: Option<Vec<u8>>,
    /// Shadow heights ([`super::sun`]), kept in step with the heights.
    shadow: Vec<f32>,
    info: TerrainInfo,
}

fn region<T: Copy>(mip: u32, rect: Rect, row_len: usize, data: &[T]) -> Region {
    let (x0, y0, x1, y1) = rect;
    let mut bytes = Vec::with_capacity((x1 - x0 + 1) * (y1 - y0 + 1) * size_of::<T>());
    for y in y0..=y1 {
        let row = &data[y * row_len + x0..=y * row_len + x1];
        // Safety: plain-old-data (floats and bytes) only.
        bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(row.as_ptr().cast::<u8>(), std::mem::size_of_val(row)) });
    }
    Region { mip, x: x0 as u32, y: y0 as u32, width: (x1 - x0 + 1) as u32, height: (y1 - y0 + 1) as u32, bytes }
}

impl TerrainEditor {
    /// A copy of `source` to edit (the renderer's textures hold the same).
    pub fn new(source: &TerrainSource) -> Self {
        let heightmap = Heightmap { width: source.heightmap.width, height: source.heightmap.height, data: source.heightmap.data.clone() };
        let pyramid = Pyramid { root_cells: source.pyramid.root_cells, levels: source.pyramid.levels.clone(), errors: source.pyramid.errors.clone() };
        let i = source.info;
        let shadow = sun::shadow_heights(&heightmap, i.cell_size, i.height_scale, i.height_offset, sun::direction());
        Self { heightmap, pyramid, materials: source.materials.clone(), shadow, info: source.info }
    }

    pub fn heightmap(&self) -> &Heightmap {
        &self.heightmap
    }

    pub fn pyramid(&self) -> &Pyramid {
        &self.pyramid
    }

    /// Shadow heights of the current terrain ([`super::sun::shadow_heights`]).
    pub fn shadow(&self) -> &[f32] {
        &self.shadow
    }

    pub fn materials(&self) -> Option<&[u8]> {
        self.materials.as_deref()
    }

    /// The cells in `rects` (inclusive cell rectangles `(x0, y0, x1, y1)`,
    /// clipped to the map) changed: `ground` gives any cell's world height
    /// (cells around the rectangles are read too, for the corners they
    /// share), `material` the material of a cell in them. Returns the
    /// regions to upload, a few per rectangle (`None` if every rectangle
    /// misses the map).
    pub fn edit(&mut self, version: u32, rects: &[(i32, i32, i32, i32)], ground: impl Fn(usize, usize) -> f32, material: impl Fn(usize, usize) -> u8) -> Option<TerrainUpdate> {
        let (cx, cy) = self.heightmap.cells();
        let clip = |v: i32, n: usize| v.clamp(0, n as i32 - 1) as usize;
        let cells: Vec<Rect> = rects
            .iter()
            .filter(|&&(x0, y0, x1, y1)| x1 >= 0 && y1 >= 0 && x0 < cx as i32 && y0 < cy as i32 && x1 >= x0 && y1 >= y0)
            .map(|&(x0, y0, x1, y1)| (clip(x0, cx), clip(y0, cy), clip(x1, cx), clip(y1, cy)))
            .collect();
        if cells.is_empty() {
            return None;
        }
        // A cell moves its four corners.
        let corners: Vec<Rect> = cells.iter().map(|&(x0, y0, x1, y1)| (x0, y0, x1 + 1, y1 + 1)).collect();
        let (scale, offset) = (self.info.height_scale, self.info.height_offset);
        let w = self.heightmap.width;
        for &(x0, y0, x1, y1) in &corners {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    self.heightmap.data[y * w + x] = (Heightmap::corner((cx, cy), &ground, x, y) - offset) / scale;
                }
            }
        }
        let change = self.pyramid.update(&self.heightmap, scale, offset, &corners);
        let shadowed = sun::update(&mut self.shadow, &self.heightmap, self.info.cell_size, scale, offset, sun::direction(), &corners);
        let mut update = TerrainUpdate { version, ..TerrainUpdate::default() };
        update.heights = corners.iter().map(|&r| region(0, r, w, &self.heightmap.data)).collect();
        update.shadows = shadowed.iter().map(|&r| region(0, r, w, &self.shadow)).collect();
        for (i, rects) in change.levels.iter().enumerate() {
            let side = self.pyramid.root_cells >> (PYRAMID_BASE_LEVEL as usize + i);
            update.pyramid.extend(rects.iter().map(|&r| region(i as u32, r, side, &self.pyramid.levels[i])));
        }
        for (i, rects) in change.errors.iter().enumerate() {
            let side = self.pyramid.root_cells >> (i + 1);
            update.errors.extend(rects.iter().map(|&r| region(i as u32, r, side, &self.pyramid.errors[i])));
        }
        if let Some(m) = self.materials.as_mut() {
            for &(x0, y0, x1, y1) in &cells {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        m[y * cx + x] = material(x, y);
                    }
                }
            }
            update.materials = cells.iter().map(|&r| region(0, r, cx, m)).collect();
        }
        Some(update)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::heightmap::build_pyramid;

    /// Cell heights of a small test map; each bump sets a rectangle (later
    /// ones win).
    fn ground(x: usize, y: usize, bumps: &[(Rect, f32)]) -> f32 {
        let base = ((x * 37 + y * 11) % 23) as f32 * 1.5 + y as f32 * 0.25;
        bumps.iter().rev().find(|((x0, y0, x1, y1), _)| (*x0..=*x1).contains(&x) && (*y0..=*y1).contains(&y)).map_or(base, |b| b.1)
    }

    fn source(cells: (usize, usize), bumps: &[(Rect, f32)], scale_offset: Option<(f32, f32)>) -> TerrainSource {
        let (hm, scale, offset) = Heightmap::from_cells(cells, |x, y| ground(x, y, bumps), scale_offset);
        let pyramid = build_pyramid(&hm, scale, offset);
        let info = TerrainInfo::new(&hm, &pyramid, 8.0, scale, offset, None);
        let materials = Some((0..cells.0 * cells.1).map(|i| (i % 5) as u8).collect());
        TerrainSource { heightmap: hm, pyramid, info, materials }
    }

    /// An edit leaves the editor's copy exactly as a map built from scratch
    /// from the edited cells (with the load-time scale and offset), and the
    /// regions it returns hold those texels.
    #[test]
    fn edits_equal_a_map_built_from_the_edited_cells() {
        let cells = (60, 44);
        let start = source(cells, &[], None);
        let (scale, offset) = (start.info.height_scale, start.info.height_offset);
        let mut editor = TerrainEditor::new(&start);
        let mut bumps = Vec::new();
        for (rect, h) in [((10, 10, 14, 12), 90.0), ((0, 0, 2, 2), -20.0), ((55, 40, 59, 43), 3.0), ((12, 8, 20, 11), 40.0)] {
            bumps.push((rect, h));
            let update = editor.edit(7, &[(rect.0 as i32, rect.1 as i32, rect.2 as i32, rect.3 as i32)], |x, y| ground(x, y, &bumps), |_, _| 9).unwrap();
            let fresh = source(cells, &bumps, Some((scale, offset)));
            let bits = |d: &[f32]| d.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&editor.heightmap().data), bits(&fresh.heightmap.data), "heights after {rect:?}");
            assert_eq!(editor.pyramid().levels, fresh.pyramid.levels);
            assert_eq!(editor.pyramid().errors, fresh.pyramid.errors);
            let i = fresh.info;
            assert_eq!(editor.shadow(), sun::shadow_heights(&fresh.heightmap, i.cell_size, i.height_scale, i.height_offset, sun::direction()));
            // The height region is the cells' corners.
            let r = &update.heights[0];
            assert_eq!((r.x, r.y, r.width, r.height), (rect.0 as u32, rect.1 as u32, (rect.2 - rect.0 + 2) as u32, (rect.3 - rect.1 + 2) as u32));
            assert_eq!(&r.bytes[..4], &editor.heightmap().data[rect.1 * 61 + rect.0].to_le_bytes());
            let m = &update.materials[0];
            assert!(m.bytes.iter().all(|&b| b == 9) && m.bytes.len() == (rect.2 - rect.0 + 1) * (rect.3 - rect.1 + 1));
            assert_eq!(update.pyramid.len(), start.pyramid.levels.len(), "every level has a block over the edit");
        }
        // Several rectangles in one edit: the same as one at a time.
        let more = [((30, 30, 31, 31), 70.0), ((40, 5, 44, 6), -5.0), ((2, 38, 3, 43), 12.0)];
        bumps.extend(more);
        let rects: Vec<_> = more.iter().map(|&((x0, y0, x1, y1), _)| (x0 as i32, y0 as i32, x1 as i32, y1 as i32)).collect();
        let update = editor.edit(8, &rects, |x, y| ground(x, y, &bumps), |_, _| 9).unwrap();
        let fresh = source(cells, &bumps, Some((scale, offset)));
        assert_eq!(editor.heightmap().data, fresh.heightmap.data);
        assert_eq!(editor.pyramid().levels, fresh.pyramid.levels);
        assert_eq!(editor.pyramid().errors, fresh.pyramid.errors);
        assert_eq!((update.heights.len(), update.materials.len()), (3, 3));
        // Clipped to the map; outside it, nothing.
        assert!(editor.edit(1, &[(-5, -5, -1, 3)], |x, y| ground(x, y, &bumps), |_, _| 0).is_none());
        let u = editor.edit(1, &[(-5, -5, -1, 3), (58, -3, 70, 1)], |x, y| ground(x, y, &bumps), |_, _| 0).unwrap();
        assert_eq!((u.heights[0].x, u.heights[0].y, u.heights[0].width, u.heights[0].height), (58, 0, 3, 3));
    }
}
