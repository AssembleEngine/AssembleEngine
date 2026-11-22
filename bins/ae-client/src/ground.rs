//! Terrain edits from the sim to the renderer: the nav service publishes
//! the terrain after each landed ground change ([`GroundFeed`]) with one
//! rectangle around everything that changed, which for edits scattered
//! over the map is most of it. A thread here compares the new terrain with
//! the previous one inside that rectangle, tile by tile, and turns the
//! changed tiles into texture regions with a [`TerrainEditor`] (the error
//! pyramid update reads about the whole map, so it stays off the render
//! and sim threads), then hands them to the render thread
//! ([`TerrainApp::terrain_edits`](ae_render::terrain::app::TerrainApp::terrain_edits)).

use ae_nav::terrain::{SharedTerrain, Terrain};
use ae_render::terrain::app::{TerrainEdits, TerrainSource};
use ae_render::terrain::edit::TerrainEditor;
use ae_render::terrain::renderer::TerrainUpdate;
use ae_services::nav::{GroundFeed, GroundUpdate};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// World height of a cell's ground (the bed under water): nav stores
/// decimetres, render takes 1 world unit = 1 m, times `relief`.
pub fn cell_ground(terrain: &Terrain, relief: f32) -> impl Fn(usize, usize) -> f32 + '_ {
    let cx = terrain.geom.cells_x as usize;
    move |x, y| terrain.ground_height(y * cx + x) as f32 * 0.1 * relief
}

/// Side of the tiles changes are found in (cells).
pub const TILE: i32 = 32;

/// The cells of `cells` (inclusive) whose ground height or material
/// differs between `old` and `new`: per `TILE`-square tile with any, the
/// rectangle around its changed cells, in row-major tile order.
pub fn changed_rects(old: &Terrain, new: &Terrain, cells: (i32, i32, i32, i32)) -> Vec<(i32, i32, i32, i32)> {
    let (cx, cy) = (new.geom.cells_x, new.geom.cells_y);
    let (x0, y0) = (cells.0.max(0), cells.1.max(0));
    let (x1, y1) = (cells.2.min(cx - 1), cells.3.min(cy - 1));
    let mut rects = Vec::new();
    if x1 < x0 || y1 < y0 {
        return rects;
    }
    for ty in (y0 / TILE)..=(y1 / TILE) {
        for tx in (x0 / TILE)..=(x1 / TILE) {
            let mut r = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            for y in (ty * TILE).max(y0)..=((ty + 1) * TILE - 1).min(y1) {
                for x in (tx * TILE).max(x0)..=((tx + 1) * TILE - 1).min(x1) {
                    let i = (y * cx + x) as usize;
                    if old.ground_height(i) != new.ground_height(i) || old.material[i] != new.material[i] {
                        r = (r.0.min(x), r.1.min(y), r.2.max(x), r.3.max(y));
                    }
                }
            }
            if r.0 <= r.2 {
                rects.push(r);
            }
        }
    }
    rects
}

/// The texture regions for `update`, against the terrain the textures
/// show (`shown`, then replaced by the update's).
pub fn edit(
    editor: &mut TerrainEditor,
    shown: &mut SharedTerrain,
    update: &GroundUpdate,
    relief: f32,
) -> Option<(TerrainUpdate, usize)> {
    let t = &update.terrain;
    let rects = changed_rects(shown, t, update.cells);
    let cx = t.geom.cells_x as usize;
    let result = editor.edit(update.version, &rects, cell_ground(t, relief), |x, y| {
        t.material[y * cx + x]
    });
    *shown = t.clone();
    result.map(|u| (u, rects.len()))
}

pub struct GroundThread {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl GroundThread {
    /// `shown`: the terrain `source` was built from.
    pub fn start(
        feed: Arc<GroundFeed>,
        source: Arc<TerrainSource>,
        mut shown: SharedTerrain,
        relief: f32,
        edits: Arc<Mutex<TerrainEdits>>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let handle = std::thread::Builder::new()
            .name("terrain edits".into())
            .spawn(move || {
                let mut editor = TerrainEditor::new(&source);
                drop(source);
                while !stop2.load(Ordering::Relaxed) {
                    let Some(update) = feed.take() else {
                        std::thread::sleep(Duration::from_millis(15));
                        continue;
                    };
                    let started = Instant::now();
                    if let Some((u, tiles)) = edit(&mut editor, &mut shown, &update, relief) {
                        eprintln!("terrain version {}: {tiles} changed tiles in cells {:?}, {} bytes to upload, made in {:.1} ms", u.version, update.cells, u.bytes(), started.elapsed().as_secs_f32() * 1e3);
                        edits.lock().unwrap_or_else(|p| p.into_inner()).pending.push(u);
                    }
                }
            })
            .expect("spawn the terrain edit thread");
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for GroundThread {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Changed cells come back as one tight rectangle per tile, and only
    /// inside the rectangle asked about; cost-only changes do not count.
    #[test]
    fn changed_cells_are_found_tile_by_tile() {
        let old = ae_nav::scenario::water_crossing_terrain(256);
        let mut new = old.clone();
        let cx = 256usize;
        let set_ground = |t: &mut Terrain, x: usize, y: usize| {
            let i = y * cx + x;
            t.height[i] += 30;
        };
        // Two cells in one tile, one across the tile boundary from them,
        // a material change, a cost change (ignored) and a change outside
        // the asked rectangle.
        set_ground(&mut new, 40, 40);
        set_ground(&mut new, 45, 50);
        set_ground(&mut new, 64, 50);
        new.material[70 * cx + 100] ^= 1;
        new.move_cost[10 * cx + 10] = 99;
        set_ground(&mut new, 200, 200);
        let rects = changed_rects(&old, &new, (0, 0, 150, 150));
        assert_eq!(rects, [(40, 40, 45, 50), (64, 50, 64, 50), (100, 70, 100, 70)]);
        assert_eq!(changed_rects(&old, &new, (0, 0, 255, 255)).len(), 4);
        assert!(changed_rects(&old, &old, (0, 0, 255, 255)).is_empty());
        assert!(changed_rects(&old, &new, (300, 300, 400, 400)).is_empty());
    }
}
