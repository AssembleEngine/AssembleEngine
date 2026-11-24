//! M16.2's coast starts reach the host: `NavPlugin::new(Map::Coast, cells)`
//! carries the generated points, which `map_starts` emits and the skirmish
//! game places its bases on. They are on dry, open cells; the other maps
//! keep their empty start list.

use ae_nav::math::WorldPos;
use ae_nav::scenario::{coast_starts, coast_terrain};
use ae_services::nav::{Map, NavPlugin};

#[test]
fn the_coast_plugin_publishes_starts_on_dry_land() {
    for cells in [256, 768] {
        let plugin = NavPlugin::new(Map::Coast, cells);
        assert_eq!(
            plugin.starts,
            coast_starts(cells),
            "the plugin's starts are the map's ({cells} cells)"
        );
        assert_eq!(plugin.starts.len(), 2);
        let terrain = coast_terrain(cells);
        let side = terrain.geom.world_width();
        for (k, start) in plugin.starts.iter().enumerate() {
            assert!(start[0] >= 0.0 && start[1] >= 0.0 && start[0] < side && start[1] < side);
            let c = terrain.geom.world_to_cell(WorldPos::new(start[0], start[1]));
            let i = terrain.geom.cell_index(c);
            assert_eq!(terrain.depth(i), 0, "start {k} at {start:?} is under water");
            assert_eq!(terrain.blocked[i], 0, "start {k} at {start:?} is blocked");
        }
    }
}

#[test]
fn the_other_maps_keep_their_empty_starts() {
    assert!(NavPlugin::new(Map::WaterCrossing, 768).starts.is_empty());
    assert!(NavPlugin::new(Map::LandCrossing, 768).starts.is_empty());
}
