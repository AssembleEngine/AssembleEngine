//! Coordinate and timing types. The four coordinate spaces are deliberately
//! distinct newtypes so accidental mixing is hard.
use serde::{Deserialize, Serialize};

pub const TICK_HZ: u32 = 30;
pub const TICK_SECONDS: f32 = 1.0 / TICK_HZ as f32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub struct CellPos {
    pub x: i32,
    pub y: i32,
}

impl CellPos {
    pub const fn new(x: i32, y: i32) -> Self { Self { x, y } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub struct SectorPos {
    pub x: i32,
    pub y: i32,
}

impl SectorPos {
    pub const fn new(x: i32, y: i32) -> Self { Self { x, y } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub struct ScreenPos {
    pub x: i32,
    pub y: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct WorldPos {
    pub x: f32,
    pub y: f32,
}

impl WorldPos {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub const fn new(x: f32, y: f32) -> Self { Self { x, y } }
    pub fn dist(self, o: Self) -> f32 { let (dx, dy) = (self.x - o.x, self.y - o.y); (dx * dx + dy * dy).sqrt() }
    pub fn lerp(self, o: Self, t: f32) -> Self { Self::new(self.x + (o.x - self.x) * t, self.y + (o.y - self.y) * t) }
    pub fn add(self, o: Self) -> Self { Self::new(self.x + o.x, self.y + o.y) }
    pub fn sub(self, o: Self) -> Self { Self::new(self.x - o.x, self.y - o.y) }
    pub fn scaled(self, s: f32) -> Self { Self::new(self.x * s, self.y * s) }
    pub fn length(self) -> f32 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn normalized_or(self, fallback: Self) -> Self {
        let l = self.length();
        if l > 1e-6 { self.scaled(1.0 / l) } else { fallback }
    }
}

impl std::ops::Add for WorldPos {
    type Output = WorldPos;
    fn add(self, o: WorldPos) -> WorldPos { WorldPos::new(self.x + o.x, self.y + o.y) }
}
impl std::ops::Sub for WorldPos {
    type Output = WorldPos;
    fn sub(self, o: WorldPos) -> WorldPos { WorldPos::new(self.x - o.x, self.y - o.y) }
}
impl std::ops::Mul<f32> for WorldPos {
    type Output = WorldPos;
    fn mul(self, s: f32) -> WorldPos { WorldPos::new(self.x * s, self.y * s) }
}
impl std::ops::AddAssign for WorldPos {
    fn add_assign(&mut self, o: WorldPos) { self.x += o.x; self.y += o.y; }
}

#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Rect {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

impl Rect {
    pub fn from_points(a: WorldPos, b: WorldPos) -> Self {
        Self { min_x: a.x.min(b.x), min_y: a.y.min(b.y), max_x: a.x.max(b.x), max_y: a.y.max(b.y) }
    }
    pub fn from_center(c: WorldPos, half_w: f32, half_h: f32) -> Self {
        Self { min_x: c.x - half_w, min_y: c.y - half_h, max_x: c.x + half_w, max_y: c.y + half_h }
    }
    pub fn contains(&self, p: WorldPos) -> bool {
        p.x >= self.min_x && p.x <= self.max_x && p.y >= self.min_y && p.y <= self.max_y
    }
    pub fn width(&self) -> f32 { self.max_x - self.min_x }
    pub fn height(&self) -> f32 { self.max_y - self.min_y }
    pub fn intersects(&self, o: &Rect) -> bool {
        self.min_x <= o.max_x && self.max_x >= o.min_x && self.min_y <= o.max_y && self.max_y >= o.min_y
    }
}

/// Immutable description of the map's dimensions and coordinate transforms.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldGeometry {
    pub cells_x: i32,
    pub cells_y: i32,
    pub cell_size: f32,
    pub sector_cells: i32,
}

impl WorldGeometry {
    pub fn new(cells_x: i32, cells_y: i32, cell_size: f32, sector_cells: i32) -> Self {
        assert!(cells_x > 0 && cells_y > 0 && cell_size > 0.0 && sector_cells > 0);
        Self { cells_x, cells_y, cell_size, sector_cells }
    }
    pub fn world_width(&self) -> f32 { self.cells_x as f32 * self.cell_size }
    pub fn world_height(&self) -> f32 { self.cells_y as f32 * self.cell_size }
    pub fn sectors_x(&self) -> i32 { (self.cells_x + self.sector_cells - 1) / self.sector_cells }
    pub fn sectors_y(&self) -> i32 { (self.cells_y + self.sector_cells - 1) / self.sector_cells }
    pub fn cell_index(&self, c: CellPos) -> usize { (c.y * self.cells_x + c.x) as usize }
    pub fn cell_index_xy(&self, x: i32, y: i32) -> Option<usize> {
        if x >= 0 && y >= 0 && x < self.cells_x && y < self.cells_y {
            Some((y * self.cells_x + x) as usize)
        } else { None }
    }
    pub fn cell_of(&self, x: i32, y: i32) -> Option<CellPos> {
        if x >= 0 && y >= 0 && x < self.cells_x && y < self.cells_y { Some(CellPos { x, y }) } else { None }
    }
    pub fn cell_center(&self, c: CellPos) -> WorldPos {
        WorldPos::new((c.x as f32 + 0.5) * self.cell_size, (c.y as f32 + 0.5) * self.cell_size)
    }
    pub fn world_to_cell(&self, p: WorldPos) -> CellPos {
        CellPos::new(
            (p.x / self.cell_size).floor() as i32,
            (p.y / self.cell_size).floor() as i32,
        )
    }
    pub fn clamp_cell(&self, c: CellPos) -> CellPos {
        CellPos::new(c.x.clamp(0, self.cells_x - 1), c.y.clamp(0, self.cells_y - 1))
    }
    pub fn in_bounds(&self, c: CellPos) -> bool {
        c.x >= 0 && c.y >= 0 && c.x < self.cells_x && c.y < self.cells_y
    }
    pub fn sector_of(&self, c: CellPos) -> SectorPos {
        SectorPos::new(c.x / self.sector_cells, c.y / self.sector_cells)
    }
    pub fn sector_index(&self, s: SectorPos) -> usize {
        (s.y * self.sectors_x() + s.x) as usize
    }
    pub fn sector_bounds(&self, s: SectorPos) -> (i32, i32, i32, i32) {
        let x0 = s.x * self.sector_cells;
        let y0 = s.y * self.sector_cells;
        (x0, y0, (x0 + self.sector_cells).min(self.cells_x), (y0 + self.sector_cells).min(self.cells_y))
    }
    pub fn bounds(&self) -> Rect {
        Rect { min_x: 0.0, min_y: 0.0, max_x: self.world_width(), max_y: self.world_height() }
    }
}

/// Integer-coordinate helpers used by the authoritative navigation layer.
pub const NEIGHBORS8: [(i32, i32); 8] = [
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
];
pub const NEIGHBORS4: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];
