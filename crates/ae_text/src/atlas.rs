//! MSDF tiles shared by every display size of a font face and instance.

use crate::{Color, Layout};
use anyhow::{Context as _, Result, ensure};
use glyphux::{
    Bitmap, Contour, DEFAULT_ANGLE_THRESHOLD, Edge, FillRule, GeneratorConfig, Projection, Shape, Transform, Vec2,
};
use parley::layout::PositionedLayoutItem;
use skrifa::{
    FontRef, GlyphId, MetadataProvider,
    instance::{LocationRef, NormalizedCoord, Size},
    outline::{DrawSettings, OutlinePen},
    raw::TableProvider,
};
use std::{collections::HashMap, ops::Range};

const TEXELS_PER_EM: f64 = 32.0;
const RANGE: f64 = 4.0;
const MARGIN: f64 = RANGE / 2.0 + 1.0;
// Increase stroke width by about 1/24 em when there is no bold face.
const BOLD_EXPANSION: f64 = 1.0 / 48.0;

#[derive(Clone, Copy, Debug)]
pub struct Shadow {
    pub offset: [f32; 2],
    pub color: Color,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Quad {
    /// Left, top, right, and bottom in local text pixels.
    pub bounds: [f32; 4],
    /// Matching coordinates in atlas texels, with the origin at top left.
    pub uv: [f32; 4],
    pub color: Color,
}

#[derive(Clone, Copy)]
struct Tile {
    bounds: [f32; 4],
    uv: [f32; 4],
}

#[derive(Eq, Hash, PartialEq)]
struct FontInstance {
    data_id: u64,
    face: u32,
    coords: Vec<NormalizedCoord>,
    skew: Option<u32>,
    embolden: bool,
}

struct Packing {
    limit: u32,
    x: u32,
    y: u32,
    row_height: u32,
    rgba: Vec<u8>,
}

impl Packing {
    fn add(&mut self, image: &Bitmap<3>) -> Result<[f32; 4]> {
        let (width, height) = (image.width(), image.height());
        ensure!(
            width <= self.limit && height <= self.limit,
            "MSDF tile exceeds atlas dimensions"
        );
        let (x, y, row_height) = if self.x + width > self.limit {
            (0, self.y + self.row_height, 0)
        } else {
            (self.x, self.y, self.row_height)
        };
        let right = x + width;
        let bottom = y + height;
        ensure!(bottom <= self.limit, "MSDF atlas is full");
        self.rgba
            .resize(self.rgba.len().max((self.limit * bottom * 4) as usize), 0);
        // The outer texel centers are beyond the field's range. Keep all
        // their channels zero so filtering can't draw dots at tile seams.
        let inner_width = width as usize - 2;
        for row in 1..height - 1 {
            let at = (((y + row) * self.limit + x + 1) * 4) as usize;
            let pixels = &mut self.rgba[at..at + inner_width * 4];
            for (target, source) in pixels
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(&image.row(row).as_chunks::<3>().0[1..1 + inner_width])
            {
                *target = [
                    glyphux::float_to_byte(source[0]),
                    glyphux::float_to_byte(source[1]),
                    glyphux::float_to_byte(source[2]),
                    255,
                ];
            }
        }
        (self.x, self.y, self.row_height) = (right, y, row_height.max(height));
        Ok([x, y, right, bottom].map(|coordinate| coordinate as f32))
    }
}

#[derive(Default)]
struct GlyphOutline {
    shape: Shape,
    contour: Contour,
    start: Vec2,
    current: Vec2,
    scale: f64,
    skew: f64,
}

impl GlyphOutline {
    fn point(&self, x: f32, y: f32) -> Vec2 {
        let y = f64::from(y) * self.scale;
        Vec2::new(f64::from(x) * self.scale + self.skew * y, y)
    }

    fn edge(&mut self, edge: Edge, to: Vec2) {
        self.contour.add_edge(edge);
        self.current = to;
    }
}

impl OutlinePen for GlyphOutline {
    fn move_to(&mut self, x: f32, y: f32) {
        self.close();
        self.start = self.point(x, y);
        self.current = self.start;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let to = self.point(x, y);
        if to != self.current {
            self.edge(Edge::line(self.current, to), to);
        }
    }

    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let control = self.point(cx, cy);
        let to = self.point(x, y);
        if control != self.current || to != self.current {
            self.edge(Edge::quadratic(self.current, control, to), to);
        }
    }

    fn curve_to(&mut self, cx: f32, cy: f32, dx: f32, dy: f32, x: f32, y: f32) {
        let c = self.point(cx, cy);
        let d = self.point(dx, dy);
        let to = self.point(x, y);
        if c != self.current || d != self.current || to != self.current {
            self.edge(Edge::cubic(self.current, c, d, to), to);
        }
    }

    fn close(&mut self) {
        if !self.contour.edges.is_empty() {
            if self.current != self.start {
                self.edge(Edge::line(self.current, self.start), self.start);
            }
            self.shape.add_contour(std::mem::take(&mut self.contour));
        }
    }
}

fn tile(mut shape: Shape, embolden: bool, packing: &mut Packing) -> Result<Option<Tile>> {
    let bbox = shape.bounds();
    if !(bbox.left < bbox.right && bbox.bottom < bbox.top) {
        return Ok(None);
    }
    let expansion = if embolden { BOLD_EXPANSION } else { 0.0 };
    let margin = MARGIN + (expansion * TEXELS_PER_EM).ceil();
    let left = (bbox.left * TEXELS_PER_EM).floor() - margin;
    let bottom = (bbox.bottom * TEXELS_PER_EM).floor() - margin;
    let right = (bbox.right * TEXELS_PER_EM).ceil() + margin;
    let top = (bbox.top * TEXELS_PER_EM).ceil() + margin;
    let width = right - left;
    let height = top - bottom;
    ensure!(
        width <= f64::from(packing.limit) && height <= f64::from(packing.limit),
        "MSDF glyph exceeds atlas dimensions"
    );

    let projection = Projection::uniform(TEXELS_PER_EM, Vec2::new(-left, -bottom) / TEXELS_PER_EM);
    let half_range = RANGE / (2.0 * TEXELS_PER_EM);
    let field_range = glyphux::Range::new(-half_range - expansion, half_range - expansion);
    let transform = Transform::new(projection, field_range);
    shape.normalize();
    glyphux::edge_coloring_simple(&mut shape, DEFAULT_ANGLE_THRESHOLD, 0);
    let mut image = Bitmap::new(width as u32, height as u32);
    let config = GeneratorConfig::default().with_sign_correction(Some(FillRule::NonZero));
    glyphux::generate_msdf(&mut image, &shape, &transform, &config);
    Ok(Some(Tile {
        bounds: [left, -top, right, -bottom].map(|value| (value / TEXELS_PER_EM) as f32),
        uv: packing.add(&image)?,
    }))
}

/// Packs fields on demand. Appending layouts preserves their quad ranges
/// and atlas texel coordinates. Unchanged text can keep its GPU batch.
pub struct Atlas {
    packing: Packing,
    glyphs: HashMap<FontInstance, HashMap<u32, Option<Tile>>>,
    solid_uv: [f32; 4],
    quads: Vec<Quad>,
}

impl Atlas {
    pub fn new(max_size: u32) -> Result<Self> {
        ensure!((16..=4096).contains(&max_size), "invalid MSDF atlas limit {max_size}");
        let mut packing = Packing {
            limit: max_size,
            x: 0,
            y: 0,
            row_height: 0,
            rgba: Vec::new(),
        };
        let [left, top, right, bottom] = packing.add(&Bitmap::from_data(3, 3, vec![1.0; 27]))?;
        let (x, y) = ((left + right) * 0.5, (top + bottom) * 0.5);
        Ok(Self {
            packing,
            glyphs: HashMap::new(),
            solid_uv: [x, y, x, y],
            quads: Vec::new(),
        })
    }

    pub fn size(&self) -> [u32; 2] {
        [self.packing.limit, self.packing.y + self.packing.row_height]
    }

    pub fn rgba(&self) -> &[u8] {
        &self.packing.rgba
    }

    pub fn quads(&self) -> &[Quad] {
        &self.quads
    }

    /// Append glyphs and decorations as one draw range, with optional shadows
    /// drawn behind them. A failed append leaves previously completed ranges usable.
    pub fn append(&mut self, layout: &Layout, shadow: Option<Shadow>) -> Result<Range<u32>> {
        let start = self.quads.len();
        let foreground = self.prepare(layout)?;
        let end = start + foreground.len() * if shadow.is_some() { 2 } else { 1 };
        let range = u32::try_from(start)?..u32::try_from(end)?;
        if let Some(shadow) = shadow {
            self.quads.extend(foreground.iter().map(|quad| {
                let [left, top, right, bottom] = quad.bounds;
                let [x, y] = shadow.offset;
                Quad {
                    bounds: [left + x, top + y, right + x, bottom + y],
                    color: shadow.color,
                    ..*quad
                }
            }));
        }
        self.quads.extend(foreground);
        Ok(range)
    }

    fn prepare(&mut self, layout: &Layout) -> Result<Vec<Quad>> {
        let mut quads = Vec::new();
        for item in layout.lines().flat_map(|line| line.items()) {
            let PositionedLayoutItem::GlyphRun(glyph_run) = item else {
                continue;
            };
            let run = glyph_run.run();
            let font = run.font();
            let coords: Vec<_> = run
                .normalized_coords()
                .iter()
                .map(|coord| NormalizedCoord::from_bits(coord.to_bits()))
                .collect();
            let synthesis = run.synthesis();
            let key = FontInstance {
                data_id: font.data.id(),
                face: font.index,
                coords: coords.clone(),
                skew: synthesis.skew().map(f32::to_bits),
                embolden: synthesis.embolden(),
            };
            let cache = self.glyphs.entry(key).or_default();
            let face = FontRef::from_index(font.data.as_ref(), font.index).context("invalid font face for MSDF")?;
            let outlines = face.outline_glyphs();
            let units = face.head().context("font head table")?.units_per_em();
            ensure!(units != 0, "the font has zero units per em");
            let scale = 1.0 / f64::from(units);
            let skew = f64::from(synthesis.skew().unwrap_or(0.0)).to_radians().tan();
            let size = run.font_size();
            let style = glyph_run.style();
            for glyph in glyph_run.positioned_glyphs() {
                let sprite = match cache.entry(glyph.id) {
                    std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let sprite = if let Some(outline) = outlines.get(GlyphId::new(glyph.id)) {
                            let mut pen = GlyphOutline {
                                scale,
                                skew,
                                ..Default::default()
                            };
                            outline.draw(
                                DrawSettings::unhinted(Size::unscaled(), LocationRef::new(&coords)),
                                &mut pen,
                            )?;
                            pen.close();
                            tile(pen.shape, synthesis.embolden(), &mut self.packing)?
                        } else {
                            None
                        };
                        *entry.insert(sprite)
                    }
                };
                if let Some(sprite) = sprite {
                    let [left, top, right, bottom] = sprite.bounds;
                    quads.push(Quad {
                        bounds: [
                            glyph.x + left * size,
                            glyph.y + top * size,
                            glyph.x + right * size,
                            glyph.y + bottom * size,
                        ],
                        uv: sprite.uv,
                        color: style.brush,
                    });
                }
            }
            let metrics = run.font_metrics();
            for (decoration, offset, thickness) in [
                (&style.underline, metrics.underline_offset, metrics.underline_size),
                (
                    &style.strikethrough,
                    metrics.strikethrough_offset,
                    metrics.strikethrough_size,
                ),
            ] {
                if let Some(decoration) = decoration {
                    let x = glyph_run.offset();
                    let y = glyph_run.baseline() - decoration.offset.unwrap_or(offset);
                    let thickness = decoration.size.unwrap_or(thickness).max(1.0);
                    quads.push(Quad {
                        bounds: [x, y, x + glyph_run.advance(), y + thickness],
                        uv: self.solid_uv,
                        color: decoration.brush,
                    });
                }
            }
        }
        Ok(quads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faux_bold_expands_distance_field_coverage() -> Result<()> {
        let mut packing = Atlas::new(128)?.packing;
        let shape = Shape::from_description("{ 0,0; 0,1; 1,1; 1,0; # }")?;
        let regular = tile(shape.clone(), false, &mut packing)?.context("regular tile")?;
        let bold = tile(shape, true, &mut packing)?.context("bold tile")?;
        let coverage = |tile: Tile| {
            let [left, top, right, bottom] = tile.uv.map(|v| v as u32);
            (top..bottom)
                .flat_map(|y| {
                    let start = ((y * packing.limit + left) * 4) as usize;
                    let end = ((y * packing.limit + right) * 4) as usize;
                    packing.rgba[start..end].as_chunks::<4>().0
                })
                .filter(|pixel| pixel[..3].iter().filter(|value| **value >= 128).count() >= 2)
                .count()
        };
        assert!(coverage(bold) > coverage(regular));
        Ok(())
    }

    #[test]
    fn adjacent_tiles_keep_the_bilinear_footprint_clear_of_other_edge_colors() -> Result<()> {
        let mut atlas = Atlas::new(16)?.packing;
        // Both colors represent the exterior, but a filter mixing them
        // at the seam would produce a median of 0.5.
        let left = Bitmap::from_data(6, 5, [1.0, 0.0, 0.0].repeat(30));
        let right = Bitmap::from_data(6, 5, [0.0, 1.0, 0.0].repeat(30));
        let left = atlas.add(&left)?.map(|value| value as u32);
        let right = atlas.add(&right)?.map(|value| value as u32);
        assert_eq!(left[2], right[0]);

        let texel = |x, y| {
            let at = ((y * atlas.limit + x) * 4) as usize;
            &atlas.rgba[at..at + 3]
        };
        for [left, top, right, bottom] in [left, right] {
            for x in left..right {
                assert_eq!(texel(x, top), &[0; 3]);
                assert_eq!(texel(x, bottom - 1), &[0; 3]);
            }
            for y in top..bottom {
                assert_eq!(texel(left, y), &[0; 3]);
                assert_eq!(texel(right - 1, y), &[0; 3]);
            }
        }
        assert_eq!(texel(left[0] + 1, left[1] + 1), &[255, 0, 0]);
        assert_eq!(texel(right[0] + 1, right[1] + 1), &[0, 255, 0]);
        Ok(())
    }
}
