//! Engine text layout. Fonts and scratch memory are shared across labels.
//! Layouts are owned and can be retained, rewrapped, measured, or rendered.

pub use parley;

use parley::fontique::{Collection, CollectionOptions, SourceCache};
use parley::{FontContext, LayoutContext};

mod atlas;

pub use atlas::{Atlas, Quad, Shadow};

/// RGBA in the draw target's working color space. Alpha is linear.
pub type Color = [f32; 4];
pub type Layout = parley::Layout<Color>;

pub struct TextSystem {
    pub fonts: FontContext,
    scratch: LayoutContext<Color>,
}

impl TextSystem {
    /// Enable system discovery if it was compiled in. Pass false for a
    /// collection populated exclusively with the application's bundled fonts.
    pub fn new(system_fonts: bool) -> Self {
        Self {
            fonts: FontContext {
                collection: Collection::new(CollectionOptions {
                    system_fonts,
                    ..Default::default()
                }),
                source_cache: SourceCache::default(),
            },
            scratch: LayoutContext::new(),
        }
    }

    /// Range indices are UTF-8 byte offsets. Set `quantize` for pixel-aligned
    /// screen text, and leave it off for text on a projected plane.
    pub fn builder<'a>(&'a mut self, text: &'a str, quantize: bool) -> parley::RangedBuilder<'a, Color> {
        let mut builder = self.scratch.ranged_builder(&mut self.fonts, text, 1.0, quantize);
        builder.push_default(parley::style::StyleProperty::Brush([1.0; 4]));
        builder
    }
}
