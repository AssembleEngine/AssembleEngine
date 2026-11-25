//! Headless MSDF text benchmark and Vulkan readback check.

use ae_framegraph::{FrameGraph, ImageDesc, LoadOp, PassKind, ResourceClass, ResourceId};
use ae_render::device::{GpuDevice, GpuInstance};
use ae_render::executor::{Binding, Executor, ExecutorConfig, ImageBinding, PassContext, Recorder};
use ae_render::memory::OwnedImage;
use ae_render::text::ae_text::parley::style::StyleProperty;
use ae_render::text::ae_text::parley::{
    FontStyle, FontWeight, GenericFamily, PositionedLayoutItem,
    fontique::{Blob, Collection, FamilyId},
};
use ae_render::text::ae_text::{Atlas, Layout, TextSystem};
use ae_render::text::{TextBatch, screen_to_clip};
use ae_render::upload;
use anyhow::{Context as _, Result, ensure};
use ash::vk;
use glam::{Mat4, Vec3};
use std::time::Instant;
use std::{collections::BTreeSet, ffi::OsStr, sync::Arc};

const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const ATLAS_SIZE: u32 = 512;
const WARMUP: usize = 48;
const SAMPLES: usize = 180;
const PREPARE_SAMPLES: usize = 9;
const INSET: f32 = 24.0;
const PARAGRAPH: &str = "Survey crews mapped the ridge before the light reached the valley.     A narrow bridge connects the two roads, and the eastern trail bends around the lake.     The watch changes at 06:30. Café orders, spare tools, and radio batteries go in the blue crate.     Please copy route changes onto the map so the next crew can read them at a glance. ";

#[derive(Clone, Copy)]
struct Scene {
    name: &'static str,
    extent: vk::Extent2D,
    pixels: f32,
    tilted: bool,
    mixed: bool,
}

impl Scene {
    fn local_size(self) -> [f32; 2] {
        [
            self.extent.width as f32 - INSET * 2.0,
            self.extent.height as f32 - INSET * 2.0,
        ]
    }

    fn transform(self) -> Mat4 {
        if !self.tilted {
            return screen_to_clip(self.extent, [INSET; 2]);
        }
        let [width, height] = self.local_size();
        let aspect = self.extent.width as f32 / self.extent.height as f32;
        let fov = 50.0_f32.to_radians();
        let distance = 5.0;
        let units_per_pixel = 2.0 * (fov * 0.5).tan() * distance * 0.64 / height;
        Mat4::perspective_rh(fov, aspect, 0.1, 100.0)
            * Mat4::from_translation(Vec3::new(0.0, 0.0, -distance))
            * Mat4::from_rotation_y(-0.52)
            * Mat4::from_rotation_x(0.31)
            * Mat4::from_scale(Vec3::splat(units_per_pixel))
            * Mat4::from_translation(Vec3::new(-width * 0.5, -height * 0.5, 0.0))
    }

    fn layout(self, fonts: &mut TextSystem, text: &str) -> Layout {
        let mut builder = fonts.builder(text, !self.tilted);
        builder.push_default(StyleProperty::FontSize(self.pixels));
        if self.mixed {
            for (phrase, size, serif, bold, italic) in [
                ("Survey crews mapped", 32.0, true, false, false),
                ("bridge connects the two roads", 20.0, false, true, false),
                ("The watch changes at 06:30.", 24.0, true, false, true),
                ("spare tools, and radio batteries", 28.0, false, true, false),
                ("next crew can read them", 18.0, true, false, false),
            ] {
                for (start, _) in text.match_indices(phrase) {
                    let range = start..start + phrase.len();
                    builder.push(StyleProperty::FontSize(size), range.clone());
                    if serif {
                        builder.push(StyleProperty::FontFamily(GenericFamily::Serif.into()), range.clone());
                    }
                    if bold {
                        builder.push(StyleProperty::FontWeight(FontWeight::BOLD), range.clone());
                    }
                    if italic {
                        builder.push(StyleProperty::FontStyle(FontStyle::Italic), range);
                    }
                }
            }
        }
        let mut layout = builder.build(text);
        layout.break_all_lines(Some(self.local_size()[0]));
        layout
    }
}

struct Target {
    gpu: Arc<GpuDevice>,
    image: OwnedImage,
}

impl Drop for Target {
    fn drop(&mut self) {
        self.gpu.wait_idle();
        self.image.destroy(&self.gpu);
    }
}

struct Draw {
    text: TextBatch,
    quad_count: u32,
    transform: Mat4,
}

impl Recorder<()> for Draw {
    fn record(&mut self, _: &(), ctx: &PassContext<'_>) -> Result<()> {
        self.text.draw(ctx, 0..self.quad_count, self.transform)
    }
}

fn register(collection: &mut Collection, path: &OsStr) -> Result<FamilyId> {
    let data = std::fs::read(path).with_context(|| format!("read font {path:?}"))?;
    let families = collection.register_fonts(Blob::new(Arc::new(data)), None);
    ensure!(
        families.len() == 1 && families[0].1.len() == 1,
        "provide one outline font face per file"
    );
    Ok(families[0].0)
}

fn main() -> Result<()> {
    const USAGE: &str =
        "usage: text_renderers SANS_REGULAR.ttf [--mixed SANS_BOLD.ttf SERIF_REGULAR.ttf SERIF_ITALIC.ttf] [--verify]";
    let mut args = std::env::args_os().skip(1).peekable();
    let font = args.next().context(USAGE)?;
    let mut fonts = TextSystem::new(false);
    let collection = &mut fonts.fonts.collection;
    let sans = register(collection, &font)?;
    collection.set_generic_families(GenericFamily::SansSerif, [sans].into_iter());
    let mixed = args.peek().is_some_and(|flag| flag == "--mixed");
    if mixed {
        args.next();
        let bold = args.next().context(USAGE)?;
        let serif = args.next().context(USAGE)?;
        let italic = args.next().context(USAGE)?;
        let sans_bold = register(collection, &bold)?;
        let serif_id = register(collection, &serif)?;
        let serif_italic = register(collection, &italic)?;
        ensure!(
            sans == sans_bold && serif_id == serif_italic && sans != serif_id,
            "use two font families, each with the matching regular and styled faces"
        );
        collection.set_generic_families(GenericFamily::Serif, [serif_id].into_iter());
        println!("Additional faces: {bold:?}, {serif:?}, {italic:?}");
    }
    let verify = match args.next().as_deref() {
        None => false,
        Some(flag) if flag == "--verify" => true,
        _ => anyhow::bail!(USAGE),
    };
    ensure!(args.next().is_none(), "{USAGE}");

    let instance = Arc::new(GpuInstance::new(&[], Some(verify))?);
    ensure!(
        !verify || instance.validation,
        "--verify requires VK_LAYER_KHRONOS_validation"
    );
    let gpu = Arc::new(GpuDevice::new(instance, None)?);
    ensure!(verify || gpu.timestamp_period > 0.0, "GPU timestamps are unavailable");
    println!("GPU: {} | font: {font:?}", gpu.device_name);
    let hd = vk::Extent2D {
        width: 1920,
        height: 1080,
    };
    let uhd = vk::Extent2D {
        width: 3840,
        height: 2160,
    };
    let mixed_scenes = [
        Scene {
            name: "4K / mixed 16–32 px",
            extent: uhd,
            pixels: 16.0,
            tilted: false,
            mixed: true,
        },
        Scene {
            name: "4K / tilted mixed 16–32 px",
            extent: uhd,
            pixels: 16.0,
            tilted: true,
            mixed: true,
        },
    ];
    let regular = [
        Scene {
            name: "1080p / 16 px",
            extent: hd,
            pixels: 16.0,
            tilted: false,
            mixed: false,
        },
        Scene {
            name: "4K / 16 px",
            extent: uhd,
            pixels: 16.0,
            tilted: false,
            mixed: false,
        },
        Scene {
            name: "4K / 32 px",
            extent: uhd,
            pixels: 32.0,
            tilted: false,
            mixed: false,
        },
        Scene {
            name: "4K / tilted 32 px",
            extent: uhd,
            pixels: 32.0,
            tilted: true,
            mixed: false,
        },
    ];
    for &scene in if mixed { &mixed_scenes[..] } else { &regular[..] } {
        run_scene(&gpu, &mut fonts, scene, verify)?;
    }
    let counters = &gpu.instance.counters;
    ensure!(
        counters.errors() == 0 && counters.warnings() == 0,
        "Vulkan validation reported warnings or errors"
    );
    Ok(())
}

fn prepare<T>(measure: bool, mut build: impl FnMut() -> Result<T>) -> Result<(T, f64)> {
    if !measure {
        return Ok((build()?, 0.0));
    }
    let mut samples = [0.0; PREPARE_SAMPLES];
    let mut result = None;
    for sample in &mut samples {
        drop(result.take());
        let start = Instant::now();
        result = Some(build()?);
        *sample = start.elapsed().as_secs_f64() * 1000.0;
    }
    samples.sort_unstable_by(f64::total_cmp);
    Ok((result.expect("at least one sample"), samples[PREPARE_SAMPLES / 2]))
}

fn run_scene(gpu: &Arc<GpuDevice>, fonts: &mut TextSystem, scene: Scene, verify: bool) -> Result<()> {
    let long = PARAGRAPH.repeat(180);
    let layout = scene.layout(fonts, &long);
    let end = layout
        .lines()
        .take_while(|line| line.metrics().block_max_coord <= scene.local_size()[1])
        .last()
        .context("the viewport has no visible lines")?
        .text_range()
        .end;
    let block = long[..end].trim_end();
    let (layout, shaping_ms) = prepare(!verify, || Ok(scene.layout(fonts, block)))?;
    if scene.mixed {
        let mut faces = BTreeSet::new();
        let mut sizes = BTreeSet::new();
        for item in layout.lines().flat_map(|line| line.items()) {
            if let PositionedLayoutItem::GlyphRun(glyphs) = item {
                let run = glyphs.run();
                faces.insert((run.font().data.id(), run.font().index));
                sizes.insert(run.font_size().to_bits());
            }
        }
        ensure!(
            faces.len() == 4 && sizes.len() == 6,
            "expected four resolved faces and six sizes"
        );
    }

    let (atlas, prepare_ms) = prepare(!verify, || {
        let mut atlas = Atlas::new(ATLAS_SIZE)?;
        atlas.append(&layout, None)?;
        Ok(atlas)
    })?;
    let glyphs = atlas.quads().len();
    ensure!(glyphs > 1000, "expected a dense layout, got {glyphs} glyphs");

    println!("{} | {} lines | {} glyphs", scene.name, layout.len(), glyphs);
    let mut draw = Draw {
        text: TextBatch::new(gpu.clone(), &atlas, FORMAT, None)?,
        quad_count: u32::try_from(glyphs)?,
        transform: scene.transform(),
    };
    let mut graph = FrameGraph::new();
    let desc = ImageDesc::color_2d(
        FORMAT,
        scene.extent.width,
        scene.extent.height,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    let id = graph.image("benchmark target", desc, ResourceClass::External);
    graph
        .pass("text draw", PassKind::Graphics, ())
        .color(id, LoadOp::Clear)
        .root();
    let target = Target {
        gpu: gpu.clone(),
        image: OwnedImage::new(gpu, desc, "text benchmark target")?,
    };
    let binding = Binding::Image(ImageBinding {
        initial: vk::ImageLayout::UNDEFINED,
        image: target.image.image,
        view: target.image.view,
        format: FORMAT,
        extent: scene.extent,
        range: target.image.range(),
        identity: 1,
        sampled: None,
        storage: None,
    });
    let external = |resource| (resource == id).then_some(binding);
    let mut executor = Executor::new(
        gpu.clone(),
        ExecutorConfig {
            frames_in_flight: 1,
            profile: !verify,
            ..Default::default()
        },
    )?;
    executor.set_graph(graph)?;

    if verify {
        let slot = executor.begin_frame()?;
        executor.execute(slot, &mut draw, &external, &[], &[])?;
        executor.wait(slot)?;
        let image = upload::read_image_from(gpu, &target.image, 4, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)?
            .pop()
            .context("missing image readback")?;
        verify_coverage(&image, scene)?;
    } else {
        println!("  Parley layout: {shaping_ms:.3} ms p50 (warm font)");
        let samples = sample_gpu(&mut executor, &mut draw, &external)?;
        let p50 = samples[samples.len() / 2];
        let p95 = samples[(samples.len() * 95).div_ceil(100) - 1];
        let image = atlas.rgba().len();
        let quads = std::mem::size_of_val(atlas.quads());
        println!(
            "  MSDF GPU p50 {p50:.3} ms | p95 {p95:.3} ms | CPU prepare p50 {prepare_ms:.3} ms | image {image} B | instances {quads} B"
        );
    }
    Ok(())
}

fn sample_gpu(
    executor: &mut Executor<()>,
    draw: &mut Draw,
    external: &dyn Fn(ResourceId) -> Option<Binding>,
) -> Result<Vec<f32>> {
    let mut times = Vec::with_capacity(SAMPLES);
    let frames = WARMUP + SAMPLES;
    for frame in 0..=frames {
        let slot = executor.begin_frame()?;
        let recorded = executor.take_gpu_frames();
        if frame > WARMUP {
            times.push(read_timestamp(&recorded)?);
        }
        if frame != frames {
            executor.execute(slot, draw, external, &[], &[])?;
        }
    }
    ensure!(times.len() == SAMPLES, "wrong sample count");
    times.sort_unstable_by(f32::total_cmp);
    Ok(times)
}

fn read_timestamp(samples: &[Vec<f32>]) -> Result<f32> {
    match samples {
        [frame] if frame.len() == 1 && frame[0].is_finite() && frame[0] > 0.0 => Ok(frame[0]),
        _ => anyhow::bail!("expected one valid GPU timestamp interval"),
    }
}

fn verify_coverage(image: &[u8], scene: Scene) -> Result<()> {
    let len = scene.extent.width as usize * scene.extent.height as usize * 4;
    ensure!(image.len() == len, "unexpected readback size");
    let foreground = image.as_chunks::<4>().0.iter().filter(|pixel| pixel[0] >= 48).count();
    println!("  MSDF readback: {foreground} foreground pixels");
    ensure!(
        foreground > len / 100 && foreground < len / 8,
        "unexpected MSDF text coverage"
    );
    Ok(())
}
