//! Headless GPU tests (ignored by default: they need a Vulkan
//! 1.3 device). Run with
//! `cargo test -p ae_render --test gpu -- --ignored --test-threads=1`.
//! Every test runs with validation (including synchronization validation)
//! and fails on any validation error or warning.


use ash::vk;
use ae_render::demo::{DemoOutput, DemoRenderer, build_demo_graph};
use ae_render::device::{GpuDevice, GpuInstance};
use ae_render::executor::{Binding, BufferBinding, Executor, ExecutorConfig, ImageBinding};
use ae_framegraph::{BakeOptions, BufferDesc, ImageDesc, ResourceClass};
use ae_render::memory::{OwnedBuffer, OwnedImage};
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

fn gpu() -> Arc<GpuDevice> {
    let instance = Arc::new(GpuInstance::new(&[], Some(true)).expect("Vulkan instance"));
    assert!(instance.validation, "these tests need VK_LAYER_KHRONOS_validation");
    Arc::new(GpuDevice::new(instance, None).expect("Vulkan device"))
}

fn assert_clean(gpu: &GpuDevice) {
    let counters = &gpu.instance.counters;
    assert_eq!(
        (counters.errors(), counters.warnings()),
        (0, 0),
        "validation errors/warnings (see stderr)"
    );
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

/// Render the demo graph offscreen for `frames` frames and return the
/// readback bytes of the last one.
fn run_demo(gpu: &Arc<GpuDevice>, extent: vk::Extent2D, bake: BakeOptions, frames: usize, profile: bool) -> (Vec<u8>, usize) {
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile,
        bake,
    };
    let mut executor = Executor::new(gpu.clone(), config).unwrap();
    let (graph, resources) = build_demo_graph(extent, DemoOutput::Readback);
    let target_desc = match graph.resources[resources.target.index()].desc {
        ae_framegraph::ResourceDesc::Image(desc) => desc,
        _ => unreachable!(),
    };
    let readback_id = resources.readback.unwrap();
    let size = extent.width as u64 * extent.height as u64 * 4;
    let mut target = OwnedImage::new(gpu, target_desc, "demo target").unwrap();
    let mut readback = OwnedBuffer::new(
        gpu,
        BufferDesc {
            size,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
        },
        MemoryLocation::GpuToCpu,
        "demo readback",
    )
    .unwrap();
    assert!(executor.set_graph(graph).unwrap());
    let physical = executor.baked().unwrap().physical.len();
    let mut renderer = DemoRenderer::new(gpu.clone(), target_desc.format, 2).unwrap();
    let target_binding = Binding::Image(ImageBinding {
        image: target.image,
        view: target.view,
        format: target_desc.format,
        extent,
        range: target.range(),
        identity: 1 << 48,
    });
    let readback_binding = Binding::Buffer(BufferBinding {
        buffer: readback.buffer,
        offset: 0,
        size,
        identity: (1 << 48) + 1,
    });
    let mut params = OwnedBuffer::new(
        gpu,
        BufferDesc {
            size: ae_render::demo::PARAMS_BYTES,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
        },
        MemoryLocation::CpuToGpu,
        "demo params",
    )
    .unwrap();
    params.mapped().unwrap()[..16].copy_from_slice(&ae_render::demo::params_bytes([1.0, 0.8, 0.5, 1.0]));
    let params_binding = Binding::Buffer(BufferBinding {
        buffer: params.buffer,
        offset: 0,
        size: 16,
        identity: (1 << 48) + 2,
    });
    let params_id = resources.params_staging;
    let target_id = resources.target;
    let mut last = None;
    for _ in 0..frames {
        let token = executor.begin_frame().unwrap();
        executor
            .execute(
                token,
                &mut renderer,
                &|resource| {
                    if resource == target_id {
                        Some(target_binding)
                    } else if resource == readback_id {
                        Some(readback_binding)
                    } else if resource == params_id {
                        Some(params_binding)
                    } else {
                        None
                    }
                },
                &[],
                &[],
            )
            .unwrap();
        last = Some(token);
    }
    executor.wait(last.unwrap()).unwrap();
    let bytes = readback.mapped().expect("host-visible readback").to_vec();
    let stats = executor.stats();
    assert_eq!(stats.bakes, 1);
    assert_eq!(stats.frames, frames as u64);
    if profile {
        assert_eq!(stats.pass_gpu_ms.len(), 9, "one timing per baked pass");
        assert!(stats.pass_gpu_ms.iter().any(|ms| *ms > 0.0), "timestamps were read back");
    }
    drop(renderer);
    drop(executor);
    target.destroy(gpu);
    readback.destroy(gpu);
    params.destroy(gpu);
    let _ = ResourceClass::External;
    let _ = ImageDesc::extent;
    (bytes, physical)
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn demo_graph_renders_validation_clean_and_aliasing_is_invisible() {
    let gpu = gpu();
    let extent = vk::Extent2D { width: 256, height: 128 };
    let (aliased, aliased_physical) = run_demo(&gpu, extent, BakeOptions { aliasing: true }, 5, true);
    let (plain, plain_physical) = run_demo(&gpu, extent, BakeOptions { aliasing: false }, 5, true);
    // R5: profiling changes no rendering behaviour.
    let (unprofiled, _) = run_demo(&gpu, extent, BakeOptions { aliasing: true }, 5, false);
    assert_eq!(fnv(&unprofiled), fnv(&aliased), "profiling must not change the image");
    assert!(aliased_physical < plain_physical, "{aliased_physical} vs {plain_physical}");
    assert_eq!(fnv(&aliased), fnv(&plain), "aliasing must not change the image");
    // The image is not blank: the left (blurred) and right (inverted, tinted)
    // halves differ, and the triangle shows up against the background.
    let pixel = |x: u32, y: u32| {
        let at = ((y * extent.width + x) * 4) as usize;
        [aliased[at], aliased[at + 1], aliased[at + 2]]
    };
    assert_ne!(pixel(64, 80), pixel(192, 80));
    assert_ne!(pixel(64, 80), pixel(4, 4));
    assert_clean(&gpu);
}

/// The messenger really counts: an invalid create-info (size 0) is caught.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn validation_counters_catch_errors() {
    let gpu = gpu();
    let info = vk::BufferCreateInfo::default()
        .size(0)
        .usage(vk::BufferUsageFlags::TRANSFER_DST);
    eprintln!("(an intentional validation error follows)");
    if let Ok(buffer) = unsafe { gpu.device.create_buffer(&info, None) } {
        unsafe { gpu.device.destroy_buffer(buffer, None) };
    }
    assert!(gpu.instance.counters.errors() > 0, "validation layer did not report");
}

/// Synchronization validation is on: two unsynchronized writes to one buffer
/// in a command buffer are a reported hazard. (Proves the clean runs above
/// mean something.)
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn sync_validation_catches_hazards() {
    let gpu = gpu();
    let device = &gpu.device;
    let mut buffer = OwnedBuffer::new(
        &gpu,
        BufferDesc {
            size: 256,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
        },
        MemoryLocation::GpuOnly,
        "hazard",
    )
    .unwrap();
    unsafe {
        let pool = device
            .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(gpu.queue_family), None)
            .unwrap();
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(1),
            )
            .unwrap()[0];
        device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
        device.cmd_fill_buffer(cmd, buffer.buffer, 0, 256, 1);
        device.cmd_fill_buffer(cmd, buffer.buffer, 0, 256, 2);
        device.end_command_buffer(cmd).unwrap();
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        eprintln!("(an intentional hazard follows)");
        device.queue_submit(gpu.queue, &[submit], vk::Fence::null()).unwrap();
        device.queue_wait_idle(gpu.queue).unwrap();
        device.destroy_command_pool(pool, None);
    }
    buffer.destroy(&gpu);
    assert!(
        gpu.instance.counters.errors() + gpu.instance.counters.warnings() > 0,
        "synchronization validation did not report the WAW hazard"
    );
}

/// First use of an image with `ColorAttachment(Load)` (layout `UNDEFINED`,
/// nothing to load) clears it, so the first frame is deterministic; later
/// frames really load (the contents survive).
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn first_use_load_becomes_a_clear() {
    use ae_render::executor::{PassContext, Recorder};
    use ae_framegraph::{FrameGraph, LoadOp, PassKind, ResourceId, Role};

    const CLEAR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
    struct Paint {
        canvas: ResourceId,
        readback: ResourceId,
    }
    impl Recorder<u8> for Paint {
        fn clear_color(&self, _pass: &u8, _resource: ResourceId) -> [f32; 4] {
            CLEAR
        }
        fn record(&mut self, pass: &u8, ctx: &PassContext<'_>) -> anyhow::Result<()> {
            if *pass == 1 {
                ae_render::demo::record_readback(ctx, ctx.image(self.canvas)?, ctx.buffer(self.readback)?);
            }
            Ok(())
        }
    }

    let gpu = gpu();
    let extent = vk::Extent2D { width: 8, height: 8 };
    let desc = ImageDesc::color_2d(
        vk::Format::R8G8B8A8_UNORM,
        extent.width,
        extent.height,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    let size = 8 * 8 * 4;
    let buffer_desc = BufferDesc {
        size,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
    };
    let mut graph = FrameGraph::<u8>::new();
    let canvas = graph.image("canvas", desc, ResourceClass::External);
    let readback = graph.buffer("readback", buffer_desc, ResourceClass::External);
    // The paint pass draws nothing: whatever the attachment holds afterwards
    // comes from its load op.
    graph.pass("paint", PassKind::Graphics, 0).color(canvas, LoadOp::Load).id();
    graph
        .pass("copy", PassKind::Transfer, 1)
        .access(canvas, Role::TransferSrc)
        .access(readback, Role::TransferDst)
        .id();
    graph.pass("host read", PassKind::Transfer, 2).access(readback, Role::HostRead).root().id();

    let mut image = OwnedImage::new(&gpu, desc, "canvas").unwrap();
    let mut buffer = OwnedBuffer::new(&gpu, buffer_desc, MemoryLocation::GpuToCpu, "readback").unwrap();
    let bindings = [
        Binding::Image(ImageBinding {
            image: image.image,
            view: image.view,
            format: desc.format,
            extent,
            range: image.range(),
            identity: 1 << 48,
        }),
        Binding::Buffer(BufferBinding {
            buffer: buffer.buffer,
            offset: 0,
            size,
            identity: (1 << 48) + 1,
        }),
    ];
    let mut executor = Executor::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    executor.set_graph(graph).unwrap();
    let mut recorder = Paint { canvas, readback };
    let expected: Vec<u8> = CLEAR.iter().map(|c| (c * 255.0).round() as u8).collect();
    for frame in 0..3 {
        let token = executor.begin_frame().unwrap();
        executor
            .execute(token, &mut recorder, &|resource| bindings.get(resource.index()).copied(), &[], &[])
            .unwrap();
        executor.wait(token).unwrap();
        let bytes = buffer.mapped().unwrap();
        for (index, pixel) in bytes.chunks(4).enumerate() {
            let close = pixel.iter().zip(&expected).all(|(a, b)| a.abs_diff(*b) <= 1);
            assert!(close, "frame {frame}, pixel {index}: {pixel:?} vs {expected:?}");
        }
    }
    drop(executor);
    image.destroy(&gpu);
    buffer.destroy(&gpu);
    assert_clean(&gpu);
}


/// Depth and multiple render targets through the graph. Three passes share
/// one transient depth buffer: `near` clears it and draws red on the left
/// half at depth 0.3 into two colour targets; `far` loads both targets and
/// the depth, draws green everywhere at 0.6 (hidden on the left) and a blue
/// square at 0.1; `decal` tests against the depth read-only and paints white
/// at 0.45 over the bottom half (only where the far green is). Target 1 gets
/// half of each colour from the two MRT passes and nothing from `decal`.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn depth_and_multiple_targets() {
    depth_and_multiple_targets_with(vk::Format::D32_SFLOAT);
}

/// The same with a combined depth-stencil format (stencil attachment bound,
/// barriers over both aspects).
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn depth_stencil_and_multiple_targets() {
    depth_and_multiple_targets_with(vk::Format::D32_SFLOAT_S8_UINT);
}

fn depth_and_multiple_targets_with(depth_format: vk::Format) {
    use ae_framegraph::{FrameGraph, LoadOp, PassKind, ResourceId, Role};
    use ae_render::executor::{PassContext, Recorder};
    use ae_render::pipeline::{self, DepthState, GraphicsDesc};

    const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
    const SIZE: u32 = 32;
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct Rect {
        rect: [f32; 4],
        color: [f32; 4],
        depth: [f32; 4],
    }
    let rect = |rect: [f32; 4], color: [f32; 3], depth: f32| Rect { rect, color: [color[0], color[1], color[2], 1.0], depth: [depth, 0.0, 0.0, 0.0] };
    let red = rect([-1.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0], 0.3);
    let green = rect([-1.0, -1.0, 1.0, 1.0], [0.0, 1.0, 0.0], 0.6);
    let blue = rect([-0.75, -0.75, -0.25, -0.25], [0.0, 0.0, 1.0], 0.1);
    let white = rect([-1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0], 0.45);

    struct Draw {
        mrt: vk::Pipeline,
        decal: vk::Pipeline,
        layout: vk::PipelineLayout,
        targets: [ResourceId; 2],
        readback: ResourceId,
        rects: [Vec<Rect>; 3],
    }
    impl Recorder<u8> for Draw {
        fn record(&mut self, pass: &u8, ctx: &PassContext<'_>) -> anyhow::Result<()> {
            let d = ctx.device;
            match *pass {
                0..=2 => unsafe {
                    let pipeline = if *pass == 2 { self.decal } else { self.mrt };
                    d.cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    for r in &self.rects[*pass as usize] {
                        d.cmd_push_constants(ctx.cmd, self.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pipeline::push_bytes(r));
                        d.cmd_draw(ctx.cmd, 6, 1, 0, 0);
                    }
                },
                3 => {
                    let buffer = ctx.buffer(self.readback)?;
                    for (index, target) in self.targets.iter().enumerate() {
                        let image = ctx.image(*target)?;
                        let region = vk::BufferImageCopy::default()
                            .buffer_offset((index as u32 * SIZE * SIZE * 4) as u64)
                            .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                            .image_extent(vk::Extent3D { width: SIZE, height: SIZE, depth: 1 });
                        unsafe { d.cmd_copy_image_to_buffer(ctx.cmd, image.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buffer.buffer, &[region]) };
                    }
                }
                _ => {}
            }
            Ok(())
        }
    }

    let gpu = gpu();
    let extent = vk::Extent2D { width: SIZE, height: SIZE };
    let color = ImageDesc::color_2d(FORMAT, SIZE, SIZE, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC);
    let bytes = (SIZE * SIZE * 4) as u64;
    let buffer_desc = BufferDesc { size: 2 * bytes, usage: vk::BufferUsageFlags::TRANSFER_DST };
    let mut graph = FrameGraph::<u8>::new();
    let t0 = graph.image("target 0", color, ResourceClass::External);
    let t1 = graph.image("target 1", color, ResourceClass::External);
    let readback = graph.buffer("readback", buffer_desc, ResourceClass::External);
    let depth = graph.image("depth", ImageDesc::depth_2d(depth_format, SIZE, SIZE, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT), ResourceClass::Transient);
    graph.pass("near", PassKind::Graphics, 0).color(t0, LoadOp::Clear).color(t1, LoadOp::Clear).depth(depth, LoadOp::Clear).id();
    graph.pass("far", PassKind::Graphics, 1).color(t0, LoadOp::Load).color(t1, LoadOp::Load).depth(depth, LoadOp::Load).id();
    graph.pass("decal", PassKind::Graphics, 2).color(t0, LoadOp::Load).depth_read(depth).id();
    graph.pass("copy", PassKind::Transfer, 3).access(t0, Role::TransferSrc).access(t1, Role::TransferSrc).discard(readback, Role::TransferDst).id();
    graph.pass("host read", PassKind::Transfer, 4).access(readback, Role::HostRead).root().id();

    let device = &gpu.device;
    let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let layout = pipeline::pipeline_layout(device, &[], std::mem::size_of::<Rect>() as u32, stages).unwrap();
    let vertex = include_bytes!(concat!(env!("OUT_DIR"), "/test_rect.vert.spv"));
    let make = |fragment: &[u8], formats: &[vk::Format], write| {
        let depth = Some(DepthState { format: depth_format, write, compare: vk::CompareOp::LESS });
        let desc = GraphicsDesc { vertex, fragment, layout, color_formats: formats, depth, topology: vk::PrimitiveTopology::TRIANGLE_LIST, blend: false };
        pipeline::graphics_pipeline(device, &desc).unwrap()
    };
    let mrt = make(include_bytes!(concat!(env!("OUT_DIR"), "/test_mrt.frag.spv")), &[FORMAT, FORMAT], true);
    let decal = make(include_bytes!(concat!(env!("OUT_DIR"), "/test_flat.frag.spv")), &[FORMAT], false);

    let mut images = [OwnedImage::new(&gpu, color, "target 0").unwrap(), OwnedImage::new(&gpu, color, "target 1").unwrap()];
    let mut buffer = OwnedBuffer::new(&gpu, buffer_desc, MemoryLocation::GpuToCpu, "readback").unwrap();
    let image_binding = |image: &OwnedImage, identity| Binding::Image(ImageBinding { image: image.image, view: image.view, format: FORMAT, extent, range: image.range(), identity });
    let bindings = [
        image_binding(&images[0], 1 << 48),
        image_binding(&images[1], (1 << 48) + 1),
        Binding::Buffer(BufferBinding { buffer: buffer.buffer, offset: 0, size: 2 * bytes, identity: (1 << 48) + 2 }),
    ];
    let mut executor = Executor::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    executor.set_graph(graph).unwrap();
    let mut recorder = Draw { mrt, decal, layout, targets: [t0, t1], readback, rects: [vec![red], vec![green, blue], vec![white]] };
    for frame in 0..3 {
        let token = executor.begin_frame().unwrap();
        executor.execute(token, &mut recorder, &|resource| bindings.get(resource.index()).copied(), &[], &[]).unwrap();
        executor.wait(token).unwrap();
        let data = buffer.mapped().unwrap();
        let pixel = |target: u32, x: u32, y: u32| {
            let at = (target as u64 * bytes) as usize + ((y * SIZE + x) * 4) as usize;
            [data[at], data[at + 1], data[at + 2]]
        };
        // Clip y = -1 is the top row. Blue square: x, y in [4, 12).
        let (top, bottom, left, right) = (6, 24, 6, 24);
        assert_eq!(pixel(0, left, top), [0, 0, 255], "frame {frame}: blue in front of red");
        assert_eq!(pixel(0, 2, top), [255, 0, 0], "frame {frame}: red hides far green");
        assert_eq!(pixel(0, right, top), [0, 255, 0], "frame {frame}: green where nothing is nearer");
        assert_eq!(pixel(0, right, bottom), [255, 255, 255], "frame {frame}: decal over green");
        assert_eq!(pixel(0, left, bottom), [255, 0, 0], "frame {frame}: decal behind red");
        // Half intensity: 127 or 128 depending on the device's rounding.
        let half = |p: [u8; 3], want: [u8; 3]| p.iter().zip(want).all(|(a, b)| a.abs_diff(b) <= 1);
        assert!(half(pixel(1, left, top), [0, 0, 128]), "frame {frame}: second target {:?}", pixel(1, left, top));
        assert!(half(pixel(1, 2, bottom), [128, 0, 0]), "frame {frame}: second target, no decal {:?}", pixel(1, 2, bottom));
        assert!(half(pixel(1, right, bottom), [0, 128, 0]), "frame {frame}: second target, no decal {:?}", pixel(1, right, bottom));
    }
    drop(executor);
    unsafe {
        device.destroy_pipeline(mrt, None);
        device.destroy_pipeline(decal, None);
        device.destroy_pipeline_layout(layout, None);
    }
    for image in &mut images {
        image.destroy(&gpu);
    }
    buffer.destroy(&gpu);
    assert_clean(&gpu);
}

/// The terrain through the frame graph, offscreen: a generated 513² map,
/// LOD compute pass, draw with reverse-Z depth, readback. The subdivision
/// refines over the frames (one split and one merge pass per frame), the
/// view shows terrain and sky, and validation stays silent.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn terrain_renders_offscreen() {
    use ae_render::terrain::app::{SKY, TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{build_pyramid, generate};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};

    let gpu = gpu();
    let heightmap = generate(513, 513, 7);
    let pyramid = build_pyramid(&heightmap, 400.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 400.0, 0.0, Some(30.0));
    let source = Arc::new(TerrainSource { heightmap, pyramid, info, materials: None });
    let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback);
    {
        // Low over the south edge, looking north: terrain below, sky above.
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera.pitch = 0.25;
        view.camera.distance = 1500.0;
    }
    let extent = vk::Extent2D { width: 320, height: 200 };
    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo { extent, format: vk::Format::R8G8B8A8_SRGB };
    let config = ExecutorConfig { frames_in_flight: 2, profile: false, bake: BakeOptions::default() };
    run_headless(&gpu, &mut app, target, 24, config).unwrap();
    let leaves = app.stats().lock().unwrap().leaves;
    let pixels = app.readback().unwrap();
    if let Ok(path) = std::env::var("AE_TEST_CAPTURE") {
        // Binary PPM, for looking at the result.
        let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
        ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
        std::fs::write(&path, ppm).unwrap();
    }
    // The capture is sRGB-encoded, like the swapchain.
    let encode = |c: f32| if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    let sky: Vec<u8> = SKY.iter().map(|&c| (encode(c) * 255.0).round() as u8).collect();
    let is_sky = |p: &[u8]| p[..3].iter().zip(&sky).all(|(a, b)| a.abs_diff(*b) <= 2);
    let sky_pixels = pixels.chunks(4).filter(|p| is_sky(p)).count();
    let total = (extent.width * extent.height) as usize;
    eprintln!("{leaves} leaves, sky {sky_pixels} of {total} pixels");
    assert!(sky_pixels > total / 50 && sky_pixels < total * 9 / 10, "terrain and sky both visible: {sky_pixels} of {total}");
    let top = &pixels[..(extent.width * 4) as usize];
    let bottom = &pixels[pixels.len() - (extent.width * 4) as usize..];
    assert!(top.chunks(4).filter(|p| is_sky(p)).count() > bottom.chunks(4).filter(|p| is_sky(p)).count(), "sky is at the top");
    assert!(leaves > 64, "refined well beyond the two roots: {leaves}");
    ae_render::thread::RenderApp::destroy(&mut app, &gpu);
    assert_clean(&gpu);
}

/// Units over the terrain: an instance stream of four units, reset + cull +
/// indirect draw after the terrain draw into the same target and depth
/// buffer. The cull passes three (the fourth is behind the camera: the
/// instance count read back is 3); each shows its class colour at its
/// projected position; the previous position of a moved unit (alpha 1
/// shows the latest) shows none; validation stays silent.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn units_render_over_terrain() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::units::{InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Vec2, Vec4};

    let gpu = gpu();
    // Flat ground at 50 with water at 60 (class 1 floats on it).
    let heightmap = Heightmap { width: 129, height: 129, data: vec![0.5; 129 * 129] };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, Some(60.0));
    let source = Arc::new(TerrainSource { heightmap, pyramid, info, materials: None });

    let red = [1.0, 0.02, 0.02];
    let blue = [0.02, 0.02, 1.0];
    let magenta = [1.0, 0.02, 1.0];
    let classes = vec![UnitClass { color: red, floats: false }, UnitClass { color: blue, floats: true }, UnitClass { color: magenta, floats: false }];
    // (x, y, prev_x, prev_y, class): the red unit moved from (420, 512).
    let units = [(560.0f32, 512.0f32, 420.0f32, 512.0f32, 0u32), (512.0, 600.0, 512.0, 600.0, 1), (470.0, 440.0, 470.0, 440.0, 2), (512.0, -4000.0, 512.0, -4000.0, 0)];
    let mut columns: [Vec<u32>; 8] = Default::default();
    for &(x, y, px, py, class) in &units {
        for (a, word) in [x.to_bits(), y.to_bits(), px.to_bits(), py.to_bits(), 1.0f32.to_bits(), 0.0f32.to_bits(), 4.0f32.to_bits(), class].into_iter().enumerate() {
            columns[a].push(word);
        }
    }
    let feed = Arc::new(UnitFeed::default());
    feed.publish(Arc::new(InstanceStream { tick: 7, published: std::time::Instant::now(), tick_seconds: 0.0, columns }));
    let setup = UnitsSetup { feed, capacity: 64, classes, min_pixels: 28.0 };
    let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(setup);
    let camera = {
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera.target = Vec2::new(512.0, 512.0);
        view.camera.pitch = 1.4;
        view.camera.distance = 700.0;
        view.fog = false;
        view.camera
    };
    let extent = vk::Extent2D { width: 320, height: 240 };
    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo { extent, format: vk::Format::R8G8B8A8_SRGB };
    let config = ExecutorConfig { frames_in_flight: 2, profile: false, bake: BakeOptions::default() };
    run_headless(&gpu, &mut app, target, 12, config).unwrap();
    let pixels = app.readback().unwrap();
    if let Ok(path) = std::env::var("AE_TEST_CAPTURE") {
        let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
        ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
        std::fs::write(&path, ppm).unwrap();
    }
    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let pixel = |x: f32, y: f32, h: f32| -> [u8; 3] {
        let clip = view_proj * Vec4::new(x, h, y, 1.0);
        let (u, v) = ((clip.x / clip.w * 0.5 + 0.5) * extent.width as f32, (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32);
        assert!(u >= 0.0 && v >= 0.0 && u < extent.width as f32 && v < extent.height as f32, "({x}, {y}) projects into the view");
        let i = (v as usize * extent.width as usize + u as usize) * 4;
        [pixels[i], pixels[i + 1], pixels[i + 2]]
    };
    let [r, g, b] = pixel(560.0, 512.0, 50.0).map(|c| c as i32);
    assert!(r > 2 * g && r > 2 * b, "red unit at its latest position: {r} {g} {b}");
    let [r, g, b] = pixel(512.0, 600.0, 60.0).map(|c| c as i32);
    assert!(b > 2 * r && b > 2 * g, "blue unit on the water: {r} {g} {b}");
    let [r, g, b] = pixel(470.0, 440.0, 50.0).map(|c| c as i32);
    assert!(r > 2 * g && b > 2 * g, "magenta unit: {r} {g} {b}");
    let [r, g, b] = pixel(420.0, 512.0, 50.0).map(|c| c as i32);
    assert!(!(r > 2 * g && r > 2 * b), "nothing red where the red unit was a tick ago: {r} {g} {b}");
    assert_eq!(app.last_unit_counts(), Some((4, 3)), "four rows uploaded, three pass the cull");
    ae_render::thread::RenderApp::destroy(&mut app, &gpu);
    assert_clean(&gpu);
}
