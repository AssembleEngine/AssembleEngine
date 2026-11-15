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
        record_threads: 1,
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
    depth_and_multiple_targets_with(vk::Format::D32_SFLOAT, 1);
    depth_and_multiple_targets_with(vk::Format::D32_SFLOAT, 3);
}

/// The same with a combined depth-stencil format (stencil attachment bound,
/// barriers over both aspects).
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn depth_stencil_and_multiple_targets() {
    depth_and_multiple_targets_with(vk::Format::D32_SFLOAT_S8_UINT, 1);
    depth_and_multiple_targets_with(vk::Format::D32_SFLOAT_S8_UINT, 3);
}

/// With `record_threads > 1` the passes are recorded in parallel (one
/// command buffer each): the same pixels.
fn depth_and_multiple_targets_with(depth_format: vk::Format, record_threads: usize) {
    use ae_framegraph::{FrameGraph, LoadOp, PassKind, ResourceId, Role};
    use ae_render::executor::{PassContext, Recorder, SharedRecorder};
    use ae_render::pipeline::{self, Blend, DepthState, GraphicsDesc};

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
            SharedRecorder::record(self, pass, ctx)
        }

        fn shared(&self) -> Option<Box<dyn SharedRecorder<u8> + '_>> {
            Some(Box::new(self))
        }
    }
    impl SharedRecorder<u8> for &Draw {
        fn record(&self, pass: &u8, ctx: &PassContext<'_>) -> anyhow::Result<()> {
            SharedRecorder::record(*self, pass, ctx)
        }
    }
    impl SharedRecorder<u8> for Draw {
        fn record(&self, pass: &u8, ctx: &PassContext<'_>) -> anyhow::Result<()> {
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
        let desc = GraphicsDesc { vertex, fragment, layout, color_formats: formats, depth, topology: vk::PrimitiveTopology::TRIANGLE_LIST, blend: Blend::Off };
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
    let mut executor = Executor::new(gpu.clone(), ExecutorConfig { record_threads, ..ExecutorConfig::default() }).unwrap();
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
    assert_eq!(executor.stats().parallel_frames, if record_threads > 1 { 3 } else { 0 }, "recorded on the threads");
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
    let config = ExecutorConfig { frames_in_flight: 2, profile: false, bake: BakeOptions::default(), record_threads: 1 };
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
    let config = ExecutorConfig { frames_in_flight: 2, profile: false, bake: BakeOptions::default(), record_threads: 1 };
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

/// Terrain edits reach the GPU while rendering: a raised, re-materialled
/// square uploaded as texture regions (`TerrainEditor` →
/// `TerrainApp::terrain_edits`) lifts the unit standing in it to the new
/// ground and changes the terrain there, and leaves the rest of the
/// picture as it was (a region update that wrote the wrong texels would
/// not; one that discarded the old contents would not either on a driver
/// that discards, which this test's NVIDIA driver does not). A second
/// update is pushed while frames are in flight.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn terrain_edits_reach_the_heightfield() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::edit::TerrainEditor;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{Headless, TargetInfo};
    use ae_render::units::{InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Vec2, Vec4};

    let gpu = gpu();
    // 128 x 128 cells of flat grass at 50.
    let cells = (128usize, 128usize);
    let (heightmap, scale, offset) = Heightmap::from_cells(cells, |_, _| 50.0, Some((100.0, 0.0)));
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, scale, offset, None);
    let source = Arc::new(TerrainSource { heightmap, pyramid, info, materials: Some(vec![0; cells.0 * cells.1]) });

    // A red unit in the middle of the square to raise, a blue one outside.
    let mut columns: [Vec<u32>; 8] = Default::default();
    for (x, y, class) in [(512.0f32, 512.0f32, 0u32), (360.0, 600.0, 1)] {
        for (a, word) in [x.to_bits(), y.to_bits(), x.to_bits(), y.to_bits(), 1.0f32.to_bits(), 0.0f32.to_bits(), 5.0f32.to_bits(), class].into_iter().enumerate() {
            columns[a].push(word);
        }
    }
    let feed = Arc::new(UnitFeed::default());
    feed.publish(Arc::new(InstanceStream { tick: 1, published: std::time::Instant::now(), tick_seconds: 0.0, columns }));
    let classes = vec![UnitClass { color: [1.0, 0.02, 0.02], floats: false }, UnitClass { color: [0.02, 0.02, 1.0], floats: false }];
    let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(UnitsSetup { feed, capacity: 8, classes, min_pixels: 6.0 });
    let camera = {
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera.target = Vec2::new(512.0, 512.0);
        view.camera.pitch = 0.7;
        view.camera.distance = 600.0;
        view.fog = false;
        view.camera
    };
    let extent = vk::Extent2D { width: 320, height: 240 };
    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo { extent, format: vk::Format::R8G8B8A8_SRGB };
    let config = ExecutorConfig { frames_in_flight: 2, profile: false, bake: BakeOptions::default(), record_threads: 1 };
    let mut run = Headless::new(&gpu, &mut app, target, config).unwrap();
    let step = |run: &mut Headless<_>, app: &mut TerrainApp| {
        run.frames(app, 12).unwrap();
        run.wait().unwrap();
        app.readback().unwrap()
    };
    let before = step(&mut run, &mut app);

    // Cells 58..=70 (world 464..568) become rock at 90.
    let raised = (58usize, 58usize, 70usize, 70usize);
    let inside = |x: usize, y: usize| (raised.0..=raised.2).contains(&x) && (raised.1..=raised.3).contains(&y);
    let mut editor = TerrainEditor::new(&source);
    let update = editor.edit(3, &[(58, 58, 70, 70)], |x, y| if inside(x, y) { 90.0 } else { 50.0 }, |_, _| 3).unwrap();
    eprintln!("terrain update: {} bytes in {} pyramid and {} error regions", update.bytes(), update.pyramid.len(), update.errors.len());
    let edits = app.terrain_edits();
    edits.lock().unwrap().pending.push(update);
    let after = step(&mut run, &mut app);
    {
        let e = edits.lock().unwrap();
        assert_eq!((e.applied, e.version, e.pending.len()), (1, 3, 0), "applied at the next frame");
    }
    if let Ok(dir) = std::env::var("AE_TEST_CAPTURE") {
        for (name, pixels) in [("before", &before), ("after", &after)] {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            std::fs::write(format!("{dir}/edit-{name}.ppm"), ppm).unwrap();
        }
    }

    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let project = |x: f32, y: f32, h: f32| -> (f32, f32) {
        let clip = view_proj * Vec4::new(x, h, y, 1.0);
        ((clip.x / clip.w * 0.5 + 0.5) * extent.width as f32, (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32)
    };
    let rgb = |pixels: &[u8], (u, v): (f32, f32)| -> [i32; 3] {
        let i = (v as usize * extent.width as usize + u as usize) * 4;
        [pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32]
    };
    let red = |[r, g, b]: [i32; 3]| r > 2 * g && r > 2 * b;
    let blue = |[r, g, b]: [i32; 3]| b > 2 * r && b > 2 * g;
    let (low, high) = (project(512.0, 512.0, 50.0), project(512.0, 512.0, 90.0));
    eprintln!("red unit projects to {low:?} on the old ground, {high:?} on the new");
    assert!((low.1 - high.1).abs() > 8.0, "the camera sees the lift");
    assert!(red(rgb(&before, low)), "before: the red unit on the old ground {:?}", rgb(&before, low));
    assert!(red(rgb(&after, high)), "after: the red unit on the raised ground {:?}", rgb(&after, high));
    assert!(!red(rgb(&after, low)), "after: nothing red at the old ground height {:?}", rgb(&after, low));
    assert!(blue(rgb(&before, project(360.0, 600.0, 50.0))) && blue(rgb(&after, project(360.0, 600.0, 50.0))), "the blue unit stays");
    // The raised top looks different (rock, lit as the top of a block).
    let top = project(480.0, 540.0, 90.0);
    let diff = rgb(&before, top).iter().zip(rgb(&after, top)).map(|(a, b)| (a - b).abs()).sum::<i32>();
    assert!(diff > 30, "the raised square changed colour: {:?} -> {:?}", rgb(&before, top), rgb(&after, top));

    // Outside the raised block's screen box (with a margin), nothing moved.
    // The box is a cell wider than the changed corners (world 464..=568)
    // on each side: shading takes normals from neighbouring height texels.
    let mut box_ = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (x, y) in [(456.0, 456.0), (576.0, 456.0), (456.0, 576.0), (576.0, 576.0)] {
        for h in [50.0, 90.0] {
            let (u, v) = project(x, y, h);
            box_ = (box_.0.min(u), box_.1.min(v), box_.2.max(u), box_.3.max(v));
        }
    }
    let margin = 6.0;
    let (mut outside, mut moved) = (0, 0);
    for v in 0..extent.height {
        for u in 0..extent.width {
            let (uf, vf) = (u as f32, v as f32);
            if uf > box_.0 - margin && uf < box_.2 + margin && vf > box_.1 - margin && vf < box_.3 + margin {
                continue;
            }
            outside += 1;
            let a = rgb(&before, (uf, vf));
            let b = rgb(&after, (uf, vf));
            moved += a.iter().zip(b).any(|(a, b)| (a - b).abs() > 2) as usize;
        }
    }
    eprintln!("raised block on screen {box_:?}; {moved} of {outside} pixels outside it changed");
    assert!(outside > 40_000 && moved == 0, "{moved} of {outside} pixels outside the edit changed");

    // Several small rectangles in one update, of sizes that leave the
    // regions after them unaligned unless the upload aligns them (a 2 x 1
    // cell rectangle is 3 x 2 corners, 24 bytes of heights).
    let update = editor.edit(4, &[(20, 20, 21, 20), (100, 7, 100, 7), (3, 90, 5, 90)], |x, y| if inside(x, y) { 90.0 } else { 50.0 + (x % 3) as f32 }, |_, _| 2).unwrap();
    assert_eq!(update.heights[0].bytes.len(), 24);
    run.frames(&mut app, 6).unwrap();
    edits.lock().unwrap().pending.push(update);
    run.frames(&mut app, 6).unwrap();
    run.wait().unwrap();
    assert_eq!(edits.lock().unwrap().applied, 2);
    run.finish().unwrap();
    ae_render::thread::RenderApp::destroy(&mut app, &gpu);
    assert_clean(&gpu);
}

/// A 64 x 64-cell map (8 m cells) from `ground(x, y)` and `material(x, y)`
/// seen from straight above its centre at `distance`, rendered offscreen with `tileset`
/// (and its weights view if `weights`): the pixels and a world → pixel
/// projection.
fn tileset_scene(
    gpu: &Arc<GpuDevice>,
    ground: impl Fn(usize, usize) -> f32,
    material: impl Fn(usize, usize) -> u8,
    tileset: Option<Arc<ae_render::terrain::tileset::Tileset>>,
    weights: bool,
    distance: f32,
    vision: Option<Arc<ae_render::terrain::app::VisionFeed>>,
) -> (Vec<u8>, impl Fn(f32, f32, f32) -> (f32, f32)) {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use glam::{Vec2, Vec4};
    let cells = (64usize, 64usize);
    let (heightmap, scale, offset) = Heightmap::from_cells(cells, &ground, None);
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, scale, offset, None);
    let materials = (0..cells.1).flat_map(|y| (0..cells.0).map(move |x| (x, y))).map(|(x, y)| material(x, y)).collect();
    let source = Arc::new(TerrainSource { heightmap, pyramid, info, materials: Some(materials) });
    let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback);
    if let Some(t) = tileset {
        app = app.with_tileset(t);
    }
    if let Some(v) = vision {
        app = app.with_vision(v);
    }
    let camera = {
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera.target = Vec2::new(256.0, 256.0);
        view.camera.pitch = 1.55;
        view.camera.distance = distance;
        view.fog = false;
        view.weights = weights;
        view.camera
    };
    let extent = vk::Extent2D { width: 320, height: 240 };
    app.create_offscreen(gpu, extent).unwrap();
    let target = TargetInfo { extent, format: vk::Format::R8G8B8A8_SRGB };
    run_headless(gpu, &mut app, target, 16, ExecutorConfig { frames_in_flight: 2, profile: false, bake: BakeOptions::default(), record_threads: 1 }).unwrap();
    let pixels = app.readback().unwrap();
    if let Ok(dir) = std::env::var("AE_TEST_CAPTURE") {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
        ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
        std::fs::write(format!("{dir}/tileset-{}.ppm", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)), ppm).unwrap();
    }
    ae_render::thread::RenderApp::destroy(&mut app, gpu);
    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let project = move |x: f32, y: f32, h: f32| {
        let clip = view_proj * Vec4::new(x, h, y, 1.0);
        ((clip.x / clip.w * 0.5 + 0.5) * extent.width as f32, (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32)
    };
    (pixels, project)
}

/// The pixel's nearest colour among `colors` (sRGB bytes).
fn nearest(p: &[u8], colors: &[[u8; 3]]) -> usize {
    let d = |c: &[u8; 3]| (0..3).map(|i| (p[i] as i32 - c[i] as i32).pow(2)).sum::<i32>();
    (0..colors.len()).min_by_key(|&i| d(&colors[i])).unwrap()
}

fn srgb_bytes(c: [f32; 3]) -> [u8; 3] {
    c.map(|v| ((if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }) * 255.0).round() as u8)
}

/// Per screen row in `rows`, the first column from `start` rightwards
/// whose pixel is closer to `to` than to `from`.
fn switch_columns(pixels: &[u8], rows: std::ops::Range<u32>, start: u32, from: [u8; 3], to: [u8; 3]) -> Vec<u32> {
    rows.filter_map(|v| (start..320u32).find(|&u| nearest(&pixels[((v * 320 + u) * 4) as usize..], &[from, to]) == 1)).collect()
}

fn spread(columns: &[u32]) -> u32 {
    columns.iter().max().unwrap() - columns.iter().min().unwrap()
}

/// The tileset's two-level blending, measured in its weights view from
/// straight above. Material boundary: grass cells left of x = 32, sand
/// right of it; with the tileset the boundary meanders and frays (the
/// switch column varies across rows by many pixels) but stays within a
/// band of the cell boundary as wide as the warp plus the bilinear cell;
/// with the palette the same boundary is a straight line. Cliffs: a ramp
/// that steepens past the cliff threshold between two flats shows the
/// cliff layer where it is steep and grass on the flats, with an irregular
/// edge.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn tileset_boundaries_are_irregular_and_bounded() {
    use ae_render::terrain::tileset;
    let gpu = gpu();
    let t = Arc::new(tileset::temperate(64));
    // LAYER_COLORS in terrain.frag: grass (layer 0), sand (2), rock (3).
    let (grass, sand, rock) = (srgb_bytes([0.10, 0.75, 0.10]), srgb_bytes([0.95, 0.85, 0.25]), srgb_bytes([0.55, 0.55, 0.60]));

    let split = |x: usize, _: usize| if x < 32 { 0 } else { 2 };
    // Zoomed in on the boundary: about 14 px per cell.
    let (pixels, project) = tileset_scene(&gpu, |_, _| 50.0, split, Some(t.clone()), true, 200.0, None);
    let rows = 4..236;
    let start = project(200.0, 256.0, 50.0).0.max(0.0) as u32;
    let cols = switch_columns(&pixels, rows.clone(), start, grass, sand);
    assert_eq!(cols.len(), rows.len(), "every row crosses from grass to sand");
    // The boundary is at x = 256 (between cell centres 31.5 and 32.5);
    // allowed: the warp (2 cells) plus half a cell each side, plus a
    // pixel.
    let reach = 8.0 * (t.warp.0 + 0.5);
    let (lo, hi) = (project(256.0 - reach, 256.0, 50.0).0 - 1.0, project(256.0 + reach, 256.0, 50.0).0 + 1.0);
    let (lo, hi) = (lo.min(hi), lo.max(hi));
    eprintln!("tileset boundary columns {}..={} over {} rows (band {lo:.0}..{hi:.0})", cols.iter().min().unwrap(), cols.iter().max().unwrap(), cols.len());
    // Row to row it frays (the height blend), not only bends.
    let jumps = cols.windows(2).filter(|w| w[0].abs_diff(w[1]) >= 2).count();
    eprintln!("{jumps} rows jump 2+ px from the row above");
    assert!(jumps >= 4, "{jumps} rows jump 2+ px from the row above");
    let cell_px = project(264.0, 256.0, 50.0).0 - project(256.0, 256.0, 50.0).0;
    assert!(spread(&cols) as f32 >= cell_px, "the boundary meanders over a cell ({cell_px:.1} px) or more: spread {}", spread(&cols));
    assert!(cols.iter().all(|&c| (lo..=hi).contains(&(c as f32))), "and stays in its band");

    // The palette: a straight boundary (grass and sand palette colours).
    let (pixels, _) = tileset_scene(&gpu, |_, _| 50.0, split, None, false, 200.0, None);
    let cols = switch_columns(&pixels, rows.clone(), start, srgb_bytes([0.23, 0.33, 0.12]), srgb_bytes([0.60, 0.54, 0.38]));
    eprintln!("palette boundary columns {}..={}", cols.iter().min().unwrap(), cols.iter().max().unwrap());
    assert!(spread(&cols) <= 1, "the palette's boundary is straight: spread {}", spread(&cols));

    // A ramp that steepens: flat at 0 for x < 16 cells, then h = (x −
    // 16)² / 4 m to x = 48 (slope 0 → 2), flat on top. The slope passes
    // the cliff thresholds (1 − n.y = 0.18 .. 0.42) between about x = 27
    // and x = 38, a band the detail tilt and the height blend can fray.
    let ramp = |x: usize, _: usize| (x.clamp(16, 48) - 16).pow(2) as f32 * 0.25;
    let (pixels, project) = tileset_scene(&gpu, ramp, |_, _| 0, Some(t.clone()), true, 620.0, None);
    let class = |x: f32, y: f32, h: f32| {
        let (u, v) = project(x, y, h);
        nearest(&pixels[((v as u32 * 320 + u as u32) * 4) as usize..], &[grass, rock])
    };
    for y in [160.0, 256.0, 352.0] {
        assert_eq!(class(64.0, y, 0.0), 0, "grass on the low flat");
        assert_eq!(class(360.0, y, 196.0), 1, "rock where the ramp is steep");
        assert_eq!(class(456.0, y, 256.0), 0, "grass on the top flat");
    }
    // Where the ramp turns to rock, the switch column varies along it.
    let (top, bottom) = (project(160.0, 96.0, 0.0).1 as u32, project(160.0, 416.0, 0.0).1 as u32);
    let cols = switch_columns(&pixels, top.min(bottom)..top.max(bottom), project(64.0, 256.0, 0.0).0 as u32, grass, rock);
    let cell_px = project(264.0, 256.0, 100.0).0 - project(256.0, 256.0, 100.0).0;
    eprintln!("cliff edge columns {}..={} over {} rows ({cell_px:.1} px per cell)", cols.iter().min().unwrap(), cols.iter().max().unwrap(), cols.len());
    assert!(spread(&cols) as f32 >= cell_px, "the cliff edge is irregular over a cell or more: spread {}", spread(&cols));
    assert_clean(&gpu);
}

/// A flat map (ground at 50, 1024 world units square), units from `rows`
/// `(x, y, radius, class)`, rendered offscreen for a few frames from
/// `camera`; returns `(rows uploaded, units the GPU cull kept)` and the
/// frame the units saw.
fn cull_on_gpu(
    gpu: &Arc<GpuDevice>,
    rows: &[(f32, f32, f32, u32)],
    capacity: u32,
    min_pixels: f32,
    camera: impl FnOnce(&mut ae_render::terrain::camera::Camera),
) -> ((u32, u32), ae_render::units::InstanceStream, ae_render::units::UnitFrame) {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::units::{InstanceStream, UnitClass, UnitFeed, UnitFrame, UnitsSetup};

    let heightmap = Heightmap { width: 129, height: 129, data: vec![0.5; 129 * 129] };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource { heightmap, pyramid, info, materials: None });
    let mut columns: [Vec<u32>; 8] = Default::default();
    for &(x, y, radius, class) in rows {
        for (a, word) in [x.to_bits(), y.to_bits(), x.to_bits(), y.to_bits(), 1.0f32.to_bits(), 0.0f32.to_bits(), radius.to_bits(), class].into_iter().enumerate() {
            columns[a].push(word);
        }
    }
    let stream = InstanceStream { tick: 1, published: std::time::Instant::now(), tick_seconds: 0.0, columns };
    let feed = Arc::new(UnitFeed::default());
    feed.publish(Arc::new(stream.clone()));
    let classes = vec![UnitClass { color: [1.0, 0.0, 0.0], floats: false }];
    let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(UnitsSetup { feed, capacity, classes, min_pixels });
    let cam = {
        let view = app.view();
        let mut view = view.lock().unwrap();
        camera(&mut view.camera);
        view.camera
    };
    let extent = vk::Extent2D { width: 320, height: 200 };
    app.create_offscreen(gpu, extent).unwrap();
    let target = TargetInfo { extent, format: vk::Format::R8G8B8A8_SRGB };
    let config = ExecutorConfig { frames_in_flight: 2, profile: false, bake: BakeOptions::default(), record_threads: 1 };
    run_headless(gpu, &mut app, target, 4, config).unwrap();
    let counts = app.last_unit_counts().unwrap();
    ae_render::thread::RenderApp::destroy(&mut app, gpu);
    let view = lod_view(&source, &cam, extent.width, extent.height);
    let frame = UnitFrame { view_proj: view.view_proj, cam_pos: view.pos, proj_scale: view.proj_scale, fog: [0.0; 4], shadows: false };
    (counts, stream, frame)
}

/// The GPU cull keeps exactly the units the CPU copy of its test keeps:
/// 3,000 units scattered around and behind an oblique camera, some far
/// away and tiny. Units within half a world unit of a plane (where
/// rounding decides) are left out of the set. The set covers what the
/// cull must get right: units culled by each side plane, units whose
/// centre is outside the view but whose body reaches into it (kept), and
/// far units kept only because they are drawn at the minimum pixel size.
/// Then a stream longer than the capacity: only `capacity` rows are
/// uploaded and culled.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn units_cull_matches_the_cpu_reference() {
    use ae_render::units::cull_margins;
    const MIN_PIXELS: f32 = 7.0;
    let gpu = gpu();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut random = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32
    };
    let camera = |c: &mut ae_render::terrain::camera::Camera| {
        c.target = glam::Vec2::new(512.0, 512.0);
        c.distance = 450.0;
        c.yaw = 0.6;
        c.pitch = 0.9;
    };
    let ground = |_: f32, _: f32, _: u32| 50.0;
    // Candidates near the view, then far small ones along the view
    // direction; keep only those clear of every plane by half a unit.
    let mut candidates: Vec<(f32, f32, f32, u32)> = (0..6000).map(|_| (-400.0 + 1800.0 * random(), -400.0 + 1800.0 * random(), 1.0 + 11.0 * random(), 0)).collect();
    candidates.extend((0..1500).map(|_| (-6000.0 + 12000.0 * random(), -6000.0 + 12000.0 * random(), 0.3 + 0.7 * random(), 0)));
    // The frame does not depend on the units: probe it with one.
    let (_, _, frame) = cull_on_gpu(&gpu, &candidates[..1], 1, MIN_PIXELS, camera);
    let probe = ae_render::units::InstanceStream {
        tick: 0,
        published: std::time::Instant::now(),
        tick_seconds: 0.0,
        columns: std::array::from_fn(|a| candidates.iter().map(|&(x, y, r, c)| [x.to_bits(), y.to_bits(), x.to_bits(), y.to_bits(), 1.0f32.to_bits(), 0, r.to_bits(), c][a]).collect()),
    };
    let margins = cull_margins(&probe, &frame, MIN_PIXELS, 1.0, ground);
    let rows: Vec<(f32, f32, f32, u32)> = candidates.iter().zip(&margins).filter(|(_, m)| m.0.abs() > 0.5).map(|(c, _)| *c).take(3000).collect();
    assert_eq!(rows.len(), 3000);

    let ((uploaded, visible), stream, frame2) = cull_on_gpu(&gpu, &rows, 4096, MIN_PIXELS, camera);
    assert_eq!(frame2.view_proj, frame.view_proj, "same camera, same frame");
    let margins = cull_margins(&stream, &frame, MIN_PIXELS, 1.0, ground);
    let unenlarged = cull_margins(&stream, &frame, 0.0, 1.0, ground);
    let kept = margins.iter().filter(|m| m.0 >= 0.0).count() as u32;
    let mut culled_by = [0usize; 6];
    for m in margins.iter().filter(|m| m.0 < 0.0) {
        culled_by[m.1] += 1;
    }
    let straddling = margins.iter().filter(|m| m.0 >= 0.0 && m.0 < m.2).count();
    let by_min_size = margins.iter().zip(&unenlarged).filter(|(m, u)| m.0 >= 0.0 && u.0 < 0.0).count();
    eprintln!("{uploaded} rows, GPU kept {visible}, CPU keeps {kept}; culled by plane {culled_by:?}; {straddling} kept with the centre outside; {by_min_size} kept only at the minimum size");
    assert_eq!(uploaded, 3000);
    assert_eq!(visible, kept, "GPU and CPU cull agree");
    assert!(kept > 300 && kept < 2700, "a mix of kept and culled units: {kept}");
    assert!(culled_by[..4].iter().all(|&n| n > 20), "every side plane culls some: {culled_by:?}");
    assert!(straddling > 10, "units across the view's edge are kept: {straddling}");
    assert!(by_min_size > 5, "far units kept by the minimum size: {by_min_size}");

    // More rows than capacity: the first 64 are uploaded; all in view.
    let crowd: Vec<(f32, f32, f32, u32)> = (0..100).map(|i| (500.0 + (i % 10) as f32 * 4.0, 500.0 + (i / 10) as f32 * 4.0, 1.0, 0)).collect();
    let ((uploaded, visible), _, _) = cull_on_gpu(&gpu, &crowd, 64, MIN_PIXELS, camera);
    assert_eq!((uploaded, visible), (64, 64), "capacity caps the rows");
    assert_clean(&gpu);
}

/// The default render module drives the same scene as the native graph:
/// terrain and units, offscreen, 16 frames; the pixels are byte for byte
/// the same, and the module is reported active with no errors.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn the_default_module_renders_like_the_native_graph() {
    let gpu = gpu();
    let wasm = ae_render::module::build("default").expect("build render/default");
    let native = scene_pixels(&gpu, None, 16);
    let from_module = scene_pixels(&gpu, Some(wasm), 16);
    assert_eq!(native.0.len(), from_module.0.len());
    let differing = native.0.iter().zip(&from_module.0).filter(|(a, b)| a != b).count();
    assert_eq!(differing, 0, "{differing} bytes differ");
    assert_eq!(native.1, from_module.1, "same unit counts");
    assert!(native.1.1 > 0, "units were drawn: {:?}", native.1);
    assert_clean(&gpu);
}

/// A generated map with a few hundred units, offscreen at 320x200,
/// natively or driven by `module`.
fn scene_app(gpu: &Arc<GpuDevice>, module: Option<Vec<u8>>) -> ae_render::terrain::app::TerrainApp {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{build_pyramid, generate};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::units::{InstanceStream, UnitClass, UnitFeed, UnitsSetup};

    let heightmap = generate(257, 257, 3);
    let pyramid = build_pyramid(&heightmap, 300.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 300.0, 0.0, Some(40.0));
    let source = Arc::new(TerrainSource { heightmap, pyramid, info, materials: None });
    let mut columns: [Vec<u32>; 8] = Default::default();
    for i in 0..400u32 {
        let (x, y) = (300.0 + (i % 20) as f32 * 70.0, 300.0 + (i / 20) as f32 * 70.0);
        for (a, word) in [x.to_bits(), y.to_bits(), x.to_bits(), y.to_bits(), 1.0f32.to_bits(), 0.5f32.to_bits(), 6.0f32.to_bits(), i % 3].into_iter().enumerate() {
            columns[a].push(word);
        }
    }
    let feed = Arc::new(UnitFeed::default());
    feed.publish(Arc::new(InstanceStream { tick: 1, published: std::time::Instant::now(), tick_seconds: 0.0, columns }));
    let classes = vec![UnitClass { color: [1.0, 0.8, 0.1], floats: false }, UnitClass { color: [0.9, 0.9, 0.95], floats: true }, UnitClass { color: [0.8, 0.1, 0.1], floats: false }];
    let mut app = TerrainApp::new(source, TerrainOutput::Readback).with_units(UnitsSetup { feed, capacity: 512, classes, min_pixels: 5.0 });
    if let Some(wasm) = module {
        app = app.with_module(wasm);
    }
    {
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera.pitch = 0.6;
        view.camera.distance = 1600.0;
    }
    app.create_offscreen(gpu, SCENE_EXTENT).unwrap();
    app
}

const SCENE_EXTENT: vk::Extent2D = vk::Extent2D { width: 320, height: 200 };

/// The scene of [`scene_app`] rendered for `frames` frames: the pixels and
/// the unit counts (rows, visible).
fn scene_pixels(gpu: &Arc<GpuDevice>, module: Option<Vec<u8>>, frames: usize) -> (Vec<u8>, (u32, u32)) {
    use ae_render::thread::{TargetInfo, run_headless};
    let mut app = scene_app(gpu, module);
    let target = TargetInfo { extent: SCENE_EXTENT, format: vk::Format::R8G8B8A8_SRGB };
    let config = ExecutorConfig { frames_in_flight: 2, profile: false, bake: BakeOptions::default(), record_threads: 1 };
    run_headless(gpu, &mut app, target, frames, config).unwrap();
    let status = app.module_status().lock().unwrap().clone();
    assert!(status.errors.is_empty(), "{:?}", status.errors);
    assert_eq!(status.active, app.module_status().lock().unwrap().loads > 0);
    let pixels = app.readback().unwrap();
    let counts = app.last_unit_counts().unwrap();
    ae_render::thread::RenderApp::destroy(&mut app, gpu);
    (pixels, counts)
}

/// Mean of the sRGB-encoded RGB values over a `w`x`h` box at (x, y).
fn mean_rgb(pixels: &[u8], x: u32, y: u32, w: u32, h: u32) -> f64 {
    let mut sum = 0u64;
    for row in y..y + h {
        for col in x..x + w {
            let i = ((row * SCENE_EXTENT.width + col) * 4) as usize;
            sum += pixels[i] as u64 + pixels[i + 1] as u64 + pixels[i + 2] as u64;
        }
    }
    sum as f64 / (3 * w * h) as f64
}

/// A WAT module that declares `graph` and `programs` (bytes for the test
/// scene's target) and traps in `frame`.
fn trapping_module(graph: &[u8], programs: &[u8]) -> Vec<u8> {
    let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("\\{b:02x}")).collect::<String>();
    let (g, p) = (4096u32, 4096 + graph.len() as u32);
    let words: Vec<u8> = [g, graph.len() as u32, p, programs.len() as u32].iter().flat_map(|w| w.to_le_bytes()).collect();
    format!(
        r#"(module
            (memory (export "memory") 2)
            (data (i32.const 64) "{}")
            (data (i32.const {g}) "{}")
            (data (i32.const {p}) "{}")
            (func (export "ae_render_version") (result i32) i32.const 1)
            (func (export "ae_render_declare") (param i32 i32 i32 i32) (result i32) i32.const 64)
            (func (export "ae_render_input") (result i32) i32.const 1024)
            (func (export "ae_render_frame") (result i32) unreachable))"#,
        hex(&words),
        hex(graph),
        hex(programs)
    )
    .into_bytes()
}

/// Render modules replaced while running, in one executor:
/// default → post (another topology: one rebake; the grade changes the
/// picture) → the post module's vignette switched on by a host flag (a
/// conditional pass: no rebake; corners darken, the centre stays) →
/// garbage handed over (rejected, the post module keeps rendering the same
/// picture) → a graph running one fullscreen program in two passes
/// (rejected by name, likewise) → a module that declares the default graph but traps in
/// `frame` (installed, then stopped: its graph keeps rendering, like the
/// default module's). Validation stays silent throughout.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn render_modules_reload_while_running() {
    use ae_render::thread::{Headless, TargetInfo};
    let gpu = gpu();
    let default = ae_render::module::build("default").expect("build render/default");
    let post = ae_render::module::build("post").expect("build render/post");
    let mut app = scene_app(&gpu, Some(default.clone()));
    let target = TargetInfo { extent: SCENE_EXTENT, format: vk::Format::R8G8B8A8_SRGB };
    let config = ExecutorConfig { frames_in_flight: 2, profile: false, bake: BakeOptions::default(), record_threads: 1 };
    let mut run = Headless::new(&gpu, &mut app, target, config).unwrap();
    let status = app.module_status();
    let reload = app.module_reload();
    let mut step = |app: &mut ae_render::terrain::app::TerrainApp, frames: usize| -> (Vec<u8>, u64) {
        run.frames(app, frames).unwrap();
        run.wait().unwrap();
        (app.readback().unwrap(), run.stats().bakes)
    };
    let (w, h) = (SCENE_EXTENT.width, SCENE_EXTENT.height);
    let corners = |p: &[u8]| (mean_rgb(p, 0, 0, 24, 24) + mean_rgb(p, w - 24, 0, 24, 24) + mean_rgb(p, 0, h - 24, 24, 24) + mean_rgb(p, w - 24, h - 24, 24, 24)) / 4.0;
    let centre = |p: &[u8]| mean_rgb(p, w / 2 - 20, h / 2 - 20, 40, 40);

    let (plain, bakes) = step(&mut app, 12);
    assert_eq!((bakes, status.lock().unwrap().loads), (1, 1));

    *reload.lock().unwrap() = Some(post.clone());
    let (graded, bakes) = step(&mut app, 12);
    assert_eq!((bakes, status.lock().unwrap().loads), (2, 2), "the post module's graph rebaked once");
    let changed = plain.iter().zip(&graded).filter(|(a, b)| a.abs_diff(**b) > 2).count();
    assert!(changed > plain.len() / 4, "the grade changes the picture: {changed} of {} bytes", plain.len());
    assert!((centre(&graded) - centre(&plain)).abs() < 40.0, "but keeps it recognisable");

    app.view().lock().unwrap().module_flags = 1;
    let (vignetted, bakes) = step(&mut app, 4);
    assert_eq!(bakes, 2, "switching a conditional pass does not rebake");
    eprintln!("corners {:.1} -> {:.1}, centre {:.1} -> {:.1}", corners(&graded), corners(&vignetted), centre(&graded), centre(&vignetted));
    assert!(corners(&vignetted) < corners(&graded) * 0.6, "the vignette darkens the corners");
    assert!((centre(&vignetted) - centre(&graded)).abs() < 2.0, "and leaves the centre");

    *reload.lock().unwrap() = Some(b"not a module".to_vec());
    let (kept, bakes) = step(&mut app, 4);
    {
        let s = status.lock().unwrap();
        assert_eq!((s.loads, s.errors.len()), (2, 1));
        assert!(s.errors[0].contains("reload rejected"), "{}", s.errors[0]);
    }
    assert_eq!(bakes, 2);
    let drift = kept.iter().zip(&vignetted).filter(|(a, b)| a.abs_diff(**b) > 2).count();
    assert!(drift < kept.len() / 100, "the post module keeps rendering: {drift} bytes moved");

    // A graph running one fullscreen program in two passes (the grade of
    // the post module, scene -> mid -> target): rejected by name, the post
    // module keeps rendering.
    let shared = {
        use ae_render_abi::module::{Program, ProgramTable};
        use ae_render_abi::{Class, GraphBuilder, ImageSpec, LoadOp, PassKind, Role, ShaderStage};
        let engine = ae_render::module::engine();
        let mut m = ae_render::module::RenderModule::load(&engine, &post).unwrap();
        let flags = ae_render_abi::module::DECLARE_UNITS | ae_render_abi::module::DECLARE_READBACK;
        let declared = m.declare(w, h, vk::Format::R8G8B8A8_SRGB.as_raw() as u32, flags).unwrap().unwrap();
        let tables = ProgramTable::decode(&declared.programs).unwrap();
        let grade = tables.iter().find_map(|p| match *p {
            Program::Fullscreen { label: "grade", spirv, .. } => Some(spirv.to_vec()),
            _ => None,
        });
        let srgb = vk::Format::R8G8B8A8_SRGB.as_raw() as u32;
        let (mut g, mut p) = (GraphBuilder::new(), ProgramTable::new());
        let leaves = g.buffer("terrain.leaves", 0, 0, Class::External);
        let indirect = g.buffer("terrain.indirect", 0, 0, Class::External);
        let depth = g.image("depth", ImageSpec::depth_2d(vk::Format::D32_SFLOAT.as_raw() as u32, w, h, 0x20), Class::Transient);
        let scene = g.image("scene", ImageSpec::color_2d(srgb, w, h, 0x10 | 0x4), Class::Transient);
        let mid = g.image("mid", ImageSpec::color_2d(srgb, w, h, 0x10 | 0x4), Class::Transient);
        let target = g.image("readback.target", ImageSpec::color_2d(srgb, w, h, 0x10 | 0x1), Class::External);
        let buffer = g.buffer("readback.buffer", 0, 0, Class::External);
        let lod = p.library("terrain.lod");
        g.pass("terrain lod", PassKind::Compute, lod).access(leaves, Role::StorageBufferWrite(ShaderStage::Compute)).access(indirect, Role::StorageBufferWrite(ShaderStage::Compute)).id();
        let draw = p.library("terrain.draw");
        g.pass("terrain", PassKind::Graphics, draw).access(leaves, Role::StorageBufferRead(ShaderStage::Vertex)).access(indirect, Role::IndirectBuffer).color(scene, LoadOp::Clear).depth(depth, LoadOp::Clear).id();
        let program = p.push(Program::Fullscreen { label: "grade", spirv: &grade.unwrap(), uniform_bytes: 32 });
        g.pass("grade once", PassKind::Graphics, program).sampled(scene, ShaderStage::Fragment).color(mid, LoadOp::DontCare).id();
        g.pass("grade twice", PassKind::Graphics, program).sampled(mid, ShaderStage::Fragment).color(target, LoadOp::DontCare).id();
        let copy = p.library("readback.copy");
        g.pass("copy", PassKind::Transfer, copy).access(target, Role::TransferSrc).discard(buffer, Role::TransferDst).id();
        let read = p.library("host.read");
        g.pass("host read", PassKind::Transfer, read).access(buffer, Role::HostRead).root().id();
        trapping_module(&g.finish(), &p.finish())
    };
    *reload.lock().unwrap() = Some(shared);
    let (still, bakes) = step(&mut app, 4);
    {
        let s = status.lock().unwrap();
        assert_eq!((s.loads, s.errors.len()), (2, 2), "{:?}", s.errors);
        assert!(s.errors[1].contains("`grade once` and `grade twice`"), "{}", s.errors[1]);
    }
    assert_eq!(bakes, 2);
    let drift = still.iter().zip(&vignetted).filter(|(a, b)| a.abs_diff(**b) > 2).count();
    assert!(drift < still.len() / 100, "the post module keeps rendering: {drift} bytes moved");

    // The default module's declaration for this target, served by a module
    // that traps every frame.
    let declared = {
        let engine = ae_render::module::engine();
        let mut m = ae_render::module::RenderModule::load(&engine, &default).unwrap();
        let flags = ae_render_abi::module::DECLARE_UNITS | ae_render_abi::module::DECLARE_READBACK;
        m.declare(w, h, vk::Format::R8G8B8A8_SRGB.as_raw() as u32, flags).unwrap().unwrap()
    };
    *reload.lock().unwrap() = Some(trapping_module(&declared.graph, &declared.programs));
    let (trapped, bakes) = step(&mut app, 8);
    {
        let s = status.lock().unwrap();
        assert_eq!((s.loads, s.errors.len()), (3, 3), "{:?}", s.errors);
        assert!(s.errors[2].contains("stopped"), "{}", s.errors[2]);
    }
    assert_eq!(bakes, 3, "back to the default topology");
    let off = trapped.iter().zip(&plain).filter(|(a, b)| a.abs_diff(**b) > 2).count();
    assert!(off < trapped.len() / 100, "its graph renders like the default module's: {off} bytes differ");

    ae_render::thread::RenderApp::destroy(&mut app, &gpu);
    drop(run);
    assert_clean(&gpu);
}

/// Many graphics passes with many draws each, recorded on one thread and
/// on four (one command buffer per pass): the same image, every frame on
/// the threads, and the recording time of each (printed; not asserted,
/// since it depends on the machine). Later passes overwrite earlier ones,
/// so a wrong submission order changes pixels.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn parallel_recording_matches_serial() {
    use ae_framegraph::{FrameGraph, LoadOp, PassKind, ResourceId, Role};
    use ae_render::executor::{PassContext, Recorder, SharedRecorder};
    use ae_render::pipeline::{self, Blend, GraphicsDesc};

    const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
    const SIZE: u32 = 64;
    const PASSES: u16 = 24;
    const DRAWS: usize = 3000;
    const FRAMES: usize = 6;
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct Rect {
        rect: [f32; 4],
        color: [f32; 4],
        depth: [f32; 4],
    }
    struct Stress {
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        rects: Vec<Vec<Rect>>,
        target: ResourceId,
        readback: ResourceId,
    }
    impl Recorder<u16> for Stress {
        fn record(&mut self, pass: &u16, ctx: &PassContext<'_>) -> anyhow::Result<()> {
            SharedRecorder::record(self, pass, ctx)
        }

        fn shared(&self) -> Option<Box<dyn SharedRecorder<u16> + '_>> {
            Some(Box::new(self))
        }
    }
    impl SharedRecorder<u16> for &Stress {
        fn record(&self, pass: &u16, ctx: &PassContext<'_>) -> anyhow::Result<()> {
            SharedRecorder::record(*self, pass, ctx)
        }
    }
    impl SharedRecorder<u16> for Stress {
        fn record(&self, pass: &u16, ctx: &PassContext<'_>) -> anyhow::Result<()> {
            let d = ctx.device;
            if let Some(rects) = self.rects.get(*pass as usize) {
                unsafe {
                    d.cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                    for r in rects {
                        d.cmd_push_constants(ctx.cmd, self.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pipeline::push_bytes(r));
                        d.cmd_draw(ctx.cmd, 6, 1, 0, 0);
                    }
                }
            } else if *pass == PASSES {
                let (image, buffer) = (ctx.image(self.target)?, ctx.buffer(self.readback)?);
                let region = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                    .image_extent(vk::Extent3D { width: SIZE, height: SIZE, depth: 1 });
                unsafe { d.cmd_copy_image_to_buffer(ctx.cmd, image.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buffer.buffer, &[region]) };
            }
            Ok(())
        }
    }

    // Deterministic rects: small squares, later passes on top.
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32
    };
    let rects: Vec<Vec<Rect>> = (0..PASSES)
        .map(|_| {
            (0..DRAWS)
                .map(|_| {
                    let (x, y, w) = (next() * 2.0 - 1.0, next() * 2.0 - 1.0, 0.02 + next() * 0.2);
                    Rect { rect: [x, y, x + w, y + w], color: [next(), next(), next(), 1.0], depth: [0.5, 0.0, 0.0, 0.0] }
                })
                .collect()
        })
        .collect();

    // Pixels of the last frame and the median recording time.
    let run = |gpu: &Arc<GpuDevice>, threads: usize| -> (Vec<u8>, f32) {
        let device = &gpu.device;
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let layout = pipeline::pipeline_layout(device, &[], std::mem::size_of::<Rect>() as u32, stages).unwrap();
        let desc = GraphicsDesc {
            vertex: include_bytes!(concat!(env!("OUT_DIR"), "/test_rect.vert.spv")),
            fragment: include_bytes!(concat!(env!("OUT_DIR"), "/test_flat.frag.spv")),
            layout,
            color_formats: &[FORMAT],
            depth: None,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            blend: Blend::Off,
        };
        let pipe = pipeline::graphics_pipeline(device, &desc).unwrap();
        let color = ImageDesc::color_2d(FORMAT, SIZE, SIZE, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC);
        let bytes = (SIZE * SIZE * 4) as u64;
        let buffer_desc = BufferDesc { size: bytes, usage: vk::BufferUsageFlags::TRANSFER_DST };
        let mut image = OwnedImage::new(gpu, color, "stress target").unwrap();
        let mut buffer = OwnedBuffer::new(gpu, buffer_desc, MemoryLocation::GpuToCpu, "stress readback").unwrap();
        let extent = vk::Extent2D { width: SIZE, height: SIZE };
        let bindings = [
            Binding::Image(ImageBinding { image: image.image, view: image.view, format: FORMAT, extent, range: image.range(), identity: 1 << 48 }),
            Binding::Buffer(BufferBinding { buffer: buffer.buffer, offset: 0, size: bytes, identity: (1 << 48) + 1 }),
        ];
        let mut graph = FrameGraph::<u16>::new();
        let target = graph.image("target", color, ResourceClass::External);
        let readback = graph.buffer("readback", buffer_desc, ResourceClass::External);
        for p in 0..PASSES {
            graph.pass(&format!("draw {p}"), PassKind::Graphics, p).color(target, if p == 0 { LoadOp::Clear } else { LoadOp::Load }).id();
        }
        graph.pass("copy", PassKind::Transfer, PASSES).access(target, Role::TransferSrc).discard(readback, Role::TransferDst).id();
        graph.pass("host read", PassKind::Transfer, PASSES + 1).access(readback, Role::HostRead).root().id();
        let mut executor = Executor::new(gpu.clone(), ExecutorConfig { record_threads: threads, ..ExecutorConfig::default() }).unwrap();
        executor.set_graph(graph).unwrap();
        let mut recorder = Stress { pipeline: pipe, layout, rects: rects.clone(), target, readback };
        let mut record_ms = Vec::new();
        for _ in 0..FRAMES {
            let token = executor.begin_frame().unwrap();
            executor.execute(token, &mut recorder, &|resource| bindings.get(resource.index()).copied(), &[], &[]).unwrap();
            executor.wait(token).unwrap();
            record_ms.push(executor.stats().record_ms);
        }
        let pixels = buffer.mapped().unwrap().to_vec();
        assert_eq!(executor.stats().parallel_frames, if threads > 1 { FRAMES as u64 } else { 0 }, "{threads} threads");
        drop(executor);
        unsafe {
            device.destroy_pipeline(pipe, None);
            device.destroy_pipeline_layout(layout, None);
        }
        image.destroy(gpu);
        buffer.destroy(gpu);
        record_ms.sort_by(f32::total_cmp);
        (pixels, record_ms[FRAMES / 2])
    };

    // Correctness under validation (with synchronization validation).
    let gpu = gpu();
    let (serial, _) = run(&gpu, 1);
    let (parallel, _) = run(&gpu, 4);
    let distinct = serial.chunks(4).collect::<std::collections::BTreeSet<_>>().len();
    assert!(distinct > 50, "the passes drew something ({distinct} colours)");
    assert!(serial == parallel, "parallel recording changed {} bytes", serial.iter().zip(&parallel).filter(|(a, b)| a != b).count());
    assert_clean(&gpu);
    drop(gpu);

    // Recording time without the validation layer (it dominates otherwise).
    let instance = Arc::new(GpuInstance::new(&[], Some(false)).expect("Vulkan instance"));
    let plain = Arc::new(GpuDevice::new(instance, None).expect("Vulkan device"));
    for threads in [1, 2, 4, 8] {
        let (pixels, ms) = run(&plain, threads);
        assert!(pixels == serial, "{threads} threads without validation: other pixels");
        println!("{threads} recording thread(s): {PASSES} passes x {DRAWS} draws recorded in {ms:.2} ms (median of {FRAMES} frames, no validation)");
    }
}

/// Fog of war on the terrain (`TerrainApp::with_vision`), from straight
/// above over a flat 512 m map with 32 m vision cells:
/// * every cell in line of sight draws exactly what no vision draws;
/// * line of sight on the left half only: the right half (never explored)
///   is much darker than the left, and the left half matches the
///   no-vision image there; explored-but-not-seen cells sit in between;
/// * before the first vision frame everything reads as unexplored.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn vision_shades_the_terrain() {
    use ae_render::terrain::app::{VisionFeed, VisionFrame};
    use ae_render::terrain::renderer::VisionGrid;
    let gpu = gpu();
    let grid = VisionGrid { width: 16, height: 16, cell_size: 32.0 };
    let plane = |f: &dyn Fn(u32, u32) -> bool| -> Vec<u64> {
        let mut words = vec![0u64; grid.words()];
        for y in 0..grid.height {
            for x in 0..grid.width {
                if f(x, y) {
                    words[y as usize * grid.stride() + (x / 64) as usize] |= 1 << (x % 64);
                }
            }
        }
        words
    };
    let scene = |feed: Option<Arc<VisionFeed>>| tileset_scene(&gpu, |_, _| 50.0, |_, _| 0, None, false, 900.0, feed).0;
    let feed = |los: &dyn Fn(u32, u32) -> bool, explored: &dyn Fn(u32, u32) -> bool| {
        let f = Arc::new(VisionFeed::new(grid));
        f.publish(Arc::new(VisionFrame { tick: 1, los: plane(los), explored: plane(explored) }));
        Some(f)
    };
    let plain = scene(None);
    let all = scene(feed(&|_, _| true, &|_, _| true));
    assert!(plain == all, "full line of sight changed {} bytes", plain.iter().zip(&all).filter(|(a, b)| a != b).count());

    // Left half seen, right half: top explored, bottom never.
    let half = scene(feed(&|x, _| x < 8, &|x, y| x < 8 || y < 8));
    let luma = |p: &[u8], x0: u32, x1: u32, y0: u32, y1: u32| {
        let mut sum = 0.0;
        for y in y0..y1 {
            for x in x0..x1 {
                let i = ((y * 320 + x) * 4) as usize;
                sum += 0.3 * p[i] as f64 + 0.59 * p[i + 1] as f64 + 0.11 * p[i + 2] as f64;
            }
        }
        sum / ((x1 - x0) * (y1 - y0)) as f64
    };
    // The map spans the middle of the frame from above; sample well inside
    // each quarter (screen y grows downwards, world y with it here or not:
    // both right quarters are compared to the left).
    let (left, right_a, right_b) = (luma(&half, 70, 140, 40, 200), luma(&half, 190, 250, 40, 100), luma(&half, 190, 250, 140, 200));
    let (lit_left, plain_left) = (left, luma(&plain, 70, 140, 40, 200));
    let (dim, dark) = (right_a.max(right_b), right_a.min(right_b));
    println!("luma: left {left:.1} (no vision {plain_left:.1}), right explored {dim:.1}, never seen {dark:.1}");
    assert!((lit_left - plain_left).abs() < 1.0, "line of sight draws as without vision");
    // sRGB bytes: the shader's 0.2 and 0.5 factors (plus desaturation)
    // read as about 0.5 and 0.75 of the brightness.
    assert!(lit_left > 1.5 * dark, "never-seen ground is much darker");
    assert!(dim > dark * 1.2 && dim < lit_left * 0.9, "explored ground sits between");

    // A feed with no frame yet: all unexplored.
    let empty = scene(Some(Arc::new(VisionFeed::new(grid))));
    let unseen = luma(&empty, 190, 250, 40, 100).min(luma(&empty, 190, 250, 140, 200));
    assert!((unseen - dark).abs() < 1.0, "before any vision frame, ground reads as never seen ({unseen:.1} vs {dark:.1})");
    assert!(luma(&empty, 70, 140, 40, 200) < 0.7 * lit_left, "and the left is not lit");
    assert_clean(&gpu);
}

/// The water surface (M7): a flat map, its left half 20 m above the water,
/// its right half 20 m under it with an island rising 35 m out of it, a
/// land unit on the left and a ship on the right, seen from 40° up.
/// * With water added but switched off, the picture is byte-identical to
///   the scene without water.
/// * With water on, every land point is unchanged, the island's top
///   included, though from this angle water lies behind its far edge (the
///   depth test keeps the surface behind the terrain), and nearly every
///   water point changes.
/// * The ship, above the plane, keeps its class colour.
/// * Out of sight (fog of war on the water half), what the surface adds
///   is dimmed with the ground under it.
/// * The waves move: half a second later only water points change.
///
/// Turning the water pipeline's depth test off fails the land check (the
/// island's far edge; over land itself the waterline fade leaves nothing to
/// see).
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn water_covers_the_low_ground_only() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, VisionFeed, VisionFrame, lod_view};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::{TerrainInfo, VisionGrid};
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::units::{InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Vec2, Vec4};

    let gpu = gpu();
    let n = 129;
    let island = |cx: usize, cz: usize| (88..=100).contains(&cx) && (88..=100).contains(&cz);
    let heightmap = Heightmap { width: n, height: n, data: (0..n * n).map(|i| if i % n < 64 { 0.8 } else if island(i % n, i / n) { 0.95 } else { 0.4 }).collect() };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, Some(60.0));
    let source = Arc::new(TerrainSource { heightmap, pyramid, info, materials: None });
    let extent = vk::Extent2D { width: 320, height: 240 };
    let grid = VisionGrid { width: 32, height: 32, cell_size: 32.0 };
    let land_unit = (250.0f32, 512.0f32);
    let ship = (760.0f32, 512.0f32);

    #[derive(Clone, Copy)]
    struct Run {
        water: bool,
        view_water: bool,
        vision: bool,
        frames: usize,
    }
    let mut camera = None;
    let mut render = |r: Run| -> Vec<u8> {
        let mut columns: [Vec<u32>; 8] = Default::default();
        for (&(x, y), class) in [land_unit, ship].iter().zip(0u32..) {
            for (a, word) in [x.to_bits(), y.to_bits(), x.to_bits(), y.to_bits(), 1.0f32.to_bits(), 0.0f32.to_bits(), 8.0f32.to_bits(), class].into_iter().enumerate() {
                columns[a].push(word);
            }
        }
        let feed = Arc::new(UnitFeed::default());
        feed.publish(Arc::new(InstanceStream { tick: 1, published: std::time::Instant::now(), tick_seconds: 0.0, columns }));
        let classes = vec![UnitClass { color: [1.0, 0.02, 0.02], floats: false }, UnitClass { color: [0.02, 0.02, 1.0], floats: true }];
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(UnitsSetup { feed, capacity: 16, classes, min_pixels: 6.0 });
        if r.water {
            app = app.with_water();
        }
        if r.vision {
            // Line of sight over the land half only; the water half never seen.
            let mut los = vec![0u64; grid.words()];
            for y in 0..grid.height as usize {
                los[y * grid.stride()] = (1u64 << 16) - 1;
            }
            let feed = Arc::new(VisionFeed::new(grid));
            feed.publish(Arc::new(VisionFrame { tick: 1, los, explored: vec![0; grid.words()] }));
            app = app.with_vision(feed);
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = Vec2::new(512.0, 512.0);
            view.camera.pitch = 0.7;
            view.camera.distance = 1100.0;
            view.fog = false;
            view.water = r.view_water;
            camera = Some(view.camera);
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo { extent, format: vk::Format::R8G8B8A8_SRGB };
        let config = ExecutorConfig { frames_in_flight: 2, profile: false, bake: BakeOptions::default(), record_threads: 1 };
        run_headless(&gpu, &mut app, target, r.frames, config).unwrap();
        let pixels = app.readback().unwrap();
        if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            let name = format!("water-{}{}{}-{}.ppm", r.water as u8, r.view_water as u8, r.vision as u8, r.frames);
            std::fs::write(std::path::Path::new(&dir).join(name), ppm).unwrap();
        }
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let base = Run { water: false, view_water: true, vision: false, frames: 12 };
    let plain = render(base);
    let switched_off = render(Run { water: true, view_water: false, ..base });
    let on = render(Run { water: true, ..base });
    let later = render(Run { water: true, frames: 42, ..base });
    let fogged_plain = render(Run { vision: true, ..base });
    let fogged = render(Run { water: true, vision: true, ..base });

    let view_proj = lod_view(&source, &camera.unwrap(), extent.width, extent.height).view_proj;
    let at = |x: f32, y: f32, h: f32| -> Option<usize> {
        let clip = view_proj * Vec4::new(x, h, y, 1.0);
        let (u, v) = ((clip.x / clip.w * 0.5 + 0.5) * extent.width as f32, (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32);
        (clip.w > 0.0 && u >= 2.0 && v >= 2.0 && u < extent.width as f32 - 2.0 && v < extent.height as f32 - 2.0).then(|| (v as usize * extent.width as usize + u as usize) * 4)
    };
    // Points on the land (clear of the cliff and the land unit) and on the
    // water (clear of the cliff's foot and the ship).
    let near = |x: f32, y: f32, (ux, uy): (f32, f32)| (x - ux).abs() < 40.0 && (y - uy).abs() < 40.0;
    let points = |xs: std::ops::RangeInclusive<u32>, ys: std::ops::RangeInclusive<u32>, step: usize, h: f32, unit: (f32, f32)| -> Vec<usize> {
        let mut out = Vec::new();
        for x in xs.step_by(step) {
            for y in ys.clone().step_by(step) {
                let (x, y) = (x as f32, y as f32);
                // Clear of the unit and of the island and its slopes.
                if !near(x, y, unit) && !((x - 752.0).abs() < 64.0 && (y - 752.0).abs() < 64.0) {
                    out.extend(at(x, y, h));
                }
            }
        }
        out
    };
    let mut land = points(16..=480, 16..=1008, 16, 80.0, land_unit);
    let mut top = Vec::new();
    for x in (712..=792).step_by(8) {
        for y in (712..=792).step_by(8) {
            top.extend(at(x as f32, y as f32, 95.0));
        }
    }
    assert!(top.len() > 60, "{} island points in view", top.len());
    land.extend(top);
    let water = points(560..=1008, 16..=1008, 16, 60.0, ship);
    assert!(land.len() > 300 && water.len() > 300, "{} land and {} water points in view", land.len(), water.len());
    let rgb = |p: &[u8], i: usize| [p[i], p[i + 1], p[i + 2]];
    let changed = |a: &[u8], b: &[u8], at: &[usize]| at.iter().filter(|&&i| rgb(a, i) != rgb(b, i)).count();
    let mean_diff = |a: &[u8], b: &[u8], at: &[usize]| at.iter().map(|&i| (0..3).map(|c| (a[i + c] as f64 - b[i + c] as f64).abs()).sum::<f64>()).sum::<f64>() / at.len() as f64;

    let off_bytes = plain.iter().zip(&switched_off).filter(|(a, b)| a != b).count();
    assert_eq!(off_bytes, 0, "water switched off draws as no water");
    assert_eq!(changed(&plain, &on, &land), 0, "land points under the water surface");
    let wet = changed(&plain, &on, &water);
    println!("water: {wet} of {} water points changed; mean difference {:.1}", water.len(), mean_diff(&plain, &on, &water));
    assert!(wet * 10 >= water.len() * 9, "the surface covers the water: {wet} of {}", water.len());

    let ship_at = at(ship.0, ship.1, 61.0).expect("the ship is in view");
    let [r, g, b] = rgb(&on, ship_at).map(|c| c as i32);
    assert!(b > 2 * r && b > 2 * g, "the ship keeps its colour over the water: {r} {g} {b}");

    assert_eq!(changed(&fogged_plain, &fogged, &land), 0, "land in sight: unchanged by the water");
    let (lit, unseen) = (mean_diff(&plain, &on, &water), mean_diff(&fogged_plain, &fogged, &water));
    println!("what the surface adds: {lit:.1} in sight, {unseen:.1} never seen");
    assert!(unseen < 0.6 * lit, "out of sight the surface's share is dimmed: {unseen:.1} vs {lit:.1}");

    assert_eq!(changed(&on, &later, &land), 0, "half a second later the land is the same");
    let moved = changed(&on, &later, &water);
    assert!(moved * 5 >= water.len(), "half a second later the waves have moved: {moved} of {}", water.len());
    assert_clean(&gpu);
}

/// The hills' shadows (`TerrainApp::with_shadows`): on a flat map with two
/// towers, one in view and one whose footprint is off screen, the ground
/// the CPU heightfield ([`ae_render::terrain::sun`]) puts in shadow darkens,
/// the off-screen tower's shadow included; sunlit ground does not change;
/// a unit in a tower's shadow darkens and one in the sun does not; shadows
/// made but switched off draw exactly as none.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn hills_shade_away_from_the_sun() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::edit::TerrainEditor;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::terrain::sun;
    use ae_render::thread::{Headless, TargetInfo};
    use ae_render::units::{InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Vec2, Vec4};

    let gpu = gpu();
    let (n, cell, scale) = (129usize, 8.0f32, 100.0f32);
    // Corner rectangles of the towers (90 m over ground at 10 m).
    let towers = [(60usize, 60usize, 67usize, 67usize), (TOWER_B.0, TOWER_B.1, TOWER_B.2, TOWER_B.3)];
    let tall = |x: usize, y: usize| towers.iter().any(|&(x0, y0, x1, y1)| (x0..=x1).contains(&x) && (y0..=y1).contains(&y));
    let heightmap = Heightmap { width: n, height: n, data: (0..n * n).map(|i| if tall(i % n, i / n) { 1.0 } else { 0.1 }).collect() };
    let shadow = sun::shadow_heights(&heightmap, cell, scale, 0.0, sun::direction());
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, cell, scale, 0.0, None);
    let source = Arc::new(TerrainSource { heightmap, pyramid, info, materials: None });
    let extent = vk::Extent2D { width: 320, height: 240 };
    // A unit in the first tower's shadow, one in the sun.
    let shaded_unit = (590.0f32, 450.0f32);
    let sunny_unit = (430.0f32, 600.0f32);

    #[derive(Clone, Copy)]
    struct Run {
        shadows: bool,
        view_shadows: bool,
        /// Then level tower A by a terrain edit.
        level_a: bool,
    }
    let mut camera = None;
    let mut render = |r: Run| -> Vec<u8> {
        let mut columns: [Vec<u32>; 8] = Default::default();
        for &(x, y) in &[shaded_unit, sunny_unit] {
            for (a, word) in [x.to_bits(), y.to_bits(), x.to_bits(), y.to_bits(), 1.0f32.to_bits(), 0.0f32.to_bits(), 6.0f32.to_bits(), 0].into_iter().enumerate() {
                columns[a].push(word);
            }
        }
        let feed = Arc::new(UnitFeed::default());
        feed.publish(Arc::new(InstanceStream { tick: 1, published: std::time::Instant::now(), tick_seconds: 0.0, columns }));
        let classes = vec![UnitClass { color: [0.45, 0.45, 0.45], floats: false }];
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(UnitsSetup { feed, capacity: 16, classes, min_pixels: 6.0 });
        if r.shadows {
            app = app.with_shadows();
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = Vec2::new(CAMERA.0, CAMERA.1);
            view.camera.pitch = 0.75;
            view.camera.distance = CAMERA.2;
            view.fog = false;
            view.shadows = r.view_shadows;
            camera = Some(view.camera);
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo { extent, format: vk::Format::R8G8B8A8_SRGB };
        let config = ExecutorConfig { frames_in_flight: 2, profile: false, bake: BakeOptions::default(), record_threads: 1 };
        let mut run = Headless::new(&gpu, &mut app, target, config).unwrap();
        run.frames(&mut app, 12).unwrap();
        if r.level_a {
            // Cells 60..=66 hold tower A's corners (60..=67) only.
            let mut editor = TerrainEditor::new(&source);
            let update = editor.edit(2, &[(59, 59, 67, 67)], |x, y| if (TOWER_B.0..TOWER_B.2).contains(&x) && (TOWER_B.1..TOWER_B.3).contains(&y) { 100.0 } else { 10.0 }, |_, _| 0).unwrap();
            assert!(!update.shadows.is_empty(), "the edit rewrites shadow heights");
            app.terrain_edits().lock().unwrap().pending.push(update);
            run.frames(&mut app, 12).unwrap();
        }
        run.wait().unwrap();
        run.finish().unwrap();
        let pixels = app.readback().unwrap();
        if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            let name = format!("shadows-{}{}{}.ppm", r.shadows as u8, r.view_shadows as u8, r.level_a as u8);
            std::fs::write(std::path::Path::new(&dir).join(name), ppm).unwrap();
        }
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let plain = render(Run { shadows: false, view_shadows: true, level_a: false });
    let switched_off = render(Run { shadows: true, view_shadows: false, level_a: false });
    let on = render(Run { shadows: true, view_shadows: true, level_a: false });
    let leveled_plain = render(Run { shadows: false, view_shadows: true, level_a: true });
    let leveled = render(Run { shadows: true, view_shadows: true, level_a: true });

    let view_proj = lod_view(&source, &camera.unwrap(), extent.width, extent.height).view_proj;
    let at = |x: f32, y: f32, h: f32| -> Option<usize> {
        let clip = view_proj * Vec4::new(x, h, y, 1.0);
        let (u, v) = ((clip.x / clip.w * 0.5 + 0.5) * extent.width as f32, (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32);
        (clip.w > 0.0 && u >= 2.0 && v >= 2.0 && u < extent.width as f32 - 2.0 && v < extent.height as f32 - 2.0).then(|| (v as usize * extent.width as usize + u as usize) * 4)
    };
    // The off-screen tower: none of it in view.
    let (bx0, by0, bx1, by1) = towers[1];
    for x in bx0..=bx1 + 1 {
        for y in by0..=by1 + 1 {
            for h in [10.0, 100.0] {
                assert!(at(x as f32 * cell - 4.0, y as f32 * cell - 4.0, h).is_none(), "tower B at corner {x},{y} height {h} is in view");
            }
        }
    }
    // Flat ground at corner resolution, sorted by the CPU heightfield:
    // lit (no shadow within two corners) or deep in shadow (all four
    // corners of the cell at least 8 m under the shadow height), clear of
    // the towers and the units.
    let s = |x: usize, y: usize| shadow[y * n + x] - 10.0;
    let near_unit = |x: f32, y: f32| [shaded_unit, sunny_unit].iter().any(|&(ux, uy)| (x - ux).abs() < 24.0 && (y - uy).abs() < 24.0);
    let near_tower = |x: usize, y: usize| towers.iter().any(|&(x0, y0, x1, y1)| x + 3 >= x0 && x <= x1 + 3 && y + 3 >= y0 && y <= y1 + 3);
    let (mut lit, mut shaded, mut shaded_b) = (Vec::new(), Vec::new(), Vec::new());
    for y in 3..n - 4 {
        for x in 3..n - 4 {
            let (wx, wy) = (x as f32 * cell + 4.0, y as f32 * cell + 4.0);
            if near_tower(x, y) || near_unit(wx, wy) {
                continue;
            }
            let Some(i) = at(wx, wy, 10.0) else { continue };
            if (x - 2..=x + 3).all(|cx| (y - 2..=y + 3).all(|cy| s(cx, cy) == 0.0)) {
                lit.push(i);
            } else if [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)].iter().all(|&(cx, cy)| s(cx, cy) > 8.0) {
                // Whose shadow: B's lies at smaller x than A's footprint.
                if x < 60 { shaded_b.push(i) } else { shaded.push(i) }
            }
        }
    }
    println!("{} lit, {} shaded by tower A, {} by tower B (off screen)", lit.len(), shaded.len(), shaded_b.len());
    assert!(lit.len() > 300 && shaded.len() > 20 && shaded_b.len() > 20, "{} lit, {} shaded, {} shaded by B", lit.len(), shaded.len(), shaded_b.len());

    let rgb = |p: &[u8], i: usize| [p[i], p[i + 1], p[i + 2]];
    let luma = |p: &[u8], i: usize| p[i] as f32 * 0.3 + p[i + 1] as f32 * 0.59 + p[i + 2] as f32 * 0.11;
    let off_bytes = plain.iter().zip(&switched_off).filter(|(a, b)| a != b).count();
    assert_eq!(off_bytes, 0, "shadows switched off draw as none");
    let changed = lit.iter().filter(|&&i| rgb(&plain, i) != rgb(&on, i)).count();
    assert_eq!(changed, 0, "sunlit ground changes");
    for (name, points) in [("tower A", &shaded), ("tower B (off screen)", &shaded_b)] {
        let darker = points.iter().filter(|&&i| luma(&on, i) < 0.75 * luma(&plain, i)).count();
        println!("{name}: {darker} of {} shaded points a quarter darker or more", points.len());
        assert!(darker * 10 >= points.len() * 9, "{name}'s shadow: {darker} of {}", points.len());
    }
    // The units: their sunlit faces (the top of the dart, 3 m up).
    let unit_luma = |p: &[u8], (x, y): (f32, f32)| {
        let i = at(x, y, 13.0).expect("unit in view");
        luma(p, i)
    };
    let (sun_plain, sun_on) = (unit_luma(&plain, sunny_unit), unit_luma(&on, sunny_unit));
    let (shade_plain, shade_on) = (unit_luma(&plain, shaded_unit), unit_luma(&on, shaded_unit));
    println!("unit in the sun {sun_plain:.0} -> {sun_on:.0}, in the shadow {shade_plain:.0} -> {shade_on:.0}");
    assert_eq!(sun_plain, sun_on, "the unit in the sun is unchanged");
    assert!(shade_on < 0.8 * shade_plain, "the unit in the shadow darkens");

    // Tower A levelled by an edit: its shadow (and the unit's) goes, B's stays.
    let gone = shaded.iter().filter(|&&i| rgb(&leveled_plain, i) != rgb(&leveled, i)).count();
    assert_eq!(gone, 0, "tower A's shadow after levelling it");
    assert_eq!(unit_luma(&leveled_plain, shaded_unit), unit_luma(&leveled, shaded_unit), "the unit after tower A is levelled");
    let still = shaded_b.iter().filter(|&&i| luma(&leveled, i) < 0.75 * luma(&leveled_plain, i)).count();
    assert!(still * 10 >= shaded_b.len() * 9, "tower B's shadow after the edit: {still} of {}", shaded_b.len());
    assert_clean(&gpu);
}

/// [`hills_shade_away_from_the_sun`]: the off-screen tower (corner
/// rectangle) and the camera (target x, z, distance).
const TOWER_B: (usize, usize, usize, usize) = (27, 70, 32, 75);
const CAMERA: (f32, f32, f32) = (560.0, 520.0, 520.0);
