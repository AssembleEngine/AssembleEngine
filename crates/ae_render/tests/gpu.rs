//! Headless GPU tests (ignored by default: they need a Vulkan
//! 1.3 device). Run with
//! `cargo test -p ae_render --test gpu -- --ignored --test-threads=1`.
//! Every test runs with validation (including synchronization validation)
//! and fails on any validation error or warning.

use ae_framegraph::{BakeOptions, BufferDesc, FrameGraph, ImageDesc, PassKind, ResourceClass, ShaderStage};
use ae_render::demo::{DemoOutput, DemoRenderer, build_demo_graph};
use ae_render::device::{GpuDevice, GpuInstance};
use ae_render::executor::{Binding, BufferBinding, Executor, ExecutorConfig, ImageBinding};
use ae_render::memory::{OwnedBuffer, OwnedImage};
use ae_render::units::{ATTR_POSE, ATTR_VIEW, ATTR_WATER, ATTRIBUTE_COUNT};
use ash::vk;
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

fn gpu() -> Arc<GpuDevice> {
    let instance = Arc::new(GpuInstance::new(&[], Some(true)).expect("Vulkan instance"));
    assert!(instance.validation, "these tests need VK_LAYER_KHRONOS_validation");
    Arc::new(GpuDevice::new(instance, None).expect("Vulkan device"))
}

/// Fill omitted poses, water depths and view states with their defaults.
fn at_rest(mut columns: [Vec<u32>; ATTRIBUTE_COUNT]) -> [Vec<u32>; ATTRIBUTE_COUNT] {
    let n = columns[0].len();
    columns[ATTR_POSE].resize(n, ae_render::units::NO_POSE);
    columns[ATTR_WATER].resize(n, 0);
    columns[ATTR_VIEW].resize(n, 0);
    columns
}

/// The `Orientation` of a unit heading `(hx, hy)` (nav x, nav y; need not
/// be unit length; zero: the identity): the shortest turn from +x about up.
fn yaw(hx: f32, hy: f32) -> [f32; 4] {
    let len = (hx * hx + hy * hy).sqrt();
    if len == 0.0 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let (c, s) = (hx / len, hy / len);
    if c <= -1.0 + 1e-6 {
        return [0.0, 1.0, 0.0, 0.0];
    }
    let q = [0.0, -s, 0.0, 1.0 + c];
    let n = (q[1] * q[1] + q[3] * q[3]).sqrt();
    [0.0, q[1] / n, 0.0, q[3] / n]
}

/// A graph with one sampled image, for the bindless rebake test.
fn one_sampled_graph(extent: u32) -> FrameGraph<()> {
    let mut graph = FrameGraph::<()>::new();
    let tex = graph.image(
        "tex",
        ImageDesc::color_2d(vk::Format::R8G8B8A8_UNORM, extent, extent, vk::ImageUsageFlags::SAMPLED),
        ResourceClass::Persistent,
    );
    graph
        .pass("sample", PassKind::Compute, ())
        .sampled(tex, ShaderStage::Compute)
        .root();
    graph
}

#[test]
#[ignore]
fn graph_handles_survive_resize_and_rebake() {
    let gpu = gpu();
    let mut executor = Executor::<()>::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    executor.set_graph(one_sampled_graph(4)).unwrap();
    assert_eq!(executor.graph_index("tex"), Some([Some(3), None, None]));
    // Same topology: no rebake, same index.
    assert!(!executor.set_graph(one_sampled_graph(4)).unwrap());
    assert_eq!(executor.graph_index("tex"), Some([Some(3), None, None]));
    // A resize rebakes and recreates the physical image; the name keeps
    // its index.
    assert!(executor.set_graph(one_sampled_graph(8)).unwrap());
    assert_eq!(executor.graph_index("tex"), Some([Some(3), None, None]));
    // A second resource takes the next index in declaration order.
    let mut graph = one_sampled_graph(8);
    let other = graph.image(
        "other",
        ImageDesc::color_2d(vk::Format::R8G8B8A8_UNORM, 8, 8, vk::ImageUsageFlags::SAMPLED),
        ResourceClass::Persistent,
    );
    graph
        .pass("sample2", PassKind::Compute, ())
        .sampled(other, ShaderStage::Compute)
        .root();
    assert!(executor.set_graph(graph.clone()).unwrap());
    assert_eq!(executor.graph_index("tex"), Some([Some(3), None, None]));
    assert_eq!(executor.graph_index("other"), Some([Some(4), None, None]));
    // A rename with the same topology re-keys the handles without a rebake
    // (names are not in the signature; the cached name hash catches it).
    let mut renamed = graph;
    renamed.resources[0].name = "renamed".into();
    assert!(!executor.set_graph(renamed).unwrap());
    assert!(executor.graph_index("renamed").is_some());
    assert_eq!(executor.graph_index("tex"), None);
    assert_eq!(executor.graph_index("other"), Some([Some(4), None, None]));
    assert_clean(&gpu);
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
fn run_demo(
    gpu: &Arc<GpuDevice>,
    extent: vk::Extent2D,
    bake: BakeOptions,
    frames: usize,
    profile: bool,
    text: bool,
) -> (Vec<u8>, usize) {
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
    let renderer = DemoRenderer::new(gpu.clone(), executor.bindless_mut(), target_desc.format).unwrap();
    let mut renderer = if text { renderer.with_text().unwrap() } else { renderer };
    let target_binding = Binding::Image(ImageBinding {
        initial: vk::ImageLayout::UNDEFINED,
        image: target.image,
        view: target.view,
        format: target_desc.format,
        extent,
        range: target.range(),
        identity: 1 << 48,
        sampled: None,
        storage: None,
    });
    let readback_binding = Binding::Buffer(BufferBinding {
        buffer: readback.buffer,
        offset: 0,
        size,
        identity: (1 << 48) + 1,
        storage: None,
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
        storage: None,
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
        assert!(
            stats.pass_gpu_ms.iter().any(|ms| *ms > 0.0),
            "timestamps were read back"
        );
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
    let extent = vk::Extent2D {
        width: 256,
        height: 128,
    };
    let (aliased, aliased_physical) = run_demo(&gpu, extent, BakeOptions { aliasing: true }, 5, true, false);
    let (plain, plain_physical) = run_demo(&gpu, extent, BakeOptions { aliasing: false }, 5, true, false);
    // R5: profiling changes no rendering behaviour.
    let (unprofiled, _) = run_demo(&gpu, extent, BakeOptions { aliasing: true }, 5, false, false);
    assert_eq!(fnv(&unprofiled), fnv(&aliased), "profiling must not change the image");
    assert!(
        aliased_physical < plain_physical,
        "{aliased_physical} vs {plain_physical}"
    );
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

#[test]
#[ignore = "needs a Vulkan 1.3 GPU and installed sans-serif fonts"]
fn parley_text_renders_in_screen_and_world_space() {
    let gpu = gpu();
    let extent = vk::Extent2D {
        width: 512,
        height: 256,
    };
    let (plain, _) = run_demo(&gpu, extent, BakeOptions::default(), 2, false, false);
    let (text, _) = run_demo(&gpu, extent, BakeOptions::default(), 2, false, true);
    let row_bytes = extent.width as usize * 4;
    for (top, bottom) in [(20, 100), (135, 210)] {
        let range = top * row_bytes..bottom * row_bytes;
        let changed = plain[range.clone()]
            .as_chunks::<4>()
            .0
            .iter()
            .zip(text[range].as_chunks::<4>().0)
            .filter(|(before, after)| before[..3] != after[..3])
            .count();
        assert!(
            changed > 100,
            "only {changed} changed pixels between rows {top} and {bottom}"
        );
    }
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
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(gpu.queue_family),
                None,
            )
            .unwrap();
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(1),
            )
            .unwrap()[0];
        device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .unwrap();
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
    use ae_framegraph::{FrameGraph, LoadOp, PassKind, ResourceId, Role};
    use ae_render::executor::{PassContext, Recorder};

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
    graph
        .pass("paint", PassKind::Graphics, 0)
        .color(canvas, LoadOp::Load)
        .id();
    graph
        .pass("copy", PassKind::Transfer, 1)
        .access(canvas, Role::TransferSrc)
        .access(readback, Role::TransferDst)
        .id();
    graph
        .pass("host read", PassKind::Transfer, 2)
        .access(readback, Role::HostRead)
        .root()
        .id();

    let mut image = OwnedImage::new(&gpu, desc, "canvas").unwrap();
    let mut buffer = OwnedBuffer::new(&gpu, buffer_desc, MemoryLocation::GpuToCpu, "readback").unwrap();
    let bindings = [
        Binding::Image(ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: image.image,
            view: image.view,
            format: desc.format,
            extent,
            range: image.range(),
            identity: 1 << 48,
            sampled: None,
            storage: None,
        }),
        Binding::Buffer(BufferBinding {
            buffer: buffer.buffer,
            offset: 0,
            size,
            identity: (1 << 48) + 1,
            storage: None,
        }),
    ];
    let mut executor = Executor::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    executor.set_graph(graph).unwrap();
    let mut recorder = Paint { canvas, readback };
    let expected: Vec<u8> = CLEAR.iter().map(|c| (c * 255.0).round() as u8).collect();
    for frame in 0..3 {
        let token = executor.begin_frame().unwrap();
        executor
            .execute(
                token,
                &mut recorder,
                &|resource| bindings.get(resource.index()).copied(),
                &[],
                &[],
            )
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
    let rect = |rect: [f32; 4], color: [f32; 3], depth: f32| Rect {
        rect,
        color: [color[0], color[1], color[2], 1.0],
        depth: [depth, 0.0, 0.0, 0.0],
    };
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
                        d.cmd_push_constants(
                            ctx.cmd,
                            self.layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            pipeline::push_bytes(r),
                        );
                        d.cmd_draw(ctx.cmd, 6, 1, 0, 0);
                    }
                },
                3 => {
                    let buffer = ctx.buffer(self.readback)?;
                    for (index, target) in self.targets.iter().enumerate() {
                        let image = ctx.image(*target)?;
                        let region = vk::BufferImageCopy::default()
                            .buffer_offset((index as u32 * SIZE * SIZE * 4) as u64)
                            .image_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .image_extent(vk::Extent3D {
                                width: SIZE,
                                height: SIZE,
                                depth: 1,
                            });
                        unsafe {
                            d.cmd_copy_image_to_buffer(
                                ctx.cmd,
                                image.image,
                                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                                buffer.buffer,
                                &[region],
                            )
                        };
                    }
                }
                _ => {}
            }
            Ok(())
        }
    }

    let gpu = gpu();
    let extent = vk::Extent2D {
        width: SIZE,
        height: SIZE,
    };
    let color = ImageDesc::color_2d(
        FORMAT,
        SIZE,
        SIZE,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    let bytes = (SIZE * SIZE * 4) as u64;
    let buffer_desc = BufferDesc {
        size: 2 * bytes,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
    };
    let mut graph = FrameGraph::<u8>::new();
    let t0 = graph.image("target 0", color, ResourceClass::External);
    let t1 = graph.image("target 1", color, ResourceClass::External);
    let readback = graph.buffer("readback", buffer_desc, ResourceClass::External);
    let depth = graph.image(
        "depth",
        ImageDesc::depth_2d(depth_format, SIZE, SIZE, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
        ResourceClass::Persistent,
    );
    graph
        .pass("near", PassKind::Graphics, 0)
        .color(t0, LoadOp::Clear)
        .color(t1, LoadOp::Clear)
        .depth(depth, LoadOp::Clear)
        .id();
    graph
        .pass("far", PassKind::Graphics, 1)
        .color(t0, LoadOp::Load)
        .color(t1, LoadOp::Load)
        .depth(depth, LoadOp::Load)
        .id();
    graph
        .pass("decal", PassKind::Graphics, 2)
        .color(t0, LoadOp::Load)
        .depth_read(depth)
        .id();
    graph
        .pass("copy", PassKind::Transfer, 3)
        .access(t0, Role::TransferSrc)
        .access(t1, Role::TransferSrc)
        .discard(readback, Role::TransferDst)
        .id();
    graph
        .pass("host read", PassKind::Transfer, 4)
        .access(readback, Role::HostRead)
        .root()
        .id();

    let device = &gpu.device;
    let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let layout = pipeline::pipeline_layout(device, &[], std::mem::size_of::<Rect>() as u32, stages).unwrap();
    let vertex = include_bytes!(concat!(env!("OUT_DIR"), "/test_rect.vert.spv"));
    let make = |fragment: &[u8], formats: &[vk::Format], write| {
        let depth = Some(DepthState {
            format: depth_format,
            write,
            compare: vk::CompareOp::LESS,
        });
        let desc = GraphicsDesc {
            vertex,
            fragment,
            layout,
            color_formats: formats,
            depth,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            blend: Blend::Off,
            glow_blend: Blend::Off,
        };
        pipeline::graphics_pipeline(device, &desc).unwrap()
    };
    let mrt = make(
        include_bytes!(concat!(env!("OUT_DIR"), "/test_mrt.frag.spv")),
        &[FORMAT, FORMAT],
        true,
    );
    let decal = make(
        include_bytes!(concat!(env!("OUT_DIR"), "/test_flat.frag.spv")),
        &[FORMAT],
        false,
    );

    let mut images = [
        OwnedImage::new(&gpu, color, "target 0").unwrap(),
        OwnedImage::new(&gpu, color, "target 1").unwrap(),
    ];
    let mut buffer = OwnedBuffer::new(&gpu, buffer_desc, MemoryLocation::GpuToCpu, "readback").unwrap();
    let image_binding = |image: &OwnedImage, identity| {
        Binding::Image(ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: image.image,
            view: image.view,
            format: FORMAT,
            extent,
            range: image.range(),
            identity,
            sampled: None,
            storage: None,
        })
    };
    let bindings = [
        image_binding(&images[0], 1 << 48),
        image_binding(&images[1], (1 << 48) + 1),
        Binding::Buffer(BufferBinding {
            buffer: buffer.buffer,
            offset: 0,
            size: 2 * bytes,
            identity: (1 << 48) + 2,
            storage: None,
        }),
    ];
    let mut executor = Executor::new(
        gpu.clone(),
        ExecutorConfig {
            record_threads,
            ..ExecutorConfig::default()
        },
    )
    .unwrap();
    executor.set_graph(graph).unwrap();
    let mut recorder = Draw {
        mrt,
        decal,
        layout,
        targets: [t0, t1],
        readback,
        rects: [vec![red], vec![green, blue], vec![white]],
    };
    for frame in 0..3 {
        let token = executor.begin_frame().unwrap();
        executor
            .execute(
                token,
                &mut recorder,
                &|resource| bindings.get(resource.index()).copied(),
                &[],
                &[],
            )
            .unwrap();
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
        assert_eq!(
            pixel(0, right, top),
            [0, 255, 0],
            "frame {frame}: green where nothing is nearer"
        );
        assert_eq!(
            pixel(0, right, bottom),
            [255, 255, 255],
            "frame {frame}: decal over green"
        );
        assert_eq!(pixel(0, left, bottom), [255, 0, 0], "frame {frame}: decal behind red");
        // Half intensity: 127 or 128 depending on the device's rounding.
        let half = |p: [u8; 3], want: [u8; 3]| p.iter().zip(want).all(|(a, b)| a.abs_diff(b) <= 1);
        assert!(
            half(pixel(1, left, top), [0, 0, 128]),
            "frame {frame}: second target {:?}",
            pixel(1, left, top)
        );
        assert!(
            half(pixel(1, 2, bottom), [128, 0, 0]),
            "frame {frame}: second target, no decal {:?}",
            pixel(1, 2, bottom)
        );
        assert!(
            half(pixel(1, right, bottom), [0, 128, 0]),
            "frame {frame}: second target, no decal {:?}",
            pixel(1, right, bottom)
        );
    }
    assert_eq!(
        executor.stats().parallel_frames,
        if record_threads > 1 { 3 } else { 0 },
        "recorded on the threads"
    );
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
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{build_pyramid, generate};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};

    let gpu = gpu();
    let heightmap = generate(513, 513, 7);
    let pyramid = build_pyramid(&heightmap, 400.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 400.0, 0.0, Some(30.0));
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback);
    {
        // Low over the south edge, looking north: terrain below, sky above.
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera.pitch = 0.25;
        view.camera.distance = 1500.0;
    }
    let extent = vk::Extent2D {
        width: 320,
        height: 200,
    };
    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
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
    let encode = |c: f32| {
        if c <= 0.003_130_8 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    };
    // The clear colour: the sky's radiance at the default exposure, through
    // the default curve (none: the sRGB target saturates at 1).
    let exposure = ae_render::lighting::exposure(ae_render::lighting::DEFAULT_EV100);
    let sky_radiance = ae_render::lighting::sky_radiance(ae_render::colorspace::WorkingSpace::Bt709);
    let sky: Vec<u8> = sky_radiance
        .iter()
        .map(|&c| (encode((c * exposure).min(1.0)) * 255.0).round() as u8)
        .collect();
    let is_sky = |p: &[u8]| p[..3].iter().zip(&sky).all(|(a, b)| a.abs_diff(*b) <= 2);
    let sky_pixels = pixels.chunks(4).filter(|p| is_sky(p)).count();
    let total = (extent.width * extent.height) as usize;
    eprintln!("{leaves} leaves, sky {sky_pixels} of {total} pixels");
    assert!(
        sky_pixels > total / 50 && sky_pixels < total * 9 / 10,
        "terrain and sky both visible: {sky_pixels} of {total}"
    );
    let top = &pixels[..(extent.width * 4) as usize];
    let bottom = &pixels[pixels.len() - (extent.width * 4) as usize..];
    assert!(
        top.chunks(4).filter(|p| is_sky(p)).count() > bottom.chunks(4).filter(|p| is_sky(p)).count(),
        "sky is at the top"
    );
    assert!(leaves > 64, "refined well beyond the two roots: {leaves}");
    ae_render::thread::RenderApp::destroy(&mut app, &gpu);
    assert_clean(&gpu);
}

/// A rebuild of the whole renderer gives its global-set handles back
/// (M15): ten full rebuilds and ten full retargets do not grow the
/// registry's live counts, which would otherwise exhaust the 64 samplers
/// after about five, with no module and with `render/post` (whose programs
/// are declared again after each rebuild and still draw the same frame).
/// Then the uniform array is filled to the device's capacity.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn rebuilding_the_renderer_does_not_leak_handles() {
    let gpu = gpu();
    rebuild_cycle(&gpu, None);
    let post = ae_render::module::build("post").expect("build render/post");
    rebuild_cycle(&gpu, Some(post));
    assert_clean(&gpu);
}

fn rebuild_cycle(gpu: &Arc<GpuDevice>, module: Option<Vec<u8>>) {
    use ae_render::thread::{Headless, RenderApp, TargetInfo};

    let label = if module.is_some() { "post" } else { "no module" };
    let mut app = scene_app(gpu, module);
    let target = TargetInfo {
        extent: SCENE_EXTENT,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
    let mut headless = Headless::new(gpu, &mut app, target, config).unwrap();
    // A running scene first: the baseline counts the graph's own handles.
    headless.frames(&mut app, 4).unwrap();
    headless.wait().unwrap();
    let baseline = headless.stats().bindless_live;
    let first = app.readback().unwrap();
    for (step, full_retarget) in (0..20).map(|step| (step, step >= 10)) {
        headless.wait().unwrap();
        if full_retarget {
            // Retarget's full path (an offscreen app always takes it)
            // destroys the post chain and the module's programs; its destroy
            // takes the offscreen target too.
            assert!(RenderApp::retarget(&mut app, gpu, headless.bindless_mut(), target, 2).unwrap());
            app.create_offscreen(gpu, SCENE_EXTENT).unwrap();
        } else {
            // A rebuild of the whole renderer over the running one: every
            // module is destroyed and created again (the offscreen output
            // stays).
            RenderApp::create(&mut app, gpu, headless.bindless_mut(), target, 2).unwrap();
        }
        // Every module's images are new objects under reused identities.
        headless.forget_external_states();
        headless.frames(&mut app, 4).unwrap();
        headless.wait().unwrap();
        let live = headless.stats().bindless_live;
        assert_eq!(
            live, baseline,
            "{label}: step {step} leaked handles: {live:?}, baseline {baseline:?}"
        );
        assert!(
            app.readback().unwrap() == first,
            "{label}: step {step} draws a different frame"
        );
    }
    fill_the_uniform_array(gpu, &mut headless, &mut app, target, baseline);
    headless.wait().unwrap();
    RenderApp::destroy(&mut app, gpu);
}

/// The uniform array (binding 4) is sized from the device, at least
/// `UNIFORM_BUFFER_MIN` and at most `UNIFORM_BUFFER_MAX`: every entry the
/// device gives can be written and used under a running scene, the next one
/// is refused with the class named, and giving them back restores the
/// counts and the frame.
fn fill_the_uniform_array(
    gpu: &Arc<GpuDevice>,
    headless: &mut ae_render::thread::Headless<
        <ae_render::terrain::app::TerrainApp as ae_render::thread::RenderApp>::Pass,
    >,
    app: &mut ae_render::terrain::app::TerrainApp,
    target: ae_render::thread::TargetInfo,
    baseline: [u32; ae_render::bindless::CLASS_COUNT],
) {
    use ae_render::bindless::{UNIFORM_BUFFER_MAX, UNIFORM_BUFFER_MIN, uniform_buffer_capacity};

    let properties = unsafe { gpu.instance.instance.get_physical_device_properties(gpu.physical) };
    let capacity = headless.bindless_mut().uniform_buffer_capacity();
    assert_eq!(capacity, uniform_buffer_capacity(&properties.limits));
    assert!(
        (UNIFORM_BUFFER_MIN..=UNIFORM_BUFFER_MAX).contains(&capacity),
        "{capacity}"
    );
    let first = app.readback().unwrap();
    let mut block =
        ae_render::upload::buffer(gpu, &[0u8; 256], vk::BufferUsageFlags::UNIFORM_BUFFER, "uniform fill").unwrap();
    let free = capacity - baseline[4];
    let ((), group) = headless
        .bindless_mut()
        .group(|bindless| {
            for _ in 0..free {
                bindless.whole_uniform_buffer(&block)?;
            }
            Ok(())
        })
        .unwrap();
    let err = headless
        .bindless_mut()
        .whole_uniform_buffer(&block)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("uniform buffer") && err.contains(&format!("capacity {capacity}")),
        "{err}"
    );
    // A frame with every entry written, the highest index included.
    headless.frames(app, 2).unwrap();
    headless.wait().unwrap();
    assert_eq!(headless.stats().bindless_live[4], capacity);
    headless.bindless_mut().retire_group(group);
    // Retired indices come back once every frame slot has dropped them.
    headless.frames(app, 2).unwrap();
    headless.wait().unwrap();
    // The frame adapts over frames (exposure), so compare after a rebuild,
    // as the rebuild steps do.
    ae_render::thread::RenderApp::create(app, gpu, headless.bindless_mut(), target, 2).unwrap();
    headless.forget_external_states();
    headless.frames(app, 4).unwrap();
    headless.wait().unwrap();
    assert_eq!(headless.stats().bindless_live, baseline);
    assert!(app.readback().unwrap() == first, "the frame after the fill");
    block.destroy(gpu);
}

/// Units over the terrain: an instance stream of four units, reset + cull +
/// indirect draw after the terrain draw into the same target and depth
/// buffer. The cull passes three (the fourth is behind the camera: the
/// instance count read back is 3); each shows its class colour at its
/// projected position; the previous position of a moved unit (alpha 1
/// shows the latest) shows none. A turn at a quarter tick draws the
/// silhouette a quarter of the way around the shortest arc.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn units_render_over_terrain() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{Headless, TargetInfo};
    use ae_render::units::{
        ATTR_PREV_ROT_X, ATTR_PREV_X, ATTR_RADIUS, ATTR_ROT_X, ATTR_X, AnimWrites, InstanceStream, UnitClass, UnitFeed,
        UnitsSetup,
    };
    use glam::{Vec2, Vec4};
    use std::time::{Duration, Instant};

    let gpu = gpu();
    // Flat ground at 50 with water at 60 (class 1 floats on it).
    let heightmap = Heightmap {
        width: 129,
        height: 129,
        data: vec![0.5; 129 * 129],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, Some(60.0));
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });

    let red = [1.0, 0.02, 0.02];
    let blue = [0.02, 0.02, 1.0];
    let magenta = [1.0, 0.02, 1.0];
    let classes = vec![
        UnitClass {
            color: red,
            gamut: ae_render::colorspace::Gamut::Bt709,
            floats: false,
        },
        UnitClass {
            color: blue,
            gamut: ae_render::colorspace::Gamut::Bt709,
            floats: true,
        },
        UnitClass {
            color: magenta,
            gamut: ae_render::colorspace::Gamut::Bt709,
            floats: false,
        },
    ];
    // (x, y, prev_x, prev_y, class): the red unit moved from (420, 512).
    let units = [
        (560.0f32, 512.0f32, 420.0f32, 512.0f32, 0u32),
        (512.0, 600.0, 512.0, 600.0, 1),
        (470.0, 440.0, 470.0, 440.0, 2),
        (512.0, -4000.0, 512.0, -4000.0, 0),
    ];
    let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
    for &(x, y, px, py, class) in &units {
        for (a, word) in [
            x.to_bits(),
            y.to_bits(),
            px.to_bits(),
            py.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            4.0f32.to_bits(),
            class,
        ]
        .into_iter()
        .enumerate()
        {
            columns[a].push(word);
        }
    }
    let feed = Arc::new(UnitFeed::default());
    let columns = at_rest(columns);
    let publish = |tick, columns, published, tick_seconds| {
        feed.publish(Arc::new(InstanceStream {
            tick,
            published,
            tick_seconds,
            sim_seconds: 0.0,
            columns,
            anim: AnimWrites::default(),
        }))
    };
    publish(7, columns.clone(), Instant::now(), 0.0);
    let setup = UnitsSetup {
        feed: feed.clone(),
        capacity: 64,
        classes,
        min_pixels: 28.0,
        models: Vec::new(),
        rigs: Vec::new(),
    };
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
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
    let mut run = Headless::new(&gpu, &mut app, target, config).unwrap();
    run.frames(&mut app, 12).unwrap();
    run.wait().unwrap();
    let pixels = app.readback().unwrap();
    if let Ok(path) = std::env::var("AE_TEST_CAPTURE") {
        let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
        ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
        std::fs::write(&path, ppm).unwrap();
    }
    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let pixel = |x: f32, y: f32, h: f32| -> [u8; 3] {
        let clip = view_proj * Vec4::new(x, h, y, 1.0);
        let (u, v) = (
            (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
            (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
        );
        assert!(
            u >= 0.0 && v >= 0.0 && u < extent.width as f32 && v < extent.height as f32,
            "({x}, {y}) projects into the view"
        );
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
    assert!(
        !(r > 2 * g && r > 2 * b),
        "nothing red where the red unit was a tick ago: {r} {g} {b}"
    );
    assert_eq!(
        app.last_unit_counts(),
        Some((4, 3)),
        "four rows uploaded, three pass the cull"
    );

    let render_turn = |app: &mut TerrainApp, run: &mut Headless<_>, tick, rot: [f32; 4], quarter_tick: bool| {
        let mut columns = columns.clone();
        columns[ATTR_PREV_X][0] = columns[ATTR_X][0];
        columns[ATTR_RADIUS][0] = 80.0f32.to_bits();
        for (k, word) in rot.map(f32::to_bits).into_iter().enumerate() {
            columns[ATTR_ROT_X + k][0] = word;
            columns[ATTR_PREV_ROT_X + k][0] = 0; // Identity encoded as zero.
        }
        // Keep alpha close to 0.25 even if a frame takes a while to prepare.
        let now = Instant::now();
        if quarter_tick {
            publish(tick, columns, now - Duration::from_secs(900), 3600.0);
        } else {
            publish(tick, columns, now, 0.0);
        }
        run.frames(app, 1).unwrap();
        run.wait().unwrap();
        app.readback()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| u16::from(p[0]) > 2 * u16::from(p[1]) && u16::from(p[0]) > 2 * u16::from(p[2]))
            .collect::<Vec<_>>()
    };
    // Exact 180-degree turns have a zero dot product. SLERP gives 45 degrees
    // at a quarter tick, while NLERP would give about 37 degrees.
    let quarter = render_turn(&mut app, &mut run, 8, [0.0, -2.0, 0.0, 0.0], true);
    let expected = render_turn(&mut app, &mut run, 9, yaw(1.0, 1.0), false);
    // The negative representative of a 160-degree turn takes the same arc.
    let (sin, cos) = 160.0f32.to_radians().sin_cos();
    let opposite = render_turn(&mut app, &mut run, 10, yaw(cos, sin).map(|c| -2.0 * c), true);
    let (sin, cos) = 40.0f32.to_radians().sin_cos();
    let expected_opposite = render_turn(&mut app, &mut run, 11, yaw(cos, sin), false);
    let different_pixels = |a: &[bool], b: &[bool]| a.iter().zip(b).filter(|(a, b)| a != b).count();
    for (got, want) in [(&quarter, &expected), (&opposite, &expected_opposite)] {
        let error = different_pixels(got, want);
        assert!(error <= 5, "quarter-tick silhouette differs by {error} red pixels");
    }

    run.finish().unwrap();
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
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Vec2, Vec4};

    let gpu = gpu();
    // 128 x 128 cells of flat grass at 50.
    let cells = (128usize, 128usize);
    let (heightmap, scale, offset) = Heightmap::from_cells(cells, |_, _| 50.0, Some((100.0, 0.0)));
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, scale, offset, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: Some(vec![0; cells.0 * cells.1]),
    });

    // A red unit in the middle of the square to raise, a blue one outside.
    let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
    for (x, y, class) in [(512.0f32, 512.0f32, 0u32), (360.0, 600.0, 1)] {
        for (a, word) in [
            x.to_bits(),
            y.to_bits(),
            x.to_bits(),
            y.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            5.0f32.to_bits(),
            class,
        ]
        .into_iter()
        .enumerate()
        {
            columns[a].push(word);
        }
    }
    let feed = Arc::new(UnitFeed::default());
    feed.publish(Arc::new(InstanceStream {
        tick: 1,
        published: std::time::Instant::now(),
        tick_seconds: 0.0,
        sim_seconds: 0.0,
        columns: at_rest(columns),
        anim: AnimWrites::default(),
    }));
    let classes = vec![
        UnitClass {
            gamut: ae_render::colorspace::Gamut::Bt709,
            color: [1.0, 0.02, 0.02],
            floats: false,
        },
        UnitClass {
            gamut: ae_render::colorspace::Gamut::Bt709,
            color: [0.02, 0.02, 1.0],
            floats: false,
        },
    ];
    let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(UnitsSetup {
        feed,
        capacity: 8,
        classes,
        min_pixels: 6.0,
        models: Vec::new(),
        rigs: Vec::new(),
    });
    let camera = {
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera.target = Vec2::new(512.0, 512.0);
        view.camera.pitch = 0.7;
        view.camera.distance = 600.0;
        view.fog = false;
        view.camera
    };
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
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
    let update = editor
        .edit(
            3,
            &[(58, 58, 70, 70)],
            |x, y| if inside(x, y) { 90.0 } else { 50.0 },
            |_, _| 3,
        )
        .unwrap();
    eprintln!(
        "terrain update: {} bytes in {} pyramid and {} error regions",
        update.bytes(),
        update.pyramid.len(),
        update.errors.len()
    );
    let edits = app.terrain_edits();
    edits.lock().unwrap().pending.push(update);
    let after = step(&mut run, &mut app);
    {
        let e = edits.lock().unwrap();
        assert_eq!(
            (e.applied, e.version, e.pending.len()),
            (1, 3, 0),
            "applied at the next frame"
        );
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
        (
            (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
            (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
        )
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
    assert!(
        red(rgb(&before, low)),
        "before: the red unit on the old ground {:?}",
        rgb(&before, low)
    );
    assert!(
        red(rgb(&after, high)),
        "after: the red unit on the raised ground {:?}",
        rgb(&after, high)
    );
    assert!(
        !red(rgb(&after, low)),
        "after: nothing red at the old ground height {:?}",
        rgb(&after, low)
    );
    assert!(
        blue(rgb(&before, project(360.0, 600.0, 50.0))) && blue(rgb(&after, project(360.0, 600.0, 50.0))),
        "the blue unit stays"
    );
    // The raised top looks different (rock, lit as the top of a block).
    let top = project(480.0, 540.0, 90.0);
    let diff = rgb(&before, top)
        .iter()
        .zip(rgb(&after, top))
        .map(|(a, b)| (a - b).abs())
        .sum::<i32>();
    assert!(
        diff > 30,
        "the raised square changed colour: {:?} -> {:?}",
        rgb(&before, top),
        rgb(&after, top)
    );

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
    assert!(
        outside > 40_000 && moved == 0,
        "{moved} of {outside} pixels outside the edit changed"
    );

    // Several small rectangles in one update, of sizes that leave the
    // regions after them unaligned unless the upload aligns them (a 2 x 1
    // cell rectangle is 3 x 2 corners, 24 bytes of heights).
    let update = editor
        .edit(
            4,
            &[(20, 20, 21, 20), (100, 7, 100, 7), (3, 90, 5, 90)],
            |x, y| if inside(x, y) { 90.0 } else { 50.0 + (x % 3) as f32 },
            |_, _| 2,
        )
        .unwrap();
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

/// Photometric lighting (M12): the scene's sun scales with its illuminance,
/// the exposure applies before the scene is stored (one EV halves it), and
/// a point light with no authored range follows the inverse square: at 4 m
/// it adds a quarter of what it adds at 2 m. All read back through the
/// clamp curve, so linear values decode exactly.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn photometric_sun_exposure_and_inverse_square() {
    use ae_render::lights::Light;
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::tonemap::ToneParams;
    use glam::{Vec2, Vec3, Vec4};

    let gpu = gpu();
    let heightmap = Heightmap {
        width: 129,
        height: 129,
        data: vec![0.5; 129 * 129],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let target = Vec2::new(512.0, 512.0);
    let ground = 50.0;
    let camera = std::cell::Cell::new(None);
    let render = |sun: f32, ev100: f32, lights: Vec<Light>, unculled: bool| -> Vec<u8> {
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_lights(lights)
            .with_tone(ToneParams {
                ev100,
                sun_illuminance: sun,
                bloom: 0.0,
                auto: false,
            });
        if unculled {
            app = app.with_unculled_lights();
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = target;
            view.camera.pitch = 1.55;
            view.camera.distance = 100.0;
            view.fog = false;
            camera.set(Some(view.camera));
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target_info = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target_info, 4, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    // The sun: half the illuminance gives half the added (sun-only) light,
    // and one EV more doubles the stored scene.
    let sky14 = render(0.0, 14.0, Vec::new(), false);
    let sun100 = render(100_000.0, 14.0, Vec::new(), false);
    let sun50 = render(50_000.0, 14.0, Vec::new(), false);
    let sky13 = render(0.0, 13.0, Vec::new(), false);
    let dark = render(0.0, 14.0, Vec::new(), false);
    let light = |h: f32| {
        Light::point(
            Vec3::new(target.x, ground + h, target.y),
            Vec3::ONE,
            100_000.0,
            0.0,
            0.05,
        )
    };
    let at2 = render(0.0, 14.0, vec![light(2.0)], false);
    let at4 = render(0.0, 14.0, vec![light(4.0)], false);

    let view_proj = lod_view(&source, &camera.get().unwrap(), extent.width, extent.height).view_proj;
    let pixel = |x: f32, y: f32, h: f32| -> usize {
        let clip = view_proj * Vec4::new(x, h, y, 1.0);
        let u = (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32;
        let v = (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32;
        (v as usize * extent.width as usize + u as usize) * 4
    };
    let at = pixel(target.x, target.y, ground);
    let lin = |pixels: &[u8], c: usize| -> f32 {
        let v = pixels[at + c] as f32 / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let add = |a: &[u8], b: &[u8]| -> f32 { (0..3).map(|c| lin(a, c) - lin(b, c)).sum::<f32>() / 3.0 };
    let (d100, d50) = (add(&sun100, &sky14), add(&sun50, &sky14));
    assert!(d100 > 0.2 && d50 > 0.1, "the sun lights the ground: {d100} {d50}");
    assert!(
        (d100 / d50 - 2.0).abs() < 0.05,
        "sun illuminance scales: {:.4} vs {:.4}",
        d100 / d50,
        2.0
    );
    // The scene is stored exposed (the tonemap only curves): one EV more
    // doubles the stored sky, which stays below the clamp's ceiling.
    let (s14, s13) = (lin(&sky14, 1), lin(&sky13, 1));
    assert!(s14 > 0.1 && s13 < 0.98, "the sky is in range: {s14} {s13}");
    assert!(
        (s13 / s14 - 2.0).abs() < 0.08,
        "one EV doubles the stored light: {:.4}",
        s13 / s14
    );

    // A point light with no range: 4 m adds a quarter of 2 m.
    let (a2, a4) = (add(&at2, &dark), add(&at4, &dark));
    assert!(a2 > 0.05 && a4 > 0.01, "the point light adds light: {a2} {a4}");
    assert!(
        (a2 / a4 - 4.0).abs() < 0.2,
        "inverse square: {:.3} (2 m vs 4 m)",
        a2 / a4
    );
    // A source radius larger than the authored range clamps the near field
    // without extinguishing the light: the range window still applies.
    let big = render(
        0.0,
        14.0,
        vec![Light::point(
            Vec3::new(target.x, ground + 1.0, target.y),
            Vec3::ONE,
            1.0e9,
            2.0,
            100.0,
        )],
        false,
    );
    assert!(
        add(&big, &dark) > 0.02,
        "the regularisation radius does not put the light out: {}",
        add(&big, &dark)
    );
    assert_clean(&gpu);
}

/// Automatic exposure (M12): the metered, eased EV (a GPU histogram, a
/// 1x1 exposure image, no readback) holds the display's mean near the key
/// across a fourfold sun change, while the same change at a manual EV
/// brightens the capture. At the reference sun auto matches the manual
/// reference EV (the key is calibrated there). A frame that is all fog of
/// war meters as if it were in sight, so the fog darkens the auto capture
/// as much as the manual one instead of being exposed away. Starts at the
/// reference EV and eases exponentially (1.5/s toward a brighter scene,
/// 0.5/s toward a darker one), so the darker scene needs the longer run.
/// Automatic exposure meters what the player sees and where they look: a
/// fireball-sized cluster of glowing scars darkens the plain ground around it a little more
/// with the bloom on (the glow the bloom shows is metered) than with it off, and
/// more when it sits at the centre of the frame than at its edge (centre-
/// weighted metering).
#[test]
#[ignore = "needs a Vulkan GPU"]
fn auto_exposure_meters_the_glow_and_the_centre() {
    use ae_render::decals::{Decal, DecalAtlas, DecalFeed, DecalsSetup, colour_texel, surface_texel};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::tonemap::ToneParams;
    use glam::{Vec2, Vec4};

    let gpu = gpu();
    let n = 65;
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![0.1; n * n],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 4.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    // Straight down, so the frame is ground only.
    let camera = Camera {
        target: Vec2::new(128.0, 128.0),
        ground: 10.0,
        distance: 160.0,
        yaw: 0.0,
        pitch: 1.55,
        fov_y: 0.8,
    };
    let size = 16u32;
    let texels = (size * size) as usize;
    let atlas = Arc::new(DecalAtlas {
        size,
        colour: vec![vec![colour_texel([0.3, 0.3, 0.3, 1.0]); texels]],
        surface: vec![vec![surface_texel([0.0, 0.0, 1.0], 1.0, 0.9); texels]],
        height_layers: Vec::new(),
        materials: Vec::new(),
    });
    let screen = |x: f32, z: f32| {
        let clip = lod_view(&source, &camera, extent.width, extent.height).view_proj * Vec4::new(x, 10.0, z, 1.0);
        (
            (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
            (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
        )
    };
    let render = |at: [f32; 2], bloom: f32| -> Vec<u8> {
        let feed = Arc::new(DecalFeed::default());
        // A 5x5 cluster of glowing scars, 12 m apart: a fireball-sized
        // patch of light (one scar glows only at its core).
        let scars: Vec<Decal> = (0..25)
            .map(|k| {
                let (i, j) = ((k % 5) as f32 - 2.0, (k / 5) as f32 - 2.0);
                Decal {
                    glow: 1.0,
                    glow_life: 1000.0,
                    ..Decal::permanent([at[0] + 12.0 * i, at[1] + 12.0 * j], [8.0, 8.0], 0.0, 0)
                }
            })
            .collect();
        feed.place_all(&scars);
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_decals(DecalsSetup {
                atlas: Some(atlas.clone()),
                ..DecalsSetup::new(feed, 32)
            })
            .with_tone(ToneParams {
                bloom,
                auto: true,
                ..Default::default()
            });
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera = camera;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        // 10 s: the exposure settles.
        run_headless(&gpu, &mut app, target, 600, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    // The centre and a spot near the frame's left or right edge.
    let (centre, edge) = ([128.0, 128.0], [128.0 + 70.0, 128.0]);
    let (eu, _) = screen(edge[0], edge[1]);
    // Plain ground on the far side from the edge spot, clear of both
    // scars' glow and bloom.
    let (x0, x1) = if eu > extent.width as f32 / 2.0 {
        (8, 40)
    } else {
        (extent.width as usize - 40, extent.width as usize - 8)
    };
    let ground = |p: &[u8]| -> f32 {
        let mut sum = 0.0;
        let mut count = 0.0;
        for y in 8..extent.height as usize - 8 {
            for x in x0..x1 {
                let i = (y * extent.width as usize + x) * 4;
                sum += 0.3 * p[i] as f32 + 0.59 * p[i + 1] as f32 + 0.11 * p[i + 2] as f32;
                count += 1.0;
            }
        }
        sum / count
    };
    let strength = ae_render::tonemap::BLOOM_STRENGTH;
    let shot = render(centre, strength);
    if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
        let mut ppm = format!("P6 {} {} 255\n", extent.width, extent.height).into_bytes();
        ppm.extend(shot.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
        std::fs::write(format!("{dir}/auto-exposure-scar.ppm"), ppm).unwrap();
    }
    let centre_bloom = ground(&shot);
    let centre_plain = ground(&render(centre, 0.0));
    let edge_plain = ground(&render(edge, 0.0));
    eprintln!(
        "edge scar at x {eu:.0}; ground luma: centre scar with bloom {centre_bloom:.1}, without {centre_plain:.1}; edge scar without bloom {edge_plain:.1}"
    );
    // The glow joins at the bloom's strength (0.3) on top of the same light
    // in the scene, so it moves the exposure a little; readback is
    // deterministic, so a small margin is a real difference.
    assert!(
        centre_bloom < centre_plain - 0.15,
        "the bloom's glow is metered: {centre_bloom:.1} vs {centre_plain:.1}"
    );
    assert!(
        centre_plain < edge_plain - 1.0,
        "the centre weighs more: {centre_plain:.1} vs {edge_plain:.1}"
    );
    assert_clean(&gpu);
}

#[test]
#[ignore = "needs a Vulkan GPU"]
fn auto_exposure_holds_the_key_across_sunlight() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::tonemap::ToneParams;
    use glam::Vec2;

    let gpu = gpu();
    let heightmap = Heightmap {
        width: 129,
        height: 129,
        data: vec![0.5; 129 * 129],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    // A vision feed with no frame published: every cell reads as never
    // seen (dimmed to 0.2).
    let grid = ae_render::terrain::renderer::VisionGrid {
        width: 32,
        height: 32,
        cell_size: 32.0,
    };
    let render = |sun: f32, auto: bool, fog: bool| -> Vec<u8> {
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_tone(ToneParams {
            sun_illuminance: sun,
            auto,
            bloom: 0.0,
            ..Default::default()
        });
        if fog {
            app = app.with_vision(Arc::new(ae_render::terrain::app::VisionFeed::new(grid)));
        }
        {
            // Straight down at the flat map: the ground fills the frame, no
            // sky, so the meter sees one surface at two sun levels.
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = Vec2::new(512.0, 512.0);
            view.camera.pitch = 1.55;
            view.camera.distance = 100.0;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        // 10 s at the readback clock's 60 Hz: the 25% scene is two EV down,
        // which the darkening rate (0.5/s) closes to about 0.013 EV.
        run_headless(&gpu, &mut app, target, 600, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let linear = |v: u8| -> f32 {
        let v = v as f32 / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let mean = |pixels: &[u8]| -> f32 {
        let total = (extent.width * extent.height) as f32;
        pixels
            .chunks(4)
            .map(|p| (linear(p[0]) + linear(p[1]) + linear(p[2])) / 3.0)
            .sum::<f32>()
            / total
    };
    let bright_auto = mean(&render(100_000.0, true, false));
    let dim_auto = mean(&render(25_000.0, true, false));
    let bright_manual = mean(&render(100_000.0, false, false));
    let dim_manual = mean(&render(25_000.0, false, false));
    let fog_auto = mean(&render(100_000.0, true, true));
    let fog_manual = mean(&render(100_000.0, false, true));
    let auto_ratio = bright_auto / dim_auto;
    let manual_ratio = bright_manual / dim_manual;
    let (fog_auto_ratio, fog_manual_ratio) = (fog_auto / bright_auto, fog_manual / bright_manual);
    eprintln!(
        "mean auto {bright_auto:.3} / {dim_auto:.3} = {auto_ratio:.3}; manual {bright_manual:.3} / {dim_manual:.3} = {manual_ratio:.3}; \
         fogged auto {fog_auto:.3} ({fog_auto_ratio:.3} of clear), manual {fog_manual:.3} ({fog_manual_ratio:.3})"
    );
    // The key sits below the untextured reference's look (KEY_LUMINANCE:
    // it is tuned on the textured default map), so auto settles at the key,
    // not at the manual frame; the frame's arithmetic mean of display-linear
    // light reads a little under the meter's trimmed mean log (0.297 against
    // 0.32 measured).
    let key = ae_render::lighting::KEY_LUMINANCE;
    assert!(
        (bright_auto / key - 1.0).abs() < 0.1,
        "auto settles at the key: {bright_auto:.3} vs {key}"
    );
    assert!(
        bright_auto < bright_manual,
        "the key is below the reference's manual look: {bright_auto:.3} vs {bright_manual:.3}"
    );
    assert!(
        (fog_auto_ratio / fog_manual_ratio - 1.0).abs() < 0.15,
        "the fog of war is not exposed away: auto keeps {fog_auto_ratio:.3} of the clear frame, manual {fog_manual_ratio:.3}"
    );
    assert!(
        bright_auto > 0.05 && dim_auto > 0.05,
        "auto-exposure keeps a visible image: {bright_auto} {dim_auto}"
    );
    assert!(
        (auto_ratio - 1.0).abs() < 0.15,
        "auto-exposure holds the key: {auto_ratio:.3}"
    );
    assert!(manual_ratio > 1.6, "manual exposure follows the sun: {manual_ratio:.3}");
    assert_clean(&gpu);
}

/// The models' punctual lights (M12, `KHR_lights_punctual`): with no host
/// light, a unit whose model carries a point light lights the ground
/// around it, and a spot lights the ground its node's -Z faces and not the
/// other side. The clustered frame equals the unculled reference (the
/// append is the only source of both), the light is counted, a model
/// without lights changes nothing, and a second frame is byte-identical.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn model_lights_light_the_ground() {
    use ae_model::{Bone, Joint, LightKind, Material, Model, ModelLight};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Mat4, Vec2, Vec4};

    let gpu = gpu();
    let heightmap = Heightmap {
        width: 129,
        height: 129,
        data: vec![0.5; 129 * 129],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });

    let identity = Mat4::IDENTITY.to_cols_array();
    let bone = |name: &str, parent: Option<u32>, y: f32| Bone {
        name: name.into(),
        parent,
        rest: Mat4::from_translation(glam::Vec3::new(0.0, y, 0.0)).to_cols_array(),
        translation: [0.0, y, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
    };
    let model_with = |light: ModelLight| {
        let mut m = Model {
            bones: vec![bone("root", None, 0.0), bone("lamp", Some(0), 1.0)],
            joints: vec![
                Joint {
                    bone: 0,
                    inverse_bind: identity,
                },
                Joint {
                    bone: 1,
                    inverse_bind: identity,
                },
            ],
            lights: vec![light],
            materials: vec![Material {
                name: "m".into(),
                base_color: [0.5, 0.5, 0.5, 1.0],
                metallic: 0.0,
                roughness: 0.8,
                ..Default::default()
            }],
            ..Default::default()
        };
        model_box(&mut m, [-1.0, 0.0, -1.0], [1.0, 1.0, 1.0], 0);
        m.bounds = ae_model::Bounds {
            min: [-1.0, 0.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };
        m.merge_primitives();
        m.validate().unwrap();
        m
    };
    let point = model_with(ModelLight {
        name: "lamp".into(),
        joint: 1,
        kind: LightKind::Point,
        color: [1.0, 0.4, 0.12],
        intensity: 6_000_000.0,
        range: 40.0,
        inner_angle: 0.0,
        outer_angle: 0.0,
        size: 4.0,
    });
    // A wide cone, so the ground in front of the unit (past its own model)
    // is inside it; model -z is world -z with the heading +x.
    let spot = model_with(ModelLight {
        name: "headlight".into(),
        joint: 1,
        kind: LightKind::Spot,
        color: [0.2, 0.4, 1.0],
        intensity: 5_000_000.0,
        range: 40.0,
        inner_angle: 0.9,
        outer_angle: 1.2,
        size: 4.0,
    });
    let mut unlit = point.clone();
    unlit.lights.clear();
    unlit.validate().unwrap();

    // Two units, both facing +x, far enough apart that their lights miss
    // each other: the first has the point light, the second the spot.
    let units = [(450.0f32, 512.0f32, 0u32), (574.0, 512.0, 1u32)];
    let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
    for &(x, y, class) in &units {
        for (a, word) in [
            x.to_bits(),
            y.to_bits(),
            x.to_bits(),
            y.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            5.0f32.to_bits(),
            class,
        ]
        .into_iter()
        .enumerate()
        {
            columns[a].push(word);
        }
    }
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let mut camera = None;
    let mut render =
        |point: Arc<Model>, spot: Arc<Model>, unculled: bool| -> (Vec<u8>, ae_render::lights::LightStats) {
            let feed = Arc::new(UnitFeed::default());
            feed.publish(Arc::new(InstanceStream {
                tick: 1,
                published: std::time::Instant::now(),
                tick_seconds: 0.0,
                sim_seconds: 0.0,
                columns: at_rest(columns.clone()),
                anim: AnimWrites::default(),
            }));
            let classes = vec![
                UnitClass {
                    color: [0.5; 3],
                    gamut: ae_render::colorspace::Gamut::Bt709,
                    floats: false,
                },
                UnitClass {
                    color: [0.5; 3],
                    gamut: ae_render::colorspace::Gamut::Bt709,
                    floats: false,
                },
            ];
            let setup = UnitsSetup {
                feed,
                capacity: 16,
                classes,
                min_pixels: 8.0,
                models: vec![Some(point), Some(spot)],
                rigs: Vec::new(),
            };
            let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(setup);
            if unculled {
                app = app.with_unculled_lights();
            }
            let view = app.view();
            {
                let mut view = view.lock().unwrap();
                view.camera.target = Vec2::new(512.0, 512.0);
                view.camera.pitch = 1.45;
                view.camera.distance = 260.0;
                view.fog = false;
                camera = Some(view.camera);
            }
            app.create_offscreen(&gpu, extent).unwrap();
            let target = TargetInfo {
                extent,
                format: vk::Format::R8G8B8A8_SRGB,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            };
            let config = ExecutorConfig {
                frames_in_flight: 2,
                profile: false,
                bake: BakeOptions::default(),
                record_threads: 1,
            };
            run_headless(&gpu, &mut app, target, 8, config).unwrap();
            let pixels = app.readback().unwrap();
            let stats = *app.light_stats().lock().unwrap();
            ae_render::thread::RenderApp::destroy(&mut app, &gpu);
            (pixels, stats)
        };
    let dark = render(Arc::new(unlit.clone()), Arc::new(unlit.clone()), false).0;
    let (lit, stats) = render(Arc::new(point.clone()), Arc::new(spot.clone()), false);
    let reference = render(Arc::new(point.clone()), Arc::new(spot.clone()), true).0;
    let again = render(Arc::new(point.clone()), Arc::new(spot.clone()), false).0;
    assert_eq!(lit, again, "the appended lights are deterministic");
    assert_eq!(lit, reference, "clustered equals every light at every pixel");
    assert_eq!(
        (stats.lights, stats.dropped),
        (2, 0),
        "one light per unit was appended: {stats:?}"
    );

    let view_proj = lod_view(&source, &camera.unwrap(), extent.width, extent.height).view_proj;
    let project = |x: f32, y: f32, h: f32| -> (f32, f32) {
        let clip = view_proj * Vec4::new(x, h, y, 1.0);
        (
            (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
            (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
        )
    };
    let rgb = |pixels: &[u8], (u, v): (f32, f32)| -> [i32; 3] {
        let i = (v as usize * extent.width as usize + u as usize) * 4;
        [pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32]
    };
    let ground = 50.0;
    // Beside the point light's unit, clear of its model: its warm light
    // raises red much more than blue.
    let near = project(462.0, 512.0, ground);
    let (was, now) = (rgb(&dark, near), rgb(&lit, near));
    let gain = |c: usize| now[c] - was[c];
    assert!(gain(0) > 3 && gain(0) > 2 * gain(2), "point light: {was:?} -> {now:?}");
    // In front of the spot (world -z) and behind it (world +z): its blue
    // light raises blue, and nothing behind it.
    let front = project(574.0, 500.0, ground);
    let behind = project(574.0, 526.0, ground);
    let (was, now) = (rgb(&dark, front), rgb(&lit, front));
    let gain = |c: usize| now[c] - was[c];
    assert!(gain(2) > 3 && gain(2) > 2 * gain(0), "spot front: {was:?} -> {now:?}");
    assert_eq!(
        rgb(&lit, behind),
        rgb(&dark, behind),
        "the spot lights nothing behind its node"
    );
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
    let materials = (0..cells.1)
        .flat_map(|y| (0..cells.0).map(move |x| (x, y)))
        .map(|(x, y)| material(x, y))
        .collect();
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: Some(materials),
    });
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
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    app.create_offscreen(gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    run_headless(
        gpu,
        &mut app,
        target,
        16,
        ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        },
    )
    .unwrap();
    let pixels = app.readback().unwrap();
    if let Ok(dir) = std::env::var("AE_TEST_CAPTURE") {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
        ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
        std::fs::write(
            format!(
                "{dir}/tileset-{}.ppm",
                N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ),
            ppm,
        )
        .unwrap();
    }
    ae_render::thread::RenderApp::destroy(&mut app, gpu);
    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let project = move |x: f32, y: f32, h: f32| {
        let clip = view_proj * Vec4::new(x, h, y, 1.0);
        (
            (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
            (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
        )
    };
    (pixels, project)
}

/// The pixel's nearest colour among `colors` (sRGB bytes).
fn nearest(p: &[u8], colors: &[[u8; 3]]) -> usize {
    let d = |c: &[u8; 3]| (0..3).map(|i| (p[i] as i32 - c[i] as i32).pow(2)).sum::<i32>();
    (0..colors.len()).min_by_key(|&i| d(&colors[i])).unwrap()
}

fn srgb_bytes(c: [f32; 3]) -> [u8; 3] {
    c.map(|v| {
        ((if v <= 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }) * 255.0)
            .round() as u8
    })
}

/// Per screen row in `rows`, the first column from `start` rightwards
/// whose pixel is closer to `to` than to `from`.
fn switch_columns(pixels: &[u8], rows: std::ops::Range<u32>, start: u32, from: [u8; 3], to: [u8; 3]) -> Vec<u32> {
    rows.filter_map(|v| (start..320u32).find(|&u| nearest(&pixels[((v * 320 + u) * 4) as usize..], &[from, to]) == 1))
        .collect()
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
    let (grass, sand, rock) = (
        srgb_bytes([0.10, 0.75, 0.10]),
        srgb_bytes([0.95, 0.85, 0.25]),
        srgb_bytes([0.55, 0.55, 0.60]),
    );

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
    let (lo, hi) = (
        project(256.0 - reach, 256.0, 50.0).0 - 1.0,
        project(256.0 + reach, 256.0, 50.0).0 + 1.0,
    );
    let (lo, hi) = (lo.min(hi), lo.max(hi));
    eprintln!(
        "tileset boundary columns {}..={} over {} rows (band {lo:.0}..{hi:.0})",
        cols.iter().min().unwrap(),
        cols.iter().max().unwrap(),
        cols.len()
    );
    // Row to row it frays (the height blend), not only bends.
    let jumps = cols.windows(2).filter(|w| w[0].abs_diff(w[1]) >= 2).count();
    eprintln!("{jumps} rows jump 2+ px from the row above");
    assert!(jumps >= 4, "{jumps} rows jump 2+ px from the row above");
    let cell_px = project(264.0, 256.0, 50.0).0 - project(256.0, 256.0, 50.0).0;
    assert!(
        spread(&cols) as f32 >= cell_px,
        "the boundary meanders over a cell ({cell_px:.1} px) or more: spread {}",
        spread(&cols)
    );
    assert!(
        cols.iter().all(|&c| (lo..=hi).contains(&(c as f32))),
        "and stays in its band"
    );

    // The palette: a straight boundary (grass and sand palette colours).
    let (pixels, _) = tileset_scene(&gpu, |_, _| 50.0, split, None, false, 200.0, None);
    let cols = switch_columns(
        &pixels,
        rows.clone(),
        start,
        srgb_bytes([0.23, 0.33, 0.12]),
        srgb_bytes([0.60, 0.54, 0.38]),
    );
    eprintln!(
        "palette boundary columns {}..={}",
        cols.iter().min().unwrap(),
        cols.iter().max().unwrap()
    );
    assert!(
        spread(&cols) <= 1,
        "the palette's boundary is straight: spread {}",
        spread(&cols)
    );

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
    let cols = switch_columns(
        &pixels,
        top.min(bottom)..top.max(bottom),
        project(64.0, 256.0, 0.0).0 as u32,
        grass,
        rock,
    );
    let cell_px = project(264.0, 256.0, 100.0).0 - project(256.0, 256.0, 100.0).0;
    eprintln!(
        "cliff edge columns {}..={} over {} rows ({cell_px:.1} px per cell)",
        cols.iter().min().unwrap(),
        cols.iter().max().unwrap(),
        cols.len()
    );
    assert!(
        spread(&cols) as f32 >= cell_px,
        "the cliff edge is irregular over a cell or more: spread {}",
        spread(&cols)
    );
    assert_clean(&gpu);
}

/// Anti-tiling on flat fields of mud (temperate layer 6, 20 m repeats) and
/// grass (layer 0, 24 m), from straight above,
/// about 20 repeats across: without it the image correlates with itself
/// shifted by one repeat (above its correlation at 1.5 repeats, the broad
/// variation's share); with it much less, while the mean and the
/// contrast (standard deviation) stay close, as the variance-preserving
/// blend promises.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn anti_tiling_breaks_the_repeat_and_keeps_the_contrast() {
    use ae_render::terrain::tileset;
    let gpu = gpu();
    let on = tileset::temperate(64);
    for (material, layer, repeat) in [(6u8, 6usize, 20.0f32), (0, 0, 24.0)] {
        let shot = |t: tileset::Tileset| {
            let (pixels, project) = tileset_scene(
                &gpu,
                |_, _| 50.0,
                |_, _| material,
                Some(Arc::new(t)),
                false,
                300.0,
                None,
            );
            let (x0, y0) = project(0.0, 0.0, 50.0);
            let (x1, y1) = project(512.0, 512.0, 50.0);
            assert!(
                x0.min(x1) < 0.0 && x0.max(x1) > 320.0 && y0.min(y1) < 0.0 && y0.max(y1) > 240.0,
                "the map fills the view"
            );
            let luma: Vec<f64> = pixels
                .chunks(4)
                .map(|p| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64)
                .collect();
            let period = (project(256.0 + repeat, 256.0, 50.0).0 - project(256.0, 256.0, 50.0).0).abs();
            (luma, period)
        };
        assert!(on.anti_tiling > 0.0, "tileset anti-tiling is on by default");
        let at_degree = |degree: f32| {
            let mut t = on.clone();
            t.anti_tiling = degree;
            shot(t)
        };
        let (luma_off, period) = at_degree(0.0);
        let (luma_soft, _) = at_degree(0.1);
        let (luma_on, _) = at_degree(1.0);
        // A margin inside the map (it spans the view and more).
        let (w, h, m) = (320usize, 240usize, 8usize);
        let stats = |l: &[f64]| {
            let px: Vec<f64> = (m..h - m)
                .flat_map(|y| (m..w - m).map(move |x| (x, y)))
                .map(|(x, y)| l[y * w + x])
                .collect();
            let mean = px.iter().sum::<f64>() / px.len() as f64;
            (
                mean,
                (px.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / px.len() as f64).sqrt(),
            )
        };
        // Correlation of the image with itself shifted by `d` pixels (x, y).
        let corr = |l: &[f64], (dx, dy): (usize, usize)| {
            let pairs: Vec<(f64, f64)> = (m..h - m - dy)
                .flat_map(|y| (m..w - m - dx).map(move |x| (x, y)))
                .map(|(x, y)| (l[y * w + x], l[(y + dy) * w + x + dx]))
                .collect();
            let n = pairs.len() as f64;
            let (ma, mb) = (
                pairs.iter().map(|p| p.0).sum::<f64>() / n,
                pairs.iter().map(|p| p.1).sum::<f64>() / n,
            );
            let cov = pairs.iter().map(|p| (p.0 - ma) * (p.1 - mb)).sum::<f64>();
            let va = pairs.iter().map(|p| (p.0 - ma).powi(2)).sum::<f64>();
            let vb = pairs.iter().map(|p| (p.1 - mb).powi(2)).sum::<f64>();
            cov / (va * vb).sqrt()
        };
        let p = period.round() as usize;
        let at = |l: &[f64]| (corr(l, (p, 0)) + corr(l, (0, p))) / 2.0;
        let ((mean_off, sd_off), (mean_on, sd_on)) = (stats(&luma_off), stats(&luma_on));
        // At one and a half repeats the tiling adds nothing: what remains
        // there is the broad variation (macro brightness), the baseline.
        let q = (period * 1.5).round() as usize;
        let base = |l: &[f64]| (corr(l, (q, 0)) + corr(l, (0, q))) / 2.0;
        let (c_off, c_on) = (at(&luma_off) - base(&luma_off), at(&luma_on) - base(&luma_on));
        let c_soft = at(&luma_soft) - base(&luma_soft);
        eprintln!(
            "layer {layer}: period {period:.1} px; correlation at one repeat above the baseline {c_off:.3} off, {c_soft:.3} at degree 0.1, {c_on:.3} on (baselines {:.3}, {:.3}, {:.3});",
            base(&luma_off),
            base(&luma_soft),
            base(&luma_on)
        );
        eprintln!("  mean {mean_off:.1} / {mean_on:.1}; sd {sd_off:.2} / {sd_on:.2}");
        // Measured above the baseline: mud 0.56 off, 0.21 at degree 0.1,
        // 0.02 on; grass 0.41 off, 0.25, 0.11 on (its low contrast leaves
        // the macro variation a baseline of 0.4-0.5).
        assert!(c_off > 0.3, "without anti-tiling the field repeats: {c_off:.3}");
        assert!(
            c_soft < c_off * 0.7 && c_soft > c_on,
            "a tenth of the degree is partway: {c_soft:.3} against {c_off:.3} off and {c_on:.3} on"
        );
        assert!(
            c_on < c_off * 0.35,
            "anti-tiling breaks the repeat: {c_on:.3} against {c_off:.3}"
        );
        assert!(
            (mean_on - mean_off).abs() < mean_off * 0.02,
            "the mean holds: {mean_on:.1} against {mean_off:.1}"
        );
        assert!(
            (sd_on / sd_off - 1.0).abs() < 0.08,
            "the contrast holds: sd {sd_on:.2} against {sd_off:.2}"
        );
    }
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
) -> (
    (u32, u32),
    ae_render::units::InstanceStream,
    ae_render::units::UnitFrame,
) {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitFrame, UnitsSetup};

    let heightmap = Heightmap {
        width: 129,
        height: 129,
        data: vec![0.5; 129 * 129],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
    for &(x, y, radius, class) in rows {
        for (a, word) in [
            x.to_bits(),
            y.to_bits(),
            x.to_bits(),
            y.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            radius.to_bits(),
            class,
        ]
        .into_iter()
        .enumerate()
        {
            columns[a].push(word);
        }
    }
    let stream = InstanceStream {
        tick: 1,
        published: std::time::Instant::now(),
        tick_seconds: 0.0,
        sim_seconds: 0.0,
        columns: at_rest(columns),
        anim: AnimWrites::default(),
    };
    let feed = Arc::new(UnitFeed::default());
    feed.publish(Arc::new(stream.clone()));
    let classes = vec![UnitClass {
        gamut: ae_render::colorspace::Gamut::Bt709,
        color: [1.0, 0.0, 0.0],
        floats: false,
    }];
    let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(UnitsSetup {
        feed,
        capacity,
        classes,
        min_pixels,
        models: Vec::new(),
        rigs: Vec::new(),
    });
    let cam = {
        let view = app.view();
        let mut view = view.lock().unwrap();
        camera(&mut view.camera);
        view.camera
    };
    let extent = vk::Extent2D {
        width: 320,
        height: 200,
    };
    app.create_offscreen(gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
    run_headless(gpu, &mut app, target, 4, config).unwrap();
    let counts = app.last_unit_counts().unwrap();
    ae_render::thread::RenderApp::destroy(&mut app, gpu);
    let view = lod_view(&source, &cam, extent.width, extent.height);
    let frame = UnitFrame {
        view_proj: view.view_proj,
        cam_pos: view.pos,
        proj_scale: view.proj_scale,
        fog: [0.0; 4],
        shadows: false,
        unit_shadows: None,
        photometry: [
            ae_render::lighting::exposure(ae_render::lighting::DEFAULT_EV100),
            0.0,
            ae_render::lighting::SUN_ILLUMINANCE,
            ae_render::lighting::SKY_ILLUMINANCE,
        ],
    };
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
    let mut candidates: Vec<(f32, f32, f32, u32)> = (0..6000)
        .map(|_| {
            (
                -400.0 + 1800.0 * random(),
                -400.0 + 1800.0 * random(),
                1.0 + 11.0 * random(),
                0,
            )
        })
        .collect();
    candidates.extend((0..1500).map(|_| {
        (
            -6000.0 + 12000.0 * random(),
            -6000.0 + 12000.0 * random(),
            0.3 + 0.7 * random(),
            0,
        )
    }));
    // The frame does not depend on the units: probe it with one.
    let (_, _, frame) = cull_on_gpu(&gpu, &candidates[..1], 1, MIN_PIXELS, camera);
    let probe = ae_render::units::InstanceStream {
        tick: 0,
        published: std::time::Instant::now(),
        tick_seconds: 0.0,
        sim_seconds: 0.0,
        columns: std::array::from_fn(|a| {
            candidates
                .iter()
                .map(|&(x, y, r, c)| {
                    [
                        x.to_bits(),
                        y.to_bits(),
                        x.to_bits(),
                        y.to_bits(),
                        0.0f32.to_bits(),
                        0.0f32.to_bits(),
                        0.0f32.to_bits(),
                        1.0f32.to_bits(),
                        0.0f32.to_bits(),
                        0.0f32.to_bits(),
                        0.0f32.to_bits(),
                        1.0f32.to_bits(),
                        r.to_bits(),
                        c,
                        ae_render::units::NO_POSE,
                        0, // Surfaced.
                        0, // Unfiltered.
                    ][a]
                })
                .collect()
        }),
        anim: ae_render::units::AnimWrites::default(),
    };
    let margins = cull_margins(&probe, &frame, MIN_PIXELS, 1.0, ground);
    let rows: Vec<(f32, f32, f32, u32)> = candidates
        .iter()
        .zip(&margins)
        .filter(|(_, m)| m.0.abs() > 0.5)
        .map(|(c, _)| *c)
        .take(3000)
        .collect();
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
    let by_min_size = margins
        .iter()
        .zip(&unenlarged)
        .filter(|(m, u)| m.0 >= 0.0 && u.0 < 0.0)
        .count();
    eprintln!(
        "{uploaded} rows, GPU kept {visible}, CPU keeps {kept}; culled by plane {culled_by:?}; {straddling} kept with the centre outside; {by_min_size} kept only at the minimum size"
    );
    assert_eq!(uploaded, 3000);
    assert_eq!(visible, kept, "GPU and CPU cull agree");
    assert!(kept > 300 && kept < 2700, "a mix of kept and culled units: {kept}");
    assert!(
        culled_by[..4].iter().all(|&n| n > 20),
        "every side plane culls some: {culled_by:?}"
    );
    assert!(straddling > 10, "units across the view's edge are kept: {straddling}");
    assert!(by_min_size > 5, "far units kept by the minimum size: {by_min_size}");

    // More rows than capacity: the first 64 are uploaded; all in view.
    let crowd: Vec<(f32, f32, f32, u32)> = (0..100)
        .map(|i| (500.0 + (i % 10) as f32 * 4.0, 500.0 + (i / 10) as f32 * 4.0, 1.0, 0))
        .collect();
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

/// The default module's HDR10 readback is the native graph's: the same
/// display intermediate and `post.encode` (M13 stage 6), within one PQ code
/// per channel (the two paths bake the same plan, so the tolerance is for
/// arithmetic order alone).
#[test]
#[ignore = "needs a Vulkan GPU"]
fn the_default_module_renders_like_the_native_graph_in_hdr10() {
    use ae_render::encode::{DisplayLevels, OutputEncoding};

    let gpu = gpu();
    let levels = DisplayLevels {
        paper_white: 203.0,
        peak: 361.0,
    };
    let wasm = ae_render::module::build("default").expect("build render/default");
    let (native, native_counts) = scene_pixels_with(&gpu, None, 16, Some((OutputEncoding::Hdr10Pq, levels)));
    let (from_module, module_counts) = scene_pixels_with(&gpu, Some(wasm), 16, Some((OutputEncoding::Hdr10Pq, levels)));
    assert_eq!(native_counts, module_counts, "same unit counts");
    assert_eq!(native.len(), from_module.len());
    let mut worst = 0i32;
    let mut differing = 0usize;
    for (a, b) in native.chunks(4).zip(from_module.chunks(4)) {
        let (ua, ub) = (
            u32::from_le_bytes(a.try_into().unwrap()),
            u32::from_le_bytes(b.try_into().unwrap()),
        );
        let mut pixel = 0;
        for shift in [0u32, 10, 20] {
            let d = (((ua >> shift) & 0x3ff) as i32 - ((ub >> shift) & 0x3ff) as i32).abs();
            pixel = pixel.max(d);
            worst = worst.max(d);
        }
        if pixel > 0 {
            differing += 1;
        }
    }
    eprintln!(
        "HDR10 native vs default module: worst channel {worst} PQ code(s), {differing} of {} pixels differ",
        native.len() / 4
    );
    assert!(worst <= 1, "the module's HDR10 frame differs by {worst} PQ codes");
    assert_clean(&gpu);
}

/// Replacing the graph while HDR (M13 stage 6): default -> post in HDR10
/// readback, one rebake, the grade changes the frame, validation silent; and
/// the vignette flag (a pass that always runs, multiplying by zero when off)
/// darkens the corners without a rebake.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn render_modules_swap_graphs_in_hdr10() {
    use ae_render::encode::{DisplayLevels, OutputEncoding};
    use ae_render::thread::{Headless, TargetInfo};

    let gpu = gpu();
    let levels = DisplayLevels {
        paper_white: 203.0,
        peak: 361.0,
    };
    let default = ae_render::module::build("default").expect("build render/default");
    let post = ae_render::module::build("post").expect("build render/post");
    let mut app = scene_app_with(&gpu, Some(default), Some((OutputEncoding::Hdr10Pq, levels)));
    let target = TargetInfo {
        extent: SCENE_EXTENT,
        format: vk::Format::A2B10G10R10_UNORM_PACK32,
        color_space: vk::ColorSpaceKHR::HDR10_ST2084_EXT,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
    let mut run = Headless::new(&gpu, &mut app, target, config).unwrap();
    let reload = app.module_reload();
    let mut step = |app: &mut ae_render::terrain::app::TerrainApp, frames: usize| -> (Vec<u8>, u64) {
        run.frames(app, frames).unwrap();
        run.wait().unwrap();
        (app.readback().unwrap(), run.stats().bakes)
    };
    let (w, h) = (SCENE_EXTENT.width, SCENE_EXTENT.height);
    let corner = |p: &[u8]| -> f32 {
        // Display units from the PQ code (the BT.2020 channels are near the
        // achromatic axis here, so the green channel's luminance is enough).
        let at = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            let u = u32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]]);
            ae_render::encode::pq_eotf(((u >> 10) & 0x3ff) as f32 / 1023.0)
        };
        (at(8, 8) + at(w - 8, 8) + at(8, h - 8) + at(w - 8, h - 8)) / 4.0
    };

    let (plain, bakes) = step(&mut app, 12);
    assert_eq!(bakes, 1);
    *reload.lock().unwrap() = Some(post);
    let (graded, bakes) = step(&mut app, 12);
    assert_eq!(bakes, 2, "the post module's graph rebaked once");
    let changed = plain.iter().zip(&graded).filter(|(a, b)| a.abs_diff(**b) > 2).count();
    assert!(
        changed > plain.len() / 10,
        "the grade changes the HDR frame: {changed} of {} bytes",
        plain.len()
    );

    // The vignette: the flag switches a uniform, not the graph.
    app.view().lock().unwrap().module_flags = 1;
    let (vignetted, bakes) = step(&mut app, 4);
    assert_eq!(bakes, 2, "switching the HDR vignette does not rebake");
    eprintln!(
        "HDR10 corners {:.1} -> {:.1} nits, centre {:?}",
        corner(&graded),
        corner(&vignetted),
        hdr10_nits(&graded, ((h / 2) * w + w / 2) as usize)
    );
    assert!(
        corner(&vignetted) < corner(&graded) * 0.8,
        "the vignette darkens the corners"
    );
    drop(run);
    ae_render::thread::RenderApp::destroy(&mut app, &gpu);
    assert_clean(&gpu);
}

/// The post module's grade runs on the display, above paper white (M13
/// stage 6): the real grade SPIR-V (from the built module) followed by
/// `post.encode` decodes to the grade's maths at display 1.5 and 2.0 paper
/// white, and saturation keeps luminance. The native (ungraded) encode of
/// the same values is reported beside it.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn the_post_grade_works_above_paper_white() {
    use ae_render::colorspace::{Gamut, WorkingSpace};
    use ae_render::encode::{DisplayLevels, OutputEncoding};
    use ae_render_abi::module::{Program, ProgramTable};

    let gpu = gpu();
    let wasm = ae_render::module::build("post").expect("build render/post");
    let engine = ae_render::module::engine();
    let mut module = ae_render::module::RenderModule::load(&engine, &wasm).unwrap();
    let target = declare_sdr(4, 4, vk::Format::B8G8R8A8_SRGB, 0);
    let declared = module.declare(&target).unwrap().expect("declares in SDR");
    let programs = ProgramTable::decode(&declared.programs).unwrap();
    let grade = programs
        .iter()
        .find_map(|p| match *p {
            Program::Fullscreen {
                label: "grade", spirv, ..
            } => Some(spirv.to_vec()),
            _ => None,
        })
        .expect("the post module's grade program");

    // The grade's constants (its uniform block in `frame`).
    let (contrast, saturation, tint) = (1.12f32, 1.15f32, [1.04f32, 1.0, 0.94]);
    let expected = |c: [f32; 3]| -> [f32; 3] {
        let luma = y709(c);
        let s: [f32; 3] = std::array::from_fn(|k| luma + saturation * (c[k] - luma));
        let t: [f32; 3] = s.map(|v| (v - 0.18) * contrast + 0.18);
        [t[0] * tint[0], t[1] * tint[1], t[2] * tint[2]]
    };
    let levels = DisplayLevels {
        paper_white: 203.0,
        peak: 1000.0,
    };
    let values: [[f32; 4]; 4] = [
        [1.5, 1.5, 1.5, 1.0],
        [2.0, 2.0, 2.0, 1.0],
        [1.8, 1.2, 0.9, 1.0],
        [2.4, 1.6, 1.2, 1.0],
    ];
    let graded = grade_values(&gpu, &grade, &values);
    let native = encode_values(
        &gpu,
        OutputEncoding::Hdr10Pq,
        vk::Extent2D { width: 4, height: 4 },
        levels,
        WorkingSpace::Bt709,
        Gamut::P3,
        &values,
    );
    let display = |frame: &[u8]| -> [f32; 3] {
        // BT.2020 nits back to the working space's paper-white units (P3
        // contains BT.709, so the map is a pure re-expression).
        let c = ae_render::colorspace::convert(Gamut::Bt2020, Gamut::Bt709, hdr10_nits(frame, 0));
        c.map(|v| v / levels.paper_white)
    };
    let mut worst = 0.0f32;
    for (k, (want, frame)) in values.iter().zip(&native).enumerate() {
        let (input, graded, plain) = ([want[0], want[1], want[2]], graded[k], display(frame));
        let expected = expected(input);
        let delta = (0..3).map(|c| (graded[c] - expected[c]).abs()).fold(0.0, f32::max);
        worst = worst.max(delta);
        eprintln!(
            "display {:.1} input {:?}: native {:?} | graded {:?} | expected {:?} (max {:.4} off); luma native {:.4} -> graded {:.4}",
            input[0],
            input.map(|v| (v * 100.0).round() / 100.0),
            plain.map(|v| (v * 1000.0).round() / 1000.0),
            graded.map(|v| (v * 1000.0).round() / 1000.0),
            expected.map(|v| (v * 1000.0).round() / 1000.0),
            delta,
            y709(plain),
            y709(graded)
        );
        // Saturation keeps luminance before the contrast: the graded
        // luminance is the input's through contrast and the tint.
        let luma_expected = y709(expected);
        assert!(
            (y709(graded) - luma_expected).abs() < 0.01,
            "row {k}: luminance {} vs {}",
            y709(graded),
            luma_expected
        );
    }
    assert!(worst < 0.01, "the decoded grade is {worst} off the maths");
    assert_clean(&gpu);
}

/// The post module's grade over a clear value, then `post.encode` to HDR10,
/// read back as the display's values in the working space (paper-white
/// units).
fn grade_values(gpu: &Arc<GpuDevice>, grade: &[u8], values: &[[f32; 4]]) -> Vec<[f32; 3]> {
    use ae_framegraph::{FrameGraph, LoadOp, PassKind, ResourceId, Role, ShaderStage};
    use ae_render::colorspace::{Gamut, WorkingSpace};
    use ae_render::encode::{DISPLAY_FORMAT, DisplayLevels, Encode, OutputEncoding};
    use ae_render::executor::{Binding, BufferBinding, Executor, ExecutorConfig, ImageBinding, PassContext, Recorder};
    use ae_render::fullscreen::FullscreenProgram;

    struct Paint {
        display: ResourceId,
        graded: ResourceId,
        target: ResourceId,
        buffer: ResourceId,
        program: FullscreenProgram,
        encode: Encode,
        sampler_index: u32,
        uniforms: [u8; 32],
        value: [f32; 4],
        levels: DisplayLevels,
    }
    impl Recorder<u32> for Paint {
        fn clear_color(&self, _pass: &u32, resource: ResourceId) -> [f32; 4] {
            if resource == self.display { self.value } else { [0.0; 4] }
        }
        fn record(&mut self, pass: &u32, ctx: &PassContext<'_>) -> anyhow::Result<()> {
            match *pass {
                1 => {
                    self.program.write_uniforms(ctx.slot, &self.uniforms);
                    self.program.record(
                        ctx,
                        &[ctx.image(self.display)?],
                        ae_render::fullscreen::Sampler {
                            index: self.sampler_index,
                        },
                    )?;
                }
                2 => {
                    self.encode.write(
                        ctx.slot,
                        OutputEncoding::Hdr10Pq,
                        self.levels,
                        WorkingSpace::Bt709,
                        Gamut::P3,
                    );
                    self.encode.record(ctx, ctx.image(self.graded)?, self.sampler_index)?;
                }
                4 => ae_render::demo::record_readback(ctx, ctx.image(self.target)?, ctx.buffer(self.buffer)?),
                _ => {}
            }
            Ok(())
        }
    }

    let extent = vk::Extent2D { width: 4, height: 4 };
    let levels = DisplayLevels {
        paper_white: 203.0,
        peak: 1000.0,
    };
    let format = OutputEncoding::Hdr10Pq.readback_format();
    let display_desc = ImageDesc::color_2d(
        DISPLAY_FORMAT,
        extent.width,
        extent.height,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
    );
    let target_desc = ImageDesc::color_2d(
        format,
        extent.width,
        extent.height,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    let buffer_desc = BufferDesc {
        size: extent.width as u64 * extent.height as u64 * 4,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
    };
    let mut graph = FrameGraph::<u32>::new();
    let display = graph.image("display", display_desc, ResourceClass::External);
    let graded = graph.image("graded", display_desc, ResourceClass::External);
    let target = graph.image("target", target_desc, ResourceClass::External);
    let buffer = graph.buffer("readback", buffer_desc, ResourceClass::External);
    graph
        .pass("fill", PassKind::Graphics, 0)
        .color(display, LoadOp::Clear)
        .id();
    graph
        .pass("grade", PassKind::Graphics, 1)
        .sampled(display, ShaderStage::Fragment)
        .color(graded, LoadOp::DontCare)
        .id();
    graph
        .pass("encode", PassKind::Graphics, 2)
        .sampled(graded, ShaderStage::Fragment)
        .color(target, LoadOp::DontCare)
        .id();
    graph
        .pass("copy", PassKind::Transfer, 3)
        .access(target, Role::TransferSrc)
        .access(buffer, Role::TransferDst)
        .id();
    graph
        .pass("host read", PassKind::Transfer, 4)
        .access(buffer, Role::HostRead)
        .root()
        .id();

    let mut image = OwnedImage::new(gpu, display_desc, "display").unwrap();
    let mut graded_image = OwnedImage::new(gpu, display_desc, "graded").unwrap();
    let mut out = OwnedImage::new(gpu, target_desc, "target").unwrap();
    let mut readback = OwnedBuffer::new(gpu, buffer_desc, MemoryLocation::GpuToCpu, "readback").unwrap();
    let mut executor = Executor::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    let display_handle = executor
        .bindless_mut()
        .sampled_2d(image.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .unwrap();
    let graded_handle = executor
        .bindless_mut()
        .sampled_2d(graded_image.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .unwrap();
    let bindings = [
        Binding::Image(ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: image.image,
            view: image.view,
            format: display_desc.format,
            extent,
            range: image.range(),
            identity: (0x6EA7E << 40) | 1,
            sampled: Some(display_handle.0.index),
            storage: None,
        }),
        Binding::Image(ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: graded_image.image,
            view: graded_image.view,
            format: display_desc.format,
            extent,
            range: graded_image.range(),
            identity: (0x6EA7E << 40) | 2,
            sampled: Some(graded_handle.0.index),
            storage: None,
        }),
        Binding::Image(ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: out.image,
            view: out.view,
            format: target_desc.format,
            extent,
            range: out.range(),
            identity: (0x6EA7E << 40) | 3,
            sampled: None,
            storage: None,
        }),
        Binding::Buffer(BufferBinding {
            buffer: readback.buffer,
            offset: 0,
            size: buffer_desc.size,
            identity: (0x6EA7E << 40) | 4,
            storage: None,
        }),
    ];
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST);
    let sampler_handle = executor.bindless_mut().sampler(&sampler_info).unwrap();
    let program = FullscreenProgram::new(
        gpu,
        executor.bindless_mut(),
        "grade",
        grade,
        32,
        1,
        &[DISPLAY_FORMAT],
        2,
    )
    .unwrap();
    let sampler_index = sampler_handle.index;
    let encode = Encode::new(gpu, executor.bindless_mut(), OutputEncoding::Hdr10Pq, format, 2).unwrap();
    let uniforms: Vec<u8> = [1.04f32, 1.0, 0.94, 1.0, 1.12, 1.15, 0.0, 0.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut paint = Paint {
        display,
        graded,
        target,
        buffer,
        program,
        encode,
        sampler_index,
        uniforms: uniforms.try_into().unwrap(),
        value: [0.0; 4],
        levels,
    };
    executor.set_graph(graph).unwrap();
    let mut frames = Vec::new();
    for value in values {
        paint.value = *value;
        let token = executor.begin_frame().unwrap();
        executor
            .execute(
                token,
                &mut paint,
                &|resource| bindings.get(resource.index()).copied(),
                &[],
                &[],
            )
            .unwrap();
        executor.wait(token).unwrap();
        frames.push(readback.mapped().unwrap().to_vec());
    }
    paint.program.destroy(gpu);
    paint.encode.destroy(gpu);
    image.destroy(gpu);
    graded_image.destroy(gpu);
    out.destroy(gpu);
    readback.destroy(gpu);
    frames
        .iter()
        .map(|frame| {
            let c = ae_render::colorspace::convert(Gamut::Bt2020, Gamut::Bt709, hdr10_nits(frame, 0));
            c.map(|v| v / levels.paper_white)
        })
        .collect()
}

/// A generated map with a few hundred units, offscreen at 320x200,
/// natively or driven by `module`.
fn scene_app(gpu: &Arc<GpuDevice>, module: Option<Vec<u8>>) -> ae_render::terrain::app::TerrainApp {
    scene_app_with(gpu, module, None)
}

/// [`scene_app`], with an HDR display spec (M13: the readback path can
/// encode HDR10 or scRGB too).
fn scene_app_with(
    gpu: &Arc<GpuDevice>,
    module: Option<Vec<u8>>,
    display: Option<(ae_render::encode::OutputEncoding, ae_render::encode::DisplayLevels)>,
) -> ae_render::terrain::app::TerrainApp {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{build_pyramid, generate};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};

    let heightmap = generate(257, 257, 3);
    let pyramid = build_pyramid(&heightmap, 300.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 300.0, 0.0, Some(40.0));
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
    for i in 0..400u32 {
        let (x, y) = (300.0 + (i % 20) as f32 * 70.0, 300.0 + (i / 20) as f32 * 70.0);
        for (a, word) in [
            x.to_bits(),
            y.to_bits(),
            x.to_bits(),
            y.to_bits(),
            0.0f32.to_bits(),
            (-0.5f32 / (1.25f32).sqrt()).to_bits(),
            0.0f32.to_bits(),
            (1.0f32 / (1.25f32).sqrt()).to_bits(),
            0.0f32.to_bits(),
            (-0.5f32 / (1.25f32).sqrt()).to_bits(),
            0.0f32.to_bits(),
            (1.0f32 / (1.25f32).sqrt()).to_bits(),
            6.0f32.to_bits(),
            i % 3,
        ]
        .into_iter()
        .enumerate()
        {
            columns[a].push(word);
        }
    }
    let feed = Arc::new(UnitFeed::default());
    feed.publish(Arc::new(InstanceStream {
        tick: 1,
        published: std::time::Instant::now(),
        tick_seconds: 0.0,
        sim_seconds: 0.0,
        columns: at_rest(columns),
        anim: AnimWrites::default(),
    }));
    let classes = vec![
        UnitClass {
            gamut: ae_render::colorspace::Gamut::Bt709,
            color: [1.0, 0.8, 0.1],
            floats: false,
        },
        UnitClass {
            gamut: ae_render::colorspace::Gamut::Bt709,
            color: [0.9, 0.9, 0.95],
            floats: true,
        },
        UnitClass {
            gamut: ae_render::colorspace::Gamut::Bt709,
            color: [0.8, 0.1, 0.1],
            floats: false,
        },
    ];
    let mut app = TerrainApp::new(source, TerrainOutput::Readback).with_units(UnitsSetup {
        feed,
        capacity: 512,
        classes,
        min_pixels: 5.0,
        models: Vec::new(),
        rigs: Vec::new(),
    });
    if let Some((encoding, levels)) = display {
        app = app.with_display(encoding, levels);
    }
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

const SCENE_EXTENT: vk::Extent2D = vk::Extent2D {
    width: 320,
    height: 200,
};

/// The scene of [`scene_app`] rendered for `frames` frames: the pixels and
/// the unit counts (rows, visible).
fn scene_pixels(gpu: &Arc<GpuDevice>, module: Option<Vec<u8>>, frames: usize) -> (Vec<u8>, (u32, u32)) {
    scene_pixels_with(gpu, module, frames, None)
}

/// [`scene_pixels`] for an output encoding (the target's format and colour
/// space follow it; the readback image uses [`ae_render::encode::OutputEncoding::readback_format`]).
fn scene_pixels_with(
    gpu: &Arc<GpuDevice>,
    module: Option<Vec<u8>>,
    frames: usize,
    display: Option<(ae_render::encode::OutputEncoding, ae_render::encode::DisplayLevels)>,
) -> (Vec<u8>, (u32, u32)) {
    use ae_render::encode::OutputEncoding;
    use ae_render::thread::{TargetInfo, run_headless};
    let mut app = scene_app_with(gpu, module, display);
    let (format, color_space) = match display.map(|(e, _)| e).unwrap_or_default() {
        OutputEncoding::Sdr => (vk::Format::R8G8B8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
        OutputEncoding::Hdr10Pq => (
            vk::Format::A2B10G10R10_UNORM_PACK32,
            vk::ColorSpaceKHR::HDR10_ST2084_EXT,
        ),
        OutputEncoding::ScRgb => (
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
        ),
    };
    let target = TargetInfo {
        extent: SCENE_EXTENT,
        format,
        color_space,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
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

/// An SDR declaration target for the reload tests.
fn declare_sdr(width: u32, height: u32, format: vk::Format, flags: u32) -> ae_render::module::DeclareTarget {
    use ae_render::encode::{DisplayLevels, OutputEncoding};
    ae_render::module::DeclareTarget {
        width,
        height,
        color_format: format,
        flags,
        encoding: OutputEncoding::Sdr,
        levels: DisplayLevels::default(),
        working: Default::default(),
        gamut: Default::default(),
    }
}

/// A WAT module that declares `graph` and `programs` (bytes for the test
/// scene's target) and traps in `frame`.
fn trapping_module(graph: &[u8], programs: &[u8]) -> Vec<u8> {
    let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("\\{b:02x}")).collect::<String>();
    let (g, p) = (4096u32, 4096 + graph.len() as u32);
    let words: Vec<u8> = [g, graph.len() as u32, p, programs.len() as u32]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    format!(
        r#"(module
            (memory (export "memory") 2)
            (data (i32.const 64) "{}")
            (data (i32.const {g}) "{}")
            (data (i32.const {p}) "{}")
            (func (export "ae_render_version") (result i32) i32.const 2)
            (func (export "ae_render_declare_input") (result i32) i32.const 512)
            (func (export "ae_render_declare") (result i32) i32.const 64)
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
    let target = TargetInfo {
        extent: SCENE_EXTENT,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
    let mut run = Headless::new(&gpu, &mut app, target, config).unwrap();
    let status = app.module_status();
    let reload = app.module_reload();
    let mut step = |app: &mut ae_render::terrain::app::TerrainApp, frames: usize| -> (Vec<u8>, u64) {
        run.frames(app, frames).unwrap();
        run.wait().unwrap();
        (app.readback().unwrap(), run.stats().bakes)
    };
    let (w, h) = (SCENE_EXTENT.width, SCENE_EXTENT.height);
    let corners = |p: &[u8]| {
        (mean_rgb(p, 0, 0, 24, 24)
            + mean_rgb(p, w - 24, 0, 24, 24)
            + mean_rgb(p, 0, h - 24, 24, 24)
            + mean_rgb(p, w - 24, h - 24, 24, 24))
            / 4.0
    };
    let centre = |p: &[u8]| mean_rgb(p, w / 2 - 20, h / 2 - 20, 40, 40);

    let (plain, bakes) = step(&mut app, 12);
    assert_eq!((bakes, status.lock().unwrap().loads), (1, 1));

    *reload.lock().unwrap() = Some(post.clone());
    let (graded, bakes) = step(&mut app, 12);
    assert_eq!(
        (bakes, status.lock().unwrap().loads),
        (2, 2),
        "the post module's graph rebaked once"
    );
    let changed = plain.iter().zip(&graded).filter(|(a, b)| a.abs_diff(**b) > 2).count();
    assert!(
        changed > plain.len() / 4,
        "the grade changes the picture: {changed} of {} bytes",
        plain.len()
    );
    assert!(
        (centre(&graded) - centre(&plain)).abs() < 40.0,
        "but keeps it recognisable"
    );

    app.view().lock().unwrap().module_flags = 1;
    let (vignetted, bakes) = step(&mut app, 4);
    assert_eq!(bakes, 2, "switching a conditional pass does not rebake");
    eprintln!(
        "corners {:.1} -> {:.1}, centre {:.1} -> {:.1}",
        corners(&graded),
        corners(&vignetted),
        centre(&graded),
        centre(&vignetted)
    );
    assert!(
        corners(&vignetted) < corners(&graded) * 0.6,
        "the vignette darkens the corners"
    );
    assert!(
        (centre(&vignetted) - centre(&graded)).abs() < 2.0,
        "and leaves the centre"
    );

    *reload.lock().unwrap() = Some(b"not a module".to_vec());
    let (kept, bakes) = step(&mut app, 4);
    {
        let s = status.lock().unwrap();
        assert_eq!((s.loads, s.errors.len()), (2, 1));
        assert!(s.errors[0].contains("reload rejected"), "{}", s.errors[0]);
    }
    assert_eq!(bakes, 2);
    let drift = kept.iter().zip(&vignetted).filter(|(a, b)| a.abs_diff(**b) > 2).count();
    assert!(
        drift < kept.len() / 100,
        "the post module keeps rendering: {drift} bytes moved"
    );

    // A graph running one fullscreen program in two passes (the grade of
    // the post module, scene -> mid -> target): rejected by name, the post
    // module keeps rendering.
    let shared = {
        use ae_render_abi::module::{Program, ProgramTable};
        use ae_render_abi::{Class, GraphBuilder, ImageSpec, LoadOp, PassKind, Role, ShaderStage};
        let engine = ae_render::module::engine();
        let mut m = ae_render::module::RenderModule::load(&engine, &post).unwrap();
        let flags = ae_render_abi::module::DECLARE_UNITS | ae_render_abi::module::DECLARE_READBACK;
        let declared = m
            .declare(&declare_sdr(w, h, vk::Format::R8G8B8A8_SRGB, flags))
            .unwrap()
            .unwrap();
        let tables = ProgramTable::decode(&declared.programs).unwrap();
        let grade = tables.iter().find_map(|p| match *p {
            Program::Fullscreen {
                label: "grade", spirv, ..
            } => Some(spirv.to_vec()),
            _ => None,
        });
        let srgb = vk::Format::R8G8B8A8_SRGB.as_raw() as u32;
        let (mut g, mut p) = (GraphBuilder::new(), ProgramTable::new());
        let leaves = g.buffer("terrain.leaves", 0, 0, Class::External);
        let indirect = g.buffer("terrain.indirect", 0, 0, Class::External);
        let depth = g.image(
            "depth",
            ImageSpec::depth_2d(vk::Format::D32_SFLOAT.as_raw() as u32, w, h, 0x20),
            Class::Transient,
        );
        let scene = g.image("scene", ae_render_abi::module::scene_spec(w, h), Class::Transient);
        let ldr = g.image("ldr", ImageSpec::color_2d(srgb, w, h, 0x10 | 0x4), Class::Transient);
        let mid = g.image("mid", ImageSpec::color_2d(srgb, w, h, 0x10 | 0x4), Class::Transient);
        let target = g.image(
            "readback.target",
            ImageSpec::color_2d(srgb, w, h, 0x10 | 0x1),
            Class::External,
        );
        let buffer = g.buffer("readback.buffer", 0, 0, Class::External);
        let lod = p.library("terrain.lod");
        g.pass("terrain lod", PassKind::Compute, lod)
            .access(leaves, Role::StorageBufferWrite(ShaderStage::Compute))
            .access(indirect, Role::StorageBufferWrite(ShaderStage::Compute))
            .id();
        let draw = p.library("terrain.draw");
        g.pass("terrain", PassKind::Graphics, draw)
            .access(leaves, Role::StorageBufferRead(ShaderStage::Vertex))
            .access(indirect, Role::IndirectBuffer)
            .color(scene, LoadOp::Clear)
            .depth(depth, LoadOp::Clear)
            .id();
        let tonemap = p.library("post.tonemap");
        g.pass("tonemap", PassKind::Graphics, tonemap)
            .sampled(scene, ShaderStage::Fragment)
            .color(ldr, LoadOp::DontCare)
            .id();
        let program = p.push(Program::Fullscreen {
            label: "grade",
            spirv: &grade.unwrap(),
            uniform_bytes: 32,
        });
        g.pass("grade once", PassKind::Graphics, program)
            .sampled(ldr, ShaderStage::Fragment)
            .color(mid, LoadOp::DontCare)
            .id();
        g.pass("grade twice", PassKind::Graphics, program)
            .sampled(mid, ShaderStage::Fragment)
            .color(target, LoadOp::DontCare)
            .id();
        let copy = p.library("readback.copy");
        g.pass("copy", PassKind::Transfer, copy)
            .access(target, Role::TransferSrc)
            .discard(buffer, Role::TransferDst)
            .id();
        let read = p.library("host.read");
        g.pass("host read", PassKind::Transfer, read)
            .access(buffer, Role::HostRead)
            .root()
            .id();
        trapping_module(&g.finish(), &p.finish())
    };
    *reload.lock().unwrap() = Some(shared);
    let (still, bakes) = step(&mut app, 4);
    {
        let s = status.lock().unwrap();
        assert_eq!((s.loads, s.errors.len()), (2, 2), "{:?}", s.errors);
        assert!(
            s.errors[1].contains("`grade once` and `grade twice`"),
            "{}",
            s.errors[1]
        );
    }
    assert_eq!(bakes, 2);
    let drift = still
        .iter()
        .zip(&vignetted)
        .filter(|(a, b)| a.abs_diff(**b) > 2)
        .count();
    assert!(
        drift < still.len() / 100,
        "the post module keeps rendering: {drift} bytes moved"
    );

    // The default module's declaration for this target, served by a module
    // that traps every frame.
    let declared = {
        let engine = ae_render::module::engine();
        let mut m = ae_render::module::RenderModule::load(&engine, &default).unwrap();
        let flags = ae_render_abi::module::DECLARE_UNITS | ae_render_abi::module::DECLARE_READBACK;
        m.declare(&declare_sdr(w, h, vk::Format::R8G8B8A8_SRGB, flags))
            .unwrap()
            .unwrap()
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
    assert!(
        off < trapped.len() / 100,
        "its graph renders like the default module's: {off} bytes differ"
    );

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
                        d.cmd_push_constants(
                            ctx.cmd,
                            self.layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            pipeline::push_bytes(r),
                        );
                        d.cmd_draw(ctx.cmd, 6, 1, 0, 0);
                    }
                }
            } else if *pass == PASSES {
                let (image, buffer) = (ctx.image(self.target)?, ctx.buffer(self.readback)?);
                let region = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: SIZE,
                        height: SIZE,
                        depth: 1,
                    });
                unsafe {
                    d.cmd_copy_image_to_buffer(
                        ctx.cmd,
                        image.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        buffer.buffer,
                        &[region],
                    )
                };
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
                    Rect {
                        rect: [x, y, x + w, y + w],
                        color: [next(), next(), next(), 1.0],
                        depth: [0.5, 0.0, 0.0, 0.0],
                    }
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
            glow_blend: Blend::Off,
        };
        let pipe = pipeline::graphics_pipeline(device, &desc).unwrap();
        let color = ImageDesc::color_2d(
            FORMAT,
            SIZE,
            SIZE,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        );
        let bytes = (SIZE * SIZE * 4) as u64;
        let buffer_desc = BufferDesc {
            size: bytes,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
        };
        let mut image = OwnedImage::new(gpu, color, "stress target").unwrap();
        let mut buffer = OwnedBuffer::new(gpu, buffer_desc, MemoryLocation::GpuToCpu, "stress readback").unwrap();
        let extent = vk::Extent2D {
            width: SIZE,
            height: SIZE,
        };
        let bindings = [
            Binding::Image(ImageBinding {
                initial: vk::ImageLayout::UNDEFINED,
                image: image.image,
                view: image.view,
                format: FORMAT,
                extent,
                range: image.range(),
                identity: 1 << 48,
                sampled: None,
                storage: None,
            }),
            Binding::Buffer(BufferBinding {
                buffer: buffer.buffer,
                offset: 0,
                size: bytes,
                identity: (1 << 48) + 1,
                storage: None,
            }),
        ];
        let mut graph = FrameGraph::<u16>::new();
        let target = graph.image("target", color, ResourceClass::External);
        let readback = graph.buffer("readback", buffer_desc, ResourceClass::External);
        for p in 0..PASSES {
            graph
                .pass(format!("draw {p}"), PassKind::Graphics, p)
                .color(target, if p == 0 { LoadOp::Clear } else { LoadOp::Load })
                .id();
        }
        graph
            .pass("copy", PassKind::Transfer, PASSES)
            .access(target, Role::TransferSrc)
            .discard(readback, Role::TransferDst)
            .id();
        graph
            .pass("host read", PassKind::Transfer, PASSES + 1)
            .access(readback, Role::HostRead)
            .root()
            .id();
        let mut executor = Executor::new(
            gpu.clone(),
            ExecutorConfig {
                record_threads: threads,
                ..ExecutorConfig::default()
            },
        )
        .unwrap();
        executor.set_graph(graph).unwrap();
        let mut recorder = Stress {
            pipeline: pipe,
            layout,
            rects: rects.clone(),
            target,
            readback,
        };
        let mut record_ms = Vec::new();
        for _ in 0..FRAMES {
            let token = executor.begin_frame().unwrap();
            executor
                .execute(
                    token,
                    &mut recorder,
                    &|resource| bindings.get(resource.index()).copied(),
                    &[],
                    &[],
                )
                .unwrap();
            executor.wait(token).unwrap();
            record_ms.push(executor.stats().record_ms);
        }
        let pixels = buffer.mapped().unwrap().to_vec();
        assert_eq!(
            executor.stats().parallel_frames,
            if threads > 1 { FRAMES as u64 } else { 0 },
            "{threads} threads"
        );
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
    assert!(
        serial == parallel,
        "parallel recording changed {} bytes",
        serial.iter().zip(&parallel).filter(|(a, b)| a != b).count()
    );
    assert_clean(&gpu);
    drop(gpu);

    // Recording time without the validation layer (it dominates otherwise).
    let instance = Arc::new(GpuInstance::new(&[], Some(false)).expect("Vulkan instance"));
    let plain = Arc::new(GpuDevice::new(instance, None).expect("Vulkan device"));
    for threads in [1, 2, 4, 8] {
        let (pixels, ms) = run(&plain, threads);
        assert!(pixels == serial, "{threads} threads without validation: other pixels");
        println!(
            "{threads} recording thread(s): {PASSES} passes x {DRAWS} draws recorded in {ms:.2} ms (median of {FRAMES} frames, no validation)"
        );
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
    let grid = VisionGrid {
        width: 16,
        height: 16,
        cell_size: 32.0,
    };
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
        f.publish(Arc::new(VisionFrame {
            tick: 1,
            los: plane(los),
            explored: plane(explored),
        }));
        Some(f)
    };
    let plain = scene(None);
    let all = scene(feed(&|_, _| true, &|_, _| true));
    assert!(
        plain == all,
        "full line of sight changed {} bytes",
        plain.iter().zip(&all).filter(|(a, b)| a != b).count()
    );

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
    let (left, right_a, right_b) = (
        luma(&half, 70, 140, 40, 200),
        luma(&half, 190, 250, 40, 100),
        luma(&half, 190, 250, 140, 200),
    );
    let (lit_left, plain_left) = (left, luma(&plain, 70, 140, 40, 200));
    let (dim, dark) = (right_a.max(right_b), right_a.min(right_b));
    println!("luma: left {left:.1} (no vision {plain_left:.1}), right explored {dim:.1}, never seen {dark:.1}");
    assert!(
        (lit_left - plain_left).abs() < 1.0,
        "line of sight draws as without vision"
    );
    // sRGB bytes: the shader's 0.2 and 0.5 factors (plus desaturation)
    // read as about 0.5 and 0.75 of the brightness.
    assert!(lit_left > 1.5 * dark, "never-seen ground is much darker");
    assert!(dim > dark * 1.2 && dim < lit_left * 0.9, "explored ground sits between");

    // A feed with no frame yet: all unexplored.
    let empty = scene(Some(Arc::new(VisionFeed::new(grid))));
    let unseen = luma(&empty, 190, 250, 40, 100).min(luma(&empty, 190, 250, 140, 200));
    assert!(
        (unseen - dark).abs() < 1.0,
        "before any vision frame, ground reads as never seen ({unseen:.1} vs {dark:.1})"
    );
    assert!(
        luma(&empty, 70, 140, 40, 200) < 0.7 * lit_left,
        "and the left is not lit"
    );
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
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Vec2, Vec4};

    let gpu = gpu();
    let n = 129;
    let island = |cx: usize, cz: usize| (88..=100).contains(&cx) && (88..=100).contains(&cz);
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: (0..n * n)
            .map(|i| {
                if i % n < 64 {
                    0.8
                } else if island(i % n, i / n) {
                    0.95
                } else {
                    0.4
                }
            })
            .collect(),
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, Some(60.0));
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let grid = VisionGrid {
        width: 32,
        height: 32,
        cell_size: 32.0,
    };
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
        let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
        for (&(x, y), class) in [land_unit, ship].iter().zip(0u32..) {
            for (a, word) in [
                x.to_bits(),
                y.to_bits(),
                x.to_bits(),
                y.to_bits(),
                0.0f32.to_bits(),
                0.0f32.to_bits(),
                0.0f32.to_bits(),
                1.0f32.to_bits(),
                0.0f32.to_bits(),
                0.0f32.to_bits(),
                0.0f32.to_bits(),
                1.0f32.to_bits(),
                8.0f32.to_bits(),
                class,
            ]
            .into_iter()
            .enumerate()
            {
                columns[a].push(word);
            }
        }
        let feed = Arc::new(UnitFeed::default());
        feed.publish(Arc::new(InstanceStream {
            tick: 1,
            published: std::time::Instant::now(),
            tick_seconds: 0.0,
            sim_seconds: 0.0,
            columns: at_rest(columns),
            anim: AnimWrites::default(),
        }));
        let classes = vec![
            UnitClass {
                gamut: ae_render::colorspace::Gamut::Bt709,
                color: [1.0, 0.02, 0.02],
                floats: false,
            },
            UnitClass {
                gamut: ae_render::colorspace::Gamut::Bt709,
                color: [0.02, 0.02, 1.0],
                floats: true,
            },
        ];
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(UnitsSetup {
            feed,
            capacity: 16,
            classes,
            min_pixels: 6.0,
            models: Vec::new(),
            rigs: Vec::new(),
        });
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
            feed.publish(Arc::new(VisionFrame {
                tick: 1,
                los,
                explored: vec![0; grid.words()],
            }));
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
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, r.frames, config).unwrap();
        let pixels = app.readback().unwrap();
        if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            let name = format!(
                "water-{}{}{}-{}.ppm",
                r.water as u8, r.view_water as u8, r.vision as u8, r.frames
            );
            std::fs::write(std::path::Path::new(&dir).join(name), ppm).unwrap();
        }
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let base = Run {
        water: false,
        view_water: true,
        vision: false,
        frames: 12,
    };
    let plain = render(base);
    let switched_off = render(Run {
        water: true,
        view_water: false,
        ..base
    });
    let on = render(Run { water: true, ..base });
    let later = render(Run {
        water: true,
        frames: 42,
        ..base
    });
    let fogged_plain = render(Run { vision: true, ..base });
    let fogged = render(Run {
        water: true,
        vision: true,
        ..base
    });

    let view_proj = lod_view(&source, &camera.unwrap(), extent.width, extent.height).view_proj;
    let at = |x: f32, y: f32, h: f32| -> Option<usize> {
        let clip = view_proj * Vec4::new(x, h, y, 1.0);
        let (u, v) = (
            (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
            (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
        );
        (clip.w > 0.0 && u >= 2.0 && v >= 2.0 && u < extent.width as f32 - 2.0 && v < extent.height as f32 - 2.0)
            .then(|| (v as usize * extent.width as usize + u as usize) * 4)
    };
    // Points on the land (clear of the cliff and the land unit) and on the
    // water (clear of the cliff's foot and the ship).
    let near = |x: f32, y: f32, (ux, uy): (f32, f32)| (x - ux).abs() < 40.0 && (y - uy).abs() < 40.0;
    let points = |xs: std::ops::RangeInclusive<u32>,
                  ys: std::ops::RangeInclusive<u32>,
                  step: usize,
                  h: f32,
                  unit: (f32, f32)|
     -> Vec<usize> {
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
    assert!(
        land.len() > 300 && water.len() > 300,
        "{} land and {} water points in view",
        land.len(),
        water.len()
    );
    let rgb = |p: &[u8], i: usize| [p[i], p[i + 1], p[i + 2]];
    let changed = |a: &[u8], b: &[u8], at: &[usize]| at.iter().filter(|&&i| rgb(a, i) != rgb(b, i)).count();
    let mean_diff = |a: &[u8], b: &[u8], at: &[usize]| {
        at.iter()
            .map(|&i| (0..3).map(|c| (a[i + c] as f64 - b[i + c] as f64).abs()).sum::<f64>())
            .sum::<f64>()
            / at.len() as f64
    };

    let off_bytes = plain.iter().zip(&switched_off).filter(|(a, b)| a != b).count();
    assert_eq!(off_bytes, 0, "water switched off draws as no water");
    assert_eq!(changed(&plain, &on, &land), 0, "land points under the water surface");
    let wet = changed(&plain, &on, &water);
    println!(
        "water: {wet} of {} water points changed; mean difference {:.1}",
        water.len(),
        mean_diff(&plain, &on, &water)
    );
    assert!(
        wet * 10 >= water.len() * 9,
        "the surface covers the water: {wet} of {}",
        water.len()
    );

    let ship_at = at(ship.0, ship.1, 61.0).expect("the ship is in view");
    let [r, g, b] = rgb(&on, ship_at).map(|c| c as i32);
    assert!(
        b > 2 * r && b > 2 * g,
        "the ship keeps its colour over the water: {r} {g} {b}"
    );

    assert_eq!(
        changed(&fogged_plain, &fogged, &land),
        0,
        "land in sight: unchanged by the water"
    );
    let luma_at = |pixels: &[u8], at: &[usize]| -> f64 {
        at.iter()
            .map(|&i| 0.3 * pixels[i] as f64 + 0.59 * pixels[i + 1] as f64 + 0.11 * pixels[i + 2] as f64)
            .sum::<f64>()
            / at.len() as f64
    };
    let (seen, unseen) = (luma_at(&on, &water), luma_at(&fogged, &water));
    println!("water in sight {seen:.1} luma, never seen {unseen:.1}");
    assert!(
        unseen < 0.8 * seen,
        "out of sight the water is dimmed: {unseen:.1} vs {seen:.1}"
    );

    assert_eq!(
        changed(&on, &later, &land),
        0,
        "half a second later the land is the same"
    );
    let moved = changed(&on, &later, &water);
    // At this distance the waves move the water by under one code: every
    // change is a single code (none saturates), so the count is how many
    // pixels cross a rounding step. 34 of 1678 measured (2.0%) with no tone
    // curve; it was 105 under Lottes' curve and over a fifteenth under
    // ACES's, whose slopes here differed.
    assert!(
        moved * 60 >= water.len(),
        "half a second later the waves have moved: {moved} of {}",
        water.len()
    );
    assert_clean(&gpu);
}

/// Pausing holds the renderer's clocks: the water after a pause is exactly
/// the water at the frame the pause began, however many frames are drawn
/// while paused, while an unpaused run of the same length has moved on.
/// (Decals and the particles read the same frame time; the unit stream is
/// stopped by the sim thread's pause.)
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn pause_holds_the_water_clock() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{Headless, RenderApp, TargetInfo};
    use glam::Vec2;

    let gpu = gpu();
    let (n, cell, scale, water) = (65usize, 16.0f32, 100.0f32, 60.0f32);
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![0.4; n * n],
    };
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, cell, scale, 0.0, Some(water));
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };

    let render = |pause_at: Option<usize>, frames: usize| -> Vec<u8> {
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_water();
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = Vec2::new(512.0, 512.0);
            view.camera.distance = 700.0;
            view.camera.pitch = 0.8;
            view.fog = false;
            view.vision = false;
            view.water = true;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let mut headless = Headless::new(&gpu, &mut app, target, config).unwrap();
        match pause_at {
            Some(at) => {
                headless.frames(&mut app, at).unwrap();
                app.view().lock().unwrap().paused = true;
                headless.frames(&mut app, frames - at).unwrap();
            }
            None => headless.frames(&mut app, frames).unwrap(),
        }
        headless.finish().unwrap();
        let pixels = app.readback().unwrap();
        RenderApp::destroy(&mut app, &gpu);
        pixels
    };

    // 12 frames end with frame time 11/60; the 13th draws at 12/60, where
    // the pause begins. The paused run draws 30 more frames at that time.
    let base = render(None, 13);
    let paused = render(Some(12), 42);
    let moved = render(None, 42);
    let differing = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(a, b)| a != b).count();
    let still = differing(&base, &paused);
    let waves = differing(&base, &moved);
    println!("pause: {still} bytes differ from the pause frame; unpaused, {waves}");
    assert_eq!(still, 0, "the water moved while paused");
    assert!(waves > 100, "the water did not move when unpaused: {waves}");
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
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Vec2, Vec4};

    let gpu = gpu();
    let (n, cell, scale) = (129usize, 8.0f32, 100.0f32);
    // Corner rectangles of the towers (90 m over ground at 10 m).
    let towers = [
        (60usize, 60usize, 67usize, 67usize),
        (TOWER_B.0, TOWER_B.1, TOWER_B.2, TOWER_B.3),
    ];
    let tall = |x: usize, y: usize| {
        towers
            .iter()
            .any(|&(x0, y0, x1, y1)| (x0..=x1).contains(&x) && (y0..=y1).contains(&y))
    };
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: (0..n * n).map(|i| if tall(i % n, i / n) { 1.0 } else { 0.1 }).collect(),
    };
    let shadow = sun::shadow_heights(&heightmap, cell, scale, 0.0, sun::direction());
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, cell, scale, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
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
        let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
        for &(x, y) in &[shaded_unit, sunny_unit] {
            for (a, word) in [
                x.to_bits(),
                y.to_bits(),
                x.to_bits(),
                y.to_bits(),
                0.0f32.to_bits(),
                0.0f32.to_bits(),
                0.0f32.to_bits(),
                1.0f32.to_bits(),
                0.0f32.to_bits(),
                0.0f32.to_bits(),
                0.0f32.to_bits(),
                1.0f32.to_bits(),
                6.0f32.to_bits(),
                0,
            ]
            .into_iter()
            .enumerate()
            {
                columns[a].push(word);
            }
        }
        let feed = Arc::new(UnitFeed::default());
        feed.publish(Arc::new(InstanceStream {
            tick: 1,
            published: std::time::Instant::now(),
            tick_seconds: 0.0,
            sim_seconds: 0.0,
            columns: at_rest(columns),
            anim: AnimWrites::default(),
        }));
        let classes = vec![UnitClass {
            gamut: ae_render::colorspace::Gamut::Bt709,
            color: [0.45, 0.45, 0.45],
            floats: false,
        }];
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(UnitsSetup {
            feed,
            capacity: 16,
            classes,
            min_pixels: 6.0,
            models: Vec::new(),
            rigs: Vec::new(),
        });
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
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        let mut run = Headless::new(&gpu, &mut app, target, config).unwrap();
        run.frames(&mut app, 12).unwrap();
        if r.level_a {
            // Cells 60..=66 hold tower A's corners (60..=67) only.
            let mut editor = TerrainEditor::new(&source);
            let update = editor
                .edit(
                    2,
                    &[(59, 59, 67, 67)],
                    |x, y| {
                        if (TOWER_B.0..TOWER_B.2).contains(&x) && (TOWER_B.1..TOWER_B.3).contains(&y) {
                            100.0
                        } else {
                            10.0
                        }
                    },
                    |_, _| 0,
                )
                .unwrap();
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
            let name = format!(
                "shadows-{}{}{}.ppm",
                r.shadows as u8, r.view_shadows as u8, r.level_a as u8
            );
            std::fs::write(std::path::Path::new(&dir).join(name), ppm).unwrap();
        }
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let plain = render(Run {
        shadows: false,
        view_shadows: true,
        level_a: false,
    });
    let switched_off = render(Run {
        shadows: true,
        view_shadows: false,
        level_a: false,
    });
    let on = render(Run {
        shadows: true,
        view_shadows: true,
        level_a: false,
    });
    let leveled_plain = render(Run {
        shadows: false,
        view_shadows: true,
        level_a: true,
    });
    let leveled = render(Run {
        shadows: true,
        view_shadows: true,
        level_a: true,
    });

    let view_proj = lod_view(&source, &camera.unwrap(), extent.width, extent.height).view_proj;
    let at = |x: f32, y: f32, h: f32| -> Option<usize> {
        let clip = view_proj * Vec4::new(x, h, y, 1.0);
        let (u, v) = (
            (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
            (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
        );
        (clip.w > 0.0 && u >= 2.0 && v >= 2.0 && u < extent.width as f32 - 2.0 && v < extent.height as f32 - 2.0)
            .then(|| (v as usize * extent.width as usize + u as usize) * 4)
    };
    // The off-screen tower: none of it in view.
    let (bx0, by0, bx1, by1) = towers[1];
    for x in bx0..=bx1 + 1 {
        for y in by0..=by1 + 1 {
            for h in [10.0, 100.0] {
                assert!(
                    at(x as f32 * cell - 4.0, y as f32 * cell - 4.0, h).is_none(),
                    "tower B at corner {x},{y} height {h} is in view"
                );
            }
        }
    }
    // Flat ground at corner resolution, sorted by the CPU heightfield:
    // lit (no shadow within two corners) or deep in shadow (all four
    // corners of the cell at least 8 m under the shadow height), clear of
    // the towers and the units.
    let s = |x: usize, y: usize| shadow[y * n + x] - 10.0;
    let near_unit = |x: f32, y: f32| {
        [shaded_unit, sunny_unit]
            .iter()
            .any(|&(ux, uy)| (x - ux).abs() < 24.0 && (y - uy).abs() < 24.0)
    };
    let near_tower = |x: usize, y: usize| {
        towers
            .iter()
            .any(|&(x0, y0, x1, y1)| x + 3 >= x0 && x <= x1 + 3 && y + 3 >= y0 && y <= y1 + 3)
    };
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
            } else if [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
                .iter()
                .all(|&(cx, cy)| s(cx, cy) > 8.0)
            {
                // Whose shadow: B's lies at smaller x than A's footprint.
                if x < 60 { shaded_b.push(i) } else { shaded.push(i) }
            }
        }
    }
    println!(
        "{} lit, {} shaded by tower A, {} by tower B (off screen)",
        lit.len(),
        shaded.len(),
        shaded_b.len()
    );
    assert!(
        lit.len() > 300 && shaded.len() > 20 && shaded_b.len() > 20,
        "{} lit, {} shaded, {} shaded by B",
        lit.len(),
        shaded.len(),
        shaded_b.len()
    );

    let rgb = |p: &[u8], i: usize| [p[i], p[i + 1], p[i + 2]];
    let luma = |p: &[u8], i: usize| p[i] as f32 * 0.3 + p[i + 1] as f32 * 0.59 + p[i + 2] as f32 * 0.11;
    let off_bytes = plain.iter().zip(&switched_off).filter(|(a, b)| a != b).count();
    assert_eq!(off_bytes, 0, "shadows switched off draw as none");
    let changed = lit.iter().filter(|&&i| rgb(&plain, i) != rgb(&on, i)).count();
    assert_eq!(changed, 0, "sunlit ground changes");
    for (name, points) in [("tower A", &shaded), ("tower B (off screen)", &shaded_b)] {
        let darker = points
            .iter()
            .filter(|&&i| luma(&on, i) < 0.75 * luma(&plain, i))
            .count();
        println!(
            "{name}: {darker} of {} shaded points a quarter darker or more",
            points.len()
        );
        assert!(
            darker * 10 >= points.len() * 9,
            "{name}'s shadow: {darker} of {}",
            points.len()
        );
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
    let gone = shaded
        .iter()
        .filter(|&&i| rgb(&leveled_plain, i) != rgb(&leveled, i))
        .count();
    assert_eq!(gone, 0, "tower A's shadow after levelling it");
    assert_eq!(
        unit_luma(&leveled_plain, shaded_unit),
        unit_luma(&leveled, shaded_unit),
        "the unit after tower A is levelled"
    );
    let still = shaded_b
        .iter()
        .filter(|&&i| luma(&leveled, i) < 0.75 * luma(&leveled_plain, i))
        .count();
    assert!(
        still * 10 >= shaded_b.len() * 9,
        "tower B's shadow after the edit: {still} of {}",
        shaded_b.len()
    );
    assert_clean(&gpu);
}

/// [`hills_shade_away_from_the_sun`]: the off-screen tower (corner
/// rectangle) and the camera (target x, z, distance).
const TOWER_B: (usize, usize, usize, usize) = (27, 70, 32, 75);
const CAMERA: (f32, f32, f32) = (560.0, 520.0, 520.0);

/// The units' shadows (`shadows.draw` into the cascaded `shadows.map`): on
/// flat ground seen from 75° up, a unit in view (A, pointing at the sun),
/// a big one placed just off the left edge of the view (B, every corner
/// of it off screen) and one in the sun away from both (C).
/// * Shadows made but switched off draw exactly as none.
/// * Ground the CPU (a ray toward the sun against each dart) puts deep in
///   A's or B's shadow darkens by a quarter or more, B's included: a
///   unit out of view casts into it. Ground with no shadow within 2 m does
///   not change.
/// * C's faces turned to the sun do not change (no self-shadowing acne).
///
/// Forcing the lookup to 1 in `unit_shadows.glsl`, or drawing no casters
/// in `UnitRenderer::record_shadows`, fails it.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn units_cast_shadows() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::terrain::sun;
    use ae_render::thread::{Headless, TargetInfo};
    use ae_render::units::{ATTR_PREV_ROT_X, ATTR_ROT_X, AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Vec2, Vec3};

    let gpu = gpu();
    let (n, cell, scale, ground) = (65usize, 4.0f32, 100.0f32, 10.0f32);
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![ground / scale; n * n],
    };
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, cell, scale, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let mut camera = Camera {
        target: Vec2::new(128.0, 128.0),
        ground,
        distance: 160.0,
        yaw: 0.0,
        pitch: 1.3,
        fov_y: 0.8,
    };
    {
        // As the app sets it up (the ground under the target).
        let defaults = TerrainApp::new(source.clone(), TerrainOutput::Readback);
        camera.fov_y = defaults.view().lock().unwrap().camera.fov_y;
    }
    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let eye = camera.eye(|_| ground);
    let project = |p: Vec3| -> Option<(f32, f32)> {
        let clip = view_proj * p.extend(1.0);
        (clip.w > 0.0).then(|| {
            (
                (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
                (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
            )
        })
    };
    let on_screen = |p: Vec3| {
        project(p).is_some_and(|(u, v)| u >= 0.0 && v >= 0.0 && u < extent.width as f32 && v < extent.height as f32)
    };
    let pixel = |p: Vec3| -> Option<usize> {
        let (u, v) = project(p)?;
        (u >= 2.0 && v >= 2.0 && u < extent.width as f32 - 2.0 && v < extent.height as f32 - 2.0)
            .then(|| (v as usize * extent.width as usize + u as usize) * 4)
    };

    // The darts as units.vert builds them: (x, z, heading x, heading z, radius).
    const CORNERS: [[f32; 3]; 4] = [[1.3, 0.25, 0.0], [-0.9, 0.0, -0.8], [-0.9, 0.0, 0.8], [-0.5, 0.9, 0.0]];
    const FACES: [usize; 12] = [0, 1, 3, 0, 3, 2, 1, 2, 3, 0, 2, 1];
    let corners = |(x, z, hx, hz, r): (f32, f32, f32, f32, f32)| -> [Vec3; 4] {
        let (forward, side) = (Vec3::new(hx, 0.0, hz), Vec3::new(-hz, 0.0, hx));
        CORNERS.map(|c| Vec3::new(x, ground, z) + (forward * c[0] + Vec3::Y * c[1] + side * c[2]) * r)
    };
    let triangles = |unit| {
        let c = corners(unit);
        (0..4).map(move |f| [c[FACES[f * 3]], c[FACES[f * 3 + 1]], c[FACES[f * 3 + 2]]])
    };
    // Möller–Trumbore: whether the ray o + t d (t > 0) hits the triangle.
    let hits = |o: Vec3, d: Vec3, [a, b, c]: [Vec3; 3]| {
        let (e1, e2) = (b - a, c - a);
        let p = d.cross(e2);
        let det = e1.dot(p);
        if det.abs() < 1e-8 {
            return false;
        }
        let s = o - a;
        let u = s.dot(p) / det;
        let q = s.cross(e1);
        let v = d.dot(q) / det;
        u >= 0.0 && v >= 0.0 && u + v <= 1.0 && e2.dot(q) / det > 1e-3
    };

    // A points at the sun: its top, behind its middle, casts past its tails.
    let toward_sun = Vec2::new(sun::direction().x, sun::direction().z).normalize();
    let a = (128.0f32, 128.0f32, toward_sun.x, toward_sun.y, 10.0f32);
    let c = (88.0f32, 160.0f32, 0.0f32, 1.0f32, 8.0f32);
    // B, heading away from the view: as far toward +x (the way shadows
    // fall, to the right on screen) as it can stand with every corner
    // still off the left edge.
    let mut b = (0.0f32, 128.0f32, -1.0f32, 0.0f32, 20.0f32);
    while !corners((b.0 + 0.25, b.1, b.2, b.3, b.4)).iter().any(|&p| on_screen(p)) {
        b.0 += 0.25;
    }
    assert!(b.0 > 5.0, "B starts in view");
    let units = [a, b, c];
    let sun_dir = sun::direction();
    let shaded = |p: Vec3| units.iter().any(|&u| triangles(u).any(|t| hits(p, sun_dir, t)));
    let hidden = |p: Vec3| units.iter().any(|&u| triangles(u).any(|t| hits(p, eye - p, t)));

    let render = |shadows: bool, view_shadows: bool, states: [u32; 3], prime_cache: bool| -> Vec<u8> {
        let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
        columns[ATTR_VIEW].extend(states);
        for &(x, z, hx, hz, r) in &units {
            for (k, word) in [x, z, x, z]
                .map(f32::to_bits)
                .into_iter()
                .chain(yaw(hx, hz).map(f32::to_bits))
                .chain(yaw(hx, hz).map(f32::to_bits))
                .chain([r.to_bits(), 0])
                .enumerate()
            {
                columns[k].push(word);
            }
        }
        let feed = Arc::new(UnitFeed::default());
        let publish = |tick, columns| {
            feed.publish(Arc::new(InstanceStream {
                tick,
                published: std::time::Instant::now(),
                tick_seconds: 0.0,
                sim_seconds: 0.0,
                columns: at_rest(columns),
                anim: AnimWrites::default(),
            }));
        };
        if prime_cache {
            let mut old = columns.clone();
            old[ATTR_VIEW][1] = 0;
            for (i, word) in yaw(0.0, 1.0).map(f32::to_bits).into_iter().enumerate() {
                old[ATTR_ROT_X + i][0] = word;
                old[ATTR_PREV_ROT_X + i][0] = word;
            }
            publish(1, old);
        } else {
            publish(1, columns.clone());
        }
        let classes = vec![UnitClass {
            gamut: ae_render::colorspace::Gamut::Bt709,
            color: [0.45, 0.45, 0.45],
            floats: false,
        }];
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(UnitsSetup {
            feed: feed.clone(),
            capacity: 16,
            classes,
            min_pixels: 6.0,
            models: Vec::new(),
            rigs: Vec::new(),
        });
        if shadows {
            app = app.with_shadows();
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera = camera;
            view.fog = false;
            view.shadows = view_shadows;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        let mut run = Headless::new(&gpu, &mut app, target, config).unwrap();
        if prime_cache {
            run.frames(&mut app, 4).unwrap();
            publish(2, columns);
        }
        run.frames(&mut app, 8).unwrap();
        run.wait().unwrap();
        run.finish().unwrap();
        let pixels = app.readback().unwrap();
        if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            std::fs::write(
                std::path::Path::new(&dir).join(format!("unit-shadows-{}{}.ppm", shadows as u8, view_shadows as u8)),
                ppm,
            )
            .unwrap();
        }
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let plain = render(false, true, [0; 3], false);
    let switched_off = render(true, false, [0; 3], false);
    let on = render(true, true, [0; 3], false);
    let blip = render(true, true, [0, 2, 0], false);
    let ghost = render(true, true, [0, 3, 0], true);

    // The ground on a 0.5 m grid: deep in a dart's shadow (and its
    // neighbours 0.75 m away), or lit with no shadow within 2 m; visible
    // (no dart between it and the camera, 0.5 m around).
    let at = |x: f32, z: f32| Vec3::new(x, ground + 0.01, z);
    let ring =
        |x: f32, z: f32, d: f32| [(-d, -d), (d, -d), (-d, d), (d, d), (0.0, 0.0)].map(|(dx, dz)| at(x + dx, z + dz));
    let (mut deep_a, mut deep_b, mut lit) = (
        std::collections::BTreeSet::new(),
        std::collections::BTreeSet::new(),
        std::collections::BTreeSet::new(),
    );
    for iz in 0..=480 {
        for ix in 0..=480 {
            let (x, z) = (8.0 + ix as f32 * 0.5, 8.0 + iz as f32 * 0.5);
            let Some(i) = pixel(at(x, z)) else { continue };
            if ring(x, z, 0.5).iter().any(|&p| hidden(p)) {
                continue;
            }
            if ring(x, z, 0.75).iter().all(|&p| shaded(p)) {
                if x < 64.0 {
                    deep_b.insert(i)
                } else {
                    deep_a.insert(i)
                };
            } else if !ring(x, z, 2.0).iter().chain(&ring(x, z, 1.0)).any(|&p| shaded(p))
                && (x - c.0).hypot(z - c.1) > 2.5 * c.4
            {
                lit.insert(i);
            }
        }
    }
    println!(
        "B at x {:.2}; {} lit points, {} deep in A's shadow, {} in B's (B off screen)",
        b.0,
        lit.len(),
        deep_a.len(),
        deep_b.len()
    );
    assert!(
        lit.len() > 5000 && deep_a.len() > 50 && deep_b.len() > 15,
        "{} lit, {} A, {} B",
        lit.len(),
        deep_a.len(),
        deep_b.len()
    );

    let rgb = |p: &[u8], i: usize| [p[i], p[i + 1], p[i + 2]];
    let luma = |p: &[u8], i: usize| p[i] as f32 * 0.3 + p[i + 1] as f32 * 0.59 + p[i + 2] as f32 * 0.11;
    assert_eq!(
        blip, ghost,
        "cached pages must match a fresh render after a caster turns and another becomes a ghost"
    );
    assert!(
        deep_a.iter().all(|&i| rgb(&blip, i) == rgb(&on, i)),
        "A still casts its shadow"
    );
    assert!(
        deep_b.iter().all(|&i| rgb(&blip, i) == rgb(&plain, i)),
        "B's shadow is gone"
    );
    assert_eq!(
        plain.iter().zip(&switched_off).filter(|(x, y)| x != y).count(),
        0,
        "shadows switched off draw as none"
    );
    let changed = lit.iter().filter(|&&i| rgb(&plain, i) != rgb(&on, i)).count();
    assert_eq!(changed, 0, "lit ground changes");
    for (name, points) in [("A", &deep_a), ("B (off screen)", &deep_b)] {
        let darker = points
            .iter()
            .filter(|&&i| luma(&on, i) < 0.75 * luma(&plain, i))
            .count();
        println!(
            "{name}: {darker} of {} shaded points a quarter darker or more",
            points.len()
        );
        assert!(
            darker * 10 >= points.len() * 9,
            "{name}'s shadow: {darker} of {}",
            points.len()
        );
    }
    // C's faces turned to the sun, at their centres (pulled a little in).
    let mut faces = 0;
    for t in triangles(c) {
        let normal = (t[1] - t[0]).cross(t[2] - t[0]).normalize();
        let centre = (t[0] + t[1] + t[2]) / 3.0;
        let outward = if normal.dot(centre - Vec3::new(c.0, ground + 0.3 * c.4, c.1)) < 0.0 {
            -normal
        } else {
            normal
        };
        if outward.dot(sun_dir) <= 0.1 || outward.dot(eye - centre) <= 0.0 {
            continue;
        }
        for w in [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [1.0 / 3.0; 3]] {
            let p = t[0] * w[0] + t[1] * w[1] + t[2] * w[2];
            let i = pixel(p).expect("C in view");
            assert_eq!(rgb(&plain, i), rgb(&on, i), "C's sunlit face at {p:?}");
        }
        faces += 1;
    }
    assert!(faces >= 1, "C shows a sunlit face");
    assert_clean(&gpu);
}

/// The particles' coloured shadows (`shadows.particles` into
/// `shadows.color`), on flat ground with one unit (the units' map, and so
/// the colour target, exist only with units), still test effects:
/// * Orange smoke 25 m up tints the ground in its shadow: green and blue
///   drop well below red there (the sky light, not tinted, stays).
/// * The ground away from the shadow and from the puff on screen is
///   byte-identical to no particles.
/// * A puff buried just under the unit and the ground, all of them nearer
///   the sun than it although within its disc as the sun sees it, leaves
///   the frame byte-identical to no particles (the depth in alpha).
/// * The same puff off screen still tints its shadow in view (the casters
///   are every live particle, not the camera's list).
/// * The same puff emissive (fire, sparks) casts nothing.
/// * Shadows switched off with smoke draw as no shadows at all.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn particles_cast_coloured_shadows() {
    use ae_render::particles::{Burst, ParticleBlend, ParticleFeed, ParticlesSetup};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::terrain::sun;
    use ae_render::thread::{Headless, TargetInfo};
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Vec2, Vec3};

    let gpu = gpu();
    let (n, cell, scale, ground) = (65usize, 4.0f32, 100.0f32, 10.0f32);
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![ground / scale; n * n],
    };
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, cell, scale, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let mut camera = TerrainApp::new(source.clone(), TerrainOutput::Readback)
        .view()
        .lock()
        .unwrap()
        .camera;
    camera = Camera {
        target: Vec2::new(128.0, 128.0),
        ground,
        distance: 160.0,
        yaw: 0.0,
        pitch: 1.3,
        ..camera
    };
    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let eye = camera.eye(|_| ground);
    let pixel = |p: Vec3| -> Option<usize> {
        let clip = view_proj * p.extend(1.0);
        let (u, v) = (
            (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
            (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
        );
        (clip.w > 0.0 && u >= 2.0 && v >= 2.0 && u < extent.width as f32 - 2.0 && v < extent.height as f32 - 2.0)
            .then(|| (v as usize * extent.width as usize + u as usize) * 4)
    };
    let sun_dir = sun::direction();
    // Distance from sphere centre `c` to the ray from `o` along `d`
    // (infinite when the centre is behind it).
    let ray_gap = |o: Vec3, d: Vec3, c: Vec3| {
        let d = d.normalize();
        let t = (c - o).dot(d);
        if t <= 0.0 {
            f32::INFINITY
        } else {
            (c - o - d * t).length()
        }
    };

    // The unit at the target; the smoke's shadow centred 30 m from it.
    let unit = (128.0f32, 128.0f32, 1.0f32, 0.0f32, 6.0f32);
    let radius = 10.0f32;
    let shadow_at = Vec3::new(150.0, ground, 150.0);
    let smoke_at = shadow_at + sun_dir * (25.0 / sun_dir.y);
    // Buried under the unit: its sunward surface 0.5 m under the ground.
    let buried_at = Vec3::new(unit.0, ground - radius - 0.5, unit.1);
    let effects = vec![
        still_effect([1.0, 0.45, 0.1, 0.9], 2.0 * radius, false),
        still_effect([1.0, 0.45, 0.1, 0.9], 2.0 * radius, true),
    ];
    let one = |pos: Vec3, effect: u32| Burst {
        pos: pos.to_array(),
        ground: false,
        effect,
        count: 1,
        scale: 1.0,
        seed: 3,
    };

    let render = |bursts: Option<&[Burst]>, shadows: bool, view_shadows: bool, name: &str| -> Vec<u8> {
        let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
        let (x, z, hx, hz, r) = unit;
        for (k, word) in [x, z, x, z]
            .map(f32::to_bits)
            .into_iter()
            .chain(yaw(hx, hz).map(f32::to_bits))
            .chain(yaw(hx, hz).map(f32::to_bits))
            .chain([r.to_bits(), 0])
            .enumerate()
        {
            columns[k].push(word);
        }
        let feed = Arc::new(UnitFeed::default());
        feed.publish(Arc::new(InstanceStream {
            tick: 1,
            published: std::time::Instant::now(),
            tick_seconds: 0.0,
            sim_seconds: 0.0,
            columns: at_rest(columns),
            anim: AnimWrites::default(),
        }));
        let classes = vec![UnitClass {
            gamut: ae_render::colorspace::Gamut::Bt709,
            color: [0.45, 0.45, 0.45],
            floats: false,
        }];
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_units(UnitsSetup {
                feed,
                capacity: 16,
                classes,
                min_pixels: 6.0,
                models: Vec::new(),
                rigs: Vec::new(),
            })
            .with_tone(ae_render::tonemap::ToneParams {
                bloom: 0.0,
                ..Default::default()
            });
        if shadows {
            app = app.with_shadows();
        }
        if let Some(bursts) = bursts {
            let feed = Arc::new(ParticleFeed::default());
            feed.emit_all(bursts);
            app = app.with_particles(ParticlesSetup {
                feed,
                capacity: 1024,
                effects: effects.clone(),
                downscale: 2,
                blend: ParticleBlend::Sorted,
            });
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera = camera;
            view.fog = false;
            view.shadows = view_shadows;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        let mut run = Headless::new(&gpu, &mut app, target, config).unwrap();
        run.frames(&mut app, 8).unwrap();
        run.wait().unwrap();
        run.finish().unwrap();
        let pixels = app.readback().unwrap();
        if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            std::fs::write(
                std::path::Path::new(&dir).join(format!("particle-shadows-{name}.ppm")),
                ppm,
            )
            .unwrap();
        }
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let plain = render(None, true, true, "plain");
    let smoke = render(Some(&[one(smoke_at, 0)]), true, true, "smoke");
    let sparks = render(Some(&[one(smoke_at, 1)]), true, true, "emissive");
    let buried = render(Some(&[one(buried_at, 0)]), true, true, "buried");
    // Off screen, its shadow on screen: the first place up-sun of the
    // target where no point within 1.6 radii of its centre is in view
    // (clear of the simulation's frustum test, which keeps a sphere
    // touching the view).
    let up_sun = Vec3::new(sun_dir.x, 0.0, sun_dir.z).normalize();
    let sphere: Vec<Vec3> = (0..27)
        .filter(|&k| k != 13)
        .map(|k| {
            Vec3::new((k % 3) as f32 - 1.0, (k / 3 % 3) as f32 - 1.0, (k / 9) as f32 - 1.0).normalize() * 1.6 * radius
        })
        .collect();
    let unseen_at = (0..400)
        .map(|t| Vec3::new(camera.target.x, ground, camera.target.y) + up_sun * t as f32 + sun_dir * (25.0 / sun_dir.y))
        .find(|&c| sphere.iter().all(|&d| pixel(c + d).is_none()))
        .expect("an off-screen place");
    let unseen = render(Some(&[one(unseen_at, 0)]), true, true, "unseen");
    let off = render(Some(&[one(smoke_at, 0)]), true, false, "off");
    let never = render(Some(&[one(smoke_at, 0)]), false, true, "never");

    // The ground on a 0.5 m grid, away from the unit and from the puff as
    // the camera sees it: deep in the smoke's shadow (the sun ray through
    // its inner 40 %), or clear of it (outside 1.6 radii).
    let (mut deep, mut clear, mut under, mut deep_unseen) = (Vec::new(), Vec::new(), 0, Vec::new());
    let light = |p: Vec3| p - sun_dir * p.dot(sun_dir);
    for iz in 0..=400 {
        for ix in 0..=400 {
            let p = Vec3::new(28.0 + ix as f32 * 0.5, ground + 0.01, 28.0 + iz as f32 * 0.5);
            let Some(i) = pixel(p) else { continue };
            if (Vec2::new(p.x, p.z) - Vec2::new(unit.0, unit.1)).length() < 2.0 * unit.4 {
                continue;
            }
            if ray_gap(p, sun_dir, unseen_at) < 0.4 * radius {
                deep_unseen.push(i);
            }
            // Within the buried puff's disc as the sun sees it.
            if (light(p) - light(buried_at)).length() < 0.9 * radius {
                under += 1;
            }
            if ray_gap(eye, p - eye, smoke_at) < 1.6 * radius {
                continue;
            }
            let gap = ray_gap(p, sun_dir, smoke_at);
            if gap < 0.4 * radius {
                deep.push(i);
            } else if gap > 1.6 * radius {
                clear.push(i);
            }
        }
    }
    println!(
        "{} ground points deep in the smoke's shadow, {} clear of it, {under} within the buried puff's disc, {} deep in the off-screen smoke's shadow",
        deep.len(),
        clear.len(),
        deep_unseen.len()
    );
    assert!(
        deep.len() > 200 && clear.len() > 20000 && under > 300 && deep_unseen.len() > 100,
        "{} deep, {} clear, {under} under, {} unseen",
        deep.len(),
        clear.len(),
        deep_unseen.len()
    );

    let ratio = |on: &[u8], i: usize, c: usize| on[i + c] as f32 / plain[i + c].max(1) as f32;
    // Orange smoke: red passes, blue least (green between); in sRGB bytes,
    // with the sky's light untinted.
    let tinted = deep
        .iter()
        .filter(|&&i| {
            ratio(&smoke, i, 2) < 0.9 * ratio(&smoke, i, 0)
                && ratio(&smoke, i, 2) < ratio(&smoke, i, 1)
                && ratio(&smoke, i, 1) < ratio(&smoke, i, 0)
        })
        .count();
    let mean = |c: usize| deep.iter().map(|&i| ratio(&smoke, i, c)).sum::<f32>() / deep.len() as f32;
    println!(
        "smoke's shadow: mean ratio to no particles r {:.2} g {:.2} b {:.2}; {tinted} of {} tinted",
        mean(0),
        mean(1),
        mean(2),
        deep.len()
    );
    assert!(
        tinted * 10 >= deep.len() * 9,
        "the smoke's shadow is not tinted: {tinted} of {}",
        deep.len()
    );
    assert!(
        mean(0) > 0.95 && mean(2) < 0.85,
        "the smoke lets red through and stops blue: r {:.2} b {:.2}",
        mean(0),
        mean(2)
    );
    let bluer = deep_unseen
        .iter()
        .filter(|&&i| ratio(&unseen, i, 2) < 0.9 * ratio(&unseen, i, 0))
        .count();
    println!("off-screen smoke: {bluer} of {} tinted", deep_unseen.len());
    assert!(
        bluer * 10 >= deep_unseen.len() * 9,
        "smoke off screen does not shade the view: {bluer} of {}",
        deep_unseen.len()
    );
    let changed =
        |frame: &[u8], points: &[usize]| points.iter().filter(|&&i| frame[i..i + 3] != plain[i..i + 3]).count();
    assert_eq!(
        changed(&smoke, &clear),
        0,
        "the ground clear of the smoke's shadow changes"
    );
    assert_eq!(
        changed(&sparks, &deep) + changed(&sparks, &clear),
        0,
        "an emissive puff casts a shadow"
    );
    let differing = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(x, y)| x != y).count();
    assert_eq!(
        differing(&plain, &buried),
        0,
        "a puff under the ground tints what is above it"
    );
    assert_eq!(differing(&off, &never), 0, "shadows switched off draw as no shadows");
    assert_clean(&gpu);
}

/// Particles (`particles.sim` + `particles.draw`) on flat ground, with
/// static test effects (no motion, long lives):
/// * A blue puff in front of a red one on the same line of sight shows
///   blue, and the frame is byte-identical whichever was emitted first
///   (the sort key carries depth, then slot); swapped in depth, red shows.
/// * A puff under the ground draws nothing (the shader's own depth test).
/// * A puff cut by the ground fades into it: along a screen column through
///   it, no step between neighbouring pixels exceeds a quarter of its
///   strongest contribution (no hard line).
/// * Particles that have died, and a pool never emitted into, draw exactly
///   as no particle renderer at all.
/// * An explosion of the default effects changes the frame, and renders
///   byte-identically twice (clustered depths: the slot in the key orders
///   what the depth does not, whatever order the simulation appended in).
///
/// All of the above at full and at half resolution. Then a depth edge: a
/// puff behind a plateau's rim leaves the plateau untouched at full
/// resolution, and at half resolution the composite does not bleed it onto
/// the plateau (a plain bilinear upsample changes 55 of its pixels), nor
/// lose more than half of it anywhere above the rim (a gross halo).
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn particles_sort_blend_and_soften() {
    use ae_render::particles::{Burst, EffectDef, ParticleBlend, default_effects, explosion};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use glam::{Vec2, Vec3};

    let gpu = gpu();
    let (n, cell, scale, ground) = (65usize, 4.0f32, 100.0f32, 10.0f32);
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![ground / scale; n * n],
    };
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, cell, scale, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let mut camera = TerrainApp::new(source.clone(), TerrainOutput::Readback)
        .view()
        .lock()
        .unwrap()
        .camera;
    camera = Camera {
        target: Vec2::new(128.0, 128.0),
        ground,
        distance: 160.0,
        yaw: 0.0,
        pitch: 0.6,
        ..camera
    };
    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let eye = camera.eye(|_| ground);
    let pixel = |p: Vec3| -> (u32, u32) {
        let clip = view_proj * p.extend(1.0);
        (
            ((clip.x / clip.w * 0.5 + 0.5) * extent.width as f32) as u32,
            ((clip.y / clip.w * 0.5 + 0.5) * extent.height as f32) as u32,
        )
    };
    let index = |(x, y): (u32, u32)| (y * extent.width + x) as usize * 4;

    let still = |color: [f32; 4], size: f32, emissive: bool, life: f32| EffectDef {
        gamut: ae_render::colorspace::Gamut::Bt709,
        color_start: color,
        color_end: color,
        size: [size, size],
        life: [life, life],
        speed: [0.0, 0.0],
        up: 0.0,
        spread: 0.0,
        gravity: 0.0,
        drag: 0.0,
        emissive,
        // The old additive value (exposed at EV14) as cd/m²: 1 / 5.086e-5.
        emission: if emissive { [19_660.8; 2] } else { [0.0; 2] },
        cover: 0.0,
        fade_in: 0.0,
        light: None,
    };
    // 0 red, 1 blue, 2 a big grey puff, 3 red that dies at once.
    let effects = vec![
        still([0.9, 0.05, 0.05, 1.0], 12.0, false, 100.0),
        still([0.05, 0.05, 0.9, 1.0], 12.0, false, 100.0),
        still([0.6, 0.6, 0.6, 0.9], 40.0, false, 100.0),
        still([0.9, 0.05, 0.05, 1.0], 12.0, false, 0.02),
    ];
    let one = |pos: Vec3, effect: u32| Burst {
        pos: pos.to_array(),
        ground: false,
        effect,
        count: 1,
        scale: 1.0,
        seed: effect,
    };

    let render_on = |source: &Arc<TerrainSource>, bursts: Option<(&[Burst], Vec<EffectDef>, u32)>| -> Vec<u8> {
        render_particles(
            &gpu,
            source,
            camera,
            extent,
            bursts.map(|(b, e, downscale)| (b, e, downscale, ParticleBlend::Sorted)),
        )
    };
    let render = |bursts: Option<(&[Burst], Vec<EffectDef>, u32)>| render_on(&source, bursts);
    let luma = |p: &[u8], i: usize| p[i] as f32 * 0.3 + p[i + 1] as f32 * 0.59 + p[i + 2] as f32 * 0.11;
    let none = render(None);

    for downscale in [1, 2] {
        println!("downscale {downscale}:");
        let empty = render(Some((&[], effects.clone(), downscale)));
        assert_eq!(
            none.iter().zip(&empty).filter(|(a, b)| a != b).count(),
            0,
            "an empty pool draws nothing"
        );

        // Two puffs 30 m apart on one line of sight, 30 m above the ground.
        let centre = Vec3::new(128.0, 40.0, 128.0);
        let dir = (centre - eye).normalize();
        let (near, far) = (centre - dir * 30.0, centre + dir * 30.0);
        let at = index(pixel(centre));
        let blue_front_a = render(Some((&[one(far, 0), one(near, 1)], effects.clone(), downscale)));
        let blue_front_b = render(Some((&[one(near, 1), one(far, 0)], effects.clone(), downscale)));
        let red_front = render(Some((&[one(near, 0), one(far, 1)], effects.clone(), downscale)));
        let differing = blue_front_a.iter().zip(&blue_front_b).filter(|(a, b)| a != b).count();
        assert_eq!(differing, 0, "the emission order changes {differing} bytes");
        let (b, r) = (&blue_front_a[at..at + 3], &red_front[at..at + 3]);
        println!(
            "  blue in front: {b:?}, red in front: {r:?}, ground: {:?}",
            &none[at..at + 3]
        );
        // Under ACES the half-resolution composite's bilinear mixes some
        // ground into the pixel; the puff still dominates.
        assert!(b[2] as f32 > 1.5 * b[0] as f32, "blue in front shows blue: {b:?}");
        assert!(r[0] as f32 > 1.5 * r[2] as f32, "red in front shows red: {r:?}");

        // Under the ground: nothing; dead: nothing.
        let buried = render(Some((
            &[one(Vec3::new(128.0, ground - 20.0, 128.0), 0)],
            effects.clone(),
            downscale,
        )));
        assert_eq!(
            none.iter().zip(&buried).filter(|(a, b)| a != b).count(),
            0,
            "a buried puff draws"
        );
        let dead = render(Some((&[one(centre, 3)], effects.clone(), downscale)));
        assert_eq!(
            none.iter().zip(&dead).filter(|(a, b)| a != b).count(),
            0,
            "a dead puff draws"
        );

        // A big puff half in the ground: its contribution along the screen
        // column through it changes smoothly.
        let low = Vec3::new(128.0, ground, 128.0);
        let soft = render(Some((&[one(low, 2)], effects.clone(), downscale)));
        let (cx, cy) = pixel(low);
        let column: Vec<f32> = (cy.saturating_sub(60)..(cy + 60).min(extent.height))
            .map(|y| (luma(&soft, index((cx, y))) - luma(&none, index((cx, y)))).abs())
            .collect();
        let strongest = column.iter().cloned().fold(0.0, f32::max);
        let step = column.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max);
        println!("  ground-cut puff: strongest {strongest:.1}, largest step {step:.1}");
        assert!(strongest > 25.0, "the puff shows: {strongest}");
        assert!(step < 0.25 * strongest, "a hard edge: step {step} of {strongest}");

        // The default effects.
        let boom = render(Some((
            &explosion([128.0, 2.0, 128.0], true, 1.0, 7),
            default_effects(),
            downscale,
        )));
        let changed = none.iter().zip(&boom).filter(|(a, b)| a != b).count();
        println!("  explosion: {changed} bytes changed");
        assert!(changed > 1000, "an explosion shows: {changed} bytes");
        // About 90 particles at clustered depths, appended by the simulation
        // in whatever order its invocations run: the same frame every time.
        let again = render(Some((
            &explosion([128.0, 2.0, 128.0], true, 1.0, 7),
            default_effects(),
            downscale,
        )));
        let differing = boom.iter().zip(&again).filter(|(a, b)| a != b).count();
        assert_eq!(differing, 0, "the same explosion twice differs in {differing} bytes");
    }

    // A depth edge: a plateau 30 m above the ground on the camera's side,
    // its rim 10 m before the target, and a puff on the line of sight
    // through the rim, 40 m beyond it (its sphere's front is behind every
    // plateau pixel). At full resolution the plateau is untouched; at half
    // resolution, the depth-aware composite must not bleed the smoke onto
    // the plateau either, and must still show it over the ground behind.
    let toward_eye = (Vec2::new(eye.x, eye.z) - camera.target).normalize();
    let plateau = ground + 30.0;
    let mut data = vec![ground / scale; n * n];
    for (i, h) in data.iter_mut().enumerate() {
        let p = Vec2::new((i % n) as f32 * cell, (i / n) as f32 * cell);
        if (p - camera.target).dot(toward_eye) > 10.0 {
            *h = plateau / scale;
        }
    }
    let heightmap = Heightmap {
        width: n,
        height: n,
        data,
    };
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, cell, scale, 0.0, None);
    let cliff = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let rim = (camera.target + toward_eye * 10.0).extend(plateau);
    let rim = Vec3::new(rim.x, plateau, rim.y);
    let behind = rim + (rim - eye).normalize() * 40.0;
    let bare = render_on(&cliff, None);
    let full = render_on(&cliff, Some((&[one(behind, 2)], effects.clone(), 1)));
    let half = render_on(&cliff, Some((&[one(behind, 2)], effects.clone(), 2)));
    let (mut shown, mut bled, mut worst, mut halo, mut lost) = (0, 0, 0.0f32, 0, 0.0f32);
    for i in (0..bare.len()).step_by(4) {
        let (f, h) = (
            (luma(&full, i) - luma(&bare, i)).abs(),
            (luma(&half, i) - luma(&bare, i)).abs(),
        );
        if f > 20.0 {
            shown += 1;
            // Smoke over the ground right above the rim, lost at half
            // resolution (a halo around the plateau).
            if h < 0.5 * f {
                halo += 1;
            }
            lost += (f - h).max(0.0);
        }
        // Untouched at full resolution: the plateau (and what the puff misses).
        if f == 0.0 && h > 4.0 {
            bled += 1;
            worst = worst.max(h);
        }
    }
    println!(
        "depth edge: {shown} pixels show the puff at full resolution; at half, {bled} untouched pixels changed (worst {worst:.1}), {halo} lost more than half of it, lost {:.0} luma in all",
        lost
    );
    assert!(shown > 200, "the puff shows above the rim: {shown}");
    assert!(
        bled <= 4,
        "smoke bleeds onto the plateau at half resolution: {bled} pixels (worst {worst:.1})"
    );
    assert!(
        halo <= 4,
        "smoke leaves a halo above the rim at half resolution: {halo} pixels"
    );
    assert_clean(&gpu);
}

/// The tile-binned packets (mode 8 of `~/particles`) draw the same puffs as
/// the sorted draw at both resolutions, keep the centre sort's order, and
/// carry the per-particle glow through their own composite:
/// * a blue puff in front of a red one shows blue at full and at half
///   resolution;
/// * a puff under the ground and a dead one draw nothing (byte-identical to
///   no particle renderer);
/// * an explosion changes the frame;
/// * a fire in front of smoke still blooms at both resolutions.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn particles_packets_both_resolutions() {
    use ae_render::particles::{Burst, ParticleBlend, explosion};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::tonemap::{BLOOM_STRENGTH, ToneParams};
    use glam::{Vec2, Vec3, Vec4};

    let gpu = gpu();
    let (n, cell, scale, ground) = (65usize, 4.0f32, 100.0f32, 10.0f32);
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![ground / scale; n * n],
    };
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, cell, scale, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let mut camera = TerrainApp::new(source.clone(), TerrainOutput::Readback)
        .view()
        .lock()
        .unwrap()
        .camera;
    camera = Camera {
        target: Vec2::new(128.0, 128.0),
        ground,
        distance: 160.0,
        yaw: 0.0,
        pitch: 0.6,
        ..camera
    };
    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let eye = camera.eye(|_| ground);
    let index = |p: Vec3| {
        let clip = view_proj * p.extend(1.0);
        (((clip.y / clip.w * 0.5 + 0.5) * extent.height as f32) as usize * extent.width as usize
            + ((clip.x / clip.w * 0.5 + 0.5) * extent.width as f32) as usize)
            * 4
    };
    let still = |color: [f32; 4], size: f32, emissive: bool| -> ae_render::particles::EffectDef {
        ae_render::particles::EffectDef {
            gamut: ae_render::colorspace::Gamut::Bt709,
            color_start: color,
            color_end: color,
            size: [size, size],
            life: [100.0, 100.0],
            speed: [0.0, 0.0],
            up: 0.0,
            spread: 0.0,
            gravity: 0.0,
            drag: 0.0,
            emissive,
            emission: if emissive { [40_000.0; 2] } else { [0.0; 2] },
            cover: 0.0,
            fade_in: 0.0,
            light: None,
        }
    };
    let effects = vec![
        still([0.9, 0.05, 0.05, 1.0], 12.0, false),
        still([0.05, 0.05, 0.9, 1.0], 12.0, false),
    ];
    let one = |pos: Vec3, effect: u32| Burst {
        pos: pos.to_array(),
        ground: false,
        effect,
        count: 1,
        scale: 1.0,
        seed: effect,
    };
    let centre = Vec3::new(128.0, 40.0, 128.0);
    let dir = (centre - eye).normalize();
    let (near, far) = (centre - dir * 30.0, centre + dir * 30.0);
    let none = render_particles(&gpu, &source, camera, extent, None);
    let luma = |p: &[u8], i: usize| p[i] as f32 * 0.3 + p[i + 1] as f32 * 0.59 + p[i + 2] as f32 * 0.11;

    for downscale in [1, 2] {
        let render = |blend, bursts: &[Burst]| {
            render_particles(
                &gpu,
                &source,
                camera,
                extent,
                Some((bursts, effects.clone(), downscale, blend)),
            )
        };
        let packets = render(ParticleBlend::Packets, &[one(far, 0), one(near, 1)]);
        let sorted = render(ParticleBlend::Sorted, &[one(far, 0), one(near, 1)]);
        let at = index(centre);
        let (p, s) = (&packets[at..at + 3], &sorted[at..at + 3]);
        println!("downscale {downscale}: packets {p:?}, sorted {s:?}");
        if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(packets.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            std::fs::write(
                std::path::Path::new(&dir).join(format!("particles-packets-{downscale}.ppm")),
                ppm,
            )
            .unwrap();
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(sorted.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            std::fs::write(
                std::path::Path::new(&dir).join(format!("particles-sorted-{downscale}.ppm")),
                ppm,
            )
            .unwrap();
        }
        assert!(
            p[2] as f32 > 1.5 * p[0] as f32,
            "packets show blue in front at {downscale}: {p:?}"
        );
        assert!(
            s[2] as f32 > 1.5 * s[0] as f32,
            "sorted show blue in front at {downscale}: {s:?}"
        );

        let buried = render(
            ParticleBlend::Packets,
            &[one(Vec3::new(128.0, ground - 20.0, 128.0), 0)],
        );
        assert_eq!(
            none.iter().zip(&buried).filter(|(a, b)| a != b).count(),
            0,
            "a buried puff draws with the packets at {downscale}"
        );
        let boom = render(ParticleBlend::Packets, &explosion([128.0, 2.0, 128.0], true, 1.0, 7));
        let changed = none.iter().zip(&boom).filter(|(a, b)| a != b).count();
        assert!(
            changed > 500,
            "an explosion shows with the packets at {downscale}: {changed} bytes"
        );
    }

    // The fire in front of the smoke, as the half-resolution glow test.
    let fire_effect = still([1.0, 0.6, 0.2, 1.0], 6.0, true);
    let smoke_effect = still([0.09, 0.085, 0.08, 0.8], 16.0, false);
    let glow_effects = vec![fire_effect, smoke_effect];
    let fire = Vec3::new(128.0, ground + 3.0, 128.0);
    let back = (fire - eye).normalize();
    let smoke = Vec3::new(fire.x + back.x * 3.0, ground + 3.0, fire.z + back.z * 3.0);
    let bursts = [
        Burst {
            pos: fire.to_array(),
            ground: true,
            effect: 0,
            count: 1,
            scale: 1.0,
            seed: 1,
        },
        Burst {
            pos: smoke.to_array(),
            ground: true,
            effect: 1,
            count: 1,
            scale: 1.0,
            seed: 2,
        },
    ];
    let clip = view_proj * Vec4::new(fire.x, fire.y, fire.z, 1.0);
    let (cu, cv) = (
        ((clip.x / clip.w * 0.5 + 0.5) * extent.width as f32) as i32,
        ((clip.y / clip.w * 0.5 + 0.5) * extent.height as f32) as i32,
    );
    let ring = |on: &[u8], off: &[u8]| -> f32 {
        let luma_at = |p: &[u8], x: i32, y: i32| {
            let i = ((y * extent.width as i32 + x) * 4) as usize;
            luma(p, i)
        };
        let mut spread = 0.0f32;
        for y in (cv - 30)..=(cv + 30) {
            for x in (cu - 30)..=(cu + 30) {
                let d = ((x - cu).pow(2) + (y - cv).pow(2)) as f32;
                if ((10.0 * 10.0)..=(30.0 * 30.0)).contains(&d) {
                    spread = spread.max(luma_at(on, x, y) - luma_at(off, x, y));
                }
            }
        }
        spread
    };
    for downscale in [1, 2] {
        let render = |bloom: f32| {
            render_particles_lit(
                &gpu,
                &source,
                camera,
                extent,
                Some((&bursts, glow_effects.clone(), downscale, ParticleBlend::Packets)),
                &[],
                ToneParams {
                    bloom,
                    ..Default::default()
                },
            )
        };
        let (off, on) = (render(0.0), render(BLOOM_STRENGTH));
        let glow = ring(&on, &off);
        println!("packets downscale {downscale}: fire glow spread {glow:.1} luma");
        assert!(
            glow > 2.0,
            "the packets' fire glow is missing at {downscale}: {glow:.1}"
        );
    }
    assert_clean(&gpu);
}

/// A medium's light from the clustered lights arrives as illuminance: at
/// the same distance, four times the intensity adds four times the light,
/// well past the old 10-bit ceiling (1022) that saturated every light over
/// about 1 000 lx.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn particles_light_by_illuminance() {
    use ae_render::lights::Light;
    use ae_render::particles::{Burst, ParticleBlend};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::tonemap::ToneParams;
    use glam::{Vec2, Vec3};

    let gpu = gpu();
    let (n, cell, scale, ground) = (65usize, 4.0f32, 100.0f32, 10.0f32);
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![ground / scale; n * n],
    };
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, cell, scale, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let mut camera = TerrainApp::new(source.clone(), TerrainOutput::Readback)
        .view()
        .lock()
        .unwrap()
        .camera;
    camera = Camera {
        target: Vec2::new(128.0, 128.0),
        ground,
        distance: 160.0,
        yaw: 0.0,
        pitch: 0.6,
        ..camera
    };
    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let puff = Vec3::new(128.0, 40.0, 128.0);
    let clip = view_proj * puff.extend(1.0);
    let at = (
        ((clip.x / clip.w * 0.5 + 0.5) * extent.width as f32) as u32,
        ((clip.y / clip.w * 0.5 + 0.5) * extent.height as f32) as u32,
    );
    let index = (at.1 * extent.width + at.0) as usize * 4;

    // Opaque smoke, so the pixel is the puff's radiance alone; the sun is
    // off and Clamp decodes the stored values linearly.
    let effect = still_effect([0.2, 0.2, 0.2, 1.0], 12.0, false);
    let burst = Burst {
        pos: puff.to_array(),
        ground: false,
        effect: 0,
        count: 1,
        scale: 1.0,
        seed: 7,
    };
    let tone = ToneParams {
        ev100: 14.0,
        sun_illuminance: 0.0,
        bloom: 0.0,
        auto: false,
    };
    // 10 m from the puff: 10 000 lx at 1e6 cd, 40 000 lx at 4e6 cd.
    let light = |intensity: f32| Light::point(Vec3::new(128.0, 40.0, 138.0), Vec3::ONE, intensity, 0.0, 0.05);
    let render = |lights: &[Light]| {
        render_particles_lit(
            &gpu,
            &source,
            camera,
            extent,
            Some((&[burst], vec![effect], 1, ParticleBlend::Sorted)),
            lights,
            tone,
        )
    };
    let dark = render(&[]);
    let dim = render(&[light(1.0e6)]);
    let bright = render(&[light(4.0e6)]);
    let decode = |p: &[u8], c: usize| {
        let v = p[index + c] as f32 / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let add = |p: &[u8]| (0..3).map(|c| decode(p, c) - decode(&dark, c)).sum::<f32>() / 3.0;
    let (dim, bright) = (add(&dim), add(&bright));
    println!(
        "clustered light on a puff: {dim:.4}, 4x: {bright:.4} (ratio {:.2})",
        bright / dim
    );
    assert!(dim > 0.02, "the light lights the puff: {dim}");
    assert!(
        (bright / dim - 4.0).abs() < 0.2,
        "the puff sees illuminance, not a clipped coefficient: {bright:.4} / {dim:.4}"
    );
    assert_clean(&gpu);
}

/// A frame of the flat test terrain seen by `camera`; with Some, a particle
/// renderer (`downscale`, `blend`) whose bursts are emitted before the
/// first frame, still for the 8 frames rendered.
fn render_particles(
    gpu: &Arc<GpuDevice>,
    source: &Arc<ae_render::terrain::app::TerrainSource>,
    camera: ae_render::terrain::camera::Camera,
    extent: vk::Extent2D,
    particles: Option<(
        &[ae_render::particles::Burst],
        Vec<ae_render::particles::EffectDef>,
        u32,
        ae_render::particles::ParticleBlend,
    )>,
) -> Vec<u8> {
    render_particles_lit(
        gpu,
        source,
        camera,
        extent,
        particles,
        &[],
        ae_render::tonemap::ToneParams {
            bloom: 0.0,
            ..Default::default()
        },
    )
}

/// [`render_particles`] with the host's lights and a tone setting (for the
/// tests that light the particles).
fn render_particles_lit(
    gpu: &Arc<GpuDevice>,
    source: &Arc<ae_render::terrain::app::TerrainSource>,
    camera: ae_render::terrain::camera::Camera,
    extent: vk::Extent2D,
    particles: Option<(
        &[ae_render::particles::Burst],
        Vec<ae_render::particles::EffectDef>,
        u32,
        ae_render::particles::ParticleBlend,
    )>,
    lights: &[ae_render::lights::Light],
    tone: ae_render::tonemap::ToneParams,
) -> Vec<u8> {
    render_particles_frames(gpu, source, camera, extent, particles, lights, tone, 8)
}

/// [`render_particles_lit`] after `frames` frames (the readback clock is
/// frame / 60 s, so the bursts are `(frames - 1) / 60` s old).
#[allow(clippy::too_many_arguments)]
fn render_particles_frames(
    gpu: &Arc<GpuDevice>,
    source: &Arc<ae_render::terrain::app::TerrainSource>,
    camera: ae_render::terrain::camera::Camera,
    extent: vk::Extent2D,
    particles: Option<(
        &[ae_render::particles::Burst],
        Vec<ae_render::particles::EffectDef>,
        u32,
        ae_render::particles::ParticleBlend,
    )>,
    lights: &[ae_render::lights::Light],
    tone: ae_render::tonemap::ToneParams,
    frames: usize,
) -> Vec<u8> {
    use ae_render::particles::{ParticleFeed, ParticlesSetup};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput};
    use ae_render::thread::{Headless, TargetInfo};
    let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
        .with_lights(lights.to_vec())
        .with_tone(tone);
    if let Some((bursts, effects, downscale, blend)) = particles {
        let feed = Arc::new(ParticleFeed::default());
        feed.emit_all(bursts);
        app = app.with_particles(ParticlesSetup {
            feed,
            capacity: 1024,
            effects,
            downscale,
            blend,
        });
    }
    {
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera = camera;
        view.fog = false;
    }
    app.create_offscreen(gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
    let mut run = Headless::new(gpu, &mut app, target, config).unwrap();
    run.frames(&mut app, frames).unwrap();
    run.wait().unwrap();
    run.finish().unwrap();
    let pixels = app.readback().unwrap();
    ae_render::thread::RenderApp::destroy(&mut app, gpu);
    pixels
}

/// Particles that hold still: one colour and size over a long life.
fn still_effect(color: [f32; 4], size: f32, emissive: bool) -> ae_render::particles::EffectDef {
    ae_render::particles::EffectDef {
        gamut: ae_render::colorspace::Gamut::Bt709,
        color_start: color,
        color_end: color,
        size: [size, size],
        life: [100.0, 100.0],
        speed: [0.0, 0.0],
        up: 0.0,
        spread: 0.0,
        gravity: 0.0,
        drag: 0.0,
        emissive,
        // The old additive value (exposed at EV14) as cd/m²: 1 / 5.086e-5.
        emission: if emissive { [19_660.8; 2] } else { [0.0; 2] },
        cover: 0.0,
        fade_in: 0.0,
        light: None,
    }
}

/// Order independence, measured (all at half resolution, against the exact
/// reference: every fragment sorted per pixel by its sphere's front depth):
/// * a dense cluster of smoke of several colours and sizes, with fire in
///   it: the error of the centre sort and of the moments, per pixel the
///   cluster covers;
/// * two puffs of different sizes crossing in depth, one metre a frame:
///   the largest change between consecutive frames (popping), and the
///   error, for each method.
/// * the same, as a volume (each fragment its sphere's chord; overlapping
///   chords mixed as media), against the flat layers in the cluster.
///
/// The numbers are printed; the assertions hold what was measured: the
/// centre sort is close to exact in the cluster, the exact per-pixel
/// order pops about as much as the sort when the two cross (the moments,
/// whose depth warp follows the frame's particle range, pop more: 372),
/// and the volume does not pop; nor do the slices or the froxels, each
/// compared with the volume (and, the froxels' blur taken out, over 8x8
/// blocks).
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn particles_order_independence() {
    use ae_render::particles::{Burst, ParticleBlend};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use glam::{Vec2, Vec3};

    let gpu = gpu();
    if !gpu.fragment_stores {
        println!("no fragmentStoresAndAtomics: no reference, skipped");
        return;
    }
    let (n, cell, scale, ground) = (65usize, 4.0f32, 100.0f32, 10.0f32);
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![ground / scale; n * n],
    };
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, cell, scale, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let mut camera = TerrainApp::new(source.clone(), TerrainOutput::Readback)
        .view()
        .lock()
        .unwrap()
        .camera;
    camera = Camera {
        target: Vec2::new(128.0, 128.0),
        ground,
        distance: 160.0,
        yaw: 0.0,
        pitch: 0.6,
        ..camera
    };
    let eye = camera.eye(|_| ground);
    let modes = [
        ParticleBlend::Sorted,
        ParticleBlend::Moments,
        ParticleBlend::Reference,
        ParticleBlend::Volume,
        ParticleBlend::Slices,
        ParticleBlend::Froxels,
        ParticleBlend::Packets,
    ];
    let render = |bursts: &[Burst], effects: &[ae_render::particles::EffectDef], blend| {
        render_particles(
            &gpu,
            &source,
            camera,
            extent,
            Some((bursts, effects.to_vec(), 2, blend)),
        )
    };
    let none = render_particles(&gpu, &source, camera, extent, None);
    // Per pixel the reference changes: mean and largest channel error.
    let error = |image: &[u8], reference: &[u8]| -> (f32, u8) {
        let (mut sum, mut count, mut worst) = (0.0, 0, 0u8);
        for i in (0..reference.len()).step_by(4) {
            if reference[i..i + 3] != none[i..i + 3] {
                count += 1;
                for c in 0..3 {
                    let e = image[i + c].abs_diff(reference[i + c]);
                    sum += e as f32;
                    worst = worst.max(e);
                }
            }
        }
        (sum / (3 * count.max(1)) as f32, worst)
    };
    // The same over blocks of a froxel column (8x8 here, at full
    // resolution), each averaged first: the error with the columns' blur
    // taken out (blocks the reference changes).
    let block_error = |image: &[u8], reference: &[u8]| -> (f32, f32) {
        let (w, h) = (extent.width as usize, extent.height as usize);
        let (mut sum, mut count, mut worst) = (0.0, 0, 0.0f32);
        for by in (0..h).step_by(8) {
            for bx in (0..w).step_by(8) {
                let (mut a, mut r, mut changed, mut n) = ([0.0f32; 3], [0.0f32; 3], false, 0.0);
                for y in by..(by + 8).min(h) {
                    for x in bx..(bx + 8).min(w) {
                        let i = (y * w + x) * 4;
                        changed |= reference[i..i + 3] != none[i..i + 3];
                        for c in 0..3 {
                            a[c] += image[i + c] as f32;
                            r[c] += reference[i + c] as f32;
                        }
                        n += 1.0;
                    }
                }
                if changed {
                    count += 1;
                    for c in 0..3 {
                        let e = (a[c] - r[c]).abs() / n;
                        sum += e;
                        worst = worst.max(e);
                    }
                }
            }
        }
        (sum / (3 * count.max(1)) as f32, worst)
    };
    let magenta = |image: &[u8]| image.chunks(4).filter(|p| p[0] > 250 && p[1] < 5 && p[2] > 250).count();

    // A cluster: 48 puffs of smoke (dark, light, brown, bluish) of 8 to 30 m
    // in a ball of 20 m, and 10 fire puffs in it.
    let effects = vec![
        still_effect([0.08, 0.08, 0.08, 0.75], 1.0, false),
        still_effect([0.7, 0.7, 0.7, 0.6], 1.0, false),
        still_effect([0.45, 0.3, 0.15, 0.7], 1.0, false),
        still_effect([0.3, 0.4, 0.7, 0.5], 1.0, false),
        still_effect([1.8, 0.8, 0.2, 0.9], 1.0, true),
    ];
    let centre = Vec3::new(128.0, 45.0, 128.0);
    let mut state = 12345u32;
    let mut rand = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as f32 / u32::MAX as f32
    };
    let mut cluster = Vec::new();
    for k in 0..58u32 {
        let offset = Vec3::new(rand() - 0.5, rand() - 0.5, rand() - 0.5) * 40.0;
        let effect = if k < 48 { k % 4 } else { 4 };
        let size = if k < 48 {
            8.0 + 22.0 * rand()
        } else {
            6.0 + 6.0 * rand()
        };
        cluster.push(Burst {
            pos: (centre + offset).to_array(),
            ground: false,
            effect,
            count: 1,
            scale: size,
            seed: k,
        });
    }
    let images: Vec<Vec<u8>> = modes.iter().map(|&m| render(&cluster, &effects, m)).collect();
    let reference = &images[2];
    assert_eq!(magenta(reference), 0, "the reference overflows its lists");
    let covered = reference
        .chunks(4)
        .zip(none.chunks(4))
        .filter(|(a, b)| a[..3] != b[..3])
        .count();
    let (sorted_mean, sorted_worst) = error(&images[0], reference);
    let (moments_mean, moments_worst) = error(&images[1], reference);
    let (volume_mean, volume_worst) = error(&images[3], reference);
    let (packets_mean, packets_worst) = error(&images[6], reference);
    assert_eq!(magenta(&images[3]), 0, "the volume overflows its lists");
    if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
        for (m, image) in modes.iter().zip(&images) {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(image.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            std::fs::write(
                std::path::Path::new(&dir).join(format!("particles-cluster-{m:?}.ppm")),
                ppm,
            )
            .unwrap();
        }
    }
    for (k, m) in modes.iter().enumerate() {
        let (mean, max) = error(&images[k], &images[3]);
        let (bmean, bmax) = block_error(&images[k], &images[3]);
        println!(
            "cluster, {m:?} against the volume: mean {mean:.2} max {max}; in 8x8 blocks mean {bmean:.2} max {bmax:.1}"
        );
    }
    // Measured: the slices 0.57 (max 10) from the volume; the layers 1.6.
    let (slices_mean, slices_max) = error(&images[4], &images[3]);
    assert!(
        slices_mean < 1.2 && slices_max < 30,
        "cluster, the slices against the volume: mean {slices_mean} max {slices_max}"
    );
    // Measured: the froxels 6.4 (max 55); 4.5 over 8x8 blocks.
    let (froxels_mean, _) = error(&images[5], &images[3]);
    assert!(
        froxels_mean < 9.0,
        "cluster, the froxels against the volume: {froxels_mean}"
    );
    println!(
        "cluster ({covered} pixels): sorted mean {sorted_mean:.2} max {sorted_worst}; moments mean {moments_mean:.2} max {moments_worst}; volume (against the layers) mean {volume_mean:.2} max {volume_worst}; packets mean {packets_mean:.2} max {packets_worst}"
    );
    // Emitted in reverse: the order-independent methods give the same frame.
    let reversed: Vec<Burst> = cluster.iter().rev().cloned().collect();
    for (m, image) in modes.iter().zip(&images) {
        let again = render(&reversed, &effects, *m);
        let worst = again.iter().zip(image).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
        println!("  {m:?} emitted in reverse: largest change {worst}");
        // Sorted: depth-key ties fall back to the slot. Moments: float sums
        // in another order (the additive blend rounds differently).
        if *m != ParticleBlend::Sorted {
            // 3 codes with no tone curve (2 under Lottes, whose shoulder
            // halved the bright differences): float sums in another order.
            assert!(worst <= 3, "{m:?} depends on the emission order: {worst}");
        }
    }

    // Crossing: a red puff of 16 m and a blue one of 26 m, 3 m apart
    // sideways, the blue one moving from 14 m in front of the red to 14 m
    // behind it along the line of sight.
    let effects = vec![
        still_effect([0.9, 0.08, 0.05, 0.85], 16.0, false),
        still_effect([0.08, 0.15, 0.9, 0.85], 26.0, false),
    ];
    let red = centre;
    let dir = (red - eye).normalize();
    let side = dir.cross(Vec3::Y).normalize() * 3.0;
    let steps: Vec<f32> = (0..=28).map(|i| i as f32 - 14.0).collect();
    let mut frames: Vec<Vec<Vec<u8>>> = vec![Vec::new(); modes.len()];
    for &t in &steps {
        let blue = red + side + dir * t;
        let bursts = [
            Burst {
                pos: red.to_array(),
                ground: false,
                effect: 0,
                count: 1,
                scale: 1.0,
                seed: 1,
            },
            Burst {
                pos: blue.to_array(),
                ground: false,
                effect: 1,
                count: 1,
                scale: 1.0,
                seed: 2,
            },
        ];
        for (k, &m) in modes.iter().enumerate() {
            frames[k].push(render(&bursts, &effects, m));
        }
    }
    let pop = |frames: &[Vec<u8>]| -> (f32, usize) {
        // The largest per-pixel channel change between consecutive frames
        // (a red-blue swap keeps the luma), and how many pixels change by
        // more than 40 in one frame.
        let (mut worst, mut jumps) = (0.0f32, 0);
        for w in frames.windows(2) {
            for i in (0..w[0].len()).step_by(4) {
                let d = (0..3).map(|c| w[1][i + c].abs_diff(w[0][i + c])).max().unwrap_or(0) as f32;
                worst = worst.max(d);
                if d > 40.0 {
                    jumps += 1;
                }
            }
        }
        (worst, jumps)
    };
    // Apart (25 m in front, 25 m behind: the chords never overlap) the
    // volume is the layers' "over", and the order shows: it keeps depth
    // order, it is not smooth by ignoring it.
    let apart: Vec<[Vec<u8>; 2]> = [-25.0f32, 25.0]
        .iter()
        .map(|&t| {
            let bursts = [
                Burst {
                    pos: red.to_array(),
                    ground: false,
                    effect: 0,
                    count: 1,
                    scale: 1.0,
                    seed: 1,
                },
                Burst {
                    pos: (red + side + dir * t).to_array(),
                    ground: false,
                    effect: 1,
                    count: 1,
                    scale: 1.0,
                    seed: 2,
                },
            ];
            [
                render(&bursts, &effects, ParticleBlend::Reference),
                render(&bursts, &effects, ParticleBlend::Volume),
            ]
        })
        .collect();
    for (t, [layers, volume]) in [-25, 25].iter().zip(&apart) {
        let (mean, max) = error(volume, layers);
        println!("apart ({t} m): volume against the layers mean {mean:.2} max {max}");
        assert!(
            mean < 1.0 && max <= 8,
            "apart, the volume is not the layers' over: mean {mean} max {max}"
        );
    }
    let swap = apart[0][1]
        .iter()
        .zip(&apart[1][1])
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    println!("apart: the volume's largest change between blue in front and behind {swap}");
    assert!(swap > 60, "the volume loses the order: {swap}");

    let mut pops = Vec::new();
    let mut against_volume = Vec::new();
    for (k, m) in modes.iter().enumerate() {
        let (worst, jumps) = pop(&frames[k]);
        let errors: Vec<(f32, u8)> = frames[k].iter().zip(&frames[2]).map(|(a, r)| error(a, r)).collect();
        let mean = errors.iter().map(|e| e.0).sum::<f32>() / errors.len() as f32;
        let max = errors.iter().map(|e| e.1).max().unwrap_or(0);
        let errors: Vec<(f32, u8)> = frames[k].iter().zip(&frames[3]).map(|(a, r)| error(a, r)).collect();
        let vmean = errors.iter().map(|e| e.0).sum::<f32>() / errors.len() as f32;
        let vmax = errors.iter().map(|e| e.1).max().unwrap_or(0);
        against_volume.push((vmean, vmax));
        let blocks: Vec<(f32, f32)> = frames[k]
            .iter()
            .zip(&frames[3])
            .map(|(a, r)| block_error(a, r))
            .collect();
        let bmean = blocks.iter().map(|e| e.0).sum::<f32>() / blocks.len() as f32;
        let bmax = blocks.iter().map(|e| e.1).fold(0.0f32, f32::max);
        println!(
            "crossing, {m:?}: largest frame-to-frame change {worst:.1} ({jumps} pixel-frames over 40); error against the layers mean {mean:.2} max {max}, against the volume mean {vmean:.2} max {vmax}, in 8x8 blocks mean {bmean:.2} max {bmax:.1}"
        );
        pops.push((worst, jumps));
    }
    for frame in frames.iter().take(modes.len()) {
        assert_eq!(frame.iter().map(|f| magenta(f)).sum::<usize>(), 0);
    }
    // Measured: in the cluster, the centre sort is within 1.6 (mean) of the
    // exact order, and the moments about as close.
    assert!(
        sorted_mean < 2.5 && moments_mean < 2.5,
        "cluster error: sorted {sorted_mean}, moments {moments_mean}"
    );
    // The packets shade the centre sort's order per pixel (like the sorted
    // draw), with the same puff and soft fade, so they are about as close
    // to the exact order as the sorted draw. Measured: 0.75 against the
    // sorted 0.74 (a candidate skipped by a packet, as when a saturated
    // lane stopped classifying its rank, showed as 0.84).
    assert!(
        packets_mean < 3.0 && (packets_mean - sorted_mean).abs() < 0.05,
        "cluster error, the packets against the reference: {packets_mean} (max {packets_worst}; sorted {sorted_mean})"
    );
    // Measured: the exact per-pixel order pops about as much as the centre
    // sort (77 against 106 pixel-frames over 40; both 73 at worst under the
    // ACES curve): the jump comes from blending particles as flat layers,
    // whatever the order.
    let (sorted_pop, reference_pop) = (pops[0], pops[2]);
    assert!(
        reference_pop.0 > 60.0 && reference_pop.1 * 2 > sorted_pop.1,
        "the reference pops much less than the sort: {reference_pop:?} against {sorted_pop:?}"
    );
    // Measured: the volume (puffs mixed by how much of their thickness
    // overlaps) changes by 7 at most between frames, no pixel by over 40.
    let volume_pop = pops[3];
    assert!(
        volume_pop.0 < 20.0 && volume_pop.1 == 0,
        "the volume pops: {volume_pop:?}"
    );
    assert!(
        volume_mean < 3.0,
        "a volume of separate puffs differs from their layers: {volume_mean}"
    );
    // Measured: the slices pop by 6, and stay within 0.19 (max 7) of the
    // volume.
    let (slices_pop, slices_error) = (pops[4], against_volume[4]);
    assert!(
        slices_pop.0 < 20.0 && slices_pop.1 == 0 && slices_error.0 < 1.0 && slices_error.1 < 30,
        "crossing, the slices: pop {slices_pop:?}, against the volume {slices_error:?}"
    );
    // Measured: the froxels pop by 10, and are 5.6 (max 42) from the volume:
    // blurred to their columns (8 pixels here), small puffs missed or caught
    // by a column's centre ray.
    let (froxels_pop, froxels_error) = (pops[5], against_volume[5]);
    assert!(
        froxels_pop.0 < 20.0 && froxels_pop.1 == 0 && froxels_error.0 < 8.0,
        "crossing, the froxels: pop {froxels_pop:?}, against the volume {froxels_error:?}"
    );
    // The packets are flat layers in the centre sort's order (like the
    // sorted draw), so they pop with the sort rather than with the volumes.
    let (packets_pop, packets_error) = (pops[6], against_volume[6]);
    assert!(
        packets_pop.0 > 40.0 && packets_error.0 < 6.0,
        "crossing, the packets: pop {packets_pop:?}, against the volume {packets_error:?}"
    );

    // Stretched: the same with a dim near puff (at a third of the pair's
    // distance: a depth range about three to one, as the battle bench's), so
    // the froxels' slices, spread over the frame's range, are thicker; and a
    // puff half in the ground.
    let mut stretched_effects = effects.clone();
    stretched_effects.push(still_effect([0.5, 0.5, 0.5, 0.2], 30.0, false));
    stretched_effects.push(still_effect([0.6, 0.5, 0.4, 0.7], 20.0, false));
    let horizontal = Vec3::new(dir.x, 0.0, dir.z).normalize();
    let near = Burst {
        pos: (eye + (Vec3::new(128.0, ground, 128.0) - eye) * 0.35).to_array(),
        ground: false,
        effect: 2,
        count: 1,
        scale: 0.4,
        seed: 3,
    };
    let buried = Burst {
        pos: (red - side * 8.0 - horizontal * 30.0 + Vec3::Y * (ground - red.y)).to_array(),
        ground: false,
        effect: 3,
        count: 1,
        scale: 1.0,
        seed: 4,
    };
    let stretched_modes = [ParticleBlend::Volume, ParticleBlend::Slices, ParticleBlend::Froxels];
    let mut stretched: Vec<Vec<Vec<u8>>> = vec![Vec::new(); stretched_modes.len()];
    for &t in &steps {
        let bursts = [
            Burst {
                pos: red.to_array(),
                ground: false,
                effect: 0,
                count: 1,
                scale: 1.0,
                seed: 1,
            },
            Burst {
                pos: (red + side + dir * t).to_array(),
                ground: false,
                effect: 1,
                count: 1,
                scale: 1.0,
                seed: 2,
            },
            near,
            buried,
        ];
        for (k, &m) in stretched_modes.iter().enumerate() {
            stretched[k].push(render(&bursts, &stretched_effects, m));
        }
    }
    if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
        for (m, frames) in stretched_modes.iter().zip(&stretched) {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(frames[14].chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            std::fs::write(
                std::path::Path::new(&dir).join(format!("particles-stretched-{m:?}.ppm")),
                ppm,
            )
            .unwrap();
        }
    }
    for (k, m) in stretched_modes.iter().enumerate() {
        let (worst, jumps) = pop(&stretched[k]);
        let errors: Vec<(f32, u8)> = stretched[k]
            .iter()
            .zip(&stretched[0])
            .map(|(a, r)| error(a, r))
            .collect();
        let mean = errors.iter().map(|e| e.0).sum::<f32>() / errors.len() as f32;
        let max = errors.iter().map(|e| e.1).max().unwrap_or(0);
        let blocks: Vec<(f32, f32)> = stretched[k]
            .iter()
            .zip(&stretched[0])
            .map(|(a, r)| block_error(a, r))
            .collect();
        let bmean = blocks.iter().map(|e| e.0).sum::<f32>() / blocks.len() as f32;
        println!(
            "stretched crossing, {m:?}: largest frame-to-frame change {worst:.1} ({jumps} pixel-frames over 40); against the volume mean {mean:.2} max {max}, in 8x8 blocks mean {bmean:.2}"
        );
        // Measured: the froxels pop by 10, 3.0 from the volume (as many
        // slices or twice as many).
        assert!(
            worst < 20.0 && jumps == 0 && mean < 6.0,
            "stretched crossing, {m:?}: pop {worst} ({jumps}), against the volume {mean}"
        );
    }
    // The buried puff alone (no pair), against the volume, which caps it at
    // the ground exactly.
    let alone: Vec<Vec<u8>> = stretched_modes
        .iter()
        .map(|&m| render(&[near, buried], &stretched_effects, m))
        .collect();
    for (k, m) in stretched_modes.iter().enumerate() {
        let (mean, max) = error(&alone[k], &alone[0]);
        let (bmean, bmax) = block_error(&alone[k], &alone[0]);
        println!(
            "buried puff, {m:?} against the volume: mean {mean:.2} max {max}; in 8x8 blocks mean {bmean:.2} max {bmax:.1}"
        );
        // Measured: the froxels 1.4 (max 16).
        assert!(mean < 3.0, "buried puff, {m:?} against the volume: {mean}");
    }
    // Apart: the froxels keep the order, stretched or not (the volume: 116).
    for (what, extra) in [("", None), (", stretched", Some(near))] {
        let swaps: Vec<u8> = [ParticleBlend::Volume, ParticleBlend::Froxels]
            .iter()
            .map(|&m| {
                let apart: Vec<Vec<u8>> = [-25.0f32, 25.0]
                    .iter()
                    .map(|&t| {
                        let mut bursts = vec![
                            Burst {
                                pos: red.to_array(),
                                ground: false,
                                effect: 0,
                                count: 1,
                                scale: 1.0,
                                seed: 1,
                            },
                            Burst {
                                pos: (red + side + dir * t).to_array(),
                                ground: false,
                                effect: 1,
                                count: 1,
                                scale: 1.0,
                                seed: 2,
                            },
                        ];
                        bursts.extend(extra);
                        render(&bursts, &stretched_effects, m)
                    })
                    .collect();
                apart[0]
                    .iter()
                    .zip(&apart[1])
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        println!(
            "apart{what}: the largest change between blue in front and behind, volume {}, froxels {}",
            swaps[0], swaps[1]
        );
        // Measured under ACES: 49 (the columns' blur lowers the peak).
        assert!(swaps[1] > 40, "apart{what}, the froxels lose the order: {}", swaps[1]);
    }
    assert_clean(&gpu);
}

/// Decals on a step heightmap (2 m cells, steps along x): a plateau 30 m
/// high ending in a 20 m cliff facing +x down to ground at 10 m, and a 6 m
/// step up to 36 m at its far side. A test atlas of eight stripes across
/// the decal's x. Decals of half-size 12 m: E on the cliff's top edge
/// (convex), F at its foot (concave), L on the low step's top edge (all
/// three centred on corner samples), G 4.5 m back from the top edge; in a
/// frame of its own, R like G but turned by 0.6 rad.
/// * Along each decal's x, unfolded over the edge (s: the length along the
///   ground from the centre), the stripes match the atlas at that s on both
///   faces of both corners: both are painted, the face continues the flat
///   ground's stripes in phase and unstretched (projections around the
///   centre put G's face out of phase; unfolding per axis bends R's).
/// * Ground past every decal's unfolded square is byte-identical to no
///   decals.
/// * A unit standing in F's footprint is not painted.
/// * Decals made but none placed draw exactly as no decals.
/// * L's ground at the foot of the 6 m step, past the unfolded footprint,
///   is not painted (the first version's three projections printed it
///   again there).
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn decals_wrap_over_cliffs() {
    use ae_render::decals::{Decal, DecalAtlas, DecalFeed, DecalsSetup, colour_texel};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{Headless, TargetInfo};
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Vec2, Vec3};

    let gpu = gpu();
    let (n, cell, scale) = (65usize, 2.0f32, 100.0f32);
    let column = |x: usize| {
        if x <= 12 {
            36.0
        } else if x <= 32 {
            30.0
        } else {
            10.0
        }
    };
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: (0..n * n).map(|i| column(i % n) / scale).collect(),
    };
    // The surface at world x (linear across a step's cell).
    let height = |x: f32| {
        let c = (x / cell).clamp(0.0, (n - 1) as f32);
        let (i, f) = (c.floor() as usize, c.fract());
        column(i) + (column((i + 1).min(n - 1)) - column(i)) * f
    };
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, cell, scale, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 480,
        height: 360,
    };
    let mut camera = TerrainApp::new(source.clone(), TerrainOutput::Readback)
        .view()
        .lock()
        .unwrap()
        .camera;
    camera = Camera {
        target: Vec2::new(56.0, 64.0),
        ground: 30.0,
        distance: 95.0,
        yaw: std::f32::consts::FRAC_PI_2,
        pitch: 0.42,
        ..camera
    };
    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let pixel = |p: Vec3| -> Option<usize> {
        let clip = view_proj * p.extend(1.0);
        let (u, v) = (
            (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
            (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
        );
        (clip.w > 0.0 && u >= 2.0 && v >= 2.0 && u < extent.width as f32 - 2.0 && v < extent.height as f32 - 2.0)
            .then(|| (v as usize * extent.width as usize + u as usize) * 4)
    };

    // Eight stripes across the atlas's x, black and white, opaque.
    const STRIPES: f32 = 8.0;
    let size = 64u32;
    let stripes = (0..size * size)
        .map(|i| {
            colour_texel(if ((i % size) as f32 / size as f32 * STRIPES) as u32 % 2 == 1 {
                [0.9, 0.9, 0.9, 1.0]
            } else {
                [0.02, 0.02, 0.02, 1.0]
            })
        })
        .collect();
    let atlas = Arc::new(DecalAtlas {
        size,
        colour: vec![stripes],
        surface: Vec::new(),
        height_layers: Vec::new(),
        materials: Vec::new(),
    });
    let half = 12.0f32;
    let decal = |x: f32, z: f32| Decal::permanent([x, z], [half, half], 0.0, 0);
    // Centres on corner samples: E (32, 16), F (33, 48), L (12, 32); G
    // 4.5 m back from the cliff's top edge (a stripe and a half: projections
    // around the centre would put the face's stripes in opposite phase).
    let centres = [(64.0f32, 32.0f32), (66.0, 96.0), (24.0, 64.0), (59.5, 66.0)];
    let centre_y = [30.0f32, 10.0, 36.0, 30.0];
    let unit = (74.0f32, 104.0f32);
    const ROTATION: f32 = 0.6;
    let rotated = (59.5f32, 32.0f32);

    #[derive(Clone, Copy, PartialEq)]
    enum Decals {
        None,
        Empty,
        On,
        /// Placed at the first frame (time 0) to live 0.05 s: gone by the
        /// last (7 / 60 s).
        Expired,
        /// Halfway through a 1 s fade-out at the last frame.
        Fading,
        /// Only R: turned by 0.6 rad, 4.5 m back from the cliff's top edge.
        Rotated,
    }
    let render = |decals: Decals| -> Vec<u8> {
        let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
        for (k, word) in [unit.0, unit.1, unit.0, unit.1]
            .map(f32::to_bits)
            .into_iter()
            .chain([0.0, 0.0, 0.0, 1.0].map(f32::to_bits))
            .chain([0.0, 0.0, 0.0, 1.0].map(f32::to_bits))
            .chain([4.0f32.to_bits(), 0])
            .enumerate()
        {
            columns[k].push(word);
        }
        let feed = Arc::new(UnitFeed::default());
        feed.publish(Arc::new(InstanceStream {
            tick: 1,
            published: std::time::Instant::now(),
            tick_seconds: 0.0,
            sim_seconds: 0.0,
            columns: at_rest(columns),
            anim: AnimWrites::default(),
        }));
        let classes = vec![UnitClass {
            gamut: ae_render::colorspace::Gamut::Bt709,
            color: [0.45, 0.45, 0.45],
            floats: false,
        }];
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(UnitsSetup {
            feed,
            capacity: 16,
            classes,
            min_pixels: 6.0,
            models: Vec::new(),
            rigs: Vec::new(),
        });
        if decals != Decals::None {
            let feed = Arc::new(DecalFeed::default());
            let life = match decals {
                Decals::Expired => Some((0.05, 0.0)),
                Decals::Fading => Some((7.0 / 60.0 + 0.5, 1.0)),
                _ => None,
            };
            if decals == Decals::Rotated {
                feed.place_all(&[Decal::permanent([rotated.0, rotated.1], [half, half], ROTATION, 0)]);
            } else if decals != Decals::Empty {
                feed.place_all(&centres.map(|(x, z)| match life {
                    Some((life, fade_out)) => Decal {
                        life,
                        fade_out,
                        ..decal(x, z)
                    },
                    None => decal(x, z),
                }));
            }
            app = app.with_decals(DecalsSetup {
                atlas: Some(atlas.clone()),
                ..DecalsSetup::new(feed, 64)
            });
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera = camera;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        let mut run = Headless::new(&gpu, &mut app, target, config).unwrap();
        run.frames(&mut app, 8).unwrap();
        run.wait().unwrap();
        run.finish().unwrap();
        let pixels = app.readback().unwrap();
        if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            std::fs::write(
                std::path::Path::new(&dir).join(format!("decals-{}.ppm", decals as u8)),
                ppm,
            )
            .unwrap();
        }
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let none = render(Decals::None);
    let empty = render(Decals::Empty);
    let on = render(Decals::On);
    let expired = render(Decals::Expired);
    let fading = render(Decals::Fading);
    let turned = render(Decals::Rotated);
    let rgb = |p: &[u8], i: usize| [p[i], p[i + 1], p[i + 2]];
    let luma = |p: &[u8], i: usize| p[i] as f32 * 0.3 + p[i + 1] as f32 * 0.59 + p[i + 2] as f32 * 0.11;
    assert_eq!(
        none.iter().zip(&empty).filter(|(a, b)| a != b).count(),
        0,
        "decals made but none placed draw as none"
    );
    assert_eq!(
        none.iter().zip(&expired).filter(|(a, b)| a != b).count(),
        0,
        "decals past their life draw as none"
    );

    // A point's distance from decal k's centre, as the shader fades it:
    // the square in unfolded metres (the length along the ground from the
    // centre line along x; the ground varies only along x), 1 at the edge.
    let extent3 = |p: Vec3, k: usize| {
        let (cx, cz) = centres[k];
        let steps = ((p.x - cx).abs() / 0.01).ceil().max(1.0) as usize;
        let dx = (p.x - cx) / steps as f32;
        let arc: f32 = (0..steps)
            .map(|i| dx.hypot(height(cx + dx * (i + 1) as f32) - height(cx + dx * i as f32)))
            .sum();
        arc.max((p.z - cz).abs()) / half
    };
    // The old extent (square in plan, rounded in the vertical), which let L
    // print again at the foot of its step.
    let rounded = |p: Vec3, k: usize| {
        let (cx, cz) = centres[k];
        let d = p - Vec3::new(cx, centre_y[k], cz);
        (d.x.abs().max(d.z.abs())).hypot(d.y) / half
    };
    // The stripe the atlas has at unfolded coordinate s (1 light).
    let expected = |s: f32| ((s / half * 0.5 + 0.5) * STRIPES).floor() as i32 % 2 == 1;
    let near_boundary = |s: f32| {
        let k = (s / half * 0.5 + 0.5) * STRIPES;
        (k - k.round()).abs() * 2.0 * half / STRIPES < 0.5
    };
    // Segments: name, points (world, s). Rows at three z across each decal.
    type Segment = (&'static str, Vec<(Vec3, f32)>);
    let mut segments: Vec<Segment> = Vec::new();
    for ((cx, cz), top) in [(centres[0], true), (centres[1], false)] {
        let (mut flat, mut face) = (Vec::new(), Vec::new());
        for dz in [-6.0f32, -2.0, 1.0] {
            let z = cz + dz;
            for k in 0..=100 {
                let t = k as f32 / 100.0;
                if top {
                    // The plateau before the edge (its last cell's normal
                    // already leans), then down the face 0.5..11 m.
                    let x = cx - half + t * (half - 2.5);
                    flat.push((Vec3::new(x, 30.0, z), x - cx));
                    let drop = 0.5 + t * 10.5;
                    face.push((Vec3::new(64.0 + 2.0 * drop / 20.0, 30.0 - drop, z), drop));
                } else {
                    let x = cx + 2.5 + t * (half - 2.5);
                    flat.push((Vec3::new(x, 10.0, z), x - cx));
                    let rise = 0.5 + t * 10.5;
                    face.push((Vec3::new(66.0 - 2.0 * rise / 20.0, 10.0 + rise, z), -rise));
                }
            }
        }
        segments.push((if top { "E top" } else { "F ground" }, flat));
        segments.push((if top { "E face" } else { "F face" }, face));
    }
    // G: the plateau up to the edge 4.5 m out, then 0.5..7 m down the
    // face, continuing from 4.5 m.
    let (mut flat, mut face) = (Vec::new(), Vec::new());
    let (gx, gz) = centres[3];
    for dz in [-6.0f32, -2.0, 1.0] {
        for k in 0..=100 {
            let t = k as f32 / 100.0;
            let x = gx - half + t * (half + 4.5 - 2.5);
            flat.push((Vec3::new(x, 30.0, gz + dz), x - gx));
            let drop = 0.5 + t * 6.5;
            let fx = 64.0 + 2.0 * drop / 20.0;
            face.push((Vec3::new(fx, 30.0 - drop, gz + dz), 4.5 + (fx - 64.0).hypot(drop)));
        }
    }
    segments.push(("G top", flat));
    segments.push(("G face", face));
    for (name, points) in &segments {
        let samples: Vec<(f32, bool)> = points
            .iter()
            .filter(|(p, s)| {
                !near_boundary(*s)
                    && (p.x - unit.0).hypot(p.z - unit.1) > 6.0
                    && (0..centres.len()).any(|k| extent3(*p, k) < 0.85)
            })
            .filter_map(|&(p, s)| pixel(p).map(|i| (luma(&on, i), expected(s))))
            .collect();
        let (lo, hi) = samples
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &(l, _)| (lo.min(l), hi.max(l)));
        let mid = (lo + hi) * 0.5;
        let matching = samples.iter().filter(|&&(l, e)| (l > mid) == e).count();
        // Both stripes show, far apart.
        let lights = samples.iter().filter(|&&(_, e)| e).count();
        println!(
            "{name}: {matching} of {} samples match the unfolded stripes (luma {lo:.0}..{hi:.0}, {lights} light)",
            samples.len()
        );
        assert!(
            samples.len() > 120 && lights > 30 && lights < samples.len() - 30,
            "{name}: {} samples, {lights} light",
            samples.len()
        );
        assert!(hi > 2.0 * lo.max(1.0), "{name}: no stripes (luma {lo:.0}..{hi:.0})");
        assert!(
            matching * 100 >= samples.len() * 95,
            "{name}: {matching} of {} match",
            samples.len()
        );
    }

    // R, turned across the cliff: the ground unfolded in world (x by its
    // length along the ground from R's centre, the cliff running along z),
    // then turned into R's frame; the stripes run across its u.
    {
        let (sn, cs) = ROTATION.sin_cos();
        let (rx, rz) = rotated;
        let arc = |x: f32| {
            let steps = ((x - rx).abs() / 0.01).ceil().max(1.0) as usize;
            let dx = (x - rx) / steps as f32;
            (0..steps)
                .map(|i| dx.hypot(height(rx + dx * (i + 1) as f32) - height(rx + dx * i as f32)))
                .sum::<f32>()
                .copysign(x - rx)
        };
        for (name, on_face) in [("R top", false), ("R face", true)] {
            let mut samples = Vec::new();
            for k in 0..=40 {
                for j in 0..=40 {
                    let dz = -half + j as f32 * half / 20.0;
                    let p = if on_face {
                        let drop = 0.5 + k as f32 * 9.0 / 40.0;
                        Vec3::new(64.0 + 2.0 * drop / 20.0, 30.0 - drop, rz + dz)
                    } else {
                        Vec3::new(rx - half + k as f32 * (half + 4.5 - 2.5) / 40.0, 30.0, rz + dz)
                    };
                    let w = arc(p.x);
                    let (u, v) = (cs * w + sn * dz, -sn * w + cs * dz);
                    if u.abs().max(v.abs()) > 0.85 * half || near_boundary(u) {
                        continue;
                    }
                    if let Some(i) = pixel(p) {
                        samples.push((luma(&turned, i), expected(u)));
                    }
                }
            }
            let (lo, hi) = samples
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), &(l, _)| (lo.min(l), hi.max(l)));
            let mid = (lo + hi) * 0.5;
            let matching = samples.iter().filter(|&&(l, e)| (l > mid) == e).count();
            let lights = samples.iter().filter(|&&(_, e)| e).count();
            println!(
                "{name}: {matching} of {} samples match the unfolded stripes (luma {lo:.0}..{hi:.0}, {lights} light)",
                samples.len()
            );
            assert!(
                samples.len() > 120 && lights > 30 && lights < samples.len() - 30,
                "{name}: {} samples, {lights} light",
                samples.len()
            );
            assert!(
                matching * 100 >= samples.len() * 95,
                "{name}: {matching} of {} match",
                samples.len()
            );
        }
    }

    // Halfway through the fade-out: every painted sample of the segments
    // between none and full (the median at a quarter to three quarters of
    // the way, in sRGB bytes).
    let mut between: Vec<f32> = segments
        .iter()
        .flat_map(|(_, points)| points.iter())
        .filter_map(|&(p, _)| pixel(p))
        .filter(|&i| (luma(&on, i) - luma(&none, i)).abs() > 20.0)
        .map(|i| (luma(&fading, i) - luma(&none, i)) / (luma(&on, i) - luma(&none, i)))
        .collect();
    between.sort_by(f32::total_cmp);
    let median = between[between.len() / 2];
    println!(
        "fading: {} samples, {:.2}..{:.2} of the way, median {median:.2}",
        between.len(),
        between[0],
        between[between.len() - 1]
    );
    assert!(
        between.len() > 300 && (0.25..0.75).contains(&median),
        "{} samples, median {median}",
        between.len()
    );
    let inside = between.iter().filter(|&&f| f > 0.05 && f < 0.95).count();
    assert!(
        inside * 10 >= between.len() * 9,
        "{inside} of {} samples between none and full",
        between.len()
    );

    // The ground on a 0.5 m grid: outside every decal's 3D extent (with a
    // margin of a few pixels) it is unchanged.
    let near_unit = |p: Vec3| (p.x - unit.0).hypot(p.z - unit.1) < 6.0;
    let (mut outside, mut changed, mut copy, mut copy_changed) = (0, 0, 0, 0);
    for iz in 0..256 {
        for ix in 0..256 {
            let (x, z) = (ix as f32 * 0.5, iz as f32 * 0.5);
            let p = Vec3::new(x, height(x), z);
            if near_unit(p) {
                continue;
            }
            let Some(i) = pixel(p) else { continue };
            let r = (0..centres.len()).map(|k| extent3(p, k)).fold(f32::MAX, f32::min);
            if r > 1.06 {
                outside += 1;
                // More than one code: a decal's light, not float noise from
                // the decal path landing on a rounding step (one point, one
                // channel, 102 -> 101, measured with no tone curve).
                changed += (0..3).any(|c| none[i + c].abs_diff(on[i + c]) > 1) as usize;
            }
            // L's copy: the foot of the 6 m step (x past 28, its leaning
            // cell), within its 3D extent, past its unfolded footprint
            // (6 m down the face, then along the ground).
            let lz = centres[2].1;
            if x > 28.0
                && p.y == 30.0
                && rounded(p, 2) < 0.94
                && (x - 26.0) + 6.0 > half + 0.5
                && (z - lz).abs() < half - 1.0
            {
                copy += 1;
                copy_changed += (rgb(&none, i) != rgb(&on, i)) as usize;
            }
        }
    }
    println!(
        "{changed} of {outside} points outside the decals changed; L's copy past its unfolded footprint: {copy_changed} of {copy} points"
    );
    assert!(outside > 30_000, "{outside} points outside");
    assert_eq!(changed, 0, "ground outside the decals changes by more than a code");
    assert!(copy > 50, "{copy} points past L's unfolded footprint");
    assert_eq!(copy_changed, 0, "L prints again at the foot of its step");

    // The unit in F's footprint: its top unchanged, the ground beside it
    // painted.
    let top = pixel(Vec3::new(unit.0, 10.0 + 3.0, unit.1)).expect("unit in view");
    assert_eq!(rgb(&none, top), rgb(&on, top), "the unit is painted");
    let beside: Vec<usize> = (0..40)
        .filter_map(|k| pixel(Vec3::new(unit.0 - 6.0 + k as f32 * 0.1, 10.0, unit.1 - 6.0)))
        .collect();
    assert!(
        beside.iter().any(|&i| rgb(&none, i) != rgb(&on, i)),
        "the ground by the unit is not painted"
    );
    assert_clean(&gpu);
}

/// Decal shading (Recoil's scars) on a plateau 30 m high over ground at
/// 10 m (2 m cells): its west cliff faces -x, toward the sun
/// (-0.5, 0.55, 0.35), its east edge drops into the hill's shadow.
/// * Relit by its normal map: a decal whose normals all lean toward its +x
///   (u), centred on the west cliff's top edge, against the same decal with
///   flat normals. At angle 0 the plateau leans away from the sun (darker)
///   and the face, unfolded downward, leans up toward it (brighter);
///   turned by pi both reverse. A sign error in the tangent frame on
///   either fails one of the four. A decal leaning toward +v turned by
///   pi / 2 lights as the one leaning +x turned by pi (the green channel's
///   sign).
/// * Modulate: a mid-grey (sRGB 0.5) decal leaves the ground as it is; a
///   quarter grey (sRGB 0.25) halves it (in sRGB bytes), Recoil's
///   `2 * decal * ground`.
/// * Glow: a glowing decal in the hill's shadow brightens its centre and
///   reddens it, also while the decal fades in; once its glow life is
///   over the frame equals the same decal without glow.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn decals_relit_modulated_and_glowing() {
    use ae_render::decals::{Blend, Decal, DecalAtlas, DecalFeed, DecalsSetup, colour_texel, surface_texel};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{Headless, TargetInfo};
    use glam::{Vec2, Vec3};

    let gpu = gpu();
    let (n, scale) = (65usize, 100.0f32);
    let column = |x: usize| if (21..=40).contains(&x) { 30.0 } else { 10.0 };
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: (0..n * n).map(|i| column(i % n) / scale).collect(),
    };
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 2.0, scale, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 480,
        height: 360,
    };
    let base = TerrainApp::new(source.clone(), TerrainOutput::Readback)
        .view()
        .lock()
        .unwrap()
        .camera;
    // From the west for the cliff, from the east for the shadowed ground.
    let west = Camera {
        target: Vec2::new(44.0, 64.0),
        ground: 25.0,
        distance: 70.0,
        yaw: -std::f32::consts::FRAC_PI_2,
        pitch: 0.42,
        ..base
    };
    let east = Camera {
        target: Vec2::new(88.0, 64.0),
        ground: 10.0,
        distance: 60.0,
        yaw: std::f32::consts::FRAC_PI_2,
        pitch: 0.7,
        ..base
    };
    let pixel = |camera: &Camera, p: Vec3| -> Option<usize> {
        let view_proj = lod_view(&source, camera, extent.width, extent.height).view_proj;
        let clip = view_proj * p.extend(1.0);
        let (u, v) = (
            (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
            (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
        );
        (clip.w > 0.0 && u >= 2.0 && v >= 2.0 && u < extent.width as f32 - 2.0 && v < extent.height as f32 - 2.0)
            .then(|| (v as usize * extent.width as usize + u as usize) * 4)
    };

    // Layers: 0 grey leaning toward +u; 1 grey, flat, glow mask 0.6 (Recoil's
    // scars peak at 0.67 and 0.95); 2 sRGB
    // 0.5; 3 sRGB 0.25 (both flat); 4 grey leaning toward +v.
    let size = 32u32;
    let texels = (size * size) as usize;
    let linear = |s: f32| ((s + 0.055) / 1.055).powf(2.4);
    let grey = |c: f32| vec![colour_texel([c, c, c, 1.0]); texels];
    let lean = 0.5f32.atan();
    let colour = vec![grey(0.3), grey(0.3), grey(linear(0.5)), grey(linear(0.25)), grey(0.3)];
    let surface = |n: [f32; 3], glow: f32| vec![surface_texel(n, 1.0, glow); texels];
    let flat = surface([0.0, 0.0, 1.0], 0.0);
    let surface = vec![
        surface([lean.sin(), 0.0, lean.cos()], 0.0),
        surface([0.0, 0.0, 1.0], 0.6),
        flat.clone(),
        flat,
        surface([0.0, lean.sin(), lean.cos()], 0.0),
    ];
    let atlas = Arc::new(DecalAtlas {
        size,
        colour,
        surface,
        height_layers: Vec::new(),
        materials: Vec::new(),
    });
    let half = 12.0f32;
    let edge = (42.0f32, 64.0f32);
    let at_edge = |layer: u32, angle: f32| Decal::permanent([edge.0, edge.1], [half, half], angle, layer);
    let glow_at = (88.0f32, 64.0f32);
    let glowing = |glow: f32, glow_life: f32| Decal {
        glow,
        glow_life,
        ..Decal::permanent([glow_at.0, glow_at.1], [6.0, 6.0], 0.0, 1)
    };

    let render = |decals: &[Decal], camera: Camera| -> Vec<u8> {
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_shadows();
        let feed = Arc::new(DecalFeed::default());
        feed.place_all(decals);
        app = app.with_decals(DecalsSetup {
            atlas: Some(atlas.clone()),
            ..DecalsSetup::new(feed, 16)
        });
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera = camera;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        let mut run = Headless::new(&gpu, &mut app, target, config).unwrap();
        run.frames(&mut app, 8).unwrap();
        run.wait().unwrap();
        run.finish().unwrap();
        let pixels = app.readback().unwrap();
        if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            let name = decals.first().map_or("none".to_string(), |d| {
                format!("{}-{}-{:.1}-{}", d.layer, d.angle, d.glow, d.glow_life)
            });
            std::fs::write(
                std::path::Path::new(&dir).join(format!("decal-shading-{name}.ppm")),
                ppm,
            )
            .unwrap();
        }
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let luma = |p: &[u8], i: usize| p[i] as f32 * 0.3 + p[i + 1] as f32 * 0.59 + p[i + 2] as f32 * 0.11;
    let median = |mut v: Vec<f32>| {
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };

    // The plateau 2.5..9 m in from the edge, and the face 1..10 m down.
    let (mut top, mut face) = (Vec::new(), Vec::new());
    for dz in [-6.0f32, -2.0, 3.0] {
        for k in 0..=40 {
            let t = k as f32 / 40.0;
            top.push(Vec3::new(edge.0 + 2.5 + t * 6.5, 30.0, edge.1 + dz));
            let drop = 1.0 + t * 9.0;
            face.push(Vec3::new(40.0 + 2.0 * (1.0 - drop / 20.0), 30.0 - drop, edge.1 + dz));
        }
    }
    let lumas = |frame: &[u8], points: &[Vec3]| -> Vec<f32> {
        points
            .iter()
            .filter_map(|&p| pixel(&west, p))
            .map(|i| luma(frame, i))
            .collect()
    };
    let flat = render(&[at_edge(1, 0.0)], west);
    let lean0 = render(&[at_edge(0, 0.0)], west);
    let lean_pi = render(&[at_edge(0, std::f32::consts::PI)], west);
    // Leaning toward +v, turned by pi / 2: its +v is the world's -x, as
    // the decal leaning +x turned by pi (the green channel's sign).
    let lean_v = render(&[at_edge(4, std::f32::consts::FRAC_PI_2)], west);
    let none = render(&[], west);
    for (name, points, sign) in [("plateau", &top, -1.0f32), ("face", &face, 1.0)] {
        let (f, a, b, v) = (
            median(lumas(&flat, points)),
            median(lumas(&lean0, points)),
            median(lumas(&lean_pi, points)),
            median(lumas(&lean_v, points)),
        );
        println!(
            "relit {name}: flat {f:.0}, leaning +x {a:.0}, turned by pi {b:.0}, leaning +v turned by pi/2 {v:.0} ({} samples)",
            lumas(&flat, points).len()
        );
        assert!(lumas(&flat, points).len() > 80, "{name} out of view");
        assert!(
            sign * (a - f) > 8.0 && sign * (f - b) > 8.0,
            "{name}: flat {f}, angle 0 {a}, angle pi {b}"
        );
        assert!(
            (v - b).abs() < 6.0,
            "{name}: leaning +v at pi / 2 {v}, leaning +x at pi {b}"
        );
    }

    // Modulate on the plateau and the face.
    let points: Vec<Vec3> = top.iter().chain(&face).copied().collect();
    let modulated = |layer| {
        render(
            &[Decal {
                blend: Blend::Modulate,
                ..at_edge(layer, 0.0)
            }],
            west,
        )
    };
    let (mid, quarter) = (modulated(2), modulated(3));
    let ground = lumas(&none, &points);
    let mid_ratio = median(lumas(&mid, &points).iter().zip(&ground).map(|(m, g)| m / g).collect());
    let quarter_ratio = median(
        lumas(&quarter, &points)
            .iter()
            .zip(&ground)
            .map(|(m, g)| m / g)
            .collect(),
    );
    println!("modulate: sRGB 0.5 leaves {mid_ratio:.3} of the ground, 0.25 leaves {quarter_ratio:.3}");
    assert!(
        (mid_ratio - 1.0).abs() < 0.04,
        "mid grey changes the ground: {mid_ratio}"
    );
    // The specular's ambient floor does not scale with the albedo (M12
    // step 3): the quarter grey ground lands higher than the diffuse-only
    // 0.51 the first version pinned.
    assert!((0.42..0.68).contains(&quarter_ratio), "quarter grey: {quarter_ratio}");

    // Glow in the hill's shadow.
    let dark = render(&[glowing(0.0, 0.0)], east);
    let lit = render(&[glowing(1.0, 100.0)], east);
    let ended = render(&[glowing(1.0, 0.05)], east);
    // Fading in over 0.4 s (as the client's scars), read at 0.12 s: the
    // glow is not dimmed by it.
    let fading_in = render(
        &[Decal {
            fade_in: 0.4,
            ..glowing(1.0, 100.0)
        }],
        east,
    );
    let centre = pixel(&east, Vec3::new(glow_at.0, 10.0, glow_at.1)).expect("glow in view");
    let sunny = pixel(&east, Vec3::new(112.0, 10.0, glow_at.1)).expect("sunlit ground in view");
    let (d, l) = (luma(&dark, centre), luma(&lit, centre));
    println!(
        "glow: centre {d:.0} without, {l:.0} with (rgb {:?}); sunlit ground {:.0}",
        &lit[centre..centre + 3],
        luma(&dark, sunny)
    );
    assert!(luma(&dark, sunny) > d + 20.0, "the glowing decal is not in shadow");
    // 25 codes, from 60: Lottes' curve compresses the glow's added light
    // (the test measured 60+ codes under Narkowicz's ACES fit at 0.596
    // ground; now 95 -> 129, a doubling in linear terms).
    assert!(
        l > d + 25.0 && lit[centre] > lit[centre + 2] + 30,
        "no glow: {d} -> {l}, {:?}",
        &lit[centre..centre + 3]
    );
    // Away from the centre the glow is held back.
    let rim = pixel(&east, Vec3::new(glow_at.0 + 5.0, 10.0, glow_at.1)).unwrap();
    assert!(
        luma(&lit, rim) - luma(&dark, rim) < (l - d) * 0.25,
        "glow not held to the centre"
    );
    assert_eq!(
        dark.iter().zip(&ended).filter(|(a, b)| a != b).count(),
        0,
        "a glow past its life still shows"
    );
    let f = luma(&fading_in, centre);
    println!("glow while fading in: centre {f:.0}");
    assert!(f > l - 12.0, "the fade-in dims the glow: {f} against {l}");
    assert_clean(&gpu);
}

/// A decal paints its layer's material, not just its colour (M12): a
/// clustered light at the view's mirror lights a smooth (roughness 0.15)
/// decal much more than a rough (0.95) one with the same albedo, and a
/// metallic decal's F0 (its colour) beats the dielectric's 0.04 at the
/// same roughness. The roughness is the product of a per-layer factor and
/// a per-texel texture (0.5 x 0.3, 1.0 x 0.95, 0.15 x 1.0), so the surface
/// channel and the factors are both exercised. Without the material
/// reaching the BRDF, the frames would be identical (the albedo and the
/// normals are).
#[test]
#[ignore = "needs a Vulkan GPU"]
fn decals_carry_their_roughness_and_metallic() {
    use ae_render::decals::{Decal, DecalAtlas, DecalFeed, DecalsSetup, LayerMaterial, colour_texel, surface_texel};
    use ae_render::lights::Light;
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use glam::{Vec2, Vec3, Vec4};

    let gpu = gpu();
    let n = 129;
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![0.5; n * n],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });

    // Smooth and rough dielectrics, then matching metals authored through
    // the legacy factor and through the material texture.
    let size = 16u32;
    let texels = (size * size) as usize;
    let grey = |c: f32| vec![colour_texel([c, c, c, 1.0]); texels];
    let rough = |r: f32| vec![surface_texel([0.0, 0.0, 1.0], r, 0.0); texels];
    let atlas = Arc::new(DecalAtlas {
        size,
        colour: vec![grey(0.5), grey(0.5), grey(0.5), grey(0.5), grey(0.0), grey(0.0)],
        surface: vec![
            rough(0.15),
            rough(0.95),
            rough(0.15),
            vec![[128, 255, 38, 0]; texels],
            rough(0.15),
            rough(0.95),
        ],
        height_layers: vec![3],
        materials: vec![
            LayerMaterial { metallic: 0.0 },
            LayerMaterial { metallic: 0.0 },
            LayerMaterial { metallic: 1.0 },
            LayerMaterial::default(),
            LayerMaterial::default(),
            LayerMaterial::default(),
        ],
    });

    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let camera = Camera {
        target: Vec2::new(512.0, 512.0),
        ground: 50.0,
        distance: 150.0,
        yaw: 0.0,
        pitch: 0.6,
        fov_y: 0.8,
    };
    let ground = Vec3::new(512.0, 50.0, 512.0);
    let eye = lod_view(&source, &camera, extent.width, extent.height).pos;
    let v = (eye - ground).normalize();
    let mirror = Vec3::new(-v.x, v.y, -v.z);
    let light = Light::point(ground + mirror * 60.0, Vec3::ONE, 1_500_000.0, 100.0, 3.0);

    let render = |layer: u32, camera: Camera, sky_only: bool| -> Vec<u8> {
        let feed = Arc::new(DecalFeed::default());
        feed.place_all(&[Decal::permanent([512.0, 512.0], [12.0, 12.0], 0.0, layer)]);
        // Turn off the sun to isolate the point light and sky.
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_decals(DecalsSetup {
                atlas: Some(atlas.clone()),
                ..DecalsSetup::new(feed, 8)
            })
            .with_tone(ae_render::tonemap::ToneParams {
                sun_illuminance: 0.0,
                ..Default::default()
            });
        if !sky_only {
            app = app.with_lights(vec![light]);
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera = camera;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 8, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let smooth = render(0, camera, false);
    let rough = render(1, camera, false);
    let metal = render(2, camera, false);
    assert!(
        metal == render(3, camera, false),
        "texture metalness should match the legacy metallic factor"
    );

    let clip = lod_view(&source, &camera, extent.width, extent.height).view_proj
        * Vec4::new(ground.x, ground.y, ground.z, 1.0);
    let (cu, cv) = (
        (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
        (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
    );
    assert!(
        cu > 10.0 && cv > 10.0 && cu < extent.width as f32 - 10.0 && cv < extent.height as f32 - 10.0,
        "the decal projects into the view at ({cu}, {cv})"
    );
    let (cu, cv) = (cu as i32, cv as i32);
    let peak = |p: &[u8]| -> f32 {
        let mut best = 0.0f32;
        for y in (cv - 4)..(cv + 4) {
            for x in (cu - 4)..(cu + 4) {
                let i = ((y * extent.width as i32 + x) * 4) as usize;
                best = best.max(0.3 * p[i] as f32 + 0.59 * p[i + 1] as f32 + 0.11 * p[i + 2] as f32);
            }
        }
        best
    };
    println!(
        "decal material peak: smooth dielectric {:.1}, rough dielectric {:.1}, smooth metal {:.1}",
        peak(&smooth),
        peak(&rough),
        peak(&metal)
    );
    assert!(
        peak(&smooth) > peak(&rough) + 20.0,
        "a rough decal keeps the ground glossy: {:.1} vs {:.1}",
        peak(&smooth),
        peak(&rough)
    );
    assert!(
        peak(&metal) > peak(&smooth) + 10.0,
        "the metallic decal's F0 does not reach the BRDF: {:.1} vs {:.1}",
        peak(&metal),
        peak(&smooth)
    );
    // Black dielectrics remove the diffuse term. Without the point light
    // the centre of the decal isolates sky specular at a grazing view.
    let grazing = Camera { pitch: 0.12, ..camera };
    let sample = ((extent.height / 2 * extent.width + extent.width / 2) * 4) as usize;
    let smooth_sky = render(4, grazing, true)[sample];
    let rough_sky = render(5, grazing, true)[sample];
    assert!(
        rough_sky > 0 && rough_sky + 24 < smooth_sky,
        "sky reflection: rough {rough_sky}, smooth {smooth_sky}"
    );
    println!("grazing sky reflection: smooth {smooth_sky}, rough {rough_sky}");
    assert_clean(&gpu);
}

/// The bloom (M12 step 4): a glowing decal's light spreads to the ground
/// around it when the bloom is on, while sunlit ground far from any light
/// source does not change (the glow holds only light sources, not lit
/// surfaces). The bloom is compared on and off with the same scene, and
/// the view's off switch (the `Z` hotkey) draws exactly the tone-mapped
/// zero.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn bloom_spreads_a_glow_but_not_lit_ground() {
    use ae_render::decals::{Decal, DecalAtlas, DecalFeed, DecalsSetup, colour_texel, surface_texel};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::tonemap::ToneParams;
    use glam::{Vec2, Vec4};

    let gpu = gpu();
    let n = 65;
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![0.1; n * n],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 4.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let camera = Camera {
        target: Vec2::new(128.0, 128.0),
        ground: 10.0,
        distance: 120.0,
        yaw: 0.0,
        pitch: 0.7,
        fov_y: 0.8,
    };

    // One grey layer whose normal's alpha is a full glow mask.
    let size = 16u32;
    let texels = (size * size) as usize;
    let atlas = Arc::new(DecalAtlas {
        size,
        colour: vec![vec![colour_texel([0.3, 0.3, 0.3, 1.0]); texels]],
        surface: vec![vec![surface_texel([0.0, 0.0, 1.0], 1.0, 0.9); texels]],
        height_layers: Vec::new(),
        materials: Vec::new(),
    });
    let decal = Decal {
        glow: 1.0,
        glow_life: 100.0,
        ..Decal::permanent([128.0, 128.0], [4.0, 4.0], 0.0, 0)
    };

    let render = |bloom: f32, view_bloom: bool| -> Vec<u8> {
        let feed = Arc::new(DecalFeed::default());
        feed.place_all(&[decal]);
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_decals(DecalsSetup {
                atlas: Some(atlas.clone()),
                ..DecalsSetup::new(feed, 4)
            })
            .with_tone(ToneParams {
                bloom,
                ..Default::default()
            });
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera = camera;
            view.fog = false;
            view.bloom = view_bloom;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 8, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let off = render(0.0, true);
    let on = render(ae_render::tonemap::BLOOM_STRENGTH, true);
    let hotkey_off = render(ae_render::tonemap::BLOOM_STRENGTH, false);
    assert_eq!(hotkey_off, off, "the view's bloom switch does not match strength 0");

    let clip = lod_view(&source, &camera, extent.width, extent.height).view_proj * Vec4::new(128.0, 10.0, 128.0, 1.0);
    let (cu, cv) = (
        (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
        (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
    );
    let (cu, cv) = (cu as i32, cv as i32);
    let luma = |p: &[u8], x: i32, y: i32| -> f32 {
        let i = ((y * extent.width as i32 + x) * 4) as usize;
        0.3 * p[i] as f32 + 0.59 * p[i + 1] as f32 + 0.11 * p[i + 2] as f32
    };
    let mut spread = 0.0f32;
    for y in (cv - 28)..=(cv + 28) {
        for x in (cu - 28)..=(cu + 28) {
            let d = ((x - cu).pow(2) + (y - cv).pow(2)) as f32;
            if ((12.0 * 12.0)..=(28.0 * 28.0)).contains(&d) {
                spread = spread.max(luma(&on, x, y) - luma(&off, x, y));
            }
        }
    }
    let far = luma(&on, cu + 80, cv) - luma(&off, cu + 80, cv);
    println!("bloom: ring spread {spread:.1} luma; sunlit ground 80 px away changes {far:.1}");
    assert!(spread > 3.0, "the glow does not bloom: {spread:.1}");
    assert!(far.abs() < 1.01, "lit ground blooms: {far:.1}");
    assert_clean(&gpu);
}

/// The half-resolution sorted particles keep their glow per particle: a
/// fire in front of smoke still blooms (the smoke behind it must not mask
/// it). The composite adds a separate half-resolution glow target instead
/// of classifying the accumulated colour by its alpha, which lost the
/// fire's emission wherever smoke shared the half texel.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn half_resolution_fire_in_front_of_smoke_blooms() {
    use ae_render::particles::{Burst, ParticleBlend};
    use ae_render::terrain::app::{TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::tonemap::{BLOOM_STRENGTH, ToneParams};
    use glam::{Vec2, Vec3, Vec4};

    let gpu = gpu();
    let (n, scale, ground) = (65usize, 100.0f32, 10.0f32);
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![ground / scale; n * n],
    };
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 4.0, scale, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let camera = Camera {
        target: Vec2::new(128.0, 128.0),
        ground,
        distance: 120.0,
        yaw: 0.0,
        pitch: 0.7,
        fov_y: 0.8,
    };

    // The fire in front of the smoke, and the smoke in front of it (the
    // offset along the view ray flips the sort order). Both hold still:
    // this is about the sorted glow, not the simulation.
    let eye = lod_view(&source, &camera, extent.width, extent.height).pos;
    let fire = Vec3::new(128.0, ground + 3.0, 128.0);
    let back = (fire - eye).normalize();
    let mut fire_effect = still_effect([1.0, 0.6, 0.2, 1.0], 6.0, true);
    fire_effect.emission = [40_000.0; 2];
    let smoke_effect = still_effect([0.09, 0.085, 0.08, 0.8], 16.0, false);
    let effects = vec![fire_effect, smoke_effect];
    let bursts = |toward_eye: f32| -> [Burst; 2] {
        let smoke = Vec3::new(
            fire.x + back.x * 3.0 * toward_eye,
            ground + 3.0,
            fire.z + back.z * 3.0 * toward_eye,
        );
        [
            Burst {
                pos: fire.to_array(),
                ground: true,
                effect: 0,
                count: 1,
                scale: 1.0,
                seed: 1,
            },
            Burst {
                pos: smoke.to_array(),
                ground: true,
                effect: 1,
                count: 1,
                scale: 1.0,
                seed: 2,
            },
        ]
    };
    let behind = bursts(1.0);
    let front = bursts(-1.0);

    let render = |bursts: &[Burst], bloom: f32, downscale: u32| -> Vec<u8> {
        render_particles_lit(
            &gpu,
            &source,
            camera,
            extent,
            Some((bursts, effects.clone(), downscale, ParticleBlend::Sorted)),
            &[],
            ToneParams {
                bloom,
                ..Default::default()
            },
        )
    };
    let ring = |on: &[u8], off: &[u8]| -> f32 {
        let clip =
            lod_view(&source, &camera, extent.width, extent.height).view_proj * Vec4::new(fire.x, fire.y, fire.z, 1.0);
        let (cu, cv) = (
            (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
            (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
        );
        let (cu, cv) = (cu as i32, cv as i32);
        let luma = |p: &[u8], x: i32, y: i32| -> f32 {
            let i = ((y * extent.width as i32 + x) * 4) as usize;
            0.3 * p[i] as f32 + 0.59 * p[i + 1] as f32 + 0.11 * p[i + 2] as f32
        };
        let mut spread = 0.0f32;
        for y in (cv - 30)..=(cv + 30) {
            for x in (cu - 30)..=(cu + 30) {
                let d = ((x - cu).pow(2) + (y - cv).pow(2)) as f32;
                if ((10.0 * 10.0)..=(30.0 * 30.0)).contains(&d) {
                    spread = spread.max(luma(on, x, y) - luma(off, x, y));
                }
            }
        }
        spread
    };
    let half_off = render(&behind, 0.0, 2);
    let half_on = render(&behind, BLOOM_STRENGTH, 2);
    let full_off = render(&behind, 0.0, 1);
    let full_on = render(&behind, BLOOM_STRENGTH, 1);
    let front_off = render(&front, 0.0, 2);
    let front_on = render(&front, BLOOM_STRENGTH, 2);
    let half = ring(&half_on, &half_off);
    let full = ring(&full_on, &full_off);
    let occluded = ring(&front_on, &front_off);
    println!(
        "particle glow: half-resolution spread {half:.1} luma; full-resolution {full:.1}; smoke in front {occluded:.1}"
    );
    assert!(
        half > 2.0,
        "the fire's glow is lost behind the smoke at half resolution: {half:.1}"
    );
    assert!(full > 2.0, "the full-resolution fire's glow is missing: {full:.1}");
    assert!(
        occluded > 0.5,
        "smoke in front removed all of the fire's glow: {occluded:.1}"
    );
    assert!(
        occluded < half * 0.9,
        "smoke in front did not take its share: {occluded:.1} of {half:.1}"
    );
    assert_clean(&gpu);
}

/// The explosion's fire covers the ground: with its light off (emission 0,
/// so only what it covers shows), the ground behind the fireball's core is
/// mostly hidden while it burns. Purely additive fire hid nothing, so
/// wherever SDR did not clip (an HDR display's headroom, or a fading fire)
/// the ground and its scorch mark showed through it.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn explosion_fire_covers_the_ground() {
    use ae_render::particles::{Burst, ParticleBlend, default_effects, effect};
    use ae_render::terrain::app::{TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::tonemap::ToneParams;
    use glam::{Vec2, Vec3, Vec4};

    let gpu = gpu();
    let (n, scale, ground) = (65usize, 100.0f32, 10.0f32);
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![ground / scale; n * n],
    };
    let pyramid = build_pyramid(&heightmap, scale, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 4.0, scale, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let camera = Camera {
        target: Vec2::new(128.0, 128.0),
        ground,
        distance: 80.0,
        yaw: 0.0,
        pitch: 0.9,
        fov_y: 0.8,
    };
    let at = Vec3::new(128.0, 1.0, 128.0);
    // Flash and fire only (the smoke covers by its own opacity), giving no
    // light and lighting nothing: what is left of the ground is what they
    // let through.
    let mut effects = default_effects();
    for e in [effect::FLASH, effect::FIRE] {
        effects[e as usize].emission = [0.0; 2];
        effects[e as usize].light = None;
    }
    let bursts = [effect::FLASH, effect::FIRE].map(|effect| Burst {
        pos: at.to_array(),
        ground: true,
        effect,
        count: if effect == effect::FLASH { 2 } else { 24 },
        scale: 1.0,
        seed: 7,
    });
    let tone = ToneParams {
        bloom: 0.0,
        ..Default::default()
    };
    let clip =
        lod_view(&source, &camera, extent.width, extent.height).view_proj * Vec4::new(at.x, ground + 2.0, at.z, 1.0);
    let (cu, cv) = (
        ((clip.x / clip.w * 0.5 + 0.5) * extent.width as f32) as i32,
        ((clip.y / clip.w * 0.5 + 0.5) * extent.height as f32) as i32,
    );
    let linear = |c: u8| -> f32 {
        let c = c as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let luma = |p: &[u8], x: i32, y: i32| -> f32 {
        let i = ((y * extent.width as i32 + x) * 4) as usize;
        0.2126 * linear(p[i]) + 0.7152 * linear(p[i + 1]) + 0.0722 * linear(p[i + 2])
    };
    // The ground's share left within `r` pixels of the burst's centre.
    let through = |on: &[u8], off: &[u8], r: i32| -> f32 {
        let (mut sum, mut count) = (0.0f32, 0usize);
        for y in (cv - r)..=(cv + r) {
            for x in (cu - r)..=(cu + r) {
                if (x - cu).pow(2) + (y - cv).pow(2) <= r * r {
                    sum += luma(on, x, y) / luma(off, x, y).max(1e-4);
                    count += 1;
                }
            }
        }
        sum / count as f32
    };
    // At 0.6 s the fire has risen off the ground and faded (its opacity
    // falls to 0 over its life): the scorch mark's fade-in takes over.
    for (frames, most) in [(7usize, 0.1f32), (19, 0.2), (37, 0.9)] {
        let bare = render_particles_frames(&gpu, &source, camera, extent, None, &[], tone, frames);
        let shot = render_particles_frames(
            &gpu,
            &source,
            camera,
            extent,
            Some((&bursts, effects.clone(), 2, ParticleBlend::Sorted)),
            &[],
            tone,
            frames,
        );
        let (core, wide) = (through(&shot, &bare, 6), through(&shot, &bare, 20));
        let age = (frames - 1) as f32 / 60.0;
        println!("fire cover at {age:.2} s: ground through the core {core:.3}, within 20 px {wide:.3}");
        assert!(
            core <= most,
            "the fire lets {core:.3} of the ground through its core at {age:.2} s"
        );
    }
    assert_clean(&gpu);
}

/// A box of `color` (alpha in w) from `min` to `max` on joint 0: 24
/// vertices with face normals, 36 indices (appended to `m`, one primitive
/// of material `material`).
fn model_box(m: &mut ae_model::Model, min: [f32; 3], max: [f32; 3], material: u32) {
    use ae_model::{Layout, Primitive, Vertex};
    let first_vertex = m.vertices.len() as u32;
    let first_index = m.indices.len() as u32;
    // Per axis and side: the face's normal and its four corners.
    for axis in 0..3 {
        for side in [0usize, 1] {
            let mut normal = [0.0f32; 3];
            normal[axis] = if side == 1 { 1.0 } else { -1.0 };
            let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
            let base = (m.vertices.len() as u32) - first_vertex;
            for (ua, ub) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
                let mut p = [0.0f32; 3];
                p[axis] = if side == 1 { max[axis] } else { min[axis] };
                p[a] = if ua == 1 { max[a] } else { min[a] };
                p[b] = if ub == 1 { max[b] } else { min[b] };
                m.vertices.push(Vertex {
                    position: p,
                    normal,
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    color: [65535; 4],
                    joints: [0; 4],
                    weights: [65535, 0, 0, 0],
                    ..Default::default()
                });
            }
            m.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    let vertex_count = m.vertices.len() as u32 - first_vertex;
    m.primitives.push(Primitive {
        material,
        layout: Layout::Skinned,
        first_vertex,
        vertex_count,
        first_index,
        index_count: 36,
    });
}

/// Units drawn with models (M11 step 2), validation clean: a piece model
/// built here (a red opaque block at its front, +X; a blue alpha-masked
/// block at its right, +Z; a green block in the middle masked away by its
/// alpha), the Khronos Fox (textured, skinned, at rest) and a dart in one
/// frame. The front and the right land where the heading puts them (the
/// model is neither turned nor mirrored), the masked-away block shows the
/// ground, the Fox shows its texture, and every unit passes the cull. With
/// the units' shadow map the models cast shadows (the masked-away block
/// none).
#[test]
#[ignore = "needs a Vulkan GPU"]
fn units_draw_models() {
    use ae_model::{AlphaMode, Bone, Joint, Material, Model};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Mat4, Vec2, Vec3, Vec4};

    let gpu = gpu();
    let heightmap = Heightmap {
        width: 129,
        height: 129,
        data: vec![0.5; 129 * 129],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });

    let identity = Mat4::IDENTITY.to_cols_array();
    let mut blocks = Model {
        bones: vec![Bone {
            name: "root".into(),
            parent: None,
            rest: identity,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }],
        joints: vec![Joint {
            bone: 0,
            inverse_bind: identity,
        }],
        ..Default::default()
    };
    let material = |name: &str, base_color: [f32; 4], alpha: AlphaMode| Material {
        name: name.into(),
        base_color,
        alpha,
        metallic: 0.0,
        roughness: 0.8,
        ..Default::default()
    };
    blocks.materials = vec![
        material("red", [1.0, 0.02, 0.02, 1.0], AlphaMode::Opaque),
        material("blue", [0.02, 0.02, 1.0, 1.0], AlphaMode::Mask { cutoff: 0.5 }),
        material("green", [0.02, 1.0, 0.02, 0.2], AlphaMode::Mask { cutoff: 0.5 }),
    ];
    model_box(&mut blocks, [0.55, 0.0, -0.3], [1.0, 0.6, 0.3], 0);
    model_box(&mut blocks, [-0.3, 0.0, 0.55], [0.3, 0.6, 1.0], 1);
    model_box(&mut blocks, [-0.4, 0.0, -0.4], [0.4, 1.2, 0.4], 2);
    blocks.bounds = ae_model::Bounds {
        min: [-0.4, 0.0, -0.4],
        max: [1.0, 1.2, 1.0],
    };
    blocks.merge_primitives();
    blocks.validate().unwrap();
    let fox = ae_gltf::convert_file(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/khronos/Fox/Fox.glb"),
    )
    .unwrap()
    .model;

    // (x, y, heading, radius, class): blocks facing world +z, the Fox
    // facing +x, a dart.
    let units = [
        (440.0f32, 512.0f32, (0.0f32, 1.0f32), 40.0f32, 0u32),
        (590.0, 470.0, (1.0, 0.0), 40.0, 1),
        (560.0, 580.0, (1.0, 0.0), 6.0, 2),
        (560.0, 600.0, (0.0, -1.0), 40.0, 0),
    ];
    let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
    for &(x, y, (hx, hy), radius, class) in &units {
        for (a, word) in [
            x.to_bits(),
            y.to_bits(),
            x.to_bits(),
            y.to_bits(),
            yaw(hx, hy)[0].to_bits(),
            yaw(hx, hy)[1].to_bits(),
            yaw(hx, hy)[2].to_bits(),
            yaw(hx, hy)[3].to_bits(),
            yaw(hx, hy)[0].to_bits(),
            yaw(hx, hy)[1].to_bits(),
            yaw(hx, hy)[2].to_bits(),
            yaw(hx, hy)[3].to_bits(),
            radius.to_bits(),
            class,
        ]
        .into_iter()
        .enumerate()
        {
            columns[a].push(word);
        }
    }
    let classes = vec![
        UnitClass {
            color: [0.5; 3],
            gamut: ae_render::colorspace::Gamut::Bt709,
            floats: false,
        },
        UnitClass {
            color: [0.5; 3],
            gamut: ae_render::colorspace::Gamut::Bt709,
            floats: false,
        },
        UnitClass {
            gamut: ae_render::colorspace::Gamut::Bt709,
            color: [1.0, 0.02, 1.0],
            floats: false,
        },
    ];
    let models = vec![Some(Arc::new(blocks)), Some(Arc::new(fox))];
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let mut camera = None;
    // The frame and the (rows, visible) counts, with or without the units'
    // shadow map (the models' caster).
    let mut render = |shadows: bool| -> (Vec<u8>, Option<(u32, u32)>) {
        let feed = Arc::new(UnitFeed::default());
        feed.publish(Arc::new(InstanceStream {
            tick: 1,
            published: std::time::Instant::now(),
            tick_seconds: 0.0,
            sim_seconds: 0.0,
            columns: at_rest(columns.clone()),
            anim: AnimWrites::default(),
        }));
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(UnitsSetup {
            feed,
            capacity: 16,
            classes: classes.clone(),
            min_pixels: 6.0,
            models: models.clone(),
            rigs: Vec::new(),
        });
        if shadows {
            app = app.with_shadows();
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = Vec2::new(512.0, 512.0);
            view.camera.pitch = 1.4;
            view.camera.distance = 700.0;
            view.fog = false;
            view.shadows = shadows;
            camera = Some(view.camera);
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 8, config).unwrap();
        let pixels = app.readback().unwrap();
        if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            std::fs::write(
                std::path::Path::new(&dir).join(format!("models-{}.ppm", shadows as u8)),
                ppm,
            )
            .unwrap();
        }
        let counts = app.last_unit_counts();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        (pixels, counts)
    };
    let (pixels, counts) = render(false);
    let (shaded, shaded_counts) = render(true);
    let camera = camera.unwrap();
    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let index = |p: Vec3| -> usize {
        let clip = view_proj * Vec4::new(p.x, p.y, p.z, 1.0);
        let (u, v) = (
            (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
            (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
        );
        assert!(
            u >= 0.0 && v >= 0.0 && u < extent.width as f32 && v < extent.height as f32,
            "{p} projects into the view"
        );
        (v as usize * extent.width as usize + u as usize) * 4
    };
    let pixel = |p: Vec3| -> [i32; 3] {
        let i = index(p);
        [pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32]
    };
    // Model point (forward, up, right) of a unit at (x, y) with heading h,
    // radius r (the blocks' footprint is 1).
    let at = |x: f32, y: f32, (hx, hy): (f32, f32), r: f32, local: Vec3| -> Vec3 {
        let (forward, side) = (Vec3::new(hx, 0.0, hy), Vec3::new(-hy, 0.0, hx));
        Vec3::new(x, 50.0, y) + (forward * local.x + Vec3::Y * local.y + side * local.z) * r
    };
    for &(x, y, h, r, class) in units.iter().filter(|u| u.4 == 0) {
        let [cr, cg, cb] = pixel(at(x, y, h, r, Vec3::new(0.8, 0.6, 0.0)));
        assert!(
            cr > 2 * cg && cr > 2 * cb,
            "class {class} at ({x}, {y}): red at its front: {cr} {cg} {cb}"
        );
        let [cr, cg, cb] = pixel(at(x, y, h, r, Vec3::new(0.0, 0.6, 0.8)));
        assert!(
            cb > 2 * cr && cb > 2 * cg,
            "class {class} at ({x}, {y}): blue (masked, kept) at its right: {cr} {cg} {cb}"
        );
        let [cr, cg, cb] = pixel(at(x, y, h, r, Vec3::new(0.0, 0.0, 0.0)));
        assert!(
            !(cg > 2 * cr && cg > 2 * cb),
            "class {class} at ({x}, {y}): the green block is masked away: {cr} {cg} {cb}"
        );
        // Mirrored, blue would be at the left.
        let [cr, cg, cb] = pixel(at(x, y, h, r, Vec3::new(0.0, 0.3, -0.8)));
        assert!(
            !(cb > 2 * cr && cb > 2 * cg),
            "class {class} at ({x}, {y}): nothing blue at its left: {cr} {cg} {cb}"
        );
    }
    // The Fox's fur (orange-brown) over its middle.
    let (fx, fy, fh, fr, _) = units[1];
    let [cr, cg, cb] = pixel(at(fx, fy, fh, fr, Vec3::new(0.0, 0.3, 0.0)));
    assert!(cr > cb + 30 && cr > 60, "the Fox's texture: {cr} {cg} {cb}");
    let [cr, cg, cb] = pixel(Vec3::new(560.0, 50.0, 580.0));
    assert!(cr > 2 * cg && cb > 2 * cg, "the dart: {cr} {cg} {cb}");
    assert_eq!(counts, Some((4, 4)), "four rows, all pass the cull");
    assert_eq!(shaded_counts, Some((4, 4)));
    // With the units' shadow map: the ground down-sun of the first block
    // unit's red block (its top centre's shadow) darkens; open ground does
    // not change.
    let sun = ae_render::terrain::sun::direction();
    let (x, y, h, r, _) = units[0];
    let top = at(x, y, h, r, Vec3::new(0.78, 0.6, 0.0));
    let ground = top - sun * ((top.y - 50.0) / sun.y);
    let luma = |px: &[u8], p: Vec3| -> i32 {
        let i = index(p);
        px[i] as i32 + px[i + 1] as i32 + px[i + 2] as i32
    };
    let (lit, dark) = (luma(&pixels, ground), luma(&shaded, ground));
    assert!(
        dark * 10 < lit * 8,
        "the red block's shadow at {ground}: {lit} without the map, {dark} with it"
    );
    // The green block, masked away, casts none (the masked caster tests
    // alpha as the camera does).
    let green = at(x, y, h, r, Vec3::new(0.0, 1.15, 0.0));
    let ground = green - sun * ((green.y - 50.0) / sun.y);
    let (lit, dark) = (luma(&pixels, ground), luma(&shaded, ground));
    assert!(
        (lit - dark).abs() <= 6,
        "no shadow of the masked-away block at {ground}: {lit} without the map, {dark} with it"
    );
    let open = Vec3::new(512.0, 50.0, 400.0);
    assert!(
        (luma(&pixels, open) - luma(&shaded, open)).abs() <= 6,
        "open ground unchanged"
    );
    assert_clean(&gpu);
}

/// The units' shadow map draws a model unit into the cascades the cull
/// finds it in, not every unit into every cascade: a pillar far outside
/// every cascade is counted in none, and a pillar off screen toward the
/// sun (outside the camera's frustum) still shadows the ground in view.
/// Checked to fail when the cascades only take units the camera sees.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn units_model_shadows_culled() {
    use ae_model::{Bone, Joint, Material, Model};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{Headless, TargetInfo};
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Mat4, Vec2, Vec3, Vec4};
    /// What one frame renders: colour bytes, its extent and the cascade lists.
    type Frames = (Vec<u8>, Option<(u32, u32)>, Vec<[u32; 3]>);

    let gpu = gpu();
    let heightmap = Heightmap {
        width: 129,
        height: 129,
        data: vec![0.5; 129 * 129],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });

    // A pillar eight times as tall as its footprint (0.2), which is drawn
    // at the unit's radius: radius 50 makes it 400 m tall.
    let identity = Mat4::IDENTITY.to_cols_array();
    let mut pillar = Model {
        bones: vec![Bone {
            name: "root".into(),
            parent: None,
            rest: identity,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }],
        joints: vec![Joint {
            bone: 0,
            inverse_bind: identity,
        }],
        materials: vec![Material {
            name: "grey".into(),
            base_color: [0.5, 0.5, 0.5, 1.0],
            metallic: 0.0,
            roughness: 0.8,
            ..Default::default()
        }],
        ..Default::default()
    };
    model_box(&mut pillar, [-0.2, 0.0, -0.2], [0.2, 1.6, 0.2], 0);
    pillar.bounds = ae_model::Bounds {
        min: [-0.2, 0.0, -0.2],
        max: [0.2, 1.6, 0.2],
    };
    pillar.validate().unwrap();
    let pillar = Arc::new(pillar);
    let (radius, height) = (50.0f32, 400.0f32);

    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let (target, pitch, distance) = (Vec2::new(512.0, 512.0), 1.4f32, 700.0f32);
    let camera = {
        let app = TerrainApp::new(source.clone(), TerrainOutput::Readback);
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera.target = target;
        view.camera.pitch = pitch;
        view.camera.distance = distance;
        view.camera
    };
    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    // Whether a sphere is inside the camera's side planes.
    let rows = view_proj.transpose();
    let sides = [
        rows.w_axis + rows.x_axis,
        rows.w_axis - rows.x_axis,
        rows.w_axis + rows.y_axis,
        rows.w_axis - rows.y_axis,
    ];
    let touches = |c: Vec3, r: f32| {
        sides
            .iter()
            .all(|p| p.truncate().dot(c) + p.w >= -r * p.truncate().length())
    };
    let in_view = |p: Vec3| -> Option<usize> {
        let clip = view_proj * Vec4::new(p.x, p.y, p.z, 1.0);
        let (u, v) = (
            (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
            (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
        );
        (u >= 8.0 && v >= 8.0 && u < extent.width as f32 - 8.0 && v < extent.height as f32 - 8.0)
            .then(|| (v as usize * extent.width as usize + u as usize) * 4)
    };
    // The pillar's upper part's shadow on the ground (height 50), and the
    // pillar's foot toward the sun from it: the nearest foot whose whole
    // pillar (a generous sphere) is off screen while that shadow is in view.
    let sun = ae_render::terrain::sun::direction();
    let toward_sun = Vec2::new(sun.x, sun.z).normalize();
    let ground_y = 50.0;
    let shadow_of = |foot: Vec2| {
        let top = Vec3::new(foot.x, ground_y + 0.9 * height, foot.y);
        top - sun * ((top.y - ground_y) / sun.y)
    };
    let foot = (0..400)
        .map(|k| target + toward_sun * (5.0 * k as f32))
        .find(|&f| {
            !touches(Vec3::new(f.x, ground_y + 0.5 * height, f.y), 0.6 * height) && in_view(shadow_of(f)).is_some()
        })
        .expect("a foot off screen whose shadow is in view");
    let far = Vec2::new(20_000.0, 512.0);

    // Class 0 the pillar toward the sun, class 1 the far one.
    let units = [(foot, 0u32), (far, 1)];
    let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
    for &(p, class) in &units {
        for (a, word) in [
            p.x.to_bits(),
            p.y.to_bits(),
            p.x.to_bits(),
            p.y.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            radius.to_bits(),
            class,
        ]
        .into_iter()
        .enumerate()
        {
            columns[a].push(word);
        }
    }
    let classes = vec![
        UnitClass {
            color: [0.5; 3],
            gamut: ae_render::colorspace::Gamut::Bt709,
            floats: false
        };
        2
    ];
    let models = vec![Some(pillar.clone()), Some(pillar.clone())];
    let render = |shadows: bool, state: u32| -> Frames {
        let feed = Arc::new(UnitFeed::default());
        let mut columns = at_rest(columns.clone());
        columns[ATTR_VIEW][0] = state;
        feed.publish(Arc::new(InstanceStream {
            tick: 1,
            published: std::time::Instant::now(),
            tick_seconds: 0.0,
            sim_seconds: 0.0,
            columns,
            anim: AnimWrites::default(),
        }));
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(UnitsSetup {
            feed,
            capacity: 16,
            classes: classes.clone(),
            min_pixels: 1.0,
            models: models.clone(),
            rigs: Vec::new(),
        });
        if shadows {
            app = app.with_shadows();
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera = camera;
            view.fog = false;
            view.shadows = shadows;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let info = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        let mut run = Headless::new(&gpu, &mut app, info, config).unwrap();
        run.frames(&mut app, 1).unwrap();
        run.wait().unwrap();
        let cascades = app.last_unit_cascade_counts();
        run.frames(&mut app, 7).unwrap();
        run.finish().unwrap();
        let pixels = app.readback().unwrap();
        if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            std::fs::write(
                std::path::Path::new(&dir).join(format!("model-shadows-{}.ppm", shadows as u8)),
                ppm,
            )
            .unwrap();
        }
        let counts = app.last_unit_counts();
        assert_eq!(
            app.last_unit_cascade_counts(),
            vec![[0; 3]; 2],
            "cached models skip shadow draws"
        );
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        (pixels, counts, cascades)
    };
    let (lit, lit_counts, lit_cascades) = render(false, 0);
    let (shaded, counts, cascades) = render(true, 0);
    let (filtered, _, _) = render(true, 2);
    assert_eq!(filtered, lit, "the off-screen blip casts no model shadow");
    assert_eq!(lit_counts, Some((2, 0)), "both pillars off screen");
    assert_eq!(counts, Some((2, 0)));
    assert_eq!(lit_cascades, vec![[0; 3]; 2], "no shadow map: no cascade lists");
    assert_eq!(cascades[1], [0, 0, 0], "the far pillar is in no cascade");
    assert!(
        cascades[0].contains(&1) && cascades[0].iter().all(|&n| n <= 1),
        "the pillar toward the sun is in a cascade: {:?}",
        cascades[0]
    );
    let i = in_view(shadow_of(foot)).unwrap();
    let luma = |px: &[u8]| px[i] as i32 + px[i + 1] as i32 + px[i + 2] as i32;
    assert!(
        luma(&shaded) * 10 < luma(&lit) * 8,
        "the off-screen pillar's shadow at {}: {} without the map, {} with it",
        shadow_of(foot),
        luma(&lit),
        luma(&shaded)
    );
    assert_clean(&gpu);
}

/// Units with animation records: the pose pass's palettes match the sim's
/// (`ae_model::anim`: `Rig::pose` of each record's channels, lerped
/// between the two at the drawn tick, then `Rig::globals` and
/// `Rig::palette`), drawn three quarters through the records (ticks 3 and 7,
/// drawn at 6) and on the newer; units without records, or whose records
/// are not in the table, on the rest palette. Layers blended, weighted,
/// masked (the sim's rig's mask) and additive, every play mode, a loop
/// across its wrap, manual turns, moves and scales, including snaps at
/// the start of their tick and one pending move,
/// records newer first, rotations on opposite hemispheres. For a model
/// built here (a base and an arm, two clips through the hierarchy), the
/// Fox (skinned) and BoxAnimated (animated pieces); the arm is drawn where
/// its pose puts it; a unit whose rest sphere is off screen but whose clip
/// reaches into it is not culled, nor one with a manual joint whose clips'
/// sphere is off screen but whose sphere covering any turn is not.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn units_pose_models() {
    use ae_model::anim::{Channels, Layer, Manual, Pose, Rig, Trs, kind, manual_flags, manual_kind};
    use ae_model::pose::{PlayMode, rest_palette};
    use ae_model::{Bone, Channel, Clip, Interpolation, Joint, Material, Model, Path, Range};
    use ae_render::models::rows;
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::{TerrainInfo, frustum_planes};
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::units::{
        AnimWrites, Cell, InstanceStream, NO_POSE, POSE_MANUAL, RECORD_CELLS, UnitClass, UnitFeed, UnitsSetup,
    };
    use glam::{Mat4, Quat, Vec2, Vec3, Vec4};

    let gpu = gpu();
    let heightmap = Heightmap {
        width: 129,
        height: 129,
        data: vec![0.5; 129 * 129],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });

    // A red base (bone 1) carrying a blue arm (bone 2). Clip "reach" moves
    // the arm 6 to the right (+z) and turns it a quarter about +y over 1 s;
    // "hop" lifts the base (and so the arm) by 1 over 2 s.
    let identity = Mat4::IDENTITY.to_cols_array();
    let bone = |name: &str, parent: Option<u32>| Bone {
        name: name.into(),
        parent,
        rest: identity,
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
    };
    let mut arm = Model {
        bones: vec![bone("root", None), bone("base", Some(0)), bone("arm", Some(1))],
        joints: vec![
            Joint {
                bone: 1,
                inverse_bind: identity,
            },
            Joint {
                bone: 2,
                inverse_bind: identity,
            },
        ],
        materials: vec![
            Material {
                name: "red".into(),
                base_color: [1.0, 0.02, 0.02, 1.0],
                metallic: 0.0,
                roughness: 0.8,
                ..Default::default()
            },
            Material {
                name: "blue".into(),
                base_color: [0.02, 0.02, 1.0, 1.0],
                metallic: 0.0,
                roughness: 0.8,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    model_box(&mut arm, [-1.0, 0.0, -1.0], [1.0, 0.6, 1.0], 0);
    let first_arm_vertex = arm.vertices.len();
    model_box(&mut arm, [-0.3, 0.6, -0.3], [0.3, 1.4, 0.3], 1);
    for v in &mut arm.vertices[first_arm_vertex..] {
        v.joints = [1, 0, 0, 0];
    }
    arm.bounds = ae_model::Bounds {
        min: [-1.0, 0.0, -1.0],
        max: [1.0, 1.4, 1.0],
    };
    let quarter = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    arm.clip_data = vec![
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0, quarter.x, quarter.y, quarter.z, quarter.w, 0.0,
        2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let channel = |bone, path, times: (u64, u64), values: (u64, u64)| Channel {
        bone,
        path,
        interpolation: Interpolation::Linear,
        times: Range {
            offset: times.0,
            len: times.1,
        },
        values: Range {
            offset: values.0,
            len: values.1,
        },
    };
    arm.clips = vec![
        Clip {
            name: "reach".into(),
            duration: 1.0,
            channels: vec![
                channel(2, Path::Translation, (0, 2), (2, 6)),
                channel(2, Path::Rotation, (0, 2), (8, 8)),
            ],
        },
        Clip {
            name: "hop".into(),
            duration: 2.0,
            channels: vec![channel(1, Path::Translation, (16, 2), (18, 6))],
        },
    ];
    arm.merge_primitives();
    arm.validate().unwrap();
    let khronos = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/khronos");
    let fox = ae_gltf::convert_file(&khronos.join("Fox/Fox.glb")).unwrap().model;
    let boxes = ae_gltf::convert_file(&khronos.join("BoxAnimated/BoxAnimated.glb"))
        .unwrap()
        .model;
    assert!(fox.clips.len() >= 2 && !boxes.clips.is_empty());
    let models = [arm, fox, boxes];

    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let set_view = |app: &TerrainApp| {
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera.target = Vec2::new(512.0, 512.0);
        view.camera.pitch = 1.4;
        view.camera.distance = 700.0;
        view.fog = false;
        view.camera
    };
    let camera = set_view(&TerrainApp::new(source.clone(), TerrainOutput::Readback));
    let lod = lod_view(&source, &camera, extent.width, extent.height);
    let view_proj = lod.view_proj;
    let planes = frustum_planes(view_proj);
    let outside = |c: Vec3, r: f32| planes.iter().any(|p| Vec3::new(p[0], p[1], p[2]).dot(c) + p[3] < -r);
    // Model point (forward, up, right) local to a unit at (x, y) turned by
    // `rot`, radius r (the arm model's footprint is 1).
    let at = |x: f32, y: f32, rot: [f32; 4], r: f32, local: Vec3| -> Vec3 {
        Vec3::new(x, 50.0, y) + Quat::from_array(rot) * (local * r)
    };
    let at_heading =
        |x: f32, y: f32, (hx, hy): (f32, f32), r: f32, local: Vec3| -> Vec3 { at(x, y, yaw(hx, hy), r, local) };

    // What a unit's records hold: none (at rest), records not in the
    // table, or channels at ticks 5 and 7 (stored newer first when
    // `swapped`).
    #[derive(Clone)]
    enum Spec {
        Rest,
        Missing,
        Records(Box<Channels>, Box<Channels>, bool),
    }
    /// A placed unit: position, facing, size, layer count and its channels.
    type Unit = ((f32, f32), (f32, f32), f32, u32, Spec);
    let (looped, once, pingpong) = (PlayMode::Loop, PlayMode::Once, PlayMode::PingPong);
    let layer = |clip: u16, mode: PlayMode, phase: f32| Layer {
        clip,
        mode: mode as u8,
        weight: 1.0,
        target: 1.0,
        phase,
        ..Default::default()
    };
    let with = |layers: &[Layer], manual: &[Manual]| {
        let mut ch = Channels::default();
        ch.layers[..layers.len()].copy_from_slice(layers);
        ch.manual[..manual.len()].copy_from_slice(manual);
        ch
    };
    let one = |clip: u16, mode: PlayMode, ta: f32, tb: f32| {
        Spec::Records(
            Box::new(with(&[layer(clip, mode, ta)], &[])),
            Box::new(with(&[layer(clip, mode, tb)], &[])),
            false,
        )
    };
    let entry = |bone: u16, kind: u8, value: [f32; 4]| Manual {
        bone,
        kind,
        value,
        speed: 1.0,
        ..Default::default()
    };
    let about_y = |deg: f32| Quat::from_rotation_y(deg.to_radians()).to_array();
    let additive = |l: Layer, mask: u16, weight: f32| Layer {
        kind: kind::ADDITIVE,
        mask,
        weight,
        ..l
    };
    let weighted = |l: Layer, weight: f32| Layer { weight, ..l };
    let mut units: Vec<Unit> = vec![
        ((420.0, 440.0), (1.0, 0.0), 20.0, 0, Spec::Rest),
        ((510.0, 440.0), (1.0, 0.0), 20.0, 0, one(1, looped, 0.5, 0.5)),
        ((600.0, 440.0), (1.0, 0.0), 20.0, 0, one(2, looped, 0.6, 1.0)),
        ((420.0, 540.0), (1.0, 0.0), 20.0, 0, one(1, looped, 0.25, 0.75)),
        ((510.0, 540.0), (1.0, 0.0), 20.0, 0, Spec::Missing),
        (
            (610.0, 560.0),
            (1.0, 0.0),
            40.0,
            1,
            Spec::Records(
                Box::new(with(
                    &[layer(2, looped, 0.3), weighted(layer(1, looped, 0.1), 0.5)],
                    &[],
                )),
                Box::new(with(
                    &[layer(2, looped, 0.4), weighted(layer(1, looped, 0.2), 0.5)],
                    &[],
                )),
                true,
            ),
        ),
        ((530.0, 640.0), (1.0, 0.0), 30.0, 2, one(1, looped, 0.1, 0.2)),
        ((620.0, 650.0), (1.0, 0.0), 6.0, 3, one(1, looped, 0.0, 0.0)),
        ((470.0, 390.0), (1.0, 0.0), 20.0, 0, one(2, pingpong, 3.3, 3.3)),
        ((560.0, 390.0), (1.0, 0.0), 20.0, 0, one(1, once, 7.5, 7.5)),
        ((650.0, 390.0), (1.0, 0.0), 30.0, 2, one(1, pingpong, 0.2, 0.3)),
        // A loop across its wrap: the poses lerped.
        ((440.0, 600.0), (1.0, 0.0), 20.0, 0, one(1, looped, 0.9, 0.1)),
        // The base hopping, the arm's reach added on its mask only.
        (
            (560.0, 600.0),
            (1.0, 0.0),
            20.0,
            0,
            Spec::Records(
                Box::new(with(
                    &[layer(2, looped, 0.5), additive(layer(1, looped, 0.4), 1, 0.7)],
                    &[],
                )),
                Box::new(with(
                    &[layer(2, looped, 1.0), additive(layer(1, looped, 0.6), 1, 0.7)],
                    &[],
                )),
                true,
            ),
        ),
        // Manual: the arm turned (out of its clip), the base moved then
        // scaled; a move not shown yet.
        (
            (480.0, 480.0),
            (1.0, 0.0),
            20.0,
            0,
            Spec::Records(
                Box::new(with(
                    &[layer(1, looped, 0.5)],
                    &[
                        entry(2, manual_kind::TURN, about_y(30.0)),
                        entry(2, manual_kind::MOVE, [0.0, 1.0, 0.0, 0.0]),
                        entry(1, manual_kind::MOVE, [0.0, 0.5, 0.0, 0.0]),
                    ],
                )),
                Box::new(with(
                    &[layer(1, looped, 0.5)],
                    &[
                        Manual {
                            speed: 0.0,
                            ..entry(2, manual_kind::TURN, about_y(120.0))
                        },
                        entry(2, manual_kind::MOVE, [0.0, 1.5, 0.0, 0.0]),
                        Manual {
                            speed: 0.0,
                            ..entry(2, manual_kind::SCALE, [1.0, 1.25, 1.0, 0.0])
                        },
                        Manual {
                            speed: 0.0,
                            ..entry(1, manual_kind::MOVE, [0.0, 0.75, 0.0, 0.0])
                        },
                        entry(1, manual_kind::SCALE, [1.5, 1.0, 1.0, 0.0]),
                        Manual {
                            flags: manual_flags::FRESH,
                            ..entry(0, manual_kind::MOVE, [0.0, 9.0, 0.0, 0.0])
                        },
                    ],
                )),
                true,
            ),
        ),
        // A wide turn whose records are on opposite hemispheres.
        (
            (580.0, 480.0),
            (1.0, 0.0),
            20.0,
            0,
            Spec::Records(
                Box::new(with(&[], &[entry(2, manual_kind::SPIN, about_y(10.0))])),
                Box::new(with(&[], &[entry(2, manual_kind::SPIN, about_y(170.0).map(|v| -v))])),
                true,
            ),
        ),
        // Half a blend over the rest, a layer at weight 0, a clip and a
        // mask the model lacks (skipped; the whole model).
        (
            (650.0, 480.0),
            (1.0, 0.0),
            20.0,
            0,
            Spec::Records(
                Box::new(with(
                    &[
                        weighted(layer(1, looped, 0.5), 0.5),
                        weighted(layer(2, looped, 0.5), 0.0),
                        layer(9, looped, 0.1),
                        Layer {
                            mask: 5,
                            ..layer(2, looped, 0.25)
                        },
                    ],
                    &[],
                )),
                Box::new(with(
                    &[
                        weighted(layer(1, looped, 0.7), 0.5),
                        weighted(layer(2, looped, 0.5), 0.0),
                        layer(9, looped, 0.1),
                        Layer {
                            mask: 5,
                            ..layer(2, looped, 0.75)
                        },
                    ],
                    &[],
                )),
                false,
            ),
        ),
    ];
    // Off the right edge, facing +y (its right is world -x): its rest
    // sphere is culled, its arm at 0.99 s of "reach" (5.94 to its right)
    // is in view.
    let (_, rest_sphere) = ae_render::models::fit(&models[0]);
    let heading = (0.0, 1.0);
    let mut x = 512.0;
    while !outside(
        at_heading(
            x,
            512.0,
            heading,
            20.0,
            Vec3::new(rest_sphere[0], rest_sphere[1], rest_sphere[2]),
        ),
        rest_sphere[3] * 20.0,
    ) {
        x += 2.0;
    }
    units.push(((x + 4.0, 512.0), heading, 20.0, 0, one(1, looped, 0.99, 0.99)));
    let reach_tip = at_heading(x + 4.0, 512.0, heading, 20.0, Vec3::new(0.0, 1.39, 5.94));
    assert!(
        !outside(reach_tip, 0.0),
        "the off-screen unit's arm reaches into the view"
    );
    // A unit with a manual joint shown is culled with the sphere covering
    // any turn (models.rs turned_sphere): off the right edge where its
    // clips' sphere is out of view and that one is not, it is drawn.
    let cull_sphere = ae_render::models::fit(&models[0]).1;
    let clip_r = cull_sphere[3].max(
        ae_model::pose::animated_radius(&models[0], Vec3::new(cull_sphere[0], cull_sphere[1], cull_sphere[2])) * 1.05,
    );
    let [turned_r, along] = ae_render::models::turned_sphere(
        [cull_sphere[0], cull_sphere[1], cull_sphere[2], clip_r],
        ae_model::pose::turned_reach(&models[0]),
    );
    let centre = Vec3::new(cull_sphere[0], cull_sphere[1], cull_sphere[2]);
    let mut turned_x = 512.0;
    while !outside(at_heading(turned_x, 512.0, (1.0, 0.0), 20.0, centre), clip_r * 20.0) {
        turned_x += 2.0;
    }
    turned_x += 2.0;
    assert!(
        !outside(
            at_heading(turned_x, 512.0, (1.0, 0.0), 20.0, centre * along),
            turned_r * 20.0
        ),
        "the turned unit's sphere is in view"
    );
    let half_round = with(
        &[layer(1, looped, 0.99)],
        &[entry(1, manual_kind::TURN, about_y(180.0))],
    );
    units.push((
        (turned_x, 512.0),
        (1.0, 0.0),
        20.0,
        0,
        Spec::Records(Box::new(half_round), Box::new(half_round), false),
    ));

    // The sim's rigs; the arm's has a mask (its arm only), which the
    // renderer gets with the rig.
    let mut rigs: Vec<Rig> = models.iter().map(Rig::new).collect();
    let arm_mask = rigs[0].add_mask("arm", vec![0.0, 0.0, 1.0]);
    assert_eq!(arm_mask, 1);
    // A record as the client packs it (ae-client poser.rs, units.rs
    // RECORD_CELLS).
    let record = |ch: &Channels, tick: u32| -> [Cell; RECORD_CELLS] {
        let mut r = [[0; 4]; RECORD_CELLS];
        for (i, l) in ch.layers.iter().enumerate().filter(|(_, l)| l.clip != 0) {
            r[1 + i] = [
                l.clip as u32 | (l.mask as u32) << 16,
                l.mode as u32 | (l.kind as u32) << 8,
                l.weight.to_bits(),
                l.phase.to_bits(),
            ];
        }
        let mut n = 0;
        let mut instant = 0;
        for m in ch
            .manual
            .iter()
            .filter(|m| m.kind != manual_kind::NONE && m.flags & manual_flags::FRESH == 0)
        {
            r[5 + n / 4][n % 4] = m.bone as u32 | (m.kind as u32) << 16;
            r[7 + n] = m.value.map(f32::to_bits);
            if matches!(m.kind, manual_kind::TURN | manual_kind::MOVE | manual_kind::SCALE)
                && m.speed == 0.0
                && m.flags & manual_flags::PENDING == 0
            {
                instant |= 1 << n;
            }
            n += 1;
        }
        r[0] = [tick, n as u32, instant, 0];
        r
    };
    let (older, newer) = (3u32, 7u32);
    let mut table: Vec<Cell> = Vec::new();
    let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
    for ((x, y), (hx, hy), radius, class, spec) in &units {
        let pose = match spec {
            Spec::Records(a, b, swapped) if (*class as usize) < rigs.len() => {
                let first = table.len() as u32;
                let b = record(b, newer);
                let a = record(a, if b[0][2] != 0 { newer - 1 } else { older });
                match swapped {
                    true => table.extend(b.iter().chain(&a)),
                    false => table.extend(a.iter().chain(&b)),
                }
                // As the client marks it: a manual joint shown in either.
                match a[0][1].max(b[0][1]) > 0 {
                    true => first | POSE_MANUAL,
                    false => first,
                }
            }
            Spec::Records(..) | Spec::Missing => 1 << 20,
            Spec::Rest => NO_POSE,
        };
        for (a, word) in [
            x.to_bits(),
            y.to_bits(),
            x.to_bits(),
            y.to_bits(),
            yaw(*hx, *hy)[0].to_bits(),
            yaw(*hx, *hy)[1].to_bits(),
            yaw(*hx, *hy)[2].to_bits(),
            yaw(*hx, *hy)[3].to_bits(),
            yaw(*hx, *hy)[0].to_bits(),
            yaw(*hx, *hy)[1].to_bits(),
            yaw(*hx, *hy)[2].to_bits(),
            yaw(*hx, *hy)[3].to_bits(),
            radius.to_bits(),
            *class,
            pose,
            0, // Surfaced.
            0, // Unfiltered.
        ]
        .into_iter()
        .enumerate()
        {
            columns[a].push(word);
        }
    }
    // Unit 3 is drawn tilted (a roll about its forward axis), so the
    // stream's rotation reaches the model matrix: its turn replaces the
    // heading's identity.
    let tilt = Quat::from_rotation_x(-0.5).to_array();
    for (k, w) in tilt.iter().enumerate() {
        columns[4 + k][3] = w.to_bits();
    }
    let classes = vec![
        UnitClass {
            color: [0.5; 3],
            gamut: ae_render::colorspace::Gamut::Bt709,
            floats: false
        };
        4
    ];
    let setup_models: Vec<_> = models.iter().map(|m| Some(Arc::new(m.clone()))).collect();
    let setup_rigs: Vec<_> = rigs.iter().map(|r| Some(Arc::new(r.clone()))).collect();
    // Frame, counts and poses; `behind`: drawn a tick behind the stream
    // (published in the future: interpolation weight 0), three quarters through
    // the records; else at the stream's tick, on the newer.
    let render = |behind: bool| {
        let feed = Arc::new(UnitFeed::default());
        let (published, tick_seconds) = match behind {
            true => (std::time::Instant::now() + std::time::Duration::from_secs(3600), 0.25),
            false => (std::time::Instant::now(), 0.0),
        };
        // A whole table of stale records (empty, ticks 1 and 2), then the
        // records as writes over it: the renderer gets the feed's table,
        // both streams' writes coalesced.
        let stale: Vec<Cell> = (0..table.len())
            .map(|c| {
                if c % RECORD_CELLS == 0 {
                    [1 + (c / RECORD_CELLS % 2) as u32, 0, 0, 0]
                } else {
                    [0; 4]
                }
            })
            .collect();
        let stream = |anim: AnimWrites| {
            Arc::new(InstanceStream {
                tick: newer as u64,
                published,
                tick_seconds,
                sim_seconds: tick_seconds,
                columns: columns.clone(),
                anim,
            })
        };
        feed.publish(stream(AnimWrites::whole(&stale)));
        let mut writes = AnimWrites::default();
        for (i, c) in table.iter().enumerate().filter(|(i, c)| stale[*i] != **c) {
            writes.push(i as u32, *c);
        }
        feed.publish(stream(writes));
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_units(UnitsSetup {
            feed,
            capacity: 32,
            classes: classes.clone(),
            min_pixels: 6.0,
            models: setup_models.clone(),
            rigs: setup_rigs.clone(),
        });
        set_view(&app);
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 6, config).unwrap();
        let pixels = app.readback().unwrap();
        if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
            let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
            ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
            std::fs::write(
                std::path::Path::new(&dir).join(format!("poses-{}.ppm", behind as u8)),
                ppm,
            )
            .unwrap();
        }
        let counts = app.last_unit_counts();
        let poses = app.last_unit_poses(&gpu).unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        (pixels, counts, poses)
    };
    // The sim's pose of `ch`.
    let pose_of = |rig: &Rig, ch: &Channels| -> Pose {
        let mut p = rig.rest_pose();
        rig.pose(ch, &mut p);
        p
    };
    let lerp3 = |a: [f32; 3], b: [f32; 3], s: f32| Vec3::from(a).lerp(Vec3::from(b), s).to_array();

    for behind in [false, true] {
        let (pixels, counts, poses) = render(behind);
        assert_eq!(
            counts,
            Some((units.len() as u32, units.len() as u32)),
            "every unit drawn (behind: {behind})"
        );
        assert_eq!(poses.len(), units.len());
        // Drawn at tick 6 (three quarters through) or 7 (the newer).
        for (r, (.., class, spec)) in units.iter().enumerate() {
            let Some(model) = models.get(*class as usize) else {
                assert!(poses[r].is_empty(), "row {r}: a dart has no palette");
                continue;
            };
            let rig = &rigs[*class as usize];
            let want: Vec<[f32; 12]> = match spec {
                Spec::Records(a, b, _) => {
                    let (pa, pb) = (pose_of(rig, a), pose_of(rig, b));
                    let s = if !behind {
                        1.0
                    } else if record(b, newer)[0][2] != 0 {
                        0.0
                    } else {
                        0.75
                    };
                    let local = pa
                        .local
                        .iter()
                        .zip(&pb.local)
                        .enumerate()
                        .map(|(bone, (old, new))| {
                            let snaps = |kind| {
                                b.manual.iter().any(|m| {
                                    m.bone as usize == bone && m.kind == kind && m.speed == 0.0 && m.flags == 0
                                })
                            };
                            Trs {
                                t: if snaps(manual_kind::MOVE) {
                                    new.t
                                } else {
                                    lerp3(old.t, new.t, s)
                                },
                                r: if snaps(manual_kind::TURN) {
                                    new.r
                                } else {
                                    Quat::from_array(old.r)
                                        .normalize()
                                        .slerp(Quat::from_array(new.r).normalize(), s)
                                        .to_array()
                                },
                                s: if snaps(manual_kind::SCALE) {
                                    new.s
                                } else {
                                    lerp3(old.s, new.s, s)
                                },
                            }
                        })
                        .collect();
                    let pose = Pose {
                        local,
                        posed: pa.posed.iter().zip(&pb.posed).map(|(a, b)| *a || *b).collect(),
                    };
                    let mut globals = vec![[0.0; 12]; rig.bones()];
                    rig.globals(&pose, &mut globals);
                    let mut palette = vec![[0.0; 12]; rig.joints()];
                    rig.palette(&globals, &mut palette);
                    palette
                }
                _ => rest_palette(model).iter().map(|m| rows(*m)).collect(),
            };
            assert_eq!(poses[r].len(), want.len(), "row {r}: one entry per joint");
            for (j, (got, want)) in poses[r].iter().zip(&want).enumerate() {
                for i in 0..12 {
                    assert!(
                        (got[i] - want[i]).abs() <= 1e-3 * want[i].abs().max(1.0),
                        "row {r} (class {class}, behind {behind}) joint {j}: {got:?} vs {want:?}"
                    );
                }
            }
        }
        if behind {
            continue;
        }
        let index = |p: Vec3| -> usize {
            let clip = view_proj * Vec4::new(p.x, p.y, p.z, 1.0);
            let (u, v) = (
                (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
                (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
            );
            assert!(
                u >= 0.0 && v >= 0.0 && u < extent.width as f32 && v < extent.height as f32,
                "{p} projects into the view"
            );
            (v as usize * extent.width as usize + u as usize) * 4
        };
        let blue = |p: Vec3| {
            let i = index(p);
            let [r, g, b] = [pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32];
            b > 2 * r && b > 2 * g
        };
        let unit = |k: usize| (units[k].0.0, units[k].0.1, units[k].1, units[k].2);
        let (x, y, h, r) = unit(0);
        assert!(
            blue(at_heading(x, y, h, r, Vec3::new(0.0, 1.39, 0.0))),
            "at rest, the arm on the base"
        );
        let (x, y, h, r) = unit(1);
        assert!(
            blue(at_heading(x, y, h, r, Vec3::new(0.0, 1.39, 3.0))),
            "half way through the reach, the arm 3 to the right"
        );
        assert!(
            !blue(at_heading(x, y, h, r, Vec3::new(0.0, 1.39, 0.0))),
            "and not on the base"
        );
        let (x, y, h, r) = unit(4);
        assert!(
            blue(at_heading(x, y, h, r, Vec3::new(0.0, 1.39, 0.0))),
            "records not in the table: at rest"
        );
        assert!(blue(reach_tip), "the off-screen unit's arm is drawn");
        let (x, y, h, r) = unit(9);
        assert!(
            blue(at_heading(x, y, h, r, Vec3::new(0.0, 1.39, 5.9))),
            "played once, held at full reach (looping it would be half way)"
        );
        // Unit 3 is tilted: its arm's bone, placed as `Anim::bone_world`
        // places it (the ground point, the rotation, then the fit), where
        // the GPU draws it. Its bone is 2, its middle one unit above its
        // origin.
        let (x, y, _h, r) = unit(3);
        let ch = match &units[3].4 {
            Spec::Records(_, newer, _) => newer,
            _ => unreachable!("unit 3 has records"),
        };
        // The size it is drawn at (the minimum pixel size included).
        let ground = Vec3::new(x, 50.0, y);
        let pixels_ = 2.0 * r * lod.proj_scale / ground.distance(lod.pos).max(1e-3);
        let size = r * (6.0 / pixels_.max(1e-6)).max(1.0);
        let (p, q) = rigs[0].bone_frame(ch, 2);
        let facing = Quat::from_array(tilt);
        let (bone, rot) = (ground + facing * (Vec3::from(p) * size), facing * Quat::from_array(q));
        let middle = bone + rot * (Vec3::Y * size);
        assert!(blue(middle), "the tilted unit's arm at {middle}");
        let untilted = ground + (Vec3::from(p) + Vec3::Y) * size;
        assert!(
            (middle - untilted).length() > 5.0 && !blue(untilted),
            "the tilt moves the arm: {middle} vs {untilted}"
        );
    }
    assert_clean(&gpu);
}

/// Speculars (M12 step 3): glTF metallic-roughness on a model's flat top
/// at the sun's mirror point. A smooth metal (F0 = base colour) has a wide
/// white highlight, a rough metal (roughness 0.9) none, and a dielectric
/// (F0 0.04) a small one: the highlight's area follows F0 and roughness.
/// With the sun off, a clustered light at the same mirror adds its own
/// highlight to the smooth metal, and the same light at the opposite
/// azimuth (same distance and elevation, so the same diffuse) adds none;
/// the rough metal and the dielectric get no highlight either.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn speculars_follow_material_and_lights() {
    use ae_model::{Bone, Joint, Material, Model};
    use ae_render::lights::Light;
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::tonemap::ToneParams;
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Mat4, Vec2, Vec3, Vec4};

    let gpu = gpu();
    let n = 129;
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![0.5; n * n],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });

    let identity = Mat4::IDENTITY.to_cols_array();
    let model = |metallic: f32, roughness: f32| -> Arc<Model> {
        let mut m = Model {
            bones: vec![Bone {
                name: "root".into(),
                parent: None,
                rest: identity,
                translation: [0.0; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            }],
            joints: vec![Joint {
                bone: 0,
                inverse_bind: identity,
            }],
            materials: vec![Material {
                name: "m".into(),
                base_color: [0.8, 0.8, 0.8, 1.0],
                metallic,
                roughness,
                ..Default::default()
            }],
            ..Default::default()
        };
        model_box(&mut m, [-1.0, 0.0, -1.0], [1.0, 1.0, 1.0], 0);
        m.bounds = ae_model::Bounds {
            min: [-1.0, 0.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };
        m.merge_primitives();
        m.validate().unwrap();
        Arc::new(m)
    };

    // Ground 50, box fitted to radius 40 (its footprint is 1): its top is
    // at 90. The camera looks from the sun's mirror direction (42 degrees
    // up, the sun's elevation, on the opposite side), so the mirror point
    // is the top's centre.
    let top = Vec3::new(512.0, 90.0, 512.0);
    let camera = Camera {
        target: Vec2::new(512.0, 512.0),
        ground: 0.0,
        distance: 400.0,
        yaw: 2.183,
        pitch: 0.835,
        fov_y: 0.8,
    };
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let eye = lod_view(&source, &camera, extent.width, extent.height).pos;
    let v = (eye - top).normalize();
    let mirror = Vec3::new(-v.x, v.y, -v.z); // reflect(-v, up)
    let opposite = Vec3::new(-mirror.x, mirror.y, -mirror.z);
    let light = |dir: Vec3| Light::point(top + dir * 60.0, Vec3::ONE, 500_000.0, 120.0, 2.0);

    let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
    for (a, word) in [
        512.0f32.to_bits(),
        512.0f32.to_bits(),
        512.0f32.to_bits(),
        512.0f32.to_bits(),
        0.0f32.to_bits(),
        0.0f32.to_bits(),
        0.0f32.to_bits(),
        1.0f32.to_bits(),
        0.0f32.to_bits(),
        0.0f32.to_bits(),
        0.0f32.to_bits(),
        1.0f32.to_bits(),
        40.0f32.to_bits(),
        0u32,
    ]
    .into_iter()
    .enumerate()
    {
        columns[a].push(word);
    }
    let render = |model: Arc<Model>, light: Option<Light>, sun: f32| -> Vec<u8> {
        let feed = Arc::new(UnitFeed::default());
        feed.publish(Arc::new(InstanceStream {
            tick: 1,
            published: std::time::Instant::now(),
            tick_seconds: 0.0,
            sim_seconds: 0.0,
            columns: at_rest(columns.clone()),
            anim: AnimWrites::default(),
        }));
        let setup = UnitsSetup {
            feed,
            capacity: 4,
            classes: vec![UnitClass {
                color: [0.5; 3],
                gamut: ae_render::colorspace::Gamut::Bt709,
                floats: false,
            }],
            min_pixels: 8.0,
            models: vec![Some(model)],
            rigs: Vec::new(),
        };
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_units(setup)
            .with_tone(ToneParams {
                sun_illuminance: sun,
                ..Default::default()
            });
        if let Some(l) = light {
            app = app.with_lights(vec![l]);
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera = camera;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 8, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };

    let clip = view_proj * Vec4::new(top.x, top.y, top.z, 1.0);
    let (cu, cv) = (
        (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
        (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
    );
    assert!(
        cu > 30.0 && cv > 30.0 && cu < extent.width as f32 - 30.0 && cv < extent.height as f32 - 30.0,
        "the box top projects into the view at ({cu}, {cv})"
    );
    let (cu, cv) = (cu as i32, cv as i32);
    let luma = |p: &[u8], x: i32, y: i32| -> f32 {
        let i = ((y * extent.width as i32 + x) * 4) as usize;
        0.3 * p[i] as f32 + 0.59 * p[i + 1] as f32 + 0.11 * p[i + 2] as f32
    };
    let peak = |p: &[u8]| -> f32 {
        let mut best = 0.0f32;
        for y in (cv - 10)..(cv + 10) {
            for x in (cu - 10)..(cu + 10) {
                best = best.max(luma(p, x, y));
            }
        }
        best
    };

    // The sun's specular alone, at a lowered sun so the highlight does not
    // clip (the sky's ambient does not depend on it).
    let sun = 1_000.0;
    let smooth_metal = render(model(1.0, 0.15), None, sun);
    let rough_metal = render(model(1.0, 0.90), None, sun);
    let dielectric = render(model(0.0, 0.15), None, sun);
    let at = |p: &[u8], dx: i32, dy: i32| -> f32 { luma(p, cu + dx, cv + dy) };
    let (m, r, d) = (at(&smooth_metal, 0, 0), at(&rough_metal, 0, 0), at(&dielectric, 0, 0));
    let (m2, r2, d2) = (
        at(&smooth_metal, 0, 24),
        at(&rough_metal, 0, 24),
        at(&dielectric, 0, 24),
    );
    println!(
        "sun specular luma at the mirror: metal {m:.0}, rough {r:.0}, dielectric {d:.0}; 24 px up: {m2:.0}, {r2:.0}, {d2:.0}"
    );
    assert!(m > r + 30.0, "a metal's F0 beats a rough metal's: {m:.1} vs {r:.1}");
    assert!(m > d + 30.0, "and a dielectric's F0 0.04 (20x lower): {m:.1} vs {d:.1}");
    assert!(
        m2 > r2 + 10.0 && m2 > d2 + 10.0,
        "the highlight reaches past its centre: {m2:.1} vs {r2:.1}, {d2:.1}"
    );

    // With the sun off, the ambient sky keeps all three visible; a light at
    // the view's mirror adds a highlight, and the same light at the
    // opposite azimuth (same distance and elevation, so the same diffuse)
    // does not. Only the smooth metal has the F0 for it.
    let on = render(model(1.0, 0.15), Some(light(mirror)), 0.0);
    let off = render(model(1.0, 0.15), Some(light(opposite)), 0.0);
    let rough_on = render(model(1.0, 0.90), Some(light(mirror)), 0.0);
    let diel_on = render(model(0.0, 0.15), Some(light(mirror)), 0.0);
    println!(
        "clustered light peak: metal {:.0}, off {:.0}, rough {:.0}, dielectric {:.0}",
        peak(&on),
        peak(&off),
        peak(&rough_on),
        peak(&diel_on)
    );
    assert!(
        peak(&on) > peak(&off) + 60.0,
        "the mirror light's highlight: {:.1} vs {:.1}",
        peak(&on),
        peak(&off)
    );
    assert!(
        peak(&rough_on) < peak(&on) - 40.0,
        "a rough metal has no highlight: {:.1}",
        peak(&rough_on)
    );
    assert!(
        peak(&diel_on) < peak(&on) - 40.0,
        "a dielectric's F0 is 20x lower: {:.1}",
        peak(&diel_on)
    );
    assert_clean(&gpu);
}

/// A clustered light's glint on the water (M12 step 3): on a still, deep
/// map the surface is the sky's reflection plus the light's GGX lobe. A
/// point light at the view's mirror brightens the water's glint region; the
/// same light at the opposite azimuth (same distance and elevation) leaves
/// it at the sky's reflection, and so does no light at all.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn lights_glint_on_water() {
    use ae_render::lights::Light;
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use glam::{Vec2, Vec3, Vec4};

    let gpu = gpu();
    let n = 129;
    // All water: 40 m under a surface at 60.
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![0.2; n * n],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, Some(60.0));
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });

    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let camera = Camera {
        target: Vec2::new(512.0, 512.0),
        ground: 0.0,
        distance: 700.0,
        yaw: 0.0,
        pitch: 0.4,
        fov_y: 0.8,
    };
    let eye = lod_view(&source, &camera, extent.width, extent.height).pos;
    let water = Vec3::new(512.0, 60.0, 512.0);
    let v = (eye - water).normalize();
    let mirror = Vec3::new(-v.x, v.y, -v.z); // reflect(-v, up)
    let light = |dir: Vec3| Light::point(water + dir * 60.0, Vec3::ONE, 2_000_000_000.0, 100.0, 3.0);

    let render = |light: Option<Light>| -> Vec<u8> {
        // The sun off: the sky's reflection and the light are all there is.
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_water()
            .with_tone(ae_render::tonemap::ToneParams {
                sun_illuminance: 0.0,
                ..Default::default()
            });
        if let Some(l) = light {
            app = app.with_lights(vec![l]);
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera = camera;
            view.fog = false;
            view.water = true;
            view.time = Some(1.0);
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 8, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let on = render(Some(light(mirror)));
    // The waves put a glint wherever a normal meets the light's half
    // vector, so an opposite-azimuth light also glints somewhere: the
    // mirror geometry is pinned by the models' flat top (the test above).
    let none = render(None);

    let clip =
        lod_view(&source, &camera, extent.width, extent.height).view_proj * Vec4::new(water.x, water.y, water.z, 1.0);
    let (cu, cv) = (
        (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
        (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
    );
    assert!(
        cu > 70.0 && cv > 70.0 && cu < extent.width as f32 - 70.0 && cv < extent.height as f32 - 70.0,
        "the water point projects into the view at ({cu}, {cv})"
    );
    let (cu, cv) = (cu as i32, cv as i32);
    // The waves tilt the normal, so the glint can sit tens of pixels from
    // the flat mirror point.
    let peak = |p: &[u8]| -> f32 {
        let mut best = 0.0f32;
        for y in (cv - 70)..(cv + 70) {
            for x in (cu - 70)..(cu + 70) {
                let i = ((y * extent.width as i32 + x) * 4) as usize;
                best = best.max(0.3 * p[i] as f32 + 0.59 * p[i + 1] as f32 + 0.11 * p[i + 2] as f32);
            }
        }
        best
    };
    println!("water glint peak: light {:.1}, none {:.1}", peak(&on), peak(&none));
    assert!(
        peak(&on) > peak(&none) + 40.0,
        "the light's glint: {:.1} vs {:.1}",
        peak(&on),
        peak(&none)
    );
    assert!(peak(&on) > 230.0, "a bright glint: {:.1}", peak(&on));
    assert_clean(&gpu);
}

/// A clustered light's specular on the terrain (M12 step 3): a grazing
/// view raises the dielectric's Fresnel, so a light at the view's mirror
/// brightens the ground's specular and the same light at the opposite
/// azimuth (same distance and elevation, so the same diffuse) does not.
/// The sun is off, so the only difference is the specular.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn lights_specular_on_the_ground() {
    use ae_render::lights::Light;
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use glam::{Vec2, Vec3, Vec4};

    let gpu = gpu();
    let n = 129;
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![0.8; n * n],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });

    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let camera = Camera {
        target: Vec2::new(512.0, 512.0),
        ground: 0.0,
        distance: 200.0,
        yaw: 0.0,
        pitch: 0.15,
        fov_y: 0.8,
    };
    let eye = lod_view(&source, &camera, extent.width, extent.height).pos;
    let ground = Vec3::new(512.0, 80.0, 512.0);
    let v = (eye - ground).normalize();
    let mirror = Vec3::new(-v.x, v.y, -v.z);

    let render = |dir: Vec3| -> Vec<u8> {
        // Bright enough to light the ground well below white with no tone
        // curve (the diffuse alone at 2e9 filled the range), so the
        // specular has room above it.
        let l = Light::point(ground + dir * 60.0, Vec3::ONE, 125_000_000.0, 100.0, 3.0);
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_tone(ae_render::tonemap::ToneParams {
                sun_illuminance: 0.0,
                ..Default::default()
            })
            .with_lights(vec![l]);
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera = camera;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 8, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let on = render(mirror);
    let off = render(Vec3::new(-mirror.x, mirror.y, -mirror.z));

    let clip = lod_view(&source, &camera, extent.width, extent.height).view_proj
        * Vec4::new(ground.x, ground.y, ground.z, 1.0);
    let (cu, cv) = (
        (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32,
        (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32,
    );
    assert!(
        cu > 10.0 && cv > 10.0 && cu < extent.width as f32 - 10.0 && cv < extent.height as f32 - 10.0,
        "the ground point projects into the view at ({cu}, {cv})"
    );
    let luma = |p: &[u8]| -> f32 {
        let i = ((cv as i32 * extent.width as i32 + cu as i32) * 4) as usize;
        0.3 * p[i] as f32 + 0.59 * p[i + 1] as f32 + 0.11 * p[i + 2] as f32
    };
    println!("ground specular: mirror {:.1}, opposite {:.1}", luma(&on), luma(&off));
    assert!(
        luma(&on) > luma(&off) + 10.0,
        "the mirror light's ground specular: {:.1} vs {:.1}",
        luma(&on),
        luma(&off)
    );
    assert_clean(&gpu);
}

/// The clustered lights (M12): the frames lit through the tiles and
/// clusters are exactly the frames lit by every light at every pixel and
/// particle (a light the culls leave out of a cluster adds exactly 0
/// there), from above and from low over the ground (where the slices
/// matter), with points, cones and beams; and the lights do light the
/// terrain, the units and the smoke.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn lights_clusters_match_the_unculled_reference() {
    use ae_render::lights::{Light, scatter};
    use ae_render::particles::{Burst, ParticleBlend, ParticleFeed, ParticlesSetup};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{build_pyramid, generate};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Vec2, Vec3};

    let gpu = gpu();
    let heightmap = generate(257, 257, 11);
    let pyramid = build_pyramid(&heightmap, 200.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 4.0, 200.0, 0.0, Some(40.0));
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let centre = Vec2::new(512.0, 512.0);
    let height = |p: Vec2| source.height_at(p);
    let mut lights = scatter(600, centre, 450.0, 3.0, 5, &height);
    for k in 0..40 {
        let a = k as f32 * 0.7;
        let p = centre + Vec2::new(a.cos(), a.sin()) * (60.0 + 8.0 * k as f32);
        let base = Vec3::new(p.x, height(p) + 25.0, p.y);
        // The old arbitrary-unit colour as candela at the EV14 reference:
        // pi / 5.086e-5.
        let scale = 61_760.0;
        let cone_color = Vec3::new(0.4, 0.8, 2.0);
        let beam_color = Vec3::new(2.0, 0.2, 0.2);
        lights.push(Light::cone(
            base,
            Vec3::new(0.3 * a.sin(), -1.0, 0.3 * a.cos()),
            cone_color,
            ae_render::lighting::luminance(cone_color.to_array()) * scale,
            60.0,
            12.0,
            0.25,
            0.45,
        ));
        lights.push(Light::beam(
            base - Vec3::X * 30.0,
            base + Vec3::new(30.0, -20.0, 10.0),
            beam_color,
            ae_render::lighting::luminance(beam_color.to_array()) * scale,
            6.0,
            2.0,
        ));
    }
    // Darts in a ring, smoke over the middle.
    let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
    for k in 0..48 {
        let a = k as f32 * std::f32::consts::TAU / 48.0;
        let (x, y) = (centre.x + 150.0 * a.cos(), centre.y + 150.0 * a.sin());
        for (attr, word) in [
            x.to_bits(),
            y.to_bits(),
            x.to_bits(),
            y.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            5.0f32.to_bits(),
            0,
        ]
        .into_iter()
        .enumerate()
        {
            columns[attr].push(word);
        }
    }
    let render = |lights: Vec<Light>, unculled: bool, pitch: f32| -> Vec<u8> {
        let listed = lights.len();
        let units = Arc::new(UnitFeed::default());
        units.publish(Arc::new(InstanceStream {
            tick: 1,
            published: std::time::Instant::now(),
            tick_seconds: 0.0,
            sim_seconds: 0.0,
            columns: at_rest(columns.clone()),
            anim: AnimWrites::default(),
        }));
        let setup = UnitsSetup {
            feed: units,
            capacity: 64,
            classes: vec![UnitClass {
                gamut: ae_render::colorspace::Gamut::Bt709,
                color: [0.7, 0.7, 0.7],
                floats: false,
            }],
            min_pixels: 8.0,
            models: Vec::new(),
            rigs: Vec::new(),
        };
        let particles = Arc::new(ParticleFeed::default());
        let effects = vec![still_effect([0.4, 0.4, 0.4, 0.6], 30.0, false)];
        for k in 0..12 {
            particles.emit(Burst {
                pos: [centre.x + 20.0 * k as f32 - 120.0, 30.0, centre.y],
                ground: true,
                effect: 0,
                count: 1,
                scale: 1.0,
                seed: k,
            });
        }
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_water()
            .with_units(setup)
            .with_particles(ParticlesSetup {
                feed: particles,
                capacity: 64,
                effects,
                downscale: 2,
                blend: ParticleBlend::Sorted,
            })
            .with_lights(lights);
        if unculled {
            app = app.with_unculled_lights();
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = centre;
            view.camera.pitch = pitch;
            view.camera.distance = 700.0;
        }
        let extent = vk::Extent2D {
            width: 320,
            height: 200,
        };
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 8, config).unwrap();
        let pixels = app.readback().unwrap();
        let stats = *app.light_stats().lock().unwrap();
        if !unculled && listed > 0 {
            // The culls listed some lights in the clusters (far fewer than
            // every light in every cluster) and cut none.
            assert!(
                stats.indices > 0 && (stats.indices as usize) < listed * 1000,
                "{stats:?}"
            );
            assert_eq!((stats.overflow, stats.dropped), (0, 0), "no list cut: {stats:?}");
        }
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    for pitch in [1.1, 0.3] {
        let dark = render(Vec::new(), false, pitch);
        // The reference first, so it cannot reuse the clustered run's
        // memory.
        let reference = render(lights.clone(), true, pitch);
        let lit = render(lights.clone(), false, pitch);
        let differ = lit.chunks(4).zip(reference.chunks(4)).filter(|(a, b)| a != b).count();
        assert_eq!(differ, 0, "pitch {pitch}: the clustered frame equals the unculled one");
        let brighter = |frame: &[u8]| {
            frame
                .chunks(4)
                .zip(dark.chunks(4))
                .filter(|(a, b)| {
                    a[..3]
                        .iter()
                        .zip(&b[..3])
                        .map(|(x, y)| *x as i32 - *y as i32)
                        .sum::<i32>()
                        > 12
                })
                .count()
        };
        let total = lit.len() / 4;
        for (name, frame) in [("reference", &reference), ("clustered", &lit)] {
            let n = brighter(frame);
            assert!(
                n > total / 50,
                "pitch {pitch}: the {name} lights light the ground ({n} of {total} pixels brighter)"
            );
        }
    }
    assert_clean(&gpu);
}

/// The bundled scars' normal maps relight the ground. One real scar
/// (`assets/decals/scars/scar1`, loaded as the client loads it: `read_tga`,
/// `add_material_layer`, Recoil's modulate blend) on flat ground
/// in the sun, against the same atlas with that layer's height made constant
/// (roughness and glow kept), close up and at a gameplay distance. Inside
/// the scar's footprint the two frames must differ: a normal map lost on
/// the way (channels, mips, the tangent frame) leaves them equal.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn scar_height_relief_relights_the_ground() {
    use ae_render::decals::{
        Blend, Decal, DecalDebug, DecalFeed, DecalsSetup, Relief, SCAR_HEIGHT_RATIO, default_atlas, read_tga,
    };
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use glam::{Vec2, Vec3, Vec4};

    let gpu = gpu();
    let n = 129;
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![0.5; n * n],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/decals/scars");
    let read = |name: &str| read_tga(&std::fs::read(dir.join(name)).unwrap()).unwrap();
    let (colour, material) = (read("scar1.tga"), read("scar1_mat.tga"));
    let mut atlas = default_atlas(512);
    let layer = atlas.add_material_layer(&colour, &material).unwrap();
    let mut flat = atlas.clone();
    for t in &mut flat.surface[layer as usize] {
        t[0] = 128;
    }
    let (atlas, flat) = (Arc::new(atlas), Arc::new(flat));

    let extent = vk::Extent2D {
        width: 640,
        height: 480,
    };
    let centre = [512.0f32, 512.0];
    let render = |atlas: &Arc<ae_render::decals::DecalAtlas>, camera: Camera, decal: bool, debug| -> Vec<u8> {
        let feed = Arc::new(DecalFeed::default());
        if decal {
            feed.place_all(&[Decal {
                blend: Blend::Modulate,
                relief: Relief::Proportional(SCAR_HEIGHT_RATIO),
                ..Decal::permanent(centre, [16.0, 16.0], 0.6, layer)
            }]);
        }
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_decals(DecalsSetup {
            atlas: Some(atlas.clone()),
            debug,
            ..DecalsSetup::new(feed, 8)
        });
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera = camera;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 8, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let luma = |p: &[u8], i: usize| p[i] as f32 * 0.3 + p[i + 1] as f32 * 0.59 + p[i + 2] as f32 * 0.11;
    // The normals' relight is measured in 8-bit luma: the tone curve's
    // compressed midtones shrink it (3.0 was under ACES at 0.596 ground,
    // 2.93 now), so "close" asks for 2.5.
    for (name, distance, min_mean) in [("close", 50.0f32, 2.5f32), ("gameplay", 250.0, 1.0)] {
        let camera = Camera {
            target: Vec2::from(centre),
            ground: 50.0,
            distance,
            yaw: 0.4,
            pitch: 0.9,
            fov_y: 0.8,
        };
        let none = render(&atlas, camera, false, DecalDebug::Off);
        let mapped = render(&atlas, camera, true, DecalDebug::Off);
        let flattened = render(&flat, camera, true, DecalDebug::Off);
        if let Ok(dir) = std::env::var("AE_TEST_CAPTURE_DIR") {
            for (tag, pixels) in [("mapped", &mapped), ("flat", &flattened)] {
                let mut ppm = format!("P6\n{} {}\n255\n", extent.width, extent.height).into_bytes();
                ppm.extend(pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
                std::fs::write(
                    std::path::Path::new(&dir).join(format!("scar-relief-{name}-{tag}.ppm")),
                    ppm,
                )
                .unwrap();
            }
        }
        // The footprint: where the scar (either version) changed the ground.
        let footprint: Vec<usize> = (0..none.len() / 4)
            .map(|i| i * 4)
            .filter(|&i| {
                (luma(&none, i) - luma(&mapped, i)).abs() > 1.0 || (luma(&none, i) - luma(&flattened, i)).abs() > 1.0
            })
            .collect();
        let deltas: Vec<f32> = footprint
            .iter()
            .map(|&i| (luma(&mapped, i) - luma(&flattened, i)).abs())
            .collect();
        let mean = deltas.iter().sum::<f32>() / deltas.len().max(1) as f32;
        let over2 = deltas.iter().filter(|&&d| d > 2.0).count() as f32 / deltas.len().max(1) as f32;
        let scar = footprint.iter().map(|&i| luma(&flattened, i)).sum::<f32>() / footprint.len().max(1) as f32;
        println!(
            "scar relief {name}: footprint {} px, mean |dluma| {mean:.2}, {:.0} % over 2, scar luma {scar:.0}",
            footprint.len(),
            over2 * 100.0
        );
        assert!(footprint.len() > 200, "{name}: scar out of view");
        assert!(
            mean > min_mean,
            "{name}: the normal map barely changes the frame (mean |dluma| {mean})"
        );
    }
    // Compare matched world positions on the real, asymmetric heightmap.
    // Read back the normal to isolate the decal basis from view-dependent lighting.
    let camera = Camera {
        target: Vec2::from(centre),
        ground: 50.0,
        distance: 25.0,
        yaw: 0.4,
        pitch: 0.9,
        fov_y: 0.8,
    };
    let normals = |camera| {
        let pixels = render(&atlas, camera, true, DecalDebug::Normal);
        let vp = lod_view(&source, &camera, extent.width, extent.height).view_proj;
        (0..81).map(move |i| {
            let x = centre[0] + (i % 9) as f32 - 4.0;
            let z = centre[1] + (i / 9) as f32 - 4.0;
            let clip = vp * Vec4::new(x, 50.0, z, 1.0);
            let u = (clip.x / clip.w * 0.5 + 0.5) * extent.width as f32 - 0.5;
            let v = (clip.y / clip.w * 0.5 + 0.5) * extent.height as f32 - 0.5;
            let (x, y) = (u.floor() as usize, v.floor() as usize);
            let at = |x, y| {
                let px = (y * extent.width as usize + x) * 4;
                Vec3::from_array(std::array::from_fn(|k| {
                    let s = f32::from(pixels[px + k]) / 255.0;
                    let linear = if s <= 0.04045 {
                        s / 12.92
                    } else {
                        ((s + 0.055) / 1.055).powf(2.4)
                    };
                    linear * 2.0 - 1.0
                }))
            };
            let row = |y| at(x, y).lerp(at(x + 1, y), u.fract());
            row(y).lerp(row(y + 1), v.fract()).normalize()
        })
    };
    let original: Vec<_> = normals(camera).collect();
    let opposite = Camera {
        yaw: camera.yaw + std::f32::consts::PI,
        ..camera
    };
    let mean_difference = original
        .iter()
        .zip(normals(opposite))
        .map(|(a, b)| a.angle_between(b))
        .sum::<f32>()
        / original.len() as f32;
    println!(
        "opposite views: mean normal angle {:.2} degrees",
        mean_difference.to_degrees()
    );
    // The projected pixels can land on different parts of a texel.
    assert!(mean_difference.to_degrees() < 6.0);

    // A real crater in the terrain mesh receives a decal while the same
    // renderer and instance switch from normal mapping to cone tracing.
    use ae_render::decals::DecalReliefMode;
    use ae_render::thread::Headless;
    let side = 65;
    let crater = Heightmap {
        width: side,
        height: side,
        data: (0..side * side)
            .map(|i| {
                let x = ((i % side) as f32 - 32.0) * 2.0;
                let z = ((i / side) as f32 - 32.0) * 2.0;
                let radius = x.hypot(z);
                0.5 - 0.075 * (-(radius / 7.0).powi(2)).exp() + 0.015 * (-((radius - 10.0) / 2.0).powi(2)).exp()
            })
            .collect(),
    };
    let pyramid = build_pyramid(&crater, 100.0, 0.0);
    let info = TerrainInfo::new(&crater, &pyramid, 2.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap: crater,
        pyramid,
        info,
        materials: None,
    });
    let feed = Arc::new(DecalFeed::default());
    feed.place(Decal::permanent([64.0, 64.0], [10.0, 8.0], 0.6, layer));
    let mut app = TerrainApp::new(source, TerrainOutput::Readback).with_decals(DecalsSetup {
        atlas: Some(atlas),
        debug: DecalDebug::DisplacedUv,
        ..DecalsSetup::new(feed, 8)
    });
    let view = app.view();
    view.lock().unwrap().camera = Camera {
        target: Vec2::splat(64.0),
        ..camera
    };
    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let mut headless = Headless::new(&gpu, &mut app, target, ExecutorConfig::default()).unwrap();
    let capture = |headless: &mut Headless<_>, app: &mut TerrainApp| {
        headless.frames(app, 5).unwrap();
        headless.wait().unwrap();
        app.readback().unwrap()
    };
    let centre_hits = |pixels: &[u8]| {
        let centre = (extent.width * (extent.height / 2) + extent.width / 2) as usize;
        pixels.as_chunks::<4>().0[centre - 12..=centre + 12]
            .iter()
            .filter(|px| px[2] >= 245)
            .count()
    };
    assert_eq!(centre_hits(&capture(&mut headless, &mut app)), 0);
    {
        let mut view = view.lock().unwrap();
        view.camera.yaw += std::f32::consts::PI;
        view.decal_relief.mode = DecalReliefMode::RobustConeStepMapping;
        view.decal_relief.adaptive_quality = false;
    }
    assert!(centre_hits(&capture(&mut headless, &mut app)) > 15);
    drop(headless);
    ae_render::thread::RenderApp::destroy(&mut app, &gpu);
    assert_clean(&gpu);
}

#[test]
#[ignore = "needs a Vulkan GPU"]
fn decal_normals_follow_the_footprint_and_compose() {
    use ae_render::decals::{
        Decal, DecalAtlas, DecalDebug, DecalFeed, DecalReliefMode, DecalReliefSettings, DecalsSetup, Image,
        RCSM_HEIGHT_SCALE, Relief,
    };
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::terrain::tileset::{Tileset, temperate};
    use ae_render::thread::{TargetInfo, run_headless};
    use glam::{Vec2, Vec3, Vec4};

    let gpu = gpu();
    let heightmap = Heightmap {
        width: 65,
        height: 65,
        data: vec![0.5; 65 * 65],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 4.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let mut atlas = DecalAtlas {
        size: 16,
        colour: vec![],
        surface: vec![],
        height_layers: vec![],
        materials: vec![],
    };
    for (alpha, axis) in [(255, 0), (255, 1), (0, 0), (255, 2)] {
        let colour = Image {
            width: 16,
            height: 16,
            texels: vec![[128, 128, 128, alpha]; 256],
        };
        let material = Image {
            width: 16,
            height: 16,
            texels: (0..256)
                .map(|i| {
                    let step = [i % 16, i / 16, 0][axis];
                    [64 + 8 * step as u8, 77, 153, 255]
                })
                .collect(),
        };
        atlas.add_material_layer(&colour, &material).unwrap();
    }
    let atlas = Arc::new(atlas);
    let extent = vk::Extent2D {
        width: 384,
        height: 288,
    };
    let camera = Camera {
        target: Vec2::splat(128.0),
        ground: 50.0,
        distance: 128.0,
        yaw: 0.4,
        pitch: 1.5,
        fov_y: 1.0,
    };
    let pos = [
        [100.0, 114.0],
        [128.0, 114.0],
        [156.0, 114.0],
        [100.0, 140.0],
        [128.0, 140.0],
        [156.0, 140.0],
        [86.0, 128.0],
        [170.0, 128.0],
    ];
    let decal = |i, layer, angle| Decal {
        relief: Relief::World(12.75),
        ..Decal::permanent(pos[i], [8.0, 4.0], angle, layer)
    };
    let decals = [
        decal(0, 0, 0.0),
        decal(1, 1, 0.0),
        decal(2, 0, std::f32::consts::FRAC_PI_2),
        decal(3, 0, 0.0),
        decal(3, 1, 0.0),
        decal(4, 1, 0.0),
        decal(4, 0, 0.0),
        decal(5, 2, 0.0),
        Decal {
            relief: Relief::World(0.0),
            ..decal(6, 0, 0.0)
        },
        decal(7, 3, 0.0),
    ];
    let render = |camera: Camera, tileset: Option<Arc<Tileset>>, relief, debug| {
        let feed = Arc::new(DecalFeed::default());
        feed.place_all(&decals);
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_decals(DecalsSetup {
            atlas: Some(atlas.clone()),
            debug,
            relief,
            ..DecalsSetup::new(feed, 16)
        });
        if let Some(tileset) = tileset {
            app = app.with_tileset(tileset);
        }
        app.view().lock().unwrap().camera = camera;
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        run_headless(&gpu, &mut app, target, 4, ExecutorConfig::default()).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        let view_proj = lod_view(&source, &camera, extent.width, extent.height).view_proj;
        pos.map(|[x, z]| {
            let clip = view_proj * Vec4::new(x, 50.0, z, 1.0);
            let u = ((clip.x / clip.w * 0.5 + 0.5) * extent.width as f32) as usize;
            let v = ((clip.y / clip.w * 0.5 + 0.5) * extent.height as f32) as usize;
            let px = (v * extent.width as usize + u) * 4;
            Vec3::from_array(std::array::from_fn(|k| {
                let s = f32::from(pixels[px + k]) / 255.0;
                let linear = if s <= 0.04045 {
                    s / 12.92
                } else {
                    ((s + 0.055) / 1.055).powf(2.4)
                };
                linear * 2.0 - 1.0
            }))
        })
    };
    let u = Vec3::new(-0.4, 1.0, 0.0);
    let v = Vec3::new(0.0, 1.0, -0.8);
    let rotated_u = Vec3::new(0.0, 1.0, -0.4);
    let opposite = Camera {
        yaw: camera.yaw + std::f32::consts::PI,
        ..camera
    };
    let oblique = Camera {
        yaw: camera.yaw + std::f32::consts::FRAC_PI_2,
        pitch: 0.65,
        ..camera
    };
    let farther = Camera {
        distance: 220.0,
        yaw: 1.6,
        ..camera
    };
    let normal = DecalReliefSettings::default();
    assert_eq!(normal.mode, DecalReliefMode::NormalMapping);
    let want = [u, v, rotated_u, v, u, Vec3::Y, Vec3::Y, Vec3::Y];
    for camera in [camera, opposite, oblique, farther] {
        for (i, (got, expected)) in render(camera, None, normal, DecalDebug::Normal)
            .into_iter()
            .zip(want)
            .enumerate()
        {
            assert!(
                got.distance(expected.normalize()) < 0.018,
                "normal at yaw {}, pitch {}, case {i}: {got} vs {expected}",
                camera.yaw,
                camera.pitch
            );
        }
    }
    let disabled = DecalReliefSettings {
        mode: DecalReliefMode::Disabled,
        ..normal
    };
    for actual in render(camera, None, disabled, DecalDebug::Normal) {
        assert!(actual.distance(Vec3::Y) < 0.018, "disabled normal {actual}");
    }
    let cone = DecalReliefSettings {
        mode: DecalReliefMode::RobustConeStepMapping,
        adaptive_quality: false,
        ..normal
    };
    let markers = render(camera, None, cone, DecalDebug::OverlapFallback);
    assert!(
        markers[3].y > 0.9 && markers[4].y > 0.9,
        "overlapped instances fall back"
    );
    assert!(markers[0].y < -0.9, "the isolated decal can trace");
    let cone_normals = render(camera, None, cone, DecalDebug::Normal);
    let scaled = |n: Vec3| Vec3::new(n.x * RCSM_HEIGHT_SCALE, n.y, n.z * RCSM_HEIGHT_SCALE).normalize();
    assert!(
        cone_normals[3].distance(scaled(v)) < 0.018,
        "RCSM overlap, V ramp newest: {}",
        cone_normals[3]
    );
    assert!(
        cone_normals[4].distance(scaled(u)) < 0.018,
        "RCSM overlap, U ramp newest: {}",
        cone_normals[4]
    );
    // A terrain detail tilted along Z remains under a full-coverage U ramp.
    let mut tileset = temperate(16);
    tileset.anti_tiling = 0.0;
    for layer in &mut tileset.layers {
        layer.settings.normal_strength = 1.0;
        for normal in layer.normal.as_chunks_mut::<2>().0 {
            *normal = [128, 179];
        }
    }
    let normals = render(camera, Some(Arc::new(tileset)), normal, DecalDebug::Normal);
    let terrain = normals[7].normalize();
    assert!(
        terrain.z > 0.3,
        "the tileset's detail normal should tilt along Z: {terrain}"
    );
    let expected = glam::Quat::from_rotation_arc(Vec3::Y, terrain) * u.normalize();
    assert!(normals[0].distance(expected) < 0.02, "{} vs {expected}", normals[0]);
    assert_clean(&gpu);
}

/// The GPU mip generator (`upload::image_mipped`, `shaders/mipgen.comp`):
/// every level a 2x2 box of the one above, down to 1x1, every layer.
/// * Coverage (the decal atlas): a transparent texel leaves no colour.
/// * sRGB is averaged in linear light (black and white make sRGB 188).
/// * Normal: x, y in rg with z rebuilt, summed and renormalised (not the
///   plain average of x), b and a averaged; also as RG8 (the tileset).
/// * R8 averages (the particle puff, the anti-tiling noise).
#[test]
#[ignore = "needs a Vulkan GPU"]
fn mips_are_made_on_the_gpu() {
    use ae_render::upload::{MipFilter, image_mipped, read_image};
    let gpu = gpu();
    let array = vk::ImageViewType::TYPE_2D_ARRAY;
    let make = |format, size: u32, layers: &[&[u8]], filter, texel: u64| {
        let mut image = image_mipped(&gpu, format, size, size, layers, filter, array, "mip test").unwrap();
        let levels = read_image(&gpu, &image, texel).unwrap();
        image.destroy(&gpu);
        levels
    };

    // Two layers, 2x2: opaque red with one transparent blue texel; white
    // and black, opaque.
    let red: Vec<u8> = [[255, 0, 0, 255], [255, 0, 0, 255], [255, 0, 0, 255], [0, 0, 255, 0]].concat();
    let grey: Vec<u8> = [
        [255, 255, 255, 255],
        [0, 0, 0, 255],
        [255, 255, 255, 255],
        [0, 0, 0, 255],
    ]
    .concat();
    let levels = make(vk::Format::R8G8B8A8_SRGB, 2, &[&red, &grey], MipFilter::Coverage, 4);
    assert_eq!(levels.len(), 2);
    assert_eq!(&levels[0][..16], &red[..], "mip 0 is the upload");
    println!("coverage: {:?}", levels[1]);
    assert_eq!(
        &levels[1][..4],
        &[255, 0, 0, 191],
        "the transparent blue leaves no colour"
    );
    let g = levels[1][4];
    assert!(
        (187..=189).contains(&g) && levels[1][7] == 255,
        "black and white average in linear light: {:?}",
        &levels[1][4..]
    );

    // Normals: two texels leaning +x (x 0.8, z 0.6), two flat; roughness
    // 0.2 / 0.6, glow 0 / 1.
    let lean = [230u8, 128, 51, 0];
    let flat = [128u8, 128, 153, 255];
    let surface: Vec<u8> = [lean, flat, lean, flat].concat();
    let levels = make(vk::Format::R8G8B8A8_UNORM, 2, &[&surface], MipFilter::Normal, 4);
    let x = |c: u8| c as f32 / 255.0 * 2.0 - 1.0;
    let (lx, ly) = (x(230), x(128));
    let lz = (1.0 - lx * lx - ly * ly).sqrt();
    let (fx, fy) = (x(128), x(128));
    let fz = (1.0 - fx * fx - fy * fy).sqrt();
    let n = [lx + fx, ly + fy, lz + fz];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let want = ((n[0] / len * 0.5 + 0.5) * 255.0).round() as i32;
    let got = levels[1][0] as i32;
    println!("normal: {:?}, want x {want}", levels[1]);
    assert!((got - want).abs() <= 1, "renormalised x: {got}, want {want}");
    assert!((got - 179).abs() > 3, "not the plain average of x");
    assert!(
        (levels[1][2] as i32 - 102).abs() <= 1 && (levels[1][3] as i32 - 128).abs() <= 1,
        "roughness and glow averaged: {:?}",
        levels[1]
    );
    let rg: Vec<u8> = surface.chunks(4).flat_map(|t| [t[0], t[1]]).collect();
    let levels = make(vk::Format::R8G8_UNORM, 2, &[&rg], MipFilter::Normal, 2);
    assert!((levels[1][0] as i32 - want).abs() <= 1, "RG8 normals: {:?}", levels[1]);

    // The signed decal texture averages full vectors as well, preserving
    // the sign and adding no [0,1] encoding bias.
    let lean = [-26214i16, 0].map(i16::to_le_bytes).concat();
    let flat = [0u8; 4];
    let rgba = [lean.as_slice(), &flat, lean.as_slice(), &flat].concat();
    let levels = make(vk::Format::R16G16_SNORM, 2, &[&rgba], MipFilter::NormalSigned, 4);
    let got = f32::from(i16::from_le_bytes([levels[1][0], levels[1][1]])) / 32767.0;
    assert!((got + 0.4 / 0.4_f32.hypot(0.8)).abs() < 1e-4, "signed normal X: {got}");

    // R8, 4x4 down to one texel: the mean.
    let r8: Vec<u8> = (0..16u8).map(|i| i * 16).collect();
    let levels = make(vk::Format::R8_UNORM, 4, &[&r8], MipFilter::Average, 1);
    assert_eq!(levels.len(), 3);
    assert_eq!(levels[1].len(), 4);
    assert!((levels[2][0] as i32 - 120).abs() <= 1, "R8 mean: {:?}", levels);
    assert_clean(&gpu);
}

/// Decode an HDR10 readback's pixel `i` to nits: the 10-bit code through
/// ST.2084's EOTF. (The channels are in BT.2020; compare luminance with
/// BT.2020's coefficients.)
fn hdr10_nits(bytes: &[u8], i: usize) -> [f32; 3] {
    let u = u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    let code = |shift: u32| ((u >> shift) & 0x3ff) as f32 / 1023.0;
    [code(0), code(10), code(20)].map(ae_render::encode::pq_eotf)
}

/// IEEE half to f32 (the scRGB readback is RGBA16F).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let man = (bits & 0x3ff) as f32;
    match exp {
        0 => sign * man * 2f32.powi(-24),
        31 if man == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

fn scrgb_pixel(bytes: &[u8], i: usize) -> [f32; 3] {
    let half = |k: usize| f16_to_f32(u16::from_le_bytes([bytes[i + 2 * k], bytes[i + 2 * k + 1]]));
    [half(0), half(1), half(2)]
}

fn y709(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn y2020(c: [f32; 3]) -> f32 {
    0.2627 * c[0] + 0.6780 * c[1] + 0.0593 * c[2]
}

fn srgb_linear(v: u8) -> f32 {
    let v = v as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Run `post.encode` over a constant display image and return the target's
/// bytes for each value: a hand-built graph (no scene), so the values the
/// shader encodes are exact.
fn encode_values(
    gpu: &Arc<GpuDevice>,
    encoding: ae_render::encode::OutputEncoding,
    extent: vk::Extent2D,
    levels: ae_render::encode::DisplayLevels,
    working: ae_render::colorspace::WorkingSpace,
    gamut: ae_render::colorspace::Gamut,
    values: &[[f32; 4]],
) -> Vec<Vec<u8>> {
    use ae_framegraph::{FrameGraph, LoadOp, PassKind, ResourceId, Role, ShaderStage};
    use ae_render::encode::{DISPLAY_FORMAT, Encode};
    use ae_render::executor::{Binding, BufferBinding, ImageBinding, PassContext, Recorder};

    struct Paint {
        display: ResourceId,
        target: ResourceId,
        buffer: ResourceId,
        encode: Encode,
        sampler: u32,
        encoding: ae_render::encode::OutputEncoding,
        levels: ae_render::encode::DisplayLevels,
        working: ae_render::colorspace::WorkingSpace,
        gamut: ae_render::colorspace::Gamut,
        value: [f32; 4],
    }
    impl Recorder<u8> for Paint {
        fn clear_color(&self, _pass: &u8, resource: ResourceId) -> [f32; 4] {
            if resource == self.display { self.value } else { [0.0; 4] }
        }
        fn record(&mut self, pass: &u8, ctx: &PassContext<'_>) -> anyhow::Result<()> {
            match *pass {
                // The fill pass draws nothing: the clear is the display's
                // value.
                1 => {
                    self.encode
                        .write(ctx.slot, self.encoding, self.levels, self.working, self.gamut);
                    self.encode.record(ctx, ctx.image(self.display)?, self.sampler)?;
                }
                3 => ae_render::demo::record_readback(ctx, ctx.image(self.target)?, ctx.buffer(self.buffer)?),
                _ => {}
            }
            Ok(())
        }
    }

    let format = encoding.readback_format();
    let size = extent.width as u64 * extent.height as u64 * encoding.bytes_per_pixel();
    let display_desc = ImageDesc::color_2d(
        DISPLAY_FORMAT,
        extent.width,
        extent.height,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
    );
    let target_desc = ImageDesc::color_2d(
        format,
        extent.width,
        extent.height,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    let buffer_desc = BufferDesc {
        size,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
    };
    let mut graph = FrameGraph::<u8>::new();
    let display = graph.image("display", display_desc, ResourceClass::External);
    let target = graph.image("target", target_desc, ResourceClass::External);
    let buffer = graph.buffer("readback", buffer_desc, ResourceClass::External);
    graph
        .pass("fill", PassKind::Graphics, 0)
        .color(display, LoadOp::Clear)
        .id();
    graph
        .pass("encode", PassKind::Graphics, 1)
        .sampled(display, ShaderStage::Fragment)
        .color(target, LoadOp::DontCare)
        .id();
    graph
        .pass("copy", PassKind::Transfer, 2)
        .access(target, Role::TransferSrc)
        .access(buffer, Role::TransferDst)
        .id();
    graph
        .pass("host read", PassKind::Transfer, 3)
        .access(buffer, Role::HostRead)
        .root()
        .id();

    let mut image = OwnedImage::new(gpu, display_desc, "display").unwrap();
    let mut out = OwnedImage::new(gpu, target_desc, "target").unwrap();
    let mut readback = OwnedBuffer::new(gpu, buffer_desc, MemoryLocation::GpuToCpu, "readback").unwrap();
    let mut executor = Executor::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    let display_handle = executor
        .bindless_mut()
        .sampled_2d(image.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .unwrap();
    let bindings = [
        Binding::Image(ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: image.image,
            view: image.view,
            format: display_desc.format,
            extent,
            range: image.range(),
            identity: (0xBEEF << 40) | 1,
            sampled: Some(display_handle.0.index),
            storage: None,
        }),
        Binding::Image(ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: out.image,
            view: out.view,
            format: target_desc.format,
            extent,
            range: out.range(),
            identity: (0xBEEF << 40) | 2,
            sampled: None,
            storage: None,
        }),
        Binding::Buffer(BufferBinding {
            buffer: readback.buffer,
            offset: 0,
            size,
            identity: (0xBEEF << 40) | 3,
            storage: None,
        }),
    ];
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST);
    let sampler_index = executor.bindless_mut().sampler(&sampler_info).unwrap().index;
    let encode = Encode::new(gpu, executor.bindless_mut(), encoding, format, 2).unwrap();
    let mut paint = Paint {
        display,
        target,
        buffer,
        encode,
        sampler: sampler_index,
        encoding,
        levels,
        working,
        gamut,
        value: [0.0; 4],
    };
    executor.set_graph(graph).unwrap();
    let mut frames = Vec::new();
    for value in values {
        paint.value = *value;
        let token = executor.begin_frame().unwrap();
        executor
            .execute(
                token,
                &mut paint,
                &|resource| bindings.get(resource.index()).copied(),
                &[],
                &[],
            )
            .unwrap();
        executor.wait(token).unwrap();
        frames.push(readback.mapped().unwrap().to_vec());
    }
    paint.encode.destroy(gpu);
    image.destroy(gpu);
    out.destroy(gpu);
    readback.destroy(gpu);
    frames
}

/// `post.encode` writes ST.2084 (M13 stage 2): display values at 100, 203 and
/// 1000 nits with a 203-nit paper white and a 1000-nit peak encode to the
/// standard's 10-bit codes (within 1 LSB), and a value above the peak stops
/// at the peak's code.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn post_encode_writes_st_2084_codes() {
    use ae_render::encode::{DisplayLevels, OutputEncoding, pq_code};

    let gpu = gpu();
    let extent = vk::Extent2D { width: 4, height: 4 };
    let levels = DisplayLevels {
        paper_white: 203.0,
        peak: 1000.0,
    };
    // Display-linear values: 100, 203 and 1000 nits over the paper white,
    // then one past the peak (8.0 = 1624 nits).
    let values: [[f32; 4]; 4] = [
        [100.0 / 203.0, 100.0 / 203.0, 100.0 / 203.0, 1.0],
        [1.0; 4],
        [1000.0 / 203.0; 4],
        [8.0; 4],
    ];
    let nits = [100.0f32, 203.0, 1000.0, 1000.0];
    let frames = encode_values(
        &gpu,
        OutputEncoding::Hdr10Pq,
        extent,
        levels,
        ae_render::colorspace::WorkingSpace::Bt709,
        OutputEncoding::Hdr10Pq.default_gamut(),
        &values,
    );
    for (frame, want) in frames.iter().zip(nits) {
        let code = pq_code(want);
        let mut worst = 0i32;
        for p in 0..(extent.width * extent.height) as usize {
            let u = u32::from_le_bytes([frame[p * 4], frame[p * 4 + 1], frame[p * 4 + 2], frame[p * 4 + 3]]);
            for shift in [0u32, 10, 20] {
                worst = worst.max((((u >> shift) & 0x3ff) as i32 - code as i32).abs());
            }
        }
        eprintln!("{want:.0} nits: code {code}, worst channel off by {worst} LSB");
        assert!(worst <= 1, "{want} nits is {} LSB from ST.2084's code {code}", worst);
    }
    assert_clean(&gpu);
}

/// `post.encode` in scRGB (M13 stage 2): the display's light times paper
/// white / 80, in RGBA16F: at an 80-nit paper white the values are the
/// display's own, at 203 nits they are 2.5375 times, and a negative display
/// value (out of gamut, which the gamut map may produce) stays negative.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn post_encode_writes_scrgb_relative_to_80_nits() {
    use ae_render::encode::{DisplayLevels, OutputEncoding};

    let gpu = gpu();
    let extent = vk::Extent2D { width: 4, height: 4 };
    let value: [f32; 4] = [0.4, 0.2, -0.1, 1.0];
    let frames = encode_values(
        &gpu,
        OutputEncoding::ScRgb,
        extent,
        DisplayLevels {
            paper_white: 203.0,
            peak: 1000.0,
        },
        ae_render::colorspace::WorkingSpace::Bt709,
        OutputEncoding::ScRgb.default_gamut(),
        &[value],
    );
    let p = scrgb_pixel(&frames[0], 0);
    let scale = 203.0 / 80.0;
    eprintln!("scRGB: {p:?} for display {value:?}, scale {scale}");
    for k in 0..3 {
        assert!(
            (p[k] - value[k] * scale).abs() < 1e-3,
            "channel {k}: {} vs {}",
            p[k],
            value[k] * scale
        );
    }
    // Paper white at 80 nits: no scale.
    let frames = encode_values(
        &gpu,
        OutputEncoding::ScRgb,
        extent,
        DisplayLevels::FLAT,
        ae_render::colorspace::WorkingSpace::Bt709,
        OutputEncoding::ScRgb.default_gamut(),
        &[value],
    );
    let p = scrgb_pixel(&frames[0], 0);
    for k in 0..3 {
        assert!((p[k] - value[k]).abs() < 1e-3, "channel {k}: {} vs {}", p[k], value[k]);
    }
    assert_clean(&gpu);
}

/// HDR output (M13 stage 2): with the display's peak equal to its paper
/// white the HDR10 frame carries the SDR frame's light. The curve's peak
/// anchor is 1 in both, so the display is the SDR frame's linear values;
/// decoded out of PQ and BT.2020, the luminance matches the sRGB frame's
/// within the two quantizations (8-bit sRGB, 10-bit PQ).
#[test]
#[ignore = "needs a Vulkan GPU"]
fn hdr10_readback_matches_the_sdr_frame_at_paper_white_peak() {
    use ae_render::encode::{DisplayLevels, OutputEncoding};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::tonemap::ToneParams;
    use glam::Vec2;

    let gpu = gpu();
    let heightmap = Heightmap {
        width: 129,
        height: 129,
        data: vec![0.5; 129 * 129],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let render = |encoding: OutputEncoding, levels: DisplayLevels| -> Vec<u8> {
        // Straight down at flat, untextured ground under the full sun, as
        // the curve's gate: one sunlit surface and the sky above it.
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_tone(ToneParams {
                bloom: 0.0,
                ..Default::default()
            })
            .with_display(encoding, levels);
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = Vec2::new(512.0, 512.0);
            view.camera.pitch = 1.55;
            view.camera.distance = 100.0;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 4, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let sdr = render(OutputEncoding::Sdr, DisplayLevels::default());
    let levels = DisplayLevels {
        paper_white: 203.0,
        peak: 203.0,
    };
    let hdr = render(OutputEncoding::Hdr10Pq, levels);
    let mut diffs = Vec::new();
    for p in 0..(extent.width * extent.height) as usize {
        let s = [
            srgb_linear(sdr[p * 4]),
            srgb_linear(sdr[p * 4 + 1]),
            srgb_linear(sdr[p * 4 + 2]),
        ];
        let h = hdr10_nits(&hdr, p * 4).map(|c| c / levels.paper_white);
        diffs.push((y709(s) - y2020(h)).abs());
    }
    let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
    diffs.sort_by(f32::total_cmp);
    let p99 = diffs[diffs.len() * 99 / 100];
    let max = diffs[diffs.len() - 1];
    eprintln!("HDR10 at peak = paper white: luminance diff mean {mean:.5}, p99 {p99:.5}, max {max:.5}");
    assert!(
        mean < 0.0025,
        "the HDR frame carries the SDR frame's light: mean {mean}"
    );
    assert!(max < 0.01, "no pixel strays: max {max}");
    assert_clean(&gpu);
}

/// M13 stage 3: in HDR the tonemap's output (the display intermediate) stays
/// in the working space and `post.encode` maps it to the output gamut once.
/// The unlit sky through the whole HDR10 chain decodes to the same light in
/// a BT.2020 working space as in BT.709, per BT.2020 container channel,
/// within the 10-bit PQ step.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn hdr10_bt2020_working_space_maps_to_the_output_once() {
    use ae_render::colorspace::WorkingSpace;
    use ae_render::encode::{DisplayLevels, OutputEncoding};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::tonemap::ToneParams;
    use glam::Vec2;

    let gpu = gpu();
    let heightmap = Heightmap {
        width: 65,
        height: 65,
        data: vec![0.5; 65 * 65],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 160,
        height: 120,
    };
    let levels = DisplayLevels {
        paper_white: 203.0,
        peak: 1000.0,
    };
    let render = |working: WorkingSpace| -> Vec<u8> {
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_working_space(working)
            .with_tone(ToneParams {
                bloom: 0.0,
                ..Default::default()
            })
            .with_display(OutputEncoding::Hdr10Pq, levels);
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = Vec2::new(256.0, 256.0);
            // Up: the clear colour (the sky) fills the frame.
            view.camera.pitch = -1.4;
            view.camera.distance = 100.0;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 4, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let narrow = render(WorkingSpace::Bt709);
    let wide = render(WorkingSpace::Bt2020);
    let mut worst = 0f32;
    for p in 0..(extent.width * extent.height) as usize {
        let (a, b) = (hdr10_nits(&narrow, p * 4), hdr10_nits(&wide, p * 4));
        for k in 0..3 {
            // Relative to the channel's light: one PQ code near 100 nits is
            // about 1%.
            worst = worst.max((a[k] - b[k]).abs() / a[k].max(1.0));
        }
    }
    eprintln!(
        "HDR10 sky, BT.2020 against BT.709 working space: worst channel {:.2}%",
        100.0 * worst
    );
    assert!(
        worst < 0.03,
        "the wide working space maps to the output once: worst {worst}"
    );
    assert_clean(&gpu);
}

/// scRGB output (M13 stage 2): with the display's peak at its paper white,
/// the 16F readback holds the SDR frame's linear values (1.0 = 80 nits), and
/// at a 203-nit paper white the same frame is 2.5375 times: the encode's
/// scale is exactly paper white / 80.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn scrgb_readback_scales_the_display_by_paper_white() {
    use ae_render::encode::{DisplayLevels, OutputEncoding};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::tonemap::ToneParams;
    use glam::Vec2;

    let gpu = gpu();
    let heightmap = Heightmap {
        width: 129,
        height: 129,
        data: vec![0.5; 129 * 129],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 160,
        height: 120,
    };
    let render = |encoding: OutputEncoding, levels: DisplayLevels| -> Vec<u8> {
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_tone(ToneParams {
                bloom: 0.0,
                ..Default::default()
            })
            .with_display(encoding, levels);
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = Vec2::new(512.0, 512.0);
            view.camera.pitch = 1.55;
            view.camera.distance = 100.0;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 4, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let sdr = render(OutputEncoding::Sdr, DisplayLevels::default());
    let flat = render(OutputEncoding::ScRgb, DisplayLevels::FLAT);
    let scale = |sdr: &[u8], hdr: &[u8], factor: f32| -> (f32, f32) {
        let (mut sum, mut worst) = (0.0, 0.0f32);
        for p in 0..(extent.width * extent.height) as usize {
            let s = [
                srgb_linear(sdr[p * 4]),
                srgb_linear(sdr[p * 4 + 1]),
                srgb_linear(sdr[p * 4 + 2]),
            ];
            let h = scrgb_pixel(hdr, p * 8);
            for k in 0..3 {
                let d = (s[k] - h[k] / factor).abs();
                sum += d;
                worst = worst.max(d);
            }
        }
        (sum / (3.0 * (extent.width * extent.height) as f32), worst)
    };
    let (mean, max) = scale(&sdr, &flat, 1.0);
    eprintln!("scRGB at 80 nits: channel diff mean {mean:.5}, max {max:.5}");
    assert!(
        mean < 0.002 && max < 0.01,
        "80-nit scRGB is the SDR frame: mean {mean}, max {max}"
    );
    let levels = DisplayLevels {
        paper_white: 203.0,
        peak: 203.0,
    };
    let scaled = render(OutputEncoding::ScRgb, levels);
    let (mean, max) = scale(&sdr, &scaled, 203.0 / 80.0);
    eprintln!("scRGB at 203 nits (/ 2.5375): channel diff mean {mean:.5}, max {max:.5}");
    assert!(
        mean < 0.002 && max < 0.01,
        "203-nit scRGB over 2.5375 is the SDR frame: mean {mean}, max {max}"
    );
    assert_clean(&gpu);
}

/// HDR output (M13 stage 2): an emitter decodes above paper white, and
/// nothing decodes above the display's peak. A 203-nit paper white and a
/// 441-nit peak (the reference display's): the explosion's flash is several
/// scene units, which the curve places above 1.0 (paper white) and below
/// peak / paper white.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn hdr_emitters_reach_above_paper_white_and_stop_at_peak() {
    use ae_render::encode::{DisplayLevels, OutputEncoding};
    use ae_render::particles::{ParticleBlend, ParticleFeed, ParticlesSetup, default_effects, explosion};
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::camera::Camera;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{Headless, TargetInfo};
    use ae_render::tonemap::ToneParams;
    use glam::Vec2;

    let gpu = gpu();
    let n = 65;
    let heightmap = Heightmap {
        width: n,
        height: n,
        data: vec![0.1; n * n],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 4.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let levels = DisplayLevels {
        paper_white: 203.0,
        peak: 441.0,
    };
    let feed = Arc::new(ParticleFeed::default());
    feed.emit_all(&explosion([128.0, 2.0, 128.0], true, 3.0, 7));
    let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
        .with_display(OutputEncoding::Hdr10Pq, levels)
        .with_particles(ParticlesSetup {
            feed,
            capacity: 1024,
            effects: default_effects(),
            downscale: 1,
            blend: ParticleBlend::Sorted,
        })
        .with_tone(ToneParams {
            bloom: 0.0,
            ..Default::default()
        });
    {
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera = Camera {
            target: Vec2::new(128.0, 128.0),
            ground: 0.0,
            distance: 40.0,
            yaw: 0.0,
            pitch: 0.5,
            fov_y: 0.8,
        };
        view.fog = false;
    }
    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
    let mut run = Headless::new(&gpu, &mut app, target, config).unwrap();
    // The flash lives about 0.2 s: watch the first ten frames. The brightest
    // pixel is its core; across every frame, no channel is above the peak.
    let mut best = (0.0f32, 0u32, 0u32);
    let mut worst_channel = 0.0f32;
    let mut band = false;
    for _ in 0..10 {
        run.frames(&mut app, 1).unwrap();
        run.wait().unwrap();
        let pixels = app.readback().unwrap();
        for p in 0..(extent.width * extent.height) as usize {
            let nits = hdr10_nits(&pixels, p * 4);
            worst_channel = worst_channel.max(nits[0].max(nits[1]).max(nits[2]));
            let y = y2020(nits);
            band |= y > levels.paper_white && y < levels.peak;
            if y > best.0 {
                best = (y, p as u32 % extent.width, p as u32 / extent.width);
            }
        }
    }
    run.finish().unwrap();
    ae_render::thread::RenderApp::destroy(&mut app, &gpu);
    let (y, x, py) = best;
    eprintln!(
        "explosion: brightest pixel ({x}, {py}) decodes to {y:.1} nits; worst channel {worst_channel:.1}; a gradient band above paper white: {} (paper white {:.0}, peak {:.0})",
        band, levels.paper_white, levels.peak
    );
    assert!(y > levels.paper_white, "the emitter is above paper white: {y}");
    assert!(
        band,
        "some pixel is between paper white and the peak (a gradient, not only a clamp)"
    );
    assert!(
        worst_channel <= levels.peak * 1.01,
        "nothing is above the peak: {worst_channel}"
    );
    assert_clean(&gpu);
}

/// M13 stage 3: unlit content in a BT.2020 working space reaches SDR within
/// one 8-bit code of the same content lit in BT.709. The sky is the unlit
/// case (the clear colour, converted into the working space and back by the
/// tonemap): a camera looking up sees only it, and the default curve's clamp
/// is a linear read.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn bt2020_unlit_content_reaches_sdr_within_one_code() {
    use ae_render::colorspace::WorkingSpace;
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::tonemap::ToneParams;
    use glam::Vec2;

    let gpu = gpu();
    let heightmap = Heightmap {
        width: 65,
        height: 65,
        data: vec![0.5; 65 * 65],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 160,
        height: 120,
    };
    let render = |working: WorkingSpace| -> Vec<u8> {
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_working_space(working)
            .with_tone(ToneParams {
                bloom: 0.0,
                ..Default::default()
            });
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = Vec2::new(256.0, 256.0);
            // Up: the clear colour (the sky) fills the frame, no terrain.
            view.camera.pitch = -1.4;
            view.camera.distance = 100.0;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 4, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let srgb = render(WorkingSpace::Bt709);
    let wide = render(WorkingSpace::Bt2020);
    let (mut worst, mut differing) = (0i32, 0usize);
    for p in 0..(extent.width * extent.height) as usize {
        for k in 0..3 {
            let d = srgb[p * 4 + k].abs_diff(wide[p * 4 + k]) as i32;
            worst = worst.max(d);
            differing += (d > 0) as usize;
        }
    }
    eprintln!(
        "unlit BT.2020 against BT.709: worst {worst} codes, {differing} of {} channels differ",
        3 * (extent.width * extent.height) as usize
    );
    assert!(worst <= 1, "unlit content reaches SDR within 1 code: worst {worst}");
    assert_clean(&gpu);
}

/// M13 stage 3: lit content differs between the working spaces, by how much
/// is measured here. The same sunlit ground with the same colours in BT.709
/// and BT.2020: the conversions are exact for unlit colours, but the lighting
/// multiplies each colour in its own space (and the tone curve maps the max
/// channel), so the two frames are close rather than identical.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn bt2020_lit_content_differs_from_bt709_by_measured_numbers() {
    use ae_render::colorspace::WorkingSpace;
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::tonemap::ToneParams;
    use glam::Vec2;

    let gpu = gpu();
    let heightmap = Heightmap {
        width: 129,
        height: 129,
        data: vec![0.5; 129 * 129],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let render = |working: WorkingSpace| -> Vec<u8> {
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_working_space(working)
            .with_tone(ToneParams {
                bloom: 0.0,
                ..Default::default()
            });
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = Vec2::new(512.0, 512.0);
            view.camera.pitch = 1.55;
            view.camera.distance = 100.0;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 4, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let srgb = render(WorkingSpace::Bt709);
    let wide = render(WorkingSpace::Bt2020);
    let mut diffs = Vec::new();
    let mut close = 0usize;
    for p in 0..(extent.width * extent.height) as usize {
        for k in 0..3 {
            let d = srgb[p * 4 + k].abs_diff(wide[p * 4 + k]) as f32;
            diffs.push(d);
            close += (d <= 1.0) as usize;
        }
    }
    let n = diffs.len() as f32;
    let mean = diffs.iter().sum::<f32>() / n;
    diffs.sort_by(f32::total_cmp);
    let p99 = diffs[(diffs.len() * 99 / 100).min(diffs.len() - 1)];
    let max = diffs[diffs.len() - 1];
    eprintln!(
        "lit BT.2020 against BT.709: channel diff mean {mean:.2}, p99 {p99:.0}, max {max:.0} codes; {:.2}% within 1 code",
        100.0 * close as f32 / n
    );
    assert!(mean < 3.0, "the lit frames stay close: mean {mean}");
    assert!(max < 24.0, "no gross difference: max {max}");
    assert_clean(&gpu);
}

/// M13 stage 3: `post.encode` maps the working space's wide colour into the
/// output gamut by compression toward the achromatic axis. A BT.2020
/// working space with P3 output: a P3 patch decodes inside P3 and outside
/// BT.709, a BT.2020 patch outside P3 comes back inside with its luminance
/// and hue kept, and an in-gamut patch is untouched.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn post_encode_compresses_wide_content_into_the_output_gamut() {
    use ae_render::colorspace::{self, Gamut, WorkingSpace};
    use ae_render::encode::{DisplayLevels, OutputEncoding};

    let gpu = gpu();
    let extent = vk::Extent2D { width: 4, height: 4 };
    let levels = DisplayLevels {
        paper_white: 203.0,
        peak: 1000.0,
    };
    // Display values are in the working space (BT.2020): P3 green expressed
    // there, a P3-inside patch (70% P3 white + 30% P3 green), and BT.2020
    // green, which is outside P3.
    let p3_green = [0.0, 1.0, 0.0];
    let p3_in = [0.3, 0.5, 0.2];
    let values = [
        colorspace::convert(Gamut::P3, Gamut::Bt2020, p3_green),
        colorspace::convert(Gamut::P3, Gamut::Bt2020, p3_in),
        [0.0, 1.0, 0.0],
    ];
    let frames = encode_values(
        &gpu,
        OutputEncoding::Hdr10Pq,
        extent,
        levels,
        WorkingSpace::Bt2020,
        Gamut::P3,
        &values.map(|v| [v[0], v[1], v[2], 1.0]),
    );
    // Decode PQ to paper-white units (the BT.2020 container), then to P3 and
    // BT.709 challenge values.
    let decode = |frame: &[u8]| -> [f32; 3] {
        let nits = hdr10_nits(frame, 0);
        colorspace::convert(Gamut::Bt2020, Gamut::P3, nits.map(|n| n / levels.paper_white))
    };
    let decoded: Vec<[f32; 3]> = frames.iter().map(|f| decode(f)).collect();
    let to709 = |c: [f32; 3]| colorspace::convert(Gamut::P3, Gamut::Bt709, c);
    eprintln!(
        "P3 green: {:.4?} -> P3 {:.4?}, 709 {:.4?}",
        values[0],
        decoded[0],
        to709(decoded[0])
    );
    eprintln!("P3 patch: {:.4?} -> P3 {:.4?}", values[1], decoded[1]);
    eprintln!(
        "BT.2020 green: {:.4?} -> P3 {:.4?}, 709 {:.4?}",
        values[2],
        decoded[2],
        to709(decoded[2])
    );
    // A P3 patch decodes inside P3 (all channels non-negative) and outside
    // BT.709 (a negative channel there).
    assert!(
        decoded[0].iter().all(|v| *v >= -1e-3),
        "P3 green inside P3: {:?}",
        decoded[0]
    );
    assert!(
        to709(decoded[0]).iter().any(|v| *v < -0.01),
        "P3 green outside BT.709: {:?}",
        to709(decoded[0])
    );
    // In gamut: untouched, within the 10-bit quantization (about 1.5% here).
    for k in 0..3 {
        assert!(
            (decoded[0][k] - p3_green[k]).abs() < 0.02,
            "P3 green unchanged: {:?}",
            decoded[0]
        );
        assert!(
            (decoded[1][k] - p3_in[k]).abs() < 0.02,
            "P3 patch unchanged: {:?}",
            decoded[1]
        );
    }
    // BT.2020 green: compressed into P3 (no negative channel), still outside
    // BT.709, with its luminance and hue kept.
    let wide = colorspace::convert(Gamut::Bt2020, Gamut::P3, values[2]);
    let want = colorspace::compress(Gamut::P3, wide);
    assert!(
        decoded[2].iter().all(|v| *v >= -1e-3),
        "BT.2020 green compresses into P3: {:?}",
        decoded[2]
    );
    assert!(
        to709(decoded[2]).iter().any(|v| *v < -0.01),
        "still outside BT.709: {:?}",
        to709(decoded[2])
    );
    let y = |c: [f32; 3]| Gamut::P3.luminance(c);
    assert!(
        (y(decoded[2]) - y(want)).abs() < 0.01,
        "luminance kept: {} vs {}",
        y(decoded[2]),
        y(want)
    );
    for k in 0..3 {
        assert!(
            (decoded[2][k] - want[k]).abs() < 0.02,
            "channel {k}: {:?} vs {want:?}",
            decoded[2]
        );
    }
    assert_clean(&gpu);
}

/// M13 stage 3: automatic exposure meters the stored scene in its own working
/// space, so a BT.2020 scene is metered with BT.2020's luminance row and
/// adapts to the same key EV. The residual between the two displays is the
/// lit content's own difference (the max-channel curve in a different basis),
/// not an exposure change.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn auto_exposure_meters_bt2020_in_its_own_space() {
    use ae_render::colorspace::WorkingSpace;
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::tonemap::ToneParams;
    use glam::Vec2;

    let gpu = gpu();
    let heightmap = Heightmap {
        width: 129,
        height: 129,
        data: vec![0.5; 129 * 129],
    };
    let pyramid = build_pyramid(&heightmap, 100.0, 0.0);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, 100.0, 0.0, None);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let extent = vk::Extent2D {
        width: 160,
        height: 120,
    };
    let render = |working: WorkingSpace| -> Vec<u8> {
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_working_space(working)
            .with_tone(ToneParams {
                auto: true,
                bloom: 0.0,
                ..Default::default()
            });
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = Vec2::new(512.0, 512.0);
            view.camera.pitch = 1.55;
            view.camera.distance = 100.0;
            view.fog = false;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        // 10 s at the readback clock: adaptation has settled.
        run_headless(&gpu, &mut app, target, 600, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let linear = |v: u8| -> f32 {
        let v = v as f32 / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let mean = |pixels: &[u8]| -> f32 {
        let total = (extent.width * extent.height) as f32;
        pixels
            .chunks(4)
            .map(|p| (linear(p[0]) + linear(p[1]) + linear(p[2])) / 3.0)
            .sum::<f32>()
            / total
    };
    let narrow = mean(&render(WorkingSpace::Bt709));
    let wide = mean(&render(WorkingSpace::Bt2020));
    eprintln!(
        "auto-exposed ground: BT.709 {narrow:.4}, BT.2020 {wide:.4}, ratio {:.4}",
        wide / narrow
    );
    assert!(
        (wide / narrow - 1.0).abs() < 0.05,
        "the meter settles the wide scene at the same key: {:.4} against {:.4}",
        wide,
        narrow
    );
    assert_clean(&gpu);
}

/// M13 stage 4: a colour authored in P3 and uploaded into the default
/// BT.709 working space keeps its wide gamut through `post.encode`. The
/// fire's death tint (`assets/effects.json`, tagged `p3`) has a negative
/// blue channel in BT.709; the display intermediate keeps the colour in the
/// working space, and only `post.encode` maps it to P3, where it decodes
/// back as the authored colour instead of being clipped to BT.709.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn p3_authored_effect_survives_post_encode() {
    use ae_render::colorspace::{self, Gamut, WorkingSpace};
    use ae_render::encode::{DisplayLevels, OutputEncoding};
    use ae_render::particles::{default_effects, effect};

    let gpu = gpu();
    let extent = vk::Extent2D { width: 4, height: 4 };
    let levels = DisplayLevels {
        paper_white: 203.0,
        peak: 1000.0,
    };
    let fire = default_effects()[effect::FIRE as usize];
    let authored = [fire.color_end[0], fire.color_end[1], fire.color_end[2]];
    // The upload (ParticleRenderer::new): from the effect's gamut into the
    // working space.
    let display = colorspace::convert(Gamut::P3, Gamut::Bt709, authored);
    assert!(display[2] < -0.01, "the display value is outside BT.709: {display:?}");
    let frames = encode_values(
        &gpu,
        OutputEncoding::Hdr10Pq,
        extent,
        levels,
        WorkingSpace::Bt709,
        Gamut::P3,
        &[[display[0], display[1], display[2], 1.0]],
    );
    let nits = hdr10_nits(&frames[0], 0);
    let p3 = colorspace::convert(Gamut::Bt2020, Gamut::P3, nits.map(|n| n / levels.paper_white));
    let back = colorspace::convert(Gamut::P3, Gamut::Bt709, p3);
    eprintln!(
        "P3 fire end: authored {authored:.4?} -> display (709) {display:.4?} -> decoded P3 {p3:.4?} (709 {back:.4?})"
    );
    // The decoded P3 colour is the authored one within the 10-bit PQ's
    // quantization, and it is still outside BT.709.
    for k in 0..3 {
        assert!(
            (p3[k] - authored[k]).abs() < 0.02,
            "P3 channel {k}: {} vs {}",
            p3[k],
            authored[k]
        );
    }
    assert!(back.iter().any(|v| *v < -0.01), "still outside BT.709: {back:?}");
    assert_clean(&gpu);
}

/// A shader with a per-pass binding (here beside the global set's arrays)
/// fails when its program is built, and the error names the binding and the
/// contract version that dropped it.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn a_per_pass_program_fails_at_load() {
    use ae_render::fullscreen::FullscreenProgram;

    let gpu = gpu();
    let mut executor = Executor::<()>::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    let error = match FullscreenProgram::new(
        &gpu,
        executor.bindless_mut(),
        "mixed",
        ae_render::spirv!("test_mixed.frag"),
        16,
        1,
        &[vk::Format::R8G8B8A8_UNORM],
        2,
    ) {
        Ok(_) => panic!("a per-pass program built"),
        Err(error) => format!("{error:#}"),
    };
    assert!(
        error.contains("per-pass") && error.contains("version 4") && error.contains("binding"),
        "{error}"
    );
    assert_clean(&gpu);
}

/// One fullscreen pass over 64 sampled images (M15 F): the bindless form's
/// handle table has no per-image binding, so the count is not limited to the
/// old layout's.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn a_program_samples_64_images() {
    use ae_framegraph::{LoadOp, ResourceId, Role};
    use ae_render::executor::{PassContext, Recorder};
    use ae_render::fullscreen::{FullscreenProgram, Sampler};

    struct Sample {
        images: Vec<ResourceId>,
        target: ResourceId,
        buffer: ResourceId,
        program: FullscreenProgram,
        sampler: Sampler,
        uniforms: [u8; 32],
    }
    impl Recorder<u32> for Sample {
        fn record(&mut self, pass: &u32, ctx: &PassContext<'_>) -> anyhow::Result<()> {
            match *pass {
                1 => {
                    let images = self
                        .images
                        .iter()
                        .map(|&r| ctx.image(r))
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    self.program.write_uniforms(ctx.slot, &self.uniforms);
                    self.program.record(ctx, &images, self.sampler)?;
                }
                2 => ae_render::demo::record_readback(ctx, ctx.image(self.target)?, ctx.buffer(self.buffer)?),
                _ => {}
            }
            Ok(())
        }
    }

    let gpu = gpu();
    let extent = vk::Extent2D { width: 4, height: 4 };
    let format = vk::Format::R8G8B8A8_UNORM;
    let source_desc = ImageDesc::color_2d(format, extent.width, extent.height, vk::ImageUsageFlags::SAMPLED);
    let target_desc = ImageDesc::color_2d(
        format,
        extent.width,
        extent.height,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    let buffer_desc = BufferDesc {
        size: extent.width as u64 * extent.height as u64 * 4,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
    };
    const COUNT: usize = 64;
    let mut graph = FrameGraph::<u32>::new();
    let images: Vec<ResourceId> = (0..COUNT)
        .map(|i| graph.image(format!("src{i}"), source_desc, ResourceClass::External))
        .collect();
    let target = graph.image("target", target_desc, ResourceClass::External);
    let buffer = graph.buffer("readback", buffer_desc, ResourceClass::External);
    let mut pass = graph.pass("sample", PassKind::Graphics, 1);
    for &image in &images {
        pass = pass.sampled(image, ShaderStage::Fragment);
    }
    pass.color(target, LoadOp::DontCare).id();
    graph
        .pass("copy", PassKind::Transfer, 2)
        .access(target, Role::TransferSrc)
        .access(buffer, Role::TransferDst)
        .id();
    graph
        .pass("host read", PassKind::Transfer, 3)
        .access(buffer, Role::HostRead)
        .root()
        .id();

    // Every source is the same mid grey, so whichever slots the shader
    // reads, the result is the same.
    let grey = [128u8, 128, 128, 255];
    let sources: Vec<OwnedImage> = (0..COUNT)
        .map(|i| ae_render::upload::image(&gpu, format, 1, 1, &[&grey], &format!("source {i}")).unwrap())
        .collect();
    let out = OwnedImage::new(&gpu, target_desc, "target").unwrap();
    let mut readback = OwnedBuffer::new(&gpu, buffer_desc, MemoryLocation::GpuToCpu, "readback").unwrap();

    let mut executor = Executor::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    let handles: Vec<_> = sources
        .iter()
        .map(|image| {
            executor
                .bindless_mut()
                .sampled_2d(image.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .unwrap()
        })
        .collect();
    let mut bindings: Vec<Option<Binding>> = sources
        .iter()
        .zip(&handles)
        .enumerate()
        .map(|(i, (image, handle))| {
            Some(Binding::Image(ImageBinding {
                initial: vk::ImageLayout::UNDEFINED,
                image: image.image,
                view: image.view,
                format,
                extent: vk::Extent2D { width: 1, height: 1 },
                range: image.range(),
                identity: (0x64 << 40) | i as u64,
                sampled: Some(handle.0.index),
                storage: None,
            }))
        })
        .collect();
    bindings.push(Some(Binding::Image(ImageBinding {
        initial: vk::ImageLayout::UNDEFINED,
        image: out.image,
        view: out.view,
        format,
        extent,
        range: out.range(),
        identity: (0x64 << 40) | 100,
        sampled: None,
        storage: None,
    })));
    bindings.push(Some(Binding::Buffer(BufferBinding {
        buffer: readback.buffer,
        offset: 0,
        size: buffer_desc.size,
        identity: (0x64 << 40) | 101,
        storage: None,
    })));

    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST);
    let sampler_handle = executor.bindless_mut().sampler(&sampler_info).unwrap();
    let sampler = Sampler {
        index: sampler_handle.index,
    };
    let program = FullscreenProgram::new(
        &gpu,
        executor.bindless_mut(),
        "64 images",
        ae_render::spirv!("test_program.frag"),
        32,
        COUNT,
        &[format],
        2,
    )
    .unwrap();
    // tint 0.5 (grey 0.502 becomes 0.753), no offset.
    let uniforms: Vec<u8> = [0.5f32, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut sample = Sample {
        images,
        target,
        buffer,
        program,
        sampler,
        uniforms: uniforms.try_into().unwrap(),
    };
    executor.set_graph(graph).unwrap();
    let token = executor.begin_frame().unwrap();
    executor
        .execute(
            token,
            &mut sample,
            &|resource| bindings.get(resource.index()).copied().flatten(),
            &[],
            &[],
        )
        .unwrap();
    executor.wait(token).unwrap();
    let pixels = readback.mapped().unwrap().to_vec();
    // 128/255 * 0.5 + 128/255 = 0.753, within one UNORM step.
    let want = 128.0 / 255.0 * 1.5;
    let got = pixels[0] as f32 / 255.0;
    assert!((got - want).abs() < 0.01, "pixel {got} vs {want}");
    assert_clean(&gpu);
}

/// A retired handle draws the fallback (the index names the white texel
/// after every slot has flushed) and the registry logs once (M15 F).
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn a_retired_handle_draws_the_fallback_and_logs_once() {
    use ae_framegraph::{LoadOp, ResourceId, Role};
    use ae_render::executor::{PassContext, Recorder};
    use ae_render::fullscreen::{FullscreenProgram, Sampler};

    struct Sample {
        source: ResourceId,
        target: ResourceId,
        buffer: ResourceId,
        program: FullscreenProgram,
        sampler: Sampler,
        stale: u32,
        uniforms: [u8; 32],
    }
    impl Recorder<u32> for Sample {
        fn record(&mut self, pass: &u32, ctx: &PassContext<'_>) -> anyhow::Result<()> {
            match *pass {
                1 => {
                    // The push names the retired index: the set's descriptor
                    // there is the fallback by the time this records.
                    let mut image = ctx.image(self.source)?;
                    image.sampled = Some(self.stale);
                    self.program.write_uniforms(ctx.slot, &self.uniforms);
                    self.program.record(ctx, &[image, image], self.sampler)?;
                }
                2 => ae_render::demo::record_readback(ctx, ctx.image(self.target)?, ctx.buffer(self.buffer)?),
                _ => {}
            }
            Ok(())
        }
    }

    let gpu = gpu();
    let extent = vk::Extent2D { width: 4, height: 4 };
    let format = vk::Format::R8G8B8A8_UNORM;
    let source_desc = ImageDesc::color_2d(format, extent.width, extent.height, vk::ImageUsageFlags::SAMPLED);
    let target_desc = ImageDesc::color_2d(
        format,
        extent.width,
        extent.height,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    let buffer_desc = BufferDesc {
        size: extent.width as u64 * extent.height as u64 * 4,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
    };
    let mut graph = FrameGraph::<u32>::new();
    let source = graph.image("source", source_desc, ResourceClass::External);
    let target = graph.image("target", target_desc, ResourceClass::External);
    let buffer = graph.buffer("readback", buffer_desc, ResourceClass::External);
    graph
        .pass("sample", PassKind::Graphics, 1)
        .sampled(source, ShaderStage::Fragment)
        .color(target, LoadOp::DontCare)
        .id();
    graph
        .pass("copy", PassKind::Transfer, 2)
        .access(target, Role::TransferSrc)
        .access(buffer, Role::TransferDst)
        .id();
    graph
        .pass("host read", PassKind::Transfer, 3)
        .access(buffer, Role::HostRead)
        .root()
        .id();

    let red = [255u8, 0, 0, 255];
    let image = ae_render::upload::image(&gpu, format, 1, 1, &[&red], "red").unwrap();
    let out = OwnedImage::new(&gpu, target_desc, "target").unwrap();
    let mut readback = OwnedBuffer::new(&gpu, buffer_desc, MemoryLocation::GpuToCpu, "readback").unwrap();

    let mut executor = Executor::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    let handle = executor
        .bindless_mut()
        .sampled_2d(image.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .unwrap();
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST);
    let sampler_handle = executor.bindless_mut().sampler(&sampler_info).unwrap();
    let sampler = Sampler {
        index: sampler_handle.index,
    };
    let program = FullscreenProgram::new(
        &gpu,
        executor.bindless_mut(),
        "fallback",
        ae_render::spirv!("test_program.frag"),
        32,
        // The shader samples two images; both are the same index.
        2,
        &[format],
        2,
    )
    .unwrap();
    // tint 0 and no offset: the result is the sampled texel alone.
    let uniforms: Vec<u8> = [0.0f32; 8].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut sample = Sample {
        source,
        target,
        buffer,
        program,
        sampler,
        stale: handle.0.index,
        uniforms: uniforms.try_into().unwrap(),
    };
    // Retired before anything records: every slot's descriptor for the
    // index becomes the white fallback at its next frame.
    executor.bindless_mut().retire_sampled(handle.0);
    let bindings = [
        Some(Binding::Image(ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: image.image,
            view: image.view,
            format,
            extent: vk::Extent2D { width: 1, height: 1 },
            range: image.range(),
            identity: (0xF00 << 40) | 1,
            sampled: Some(handle.0.index),
            storage: None,
        })),
        Some(Binding::Image(ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: out.image,
            view: out.view,
            format,
            extent,
            range: out.range(),
            identity: (0xF00 << 40) | 2,
            sampled: None,
            storage: None,
        })),
        Some(Binding::Buffer(BufferBinding {
            buffer: readback.buffer,
            offset: 0,
            size: buffer_desc.size,
            identity: (0xF00 << 40) | 3,
            storage: None,
        })),
    ];
    executor.set_graph(graph).unwrap();
    // Both slots flush the fallback, so the red texel can go.
    for _ in 0..2 {
        let token = executor.begin_frame().unwrap();
        executor
            .execute(
                token,
                &mut sample,
                &|resource| bindings.get(resource.index()).copied().flatten(),
                &[],
                &[],
            )
            .unwrap();
        executor.wait(token).unwrap();
    }
    let pixels = readback.mapped().unwrap().to_vec();
    assert_eq!(
        &pixels[..4],
        &[255, 255, 255, 255],
        "the white fallback, not the red image"
    );
    // The stale handle resolves to the fallback and is reported once.
    let bindless = executor.bindless_mut();
    assert_eq!(bindless.index_2d(ae_render::bindless::Sampled2D(handle.0)), 0);
    assert_eq!(bindless.index_2d(ae_render::bindless::Sampled2D(handle.0)), 0);
    assert_eq!(bindless.stale_warnings(), 1, "a stale handle logs once");
    assert_clean(&gpu);
}

/// A sampler retired and asked for again while its object is alive is
/// revived under its index and samples with its own settings, not the
/// fallback's (M15): once when no frame applied its first write, and once
/// when both slots had applied the fallback through `flush_open_slot`. The
/// source is red|blue; at the second pixel NEAREST gives pure red, the
/// fallback's LINEAR a quarter of blue.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn a_revived_sampler_keeps_its_settings() {
    use ae_framegraph::{LoadOp, ResourceId, Role};
    use ae_render::executor::{PassContext, Recorder};
    use ae_render::fullscreen::{FullscreenProgram, Sampler};

    struct Sample {
        source: ResourceId,
        target: ResourceId,
        buffer: ResourceId,
        program: FullscreenProgram,
        sampler: Sampler,
        uniforms: [u8; 32],
    }
    impl Recorder<u32> for Sample {
        fn record(&mut self, pass: &u32, ctx: &PassContext<'_>) -> anyhow::Result<()> {
            match *pass {
                1 => {
                    let image = ctx.image(self.source)?;
                    self.program.write_uniforms(ctx.slot, &self.uniforms);
                    self.program.record(ctx, &[image, image], self.sampler)?;
                }
                2 => ae_render::demo::record_readback(ctx, ctx.image(self.target)?, ctx.buffer(self.buffer)?),
                _ => {}
            }
            Ok(())
        }
    }

    let gpu = gpu();
    let extent = vk::Extent2D { width: 4, height: 1 };
    let format = vk::Format::R8G8B8A8_UNORM;
    let source_desc = ImageDesc::color_2d(format, 2, 1, vk::ImageUsageFlags::SAMPLED);
    let target_desc = ImageDesc::color_2d(
        format,
        extent.width,
        extent.height,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    let buffer_desc = BufferDesc {
        size: extent.width as u64 * extent.height as u64 * 4,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
    };
    let mut graph = FrameGraph::<u32>::new();
    let source = graph.image("source", source_desc, ResourceClass::External);
    let target = graph.image("target", target_desc, ResourceClass::External);
    let buffer = graph.buffer("readback", buffer_desc, ResourceClass::External);
    graph
        .pass("sample", PassKind::Graphics, 1)
        .sampled(source, ShaderStage::Fragment)
        .color(target, LoadOp::DontCare)
        .id();
    graph
        .pass("copy", PassKind::Transfer, 2)
        .access(target, Role::TransferSrc)
        .access(buffer, Role::TransferDst)
        .id();
    graph
        .pass("host read", PassKind::Transfer, 3)
        .access(buffer, Role::HostRead)
        .root()
        .id();

    let texels = [255u8, 0, 0, 255, 0, 0, 255, 255];
    let image = ae_render::upload::image(&gpu, format, 2, 1, &[&texels], "red|blue").unwrap();
    let out = OwnedImage::new(&gpu, target_desc, "target").unwrap();
    let mut readback = OwnedBuffer::new(&gpu, buffer_desc, MemoryLocation::GpuToCpu, "readback").unwrap();

    let mut executor = Executor::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    let handle = executor
        .bindless_mut()
        .sampled_2d(image.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .unwrap();
    let nearest = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    // Retired before any frame applied its write, then asked for again.
    let first = executor.bindless_mut().sampler(&nearest).unwrap();
    executor.bindless_mut().retire_sampler(first);
    let revived = executor.bindless_mut().sampler(&nearest).unwrap();
    assert_eq!(revived.index, first.index, "the retiring index is revived");
    let program = FullscreenProgram::new(
        &gpu,
        executor.bindless_mut(),
        "revive",
        ae_render::spirv!("test_program.frag"),
        32,
        2,
        &[format],
        2,
    )
    .unwrap();
    // tint 0 and no offset: the result is the sampled texel alone.
    let uniforms: Vec<u8> = [0.0f32; 8].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut sample = Sample {
        source,
        target,
        buffer,
        program,
        sampler: Sampler { index: revived.index },
        uniforms: uniforms.try_into().unwrap(),
    };
    let bindings = [
        Some(Binding::Image(ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: image.image,
            view: image.view,
            format,
            extent: vk::Extent2D { width: 2, height: 1 },
            range: image.range(),
            identity: (0xF01 << 40) | 1,
            sampled: Some(handle.0.index),
            storage: None,
        })),
        Some(Binding::Image(ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: out.image,
            view: out.view,
            format,
            extent,
            range: out.range(),
            identity: (0xF01 << 40) | 2,
            sampled: None,
            storage: None,
        })),
        Some(Binding::Buffer(BufferBinding {
            buffer: readback.buffer,
            offset: 0,
            size: buffer_desc.size,
            identity: (0xF01 << 40) | 3,
            storage: None,
        })),
    ];
    executor.set_graph(graph).unwrap();
    let mut frames = |executor: &mut Executor<u32>, sample: &mut Sample| {
        // One frame per slot, so both slots' sets are checked.
        for _ in 0..2 {
            let token = executor.begin_frame().unwrap();
            executor
                .execute(
                    token,
                    sample,
                    &|resource| bindings.get(resource.index()).copied().flatten(),
                    &[],
                    &[],
                )
                .unwrap();
            executor.wait(token).unwrap();
            let pixels = readback.mapped().unwrap().to_vec();
            assert_eq!(&pixels[4..8], &[255, 0, 0, 255], "NEAREST, not the fallback's LINEAR");
        }
    };
    frames(&mut executor, &mut sample);
    // Retired again, and both slots apply the fallback before it is asked
    // for again (flush_open_slot does not mark the retirement).
    executor.bindless_mut().retire_sampler(revived);
    for slot in 0..2 {
        executor.bindless_mut().flush_open_slot(&gpu.device, slot);
    }
    let again = executor.bindless_mut().sampler(&nearest).unwrap();
    assert_eq!(again.index, revived.index, "the retiring index is revived again");
    frames(&mut executor, &mut sample);
    assert_clean(&gpu);
}

/// GPU-assisted validation (`AE_VK_GPU_AV=1`): a program that samples a
/// descriptor the set never bound is reported, and the frames before it are
/// clean. Skipped when the layer has no GPU-AV.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU and a validation layer with GPU-AV"]
fn gpu_assisted_validation_reports_an_unbound_handle() {
    use ae_framegraph::{LoadOp, ResourceId, Role};
    use ae_render::executor::{PassContext, Recorder};
    use ae_render::fullscreen::{FullscreenProgram, Sampler};

    struct Sample {
        source: ResourceId,
        target: ResourceId,
        buffer: ResourceId,
        program: FullscreenProgram,
        sampler: Sampler,
        index: u32,
    }
    impl Recorder<u32> for Sample {
        fn record(&mut self, pass: &u32, ctx: &PassContext<'_>) -> anyhow::Result<()> {
            match *pass {
                1 => {
                    let mut image = ctx.image(self.source)?;
                    image.sampled = Some(self.index);
                    self.program.record(ctx, &[image], self.sampler)?;
                }
                2 => ae_render::demo::record_readback(ctx, ctx.image(self.target)?, ctx.buffer(self.buffer)?),
                _ => {}
            }
            Ok(())
        }
    }

    // The layer reads this when the instance is made; the test owns the
    // process's device for its length (the GPU tests run one thread).
    // Safety: this test owns the process's environment for its length (the
    // GPU tests run one thread at a time); the previous values come back
    // before it returns.
    let previous = (
        std::env::var("AE_VK_GPU_AV").ok(),
        std::env::var("AE_VK_SYNC_VALIDATION").ok(),
    );
    let restore = |previous: &(Option<String>, Option<String>)| unsafe {
        match &previous.0 {
            Some(v) => std::env::set_var("AE_VK_GPU_AV", v),
            None => std::env::remove_var("AE_VK_GPU_AV"),
        }
        match &previous.1 {
            Some(v) => std::env::set_var("AE_VK_SYNC_VALIDATION", v),
            None => std::env::remove_var("AE_VK_SYNC_VALIDATION"),
        }
    };
    unsafe {
        std::env::set_var("AE_VK_GPU_AV", "1");
        std::env::set_var("AE_VK_SYNC_VALIDATION", "0");
    }
    let instance = Arc::new(GpuInstance::new(&[], Some(true)).expect("Vulkan instance"));
    if !instance.gpu_assisted {
        eprintln!("GPU-assisted validation is not available; skipping");
        restore(&previous);
        return;
    }
    let gpu = Arc::new(GpuDevice::new(instance, None).expect("Vulkan device"));

    let extent = vk::Extent2D { width: 4, height: 4 };
    let format = vk::Format::R8G8B8A8_UNORM;
    let source_desc = ImageDesc::color_2d(format, extent.width, extent.height, vk::ImageUsageFlags::SAMPLED);
    let target_desc = ImageDesc::color_2d(
        format,
        extent.width,
        extent.height,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    let buffer_desc = BufferDesc {
        size: extent.width as u64 * extent.height as u64 * 4,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
    };
    let mut graph = FrameGraph::<u32>::new();
    let source = graph.image("source", source_desc, ResourceClass::External);
    let target = graph.image("target", target_desc, ResourceClass::External);
    let buffer = graph.buffer("readback", buffer_desc, ResourceClass::External);
    graph
        .pass("sample", PassKind::Graphics, 1)
        .sampled(source, ShaderStage::Fragment)
        .color(target, LoadOp::DontCare)
        .id();
    graph
        .pass("copy", PassKind::Transfer, 2)
        .access(target, Role::TransferSrc)
        .access(buffer, Role::TransferDst)
        .id();
    graph
        .pass("host read", PassKind::Transfer, 3)
        .access(buffer, Role::HostRead)
        .root()
        .id();

    let white = [255u8; 4];
    let image = ae_render::upload::image(&gpu, format, 1, 1, &[&white], "source").unwrap();
    let out = OwnedImage::new(&gpu, target_desc, "target").unwrap();
    let readback = OwnedBuffer::new(&gpu, buffer_desc, MemoryLocation::GpuToCpu, "readback").unwrap();

    let mut executor = Executor::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    let handle = executor
        .bindless_mut()
        .sampled_2d(image.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .unwrap();
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST);
    let sampler_handle = executor.bindless_mut().sampler(&sampler_info).unwrap();
    let sampler = Sampler {
        index: sampler_handle.index,
    };
    let program = FullscreenProgram::new(
        &gpu,
        executor.bindless_mut(),
        "out of bounds",
        ae_render::spirv!("test_oob.frag"),
        0,
        1,
        &[format],
        2,
    )
    .unwrap();
    let bindings = [
        Some(Binding::Image(ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: image.image,
            view: image.view,
            format,
            extent: vk::Extent2D { width: 1, height: 1 },
            range: image.range(),
            identity: (0x0B0 << 40) | 1,
            sampled: Some(handle.0.index),
            storage: None,
        })),
        Some(Binding::Image(ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: out.image,
            view: out.view,
            format,
            extent,
            range: out.range(),
            identity: (0x0B0 << 40) | 2,
            sampled: None,
            storage: None,
        })),
        Some(Binding::Buffer(BufferBinding {
            buffer: readback.buffer,
            offset: 0,
            size: buffer_desc.size,
            identity: (0x0B0 << 40) | 3,
            storage: None,
        })),
    ];
    let mut sample = Sample {
        source,
        target,
        buffer,
        program,
        sampler,
        index: handle.0.index,
    };
    executor.set_graph(graph).unwrap();
    // A bound index first: the layer reports nothing.
    for _ in 0..2 {
        let token = executor.begin_frame().unwrap();
        executor
            .execute(
                token,
                &mut sample,
                &|resource| bindings.get(resource.index()).copied().flatten(),
                &[],
                &[],
            )
            .unwrap();
        executor.wait(token).unwrap();
    }
    assert_eq!(gpu.instance.counters.errors(), 0, "a bound handle is clean");
    // Now an index nothing wrote: GPU-AV reports the read.
    sample.index = 63;
    let token = executor.begin_frame().unwrap();
    executor
        .execute(
            token,
            &mut sample,
            &|resource| bindings.get(resource.index()).copied().flatten(),
            &[],
            &[],
        )
        .unwrap();
    executor.wait(token).unwrap();
    let errors = gpu.instance.counters.errors();
    assert!(errors > 0, "GPU-assisted validation reported nothing");
    restore(&previous);
}

// ---- M16.1 baseline gates (docs/PLAN.md, M16) -------------------------------
//
// Before any M16 shader or pass changes, these pin what "unchanged" means:
// one frame hash per pre-M16 fixture, keyed by device and driver in
// `tests/data/m16_baseline.txt`. The frame identity is the binding gate
// (the advisory SPIR-V gate is `tests/spirv.rs`). A hash changes only in a
// commit whose message gives the reason; M16.7's "blips and ghosts cast no
// shadow" is the planned one, and it re-pins `observer` alone.
//
// On an unknown device the tests print the hashes they measured and skip:
// frames compare only on the same GPU and driver.

use std::collections::BTreeMap;
use std::sync::OnceLock;

/// The pinned hashes, as committed.
const M16_PINNED: &str = include_str!("data/m16_baseline.txt");

/// The baseline fixtures' extent (M16.1: 320x180).
const M16_EXTENT: vk::Extent2D = vk::Extent2D {
    width: 320,
    height: 180,
};

/// Frames drawn per fixture: at 1/60 s each, the last is a fixed animation
/// time (8/60 s) with the water, decals and particles all moving.
const M16_FRAMES: usize = 8;

/// `device | driver`, the key `tests/data/m16_baseline.txt` sections use.
/// The driver is `VkPhysicalDeviceProperties::driverVersion` as the driver
/// reports it (hexadecimal: vendor-specific encodings decode differently).
fn m16_device_key(gpu: &GpuDevice) -> String {
    let properties = unsafe { gpu.instance.instance.get_physical_device_properties(gpu.physical) };
    format!("{} | {:#010x}", gpu.device_name, properties.driver_version)
}

/// The pinned hashes: device key -> fixture -> hash.
fn m16_pinned() -> &'static BTreeMap<String, BTreeMap<String, u64>> {
    static PARSED: OnceLock<BTreeMap<String, BTreeMap<String, u64>>> = OnceLock::new();
    PARSED.get_or_init(|| {
        let mut out: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
        let mut key = String::new();
        for line in M16_PINNED.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("# ") {
                key = rest.trim().to_string();
                continue;
            }
            if line.starts_with('#') {
                continue;
            }
            let (name, hash) = line
                .split_once(' ')
                .unwrap_or_else(|| panic!("m16_baseline.txt: line without a hash: {line:?}"));
            assert!(!key.is_empty(), "m16_baseline.txt: {name} before a device line");
            let hash = u64::from_str_radix(hash, 16).unwrap_or_else(|_| panic!("bad hash {hash:?}"));
            out.entry(key.clone()).or_default().insert(name.to_string(), hash);
        }
        out
    })
}

/// One pre-M16 fixture's frame: [`M16_EXTENT`], a fixed camera and time on
/// the crossing maps (the land crossing when `wet` is false, the water
/// crossing otherwise), their units, and, with `observer`, the observer's
/// looks (own, enemy, blip and selection/ghost) so a blip's shadow is in
/// frame.
fn m16_frame(gpu: &Arc<GpuDevice>, wet: bool, observer: bool, surface: bool) -> Vec<u8> {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource, VisionFeed, VisionFrame};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::{TerrainInfo, VisionGrid};
    use ae_render::thread::{TargetInfo, run_headless};
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};

    let terrain = if wet {
        ae_nav::scenario::water_crossing_terrain(512)
    } else {
        ae_nav::scenario::land_crossing_terrain(512)
    };
    let cells = (terrain.geom.cells_x as usize, terrain.geom.cells_y as usize);
    let (heightmap, scale, offset) =
        Heightmap::from_cells(cells, |x, y| terrain.ground_height(y * cells.0 + x) as f32 * 0.1, None);
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let level = terrain.has_water().then_some(terrain.water_level as f32 * 0.1);
    let info = TerrainInfo::new(&heightmap, &pyramid, terrain.geom.cell_size, scale, offset, level);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: Some(terrain.material.clone()),
    });
    let side = terrain.geom.world_width();
    // The water crossing's big lake; the land crossing's middle.
    let target = if wet {
        glam::Vec2::new(side * 0.52, side * 0.63)
    } else {
        glam::Vec2::new(side * 0.5, side * 0.5)
    };
    // The looks: three plain classes, or the observer's own, enemy, blip
    // and selection classes (the client's order and count for two real
    // classes).
    let colors: Vec<[f32; 3]> = if observer {
        vec![
            [0.95, 0.75, 0.10],
            [0.80, 0.12, 0.08],
            [0.88, 0.55, 0.14],
            [0.55, 0.20, 0.10],
            [0.6, 0.6, 0.6],
            [1.0, 1.0, 1.0],
        ]
    } else {
        vec![[1.0, 0.8, 0.1], [0.9, 0.9, 0.95], [0.8, 0.1, 0.1]]
    };
    let classes: Vec<UnitClass> = colors
        .iter()
        .enumerate()
        .map(|(i, &color)| UnitClass {
            gamut: ae_render::colorspace::Gamut::Bt709,
            color,
            floats: i % 2 == 1,
        })
        .collect();
    let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
    for i in 0..400u32 {
        let (x, y) = (
            target.x - 665.0 + (i % 20) as f32 * 70.0,
            target.y - 665.0 + (i / 20) as f32 * 70.0,
        );
        let class = i % classes.len() as u32;
        for (a, word) in [
            x.to_bits(),
            y.to_bits(),
            x.to_bits(),
            y.to_bits(),
            0.0f32.to_bits(),
            (-0.5f32 / (1.25f32).sqrt()).to_bits(),
            0.0f32.to_bits(),
            (1.0f32 / (1.25f32).sqrt()).to_bits(),
            0.0f32.to_bits(),
            (-0.5f32 / (1.25f32).sqrt()).to_bits(),
            0.0f32.to_bits(),
            (1.0f32 / (1.25f32).sqrt()).to_bits(),
            6.0f32.to_bits(),
            class,
        ]
        .into_iter()
        .enumerate()
        {
            columns[a].push(word);
        }
        columns[ATTR_VIEW].push(match (observer, class) {
            (true, 2 | 3) => 1, // Seen.
            (true, 4) => 2,     // Blip.
            _ => 0,             // Own or unfiltered.
        });
    }
    let feed = Arc::new(UnitFeed::default());
    feed.publish(Arc::new(InstanceStream {
        tick: 1,
        published: std::time::Instant::now(),
        tick_seconds: 0.0,
        sim_seconds: 0.0,
        columns: at_rest(columns),
        anim: AnimWrites::default(),
    }));
    let mut app = TerrainApp::new(source, TerrainOutput::Readback)
        .with_water()
        .with_shadows()
        .with_units(UnitsSetup {
            feed,
            capacity: 512,
            classes,
            min_pixels: 5.0,
            models: Vec::new(),
            rigs: Vec::new(),
        });
    if observer {
        // The observer's vision, as the client attaches it: a fixed feed of
        // line of sight and explored cells, so the water's fog-of-war branch
        // is in frame (the lake straddles the sight boundary and the river
        // beyond it is never seen). A regression that draws every water
        // pixel as seen changes the frame.
        let grid = VisionGrid {
            width: 32,
            height: 32,
            cell_size: side / 32.0,
        };
        let (stride, words) = (grid.stride(), grid.words());
        let (mut los, mut explored) = (vec![0u64; words], vec![0u64; words]);
        for row in 0..grid.height as usize {
            for col in 0..grid.width as usize {
                let bit = 1u64 << (col % 64);
                if col < 17 {
                    los[row * stride + col / 64] |= bit;
                }
                if col < 19 {
                    explored[row * stride + col / 64] |= bit;
                }
            }
        }
        let feed = Arc::new(VisionFeed::new(grid));
        feed.publish(Arc::new(VisionFrame { tick: 1, los, explored }));
        app = app.with_vision(feed);
    }
    {
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera.target = target;
        view.camera.pitch = 0.6;
        view.camera.distance = 1600.0;
        // A fixed heading, so the frame does not depend on the default.
        view.camera.yaw = 0.35;
        view.water = surface;
    }
    app.create_offscreen(gpu, M16_EXTENT).unwrap();
    let target = TargetInfo {
        extent: M16_EXTENT,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
    run_headless(gpu, &mut app, target, M16_FRAMES, config).unwrap();
    let pixels = app.readback().unwrap();
    assert_clean(gpu);
    ae_render::thread::RenderApp::destroy(&mut app, gpu);
    pixels
}

/// Render `name`'s fixture and compare its frame hash with the pinned one,
/// or print the hashes and skip on an unknown device.
fn m16_check(gpu: &Arc<GpuDevice>, name: &str, water: bool, observer: bool, surface: bool) {
    let key = m16_device_key(gpu);
    let pinned = m16_pinned();
    let pixels = m16_frame(gpu, water, observer, surface);
    let hash = fnv(&pixels);
    let Some(section) = pinned.get(&key) else {
        eprintln!(
            "M16 baseline skipped on an unknown device; paste into tests/data/m16_baseline.txt:\n\
             # {key}\ndry {dry:016x}\nwet {wet:016x}\nobserver {observer:016x}\noff {off:016x}",
            dry = fnv(&m16_frame(gpu, false, false, true)),
            wet = fnv(&m16_frame(gpu, true, false, true)),
            observer = fnv(&m16_frame(gpu, true, true, true)),
            off = fnv(&m16_frame(gpu, true, false, false)),
        );
        return;
    };
    let Some(&want) = section.get(name) else {
        panic!("tests/data/m16_baseline.txt has no {name} for {key}");
    };
    assert_eq!(
        hash, want,
        "the {name} frame changed ({hash:016x}, pinned {want:016x}). Frames change only in a \
         commit whose message gives the reason, which also re-pins this hash."
    );
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn m16_baseline_dry_frame_is_unchanged() {
    m16_check(&gpu(), "dry", false, false, true);
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn m16_baseline_wet_frame_is_unchanged() {
    m16_check(&gpu(), "wet", true, false, true);
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn m16_baseline_observer_frame_is_unchanged() {
    m16_check(&gpu(), "observer", true, true, true);
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn m16_baseline_water_off_frame_is_unchanged() {
    m16_check(&gpu(), "off", true, false, false);
}

/// The M16.2 coast fixture renders (criterion 4): the map the client's
/// `--map coast` builds, at 512 cells, from the navigation crate's own
/// generator, with its water surface. The seaward half reads blue under the
/// water and the landward half green, and the surface changes the frame.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn coast_fixture_renders() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{TargetInfo, run_headless};
    use glam::Vec2;

    let gpu = gpu();
    let terrain = ae_nav::scenario::coast_terrain(512);
    let cells = (terrain.geom.cells_x as usize, terrain.geom.cells_y as usize);
    let (heightmap, scale, offset) =
        Heightmap::from_cells(cells, |x, y| terrain.ground_height(y * cells.0 + x) as f32 * 0.1, None);
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let water = Some(terrain.water_level as f32 * 0.1);
    let info = TerrainInfo::new(&heightmap, &pyramid, terrain.geom.cell_size, scale, offset, water);
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: Some(terrain.material.clone()),
    });
    let extent = vk::Extent2D {
        width: 320,
        height: 240,
    };
    let side = terrain.geom.world_width();
    let render = |surface: bool| -> Vec<u8> {
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_water();
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.camera.target = Vec2::new(side * 0.5, side * 0.5);
            view.camera.distance = 2400.0;
            view.camera.pitch = 0.6;
            view.fog = false;
            view.water = surface;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let target = TargetInfo {
            extent,
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let config = ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        };
        run_headless(&gpu, &mut app, target, 8, config).unwrap();
        let pixels = app.readback().unwrap();
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let wet = render(true);
    let dry = render(false);
    let differing = wet.iter().zip(&dry).filter(|(a, b)| a != b).count();
    assert!(
        differing > wet.len() / 10,
        "the water covers the shelf: {differing} of {} bytes differ",
        wet.len()
    );
    // The bottom third avoids the sky; the left is land, the right is sea.
    let mean = |pixels: &[u8], x0: u32, x1: u32| -> [f64; 3] {
        let mut sum = [0.0; 3];
        let mut n = 0.0;
        for y in (extent.height * 2 / 3)..extent.height {
            for x in x0..x1 {
                let i = ((y * extent.width + x) * 4) as usize;
                for c in 0..3 {
                    sum[c] += pixels[i + c] as f64;
                }
                n += 1.0;
            }
        }
        sum.map(|v| v / n)
    };
    let sea = mean(&wet, extent.width * 2 / 3, extent.width);
    let land = mean(&wet, 0, extent.width / 3);
    assert!(sea[2] > sea[1], "the sea reads blue: {sea:?}");
    assert!(land[1] > land[2], "the land reads green: {land:?}");
    assert_clean(&gpu);
}

// ---- M16.5: the shore field and the caustic web -----------------------------

/// A ground height texture for `terrain` in metres (one texel per corner),
/// and its values.
fn shore_ground(gpu: &Arc<GpuDevice>, terrain: &ae_nav::terrain::Terrain) -> (OwnedImage, Vec<f32>, u32, u32) {
    use ae_render::terrain::heightmap::Heightmap;
    let cells = (terrain.geom.cells_x as usize, terrain.geom.cells_y as usize);
    let ground = |x: usize, y: usize| terrain.ground_height(y * cells.0 + x) as f32 * 0.1;
    let (hm, _, _) = Heightmap::from_cells(cells, ground, Some((1.0, 0.0)));
    let bytes: Vec<u8> = hm.data.iter().flat_map(|h| h.to_le_bytes()).collect();
    let image = ae_render::upload::image(
        gpu,
        vk::Format::R32_SFLOAT,
        hm.width as u32,
        hm.height as u32,
        &[&bytes],
        "shore test ground",
    )
    .expect("the ground texture");
    (image, hm.data, hm.width as u32, hm.height as u32)
}

/// The shore pass's recorder: the water data and the ground texture's
/// bindless handles.
struct ShoreRecorder {
    /// `None` only once the test has taken it to destroy its objects.
    water: Option<ae_render::water::WaterRenderer>,
    ground: u32,
    sampler: u32,
}

impl ae_render::executor::Recorder<ae_render::library::LibraryPass> for ShoreRecorder {
    fn skip(&self, pass: &ae_render::library::LibraryPass) -> bool {
        matches!(pass, ae_render::library::LibraryPass::WaterShore)
            && self.water.as_ref().is_none_or(|w| !w.shore_dirty())
    }

    fn record(
        &mut self,
        pass: &ae_render::library::LibraryPass,
        ctx: &ae_render::executor::PassContext<'_>,
    ) -> anyhow::Result<()> {
        match pass {
            ae_render::library::LibraryPass::WaterShore => {
                self.water
                    .as_ref()
                    .expect("the water data")
                    .record_shore(ctx, self.ground, self.sampler)
            }
            other => anyhow::bail!("the shore test's graph has {other:?}"),
        }
    }
}

/// A one-pass graph writing `water.shore`, with the library's checks.
fn shore_graph(env: &ae_render::library::LibraryEnv, desc: ImageDesc) -> FrameGraph<ae_render::library::LibraryPass> {
    use ae_framegraph::Role;
    use ae_render::library::{LibraryPass, LibraryResource};
    let mut graph = FrameGraph::<LibraryPass>::new();
    let shore = graph.image(LibraryResource::WaterShore.name(), desc, ResourceClass::External);
    graph
        .pass(
            LibraryPass::WaterShore.name(),
            PassKind::Compute,
            LibraryPass::WaterShore,
        )
        .access(shore, Role::StorageImageWrite(ShaderStage::Compute))
        .conditional()
        .root();
    let map = ae_render::library::resolve(&mut graph, env, |_| None).expect("the library resolves");
    for (index, pass) in graph.passes.iter().enumerate() {
        if pass.record == LibraryPass::WaterShore {
            ae_render::library::check(&graph, index, LibraryPass::WaterShore, &map, env)
                .expect("the shore pass is wired as the library needs");
        }
    }
    graph
}

/// The environment for a water-only library graph at `(w, h)` ground samples.
fn shore_env((w, h): (u32, u32)) -> ae_render::library::LibraryEnv {
    ae_render::library::LibraryEnv {
        color_format: vk::Format::R8G8B8A8_SRGB,
        scene_format: vk::Format::R16G16B16A16_SFLOAT,
        depth_format: vk::Format::D32_SFLOAT,
        extent: vk::Extent2D {
            width: 320,
            height: 240,
        },
        units: false,
        water: false,
        shadows: false,
        particles: false,
        ground: Some((w, h)),
        bake_water: false,
        encoding: ae_render::encode::OutputEncoding::Sdr,
        levels: ae_render::encode::DisplayLevels::default(),
        working: Default::default(),
        gamut: Default::default(),
        reflection_scale: 2,
    }
}

/// M16.5: the shore travel-time field on the coast fixture. Zero at the
/// waterline, rising seaward (monotone along a probe line), a real time in
/// deep water, and a terrain edit changes it only around the edit (the
/// pass is skipped while the ground is unchanged).
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn the_shore_field_starts_at_the_waterline_and_rises_seaward() {
    use ae_framegraph::ResourceId;
    use ae_render::library::LibraryPass;

    let gpu = gpu();
    let terrain = ae_nav::scenario::coast_terrain(256);
    let (mut ground_img, heights, w, h) = shore_ground(&gpu, &terrain);
    let level = terrain.water_level as f32 * 0.1;
    let mut executor = Executor::<LibraryPass>::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    let water = ae_render::water::WaterRenderer::new(
        &gpu,
        executor.bindless_mut(),
        w,
        h,
        level,
        terrain.geom.cell_size,
        true,
        false,
    )
    .expect("the water data");
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    let sampler = executor.bindless_mut().sampler(&sampler_info).unwrap();
    let ground_index = executor
        .bindless_mut()
        .sampled_2d(ground_img.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .unwrap()
        .0
        .index;
    let mut recorder = ShoreRecorder {
        water: Some(water),
        ground: ground_index,
        sampler: sampler.index,
    };
    let env = shore_env((w, h));
    let graph = shore_graph(&env, recorder.water.as_ref().unwrap().shore().unwrap().desc());
    let shore = ResourceId(
        graph
            .resources
            .iter()
            .position(|r| r.name == "water.shore")
            .expect("the graph declares water.shore") as u32,
    );
    executor.set_graph(graph).unwrap();
    let binding = Binding::Image(recorder.water.as_ref().unwrap().shore_binding().unwrap());
    let run = |recorder: &mut ShoreRecorder, executor: &mut Executor<LibraryPass>| {
        let token = executor.begin_frame().unwrap();
        executor
            .execute(
                token,
                recorder,
                &|resource| (resource == shore).then_some(binding),
                &[],
                &[],
            )
            .unwrap();
        executor.wait(token).unwrap();
    };
    run(&mut recorder, &mut executor);
    let field = recorder
        .water
        .as_ref()
        .unwrap()
        .shore()
        .unwrap()
        .read(&gpu)
        .expect("the field reads back");
    assert_eq!(field.len(), (w * h) as usize);
    let at = |x: u32, y: u32| field[(y * w + x) as usize];

    // The probe: a row south of the river's mouth (so the first water is the
    // sea, not the river channel), from the first water texel seaward.
    let row = (h as f32 * 0.9) as u32;
    let first_water = (1..w)
        .find(|&x| heights[(row * w + x) as usize] < level)
        .expect("the coast has water on the middle row");
    assert!(
        heights[(row * w + first_water - 1) as usize] >= level,
        "the probe starts at the waterline"
    );
    assert!(
        at(first_water, row) < 1e-3,
        "zero at the waterline: {}",
        at(first_water, row)
    );
    let mut last = 0.0f32;
    let mut far = 0.0f32;
    for x in first_water..(first_water + 60).min(w) {
        let t = at(x, row);
        assert!(
            t + 1e-3 >= last,
            "the travel time rises seaward along the probe: {last} then {t} at x {x}"
        );
        assert!(t.is_finite() && t < 60000.0);
        last = t;
        far = t;
    }
    assert!(far > 1.0, "a minute seaward the swell takes a while: {far}");
    // Deep water: tens of seconds. Land: zero (nothing to travel).
    let deep_x = (w as f32 * 0.9) as u32;
    assert!(at(deep_x, row) > 20.0, "deep water: {}", at(deep_x, row));
    let land_x = (w as f32 * 0.1) as u32;
    assert!(heights[(row * w + land_x) as usize] >= level);
    assert_eq!(at(land_x, row), 0.0);

    // A ground edit (raising a small island in the river's narrow channel)
    // changes the field only around it: the channel's banks are the nearest
    // shore on both sides, so no far texel's nearest shore moves. The pass
    // runs again only because the edit marked it dirty.
    let river_row = (h as f32 * 0.4) as u32;
    let river_x = w / 4; // the channel runs near a quarter of the width
    let (x0, y0, side) = (river_x, river_row, 4u32);
    let raised: Vec<u8> = (0..side * side).flat_map(|_| (level + 20.0).to_le_bytes()).collect();
    ae_render::upload::update_images(
        &gpu,
        &[(
            &ground_img,
            &[ae_render::upload::Region {
                mip: 0,
                x: x0,
                y: y0,
                width: side,
                height: side,
                bytes: raised,
            }],
        )],
    )
    .expect("the edit lands");
    recorder
        .water
        .as_ref()
        .unwrap()
        .mark_dirty(Some((x0, y0, x0 + side, y0 + side)));
    assert_eq!(
        recorder.water.as_ref().unwrap().shore().unwrap().last_rect(),
        Some((x0, y0, x0 + side, y0 + side))
    );
    run(&mut recorder, &mut executor);
    let edited = recorder
        .water
        .as_ref()
        .unwrap()
        .shore()
        .unwrap()
        .read(&gpu)
        .expect("the field reads back");
    let mut changed = 0;
    let radius = 24u32;
    for y in 0..h {
        for x in 0..w {
            let (a, b) = (at(x, y), edited[(y * w + x) as usize]);
            if (a - b).abs() > 1e-3 {
                changed += 1;
                assert!(
                    x + radius >= x0 && x <= x0 + side + radius && y + radius >= y0 && y <= y0 + side + radius,
                    "the field changed at ({x}, {y}), outside the edit rect plus {radius} cells"
                );
            }
        }
    }
    assert!(changed > 0, "raising an island changes the field around it");
    let new_at = |x: u32, y: u32| edited[(y * w + x) as usize];
    assert_eq!(new_at(x0 + 1, y0 + 1), 0.0, "the island itself is dry");

    // The recorder's skip: clean now, so the next frame records nothing.
    assert!(!recorder.water.as_ref().unwrap().shore_dirty());
    let mut water = recorder.water.take().expect("the water data");
    water.retire(executor.bindless_mut());
    water.destroy(&gpu);
    drop(recorder);
    drop(executor);
    ground_img.destroy(&gpu);
    assert_clean(&gpu);
}

/// M16.5: the seed coordinates are exact at a wide map's edge. A half-float
/// centre cannot represent x >= 1024 texels, so a waterline at x 2048 used
/// to start at 1.06 s; the integer seed image keeps the first wet texel at
/// exactly zero and the adjacent step at one texel over the break-depth
/// speed (the default map is 2560 cells, archives reach 8192).
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn the_shore_field_is_exact_on_a_wide_map() {
    use ae_render::library::LibraryPass;

    let gpu = gpu();
    let (w, h) = (2561u32, 8u32);
    let coast = 2048u32;
    let level = 0.5f32;
    let cell = 4.0f32;
    let bytes: Vec<u8> = (0..h)
        .flat_map(|_| (0..w).flat_map(|x| (if x < coast { 1.0f32 } else { 0.0 }).to_le_bytes()))
        .collect();
    let mut ground = ae_render::upload::image(&gpu, vk::Format::R32_SFLOAT, w, h, &[&bytes], "wide map ground")
        .expect("the ground texture");
    let mut executor = Executor::<LibraryPass>::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    let water = ae_render::water::WaterRenderer::new(&gpu, executor.bindless_mut(), w, h, level, cell, true, false)
        .expect("the water data");
    let sampler = executor
        .bindless_mut()
        .sampler(&vk::SamplerCreateInfo::default())
        .unwrap();
    let ground_index = executor
        .bindless_mut()
        .sampled_2d(ground.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .unwrap()
        .0
        .index;
    let mut recorder = ShoreRecorder {
        water: Some(water),
        ground: ground_index,
        sampler: sampler.index,
    };
    let graph = shore_graph(
        &shore_env((w, h)),
        recorder.water.as_ref().unwrap().shore().unwrap().desc(),
    );
    let shore = ae_framegraph::ResourceId(
        graph
            .resources
            .iter()
            .position(|r| r.name == "water.shore")
            .expect("the graph declares water.shore") as u32,
    );
    executor.set_graph(graph).unwrap();
    let binding = Binding::Image(recorder.water.as_ref().unwrap().shore_binding().unwrap());
    let token = executor.begin_frame().unwrap();
    executor
        .execute(
            token,
            &mut recorder,
            &|resource| (resource == shore).then_some(binding),
            &[],
            &[],
        )
        .unwrap();
    executor.wait(token).unwrap();
    let field = recorder
        .water
        .as_ref()
        .unwrap()
        .shore()
        .unwrap()
        .read(&gpu)
        .expect("the field reads back");
    let row = 3u32;
    let at = |x: u32| field[(row * w + x) as usize];
    let step = cell / (0.6 * (9.81 * 1.0f32).sqrt());
    assert_eq!(at(coast - 1), 0.0, "land before the waterline");
    assert_eq!(at(coast), 0.0, "the first wet texel is exactly zero");
    assert!(
        (at(coast + 1) - step).abs() < 0.01,
        "one texel seaward is {step} s: {}",
        at(coast + 1)
    );
    assert!(
        (at(coast + 2) - 2.0 * step).abs() < 0.01,
        "two texels: {}",
        at(coast + 2)
    );
    let mut water = recorder.water.take().expect("the water data");
    water.retire(executor.bindless_mut());
    water.destroy(&gpu);
    drop(recorder);
    drop(executor);
    ground.destroy(&gpu);
    assert_clean(&gpu);
}

/// M16.5: the caustic web is the deterministic tileable pattern, uploaded
/// with its full GPU mip chain (256² down to 1).
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn the_caustic_web_is_tileable_and_mipped() {
    let gpu = gpu();
    let mut executor = Executor::<()>::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    let mut water = ae_render::water::WaterRenderer::new(&gpu, executor.bindless_mut(), 8, 8, 0.5, 8.0, false, true)
        .expect("the water data");
    let mips = water.caustics().unwrap().read(&gpu).expect("the web reads back");
    assert_eq!(mips.len(), 9, "256² has nine mips");
    assert_eq!(
        mips[0],
        ae_render::water::caustics::base(ae_render::water::caustics::SIZE)
    );
    assert_eq!(mips[1].len(), (128 * 128 * 4) as usize);
    assert_eq!(mips[8].len(), 4, "the last mip is one texel");
    assert!(mips[0].chunks(4).any(|t| t[0] > 128), "the web has bright threads");
    assert!(mips[0].chunks(4).any(|t| t[0] < 32), "and dark cells");
    water.retire(executor.bindless_mut());
    water.destroy(&gpu);
    drop(executor);
    assert_clean(&gpu);
}

// ---- M16.5 review: the bake path, allocation gating and cleanup ------------

/// The coast terrain as a render source, with `water` from `terrain`.
fn coast_source(terrain: &ae_nav::terrain::Terrain) -> (Arc<ae_render::terrain::app::TerrainSource>, Option<f32>) {
    use ae_render::terrain::app::TerrainSource;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    let cells = (terrain.geom.cells_x as usize, terrain.geom.cells_y as usize);
    let ground = |x: usize, y: usize| terrain.ground_height(y * cells.0 + x) as f32 * 0.1;
    let (heightmap, scale, offset) = Heightmap::from_cells(cells, ground, None);
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let water = (terrain.water_level > ae_nav::terrain::NO_WATER).then_some(terrain.water_level as f32 * 0.1);
    let info = TerrainInfo::new(&heightmap, &pyramid, terrain.geom.cell_size, scale, offset, water);
    (
        Arc::new(TerrainSource {
            heightmap,
            pyramid,
            info,
            materials: Some(terrain.material.clone()),
        }),
        water,
    )
}

/// M16.5 review: the app's own graph declares the water data, bakes it on
/// the first frame, skips it while the ground is unchanged, and bakes again
/// after a terrain edit lands through the real `TerrainApp` feed.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn the_app_bakes_the_water_data_and_refreshes_it_after_an_edit() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput};
    use ae_render::thread::{Headless, TargetInfo};

    let gpu = gpu();
    let terrain = ae_nav::scenario::coast_terrain(128);
    let (source, water) = coast_source(&terrain);
    assert!(water.is_some());
    let extent = vk::Extent2D {
        width: 320,
        height: 180,
    };
    let mut app = TerrainApp::new(source, TerrainOutput::Readback)
        .with_water()
        .with_water_data();
    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
    let mut headless = Headless::new(&gpu, &mut app, target, config).unwrap();
    headless.frames(&mut app, 1).unwrap();
    assert!(app.has_water_data(), "the opt-in graph allocated the water data");
    assert_eq!(app.water_bakes(), 1, "the first frame bakes");
    headless.frames(&mut app, 1).unwrap();
    assert_eq!(app.water_bakes(), 1, "a clean frame skips the bake");
    // An edit lands through the app's feed (the client's path): the next
    // frame bakes once more, then the field is clean again.
    let bytes: Vec<u8> = (0..4 * 4).flat_map(|_| 0.5f32.to_le_bytes()).collect();
    app.terrain_edits()
        .lock()
        .unwrap()
        .pending
        .push(ae_render::terrain::renderer::TerrainUpdate {
            version: 1,
            heights: vec![ae_render::upload::Region {
                mip: 0,
                x: 64,
                y: 64,
                width: 4,
                height: 4,
                bytes,
            }],
            ..Default::default()
        });
    headless.frames(&mut app, 1).unwrap();
    assert_eq!(app.water_bakes(), 2, "the edit bakes again");
    headless.frames(&mut app, 1).unwrap();
    assert_eq!(app.water_bakes(), 2, "and once more clean");
    headless.finish().unwrap();
    ae_render::thread::RenderApp::destroy(&mut app, &gpu);
    assert_clean(&gpu);
}

/// A v3 module that ignores the new bits still renders its M7 graph.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn m16_post_module_keeps_m7_with_all_water_tiers_requested() {
    use ae_render::library::WaterTiers;
    use ae_render::terrain::app::{TerrainApp, TerrainOutput};
    use ae_render::thread::{RenderApp, TargetInfo, run_headless};

    let gpu = gpu();
    let module = ae_render::module::build("post").expect("build render/post");
    let wet = ae_nav::scenario::coast_terrain(128);
    let (source, _) = coast_source(&wet);
    let extent = vk::Extent2D { width: 160, height: 90 };
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        profile: false,
        ..ExecutorConfig::default()
    };
    let render = |request, surface| {
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_water()
            .with_water_tiers(request)
            .with_module(module.clone());
        app.view().lock().unwrap().water = surface;
        app.create_offscreen(&gpu, extent).unwrap();
        run_headless(&gpu, &mut app, target, 8, config).unwrap();
        assert_eq!(app.water_tiers(), WaterTiers::default());
        assert!(!app.has_water_data());
        let pixels = app.readback().unwrap();
        RenderApp::destroy(&mut app, &gpu);
        pixels
    };
    let m7 = render(WaterTiers::default(), true);
    let requested = render(
        WaterTiers {
            reflect: true,
            refract: true,
            waves: true,
            volume: true,
        },
        true,
    );
    assert!(m7 == requested, "the extra flags changed the v3 module's M7 frame");
    assert!(m7 != render(WaterTiers::default(), false), "the M7 surface is visible");
    assert_clean(&gpu);
}

/// M16.5 review: M7, water-off and dry maps allocate no water data (the
/// shell of the finding was ~60-640 MiB of unused scratch), and no bake
/// runs.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn m7_off_and_dry_allocate_no_water_data() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput};
    use ae_render::thread::{TargetInfo, run_headless};

    let gpu = gpu();
    let extent = vk::Extent2D { width: 160, height: 90 };
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
    let wet = ae_nav::scenario::coast_terrain(128);
    let dry = ae_nav::scenario::land_crossing_terrain(128);
    for (name, terrain, water, surface) in [
        ("m7", &wet, true, true),
        ("off", &wet, true, false),
        ("dry", &dry, false, true),
    ] {
        let (source, level) = coast_source(terrain);
        assert_eq!(level.is_some(), water, "{name}");
        let mut app = TerrainApp::new(source, TerrainOutput::Readback);
        if water {
            app = app.with_water();
        }
        {
            let view = app.view();
            let mut view = view.lock().unwrap();
            view.water = surface;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        run_headless(&gpu, &mut app, target, 2, config).unwrap();
        assert!(!app.has_water_data(), "{name}: no water data");
        assert_eq!(app.water_bakes(), 0, "{name}: no bake");
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
    }
    assert_clean(&gpu);
}

/// M16.5 review: the caustic web is uploaded once, so a graph's first *real*
/// read must keep it. The executor starts a new external image at the
/// binding's initial layout (the upload's SHADER_READ_ONLY), not UNDEFINED: a
/// pass whose shader samples it reads the uploaded bytes. The water objects
/// are retired and destroyed before the device, and the validation counters
/// are checked after teardown.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn a_first_graph_read_keeps_the_uploaded_caustics() {
    use ae_framegraph::Role;
    use ae_render::bindless::{HandleTable, Push};
    use ae_render::executor::{BufferBinding, PassContext, Recorder};
    use gpu_allocator::MemoryLocation;

    struct Sample {
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        table: u32,
        caustics: u32,
        sampler: u32,
    }
    impl Recorder<()> for Sample {
        fn record(&mut self, _: &(), ctx: &PassContext<'_>) -> anyhow::Result<()> {
            let mut push = Push {
                words: [0, 0, 0, 0],
                ..Push::default()
            };
            push.table(self.table);
            push.image(0, self.caustics);
            push.sampler(0, self.sampler);
            unsafe {
                ctx.device
                    .cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
                ctx.device.cmd_bind_descriptor_sets(
                    ctx.cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[ctx.global_set],
                    &[],
                );
                ctx.device.cmd_push_constants(
                    ctx.cmd,
                    self.layout,
                    ae_render::bindless::PUSH_STAGES,
                    0,
                    ae_render::pipeline::push_bytes(&push),
                );
                ctx.device.cmd_dispatch(ctx.cmd, 1, 1, 1);
            }
            Ok(())
        }
    }

    let gpu = gpu();
    let instance = gpu.instance.clone();
    let mut executor = Executor::<()>::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    let mut water = ae_render::water::WaterRenderer::new(&gpu, executor.bindless_mut(), 8, 8, 0.5, 8.0, true, true)
        .expect("the water data");
    assert!(
        water.shore_binding().unwrap().sampled.is_some(),
        "the shore field has a bindless sampled handle for a graph consumer"
    );
    let before = water.caustics().unwrap().read(&gpu).expect("the web reads back");
    let sampled = water.caustics_binding().unwrap().sampled.unwrap();
    let mut out = OwnedBuffer::new(
        &gpu,
        BufferDesc {
            size: 4,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        },
        MemoryLocation::GpuToCpu,
        "caustic sample",
    )
    .unwrap();
    let out_handle = executor.bindless_mut().whole_buffer(&out).unwrap().index;
    let mut table = HandleTable::new(&gpu, executor.bindless_mut(), &[out_handle], "sample handles").unwrap();
    let sampler = executor
        .bindless_mut()
        .sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .max_lod(vk::LOD_CLAMP_NONE),
        )
        .unwrap();
    let pipeline = ae_render::pipeline::compute_pipeline(
        &gpu.device,
        ae_render::spirv!("test_sample_texel.comp"),
        executor.bindless().pipeline_layout(),
    )
    .unwrap();
    let mut graph = FrameGraph::<()>::new();
    let caustics = graph.image(
        "water.caustics",
        water.caustics().unwrap().desc(),
        ResourceClass::External,
    );
    let target = graph.buffer(
        "sample.out",
        BufferDesc {
            size: 4,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        },
        ResourceClass::External,
    );
    graph
        .pass("sample", PassKind::Compute, ())
        .sampled(caustics, ShaderStage::Compute)
        .access(target, Role::StorageBufferWrite(ShaderStage::Compute))
        .root();
    executor.set_graph(graph).unwrap();
    let caustics_binding = Binding::Image(water.caustics_binding().unwrap());
    let out_binding = Binding::Buffer(BufferBinding {
        buffer: out.buffer,
        offset: 0,
        size: 4,
        identity: 0x5A11_0001,
        storage: None,
    });
    let mut sample = Sample {
        pipeline,
        layout: executor.bindless().pipeline_layout(),
        table: table.index,
        caustics: sampled,
        sampler: sampler.index,
    };
    let token = executor.begin_frame().unwrap();
    executor
        .execute(
            token,
            &mut sample,
            &|resource| {
                if resource == caustics {
                    Some(caustics_binding)
                } else if resource == target {
                    Some(out_binding)
                } else {
                    None
                }
            },
            &[],
            &[],
        )
        .unwrap();
    executor.wait(token).unwrap();
    let word = u32::from_le_bytes(
        ae_render::upload::read_buffer(&gpu, out.buffer, 0, 4)
            .unwrap()
            .try_into()
            .unwrap(),
    );
    let texel = &before[0][..4];
    let want = texel[0] as u32 | (texel[1] as u32) << 8 | (texel[2] as u32) << 16 | (texel[3] as u32) << 24;
    assert_eq!(word, want, "the shader read the uploaded caustic texel");
    unsafe { gpu.device.destroy_pipeline(pipeline, None) };
    table.destroy(&gpu);
    water.retire(executor.bindless_mut());
    water.destroy(&gpu);
    drop(executor);
    out.destroy(&gpu);
    // Device teardown is where leaked objects are reported.
    drop(gpu);
    assert_eq!(instance.counters.errors(), 0, "validation errors after teardown");
    assert_eq!(instance.counters.warnings(), 0, "validation warnings after teardown");
}

/// M16.5 review: a failure while making the water data (an exhausted
/// storage-image registry) releases what it made; the device reports no
/// leaked objects at teardown.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn a_failed_water_construction_releases_its_objects() {
    let instance = Arc::new(GpuInstance::new(&[], Some(true)).unwrap());
    let gpu = Arc::new(GpuDevice::new(instance.clone(), None).unwrap());
    let mut executor = Executor::<()>::new(gpu.clone(), ExecutorConfig::default()).unwrap();
    let desc = ImageDesc::color_2d(vk::Format::R16_SFLOAT, 8, 8, vk::ImageUsageFlags::STORAGE);
    let mut image = OwnedImage::new(&gpu, desc, "descriptor exhaustion probe").unwrap();
    while executor
        .bindless_mut()
        .storage_image(image.view, vk::ImageLayout::GENERAL)
        .is_ok()
    {}
    let error = ae_render::water::WaterRenderer::new(&gpu, executor.bindless_mut(), 8, 8, 0.5, 8.0, true, true)
        .err()
        .expect("the storage-image registry is full");
    eprintln!("{error:#}");
    drop(executor);
    image.destroy(&gpu);
    drop(gpu);
    assert_eq!(
        instance.counters.errors(),
        0,
        "no leaked objects (see the validation log)"
    );
}

/// M16.5 review: each water input is allocated only when the graph declares
/// it. A shore-only graph gets the field but no caustic web; a caustics-only
/// graph gets the small web and none of the full-map shore images (which
/// used to allocate ~62.5 MiB for a web-only graph).
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn a_graph_allocates_only_the_water_inputs_it_declares() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput};
    use ae_render::thread::{TargetInfo, run_headless};

    let gpu = gpu();
    let wet = ae_nav::scenario::coast_terrain(128);
    let default: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../render/default.graph.json"
    )))
    .unwrap();
    let graph = |shore: bool, caustics: bool| -> Vec<u8> {
        let mut g = without_shore(default.clone());
        if shore {
            g["resources"].as_array_mut().unwrap().push(serde_json::json!({
                "name": "water.shore", "kind": "library", "when": "water"
            }));
            g["passes"].as_array_mut().unwrap().insert(
                0,
                serde_json::json!({
                    "name": "shore bake", "kind": "compute", "library": "water.shore",
                    "when": "water", "access": ["water.shore storage_image_write:compute"],
                    "root": true
                }),
            );
        }
        if caustics {
            let resource = g["resources"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|r| r["name"] == "water.caustics")
                .unwrap();
            resource["when"] = serde_json::json!("water");
        }
        serde_json::to_vec(&g).unwrap()
    };
    let extent = vk::Extent2D { width: 160, height: 90 };
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        frames_in_flight: 2,
        profile: false,
        bake: BakeOptions::default(),
        record_threads: 1,
    };
    for (shore, caustics) in [(true, false), (false, true)] {
        let (source, _) = coast_source(&wet);
        let mut app = TerrainApp::new(source, TerrainOutput::Readback)
            .with_water()
            .with_module(graph(shore, caustics));
        app.create_offscreen(&gpu, extent).unwrap();
        run_headless(&gpu, &mut app, target, 1, config).unwrap();
        assert_eq!(app.has_shore_data(), shore, "shore={shore} caustics={caustics}");
        assert_eq!(app.has_caustics_data(), caustics, "shore={shore} caustics={caustics}");
        assert_eq!(app.water_bakes(), u64::from(shore), "only a shore graph bakes");
        ae_render::thread::RenderApp::destroy(&mut app, &gpu);
    }
    assert_clean(&gpu);
}

// ---- M16.5 third review: acceptance gates and handle retirement -------------

/// The stock default graph as JSON (the client's module graph).
fn default_graph_json() -> serde_json::Value {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../render/default.graph.json"
    )))
    .unwrap()
}

/// The default graph without its own shore field (the waves tier declares
/// and samples it), so a test can declare the field its own way.
fn without_shore(mut graph: serde_json::Value) -> serde_json::Value {
    graph["resources"]
        .as_array_mut()
        .unwrap()
        .retain(|r| r["name"] != "water.shore");
    let passes = graph["passes"].as_array_mut().unwrap();
    passes.retain(|p| p["library"] != "water.shore");
    for pass in passes {
        if let Some(access) = pass["access"].as_array_mut() {
            access.retain(|a| !a.as_str().is_some_and(|a| a.starts_with("water.shore ")));
        }
    }
    graph
}

/// The default graph plus the water data: an optional shore bake (with an
/// optional module flag) and/or a `water.shore` sample on `water.draw`.
fn shore_json_graph(flag: Option<u32>, bake: bool, sample: bool) -> Vec<u8> {
    let mut graph = without_shore(default_graph_json());
    graph["resources"].as_array_mut().unwrap().push(serde_json::json!({
        "name": "water.shore", "kind": "library", "when": "water"
    }));
    let passes = graph["passes"].as_array_mut().unwrap();
    if bake {
        let mut pass = serde_json::json!({
            "name": "shore bake", "kind": "compute", "library": "water.shore",
            "when": "water", "access": ["water.shore storage_image_write:compute"],
            "root": true
        });
        if let Some(flag) = flag {
            pass["flag"] = flag.into();
        }
        passes.insert(0, pass);
    }
    if sample {
        passes.iter_mut().find(|p| p["library"] == "water.draw").unwrap()["access"]
            .as_array_mut()
            .unwrap()
            .push("water.shore sampled:fragment".into());
    }
    serde_json::to_vec(&graph).unwrap()
}

/// M16.5 third review: a graph that passes the app's checks but fails the
/// executor's frame-graph bake (here `depth` loses its sampled usage) must be
/// rejected *before* the water data is committed or the module counted as
/// loaded: the active module, its clean bake and its resources stay, and the
/// frame is drawn with them.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn an_invalid_graph_reload_keeps_the_active_water() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput};
    use ae_render::thread::{Headless, RenderApp, TargetInfo};

    let instance = Arc::new(GpuInstance::new(&[], Some(true)).unwrap());
    let gpu = Arc::new(GpuDevice::new(instance.clone(), None).unwrap());
    let wet = ae_nav::scenario::coast_terrain(128);
    let (source, _) = coast_source(&wet);
    let extent = vk::Extent2D { width: 160, height: 90 };
    let mut app = TerrainApp::new(source, TerrainOutput::Readback)
        .with_water()
        .with_module(shore_json_graph(None, true, true));
    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let mut headless = Headless::new(&gpu, &mut app, target, ExecutorConfig::default()).unwrap();
    headless.frames(&mut app, 2).unwrap();
    headless.wait().unwrap();
    assert_eq!(app.water_bakes(), 1);

    // The stock graph, with the depth's sampled usage removed: the app's
    // wiring accepts it, the frame-graph bake refuses it.
    let mut broken = default_graph_json();
    let depth = broken["resources"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|r| r["name"] == "depth")
        .unwrap();
    depth["usage"] = serde_json::json!(["depth_stencil_attachment"]);
    *app.module_reload().lock().unwrap() = Some(serde_json::to_vec(&broken).unwrap());
    let result = headless.frames(&mut app, 1);
    assert!(
        result.is_ok(),
        "the rejected reload must not break the frame: {result:?}"
    );
    assert!(app.has_shore_data(), "the active shore field stays");
    assert_eq!(app.water_bakes(), 1, "its clean bake stays clean");
    let status = app.module_status().lock().unwrap().clone();
    assert_eq!(status.loads, 1, "the rejected graph is not counted as loaded");
    assert_eq!(status.errors.len(), 1, "{:?}", status.errors);
    assert!(status.errors[0].contains("frame graph bake"), "{:?}", status.errors);
    headless.finish().unwrap();
    app.destroy(&gpu);
    drop(gpu);
    assert_eq!(instance.counters.errors(), 0, "validation errors after teardown");
    assert_eq!(instance.counters.warnings(), 0, "validation warnings after teardown");
}

/// M16.5 third review: a full retarget destroys the water data, so it must
/// give its bindless handles back with the other recreated objects. The live
/// registry stays flat through repeated same-executor rebuilds, and the
/// recreated field bakes and is readable with validation clean.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn a_full_retarget_retires_the_water_handles() {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput};
    use ae_render::thread::{Headless, RenderApp, TargetInfo};

    let instance = Arc::new(GpuInstance::new(&[], Some(true)).unwrap());
    let gpu = Arc::new(GpuDevice::new(instance.clone(), None).unwrap());
    let wet = ae_nav::scenario::coast_terrain(128);
    let (source, _) = coast_source(&wet);
    let extent = vk::Extent2D { width: 160, height: 90 };
    let mut app = TerrainApp::new(source, TerrainOutput::Readback)
        .with_water()
        .with_module(shore_json_graph(None, true, true));
    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let mut headless = Headless::new(&gpu, &mut app, target, ExecutorConfig::default()).unwrap();
    headless.frames(&mut app, 2).unwrap();
    headless.wait().unwrap();
    let live = headless.bindless_mut().live();
    for round in 0..3 {
        assert!(
            app.retarget(&gpu, headless.bindless_mut(), target, 2).unwrap(),
            "a full retarget rebuilds"
        );
        app.create_offscreen(&gpu, target.extent).unwrap();
        headless.forget_external_states();
        headless.frames(&mut app, 1).unwrap();
        headless.wait().unwrap();
        assert_eq!(
            headless.bindless_mut().live(),
            live,
            "round {round}: live handles stay flat through a retarget"
        );
        assert!(app.has_shore_data(), "round {round}: the field is recreated");
        assert_eq!(
            app.water_bakes(),
            1,
            "round {round}: the new field bakes on its first frame"
        );
    }
    headless.finish().unwrap();
    app.destroy(&gpu);
    drop(gpu);
    assert_eq!(instance.counters.errors(), 0, "validation errors after teardown");
    assert_eq!(instance.counters.warnings(), 0, "validation warnings after teardown");
}

// ---- M16.5 fourth review: explicit library descriptors ---------------------

/// The stock default graph declared for the water fixture, as a WASM guest
/// builds it, with the given edit applied; `wat_render_module` packs the
/// encoded graph and program table into the module layout a real render
/// module exports.
fn stock_render_module(edit: impl FnOnce(&mut FrameGraph<u32>, &mut ae_render_abi::module::ProgramTable)) -> Vec<u8> {
    use ae_render::encode::{DisplayLevels, OutputEncoding};
    use ae_render::module::DeclareTarget;
    use ae_render_abi::module::ProgramTable;

    let base_json = serde_json::to_vec(&default_graph_json()).unwrap();
    let mut file = ae_render::graph_file::GraphFile::parse(&base_json).unwrap();
    let target = DeclareTarget {
        width: 160,
        height: 90,
        color_format: vk::Format::R8G8B8A8_SRGB,
        flags: ae_render_abi::module::DECLARE_READBACK | ae_render_abi::module::DECLARE_WATER,
        encoding: OutputEncoding::Sdr,
        levels: DisplayLevels::default(),
        working: Default::default(),
        gamut: Default::default(),
    };
    let declared = file.declare(&target).unwrap().unwrap();
    let mut graph: FrameGraph<u32> = ae_framegraph::descriptor::decode(&declared.graph).unwrap();
    let mut table = ProgramTable::new();
    for program in ProgramTable::decode(&declared.programs).unwrap() {
        table.push(program);
    }
    edit(&mut graph, &mut table);
    let encoded = ae_framegraph::descriptor::encode(&graph, |id| *id);
    wat_render_module(&encoded, &table.finish())
}

/// The module layout the sim and render guests export: graph and program
/// blobs in linear memory behind `ae_render_declare`.
fn wat_render_module(graph: &[u8], programs: &[u8]) -> Vec<u8> {
    let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("\\{b:02x}")).collect::<String>();
    let (g, p) = (4096u32, 4096 + graph.len() as u32);
    let words: Vec<u8> = [g, graph.len() as u32, p, programs.len() as u32]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    format!(
        r#"(module
            (memory (export "memory") 2)
            (data (i32.const 64) "{}")
            (data (i32.const {g}) "{}")
            (data (i32.const {p}) "{}")
            (func (export "ae_render_version") (result i32) i32.const 3)
            (func (export "ae_render_declare_input") (result i32) i32.const 512)
            (func (export "ae_render_declare") (result i32) i32.const 64)
            (func (export "ae_render_input") (result i32) i32.const 1024)
            (func (export "ae_render_frame") (result i32) i32.const 128))"#,
        hex(&words),
        hex(graph),
        hex(programs)
    )
    .into_bytes()
}

/// A wet app with a live, clean shore bake (the stock graph plus a JSON
/// `water.shore` bake and sample) and a reload slot.
fn module_app(
    module: Vec<u8>,
) -> (
    Arc<GpuInstance>,
    Arc<GpuDevice>,
    ae_render::terrain::app::TerrainApp,
    ae_render::thread::Headless<<ae_render::terrain::app::TerrainApp as ae_render::thread::RenderApp>::Pass>,
) {
    use ae_render::terrain::app::{TerrainApp, TerrainOutput};
    use ae_render::thread::{Headless, TargetInfo};

    let instance = Arc::new(GpuInstance::new(&[], Some(true)).unwrap());
    let gpu = Arc::new(GpuDevice::new(instance.clone(), None).unwrap());
    let wet = ae_nav::scenario::coast_terrain(128);
    let (source, _) = coast_source(&wet);
    let extent = vk::Extent2D { width: 160, height: 90 };
    let mut app = TerrainApp::new(source, TerrainOutput::Readback)
        .with_water()
        .with_module(module);
    app.view().lock().unwrap().module_flags = ae_render_abi::module::DECLARE_WATER;
    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let headless = Headless::new(&gpu, &mut app, target, ExecutorConfig::default()).unwrap();
    (instance, gpu, app, headless)
}

fn water_reload_setup() -> (
    Arc<GpuInstance>,
    Arc<GpuDevice>,
    ae_render::terrain::app::TerrainApp,
    ae_render::thread::Headless<<ae_render::terrain::app::TerrainApp as ae_render::thread::RenderApp>::Pass>,
) {
    module_app(shore_json_graph(None, true, true))
}

/// M16.5 fourth review: a module that declares its own `water.caustics`
/// descriptor may not add a usage the uploaded web lacks. The reload is
/// rejected by name before it commits: the live graph (and its clean shore
/// bake) keeps drawing, the load count stays, and validation is clean.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn an_explicit_caustics_attachment_usage_is_rejected() {
    use ae_render::thread::RenderApp;
    use ae_render::water::caustics;

    let (instance, gpu, mut app, mut headless) = water_reload_setup();
    headless.frames(&mut app, 2).unwrap();
    headless.wait().unwrap();
    assert_eq!(app.water_bakes(), 1);

    let module = stock_render_module(|graph, _| {
        graph.image(
            "water.caustics",
            ImageDesc {
                mip_levels: ae_render::upload::mip_levels(caustics::SIZE, caustics::SIZE),
                ..ImageDesc::color_2d(
                    caustics::FORMAT,
                    caustics::SIZE,
                    caustics::SIZE,
                    vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                )
            },
            ResourceClass::External,
        );
    });
    *app.module_reload().lock().unwrap() = Some(module);
    let result = headless.frames(&mut app, 1);
    assert!(result.is_ok(), "the rejected reload keeps drawing: {result:?}");
    let status = app.module_status().lock().unwrap().clone();
    assert!(app.has_shore_data(), "the active shore data stays");
    assert_eq!(app.water_bakes(), 1, "its clean bake stays clean");
    assert_eq!(status.loads, 1, "the rejected graph is not counted as loaded");
    assert_eq!(status.errors.len(), 1, "{:?}", status.errors);
    assert!(
        status.errors[0].contains("water.caustics") && status.errors[0].contains("COLOR_ATTACHMENT"),
        "{:?}",
        status.errors
    );
    headless.finish().unwrap();
    app.destroy(&gpu);
    drop(gpu);
    assert_eq!(instance.counters.errors(), 0, "validation errors after teardown");
    assert_eq!(instance.counters.warnings(), 0, "validation warnings after teardown");
}

/// M16.5 fourth review: the same for `water.shore`, whose bound field is
/// storage, sampled and a transfer source, never a colour attachment. The
/// module also declares the field's producer, so only the descriptor can be
/// the reason the reload is refused.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn an_explicit_shore_attachment_usage_is_rejected() {
    use ae_render::thread::RenderApp;
    use ae_render::water::shore;

    let (instance, gpu, mut app, mut headless) = water_reload_setup();
    headless.frames(&mut app, 2).unwrap();
    headless.wait().unwrap();
    assert_eq!(app.water_bakes(), 1);

    let module = stock_render_module(|graph, table| {
        let bake = table.library("water.shore");
        let field = graph.image(
            "water.shore",
            ImageDesc::color_2d(
                shore::FIELD_FORMAT,
                129,
                129,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
            ResourceClass::External,
        );
        graph
            .pass("shore bake", PassKind::Compute, bake)
            .access(field, ae_framegraph::Role::StorageImageWrite(ShaderStage::Compute))
            .root();
    });
    *app.module_reload().lock().unwrap() = Some(module);
    let result = headless.frames(&mut app, 1);
    assert!(result.is_ok(), "the rejected reload keeps drawing: {result:?}");
    let status = app.module_status().lock().unwrap().clone();
    assert!(app.has_shore_data(), "the active shore data stays");
    assert_eq!(app.water_bakes(), 1, "its clean bake stays clean");
    assert_eq!(status.loads, 1, "the rejected graph is not counted as loaded");
    assert_eq!(status.errors.len(), 1, "{:?}", status.errors);
    assert!(
        status.errors[0].contains("water.shore") && status.errors[0].contains("COLOR_ATTACHMENT"),
        "{:?}",
        status.errors
    );
    headless.finish().unwrap();
    app.destroy(&gpu);
    drop(gpu);
    assert_eq!(instance.counters.errors(), 0, "validation errors after teardown");
    assert_eq!(instance.counters.warnings(), 0, "validation warnings after teardown");
}

/// M16.5 fourth review: the control. A module that declares the caustic web
/// exactly as the host allocates it (256², the nine-mip chain, sampled only)
/// and samples it from the water pass loads and draws with no messages.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn an_explicit_caustics_sampled_descriptor_is_accepted() {
    use ae_render::thread::RenderApp;
    use ae_render::water::caustics;

    let (instance, gpu, mut app, mut headless) = water_reload_setup();
    headless.frames(&mut app, 2).unwrap();
    headless.wait().unwrap();

    let module = stock_render_module(|graph, _| {
        let web = graph.image(
            "water.caustics",
            ImageDesc {
                mip_levels: ae_render::upload::mip_levels(caustics::SIZE, caustics::SIZE),
                ..ImageDesc::color_2d(
                    caustics::FORMAT,
                    caustics::SIZE,
                    caustics::SIZE,
                    vk::ImageUsageFlags::SAMPLED,
                )
            },
            ResourceClass::External,
        );
        let water = graph.passes.iter_mut().find(|p| p.name == "water").unwrap();
        water.intents.push(ae_framegraph::Intent {
            resource: web,
            role: ae_framegraph::Role::Sampled(ShaderStage::Fragment),
            discard: false,
        });
    });
    *app.module_reload().lock().unwrap() = Some(module);
    let result = headless.frames(&mut app, 1);
    assert!(result.is_ok(), "{result:?}");
    let status = app.module_status().lock().unwrap().clone();
    assert!(status.errors.is_empty(), "{:?}", status.errors);
    assert_eq!(status.loads, 2, "the accepted graph is loaded");
    assert!(app.has_caustics_data(), "the graph's web is allocated");
    headless.finish().unwrap();
    app.destroy(&gpu);
    drop(gpu);
    assert_eq!(instance.counters.errors(), 0, "validation errors after teardown");
    assert_eq!(instance.counters.warnings(), 0, "validation warnings after teardown");
}

// ---- M16.5 fifth review: one library resource, one id -----------------------

/// A render module whose tonemap pass samples the shore field's travel time
/// through a test fullscreen shader and writes it to the readback target.
/// The shore bake is deliberately unrooted: only a reader of the *same*
/// resource id keeps it alive. `duplicate` declares `water.shore` twice and
/// samples the second id, which used to bind the same image without making
/// the first writer a dependency (so the bake was pruned and the read zero).
fn shore_consumer_module(duplicate: bool) -> Vec<u8> {
    use ae_framegraph::{Intent, Role};

    stock_render_module(|graph, table| {
        let sample = table.push(ae_render_abi::module::Program::Fullscreen {
            label: "shore probe",
            spirv: include_bytes!(concat!(env!("OUT_DIR"), "/test_sample_shore.frag.spv")),
            uniform_bytes: 0,
        });
        let spec = ImageDesc::color_2d(
            vk::Format::R16_SFLOAT,
            129,
            129,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        );
        let write = graph.image("water.shore", spec, ResourceClass::External);
        let read = if duplicate {
            graph.image("water.shore", spec, ResourceClass::External)
        } else {
            write
        };
        let bake = table.library("water.shore");
        graph
            .pass("shore bake", PassKind::Compute, bake)
            .access(write, Role::StorageImageWrite(ShaderStage::Compute));
        let tonemap = graph.passes.iter_mut().find(|pass| pass.name == "tonemap").unwrap();
        tonemap.record = sample;
        tonemap
            .intents
            .retain(|intent| matches!(intent.role, Role::ColorAttachment(_)));
        tonemap.intents.push(Intent {
            resource: read,
            role: Role::Sampled(ShaderStage::Fragment),
            discard: false,
        });
    })
}

/// M16.5 fifth review: two declarations of the singleton `water.shore`
/// resource bind the same image under two ids, and the frame graph would let
/// the second reader outlive the first writer's bake. The module is refused
/// on its first declaration: no water data is committed, the frame reports
/// the resource by name and validation stays clean.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn a_duplicate_water_shore_declaration_is_rejected_on_load() {
    use ae_render::thread::RenderApp;

    let (instance, gpu, mut app, mut headless) = module_app(shore_consumer_module(true));
    let result = headless.frames(&mut app, 1);
    let error = format!("{:#}", result.expect_err("the duplicate declaration is refused"));
    assert!(
        error.contains("water.shore") && error.contains("more than once"),
        "{error}"
    );
    assert!(!app.has_water_data(), "a refused first declaration commits nothing");
    app.destroy(&gpu);
    drop(headless);
    drop(gpu);
    assert_eq!(instance.counters.errors(), 0, "validation errors after teardown");
    assert_eq!(instance.counters.warnings(), 0, "validation warnings after teardown");
}

/// The same duplicate module on a live reload: it is rejected by name, the
/// running graph, its shore field and its clean bake stay, the load count
/// stays, and the frame keeps drawing.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn a_duplicate_water_shore_declaration_is_rejected_on_reload() {
    use ae_render::thread::RenderApp;

    let (instance, gpu, mut app, mut headless) = water_reload_setup();
    headless.frames(&mut app, 2).unwrap();
    headless.wait().unwrap();
    assert_eq!(app.water_bakes(), 1);

    *app.module_reload().lock().unwrap() = Some(shore_consumer_module(true));
    let result = headless.frames(&mut app, 1);
    assert!(result.is_ok(), "the rejected reload keeps drawing: {result:?}");
    let status = app.module_status().lock().unwrap().clone();
    assert!(app.has_shore_data(), "the active shore field stays");
    assert_eq!(app.water_bakes(), 1, "its clean bake stays clean");
    assert_eq!(status.loads, 1, "the rejected graph is not counted as loaded");
    assert_eq!(status.errors.len(), 1, "{:?}", status.errors);
    assert!(
        status.errors[0].contains("water.shore") && status.errors[0].contains("more than once"),
        "{:?}",
        status.errors
    );
    headless.finish().unwrap();
    app.destroy(&gpu);
    drop(gpu);
    assert_eq!(instance.counters.errors(), 0, "validation errors after teardown");
    assert_eq!(instance.counters.warnings(), 0, "validation warnings after teardown");
}

/// The control: one declaration, the same unrooted bake and sample shader.
/// The producer is a dependency of the reader, so the bake runs once and the
/// target's red carries the field's travel time (the shader scales 100 s).
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn a_unique_water_shore_consumer_keeps_its_bake_live() {
    use ae_render::thread::RenderApp;

    let (instance, gpu, mut app, mut headless) = module_app(shore_consumer_module(false));
    let result = headless.frames(&mut app, 1);
    headless.finish().unwrap();
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(app.water_bakes(), 1, "the sampled producer bakes");
    let red = app.readback().unwrap()[0];
    assert!(red > 0, "the target's red carries the shore travel time");
    app.destroy(&gpu);
    drop(gpu);
    assert_eq!(instance.counters.errors(), 0, "validation errors after teardown");
    assert_eq!(instance.counters.warnings(), 0, "validation warnings after teardown");
}

/// The coast's reflection target before the surface samples it. Draws a
/// tower, and three darts that can be hidden independently by view state
/// or the water plane.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn coast_planar_reflection_is_clipped_and_observer_safe() {
    use ae_framegraph::{ResourceId, Role};
    use ae_model::{Bone, Bounds, Joint, Material, Model};
    use ae_render::library::{LibraryPass, WaterTiers};
    use ae_render::terrain::app::{NATIVE, PassWork, ScenePass, TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{RenderApp, TargetInfo};
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Mat4, Vec2, Vec3};

    let gpu = gpu();
    let terrain = ae_nav::scenario::coast_terrain(512);
    let cells = (terrain.geom.cells_x as usize, terrain.geom.cells_y as usize);
    let (heightmap, scale, offset) =
        Heightmap::from_cells(cells, |x, z| terrain.ground_height(z * cells.0 + x) as f32 * 0.1, None);
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let level = terrain.water_level as f32 * 0.1;
    let info = TerrainInfo::new(&heightmap, &pyramid, terrain.geom.cell_size, scale, offset, Some(level));
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: Some(terrain.material),
    });
    let identity = Mat4::IDENTITY.to_cols_array();
    let mut tower = Model {
        bones: vec![Bone {
            name: "root".into(),
            parent: None,
            rest: identity,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }],
        joints: vec![Joint {
            bone: 0,
            inverse_bind: identity,
        }],
        materials: vec![Material {
            base_color: [1.0, 0.001, 0.001, 1.0],
            ..Default::default()
        }],
        bounds: Bounds {
            min: [-0.35, 0.0, -0.35],
            max: [0.35, 1.6, 0.35],
        },
        ..Default::default()
    };
    model_box(&mut tower, [-0.35, 0.0, -0.35], [0.35, 1.6, 0.35], 0);
    tower.validate().unwrap();
    let feed = Arc::new(UnitFeed::default());
    let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
        .with_water()
        .with_water_tiers(WaterTiers {
            reflect: true,
            ..Default::default()
        })
        .with_units(UnitsSetup {
            feed: feed.clone(),
            capacity: 16,
            classes: vec![
                UnitClass {
                    color: [1.0, 0.001, 0.001],
                    gamut: ae_render::colorspace::Gamut::Bt709,
                    floats: true,
                };
                2
            ],
            min_pixels: 1.0,
            models: vec![Some(Arc::new(tower)), None],
            rigs: Vec::new(),
        });
    let extent = vk::Extent2D {
        width: 321,
        height: 241,
    };
    let (rw, rh) = (extent.width.div_ceil(2), extent.height.div_ceil(2));
    let view = app.view();
    let (center, camera) = {
        let mut view = view.lock().unwrap();
        view.camera.target = Vec2::new(4096.0 * 0.56, 4096.0 * 0.5);
        view.camera.yaw = 0.0;
        view.camera.pitch = 0.52;
        view.camera.distance = 530.0;
        view.fog = false;
        view.time = Some(0.0);
        (view.camera.target, view.camera)
    };
    let offsets = [-136.0, -49.0, 42.0, 135.0];
    let radius = 23.0f32;
    let publish = |tick, filtered| {
        let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
        for (i, dx) in offsets.into_iter().enumerate() {
            let x = center.x + dx;
            let z = center.y;
            let words: [u32; ATTRIBUTE_COUNT] = [
                x.to_bits(),
                z.to_bits(),
                x.to_bits(),
                z.to_bits(),
                0,
                0,
                0,
                1.0f32.to_bits(),
                0,
                0,
                0,
                1.0f32.to_bits(),
                radius.to_bits(),
                u32::from(i != 0),
                ae_render::units::NO_POSE,
                if filtered && i == 3 { 55.0f32.to_bits() } else { 0 },
                if filtered { [0, 2, 3, 1][i] } else { 0 },
            ];
            for (column, word) in columns.iter_mut().zip(words) {
                column.push(word);
            }
        }
        feed.publish(Arc::new(InstanceStream {
            tick,
            published: std::time::Instant::now(),
            tick_seconds: 0.0,
            sim_seconds: 0.0,
            columns,
            anim: AnimWrites::default(),
        }));
    };

    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let mut executor = Executor::new(
        gpu.clone(),
        ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 2,
        },
    )
    .unwrap();
    app.create(&gpu, executor.bindless_mut(), target, 2).unwrap();
    let (mut graph, _) = app.declare(target, executor.bindless_mut()).unwrap();
    assert!(app.water_tiers().reflect);
    let image = ResourceId(
        graph
            .resources
            .iter()
            .position(|r| r.name == "water.reflection")
            .unwrap() as u32,
    );
    let desc = BufferDesc {
        size: u64::from(rw) * u64::from(rh) * 8,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
    };
    let buffer = graph.buffer("reflection probe", desc, ResourceClass::External);
    graph
        .pass(
            "reflection probe copy",
            PassKind::Transfer,
            ScenePass {
                program: NATIVE,
                work: PassWork::Readback { image, buffer },
            },
        )
        .access(image, Role::TransferSrc)
        .discard(buffer, Role::TransferDst)
        .id();
    graph
        .pass(
            "reflection probe read",
            PassKind::Transfer,
            ScenePass {
                program: NATIVE,
                work: PassWork::Library(LibraryPass::HostRead),
            },
        )
        .access(buffer, Role::HostRead)
        .root();
    let mut readback = OwnedBuffer::new(&gpu, desc, MemoryLocation::GpuToCpu, "reflection probe").unwrap();
    executor.set_graph(graph).unwrap();
    let (all, filtered, off) = {
        let mut render = || {
            let mut last = None;
            for _ in 0..6 {
                let token = executor.begin_frame().unwrap();
                app.prepare(executor.bindless_mut(), token.slot).unwrap();
                let mut externals = (0..executor.graph().resources.len() as u32)
                    .map(|index| app.external(ResourceId(index), token.slot))
                    .collect::<Vec<_>>();
                externals[buffer.index()] = Some(Binding::Buffer(BufferBinding {
                    buffer: readback.buffer,
                    size: desc.size,
                    offset: 0,
                    identity: 0x5245464c,
                    storage: None,
                }));
                executor
                    .execute(token, app.recorder(), &|id| externals[id.index()], &[], &[])
                    .unwrap();
                last = Some(token);
            }
            executor.wait(last.unwrap()).unwrap();
            (readback.mapped().unwrap().to_vec(), app.readback().unwrap())
        };
        publish(1, false);
        render();
        view.lock().unwrap().frozen = Some(lod_view(&source, &camera, extent.width, extent.height));
        let (all, surface_all) = render();
        publish(2, true);
        let (filtered, surface_filtered) = render();
        assert_ne!(surface_all, surface_filtered, "water samples the reflected geometry");
        view.lock().unwrap().water = false;
        let (off, _) = render();
        (all, filtered, off)
    };
    assert!(
        off.iter().all(|&byte| byte == 0),
        "the water toggle skips the mirror draws"
    );

    let project = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let red_pixels = |image: &[u8], x: f32, height: f32| {
        let p = project * Vec3::new(x, level - height, center.y).extend(1.0);
        let cx = ((p.x / p.w * 0.5 + 0.5) * rw as f32) as i32;
        let cy = ((p.y / p.w * 0.5 + 0.5) * rh as f32) as i32;
        assert!((7..rw as i32 - 7).contains(&cx) && (7..rh as i32 - 7).contains(&cy));
        (cy - 7..=cy + 7)
            .flat_map(|y| (cx - 7..=cx + 7).map(move |x| (y * rw as i32 + x) as usize * 8))
            .filter(|&i| {
                let [r, g, b] = scrgb_pixel(image, i);
                r > 0.004 && r > 3.0 * g.max(b)
            })
            .count()
    };
    for (i, dx) in offsets.into_iter().enumerate() {
        let height = if i == 0 { radius * 0.8 } else { radius * 0.3 };
        let (unfiltered, visible) = (
            red_pixels(&all, center.x + dx, height),
            red_pixels(&filtered, center.x + dx, height),
        );
        assert!(
            unfiltered >= 2,
            "row {i} appears at its mirrored position ({unfiltered} red pixels)"
        );
        if i == 0 {
            assert_eq!(visible, unfiltered, "Own tower stays in the mirror");
        } else {
            assert_eq!(visible, 0, "blip, ghost and submerged row are absent ({i})");
        }
    }
    assert!(
        filtered.as_chunks::<8>().0.contains(&[0; 8]),
        "sky keeps the clear value"
    );
    gpu.wait_idle();
    drop(executor);
    app.destroy(&gpu);
    readback.destroy(&gpu);
    assert_clean(&gpu);
}

/// The mirror cull keeps only units whose reflection can land on water: on
/// a map whose left half is dry land 20 m above the water and right half
/// sea, a tower and a dart far inland are not drawn into the reflection
/// (their mirrored boxes, widened by the water's sample shift, meet only
/// dry ground), the same two over the sea are. Counted from the mirror's
/// indirect arguments: the raised land would hide the inland pair in the
/// reflection's pixels anyway.
#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn mirror_cull_drops_units_whose_reflection_misses_the_water() {
    use ae_framegraph::{ResourceId, Role};
    use ae_model::{Bone, Bounds, Joint, Material, Model};
    use ae_render::library::{LibraryPass, WaterTiers};
    use ae_render::terrain::app::{NATIVE, PassWork, ScenePass, TerrainApp, TerrainOutput, TerrainSource, lod_view};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{RenderApp, TargetInfo};
    use ae_render::units::{AnimWrites, InstanceStream, UnitClass, UnitFeed, UnitsSetup};
    use glam::{Mat4, Vec2, Vec3};

    let gpu = gpu();
    // 256 x 256 cells of 4 m: the shore is the line x = 512 m.
    let land = 20.0f32;
    let (heightmap, scale, offset) = Heightmap::from_cells((256, 256), |x, _| if x < 128 { land } else { -20.0 }, None);
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let info = TerrainInfo::new(&heightmap, &pyramid, 4.0, scale, offset, Some(0.0));
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let identity = Mat4::IDENTITY.to_cols_array();
    let mut tower = Model {
        bones: vec![Bone {
            name: "root".into(),
            parent: None,
            rest: identity,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }],
        joints: vec![Joint {
            bone: 0,
            inverse_bind: identity,
        }],
        materials: vec![Material {
            base_color: [1.0, 0.001, 0.001, 1.0],
            ..Default::default()
        }],
        bounds: Bounds {
            min: [-0.35, 0.0, -0.35],
            max: [0.35, 1.6, 0.35],
        },
        ..Default::default()
    };
    model_box(&mut tower, [-0.35, 0.0, -0.35], [0.35, 1.6, 0.35], 0);
    tower.validate().unwrap();
    let feed = Arc::new(UnitFeed::default());
    let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
        .with_water()
        .with_water_tiers(WaterTiers {
            reflect: true,
            ..Default::default()
        })
        .with_units(UnitsSetup {
            feed: feed.clone(),
            capacity: 16,
            classes: vec![
                UnitClass {
                    color: [1.0, 0.001, 0.001],
                    gamut: ae_render::colorspace::Gamut::Bt709,
                    floats: true,
                };
                2
            ],
            min_pixels: 1.0,
            models: vec![Some(Arc::new(tower)), None],
            rigs: Vec::new(),
        });
    let extent = vk::Extent2D {
        width: 321,
        height: 241,
    };
    let (rw, rh) = (extent.width.div_ceil(2), extent.height.div_ceil(2));
    let view = app.view();
    let (center, camera) = {
        let mut view = view.lock().unwrap();
        view.camera.target = Vec2::new(400.0, 512.0);
        view.camera.yaw = 0.0;
        view.camera.pitch = 0.52;
        view.camera.distance = 530.0;
        view.fog = false;
        view.time = Some(0.0);
        (view.camera.target, view.camera)
    };
    // (x offset from the view's centre, z offset, class): the tower and
    // the dart 248 m inland, then 24 m out to sea.
    let units = [
        (-136.0, -40.0, 0u32),
        (-136.0, 40.0, 1),
        (136.0, -40.0, 0),
        (136.0, 40.0, 1),
    ];
    let radius = 8.0f32;
    let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = Default::default();
    for (dx, dz, class) in units {
        let (x, z) = (center.x + dx, center.y + dz);
        let words: [u32; ATTRIBUTE_COUNT] = [
            x.to_bits(),
            z.to_bits(),
            x.to_bits(),
            z.to_bits(),
            0,
            0,
            0,
            1.0f32.to_bits(),
            0,
            0,
            0,
            1.0f32.to_bits(),
            radius.to_bits(),
            class,
            ae_render::units::NO_POSE,
            0,
            0,
        ];
        for (column, word) in columns.iter_mut().zip(words) {
            column.push(word);
        }
    }
    feed.publish(Arc::new(InstanceStream {
        tick: 1,
        published: std::time::Instant::now(),
        tick_seconds: 0.0,
        sim_seconds: 0.0,
        columns,
        anim: AnimWrites::default(),
    }));

    app.create_offscreen(&gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let mut executor = Executor::new(
        gpu.clone(),
        ExecutorConfig {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 2,
        },
    )
    .unwrap();
    app.create(&gpu, executor.bindless_mut(), target, 2).unwrap();
    let (mut graph, _) = app.declare(target, executor.bindless_mut()).unwrap();
    let image = ResourceId(
        graph
            .resources
            .iter()
            .position(|r| r.name == "water.reflection")
            .unwrap() as u32,
    );
    let desc = BufferDesc {
        size: u64::from(rw) * u64::from(rh) * 8,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
    };
    let buffer = graph.buffer("reflection probe", desc, ResourceClass::External);
    graph
        .pass(
            "reflection probe copy",
            PassKind::Transfer,
            ScenePass {
                program: NATIVE,
                work: PassWork::Readback { image, buffer },
            },
        )
        .access(image, Role::TransferSrc)
        .discard(buffer, Role::TransferDst)
        .id();
    graph
        .pass(
            "reflection probe read",
            PassKind::Transfer,
            ScenePass {
                program: NATIVE,
                work: PassWork::Library(LibraryPass::HostRead),
            },
        )
        .access(buffer, Role::HostRead)
        .root();
    // The mirrored units' indirect arguments: the darts' instance count,
    // then one command per model primitive.
    let draws = ResourceId(
        graph
            .resources
            .iter()
            .position(|r| r.name == ae_render_abi::module::UNITS_INDIRECT_REFLECT_NAME)
            .unwrap() as u32,
    );
    let counts_desc = BufferDesc {
        size: 4096,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
    };
    let counts_buffer = graph.buffer("mirror counts probe", counts_desc, ResourceClass::External);
    graph
        .pass(
            "mirror counts copy",
            PassKind::Transfer,
            ScenePass {
                program: NATIVE,
                work: PassWork::BufferReadback {
                    source: draws,
                    buffer: counts_buffer,
                },
            },
        )
        .access(draws, Role::TransferSrc)
        .discard(counts_buffer, Role::TransferDst)
        .id();
    graph
        .pass(
            "mirror counts read",
            PassKind::Transfer,
            ScenePass {
                program: NATIVE,
                work: PassWork::Library(LibraryPass::HostRead),
            },
        )
        .access(counts_buffer, Role::HostRead)
        .root();
    let mut readback = OwnedBuffer::new(&gpu, desc, MemoryLocation::GpuToCpu, "reflection probe").unwrap();
    let mut counts = OwnedBuffer::new(&gpu, counts_desc, MemoryLocation::GpuToCpu, "mirror counts probe").unwrap();
    // Past the source's end the copy leaves these zeros.
    counts.mapped().unwrap().fill(0);
    executor.set_graph(graph).unwrap();
    view.lock().unwrap().frozen = Some(lod_view(&source, &camera, extent.width, extent.height));
    let mut last = None;
    for _ in 0..6 {
        let token = executor.begin_frame().unwrap();
        app.prepare(executor.bindless_mut(), token.slot).unwrap();
        let mut externals = (0..executor.graph().resources.len() as u32)
            .map(|index| app.external(ResourceId(index), token.slot))
            .collect::<Vec<_>>();
        externals[buffer.index()] = Some(Binding::Buffer(BufferBinding {
            buffer: readback.buffer,
            size: desc.size,
            offset: 0,
            identity: 0x5245464c,
            storage: None,
        }));
        externals[counts_buffer.index()] = Some(Binding::Buffer(BufferBinding {
            buffer: counts.buffer,
            size: counts_desc.size,
            offset: 0,
            identity: 0x4d495252,
            storage: None,
        }));
        executor
            .execute(token, app.recorder(), &|id| externals[id.index()], &[], &[])
            .unwrap();
        last = Some(token);
    }
    executor.wait(last.unwrap()).unwrap();
    let mirror = readback.mapped().unwrap().to_vec();
    let words: Vec<u32> = counts
        .mapped()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    // The darts' count (word 1), then each command's (word 4 + 5k + 1).
    let mirrored = words[1] + (4..words.len() - 1).step_by(5).map(|k| words[k + 1]).sum::<u32>();
    assert_eq!(mirrored, 2, "only the tower and the dart over the sea are mirrored");

    let project = lod_view(&source, &camera, extent.width, extent.height).view_proj;
    let red_pixels = |x: f32, y: f32, z: f32| {
        let p = project * Vec3::new(x, -y, z).extend(1.0);
        let cx = ((p.x / p.w * 0.5 + 0.5) * rw as f32) as i32;
        let cy = ((p.y / p.w * 0.5 + 0.5) * rh as f32) as i32;
        assert!((7..rw as i32 - 7).contains(&cx) && (7..rh as i32 - 7).contains(&cy));
        (cy - 7..=cy + 7)
            .flat_map(|y| (cx - 7..=cx + 7).map(move |x| (y * rw as i32 + x) as usize * 8))
            .filter(|&i| {
                let [r, g, b] = scrgb_pixel(&mirror, i);
                r > 0.004 && r > 3.0 * g.max(b)
            })
            .count()
    };
    for (i, (dx, dz, _)) in units.into_iter().enumerate() {
        let ground = if dx < 0.0 { land } else { 0.0 };
        let red = red_pixels(center.x + dx, ground + radius * 0.5, center.y + dz);
        if dx > 0.0 {
            assert!(red >= 2, "unit {i} over the sea is mirrored ({red} red pixels)");
        }
    }
    gpu.wait_idle();
    drop(executor);
    app.destroy(&gpu);
    readback.destroy(&gpu);
    counts.destroy(&gpu);
    assert_clean(&gpu);
}

struct RefractionProbe {
    color: Vec<u8>,
    surface: Vec<u8>,
    /// The last finished frame's rows and units the cull passed.
    counts: Option<(u32, u32)>,
}

struct WaterDart {
    depth: f32,
    radius: f32,
    /// The row's `ATTR_VIEW` (`RowState`: 0 own, 2 blip, 3 ghost).
    view: u32,
    /// Drawn as a red box model, not a dart.
    model: bool,
}

/// A red box, one unit radius wide, with one bone.
fn red_box_model() -> Arc<ae_model::Model> {
    let identity = glam::Mat4::IDENTITY.to_cols_array();
    let mut model = ae_model::Model {
        bones: vec![ae_model::Bone {
            name: "root".into(),
            parent: None,
            rest: identity,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }],
        joints: vec![ae_model::Joint {
            bone: 0,
            inverse_bind: identity,
        }],
        materials: vec![ae_model::Material {
            name: "red".into(),
            base_color: [1.0, 0.005, 0.004, 1.0],
            metallic: 0.0,
            roughness: 0.8,
            ..Default::default()
        }],
        ..Default::default()
    };
    model_box(&mut model, [-0.5, 0.0, -0.5], [0.5, 0.6, 0.5], 0);
    model.bounds = ae_model::Bounds {
        min: [-0.5, 0.0, -0.5],
        max: [0.5, 0.6, 0.5],
    };
    model.validate().unwrap();
    Arc::new(model)
}

fn refraction_probe(
    gpu: &Arc<GpuDevice>,
    depth: f32,
    water: ae_render::water::WaterParams,
    dart: Option<WaterDart>,
    shadows: bool,
) -> RefractionProbe {
    use ae_framegraph::{ResourceId, Role};
    use ae_render::library::{LibraryPass, WaterTiers};
    use ae_render::terrain::app::{NATIVE, PassWork, ScenePass, TerrainApp, TerrainOutput, TerrainSource};
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;
    use ae_render::thread::{RenderApp, TargetInfo};
    use ae_render::units::{InstanceStream, UnitClass, UnitFeed, UnitsSetup};

    let (heightmap, scale, offset) = Heightmap::from_cells((32, 32), |_, _| -depth, None);
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let info = TerrainInfo::new(&heightmap, &pyramid, 4.0, scale, offset, Some(0.0));
    let source = Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    });
    let mut app = TerrainApp::new(source, TerrainOutput::Readback)
        .with_water()
        .with_water_tiers(WaterTiers {
            refract: true,
            ..Default::default()
        })
        .with_water_params(water);
    if let Some(dart) = dart {
        let feed = Arc::new(UnitFeed::default());
        let xy = 64.0f32.to_bits();
        let columns = [
            xy,
            xy,
            xy,
            xy,
            0,
            0,
            0,
            1.0f32.to_bits(),
            0,
            0,
            0,
            1.0f32.to_bits(),
            dart.radius.to_bits(),
            0,
            ae_render::units::NO_POSE,
            dart.depth.to_bits(),
            dart.view,
        ]
        .map(|word| vec![word]);
        feed.publish(Arc::new(InstanceStream {
            columns,
            tick: 1,
            published: std::time::Instant::now(),
            tick_seconds: 0.0,
            sim_seconds: 0.0,
            anim: Default::default(),
        }));
        app = app.with_units(UnitsSetup {
            feed,
            capacity: 4,
            classes: vec![UnitClass {
                color: [1.0, 0.005, 0.004],
                gamut: ae_render::colorspace::Gamut::Bt709,
                floats: true,
            }],
            min_pixels: 0.0,
            models: if dart.model {
                vec![Some(red_box_model())]
            } else {
                Vec::new()
            },
            rigs: Vec::new(),
        });
    }
    if shadows {
        app = app.with_shadows();
    }
    {
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera.target = glam::Vec2::splat(64.0);
        view.camera.distance = 62.0;
        view.camera.pitch = 0.8;
        view.camera.yaw = 0.0;
        view.time = Some(0.0);
        view.fog = false;
    }
    let extent = vk::Extent2D {
        width: 192,
        height: 144,
    };
    app.create_offscreen(gpu, extent).unwrap();
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let mut executor = Executor::new(
        gpu.clone(),
        ExecutorConfig {
            profile: false,
            frames_in_flight: 2,
            record_threads: 2,
            ..Default::default()
        },
    )
    .unwrap();
    app.create(gpu, executor.bindless_mut(), target, 2).unwrap();
    let (mut graph, _) = app.declare(target, executor.bindless_mut()).unwrap();
    let image = ResourceId(
        graph
            .resources
            .iter()
            .position(|r| r.name == "water.refraction")
            .unwrap() as u32,
    );
    let desc = BufferDesc {
        size: u64::from(extent.width) * u64::from(extent.height) * 8,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
    };
    let buffer = graph.buffer("refraction probe", desc, ResourceClass::External);
    graph
        .pass(
            "refraction probe copy",
            PassKind::Transfer,
            ScenePass {
                program: NATIVE,
                work: PassWork::Readback { image, buffer },
            },
        )
        .access(image, Role::TransferSrc)
        .discard(buffer, Role::TransferDst)
        .id();
    graph
        .pass(
            "refraction probe read",
            PassKind::Transfer,
            ScenePass {
                program: NATIVE,
                work: PassWork::Library(LibraryPass::HostRead),
            },
        )
        .access(buffer, Role::HostRead)
        .root();
    let mut readback = OwnedBuffer::new(gpu, desc, MemoryLocation::GpuToCpu, "refraction probe").unwrap();
    executor.set_graph(graph).unwrap();
    for _ in 0..4 {
        let token = executor.begin_frame().unwrap();
        app.prepare(executor.bindless_mut(), token.slot).unwrap();
        let mut externals = (0..executor.graph().resources.len() as u32)
            .map(|index| app.external(ResourceId(index), token.slot))
            .collect::<Vec<_>>();
        externals[buffer.index()] = Some(Binding::Buffer(BufferBinding {
            buffer: readback.buffer,
            size: desc.size,
            offset: 0,
            identity: 0x52454652,
            storage: None,
        }));
        executor
            .execute(token, app.recorder(), &|id| externals[id.index()], &[], &[])
            .unwrap();
        executor.wait(token).unwrap();
    }
    let color = readback.mapped().unwrap().to_vec();
    let surface = app.readback().unwrap();
    let counts = app.last_unit_counts();
    drop(executor);
    app.destroy(gpu);
    readback.destroy(gpu);
    RefractionProbe { color, surface, counts }
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn refractive_water_reconstructs_seabed_depth_and_matches_absorption() {
    use ae_render::water::WaterParams;
    let gpu = gpu();
    let absorption = [0.035_f32, 0.075, 0.16];
    let plain = WaterParams {
        absorption: [0.0; 3],
        caustics: 0.0,
        ..Default::default()
    };
    let tint = WaterParams { absorption, ..plain };
    let exposure = ae_render::tonemap::ToneParams::default().exposure();
    for depth in [1.0_f32, 4.0, 16.0] {
        let scene = refraction_probe(&gpu, depth, plain, None, false);
        let refracted = refraction_probe(&gpu, depth, tint, None, false);
        let deep_mix = 1.0 - (-depth * absorption.iter().sum::<f32>() * 0.1).exp();
        let mut count = 0;
        for (base, actual) in scene
            .color
            .as_chunks::<8>()
            .0
            .iter()
            .zip(refracted.color.as_chunks::<8>().0)
        {
            if base[6..8] != [0, 0x3c] {
                continue;
            }
            assert_eq!(actual[6..8], [0, 0x3c]);
            let base = scrgb_pixel(base, 0);
            let actual = scrgb_pixel(actual, 0);
            for i in 0..3 {
                let t = (-depth * tint.absorption[i]).exp();
                let scatter = tint.shallow[i] + deep_mix * (tint.deep[i] - tint.shallow[i]);
                let want = base[i] * t + (1.0 - t) * scatter * exposure;
                // Half-float readback and the reverse-Z world reconstruction.
                assert!(
                    (actual[i] - want).abs() <= want * 0.004 + 0.001,
                    "{depth} m channel {i}: {} vs {want}",
                    actual[i]
                );
            }
            count += 1;
        }
        assert!(count > 8000, "only {count} seabed samples");
        assert_ne!(
            scene.surface, refracted.surface,
            "the surface must consume the underwater result"
        );
    }
    assert_eq!(gpu.instance.counters.errors(), 0);
    assert_eq!(gpu.instance.counters.warnings(), 0);
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn refractive_water_masks_above_water_hulls_and_shadows_caustics() {
    use ae_render::water::WaterParams;
    let gpu = gpu();
    let plain = WaterParams {
        caustics: 0.0,
        ..Default::default()
    };
    // A coarse web (a 33 m tile): the underwater pass fades out threads
    // finer than a few pixels, and the probe looks from RTS height.
    let web = WaterParams {
        caustics: 0.03,
        ..plain
    };
    let hull = || {
        Some(WaterDart {
            depth: 0.0,
            radius: 8.0,
            view: 0,
            model: false,
        })
    };
    let open = refraction_probe(&gpu, 4.0, plain, hull(), false);
    let open_caustics = refraction_probe(&gpu, 4.0, web, hull(), false);
    let shadow = refraction_probe(&gpu, 4.0, plain, hull(), true);
    let shadow_caustics = refraction_probe(&gpu, 4.0, web, hull(), true);
    let absent = refraction_probe(&gpu, 4.0, web, None, true);
    let mut masked = 0;
    let mut brightened = 0;
    let mut suppressed = 0;
    for i in (0..open.color.len()).step_by(8) {
        if open.color[i + 6..i + 8] == [0, 0] && absent.color[i + 6..i + 8] == [0, 0x3c] {
            assert_eq!(
                open_caustics.color[i..i + 8],
                [0; 8],
                "above-water geometry enters the copy"
            );
            masked += 1;
        }
        let y = |data: &RefractionProbe| y709(scrgb_pixel(&data.color, i));
        let lit_caustics = y(&open_caustics) - y(&open);
        if lit_caustics > 0.008 {
            brightened += 1;
            if y(&shadow) > 0.0
                && y(&shadow) < y(&open) * 0.9
                && (y(&shadow_caustics) - y(&shadow)).abs() < lit_caustics * 0.2
            {
                suppressed += 1;
            }
        }
    }
    assert!(
        masked > 30 && brightened > 100 && suppressed > 5,
        "masked {masked}, lit {brightened}, suppressed {suppressed}"
    );
    assert_eq!(gpu.instance.counters.errors(), 0);
    assert_eq!(gpu.instance.counters.warnings(), 0);
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn refraction_frames_draw_blips_and_ghosts_over_the_water_only() {
    use ae_render::water::WaterParams;
    let gpu = gpu();
    let plain = WaterParams {
        caustics: 0.0,
        ..Default::default()
    };
    let hull = |view, model| {
        Some(WaterDart {
            depth: 0.0,
            radius: 8.0,
            view,
            model,
        })
    };
    let absent = refraction_probe(&gpu, 4.0, plain, None, false);
    let red = |p: &RefractionProbe| {
        p.surface
            .chunks(4)
            .filter(|c| c[0] > 60 && u32::from(c[0]) > 3 * u32::from(c[1]))
            .count()
    };
    assert_eq!(red(&absent), 0);
    // The SUB cull puts an Own row in the scene's list and a blip or ghost
    // in the water overlay's (the darts' and the models' commands): each
    // draws once, on the surface, and the count report has it either way.
    for (view, model, what) in [
        (0, false, "own dart"),
        (2, false, "blip dart"),
        (3, false, "ghost dart"),
        (0, true, "own model"),
        (2, true, "blip model"),
        (3, true, "ghost model"),
    ] {
        let probe = refraction_probe(&gpu, 4.0, plain, hull(view, model), false);
        assert!(red(&probe) > 50, "{what}: {} red pixels", red(&probe));
        assert_eq!(probe.counts, Some((1, 1)), "{what}");
        if view != 0 {
            assert!(probe.color == absent.color, "{what} enters the refraction copy");
        }
    }
    assert_eq!(gpu.instance.counters.errors(), 0);
    assert_eq!(gpu.instance.counters.warnings(), 0);
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn submerged_dart_keeps_paint_and_gets_only_the_requested_outline() {
    use ae_render::water::WaterParams;
    let gpu = gpu();
    let no_tint = WaterParams {
        caustics: 0.0,
        absorption: [0.0; 3],
        ..Default::default()
    };
    let tint = WaterParams {
        absorption: [0.1, 0.2, 0.5],
        shallow: [0.0; 3],
        deep: [0.0; 3],
        sub_keep: 0.6,
        sub_tint: 4.0,
        sub_tint_max: 3.0,
        ..no_tint
    };
    let sub = || {
        Some(WaterDart {
            depth: 4.0,
            radius: 2.6,
            view: 0,
            model: false,
        })
    };
    let base = refraction_probe(&gpu, 10.0, no_tint, sub(), false);
    let painted = refraction_probe(&gpu, 10.0, tint, sub(), false);
    let outlined_tint = WaterParams {
        outline: true,
        ..no_tint
    };
    let outlined = refraction_probe(&gpu, 10.0, outlined_tint, sub(), false);
    let mut sub_pixels = 0;
    let mut edge_pixels = 0;
    for i in (0..base.color.len()).step_by(8) {
        let b = scrgb_pixel(&base.color, i);
        let p = scrgb_pixel(&painted.color, i);
        if b[0] > 4.0 * b[1] && b[0] > 0.08 {
            sub_pixels += 1;
            for j in 0..3 {
                let want = b[j] * (-tint.absorption[j] * 3.0).exp().max(0.6);
                assert!(
                    (p[j] - want).abs() < 0.002 + want * 0.004,
                    "channel {j}: {} vs {want}",
                    p[j]
                );
            }
        }
        if outlined.color[i..i + 8] != base.color[i..i + 8] {
            edge_pixels += 1;
        }
    }
    assert!(
        sub_pixels > 12 && edge_pixels > 3 && edge_pixels < sub_pixels,
        "{sub_pixels} hull pixels, {edge_pixels} edge pixels"
    );
    assert_eq!(gpu.instance.counters.errors(), 0);
    assert_eq!(gpu.instance.counters.warnings(), 0);
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn surface_refraction_grows_with_depth_and_rejects_dry_or_bright_samples() {
    use ae_framegraph::{ResourceId, Role};
    use ae_render::bindless::{PUSH_STAGES, Push};
    use ae_render::executor::{PassContext, Recorder};

    struct Shader {
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        out: ResourceId,
    }

    impl Recorder<()> for Shader {
        fn record(&mut self, _: &(), ctx: &PassContext<'_>) -> anyhow::Result<()> {
            let mut push = Push::default();
            push.words[0] = ctx.buffer(self.out)?.storage.unwrap();
            unsafe {
                ctx.device
                    .cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
                ctx.device.cmd_bind_descriptor_sets(
                    ctx.cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[ctx.global_set],
                    &[],
                );
                ctx.device.cmd_push_constants(
                    ctx.cmd,
                    self.layout,
                    PUSH_STAGES,
                    0,
                    ae_render::pipeline::push_bytes(&push),
                );
                ctx.device.cmd_dispatch(ctx.cmd, 1, 1, 1);
            }
            Ok(())
        }
    }

    let gpu = gpu();
    let mut executor = Executor::new(
        gpu.clone(),
        ExecutorConfig {
            profile: false,
            ..Default::default()
        },
    )
    .unwrap();
    let mut graph = FrameGraph::new();
    let desc = BufferDesc {
        size: 32,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
    };
    let out = graph.buffer("refraction offset and guard", desc, ResourceClass::External);
    graph
        .pass("probe", PassKind::Compute, ())
        .access(out, Role::StorageBufferWrite(ShaderStage::Compute))
        .id();
    graph
        .pass("host read", PassKind::Transfer, ())
        .access(out, Role::HostRead)
        .root();
    let mut buffer = OwnedBuffer::new(&gpu, desc, MemoryLocation::GpuToCpu, "refraction guard probe").unwrap();
    let storage = executor.bindless_mut().whole_buffer(&buffer).unwrap();
    let layout = executor.bindless().pipeline_layout();
    let pipeline =
        ae_render::pipeline::compute_pipeline(&gpu.device, ae_render::spirv!("test_water_refract.comp"), layout)
            .unwrap();
    let mut shader = Shader { pipeline, layout, out };
    executor.set_graph(graph).unwrap();
    let token = executor.begin_frame().unwrap();
    let binding = Binding::Buffer(BufferBinding {
        buffer: buffer.buffer,
        size: 32,
        offset: 0,
        identity: 0x52454647,
        storage: Some(storage.index),
    });
    executor
        .execute(token, &mut shader, &|id| (id == out).then_some(binding), &[], &[])
        .unwrap();
    executor.wait(token).unwrap();
    let values: Vec<_> = buffer.mapped().unwrap()[..32]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert_eq!(values[0], 0.0);
    assert!(values[..4].windows(2).all(|w| w[0] < w[1]), "{values:?}");
    assert_eq!(values[4..7], [1.0, 0.0, 0.0]);
    assert!(values[7] > 0.0 && values[7] < 1.0);
    unsafe { gpu.device.destroy_pipeline(pipeline, None) };
    drop(executor);
    buffer.destroy(&gpu);
    assert_clean(&gpu);
}

fn volume_test_source() -> Arc<ae_render::terrain::app::TerrainSource> {
    use ae_render::terrain::app::TerrainSource;
    use ae_render::terrain::heightmap::{Heightmap, build_pyramid};
    use ae_render::terrain::renderer::TerrainInfo;

    let (heightmap, scale, offset) = Heightmap::from_cells((64, 64), |_, _| -24.0, None);
    let pyramid = build_pyramid(&heightmap, scale, offset);
    let info = TerrainInfo::new(&heightmap, &pyramid, 8.0, scale, offset, Some(0.0));
    Arc::new(TerrainSource {
        heightmap,
        pyramid,
        info,
        materials: None,
    })
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn water_volume_shows_snells_window_and_crosses_the_near_plane_smoothly() {
    use ae_render::library::WaterTiers;
    use ae_render::terrain::app::{TerrainApp, TerrainOutput};
    use ae_render::thread::{Headless, RenderApp, TargetInfo};
    use glam::Vec2;

    let gpu = gpu();
    let source = volume_test_source();
    let extent = vk::Extent2D {
        width: 161,
        height: 161,
    };
    let target = TargetInfo {
        extent,
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    let config = ExecutorConfig {
        profile: false,
        record_threads: 2,
        ..Default::default()
    };
    let full = WaterTiers {
        reflect: true,
        refract: true,
        waves: true,
        volume: true,
    };
    let init = |tiers| {
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback)
            .with_water()
            .with_water_tiers(tiers);
        let view = app.view();
        {
            let mut v = view.lock().unwrap();
            v.time = Some(0.0);
            v.fog = false;
            v.camera.target = Vec2::splat(256.0);
            v.camera.yaw = 0.0;
            v.camera.distance = 28.0;
        }
        app.create_offscreen(&gpu, extent).unwrap();
        let headless = Headless::new(&gpu, &mut app, target, config).unwrap();
        (app, headless)
    };
    let (mut app, mut headless) = init(full);
    let (mut surface_only, mut reference) = init(WaterTiers { volume: false, ..full });
    let place = |app: &TerrainApp, eye_height, pitch, fov_y| {
        let view = app.view();
        let mut view = view.lock().unwrap();
        view.camera.pitch = pitch;
        view.camera.fov_y = fov_y;
        view.camera_focus = Some(eye_height - view.camera.offset_dir().y * view.camera.distance);
    };

    place(&app, 4.0, 0.6, 0.8);
    place(&surface_only, 4.0, 0.6, 0.8);
    headless.frames(&mut app, 8).unwrap();
    reference.frames(&mut surface_only, 8).unwrap();
    headless.wait().unwrap();
    reference.wait().unwrap();
    assert!(
        app.readback() == surface_only.readback(),
        "above-water output must be byte-identical"
    );

    place(&app, -3.0, -1.5, 2.16);
    place(&surface_only, -3.0, -1.5, 2.16);
    headless.frames(&mut app, 6).unwrap();
    reference.frames(&mut surface_only, 6).unwrap();
    headless.wait().unwrap();
    reference.wait().unwrap();
    let below = app.readback().unwrap();
    let sky = surface_only.readback().unwrap();
    let pixel = |x, y| ((y * extent.width + x) * 4) as usize;
    let center = pixel(extent.width / 2, extent.height / 2);
    let sky_color = &sky[center..center + 3];
    assert!(sky.as_chunks::<4>().0.iter().filter(|p| &p[..3] == sky_color).count() > below.len() / 8);
    assert!(
        below.as_chunks::<4>().0.iter().all(|p| &p[..3] != sky_color),
        "no clear sky below the water"
    );
    let center_color = &below[center..center + 3];
    let outside = &below[0..3];
    assert!(
        center_color[1] >= outside[1] + 20,
        "Snell's window {center_color:?}, internal reflection {outside:?}"
    );
    assert!(
        outside[2] > outside[0],
        "water in-scatter outside the window {outside:?}"
    );

    let mut previous: Option<Vec<u8>> = None;
    // Per step: the channels that changed by more than 48/255, and where.
    let mut max_step = (0, 0, 0, 0);
    for step in 0..=16 {
        place(&app, 2.0 - step as f32 * 0.25, -0.37, 0.8);
        headless.frames(&mut app, 2).unwrap();
        headless.wait().unwrap();
        let frame = app.readback().unwrap();
        if let Some(previous) = previous {
            let jumps: Vec<usize> = frame
                .iter()
                .zip(previous)
                .enumerate()
                .filter(|(_, (a, b))| a.abs_diff(*b) > 48)
                .map(|(i, _)| i / 4 / extent.width as usize)
                .collect();
            if jumps.len() > max_step.0 {
                let rows = (jumps.iter().min().copied(), jumps.iter().max().copied());
                max_step = (jumps.len(), step, rows.0.unwrap_or(0), rows.1.unwrap_or(0));
            }
        }
        previous = Some(frame);
    }
    RenderApp::destroy(&mut app, &gpu);
    RenderApp::destroy(&mut surface_only, &gpu);
    drop((headless, reference));
    assert_clean(&gpu);
    // A 25 cm step moves the waterline's band across the near plane (the
    // full tier's swell and static octaves make the surface at the eye
    // wavy, so the band is a few rows deep), but never pops the frame: at
    // most 2% of the channels change by more than 48/255.
    let budget = (extent.width * extent.height * 4) as usize / 50;
    assert!(
        max_step.0 <= budget,
        "largest 25 cm step changed {} channels by more than 48 (step, rows): {max_step:?}",
        max_step.0
    );
}

#[test]
#[ignore = "needs a Vulkan 1.3 GPU"]
fn water_quality_draws_and_resizes_with_the_surface_enabled_and_disabled() {
    use ae_framegraph::ResourceId;
    use ae_render::library::WaterTiers;
    use ae_render::terrain::app::{TerrainApp, TerrainOutput};
    use ae_render::thread::{RenderApp, TargetInfo};

    let gpu = gpu();
    let source = volume_test_source();
    let mut off = [Vec::new(), Vec::new()];
    for (water, tiers) in [
        (false, WaterTiers::default()),
        (true, WaterTiers::default()),
        (
            true,
            WaterTiers {
                reflect: true,
                ..Default::default()
            },
        ),
        (
            true,
            WaterTiers {
                refract: true,
                ..Default::default()
            },
        ),
        (
            true,
            WaterTiers {
                reflect: true,
                refract: true,
                waves: true,
                volume: false,
            },
        ),
        (
            true,
            WaterTiers {
                reflect: true,
                refract: true,
                waves: true,
                volume: true,
            },
        ),
    ] {
        let mut app = TerrainApp::new(source.clone(), TerrainOutput::Readback).with_water_tiers(tiers);
        if water {
            app = app.with_water();
        }
        let view = app.view();
        view.lock().unwrap().time = Some(0.0);
        let mut executor = Executor::new(
            gpu.clone(),
            ExecutorConfig {
                profile: false,
                record_threads: 2,
                ..Default::default()
            },
        )
        .unwrap();
        for (index, (width, height)) in [(96, 72), (117, 81)].into_iter().enumerate() {
            let target = TargetInfo {
                extent: vk::Extent2D { width, height },
                format: vk::Format::R8G8B8A8_SRGB,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            };
            if index == 0 {
                app.create(&gpu, executor.bindless_mut(), target, 2).unwrap();
            } else {
                assert!(app.retarget(&gpu, executor.bindless_mut(), target, 2).unwrap());
                executor.forget_external_states();
            }
            app.create_offscreen(&gpu, target.extent).unwrap();
            let (graph, _) = app.declare(target, executor.bindless_mut()).unwrap();
            assert_eq!(app.water_tiers(), tiers);
            executor.set_graph(graph).unwrap();
            let mut render = |enabled| {
                view.lock().unwrap().water = enabled;
                for _ in 0..6 {
                    let token = executor.begin_frame().unwrap();
                    app.prepare(executor.bindless_mut(), token.slot).unwrap();
                    let bindings = (0..executor.graph().resources.len() as u32)
                        .map(|id| app.external(ResourceId(id), token.slot))
                        .collect::<Vec<_>>();
                    executor
                        .execute(token, app.recorder(), &|id| bindings[id.index()], &[], &[])
                        .unwrap();
                    executor.wait(token).unwrap();
                }
                app.readback().unwrap()
            };
            let enabled = render(water);
            assert_eq!(enabled.len(), (width * height * 4) as usize);
            if water {
                let disabled = render(false);
                assert!(
                    disabled == off[index],
                    "{tiers:?} after water switch at {width}×{height}"
                );
                assert!(enabled != disabled, "{tiers:?} draws the surface at {width}×{height}");
            } else {
                off[index] = enabled;
            }
        }
        RenderApp::destroy(&mut app, &gpu);
    }
    assert_clean(&gpu);
}
