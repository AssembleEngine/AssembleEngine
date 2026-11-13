//! Render ABI spike: a frame graph declared by a render module through
//! `ae_render_abi::GraphBuilder` must decode into exactly the graph the
//! engine would declare natively (same topology signature, same bake, same
//! dot export). The reference is the ecs_sim demo graph
//! (`~/ecs_sim/src/render/demo.rs`, `build_demo_graph`), reproduced here.

use ae_framegraph::descriptor::{DescriptorError, decode, encode};
use ae_framegraph::dot::to_dot;
use ae_framegraph::{
    BakeOptions, BufferDesc, FrameGraph, ImageDesc, LoadOp, PassKind, ResourceClass, Role, ShaderStage, bake,
};
use ae_render_abi as abi;
use ash::vk;

const SCENE_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const PARAMS_BYTES: u64 = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Output {
    Present(vk::Format),
    Readback,
}

/// `build_demo_graph` from the ecs_sim prototype, with `()` payloads.
fn native_demo(width: u32, height: u32, output: Output) -> FrameGraph {
    let mut graph = FrameGraph::new();
    let work_usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
    let scene = graph.image(
        "scene",
        ImageDesc::color_2d(SCENE_FORMAT, width, height, vk::ImageUsageFlags::COLOR_ATTACHMENT | work_usage),
        ResourceClass::Transient,
    );
    let work = |graph: &mut FrameGraph, name: &str| {
        graph.image(name, ImageDesc::color_2d(SCENE_FORMAT, width, height, work_usage), ResourceClass::Transient)
    };
    let params_staging = graph.buffer(
        "params staging",
        BufferDesc { size: PARAMS_BYTES, usage: vk::BufferUsageFlags::TRANSFER_SRC },
        ResourceClass::External,
    );
    let params = graph.buffer(
        "params",
        BufferDesc { size: PARAMS_BYTES, usage: vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST },
        ResourceClass::Transient,
    );
    let a1 = work(&mut graph, "a1");
    let a2 = work(&mut graph, "a2");
    let b1 = work(&mut graph, "b1");
    let b2 = work(&mut graph, "b2");
    let (target, readback) = match output {
        Output::Present(format) => (
            graph.image(
                "backbuffer",
                ImageDesc::color_2d(format, width, height, vk::ImageUsageFlags::COLOR_ATTACHMENT),
                ResourceClass::Backbuffer,
            ),
            None,
        ),
        Output::Readback => {
            let target = graph.image(
                "target",
                ImageDesc::color_2d(SCENE_FORMAT, width, height, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC),
                ResourceClass::External,
            );
            let readback = graph.buffer(
                "readback",
                BufferDesc { size: width as u64 * height as u64 * 4, usage: vk::BufferUsageFlags::TRANSFER_DST },
                ResourceClass::External,
            );
            (target, Some(readback))
        }
    };
    graph.pass("upload params", PassKind::Transfer, ()).access(params_staging, Role::TransferSrc).discard(params, Role::TransferDst).id();
    graph.pass("triangle", PassKind::Graphics, ()).color(scene, LoadOp::Clear).id();
    let filter = |graph: &mut FrameGraph, name: &str, src, dst| {
        graph
            .pass(name, PassKind::Compute, ())
            .access(src, Role::StorageImageRead(ShaderStage::Compute))
            .discard(dst, Role::StorageImageWrite(ShaderStage::Compute))
            .access(params, Role::UniformBuffer(ShaderStage::Compute))
            .id();
    };
    filter(&mut graph, "blur_h", scene, a1);
    filter(&mut graph, "blur_v", a1, a2);
    filter(&mut graph, "invert", scene, b1);
    filter(&mut graph, "tint", b1, b2);
    graph
        .pass("composite", PassKind::Graphics, ())
        .sampled(a2, ShaderStage::Fragment)
        .sampled(b2, ShaderStage::Fragment)
        .color(target, LoadOp::DontCare)
        .id();
    match readback {
        Some(buffer) => {
            graph.pass("readback", PassKind::Transfer, ()).access(target, Role::TransferSrc).discard(buffer, Role::TransferDst).id();
            graph.pass("host read", PassKind::Transfer, ()).access(buffer, Role::HostRead).root().id();
        }
        None => {
            graph.pass("present", PassKind::Present, ()).present(target).root().id();
        }
    }
    graph
}

/// The same graph written the way a render module would: only
/// `ae_render_abi` and raw Vulkan numbers (an SDK would name them).
fn guest_demo(width: u32, height: u32, output: Output) -> Vec<u8> {
    use abi::{Class, GraphBuilder, ImageSpec, LoadOp, PassKind, Role, ShaderStage};
    const RGBA8: u32 = 37; // VK_FORMAT_R8G8B8A8_UNORM
    const IMG_TRANSFER_SRC: u32 = 0x1;
    const IMG_SAMPLED: u32 = 0x4;
    const IMG_STORAGE: u32 = 0x8;
    const IMG_COLOR: u32 = 0x10;
    const BUF_TRANSFER_SRC: u32 = 0x1;
    const BUF_TRANSFER_DST: u32 = 0x2;
    const BUF_UNIFORM: u32 = 0x10;

    let mut g = GraphBuilder::new();
    let work_usage = IMG_STORAGE | IMG_SAMPLED;
    let scene = g.image("scene", ImageSpec::color_2d(RGBA8, width, height, IMG_COLOR | work_usage), Class::Transient);
    let params_staging = g.buffer("params staging", PARAMS_BYTES, BUF_TRANSFER_SRC, Class::External);
    let params = g.buffer("params", PARAMS_BYTES, BUF_UNIFORM | BUF_TRANSFER_DST, Class::Transient);
    let mut work = |name| g.image(name, ImageSpec::color_2d(RGBA8, width, height, work_usage), Class::Transient);
    let (a1, a2, b1, b2) = (work("a1"), work("a2"), work("b1"), work("b2"));
    let (target, readback) = match output {
        Output::Present(format) => {
            let spec = ImageSpec::color_2d(format.as_raw() as u32, width, height, IMG_COLOR);
            (g.image("backbuffer", spec, Class::Backbuffer), None)
        }
        Output::Readback => {
            let target = g.image("target", ImageSpec::color_2d(RGBA8, width, height, IMG_COLOR | IMG_TRANSFER_SRC), Class::External);
            (target, Some(g.buffer("readback", width as u64 * height as u64 * 4, BUF_TRANSFER_DST, Class::External)))
        }
    };
    g.pass("upload params", PassKind::Transfer, 1).access(params_staging, Role::TransferSrc).discard(params, Role::TransferDst).id();
    g.pass("triangle", PassKind::Graphics, 2).color(scene, LoadOp::Clear).id();
    for (name, src, dst) in [("blur_h", scene, a1), ("blur_v", a1, a2), ("invert", scene, b1), ("tint", b1, b2)] {
        g.pass(name, PassKind::Compute, 3)
            .access(src, Role::StorageImageRead(ShaderStage::Compute))
            .discard(dst, Role::StorageImageWrite(ShaderStage::Compute))
            .access(params, Role::UniformBuffer(ShaderStage::Compute))
            .id();
    }
    g.pass("composite", PassKind::Graphics, 4)
        .sampled(a2, ShaderStage::Fragment)
        .sampled(b2, ShaderStage::Fragment)
        .color(target, LoadOp::DontCare)
        .id();
    match readback {
        Some(buffer) => {
            g.pass("readback", PassKind::Transfer, 5).access(target, Role::TransferSrc).discard(buffer, Role::TransferDst).id();
            g.pass("host read", PassKind::Transfer, 6).access(buffer, Role::HostRead).root().id();
        }
        None => {
            g.pass("present", PassKind::Present, 7).present(target).root().id();
        }
    }
    g.finish()
}

const OUTPUTS: [Output; 2] = [Output::Present(vk::Format::B8G8R8A8_SRGB), Output::Readback];

#[test]
fn guest_declaration_matches_native_graph() {
    for output in OUTPUTS {
        let native = native_demo(1280, 720, output);
        let bytes = guest_demo(1280, 720, output);
        let decoded = decode(&bytes).expect("decode");
        assert_eq!(decoded.signature(), native.signature(), "topology signatures differ");
        let (a, b) = (bake(&native, BakeOptions::default()).unwrap(), bake(&decoded, BakeOptions::default()).unwrap());
        assert_eq!(a, b, "bakes differ");
        assert_eq!(to_dot(&native, &a), to_dot(&decoded, &b), "dot exports differ");
        // Payloads arrive per pass, in declaration order.
        let payloads: Vec<u32> = decoded.passes.iter().map(|p| p.record).collect();
        assert_eq!(&payloads[..6], &[1, 2, 3, 3, 3, 3]);
    }
}

#[test]
fn a_topology_change_changes_the_signature() {
    let present = decode(&guest_demo(1280, 720, OUTPUTS[0])).unwrap().signature();
    let resized = decode(&guest_demo(1920, 1080, OUTPUTS[0])).unwrap().signature();
    let readback = decode(&guest_demo(1280, 720, OUTPUTS[1])).unwrap().signature();
    assert_ne!(present, resized);
    assert_ne!(present, readback);
    assert_eq!(present, decode(&guest_demo(1280, 720, OUTPUTS[0])).unwrap().signature(), "same declaration, same signature");
}

#[test]
fn encode_decode_round_trip() {
    for output in OUTPUTS {
        let native = native_demo(640, 360, output);
        let decoded = decode(&encode(&native, |_| 0)).unwrap();
        assert_eq!(decoded.signature(), native.signature());
        assert_eq!(bake(&decoded, BakeOptions::default()).unwrap(), bake(&native, BakeOptions::default()).unwrap());
    }
}

#[test]
fn malformed_declarations_are_rejected() {
    let good = guest_demo(64, 64, OUTPUTS[0]);
    let word = |bytes: &mut Vec<u8>, index: usize, value: u32| bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    let resource_word = |r: usize, k: usize| abi::HEADER_WORDS + r * abi::RESOURCE_WORDS + k;
    let resources = 8; // scene, params staging, params, a1, a2, b1, b2, backbuffer
    let pass_word = |p: usize, k: usize| abi::HEADER_WORDS + resources * abi::RESOURCE_WORDS + p * abi::PASS_WORDS + k;
    let passes = 8;
    let intent_word = |i: usize, k: usize| pass_word(passes, 0) + i * abi::INTENT_WORDS + k;

    let check = |patch: &dyn Fn(&mut Vec<u8>), expect: &dyn Fn(&DescriptorError) -> bool| {
        let mut bytes = good.clone();
        patch(&mut bytes);
        let error = decode(&bytes).expect_err("malformed declaration was accepted");
        assert!(expect(&error), "unexpected error: {error}");
    };
    check(&|b| b.truncate(20), &|e| matches!(e, DescriptorError::Truncated { .. }));
    check(&|b| b.push(0), &|e| matches!(e, DescriptorError::TrailingBytes(1)));
    check(&|b| word(b, 0, 0xDEAD), &|e| matches!(e, DescriptorError::BadMagic(_)));
    check(&|b| word(b, 1, 99), &|e| matches!(e, DescriptorError::UnsupportedVersion(99)));
    check(&|b| word(b, 4, u32::MAX), &|e| matches!(e, DescriptorError::TooMany { .. }));
    check(&|b| word(b, resource_word(0, 1), 9), &|e| matches!(e, DescriptorError::BadEnum { what: "resource class", .. }));
    check(&|b| word(b, resource_word(0, 6), 0), &|e| matches!(e, DescriptorError::BadImage { .. }));
    check(&|b| word(b, resource_word(0, 4), 10_000), &|e| matches!(e, DescriptorError::BadName { .. }));
    check(&|b| word(b, pass_word(0, 0), 7), &|e| matches!(e, DescriptorError::BadEnum { what: "pass kind", .. }));
    check(&|b| word(b, pass_word(0, 1), 4), &|e| matches!(e, DescriptorError::BadEnum { what: "pass flags", .. }));
    check(&|b| word(b, pass_word(1, 5), 0), &|e| matches!(e, DescriptorError::BadIntentRange { .. }));
    check(&|b| word(b, intent_word(0, 0), 500), &|e| matches!(e, DescriptorError::BadResource { .. }));
    check(&|b| word(b, intent_word(0, 1), 77), &|e| matches!(e, DescriptorError::BadEnum { what: "intent role", .. }));
    // TransferSrc takes no argument.
    check(&|b| word(b, intent_word(0, 2), 1), &|e| matches!(e, DescriptorError::BadEnum { what: "intent argument", .. }));
}

/// Random corruption must never panic, neither in decode nor in bake.
#[test]
fn corrupted_declarations_never_panic() {
    let good = guest_demo(256, 256, OUTPUTS[1]);
    let words = good.len() / 4;
    let mut state = 0x5EED_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let (mut decoded, mut baked) = (0, 0);
    for _ in 0..20_000 {
        let mut bytes = good.clone();
        for _ in 0..1 + next() % 4 {
            let index = (next() as usize) % words;
            let value = match next() % 4 {
                0 => (next() % 16) as u32,
                1 => u32::MAX,
                2 => (next() as u32) ^ u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap()),
                _ => 1 << (next() % 32),
            };
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        if next() % 8 == 0 {
            bytes.truncate((next() as usize) % bytes.len());
        }
        if let Ok(graph) = decode(&bytes) {
            decoded += 1;
            if bake(&graph, BakeOptions::default()).is_ok() {
                baked += 1;
            }
        }
    }
    eprintln!("20000 corrupted declarations: {decoded} decoded, {baked} also baked");
    assert!(decoded > 0, "mutations were too destructive to exercise bake");
}
