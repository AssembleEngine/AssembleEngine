//! Micro benches for the frame graph core: `cargo bench -p ae_framegraph`.

use ae_framegraph::barriers::plan_frame;
use ae_framegraph::{
    AccessState, BakeOptions, FrameGraph, ImageDesc, LoadOp, PassKind, ResourceClass, Role, ShaderStage, StateKey, bake,
};
use ash::vk;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::collections::BTreeMap;
use std::hint::black_box;

/// A straight chain: each compute pass reads the previous image and writes the next.
fn chain(passes: usize) -> FrameGraph {
    let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
    let mut graph = FrameGraph::new();
    let mut previous = None;
    for index in 0..passes {
        let image = graph.image(
            format!("t{index}"),
            ImageDesc::color_2d(vk::Format::R16G16B16A16_SFLOAT, 512, 512, usage),
            ResourceClass::Transient,
        );
        let mut pass = graph.pass(format!("p{index}"), PassKind::Compute, ());
        if let Some(previous) = previous {
            pass = pass.access(previous, Role::StorageImageRead(ShaderStage::Compute));
        }
        pass.discard(image, Role::StorageImageWrite(ShaderStage::Compute)).id();
        previous = Some(image);
    }
    let back = graph.image(
        "backbuffer",
        ImageDesc::color_2d(vk::Format::B8G8R8A8_UNORM, 1920, 1080, vk::ImageUsageFlags::COLOR_ATTACHMENT),
        ResourceClass::Backbuffer,
    );
    graph.pass("composite", PassKind::Graphics, ()).sampled(previous.unwrap(), ShaderStage::Fragment).color(back, LoadOp::Clear).id();
    graph.pass("present", PassKind::Present, ()).present(back).root().id();
    graph
}

fn benches(c: &mut Criterion) {
    for passes in [16usize, 128] {
        let graph = chain(passes);
        let baked = bake(&graph, BakeOptions::default()).unwrap();
        c.bench_with_input(BenchmarkId::new("signature", passes), &graph, |b, graph| b.iter(|| black_box(graph).signature()));
        c.bench_with_input(BenchmarkId::new("bake", passes), &graph, |b, graph| {
            b.iter(|| bake(black_box(graph), BakeOptions::default()))
        });
        c.bench_with_input(BenchmarkId::new("plan_barriers", passes), &baked, |b, baked| {
            let mut store = BTreeMap::<StateKey, AccessState>::new();
            b.iter(|| plan_frame(black_box(baked), &mut store, |_| false))
        });
    }
}

criterion_group!(group, benches);
criterion_main!(group);
