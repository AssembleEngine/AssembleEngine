//! Frame graph bench: a synthetic layered graph, no GPU.

use super::Options;
use crate::measure::time;
use crate::record::Report;
use ae_framegraph::{
    AccessState, BakeOptions, descriptor, FrameGraph, ImageDesc, LoadOp, PassKind, ResourceClass, Role, ShaderStage, StateKey, bake,
    barriers::plan_frame,
};
use anyhow::{Result, anyhow};
use ash::vk;
use std::collections::BTreeMap;

/// `layers` rows of `width` compute passes. Every pass writes one transient
/// image and (after the first row) reads two images of the row above; a final
/// graphics pass samples the last row into the backbuffer, then present.
pub fn layered_graph(layers: usize, width: usize) -> FrameGraph {
    let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
    let mut graph = FrameGraph::new();
    let mut above: Vec<_> = Vec::new();
    for layer in 0..layers {
        let mut row = Vec::with_capacity(width);
        for column in 0..width {
            // Three distinct sizes, so aliasing has to match descriptions.
            let size = 256 << (column % 3);
            let image = graph.image(
                format!("t{layer}_{column}"),
                ImageDesc::color_2d(vk::Format::R16G16B16A16_SFLOAT, size, size, usage),
                ResourceClass::Transient,
            );
            let mut pass = graph.pass(format!("p{layer}_{column}"), PassKind::Compute, ());
            if layer > 0 {
                pass = pass
                    .access(above[column], Role::StorageImageRead(ShaderStage::Compute))
                    .access(above[(column + 1) % width], Role::StorageImageRead(ShaderStage::Compute));
            }
            pass.discard(image, Role::StorageImageWrite(ShaderStage::Compute)).id();
            row.push(image);
        }
        above = row;
    }
    let back = graph.image(
        "backbuffer",
        ImageDesc::color_2d(vk::Format::B8G8R8A8_UNORM, 1920, 1080, vk::ImageUsageFlags::COLOR_ATTACHMENT),
        ResourceClass::Backbuffer,
    );
    let mut composite = graph.pass("composite", PassKind::Graphics, ());
    for &image in &above {
        composite = composite.sampled(image, ShaderStage::Fragment);
    }
    composite.color(back, LoadOp::Clear).id();
    graph.pass("present", PassKind::Present, ()).present(back).root().id();
    graph
}

pub fn run(options: &Options) -> Result<Report> {
    let mut report = Report::new("framegraph-bake");
    let shapes: &[(usize, usize)] = if options.small { &[(8, 4)] } else { &[(8, 4), (32, 8), (64, 16)] };
    for &(layers, width) in shapes {
        ae_profile::zone!("framegraph-bake shape");
        let tag = format!("{}p", layers * width + 2);
        let graph = layered_graph(layers, width);
        let baked = bake(&graph, BakeOptions::default()).map_err(|e| anyhow!("bake failed: {e}"))?;

        report.info(&format!("{tag}/passes"), baked.passes.len() as u64, "");
        report.info(&format!("{tag}/resources"), baked.resources.len() as u64, "");
        report.counter(&format!("{tag}/physical_resources"), baked.physical.len() as u64, "");
        report.counter(&format!("{tag}/estimated_bytes"), baked.estimated_bytes(), "B");
        let barriers = plan_frame(&baked, &mut BTreeMap::<StateKey, AccessState>::new(), |_| false);
        report.counter(&format!("{tag}/barriers_first_frame"), barriers.iter().map(Vec::len).sum::<usize>() as u64, "");

        report.time(&format!("{tag}/declare"), time(options.budget, || layered_graph(layers, width)));
        // A render module hands the host this declaration; the host decodes it before baking.
        let bytes = descriptor::encode(&graph, |_| 0);
        report.info(&format!("{tag}/descriptor_bytes"), bytes.len() as u64, "B");
        report.time(&format!("{tag}/descriptor_decode"), time(options.budget, || descriptor::decode(&bytes)));
        report.time(&format!("{tag}/signature"), time(options.budget, || graph.signature()));
        report.time(&format!("{tag}/bake"), time(options.budget, || bake(&graph, BakeOptions::default())));
        report.time(&format!("{tag}/bake_no_alias"), time(options.budget, || bake(&graph, BakeOptions { aliasing: false })));
        // Steady state: the store carries last frame's states into this one.
        let mut store = BTreeMap::<StateKey, AccessState>::new();
        report.time(&format!("{tag}/plan_barriers"), time(options.budget, || plan_frame(&baked, &mut store, |_| false)));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layered_graph_bakes_and_aliases() {
        let graph = layered_graph(8, 4);
        let baked = bake(&graph, BakeOptions::default()).unwrap();
        assert_eq!(baked.passes.len(), 8 * 4 + 2, "no pass may be pruned");
        let unaliased = bake(&graph, BakeOptions { aliasing: false }).unwrap();
        assert!(baked.physical.len() < unaliased.physical.len(), "aliasing should merge transients");
    }
}
