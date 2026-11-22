//! Barrier derivation: a per-resource access state machine that turns the
//! baked, declarative accesses into concrete `synchronization2` barriers at
//! record time (§3.5). Pure; the executor maps `BarrierDesc` to
//! `vk::ImageMemoryBarrier2` / `vk::BufferMemoryBarrier2`.
//!
//! State is keyed by *physical* allocation (`StateKey::Physical`), so an
//! alias handoff needs no state copy: the next logical member simply
//! continues the physical history (and starts with a discard). External and
//! backbuffer resources are keyed by logical id; the executor keeps one state
//! per bound object (per swapchain image, per frame slot).
//!
//! Rules (tightened relative to the baseline):
//! - a layout change always emits a barrier (old layout `UNDEFINED` on
//!   discard), sourced from the last write *and* the reads since it; a
//!   read-triggered change keeps the last write's access, so a later read
//!   at another stage still carries it;
//! - a write after reads or a write emits an execution/memory dependency
//!   (WAR/WAW);
//! - a read emits a barrier only if the last write is not yet visible to its
//!   stages and access; read-after-read emits nothing;
//! - nothing is reset per frame: persistent resources keep their layout,
//!   transients start over through their discarding first write.

use super::{BakedGraph, ResourceId, Usage};
use ash::vk;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessState {
    pub layout: vk::ImageLayout,
    /// Last write (or layout transition) that later accesses must follow.
    pub write_stages: vk::PipelineStageFlags2,
    pub write_access: vk::AccessFlags2,
    /// Stages that read since the last write (write-after-read sources).
    pub read_stages: vk::PipelineStageFlags2,
    /// Where the last write has already been made visible.
    pub visible_stages: vk::PipelineStageFlags2,
    pub visible_access: vk::AccessFlags2,
}

impl Default for AccessState {
    fn default() -> Self {
        Self {
            layout: vk::ImageLayout::UNDEFINED,
            write_stages: vk::PipelineStageFlags2::NONE,
            write_access: vk::AccessFlags2::NONE,
            read_stages: vk::PipelineStageFlags2::NONE,
            visible_stages: vk::PipelineStageFlags2::NONE,
            visible_access: vk::AccessFlags2::NONE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateKey {
    Physical(u32),
    External(ResourceId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BarrierDesc {
    pub resource: ResourceId,
    pub key: StateKey,
    pub image: bool,
    pub src_stages: vk::PipelineStageFlags2,
    pub src_access: vk::AccessFlags2,
    pub dst_stages: vk::PipelineStageFlags2,
    pub dst_access: vk::AccessFlags2,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
}

/// Where the executor keeps access states.
pub trait StateStore {
    fn state(&mut self, key: StateKey) -> &mut AccessState;
}

impl StateStore for std::collections::BTreeMap<StateKey, AccessState> {
    fn state(&mut self, key: StateKey) -> &mut AccessState {
        self.entry(key).or_default()
    }
}

/// Access implications: a broad read access covers the specific ones.
fn expand_access(access: vk::AccessFlags2) -> vk::AccessFlags2 {
    let mut out = access;
    if access.contains(vk::AccessFlags2::SHADER_READ) {
        out |= vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_READ;
    }
    if access.contains(vk::AccessFlags2::MEMORY_READ) {
        out |= vk::AccessFlags2::SHADER_READ
            | vk::AccessFlags2::SHADER_SAMPLED_READ
            | vk::AccessFlags2::SHADER_STORAGE_READ
            | vk::AccessFlags2::UNIFORM_READ
            | vk::AccessFlags2::TRANSFER_READ
            | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
            | vk::AccessFlags2::INDEX_READ
            | vk::AccessFlags2::INDIRECT_COMMAND_READ
            | vk::AccessFlags2::COLOR_ATTACHMENT_READ;
    }
    out
}

fn expand_stages(stages: vk::PipelineStageFlags2) -> vk::PipelineStageFlags2 {
    if stages.contains(vk::PipelineStageFlags2::ALL_COMMANDS) {
        return vk::PipelineStageFlags2::from_raw(!0);
    }
    let mut out = stages;
    if stages.contains(vk::PipelineStageFlags2::ALL_GRAPHICS) {
        out |= vk::PipelineStageFlags2::VERTEX_SHADER
            | vk::PipelineStageFlags2::FRAGMENT_SHADER
            | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
            | vk::PipelineStageFlags2::INDEX_INPUT
            | vk::PipelineStageFlags2::DRAW_INDIRECT
            | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
    }
    if stages.contains(vk::PipelineStageFlags2::ALL_TRANSFER) {
        out |= vk::PipelineStageFlags2::COPY
            | vk::PipelineStageFlags2::BLIT
            | vk::PipelineStageFlags2::RESOLVE
            | vk::PipelineStageFlags2::CLEAR;
    }
    out
}

/// Apply one access to `state`, returning the barrier it needs, if any.
/// `(src stages, src access, dst stages, dst access, old layout, new layout)`
/// in the returned `BarrierDesc` (resource and key filled by the caller).
pub fn transition(state: &mut AccessState, usage: Usage, image: bool, discard: bool) -> Option<BarrierDesc> {
    let make = |state: &AccessState, src_stages, src_access| BarrierDesc {
        resource: ResourceId(u32::MAX),
        key: StateKey::External(ResourceId(u32::MAX)),
        image,
        src_stages,
        src_access,
        dst_stages: usage.stages,
        dst_access: usage.access,
        old_layout: if !image || discard {
            vk::ImageLayout::UNDEFINED
        } else {
            state.layout
        },
        new_layout: if image {
            usage.layout
        } else {
            vk::ImageLayout::UNDEFINED
        },
    };
    let layout_change = image && state.layout != usage.layout;
    let barrier = if layout_change {
        // The transition is a write: it must follow every prior access.
        let barrier = make(state, state.write_stages | state.read_stages, state.write_access);
        state.layout = usage.layout;
        // A read-triggered transition keeps the last write as the source
        // for later reads at other stages (adding the transition's own
        // stage); a write-triggered one is replaced below.
        state.write_stages |= usage.stages;
        state.read_stages = vk::PipelineStageFlags2::NONE;
        state.visible_stages = usage.stages;
        state.visible_access = usage.access;
        Some(barrier)
    } else if usage.write {
        let prior = state.write_stages | state.read_stages;
        (!prior.is_empty() || !state.write_access.is_empty()).then(|| make(state, prior, state.write_access))
    } else {
        let has_write = !state.write_stages.is_empty() || !state.write_access.is_empty();
        let covered = expand_stages(state.visible_stages).contains(usage.stages)
            && expand_access(state.visible_access).contains(usage.access);
        if has_write && !covered {
            let barrier = make(state, state.write_stages, state.write_access);
            state.visible_stages |= usage.stages;
            state.visible_access |= usage.access;
            Some(barrier)
        } else {
            None
        }
    };
    if usage.write {
        state.write_stages = usage.stages;
        state.write_access = usage.access;
        state.read_stages = vk::PipelineStageFlags2::NONE;
        state.visible_stages = vk::PipelineStageFlags2::NONE;
        state.visible_access = vk::AccessFlags2::NONE;
    } else {
        state.read_stages |= usage.stages;
    }
    barrier
}

pub fn state_key(baked: &BakedGraph, resource: ResourceId) -> StateKey {
    match baked.resources[resource.index()].physical {
        Some(physical) => StateKey::Physical(physical),
        None => StateKey::External(resource),
    }
}

/// Barriers to record before baked pass `index`, advancing `store`. A
/// skipped (conditional) pass must not call this: it neither records nor
/// changes state, and the next access derives its own transition.
pub fn pass_barriers(baked: &BakedGraph, index: usize, store: &mut impl StateStore) -> Vec<BarrierDesc> {
    let pass = &baked.passes[index];
    let mut out = Vec::with_capacity(pass.accesses.len());
    for access in &pass.accesses {
        let key = state_key(baked, access.resource);
        let image = baked.resources[access.resource.index()].image;
        if let Some(mut barrier) = transition(store.state(key), access.usage, image, access.discard) {
            barrier.resource = access.resource;
            barrier.key = key;
            out.push(barrier);
        }
    }
    out
}

/// Barriers for one frame, per baked pass (empty for skipped passes).
pub fn plan_frame(
    baked: &BakedGraph,
    store: &mut impl StateStore,
    skipped: impl Fn(super::PassId) -> bool,
) -> Vec<Vec<BarrierDesc>> {
    (0..baked.passes.len())
        .map(|index| {
            if skipped(baked.passes[index].pass) {
                Vec::new()
            } else {
                pass_barriers(baked, index, store)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BakeOptions, FrameGraph, ImageDesc, LoadOp, PassKind, ResourceClass, Role, ShaderStage, bake};
    use std::collections::BTreeMap;

    type Store = BTreeMap<StateKey, AccessState>;

    fn usage(role: Role) -> Usage {
        role.usage()
    }

    #[test]
    fn read_after_read_emits_nothing() {
        let mut state = AccessState::default();
        let write = transition(
            &mut state,
            usage(Role::StorageImageWrite(ShaderStage::Compute)),
            true,
            true,
        );
        assert!(write.is_some(), "UNDEFINED -> GENERAL");
        let first = transition(
            &mut state,
            usage(Role::StorageImageRead(ShaderStage::Compute)),
            true,
            false,
        );
        let barrier = first.expect("write -> read needs visibility");
        assert_eq!(barrier.src_access, vk::AccessFlags2::SHADER_STORAGE_WRITE);
        assert_eq!(barrier.old_layout, barrier.new_layout, "no layout change");
        let second = transition(
            &mut state,
            usage(Role::StorageImageRead(ShaderStage::Compute)),
            true,
            false,
        );
        assert_eq!(second, None, "read after read");
    }

    #[test]
    fn layout_change_waits_for_prior_reads_and_write() {
        let mut state = AccessState::default();
        transition(&mut state, usage(Role::ColorAttachment(LoadOp::Clear)), true, true);
        transition(&mut state, usage(Role::Sampled(ShaderStage::Fragment)), true, false);
        let to_transfer = transition(&mut state, usage(Role::TransferSrc), true, false).unwrap();
        assert_eq!(to_transfer.old_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(to_transfer.new_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert!(
            to_transfer
                .src_stages
                .contains(vk::PipelineStageFlags2::FRAGMENT_SHADER)
        );
    }

    #[test]
    fn discard_uses_undefined_and_write_after_read_syncs() {
        let mut state = AccessState::default();
        transition(
            &mut state,
            usage(Role::StorageImageWrite(ShaderStage::Compute)),
            true,
            true,
        );
        transition(
            &mut state,
            usage(Role::StorageImageRead(ShaderStage::Fragment)),
            true,
            false,
        );
        // Same layout, a write after a read: WAR dependency, no transition.
        let war = transition(
            &mut state,
            usage(Role::StorageImageWrite(ShaderStage::Compute)),
            true,
            true,
        )
        .unwrap();
        assert!(war.src_stages.contains(vk::PipelineStageFlags2::FRAGMENT_SHADER));
        assert_eq!(war.new_layout, vk::ImageLayout::GENERAL);
        // A discarding change of layout starts from UNDEFINED.
        let clear = transition(&mut state, usage(Role::ColorAttachment(LoadOp::Clear)), true, true).unwrap();
        assert_eq!(clear.old_layout, vk::ImageLayout::UNDEFINED);
    }

    /// Write → read at one stage (with a layout change) → read at another
    /// stage: the second barrier must still carry the write's access.
    #[test]
    fn second_read_stage_after_a_transition_keeps_the_write() {
        let mut state = AccessState::default();
        transition(&mut state, usage(Role::ColorAttachment(LoadOp::Clear)), true, true);
        let first = transition(&mut state, usage(Role::Sampled(ShaderStage::Vertex)), true, false).unwrap();
        assert_eq!(first.src_access, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);
        assert_eq!(first.new_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let second = transition(&mut state, usage(Role::Sampled(ShaderStage::Fragment)), true, false)
            .expect("the fragment stage has not seen the write");
        assert_eq!(second.old_layout, second.new_layout, "no second transition");
        assert_eq!(second.src_access, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);
        assert!(
            second
                .src_stages
                .contains(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        );
        assert!(second.src_stages.contains(vk::PipelineStageFlags2::VERTEX_SHADER));
        assert_eq!(second.dst_stages, vk::PipelineStageFlags2::FRAGMENT_SHADER);
        let third = transition(&mut state, usage(Role::Sampled(ShaderStage::Fragment)), true, false);
        assert_eq!(third, None, "read after read");
        // A later write waits for both read stages and the transition.
        let war = transition(&mut state, usage(Role::ColorAttachment(LoadOp::Clear)), true, true).unwrap();
        assert!(
            war.src_stages
                .contains(vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER)
        );
    }

    #[test]
    fn buffers_never_change_layout_and_first_use_is_free() {
        let mut state = AccessState::default();
        assert_eq!(transition(&mut state, usage(Role::TransferDst), false, true), None);
        let read = transition(
            &mut state,
            usage(Role::StorageBufferRead(ShaderStage::Vertex)),
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            (read.old_layout, read.new_layout),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::UNDEFINED)
        );
        assert_eq!(read.src_access, vk::AccessFlags2::TRANSFER_WRITE);
    }

    fn demo_graph() -> FrameGraph {
        let mut graph = FrameGraph::new();
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
        let scene = graph.image(
            "scene",
            ImageDesc::color_2d(vk::Format::R8G8B8A8_UNORM, 16, 16, usage),
            ResourceClass::Persistent,
        );
        let back = graph.image(
            "backbuffer",
            ImageDesc::color_2d(
                vk::Format::B8G8R8A8_UNORM,
                16,
                16,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
            ResourceClass::Backbuffer,
        );
        graph
            .pass("scene", PassKind::Graphics, ())
            .color(scene, LoadOp::Clear)
            .conditional()
            .id();
        graph
            .pass("composite", PassKind::Graphics, ())
            .sampled(scene, ShaderStage::Fragment)
            .color(back, LoadOp::Clear)
            .id();
        graph.pass("present", PassKind::Present, ()).present(back).root().id();
        graph
    }

    /// Persistent layouts survive frames: frame 2 does not re-transition
    /// from UNDEFINED, and the present transition comes back each frame.
    #[test]
    fn persistent_state_survives_frames() {
        let graph = demo_graph();
        let baked = bake(&graph, BakeOptions::default()).unwrap();
        let mut store = Store::new();
        let frame1 = plan_frame(&baked, &mut store, |_| false);
        assert_eq!(frame1[0][0].old_layout, vk::ImageLayout::UNDEFINED, "scene first clear");
        let frame2 = plan_frame(&baked, &mut store, |_| false);
        assert_eq!(
            frame2[0][0].old_layout,
            vk::ImageLayout::UNDEFINED,
            "a clear discards, but syncs with last frame's sampling"
        );
        assert!(
            frame2[0][0]
                .src_stages
                .contains(vk::PipelineStageFlags2::FRAGMENT_SHADER)
        );
        let present = frame2[2][0];
        assert_eq!(present.new_layout, vk::ImageLayout::PRESENT_SRC_KHR);
    }

    /// Skipping a conditional pass changes only the concrete barriers of
    /// later accesses (derived from the state machine), never the plan.
    #[test]
    fn skipped_pass_records_no_barriers_and_keeps_state() {
        let graph = demo_graph();
        let baked = bake(&graph, BakeOptions::default()).unwrap();
        let mut store = Store::new();
        plan_frame(&baked, &mut store, |_| false);
        let scene_pass = baked.passes[0].pass;
        let skipped = plan_frame(&baked, &mut store, |pass| pass == scene_pass);
        assert!(skipped[0].is_empty());
        // The composite still samples `scene` in its existing layout: no
        // transition, and the write from frame 1 is already visible.
        assert!(
            skipped[1]
                .iter()
                .all(|barrier| barrier.resource != graph.passes[0].intents[0].resource)
        );
    }
}
