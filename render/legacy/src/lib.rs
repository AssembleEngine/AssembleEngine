//! A contract-2 render module: a fullscreen program fills a transient image
//! with a tint, a second one (sampling it through a combined image sampler)
//! scales it into the target, then present or readback. Both programs use
//! the per-pass form (set 0: the uniform block, then the sampled images),
//! which contract version 3 still accepts. The host's tests use it to prove
//! that form still loads and renders.

use ae_render_sdk::{
    Class, Declaration, DeclareEnv, FrameInput, FrameOutput, GraphBuilder, ImageSpec, LoadOp, PassKind, Program,
    ProgramTable, RenderModule, Role, ShaderStage, render_module, vk,
};

const FILL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fill.frag.spv"));
const COPY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy.frag.spv"));

/// The fill's colour, and the copy's gain.
pub const TINT: [f32; 3] = [0.5, 0.25, 0.75];
pub const GAIN: [f32; 3] = [1.0, 0.5, 1.0];

#[derive(Default)]
struct Legacy {
    copy: u32,
}

fn library(graph: &mut GraphBuilder, name: &str) -> ae_render_sdk::ResourceRef {
    graph.buffer(name, 0, 0, Class::External)
}

impl RenderModule for Legacy {
    /// The per-pass fullscreen form is contract 2's.
    const VERSION: u32 = 2;

    fn declare(&mut self, env: &DeclareEnv) -> Option<Declaration> {
        let (w, h) = (env.width, env.height);
        let mut g = GraphBuilder::new();
        let mut p = ProgramTable::new();
        let source = g.image(
            "legacy.source",
            ImageSpec::color_2d(env.color_format, w, h, vk::USAGE_COLOR_ATTACHMENT | vk::USAGE_SAMPLED),
            Class::Transient,
        );
        let target = if env.readback {
            g.image(
                "readback.target",
                ImageSpec::color_2d(
                    env.color_format,
                    w,
                    h,
                    vk::USAGE_COLOR_ATTACHMENT | vk::USAGE_TRANSFER_SRC,
                ),
                Class::External,
            )
        } else {
            g.image(
                "target",
                ImageSpec::color_2d(env.color_format, w, h, vk::USAGE_COLOR_ATTACHMENT),
                Class::Backbuffer,
            )
        };
        let fill = p.push(Program::Fullscreen {
            label: "legacy.fill",
            spirv: FILL,
            uniform_bytes: 16,
        });
        g.pass("fill", PassKind::Graphics, fill)
            .color(source, LoadOp::Clear)
            .id();
        self.copy = p.push(Program::Fullscreen {
            label: "legacy.copy",
            spirv: COPY,
            uniform_bytes: 16,
        });
        g.pass("copy", PassKind::Graphics, self.copy)
            .sampled(source, ShaderStage::Fragment)
            .color(target, LoadOp::DontCare)
            .id();
        if env.readback {
            let buffer = library(&mut g, "readback.buffer");
            let copy = p.library("readback.copy");
            g.pass("copy out", PassKind::Transfer, copy)
                .access(target, Role::TransferSrc)
                .discard(buffer, Role::TransferDst)
                .id();
            let read = p.library("host.read");
            g.pass("host read", PassKind::Transfer, read)
                .access(buffer, Role::HostRead)
                .root()
                .id();
        } else {
            let present = p.library("present");
            g.pass("present", PassKind::Present, present)
                .present(target)
                .root()
                .id();
        }
        Some(Declaration { graph: g, programs: p })
    }

    fn frame(&mut self, _input: &FrameInput, out: &mut FrameOutput) {
        // Program order: the fill's block, then the copy's.
        for value in [TINT[0], TINT[1], TINT[2], 1.0, GAIN[0], GAIN[1], GAIN[2], 1.0] {
            out.uniforms.extend_from_slice(&value.to_le_bytes());
        }
    }
}

render_module!(Legacy);
