//! Rust SDK for render modules (built for `wasm32-unknown-unknown`).
//!
//! A render module implements [`RenderModule`] and names its type in
//! [`render_module!`], which writes the exports the host calls (see
//! [`ae_render_abi::module`]). `declare` describes the frame graph and the
//! program each pass runs; `frame` runs every frame with the host's
//! [`FrameInput`] and may disable conditional passes and fill fullscreen
//! programs' uniform blocks.
//!
//! ```ignore
//! #[derive(Default)]
//! struct Scene;
//! impl ae_render_sdk::RenderModule for Scene {
//!     fn declare(&mut self, env: &DeclareEnv) -> Option<Declaration> { ... }
//! }
//! ae_render_sdk::render_module!(Scene);
//! ```
//!
//! # The display rule
//!
//! The output's [`OutputEncoding`] decides the tail of the graph, and
//! [`DeclareEnv::encoding`] carries it (with the levels, the working space
//! and the output gamut):
//!
//! * **HDR** ([`OutputEncoding::Hdr10Pq`] or [`OutputEncoding::ScRgb`]):
//!   declare the display intermediate ([`display_spec`], [`DISPLAY_NAME`]:
//!   `RGBA16F`, linear light in units of paper white, in the working space),
//!   have `post.tonemap` write it, run every colour-changing pass (a grade,
//!   a vignette) on it, and end with a `post.encode` pass that writes the
//!   target. Only `post.encode` maps the display to the output's encoding
//!   and gamut. Values may exceed 1 (paper white): that is the point.
//! * **SDR** ([`OutputEncoding::Sdr`]): no display image and no encode
//!   step. `post.tonemap` writes the target itself.
//!
//! The host checks the rule on the declared graph: `post.encode` must be
//! present exactly when the output is HDR, and the error names the missing
//! or extra pass and the encoding.
//!
//! Antialiasing and effects that need range should work in the display's
//! units and leave the encode to the end; a pass must not clamp at paper
//! white, or the values the curve put above it are lost.
//!
//! # Fullscreen programs
//!
//! [`Program::Fullscreen`] is one fragment shader drawn as a fullscreen
//! triangle into its pass's colour attachments. It has two forms, and a
//! shader uses exactly one (the host reflects the SPIR-V and refuses a mix):
//!
//! * **Per-pass** (contract version 2, accepted by 3 for one version): set
//!   0, binding 0 is the `uniform` block (`uniform_bytes` long, std140),
//!   bindings 1.. are one combined image sampler per `Sampled` intent of the
//!   pass, in intent order.
//! * **Bindless** (contract version 3): only the engine's global set.
//!   Binding 0 is `texture2D images[]`, binding 1 `sampler samplers[]`,
//!   binding 3 the program's `uint handles[]` table and binding 4 its
//!   `std140` block (`uniform` blocks are uniform buffers: they cost less
//!   per access than storage buffers). The push block is at most 16 bytes:
//!   `uniforms` (the block's index), `samplerIndex`, `handles` (the table's
//!   index) and a pad word.
//!
//! A bindless shader includes the engine's interface headers
//! (`crates/ae_render/shaders/bindless.glsl` and `program.glsl`; add that
//! directory to `glslc`'s include path) with `AE_DECLARE_PUSH 0`, declares
//! its block as `ae_uniforms[]`, then uses `AE_UNIFORMS` and
//! `AE_PROGRAM_IMAGE(k)` (its k-th sampled image, in the pass's order):
//!
//! ```glsl
//! #define AE_DECLARE_PUSH 0
//! #include "bindless.glsl"
//! #include "program.glsl"
//! layout(std140, set = AE_BINDLESS_SET, binding = AE_UNIFORM_BLOCKS) uniform Grade {
//!     vec4 tint;
//! } ae_uniforms[];
//! #define g AE_UNIFORMS
//! #define scene AE_PROGRAM_IMAGE(0)
//! ```
//!
//! The host writes the program's table with the pass's sampled-image
//! indices before the pass records; the indices are the same in every frame
//! slot. `FrameOutput::uniforms` still carries each program's block in
//! program order, `uniform_bytes` long.
//!
//! `RenderModule::VERSION` pins the contract version a module is built for
//! (the SDK's current one by default); `render/legacy` is a contract-2
//! fixture.

pub use ae_render_abi as abi;
pub use ae_render_abi::module::{
    BLOOM_FORMAT, BLOOM_NAME, BLOOM_SCRATCH_NAME, DISPLAY_FORMAT, DISPLAY_NAME, DeclareInput, EXPOSURE_FORMAT,
    EXPOSURE_HISTOGRAM_NAME, EXPOSURE_LOG2_MAX, EXPOSURE_LOG2_MIN, EXPOSURE_NAME, FrameInput, GLOW_FORMAT, GLOW_NAME,
    Gamut, OutputEncoding, PARTICLE_SHADOW_NAME, Program, ProgramTable, SCENE_FORMAT, SCENE_NAME, SHADOW_MAP_NAME,
    WorkingSpace, bloom_size, bloom_spec, display_spec, exposure_spec, glow_spec, particle_shadow_spec, scene_spec,
    shadow_map_spec,
};
pub use ae_render_abi::{Class, GraphBuilder, ImageSpec, LoadOp, PassKind, PassWriter, ResourceRef, Role, ShaderStage};

/// Raw Vulkan values a module needs to describe images.
pub mod vk {
    pub const FORMAT_R8G8B8A8_UNORM: u32 = 37;
    pub const FORMAT_R8G8B8A8_SRGB: u32 = 43;
    pub const FORMAT_B8G8R8A8_SRGB: u32 = 50;
    pub const FORMAT_R16G16B16A16_SFLOAT: u32 = 97;
    pub const FORMAT_A2B10G10R10_UNORM_PACK32: u32 = 48;
    pub const FORMAT_D32_SFLOAT: u32 = 126;
    pub const USAGE_TRANSFER_SRC: u32 = 0x1;
    pub const USAGE_TRANSFER_DST: u32 = 0x2;
    pub const USAGE_SAMPLED: u32 = 0x4;
    pub const USAGE_STORAGE: u32 = 0x8;
    pub const USAGE_COLOR_ATTACHMENT: u32 = 0x10;
    pub const USAGE_DEPTH_STENCIL_ATTACHMENT: u32 = 0x20;
}

/// The target and the output's display spec a declaration is for: what the
/// host wrote into [`DeclareInput`] before calling `declare`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeclareEnv {
    pub width: u32,
    pub height: u32,
    /// `VkFormat` of the target (the swapchain's, or the readback image's).
    pub color_format: u32,
    /// The output's encoding: HDR keeps the display intermediate and ends
    /// in `post.encode`, SDR has neither (see the crate docs).
    pub encoding: OutputEncoding,
    /// Paper white in signal nits: the display intermediate's 1.0.
    pub paper_white: f32,
    /// The display's peak in the same signal nits (`peak / paper_white` is
    /// the headroom the tone curve may use).
    pub peak: f32,
    /// The space the scene is lit in and the display intermediate stores.
    pub working_space: WorkingSpace,
    /// The gamut `post.encode` maps the display to (unused in SDR).
    pub output_gamut: Gamut,
    /// The scene has units: `units.*` library passes and resources exist.
    pub units: bool,
    /// Render to the library's `readback.target` and copy it to
    /// `readback.buffer` instead of presenting a `Backbuffer`.
    pub readback: bool,
    /// The scene draws water: the `water.draw` library pass exists.
    pub water: bool,
    /// The units cast shadows (only with `units`): the `shadows.draw`
    /// library pass and the `shadows.map` library image
    /// ([`shadow_map_spec`], `External`) exist. Declaring the map, the
    /// graph has `shadows.draw` write it (`depth:clear`, reading
    /// `units.instances` in the vertex stage) and every terrain, unit and
    /// water draw sample it (`sampled:fragment`).
    pub shadows: bool,
    /// The scene has particles: the `particles.sim` and `particles.draw`
    /// library passes and the `particles.state`, `particles.keys` and
    /// `particles.indirect` library buffers exist. `particles.draw` samples
    /// the depth buffer (`sampled:fragment`, so declare it with sampled
    /// usage) and blends into one colour attachment, after everything
    /// opaque and the water. With `shadows` too, the particles' coloured
    /// shadows can be declared: the `shadows.color` library image
    /// ([`particle_shadow_spec`], `External`), written by the
    /// `shadows.particles` library pass (`color:clear`; `particles.state`
    /// and `particles.keys` read in the vertex stage, `particles.indirect`
    /// as an indirect buffer) and sampled
    /// (`sampled:fragment`) by every pass that samples `shadows.map`.
    pub particles: bool,
    // The glow and bloom are not flags: a graph declares the `glow` image
    // (`glow_spec`) and the three `post.bloom` passes itself, or neither.
}

impl DeclareEnv {
    /// Whether the graph keeps a display intermediate and ends in
    /// `post.encode`.
    pub fn uses_display(&self) -> bool {
        self.encoding.uses_display()
    }

    /// How far the peak is above paper white (the tone curve's headroom).
    pub fn headroom(&self) -> f32 {
        if self.paper_white > 0.0 {
            self.peak / self.paper_white
        } else {
            1.0
        }
    }
}

pub struct Declaration {
    pub graph: GraphBuilder,
    pub programs: ProgramTable,
}

/// What `frame` hands back.
#[derive(Default)]
pub struct FrameOutput {
    disabled: Vec<u32>,
    /// Every fullscreen program's uniform block, in program order.
    pub uniforms: Vec<u8>,
}

impl FrameOutput {
    /// Skip the conditional passes whose payload is `program` this frame.
    pub fn disable(&mut self, program: u32) {
        let word = program as usize / 32;
        if self.disabled.len() <= word {
            self.disabled.resize(word + 1, 0);
        }
        self.disabled[word] |= 1 << (program % 32);
    }
}

pub trait RenderModule: Default {
    /// The contract version this module is built for: the SDK's current one
    /// ([`abi::module::MODULE_VERSION`]), unless a module pins an older one
    /// it still wants to run under (a contract-2 module keeps the per-pass
    /// fullscreen form).
    const VERSION: u32 = abi::module::MODULE_VERSION;

    /// The graph for this target, or `None` if the module cannot render it.
    fn declare(&mut self, env: &DeclareEnv) -> Option<Declaration>;

    /// Per-frame: called after the host wrote `input`. `out` starts empty
    /// (nothing disabled, no uniforms) every frame.
    fn frame(&mut self, _input: &FrameInput, _out: &mut FrameOutput) {}
}

/// Export glue; used by [`render_module!`].
#[doc(hidden)]
pub mod rt {
    use super::*;

    pub struct Exports<M> {
        pub module: M,
        declaration: (Vec<u8>, Vec<u8>),
        declare_input: [u8; abi::module::DECLARE_INPUT_BYTES],
        declared: [u32; 4],
        input: [u8; abi::module::FRAME_INPUT_BYTES],
        output: FrameOutput,
        returned: [u32; 4],
    }

    impl<M: RenderModule> Default for Exports<M> {
        fn default() -> Self {
            Self {
                module: M::default(),
                declaration: (Vec::new(), Vec::new()),
                declare_input: [0; abi::module::DECLARE_INPUT_BYTES],
                declared: [0; 4],
                input: [0; abi::module::FRAME_INPUT_BYTES],
                output: FrameOutput::default(),
                returned: [0; 4],
            }
        }
    }

    impl<M: RenderModule> Exports<M> {
        pub fn declare_input(&mut self) -> u32 {
            self.declare_input.as_mut_ptr() as u32
        }

        pub fn declare(&mut self) -> u32 {
            let input = DeclareInput::from_bytes(&self.declare_input);
            let env = DeclareEnv {
                width: input.width,
                height: input.height,
                color_format: input.color_format,
                // Unknown wire values mean an ABI this module does not know;
                // the host only writes the ones it defines.
                encoding: input.output_encoding().unwrap_or(OutputEncoding::Sdr),
                paper_white: input.paper_white,
                peak: input.peak,
                working_space: input.working().unwrap_or(WorkingSpace::Bt709),
                output_gamut: input.gamut().unwrap_or(Gamut::Bt709),
                units: input.flags & abi::module::DECLARE_UNITS != 0,
                readback: input.flags & abi::module::DECLARE_READBACK != 0,
                water: input.flags & abi::module::DECLARE_WATER != 0,
                shadows: input.flags & abi::module::DECLARE_SHADOWS != 0,
                particles: input.flags & abi::module::DECLARE_PARTICLES != 0,
            };
            let Some(d) = self.module.declare(&env) else { return 0 };
            self.declaration = (d.graph.finish(), d.programs.finish());
            let (g, p) = &self.declaration;
            self.declared = [g.as_ptr() as u32, g.len() as u32, p.as_ptr() as u32, p.len() as u32];
            self.declared.as_ptr() as u32
        }

        pub fn input(&mut self) -> u32 {
            self.input.as_mut_ptr() as u32
        }

        pub fn frame(&mut self) -> u32 {
            let input = FrameInput::from_bytes(&self.input);
            self.output.disabled.clear();
            self.output.uniforms.clear();
            self.module.frame(&input, &mut self.output);
            let o = &self.output;
            self.returned = [
                o.disabled.as_ptr() as u32,
                o.disabled.len() as u32,
                o.uniforms.as_ptr() as u32,
                o.uniforms.len() as u32,
            ];
            self.returned.as_ptr() as u32
        }
    }
}

/// Write the render module exports for `$ty` (a [`RenderModule`]).
#[macro_export]
macro_rules! render_module {
    ($ty:ty) => {
        thread_local! {
            static AE_RENDER_MODULE: ::std::cell::RefCell<$crate::rt::Exports<$ty>> = ::std::cell::RefCell::new(::std::default::Default::default());
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn ae_render_version() -> u32 {
            <$ty as $crate::RenderModule>::VERSION
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn ae_render_declare_input() -> u32 {
            AE_RENDER_MODULE.with(|m| m.borrow_mut().declare_input())
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn ae_render_declare() -> u32 {
            AE_RENDER_MODULE.with(|m| m.borrow_mut().declare())
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn ae_render_input() -> u32 {
            AE_RENDER_MODULE.with(|m| m.borrow_mut().input())
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn ae_render_frame() -> u32 {
            AE_RENDER_MODULE.with(|m| m.borrow_mut().frame())
        }
    };
}
