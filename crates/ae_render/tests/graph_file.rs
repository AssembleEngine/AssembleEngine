//! Graph files declare what the code modules declare: `render/default.graph.json`
//! and `render/post.graph.json` against the in-repo WASM modules
//! `render/default` and `render/post` (built on first use): the same graph
//! descriptor (decoded and encoded again: name order in the blob follows
//! call order) and the same program table, byte for byte with the
//! SPIR-V, for every scene
//! kind and a few targets, and the same per-frame output (uniforms, which
//! conditional passes run) for each flag value. No GPU needed.

use ae_render::encode::{DisplayLevels, OutputEncoding};
use ae_render::module::{DeclareTarget, RenderModule, build, engine};
use ae_render_abi::module::FrameInput;
use ash::vk;
use std::path::Path;

fn repo(path: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
        .to_string_lossy()
        .into_owned()
}

/// A descriptor decoded and encoded again: the builder writes names into
/// its blob in call order, so two declarations of the same graph can
/// differ in bytes when resources and passes are declared interleaved.
fn canonical(graph: &[u8]) -> Vec<u8> {
    let decoded = ae_framegraph::descriptor::decode(graph).expect("the descriptor decodes");
    ae_framegraph::descriptor::encode(&decoded, |payload| *payload)
}

#[test]
fn graph_files_declare_what_the_modules_declare() {
    let engine = engine();
    for name in ["default", "post"] {
        let mut wasm = RenderModule::load(&engine, &build(name).unwrap()).unwrap();
        let file_bytes = build(&repo(&format!("render/{name}.graph.json"))).unwrap();
        let mut file = RenderModule::load(&engine, &file_bytes).unwrap();
        assert!(matches!(file, RenderModule::File(_)) && matches!(wasm, RenderModule::Wasm(_)));
        let levels = DisplayLevels {
            paper_white: 203.0,
            peak: 361.0,
        };
        for encoding in [OutputEncoding::Sdr, OutputEncoding::Hdr10Pq, OutputEncoding::ScRgb] {
            for (width, height, format) in [(1280, 720, 50), (64, 32, 43), (1, 1, 37)] {
                for flags in ae_render::graph_file::SCENE_FLAGS {
                    let target = DeclareTarget {
                        width,
                        height,
                        color_format: vk::Format::from_raw(format as i32),
                        flags,
                        encoding,
                        levels,
                        working: Default::default(),
                        gamut: encoding.default_gamut(),
                    };
                    let session = || DeclareTarget { ..target };
                    let want = wasm.declare(&session()).unwrap().expect("the module declares");
                    let got = file.declare(&session()).unwrap().expect("the file declares");
                    assert!(
                        canonical(&got.graph) == canonical(&want.graph),
                        "{name} {encoding:?} {width}x{height} flags {flags}: graph descriptors differ"
                    );
                    assert!(
                        got.programs == want.programs,
                        "{name} {encoding:?} {width}x{height} flags {flags}: program tables differ"
                    );
                    let programs = ae_render_abi::module::ProgramTable::decode(&want.programs)
                        .unwrap()
                        .len() as u32;
                    for app_flags in [0, 1, 2, 3] {
                        let input = FrameInput {
                            width,
                            height,
                            flags: app_flags,
                            ..FrameInput::default()
                        };
                        let (w, f) = (wasm.frame(&input).unwrap(), file.frame(&input).unwrap());
                        assert_eq!(
                            f.uniforms, w.uniforms,
                            "{name} {encoding:?}: uniforms with flags {app_flags}"
                        );
                        for p in 0..programs {
                            assert_eq!(
                                f.is_disabled(p),
                                w.is_disabled(p),
                                "{name} {encoding:?}: program {p} with flags {app_flags}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// A graph file that names a shader keeps working only through `build`
/// (which inlines it); loading the raw file says so.
#[test]
fn raw_graph_files_with_shader_paths_are_refused() {
    let raw = std::fs::read(repo("render/post.graph.json")).unwrap();
    let err = RenderModule::load(&engine(), &raw).err().expect("refused");
    assert!(format!("{err:#}").contains("is not inlined"), "{err:#}");
    assert!(RenderModule::load(&engine(), b"not a module").is_err());
    assert!(
        format!(
            "{:#}",
            RenderModule::load(&engine(), b"{\"version\": 1}").err().unwrap()
        )
        .contains("parsing the graph file")
    );
}
