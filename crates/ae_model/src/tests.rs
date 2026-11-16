use super::*;
use crate::pose::{Pose, globals, palette, rest_palette};

fn bone(name: &str, parent: Option<u32>, t: [f32; 3]) -> Bone {
    Bone { name: name.into(), parent, rest: trs(t, [0.0, 0.0, 0.0, 1.0], [1.0; 3]).to_cols_array(), translation: t, rotation: [0.0, 0.0, 0.0, 1.0], scale: [1.0; 3] }
}

fn sample_model() -> Model {
    let tex = ktx::write(&ktx::Image { format: ktx::Format::RGBA8_SRGB, width: 2, height: 1, levels: vec![vec![255, 0, 0, 255, 0, 255, 0, 128], vec![128, 128, 0, 192]] }, true).unwrap();
    let v = |x: f32, joint: u16| Vertex { position: [x, 0.0, 0.0], normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0], color: [65535; 4], joints: [joint, 0, 0, 0], weights: [65535, 0, 0, 0], ..Default::default() };
    Model {
        bones: vec![bone("root", None, [0.0; 3]), bone("arm", Some(0), [1.0, 0.0, 0.0])],
        joints: vec![Joint { bone: 0, inverse_bind: Mat4::IDENTITY.to_cols_array() }, Joint { bone: 1, inverse_bind: Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0)).to_cols_array() }],
        primitives: vec![Primitive { material: 0, first_vertex: 0, vertex_count: 3, first_index: 0, index_count: 3 }],
        materials: vec![Material {
            name: "m".into(),
            base_color_texture: Some(TextureRef { texture: 0, uv: 0, sampler: Sampler::default(), transform: Some(UvTransform { offset: [0.5, 0.0], rotation: 0.3, scale: [2.0, 2.0] }) }),
            alpha: AlphaMode::Mask { cutoff: 0.5 },
            ..Default::default()
        }],
        vertices: vec![v(0.0, 0), v(1.0, 1), v(2.0, 1)],
        indices: vec![0, 1, 2],
        texture_names: vec!["t".into()],
        textures: vec![tex],
        clips: vec![Clip {
            name: "swing".into(),
            duration: 2.0,
            channels: vec![Channel { bone: 1, path: Path::Translation, interpolation: Interpolation::Linear, times: Range { offset: 0, len: 2 }, values: Range { offset: 2, len: 6 } }],
        }],
        clip_data: vec![0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 4.0, 0.0],
        bounds: Bounds { min: [0.0; 3], max: [2.0, 0.0, 0.0] },
    }
}

#[test]
fn round_trip() {
    let m = sample_model();
    m.validate().unwrap();
    let bytes = m.to_bytes();
    assert_eq!(&bytes[..8], b"AEMODEL\0");
    let back = Model::from_bytes(&bytes).unwrap();
    assert_eq!(back, m);
    // Every blob range 16-aligned.
    let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    assert_eq!(header_len % 16, 0);
}

#[test]
fn refuses_other_versions_and_bad_references() {
    let m = sample_model();
    let mut bytes = m.to_bytes();
    bytes[8] = 99;
    assert!(Model::from_bytes(&bytes).unwrap_err().to_string().contains("version 99"));
    let mut bad = sample_model();
    bad.vertices[1].joints[0] = 7;
    assert!(bad.validate().is_err());
    let mut bad = sample_model();
    bad.bones[1].parent = Some(1);
    assert!(bad.validate().is_err());
}

#[test]
fn ktx_round_trip_and_the_reference_reader() {
    for format in [ktx::Format::RGBA8_UNORM, ktx::Format::RGBA8_SRGB, ktx::Format::BC7_SRGB] {
        for zstd in [false, true] {
            let (w, h) = (5u32, 3u32);
            let levels: Vec<Vec<u8>> =
                (0..3).map(|l| (0..format.level_bytes((w >> l).max(1), (h >> l).max(1)).unwrap()).map(|i| (i * 7 + l * 13) as u8).collect()).collect();
            let image = ktx::Image { format, width: w, height: h, levels };
            let bytes = ktx::write(&image, zstd).unwrap();
            let back = ktx::read(&bytes).unwrap();
            assert_eq!(back, image, "{format:?} zstd {zstd}");
            let r = ktx2::Reader::new(&bytes[..]).unwrap();
            let dfd = r.basic_dfd().expect("a basic DFD");
            assert_eq!(dfd.transfer_function == Some(ktx2::TransferFunction::SRGB), format.is_srgb());
            assert_eq!(r.header().format.map(|f| f.value()), Some(format.0));
        }
    }
    // A level of the wrong size is refused.
    let bad = ktx::Image { format: ktx::Format::RGBA8_UNORM, width: 2, height: 2, levels: vec![vec![0; 15]] };
    assert!(ktx::write(&bad, false).is_err());
}

#[test]
fn pose_and_clip() {
    let m = sample_model();
    // Rest: every joint's palette matrix is the identity (bind = rest).
    for p in rest_palette(&m) {
        assert!(p.abs_diff_eq(Mat4::IDENTITY, 1e-6));
    }
    let mut pose = Pose::rest(&m);
    pose.sample(&m, 0, 1.0);
    // Half way from (1, 0, 0) to (1, 4, 0).
    assert!(pose.translation[1].abs_diff_eq(Vec3::new(1.0, 2.0, 0.0), 1e-6));
    let pal = palette(&m, &globals(&m, &pose.locals(&m)));
    assert!(pal[1].transform_point3(Vec3::new(1.0, 0.0, 0.0)).abs_diff_eq(Vec3::new(1.0, 2.0, 0.0), 1e-6));
    // Held outside the keys.
    pose.sample(&m, 0, 9.0);
    assert!(pose.translation[1].abs_diff_eq(Vec3::new(1.0, 4.0, 0.0), 1e-6));
}

#[test]
fn step_slerp_and_cubic() {
    let mut m = sample_model();
    // Rotation, linear: a quarter turn about +Y over 1 s; half way is an eighth.
    let q = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    m.clip_data = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, q.x, q.y, q.z, q.w];
    m.clips[0].channels[0] = Channel { bone: 1, path: Path::Rotation, interpolation: Interpolation::Linear, times: Range { offset: 0, len: 2 }, values: Range { offset: 2, len: 8 } };
    m.validate().unwrap();
    let mut pose = Pose::rest(&m);
    pose.sample(&m, 0, 0.5);
    assert!(pose.rotation[1].abs_diff_eq(Quat::from_rotation_y(std::f32::consts::FRAC_PI_4), 1e-5));
    m.clips[0].channels[0].interpolation = Interpolation::Step;
    pose.sample(&m, 0, 0.99);
    assert!(pose.rotation[1].abs_diff_eq(Quat::IDENTITY, 1e-6));
    // Cubic spline, translation, zero tangents: smoothstep between the keys.
    m.clip_data = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    m.clips[0].channels[0] = Channel { bone: 1, path: Path::Translation, interpolation: Interpolation::CubicSpline, times: Range { offset: 0, len: 2 }, values: Range { offset: 2, len: 18 } };
    m.validate().unwrap();
    pose.sample(&m, 0, 0.25);
    let s: f32 = 0.25;
    assert!((pose.translation[1].x - 10.0 * (3.0 * s * s - 2.0 * s * s * s)).abs() < 1e-5);
}

#[test]
fn gltf_axes() {
    // glTF's front (+Z) is the engine's forward (+X); glTF's +X (the
    // model's left) the engine's -Z (left, +Z being right).
    assert!((GLTF_TO_ENGINE * Vec3::Z).abs_diff_eq(Vec3::X, 1e-6));
    assert!((GLTF_TO_ENGINE * Vec3::X).abs_diff_eq(-Vec3::Z, 1e-6));
    assert!((GLTF_TO_ENGINE * Vec3::Y).abs_diff_eq(Vec3::Y, 1e-6));
}

#[test]
fn uv_transform_matrix() {
    // KHR_texture_transform's own example order: T * R * S.
    let t = UvTransform { offset: [0.5, 0.25], rotation: 0.4, scale: [2.0, 3.0] };
    let m = t.matrix();
    let uv = [0.3f32, 0.7];
    let got = [m[0][0] * uv[0] + m[1][0] * uv[1] + m[2][0], m[0][1] * uv[0] + m[1][1] * uv[1] + m[2][1]];
    let (s, c) = 0.4f32.sin_cos();
    let (x, y) = (uv[0] * 2.0, uv[1] * 3.0);
    // The extension's rotation matrix: [c s; -s c].
    let want = [c * x + s * y + 0.5, -s * x + c * y + 0.25];
    assert!((got[0] - want[0]).abs() < 1e-6 && (got[1] - want[1]).abs() < 1e-6, "{got:?} {want:?}");
}
