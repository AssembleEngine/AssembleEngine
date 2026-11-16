use super::*;
use crate::pose::{PlayMode, Pose, animated_radius, bake, globals, palette, rest_palette};

fn bone(name: &str, parent: Option<u32>, t: [f32; 3]) -> Bone {
    Bone { name: name.into(), parent, rest: trs(t, [0.0, 0.0, 0.0, 1.0], [1.0; 3]).to_cols_array(), translation: t, rotation: [0.0, 0.0, 0.0, 1.0], scale: [1.0; 3] }
}

fn sample_model() -> Model {
    let tex = ktx::write(&ktx::Image { format: ktx::Format::RGBA8_SRGB, width: 2, height: 1, levels: vec![vec![255, 0, 0, 255, 0, 255, 0, 128], vec![128, 128, 0, 192]] }, true).unwrap();
    let v = |x: f32, joint: u16| Vertex { position: [x, 0.0, 0.0], normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0], color: [65535; 4], joints: [joint, 0, 0, 0], weights: [65535, 0, 0, 0], ..Default::default() };
    Model {
        bones: vec![bone("root", None, [0.0; 3]), bone("arm", Some(0), [1.0, 0.0, 0.0])],
        joints: vec![Joint { bone: 0, inverse_bind: Mat4::IDENTITY.to_cols_array() }, Joint { bone: 1, inverse_bind: Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0)).to_cols_array() }],
        primitives: vec![Primitive { material: 0, layout: Layout::Rigid, first_vertex: 0, vertex_count: 3, first_index: 0, index_count: 3 }],
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
fn baked_clips_match_the_sampler_and_loop() {
    // Translation, linear: exact at every time (a lerp of lerps).
    let m = sample_model();
    let baked = bake(&m, 0);
    assert_eq!(baked.tracks, vec![1]);
    assert_eq!(baked.frames, 121);
    assert!((baked.rate - 60.0).abs() < 1e-4);
    for t in [0.0, 0.01, 0.5, 1.0, 1.337, 1.99] {
        let mut pose = Pose::rest(&m);
        pose.sample(&m, 0, t);
        let want = pose.locals(&m);
        let got = baked.locals(&m, t, PlayMode::Loop);
        for (w, g) in want.iter().zip(&got) {
            assert!(w.abs_diff_eq(*g, 1e-4), "t {t}: {w} vs {g}");
        }
        // Looping, ping-pong, once.
        let again = baked.locals(&m, t + 2.0 * baked.duration, PlayMode::Loop);
        assert!(again[1].abs_diff_eq(got[1], 1e-4), "t {t} two loops later");
        let back = baked.locals(&m, 2.0 * baked.duration - t, PlayMode::PingPong);
        assert!(back[1].abs_diff_eq(got[1], 1e-4), "t {t} on the way back");
        let forth = baked.locals(&m, 4.0 * baked.duration + t, PlayMode::PingPong);
        assert!(forth[1].abs_diff_eq(got[1], 1e-4), "t {t} forwards again");
        assert!(baked.locals(&m, t, PlayMode::Once)[1].abs_diff_eq(got[1], 1e-4), "t {t} once");
        let held = baked.locals(&m, baked.duration + 5.0 + t, PlayMode::Once);
        assert!(held[1].abs_diff_eq(baked.locals(&m, baked.duration, PlayMode::Once)[1], 1e-6), "t {t}: once holds the last frame");
    }
    // A duration that is not a whole number of frames: the rate rounds up
    // so the last frame falls on it.
    let mut m = sample_model();
    m.clip_data[1] = 0.3337;
    m.clips[0].duration = 0.3337;
    let baked = bake(&m, 0);
    assert_eq!(baked.frames, 22);
    assert!(((baked.frames - 1) as f32 / baked.rate - 0.3337).abs() < 1e-6);

    // Rotation: a quarter turn about +Y in two keys, the second stored
    // negated (the same rotation): the baked frames stay on one hemisphere
    // and the nlerp follows the slerp closely.
    let mut m = sample_model();
    let q = -Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    m.clip_data = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, q.x, q.y, q.z, q.w];
    m.clips[0].duration = 1.0;
    m.clips[0].channels[0] = Channel { bone: 1, path: Path::Rotation, interpolation: Interpolation::Linear, times: Range { offset: 0, len: 2 }, values: Range { offset: 2, len: 8 } };
    m.validate().unwrap();
    let baked = bake(&m, 0);
    assert!(baked.keys.windows(2).all(|k| k[0].r.dot(k[1].r) >= 0.0));
    for t in [0.1, 0.25, 0.5, 0.77] {
        let mut pose = Pose::rest(&m);
        pose.sample(&m, 0, t);
        let got = baked.locals(&m, t, PlayMode::Loop);
        assert!(pose.locals(&m)[1].abs_diff_eq(got[1], 1e-4), "t {t}");
    }
}

#[test]
fn animation_reach() {
    // The arm's tip (2, 0, 0) is carried up to (2, 4, 0) at the clip's end.
    let m = sample_model();
    let r = animated_radius(&m, Vec3::ZERO);
    assert!((r - 20f32.sqrt()).abs() < 0.01, "{r}");
    let mut still = sample_model();
    still.clips.clear();
    assert_eq!(animated_radius(&still, Vec3::ZERO), 0.0);
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

#[test]
fn merge_by_material_and_layout() {
    let mut m = sample_model();
    let mut skinned = m.vertices[0];
    skinned.joints = [0, 1, 0, 0];
    skinned.weights = [30000, 35535, 0, 0];
    let piece = |x: f32, joint: u16| Vertex { position: [x, 1.0, 0.0], joints: [joint, 0, 0, 0], weights: [65535, 0, 0, 0], ..m.vertices[0] };
    // Primitive 0 (rigid, joint 0..1), a second rigid piece on joint 1 with
    // the same material, and a skinned triangle with it too.
    m.vertices.extend([piece(0.0, 1), piece(1.0, 1), piece(2.0, 1), skinned, skinned, skinned]);
    m.indices.extend([2, 1, 0, 0, 1, 2]);
    m.primitives.push(Primitive { material: 0, layout: Layout::Skinned, first_vertex: 3, vertex_count: 3, first_index: 3, index_count: 3 });
    m.primitives.push(Primitive { material: 0, layout: Layout::Skinned, first_vertex: 6, vertex_count: 3, first_index: 6, index_count: 3 });
    m.validate().unwrap();
    let before: Vec<[f32; 3]> = m.primitives.iter().flat_map(|p| m.indices[p.first_index as usize..(p.first_index + p.index_count) as usize].iter().map(|&i| m.vertices[(p.first_vertex + i) as usize].position)).collect();
    m.merge_primitives();
    m.validate().unwrap();
    assert_eq!(m.primitives.iter().map(|p| (p.layout, p.vertex_count, p.index_count)).collect::<Vec<_>>(), [(Layout::Rigid, 6, 6), (Layout::Skinned, 3, 3)]);
    // The same triangles, in the same order.
    let after: Vec<[f32; 3]> = m.primitives.iter().flat_map(|p| m.indices[p.first_index as usize..(p.first_index + p.index_count) as usize].iter().map(|&i| m.vertices[(p.first_vertex + i) as usize].position)).collect();
    assert_eq!(before, after);
    let rigid = m.gpu_words(&m.primitives[0]);
    assert_eq!(rigid.len(), 6 * RIGID_WORDS);
    assert_eq!(rigid[RIGID_WORDS * 3 + 16], 1, "the piece's joint");
    assert_eq!(f32::from_bits(rigid[RIGID_WORDS * 4]), 1.0, "position x of the second piece's second vertex");
    let skin = m.gpu_words(&m.primitives[1]);
    assert_eq!(skin.len(), 3 * SKINNED_WORDS);
    assert_eq!(&skin[16..20], &[1 << 16, 0, 30000 | 35535 << 16, 0]);
    // A rigid primitive with a blended vertex is refused.
    m.primitives[0].layout = Layout::Rigid;
    m.primitives[1].layout = Layout::Rigid;
    assert!(m.validate().is_err());
}

mod anim_tests {
    use super::*;
    use crate::anim::*;

    /// Root, `left` and `right` under it, `tip` under `left`. Clip 0 moves
    /// `left` and `right` up by 2 over 2 s; clip 1 turns `left` a quarter
    /// about +Y over 1 s; clip 2 lifts `tip` by 1 over 1 s.
    fn model() -> Model {
        let mut m = sample_model();
        m.bones = vec![bone("root", None, [0.0; 3]), bone("left", Some(0), [0.0, 0.0, 1.0]), bone("right", Some(0), [0.0, 0.0, -1.0]), bone("tip", Some(1), [1.0, 0.0, 0.0])];
        m.joints = (0..4).map(|b| Joint { bone: b, inverse_bind: Mat4::IDENTITY.to_cols_array() }).collect();
        let q = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        #[rustfmt::skip]
        let data = vec![
            0.0, 2.0,                     // 0: times of clip 0
            0.0, 0.0, 1.0, 0.0, 2.0, 1.0, // 2: left
            0.0, 0.0, -1.0, 0.0, 2.0, -1.0, // 8: right
            0.0, 1.0,                     // 14: times of clips 1 and 2
            0.0, 0.0, 0.0, 1.0, q.x, q.y, q.z, q.w, // 16: left's turn
            1.0, 0.0, 0.0, 1.0, 1.0, 0.0, // 24: tip's lift
        ];
        let r = |offset, len| Range { offset, len };
        let ch = |bone, path, times, values| Channel { bone, path, interpolation: Interpolation::Linear, times, values };
        m.clips = vec![
            Clip { name: "rise".into(), duration: 2.0, channels: vec![ch(1, Path::Translation, r(0, 2), r(2, 6)), ch(2, Path::Translation, r(0, 2), r(8, 6))] },
            Clip { name: "turn".into(), duration: 1.0, channels: vec![ch(1, Path::Rotation, r(14, 2), r(16, 8))] },
            Clip { name: "lift".into(), duration: 1.0, channels: vec![ch(3, Path::Translation, r(14, 2), r(24, 6))] },
        ];
        m.clip_data = data;
        m.validate().unwrap();
        m
    }

    fn palette_of(rig: &Rig, ch: &Channels) -> Vec<Affine> {
        let mut pose = rig.rest_pose();
        rig.pose(ch, &mut pose);
        let mut g = vec![IDENTITY; rig.bones()];
        rig.globals(&pose, &mut g);
        let mut p = vec![IDENTITY; rig.joints()];
        rig.palette(&g, &mut p);
        p
    }

    fn assert_close(got: &[Affine], want: &[Mat4], eps: f32, what: &str) {
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            let w = affine_of_cols(&w.to_cols_array());
            assert!(g.iter().zip(&w).all(|(a, b)| (a - b).abs() <= eps), "{what}: joint {k}: {g:?} vs {w:?}");
        }
    }

    fn layer(clip: u16, phase: f32) -> Layer {
        // `clip` is the model's index; layers number clips from 1.
        Layer { clip: clip + 1, weight: 1.0, target: 1.0, speed: 1.0, phase, ..Default::default() }
    }

    /// The rest pose, and a clip at full weight, as the glam path draws them.
    #[test]
    fn matches_the_glam_path() {
        let m = model();
        let rig = Rig::new(&m);
        assert_close(&palette_of(&rig, &Channels::default()), &rest_palette(&m), 1e-6, "rest");
        for (clip, t) in [(0u16, 0.0f32), (0, 0.7), (0, 1.99), (1, 0.33), (1, 0.5)] {
            let mut ch = Channels::default();
            ch.layers[0] = layer(clip, t);
            let want = palette(&m, &globals(&m, &bake(&m, clip as usize).locals(&m, t, PlayMode::Loop)));
            assert_close(&palette_of(&rig, &ch), &want, 1e-5, &format!("clip {clip} at {t}"));
        }
        assert_eq!(rig.clip("turn"), Some(2));
        assert_eq!(rig.duration(1), 2.0);
        assert_eq!(rig.duration(0), 0.0);
    }

    /// An override on a mask leaves the other bones; a half-weight blend is
    /// half way; a mask's own weights scale it per bone.
    #[test]
    fn masks_and_weights() {
        let m = model();
        let mut rig = Rig::new(&m);
        let left = rig.bone("left").unwrap();
        let only_left = rig.add_mask("left", rig.subtree(left));
        assert_eq!(rig.mask("left"), Some(only_left));
        assert_eq!(rig.subtree(left), vec![0.0, 1.0, 0.0, 1.0]);
        let mut ch = Channels::default();
        ch.layers[0] = Layer { mask: only_left, ..layer(0, 1.0) };
        let p = palette_of(&rig, &ch);
        assert_eq!(p[1][7], 1.0, "left risen half way");
        assert_eq!(p[2], palette_of(&rig, &Channels::default())[2], "right untouched");
        assert_eq!(p[3][7], 1.0, "the tip rides on left");
        ch.layers[0] = Layer { weight: 0.5, ..layer(0, 2.0 - 1e-6) };
        assert!((palette_of(&rig, &ch)[2][7] - 1.0).abs() < 1e-4, "half weight: half of the full rise");
        let mut w = vec![0.0; 4];
        w[2] = 0.25;
        let quarter = rig.add_mask("quarter right", w);
        ch.layers[0] = Layer { mask: quarter, ..layer(0, 2.0 - 1e-6) };
        let p = palette_of(&rig, &ch);
        assert!((p[2][7] - 0.5).abs() < 1e-4 && p[1][7] == 0.0);
        // A second blend layer over the first.
        ch.layers[0] = layer(0, 1.0);
        ch.layers[1] = Layer { mask: only_left, ..layer(1, 1.0 - 1e-6) };
        let p = palette_of(&rig, &ch);
        assert_eq!(p[2][7], 1.0, "right still rises");
        assert!(p[3][11].abs() < 1e-4 && (p[3][3]).abs() < 1e-4, "left turned a quarter: the tip at {:?}", p[3]);
    }

    /// An additive layer adds its clip's change from its first frame: at
    /// that frame nothing changes, later it adds to what is below.
    #[test]
    fn additive_layers() {
        let m = model();
        let rig = Rig::new(&m);
        let mut ch = Channels::default();
        ch.layers[0] = layer(0, 1.0);
        let base = palette_of(&rig, &ch);
        ch.layers[1] = Layer { kind: kind::ADDITIVE, ..layer(0, 0.0) };
        assert_close(&palette_of(&rig, &ch), &base.iter().map(|a| Mat4::from_cols_array(&cols_of(a))).collect::<Vec<_>>(), 1e-6, "additive at its reference");
        ch.layers[1].phase = 1.0;
        assert_eq!(palette_of(&rig, &ch)[1][7], 2.0, "half way plus half way");
        ch.layers[1].weight = 0.5;
        assert_eq!(palette_of(&rig, &ch)[1][7], 1.5);
    }

    fn cols_of(a: &Affine) -> [f32; 16] {
        [a[0], a[4], a[8], 0.0, a[1], a[5], a[9], 0.0, a[2], a[6], a[10], 0.0, a[3], a[7], a[11], 1.0]
    }

    /// The ticker: fades, phases, play modes, ground speed.
    #[test]
    fn advance_layers() {
        let rig = Rig::new(&model());
        let dt = 1.0 / 30.0;
        let mut ch = Channels::default();
        ch.layers[0] = Layer { weight: 0.0, target: 1.0, fade: 5.0, ..layer(1, 0.0) };
        let mut ticks = 0;
        while ch.layers[0].weight < 1.0 {
            rig.advance(&mut ch, dt, 0.0);
            ticks += 1;
        }
        assert_eq!(ticks, 6, "a 200 ms blend-in");
        assert!((ch.layers[0].phase - 0.2).abs() < 1e-5);
        for _ in 0..30 {
            rig.advance(&mut ch, dt, 0.0);
        }
        assert!(ch.layers[0].phase >= 0.0 && ch.layers[0].phase < 1.0, "looped: {}", ch.layers[0].phase);
        assert!((ch.layers[0].phase - 0.2).abs() < 1e-4);
        // Fading out clears the layer.
        ch.layers[0].target = 0.0;
        for _ in 0..6 {
            rig.advance(&mut ch, dt, 0.0);
        }
        assert!(ch.layers[0].is_empty());
        // Once: ends and holds.
        ch.layers[0] = Layer { mode: PlayMode::Once as u8, ..layer(1, 0.9) };
        rig.advance(&mut ch, dt, 0.0);
        assert!(!ch.layers[0].ended());
        for _ in 0..3 {
            rig.advance(&mut ch, dt, 0.0);
        }
        assert!(ch.layers[0].ended() && ch.layers[0].phase == 1.0);
        // Ping-pong keeps two lengths; ground speed scales the clip.
        ch.layers[0] = Layer { mode: PlayMode::PingPong as u8, ..layer(1, 1.9) };
        rig.advance(&mut ch, 0.2, 0.0);
        assert!((ch.layers[0].phase - 0.1).abs() < 1e-5);
        ch.layers[0] = Layer { flags: layer_flags::GROUND_SPEED, speed: 0.1, ..layer(0, 0.0) };
        rig.advance(&mut ch, 0.5, 6.0);
        assert!((ch.layers[0].phase - 0.3).abs() < 1e-6, "6 m/s for 0.5 s at 0.1 s/m");
    }

    /// A command as the SDK leaves it: pending, and fresh unless it replaces
    /// an entry.
    fn command(bone: u16, kind: u8, target: [f32; 4], speed: f32) -> Manual {
        Manual { bone, kind, flags: manual_flags::PENDING | manual_flags::FRESH, target, speed, ..Default::default() }
    }

    fn reissue(m: &mut Manual, target: [f32; 4]) {
        (m.flags, m.target) = (manual_flags::PENDING, target);
    }

    fn ticks_until_done(rig: &Rig, ch: &mut Channels, k: usize) -> u32 {
        let mut ticks = 0;
        while !ch.manual[k].done() {
            rig.advance(ch, 1.0 / 30.0, 0.0);
            ticks += 1;
            assert!(ticks < 1000);
        }
        ticks
    }

    /// Turns: in the ticks their speed needs; the long way past 180°; with
    /// acceleration and deceleration; about a model axis through the bone's
    /// rest rotation; a command given again goes on from where it is.
    #[test]
    fn manual_turns() {
        let rig = Rig::new(&model());
        let pi = std::f32::consts::PI;
        let up = |angle| [0.0, 1.0, 0.0, angle];
        let mut ch = Channels::default();
        ch.manual[0] = command(1, manual_kind::TURN, up(pi / 2.0), pi);
        assert_eq!(ticks_until_done(&rig, &mut ch, 0), 15, "a quarter turn at half a turn a second");
        let p = palette_of(&rig, &ch);
        assert!(p[3][11].abs() < 1e-5 && p[3][3].abs() < 1e-5, "tip turned onto -Z of left: {:?}", p[3]);
        // Back to rest, then to 270°: the long way, through +90°.
        reissue(&mut ch.manual[0], up(0.0));
        ticks_until_done(&rig, &mut ch, 0);
        reissue(&mut ch.manual[0], up(1.5 * pi));
        for _ in 0..15 {
            rig.advance(&mut ch, 1.0 / 30.0, 0.0);
        }
        let p = palette_of(&rig, &ch);
        assert!(p[3][11].abs() < 1e-4 && p[3][3].abs() < 1e-4, "through +90°: {:?}", p[3]);
        assert_eq!(ticks_until_done(&rig, &mut ch, 0), 30);
        let p = palette_of(&rig, &ch);
        assert!((p[3][11] - 2.0).abs() < 1e-4 && p[3][3].abs() < 1e-4, "at 270° (-90°): {:?}", p[3]);
        // Ramps: from rest with 2π/s² up and down, a quarter turn takes
        // 2·sqrt(θ/a) = 1 s, peaking at π/s (give or take a tick).
        let mut ch = Channels::default();
        ch.manual[0] = Manual { accel: 2.0 * pi, decel: 2.0 * pi, ..command(1, manual_kind::TURN, up(pi / 2.0), 10.0) };
        let mut peak = 0.0f32;
        let mut ticks = 0;
        while !ch.manual[0].done() {
            rig.advance(&mut ch, 1.0 / 30.0, 0.0);
            peak = peak.max(ch.manual[0].rate);
            ticks += 1;
        }
        assert!((27..=31).contains(&ticks), "{ticks} ticks");
        assert!(peak > 0.9 * pi && peak < 1.01 * pi, "top speed reached {peak}");
        // Given again mid-way, a turn goes on as if it had not been.
        let mut a = Channels::default();
        a.manual[0] = Manual { accel: 2.0 * pi, ..command(1, manual_kind::TURN, up(pi / 2.0), pi) };
        for _ in 0..5 {
            rig.advance(&mut a, 1.0 / 30.0, 0.0);
        }
        let mut b = a;
        reissue(&mut b.manual[0], up(pi / 2.0));
        b.manual[0].accel = 2.0 * pi;
        rig.advance(&mut a, 1.0 / 30.0, 0.0);
        rig.advance(&mut b, 1.0 / 30.0, 0.0);
        assert_eq!(a, b);
        // `left` lying on its side at rest (a quarter about +X, so its +Z is
        // the model's -Y): a turn about the model's up still swings the tip round it,
        // and about the bone's own Z the same turn is the other way round.
        let mut m = model();
        let q = Quat::from_rotation_x(pi / 2.0);
        m.bones[1].rotation = q.to_array();
        m.bones[1].rest = (Mat4::from_translation(Vec3::new(0.0, 0.0, 1.0)) * Mat4::from_quat(q)).to_cols_array();
        let rig = Rig::new(&m);
        let mut ch = Channels::default();
        ch.manual[0] = command(1, manual_kind::TURN, up(pi / 2.0), 0.0);
        rig.advance(&mut ch, 1.0 / 30.0, 0.0);
        let p = palette_of(&rig, &ch);
        assert!(p[3][3].abs() < 1e-5 && p[3][7].abs() < 1e-5 && p[3][11].abs() < 1e-5, "tip turned onto -Z of left: {:?}", p[3]);
        ch.manual[0] = Manual { flags: manual_flags::PENDING | manual_flags::FRESH | manual_flags::BONE_AXIS, ..command(1, manual_kind::TURN, [0.0, 0.0, 1.0, pi / 2.0], 0.0) };
        rig.advance(&mut ch, 1.0 / 30.0, 0.0);
        let p = palette_of(&rig, &ch);
        assert!((p[3][11] - 2.0).abs() < 1e-5, "about the bone's Z (the model's -Y): onto +Z: {:?}", p[3]);
    }

    /// Aiming: a direction as a bone sees it, with its parents where they
    /// are going (a pending turn included, a clip's value now); a short turn
    /// goes the short way whatever the angles say.
    #[test]
    fn aiming() {
        let rig = Rig::new(&model());
        let pi = std::f32::consts::PI;
        let close = |a: [f32; 3], b: [f32; 3]| a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5);
        let mut ch = Channels::default();
        assert!(close(rig.dir_in(&ch, 3, [1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]), "parents at rest");
        // `left` told to turn a quarter about up: the tip sees the model's +X
        // where its own +Z was at rest.
        ch.manual[0] = command(1, manual_kind::TURN, [0.0, 1.0, 0.0, pi / 2.0], 1.0);
        let d = rig.dir_in(&ch, 3, [1.0, 0.0, 0.0]);
        assert!(close(d, [0.0, 0.0, 1.0]), "{d:?}");
        rig.advance(&mut ch, 1.0 / 30.0, 0.0);
        assert!(close(rig.dir_in(&ch, 3, [1.0, 0.0, 0.0]), d), "where it is going, not where it is");
        // A clip turning `left` a quarter: the same, now.
        let mut ch = Channels::default();
        ch.layers[0] = layer(1, 1.0 - 1e-6);
        let d = rig.dir_in(&ch, 3, [1.0, 0.0, 0.0]);
        assert!(close(d, [0.0, 0.0, 1.0]), "{d:?}");
        // From 170° to -170°: the long way (340°) by the numbers, 20° short.
        let up = |deg: f32| [0.0, 1.0, 0.0, deg.to_radians()];
        let mut ch = Channels::default();
        ch.manual[0] = command(1, manual_kind::TURN, up(170.0), 0.0);
        rig.advance(&mut ch, 1.0 / 30.0, 0.0);
        let mut long = ch;
        reissue(&mut long.manual[0], up(-170.0));
        long.manual[0].speed = 1.0;
        let mut short = long;
        short.manual[0].flags |= manual_flags::SHORT;
        let (lt, st) = (ticks_until_done(&rig, &mut long, 0), ticks_until_done(&rig, &mut short, 0));
        assert_eq!((lt, st), (179, 11), "340° and 20° at 1 rad/s (178.02 and 10.5 ticks)");
        let (pl, ps) = (palette_of(&rig, &long), palette_of(&rig, &short));
        assert!(pl.iter().flatten().zip(ps.iter().flatten()).all(|(a, b)| (a - b).abs() < 1e-4), "the same place");
    }

    /// A bone with a manual entry takes no clip (its children still do); a
    /// clip played later takes it back; a turn on a clipped bone starts where
    /// the clip had it; spins, moves and scales.
    #[test]
    fn manual_and_clips() {
        let rig = Rig::new(&model());
        let pi = std::f32::consts::PI;
        let dt = 1.0 / 30.0;
        let mut ch = Channels::default();
        // Clip "turn" has `left` a quarter round; "lift" raises `tip` (held).
        ch.layers[0] = Layer { speed: 0.0, ..layer(1, 1.0 - 1e-6) };
        ch.layers[1] = Layer { speed: 0.0, ..layer(2, 1.0 - 1e-6) };
        let quarter = palette_of(&rig, &ch);
        ch.manual[0] = command(1, manual_kind::TURN, [0.0, 0.0, 1.0, 0.0], 1.0);
        assert_eq!(palette_of(&rig, &ch), quarter, "a fresh entry shows nothing yet");
        let shown = rig.clone();
        let mut started = ch;
        shown.advance(&mut started, 0.0, 0.0);
        let p = palette_of(&rig, &started);
        assert!(p.iter().flatten().zip(quarter.iter().flatten()).all(|(a, b)| (a - b).abs() < 1e-5), "started where the clip had it");
        ticks_until_done(&rig, &mut started, 0);
        let p = palette_of(&rig, &started);
        assert!((p[3][3] - 1.0).abs() < 1e-4 && (p[3][11] - 1.0).abs() < 1e-4, "left back at rest, the clip's turn gone: {:?}", p[3]);
        assert!((p[3][7] - 1.0).abs() < 1e-4, "the tip still lifted by its clip: {:?}", p[3]);
        // Playing a clip on `left` takes it back.
        started.layers[0].flags |= layer_flags::NEW;
        rig.advance(&mut started, 0.0, 0.0);
        assert!(started.manual[0].is_empty() && started.layers[0].flags & layer_flags::NEW == 0);
        // ... but not from a command given in the same tick, nor an additive
        // clip, nor a clip without the bone.
        started.manual[0] = command(1, manual_kind::TURN, [0.0, 1.0, 0.0, 0.0], 0.0);
        started.layers[0].flags |= layer_flags::NEW;
        rig.advance(&mut started, 0.0, 0.0);
        assert!(!started.manual[0].is_empty());
        started.layers[1].flags |= layer_flags::NEW;
        started.layers[2] = Layer { kind: kind::ADDITIVE, flags: layer_flags::NEW, ..layer(1, 0.0) };
        rig.advance(&mut started, 0.0, 0.0);
        assert!(!started.manual[0].is_empty());
        // A spin speeds up and down by its ramps, staying on the rest side.
        let mut ch = Channels::default();
        ch.manual[3] = Manual { accel: 8.0 * pi, decel: 8.0 * pi, ..command(2, manual_kind::SPIN, [0.0, 1.0, 0.0, 0.0], 4.0 * pi) };
        assert_eq!(ticks_until_done(&rig, &mut ch, 3), 15);
        assert_eq!(ch.manual[3].rate, 4.0 * pi);
        for _ in 0..40 {
            rig.advance(&mut ch, dt, 0.0);
            assert!(ch.manual[3].value[3] >= 0.0 && ch.manual[3].done());
        }
        ch.manual[3].speed = 0.0;
        assert_eq!(ticks_until_done(&rig, &mut ch, 3), 15);
        // Moves (an offset in the model's frame) and scales, at once.
        ch.manual[0] = command(1, manual_kind::MOVE, [0.0, 2.0, 0.0, 0.0], 0.0);
        ch.manual[1] = command(1, manual_kind::SCALE, [2.0, 2.0, 2.0, 0.0], 0.0);
        rig.advance(&mut ch, dt, 0.0);
        let p = palette_of(&rig, &ch);
        assert_eq!((p[1][7], p[1][0], p[1][11]), (2.0, 2.0, 1.0));
        assert!((p[3][3] - 2.0).abs() < 1e-5, "the tip rides on the scaled bone: {:?}", p[3]);
    }

    /// The same channels give the same bits.
    #[test]
    fn repeatable() {
        let rig = Rig::new(&model());
        let mut ch = Channels::default();
        ch.layers[0] = layer(0, 0.3);
        ch.layers[1] = Layer { kind: kind::ADDITIVE, weight: 0.3, ..layer(1, 0.6) };
        ch.manual[0] = Manual { accel: 3.0, decel: 2.0, ..command(3, manual_kind::TURN, [0.3, 0.0, 1.0, 1.0], 2.0) };
        ch.manual[1] = Manual { accel: 3.0, ..command(2, manual_kind::SPIN, [1.0, 1.0, 0.0, 0.0], 5.0) };
        let mut a = ch;
        let mut b = ch;
        for _ in 0..100 {
            rig.advance(&mut a, 1.0 / 30.0, 1.5);
            rig.advance(&mut b, 1.0 / 30.0, 1.5);
        }
        assert_eq!(a, b);
        let (pa, pb) = (palette_of(&rig, &a), palette_of(&rig, &b));
        assert!(pa.iter().flatten().zip(pb.iter().flatten()).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}
