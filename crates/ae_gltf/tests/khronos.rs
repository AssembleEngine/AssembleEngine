//! The Khronos sample models in `assets/models/khronos` converted, written,
//! read back and posed.

use ae_gltf::convert_file;
use ae_model::pose::{Pose, globals, palette};
use ae_model::{Model, ktx};
use glam::Vec3;
use std::path::PathBuf;

fn asset(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/khronos").join(rel)
}

fn convert(rel: &str) -> (Model, Vec<String>) {
    let c = convert_file(&asset(rel)).unwrap();
    let back = Model::from_bytes(&c.model.to_bytes()).unwrap();
    assert_eq!(back, c.model, "{rel}: round trip");
    (back, c.warnings)
}

/// Model-space positions under `palette`.
fn skinned(m: &Model, pal: &[glam::Mat4]) -> Vec<Vec3> {
    m.vertices
        .iter()
        .map(|v| (0..4).map(|k| pal[v.joints[k] as usize].transform_point3(Vec3::from(v.position)) * (v.weights[k] as f32 / 65535.0)).sum())
        .collect()
}

fn posed(m: &Model, clip: usize, t: f32) -> Vec<Vec3> {
    let mut pose = Pose::rest(m);
    pose.sample(m, clip, t);
    skinned(m, &palette(m, &globals(m, &pose.locals(m))))
}

#[test]
fn simple_skin() {
    let (m, warnings) = convert("SimpleSkin/SimpleSkin.gltf");
    assert!(warnings.is_empty(), "{warnings:?}");
    // Root + the three nodes; the skin's two joints (the mesh node adds
    // none: its mesh is skinned).
    assert_eq!((m.bones.len(), m.joints.len(), m.vertices.len(), m.indices.len()), (4, 2, 10, 24));
    assert_eq!(m.clips.len(), 1);
    // The clip bends the upper half about the joint at height 1; the
    // bottom row stays.
    let rest = posed(&m, 0, 0.0);
    let bent = posed(&m, 0, 2.0);
    for (a, b) in rest.iter().zip(&bent) {
        if a.y < 0.01 {
            assert!(a.abs_diff_eq(*b, 1e-5), "{a} moved to {b}");
        }
    }
    let top = rest.iter().zip(&bent).filter(|(a, _)| a.y > 1.99).map(|(a, b)| a.distance(*b)).fold(0.0f32, f32::max);
    assert!(top > 0.3, "the top did not move ({top})");
    // Bending keeps the distance from the joint (a rotation about it).
    let joint = Vec3::new(0.0, 1.0, 0.0);
    for (a, b) in rest.iter().zip(&bent) {
        if a.y > 1.99 {
            assert!((a.distance(joint) - b.distance(joint)).abs() < 1e-4);
        }
    }
}

#[test]
fn box_animated_pieces() {
    let (m, warnings) = convert("BoxAnimated/BoxAnimated.glb");
    assert!(warnings.is_empty(), "{warnings:?}");
    // Two meshes on two nodes: two pieces, every vertex on one joint at
    // weight 1 with the identity inverse bind.
    assert_eq!(m.primitives.len(), 2);
    assert_eq!(m.joints.len(), 2);
    for j in &m.joints {
        assert_eq!(j.inverse_bind, glam::Mat4::IDENTITY.to_cols_array());
    }
    assert!(m.vertices.iter().all(|v| v.weights == [65535, 0, 0, 0]));
    let clip = &m.clips[0];
    assert_eq!(clip.channels.len(), 2);
    // Over the clip one piece moves and the other turns: positions change.
    let rest = posed(&m, 0, 0.0);
    let mid = posed(&m, 0, clip.duration * 0.5);
    let moved = rest.iter().zip(&mid).filter(|(a, b)| a.distance(**b) > 1e-3).count();
    assert!(moved > 0);
}

#[test]
fn fox_textured_skin_and_clips() {
    let (m, warnings) = convert("Fox/Fox.glb");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(m.clips.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["Survey", "Walk", "Run"]);
    assert_eq!(m.joints.len(), 24, "the skin's joints only");
    let tex = ktx::read(&m.textures[0]).unwrap();
    assert_eq!((tex.format, tex.width, tex.height, tex.levels.len()), (ktx::Format::RGBA8_SRGB, 1024, 1024, 11));
    // Standing on the ground, longer than wide, and long along the
    // engine's forward axis (glTF +Z): the axis turn applied.
    let b = m.bounds;
    let size = Vec3::from(b.max) - Vec3::from(b.min);
    assert!(b.min[1].abs() < 1.0 && size.x > 2.0 * size.z, "{b:?}");
    // Every clip keeps the feet near the ground and moves the model.
    for (c, clip) in m.clips.iter().enumerate() {
        let p = posed(&m, c, clip.duration * 0.37);
        let min_y = p.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        assert!(min_y.abs() < 10.0, "{}: lowest point {min_y}", clip.name);
    }
}

/// The sim's animation path (`ae_model::anim`, scalar and libm) poses the
/// Fox as the glam path does: every clip at several times, and the rest.
#[test]
fn fox_anim_rig_matches_the_glam_path() {
    use ae_model::anim::{Channels, IDENTITY, Layer, Rig, affine_of_cols};
    use ae_model::pose::{PlayMode, bake, rest_palette};
    let (m, _) = convert("Fox/Fox.glb");
    let rig = Rig::new(&m);
    let run = |ch: &Channels| {
        let mut pose = rig.rest_pose();
        rig.pose(ch, &mut pose);
        let mut g = vec![IDENTITY; rig.bones()];
        rig.globals(&pose, &mut g);
        let mut p = vec![IDENTITY; rig.joints()];
        rig.palette(&g, &mut p);
        p
    };
    let check = |got: &[[f32; 12]], want: &[glam::Mat4], what: &str| {
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            let w = affine_of_cols(&w.to_cols_array());
            // Relative to the matrix's largest entry (Fox units are centimetres).
            let eps = 1e-4 * (1.0 + w.iter().fold(0.0f32, |a, x| a.max(x.abs())));
            assert!(g.iter().zip(&w).all(|(a, b)| (a - b).abs() <= eps), "{what}: joint {k}: {g:?} vs {w:?}");
        }
    };
    check(&run(&Channels::default()), &rest_palette(&m), "rest");
    for (c, clip) in m.clips.iter().enumerate() {
        let baked = bake(&m, c);
        for f in [0.0, 0.13, 0.5, 0.91] {
            let t = clip.duration * f;
            let mut ch = Channels::default();
            ch.layers[0] = Layer { clip: c as u16 + 1, weight: 1.0, target: 1.0, speed: 1.0, phase: t, ..Default::default() };
            check(&run(&ch), &ae_model::pose::palette(&m, &globals(&m, &baked.locals(&m, t, PlayMode::Loop))), &format!("{} at {t}", clip.name));
        }
    }
}

/// `Rig::bounds` (the sim's bounds query) is the box of the Fox skinned by
/// the glam path: at rest the converter's rest bounds; in every clip at
/// several times, and with a manual head turn, every vertex inside it and
/// each face touched by some vertex.
#[test]
fn fox_rig_bounds_box_the_posed_mesh() {
    use ae_model::anim::{Channels, IDENTITY, Layer, Manual, Rig, manual_kind};
    use ae_model::pose::{PlayMode, bake};
    let (m, _) = convert("Fox/Fox.glb");
    let rig = Rig::new(&m);
    // Fox units are centimetres: 1e-4 of its size.
    let eps = 1e-4 * (Vec3::from(m.bounds.max) - Vec3::from(m.bounds.min)).length();
    let check = |b: ae_model::Bounds, p: &[Vec3], what: &str| {
        let (lo, hi) = (Vec3::from(b.min), Vec3::from(b.max));
        assert!(p.iter().all(|v| v.cmpge(lo - eps).all() && v.cmple(hi + eps).all()), "{what}: a vertex outside {b:?}");
        let min = p.iter().fold(Vec3::INFINITY, |a, v| a.min(*v));
        let max = p.iter().fold(Vec3::NEG_INFINITY, |a, v| a.max(*v));
        assert!(min.abs_diff_eq(lo, eps) && max.abs_diff_eq(hi, eps), "{what}: {b:?} vs {min} {max}");
    };
    let rest = rig.bounds(&Channels::default());
    check(rest, &skinned(&m, &ae_model::pose::rest_palette(&m)), "rest");
    assert!(Vec3::from(rest.min).abs_diff_eq(Vec3::from(m.bounds.min), eps) && Vec3::from(rest.max).abs_diff_eq(Vec3::from(m.bounds.max), eps));
    for (c, clip) in m.clips.iter().enumerate() {
        let baked = bake(&m, c);
        for f in [0.0, 0.29, 0.77] {
            let t = clip.duration * f;
            let mut ch = Channels::default();
            ch.layers[0] = Layer { clip: c as u16 + 1, weight: 1.0, target: 1.0, speed: 1.0, phase: t, ..Default::default() };
            let pal = palette(&m, &globals(&m, &baked.locals(&m, t, PlayMode::Loop)));
            check(rig.bounds(&ch), &skinned(&m, &pal), &format!("{} at {t}", clip.name));
        }
    }
    // The head turned (manual, over the walk): checked against the scalar
    // path's own palette, skinned by glam.
    let head = rig.bone("b_Head_05").unwrap();
    let mut ch = Channels::default();
    ch.layers[0] = Layer { clip: 2, weight: 1.0, target: 1.0, speed: 1.0, phase: 0.3, ..Default::default() };
    let q = glam::Quat::from_array(rig.rest()[head as usize].r) * glam::Quat::from_rotation_y(1.2);
    ch.manual[0] = Manual { bone: head, kind: manual_kind::TURN, value: q.to_array(), target: q.to_array(), ..Default::default() };
    let mut pose = rig.rest_pose();
    rig.pose(&ch, &mut pose);
    let mut g = vec![IDENTITY; rig.bones()];
    rig.globals(&pose, &mut g);
    let mut p = vec![IDENTITY; rig.joints()];
    rig.palette(&g, &mut p);
    let pal: Vec<glam::Mat4> = p.iter().map(|a| glam::Mat4::from_cols_array(&[a[0], a[4], a[8], 0.0, a[1], a[5], a[9], 0.0, a[2], a[6], a[10], 0.0, a[3], a[7], a[11], 1.0])).collect();
    let turned = rig.bounds(&ch);
    check(turned, &skinned(&m, &pal), "head turned");
    let mut walk = ch;
    walk.manual[0] = Manual::default();
    assert_ne!(turned, rig.bounds(&walk), "the turn moves the box");
}
