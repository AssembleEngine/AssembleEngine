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
