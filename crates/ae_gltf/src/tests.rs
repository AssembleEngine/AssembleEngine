//! glTFs built here (embedded buffers), converted and checked.

use super::*;
use ae_model::ktx;
use serde_json::{Value, json};

/// A glTF under construction: one buffer, accessors appended.
#[derive(Default)]
struct Builder {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Builder {
    fn view(&mut self, bytes: &[u8]) -> usize {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        self.views.push(json!({"buffer": 0, "byteOffset": self.bin.len(), "byteLength": bytes.len()}));
        self.bin.extend_from_slice(bytes);
        self.views.len() - 1
    }

    fn floats(&mut self, data: &[f32], ty: &str) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let v = self.view(&bytes);
        let comps = match ty {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            _ => 16,
        };
        let mut a = json!({"bufferView": v, "componentType": 5126, "count": data.len() / comps, "type": ty});
        if ty == "VEC3" {
            let mut min = [f32::INFINITY; 3];
            let mut max = [f32::NEG_INFINITY; 3];
            for p in data.chunks(3) {
                for k in 0..3 {
                    min[k] = min[k].min(p[k]);
                    max[k] = max[k].max(p[k]);
                }
            }
            a["min"] = json!(min);
            a["max"] = json!(max);
        }
        self.accessors.push(a);
        self.accessors.len() - 1
    }

    fn u16s(&mut self, data: &[u16], ty: &str, count: usize) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let v = self.view(&bytes);
        self.accessors.push(json!({"bufferView": v, "componentType": 5123, "count": count, "type": ty}));
        self.accessors.len() - 1
    }

    /// The document with `rest` merged in; the buffer as a data URI.
    fn finish(self, mut rest: Value) -> Vec<u8> {
        use base64::Engine as _;
        let uri = format!("data:application/octet-stream;base64,{}", base64::engine::general_purpose::STANDARD.encode(&self.bin));
        rest["asset"] = json!({"version": "2.0"});
        rest["buffers"] = json!([{"byteLength": self.bin.len(), "uri": uri}]);
        rest["bufferViews"] = Value::Array(self.views);
        rest["accessors"] = Value::Array(self.accessors);
        serde_json::to_vec(&rest).unwrap()
    }
}

/// A triangle facing glTF +Y, its tip at glTF +Z (the model's front).
const TRI: [f32; 9] = [-0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0];

fn rest_positions(m: &Model) -> Vec<Vec3> {
    let pal = pose::rest_palette(m);
    m.vertices
        .iter()
        .map(|v| {
            let mut q = Vec3::ZERO;
            for k in 0..4 {
                q += pal[v.joints[k] as usize].transform_point3(Vec3::from(v.position)) * (v.weights[k] as f32 / 65535.0);
            }
            q
        })
        .collect()
}

/// Punctual lights: a point light and a turned spot become model lights on
/// their nodes' joints (engine position and direction from the pose), a
/// directional light is ignored with a warning naming it, and the model
/// round-trips with its lights.
#[test]
fn punctual_lights_ride_their_nodes() {
    use ae_model::LightKind;

    let mut b = Builder::default();
    let pos = b.floats(&TRI, "VEC3");
    let quarter = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let gltf = b.finish(json!({
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 2, 3]}],
        "extensionsUsed": ["KHR_lights_punctual"],
        "extensions": {"KHR_lights_punctual": {"lights": [
            {"type": "point", "color": [0.5, 0.4, 0.3], "intensity": 40.0, "range": 25.0},
            {"type": "spot", "color": [1.0, 1.0, 1.0], "intensity": 10.0, "spot": {"innerConeAngle": 0.2, "outerConeAngle": 0.5}},
            {"type": "directional", "color": [1.0, 1.0, 1.0], "intensity": 3.0}
        ]}},
        "nodes": [
            {"name": "hull", "mesh": 0},
            {"name": "lamp", "translation": [1.0, 2.0, 3.0], "extensions": {"KHR_lights_punctual": {"light": 0}}},
            {"name": "headlight", "translation": [-2.0, 1.0, 0.0], "rotation": [quarter.x, quarter.y, quarter.z, quarter.w], "extensions": {"KHR_lights_punctual": {"light": 1}}},
            {"name": "sun", "extensions": {"KHR_lights_punctual": {"light": 2}}}
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": pos}}]}]
    }));
    let c = convert(&gltf, None).unwrap();
    let m = &c.model;
    assert_eq!(m.lights.len(), 2, "the directional light is not a model light");
    assert!(c.warnings.iter().any(|w| w.contains("sun") && w.contains("directional")), "{:?}", c.warnings);

    let point = &m.lights[0];
    assert_eq!(point.kind, LightKind::Point);
    assert_eq!(point.name, "light (unnamed) on node lamp");
    assert!(Vec3::from(point.color).abs_diff_eq(Vec3::new(0.5, 0.4, 0.3), 1e-6), "{:?}", point.color);
    assert_eq!(point.intensity, 40.0);
    assert_eq!((point.range, point.inner_angle, point.outer_angle), (25.0, 0.0, 0.0));
    assert_eq!(point.size, 25.0 / 8.0);

    // Its joint is the lamp node: the rest palette puts it where the node
    // is, and the node's local -Z is the light's direction.
    let palette = pose::rest_palette(m);
    let m4 = palette[point.joint as usize];
    let want = GLTF_TO_ENGINE * Vec3::new(1.0, 2.0, 3.0);
    assert!(m4.transform_point3(Vec3::ZERO).abs_diff_eq(want, 1e-5), "{} vs {want}", m4.transform_point3(Vec3::ZERO));
    // The hull's joint stays at the origin (no translation crept in).
    assert!(palette[m.joints[0].bone as usize].transform_point3(Vec3::ZERO).length() < 1e-6);

    // The spot: two angles, and a quarter turn about glTF +Y turns its
    // local -Z to glTF -X (engine -Z, since glTF -X maps to engine -Z?):
    // root (+Z to +X) after the turn: -Z -> -X -> engine -Z.
    let spot = &m.lights[1];
    assert_eq!(spot.kind, LightKind::Spot);
    assert!(spot.inner_angle == 0.2 && spot.outer_angle == 0.5);
    // No range in the glTF: no authored cutoff, and the default source
    // radius.
    assert_eq!((spot.range, spot.size), (0.0, ae_model::DEFAULT_SOURCE_RADIUS));
    let m4 = palette[spot.joint as usize];
    let dir = m4.transform_vector3(Vec3::NEG_Z).normalize();
    let want = (GLTF_TO_ENGINE * (quarter * Vec3::NEG_Z)).normalize();
    assert!(dir.abs_diff_eq(want, 1e-5), "{dir} vs {want}");
    let _ = LightKind::Spot;

    // The round trip keeps them.
    let back = Model::from_bytes(&m.to_bytes()).unwrap();
    assert_eq!(&back, m);
}

#[test]
fn pieces_and_axes() {
    let mut b = Builder::default();
    let pos = b.floats(&TRI, "VEC3");
    // Parent moved 2 along glTF +Z; the child turned a quarter about +Y
    // (glTF +Z to +X) and carrying the mesh. No normals: flat ones.
    let q = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let gltf = b.finish(json!({
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "hull", "translation": [0.0, 0.0, 2.0], "children": [1]},
            {"name": "turret", "rotation": [q.x, q.y, q.z, q.w], "scale": [1.0, 2.0, 1.0], "mesh": 0}
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": pos}}]}]
    }));
    let c = convert(&gltf, None).unwrap();
    let m = &c.model;
    assert_eq!(m.bones.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(), ["__gltf_root", "hull", "turret"]);
    assert_eq!(m.bones[2].parent, Some(1));
    assert_eq!(m.joints.len(), 1);
    assert_eq!(m.joints[0].bone, 2);
    assert_eq!(m.materials.len(), 1, "the default material");
    // Expected: glTF world positions, then glTF -> engine.
    let world = Mat4::from_translation(Vec3::new(0.0, 0.0, 2.0)) * Mat4::from_scale_rotation_translation(Vec3::new(1.0, 2.0, 1.0), q, Vec3::ZERO);
    let got = rest_positions(m);
    for (i, p) in TRI.chunks(3).enumerate() {
        let want = GLTF_TO_ENGINE * world.transform_point3(Vec3::from_slice(p));
        assert!(got[i].abs_diff_eq(want, 1e-5), "{i}: {} vs {want}", got[i]);
    }
    // The front: the hull sits 2 m ahead (engine +X), not to a side.
    let centre = got.iter().copied().sum::<Vec3>() / 3.0;
    assert!(centre.x > 1.5 && centre.z.abs() < 1.0, "{centre}");
    // Flat normal of a triangle facing glTF +Y.
    for v in &m.vertices {
        assert!(Vec3::from(v.normal).abs_diff_eq(Vec3::Y, 1e-6) || Vec3::from(v.normal).abs_diff_eq(-Vec3::Y, 1e-6));
    }
    assert!(Vec3::from(m.bounds.max).x > 1.5);
    assert!(c.warnings.is_empty(), "{:?}", c.warnings);
}

#[test]
fn skin_reduced_to_four_influences() {
    let mut b = Builder::default();
    let pos = b.floats(&TRI, "VEC3");
    let nrm = b.floats(&[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], "VEC3");
    // Vertex 0: six influences (joints 0..5), weights 6..1; the others on joint 1.
    let j0 = b.u16s(&[0, 1, 2, 3, 1, 0, 0, 0, 1, 0, 0, 0], "VEC4", 3);
    let w0 = b.floats(&[6.0 / 21.0, 5.0 / 21.0, 4.0 / 21.0, 3.0 / 21.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "VEC4");
    let j1 = b.u16s(&[4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "VEC4", 3);
    let w1 = b.floats(&[2.0 / 21.0, 1.0 / 21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "VEC4");
    let uv = b.floats(&[0.0; 6], "VEC2");
    let col = b.floats(&[1.0; 12], "VEC4");
    let mut ibm = Vec::new();
    for k in 0..6 {
        ibm.extend(Mat4::from_translation(Vec3::new(-(k as f32), 0.0, 0.0)).to_cols_array());
    }
    let ibm = b.floats(&ibm, "MAT4");
    let joints: Vec<Value> = (0..6).map(|k| json!({"name": format!("j{k}"), "translation": [if k == 0 { 0.0 } else { 1.0 }, 0.0, 0.0], "children": if k < 5 { json!([k + 2]) } else { json!([]) }})).collect();
    let mut nodes = vec![json!({"name": "body", "mesh": 0, "skin": 0, "translation": [50.0, 0.0, 0.0]})];
    nodes.extend(joints);
    let gltf = b.finish(json!({
        "scenes": [{"nodes": [0, 1]}],
        "nodes": nodes,
        "skins": [{"joints": [1, 2, 3, 4, 5, 6], "inverseBindMatrices": ibm}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": pos, "NORMAL": nrm, "JOINTS_0": j0, "WEIGHTS_0": w0, "JOINTS_1": j1, "WEIGHTS_1": w1, "TEXCOORD_2": uv, "COLOR_1": col}}]}]
    }));
    let c = convert(&gltf, None).unwrap();
    let m = &c.model;
    let w = |s: &str| c.warnings.iter().any(|x| x.contains(s));
    assert!(w("more than 4 influences") && w("TEXCOORD_2 dropped") && w("COLOR_1 dropped"), "{:?}", c.warnings);
    let v0 = m.vertices[0];
    assert_eq!(v0.weights.iter().map(|&w| w as u32).sum::<u32>(), 65535);
    let names: Vec<&str> = v0.joints.iter().map(|&j| m.bones[m.joints[j as usize].bone as usize].name.as_str()).collect();
    assert_eq!(names, ["j0", "j1", "j2", "j3"]);
    // 6:5:4:3 renormalised over 18.
    assert!((v0.weights[0] as f32 / 65535.0 - 6.0 / 18.0).abs() < 1e-4);
    // The mesh node's own transform (50 m) is ignored for a skinned mesh:
    // at rest (bind = rest), positions are the glTF ones, turned.
    for (i, p) in rest_positions(m).iter().enumerate() {
        let want = GLTF_TO_ENGINE * Vec3::from_slice(&TRI[i * 3..i * 3 + 3]);
        assert!(p.abs_diff_eq(want, 1e-4), "{i}: {p} vs {want}");
    }
}

#[test]
fn extensions_required_and_used() {
    let mut b = Builder::default();
    let pos = b.floats(&TRI, "VEC3");
    let doc = |required: &[&str], used: &[&str]| {
        json!({
            "extensionsRequired": required, "extensionsUsed": used,
            "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": pos}}]}]
        })
    };
    let bin = std::mem::take(&mut b.bin);
    let views = b.views.clone();
    let accessors = b.accessors.clone();
    let make = |d: Value| Builder { bin: bin.clone(), views: views.clone(), accessors: accessors.clone() }.finish(d);
    let err = convert(&make(doc(&["KHR_draco_mesh_compression"], &["KHR_draco_mesh_compression"])), None).err().unwrap().to_string();
    assert!(err.contains("KHR_draco_mesh_compression"), "{err}");
    let err = convert(&make(doc(&["EXT_meshopt_compression"], &["EXT_meshopt_compression"])), None).err().unwrap().to_string();
    assert!(err.contains("EXT_meshopt_compression"), "{err}");
    let c = convert(&make(doc(&[], &["EXT_something_optional"])), None).unwrap();
    assert!(c.warnings.iter().any(|w| w.contains("EXT_something_optional")), "{:?}", c.warnings);
}

fn png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let img = image::RgbaImage::from_fn(w, h, |x, y| image::Rgba(f(x, y)));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

#[test]
fn material_textures_transform_and_tangents() {
    let mut b = Builder::default();
    // A quad facing +Y with UVs (two triangles, indexed).
    let pos = b.floats(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0], "VEC3");
    let nrm = b.floats(&[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], "VEC3");
    let uv0 = b.floats(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], "VEC2");
    let uv1 = b.floats(&[0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0], "VEC2");
    let idx = b.u16s(&[0, 2, 1, 0, 3, 2], "SCALAR", 6);
    let color = png(4, 2, |x, y| [x as u8 * 60, y as u8 * 100, 255, 255]);
    let normal = png(3, 3, |_, _| [128, 128, 255, 255]);
    let cv = b.view(&color);
    let nv = b.view(&normal);
    let gltf = b.finish(json!({
        "extensionsUsed": ["KHR_texture_transform", "KHR_materials_emissive_strength"],
        "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "images": [{"bufferView": cv, "mimeType": "image/png"}, {"bufferView": nv, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9728, "minFilter": 9729, "wrapS": 33071, "wrapT": 33648}],
        "textures": [{"source": 0, "sampler": 0}, {"source": 1}],
        "materials": [{
            "name": "paint",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 0.5, 0.25, 1.0],
                "baseColorTexture": {"index": 0, "extensions": {"KHR_texture_transform": {"offset": [0.5, 0.0], "rotation": 0.25, "scale": [2.0, 3.0], "texCoord": 1}}},
                "metallicFactor": 0.2, "roughnessFactor": 0.7
            },
            "normalTexture": {"index": 1, "scale": 0.5},
            "emissiveFactor": [1.0, 0.5, 0.0],
            "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 4.0}},
            "alphaMode": "MASK", "alphaCutoff": 0.3, "doubleSided": true
        }],
        "meshes": [{"primitives": [{"attributes": {"POSITION": pos, "NORMAL": nrm, "TEXCOORD_0": uv0, "TEXCOORD_1": uv1}, "indices": idx, "material": 0}]}]
    }));
    let c = convert(&gltf, None).unwrap();
    let m = &c.model;
    let mat = &m.materials[0];
    assert_eq!(mat.base_color, [1.0, 0.5, 0.25, 1.0]);
    assert_eq!(mat.emissive, [4.0, 2.0, 0.0]);
    assert_eq!(mat.alpha, AlphaMode::Mask { cutoff: 0.3 });
    assert!(mat.double_sided && mat.normal_scale == 0.5 && mat.metallic == 0.2);
    let base = mat.base_color_texture.unwrap();
    assert_eq!(base.uv, 1, "texCoord override");
    assert_eq!(base.transform, Some(UvTransform { offset: [0.5, 0.0], rotation: 0.25, scale: [2.0, 3.0] }));
    assert_eq!(base.sampler, Sampler { mag: Filter::Nearest, min: Filter::Linear, mip: None, wrap_s: Wrap::ClampToEdge, wrap_t: Wrap::MirroredRepeat });
    let base_img = ktx::read(&m.textures[base.texture as usize]).unwrap();
    // Level 0 only: the renderer makes the mips on the GPU.
    assert_eq!((base_img.format, base_img.width, base_img.height, base_img.levels.len()), (ktx::Format::RGBA8_SRGB, 4, 2, 1));
    assert_eq!(&base_img.levels[0][4..8], &[60, 0, 255, 255]);
    let n = mat.normal_texture.unwrap();
    let n_img = ktx::read(&m.textures[n.texture as usize]).unwrap();
    assert_eq!((n_img.format, n_img.levels.len()), (ktx::Format::RGBA8_UNORM, 1));
    // Tangents generated along +U (= +X here), handedness from the UVs.
    for v in &m.vertices {
        assert!(Vec3::from_slice(&v.tangent[..3]).abs_diff_eq(Vec3::X, 1e-4), "{:?}", v.tangent);
        assert!(v.tangent[3].abs() == 1.0);
    }
    assert_eq!(m.vertices.len(), 4, "welded back");
    assert_eq!(m.indices.len(), 6);
}

#[test]
fn sparse_strip_and_clip() {
    let mut b = Builder::default();
    // Four positions, the third overridden by a sparse accessor.
    let base = b.floats(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 9.0, 9.0, 9.0, 1.0, 0.0, 1.0], "VEC3");
    let idx = b.view(&[2u16.to_le_bytes(), [0, 0]].concat());
    let vals: Vec<u8> = [0.0f32, 0.0, 1.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    let val = b.view(&vals);
    b.accessors[base]["sparse"] = json!({"count": 1, "indices": {"bufferView": idx, "componentType": 5123}, "values": {"bufferView": val}});
    b.accessors[base]["min"] = json!([0.0, 0.0, 0.0]);
    b.accessors[base]["max"] = json!([1.0, 0.0, 1.0]);
    let times = b.floats(&[0.0, 1.0], "SCALAR");
    let q = glam::Quat::from_rotation_y(1.0);
    let rots = b.floats(&[0.0, 0.0, 0.0, 1.0, q.x, q.y, q.z, q.w], "VEC4");
    let tr = b.floats(&[0.0, 0.0, 0.0, 0.0, 5.0, 0.0], "VEC3");
    let weights = b.floats(&[0.0, 1.0], "SCALAR");
    let gltf = b.finish(json!({
        "scenes": [{"nodes": [0]}], "nodes": [{"name": "spin", "mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": base}, "mode": 5, "targets": [{"POSITION": base}]}], "weights": [0.0]}],
        "animations": [{"name": "spin", "samplers": [
            {"input": times, "output": rots},
            {"input": times, "output": tr, "interpolation": "STEP"},
            {"input": times, "output": weights}
        ], "channels": [
            {"sampler": 0, "target": {"node": 0, "path": "rotation"}},
            {"sampler": 1, "target": {"node": 0, "path": "translation"}},
            {"sampler": 2, "target": {"node": 0, "path": "weights"}}
        ]}]
    }));
    let c = convert(&gltf, None).unwrap();
    let w = |s: &str| c.warnings.iter().any(|x| x.contains(s));
    assert!(w("morph targets dropped") && w("morph target weights dropped"), "{:?}", c.warnings);
    // A strip of four: two triangles; the sparse value replaced (9, 9, 9).
    let m = Model::from_bytes(&c.model.to_bytes()).unwrap();
    assert_eq!(m.indices.len(), 6);
    assert!(m.vertices.iter().all(|v| v.position != [9.0, 9.0, 9.0]));
    assert!(m.vertices.iter().any(|v| v.position == [0.0, 0.0, 1.0]));
    // Both triangles wound the same way (normals agree).
    let ns: Vec<Vec3> = m.vertices.iter().map(|v| Vec3::from(v.normal)).collect();
    assert!(ns.windows(2).all(|p| p[0].abs_diff_eq(p[1], 1e-6)), "{ns:?}");
    assert_eq!(m.clips.len(), 1);
    let clip = &m.clips[0];
    assert_eq!((clip.name.as_str(), clip.duration, clip.channels.len()), ("spin", 1.0, 2));
    let mut p = pose::Pose::rest(&m);
    p.sample(&m, 0, 0.5);
    assert!(p.rotation[1].abs_diff_eq(glam::Quat::from_rotation_y(0.5), 1e-5));
    assert_eq!(p.translation[1], Vec3::ZERO, "STEP holds the first key");
}

#[test]
fn basis_ktx2_transcoded_to_bc7() {
    use basisu_c_sys::extra::{BasisuEncoder, BasisuEncoderParams, SourceImage, SourceImageFormat, basisu_encoder_init};
    use basisu_c_sys::BasisTextureFormat;
    basisu_encoder_init();
    let (w, h) = (16u32, 8u32);
    let rgba: Vec<u8> = (0..w * h).flat_map(|i| [(i * 3) as u8, (i * 5) as u8, 200, 255]).collect();
    let mut enc = BasisuEncoder::new();
    enc.set_image(SourceImage { data: &rgba, format: SourceImageFormat::Rgba8, size: basisu_c_sys::extra::types::Extent3d { width: w, height: h, depth_or_array_layers: 1 } }).unwrap();
    let params = BasisuEncoderParams::new_with_srgb_defaults(BasisTextureFormat::UastcLdr4x4).with_flags(basisu_c_sys::common::BU_COMP_FLAGS_GEN_MIPS_CLAMP);
    let ktx2 = enc.compress(params).unwrap();
    let mut b = Builder::default();
    let pos = b.floats(&TRI, "VEC3");
    let uv = b.floats(&[0.0, 0.0, 1.0, 0.0, 0.5, 1.0], "VEC2");
    let kv = b.view(&ktx2);
    let gltf = b.finish(json!({
        "extensionsUsed": ["KHR_texture_basisu"], "extensionsRequired": ["KHR_texture_basisu"],
        "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "images": [{"bufferView": kv, "mimeType": "image/ktx2"}],
        "textures": [{"extensions": {"KHR_texture_basisu": {"source": 0}}}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": pos, "TEXCOORD_0": uv}, "material": 0}]}]
    }));
    let c = convert(&gltf, None).unwrap();
    let img = ktx::read(&c.model.textures[0]).unwrap();
    assert_eq!((img.format, img.width, img.height), (ktx::Format::BC7_SRGB, w, h));
    assert_eq!(img.levels.len(), 5, "the encoder's mips kept");
    assert_eq!(img.levels[0].len(), 4 * 2 * 16);
}
