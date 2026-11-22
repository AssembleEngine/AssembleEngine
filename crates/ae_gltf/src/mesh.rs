//! One glTF primitive to the model's vertices and triangle list.

use ae_model::{MAX_INFLUENCES, Vertex};
use anyhow::{Context as _, Result, ensure};
use std::collections::HashMap;

/// Influences read per vertex: JOINTS_0/1 and WEIGHTS_0/1.
const READ_INFLUENCES: usize = 8;

/// A primitive's attributes as read, its triangles as a list.
pub struct Source {
    positions: Vec<[f32; 3]>,
    normals: Option<Vec<[f32; 3]>>,
    tangents: Option<Vec<[f32; 4]>>,
    uv: [Option<Vec<[f32; 2]>>; 2],
    colors: Option<Vec<[u16; 4]>>,
    /// (skin joint index, weight).
    influences: Option<Vec<[(u16, f32); READ_INFLUENCES]>>,
    indices: Vec<u32>,
    pub has_joints: bool,
}

pub enum Joints {
    /// A skinned mesh: skin joint index -> model joint.
    Skin(Vec<u16>),
    /// A mesh on a node: every vertex on this model joint.
    Piece(u16),
}

fn warn(warnings: &mut Vec<String>, w: String) {
    if !warnings.contains(&w) {
        warnings.push(w);
    }
}

/// `None` when the primitive has no triangles (points, lines: warned).
pub fn read_primitive<'a, 's, F>(
    prim: &'a gltf::Primitive<'a>,
    buffer: F,
    what: &str,
    warnings: &mut Vec<String>,
) -> Result<Option<Source>>
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    use gltf::Semantic as S;
    use gltf::mesh::Mode;
    let mode = prim.mode();
    if !matches!(mode, Mode::Triangles | Mode::TriangleStrip | Mode::TriangleFan) {
        warn(warnings, format!("{what}: {mode:?} dropped (triangles only)"));
        return Ok(None);
    }
    for (semantic, _) in prim.attributes() {
        match semantic {
            S::TexCoords(n) if n >= 2 => warn(warnings, format!("{what}: TEXCOORD_{n} dropped")),
            S::Colors(n) if n >= 1 => warn(warnings, format!("{what}: COLOR_{n} dropped")),
            S::Joints(n) | S::Weights(n) if n >= 2 => warn(
                warnings,
                format!("{what}: JOINTS_{n}/WEIGHTS_{n} dropped (more than 8 influences)"),
            ),
            _ => {}
        }
    }
    if prim.morph_targets().len() > 0 {
        warn(warnings, format!("{what}: morph targets dropped"));
    }
    let r = prim.reader(buffer);
    let positions: Vec<[f32; 3]> = r
        .read_positions()
        .with_context(|| format!("{what}: no POSITION"))?
        .collect();
    let n = positions.len();
    let check = |len: usize, name: &str| -> Result<()> {
        ensure!(len == n, "{what}: {name} has {len} values for {n} positions");
        Ok(())
    };
    let normals: Option<Vec<[f32; 3]>> = r.read_normals().map(|it| it.collect());
    let tangents: Option<Vec<[f32; 4]>> = r.read_tangents().map(|it| it.collect());
    let uv0: Option<Vec<[f32; 2]>> = r.read_tex_coords(0).map(|it| it.into_f32().collect());
    let uv1: Option<Vec<[f32; 2]>> = r.read_tex_coords(1).map(|it| it.into_f32().collect());
    let colors: Option<Vec<[u16; 4]>> = r.read_colors(0).map(|it| it.into_rgba_u16().collect());
    for (len, name) in [
        (normals.as_ref().map(Vec::len), "NORMAL"),
        (tangents.as_ref().map(Vec::len), "TANGENT"),
        (uv0.as_ref().map(Vec::len), "TEXCOORD_0"),
        (uv1.as_ref().map(Vec::len), "TEXCOORD_1"),
        (colors.as_ref().map(Vec::len), "COLOR_0"),
    ] {
        if let Some(len) = len {
            check(len, name)?;
        }
    }
    let mut influences = None;
    let has_joints = r.read_joints(0).is_some() && r.read_weights(0).is_some();
    if has_joints {
        let mut inf = vec![[(0u16, 0.0f32); READ_INFLUENCES]; n];
        for set in 0..2u32 {
            let (Some(j), Some(w)) = (r.read_joints(set), r.read_weights(set)) else {
                continue;
            };
            let j: Vec<[u16; 4]> = j.into_u16().collect();
            let w: Vec<[f32; 4]> = w.into_f32().collect();
            check(j.len(), "JOINTS")?;
            check(w.len(), "WEIGHTS")?;
            for v in 0..n {
                for k in 0..4 {
                    inf[v][set as usize * 4 + k] = (j[v][k], w[v][k]);
                }
            }
        }
        influences = Some(inf);
    }
    let list: Vec<u32> = match r.read_indices() {
        Some(it) => it.into_u32().collect(),
        None => (0..n as u32).collect(),
    };
    ensure!(list.iter().all(|&i| (i as usize) < n), "{what}: index out of range");
    let mut indices = Vec::new();
    match mode {
        Mode::Triangles => {
            ensure!(
                list.len().is_multiple_of(3),
                "{what}: {} indices, not whole triangles",
                list.len()
            );
            indices = list;
        }
        Mode::TriangleStrip => {
            for i in 0..list.len().saturating_sub(2) {
                let (a, b) = if i % 2 == 0 {
                    (list[i], list[i + 1])
                } else {
                    (list[i + 1], list[i])
                };
                indices.extend([a, b, list[i + 2]]);
            }
        }
        _ => {
            for i in 1..list.len().saturating_sub(1) {
                indices.extend([list[i], list[i + 1], list[0]]);
            }
        }
    }
    Ok(Some(Source {
        positions,
        normals,
        tangents,
        uv: [uv0, uv1],
        colors,
        influences,
        indices,
        has_joints,
    }))
}

/// Joints and weights of one vertex: the four strongest, renormalised to
/// unorm16 summing to 65535. `None` if every weight is 0.
/// Joints, weights, and whether influences were dropped.
type Reduced = ([u16; 4], [u16; 4], bool);

fn reduce(inf: &[(u16, f32); READ_INFLUENCES], map: &[u16]) -> Result<Option<Reduced>> {
    let mut merged: Vec<(u16, f32)> = Vec::new();
    for &(j, w) in inf {
        if w <= 0.0 {
            continue;
        }
        let mj = *map
            .get(j as usize)
            .with_context(|| format!("joint {j} outside its skin ({} joints)", map.len()))?;
        match merged.iter_mut().find(|(m, _)| *m == mj) {
            Some(e) => e.1 += w,
            None => merged.push((mj, w)),
        }
    }
    if merged.is_empty() {
        return Ok(None);
    }
    merged.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let dropped = merged.len() > MAX_INFLUENCES;
    merged.truncate(MAX_INFLUENCES);
    let total: f32 = merged.iter().map(|e| e.1).sum();
    let mut joints = [merged[0].0; 4];
    let mut weights = [0u16; 4];
    for (k, &(j, w)) in merged.iter().enumerate() {
        joints[k] = j;
        weights[k] = (w / total * 65535.0).round() as u16;
    }
    // Exact sum: the rounding error onto the largest.
    let sum: i32 = weights.iter().map(|&w| w as i32).sum();
    weights[0] = (weights[0] as i32 + 65535 - sum) as u16;
    Ok(Some((joints, weights, dropped)))
}

/// The vertices and triangle list; `tangent_uv` the normal map's UV set
/// when the material has one (tangents generated if absent).
pub fn build(
    s: Source,
    joints: &Joints,
    tangent_uv: Option<u32>,
    what: &str,
    warnings: &mut Vec<String>,
) -> Result<(Vec<Vertex>, Vec<u32>)> {
    let n = s.positions.len();
    let mut vertices = vec![Vertex::default(); n];
    let mut dropped = false;
    let mut unweighted = false;
    for (i, v) in vertices.iter_mut().enumerate() {
        v.position = s.positions[i];
        v.normal = s.normals.as_ref().map_or([0.0, 1.0, 0.0], |a| a[i]);
        v.tangent = s.tangents.as_ref().map_or([1.0, 0.0, 0.0, 1.0], |a| a[i]);
        v.uv0 = s.uv[0].as_ref().map_or([0.0; 2], |a| a[i]);
        v.uv1 = s.uv[1].as_ref().map_or([0.0; 2], |a| a[i]);
        v.color = s.colors.as_ref().map_or([65535; 4], |a| a[i]);
        match joints {
            Joints::Piece(p) => {
                v.joints = [*p; 4];
                v.weights = [65535, 0, 0, 0];
            }
            Joints::Skin(map) => {
                let inf = &s.influences.as_ref().expect("skinned source has influences")[i];
                match reduce(inf, map).with_context(|| format!("{what}: vertex {i}"))? {
                    Some((j, w, d)) => {
                        v.joints = j;
                        v.weights = w;
                        dropped |= d;
                    }
                    None => {
                        unweighted = true;
                        v.joints = [map[0]; 4];
                        v.weights = [65535, 0, 0, 0];
                    }
                }
            }
        }
    }
    if dropped {
        warn(
            warnings,
            format!(
                "{what}: more than {MAX_INFLUENCES} influences on some vertices; the strongest {MAX_INFLUENCES} kept, renormalised"
            ),
        );
    }
    if unweighted {
        warn(
            warnings,
            format!("{what}: vertices with all weights 0; put on the skin's first joint"),
        );
    }
    let mut indices = s.indices;
    let flat = s.normals.is_none();
    let gen_tangents = match tangent_uv {
        Some(uv) if s.tangents.is_none() => {
            if s.uv[uv as usize].is_some() {
                true
            } else {
                warn(
                    warnings,
                    format!("{what}: normal map on TEXCOORD_{uv}, which the primitive lacks; no tangents"),
                );
                false
            }
        }
        _ => false,
    };
    if !(flat || gen_tangents) {
        return Ok((vertices, indices));
    }
    // Unweld: one vertex per triangle corner.
    let mut corners: Vec<Vertex> = indices.iter().map(|&i| vertices[i as usize]).collect();
    if flat {
        for tri in corners.as_chunks_mut::<3>().0 {
            let p = |k: usize| glam::Vec3::from(tri[k].position);
            let n = (p(1) - p(0)).cross(p(2) - p(0)).normalize_or(glam::Vec3::Y);
            for v in tri.iter_mut() {
                v.normal = n.to_array();
            }
        }
    }
    if gen_tangents {
        let mut g = Mikk {
            corners: &mut corners,
            uv: tangent_uv.unwrap() as usize,
        };
        if bevy_mikktspace::generate_tangents(&mut g).is_err() {
            warn(warnings, format!("{what}: MikkTSpace failed; default tangents"));
        }
    }
    // Weld identical corners back.
    let mut seen: HashMap<[u32; 20], u32> = HashMap::new();
    vertices = Vec::new();
    indices = Vec::with_capacity(corners.len());
    for c in corners {
        let key = vertex_key(&c);
        let i = *seen.entry(key).or_insert_with(|| {
            vertices.push(c);
            vertices.len() as u32 - 1
        });
        indices.push(i);
    }
    Ok((vertices, indices))
}

fn vertex_key(v: &Vertex) -> [u32; 20] {
    let mut k = [0u32; 20];
    let f = v
        .position
        .iter()
        .chain(&v.normal)
        .chain(&v.tangent)
        .chain(&v.uv0)
        .chain(&v.uv1);
    for (i, x) in f.enumerate() {
        k[i] = x.to_bits();
    }
    let h = |a: [u16; 4]| [a[0] as u32 | (a[1] as u32) << 16, a[2] as u32 | (a[3] as u32) << 16];
    k[14..16].copy_from_slice(&h(v.color));
    k[16..18].copy_from_slice(&h(v.joints));
    k[18..20].copy_from_slice(&h(v.weights));
    k
}

struct Mikk<'a> {
    corners: &'a mut [Vertex],
    uv: usize,
}

impl bevy_mikktspace::Geometry for Mikk<'_> {
    fn num_faces(&self) -> usize {
        self.corners.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.corners[face * 3 + vert].position
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.corners[face * 3 + vert].normal
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        let v = &self.corners[face * 3 + vert];
        if self.uv == 0 { v.uv0 } else { v.uv1 }
    }

    fn set_tangent(&mut self, t: Option<bevy_mikktspace::TangentSpace>, face: usize, vert: usize) {
        if let Some(t) = t {
            let mut e = t.tangent_encoded();
            // glTF's bitangent = cross(normal, tangent) * w; MikkTSpace's
            // encoding uses the same convention.
            e[3] = if e[3] < 0.0 { -1.0 } else { 1.0 };
            self.corners[face * 3 + vert].tangent = e;
        }
    }
}
