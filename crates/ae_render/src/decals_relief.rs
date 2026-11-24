//! Acceleration data for the bilinear micro-heightfields, following
//! Bán, Valasek, Bálint and Vad, Robust Cone Step Mapping (2024).
//! https://doi.org/10.2312/sr.20241146
//! Implementation based on the CC BY paper's algorithms, with clamped
//! edges and conservative cell verification of the proposed jumps.

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context as _, Result};
use ash::vk;
use sha2::{Digest, Sha256};

use crate::decals::{DecalAtlas, GpuSurfaceLayer};
use crate::device::GpuDevice;
use crate::memory::OwnedImage;
use crate::{pipeline, upload};

const FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
const TEXEL_BYTES: usize = 16;
const DOMAIN: &[u8] =
    b"AE RCSM v1: UNORM8 heights, clamp-to-edge, bilinear, corrected R32F slope, Lipschitz, cell max/min";

fn cache_dir() -> Option<PathBuf> {
    let root = std::env::var_os("XDG_CACHE_HOME")
        .filter(|v| !v.is_empty())
        .or_else(|| std::env::var_os("HOME").map(|v| PathBuf::from(v).join(".cache").into_os_string()))?;
    Some(PathBuf::from(root).join("assembleengine/decal-cones"))
}

fn key(size: usize, height: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(DOMAIN);
    hash.update((size as u64).to_le_bytes());
    hash.update(include_bytes!("../shaders/decal_cones.comp"));
    hash.update(height);
    format!("{:x}", hash.finalize())
}

fn read_cache(path: &Path, size: usize) -> Option<Vec<u8>> {
    let mut bytes = fs::read(path).ok()?;
    if bytes.len() != size * size * TEXEL_BYTES + 32 {
        return None;
    }
    let end = bytes.len() - 32;
    if Sha256::digest(&bytes[..end])[..] != bytes[end..] {
        return None;
    }
    bytes.truncate(end);
    Some(bytes)
}

fn write_cache(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let temp = path.with_extension(format!(
        "{}.{}.tmp",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let mut out = fs::OpenOptions::new().create_new(true).write(true).open(&temp)?;
    let result = (|| {
        out.write_all(bytes)?;
        out.write_all(&Sha256::digest(bytes))?;
        out.sync_all()?;
        fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

/// Array sectors are [relaxed aperture, certified aperture, cell max, cell min].
pub(crate) fn upload_cones(gpu: &GpuDevice, atlas: &DecalAtlas, layers: &[GpuSurfaceLayer]) -> Result<OwnedImage> {
    let n = atlas.size as usize;
    let cache = cache_dir().filter(|path| fs::create_dir_all(path).is_ok());
    let mut unique: HashMap<String, Vec<u8>> = HashMap::new();
    let mut layer_keys = Vec::with_capacity(layers.len());
    let mut generator = None;
    let started = Instant::now();
    let mut generated = 0;
    for (i, layer) in layers.iter().enumerate() {
        let height = if atlas.height_layers.contains(&(i as u32)) {
            layer.material.iter().map(|t| t[0]).collect::<Vec<_>>()
        } else {
            vec![128; n * n]
        };
        let id = key(n, &height);
        if let std::collections::hash_map::Entry::Vacant(entry) = unique.entry(id.clone()) {
            let file = cache.as_ref().map(|dir| dir.join(&id));
            let bytes = if let Some(hit) = file.as_ref().and_then(|p| read_cache(p, n)) {
                hit
            } else {
                let raw = if height.iter().all(|&h| h == height[0]) {
                    vec![1.0; n * n]
                } else {
                    if generator.is_none() {
                        generator = Some(ConeGenerator::new(gpu)?);
                    }
                    generated += 1;
                    generator.as_ref().unwrap().run(n, &height)?
                };
                let bytes = correct_cones(n, &height, &raw);
                if let Some(path) = &file
                    && let Err(err) = write_cache(path, &bytes)
                {
                    eprintln!("decal cone cache {}: {err}", path.display());
                }
                bytes
            };
            entry.insert(bytes);
        }
        layer_keys.push(id);
    }
    if generated > 0 && std::env::var_os("AE_DECAL_CONE_REPORT").is_some() {
        eprintln!(
            "decal cones: {generated} GPU generations, {:.1} ms with readback, correction and cache, {} payload bytes",
            started.elapsed().as_secs_f64() * 1e3,
            unique.values().map(Vec::len).sum::<usize>()
        );
    }
    let mips: Vec<_> = layer_keys.iter().map(|id| vec![unique[id].as_slice()]).collect();
    upload::image_layers(
        gpu,
        FORMAT,
        atlas.size,
        atlas.size,
        &mips,
        vk::ImageViewType::TYPE_2D_ARRAY,
        "decal cones",
    )
}

fn max_tree(n: usize, height: &[u8]) -> Vec<u32> {
    let leaf_offset = (n * n - 1) / 3;
    let mut tree = vec![0; leaf_offset];
    tree.extend(height.iter().map(|&h| u32::from(h)));
    let mut count = n / 2;
    while count != 0 {
        let parent = (count * count - 1) / 3;
        let child = (4 * count * count - 1) / 3;
        for y in 0..count {
            for x in 0..count {
                let at = child + y * 4 * count + x * 2;
                tree[parent + y * count + x] = tree[at]
                    .max(tree[at + 1])
                    .max(tree[at + 2 * count])
                    .max(tree[at + 2 * count + 1]);
            }
        }
        count /= 2;
    }
    tree
}

fn correct_cones(n: usize, height: &[u8], relaxed: &[f32]) -> Vec<u8> {
    let index = |x, y| y * n + x;
    let mut max_du = 0;
    let mut max_dv = 0;
    for y in 0..n {
        for x in 0..n {
            let h = height[index(x, y)];
            max_du = max_du.max(h.abs_diff(height[index((x + 1).min(n - 1), y)]));
            max_dv = max_dv.max(h.abs_diff(height[index(x, (y + 1).min(n - 1))]));
        }
    }
    // Bilinear partial derivatives stay inside these edge slope bounds.
    // This global Lipschitz cone certifies a prefix of each relaxed jump.
    let max_slope = f64::from(max_du).hypot(f64::from(max_dv)) * n as f64 / 255.0;
    let certified = ((1.0 / max_slope).min(1.0) * 0.999_9) as f32;
    let mut bytes = Vec::with_capacity(n * n * TEXEL_BYTES);
    for y in 0..n {
        for x in 0..n {
            let mut cone: f32 = 1.0;
            // Equation 2: min filter over each texel's 3x3 neighborhood.
            for ny in y.saturating_sub(1)..=(y + 1).min(n - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(n - 1) {
                    cone = cone.min(relaxed[index(nx, ny)]);
                }
            }
            let (right, down) = ((x + 1).min(n - 1), (y + 1).min(n - 1));
            let h = [
                height[index(x, y)],
                height[index(right, y)],
                height[index(x, down)],
                height[index(right, down)],
            ];
            let lo = f32::from(*h.iter().min().unwrap()) / 255.0;
            let hi = f32::from(*h.iter().max().unwrap()) / 255.0;
            for value in [cone, certified, hi, lo] {
                bytes.extend(value.to_le_bytes());
            }
        }
    }
    bytes
}

struct ConeGenerator<'a> {
    gpu: &'a GpuDevice,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl<'a> ConeGenerator<'a> {
    fn new(gpu: &'a GpuDevice) -> Result<Self> {
        Self::program(gpu, pipeline::spirv!("decal_cones.comp"), 0, 4)
    }

    fn program(gpu: &'a GpuDevice, shader: &[u8], sampled: u32, push_size: u32) -> Result<Self> {
        let d = &gpu.device;
        let mut out = Self {
            gpu,
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
        };
        let stage = vk::ShaderStageFlags::COMPUTE;
        let storage = vk::DescriptorType::STORAGE_BUFFER;
        let mut bindings = vec![(storage, stage); 2];
        bindings.resize(
            2 + sampled as usize,
            (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, stage),
        );
        out.set_layout = pipeline::set_layout(d, &bindings)?;
        out.layout = pipeline::pipeline_layout(d, &[out.set_layout], push_size, stage)?;
        out.pipeline = pipeline::compute_pipeline(d, shader, out.layout)?;
        let mut sizes = vec![vk::DescriptorPoolSize {
            ty: storage,
            descriptor_count: 2,
        }];
        if sampled > 0 {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: sampled,
            });
        }
        unsafe {
            out.pool = d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes),
                None,
            )?;
            out.set = d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(out.pool)
                    .set_layouts(&[out.set_layout]),
            )?[0];
        }
        Ok(out)
    }

    fn bind_buffers(&self, buffers: [vk::Buffer; 2]) {
        let infos = buffers.map(|buffer| [vk::DescriptorBufferInfo::default().buffer(buffer).range(vk::WHOLE_SIZE)]);
        let writes = std::array::from_fn::<_, 2, _>(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos[i])
        });
        unsafe { self.gpu.device.update_descriptor_sets(&writes, &[]) };
    }

    fn dispatch(&self, [x, y]: [u32; 2], push: &[u8]) -> Result<()> {
        let gpu = self.gpu;
        upload::one_shot(gpu, |cmd| unsafe {
            let d = &gpu.device;
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[self.set], &[]);
            d.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            d.cmd_dispatch(cmd, x, y, 1);
        })
    }

    fn run(&self, n: usize, height: &[u8]) -> Result<Vec<f32>> {
        let gpu = self.gpu;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let tree = max_tree(n, height);
        let raw: Vec<_> = tree.into_iter().flat_map(u32::to_le_bytes).collect();
        let mut heights = upload::buffer(gpu, &raw, usage, "decal height max tree")?;
        let result = (|| {
            let size = (n * n * 4) as u64;
            let mut output = upload::buffer_sized(
                gpu,
                size,
                &[],
                usage | vk::BufferUsageFlags::TRANSFER_SRC,
                "relaxed cone output",
            )?;
            self.bind_buffers([heights.buffer, output.buffer]);
            let groups = (n as u32).div_ceil(8);
            let copied = self
                .dispatch([groups, groups], &(n as i32).to_le_bytes())
                .and_then(|()| upload::read_buffer(gpu, output.buffer, 0, size));
            output.destroy(gpu);
            let copied = copied.context("generating decal relaxed cones")?;
            Ok(copied
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&word| f32::from_le_bytes(word))
                .collect())
        })();
        heights.destroy(gpu);
        result
    }
}

impl Drop for ConeGenerator<'_> {
    fn drop(&mut self) {
        let d = &self.gpu.device;
        unsafe {
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuInstance;
    use std::sync::Arc;

    fn height_at(n: usize, height: &[u8], uv: [f64; 2]) -> f64 {
        let [x, y] = uv.map(|u| u * n as f64 - 0.5);
        let (i, j) = (x.floor() as i32, y.floor() as i32);
        let get = |x: i32, y: i32| {
            let clamp = |c: i32| c.clamp(0, n as i32 - 1) as usize;
            f64::from(height[clamp(y) * n + clamp(x)]) / 255.0
        };
        let (u, v) = (x - x.floor(), y - y.floor());
        let lerp = |a, b, t| a * (1.0 - t) + b * t;
        lerp(
            lerp(get(i, j), get(i + 1, j), u),
            lerp(get(i, j + 1), get(i + 1, j + 1), u),
            v,
        )
    }

    // Reference: split the complete ray at every texel centre line, then
    // interpolate its signed residual at three points to obtain the cell
    // quadratic. Uses f64 and no acceleration or sampling interval.
    fn first_hit(n: usize, height: &[u8], ray: [f32; 8]) -> Option<f64> {
        let [u, v, du, dv, lo, hi, _, _] = ray.map(f64::from);
        let mut times = vec![0.0, 1.0];
        for (p, dir) in [(u, du), (v, dv)] {
            if dir == 0.0 {
                continue;
            }
            for i in 0..n {
                let t = ((i as f64 + 0.5) / n as f64 - p) / dir;
                if (0.0..1.0).contains(&t) {
                    times.push(t);
                }
            }
        }
        times.sort_by(f64::total_cmp);
        times.dedup();
        let residual = |t| hi + t * (lo - hi) - height_at(n, height, [u + t * du, v + t * dv]);
        for w in times.windows(2) {
            let (start, end) = (w[0], w[1]);
            let c = residual(start);
            if c < 1e-12 {
                return Some(start);
            }
            let middle = residual((start + end) * 0.5);
            let last = residual(end);
            let b = 4.0 * middle - last - 3.0 * c;
            let a = last - c - b;
            let mut roots = if a.abs() < 1e-12 {
                [-c / b, f64::INFINITY]
            } else {
                let disc = (b * b - 4.0 * a * c).sqrt();
                [(-b - disc) / (2.0 * a), (-b + disc) / (2.0 * a)]
            };
            roots.sort_by(f64::total_cmp);
            if let Some(local) = roots.into_iter().find(|r| (0.0..=1.0 + 1e-10).contains(r)) {
                return Some(start + local.min(1.0) * (end - start));
            }
        }
        None
    }

    fn brute_cones(n: usize, height: &[u8]) -> Vec<f32> {
        let get = |x: i32, y: i32| {
            let clamp = |c: i32| c.clamp(0, n as i32 - 1) as usize;
            height[clamp(y) * n + clamp(x)]
        };
        (0..n * n)
            .map(|base| {
                let (x, y) = ((base % n) as i32, (base / n) as i32);
                let mut cone: f64 = 1.0;
                for (q, &h) in height.iter().enumerate() {
                    if h <= height[base] {
                        continue;
                    }
                    let (u, v) = ((q % n) as i32, (q / n) as i32);
                    for dx in [-1, 1] {
                        for dy in [-1, 1] {
                            if (u - x) * dx < 0 || (v - y) * dy < 0 {
                                continue;
                            }
                            let (hx, hy, diagonal) = (get(u + dx, v), get(u, v + dy), get(u + dx, v + dy));
                            if h > hx || h > hy || hx > diagonal || hy > diagonal {
                                let r = f64::from(u - x).hypot(f64::from(v - y)) / n as f64;
                                cone = cone.min(r * 255.0 / f64::from(h - height[base]));
                            }
                        }
                    }
                }
                cone as f32
            })
            .collect()
    }

    #[test]
    #[ignore = "needs a Vulkan GPU"]
    fn gpu_relaxed_cones_and_first_hits_match_the_bilinear_reference() {
        let instance = Arc::new(GpuInstance::new(&[], Some(true)).unwrap());
        assert!(instance.validation);
        let gpu = GpuDevice::new(instance, None).unwrap();
        let n = 16;
        let height = [
            vec![128; n * n],
            (0..n * n).map(|i| 32 + 12 * (i % n) as u8).collect(),
            (0..n * n).map(|i| 32 + 12 * (i / n) as u8).collect(),
            (0..n * n)
                .map(|i| (128.0 + 95.0 * (i as f64 * 0.44).sin()) as u8)
                .collect(),
            (0..n * n)
                .map(|i| {
                    if i % n == 7 {
                        0
                    } else if i / n == 8 {
                        255
                    } else {
                        191
                    }
                })
                .collect(),
            (0..n * n)
                .map(|i| {
                    if i / n < 8 && i % n > 8 {
                        229
                    } else if (i / n + i % n) % 2 == 0 {
                        24
                    } else {
                        60
                    }
                })
                .collect(),
            (0..n * n)
                .map(|i| if (i % n == 8) ^ (i / n == 8) { 255 } else { 0 })
                .collect(),
        ];
        let generator = ConeGenerator::new(&gpu).unwrap();
        let packed: Vec<_> = height
            .iter()
            .map(|layer| {
                let actual = generator.run(n, layer).unwrap();
                for (a, b) in actual.iter().zip(brute_cones(n, layer)) {
                    assert!((a - b).abs() < 2e-6, "{a} vs {b}");
                }
                correct_cones(n, layer, &actual)
            })
            .collect();
        let img = |format, bytes: &[Vec<u8>]| {
            upload::image_layers(
                &gpu,
                format,
                n as u32,
                n as u32,
                &bytes.iter().map(|b| vec![b.as_slice()]).collect::<Vec<_>>(),
                vk::ImageViewType::TYPE_2D_ARRAY,
                "relief probe",
            )
            .unwrap()
        };
        let mut h_img = img(vk::Format::R8_UNORM, &height);
        let mut c_img = img(FORMAT, &packed);
        let desc = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler = unsafe { gpu.device.create_sampler(&desc, None).unwrap() };
        let probe = ConeGenerator::program(&gpu, pipeline::spirv!("decal_relief_probe.comp"), 2, 12).unwrap();
        let images = [h_img.view, c_img.view].map(|view| {
            [vk::DescriptorImageInfo::default()
                .image_view(view)
                .sampler(sampler)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
        });
        let writes = std::array::from_fn::<_, 2, _>(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(probe.set)
                .dst_binding(i as u32 + 2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&images[i])
        });
        unsafe { gpu.device.update_descriptor_sets(&writes, &[]) };
        let mut seed = 333u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let mut rays = Vec::new();
        for layer in 0..height.len() {
            for i in 0..257 {
                let [x, y, u, v] = if i == 0 {
                    [0.0, 0.0, 1.0, 1.0]
                } else {
                    [next(), next(), next(), next()]
                };
                rays.push([x, y, u - x, v - y, 0.0, 1.0, 0.2 + 2.0 * next(), layer as f32]);
            }
        }
        rays.push([0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        // The ray starts and ends above the two low diagonal corners of
        // one cell, but passes through the bilinear hump in between.
        rays.push([
            7.5 / 16.0,
            7.5 / 16.0,
            1.0 / 16.0,
            1.0 / 16.0,
            0.3,
            0.4,
            1.0,
            (height.len() - 1) as f32,
        ]);
        // The final ray leaves the domain. The tracer must report fallback.
        rays.push([0.9, 0.9, 0.2, 0.2, 0.0, 1.0, 1.0, 0.0]);
        let raw: Vec<_> = rays
            .iter()
            .flat_map(|ray| ray.iter().flat_map(|v| v.to_le_bytes()))
            .collect();
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let mut input = upload::buffer(&gpu, &raw, usage, "probe rays").unwrap();
        let size = (rays.len() * 16) as u64;
        let mut output = upload::buffer_sized(
            &gpu,
            size,
            &[],
            usage | vk::BufferUsageFlags::TRANSFER_SRC,
            "probe hits",
        )
        .unwrap();
        probe.bind_buffers([input.buffer, output.buffer]);
        let run = |budget| {
            let count = rays.len() as u32;
            probe
                .dispatch([count.div_ceil(64), 1], pipeline::push_bytes(&[count, budget, 6]))
                .unwrap();
            upload::read_buffer(&gpu, output.buffer, 0, size).unwrap()
        };
        let results = run(128);
        let zero_budget = run(0);
        let one_step = run(1);
        let mut max_error = 0.0f64;
        let decode = |bytes: &[u8]| {
            let words: [[u8; 4]; 4] = bytes.as_chunks::<4>().0.try_into().unwrap();
            words.map(f32::from_le_bytes)
        };
        let exhausted = one_step
            .as_chunks::<16>()
            .0
            .iter()
            .filter(|bytes| decode(*bytes)[3] == -2.0)
            .count();
        assert!(exhausted > 0, "one proposal should leave some rays unfinished");
        for (i, ray) in rays.iter().enumerate() {
            let (_, data) = results.split_at(i * 16);
            let [u, v, h, steps] = decode(&data[..16]);
            assert!(decode(&zero_budget[i * 16..][..16])[3] < 0.0);
            if i == rays.len() - 1 {
                assert!(steps < 0.0);
                continue;
            }
            let expected = first_hit(n, &height[ray[7] as usize], *ray).unwrap();
            assert!(steps > 0.0, "valid ray {i} exhausted: {ray:?} vs {expected}");
            let [x, y, dx, dy, lo, hi, range, _] = ray.map(f64::from);
            let actual = (hi - (f64::from(h) / range + 128.0 / 255.0)) / (hi - lo);
            let error = (actual - expected).abs();
            max_error = max_error.max(error);
            assert!(error < 1e-5, "ray {i}, error {error:.7}, {actual} vs {expected}");
            assert!((f64::from(u) - (x + dx * expected)).abs() < 1e-5);
            assert!((f64::from(v) - (y + dy * expected)).abs() < 1e-5);
        }
        println!(
            "{} valid rays, zero misses, max normalized error {max_error:.7}, {exhausted} one-step fallbacks",
            rays.len() - 1
        );
        output.destroy(&gpu);
        input.destroy(&gpu);
        unsafe { gpu.device.destroy_sampler(sampler, None) };
        h_img.destroy(&gpu);
        c_img.destroy(&gpu);
        assert_eq!(
            (gpu.instance.counters.errors(), gpu.instance.counters.warnings()),
            (0, 0)
        );
    }
}
