//! A model's textures: KTX2 files in a GPU format, read here and written
//! by the converter.
//!
//! Formats the runtime takes ([`Format`]): RGBA8 and the BC family, each
//! UNORM or sRGB; no supercompression or zstd. No BasisLZ / UASTC (the
//! converter transcodes those), cube maps, arrays or 3D. The level count
//! must be given (not 0).

use anyhow::{Context as _, Result, bail, ensure};

/// A `VkFormat` the runtime samples directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format(pub u32);

impl Format {
    pub const RGBA8_UNORM: Format = Format(37);
    pub const RGBA8_SRGB: Format = Format(43);
    pub const BC7_UNORM: Format = Format(145);
    pub const BC7_SRGB: Format = Format(146);

    /// (block width and height, bytes per block); `None` if not taken.
    pub fn block(self) -> Option<(u32, u32)> {
        match self.0 {
            37 | 43 => Some((1, 4)),
            // BC1 RGB/RGBA, BC4: 8 bytes per 4x4 block.
            131..=134 | 139 | 140 => Some((4, 8)),
            // BC2, BC3, BC5, BC6H, BC7: 16.
            135..=138 | 141..=146 => Some((4, 16)),
            _ => None,
        }
    }

    pub fn is_srgb(self) -> bool {
        matches!(self.0, 43 | 132 | 134 | 136 | 138 | 146)
    }

    /// The sRGB or UNORM twin; `None` where there is none (BC4, BC5, BC6H).
    pub fn with_srgb(self, srgb: bool) -> Option<Format> {
        let (unorm, s) = match self.0 {
            37 | 43 => (37, 43),
            131 | 132 => (131, 132),
            133 | 134 => (133, 134),
            135 | 136 => (135, 136),
            137 | 138 => (137, 138),
            145 | 146 => (145, 146),
            _ => return None,
        };
        Some(Format(if srgb { s } else { unorm }))
    }

    /// Bytes of one level `width` x `height`.
    pub fn level_bytes(self, width: u32, height: u32) -> Option<usize> {
        let (b, bytes) = self.block()?;
        Some(width.div_ceil(b) as usize * height.div_ceil(b) as usize * bytes as usize)
    }
}

/// A decoded texture: every level, largest first, uncompressed by zstd.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<Vec<u8>>,
}

impl Image {
    pub fn level_size(&self, level: usize) -> (u32, u32) {
        ((self.width >> level).max(1), (self.height >> level).max(1))
    }

    pub fn check(&self) -> Result<()> {
        ensure!(self.width > 0 && self.height > 0, "empty texture");
        ensure!(!self.levels.is_empty(), "texture without levels");
        ensure!(self.levels.len() as u32 <= 32 - self.width.max(self.height).leading_zeros(), "{} levels for {}x{}", self.levels.len(), self.width, self.height);
        for (i, l) in self.levels.iter().enumerate() {
            let (w, h) = self.level_size(i);
            let want = self.format.level_bytes(w, h).with_context(|| format!("VkFormat {} not taken", self.format.0))?;
            ensure!(l.len() == want, "level {i}: {} bytes, {w}x{h} needs {want}", l.len());
        }
        Ok(())
    }
}

const SUPERCOMPRESSION_NONE: u32 = 0;
const SUPERCOMPRESSION_ZSTD: u32 = 2;

/// Reads a KTX2 file in a taken format.
pub fn read(bytes: &[u8]) -> Result<Image> {
    let reader = ktx2::Reader::new(bytes).map_err(|e| anyhow::anyhow!("KTX2: {e:?}"))?;
    let h = reader.header();
    let Some(vk) = h.format else { bail!("KTX2 without a VkFormat (Basis: convert the model again)") };
    let format = Format(vk.value());
    ensure!(format.block().is_some(), "KTX2 VkFormat {} not taken", format.0);
    ensure!(h.pixel_depth == 0 && h.layer_count == 0 && h.face_count == 1, "KTX2 not a plain 2D texture");
    ensure!(h.level_count > 0, "KTX2 without its levels");
    let zstd = match h.supercompression_scheme {
        None => false,
        Some(ktx2::SupercompressionScheme::Zstandard) => true,
        Some(s) => bail!("KTX2 supercompression {s:?} not taken"),
    };
    let mut levels = Vec::new();
    for level in reader.levels() {
        if zstd {
            let mut out = Vec::with_capacity(level.uncompressed_byte_length as usize);
            let mut decoder = ruzstd::decoding::StreamingDecoder::new(level.data).map_err(|e| anyhow::anyhow!("zstd: {e}"))?;
            std::io::Read::read_to_end(&mut decoder, &mut out).context("zstd")?;
            levels.push(out);
        } else {
            levels.push(level.data.to_vec());
        }
    }
    let image = Image { format, width: h.pixel_width, height: h.pixel_height.max(1), levels };
    image.check()?;
    Ok(image)
}

/// The basic data format descriptor for `format`.
fn dfd(format: Format, zstd: bool) -> Vec<u8> {
    // Khronos Data Format Specification 1.3, the basic descriptor block.
    const MODEL_RGBSDA: u32 = 1;
    const BT709: u32 = 1;
    let transfer = if format.is_srgb() { 2 } else { 1 };
    let (model, dims, bytes_plane, samples): (u32, u32, u32, Vec<[u32; 4]>) = match format.0 {
        37 | 43 => {
            let mut s = Vec::new();
            for (i, id) in [0u32, 1, 2, 15].into_iter().enumerate() {
                // Alpha is linear in an sRGB texture.
                let qualifiers = if id == 15 && format.is_srgb() { 0x10 } else { 0 };
                s.push([(i as u32 * 8) | (7 << 16) | ((id | qualifiers) << 24), 0, 0, 255]);
            }
            (MODEL_RGBSDA, 0, 4, s)
        }
        _ => {
            // One sample covering the 64 or 128-bit block; the colour
            // models KHR_DF_MODEL_BC1A 128, BC2 129, BC3 130, BC4 131,
            // BC5 132, BC6H 133, BC7 134.
            let (model, bits) = match format.0 {
                131..=134 => (128, 64),
                135 | 136 => (129, 128),
                137 | 138 => (130, 128),
                139 | 140 => (131, 64),
                141 | 142 => (132, 128),
                143 | 144 => (133, 128),
                _ => (134, 128),
            };
            (model, 3 | (3 << 8), bits / 8, vec![[(bits - 1) << 16, 0, 0, u32::MAX]])
        }
    };
    let block_size = 24 + 16 * samples.len() as u32;
    let mut words = vec![block_size + 4, 0, 2 | (block_size << 16), model | (BT709 << 8) | (transfer << 16), dims];
    // libktx zeroes bytesPlane0 for supercompressed data.
    words.push(if zstd { 0 } else { bytes_plane });
    words.push(0);
    for s in samples {
        words.extend(s);
    }
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A zstd frame encoder.
pub type Compress = dyn Fn(&[u8]) -> Vec<u8>;

/// Writes `image` as KTX2, its levels zstd-compressed (ruzstd's fastest
/// level) when `zstd`.
pub fn write(image: &Image, zstd: bool) -> Result<Vec<u8>> {
    if zstd {
        write_with(image, Some(&|l: &[u8]| ruzstd::encoding::compress_to_vec(l, ruzstd::encoding::CompressionLevel::Fastest)))
    } else {
        write_with(image, None)
    }
}

/// Writes `image` as KTX2, each level through `zstd` (a zstd frame
/// encoder) when given.
pub fn write_with(image: &Image, zstd: Option<&Compress>) -> Result<Vec<u8>> {
    image.check()?;
    let n = image.levels.len();
    let compress = zstd;
    let zstd = compress.is_some();
    let dfd = dfd(image.format, zstd);
    let index_end = 80 + 24 * n;
    let dfd_offset = index_end;
    let data_start = dfd_offset + dfd.len();
    let (_, block_bytes) = image.format.block().unwrap();
    // Uncompressed levels align to lcm(block bytes, 4); compressed to 1.
    let align = if zstd { 1 } else { block_bytes.max(4) as usize };
    let payloads: Vec<Vec<u8>> = image
        .levels
        .iter()
        .map(|l| match compress {
            Some(c) => c(l),
            None => l.clone(),
        })
        .collect();
    // Smallest level first in the file.
    let mut out = vec![0u8; data_start];
    let mut index = vec![(0u64, 0u64, 0u64); n];
    for level in (0..n).rev() {
        out.resize(out.len().div_ceil(align) * align, 0);
        index[level] = (out.len() as u64, payloads[level].len() as u64, image.levels[level].len() as u64);
        out.extend_from_slice(&payloads[level]);
    }
    let mut head = Vec::with_capacity(80);
    head.extend_from_slice(&[0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A]);
    // typeSize: 1 for 8-bit components and for block-compressed formats.
    for w in [image.format.0, 1, image.width, image.height, 0, 0, 1, n as u32, if zstd { SUPERCOMPRESSION_ZSTD } else { SUPERCOMPRESSION_NONE }, dfd_offset as u32, dfd.len() as u32, 0, 0] {
        head.extend_from_slice(&w.to_le_bytes());
    }
    head.extend_from_slice(&0u64.to_le_bytes());
    head.extend_from_slice(&0u64.to_le_bytes());
    for (o, l, u) in index {
        head.extend_from_slice(&o.to_le_bytes());
        head.extend_from_slice(&l.to_le_bytes());
        head.extend_from_slice(&u.to_le_bytes());
    }
    out[..index_end].copy_from_slice(&head);
    out[dfd_offset..data_start].copy_from_slice(&dfd);
    Ok(out)
}
