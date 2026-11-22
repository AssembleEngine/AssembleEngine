//! glTF images to the model's KTX2 textures.
//!
//! * PNG and JPEG: RGBA8 (sRGB or UNORM by use), level 0 only (the
//!   renderer makes the mips on the GPU, `ae_render::upload::image_mipped`),
//!   zstd-supercompressed: lossless.
//! * KTX2 in a GPU format the runtime takes: kept, its VkFormat switched
//!   to the sRGB or UNORM twin its use asks for.
//! * KTX2 with Basis Universal (`KHR_texture_basisu`; VkFormat undefined):
//!   transcoded to BC7, every level it has.

use ae_model::ktx::{self, Format, Image};
use anyhow::{Context as _, Result, bail, ensure};

const KTX2_MAGIC: [u8; 12] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];

/// Encodes one source image as a KTX2 for a use that is (or not) sRGB.
pub fn to_ktx2(bytes: &[u8], srgb: bool, warnings: &mut Vec<String>, what: &str) -> Result<Vec<u8>> {
    if bytes.starts_with(&KTX2_MAGIC) {
        return from_ktx2(bytes, srgb, warnings, what);
    }
    let decoded = image::load_from_memory(bytes)
        .with_context(|| format!("{what}: decode"))?
        .to_rgba8();
    let (w, h) = decoded.dimensions();
    let levels = vec![decoded.into_raw()];
    let format = if srgb { Format::RGBA8_SRGB } else { Format::RGBA8_UNORM };
    write(&Image {
        format,
        width: w,
        height: h,
        levels,
    })
}

fn from_ktx2(bytes: &[u8], srgb: bool, warnings: &mut Vec<String>, what: &str) -> Result<Vec<u8>> {
    // VkFormat at byte 12; 0 is undefined (Basis).
    let vk = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let mut image = if vk == 0 {
        transcode_basis(bytes, srgb).with_context(|| format!("{what}: Basis transcode"))?
    } else {
        ktx::read(bytes).with_context(|| format!("{what}: KTX2"))?
    };
    match image.format.with_srgb(srgb) {
        Some(f) => image.format = f,
        None if srgb => warnings.push(format!(
            "{what}: VkFormat {} has no sRGB twin; sampled as linear",
            image.format.0
        )),
        None => {}
    }
    write(&image)
}

/// KTX2 with zstd at level 19 (the runtime decodes with ruzstd).
fn write(image: &Image) -> Result<Vec<u8>> {
    ktx::write_with(
        image,
        Some(&|l: &[u8]| zstd::bulk::compress(l, 19).expect("zstd compresses")),
    )
}

/// Basis Universal (ETC1S or UASTC LDR) KTX2 to BC7.
fn transcode_basis(bytes: &[u8], srgb: bool) -> Result<Image> {
    use basisu_c_sys::TranscodeTargetFormat;
    use basisu_c_sys::extra::{BasisuTranscoder, ChannelType, SupportedTextureCompression, basisu_transcoder_init};
    basisu_transcoder_init();
    let t = BasisuTranscoder::new(bytes, SupportedTextureCompression::all(), ChannelType::Auto)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let info = t.get_info();
    ensure!(
        info.faces == 1 && info.layers <= 1,
        "Basis KTX2 with {} faces, {} layers: only plain 2D textures",
        info.faces,
        info.layers
    );
    let out = t
        .transcode(Some(TranscodeTargetFormat::Bc7Rgba), Some(srgb))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let format = if srgb { Format::BC7_SRGB } else { Format::BC7_UNORM };
    let mut levels = Vec::new();
    let mut at = 0;
    for l in 0..info.levels.max(1) {
        let n = format
            .level_bytes((info.width >> l).max(1), (info.height >> l).max(1))
            .unwrap();
        ensure!(at + n <= out.data.len(), "transcoded data shorter than its levels");
        levels.push(out.data[at..at + n].to_vec());
        at += n;
    }
    if at != out.data.len() {
        bail!("transcoded data {} bytes, levels {at}", out.data.len());
    }
    Ok(Image {
        format,
        width: info.width,
        height: info.height,
        levels,
    })
}
