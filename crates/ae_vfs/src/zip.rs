//! Zip archives: the subset content packages need, so any zip tool can
//! make or open one.
//!
//! Reading takes stored (0) and deflated (8) entries, checks each entry's
//! CRC-32 and size, and refuses encryption, zip64, multi-disk archives,
//! duplicate paths and paths that are not content paths. Directory entries
//! are skipped.
//!
//! [`write`] is deterministic: entries sorted by path, a fixed timestamp
//! (1980-01-01 00:00), deflate at level 6 unless storing is smaller, no
//! extra fields. The same files always give the same bytes.

use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::sync::Arc;

const LOCAL: u32 = 0x0403_4b50;
const CENTRAL: u32 = 0x0201_4b50;
const END: u32 = 0x0605_4b50;
const STORED: u16 = 0;
const DEFLATED: u16 = 8;
/// DOS date of 1980-01-01 (time 00:00 is 0).
const DOS_DATE: u16 = 0x21;
/// Largest uncompressed entry accepted (a corrupt size must not allocate
/// without bound).
const MAX_ENTRY: usize = 1 << 31;

#[derive(Clone, Copy, Debug)]
struct Entry {
    method: u16,
    crc: u32,
    compressed: usize,
    size: usize,
    local: usize,
}

/// An opened archive: its bytes and its entries by content path.
#[derive(Clone, Debug)]
pub struct Archive {
    bytes: Arc<Vec<u8>>,
    entries: BTreeMap<String, Entry>,
}

fn u16_at(b: &[u8], at: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(b.get(at..at + 2).context("truncated zip")?.try_into().unwrap()))
}

fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(b.get(at..at + 4).context("truncated zip")?.try_into().unwrap()))
}

impl Archive {
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        let b = &bytes[..];
        // The end record: 22 bytes plus a comment of up to 64 KiB.
        ensure!(b.len() >= 22, "not a zip archive (too short)");
        let lowest = b.len().saturating_sub(22 + 0xFFFF);
        let end = (lowest..=b.len() - 22).rev().find(|&i| u32_at(b, i).is_ok_and(|s| s == END)).context("not a zip archive (no end record)")?;
        ensure!(u16_at(b, end + 4)? == 0 && u16_at(b, end + 6)? == 0, "multi-disk zip archives are not supported");
        let count = u16_at(b, end + 10)? as usize;
        let (cd_size, cd_at) = (u32_at(b, end + 12)? as usize, u32_at(b, end + 16)? as usize);
        ensure!(count != 0xFFFF && cd_at != 0xFFFF_FFFF, "zip64 archives are not supported");
        ensure!(cd_at + cd_size <= end, "zip central directory out of range");
        let mut entries = BTreeMap::new();
        let mut at = cd_at;
        for _ in 0..count {
            ensure!(u32_at(b, at)? == CENTRAL, "bad zip central directory entry at {at}");
            let flags = u16_at(b, at + 8)?;
            let method = u16_at(b, at + 10)?;
            let crc = u32_at(b, at + 16)?;
            let compressed = u32_at(b, at + 20)? as usize;
            let size = u32_at(b, at + 24)? as usize;
            let (n, x, c) = (u16_at(b, at + 28)? as usize, u16_at(b, at + 30)? as usize, u16_at(b, at + 32)? as usize);
            let local = u32_at(b, at + 42)? as usize;
            let name = std::str::from_utf8(b.get(at + 46..at + 46 + n).context("truncated zip")?).context("zip entry name is not UTF-8")?.to_owned();
            at += 46 + n + x + c;
            if name.ends_with('/') {
                continue;
            }
            crate::check_path(&name).with_context(|| format!("zip entry {name:?}"))?;
            ensure!(flags & 1 == 0, "{name}: encrypted zip entries are not supported");
            ensure!(method == STORED || method == DEFLATED, "{name}: zip method {method} (only stored and deflated)");
            ensure!(size <= MAX_ENTRY && compressed != 0xFFFF_FFFF, "{name}: too large");
            if entries.insert(name.clone(), Entry { method, crc, compressed, size, local }).is_some() {
                bail!("{name} is in the archive twice");
            }
        }
        Ok(Self { bytes: Arc::new(bytes), entries })
    }

    pub fn open(path: &std::path::Path) -> Result<Self> {
        Self::parse(std::fs::read(path).with_context(|| format!("read {}", path.display()))?).with_context(|| format!("archive {}", path.display()))
    }

    /// Every file path, sorted.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn contains(&self, path: &str) -> bool {
        self.entries.contains_key(path)
    }

    /// The file's bytes (checked against its size and CRC), or `None`.
    pub fn read(&self, path: &str) -> Option<Result<Vec<u8>>> {
        let e = *self.entries.get(path)?;
        Some(self.extract(e).with_context(|| format!("{path} in the archive")))
    }

    fn extract(&self, e: Entry) -> Result<Vec<u8>> {
        let b = &self.bytes[..];
        ensure!(u32_at(b, e.local)? == LOCAL, "bad local header");
        let start = e.local + 30 + u16_at(b, e.local + 26)? as usize + u16_at(b, e.local + 28)? as usize;
        let data = b.get(start..start + e.compressed).context("data out of range")?;
        let out = match e.method {
            STORED => data.to_vec(),
            _ => miniz_oxide::inflate::decompress_to_vec_with_limit(data, e.size).map_err(|err| anyhow::anyhow!("inflate: {err:?}"))?,
        };
        ensure!(out.len() == e.size, "size {} where the directory says {}", out.len(), e.size);
        ensure!(crc32fast::hash(&out) == e.crc, "CRC mismatch (corrupt archive)");
        Ok(out)
    }
}

/// A zip of `files` (content paths and bytes), deterministic.
pub fn write(files: &BTreeMap<String, Vec<u8>>) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    ensure!(files.len() < 0xFFFF, "too many files for a zip without zip64");
    for (name, data) in files {
        crate::check_path(name)?;
        ensure!(data.len() < 0xFFFF_FFFF, "{name} is too large for a zip without zip64");
        let deflated = miniz_oxide::deflate::compress_to_vec(data, 6);
        let (method, body) = if deflated.len() < data.len() { (DEFLATED, &deflated[..]) } else { (STORED, &data[..]) };
        let crc = crc32fast::hash(data);
        let local = out.len() as u32;
        let header = |sig: u32, out: &mut Vec<u8>, central: bool| {
            out.extend_from_slice(&sig.to_le_bytes());
            if central {
                out.extend_from_slice(&20u16.to_le_bytes()); // made by
            }
            out.extend_from_slice(&20u16.to_le_bytes()); // needed
            out.extend_from_slice(&0x0800u16.to_le_bytes()); // flags: UTF-8 names
            out.extend_from_slice(&method.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // time
            out.extend_from_slice(&DOS_DATE.to_le_bytes());
            out.extend_from_slice(&crc.to_le_bytes());
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // extra
            if central {
                out.extend_from_slice(&[0; 6]); // comment length, disk, internal attributes
                out.extend_from_slice(&0u32.to_le_bytes()); // external attributes
                out.extend_from_slice(&local.to_le_bytes());
            }
            out.extend_from_slice(name.as_bytes());
        };
        header(LOCAL, &mut out, false);
        out.extend_from_slice(body);
        header(CENTRAL, &mut central, true);
    }
    let cd_at = out.len();
    ensure!(cd_at < 0xFFFF_FFFF, "archive too large for a zip without zip64");
    out.extend_from_slice(&central);
    out.extend_from_slice(&END.to_le_bytes());
    out.extend_from_slice(&[0; 4]); // disks
    out.extend_from_slice(&(files.len() as u16).to_le_bytes());
    out.extend_from_slice(&(files.len() as u16).to_le_bytes());
    out.extend_from_slice(&(central.len() as u32).to_le_bytes());
    out.extend_from_slice(&(cd_at as u32).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_deterministic_and_checked() {
        let mut files = BTreeMap::new();
        files.insert("defs/units.json".to_string(), b"{\"a\": 1}\n".repeat(50));
        files.insert("maps/x.bin".to_string(), vec![7u8, 1, 200]);
        files.insert("empty".to_string(), Vec::new());
        let zip = write(&files).unwrap();
        assert_eq!(write(&files).unwrap(), zip, "same files, same bytes");
        let a = Archive::parse(zip.clone()).unwrap();
        assert_eq!(a.paths().collect::<Vec<_>>(), ["defs/units.json", "empty", "maps/x.bin"]);
        for (p, data) in &files {
            assert_eq!(&a.read(p).unwrap().unwrap(), data, "{p}");
        }
        assert!(a.read("none").is_none());
        // The repeated text is deflated; a flipped byte in its data fails the CRC or inflate.
        let mut bad = zip.clone();
        let at = 30 + "defs/units.json".len() + 3;
        bad[at] ^= 0x55;
        assert!(Archive::parse(bad).unwrap().read("defs/units.json").unwrap().is_err());
        // A path that is not a content path is refused.
        let mut evil = BTreeMap::new();
        evil.insert("../x".to_string(), vec![1]);
        assert!(write(&evil).is_err());
        assert!(Archive::parse(b"not a zip at all, just some bytes".to_vec()).is_err());
    }
}
