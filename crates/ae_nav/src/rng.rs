//! Small deterministic PRNG and value-noise helpers. No wall clock or OS RNG is
//! consulted by simulation generation.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self { Self(seed ^ 0x9e37_79b9_7f4a_7c15) }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn next_u32(&mut self) -> u32 { (self.next_u64() >> 32) as u32 }
    pub fn f32(&mut self) -> f32 { (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 }
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + (hi - lo) * self.f32() }
    pub fn range_i32(&mut self, lo: i32, hi_excl: i32) -> i32 {
        if hi_excl <= lo { lo } else { lo + (self.next_u32() % (hi_excl - lo) as u32) as i32 }
    }
    pub fn chance(&mut self, p: f32) -> bool { self.f32() < p }
}

#[inline]
pub fn hash2(seed: u64, x: i32, y: i32) -> u32 {
    let mut h = seed ^ (x as u32 as u64).wrapping_mul(0xa076_1d64_78bd_642f)
        ^ (y as u32 as u64).wrapping_mul(0xe703_7ed1_a0b4_28db);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h as u32
}

fn smooth(t: f32) -> f32 { t * t * (3.0 - 2.0 * t) }

pub fn value_noise(seed: u64, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = smooth(x - x0 as f32);
    let fy = smooth(y - y0 as f32);
    let a = hash2(seed, x0, y0) as f32 / u32::MAX as f32;
    let b = hash2(seed, x0 + 1, y0) as f32 / u32::MAX as f32;
    let c = hash2(seed, x0, y0 + 1) as f32 / u32::MAX as f32;
    let d = hash2(seed, x0 + 1, y0 + 1) as f32 / u32::MAX as f32;
    let ab = a + (b - a) * fx;
    let cd = c + (d - c) * fx;
    ab + (cd - ab) * fy
}

pub fn fbm(seed: u64, x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for o in 0..octaves {
        sum += amp * value_noise(seed.wrapping_add(o as u64 * 0x517c_c1b7_2722_0a95), x * freq, y * freq);
        norm += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    sum / norm
}

/// Deterministic 64-bit hash used for authoritative state digests.
#[derive(Clone, Debug, Default)]
pub struct StateHasher(u64);

impl StateHasher {
    pub fn new() -> Self { Self(0xcbf2_9ce4_8422_2325) }
    pub fn u32(&mut self, v: u32) { self.u64(v as u64); }
    pub fn u64(&mut self, v: u64) {
        self.0 ^= v;
        self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        self.0 ^= self.0 >> 29;
    }
    pub fn i32(&mut self, v: i32) { self.u32(v as u32); }
    pub fn f32_bits(&mut self, v: f32) { self.u32(v.to_bits()); }
    pub fn finish(self) -> u64 { self.0 }
}
