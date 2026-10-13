//! Deterministic utilities: RNG, hashing for reproducibility checks, and a
//! libm-free sine/cosine.

use crate::vec2::Vec2;

/// xorshift32* — tiny, fast, fully deterministic (integer-only state transition).
#[derive(Clone, Debug)]
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Rng(if seed == 0 { 0x9E37_79B9 } else { seed })
    }
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x.wrapping_mul(0x2545_F491)
    }
    /// Uniform in [0, 1). Uses the top 24 bits so the f32 conversion is exact.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }
    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

/// FNV-1a over raw f32 bit patterns. Used to assert bit-exact reproducibility.
#[derive(Clone, Copy)]
pub struct BitHasher(u64);
impl BitHasher {
    pub fn new() -> Self {
        BitHasher(0xcbf2_9ce4_8422_2325)
    }
    #[inline]
    pub fn push_u32(&mut self, v: u32) {
        for b in v.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    #[inline]
    pub fn push_f32(&mut self, v: f32) {
        self.push_u32(v.to_bits());
    }
    pub fn finish(self) -> u64 {
        self.0
    }
}
impl Default for BitHasher {
    fn default() -> Self {
        Self::new()
    }
}

/// Deterministic `(sin θ, cos θ)` using only `+ − × floor` — no libm.
///
/// `f32::sin` calls the platform libm, whose last-ulp behaviour differs between
/// glibc / musl / macOS / wasm, which would silently break cross-platform
/// bit-exactness the moment a rotated box is created.
///
/// Method: reduce θ to r ∈ [−π/4, π/4] with k = round(θ·2/π), evaluate Taylor
/// polynomials (truncation error < 2e-9 on that interval, i.e. below f32 ε), then
/// rotate by k quadrants.
pub fn det_sin_cos(theta: f32) -> (f32, f32) {
    const TWO_OVER_PI: f32 = 0.636_619_77;
    // π/2 split in two parts (Cody–Waite) so k·π/2 is subtracted accurately.
    const PIO2_HI: f32 = 1.570_312_5; // exactly representable
    const PIO2_LO: f32 = 4.838_267_9e-4;
    let k = (theta * TWO_OVER_PI + 0.5).floor();
    let r = (theta - k * PIO2_HI) - k * PIO2_LO;
    let r2 = r * r;
    let s = r * (1.0 + r2 * (-1.0 / 6.0 + r2 * (1.0 / 120.0 + r2 * (-1.0 / 5040.0 + r2 * (1.0 / 362_880.0)))));
    let c = 1.0 + r2 * (-0.5 + r2 * (1.0 / 24.0 + r2 * (-1.0 / 720.0 + r2 * (1.0 / 40_320.0 + r2 * (-1.0 / 3_628_800.0)))));
    match (k as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// Unit vector for angle `theta`: (cos θ, sin θ).
pub fn unit_from_angle(theta: f32) -> Vec2 {
    let (s, c) = det_sin_cos(theta);
    Vec2::new(c, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sincos_accuracy_vs_libm() {
        let mut worst = 0f32;
        let mut t = -20.0f32;
        while t < 20.0 {
            let (s, c) = det_sin_cos(t);
            worst = worst.max((s - t.sin()).abs()).max((c - t.cos()).abs());
            t += 0.00137;
        }
        assert!(worst < 2.0e-6, "worst abs error {worst}");
    }

    #[test]
    fn sincos_is_unit() {
        let mut t = -10.0f32;
        while t < 10.0 {
            let (s, c) = det_sin_cos(t);
            assert!((s * s + c * c - 1.0).abs() < 2e-6);
            t += 0.01;
        }
        assert_eq!(det_sin_cos(0.0), (0.0, 1.0));
    }

    #[test]
    fn rng_is_reproducible_and_in_range() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            let x = a.next_f32();
            assert_eq!(x.to_bits(), b.next_f32().to_bits());
            assert!((0.0..1.0).contains(&x));
        }
    }
}
