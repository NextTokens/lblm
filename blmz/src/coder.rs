//! Binary arithmetic coder (carry-less 32-bit range coder in the lpaq1/paq8 family) with
//! 32-bit probability precision.
//!
//! Invariant after every normalisation: x1 < x2 and their top bytes differ, so the range
//! x2 - x1 >= 1 and both sub-intervals of a split are non-empty for any p in [1, 2^32 - 1].
//! The decoder consumes exactly the bytes the encoder produced (4 initial bytes + one per shift
//! == one per shift + the 4-byte flush), which lets a container place data right after the
//! payload without storing the payload length.

/// Convert a model probability P(bit = 1) to the coder's 32-bit fixed point, in [1, 2^32 - 1].
/// NaN and out-of-range inputs are clamped deterministically (NaN -> 1).
#[inline]
pub fn quantize(p: f64) -> u32 {
    let q = p * 4294967296.0;
    if q >= 4294967295.0 {
        4294967295
    } else if q >= 1.0 {
        q as u32
    } else {
        1 // also NaN: comparisons are false
    }
}

#[inline]
fn split(x1: u32, x2: u32, p1: u32) -> u32 {
    x1 + (((x2 - x1) as u64 * p1 as u64) >> 32) as u32
}

pub struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    pub fn new() -> Self {
        Encoder {
            x1: 0,
            x2: u32::MAX,
            out: Vec::with_capacity(1 << 16),
        }
    }

    /// Code `bit` with P(bit = 1) = p1 / 2^32.
    #[inline]
    pub fn encode(&mut self, bit: u32, p1: u32) {
        let xmid = split(self.x1, self.x2, p1);
        if bit != 0 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
        }
    }

    /// Bytes produced so far that will not change any more.
    pub fn take_output(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.out)
    }

    /// Write the final 4 bytes (x1 exactly, so the decoder's code value lies in the final interval).
    pub fn finish(mut self) -> Vec<u8> {
        self.out.extend_from_slice(&self.x1.to_be_bytes());
        self.out
    }
}

/// Pull-based byte source for the decoder. Returns None at end of input.
pub trait ByteSource {
    fn next_byte(&mut self) -> Option<u8>;
}

pub struct Decoder<S: ByteSource> {
    x1: u32,
    x2: u32,
    x: u32,
    src: S,
    /// bytes requested past the end of the source (a valid stream never needs any)
    overrun: u32,
}

impl<S: ByteSource> Decoder<S> {
    pub fn new(mut src: S) -> Self {
        let mut x = 0u32;
        let mut overrun = 0;
        for _ in 0..4 {
            let b = match src.next_byte() {
                Some(b) => b,
                None => {
                    overrun += 1;
                    0
                }
            };
            x = (x << 8) | b as u32;
        }
        Decoder {
            x1: 0,
            x2: u32::MAX,
            x,
            src,
            overrun,
        }
    }

    #[inline]
    pub fn decode(&mut self, p1: u32) -> u32 {
        let xmid = split(self.x1, self.x2, p1);
        let bit = if self.x <= xmid {
            self.x2 = xmid;
            1
        } else {
            self.x1 = xmid + 1;
            0
        };
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
            let b = match self.src.next_byte() {
                Some(b) => b,
                None => {
                    self.overrun = self.overrun.saturating_add(1);
                    0
                }
            };
            self.x = (self.x << 8) | b as u32;
        }
        bit
    }

    /// Number of bytes the decoder wanted beyond the end of its source. Non-zero means the
    /// payload was truncated or corrupt.
    pub fn overrun(&self) -> u32 {
        self.overrun
    }

    pub fn into_source(self) -> S {
        self.src
    }

    pub fn source_mut(&mut self) -> &mut S {
        &mut self.src
    }
}

pub struct SliceSource<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

impl ByteSource for SliceSource<'_> {
    #[inline]
    fn next_byte(&mut self) -> Option<u8> {
        let b = self.data.get(self.pos).copied();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 11
    }

    #[test]
    fn roundtrip_random_bits_and_extreme_probabilities() {
        let mut s = 7u64;
        let n = 200_000;
        let mut bits = Vec::with_capacity(n);
        let mut ps = Vec::with_capacity(n);
        for i in 0..n {
            let p = match i % 5 {
                0 => 1,
                1 => u32::MAX,
                2 => (lcg(&mut s) as u32) | 1,
                3 => 1 << 31,
                _ => quantize(f64::NAN),
            };
            // bits deliberately disagree with extreme probabilities sometimes
            bits.push((lcg(&mut s) & 1) as u32);
            ps.push(p);
        }
        let mut e = Encoder::new();
        for i in 0..n {
            e.encode(bits[i], ps[i]);
        }
        let out = e.finish();
        let mut d = Decoder::new(SliceSource { data: &out, pos: 0 });
        for i in 0..n {
            assert_eq!(d.decode(ps[i]), bits[i], "bit {}", i);
        }
        assert_eq!(d.overrun(), 0);
        assert_eq!(d.into_source().pos, out.len(), "decoder must consume exactly the payload");
    }

    #[test]
    fn quantize_bounds() {
        assert_eq!(quantize(0.0), 1);
        assert_eq!(quantize(-3.0), 1);
        assert_eq!(quantize(f64::NAN), 1);
        assert_eq!(quantize(1.0), u32::MAX);
        assert_eq!(quantize(f64::INFINITY), u32::MAX);
        assert_eq!(quantize(0.5), 1 << 31);
    }
}
