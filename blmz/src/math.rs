//! Frozen, portable transcendental functions.
//!
//! The model's probabilities feed an arithmetic coder, so the DECODER must recompute every
//! probability bit-for-bit. IEEE-754 guarantees `+ - * / sqrt` are correctly rounded on every
//! platform Rust supports (and Rust never contracts `a*b+c` into an FMA on its own), but `ln`,
//! `exp` and `log2` come from the platform libm (glibc, the macOS libm, the MSVC CRT, ...) and
//! are NOT required to agree in the last bit. One differing ulp changes a probability, the coder
//! interval, and every byte after it.
//!
//! So the format pins its own implementations: the FreeBSD msun algorithms (via the pure-Rust
//! `libm` crate 0.2.8, MIT OR Apache-2.0), vendored here so that no dependency update can ever
//! change a decoded byte. They use only IEEE basic operations and integer bit manipulation, so
//! they give the same result on every IEEE-754 target. Changing ANY line of this file is a
//! format change: it must bump the model id in `format.rs`.
//!
//! Building with `RUSTFLAGS="--cfg blmz_std_math"` swaps in the platform functions. It exists only
//! to prove the port is bit-identical to the research engine `blmrs/src/bin/strong.rs` on the same
//! machine (tools/parity.sh); streams written by such a build carry a different model id and are
//! refused by normal builds. It is a compiler cfg, not a Cargo feature, so no dependent crate can
//! switch it on by accident.
//!
//! Basic operations are IEEE-exact only with SSE2 (or any non-x87) floating point: 32-bit x86
//! without SSE2 is refused at compile time (lib.rs).
//!
//! Original notice of the vendored algorithms (exp, log, log2):
//!
//! ```text
//! ====================================================
//! Copyright (C) 1993, 2004 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunSoft, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================
//! ```

#[cfg(blmz_std_math)]
#[inline]
pub fn ln(x: f64) -> f64 {
    x.ln()
}
#[cfg(blmz_std_math)]
#[inline]
pub fn exp(x: f64) -> f64 {
    x.exp()
}
#[cfg(blmz_std_math)]
#[inline]
pub fn log2(x: f64) -> f64 {
    x.log2()
}

#[cfg(not(blmz_std_math))]
pub use portable::{exp, ln, log2};

#[cfg_attr(blmz_std_math, allow(dead_code))]
// vendored verbatim in substance: keep msun's constants and idioms exactly
#[allow(clippy::eq_op, clippy::approx_constant, clippy::excessive_precision)]
pub mod portable {
    // ---- exp: origin FreeBSD /usr/src/lib/msun/src/e_exp.c (via libm 0.2.8) ----
    const HALF: [f64; 2] = [0.5, -0.5];
    const LN2HI: f64 = 6.93147180369123816490e-01; /* 0x3fe62e42, 0xfee00000 */
    const LN2LO: f64 = 1.90821492927058770002e-10; /* 0x3dea39ef, 0x35793c76 */
    const INVLN2: f64 = 1.44269504088896338700e+00; /* 0x3ff71547, 0x652b82fe */
    const P1: f64 = 1.66666666666666019037e-01; /* 0x3FC55555, 0x5555553E */
    const P2: f64 = -2.77777777770155933842e-03; /* 0xBF66C16C, 0x16BEBD93 */
    const P3: f64 = 6.61375632143793436117e-05; /* 0x3F11566A, 0xAF25DE2C */
    const P4: f64 = -1.65339022054652515390e-06; /* 0xBEBBBD41, 0xC5D26BF1 */
    const P5: f64 = 4.13813679705723846039e-08; /* 0x3E663769, 0x72BEA4D0 */

    fn scalbn(x: f64, mut n: i32) -> f64 {
        let x1p1023 = f64::from_bits(0x7fe0000000000000);
        let x1p53 = f64::from_bits(0x4340000000000000);
        let x1p_1022 = f64::from_bits(0x0010000000000000);
        let mut y = x;
        if n > 1023 {
            y *= x1p1023;
            n -= 1023;
            if n > 1023 {
                y *= x1p1023;
                n -= 1023;
                if n > 1023 {
                    n = 1023;
                }
            }
        } else if n < -1022 {
            y *= x1p_1022 * x1p53;
            n += 1022 - 53;
            if n < -1022 {
                y *= x1p_1022 * x1p53;
                n += 1022 - 53;
                if n < -1022 {
                    n = -1022;
                }
            }
        }
        y * f64::from_bits(((0x3ff + n) as u64) << 52)
    }

    /// e^x, < 1 ulp (FreeBSD msun).
    pub fn exp(mut x: f64) -> f64 {
        let x1p1023 = f64::from_bits(0x7fe0000000000000);
        let hi: f64;
        let lo: f64;
        let k: i32;
        let mut hx = (x.to_bits() >> 32) as u32;
        let sign = (hx >> 31) as i32;
        hx &= 0x7fffffff;
        if hx >= 0x4086232b {
            if x.is_nan() {
                return x;
            }
            if x > 709.782712893383973096 {
                x *= x1p1023;
                return x;
            }
            if x < -745.13321910194110842 {
                return 0.;
            }
        }
        if hx > 0x3fd62e42 {
            if hx >= 0x3ff0a2b2 {
                k = (INVLN2 * x + HALF[sign as usize]) as i32;
            } else {
                k = 1 - sign - sign;
            }
            hi = x - k as f64 * LN2HI;
            lo = k as f64 * LN2LO;
            x = hi - lo;
        } else if hx > 0x3e300000 {
            k = 0;
            hi = x;
            lo = 0.;
        } else {
            return 1. + x;
        }
        let xx = x * x;
        let c = x - xx * (P1 + xx * (P2 + xx * (P3 + xx * (P4 + xx * P5))));
        let y = 1. + (x * c / (2. - c) - lo + hi);
        if k == 0 {
            y
        } else {
            scalbn(y, k)
        }
    }

    // ---- log / log2: origin FreeBSD /usr/src/lib/msun/src/e_log.c, e_log2.c (via libm 0.2.8) ----
    const LN2_HI: f64 = 6.93147180369123816490e-01; /* 3fe62e42 fee00000 */
    const LN2_LO: f64 = 1.90821492927058770002e-10; /* 3dea39ef 35793c76 */
    const IVLN2HI: f64 = 1.44269504072144627571e+00; /* 0x3ff71547, 0x65200000 */
    const IVLN2LO: f64 = 1.67517131648865118353e-10; /* 0x3de705fc, 0x2eefa200 */
    const LG1: f64 = 6.666666666666735130e-01; /* 3FE55555 55555593 */
    const LG2: f64 = 3.999999999940941908e-01; /* 3FD99999 9997FA04 */
    const LG3: f64 = 2.857142874366239149e-01; /* 3FD24924 94229359 */
    const LG4: f64 = 2.222219843214978396e-01; /* 3FCC71C5 1D8E78AF */
    const LG5: f64 = 1.818357216161805012e-01; /* 3FC74664 96CB03DE */
    const LG6: f64 = 1.531383769920937332e-01; /* 3FC39A09 D078C69F */
    const LG7: f64 = 1.479819860511658591e-01; /* 3FC2F112 DF3E5244 */

    /// Shared argument reduction: x = 2^k * (1+f), sqrt(2)/2 < 1+f < sqrt(2).
    /// Returns None for the special cases, with the value to return.
    #[inline]
    fn reduce(mut x: f64) -> Result<(f64, i32), f64> {
        let x1p54 = f64::from_bits(0x4350000000000000);
        let mut ui = x.to_bits();
        let mut hx = (ui >> 32) as u32;
        let mut k: i32 = 0;
        if hx < 0x00100000 || (hx >> 31) != 0 {
            if ui << 1 == 0 {
                return Err(-1. / (x * x));
            }
            if hx >> 31 != 0 {
                return Err((x - x) / 0.0);
            }
            k -= 54;
            x *= x1p54;
            ui = x.to_bits();
            hx = (ui >> 32) as u32;
        } else if hx >= 0x7ff00000 {
            return Err(x);
        } else if hx == 0x3ff00000 && ui << 32 == 0 {
            return Err(0.);
        }
        hx += 0x3ff00000 - 0x3fe6a09e;
        k += ((hx >> 20) as i32) - 0x3ff;
        hx = (hx & 0x000fffff) + 0x3fe6a09e;
        ui = ((hx as u64) << 32) | (ui & 0xffffffff);
        Ok((f64::from_bits(ui), k))
    }

    /// Natural logarithm, < 1 ulp (FreeBSD msun).
    pub fn ln(x: f64) -> f64 {
        let (x, k) = match reduce(x) {
            Ok(v) => v,
            Err(special) => return special,
        };
        let f: f64 = x - 1.0;
        let hfsq: f64 = 0.5 * f * f;
        let s: f64 = f / (2.0 + f);
        let z: f64 = s * s;
        let w: f64 = z * z;
        let t1: f64 = w * (LG2 + w * (LG4 + w * LG6));
        let t2: f64 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
        let r: f64 = t2 + t1;
        let dk: f64 = k as f64;
        s * (hfsq + r) + dk * LN2_LO - hfsq + f + dk * LN2_HI
    }

    /// Base-2 logarithm, < 1 ulp (FreeBSD msun).
    pub fn log2(x: f64) -> f64 {
        let (x, k) = match reduce(x) {
            Ok(v) => v,
            Err(special) => return special,
        };
        let f = x - 1.0;
        let hfsq = 0.5 * f * f;
        let s = f / (2.0 + f);
        let z = s * s;
        let mut w = z * z;
        let t1 = w * (LG2 + w * (LG4 + w * LG6));
        let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
        let r = t2 + t1;
        let mut hi = f - hfsq;
        hi = f64::from_bits(hi.to_bits() & ((-1i64 as u64) << 32));
        let lo = f - hi - hfsq + s * (hfsq + r);
        let mut val_hi = hi * IVLN2HI;
        let mut val_lo = (lo + hi) * IVLN2LO + lo * IVLN2HI;
        let y: f64 = k.into();
        w = y + val_hi;
        val_lo += (y - w) + val_hi;
        val_hi = w;
        val_lo + val_hi
    }
}

/// logit: ln(p / (1 - p)) with p clamped to [1e-6, 1 - 1e-6] (strong.rs `stretch`).
#[inline]
pub fn stretch(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    ln(p / (1.0 - p))
}

/// logistic: 1 / (1 + e^-t), saturating outside |t| <= 30 (strong.rs `squash`).
#[inline]
pub fn squash(t: f64) -> f64 {
    if t > 30.0 {
        1.0 - 1e-6
    } else if t < -30.0 {
        1e-6
    } else {
        1.0 / (1.0 + exp(-t))
    }
}

#[cfg(test)]
mod tests {
    use super::portable;

    const FROZEN_GRID_HASH: u64 = 16261153804132893190;

    /// The portable functions must stay within 1 ulp of the platform's on the engine's domain.
    #[test]
    fn portable_matches_platform_within_one_ulp() {
        let mut worst = 0u64;
        let ulps = |a: f64, b: f64| (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs();
        let mut x = 1e-9f64;
        while x < 1e6 {
            worst = worst.max(ulps(portable::ln(x), x.ln()));
            worst = worst.max(ulps(portable::log2(x), x.log2()));
            x *= 1.000_123;
        }
        let mut t = -40.0f64;
        while t < 40.0 {
            worst = worst.max(ulps(portable::exp(t), t.exp()));
            t += 0.000_731;
        }
        assert!(worst <= 1, "worst ulp distance {}", worst);
    }

    /// FNV-1a over the IEEE bits of ln/log2/exp on a dense grid covering the engine's domain.
    /// Pinned: if this changes, decoded output changes, so it is a FORMAT change (new model id).
    pub(crate) fn grid_hash() -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut eat = |v: f64| {
            for b in v.to_bits().to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            }
        };
        let mut x = 1e-9f64;
        while x < 1e6 {
            eat(portable::ln(x));
            eat(portable::log2(x));
            x *= 1.000_37;
        }
        let mut t = -40.0f64;
        while t < 40.0 {
            eat(portable::exp(t));
            t += 0.001_9;
        }
        h
    }

    #[test]
    fn frozen_values() {
        assert_eq!(grid_hash(), FROZEN_GRID_HASH, "portable math changed: this is a format change");
    }
}
