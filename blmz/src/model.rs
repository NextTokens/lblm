//! The bit predictor: a faithful port of the ADOPTED DEFAULT path of the research engine
//! `blmrs/src/bin/strong.rs` (ledger §63, §79/§80, §82–§84, §91).
//!
//! Inputs to the mixers (index: model):
//!   0..8   byte-aware orders 0..7 (flat tables, 8-bit checksum tags, evict-on-mismatch)
//!   8..13  hashed high orders {8,12,16,24,32} (merged tables, no tags)
//!   13..16 sparse models over byte pairs at offsets (2,3) (1,4) (3,6)
//!   16     word model, 17 previous-word (bigram) model — nonstationary counts (BLWNS=1)
//!   18,19  byte-match models (min length 5 and 8)
//!   20..25 indirect models (ICM): orders 2..6 bit-history byte -> adaptive StateMap
//!   25..33 the current word prefix's learned 8-dim vector (BLPVEC=2)
//!   33     constant 0.0 (the research engine's dead bias slot under BLSEL, ledger §86 — kept so
//!          the float summation order, and therefore every probability, is identical)
//!   34     the lean word-vote selector (BLSEL=1, N=2 slots, TOPM=4, tagged vote cells BLSELTAG=1)
//! Three logistic mixers (selected by order-1, by an order-2 hash, and one global; RMSProp), a
//! 4-weight final mixer, then six chained APM/SSE stages.
//!
//! Every read made while predicting bit i depends only on bits < i: an encoder and a decoder
//! running this code see the same probabilities. Differences from strong.rs are storage-only and
//! value-preserving: counts live in u8 (they never exceed 255 before halving — CLIMIT), tags and
//! counts share one slot, and history is a bounded window instead of the whole input (identical
//! for inputs no longer than the window).

use crate::math::{exp, log2, squash, stretch};

const NM: usize = 8; // byte orders 0..7
const NH: usize = 5; // hashed high orders
const HORDERS: [usize; NH] = [8, 12, 16, 24, 32];
const NSP: usize = 3;
const SPOFF: [(usize, usize); NSP] = [(2, 3), (1, 4), (3, 6)];
const NICM: usize = 5;
const ICM_K: [usize; NICM] = [2, 3, 4, 5, 6];
const NIN: usize = NM + NH + NSP + 4 + NICM; // 25
const PVD: usize = 8;
const I_DEAD: usize = NIN + PVD; // 33
const I_SEL: usize = NIN + PVD + 1; // 34
/// Active mixer width (inputs incl. the dead slot and the vote).
pub const NW: usize = I_SEL + 1; // 35

const NSEL: usize = 8 * 256;
const NSEL2: usize = 8 * 2048;
const CLIMIT: u32 = 255;
const MAXB: usize = 7;
const MULT: u64 = 0x9E37_79B9_7F4A_7C15;
const RMS_DECAY: f64 = 0.9999;
const RMS_EPS: f64 = 1e-4;

// adopted hyper-parameters (strong.rs env defaults)
const DELTA: f64 = 0.08;
const ALR: f64 = 0.0010;
const ALRF: f64 = 0.0010;
const LR_S: f64 = 0.02;

// §82 lean selector, adopted values (NSELSLOTS=2, SELTOPM=4, SELVORD=3, SELSENT=0, BLSELTAG=1)
const SELSMAX: usize = 64;
const SEL_TOPM: usize = 8;
const NSELSLOTS: usize = 2;
const SELTOPM: usize = 4;
const SEL_UGAIN: f64 = 4.0;
const SEL_PBD: f64 = 0.05;
const SEL_TSC: f64 = 2.0;
const SEL_UDC: f64 = 0.995;
const SEL_UCLIP: f64 = 4.0;
const SELCTXMASK: u64 = (1u64 << 24) - 1;
const SELTAGMASK: u8 = 0xFF;

/// Table sizes (log2). Part of the stream header: the decoder must allocate exactly these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    /// order / sparse / word / ICM tables: 2^obits slots each (strong.rs `obits`)
    pub obits: u8,
    /// hashed high-order tables: 2^hbits count pairs each (strong.rs HBITS = 22)
    pub hbits: u8,
    /// match-model hash tables: 2^mbits positions each (strong.rs 22)
    pub mbits: u8,
    /// prefix-vector table: 2^sbits vectors (strong.rs SBITS = 22)
    pub sbits: u8,
    /// selector vote cells: 2^selvbits (strong.rs SELVBITS = 23)
    pub selvbits: u8,
    /// selector usefulness cells: 2^selubits (strong.rs SELUBITS = 23)
    pub selubits: u8,
    /// history window: 2^window_log bytes. Identical to strong.rs for inputs <= the window.
    pub window_log: u8,
}

impl Params {
    /// The research engine's adopted configuration at `obits` (all other sizes = strong.rs defaults).
    pub const fn research(obits: u8) -> Params {
        Params {
            obits,
            hbits: 22,
            mbits: 22,
            sbits: 22,
            selvbits: 23,
            selubits: 23,
            window_log: 32,
        }
    }

    pub const MIN_BITS: u8 = 10;
    pub const MAX_BITS: u8 = 30;
    pub const MIN_WINDOW: u8 = 16;
    pub const MAX_WINDOW: u8 = 40;

    /// Reject anything a hostile header could use to make us allocate absurd memory or overflow
    /// shifts. (Memory limits on top of this are the caller's policy: see `memory_bytes`.)
    pub fn validate(&self) -> Result<(), String> {
        let tab = [
            ("obits", self.obits),
            ("hbits", self.hbits),
            ("mbits", self.mbits),
            ("sbits", self.sbits),
            ("selvbits", self.selvbits),
            ("selubits", self.selubits),
        ];
        for (name, v) in tab {
            if !(Self::MIN_BITS..=Self::MAX_BITS).contains(&v) {
                return Err(format!("{} = {} outside [{}, {}]", name, v, Self::MIN_BITS, Self::MAX_BITS));
            }
        }
        if !(Self::MIN_WINDOW..=Self::MAX_WINDOW).contains(&self.window_log) {
            return Err(format!(
                "window_log = {} outside [{}, {}]",
                self.window_log,
                Self::MIN_WINDOW,
                Self::MAX_WINDOW
            ));
        }
        if usize::BITS < 64 && (self.obits.max(self.selvbits) as u32 + 4 >= usize::BITS) {
            return Err("table sizes too large for this platform's address space".into());
        }
        Ok(())
    }

    /// Model memory in bytes (tables + mixers), excluding the history window, which grows with
    /// the input up to 2^window_log bytes.
    pub fn memory_bytes(&self) -> u64 {
        let o = 1u64 << self.obits;
        let order_like = (NM + NSP + 2) as u64 * o * 3; // tag + n0 + n1 (u8)
        let icm = NICM as u64 * o;
        let high = NH as u64 * (2u64 << self.hbits);
        let mm = 2 * 4 * (1u64 << self.mbits);
        let pv = (1u64 << self.sbits) * (8 * PVD as u64 + 8);
        let selv = (1u64 << self.selvbits) * (16 + 1);
        let selu = (1u64 << self.selubits) * 8;
        let mixers = (NSEL + NSEL2 + 1) as u64 * NW as u64 * 16;
        let apm = (2048 + 1024 + 2048 + 256 + 4096 + 4096) as u64 * 33 * 8;
        order_like + icm + high + mm + pv + selv + selu + mixers + apm
    }
}

/// A count slot: [8-bit checksum tag, n0, n1]; counts saturate by halving (always <= 254 at rest).
/// A plain byte array (not a struct) so `vec![[0; 3]; n]` gets lazily-zeroed pages from the OS.
type Slot = [u8; 3];

/// strong.rs count bump on a pair [n0, n1]: c[y] += 1; if c[y] >= CLIMIT { both = (c + 1) >> 1 }.
#[inline]
fn bump_pair(c: &mut [u8; 2], y: usize) {
    let v = c[y] as u32 + 1;
    if v >= CLIMIT {
        c[y] = ((v + 1) >> 1) as u8;
        c[1 - y] = ((c[1 - y] as u32 + 1) >> 1) as u8;
    } else {
        c[y] = v as u8;
    }
}

#[inline]
fn bump(s: &mut Slot, y: usize) {
    let mut c = [s[1], s[2]];
    bump_pair(&mut c, y);
    s[1] = c[0];
    s[2] = c[1];
}

/// ledger §79 PAQ nonstationary rule on the opposite count: if n_{1-y} > 2, n_{1-y} = n/2 + 1.
#[inline]
fn ns_discount(s: &mut Slot, y: usize) {
    let o = 2 - y; // y=1 -> n0 at [1]; y=0 -> n1 at [2]
    if s[o] > 2 {
        s[o] = s[o] / 2 + 1;
    }
}

#[inline]
fn count_p(n0: u8, n1: u8) -> f64 {
    let (n0, n1) = (n0 as f64, n1 as f64);
    (n1 + DELTA) / (n0 + n1 + 2.0 * DELTA)
}

/// stretch(count_p(n0, n1)) for every (n0, n1): counts are 8-bit, so the 18 per-bit `ln` calls on
/// count pairs become one table read each. Same function of the same inputs -> identical values.
fn stretch_table() -> &'static [f64] {
    static T: std::sync::OnceLock<Box<[f64]>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = vec![0.0f64; 1 << 16];
        for n0 in 0..256usize {
            for n1 in 0..256usize {
                t[(n0 << 8) | n1] = stretch(count_p(n0 as u8, n1 as u8));
            }
        }
        t.into_boxed_slice()
    })
}

#[inline]
fn st_count(t: &[f64], n0: u8, n1: u8) -> f64 {
    t[((n0 as usize) << 8) | n1 as usize]
}

#[inline]
fn sel_mix(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 32)
}

#[inline]
fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn is_letter(b: u8) -> bool {
    b.is_ascii_alphabetic()
}

/// Adaptive probability map (SSE): 33 knots over the stretch domain, per context.
struct Apm {
    t: Vec<f64>,
    rate: f64,
    lo: usize,
    w: f64,
    c: usize,
}

impl Apm {
    const K: usize = 33;
    const SMAX: f64 = 8.0;

    fn new(n_ctx: usize, rate: f64) -> Self {
        let step = 2.0 * Self::SMAX / (Self::K as f64 - 1.0);
        let mut t = vec![0.0f64; n_ctx * Self::K];
        for j in 0..Self::K {
            let v = squash(-Self::SMAX + j as f64 * step);
            for c in 0..n_ctx {
                t[c * Self::K + j] = v;
            }
        }
        Apm {
            t,
            rate,
            lo: 0,
            w: 0.0,
            c: 0,
        }
    }

    #[inline]
    fn refine(&mut self, p: f64, cx: usize) -> f64 {
        let step = 2.0 * Self::SMAX / (Self::K as f64 - 1.0);
        let s = stretch(p);
        let (lo, w);
        if s <= -Self::SMAX {
            lo = 0;
            w = 0.0;
        } else if s >= Self::SMAX {
            lo = Self::K - 2;
            w = 1.0;
        } else {
            let x = (s + Self::SMAX) / step;
            let mut l = x as usize;
            if l >= Self::K - 1 {
                l = Self::K - 2;
            }
            lo = l;
            w = x - l as f64;
        }
        self.lo = lo;
        self.w = w;
        self.c = cx;
        let base = cx * Self::K;
        self.t[base + lo] * (1.0 - w) + self.t[base + lo + 1] * w
    }

    #[inline]
    fn update(&mut self, y: f64) {
        let base = self.c * Self::K;
        let (lo, w, rt) = (self.lo, self.w, self.rate);
        self.t[base + lo] += rt * (1.0 - w) * (y - self.t[base + lo]);
        self.t[base + lo + 1] += rt * w * (y - self.t[base + lo + 1]);
    }
}

/// Bounded byte history. Positions are absolute (u64); reads older than the window are refused
/// by the callers, so behaviour equals strong.rs's unbounded `hist` while the input fits.
struct History {
    buf: Vec<u8>,
    mask: u64,
    cap: u64,
}

impl History {
    fn new(window_log: u8) -> Self {
        let cap = 1u64 << window_log;
        History {
            buf: Vec::new(),
            mask: cap - 1,
            cap,
        }
    }
    #[inline]
    fn push(&mut self, pos: u64, b: u8) {
        if (self.buf.len() as u64) < self.cap {
            self.buf.push(b);
        } else {
            self.buf[(pos & self.mask) as usize] = b;
        }
    }
    /// byte at absolute position `pos` (caller guarantees cur - pos <= cap)
    #[inline]
    fn at(&self, pos: u64) -> u8 {
        self.buf[(pos & self.mask) as usize]
    }
    #[inline]
    fn in_window(&self, pos: u64, byte_pos: u64) -> bool {
        pos < byte_pos && byte_pos - pos <= self.cap
    }
}

struct MatchModel {
    tab: Vec<u32>,
    mask: u64,
    minlen: u64,
    ptr: u64,
    len: usize,
    h: u64,
}

impl MatchModel {
    fn new(hash_bits: u8, minlen: u64) -> Self {
        MatchModel {
            tab: vec![0u32; 1usize << hash_bits],
            mask: (1u64 << hash_bits) - 1,
            minlen,
            ptr: 0,
            len: 0,
            h: 0,
        }
    }

    #[inline]
    fn predicted(&self, hist: &History, phase: usize, byte_pos: u64) -> f64 {
        if self.len == 0 || self.ptr >= byte_pos {
            return 0.0;
        }
        let pb = (hist.at(self.ptr) >> (7 - phase)) & 1;
        let st = 1.6 + 0.35 * (self.len.min(28) as f64);
        if pb == 1 {
            st
        } else {
            -st
        }
    }

    /// Called after byte `byte_pos` (now in `hist`) is known.
    fn update_after_byte(&mut self, hist: &History, byte_pos: u64) {
        if self.len > 0 && self.ptr < byte_pos {
            if hist.at(self.ptr) == hist.at(byte_pos) {
                self.ptr += 1;
                self.len = (self.len + 1).min(65535);
            } else {
                self.len = 0;
                self.ptr = 0;
            }
        }
        let b = hist.at(byte_pos) as u64;
        self.h = ((self.h << 8) | b) & 0xFFFF_FFFF_FFFF;
        if byte_pos + 1 >= self.minlen {
            let hk = (self.h.wrapping_mul(2654435761) & self.mask) as usize;
            // strong.rs stores (byte_pos + 1) as u32; recover the absolute position (inputs > 4 GiB)
            let stored = self.tab[hk] as u64;
            self.tab[hk] = (byte_pos + 1) as u32;
            if self.len == 0 && stored != 0 {
                let here = byte_pos + 1;
                let mut prev = (here & !0xFFFF_FFFFu64) | stored;
                if prev > here {
                    prev = prev.wrapping_sub(1u64 << 32);
                }
                // strong.rs: prev <= byte_pos (and the byte must still be in the window)
                if prev <= byte_pos && prev != 0 && hist.in_window(prev, byte_pos + 1) {
                    self.ptr = prev;
                    self.len = self.minlen as usize;
                }
            }
        }
    }
}

/// The predictor. Call `p()` then `update(bit)` for each bit, most significant bit first.
pub struct Model {
    params: Params,
    st: &'static [f64],
    // flat count tables
    ord: Vec<Vec<Slot>>,      // NM x 2^obits
    sp: Vec<Vec<Slot>>,       // NSP x 2^obits
    wd: Vec<Slot>,            // word
    wd2: Vec<Slot>,           // previous word
    high: Vec<Vec<[u8; 2]>>,  // NH x 2^hbits (no tags)
    icm_bh: Vec<Vec<u8>>,     // NICM x 2^obits bit-history bytes
    sm_p: [[f64; 256]; NICM], // StateMaps
    sm_n: [[u32; 256]; NICM],
    // mixers
    mixers: Vec<[f64; NW]>,
    mixers_g: Vec<[f64; NW]>,
    mixers2: Vec<[f64; NW]>,
    mixers2_g: Vec<[f64; NW]>,
    gmix: [f64; NW],
    gmix_g: [f64; NW],
    final_w: [f64; 4],
    final_g: [f64; 4],
    apm: [Apm; 6],
    mm: MatchModel,
    mm2: MatchModel,
    hist: History,
    // §82 selector
    sl2: [u64; SELSMAX],
    selvt: Vec<[f64; 2]>,
    selvtg: Vec<u8>,
    selut: Vec<f64>,
    sel_nc: usize,
    sel_gk: u64,
    sel_t: [usize; SEL_TOPM],
    sel_nt: usize,
    sel_wt: [f64; SELSMAX],
    sel_ix: [usize; SELSMAX],
    sel_p1: [f64; SELSMAX],
    sel_ll: [f64; SELSMAX],
    sel_uw: f64,
    sel_active: bool,
    // §79 prefix vectors
    pv_tab: Vec<[f64; PVD]>,
    pv_tag: Vec<u64>,
    pv_id: u64,
    pv_ti: usize,
    pv_feat: [f64; PVD],
    pv_grad: [f64; PVD],
    // stream state
    cur: u64,
    phase: usize,
    prev_byte: u64,
    prev2: u64,
    word_hash: u64,
    prev_word_hash: u64,
    byte_pos: u64,
    htail: u64,
    hbase: [u64; NH],
    // per-bit caches (set by p(), consumed by update())
    sts: [f64; NW],
    oslot: [usize; NM],
    sp_slot: [usize; NSP],
    wd_slot: usize,
    wd2_slot: usize,
    hslot: [usize; NH],
    icm_bv: [usize; NICM],
    icm_ti: [usize; NICM],
    sel: usize,
    sel2: usize,
    p_sel: f64,
    p_sel2: f64,
    p_g: f64,
    p_mix: f64,
    ssel: f64,
    ssel2: f64,
    sg: f64,
    pending: bool,
}

impl Model {
    /// # Panics
    /// If `params` fails `Params::validate` (check it first for untrusted values).
    pub fn new(params: Params) -> Model {
        params.validate().expect("invalid model parameters");
        let o = 1usize << params.obits;
        let apm = [
            Apm::new(256 * 8, 0.007),
            Apm::new(1024, 0.005),
            Apm::new(256 * 8, 0.006),
            Apm::new(256, 0.005),
            Apm::new(4096, 0.005),
            Apm::new(4096, 0.005),
        ];
        Model {
            params,
            st: stretch_table(),
            ord: (0..NM).map(|_| vec![[0u8; 3]; o]).collect(),
            sp: (0..NSP).map(|_| vec![[0u8; 3]; o]).collect(),
            wd: vec![[0u8; 3]; o],
            wd2: vec![[0u8; 3]; o],
            high: (0..NH).map(|_| vec![[0u8; 2]; 1usize << params.hbits]).collect(),
            icm_bh: (0..NICM).map(|_| vec![0u8; o]).collect(),
            sm_p: [[0.5f64; 256]; NICM],
            sm_n: [[0u32; 256]; NICM],
            mixers: vec![[0.0f64; NW]; NSEL],
            mixers_g: vec![[0.0f64; NW]; NSEL],
            mixers2: vec![[0.0f64; NW]; NSEL2],
            mixers2_g: vec![[0.0f64; NW]; NSEL2],
            gmix: [0.0; NW],
            gmix_g: [0.0; NW],
            final_w: [0.3, 0.3, 0.2, 0.0],
            final_g: [0.0; 4],
            apm,
            mm: MatchModel::new(params.mbits, 5),
            mm2: MatchModel::new(params.mbits, 8),
            hist: History::new(params.window_log),
            sl2: [0; SELSMAX],
            selvt: vec![[0.0f64; 2]; 1usize << params.selvbits],
            selvtg: vec![0u8; 1usize << params.selvbits],
            selut: vec![0.0f64; 1usize << params.selubits],
            sel_nc: 0,
            sel_gk: 0,
            sel_t: [0; SEL_TOPM],
            sel_nt: 0,
            sel_wt: [0.0; SELSMAX],
            sel_ix: [0; SELSMAX],
            sel_p1: [0.0; SELSMAX],
            sel_ll: [0.0; SELSMAX],
            sel_uw: 0.0,
            sel_active: false,
            pv_tab: vec![[0.0f64; PVD]; 1usize << params.sbits],
            pv_tag: vec![0u64; 1usize << params.sbits],
            pv_id: 0,
            pv_ti: 0,
            pv_feat: [0.0; PVD],
            pv_grad: [0.0; PVD],
            cur: 0,
            phase: 0,
            prev_byte: 0,
            prev2: 0,
            word_hash: 0,
            prev_word_hash: 0,
            byte_pos: 0,
            htail: 0,
            hbase: [0; NH],
            sts: [0.0; NW],
            oslot: [0; NM],
            sp_slot: [0; NSP],
            wd_slot: 0,
            wd2_slot: 0,
            hslot: [0; NH],
            icm_bv: [0; NICM],
            icm_ti: [0; NICM],
            sel: 0,
            sel2: 0,
            p_sel: 0.5,
            p_sel2: 0.5,
            p_g: 0.5,
            p_mix: 0.5,
            ssel: 0.0,
            ssel2: 0.0,
            sg: 0.0,
            pending: false,
        }
    }

    pub fn params(&self) -> Params {
        self.params
    }

    /// Bytes of input seen so far.
    pub fn position(&self) -> u64 {
        self.byte_pos
    }

    /// Issue cache prefetches for the slots `p()` will read at (phase, cur). Mirrors the key
    /// formulas in `p()`; a mismatch here would cost speed only, never correctness.
    #[inline]
    fn prefetch_bit(&self, phase: usize, cur: u64) {
        let bp = self.byte_pos;
        for k in 0..NM {
            let l = if bp >= k as u64 { k } else { bp as usize };
            let m = if 8 * l >= 64 { u64::MAX } else { (1u64 << (8 * l)) - 1 };
            let key = (((1u64 << (8 * l + phase)) | ((self.htail & m) << phase) | cur) << 3) | phase as u64;
            let ti = self.slot_index(key).0;
            prefetch(&self.ord[k][ti]);
            if (2..=6).contains(&k) {
                prefetch(&self.icm_bh[k - 2][ti]);
            }
        }
        let hmask = (1u64 << self.params.hbits) - 1;
        for hk in 0..NH {
            let slot = ((self.hbase[hk].wrapping_mul(2654435761) ^ (phase as u64).wrapping_mul(0x9E37_79B1) ^ cur.wrapping_mul(2246822519))
                & hmask) as usize;
            prefetch(&self.high[hk][slot]);
        }
        for j in 0..NSP {
            let (oa, ob) = SPOFF[j];
            let ba = if bp >= oa as u64 { self.hist.at(bp - oa as u64) as u64 } else { 0 };
            let bb = if bp >= ob as u64 { self.hist.at(bp - ob as u64) as u64 } else { 0 };
            let sk = (((((((1u64 << phase) | cur) << 8) | ba) << 8 | bb) << 5) | ((j as u64) << 3)) | (phase as u64);
            prefetch(&self.sp[j][self.slot_index(sk).0]);
        }
        let low = (((1u64 << phase) | cur) << 3) | (phase as u64);
        prefetch(&self.wd[self.slot_index((self.word_hash << 12) | low).0]);
        let wctx = self
            .prev_word_hash
            .wrapping_mul(0x9E37_79B1)
            .wrapping_add(self.word_hash.wrapping_mul(2654435761));
        prefetch(&self.wd2[self.slot_index((wctx << 12) | low).0]);
        if self.sel_active {
            let ctx3 = self.htail & SELCTXMASK;
            let vbits = self.params.selvbits as u32;
            for j in 0..self.sel_nc {
                let h = self.sl2[j].wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ ctx3.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                    ^ (((phase << 7) | cur as usize) as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
                let ix = (sel_mix(h) >> (64 - vbits)) as usize;
                prefetch(&self.selvt[ix]);
                prefetch(&self.selvtg[ix]);
            }
        }
    }

    #[inline]
    fn slot_index(&self, key: u64) -> (usize, u8) {
        let obits = self.params.obits as u32;
        let h = key.wrapping_mul(MULT);
        ((h >> (64 - obits)) as usize, ((h >> (64 - obits - 8)) & 0xFF) as u8)
    }

    /// Probability that the next bit is 1, in [1e-6, 1 - 1e-6] (NaN-free by construction of the
    /// clamps; the coder still sanitises).
    pub fn p(&mut self) -> f64 {
        debug_assert!(!self.pending, "p() called twice without update()");
        self.pending = true;
        let phase = self.phase;
        let cur = self.cur;
        let prefix = cur;
        let bp = self.byte_pos;
        let mut oreset = [false; NM];

        // byte-aware orders 0..7
        let maskb = |l: usize| -> u64 {
            if 8 * l >= 64 {
                u64::MAX
            } else {
                (1u64 << (8 * l)) - 1
            }
        };
        for k in 0..NM {
            let l = if bp >= k as u64 { k } else { bp as usize };
            let run_ = ((self.htail & maskb(l)) << phase) | cur;
            let key = (((1u64 << (8 * l + phase)) | run_) << 3) | (phase as u64);
            let (ti, want) = self.slot_index(key);
            let s = &mut self.ord[k][ti];
            oreset[k] = s[0] != want;
            if oreset[k] {
                *s = [want, 0, 0];
            }
            self.oslot[k] = ti;
            self.sts[k] = st_count(self.st, s[1], s[2]);
        }
        // hashed high orders (merged, untagged)
        let hmask = (1u64 << self.params.hbits) - 1;
        for hk in 0..NH {
            let hv = self.hbase[hk];
            let slot = ((hv.wrapping_mul(2654435761) ^ (phase as u64).wrapping_mul(0x9E37_79B1) ^ prefix.wrapping_mul(2246822519)) & hmask)
                as usize;
            self.hslot[hk] = slot;
            let c = self.high[hk][slot];
            self.sts[NM + hk] = st_count(self.st, c[0], c[1]);
        }
        // sparse byte pairs
        for j in 0..NSP {
            let (oa, ob) = SPOFF[j];
            let ba = if bp >= oa as u64 { self.hist.at(bp - oa as u64) as u64 } else { 0 };
            let bb = if bp >= ob as u64 { self.hist.at(bp - ob as u64) as u64 } else { 0 };
            let sk = (((((((1u64 << phase) | cur) << 8) | ba) << 8 | bb) << 5) | ((j as u64) << 3)) | (phase as u64);
            let (ti, want) = self.slot_index(sk);
            let s = &mut self.sp[j][ti];
            if s[0] != want {
                *s = [want, 0, 0];
            }
            self.sp_slot[j] = ti;
            self.sts[NM + NH + j] = st_count(self.st, s[1], s[2]);
        }
        // word and previous-word
        {
            let wk = (self.word_hash << 12) | ((((1u64 << phase) | cur) << 3) | (phase as u64));
            let (ti, want) = self.slot_index(wk);
            let s = &mut self.wd[ti];
            if s[0] != want {
                *s = [want, 0, 0];
            }
            self.wd_slot = ti;
            self.sts[NM + NH + NSP] = st_count(self.st, s[1], s[2]);
        }
        {
            let wctx = self
                .prev_word_hash
                .wrapping_mul(0x9E37_79B1)
                .wrapping_add(self.word_hash.wrapping_mul(2654435761));
            let wk2 = (wctx << 12) | ((((1u64 << phase) | cur) << 3) | (phase as u64));
            let (ti, want) = self.slot_index(wk2);
            let s = &mut self.wd2[ti];
            if s[0] != want {
                *s = [want, 0, 0];
            }
            self.wd2_slot = ti;
            self.sts[NM + NH + NSP + 1] = st_count(self.st, s[1], s[2]);
        }
        self.sts[NM + NH + NSP + 2] = self.mm.predicted(&self.hist, phase, bp);
        self.sts[NM + NH + NSP + 3] = self.mm2.predicted(&self.hist, phase, bp);
        // indirect models
        for c in 0..NICM {
            let ti = self.oslot[ICM_K[c]];
            if oreset[ICM_K[c]] {
                self.icm_bh[c][ti] = 0;
            }
            let bv = self.icm_bh[c][ti] as usize;
            self.icm_bv[c] = bv;
            self.icm_ti[c] = ti;
            self.sts[NM + NH + NSP + 4 + c] = stretch(self.sm_p[c][bv]);
        }
        // prefix vector (BLPVEC=2)
        for k in 0..PVD {
            self.sts[NIN + k] = if self.pv_id != 0 { self.pv_feat[k] } else { 0.0 };
        }
        self.sts[I_DEAD] = 0.0;
        // selector vote (BLSEL=1, tagged cells)
        if self.sel_active && self.sel_nc > 0 {
            let ctx3 = self.htail & SELCTXMASK;
            let vbits = self.params.selvbits as u32;
            let vmask = (1usize << vbits) - 1;
            for j in 0..self.sel_nc {
                let h = self.sl2[j].wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ ctx3.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                    ^ (((phase << 7) | cur as usize) as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
                let hm = sel_mix(h);
                let ix = ((hm >> (64 - vbits)) as usize) & vmask;
                let want = (((hm >> (64 - vbits - 8)) & 0xFF) as u8) & SELTAGMASK;
                if self.selvtg[ix] != want {
                    self.selvtg[ix] = want;
                    self.selvt[ix] = [0.0, 0.0];
                }
                self.sel_ix[j] = ix;
                let [n0, n1] = self.selvt[ix];
                self.sel_p1[j] = (n1 + 0.2) / (n0 + n1 + 0.4);
            }
            let mut pm = 0.0f64;
            for t in 0..self.sel_nt {
                pm += self.sel_wt[t] * self.sel_p1[self.sel_t[t]];
            }
            let f = stretch(pm);
            self.sel_uw += self.gmix[I_SEL].abs();
            self.sts[I_SEL] = f;
        } else {
            self.sts[I_SEL] = 0.0;
        }

        // mixers
        let pb = self.prev_byte;
        self.sel = ((phase << 8) | pb as usize) & (NSEL - 1);
        self.sel2 = ((phase << 11) | ((pb.wrapping_mul(769) ^ self.prev2.wrapping_mul(2246822519)) as usize & 2047)) & (NSEL2 - 1);
        // three dot products in one loop: each accumulator keeps strong.rs's exact summation
        // order, but the three dependency chains overlap in the pipeline
        let sts = &self.sts;
        let (w1, w2, wg) = (&self.mixers[self.sel], &self.mixers2[self.sel2], &self.gmix);
        let (mut d, mut d2, mut dg) = (0.0f64, 0.0f64, 0.0f64);
        for k in 0..NW {
            d += w1[k] * sts[k];
            d2 += w2[k] * sts[k];
            dg += wg[k] * sts[k];
        }
        self.p_sel = squash(d);
        self.p_sel2 = squash(d2);
        self.p_g = squash(dg);
        self.ssel = stretch(self.p_sel);
        self.ssel2 = stretch(self.p_sel2);
        self.sg = stretch(self.p_g);
        let fw = &self.final_w;
        self.p_mix = squash(fw[0] * self.ssel + fw[1] * self.sg + fw[2] * self.ssel2 + fw[3]);

        // APM/SSE chain
        let p_mix = self.p_mix;
        let ph = phase as u64;
        let pa0 = self.apm[0].refine(p_mix, ((pb << 3) | ph) as usize);
        let pa = 0.3 * p_mix + 0.7 * pa0;
        let pb0 = self.apm[1].refine(pa, ((pb.wrapping_mul(769) + self.prev2.wrapping_mul(31) + ph) & 1023) as usize);
        let mut p = 0.3 * pa + 0.7 * pb0;
        let pc0 = self.apm[2].refine(p, ((cur << 3) | ph) as usize);
        p = 0.3 * p + 0.7 * pc0;
        let pd0 = self.apm[3].refine(p, (self.mm.len.min(31) << 3) | phase);
        p = 0.3 * p + 0.7 * pd0;
        let pe0 = self.apm[4].refine(p, (((self.word_hash & 511) << 3) | ph) as usize);
        p = 0.3 * p + 0.7 * pe0;
        let pf0 = self.apm[5].refine(p, ((((self.htail ^ (self.htail >> 13)) & 511) << 3) | ph) as usize);
        p = 0.3 * p + 0.7 * pf0;
        p.clamp(1e-6, 1.0 - 1e-6)
    }

    /// Learn from the actual bit (0 or 1). Must follow exactly one `p()`.
    pub fn update(&mut self, y: u32) {
        debug_assert!(self.pending, "update() without p()");
        self.pending = false;
        let y = (y & 1) as usize;
        let yf = y as f64;
        if self.phase < 7 {
            // the next bit's table slots are known now: start their cache misses before the
            // mixer updates below (a hint only; never changes a value)
            self.prefetch_bit(self.phase + 1, (self.cur << 1) | y as u64);
        }

        // final mixer
        let em = yf - self.p_mix;
        let gf = [em * self.ssel, em * self.sg, em * self.ssel2, em];
        let ord_ = 1.0 - RMS_DECAY;
        for i in 0..4 {
            self.final_g[i] = RMS_DECAY * self.final_g[i] + ord_ * gf[i] * gf[i];
        }
        for i in 0..4 {
            self.final_w[i] += ALRF * gf[i] / (self.final_g[i].sqrt() + RMS_EPS);
        }
        let e_sel = yf - self.p_sel;
        let e_sel2 = yf - self.p_sel2;
        let e_g = yf - self.p_g;
        // prefix-vector credit, PRE-update weights
        if self.pv_id != 0 {
            for k in 0..PVD {
                self.pv_grad[k] +=
                    e_sel * self.mixers[self.sel][NIN + k] + e_sel2 * self.mixers2[self.sel2][NIN + k] + e_g * self.gmix[NIN + k];
            }
        }
        let sts = &self.sts;
        rms_update(&mut self.mixers[self.sel], &mut self.mixers_g[self.sel], sts, e_sel, ord_);
        rms_update(&mut self.mixers2[self.sel2], &mut self.mixers2_g[self.sel2], sts, e_sel2, ord_);
        rms_update(&mut self.gmix, &mut self.gmix_g, sts, e_g, ord_);

        // selector cells for ALL candidates + per-candidate log-likelihood
        if self.sel_active && self.sel_nc > 0 {
            for j in 0..self.sel_nc {
                let pr = if y == 1 { self.sel_p1[j] } else { 1.0 - self.sel_p1[j] };
                self.sel_ll[j] += log2(pr.max(1e-9));
                let cell = &mut self.selvt[self.sel_ix[j]];
                let cy = cell[y] + 1.0;
                if cy + cell[1 - y] >= 255.0 {
                    cell[y] = cy * 0.5;
                    cell[1 - y] *= 0.5;
                } else {
                    cell[y] = cy;
                }
            }
        }
        // counts
        for k in 0..NM {
            let s = &mut self.ord[k][self.oslot[k]];
            bump(s, y);
        }
        for c in 0..NICM {
            let bv = self.icm_bv[c];
            let pr = self.sm_p[c][bv];
            self.sm_p[c][bv] = pr + (yf - pr) / (self.sm_n[c][bv] as f64 + 1.5);
            if self.sm_n[c][bv] < 1023 {
                self.sm_n[c][bv] += 1;
            }
            self.icm_bh[c][self.icm_ti[c]] = (((bv << 1) | y) & 0xFF) as u8;
        }
        for j in 0..NSP {
            let s = &mut self.sp[j][self.sp_slot[j]];
            bump(s, y);
        }
        {
            let s = &mut self.wd[self.wd_slot];
            bump(s, y);
            ns_discount(s, y);
            let s = &mut self.wd2[self.wd2_slot];
            bump(s, y);
            ns_discount(s, y);
        }
        for hk in 0..NH {
            let c = &mut self.high[hk][self.hslot[hk]];
            bump_pair(c, y);
        }
        for a in self.apm.iter_mut() {
            a.update(yf);
        }

        self.cur = (self.cur << 1) | y as u64;
        self.phase += 1;
        if self.phase == 8 {
            self.end_of_byte();
        }
    }

    fn end_of_byte(&mut self) {
        // prefix vector: apply the byte's accumulated credit (v += LR_S * g, clipped)
        if self.pv_id != 0 {
            let v = &mut self.pv_tab[self.pv_ti];
            for k in 0..PVD {
                let e = v[k] + LR_S * self.pv_grad[k];
                v[k] = e.clamp(-1.0, 1.0);
                self.pv_grad[k] = 0.0;
            }
            self.pv_feat = *v;
        }
        let b = (self.cur & 0xFF) as u8;
        let bp = self.byte_pos;
        self.hist.push(bp, b);
        self.mm.update_after_byte(&self.hist, bp);
        self.mm2.update_after_byte(&self.hist, bp);
        self.htail = ((self.htail << 8) | b as u64) & ((1u64 << (8 * MAXB)) - 1);

        // selector usefulness for the byte just served
        if self.sel_active && self.sel_nc > 0 {
            let mut mu = 0.0f64;
            for j in 0..self.sel_nc {
                mu += self.sel_ll[j];
            }
            mu /= self.sel_nc as f64;
            let mut wu = self.sel_uw / 8.0;
            if wu > 1.0 {
                wu = 1.0;
            }
            if wu > 0.02 {
                let ubits = self.params.selubits as u32;
                let umask = (1usize << ubits) - 1;
                for j in 0..self.sel_nc {
                    let adv = (self.sel_ll[j] - mu).clamp(-1.0, 1.0);
                    let h = sel_mix(self.sel_gk ^ self.sl2[j].wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let ui = ((h >> (64 - ubits)) as usize) & umask;
                    let u = self.selut[ui] * SEL_UDC + (1.0 - SEL_UDC) * wu * adv;
                    self.selut[ui] = u.clamp(-SEL_UCLIP, SEL_UCLIP);
                }
            }
        }
        self.sel_ll = [0.0; SELSMAX];
        self.sel_uw = 0.0;

        // words: letters extend the hash; a non-letter completes the word
        if is_letter(b) {
            self.word_hash = (self.word_hash.wrapping_mul(131) + ((b | 0x20) as u64)) & 0xFFF_FFFF;
        } else {
            if self.word_hash != 0 {
                self.prev_word_hash = self.word_hash;
                // whole-stream LRU of completed words (depth NSELSLOTS)
                let wid = self.word_hash;
                let nc = self.sel_nc;
                let pos = (0..nc).find(|&j| self.sl2[j] == wid).unwrap_or(nc);
                if pos < nc {
                    self.sl2.copy_within(0..pos, 1);
                } else if nc < NSELSLOTS {
                    self.sl2.copy_within(0..nc, 1);
                    self.sel_nc += 1;
                } else {
                    self.sl2.copy_within(0..NSELSLOTS - 1, 1);
                }
                self.sl2[0] = wid;
            }
            self.word_hash = 0;
        }

        // selector forward: a query-free gate over the current slots serves the next byte
        let ctx3 = self.htail & SELCTXMASK;
        self.sel_gk = ctx3;
        if self.sel_nc > 0 {
            let ubits = self.params.selubits as u32;
            let umask = (1usize << ubits) - 1;
            let nc = self.sel_nc;
            let mut e = [0.0f64; SELSMAX];
            let mut mx = -1e30f64;
            for j in 0..nc {
                let h = sel_mix(ctx3 ^ self.sl2[j].wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let ui = ((h >> (64 - ubits)) as usize) & umask;
                let ev = (SEL_UGAIN * self.selut[ui] - SEL_PBD * j as f64).clamp(-12.0, 12.0);
                e[j] = ev;
                if ev > mx {
                    mx = ev;
                }
            }
            let mut z = 0.0f64;
            let mut a = [0.0f64; SELSMAX];
            for j in 0..nc {
                let ex = exp((e[j] - mx) / SEL_TSC);
                a[j] = ex;
                z += ex;
            }
            for j in 0..nc {
                a[j] /= z;
            }
            self.sel_nt = SELTOPM.min(nc);
            let mut taken = [false; SELSMAX];
            let mut at = 0.0f64;
            for t in 0..self.sel_nt {
                let mut best = 0usize;
                let mut bv = -1.0f64;
                for j in 0..nc {
                    if !taken[j] && a[j] > bv {
                        bv = a[j];
                        best = j;
                    }
                }
                taken[best] = true;
                self.sel_t[t] = best;
                at += bv;
            }
            if at <= 0.0 {
                at = 1.0;
            }
            for t in 0..self.sel_nt {
                self.sel_wt[t] = a[self.sel_t[t]] / at;
            }
            self.sel_active = true;
        } else {
            self.sel_nt = 0;
            self.sel_active = false;
        }

        // prefix key: a letter moves to the new letters-only prefix; a non-letter keeps it
        if is_letter(b) {
            let id = self.word_hash | (1u64 << 40);
            let sbits = self.params.sbits as u32;
            let ti = (id.wrapping_mul(MULT) >> (64 - sbits)) as usize;
            if self.pv_tag[ti] != id {
                self.pv_tag[ti] = id;
                let mut sd = id;
                let mut v = [0.0f64; PVD];
                for x in v.iter_mut() {
                    *x = (splitmix(&mut sd) as f64 / u64::MAX as f64 - 0.5) * 0.1;
                }
                self.pv_tab[ti] = v;
            }
            self.pv_id = id;
            self.pv_ti = ti;
            self.pv_feat = self.pv_tab[ti];
        }

        self.prev2 = self.prev_byte;
        self.prev_byte = b as u64;
        self.cur = 0;
        self.phase = 0;
        self.byte_pos += 1;
        let bp = self.byte_pos;
        for hk in 0..NH {
            let lo = bp.saturating_sub(HORDERS[hk] as u64);
            let mut hv: u64 = 1469598103934665603;
            for q in lo..bp {
                hv = (hv ^ self.hist.at(q) as u64).wrapping_mul(1099511628211);
            }
            self.hbase[hk] = hv;
        }
    }
}

#[inline(always)]
fn prefetch<T>(r: &T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetch is a hint; any address is allowed and nothing is dereferenced.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(r as *const T as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = r;
}

#[inline]
fn rms_update(w: &mut [f64; NW], wg: &mut [f64; NW], sts: &[f64; NW], e: f64, ord_: f64) {
    for k in 0..NW {
        let gk = e * sts[k];
        wg[k] = RMS_DECAY * wg[k] + ord_ * gk * gk;
        w[k] += ALR * gk / (wg[k].sqrt() + RMS_EPS);
    }
}
