//! blmrs strong — the strong bit-native context-mixing predictor (extends the `mixnsfast.py` port).
//!
//! Models: byte-orders 0..7, hashed high orders {8,12,16,24,32}, 3 sparse models (non-adjacent byte
//! pairs), a word model + a previous-word model (text bigrams), two byte-match models (min len 5/8),
//! and 5 INDIRECT context models (orders 2..6): each maps an order-K context's bit-HISTORY byte to an
//! adaptive StateMap, a nonstationary prediction the raw counts miss (the ISSE/ICM idea). These feed
//! TWO context-selected logistic mixers (selected by order-1 and by order-2) plus a global mixer,
//! combined by a 4-weight final mixer, then a chain of 6 SSE/APM stages (prev-byte, order-2,
//! partial-byte, match-length, word-hash, order-3 contexts). Non-stationary count recency-halving +
//! RMSProp adaptive learning. Orders/sparse/word use flat fixed-size open-addressing tables (high-bit
//! multiplicative hash + 8-bit checksum tags) -> bounded memory; the high orders use merged hashed tables.
//!
//! History: started as a faithful port of `mixnsfast.py` (verified bit-equal at small scale), then
//! improved — order-7, a 2nd (order-2) mixer, +2 SSE stages, +2 sparse models, the previous-word
//! model, tuned DELTA/learning-rates, and the indirect (ICM/StateMap) models — for ~3.4% lower bits/bit
//! on real text (measured 11 MB, see learned_binary_address_machine.md §63). DELTA/ALR/ALRF are
//! env-overridable. Usage: strong <path> [byte_cap] [obits].
//!
//! §72 WORD-SLOT CHANNEL (env BLMSLOTS=1, default OFF = bit-identical baseline): an LRU of the
//! last SLOTS=6 COMPLETED words, each bound to its own SD=8-dim vector learned ONLINE by the
//! compression loss (wstate.py: the EMA superposition cannot bind word identity — no readout of it
//! can — but slots isolate it; learned vectors beat frozen identity vectors on copy-ablated,
//! decontaminated held-out, crossing below the order-only baseline at >=1.2MB). The SLOTS*SD slot
//! dims feed a dedicated head added into the GLOBAL mixer's input (before squash), trained by the
//! same e_g signal; per-bit credit slot_vec[k] += (y-p_g)*head_w[k] accumulates per byte and is
//! applied to the word's vector at byte end (LR_S). Vectors live in a tagged flat table keyed by
//! word hash (deterministic splitmix init), evict-on-collision like every other table.
//! §77 SIGN FIX: until §77 the code applied v -= LR_S*g with g = sum (y-p_g)*w (gradient ASCENT), so
//! every §72/§74/§75 BLMSLOTS/BLSOFT A/B ran uphill-trained vectors. Fixed to v += LR_S*g.
//! §77 ALSO CORRECTS THE TEXT ABOVE: the wstate.py "crossing" cited in the §72 paragraph came from an LRU of word
//! PREFIXES (a learned current-word model), not completed words -- whole-word slots lose there -- and §73's
//! hard-key result is unsupported (the bucketed vectors were per-prefix predictors). See ledger §77.5.
//!
//! §74 SOFT RETRIEVAL (env BLSOFT=1, implies the §72 machinery): §73 proved HARD similarity keys
//! (sign buckets as count-table keys) fail — quantisation destroys the graded information. The
//! soft readout instead: the §72 vectors are indexed in an LSH structure (sign bucket + the 8
//! one-bit-flip neighbours, capped lists); once per byte the last completed word retrieves its
//! top-SOFTK neighbours by NORMALISED dot (>= SIMMIN); per bit, the neighbours' OWN word-model
//! counts (the existing wdc tables, same keys) are voted similarity-weighted into a single
//! stretch() head on the global mixer. This is "predict what follows words SIMILAR to the last
//! word" — generalisation across the word tail that identity keys cannot give. Port rule (§72):
//! earns its place only by beating strong's own 11MB baseline (0.217011).
//!
//! §77 ABSORBER ABLATION (env BLSTRIPW=1 / BLSTRIPH=1, both default OFF = bit-identical baseline):
//! §75 named strong's word-keyed family as the absorber of the §72 slot signal; these flags remove
//! it so the slot channel can be A/B'd against a strong WITHOUT it. A stripped model's mixer input
//! (its stretch) is exactly 0.0 every bit, so its weights in all three mixers get zero gradient and
//! stay 0, and its tables are neither read nor updated. BLSTRIPW=1 strips the word model (wdc) and
//! the previous-word model (wdc2), AND the one place word_hash is used as a CONTEXT rather than a
//! model input: the APM5 SSE stage, whose context (word_hash&511, phase) collapses to (0, phase) --
//! the stage still runs (same chain shape) but carries no word identity. word_hash itself is still
//! computed because the §72 slot LRU needs it (BLMSLOTS=1 works with either flag); prev_word_hash
//! has no other consumer. BLSTRIPH=1 strips the hashed high orders HORDERS {8,12,16,24,32} (htab;
//! hbase is not even computed). Nothing else reads word_hash/prev_word_hash/wdc/wdc2/htab/hbase:
//! the ICMs index the ORDERS 2..6 slots, the sparse models hash raw byte pairs, the match models
//! keep their own rolling hash, mixer selection uses prev_byte/prev2, APM6's order-3 context comes
//! from htail. BLSOFT=1 votes the wdc tables, so BLSOFT=1 with BLSTRIPW=1 is undefined and exits
//! with an error. The active flags are printed on a `flags:` line under the header.
//! §77 KNOWN DEFECT (not fixed; track closed): the BLSOFT vote reads wdc[(neighbour<<12)|partial], the
//! word-model statistic for the byte AFTER the neighbour's letters (its delimiter), and applies it to
//! the letters of the next word -- inert by construction. Result with the sign fix (§77.3): whole-word
//! slots add -0.000005 bits/bit on corpus_big and +0.00004 on stdlib, with or without BLSTRIPW/H.
//!
//! §79 PER-PREFIX VECTOR + NONSTATIONARY COUNTS (env BLPVEC / BLWNS / BLNSALL, all default OFF =
//! bit-identical baseline). BLPVEC=2 ports the rr78 scratch mode 2 (which beat both ~11 MB baselines (corpus_big 11 MB, stdlib.bin 10.6 MB):
//! corpus_big 0.217011 -> 0.216509, stdlib 0.127624 -> 0.127204; frozen LR_S=0 only 0.216965 /
//! 0.127591): one PVD=8-dim vector per CURRENT WORD PREFIX, key = word_hash of the letters-only prefix
//! (tagged id word_hash|1<<40), set on every letter and KEPT after the word ends until the next letter
//! (the instrument's prefix slot 0). Vectors live in a 2^SBITS tagged table, evict-on-collision, with
//! the §72 splitmix init. The 8 dims are ordinary inputs to ALL THREE mixers: under BLPVEC=2 the active
//! mixer width grows from NW to NIN+PVD+1 with the bias moved after the 8 dims (the scratch layout, so
//! the float summation order matches it); otherwise the loops run over the original NW and the dims do
//! not exist. Per bit, credit g[k] += e_sel*w_sel[k] + e_sel2*w_sel2[k] + e_g*w_g[k] with PRE-update
//! weights; at byte end v += LR_S*g, clipped to [-1,1] (LR_S=0 = frozen random-vector control).
//! BLPVEC=1 is the scratch's global-mixer-only head (§72 placement, own RMSProp head at ALRS), kept for
//! the record: it was nearly inert. A red-team found the instrument-side gain of such a vector is mostly
//! RECENCY adaptation that cumulative counts lack; strong only halves both counts at CLIMIT. So:
//! BLWNS=1 applies the PAQ nonstationary update to the word and previous-word tables (wdc, wdc2): on a
//! bump with bit y, n_y += 1 with the CLIMIT halving exactly as before, THEN if n_{1-y} > 2,
//! n_{1-y} = n_{1-y}/2 + 1. BLNSALL=1 (exploratory) applies the same rule to every count table: orders
//! 0..7, sparse, word, word2 and the hashed high orders (not the ICM StateMap counts). BLPVEC>2 exits.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Instant;

const ORDERS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
const HORDERS: [usize; 5] = [8, 12, 16, 24, 32];
const NH: usize = 5;
const NM: usize = 8;
const NSP: usize = 3; // sparse models over non-adjacent byte pairs (offsets below)
const SPOFF: [(usize, usize); NSP] = [(2, 3), (1, 4), (3, 6)];
const NICM: usize = 5;               // indirect context models (bit-history -> adaptive StateMap)
const ICM_K: [usize; NICM] = [2, 3, 4, 5, 6]; // which ORDERS positions get an ICM
const NIN: usize = NM + NH + NSP + 4 + NICM; // [orders][hi][sparse..][word][word2][match][match2][icm..] bias
const NW: usize = NIN + 1; // mixer weights (inputs + bias)
const PVD: usize = 8; // §79 per-prefix vector dims (BLPVEC)
const NWMAX: usize = NIN + PVD + 1; // §79 storage width; ACTIVE width nw = NW, or NIN+PVD+1 under BLPVEC=2
const SELW: usize = NWMAX + 1;      // §82 storage width with BLSEL (one more input: the selector vote)

// ---- §82 the lean selector port (wstate.py v10 §81; env BLSEL, default OFF = bit-identical) ----
const SELSMAX: usize = 64;   // max word-slot depth (active = NSELSLOTS, default 32)
const SEL_TOPM: usize = 8;    // MAX vote-set size (active = env SELTOPM, default 4)
const SEL_UGAIN: f64 = 4.0;  // gate gain on usefulness
const SEL_PBD: f64 = 0.05;   // position (recency) bias per slot step
const SEL_TSC: f64 = 2.0;    // softmax temperature
const SEL_UDC: f64 = 0.995;  // usefulness EWMA decay
const SEL_UCLIP: f64 = 4.0;  // usefulness clamp

#[inline]
fn sel_mix(mut h: u64) -> u64 {
    h ^= h >> 33; h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29; h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 32)
}
const NSEL: usize = 8 * 256;
const HBITS: u32 = 22;
const CLIMIT: u32 = 255;
const MAXB: usize = 7;
const MULT: u64 = 0x9E37_79B9_7F4A_7C15;

const RMS_DECAY: f64 = 0.9999;
const RMS_EPS: f64 = 1e-4;

#[inline]
fn envf(name: &str, default: f64) -> f64 {
    env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

// ---- §72 word-slot channel ----
const SLOTS: usize = 64;              // MAX LRU depth (active depth = env NSLOTS, default 6;
                                      // deep slots reach BEYOND the hashed order-32 contexts)
const SD: usize = 8;                  // vector dims per word

#[inline]
fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn bucket_of(v: &[f64; SD]) -> u8 {
    let mut b = 0u8;
    for k in 0..SD { b = (b << 1) | (v[k] > 0.0) as u8; }
    b
}

/// §74: keep the LSH index current for a word whose vector exists (sign bucket, capped lists).
fn index_update(w: u64, sbits: u32, semb: &[[f64; SD]], setag: &[u64],
                bucket_vec: &mut [Vec<u64>], word_bucket: &mut HashMap<u64, u8>) {
    let ti = (w.wrapping_mul(MULT) >> (64 - sbits)) as usize;
    if setag[ti] != w { return; }                       // no vector resident
    let b = bucket_of(&semb[ti]) as usize;
    let unchanged = matches!(word_bucket.get(&w), Some(&old) if old as usize == b);
    if !unchanged {
        if let Some(&old) = word_bucket.get(&w) {
            bucket_vec[old as usize].retain(|&x| x != w);
        }
        let bv = &mut bucket_vec[b];
        if !bv.contains(&w) {
            if bv.len() >= 64 { bv.remove(0); }
            bv.push(w);
        }
        word_bucket.insert(w, b as u8);
    }
}

/// §79 PAQ nonstationary rule: after a bump of count pair c[s..s+2] with bit yi (CLIMIT halving
/// already applied), discount the OPPOSITE count: if n_{1-y} > 2 then n_{1-y} = n_{1-y}/2 + 1.
#[inline]
fn ns_discount(c: &mut [u32], s: usize, yi: usize) {
    let o = s + 1 - yi;
    if c[o] > 2 { c[o] = c[o] / 2 + 1; }
}

#[inline]
fn stretch(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}
#[inline]
fn squash(t: f64) -> f64 {
    if t > 30.0 { 1.0 - 1e-6 } else if t < -30.0 { 1e-6 } else { 1.0 / (1.0 + (-t).exp()) }
}

/// Adaptive Probability Map (SSE) — flat knot table over the stretch domain, per context.
struct Apm {
    k: usize,
    smax: f64,
    rate: f64,
    step: f64,
    t: Vec<f64>, // n_ctx * k
    lo: usize,
    w: f64,
    c: usize,
}
impl Apm {
    fn new(n_ctx: usize, k: usize, rate: f64) -> Self {
        let smax = 8.0;
        let step = 2.0 * smax / (k as f64 - 1.0);
        let mut t = vec![0.0f64; n_ctx * k];
        for j in 0..k {
            let v = squash(-smax + j as f64 * step);
            for c in 0..n_ctx {
                t[c * k + j] = v;
            }
        }
        Apm { k, smax, rate, step, t, lo: 0, w: 0.0, c: 0 }
    }
    #[inline]
    fn refine(&mut self, p: f64, cx: usize) -> f64 {
        let s = stretch(p);
        let (lo, w);
        if s <= -self.smax {
            lo = 0; w = 0.0;
        } else if s >= self.smax {
            lo = self.k - 2; w = 1.0;
        } else {
            let x = (s + self.smax) / self.step;
            let mut l = x as usize;
            if l >= self.k - 1 { l = self.k - 2; }
            lo = l; w = x - l as f64;
        }
        self.lo = lo; self.w = w; self.c = cx;
        let base = cx * self.k;
        self.t[base + lo] * (1.0 - w) + self.t[base + lo + 1] * w
    }
    #[inline]
    fn update(&mut self, y: f64) {
        let base = self.c * self.k;
        let (lo, w, rt) = (self.lo, self.w, self.rate);
        self.t[base + lo] += rt * (1.0 - w) * (y - self.t[base + lo]);
        self.t[base + lo + 1] += rt * w * (y - self.t[base + lo + 1]);
    }
}

struct MatchModel {
    mask: u64,
    tab: Vec<u32>,
    minlen: usize,
    ptr: usize,
    len: usize,
    h: u64,
}
impl MatchModel {
    fn new(hash_bits: u32, minlen: usize) -> Self {
        MatchModel { mask: (1u64 << hash_bits) - 1, tab: vec![0u32; 1usize << hash_bits], minlen, ptr: 0, len: 0, h: 0 }
    }
    #[inline]
    fn predicted(&self, hist: &[u8], phase: usize, byte_pos: usize) -> f64 {
        if self.len == 0 || self.ptr >= byte_pos {
            return 0.0;
        }
        let pb = (hist[self.ptr] >> (7 - phase)) & 1;
        let st = 1.6 + 0.35 * (self.len.min(28) as f64);
        if pb == 1 { st } else { -st }
    }
    fn update_after_byte(&mut self, hist: &[u8], byte_pos: usize) {
        if self.len > 0 && self.ptr < byte_pos {
            if hist[self.ptr] == hist[byte_pos] {
                self.ptr += 1; self.len = (self.len + 1).min(65535);
            } else {
                self.len = 0; self.ptr = 0;
            }
        }
        let b = hist[byte_pos] as u64;
        self.h = ((self.h << 8) | b) & 0xFFFF_FFFF_FFFF;
        if byte_pos + 1 >= self.minlen {
            let hk = (self.h.wrapping_mul(2654435761) & self.mask) as usize;
            let prev = self.tab[hk] as usize;
            self.tab[hk] = (byte_pos + 1) as u32;
            if self.len == 0 && prev != 0 && prev <= byte_pos {
                self.ptr = prev; self.len = self.minlen;
            }
        }
    }
}

#[allow(unused_assignments)] // sp_slot/wd_slot are seeded then reassigned every iteration
fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("data/corpus.txt");
    let cap: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300_000);
    let obits: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(23);
    let osize: usize = 1usize << obits;
    // env-tunable hyperparameters (defaults = the strong baseline) for fast sweeps without recompiling
    let delta = envf("DELTA", 0.08);     // tuned (was 0.18): lower smoothing -> sharper high-count contexts
    let d2 = 2.0 * delta;
    let alr_lr = envf("ALR", 0.0010);    // tuned mixer LR (was 0.0013)
    let alr_lrf = envf("ALRF", 0.0010);  // tuned final-mixer LR
    let use_slots = env::var("BLMSLOTS").map(|s| s == "1").unwrap_or(false);
    let blsoft = env::var("BLSOFT").map(|s| s == "1").unwrap_or(false);
    let use_slots = use_slots || blsoft;      // §74 soft retrieval needs the §72 vectors
    let strip_w = env::var("BLSTRIPW").map(|s| s == "1").unwrap_or(false);  // §77 no word/prev-word
    let strip_h = env::var("BLSTRIPH").map(|s| s == "1").unwrap_or(false);  // §77 no hashed orders 8..32
    if blsoft && strip_w {
        eprintln!("error: BLSOFT=1 with BLSTRIPW=1 is undefined (the soft vote reads the stripped word-model tables wdc)");
        std::process::exit(2);
    }
    let lr_s = envf("LR_S", 0.02);       // §72 slot-vector learning rate
    let sbits: u32 = envf("SBITS", 22.0) as u32;   // slot-vector table bits (evict-on-collision)
    let alr_slot = envf("ALRS", 0.0015); // §72 slot-head mixer LR (own knob, default between strong's and wstate's)
    let soft_k = envf("SOFTK", 8.0) as usize;      // §74 retrieved neighbours per byte
    let simmin = envf("SIMMIN", 0.25);             // §74 cosine floor for a neighbour vote
    let nslots: usize = (envf("NSLOTS", 6.0) as usize).clamp(1, SLOTS);   // §75 active LRU depth
    let blpvec: u32 = envf("BLPVEC", 2.0) as u32;   // §79 per-prefix vector: 0 off, 1 global head, 2 all mixers.
    // §82: the §81 lean selector, ported. Sentence-scoped word LRU + full-weight vote cells for
    // ALL candidates + a query-free contextual-usefulness gate + the vote fed to ALL THREE mixers
    // (the §78.3 placement). Default OFF = bit-identical; NSELSLOTS/SELVBITS/SELUBITS tune it.
    // DEFAULT ON since the §83 owner adoption (gains on all text corpora incl. held-out
    // enwik8 tail; tie on code; §80 standing directive). BLSEL=0 recovers the pre-§83 engine.
    let blsel = env::var("BLSEL").map(|s| s != "0").unwrap_or(true);
    let nselslots: usize = (envf("NSELSLOTS", 2.0) as usize).clamp(1, SELSMAX);   // §84: value is at depth <=2; deeper candidates dilute
    let selvbits: u32 = envf("SELVBITS", 23.0) as u32;   // §84: 2^23 vote cells (bigger corpora need them)
    let selubits: u32 = envf("SELUBITS", 23.0) as u32;
    let sel_ugain = envf("SELUGAIN", 4.0);    // §83 sweep knobs (§81 defaults shown)
    let sel_pbd = envf("SELPBD", 0.05);
    let sel_tsc = envf("SELTSC", 2.0);
    let sel_udc = envf("SELUDC", 0.995);
    // §83 binding scope: clear the candidate set every k-th '.' (1 = §81 sentence scope;
    // 0 = never clear = whole-stream LRU). Stale words' cells pollute, but the CONTEXTUAL
    // usefulness is expected to self-correct at the gate -- this is the experiment.
    // §83 sweep verdict: whole-stream scope (0) beats sentence scope on every text corpus --
    // the CONTEXTUAL usefulness self-corrects the cross-sentence pollution §81 had to scope away.
    let selsent: u32 = envf("SELSENT", 0.0) as u32;
    let seltopm: usize = (envf("SELTOPM", 4.0) as usize).clamp(1, SEL_TOPM);   // §84 sweep
    let selvord: u32 = envf("SELVORD", 3.0) as u32;      // §84: vote/gate context width in bytes
    let selctxmask: u64 = if selvord >= 8 { u64::MAX } else { (1u64 << (8 * selvord)) - 1 };
    let mut sent_ctr: u32 = 0;
    // DEFAULT 2 since the §80 owner adoption of the §79 result (enwik8 0.199145 -> 0.196123,
    // decodability verified §79R); set BLPVEC=0 to recover the pre-§80 engine bit-identically.
    if blpvec > 2 {
        eprintln!("error: BLPVEC must be 0, 1 or 2 (got {})", blpvec);
        std::process::exit(2);
    }
    // DEFAULT ON since the §80 owner adoption (BLWNS=1 was the larger, free share of the §79 gain);
    // set BLWNS=0 to recover the pre-§80 engine bit-identically.
    let blwns = env::var("BLWNS").map(|s| s != "0").unwrap_or(true);     // §79 NS rule on wdc/wdc2
    let ns_all = env::var("BLNSALL").map(|s| s == "1").unwrap_or(false); // §79 NS rule on every count table
    let ns_w = blwns || ns_all;
    let nin = if blsel { NWMAX } else if blpvec == 2 { NIN + PVD } else { NIN };  // §82: bias index
    let nw = if blsel { SELW } else if blpvec == 2 { NWMAX } else { NW };         // §82: active mixer width

    let mut raw = fs::read(path).expect("read input");
    if cap > 0 && raw.len() > cap { raw.truncate(cap); }
    let n = raw.len() * 8;
    let mut bits: Vec<u8> = Vec::with_capacity(n);
    for &byte in &raw {
        for j in (0..8).rev() { bits.push((byte >> j) & 1); }
    }

    let maskb: Vec<u64> = (0..=MAXB).map(|l| if 8 * l >= 64 { u64::MAX } else { (1u64 << (8 * l)) - 1 }).collect();
    let lr_final = 0.01f64;
    let hmask: u64 = (1u64 << HBITS) - 1;

    // flat tables (orders / sparse / word): counts (2 per slot) + 8-bit checksum tags
    let mut ocount: Vec<Vec<u32>> = (0..NM).map(|_| vec![0u32; 2 * osize]).collect();
    let mut otag: Vec<Vec<u8>> = (0..NM).map(|_| vec![0u8; osize]).collect();
    let mut spc: Vec<Vec<u32>> = (0..NSP).map(|_| vec![0u32; 2 * osize]).collect();
    let mut sptag: Vec<Vec<u8>> = (0..NSP).map(|_| vec![0u8; osize]).collect();
    let mut wdc = vec![0u32; 2 * osize]; let mut wdtag = vec![0u8; osize];
    let mut wdc2 = vec![0u32; 2 * osize]; let mut wdtag2 = vec![0u8; osize]; // previous-word model
    // high orders: merged hashed tables (no tags), exactly like the Python
    let mut htab: Vec<Vec<u32>> = (0..NH).map(|_| vec![0u32; 2 * (1usize << HBITS)]).collect();

    let mut mixers = vec![[0.0f64; SELW]; NSEL];
    let mut mixers_g = vec![[0.0f64; SELW]; NSEL];
    // a SECOND context-selected mixer, partitioned by order-2 (prev_byte,prev2) instead of order-1
    const NSEL2: usize = 8 * 2048;
    let mut mixers2 = vec![[0.0f64; SELW]; NSEL2];
    let mut mixers2_g = vec![[0.0f64; SELW]; NSEL2];
    let mut gmix = [0.0f64; SELW];
    let mut gmix_g = [0.0f64; SELW];
    let mut final_w = [0.3f64, 0.3, 0.2, 0.0]; // [w_sel, w_global, w_sel2, bias]
    let mut final_g = [0.0f64; 4];
    let mut apm1 = Apm::new(256 * 8, 33, 0.007);
    let mut apm2 = Apm::new(1024, 33, 0.005);
    let mut apm3 = Apm::new(256 * 8, 33, 0.006); // ctx = current partial byte + phase
    let mut apm4 = Apm::new(256, 33, 0.005); // ctx = match-length bucket + phase
    let mut apm5 = Apm::new(4096, 33, 0.005); // ctx = word hash + phase (text structure)
    let mut apm6 = Apm::new(4096, 33, 0.005); // ctx = order-3 byte hash + phase
    let mut mm = MatchModel::new(22, 5);
    let mut mm2 = MatchModel::new(22, 8); // a second, longer match model (min length 8)

    // §72 word-slot channel state
    let mut semb: Vec<[f64; SD]> = vec![[0.0f64; SD]; 1usize << sbits];   // word vectors
    let mut setag: Vec<u64> = vec![0u64; 1usize << sbits];                // word id as tag
    let mut sl = [0u64; SLOTS];                    // LRU word ids (0 = empty)
    let mut slot_w = [0.0f64; SLOTS * SD];         // head weights into the global mixer
    let mut slot_wg = [0.0f64; SLOTS * SD];        // RMSProp state for the head
    let mut sgrad = [[0.0f64; SD]; SLOTS];         // per-byte exact credit per slot
    let mut slot_feat = [[0.0f64; SD]; SLOTS];     // cached features for the current byte
    let mut slot_tabi = [0usize; SLOTS];           // table index of each slot's word

    // §74 soft-retrieval state: LSH index over the §72 vectors + one trained mixer head
    let mut bucket_vec: Vec<Vec<u64>> = vec![Vec::new(); 256];
    let mut word_bucket: HashMap<u64, u8> = HashMap::new();
    let mut soft_nbrs: Vec<(f64, u64)> = Vec::new();   // (cosine, word) refreshed once per byte
    let mut soft_w = 0.0f64;                           // the head weight into the global mixer
    let mut soft_wg = 0.0f64;                          // RMSProp state for the head

    // §82 selector state (allocated only when BLSEL)
    let selvsz = 2 * (1usize << selvbits);
    let selusz = 1usize << selubits;
    let selvmask = (1usize << selvbits) - 1;
    let selumask = selusz - 1;
    let mut sl2 = [0u64; SELSMAX];                     // sentence-scoped word LRU
    let mut selvt: Vec<f64> = if blsel { vec![0.0f64; selvsz] } else { Vec::new() };  // vote (n0,n1)
    let mut selut: Vec<f64> = if blsel { vec![0.0f64; selusz] } else { Vec::new() };  // usefulness u[(ctx3,word)]
    let mut sel_nc = 0usize;                           // current candidates (packed in sl2[0..sel_nc])
    let mut sel_gk: u64 = 0;                           // the gate ctx3 used for the byte being served
    let mut sel_T = [0usize; SEL_TOPM];   // SEL_TOPM is the MAX (8)                // candidate indices of the vote set
    let mut sel_nT = 0usize;                           // vote-set size
    let mut sel_wT = [0.0f64; SELSMAX];                // renormalised mixture weight per T member
    let mut sel_ix = [0usize; SELSMAX];                // per-bit vote-cell indices (all candidates)
    let mut sel_p1 = [0.0f64; SELSMAX];                // per-bit p_k(1) (all candidates)
    let mut sel_ll = [0.0f64; SELSMAX];                // per-byte log-likelihood accumulators
    let mut sel_uw = 0.0f64;                           // per-byte sum |mixer weight on f|
    let mut sel_f = 0.0f64;                            // per-bit vote feature stretch(pm)
    let mut sel_active = false;                        // whether the per-bit caches are valid

    // §79 per-prefix vector state (tables allocated only when BLPVEC != 0)
    let pvsz = if blpvec != 0 { 1usize << sbits } else { 1 };
    let mut pv_tab: Vec<[f64; PVD]> = vec![[0.0f64; PVD]; pvsz];
    let mut pv_tag: Vec<u64> = vec![0u64; pvsz];
    let mut pv_id: u64 = 0;                            // current prefix id (0 = no letter seen yet)
    let mut pv_ti: usize = 0;                          // its table index
    let mut pv_feat = [0.0f64; PVD];                   // cached vector for the current byte
    let mut pv_w = [0.0f64; PVD];                      // BLPVEC=1 head weights (global mixer)
    let mut pv_wg = [0.0f64; PVD];                     // BLPVEC=1 head RMSProp state
    let mut pv_grad = [0.0f64; PVD];                   // per-byte accumulated credit

    let mut hist: Vec<u8> = Vec::with_capacity(raw.len());
    let mut cur: u64 = 0;
    let mut phase: usize = 0;
    let mut prev_byte: u64 = 0;
    let mut prev2: u64 = 0;
    let mut word_hash: u64 = 0;
    let mut byte_pos: usize = 0;
    let mut htail: u64 = 0;

    let mut sts = [0.0f64; SELW];
    let mut oslot = [0usize; NM];
    let mut oreset = [false; NM];
    // indirect context models: per-context bit-history byte + an adaptive StateMap over the 256 histories
    let mut icm_bh: Vec<Vec<u8>> = (0..NICM).map(|_| vec![0u8; osize]).collect();
    let mut sm_p: Vec<[f64; 256]> = (0..NICM).map(|_| [0.5f64; 256]).collect();
    let mut sm_n: Vec<[u32; 256]> = (0..NICM).map(|_| [0u32; 256]).collect();
    let mut icm_bv = [0usize; NICM];
    let mut icm_ti = [0usize; NICM];
    let mut hslot = [0usize; NH];
    let mut hbase = [0u64; NH];
    let mut sp_slot = [0usize; NSP];
    let mut wd_slot = 0usize;
    let mut wd2_slot = 0usize;
    let mut prev_word_hash: u64 = 0;

    let split = (n as f64 * 0.8) as usize;
    let mut tot = 0.0f64;
    let mut tail = 0.0f64;
    let mut tailn = 0usize;
    let t0 = Instant::now();

    for i in 0..n {
        let prefix = cur;
        // byte-aware orders 0..6 (flat, high-bit hash + tag)
        for k in 0..NM {
            let b = ORDERS[k];
            let l = if byte_pos >= b { b } else { byte_pos };
            let run_ = ((htail & maskb[l]) << phase) | cur;
            let key = (((1u64 << (8 * l + phase)) | run_) << 3) | (phase as u64);
            let h = key.wrapping_mul(MULT);
            let ti = (h >> (64 - obits)) as usize;
            let want = ((h >> (64 - obits - 8)) & 0xFF) as u8;
            oreset[k] = otag[k][ti] != want;
            if oreset[k] { otag[k][ti] = want; ocount[k][2 * ti] = 0; ocount[k][2 * ti + 1] = 0; }
            oslot[k] = ti;
            let (n0, n1) = (ocount[k][2 * ti] as f64, ocount[k][2 * ti + 1] as f64);
            sts[k] = stretch((n1 + delta) / (n0 + n1 + d2));
        }
        // high orders (merged hashed)
        if strip_h {
            for hk in 0..NH { sts[NM + hk] = 0.0; } // §77: stripped -> exact-zero mixer input
        } else {
            for hk in 0..NH {
                let hv = hbase[hk];
                let slot = ((hv.wrapping_mul(2654435761)
                    ^ (phase as u64).wrapping_mul(0x9E37_79B1)
                    ^ prefix.wrapping_mul(2246822519)) & hmask) as usize * 2;
                hslot[hk] = slot;
                let (n0, n1) = (htab[hk][slot] as f64, htab[hk][slot + 1] as f64);
                sts[NM + hk] = stretch((n1 + delta) / (n0 + n1 + d2));
            }
        }
        // sparse models: non-adjacent byte pairs (capture gaps the contiguous orders miss)
        for j in 0..NSP {
            let (oa, ob) = SPOFF[j];
            let ba = if byte_pos >= oa { hist[byte_pos - oa] as u64 } else { 0 };
            let bb = if byte_pos >= ob { hist[byte_pos - ob] as u64 } else { 0 };
            let sk = ((((((1u64 << phase) | cur) << 8) | ba) << 8 | bb) << 5) | ((j as u64) << 3) | (phase as u64);
            let h = sk.wrapping_mul(MULT);
            let ti = (h >> (64 - obits)) as usize;
            let want = ((h >> (64 - obits - 8)) & 0xFF) as u8;
            if sptag[j][ti] != want { sptag[j][ti] = want; spc[j][2 * ti] = 0; spc[j][2 * ti + 1] = 0; }
            sp_slot[j] = ti;
            let (n0, n1) = (spc[j][2 * ti] as f64, spc[j][2 * ti + 1] as f64);
            sts[NM + NH + j] = stretch((n1 + delta) / (n0 + n1 + d2));
        }
        if strip_w {
            // §77: word + previous-word models stripped -> exact-zero mixer inputs, tables untouched
            sts[NM + NH + NSP] = 0.0;
            sts[NM + NH + NSP + 1] = 0.0;
        } else {
            // word
            let wk = (word_hash << 12) | ((((1u64 << phase) | cur) << 3) | (phase as u64));
            {
                let h = wk.wrapping_mul(MULT);
                let ti = (h >> (64 - obits)) as usize;
                let want = ((h >> (64 - obits - 8)) & 0xFF) as u8;
                if wdtag[ti] != want { wdtag[ti] = want; wdc[2 * ti] = 0; wdc[2 * ti + 1] = 0; }
                wd_slot = ti;
                let (n0, n1) = (wdc[2 * ti] as f64, wdc[2 * ti + 1] as f64);
                sts[NM + NH + NSP] = stretch((n1 + delta) / (n0 + n1 + d2));
            }
            // previous-word model: predict the current word's bits from the WORD BEFORE it (text bigrams)
            let wctx = prev_word_hash.wrapping_mul(0x9E37_79B1).wrapping_add(word_hash.wrapping_mul(2654435761));
            let wk2 = (wctx << 12) | ((((1u64 << phase) | cur) << 3) | (phase as u64));
            {
                let h = wk2.wrapping_mul(MULT);
                let ti = (h >> (64 - obits)) as usize;
                let want = ((h >> (64 - obits - 8)) & 0xFF) as u8;
                if wdtag2[ti] != want { wdtag2[ti] = want; wdc2[2 * ti] = 0; wdc2[2 * ti + 1] = 0; }
                wd2_slot = ti;
                let (n0, n1) = (wdc2[2 * ti] as f64, wdc2[2 * ti + 1] as f64);
                sts[NM + NH + NSP + 1] = stretch((n1 + delta) / (n0 + n1 + d2));
            }
        }
        sts[NM + NH + NSP + 2] = mm.predicted(&hist, phase, byte_pos);
        sts[NM + NH + NSP + 3] = mm2.predicted(&hist, phase, byte_pos);
        // indirect context models: the order-K context's bit-HISTORY (recency, not just counts)
        // indexes an adaptive StateMap -> a nonstationary prediction (the ISSE/ICM idea)
        for c in 0..NICM {
            let ti = oslot[ICM_K[c]];
            if oreset[ICM_K[c]] { icm_bh[c][ti] = 0; }
            let bv = icm_bh[c][ti] as usize;
            icm_bv[c] = bv; icm_ti[c] = ti;
            sts[NM + NH + NSP + 4 + c] = stretch(sm_p[c][bv]);
        }
        sts[nin] = 1.0;
        // §79 BLPVEC=2: the current prefix's vector as 8 ordinary inputs (between the models and the bias)
        if blpvec == 2 { for k in 0..PVD { sts[NIN + k] = if pv_id != 0 { pv_feat[k] } else { 0.0 }; } }
        // §82 BLSEL: the selector vote as ONE more input to ALL THREE mixers (§78.3 placement;
        // the mixers are already context-selected, which is the engine's analog of §81's own
        // context-selected weight). Reads EVERY candidate's cell (cells train for all, §81).
        if blsel {
            if sel_active && sel_nc > 0 {
                let ctx3 = htail & selctxmask;
                for j in 0..sel_nc {
                    let h = sl2[j].wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ ctx3.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                        ^ (((phase << 7) | cur as usize) as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
                    let ix = ((sel_mix(h) >> (64 - selvbits)) as usize) & selvmask;
                    sel_ix[j] = ix;
                    let (n0, n1) = (selvt[2 * ix], selvt[2 * ix + 1]);
                    sel_p1[j] = (n1 + 0.2) / (n0 + n1 + 0.4);
                }
                let mut pm = 0.0f64;
                for t in 0..sel_nT { pm += sel_wT[t] * sel_p1[sel_T[t]]; }
                sel_f = stretch(pm);
                sel_uw += gmix[NWMAX].abs();
                sts[NWMAX] = sel_f;
            } else {
                sts[NWMAX] = 0.0;
            }
        }

        let sel = ((phase << 8) | prev_byte as usize) & (NSEL - 1);
        let sel2 = ((phase << 11) | ((prev_byte.wrapping_mul(769) ^ prev2.wrapping_mul(2246822519)) as usize & 2047)) & (NSEL2 - 1);
        let mut d = 0.0;
        for (w, s) in mixers[sel][..nw].iter().zip(&sts[..nw]) { d += w * s; }
        let p_sel = squash(d);
        let mut d2 = 0.0;
        for (w, s) in mixers2[sel2][..nw].iter().zip(&sts[..nw]) { d2 += w * s; }
        let p_sel2 = squash(d2);
        let mut dg = 0.0;
        for (w, s) in gmix[..nw].iter().zip(&sts[..nw]) { dg += w * s; }
        if blpvec == 1 && pv_id != 0 { for k in 0..PVD { dg += pv_w[k] * pv_feat[k]; } }
        if use_slots {
            for si in 0..nslots {
                let f = &slot_feat[si];
                let b = si * SD;
                for k in 0..SD { dg += slot_w[b + k] * f[k]; }
            }
        }
        // §74: similarity-weighted vote of the retrieved neighbours' OWN word-model counts
        let mut soft_st = 0.0f64;
        let mut use_soft = false;
        if blsoft && !soft_nbrs.is_empty() {
            let mut psum = 0.0f64;
            let mut ssum = 0.0f64;
            for &(sim, w) in &soft_nbrs {
                let wk = (w << 12) | ((((1u64 << phase) | cur) << 3) | (phase as u64));
                let h = wk.wrapping_mul(MULT);
                let ti = (h >> (64 - obits)) as usize;
                let want = ((h >> (64 - obits - 8)) & 0xFF) as u8;
                let pi = if wdtag[ti] == want {
                    let (n0, n1) = (wdc[2 * ti] as f64, wdc[2 * ti + 1] as f64);
                    (n1 + delta) / (n0 + n1 + d2)
                } else {
                    0.5
                };
                psum += sim * pi;
                ssum += sim;
            }
            if ssum > 0.0 {
                soft_st = stretch(psum / ssum);
                use_soft = true;
                dg += soft_w * soft_st;
            }
        }
        let p_g = squash(dg);
        let ssel = stretch(p_sel);
        let ssel2 = stretch(p_sel2);
        let sg = stretch(p_g);
        let p_mix = squash(final_w[0] * ssel + final_w[1] * sg + final_w[2] * ssel2 + final_w[3]);
        let pa0 = apm1.refine(p_mix, ((prev_byte << 3) | phase as u64) as usize);
        let pa = 0.3 * p_mix + 0.7 * pa0;
        let pb0 = apm2.refine(pa, ((prev_byte.wrapping_mul(769) + prev2.wrapping_mul(31) + phase as u64) & 1023) as usize);
        let mut p = 0.3 * pa + 0.7 * pb0;
        let pc0 = apm3.refine(p, ((cur << 3) | phase as u64) as usize);
        p = 0.3 * p + 0.7 * pc0;
        let pd0 = apm4.refine(p, ((mm.len.min(31) << 3) | phase) as usize);
        p = 0.3 * p + 0.7 * pd0;
        // §77: APM5 is the only CONTEXT use of word_hash; under BLSTRIPW its context is phase only
        let apm5_w = if strip_w { 0 } else { word_hash & 511 };
        let pe0 = apm5.refine(p, ((apm5_w << 3) | phase as u64) as usize);
        p = 0.3 * p + 0.7 * pe0;
        let pf0 = apm6.refine(p, ((((htail ^ (htail >> 13)) & 511) << 3) | phase as u64) as usize);
        p = 0.3 * p + 0.7 * pf0;
        p = p.clamp(1e-6, 1.0 - 1e-6);

        let y = bits[i];
        let yf = y as f64;
        let cost = -(if y == 1 { p } else { 1.0 - p }).log2();
        tot += cost;
        if i >= split { tail += cost; tailn += 1; }

        // --- updates ---
        let em = yf - p_mix;
        let (gf0, gf1, gf2, gf3) = (em * ssel, em * sg, em * ssel2, em);
        let ord_ = 1.0 - RMS_DECAY;
        final_g[0] = RMS_DECAY * final_g[0] + ord_ * gf0 * gf0;
        final_g[1] = RMS_DECAY * final_g[1] + ord_ * gf1 * gf1;
        final_g[2] = RMS_DECAY * final_g[2] + ord_ * gf2 * gf2;
        final_g[3] = RMS_DECAY * final_g[3] + ord_ * gf3 * gf3;
        final_w[0] += alr_lrf * gf0 / (final_g[0].sqrt() + RMS_EPS);
        final_w[1] += alr_lrf * gf1 / (final_g[1].sqrt() + RMS_EPS);
        final_w[2] += alr_lrf * gf2 / (final_g[2].sqrt() + RMS_EPS);
        final_w[3] += alr_lrf * gf3 / (final_g[3].sqrt() + RMS_EPS);
        let _ = lr_final;
        let e_sel = yf - p_sel;
        let e_sel2 = yf - p_sel2;
        let e_g = yf - p_g;
        // §79 BLPVEC=2: credit for the prefix vector from all three mixers, PRE-update weights
        if blpvec == 2 && pv_id != 0 {
            for k in 0..PVD {
                pv_grad[k] += e_sel * mixers[sel][NIN + k] + e_sel2 * mixers2[sel2][NIN + k] + e_g * gmix[NIN + k];
            }
        }
        {
            let w = &mut mixers[sel][..nw];
            let wg = &mut mixers_g[sel][..nw];
            let sts = &sts[..nw];
            for k in 0..nw {
                let gk = e_sel * sts[k];
                wg[k] = RMS_DECAY * wg[k] + ord_ * gk * gk;
                w[k] += alr_lr * gk / (wg[k].sqrt() + RMS_EPS);
            }
        }
        {
            let w = &mut mixers2[sel2][..nw];
            let wg = &mut mixers2_g[sel2][..nw];
            let sts = &sts[..nw];
            for k in 0..nw {
                let gk = e_sel2 * sts[k];
                wg[k] = RMS_DECAY * wg[k] + ord_ * gk * gk;
                w[k] += alr_lr * gk / (wg[k].sqrt() + RMS_EPS);
            }
        }
        {
            let w = &mut gmix[..nw];
            let wg = &mut gmix_g[..nw];
            let sts = &sts[..nw];
            for k in 0..nw {
                let gk = e_g * sts[k];
                wg[k] = RMS_DECAY * wg[k] + ord_ * gk * gk;
                w[k] += alr_lr * gk / (wg[k].sqrt() + RMS_EPS);
            }
        }
        // §82 BLSEL: full-weight counts for ALL candidates (the §81 chicken-and-egg fix) +
        // per-candidate log-likelihoods (the usefulness signal)
        if blsel && sel_active && sel_nc > 0 {
            let ysi = y as usize;
            for j in 0..sel_nc {
                let pr = if y == 1 { sel_p1[j] } else { 1.0 - sel_p1[j] };
                sel_ll[j] += pr.max(1e-9).log2();
                let j0 = 2 * sel_ix[j];
                let cy = selvt[j0 + ysi] + 1.0;
                if cy + selvt[j0 + 1 - ysi] >= 255.0 {
                    selvt[j0 + ysi] = cy * 0.5;
                    selvt[j0 + 1 - ysi] *= 0.5;
                } else {
                    selvt[j0 + ysi] = cy;
                }
            }
        }
        // §79 BLPVEC=1: train the global-mixer prefix head; credit uses the PRE-update head weight
        if blpvec == 1 && pv_id != 0 {
            for k in 0..PVD {
                let gk = e_g * pv_feat[k];
                let w_old = pv_w[k];
                pv_wg[k] = RMS_DECAY * pv_wg[k] + ord_ * gk * gk;
                pv_w[k] += alr_slot * gk / (pv_wg[k].sqrt() + RMS_EPS);
                pv_grad[k] += e_g * w_old;
            }
        }
        // §72: train the slot head on e_g and accumulate EXACT per-bit credit for the vectors
        // (wstate.py semantics: E-credit uses the PRE-update head weight)
        if use_slots {
            for si in 0..nslots {
                if sl[si] == 0 { continue; }
                let f = &slot_feat[si];
                let b = si * SD;
                let g = &mut sgrad[si];
                for k in 0..SD {
                    let gk = e_g * f[k];
                    let idx = b + k;
                    let w_old = slot_w[idx];
                    slot_wg[idx] = RMS_DECAY * slot_wg[idx] + ord_ * gk * gk;
                    slot_w[idx] += alr_slot * gk / (slot_wg[idx].sqrt() + RMS_EPS);
                    g[k] += e_g * w_old;
                }
            }
        }
        // §74: train the soft-retrieval head on the same global-mixer error
        if blsoft && use_soft {
            let gk = e_g * soft_st;
            soft_wg = RMS_DECAY * soft_wg + ord_ * gk * gk;
            soft_w += alr_lr * gk / (soft_wg.sqrt() + RMS_EPS);
        }
        // count bumps + recency halving
        let yi = y as usize;
        for k in 0..NM {
            let s = 2 * oslot[k];
            ocount[k][s + yi] += 1;
            if ocount[k][s + yi] >= CLIMIT { ocount[k][s] = (ocount[k][s] + 1) >> 1; ocount[k][s + 1] = (ocount[k][s + 1] + 1) >> 1; }
            if ns_all { ns_discount(&mut ocount[k], s, yi); }
        }
        // ICM: adapt the StateMap toward y at the observed history, then shift y into the history byte
        for c in 0..NICM {
            let bv = icm_bv[c];
            let pr = sm_p[c][bv];
            sm_p[c][bv] = pr + (yf - pr) / (sm_n[c][bv] as f64 + 1.5);
            if sm_n[c][bv] < 1023 { sm_n[c][bv] += 1; }
            icm_bh[c][icm_ti[c]] = (((bv << 1) | yi) & 0xFF) as u8;
        }
        for j in 0..NSP {
            let s = 2 * sp_slot[j]; spc[j][s + yi] += 1;
            if spc[j][s + yi] >= CLIMIT { spc[j][s] = (spc[j][s] + 1) >> 1; spc[j][s + 1] = (spc[j][s + 1] + 1) >> 1; }
            if ns_all { ns_discount(&mut spc[j], s, yi); }
        }
        if !strip_w {
            {
                let s = 2 * wd_slot; wdc[s + yi] += 1;
                if wdc[s + yi] >= CLIMIT { wdc[s] = (wdc[s] + 1) >> 1; wdc[s + 1] = (wdc[s + 1] + 1) >> 1; }
                if ns_w { ns_discount(&mut wdc, s, yi); }
            }
            {
                let s = 2 * wd2_slot; wdc2[s + yi] += 1;
                if wdc2[s + yi] >= CLIMIT { wdc2[s] = (wdc2[s] + 1) >> 1; wdc2[s + 1] = (wdc2[s + 1] + 1) >> 1; }
                if ns_w { ns_discount(&mut wdc2, s, yi); }
            }
        }
        if !strip_h {
            for hk in 0..NH {
                let s = hslot[hk]; htab[hk][s + yi] += 1;
                if htab[hk][s + yi] >= CLIMIT { htab[hk][s] = (htab[hk][s] + 1) >> 1; htab[hk][s + 1] = (htab[hk][s + 1] + 1) >> 1; }
                if ns_all { ns_discount(&mut htab[hk], s, yi); }
            }
        }
        apm1.update(yf);
        apm2.update(yf);
        apm3.update(yf);
        apm4.update(yf);
        apm5.update(yf);
        apm6.update(yf);

        cur = (cur << 1) | (y as u64);
        phase += 1;
        if phase == 8 {
            // §79: apply this byte's accumulated credit to the current prefix vector (v += LR_S*g, clip)
            if blpvec != 0 && pv_id != 0 {
                for k in 0..PVD {
                    let e = pv_tab[pv_ti][k] + lr_s * pv_grad[k];
                    pv_tab[pv_ti][k] = if e > 1.0 { 1.0 } else if e < -1.0 { -1.0 } else { e };
                    pv_grad[k] = 0.0;
                }
                pv_feat = pv_tab[pv_ti];
            }
            // §72: apply this byte's accumulated vector credit to each slot's word
            if use_slots {
                for si in 0..nslots {
                    if sl[si] == 0 { continue; }
                    let ti = slot_tabi[si];
                    let g = &mut sgrad[si];
                    for k in 0..SD {
                        // §77 SIGN FIX: g accumulates e_g*w = (y-p_g)*w = -dL/df, so descent is
                        // v += lr_s*g. §72-§76 subtracted it (gradient ASCENT on the vectors).
                        let e = semb[ti][k] + lr_s * g[k];
                        semb[ti][k] = if e > 1.0 { 1.0 } else if e < -1.0 { -1.0 } else { e };
                        g[k] = 0.0;
                    }
                }
            }
            let b = (cur & 0xFF) as u8;
            hist.push(b);
            mm.update_after_byte(&hist, byte_pos);
            mm2.update_after_byte(&hist, byte_pos);
            htail = ((htail << 8) | (b as u64)) & maskb[MAXB];
            // §82: usefulness update for the byte just SERVED (candidates still pre-LRU)
            if blsel && sel_active && sel_nc > 0 {
                let mut mu = 0.0f64;
                for j in 0..sel_nc { mu += sel_ll[j]; }
                mu /= sel_nc as f64;
                let mut wu = sel_uw / 8.0;
                if wu > 1.0 { wu = 1.0; }
                if wu > 0.02 {
                    for j in 0..sel_nc {
                        let mut adv = sel_ll[j] - mu;
                        if adv > 1.0 { adv = 1.0; } else if adv < -1.0 { adv = -1.0; }
                        let h = sel_mix(sel_gk ^ sl2[j].wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        let ui = ((h >> (64 - selubits)) as usize) & selumask;
                        let mut u = selut[ui] * sel_udc + (1.0 - sel_udc) * wu * adv;
                        if u > SEL_UCLIP { u = SEL_UCLIP; } else if u < -SEL_UCLIP { u = -SEL_UCLIP; }
                        selut[ui] = u;
                    }
                }
            }
            if blsel {
                for j in 0..SELSMAX { sel_ll[j] = 0.0; }
                sel_uw = 0.0;
                if b == 46 {                     // §81 sentence scoping / §83 scope knob
                    sent_ctr = sent_ctr.wrapping_add(1);
                    if selsent >= 1 && sent_ctr >= selsent {
                        sent_ctr = 0;
                        for j in 0..SELSMAX { sl2[j] = 0; }
                        sel_nc = 0;
                    }
                }
            }
            if (65..=90).contains(&b) || (97..=122).contains(&b) {
                word_hash = (word_hash.wrapping_mul(131) + ((b | 0x20) as u64)) & 0xFFF_FFFF;
            } else {
                if word_hash != 0 {
                    prev_word_hash = word_hash; // remember the word that just ended
                    if blsel {
                        // §82: sentence-scoped LRU of completed words (packed, depth nselslots)
                        let wid = word_hash;
                        let mut pos = sel_nc;
                        for j in 0..sel_nc { if sl2[j] == wid { pos = j; break; } }
                        if pos < sel_nc {
                            sl2.copy_within(0..pos, 1);
                        } else if sel_nc < nselslots {
                            sl2.copy_within(0..sel_nc, 1);
                            sel_nc += 1;
                        } else {
                            sl2.copy_within(0..nselslots - 1, 1);
                        }
                        sl2[0] = wid;
                    }
                    if use_slots {
                        // LRU move-to-front / insert of the completed word
                        let wid = word_hash;
                        let mut pos = nslots;
                        for j in 0..nslots { if sl[j] == wid { pos = j; break; } }
                        if pos < nslots { sl.copy_within(0..pos, 1); }
                        else { sl.copy_within(0..nslots - 1, 1); }
                        sl[0] = wid;
                        // reload features for every slot (handles LRU shifts + rare evictions)
                        for si in 0..nslots {
                            let w = sl[si];
                            if w == 0 { slot_feat[si] = [0.0; SD]; continue; }
                            let h = w.wrapping_mul(MULT);
                            let ti = (h >> (64 - sbits)) as usize;
                            if setag[ti] != w {
                                setag[ti] = w;
                                let mut s = w;
                                let mut v = [0.0f64; SD];
                                for k in 0..SD {
                                    v[k] = (splitmix(&mut s) as f64 / u64::MAX as f64 - 0.5) * 0.1;
                                }
                                semb[ti] = v;
                            }
                            slot_feat[si] = semb[ti];
                            slot_tabi[si] = ti;
                        }
                        // §74: refresh the LSH index for slot words, then retrieve the
                        // neighbours of sl[0] (its sign bucket + the 8 one-bit flips)
                        for si in 0..nslots {
                            if sl[si] != 0 {
                                index_update(sl[si], sbits, &semb, &setag,
                                             &mut bucket_vec, &mut word_bucket);
                            }
                        }
                        if blsoft {
                            soft_nbrs.clear();
                            let w0 = sl[0];
                            if w0 != 0 {
                                let ti0 = (w0.wrapping_mul(MULT) >> (64 - sbits)) as usize;
                                if setag[ti0] == w0 {
                                    let v0 = semb[ti0];
                                    let mut n0 = 0.0f64;
                                    for k in 0..SD { n0 += v0[k] * v0[k]; }
                                    n0 = n0.sqrt();
                                    if n0 > 1e-9 {
                                        let b0 = bucket_of(&v0) as usize;
                                        for j in 0..9usize {
                                            let bb = if j == 8 { b0 } else { b0 ^ (1 << j) };
                                            for ii in 0..bucket_vec[bb].len() {
                                                let w = bucket_vec[bb][ii];
                                                if w == w0 { continue; }
                                                if soft_nbrs.iter().any(|&(_, x)| x == w) { continue; }
                                                let tiw = (w.wrapping_mul(MULT) >> (64 - sbits)) as usize;
                                                if setag[tiw] != w { continue; }
                                                let vw = semb[tiw];
                                                let mut nw = 0.0f64;
                                                let mut dt = 0.0f64;
                                                for k in 0..SD {
                                                    dt += v0[k] * vw[k];
                                                    nw += vw[k] * vw[k];
                                                }
                                                nw = nw.sqrt();
                                                if nw < 1e-9 { continue; }
                                                let sim = dt / (n0 * nw);
                                                if sim > simmin { soft_nbrs.push((sim, w)); }
                                            }
                                        }
                                        soft_nbrs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                                        soft_nbrs.truncate(soft_k);
                                    }
                                }
                            }
                        }
                    }
                    word_hash = 0;
                } else {
                    word_hash = 0;
                }
            }
            // §82: forward -- the query-free gate over the CURRENT (post-LRU) slots -> serves the next byte
            if blsel {
                let ctx3 = htail & selctxmask;
                sel_gk = ctx3;
                if sel_nc > 0 {
                    let mut e = [0.0f64; SELSMAX];
                    let mut mx = -1e30f64;
                    for j in 0..sel_nc {
                        let h = sel_mix(ctx3 ^ sl2[j].wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        let ui = ((h >> (64 - selubits)) as usize) & selumask;
                        let mut ev = sel_ugain * selut[ui] - sel_pbd * j as f64;
                        if ev > 12.0 { ev = 12.0; } else if ev < -12.0 { ev = -12.0; }
                        e[j] = ev;
                        if ev > mx { mx = ev; }
                    }
                    let mut z = 0.0f64;
                    let mut a = [0.0f64; SELSMAX];
                    for j in 0..sel_nc { let ex = ((e[j] - mx) / sel_tsc).exp(); a[j] = ex; z += ex; }
                    for j in 0..sel_nc { a[j] /= z; }
                    sel_nT = seltopm.min(sel_nc);
                    let mut taken = [false; SELSMAX];
                    let mut at = 0.0f64;
                    for t in 0..sel_nT {
                        let mut best = 0usize;
                        let mut bv = -1.0f64;
                        for j in 0..sel_nc { if !taken[j] && a[j] > bv { bv = a[j]; best = j; } }
                        taken[best] = true;
                        sel_T[t] = best;
                        at += bv;
                    }
                    if at <= 0.0 { at = 1.0; }
                    for t in 0..sel_nT { sel_wT[t] = a[sel_T[t]] / at; }
                    sel_active = true;
                } else {
                    sel_nT = 0;
                    sel_active = false;
                }
            }
            // §79: a letter moves the prefix key to the new letters-only prefix; a non-letter KEEPS the
            // last prefix (the finished word) until the next letter
            if blpvec != 0 && ((65..=90).contains(&b) || (97..=122).contains(&b)) {
                let id = word_hash | (1u64 << 40);
                let ti = (id.wrapping_mul(MULT) >> (64 - sbits)) as usize;
                if pv_tag[ti] != id {
                    pv_tag[ti] = id;
                    let mut sd = id;
                    let mut v = [0.0f64; PVD];
                    for k in 0..PVD { v[k] = (splitmix(&mut sd) as f64 / u64::MAX as f64 - 0.5) * 0.1; }
                    pv_tab[ti] = v;
                }
                pv_id = id; pv_ti = ti; pv_feat = pv_tab[ti];
            }
            prev2 = prev_byte; prev_byte = b as u64; cur = 0; phase = 0; byte_pos += 1;
            if !strip_h {
                for hk in 0..NH {
                    let bb = HORDERS[hk];
                    let lo = if byte_pos >= bb { byte_pos - bb } else { 0 };
                    let mut hv: u64 = 1469598103934665603;
                    for bp in lo..byte_pos { hv = (hv ^ hist[bp] as u64).wrapping_mul(1099511628211); }
                    hbase[hk] = hv;
                }
            }
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let whole = tot / n as f64;
    let last = if tailn > 0 { tail / tailn as f64 } else { 0.0 };
    println!("corpus={}  bytes={}  bits={}  obits={}", path, raw.len(), n, obits);
    println!("  flags: BLSTRIPW={} BLSTRIPH={} BLMSLOTS={} BLSOFT={} NSLOTS={} BLPVEC={} BLWNS={} BLNSALL={} LR_S={} BLSEL={} NSELSLOTS={} SELSENT={}",
             strip_w as u8, strip_h as u8, use_slots as u8, blsoft as u8, nslots, blpvec, blwns as u8, ns_all as u8, lr_s,
             blsel as u8, nselslots, selsent);
    println!("  blmrs-strong  whole-stream = {:.6}   last-20% = {:.6}  bits/bit   [{:.1}s, {:.1} Mbits/s]",
             whole, last, secs, (n as f64 / 1e6) / secs);
}
