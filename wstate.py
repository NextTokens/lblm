#!/usr/bin/env python3
"""
wstate.py -- PHASE 1, the decisive experiment: does a SEMANTIC recurrent state cross BELOW
the order-n baseline where SS70's byte-bit state (lstate.py) only reached parity?

Diagnosis being tested (SS70/SS71): a compact state fed the 8 RAW BITS of each byte can only
track byte-class running statistics -- structure the order models already have. The named fix
("richer input projection -- not a new idea"): feed the state the CURRENT WORD'S EMBEDDING
alongside the bits, with the embedding LEARNED ONLINE BY THE COMPRESSION LOSS ITSELF (one-step-
truncated RTRL through the state, mixer weights as feedback). m=32 fixed-long-timescale diagonal
EMA units (timescales ~7 .. ~2000 bytes; SS70: forced-long decays tie learned ones).

    h_j(t+1) = a_j h_j(t) + (1-a_j) * W_j . x(t),   x = [8 signed bits of byte, E[w_t] (D=16)]
    W learned (exact diagonal RTRL):  W_jk  -= lr_rec  * G_j(t) * e_jk,
                                      e_jk  <- (1-a_j) x_k + a_j e_jk
    E learned (word-trace credit): each recent word w carries an m-vector eligibility
           A[w,j] <- a_j A[w,j] + (1-a_j) G_j  [only while w was the last active word]
           flushed on eviction/end-of-run as  E[w,k] -= lr_emb * sum_j A[w,j] W_j[8+k]
           (exact long-range credit for frozen W -- the 1-step truncation taught embeddings
            only to predict their OWN next letter, which random vectors already do; v2 fix)
    G_j(t) = sum over the byte's 8 bits of (p - y) * w_mixer[state_j]   (RTRL learning signal)

v3: +BUCKET EXPERT -- the probe (_assoc_check) showed the LINEAR mixer cannot read word
identity out of the EMA superposition beyond ~1 word back (bitsonly==learned==chance at a
2-word cue; the v1/v2 embedding credit is not the bottleneck, the readout is). So every
state arm additionally feeds a NONLINEAR count-table expert keyed
    (16 sign bits of mid-timescale units 12..27, bit phase, partial byte)
(the prims.py mechanism, on the learned state).

v4: +WORD-SLOT BINDING -- buckets on the EMA still fail the 2-word-cue probe: the EMA
SUPERPOSES ~5 words, so no function of it isolates one word's identity. The fix is the
project's own cycle-5 idea reborn: an LRU of the last S=6 DISTINCT words, each slot holding
that word's OWN small vector (SD=8), fed DIRECTLY as mixer features (no superposition, no
RTRL needed: a slot persists for ~6 words, so every bit while resident gives EXACT credit
to the embedding). Arms:
    slotsr : EMA(bits) + bucket + slots with FROZEN random vectors  (identity control)
    slots  : EMA(bits) + bucket + slots with vectors LEARNED by the loss (THE HEADLINE)
  slots vs bitsonly = the slot channel's effect; slots vs slotsr = MEANING vs identity.
  RESULT (§72): slots CROSS below the orders baseline at >=1.2MB, margin growing with data,
  replicated across seeds; slotsr never crosses -> the win is the learned vectors.
  CORRECTED (SS77): in that run the LRU held word PREFIXES (see WSLOTMODE below), so slot 0 was the
  current prefix and the crossing is a learned current-word model: one prefix slot (WSLOTS=1) beats
  S=6, and real whole-word slots (WSLOTMODE=word) lose to the orders baseline at every size measured
  (seed 0; ledger SS77.2).

v5 (Phase 2, similarity): the strong.rs port showed identity slots add NOTHING to an
lpaq-class engine (hashed order-8..32 already span 1-5 words) -- the open lever is
SIMILARITY, generalisation across related words. Two new arms, both = slots + ONE extra
readout of the CURRENT word (prefix id, like the word model):
    slotsw : + an IDENTITY-keyed count expert  (word-hash bucket, phase, partial) [control]
    semsim : + an EMBEDDING-BUCKET count expert (8 sign bits of the current word's vector ->
             256 buckets; words the loss has ALIGNED share counts -> evidence transfers
             across the word tail)  + S coherence features dot(E[cur], E[slot_i])
  semsim < slotsw (same capacity, same placement) => SIMILARITY beats IDENTITY keying --
  the Phase-2 thesis. semsim < slots => the similarity layer adds margin on its own.
  CORRECTED (SS77): under WSLOTMODE=prefix slots[0] is the current prefix at word-interior bytes, so slotsw
  is a word model and semsim buckets per-prefix predictor vectors; this could not measure similarity.

v7 (SS77, ATTENTIONAL READ): SS75/76 read the deep slots as HOLDING information a fixed recency readout
cannot use (SS77: that was a noise-dimension artifact). The pre-registered test reads the LRU with CONTENT-based
attention, served once per byte (forward at byte end, after the LRU update):
    c = E[slot0] (last completed word under WSLOTMODE=word);  q = Wq c + bq;  e_k = q.E_k + beta_k;  a = softmax(e)
    pooled r = sum_k a_k E_k  (SD features);  T = top-m slots by a (WTOPM=4)
    vote v = sum_{k in T} (a_k/A_T) stretch(P_vote[hash(word_k, order-WVORD ctx, phase, partial)])
  credit is EXACT for parameters frozen within the byte (per-bit dL/dr and dL/da accumulated with
  the predict-time mixer weights, backprop through the softmax applied at byte end). Arms:
    attn    : learned vectors + learned query + learned position bias       (THE HEADLINE)
    attnr   : vectors FROZEN, query/bias learn                              -> MEANING control
    attnrec : learned vectors, a_k = recency weights GP[k] (SS76), T = m most recent
                                                                            -> SELECTION control
    attnscr : the SD+1 attention features replaced by noise                 -> floor
    attncos : = attn, but COSINE content score at a fixed scale (SS77c; WKAPPA, default 8.0):
              e_k = KAPPA (q.E_k) / (|q||E_k| + 1e-8) + beta_k  -- content logits O(KAPPA) whatever the
              vector norms (attn's q.E_k spread is ~0.002 at init norms ~0.08, so attn ran near-uniform and
              position-driven). Exact backward through the normalisation for q and every E_k.
  SS77 finding (instrument bug): the slot LRU was updated at EVERY letter byte with the growing
  PREFIX id, so S=6 held ~2 words ('cat','ca','c','quick','quic','qui') and S=32 ~6-8 words;
  strong.rs BLMSLOTS inserts only COMPLETED words. WSLOTMODE=prefix (default, SS72-76 bit-identical)
  | word (completed words only -- the intended mode for the attn arms, with WSLOTS=32).
  SS77 RESULT: the long-gap probe (_bind_probe.py) killed the attention arms -- no content arm binds a
  cue at any gap; the vote's mixer weight settles negative, so correct votes hurt (ledger SS77.4).

Words: maximal [A-Za-z0-9] runs, lowercased, FNV-1a rolling hash -> id (prefix-visible, like
the core's word model). id 0 (no active word) -> zero embedding. The EMA of word embeddings is
a running TOPIC vector -- long-range structure no order-n byte table represents.

ARMS (genmem.py protocol: match/copy OFF everywhere, 13-byte-DECONTAMINATED held-out wt103,
orders 0..6 + state features in one online logistic mixer; deterministic mask, see below):
  baseline : orders only                            -- the rail (== SS70 baseline)
  bitsonly : m=32 state, x = bits only              -- attribution: is any win just m=32?
  randemb  : m=32 state, x = bits + FROZEN random per-word vectors (never trained)
             -- identity control: consistent word identity WITHOUT semantic generalization
  learned  : m=32 state, x = bits + embeddings TRAINED by the loss   -- THE HEADLINE
  scrambled: state features replaced by noise (input too)            -- the noise floor

Attribution ladder (all on the same decontaminated scored bytes):
  learned < baseline   -> the state helps at all
  learned < bitsonly   -> the EMBEDDING did it, not capacity           [the named fix works]
  learned < randemb    -> the win is MEANING, not lexical identity     [the words wall]
  learned < scrambled  -> real learned signal, not noise               [SS70's bar]

NOTE: genmem.clean_mask uses Python's process-salted hash(); this script runs arms as parallel
PROCESSES, so it re-implements the identical 13-gram excision with a deterministic FNV-1a hash
-> every arm scores byte-identical test positions.

Run:  python wstate.py                (5 arms x sizes [150,450,1200,2700] KB, arms in parallel)
      python wstate.py --selftest     (tiny sequential sanity run)
"""
import sys, math, random, time, os
from array import array
from concurrent.futures import ProcessPoolExecutor

from genmem import stretch, squash

ORDERS = [0, 1, 2, 3, 4, 5, 6]
M = 32            # state units (SS70 used 12)
XDIM_BITS = 8     # raw signed bits of the just-finished byte
D = 16            # word-embedding dimension
XDIM = XDIM_BITS + D
AMIN, AMAX = 0.85, 0.9995      # fixed decay spread -> timescales ~7 .. ~2000 bytes
FNV0 = 0x811C9DC5
MAXREC = 48                    # recent-word trace bank size (eligibility, evict-oldest)
S = int(os.environ.get("WSLOTS", "6"))   # word slots (LRU depth; 6 = SS72; deep = beyond order-32 reach)
SD = 8                         # slot vector dims
H = int(os.environ.get("WHEADS", "4"))  # SS76: shared recency-discounted projection heads
GAMMA = float(os.environ.get("WGAMMA", "0.85"))
# SS77: WHAT the slot LRU binds. "prefix" (default, the SS72-76 behaviour, kept bit-identical) inserts
# the growing PREFIX id at EVERY letter byte -- so S=6 holds ~2 words ('cat','ca','c','quick','quic',
# 'qui') and S=32 ~6-8 words. "word" inserts only COMPLETED words (on the byte that ends a word),
# matching strong.rs BLMSLOTS -- so S slots really are the last S distinct words.
SLOTMODE = os.environ.get("WSLOTMODE", "prefix")
assert SLOTMODE in ("prefix", "word"), SLOTMODE
GP = [GAMMA ** k for k in range(256)]   # precomputed recency discounts
PROJ_ARMS = ("proj", "projr", "projscr")
ATTN_ARMS = ("attn", "attnr", "attnrec", "attnscr", "attncos")   # SS77: attentional read over the slot LRU
ATTN_READ_ARMS = ("attn", "attnr", "attnrec", "attncos")         # the non-floor attention arms
KAPPA = float(os.environ.get("WKAPPA", "8.0"))           # SS77c: attncos fixed cosine scale
TOPM = int(os.environ.get("WTOPM", "4"))                 # SS77: vote over the top-m attended slots
VORD = int(os.environ.get("WVORD", "3"))                 # SS77: vote context order (bytes)
VBITS = int(os.environ.get("WVBITS", "22"))              # SS77: vote table = 2 * 2^VBITS counts
M64 = (1 << 64) - 1
SLOT_ARMS = ("slots", "slotsr", "slotsw", "semsim", "semfast", "matchslots",
             "proj", "projr", "projscr", "scrambled") + ATTN_ARMS
LEARNED_SLOT_ARMS = ("slots", "slotsw", "semsim", "semfast", "matchslots")  # slot vectors trained by the loss
SIM_ARMS = ("slotsw", "semsim")            # v5: +word expert (identity vs embedding bucket)
XKEY_ARMS = ("slotsw", "semsim", "semfast")
SUBWORD_ARMS = ("semfast",)                # v6: char-3gram-composed vector init (fastText-style)
MATCH_ARMS = ("matchbase", "matchslots")   # §75: the reconciliation test -- copy ON in the
# instrument too. If slots' crossing collapses with the match channel present, the no-copy win
# lived on the signal the match model harvests in production (repeated word usage).


def is_wb(b):
    return (65 <= b <= 90) or (97 <= b <= 122) or (48 <= b <= 57)


def clean_mask_det(train, test, K=13):
    """genmem.clean_mask, but with a process-independent FNV-1a gram hash (deterministic)."""
    grams = set()
    for i in range(len(train) - K + 1):
        h = FNV0
        for j in range(K):
            h = ((h ^ train[i + j]) * 0x01000193) & 0xFFFFFFFF
        grams.add(h)
    contam = bytearray(len(test))
    for i in range(len(test) - K + 1):
        h = FNV0
        for j in range(K):
            h = ((h ^ test[i + j]) * 0x01000193) & 0xFFFFFFFF
        if h in grams:
            for j in range(i, i + K):
                contam[j] = 1
    return [c == 0 for c in contam]


class Model:
    def __init__(self, arm="baseline", lr=0.004, lr_rec=0.02, lr_emb=0.03, lr_s=0.02, lr_head=0.02, seed=0):
        self.arm = arm
        self.use_state = arm not in ("baseline", "matchbase")
        self.use_match = arm in MATCH_ARMS     # §75 reconciliation arms: copy ON in the instrument
        self.lr, self.lr_rec, self.lr_emb, self.lr_s, self.lr_head = lr, lr_rec, lr_emb, lr_s, lr_head
        self.NM = len(ORDERS)
        self.NIN = self.NM + ((M + 1) if self.use_state else 0)     # +1 = bucket expert
        self.NIN = self.NIN + (1 if self.use_match else 0)          # match/copy vote
        if arm in SLOT_ARMS and arm not in PROJ_ARMS and arm not in ATTN_ARMS:
            self.NIN += S * SD                                     # word-slot features
        if arm in PROJ_ARMS:
            self.NIN += H                                          # SS76 projection-head features
        if arm in ATTN_ARMS:
            self.NIN += SD + 1                                     # SS77 pooled features + vote
        if arm in XKEY_ARMS:
            self.NIN += 1                                          # word-expert feature
        if arm in ("semsim", "semfast"):
            self.NIN += S                                          # coherence dot features
        self.w = [0.0] * (self.NIN + 1)
        self.wg = [0.0] * (self.NIN + 1)
        self.tab = [dict() for _ in ORDERS]
        self.btab = {}                        # bucket expert counts (state arms)
        self.buck = 0
        self.xtab = {}                        # v5 word-expert counts (slotsw/semsim)
        self.xb = 0                           # its key bucket, refreshed per byte
        self.curv = None                      # current word's vector cache (semsim)
        self.curdots = [0.0] * S              # dot(E[cur], E[slot_i]) cache (semsim)
        self.slots = [0] * S                   # LRU word ids (0 = empty)
        self.semb = {}                         # word id -> SD floats (slot content)
        self.sgrad = {}                        # word id -> per-byte accumulated exact credit
        if self.use_match:                     # §75: genmem's byte match/copy model
            self.MINLEN, self.GATE = 16, 18
            self.mtab = {}
            self.hist = bytearray()
            self.mptr, self.mlen = -1, 0
        if arm in PROJ_ARMS:                   # §76: shared recency-discounted heads
            rng = random.Random(seed + 991)
            self.hw = [[(rng.random() - 0.5) * 0.2 for _ in range(SD)] for _ in range(H)]
            self.hgrad = [0.0] * H             # per-byte exact credit per head
            self.hfeat = [0.0] * H             # cached features (serve the next byte)
            self.hdw = [[0.0] * SD for _ in range(H)]   # d feat_h / d w_h  (cached)
            self.hck = [0.0] * max(S, 1)       # d feat_h / d E[slot_k] coefficient per slot
        if arm == "scrambled" or arm == "projscr" or arm == "attnscr":
            self._sr = random.Random(1234)
        if arm in ATTN_READ_ARMS:                   # SS77 attentional read
            if arm != "attnrec":
                rng = random.Random(seed + 1777)
                self.Wq = [[(1.0 if i == j else 0.0) + rng.uniform(-0.05, 0.05) for j in range(SD)]
                           for i in range(SD)]
                self.bq = [0.0] * SD
                self.beta = [0.0] * S
            self.vtab = array("H", [0]) * (2 << VBITS)   # flat (n0, n1) vote counts
            self.at_n = 0                          # non-empty slots served (0 = features/vote off)
            self.at_ks = []; self.at_E = []; self.at_a = []; self.at_T = []; self.at_wT = []
            self.at_AT = 1.0; self.at_c = None; self.at_q = None
            self.at_cos = None                     # attncos backward cache (|q|, |E_k|, q.E_k, den_k)
            self.at_r = [0.0] * SD; self.at_hb = []
            self.aGr = [0.0] * SD                  # per-byte dL/dr
            self.aGa = []                          # per-byte dL/da_k (vote path), aligned with at_T
            self.vt_idx = []; self.vt_s = []; self.vt_v = 0.0   # per-bit vote cache (predict time)
        self.htail = 0; self.cur = 0; self.phase = 0
        self.sbase = self.NM
        self.wh = FNV0                 # rolling FNV-1a over lowercased word bytes
        self.last_word = 0
        self.gtab = {}                 # v6: char-3gram id -> SD vector (subword table)
        self.tri_of = {}               # v6: word id -> its 3gram ids (for init + credit share)
        self.wtri = []                 # v6: rolling 3grams of the current word
        self.wtail2 = (0, 0)           # v6: last two letter bytes (for 3gram ids)
        self.inw = False               # SS77: inside a word run
        self.done = 0                  # SS77: id of the word COMPLETED by the byte just seen (else 0)
        if self.use_state:
            self.a = [AMIN + (AMAX - AMIN) * (j / (M - 1)) for j in range(M)]
            rng = random.Random(seed)
            self.W = [[(rng.random() - 0.5) * 0.1 for _ in range(XDIM)] for _ in range(M)]
            self.h = [0.0] * M
            self.eW = [[0.0] * XDIM for _ in range(M)]
            self.G = [0.0] * M
            self.emb = {}                  # word id -> D floats (learned, or frozen-random)
            self.wcount = {}               # word id -> occurrences (diagnostics only)
            self.rec = {}                  # word id -> m-vector eligibility trace (learned arm)
            self.recorder = []             # insertion order for evict-oldest

    def _octx(self, k):
        b = ORDERS[k]
        mask = (1 << (8 * b)) - 1 if b else 0
        return (self.phase, self.cur, (self.htail & mask) if b else 0)

    # ---- word id of the [A-Za-z0-9] run ENDING at byte b (prefix-visible); 0 = none ----
    def _word_after(self, b):
        if is_wb(b):
            self.wh = ((self.wh ^ (b | 32)) * 0x01000193) & 0xFFFFFFFF
            c = b | 32
            b1, b2 = self.wtail2
            if b1:                                   # a full 3gram is available
                self.wtri.append((b1 << 16) | (b2 << 8) | c)
            self.wtail2 = (b2, c)
            self.inw = True; self.done = 0
            return self.wh | (1 << 31)     # never 0
        if self.wh and self.arm in SUBWORD_ARMS:   # v6: a word just completed — stash its 3grams
            self.tri_of.setdefault(self.wh | (1 << 31), tuple(self.wtri))
        self.done = (self.wh | (1 << 31)) if self.inw else 0
        self.inw = False
        self.wh = FNV0
        self.wtri = []
        self.wtail2 = (0, 0)
        return 0

    def _vec(self, wid):
        v = self.emb.get(wid)
        if v is None:
            r = random.Random((wid * 2654435761) & 0x7FFFFFFF)
            v = [(r.random() - 0.5) * 0.1 for _ in range(D)]   # same init scale in every arm
            self.emb[wid] = v
        self.wcount[wid] = self.wcount.get(wid, 0) + 1
        return v

    def _svec(self, wid):
        v = self.semb.get(wid)
        if v is None:
            tris = self.tri_of.get(wid) if self.arm in SUBWORD_ARMS else None
            base = None
            if tris:
                acc = [0.0] * SD; n = 0
                for t in tris:
                    g = self.gtab.get(t)
                    if g is not None:
                        for k in range(SD):
                            acc[k] += g[k]
                        n += 1
                if n:
                    base = [a / n for a in acc]      # mean of TRAINED 3gram vectors
            r = random.Random((wid * 40503) & 0x7FFFFFFF)
            if base is not None:
                v = [base[k] + (r.random() - 0.5) * 0.02 for k in range(SD)]   # composed init
            else:
                v = [(r.random() - 0.5) * 0.1 for _ in range(SD)]
            self.semb[wid] = v
        self.wcount[wid] = self.wcount.get(wid, 0) + 1
        return v

    def predict(self):
        sts = [0.0] * (self.NIN + 1); i = 0
        for k in range(self.NM):
            c = self.tab[k].get(self._octx(k))
            n0, n1 = (c[0], c[1]) if c else (0, 0)
            sts[i] = stretch((n1 + 0.2) / (n0 + n1 + 0.4)); i += 1
        if self.use_state:
            if self.arm == "scrambled":
                for j in range(M):
                    sts[i] = self._sr.uniform(-1, 1); i += 1
            else:
                for j in range(M):
                    h = self.h[j]
                    sts[i] = h if -5.0 < h < 5.0 else (5.0 if h > 0 else -5.0); i += 1
            c = self.btab.get(self._bkey())
            n0, n1 = (c[0], c[1]) if c else (0, 0)
            sts[i] = stretch((n1 + 0.2) / (n0 + n1 + 0.4))
            i += 1
        if self.arm in SLOT_ARMS and self.arm not in PROJ_ARMS and self.arm not in ATTN_ARMS:
            if self.arm == "scrambled":
                for j in range(S * SD):
                    sts[i] = self._sr.uniform(-1, 1); i += 1
            else:
                for w_id in self.slots:
                    if w_id:
                        v = self._svec(w_id)
                        for k in range(SD):
                            sts[i] = v[k]; i += 1
                    else:
                        i += SD
        if self.arm in PROJ_ARMS:
            if self.arm == "projscr":
                for hh in range(H):
                    sts[i] = self._sr.uniform(-1, 1); i += 1
            else:
                for hh in range(H):
                    sts[i] = self.hfeat[hh]; i += 1
        if self.arm in ATTN_ARMS:
            if self.arm == "attnscr":
                for j in range(SD + 1):
                    sts[i] = self._sr.uniform(-1, 1); i += 1
            elif self.at_n:
                r = self.at_r
                for j in range(SD):
                    sts[i] = r[j]; i += 1
                key2 = (self.phase << 8) | self.cur
                vt = self.vtab; vmask = (1 << VBITS) - 1
                idxs = []; ss = []; v = 0.0; wT = self.at_wT
                for t, hb in enumerate(self.at_hb):
                    hx = ((hb ^ key2) * 0x94D049BB133111EB) & M64
                    ix = (hx ^ (hx >> 29)) & vmask
                    n0 = vt[2 * ix]; n1 = vt[2 * ix + 1]
                    s = stretch((n1 + 0.2) / (n0 + n1 + 0.4))
                    idxs.append(ix); ss.append(s)
                    v += wT[t] * s
                self.vt_idx = idxs; self.vt_s = ss; self.vt_v = v
                sts[i] = v; i += 1
            else:
                i += SD + 1
        if self.arm in XKEY_ARMS:
            c = self.xtab.get((self.xb << 10) | (self.phase << 7) | self.cur)
            n0, n1 = (c[0], c[1]) if c else (0, 0)
            sts[i] = stretch((n1 + 0.2) / (n0 + n1 + 0.4)); i += 1
        if self.arm == "semsim":
            for j in range(S):
                sts[i] = self.curdots[j]; i += 1
        if self.use_match:
            st = 0.0
            if self.mlen >= self.GATE and 0 <= self.mptr < len(self.hist):
                pb = self.hist[self.mptr]
                if self.phase == 0 or (pb >> (8 - self.phase)) == self.cur:
                    bit = (pb >> (7 - self.phase)) & 1
                    st = (1.6 + 0.35 * min(self.mlen, 28)) * (1 if bit else -1)
            sts[i] = st
        sts[self.NIN] = 1.0
        d = 0.0; w = self.w
        for j in range(self.NIN + 1):
            d += w[j] * sts[j]
        return squash(d), sts

    def _bkey(self):
        return (self.buck << 10) | (self.phase << 7) | self.cur

    def step(self, y, learn):
        p, sts = self.predict()
        cost = -math.log2(p if y == 1 else 1 - p)
        if self.use_state and self.arm != "scrambled":
            sb = self.sbase; w = self.w; g = p - y          # dL/dz, nats
            for j in range(M):
                self.G[j] += g * w[sb + j]                  # accumulate dL/dh_j over the byte
        if self.arm in ("proj", "projr"):
            # §76: exact per-bit credit for the head features (mixer weight at predict time)
            g = p - y
            base = self.NM + M + 1; w = self.w
            for hh in range(H):
                self.hgrad[hh] += g * w[base + hh]
        if self.arm in ATTN_READ_ARMS and self.at_n:
            # SS77: exact per-bit credit (mixer weights at predict time) to r and, via the vote, to a
            g = p - y
            base = self.NM + M + 1; w = self.w; Gr = self.aGr
            for j in range(SD):
                Gr[j] += g * w[base + j]
            if self.arm != "attnrec":
                gv = g * w[base + SD] / self.at_AT
                v = self.vt_v; ss = self.vt_s; Ga = self.aGa
                for t in range(len(ss)):
                    Ga[t] += gv * (ss[t] - v)
        if self.arm in LEARNED_SLOT_ARMS:
            # exact credit to slot-embedding dims: mixer weight at predict time, per bit
            g = p - y
            base = self.NM + M + 1; w = self.w
            for si in range(S):
                w_id = self.slots[si]
                if w_id:
                    gr = self.sgrad.get(w_id)
                    if gr is None:
                        gr = [0.0] * SD
                        self.sgrad[w_id] = gr
                    bi = base + si * SD
                    for k in range(SD):
                        gr[k] += g * w[bi + k]
        if learn:
            err = y - p
            for j in range(self.NIN + 1):
                q = err * sts[j]
                self.wg[j] = 0.999 * self.wg[j] + 0.001 * q * q
                self.w[j] += self.lr * q / (math.sqrt(self.wg[j]) + 1e-4)
            for k in range(self.NM):
                key = self._octx(k); c = self.tab[k].get(key)
                if c is None:
                    c = [0, 0]; self.tab[k][key] = c
                c[y] += 1
            if self.use_state:
                key = self._bkey(); c = self.btab.get(key)
                if c is None:
                    c = [0, 0]; self.btab[key] = c
                c[y] += 1
            if self.arm in XKEY_ARMS:
                key = (self.xb << 10) | (self.phase << 7) | self.cur
                c = self.xtab.get(key)
                if c is None:
                    c = [0, 0]; self.xtab[key] = c
                c[y] += 1
            if self.arm in ATTN_READ_ARMS and self.at_n:
                vt = self.vtab
                for ix in self.vt_idx:
                    j0 = 2 * ix; jy = j0 + y
                    cy = vt[jy] + 1
                    if cy >= 255:
                        vt[jy] = (cy + 1) >> 1
                        jo = j0 + 1 - y
                        vt[jo] = (vt[jo] + 1) >> 1
                    else:
                        vt[jy] = cy
        self.cur = (self.cur << 1) | y; self.phase += 1
        if self.phase == 8:
            self._byte_end(learn)
        return cost

    def _byte_end(self, learn):
        b = self.cur & 0xFF
        wid = self._word_after(b)
        if self.arm in SUBWORD_ARMS and wid and wid not in self.tri_of:
            self.tri_of[wid] = tuple(self.wtri)     # stash the prefix's 3grams so far
        # §76: apply the head/vector credit for the byte just SERVED (slots still pre-LRU)
        if self.arm in ("proj", "projr") and learn:
            if self.arm == "proj":
                for k in range(S):
                    ck = self.hck[k]
                    if not ck:
                        continue
                    w_id = self.slots[k]
                    v = self.semb.get(w_id)
                    if not w_id or v is None:
                        continue
                    for dd in range(SD):
                        acc = 0.0
                        for hh in range(H):
                            acc += self.hgrad[hh] * ck * self.hw[hh][dd]
                        e = v[dd] - self.lr_s * acc
                        v[dd] = 1.0 if e > 1.0 else (-1.0 if e < -1.0 else e)
            for hh in range(H):
                g = self.hgrad[hh]
                if g:
                    hdw = self.hdw[hh]
                    wh = self.hw[hh]
                    for dd in range(SD):
                        wh[dd] -= self.lr_head * g * hdw[dd]
            self.hgrad = [0.0] * H
        if self.arm in ATTN_READ_ARMS:
            # SS77: apply the attention credit for the byte just SERVED (slots still pre-LRU)
            if learn and self.at_n:
                self._attn_apply(self._attn_grads())
            self.aGr = [0.0] * SD
            self.aGa = [0.0] * len(self.at_T)
        swid = wid if SLOTMODE == "prefix" else self.done
        if self.arm in SLOT_ARMS and swid:
            sl = self.slots
            if swid in sl:
                sl.remove(swid)
            else:
                del sl[-1]
            sl.insert(0, swid)
        if self.arm in ("proj", "projr") and self.use_state:
            # §76: refresh head features + exact-credit caches from the post-LRU slots
            den = 0.0
            for k in range(S):
                if self.slots[k]:
                    den += GP[k]
            if den <= 0.0:
                den = 1.0
            for k in range(S):
                self.hck[k] = (GP[k] / den) if self.slots[k] else 0.0
            for hh in range(H):
                wh = self.hw[hh]
                num = 0.0
                acc = [0.0] * SD
                for k in range(S):
                    w_id = self.slots[k]
                    if not w_id:
                        continue
                    v = self.semb.get(w_id)
                    if v is None:
                        v = self._svec(w_id)
                    c = GP[k]
                    d = 0.0
                    for dd in range(SD):
                        d += wh[dd] * v[dd]
                        acc[dd] += c * v[dd]
                    num += c * d
                self.hfeat[hh] = num / den
                self.hdw[hh] = [a / den for a in acc]
        if self.arm in LEARNED_SLOT_ARMS and self.sgrad:
            for w_id, gr in self.sgrad.items():
                v = self.semb.get(w_id)
                if v is None:
                    continue
                for k in range(SD):
                    e = v[k] - self.lr_s * gr[k]
                    v[k] = 1.0 if e > 1.0 else (-1.0 if e < -1.0 else e)
                if self.arm in SUBWORD_ARMS:
                    # v6: share the credit with the word's char-3grams (1/len each) so future
                    # UNSEEN words composing those 3grams start inside their morphological family
                    tris = self.tri_of.get(w_id)
                    if tris:
                        sc = self.lr_s / len(tris)
                        for t in tris:
                            gv = self.gtab.get(t)
                            if gv is None:
                                gv = [0.0] * SD
                                self.gtab[t] = gv
                            for k in range(SD):
                                e = gv[k] - sc * gr[k]
                                gv[k] = 1.0 if e > 1.0 else (-1.0 if e < -1.0 else e)
            self.sgrad.clear()
        if self.use_state and self.arm != "scrambled" and learn:
            # 1) per-word eligibility traces (v2 long-range credit): decay every recent word's
            #    trace; the word that shaped the state serving THIS byte collects its G fresh
            if self.arm == "learned":
                a = self.a; G = self.G; lw = self.last_word
                for wj, A in self.rec.items():
                    for j in range(M):
                        A[j] = a[j] * A[j]
                if lw:
                    A = self.rec.get(lw)
                    if A is None:
                        if len(self.recorder) >= MAXREC:            # evict oldest -> flush
                            old = self.recorder.pop(0)
                            A = self.rec.pop(old, None)
                            if A is not None:
                                self._flush(old, A)
                        A = [0.0] * M
                        self.rec[lw] = A
                        self.recorder.append(lw)
                    for j in range(M):
                        A[j] += (1.0 - a[j]) * G[j]
            # 2) exact diagonal RTRL for W with THIS byte's learning signal
            for j in range(M):
                gj = self.G[j]
                if gj:
                    Wj = self.W[j]; eWj = self.eW[j]
                    for k in range(XDIM):
                        e = Wj[k] - self.lr_rec * gj * eWj[k]
                        Wj[k] = 2.0 if e > 2.0 else (-2.0 if e < -2.0 else e)
        # 3) build x, roll eligibility traces and state (h(t+1) uses the UPDATED W)
        if self.use_state:
            x = [0.0] * XDIM
            for k in range(8):
                x[k] = 2.0 * ((b >> k) & 1) - 1.0
            if self.arm == "scrambled":
                for k in range(XDIM):
                    x[k] = self._sr.uniform(-1, 1)
            elif self.arm in ("learned", "randemb") and wid:
                x[8:] = self._vec(wid)
            a = self.a; W = self.W; eW = self.eW; h = self.h
            for j in range(M):
                aj = a[j]; Wj = W[j]; eWj = eW[j]
                wr = 0.0
                for k in range(XDIM):
                    wr += Wj[k] * x[k]
                    eWj[k] = (1.0 - aj) * x[k] + aj * eWj[k]
                h[j] = aj * h[j] + (1.0 - aj) * wr
            buck = 0
            for u in range(12, 28):
                buck = (buck << 1) | (1 if h[u] > 0.0 else 0)
            self.buck = buck
            self.G = [0.0] * M
            self.last_word = wid
        if self.arm in XKEY_ARMS:
            # v5 (revised): key on the LAST COMPLETED word (slots[0]) — its vector is TRAINED
            # (it was slot-resident), the key fires at word boundaries AND interiors, and the
            # identity/embedding arms key the SAME positions. (The first version keyed on the
            # growing prefix id, whose vectors are untrained for multi-letter interiors and dead
            # at boundaries — a flaw the identity control is immune to, i.e. an unfair A/B.)
            w_id = self.slots[0] if self.use_state else 0
            if not w_id:
                self.xb = 0; self.curv = None
                for si in range(S):
                    self.curdots[si] = 0.0
            elif self.arm == "slotsw":
                self.xb = (((w_id * 2654435761) & 0xFFFFFFFF) >> 20) & 0xFFF     # 4096 identity slots
            else:
                v = self._svec(w_id)
                xb = 0
                for k in range(SD):
                    xb = (xb << 1) | (1 if v[k] > 0.0 else 0)
                self.xb = xb
                self.curv = v
                self.curdots[0] = 0.0                       # slot 0 IS the key word; skip self
                for si in range(1, S):
                    w2 = self.slots[si]
                    if w2:
                        sv = self.semb.get(w2)
                        if sv is None:
                            sv = self._svec(w2)
                        d = 0.0
                        for k in range(SD):
                            d += v[k] * sv[k]
                        self.curdots[si] = d
                    else:
                        self.curdots[si] = 0.0
        self.htail = ((self.htail << 8) | b) & ((1 << 48) - 1); self.cur = 0; self.phase = 0
        if self.arm in ATTN_READ_ARMS:
            self._attn_forward()       # post-LRU slots, post-htail context -> serves the next byte
        if self.use_match:
            self._match_after(b)
        return None

    # ---- SS77 attentional read ----
    def _attn_forward(self):
        """pooled features, top-m vote set and backward caches from the CURRENT slots/params."""
        sl = self.slots
        if not sl[0]:
            self.at_n = 0; self.at_T = []; self.at_hb = []; self.aGa = []
            self.aGr = [0.0] * SD
            return
        ks = [k for k in range(S) if sl[k]]
        E = []
        for k in ks:
            v = self.semb.get(sl[k])
            if v is None:
                v = self._svec(sl[k])
            E.append(v)
        n = len(ks)
        if self.arm == "attnrec":
            den = 0.0
            for k in ks:
                den += GP[k]
            a = [GP[k] / den for k in ks]
            c = None; q = None
        else:
            c = list(E[0])
            q = [0.0] * SD
            for i2 in range(SD):
                Wi = self.Wq[i2]; acc = self.bq[i2]
                for j in range(SD):
                    acc += Wi[j] * c[j]
                q[i2] = acc
            e = [0.0] * n
            if self.arm == "attncos":
                # SS77c: e_k = KAPPA (q.E_k) / (|q||E_k| + 1e-8) + beta_k
                nq = 0.0
                for j in range(SD):
                    nq += q[j] * q[j]
                nq = math.sqrt(nq)
                nE = [0.0] * n; sc = [0.0] * n; dn = [0.0] * n
                for t in range(n):
                    Et = E[t]; d = 0.0; ne = 0.0
                    for j in range(SD):
                        d += q[j] * Et[j]
                        ne += Et[j] * Et[j]
                    ne = math.sqrt(ne)
                    den = nq * ne + 1e-8
                    nE[t] = ne; sc[t] = d; dn[t] = den
                    e[t] = KAPPA * d / den + self.beta[ks[t]]
                self.at_cos = (nq, nE, sc, dn)
            else:
                for t in range(n):
                    Et = E[t]; d = self.beta[ks[t]]
                    for j in range(SD):
                        d += q[j] * Et[j]
                    e[t] = d
            mx = max(e)
            ex = [math.exp(x - mx) for x in e]
            z = sum(ex)
            a = [x / z for x in ex]
        r = [0.0] * SD
        for t in range(n):
            at = a[t]; Et = E[t]
            for j in range(SD):
                r[j] += at * Et[j]
        T = sorted(range(n), key=lambda t: -a[t])[:TOPM]      # stable: ties -> more recent slot
        AT = 0.0
        for t in T:
            AT += a[t]
        ctx = self.htail & ((1 << (8 * VORD)) - 1)
        hb = []
        for t in T:
            h = ((sl[ks[t]] * 0x9E3779B97F4A7C15) ^ (ctx * 0xC2B2AE3D27D4EB4F)) & M64
            h = ((h ^ (h >> 31)) * 0xBF58476D1CE4E5B9) & M64
            hb.append(h ^ (h >> 27))
        self.at_n = n; self.at_ks = ks; self.at_E = E; self.at_a = a; self.at_T = T
        self.at_AT = AT; self.at_wT = [a[t] / AT for t in T]; self.at_c = c; self.at_q = q
        self.at_r = r; self.at_hb = hb
        self.aGr = [0.0] * SD; self.aGa = [0.0] * len(T)

    def _attn_grads(self):
        """exact gradients of the byte's summed nat loss (params frozen within the byte).
        returns (dE per served slot position, dWq, dbq, dbeta{k: g}); query parts None for attnrec."""
        n = self.at_n; E = self.at_E; a = self.at_a; Gr = self.aGr
        if self.arm == "attnrec":
            dE = [[a[t] * Gr[j] for j in range(SD)] for t in range(n)]
            return dE, None, None, None
        dA = [0.0] * n
        for t in range(n):
            Et = E[t]; d = 0.0
            for j in range(SD):
                d += Gr[j] * Et[j]
            dA[t] = d
        for u, t in enumerate(self.at_T):
            dA[t] += self.aGa[u]
        sbar = 0.0
        for t in range(n):
            sbar += a[t] * dA[t]
        de = [a[t] * (dA[t] - sbar) for t in range(n)]
        q = self.at_q; c = self.at_c
        dq = [0.0] * SD
        if self.arm == "attncos":
            # SS77c: e_k = K s_k / den_k + beta_k,  s_k = q.E_k,  den_k = |q||E_k| + 1e-8
            #   de_k/dq   = K E_k / den_k - K s_k |E_k| (q/|q|)   / den_k^2
            #   de_k/dE_k = K q   / den_k - K s_k |q|   (E_k/|E_k|) / den_k^2
            # (|q| = 0 forces s_k = 0, so the second term's exact limit is 0; likewise |E_k| = 0.)
            nq, nE, sc, dn = self.at_cos
            dE = [[a[t] * Gr[j] for j in range(SD)] for t in range(n)]
            for t in range(n):
                g1 = de[t] * KAPPA / dn[t]
                h = de[t] * KAPPA * sc[t] / (dn[t] * dn[t])
                hq = h * nE[t] / nq if nq > 0.0 else 0.0
                hE = h * nq / nE[t] if nE[t] > 0.0 else 0.0
                Et = E[t]; dEt = dE[t]
                for j in range(SD):
                    dq[j] += g1 * Et[j] - hq * q[j]
                    dEt[j] += g1 * q[j] - hE * Et[j]
        else:
            for t in range(n):
                dt = de[t]; Et = E[t]
                for j in range(SD):
                    dq[j] += dt * Et[j]
        dbeta = {self.at_ks[t]: de[t] for t in range(n)}
        dWq = [[dq[i2] * c[j] for j in range(SD)] for i2 in range(SD)]
        dbq = list(dq)
        if self.arm != "attncos":
            dE = [[a[t] * Gr[j] + de[t] * q[j] for j in range(SD)] for t in range(n)]
        Wq = self.Wq; d0 = dE[0]                     # slot 0 is also the query source
        for j in range(SD):
            acc = 0.0
            for i2 in range(SD):
                acc += Wq[i2][j] * dq[i2]
            d0[j] += acc
        return dE, dWq, dbq, dbeta

    def _attn_apply(self, grads):
        dE, dWq, dbq, dbeta = grads
        if self.arm != "attnr":
            lr = self.lr_s
            for t in range(self.at_n):
                v = self.at_E[t]; g = dE[t]
                for j in range(SD):
                    e = v[j] - lr * g[j]
                    v[j] = 1.0 if e > 1.0 else (-1.0 if e < -1.0 else e)
        if dWq is not None:
            lr = self.lr_head
            for i2 in range(SD):
                Wi = self.Wq[i2]; gi = dWq[i2]
                for j in range(SD):
                    e = Wi[j] - lr * gi[j]
                    Wi[j] = 8.0 if e > 8.0 else (-8.0 if e < -8.0 else e)
                e = self.bq[i2] - lr * dbq[i2]
                self.bq[i2] = 8.0 if e > 8.0 else (-8.0 if e < -8.0 else e)
            for k, g in dbeta.items():
                e = self.beta[k] - lr * g
                self.beta[k] = 8.0 if e > 8.0 else (-8.0 if e < -8.0 else e)

    def _match_after(self, b):
        """genmem's match model update: extend/break the candidate, register the context."""
        n = len(self.hist)
        self.hist.append(b)
        if self.mlen > 0 and self.mptr < n - 1:
            if self.hist[self.mptr] == self.hist[n - 1]:
                self.mptr += 1; self.mlen = min(self.mlen + 1, 65535)
            else:
                self.mlen = 0; self.mptr = -1
        if n >= self.MINLEN:
            key = bytes(self.hist[n - self.MINLEN:n])
            prev = self.mtab.get(key, -1)
            self.mtab[key] = n
            if self.mlen == 0 and 0 <= prev < n:
                self.mptr = prev; self.mlen = self.MINLEN

    def _flush(self, wid, A):
        """apply accumulated eligibility: E[wid] -= lr_emb * (A . W[:, emb])"""
        v = self.emb.get(wid)
        if v is None:
            return
        W = self.W
        for k in range(D):
            kk = XDIM_BITS + k
            acc = 0.0
            for j in range(M):
                acc += A[j] * W[j][kk]
            e = v[k] - self.lr_emb * acc
            v[k] = 1.0 if e > 1.0 else (-1.0 if e < -1.0 else e)

    def _flush_all(self):
        for wid, A in self.rec.items():
            self._flush(wid, A)
        self.rec.clear(); self.recorder.clear()

    def run(self, raw, learn, mask=None):
        tot = 0.0; nb = 0; bp = 0
        for b in raw:
            scored = (mask is None) or (bp < len(mask) and mask[bp])
            bc = 0.0
            for j in range(7, -1, -1):
                bc += self.step((b >> j) & 1, learn)
            if scored:
                tot += bc; nb += 1
            bp += 1
        if learn and self.arm == "learned":
            self._flush_all()
        return tot / max(1, nb)


# ---------------- worker: one arm across all sizes ----------------
def run_arm(arm, sizes, seed=0, train_path="data/wt103_train.txt", test_path="data/wt103_test.txt"):
    train_all = open(train_path, "rb").read()
    test_all = open(test_path, "rb").read()
    rows = []
    t0 = time.time()
    model = None
    tr = te = b""
    for kb in sizes:
        tr = train_all[:kb * 1024]
        te = test_all[:min(400 * 1024, kb * 1024 // 2)]
        mask = clean_mask_det(tr, te, 13)
        m = Model(arm=arm, seed=seed)
        m.run(tr, learn=True)
        bpb = m.run(te, learn=True, mask=mask)
        rows.append((kb, bpb))
        model = m
    return arm, rows, time.time() - t0, neighbor_payload(model, tr, te)


def id2str(data):
    """rebuild word-id -> string with the exact rolling hash the model uses (diagnostics)."""
    m = {}; s = bytearray(); h = FNV0
    for b in data:
        if is_wb(b):
            h = ((h ^ (b | 32)) * 0x01000193) & 0xFFFFFFFF
            s.append(b | 32)
        else:
            if s:
                m.setdefault(h | (1 << 31), bytes(s).decode("latin1"))
            h = FNV0; s = bytearray()
    if s:
        m.setdefault(h | (1 << 31), bytes(s).decode("latin1"))
    return m


def neighbor_payload(model, tr, te):
    """for vector-armed models: top-frequency words (len>=3) + 3 nearest neighbours by cosine."""
    if model is None:
        return None
    table = model.semb if model.arm in SLOT_ARMS else getattr(model, "emb", {})
    if not table:
        return None
    ids = id2str(tr + te)
    top = [(w, c) for w, c in sorted(model.wcount.items(), key=lambda kv: -kv[1])
           if len(ids.get(w, "")) >= 3][:40]
    vecs = [(wid, cnt, table[wid]) for wid, cnt in top if wid in table]
    out = []
    for i, (wa, ca, va) in enumerate(vecs):
        best = []
        for j, (wb_, cb, vb) in enumerate(vecs):
            if i == j:
                continue
            na = math.sqrt(sum(t * t for t in va)) or 1e-9
            nb_ = math.sqrt(sum(t * t for t in vb)) or 1e-9
            cos = sum(t * u for t, u in zip(va, vb)) / (na * nb_)
            best.append((cos, ids.get(wb_, hex(wb_))))
        best.sort(reverse=True)
        out.append((ids.get(wa, hex(wa)), ca, [(f"{c:+.2f}", s) for c, s in best[:3]]))
    return out[:12]


# ---------------- main ----------------
ARMS = ["baseline", "slots", "slotsw", "semsim", "semfast", "scrambled"]


def report(results, sizes, secs):
    by = {arm: dict(rows) for arm, rows, _, _ in results}
    present = set(by)
    print()
    hdr = (f"{'train':>8}" + "".join(f"{a:>10}" for a in ARMS))
    print(hdr); print("-" * len(hdr))
    for kb in sizes:
        print(f"{kb:>6}KB" + "".join(f"{by[a][kb]:>10.4f}" for a in ARMS))
    print("-" * len(hdr))
    def series(f):
        return [round(f(by, kb), 4) for kb in sizes]
    if "slots" in present and "baseline" in present:
        help_s = series(lambda B, kb: B["baseline"][kb] - B["slots"][kb])     # >0 slots help
        print(f"slots  vs baseline (the crossing)   : {help_s}")
        if "slotsr" in present:
            sem_s = series(lambda B, kb: B["slotsr"][kb] - B["slots"][kb])    # >0 MEANING vs identity
            print(f"slots  vs slotsr (MEANING)         : {sem_s}")
        if "scrambled" in present:
            noi_s = series(lambda B, kb: B["scrambled"][kb] - B["slots"][kb]) # >0 real signal
            print(f"slots  vs scrambled (noise floor)  : {noi_s}")
            if any(h > 0.001 for h in help_s) and all(n > 0.003 for n in noi_s):
                verdict_s = "CROSS -- word-slot binding beats the orders baseline on THIS data too"
            elif all(n > 0.003 for n in noi_s):
                verdict_s = "signal present (above noise) but no cross at these sizes"
            else:
                verdict_s = "NEGATIVE on this data"
            print(f"  VERDICT (slots): {verdict_s}")
    if {"semsim", "baseline", "slotsw"} <= present:
        help_m = series(lambda B, kb: B["baseline"][kb] - B["semsim"][kb])
        sim_id = series(lambda B, kb: B["slotsw"][kb] - B["semsim"][kb])
        sim_marg = series(lambda B, kb: B["slots"][kb] - B["semsim"][kb]) if "slots" in present else None
        print(f"semsim vs baseline (help)          : {help_m}")
        print(f"semsim vs slotsw (SIMILARITY>IDENT): {sim_id}")
        if sim_marg is not None:
            print(f"semsim vs slots (layer margin)     : {sim_marg}")
    if "semfast" in present and {"semsim", "slotsw"} <= present:
        sub = series(lambda B, kb: B["semsim"][kb] - B["semfast"][kb])       # >0 subword init helps
        sub_id = series(lambda B, kb: B["slotsw"][kb] - B["semfast"][kb])    # >0 subword+sem beats identity
        print(f"semfast vs semsim (subword init)   : {sub}")
        print(f"semfast vs slotsw (vs identity)    : {sub_id}")
    if "proj" in present and "baseline" in present:
        p_help = series(lambda B, kb: B["baseline"][kb] - B["proj"][kb])          # >0 deep readout crosses
        print(f"proj   vs baseline (deep crossing)   : {p_help}")
        if "projr" in present:
            p_sem = series(lambda B, kb: B["projr"][kb] - B["proj"][kb])          # >0 MEANING
            print(f"proj   vs projr (MEANING)           : {p_sem}")
        if "projscr" in present:
            p_noi = series(lambda B, kb: B["projscr"][kb] - B["proj"][kb])        # >0 real signal
            print(f"proj   vs projscr (noise floor)     : {p_noi}")
            if any(h > 0.001 for h in p_help) and all(n > 0.003 for n in p_noi):
                print("  VERDICT (proj): CROSS -- the parameter-efficient deep readout crosses.")
            elif all(n > 0.003 for n in p_noi):
                print("  VERDICT (proj): signal present, no cross at these sizes.")
            else:
                print("  VERDICT (proj): NEGATIVE.")
    if "attn" in present and "baseline" in present:
        a_help = series(lambda B, kb: B["baseline"][kb] - B["attn"][kb])         # >0 crosses
        print(f"attn   vs baseline (crossing)        : {a_help}")
        a_sel = None
        if "attnr" in present:
            a_sem = series(lambda B, kb: B["attnr"][kb] - B["attn"][kb])         # >0 MEANING
            print(f"attn   vs attnr (MEANING)           : {a_sem}")
        if "attnrec" in present:
            a_sel = series(lambda B, kb: B["attnrec"][kb] - B["attn"][kb])       # >0 content beats recency
            print(f"attn   vs attnrec (SELECTION)       : {a_sel}")
        if "attnscr" in present:
            a_noi = series(lambda B, kb: B["attnscr"][kb] - B["attn"][kb])       # >0 real signal
            print(f"attn   vs attnscr (floor)           : {a_noi}")
            if any(h > 0.001 for h in a_help) and all(x > 0.003 for x in a_noi):
                v_a = "CROSS -- the attentional read crosses the orders baseline"
            elif all(x > 0.003 for x in a_noi):
                v_a = "signal present, no cross at these sizes"
            else:
                v_a = "NEGATIVE"
        else:
            v_a = "no floor arm (attnscr) -- verdict undetermined"
        if a_sel is not None:
            v_a += ("; SELECTION > 0 at every size (content beats recency)" if all(x > 0 for x in a_sel)
                    else "; SELECTION NOT > 0 at every size (content does not beat recency)")
        else:
            v_a += "; SELECTION untested (no attnrec arm)"
        print(f"  VERDICT (attn): {v_a}.")
    if "attncos" in present:
        if "baseline" in present:
            c_help = series(lambda B, kb: B["baseline"][kb] - B["attncos"][kb])  # >0 crosses
            print(f"attncos vs baseline (crossing)       : {c_help}")
        if "attnrec" in present:
            c_sel = series(lambda B, kb: B["attnrec"][kb] - B["attncos"][kb])    # >0 content beats recency
            print(f"attncos vs attnrec (SELECTION)       : {c_sel}")
    print(f"\n[{secs:.0f}s total]")


def main():
    global ARMS
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    selftest = "--selftest" in sys.argv
    sizes = [40] if selftest else [150, 450, 1200, 2700]
    train_path = "data/wt103_train.txt"
    test_path = "data/wt103_test.txt"
    if "--train" in sys.argv:
        train_path = sys.argv[sys.argv.index("--train") + 1]
        test_path = sys.argv[sys.argv.index("--test") + 1]
    if "--sizes" in sys.argv:
        sizes = [int(x) for x in sys.argv[sys.argv.index("--sizes") + 1].split(",")]
    if "--arms" in sys.argv:
        arms = sys.argv[sys.argv.index("--arms") + 1].split(",")
    else:
        arms = ARMS
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    ARMS = arms                       # report() prints what actually ran
    print("=" * 104)
    print("SEMANTIC-STATE instrument (SS72-74 lineage)")
    print(f"  data: {train_path} / {test_path}")
    print(f"  arms: {' | '.join(ARMS)}   sizes(KB): {sizes}   seed: {seed}")
    print(f"  slots: WSLOTMODE={SLOTMODE}  WSLOTS={S}  WHEADS={H}  WGAMMA={GAMMA}"
          f"  WTOPM={TOPM}  WVORD={VORD}  WVBITS={VBITS}  WKAPPA={KAPPA}")
    print("  gate: copy/match OFF, 13-byte-decontaminated held-out, deterministic mask")
    print("=" * 104)
    t0 = time.time()
    if selftest:
        results = [run_arm(a, sizes, seed, train_path, test_path) for a in ARMS]
    else:
        with ProcessPoolExecutor(max_workers=len(ARMS)) as ex:
            futs = [ex.submit(run_arm, a, sizes, seed, train_path, test_path) for a in ARMS]
            results = [f.result() for f in futs]
    secs = time.time() - t0
    for arm, rows, dt, _ in results:
        print(f"  [{arm:<10}] " + "  ".join(f"{kb}KB={b:.4f}" for kb, b in rows) + f"   [{dt:.0f}s]")
    report(results, sizes, secs)
    for arm, rows, dt, nn in results:
        if nn:
            print(f"\nEMBEDDING NEIGHBOURS ({arm} arm, top words, cosine):")
            for w, c, nbrs in nn:
                print(f"  {w:<14} x{c:<6} ->  " + "   ".join(f"{s}({cs})" for cs, s in nbrs))
    print("=" * 104)


if __name__ == "__main__":
    main()
