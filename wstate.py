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
import sys, math, random, time
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
S = 6                          # word slots (LRU of recent distinct words)
SD = 8                         # slot vector dims
SLOT_ARMS = ("slots", "slotsr", "scrambled")


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
    def __init__(self, arm="baseline", lr=0.004, lr_rec=0.02, lr_emb=0.03, lr_s=0.02, seed=0):
        self.arm = arm                     # baseline|bitsonly|randemb|learned|slotsr|slots|scrambled
        self.use_state = arm != "baseline"
        self.lr, self.lr_rec, self.lr_emb, self.lr_s = lr, lr_rec, lr_emb, lr_s
        self.NM = len(ORDERS)
        self.NIN = self.NM + ((M + 1) if self.use_state else 0)     # +1 = bucket expert
        if arm in SLOT_ARMS:
            self.NIN += S * SD                                     # word-slot features
        self.w = [0.0] * (self.NIN + 1)
        self.wg = [0.0] * (self.NIN + 1)
        self.tab = [dict() for _ in ORDERS]
        self.btab = {}                        # bucket expert counts (state arms)
        self.buck = 0
        self.slots = [0] * S                   # LRU word ids (0 = empty)
        self.semb = {}                         # word id -> SD floats (slot content)
        self.sgrad = {}                        # word id -> per-byte accumulated exact credit
        self.htail = 0; self.cur = 0; self.phase = 0
        self.sbase = self.NM
        self.wh = FNV0                 # rolling FNV-1a over lowercased word bytes
        self.last_word = 0
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
        if arm == "scrambled":
            self._sr = random.Random(1234)

    def _octx(self, k):
        b = ORDERS[k]
        mask = (1 << (8 * b)) - 1 if b else 0
        return (self.phase, self.cur, (self.htail & mask) if b else 0)

    # ---- word id of the [A-Za-z0-9] run ENDING at byte b (prefix-visible); 0 = none ----
    def _word_after(self, b):
        if is_wb(b):
            self.wh = ((self.wh ^ (b | 32)) * 0x01000193) & 0xFFFFFFFF
            return self.wh | (1 << 31)     # never 0
        self.wh = FNV0
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
            r = random.Random((wid * 40503) & 0x7FFFFFFF)
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
        if self.arm in SLOT_ARMS:
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
        if self.arm == "slots":
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
        self.cur = (self.cur << 1) | y; self.phase += 1
        if self.phase == 8:
            self._byte_end(learn)
        return cost

    def _byte_end(self, learn):
        b = self.cur & 0xFF
        wid = self._word_after(b)
        if self.arm in SLOT_ARMS and wid:
            sl = self.slots
            if wid in sl:
                sl.remove(wid)
            else:
                del sl[-1]
            sl.insert(0, wid)
        if self.arm == "slots" and self.sgrad:
            for w_id, gr in self.sgrad.items():
                v = self.semb.get(w_id)
                if v is None:
                    continue
                for k in range(SD):
                    e = v[k] - self.lr_s * gr[k]
                    v[k] = 1.0 if e > 1.0 else (-1.0 if e < -1.0 else e)
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
        self.htail = ((self.htail << 8) | b) & ((1 << 48) - 1); self.cur = 0; self.phase = 0

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
def run_arm(arm, sizes, seed=0):
    train_all = open("data/wt103_train.txt", "rb").read()
    test_all = open("data/wt103_test.txt", "rb").read()
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
    table = model.semb if model.arm in ("slots", "slotsr") else getattr(model, "emb", {})
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
ARMS = ["baseline", "bitsonly", "randemb", "learned", "slotsr", "slots", "scrambled"]


def report(results, sizes, secs):
    by = {arm: dict(rows) for arm, rows, _, _ in results}
    print()
    hdr = (f"{'train':>8}" + "".join(f"{a:>10}" for a in ARMS))
    print(hdr); print("-" * len(hdr))
    for kb in sizes:
        print(f"{kb:>6}KB" + "".join(f"{by[a][kb]:>10.4f}" for a in ARMS))
    print("-" * len(hdr))
    def series(f):
        return [round(f(by, kb), 4) for kb in sizes]
    # primary (v4): the word-slot channel
    help_s = series(lambda B, kb: B["baseline"][kb] - B["slots"][kb])         # >0 slots help
    chan = series(lambda B, kb: B["bitsonly"][kb] - B["slots"][kb])           # >0 slot channel works
    sem_s = series(lambda B, kb: B["slotsr"][kb] - B["slots"][kb])            # >0 MEANING vs identity
    noi_s = series(lambda B, kb: B["scrambled"][kb] - B["slots"][kb])         # >0 real signal
    # secondary (v3): the EMA embedding projection
    help_e = series(lambda B, kb: B["baseline"][kb] - B["learned"][kb])
    sem_e = series(lambda B, kb: B["randemb"][kb] - B["learned"][kb])
    print(f"SLOTS  vs baseline (help)         : {help_s}")
    print(f"SLOTS  vs bitsonly (channel)      : {chan}")
    print(f"SLOTS  vs slotsr (MEANING)        : {sem_s}")
    print(f"SLOTS  vs scrambled (noise floor) : {noi_s}")
    print(f"EMAemb vs baseline (help)         : {help_e}")
    print(f"EMAemb vs randemb (meaning)       : {sem_e}")
    print()
    crossed = any(h > 0.001 for h in help_s)
    semantic = sem_s[-1] > 0.002 and sum(1 for s in sem_s if s > 0) >= len(sem_s) // 2 + 1
    real = all(n > 0.003 for n in noi_s)
    print("VERDICT:", end=" ")
    if crossed and semantic and real:
        print("CROSS -- the word-slot semantic state beats the orders baseline on copy-ablated,")
        print("  decontaminated held-out; the win is the SLOT CHANNEL (vs bitsonly) and MEANING")
        print("  beyond lexical identity (vs slotsr). The SS70 wall fell. Next: strong.rs port.")
    elif crossed and real:
        print("LEXICAL -- slots help (beat baseline + noise floor) but slots ~= slotsr: the gain")
        print("  is word-identity binding, not semantic generalization. Half the wall fell: the")
        print("  state now CARRIES words; making the vectors MEAN is the remaining step.")
    elif real and any(s > 0 for s in sem_s) and (help_s[-1] > help_s[0] - 0.001):
        print("PROMISING -- real signal and a semantic margin, crossing needs more scale/capacity.")
    else:
        print("NEGATIVE -- no robust signal beyond the noise floor. Honest; iterate or stop.")
    print(f"\n[{secs:.0f}s total]")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    selftest = "--selftest" in sys.argv
    sizes = [40] if selftest else [150, 450, 1200, 2700]
    print("=" * 104)
    print("PHASE 1 -- SEMANTIC STATE vs the SS70 parity wall (word vectors trained BY compression)")
    print(f"  arms: {' | '.join(ARMS)}   sizes(KB): {sizes}")
    print("  gate: copy/match OFF, 13-byte-decontaminated held-out wt103, deterministic mask")
    print("=" * 104)
    t0 = time.time()
    if selftest:
        results = [run_arm(a, sizes) for a in ARMS]
    else:
        with ProcessPoolExecutor(max_workers=len(ARMS)) as ex:
            futs = [ex.submit(run_arm, a, sizes) for a in ARMS]
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
