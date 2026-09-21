#!/usr/bin/env python3
"""jevprobe.py -- §96: Jev as a RIVAL DETECTOR on honestmap's job.

REGISTRATION: prereg/96_jev.md, committed ALONE at 03a624a before any scored call.

  The question honestmap exists to answer is "can the bit-native engine tell STRUCTURED-but-high-
  entropy bytes from TRULY-RANDOM bytes, where order-0 entropy is blind, and does it beat gzip?"
  Its own docstring sets the standard: "If a 20-line order-2 script matches it, there is no moat
  and we say so." A frozen commercial decision model is a stronger rival than a 20-line script.
  §96 adds Jev as ONE MORE COLUMN. Nothing is integrated into the machine.

WHAT IS FIXED BY THE REGISTRATION AND MUST NOT BE EDITED AFTER A SCORE IS SEEN:
  * INSTRUCTION (below), verbatim;
  * hex encoding -- measured before any score existed: on a blob with 250 repeated 16-byte ECB
    blocks base64 preserves 101 and hex preserves 250, because 16 mod 3 = 1 shifts the base64
    phase between successive blocks;
  * 8 windows x 2048 bytes per blob at evenly spaced deterministic offsets;
  * AUC as the primary metric, blob-clustered bootstrap (windows within a blob are NOT independent
    -- §93 and §95 were both corrected for this class of error at the document level).

DETERMINISM: honestmap.corpus() calls os.urandom, so every run builds different blobs and nothing
could be cached or reproduced. §96 substitutes a seeded PRNG and records each blob's sha256.

CACHING is load-bearing, not an optimisation: every response is cached by sha256 of the exact
request body and the cache is committed, so every number replays offline with no key and no
network. Without it this would be the only ledger section not re-derivable from a clone -- the
defect §92 charged §87-§91 with.

  python jevprobe.py --build          # deterministic corpus + windows, print shas, NO network
  python jevprobe.py --run            # score every window (cache-first), then report
  python jevprobe.py --report         # recompute the report from cache only, offline
"""
import os, sys, math, json, gzip, lzma, base64, hashlib, hmac, struct, time, random
import subprocess, tempfile, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import honestmap as HM

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
CACHE = os.path.join(HERE, "_96", "cache")
SEED = 20260921
WIN, NWIN = 2048, 8

# ---- REGISTERED VERBATIM (prereg/96_jev.md §3). Editing this after a score invalidates the run.
INSTRUCTION = ("These bytes are cryptographically random, with no repeated blocks, "
               "no periodic structure, and no encoding pattern.")
SCORE_LEVELS = ["obvious repeated structure", "some periodic or encoded pattern",
                "unclear", "mostly random-looking", "cryptographically random"]


# ---------------------------------------------------------------- deterministic corpus
def _rb(rng, n):
    return bytes(rng.randrange(256) for _ in range(n)) if n < 64 else rng.randbytes(n)


def copy_packed_det(n, rng):
    """honestmap.copy_packed with the PRNG seeded; ~78% literal repeats of one random chunk."""
    chunk = rng.randbytes(512)
    out = bytearray()
    while len(out) < n:
        out += chunk if rng.randrange(256) < 200 else rng.randbytes(512)
    return bytes(out[:n])


def corpus_det(n=48 * 1024):
    """honestmap.corpus(), byte-for-byte, with every os.urandom replaced by a seeded PRNG."""
    rng = random.Random(SEED)
    t = HM.real_text(n)
    return [
        ("english text",         "low-entropy", t),
        ("base64(text)",         "STRUCTURED",  base64.b64encode(t)[:n]),
        ("base64(random)",       "STRUCTURED",  base64.b64encode(rng.randbytes(n))[:n]),
        ("XOR repeat-key(text)", "STRUCTURED",  HM.xor_repeat(t)),
        ("ECB(structured recs)", "STRUCTURED",  HM.ecb_like(HM.structured_records(n))[:n]),
        ("copy/packed",          "STRUCTURED",  copy_packed_det(n, rng)),
        ("gzip(text)",           "random-ish",  gzip.compress(HM.real_text(300 * 1024), 9, mtime=0)[:n]),
        ("lzma(text)",           "random-ish",  lzma.compress(HM.real_text(300 * 1024), preset=9)[:n]),
        ("AES-CTR(text)",        "RANDOM",      HM.aes_ctr_like(t)),
        ("urandom",              "RANDOM",      rng.randbytes(n)),
    ]


def windows(b):
    """8 windows of 2048 B at evenly spaced deterministic offsets."""
    span = len(b) - WIN
    return [b[(span * i) // (NWIN - 1):(span * i) // (NWIN - 1) + WIN] for i in range(NWIN)]


# ---------------------------------------------------------------- detectors
def engine_bpb(b):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(b); path = f.name
    try:
        out = subprocess.run([HM.STRONG, path, "0", "21"], capture_output=True, text=True).stdout
        import re
        m = re.search(r"whole-stream = ([0-9.]+)", out)
        return float(m.group(1)) * 8.0 if m else float("nan")
    finally:
        os.unlink(path)


def _body(hexstr, kind):
    if kind == "noul":
        q = {"type": "noul", "instructions": INSTRUCTION}
    else:
        q = {"type": "score", "instructions": INSTRUCTION, "criteria": SCORE_LEVELS}
    return {"state": hexstr, "model": MODEL, "questions": {"q": q}}


def jev(hexstr, kind="noul", key=None, tries=5):
    """cache-first. Returns (value, resolved_model, usage) or (None, None, None) on failure."""
    body = json.dumps(_body(hexstr, kind), sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(body.encode()).hexdigest()
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, h + ".json")
    if os.path.exists(path):
        d = json.load(open(path))
    else:
        if key is None:
            return None, None, None                      # --report is offline by construction
        d = None
        for a in range(tries):
            req = urllib.request.Request(ENDPOINT, data=body.encode(),
                                         headers={"Authorization": f"Bearer {key}",
                                                  "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 529) and a < tries - 1:
                    time.sleep(2 ** a); continue
                sys.stderr.write(f"  HTTP {e.code} on {h[:8]}\n"); return None, None, None
            except Exception as e:
                if a < tries - 1:
                    time.sleep(2 ** a); continue
                sys.stderr.write(f"  {type(e).__name__} on {h[:8]}\n"); return None, None, None
        if d is None:
            return None, None, None
        json.dump(d, open(path, "w"))
    ans = d["answers"]["q"]
    v = ans["noul"] if kind == "noul" else ans["score"]
    return v, d.get("model"), d.get("usage")


# ---------------------------------------------------------------- metrics
def auc(pos, neg):
    """P(a random POS scores above a random NEG), ties counted as 1/2. Scale-free."""
    if not pos or not neg:
        return float("nan")
    w = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def auc_boot(rows, det, reps=2000, seed=SEED):
    """blob-clustered: resample BLOBS with replacement, then windows within each blob."""
    byblob = {}
    for r in rows:
        byblob.setdefault((r["blob"], r["kind"]), []).append(r[det])
    P = [k for k in byblob if k[1] == "RANDOM"]
    N = [k for k in byblob if k[1] == "STRUCTURED"]
    obs = auc([v for k in P for v in byblob[k]], [v for k in N for v in byblob[k]])
    rng = random.Random(seed); out = []
    for _ in range(reps):
        pv, nv = [], []
        for _ in range(len(P)):
            b = byblob[P[rng.randrange(len(P))]]
            pv += [b[rng.randrange(len(b))] for _ in b]
        for _ in range(len(N)):
            b = byblob[N[rng.randrange(len(N))]]
            nv += [b[rng.randrange(len(b))] for _ in b]
        out.append(auc(pv, nv))
    out.sort()
    return obs, out[int(0.025 * reps) - 1], out[int(0.975 * reps)]


def paired_auc_boot(rows, d1, d2, reps=20000, seed=SEED):
    """PAIRED blob-clustered CI on A(d1) - A(d2). The statistic the registered branches turn on,
    and which the first version of this file did not implement -- §96's own §92-class defect."""
    byblob = {}
    for r in rows:
        byblob.setdefault((r["blob"], r["kind"]), []).append(r)
    P = [k for k in byblob if k[1] == "RANDOM"]
    N = [k for k in byblob if k[1] == "STRUCTURED"]
    def A(det, pk, nk):
        return auc([w[det] for k in pk for w in byblob[k]], [w[det] for k in nk for w in byblob[k]])
    obs = A(d1, P, N) - A(d2, P, N)
    rng = random.Random(seed); out = []
    for _ in range(reps):
        pk = [P[rng.randrange(len(P))] for _ in P]
        nk = [N[rng.randrange(len(N))] for _ in N]
        out.append(A(d1, pk, nk) - A(d2, pk, nk))
    out.sort()
    zero = sum(1 for v in out if v == 0.0) / reps
    return obs, out[int(0.025 * reps) - 1], out[int(0.975 * reps)], zero


# ---------------------------------------------------------------- driver
def build():
    C = corpus_det()
    print(f"{'blob':<24}{'kind':<13}{'sha256[:16]':<18}{'windows'}")
    for label, kind, b in C:
        print(f"{label:<24}{kind:<13}{hashlib.sha256(b).hexdigest()[:16]:<18}{len(windows(b))}")
    return C


def run(live=True):
    key = None
    if live:
        for ln in open(os.path.join(HERE, ".env")):
            if ln.startswith("TYPESAFE_API_KEY="):
                key = ln.split("=", 1)[1].strip()
        if not key:
            print("no TYPESAFE_API_KEY in .env"); return
    C = corpus_det()
    rows, fails, calls, toks, models = [], 0, 0, 0, set()
    for label, kind, b in C:
        for i, w in enumerate(windows(b)):
            hx = w.hex()
            v, m, u = jev(hx, "noul", key)
            if v is None:
                fails += 1; continue
            if u:
                calls += 1; toks += u.get("input_tokens", 0); models.add(m)
            rows.append({"blob": label, "kind": kind, "win": i,
                         "jev": v, "order0": HM.order0(w), "order1": HM.cond_entropy(w, 1),
                         "gzip": HM.gzip_bpb(w), "engine": engine_bpb(w)})
        done = [r for r in rows if r["blob"] == label]
        if done:
            print(f"  {label:<24} jev {sum(r['jev'] for r in done)/len(done):.3f}   "
                  f"engine {sum(r['engine'] for r in done)/len(done):.2f}   "
                  f"gzip {sum(r['gzip'] for r in done)/len(done):.2f}")
    os.makedirs(os.path.join(HERE, "_96"), exist_ok=True)
    json.dump({"blob_sha256": {l: hashlib.sha256(b).hexdigest() for l, _k, b in C},
               "rows": rows, "fails": fails, "new_calls": calls, "input_tokens": toks,
               "resolved_models": sorted(m for m in models if m), "seed": SEED,
               "instruction": INSTRUCTION, "win": WIN, "nwin": NWIN},
              open(os.path.join(HERE, "_96", "rows.json"), "w"), indent=1)
    print(f"\n  new API calls {calls}  input_tokens {toks:,}  failures {fails}  "
          f"resolved {sorted(m for m in models if m)}")
    report()


def report():
    d = json.load(open(os.path.join(HERE, "_96", "rows.json")))
    rows = d["rows"]
    n_exp = len(corpus_det()) * NWIN
    print("\n" + "=" * 96)
    print("§96  Jev as a RIVAL DETECTOR -- AUC, RANDOM vs STRUCTURED windows (registered primary)")
    print("=" * 96)
    print(f"  resolved model {d['resolved_models']}   windows {len(rows)}/{n_exp}   "
          f"failures {d['fails']}")
    if d["fails"] > 0.10 * n_exp:
        print("  BRANCH 6: >10% of calls failed -> THE RUN IS VOID. No branch fires."); return
    print(f"\n  {'blob':<24}{'kind':<13}{'jev':>8}{'order0':>9}{'order1':>9}{'gzip':>8}{'engine':>9}")
    for label, kind, _ in corpus_det():
        s = [r for r in rows if r["blob"] == label]
        if not s:
            continue
        f = lambda k: sum(r[k] for r in s) / len(s)
        print(f"  {label:<24}{kind:<13}{f('jev'):>8.3f}{f('order0'):>9.2f}{f('order1'):>9.2f}"
              f"{f('gzip'):>8.2f}{f('engine'):>9.2f}")
    print(f"\n  {'detector':<12}{'AUC':>8}   {'blob-clustered 95% CI':>26}")
    A = {}
    for det in ("jev", "order0", "order1", "gzip", "engine"):
        o, lo, hi = auc_boot(rows, det)
        A[det] = o
        print(f"  {det:<12}{o:>8.3f}   [{lo:>10.3f},{hi:>10.3f}]")
    # the registered fairness control: English-as-hex vs urandom-as-hex
    eng = [r["jev"] for r in rows if r["blob"] == "english text"]
    rnd = [r["jev"] for r in rows if r["blob"] == "urandom"]
    ctrl = auc(rnd, eng)
    print(f"\n  FAIRNESS CONTROL (registered): urandom-as-hex vs English-as-hex, Jev AUC {ctrl:.3f}")
    print(f"    branch 5 fires below 0.700 -> the ENCODING is the binding constraint, and every")
    print(f"    Jev number is scoped to 'from hex' rather than read as a capability claim.")
    print(f"\n  REGISTERED BRANCHES (prereg/96_jev.md §4):")
    print(f"    1 jev>=engine (CI excl 0)  2 jev>=gzip but <engine  3 jev<gzip  4 jev CI incl 0.5")
    print(f"    jev {A['jev']:.3f}   gzip {A['gzip']:.3f}   engine {A['engine']:.3f}")
    print("=" * 96)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--build":
        build()
    elif a[0] == "--run":
        run(live=True)
    elif a[0] == "--report":
        report()
    else:
        print(__doc__)
