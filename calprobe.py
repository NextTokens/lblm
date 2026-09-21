"""calprobe.py -- §92A: is the machine's probability actually calibrated?

REGISTRATION: prereg/92A_calibration.md, written before any number in this file was computed.
Read it first. The short version:

  Two live claims in this project assert calibration -- ledger line 1352 ("Confidence is
  calibrated: accuracy@commit rises monotonically 0.57 -> 0.95") and, at the time of writing,
  the poc.py dashboard ("calibrated -- commits/abstains at a fixed tau, measurably monotone";
  the v2 rewrite has since dropped the word). Both describe a SELECTIVE-PREDICTION
  curve, which is monotone for any model with an informative confidence ORDERING, including one
  that is uniformly 2x overconfident. Calibration is the different property that among the bits
  assigned probability ~q, a 1 occurs with frequency ~q. Before this file, the repository
  contained zero calibration measurements of any kind.

WHAT IT MEASURES
  Prequential (online, one-pass) calibration of the per-bit forecast, on the adopted rail and the
  standard decontaminated held-out slice, so the population is the one every §86-§91 corpus number
  is measured on.

  * ECE at 10 and 20 bins, on BOTH uniform-width and equal-mass (quantile) bins. Bit forecasts are
    extremely concentrated near 0 and 1; uniform bins put nearly all mass in two cells and can hide
    interior miscalibration. Both are pre-committed so neither can be chosen after the fact.
  * MCE over bins with n >= 100; Brier score with its reliability/resolution/uncertainty split.
  * THE NOISE FLOOR, simulated from the machine's OWN predictions -- never looked up by sample size.
    ECE is biased upward at finite n because each bin's hit rate is a noisy estimate, and the size
    of that bias depends on how CONCENTRATED the forecasts are, not on n. So:
        y_sim[i] ~ Bernoulli(p[i])   (calibrated by construction)  ->  ECE(p, y_sim)
    The headline is the ratio ECE / floor. An ECE sitting at its floor measures nothing in either
    direction and is reported as such rather than as evidence of calibration.
    Computed two ways that must agree: an O(N) analytic form (per bin, E|sum y - sum p| ~
    sqrt(2*V/pi) with V = sum p(1-p), by CLT) and a Monte Carlo, checked against each other in S3.
  * The selective-prediction curve alongside, so ledger 1352's own claim is reproduced rather than
    argued with, and the two properties can be read side by side.
  * SECONDARY, byte level: the next-byte top-1 decision, which is what poc.py actually displays.
    Reported on the true byte probability P = prod(p_bit) AND on poc.py's displayed quantity
    P**(1/8), because those are not the same number and the dashboard thresholds the second one.

WHAT IT DOES NOT MEASURE
  Anything about a window-level anomaly score. Per-bit calibration does not make a surprisal total
  over a window calibrated for a decision; that needs its own null distribution over windows, which
  does not exist anywhere in this project. No branch of §92A licenses a monitor-alarm claim.

NO CHANGE TO THE MODEL. p is recovered from the cost step() already returns
(cost = -log2(p if y else 1-p)), so wstate.py is not touched and no bit-identity risk is taken.
S1 verifies the recovery against predict() exactly.

RUNNING IT
  python calprobe.py --selftest                     S1, S2, S3 (fast, ~40 KB)
  python calprobe.py --run ARM TRAIN_KB OUT.npz     capture (p, y) over the held-out slice
  python calprobe.py --report OUT.npz               the registered verdict
  python calprobe.py --bytes ARM TRAIN_KB STRIDE    the byte-level secondary
Env for every model call, the adopted rail: WNS=1 WSLOTMODE=word WSLOTS=32 WVBITS2=22
"""
import os, sys, math, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import wstate as W


# ----------------------------------------------------------------------------------------------------
# capture
# ----------------------------------------------------------------------------------------------------
def capture(arm, train_kb, seed=0, train_path="data/wt103_train.txt", test_path="data/wt103_test.txt",
            nomask=False):
    """run the adopted protocol and record (p, y) for every SCORED bit of the held-out slice.

    Protocol is run_arm()'s, byte for byte: train prefix at train_kb, held-out slice
    min(400 KB, train_kb/2), decontaminated with clean_mask_det(tr, te, 13). Learning stays ON
    during the measured slice exactly as the corpus numbers have it -- this is an ONLINE forecast
    and turning learning off would measure a different machine.
    """
    train_all = open(os.path.join(HERE, train_path), "rb").read()
    test_all = open(os.path.join(HERE, test_path), "rb").read()
    tr = train_all[:train_kb * 1024]
    te = test_all[:min(400 * 1024, train_kb * 1024 // 2)]
    # AMENDMENT 1: clean_mask_det picks bytes by looking at the text being scored, so the measured
    # population is outcome-selected. nomask=True scores EVERY byte. Model state is identical either
    # way -- learning already runs on every byte; the mask only chooses what is RECORDED.
    mask = bytearray([1]) * len(te) if nomask else W.clean_mask_det(tr, te, 13)

    m = W.Model(arm=arm, seed=seed)
    t0 = time.time()
    m.run(tr, learn=True)
    t_train = time.time() - t0

    n_bits = 8 * sum(1 for bp in range(len(te)) if (bp < len(mask) and mask[bp]))
    p_arr = np.empty(n_bits, dtype=np.float64)
    y_arr = np.empty(n_bits, dtype=np.uint8)

    i = 0
    tot = 0.0
    nb = 0
    t0 = time.time()
    for bp, b in enumerate(te):
        scored = (bp < len(mask) and mask[bp])
        bc = 0.0
        for j in range(7, -1, -1):
            y = (b >> j) & 1
            c = m.step(y, True)
            bc += c
            if scored:
                # cost = -log2(p if y==1 else 1-p)  ->  invert it exactly
                q = 2.0 ** (-c)
                p_arr[i] = q if y == 1 else 1.0 - q
                y_arr[i] = y
                i += 1
        if scored:
            tot += bc
            nb += 1
    assert i == n_bits, (i, n_bits)
    return {"p": p_arr, "y": y_arr, "bpb": tot / max(1, nb), "n_bytes_scored": nb,
            "arm": arm, "train_kb": train_kb, "test_bytes": len(te),
            "t_train": t_train, "t_measure": time.time() - t0,
            "nomask": bool(nomask), "retained": nb / max(1, len(te)),
            "env": {k: os.environ.get(k) for k in
                    ("WNS", "WSLOTMODE", "WSLOTS", "WVBITS2", "WSELTAG", "WSELTOPM")}}


# ----------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------
def _bin_edges_uniform(nbins):
    return np.linspace(0.0, 1.0, nbins + 1)


def _bin_edges_quantile(p, nbins):
    qs = np.linspace(0.0, 1.0, nbins + 1)
    e = np.unique(np.quantile(p, qs))
    e[0], e[-1] = 0.0, 1.0
    return e


def ece(p, y, edges):
    """expected calibration error + the per-bin table. Weighted by bin mass, as standard."""
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    N = len(p)
    tot = 0.0
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        conf = float(p[sel].mean())
        acc = float(y[sel].mean())
        tot += (n / N) * abs(acc - conf)
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "n": n, "mean_p": conf, "obs_freq": acc, "gap": acc - conf})
    return tot, rows


def mce(rows, min_n=100):
    r = [abs(x["gap"]) for x in rows if x["n"] >= min_n]
    return max(r) if r else float("nan")


def brier(p, y):
    """Brier + Murphy decomposition: reliability - resolution + uncertainty."""
    bs = float(np.mean((p - y) ** 2))
    edges = _bin_edges_uniform(20)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    N = len(p)
    ybar = float(y.mean())
    rel = res = 0.0
    for b in range(len(edges) - 1):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        pk = float(p[sel].mean())
        ok = float(y[sel].mean())
        rel += n * (pk - ok) ** 2
        res += n * (ok - ybar) ** 2
    return {"brier": bs, "reliability": rel / N, "resolution": res / N,
            "uncertainty": ybar * (1 - ybar)}


def ece_floor_analytic(p, edges):
    """E[ECE] under the null that the model IS calibrated, computed from p alone.

    Per bin the contribution to ECE is |sum(y) - sum(p)| / N. Under the null sum(y) is
    Poisson-binomial with mean sum(p) and variance V = sum p(1-p), so by CLT the expected
    absolute deviation is sqrt(2V/pi). O(N), no simulation.
    """
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    N = len(p)
    tot = 0.0
    for b in range(len(edges) - 1):
        sel = idx == b
        if not sel.any():
            continue
        V = float((p[sel] * (1.0 - p[sel])).sum())
        tot += math.sqrt(2.0 * V / math.pi)
    return tot / N


def ece_floor_mc(p, edges, trials=200, seed=20260919):
    """the same floor by simulation: y ~ Bernoulli(p), i.e. calibrated by construction."""
    rng = np.random.default_rng(seed)
    out = np.empty(trials)
    for t in range(trials):
        ys = (rng.random(len(p)) < p).astype(np.uint8)
        out[t], _ = ece(p, ys, edges)
    return {"mean": float(out.mean()), "p95": float(np.quantile(out, 0.95)),
            "sd": float(out.std()), "trials": trials}


def selective_curve(p, y, taus=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)):
    """ledger 1352's statistic, at bit level: commit when max(p,1-p) >= tau.

    Reported so the two properties can be read together. A monotone rise here is NOT calibration.
    """
    conf = np.maximum(p, 1.0 - p)
    pred = (p >= 0.5).astype(np.uint8)
    rows = []
    for t in taus:
        sel = conf >= t
        n = int(sel.sum())
        rows.append({"tau": t, "coverage": n / len(p),
                     "accuracy": float((pred[sel] == y[sel]).mean()) if n else float("nan"),
                     "mean_conf": float(conf[sel].mean()) if n else float("nan")})
    return rows


# ----------------------------------------------------------------------------------------------------
# self-tests
# ----------------------------------------------------------------------------------------------------
def selftest(train_kb=40):
    print("S1  p recovered from cost == p from predict() (predict() is pure, self-test S1 in wstate)")
    train_all = open(os.path.join(HERE, "data/wt103_train.txt"), "rb").read()
    tr = train_all[:train_kb * 1024]
    te = open(os.path.join(HERE, "data/wt103_test.txt"), "rb").read()[:4096]
    m = W.Model(arm="selleanu", seed=0)
    m.run(tr, learn=True)
    worst = 0.0
    n = 0
    for b in te[:1500]:
        for j in range(7, -1, -1):
            y = (b >> j) & 1
            p_direct, _ = m.predict()            # pure: safe to call before step()
            c = m.step(y, True)
            q = 2.0 ** (-c)
            p_rec = q if y == 1 else 1.0 - q
            worst = max(worst, abs(p_rec - p_direct))
            n += 1
    print(f"    {n} bits, max |p_recovered - p_predict| = {worst:.3e}   "
          f"{'PASS' if worst < 1e-9 else 'FAIL'}")

    print("S2  bpb recomputed from the captured (p, y) == the model's own returned bpb")
    d = capture("selleanu", train_kb)
    p, y = d["p"], d["y"]
    bpb_recon = float(-(np.log2(np.where(y == 1, p, 1.0 - p))).sum()) / d["n_bytes_scored"]
    print(f"    model {d['bpb']:.9f}  reconstructed {bpb_recon:.9f}  "
          f"delta {abs(bpb_recon - d['bpb']):.3e}   "
          f"{'PASS' if abs(bpb_recon - d['bpb']) < 1e-7 else 'FAIL'}")

    print("S4  the beam's served p along the TRUE byte path == step()'s own cost for that byte")
    print("    (the check that would have caught the htail bug; S1-S3 do not perform it)")
    m2 = W.Model(arm="selleanu", seed=0)
    m2.run(tr, learn=True)
    worst4 = 0.0
    for b in te[:400]:
        save = (m2.cur, m2.phase, m2.htail)
        spec = 0.0
        for j in range(7, -1, -1):
            bit = (b >> j) & 1
            p4, _ = m2.predict()                # p4, not p: S3 below reads S2's captured p array
            spec += -math.log2(p4 if bit == 1 else 1.0 - p4)
            m2.cur = ((m2.cur << 1) | bit) & 0xFF
            m2.phase += 1                       # htail deliberately NOT advanced
        m2.cur, m2.phase, m2.htail = save
        # learn=False: the speculative pass does not learn, so the comparator must not either.
        # With learn=True the engine updates its weights after every bit and is a DIFFERENT
        # predictor from bit 2 onward -- equality is then impossible and the gate can never pass.
        real = sum(m2.step((b >> j) & 1, False) for j in range(7, -1, -1))
        worst4 = max(worst4, abs(spec - real))
    print(f"    400 bytes, max |speculative - real| = {worst4:.3e}   "
          f"{'PASS' if worst4 < 1e-9 else 'FAIL'}")

    print("S3  analytic ECE noise floor == Monte Carlo floor")
    for nb in (10, 20):
        e = _bin_edges_quantile(p, nb)
        a = ece_floor_analytic(p, e)
        mc = ece_floor_mc(p, e, trials=100)
        rel = abs(a - mc["mean"]) / max(mc["mean"], 1e-12)
        print(f"    quantile {nb:>2} bins: analytic {a:.6f}  MC {mc['mean']:.6f} "
              f"(sd {mc['sd']:.6f})  rel.diff {rel:.3%}   {'PASS' if rel < 0.10 else 'FAIL'}")
    return d


# ----------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------
def report(d):
    p, y = d["p"], d["y"]
    N = len(p)
    print(f"\n{'=' * 96}")
    print(f"§92A CALIBRATION -- arm {d['arm']}  train {d['train_kb']} KB  "
          f"held-out {d['test_bytes']/1024:.0f} KB  scored {d['n_bytes_scored']} bytes")
    print(f"  bits measured {N:,}   bpb {d['bpb']:.6f}   base rate P(bit=1) {float(y.mean()):.4f}")
    print(f"{'=' * 96}")

    print(f"\n  {'binning':<18} {'ECE':>9} {'floor':>9} {'floor p95':>10} {'ratio':>7} "
          f"{'MCE':>8}   verdict")
    headline = None
    for name, mk in (("uniform", _bin_edges_uniform), ("quantile", _bin_edges_quantile)):
        for nb in (10, 20):
            e = mk(nb) if name == "uniform" else mk(p, nb)
            v, rows = ece(p, y, e)
            fl = ece_floor_analytic(p, e)
            mc = ece_floor_mc(p, e, trials=200)
            r = v / max(fl, 1e-12)
            verdict = ("AT FLOOR - measures nothing" if v < mc["p95"]
                       else f"{r:.1f}x floor - real")
            print(f"  {name+' '+str(nb):<18} {v:9.5f} {fl:9.5f} {mc['p95']:10.5f} "
                  f"{r:7.2f} {mce(rows):8.5f}   {verdict}")
            if name == "quantile" and nb == 10:
                headline = (v, fl, mc, r, rows)

    v, fl, mc, r, rows = headline
    print(f"\n  REGISTERED HEADLINE (quantile, 10 bins): ECE {v:.5f}, floor {fl:.5f}, "
          f"ratio r = {r:.2f}")
    print(f"  Registered branch: r <= 1.2 and ECE < floor_p95 -> calibrated as far as this resolves")
    if r <= 1.2 and v < mc["p95"]:
        print(f"  -> BRANCH 1: calibrated at bit level within this test's resolution.")
    else:
        signed = sum((x["n"] / N) * x["gap"] for x in rows)
        toward = sum((x["n"] / N) * (abs(x["mean_p"] - 0.5) - abs(x["obs_freq"] - 0.5)) for x in rows)
        direction = "OVERCONFIDENT" if toward > 0 else "UNDERCONFIDENT"
        print(f"  -> BRANCH 2/3: NOT calibrated. r = {r:.2f}. Direction: {direction} "
              f"(mean |p-0.5| exceeds mean |obs-0.5| by {toward:+.5f}); signed gap {signed:+.5f}")

    print(f"\n  per-bin reliability table (quantile, 10 bins):")
    print(f"    {'range':<22} {'n':>12} {'mean p':>9} {'observed':>9} {'gap':>9}")
    for x in rows:
        print(f"    [{x['lo']:.6f},{x['hi']:.6f}] {x['n']:>12,} {x['mean_p']:9.5f} "
              f"{x['obs_freq']:9.5f} {x['gap']:+9.5f}")

    b = brier(p, y)
    print(f"\n  Brier {b['brier']:.6f} = reliability {b['reliability']:.6f} "
          f"- resolution {b['resolution']:.6f} + uncertainty {b['uncertainty']:.6f}")

    print(f"\n  SELECTIVE PREDICTION (ledger 1352's statistic, at bit level) -- reported ALONGSIDE,")
    print(f"  not as calibration. A monotone rise here is produced by any informative ordering.")
    print(f"    {'tau':>6} {'coverage':>10} {'accuracy':>10} {'mean conf':>10} {'conf - acc':>11}")
    for x in selective_curve(p, y):
        d_ = x["mean_conf"] - x["accuracy"]
        print(f"    {x['tau']:6.2f} {x['coverage']:10.4f} {x['accuracy']:10.4f} "
              f"{x['mean_conf']:10.4f} {d_:+11.4f}")
    print(f"\n  Read the last column: if it is ~0 the confidence is honest at that threshold; if it")
    print(f"  is positive the model is overconfident there, however monotone the accuracy column is.")


# ----------------------------------------------------------------------------------------------------
# the byte-level secondary -- what poc.py actually displays
# ----------------------------------------------------------------------------------------------------
def capture_bytes(arm, train_kb, stride=50, width=6, seed=0, nomask=False):
    """next-BYTE top-1 decision, at a deterministic stride over the same held-out slice.

    The poc.py dashboard USED to threshold `conf = 2 ** (sum log2 p_bit / 8)` -- the GEOMETRIC
    MEAN per bit, i.e. P(byte) ** (1/8), not P(byte) -- so tau = 0.90 on the displayed quantity was
    tau = 0.9**8 = 0.4305 on the real one. poc.py v2 displays 2**(-lp) and has no tau gate, so the
    gate block below measures the SUPERSEDED behaviour; both quantities are still recorded here
    against the same event ("is top-1 the byte that actually came next").

    The beam is width-limited, so P(top1) is a lower bound on the true argmax probability; that
    direction makes the overconfidence finding conservative, not generous.
    """
    train_all = open(os.path.join(HERE, "data/wt103_train.txt"), "rb").read()
    test_all = open(os.path.join(HERE, "data/wt103_test.txt"), "rb").read()
    tr = train_all[:train_kb * 1024]
    te = test_all[:min(400 * 1024, train_kb * 1024 // 2)]
    # AMENDMENT 1 applies to the secondary verbatim: clean_mask_det selects bytes by looking
    # at the text being scored, so nomask=True scores every byte. Same branches, on ECE.
    mask = bytearray([1]) * len(te) if nomask else W.clean_mask_det(tr, te, 13)
    m = W.Model(arm=arm, seed=seed)
    m.run(tr, learn=True)

    P_top, hit = [], []
    for bp, b in enumerate(te):
        if (bp < len(mask) and mask[bp]) and bp % stride == 0:
            save = (m.cur, m.phase, m.htail)
            ht0 = m.htail
            beam = [(0.0, m.cur, m.phase, ht0, 0)]
            for _lvl in range(8):
                nxt = []
                for lp, cur, phase, ht, val in beam:
                    for bit in (0, 1):
                        m.cur, m.phase = cur, phase
                        p, _ = m.predict()
                        pb = p if bit else 1.0 - p
                        # htail is NOT advanced: wstate.py:1443 shifts it one BYTE at a time inside
                        # _byte_end, so it is frozen across all eight bits of the byte being coded.
                        # poc.py:peek() shifts it one BIT per bit and so corrupts the order-context
                        # keying on bits 1-7 of every speculative byte. S4 gates this.
                        nxt.append((lp + math.log2(max(pb, 1e-12)),
                                    ((cur << 1) | bit) & 0xFF, phase + 1, ht, (val << 1) | bit))
                nxt.sort(key=lambda x: -x[0])
                beam = nxt[:width]
            m.cur, m.phase, m.htail = save
            lp, _c, _ph, _ht, val = beam[0]
            P_top.append(2.0 ** lp)
            hit.append(1 if val == b else 0)
        for j in range(7, -1, -1):
            m.step((b >> j) & 1, True)
    return np.array(P_top), np.array(hit, dtype=np.uint8)


def report_bytes(P_top, hit, tau=0.90):
    N = len(P_top)
    disp = P_top ** (1.0 / 8.0)          # poc.py:84's displayed "conf"
    print(f"\n{'=' * 96}")
    print(f"§92A SECONDARY -- the byte decision poc.py displays   n = {N:,} sampled bytes")
    print(f"{'=' * 96}")
    print(f"  top-1 accuracy {float(hit.mean()):.4f}   mean P(top1) {float(P_top.mean()):.4f}   "
          f"mean displayed P^(1/8) {float(disp.mean()):.4f}")
    e = _bin_edges_quantile(P_top, 10)
    v, rows = ece(P_top, hit, e)
    fl = ece_floor_analytic(P_top, e)
    mc = ece_floor_mc(P_top, e, trials=500)
    print(f"\n  TRUE P(top1) as a forecast of 'top-1 is correct':")
    print(f"    ECE {v:.5f}  floor {fl:.5f} (p95 {mc['p95']:.5f})  ratio {v/max(fl,1e-12):.2f}")
    print(f"    {'range':<22} {'n':>7} {'mean P':>9} {'observed':>9} {'gap':>9}")
    for x in rows:
        print(f"    [{x['lo']:.5f},{x['hi']:.5f}] {x['n']:>7,} {x['mean_p']:9.5f} "
              f"{x['obs_freq']:9.5f} {x['gap']:+9.5f}")
    sel = disp >= tau
    n = int(sel.sum())
    print(f"\n  poc.py's gate, as shipped: COMMIT when displayed P^(1/8) >= {tau}")
    print(f"    that is COMMIT when the true byte probability >= {tau**8:.4f}")
    print(f"    coverage {n/N:.4f}   accuracy-at-commit {float(hit[sel].mean()) if n else float('nan'):.4f}")
    print(f"    mean DISPLAYED confidence on committed bytes {float(disp[sel].mean()) if n else float('nan'):.4f}")
    print(f"    mean TRUE probability      on committed bytes {float(P_top[sel].mean()) if n else float('nan'):.4f}")
    if n:
        print(f"    -> the dashboard shows {float(disp[sel].mean()):.4f} where the honest number is "
              f"{float(P_top[sel].mean()):.4f} (overstated by "
              f"{float(disp[sel].mean()) - float(P_top[sel].mean()):+.4f})")


def save(d, path):
    np.savez_compressed(path, p=d["p"], y=d["y"],
                        meta=json.dumps({k: v for k, v in d.items() if k not in ("p", "y")}))
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def load(path):
    z = np.load(path, allow_pickle=False)
    d = json.loads(str(z["meta"]))
    d["p"] = z["p"]; d["y"] = z["y"]
    return d


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--selftest":
        d = selftest(*(int(x) for x in a[1:]))
        report(d)
    elif a[0] == "--run":
        d = capture(a[1], int(a[2]))
        save(d, a[3])
        report(d)
    elif a[0] == "--bytes":
        Pt, hit = capture_bytes(a[1], int(a[2]), int(a[3]) if len(a) > 3 else 50,
                                nomask=(len(a) > 4 and a[4] == "nomask"))
        np.savez_compressed("_cal/bytes.npz", P=Pt, hit=hit)
        report_bytes(Pt, hit)
    elif a[0] == "--report":
        report(load(a[1]))
    else:
        print(__doc__)
