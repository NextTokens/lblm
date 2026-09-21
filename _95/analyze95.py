"""§95 analysis -- computes EXACTLY what prereg/95_tag.md registered, and nothing else.

Written while the arms were still running, so no number could steer a choice of statistic.

Run:  python _95/analyze95.py
"""
import os, sys, json, math, collections

HERE = r"D:\Dev\bitexp"
os.chdir(HERE)
sys.path.insert(0, HERE)
import _factprobe as F

ARMS = {
    "ref  b22 tag-OFF": "_93/scope0.json",      # on disk, §93
    "gate b22 tag-OFF": "_95/gate_b22_notag.json",
    "     b22 tag-ON ": "_95/s0_b22_tag.json",
    "     b28 tag-OFF": "_94/s0_b28.json",      # on disk, §94
    "     b28 tag-ON ": "_95/s0_b28_tag.json",
}
REF = "ref  b22 tag-OFF"


def load(p):
    return json.load(open(p))


def sgn(x):
    return 0.0 if x == 0 else (1.0 if x > 0 else -1.0)


def facts(d, docof, nwords):
    return F.populations(d, docof, nwords)["all"]


def key(r):
    return (r["pid"], r["j"], r["cue"])


def summarise(rows):
    """the registered PAIR plus the scale-immune statistic. Never the product alone."""
    n = len(rows)
    srv = [r for r in rows if r["inT_first"] == 1]
    e = [r["e_first"] for r in rows]
    es = [r["e_first"] for r in srv]
    return {
        "n": n,
        "P_served": len(srv) / max(1, n),
        "E_e_given_served": sum(es) / max(1, len(es)),
        "mean_abs_e_served": sum(abs(x) for x in es) / max(1, len(es)),
        "sign_mean": sum(sgn(x) for x in e) / max(1, n),
        "E_sign_given_served": sum(sgn(x) for x in es) / max(1, len(es)),
        "product_e_first": sum(e) / max(1, n),
        "matched_contrast_served": (sum(r["e_first"] - r["e_match_first"] for r in srv)
                                    / max(1, len(srv))),
        "cand_n_served": sum(r["cand_n"] for r in srv) / max(1, len(srv)),
        "a_cue_served": sum(r["a_cue"] for r in srv) / max(1, len(srv)),
    }


def paired(ref_rows, arm_rows, docof, val):
    """paired delta arm - ref on the rows present in BOTH, two-level (primary) + pid (secondary)."""
    a = {key(r): r for r in arm_rows}
    merged = []
    for r in ref_rows:
        o = a.get(key(r))
        if o is None:
            continue
        merged.append({"pid": r["pid"], "j": r["j"], "_d": val(o) - val(r)})
    if not merged:
        return None
    two = F.two_level_boot(merged, docof, lambda r: r["_d"])
    one = F.one_level_boot(merged, lambda r: r["pid"], lambda r: r["_d"])
    return {"n": len(merged), "obs": two[0], "two_lo": two[1], "two_hi": two[2],
            "one_lo": one[1], "one_hi": one[2]}


def la_lm_87(rows):
    """§87's own construction: each arm's OWN served fact events with a served C-MATCH partner.
    LM only exists for kind in (fact, null) -- _factprobe.py:580 -- so fact rows only."""
    sel = [r for r in rows if r["inT_first"] == 1 and r["match_first"]]
    if not sel:
        return float("nan"), 0
    return sum(r["LA"] - r["LM"] for r in sel) / len(sel), len(sel)


def main():
    docof, nwords, ndocs = F.doc_index(400)
    print(f"documents in the 400 KB measured slice: {ndocs}   words {nwords:,}")
    print(f"BOOT={F.BOOT} SEED={F.SEED}  (two-level document bootstrap = REGISTERED PRIMARY)\n")

    for adj_on in (True, False):
        F.ADJ = F.build_adj(400) if adj_on else set()
        tag = "ADJ FILTER ON  (registered §87 protocol)" if adj_on else "ADJ FILTER OFF (§93/§94's published basis)"
        print("=" * 118)
        print(tag)
        print("=" * 118)

        D, R = {}, {}
        for name, path in ARMS.items():
            if not os.path.exists(path):
                print(f"  {name}: MISSING {path}")
                continue
            d = load(path)
            D[name] = d
            R[name] = facts(d, docof, nwords)

        if REF not in R:
            return

        print(f"\n{'arm':18} {'n':>6} {'P(srv)':>8} {'E[e|srv]':>10} {'mean|e|':>9} "
              f"{'SIGN-MEAN':>11} {'E[sgn|srv]':>11} {'product':>9} {'e-e_match':>10} {'bpb':>10} {'evict':>9}")
        for name in ARMS:
            if name not in R:
                continue
            s = summarise(R[name])
            d = D[name]
            print(f"{name:18} {s['n']:>6} {s['P_served']:>8.4f} {s['E_e_given_served']:>10.4f} "
                  f"{s['mean_abs_e_served']:>9.4f} {s['sign_mean']:>+11.4f} {s['E_sign_given_served']:>+11.4f} "
                  f"{s['product_e_first']:>+9.4f} {s['matched_contrast_served']:>+10.4f} "
                  f"{d['test_bpb']:>10.6f} {d.get('evictions', 0):>9,}")

        print(f"\n  PAIRED vs {REF}  --  PRIMARY statistic = sign(e_first), two-level CI")
        print(f"  {'arm':18} {'n':>6} {'Dsign':>9} {'two-level CI':>22} {'pid CI':>22}")
        for name in ARMS:
            if name == REF or name not in R:
                continue
            p = paired(R[REF], R[name], docof, lambda r: sgn(r["e_first"]))
            if p:
                print(f"  {name:18} {p['n']:>6} {p['obs']:>+9.4f} "
                      f"[{p['two_lo']:>+8.4f},{p['two_hi']:>+8.4f}] "
                      f"[{p['one_lo']:>+8.4f},{p['one_hi']:>+8.4f}]")

        print(f"\n  PAIRED vs {REF}  --  secondary: e_first (a PRODUCT, not served evidence)")
        for name in ARMS:
            if name == REF or name not in R:
                continue
            p = paired(R[REF], R[name], docof, lambda r: r["e_first"])
            if p:
                print(f"  {name:18} {p['n']:>6} {p['obs']:>+9.4f} "
                      f"[{p['two_lo']:>+8.4f},{p['two_hi']:>+8.4f}] "
                      f"[{p['one_lo']:>+8.4f},{p['one_hi']:>+8.4f}]")

        print(f"\n  LA-LM, §87's construction (own served fact events w/ served C-MATCH partner)")
        for name in ARMS:
            if name not in R:
                continue
            v, n = la_lm_87(R[name])
            print(f"  {name:18} {v:>+9.4f}  n={n:,}")
        print()

    # ---- the registered bit-identity gate -------------------------------------------------
    print("=" * 118)
    print("REGISTERED BIT-IDENTITY GATE: WSELTAG=0 must reproduce _93/scope0.json EXACTLY")
    print("=" * 118)
    if os.path.exists("_95/gate_b22_notag.json"):
        a, b = load("_93/scope0.json"), load("_95/gate_b22_notag.json")
        same_bpb = (repr(a["test_bpb"]) == repr(b["test_bpb"]))
        ra = {key(r): r["e_first"] for r in a["rows"] if r["kind"] == "fact"}
        rb = {key(r): r["e_first"] for r in b["rows"] if r["kind"] == "fact"}
        same_keys = (set(ra) == set(rb))
        worst = max((abs(ra[k] - rb[k]) for k in ra if k in rb), default=float("nan"))
        print(f"  test_bpb  §93 {a['test_bpb']!r}")
        print(f"  test_bpb  §95 {b['test_bpb']!r}")
        print(f"  identical bpb: {same_bpb}   identical fact-row keys: {same_keys}   "
              f"max |de_first| = {worst:.3e}")
        print(f"  GATE: {'PASS' if (same_bpb and same_keys and worst == 0.0) else 'FAIL -- RUN IS VOID'}")
    else:
        print("  gate arm not present")


if __name__ == "__main__":
    main()
