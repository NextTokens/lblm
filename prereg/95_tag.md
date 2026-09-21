# §95 pre-registration — does the 8-bit tag buy de-collision's *vote quality* for 4 MB?

**Written 2026-09-20, before any §95 run, and committed ALONE before the arms are launched.**
Against commit `ef5f24e`.

This registration is committed on its own, with no result in the same commit, because the §92–§94
audit found that `prereg/93_scope.md` and `prereg/94_ladder.md` were both first committed *in their
own result commits* (`d2f334c`, `77f6f0b`), so the ledger sentences "Registered before any run
(commit `8dc730b`)" and "(commit `d2f334c`)" name commits that do not contain the file. The
filesystem evidence supported pre-registration in both cases; git did not. This file fixes the
process, not by asserting harder but by committing earlier.

---

## 1. What §94 established, and the one question it leaves

§94 laddered two levers under whole-stream scope (`WSELSENT=0`) and reported both as passing. The
§92–§94 audit showed that only one of them did:

| arm | P(served) | E[e \| served] | mean \|e\| served | **sign-mean** | E[sign \| served] | bpb |
|---|---|---|---|---|---|---|
| ref `s0_b22/T4` | 0.1091 | −0.4726 | 0.7227 | **−0.0451** | −0.4135 | 2.275212 |
| `s0_t16` | 0.7211 | −0.1081 | 0.1532 | **−0.3927** | −0.5446 | 2.276250 |
| `s0_t32` | 1.0000 | −0.0314 | 0.0674 | **−0.4271** | −0.4271 | 2.275946 |
| **`s0_b28`** | 0.1627 | −0.1136 | **0.8052** | **−0.0054** | **−0.0329** | 2.280652 |

**The reach lever is a mixture-denominator artifact.** On §89A's own |T|-invariant sign-mean
(ledger:4633) the reach ladder gets monotonically *worse*; at `t32` roughly 71 % of served cue votes
still point the wrong way, no better than the reference. Per-served-event informativeness collapses
10.7×. §94's "+0.0201" is a still-wrong vote shrunk by the denominator and multiplied by a service
rate driven to 1.000. The registered branch that actually fired was `94_ladder.md`'s *other* one:
"service without evidence: the newly-served votes are uninformative".

**The saturation lever is clean.** `b28` holds `mean|e|` steady at 0.8052 (|T| unchanged), improves
the conditional 4.2×, and moves the sign-mean from −0.0451 to −0.0054 — within a hair of the first
positive absolute sign this project has ever produced on §87's estimand.

**But `b28` costs 4.29 GB of table.** §89B established that an 8-bit tag per vote cell with
evict-on-mismatch (`WSELTAG=1`, `wstate.py` v14) reaches a 64× larger table's *fact-level memory*
for **4 MB on a 67 MB table** — parity at 1/60 of the memory, past its registered bar on both
streams. That was measured under **sentence** scope (`WSELSENT=1`) and on §87's `LA − LM` estimand.

**§95's question, in one sentence:** does the tag reproduce `b28`'s vote-quality gain at 2^22, under
whole-stream scope, on the statistic §94's audit established as the honest one?

This matters beyond the arithmetic. De-collision is now the only surviving fact-recall lever in the
project, and it is unusable at 4.29 GB. If the tag carries it, the lever becomes affordable. If it
does not, §89B's parity result is specific to its own estimand and does not generalise.

## 2. The design — a 2 × 2, with two cells already on disk

This project has been corrected **six times** for reading a mechanism off a one-armed or
multi-variable experiment (§89A, then §90 and §91 in the same file, then §92B's decomposition, then
§93's and §94's estimands). The registration is therefore a complete factorial, and two of its four
cells already exist and are **not re-run**:

| | tag OFF | tag ON |
|---|---|---|
| **b22** | `_93/scope0.json` (on disk) | **`s0_b22_tag` — NEW** |
| **b28** | `_94/s0_b28.json` (on disk) | **`s0_b28_tag` — NEW** |

Two new arms, run in parallel, ~27 min each.

```
WNS=1 WSLOTMODE=word WSLOTS=32 WSELSENT=0 WVBITS2=<22|28> WSELTAG=1 \
  python _factprobe.py --job selleanu <22|28> 2700 400 100 _95/s0_b<22|28>_tag.json
```

Everything else is §87/§93/§94's protocol byte for byte: `selleanu`, 2700 KB train, 400 KB
decontaminated measure, 100 KB calibration, online scoring, one run per arm (the model is
deterministic).

**Why the b28 × tag cell is in the design and not dropped as redundant.** It is the cell that
separates two indistinguishable hypotheses. If the tag and de-collision repair the *same* defect,
`b22_tag ≈ b28` and `b28_tag ≈ b28` — the tag is a cheap substitute. If they repair *different*
defects, `b28_tag > b28`, and the two stack. Without this cell, a positive `b22_tag` cannot tell
those apart, and this project's whole recent history is of exactly that ambiguity being resolved the
wrong way.

## 3. Statistics — fixed now, and chosen to survive the §93/§94 audit

**Primary, registered: the |T|-invariant sign-mean of `e_first` on fact-kind events**, paired against
the b22 tag-off reference. This is §89A's own scale-immune statistic (ledger:4633), and it is
primary precisely because §94's headline was overturned on it. `|T|` is in fact matched across all
four cells here (`WSELTOPM = 4` throughout), so magnitude comparisons are legitimate — the sign-mean
is registered as primary anyway, so that no outcome can later be rescued by switching statistics.

**The §89A amendment is reinstated, having been dropped by both `93_scope.md` and `94_ladder.md`.**
Ledger:4614: the estimand is *"the pair (P(served), E[e | served]) reported together, **never their
product alone**"*. Every §95 table reports the pair. `e_first`'s unconditional mean is reported
beside them, labelled as a product, never as "served evidence".

**Also recorded, all pre-committed, none selected afterwards:**

* `e_first − e_match_first`, the |T|-matched within-event contrast (§89A, ledger:4634-4637) — already
  in every row and never reported by §93 or §94.
* `LA − LM` on **§87's own construction** (each arm's own served fact events with a served C-MATCH
  partner) **and** on §93's doubly-served restriction. Both, because the audit found they carry
  opposite signs and §93 published only one. The §93/§94 practice of averaging `LA − LM` over `else`
  and `cross` rows is **excluded**: `_factprobe.py:580` never writes `LM` for those kinds, so the
  statistic does not exist there.
* The controls the probe already prints and §93 omitted: C-MATCH, C-ELSEWHERE, and C-NULL **split on
  contamination** (ledger:4430-4432 — C-NULL is 70 % facts and is not a specificity control unless
  split).
* `inT_first`, `cand_n`, `a_cue`, `max|w|`, cell count, **evictions** (the tag's own activity — an
  arm with `evictions = 0` has not engaged the mechanism and is void as a test of it).
* Whole-stream bpb on measure and calibration slices.
* The `§87 ADJ filter` **on** (the registered §87 protocol, which §93 and §94 both silently ran with
  it off, dropping 1,736 rows / 5 %) **and** off, so the rows remain comparable with §93/§94's
  published numbers. The ADJ-on figures are the registered ones.

**Interval estimator.** Both are reported and the **two-level (document) bootstrap is primary**,
`BOOT=2000 SEED=20260917`, which is `_factprobe.py`'s own default analysis path. §93 reported
intervals that match no registered resampler, and under the registered two-level bootstrap its
all-events lower bound falls to +0.0024, below its own +0.02 bar. Registering the stricter estimator
as primary in advance removes that degree of freedom.

**Recording.** Each JSON must carry the environment block and the **sha256 of the `wstate.py` blob**
used. §92's captures recorded neither, so no §92 figure is checkable against a changed engine — and
`wstate.py` has changed since (v17). If the hash is absent from the artifact it is recorded in the
run log and quoted in the write-up.

## 4. Bit-identity gate

`WSELTAG=0` under `WSELSENT=0` at b22 must reproduce `_93/scope0.json` **exactly** — `test_bpb`
2.275212334662630 and the fact-row `e_first` vector unchanged. §89B registered the same gate and
`wstate.py:302` states the tag is bit-identical when off. **If it does not reproduce, the run is void,
the defect is found and fixed, and every arm is re-run from scratch.** No result from a
non-identical build is reported, and no §95 number is quoted before this gate is shown green.

## 5. Branches — one per outcome, written now

Let `Δsign` be the paired change in fact-kind sign-mean against the b22 tag-off reference
(−0.0451). `b28`'s value is **+0.0397**; half of it is **+0.0199**, i.e. a tagged sign-mean of
−0.0252.

1. **`b22_tag` sign-mean ≥ 0, CI excluding 0 upward** → the first **positive absolute** served-vote
   sign in this project's history, at 2^22. This is the outcome the handover's "§95 must demand the
   SIGN" was written for. Action: report it, then **stop and red-team before anything else** — no
   adoption talk, no engine port, no real-text extension in the same session. A first-ever sign
   change is exactly the kind of result this project has had overturned before (§72–§76, §86's
   binding claim, §89A's purity claim).
2. **`Δsign ≥ +0.0199` (≥ half of `b28`'s gain) and bpb cost < `b28`'s +0.005440** → the tag buys
   de-collision's vote-quality gain at 1/60 of the memory; §89B's parity result generalises from its
   own estimand to §94's. Recorded, **not adopted** — §80's held-out ritual governs adoption and is
   not part of this registration.
3. **`0 < Δsign < +0.0199`** → the tag helps but does not carry the lever. Report as a partial
   transfer, with the honest consequence stated: de-collision remains the only full lever and it
   still costs 4.29 GB.
4. **`Δsign ≤ 0`** → the tag does **not** transfer to this estimand. §89B's result is specific to
   `LA − LM` under sentence scope. Action: the tag line closes for the fact-recall track, and §89B's
   ledger text gains a scope note. This is a genuinely likely outcome and is named here so it cannot
   be re-analysed into branch 3: scope0 puts **more** keys through the **same** cells (§93's own
   named follow-up), so evict-on-mismatch may thrash rather than protect.
5. **`b28_tag ≈ b28`** (paired CI includes 0) → the tag and de-collision repair the **same** defect.
   Combined with branch 2 this is the strong result: one mechanism, two prices, take the cheap one.
6. **`b28_tag > b28`, CI excluding 0** → they repair **different** defects and stack. Records a new
   open lever; does not license a claim about which defect is which without a further decomposition.
7. **Any arm improves bpb *and* sign-mean together** → the second configuration in project history to
   move both axes (§89A was the first, and its own correction 1 downgraded it to a tag × width
   interaction). Flagged for §80; not adopted here.
8. **`evictions = 0` on a tagged arm** → the mechanism never fired; the arm is void as a test and the
   cause is found before anything is reported.

## 6. Reverse-course rule

Mirroring §90's, at the same standard and at the owner's standing instruction: **if branch 4 fires on
`b22_tag`, the tag is retired for the fact-recall track** — no further tag arms on this estimand, no
"try it at b24", no re-parameterisation to rescue it. The purpose of this registration is to find out
whether the affordable version of the only surviving lever works, not to keep the tag alive.

And the symmetric rule, which matters more: **no branch of this registration licenses a claim about
real-text deployment, the engine, or compression.** §94's five arms all cost bpb; §86.8's regime
account says corpus bits/byte can reward the very defect that destroys a specific memory. A fact-recall
win here is a fact-recall win and nothing else.

## 7. What would make this wrong

* Reading the product `e_first` as "served evidence". Guarded: §3, the pair is mandatory.
* Comparing magnitudes across different `|T|`. Guarded: `WSELTOPM = 4` in all four cells, and the
  sign-mean is primary regardless.
* Publishing whichever interval estimator passes. Guarded: both reported, two-level registered primary.
* Conditioning on a post-treatment served set. Guarded: the pair is reported on each arm's own rows;
  any fixed-subset analysis is labelled as such and is secondary.
* A tagged arm that never evicts. Guarded: branch 8.
* Attributing a `b22_tag` gain to the tag when de-collision would have done it anyway. Guarded: the
  2 × 2, branches 5 and 6.

**Red-team:** required on the result before any ledger text is written, per the standing rule.
