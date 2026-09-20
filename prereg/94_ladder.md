# §94 pre-registration — the scope0 ladder: saturation and service-rate under whole-stream scope

**Written 2026-09-19, before any §94 run, at commit `d2f334c`.**

## 1. Why this exists

§93 (branch 1) made the sentence wipe a tension-free fact-recall lever — and its data named two
limits in the same breath. First, the **service rate**: under `WSELSENT=0` the gate serves the cue
half as often (`inT_first` 0.204 → 0.109) because cross-sentence candidates crowd the top-4 —
`WSELTOPM` is the direct knob and §89A already showed reach → 1.000 at `WSELTOPM=32` (on the
synthetic probe). Second, the **scope–saturation interaction**: scope0 makes MORE keys resident and
updates the SAME cells — §87's R4 ladder (at sentence scope) found de-collision repairs the
cell-level defect (`LA−LM` −0.121 at 2^28) while costing compression monotonically; whether
scope0 changes that regime is untested. This section runs both axes in one grid.

## 2. Protocol

Identical to §87/§93 (`selleanu`, `WNS=1 WSLOTMODE=word WSLOTS=32`, 2700 KB train, 400 KB
decontaminated measure, online scoring, deterministic, one run per arm). All arms run
`WSELSENT=0`. The §93 `b22/TOPM4` run (`_93/scope0.json`) is the reference and is NOT rerun.
New arms:

| arm | WVBITS2 | WSELTOPM | what it isolates |
|---|---|---|---|
| s0_b24 | 24 | 4 | saturation step 1 |
| s0_b26 | 26 | 4 | saturation step 2 (§87: still collided) |
| s0_b28 | 28 | 4 | saturation step 3 (§87: de-collides; 4.3 GB table) |
| s0_t16 | 22 | 16 | service rate step 1 |
| s0_t32 | 22 | 32 | service rate step 2 (§89A's full reach) |

Metrics recomputed identically to §93: paired cluster-bootstrap (fact types) CIs on
`e_first` (fact-kind and all-events), `inT_first` service rate, `LA−LM` on doubly-served rows,
`LA` on served rows, whole-stream bpb.

## 3. Registered criteria (each vs the §93 reference `s0_b22`; paired, cluster CI excluding 0)

* **SA SATURATION.** `s0_b28 − s0_b22` on fact-kind `e_first` ≥ +0.02 with CI excluding 0
  → "cells bind under scope0; de-collision transfers". CI including 0 → "at 400 KB the b22
  cell population is not the binding constraint under scope0" (§87 measured 104.9M distinct
  keys vs 4.19M cells — if that were binding at the margin, b28 shows it; if not, the §87 R4
  effect was a sentence-scope artifact on this estimand).
* **SB SERVICE RATE.** `s0_t32 − s0_b22` on fact-kind `e_first` ≥ +0.02 with CI excluding 0
  **AND** `inT_first` rising above 0.15 → "the crowded top-4 was suppressing evidence; reach
  restores service". If `inT` rises but `e_first` does not → "service without evidence: the
  newly-served votes are uninformative" (§89A's dilution finding, real-text edition).
* **S3 COST.** bpb reported per arm. Any arm whose bpb exceeds the sentence-scope §87 reference
  2.278784 by > 0.01 is marked TENSION regardless of evidence gains.
* **S4 STACKING (exploratory, no gate).** If SA and SB both pass individually, the named
  follow-up is `b28 × t32` (not run in this grid).

## 4. Registered branches

1. **SA and SB pass:** the two levers are real and separable; §94 names `scope0 + de-collision +
   reach` the candidate configuration for the first attempt at a POSITIVE absolute served-evidence
   sign on fact events (currently −0.052).
2. **SB passes, SA fails:** under scope0 the binding limit is the gate, not the cells; the §87 R4
   de-collision result does not transfer to whole-stream scope; saturation drops behind reach.
3. **SA passes, SB fails:** the §86.8 collision regime deepens — the memory improves only by
   de-colliding, and the compression cost is reported as the third regime instance.
4. **Neither passes:** §93's one-knob gain is near the ceiling at this scale; the scope thread
   closes at instrument scale and the frontier moves to §92B's mixing space.
5. Any arm marking TENSION: reported as memory-vs-compression trade, no adoption, per §86.8.

No engine change follows in any branch (the engine runs `SELSENT=0` and `NSELSLOTS=2`; its
candidate-space differs and nothing here licenses a port).
