# §93 pre-registration — does removing the sentence scope repair fact recall on real text?

**Written 2026-09-19, before any §93 run, at commit `8dc730b`.**

## 1. Why this exists

§87 found the instrument's fact memory **sentence-scoped**: a `.` wipes the whole slot LRU, and
**2,220 of 11,751 fact events (18.9 %) are structural zeros** — cross-sentence events whose cue
*cannot* be resident, contributing exactly 0.000 by construction (probe self-test S3: 21/21 exact
zeros). §83 later discovered, at engine scale and post-hoc, that whole-stream scope
(`SELSENT=0`, never clear) beats sentence scope on every text corpus (+0.00002–0.0001 bits/bit) —
the engine adopted it. The instrument never re-measured. §92B refocused attention on the mixing
space; this registration keeps the scope question separate and falsifiable, because it is the
cheapest untested structural limit named in §91's close-out (scope, saturation, scale).

The knob: `wstate.py` `WSELSENT` (added for this section, default 1 = §81 sentence scoping,
verified bit-identical: 30 KB cost 78759.383835 before and after). `WSELSENT=0` = never clear,
matching the engine's adopted `SELSENT=0`.

## 2. Estimand and protocol

Identical to §87's registered protocol in every respect except the knob: arm `selleanu`,
`WNS=1 WSLOTMODE=word WSLOTS=32 WVBITS2=22`, train `data/wt103_train.txt[:2700 KB]`, measured
stream `data/wt103_test.txt[:400 KB]`, 13-byte deterministic decontamination, online scoring.
Two arms, same data, same seeds: `WSELSENT=1` (the §87 measurement, rerun fresh) and
`WSELSENT=0`. The §87 metrics recomputed identically by `_factprobe.py`'s own extraction:
mean fact gain at the identity byte (with cluster-bootstrap CIs), served-evidence at fact
outcomes, `LA − LM` paired estimand, bits/byte of the whole stream, and the S3 structural-zero
count (which must drop toward 0 under `WSELSENT=0` — the mechanism check).

## 3. Registered criteria (both arms, wt103, one run each; deterministic model ⇒ seeds fixed)

* **S1 MECHANISM.** Under `WSELSENT=0`, cross-sentence events are no longer structurally zero:
  the resident fraction of cross-sentence cues exceeds 0.5 (it was 0.000 by construction).
  If S1 fails the knob is broken and nothing else is read.
* **S2 EVIDENCE (primary).** Mean served evidence at fact outcomes under `WSELSENT=0` exceeds
  the `WSELSENT=1` arm by ≥ 0.02 bits/event with the cluster-bootstrap 95 % CI of the paired
  difference excluding 0. (§87's served evidence was negative at every rung, −0.64 → −0.25;
  this criterion asks for improvement, not sign reversal.)
* **S3 COST (the §86.8 tension).** Whole-stream bits/byte reported for both arms. No threshold:
  the pairing is reported, not gated — but if `WSELSENT=0` is worse by more than 0.01 bpb the
  reading is "repairs memory, damages compression," which §86.8 already frames as the regime
  question, and no adoption follows.
* **S4 `LA − LM`.** Reported both arms; improvement in the negative direction (toward memory
  repair) is supporting evidence, not a gate.

## 4. Registered branches

1. **S1 and S2 pass:** the sentence wipe is a real cause of the §87 negative → §93 names scope
   the cheapest fact-recall lever since §86; next step is the saturation interaction
   (`WSELSENT=0 × WVBITS2` ladder).
2. **S1 passes, S2 fails:** the structural zeros were zeros anyway — cross-sentence cues,
   once resident, still do not serve usable memory → the §87 negative deepens to "not
   scope-limited"; the close-out extends to scope and only saturation/scale remain.
3. **S2 passes, S3 cost > 0.01 bpb:** memory/compression tension reproduces on the scope axis;
   recorded as the third instance of the §86.8 regime (after de-collision and tagging).
4. Anything ambiguous: report, do not interpret past the numbers.

No arm is adopted and no engine change follows from this section in any branch (the engine
already runs `SELSENT=0`).
