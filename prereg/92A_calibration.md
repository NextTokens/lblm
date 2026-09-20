# §92A pre-registration — is the machine's probability calibrated?

**Written before any calibration number has been computed. 2026-09-19.**
Nothing in this repository has ever measured calibration: `grep -rniE "\b(ECE|brier|reliability
curve|calibration curve|proper scoring|observed frequency)\b" --include=*.py --include=*.md .`
returns **0 hits** at commit `0d974db`. This registration is written against that zero.

---

## 1. Why this exists

Two live claims in the project assert calibration, and neither is supported by a calibration
measurement:

1. **Ledger line 1352** — "**Confidence is calibrated:** accuracy@commit rises monotonically
   (0.57 → 0.95) as the threshold tightens — higher self-confidence really does mean higher accuracy."
2. **`poc.py:244`**, in the user-facing dashboard under the heading *What makes it a class of one* —
   "**calibrated** — commits/abstains at a fixed τ, measurably monotone".

Both describe a **selective-prediction** (accuracy-at-coverage) curve. That curve is monotone for any
model whose confidence ordering is informative, including a model that is uniformly 2× overconfident.
It is not calibration. Calibration is the distinct property that among the bits the machine assigned
probability ≈ q, the observed frequency of a 1 is ≈ q.

A third, larger claim was made in conversation on 2026-09-18 and is withdrawn in advance here: that
the machine is "calibrated by construction because log loss is a proper scoring rule". A proper
scoring rule makes truthful reporting the **optimum** of the objective; it does not establish that an
under-trained online mixer attains it. This registration exists to replace the argument with a number.

## 2. The estimand

Prequential (online, one-pass) calibration of the per-bit forecast. For every bit the instrument
codes on the held-out slice, record the pair `(p, y)` where `p` is the probability the model served
for that bit **before** seeing it and `y ∈ {0,1}` is the bit. The forecast is sequential and adaptive;
this is the standard prequential setting and calibration is well defined in it.

Reported on the probability of the **observed-class** convention made explicit: all binning is on
`p1 = P(bit = 1)` against outcome `y`, with no folding of `p` to `max(p, 1-p)`. Folding is reported
separately as a secondary view because it is what a "confidence" claim implies, and the two are not
interchangeable.

## 3. Method, fixed in advance

**Arm and protocol.** The instrument's adopted configuration: `selleanu`, `WNS=1`,
`WSLOTMODE=word`, `WSLOTS=32`, `WVBITS2=22` — the same rail every §86–§91 corpus number uses.
Stream: `data/wt103_test.txt`. Train prefix then a held-out measured slice, using the same
train/measure/calibration split the §87 job uses, so the population is one already characterised.

**Statistics, all four reported, none selected after the fact:**

- **ECE** — expected calibration error, 10 bins, and repeated at 20 bins.
- **ECE on equal-mass (quantile) bins**, same counts. Bit-level forecasts are extremely concentrated
  near 0 and 1; uniform-width bins put almost all mass in two bins and can hide interior
  miscalibration. Both binnings are pre-committed; neither is the headline alone.
- **MCE** — maximum calibration error over bins with n ≥ 100.
- **Brier score**, and its decomposition into reliability / resolution / uncertainty.

**The noise floor — the load-bearing method, and the reason this registration is specific about it.**
ECE is biased upward at finite sample because each bin's hit rate is a noisy estimate. The floor is
**not** a function of n alone: it depends on how concentrated the forecasts are. The floor will
therefore be simulated from the machine's **own** predictions:

    for trial in 1..1000:
        y_sim[i] ~ Bernoulli(p[i])        # the model is calibrated BY CONSTRUCTION in the sim
        floor[trial] = ECE(p, y_sim)
    floor_mean, floor_p95 = mean(floor), percentile(floor, 95)

The headline is the **ratio ECE / floor_mean**, with `floor_p95` as the significance line. An ECE at
its floor measures nothing in either direction and will be reported as such, not as evidence of
calibration.

This method is adopted from an external critique of an unrelated product (the public
`jev-exploration` audit of TypeSafe's Jev, Sept 2026), which found that quoting a floor indexed on
sample size instead of recomputing it from each study's own predictions inverted the conclusion for
two of three studies. The same error was made in this project's own 2026-09-18 analysis. It is
written into the method here so it cannot recur.

**Separation of the two claims.** The selective-prediction curve (accuracy@commit vs τ) will be
recomputed alongside, so ledger 1352's actual number is reproduced rather than disputed from memory.
The report states explicitly whether each of the two properties holds, independently.

**Secondary population.** The same statistics on the next-**byte** top-1 decision at τ = 0.90, which
is what `poc.py` actually displays. Per-bit calibration does not imply per-byte-decision calibration;
if only one is measured, only one may be claimed.

## 4. Branches — every outcome has one, written now

Let `r = ECE / floor_mean` on equal-mass bins, 10 bins, per-bit.

- **`r ≤ 1.2` and ECE < floor_p95** → the per-bit forecast is calibrated as far as this test can
  resolve. **Action:** ledger 1352 is corrected anyway — it currently cites the wrong evidence for a
  right conclusion. `poc.py:244` may keep the word *calibrated* but must cite this section, and must
  not be extended to the byte decision or to any window-level anomaly score without its own measurement.
- **`r > 1.2`, model overconfident** (observed frequency closer to 0.5 than stated) → **the claim is
  false as stated.** `poc.py:244` is rewritten to the property actually held (informative confidence
  ordering / selective prediction), ledger 1352 is corrected in place with a dated note, and the
  conversational "calibrated by construction" claim is recorded as refuted.
- **`r > 1.2`, model underconfident** → same corrections, different wording; additionally recorded as a
  compression opportunity, since an underconfident coder is leaving bits on the table.
- **ECE and floor both ≈ 0 and the bins are degenerate** (all mass in ≤ 2 bins even under quantile
  binning) → the test is **uninformative at bit level** and is reported as such, with the byte-level
  secondary becoming the headline. This branch exists because it is a genuinely likely outcome for a
  bit predictor and must not be silently re-analysed into one of the above.
- **Any outcome** → a window-level anomaly score remains unmeasured. No branch of this experiment
  licenses a claim about alarm calibration for the monitor use case. That needs a null distribution
  over windows, which does not exist anywhere in this project.

## 5. What would make this wrong

- Measuring on the training prefix rather than held-out. Guarded: the split is the §87 job's.
- Reporting whichever binning flatters the result. Guarded: both pre-committed, both reported.
- Quoting a floor from a table instead of simulating it from these predictions. Guarded: §3.
- Treating a folded "confidence" ECE as the per-bit result. Guarded: both reported, labelled.

**Red-team:** required before any wording in `poc.py` or the ledger changes on the strength of this.


---

## AMENDMENT 1 -- the decontamination-selection control (registered 2026-09-19, BEFORE the run)

**Why.** The §92A red-team found a confound neither the registration nor I anticipated. Every number
above is measured on the bytes that survive `clean_mask_det(tr, te, 13)`, and that filter chooses
bytes by looking at the very text being scored. The retained fraction is **69.1 %** at 400 KB and
**41.6 %** at 2700 KB -- so across the two scale points, "more training" and "more aggressive
outcome-dependent selection" move together, and the selection hypothesis **alone** predicts the
direction and rough size of the observed change. Until this control runs, no scale sentence may be
written in either direction -- not "flat", not "training makes it worse". The §92A headline must
carry the scope "on the held-out text it scores".

**The run.** `capture()` with `mask = [True] * len(te)`, everything else byte-for-byte identical, at
both 400 KB and 2700 KB. Model state is unaffected: learning already runs on every byte, so the mask
only selects which bits are *recorded*. Recorded from the same run: the ECE of the **masked-out**
bytes alone, which tests the confound directly rather than by inference.

**Branches, fixed now, on ECE -- never on r.**

- **Full-slice ECE within ±15 % of the masked figure (0.00695 / 0.01026), same sign** -> the selection
  confound is dead. The headline drops the "on the text it scores" scope and applies to the machine.
- **Full-slice ECE falls by more than a third toward the floor** -> the headline **is** a selection
  artifact and is rewritten as one: the finding becomes "on the decontaminated subset the machine is
  overconfident", which is a much narrower claim, and the scale comparison is withdrawn entirely.
- **Between those** -> the effect is real but partly selection-driven; report both figures side by
  side and keep the scope clause permanently.
- **Masked-out bytes UNDERconfident by a matching amount** -> direct confirmation of the confound,
  reported whatever the full-slice number does.

**Also registered now: r is retired as a headline statistic.** It is a power statistic, not an effect
size: on identical data it runs 18.36 / 14.77 / 14.51 / 10.26 across the four pre-committed binnings
and 4.63 at uniform-100, and it grows as √N under the pure no-effect null -- the red-team recovered
the whole "5.33 -> 18.36" trend by **subsampling the 400 KB capture alone** (r = 6.32 at the 40 KB
run's extent). r keeps exactly one job: establishing that the effect is not noise. Every magnitude
statement uses ECE, the Platt slope, and the plain-language odds ratio.

**Withdrawn from §92A before drafting, per the red-team:**
1. "ECE is roughly flat with 10× more training" -- confounded three ways (train size, evaluation
   extent, retained fraction). At matched evaluation extent it is 0.00879 vs 0.00625, a 41 %
   *increase*. The by-fifths split is different and better evidence and is what should be cited.
2. "the selective-prediction curve reproduces ledger 1352" -- it does not. Ledger 1352 is a **byte**-level
   greedy rollout with confidence = the product of per-bit max-probs (0.565 -> 0.947). `selective_curve()`
   is a **bit**-level analogue (0.9161 -> 0.9984). Different population, different confidence definition,
   different numbers. The *logical* point -- that a monotone accuracy-at-commit curve is produced by any
   informative ordering and is therefore not evidence of calibration -- stands without it.
3. The registration's own claim that "the population is one already characterised" is **false at 400 KB**:
   the wt103 ladder is [150, 450, 1200, 2700] KB (`wstate.py:2030`) and 400 KB is not on it. It is true
   at 2700 KB, where the measured bpb 2.278784 matches §86.8's recorded 2.2788.

**Two defects found by the red-team, both real, both verified here before this amendment was written:**
- `poc.py:peek()` advances `htail` one BIT at a time. `wstate.py:1435` advances it one BYTE at a time in
  `_byte_end`, i.e. it is **frozen across all eight bits**. The speculative beam therefore corrupts the
  order-context keying on bits 1-7 of every candidate byte. `calprobe.capture_bytes()` inherited the bug
  verbatim. Both are fixed before the byte-level secondary runs, and a new self-test S4 requires the
  served per-bit p along the TRUE byte's path to equal `step()`'s own cost to ~1e-12 -- the check that
  would have caught it and which S1-S3 do not perform.
- `genmem.py:35-38`: `squash(t)` returns `1e-6` for `t < -30` while the logistic at `-30` is `9.358e-14`,
  so the function jumps seven orders of magnitude at the boundary and is **non-monotone** there -- a more
  negative logit returns a higher `P(bit=1)`. The capture's minimum p is `9.581e-14` (`t = -29.976`): the
  machine ran to the cliff without crossing it. §92A's numbers are unaffected. Recorded as an engine
  defect; the fix is to clip `t`, not the output. Not fixed here -- it changes coded output and belongs
  under the §80 ritual.
