# §92B pre-registration — the missing cell: does the readout's MIXING SPACE carry §90/§91's effect?

**Written before any `selmix` run. 2026-09-19.** Against commit `0d974db` (§91 addendum).

---

## 1. The defect this exists to repair

§91A concludes: *"the spectrum is now fully measured … the §85 lever's readout component is unsolved
at this data scale … no further readout iterations on this probe."* That closure is attributed to the
**dial keying** variable (shared → per-word). The experiment that produced it changed **four** things
simultaneously.

Reading `wstate.py` at `0d974db`, the shipped `sel` path is:

```python
pm += self.sel_wT[t] * p1all[ci]      # :935   weighted average in PROBABILITY space, Σ wT = 1
f = stretch(pm)                        # :967   one logit transform, after mixing
d += self.vsw.get(wk, 0.0) * f         # :980   one shared dial
```

and the `seliso` / `selhyb` path is:

```python
for t, ci in enumerate(self.sel_T):    # :954
    f_t = stretch(p1all[ci])           # :957   per-candidate transform, BEFORE mixing
    w_t = a_cx + self.viso.get(wk_t, 0.0)
    d += w_t * f_t                     # :960   UNNORMALISED sum; sel_wT never used
```

| variable | `sel` | `seliso` / `selhyb` |
|---|---|---|
| mixing space | probability | logit |
| normalisation | `Σ wT = 1` (`:1510`) | unnormalised sum over \|T\| |
| gate weighting | `wT[t]` weights each vote | **dropped — `wT` unused on this path** |
| dial keying | shared `vsw[cx]` | per-word `a_cx + viso[(cx, wid)]` |

Three variables moved besides the one the conclusion names. Additionally, §90/§91's central
**mechanism** claim — "per-word dials starve, `max|w|` 4.5 vs the shared readout's 30.6" — is not an
apples-to-apples magnitude comparison: with `|T|` unnormalised dials summing and the gate weights
removed, each dial needs roughly `1/|T|` the magnitude to make the same contribution to `d`. The
observed gap is consistent with arithmetic before it is evidence of starvation.

This is the same defect §89A was corrected for, in its own words: *"asserted from a one-armed
experiment and is withdrawn in favour of the 2 × 2."*

## 2. Why the mixing space is a candidate cause, not a nuisance variable

Mixing in probability space and transforming afterwards destroys the evidence of a confident cell.
Measured directly from the shipped expressions (cell prior `(n+0.2)/(N+0.4)`, one cell at `(100,0)`
served alongside empty cells):

| served set | shipped `f` | logit-space `f` | evidence retained |
|---|---|---|---|
| \|T\| = 2 | −1.093 | −3.108 | 35.2 % |
| \|T\| = 4 | −0.509 | −1.554 | 32.7 % |
| \|T\| = 32 | −0.062 | −0.194 | 32.1 % |

About two-thirds of a confident cell's evidence is lost, and the ratio is near-constant in `|T|`, so
this is a representation defect rather than a mixture-size one. §89A's own correction #2 measured the
same thing from the other side — *"the same empty cell costs −1.43 at |T| = 4 and −0.32 at |T| = 32"* —
without naming the cause.

## 3. The arms — a decomposition, not a single new arm

| arm | mixing space | normalisation + gate weights | dial keying | status |
|---|---|---|---|---|
| `sel` | probability | `wT`, Σ = 1 | shared | reference (shipped) |
| **`selmix`** | **logit** | `wT`, Σ = 1 | shared | **NEW — isolates mixing space** |
| **`selisn`** | logit | `wT`, Σ = 1 | per-word | **NEW — isolates keying, normalisation held** |
| `seliso` | logit | unnormalised, no gate | per-word | §90, as built |
| `selhyb` | logit | unnormalised, no gate | shared prior + per-word | §91, as built |

The chain gives clean attribution: `sel → selmix` isolates the mixing space; `selmix → selisn`
isolates the dial keying with normalisation held fixed; `selisn → seliso` isolates the normalisation
and gate-weight drop.

`selmix` is exactly `sel` with `f = Σ_t wT[t] · stretch(p1_t)` in place of `f = stretch(Σ_t wT[t] · p1_t)`.
Everything else — gate, candidate set, counts, updates, the single shared `vsw[cx]` dial and its
gradient — is untouched. `selisn` is `selmix` with the dial keyed `(cx, wid)` per served word, still
weighted by `wT` and still summing to one unit of weight.

**Bit-identity requirement:** every pre-existing arm must remain bit-identical after these are added.
Verified before any measurement by re-running the standing reference (`selleanu 60/25 KB` →
MEASURE `2.372273`, CALIB `2.525331`) and the §91 grid's `sel` row.

## 4. Criteria, and the grid

The §86 criteria exactly as §90 and §91 ran them — I1 (interference, k-matched), I2a (few-shot vote),
I2b (exposures to vote), I3 (no-regression on known cues) — on `_fewshot_probe.py`, G = 24, seeds 7/8,
so the new rows drop into §91A's published table without re-scoring it. Recorded alongside: each
arm's dial magnitudes (`max|w|`, cell count) and the instrument's bits/bit on the same grid.

## 5. Branches — written now, one per outcome

- **`selmix` passes I1 or I3 where `sel` fails** → **§91A's closure is withdrawn.** The readout story
  reopens, the lever is the mixing space rather than the keying, and a real-text run on §87's
  fact-level estimand is licensed (and required before any adoption talk).
- **`selmix` ≈ `sel` on all four criteria** → the mixing space is not the lever at this data scale.
  §91A's conclusion **stands**, but its text is corrected: it is licensed by this decomposition, not by
  the four-variable comparison it currently cites.
- **`selmix` materially better than `sel` but still failing every criterion** → recorded as a real but
  insufficient effect; the criteria are not moved to meet it. No adoption, no real-text run.
- **`selisn` differs materially from `seliso`** → §90/§91's *starvation* mechanism is confounded by the
  normalisation and gate-weight drop, and both sections' mechanism paragraphs are corrected. This is
  scored even if every arm fails the criteria, because it is a claim about **why**, and this project's
  red-teams have overturned the mechanism sentence six times while never overturning a measurement.
- **`selmix` improves the grid's bits/bit** → recorded, **not adopted**. §80's held-out ritual governs
  adoption and is not part of this registration.
- **Any arm fails to leave the pre-existing arms bit-identical** → the run is void, the defect is fixed,
  and the grid is re-run from scratch. No results from a non-identical build are reported.

## 6. Reverse-course rule

Mirroring §90's, at the same standard: if `selmix` and `selisn` both land in the "≈ `sel`" branch, the
readout story closes **for real** — no further readout arms on this probe, and the correction to §91A
is a strengthening of its conclusion rather than a withdrawal. The purpose of this registration is to
find out whether §91A's closure was earned, not to reopen the question indefinitely.

**Red-team:** required on the result before any ledger text changes, per the standing rule.
