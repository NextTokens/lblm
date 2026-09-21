# §96 pre-registration — Jev as a RIVAL DETECTOR on honestmap's job

**Written 2026-09-21, before any scored call. Committed ALONE, before the run**, per the §95 process
fix (§93's and §94's registrations were first committed in their own result commits).

Against commit `ac5bf4d`.

---

## 1. The question, and why it is not §90 again

§90 imported Jev's one testable **mechanism** (per-question isolation → a per-word readout, `seliso`),
tested it against a pre-registered bar, and retired it. §92B then showed §90 had attributed its own
result to the wrong variable anyway.

§96 asks a different question, and one the project has never asked of anything: **is the external
model better than us at our own job?** `honestmap.py` exists to answer "can the bit-native engine
tell STRUCTURED-but-high-entropy bytes from TRULY-RANDOM bytes, where order-0 entropy is blind, and
does it beat gzip?" Its own docstring sets the standard: *"If a 20-line order-2 script matches it,
there is no moat and we say so."* A frozen commercial decision model is a stronger rival than a
20-line script, and it has never been run against this corpus.

Nothing is integrated into the machine by this experiment. Jev is added as **one more column** beside
`order0`, `cond_entropy(k=1)`, `gzip_bpb` and `engine_bpb`. If it wins, the moat claim in
`honestmap.py` is weakened and must be rewritten; if it loses, that is the strongest evidence for
the moat this repository has.

## 2. The API contract, pinned

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <TYPESAFE_API_KEY>       # from .env, gitignored, never logged
Content-Type: application/json
{"state": <hex string>, "model": "jev-latest", "questions": {...}}
```

`model` is sent as `"jev-latest"`; the **resolved version is read from the response and recorded**
with every row. The probe on 2026-09-21 resolved to **`jev-1.13.0`**. If the resolved version changes
mid-run the run is void and is re-run in full — a rival's version is part of the measurement.

Measured on three probe calls, and used only for sizing: **~1.77 input tokens per blob byte plus
~285 tokens of fixed overhead**; latency ~0.5 s; no rate-limit headers are advertised. Errors
documented: 401, 422, 429, 529, with exponential backoff advised.

## 3. Method, fixed in advance

**Primitive: `Noul`**, which returns a bare scalar in [0, 1] — directly usable as a detector score
and the only primitive that needs no rubric design. **The instruction string is fixed here verbatim
and may not be edited after any score is seen:**

> `These bytes are cryptographically random, with no repeated blocks, no periodic structure, and no encoding pattern.`

Higher `noul` = "more random", which is the same direction as `engine_bpb` and `gzip_bpb`, so no
sign-flipping is needed anywhere.

**Secondary primitive: `Score`**, 5 ordered levels, run on the same windows, to obtain the
`probabilities` and `confidence` fields `Noul` does not return. Reported, not used for the verdict.

**Encoding: hex. Chosen on measurement, before any score existed.** A 16-byte ECB block is not a
multiple of 3, so base64 shifts phase between successive blocks: on a blob with 250 repeated
16-byte blocks, base64 preserves **101** repeated units and hex preserves **250**. Base64 would
destroy 60 % of the exact signal the STRUCTURED class is built from. Hex is byte-aligned, at 2× size.

**Windows.** 8 windows per blob, 2,048 bytes each, at evenly spaced deterministic offsets. Windows
rather than whole blobs because 48 KB of hex is ~85 k tokens per call, and because windows give a
within-blob variance the present harness has none of.

**The corpus must be made deterministic.** `honestmap.corpus()` calls `os.urandom`, so every run
builds different blobs — responses could not be cached and no result would be reproducible. §96 uses
a seeded PRNG in place of `os.urandom`, holds everything else byte-for-byte, and records the
**sha256 of every blob** with the results.

**Caching, and why it is load-bearing.** Every response is cached keyed by the sha256 of the exact
request body, and **the cache is committed**. Every number is then replayable offline, with no key
and no network. Without this, a remote frozen model makes this the only section in the ledger that
cannot be re-derived from a clone — which is the defect §92 criticised §87–§91 for.

**Comparability across detectors.** Raw gaps in bits/byte are not comparable to a [0, 1] scalar, so
the primary metric is rank-based:

- **PRIMARY: AUC** — the probability that a randomly drawn RANDOM window scores "more random" than a
  randomly drawn STRUCTURED window. Scale-free, computed identically for all five detectors on the
  **same windows**. 0.5 = no separation; 1.0 = perfect.
- Intervals by **blob-clustered bootstrap**, resampling blobs and then windows within them. Windows
  inside one blob are not independent, and §93/§95 were both corrected for exactly this class of
  error at the document level. 5 STRUCTURED and 2 RANDOM blobs is a thin cluster count and the
  interval will be wide — that is reported, not hidden.
- Secondary: each detector's raw per-class means, and the `low-entropy` and `random-ish` blobs
  reported but excluded from the AUC (the registered contrast is RANDOM vs STRUCTURED).

**The fairness control, and why this test could be unfair without it.** Asking a text model to judge
randomness from a hex dump is out of distribution, and a failure could be the *encoding* rather than
the model. Control: the same question on the `english text` blob's windows, hex-encoded identically.
A detector that cannot separate English-as-hex from urandom-as-hex is being limited by the encoding,
and **that reading is registered in advance** so a negative cannot be over-claimed as "Jev cannot do
this".

## 4. Branches — one per outcome, written now

Let `A_jev`, `A_eng`, `A_gzip` be the AUCs, and `Δ = A_jev − A_eng` with a blob-clustered CI.

1. **`A_jev ≥ A_eng`, CI on Δ excluding 0** → **the external model beats the engine at the job
   honestmap exists to demonstrate.** Action: report it plainly and first; `honestmap.py`'s
   "HONEST READ" verdict and the moat claim are rewritten in the same change. No softening.
2. **`A_jev ≥ A_gzip`, but Δ < 0 with CI excluding 0** → a real rival to the cheap baseline, not to
   the engine. The moat narrows to "better than gzip *and* better than a frozen commercial model",
   and the honestmap text says so.
3. **`A_jev < A_gzip`** → Jev does not do this job. No integration on this seam. The moat claim
   stands and is *strengthened*, and this is recorded as the first external-rival test it survived.
4. **`A_jev` CI includes 0.5** → Jev cannot separate the classes at all on hex. Same action as 3,
   plus the fairness control decides whether the honest sentence is "cannot do this" (control also
   at 0.5) or "cannot do this **from hex**" (control well above 0.5).
5. **Fairness control below 0.7 AUC** → the encoding is the binding constraint. Every Jev number is
   reported with that scope and no claim about the model's capability is made in either direction.
6. **More than 10 % of calls fail** (non-200 after backoff) → the run is **void**. No partial
   results are reported, no branch fires.

## 5. Reverse-course rule

If branch 3 or 4 fires, **the honestmap seam closes**: no prompt tuning, no re-rubricking, no
switching primitives to rescue the number. The instruction text in §3 is the one that ran. This rule
exists because prompt-tuning against a visible score is the purest forking path available here, and
because this project's own §95 disclosed that 5 of 8 post-hoc estimator choices would have crossed a
threshold the registered one did not.

## 6. What this cannot establish

- Nothing about the other two integration seams (window-level alarm adjudication; the commit/abstain
  layer). Those need their own registrations.
- Nothing about production cost, latency, or availability.
- Nothing about Jev's behaviour on its intended inputs. This is a deliberately hostile domain for it,
  chosen because it is *our* domain — which is the point of a rival test, and also its limit.
- **No claim about the ledger's window-level firewall.** §92A: no null distribution over windows
  exists in this project, and §96 does not create one.

**Red-team:** required on the result before any wording in `honestmap.py` or the ledger changes.

## 7. Egress

Everything transmitted is built from `data/` public corpora (wikitext / corpus.txt) plus standard
stdlib crypto and encoding constructions. **No unpublished project source, no results, and no
credentials leave the machine.** The owner authorised public-corpora egress only, on 2026-09-21.
