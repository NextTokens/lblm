# HANDOVER — the LBLM intelligence track, post-§76

**Scope of this document:** everything needed to continue the "new strain of models" effort after
the 2026-09-13/14 sessions (ledger §72–§76, commits `7d1df61`…`37eeef7`, on top of the June
§1–§71 arc). Written to be self-contained: read this + `learned_binary_address_machine.md` §69–76
and you can continue without archaeology.

---

## 1. The one-paragraph state of the project

The project is an online, single-pass, bit-level predictive machine (compressor = intelligence
substrate). Its production engine `blmrs-strong` is lpaq1-class (0.209 bits/bit enwik8; 0.217011
on the 11 MB `corpus_big` A/B; 0.205801 at 30 MB enwik8). The intelligence track asks whether
*learned memory channels* can beat plain statistics on leak-free held-out data. After §69–71
every no-copy channel only tied the order-n models. This session **broke that wall** with
**word-slot binding** (§72), mapped exactly where that signal lives (code ≫ text ≫ DNA, §75),
identified why the production engine absorbs it (the word-model family, §75), and measured the
readout-capacity ladder that governs exploiting *depth* (§76). The next concrete build is named
at the end of this document.

## 2. What was done this session (verdicts, with numbers)

### §72 — the breakthrough: word-slot binding crosses the parity wall (`wstate.py`, `BLMSLOTS` in `strong.rs`)

- Probe-driven elimination: EMA-projection, exact trace-bank credit, and nonlinear bucket
  readout all fail identically because the EMA **superposes ~5 words** — no readout of a
  superposition can isolate one word's identity. Binding, not projection or credit, was the wall.
- **Word slots**: an LRU of the last S=6 distinct words, each holding its own 8-dim vector fed
  DIRECTLY as mixer features. Credit is exact by construction (a slot persists ~6 words; every
  bit while resident gradients its word's vector). No RTRL needed.
- Gate result (copy OFF, 13-byte-decontaminated wt103, deterministic mask, scrambled floor):
  slots cross BELOW the orders-only baseline at ≥1.2 MB, margin growing with data
  (−0.0031 → +0.0027); **replicated seeds 0/1/2** (+0.0015…+0.0031 vs baseline everywhere).
- Attribution: slots beat the frozen-vector control (`slotsr`) by +0.011…+0.014 at every
  seed×size, and `slotsr` never crosses → **the win is the vectors the compression loss
  teaches** (meaning beyond lexical identity).
- Rust port (`BLMSLOTS=1`, default OFF = bit-identical): **no gain at 11 MB** (+0.0001).
  This established the **port rule**: a channel earns its engine place only by beating
  strong's own baseline (0.217011 @11 MB `corpus_big`, obits 25), never the orders-only rail.

### §73 — hard similarity keying honestly fails

- `semsim` (count expert keyed by embedding-bucket = 8 sign bits + coherence dots) vs `slotsw`
  (identity-keyed expert control): at scale `semsim` never crosses and hurts bare slots by
  0.004–0.006; `slotsw` shows the §71 pattern (best at 150–450 KB, drag at 2.7 MB).
- Zero-shot probe: fastText-style subword-composed init (`semfast`) lands an unseen word in its
  family's direction, yet sign buckets still misroute — **hard quantisation destroys the graded
  information similarity carries**. Requirement established: similarity needs a SOFT readout.

### §74 — soft retrieval built in the engine, measured inert

- `strong.rs BLSOFT=1`: LSH over the §72 vectors (sign bucket + 8 one-bit-flip probes) → top-k
  neighbours of the last word by normalised dot → **similarity-weighted votes of the
  neighbours' own word-model counts** → one trained mixer head. Default bit-identical.
- 11 MB A/B: all configs +0.00009…+0.00015 vs 0.217011. Cause: single-pass tail vectors are
  too under-trained for informative neighbourhoods.

### §75 — the frontier sweep: where the signal lives, and the absorber identified

- **Scale (enwik8 30 MB production):** baseline 0.205801; slots +0.00009; soft +0.00009.
  Production verdict unchanged at 2.7× data.
- **Code is the channel's home turf:** slots cross at EVERY size from 100 KB
  (+0.0074…+0.0122, 3–4× text margins), meaning margin +0.017…+0.020.
- **DNA is an honest negative** (6-mer-tokenised E. coli): slots −0.02 vs baseline; learned
  worse than frozen. k-mer "words" have no selectional structure; codon/revcomp specialists
  (§55) remain the right DNA tools.
- **Absorber hunt (3 hypotheses, 2 killed):** depth fails everywhere (S=32/64 worse in both
  engines — but the S=32 scrambled-floor separation GREW to +0.64 vs +0.17: information exists
  at depth, the linear readout can't extract it); copy is NOT the absorber (match-ON instrument
  arms `matchbase`/`matchslots`: slots still beat orders+match, +0.002 wt103 / +0.006…+0.013
  code); **the word-model family IS** (identity word-expert on top of slots adds
  +0.028…+0.037 on code — same window, more signal — and strong has word + prev-word +
  hashed orders 8–32).

### §76 — attacking the depth gap with shared recency-discounted heads (`proj` arms)

- `feat_h = Σ_k γ^k (w_h · E[slot_k]) / Σ_k γ^k`, H=4 heads, γ=0.85 — 9 parameters per head,
  depth-independent; exact credit to heads and vectors. New corpus: **10.4 MB Python stdlib**
  (596 files, real cross-file code).
- **Variance penalty fixed** (raw S=32 was −0.03 below baseline; proj@32 at parity everywhere;
  meaning +0.004…+0.010, floor +0.012…+0.018 at every depth/corpus). **But 4 linear heads
  don't cash the depth**: proj@32 never crosses consistently; proj@6 crosses on code
  (+0.0012…+0.0014, ~1/10 the free-readout margin) — recency-averaging dilutes the recent
  window.
- **The measured ladder:** free-48-dim@6 ≫ 4head@6 > 4head@32 ≈ raw@32.
  *Value per unit of depth tracks readout capacity per unit of depth.*

## 3. The instruments — what exists and how to run it

| Tool | Command | Notes |
|---|---|---|
| The instrument | `python wstate.py --train T --test E --sizes a,b,c --arms A1,A2,…` | parallel over arms (24 cores). Default data = wt103; `--selftest` for a 40 KB sanity pass |
| Slot depth / heads | env `WSLOTS` (default 6), `WHEADS` (4), `WGAMMA` (0.85) | read at import time |
| Arms available | `baseline bitsonly randemb learned slotsr slots slotsw semsim semfast matchbase matchslots proj projr projscr scrambled` | see `wstate.py` docstring for the attribution ladder each controls |
| Production engine | `./blmrs/target/release/strong.exe <path> <cap> <obits>` | knobs: `BLMSLOTS`, `BLSOFT`, `NSLOTS` (≤64), `SBITS`, `LR_S`, `ALRS`, `SOFTK`, `SIMMIN`; all default OFF = bit-identical |
| The gate protocol | copy/match OFF · 13-gram decontamination (`clean_mask_det`, deterministic FNV — genmem's `clean_mask` is process-salted, do NOT use across processes) · frozen-vector control · scrambled noise floor | comparisons valid WITHIN a size only (test slice grows with train) |
| Scratch probes | `_assoc_check.py` (delayed word→outcome binding), `_simprobe.py` (tail + zero-shot families), `_replicate.py` (seed replication) | gitignored; keep using this probe-first pattern |

**Data on disk:** wt103 train/test (2.8 MB/1 MB), `corpus_big.txt` (11 MB, the production A/B),
enwik8 (100 MB, downloaded this session), `stdlib_train/test.txt` (7.8/2.6 MB, NEW cross-file
code), `code_train/test.txt` (585/195 KB), `dna_train/test.txt` (E. coli, 6-mer `.`-separated),
plus the June corpora and genomes.

**Timings (this machine):** instrument ≈ 0.04–0.06 Mbit/s per arm (a 5.6 MB arm ≈ 60–90 min);
`strong.exe` ≈ 0.1–0.2 Mbit/s (11 MB ≈ 12–18 min; 30 MB ≈ 30–50 min). Full multi-arm runs are
parallel; launch them in background and poll the output files.

## 4. Established laws — do not re-litigate

1. **Superposition cannot bind.** Any state that averages ~5 words is unreadable per-word, by
   any readout. Bind identity in slots.
2. **The slot channel's signal is real, replicated, copy-independent, and strongest on code** —
   but it lives in the ~6-word window that word-keyed statistics already harvest. Any engine
   with a decent word-model family absorbs it (that family is THE absorber — not match, not range).
3. **Depth holds information but linear recency readouts can't extract it** (the §76 ladder).
4. **Hard quantisation of similarity destroys its value**; similarity must be read softly.
5. **Domains rank: code ≫ text ≫ DNA** for word-binding signal; DNA needs its specialists.
6. **Process laws:** probe at byte-scale before trusting an hour-long run; replicate across
   seeds before writing a verdict; ports only on beats-strong; negatives go in the ledger.

## 5. The way forward — the next build, concretely

**Primary: an attentional read of the deep slot memory.** The problem is now precisely
"select WHICH bound word matters for THIS context," not "remember more." Design sketch,
reusing everything that exists:

- Candidates: the S=32 bound slots (always trained by residency credit — unlike §74's LSH
  neighbours, no cold-start).
- Query: a learned projection of the current context (start: `E[current word]` or the existing
  EMA state; a tiny learned query vector is the §76 head minus the fixed recency discount).
- Scores: `s_k = dot(query, E[slot_k]) / τ`; selection: softmax (soft) — **this is §74's
  similarity-weighted word-model vote with the candidate set = slots instead of LSH neighbours,
  and content-based instead of recency-based weighting.**
- Vote: each selected word's own word-model counts (exists in both engines), one mixer input.
- Acceptance: instrument CROSS vs baseline with meaning-vs-frozen and noise-floor controls on
  code + stdlib at S=32 (where §76 stopped winning); then the port rule vs 0.217011 (11 MB
  `corpus_big`) AND the stdlib corpus at 10 MB.
- Expected failure mode to watch: query cold-start for rare current words — the `semfast`
  subword-composed init already exists to mitigate it.

**Secondary paths, in order:**
1. **Slots for engines WITHOUT a word-model family.** The absorber is strong's word models; a
   minimal/embedded engine variant (or the DNA engine with domain-appropriate tokens) lacks
   them — slots may pay immediately there. Cheap to test: strip word/prev-word/high orders from
   a strong variant and A/B slots on `corpus_big`/stdlib.
2. **Genuinely long-range domains** (the stdlib corpus is in place; add cross-referential
   corpora: legal/technical documents, configuration graphs).
3. **Scale beyond CPU** when a channel finally clears beats-strong: the layout is already
   GPU-shaped (flat tables, dense mixer); the serial wall is broken by batching independent
   streams in lockstep (see the §39/README notes).
4. **Unchanged side assets:** `honestmap.py` structure triage remains productizable now;
   `separation.py` engine/knowledge separation remains the strain's distinctive guarantee.

## 6. Practical gotchas (learned the hard way)

- **Windows file lock:** `cargo build` silently produces "failed to remove strong.exe" while a
  run is live, and the OLD binary then answers your "new" A/B (it bit us once — the tell is
  identical numbers across configs). Kill all `strong.exe` before rebuilding; re-verify the
  default bit-identity (0.231704 @300 KB `corpus.txt`) after every rebuild.
- **`.gitignore` whitelist:** new top-level deliverable `.py` files must be added to the
  whitelist or they vanish from `git status`. `_*.py`, `probe_*.py`, `*.txt`, `data/` are
  deliberately ignored (scratch).
- **Mask determinism:** `genmem.clean_mask` uses salted `hash()` — fine within one process,
  wrong across processes. `wstate.clean_mask_det` is the deterministic replacement.
- **Conventions:** §-numbered commits; every ledger section carries its controls and negatives;
  a crossing without seed replication is not a verdict; a production claim needs the paired
  same-binary baseline.
- **Rust toolchain:** `/c/Users/aali_/.cargo/bin/cargo` (not on PATH in Git Bash);
  `cd blmrs && cargo build --release --bin strong`.

## 7. Commits and ledger map for this session

| § | Commit | One line |
|---|---|---|
| 72 | `7d1df61` | the parity wall falls on the instrument; port honest negative |
| — | `073be07` | ledger §72 + README |
| 73 | `6a6414b`, `4f38f91` | hard similarity fails; subword init; baseline-arm crash fix |
| 74 | `ac9b603` | soft retrieval in strong.rs, measured inert |
| 75 | `7c06e8b` | frontier sweep: enwik8 30 MB, code home turf, DNA negative, absorber |
| 76 | `37eeef7` | shared recency-discounted heads; the readout-capacity ladder |

Ledger: `learned_binary_address_machine.md` §69–§76. README row for `wstate.py` is current.
