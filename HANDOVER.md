# HANDOVER — the LBLM intelligence track, post-§81

> **§80–§81 update (2026-09-14):** the owner adopted the §79 engine flags as defaults
> (`BLWNS=1`, `BLPVEC=2`; `BLWNS=0 BLPVEC=0` recovers the pre-§80 engine bit-identically;
> commit cc0af5d). §81 then built the "simplest selector" and **learned long-range selection
> binds for the first time in the project's history** — +1.6…+2.3 bits/outcome at 2/12/24-word
> gaps on the synthetic probe, and on real corpora a measured binding signal (+0.008…+0.027
> beyond the same-machinery no-binding control, growing with data) that the LEAN form
> (no EMA/bucket overhead; post-hoc, labelled) converts into net held-out wins against the
> nonstationary rail: stdlib +0.0008/+0.0041 (1.2/2.4 MB), wt103 +0.0039 (2.7 MB) — exact
> deterministic numbers. Mechanism: full-weight cells for ALL candidates (kills the §77.4
> chicken-and-egg), a QUERY-FREE contextual usefulness gate keyed by the same 3-byte context
> as the cells (a global per-word score measurably rewards merely-predictable words), a
> CONTEXT-SELECTED vote readout (§78B's dilution fix, measured +2…+4 at deciding contexts),
> and sentence-scoped candidates (cross-sentence cell pollution otherwise measured 183
> updates for 37 own sentences). Next: port the lean channel to strong.rs per the §78.3
> lesson (all three mixers) under the beats-strong rule. Details: ledger §81.

---

# Previous state (post-§79)

**Scope of this document:** everything needed to continue after the 2026-09-14 session (ledger §77–§79),
which audited the §72–§76 arc, settled its two loose ends, found a production improvement and closed
learned long-range selection for the gates tested. Read this + `learned_binary_address_machine.md`
§77–§79 first. §72–§76 remain in the ledger as a record; §77.5 lists which of their claims still stand.

---

## 1. The one-paragraph state of the project

The project is an online, single-pass, bit-level predictive machine (compressor = intelligence
substrate). Its production engine `blmrs-strong` is lpaq1-class: full enwik8 0.199145 bits/bit
(19.91 MB), 0.217011 on the 11 MB `corpus_big`, 0.127624 on the 10.6 MB Python stdlib. The intelligence
track asked whether *learned memory channels* beat plain statistics on leak-free held-out data. §77
found the §72–§76 "crossing" was a learned current-word model created by an instrument bug, and the
engine port trained its vectors the wrong way. §78 found that learned vector's instrument gain is mostly
*forgetting* that the instrument's cumulative counts lack. Chasing that into the engine (§79) produced
the arc's one production improvement: **forgetting in the word models (`BLWNS=1`) plus the learned
prefix vector (`BLPVEC=2`) takes full enwik8 to 0.196123 (19.61 MB, −1.5 %), decodability verified by a real encode/decode round trip on corpus_big (11 MB).** Its pre-registered adoption rule did not fire, so both flags stay default
OFF pending the owner's decision (§5). Every learned way of *selecting* a distant memory slot tested so far failed: softmax and cosine attention, a responsibility gate and its amendment, all on a synthetic probe and all querying from slot 0.

## 2. What §77–§79 found (numbers in the ledger)

1. **§77 instrument defect.** `wstate.py` put the growing prefix id into the slot LRU at every letter byte
   (six slots ≈ two words). `WSLOTMODE=word` stores completed words; the default `prefix` stays
   bit-identical to §72–§76.
2. **§77 engine defect.** `strong.rs` trained slot vectors by gradient ascent since §72. Fixed. `BLSOFT`
   votes a mis-keyed statistic; documented, not fixed.
3. **§77 real word memory loses.** Whole-word slots (6 and 32) are worse than the orders baseline at every
   size on code and wt103 (seed 0); in production they add ~0.
4. **§77/§78 selection.** No content-attention arm bound the cue at any gap, even two words back. A forced oracle shows
   the vote store works once the right slot is handed to it (+2.5 of 3 bits at 24 words), so selection
   is the bottleneck (§78.4).
5. **§78 the learned prefix vector is mostly forgetting.** On top of an exact count word model it adds
   +0.011…+0.030 bpb on code, wt103 and stdlib (pre-registered pass), but with PAQ nonstationary counts
   that margin falls to +0.0004…+0.0031 and fails on wt103 at 150–1200 KB (nonstationary re-run on code and
   wt103 only). **The instrument's orders-only rail is
   0.13–0.37 bpb weak** because its counts never forget; every instrument crossing in §70–§78 was
   measured against it.
6. **§79 production.** Pre-registered Q1–Q3 pass on corpus_big, stdlib and enwik8 first 30 MB (Q1/Q2 are new
   evidence only on enwik8, since the other two repeated §78.3's scratch numbers); on
   held-out data (79H) the gains hold on the enwik8 tail and repo code, and also on full enwik8 (not held out: it contains the 79A first 30 MB):

   | corpus | base | `BLWNS=1` | `BLWNS=1 BLPVEC=2` |
   |---|---|---|---|
   | full enwik8 100 MB | 0.199145 | 0.196841 | 0.196123 |
   | enwik8 last 30 MB | 0.204666 | 0.202230 | 0.201627 |
   | repo code 770 KB (obits 24) | 0.185380 | 0.182759 | 0.182558 |
   | E. coli 4.6 MB | 0.240340 | 0.240340 | 0.240341 |

   H1 failed only because the DNA file has no word boundaries, so the forgetting rule never triggers
   there (0 times in 37.1 M bits). The vector is not a memory effect (a 4.7 MB table beats +4.1 GB of
   counts) and costs +1.3–1.5 % time; `BLWNS=1` costs nothing measurable.
7. **§79 learned gates fail.** A responsibility-trained mixture-of-experts gate collapsed onto one fixed
   word; an amendment removing both diagnosed loops also failed. Kill rules fired. All tested gates built
   their query from slot 0, which is always "then" at the deciding byte.

## 3. The instruments — what exists and how to run it

| Tool | Command | Notes |
|---|---|---|
| The instrument | `python wstate.py --train T --test E --sizes a,b,c --arms A1,A2,… [--seed N]` | one process per arm; `--selftest` runs arms sequentially |
| Slot memory mode | env `WSLOTMODE=prefix` (default, legacy) or `word` | **use `word` for any claim about word memory** |
| Slot / attention knobs | env `WSLOTS` (6), `WHEADS` (4), `WGAMMA` (0.85), `WTOPM` (4), `WVORD` (3), `WVBITS` (22), `WKAPPA` (8.0) | read at import time |
| Arm families | legacy slots/proj; `attn attnr attnrec attnscr attncos`; oracles `attnorc1 attnorc4`; §78A `wcnt pvec pvecr wcntpvec wcntpvecr`; §79 `attnmoe attnmoe_uni attnmoe_orc attnmoe_fc attnmoe_fcnb attnmoe_uni_fc` | docstring v7–v9b documents each and its controls |
| Binding probe | `python _bind_probe.py --grid` / `--grid78` / `--grid79` / `--grid79b` | gitignored; criteria in its docstring; outputs `_77p_probe.txt`, `_78b_probe.txt`, `_79_probe.txt`, `_79b_probe.txt` |
| Production engine | `./blmrs/target/release/strong.exe <path> <cap> <obits>` | cap 0 = full file. §79 flags `BLWNS=1`, `BLPVEC=2` (`LR_S`, `SBITS`), `BLNSALL=1`; older `BLMSLOTS`, `BLSOFT` (mis-keyed), `NSLOTS`, `ALRS`, `SOFTK`, `SIMMIN`, `BLSTRIPW`, `BLSTRIPH`; all default OFF = bit-identical |
| Gate protocol | copy/match OFF · 13-gram decontamination (`clean_mask_det`) · frozen-vector control · scrambled floor | **the orders rail's counts never forget (§78.2)**; the scrambled floor measures noise-dimension penalty |

**Data on disk:** wt103 train/test (2.8 MB/1 MB), `corpus_big.txt` (11 MB), enwik8 (100 MB) and
`enwik8_tail30` (its last 30 MB), `stdlib_train/test.txt` (8.0/2.7 MB) and `stdlib.bin` (10.6 MB),
`code_train/test.txt` (585/195 KB), `repo_code.txt` (this repo's own Python and Rust, 770 KB),
`dna_train/test.txt` (6-mer tokens), `ecoli.txt` (one line of A/C/G/T), plus the June corpora.

**Timings (this machine):** instrument baseline ≈ 6–10 KB/s; slot arms ≈ 0.7–2.2 KB/s. `strong.exe`
≈ 0.1–0.2 Mbits/s: 11 MB ≈ 9–12 min, enwik8 30 MB ≈ 33 min, full enwik8 ≈ 90 min. Each obits-25 run
needs ~4.5–5 GB RAM.

## 4. Established laws — corrected after §79

1. **No learned memory channel measured on leak-free held-out data beats order-n plus word statistics by more than a forgetting effect.** §70 EMA state and §72–§77 whole-word slots (≤2.7 MB, seed 0) do not beat them. The §72 learned prefix vector beats a cumulative count word model by +0.011…+0.030 bpb (§78A) but by only +0.0004…+0.0031 once all counts are nonstationary (§78.2, code and wt103). Not covered: soft
   similarity with correct keying, and §76 heads and attention on real corpora in word mode.
2. **A learned per-prefix vector is mostly a forgetting word model at instrument scale** (§78.2); in the engine it
   still adds beyond forgetting (§79 Q3, H2). Compare any such channel
   against nonstationary counts, not cumulative ones.
3. **Counts that never forget make a weak baseline** (0.13–0.37 bpb in the instrument). Nonstationary
   counters are required in any instrument claim.
4. **Selection, not storage, blocks long-range binding on the synthetic probe** (§78.4), and selection must be
   near one-hot (cue plus 3 distractors loses most of the gain to dilution): a hand-given slot pays ~2.5 of 3 bits
   at 24 words; every learned selector tested (softmax, cosine, responsibility gate, amended gate) failed,
   all with the query taken from slot 0.
5. **Noise floors scale with dimension.** Separation from a scrambled arm is not evidence of
   information unless the floor has the same dimension count.
6. **Process laws (reinforced):** print what a mechanism actually holds; port the exact mechanism that
   won and say where it enters; test gradient signs with a toy; red-team the premise before building, not
   only the result; pre-register criteria before runs and keep the text unedited, adding corrections
   after; check that every corpus in a gate is one the flag can act on.

## 5. The way forward

**Decision for the project owner: turn on `BLWNS=1`, with or without `BLPVEC=2`, by default?**
The evidence is strong but post hoc with respect to the adoption rule: `BLWNS=1 BLPVEC=2` is 0.7 % to
2.0 % smaller on every text and code corpus of 770 KB or more (only 0.15 % on the 300 KB corpus.txt smoke run) (corpus_big −0.7 %, stdlib −2.0 %, enwik8 and repo code
−1.5 %), including the held-out ones, no harm on DNA (where `BLWNS=1` cannot act and `BLPVEC=2` costs 0.000001), decodability verified,
`BLWNS=1` free and `BLPVEC=2` +1.3–1.5 % time. `BLPVEC=2` keeps 68 % (corpus_big) and 90 % (stdlib) of its
gain with `SBITS=16` (4.7 MB). A default change belongs in its own reviewed commit, re-verifying the new default numbers.

Open, cheaper-than-before engine leads the §78–§79 red-teams measured:
- **Stronger forgetting across tables.** `BLNSALL=1` beats `BLWNS=1` on stdlib by +0.000814 (+0.003165 over base) but not on
  text; a tuned per-table rule is untested.
- **Learned per-(word prefix, phase) scalar biases** gained a further +0.008…+0.010 bpb over all-nonstationary counts in the §78.2 red-team's scratch runs (code 100/200 KB, wt103 150/450 KB; not recorded in the ledger) — an lpaq-style adaptive probability stage, not yet tried in the engine.

Model-track leads, all low prior:
- **A query not taken from slot 0** is the only untested lever for learned selection that the audit identified (§79B, §79.6).
- **Rebase the instrument on nonstationary counts** before any new instrument claim.

Assets that stand independently of the model track:
- **The production engine**, now with measured, decodable improvements behind flags.
- **`honestmap.py` structure triage** and **`separation.py` engine/knowledge separation**.
- **The byte-stream surprise signal**: the engine's per-bit cost is a training-free novelty score over
  opaque streams, the use case earlier assessments found most practical.

## 6. Practical gotchas (learned the hard way)

- **Print what the memory holds.** The prefix bug survived five sections because nobody dumped the slot
  contents.
- **Know what a flag can act on.** `word_hash` resets only on non-letters: a file with no word
  boundaries makes almost every word-model key new (99.1 % distinct on E. coli), so word-model flags are no-ops there.
- **Windows file lock:** `cargo build` silently leaves the old `strong.exe` while a run is live. Check
  `tasklist | grep -i strong` before rebuilding; re-verify 0.231704 @ 300 KB `corpus.txt` (obits 23).
- **Run long experiments from frozen copies** of `wstate.py`/`genmem.py` and of `strong.exe` (Windows
  process pools re-import from disk; agents edit files mid-run).
- **Floating-point decoding:** a real decoder must be the same binary or a pinned build; the model uses
  f64 ln/exp and RMSProp state.
- **`.gitignore` whitelist:** new top-level deliverable `.py` files must be whitelisted.
  `_*.py`, `probe_*.py`, `*.txt`, `data/` are deliberately ignored (scratch).
- **Mask determinism:** use `wstate.clean_mask_det`, not `genmem.clean_mask`, across processes.
- **Rust toolchain:** `/c/Users/aali_/.cargo/bin/cargo`; `cd blmrs && cargo build --release --bin strong`.

## 7. Commits and ledger map

| § | Commit | One line |
|---|---|---|
| 72 | `7d1df61`, `073be07` | word-slot "crossing" (§77: a learned word model via the prefix bug); port ran uphill vectors |
| 73 | `6a6414b`, `4f38f91` | hard similarity fails (§77: bucketed per-prefix predictors) |
| 74 | `ac9b603` | soft retrieval inert (§77: sign bug + mis-keyed vote) |
| 75 | `7c06e8b` | frontier sweep (§77: absorber and depth claims fall) |
| 76 | `37eeef7` | recency heads (§77: the depth target was a noise artifact) |
| 77 | `5439a5f` | audit: two defects fixed, re-gate, strip grid, attention probe kill |
| 78–79 | uncommitted | prefix vector = forgetting; oracle; engine flags `BLWNS`/`BLPVEC`/`BLNSALL`; gate kills |

Ledger: `learned_binary_address_machine.md` §69–§79.
