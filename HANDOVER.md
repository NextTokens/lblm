# HANDOVER — the LBLM intelligence track, post-§77

**Scope of this document:** everything needed to continue after the 2026-09-14 session (ledger §77),
which audited and corrected the §72–§76 arc. Read this + `learned_binary_address_machine.md` §77
first. §72–§76 remain in the ledger as a record, but §77.5 lists which of their claims still stand.

---

## 1. The one-paragraph state of the project

The project is an online, single-pass, bit-level predictive machine (compressor = intelligence
substrate). Its production engine `blmrs-strong` is lpaq1-class (0.209 bits/bit enwik8; 0.217011
on the 11 MB `corpus_big` A/B; 0.127624 on the 10.6 MB Python stdlib; 0.205801 at 30 MB enwik8).
The intelligence track asked whether *learned memory channels* can beat plain statistics on
leak-free held-out data. §72–§76 reported that word-slot binding crossed that wall. **§77 found
that the crossing was a learned current-word model created by an instrument bug (the slot memory
held word prefixes), that the engine port had trained its vectors in the wrong direction, and that
real word memory does not beat the orders baseline where it was measured correctly: whole-word slots
lose on code and wt103 (seed 0) and add ~nothing in production, and recency heads and content
attention fail a synthetic binding probe; soft similarity was never re-measured with correct keying.**
The recommendation is to close the learned-memory model track (§5).

## 2. What §77 found (numbers in ledger §77)

1. **Instrument defect.** `wstate.py` put the growing prefix id into the slot LRU at every letter
   byte. Six slots held about two words (`cat, ca, c, quick, quic, qui`); 32 slots held about 7.5.
   New env `WSLOTMODE=word` stores completed words only; the default `prefix` stays bit-identical to
   §72–§76 so old numbers reproduce.
2. **Engine defect.** `strong.rs` applied the slot-vector credit with the wrong sign (gradient
   ascent) since §72. Fixed; the default engine is bit-identical (0.231704 @ 300 KB `corpus.txt`).
   `BLSOFT` also votes the wrong statistic (the statistic of the delimiter after a similar word,
   applied to the next word's letters); documented, not fixed.
3. **What the §72 win was.** One slot holding the current prefix beats the orders baseline by
   +0.015…+0.020 bpb on code and +0.007…+0.015 on wt103 (seed 0), more than the §72 six-slot arm.
   Frozen vectors in that slot lose. It is a learned word model; production has a count word model, but
   the two were never compared.
4. **Real word memory loses.** Whole-word slots, at 6 and 32 slots, are worse than the orders
   baseline at every size on code and wt103 (seed 0), and the loss grows with data on wt103. Learned
   vectors are mostly worse than frozen ones.
5. **Production, sign fixed.** Whole-word slots add nothing on text (−0.000005 bits/bit) and +0.00004
   on code, with or without the word models (`BLSTRIPW`) and high orders (`BLSTRIPH`). Removing the word
   models reveals no hidden whole-word slot gain; the §72 channel (a learned prefix vector) was never
   tested in production.
6. **The "information at depth" was an artifact.** Scrambled arms lose ≈0.002 bpb per noise
   dimension; the growing "floor separation" measured that, not information.
7. **The attentional read fails its pre-registered probe.** On a synthetic cue→outcome stream, no
   content-attention arm (default settings) retrieves a cue 2, 12 or 24 words back, although the cue
   is in memory 100 % of the time. Kill rule fired; no corpus grid was run. The pre-registered sanity
   check (P1) also failed; the evidence that the probe can register binding (a recency vote recovers
   ~70 % of the information when the cue is recent) is post hoc. Red-team measurement (3 cells, one
   seed each): when the cue is in the vote set its entry is right on 84–94 % of informative bits, but
   the mixer weight on the vote is negative, so correct votes hurt; that training then pushes attention
   away from them is inferred from the sign, not traced.

## 3. The instruments — what exists and how to run it

| Tool | Command | Notes |
|---|---|---|
| The instrument | `python wstate.py --train T --test E --sizes a,b,c --arms A1,A2,… [--seed N]` | one process per arm; `--selftest` runs arms sequentially |
| Slot memory mode | env `WSLOTMODE=prefix` (default, legacy) or `word` | **use `word` for any claim about word memory** |
| Slot depth / heads | env `WSLOTS` (6), `WHEADS` (4), `WGAMMA` (0.85) | read at import time |
| Attention knobs | env `WTOPM` (4), `WVORD` (3), `WVBITS` (22), `WKAPPA` (8.0, `attncos`) | read at import time |
| Arms | `baseline bitsonly randemb learned slotsr slots slotsw semsim semfast matchbase matchslots proj projr projscr scrambled attn attnr attnrec attnscr attncos` | docstring has the control ladder for each family |
| Binding probe | `python _bind_probe.py --smoke` / `--grid` | gitignored scratch; pre-registered criteria in its docstring; writes `_77p_probe.txt` |
| Production engine | `./blmrs/target/release/strong.exe <path> <cap> <obits>` | cap 0 = full file. Knobs: `BLMSLOTS`, `BLSOFT` (mis-keyed), `NSLOTS` (≤64), `SBITS`, `LR_S`, `ALRS`, `SOFTK`, `SIMMIN`, `BLSTRIPW`, `BLSTRIPH`; all default OFF = bit-identical |
| Gate protocol | copy/match OFF · 13-gram decontamination (`clean_mask_det`) · frozen-vector control · scrambled floor | comparisons valid only within a size; the scrambled floor measures noise-dim penalty, so size it to the arm |

**Data on disk:** wt103 train/test (2.8 MB/1 MB), `corpus_big.txt` (11 MB), enwik8 (100 MB),
`stdlib_train/test.txt` (8.0/2.7 MB) and `stdlib.bin` (10.6 MB), `code_train/test.txt`
(585/195 KB), `dna_train/test.txt`, plus the June corpora and genomes.

**Timings (this machine):** instrument baseline ≈ 6–10 KB/s; slot arms ≈ 0.7–2.2 KB/s (a 32-slot wt103
arm over all four sizes took ≈ 2.2 h with other jobs running).
`strong.exe` full engine on 11 MB ≈ 9–11 min; stripped variants 5–10 min.

## 4. Established laws — corrected after §77

1. **No learned memory channel measured on leak-free held-out data beats order-n plus word
   statistics** (§70 EMA state; §72–§77 whole-word slots at ≤2.7 MB, seed 0). §71's hand-built
   counters reached parity at scale. Not covered: soft similarity (never measured with correct keying),
   §76 heads and §77 attention in word mode (synthetic probe only), and the §72 learned prefix vector
   against a count word model (never run). The match model gave +0.011…+0.014 in §69–§71 but was worse
   than no match in the §75 instrument arms under the same decontaminated gate, so even the
   memorisation gain is unsettled.
2. **A 32-unit EMA superposition did not bind a 2-word cue with a linear or bucketed readout** (§72
   v1–v3; not re-tested after §77, and the prefix-slot "pass" it was contrasted with was not cue
   binding). Binding identity in whole-word slots does not help either: they lose on code and text.
3. **A learned per-word vector is a word model.** Any "win" by a channel whose slot 0 is the
   current word must be compared against a word model, not an orders-only baseline.
4. **Noise floors scale with dimension.** Separation from a scrambled arm is not evidence of
   information unless the floor has the same dimension count.
5. **Process laws (reinforced):** check that the mechanism does what the text says (print what the
   memory holds); port the exact mechanism that won; test gradient signs with a toy before an A/B;
   replicate across seeds before a verdict; pre-register kill rules; negatives go in the ledger.

## 5. The way forward

**Recommendation: close the learned-memory model track.** The arc's own kill rule fired, and every
§72–§76 positive that was decomposed traced to a bug or a word model.

If model work continues anyway, two narrow loose ends have measured targets:
- **Forced-selection oracle for the vote-readout trap (§77.4).** Force the cue into the vote set
  during training and test, log the vote mixer weight. If the weight turns positive and the gain
  appears at G=12/24, the vote readout works once selection is given and selection is the remaining
  problem; if not, the vote readout itself fails at long gaps. Needs a new arm in `wstate.py`; the
  whole 58-job probe grid took about 32 min at 4 jobs in parallel.
- **The §72 learned prefix vector against a count word model** on the same gate. It settles whether
  that channel is anything more than a word model. Low prior.

Assets that stand independently of the model track:
- **The production engine** (lpaq1-class, bit-identical defaults, full ablation flags).
- **`honestmap.py` structure triage** and **`separation.py` engine/knowledge separation**.
- **The byte-stream surprise signal**: the engine's per-bit cost is a training-free novelty score
  over opaque streams, the use case earlier assessments found most practical.

## 6. Practical gotchas (learned the hard way)

- **Print what the memory holds.** The prefix bug survived five sections because nobody dumped the
  slot contents. A short script that printed them found it.
- **Windows file lock:** `cargo build` silently leaves the old `strong.exe` in place while a run is
  live. Kill all `strong.exe` before rebuilding; re-verify 0.231704 @ 300 KB `corpus.txt` (obits 23).
- **Run long experiments from a frozen copy** of `wstate.py` + `genmem.py` (Windows process pools
  re-import from disk); this session used snapshots in the scratchpad.
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
| 77 | uncommitted | audit: two defects fixed, re-gate, strip grid, attention probe kill |

Ledger: `learned_binary_address_machine.md` §69–§77.
