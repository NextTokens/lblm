# HANDOVER — the LBLM intelligence track, post-§89

> **§91 update (2026-09-18) — the readout story closed; the tag adopted.** The hybrid readout
> (`selhyb`: shared prior + per-word residual, the spectrum's fourth and last point) fails I1 and I3
> under its own pre-registered reverse-course rule: it inherits the per-word starvation, and the
> §85 lever's readout component is **unsolved at this data scale**. The track's open limits are now
> scope (the fact instrument's memory still wipes at '.'), saturation (104.8M keys vs 4.19M cells,
> the §86.8 regime account), and scale (GPU/batch-over-streams, never started). In the engine,
> `BLSELTAG=1` is adopted as the default under the §80 standing directive (beats untagged on every
> corpus incl. held-out; full enwik8 0.194567; `BLSELTAG=0` recovers bit-identically; new 300 KB ref
> 0.230742). Details: ledger §91.
>
> **§89 update (2026-09-17) — the cheap fix, and a mechanism claim I got wrong. Read ledger §89, then §87.**
>
> **What holds.** An 8-bit tag per vote cell with evict-on-mismatch (`WSELTAG=1`, v14, default OFF and
> bit-identical when off) costs **4 MB on a 67 MB table** and reaches the fact-level memory of a **4.29 GB**
> table — **parity at 1/60 of the memory**, past its registered bar on both streams. On §87's estimand
> (paired `LA − LM`, 4,079 events): adopted 2^22 **+0.412** → 2^28 −0.231 → **tagged 2^22 −0.315**. The
> paired tagged-minus-2^28 difference is −0.0730 [−0.2175, +0.0766] — a **tie**, not a win; an earlier draft
> claimed it beat 2^28 and that is withdrawn.
>
> **What I got wrong.** I read the gain as *purity* — one key's estimate instead of a blend of twenty-five.
> A red-team ablation refuted it: with **no tags at all**, cells wiped at random at a matched rate and reads
> refused at random at a matched rate — **zero identity information anywhere** — `LA − LM` reaches
> **99.3 % of the same travel**, with cells holding a median of **one count**. It is **shallowness**, not
> ownership, that moves the number. "Sharing, not scarcity, was the defect" is withdrawn.
>
> **Cost.** Tagging worsens bits/byte on both measured streams and on the calibration slice, but *improves*
> it on the 2.7 MB train slice of the same run (−0.002141), so §86.8's regime account is **left open**, not
> confirmed. And the claim that the engine "can have both" is withdrawn as unlicensed: §86.6 measured the
> engine's tag in bits/bit, and §87's whole point is that bits/bit cannot answer the memory question.
>
> **Still true after all of it:** the machine does not recall facts. Cue evidence −0.2832; the served gain is
> significantly negative (−0.00462 [−0.00897, −0.00038]); the repaired cell still codes the fact's identity
> byte at 5.105 bits against the plain rail's 3.746.
>
> **§89A — the first configuration where better memory and better compression coexist.** Laddering
> `WSELTOPM` over {4, 8, 16, 32} with the tag on: reach at g ≥ 6 goes 0.012 → 1.000, and on the `|T|`-matched
> within-event contrast (the cue against its served frequency-matched partner, same event, same bits, same
> mixture size) the cue goes from **−0.1030 [−0.3009, +0.0948] to +0.1338 [+0.0950, +0.1687]**. bits/byte
> improves monotonically to 2.242265. **Tagged + `WSELTOPM = 32` beats untagged + `WSELTOPM = 4` on BOTH axes
> at once** (`LA` 5.081 vs 7.261; bpb 2.242265 vs 2.244060) — the first time in this project. Cost: **+2.6 %
> wall time, zero memory.** The intervention is surgically clean: the vote cells are bit-identical at all four
> rungs, so nothing is learned differently, only read differently.
>
> **Three corrections the red-team forced, all mine.** (1) The compression gain is a **tag × width
> interaction** — the untagged ladder goes the other way (2.244060 → 2.244480), so "selection moves both
> together" is false in general. (2) The ladder's negative left end is carried by empty and mis-owned cells
> at `|T| = 4` (112.5 % of it), not by anti-informative memory. (3) Only about half is fact-specific
> (+0.0218 [+0.0109, +0.0344]); the same cue crosses against a neighbouring non-fact outcome too.
>
> **WITHDRAWN IN FULL: "the gate was serving the cues whose memory was least needed."** That was Simpson's
> paradox — `a_cue` is 0.89-correlated with the softmax denominator (sentence length) and `|e|` scales as
> `1/cand_n`. Hold it fixed and the ordering reverses; on the within-event control the gate's ordering is
> monotone **in its favour**, and its learned term predicts better evidence in 13 of 13 cells. **The shipped
> gate selects well; what limited it was how few it was allowed to serve.**
>
> **Next: the readout** — §87's untouched third limit. The served gain in bits at fact outcomes is still ≈ 0
> even at `WSELTOPM = 32`. §86.3 says the bucketed fix is the worst arm on real text, so it must be per-word
> or per-cell.
>
---

# Previous state (post-§87)

> **§87 update (2026-09-17) — the fact-level instrument for real text. Read ledger §87 first.**
>
> **Why it exists.** §86.8 proved corpus bits/byte can *reward* the very defect that destroys a specific
> memory, so every binding claim about real text so far (§81's corpus crossings, §86.3's arm ranking) rested
> on a measure that cannot see this track's question. §87 is the missing measure: for a pair of words that
> genuinely go together — `steiner → hitler`, `miles → km`, `superfamily → family` — it asks what the machine's
> memory *of the cue* was worth at the moment the outcome was coded, one fact at a time.
>
> **The headline, on the load-bearing population** (1,661 fact events served at every rung with a served
> frequency-matched partner — a within-event, same-byte, same-bits contrast no selection imbalance can enter):
> at the **shipped** table size a word's own memory of its fact is significantly *worse* than the memory of a
> word standing beside it by coincidence (`LA − LM` = **+0.290 bits [+0.046, +0.535]**). De-collide the table
> and that reverses (**−0.121** at 2^28), a fact-specific repair of **−0.4106 bits [−0.7284, −0.1570]** (paired
> DiD; **+0.4514 logits [+0.2478, +0.6871]** on cue evidence). **And bits/byte gets monotonically WORSE at
> every step** (2.278784 → 2.284920, ρ = +1.000).
>
> **The independent 890 KB replication is stronger** (4,254 paired events): `LA − LM` runs
> **+0.384 [+0.233, +0.540] → −0.237 [−0.396, −0.073]**, so *both* endpoints exclude zero and the sign change
> is significant at both ends; paired DiD **−0.6211 bits [−0.8095, −0.4305]**. bits/byte is monotone worse
> there too (2.244060 → 2.250980 against a rail of 2.253037) — de-collision hands back three-quarters of the
> whole channel's compression value while repairing the memory.
>
> **The machine still does not recall facts on real text.** At every rung the served evidence is negative
> (−0.64 → −0.25). The cue is in the slot LRU on 100 % of events, in the served vote set on 21 %, and beyond
> nine words on 0.15 %. The shared readout converts 2 bits of content into +0.0006 bits.
>
> **Two structural facts, from the code and a model-free replay:** the memory is **sentence-scoped** (a `.`
> wipes the whole slot LRU — 2,220 of 11,751 events are structural zeros), and it is **saturated**:
> 104,860,139 distinct vote keys over 301 M cell reads against 4,194,304 cells — 100 % collided, ~25 keys per
> cell (99.8 / 79.0 / 32.3 % at 2^24/26/28).
>
> **Both red-teams changed the result.** The first (pre-run) showed §86's cue-attributable gain is
> sign-inverted on real text and forced a readout-free estimand. The second (pre-write-up) withdrew three
> headline sentences of the first draft: the unconditional mean factorises as `P(served) × E[e|served]` and its
> sign was set by the service rate, not by memory content. Two of §87's own controls are defective and are
> reported as such (C-NULL is 70 % facts; C-ELSEWHERE's pooled sign comes from its `j+1` half).
>
> **Registered verdicts:** R4 **PASS** (thin on its registered 17-row stratum; carried by the 1,900-event
> served-conditional population and the paired DiD). R1 **FAIL** and uninformative in both directions. R2
> **indeterminate**, and the draft's reading of it withdrawn. R3 0.056 → 0.500 across the ladder.
>
> **New:** `_factprobe.py` (registration verbatim in its docstring, extraction, job, self-tests, report, grid);
> `wstate.py` v13 — `sel_dnv` (diagnostic-only exact pre-squash dot) and the `WSELTOPM` / `WSELPBD` env knobs,
> all defaulting to the §81 literals and verified bit-identical. **§89 is pre-registered blind** in
> `scratchpad/p87/prereg_89.md`: §89B ports `BLSELTAG` into the instrument (does tagging at 2^22 buy the 2^28
> rung for 4 MB instead of 4.29 GB?), then §89A ladders the two selection knobs.

---

# Previous state (post-§86)

> **§86 update (2026-09-17) — an 8-hour session on the intelligence path. Read ledger §86 first.**
>
> **The capability, stated plainly:** the machine is an online associative memory that learns a new
> long-range fact (a never-seen cue bound to a never-seen outcome, 24 words apart) from **ONE exposure**,
> and with the two §86 fixes it no longer damages the facts it already holds when unfamiliar material
> arrives. A second, unlooked-for capability: trained only on familiar material it detects a novel
> context and *inverts* its memory readout, saving 5–7 bits before it has any fact to recall.
>
> **The binding limit is ADDRESSING, not capacity or selection.** The instrument's "dead facts" (8 of 96)
> and the engine's 42 % colliding vote-cell reads are the same defect: collisions poison cells rather
> than dilute them. An 8-bit tag (`BLSELTAG=1`, default OFF) buys ~3 address bits — held-out gains
> +0.000374 (enwik8 tail) and +0.000587 (repo code), and +0.000935 on stdlib — for 8 MB, matching what
> gigabytes of untagged table buy. Full-file headline with it: **enwik8 0.194567 (19.46 MB)** vs the adopted
> default's 0.195780 — +0.001213, the largest single step of the §79–§86 arc. Validated, documented,
> **not adopted**: the owner's call under §80.
>
> **Two structural bugs in the standing record:** (1) `sellean`, the arm that produced §81's corpus
> crossings, has NO learned selection (two stale `arm == "sel"` guards); the real arm is added as
> `selleanu` and beats it everywhere (wt103 2.7 MB +0.0073 vs +0.0039). (2) The shipped engine has had
> no bias input to its context-selected mixers since §82 (`BLSELBIAS=1` fixes it; effect ~1e-5).
>
> **Probe results do not transfer by default:** the reliability-bucketed readout is decisive on the probe,
> the *worst* arm on real text, and near-noise in the engine. Fast trust transfers; bucketing does not.
>
> **Collisions destroy specific memories but help aggregate compression (§86.8).** A bigger, less-colliding
> instrument table is WORSE on corpora (stdlib 2.1150 → 2.1181 at 2^26) while it revives every dead fact on the
> probe and tagging pays in the engine. Merging keys is fatal to one fact and mild backoff in aggregate; the
> regimes separate by data per cell. Consequence: corpus bits/byte can reward the very defect that destroys a
> specific memory — binding claims on real data need a fact-level recall measure, not a compression measure.
>
> **Method lesson, twice learned:** a metric over a whole channel does not measure the part under test.
> §86's registered metric passed arms whose binding was actively harmful; the cue-attributable
> counterfactual is the correct one and is what every §86 number above uses.

---

# Previous state (post-§85)

**Scope:** everything needed to continue after the 2026-09-14…16 sessions (ledger §77–§85). Read this +
ledger §84–§85 first, then §81 and §77–§79 as needed. §72–§76 remain as a record; §77.5 lists which of
their claims still stand. The owner's stated priority is the **intelligence track** (learned memory and
selection), not compression ratio; compressor gains are side results.

---

## 1. The one-paragraph state of the project

The project is an online, single-pass, bit-level predictive machine. Its production engine `blmrs-strong`
now defaults to forgetting word models (§79/§80), a learned prefix vector (§79/§80) and a shallow learned
word-vote selector (§82–§84): full enwik8 **0.195780 bits/bit (19.58 MB)**, −1.7 % over the arc, every
step verified decodable. On the intelligence track: §81 built the first learned selector that binds distant
content on the synthetic probe (2.0 of 3 bits at 12 words, 1.7 at 24); on real text the instrument shows
depth pays (§85.1) but the engine's word and match models already cover it, so production keeps only the
depth-2 vote (§84). §85's few-shot probe then found the machine **stores a new fact on first exposure but
trusts it slowly, and its one shared readout dial lets unreliable new memories degrade reliable old ones**
(known-cue gain 1.7 → 0.5 bits when eight unknown pairs join the stream). That shared readout is the
next structural lever.

## 2. What is established (numbers in the ledger)

1. **§77 defects (fixed):** the instrument's slot memory held prefixes; the engine trained slot vectors
   uphill. **§78:** the §72 "meaning" vector was mostly forgetting; the instrument's cumulative-count rail
   is 0.13–0.37 bpb weak. **§79/§80:** `BLWNS=1 BLPVEC=2` adopted.
2. **§81 selection binds on the probe** (query-free contextual-usefulness gate, full-weight cells for all
   candidates, context-selected readout). §85.1 corrects one wording: its C3 oracle control was
   mis-specified, not "failed as designed"; the claim rests on C2a/C2b, which passed.
3. **§84 the engine keeps a bigram.** In the engine, 2 candidate slots beat 16 by ~6×; the top pick on real
   text is the previous word. **§85.1:** in the instrument, depth helps at every step and only depth 32
   beats the rail. Both hold at their own scale: the engine's word family absorbs the deep part.
4. **§85.1 the adopted channel is verified:** real arithmetic-coded round trip on 11 MB (4,601 bytes saved),
   causal under byte flips, `BLSEL=0` equals the pre-§82 engine rebuilt from git, +192 MB, +5 % time.
5. **§85.2 few-shot binding at 24 words: F1/F2/F3 fail.** Storage is immediate (new cell at 0.16 bits by
   ten exposures); the gate gives a new cue half an old cue's share; the shared per-context readout weight is
   driven down by confidently wrong votes on new-cue sentences (phase-5 weight 0.06 vs 0.84 in the
   known-only control), and re-reading through the control's weights recovers 88 % of the loss.

## 3. The instruments — what exists and how to run it

| Tool | Command | Notes |
|---|---|---|
| The instrument | `python wstate.py --train T --test E --sizes a,b,c --arms A1,… [--seed N]` | one process per arm; env `WNS=1` for the nonstationary rail (**required**), `WSLOTMODE=word`, `WSLOTS` |
| Selector arms | `sel selpos selorc4 sellean` (v10); older `attn*`, `attnmoe*`, `wcnt pvec …` | docstring v7–v10 documents each and its controls |
| Binding probes | `python _bind_probe.py --grid` / `--grid78` / `--grid79` / `--grid79b` / `--grid81`; `python _fewshot_probe.py --grid` (env `FS_SNAP=<snapshot dir>`) | gitignored; criteria in docstrings; outputs `_77p/_78b/_79/_79b/_81_probe.txt`, `_85_fewshot.txt`; per-job JSON in the scratchpad |
| Production engine | `./blmrs/target/release/strong.exe <path> <cap> <obits>` | defaults ON: `BLWNS=1 BLPVEC=2 BLSEL=1 NSELSLOTS=2 SELVBITS=SELUBITS=23`; `BLSEL=0` → 0.231368, `BLWNS=0 BLPVEC=0 BLSEL=0` → 0.231704 at 300 KB `corpus.txt` obits 23; **current reference 0.230863** |
| Round-trip harness | scratch `rt79/patch_cod.py` applied to a copy of `strong.rs` (`MODE=enc/dec`, `DUMP=`) | rebuild it for any new channel before adoption |

**Instrument snapshots matter:** the §81 probe numbers come from a dict vote store (scratch `snap81`);
the committed instrument uses a hashed 2^22 table whose collisions cut the probe gain from 1.68 to
1.43 bits at 24 words. Run probe comparisons on the snapshot the reference came from.

**Data on disk:** wt103, `corpus_big.txt`, enwik8 and `enwik8_tail30`, stdlib train/test and `stdlib.bin`,
`code_train/test.txt`, `repo_code.txt`, `dna_train/test.txt`, `ecoli.txt`, the June corpora.
**Timings:** instrument selector arms ≈ 25–60 min per corpus; `strong.exe` 11 MB ≈ 8–10 min, full enwik8
≈ 80–90 min, ~4 GB RAM per obits-25 run.

## 4. Established laws — after §85

1. **No learned memory channel beats order-n plus word statistics on leak-free held-out text by more than
   forgetting plus a shallow word vote** (§78, §84). Deep binding is real in the instrument (§85.1) and
   absorbed by the engine's word and match models.
2. **Counts that never forget make a weak baseline**; nonstationary counters are required in every
   instrument claim (§78.2).
3. **Selection is learnable online** with cells-for-all, a query-free contextual gate and a
   context-selected readout (§81); attention-style gates queried from a constant slot were not (§77–§79).
4. **Storage is immediate, trust is slow, and a shared readout is an interference channel** (§85.2).
5. **Noise floors scale with dimension** (§77); calibrate registered thresholds against the instrument's
   known-good values before registering (§85.2 F2); never use whole-word gain on never-seen words as a
   binding measure (§85.2 F1).
6. **Process:** print what a mechanism holds; port the exact mechanism and say where it enters; red-team
   the premise before building; pre-register with a branch for every outcome (§81's tree lacked one) and
   keep the text unedited, adding corrections after; verify decodability before adopting a default.

## 5. The way forward

**Next lever (pre-register before building): per-word or per-reliability readout weighting.** Replace the
one readout dial shared by every word at a context with one keyed by the selected word's own usefulness
bucket, or gate the vote by the candidate's recent reliability. Criteria on the §85 probe: known-cue gain
in the 16-cue stream returns to within 0.3 bits of the 8-cue control (1.70 / 1.64), and exposures-to-bind
at 24 words becomes finite. Cost: a day. If it passes, re-run §81's corpus phase and the engine port rule.

**Known structural limits (do not spend a week finding them):** every learned parameter is per word, so
the machine cannot compose two cues or apply a rule to an unseen cue; shared representations trained by
gradient would be needed, which is the language-model path.

**Engine side-leads (owner's call):** `BLNSALL=1` beats `BLWNS=1` on stdlib by +0.0008; a per-(prefix,
phase) adaptive bias stage gained +0.008…+0.010 bpb over nonstationary counts in the §78.2 red-team's
scratch runs; `SELVBITS=16`-style smaller tables were not tried for the selector.

## 6. Practical gotchas

- **Print what the memory holds**, and dump what a gate selects (§84's probe predicted the controls).
- **Know what a flag can act on**: a file with no word boundaries makes word-model flags no-ops.
- **Windows file lock:** check `tasklist | grep -i strong` before `cargo build`; re-verify 0.230863 at
  300 KB after every rebuild.
- **Run long experiments from frozen copies** of `wstate.py`/`genmem.py` and of `strong.exe`; record the
  snapshot sha in every result.
- **Background agents can stall** after their own background children finish; check the scratch dir's
  mtimes and resume them.
- **Floating-point decoding:** a real decoder must be the same binary or a pinned build.
- **`.gitignore` whitelist:** deliverable `.py` files must be whitelisted; `_*.py`, `*.txt`, `data/` are scratch.
- **Rust toolchain:** `/c/Users/aali_/.cargo/bin/cargo`; `cd blmrs && cargo build --release --bin strong`.

## 7. Commits and ledger map

| § | Commit | One line |
|---|---|---|
| 72–76 | `7d1df61`…`37eeef7` | the slot-binding arc (corrected by §77) |
| 77 | `5439a5f` | audit: two defects fixed, re-gate, strip grid, attention probe kill |
| 78–79 | `4949ef1` | prefix vector = forgetting; oracle; engine flags; gate kills |
| 80 | `cc0af5d` | `BLWNS=1 BLPVEC=2` adopted as defaults |
| 81 | `30ca0d8`…`8b23201` | the simplest selector binds on the probe and (lean) on corpora |
| 82–84 | `34ffeec`, `904a9c7`, `d5601af`, `eaf2ec6` | port, adoption, the bigram revision (N=2) |
| 85 | this commit | §81–§84 verified; few-shot probe: storage immediate, trust slow, shared readout interferes |

Ledger: `learned_binary_address_machine.md` §69–§85.
