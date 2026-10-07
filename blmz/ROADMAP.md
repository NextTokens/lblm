# blmz roadmap: from a stable codec to a defensible one

*Draft for the owner. 2026-10-07, branch `ccr-5220a43a-vaecp0`, written against `98b9763`. Scope: the `blmz/` crate; the research
track (`blmrs/`, ledger, `prereg/`) only where it feeds model changes. Paths are relative to `blmz/` unless they start
with `.github/` or `blmrs/`.*

**Evidence tags.** **[M]** measured in the stabilisation session (contended 4-core Xeon @ 2.1 GHz unless stated).
**[S]** sourced, mostly lzbench on an EPYC 9555P. **[D]** derived from [M]/[S] numbers or source constants (Appendix A).
**[I]** landscape-audit inference under its stated assumptions. **[repo]** read from the repository at HEAD. **[A]**
assumption, target or estimate, not evidence; every effort figure and every gate threshold that does not cite a
measurement is [A]. **Units:** MB = 10^6 B, MiB = 2^20 B; `blmz` prints KB/s with 1 KB = 1024 B (`src/bin/blmz.rs`) and
the evidence does not define every competitor's unit — a ≤ 2.4 % ambiguity, far below any effect here, removed in M0.

## 0. The hard facts
1. **Stable, verified, portable.** 37 tests (36 run; `write_fixtures` is an ignored generator); all 12 Silesia streams round-trip byte-identical; bit-identical on x86_64,
   i686 and aarch64; every confirmed high/medium finding of a 31-agent review fixed [M].
2. **Pareto-dominated.** On 12 × 1 MB Silesia slices zpaq -m5 is 3.5 % smaller than blmz's real stream and ~2.3–4.1×
   faster (blmz CPU time against competitor speeds of unstated basis on a loaded host; screening-grade); kanzi -9 is 1.1 % larger and ~6.1–10.7× faster [D]. (The often-quoted "3.3 %, 10×, 28×"
   compares against the research engine's ideal size and 18.8 KB/s *wall* time.) On full `dickens` blmz -6 is 4.9 %
   larger than zpaq -m5 and 2.7 % larger than kanzi -9, and slower than both [M, D]. Full Silesia: blmz -6 42.04 MB [M];
   zpaq -m5 39.11 MB, kanzi -9 41.81 MB @ 2.48 MB/s, Gleipnir -9 35.47 MB @ 0.32 MB/s [S].
3. **Speed is the production gate.** 53.9 KB/s aggregate on Silesia (30–82 per file), symmetric: **~5 h per GB each
   way** [M, D]. The f64 compute floor is ~82 KB/s [M]; no change that keeps model 1's bitstream reaches zpaq's
   ~200 KB/s here [D] (floor ~82–95 KB/s vs ~200 KB/s).
4. **Text ratio is the asset, unproven at scale.** The research engine's ideal size beats zpaq -m5 on 1 MB slices of xml,
   reymont and webster (−5.3 / −4.5 / −2.0 %) [M]; blmz's coded files are 0.1–1.1 % larger than that ideal. The one full
   text file measured, dickens, is not among those wins and loses to zpaq; G0.2 decides at full size.
5. **Every model change is a format change; released streams decode forever** (`FORMAT.md`, `tests/fixtures/`). Preview
   streams too — so **every release is a freeze**.

## 1. Thesis
1. **A correct codec nobody should adopt yet**; until it leaves the dominated corner, the README says *preview*.
2. **With known levers, speed is the only exit from every dominator.** A ratio-only win (≤ 39 MB) still leaves Gleipnir
   -9 smaller and probably faster (0.32 MB/s there ≈ 67–148 KB/s here [D]). Throughput comes from **one new model id
   built for speed** (model 2) plus frames on every core.
3. **The text lead is narrower than the headline**, so M0 re-tests it at full size before the 10–16 eng-week model-2
   spend (go/no-go D2).
4. **Product: a general-purpose high-ratio codec for data that is written once and read rarely** — storage is scarce
   and compression demand is broad, so the target is the general cold/warm tier (backups, archives, datasets, logs,
   binaries), not a text niche. Text is where blmz already leads; general data is where the measured gaps are, and that
   is what the model work targets (§3, M4). Delivery is library/CLI first and must be **SaaS-ready**: per-job CPU and
   memory bounds, cancellation, progress, metering (§3.3). Non-goals are explicit (§3.4).
5. **Research culture becomes CI-enforced change control:** pre-registration → an RFC committed alone; "flag=0 recovers
   bit-identically" → a gate; a red-team gates every freeze (§5).

## 2. Where we are (measured)

### 2.1 Done: stabilisation
- **Model id 1** ports the adopted `strong.rs` defaults; every per-bit probability is FNV-hashed against the engine by
  `tools/parity.sh` on dickens, mozilla, sao, x-ray, ooffice (locally, obits 20–22), the ledger file and a PNG; CI `parity` re-checks only the ledger file (200,000 B) and the PNG (150,000 B) at obits 20; ~2.1× faster at
  equal tables [M]. **Portable math:** vendored FreeBSD msun `ln`/`exp`/`log2`, pinned by `math::tests::frozen_values`,
  bit-identical on x86_64/i686/aarch64, gnu and musl [M]; x87 refused at compile time.
- **Container v1** (`FORMAT.md`): CRC-checked header with the table params, per-byte end flag, length + CRC-32 trailer;
  47 B overhead; worst-case memory 36 MiB (L1) to 3.7 GiB (L9), 485 MiB at L6 [repo: `examples/levels.rs`, after the ICM fold].
- **Library, CLI, tests, CI:** typed errors, fallible allocation, `Limits`, atomic temp files; 37 tests including
  `fixtures_decode_forever` (5 streams), `golden_streams` and `hostile_header_params_are_refused_before_allocating`; CI
  jobs `test` (ubuntu, macos-arm64, windows), `test-32bit` (+ x87 must-fail), `msrv`, `lint`, `fuzz-build`, `parity`.

### 2.2 Ratio: full Silesia, blmz -6, single process [M]

| file | raw | blmz -6 | xz -9e | vs xz | KB/s |
|---|---:|---:|---:|---:|---:|
| xml | 5,345,280 | 325,310 | 434,892 | −25.2 % | 69.9 |
| dickens | 10,192,446 | 2,198,331 | 2,831,212 | −22.4 % | 51.9 |
| reymont | 6,627,202 | 940,165 | 1,315,592 | −28.5 % | 40.7 |
| ooffice | 6,152,192 | 2,157,963 | 2,427,224 | −11.1 % | 35.2 |
| sao | 7,251,944 | 4,495,622 | 4,425,664 | **+1.6 %** | 30.0 |
| x-ray | 8,474,240 | 3,671,441 | 4,491,264 | −18.3 % | 36.9 |
| osdb | 10,085,684 | 2,274,731 | 2,844,556 | −20.0 % | 44.8 |
| mr | 9,970,564 | 2,197,970 | 2,751,892 | −20.1 % | 72.0 |
| samba | 21,606,400 | 3,231,569 | 3,739,524 | −13.6 % | 59.2 |
| nci | 33,553,445 | 1,253,069 | 1,449,272 | −13.5 % | 82.4 |
| mozilla | 51,220,480 | 13,194,050 | 13,376,240 | −1.4 % | 61.8 |
| webster | 41,458,703 | 6,097,703 | 8,368,672 | −27.1 % | 46.7 |
| **total** | 211,938,580 | **42,037,924** | 48,456,004 | **−13.2 %** | 53.9 |

bzip2 -9: 54,506,769; gzip -9: 67,631,990. Decoding costs as much as encoding: a full-Silesia pass is ~64–66 min each
way [D], which slows every experiment below.

### 2.3 The Pareto picture
12 × 1 MB Silesia prefixes, same host, single thread [M]; table deltas against blmz's real output [D]. Per file vs zpaq
-m5, the research engine's ideal size wins on text (xml −5.3 %, reymont −4.5 %, webster −2.0 %) and loses on binaries
(ooffice +13.9 %, sao +14.4 %) [M]; blmz's coded files are 0.1–1.1 % larger than the ideal, so its margins are up to ~1
point smaller.
The speeds mix wall and CPU time on a loaded host: screening-grade only.

| codec | total bytes | vs blmz -6 | speed here |
|---|---:|---:|---|
| blmz -6 (real; engine ideal 3,422,088) | 3,432,826 | — | 49–86 KB/s CPU |
| zpaq 7.15 -m5 -t1 | 3,313,882 | −3.5 % | ~200 KB/s |
| kanzi 2.6 -l9 -j1 | 3,468,999 | +1.1 % | 522 KB/s |
| paq8px v217 -8 | 2,484,548 | −27.6 % | 2.1–6.3 KB/s |
| xz -9e / bzip2 -9 / gzip -9 | 3,981,316 / 3,934,362 / 4,812,226 | +16.0 / +14.6 / +40.2 % | xz 0.4–1.2 s/MB |

**Full `dickens`** (10,192,446 B) [M]: blmz -6 is dominated by kanzi and zpaq, and even the research engine's *ideal*
code length at obits 24 is 2.1 % above zpaq's file [D]. Sourced: paq8px -12L 1,856,083, cmix v21+precomp 1,802,071 [S].

| codec | bytes | vs blmz -6 | time / speed |
|---|---:|---:|---|
| blmz -6 | 2,198,331 | — | ~192 s, 51.9 KB/s |
| research engine obits 22 / 24 (ideal, not files) | 2,163,000 / 2,138,762 | −1.6 / −2.7 % | — |
| kanzi -9 | 2,140,182 | −2.6 % | 8.9 s, ~1.1 MB/s [D] |
| zpaq -m5 | 2,094,742 | −4.7 % | 54 s, ~184 KB/s [D] |

**Full Silesia, sourced** [S] (lzbench, EPYC 9555P): paq8px -12L 27.68 MB (~64,402 s); cmix v21+precomp 28.26; Gleipnir
-9 35.47 @ 0.32 MB/s, 1.19 GB; zpaq -m5 39.11; kanzi -9 41.81 @ 2.48 MB/s; ppmd8 -9 48.21 @ 2.6 MB/s; xz -9 48.77;
brotli -11 50.41; zstd -22 52.28. Ours: **blmz -6 42.04 MB @ ~54 KB/s** [M].

### 2.4 Where the time goes
- **Research engine under callgrind** [M] (before the count-pair stretch table): 6,622 instructions/bit, libm 27 %, 20.8 last-level misses/bit (simulated
  32 MiB cache), table lookups 66 % of cycles. **blmz itself was never profiled** (M0).
- **One model-1 bit** [D, `src/model.rs`]: a slot in each of 18 count tables (8 orders, 5 hashed high orders, 3 sparse,
  2 word), 2 selector cells, 2 match models, 6 APMs, three 35-wide f64 dot products, and RMS-normalised updates costing
  **109 sqrt + 109 div**, over 9.84 MiB of f64 weight/RMS state. **Cycle budget** at 2.1 GHz [D]: ~4,940 cycles/bit
  today; ~1,300 at zpaq-here speed; ~260 at 1 MB/s.
- **The ~82 KB/s floor is approximate, not hard:** L1 measured 83 KB/s on a 300 KB slice *before* the ICM fold made the
  engine 10–14 % faster [M]. Even at ~95 KB/s it is ≥ 2× short of zpaq here and ~10× short of 1 MB/s.
- **Format-preserving work** [M]. Kept (bit-identical): count-pair stretch table −24…−33 % (auditor / this host), fused dot products, next-bit
  prefetch −17 %, ICM history folded into the order slot −14 %. Small or conditional: `target-cpu=native` ≤ 5 % (left to
  M3's runtime dispatch); THP −14 % CPU at 10 MB but a loss below ~1 MB, needs `libc`; byte-boundary prefetch neutral.
  Rejected: prefetching both children of the next bit (35 % slower); GPUs (serial per-bit dependency, ~2–3 µs/bit [I]).

**Format-changing ablations** (1 MB, obits 22) [M, D], input to the exchange-rate rule (D6):

| ablation | dickens size | ooffice size | CPU | note |
|---|---:|---:|---:|---|
| STRIPH (no hashed orders 8..32) | −0.08 % | −0.31 % | −20 % | **+0.18 % on 10 MB dickens**: sign flips at scale |
| SEL0 (no word-vote selector) / PVEC0 (no prefix vector) | +0.39 / +0.12 % | +0.17 / +0.05 % | n/m | |
| SEL0 + PVEC0 + STRIPH | +0.47 % | −0.09 % | −33 % | 1.5× faster |
| no mix2 / only 2 APMs | +0.51 / +0.20 % | +0.55 / +0.78 % | n/m | these earn their keep |

### 2.5 Where the bytes are
**Headroom map** [D]: paq8px's per-file slice lead applied to blmz's full files gives 28.16 MB (sourced: 27.68 MB).
mozilla (7.96 MB) and samba (1.72 MB) hold **70 % of the 13.88 MB headroom**, mostly embedded deflate (jar, PDF) — a
non-goal; the rest is thin (webster 1.06, ooffice 0.77, sao 0.77, nci 0.37 MB). The levers that fit are in M3 (L5) and
M4, which projects their sum.

### 2.6 Known gaps
- **Robustness.** No stored fallback (random +0.13 %, model-adversarial +0.6 % [M]); no frames, so a bit flip loses the
  rest of the stream and is found only at its end (mozilla ~14 min [D]); one thread; one model id per decoder; no SIGINT
  cleanup or progress.
- **Assurance.** Fuzzing at 9–22 exec/s [M]; Actions tag-pinned (`actions/checkout@v4`) and `cargo-fuzz` unversioned; no
  big-endian or wasm CI; no `LICENSE`/`NOTICE` in `blmz/` although Cargo.toml says Apache-2.0 and `src/math.rs` carries
  Sun's msun notice [repo].
- **Spec, coupling, reproducibility.** `FORMAT.md` makes `src/model.rs` + `src/math.rs` normative, so no independent
  decoder can be written from it. `parity.sh` patches `strong.rs` HEAD, asserts its shape, and runs on every `strong.rs`
  edit. Corpora are not hash-pinned; enwik8 is unreachable here; `.blz` collides with Nintendo BLZ [S].

### 2.7 Corrections to existing docs
- `README.md` (corrected in the commit that adds this roadmap): "~54 KB/s … roughly 1 hour per GB" was really **~5.0–5.2
  h per GB** each way [D]; "~1.6–1.8×" predated the ICM fold (now ~2.1× [M]); L6 memory 223/479 MiB predated it too (now
  229/485 MiB); and its comparison listed only xz, bzip2 and gzip, hiding the two codecs that dominate blmz — zpaq and
  kanzi rows are now in it (M0 replaces them with co-measured full-size numbers).
- The decode timings (webster 851 s, nci 1,232 s, mozilla 1,730 s; ~27–48 KB/s) came from an older binary on the
  contended host: correctness, not speed.
- The evidence calls mozilla "x86 executables"; Silesia's mozilla is a Tru64 UNIX (Alpha) build ([A]: recalled from the
  corpus documentation; M0 verifies). E8/E9 is credited to ooffice only; the −1.7 % Alpha alignment context (file not recorded; presumably a mozilla
  slice, M0 verifies) is credited to mozilla.

## 3. Positioning, beachhead, non-goals

### 3.1 Position
**A general-purpose, highest-practical-ratio codec for cold and warm storage that still behaves like a codec:** a
frozen format that decodes forever, bit-exact on every IEEE platform, bounded memory and CPU per job, safe on untrusted
input, and **no slower than zpaq -m5**. Storage is scarce and compression demand is broad, so the product is judged on
*general* data (Silesia's binaries, records, executables and images as much as its text), delivered as a library and
CLI that a hosted service can meter. Forever-decodability is necessary but not a differentiator — zpaq has it [S] and
dominates blmz; paq8px and cmix own the ratio frontier but break compatibility [S]; kanzi -9 matches blmz's aggregate
ratio far faster [M]. **The open slot: ratio between zpaq -m5 and Gleipnir on general data, at zpaq speed and
zpaq-grade stability.** Text ratio is today's strength, not the product boundary.

### 3.2 Why speed decides whether there is a market (and why it depends on who pays for CPU)
Inference [I], assuming CM output 15–25 % smaller than xz, $0.02/core-hour, Deep Archive pricing [S] and S3 Standard
pricing as recalled. At **50 KB/s** (today) the extra CPU pays back against storage saved in 160–266 years on Deep
Archive and 7–12 years on S3 Standard, and over a network CM wins end-to-end only below ~14–23 kbit/s. At **1 MB/s**
these become 5–9 years, 0.2–0.4 years and ~280–460 kbit/s. Every restore pays the CPU again. **Speed sets the size of
the market:** at ~1 MB/s, S3-Standard-class archives pay back in months; the cheapest cold tier still takes 5–9 years.
**Two deployment cases, two gates.** *Hosted / SaaS:* compute is billed per job, so the payback math above applies and
M3's speed gate decides viability; per-job CPU must be bounded and metered (`max_output` is already a CPU budget; add
cancellation and progress in M1). *Self-hosted on idle machines (capacity-constrained, CPU effectively free):* the
metric is bytes reclaimed per idle core-hour, and M2's multithreading alone makes blmz usable for a cold backlog today
(8 cores ≈ 430 KB/s ≈ 35 GB/day per machine [D]) — provided the data is rarely read back, because restore costs the same.

### 3.3 Target segments and delivery

| segment | evidence | verdict |
|---|---|---|
| B0: general cold/warm storage reclamation (backups, archives, datasets, logs, binaries; self-hosted or hosted) | full Silesia −13.2 % vs xz but +7.0 % vs zpaq -m5; the losses are on records (sao +1.6 % vs xz), executables (mozilla −1.4 %), images (~8 % under bzip2) [M]; measured levers exist for each (M4) | **primary**: the storage market is general data; M4's per-type work is the product work, not a follow-up |
| B1: text-heavy archives (legal/compliance, mail, wiki/book corpora) | the measured lead over zpaq is on text only (−2…−5 % on slices, engine ideal [M]); against xz, full text files are −13.5…−28.5 % (nci … reymont) [M] | **first proof point** inside B0: the segment where blmz can be best today, used to validate the pipeline, not the product boundary |
| B2: embedded in an existing tool, or a hosted compression service | the library API exists; SaaS needs metering, cancellation, progress, per-tenant limits (M1) | **channels** for B0 |
| B3: very-low-bandwidth links (LoRa, satellite IoT, HF) | today's speed already wins below ~14–23 kbit/s [I] | **parked**: small messages, 36 MiB at L1, 47 B overhead, cold model; needs a primed small model |
| B4: LTCB entry after the freeze; genomics (`dna.rs`: E. coli 1.908, chr21 1.6255 bits/base, ledger cross-entropy, not re-measured [repo]), logs | cheap credibility [I]; CRAM 3.1 fqzcomp is already CM, CLP/OpenZL serve logs [S] | LTCB **at 1.0**; domains **no** |

### 3.4 Non-goals (through 1.x)
- **Matching paq8px or cmix** (paq8px -8 is 27.6 % smaller on slices [D]); the Hutter Prize (record 100,424,672 B;
  engine ideal ~1.34× cmix's file on enwik8 [D]: 0.195525 bits/bit cross-entropy from the ledger, not a coded size).
- **Deflate/zip/PDF recompression**: most of paq8px's lead on mozilla (+152 %) and samba (+114 %) [M], but a separate
  attack surface (post-1.0, D11).
- **Web or real-time encoding, random access inside a frame, general backup/dedup** (zpaq and kanzi own that frontier
  [S]); GPUs; non-IEEE or x87 targets; automatic adoption of research defaults (§5.5).
- **Encoder-output stability across releases** (D5): streams decode forever, bytes are identical across platforms and
  thread counts within a release, a newer encoder may differ (as with zstd and xz).

## 4. Milestones

| # | milestone | format impact | value | risk | effort [A] |
|---|---|---|---|---|---|
| M0 | measurement truth | none | gates every decision | low | 3–4 eng-weeks |
| M1 | model-1 hygiene and assurance | none (one Class C preset retune) | trust; needed for any release | low | 3–4 eng-weeks |
| M2 | container v2: frames, stored frames, threads | container version 2 | high: wall clock, robustness | low–mid | 4–6 eng-weeks |
| M3 | model 2, the fast core | new model id (experimental until frozen) | highest | high | 10–16 eng-weeks |
| M4 | per-type routing and filters, inside model 2 | same id; filter ids | mid–high | mid | 6–10 eng-weeks |
| M5 | 1.0: freeze, package, launch | freeze | — | mid | 5–7 eng-weeks |

**Schedule [A].** Critical path M0 → M3 → M4 → M5: **24–37 eng-weeks**; M1 and M2 run alongside M3 with a second
engineer. Total 31–47 eng-weeks: ~7–11 months for one engineer, ~6–9 for two. **Model 2 freezes at the end of M4**, not
M3, so 1.0 adds exactly one released id; container v2 freezes when it first ships (D7).

**Ordering for a general-storage product.** M2 (frames, stored blocks, threads) is pulled forward to ship as the 0.2
preview as soon as G1/G2 pass: it is the first version a capacity-constrained self-hosted user can run on a cold
backlog with all cores, and it stops burning CPU on incompressible data. M3 and M4 are one programme ("model 2"): M4's
per-type routing is not a refinement after the fast core but the part that makes the ratio competitive on the data
that fills storage, so its RFCs are written during M3 and its levers are benchmarked on the fast core as it lands.
For a hosted service M3's speed gate still decides viability (§3.2).

### M0: Measurement truth (first; 3–4 eng-weeks [A])
**Goal.** Replace contended, slice-based, wall/CPU-mixed numbers with the §6 protocol; re-test the text premise at full
size before the throughput spend.

**Work.**
- **Corpora and competitors.** `bench/corpora.toml` pins §6's sets by sha256 (E fetched outside this environment; H from
  the owner, D1). `bench/competitors.toml` pins and builds xz, zstd `--long`, brotli, bzip2, ppmd8, kanzi 2.6, zpaq
  7.15, paq8px v217 (slices only) and Gleipnir -9 if a build is obtainable (else its [S] row is labelled "not
  co-measured").
- **Harness.** `bench/run.sh` → JSON per run (bytes, user/sys/wall, peak RSS, `cmp`-checked decode); `bench/report.py` →
  tables and a Pareto plot in `bench/results/`.
- **Profile blmz itself** (perf, callgrind): cycles and last-level misses for orders, high orders, sparse, word,
  selector, match, mixers + `rms_update`, APMs. This ranks M3's levers.
- **Full-size runs** of blmz -1/-6/-9, zpaq -m5 -t1 and kanzi -9 -j1 on S, T and E — including **the first coded,
  round-tripped enwik8 number**.
- **Frame-size sweep** with fresh models (4–64 MiB) on dickens, webster, mozilla, enwik8. Only 1 MB (+3.3 %) and 250 KB
  (+16.0 %) frames are measured [M]; "≥ 16 MB costs < 1 %" is inference.
- **Scratch discovery and docs.** Is mozilla Alpha or x86? What share of mozilla and samba is bit-exactly re-deflatable
  (D11)? Apply §2.7.

**Exit gates.** **G0.1** The harness reproduces 42,037,924 B for L6 Silesia exactly; every number in `README.md` and
here traces to a results file (commit, host, corpus hash). **G0.2** A per-file full-size text verdict: blmz -6/-9 vs
zpaq -m5 and kanzi -9 on T and enwik8. **G0.3** The blmz profile. **G0.4** The frame-size curve, which sets M2's
default. **G0.5** The discovery reports.

**Decision point D2** (pre-registered now). Proceed if blmz -9 is smaller than zpaq -m5 on T's total and on most T
files. Otherwise the text lead is a small-input effect: M3's ratio budget goes to zero and M4's per-type levers (the
general-data work) become the whole ratio case for model 2.

**Risks.** No uncontended host yet; enwik8 blocked here; ~2 CPU-hours per full-Silesia configuration [D].

### M1: Model-1 hygiene and assurance, no bitstream change (3–4 eng-weeks [A])
**Goal.** Close the open items that leave model 1's arithmetic untouched, and build the gates every later change must
pass.

**Work.**
- **Operations and API.** SIGINT/SIGTERM removes `.blmz-partial`; `-v` progress (`src/bin/blmz.rs`, `tests/cli.rs`).
  `Encoder<W: Write>`/`Decoder<R: Read>` with `finish()`, progress and cancellation (`Error::Cancelled` is additive:
  `Error` is `non_exhaustive`); `cargo-semver-checks` in CI.
- **Dispatch.** A (version, model id) decoder table; a `Predictor` trait monomorphised per id (no per-bit branch). Model
  1 moves to `src/m1/` and freezes. **Bugs in a frozen model are part of the format**: mm2 duplicating mm stays in model
  1; the fix is model 2.
- **Change control** (§5). `MODELS.md` + test `registry_matches_format` (every id in `src/format.rs` has a row,
  fixtures, a golden line); `rfcs/0000-template.md`, `GOVERNANCE.md`, `CHANGELOG.md`, append-only `LEDGER.md`; CI
  `rfc-check` and `fixtures-immutable` (fails on `git diff --diff-filter=MD -- tests/fixtures/`); `.github/CODEOWNERS`
  with two reviewers on `src/{math,model,coder,format}.rs`, `FORMAT.md`, `tests/fixtures/`, `.github/`.
- **Supply chain.** Actions pinned by SHA, `cargo-fuzz` by version; a `deny` job (zero runtime dependencies);
  `blmz/LICENSE` + `NOTICE` (msun, `libm`); provenance test `fixtures_reproduce` (re-encoding `fixture_contents()`
  reproduces the committed bytes) + `tests/fixtures/SHA256SUMS` — the xz CVE-2024-3094 lesson [S].
- **Platforms.** Tier-1 `aarch64-linux`, `cross-be` (s390x, qemu), `wasm32-wasip1` — a diverging target is refused at
  compile time, like x87; `cross-decode` (each tier-1 job decodes the others' streams); `ulp-mutation` (a one-ULP change
  in `src/math.rs` must fail everywhere); `canary` (Rust beta/nightly, for float codegen changes).
- **Fuzzing.** Throughput first (reuse a zeroed model, 2^10 tables); `fuzz-nightly` ≥ 1 h per target with a cached
  corpus, timeouts are findings; a `timeouts` target bounded by `max_output`.
- **Decoupling and presets.** `parity.sh` points at a frozen snapshot (§5.5). Class C retune: selector usefulness table
  from 2^selubits entries (16 MiB at L6, 128 MiB at L9 [D]) to 2^16 (0.5 MiB); 0.222197 vs 0.222196 bits/bit at 1 MB
  [M]. Optional THP (D8).

**Exit gates.** **G1.1** Fixtures unchanged on every target and in `cross-decode`; goldens change only by the retune.
**G1.2** Fuzz throughput ≥ 10× today, 14 consecutive clean nightly runs. **G1.3** Test PRs prove a fixture edit, an
unpinned action and a one-ULP math change are each rejected. **G1.4** SIGINT leaves no partial file on Linux, macOS,
Windows. **G1.5** The retune costs ≤ +0.01 % on full Silesia [A]. **G1.6** `cargo deny check` green.

**Risks.** wasm or big-endian may truly diverge (x87 diverged at bit 1 [M]); THP is Linux-only.

### M2: Container v2, with frames, stored frames and multiple cores (4–6 eng-weeks [A])
**Goal.** The first user-visible speed win with no predictor change (model 1 runs inside frames): wall clock that scales
with cores, memcpy speed on incompressible data, corruption confined to one frame.

**Work.**
- **`FORMAT.md` v2** (v1 stays decodable): header → frames → end marker → optional backward-readable index (offsets,
  sizes, CRCs; cf. zstd's seek table, xz's index [S]) → trailer. Reserved bits must be zero and are checked (xz); a
  check-type field is reserved (2.0: CRC-32). Frame header: type (model / stored / skippable), model id, a
  **length-prefixed model-parameter block** validated by the model, filter id (0 = none; reserved so M4's E8/E9 needs no
  container change), raw and coded lengths, CRC-32 of the raw frame. Every model frame starts from a fresh model.
- **Encoder and decoder.** Default frame size from G0.4, `--solid` for one frame; per-frame stored-vs-model heuristic; a
  `std::thread::scope` pool behind `-T`, `--memlimit` capping threads × frame memory (L6: 485 MiB/thread [repo]);
  byte-identical output for any thread count. Parallel decode; `-t` names the failing frame; `-d --salvage` writes every
  verified frame; `-l` lists frames; mixed v1/v2 concatenations decode in order.
- **Tests.** ≥ 7 `v2-m1-*` fixtures (empty, single, multi, stored-only, mixed, indexed, v1+v2 concatenated); an
  exhaustive per-frame bit-flip test; fuzz targets `container_v2` (structure-aware) and `differential` (`-T1` vs `-Tn`;
  v1 solid vs v2 single frame).

**Exit gates.** **G2.1** At the default frame size S and enwik8 grow ≤ 1.0 % per file vs one model-1 stream [A].
**G2.2** Wall-clock speedup ≥ 0.8 × N at N = 4 and 8 on a Silesia tar, peak RSS reported [A]. **G2.3** Random and
model-adversarial inputs expand by no more than the container overhead (today +0.13 % / +0.6 % [M]) at ≥ 100 MB/s [A].
**G2.4** v1 fixtures decode; v2 fixtures pinned everywhere and in `cross-decode`. **G2.5** `-T1` and `-T8` produce
identical bytes. **G2.6** A flip in frame k loses only frame k under `--salvage` and is detected within one frame
(~5 min for 16 MiB [D]).

**Risks.** Frames cut long-range context (cost only inferred); memory scales with threads; small inputs gain nothing.
zpaq and kanzi multithread too: **M2 alone cannot pass the Pareto gate**.

### M3: Model 2, the fast core (10–16 eng-weeks [A]; highest value, highest risk)
**Goal.** Single-core throughput ≥ zpaq -m5 -t1 on the reference host within a pre-registered ratio budget.
`rfcs/0002-model-2.md` is committed alone before any arm is measured; development uses an experimental id (§5.1).

| lever | evidence | speed hope [A] | ratio risk |
|---|---|---|---|
| L1: cache-line bucketed hashing — one 64-byte bucket per (context, nibble) holding the nibble's counters + a check byte, for orders, sparse, word, high orders; prefetch at nibble boundaries | lookups 66 % of cycles, 20.8 LL misses/bit (research engine) [M]; next-bit prefetch alone −17 % [M]; 36 lines/byte instead of up to 144 (18 tables × 8 bits) [D] | 2–3× | low–mid: collision/replacement behaviour changes |
| L2: SIMD mixers (f32 × 8 with a normative reduction order, or int16/int32 fixed point); a cheaper normalised update than 109 sqrt + 109 div | 9.84 MiB f64 mixer state [D]; libm 27 % of instructions in the research engine before the stretch table [M]; blmz unprofiled (G0.3) | 1.5–2× | 12-bit integer stretch/squash +0.08 % text, +0.63 % binary [M]; 16-bit coder quantisation +0.000011 bits/bit [M]; f32 unmeasured |
| L3: table-driven stretch, squash, APM interpolation, specified to the bit | count-pair stretch table gave −24…−33 % [M] | 1.2–1.5× | measure 12- vs 16-bit |
| L4: fewer models | mm2 duplicates mm (291,553 of 291,554 active bytes, same pointer) [M]; STRIPH, SEL0+PVEC0+STRIPH (§2.4) | 1.2–1.5× | +0.2…0.5 % unless M4 routes per type |
| L5: measured ratio fixes | match model (hash true min length, verify candidate, cut at first mismatching bit): xml −5.38 %, dickens −0.32 %, samba −0.17 %, sao **+0.10 %** [M]; order-7 key overflow; weight clamp ±16 (no effect, safety); monotone squash (no cliff at \|t\| = 30) | ratio only | sao must be offset |

Speed hopes are guesses [A]; they multiply only if the bottleneck does not move, and G0.3 re-ranks them.

**Determinism, normative in `FORMAT.md`.**
- *f32:* zero-pad to a multiple of 8; lane j sums k ≡ j (mod 8) in increasing k; reduce as
  `((l0+l4)+(l2+l6))+((l1+l5)+(l3+l7))`; only IEEE add/mul/div/sqrt (no FMA, rsqrt/rcp or FTZ/DAZ dependence); SSE2,
  AVX2 (runtime dispatch) and NEON match a scalar reference bit for bit. *Integer:* no floating point at all — an easier
  second implementation and WASM.
- *CI:* `simd-diff`; fuzz target `fuzz_targets/scalar_vs_simd.rs`; model-2 goldens on every tier-1 target and in
  `cross-decode`; **`lever-off-parity`** — a model-2 dev build with every lever off reproduces model 1's per-bit hash
  ("flag=0 recovers bit-identically" as a gate; once a lever changes layout for good, its own fixtures take over).

**Exit gates (pre-registered).** **G3.1 speed:** single-thread compression *and* decompression ≥ 1.0× zpaq -m5 -t1 on S
and T (today ~0.25–0.43× on slices, ~0.28× on full dickens [D]); stretch ≥ 0.4× kanzi -9 -j1, i.e. 1 MB/s/core on
lzbench's EPYC [S, D]. **G3.2 ratio budget:** the core (L1–L4) ≤ +1.0 % vs model 1 at equal level on T [A], target
≤ +0.5 % (D5); with L5, no Silesia file larger than model 1 except losses listed in the RFC. **G3.3 determinism:**
identical streams across scalar/SSE2/AVX2/NEON, x86-64/aarch64/i686, Linux/macOS/Windows; 24 h of clean differential
fuzzing. **G3.4** red-team closed (§5.4).

**Kill or reposition** (pre-registered): if the best in-budget arm is < 0.6× zpaq -m5 -t1 when the 16 eng-weeks run out
[A], stop; default-tier claims go and blmz becomes a max-ratio preview (D3).

**Risks.** Ratio loss beyond budget (the 12-bit integer path alone costs +0.63 % on binary [M]); SIMD nondeterminism
repeating the research engine's rare, total, silent decode failures (est. 4.6 mismatches/MB [M]); misses still
dominating after bucketing, so L4 must remove tables, not reorganise them.

### M4: Per-type routing — each type runs only the models that pay (6–10 eng-weeks [A])
**Goal.** Spend CPU only where it pays and close the measured binary gaps, inside model 2 before it freezes; each item
is an RFC gated at 1 MB and full-file scale.

**Work.**
- **Classifier and model sets.** The encoder writes the class (text, x86, RISC, records, image, incompressible → stored)
  into the frame's model-parameter block. Text keeps word, selector and prefix vector; others drop them (SEL0/PVEC0 cost
  +0.17/+0.05 % on ooffice vs +0.39/+0.12 % on dickens [M]); D6 decides each component by measurement, not provenance.
- **E8/E9 x86 call filter** (filter id): ooffice −13.9 % [M] (xz+BCJ is only 1.3 % worse than the engine without it
  [M]); auto-detected per frame, no-regress bar ≤ +0.05 % on any file [A].
- **Contexts.** Record context with length detection: sao −12.3 % with the length hard-coded to 28 [M]; the RFC
  pre-registers how much detection may lose (keep ≥ 2/3 [A]). Line/column: nci −13.5 %, dickens −2.4 %, xml −0.02 % [M].
  4-byte alignment for RISC code (mozilla, if Alpha): −1.7 % [M].
- **Larger text tables paid for by model 2's speed** (obits 24 is −2.7 % vs blmz -6 on full dickens [M, ideal]). Pixel
  predictors (W, N, NW, gradient) for x-ray and mr last; stride contexts alone gave −0.7/−2.2 % [M].

**Projection** [D, Appendix A] — L5 plus the slice deltas above on the full-file L6 sizes, assuming additivity and that
the −1.7 % was measured on mozilla:
- Full Silesia **40.63 MB** (40.82 if sao detection keeps 2/3). With M3's +1.0 % core budget *and* G2.1's +1.0 % frame
  cost spent: **41.3–41.75 MB**, only 0.1–1.3 % under kanzi -9's 41.81 MB [S] — **G5.3 is at risk**. Even the best case
  is 3.9 % above zpaq -m5's 39.11 MB: **the landscape audit's "≤ 39 MB default tier" is unreachable with known levers**
  (the rest of paq8px's lead is embedded deflate and images).
- **On dickens, known levers leave blmz +2.1 % vs zpaq (dominated unless faster) and level with kanzi, far short of
  G5.2's 2 % margin:** 2,138,813 B, or 2,160,201 B (+3.1 % / +0.9 %) with the core budget spent. G5.1 needs at least one of: G0.2 showing the slice lead holds on T's total (xml,
  reymont, webster, enwik8); larger text tables (obits 24 + levers ≈ 2,080,856 B ideal code length, −0.7 % vs zpaq's file before blmz's 0.1–1.1 %
  coding overhead, so roughly −0.7…+0.4 % [D]; speed and memory cost unmeasured); new text levers via RFC.

**Exit gates.** **G4.1** Each lever meets its RFC gate on full files. **G4.2** Non-text frames ≥ 1.3× faster than text
frames [A]. **G4.3** No file worse than under model 1 without an RFC reason. **G4.4** Misclassification cost measured
and bounded by falling back to the text set. **G4.5** The projection re-run on measured numbers before model 2 freezes.

**Risks.** Classifier false positives; record-length detection on real data; scope creep toward paq8px.

### M5: 1.0, with freeze, packaging and launch (5–7 eng-weeks [A])
**Goal.** Make the frozen container v2 + model 2 adoptable by one beachhead user.

**Work.**
- **Freeze.** `FORMAT.md` fully normative (reduction order, table generation, model 2 and filters as prose +
  pseudo-code; model 1's section completed from `src/m1/`). `blmz-ref`, a slow decoder written only from `FORMAT.md`,
  decodes every fixture (CI `ref-decoder`). 30-day soak, no Class A/B change [A].
- **Packaging and release.** crates.io (name free per the landscape audit [S]), C ABI (cargo-c), Python (PyO3/maturin
  abi3), WASM decoder, `file(1)` magic, man page; two independent builders produce identical binaries; signed
  artefacts + SBOM; `blmz-release.yml` with two approvers; `SECURITY.md`; OSS-Fuzz once public (D9).
- **Launch.** One real pipeline on `Encoder`/`Decoder` (D1); a "when not to use blmz" page; an LTCB entry; README
  headlines with coded, round-tripped numbers only.

**Exit gates — the Pareto gate.** **G5.1 non-dominance:** on T's total no pinned competitor (Gleipnir -9 and ppmd8
included) is both smaller and at least as fast as blmz's default level — reference host, full files, one thread, one
session (§6); per-file dominations are published. **G5.2 margin:** ≥ 2 % smaller on T than the best competitor at least
as fast, with H confirming the sign [A, D5]. **G5.3 general data:** full Silesia ≤ kanzi -9 in the same session; no
input expands beyond container overhead. **G5.4** Every fixture (v1-m1, v2-m1, v2-m2) decodes on every target, in
`cross-decode` and in `blmz-ref`. **G5.5** ≥ 4 weeks of nightly fuzzing with no unfixed crash; release red-team closed.

**If G5.1 fails, there is no 1.0** (D12): releases stay 0.x previews ("max ratio, not production"), every written stream
still decodes, and work returns to M3/M4 for one dated cycle.

**After 1.0** (options): a **max tier** (Silesia ≤ 35 MB at ≥ 0.3 MB/s/core in ≤ 2 GB [I], beating Gleipnir -9) needing
deflate recompression with a vendored, pinned deflate inside the filter spec (D11) and image models; a primed small
model for B3; a genomics id from `dna.rs`; bigger vote tables (a claimed −0.001 bits/bit on enwik8 at 1.6 GiB from ledger §86,
withdrawn in §91B as not reproduced; re-measure in M0 — likely too little for an id alone).

## 5. Model-change and format governance

### 5.1 Two clocks, four knobs
**Two clocks.** The **container** changes rarely and each version freezes when first shipped; the **predictors** are an
append-only registry, `MODELS.md` (id, status, RFC, fixtures, golden hash, release). v2 dispatches per frame, so a new
id needs no container change. **Four knobs:** container *version* (layout); *model id* (the predictor's arithmetic, math
included); *filter id* (reversible transforms); *level, params and encoder heuristics* (table sizes, frame size,
stored-vs-model, threads), which never need a decoder change.

**Id ranges.** `0x01–0x3F` released, decoded forever. `0x40–0x7F` experimental: written only under
`--cfg blmz_experimental`, refused by release decoders, deletable, never in "forever" fixtures. `0x80 | id`:
platform-libm parity variant of `id`, refused by normal readers (`0x81` exists). **Lifecycle:** RFC → experimental →
candidate (bits frozen, fixtures, spec, red-team closed) → released (encoder default) → decode-only; nothing is removed;
model 1 becomes decode-only once model 2 ships (D10). At most one released id per minor release — before 1.0, exactly
one more.

### 5.2 Change classes

| class | covers | requirement |
|---|---|---|
| 0: no output change | speed (stretch table, prefetch, ICM fold), hardening, API | parity hash, goldens, fixtures unchanged on every target |
| C: encoder-only | presets (SELUBITS), frame size, stored-vs-model, threads | one decoder reads old and new streams; goldens re-pinned in the same PR on all tier-1 targets; fixtures untouched; benchmark row + CHANGELOG |
| B: predictor or filter | anything changing one probability bit, or a transform | **always a new id, however small** (the order-7 fix moves ratio ≤ 0.006 % [M] yet breaks every stream); full §5.4 |
| A: container | layout, frame types, checksum, index | new container version, owner sign-off, spec, fixtures, fuzz targets; after 1.0 at most one per major release |

### 5.3 Research practice becomes production rule

| research practice (evidence) | production rule | enforced by |
|---|---|---|
| pre-registration committed alone before any arm (`prereg/95_tag.md`); "a branch for every outcome" (HANDOVER §4, law 6) | `rfcs/NNNN-*.md` committed alone: lever, where it enters, speed and ratio bars at **both** scales, budget, reverse-course rule, a branch per outcome | `rfc-check`: a PR touching `src/{model,math,coder,format}.rs` or `FORMAT.md` cites an RFC committed before its benchmark commit |
| "BLxxx=0 recovers bit-identically" held for all 14 rebuilt versions [M] | released ids never change a bit | `lever-off-parity`, `golden_streams`, `fixtures-immutable`, `registry_matches_format` |
| red-teams: the 31-agent review fixed the window-distance self-compare (+44–51 %) [M]; §95's red-team overturned 3 of 4 drafted sentences (HANDOVER.md) | red-team before every freeze and release (determinism, resources, format, ratio claim); findings CONFIRMED/PLAUSIBLE, fixed or accepted in writing | §5.4 checklist |
| the ledger withdraws rather than edits (0.194567, §91) | append-only `LEDGER.md`; every gate result with commit, host, corpus hashes | review |
| the §80 rule lived only in a commit message; 1 of 4 default changes was round-tripped; output changed 4 times in 5 days [M] | no output change merges without §5.4 | CODEOWNERS (two people) |

### 5.4 Definition of done for a Class A or B change
(1) RFC committed alone, beforehand. (2) `lever-off-parity` green; older ids' goldens and fixtures unchanged on every
target and in `cross-decode`. (3) A §6 full-size benchmark at both scales, reporting T, H (never tuned on) and enwik8,
every size decode-verified and in the ledger. (4) ≥ 7 fixtures (empty, text, records, streamed, wrap, multi-frame,
stored mix) from a committed test, with `fixtures_reproduce`, `SHA256SUMS` and the `MODELS.md` row in the same reviewed
commit. (5) Normative `FORMAT.md` section. (6) ≥ 24 CPU-hours of fuzzing on the new id, no crash, no timeout over budget
[A]. (7) Red-team findings closed. (8) Two approvals, one the owner's.

### 5.5 Compatibility promise and research intake
**Promise (D5):** every released (version, id) decodes forever; within a release, encoder bytes are identical across
platforms and thread counts. **Intake:** freeze the snapshot model 1 was ported from as `tools/strong_m1_ref.rs`, point
`parity.sh` at it, drop the `strong.rs` trigger in `blmz.yml`; research enters only as RFCs against the development
model, measured by §6; cross-entropy is never a product number.

## 6. Benchmark and reporting protocol
1. **Pinned sets** (`bench/corpora.toml`, sha256): **Q** 12 × 1 MB slices (screening); **S** full Silesia; **T** text
   (dickens, reymont, webster, xml, nci, enwik8 + D1's beachhead files); **E** enwik8/enwik9; **H** held-out text, never
   tuned on; **A** adversarial. Reports note that Silesia is both tuning and reporting data.
2. **Bytes, not bits/bit; gate on full files.** Sizes include the container and are `cmp`-checked in the same run;
   cross-entropy (like the research headlines 0.209 stale, 0.194567 withdrawn, 0.195525 current) is labelled "ideal" and
   never compared with other codecs' files. Slices only screen: STRIPH flips sign between 1 MB and 10 MB [M], and every
   text win over zpaq (xml, reymont, webster) exists only on slices; the one full text file measured, dickens, is not among
   them and loses [M], so no slice-to-full reversal has been observed yet: G0.2 decides.
3. **A dedicated, uncontended reference host** (CPU, cores, L2/L3, RAM, kernel, THP mode, governor recorded; `taskset`).
   Every speed in §2 is screening-grade.
4. **Speed relative to co-measured competitors.** User/sys/wall and peak RSS per direction (median of 5, 3 above 10 MB,
   with spread); gates are multiples of **zpaq -m5 -t1 and kanzi -9 -j1 from the same session**. Absolute speeds don't
   transfer: kanzi -9 runs 522 KB/s on slices and ~1.1 MB/s on full dickens here, 2.48 MB/s on the EPYC [M, S] — a
   2.2–4.8× host factor depending on input size [D].
5. **Gate single-threaded;** multithreaded runs compared at equal thread counts, with memory. **Show the whole
   frontier:** zpaq, kanzi, Gleipnir and ppmd8 beside xz/bzip2/gzip, built from source at pinned versions; [S] figures
   in their own column.
6. **Generate, don't type.** Results in `bench/results/<date>-<host>-<commit>.json`; README and roadmap tables are
   generated from them and re-run after any output-changing commit. **CI:** `bench-smoke` per PR (exact coded sizes of
   in-repo inputs at L1 and L6; any diff needs a C or B label); `bench-full` weekly on the self-hosted reference runner
   (S, T, competitors, JSON artefacts); shared runners never gate speed.

## 7. Risk register

| id | risk | L / I | mitigation | trigger |
|---|---|---|---|---|
| R1 | speed gate unreachable within the ratio budget | M / H | profile first (G0.3); f32 and integer arms; levers ranked by profile | M3 kill criterion |
| R2 | the text lead is a small-input artefact | **H** / H | G0.2 before M3; full dickens loses to zpaq even at obits 24 (ideal +2.1 %) [D] | D2 |
| R3 | known levers may not pass G5 (dickens +2.1–3.1 % vs zpaq; Silesia 41.3–41.75 MB clears kanzi's 41.81 MB by only 0.1–1.3 %) [D] | H / H | larger text tables; core budget target ≤ +0.5 %; RFC text levers; "no 1.0" branch | G4.5 |
| R4 | nondeterminism (SIMD, float, compiler) or a change without a new id → silent, total decode failure | M / critical | normative reduction; scalar reference; `simd-diff`, `cross-decode`, `ulp-mutation`, `canary`; `golden_streams`, `fixtures-immutable`, `rfc-check`, CODEOWNERS | any golden mismatch blocks merge |
| R5 | supply chain via fixtures or release tooling (CVE-2024-3094 [S]) | L / critical | provenance test; SHA-pinned Actions; two-person review; reproducible signed builds; zero deps | unreviewed `.github/` or fixture change |
| R6 | model-id sprawl; forever-decoder burden | M / M | batching; experimental range; decode-only retirement | 2nd id proposed before 1.0 |
| R7 | resources: memory × threads (L6 485 MiB/thread [repo]); symmetric decode cost; decompression bombs | H / M | `--memlimit` bounds threads; `max_output` 1 GiB default (done); costs documented | OOM reports |
| R8 | economics never close on cold tiers (Deep Archive 5–9 years even at 1 MB/s [I]) | H / M | target pricier tiers (B1) | — |
| R9 | fuzzing gives false confidence (9–22 exec/s [M]) | H / M | throughput work; nightly; structure-aware and differential targets; OSS-Fuzz | exec/s below target |
| R10 | research churn breaks production; irreproducible claims; overfitting to Silesia | H / M | frozen snapshot; RFC-only intake; pinned corpora; held-out H | — |
| R11 | frames cost more ratio than inferred | M / M | measured curve (G0.4); `--solid`; larger frames at high levels | G2.1 fails |
| R12 | owner attention split with the intelligence track; bus factor on a forever promise | M / H | staffing decision (D3); `blmz-ref` and a complete `FORMAT.md` (M5) | — |
| R13 | competitors move, or the frontier is incomplete | M / M | relative gates; Gleipnir and ppmd8 pinned; quarterly re-runs | — |

## 8. Decisions for the owner

| # | decision | options | recommendation | by |
|---|---|---|---|---|
| D1 | product form and first proof point | library+CLI embedded in tools / hosted service / both | **both, library first**; B0 general cold storage is the product, B1 text archives the first proof point; decide hosted vs self-hosted pricing before M3, since it sets whether speed or idle-CPU economics gate 0.x adoption | M0 end |
| D2 | go/no-go on the text premise | proceed / text levers before M3 / re-choose beachhead | **pre-register the M0 criterion now**; proceed only if G0.2 meets it | M0 end |
| D3 | priority vs the intelligence track | compressor first (2 engineers; no research adoptions until model 2 freezes) / shared / research first (blmz stays 0.1) | **compressor first if D2 says go**, with M3's kill criterion | M0 end |
| D4 | model-2 numerics | f32 SIMD with normative reduction / integer fixed point | **pre-register both; integer wins a tie** (no float spec, easier second implementation, WASM) | M3 RFC |
| D5 | gates and promise | absolute KB/s / relative to co-measured codecs | **relative**; 1.0 margin ≥ 2 %; core budget ≤ +1.0 % ceiling, ≤ +0.5 % target; adopt the §5.5 promise | M0 |
| D6 | exchange rate for keeping a component | X % smaller per 1 % CPU | **≥ 0.05 %** [A]: hashed orders 8..32 buy ≈ 0.009 % (+0.18 % at 10 MB for −20 % time measured at 1 MB) and would go unless M0's full-size timing differs | M3 RFC |
| D7 | container v2 with model 1 as a 0.2 preview? | yes / wait for model 2 | **yes, once G1/G2 pass**: multi-core win early, frames exercised; note this freezes container v2 | M2 end |
| D8 | dependencies | std-only / audited crates | **zero runtime deps in core**; wrappers may; THP only as an opt-in `libc` feature if ≥ 10 % at default sizes | M1 |
| D9 | name, extension, visibility | `.blz` / `.blmz`; public now / at 0.2 / at 1.0 | **`.blmz` for v2** (still read `.blz`), reserve the crates.io name; **public at 0.2** — outside fuzzing pays most at the freeze, and the README now shows the zpaq/kanzi comparison | before 0.2 |
| D10 | model 1 after model 2 ships | default for some levels / decode-only | **decode-only** | M5 |
| D11 | deflate recompression (max tier) | in 1.x / post-1.0 / never | **post-1.0**, sized by M0's discovery | after 1.0 |
| D12 | G5.1 fails at M5 | stay 0.x / ship 1.0 without default-tier claims | **stay 0.x** (streams already decode forever, so users lose nothing); one more dated M3/M4 cycle, then D3's reposition | M5 |

## Appendix A: derivations
- **Ratios.** Slices: zpaq −3.5 % vs blmz's 3,432,826 B, i.e. blmz +3.6 % vs zpaq (engine ideal +3.3 %); speed
  200/86–200/49 = 2.3–4.1×, 522/86–522/49 = 6.1–10.7×. Dickens: zpaq −4.7 % (blmz +4.9 %), kanzi −2.6 % (blmz +2.7 %),
  obits 24 still +2.1 % vs zpaq; zpaq ≈ 184 KB/s and kanzi ≈ 1.1 MB/s, so blmz's 51.9 KB/s is 0.28× zpaq.
- **Time and cycles.** 10^9 / (53.9 × 1024) s ≈ 5.0 h/GB (5.15 h if KB = 1000 B); Silesia 64–66 min; 16 MiB frame
  ~5 min; mozilla at 61.8 KB/s ~14 min. 2.1×10^9 / (51.9 × 1024 × 8) ≈ 4,940 cycles/bit; 200 KB/s ≈ 1,280–1,310; 1 MB/s
  ≈ 262.
- **Host factor.** 2.48/0.522 = 4.75; 2.48/1.145 = 2.17; Gleipnir here 320/4.75–320/2.17 ≈ 67–148 KB/s, assuming the
  same factor.
- **Model 1** (`src/model.rs`). Tables 8 + 5 + 3 + 2 = 18; 109 = 3 × NW (35) in `rms_update` + 4 in the final mixer;
  mixers (2048 + 16384 + 1) × 35 × 16 B = 9.84 MiB; selut 2^21 × 8 B = 16 MiB (L6), 2^24 × 8 B = 128 MiB (L9), 0.5 MiB
  at 2^16.
- **Projection** (each file's L6 size × its slice deltas). L5 bundle −32.6 KB (xml −17,492; dickens −6,925; samba
  −5,558; mozilla −3,167; ooffice −2,935; sao +3,462). M4: sao −553,387; ooffice −299,549; mozilla alignment −224,245;
  nci −169,164; dickens −52,594; mr −48,355; x-ray −25,700. Totals 40,632,253 B; 40,816,715 if sao keeps 2/3; 41,274,656
  with +1 % core everywhere and +1 % frame cost on files > 16 MiB; 41,753,258 with frame cost everywhere and no
  alignment/image levers. Dickens 2,138,813; 2,160,201 with the core budget; 2,080,856 at obits 24.
- **Headroom.** Σ full × (1 − 1/(1 + g)) over the paq8px slice lead g = 13.88 MB (→ 28.16 MB); mozilla + samba 9.68 MB =
  70 %. **Exchange rate.** STRIPH 0.18 % / 20 % ≈ 0.009 % per 1 % CPU; SEL0+PVEC0+STRIPH (dickens) 0.47 % / 33.6 % ≈
  0.014 %.
