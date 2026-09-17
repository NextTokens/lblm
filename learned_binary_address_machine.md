# Bit-Native Predictive Machine — Learned Binary Addresses (v2 design)

> Companion to `bit_native_predictive_machine.md` (v1). v1 specified the register, the
> calibration loop, and the two data streams, but left §4.3 — the "adjustable signal
> circuit" — described only in metaphor. **This document is the concrete circuit**, plus
> the upgrade we agreed on: the responder templates become *mobile learned binary
> addresses*. Reference implementation: `blm.py`.

---

## 0. The bet, restated

The machine predicts the **next bit** from a register of recent bits, autoregressively —
vocabulary of size 2, not ~50k tokens like an LLM. Each step is a single Bernoulli
decision (cheap), at the cost of ~`log2(vocab)` ≈ **16× more steps** per equivalent token
of content. The bet pays off *only if the per-step circuit is tiny*, which it is here.

Going binary doesn't remove difficulty; it **relocates** it — from *vocabulary size* to
**dependency range measured in steps**. With 1 bit/step, structure an LLM grabs in a few
steps is smeared across ~16× more of ours. So the real enemy is long-range reach in a
single left-to-right bit chain. Three escape ideas were considered:

- **Blockchain-style chaining** where the "hash" is a *learned* summary (not cryptographic —
  a crypto hash's avalanche/pre-image-resistance is exactly wrong for a model).
- **Multi-directional relations** (cellular-automaton field, any-order/masked generation).
- **A matrix with coordinates**.

They are not three gambles — they are **one substrate**: give every unit an explicit
**address** (itself a bit-vector) and let *relation = a function of addresses*. The three
ideas are just settings of that function (local→CA, dense→attention, stored-pattern→Hopfield),
and the blockchain is just a coarse coordinate axis. We chose the most powerful knob:
**learned binary addresses**.

---

## 1. Core objects

```
unit  = ( address a ∈ {0,1}^A ,  value v ∈ {0,1} ,  strength w ∈ ℝ≥0 )
query q = the current R-bit register
```

Minimal version (what `blm.py` builds first): **H = identity**, so `A = R` and the query is
just the raw register. The leap from v1 is that a unit's address is allowed to **move** in
the Hamming cube instead of being pinned to a raw template.

---

## 2. Predict one bit

```
1. q = current R-bit register.
2. for each unit m:  sim_m = R − popcount(q XOR a_m)          # Hamming closeness
                     k_m   = exp(beta · (sim_m − R))          # 1 at exact match, decays
3. pressure  P = Σ_m  w_m · (2·v_m − 1) · k_m                 # signed strength-weighted vote
4. next bit = 1 if P > 0 else 0    (exact tie → configured default)
```

Address compare is `XOR + popcount` — the cheapest operation there is. Top-k retrieval over
learned binary codes is literally **learned LSH** (locality-sensitive hashing). This is the
fastest of all variants we discussed; the speed thesis survives intact.

---

## 3. Learn one step (online, gradient-free — a binary SOM)

For each observed `(register q, true next bit y)`:

```
move_prob = base_move_prob · anneal^t              # SOM cooling over time t

ŷ = predict(q)
rank units by Hamming(q, a_m)
nearest_correct = closest unit with value == y

if no nearest_correct within alloc_radius:
    ALLOCATE a new unit at address q, value y       # adaptive resolution
    nearest_correct = that new unit

reinforce: nearest_correct.w += lr_w
           (mobile) PULL a_m one+ Hamming steps toward q   # tighten the right region

if ŷ != y:                                          # a miss
    for the nearest wrong-voters within push_radius:
        weaken: w -= lr_w
        (mobile) PUSH a_m one+ steps away from q    # contrastive separation

periodically: merge units with identical (address, value); track address utilisation
```

This is a Self-Organizing Map's competitive update **applied to move the addresses
themselves** (Kohonen's SOM moves fixed-grid weights; we move the coordinates). It is the
gradient-free route that fits v1's "adjust strengths" calibration. Gradient alternatives:
straight-through binarization (semantic hashing) and a VQ codebook. **All three are
workarounds for the same discrete-bottleneck — none is free.**

**Two things fall out for free:** (a) genuine context collisions become the *push-apart*
signal — the conflict problem trains the addresses instead of breaking the machine; (b) if
the address is recurrent, `H(prev_addr, new_bits) → addr`, the blockchain summary chain
reappears as the address trajectory.

**Modes:** `frozen` = addresses never move (a growing template/responder table with strength
learning = the original v1 circuit). `mobile` = addresses self-organise. Comparing them is
the experiment.

---

## 4. Data & task (real — no invented data)

The only data is the two streams from v1 §5:

```
STREAM_A = 0101011111000     (question 01010 / boundary 111 / answer 11 / stop 000)
STREAM_B = 1010011100000     (question 10100 / boundary 111 / answer 00 / stop 000)
```

Training pairs = every R-bit window → its next bit, over both streams. **Task:** from the
first R bits of each stream, autoregressively reproduce its tail. **Honest baseline to
beat:** always-predict-0 (the zero-bias trap — ~80% bit accuracy at R=8 yet it gets
stream A's answer `11` wrong).

---

## 5. Results (hardened build, clean A/B, 5 seeds each)

Sweep over register width R, mode, with the hardening from §8 applied (separate move-RNG so
frozen/mobile share an identical shuffle order). `both_ok` = both stream tails reproduced
exactly; `LOWO` = leave-one-window-out generalisation (hold out a pair, train on the rest,
predict it); robustness = of 16 single-bit seed flips, how many give the right 2-bit head.

| R | mode | train | both_ok | LOWO | robust/16 | units | collisions |
|---|------|------:|--------:|-----:|----------:|------:|-----------:|
| 8 | frozen | 1.00 | 1.00 | 0.60 | 13.0 | 10.0 | 0 |
| 8 | mobile | 1.00 | 1.00 | 0.60 | 13.0 | 10.0 | 0 |
| 6 | frozen | 1.00 | 1.00 | 0.79 | 8.6 | 9.0 | 0 |
| 6 | mobile | 1.00 | 1.00 | 0.77 | **9.0** | 11.8 | 0 |
| 5 | frozen | 0.975 | 0.80 | 0.64 | 4.2 | 8.8 | 0 |
| 5 | mobile | **1.00** | **1.00** | 0.66 | 4.4 | 14.0 | 0 |
| 4 | frozen | 0.756 | 0.00 | 0.46 | 1.0 | 8.4 | 4 colliding /12 |
| 4 | mobile | 0.689 | 0.00 | 0.43 | 1.4 | 12.2 | 4 colliding /12 |

**Recurrent address** (`addr = window ++ history`, frozen mode):

| addr | R | h | A | both_ok | LOWO | note |
|------|---|---|---|--------:|-----:|------|
| register | 5 | – | 5 | 0.80 | 0.64 | memoryless baseline |
| shift    | 5 | 2 | 7 | **1.00** | 0.65 | history lifts R=5 to perfect |
| register | 8 | – | 8 | 1.00 | 0.60 | |
| shift    | 8 | 2 | 10 | 1.00 | **0.70** | history improves generalisation |
| register | 4 | – | 4 | 0.00 | 0.46 | 6 colliding rows |
| shift    | 4 | 2 | 6 | 0.00 | 0.58 | collisions 6→1, but cold-state seed (see §6.4) |
| fold     | 4 | 3 | 7 | 0.00 | 0.41 | hand-coded compression is *worse* than shift |

---

## 6. What we learned (every cycle teaches something)

1. **It does the task at R ≥ 6, but by MEMORISATION, not learning.** train_acc 1.0 and
   both_ok 1.0, but the units sit *verbatim* on the training registers (10/10 at R=8) — it
   is a Hamming-kNN over an injective lookup table. **LOWO generalisation is only
   0.43–0.79**, i.e. it does not learn the stream grammar. Honest framing: "memorises an
   injective table with Hamming smoothing."

2. **Part of the R=8 "win" is zero-bias.** Stream B's tail is `00000`, so always-predict-0
   *also* reproduces B. The machine genuinely beats the baseline only on **stream A** (`11`).
   The harness now prints the baseline's own generation so this can't be hidden.

3. **Mobile gives a modest, real edge — under a clean A/B.** With the move-RNG separated so
   both modes see the same shuffle order, mobile **beats frozen at R=5** (both_ok 1.00 vs
   0.80) and edges robustness at R=6 (9.0 vs 8.6); tied at R=8. My first-pass "mobile never
   helps" was partly an *uncontrolled-comparison* artifact. (It still wastes units and never
   helps at R=4, where the task is unlearnable by any stateless predictor.)

4. **Recurrent history is the right direction, and it exposed a new requirement.** Carrying
   history (`shift`) cut R=4 collisions 6→1, lifted **R=5 both_ok 0.80→1.00**, and improved
   R=8 LOWO 0.60→0.70. But it could *not* fix R=4 generation, because there the seed is only
   4 bits **and the recurrent state starts cold (zeros)** — generation is information-starved
   at the boundary no matter how wide the address. **Lesson: a recurrent machine must have
   its state seeded/warmed, not zero-initialised.**

5. **Hand-coded compression breaks Hamming smoothness.** The rotate/xor `fold` consistently
   did *worse* than the plain `shift` (more address collisions, lower LOWO) — empirical proof
   that you cannot squeeze history into fixed width with a fixed mixing function. This is
   exactly why the address compressor must be **learned** (a smooth binary hash), the hard
   discrete-bottleneck sub-problem.

6. **Collisions are an information-theoretic wall (confirmed, strengthened).** At R=4 the 4
   contexts are exact 50/50 splits; exhaustively **0 of 4096** deterministic tables reproduce
   both tails. The R-bit register is not Markov-sufficient — no learning rule or probabilistic
   output fixes that; only *more sufficient state* does.

7. **The toy is now saturated.** Once R ≥ 6, the memoryless register already solves the task,
   so the two short streams can no longer distinguish architectures (except in the R=4
   collision corner). Testing recurrent/learned-address mechanisms for real needs data with
   **dependencies longer than the register** — which 13-bit streams cannot exhibit.

---

## 7. Next design iteration (to fine-tune)

Priority order, per the verification workflow's diagnosis (do NOT jump to learned-H or
probabilistic output first — the state must be made sufficient before either pays off):

1. **Warm-start the recurrent state** so generation isn't cold (the §6.4 discovery), then
   re-test whether recurrent state cracks the R=4 generation.
2. **Learned smooth binary hash** for the address compressor (replaces the hand-coded
   `fold`, which §6.5 showed breaks smoothness) — with anti-collapse spread pressure from
   day one.
3. **A harder data bench** with genuine long-range structure (since §6.7: the current toy is
   saturated) — the only way to measure whether recurrent/learned addresses beat a wide
   register at scale.
4. *Deferred:* probabilistic/frequency output and generic learned-H — premature until state
   is sufficient.

---

## 8. Verdict from adversarial verification (5-agent workflow, high confidence)

- **Implementation is correct.** No behaviour-altering bugs: generation is genuinely
  autoregressive (no target leak), the kernel `exp(beta·(sim−A))` decays monotonically with
  Hamming distance (near neighbours dominate, verified by hand), the `(2v−1)` sign is right,
  `merge()` loses no units, and all sweep numbers reproduce to the digit.
- **Claim "genuinely learns" → corrected to "memorises" (see §6.1–6.2).**
- **Claim "mobile never helps" → refuted (see §6.3); it helps modestly under a clean A/B.**
- **Claim "R=4 collisions are fatal to deterministic modes" → confirmed and strengthened
  (see §6.6).**
- **Hardening applied:** separate move-RNG (clean A/B), LOWO metric, baseline-generation
  reporting. Recommended single next change: **recurrent address** — now implemented and
  tested (§5, §6.4–6.5).

---

## 9. Cycle 2 — the long-range recall bench (`bench.py`)

Since §6.7 showed the 13-bit toy is saturated, we built a **controlled capability probe** with
dependencies longer than the register. Each example:

```
[TYPE bit] [shared BODY length L] [BOUNDARY 111] [ANSWER] [STOP 000]
```

The two classes share an *identical* body and differ only by the leading TYPE bit (and the
answer it dictates: type0→`11`, type1→`00`). For **L ≥ R−3** the answer-position register is
byte-identical across classes — a deliberate collision at distance L that *only memory* can
resolve. Sweeping L gives a **memory curve**. (This is a designed test like the copy/parity
tasks used to validate LSTMs — generated by a documented seeded rule, not arbitrary mock data.)

**Memory curve (R=6, answer-accuracy, chance=0.50):**

| L | register | shift+4 | fold+4 | note |
|---|---------:|--------:|-------:|------|
| 4 | 0.42 | **0.85** | 0.83 | inside shift horizon |
| 6 | 0.27 | **0.88** | 0.75 | horizon edge (R+h−4 = 6) |
| 8 | 0.33 | 0.44 | 0.42 | past horizon → all collapse to chance |

The **memory horizon is exactly `L = R + h − 4`** (the −4 = 1 TYPE bit + 3 boundary bits),
verified as a *hard step function* across a 16-cell R×h grid. This is genuine recurrent memory,
not a leak: `register` reaches L=R−4, `shift+h` reaches L=R+h−4, `fold` has no clean horizon
(its xor-mixing hurts Hamming smoothness, confirming §6.5).

---

## 10. Cycle-2 verification verdict (5-agent workflow, high confidence)

The workflow reproduced every number and **overturned two claims I had drafted** — the reason
we verify before documenting:

1. **"Dilution" → REFUTED.** I had guessed register fails at small L because uniform Hamming
   *dilutes* a present-but-outvoted TYPE bit. Wrong: amplifying that bit 24× or raising `beta`
   to 64 changes nothing. The real cause is **global label conflict** — the answer-position
   register recurs at *non-answer* positions with the opposite next-bit target, so the exact
   unit is reinforced toward the wrong value about as often as the right one — plus
   autoregressive feedback on the 2nd answer bit. (Training on answer-position pairs only lifts
   L=2 from 0.52→0.69; up-weighting the bit did not. So it is conflict-limited, not
   weight-limited.)

2. **"It only memorises" → REFUTED.** The decisive test is a **rule-scramble control**
   (randomise type→answer per body, so no transferable rule exists). On *unseen* bodies,
   `shift+4` scores **0.72–0.78 intact vs 0.44–0.53 scrambled** at L=3–6 (gap +0.22–0.31). A
   memoriser would tie both arms; the gap proves **genuine body-invariant rule transfer**. It
   is real but bounded: only inside the memory band, and capped at ~0.7–0.8 (seed-noisy)
   because uniform Hamming cannot isolate the lone type bit from body-derived state bits.

3. **Memory horizon `R+h−4` and bench soundness (L≥3) → CONFIRMED.**

**Bench bugs fixed (per the verdict):** docstring now scopes the collision claim to L≥R−3;
`gen_body` forbids a leading `[1,1]` (was creating spurious `111` in ~48% of sequences); the
rule-scramble control is built in; the small-L / L<R cells are flagged as not-clean-recall.

---

## 11. Sharpened spec for the learned address, and the next change

The evidence (keeping only what survived verification) says the learned address must:

- **Carry-far (bounded):** inject the TYPE bit into the recurrent state so it survives to the
  answer — but only up to `h` dropped steps; beyond the horizon the bit is gone and *no*
  address recovers it. Not "arbitrarily far".
- **Weight-the-bit (feature selectivity):** attend to the type-carrying slot and down-weight
  body bits. This is the lever for the **generalisation ceiling** (push held-out 0.78→~1.0) —
  *not* the small-L fix (that is label conflict; see §10.1). The DROPPED claim: "uniform
  Hamming dilution outvotes a present bit."
- **Body-invariance:** encode `answer=f(TYPE)` independently of the body — *demonstrated* by
  the rule-scramble gap, required, but only meaningful inside the memory band.

**Attempt 1 — static MI weighting (`--weights mi`):** weight each address bit by its mutual
information with the next bit. Decisive held-out test (`shift+4`, R=6, K=24, e500, 3 seeds):

| L | uniform intact | uniform scram | MI intact | MI scram |
|---|---------------:|--------------:|----------:|---------:|
| 3 | 0.81 | 0.44 | 0.75 | 0.47 |
| 4 | 0.78 | 0.56 | 0.75 | 0.44 |
| 6 | 0.58 | 0.39 | 0.64 | 0.31 |
| 8 | 0.39 | 0.42 | 0.25 | 0.42 |

**Result: MI weighting did NOT break the ceiling** (a wash, slightly worse). The
intact≫scrambled gap confirms generalisation is real in both, but selectivity didn't improve.

**Why (diagnosed, not guessed).** Printing the learned weights for `shift+4`/L=3 (address =
6 window bits ++ 4 state bits; the two classes differ *only* at position 9, the type slot):

```
weights:  0:0.38  1:1.00  2:0.15  3:0.02  4:1.46  5:2.66  6:0.49  7:0.85  8:1.38  9:1.62
```

The **boundary bits get the highest weight** (pos 5 = 2.66, pos 4 = 1.46) because `111`→answer
is globally predictive — but those bits are *identical across the two classes*, so up-weighting
them does nothing. The lone **discriminative** bit (pos 9 = 1.62) is above average yet swamped.
**Static MI conflates *predictive* with *discriminative*:** it scores each bit against the next
bit marginally, blind to which bits separate the *confusable* classes.

**Attempt 2 — contrastive weighting (`--weights contrastive`):** weight a bit by how often it
*separates collision partners* — near addresses (Hamming ≤ radius) with **opposite** labels.
This targets discriminative bits and ignores predictive-but-invariant ones.

*Diagnostic — it worked at the weight level.* For `shift+4`/L=3 the learned weights were:
```
0:1.50  1:1.57  2:1.43  3:0.86  4:0.13  5:1.24  6:0.00  7:0.20  8:1.08  9:1.99
```
The discriminative slot (pos 9) is now the **highest** weight and a boundary bit (pos 4) is
killed to 0.13 — exactly the intended correction over MI.

*But held-out accuracy did NOT improve.* Probe (held-out L=3, 5 seeds; numbers corrected after
the verification pass — see §11a):

| weighting | held-out intact | scrambled |
|---|---:|---:|
| uniform | 0.73 | 0.42 |
| pos9 dominant (others=1, pos9=5×) | ~0.78 | — |
| **isolate pos9 only (others≈0)** | **0.50 (chance)** | 0.50 |

So concentrating weight on the discriminative bit **doesn't help** (≈ uniform), and *fully
isolating* it **collapses to chance**.

**Conclusion — the metric-tuning arc is exhausted.** No fixed per-bit weighting of the raw
address (applied at retrieval) can solve this, because the task needs **different bits at
different generation steps**: the *first* answer bit needs the type slot, the *second* needs
the window (the just-emitted first bit). A single static weight vector cannot serve both. The
need is genuinely **query-conditional**.

### 11a. Verification verdict (cycle 2c, 5-agent workflow, high confidence)

- **Confirmed — no static *retrieval* weighting works.** An agent *optimised* over arbitrary
  static vectors: best held-out caps at **~0.80**, never near 0.85; at L=3 the search optimum
  even overfits *below* uniform (0.67 < 0.73). Contrastive does not beat uniform (reproduced).
- **Mechanism confirmed independently.** The discriminative position moves per step:
  L=3 step1 = {9}, step2 = {5,8}; L=4 step1 = {8}, step2 = {5,7} (pos9/8 = the type bit's depth
  in the h=4 state; pos5 = the just-emitted answer bit). Cleanest proof it is step-dependence,
  not error propagation: under isolate-pos9, first-bit acc = 0.90 but the **teacher-forced**
  second bit (handed the *true* first bit) = 0.50 = chance.
- **Correction to my earlier number.** I had reported "pos9-dominant-5× → 0.58"; that does **not**
  reproduce — with a non-suppressed base it is ~0.78 (within seed noise of uniform). The
  conclusion (capped well below 0.85) stands; the 0.58 figure was an artifact of an
  over-suppressed base vector.
- **Scope caveat.** Weights are consumed only by retrieval (`pressure`); `learn`'s
  allocation/ranking still use *unweighted* Hamming. A static vector applied to ranking *too*
  (a different machine) showed a suggestive +0.05–0.13 at L3/L4 (reaching ~0.85) — but its
  scramble control was not confirmed, so it is not endorsed as a counterexample. The claim is
  scoped to **retrieval-only** weighting.
- **Caveat on the next step.** Query-conditional weighting targets the *step-dependence*
  problem; at L ≥ 6 the discriminative signal itself flattens toward uniform, so it is unlikely
  to rescue the long-horizon (signal-decay) regime.

---

## 12. State of the project after cycle 2

**Confirmed, reusable results:**
- Recurrent memory works, with an exact horizon `L = R+h−4` (hard step, 16-cell grid).
- Genuine body-invariant rule transfer to unseen bodies (rule-scramble control: +0.22–0.31),
  bounded at ~0.78 inside the band.
- The ~0.78 ceiling is **not** a retrieval-weighting problem: static MI fails, static contrastive
  finds the right bit but doesn't help, optimised static vectors cap ~0.80, and the
  discriminative bit moves per step (so a fixed vector cannot serve all steps).

### 12a. Cycle 2d — query-conditional readout (`--weights conditional`) also fails

The attention-style readout (weights recomputed per query from the local confusion near `q`) was
the evidence-pointed next step. Tested head-to-head with uniform (held-out, K=24, 5 seeds):

| L | uniform intact | conditional intact | Δ |
|---|---:|---:|---:|
| 3 | 0.73 | 0.73 | +0.00 |
| 4 | 0.73 | 0.70 | −0.03 |
| 6 | 0.53 | 0.58 | +0.05 |

A `cond_k × radius` sweep ({8,16,32} × {2,3,4}) never beat uniform (best ≈0.73–0.75). **Result:
query-conditional readout does not break the ceiling either.**

**The sharpened conclusion.** Changing the *readout* — static **or** query-conditional — cannot
move the ceiling, because the verification confirmed weights only touch retrieval (`pressure`);
`learn`'s allocation/ranking still use **unweighted** Hamming, so the memory is *built* uniformly
regardless. On an unseen body the stored units simply do not encode the type cleanly, and no
readout reweighting recovers what was never represented. **The ceiling is representational, not a
metric.**

**The open frontier (next chapter):** shape the *learning*, not the readout —
1. **Weight allocation/ranking** (the verification's suggestive +0.05–0.13 path) so the
   discriminative structure determines *which* units are stored; verify with the scramble control.
2. **A learned recurrent encoder** that maps (window ++ history) into a body-invariant binary code
   during training (the genuine "learned hash" — the discrete-bottleneck problem).

Known limit for both: they target the in-band ceiling, not long-horizon signal decay (L ≥ R+h−4).

---

## 13. Methodology — measuring learning vs memorization at the bit level

A caution that shapes how every result here is read: **"memorization vs generalization" is partly
an *assumption* in a bit-native model, not a clean dichotomy.** With a vocabulary of 2, every
stored pattern is shared across a large Hamming neighbourhood, so *storage is itself an
interpolation mechanism* — a bit-kNN that "looks up" patterns is functionally generative when it
recombines stored bit-fragments into sequences it never saw. Flat exact-held-out accuracy
therefore should **not** be read as "memorization."

Two assumption-free instruments are used instead:

1. **Rule-scramble control.** Randomise the rule (type→answer) per body so no transferable rule
   exists. The **scramble gap** = held-out(intact) − held-out(scrambled) is the amount of genuine
   rule transfer; a pure memoriser ties both arms (gap ≈ 0). This also controls for capacity: the
   scrambled arm has the *same* unit count, so an above-chance intact arm is not "just more units."
2. **Dataset-size learning curve.** Plot held-out vs training size `K`. Memorization → flat;
   genuine rule-learning → the scramble gap **grows with `K`** (more data → better rule → better
   transfer). Corroborating clue: in these experiments more `K` helps but more *epochs* do not —
   the signature of generalisation (epochs only re-fit the same data).

## 14. Cycle 3 — allocation weighting (option 1): the capacity control flips it

Option 1 applied the discriminative weights to `learn`'s ranking/allocation (`--weight-learn`),
not just the readout. A first learning curve looked like a win (intact ~0.85 vs uniform ~0.71).
But adding a **capacity control** — plain uniform forced to allocate as many units
(`alloc_radius=0`) — reversed the conclusion (L=4, held-out, 6 seeds):

| variant | K=24 | K=48 | K=96 | units |
|---|---|---|---|---|
| uniform (r=1) | 0.71 / +.29 | 0.65 / +.17 | 0.74 / +.36 | ~44 |
| **uniform MAXCAP (r=0)** | 0.74 / +.35 | 0.81 / +.38 | **0.83 / +.45** | ~117 |
| contr/LEARN (option 1) | 0.76 / +.32 | 0.81 / +.38 | **0.84 / +.45** | ~68 |

*(cells = intact / scramble-gap.)* **Capacity-matched uniform matches option 1.** The
discriminative weighting was *not* the active ingredient — it helped only by *incidentally
allocating more units*. **The lever is capacity (unit count) + data, not the weighting.**

**What is real (confirmed by the scramble + capacity controls):**
- **It genuinely learns the rule.** The scramble gap grows with `K` (+0.29 → +0.45) and intact
  rises to **~0.84 at K=96** — and the scrambled arm stays at chance with the *same* unit count,
  so this is not memorization-by-capacity. (See §13.)
- The earlier "~0.78 ceiling" was largely a **capacity/coverage limit**, not purely
  representational: more units + more data → ~0.84.

### 14a. Verification verdict (cycle 3, 4-agent workflow, medium confidence)

- **Capacity is the lever — confirmed.** The clean control: contrastive *readout-only* allocates
  the *identical* unit counts as uniform (seed-for-seed), and there weighting **never wins**
  (e.g. K96 0.674 vs 0.667 at 41 units). The apparent `contr/LEARN` edge is **unit-efficiency**:
  `weight_learn` shifts allocation into a ~68-unit band that integer-Hamming uniform cannot
  discretely occupy (its reachable counts are ~17/36/117). At matched-or-higher capacity, plain
  uniform *allocate-every-step* reaches **0.94**, matching/beating weighting — so accuracy tracks
  the **capacity / exact-context** axis, not the discriminative metric.
- **Genuine learning — strongly confirmed.** The scrambled arm stays at chance (~0.40–0.48) while
  carrying *strictly more* units than the intact arm (K96: scramble 174u @ ~0.45 vs intact 117u @
  ~0.88). Extra capacity with no rule buys nothing → this is transferable rule recall, not
  memorization-by-capacity. The scramble gap grows monotonically with `K`.
- **Asymptote ~0.88–0.90, not 1.0 (and not a flat 0.85).** Held-out rises monotonically to
  **0.896 at K=192**. The residual gap is **jointly data-limited** (units saturate at 117 by K=48
  yet accuracy keeps climbing to K=192) **and representation-limited**: a body-length sweep (K96,
  max capacity) gives L2 = 0.76, L4 = 0.75, **L6 = 0.69** — accuracy *falls as the body grows
  despite more units*, because the lone discriminative TYPE bit is **diluted by body-noise** in the
  unweighted vote.

**Residual confound (why medium, not high):** integer Hamming makes unit count discrete, so plain
uniform could not be placed at *exactly* `contr/LEARN`'s 68-unit point for a perfect head-to-head.

**Next step (closes the confound and tests the floor):** add a **continuous capacity knob**
(prune/subsample the unit set, or a real-valued allocation threshold) to set uniform at any unit
count. Then (a) count-matched uniform vs weighting settles claim 1 fully, and (b) if weighting the
TYPE bit *does* lift the ~0.88 ceiling at matched capacity, the asymptote is a uniform-Hamming
representational floor that a **learned encoder** (compress the body out) would remove — which the
body-length sweep already points to.

### 14b. Cycle 3b — matched-capacity comparison (continuous capacity knob)

Implemented the capacity knob: `Machine.prune_to(n)` (and `--prune-to`) keeps the `n` strongest
units after training, so any config can be placed at *any* unit count for a clean head-to-head.
Results (K=64, L=4, held-out, 4 seeds; alloc=0 pool then prune to `N`):

**Capacity curve** (intact accuracy):

| N | 40 | 55 | 70 | 85 | 100 |
|---|---|---|---|---|---|
| uniform | 0.68 | 0.73 | 0.80 | 0.77 | 0.77 |
| weighted (LEARN) | 0.73 | 0.80 | 0.80 | 0.80 | 0.83 |

**Matched N=68 head-to-head** (intact / scramble): uniform **0.80 / 0.38** == weighted **0.80 / 0.38**.

**Conclusion — claim 1 fully closed.** At the contested 68-unit point the two are an *exact tie*;
across the curve the weighting shows at most a small, noisy +0.03–0.07 edge, none robust. So the
residual count-matching confound is closed: **capacity + data is the lever, not the discriminative
weighting.** Confidence upgraded from medium toward high.

**Floor test** (matched N=85, L4 vs L6): uniform L4 0.77 → **L6 0.48**; weighted L4 0.80 → **L6 0.53**.
Both collapse toward chance as the body lengthens (the L6 absolute is also data/capacity-limited at
K=64); weighting adds only +0.05 and does **not** rescue it. The longer-body ceiling is a **genuine
representational floor** of (static-weighted) Hamming.

**Cycle 3 is concluded.** The machine genuinely learns a transferable rule (scramble + capacity
controlled), scaling with data to ~0.88–0.90; the lever is capacity + data, not any weighting; and
the residual ceiling is representational and grows with body length. The evidence-pointed next
chapter is the **learned encoder** — compress the body out so the type stays separable — not any
further metric, weighting, or capacity tweak.

---

## 15. Cycle 4 — the learned encoder (`encoder.py`)

The recurrent state update becomes a **learned** transition table `g: (state, dropped bit) →
state` (`addr_mode='learned'`, `h=3` → 16 entries); `shift` and `fold` are fixed points of this
family. Clean protocol: memory trains on *train* bodies, `g` is selected on *dev* bodies, reported
on separate *test* bodies with the rule-scramble control.

**Attempt 1 — hill-climbing the table FAILED (the search overfit).** The hill-climbed `g` overfit
the 8-body dev set (dev 0.75 → test L6 **0.46**, gap +0.06) and *degraded* L4 to 0.48 (vs shift
0.83). A free 16-entry table scored on 8 dev bodies is too overfittable.

**Diagnostic — the *structure* is sufficient (hand-built latch).** Key insight: `g` sees only the
dropped bit's *value*, not its position — but the TYPE bit is the *first* bit to drop, so "absorb
the first drop and hold it" is expressible. A hand-built **latch** (state 0 → absorbing state
4/5 keyed on the first drop, held forever) removes the memory horizon. Confirmed (K=40, 6 seeds,
held-out, intact / scramble-gap):

| L | shift | latch |
|---|---|---|
| 4 | 0.83 (+0.48) | 0.78 (+0.44) |
| 6 | 0.31 (**−0.11**) | 0.61 (**+0.21**) |
| 8 | 0.38 (−0.02) | 0.64 (+0.20) |
| 10 | 0.36 (−0.01) | 0.66 (+0.27) |

`shift` collapses to a *zero/negative gap* past its horizon (R+h−4 = 5); the **latch holds a real,
scramble-controlled signal across L6–L10**. A learned recurrent code removes the horizon.

**Conclusions (cycle 4):**
1. **The learned-code structure is sufficient** — a latch carries the discriminative bit
   *arbitrarily far*, where the fixed shift/window cannot. This is the first mechanism to beat the
   horizon, not just the in-band ceiling.
2. **The bottleneck is the *learning*, not the representation.** Naive hill-climbing on a small dev
   set did not find the latch — this is the long-range **credit-assignment** problem (the answer
   supervision is far from where the latch must fire).
3. **Residual:** even with the latch, the long-range gap (~+0.25) is below the short-range gap
   (+0.44) — the *window* body-noise still dilutes retrieval. A fuller encoder must **compress the
   window too**, not only latch the history.

**Next:** *learn* the latch — a credit-assignment method or a structured gating prior that can
discover "latch the first informative drop and hold" from answer-level supervision — plus window
compression for the residual.

---

## 16. Cycle 5 — the learnable gated latch (`gated.py`)

To make the latch *learnable* (cycle 4 showed the free table overfits), use a **structured
encoder family** — a write-gate with a latch prior, parameterised by a short binary write-schedule
`w` of only `len(w)` bits (`blm.gated_latch_table`): on the `k`-th dropped bit, `w[k]=1` ⇒ latch
that bit and hold; else advance. `w=[1,0,0,0]` is the hand latch. `w` is **learned** by
enumerating its tiny `2^len(w)` space on *dev* bodies; compared to `shift` (horizon collapse) and a
**free** transition table hill-climbed on the same dev set (the cycle-4 overfit baseline). Clean
train/dev/test split + scramble control. Hypothesis: the low-parameter structured family is
*learnable and generalises* where the high-capacity free table overfit.

**Result — the latch is learnable.** Enumerating the schedule space on dev recovered
**`w=[1,0,0,0]`** (the latch). Test (held-out, K=40, h=4, 4 seeds; intact / scramble / gap):

| L | shift | free-table | gated-latch |
|---|---|---|---|
| 4 | 0.81 / +0.47 | 0.81 / +0.50 | 0.81 / +0.53 |
| 6 | 0.64 / +0.11 | 0.70 / +0.14 | 0.59 / +0.17 |
| 8 | 0.41 / +0.03 | 0.59 / +0.17 | 0.61 / +0.12 |
| 10 | 0.36 / **+0.02** | 0.41 / +0.19 | **0.72 / +0.33** |

**Conclusions (cycle 5):**
1. **The latch is learnable via a structured prior.** The tiny schedule space recovered the clean
   latch `w=[1,0,0,0]` from data — the inductive bias makes learnable what the free `2^h` table did
   not reliably find.
2. **The learned gated-latch removes the horizon.** At L10, `shift` is dead (gap +0.02) while the
   gated-latch holds strongly (0.72, gap +0.33).
3. **Nuance vs the free table.** At `h=4` the free table *also* partially learned a latch-like rule
   (it beats `shift` at L8/L10), so it was not as catastrophic as at `h=3` — but the gated-latch is
   cleaner and clearly best at the extreme (L10: 0.72 vs 0.41). The structured family is the more
   reliable route.

_(The L10 table cell above is 4-seed and seed-noisy; the verification below corrects it.)_

### 16a. Verification verdict (cycle 5, 4-agent workflow, high confidence)

- **Latch reliably learned — confirmed.** The 16-schedule space collapses to 5 reachable
  behaviours; all 8 schedules with `w[0]=1` are byte-identical "latch-first-drop", and the bench
  TYPE is always the first drop — latching into an absorbing state, *verified to body length 1000*.
  `learn_schedule` recovers the latch **8/8** at L=4/8/10; the L=6 5/8 case was a tie-break artifact
  (now fixed — ties prefer the latch family, all-zeros excluded), not an overfit win.
- **Horizon genuinely removed — confirmed.** `shift`'s address is identical across the two classes
  for all L ≥ 8 (horizon R+h−4=6) while the gated address stays distinct; the 8-seed
  scramble-controlled gated gap is significant at every L (z = 3.4–7.6).
- **Gated beats free — confirmed (best-supported).** Gated wins 4/4 at L10/L12 (+0.27 / +0.21 raw,
  non-overlapping per-split ranges); the free hill-climbed table overfits dev (≈0 held-out scramble
  gap, negative on 2/4 splits at L12). The structured family **cannot** overfit (5-point reachable
  space); it is the only encoder with above-chance scramble-controlled transfer at long L on every split.
- **Correction:** the 4-seed "+0.33 at L10" did **not** replicate at 8 seeds — robust value
  ≈ **+0.156**, and the gap **attenuates with L**: +0.156 / +0.102 / +0.078 at L10/12/14. The latch
  removes the *horizon*, but the (body × latched-state) address space the SOM readout must cover
  **grows with L**, so the readout (not the memory) now limits long-L accuracy.

**Next — window/body compression.** The encoder is proven horizon-free; the remaining attenuation
is a *readout* problem: fold/hash/down-weight the **window body bits** so only the latched
TYPE-carrying state dominates the retrieval kernel, decoupling rule-readout from L.

---

## 17. Cycle 6 — window compression (`window.py`)

Cycle 5 left a *readout* limit: the latch holds the type, but accuracy attenuated with length
because the (window body × latched-state) space the readout must cover grows with `L`. Fix
(`win_keep`): the register still slides at width `R` (driving the latch), but the **address keeps
only the last `win_keep` window bits** — the latch carries long-range memory, so the address window
only needs the local structure (boundary + recent outputs); the body bits are dropped.

**Result — corrected after verification (§17a); my first low-power headline is retracted.** Two
mechanisms with very different evidential standing:

- **The latch is the real, robust win** (cycle 5): `w=[1,0,0,0]` recovered 8/8 on dev, horizon-free
  *by construction*.
- **Window compression genuinely helps and genuinely transfers.** A short address window
  (≈ boundary width) beats the full window on held-out (mean +0.17 across L), and the transfer is
  real and leakage-proof: the `wk=3` answer-position address provably collapses to **exactly two**
  body-invariant, type-only addresses (`111` boundary ++ the latched-type bit), so the scramble arm
  is *structurally forced* to chance. `win_keep=3` is the **mechanistic optimum** (wk2 0.67 / wk3
  0.92 / wk4 0.64 at L8/12; `wk>3` pulls in body bits and pushes scramble *below* chance — overfit).

What I first claimed and now **retract** (it was low statistical power — 16 held-out items/seed):
1. **`win_keep` is NOT reliably learned.** Dev cannot distinguish `win_keep ∈ {2,3,4}` (they tie at
   the ceiling on the 8-body dev set); the search lands on a noisy tie-break (`wk=3` in only 2/8
   runs). `win_keep=3` equals this bench's boundary width — a **principle** ("address window =
   local-structure width: boundary + autoregression"), not a reliably-learned constant.
2. **Accuracy is NOT flat near 1.0.** It is a **bimodal ~0.5/1.0 mixture** (each split solves or
   fails), plateauing ~0.79 for L≥8; the "flat ~0.9" was a noisy average (within-L std ~0.25,
   larger than the L4→L20 change).
3. **Full-window does NOT clearly attenuate** on re-run (it dips then recovers; net slope ~0). The
   honest contrast is "compressed *higher* than full", not "compressed flat vs full decaying".

### 17a. Verification verdict (cycle 6, 4-agent workflow, high confidence)

- **Latch** — robustly learnable, horizon-free by construction, code-verified. ✓
- **Window compression** — genuine improvement *and* genuine, leakage-proof transfer (mechanism
  traced to two type-only addresses; `wk>3` over-fits → sub-chance scramble). ✓
- **Downgrades** — `win_keep` not reliably learned (noisy tie-break over {2,3,4}); held-out is
  bimodal ~0.5/1.0 not flat-near-1.0; full-window does not visibly attenuate. **Root cause: low
  test power** (16 held-out items/seed → near-binary coin flips).
- **Next — raise test power and re-assess:** pool held-out across many seeds with binomial CIs;
  report the per-`win_keep` curve with CIs at several L; extend L to 24–32; reconcile the
  attenuation discrepancy. Lead the story with the latch; present window compression honestly as
  "narrowing the address to local-structure width beats full-window with scramble-controlled
  transfer", stating `win_keep=3` is the bench's boundary width (principle, not learned constant).
  _High-power re-assessment: see §17b._

### 17b. High-power re-assessment (pooled held-out, binomial CIs)

Re-ran with **pooled** held-out items (K=48, 12 seeds, n=216/cell, 95% CI) to settle the three
downgraded points with statistical power.

**Per-`win_keep`** (intact ± CI / scramble):

| wk | L8 intact | L8 scr | L16 intact | L16 scr |
|---|---|---|---|---|
| 2 | 0.67±0.06 | 0.50 | 0.67±0.06 | 0.44 |
| **3** | **0.92±0.04** | 0.51 | 0.83±0.05 | 0.47 |
| 4 | 0.73±0.06 | **0.37** | 0.95±0.03 | **0.35** |
| 6 | 0.65±0.06 | 0.35 | 0.70±0.06 | 0.34 |

**Length curve** (compressed `wk=3` vs full `wk=6`, intact ± CI):

| L | wk=3 | scr | wk=6 | scr | gap |
|---|---|---|---|---|---|
| 4 | 1.00±0.00 | 0.43 | 0.71±0.06 | 0.42 | +0.29 |
| 8 | 0.92±0.04 | 0.51 | 0.65±0.06 | 0.35 | +0.27 |
| 12 | 0.96±0.03 | 0.44 | 0.70±0.06 | 0.35 | +0.26 |
| 16 | 0.83±0.05 | 0.47 | 0.70±0.06 | 0.34 | +0.13 |
| 24 | 0.88±0.04 | 0.42 | 0.75±0.06 | 0.41 | +0.13 |

**Rigorous conclusions (supersede §17a's downgrades where power resolves them):**
1. **`win_keep=3` IS identifiable with power** — CI-separated optimum at L8 (0.92 vs 0.65–0.73);
   the earlier "not learnable" was a small-dev artifact. **But the correct selection objective is
   "high intact *with scramble ≈ chance*"** — raw intact (and even raw scramble-gap) would pick the
   *overfit* `wk=4` at L16 (intact 0.95 but scramble 0.35 = **sub-chance** = body-overfit), whereas
   `wk=3` keeps scramble ≈ 0.5 (clean). Sub-chance scramble is the overfit signature.
2. **Compressed (`wk=3`) holds ~0.85–0.95 across L4–24** — high, roughly flat (mild noise), and
   **CI-separably above full-window (~0.70) at every L** (+0.13 to +0.29). The high-power estimate
   (~0.90) is *higher* than the underpowered re-run's ~0.79; the core "high, roughly
   length-independent, beats full-window" claim is **rehabilitated** — just not "flat at 1.0".
3. **Full-window sits ~0.70 throughout** (no clear attenuation), with mildly sub-chance scramble.

**Net, CI-backed:** latch + window compression (address = local-structure width) achieve **high
(~0.90), roughly length-independent, scramble-validated rule transfer, clearly above full-window** —
solid, with `win_keep=3` the principled (boundary-width) optimum selected by the scramble-clean
criterion, not a "flat-at-1.0 reliably-learned-constant" overclaim.

---

## 18. Cycle 7 — multi-feature memory: does it compose? (`multi.py`)

Does the structured-latch approach scale to remembering **more than one thing**? A 2-feature
recall bench: `[TYPE1 TYPE2][shared BODY L][111][ANSWER 2b][000]`, answer = `f(t1,t2)` — **echo**
(`[t1,t2]`, independent) and **xor** (`[t1^t2, t1^t2]`, joint, needs both features combined). The
**multi-latch** (`blm.multi_latch_table(k,h)`) holds the first `k` dropped bits (the `k` type bits)
then freezes; `k=1` is the single latch and can hold only one feature. `k` is **learned** on dev by
the scramble-clean objective; body-disjoint split + per-body rule-scramble control.

**Result — the structured latch COMPOSES.** `k=2` is learned for both modes. Pooled held-out
(K=48, L=8, win_keep=3, 12 seeds, 95% CI):

| answer | k=1 (one feature) | k=2 (both) | scramble |
|---|---|---|---|
| xor (joint) | 0.50±0.05 | **0.96±0.02** | 0.50 |
| echo (independent) | 0.48±0.05 | **0.65±0.05** | 0.24 |

- **Multi-feature memory works:** `k=2` holds both type bits, is CI-separably above `k=1`, and the
  count `k` is itself learnable.
- **The joint task (XOR) is solved cleanly (0.96)** — the readout combines two latched features;
  `k=1` is exactly at chance (one feature carries no parity information).
- **The independent 4-way task (echo) is only partially solved (0.65)** — counter-intuitively
  *harder* than the joint XOR. The per-answer-bit diagnostic shows why: bit1 (`t1`, read from the
  latch) = **0.90**, but the second *independent* bit (`t2`) = **0.73** even teacher-forced. For XOR
  bit2 = parity = bit1, so it is **copied from the window** (1.00); for echo bit2 = `t2 ≠ t1`, so it
  must be extracted from the latched state while body-internal `[1,1,*]` window collisions intrude.

**Conclusion:** the **memory primitive composes** — multiple features are latched and held, and a
*joint* function of them (XOR) is read out cleanly. The residual (echo's independent second bit) is
the **familiar readout limit** — uniform-Hamming kNN extracting one specific bit from a
multi-feature address amid collisions — *not* a memory failure. The bottleneck, again, is the
readout, not the memory.

### 18a. Verification verdict (cycle 7, 4-agent workflow, medium confidence)

- **Composition holds in-band and is genuine — with fragility.** `k=2` is CI-separably above `k=1`
  for both modes (L≤32); `learn_k` recovers `k=2` 5/5 seed-blocks (pooled). Mechanism verified:
  `multi_latch_table(2,4)` makes **4 distinct injective absorbing states** decoding to `(t1,t2)`,
  while `k=1` collapses the 4 classes to 2 (holds `t1` only) — structurally unable to do the 2nd bit.
  *Caveats:* echo's per-seed advantage is marginal (~10/16 wins at L=8); single-split `learn_k` is a
  coin-flip for echo (reliable recovery needs multi-seed pooling); and **echo collapses at L=48**
  (`k2=k1=0.44`) — the latch does *not* compose for echo at large gaps.
- **XOR genuine — confirmed.** 0.938±0.040, scramble *exactly* at chance; leakage structurally
  impossible (kept window = `[1,1,1]` boundary for all 4 classes; type bits outside the R-window;
  body shared). The **rule** is genuine at every L, but intact **accuracy decays with L**
  (0.96→0.92→0.75) — a held-out generalisation cost, not a memory failure.
- **Echo residual = readout, NOT memory, and NOT fixable by the available levers.** Both bits are
  provably in the latched state; the 2nd independent bit is the bottleneck (full .65 / bit1 .90 /
  bit2 .73) because at that step the kept window groups *only* by the just-emitted `t1`
  (`[1,1,0]` vs `[1,1,1]`), so `t2` must be read from a register sub-bit against a same-window
  collision. Every readout lever failed (β=8 unchanged; more capacity *worse*; `wk=2/0` worse) —
  a deeper limit of the frozen-address Hamming-vote readout.
- **Methodology flags:** results are seed-fragile (xor L8 0.958→0.891 from 12→16 seeds) and
  **compute-bound** (a single k=2 eval at ep=200 didn't finish a 4-seed pool in 10 min — this is what
  maxed the CPU). Next runs: cut per-eval cost (lower epochs / cache pooled pairs across seeds),
  report per-seed win/tie/loss (not only pooled CIs), and lock the echo claim to L≤32.

**Next chapter — the readout.** The latch holds the bits; the gap is *extraction*. Replace the
single global Hamming-vote with a readout that can separate collinear register sub-bits under an
identical window — e.g. a per-state / per-window-group calibrated head (a small linear decoder over
the latched register), or an address transform exposing the register bits as separable dimensions.
Validate on echo `bit2 | true-bit1` (capped ~0.73). Do *not* chase larger L for echo until the
readout is fixed.

---

## 19. Cycle 8 — the learned decoder, and a correction: it was an address collision

We built the learned decoder the §18a verdict asked for — and the investigation **corrected the
"readout limit" diagnosis itself.**

**(a) A global learned decoder is *worse* than the vote.** A gradient-free perceptron over the
address (single bits + pairwise ANDs, so it *can* select one bit or combine two) underperforms the
kNN-vote (K=40, L=8, 8 seeds): echo full **0.38 vs 0.53**; xor full **0.56 vs 0.84**. Reason: a
*global* linear head is **swamped by the filler positions** (which vastly outnumber the answer
positions), so it underfits the rare answers — whereas the vote's **locality** (only nearby units
matter) is its strength. (Implementation: `decoder.py`.)

**(b) The real blocker is an ADDRESS COLLISION, not the readout.** Direct measurement: **160/160**
answer-bit-2 addresses *also* map to a different target elsewhere in training. The same address must
emit two different bits → **no readout — vote or decoder — can resolve it.** After the latch
freezes, the register is identical at every position and the 3-bit window cannot distinguish the
*answer region* from the *body*. So echo's cap is **address ambiguity**, not extraction — this
supersedes §18a's "readout limit" framing (the cycle-7 readout-lever probes failed precisely
because the address itself is ambiguous).

**The fix is address enrichment, not a fancier readout.** A sticky **"boundary-seen" bit** halves
the collisions (160 → 80); the rest are collisions *within* the post-boundary region, so full
disambiguation needs a little more — a short **post-boundary step counter**. Then every answer
position has a unique address and even the plain vote can extract the answer.

**Corrected next step:** extend the recurrent state to track **WHERE we are** (boundary-seen +
post-boundary position), not to build a smarter decoder. *The latch holds WHAT; the missing piece
is WHERE.* (And a global readout is the wrong shape regardless — locality/query-conditioning is what
makes the vote work.)

---

## 20. Cycle 9 — region/position latch: the multi-feature task is SOLVED (`region.py`)

Built the region/position latch: **address = [boundary_seen, post-boundary position (3 bits)] ++
[window-slice] ++ [feature-latch (the type bits)]**. Also fixed a latent bench bug — the body now
ends in `0` so the `111` boundary is **unambiguous** (a body ending in `1` fused with the boundary
into an early `111`, which had been aliasing the position counter).

Result (K=40, L=8, **8 seeds pooled**, scramble-controlled):

| mode | baseline (no region) | + region/position latch | scramble |
|---|---|---|---|
| echo | 0.72 | **1.00**  (bit2 1.00) | 0.21 |
| xor | 0.97 | **1.00** | 0.52 |

Answer-bit-2 address collisions: **160/160 → 0/160.**

**The multi-feature recall task is now solved** (echo *and* xor → 1.00, pooled, scramble at chance).
Once the address encodes **both *what* (feature latch) and *where* (region/position latch)** it is
collision-free and body-invariant, so the *simple vote* maps it perfectly and generalises to unseen
bodies.

**Completed lesson chain:** latch (*what*, long-range) → window compression (drop body-noise) →
multi-latch (several features) → **region/position latch (*where*)** ⇒ the address fully and
unambiguously encodes the situation, and a plain readout suffices.

**Honest correction to the running theme.** Across cycles I kept concluding "the *readout* is the
bottleneck." For this benchmark family that was **wrong**: the real gap was an **incomplete address**
(it encoded *what* but not *where*). Completing the address solved the task with the simplest
readout — the decoder was a red herring. (A learned/attention readout may still matter for harder
tasks, but it was *not* the blocker here.)

### 20a. Verification verdict (cycle 9, 4-agent workflow, high confidence — no correction needed)

- **Task genuinely solved.** echo & xor full = 1.00, bit2 = 1.00 held-out at every L tested (L=4,8
  over 4 seeds; L=16,24 single-seed).
- **Deterministic core (strongest evidence):** answer-bit-2 collisions **0/160 at *all* L ∈
  {4,8,16,24}** for both modes (was 160/160) — seed-independent.
- **No leakage:** the answer-position address is body-invariant (1 distinct address per class) and
  the target is never copied into it — under scramble the answer addresses collide 4/4, which is
  only possible if the address encodes (position, type), not the answer. The latch freezes holding
  both type bits *before the answer exists* (0 bad over 320 items).
- **Sound:** the `111` boundary is unique at position `2+L` with **0 violations across 140,800
  sequences**; train/dev/test are id-disjoint; the mechanism is structurally L-invariant.
- **One disclosed caveat (not a bug / not leakage):** xor's scramble control sits ~0.44–0.56 (not
  ~0.25) because xor's full answer has only 2 distinct values, raising the full-match chance
  baseline; echo's control is clean (~0.17–0.25) and xor's collision-core collapses correctly.
- **Consolidation follow-ups (recorded, not yet done):** use the collision-core as xor's scramble
  control; full 16-seed pooled-CI accuracy sweep for L=8/16/24; optional L=32/48 stress + a
  body-content-disjoint split.

---

## 21. Cycle 10 — recall vs computation (parity) (`parity.py`)

Evidence-led probe: does *recurrent-state + address + lookup* do **aggregation**, or only recall?
Task: answer = **parity of all F=6 feature bits**, `[features][111][p,p][000]`. Three recurrent
states (each + region/position + small window), held-out on **unseen feature patterns** (K=40, 6
seeds):

| recurrent state | intact | scramble |
|---|---|---|
| **accum** (1-bit running XOR, frozen at boundary) | **1.00** | 0.58 |
| latch — first 2 features | 0.48 | 0.54 |
| latch — **all 6** features | 0.60 | 0.38 |

**Findings:**
1. **Computing the aggregate solves it.** A 1-bit running-XOR accumulator (frozen at the boundary) =
   parity → exactly **2 distinct answer-addresses** → generalises to unseen patterns → 1.00.
2. **Holding the raw inputs does NOT generalise — even holding *all* of them.** `latch-all-6` (0.60)
   holds every feature bit, yet lookup can't generalise parity because **parity isn't
   Hamming-smooth** (flip one input bit → the answer flips, but nearest-neighbour returns the wrong
   one). `latch-first-2` (0.48) ≈ chance.
3. ⇒ **The recurrent state must encode the task-relevant *computed* feature, not the raw inputs.**
   *Recall = hold (latch); computation = accumulate.* The lookup readout is fine either way; the
   recurrent state is where the task-specific computation must live.

So the framework **does** extend from recall to computation — *provided the right aggregate is
computed into the address.* This raises (without yet answering) the open frontier: **how to *learn*
which computation/accumulator a task needs** (here it was hand-built as running-XOR).

*Caveat:* the `[p,p]` 2-value answer makes the scramble baseline ~0.5 (weak control, as with xor);
the deterministic core is that `accum` yields exactly 2 answer-addresses, so it generalises by
construction.

### 21a. Verification verdict (cycle 10, 4-agent workflow, high confidence — confirmed & strengthened)

- **accum solves + generalises** parity: **1.00 held-out at F = 6, 8, 10, 12** (no decay).
  *Deterministic*, not just statistical: at answer positions region+window bits are constant, so the
  only content-varying address bit *is* the frozen running-XOR = true parity → exactly 2
  parity-addresses → any unseen pattern collapses onto a seen address → lookup cannot fail (no
  Hamming-smoothness needed; address space is F-invariant because parity has 2 values).
- **holding fails even holding ALL inputs:** parity is maximally non-Hamming-smooth (100% of
  Hamming-distance-1 pattern pairs have opposite parity); nearest-neighbour over the raw held inputs
  scores 0.73 at F=6 and **0.33 (below chance) at F=8**; latch-6 is no better than latch-2.
- **sound / leak-free:** `parity_acc` reads only pre-boundary features (poisoning the answer bits
  leaves it unchanged); the `111` boundary is unique.
- **METHODOLOGY FIX (applies to earlier cycles too):** `gated.split` is **ID-disjoint, not
  content-disjoint** — `gen_feats` yields only 24/64 patterns at F=6, so test patterns overlapped
  train (6/8). The decisive **content-disjoint** control *strengthens* the result: accum 1.00 (gap
  **+0.65**) while latch-2 (0.31, **−0.27**) and latch-6 (0.21, **−0.40**) go **negative**. Removing
  the leak leaves accum untouched (as the deterministic argument predicts) and sinks the latches.
  → **adopt content-disjoint splits + the negative-gap control as the headline** (retire the weak
  duplicated-bit scramble baseline). Confidence: high.

---

## 22. Cycle 11 — compute-vs-hold holds for a non-degenerate aggregate (`aggregate.py`)

Extends cycle 10 from parity (2 values) to **popcount mod 4** (4 values), using the
**content-disjoint** split (the cycle-10 methodology fix). Held-out, 5 seeds (chance = 0.25):

| F | accum (running mod-4 counter) | latch-all (holds every input) | #patterns |
|---|---|---|---|
| 6 | **1.00** (scr 0.46) | 0.14 (scr 0.37) | 24 |
| 8 | **1.00** (scr 0.26) | 0.15 | 81 |
| 10 | **1.00** (scr 0.22) | 0.18 | 274 |

- **accum (running mod-4 counter, frozen at the boundary) solves + generalises, F-invariantly.**
  The answer-address space is the **4 aggregate values, not 2^F**, so unseen patterns collapse onto
  the 4 seen addresses → generalisation by construction; scramble → ~0.25 as the content-leak
  shrinks with more patterns (the content-disjoint control is clean here).
- **latch-all FAILS (≤ chance, 0.14–0.18)** even holding every input — a non-Hamming-smooth
  aggregate can't be generalised by lookup over raw inputs.
- ⇒ The **compute-vs-hold** principle holds beyond the degenerate 2-value case: the recurrent state
  must *compute* the task aggregate (here a running mod-m counter); the address space then scales
  with the **number of aggregate values (m)**, not the input count — hence F-invariant.

_Recorded; not yet adversarially verified (paused new runs at the user's request)._

---

## 23. Cycle 12 — LEARN the recurrent computation (the central question) (`learn_state.py`)

Every prior cycle had *me* hand-picking the recurrent state per task. This one tests whether the
machine can **learn which computation a task needs.** A structured family —
`hold-k` (latch the first k bits) and `count-m` (running popcount mod m, frozen at the boundary) —
plus a meta-learner that picks the member generalising best on a **content-disjoint dev set**,
tie-broken toward **fewest state bits** (the simplest computation).

Result (held-out, 1 seed):

| task | LEARNED | expected | test acc | dev signal |
|---|---|---|---|---|
| recall (echo) | **hold-2** | hold-2 | 1.00 | hold2=1.00, hold3=1.00, count*≤0.44 |
| parity (mod 2) | **count-2** | count-2 | 1.00 | count2=1.00, count4=1.00, hold*≤0.44 |
| popcount mod 4 | **count-4** | count-4 | 1.00 | count4=1.00, count3=0.81, count2=0.62, hold*≤0.38 |

**The machine recovers the right recurrent computation from data — without being told:** the latch
for recall, running-XOR (`count-2`) for parity, the mod-4 counter for mod-4, each at 1.00 held-out.
The parsimony tie-break recovers the **minimal** correct computation (`hold-2` not `hold-3`;
`count-2` not `count-4` for parity — `count-4` also solves parity but is over-complex). This converts
*"I hand-engineer a solver per task"* into *"the machine selects the right computation from data."*

**Honest scope:** this is **selection over a small, hand-defined structured family** (`{hold, count}`),
*not* learning an arbitrary computation from scratch. That is the practical/principled form (cf.
cycles 4–5: structured families are learnable; free tables overfit) — a library of structured
primitives + data-driven selection. Open extension: grow the family / learn the members themselves.

_1 seed; not yet adversarially verified._

---

## 24. Cycle 13 — selection COMPOSES (a task needing two computations) (`compose.py`)

Task: `[type bit][F features][111][answer = (type, parity(type+features))][000]` — **bit1 is a held
feature, bit2 is a computed aggregate.** The meta-learner chooses over **singles AND hold×count
pairs** (content-disjoint dev, parsimony tie-break).

Result (F=8, held-out): **LEARNED `{hold-1, count-2}`, test acc 1.00.**

| candidate | dev | bits |
|---|---|---|
| **{hold-1, count-2}** | 1.00 | **2** (picked) |
| {hold-1, count-4} | 1.00 | 3 |
| {hold-2, count-2} | 1.00 | 3 |
| {hold-2, count-4} | 1.00 | 4 |
| every single (hold-*, count-*) | ≤ 0.50 | — |

**Selection composes:** no single computation solves the task (all singles ≈ 0.50); four
`{hold, count}` pairs reach 1.00; parsimony recovers the **minimal** combination. So cycle 12's
"learn the computation" extends to **"select the minimal *combination* of computations"** — the
machine composes primitives into a tiny learned program for a task needing both memory and
computation.

**Honest scope:** selection over a hand-defined family *and* a hand-defined combination space
(singles + hold×count pairs); the "program" is a concatenation of state-features (no control flow),
and the answer's two bits are separable (one held, one computed). Deeper composition (joint /
non-separable answer functions; a *learned* combination space) is the open extension. 1 seed;
mechanistically transparent (singles can't, the right pairs can, parsimony picks the minimal).

---

## 25. Methodology — does more data help? data scaling vs representation (`datascale.py`)

Question raised mid-research: would increasing the dev set / training on larger data help? Two
learning curves on popcount mod 4 (F=10, content-disjoint split) separate the two failure modes.

**Curve 1 — SELECTION vs DEV size (statistical axis):**

| dev items | P(pick count-4) | test acc of pick |
|---|---|---|
| 2 | 0.43 | 0.80 |
| 4 | 0.63 | 0.88 |
| 8 | 0.90 | 0.98 |
| 16 | 1.00 | 1.00 |
| 54 | 1.00 | 1.00 |

More dev data **fixes selection**: a tiny dev mis-picks (overfits) 57% of the time; at ≥16 dev items
it reliably recovers `count-4` (1.00). → increasing the dev set is the principled lever to make a
*freer* search safe.

**Curve 2 — TEST acc vs TRAIN size (representational axis):**

| train items | raw-input (hold-all) | structured count-4 |
|---|---|---|
| 41 | 0.33 | 0.33 |
| 83 | 0.17 | 0.56 |
| 166 | 0.02 | 1.00 |

More training data only helps the **right** representation (`count-4` → 1.00). It makes the **wrong**
one *worse* (raw-input 0.33 → 0.02): with a non-Hamming-smooth target and a Hamming-retrieval
readout, more data sharpens confidently-wrong interpolation from mis-aggregated neighbours.

**Bottom line:** data scaling helps the **search** (reliable selection of the computation) but cannot
substitute for the right **representation** — it amplifies whatever representation you chose, for
better or worse. So "just train on more data" is **not** a substitute for the primitive library; they
are complementary — more dev → reliable selection; the library → the representations to select among;
more train → pays off only once the representation is right.

---

## 26. Cycle 14 — prediction → POLICY: bit-native action that learns from reward (`action.py`)

Extends "learn the computation" from supervised next-bit prediction to **reward-driven action**
(framing-doc criteria #5 action, #9 over-trials adaptation). A contextual-bandit agent: sees a bit
context, forms an address via a family member, **chooses an action**, gets a **reward bit**, and
learns the policy **online from reward** (no labels).

**Part A — improve over trials + generalise** (parity-act, 2 actions, chance 0.50):

| representation | reward over trials | held-out |
|---|---|---|
| count-2 (right) | 0.997 → 1.00 | **1.00** |
| hold-all (raw) | 0.936 → 1.00 | **0.50** |

Both improve over trials (online adaptation from reward); the **right** representation generalises to
held-out contexts (1.00), the **raw** one memorises training and collapses to chance on held-out
(0.50) — the cycle-11 / data-scaling representation lesson, now for policy.

**Part B — select the computation FROM REWARD** (no labels):

| task | LEARNED | expected | held-out reward | dev signal |
|---|---|---|---|---|
| parity (2 act) | **count-2** | count-2 | 1.00 | count2=1.00, count4=1.00, count3=0.88, holds≤0.44 |
| mod4 (4 act) | **count-4** | count-4 | 1.00 | count4=1.00, count3=0.81, count2=0.62, holds≤0.38 |

The meta-learner recovers the right (minimal) computation **from reward alone** — `count-2` for
parity-reward, `count-4` for mod4-reward. So "learn the computation" is not tied to labelled
prediction; it works under reward supervision and outputs an **action**.

**Significance:** the predictor becomes an **agent**. The core now selects the right computation to
*act* (not just predict), learns the policy online, improves over trials, and generalises to unseen
contexts — all bit-native, no language.

**Honest scope:** single-step contextual bandit, small synthetic action sets, a tabular policy over
the learned address; not yet sequential/multi-step planning or non-stationary environments. 1 seed.

---

## 27. Cycle 15 — SEQUENTIAL / multi-step action: a bit-native MDP with delayed reward (`mdp.py`)

Extends cycle 14 from a single-step bandit to a real **MDP**: navigate a 1-D corridor (P=8) to a goal
shown once at the start; the reward arrives **only at the final step**. Solved by tabular **TD
Q-learning** over a learned bit-address. Tests temporal credit assignment + memory across the horizon
+ the representation lesson, in sequence.

| representation | train reward | held-out reward | #states |
|---|---|---|---|
| `rel` = (sign(goal−pos), step) — **computed** relative direction | 1.00 | **1.00** | 21 |
| `abs` = (pos, goal, step) — raw absolute | 0.50 | **0.00** | 134 |
| `nomem` = (pos, step) — no goal memory | 0.25 | **0.00** | 35 |

`rel` learning curve (reaches-goal rate, delayed reward): **0.13 → 0.82** over training (greedy eval
= 1.00; chance ≈ 0.12).

**Findings:**
- **Credit assignment works:** with only a terminal reward, TD backup over the learned bit-address
  lifts the policy from chance (0.13) to optimal (1.00 greedy) — sequential, multi-step.
- **Memory across the horizon is required:** `nomem` (no goal memory) fails (0.25 / 0.00).
- **The representation lesson carries to sequential control:** the *computed* relative-direction
  state (21 states) generalises to held-out goals (1.00); the *raw* absolute state (134 states)
  memorises and collapses on held-out goals (0.00). Same lesson as cycles 11 / 14 / data-scaling
  (§25), now for RL.
- **Learn-the-computation extends to RL:** selecting the representation by held-out reward recovers
  `rel`.

**Significance:** the bit-native core is now a *sequential decision-maker* — delayed-reward credit
assignment over a horizon, with the right *computed* representation giving generalisation.

**Honest scope:** small tabular MDP (P=8, H=8, 3 actions), single deterministic corridor, hand-given
representation candidates; not yet function approximation at scale, stochastic dynamics, or long
horizons. 1 seed.

---

## 28. Scale on REAL data — the bit-native core as a real text compressor (`scale.py`)

The consolidation step (after a run of small single-mechanism cycles): assemble the core as a
**next-bit predictor** and run it on a **real corpus** (300–772 KB of public-domain English text,
bytes → bits), measured by held-out **bits-per-bit** (= cross-entropy = compression; raw = 1.0000;
the bit-native analogue of an LLM's perplexity), externally referenced against **gzip**. The model is
the exact-count form of the content-addressable machine: a context address → empirical next-bit
distribution (KT/Laplace smoothed); at scale exact counts replace the Hamming-kernel/SOM used on the
tiny benches.

**Representation comparison** (300 KB, held-out 20%):

| representation | held-out bits/bit |
|---|---|
| order-0 within-byte (phase, cur) | 0.5733 |
| bit-window k=8 (raw, unaligned) | 0.6617 |
| bit-window k=16 (raw, unaligned) | 0.3792 |
| byte-aware B=1 (computed phase) | 0.4296 |
| byte-aware B=2 (computed phase) | 0.3367 |
| byte-aware B=3 (computed phase) | 0.2806 |
| **CORE backoff (orders 3→2→1→0)** | **0.2727** |
| _external ref: gzip_ | _0.3585_ |

**Data scaling** (CORE backoff model):

| bytes | core bits/bit | gzip bits/bit |
|---|---|---|
| 100 K | 0.3070 | 0.3673 |
| 300 K | 0.2727 | 0.3585 |
| 600 K | 0.2689 | 0.3567 |
| 772 K | 0.2791 | 0.3559 |

**Findings:**
- **The bit-native core is a real predictor:** it compresses real English to **27 %** (0.2727
  bits/bit) and **beats gzip** (0.3585) at every data size — the "see it in real, like LLM training"
  milestone (bits/bit = the bit-native perplexity/compression metric).
- **The representation lesson holds at scale on real data:** the *computed* byte-aware representations
  dominate raw bit-windows of comparable size (byte-aware B=2 0.3367 < bit-window k=16 0.3792) and
  scale better with order; even order-0 within-byte (just phase + current partial byte) carries real
  signal (0.5733).
- **More real data → better,** to ~600 KB (0.307 → 0.269); the slight uptick at 772 KB is a held-out
  distribution shift (the file tail is the Gutenberg license/footer), not a reversal — consistent with
  the data-scaling finding (§25) on real data.

**Honest scope / relation to the design:** at scale the "learned binary address + Hamming kernel +
SOM" reduces to an exact-count context model (a PPM-like byte predictor) — Hamming generalisation is
unnecessary when data is plentiful; what is tested, and what matters, is the **representation** (the
recurrent / computed features written into the address). **gzip is a standard but weak baseline**;
strong compressors (PPM/CTW/PAQ, LLMs) reach far lower bits/bit. The claim is "**the core works and
beats gzip on real text**," not state of the art. One book, one language, ≤ 772 KB.

---

## 29. Learn the REPRESENTATION on real data — discover byte structure from scratch (`represent.py`)

§28 hand-gave the byte/phase structure. This removes that crutch: the core **discovers** which context
features predict the next bit, on the real corpus, via learn-the-computation at scale.

**Part 1 — period scan** (address = `i mod p` alone): bits/bit is minimised at **p=8** (0.77), tied
with its multiple p=16, with partial dips at the divisors/multiples 4 and 12 and ≈ chance (0.99)
elsewhere. The core **discovers the byte period of text** purely from predictive value, without being
told.

**Part 2 — greedy forward selection** over a pool {`i mod p` (p=2..16), lag bits (1..16)}; it builds
a representation from scratch:

| step | feature added | bits/bit |
|---|---|---|
| 1 | **mod-8** (byte phase, chosen first) | 0.7725 |
| 2 | **lag-8** (aligned bit in previous byte) | 0.7430 |
| 3–7 | lag-1,2,3,4,5,6 | 0.5323 |
| 8–16 | lag-5,7,9,10,11,12,14,15,16 | **0.3759** |
| — | _hand-given byte-aware B=2_ | _0.3445_ |

**Findings:**
- **The core rediscovers byte structure:** it selects the byte phase (`mod-8`) first and the
  byte-aligned previous bit (`lag-8`) second — the same structure §28 hand-coded — from data alone.
- **The discovered representation converges toward the hand-engineered one:** the gap shrinks from
  0.53 vs 0.35 (7 features) to **0.376 vs 0.345** (16 features, ~0.03 apart) and was still narrowing.
  Given enough capacity the *learned* representation is essentially as good as the *supplied* one —
  without being told about bytes.
- **Residual gap:** greedy-over-individual-bits + crude global backoff is slightly weaker than the
  structured byte-aware + order-backoff CORE (§28), and the curve had not fully converged.

**Significance:** the central "learn the computation" result now holds at scale on real data for
**representation discovery** — the core finds the predictive structure (byte period + alignment)
itself, rather than being handed it. **Honest scope:** greedy feature selection over a hand-defined
candidate pool (periods + lags); a real but still bounded search; ≤ 120 KB, one corpus.

---

## 30. Push the compressor — online logistic context mixing (`mix.py`, `mix_sse.py`, `two_layer_mix.py`)

The scaled core's predictor, improved from count-backoff (§28) to **online logistic context mixing** —
the lpaq/PAQ idea, and exactly the framing's "neural-network mapping" in bit-native form: several
byte-aware context models each vote a probability for the next bit; a single **online-trained
logistic unit** mixes them in the logit (stretch/squash) domain; the mixed P codes the bit; weights +
counts update online. Adaptive over the whole stream (like gzip on the whole file). Metric = bits/bit
(= compression; raw = 1.0000).

Results (300 KB cap, **whole-stream / last-20%**; self-run and adversarially verified):

| model | prose | code |
|---|---|---|
| _gzip (whole file)_ | _0.3585_ | _0.2386_ |
| `mix.py` logistic mixing (orders 0–4) | 0.2636 / 0.2398 | 0.2269 / 0.1890 |
| `two_layer_mix.py` (mixer-set + SSE) | **0.2400** / 0.2158 | 0.1962 / 0.1582 |
| `mix_sse.py` (SSE/APM + match + orders 0–6) | 0.2411 / 0.2167 | 0.1878 / 0.1473 |
| **`mixmax.py` (merge — best of both)** | **0.2392 / 0.2163** | **0.1835 / 0.1460** |

- All four **beat gzip** on both prose and code; the merged model compresses real prose to ~24 % and
  code to ~18 %.
- **The merge `mixmax.py` wins both corpora at once** — prose 0.2392 (< two_layer 0.2400) and code
  0.1835 (< mix_sse 0.1878) — combining `two_layer_mix`'s global + final-layer mixers with `mix_sse`'s
  byte-level match model and order-5/6 contexts. **Verified leakage-free** by a built-in future-bit-flip
  causality self-test (`python mixmax.py flip`: flipping a future bit leaves every past prediction
  bit-identical, while later predictions do change).

**Verification (multi-agent workflow — 2 independent skeptics + judge):** `mix.py` is **sound and
leakage-free** — causality proven three ways (count-invariant assertion c0+c1 ≤ i; future-bit-flip
invariance; from-scratch re-implementation matching to 4 decimals); the metric is a correct mean
−log₂ P(true bit) (a constant-0.5 predictor = exactly 1.0000); single-thread, stdlib only. Both
challengers were verified causal (future-flip test) and reproduce their numbers.

**Honest scope:** bits/bit depends on the byte budget (longer stream → lower, an online-adaptation
effect, not leakage): `mix.py` is 0.2636/0.2269 at 300 KB but 0.2743/0.2351 at 200 KB. gzip is a
standard but **weak** baseline; strong compressors (PAQ/cmix) reach far lower. This is "the bit-native
core, with a neural mixer, is a real compressor that beats gzip," not state of the art. ≤ 300 KB, one
prose + one code corpus.

**Next:** the road to *strong* compressors — CTW-style context mixing, larger / more diverse corpora,
longer context + match models, adaptive count decay for non-stationarity — or apply the same
predictor to a different modality through a dumb adapter.

---

## 31. Agency on a REAL stream — bit-native change / anomaly detection (`stream.py`)

Turns the predictor into an **agent that acts on a real stream**: the online logistic mixer predicts
each bit; the per-byte **surprise** (−log₂ P over its 8 bits) is the signal; a sustained mean-shift in
surprise makes the core **emit a flag — its action**. This realises the framing doc's own examples of
intelligence-without-language ("detecting anomaly in a stream" / "detect when a pattern changes", §6
and §11) and exercises non-stationarity. Stream = **real** data with known regime shifts: English →
Python source → English (270 KB, built from the two real corpora so the change points are ground
truth).

**Result:**
- Surprise jumps sharply at **both** true regime changes: eng→code **+1.11** bits/byte (2.12 → 3.23),
  code→eng **+0.41** (1.83 → 2.24).
- The detector flags **both** boundaries with low latency (**554** and **192** bytes).
- The per-5 KB surprise trace tracks the regimes (English ~2.1–2.6, Python ~1.7–2.0 bits/byte).

**Findings:**
- The bit-native core **detects distribution shift on real data** — its prediction surprise is a
  usable anomaly signal, and the flag is a genuine action from the *same* predictor (no separate model).
- **Asymmetry (honest):** the return boundary (code→eng) is weaker (+0.41 vs +1.11) because the model
  **retained English statistics** from the first segment — its memory makes the second English regime
  less surprising. A real property of the predictor, not a detector artefact.
- The 8 extra flags are mostly genuine **local** content anomalies (e.g. a repetitive passage dipping
  to ~1.3 bits/byte) plus 2 refractory echoes of the real detections — i.e. the detector is doing
  anomaly detection, broader than the 3 planted change-points.

**Honest scope:** a thin surprise-threshold detector on top of the `mix.py` predictor (windows/margin
tuned, not learned); one constructed real stream with 2 planted boundaries; recall 2/2, with extra
flags being real local anomalies rather than random noise. **Significance:** closes the framing
scorecard's *action* axis on **real** data — the same core that compresses also **acts** (flags
changes/anomalies) on a real stream, supporting "intelligence below language" beyond compression.

---

## 32. Toward STRONG compressors — four techniques, and the real lever (`mixns.py`)

A 4-way parallel push (adversarial multi-agent workflow) added four advancements on top of `mixmax`,
each benched on prose + code (200 KB) and causality-self-tested:

| technique | prose | code | Δ vs mixmax |
|---|---|---|---|
| `mixmax` (base) | 0.2495 | 0.1922 | — |
| **`mixns`** (recency count-halving + RMSProp mixer LR + hashed orders 8/12/16) | **0.2465** | **0.1891** | −0.003 |
| `deep_sse` (5-APM calibration chain) | 0.2481 | 0.1913 | −0.001 |
| `mixctw` (binary CTW input) | 0.2482 | 0.1911 | −0.001 |
| `multimatch` (run-length + multi-hash) | 0.2495 | 0.1918 | ~0 |

All four are **verified causal** (future-bit-flip self-test) and independently re-measured; `mixns`
wins both. The techniques are **sub-additive** (they attack overlapping high-order signal), so stacking
all four is only ≈0.244/0.187 — diminishing returns.

`mixns` verified at 300 KB: **prose 0.2364 / code 0.1806** (causal), vs gzip 0.3585 / 0.2386.

**The real lever is data, not more tricks.** `mixns` prose whole-stream by corpus size:

| bytes | bits/bit |
|---|---|
| 100 K | 0.2638 |
| 300 K | 0.2364 |
| 500 K | 0.2269 |

More data bought **≈0.037** (100 K→500 K) — an order of magnitude more than the ≈0.003 from the best
shallow trick.

**Conclusion:** shallow modelling tricks have hit diminishing returns; reaching PPM/PAQ territory
needs **scale** (more / diverse data) and deeper modelling, not more shallow mixer inputs. `mixns` is
the current strongest single model and is verified leakage-free. **Honest scope:** gzip is a weak
baseline; absolute bits/bit (~0.18–0.24) is still well above strong compressors; single-corpus,
≤ 500 KB.

---

## 33. Confidence-driven ACTION with consequences (`decide.py`)

Exercises the framing's **confidence / uncertainty** criterion together with action: the core predicts
the next **byte** by an 8-bit greedy rollout (no peeking) and **decides commit-or-abstain** by its own
confidence (product of the per-bit max-probs). Reward: +1 correct commit, −4 wrong commit, 0 abstain.
Real prose, 200 KB.

| threshold | coverage | acc@commit | net reward/byte |
|---|---|---|---|
| 0.00 (always commit) | 1.00 | 0.565 | −1.177 |
| 0.50 | 0.57 | 0.784 | −0.045 |
| 0.80 | 0.29 | 0.911 | **+0.158** |
| 0.90 | 0.18 | 0.947 | +0.129 |

- Next-byte top-1 accuracy **0.565** (chance 1/256).
- **Confidence is calibrated:** accuracy@commit rises monotonically (0.57 → 0.95) as the threshold
  tightens — higher self-confidence really does mean higher accuracy.
- **Always-commit loses** (−1.177 under the asymmetric reward); **confidence-gating wins** (+0.158 at
  τ=0.80) by committing only when confident and abstaining otherwise.

**Significance:** the core uses its **own uncertainty** to make profitable decisions — confidence-driven
action with real consequences, and the monotone accuracy-vs-confidence shows the confidence signal is
meaningful. Exercises framing criteria #5 (action) + confidence. **Honest scope:** greedy-rollout
next-byte prediction + a *swept* confidence threshold (the policy is thresholded, not learned); one
real corpus; the reward structure (λ=4) is illustrative.

---

## 34. Scaling up — multi-MB real text, and the capacity ceiling (`mix.py`, `mixns.py`)

Scaled to a **5.4 MB** real-English corpus (3 public-domain books). Whole-stream bits/bit:

| size | `mix.py` (orders 0–4) | `mixns` (strong) | gzip |
|---|---|---|---|
| 1 MB | 0.2512 | **0.2222** | 0.3560 |
| 2 MB | 0.2514 | **0.2215** | 0.3607 |
| 4 MB | 0.2462 | — | 0.3631 |

**Findings:**
- **Both beat gzip at every scale** (~0.22–0.25 vs ~0.36); gzip does **not** improve with more data
  (fixed 32 KB window), the bit-native models do.
- **The strong model wins at scale:** `mixns` (~0.22) clearly beats the simple mixer (~0.25) — its
  high-order + match components pay off with enough data.
- **Honest catch — bits/bit *plateaus* beyond ~1 MB** (`mixns` 0.2222 → 0.2215; `mix` flat ~0.25).
  Small-scale data gains were large (§25/§32: 0.26 → 0.23 from 100 → 500 KB); at MB scale they shrink
  to ~0. The bottleneck has shifted from **data** to **model capacity**: a fixed-capacity context model
  saturates once its contexts are populated, and more data can't buy structure it cannot represent.

**Conclusion — the scaling triad:** data, model **capacity**, and representation must scale *together*.
§25 showed data can't fix the wrong representation; §32 showed shallow tricks plateau at small data;
this shows **data plateaus at fixed capacity**. Pushing from our ~0.22 toward strong compressors
(~0.15 on enwik8) needs more **capacity** (longer contexts/match, deeper/larger mixing), not just more
data. **Honest scope:** pure-Python ceiling (~5 MB tractable, minutes-to-hours, one core); 0.22
bits/bit sits between gzip (0.36) and strong compressors (~0.15); LLM-scale data + capacity would need
a compiled/vectorised implementation. _(Refined in §35: with PyPy reaching 4 MB, `mixns` continued
down to 0.2161 — diminishing returns, not a hard plateau.)_

---

## 35. Overcoming the Python ceiling — PyPy (CPU, not GPU)

The hot loop is **sequential + sparse** (dict context tables), so the lever is faster per-step
**native CPU** execution, not parallelism/GPU. First, cheapest step: run the unchanged pure-Python
code under **PyPy** (a JIT Python).

Measured (bit-for-bit **identical** bits/bit — correctness preserved):

| run | CPython 3.13 | PyPy 7.3.19 | speedup |
|---|---|---|---|
| `mix.py` @500 KB | 27.5 s | 8.4 s | 3.3× |
| `mixns.py` @300 KB | 57.0 s | 14.9 s | 3.8× |

**~3.3–3.8× out of the box** — capped by per-bit `bytes(slice)` allocation and dict-with-tuple keys
that the JIT can't fully remove. This is **CPU expansion of the same architecture, not GPU**: the model
is sparse + sequential (each bit depends on the prior step), which a GPU can't exploit; GPUs would only
matter under a dense neural pivot.

Leveraged immediately to reach **4 MB** on the strong model (≈9 min on CPython → **2.5 min** on PyPy):

| size | `mixns` whole-stream | gzip |
|---|---|---|
| 1 MB | 0.2222 | 0.3560 |
| 2 MB | 0.2215 | 0.3607 |
| 4 MB | **0.2161** | 0.3631 |

**This refines §34:** bits/bit is **not** at a hard plateau — 4 MB continued down (0.2215 → 0.2161);
the earlier "plateau" was partly an artefact of only reaching 2 MB on CPython. More data still helps
(diminishing but real), and capacity would help more.

**Path forward:** bigger speedups need PyPy-friendly **integer (rolling-hash) context keys** or the
planned **Rust/C++ native core** (the route to enwik8-scale, ~0.15). PyPy now buys ~3.5× and a clearer
view of the scaling trend, with no code change and no GPU.

---

## 36. PyPy- and Rust-friendly integer keys — same quality, ~12× faster (`mixfast.py`)

Replaced the per-bit context key — `(phase, bytes(partial), bytes(prev_bytes))` — with a single
**lossless integer**: a sentinel-prefixed pack of phase + the partial current byte + the previous
bytes, carried in a rolling `htail` integer (no per-bit `bytes` allocation, no tuple hashing). The
encoding is a **bijection** with the old key (same equivalence classes).

**Proven bit-identical** to `mix.py` (whole/tail match to 1e-12). Speed @500 KB:

| | CPython 3.13 | PyPy 7.3.19 |
|---|---|---|
| `mix.py` (bytes/tuple keys) | 27.5 s | 8.4 s |
| `mixfast.py` (integer keys) | 13.6 s | **2.26 s** |

- Integer keys: **2.0×** on CPython alone (no bytes allocation).
- They let the JIT do far better: **6.0×** PyPy-vs-CPython (was 3.3× with byte keys).
- **Combined (integer keys + PyPy): ~12×** over the original (27.5 → 2.26 s), output unchanged.

This confirms the analysis: lossless keys cost **no quality**, **multiply** the PyPy win, and converge
the data model toward the eventual **Rust/C++** design (integer keys + open-addressing hash tables of
fixed-width keys) — so it **de-risks** the native port rather than complicating it.

The same change applied to the strong model gives `mixnsfast.py` — **bit-identical** to `mixns`
(0.2581495288 match) and causality-clean (future-flip test passes). Its win is smaller (CPython 57.0 →
40.5 s, ~1.4×; PyPy 10.1 s, ~5.6× combined) because `mixns`'s cost is dominated by the mixer / RMSProp
/ APM float math rather than the context dicts — but it's a verified, lossless speedup of the model we
actually scale.

---

## 37. Scaling with the fast stack — clean homogeneous curve (`mixfast.py` / `mixnsfast.py` under PyPy)

With the integer-key models under PyPy, runs that were *hours* on CPython are *minutes* (e.g. 88 M
bits / 11 MB in ~50 s for `mixfast`). Used this to test scaling properly.

**Heterogeneous corpus is not a clean scaling axis.** On 4 concatenated books (11 MB), `mixfast` went
5 M 0.2528 → 11 M 0.2609 — *up* — **but so did gzip** (0.369 → 0.374), because the appended Shakespeare
(archaic English / verse) is genuinely harder. Composition dominates; concatenating different books ≠
"more of the same data."

**Clean homogeneous scaling** (single source — War & Peace prefixes), `mixnsfast`:

| bytes | whole-stream | last-20% | gzip |
|---|---|---|---|
| 0.5 M | 0.2382 | 0.2258 | 0.3616 |
| 1 M | 0.2288 | 0.2213 | 0.3630 |
| 2 M | 0.2222 | 0.2139 | 0.3652 |
| 3 M | 0.2181 | 0.2104 | 0.3651 |

On homogeneous data, bits/bit improves **monotonically** with scale (0.238 → 0.218, still dropping at
3 M; last-20 % to 0.210), while gzip stays flat (~0.365). **More data does help cleanly** — the earlier
"plateau"/non-monotonic readings (§34) were corpus-composition artefacts, not a model ceiling.

**Takeaways:** (1) the fast stack (PyPy + lossless integer keys) makes real-scale experiments
practical — tens of millions of bits in seconds-to-minutes, one core, no GPU. (2) Clean scaling needs
**homogeneous** data; heterogeneous concatenation conflates scale with difficulty. (3) The strong model
holds ~0.21–0.22 vs gzip ~0.365 at every scale and is still descending at 3 M of one source.
**Honest scope:** ~3 M single-source is still small vs enwik8 (100 M); the dict-based orders grow with
data (RAM) — fixed-size hashing / the Rust core remain the path to much larger scale.

---

## 38. Bounded-RAM via fixed-size hashing — reaches scale, collision-limited (`mixnshash.py`)

To break the dict-RAM ceiling, the byte-orders + sparse + word tables were converted from growing
dicts to **fixed-size hashed count arrays** (the high orders already were), with **8-bit checksum tags**
(evict-on-collision — the PAQ/zpaq trick). RAM is now **bounded** (~370 MB, independent of data size)
and the model stays **causal** (flip-test passes).

**Collision cost** (1.5 MB War & Peace, vs the exact `mixnsfast`):

| model | whole-stream |
|---|---|
| `mixnsfast` (exact dict) | 0.2254 |
| `mixnshash` (fixed hash, 22-bit, tagged) | 0.2739 |

→ **+0.048 bits/bit.** Tags didn't recover it: the high orders (5–6) run at high load, so collisions
are *frequent* and eviction loses count history about as much as merging — the fundamental
memory↔quality tradeoff (no free lunch at small fixed tables).

**Reaching scale** (enwik8, the standard 100 MB Wikipedia benchmark — bounded RAM runs sizes the exact
dict can't):

| bytes | `mixnshash` whole-stream | gzip |
|---|---|---|
| 10 M | 0.2751 | 0.3688 |
| 30 M | 0.2729 | 0.3668 |

- **Reaches 30 MB** (240 M bits) on fixed ~370 MB — the bounded-RAM goal, achieved; beats gzip (0.273
  vs 0.367).
- **Collision-limited:** roughly *flat* 10→30 MB (tables saturate), stuck ~0.273 — the +0.048 cost
  caps it, and more data stops helping once the fixed tables are full.

**Conclusion:** fixed-size hashing buys **bounded RAM** (so we reach 30 MB) at the cost of collisions
that **cap quality** and grow with scale. Getting **both** large scale **and** low bits/bit needs large
*tuned* memory + efficient hash tables — practical in **Rust/C++** (the planned core), not pure Python
(370 MB+ Python arrays are clumsy and sizing iteration is slow). enwik8 reference: gzip 0.37, strong
compressors ~0.15; ours ~0.273 (bounded) / ~0.225 (exact, smaller scale) sits between. This is the
clearest signal yet that the **native core is the next real lever**. _(§39 refines this: the native
core gives bounded memory + near-exact quality and ~2× speed, but scale is memory-latency-bound, not
language-bound — Rust is not a 100× lever here.)_

---

## 39. The Rust core — `blmrs`: first results and an honest correction (`blmrs/`)

Started the native core. Installed Rust 1.96 (gnu, self-contained linker) and built `blmrs`: a faithful
port of `mixfast` (logistic mixing, orders 0–4, the same lossless integer keys).

**Correctness:** the HashMap variant is **bit-for-bit identical** to Python `mixfast`
(0.253666 == 0.253666 @500 KB) — the algorithm port is verified.

**Flat engine** (fixed-size open-addressing arrays + 8-bit checksum tags = bounded memory):

| size | `blmrs` flat | exact (Python) | Δ |
|---|---|---|---|
| 500 KB | 0.254028 | 0.253666 | +0.0004 |
| 5 MB | 0.253601 | 0.252790 | +0.0008 |

→ **near-exact**: with proper *high-bit* multiplicative hashing + large tables (2²⁴ slots), the
collision cost is tiny — unlike Python `mixnshash` (+0.048), because Rust affords big flat tables
cheaply. The native core gets **bounded memory *without* the quality hit**.

**Speed** (same model):

| | @500 KB | @5 MB |
|---|---|---|
| CPython `mixfast` | 13.6 s | ~135 s |
| PyPy `mixfast` | 2.26 s | 39.5 s |
| `blmrs` (native flat) | 1.37 s | 22 s |

→ **~1.7–1.8× over PyPy, ~6× over CPython.**

**Honest correction.** Earlier sections framed Rust as "the 100× lever for scale." **That was wrong.**
At scale this workload is **memory-latency-bound** — each bit does ~5 random accesses into tables far
larger than cache, so throughput is ~1.8 Mbits/s (~550 ns/bit ≈ 5 × a DRAM miss) *regardless of
language*. Native buys a **modest ~2×** (lower per-access overhead), not orders of magnitude. The real
Rust wins here are **(a) bounded memory with near-exact quality** (big flat tables Python can't afford)
and **(b)** a clean, fast-to-iterate native base. The genuine path to large speedups is **cache-aware
design** (smaller working sets, fewer/cheaper accesses, SIMD on contiguous data), not language alone.

**Next:** port the strong model (`mixns`) to the native core; explore cache-aware table layouts; the
bounded-memory + near-exact property already lets `blmrs` scale at fixed RAM where Python degraded.

---

## 40. The strong model in the native core — quality at scale (`blmrs/src/bin/strong.rs`)

Ported the **full strong model** (`mixnsfast`: orders 0–6, hashed high orders 8/12/16, byte-match,
sparse, word, two-layer context-selected mixer + global + final, 2 chained SSE/APM, non-stationary
count-halving, RMSProp) to the native core, with flat bounded tables (high-bit hash + checksum tags).

**Verified near-exact** (1 MB War & Peace):

| | whole-stream |
|---|---|
| Python `mixnsfast` (exact) | 0.228847 |
| `blmrs-strong` (native, obits=23) | 0.229253 |

→ Δ **+0.0004** — the port is faithful (tiny flat-collision + libm delta), and ~2.6× faster than PyPy.

**At scale on enwik8** (obits=24), vs Python's bounded `mixnshash`:

| bytes | `blmrs-strong` | Python `mixnshash` (bounded) | gzip |
|---|---|---|---|
| 10 M | **0.2253** | 0.2751 | 0.3688 |
| 30 M | **0.2203** | 0.2729 | 0.3668 |

- The native strong model beats Python's bounded version by **~0.05 bits/bit** — Rust affords bigger
  tables (2²⁴ vs 2²²) and uses the corrected high-bit hash, so collisions don't cripple the high orders.
- It **keeps improving with data** (10 M 0.2253 → 30 M 0.2203), unlike the Python bounded version which
  was collision-capped (flat ~0.273). The strong model genuinely benefits from scale.

This is the concrete payoff of the native core: **the good model at scale, with bounded memory and
near-exact quality** — what Python couldn't do. enwik8 context: gzip 0.37, this **0.220**, strong
compressors ~0.15 — a real compressor between gzip and SOTA, still improving with data. **Honest scope:**
~0.5 Mbits/s (strong model, memory-bound); SOTA needs far more models/memory/tuning; bounded tables
still cap the very highest orders at extreme scale.

---

## 41. Full enwik8 — the headline number (`blmrs-strong`)

Ran the strong native model on the **full enwik8** (100 MB / 800 M bits — the standard compression
benchmark), `obits=27` (~11 GB tables), 28 min, one core.

**Result: whole-stream `0.211107` bits/bit** (last-20% 0.2072) → enwik8 compresses to **~21.1 MB**.

Scaling held to the end — monotonic, still descending:

| size | `blmrs-strong` bits/bit |
|---|---|
| 10 M | 0.2253 |
| 30 M | 0.2203 |
| **100 M (full)** | **0.2111** |

Where it sits on the enwik8 ladder (well-known approximate compressed sizes):

| compressor | enwik8 | bits/bit |
|---|---|---|
| gzip | ~36 MB | ~0.36 |
| bzip2 | ~29 MB | ~0.29 |
| PPMd | ~24 MB | ~0.24 |
| **blmrs-strong (ours)** | **~21.1 MB** | **0.211** |
| lpaq1 | ~20 MB | ~0.20 |
| paq8 | ~16 MB | ~0.16 |
| cmix (SOTA) | ~15 MB | ~0.15 |

So the from-scratch bit-native core, scaled to the native strong model, **beats gzip, bzip2, and
PPMd**, and lands right next to **lpaq1** — a genuine, respectable context-mixing compressor between
the classical tools and the strong PAQ family. **Honest scope:** bits/bit is the model's
**cross-entropy** (the ideal arithmetic-coded size; a real archiver adds a small coder + decompressor
overhead, which the Hutter Prize counts); ~0.5 Mbits/s; SOTA (~0.15) needs many more models + GB-scale
tuned memory + cache-aware engineering. But as a measure of the *model*, **0.211 bits/bit on full
enwik8** is a real, defensible headline for a bit-native predictor built from first principles.
_(Improved to 0.209 in §42.)_

---

## 42. Push toward SOTA — a model-stacking round (`blmrs-strong`)

A round of principled additions to the strong native model, each benched on enwik8:

- SSE/APM chain extended **2 → 4 stages** (added current-partial-byte and match-length contexts);
- a **second, longer byte-match model** (min length 8);
- more **high-order reach** (orders 24, 32) + bigger high-order tables (HBITS 20 → 22).

Effect (whole-stream bits/bit, matched obits):

| size | baseline strong | improved | Δ |
|---|---|---|---|
| 10 M | 0.2253 | 0.2240 | −0.0013 |
| 30 M | 0.2203 | 0.2186 | −0.0017 |
| **100 M (full)** | 0.2111 | **0.2093** | −0.0018 |

**Updated headline: `0.2093` bits/bit → enwik8 ~20.9 MB — now essentially at the lpaq1 level.**

Per addition the gains are small (~0.0003–0.0007), but they **grow with scale** (more data populates
the high orders and the match models, so each is worth more at 100 M than at 10 M).

**Honest conclusion:** model-stacking works and the framework supports it cleanly, but each principled
addition yields ~thousandths. Closing the remaining gap to SOTA (~0.15) is a deep, separate program —
**indirect-context bit-history state machines** (the PAQ "ICM"), an **NN/LSTM mixer** (cmix's big
lever), many more context models, and heavy per-context tuning — not a handful of increments. We have
demonstrated the direction and reached **lpaq1 territory**; SOTA is research-grade from here. Causality
is preserved by construction (every addition is predict-before-update; the base was verified
bit-identical to the causal Python reference).

---

## 43. Path B — open-ended induction of bit-native computations (`induce.py`)

Toward the end goal (**bit-native intelligence**, not language modeling): can the core **induce** the
right computation by *composing* from a library of primitives — learning to compute, rather than only
*selecting* from a 6-member hand-made family (cycles 12–13)?

Induction over a 12-primitive library (latch-k, count-mod-m, count-bucket, max-run, last-k, …),
exhaustive composition to depth 3, content-disjoint held-out + a random-answer scramble control:

| task | induced program | test | scramble | verdict |
|---|---|---|---|---|
| parity | `cmod2` | 1.00 | 0.58 | SOLVED |
| recall | `latch2` | 1.00 | 0.36 | SOLVED |
| mod4 | `cmod4` | 1.00 | 0.30 | SOLVED |
| majority | `cbucket` | 1.00 | 0.59 | SOLVED |
| maxrun≥3 | `maxrun` | 1.00 | 0.70 | SOLVED |
| compose | `latch1 + cmod2` | 1.00 | 0.37 | SOLVED |
| automaton (01-transition parity) | _(spurious)_ | 0.58 | 0.62 | **BREAKS** |
| automaton + primitive | `transcount` | 1.00 | — | SOLVED |

**Findings:**
- **6/7 induced** — the correct *minimal* program for each, generalising to held-out (1.00) with
  scramble at chance, and it picks *different* primitives per task (latch for recall, counters for
  parity/mod-m, bucket for majority, max-run for runs). Genuine learning-to-compute across a diverse
  task set, not pattern-fitting.
- **Breaks honestly.** Transition-parity is not in the library's span → no composition determines it;
  the "best" found is spurious, and the **scramble control exposes it** (scramble 0.62 ≈ test 0.58; a
  real program has scramble ≪ test). The method is self-honest — the controls catch the fake.
- **Extensible** — add a transition primitive and it induces `transcount` → 1.00. The frontier extends
  when the needed primitive exists.

**Significance (honest, no hype):** this is cheap, CPU, example-driven **induction of computations** —
composing and validating programs from data — a mechanism *unlike* LLM scaling, and it learns *what to
compute* across recall / counting / comparison / run-detection / composition.

**The real open question (Path B's frontier):** it is bounded by (a) the primitive **library** (can't
induce what isn't in the span — automaton), (b) composition **depth + search** (exhaustive to 3; deeper
is combinatorial — the known hard part of program synthesis), and (c) **hand-added** primitives. The
genuine "intelligence via a different course" bet is whether the system can **grow its own library** —
*invent the missing primitive from a failure* — rather than be handed it, and whether induction
transfers to **real-data** tasks. Those are the next steps.

---

## 44. Path B, step — primitive INVENTION: grow the toolbox from a failure (`invent.py`)

A step (not the goal) toward open-ended induction: when the fixed library *breaks*, can the system
**invent** the missing primitive by searching a **generative** space — primitives parameterised by a
predicate over (prev, cur) bits × an aggregation (count mod m / ever / max-run)?

| task | base test | result |
|---|---|---|
| 01-parity | 0.57 | **INVENTED** `01:cnt2` → 1.00 (the 01-transition counter) |
| 10-parity | 0.62 | **INVENTED** `10:cnt2` → 1.00 |
| 11-parity | 0.60 | **INVENTED** `and:cnt2` → 1.00 (found "both bits 1") |
| saw-11 | 1.00 | base already solved (`maxrun`) |
| 010-parity | 0.61 | **NOT invented** — needs a 3-bit predicate, beyond the 2-bit space |

**Findings:**
- **Invention works.** On tasks the fixed library can't do (base ≈ chance), the system searches the
  generative space and *discovers* the missing primitive (transition / pair counters), solving at 1.00,
  scramble-clean. It grows its own toolbox from failures.
- **Self-honest.** base 0.57–0.62 → invention 1.00; the scramble control guards against spurious
  "inventions."
- **Invention has its own frontier.** `010-parity` needs a 3-bit predicate, beyond the 2-bit generative
  space → honestly not invented — and that frontier is *itself* extensible (a 3-bit-predicate space
  would reach it).

So there is a **hierarchy of frontiers**, each extending the last — fixed library → invention (2-bit
predicate DSL) → richer generative spaces — every level cheap, example-driven, self-honest, mappable.

**Significance + honest scope (a step, not the goal):** a recursive, *verified*, example-driven
mechanism for **discovering** computations — qualitatively unlike LLM scaling. But each generative
space is still **hand-defined** here; truly open-ended discovery needs the system to grow its **own**
space. Next: a learnable/growable generative space; deeper computations; **inventing groupings** (the
bit→byte→event unit question); and grounding on a real test case.

---

## 45. Path B step — recursive synthesis: grow the space, hit the search wall (`recurse.py`)

A step beyond invent.py (which broke at 3-bit patterns): give the system a **fixed grammar** of stream
transducers `{s, not, lag_k, and/or/xor}` + aggregations and let it **synthesise arbitrary-depth
computations by composition** (iterative deepening). A fixed finite grammar that *generates an unbounded
space*.

**Honesty:** the search tries thousands of expressions, so a single content-disjoint split is **not** a
sufficient control — every winner is **re-validated on 5 fresh seeds**, SOLVED only if it survives.

| task | found | cross-seed (5 fresh seeds) | program |
|---|---|---|---|
| 01-parity | d2 | 1.00 | `cnt4(~s ^ lag1(~s))` — a non-obvious but correct program |
| 010-parity | d2 | **1.00** | `cnt2(lag1(s) & (~s & lag2(~s)))` — the **exact 010 detector** |
| 0110-parity | — | — | NOT FOUND within budget |

**Findings:**
- **Grows its own space.** It synthesises the *exact* 010 detector by composition — crossing the
  3-bit-pattern frontier the fixed 2-bit-predicate invention (§44) could **not** — from a fixed finite
  grammar, no hand-added predicates. Verified across 5 fresh seeds (cross-seed is essential; here it
  *confirms* the programs are real, not search-overfit).
- **Hits the search wall.** `0110-parity` is expressible in the grammar, but the bounded iterative
  deepening did not find it — the combinatorial explosion of program synthesis. **The limit is now
  search, not expressiveness.**
- **Methodology (no fake realities):** at large search scale, single-split controls are insufficient;
  cross-seed validation is required, and is built in. The honesty guard must scale with the search.

**Significance + honest scope (a step):** the "grow your own space" mechanism works for moderate depth —
a fixed grammar generating *verified* computations that cross prior frontiers. It now hits the classic
program-synthesis **search wall**; scaling it (type-directed / library-learning / guided search) is the
open frontier, and the genuine "different course" bet rests there.

---

## 46. Path B step — library learning: the loop works, abstraction quality is the wall (`library.py`)

DreamCoder-style bootstrapping: process a curriculum and add each solved program (its feature stream)
to the library as a reusable primitive, so harder tasks compose from learned pieces. Cross-seed
validated throughout.

| stage | result |
|---|---|
| flat (base grammar) `0110-parity` | NOT FOUND (the §45 search wall) |
| curriculum `01-parity` | SOLVED → learned `L1 = ~s^lag1(~s)` (transition stream) |
| curriculum `10-parity` | SOLVED → learned `L2 = s^lag1(s)` (= same transition stream, **redundant**) |
| curriculum `010-parity` | SOLVED **reusing L1**: `~s & (L1 & lag1(L1))` |
| `0110-parity` WITH the library | still NOT FOUND |

**Findings:**
- **The bootstrapping loop works.** `010-parity`'s solution *reused* the learned `L1` — the library is
  genuinely composed-from; reuse happens.
- **But it did not crack `0110`.** The extracted abstractions are not the *right* reusable pieces:
  `L1` and `L2` are **redundant** (both the transition stream), and it never learned the clean
  01-/10-detectors that `0110` (= `and(lag2(01-detector), 10-detector)`) needs — the "keep the first
  solution" bias yields non-ideal primitives.
- **Honest crux:** the loop is easy; **good abstraction extraction is the wall.** Cracking it needs
  refactoring / compression to find the right common sub-programs (DreamCoder's actual hard step), not
  just keeping winning streams.

**Significance + honest scope:** the whole Path B arc — induction → invention → recursive synthesis →
library learning — is mechanically real and **self-honest** (cross-seed throughout; no fake success
here, the loop works but the wall stands). It traces the "different course" bet to its genuine crux:
**discovering *good* abstractions efficiently** — search + compression over programs. That difficulty
(not expressiveness, not the loop) is where the bet now squarely rests, and it is a known-hard open
problem.

---

## 47. Path B step — abstraction refactoring: a precise, honest miss (`refactor.py`)

Attempt to fix §46 by **mining** reusable primitives: enumerate the solution space per curriculum task
and keep every distinct **cross-seed-validated** solving stream as a candidate abstraction (since
`cnt2(01-detector)` *also* solves 01-parity, the clean detectors should be *in* the solution space).
Then test `0110` with the enriched library.

**Result:** mined 8 (01-parity), 12 (10-parity), 2 (010-parity) validated solving streams → a
13-primitive enriched library → `0110-parity` **still NOT FOUND**.

**Why (the precise crux):**
- The mined abstractions are **boundary-tricks** (transition-XOR forms like `~s^lag1(~s)`), *not* clean
  detectors. The grammar's zero-fill `lag` **contaminates** clean detection at the edge:
  `and(~lag1(s), s) = [s[0], 01-det₁, …]` → `cnt2` = `(s[0] + #01) mod 2` ≠ 01-parity. So the **exact**
  pattern-detectors don't aggregate cleanly; the synthesis finds boundary-trick solutions that pass
  validation but **don't compose** to deeper patterns.
- Mining solving streams therefore surfaces tricks, not the reusable detectors `0110` needs.

**Honest conclusion:** cracking the wall needs (a) a **boundary-aware grammar** (so exact pattern
detection aggregates correctly) and (b) **parameterised abstraction** — a detector *template*
parameterised by the pattern, discovered by anti-unifying analogous solutions — i.e. higher-order /
typed program synthesis. The "different course" is real and **self-honest at every step**; its crux is
now precisely located: **parameterised abstraction discovery + boundary handling**, the heart of
inductive program synthesis (DreamCoder/Stitch territory) and a hard open frontier. No fake success —
the attempt honestly failed and told us *exactly* why. _(A focused boundary fix — adding validity
masks `m_k` to the grammar — was also tried: it mined more streams (library of 26) but `0110` still
NOT FOUND; the synthesis folds masks into new trick-forms rather than clean composable detectors.
Confirms the wall is parameterised abstraction, not boundary handling alone.)_

---

## 48. Cross-domain generalisation — DNA (`blmrs-strong` on real genomes)

Does the core generalise off English to a domain with genuinely different structure (codon period-3,
reverse-complement palindromes, long repeats)? The "dumb adapter" is just packing each base
(A/C/G/T → 2 bits) and streaming it into the **existing** `blmrs-strong` — **no DNA-specific tuning**.
Metric = the same whole-stream bits/bit (× 2 = **bits/base**); raw 2-bit packing = the 2.000 floor.

| genome | bases | `blmrs-strong` bits/base | vs floor 2.0 | specialised DNA field |
|---|---|---|---|---|
| E. coli K-12 (low redundancy) | 4.64 M | 1.936 (last-20% 1.912) | −0.06 | ~1.85–1.90 (E. coli is hard) |
| human chr21 (repetitive) | 40.1 M | **1.679** (last-20% 1.768) | **−0.32** | ~1.6–1.7 (specialised band) |

**Findings:**
- **It generalises off English.** Below the 2-bit floor on both genomes with only a 2-bit adapter —
  capturing real DNA structure it was never built for.
- **On the repetitive human chromosome it lands at 1.679 bits/base — in the specialised
  DNA-compressor band (~1.6–1.7)** — because the byte-**match model** (its text-repeat strength)
  transfers directly to DNA's long repeats. Apples-to-apples against the same context-mixing family
  (NAF/GeCo3/JARVIS), off-the-shelf.
- **On the hard, low-redundancy E. coli it is near the floor (1.936)**, just above specialised
  (~1.85–1.90): few repeats for the match model, and the untuned weaknesses bite.

**Honest weaknesses (all fixable):** (a) **byte-misalignment** — `blmrs-strong` is hardwired to 8-bit
bytes, but DNA's units are 2-bit bases and **codons (period-3 = 6 bits)**; (b) **no reverse-complement
modelling** (DNA compressors model RC palindromes explicitly); (c) on low-redundancy genomes the match
model has nothing to bite. A base/codon-aware adapter + RC modelling would likely push **both** lower
(E. coli toward ~1.85, human below 1.6).

**Conclusion:** the bit-native core generalises cross-domain off-the-shelf, and on the genomes that
matter most (large, repetitive) is **competitive with specialised DNA compressors of the same family**
— strong evidence the architecture captures *real structure*, not English-specific quirks. (Reference
ranges are from the DNA-compression literature; bacterial ~1.72–1.9, human ~1.6–1.7. Exact same-genome
GeCo3/JARVIS numbers would sharpen the head-to-head.)

---

## 49. DNA — base/codon-aware adapter (`dna.py`)

§48's byte-aware result left two fixable weaknesses (byte-misalignment, no DNA-specific structure).
`dna.py` is a base/codon-aware context-mixing model: it operates on **2-bit bases**, conditions context
on previous **bases**, adds an explicit **codon phase** (period-3 reading frame), and runs a
**base-granular VERIFIED match model**. RMSProp-stabilised mixing.

| genome | byte-aware (§48) | base/codon | + rev-complement | specialised field | floor |
|---|---|---|---|---|---|
| E. coli (low redundancy) | 1.936 | 1.9145 | **1.9079** | ~1.85–1.90 | 2.0 |
| human chr21 (repetitive) | 1.679 | 1.6438 | **1.6156** | ~1.6–1.7 | 2.0 |

**Findings:**
- Each targeted adapter helps: aligning to DNA's natural units (2-bit bases, codon period-3) + a
  verified base-match model, then a **reverse-complement** match model (inverted repeats), improve
  **both** genomes over the byte-aware model.
- **Human chr21 → 1.6156 — at the better edge of the specialised DNA-compressor band** (~1.6–1.7,
  NAF/GeCo3/JARVIS family), essentially **matching** that class.
- E. coli → 1.9079 — within the specialised range (~1.85–1.90) on the hardest (low-redundancy) genome.
- Reverse-complement helped more on human (−0.028) than E. coli (−0.007) — as expected, since the human
  genome is dense in inverted repeats (Alu in both orientations).
- **Debugging note worth keeping:** the first version was *worse than the floor* (2.0067) because the
  match model fired on hash **collisions** with high confidence — confident-wrong predictions are
  catastrophic. **Verifying matches** (comparing the actual preceding bases) fixed it; the RC match is
  verified the same way. A real lesson for any confident sub-model: **verify before you trust.**

**Headline:** off a from-scratch *English* compressor, three targeted DNA adapters (base/codon
alignment, base-match, reverse-complement) move the core from "generalises off English" to **matching
specialised DNA compressors of the same family** (human in-band ~1.62, E. coli in-range ~1.91) — concrete
evidence the architecture models *real structure*, cheaply **retargetable per domain**. (Reference
numbers are literature ranges; an exact same-genome GeCo3/JARVIS run would sharpen the head-to-head.)
## 50. Path B — CRACKING the wall: boundary-aware detectors + induced `detect(P)` (`wake_lgg.py`, `probe_*.py`)

§47 located the crux precisely: the wall needs **(a) a boundary-aware grammar** so exact pattern
detection aggregates cleanly, **and (b) parameterised abstraction** — a `detect(P)` *template*, not
concrete winning streams. A fan-out research pass (12 angles) then **built and adversarially verified
five independent prototypes**, each re-run from scratch against the live `induce.py` harness. The
baseline wall was re-confirmed first: `recurse.py` finds `01-parity` (`cnt4(~s^lag1(~s))`) and
`010-parity` (`cnt2(lag1(s)&(~s&lag2(~s)))`) but reports **`0110-parity` NOT FOUND** at depth-6 / 9000
streams. All five prototypes cross it — cross-seed 1.00 on fresh seeds, scramble-clean, and
**generalising to UNSEEN patterns** (not just hitting `0110`):

| prototype | mechanism | `0110` | unseen-pattern generalisation | honest depth |
|---|---|---|---|---|
| `probe_antiunify_…` | boundary-aware `lit(s,k,b)` detector + **Plotkin LGG induces `detect(P)`** | ✅ cs 1.000 | **4/4** (incl. unseen arity `00110`); reconstructs the hand detector bit-for-bit | template genuinely **induced** by anti-unification |
| `probe_param_detect_cegis_…` | `detect(s,P)` + CEGIS over `(P, agg)` | ✅ cs 1.000 | **6/6** by binding only `P`, frozen `agg=cnt2` | template hand-supplied; only the parameter learned |
| `probe_sentinel_boundary_…` | sentinel (option-monad) lag fixes boundaries; the project's **own** bottom-up search then finds `0110` | ✅ depth-4 `cnt2(~s&lag1(s)&lag2(s)&~lag3(s))` | **3/3** | proves the **boundary half** independently |
| `probe_diff_soft_detector_…` | differentiable conv-detector + smooth parity head (gradient, not enumeration) | ✅ recovers `W=[0,1,1,0], m=4` | **4/4** incl. `m=5` unseen arity | the **"different course"** cross-check |
| `probe_cvec_canonical_dedup_…` | full-domain cvec dedup **excludes the trick e-class** | ✅ full-domain-exact | **15/15** unseen (lengths 2–6) | the **honesty layer** |

The verified fixes:
- **Boundary-aware detection.** Replace zero-filling `lag` with a positional literal
  `lit(s,k,b) = [1 if (i>=k and s[i-k]==b) else 0]`, AND the contiguous literals, and **mask to
  `i>=m-1`** (the validity window). Then `cnt2(detector_P) ≡ trans_pat(s,P) % 2` is an **exact
  identity** — verified *exhaustively* over all inputs at L=6, 8, 10 (zero mismatches), so it is a
  theorem, not an L=12 coincidence. The zero-fill version disagrees on 12–49 % of rows. The sentinel
  probe is the clincher: **fix the boundary and the existing blind search finds `0110` on its own** —
  the search wall was a *representation* defect wearing a search costume.
- **Parameterised abstraction.** Anti-unify the per-pattern detector ASTs (group by arity → the
  polarity slot that varies becomes a metavariable `P[j]`; a fold over the literal list lifts arity)
  into `detect(P) = cnt2(AND_j lit(s, |P|-1-j, P[j]))`. Freeze it, then for an unseen task bind **only**
  `P` from examples (leave-one-pattern-out). This is the move `library.py`/`refactor.py` could not make
  because they kept concrete streams.

**Findings:**
- **The wall is cracked, verified five ways.** `0110-parity` is solved scramble-clean, cross-seed,
  leakage-free, by a parameterised detector — by five independent routes (enumerative, CEGIS,
  boundary-algebra, gradient, e-graph). The baseline genuinely fails; the crack is non-trivial.
- **Generalisation is real, not memorisation.** Held-out patterns the solver was never given
  (`0011`, `1001`, `00110`, …) are solved by binding only the parameter, surviving cross-seed +
  balanced-scramble + negative controls (`detect(P)` correctly refuses to fire on majority /
  total-parity / mod3, and `|P|≥6` patterns are honestly UNSAT — too rare at L=12).
- **A real harness bug, fixed.** `recurse.py`'s fixed `scramble<0.7` gate is **mis-calibrated**:
  P-parity labels are class-imbalanced at L=12 (e.g. `00110` majority-baseline ≈ 0.78), so the gate
  **false-rejects** genuine solvers on rare patterns. The fix is **balanced accuracy** (mean per-class
  recall) → real 1.00 / scramble ≈ 0.50 for every pattern. (Verified separately: the `0110` crack also
  survives the *original* raw `<0.7` gate, so this change is not load-bearing for the headline — only
  for not false-rejecting imbalanced patterns. Worth back-porting to `recurse.py`.)

**Significance + honest scope (the ceiling, named not hidden):** the "different course" — *learn what to
compute, then abstract it into a reusable parameterised primitive* — now crosses the exact wall that
stopped induction → invention → recursion → library learning → refactoring. **But:** in four of the five
prototypes the *structural shape* of `detect(P)` (AND-of-delayed-literals + validity mask) is **built
into the grammar by hand**; only `antiunify` *induces* the shape (by LGG over independently-searched
detector ASTs), and even there LGG suffices only because the wake-search returns canonically-aligned
ASTs. **No prototype yet demonstrates fully autonomous discovery of the detector template from raw
failures alone**, and everything here cracks **one** task family — sliding-window pattern-count parity.
The genuinely open question — *is parameterised abstraction a general mechanism, or a per-family manual
move?* — is therefore **not yet settled**. The canonical mechanism is consolidated in `wake_lgg.py`
(wake-synthesise clean detectors → LGG-induce `detect(P)` → bind-only-`P` generalisation, with the cvec
honesty layer and balanced gate); the make-or-break test is **§51 (M3): the same machinery on a
genuinely *different* family**, with no hand-built template. No fake success — the wall is down, and the
next wall is named.

---

## 51. Path B — M3: is the abstraction mechanism GENERAL, or a per-family move? (`m3_different_family.py`)

§50 cracked the wall for **one** family (parity of contiguous-pattern occurrences) and named the real
open question: when we move to a *different* family, must we hand-write a new template, or does the same
`WAKE → SLEEP(LGG) → BIND` machinery discover the new abstraction itself? If it is a per-family manual
move, the wall just reappears one family up and "open-ended" fails. M3 runs the identical machinery —
with a **general** grammar (literals over arbitrary lag *subsets*, not just contiguous; a small set of
aggregations) — on three graduated families, with the same honest controls (balanced cross-seed on
fresh seeds, multi-draw balanced-scramble, full-domain direct-solver check over all 2¹² inputs, no
pattern supplied).

| family | what is new | result |
|---|---|---|
| **M3a — gapped patterns** (`1.1`, `1..1`, `0.1`, `1.1.1`) | literal conjunctions at **non-contiguous** lags; the §50 contiguous `detect(P:string)` literally cannot encode a gap | **PASS — 5/5 held-out**: WAKE recovers the exact gapped detector from labels, the parameter generalises *string → spec* (a set of `(lag,bit)` literals), every held-out pattern is full-domain-exact, cross-seed 1.000, scramble ≈ 0.50, recovering the canonical detector |
| **M3b — threshold counting** (`#occ(P) ≥ t`) | a **second abstraction axis**: the *aggregation* changes from parity to a count-threshold, parameterised by `t` | **PASS — 4/4 held-out**: a parity-only grammar provably cannot solve `#11≥2`; extending the aggregation lets WAKE induce a `(detector, threshold)` template and bind unseen `(P,t)` — cross-seed 1.000, scramble ≈ 0.50 (milestone M2: a second induced axis) |
| **M3c — count equality** (`#0 == #1`, the counting essence of `aⁿbⁿ`) | a two-sided count equality `sum == k` — outside the detector + {parity, threshold} span | **UNSAT, honestly** — no `(spec, aggregation)` in the grammar is a full-domain-exact solver; WAKE returns nothing. The wall **relocates** here |

**Findings:**
- **Not a per-family manual move — within a paradigm.** The *same* `WAKE → SLEEP → BIND` machinery
  generalises across families when they lie in the **detector + aggregation** span: M3a by enriching the
  **parameter** (contiguous string → arbitrary literal-spec), M3b by enriching the **aggregation**
  (parity → threshold). In both, the abstraction is *induced and bound*, not re-hand-built, and every
  held-out member passes the full-domain-exact + cross-seed + scramble gate.
- **The boundary is real and precisely located.** M3c (`#0==#1`) is **honestly UNSAT**: count equality
  is not a fixed-window parity/threshold, so the detector paradigm does not reach it. The mechanism does
  not fake a solution — it returns nothing, exactly as a self-honest system should.
- **So the honest answer is nuanced, not a yes/no.** Parameterised abstraction is a **general mechanism
  *within* a computational paradigm**, spanning multiple parameter and aggregation axes — but crossing
  into a *genuinely new* paradigm (counting/balance) still needs a **new primitive**. That is precisely
  the §44 *invention* move (grow the generative space), now relocated one level up: the open frontier is
  **inducing the new aggregation/primitive itself**, not the parameter within a fixed one.

**Significance + honest scope:** M3 settles the §50 open question in the only honest way — *partially,
with a sharp edge*. The "different course" abstraction mechanism is **not** brittle-per-family inside the
detector+aggregation paradigm (a real result: it transfers across gap-structure and across the
parity↔threshold axis with no new template), **and** it has a **named, falsifiable boundary** at
count-equality where a new primitive is required. The Path B ladder is therefore: ✅ M1 cracked
(`wake_lgg.py`), ✅ M2/M3 multi-axis generalisation within the paradigm (`m3_different_family.py`), and
the next genuine frontier — **autonomous discovery of a new primitive/aggregation when the paradigm runs
out** (M3c's `count==k`, then stateful/counting languages) — is exactly where the bet now rests. No
fake success: two families transferred, one honestly did not, and we know precisely why.

---

## 52. Path B — crossing the relocated wall: invent the missing AGGREGATION (`invent_agg.py`)

§51 left an honest boundary: `#0==#1` (count **equality**) was UNSAT because no
detector + {parity, threshold} solver exists. That is the §44 situation — *the library
breaks* — lifted one level: the missing piece is now an **aggregation**, not a detector
parameter. This step applies the §44 **invention** move to the aggregation axis: when the
known aggregations `{cnt2, ge_k}` fail, search a **generative** space of count-predicates
`compare(count, k)` for `compare ∈ {ge, le, eq, ne}` and *invent* the missing comparator —
then fold it into a parameterised template and generalise.

| step | result |
|---|---|
| **(A) invent from failure** | `{cnt2, ge_k}` is UNSAT on `#0==#1`; searching the generative comparator space **invents `eq`** → solver `count(detector) eq 6`, **full-domain-exact**, cross-seed 1.000, balanced-scramble 0.52, a genuinely new form |
| **(B) generalise the invented comparator** | fold into a template `detect(spec) op k` parameterised by `(spec, op, k)`; **4/4 held-out** count-comparison tasks (`#1 ne 6`, `#0 eq 4`, `#10 le 1`, `#11 ge 2`) bound from labels, all full-domain-exact, cross-seed 1.000, scramble ≈ 0.51 |
| **(C) the next honest boundary** | **Dyck-1** balanced parentheses (`0='(' +1`, `1=')' −1`; balanced iff total `==0` **and** every prefix sum `≥0`) stays **UNSAT even with the invented comparators** |

**Findings:**
- **The invention move lifts to the aggregation axis.** A failure that was a hard boundary in
  §51 is crossed by *inventing the missing comparator from a generative space* and binding it —
  the same self-honest, example-driven discovery as §44, now over aggregations. The invented
  `eq` is validated full-domain-exact (a coincidence cannot pass), not just split-lucky.
- **It generalises, honestly — including equivalent re-parameterisations.** Held-out
  count-comparison tasks are all solved; notably the system sometimes finds an *equivalent* exact
  program rather than the textually-intended one (e.g. `#1 ≤ 3` solved as `#0 ≥ 9`, flagged
  `canonical=False`) — which is correct (it is the same function, proven over all 2¹² inputs) and
  a fair reminder that the gold standard is full-domain equivalence, not surface form.
- **The next wall is precisely located: stateful computation.** Dyck-1's `total==0` part *is*
  `#0==#1` (now solvable via the invented `eq`), but the **prefix-`≥0`** condition is not a
  function of any global count — it needs a running **prefix-min / counter** over the stream. No
  fixed-window detector + count comparison can express it, so it is honestly UNSAT. The boundary is
  the *stateful prefix condition*, not the equality.

**Significance + honest scope:** the **hierarchy of frontiers** the project has tracked since §43
extends cleanly and self-honestly — fixed library → invented detector primitives (§44) → recursive
synthesis (§45) → induced parameterised detectors (§50) → multi-axis generalisation (§51) → **invented
aggregations (§52)** — each level crossed by discovery within a generative space, each validated by
full-domain exactness + cross-seed + scramble, each then exposing the next wall. The **standing honest
caveat** is unchanged and now sharply scoped: the *generative space itself* (here, the comparator set
`{ge,le,eq,ne}`) is hand-provided, exactly as §44's predicate DSL was; the deepest open problem remains
**growing the generative space autonomously**. And the next concrete target is no longer a pattern or a
count-predicate but **stateful computation** — a running counter / automaton (Dyck, `aⁿbⁿ`,
prefix-balance) — which connects back to §43's automaton and the latch/counter mechanisms of cycles
9–13. That is where the bit-native "learn-what-to-compute" bet now genuinely rests.

---

## 53. Path B — the STATEFUL frontier: invent a running counter (`stateful.py`)

§52 located the next wall precisely: Dyck-1 balanced parentheses stayed UNSAT even with the
invented count-comparators, because *"every prefix sum ≥ 0"* is not a function of any **global**
count — it needs a running **prefix-min**, i.e. **state** carried along the stream (the cycles 9–13
mechanism: latch / running-XOR / mod-m counter, now needed for *program induction*). This step
invents a stateful primitive and runs the same `WAKE → SLEEP → BIND` machinery over it.

- **primitive:** a signed running counter `bal[i] = bal[i-1] + step(s[i])` with `step: 0→+1, 1→−1`;
- **readouts:** `final = bal[-1]`, `min` = min prefix, `max` = max prefix;
- **atom:** `compare(readout, k)` reusing the §52 comparator DSL; a feature is one atom **or** a
  conjunction of two (Dyck needs *balanced* AND *never-negative*).

| step | result |
|---|---|
| **(A) Dyck without state** | UNSAT in the §52 detector+comparator grammar (a global count cannot see the prefix condition) — boundary holds |
| **(B) invent the counter** | WAKE (readout/op/k **not** supplied) finds Dyck = **`final le 0 AND min ge 0`**, **full-domain-exact**, cross-seed 1.000, balanced-scramble 0.50 |
| **(C) generalise** | leave-one-out over a single-counter family — curriculum `{min≥0, final==0, max≤2}`, **4/4 held-out** (`final≥1`, `max≥3`, `min≤−2`, `final≤0`) recovered from labels, all full-domain-exact, cross-seed 1.000 |
| **(D) next boundary** | **palindrome** `s == reverse(s)` is UNSAT in **both** the detector+comparator *and* the single-counter grammar |

**Findings:**
- **The hierarchy of frontiers extends into STATE.** A running counter — the cycles 9–13 mechanism,
  now *invented from a program-synthesis failure* — crosses §52's wall: Dyck-1 is solved
  full-domain-exact, scramble-clean, cross-seed. The solver found is `final le 0 AND min ge 0`, an
  *equivalent* exact form of `final == 0 AND min ≥ 0` (under `min ≥ 0`, `final ≤ 0` forces
  `final = 0`) — proven over all 2¹² inputs, an honest reminder that full-domain equivalence, not
  surface form, is the bar.
- **It generalises across the single-counter family.** Held-out `(readout, comparator, threshold)`
  tasks are all bound from labels and full-domain-exact — the stateful abstraction is parameterised
  and reused, not re-hand-built per task.
- **The next wall is, again, precisely located: non-local / multi-counter structure.** Palindrome
  compares position `i` to `L-1-i` — a two-ended relation a single left-to-right counter cannot
  carry — so it is honestly UNSAT in both grammars. The frontier relocates to **two-way / stack /
  multi-counter** computation (palindrome, `aⁿbⁿcⁿ`).

**Significance + honest scope:** the Path B ladder now reads §43 fixed library → §44 invented detector
primitives → §45 recursive synthesis → §50 induced parameterised detectors → §51 multi-axis
generalisation → §52 invented aggregations → **§53 invented stateful counter** — six honest levels, each
crossed by discovery within a generative space, each validated by full-domain exactness + cross-seed +
scramble, each exposing the next wall. The **standing caveat persists and is identical at every level**:
the generative space itself (here: the step map `{0→+1,1→−1}`, the readout set `{final,min,max}`, and
depth-2 conjunction) is **hand-provided** — autonomous growth of the space remains the deepest open
problem. The honest headline is *not* "general intelligence below language"; it is a **self-honest,
example-driven, full-domain-verified mechanism for inducing and reusing bit-native computations that
climbs a real ladder of computational power** — detectors → counts → comparisons → state — with each
boundary found, named, and crossed, and the next (non-local / stack-structured computation) now squarely
in view. That is a concrete, falsifiable foundation for the "different course" bet, not a claim that the
bet is won.

---

## 54. Path B — a REAL model on REAL data: induce the representation, mix to predict, scale (`real_test.py`, `real_mix.py`, `real_scale.py`)

§43–47 and §50–53 are mechanism on synthetic `gen()` bit-tasks (solved to full-domain-exact 1.000). Mechanism is
not the goal; the goal is a real bit-native model. This step is the honest real test: run Path B's
induction — *search bit-native computations and keep the ones that predict, by held-out bits/bit* — on
**real** streams (English text, and **E. coli DNA in native 2-bit**, the purest real bit stream), and
unify it with Path A's predictor. Metric throughout = held-out / online **bits/bit** (cross-entropy =
compression; raw 1.0000, lower better), externally referenced against gzip on the same stream; causal,
stdlib-only.

**(a) Induction on real data (`real_test.py`).** Greedy forward selection over a computation pool
(simple `{lag, mod}` vs `{… + running counts + detectors}`), held-out bits/bit. The induction
**discovers real structure** — it selects `mod 8` first on text (*rediscovers the byte from raw bits*),
`lag 3/6` on DNA (codon scale). Path B's own primitives (popcount `bucket`, pattern `det`) **are
selected and lower bits/bit** (text 0.506→0.426). Honest limit: a single sparse conjunction-table beats
order-0 but trails gzip on text (0.426 vs 0.380); on DNA it already beats gzip (0.974 vs 0.995).

**(b) The unification — induce + mix (`real_mix.py`).** Path B induces *what to compute* (the predictive
**period/unit** found by a quick scan + counter/detector contexts); Path A **mixes** them online
(logistic context mixing, the lpaq idea). One engine (the framing's M6: compression = prediction =
program-finding). It **beats gzip on both** real streams. On DNA — where the 8-bit byte assumption is
*wrong* — inducing the right unit (codon, p=6) **plus reverse-complement contexts** (base-complement is
a bit-flip in 2-bit DNA, so restriction-site RC-palindromes are the §53 palindrome structure on real
data) beats the byte model *and* gzip.

**(c) Scaling (`real_scale.py`).** Rebuilt with the project's proven levers — **integer rolling keys**
(no per-bit slicing; cf. `mixfast`), a **stretch LUT** (PAQ), an **SSE/APM** recalibration stage, and a
revcomp table — ~4× faster, multi-MB tractable. The scaling law holds:

| stream | 0.5 MB | 1 MB | 2 MB | 4 MB | gzip |
|---|---|---|---|---|---|
| **text** (corpus_big, whole) | 0.2462 | 0.2417 | 0.2403 | **0.2328** | ~0.36 |
| **text** +SSE+order8 (last-20%) | — | 0.235 | 0.229 | **0.2167** | ~0.36 |
| **text** (homogeneous Shakespeare, whole) | — | 0.2565 | 0.2510 | **0.2431** | ~0.377 |
| **DNA** (ecoli, +revcomp+SSE, whole) | — | 0.9701 | 0.9709 | **0.9658** | ~0.992 |

**Findings:**
- **It is a real, scalable model on real data.** Held-out bits/bit falls monotonically with data on
  homogeneous text (0.2565→0.2431, still dropping), reaching **0.217 (mature) on corpus_big at 4 MB —
  lpaq1 / the §41 enwik8-headline band (0.209)** — and DNA reaches **≈1.91 bits/base** (0.9534 mature),
  each beating gzip by a widening margin.
- **The induction is real and interpretable at every scale:** it rediscovers the **byte** (`p=8`) on
  text and the **codon** (`p=6`) on DNA from raw bits, and its counter/detector/revcomp primitives —
  the very computations the synthetic ladder learned to induce — measurably help, *most* on DNA where
  the representation must be discovered (the byte model is wrong there).
- **Honest scope.** Not SOTA (cmix ~0.15 on text). On *text* the byte unit is already right, so the
  induced engine ties the strong byte model (both beat gzip) — induction's clear *win* is on DNA. A 6 MB
  `corpus_big` point regressed: the documented **heterogeneous-corpus artifact** (the homogeneous
  single-work curve is cleanly monotone, confirming it). And the pure-Python per-bit loop caps practical
  scale at ~6–10 MB.

**Significance:** the "different course" is no longer only a synthetic-task mechanism — it is a **real
bit-native predictive model that scales on real data and beats gzip on two very different streams (text
and genome) by inducing what to compute**, unified with Path A into one engine. The next genuine scaling
lever is the **Rust core (`blmrs`)** — port this *induced-representation* engine there to reach
enwik8 / full-genome (100 MB) scale, exactly as §39–41 did for the byte model, now general.

---

## 55. Native DNA at full-genome scale — the Rust port (`blmrs/src/bin/dna.rs`)

§49 reached the DNA specialist band (E. coli 1.908, human chr21 1.616 bits/base) with `dna.py` — a
2-bit-base, codon-phase, reverse-complement-aware context mixer — but in pure Python, so it could not
run whole chromosomes at speed. This is the §54 "next lever" executed: a faithful native port into the
Rust core, using `strong.rs`-style **bounded-RAM flat open-addressing tables** (multiplicative hash +
8-bit checksum tag) for the base-history orders, with the forward and reverse-complement match models
ported verbatim. The byte-oriented `strong.rs` is left untouched; `dna.rs` is the period-/unit-correct
engine (the unit is the 2-bit base, the period is the codon-3 reading frame, inverted repeats via
reverse-complement — none of the 8-bit byte assumptions).

| genome | bases | `dna.py` (exact dict) | **`blmrs-dna` (Rust)** | time |
|---|---|---|---|---|
| E. coli — 200 K slice (cross-check) | 200 000 | 1.9520 | **1.9520** | 0.4 s |
| **E. coli — full** | 4 641 652 | 1.908 (§49) | **1.9081** | **8.1 s** |
| **Human chr21 — full** | 40 088 616 | 1.616 (§49) | **1.6255** | **67.6 s** |

**Findings:**
- **Verified, then scaled.** The port is **bit-exact** against `dna.py` at small scale (1.9520 =
  1.9520, where the flat table holds no collisions = the exact dict), and reproduces §49's
  specialist-band numbers **on the full genomes**: E. coli 1.9081 and human chr21 1.6255 (the latter
  within the flat-collision delta of the Python 1.616). A whole human chromosome — **40 M bases / 80 M
  bit-predictions — in 68 s** (~0.59 Mbase/s), which `dna.py` could not do (it would take ~11 min).
- **The native lever is bounded-memory-at-scale, as in §39–41.** Flat hashed tables (obits 23–24) give
  fixed RAM with near-exact quality; the genome-scale run is what the Rust core exists for.
- **It confirms the representation thesis on real genomes at scale.** The win over a byte model is the
  *unit/period* (2-bit base + codon-3) and *reverse-complement* — the bit-native "induce/choose the
  right representation" lesson, now competitive with domain specialists across a 9× size range
  (E. coli → human chr21), off a from-scratch English compressor's machinery.

**Significance + honest scope:** the bit-native core is now **natively retargetable and scalable** —
the same project, two engines (`strong.rs` for byte streams at enwik8 scale, `dna.rs` for genomes at
chromosome scale), each unit-correct, each bounded-RAM. Honest scope: `dna.rs` is a *faithful* port of
`dna.py`'s hand-built DNA representation, not an *auto-induced* one — wiring the period/unit *discovery*
(§52–54, `real_scale.py`) into the native engine so it picks the unit itself is the next step. And text
at full enwik8 scale via the induced engine (vs the already-native `strong.rs`) remains to be run.

---

## 56. The period-DISCOVERING native engine — induction at scale (`blmrs/src/bin/induced.rs`)

§55's honest gap was that `dna.rs` is a *hand-built* DNA representation. This closes it: the period/unit
*discovery* (§52–54, `real_scale.py`) is wired into the native core. `induced.rs` runs a quick held-out
**period scan** to discover the predictive unit `p` from the data, then runs unit-aligned online logistic
context mixing at `p` (integer rolling keys + `strong.rs`-style bounded-RAM flat tables + the 33-knot
APM). It is **not told the unit** — it discovers the byte on text and the codon on DNA.

**Comprehensive test battery (all pass):**

| test | result |
|---|---|
| period discovery | English text → **p=8** (byte); source code → **p=8**; 2-bit DNA → **p=6** (codon) — auto-discovered |
| correctness vs Python (`real_scale.py`, same params) | text Δ**0.0005** (0.2625 vs 0.2620); DNA Δ**0.0002** (0.9769 vs 0.9767) — matches within stretch-LUT/float noise |
| **causality** (future-bit-flip) | **PASS** — both streams processed in full; flipping a bit *after* the checkpoint leaves the prefix cost bit-identical (no look-ahead) |
| determinism | **PASS** — identical output across runs |
| edge case (5-byte input) | handled, no panic |
| scale + gzip (text 2 MB) | **0.2358** bits/bit vs gzip 0.3607 (+0.125) |
| **DNA full E. coli genome** | auto-p=6 → **0.9564 bits/bit = 1.913 bits/base** (last-20%) vs gzip 1.984 |

**Findings:**
- **The representation is induced natively, at scale.** The engine discovers the unit (byte / codon)
  from a held-out scan and predicts unit-aligned — verified against the Python reference to ≤0.0005,
  causal (future-bit-flip), deterministic, and beating gzip on both text and genome.
- **Discovery ≈ hand-building, without the priors.** On the full E. coli genome the *induced* engine
  reaches **1.913 bits/base by discovering the codon (p=6) alone** — essentially the hand-built `dna.rs`
  specialist's **1.908**, but with **no codon/reverse-complement prior supplied**. The "learn what to
  compute / induce the representation" thesis, native and at genome scale.
- **A real, named scan subtlety (fixed).** The period scan must use a **small** window: on a large
  window long periods win by *context length* (more conditioning bits), not true periodicity — on the
  near-random genome that pushed the pick to p=10. An 80 k-bit scan (matching `real_scale.py`) restores
  the true unit (codon p=6); reported, not hidden.

**Significance + honest scope:** the bit-native core now *induces* its representation in the native
engine — `induced.rs` is general (one engine, period discovered per stream) and validated by a full
test battery. Honest scope: it is the unit-discovery + mixer core, **without** `dna.rs`'s
reverse-complement/match models, so on DNA it sits just shy of the tuned specialist (1.913 vs 1.908) and
on text just shy of `strong.rs`'s match/high-order machinery; folding those into the induced engine, and
running it at full enwik8 / whole-chromosome scale, is the remaining work. The thesis stands
empirically: **the right unit can be discovered, not assumed — natively, causally, at scale.**

---

## 57. Folding the match + reverse-complement models into the induced engine, at scale (`induced.rs`)

§56's remaining work was to fold `strong.rs`/`dna.rs`'s repeat models into the induced engine and run at
full scale. Done: a **forward match** whose granularity *follows the discovered unit* (byte symbols for
text p=8; 2-bit **bases** for DNA p=6), and — when the discovered unit is base-like (DNA mode) — a
**reverse-complement match** for inverted repeats, sharing the forward base table. Still causal; the
folded models are validated by the same future-bit-flip test.

| stream | engine | bits/bit | bits/base | reference |
|---|---|---|---|---|
| **E. coli — full** (4.64 M bases) | induced: discover p=6 → base-match + RC | 0.9547 / **0.9394** (whole/last-20%) | 1.909 / **1.879** | `dna.rs` specialist **1.908 / 1.876** |
| **text — full corpus_big** (11 MB) | induced: discover p=8 → byte-match | 0.2403 / **0.2382** | — | gzip 0.374 |

**Findings:**
- **Discovery now equals hand-building on a clean genome.** On the full E. coli genome the induced
  engine reaches **1.909 / 1.879 bits/base — matching the hand-built `dna.rs` specialist (1.908 / 1.876)**
  — by *discovering* the codon unit (p=6), routing to base-granular match + reverse-complement, with **no
  DNA prior supplied**. The match granularity auto-follows the discovered unit; RC engages only in DNA
  mode (it is inactive and harmless on text). Both folded models pass the future-bit-flip causality test.
- **Scales to 11 MB text**, beating gzip 0.374 → 0.24, still discovering the byte.
- **An honest limitation, found at scale (human chr21).** Unlike E. coli (~88 % coding → a clear codon
  period), human chr21 is **mostly non-coding** (codon signal only in ~2 % exons) and *starts* with a
  long low-complexity/telomere region — so the prefix period scan finds no robust base-like unit (it
  picks p=8/10 by context-length) and the engine runs in generic mode (≈1.7 bits/base) rather than DNA
  mode. A middle-window scan did not help (it broke E. coli) — the real issue is that **weak-periodicity
  streams defeat the simple scan**; robust modality/period detection (autocorrelation, an entropy-gated
  representative window, or a dedicated base-structure test) is the honest open problem for the discovery
  step. `dna.rs` (unit hand-given) still gets chr21 to 1.6255; matching that via *discovery* needs the
  better detector.

**Significance + honest scope:** the induced native engine now folds in the repeat models and, on a
clean genome, **discovery matches the hand-built specialist** — strong evidence for "induce the
representation, don't assume it." The named gap is **robust unit discovery on weak-periodicity / mostly
non-coding streams** (human DNA), where the simple prefix scan is not enough. No fake success: E. coli
and 11 MB text validate end-to-end, causally; chr21's discovery limitation is reported, not hidden.

_(Detector update: the scan now **skips low-complexity windows** — a telomere / N-run scores near 0,
trivially compressible — and uses the first representative window, so a non-representative prefix no
longer misleads it. This preserves text=8 / E. coli=6, but **human chr21 still resolves to p=10**: its
codon signal is genuinely too weak (mostly non-coding) for a prediction scan to find. Tried prefix,
middle, multi-window-vote, and representative-window scans; none robustly routes human DNA to base-mode.
The honest conclusion stands — robust unit discovery on weak-periodicity streams (autocorrelation, a
base-structure test) is the open problem, and `dna.rs` with the unit hand-given still reaches 1.6255.)_

---

## 58. The loop closes — English in, bit-native generation, English out (`talk.py`)

`BIT_NATIVE_INTELLIGENCE_FRAMING.md` proposes a **bit-native intelligence core with dumb modality
adapters**: language is an optional I/O codec, prediction is the substrate. This realises that loop
end-to-end and makes it *visible*: type English → a **dumb adapter** (UTF-8 bytes ↔ bits, no semantics)
→ the **bit-native core** (a byte-aware next-bit predictor, the same family that compresses enwik8),
trained on real English, **generates a response by autoregressively sampling the next bit** from its own
prediction, one bit at a time → the adapter decodes the bits back to English → you read it.

```
[you]      "It is a truth universally acknowledged that "
[adapter]  English -> 44 bytes -> 352 bits
[core]     sampling 220 next-bytes (1760 bits), one bit at a time ...
[adapter]  bits -> bytes -> English
[core says] It is a truth universally acknowledged that | there was all this morning at her mother,
            ... I can remember ... And even a very day after a short survey" her again, and which she
            countenance, that he should not ...
```

**Findings:**
- **The architecture's I/O loop is real, not a diagram.** Trained on *Pride & Prejudice* (real English),
  the core produces recognisable English continuations — real words, spacing, punctuation, the corpus's
  voice — purely by predicting and sampling bits, with language living only in the dumb adapter. The same
  next-bit prediction that *compresses* a stream *generates* one (compression = prediction = generation).
- **Honest scope (stated plainly).** This is a **byte-level statistical** language model (a high-order
  n-gram / small char-LM), **not** an instruction-tuned LLM. It does **not reason or answer** — it
  *continues* the prompt in the corpus's style, with occasional non-text bytes at low temperature. The
  value is architectural: it demonstrates *intelligence-as-prediction below language*, the framing's
  central thesis, as a runnable loop — not a capability claim.
- **It ties the whole project together.** The bit-native predictor — toy benches (§1–42), Path B's
  induced computations (§43–57), the native engines — is, at bottom, a next-bit model; here that model
  *speaks*, through a dumb adapter, exactly as the framing proposed. Scaling the core (the strong /
  induced engines) and training on dialogue would make the continuation more answer-like; the loop itself
  is now closed and interactive (`python talk.py "your prompt"`).

---

## 59. From continuation to ANSWERING — strong core + Q/A + a real dataset (`chat.py`, `make_chat_data.py`)

§58 *continued* a prompt; this *answers* it, by two additions that change the data and the wrapper, not
the engine. **(#1) A stronger generative core:** online logistic context mixing (byte-aware orders, the
enwik8 family) **plus a byte match model** — when the recent context matches a span seen in training, it
*recalls and copies* the continuation (attention-like copy, in bits). **(#2) A Q/A wrapper:** your input
is formatted `Q: …\nA:` so the model falls into answer-shape, and generation **stops at the next `Q:`**
for a bounded reply. **(dataset) `make_chat_data.py`** builds a real, correct corpus — exact arithmetic +
curated true facts (capitals, science, opposites, counting), 4 643 Q/A pairs — the "serious data" lever.

```
Q: What is the capital of France?  -> A: Paris      Q: What is 7 + 8?     -> A: 15
Q: How many legs does a spider have? -> A: 8         Q: What is 25 + 17?  -> A: 42
Q: What is the opposite of hot?    -> A: cold        Q: symbol for water? -> A: H2O
```

**Findings:**
- **It answers — correctly — on what it was taught.** Capitals, arithmetic in range, opposites,
  counting, simple science all come back right, by recall: the long match locks onto the question's tail
  and copies the answer that followed it in training; the orders supply format and stop. Same next-bit
  engine as the compressor, run in sample mode — recall is just the match model (compression =
  prediction = generation = recall).
- **A real bug, found and fixed (kept honest).** A *single* learned mixer weight for the match went
  **negative** — on short Q/A data many 16-byte contexts precede different bytes, so the mixer learned to
  distrust the match. The fix is to trust the match only once it is a **long** span (gate on match
  length) and steer toward the recalled bit then, rather than mixing it by one global weight.
- **Honest scope, stated plainly.** This is **recall + format**, not reasoning. Trained questions →
  correct; **novel** questions → *answer-shaped but wrong/made-up* (`"capital of Mars?" → "10"`,
  `"123 + 456?" → "30"`) — no generalisation, no arithmetic it didn't memorise, no facts it never saw.
  The value is the architecture: a bit-native predictor, through a dumb adapter, *answers* — and gets
  better strictly by (a) **more/cleaner data** (`make_chat_data.py` is the lever) and (b) a **bigger
  core** (the strong/induced engines). It does not pretend to know what it was not shown.

**Significance:** the framing-doc loop is now not just "speaks" (§58) but "**answers**", with real recall
over a real dataset — `python chat.py "your question"`. Everything stays bit-level: prediction is the
substrate, language is the dumb adapter, recall is the match model. The path to "actually smart" is the
honest one — scale the core and the data — not a change of mechanism.

---

## 60. Recall vs GENERATION — can it produce correct *new* data? (addition, held-out) (`add.py`)

§59 was honest that chat is **recall**: delete a fact and it cannot answer; it memorised the times-table
and fails `271+314`. The sharp question: can the bit-native core produce **correct outputs for inputs it
never saw** — i.e. learn a *computation* that generalises, not a table? Addition is the microscope, with
**held-out accuracy** as the only metric (the discipline §59's chat side lacked).

Representation is the lever: addition is non-local as decimal text, but in **binary, LSB-first** it is a
1-bit-state transducer — `s_i = f_out(a_i,b_i,carry)`, `carry' = f_upd(a_i,b_i,carry)`. The carry is a
**hidden** recurrent state; it is **never supplied**. We *induce* both 3-input boolean functions by
searching the 256×256 transducer space for the one that explains the training sums — Path B's "learn the
recurrent computation" (cf. §51), applied to a generative task.

| approach | train acc | **held-out acc (3000 unseen sums)** |
|---|---|---|
| **memoriser** (key on `(a,b)`, chat-style recall) | 100 % | **0 %** |
| **induced computation** (1-bit-state transducer) | 100 % | **100 %** |

**Findings:**
- **It generates correct new data — by induced computation.** From **60** example sums the search finds
  **exactly one** transducer reproducing them: `out=XOR3` (150), `carry=MAJ3` (232) — *the full adder*.
  The hidden carry was **induced, not given**. It then computes sums it never saw (`3131+2996=6127`, …)
  at **100 % held-out**, while the memoriser is at **0 %**. That is the honest, measured **yes** to
  "can it generate new data" — wherever it induces the *rule*, not the *table*.
- **It is the representation, again.** The same numbers as decimal text → memorised; as binary
  bit-columns + a 1-bit state → the rule is inducible and generalises perfectly. "The lever is the
  representation" (the project's spine) is what separates recall from generation.
- **Knowledge stays separable — and that is fine.** The core learned a *function* (the adder), not
  stored answers; facts can live in an external addressable store (the match model is exactly that).
  A coherent **non-LLM** shape: *induce computations + retrieve knowledge + generate*, rather than
  memorise everything in weights.

**Significance + honest scope:** this closes the loop on the project's central claim with a number —
the bit-native core **generates correct novel data when it learns a generalising computation**
(held-out 100 %), and only recalls when it memorises (held-out 0 %). The honest ceiling: addition is a
*tiny* rule (1-bit state) induced by *brute* search over a *small* space; complex computations hit the
Path B search wall (§45), and discovering the right representation automatically is still open. But the
principle — recall ≠ generation, and induced computation *is* generation — is now proven end-to-end, and
it points the way: a growing **library of induced computations** composed with separable knowledge.

---

## 61. A reliable, verifiable STRUCTURED RESPONSE — and where the wall really is (`tool.py`)

The honest target is not free prose (a high, scale-bound wall) but a **proper structured response you can
rely on** — the kind that, in a future view, underlies tool-use. Stripped of that framing: can the core
turn a request into a **valid tool call that, executed, is correct**, for requests it never saw? Three
guarantees, all decidable: **valid by construction** (the response is emitted under a grammar
`op(a, b)`), **correct by induced computation** (executed by the §60 bit-native transducer — add and a
subtractor, each induced from 60 disjoint examples, so they *compute*), and **verified** (validity and
correctness measured on held-out requests). Division of labour as the framing doc prescribes: the dumb
adapter tokenises, the core **routes** the operation and **computes**, the grammar guarantees a
well-formed response.

| over 2 000 held-out requests | result |
|---|---|
| structured response **valid** (parses the grammar) | **100 %** |
| executed answer **correct** | **100 %** |
| chat-style **memoriser** (request → answer recall) | **0 %** |

**Findings:**
- **A response you can rely on is reachable — in the structured/verifiable regime.** The core emits a
  valid call and the correct answer for numbers it never saw (memoriser: 0 %), because the computation is
  *induced*, not stored, and the form is *grammar-guaranteed*. This is the trustworthy shape: not "hope
  the prose is right" but "the response cannot be malformed and the value is checkable."
- **The wall is words, not compute — shown, not asserted.** Before a fix, valid was 100 % but correct
  was **82 %**, and *every* error was one phrasing: *"subtract X from Y"* (= Y−X), which the parser got
  backwards. The computation was flawless; the failure was **language understanding**. Fixing it meant
  encoding a *words fact* (`"from" ⇒ swap the operands`) — a compute fix would not have helped. The
  residual difficulty lives exactly where intuition says: in mapping varied language to intent.
- **Honest scope.** The generalisation that *works* (held-out numbers, 100 %) comes from the induced
  computation + grammar + copy of the request's fields. The part that is **bounded** is the
  language-understanding (keyword routing + known phrasings, hand-given here); widening it to open
  language is the words-wall — solvable only by *learning* many phrasings (scale) or inducing the
  parse, which is the open frontier.

**Significance:** "a proper response you can rely on" is not blocked by computation — that generalises
cheaply (§60) and is verifiable. It is gated by **language understanding**, precisely. So the productive
path is the structured/verifiable regime — valid by grammar, correct by induced rule, knowledge
separable — where the bit-native core can be *reliable by construction*, and to push it, attack the
bounded-language-understanding frontier directly rather than chasing free-prose fluency.

---

## 62. Engine / language / knowledge — provably separable, held-out (`separation.py`, `separation_data.py`)

A sharper question than "predict the next bit": are **computation**, the **language** a query is posed
in, and the **knowledge** (facts read from text) *separable, independently-varying* parts of the core —
or are they entangled? The bet is a three-way separation: an **engine** (computation), a **language**
(the surface), and a **knowledge** store (facts), that vary independently. `separation.py` builds this
as a falsifiable, staged experiment over the §59 byte-native core and the §60 induced adder, with the
controls an adversarial panel demanded (designed by one research fan-out, audited by a separate
adversarial fan-out; both in the session log).

The three artifacts: **(E)** the *engine* = the induced full-adder transducer `(out=XOR3, upd=MAJ3)`
plus `chat.Core`'s fixed predictor code and frozen hyperparameters — a **corpus-independent** object;
**(L)** the *language* = the grammar/Q&A shape the store acquires from a **fact-free** English corpus;
**(K)** the *knowledge* = the append-only byte store (`corpus_bytes`/`mtab`/`tables`), filled **only**
by reading. We fingerprint E and K at every stage S0→S4 (arrival → learn English → read an English book
→ meet a cipher-only fact → learn to read the cipher).

| claim | result (held-out / staged) |
|---|---|
| **C1** math is *induced*, not recalled | adder `(150,232)` forced by 60 disjoint sums; **100 %** on 3 000 unseen pairs; memoriser **0 %** |
| **C2** holds no facts it has not read | with English loaded, every real-world-fact query **abstains** (0/5 correct, **5/5 abstain**) |
| **C3** language ≠ knowledge | English readability **8.00 → 0.81 bits/byte** (learned) yet **no** fact becomes answerable |
| **C4** engine/store **separation** | engine fingerprint identical S0..S4 **and sabotage-sensitive**; store fingerprint changes every stage, corpus **prefix-monotonic** |
| **C5** reading supplies knowledge, gated by language | novel fact read → **commits** (pre-book abstains); cipher-only fact **abstains in English, commits in its own language**; cipher **learned** (4.81 → 0.89 b/B) while never-taught cipher C′ stays **5.96** |

**Findings:**
- **Math is computed; facts are read.** The adder generalises 100 % on numbers it never saw (a *computed*
  rule, memoriser 0 %), while the *same* core abstains on an unread fact (e.g. the capital of France)
  until it reads it — computation and fact-recall are distinct subsystems, not one entangled blob.
- **The engine invariant is a *separation guarantee*, verified by sabotage — not a tautology.** The
  fingerprint is, by construction, a function of the corpus-independent engine only, so its constancy
  across stages is *exactly the claim* that language and facts land in the **store**, never the engine.
  We keep it from being vacuous by showing it **moves** when the math is broken (`upd_fn=233 → different
  hash`), the predictor is patched, or the gate changes — so it genuinely *would* change if who-it-is
  changed. Meanwhile the store fingerprint changes at every stage and the corpus is append-monotonic.
- **Honest abstention, earned not declared.** One fixed rule for *every* query — commit iff the question
  is found **verbatim** in read text (`prime_mlen ≥ MATCH_GATE`) **and** confidence ≥ τ — with no
  per-fact branch. The binding term is verbatim-question recall: a *fluent confabulation* (conf 0.98 but
  `mlen 0`) **abstains**; the *same* query commits only after the book is read. Calibration, not a guard.
- **Knowledge is gated by the language it is written in.** A fact stored only in cipher C abstains to an
  English question but commits when asked **in cipher C** (the model emits cipher bytes that are a
  verbatim substring of what it read; we decode **only to score**, never to answer). And "learning a
  language" is measured honestly as **held-out readability**: reading cipher-C text drops its bits/byte
  4.81 → 0.89, while a never-taught cipher C′ stays at 5.96 — learned to read, not trivially decoded.
- **The boundaries, stated and shown (not hidden).** Recall is **span-level copy**, labelled by
  longest-common-substring: a < 16-byte **paraphrase abstains** (no semantic comprehension). The engine
  does **not translate**: an English question about the cipher fact **stays gated even after** the model
  learns to read that cipher. And the learned "English" is **vocabulary/grammar-bound** — a
  novel-vocabulary control (same grammar, unseen words) stays high (≈ 3.8 b/B vs 0.81), so the claim is
  "learned *this* language," not "reads arbitrary English." Corpora are audited **fact-free** (no answer
  token, no 16-byte probe span) and the demo does **zero file I/O** (it never touches `data/chat.txt`,
  which contains France→Paris and would be a fatal leak).

**What it proves, and does not.** `separation.py` demonstrates a clean three-way separation between a
computation engine, a language, and a knowledge store on a byte-level model: math is induced computation
(100 % held-out, memoriser 0 %); the engine is corpus-independent by construction *and* sabotage-
sensitive, so language and facts provably live in the separable, append-monotonic store; facts are absent
until **read** (audited fact-free, novel entity abstains pre-book / commits post-book, one book leaks no
other fact); retrieval and abstention run through **one fixed, fact-agnostic rule** with no decoder in
the answer path; and "learning a language" is held-out readability (taught cipher C drops, never-taught
C′ does not). It does **not** prove comprehension, grounding, or understanding — fact recall is verbatim
span-copy (a paraphrase abstains), the model does not translate (a cipher fact asked in the other
language stays gated), and the learned language is a small closed vocabulary/grammar. Stated plainly:
**the core computes math that generalises, holds no fact it has not read, and is the same engine
byte-for-byte before and after learning a language — computation, language, and knowledge are separable
axes**, measurable held-out, which is what makes them improvable one at a time.

**Significance:** the §61 result said the wall is *words*. §62 maps that wall precisely by **separating
the axes**: computation generalises for free; reliable answering is recall traceable to what was read,
gated by language; and "understanding" (paraphrase, translation, open vocabulary) is the residual,
explicitly *not* yet crossed. The productive next step is the same one §61 pointed at — widen the bounded
language axis (learn the question→intent map, paraphrase-robustly) — now with a harness that can measure,
honestly and held-out, exactly how much of that axis any change actually buys.

---

## 63. Improving the strong compressor — richer models + tuning, measured (`blmrs/src/bin/strong.rs`)

Past the research phase, the first engineering target is the headline itself: **lower bits/bit**. The
strong core was already lpaq1-class (§ headline). Working in an A/B fork (`strong2`, since promoted),
each change was measured on a held-out real-text proxy (`data/corpus_big.txt`) and kept only if it
helped — a clean ablation, not a guess.

**Changes (each measured at 1 MB, whole-stream bits/bit):**

| step | change | bits/bit | Δ |
|---|---|---|---|
| baseline | byte-orders 0..6, hi {8,12,16,24,32}, 1 sparse, word, 2 match, 1 sel mixer, 4 SSE | 0.221636 | — |
| +order-7 | fill the 6→8 gap | 0.220987 | −0.0006 |
| +mixer-2 | a 2nd context-selected mixer (order-2 selector) + 4-weight final combiner | 0.219541 | −0.0014 |
| +SSE | two more APM stages (word-hash, order-3 contexts) | 0.219314 | −0.0002 |
| +tune | DELTA 0.18→0.08, mixer LRs 0.0013→0.0010 (env-overridable) | 0.218859 | −0.0005 |
| +sparse | 3 sparse models (non-adjacent byte pairs) instead of 1 | 0.218614 | −0.0002 |
| +prev-word | predict a word's bits from the **previous word** (text bigrams) — biggest single win | 0.216905 | −0.0017 |
| +ICM | 5 **indirect** models (orders 2–6): bit-history byte → adaptive StateMap (nonstationary) | 0.216588 | −0.0003¹ |

¹ ICM looks tiny at 1 MB but **scales hard**: −0.0026 at 3 MB and −0.0019 at 11 MB (whole) — it earns its
keep only once contexts accumulate bit-history (see the validation rows). On code it is the dominant win.

**Validation (cumulative engine — all of the above incl. ICM — vs the original baseline, held-out).**
Both engines are *committed, reproducible binaries*: the improved `strong` (HEAD) and the frozen
pre-§63 baseline `strongbase` (= `git show 85efe85~1:.../strong.rs`, built by `cargo build --release`).
Each row states its exact byte-cap and `obits` (memory) — re-running either binary reproduces the cell
to 6 d.p. (the engine is deterministic; no RNG):

| corpus | cap, obits | baseline (`strongbase`) | improved (`strong`) | improvement |
|---|---|---|---|---|
| corpus_big 1 MB | `1000000 23` | 0.221636 | 0.216588 | −2.28 % |
| corpus_big 3 MB | `3000000 24` | 0.218053 | 0.211762 | −2.88 % |
| **corpus_big 11 MB** (whole) | `0 25` | **0.224637** | **0.217011** | **−3.39 %** |
| corpus_big 11 MB (last-20 %) | `0 25` | 0.223195 | 0.214899 | −3.72 % |
| code 0.8 MB | `800000 23` | 0.159457 | 0.150989 | **−5.31 %** |
| b100 1 MB | `1000000 23` | 0.241401 | 0.236463 | −2.04 % |

*(Red-team note: the table previously omitted `obits`, so an auditor who measured 11 MB at `obits 23`
got 0.217759/0.216233 instead — a real reproducibility defect, now fixed by stating the obits. The gain
itself is **robust to memory**: at `obits 23` the 11 MB whole delta is still −3.22 % — smaller because
the ICM/StateMaps have less room — vs −3.39 % at `obits 25`.)*

**Findings:**
- **A real, generalising gain that grows with data.** −3.39 % bits/bit (whole) on the 11 MB text proxy,
  −3.72 % on the warmed-up last-20 %, and it **generalises** — text *and* code, 2.0–5.3 % — so it is no
  corpus-big artefact. The gain **grows with scale** (1 MB −2.28 % → 11 MB −3.39 %): richer models earn
  their keep as the stream lengthens — the opposite of overfitting. (Strict causality / no future-bit
  leakage was independently traced and confirmed in the red-team pass; bits/bit is a valid online loss.)
- **Two kinds of win.** (a) *Direct structure*: the **previous-word model** (−0.0017 at 1 MB) and the
  **second (order-2) mixer** — text-bigram structure written into the address, the project mantra,
  bankable as bits. (b) *Indirect / nonstationary*: the **ICM/StateMap** models — a context's bit-HISTORY
  (recency, not just totals) indexes an adaptive map — which is the lever that **scales** (−0.15 % at
  1 MB but **−1.2 % at 3 MB**, and **−5.3 % cumulative on code**, where repeated structure makes the
  history maximally informative).
- **Honest scope.** (1) **Speed cost:** ~2–3× slower (second full mixer + ICMs + extra models),
  ~0.15 Mbit/s pure-scalar — a roadmap-item-3 (throughput) concern, deferred deliberately. (2) **enwik8
  not re-run:** the 0.209 ladder figure was the *prior* engine; the improved engine should be at least as
  good but that was **not measured here** (enwik8 isn't local), so the ladder stands as a conservative
  floor, not a new claim. (3) Every step kept only on a measured win; `DELTA/ALR/ALRF` are env-overridable
  for one-build sweeps.

**Significance:** the bit-native compressor is improvable by ordinary, honest CM engineering — measure,
keep wins, ablate. Two levers paid off: representation (text-structure models) and **indirect bit-history
StateMaps** (the ICM/ISSE family that separates lpaq1 from paq8), the latter scaling with data. The ICM
here is the *flat-input* form (bit-history → StateMap → mixer input); the remaining headroom is the
*chained* ISSE form (each order refining the previous prediction) plus a richer state machine and ICMs on
the word/high-order contexts — the next push on this axis.

---

## 64. A first probe at the words wall — a LEARNED question→intent router (`intent.py`)

Roadmap item 2 is *capability*, and §61/§62 named its wall: **words** — mapping a varied phrasing to the
right intent. `tool.py` routed with **hand-coded** keywords (`"plus" ⇒ add`). §64 replaces that with a
**learned** map (online multiclass logistic — the compressor's own stretch/squash unit — over binary
features: word tokens, operator symbols, numbers masked to `NUM`, char 3-grams) and measures, honestly,
*how far it generalises* across three template-disjoint tiers, over **8 seeds** (mean, min–max).

> **Honesty note.** A first cut of this section over-claimed and was corrected after an adversarial
> red-team (the project's standing practice): its HARD tier *leaked* verbatim training words, and its
> hand-coded baseline was a *strawman* (whole-word match). Both are fixed below — HARD now contains only
> word forms that are **not** whole training words (asserted in code), and a **fair stem-substring**
> hand-coder is reported alongside. The numbers and the claim are the corrected ones.

| router | train | EASY¹ | HARD² | NOVEL³ |
|---|---|---|---|---|
| hand-coded, whole-word (`tool.py`) | 77.8 | 86.7 | 0.0 | 5.6 |
| hand-coded, **fair stem-substring** | 94.4 | 93.3 | **91.7** | 0.0 |
| learned (words only) | 100 | 100 | 44.8 | 42.4 |
| **learned (words + char n-grams)** | **100** | **100** | **97.9** | 38.2 |
| learned (words + char, stop-words stripped) | 100 | 100 | 96.9 | 42.4 |

(mean % over 8 seeds.)  ¹ EASY = new sentence *structure*, same key words.  ² HARD = unseen word *forms*
(`added`, `subtracting`, `multiplied`) — share stem *substrings*, **no** whole training word (verified).
³ NOVEL = true *synonyms* (`combine`, `deduct`, `scale`) — share no discriminative word with training.

**Findings (corrected, multi-seed):**
- **Routing is learned, and generalises across sentence structure.** EASY = **100 %** on every seed —
  no hand-tuned keyword list, robust to new phrasings built from known words.
- **Char n-grams give a *real* lift on unseen word forms.** On the de-leaked HARD tier, words-only scores
  **44.8 %** but words+char-grams **97.9 %** — sub-word features genuinely bridge `multiplied`→`mul`,
  `subtracting`→`sub`. This is the one solid generalisation result beyond structure.
- **But this is *not* a win over fair hand-coding.** A reasonable **stem-substring** hand-coder also
  solves HARD (**91.7 %**). So the learned router's value is *"no keyword hand-tuning"* (it discovered
  the stems from data), **not** a capability hand-coding lacks. The earlier "generalises where hand-coded
  cannot" claim was false and is retracted.
- **The synonym wall is NOT climbed.** True synonyms (NOVEL) sit at **≈38 %** — at the **33 % three-class
  chance floor** (per-seed range 33–50 %), and stripping function words leaves it at chance (**42 %**), so
  the few above-floor hits were shared *function* words, not meaning. Surface features cannot cross a pure
  semantic gap.

**Significance:** the words wall is a **gradient**, now honestly mapped: sentence *structure* (learned,
solved), *morphology* via sub-word features (learned, solved — though fair hand-coding solves it too),
and *semantics*/synonyms (**open** — at chance). The real next step is the only one that crosses the last
gap: give words **learned distributional meaning** (§62's "reading" — representations the predictor
already forms from a corpus) and route over those. That keeps item 2 tied to the core predictor rather
than bolting on an LLM — and §64's honest value is the *measurement* that says exactly where the wall is.

---

## 65. Crossing the synonym wall with learned MEANING — a mechanism, honestly scoped (`meaning.py`)

§64 left the last tier of the words wall unclimbed: true **synonyms** (`deduct`, `combine`, `scale`)
share no surface feature with the training vocabulary, so a bag-of-features router routes them at chance.
§64's own verdict named the fix — §62's *reading*: let words acquire **distributional meaning** from a
corpus, then route over that. §65 builds it and, per standing practice, **red-teams it** (the file below
was corrected after that pass; the numbers and framing here are the corrected ones).

The method is the classic count-based one: **PPMI co-occurrence vectors** from an **unlabeled** reading
corpus, then a nearest-centroid router whose operation centroids are built **only from the training
words**. If `deduct` was read in subtraction-typed contexts its vector lands near `subtract`/`minus`
(`cos(deduct, sub-centroid) ≈ 0.94` vs `0.0` to add), so it routes to sub — though the router was never
told what `deduct` means.

| over 5 seeds (chance = 33 %) | result |
|---|---|
| meaning router, TRAIN-word probes (sanity) | 100 % |
| **meaning router, NOVEL synonyms (the wall)** | **91.7 %** |
| surface router, NOVEL (word identity, control) | **0 %** — cannot route |
| meaning router, NOVEL, token-**shuffled** corpus (control) | 33 % — class-collapse |

Reading *more* → more meaning (NOVEL 17 % → 33 % → 67 % → 92 % at 8 → 24 → 80 → 240 sentences): it is
**learned**, not hardcoded.

**Findings (post red-team):**
- **The mechanism is real.** Learned distributional vectors place novel synonyms with their operation's
  training words well enough to route them (**92 %**), where surface identity is **0 %** (orthogonal) and
  a token-shuffled corpus is **chance** — so the lift is genuine co-occurrence **structure**, learned by
  reading, not leakage. A smooth cross-contamination curve (the red-team's: 0 %→100, 40 %→53, 100 %→45)
  confirms a real distributional learner, not a brittle pattern-match.
- **But the separation is SUPPLIED by the corpus, not discovered — stated plainly.** The frames place
  each operation in a *disjoint context vocabulary* (larger/bigger vs smaller/fewer vs
  repeatedly/manyfold). Collapse the frames to a *shared* vocabulary and the effect **vanishes** — NOVEL
  *and* TRAIN both drop to 33 %. So the honest claim is "meaning crosses the wall **given a corpus that
  uses the words in separated contexts**" — supervision-by-construction in continuous form, not magic.
- **Honest limits, all shown in-code.** It works only for synonyms that actually **appear** (unlabeled) in
  the reading corpus; only for a **small co-occurrence window** (window ≤ 2 → 100 %, window ≥ 8 → chance,
  as wide windows blur the short frames); and it is **classical PPMI**, a separate method — **not** the
  bit-native predictor and sharing no machinery with it. Real local corpora are too **sparse** for this
  vocabulary (a frequency scan of 17 MB of local text finds `deduct` 0×, `subtract` 1×, `multiply` 3×), so
  no real corpus is read — this is a *mechanism* demo, not a real-data result.

**Significance:** §65 finishes **mapping** the words wall that §64 measured — structure (learned),
morphology (learned), and now semantics/synonyms (**crossable by learned meaning**, demonstrated as a
mechanism with hard controls). The honest residual is **data, not method**: real synonym generalisation
needs a corpus that actually *uses* the vocabulary (a large real corpus does; our local text does not),
and a genuinely *bit-native* version would learn such word meaning **through the predictor itself**
rather than bolting on PPMI — the real next step on item 2, now precisely located.

---

## 66. Predictive variant — prediction substitutes for the count (`bitmeaning.py`)

A tightly-scoped follow-on to §65 (and red-teamed alongside it). §65 crossed the synonym wall by
**counting** (PPMI) and honestly flagged that this was *"classical NLP, not the predictor."* §66 closes
exactly that gap with a **count→predict substitution**: learn the word vectors as the **parameters of a
predictor** — skip-gram with negative sampling (word2vec, Mikolov 2013: a logistic unit + online SGD
predicting neighbours, the **same primitive class** the core's mixer uses, though sharing **none of its
code**) — then route by §65's exact protocol (centroids from training words only).

On the **same** corpus, controls, and scope as §65: NOVEL synonyms route at **100 %** (chance 33 %, 3
seeds), surface **0 %**, shuffled corpus ~chance (decorrelated). The one piece of evidence §65's count
method could not give is an **ablation that turns the predictor off**: with no SGD (`epochs=0`, random
init) NOVEL sits at **34 % (chance)**; trained, **100 %** — so the lift *is* the predictor, not the
representation lying around. (It is also strictly more robust than PPMI: skip-gram windows 1–8 all give
100 %, where §65's PPMI decayed past the frame length.)

**Honest scope (a non-claim, unchanged from §65):** this is a **substitution, not a new capability** — the
same constructed corpus that *supplies* the operation-disjoint separation, the same "synonyms must appear
in the (unlabeled) corpus" limit, and it is **self-contained word2vec** that shares the byte core's
*primitive class* but **no code** with it. The result it adds is narrow and specific: crossing the
synonym wall is **not special to count statistics** — a predictor's learned weights do it too. The
genuinely bit-native step — learning such word meaning *inside the byte predictor itself* — remains the
open work, and is the real next move on item 2.

---

## 67. Item 2 capstone — a dependable answerer, held-out and verified (`answerer.py`)

The pieces built across item 2, wired into **one pipeline** and measured end-to-end (and red-teamed):
`request → ` **learned router** (§64) ` → ` integer args (dumb adapter) ` → ` **induced computation**
(§60) ` → ` **verified** answer. The point is a *doubly-held-out* measurement neither part made alone:
unseen phrasings × unseen numbers, grammar-gated and truth-verified.

- **Computation is INDUCED, and the rule is universal.** add/sub are 1-bit-state transducers induced
  from 60 disjoint sums; the search returns the **unique full adder `(150,232)`** (out = XOR3, upd =
  MAJ3) — exact for all inputs/widths (asserted exhaustively on `[0,256)²`, not a sample). **Multiply is
  COMPOSED** from the induced adder by shift-and-add. Held-out: add/sub/mul **100 %** on 3 000 unseen
  pairs; a memoriser is **0 %**.
- **Routing is LEARNED** (§64's logistic over word/char features), so the answerer generalises to
  **unseen phrasings**; train/test templates are disjoint (HARD shares only function words).
- **The dependable property is precision.** CORRECT-when-committed = **98 % (83–100) across 8 router
  seeds** at confidence TAU = 0.4, and **100 % for every seed at TAU ≥ 0.90** — where a memoriser is 0 %.
  Every committed answer is a grammar-valid call, verified against truth.

| held-out, doubly-unseen | result |
|---|---|
| induced compute (3 000 unseen pairs): add / sub / mul | **100 % / 100 % / 100 %** (memoriser 0 %) |
| end-to-end CORRECT-when-committed (8 seeds, TAU 0.4) | **98 % (83–100)**; 100 % @ TAU ≥ 0.90 |
| coverage at TAU 0.4 | 83 % (abstains on ~17 % of in-scope phrasings) |

**Honest limits (caught by testing-as-we-go, kept — not papered over):**
- **Precision is seed-fragile at low TAU.** One router seed confidently misroutes a fragile HARD
  word-form (`decremented`) at TAU 0.4 (precision 83 %); raising TAU ≥ 0.90 restores 100 % for *every*
  seed, at a coverage cost. The single-seed "100 %" headline would have been an overclaim — the
  multi-seed `mean (min–max)` is the honest figure (matching `intent.py`'s own discipline).
- **A confidence knob does NOT reject out-of-scope synonyms.** NOVEL synonyms ride shared *function*
  words, so they are **committed-and-sometimes-wrong** (≈ chance), **not abstained** — not a clean
  refusal. Extending coverage there is a **meaning** problem (§65/§66), not a threshold to tune. This is
  the honest bridge from the capstone back to the meaning frontier.

**Significance — item 2 is complete.** The tool path is a **dependable answerer on its covered
language**: learned routing (generalises phrasing) + induced computation (generalises numbers, the
*universal* rule) + grammar + verification ⇒ correct-when-committed where recall is 0 %. Its boundary is
drawn honestly: it trades coverage for precision (a TAU knob), and the genuinely open frontier —
out-of-scope synonyms — is a *meaning* problem, exactly where §65/§66 point and where a **bit-native**
meaning (learned inside the predictor, not a bolted-on embedder) is the next move. End-to-end, the words
wall is mapped: **structure and morphology routed, computation induced and verified, synonyms named as
the meaning frontier** — item 2's "grow the tool path into a dependable answerer" delivered, with its
limits measured rather than hidden.

---

## 68. Real-data validation on a HuggingFace dataset (`realtest.py`, wikitext-103)

Before scaling (item 3), the foundation was tested on **real** data — not the local `corpus_big` proxy,
not the synthetic §65/§66 corpora. `realtest.py` fetches real Wikipedia text (`Salesforce/wikitext`,
wikitext-103-raw; ~1.4 MB, via the HF datasets-server, stdlib only — no `datasets`/`pyarrow`) and re-runs
two project claims, reproducibly and **red-teamed** (the meaning verdict below was *corrected* after the
red-team caught a spin).

**(A) Compression — the headline holds on real data.** Online (causal) cross-entropy on real Wikipedia
vs general-purpose compressors on the same bytes:

| compressor | bits/bit (real wikitext) |
|---|---|
| gzip -9 | 0.357 |
| xz/lzma -9 | 0.293 |
| bzip2 -9 | 0.283 |
| strongbase (pre-§63) | 0.2356 |
| **strong (improved)** | **0.2285** |
| *(reference, not run: lpaq1 ~0.20, paq8 ~0.16, cmix ~0.15)* | |

The bit-native core **beats every general-purpose compressor** (gzip/bzip2/xz) on real text, and the
**§63 improvement holds on real data at −3.0 %** (strongbase → strong), inside the documented corpus_big
band (2.9–3.4 %) — so the gain is *not* a proxy artefact. Honest bounds (kept): it does **not** beat its
peer class (dedicated CM/PPM: lpaq1/paq8/cmix — the README ladder), and the core's number is idealised
cross-entropy (no coder/header), an asymmetry that is < 2.2e-4 of the file here and changes no ranking.

**(B) Meaning — method, not data (a corrected verdict).** Does distributional meaning cluster *real*
synonyms on real text? Headline metric: is the synonym a top-10 nearest neighbour (usable)?

| method (same 1.4 MB real text) | synonyms in top-10 NN |
|---|---|
| naive PPMI (window 4, no smoothing) | 1/12 |
| **smoothed PPMI** (window 2, α = 0.75, rare-context drop) | **4/12** |

The naive readout barely clusters synonyms — but a **standard better method recovers 4/12 on the *same*
data, zero extra text** (`famous→popular` rank 1, `began→started` rank 2, `city→town` rank 3,
`built→constructed` rank 5). Changing **only the method, on identical data**, moves it — so the weak
result is a **method** limitation, not a data one. *An earlier draft concluded "this confirms §65/§66 —
the bottleneck is data," without testing either axis; the red-team showed **5× more data left naive NN
flat while a method change fixed it on identical data**, so that conclusion was **spin** and was removed.*
Honest statement: the distributional **mechanism is real on real text once the embedding is decent**;
whether more data helps further is untested here.

**Significance.** The real-data test does two things. It **confirms the compression headline on real
Wikipedia** (beats gzip/bzip2/xz; §63 holds) — a solid foundation to scale in item 3. And it is a clean
case of the **red-team discipline working under pressure**: a tempting "real text confirms our story"
conclusion was actually *false* (the limit is the embedding *method*, not data), and testing caught it.
The item-2 follow-on is now sharper — distributional meaning works on real text with a decent embedding,
so crossing the synonym wall on real data is a **better embedding** problem, not merely *more text*.

---

## 69. Toward generative intelligence — a leak-free memorization-vs-generalization instrument (`genmem.py`)

The goal: a stronger byte-level generative model with long-range memory, where success = **generalization
(intelligence), not memorization**. Rather than build a (worse-)LLM, we built the instrument the design
fan-out called for: keep **memorization** (the verbatim byte-match model = a copy oracle) and
**abstraction** (the logistic mixer) as *separate, individually-ablatable* channels, add **one new
long-range channel that is structurally incapable of verbatim copy** — a multi-timescale leaky-integrator
**reservoir** of byte-class features (a bounded smooth running summary; it cannot replay a span) — and
**measure**, on the official leak-free **wikitext-103** train/test split, whether it adds generalization.

**Headline metric = intelligence:** copy-ablated, **13-byte-decontaminated**, held-out bits/byte, reservoir
ON vs OFF (a gain here is generalization *by construction*: the channel can't copy, the match model is
off, and every test span sharing a ≥13-byte substring with train is excised — ~66 % of test bytes remain).

**Result (3 reservoir designs; copy OFF, decontaminated wt103 test):**

| channel | held-out bits/byte |
|---|---|
| baseline (orders only) | 2.3022 |
| **+ reservoir (the headline)** | 2.3789 — *worse* |
| + **scrambled** reservoir (sanity: random features) | 2.3593 — *also worse* |
| memorization (match/copy) gain, copy ON vs OFF | **+0.0111** — consistent, all runs |

**Findings (honest):**
- **The only long-range mechanism that improves held-out prediction is verbatim MEMORIZATION** (the match
  model, +0.011 bits/byte, consistent across every run and design).
- **The fixed reservoir adds NO generalization.** Across three principled designs — a count-table state
  (≈ noise), continuous mixer features (destabilized the simple SGD mixer), and continuous features +
  **RMSProp** (stable mixing) — it was noise or actively harmful. The **scrambled control** (30 *random*
  features) *also* hurt (≈ −0.06), revealing the dominant effect: adding fixed long-range **capacity
  overfits** on this data and hurts held-out, structured or not.
- So when intelligence is measured rigorously (copy-ablated, decontaminated, leak-free), a **hand-crafted**
  long-range memory provides none here — the available long-range gain is memorization.

**Significance + the honest implication.** This is direct, falsifiable evidence for the wall the project
keeps hitting: **generalizing long-range structure in a byte model is hard because it needs *learned*
memory** (the model learns *what* to remember) — which is what attention/SSMs do with deep learning at
scale; a from-scratch CPU context-mixer's "long range" is fundamentally the match model = **memorization**.
The durable asset is the **instrument**: a leak-free, copy-ablated, decontaminated, scrambled-controlled
testbed that *cannot mistake memorization for intelligence* (it rejected three designs). The next
genuinely-different step (the design's v2) is a **learned diagonal-linear recurrent memory** trained online
by RTRL (O(m)/step, no BPTT) — a real build with uncertain payoff at CPU scale — versus the honest
alternative the project's own evidence supports: **this architecture's strength is prediction/compression**
(lpaq1-class, beats gzip/bzip2/xz on real text, §63/§68), **not generative intelligence**. Scale caveat:
0.5–2.8 MB is tiny; the negative is robust for *fixed* features at this scale, and the scrambled-overfit
signal indicates scale is part of the difficulty — but the instrument now lets any future claim of
"added intelligence" be settled leak-free, the same way every other claim in this log was.

---

## 70. The no-copy intelligence track — a learned recurrent state that reaches parity, not past, the orders (`lstate.py`)

The bridge from memorization to intelligence, attempted concretely. Per the plan: keep §69's gate, and
replace the FIXED reservoir (which was indistinguishable from noise) with a **LEARNED, tiny (m=12),
diagonal-linear recurrent state** trained online by **RTRL** — `h_j = a_j h_j + (1−a_j) W_j·x` (a bounded
EMA; `a_j = σ(α_j)`; `x` = the byte's signed bits), fed as continuous features to the mixer. m=12 floats
**cannot store a span** → any held-out gain is abstraction, not copy. Gated by the genmem protocol:
match/copy OFF, 13-byte-decontaminated held-out, state ON/OFF/SCRAMBLED, a frozen (train-attributable)
read, and the copy gain reported separately.

**Result (leak-free wt103, copy OFF, decontaminated held-out bits/byte):**

| | 450 KB | 1200 KB |
|---|---|---|
| state − baseline (the win condition; <0 = beats orders) | +0.0016 | +0.0013 |
| state − scrambled noise floor (>0 = real learned signal) | **+0.021** | **+0.021** |
| match/copy (memorization) gain | +0.014 (consistent, every run) | |

**Findings (honest):**
- **The learned state genuinely learns.** It beats the SCRAMBLED noise floor by a robust, stable **+0.02**
  at every data size — unlike §69's fixed reservoir, which *matched* noise. The RTRL mechanism captures
  real generalisable structure, and the signal is data-scalable.
- **But it reaches PARITY with the local orders, not past them.** Only **+0.0013** above the no-state
  baseline at 1.2 MB (shrinking with data, but never crossing). What it learns is largely **redundant**
  with the order-0..6 contexts, so it ties them rather than adding net generalisation.
- **Forcing the state long-range doesn't help** (a stable long-range variant ties the learned-decay one).
  An earlier *unbounded* recurrence DIVERGED with long decays (RTRL trace explosion — a numerical artefact
  the EMA form fixes), a reminder to trust the gate's controls over a single number.
- **Memorization remains the only reliable long-range gain** (+0.014, every run).

**Significance — the honest line.** The bridge is **half-built**: a learned, tiny, no-copy memory that
*genuinely learns* (clearly above noise, data-scalable) and reaches **parity** with local context — a real
step beyond §69's noise-level reservoir — but does **not** beat the order baseline. The honest target
(a compact learned state that *beats* the match-OFF order baseline) is **not met**: the state TIES the
orders rather than transcending them. This is direct evidence that a *compact* recurrent state, even
learned, captures structure largely redundant with local n-gram context on natural text; transcending it
(true long-range generalisation) is exactly what needs **rich learned representations at scale** — the
deep-learning regime a from-scratch CPU diagonal-linear state cannot reach. The durable assets are the
**rigorous leak-free gate** (`genmem.py`) and the **corrected RTRL state channel** (`lstate.py`); the
remaining levers (Path B induced primitives as a richer state input §50/§54, residual-driven discovery,
Rust-scale data) are real but enter the rich-representation/scale frontier. The resting place the
project's own evidence supports: **a from-scratch predictor that beats standard compressors on real text
(§63/§68), plus a rigorous intelligence gate that shows exactly where compact byte-level "intelligence"
plateaus — at parity with local context, with memorization as the only reliable long-range gain.**

---

## 71. Item 3 — induced-computation channels: real signal, a sample-efficiency edge, but parity at scale (`prims.py`)

Per item 3 of the bridge plan: promote Path-B primitives — **stateful counters/detectors** — to
first-class, individually-ablatable predictor channels: bracket-nesting **depth**, quote state, digit-run
length, word-position, line-position, caps-run. Each is causal and becomes a count-model keyed by its
computed state, gated by the genmem protocol (copy OFF, 13-byte-decontaminated held-out, per-channel
ablation, a SCRAMBLED-bucket control). A counter is **unbounded state a fixed-window n-gram cannot
represent** → if it lowers copy-off held-out past the orders, the gain is *computation*, not copy.

**Per-primitive (450 KB), gap vs the order baseline (<0 = beats orders):** only **two of six** help —
`paren` nesting-depth (−0.0011) and `word` position (−0.0017); the other four (quote/digit/line/caps) are
redundant with the orders and hurt. The helpful pair beats its own scrambled control by +0.009.

**Robustness check (the decisive part) — `paren+word` vs baseline across data sizes:**

| train | gap vs baseline | vs scrambled noise |
|---|---|---|
| 450 KB | **−0.0026** | +0.0087 |
| 1200 KB | **−0.0020** | +0.0082 |
| 2700 KB | **+0.0018** | +0.0061 |

The cross is real at 450–1200 KB but **vanishes (flips positive) at 2.7 MB**: as data grows, the order
models learn the same structure the counters encode, so the counters become **redundant**. Net: **parity
at scale**, with a small **sample-efficiency** edge at low data (a counter generalises from few examples
faster than the orders, which need more data to learn the same regularity). The channels carry real signal
(they beat their scrambled noise floor at every size), just not enough to transcend the orders. *The gate
caught the small-data cross as non-robust at 2.7 MB — one slice short of a false claim.*

**Honest verdict (items 2 AND 3 together).** Every no-copy channel built — the smooth learned RTRL state
(§70) and now induced computation (§71) — captures REAL structure (robustly beats its scrambled noise
floor) but reaches **parity** with the local order models, not past them. Two independent mechanisms hit
the **same ceiling**, and that consistency is itself the finding: on natural text at CPU scale, no-copy
"intelligence" channels are largely **redundant with local n-gram context**; transcending it needs rich
learned representations at scale (the deep-learning regime). **Memorization (the match model, +0.014)
remains the only reliable long-range gain.** Durable assets: the leak-free intelligence gate plus two
clean, ablatable no-copy channels and a precise, falsifiable map of exactly where they plateau — parity
with local context, with a measured sample-efficiency edge that the orders erase as data grows.

---

## 72. Phase 1 of the new plan — the parity wall FALLS: word-slot binding, vectors the compression loss teaches (`wstate.py`)

> **Corrected by §77:** the slot LRU here held word prefixes, not words (six slots ≈ two words); the crossing is a learned current-word model, and the engine port trained its vectors with the wrong sign. See §77.5.

The §70/§71 ceiling (every no-copy channel ties the orders) is broken — not by more capacity or
better credit, but by fixing **what the state representation can hold**. The experiment,
`wstate.py` (7 arms, genmem gate: copy OFF, 13-byte-decontaminated held-out wt103, deterministic
mask, scrambled noise floor), was driven by a micro-probe (a rare word predicting an outcome two
words later — invisible to order≤6) that killed hypotheses one at a time:

- **v1 — EMA state + word-embedding projection** (the §70 diagnosis's named fix): at scale,
  `learned == randemb == bitsonly` **to four decimals**, all slightly *above* baseline. The
  projection does nothing. **v2 — exact per-word trace-bank credit** (long-range RTRL for
  embeddings, exact for frozen W): no change. **v3 — nonlinear bucket readout** (prims-style
  count table keyed on sign-quantised state): no change. Credit ✗, readout ✗.
- **The diagnosis the probe forces:** the EMA *superposes* ~5 recent words; the cue word is a
  minority of the signal, so **no function of the superposed state — linear or bucketed — can
  isolate one word's identity**. The binding, not the projection and not the credit, is the wall.
- **v4 — WORD SLOTS:** an LRU of the last 6 distinct words, each bound to its **own** 8-dim
  vector fed *directly* as mixer features. Binding is exact (a slot holds one word), and the
  credit becomes trivially exact — a slot persists ~6 words, so every bit while it is resident
  gradients the word's vector. No RTRL needed at all. The probe passes at once
  (`slots < slotsr < bitsonly` at the outcome byte).

**Result (leak-free wt103, copy OFF, decontaminated held-out bits/byte, seed 0):**

| train | baseline | bitsonly (m=32 EMA) | slotsr (frozen identity) | **slots (learned)** | scrambled |
|---|---|---|---|---|---|
| 150 KB | 2.3675 | 2.3733 | 2.3767 | **2.3705** | 2.5240 |
| 450 KB | 2.2758 | 2.2821 | 2.2858 | **2.2769** | 2.4334 |
| 1200 KB | 2.3715 | 2.3796 | 2.3818 | **2.3695** | 2.5398 |
| 2700 KB | 2.4793 | 2.4886 | 2.4904 | **2.4766** | 2.6584 |

- **The cross:** slots sit −0.003 at 150 KB and **cross below the orders baseline at ≥1.2 MB**
  (+0.0020, +0.0027 at 1.2/2.7 MB) — margin **growing with data**, the first no-copy channel in
  the project's history to pass the orders. **Replicated on seeds 1,2:** +0.0015…+0.0031 vs
  baseline at every seed×size.
- **Attribution is clean:** slots beat `bitsonly` (same EMA + bucket, no slots) by +0.012 at
  2.7 MB — the slot channel carries the win. Slots beat `slotsr` (identical architecture,
  **frozen random vectors**) by **+0.011…+0.014 at every seed×size**, and `slotsr` never crosses
  baseline — the win is specifically the **vectors the compression loss teaches**, i.e. MEANING
  beyond lexical identity, on copy-off, span-excised held-out data. The noise floor is +0.15 away.
- Every EMA-superposition variant reproduces §70's parity-minus **inside the same runs**.

**The strong.rs port — an honest negative that sharpens the thesis.** The same channel was
ported into the production engine (`BLMSLOTS=1`, env-gated; default verified **bit-identical**,
0.217011 on the 11 MB benchmark). At 11 MB it does **not** help: +0.0001…+0.0002 worse across
SBITS/LR_S/ALRS sweeps — strong's hashed order-8..32 contexts (1–5 words), word/prev-word models
and two match models **already cover short-range word identity**. Identity memory is *cheap* at
lpaq-class capacity; what production lacks is **similarity** — generalisation across related
words, soft content-based retrieval beyond exact match. That is Phase 2, and the port rule going
forward: **an instrument channel earns its Rust port only by beating strong itself, not the
orders-only rail.**

**Significance.** The §69–71 plateau was a *representation* artefact: superposed smooth state
cannot bind, so it relearns what the orders already know. Binding recent-word identity into
slots — with vectors trained by nothing but the compression loss — turns the same tiny budget
into net held-out generalisation that grows with data. The durable assets: the probe-driven
diagnosis method (kill hypotheses at byte-scale before trusting a 1-hour run), the 7-arm
instrument with its identity/noise/scrambled controls, and a replicated positive control for
the first time in the no-copy track. Next: similarity features on top of the slots
(`dot(E[cur], E[slot])` coherence, embedding-bucketed count experts that share evidence across
the word tail), then soft retrieval — the match model's generalisation.

---

## 73. Phase 2, step 1 — hard similarity keying honestly fails; the soft-retrieval requirement (`wstate.py`)

> **Corrected by §77:** `slots[0]` was the current prefix at word-interior bytes, so `slotsw` was a word model and `semsim` bucketed per-prefix predictor vectors; this section could not measure similarity. See §77.5.

Phase 2's thesis (from §72's port negative): identity memory is cheap at production capacity;
the open lever is **similarity** — generalisation across related words. Step 1 tested it in the
instrument with the cheapest possible mechanism: a count-table expert keyed not by word ID but
by the **embedding bucket** (8 sign bits of the current word's loss-trained vector — words the
loss has aligned share counts, so evidence should transfer across the word tail), plus
coherence features `dot(E[last word], E[slot word])`, against an **identity-keyed expert** of
identical placement (`slotsw`, 4096 word-hash slots ≈ a word model). Two design iterations
were forced by honesty checks: (i) the first version keyed on the *growing prefix id*, whose
vectors are untrained inside multi-letter words and dead at boundaries — a flaw the identity
control is immune to (unfair A/B); rekeyed both arms on the last completed word. (ii) A v6
**subword-composed init** (fastText-style: char-3gram vectors trained by shared slot credit;
unseen words initialise inside their morphological neighbourhood) was added and probe-verified
mechanically — composed init does place a held-out nonce word at its family-shared trigram
vector — but it cannot rescue hard buckets.

**Probes:** the tail probe (morphological families, few occurrences) — identity best, bucket
close, subword between. The **zero-shot probe** (held-out family member, first exposure,
learning off): composed init lands in the family direction, yet the sign-quantised bucket still
misroutes — **hard buckets destroy exactly the graded information similarity carries.**

**Scale (the §72 gate, same protocol, same seeds):**

| train | baseline | slotsr | **slots (§72)** | slotsw (identity expert) | semsim (bucket+dots) | scrambled |
|---|---|---|---|---|---|---|
| 150 KB | 2.3675 | 2.3767 | **2.3705** | 2.3655 | 2.3749 | 2.5240 |
| 450 KB | 2.2758 | 2.2858 | **2.2769** | 2.2754 | 2.2806 | 2.4334 |
| 1200 KB | 2.3715 | 2.3818 | **2.3695** | 2.3732 | 2.3743 | 2.5398 |
| 2700 KB | 2.4793 | 2.4904 | **2.4766** | 2.4824 | 2.4826 | 2.6584 |

- **The §72 anchors reproduce exactly** (slots row identical — determinism check).
- **Neither expert survives at scale.** `slotsw` is best at 150–450 KB (the word-model
  sample-efficiency edge) then turns into a **drag** at 2.7 MB (2.4824 vs slots' 2.4766) — the
  §71 pattern replaying: the orders absorb common-word signal, the extra correlated expert only
  adds estimation variance. `semsim` never crosses and **hurts bare slots** by 0.004–0.006 at
  every size: the hard bucket fires spurious early correlations the mixer must then unlearn.
- The bare **slots channel remains the only thing that beats baseline** (crossing intact:
  +0.002/+0.0027 at 1.2/2.7 MB).

**Verdict — a requirement, not a dead end.** Similarity keying in *hard* (quantised) form is
honestly negative on this instrument: at CPU scale it loses to identity keying, which itself
loses to doing nothing at scale. Combined with the probes, the diagnosis is precise: graded
similarity must be read out **softly** — similarity-weighted voting over retrieved
neighbours — which Python cannot afford (a per-byte k-NN over the vocabulary) but the Rust
engine can: **LSH/simhash buckets over the §72 vector table → top-k neighbours of the current
word → similarity-weighted word-model counts → one mixer input**; vectors trained by the
existing exact slot credit. That is Phase 2 proper, and per the §72 rule it earns its place
only by beating **strong itself** at 11 MB, not the orders-only rail.

---

## 74. Phase 2, step 2 — soft retrieval in the production engine: built, measured, honestly inert (`strong.rs`, `BLSOFT=1`)

> **Corrected by §77:** the vectors were trained with the wrong sign and the vote read the statistic of the delimiter that follows a similar word and applied it to the next word's letters, so this retrieval was inert by construction. See §77.5.

The §73 requirement, implemented exactly as named: the §72 word vectors indexed in an LSH
structure (sign bucket + the 8 one-bit-flip neighbours, capped lists); once per byte the last
completed word retrieves its top-`SOFTK` neighbours by normalised dot (≥ `SIMMIN`); per bit the
neighbours' **own word-model counts** (the existing `wdc` tables, same keys — no new statistics
store) are voted similarity-weighted into one trained `stretch()` head on the global mixer.
Default path verified **bit-identical** (0.231704 @300 KB; the 11 MB baseline re-run reproduces
0.217011 on the same binary).

**Port-rule A/B at 11 MB (obits 25, whole-stream bits/bit):**

| config | bits/bit | vs baseline |
|---|---|---|
| baseline (default) | **0.217011** | — |
| BLSOFT=1 (k=8, sim≥0.25) | 0.217101 | +0.00009 |
| BLSOFT=1 (k=16) | 0.217099 | +0.00009 |
| BLSOFT=1 (simmin=0, pure kNN) | 0.217097 | +0.00009 |
| BLSOFT=1 (ALRS=0.003) | 0.217159 | +0.00015 |

(1 MB pair: 0.216588 baseline vs 0.216742 soft — same sign.) The head correctly learns to
ignore the vote: at 11 MB single-pass, tail words have received too few credits for their
vector neighbourhoods to be informative, so the similarity vote degenerates toward the word
model / unigram average the engine already has — and the residual cost is the §72 slot
channel's own +0.00005 drag plus a little noise.

**The pattern across §72–§74 is now three independent mechanisms hitting the same production
wall:** identity slots, hard similarity keys, and properly-engineered soft retrieval all add
nothing to an lpaq-class engine on natural text at CPU scale — while the instrument
(`wstate.py`) proves the same channels carry real, growing, replicated signal against a weak
baseline. The reconciliation: **hashed order-8..32 + word + match models already span 1–5-word
memory; what the new channels add is real but below the engine's resolution at this scale.**
Where the evidence still points: the instrument's slots margin GROWS with data
(+0.0020 @1.2 MB → +0.0027 @2.7 MB); the production redundancy verdict is measured at ONE
scale (11 MB, single pass). The named next steps: (a) bigger single corpus (enwik8 slice)
for strong+slots at 30–100 MB, where the growth curve either survives contact with the engine
or the plateau claim hardens into a law; (b) long-range-dependency-rich streams (code, DNA,
logs — §54's induced channels helped exactly there); (c) the instrument stays the discovery
tool; ports follow the beats-strong rule.

---

## 75. The frontier sweep — scale, long-range-rich domains, and the absorber identified (`wstate.py`, `strong.rs`)

> **Corrected by §77:** instrument S=32 held ~7.5 words, the depth "information" was a noise-dimension penalty, and the absorber claim falls once the port's sign is fixed. See §77.5.

The two named next steps of §74, executed end to end, plus the mechanism question they forced.

**(1) Scale — enwik8 30 MB, production A/B (obits 25):** baseline **0.205801**; BLMSLOTS=1
0.205886 (+0.00009); BLSOFT=1 0.205893 (+0.00009). The production verdict is unchanged at
2.7× the data: the §72/§74 channels remain inert in the engine.

**(2) Long-range-rich domains — the instrument (same gate, copy OFF, decontaminated):**
- **CODE is the channel's home turf.** Slots cross at EVERY size from 100 KB
  (+0.0086/+0.0074/+0.0112/+0.0122 vs the orders baseline), with a larger meaning margin
  (+0.017…+0.020 vs frozen vectors) — 3–4× the text margins, at 10× less data.
- **DNA is an honest negative.** 6-mer-tokenised E. coli: slots WORSE than baseline at every
  size (−0.02), learned worse than frozen (−0.01). Fixed-width k-mer "words" carry no
  selectional structure; the codon/revcomp specialists (§55) are the right tools there.
- Production on code (800 KB): baseline 0.150989, slots +0.00009, soft +0.00011 — code's
  stronger signal does NOT survive the engine either.

**(3) Two wrong absorber hypotheses, killed by experiment:**
- **Depth/range ✗.** Deep slots (S=32/64; `NSLOTS` in strong.rs, `WSLOTS` in wstate.py):
  monotonically worse in production (+0.0003…+0.0009) and a clean instrument NEGATIVE on
  code (−0.03, meaning-margin gone). The channel is intrinsically SHALLOW (~6 words) —
  depth drowns the readout in estimation variance. (Note what the noise floor shows: at
  S=32 the scrambled-floor separation GREW to +0.64 vs +0.17 at S=6 — the state carries
  MORE raw information at depth; the linear mixer cannot extract it. An information-vs-
  readout gap, not an information absence.)
- **Copy ✗.** The reconciliation arms (genmem's match model added to the instrument):
  slots still beat orders+match — wt103 +0.002 at 1.2/2.7 MB, code +0.006…+0.013. The
  no-copy gate was NOT feeding the slot channel copy's lunch.
- **The word-model family ✓.** On code, adding an identity word-expert on TOP of slots
  (`slotsw` vs `slots`) adds +0.028…+0.037 — word-keyed statistics carry even more signal
  than slots in the same window. Production strong HAS that family (word, prev-word,
  hashed orders 8–32). That is the absorber.

**The law, stated.** The word-slot channel's signal is real, replicated, copy-independent,
strongest on code — and lives inside the same ~6-word window that word-keyed statistics
already harvest; any engine with a decent word model absorbs it. For a learned-memory
channel to matter in production it must bind structure OUTSIDE that window; the S=32
experiment shows the information exists at depth but the readout capacity (linear mixer +
CPU-scale data per word) does not. The two live directions that remain: a regularised,
nonlinear readout that can exploit depth (the deep-learning regime, now with a measured
target), and domains with genuine long-distance references (cross-file code, structured
documents) where the window assumption fails outright. The instrument-first method did its
job: it located the channel (code ≫ text ≫ DNA), killed three wrong explanations, and
named the true one.

---

## 76. Attacking the information-vs-readout gap — shared recency-discounted heads (`wstate.py` `proj` arms)

> **Corrected by §77:** the depth target this section attacks was a noise-dimension artifact; slots held prefixes. See §77.5.

§75 left a measured target: at slot depth 32 the state's raw information GREW (scrambled-floor
separation +0.64 vs +0.17 at depth 6) while net value went negative — the linear mixer over 256
raw dims drowns in variance. The §76 build: replace the per-(position, dim) weights with **H=4
shared recency-discounted projections** — `feat_h = Σ_k γ^k (w_h · E[slot_k]) / Σ_k γ^k`,
γ=0.85 — 9 parameters per head, depth-independent; exact per-bit credit to heads AND vectors.
Arms: `proj` (learned), `projr` (frozen vectors — meaning control), `projscr` (noise — floor).
Plus a real cross-file corpus: 10.4 MB of the Python standard library (596 files).

**Results (same gate; WSLOTS=32 unless noted):**

| run | proj − baseline | proj − projr (meaning) | proj − projscr (floor) |
|---|---|---|---|
| code 585 KB (S=32) | −0.0012…−0.0032 | +0.005…+0.008 | +0.012…+0.016 |
| code 585 KB (**S=6**) | **+0.0012…+0.0014 (cross)** | +0.007…+0.010 | +0.014…+0.018 |
| wt103 2.7 MB (S=32) | −0.002…−0.003 | +0.004…+0.007 | +0.012…+0.015 |
| stdlib 0.4–3.4 MB (S=32) | +0.001 / −0.001 / −0.000 / −0.003 | +0.004…+0.007 | +0.014…+0.016 |

**Findings, honestly:**
- **The variance penalty is fixed**: raw S=32 sat −0.03 below baseline; proj@32 sits at parity
  everywhere — depth now costs nothing. Real learned signal at every depth and corpus (meaning
  and floor margins always positive).
- **But 4 shared linear heads do not cash the depth**: proj@32 never crosses consistently
  (stdlib crosses only at the smallest size). proj@6 crosses on code — the head form itself is
  sound — but with ~1/10 the margins of the free 48-dim readout, and recency-averaging over 32
  slots DILUTES the recent-window signal that carries the value (position 0's share ≈ 15%).
- **The ladder is now measured**: free-48-dim@6 ≫ 4-head@6 > 4-head@32 ≈ raw@32. Net value
  per unit of depth tracks readout capacity per unit of depth. Closing the gap needs a readout
  whose capacity GROWS with depth non-linearly (content-selected/attentional reading of the
  bound words — "which of my 32 bound entities does THIS context concern?") — the deep-learning
  regime, now with a precise, budgeted target rather than a vague ambition.

**Where this leaves the strain.** The instrument's discoveries stand (binding > superposition;
code ≫ text ≫ DNA; the word-model-family absorber; the depth information gap). The next build
that could matter in production is an attentional read of the deep slot memory — select by
content, not by recency — developed instrument-first under the same gate, ported only on
beats-strong.

---

## 77. The audit — the §72 channel was a word model, the port trained its vectors uphill, and the attentional read fails its probe (`wstate.py` WSLOTMODE + `attn*` arms, `strong.rs` sign fix + `BLSTRIPW/H`, `_bind_probe.py`)

**This section corrects §72–§76.** The planned §77 build (an attentional read of deep slot memory,
HANDOVER §5) was preceded by a premise check, and the check found two implementation defects that
change what §72–§76 measured. Every finding below was red-teamed by independent agents
(mismatch hunt, claims audit, code review with complex-step gradients, probe re-scoring).

### 77.1 Two defects

1. **The instrument's slot memory held word PREFIXES, not words.** `wstate.py` moved the growing
   prefix id to the front of the LRU at every letter byte. After "the quick brown fox … the quick
   cat" the S=6 slots were `cat, ca, c, quick, quic, qui`. Over 300 KB of wt103, S=6 held 1.98
   distinct words on average counting the word being typed (1.21 whole previous words) and S=32
   held 7.55; slot 0 was the current prefix at 100 % of letter
   bytes. The ledger and code comments described "the last S distinct words" throughout §72–§76,
   and §73's "rekeyed on the last completed word" never happened at word-interior bytes.
   `strong.rs BLMSLOTS` inserts only completed words, so **the port never tested the channel that
   crossed**. Fix: env `WSLOTMODE=prefix` (default, bit-identical to §72–§76: sha256 of per-bit cost streams matches the
   committed code on 9 legacy arms) | `word` (completed words only, matching the engine).
2. **The engine trained slot vectors in the wrong direction.** `strong.rs` accumulated
   `g += (y − p_g)·w` (= −dL/df) and applied `v −= lr_s·g`, which is gradient ASCENT. The header
   comment described the correct step. Every §72/§74/§75 production A/B with `BLMSLOTS` or `BLSOFT`
   therefore ran vectors pushed uphill. Fixed to `v += lr_s·g`; default bit-identity re-verified
   (0.231704 @ 300 KB `corpus.txt`). At 300 KB, `BLMSLOTS=1` moves from 0.231895 (bug) to 0.231708.

Further defects the audit found (documented, not fixed): `BLSOFT` votes each neighbour's `wdc`
counts keyed as if the neighbour were the current prefix — the statistic of the delimiter after
that word — and applies it to the letters of the next word, so §74 soft retrieval was inert by
construction; the §73 hard-similarity arms bucketed per-prefix predictor vectors, so they could not
measure similarity; `semfast` carries S permanently-zero inputs.

### 77.2 The re-gate: what the §72 channel actually was

Same gate as §72 (copy OFF, 13-byte decontamination, deterministic mask), **seed 0 only**, arms in
parallel. Values are bits/byte on the scored held-out bytes; lower is better. Outputs: `_77a_*.txt`.
All slot arms also carry the M=32 EMA state and bucket expert.

**Code** (`code_train/test`):

| arm | 100 KB | 200 KB | 400 KB | 585 KB |
|---|---|---|---|---|
| baseline (orders 0–6) | 2.1948 | 1.9731 | 2.0504 | 2.1037 |
| §72 slots, prefix S=6 | 2.1862 | 1.9656 | 2.0392 | 2.0916 |
| **one prefix slot (S=1), learned** | **2.1803** | **1.9578** | **2.0321** | **2.0834** |
| one prefix slot, frozen | 2.2028 | 1.9800 | 2.0588 | 2.1117 |
| whole words S=6, learned | 2.2114 | 1.9892 | 2.0654 | 2.1189 |
| whole words S=6, frozen | 2.2060 | 1.9851 | 2.0634 | 2.1172 |
| whole words S=32, learned | 2.2194 | 1.9967 | 2.0750 | 2.1285 |
| whole words S=32, frozen | 2.2261 | 2.0032 | 2.0813 | 2.1358 |

**wt103** (`wt103_train/test`):

| arm | 150 KB | 450 KB | 1200 KB | 2700 KB |
|---|---|---|---|---|
| baseline | 2.3675 | 2.2758 | 2.3715 | 2.4793 |
| §72 slots, prefix S=6 | 2.3705 | 2.2769 | 2.3695 | 2.4766 |
| **one prefix slot (S=1), learned** | **2.3606** | **2.2667** | **2.3590** | **2.4641** |
| one prefix slot, frozen | 2.3741 | 2.2836 | 2.3797 | 2.4888 |
| whole words S=6, learned | 2.3814 | 2.2925 | 2.3915 | 2.5026 |
| whole words S=6, frozen | 2.3767 | 2.2869 | 2.3842 | 2.4930 |
| whole words S=32, learned | 2.4046 | 2.3170 | 2.4152 | 2.5286 |
| whole words S=32, frozen | 2.3962 | 2.3057 | 2.4043 | 2.5149 |

**Findings.**
- **The §72 win is a learned current-word model.** One slot that holds the current prefix beats the
  orders baseline by +0.0145…+0.0203 on code and +0.0069…+0.0152 on wt103, growing with data —
  1.7× the §72 six-slot margin on code and ~5.6× on wt103 at the largest sizes. Frozen random vectors in the same slot
  lose. Extra slots only dilute it. An 8-dim vector per prefix id is a word model, which the
  orders-0..6 baseline lacks and `strong.rs` already has (`wdc`, keyed per prefix). "Meaning"
  (learned vs frozen) was per-id trainable parameters versus none.
- **Real word binding loses.** Whole-word slots are worse than baseline at every size, at 6 and 32
  slots, on both corpora. The loss grows with data on wt103 (−0.0139 → −0.0233 at S=6, −0.0371 →
  −0.0493 at S=32). Learned vectors are worse than frozen ones at S=6 on both corpora and at S=32 on
  wt103; only code at S=32 has learned beating frozen (+0.006…+0.007), still below baseline. Part of the loss is EMA overhead, but on wt103 the S=6 arm is
  also worse than the §72 `bitsonly` arm (EMA only) by −0.008…−0.014. The effects are ~10× the §72
  seed spread, so seed 0 is likely representative, but they are not replicated.
- **The §75/§76 "information at depth" was a noise-dimension penalty.** Scrambled-minus-baseline on
  code at 100 KB is +0.078 / +0.162 / +0.669 for 40 / 80 / 288 noise dims (≈0.002 bpb per dim), and
  frozen random vectors show the same "floor separation" as learned ones. There was no measured
  depth target for the attentional read.

### 77.3 Production with the sign fixed, and the absorber ablation

New flags (default OFF, bit-identical): `BLSTRIPW=1` removes the word and previous-word models
(and the word context of APM5); `BLSTRIPH=1` removes hashed orders 8–32. Full length, obits 25,
bits/bit whole-stream. Slot gain = (X) − (X + `BLMSLOTS=1`); positive means slots helped. Outputs:
`_77b_*.txt`.

| engine variant | corpus_big 11 MB | + slots | slot gain | stdlib.bin 10.6 MB | + slots | slot gain |
|---|---|---|---|---|---|---|
| full | 0.217011 | 0.217016 | −0.000005 | 0.127624 | 0.127586 | +0.000038 |
| no word models | 0.220255 | 0.220256 | −0.000001 | 0.129935 | 0.129892 | +0.000043 |
| no high orders | 0.217073 | 0.217076 | −0.000003 | 0.128946 | 0.128897 | +0.000049 |
| neither | 0.220438 | 0.220438 | 0.000000 | 0.131433 | 0.131386 | +0.000047 |

- **Whole-word slots carry nothing on text** even with learning fixed and the word models removed.
- **On code they give a tiny consistent gain** (+0.00004 bits/bit, 0.03 %) that barely changes
  (+0.000038 → +0.000043) when the word models are removed. **Removing the word-model family reveals
  no hidden whole-word slot signal.** Whether it absorbs the §72 learned prefix-vector channel was
  never tested, because that channel was never ported. §75's `slotsw` evidence compared a word model
  against a diluted word model.
- These are whole-stream online compression numbers, not decontaminated held-out numbers.
- The word models are worth +0.0032 (text) and +0.0023 (code) bits/bit. Hashed orders 8–32 are worth
  +0.00006 on text and +0.0013 on code at full size (at ≤1 MB they were slightly negative on text).

### 77.4 The long-gap content-binding probe (`_bind_probe.py`, output `_77p_probe.txt`)

Synthetic stream: `<cue> <G distractors> then <outcome> .` with 8 nonce cues mapped to 8 outcome
words and 26 distractor words. Gain = bits saved on the outcome word versus the orders baseline
(bound ≈ 3 bits), 1200 training sentences, 200 newly drawn test sentences scored online, data seeds 7/8 (at G=2,
37/200 and 50/200 test sentences repeat a training sentence verbatim; gains are the same on repeats).
Criteria were written into the script before any run (file saved 12:38:22, first result 12:38:30;
wstate snapshot sha256 `376380201f…e860`). All word-mode 32-slot arms had the cue in memory on
100 % of test outcomes (mean LRU position 3.0 / 10.7 / 16.9 at G = 2 / 12 / 24).

Pre-registered: **P1** sanity — at G=2 the best of {free readout S=32, `attn`, `attncos`} gains
≥ 1.0 bit on both seeds. **P2** content binding — at G=12 and G=24 on both seeds an attention arm
gains ≥ 1.5 bits, beats `attnrec` by ≥ 1.0 and beats the best of {free readout, §76 heads} by ≥ 0.5.
**P3** reach — S=6 whole words and prefix S=32 gain ≤ 0.3 at G=24. **Kill rule** — if P2 fails, no
corpus grid for the attention arms.

| arm | G=2 s7 | G=2 s8 | G=12 s7 | G=12 s8 | G=24 s7 | G=24 s8 |
|---|---|---|---|---|---|---|
| whole words S=6 (reach control) | +0.769 | +0.798 | −0.036 | −0.051 | −0.075 | −0.022 |
| prefix S=32 (legacy deep arm) | +0.355 | +0.435 | −0.131 | −0.083 | −0.105 | −0.104 |
| whole words S=32, free readout | +0.506 | +0.502 | −0.196 | −0.167 | −0.179 | −0.243 |
| §76 heads, whole words S=32 | +0.034 | +0.050 | −0.019 | −0.028 | −0.016 | −0.024 |
| `attn` (content attention) | −0.044 | −0.003 | −0.011 | +0.007 | −0.019 | −0.043 |
| `attncos` (cosine attention, κ=8) | −0.041 | −0.020 | −0.039 | −0.003 | −0.036 | −0.060 |
| `attnr` (frozen vectors) | −0.024 | −0.034 | −0.022 | −0.001 | −0.032 | −0.024 |
| `attnrec` (recency vote) | +2.184 | +2.057 | −0.002 | −0.020 | −0.013 | −0.016 |

Secondary (G=12, seed 7, 3600 training sentences): `attn` −0.037, `attncos` −0.030, `attnrec` −0.055.

**Verdict: P1 FAIL, P2 FAIL, P3 PASS — the kill rule fires; no corpus grid for the attention arms.**
Scope: these arms at default settings (WTOPM=4, κ=8, default learning rates, untuned). No
content-attention arm binds a cue at any gap, even two words back, and tripling the data at G=12
(one seed) does not help. The pre-registered sanity check P1 failed; the evidence that the probe can
register binding is post hoc: `attnrec` recovers ~70 % of the information at G=2, where the cue is
recent enough to enter its vote set.

The red-team recomputed every table cell from the 58 per-job records and independently re-ran 6 G=2
cells (plus the baseline under the prefix env), matching per sentence to within 9e-16. It also
measured the vote readout in 3 cells (one seed each, 17–45 test outcomes): when a content arm puts the
cue in its vote set, the cue's vote entry predicts the true bit 84–94 % of the time on the informative
bits 3–7 of the outcome's first byte, but the mixer weight on the vote feature averages negative
(−0.16…−0.76; `attnrec` +1.22), so a correct vote pushes away from the true bit (−0.05…−0.10). The
cause of that sign was not isolated: `attnrec`'s vote set at G=2 is also 3/4 non-cue, yet its weight is
positive. With a negative weight the gradient favours moving attention off correct voters; this is
consistent with a self-reinforcing readout trap, inferred from end-of-run snapshots and not traced
during training, and suggests more than a selection failure.

Attention behaviour: `attn` at the pre-registered init was near-uniform and position-driven (content
logit spread ~0.002). `attncos` (added after that diagnosis; scored only on the synthetic probe) was
sharp. On 40 KB of wt103 and code its choice was dominated by the learned bias direction `bq` (argmax
unchanged on 83–96 % of bytes when the query word is removed); on the probe stream its argmax varied
across all 32 slots and did not select the cue.

### 77.5 What stands and what falls in §72–§76

| claim | status |
|---|---|
| §70/§71 EMA-state and induced channels reach parity only | stands (untouched by both defects) |
| §72 "word-slot binding crosses the parity wall" | falls — the crossing is a learned current-word model |
| §72 "the win is meaning beyond identity" | falls — per-id parameters versus none |
| §72/§74/§75 production numbers | stand as measurements of uphill-trained vectors and a mis-keyed vote; they say nothing about learned slots |
| §73 "hard quantisation destroys similarity; similarity must be read softly" | unsupported — the bucketed vectors were per-prefix predictors |
| §74 "soft retrieval inert because tail vectors are under-trained" | falls — inert by construction (sign + keying) |
| §75 "code ≫ text ≫ DNA" | stands for a learned word model over orders 0–6, not for word binding |
| §75 "the word-model family is the absorber" | unsupported as stated — whole-word slots have almost nothing for it to absorb (§77.3); the §72 channel it was meant to explain is itself a word model and was never ported |
| §75/§76 "information exists at depth; readout capacity is the gap" | falls — noise-dimension penalty |
| §76 "proj@6 crosses on code" | overstated — negative at 100/200 KB, single seed |
| §75 "copy is not the absorber" | weak — the match model is worse than no match under 13-gram decontamination |

The §77 re-gate verdicts in this table are seed 0 and one deterministic engine run per config.

### 77.6 Where this leaves the strain

> **Updated by §78:** the prefix vector's instrument gain is mostly forgetting that the instrument's
> cumulative counts lack, a scratch all-mixer engine port gains +0.0004–0.0005 bits/bit, and a forced-selection
> oracle shows the vote works at long gaps once selection is given. See §78.5.

After §69–§77, the only learned memory channel that beat order-n statistics on leak-free held-out data
was a learned current-prefix vector, which is a word model. Production has a count word model, but the
learned vector was never tested against one or ported. Whole-word binding failed once measured
correctly (code and wt103 at seed 0, production at 11 MB). Recency heads and content attention were
re-tested in word mode only on the synthetic probe, where they failed. Soft similarity was never
re-measured with correct keying. The only positive production result is +0.00004 bits/bit on code, at
about 1–6 % more wall-clock time (+1.8 % for the full engine on stdlib; concurrent runs, noisy).

Recommendation: close the learned-memory model track. The pre-registered kill rule fired, and every
§72–§76 positive that was decomposed traced to a bug or a word model. Two narrow, measured loose ends
remain if anyone reopens it: (1) the vote-readout trap of §77.4 — a forced-selection oracle (cue forced
into the vote set during training and test, vote weight logged; needs new code) would show whether the
vote readout works once selection is given; (2) the §72 learned prefix vector against a count word
model on the same gate, which would settle whether it is anything more than a word model.

---

## 78. The two loose ends — the prefix vector is mostly forgetting, the instrument's counts never forget, and selection is the bottleneck (`wstate.py` §78 arms, `_bind_probe.py --grid78`)

§77.6 left two narrow questions. Both were pre-registered in the code before any run, built with
self-tests (bit-identity of all 20 legacy arms, causality flips, high-precision gradient checks),
reviewed by two independent agents with their own re-implementations, then red-teamed after the runs.

### 78.1 §78A — is the §72 learned prefix vector more than a count word model? (pre-registered)

Arms have no EMA state, so they isolate the channel: `wcnt` = orders 0–6 + an exact count expert keyed
by (slot 0 id, phase, partial byte); `pvec` = orders + the learned 8-dim vector of slot 0; `pvecr` =
frozen; `wcntpvec` / `wcntpvecr` = both. Gate as §77.2, prefix mode, one slot. A1 = `wcnt` − `wcntpvec`,
A2 = `wcntpvecr` − `wcntpvec`, each required ≥ +0.002 bpb at every size on code and wt103, then stdlib.
Outputs `_78a_code.txt`, `_78a_wt103.txt`, `_78a_stdlib.txt`.

| corpus (sizes) | `wcnt` − baseline | `pvec` − baseline | A1 | A2 |
|---|---|---|---|---|
| code (100/200/400/585 KB) | +0.0620 +0.0561 +0.0516 +0.0528 | +0.0224 +0.0238 +0.0263 +0.0288 | +0.0219 +0.0240 +0.0272 +0.0297 | +0.0219 +0.0232 +0.0270 +0.0292 |
| wt103 (150/450/1200/2700 KB) | +0.0298 +0.0257 +0.0235 +0.0205 | +0.0127 +0.0166 +0.0204 +0.0245 | +0.0105 +0.0149 +0.0188 +0.0220 | +0.0106 +0.0150 +0.0186 +0.0217 |
| stdlib (400/1200/2400 KB) | +0.0478 +0.0392 +0.0366 | +0.0269 +0.0240 +0.0247 | +0.0276 +0.0243 +0.0242 | +0.0266 +0.0235 +0.0235 |

**Pre-registered verdict: A1 and A2 pass on all three corpora.** Frozen vectors add ~0 (`pvecr` −
baseline −0.0002…+0.0010).

### 78.2 Red-team (post hoc): what the vector carries is forgetting

The mechanism red-team reproduced every number bit-exactly, found no leak (code read, 18/18 bit-flip
checks, one-byte-delayed credit unchanged), then tested explanations on code 100/200 KB and wt103
150/450 KB:

- **Every count table in the instrument is cumulative and never forgets** (orders 0–6 and `wcnt`).
  The learned vector, updated at a constant rate, is an exponentially forgetting per-prefix nudge
  toward what has *recently* followed that prefix. Its gain sits on bytes whose recent rate is ≥ 2×
  their long-run rate (+0.12–0.14 bits per such byte) and turns negative where the rate recently
  dropped; 79–83 % of it is on prefixes seen ≥ 100 times; freezing the vectors during the test removes
  77–91 % of it; bytes right after a word carry only 8–9 %.
- **With the standard PAQ nonstationary count rule** (on each update the opposite count, if > 2,
  becomes n/2 + 1) on all tables, A1 falls to +0.0031 +0.0027 +0.0030 +0.0028 on code and
  +0.0004 +0.0006 +0.0015 +0.0026 on wt103 — **failing the +0.002 bar on wt103 at 150, 450 and 1200 KB.**
  A fast learned scalar per prefix recovers 73–82 % of A1; confidence or pooled count features add nothing.
- **The instrument's orders-only rail is 0.13–0.37 bpb weak.** The same orders with nonstationary counts:
  code 2.1948 → 1.9172, 1.9731 → 1.6736, 2.0504 → 1.7153, 2.1037 → 1.7349; wt103 2.3675 → 2.2411,
  2.2758 → 2.1156, 2.3715 → 2.1722, 2.4793 → 2.2531. **Every instrument "crossing" in §70–§78 was
  measured against this non-forgetting rail.** Any further instrument claim must use nonstationary
  counters in the baseline and in the arm.

**Corrected §78A reading:** A1/A2 pass as pre-registered, but the learned prefix vector is mostly
recency adaptation that cumulative counts lack. The reading "it carries something a count word model
cannot" is retired.

### 78.3 Red-team (post hoc): the engine absorbs it only under the §72 placement — and a scratch port gains

The absorption red-team rebuilt strong's mixer stack and APM chain in the instrument (code 100/200 KB,
wt103 150/450 KB):

- Word-keyed SSE (an APM5 copy, even with exact keys) absorbs none of A1 (77–105 % kept).
- strong's own context-selected mixers (by phase + previous byte, and by phase + order-2 hash) absorb a
  vector fed to the global mixer only: 1–12 % of A1 survives. This is why a §72-placement port is inert.
- **Fed to all three mixers, the margin returns at 119–162 % of A1**, and learning beats frozen vectors.

A scratch port of that placement (`BLPVEC=2`: key = current letters-only prefix kept through delimiters
until the next letter; 8 inputs to all three mixers; credit = each mixer's own error × its pre-update
weight; `v += LR_S·g`; untuned defaults) gave, whole-stream bits/bit:

| corpus | baseline | `BLPVEC=2` | gain | frozen (`LR_S=0`) gain |
|---|---|---|---|---|
| corpus_big 11 MB (obits 25) | 0.217011 | 0.216509 | +0.000502 | +0.000046 |
| stdlib.bin 10.6 MB (obits 25) | 0.127624 | 0.127204 | +0.000420 | +0.000033 |
| corpus_big without word models | 0.220255 | 0.219417 | +0.000838 | — |
| stdlib without word models | 0.129935 | 0.129275 | +0.000660 | — |

The global-only placement was +0.00004…+0.0001 at 0.3–2.8 MB. The gain is ~10× the §77.3 whole-word
slot result and meets the §72 port rule (beat 0.217011), but it is a scratch binary, one configuration
per corpus, copy ON and not decontaminated. Given §78.2, it may be a crude adaptive-rate effect that a
better counter rule gives more cheaply. **§79 builds it in the repo with forgetting controls and a third
corpus, criteria written before the runs.**

### 78.4 §78B — forced-selection oracle for the §77 vote trap (pre-registered)

`attnorc1` forces the vote set to the cue's slot; `attnorc4` forces the cue plus the 3 other top-attended
slots. Same probe data and protocol as §77.4. Gain in bits per outcome word vs baseline (bound ≈ 3);
output `_78b_probe.txt`.

| arm | G=2 s7 | G=2 s8 | G=12 s7 | G=12 s8 | G=24 s7 | G=24 s8 |
|---|---|---|---|---|---|---|
| `attn` (reference) | −0.044 | −0.003 | −0.011 | +0.007 | −0.019 | −0.043 |
| `attnorc1` (cue only) | +2.853 | +2.827 | +2.459 | +2.485 | +2.456 | +2.515 |
| `attnorc4` (cue + 3) | +1.800 | +1.884 | +0.399 | +0.524 | +0.503 | +0.569 |

**Verdict: B1 PASS (cue-only gain ≥ 1.5 at G=12/24), B2 PASS (its vote weight > 0 in all four cells),
B3 FAIL (cue + 3 < 1.0).** The vote store and readout work at long gaps once selection is given;
selection is the bottleneck.

**Corrected B3 reading (red-team, post hoc; supersedes the line the probe auto-printed):** `attnorc4`'s
vote weight is positive in 6/6 cells, so distractors in the vote set do not create the negative-weight
trap. Its loss is dilution: at G=12/24 the cue holds ≈0.30 of the weight inside the vote set, and a
renormalised stretch average scaled by one global mixer weight (pinned by the other ~99 % of bits)
cannot undo it. With distractor stretches zeroed, pure dilution predicts first-byte gains of +0.43 /
+0.61 at G=12/24 (seed 7) against +0.40 / +0.50 observed. `attn`'s negative weight has a different
source: its vote set almost never contains the cue, so the cue's cell never trains and every member
votes at chance.

### 78.5 What changes

- §77.6's closure stands for word binding, but two measured leads replace "no target": a production
  port of the learned prefix vector at all-mixer placement (§78.3), and a responsibility-trained
  mixture-of-experts gate that combines votes in probability space so no shared mixer weight can dilute
  or reverse selection (§78.4). Both are pre-registered in §79.
  (The §78.4 lead's criteria were registered in `_bind_probe.py`, not in §79 of this ledger.)
- **Instrument hygiene:** nonstationary counters are now a required control for any instrument claim.

---

## 79. Pre-registration (written 2026-09-14 before any §79 run)

### 79A — the learned prefix vector in the production engine (`strong.rs` `BLPVEC=2`, `BLWNS`, `BLNSALL`)

Build reviewed with no blockers (sign toy, 19.2 M credit finite-difference checks, key re-derivation on
every byte, nonstationary rule checked on every bit, 15-digit bit-identity of defaults and older flags,
lookahead flips). Grid: whole-stream bits/bit, obits 25, on corpus_big 11 MB, stdlib.bin 10.6 MB and the
first 30 MB of enwik8. Arms: base; `BLPVEC=2`; `BLPVEC=2 LR_S=0`; `BLWNS=1`; `BLWNS=1 BLPVEC=2`;
`BLNSALL=1`; `BLNSALL=1 BLPVEC=2`. Outputs `_79a_*.txt`.

- **Q1 gain:** `BLPVEC=2` beats base by ≥ 0.0003 bits/bit on all three corpora.
- **Q2 learning:** the frozen arm's gain is < 1/3 of the learned arm's gain on all three corpora.
- **Q3 more than word-model forgetting:** (`BLWNS=1`) − (`BLWNS=1 BLPVEC=2`) ≥ 0.0002 on all three corpora.
- **Exploratory, no gate:** `BLNSALL=1` with and without `BLPVEC=2`.
- **Reading:** Q1–Q3 pass → the learned prefix vector is a production improvement beyond cheaper forgetting.
  Q1 and Q2 pass but Q3 fails → if `BLWNS=1` alone gains at least as much as `BLPVEC=2`, the cheaper
  forgetting rule is the improvement to adopt instead. Q1 fails → the scratch result does not hold at the
  third corpus or in the repo build. No default is changed in this section either way.

---

### 79A result (runs finished before 79H was written)

| corpus | base | `BLPVEC=2` | frozen | `BLWNS=1` | `BLWNS=1 BLPVEC=2` | `BLNSALL=1` | `BLNSALL=1 BLPVEC=2` |
|---|---|---|---|---|---|---|---|
| corpus_big 11 MB | 0.217011 | 0.216509 | 0.216965 | 0.215843 | 0.215536 | 0.216431 | 0.216158 |
| stdlib.bin 10.6 MB | 0.127624 | 0.127204 | 0.127591 | 0.125273 | 0.125011 | 0.124459 | 0.124261 |
| enwik8 first 30 MB | 0.205801 | 0.204973 | 0.205728 | 0.203368 | 0.202723 | 0.203764 | 0.203139 |

Q1 PASS (+0.000502 / +0.000420 / +0.000828). Q2 PASS (frozen/learned 0.092 / 0.079 / 0.088).
Q3 PASS (+0.000307 / +0.000262 / +0.000645). Pre-registered reading: the learned prefix vector is a
production improvement beyond cheaper forgetting. The forgetting control itself (`BLWNS=1`, the PAQ
nonstationary rule on the word and previous-word tables) is a larger gain than `BLPVEC=2`'s: +0.001168 /
+0.002351 / +0.002433. Outputs `_79a_*.txt`.

### 79H — held-out confirmation (written 2026-09-14 17:52, before any 79H run)

The three 79A corpora shaped the idea, so the adoption decision uses data that played no part in it:
the last 30 MB of enwik8 (`data/enwik8_tail30`, obits 25), the E. coli genome as letters
(`data/ecoli.txt`, 4.6 MB, obits 25), this repository's own Python and Rust source
(`data/repo_code.txt`, 770 KB, obits 24, not Python-stdlib text), and the full 100 MB enwik8 headline
(obits 25). Arms: base, `BLWNS=1`, `BLWNS=1 BLPVEC=2`. Outputs `_79h_*.txt`.

- **H1 forgetting:** `BLWNS=1` beats base by ≥ 0.0005 bits/bit on all four corpora.
- **H2 vector on top:** `BLWNS=1 BLPVEC=2` beats `BLWNS=1` by ≥ 0.0002 on enwik8 tail, repo code and full
  enwik8 (DNA exploratory: its letters-only "words" are whole lines).
- **H3 safety:** neither flag is worse than base by more than 0.0002 on any corpus.
- **Reading:** H1 and H3 pass → recommend `BLWNS=1` as the new default. H1, H2 and H3 pass → recommend
  `BLWNS=1 BLPVEC=2`. Defaults change only in a separate, reviewed commit.

---

### 79A timing correction (post hoc)

"Before any §79 run" holds only for the 79A grid, which launched at 16:45:10, four seconds after these
criteria were written. Before that, the builder and reviewer had run all seven 79A arms at 300 KB
(corpus.txt, obits 23: base 0.231704, `BLPVEC=2` 0.231420, frozen 0.231684, `BLWNS=1` 0.231576,
`BLWNS=1 BLPVEC=2` 0.231368, `BLNSALL=1` 0.233588, `BLNSALL=1 BLPVEC=2` 0.233453), and the §79B smoke
probe had run at 16:30. The corpus_big and stdlib `BLPVEC=2` and frozen values equal §78.3's scratch
results to six digits. Q1 and Q2 are therefore new evidence only on enwik8's first 30 MB; Q3 is new on all
three corpora. `BLWNS=1` is a larger gain than `BLPVEC=2`'s on all three corpora (on stdlib `BLNSALL=1`
is larger still).

### 79H result

| corpus | base | `BLWNS=1` | `BLWNS=1 BLPVEC=2` | H1 gain | H2 gain |
|---|---|---|---|---|---|
| full enwik8 100 MB | 0.199145 | 0.196841 | 0.196123 | +0.002304 | +0.000718 |
| enwik8 last 30 MB | 0.204666 | 0.202230 | 0.201627 | +0.002436 | +0.000603 |
| repo code 770 KB (obits 24) | 0.185380 | 0.182759 | 0.182558 | +0.002621 | +0.000201 |
| E. coli 4.6 MB | 0.240340 | 0.240340 | 0.240341 | +0.000000 | −0.000001 (exploratory) |

**H1 FAIL** (DNA +0.000000 < 0.0005; it holds on the three text and code corpora). **H2 PASS**
(+0.000718 / +0.000603 / +0.000201; the code margin over 0.0002 is about 1e-6, a marginal pass).
**H3 PASS** (worst case `BLWNS=1 BLPVEC=2` on DNA, 0.000001 worse than base; neither tested arm is worse
by more than 0.0002). **Both pre-registered readings need H1, so neither fires, and no default is
recommended by this section. Any decision to adopt `BLWNS=1`, with or without `BLPVEC=2`, is post hoc.**

Corrections to the 79H text: `data/ecoli.txt` has no newlines and only A/C/G/T, so its one letters-only
"word" is the whole 4,641,652-byte file, not whole lines. Full enwik8 contains the 79A first 30 MB (30 %
of its bytes), so it is not held out; the clean held-out evidence is the enwik8 tail, repo code and DNA
(repo code shares 0/500 sampled 100-byte windows with the earlier code corpora). In compressed size, full
enwik8 goes from 19.91 MB to 19.68 MB (`BLWNS=1`) and 19.61 MB (both). Outputs `_79h_*.txt`.

### 79R — red-team of the engine result (post hoc)

- **Why DNA is exactly zero.** With no non-letter byte, `word_hash` never resets and becomes a rolling
  hash of the whole file prefix; 99.1 % of byte-level keys are distinct, word-table counts never exceed 4,
  and the nonstationary rule's condition was met 0 times in 37.1 M bits (text: 0.155 per bit). `BLWNS=1`
  is identical to base to 6 decimals by construction; `BLPVEC=2` sees a fresh random vector on 99.4 % of
  bytes. The H1 failure records that the gate included a corpus where the flag cannot act, not that
  forgetting fails or harms DNA statistics. Post hoc, on 6-mer-tokenised DNA (not held out) the rule does
  act (0.128 discounts per bit) but gains only +0.000107, about 24× less than text at the same size.
- **Not a memory effect.** The vector table is 2^SBITS entries (302 MB at the default 22). At matched or
  better memory for the baseline:

| corpus | `BLWNS=1` obits 25 | `BLWNS=1` obits 26 (+4.1 GB) | `+BLPVEC=2` SBITS 16 (+4.7 MB) | SBITS 18 (+19 MB) | SBITS 22 (+302 MB) |
|---|---|---|---|---|---|
| corpus_big | 0.215843 | 0.215714 | 0.215634 | 0.215567 | 0.215536 |
| stdlib | 0.125273 | 0.125206 | 0.125036 | 0.125015 | 0.125011 |

  A 4.7 MB vector table beats doubling every count table, and keeps 68 % / 90 % of the full gain.
- **Real, decodable compression.** Every update uses only already-coded bits (code audit; a flipped later
  byte leaves all earlier probabilities bit-identical). A copy with a 32-bit arithmetic coder encoded and
  decoded all of corpus_big back to identical bytes: `BLWNS=1` 2,384,018 bytes, `BLWNS=1 BLPVEC=2`
  2,380,617 bytes — 3,401 bytes saved against 3,393 predicted. Decoding needs the same binary or pinned
  floating-point build; a portable decoder is not guaranteed.
- **Speed.** `BLWNS=1` has no measurable cost; `BLPVEC=2` costs +1.3 % encode / +1.5 % decode in a paired
  measurement (grid wall times are ±5 % noise from concurrent runs).

### 79B — a responsibility-trained mixture-of-experts gate (`attnmoe`, pre-registered in the `_bind_probe.py` docstring before the §79 smoke and grid, not in this ledger) and its amendment (`attnmoe_fc`, `attnmoe_fcnb`)

Design (from the §78 red-team): every non-empty slot is a vote expert; a learned gate prior g over slots;
votes combined in probability space with within-byte posterior weights, so the product over a byte equals
Σ g_k P_k(byte) exactly; the gate trained by g_k − r_k from byte-end responsibilities, never through the
mixer weight. Checks before any grid: 60-digit finite differences of the gate credit (worst 1.6e-12 across
the three arms), exactness of the mixture (6e-15), a one-hot oracle gate reproducing the §78B cue-only gain
within 0.03 bit, causality flips, bit-identity of all 27 older arms, and an independent reviewer
re-implementation. Outputs `_79_probe.txt`, `_79b_probe.txt`. Gain in bits per outcome word vs baseline
(bound ≈ 3).

| arm | G=2 s7 | G=2 s8 | G=12 s7 | G=12 s8 | G=24 s7 | G=24 s8 |
|---|---|---|---|---|---|---|
| `attnorc4` (forced cue + 3, reference) | +1.800 | +1.884 | +0.399 | +0.524 | +0.503 | +0.569 |
| `attnmoe` (registered) | −0.094 | −0.056 | −0.044 | −0.031 | −0.019 | −0.032 |
| `attnmoe_uni` (uniform gate control) | −0.023 | −0.020 | −0.187 | −0.181 | −0.254 | −0.274 |
| `attnmoe_fc` (amendment: full-weight counts) | −0.040 | −0.056 | −0.042 | +0.010 | −0.022 | −0.027 |
| `attnmoe_fcnb` (amendment: full counts, no gate bias) | −0.005 | −0.013 | −0.123 | −0.129 | −0.097 | −0.134 |
| `attnmoe_uni_fc` (amendment control) | −0.032 | +0.002 | −0.075 | −0.070 | −0.091 | −0.110 |

**Registered verdict: M1–M4 FAIL in every cell; kill rule fired.** The builder and the reviewer both flagged
it before the grid. In the reviewer's run (G=24 seed 8) the gate locked within ~25 training sentences onto
one fixed distractor word ("golf", weight ≈ 0.99) whatever the query, because the gate bias alone selected
it and responsibility-weighted counts starved the cue's own vote cells. In the grid the max gate weight was
0.989–0.998 at G=12/24 and the cue's gate prior was < 0.0001 in all four cells.

**Amendment (labelled; designed after that diagnosis).** Its criteria (M1b–M4b and the kill rule, word for
word as in `_bind_probe.py`, which adds one clarifying line on M2b's definitions) were fixed in the
orchestrator's brief at 17:09:55, before the first §79 grid result (17:12:29). They were inserted into
`_bind_probe.py` at 17:16:18, five seconds after the §79 verdict, and before the §79b smoke (17:31) and grid
(17:55). Only the session transcript timestamp shows that they came before the grid results. Already run
before the brief: the §79 smoke (attnmoe on the first 20 test sentences of G=12 seed 7), K1 (300 training
sentences) and K3 (baseline and the oracle gate on the full G=12/24 test sets). **M1b–M4b FAIL for both arms;
kill rule fired.** Full-weight counts stopped the single-word collapse, but `attnmoe_fc` gave the max gate
weight to "golf" or "india" in 73.5–99 % of G=12/24 outcomes ("then" 19–25 % at G=12; mean max weight
0.32–0.42), and `attnmoe_fcnb` ranked the cue first in 30–36.5 % of G=12 outcomes (0 % at G=24) while giving
it only 0.033–0.054 of the weight (uniform 0.031). In the amended arms and both uniform-gate controls, the
mixer weight on the combined vote at the end of training ends negative in all six cells (−0.75…−2.05; the
registered arm −0.22…−0.69): a probability-space mixture over 32 mostly-distractor experts is a signal the
mixer learns to oppose, so a correct cue never gets enough weight to pay.

**Reading, scoped.** Across §77.4, §78B and §79B, the store works (a hand-given cue slot recovers ~2.5 of 3
bits at 24 words), and every learned way of choosing that slot tested so far failed: softmax attention,
cosine attention, a responsibility gate, and that gate with its two diagnosed loops removed. All of them
built the query from slot 0, which holds "then" at 100 % of outcome first bytes, so the query was the same
for every sentence at the deciding byte. The closure covers these gates and this query construction at this
data scale, not learned selection in general.

### 79.6 Where this leaves the strain

- **The one production improvement of the §72–§79 arc is plain forgetting in the word models**, with the
  learned prefix vector adding a smaller, memory-independent amount: full enwik8 0.199145 → 0.196123
  (−1.5 %), verified decodable. It came from the §78 red-team asking why the instrument's vector helped,
  not from the intelligence track's own thesis. The pre-registered adoption rule did not fire (H1 failed on a
  corpus where the flag cannot act), so turning either flag on by default is a post-hoc decision for the
  project owner, made in a separate, reviewed commit.
- **The instrument's cumulative counters are a weak rail** (§78.2); nonstationary counters are required in
  any future instrument claim.
- **Learned long-range selection is closed for the tested gates.** A different query source (not slot 0)
  is the only untested lever the audit identified.

---

## 81. The simplest selector — learned selection binds, on the probe and (lean form) on real corpora (`wstate.py` v10 `sel` arms, `_bind_probe.py --grid81`)

§79.6 left one untested lever (a query not taken from slot 0) and two loose ends. This section
builds the selector the diagnoses pointed to, fixes all four §77.4/§78B/§79B causes **by
construction**, and tests it at probe scale and on real corpora.

**The mechanism** (word mode, S=32, sentence-scoped candidates — a 32-deep LRU otherwise lets
the previous sentence's cue pollute its own vote cells with foreign outcomes; measured 183
updates for 37 own sentences):
1. **Vote store:** every word owns cells keyed (word, 3-byte context, phase, partial byte),
   trained FULL-WEIGHT for ALL candidates every bit (the §79b `fc` lesson; the chicken-and-egg
   of §77.4 dies because unselected words' cells train anyway). Flat bounded table (2^22×f64).
2. **Query-free gate:** `e_k = 4·u[(3-byte ctx, w_k)] − 0.05·k`, softmax T=2, top-4. The
   usefulness `u` is an EWMA of each candidate's own log-likelihood ADVANTAGE over the
   candidate mean, keyed by the SAME 3-byte context the cells key on (a global per-word score
   measurably rewards merely-predictable words; a 1-byte context key mixes every word-start
   byte and the separation dies — both diagnosed on smoke runs).
3. **Readout:** the vote `stretch(Σ w_t p_t)` is read through a **CONTEXT-SELECTED weight**
   (per (prev_byte, phase, partial) cell, own RMSProp) added to the mixer dot — never one
   global mixer weight shared with the ~99% of bits where the vote is irrelevant (§78B's
   dilution/reversal). Measured +1.97…+4.1 at deciding contexts, never negative.
4. **No responsibility loops:** direct per-candidate advantage updates only.

**Probe (pre-registered; criteria in `_bind_probe.py` before any run; identical protocol to
§77/§78B/§79):** `sel` gains **+2.336/+2.291 (G=2), +2.043/+2.014 (G=12), +1.677/+1.596
(G=24)** vs the orders baseline — cue in the vote set **200/200** at every gap and seed.
C1 PASS, C2a PASS, C2b PASS (`selpos`, same readout with a position-only gate: −0.04…+0.02 at
G=12/24 — **the gate carries the binding**). C3 FAILed as designed and the verdict tree then
printed "KILL (C2 first clause failed)" — a branch whose premise is false (C2 passed in every
cell); the tree had no branch for headline-passes/oracle-control-fails. Post-hoc diagnosis,
labelled: `selorc4`'s gate uses position-only logits, so its forced cue carries 0.045–0.076
gate weight vs `sel`'s 0.17–0.28 — the control reproduces the §78B dilution **by
construction** (it tested dilution, not the readout) and its failure *confirms* that weight
concentration, not mere inclusion, is what pays. **First learned selector in the project's
history that binds distant content** (§85.1: C3 was a mis-specified control, and the continuation was taken under an unregistered branch) — after softmax attention (§77.4), cosine attention, and
two responsibility gates (§79B) all failed.

**Corpus (pre-registered B1–B3 in `wstate.py` before any corpus run; WNS=1 everywhere per the
§78.2 mandate — the PAQ nonstationary rule on the order tables; the no-binding control is
`sel@WSLOTS=1`: the SAME machinery with candidates = the last word only):**

| corpus (sizes KB) | sel − sel@1 (B1, binding) | sel − baseline-NS (B2) | sellean − baseline-NS |
|---|---|---|---|
| code (100→585) | +0.0009…+0.0077 (2/4 sizes ≥ 0.001) | −0.008…−0.016 | −0.002…−0.009 |
| wt103 (150→2700) | +0.008…+0.027, growing | −0.0003…−0.009 | −0.0001 / −0.0026 / **+0.0039** |
| stdlib (400→2400) | +0.011…+0.022, growing | −0.012…**+0.0003** | −0.0057 / **+0.0008 / +0.0041** |

- **B1 PASSES on all three corpora** — the binding signal is real on real data, and grows
  with size (the `sel` arms carry the M=32 EMA + bucket expert inherited from §77-arm
  comparability; `sel@1` carries it too, so B1 is unaffected).
- **B2 as pre-registered FAILS** (the composite sits above the fair rail) — but the
  decomposition is clean: the inherited state overhead costs ≈ what binding pays. A **lean**
  selector (post-hoc, labelled: the same channel without the EMA/bucket block) **crosses
  below the nonstationary rail on stdlib at 1.2/2.4 MB (+0.0008/+0.0041) and on wt103 at
  2.7 MB (+0.0039)**, margins growing with data; code stays negative. `sel` beats `selpos`
  everywhere (B3 holds).
- **Seed replication of the lean crossings:** seeds 1 and 2 return numbers BIT-IDENTICAL to
  seed 0 (stdlib 2.0822/2.1171; wt103 2.2822) — the lean arm contains no seeded randomness
  (deterministic per-word vector init, zero-init weights, deterministic updates), so the
  crossings are exact reproducible values, not seed averages; the `sel`/`sel@1`/`selpos`
  comparisons that DO involve seeded state remain single-seed.

**Where this leaves the strain.** Selection — the bottleneck every §77–§79 gate died on — is
solved at probe scale by (cells-for-all, query-free contextual usefulness, context-selected
readout, sentence scoping), and the same mechanism carries a measurable, growing binding
signal on real corpora that a stripped form converts into a net held-out win against a fair,
non-forgetting rail on 2 of 3 corpora. The port rule still gates the engine: the next step is
a lean-channel port to `strong.rs` and the beats-strong test at 11 MB+, with the §78.3 lesson
applied (all three mixers, not the global one). If the engine's word-model family absorbs it
(§75's law), the corpus margins say where the ceiling is; if not, this is the strain's first
engine-bound learned-memory channel.

---

## 82. The lean selector ported to the engine — it beats strong itself (`strong.rs` `BLSEL=1`)

The §81 lean channel ported per the pre-named design: sentence-scoped LRU of completed words
(depth `NSELSLOTS`, default 32), full-weight vote cells for ALL candidates (flat 2^22 table keyed
(word, 3-byte ctx, phase, partial)), the query-free contextual-usefulness gate (2^22 table,
same hyperparameters as §81), and the vote fed as **one input to ALL THREE mixers** — the §78.3
placement; the engine's context-selected mixers are its analog of §81's own context-selected
weight. Default OFF = bit-identical (0.231368 @300 KB re-verified after the port).

**Port-rule A/B (whole-stream, obits 25; baselines = the §80 defaults `BLWNS=1 BLPVEC=2`):**

| corpus | baseline | `BLSEL=1` | gain |
|---|---|---|---|
| corpus_big 11 MB | 0.215536 | **0.215499** | **+0.000037** |
| corpus_big, `NSELSLOTS=16` | — | 0.215491 | +0.000045 |
| stdlib.bin 10.6 MB | 0.125011 | 0.125021 | −0.000010 (tie) |
| enwik8 first 30 MB | 0.202723 | **0.202645** | **+0.000078** |

- **The port rule is met on 2 of 3 corpora** (both text; stdlib ties): the first learned
  long-range memory channel in the project's history that adds net compression to the
  production engine — every §72–§79 channel was inert or negative there. At 300 KB it already
  gains +0.000076 on top of the §80 defaults.
- Honest scope: the margins are ~5–10× smaller than `BLWNS`'s; single run per config; the
  engine's whole-stream online setting differs from the instrument's copy-off decontaminated
  gate (the per-corpus pattern inverts: stdlib was the instrument's best, ties here — the
  engine's word models + match absorb part of the channel's instrument value online).
  Default stays OFF pending the owner's adoption decision (the §80 precedent, in a separate
  reviewed commit); the flag is validated, documented, and cheap (~5–15 % wall time).

**The arc §77→§82 in one line:** audit found the §72 channel was a word model and the port
trained uphill; §78 isolated what actually helps (forgetting, selection-not-storage); §79
banked the forgetting as an engine default; §81 solved selection at the instrument (the first
binding gate, and a lean form that beats a fair rail on real corpora); §82 ported it and it
beats the engine itself. The strain's first learned-memory channel is now engine-bound.

---

## 83. Replication, sweep, and the scope discovery — the selector adopted (`SELSENT=0`, `NSELSLOTS=16`)

Three pre-named follow-ups, run to completion. Baselines = the §80 defaults; gains below are
whole-stream bits/bit ×10⁻⁶ vs each corpus's baseline.

**(1) Held-out replication** (enwik8 **last** 30 MB — data untouched by every earlier decision;
its default run reproduces 79H's 0.201627 exactly): `BLSEL=1` +0.000048 (0.201579);
`NSELSLOTS=16` +0.000064 (0.201563). **The §82 port result replicates on held-out data.**

**(2) Engine-scale sweep** (corpus_big; knobs env-ized, defaults bit-identical before the runs):
sentence scope and the §81 hyperparameters are good but not optimal — `SELPBD=0.02` +0.000043,
`SELUGAIN=8` +0.000040, `NSELSLOTS=16` +0.000045, `SELUDC=0.999` −0.000004 (worse),
`SELTSC=1`/`NSELSLOTS=64` neutral. Interactions are small (best pair `UG8+N16` +0.000047).

**(3) The scope discovery — `SELSENT=0` (never clear; whole-stream 16/32-word LRU) beats
sentence scope everywhere on text**: corpus_big 0.215480 (+0.000056) vs sentence scope's
0.215499. §81's sentence scoping was a patch for the *pre-contextual* usefulness design; with
u keyed by (3-byte context, word), stale words' cells do pollute — but their advantage sinks and
**the gate demotes them on its own**. The binding scope is the whole stream; cross-sentence
entity continuity is part of what the channel now captures.

**Final configuration validated cross-corpus** (`NSELSLOTS=16`, `SELSENT=0`):

| corpus | baseline | final config | gain |
|---|---|---|---|
| enwik8 first 30 MB | 0.202723 | **0.202601** | **+0.000122** |
| enwik8 last 30 MB (held out) | 0.201627 | 0.201563 | +0.000064 |
| corpus_big 11 MB | 0.215536 | 0.215491 | +0.000045 |
| stdlib 10.6 MB | 0.125011 | 0.125012 | tie (−0.000001) |

**Adopted as the strong.exe defaults** under the owner's §80 standing directive (`BLSEL=0`
recovers the pre-§83 engine bit-identically; new 300 KB reference 0.231282): the engine now
carries a learned long-range binding channel by default. Honest scope: text corpora gain
+0.00005–0.00012 (0.02–0.06 %), code ties — the engine's word models plus match own code, as
the instrument predicted (§81: code's binding value was real but the engine's word-family
absorbs it). Cost ~5–15 % wall time. Full-enwik8 headline with the new defaults (obits 25, 68 min):
**0.196021 bits/bit (19.60 MB)** — the adoption chain reads 0.199145 (§79 base) → 0.196841
(`BLWNS`) → 0.196123 (`+BLPVEC`) → **0.196021 (`+BLSEL` N16 SENT0)**; −1.6 % total from the
forgetting-and-binding arc alone.

---

## 84. What the channel actually is — the bigram revision, and the re-adoption that followed (`_what_binds.py`, `NSELSLOTS` controls)

Three questions, one answer that revised §82–§83.

**A. The knob sweep plateaued.** At the §83 config (N=16, whole-stream scope), every knob sits
within ±0.000005 on corpus_big (TOPM 2/8, VORD 2/4, N 8/24, tables 23) — the adopted settings
looked near-optimal. On enwik8-30MB only the bigger tables moved anything (b23 0.202561 vs
0.202601).

**B. The what-binds probe** (instrument `sellean`, trained 1.2 MB wt103, frozen, 11,718 sampled
selections on held-out text): the gate's **top pick is the preceding word itself** in almost
every sample ("on → on", "Zaza → zaza", "starred → starred"); second/third picks carry
relational structure ("in → cast, was"; "role → starring, guest"); aggregate stats barely move
(recurrence@40w: 27.3% selected vs 26.1% random vs 28.2% most-recent). The channel had largely
learned a **context-enriched word-bigram vote**: the previous word's cells at the 3-byte
context predict what follows it.

**C. The controls confirmed the probe — and overturned the adopted depth** (corpus_big, over
the no-selector baseline 0.215536): `NSELSLOTS=1` +0.000164 (0.215372), **`N=2` +0.000267
(0.215224)**, N=3 0.215319, N=4 0.215398, N=16 +0.000045 (the §83 config). The value is at
depth ≤ 2; deeper candidates DILUTE (N=16 loses 0.00015 to N=2). §82/§83's depth-16 "binding"
channel was mostly dilution around a shallow vote.

**Re-adoption (`NSELSLOTS=2`, `SELVBITS=SELUBITS=23`), validated on all four corpora:**

| corpus | no selector | N=2 + b23 | gain |
|---|---|---|---|
| corpus_big 11 MB | 0.215536 | **0.215120** | **+0.000416** |
| enwik8 first 30 MB | 0.202723 | 0.202400 | +0.000323 |
| enwik8 last 30 MB (held out) | 0.201627 | 0.201343 | +0.000284 |
| stdlib 10.6 MB (code) | 0.125011 | **0.124701** | **+0.000310** |

Uniform ~+0.0003 everywhere — including **code, which tied at N=16**: the dilution had been
hiding the code gains too. New 300 KB reference 0.230863. `BLSEL=0` still recovers the
no-selector engine bit-identically.

**The honest revision, stated plainly.** The production value of the §81–§83 channel is not
long-range binding; it is **a learned (word × 3-byte-context) vote read through
context-selected weights** — a better word-bigram than the engine had (`wdc2` lacks the middle
context; `wdc` keys only the current prefix). The engine's own word family absorbs the deep
binding the instrument measured (§81's B1 stands as instrument truth; §75's absorber law
refined: what the engine lacked was never binding capacity but this shallow vote). The
instrument-first, probe-then-control method did exactly its job: the interpretability probe
predicted the N-controls before they ran. Full-enwik8 headline with the corrected defaults (obits 25, 82 min):
**0.195780 bits/bit (19.58 MB)** — the corrected channel contributes −0.000343 on the full file
(3.4× the N=16 config's −0.000102). The adoption chain now reads 0.199145 (§79 base) →
0.196841 (`BLWNS`) → 0.196123 (`+BLPVEC`) → 0.196021 (`+BLSEL` N16, §83) → **0.195780
(`N=2` + b23, §84)**; −1.7 % total from the forgetting-and-vote arc, every step online and
adopted only after beating the previous default on held-out data.

---

## 85. Verification of §81–§84, and the few-shot online-binding probe (`_fewshot_probe.py`)

Two steps, both run by independent agents with their own scripts (scratch: `rt85/`, `fs85/`, `fs85rt/`);
every number below was recomputed from the raw records by a second agent.

### 85.1 §81–§84 verified

- **Decodability and causality of the adopted selector (`BLSEL=1`, N=2, tables 2^23).** A copy of the
  current `strong.rs` with a real 32-bit arithmetic coder encodes and decodes all of `corpus_big` back to
  identical bytes: default 2,376,016 bytes, `BLSEL=0` 2,380,617 bytes — **the channel saves 4,601 real
  bytes (0.19 %)**, against 4,594 predicted from the model cost. Probability dumps under two byte flips
  (letter→space inside a word; space→letter merging two words) are bit-identical through the flipped bit
  and first differ at the very next bit. Code audit found no read of the current byte before it is coded.
- **`BLSEL=0` recovers the pre-§82 engine**, verified against a rebuild of commit `8b23201`'s source
  (0.231368 / 0.208290 at 300 KB, identical), not only against the documented number.
- **Cost:** +192 MB peak working set (exactly the two 2^23 f64 tables) and +5 % wall time (paired runs).
- **The instrument depth control §84 never ran** (lean selector, `WNS=1`, word mode, same rail as §81):

| slots | wt103 450 / 1200 / 2700 KB | stdlib 400 / 1200 / 2400 KB |
|---|---|---|
| 1 | 2.1552 / 2.2197 / 2.3056 | 2.0691 / 2.0999 / 2.1367 |
| 2 | 2.1522 / 2.2142 / 2.2955 | 2.0621 / 2.0937 / 2.1288 |
| 4 | 2.1483 / 2.2068 / 2.2876 | 2.0611 / 2.0886 / 2.1221 |
| 32 (§81) | 2.1445 / 2.2020 / 2.2822 | 2.0593 / 2.0822 / 2.1171 |
| rail (no selector) | 2.1444 / 2.2046 / 2.2861 | 2.0536 / 2.0830 / 2.1212 |

  In the instrument, depth helps at every step and only depth 32 crosses the rail. §81's real-corpus binding
  claim stands at instrument scale; §84's revision is correct for the engine, whose word models and match
  model already cover the deep part. Both hold at their own scale.
- **§81 probe verdict audit.** All gains, C1–C3 and the 200/200 cue-in-vote-set figures recompute exactly.
  The registered verdict tree had three branches and none matched the realised outcome (C1, C2a, C2b pass;
  C3 fail); the code's catch-all printed "KILL (C2 first clause failed)" on a false premise. The override's
  mechanism claim is verified (`selorc4`'s gate weight on the forced cue equals the position-only softmax to
  four decimals, 0.045–0.076, so it tested §78B's dilution, not the readout). Correction to §81's wording:
  C3 was a **mis-specified control, uninformative about the readout**, not a control that "failed as
  designed"; the continuation was taken under an unregistered branch. The binding claim rests on C2a/C2b,
  which are independent of C3.

### 85.2 Few-shot online binding (pre-registered in `_fewshot_probe.py` before any run; run on the §81 instrument snapshot, sha `4f0b9ec6…`)

Question: after a never-seen cue word first appears bound to a never-seen outcome word, how many
exposures does the §81 selector need to use it, learning online in one pass? Training as §81 (8 known
cues, 1200 sentences); test stream of 320 sentences from 16 cues (8 known + 8 new, each new cue ~20
times), scored online; exposure index k = the cue's earlier occurrences in the test stream. Output
`_85_fewshot.txt`, records `fs85/`.

| gap | new-cue gain, mean over k = 5…19 (seeds 7 / 8) | exposures-to-bind (≥ 1.5 bits sustained) | known-cue gain in this stream vs §81 |
|---|---|---|---|
| 24 words | +0.70 / −0.77 | none | +0.51 / +0.41 vs +1.68 / +1.60 |
| 12 words | (secondary) | 10 / 11 | +1.1…+1.3 vs +2.04 / +2.01 |

**Registered verdict at G=24: F1 FAIL, F2 FAIL, F3 FAIL.** The red-team recomputed every figure and then
established what each failure means:

- **Storage is immediate.** The new cue's own vote cell predicts the outcome's first byte at 8.0 bits on
  first sight, 1.8 bits after one exposure and 0.16 bits from ten exposures on. The binding is in memory
  from exposure 1.
- **Selection is slow.** The gate gives a new cue about half a known cue's share of the vote (weight 0.071
  vs 0.171; mixture share 0.22 vs 0.48) and does not catch up within 20 exposures at 24 words. F2's
  registered threshold (gate ≥ 0.3) was mis-calibrated: fully bound §81 cues sit at 0.17.
- **The early "gains" are hedging, not binding.** At k = 0–2 the whole-word gain is +6.6 bits on the first
  byte (the baseline pays for a never-seen letter; sel's orders are less confident) and −4.4 bits on the
  rest of the word (sel spells new words worse at every k). Decomposed by a counterfactual with the vote
  removed, 40–85 % of the per-k first-byte gain is that hedging.
- **F3's interference is real and its channel is the shared readout.** A control with the known 8 cues only
  and the same 320-sentence length restores +1.70 / +1.64 (F3 would pass), so the drop is caused by the new
  cues' presence. In the 16-cue stream the known cues' gate weight (0.171), vote-set membership (164/164),
  own-cell accuracy (0.014 bits) and vote mixture (1.34 vs 1.26 bits) are all unchanged; what fell is the
  **context-selected readout weight** shared by every word at the deciding bits (phase 5: 0.06 vs 0.84).
  New-cue sentences produce confidently wrong votes (4.9 bits), the shared RMSProp weight is driven down,
  and re-reading the 16-cue stream's votes through the control's weights recovers 88 % of the 1.22-bit loss.

**Plain-language statement.** The machine writes a new fact into memory the first time it sees it, and
that memory is accurate within a few exposures. What it cannot yet do is trust the new fact quickly, and
its one shared "how much to trust memory" dial per bit position means unreliable new memories turn the
dial down for the reliable old ones too: adding eight unknown pairs cut the known pairs' benefit from 1.7
to 0.5 bits. Storage works; selection is slow; **the shared readout is the interference channel.**

### 85.3 What this fixes in the record, and the next lever

- §81's wording on C3 is corrected above; §81's binding claim and §84's engine revision both stand.
- The lever the probe points to is structural and cheap: **per-word or per-reliability readout
  weighting** instead of one dial shared by every word at a context (e.g. the readout weight keyed by the
  selected word's own usefulness bucket, or a reliability-gated vote). It is the first cause since §81 that
  the depth story does not explain, and it is pre-registerable on this probe: the known-cue gain in the
  16-cue stream must return to within 0.3 bits of the 8-cue control, and exposures-to-bind at 24 words
  must become finite.
- Registration lessons: calibrate thresholds against the instrument's own known-good values before
  registering (F2), and never use whole-word gain on never-seen words as a binding measure (F1).

---

## 86. The §85 lever tested, two structural bugs found, and the memory's real constraint (`wstate.py` v11–v12, `strong.rs` §86 flags, `_fewshot_probe.py --grid86`, `_scale_probe.py`)

An eight-hour continuous session on the intelligence path. Everything below was built with criteria written
before the runs, self-tested, and attacked by independent red-team agents; two of the session's findings are
defects in configurations earlier sections treat as settled.

### 86.1 Two structural bugs in the standing record

1. **`sellean` — the arm that produced §81's real-corpus crossings — has no learned selection.** When it was
   created (commit `8b23201`) two `self.arm == "sel"` guards were left in place, so the lean arm never writes
   the usefulness table: `len(suw)` is 0 after 20 KB where `sel` has 157,769 keys, and `sel_gk` is None. It is
   `selpos` minus the state block: a position-only recency gate with a context-selected readout. Consequence:
   §81's "the lean selector converts binding into net held-out wins" (stdlib +0.0008/+0.0041, wt103 +0.0039)
   was measured on a channel with **no learned selection**, and §85.1's depth control inherited the same arm.
   The arm §81 described is added here as `selleanu` (= `sel` without the EMA/bucket block); §86.3 measures it.
2. **The shipped engine has had no bias input to its context-selected mixers since §82.** `nin = NWMAX` makes
   `sts[nin] = 1.0` write the index the selector vote then overwrites on every bit, while index `NIN+PVD` — the
   slot §79's own header documents as the bias — is never written and stays 0.0 for the whole run, with zero
   gradient in all three weight sets. Verified by instrumentation (`bias_index_held_1.0 = 0` of 2,400,000 bits;
   `sum|mixers[.][33]| = 0.0`). The final mixer keeps its own bias, so what was lost is the **per-context**
   offset. Fixed behind `BLSELBIAS=1` (default OFF): +0.000013 at 11 MB, −0.000046 at 300 KB, with last-20 %
   improving in 6 of 6 arms — a warm-up cost repaid only on long streams. Small, but the shipped default has
   been computing a dot product with one hard-zero input since §83.

### 86.2 The §85 lever on the probe: both fixes are necessary, and binding takes one exposure

Arms (each = `sel` + one change; all reductions bit-identical to `sel`): `selrb` keys the vote's readout weight
by a 4-way reliability bucket of the top candidate's usefulness instead of sharing one weight across all words;
`selfa` replaces the fixed-rate usefulness EWMA with a count-adaptive rate (running mean early, §81 rate later);
`selrbfa` is both. Registered criteria I1–I3 in `_fewshot_probe.py` before any run; a registered `selrs`
(reliability-*scaled* vote) proved unrunnable — a closed loop where the scale is zero until the weight trains and
the weight cannot train while the scale is zero — and is recorded as registered, vote-inactive, with a dated
amendment adding a bootstrap-floor variant.

**Registered verdict: `selrbfa` and `selrs2fa` pass all four criteria; `selrb` alone fixes interference only;
`selfa` alone fixes speed only.** But the red-team then falsified the headline and the metric:

- **The registered metric does not measure binding.** It scores the whole vote, so an arm can pass while the
  bound cue's own contribution is negative: under `sel` the cue contributes −0.53…−0.92 bits at every exposure
  where it is in the vote set, while I2a reads +1.075 (a pass). This is §85's own registration lesson recurring.
  On the corrected **cue-attributable** metric (a serve-time counterfactual removing the cue from the mixture):

| arm | cue contribution at k=1 | at k=10 | exposures to a useful binding |
|---|---|---|---|
| `sel` | 0.000 | −0.526 | never |
| `selrb` | 0.000 | −1.122 | never |
| `selfa` | −1.151 | +2.058 | 6 |
| **`selrbfa`** | **+3.665** | **+8.646** | **1** |

- **"Binds from the first exposure" is false.** At k=0 the cue contributes exactly 0.000 bits on 16/16 events
  across both seeds; it is not in the vote set and its cell is empty. The honest claim is **one exposure**.
- **What the k=0 gain actually is (a separate capability).** The machine still saves 5–7 bits at first sight,
  entirely from non-cue candidates: trained only on familiar material, it has learned that when nothing in
  working memory is trusted for this position the familiar continuation is *wrong*, and the bucket-0 readout
  weight is negative on exactly the letter-discriminating bits (mean −0.61…−1.62 at bits 3–7, against +1.34…+3.74
  under bucket 3). It fires before any novel outcome byte has been coded. That is **novelty detection**, not
  memory, and it would fire identically if every new answer were swapped.
- Interference is fixed: the known cues' benefit is the same in the polluted and clean pools (|d| 0.008/0.129
  against a 0.3 tolerance), where `sel` loses 1.65 bits. `selrbfa` removes 98.6 %/98.8 % of the rail's
  outcome-word cost — ceiling saturation, not open-ended headroom.

### 86.3 The same fixes on real corpora: fast trust generalizes, reliability bucketing does not

Lean arms, standard gate, nonstationary rail (§78.2), seed 0. Gain vs the rail in bits/byte:

| arm | wt103 450/1200/2700 KB | stdlib 400/1200/2400 KB | code 100/200/400/585 KB |
|---|---|---|---|
| `sellean` (§81's arm, no learned selection) | −0.0001 / +0.0026 / +0.0039 | −0.0057 / +0.0008 / +0.0041 | −0.0087 / −0.0032 / −0.0038 / −0.0022 |
| **`selleanu`** (learned selection restored) | **+0.0025 / +0.0056 / +0.0073** | −0.0044 / +0.0025 / **+0.0062** | −0.0090 / −0.0027 / −0.0026 / −0.0013 |
| `selleanfa` (+ fast trust) | +0.0033 / +0.0051 / +0.0060 | −0.0038 / +0.0033 / **+0.0060** | −0.0084 / −0.0016 / −0.0023 / −0.0015 |
| `selleanrb` (+ reliability buckets) | −0.0013 / +0.0029 / +0.0042 | −0.0203 / −0.0069 / −0.0004 | −0.0322 / −0.0224 / −0.0174 / −0.0117 |
| `selleanrbfa` (both) | +0.0012 / +0.0041 / +0.0040 | −0.0103 / −0.0024 / +0.0028 | −0.0270 / −0.0129 / −0.0093 / −0.0062 |

- **Learned selection does pay on real text**, and by more than §81 claimed with the wrong arm: `selleanu`
  beats `sellean` at every size on all three corpora, and nearly doubles §81's headline wt103 crossing
  (+0.0073 vs +0.0039 at 2.7 MB).
- **The probe's decisive fix is probe-specific.** `selrb` is the worst arm on both corpora. The engine agrees:
  in `strong.rs` the same mechanism is worth only +0.000023, and a retune of its thresholds to the engine's
  measured quartiles made it *worse* at every setting tried (the best split is one large coherent trusted class,
  not an even partition). Splitting a shared readout weight helps when the stream is half unfamiliar by
  construction, and costs training signal when it is not.
- Code never crosses the rail, as in §81.
- On wt103 every selector arm crosses at 1.2 MB and above, and `selleanfa` is second only to
  `selleanu` (+0.0060 vs +0.0073 at 2.7 MB) — fast trust is the one §86 fix that helps on all three corpora.

### 86.4 The engine's memory is collision-bound, not capacity-bound

The adopted vote table is an untagged hashed table at 2^23. Growing it alone (no other change):

| vote cells | corpus_big gain | enwik8 30 MB gain |
|---|---|---|
| 2^23 (adopted) | +0.000416 | +0.000323 |
| 2^26 | +0.000865 | +0.001025 |
| 2^28 | +0.001094 | +0.001794 |
| 2^29 | — | +0.002101 |

No saturation at 8.6 GB, and the usefulness table is irrelevant (growing it alone changes nothing; vote 2^25 with
usefulness 2^23 equals both at 2^25). With capacity free, the 3-byte context key still beats 2-byte and 4-byte,
so the key's resolution was never the issue.

**Most of that appetite is collision damage.** An 8-bit tag with evict-on-mismatch (`BLSELTAG=1`, the discipline
the engine's other tables already use) at 2^23:

| corpus | current default | tagged 2^23 | gain | equivalent untagged size |
|---|---|---|---|---|
| corpus_big 11 MB | 0.215120 | 0.214659 | +0.000461 | 2^26 (1074 MB vs 142 MB) |
| enwik8 30 MB | 0.202400 | 0.201987 | +0.000413 | — |
| stdlib 10.6 MB | 0.124701 | 0.123766 | **+0.000935** | — |

42 % of cell reads evict at 2^23, and evicting beats merging: colliding keys were *poisoning* the vote, not
merely diluting it. Tagging buys roughly three address bits; about 60 % of what raw capacity bought was
collision damage and 40 % genuine capacity. Tagged 2^26 (+0.001254 on enwik8) reaches 85 % of untagged 2^28 for
1/67 of the memory. Combined with both §86 fixes, tagged 2^23 gives +0.000551 on enwik8. Default OFF; held-out
validation in §86.6.

**Domain mixing costs the channel a third of its value** and it is not a switching effect: interleaving two
domains every 1 MB (+0.000192) and concatenating them (+0.000203) both lose against either domain alone
(+0.000303 / +0.000312), while the engine's overall compression loses only 1.4 %. Larger tables recover part of
it (66 % → 72 % retained), so it is capacity plus sharing, not switching.

### 86.5 How many facts, how far — and why facts die (the day's unifying finding)

`_scale_probe.py`, registered S1–S4, cue-attributable metric, 40 training exposures per fact: **all four pass** —
`selrbfa` holds 8, 32 and 96 facts at a 24-word gap and binds at gaps of 24, 48 and 96 words, beating `sel` by
+3.0…+5.5 bits in every cell. But the interesting part was the failure mode, and a registered follow-up
(`_lockout_probe.py`, §88 criteria D1–D3) then **overturned the first reading of it**.

**Observed:** facts die all-or-nothing. A dead fact's cue enters the served vote set in 0 of its 20 test
exposures and contributes exactly 0.000 bits. Dead rate 0/8, 0/32, **8/96**, and 1/8 at gaps of 48 and 96.

**First reading (wrong):** a gate lockout — the cell looked perfect (49.5 counts, zero errors) while usefulness
sat negative, apparently self-reinforced through §86.2's bucket-0 "don't trust memory" dial.

**Corrected:** three candidate lockout fixes (optimistic initialisation of unseen trust, forced exploration, and
removing the update gate) **all failed D1, and the diagnostics explain why the premise was wrong**: trust *was*
updating (59 updates for dead and live facts alike), no other context shared the key, and the cue's own memory
was genuinely no better than an average candidate. The "perfect cell" reading had inspected only bit 7, which is
clean for every ASCII word. Dumping all eight bits shows **12 of 12 dead facts sit on a collided cell** carrying
110–250 counts whose majority bit is *opposite* to their own outcome (pair 5, bit 6: 137.8/18.1 where the truth
is 0/59). One corrupted bit costs ~3 bits and erases the fact. The gate was reporting the truth.

**Causal confirmation** (post-hoc, table size the only change):

| cell | vote table 2^22 | 2^24 | 2^26 |
|---|---|---|---|
| 96 facts, gap 24 | 8 dead, 0.795 bits/outcome | 2 dead, 0.196 | **0 dead, 0.076** |
| 8 facts, gap 48 | 1 dead, 0.801 | 1 dead (collision re-lands on the same cell), 0.623 | **0 dead, 0.238** |

Revived facts go from 4.5–7.4 bits to 0.03–0.10, while `train_bpb` barely moves — this is the memory, not general
modelling. The 8-fact row at 2^24 is the control that makes it causal.

**So §86.4 and §86.5 are one finding.** In the engine, 42 % of vote-cell reads collide at 2^23 and evicting beats
merging; in the instrument, every dead fact is a collision. **The memory's limit is addressing, not capacity or
selection, and collisions poison cells rather than dilute them.** §87's "capacity limit / graceful degradation"
reading is corrected: at an adequate table the machine holds 96 of 96 facts at 0.076 bits per outcome.

Caveats recorded: the scaling metric is un-normalised and grows with the number of facts, so its "190 % of the
8-fact value" is not evidence about crowding; the gap axis saturates at ~27 distinct words because the distractor
vocabulary has only 26; the collision test is post-hoc and deserves a registered replication; and the registered
"dead" rule (0 of 20) is brittle — one forced exploration flipped it while recovering no value.

### 86.6 Held-out validation of the engine fix

Data that shaped none of these decisions (the §80/§83 adoption standard):

| held-out corpus | current default | tagged 2^23 | gain | tagged + §86 fixes | tagged 2^26 |
|---|---|---|---|---|---|
| enwik8 last 30 MB | 0.201343 | 0.200969 | +0.000374 | 0.200868 (+0.000475) | 0.200151 (+0.001192) |
| repo code 770 KB | 0.181710 | 0.181123 | +0.000587 | — | — |

Full-file headline with the tagged configuration (`BLSELTAG=1 SELVBITS=SELUBITS=26`, obits 25, 61 min):
**full enwik8 0.194567 bits/bit (19.46 MB)** against the adopted default's 0.195780 (19.58 MB) — +0.001213,
the largest single step of the §79–§86 arc and larger than `BLWNS`'s. The chain now reads 0.199145 (§79 base)
→ 0.196841 (`BLWNS`) → 0.196123 (`+BLPVEC`) → 0.196021 (`+BLSEL` N16) → 0.195780 (N=2+b23, §84) →
**0.194567 (`+BLSELTAG` at 2^26)**.

Cost: 8 MB of tags at 2^23 (64 MB at 2^26), no measurable time. `BLSELTAG=1` is validated, documented and default
OFF; adoption is the owner's decision under the §80 precedent, in its own reviewed commit.

### 86.7 Where this leaves the intelligence path

- **The machine is an online associative memory that learns a new long-range fact from ONE exposure** and, with
  §86.2's two fixes, no longer damages the facts it already holds when unfamiliar material arrives. That is the
  clearest statement of the capability the strain actually has.
- **Its binding limit is addressing, not capacity or selection.** Both the instrument's dead facts and the
  engine's 42 % colliding reads are the same defect, and a tag fixes it for 8 MB (§86.4–86.6).
- **A second capability was found, unlooked-for:** trained only on familiar material, the machine detects that a
  context is novel and inverts its memory readout, saving 5–7 bits before it has any fact to recall (§86.2).
- **Probe results do not transfer by default.** The reliability-bucketed readout is decisive on the probe, the
  worst arm on real text, and near-noise in the engine (§86.3). Fast trust transfers; bucketing does not.
- **Open, in order of value:** (1) a registered replication of the collision finding (§86.5 was post-hoc);
  (2) whether the novelty detector of §86.2 is useful on its own; (3) a fact-level instrument for corpus text —
  §86.8 shows corpus bits/byte can reward the very defect that destroys a specific memory, so binding claims on
  real data need a recall measure, not a compression measure; (4) the §85 lesson, now twice-learned: a metric
  over a whole channel does not measure the part of the channel under test.
  **[§87 ANSWERED (1) and (3), and made (4) a five-times-learned lesson.** The fact-level instrument exists
  (`_factprobe.py`); the collision finding is replicated on real text as a registered forward prediction at four
  doses; and corpus bits/byte does not merely fail to see binding, it moves in the *opposite* direction to
  fact-level memory quality. (2) remains open, and §87 adds a negative note: the real-text version of the
  inverted readout was tested (`negoracle.py`) and is not supported.]**
- **Answered this session:** a better-addressed instrument table does NOT improve the corpus verdict (§86.8).

### 86.8 Collisions destroy specific memories while helping aggregate compression

The §86.7 open question — every corpus number to date was measured on a colliding 2^22 instrument table, so
does fixing the addressing change the corpus verdict? — was run, and the answer is **no, and the sign is
opposite to the engine's**:

| `selleanu`, instrument vote table | 2^22 | 2^24 | 2^26 |
|---|---|---|---|
| stdlib 2400 KB (rail 2.1212) | **2.1150** | 2.1201 | 2.1181 |
| wt103 2700 KB (rail 2.2861) | **2.2788** | 2.2818 | — |

A bigger, less-colliding table is **worse** on both corpora, while on the probe (§86.5) enlarging the same table
revived every dead fact, and in the engine at 11–100 MB tagging is worth up to +0.00094.

These are consistent, and the reconciliation is the day's sharpest statement about this machine. A collision
merges two unrelated (word, context) keys into one cell. For a *specific* memory that is fatal: the cell's
majority bit flips against its owner's truth and the fact is erased (§86.5). For *aggregate* compression on a
2–3 MB stream it is mild backoff: most cells are sparse, and sharing counts across keys smooths the estimate
more than the occasional corruption costs. The regimes separate by data per cell — at engine scale (11–100 MB)
cells have enough evidence that merging only corrupts, which is why tagging pays there and hurts nothing.

**Consequence for the intelligence path.** Compression on a corpus is a poor instrument for the question this
track is about: the same defect that destroys a machine's ability to recall a particular fact can *improve* its
bits-per-byte. Reliable recall of specific facts and aggregate compression are not the same objective, and where
they conflict this section is the evidence. Probe-style, fact-level measurement is the right instrument for
binding claims; corpus bits/byte is the right instrument for production claims; neither substitutes for the other.

---

## 87. A fact-level recall instrument for real corpus text (`_factprobe.py`, `wstate.py` v13)

§86.7's highest-value open item was that **corpus bits/byte cannot answer the binding question**: §86.8 showed
the same collision defect that destroys a specific memory on the synthetic probe can *improve* aggregate
compression, so §81's corpus crossings and §86.3's arm ranking rest on a measure that does not see what this
track is about. §87 builds the missing instrument — recall of *specific facts*, measured on natural text — and
uses it to turn §86.8's post-hoc reconciliation into a forward, falsifiable prediction.

Everything below was registered before it was run, attacked by an independent four-lens red-team before the
grid (which rewrote the metric), and attacked again by a second four-lens red-team before it was written here
(which withdrew three of the first draft's headline sentences). Both attacks are recorded in §87.6.

### 87.1 What a "fact" is on real text, and what the machine can structurally hold

A **fact** is a word-type pair (A,B) with count(A), count(B) ≥ 5, at least 4 within-sentence occurrences of B
whose **nearest** preceding A lies 2–24 words back, and window-lift ≥ 4 — lift measured against the chance of a
B landing anywhere in a 23-word window, `1 − (1 − count(B)/n)^23`, not against a per-word probability. On
`wt103_test[:400 KB]` (66,274 words, 21 documents) that yields the associations a reader would name:
`miles → km`, `steiner → hitler`, `superfamily → family`, `naktong → division`, `winds → km/h`,
`kershaw → season`. The same rule on `stdlib_test` yields `src → dst`, `handle → request`, `alert → description`.

Three properties of the instrument bound what can be measured. Two were found by reading the code; the third
by replaying the slot rule over the stream without a model:

1. **The machine's memory is sentence-scoped.** For every `SEL_ARM`, `_byte_end` executes
   `if b == 46: self.slots = [0] * S` — a `.` wipes the entire slot LRU, so a cue separated from its outcome by
   a sentence end cannot be a candidate and contributes exactly 0 by construction. That is **2,220 of 11,751**
   extracted events. (One carve-out is real and flagged: the wipe happens *before* the LRU insert of the word
   that just completed, so the last word of a sentence survives into slot 0.)
2. **The memory is saturated at the adopted table size.** A model-free replay of the instrument's own slot rule
   and key expression over train+measure (`keycount.py`) gives **301,246,416 cell reads** over an estimated
   **104,860,139 distinct `(word, 3-byte context, phase, prefix)` keys** (a 2^32 bitset with the standard
   collision correction). Against `WVBITS2 = 22`'s 4,194,304 cells that implies a **100 % collision rate and
   ~25 distinct keys per cell**; the ladder is 99.8 % / 79.0 % / 32.3 % at 2^24 / 2^26 / 2^28.
3. **Only 4 of ~16.6 residents are ever served.** `SEL_TOPM = 4` truncates the candidate set and the gate score
   is `SEL_UGAIN·u − SEL_PBD·k`, so an untrained cue loses to recency by 0.05 per slot step.

### 87.2 The measurement

At every bit of an instrumented outcome word the job calls `predict()` once — pure; self-test S1 confirms the
per-byte cost stream is bit-identical to a plain `run()` — and forms three quantities from the served state:

| | what it is | what it measures |
|---|---|---|
| **served gain** | cost(cue removed from the served set T, weights renormalised) − cost as served | §86's cue-attributable metric. A verdict about the **readout**, not the memory (below). |
| **cue evidence** *e* | Σ over the byte's bits of (f − f_without_cue)·(+1 if y==1 else −1), in logits | what the cue's own cell pushed toward the bit that actually occurred, free of the readout weight |
| **cell cost** `LA` | −Σ log2 p1_cue over the outcome's identity byte | what B's identity byte would cost coded from the cue's own cell alone |

**Why the served gain is not a memory measure** — the finding that forced the instrument to be rebuilt before
it ran. Its sign is `sign(vsw) × (the cue's own direction)`, and for `selleanu` `vsw` is ONE shared RMSProp
scalar per `(previous byte, phase, prefix)` — the same key for 98.6 % of fact events — measured negative on
73.2 % of scored bits. The registered statistic is therefore **maximised by the absence of binding**: −0.0226
bits for cues whose cell really holds B, +0.0372 for cues whose cell is wrong about it. That is §85's F1 and
§86.2's metric error a third time, caught before the run rather than after it.

**Controls** (all on the identical counterfactual): **C-MATCH**, the *served* candidate whose stream frequency
is nearest the cue's, so both arms are served and the comparison is about which memory; **C-ELSEWHERE**, the
same cue at neighbouring word starts that are not B; **C-NULL**, a band-relative pair with the same outcome;
**CROSS**, the structural zeros of 87.1(1). Primary population: facts that qualify on the stream's first half,
scored on its second; primary stratum adds fan-out ≤ 2 and B absent from the preceding 25 words. Bootstrap:
two-level cluster over documents then fact types. Self-tests S1–S4 all pass (S3 after the fix in §87.5).

### 87.3 The result: what de-collision does to the memory, and what it does to compression

Six runs on the 400 KB stream (`selleanu` at `WVBITS2 ∈ {22, 24, 26, 28}`, `selleanfa`, `baseline`) plus a
replication on an 890 KB stream. **The load-bearing population is the 1,661 fact events that are served at
every rung and have a served C-MATCH partner** — a within-event, within-byte, service-matched contrast into
which no selection imbalance can enter (the first draft's population could not say that; see §87.5).

| table | collisions | `LA` fact cell | `LM` matched partner | **`LA − LM`** (95 % CI) | *e* fact | **bits/byte** |
|---|---|---|---|---|---|---|
| 2^22 (adopted) | 100.0 % | 7.176 | 6.886 | **+0.290 [+0.046, +0.535]** | −0.6444 | **2.278784** |
| 2^24 | 99.8 % | 6.678 | 6.557 | +0.122 [−0.199, +0.432] | −0.5969 | 2.281810 |
| 2^26 | 79.0 % | 6.079 | 5.986 | +0.092 [−0.236, +0.427] | −0.4431 | 2.284270 |
| 2^28 | 32.3 % | **5.189** | 5.310 | **−0.121 [−0.421, +0.130]** | **−0.2503** | **2.284920** |

**Three statements, each with its own interval.**

1. **At the shipped table size a word's memory of its own fact is significantly *worse* than a coincidence.**
   `LA − LM = +0.290 bits [+0.046, +0.535]`: the cue's own cell predicts the outcome's identity byte worse
   than the cell of a frequency-matched word that merely happens to be in the same vote set at the same
   instant. On cue evidence the same comparison is −0.6444 against −0.0108 — the fact's cue actively pushes
   away from its own outcome while the coincidental word is neutral.
2. **De-collision repairs that, and the repair is fact-specific.** Paired difference-in-differences over the
   same 1,661 events: **−0.4106 bits [−0.7284, −0.1570]** on `LA − LM`, and **+0.4514 logits
   [+0.2478, +0.6871]** on cue evidence against C-MATCH. Both exclude zero. So of the ~2 bits of cell
   sharpening de-collision buys (`LA` 7.176 → 5.189), about 0.41 bits is specific to the fact's own cell; the
   rest is generic — every cell gets sharper (`LM` 6.886 → 5.310).
3. **Every step of that repair makes compression monotonically worse.** bits/byte 2.278784 → 2.284920,
   Spearman(−collision rate, bits/byte) = **+1.000**, against Spearman(−collision rate, cue evidence) =
   **+1.000** on the served-conditional population (−0.6604 / −0.6269 / −0.4624 / −0.2566, n ≈ 1,900 per rung).

The 2^22 figure 2.278784 reproduces §86.8's independently measured wt103 2700 KB value of 2.2788 exactly, and
the `baseline` rail (2.286090) reproduces §85.1's 2.2861 to five decimals.

**This is what §87 was built to get.** §86.8 inferred the memory/compression divergence after the fact from two
disagreeing numbers. §87 registered it, with a branch for every outcome, and measured it at four doses inside
one run on the same bytes: **the machine's memory of specific facts and its compression score move in opposite
directions, monotonically.**

**The obvious confound, run and refuted in the opposite direction** (`shrink.py`, confirmed independently by
the red-team's `rt87b/shrinklens/`). A bigger table spreads the same 301 M updates over more cells (counts per
cell fall 109 → 66), so every logit-scale quantity could shrink toward zero mechanically. It does not: the
cells **sharpen** (mean |p1 − 0.5| rises 0.2533 → 0.4271), the vote feature's own scale more than doubles
(mean |f| 1.78 → 3.94), and `LA` falls toward the rail where count-starvation would drive it *up* toward 8
bits. A pure-rescaling model predicts the fact's evidence should become *more* negative at 2^28 (−0.139 to
−0.171); it is measured at −0.0545.

### 87.4 What the machine still cannot do, and why

**It does not recall facts on real text.** Even at 32 % collisions the cue's served evidence is −0.25 logits:
consulted, the memory still pushes away from the byte that occurs. Two limits survive de-collision:

* **Selection.** The cue is in the slot LRU on **100.0 %** of events and in the served vote set on **21.3 %**,
  and beyond nine words on **0.15 %** (5 of 3,337 events at 2^28) — `SEL_TOPM = 4` against 16.6 residents,
  with a recency prior no untrained cue can outrank. Forcing the cue in costs bits
  (−0.0087 [−0.0137, −0.0041]), and the cost is the **displaced member, not the cue**: forcing a matched
  non-fact cue into the same positions costs the same (−0.0098 vs −0.0099, paired difference
  −0.0001 [−0.0032, +0.0026]). Whether *enlarging* the served set rather than swapping into it pays is
  therefore untested, and is §89A's question.
* **Readout.** The served gain — §86's metric, the readout's own verdict — is −0.0005 bits over all events and
  +0.0006 on the clean stratum at 2^28. A 2-bit improvement in cell content converts to nothing, because one
  shared `vsw` per (previous byte, phase, prefix) prices every word's vote.

**`selleanfa` changes none of it, and the reason is sharper than the headline.** On all 1,391 events both arms
serve, the cue cell's code length for the outcome byte is **identical to the last bit** (paired difference
exactly 0.0000): the two arms share the vote cells and differ only in how fast trust moves. Paired cue-evidence
difference −0.0034 [−0.0167, +0.0133]; and on this stream fast trust is *worse* on bits/byte (2.280100 vs
2.278784). Trust speed is not the constraint.

**Replication on an independent, larger stream — and it is stronger there.** The registered power remedy was
invoked (the primary stratum held 38 events at 400 KB) and the whole ladder re-run on `wt103_test[:890 KB]` —
147,026 words, 40 documents, 4,498 fact types, 26,204 fact attributions, and **4,254 events in the paired
service-matched population** against 1,661 on the 400 KB stream:

| table | `LA` fact | `LM` matched partner | **`LA − LM`** (95 % CI) | *e* fact | *e* match | bits/byte |
|---|---|---|---|---|---|---|
| 2^22 | 7.309 | 6.925 | **+0.384 [+0.233, +0.540]** | −0.7779 | −0.0436 | **2.244060** |
| 2^24 | 6.832 | 6.526 | +0.306 [+0.111, +0.502] | −0.7778 | −0.0944 | 2.247285 |
| 2^26 | 6.078 | 6.023 | +0.055 [−0.137, +0.254] | −0.5747 | −0.1638 | 2.250053 |
| 2^28 | 5.177 | 5.414 | **−0.237 [−0.396, −0.073]** | −0.3449 | −0.1600 | 2.250980 |

On this stream **both endpoints exclude zero**, so the sign change is significant at both ends rather than at
one: at the shipped size the fact's own cell is worse than the coincidence's by +0.384 bits, and at 2^28 it is
better by 0.237. Paired DiD **−0.6211 bits [−0.8095, −0.4305]**; on cue evidence against C-MATCH
**+0.5494 logits [+0.3860, +0.7646]**. The bits/byte column is monotone worse across all four rungs here too
(ρ = +1.000), against a rail of 2.253037 — note the adopted 2^22 selector beats that rail by 0.0090 while the
2^28 one beats it by only 0.0021, i.e. **de-collision gives back three-quarters of the channel's entire
compression value** while repairing the memory.

### 87.5 The registered verdicts, and the three sentences the red-team withdrew

* **R4 — PASS.** Registered bar ρ ≥ +0.8 for cue evidence against falling collision rate, with bits/byte
  worsening. Both hold. **Stated with its weakness:** on the *registered* stratum (clean and unmasked) the
  population is 17 events over 6 fact types of which only three ever score (`either→or`, `more→than`, `3→4`),
  the ladder is one adjacent inversion short of monotone, and ρ = +0.800 has an exhaustive-permutation
  p = 0.167. The result is carried by the served-conditional all-facts population (ρ = +1.000, n ≈ 1,900 per
  rung) and by the paired DiD of §87.3, not by the registered 17 rows.
* **R1 — FAIL, and its registered statistic is uninformative in both directions.** The estimand was
  fact − C-MATCH, but C-MATCH is drawn from the served set and is therefore served on 100 % of rows while the
  fact is served on 19.6 %, with *e* identically 0 when unserved: the raw statistic reads +0.345
  [+0.208, +0.457] at 2^22, a "pass" produced by 0-vs-negative. Restricted to rows where both are served, the
  fact is worse (−0.644 vs −0.011) — so R1 fails on the comparable population. Its threshold was also
  mis-scaled by me: a *per-event* 95th percentile compared against a *mean*.
* **R2 — INDETERMINATE, and the first draft's reading of it was wrong.** The draft reported "a matched
  non-fact pair scores better than the fact (−0.101 vs −0.130)" and read it as the machine's memory being
  worse than useless. **Withdrawn.** `mean e = P(served) × E[e | served]` exactly, and the entire gap is the
  first term: fact cues are served on 0.196 of events against 0.129 for C-NULL and 0.142 for C-ELSEWHERE,
  and every conditional mean is negative, so the more-served arm is penalised by construction. Conditional on
  service the fact is the least negative of the three at every rung. The controlled version of the claim is
  §87.3's `LA − LM`, which survives.
* **Two of §87's own controls are defective and must not be used for specificity.** **C-NULL is 70 % facts:**
  674 of 961 partners (73.0 % of scored rows) are themselves members of the frozen fact set, because the
  partner had to clear the same ≥ 4-in-sentence-events gate and the "bottom quartile" is not a quartile when
  the median pool size is 2. **C-ELSEWHERE's pooled sign is an artifact of its `j+1` half:** j−1 gives −0.1505
  (worse than the fact) and j+1 gives −0.0621 (better), and the adjacency filter drops 1,503 j−1 rows and
  exactly 0 j+1 rows. Both are reported, neither is used.
* **R3 — 0.056 / 0.111 / 0.278 / 0.500** across the ladder on the 400 KB primary population, 0.000 → 0.235 on
  the 890 KB replication (descriptive; no threshold registered, as declared).
* **R5 — the rail costs 4.12 bits** for a fact's identity byte over all filtered events, 3.70 over served ones.
* **A4 is weaker than registered.** Only the ≥ 4-events gate is computed on the first half; lift, the count
  gates and fan-out are still computed over the whole stream. Under strict first-half qualification the split
  population is 162 events and the direction holds (e(2^28) = +0.128 [−0.270, +0.566]) but significance does not.
* **S1–S4 pass**, S3 after an event-definition fix the test itself found: 62 of 2,220 "cross-sentence" events
  had the cue *also* adjacent to the outcome (slot 0, distance 1 — invisible to a window starting at 2), so
  the record named a distant copy while the attribution went to the adjacent one. All 1,736 such rows are
  dropped everywhere; afterwards the structural zero is exact (max |gain| 0.000e+00).
* **Checked and not supported:** that the readout uses the memory *inverted* (a real-text version of §86.2's
  novelty detector). The quadrant test (`negoracle.py`) does not behave as a sign model predicts, and the
  served channel is worth ≈ −0.003 bits at fact events either way. Recorded so it is not re-run.

### 87.6 Where this leaves the intelligence path

**In plain words.** Ask the machine what it remembers about a word at the moment its partner arrives —
*Steiner … Hitler*, *miles … km* — and at the size this project ships, what comes back about that specific
pair is *worse than what a word standing next to it by coincidence has to say*. Its memory cell for the pair
is shared with about twenty-five unrelated pairs. Give the pairs room and the machine's memory of its own
facts improves specifically, by about 0.4 bits more than the coincidence's does. **And every step of that
improvement makes the compression score worse.** That is the whole reason §87 exists: §86.8 suspected it,
§87 registered it and measured it at four doses.

**The three limits, now separated and ordered.** §85 named storage / trust / readout; §87 measures the
corresponding three on real text:

1. **Addressing** — the binding one, and the only one §87 moves. Worth ~2 bits of cell sharpening, of which
   ~0.4 bits is fact-specific.
2. **Selection** — the cue reaches the served set on 21 % of events and 0.15 % beyond nine words. Untested
   whether enlarging the set pays, because the only intervention measured (substitution) costs the same for a
   fact cue and a non-fact cue.
3. **Readout** — one shared weight prices every word's vote, converting 2 bits of content into +0.0006.
   §86.3 already showed the bucketed fix is the worst arm on real text, so this is a per-word or per-cell
   problem, not a re-bucketing one.

**Next (pre-registered blind in `prereg_89.md`, before any §87 number was read).** §89B: port the engine's
`BLSELTAG` (8-bit tag, evict on mismatch — §86.6, +0.000374…+0.000935 held out) into the instrument and ask
whether tagging at 2^22 reaches the 2^28 rung's `LA − LM` **at 4 MB instead of 4.29 GB**. Tagging and
enlarging are different interventions — enlarging dilutes collisions, tagging evicts them, and §86.5 measured
eviction beating merging in the engine — so §87's ladder bounds what addressing is worth without saying how
cheaply it can be bought. §89A (the `WSELTOPM` / `WSELPBD` ladder, knobs added in v13 and verified
bit-identical) follows it, because selection only binds once addressing is fixed.

**What §87 does not claim.** That the machine recalls facts on real text — it does not, at any rung. That the
addressing fix is the cheapest one — that is §89B's question. Anything about the engine, whose tag result is a
separate held-out measurement (§86.6). And it answers §86.7's item (3), the reason §87 was run: corpus
bits/byte is not merely a poor instrument for binding, it moves in the **opposite direction** to fact-level
memory quality across four measured doses.

**Method lesson, the fifth of its kind.** §85's F1, §86.2's registered metric, §87's served gain, §87's own
threshold — and now §87's first draft, which read `P(served) × E[e | served]` as a statement about memory
content when its sign was set by the service rate. Every one is the same error: **a statistic aggregated over
a channel does not measure the part of the channel under test.** The estimand that survived is the one that
holds everything but the thing under test fixed — same event, same byte, same bits, both cells served.

---

## Appendix — prior-art map (search terms, all bit/discrete, not LLM-specific)

- **Semantic hashing** — learn compact binary codes preserving similarity (the learned "hash").
- **Locality-Sensitive Hashing (LSH)** — binary addresses for sub-linear similarity search.
- **Self-Organizing Maps (Kohonen)** — gradient-free competitive placement by similarity.
- **VQ-VAE** — discrete codebook learning (straight-through + commitment loss).
- **Hopfield / modern Hopfield networks** — binary content-addressable memory; retrieval = relaxing into a stored attractor.
- **Neural Turing Machine / DNC** — content-addressable memory with learned keys.
- **Neural Cellular Automata, Rule 110, Conway's Life** — multi-directional, parallel, Turing-complete bit machines.
- **Analog Bits, MaskGIT, discrete diffusion** — any-order / masked bit-field generation.
