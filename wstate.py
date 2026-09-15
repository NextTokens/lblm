#!/usr/bin/env python3
"""
wstate.py -- PHASE 1, the decisive experiment: does a SEMANTIC recurrent state cross BELOW
the order-n baseline where SS70's byte-bit state (lstate.py) only reached parity?

Diagnosis being tested (SS70/SS71): a compact state fed the 8 RAW BITS of each byte can only
track byte-class running statistics -- structure the order models already have. The named fix
("richer input projection -- not a new idea"): feed the state the CURRENT WORD'S EMBEDDING
alongside the bits, with the embedding LEARNED ONLINE BY THE COMPRESSION LOSS ITSELF (one-step-
truncated RTRL through the state, mixer weights as feedback). m=32 fixed-long-timescale diagonal
EMA units (timescales ~7 .. ~2000 bytes; SS70: forced-long decays tie learned ones).

    h_j(t+1) = a_j h_j(t) + (1-a_j) * W_j . x(t),   x = [8 signed bits of byte, E[w_t] (D=16)]
    W learned (exact diagonal RTRL):  W_jk  -= lr_rec  * G_j(t) * e_jk,
                                      e_jk  <- (1-a_j) x_k + a_j e_jk
    E learned (word-trace credit): each recent word w carries an m-vector eligibility
           A[w,j] <- a_j A[w,j] + (1-a_j) G_j  [only while w was the last active word]
           flushed on eviction/end-of-run as  E[w,k] -= lr_emb * sum_j A[w,j] W_j[8+k]
           (exact long-range credit for frozen W -- the 1-step truncation taught embeddings
            only to predict their OWN next letter, which random vectors already do; v2 fix)
    G_j(t) = sum over the byte's 8 bits of (p - y) * w_mixer[state_j]   (RTRL learning signal)

v3: +BUCKET EXPERT -- the probe (_assoc_check) showed the LINEAR mixer cannot read word
identity out of the EMA superposition beyond ~1 word back (bitsonly==learned==chance at a
2-word cue; the v1/v2 embedding credit is not the bottleneck, the readout is). So every
state arm additionally feeds a NONLINEAR count-table expert keyed
    (16 sign bits of mid-timescale units 12..27, bit phase, partial byte)
(the prims.py mechanism, on the learned state).

v4: +WORD-SLOT BINDING -- buckets on the EMA still fail the 2-word-cue probe: the EMA
SUPERPOSES ~5 words, so no function of it isolates one word's identity. The fix is the
project's own cycle-5 idea reborn: an LRU of the last S=6 DISTINCT words, each slot holding
that word's OWN small vector (SD=8), fed DIRECTLY as mixer features (no superposition, no
RTRL needed: a slot persists for ~6 words, so every bit while resident gives EXACT credit
to the embedding). Arms:
    slotsr : EMA(bits) + bucket + slots with FROZEN random vectors  (identity control)
    slots  : EMA(bits) + bucket + slots with vectors LEARNED by the loss (THE HEADLINE)
  slots vs bitsonly = the slot channel's effect; slots vs slotsr = MEANING vs identity.
  RESULT (§72): slots CROSS below the orders baseline at >=1.2MB, margin growing with data,
  replicated across seeds; slotsr never crosses -> the win is the learned vectors.
  CORRECTED (SS77): in that run the LRU held word PREFIXES (see WSLOTMODE below), so slot 0 was the
  current prefix and the crossing is a learned current-word model: one prefix slot (WSLOTS=1) beats
  S=6, and real whole-word slots (WSLOTMODE=word) lose to the orders baseline at every size measured
  (seed 0; ledger SS77.2).

v5 (Phase 2, similarity): the strong.rs port showed identity slots add NOTHING to an
lpaq-class engine (hashed order-8..32 already span 1-5 words) -- the open lever is
SIMILARITY, generalisation across related words. Two new arms, both = slots + ONE extra
readout of the CURRENT word (prefix id, like the word model):
    slotsw : + an IDENTITY-keyed count expert  (word-hash bucket, phase, partial) [control]
    semsim : + an EMBEDDING-BUCKET count expert (8 sign bits of the current word's vector ->
             256 buckets; words the loss has ALIGNED share counts -> evidence transfers
             across the word tail)  + S coherence features dot(E[cur], E[slot_i])
  semsim < slotsw (same capacity, same placement) => SIMILARITY beats IDENTITY keying --
  the Phase-2 thesis. semsim < slots => the similarity layer adds margin on its own.
  CORRECTED (SS77): under WSLOTMODE=prefix slots[0] is the current prefix at word-interior bytes, so slotsw
  is a word model and semsim buckets per-prefix predictor vectors; this could not measure similarity.

v7 (SS77, ATTENTIONAL READ): SS75/76 read the deep slots as HOLDING information a fixed recency readout
cannot use (SS77: that was a noise-dimension artifact). The pre-registered test reads the LRU with CONTENT-based
attention, served once per byte (forward at byte end, after the LRU update):
    c = E[slot0] (last completed word under WSLOTMODE=word);  q = Wq c + bq;  e_k = q.E_k + beta_k;  a = softmax(e)
    pooled r = sum_k a_k E_k  (SD features);  T = top-m slots by a (WTOPM=4)
    vote v = sum_{k in T} (a_k/A_T) stretch(P_vote[hash(word_k, order-WVORD ctx, phase, partial)])
  credit is EXACT for parameters frozen within the byte (per-bit dL/dr and dL/da accumulated with
  the predict-time mixer weights, backprop through the softmax applied at byte end). Arms:
    attn    : learned vectors + learned query + learned position bias       (THE HEADLINE)
    attnr   : vectors FROZEN, query/bias learn                              -> MEANING control
    attnrec : learned vectors, a_k = recency weights GP[k] (SS76), T = m most recent
                                                                            -> SELECTION control
    attnscr : the SD+1 attention features replaced by noise                 -> floor
    attncos : = attn, but COSINE content score at a fixed scale (SS77c; WKAPPA, default 8.0):
              e_k = KAPPA (q.E_k) / (|q||E_k| + 1e-8) + beta_k  -- content logits O(KAPPA) whatever the
              vector norms (attn's q.E_k spread is ~0.002 at init norms ~0.08, so attn ran near-uniform and
              position-driven). Exact backward through the normalisation for q and every E_k.
  SS77 finding (instrument bug): the slot LRU was updated at EVERY letter byte with the growing
  PREFIX id, so S=6 held ~2 words ('cat','ca','c','quick','quic','qui') and S=32 ~6-8 words;
  strong.rs BLMSLOTS inserts only COMPLETED words. WSLOTMODE=prefix (default, SS72-76 bit-identical)
  | word (completed words only -- the intended mode for the attn arms, with WSLOTS=32).
  SS77 RESULT: the long-gap probe (_bind_probe.py) killed the attention arms -- no content arm binds a
  cue at any gap; the vote's mixer weight settles negative, so correct votes hurt (ledger SS77.4).

v8 (§78A, IS THE §72 CHANNEL MORE THAN A COUNT WORD MODEL?): §77 showed the §72 "win" is a learned SD=8 vector
for slots[0], which under WSLOTMODE=prefix is the growing current prefix at letter bytes and the last completed
word after a non-letter byte (until the next letter). It was only ever compared with an orders-only baseline.
New arms, ALL with use_state=False (no M=32 EMA, no bucket expert) so they isolate the channel; each = orders
0..6 plus (inputs at an explicit per-model base, NOT the legacy NM+M+1: self.fbase = NM, the vector dims at
self.pbase, the count expert at self.cbase):
    wcnt      : ONE exact count expert keyed (slots[0] id, phase, partial byte) in a dict (no hashing, no
                collisions), stretch((n1+0.2)/(n0+n1+0.4)), updated only when learn     -> a COUNT word model
    pvec      : the learned SD=8 vector of slots[0] fed as 8 mixer features, trained by the same exact per-bit
                credit as the legacy 'slots' arm (predict-time mixer weight, applied at byte end, lr_s, clip +-1;
                applied only when learn)                                               -> the §72 channel, no EMA
    pvecr     : pvec with vectors FROZEN                                               -> extra-dims control
    wcntpvec  : wcnt + pvec
    wcntpvecr : wcnt + pvecr
  slots[0] follows the existing slot LRU with the arm's S (WSLOTMODE=prefix WSLOTS=1 for §78A). Empty slot ->
  zero vector features; the count expert keys id 0 as a normal key.
  report() prints wcnt-vs-baseline, pvec-vs-baseline, A1 = wcnt - wcntpvec, A2 = wcntpvecr - wcntpvec.
  PRE-REGISTERED (written here before any corpus run):
  Gate as §77.2 (copy OFF, 13-byte decontamination, deterministic mask), WSLOTMODE=prefix WSLOTS=1, seed 0.
  Corpora/sizes: code 100,200,400,585 KB; wt103 150,450,1200,2700 KB.
  A1: wcnt - wcntpvec >= +0.002 bpb at every size on both corpora (the learned vector adds beyond a count word model).
  A2: wcntpvecr - wcntpvec >= +0.002 bpb at every size on both corpora (the addition is learning, not extra dims).
  If A1 and A2 both pass: confirm on stdlib 400,1200,2400 KB with the same two criteria. (Seeds do not vary these arms: no seeded init.)
  Verdict: A1 and A2 (and stdlib) -> the learned prefix vector carries something a count word model does not; else it is a word model and
  the §72 channel is fully explained.
  SS78 RESULT: A1/A2 passed on code, wt103 and stdlib, but a red-team showed the gain is mostly recency adaptation that the
  cumulative (never-forgetting) counts lack; with nonstationary counts A1 fails on wt103 at 150-1200 KB (not re-run on stdlib; ledger SS78.2).

v8b (§78B, FORCED-SELECTION ORACLE for the §77.4 vote-readout trap -- DIAGNOSTIC arms with an EXTERNAL HINT,
used ONLY by _bind_probe.py): identical to attn in every respect except the vote set T. They read
model.oracle_wid, an attribute the probe sets (default 0 = no oracle) to the id of the word it wants selected
(the sentence's cue). The hint comes from outside the byte stream, so these arms are not stand-alone predictors.
    attnorc1 : T = {the slot holding oracle_wid} with renormalised weight 1 (vote = that slot's stretch); if
               oracle_wid is 0 or not in the slots, T is empty and the vote feature is 0 for that byte
               (counted: model.orc_absent out of model.orc_served attention-served bytes, both oracle arms)
    attnorc4 : T = {oracle slot} plus the top-(WTOPM-1) other slots by attention weight a (top-3 at the default
               WTOPM=4), renormalised over T as in attn; if absent, T = top-WTOPM by a exactly as attn
  Credit is attn's exact credit with the forced T held fixed (for attnorc1 the vote term gives no attention
  gradient since s_k - v = 0; its pooled-feature credit is attn's).

v9 (§79, RESPONSIBILITY-GATED MIXTURE OF VOTE EXPERTS -- the §78B follow-up): §78B showed that forcing the vote set to the
cue slot binds (attnorc1 +2.46..+2.52 bits at G=12/24) while cue + 3 attended slots (attnorc4) only reaches +0.40..+0.57:
a renormalised stretch average dilutes the cue's vote behind one scalar mixer weight, and attn's own T rarely holds the
cue, so the cue's vote cells never train (chicken-and-egg). These arms replace attn's top-m stretch vote with an exact
probability-space mixture whose gate is trained by the mixture's own responsibilities. Word mode, WSLOTS=32; use_state
and every non-vote part (EMA state, bucket expert, SD pooled features r) as attn. Arms:
    attnmoe     : (1) candidates = ALL non-empty slots (no top-m). Slot word k owns a vote expert
                      p_k(bit=1) = (n1+0.2)/(n0+n1+0.4), cell = attn's vote key/hash (word_k, order-WVORD ctx, phase,
                      partial) in a FLOAT count table (2 * 2^WVBITS doubles; attn's uint16 table is not allocated).
                  (2) gate prior g = softmax(e), e_k = q.E_k + beta_k, q = Wq E_0 + bq, exactly attn's dot-product score.
                  (3) probability-space combination with within-byte posterior weights: at the byte's first bit
                      w_k = g_k; p_moe(bit j = 1) = sum_k w_k p_k(1); after the bit's value y is seen
                      w_k <- w_k p_k(y) / sum_i w_i p_i(y)   (p_k(0) = 1 - p_k(1)). Hence prod_j p_moe(b_j) =
                      sum_k g_k P_k(byte) exactly, and the posterior after bit 8 is r_k = g_k P_k / sum_i g_i P_i. The mixer
                      gets ONE feature stretch(p_moe) (genmem.stretch, clip 1e-6) at attn's vote position NM+M+1+SD;
                      the SD pooled features are r = sum_k g_k E_k (attn's, with a = g).
                  (4) gate credit at byte end (serving slots, before the LRU update), when learn: the mixture's own byte
                      loss -ln sum_k g_k P_k gives dL/de_k = g_k - r_k; the pooled features add attn's exact mixer credit
                      (dL/dr per bit at predict-time mixer weights) through the SAME softmax, so
                          de_k = (g_k - r_k) + g_k (Gr.E_k - sum_i g_i Gr.E_i),
                      backpropagated exactly as attn's _attn_grads: dq = sum_k de_k E_k, dbeta_k = de_k, dWq = dq c^T,
                      dbq = dq, dE_k = g_k Gr + de_k q, dE_0 += Wq^T dq. The p_moe feature's own mixer gradient is NOT
                      propagated to the gate (the responsibility term replaces it). lr_head for Wq/bq/beta (clip +-8),
                      lr_s for vectors (clip +-1), as attn.
                  (5) soft expert counts, when learn, at every bit: after the posterior update of (3) (which includes the
                      bit just seen), each candidate's cell gets n_{k,y} += rho_k with rho_k = that running posterior
                      w_k p_k(y) / sum_i w_i p_i(y) (so bit 8 uses exactly the byte responsibility r_k; earlier bits use the
                      posterior given the byte's bits so far, which is what is known when the cell is written). Halving as
                      attn: if n_{k,y} + rho_k >= 255 then n_{k,y} = (n_{k,y} + rho_k)/2 and n_{k,1-y} = n_{k,1-y}/2 (exact float
                      halving; attn's integer rule rounds up). Candidates with rho_k == 0.0 are skipped (an exact no-op).
    attnmoe_uni : identical, but g frozen at 1/n over the n non-empty slots (no gate parameters are used or learned;
                  vectors learn through r = mean E with dE_k = Gr/n; (3) and (5) unchanged)      -> GATE-LEARNING control
    attnmoe_orc : identical, but g = one-hot on the slot holding model.oracle_wid (uniform 1/n if 0 or absent; counted in
                  orc_served/orc_absent like v8b); gate frozen; dE_k = g_k Gr. DIAGNOSTIC (external hint), used only for the
                  §79 pre-build check that a one-hot gate reproduces attnorc1 (whose vote cells get +1 per bit, like rho=1 here).
  Diagnostics kept for the probe: at_a = g, at_T = all candidate positions, mo_rlast = the served byte's final posterior r.

v9b (§79b AMENDMENT, made after the pre-grid collapse diagnosis of the registered attnmoe, before any §79 test-set result was
seen): the registered attnmoe gate collapsed within ~25 training sentences onto one fixed distractor word (g ~ 0.99 whatever the
query). Reviewer diagnosis: q ~ bq selects a fixed word (bq.E_word ~ +9..+12 vs the query term ~ -0.3), and responsibility-
weighted soft counts starve the cue expert (its first-outcome-byte cells cost 5.3-7.6 bits vs 2.8-4.2 for the collapsed word), so
r_cue < g_cue pushes the cue further down. Three amended arms (word mode, WSLOTS=32), identical to attnmoe unless stated:
    attnmoe_fc     : expert counts are FULL-WEIGHT and gate-independent: at every bit, when learn, EVERY candidate k's cell gets
                     n_{k,y} += 1.0 (same exact float halving: if n_{k,y} + 1.0 >= 255 then n_{k,y} = (n_{k,y} + 1.0)/2 and
                     n_{k,1-y} = n_{k,1-y}/2). The within-byte posterior, the p_moe feature, the byte-end responsibilities r and
                     the gate credit de_k = (g_k - r_k) + pooled credit are unchanged. Gate q = Wq E_0 + bq as attnmoe.
    attnmoe_fcnb   : attnmoe_fc with bq fixed at 0 and never learned (q = Wq E_0; _attn_grads returns dbq = None, no bq update).
    attnmoe_uni_fc : g frozen at uniform 1/n (as attnmoe_uni), full-weight counts as attnmoe_fc  -> control for both amended arms.
  Every pre-existing arm (attnmoe, attnmoe_uni, attnmoe_orc included) is bit-identical to v9.

Words: maximal [A-Za-z0-9] runs, lowercased, FNV-1a rolling hash -> id (prefix-visible, like
the core's word model). id 0 (no active word) -> zero embedding. The EMA of word embeddings is
a running TOPIC vector -- long-range structure no order-n byte table represents.

ARMS (genmem.py protocol: match/copy OFF everywhere, 13-byte-DECONTAMINATED held-out wt103,
orders 0..6 + state features in one online logistic mixer; deterministic mask, see below):
  baseline : orders only                            -- the rail (== SS70 baseline)
  bitsonly : m=32 state, x = bits only              -- attribution: is any win just m=32?
  randemb  : m=32 state, x = bits + FROZEN random per-word vectors (never trained)
             -- identity control: consistent word identity WITHOUT semantic generalization
  learned  : m=32 state, x = bits + embeddings TRAINED by the loss   -- THE HEADLINE
  scrambled: state features replaced by noise (input too)            -- the noise floor

Attribution ladder (all on the same decontaminated scored bytes):
  learned < baseline   -> the state helps at all
  learned < bitsonly   -> the EMBEDDING did it, not capacity           [the named fix works]
  learned < randemb    -> the win is MEANING, not lexical identity     [the words wall]
  learned < scrambled  -> real learned signal, not noise               [SS70's bar]

NOTE: genmem.clean_mask uses Python's process-salted hash(); this script runs arms as parallel
PROCESSES, so it re-implements the identical 13-gram excision with a deterministic FNV-1a hash
-> every arm scores byte-identical test positions.

Run:  python wstate.py                (5 arms x sizes [150,450,1200,2700] KB, arms in parallel)
      python wstate.py --selftest     (tiny sequential sanity run)
"""
import sys, math, random, time, os
from array import array
from concurrent.futures import ProcessPoolExecutor

from genmem import stretch, squash

ORDERS = [0, 1, 2, 3, 4, 5, 6]
M = 32            # state units (SS70 used 12)
XDIM_BITS = 8     # raw signed bits of the just-finished byte
D = 16            # word-embedding dimension
XDIM = XDIM_BITS + D
AMIN, AMAX = 0.85, 0.9995      # fixed decay spread -> timescales ~7 .. ~2000 bytes
FNV0 = 0x811C9DC5
MAXREC = 48                    # recent-word trace bank size (eligibility, evict-oldest)
S = int(os.environ.get("WSLOTS", "6"))   # word slots (LRU depth; 6 = SS72; deep = beyond order-32 reach)
SD = 8                         # slot vector dims
H = int(os.environ.get("WHEADS", "4"))  # SS76: shared recency-discounted projection heads
GAMMA = float(os.environ.get("WGAMMA", "0.85"))
# SS77: WHAT the slot LRU binds. "prefix" (default, the SS72-76 behaviour, kept bit-identical) inserts
# the growing PREFIX id at EVERY letter byte -- so S=6 holds ~2 words ('cat','ca','c','quick','quic',
# 'qui') and S=32 ~6-8 words. "word" inserts only COMPLETED words (on the byte that ends a word),
# matching strong.rs BLMSLOTS -- so S slots really are the last S distinct words.
SLOTMODE = os.environ.get("WSLOTMODE", "prefix")
assert SLOTMODE in ("prefix", "word"), SLOTMODE
GP = [GAMMA ** k for k in range(256)]   # precomputed recency discounts
PROJ_ARMS = ("proj", "projr", "projscr")
ATTN_ARMS = ("attn", "attnr", "attnrec", "attnscr", "attncos",
             "attnorc1", "attnorc4",
             "attnmoe", "attnmoe_uni", "attnmoe_orc",
             "attnmoe_fc", "attnmoe_fcnb", "attnmoe_uni_fc")     # SS77: attentional read over the slot LRU
ATTN_READ_ARMS = ("attn", "attnr", "attnrec", "attncos",
                  "attnorc1", "attnorc4",
                  "attnmoe", "attnmoe_uni", "attnmoe_orc",
                  "attnmoe_fc", "attnmoe_fcnb", "attnmoe_uni_fc")  # the non-floor attention arms
ORACLE_ARMS = ("attnorc1", "attnorc4",     # §78B: forced vote set from model.oracle_wid (diagnostic, external hint)
               "attnmoe_orc")              # §79: one-hot gate from model.oracle_wid (diagnostic, external hint)
MOE_ARMS = ("attnmoe", "attnmoe_uni", "attnmoe_orc",   # §79: responsibility-gated mixture of per-slot vote experts
            "attnmoe_fc", "attnmoe_fcnb", "attnmoe_uni_fc")   # §79b amendment arms
MOE_FIXED_GATE = ("attnmoe_uni", "attnmoe_orc",        # §79: gate not a function of learned parameters
                  "attnmoe_uni_fc")
MOE_RESP_ARMS = ("attnmoe", "attnmoe_fc", "attnmoe_fcnb")   # §79/§79b: learned gate trained by g_k - r_k
MOE_FC_ARMS = ("attnmoe_fc", "attnmoe_fcnb", "attnmoe_uni_fc")   # §79b: full-weight, gate-independent expert counts
MOE_NOBQ_ARMS = ("attnmoe_fcnb",)                      # §79b: bq fixed at 0, never learned
# §81 (v10) THE SIMPLEST SELECTOR -- the two §77.4/§78B/§79B diagnosed causes fixed directly:
#   (a) the vote's readout was ONE scalar mixer weight shared by ~99% of irrelevant bits (dilution/reversal):
#       here the vote stretch(p_sel) is read through a CONTEXT-SELECTED weight of its own
#       (per (prev_byte, phase, partial) cell, RMSProp) added to the mixer dot, so the deciding
#       context can adopt the vote without being pinned by the rest of the stream;
#   (b) every learned gate queried from slot 0, which at the deciding byte always holds the same
#       word -- here the gate is QUERY-FREE: e_k = u[w_k] - SEL_PBD*k, a directly trained per-word
#       usefulness score (EWMA of the word's own vote agreement, clamped) plus a recency position
#       bias, softmaxed at a scale that stays neither uniform nor saturated. Vote = probability-space
#       mixture over the top-SEL_TOPM candidates; vote counts are FULL-WEIGHT per candidate
#       (the §79b lesson: responsibility-weighted counts starve cells). Word mode, WSLOTS=32.
# §81 CORPUS PHASE PRE-REGISTERED (written 2026-09-14 before any §81 corpus run; the probe grid above
# had already been run and committed at 30ca0d8):
# Protocol: the standard gate (copy/match OFF, 13-byte decontamination, deterministic mask), seed 0,
# WNS=1 everywhere (PAQ nonstationary rule on the order tables -- the §78.2 mandate), WSLOTMODE=word.
# Corpora/sizes: code 100/200/400/585 KB; wt103 150/450/1200/2700 KB; stdlib 400/1200/2400 KB.
# Arms: baseline (orders-NS); sel (WSLOTS=32, full selector); selpos (WSLOTS=32, position gate);
# sel@1 (WSLOTS=1: the SAME machinery -- flat vote cells, contextual usefulness, context-selected
# readout -- with candidates = the last completed word only: the no-binding control).
# B1 (binding, load-bearing): sel beats sel@1 by >= 0.001 bpb at >= half the sizes of a corpus.
# B2 (rail): sel beats baseline-NS by >= 0.001 bpb at >= half the sizes of a corpus.
# B3 (gate): sel beats selpos at every size where sel beats sel@1.
# Reading (pre-committed): B1+B2 on any corpus -> the selector carries real-data binding value beyond
# near-word memory and beyond the orders rail; port to strong.rs under the beats-strong rule. B2 without
# B1 -> word-model-like value only (the §78 lesson repeats; no port). Neither -> the probe binding does
# not transfer to real corpora at this scale; honest negative, no port, re-assess.
#
SEL_ARMS = ("sel", "selpos", "selorc4", "sellean")
#   sel     : usefulness + position bias + context-selected readout   (THE HEADLINE)
#   selpos  : position bias only, same readout                        -> isolates the readout fix
#   selorc4 : vote set forced to {cue} + top-3 by gate (oracle_wid, probe-only diagnostic)
SEL_PBD = 0.05      # position bias per slot step (recency prior; small enough that learned usefulness outranks ~12 slots of recency)
SEL_TSC = 2.0       # softmax temperature (properly scaled scores; e is O(4) so a stays graded)
SEL_UDC = 0.995     # usefulness EWMA decay (timescale ~200 bytes)
SEL_UCLIP = 4.0     # usefulness clamp
SEL_TOPM = 4        # vote-set size (matches WTOPM=4 of the §77 arms)
SVBITS = int(os.environ.get("WVBITS2", "22"))      # §81 vote-table bits (flat, bounded; real corpora)
WNS = os.environ.get("WNS", "0") == "1"            # §81 corpus phase: PAQ nonstationary rule on the
                                                   # order tables (§78.2: cumulative counters are a weak rail)
SEL_UGAIN = 4.0     # gate gain on u (u is an EWMA of clipped log-advantage, |u| ~ 0.1 at
                    # probe scale; the gain puts a separated cue above ~12 slots of PBD)
# §78A: the slots[0] channel in isolation (use_state=False; orders 0..6 + these inputs only)
W78_ARMS = ("wcnt", "pvec", "pvecr", "wcntpvec", "wcntpvecr")
CNT_ARMS = ("wcnt", "wcntpvec", "wcntpvecr")         # exact count expert keyed (slots[0], phase, partial)
PVEC_ARMS = ("pvec", "pvecr", "wcntpvec", "wcntpvecr")   # slots[0]'s SD-dim vector as mixer inputs
PVEC_LEARN_ARMS = ("pvec", "wcntpvec")               # ... trained by the exact per-bit credit (others frozen)
KAPPA = float(os.environ.get("WKAPPA", "8.0"))           # SS77c: attncos fixed cosine scale
TOPM = int(os.environ.get("WTOPM", "4"))                 # SS77: vote over the top-m attended slots
VORD = int(os.environ.get("WVORD", "3"))                 # SS77: vote context order (bytes)
VBITS = int(os.environ.get("WVBITS", "22"))              # SS77: vote table = 2 * 2^VBITS counts
M64 = (1 << 64) - 1
SLOT_ARMS = ("slots", "slotsr", "slotsw", "semsim", "semfast", "matchslots",
             "proj", "projr", "projscr", "scrambled") + ATTN_ARMS + SEL_ARMS
LEARNED_SLOT_ARMS = ("slots", "slotsw", "semsim", "semfast", "matchslots")  # slot vectors trained by the loss
SIM_ARMS = ("slotsw", "semsim")            # v5: +word expert (identity vs embedding bucket)
XKEY_ARMS = ("slotsw", "semsim", "semfast")
SUBWORD_ARMS = ("semfast",)                # v6: char-3gram-composed vector init (fastText-style)
MATCH_ARMS = ("matchbase", "matchslots")   # §75: the reconciliation test -- copy ON in the
# instrument too. If slots' crossing collapses with the match channel present, the no-copy win
# lived on the signal the match model harvests in production (repeated word usage).


def is_wb(b):
    return (65 <= b <= 90) or (97 <= b <= 122) or (48 <= b <= 57)


def clean_mask_det(train, test, K=13):
    """genmem.clean_mask, but with a process-independent FNV-1a gram hash (deterministic)."""
    grams = set()
    for i in range(len(train) - K + 1):
        h = FNV0
        for j in range(K):
            h = ((h ^ train[i + j]) * 0x01000193) & 0xFFFFFFFF
        grams.add(h)
    contam = bytearray(len(test))
    for i in range(len(test) - K + 1):
        h = FNV0
        for j in range(K):
            h = ((h ^ test[i + j]) * 0x01000193) & 0xFFFFFFFF
        if h in grams:
            for j in range(i, i + K):
                contam[j] = 1
    return [c == 0 for c in contam]


class Model:
    def __init__(self, arm="baseline", lr=0.004, lr_rec=0.02, lr_emb=0.03, lr_s=0.02, lr_head=0.02, seed=0):
        self.arm = arm
        self.use_state = arm not in ("baseline", "matchbase", "sellean") and arm not in W78_ARMS
        self.use_match = arm in MATCH_ARMS     # §75 reconciliation arms: copy ON in the instrument
        self.lr, self.lr_rec, self.lr_emb, self.lr_s, self.lr_head = lr, lr_rec, lr_emb, lr_s, lr_head
        self.NM = len(ORDERS)
        self.NIN = self.NM + ((M + 1) if self.use_state else 0)     # +1 = bucket expert
        self.NIN = self.NIN + (1 if self.use_match else 0)          # match/copy vote
        if arm in SLOT_ARMS and arm not in PROJ_ARMS and arm not in ATTN_ARMS and arm not in SEL_ARMS:
            self.NIN += S * SD                                     # word-slot features
        if arm in PROJ_ARMS:
            self.NIN += H                                          # SS76 projection-head features
        if arm in ATTN_ARMS:
            self.NIN += SD + 1                                     # SS77 pooled features + vote
        if arm in SEL_ARMS:
            self.NIN += SD                                         # §81 pooled features (the vote is read through its OWN context-selected weight, not a mixer input)
            self.sel_rbase = self.NM + (M + 1 if self.use_state else 0)   # §81: r-feature base (no state block for sellean)
        if arm in XKEY_ARMS:
            self.NIN += 1                                          # word-expert feature
        if arm in ("semsim", "semfast"):
            self.NIN += S                                          # coherence dot features
        if arm in W78_ARMS:
            # §78A: explicit feature base (no state block, so NOT the legacy NM+M+1):
            #   [orders 0..6][slots[0] vector SD (PVEC_ARMS)][count expert 1 (CNT_ARMS)][bias]
            self.fbase = self.NIN
            self.pbase = self.fbase
            if arm in PVEC_ARMS:
                self.NIN += SD
            self.cbase = self.NIN
            if arm in CNT_ARMS:
                self.NIN += 1
            self.wctab = {}                    # (slots[0] id, phase, partial) -> [n0, n1], exact keys
            self.pv = None                     # slots[0]'s vector (the list in semb), refreshed at byte end
            self.wcount = {}                   # word id -> _svec calls (diagnostics only; no state block here)
        self.w = [0.0] * (self.NIN + 1)
        self.wg = [0.0] * (self.NIN + 1)
        self.tab = [dict() for _ in ORDERS]
        self.btab = {}                        # bucket expert counts (state arms)
        self.buck = 0
        self.xtab = {}                        # v5 word-expert counts (slotsw/semsim)
        self.xb = 0                           # its key bucket, refreshed per byte
        self.curv = None                      # current word's vector cache (semsim)
        self.curdots = [0.0] * S              # dot(E[cur], E[slot_i]) cache (semsim)
        self.slots = [0] * S                   # LRU word ids (0 = empty)
        self.semb = {}                         # word id -> SD floats (slot content)
        self.sgrad = {}                        # word id -> per-byte accumulated exact credit
        if self.use_match:                     # §75: genmem's byte match/copy model
            self.MINLEN, self.GATE = 16, 18
            self.mtab = {}
            self.hist = bytearray()
            self.mptr, self.mlen = -1, 0
        if arm in PROJ_ARMS:                   # §76: shared recency-discounted heads
            rng = random.Random(seed + 991)
            self.hw = [[(rng.random() - 0.5) * 0.2 for _ in range(SD)] for _ in range(H)]
            self.hgrad = [0.0] * H             # per-byte exact credit per head
            self.hfeat = [0.0] * H             # cached features (serve the next byte)
            self.hdw = [[0.0] * SD for _ in range(H)]   # d feat_h / d w_h  (cached)
            self.hck = [0.0] * max(S, 1)       # d feat_h / d E[slot_k] coefficient per slot
        if arm == "scrambled" or arm == "projscr" or arm == "attnscr":
            self._sr = random.Random(1234)
        if arm in ATTN_READ_ARMS:                   # SS77 attentional read
            if arm != "attnrec":
                rng = random.Random(seed + 1777)
                self.Wq = [[(1.0 if i == j else 0.0) + rng.uniform(-0.05, 0.05) for j in range(SD)]
                           for i in range(SD)]
                self.bq = [0.0] * SD
                self.beta = [0.0] * S
            if arm in MOE_ARMS:
                self.mvtab = array("d", [0.0]) * (2 << VBITS)   # §79: flat (n0, n1) FLOAT expert counts
                self.mo_w = []; self.mo_p1 = []; self.mo_rlast = None   # §79: running posterior, per-bit p_k(1), last r
            else:
                self.vtab = array("H", [0]) * (2 << VBITS)   # flat (n0, n1) vote counts
            self.at_n = 0                          # non-empty slots served (0 = features/vote off)
            self.at_ks = []; self.at_E = []; self.at_a = []; self.at_T = []; self.at_wT = []
            self.at_AT = 1.0; self.at_c = None; self.at_q = None
            self.at_cos = None                     # attncos backward cache (|q|, |E_k|, q.E_k, den_k)
            self.at_r = [0.0] * SD; self.at_hb = []
            self.aGr = [0.0] * SD                  # per-byte dL/dr
            self.aGa = []                          # per-byte dL/da_k (vote path), aligned with at_T
            self.vt_idx = []; self.vt_s = []; self.vt_v = 0.0   # per-bit vote cache (predict time)
            if arm in ORACLE_ARMS:
                self.oracle_wid = 0                # §78B: set by the probe (0 = no oracle)
                self.orc_served = 0                # attention-served bytes (slots non-empty)
                self.orc_absent = 0                # ... of which oracle_wid was 0 or not in the slots
        if arm in SEL_ARMS:                        # §81 the simplest selector
            self.wcount = {}                       # (also for lean arms: no state block initializes it)
            self.svt = array("d", [0.0]) * (2 << SVBITS)   # flat (n0, n1) vote counts, bounded
            self.svmask = (1 << SVBITS) - 1
            self.suw = {}                          # per-word usefulness score (arm sel; EWMA of own-vote agreement)
            self.vsw = {}                          # CONTEXT-SELECTED vote weights: cx -> scalar
            self.vswg = {}                         # RMSProp state per cx
            self.lr_vsw = 0.02                     # the context-selected weight's learning rate
            self.oracle_wid = 0                    # selorc4: set by the probe
            self.orc_served = 0; self.orc_absent = 0
            self.sel_pairs = []                    # [(wid, vord_ctx)] of the served vote set T
            self.sel_wT = []                       # renormalised mixture weights over T
            self.sel_cand = []                     # [(wid, a_k, vector)] over ALL candidates (vector credit)
            self.sel_p1 = []                       # per-bit p_k(1) over T (predict-time cache)
            self.sel_keys = []                     # per-bit vote-count keys over T
            self.sel_p1all = []; self.sel_keysall = []   # per-bit over ALL candidates
            self.sel_T = []                        # candidate indices of the served vote set
            self.sel_f = 0.0; self.sel_cx = 0      # per-bit stretch(p_sel) and its context cell
            self.sel_gk = None                       # the gate key (cx) used for the byte being served
        if arm in SEL_ARMS and arm not in getattr(self, "_sel_init_done", ()) :
            pass
            self.sel_r = [0.0] * SD                # pooled features r = sum_k a_k E_k
            self.sel_uwacc = 0.0                   # per-byte sum of |vsw| (usefulness weighting)
            self.sel_Gr = [0.0] * SD               # per-byte dL/dr (mixer weights at predict time)
            self.sel_acc = []                      # per-T vote agreement accumulator (usefulness)
            self.at_n = 0; self.at_ks = []; self.at_a = []; self.at_T = []   # probe diagnostics
        self.htail = 0; self.cur = 0; self.phase = 0
        self.sbase = self.NM
        self.wh = FNV0                 # rolling FNV-1a over lowercased word bytes
        self.last_word = 0
        self.gtab = {}                 # v6: char-3gram id -> SD vector (subword table)
        self.tri_of = {}               # v6: word id -> its 3gram ids (for init + credit share)
        self.wtri = []                 # v6: rolling 3grams of the current word
        self.wtail2 = (0, 0)           # v6: last two letter bytes (for 3gram ids)
        self.inw = False               # SS77: inside a word run
        self.done = 0                  # SS77: id of the word COMPLETED by the byte just seen (else 0)
        if self.use_state:
            self.a = [AMIN + (AMAX - AMIN) * (j / (M - 1)) for j in range(M)]
            rng = random.Random(seed)
            self.W = [[(rng.random() - 0.5) * 0.1 for _ in range(XDIM)] for _ in range(M)]
            self.h = [0.0] * M
            self.eW = [[0.0] * XDIM for _ in range(M)]
            self.G = [0.0] * M
            self.emb = {}                  # word id -> D floats (learned, or frozen-random)
            self.wcount = {}               # word id -> occurrences (diagnostics only)
            self.rec = {}                  # word id -> m-vector eligibility trace (learned arm)
            self.recorder = []             # insertion order for evict-oldest

    def _octx(self, k):
        b = ORDERS[k]
        mask = (1 << (8 * b)) - 1 if b else 0
        return (self.phase, self.cur, (self.htail & mask) if b else 0)

    # ---- word id of the [A-Za-z0-9] run ENDING at byte b (prefix-visible); 0 = none ----
    def _word_after(self, b):
        if is_wb(b):
            self.wh = ((self.wh ^ (b | 32)) * 0x01000193) & 0xFFFFFFFF
            c = b | 32
            b1, b2 = self.wtail2
            if b1:                                   # a full 3gram is available
                self.wtri.append((b1 << 16) | (b2 << 8) | c)
            self.wtail2 = (b2, c)
            self.inw = True; self.done = 0
            return self.wh | (1 << 31)     # never 0
        if self.wh and self.arm in SUBWORD_ARMS:   # v6: a word just completed — stash its 3grams
            self.tri_of.setdefault(self.wh | (1 << 31), tuple(self.wtri))
        self.done = (self.wh | (1 << 31)) if self.inw else 0
        self.inw = False
        self.wh = FNV0
        self.wtri = []
        self.wtail2 = (0, 0)
        return 0

    def _vec(self, wid):
        v = self.emb.get(wid)
        if v is None:
            r = random.Random((wid * 2654435761) & 0x7FFFFFFF)
            v = [(r.random() - 0.5) * 0.1 for _ in range(D)]   # same init scale in every arm
            self.emb[wid] = v
        self.wcount[wid] = self.wcount.get(wid, 0) + 1
        return v

    def _svec(self, wid):
        v = self.semb.get(wid)
        if v is None:
            tris = self.tri_of.get(wid) if self.arm in SUBWORD_ARMS else None
            base = None
            if tris:
                acc = [0.0] * SD; n = 0
                for t in tris:
                    g = self.gtab.get(t)
                    if g is not None:
                        for k in range(SD):
                            acc[k] += g[k]
                        n += 1
                if n:
                    base = [a / n for a in acc]      # mean of TRAINED 3gram vectors
            r = random.Random((wid * 40503) & 0x7FFFFFFF)
            if base is not None:
                v = [base[k] + (r.random() - 0.5) * 0.02 for k in range(SD)]   # composed init
            else:
                v = [(r.random() - 0.5) * 0.1 for _ in range(SD)]
            self.semb[wid] = v
        self.wcount[wid] = self.wcount.get(wid, 0) + 1
        return v

    def predict(self):
        sts = [0.0] * (self.NIN + 1); i = 0
        for k in range(self.NM):
            c = self.tab[k].get(self._octx(k))
            n0, n1 = (c[0], c[1]) if c else (0, 0)
            sts[i] = stretch((n1 + 0.2) / (n0 + n1 + 0.4)); i += 1
        if self.use_state:
            if self.arm == "scrambled":
                for j in range(M):
                    sts[i] = self._sr.uniform(-1, 1); i += 1
            else:
                for j in range(M):
                    h = self.h[j]
                    sts[i] = h if -5.0 < h < 5.0 else (5.0 if h > 0 else -5.0); i += 1
            c = self.btab.get(self._bkey())
            n0, n1 = (c[0], c[1]) if c else (0, 0)
            sts[i] = stretch((n1 + 0.2) / (n0 + n1 + 0.4))
            i += 1
        if self.arm in SLOT_ARMS and self.arm not in PROJ_ARMS and self.arm not in ATTN_ARMS and self.arm not in SEL_ARMS:
            if self.arm == "scrambled":
                for j in range(S * SD):
                    sts[i] = self._sr.uniform(-1, 1); i += 1
            else:
                for w_id in self.slots:
                    if w_id:
                        v = self._svec(w_id)
                        for k in range(SD):
                            sts[i] = v[k]; i += 1
                    else:
                        i += SD
        if self.arm in PROJ_ARMS:
            if self.arm == "projscr":
                for hh in range(H):
                    sts[i] = self._sr.uniform(-1, 1); i += 1
            else:
                for hh in range(H):
                    sts[i] = self.hfeat[hh]; i += 1
        if self.arm in ATTN_ARMS:
            if self.arm == "attnscr":
                for j in range(SD + 1):
                    sts[i] = self._sr.uniform(-1, 1); i += 1
            elif self.at_n and self.arm in MOE_ARMS:
                # §79: pooled r (a = g), then ONE feature stretch(p_moe), p_moe = sum_k w_k p_k(1) over ALL candidates
                r = self.at_r
                for j in range(SD):
                    sts[i] = r[j]; i += 1
                key2 = (self.phase << 8) | self.cur
                vt = self.mvtab; vmask = (1 << VBITS) - 1; wpost = self.mo_w
                idxs = []; p1s = []; pm = 0.0
                for t, hb in enumerate(self.at_hb):
                    hx = ((hb ^ key2) * 0x94D049BB133111EB) & M64
                    ix = (hx ^ (hx >> 29)) & vmask
                    n0 = vt[2 * ix]; n1 = vt[2 * ix + 1]
                    p1 = (n1 + 0.2) / (n0 + n1 + 0.4)
                    idxs.append(ix); p1s.append(p1)
                    pm += wpost[t] * p1
                self.vt_idx = idxs; self.mo_p1 = p1s
                sts[i] = stretch(pm); i += 1
            elif self.at_n:
                r = self.at_r
                for j in range(SD):
                    sts[i] = r[j]; i += 1
                key2 = (self.phase << 8) | self.cur
                vt = self.vtab; vmask = (1 << VBITS) - 1
                idxs = []; ss = []; v = 0.0; wT = self.at_wT
                for t, hb in enumerate(self.at_hb):
                    hx = ((hb ^ key2) * 0x94D049BB133111EB) & M64
                    ix = (hx ^ (hx >> 29)) & vmask
                    n0 = vt[2 * ix]; n1 = vt[2 * ix + 1]
                    s = stretch((n1 + 0.2) / (n0 + n1 + 0.4))
                    idxs.append(ix); ss.append(s)
                    v += wT[t] * s
                self.vt_idx = idxs; self.vt_s = ss; self.vt_v = v
                sts[i] = v; i += 1
            else:
                i += SD + 1
        if self.arm in SEL_ARMS:
            if self.sel_pairs:
                r = self.sel_r
                for j in range(SD):
                    sts[i] = r[j]; i += 1
            else:
                i += SD
        if self.arm in XKEY_ARMS:
            c = self.xtab.get((self.xb << 10) | (self.phase << 7) | self.cur)
            n0, n1 = (c[0], c[1]) if c else (0, 0)
            sts[i] = stretch((n1 + 0.2) / (n0 + n1 + 0.4)); i += 1
        if self.arm == "semsim":
            for j in range(S):
                sts[i] = self.curdots[j]; i += 1
        if self.arm in W78_ARMS:
            if self.arm in PVEC_ARMS:
                v = self.pv
                if v is not None:                  # empty slot -> zero features
                    pb = self.pbase
                    for k in range(SD):
                        sts[pb + k] = v[k]
            if self.arm in CNT_ARMS:
                c = self.wctab.get((self.slots[0], self.phase, self.cur))
                n0, n1 = (c[0], c[1]) if c else (0, 0)
                sts[self.cbase] = stretch((n1 + 0.2) / (n0 + n1 + 0.4))
        if self.use_match:
            st = 0.0
            if self.mlen >= self.GATE and 0 <= self.mptr < len(self.hist):
                pb = self.hist[self.mptr]
                if self.phase == 0 or (pb >> (8 - self.phase)) == self.cur:
                    bit = (pb >> (7 - self.phase)) & 1
                    st = (1.6 + 0.35 * min(self.mlen, 28)) * (1 if bit else -1)
            sts[i] = st
        sts[self.NIN] = 1.0
        d = 0.0; w = self.w
        for j in range(self.NIN + 1):
            d += w[j] * sts[j]
        if self.arm in SEL_ARMS and self.sel_pairs:
            # §81: read EVERY candidate's vote cell (the chicken-and-egg fix: unselected
            # words' cells must train so their usefulness can differentiate); the mixture
            # pm uses only the served set T; read through a CONTEXT-SELECTED weight
            nall = len(self.sel_cand)
            p1all = [0.0] * nall
            keysall = [None] * nall
            vt = self.svt
            for t in range(nall):
                wid, vctx = self.sel_cand[t][0], self.sel_cand[t][1]
                h = (wid * 0x9E3779B97F4A7C15 ^ vctx * 0xC2B2AE3D27D4EB4F
                     ^ ((self.phase << 7 | self.cur) * 0x165667B19E3779F9)) & 0xFFFFFFFFFFFFFFFF
                ix = ((h ^ (h >> 31)) * 0xBF58476D1CE4E5B9 >> 42) & self.svmask
                j0 = 2 * ix
                p1all[t] = (vt[j0 + 1] + 0.2) / (vt[j0] + vt[j0 + 1] + 0.4)
                keysall[t] = ix
            self.sel_p1all = p1all; self.sel_keysall = keysall
            pm = 0.0
            p1s = []
            keys = []
            for t, ci in enumerate(self.sel_T):
                p1s.append(p1all[ci]); keys.append(keysall[ci])
                pm += self.sel_wT[t] * p1all[ci]
            self.sel_p1 = p1s; self.sel_keys = keys
            f = stretch(pm)
            cx = ((self.htail & 0xFF) << 10) | (self.phase << 7) | self.cur
            self.sel_f = f; self.sel_cx = cx
            d += self.vsw.get(cx, 0.0) * f
        return squash(d), sts

    def _bkey(self):
        return (self.buck << 10) | (self.phase << 7) | self.cur

    def step(self, y, learn):
        p, sts = self.predict()
        cost = -math.log2(p if y == 1 else 1 - p)
        if self.use_state and self.arm != "scrambled":
            sb = self.sbase; w = self.w; g = p - y          # dL/dz, nats
            for j in range(M):
                self.G[j] += g * w[sb + j]                  # accumulate dL/dh_j over the byte
        if self.arm in ("proj", "projr"):
            # §76: exact per-bit credit for the head features (mixer weight at predict time)
            g = p - y
            base = self.NM + M + 1; w = self.w
            for hh in range(H):
                self.hgrad[hh] += g * w[base + hh]
        if self.arm in ATTN_READ_ARMS and self.at_n:
            # SS77: exact per-bit credit (mixer weights at predict time) to r and, via the vote, to a
            g = p - y
            base = self.NM + M + 1; w = self.w; Gr = self.aGr
            for j in range(SD):
                Gr[j] += g * w[base + j]
            if self.arm in MOE_ARMS:
                # §79: within-byte posterior w_k <- w_k p_k(y) / sum_i w_i p_i(y)  (runs whether or not learn)
                wpost = self.mo_w; p1s = self.mo_p1
                if y:
                    num = [wpost[t] * p1s[t] for t in range(len(p1s))]
                else:
                    num = [wpost[t] * (1.0 - p1s[t]) for t in range(len(p1s))]
                zs = 0.0
                for x in num:
                    zs += x
                self.mo_w = [x / zs for x in num]
            elif self.arm != "attnrec":
                gv = g * w[base + SD] / self.at_AT
                v = self.vt_v; ss = self.vt_s; Ga = self.aGa
                for t in range(len(ss)):
                    Ga[t] += gv * (ss[t] - v)
        if self.arm in SEL_ARMS and self.sel_pairs:
            # §81: per-bit dL/dr at predict-time mixer weights (the vote needs no a-credit:
            # the gate is not a function of the vectors)
            g = p - y
            base = self.sel_rbase; w = self.w; Gr = self.sel_Gr
            for j in range(SD):
                Gr[j] += g * w[base + j]
            # usefulness signal: EVERY candidate's own log-likelihood on this bit
            # (agreement with the observed bit under its OWN vote cell)
            p1a = self.sel_p1all; lll = self.sel_acc
            if y:
                for t in range(len(p1a)):
                    q = p1a[t]
                    lll[t] += math.log2(q if q > 1e-9 else 1e-9)
            else:
                for t in range(len(p1a)):
                    q = p1a[t]
                    lll[t] += math.log2((1.0 - q) if q < 1.0 - 1e-9 else 1e-9)
            # §81: the readout itself identifies the bytes where the vote matters -- accumulate
            # |vsw| to weight the usefulness update (dilutes the many bytes where the vote is
            # irrelevant, which otherwise reward merely-predictable words)
            self.sel_uwacc += abs(self.vsw.get(self.sel_cx, 0.0))
        if self.arm in LEARNED_SLOT_ARMS:
            # exact credit to slot-embedding dims: mixer weight at predict time, per bit
            g = p - y
            base = self.NM + M + 1; w = self.w
            for si in range(S):
                w_id = self.slots[si]
                if w_id:
                    gr = self.sgrad.get(w_id)
                    if gr is None:
                        gr = [0.0] * SD
                        self.sgrad[w_id] = gr
                    bi = base + si * SD
                    for k in range(SD):
                        gr[k] += g * w[bi + k]
        if self.arm in PVEC_LEARN_ARMS:
            # §78A: the legacy slots credit for slots[0] only, at this arm's explicit base
            w_id = self.slots[0]
            if w_id:
                g = p - y
                w = self.w; pb = self.pbase
                gr = self.sgrad.get(w_id)
                if gr is None:
                    gr = [0.0] * SD
                    self.sgrad[w_id] = gr
                for k in range(SD):
                    gr[k] += g * w[pb + k]
        if learn:
            err = y - p
            for j in range(self.NIN + 1):
                q = err * sts[j]
                self.wg[j] = 0.999 * self.wg[j] + 0.001 * q * q
                self.w[j] += self.lr * q / (math.sqrt(self.wg[j]) + 1e-4)
            for k in range(self.NM):
                key = self._octx(k); c = self.tab[k].get(key)
                if c is None:
                    c = [0, 0]; self.tab[k][key] = c
                c[y] += 1
                if WNS and c[1 - y] > 2:
                    c[1 - y] = c[1 - y] // 2 + 1
            if self.use_state:
                key = self._bkey(); c = self.btab.get(key)
                if c is None:
                    c = [0, 0]; self.btab[key] = c
                c[y] += 1
            if self.arm in XKEY_ARMS:
                key = (self.xb << 10) | (self.phase << 7) | self.cur
                c = self.xtab.get(key)
                if c is None:
                    c = [0, 0]; self.xtab[key] = c
                c[y] += 1
            if self.arm in CNT_ARMS:
                key = (self.slots[0], self.phase, self.cur)
                c = self.wctab.get(key)
                if c is None:
                    c = [0, 0]; self.wctab[key] = c
                c[y] += 1
            if self.arm in MOE_FC_ARMS and self.at_n:
                # §79b: FULL-WEIGHT gate-independent counts n_{k,y} += 1.0 for every candidate, float halving at 255
                vt = self.mvtab
                for ix in self.vt_idx:
                    j0 = 2 * ix; jy = j0 + y
                    cy = vt[jy] + 1.0
                    if cy >= 255.0:
                        vt[jy] = cy * 0.5
                        jo = j0 + 1 - y
                        vt[jo] = vt[jo] * 0.5
                    else:
                        vt[jy] = cy
            elif self.arm in MOE_ARMS and self.at_n:
                # §79: soft counts n_{k,y} += rho_k (running posterior incl. this bit), float halving at 255
                vt = self.mvtab; rho = self.mo_w
                for t, ix in enumerate(self.vt_idx):
                    rt = rho[t]
                    if rt == 0.0:
                        continue
                    j0 = 2 * ix; jy = j0 + y
                    cy = vt[jy] + rt
                    if cy >= 255.0:
                        vt[jy] = cy * 0.5
                        jo = j0 + 1 - y
                        vt[jo] = vt[jo] * 0.5
                    else:
                        vt[jy] = cy
            elif self.arm in ATTN_READ_ARMS and self.at_n:
                vt = self.vtab
                for ix in self.vt_idx:
                    j0 = 2 * ix; jy = j0 + y
                    cy = vt[jy] + 1
                    if cy >= 255:
                        vt[jy] = (cy + 1) >> 1
                        jo = j0 + 1 - y
                        vt[jo] = (vt[jo] + 1) >> 1
                    else:
                        vt[jy] = cy
        if self.arm in SEL_ARMS and self.sel_pairs and learn:
            # §81: train the CONTEXT-SELECTED vote weight on the final error, then give every
            # served candidate FULL-WEIGHT counts (the §79b lesson: shared responsibility starves cells)
            g2 = (y - p) * self.sel_f
            cx = self.sel_cx
            gg = self.vswg.get(cx, 0.0)
            gg = 0.999 * gg + 0.001 * g2 * g2
            self.vswg[cx] = gg
            self.vsw[cx] = self.vsw.get(cx, 0.0) + self.lr_vsw * g2 / (math.sqrt(gg) + 1e-4)
            vt = self.svt
            for ix in self.sel_keysall:
                j0 = 2 * ix; jy = j0 + y
                cy = vt[jy] + 1.0
                if cy + vt[j0 + 1 - y] >= 255.0:
                    vt[jy] = cy * 0.5
                    vt[j0 + 1 - y] = vt[j0 + 1 - y] * 0.5
                else:
                    vt[jy] = cy
        self.cur = (self.cur << 1) | y; self.phase += 1
        if self.phase == 8:
            self._byte_end(learn)
        return cost

    def _byte_end(self, learn):
        b = self.cur & 0xFF
        wid = self._word_after(b)
        if self.arm in SUBWORD_ARMS and wid and wid not in self.tri_of:
            self.tri_of[wid] = tuple(self.wtri)     # stash the prefix's 3grams so far
        # §76: apply the head/vector credit for the byte just SERVED (slots still pre-LRU)
        if self.arm in ("proj", "projr") and learn:
            if self.arm == "proj":
                for k in range(S):
                    ck = self.hck[k]
                    if not ck:
                        continue
                    w_id = self.slots[k]
                    v = self.semb.get(w_id)
                    if not w_id or v is None:
                        continue
                    for dd in range(SD):
                        acc = 0.0
                        for hh in range(H):
                            acc += self.hgrad[hh] * ck * self.hw[hh][dd]
                        e = v[dd] - self.lr_s * acc
                        v[dd] = 1.0 if e > 1.0 else (-1.0 if e < -1.0 else e)
            for hh in range(H):
                g = self.hgrad[hh]
                if g:
                    hdw = self.hdw[hh]
                    wh = self.hw[hh]
                    for dd in range(SD):
                        wh[dd] -= self.lr_head * g * hdw[dd]
            self.hgrad = [0.0] * H
        if self.arm in ATTN_READ_ARMS:
            # SS77: apply the attention credit for the byte just SERVED (slots still pre-LRU)
            if self.arm in MOE_ARMS:
                self.mo_rlast = self.mo_w if self.at_n else None     # §79: served byte's responsibilities r
            if learn and self.at_n:
                self._attn_apply(self._attn_grads())
            self.aGr = [0.0] * SD
            self.aGa = [0.0] * len(self.at_T)
        if self.arm in SEL_ARMS:
            # §81: same timing contract (the byte just SERVED, slots still pre-LRU)
            self._sel_apply(learn)
            if b == 46:                      # '.' ends the sentence: candidates are scoped to
                self.slots = [0] * S         # the CURRENT sentence (a 32-deep LRU otherwise lets
                                             # the previous sentence's cue pollute its own vote
                                             # cells with foreign outcomes -- measured 183 updates
                                             # for 37 own sentences before this fix)
        swid = wid if SLOTMODE == "prefix" else self.done
        if (self.arm in SLOT_ARMS or self.arm in W78_ARMS) and swid:
            sl = self.slots
            if swid in sl:
                sl.remove(swid)
            else:
                del sl[-1]
            sl.insert(0, swid)
        if self.arm in ("proj", "projr") and self.use_state:
            # §76: refresh head features + exact-credit caches from the post-LRU slots
            den = 0.0
            for k in range(S):
                if self.slots[k]:
                    den += GP[k]
            if den <= 0.0:
                den = 1.0
            for k in range(S):
                self.hck[k] = (GP[k] / den) if self.slots[k] else 0.0
            for hh in range(H):
                wh = self.hw[hh]
                num = 0.0
                acc = [0.0] * SD
                for k in range(S):
                    w_id = self.slots[k]
                    if not w_id:
                        continue
                    v = self.semb.get(w_id)
                    if v is None:
                        v = self._svec(w_id)
                    c = GP[k]
                    d = 0.0
                    for dd in range(SD):
                        d += wh[dd] * v[dd]
                        acc[dd] += c * v[dd]
                    num += c * d
                self.hfeat[hh] = num / den
                self.hdw[hh] = [a / den for a in acc]
        if self.arm in LEARNED_SLOT_ARMS and self.sgrad:
            for w_id, gr in self.sgrad.items():
                v = self.semb.get(w_id)
                if v is None:
                    continue
                for k in range(SD):
                    e = v[k] - self.lr_s * gr[k]
                    v[k] = 1.0 if e > 1.0 else (-1.0 if e < -1.0 else e)
                if self.arm in SUBWORD_ARMS:
                    # v6: share the credit with the word's char-3grams (1/len each) so future
                    # UNSEEN words composing those 3grams start inside their morphological family
                    tris = self.tri_of.get(w_id)
                    if tris:
                        sc = self.lr_s / len(tris)
                        for t in tris:
                            gv = self.gtab.get(t)
                            if gv is None:
                                gv = [0.0] * SD
                                self.gtab[t] = gv
                            for k in range(SD):
                                e = gv[k] - sc * gr[k]
                                gv[k] = 1.0 if e > 1.0 else (-1.0 if e < -1.0 else e)
            self.sgrad.clear()
        if self.arm in W78_ARMS:
            # §78A: apply the credit for the byte just served (only when learn), then refresh slots[0]'s vector
            if self.sgrad:
                if learn:
                    lr = self.lr_s
                    for w_id, gr in self.sgrad.items():
                        v = self.semb[w_id]
                        for k in range(SD):
                            e = v[k] - lr * gr[k]
                            v[k] = 1.0 if e > 1.0 else (-1.0 if e < -1.0 else e)
                self.sgrad.clear()
            if self.arm in PVEC_ARMS:
                s0 = self.slots[0]
                self.pv = self._svec(s0) if s0 else None
        if self.use_state and self.arm != "scrambled" and learn:
            # 1) per-word eligibility traces (v2 long-range credit): decay every recent word's
            #    trace; the word that shaped the state serving THIS byte collects its G fresh
            if self.arm == "learned":
                a = self.a; G = self.G; lw = self.last_word
                for wj, A in self.rec.items():
                    for j in range(M):
                        A[j] = a[j] * A[j]
                if lw:
                    A = self.rec.get(lw)
                    if A is None:
                        if len(self.recorder) >= MAXREC:            # evict oldest -> flush
                            old = self.recorder.pop(0)
                            A = self.rec.pop(old, None)
                            if A is not None:
                                self._flush(old, A)
                        A = [0.0] * M
                        self.rec[lw] = A
                        self.recorder.append(lw)
                    for j in range(M):
                        A[j] += (1.0 - a[j]) * G[j]
            # 2) exact diagonal RTRL for W with THIS byte's learning signal
            for j in range(M):
                gj = self.G[j]
                if gj:
                    Wj = self.W[j]; eWj = self.eW[j]
                    for k in range(XDIM):
                        e = Wj[k] - self.lr_rec * gj * eWj[k]
                        Wj[k] = 2.0 if e > 2.0 else (-2.0 if e < -2.0 else e)
        # 3) build x, roll eligibility traces and state (h(t+1) uses the UPDATED W)
        if self.use_state:
            x = [0.0] * XDIM
            for k in range(8):
                x[k] = 2.0 * ((b >> k) & 1) - 1.0
            if self.arm == "scrambled":
                for k in range(XDIM):
                    x[k] = self._sr.uniform(-1, 1)
            elif self.arm in ("learned", "randemb") and wid:
                x[8:] = self._vec(wid)
            a = self.a; W = self.W; eW = self.eW; h = self.h
            for j in range(M):
                aj = a[j]; Wj = W[j]; eWj = eW[j]
                wr = 0.0
                for k in range(XDIM):
                    wr += Wj[k] * x[k]
                    eWj[k] = (1.0 - aj) * x[k] + aj * eWj[k]
                h[j] = aj * h[j] + (1.0 - aj) * wr
            buck = 0
            for u in range(12, 28):
                buck = (buck << 1) | (1 if h[u] > 0.0 else 0)
            self.buck = buck
            self.G = [0.0] * M
            self.last_word = wid
        if self.arm in XKEY_ARMS:
            # v5 (revised): key on the LAST COMPLETED word (slots[0]) — its vector is TRAINED
            # (it was slot-resident), the key fires at word boundaries AND interiors, and the
            # identity/embedding arms key the SAME positions. (The first version keyed on the
            # growing prefix id, whose vectors are untrained for multi-letter interiors and dead
            # at boundaries — a flaw the identity control is immune to, i.e. an unfair A/B.)
            w_id = self.slots[0] if self.use_state else 0
            if not w_id:
                self.xb = 0; self.curv = None
                for si in range(S):
                    self.curdots[si] = 0.0
            elif self.arm == "slotsw":
                self.xb = (((w_id * 2654435761) & 0xFFFFFFFF) >> 20) & 0xFFF     # 4096 identity slots
            else:
                v = self._svec(w_id)
                xb = 0
                for k in range(SD):
                    xb = (xb << 1) | (1 if v[k] > 0.0 else 0)
                self.xb = xb
                self.curv = v
                self.curdots[0] = 0.0                       # slot 0 IS the key word; skip self
                for si in range(1, S):
                    w2 = self.slots[si]
                    if w2:
                        sv = self.semb.get(w2)
                        if sv is None:
                            sv = self._svec(w2)
                        d = 0.0
                        for k in range(SD):
                            d += v[k] * sv[k]
                        self.curdots[si] = d
                    else:
                        self.curdots[si] = 0.0
        self.htail = ((self.htail << 8) | b) & ((1 << 48) - 1); self.cur = 0; self.phase = 0
        if self.arm in ATTN_READ_ARMS:
            self._attn_forward()       # post-LRU slots, post-htail context -> serves the next byte
        if self.arm in SEL_ARMS:
            self._sel_forward()        # §81: same timing contract (post-LRU, post-htail)
        if self.use_match:
            self._match_after(b)
        return None

    # ---- §81 the simplest selector ----
    def _sel_forward(self):
        """query-free gate (usefulness + position bias) over the current slots -> serves the next byte."""
        sl = self.slots
        if not sl[0]:
            self.sel_pairs = []; self.sel_wT = []; self.sel_cand = []; self.sel_acc = []
            self.sel_T = []; self.sel_p1all = []; self.sel_keysall = []
            self.sel_r = [0.0] * SD; self.sel_Gr = [0.0] * SD
            self.at_n = 0; self.at_ks = []; self.at_a = []; self.at_T = []
            return
        ks = [k for k in range(S) if sl[k]]
        E = []
        for k in ks:
            v = self.semb.get(sl[k])
            if v is None:
                v = self._svec(sl[k])
            E.append(v)
        n = len(ks)
        e = []
        # §81: the usefulness key = the SAME 3-byte (VORD) context the vote cells key on --
        # one byte of context mixes the outcome byte with every word-start byte and the
        # separation dies (measured); three bytes isolate "after then " from "after golf "
        gk = (self.htail & ((1 << (8 * VORD)) - 1)) if self.arm == "sel" else None
        self.sel_gk = gk
        for k in ks:
            if self.arm == "sel":
                ev = SEL_UGAIN * self.suw.get((gk, sl[k]), 0.0) - SEL_PBD * k
                e.append(12.0 if ev > 12.0 else (-12.0 if ev < -12.0 else ev))
            else:
                e.append(-SEL_PBD * k)
        mx = max(e)
        ex = [math.exp((x - mx) / SEL_TSC) for x in e]
        z = sum(ex)
        a = [x / z for x in ex]
        if self.arm == "selorc4":
            self.orc_served += 1
            ow = self.oracle_wid
            if ow and ow in sl:
                ci = ks.index(sl.index(ow))
                by_a = [t for t in sorted(range(n), key=lambda t: -a[t]) if t != ci]
                T = [ci] + by_a[:SEL_TOPM - 1]
            else:
                self.orc_absent += 1
                T = sorted(range(n), key=lambda t: -a[t])[:SEL_TOPM]
        else:
            T = sorted(range(n), key=lambda t: -a[t])[:SEL_TOPM]   # stable: ties -> more recent slot
        AT = 0.0
        for t in T:
            AT += a[t]
        if AT <= 0.0:
            AT = 1.0
        r = [0.0] * SD
        for t in range(n):
            at_ = a[t]; Et = E[t]
            for j in range(SD):
                r[j] += at_ * Et[j]
        vctx = self.htail & ((1 << (8 * VORD)) - 1)
        self.sel_pairs = [(sl[ks[t]], vctx) for t in T]
        self.sel_wT = [a[t] / AT for t in T]
        self.sel_T = list(T)                        # candidate indices of the served vote set
        self.sel_cand = [(sl[ks[t]], vctx, a[t], E[t]) for t in range(n)]
        self.sel_acc = [0.0] * n                    # per-candidate per-byte log2-likelihood
        self.sel_r = r
        # probe diagnostics (attn-compatible names)
        self.at_n = n; self.at_ks = ks; self.at_a = a; self.at_T = T; self.at_AT = AT

    def _sel_apply(self, learn):
        """§81: end-of-byte credit for the SERVED byte (call pre-LRU): per-word usefulness from
        each candidate's OWN log-likelihood advantage over the candidate mean (so cells that
        specialise rise, marginals sink); vectors through the pooled features r."""
        if not self.sel_pairs:
            self.sel_Gr = [0.0] * SD
            self.sel_uwacc = 0.0
            return
        if learn:
            if self.arm == "sel" and self.sel_acc and self.sel_gk is not None:
                lll = self.sel_acc
                mu = sum(lll) / len(lll)
                wu = self.sel_uwacc / 8.0            # mean |vsw| over the byte's bits
                if wu > 1.0:
                    wu = 1.0
                if wu > 0.02:                        # bytes where the vote is wanted at all
                    for t in range(len(lll)):
                        adv = lll[t] - mu
                        if adv > 1.0: adv = 1.0
                        elif adv < -1.0: adv = -1.0
                        wid = self.sel_cand[t][0]
                        uk = (self.sel_gk, wid)       # contextual: (byte's first-bit cx, word)
                        u = self.suw.get(uk, 0.0)
                        u = SEL_UDC * u + (1.0 - SEL_UDC) * wu * adv
                        self.suw[uk] = max(-SEL_UCLIP, min(SEL_UCLIP, u))
            Gr = self.sel_Gr
            for wid, _vctx, at_, v in self.sel_cand:
                for j in range(SD):
                    e_ = v[j] - self.lr_s * at_ * Gr[j]
                    v[j] = 1.0 if e_ > 1.0 else (-1.0 if e_ < -1.0 else e_)
        self.sel_Gr = [0.0] * SD
        self.sel_uwacc = 0.0
        self.sel_acc = [0.0] * len(self.sel_cand)

    # ---- SS77 attentional read ----
    def _attn_forward(self):
        """pooled features, top-m vote set and backward caches from the CURRENT slots/params."""
        sl = self.slots
        if not sl[0]:
            self.at_n = 0; self.at_T = []; self.at_hb = []; self.aGa = []
            self.aGr = [0.0] * SD
            return
        ks = [k for k in range(S) if sl[k]]
        E = []
        for k in ks:
            v = self.semb.get(sl[k])
            if v is None:
                v = self._svec(sl[k])
            E.append(v)
        n = len(ks)
        if self.arm == "attnrec":
            den = 0.0
            for k in ks:
                den += GP[k]
            a = [GP[k] / den for k in ks]
            c = None; q = None
        elif self.arm in MOE_FIXED_GATE:
            # §79: frozen gate. uni: 1/n; orc: one-hot on oracle_wid's slot (1/n if 0 or absent)
            c = None; q = None
            a = [1.0 / n] * n
            if self.arm == "attnmoe_orc":
                self.orc_served += 1
                ow = self.oracle_wid
                if ow and ow in sl:
                    a = [0.0] * n
                    a[ks.index(sl.index(ow))] = 1.0
                else:
                    self.orc_absent += 1
        else:
            c = list(E[0])
            q = [0.0] * SD
            for i2 in range(SD):
                Wi = self.Wq[i2]; acc = self.bq[i2]
                for j in range(SD):
                    acc += Wi[j] * c[j]
                q[i2] = acc
            e = [0.0] * n
            if self.arm == "attncos":
                # SS77c: e_k = KAPPA (q.E_k) / (|q||E_k| + 1e-8) + beta_k
                nq = 0.0
                for j in range(SD):
                    nq += q[j] * q[j]
                nq = math.sqrt(nq)
                nE = [0.0] * n; sc = [0.0] * n; dn = [0.0] * n
                for t in range(n):
                    Et = E[t]; d = 0.0; ne = 0.0
                    for j in range(SD):
                        d += q[j] * Et[j]
                        ne += Et[j] * Et[j]
                    ne = math.sqrt(ne)
                    den = nq * ne + 1e-8
                    nE[t] = ne; sc[t] = d; dn[t] = den
                    e[t] = KAPPA * d / den + self.beta[ks[t]]
                self.at_cos = (nq, nE, sc, dn)
            else:
                for t in range(n):
                    Et = E[t]; d = self.beta[ks[t]]
                    for j in range(SD):
                        d += q[j] * Et[j]
                    e[t] = d
            mx = max(e)
            ex = [math.exp(x - mx) for x in e]
            z = sum(ex)
            a = [x / z for x in ex]
        r = [0.0] * SD
        for t in range(n):
            at = a[t]; Et = E[t]
            for j in range(SD):
                r[j] += at * Et[j]
        if self.arm in MOE_ARMS:
            T = list(range(n))                          # §79: every non-empty slot is a candidate expert
        elif self.arm in ORACLE_ARMS:
            T = self._oracle_T(sl, ks, a, n)
        else:
            T = sorted(range(n), key=lambda t: -a[t])[:TOPM]      # stable: ties -> more recent slot
        AT = 0.0
        for t in T:
            AT += a[t]
        if not T:
            AT = 1.0                                   # §78B attnorc1, oracle absent: no vote (feature 0)
        ctx = self.htail & ((1 << (8 * VORD)) - 1)
        hb = []
        for t in T:
            h = ((sl[ks[t]] * 0x9E3779B97F4A7C15) ^ (ctx * 0xC2B2AE3D27D4EB4F)) & M64
            h = ((h ^ (h >> 31)) * 0xBF58476D1CE4E5B9) & M64
            hb.append(h ^ (h >> 27))
        self.at_n = n; self.at_ks = ks; self.at_E = E; self.at_a = a; self.at_T = T
        self.at_AT = AT; self.at_wT = [a[t] / AT for t in T]; self.at_c = c; self.at_q = q
        self.at_r = r; self.at_hb = hb
        self.aGr = [0.0] * SD; self.aGa = [0.0] * len(T)
        if self.arm in MOE_ARMS:
            self.mo_w = list(a)                         # §79: posterior at the byte's first bit = gate prior g

    def _oracle_T(self, sl, ks, a, n):
        """§78B forced vote set (positions into ks). attnorc1: {oracle slot} or empty when absent;
        attnorc4: {oracle slot} + top-(TOPM-1) others by a, or attn's top-TOPM when absent."""
        self.orc_served += 1
        ow = self.oracle_wid
        to = ks.index(sl.index(ow)) if (ow and ow in sl) else -1
        if to < 0:
            self.orc_absent += 1
            if self.arm == "attnorc1":
                return []
            return sorted(range(n), key=lambda t: -a[t])[:TOPM]
        if self.arm == "attnorc1":
            return [to]
        return [to] + [t for t in sorted(range(n), key=lambda t: -a[t]) if t != to][:TOPM - 1]

    def _attn_grads(self):
        """exact gradients of the byte's summed nat loss (params frozen within the byte).
        returns (dE per served slot position, dWq, dbq, dbeta{k: g}); query parts None for attnrec."""
        n = self.at_n; E = self.at_E; a = self.at_a; Gr = self.aGr
        if self.arm == "attnrec" or self.arm in MOE_FIXED_GATE:
            dE = [[a[t] * Gr[j] for j in range(SD)] for t in range(n)]
            return dE, None, None, None
        dA = [0.0] * n
        for t in range(n):
            Et = E[t]; d = 0.0
            for j in range(SD):
                d += Gr[j] * Et[j]
            dA[t] = d
        if self.arm not in MOE_ARMS:
            for u, t in enumerate(self.at_T):
                dA[t] += self.aGa[u]
        sbar = 0.0
        for t in range(n):
            sbar += a[t] * dA[t]
        de = [a[t] * (dA[t] - sbar) for t in range(n)]
        if self.arm in MOE_RESP_ARMS:
            rr = self.mo_w                              # §79: after bit 8 the posterior IS r_k = g_k P_k / sum g P
            for t in range(n):
                de[t] += a[t] - rr[t]                   # d(-ln sum_k g_k P_k)/de_k
        q = self.at_q; c = self.at_c
        dq = [0.0] * SD
        if self.arm == "attncos":
            # SS77c: e_k = K s_k / den_k + beta_k,  s_k = q.E_k,  den_k = |q||E_k| + 1e-8
            #   de_k/dq   = K E_k / den_k - K s_k |E_k| (q/|q|)   / den_k^2
            #   de_k/dE_k = K q   / den_k - K s_k |q|   (E_k/|E_k|) / den_k^2
            # (|q| = 0 forces s_k = 0, so the second term's exact limit is 0; likewise |E_k| = 0.)
            nq, nE, sc, dn = self.at_cos
            dE = [[a[t] * Gr[j] for j in range(SD)] for t in range(n)]
            for t in range(n):
                g1 = de[t] * KAPPA / dn[t]
                h = de[t] * KAPPA * sc[t] / (dn[t] * dn[t])
                hq = h * nE[t] / nq if nq > 0.0 else 0.0
                hE = h * nq / nE[t] if nE[t] > 0.0 else 0.0
                Et = E[t]; dEt = dE[t]
                for j in range(SD):
                    dq[j] += g1 * Et[j] - hq * q[j]
                    dEt[j] += g1 * q[j] - hE * Et[j]
        else:
            for t in range(n):
                dt = de[t]; Et = E[t]
                for j in range(SD):
                    dq[j] += dt * Et[j]
        dbeta = {self.at_ks[t]: de[t] for t in range(n)}
        dWq = [[dq[i2] * c[j] for j in range(SD)] for i2 in range(SD)]
        dbq = None if self.arm in MOE_NOBQ_ARMS else list(dq)     # §79b attnmoe_fcnb: bq fixed at 0
        if self.arm != "attncos":
            dE = [[a[t] * Gr[j] + de[t] * q[j] for j in range(SD)] for t in range(n)]
        Wq = self.Wq; d0 = dE[0]                     # slot 0 is also the query source
        for j in range(SD):
            acc = 0.0
            for i2 in range(SD):
                acc += Wq[i2][j] * dq[i2]
            d0[j] += acc
        return dE, dWq, dbq, dbeta

    def _attn_apply(self, grads):
        dE, dWq, dbq, dbeta = grads
        if self.arm != "attnr":
            lr = self.lr_s
            for t in range(self.at_n):
                v = self.at_E[t]; g = dE[t]
                for j in range(SD):
                    e = v[j] - lr * g[j]
                    v[j] = 1.0 if e > 1.0 else (-1.0 if e < -1.0 else e)
        if dWq is not None:
            lr = self.lr_head
            for i2 in range(SD):
                Wi = self.Wq[i2]; gi = dWq[i2]
                for j in range(SD):
                    e = Wi[j] - lr * gi[j]
                    Wi[j] = 8.0 if e > 8.0 else (-8.0 if e < -8.0 else e)
                if dbq is not None:
                    e = self.bq[i2] - lr * dbq[i2]
                    self.bq[i2] = 8.0 if e > 8.0 else (-8.0 if e < -8.0 else e)
            for k, g in dbeta.items():
                e = self.beta[k] - lr * g
                self.beta[k] = 8.0 if e > 8.0 else (-8.0 if e < -8.0 else e)

    def _match_after(self, b):
        """genmem's match model update: extend/break the candidate, register the context."""
        n = len(self.hist)
        self.hist.append(b)
        if self.mlen > 0 and self.mptr < n - 1:
            if self.hist[self.mptr] == self.hist[n - 1]:
                self.mptr += 1; self.mlen = min(self.mlen + 1, 65535)
            else:
                self.mlen = 0; self.mptr = -1
        if n >= self.MINLEN:
            key = bytes(self.hist[n - self.MINLEN:n])
            prev = self.mtab.get(key, -1)
            self.mtab[key] = n
            if self.mlen == 0 and 0 <= prev < n:
                self.mptr = prev; self.mlen = self.MINLEN

    def _flush(self, wid, A):
        """apply accumulated eligibility: E[wid] -= lr_emb * (A . W[:, emb])"""
        v = self.emb.get(wid)
        if v is None:
            return
        W = self.W
        for k in range(D):
            kk = XDIM_BITS + k
            acc = 0.0
            for j in range(M):
                acc += A[j] * W[j][kk]
            e = v[k] - self.lr_emb * acc
            v[k] = 1.0 if e > 1.0 else (-1.0 if e < -1.0 else e)

    def _flush_all(self):
        for wid, A in self.rec.items():
            self._flush(wid, A)
        self.rec.clear(); self.recorder.clear()

    def run(self, raw, learn, mask=None):
        tot = 0.0; nb = 0; bp = 0
        for b in raw:
            scored = (mask is None) or (bp < len(mask) and mask[bp])
            bc = 0.0
            for j in range(7, -1, -1):
                bc += self.step((b >> j) & 1, learn)
            if scored:
                tot += bc; nb += 1
            bp += 1
        if learn and self.arm == "learned":
            self._flush_all()
        return tot / max(1, nb)


# ---------------- worker: one arm across all sizes ----------------
def run_arm(arm, sizes, seed=0, train_path="data/wt103_train.txt", test_path="data/wt103_test.txt"):
    train_all = open(train_path, "rb").read()
    test_all = open(test_path, "rb").read()
    rows = []
    t0 = time.time()
    model = None
    tr = te = b""
    for kb in sizes:
        tr = train_all[:kb * 1024]
        te = test_all[:min(400 * 1024, kb * 1024 // 2)]
        mask = clean_mask_det(tr, te, 13)
        m = Model(arm=arm, seed=seed)
        m.run(tr, learn=True)
        bpb = m.run(te, learn=True, mask=mask)
        rows.append((kb, bpb))
        model = m
    return arm, rows, time.time() - t0, neighbor_payload(model, tr, te)


def id2str(data):
    """rebuild word-id -> string with the exact rolling hash the model uses (diagnostics)."""
    m = {}; s = bytearray(); h = FNV0
    for b in data:
        if is_wb(b):
            h = ((h ^ (b | 32)) * 0x01000193) & 0xFFFFFFFF
            s.append(b | 32)
        else:
            if s:
                m.setdefault(h | (1 << 31), bytes(s).decode("latin1"))
            h = FNV0; s = bytearray()
    if s:
        m.setdefault(h | (1 << 31), bytes(s).decode("latin1"))
    return m


def neighbor_payload(model, tr, te):
    """for vector-armed models: top-frequency words (len>=3) + 3 nearest neighbours by cosine."""
    if model is None:
        return None
    table = model.semb if model.arm in SLOT_ARMS else getattr(model, "emb", {})
    if not table:
        return None
    ids = id2str(tr + te)
    top = [(w, c) for w, c in sorted(model.wcount.items(), key=lambda kv: -kv[1])
           if len(ids.get(w, "")) >= 3][:40]
    vecs = [(wid, cnt, table[wid]) for wid, cnt in top if wid in table]
    out = []
    for i, (wa, ca, va) in enumerate(vecs):
        best = []
        for j, (wb_, cb, vb) in enumerate(vecs):
            if i == j:
                continue
            na = math.sqrt(sum(t * t for t in va)) or 1e-9
            nb_ = math.sqrt(sum(t * t for t in vb)) or 1e-9
            cos = sum(t * u for t, u in zip(va, vb)) / (na * nb_)
            best.append((cos, ids.get(wb_, hex(wb_))))
        best.sort(reverse=True)
        out.append((ids.get(wa, hex(wa)), ca, [(f"{c:+.2f}", s) for c, s in best[:3]]))
    return out[:12]


# ---------------- main ----------------
ARMS = ["baseline", "slots", "slotsw", "semsim", "semfast", "scrambled"]


def report(results, sizes, secs):
    by = {arm: dict(rows) for arm, rows, _, _ in results}
    present = set(by)
    print()
    hdr = (f"{'train':>8}" + "".join(f"{a:>10}" for a in ARMS))
    print(hdr); print("-" * len(hdr))
    for kb in sizes:
        print(f"{kb:>6}KB" + "".join(f"{by[a][kb]:>10.4f}" for a in ARMS))
    print("-" * len(hdr))
    def series(f):
        return [round(f(by, kb), 4) for kb in sizes]
    if "slots" in present and "baseline" in present:
        help_s = series(lambda B, kb: B["baseline"][kb] - B["slots"][kb])     # >0 slots help
        print(f"slots  vs baseline (the crossing)   : {help_s}")
        if "slotsr" in present:
            sem_s = series(lambda B, kb: B["slotsr"][kb] - B["slots"][kb])    # >0 MEANING vs identity
            print(f"slots  vs slotsr (MEANING)         : {sem_s}")
        if "scrambled" in present:
            noi_s = series(lambda B, kb: B["scrambled"][kb] - B["slots"][kb]) # >0 real signal
            print(f"slots  vs scrambled (noise floor)  : {noi_s}")
            if any(h > 0.001 for h in help_s) and all(n > 0.003 for n in noi_s):
                verdict_s = "CROSS -- word-slot binding beats the orders baseline on THIS data too"
            elif all(n > 0.003 for n in noi_s):
                verdict_s = "signal present (above noise) but no cross at these sizes"
            else:
                verdict_s = "NEGATIVE on this data"
            print(f"  VERDICT (slots): {verdict_s}")
    if {"semsim", "baseline", "slotsw"} <= present:
        help_m = series(lambda B, kb: B["baseline"][kb] - B["semsim"][kb])
        sim_id = series(lambda B, kb: B["slotsw"][kb] - B["semsim"][kb])
        sim_marg = series(lambda B, kb: B["slots"][kb] - B["semsim"][kb]) if "slots" in present else None
        print(f"semsim vs baseline (help)          : {help_m}")
        print(f"semsim vs slotsw (SIMILARITY>IDENT): {sim_id}")
        if sim_marg is not None:
            print(f"semsim vs slots (layer margin)     : {sim_marg}")
    if "semfast" in present and {"semsim", "slotsw"} <= present:
        sub = series(lambda B, kb: B["semsim"][kb] - B["semfast"][kb])       # >0 subword init helps
        sub_id = series(lambda B, kb: B["slotsw"][kb] - B["semfast"][kb])    # >0 subword+sem beats identity
        print(f"semfast vs semsim (subword init)   : {sub}")
        print(f"semfast vs slotsw (vs identity)    : {sub_id}")
    if "proj" in present and "baseline" in present:
        p_help = series(lambda B, kb: B["baseline"][kb] - B["proj"][kb])          # >0 deep readout crosses
        print(f"proj   vs baseline (deep crossing)   : {p_help}")
        if "projr" in present:
            p_sem = series(lambda B, kb: B["projr"][kb] - B["proj"][kb])          # >0 MEANING
            print(f"proj   vs projr (MEANING)           : {p_sem}")
        if "projscr" in present:
            p_noi = series(lambda B, kb: B["projscr"][kb] - B["proj"][kb])        # >0 real signal
            print(f"proj   vs projscr (noise floor)     : {p_noi}")
            if any(h > 0.001 for h in p_help) and all(n > 0.003 for n in p_noi):
                print("  VERDICT (proj): CROSS -- the parameter-efficient deep readout crosses.")
            elif all(n > 0.003 for n in p_noi):
                print("  VERDICT (proj): signal present, no cross at these sizes.")
            else:
                print("  VERDICT (proj): NEGATIVE.")
    if "attn" in present and "baseline" in present:
        a_help = series(lambda B, kb: B["baseline"][kb] - B["attn"][kb])         # >0 crosses
        print(f"attn   vs baseline (crossing)        : {a_help}")
        a_sel = None
        if "attnr" in present:
            a_sem = series(lambda B, kb: B["attnr"][kb] - B["attn"][kb])         # >0 MEANING
            print(f"attn   vs attnr (MEANING)           : {a_sem}")
        if "attnrec" in present:
            a_sel = series(lambda B, kb: B["attnrec"][kb] - B["attn"][kb])       # >0 content beats recency
            print(f"attn   vs attnrec (SELECTION)       : {a_sel}")
        if "attnscr" in present:
            a_noi = series(lambda B, kb: B["attnscr"][kb] - B["attn"][kb])       # >0 real signal
            print(f"attn   vs attnscr (floor)           : {a_noi}")
            if any(h > 0.001 for h in a_help) and all(x > 0.003 for x in a_noi):
                v_a = "CROSS -- the attentional read crosses the orders baseline"
            elif all(x > 0.003 for x in a_noi):
                v_a = "signal present, no cross at these sizes"
            else:
                v_a = "NEGATIVE"
        else:
            v_a = "no floor arm (attnscr) -- verdict undetermined"
        if a_sel is not None:
            v_a += ("; SELECTION > 0 at every size (content beats recency)" if all(x > 0 for x in a_sel)
                    else "; SELECTION NOT > 0 at every size (content does not beat recency)")
        else:
            v_a += "; SELECTION untested (no attnrec arm)"
        print(f"  VERDICT (attn): {v_a}.")
    if present & set(W78_ARMS):
        # §78A: is the learned prefix vector more than a count word model? (values are bpb differences; > 0 means the subtracted arm compresses better)
        def raw(f):
            return [f(by, kb) for kb in sizes]
        if {"wcnt", "baseline"} <= present:
            print(f"wcnt   vs baseline (count word model)  : {series(lambda B, kb: B['baseline'][kb] - B['wcnt'][kb])}")
        if {"pvec", "baseline"} <= present:
            print(f"pvec   vs baseline (the §72 channel)   : {series(lambda B, kb: B['baseline'][kb] - B['pvec'][kb])}")
        if {"pvecr", "baseline"} <= present:
            print(f"pvecr  vs baseline (frozen vectors)    : {series(lambda B, kb: B['baseline'][kb] - B['pvecr'][kb])}")
        if {"wcnt", "wcntpvec"} <= present:
            a1 = raw(lambda B, kb: B["wcnt"][kb] - B["wcntpvec"][kb])
            ok1 = all(x >= 0.002 for x in a1)
            print(f"A1 wcnt - wcntpvec (vector beyond count): {[round(x, 4) for x in a1]}"
                  f"  -> >=+0.002 at every size on THIS corpus: {'yes' if ok1 else 'NO'}")
        if {"wcntpvecr", "wcntpvec"} <= present:
            a2 = raw(lambda B, kb: B["wcntpvecr"][kb] - B["wcntpvec"][kb])
            ok2 = all(x >= 0.002 for x in a2)
            print(f"A2 wcntpvecr - wcntpvec (learning)      : {[round(x, 4) for x in a2]}"
                  f"  -> >=+0.002 at every size on THIS corpus: {'yes' if ok2 else 'NO'}")
        print("  (the pre-registered §78A verdict needs A1 and A2 at every size on BOTH code and wt103, then stdlib;"
              " see the module docstring)")
    if "attncos" in present:
        if "baseline" in present:
            c_help = series(lambda B, kb: B["baseline"][kb] - B["attncos"][kb])  # >0 crosses
            print(f"attncos vs baseline (crossing)       : {c_help}")
        if "attnrec" in present:
            c_sel = series(lambda B, kb: B["attnrec"][kb] - B["attncos"][kb])    # >0 content beats recency
            print(f"attncos vs attnrec (SELECTION)       : {c_sel}")
    print(f"\n[{secs:.0f}s total]")


def main():
    global ARMS
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    selftest = "--selftest" in sys.argv
    sizes = [40] if selftest else [150, 450, 1200, 2700]
    train_path = "data/wt103_train.txt"
    test_path = "data/wt103_test.txt"
    if "--train" in sys.argv:
        train_path = sys.argv[sys.argv.index("--train") + 1]
        test_path = sys.argv[sys.argv.index("--test") + 1]
    if "--sizes" in sys.argv:
        sizes = [int(x) for x in sys.argv[sys.argv.index("--sizes") + 1].split(",")]
    if "--arms" in sys.argv:
        arms = sys.argv[sys.argv.index("--arms") + 1].split(",")
    else:
        arms = ARMS
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    ARMS = arms                       # report() prints what actually ran
    print("=" * 104)
    print("SEMANTIC-STATE instrument (SS72-74 lineage)")
    print(f"  data: {train_path} / {test_path}")
    print(f"  arms: {' | '.join(ARMS)}   sizes(KB): {sizes}   seed: {seed}")
    print(f"  slots: WSLOTMODE={SLOTMODE}  WSLOTS={S}  WHEADS={H}  WGAMMA={GAMMA}"
          f"  WTOPM={TOPM}  WVORD={VORD}  WVBITS={VBITS}  WKAPPA={KAPPA}")
    print("  gate: copy/match OFF, 13-byte-decontaminated held-out, deterministic mask")
    print("=" * 104)
    t0 = time.time()
    if selftest:
        results = [run_arm(a, sizes, seed, train_path, test_path) for a in ARMS]
    else:
        with ProcessPoolExecutor(max_workers=len(ARMS)) as ex:
            futs = [ex.submit(run_arm, a, sizes, seed, train_path, test_path) for a in ARMS]
            results = [f.result() for f in futs]
    secs = time.time() - t0
    for arm, rows, dt, _ in results:
        print(f"  [{arm:<10}] " + "  ".join(f"{kb}KB={b:.4f}" for kb, b in rows) + f"   [{dt:.0f}s]")
    report(results, sizes, secs)
    for arm, rows, dt, nn in results:
        if nn:
            print(f"\nEMBEDDING NEIGHBOURS ({arm} arm, top words, cosine):")
            for w, c, nbrs in nn:
                print(f"  {w:<14} x{c:<6} ->  " + "   ".join(f"{s}({cs})" for cs, s in nbrs))
    print("=" * 104)


if __name__ == "__main__":
    main()
