# [WORKING TITLE: Does Test-Time Dynamics Beat a Table at Matched Bytes? A Protocol, and an Audit That Disqualifies Its Own Authors]

**[AUTHORS PLACEHOLDER]**

> **Venue class.** TAE-class: **≤ 8 pages of main text excluding references and appendices**; appendices
> unlimited; double-blind; non-archival. Reviewers here are not required to read appendices, so **every
> load-bearing result, control and defense is in the main text**; appendices carry full cell tables, flag
> provenance, pre-registration scorecards, declared NOT-RUNs and the negatives ledger. *Placeholders per
> drafting policy: title `[WORKING TITLE: …]`, authorship `[AUTHORS PLACEHOLDER]`; both workshopped at the end,
> both blank in any anonymized build.*

## Abstract

Modern sequence memories — test-time-trained (TTT) memories, delta-rule linear attention, state-space memories,
learned associative stores — justify themselves by what their *dynamics* compute at inference. We ask a question
the family does not ask of itself: **at a matched state-byte budget, does a memory's learned test-time dynamics
beat a non-parametric table holding the same bytes?** The paper's object is the protocol that makes it
answerable across memory types: a matched-byte **table launder**, a **two-sided byte ledger** under a
**learned-initial-state rule** (an initialisation is *parameters*; only the per-sequence deviation is *state*),
a **+0 B substitute audit** over the memory's own stored bytes, a **same-keys null**, a **blank-store control**,
and a **rescue gate** disqualifying any arm within 2 SE of its own blank store. We run it on three rival
bounded-state families (TTT, delta-rule, SSD) and on a learned continuous-latent store — the CLU, introduced as
CHLU in Jawahar & Pierini (2026) — on a synthetic memory task at CPU scale (`d_in = 5`, 5–6 stored items,
~10-token streams; **nine seeds on every arm, ours included**). **(i)** On the one family surviving our own
protocol validation, **no arm beats a zero-extra-byte reader of a *raw* table holding the same bytes**: **0 of 6
rival arms** (**−0.2563 … −0.4602**, every one at least **4.4 SE** below zero, full tuning grid) **and not the
CLU either** (**−0.2897 ± 0.0328**, 8.8 SE). The gate that disqualifies one rival arm outright disqualifies
**ours**: our store's written read is statistically indistinguishable from an empty store (lift
**−0.0465 ± 0.0406**, 1.1 SE, point estimate on the wrong side of zero) — a finding, not a footnote. It
reproduces off this rig: on a **real-image** Split-CIFAR-10 stream census (`d = 12`, three encoder arms × 3
seeds, pre-registered, nothing tuned), **0 of 9 cells** show daylight for a store holding **2,364×** the table's
bytes (⚠ matched *items*, not bytes; a component-build measurement, §4.6). **(ii) Two protocol findings**, both
costing us work: the control the field would naturally write is the wrong one — reading a weight-valued memory's
byte-matched table *through the memory's own projections* costs that table **0.263 … 0.942 neg-MAE** at
identical bytes (six arms, all > 2 SE) and manufactures apparent dividends of that size, which vanish under a
raw control (both registered before measurement, the only reason this is a finding and not a re-frame); and the
**rescue gate is underpowered below nine seeds**, while fit-split best-of-grid selection makes a regulariser
axis unselectable, so a nominally 6 × 2 grid is operationally 6 × 1. **(iii) A byte-floor identity**:
`ratio = [A(D+2)+d]/(d+m) ≥ 2.20×` — exactly the byte price of one privately-deletable parameter group per item,
so **byte-exact deletion and compression are the same trade** — and the one coupling a row-selecting table
provably cannot express obeys `exp(−½(d/s)²)` on our learned store (R² = 0.995), **exponentially suppressed by
the very admission gate that keeps the store writable**. *A store organised well enough to be safe is organised
well enough to be a table.*

---

# 1. Introduction

A memory module earns its place by what it does at inference: TTT memories (Sun et al., 2024) run an inner
optimiser over the stream; delta-rule linear attentions (DeltaNet, Yang et al., 2024; Gated DeltaNet, Yang et
al., 2025; Gated DeltaNet-2, Hatamizadeh et al., 2026) run a key-conditioned erase–write recurrence;
matrix-state SSMs (Mamba-2 / SSD; Dao & Gu, ICML 2024) run a scalar-gated linear recurrence; Titans (Behrouz et
al., NeurIPS 2025) runs a momentum-accelerated associative write; Sparse Delta Memory (Cabannes et al., 2026)
routes writes into explicit slots; the CLU integrates a damped particle through a learned potential and reads
where it settles. All are *dynamics at test time over a bounded state*, and all are evaluated against **other
neural architectures**. The comparison none of them runs is the cheapest available: **put a non-parametric store
on the same byte budget and see who wins.**

That comparison is standard one field over. A learned Bloom filter counts as an improvement only if *"the size
to represent the learned function f and the size of the smaller backup filter for false negatives is smaller
than the size of a corresponding Bloom filter with the same false positive rate"* (Mitzenmacher, NeurIPS 2018) —
the *verdict* is conditioned on matched space — and learned-index benchmarks report the structure's own space
beside its latency (Kipf et al., 2019; Marcus et al., PVLDB 2020); evaluation methodology knows the same
discipline as *partial-input baselines* (Poliak et al., 2018; Feng, Wallace & Boyd-Graber, 2019). ⚠ The two
precedents differ and we keep them apart: only the learned-Bloom-filter line makes equal space a *condition of
the verdict*; the index benchmarks *report* space without equalising it. This protocol takes the stronger of the
two. We import an established discipline into a family that has not adopted it (§5), and the headline is
negative for every arm in the audit — **including ours**.

**Contributions.** **(1) A uniform matched-byte audit protocol for bounded-state memories** (§2): five mandatory
columns — table launder · two-sided byte ledger · **+0 B** substitute audit · same-keys null · blank-store
control — plus a **rescue gate** and an identical-encoder invariant enforced in code. **(2) The
learned-initial-state rule** (§2.2): the initialisation is **parameters**, only the per-sequence deviation is
**state**, both declared; applied to ourselves it costs us a *measured* deviation of 5200 B against 5376 B of
initialisation (ratio 0.967). **(3) Three measured findings about evaluation protocol**, each a control everyone
would run, shown not to be neutral: the **projected-table control manufactures dividends** (§2.3); the **rescue
gate is underpowered below nine seeds** and **fit-split best-of-grid selection makes a regulariser axis
unselectable** (§2.1, §2.5); **plain prequential evaluation at an `h`-step-ahead horizon is not a neutral
control**, handing a continuously-updated learner up to `h − 1` steps of future labels, asymmetrically in the
flattering direction (§2.4). **(4) Two structural results** (§3): the byte-floor identity `ratio ≥ 2.20×` and
its corollary that **compression and byte-exact deletion are the same trade**. **(5) The measurements** (§4):
six rival rows and the CLU column at nine seeds; the one coupling a table structurally cannot express; a
real-image substrate row at a 2,364× byte handicap. **(6) The negative, stated as the finding it is** — our own
arm disqualified by our own gate — with limitations (§6) stating the audit's thinness before a referee has to.

**Headline figure — Figure 1.** *Signed +0 B margin against a raw-metric byte-matched table, one bar per audited
arm, zero line drawn, arms failing their own blank-store control hatched*; all seven bars at a uniform `n = 9`.
**Every bar is below zero. This is the paper.** (Spec and figure → artifact → field provenance: Appendix K.)

**Scope, in our own voice, early.**

> **The matched-bytes launder tests whether a memory's *inference-time* dynamics beat a table **given the
> organisation** — both arms inherit the same placement of the same content. It does not test how that content
> came to be organised, and it is not evidence about any other stage of the system.**

⭐ One consequence cuts our way and belongs in the same place, so a null dividend is not over-read: **a store
that organises well and then reads like a cheap table scores dividend ≈ 0 by design — an inference-cost win, not
a failure.** Three hard boundaries. **(a) These laws govern the store; end-to-end performance additionally
depends on the encoder, measured separately, φ-bytes ledgered** — every arm of every cell (memory, launder,
`+0 B` readers, null, blank) sees the identical encoder output, asserted by content hash, with φ's bytes
ledgered on every arm including the launder; a null here is not an end-to-end refutation, a law here not an
end-to-end guarantee. **(b) The families are designed synthetics at CPU scale**, and our own validation (§2.4)
strikes three of four as measuring the construction rather than the memory. **(c) Nothing here transfers to a
language-model claim**: every cell runs at `d_in = 5`, 5–6 stored items, ~10-token streams, on CPU; no
language-modelling run was sized or attempted, and ⛔ **no external benchmark is claimed as won on its own
headline metric — here or anywhere in this paper.** **Verification versus evidence:** §3 and Appendices C–E are
*verification* (designed testbeds, exact arithmetic over recorded ledgers, confirming an identity is exact where
claimed — not discoveries); §4 is *evidence* (trained store, trained rival arms). Where the two meet (§4.5) we
say which is which and report the discrepancy rather than the agreement.

---

# 2. The protocol

A **bounded-state memory** maintains a state of fixed byte size across a stream, writes items into it, and
answers queries from it. We audit at a fixed **iso-state budget** — every arm sized so its declared state
occupies the same bytes (**1364 float32 = 5456 B**; head widths derived from it and registered before any run,
Appendix A.1). Task families are in Appendix N, rival arms in Appendix O, our store in Appendix P. Every arm
shares one read-in encoder `φ` whose bytes are ledgered on **every** arm including the launder; each cell
asserts a content hash of `(q₀, keys)` across arms and raises on mismatch (a 1e-9 perturbation raises; tested).

## 2.1 The five mandatory columns, and the rescue gate

| column | definition | what it rules out |
|---|---|---|
| **matched-byte table launder** | the memory's own stored content re-served as a table of `n_rows = ⌊state_floats/(d_k+d_v)⌋` rows at the same byte budget, read by a fixed reader | "the dynamics are doing the work" when the content alone suffices |
| **two-sided byte ledger** | a declared split into **F1 parameters** (shared across sequences), **F2 state** (per-sequence — the audited budget), **F4 per-read transients**; must sum to the total or the run raises | budget laundering in either direction |
| **+0 B substitute audit** | the strongest reader using **zero extra bytes** beyond the launder's own table (arg-min, 2-NN mean, 2-NN inverse-distance, echo, insertion-order, order-aware) | "our memory beat its launder" when neither is the best reader of its own bytes |
| **same-keys null** | the same read against a store written with the same keys and a permuted payload assignment | key-side leakage through the encoder |
| **blank-store control** | the identical read path against a store with nothing written in it | reads that succeed on an empty store (a measured failure mode) |

**The rescue gate.** An arm whose full read sits **within 2 SE of its own blank-store control** is **NOT
RESCUED**, and **no comparative margin in favour of another arm over it is quotable.** Direction rule (Appendix
B.5): the gate suppresses *comparative* margins in the **flattering** direction, never an arm's own loss to its
own byte-matched table — the only quantity the headline uses. It disqualifies one of six rival arms outright,
leaves one unresolved across initialisation schemes and two selection-dependent, **and disqualifies our own
store** (§4.1); on the byte-frontier column **none of the twenty (arm × head-width) cells at nine seeds clears
it** (Appendix H).

⚠ **Protocol finding 1 — the gate has a power requirement, found the hard way.** Its control is the arm's
blank-store read (for a memory with a learned initial state: that initialisation read through fitted
projections), whose seed-to-seed spread is comparable to the lift it gates — one arm's blank reads over three
seeds: **−0.962 / −2.634 / −1.390**. **At three seeds the gate is a coin flip**: three legitimate configurations
of our own harness return three different rescued sets, while at nine seeds two independent code paths agree on
four of the five arms they share. ⇒ **every rescue verdict here is a nine-seed verdict; we report no three-seed
rescue verdict, including our own first pass's.** Adopters should run ≥ 9 seeds, pair the control per seed, or
average it over initialisations. ⚠ Seeds are not the only axis: so is the best-of-grid **selection rule**. We
score three registered rules from the same fits (fit-split primary, the first pass's sub-grid, a **held-out**
stream) and quote a verdict **only where stable across all three**: `gdn`, `mamba2` stably rescued; `ttt_mlp`
stably not; `ttt_linear` INIT-UNSTABLE; **`deltanet` and `gdn2` SELECTION-DEPENDENT** (held-out lifts
**+0.0768 ± 0.0446**, **+0.6685 ± 0.3389**). ⛔ No comparative margin in the flattering direction is quoted
against a selection-dependent arm, exactly as for a non-rescued one. ⚠ Every 2 SE decision is reported **per
cell**; **no multiplicity correction is applied and none is claimed** — the headline quantity is not a gate
decision, and fragile decisions are printed as fragile rather than resolved. Admissible-cell coverage is
reported first-class, per family per seed.

## 2.2 The learned-initial-state rule

TTT's `W₀` is explicitly *"shared between all sequences"*; explicit-slot memories learn an `M₀`; our own `V_θ`
has a learned initialisation. No published convention says how to count it, and the choice moves the number a
lot in both directions.

> **Rule.** The initialisation is **parameters** (F1); only the **per-sequence deviation** is **state** (F2);
> both are declared, in the same table, for every arm.

Counting the init as state *inflates* a budget (making the matched table larger and easier to beat); counting
the deviation as parameters *launders* it. ⭐ **We apply it to ourselves and measure rather than assume**,
diffing `V_θ` before and after the stream: the write moves **192/192 atom centres (960 floats) but only 160/192
widths and amplitudes**, plus 20 codebook floats — **1300 floats = 5200 B** against an initialisation of **1344
floats = 5376 B**, a state/parameter ratio of **0.967** (Appendix G).

## 2.3 The +0 B substitute audit, and the control it must include

*Is the memory the best reader of its own bytes?* A zero-extra-byte reader gets the launder's table; if it
matches or beats the memory, the memory's advantage over its own launder is not evidence about the memory. The
idea is not ours in general form (§5) — it is the partial-input / trivial-baseline tradition, applied to a
*memory's own stored bytes* at a *state-byte* convention. We carry Feng et al.'s converse caveat because it
bounds what a *passed* audit licenses: the failure of a partial-input baseline does not show a dataset is
artifact-free, and **by the same logic a substitute audit a memory passes does not show the memory is doing real
work** (⚠ their caveat concerns datasets and annotation artifacts; the transfer to stored bytes is our analogy).
Only a *failed* audit is informative — and here it fails for every arm.

⭐ **Protocol finding 2 — the control the audit must include.** For a weight-valued memory the natural
byte-matched table is the table of the memory's own projected `(θ_K x, θ_V x)` pairs; that is what we
pre-registered and implemented, and **it is not a neutral control.** Read through the memory's own projections,
a table at the same bytes is handicapped by **0.276 / 0.263 / 0.425 / 0.856 / 0.942 / 0.614 neg-MAE**
(TTT-Linear · TTT-MLP · DeltaNet · GDN · GDN-2 · Mamba-2; nine seeds, all > 2 SE from zero — the SSD arm's gap
is **+0.6139 ± 0.1386**) relative to a **raw-metric** table holding the same bytes, for two reasons visible at
equation level: `θ_K, θ_V` are trained for the *recurrence*, not for a table; and a single stored value decoded
by the memory's own head `θ_O` is out of distribution for `θ_O`, trained on kernel-averaged reads
`o = Σ_s v_s(k_s·q)`. On two arms the handicap (**0.856 ± 0.091**, **0.942 ± 0.091**) exceeds the margin by
which the arm loses to the raw table (**0.2600 ± 0.0278**, **0.2592 ± 0.0292**), so **which control is run
decides the sign of the reported result.** Both are mandatory, both reported for every arm. ⚠ **The registration
ordering is load-bearing and belongs in the methods, not a rebuttal:** the projected control (predicted
positive) and the raw control (predicted to erase it) were **both registered before any measurement**; added
afterwards the second would be indistinguishable from a re-frame. ⭐ The dated registration documents, one per
pass, are in the supplementary material (Appendix I).

## 2.4 Validating the instrument before spending it — designed families, then real venues

We ran the protocol against a **full-attention table reader** on every candidate family and asked a
falsification question about the protocol itself: *if a ≤4 B substitute is at the metric's ceiling for every
family, including for full attention, the protocol is measuring the task and not the memory.* With
`S(f) = (sub − blank)/(M − blank)` (3 seeds, designed families):

| family | metric | `M` | blank | +0 B substitute | full attention | **`S(f)`** | verdict |
|---|---|---|---|---|---|---|---|
| `overload` (byte ratio 478.2×) | decode | 1.0 | 0.1667 | **1.0000** (settle-deleted) | 1.0000 | **1.0000** | ⛔ saturated |
| `aggregate` (54.56×) | neg-MAE | 0.0 | −0.4221 | **−0.2081** (2-NN) | −0.2493 | **0.5068** | ✅ **survives** |
| `recency` (54.56×) | acc | 1.0 | 0.5463 | **1.0000** (order-aware) | 0.4755 | **1.0000** | ⛔ saturated |
| `manifold` (52.0×) | R² | 1.0 | −0.0001 | **1.0000** (echo) | 0.0000 | **1.0000** | ⛔ saturated |

**It did not fire — but it came within one family of firing, and reshaped the paper.** Three of four families
are struck as protocol-invalid: something costing ≤4 B sits at the metric's exact maximum and **it is never the
memory** (on `overload` our own store reads 0.9722 ± 0.0139, below three readers of a table costing 1/478th of
its bytes). ⚠ **The rule cannot separate a substitutable family from a substitutable *anchor*** — one anchor per
family, no anchor sweep; unresolved, carried to §6 L1. ⚠ **Coverage, including the uncomfortable part:** by our
store's own write-admissibility criterion (endpoint loss ≤ 0.05) **`aggregate` is 0 of 3 and `manifold` is 0 of
3**, unmoved (≤ 0.005) by tripling the write budget — the family carrying our entire dividend column is one our
store does not write to its own bar, a candidate mechanism for §4.1's result and a fact about *our* store, not
the protocol or any rival arm. ⚠ `overload`'s verdict turns on one pre-registered definitional choice (excluding
the arg-min launder gives `S_excl = 0.6500` and the family survives); we report both, selected the strict one,
and carry it **only** as a labelled byte-frontier column (Appendix H). ⭐ The one generalisable design rule two
rounds of auditing support: **`aggregate` survives because its target is constructed to be absent from the
table** — *"the answer is provably not in the table"* is the only family property that has survived a +0 B
substitute audit (0-of-4, then 1-of-4).

**The same discipline, run on real streaming venues, retires both** (riders and provenance: Appendices R.2,
A.6). Before committing a store build to a public stream we ran the same tripwire — *is a trivial, byte-matched
exemplar store already at the strong-baseline frontier?* — on the two best-documented real streaming venues
available to us: INSECTS (prequential accuracy, window 1000, 79,985 scored instances) and Metro Interstate
Traffic Volume (prequential 24-h-ahead MAE, 34,848 scored pairs):

> **At laptop byte budgets an exemplar store at matched bytes sits at or above the strong-baseline frontier on
> both of the best-documented real streaming venues — and on one of them, destroying temporal order does not
> hurt it.**

⚠ **Decoding *frontier*:** **within the pre-registered admissibility margin**, not *ahead of the best baseline*
— and the venues fire differently. On INSECTS the byte-matched exemplar store (SAM-kNN at its published
0.634 MiB budget: **76.9157 %** at 665,000 B, causally standardised) lands **1.90 points below** ARF-100, 0.10
points inside the registered threshold, so the honest sentence is ***"the byte-matched exemplar store is at
ARF's shoulder"*, never *"it beats ARF"***. ⚠ ARF's byte caveat is mandatory wherever ARF anchors the frontier:
its measured state is **9,542,925 B = 14.35×** SAM-kNN's 665,000 B — **ARF is byte-matched to nothing**, the
accuracy frontier only. ⚠ Ours is `river` 0.25.0's `ARFClassifier` (78.8139 ± 0.0526 SD, 3 seeds); the published
**77.13 is a MOA number** at an unstated ensemble size (**100 is MOA's default, so reading their arm as
"ARF-100" is our inference**): `river` ARF ≠ MOA ARF, and ours is the stronger — the direction that makes firing
*harder*. ⛔ The venue's terminal persistence band is at ceiling and persistence-trivial, and **no per-band
retention or acquisition number is quoted from it**, here or anywhere. On Metro there is no threshold call: the
exemplar store beats **every** strong reference outright, firing against all **nine** (`gbdt_tuned` −0.0615 ·
`gbdt` −0.0710 · `gbdt_cat` −0.0721 · `gbdt_recent` −0.0916 · `rls` −0.1604 · `gru_big` −0.1605 ·
`ridge_batch` −0.2119 · `mlp` −0.2419 · `gru` −0.2969) by 4–25 %; the **registered `k = 5`** arm at the
store-build byte budget reads **306.76** MAE, and destroying the stream's temporal order *improves* it there
(306.762 → 301.068, −1.86 %). ⚠ The shuffle is a shuffle, **not a drift-free data source**. ⚠ We read this as **a
fallback being retired, not a venue crisis**: *"move the audit to real streaming data"* does not escape the
substitute-audit discipline any more than the designed families did. ⛔ These are **baseline-only** admissibility
runs — **no memory cell of any kind, ours or a rival's, ran on either venue** — so every real-data value leg of
this work is a **declared NOT-RUN for want of an admissible venue, never a null** (Appendix J), and no benchmark
result is claimed for any memory.

⭐ **Protocol finding 3, from the same runs** (Appendix R.3): **plain prequential evaluation at a 24 h horizon
leaks up to 23 h of future traffic to any continuously-updated learner (+10.9 % to a 250-exemplar k-NN, −0.3 %
to GBDT — asymmetric in the direction of firing); and seasonal-naive(`t − 24 h`) is degenerate at a 24 h horizon
(the non-degenerate naive is `t − 168 h`).** Mechanism: pair `t − 1`'s *label* is the volume 23 hours after pair
`t`'s forecast origin, so test-then-train hands the learner those labels — our own pre-registration asserted the
opposite and was refuted by mechanism. The fix (restrict every admission, fit and update to indices ≤ `A(t)`,
the last pair whose target time ≤ pair `t`'s origin) is used for every primary Metro number above.
⚠ **Delta convention, so it need not be reverse-engineered:** each percentage is `(embargoed − leaky)/embargoed`,
relative to the embargoed (conservative) value, so a **positive** number means plain prequential evaluation
flattered that arm by that much and the embargo took it back. ⭐ The venue verdict is unchanged by the fix
(−0.0946 leaky, **−0.0615** embargoed): the leak we closed was biased **toward** firing our own tripwire.
⚠ Scope: one stream (Metro, 34,848 hourly pairs, `h` = 24), one feature construction — **the mechanism is
protocol-general, the magnitudes are not.**

## 2.5 Tuning the rivals

*"Rivals lose to their own byte-matched tables"* is what under-tuning a rival produces, and that bias runs
*toward* our headline — so we close the attack by measurement (full detail: Appendices A.1, I.1). Every rival
number comes from the **full grid** (6 learning rates × 2 weight decays; TTT arms additionally two mini-batch
sizes), best-of-grid on the fit split with outer parameters never seeing the eval stream, 400 outer steps plus a
**2000-step (5×) re-check**, nine seeds per arm; ⚠ two sub-clauses of the standard (`β = (0.9, 0.98)`, cosine
decay) were **not** adopted and are declared deviations. **Nothing in the headline changed**: the widened grid's
points are selected in **0 of 45** incumbent cells (**1 of 54** with the SSD arm, whose fit-split optimum is
*interior* to the grid — the sharpest available refutation of "under-tuned" for the arm that has it); widening
moves an arm's `full` read by ≤ **0.031**; at 5× budget TTT-MLP's fit loss falls **64.1 %** while its eval metric
moves under one SE; restoring the SSD arm's block-level parts cuts fit loss **36 %** while moving its eval read
**worse**. ⚠ **A disclosure that matters more than the tuning did:** widening the grid required a new
initialisation-key scheme, and the control column pricing it shows that re-draw moved arms **4× to 35× more than
the tuning did**. ⭐ **Protocol finding 2b — the tuning standard's own selection rule:** fit-split best-of-grid
selects on the very objective being optimised, so `wd = 0.1` is chosen only by fourth-decimal tie-breaks
(**12 of 45**) and a lower learning rate **never** (0 of 45), while under a **held-out** stream the added points
*are* chosen (**26 of 45**, **24 of 45**). ⇒ **a nominally 6 × 2 grid is operationally 6 × 1 unless the
selection split is fixed too.** Held-out selection is a declared secondary and keeps every raw-table margin
negative (**−0.24 … −0.49**, 5 of 5).

---

# 3. What matched bytes can mean

*(Grade: **verification** — identities checked in exact arithmetic or on designed geometries; neither is a
performance claim. Assumptions and verification detail: Appendices C–D.)*

Let the store be an atom dictionary `V_θ(q) = α‖q‖² − Σ_j A_j exp(−‖q − c_j‖²/2s_j²)` with learnable
`(c_j ∈ R^D, log s_j, amp_j)`, partitioned into **one atom group per item slot** (a masked, item-local write),
and the launder a table of `K` live rows `(key ∈ R^d, payload ∈ R^m)`; write `A ≡ N_at/K`,
`D = d + m + n_spectator`.

> **Theorem T1.** `ratio ≡ full_bytes/launder_bytes = [A(D+2) + d]/(d+m)` exactly and independently of `K`; and
> since one atom group per item forces `A ≥ 1`, `ratio ≥ [(D+2)+d]/(d+m)` = **2.20×** at `(d, m, n_spec) =
> (4, 1, 0)`, **2.40×** at `(4, 1, 1)`.
>
> **Prop T1.4.** Byte-exact deletion in `O(1)` is available **exactly on an item's private-atom fraction `p`**,
> because byte exactness *is* the statement that item `i`'s parameters form a block disjoint from every other
> item's — the same property that forces `A ≥ 1`. On the shared fraction there are two options and no third:
> leave the shared atoms (deletion is not byte-exact, with a residual) or re-fit them (every co-tenant's bytes
> change, and the cost is a *write*, not a delete — the retraining baseline exact unlearning is defined
> against). **Corollary.** A private atom is indivisible, so `p ≥ 1/A_tot` forces `r ≥ [(D+2)+d]/(d+m)`:
> ⭐ **the 2.20× floor is exactly the byte price of one privately-deletable atom per item.**

**Verification** (detail: Appendix C): in exact integer/rational arithmetic over 28 recorded ledger cells the
byte decomposition is exact **28/28** and the identity reproduces the measured ledger ratio **28/28 at 0 ulp**.
⚠ **Erratum, ours:** the closed form in our own pre-registration is **wrong in 4 of 28 cells** — every cell with
a spectator dimension — and the error was **conservative** (it understated ratio and floor alike), so no claim
built on it was inflated; the corrected law above is exact in all 28. **Domain:** exactly this store family.
**The trade, quantified:** at matched bytes **at most `p ≤ 4.19e-4`** — 0.042 % of an item's parameter mass —
could remain byte-exactly deletable, at exchange rate `dp/dr = 2.10e-3`; this bounds **byte** exactness only,
not behavioural unlearning metrics. A third verification result — a settled-point read is **untrainable
end-to-end in both directions** — is in Appendix D.

---

# 4. Results

**Grade: evidence.** Sample sd (`ddof = 1`), `SE = sd/√n`; identical `φ` asserted in code; byte-ledger identity
asserted as integers. **Seed counts, stated once and never mixed silently:** on the dividend family **every
column is at a uniform `n = 9`** (seeds 0–8) — six rival arms under the full grid **and our own store**, whose
nine-seed column is a **re-aggregation of banked per-seed cells** (nothing was re-measured; Appendix A.1e). Flag
provenance, one table per run: Appendix A.

## 4.1 The audit table, and the arm it disqualifies

| arm | **full** | **+0 B margin** | ⭐ **raw-metric +0 B margin** | lift over own blank (full grid) | lift (first-pass path) | **RESCUED?** |
|---|---|---|---|---|---|---|
| ttt_linear | −0.6075 ± 0.1096 | **−0.2213 ± 0.1062** | **−0.4602 ± 0.1038** | +0.093 ± 0.134 | +0.320 ± 0.083 | ⚠ **INIT-UNSTABLE** |
| ttt_mlp | −0.5898 ± 0.0731 | **−0.2095 ± 0.0683** | **−0.4425 ± 0.0869** | −0.071 ± 0.090 | +0.093 ± 0.107 | ⛔ **NOT RESCUED** (no configuration, no selection rule) |
| deltanet | −0.4205 ± 0.0299 | **−0.0172 ± 0.0263** | **−0.2732 ± 0.0395** | +0.294 ± 0.077 | +0.141 ± 0.046 | ⚠ **SELECTION-DEPENDENT** (+0.077 ± 0.045 held-out) |
| gdn | −0.4073 ± 0.0120 | **−0.0102 ± 0.0229** | **−0.2600 ± 0.0278** | +0.880 ± 0.227 | +0.947 ± 0.149 | ✅ **RESCUED** (all three rules) |
| gdn2 | −0.4065 ± 0.0178 | **+0.0473 ± 0.0277** | **−0.2592 ± 0.0292** | +1.025 ± 0.329 | +1.384 ± 0.276 | ⚠ **SELECTION-DEPENDENT** (+0.669 ± 0.339 held-out) |
| mamba2 (SSD) | −0.4036 ± 0.0329 | **+0.0047 ± 0.0519** | **−0.2563 ± 0.0416** | +1.421 ± 0.463 | — (added after the first pass) | ✅ **RESCUED** (all three rules; lift positive 9/9 seeds) |
| **CLU (ours)** | **−0.4370 ± 0.0417** | **−0.2897 ± 0.0328** | **−0.2897 ± 0.0328** (float-identical, 9/9 seeds) | **−0.0465 ± 0.0406** (blank **−0.3906 ± 0.0124**) | — | ⛔ **NOT RESCUED** — within noise of its own blank store, point estimate on the wrong side of zero |

*Head widths, per-arm byte ledgers and the same-keys-null column: §4.3, Appendices B.2 and I.1c.*

⭐ **Our own arm fails our own gate at the same nine seeds as every rival arm, and that is a result, not an
embarrassment.** |t| = 1.14 against a 2 SE bar: the written store is **statistically indistinguishable from an
empty one** — the same category as `ttt_mlp` — with the point estimate on the wrong side of zero. The honest
reading is the one our thesis predicts: the written content does not lift the read above a blank store, and a
store whose content does not lift its own read is exactly a store whose dynamics have nothing to buy over a
table of the same content. ⚠ §2.4 records a candidate mechanism and it is ours to own (0 of 3 cells
write-admissible at tolerance 0.05). ⛔ Consequently **no comparative margin in favour of the CLU over any rival
arm is quotable anywhere in this paper**, and none is drawn. ⚠ It is a sign-and-significance statement, **not** a
demonstration that the store reads *below* blank (the lift is inside noise), and not evidence about any other
store; our **blank** read is far more stable across seeds (SE **0.0124**) than our **written** read
(SE **0.0417**), so on our side the variance the gate fights comes from the *write*; and our column is
bit-identical across both code paths, so cross-path agreement is *not* independent evidence for our own verdict.
⭐ The projected-versus-raw distinction **does not arise for our store, and that is measured**: our launder is
already a raw `(key, payload)` table and the raw-table margin is **float-identical to the `+0 B` margin on 9 of
9 seeds**; our dividend over our own launder is **−0.0561 ± 0.0315** — at **1.78 SE** a *sign* statement, not a
significant effect: the launder **reads no worse than** the store, and no sentence here says the launder *beats*
it. ⛔ **Byte-ledger discipline:** the TTT rows' ledger **and our own** are per-seed quantities (best-of-grid
selects the mini-batch `b`, which sits inside the declared state; our admission gate admits five items on eight
seeds and six on the ninth), so **no single TTT byte figure and no single CLU byte figure is *the* nine-seed
value**; ours is **modal, 8 of 9 seeds** — `5456 B / 100 B / 54.56×`, with `5472 B / 120 B / 45.60×` on seed 8,
the integer identity green on all nine (a labelling rule, not a defect). Coverage is reported first-class (mean
admitted fraction **0.639 ± 0.014**, mean query coverage **0.695 ± 0.041**).

## 4.2 The headline

> ⭐ **At byte-matched state, on the one designed family that survives protocol validation, at `d_in = 5` with
> 5–6 stored items and ~10-token streams at CPU scale, no memory in this audit — neither the three rival
> bounded-state families nor the CLU — beats a zero-extra-byte reader of a *raw* table holding the same bytes:
> 0 of 6 rival arms over nine seeds under the full tuning grid (**−0.2563 ± 0.0416 … −0.4602 ± 0.1038**, every
> one at least 4.4 SE below zero), and the CLU over the same nine seeds (**−0.2897 ± 0.0328**, 8.8 SE below
> zero).**

The scale qualifiers are the claim's extent, not decoration. **The headline survives every registered stress** —
the full grid, **5×** the outer budget, a **held-out** selection rule and nine seeds — with the margin negative
in every column we ran (Appendix I.1d; ⚠ the two stress columns are three-seed re-selections from the same fits,
less powered than the result they defend, and labelled as such at every appearance). ⚠ On optional stopping:
seeds 3–8 were a declared power addition made *before* the pooled aggregate was computed and not conditioned on
it, and the verdict change that matters ran **against** the flattering direction (DeltaNet became rescued,
*adding* a functioning rival to compare against). ⚠ **The honest qualifier on the arm count applies to our own
arm too:** four of six rival arms do not settle the gate under every registered selection rule, so for them *"it
loses to a raw table at its own bytes"* is partly a statement about an arm that may not be reading its store at
all — ⛔ **and the identical qualifier attaches to the CLU.** We apply it to ourselves in the same paragraph as
to the rivals, because a gate that disqualifies only other people's arms is not a gate. **On the two arms whose
verdict is stable across all three selection rules the headline holds with more room: −0.2600 ± 0.0278 (GDN) and
−0.2563 ± 0.0416 (Mamba-2)** — ≈ 9.4 and ≈ 6.2 SE below zero, the restricted form we would defend if only one
could be.

⚠ **The objection our own construction note invites, answered rather than left standing.** Appendix N.1 says of
the dividend family that *"the family ships with the 2-NN mean as its own strongest control, and that control is
expected to win. It does."* That is a **design property** — the answer is constructed to be absent from the
table, so an aggregating reader is the obvious beneficiary — and it is not the finding. The finding is what the
construction did **not** fix, all of it free to move: the **size and stability** of the margins (nothing sets how
far a *fitted memory* falls short — measured −0.2563 … −0.4602, every arm ≥ 4.4 SE below zero, the sign holding
on all six arms under every stress); the **sign** of the same arms against the weaker projected control (four of
six comfortably positive — the family is perfectly capable of paying test-time dynamics a dividend, and does,
against the control the field would naturally build); the **rescue-gate outcomes, including our own** (a design
that foreordained the headline would not also have disqualified the authors' arm at the same nine seeds); the
**rivals' behaviour against their own projected tables** (four of six cleared that weaker bar, which makes the
raw-table result a statement about the *control* rather than about broken arms); and the **direction the
registered stresses moved things**. Appendix N.1 itemises this in one place.

## 4.3 Where test-time dynamics *does* pay; and the two-sided ledger

**"The finding inverts" — does NOT fire in the strong form; DOES fire in the weak form, which we state plainly
rather than re-frame.** Against the **arg-min** control read through each memory's own projections the dividend
at nine seeds is positive by more than 2 SE on four arms (**DeltaNet +0.1515 ± 0.0600 · GDN +0.5960 ± 0.0933 ·
GDN-2 +0.6824 ± 0.0756 · Mamba-2 +0.3575 ± 0.1451**), negative on both TTT arms, and **−0.0561 ± 0.0315** for
the CLU (1.78 SE, a sign statement): **test-time dynamics pays for the delta-rule and SSD arms and does not pay
for ours** — true as measured, unsoftened. **Strong form: 0 of 6 rival arms beat the raw-metric +0 B table at
the same bytes, and neither does the CLU** — and the audit is not a different paper **only because the
distinction that decides it was registered before measurement**; we regard the ordering, not the outcome, as the
credible part. **"Not apples-to-apples" — does NOT fire**, with a split we never blur:
`n_rows = ⌊state_floats/(d_k+d_v)⌋` is *forced* for the six arms adjudicated **by measurement** across three
state types, while ⚠ **Sparse Delta Memory and Titans are adjudicated from published equations only, never
measured**. ⭐ **Every rival family we measured is metric-native or weakly so** (per-arm equations, verdicts and
losses: Appendix O.3; ⚠ TTT-MLP is the one arm where the equation-level argument does not close, and ⛔ it is NOT
RESCUED, so no comparative margin over it is quoted) — the matched-bytes ceiling is a property of the family,
and that line belongs to the field rather than to us.

**The two-sided ledger, and the asymmetry that is itself a finding** (full table: Appendix B.2). Every rival's
declared state sits at a state/table ratio inside **[1.0000, 1.0278]** on every seed — ⭐ **every rival's state
*can* be byte-matched to its own table; ours provably cannot.** T1's floor makes matched bytes unreachable under
a per-item group-masked write and the audited cell sits at **54.56×** (⚠ **modal, 8 of 9 seeds**; **45.60×** on
the sixth-item seed). This is the sharpest single statement the ledger produces, it runs *against* our own
system, and it is why every byte or dividend claim here carries the **≥ 2.20× (≥ 2.40× with a spectator
dimension)** caveat. ⚠ **Params are not matched** and no arm is param-matched (the SSD arm's F1 is *lower* than
the delta arms' — an asymmetry in the rival's favour). ⛔ **No cell measured in this audit is a byte-matched
dividend; the minimum ratio measured anywhere in this work is 17.11×.**

## 4.4 Deletion, in the "and also" position

⚠ **Said first, by us: a table deletes exactly by construction** — byte-exact deletion is a result only for a
*learned or superposed* store, which is the entire reason this column is reported and the entire reason it is
not a headline. **Frozen result, with instrument and conditions in the same sentence:** deleting an item leaves
the store **byte-equal to the never-written counterfactual on 3072 of 3072 compared bytes** and a
membership-inference attack against the deleted item reads **AUROC 0.5000 ± 0.0000**, at every tested load from
0.29× to 1.71× of capacity, under three explicit conditions (`budget ≥ n_cells`, zero leak, depth-ordered
eviction; an LRU policy is a hard error). ⛔ We do not call this *certified*, *unlearning* or *exact deletion*
without those qualifiers: it is **verified byte-exactness under the stated conditions** — locked into §3's trade
and **materially narrowed** by neighbouring work reaching MIA-AUROC ≈ 0.5 by design in a different setting
(Appendix Q). ⛔ **No rival family in this audit has a deletion verb at all**, so this column has no cross-family
row and we do not manufacture one.

## 4.5 What a table structurally cannot do, measured

*(The law is **verified** on designed geometries, then **measured as evidence** on the learned store; the two are
reported separately and their discrepancy is a finding. Full sweep: Appendix F.)*

> **Prop T5.4 (shared-index bottleneck).** For fixed `x`, a row-selecting table's `∂ŷ/∂(any non-selected row)
> = 0` **exactly**. A continuous-latent store has no such factorisation: its acceleration is `−M⁻¹∇V` and `∇V`
> sums over **every** well, so deleting a stored item the query did not select moves every point of its
> trajectory.

Measured on the learned store (full per-radius sweep, rulers and fits: Appendix F), the coupling falls from
**0.814 ± 0.31** at `d/s_fit = 1.10` to **0.01534 ± 0.006** at the audited cell (`d/s_fit = 3.59`), obeying
`exp(−½(d/s)²)` with implied `s` = **0.3979** at **R² = 0.9953** over **2.72 decades** (three estimates by two
independent methods, 0.7 % apart), while the per-slot table's third-party Δ is **`0.0` at every slot × every
dropped row × every cell — float equality, not a tolerance** (by construction, never a win). ⚠ **This corrects
one of our own earlier statements by a large factor:** an earlier estimate placed this store at `d/s ≈ 1.9` with
an `O(1)` coupling by reading the *admission gate's refusal radius* as an achieved spacing; the achieved
separation is `sep = 1.346`, so the audited configuration runs at **`d/s = 4.34` (atom-width ruler)** /
**`d/s_fit = 3.59` (fitted-width ruler)** with coupling **1.53e-2** — a **45–52×** correction, in the direction
that makes the table *harder* to escape. **Every `d/s` statement in this paper names its ruler.** ⚠ **Convention,
and the check we owe:** the effective-`s` estimator subtracts the store's own confining term `α‖q‖²`, and a
`d/s` computed without that subtraction is a different quantity; ⛔ on that convention **`s = 0.40` is flagged
for a check, not discharged** (a declared NOT-RUN, Appendix J) — ⭐ with the correction's direction known rather
than guessed and running *against* us (smaller `s`, larger `d/s`, *more* suppression), so every suppression
number printed here is the **conservative** one.

> ⭐ A per-slot matched-bytes table reproduces our slotted read at 37 of 38 slots, so on the answer channel the
> dynamics buy nothing a table cannot. Exactly one thing such a table structurally **cannot** do: deleting a
> stored item the query did *not* select changes its answer by **exactly zero**. A CLU has no such
> factorisation, and that coupling obeys `κ(d) = (d/σ_q)·exp(−(d²−σ_q²)/2s²)` with **R² = 0.995** across a 525×
> range. It is `O(1)` — 0.81 of the query's own item — only when neighbouring items sit **1.8 well-widths**
> (atom-width ruler) apart, and at that spacing our admission machinery refuses the write in **half** of all
> seeds because the wells have merged (`λ_min < 0`). At the spacing our shipped configuration achieves
> (**4.3 well-widths** atom-width; **3.59** fitted-width) it is **1.5e-2** — still the store's, but two orders of
> magnitude below the query's own launch noise, i.e. unusable as a read-out. ⭐ **The one capability a table
> cannot imitate is exponentially suppressed by the very gate that makes the store safe: a store organised well
> enough to be safe is organised well enough to be a table.** Not a defect of this implementation — a design
> identity.

⚠ The set of couplings a table cannot express is **not proven exhaustive**: we proved third-party attribution is
inexpressible and that richer per-item slot content is *not* a route (measured slot-vector rank **13 = 3d+1** at
`d = 4` for every `S ≥ 2`), not that no other coupling class exists.

## 4.6 The same store-versus-table indistinguishability at a real-image substrate

*(Grade: **evidence** — a component-build measurement on real-image streams, made on this program's census rig
and folded in here; provenance Appendix A.5, detail Appendix R.1. ⛔ **Not** one of this paper's matched-byte
cells; no number from it enters any other section.)*

The audit's central store-versus-table statement — a store whose reads are statistically indistinguishable from,
**or below**, a small table's (here §4.1's dividend column, **−0.0561 ± 0.0315** at 1.78 SE: the launder *reads
no worse than* the store) — has been measured at a further substrate, unrelated to both designed families and a
**third** in this paper's own counting: a **real-image** Split-CIFAR-10 stream-census rig at address dimension
`d = 12` (a 32,768-atom store of the same implementation class as Appendix P's, 16-item capacity, three encoder
arms × 3 seeds = 9 cells, 128 cue queries per cell). The registered finding, in its approved form:

> **Branch (b), 0/9 cells, 3/3 arms, pre-registered at Q6 = 0.70 with nothing tuned; |A3b| ≤ 0.047 on 7/9 ⇒ the
> store and an 832-byte table are statistically indistinguishable on held-out stream reads — and the null is
> ATTRIBUTABLE (store not inert 9/9, geometry GO and gate validation both verified in advance).**

*Branch (b)* is the pre-registered **no-daylight** branch of a two-branch registration in which both branches
were declared reportable before the first cell; *Q6 = 0.70* is that branch's prior, assigned in advance
(daylight carried 0.15); *A3b* is the store-minus-table margin on **held-out stream reads** (pooled binomial
SE); and the registered rule is *"DAYLIGHT iff a launder margin is POSITIVE beyond 2 SE on ≥ 3 seeds — cue (A3a)
or stream (A3b); NO DAYLIGHT otherwise."* Measured: **0/9 cells, 0/3 seeds on every arm, on both legs**
(per-arm margins: Appendix R.1). Three riders travel with the row. ⚠ **Matching is matched-items, not
matched-bytes, and the byte asymmetry runs in the store's favour:** the store occupies **1,966,848 B against the
832 B cue-side table (2,364×)** — 4,728× against the 416 B ring table, 3.147× with the encoder priced on both
sides — **and still buys no daylight**; it is a *weaker-instrument sibling* of the matched-byte columns, not one
of them. ⚠ **The ceiling itself was expected; the measured content is the attributable absence of daylight:** on
a metric-native cue protocol a nearest-neighbour reader of the store's own keys is the Bayes-rule reader, so a
table *ceiling* is a theorem there, stated in advance — which is why the null branch carried the 0.70 prior.
What the row adds is **attributability**: the store is demonstrably not inert (well-depth medians 0.47–0.97
across all 9 cells), the address geometry was verified GO in advance (σ_q/spacing 0.334 → 0.210 at `d = 12`, 3/3
seeds), and the instrument's validation was verified in advance — so the null cannot be dismissed as a dead
store, a bad substrate, or a blind instrument. ⚠ **A component-build measurement, not a verdict:** no
tier-level, performance, accuracy or benchmark claim, and it adjudicates nothing between its encoder arms. At
that substrate, under a rule registered before the first cell, with nothing tuned, the store and a table of its
own keys read the same — this paper's closing sentence, measured on a rig not designed to test it.

---

# 5. Related work and positioning

**The family being audited** (per-arm detail: Appendix O; verified citations and quote scopes: Appendix Q).
TTT makes the memory an inner learner with a sequence-shared `W₀` whose Theorem 2 identifies the nonparametric
TTT learner with a Nadaraya–Watson estimator; Titans adds momentum and a forget gate; delta-rule linear
attention runs DeltaNet → Gated DeltaNet → **Gated DeltaNet-2**, our reference arm because it supersedes GDN
(⭐ and concurrently **Erase-then-Delta** makes the same erase/write-decoupling move at larger scale, so the
frontier *moved* rather than being conveniently chosen); Sparse Delta Memory routes writes into explicit slots
with a learned `M₀`; **Mamba-2** carries a matrix-valued state under a scalar decay, and ⭐ the delta-rule line's
own authors place it inside their family — ⚠ with their hedge *"up to specific parameterization"*, which we
carry, our arm's identity being stated and tested (Appendix O.2b) rather than inherited from that sentence.
**Evaluation conventions:** Based owns the field's only explicit *state-bytes-during-generation* axis, populated
by six neural sequence mixers and nothing else; MAD normalises to *"an iso-state and iso-parameter setting"*
across **neural** architectures only; Sparse Delta Memory reports a state-to-parameter ratio; HOLA compares a
bounded exact KV cache against a *matched* recency cache — cited favourably: the field is already conceding that
part of the payload belongs in an exact store.

**The positioning claim, scoped, dated and checkable.** We surveyed 14 candidates against a four-part definition
of the control (non-parametric store · sized to the learned memory's **declared state bytes** · run on the same
task · verdict reported): **0 HIT · 2 PARTIAL (both out-of-family) · 7 NEAR-MISS · 5 NO**.

> **Across the modern neural sequence-memory family surveyed here — delta-rule and linear-attention models,
> SSMs, test-time-trained memories, explicit-slot memories, semiparametric hybrids, and the family's own recall
> and architecture-search benchmarks — we find no paper, as of 31 July 2026, that sizes a *non-parametric* store
> (a table, kNN index, count-based model, or explicit (k, v) rows) to a learned memory's **declared state-byte
> budget**, runs it as a control on the same task, and reports the comparison. The nearest existing conventions
> are iso-state normalisation *across neural architectures only*, a state-bytes axis populated exclusively by
> neural sequence mixers, and isoFLOP/isoParameter reporting with a state-to-parameter ratio. Budget-matched
> controls against non-learned alternatives are, by contrast, routine **outside** this family — at matched or
> explicitly accounted space in learned data structures, and, concurrently with this work, as a token-matched
> recency window in LLM-agent memory evaluation. We therefore position this audit as **importing an established
> discipline into a family that has not adopted it**, not as inventing it.**

⛔ We do **not** claim the unscoped version (*"no published rival paper runs a non-parametric matched-byte
control"*): it quantifies over all papers and is false. ⭐ One survey clause is no longer only a survey clause —
the SSM family is now represented by a **measured** arm at byte-identical state; ⚠ by **one** member (Mamba-1
and Mamba-3 are different state types, both declared NOT-RUN). **The ancestry we concede, in our own voice**
(quote scopes: Appendix Q): audit-at-equal-bits is standard outside this family, and the substitute audit is the
partial-input tradition, whose converse caveat we carry. ⭐ **What survives is stronger than a monopoly claim:**
seven of the fourteen candidates build the *adjacent* instrument and stop one step short of it. **A conceded
ancestor is worth more than a contested monopoly.**

---

# 6. Limitations

**L1 — One-family thinness, verbatim.** *Three rival families audited against **one** surviving synthetic family
is a thin cross-family audit, and the rival rows cannot carry more weight than that.* ⛔ Our validation rule
cannot separate a substitutable family from a substitutable *anchor* (§2.4), and the coverage every store-side
verdict rests on is one family; a second dividend family built to §2.4's rule is the cheapest thing that would
strengthen this work. ⭐ One folded measurement sharpens how hard that bar is rather than lowering it: the same
admissibility discipline, run against the two best-documented real streaming venues, **retires both** — the
thinness is a property of the bar, now measured on designed *and* real venues, not an idiosyncrasy of this rig.

**L2 — A second thinness: the comparative arm count is two.** One of six rival arms clears its own blank-store
control in no configuration (TTT-MLP), a second is INIT-UNSTABLE (TTT-Linear), two more are SELECTION-DEPENDENT
(DeltaNet, GDN-2); on the byte-frontier column none of twenty cells separates from its control; ⛔ **and our own
store fails the same gate** (lift −0.0465 ± 0.0406). No comparative margin in favour of any arm over TTT-MLP,
TTT-Linear, DeltaNet, GDN-2 or the CLU appears anywhere in this paper, so the *comparative* half of the audit is
carried by **two** arms — GDN and Mamba-2. ⚠ The direction rule keeps this from being vacuous: it suppresses
comparative margins only, never an arm's own loss to its own table, which is the headline's only quantity and is
quotable for all six rival arms and for ours. **L2a — The rescue gate is underpowered below nine seeds *and*
sensitive to the selection rule**, and we found out both about our own harness: every verdict is a nine-seed
verdict quoted only where stable across all three registered rules, our first pass's three-seed verdicts are
withdrawn rather than quoted, and a pre-registered prediction that rescue statuses would be stable under
re-tuning was **refuted**. ⚠ Our own column is the one case where cross-path agreement is *not* independent
evidence.

**L3–L10, in brief; full statements in Appendix S.7.** The launder's scope is §1's boxed sentence and nothing
larger (L3). The tuning standard was met and we show the grid is not the binding constraint for the arms it was
drawn around, but **not** that no tuning protocol would rescue an arm (L4); our own selection rule is weaker
than it looks and the held-out secondary gives the same headline with a *smaller* rescued set, making the audit
thinner rather than stronger (L4a). Measured and reasoned families are never blurred — **Sparse Delta Memory and
Titans are adjudicated from published equations alone** (L5). ⛔ **The byte-ratio caveat travels with every
dividend or byte claim:** `ratio ≥ 2.20×` (`≥ 2.40×` with a spectator dimension), the audited cell at 54.56×
(**modal, 8 of 9 seeds**; 45.60× on the sixth-item seed), and **no cell measured in this work is a byte-matched
dividend — the minimum ratio measured anywhere is 17.11×** (L6). A table deletes exactly by construction and the
deletion column has no cross-family row (L7); the ancestry is conceded (L8). ⚠ What the theory does not derive
(L9): the particle-gradient prefactor is open; **naming `s` for a learned multi-atom well gates the transfer of
every geometric domain statement**, and §4.5's value holds **for this store only**, ⛔ flagged for a check rather
than discharged, with the correction's known direction running *against* us; T5.4's coupling list is not proven
exhaustive; a live launch-momentum probe is NOT-RUN, its target quantified at a factor **65**. Protocol caveats
carry domains (L10): below `s/sep ≈ 0.15` the basin boundary is *inertial*; **`λ_min > 0` does not certify a
nonempty basin**; `sep/2` is not a certified inradius.

**L11 — Weight class.** Every measured cell runs at `d_in = 5`, 5–6 stored items, ~10-token streams, float32,
CPU. **Nothing here transfers to a language-model claim**, and no language-modelling run was sized or attempted.
⚠ The folded measurements do not change this: the two streaming-venue runs are baseline-only with **no memory
arm of any kind**, their value legs are declared NOT-RUN, and §4.6 is a 3-seed component-build measurement on a
different rig. **L12 — Reproducibility incident, disclosed.** A per-arm fit key used a process-salted hash, so
two runs at identical seeds differed; found and fixed **before any number in this work was recorded**, and the
reported run is post-fix and reproduces bit-identically. We report it because it briefly produced numbers.

---

# 7. Conclusion

We asked whether a bounded-state memory's learned test-time dynamics buy anything over a non-parametric table
holding the same bytes, built one protocol that makes the question answerable across memory types, and ran it on
rival families and on our own system with the same columns for everybody. At matched state bytes, on the one
designed family that survives our own protocol validation, at CPU scale with `d_in = 5`, 5–6 stored items and
~10-token streams: **nothing in the audit beats a zero-extra-byte reader of a raw table holding the same
bytes** — not the rival families (0 of 6 arms, nine seeds, full grid, −0.2563 … −0.4602, unchanged at 5× budget
and under held-out selection) and not ours (−0.2897 ± 0.0328), an arm which on those same nine seeds does not
clear the audit's own functioning check either. Against a weaker but natural control read through each memory's
own projections, **test-time dynamics does pay for the delta-rule and SSD arms and does not pay for ours** — and
we may report the stronger control beside it only because both were registered before measurement.

Two structural results bound what the answer could have been: matched bytes is unreachable by an accounting
identity whose floor is exactly the price of one privately-deletable parameter group per item, so **compression
and byte-exact deletion are the same trade**; and the one coupling a row-selecting table provably cannot express
obeys `exp(−½(d/s)²)` on our learned store (ruler named, subtraction convention declared, the fitted width
flagged for a check whose known direction makes the suppression stronger), exponentially suppressed by the same
admission gate that keeps the store writable. ⭐ Three measurements from outside this rig tighten the paper's
boundaries rather than relaxing them: the store-versus-table indistinguishability reproduces on a **real-image
stream census** in **0 of 9 cells** with the store holding **2,364×** the table's bytes (matched *items*; a
component-build measurement); the admissibility discipline **retires both** real streaming venues we gated — **a
fallback being retired, not a venue crisis** — which is why every real-data value leg here is a **declared
NOT-RUN for want of an admissible venue, never a null**; and the same runs produced the label-embargo finding.
⛔ None is a benchmark result for any memory: **no memory cell of any kind, ours or a rival's, ran on either
venue.**

⭐ **A store organised well enough to be safe is organised well enough to be a table.** That is the audit's
conclusion, it is a design identity rather than a defect of this implementation class, and it is where this
paper stops.

---
---

# Appendices

> **Appendix policy.** Main text carries the main results, every load-bearing control and every defense
> (this venue's reviewers are not required to read appendices). Everything else — full cell tables, flag
> provenance, pre-registration scorecards, declared NOT-RUNs, negative results, figure specifications and the
> folded evidence in detail — lives here, fully written. Nothing is omitted.

## Appendix A — Flag provenance

Every quantitative result travels with the configuration that produced it; cells in different sections must
not be reproducible into an apparent contradiction.

**A.1 The audit run, full tuning grid** (§2.5, §4.1–§4.4) — the source of every rival number in the main text.
Environment: JAX **0.9.0** / Equinox 0.13.4 / Optax 0.2.6, float32 on both sides of the ledger, main
environment reused (no re-resolution). Seeds **0, 1, 2** (registered primary) **+ 3–8** (declared power
addition) ⇒ `n = 9` pooled on every rival cell; sample sd (`ddof = 1`), `SE = sd/√n`. Fit-stream seeds
`seed + 101`, `seed + 102` (never the eval stream; byte-identical to the first pass's, asserted in a test);
`seed + 103` is the **held-out** stream used only for the declared secondary selection rule. Family:
`aggregate@base` only (⛔ the frontier column was **not** re-run under the full grid — Appendix J; `recency`,
`manifold` struck by §2.4). Arms: the five incumbents (`ttt_linear`, `ttt_mlp`, `deltanet`, `gdn`, `gdn2`);
the SSD arm has its own table (A.1f) and the CLU column its own (A.1e). Task flags: `family=aggregate`,
`capacity=6`, `consolidate_every=2`, staged admission on. Store flags: `capacity=6`, `budget=6`,
`min_atoms=192`, `min_atoms_base=192`, `min_atoms_c=1.0`, staged admission on (the audited cell, unmodified).
Grid: `lr ∈ {1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2} × wd ∈ {0, 0.1}`, TTT arms additionally `b ∈ {1, 16}`
⇒ 24 configurations per TTT arm, 12 per delta arm; Adam at `wd = 0`, decoupled AdamW at `wd = 0.1`; 400 outer
steps + a 2000-step re-check on the sub-grid containing every 400-step winner. ⚠ Declared deviations from the
tuning standard: `β = (0.9, 0.98)` and cosine decay **not** adopted, so exactly one variable moves between
passes. ⚠ Initialisation-key scheme **changed and priced**: one init per (arm, seed, `b`) shared across all
`(lr, wd)`; the first pass split one key sequentially across grid points (priced by the `lite control` column,
I.1a). Iso-state budget **1364 float32 = 5456 B**; head widths **29 / 12 / 36**, registered before any run and
asserted in the test suite. Identical-encoder invariant enforced per cell (`phi_id = 09dc0ee5…`). Byte law:
corrected `ratio = [A(D+2)+d]/(d+m)`; ledger identity green on every cell (`5456 B / 100 B / 54.56×` — ⛔ the
**modal** value, 8 of 9 seeds; `5472 B / 120 B / 45.60×` on seed 8); floors **2.20×** (`n_spec = 0`) /
**2.40×** (`n_spec = 1`). ⛔ Never *"verified to 1e-9 in all 28 cells"* (§3.1). Reproducibility: the first pass
reproduces **digit-for-digit** at the base code from this branch on all five arms, so every difference between
passes is attributable to a declared change. Test suite at the recorded commit: **1143 passed, 0 failed**; lint
clean. Wall clock ≈ 31 min total. ⚠ Disclosed incident (inherited from the first pass): a per-rival fit key used
a process-salted `hash()`, fixed to a stable index **before any reported number was recorded** (§6 L12).

**A.1b The first pass (reduced grid)** — provenance for Appendix I.1 only. Same environment; seeds 0–2,
`SE = sd/√3`; Adam, 400 steps, `lr ∈ {1e-3, 3.16e-3, 1e-2}`, best-of-grid on the fit split, TTT arms `b ∈ {1,
16}`, initialisation key split **sequentially** across grid points. Families: `aggregate@base` +
`overload@load1x_shipped` (the only source for the first pass's frontier rows, Appendix H.2). ⚠ Reduced grid
against the 6 × 2 standard — declared at the time as a budget choice, **not** as compliance; superseded for
every main-text number by A.1. ⛔ **No verdict from this pass is quoted in the main text.**

**A.1c The uniform nine-seed re-aggregation** (§4.1, §4.3, I.1c) — a **re-aggregation only**; no cell was
re-measured. Same environment (numpy 2.4.1, CPU); seeds 0–8 on every column; paired statistics (`full − null`,
`full − blank`) computed per seed then aggregated. Columns added at `n = 9`: projected (arg-min) launder ·
dividend against it · same-keys null · blank store · the three `+0 B` readers · the paired differences · the
per-seed byte ledger. Four selection columns re-aggregated: primary (fit-split), the first pass's code path,
the `lite control` sub-grid, the declared secondary (held-out). Fidelity check: every quantity already
published at `n = 9` reproduces digit-for-digit (raw-table margins −0.4602 / −0.4425 / −0.2732 / −0.2600 /
−0.2592; rescue lifts 0.093 ± 0.134 / −0.071 ± 0.090 / 0.294 ± 0.077 / 0.880 ± 0.227 / 1.025 ± 0.329). ⚠ TTT
byte columns are per-seed (`ttt_linear` 5220 B at `b = 16`, `d = 29` or 5328 B at `b = 1`, `d = 36`; `ttt_mlp`
5376 B or 4656 B) — ⛔ no single TTT byte figure is *the* nine-seed value.

**A.1d The byte-frontier rows at nine seeds** (§4.5, Appendix H). Family/metric `overload@load1x_shipped`,
`decode` (higher better), 24 queries, 7 stream tokens, 6 live items, chance 0.1667. Arms `deltanet`,
`ttt_linear`, `gdn2` at `d_head ∈ {2, 4, 8, 16, 36}` (15 cells) on the current code path; ⛔ `ttt_mlp` and `gdn`
NOT RUN. Seeds 0–8. CLU curve **banked, not re-measured**: decode 0.972 → 0.097 as the ratio falls 478× →
2.28× (`n = 3`).

**A.1e The CLU column at nine seeds** (§4.1–§4.3) — ⭐ **a re-aggregation of banked per-seed cells; no CLU cell
was re-measured.** Commit `eaecc91`, clean tree, no tracked file touched. Environment: main venv reused, JAX
0.9.0 / Equinox 0.13.4 / Optax 0.2.6 / numpy 2.4.1, float32, CPU (⚠ the aggregation itself is pure numpy over
recorded JSON). Seeds 0–8 on every CLU column, every margin and lift **paired per seed**. Cell `aggregate@base`
(`capacity = 6`, `consolidate_every = 2`, staged admission on — the shipped cell, unmodified); shipped read
flags (Appendix P): deterministic read, `T = 0`, zero launch momentum, **no Langevin step**. Ledger:
1364 float32 = 5456 B; F1 **5376 B** / F2 **5200 B**; ⛔ **modal (8 of 9)** `5456 B / 100 B / 54.56×`, seed 8
`5472 B / 120 B / 45.60×` (six items admitted); integer identity green on **all nine**. Metric `neg_mae`.
Code-path identity: bit-identical across both paths (max |Δ| = 0.0 on `full`, `launder`, `blank`, null) ⇒
cross-path agreement is *not* independent evidence for our own verdict. Independent check: an out-of-harness
numpy recomputation reproduces the shipped rule exactly (`full −0.437047 ± 0.041739`, `blank
−0.390587 ± 0.012374`, `lift −0.046460 ± 0.040631`, 2 SE = 0.081261 ⇒ RESCUED = False, `dividend
−0.056070 ± 0.031549`), and seeds 0–2 reproduce the banked three-seed values digit-for-digit. `+0 B`
convention: per-seed arg-max **−0.2897 ± 0.0328** vs banked fixed-reader rule −0.2862 ± 0.0317 (Δ = 0.0035; no
claim turns on the choice). ⛔ **0 new measurements.**

**A.1f The Mamba-2 (SSD) arm** (§4.1, §4.3, §4.5). Same environment as A.1 (JAX 0.9.0 / Equinox 0.13.4 / Optax
0.2.6 / NumPy 2.4.1), float32 both sides of the ledger; nine seeds; byte-identical state to the delta-rule arms
(5184 B); the run reproduces all five incumbent arms **bit-identically**, which is what makes its row
comparable to theirs. ⚠ Its 5× budget re-check, its paired `full − null` and its per-head-width `+0 B` reader
margins are **declared NOT-RUN** (Appendix J).

**A.5 The third-substrate no-daylight row** (§4.8, R.1) — ⭐ **folded, not re-run** ⟦N276⟧. Measuring commit
`0f057a0` on a dedicated worktree; main venv reused, **JAX 0.9.0**, float32, module-provenance printed per
cell. Substrate: **Split-CIFAR-10, reduced protocol** (5 tasks × 2 classes, class-incremental, 1000 train /
500 test per task); store `addr_dim = 12`, `payload_dim = 1`, **32,768 atoms**, `capacity = 16`,
`well_budget = 8`, `leak = 0.02`, `write_steps = 300`, `read_steps = 800`, `address_steps = 400`, 128 cue
queries per cell. Store-width flag `atom_width_frac_spacing = 1.5` co-scaled to each seed's measured spacing
(⚠ the **banked census value**, quoted with its instrument; the shipped config default differs and this cell is
**not** asserted reproducible at shipped defaults), `atom_kernel = wendland` (cutoff 2.5),
`atom_site_local_init = True`. Encoder arms: a fitted contrastive encoder (`phi_dim = 256`, 8,000 fit steps —
"strong" only by the CL-accuracy metric that defines it on that rig), an **unfitted** random-convolution
control (`phi_dim = 256`, 0 fit steps), a PCA reference at `phi_dim = 12`; stream fingerprints bit-identical
across arms per seed ⇒ every cross-arm quantity is paired. Seeds 0, 1, 2 per arm = 9 cells. Registration: the
two-branch daylight rule and both branches' reportability registered **before the first cell**; priors
**Q6 = 0.70** (no daylight) / Q5 = 0.15 (daylight); **nothing tuned**. Instrument label: measured under that
rig's **pass-3** instrument, before its later hardening — numbers stay labelled with the instrument that
produced them. Bytes: ⛔ **matched-items, not matched-bytes** — store **1,966,848 B** vs **832 B** (cue table,
`n = 16`) = **2,364×**; 416 B (ring table) = 4,728×; **3.147×** with the encoder priced on both sides (fitted
arms) / 13.278× (PCA); ⛔ no leg of that instrument reads bytes. Launder audit **asserted, not intended**: the
table re-derives its keys from the projected encoder output and raises unless bit-identical (9/9 cells,
`launder_key_dim = store_address_dim = 12`). Wall: 27,043 s of census cells + 2,373 s scale control.

**A.6 The two streaming-venue gate runs and the label embargo** (§2.4, R.2–R.3) — ⭐ two **folded,
baseline-only** venue-admissibility runs (⛔ **no memory cell of any kind — no CLU, no rival**) plus the
protocol finding they produced ⟦N294 / N295 / N296⟧. Repo HEAD `7fcef50`, unchanged, 0 tracked files modified.
Environments: dedicated scratch venvs — CPython 3.12.9, `river 0.25.0`, numpy 2.5.2, scikit-learn 1.9.0, scipy
1.18.0 (INSECTS); the same plus torch 2.13.0, matplotlib 3.11.1 (Metro); project venv and lockfile untouched;
`m = 1` (undecimated) for every published number. **INSECTS protocol:** prequential accuracy, window 1000 (the
benchmark authors' convention), index 0 excluded ⇒ n_scored = **79,985**; primary condition
`incremental-reoccurring`, second condition `incremental-abrupt-reoccurring`. **INSECTS arms:** exemplar stores
— SAM-kNN (⚠ **our own port; the streaming library ships none**, so every *"one-line baseline"* cost estimate
is void; validated against the authors' published Weather row, 21.70 / 21.68 % error vs published 21.74 /
21.53, and 400/400 brute-force agreement on 500 randomised kNN cases) and plain kNN windows at
`L ∈ {250 … 14,782}` — each run **raw AND causally standardised with the max consumed** (a declared,
pre-registered anti-hobbling rule); strong reference **`river` 0.25.0 `ARFClassifier` at 100 trees**, 3 seeds
(**78.8139 ± 0.0526 SD** vs the published **77.13**, a **MOA** number at an **unstated** ensemble size — 100 is
MOA's default and reading their arm as ARF-100 is our inference; ⚠ **`river` ARF ≠ MOA ARF**, and ours is the
stronger, the direction that makes firing *harder*). ⚠ ARF's measured state is **9,542,925 B = 14.35×**
SAM-kNN's 665,000 B ⇒ **ARF is byte-matched to nothing** and is the accuracy frontier only. ⚠ The SAM-kNN
**3,000-exemplar STM cap** is a **reference-implementation default** (`ltm_size = 0.4` at `L_max = 5000`), not
a published parameter — the ICDM paper publishes `k`, `L_min`, `L_max` only. **Metro protocol:** hidden clock
(`date_time` withheld from every arm); 24-h horizon; **24-h label embargo ON for every primary number**;
prequential MAE/RMSE over all **34,848** scored pairs; **32 features = 132 B/exemplar** (24 recent lags ⊕ 3
weekly echoes ⊕ 4 weather ⊕ holiday), identical for every arm (⚠ the analyst's declared choice, and the k-NN
attack is defined by it; mitigations: identical features to the strong baselines, plus a static 70/30
chronological holdout reproducing the parity outside the harness); feature gap-fill ≤ 3 h, **targets never
imputed**; ⚠ a 7,386-hour sensor hole makes the stream two eras. **Metro arms:** exemplar k-NN over past
windows at `L ∈ {250 … 34,847}` (**registered `k = 5`**; a declared anti-hobbling k-ladder beside it, labelled
at every quotation); **nine** strong references, single seed. Seeds/determinism: exemplar arms are single
deterministic runs (**no variance estimate is claimed and none exists**); ARF 3-seeded; Metro strong baselines
single-seeded. Shuffle null: fixed-seed uniform permutation of the **pair sequence** — `P(X, y)` and `P(y|X)`
preserved exactly, positive control exact (persistence 574.1346849611 ordered vs shuffled, diff 0); ⚠ a
shuffle, **not** a drift-free data source. Declared harness fix: a causal z-score blow-up at `t < 500` fixed by
a declared ±10 SD clip on every standardised arm (pre-fix numbers used nowhere).

## Appendix B — The protocol, in operational detail

**B.1 Byte classes.** `F1` = parameters shared across sequences (including any learned initial state); `F2` =
per-sequence state — **the audited budget**; `F4` = per-read transients (buffers, index sets, top-k
selections), not state, but declared and granted to the launder on equal terms. Every breakdown must sum to its
total or the run raises.

**B.2 Building a byte-matched table.** For a memory with an explicit float state and an explicit
`(θ_K x, θ_V x)` stream, `n_rows = ⌊state_floats/(d_k + d_v)⌋` is **forced by the ledger, not chosen**;
losslessness is checked per cell. ⚠ The mini-batch is inside the state, so a TTT arm's whole ledger row moves
with `b`:

| arm | `b` | `d_head` | state floats | F2 state B | table rows | matched table B |
|---|---|---|---|---|---|---|
| ttt_linear | 16 | 29 | 1305 | 5220 | 22 | 5104 |
| ttt_linear | 1 | 36 | 1332 | 5328 | 18 | 5184 |
| ttt_mlp | 16 | 12 | 1344 | 5376 | 56 | 5376 |
| ttt_mlp | 1 | 12 | 1164 | 4656 | 48 | 4608 |
| deltanet / gdn / gdn2 / mamba2 | — | 36 | 1296 | 5184 | 18 | 5184 |

**B.3 The two mandatory table variants.** *Projected*: rows are the memory's own `(θ_K x, θ_V x)` pairs, read
through its own output head. *Raw-metric*: the same bytes in the raw address/payload space, read by the best
`+0 B` reader. **Both are reported for every arm** (§2.3).

**B.4 The +0 B reader set.** arg-min over keys · 2-NN mean · 2-NN inverse-distance · echo of the query ·
insertion order · order-aware pair reader. "Zero extra bytes" means no parameter, no threshold and no stored
quantity beyond the launder's own table; a fitted scalar temperature costs 4 B and is declared as such.

**B.5 The rescue gate, formally, with its direction rule and power requirement.** Arm `a` on cell `c` is
RESCUED iff its full read exceeds its own blank-store control by more than **2 SE of that lift**.

> **Direction rule.** Failing the gate suppresses every **comparative** margin *in favour of another arm over*
> the non-rescued arm — the margins that would flatter a competitor by beating something that may not be
> reading its store at all. It does **not** suppress the arm's own loss to its own byte-matched table, which is
> a statement about that arm against its own bytes and is if anything *strengthened* by the arm's not
> functioning. This paper's headline is built entirely out of the second kind of quantity.

⚖ **Selection-stability rule.** A rescue verdict is quoted only where stable across the three registered
selection rules; an arm rescued under some and not others is printed **SELECTION-DEPENDENT** and treated
exactly as non-rescued by the direction rule. ⚠ **Power:** the control is a single initialisation draw per
seed, whose spread can exceed the lift being gated (one arm's blank reads across three seeds:
−0.962 / −2.634 / −1.390); at `n = 3` three legitimate configurations return three different rescued sets
(`{ttt_linear, gdn, gdn2}`, `{}`, `{ttt_linear}`); at `n = 9` two independent code paths agree on four of five.
⇒ report rescue verdicts at `n ≥ 9`, and prefer a control paired per seed or averaged over several
initialisation draws — we did neither of the latter two and report the nine-seed verdict instead; both remain
open improvements. **Verdicts as applied (dividend family, `n = 9`):** ✅ RESCUED `gdn`, `mamba2`; ⚠
SELECTION-DEPENDENT `deltanet` (+0.077 ± 0.045 held-out), `gdn2` (+0.669 ± 0.339 held-out); ⛔ NOT RESCUED
`ttt_mlp`; ⚠ INIT-UNSTABLE `ttt_linear`; ⛔ NOT RESCUED the **CLU** (lift −0.0465 ± 0.0406 over its own blank
store). On the byte-frontier column **none of the twenty (arm × head-width) cells at nine seeds** separates
from its control, so that column carries **no quotable margin in either direction**.

**B.6 The identical-encoder invariant.** All arms of a cell share `φ` and its byte cost; a content hash of
`(q₀, keys)` is asserted across arms; a 1e-9 perturbation raises (tested). **B.7 The ledger identity, as a
blocking check.** `full == 4[N_at(D+2) + K·d]` and `launder == 4K(d+m)`, asserted as **integers** on every
cell; a drifted store raises.

## Appendix C — Byte-floor theorem: verification detail

| check | result |
|---|---|
| byte decomposition `V/4 = N_at(D+2)`, `code/4 = Kd`, `launder/4 = K(d+m)` | ✅ **28/28 exact (integers)** |
| corrected law reproduces the recorded ledger ratio | ✅ **28/28 exact (rationals, 0 ulp)** |
| the closed form stated in our pre-registration | ⛔ **24/28**; the four spectator-dimension cells miss by **+8.6667** (52.00 measured vs 43.33 as registered) |
| shell-atom surcharge | ✅ `52.00 → 58.40` exactly; `+1/(D+2) = +12.5 %` on the atom term |
| floors | ✅ 2.20 / 2.40 (Gaussian, `n_spec = 0/1`); 2.40 / 2.60 (shell) |

**The erratum's re-score, cell by cell** (re-scored offline from the recorded artifact; the harness was not
re-run): **24 of 28 cells bitwise unchanged**; the four spectator-dimension cells change 43.3333 → 52.0000
(+8.6667, +20 %) with their printed floor 2.00× → 2.40×; the **measured minimum ratio 2.2824× is unchanged**.
⭐ The bug was invisible to the test suite because **no test exercised a spectator dimension**; a regression
test now covers it. That is the coverage lesson, not merely a formula fix.

## Appendix D — The dichotomy theorem (settled-point reads are untrainable end-to-end)

> **Theorem T3.** For the dissipative velocity-Verlet map `T_θ` with separable `H = T(p) + V_θ(q)`, for every
> `γ ∈ (0,2)`, `dt > 0`, inertial mass `M ≻ 0`: **`Fix(T_θ) = {(q, 0) : ∇V_θ(q) = 0}`**. The defining equation
> contains `θ` **only**, so the read's parameters split into **fixed-point parameters** `θ` (with
> `∂q*/∂θ = −(Hess V_θ(q*))⁻¹ ∂_θ∇V_θ(q*)`, exact, no `(γ, dt, M)` correction) and **transient parameters**
> `ζ ∈ {q₀, p₀, M, γ, dt, integrator}`, which appear in the *approach* and nowhere else, so `∂z*/∂ζ ≡ 0`
> exactly in the fixed-point limit. At finite budget the surviving sensitivity is the un-decayed remnant of the
> contraction, `‖∂z_N/∂ζ‖ ≍ K_ζ e^{−C}`, `C ≡ Σ_p N_p ln(1/ρ_p)`, with `K_ζ = O(1)` for `ζ = q₀` and `O(N)`
> for `ζ ∈ {M, γ}`.

**Measured on the learned system:** `‖∂L/∂φ‖` = **0.0** (implicit) / **2.654e-9** (unrolled) / **6.421e-3**
(trajectory read) — a ratio of **2.42e6**; the inertial-mass gradient is **exactly 0.0 bitwise on 3/3 seeds**.
**Verified on designed geometries:** `Fix(T)` is `(γ, M)`-independent to **1.67e-15**; the `e^{−C}` slope is
**−0.9941 over 143.9 decades** (per-γ −0.981 / −1.007); after dividing out the derived `N` prefactor the
`{M, γ}` law is exact to **±1 %**. For the read schedule used here `C = 18.34` ⇒ `e^{−C} = 1.084e-8`, which
brackets both measured harness gradients. ⭐ **Zero fitted parameters, and it retro-explains a prior negative
result:** an earlier attempt to *learn* an address by gradient descent through a settled-point read failed at
chance (4.2 % on one implementation; 0–2 of 18 on another; loss frozen to 7 significant digits over 4000
steps). Every mode is underdamped at those friction values, so `ρ = √(1−γ)` exactly and landscape-independently
and `‖∇_address‖ = 3.3e-1·(1−γ)^{600}` has no free parameter but the `γ = 0` anchor; measured/predicted over
five decades: **1.00 / 0.92 / 1.17 / 2.14 / 8.9**, the last two non-monotone in `γ` — that instrument's
numerical floor, not physics. At a 3000-step probe the predicted address gradient is `10^{−33.4}`, **26 orders
below float32 epsilon**: gradient descent was optimising round-off. ⛔ **Consequence, as a scope limit rather
than a proposal:** retrieval robustness *is* `∂(final)/∂q₀ → 0` and the convergence budget `C` sets both; any
claim about training *through* a settled-point read is bounded by `e^{−C}`, and we make none. **Not derived,
never quoted as settled:** the *prefactor* of the harness's own particle gradients (toy law `N e^{−C}`,
measured at `≈ e^{−C}`, a factor `N ≈ 1200` apart; the structural claim is unaffected).

## Appendix E — Protocol caveats, as numbered propositions with domains

1. **Prop D2a (when a settle is arg-min).** Under (H1) separable wells with `λ_min > 0`, (H2) equal well depth,
   (H3) a settled-point-only read and (H4) a query law inside the certified basins, the settle map is
   **exactly** arg-min over the stored centres and the dividend against a matched-key arg-min launder is
   **exactly 0**, not "small". The audited anchor cell realises it: dividend **0.0000**, disagreement mass
   `D = 0`, reproduced three times independently. Which hypothesis, dropped, breaks the conclusion (designed
   two-well geometries; every `d/s` here is a **designed**, architecturally-specified width, not a fit, so the
   fitted-width estimator and its subtraction convention do not enter): dropping (H2) shifts the axial
   separatrix by `δ = ln(A_i/A_j)/(d_ij/s² − 4/d_ij)` — measured boundary +0.0163…+0.0791 vs predicted
   +0.0156…+0.0767 over five depth ratios (**3.1–4.8 % relative**; measured/predicted **1.141** over 7 cells);
   dropping (H1) merges wells (`D = 0.0000` down to `d/s = 2.86`, **0.0550** at 2.29, **1.0000** at 1.71 — the
   conclusion fails **and the store fails with it**); (H4) is ⛔ **refuted as a hypothesis in the equal-depth
   case** (`D = 0.00000` at every `sep/σ_q ∈ [2,10]`, n = 4000/cell — with equal depths the settle boundary
   *is* the Voronoi boundary by symmetry); dropping (H3) is the **only** drop that opens a channel without
   degrading the store (within-basin sd of `q*` = 6.6e-10, piecewise constant, against within-basin sd of
   `q_t/σ_q` = 0.99 (t=1) · 0.43 (t=10) · 0.042 (t=100) · <1e-3 (t=240) · 4.7e-9 (t=1200); `p_t` peaks at
   **1.88 σ_q at t ≈ 10**) — a table returns one value per basin, the trajectory a continuum, but only for
   `t ≲ 240` steps. ⚠ **`D` is the dividend's *variance*, not its magnitude**: the cell with the largest `D`
   (0.931) has the **worst** dividend (−0.875). `D` is never a progress signal. **Status:** proven in the
   symmetric case (measured offset 2.92e-8 against a 2e-3 bar); evidenced at 3–5 % in the asymmetric case on a
   2-D toy for `|δ| ≤ 0.25 d_ij`, `s/sep ∈ [0.15, 0.30]`; **invalid below `s/sep = 0.15`**, where the basin
   boundary is inertial rather than static.
2. **Below `s/sep ≈ 0.15` the basin boundary is inertial** — measured capture radius 1.306 against a midpoint
   of 1.000, a **21× miss** for the static correction; the asymmetry is destroyed by damping, so it is
   momentum-carried. No static proxy is valid there.
3. **`λ_min > 0` does not certify a nonempty basin** — measured capture radius **0.000** at `λ_min = +0.910`,
   reproduced independently on a task-like store at `λ_min = +1.43` (2 of 6 sites).
4. **`sep/2` is not a certified inradius**, and the corrected proxy is valid only inside its four-condition
   domain; ⛔ the corrected inradius gives **no** improvement over `sep/2` on the anisotropic store the
   certificate work uses.
5. **A designed near-degeneracy does not survive superposition on a learned store** — a tilt reduces `λ_min`
   monotonically; refuted in sign on two independent implementations.
6. **A soft-certificate budget previously quoted as located is not located at all** under the corrected ruler
   (`B ≥ 0.542` is unrefuted; the edge was located by an estimator breaking down, not by a property of the
   store).
7. **Truncated backpropagation depth governs `∂q_N/∂θ` only where fixed-point sensitivity dominates the
   transient** — in a `K`-item store the far-well parameters are exactly the interference gradients, so
   truncation preserves the on-well gradient and destroys the crowding gradient.

## Appendix F — The third-party attribution sweep, in full

Deleting the query's *second*-nearest stored key and dividing by deleting its *nearest*, on the learned store,
3 seeds per radius:

| `ball_radius` | coverage | `sep` | fitted `s` | **`d/s_fit`** | **measured coupling ± 2 SE** | ⛔ per-slot **table** | `λ_min` |
|---|---|---|---|---|---|---|---|
| 0.42 | **3/6** | 0.5481 | 0.482 | 1.10 | **0.814 ± 0.31** | **0 exactly** | 1.26 |
| 0.55 | 3/3 | 0.7402 | 0.412 | 1.71 | **0.344 ± 0.18** | **0 exactly** | 2.64 |
| 0.64 | 3/3 | 0.8614 | 0.400 | 2.05 | **0.226 ± 0.04** | **0 exactly** | 2.84 |
| 0.80 | 3/3 | 1.0767 | 0.385 | 2.68 | **0.0970 ± 0.02** | **0 exactly** | 3.03 |
| **1.00 (the audited cell)** | 3/3 | **1.3459** | 0.362 | **3.59** | **0.01534 ± 0.006** | **0 exactly** | 3.24 |
| 1.20 | 3/3 | 1.6211 | 0.362 | 4.41 | **1.55e-3 ± 2e-3** | **0 exactly** | 3.16 |

| fit | slope | implied `s` | prefactor | **R²** | decades swept |
|---|---|---|---|---|---|
| static ∇V ratio | −3.158 | **0.3979** | 0.379 | **0.9953** | **2.72** |
| dynamical slot coupling at `t = 1` | −3.155 | **0.3981** | 0.378 | 0.9952 | 2.72 |
| independent well-fit of the learned store | — | **0.4006** | — | — | — |

⚠ The fitted `s` is **not a constant of the architecture** (0.482 → 0.362 across the sweep) — the ruler moves
under the thing it measures — and every `d/s` in this paper names which ruler it uses (atom-width
`atom_init_width = 0.30` vs fitted `s`); the two differ by 1.33×. ⛔ The table's third-party Δ is `0.0` by
construction (Prop T5.4), never a win; the non-vacuous half is asserted beside it — deleting the row the query
*did* select moves the same table by a whole payload level. ⚠ **Slot count buys no per-item capacity:** the
slot vector is the image of a `(3d+1)`-dimensional launch map, measured rank **13 = 3d+1** at `d = 4` for every
`S ≥ 2` and **4 = d** under the audited read.

## Appendix G — The measured write footprint

Diffing `V_θ` before and after the stream on the audited cell: the write moves **192/192 atom centres (960
floats)** but only **160/192 widths and amplitudes**, plus 20 codebook floats ⇒ **1300 floats = 5200 B** of
per-stream deviation against **1344 floats = 5376 B** of initialisation, a state/parameter ratio of **0.967**.
The 32 unmoved widths/amplitudes are the one free slot, whose centres the allocator re-draws while its widths
and amplitudes are re-set to their initial constants — benign, fully explained, and, as far as we know, the
first time a store's write footprint has been *measured* rather than inferred from its write mask.

## Appendix H — The byte-frontier column, in full

**H.1 The nine-seed rows (the column of record).** `overload@load1x_shipped`, `decode`, chance 0.1667, 24
queries. Arms `deltanet`, `ttt_linear`, `gdn2` × `d_head ∈ {2, 4, 8, 16, 36}` (15 cells) plus the SSD arm's
five (H.1b) ⇒ **20 cells**. ⛔ **0 of 20 clear the rescue gate**; the largest lift anywhere is
**+0.0694 ± 0.0491** (`deltanet` at `d_head = 4`). Every arm's read lives in **0.12–0.24** on a six-way choice,
so ⇒ **no margin on this column is quotable in either direction** — including against our own store's banked
`decode` value. ⚠ At `d_head ≤ 8` the affordable table has **fewer rows than the stream has tokens**, so those
cells' launder is a *lossy* control and must not be read as "a table holding the same information".
**H.1b The SSD arm's frontier rows** (nine seeds, current code path): NOT RESCUED at all five head widths and
in all three registered selections (lifts +0.028 ± 0.035 / +0.009 ± 0.037 / +0.014 ± 0.045 on the audit cell),
taking the column from 0 of 15 to **0 of 20**. **H.2 The first pass's three-seed rows, kept as history** —
produced *before* the initialisation-key change, therefore not on the current code path, printed separately and
never mixed with H.1. **The CLU's curve** is banked at `n = 3`: decode **0.972 → 0.097** as the ratio falls
**478× → 2.28×**. ⚠ Two arms (`ttt_mlp`, `gdn`) were never run on this column and are declared NOT-RUN rather
than reported as nulls (Appendix J). The 5× under-training check on this column shows the arms get *worse*, not
better, at a larger budget.

## Appendix I — Pre-registration scorecards

**Preamble.** The dated registration documents — one per pass, each written **before** the run it governs — are
committed to the supplementary material. Scoring convention: a prediction is CONFIRMED, PARTIAL,
WRONG-DIRECTION or NOT-RUN; wrong-direction entries are printed, never re-scored.

**I.1 The audit's rival predictions (registered before the first pass), and the full-grid before/after.**
Overall: **6 confirmed · 4 partial · 3 wrong-direction · 2 NOT-RUN**. **I.1a** prints, arm by arm, the first
pass's reduced-grid values, the `lite control` column (the same sub-grid re-selected from the new fits — the
control that prices the initialisation-key re-draw) and the full-grid values; the re-draw moved arms by
**−0.148 / +0.125 / +0.018 / −0.015 / −0.042** against tuning effects of **−0.0303 / +0.0018 / −0.0009 /
+0.0006 / +0.0034**, i.e. **4× to 35×** larger. **I.1b** records the full-grid pass's own pre-registration
scorecard, including ⛔ the **refuted** prediction that the five rescue statuses would be stable under
re-tuning (three flipped; the gate's power, not the tuning, was the cause) and ⛔ the **refuted** count
prediction on `wd = 0.1` (predicted ≤ 2 of 15, observed 6 of 15 — always by fourth-decimal tie-break, which is
how the selection-rule finding was discovered). **I.1c** is the full column set at a uniform nine seeds — (a)
the raw-metric margins, (b) the projected-control handicaps (**0.276 / 0.263 / 0.425 / 0.856 / 0.942** and the
SSD arm's **+0.6139 ± 0.1386**), (c) the same-keys-null column, (d) the blank-store column and the paired
lifts, (e) the per-seed byte ledgers with the `b → (d_head, state, table)` map and the CLU's **modal (8 of 9)**
label, (f) the superseded three-seed CLU values printed beside the nine-seed ones. Per-seed admissible-cell
coverage on `aggregate`, seeds 0–8: **58/72 · 66/80 · 55/80 · 45/80 · 48/72 · 56/80 · 64/80 · 60/80 · 51/112**
(fractions **0.806 · 0.825 · 0.688 · 0.563 · 0.667 · 0.700 · 0.800 · 0.750 · 0.455**), with **5 of 8** offered
items admitted on seeds 0–7 and **6 of 8** on seed 8 — seed 8 is simultaneously the lowest-coverage cell, the
only six-item cell and the only different ledger; one mechanism explains all three, and it is the gate doing
its job. **I.1d** is the tuning grid audited: the three registered selection rules scored from the same fits,
with the stress columns (5× budget: −0.2184 … −0.2630; held-out selection: −0.24 … −0.49) labelled
**three-seed re-selections** at every appearance. **I.2** is the theorem set's scorecard; **I.3** the
attribution sweep's, where ⛔ the registered decade span, slot-decay ratio and free-fall prefactor were **all
missed** — the ruler, not the law, was wrong.

## Appendix J — Declared NOT-RUNs (never reported as nulls)

- **A Titans arm.** No official code; the chunk size is never given a numeric value; no seeds reported. An arm
  would be our reconstruction audited against our reconstruction's table; its `2·|M_θ|` state convention stays
  ⚠ **UNPINNED — our reconstruction, captioned every time.**
- **A Sparse Delta Memory arm.** Its official implementation requires Torch ≥ 2.8 / Triton ≥ 3.4 / SM 80+
  hardware. Positioning only. ⛔ Its published state/parameter ratios are **quarantined** (two independent
  extractions disagree) and none is quoted anywhere in this paper.
- **GRU and sliding-window-attention arms** (outside the ruled arm set; both would take the measured state-type
  count from three to five). **Mamba-1 and Mamba-3** (different state types, each needing its own ledger row).
- **The 5× budget re-check for the SSD arm**; **the SSD arm's paired `full − null`** and its per-reader `+0 B`
  means; **its per-head-width `+0 B` reader margins** on the frontier column.
- **A deletion column for any rival** — no rival family has a deletion verb.
- **The `recency` and `manifold` families** (struck by protocol validation as protocol-invalid; ⛔ in
  particular **no `recency` dividend is a null** — a `+0 B` reader of the table's own row order answers the
  restricted question at 1.0000 on 3/3 seeds, by construction).
- **`ttt_mlp` and `gdn` on the byte-frontier column** (no row; not reported as nulls). **The byte-frontier
  column under the full tuning grid** — the registered re-tuning trigger did not occur.
- **A re-measurement of the CLU column** — the nine-seed column is a **re-aggregation of banked cells**, which
  is why it was cheap; **the CLU's frontier curve at nine seeds** (banked at `n = 3`, labelled `n = 3`).
- **Two sub-clauses of the tuning standard** (`β = (0.9, 0.98)`, cosine decay) — declared deviations. **A re-run
  with held-out selection as the primary** (adopting it would be a post-hoc change of a registered rule).
- **The `overload` family at the base atom budget** (0/18 admissible, including the control arm). **A *trained*
  attention reader** (ours is a table reader with a grid-fitted scalar temperature). **A soft-certificate sweep
  over the violation budget** (one demonstration cell only).
- **A live-launch-momentum third-party probe** — the one mechanism that could in principle beat the
  `exp(−½(d/s)²)` suppression; its target is quantified rather than guessed: the suppression it would have to
  beat at the audited geometry is a factor **65** (`Δ(d²)/2s² = 4.17` in the exponent).
- **The `s` re-measurement under the `α‖q‖²` subtraction convention.** ⛔ `s = 0.40` is **flagged for a check,
  never reported as a confirmation and never as a null**; ⭐ its direction is known — smaller `s`, larger `d/s`,
  *stronger* suppression — so this NOT-RUN cannot be hiding a number that would help us. Also **a `d/s` sweep
  by varying the atom width** (it would move the write's expressivity at the same time) and **the eviction-path
  deletion as a robustness arm** (eviction *re-draws* the freed group, so its Δ would not be the item's).
- ⛔ **Any real-data VALUE leg for any memory — ours or a rival's — is a declared NOT-RUN for want of an
  admissible venue, never a null**: both gated venues fail the tripwire (§2.4, R.2), so no store build and no
  memory cell ran on either; both remain admissible as *mechanics* substrates. **A third real tabular streaming
  venue** was deliberately not shopped after the two firings — a registered stop, not a budget accident. **The
  INSECTS `out-of-control` stream** (the venue's own drift-free null) ⛔ has **no data source** and is a
  declared NOT-RUN; the drift-free-null role is filled on Metro by the order-destroying shuffle, which is a
  shuffle of the pair sequence, not a stationary data source. **`rls_ff` on Metro** diverged (RMSE 5.3e4 → 2.3e5)
  and is excluded and declared NOT-RUN rather than reported.
- **Any language-modelling leg** (not sized, not attempted). **Any change to a shipped default** anywhere.

## Appendix K — Figure specifications

> **Render status.** Figures 1–5 are rendered from banked artifacts (PNG at 200 dpi + PDF twin); every plotted
> value traces to a named field in a machine-readable figure → artifact → field → value provenance table
> (14 entries for Figure 1; 48 entries covering Figures 2–5). ⛔ **No cell was measured or re-measured for any
> render**; the Figure 1 renderer fails loudly rather than drawing if the bar count is not seven or any bar is
> not at `n = 9`. Two declared non-artifact quantities are logged in the table: Figure 4's attention
> normalisation (computed with the artifact's own published rule) and Figure 3's shading boundary
> (presentational).

**Figure 1 (headline).** Signed **+0 B raw-metric margin** per audited arm, horizontal bars, zero line drawn,
gate status hatched and labelled per bar — **NOT RESCUED** (`ttt_mlp` and, ⛔ **stated in the caption in the
same breath as the rival hatching**, the **CLU**), **INIT-UNSTABLE** (`ttt_linear`), **SELECTION-DEPENDENT**
(`deltanet`, `gdn2`), unhatched for the two arms rescued under all three selection rules (`gdn`, `mamba2`); the
CLU bar in a distinct fill so the authors' own arm is identifiable. Error bars = ±1 SE. ⭐ **`n = 9` on every
bar**, with family, byte budget, tuning grid and the scale qualifier in the caption. ⛔ No mixed-`n` language.

| bar | raw-metric `+0 B` margin | `n` | gate status (hatching) |
|---|---|---|---|
| TTT-Linear | −0.4602 ± 0.1038 | 9 | INIT-UNSTABLE |
| TTT-MLP | −0.4425 ± 0.0869 | 9 | NOT RESCUED |
| DeltaNet | −0.2732 ± 0.0395 | 9 | SELECTION-DEPENDENT |
| Gated DeltaNet | −0.2600 ± 0.0278 | 9 | rescued under all three rules — unhatched |
| Gated DeltaNet-2 | −0.2592 ± 0.0292 | 9 | SELECTION-DEPENDENT |
| Mamba-2 (SSD) | −0.2563 ± 0.0416 | 9 | rescued under all three rules — unhatched |
| **CLU (ours)** | **−0.2897 ± 0.0328** | 9 | **NOT RESCUED** (distinct fill) |

**Figure 2.** Two-sided byte ledger, stacked bars per arm: F1 parameters (learned-initial-state component
hatched) and F2 state, with each arm's own-table byte count as a tick; the CLU's 54.56× excursion on a broken
axis, labelled unreachable-by-construction (T1). ⚠ Bars are the **modal** ledger over nine seeds, each arm's
other per-seed configuration drawn as a cap (TTT arms: `b`; CLU: five or six items admitted); delta and SSD
rows are seed-constant. **Figure 3.** Third-party attribution: `κ` versus `d/s` on log-linear axes, **both
rulers plotted**, the fitted `exp(−½(d/s)²)` line with R² annotated, the per-slot table's exactly-zero drawn on
the axis, the audited cell marked at **`d/s_fit = 3.59`**, and the write-inadmissible region (`λ_min < 0` on
≥ 1 seed) shaded (right edge presentational, declared). **Figure 4 (appendix).** Protocol validation: `S(f)`
per family against the saturation threshold, substitute byte cost annotated per bar, full-attention reader
overlaid. **Figure 5 (appendix).** The CLU's accuracy-versus-bytes frontier curve (**banked at `n = 3`**,
reused and not re-measured) with the rival points **omitted** and a caption stating why: at nine seeds **0 of
20 (arm × head-width) cells clear their own blank-store control**, so the rival points would be a picture of
noise, and two of six arms were never run there. ⚠ The caption must state that swept arms differ in **write
load** (6 vs 17 live items), plotted as two series with only the 1× series connected.

## Appendix L — Negative results and refutations recorded by this work

Negatives are documented and never dropped.

1. **The audit's headline is negative for our own system**: dividend **−0.0561 ± 0.0315** against its own
   launder (**1.78 SE** — the launder *reads no worse than* the store), `+0 B` margin **−0.2897 ± 0.0328**
   (8.8 SE); an earlier round of the same substitute audit went **0-for-4** (insertion order 0.776 vs 0.302;
   echo 1.0000 vs −0.180).
2. ⛔ **Our store fails our own rescue gate at the same nine seeds as every rival arm**: full **−0.4370 ±
   0.0417** against blank **−0.3906 ± 0.0124**, lift **−0.0465 ± 0.0406** (|t| = 1.14) ⇒ statistically
   indistinguishable from an empty store, point estimate on the wrong side of zero. ⚠ It supersedes this work's
   earlier three-seed reading (a point lift of −0.104 stated as reading *below* blank): the sign is unchanged,
   the significance statement is not.
3. **Three of four designed task families are struck as protocol-invalid** — including two where the memory
   reads *below* a ≤ 4 B substitute, and one (`recency`) where it reads below its own blank store, **0.4769
   written against 0.5463 blank**, kept on record with the family's struck status attached.
4. **The registered byte-matched control for weight-valued memories is a weak control** — a refutation of our
   own pre-registered protocol text, costing the table 0.263–0.942 at nine seeds.
5. **The published closed form of the byte law is wrong in 4 of 28 cells** (conservatively).
6. **The registered disagreement-mass formula was refuted** and replaced. 7. **The registered `{M, γ}`
   sensitivity band was missed by 4.8×** — the excess is exactly the `O(N)` injection prefactor.
8. **The registered third-party decade span, slot-decay ratio and free-fall prefactor were all missed** — the
   ruler, not the law, was wrong. 9. **The registered frontier knee was wrong**, from our own miscount of
   stream tokens.
10. **Our own placement of this store on the `d/s` axis was wrong by 45–52×** — an admission gate's refusal
    radius read as an achieved spacing; the correction moves *against* the interesting direction.
11. **A designed near-degeneracy does not survive superposition on a learned store** (refuted in sign on two
    implementations). 12. **A soft-certificate budget previously quoted as located is not located at all.**
13. **`λ_min > 0` does not certify a nonempty basin** (capture radius 0.000 at `λ_min = +0.910`).
14. **Slot count buys no per-item capacity** (measured rank `3d+1`; `d` under the audited read).
15. **A prior attempt to learn an address by gradient descent through a settled-point read died at chance**,
    explained in Appendix D with zero fitted parameters.
16. **Our own rescue-gate verdicts were not reproducible at three seeds**: a pre-registered prediction that the
    five statuses would be stable under re-tuning was **refuted**, three flipped, and the gate's own power —
    not the tuning — was the cause. Every three-seed rescue verdict we had recorded is withdrawn.
17. **A registered prediction about our own tuning standard was refuted on its count** (predicted ≤ 2 of 15,
    observed 6 of 15, always by fourth-decimal tie-break) — which is how we discovered that the standard's
    selection rule makes its own regulariser axis unselectable.
18. **The initialisation-key scheme, not the tuning, dominated the difference between our two passes** (4–35×)
    — invisible without a control column we had to think to run.
19. **Both TTT arms read below their own same-keys null at nine seeds** (−0.2063 ± 0.1016, −0.1995 ± 0.0665) —
    a stronger statement of malfunction than failing the blank-store gate, on the rival side of the audit.
20. **The byte-frontier column resolves nothing at nine seeds**: 0 of 15 cells, best lift +0.0694 ± 0.0491 —
    **0 of 20** once the SSD arm is added (entry 23). Extra power did not rescue the column.
21. **The projected-control dividend magnitudes fell between our two passes** (gdn 1.02 → 0.60, −42 %; gdn2
    0.88 → 0.68, −22 %) while signs held on 5 of 5; the three-seed magnitudes are superseded and both printed.
22. ⛔ **Our own byte ledger is not seed-constant, and a registered prediction that it was is refuted**:
    `5456 B / 100 B / 54.56×` on eight seeds, `5472 B / 120 B / 45.60×` on the ninth ⇒ every ratio is labelled
    **modal (8 of 9)**. The integer identity is green on all nine — a labelling defect in how we published the
    number, not a defect in the store.
23. ⛔ **Rescue verdicts depend on the best-of-grid selection rule, not only on the seed count** (`deltanet`
    +0.077 ± 0.045, `gdn2` +0.669 ± 0.339 under held-out selection) ⇒ both printed SELECTION-DEPENDENT and the
    quotable rescued set shrinks to two arms. A second free choice in a protocol we ourselves specified.
24. **The byte-frontier column stayed a labelled null when a sixth arm was added** (0 of 15 → **0 of 20**).
25. **The SSD arm's `+0 B` own-table margin is a tie, not a win** (+0.0047 ± 0.0519, sign-flipping under
    held-out selection), and **no ordering among the four delta/SSD arms on `full` is quotable** (they span
    0.017 with SEs of 0.012–0.033).
26. ⛔ **The INSECTS venue fails the admissibility tripwire, and the anti-hobbling rule is what decided it**
    ⟦N294⟧: the strong-baseline frontier clears the byte-matched exemplar store by **1.90 points** where the
    registered rule required more — 0.10 points inside the threshold — and causal standardisation (worth +4.0
    to +6.3 points to the exemplar arms) is what put the store at the frontier's shoulder. Had the exemplar
    arms been run raw only, the tripwire would **not** have fired: an audit that hobbles its trivial baseline
    admits its own venue.
27. ⛔ **The Metro venue fails three admissibility checks at once** ⟦N295⟧: the byte-matched exemplar store
    beats every one of **nine** strong references by 4–25 % relative MAE; the strong frontier's headroom over
    a one-line weekly rule (`ŷ_j = y_{j−168}`, MAE 342.6513) is **2.17 %**; and destroying temporal order
    *helps* the exemplar store at and above the audited byte budgets (−1.9 % to −2.9 % MAE).
28. ⛔ **Our own pre-registration asserted plain prequential evaluation is leak-free at a 24-hour horizon, and
    that assertion was refuted by mechanism** ⟦N296⟧ — worth up to **+10.85 %** MAE to a 250-exemplar k-NN and
    **−0.34 %** to a periodically-refit GBDT; and seasonal-naive(`t − 24 h`) is **degenerate** (bit-identical
    to persistence, max |diff| = 0.0).
29. **A pre-registered no-daylight null at a third substrate, kept as the null it is** ⟦N276⟧: 0/9 cells, at a
    byte asymmetry of **2,364×** in the store's *favour*, attributable rather than dismissible.

## Appendix M — Reproducibility and artifact notes

Seed counts are stated per section and never mixed silently: all six rival dividend arms **and our own store**
are 9-seeded on every column; four of six arms on the byte-frontier column are 9-seeded and two were not run
there; the CLU's banked frontier curve, the protocol-validation run, the first pass's frontier rows and the
attribution sweep are 3-seeded. ⭐ The CLU's nine-seed column is a **re-aggregation of banked per-seed cells**
whose seeds 0–2 reproduce the previously published three-seed values digit-for-digit and whose shipped-rule
output an independent out-of-harness recomputation reproduces exactly (A.1e); the SSD arm's run reproduces all
five incumbent arms **bit-identically** (A.1f), which is what makes its row comparable to theirs. ⭐ **The
audit's first pass reproduces digit-for-digit from the second pass's branch at the base code**, so every
difference between passes is attributable to a declared change — the strongest reproducibility statement in the
paper, and what licenses the before/after table. The protocol-validation run reproduces its reference artifact
exactly on every shared arm, per seed; the attribution sweep's per-radius aggregates recompute to the last
digit on every 3/3 point; the full test suite is green at the recorded commit (1143 passed, 0 failed) with the
byte-ledger identity and the identical-encoder invariant asserted per cell. One reproducibility incident is
disclosed (§6 L12). ⭐ The folded measurements carry their own provenance (A.5–A.6) and their own seed
disciplines: the census cells are 3-seeded per arm with bit-identical streams across arms (every cross-arm
quantity paired); the INSECTS ARF reference is 3-seeded; **every exemplar arm on both streaming venues is a
single deterministic run, and no variance estimate is claimed where none exists.**

## Appendix N — The task families, defined

**N.0 What every family shares.** A stream offers `n_offer` items in order; each item is a pair
`(address ∈ R^4, payload ∈ R^1)`, the address drawn inside a unit ball (`ball_radius = 1.0`) subject to the
store's admission rule and the payload drawn from `[−1, +1]` on a level set spaced **0.4** apart. The store has
a slot `capacity` and an atom `budget`; a stream also carries a **deletion demand** naming a still-live item, a
**revisit** of an earlier address, a near-duplicate **collision offer** (where enabled), and at least five
**consolidation windows**. Queries launch from `q₀ = (query address, 0, …)` with isotropic jitter
`σ_q = 0.15`; payload tolerance 0.1. Every arm of a cell sees the identical encoder output, asserted by content
hash.

**N.1 `aggregate` (the sole dividend family; `S = 0.5068`; the audit's entire empirical basis).**
`n_offer = 8`, `capacity = budget = 6`, `consolidate_every = 2`, staged admission on, 192 atoms, no spectator
dimension. Pairs of stored items whose addresses lie within `1.7×` the stream's minimum separation are
enumerated; for each pair **8** queries are drawn with mixing weight `λ ~ U(0.35, 0.65)`, the query address is
`(1−λ)c_i + λc_j` plus jitter, and **the target is the convex combination of the two payloads** — which lies
*between* the two stored basins. **The construction rule that makes it work:** a query whose target lands
within the payload tolerance of any stored payload is **dropped at construction**, so the answer is provably
not a stored payload and an arg-min lookup has an error bounded below by a positive constant. Metric `neg_mae`
(max 0.0). Blank store **−0.4221**, strongest `+0 B` reader (2-NN mean/inverse-distance) **−0.2081**, full
attention **−0.2493**, saturation **0.5068** — ⚠ these four are the **protocol-validation run's, at three
seeds**; the same blank control measured on the audit cell at nine seeds reads **−0.3906 ± 0.0124**, and the
two are not the same estimate at the same `n`. We print both with their runs named. *Why it is not
metric-native by construction, and where that stops:* criterion 4 (*if the query lives in the stored keys'
metric space, a classical method is the ceiling*) holds against **arg-min** but not against
aggregation-augmented classical readers — which is exactly why **the family ships with the 2-NN mean as its own
strongest control, and that control is expected to win. It does.**

⚠ **What that concession does and does not give away — itemised here because that sentence is the sharpest one
a reader can quote against this paper** (pointed to from §4.2). That the 2-NN mean is this family's strongest
`+0 B` reader is a **design property**: the family is built so that the answer is provably not in the table,
which stops an arg-min lookup from being accidentally right and makes an aggregating reader the obvious
beneficiary. It is a statement about the *reader ranking*, and the audit's headline is not that ranking.
Everything the construction does **not** fix could have come out otherwise, and each was free to move:
**(a) the size and stability of the margins** — nothing in the construction sets how far a *fitted memory*
falls short of that reader; measured **−0.2563 … −0.4602** at nine seeds, every arm at least **4.4 SE** below
zero, the sign holding on **all six** rival arms under the full grid and, on the five incumbents, additionally
at 5× the outer budget and under a held-out selection rule (both three-seed re-selections, labelled as such). A
family that merely *favoured* the table would be compatible with margins inside noise; these are not.
**(b) the sign of the same arms against the weaker control** — against the projected control four of six arms
are comfortably **positive** (**+0.1515 … +0.6824**): this family is perfectly capable of paying test-time
dynamics a dividend, and does, against the control the field would naturally build; which control is run
decides the *sign*, and that could not have been manufactured by the construction because both controls were
registered before measurement. **(c) the rescue-gate outcomes, including our own** — all six rival arms were
free to clear the gate under every rule; two did. Our own store was free to clear it and **did not**: a design
that foreordained the headline would not also have disqualified the authors' arm at the same nine seeds.
**(d) the rivals' behaviour against their own projected tables** — all six were free to fail even that weaker
bar; **four of six** cleared it, which is what makes the raw-table result a statement about the *control*
rather than about broken arms. **(e) the direction the registered stresses moved things** — the widened grid,
the 5× re-check, the held-out rule and the six added seeds were all registered as things that could change the
paper; none moved the headline, the verdict change that matters ran **against** the flattering direction, and
the one change that ran our way is a count on a column the headline does not use, printed both ways rather than
picked. **(f) the pre-registration scorecard itself** — six predictions confirmed, four partial, **three
wrong-direction**, two NOT-RUN, and one prediction about this very family's rescue stability **refuted**; the
misses are printed rather than re-scored. ⇒ The family was built so that a reader-discrimination question is
**askable**; it was not built so that the audit's answer is fixed in advance. The claim made from it is the one
in §4.2 and no larger, and §6 L1 states the thinness of resting an audit on one surviving family in the paper's
own voice.

**N.2 `overload` (retained only as the byte-frontier column).** `load1x_shipped` sets
`n_offer = capacity = budget = 6` with `atoms_per_item = 341` (2046 atoms), `consolidate_every = 4`,
`n_query_per_item = 4`, no collision offer, admission override `d_safe_override = 0.58`. Every **offered** item
is queried, live or not; the target is that item's own stored payload; scored by `decode`, a **six-way** choice
(chance 0.1667, max 1.0), 24 queries. The byte-matched table is never budget-limited here, so the family fails
the dividend criterion in advance and is **declared** a frontier instrument rather than discovered to be one:
its table launder sits at the metric's exact maximum (**1.0000**, 3/3 seeds), saturation **1.0000** (strict) or
**0.6500** (declared secondary, arg-min excluded). ⛔ At the family's *base* atom budget the cell is **0 of 18
admissible including the control arm** — which is why the shipped anchor is the one measured, and why §2.4's
anchor objection exists.

**N.3 `recency` (struck).** `n_offer = 8`, `capacity = budget = 6`, per-item lifetimes on (`leak = 0.06`); the
query asks which of two named items was written more recently (chance 0.5, accuracy). ⛔ Struck as
protocol-invalid: a `+0 B` reader of the table's own **row order** answers it exactly (**1.0000 on 3/3
seeds**), because a `(key, payload)` table already encodes insertion order. Its blank store reads **0.5463**,
above the memory's own **0.4769** — the earlier instance of the below-blank pattern §4.1 reports on
`aggregate`. **N.4 `manifold` (struck).** `n_offer = capacity = budget = 6`, **one spectator dimension** the
write objective never constrains, a 12-point launch grid spanning `±0.6` along it, read-out the settled
spectator coordinate scored by `R²`. ⛔ Struck: an **echo** reader returning the launch coordinate scores
**1.0000** at `+0 B` while full attention reads **0.0000** (the table's spectator column is written zero); the
blocker was named in advance — the write digs point wells, not valleys.

**N.5 The four rules a replacement family must satisfy.** The answer is not recoverable from the table's **row
order** (kills `recency`); it is not the query itself or a function of it alone (kills `manifold`); the store's
operating point is not one where the arg-min table sits at the metric's exact maximum (kills
`overload@load1x_shipped`); and — the rule that subsumes the other three — **the target is constructed to be
absent from the table.** These are necessary conditions for a family to discriminate *readers*; they say
nothing about whether any particular memory then wins.

## Appendix O — The rival arms, specified

Every rival arm is our own minimal faithful implementation of the published *update rule*, sized to the
protocol's iso-state budget, with the equations we implemented named so a reader can check the faithfulness
argument rather than take it. **O.1 The iso-state budget and head widths.** The budget is the CLU's own
audited-cell state, **1364 float32 = 5456 B**; each arm's head width is the largest value whose declared
per-sequence state fits under it, computed and registered before any run and asserted in the test suite
(29 / 12 / 36; per-arm state and table bytes in B.2). **O.2 The arms.** TTT-Linear and TTT-MLP (inner optimiser
over the stream, learned sequence-shared `W₀`, mini-batch `b` inside the declared state); DeltaNet, Gated
DeltaNet and Gated DeltaNet-2 (key-conditioned erase–write recurrence; GDN adds a scalar decay `α_t`, GDN-2
decouples channel-wise erase `b_t` from channel-wise write `w_t`); **O.2b** Mamba-2 (SSD): `h_t = a_t h_{t−1} +
B_t(Δ_t v_t)ᵀ` read as `o_q = h_Tᵀ C_q`, implemented and **asserted as an identity in a test** rather than
inherited from any paper's prose. ⭐ Restoring its block-level parts (`D` skip, `z` gate) cuts fit-split loss
**36 %** (0.2684 → 0.1721) and moves its eval read **worse** by 0.195 (≈ 2.3 SE), leaving the minimal
configuration the arm's best configuration on the audited metric — the same fit-to-eval gap the TTT-MLP budget
check finds, measured on a second family. ⚠ Key-norm assertion: GDN-2's key norms are 1.000 ± 1e-3, Mamba-2's
have sd > 1e-3.

**O.3 Metric-nativeness, per arm** (evidence; nine seeds, full grid). ⚠ The right-hand column is each arm's
**own loss to its own byte-matched table**, which B.5's direction rule leaves quotable for every arm; the gate
verdict is printed beside it, and the quantity the gate does suppress — a *comparative* margin of one arm over
a non-rescued one — appears nowhere in this paper.

| arm | verdict | equation-level argument | measured vs the raw table |
|---|---|---|---|
| DeltaNet | metric-native | `o = Sᵀq` with `S` a sum of outer products ⇒ `o = Σ_s z_s(k_s·q)`, a linear kernel smoother; `q, k` L2-normalised ⇒ `argmin‖q−k‖ ≡ argmax q·k` **exactly**; the only non-metric ingredient is the scalar `β_t` | loses by **0.2732 ± 0.0395** (⚠ SELECTION-DEPENDENT) |
| Gated DeltaNet | metric-native | adds a scalar decay `α_t` — a scalar reweighting | loses by **0.2600 ± 0.0278** (✅ rescued) |
| Gated DeltaNet-2 | metric-native | channel-wise erase/write ⇒ a learned diagonal, token-dependent Mahalanobis shape; still a metric, and the table it is audited against is entitled to the same shape | loses by **0.2592 ± 0.0292** (⚠ SELECTION-DEPENDENT) |
| TTT-Linear | metric-native | with gradients at `W₀` the read is `W₀q − 2η Σ_s(W₀k_s − v_s)(k_s·q)`; its own Theorem 2 makes the nonparametric TTT learner a Nadaraya–Watson estimator | loses by **0.4602 ± 0.1038**; ⚠ INIT-UNSTABLE ⇒ no comparative margin over it is quoted |
| TTT-MLP | **weakly** metric-native | the GELU means the read is *not* a kernel average of stored values, so metric-nativeness does **not** close at equation level — the only arm for which it does not | loses by **0.4425 ± 0.0869**; ⛔ NOT RESCUED ⇒ no comparative margin over it is quoted |
| Mamba-2 (SSD) | metric-native (**unnormalised**) | `o = Σ_j γ_j (C_q·B_j) Δ_j v_j`: a dot-product kernel smoother with exponential recency weighting ⇒ criterion 4 closes in the same sense as DeltaNet's. ⚠ Unlike GDN-2 it does **not** L2-normalise `B`/`C`, so `argmin‖q−k‖` and `argmax q·k` do **not** coincide — the key-norm term survives | loses by **0.2563 ± 0.0416** (6.2 SE; negative 9/9 seeds; ✅ rescued) |

## Appendix P — The store under audit

**P.1 The object.** An **atom dictionary** potential
`V_θ(q) = α‖q‖² − Σ_{j=1..N_at} A_j exp(−‖q − c_j‖²/2s_j²)` over a `D`-dimensional latent space
(`D = d + m + n_spectator`; the audited cell has `d = 4`, `m = 1`, no spectator dimension), learnable per-atom
`(c_j, log s_j, amp_j)` — seven floats per atom in the audited geometry — plus a live-address codebook of `K·d`
floats. Atoms are partitioned **one group per item slot** and a write is masked to its own group. Audited cell:
`N_at = 192`, `capacity = budget = 6`, confinement `α = 0.05`, atom width 0.3, `ball_radius = 1.0`.
**P.2 The write.** Staged admission (an offer is admitted only if its address clears the store's own separation
gate); admitted items are written by a masked local optimisation (300 Adam steps at `3e-3`, weight decay
`1e-4`, address noise `σ_addr = 0.25`, payload noise `σ_pay = 0.6`, hinge margin 0.15, barrier 0.2), touching
only the item's own atom group. Eviction is depth-ordered and **re-draws** the freed group; deletion, where
measured, is **amplitude-zeroing** of the item's atom group (exact removal, nothing else moved) rather than the
eviction path. **P.3 The read.** A query is launched at `q₀` with zero momentum and integrated by a dissipative
velocity-Verlet map with `dt = 0.05` in two phases — 400 steps at `γ_address = 0.05`, then 800 at
`γ_read = 0.02` — under a learned-Newtonian kinetic term; the answer is read at the settled point (the
trajectory read-out is used only where §4.6 says so). Temperature is zero: deterministic, no Langevin step. The
schedule's convergence budget is `C = 18.34` (Appendix D), which is what makes the read's transient parameters
exponentially insensitive. **P.4 The ledger.** The declared state is the **measured** per-stream deviation of
`V_θ` (1300 floats = 5200 B) against an initialisation of 1344 floats = 5376 B; the matched-byte launder is the
table of `K` live `(address, payload)` rows = 100 B — a ratio of **54.56×**, asserted as an integer identity per
cell. ⛔ That ratio is **unreachable-by-construction, not a budget choice** (§3.1), and ⛔ it is the **modal**
ratio (8 of 9 seeds): on the ninth the admission gate admits a sixth item, the launder becomes 120 B and the
ratio **45.60×**. The identity is green on all nine seeds; **no single figure is *the* nine-seed ledger.**

## Appendix Q — Citation notes and quote scopes

⚠ This is **not** the reference list (assembled at the typesetting pass); it carries the entries whose venue,
pagination, identifier and — where quoted — **wording** were checked against a published record or primary
artifact, with the caveats that must travel with each. **(a) Learned data structures.** Mitzenmacher (NeurIPS
2018) is quoted for the matched-space *condition*; Kipf et al. (2019, non-archival workshop) and Marcus et al.
(PVLDB 2020) for space **reported beside** speed. ⚠ Later SOSD-derived work (Chesetti & Pandey, ACDA 2025;
quotes pinned to the arXiv v2 §6.6, the camera-ready being paywalled and unread by us) finds RadixSpline and
RMI *"an order of magnitude (30 − 80×) larger than B-trees"* and *"at least 4 orders-of-magnitude more time to
build"* — ⚠ but the same paragraph reports the PGM index as **4× smaller** than the B-tree, the B-tree baseline
is deliberately sparsified (*"we only build the B-tree by uniformly sampling every 256th key"*), and that
paper's own headline is a rough **tie** in external memory; ⛔ the benchmark suite itself reaches the opposite
conclusion on size (RMI 3 %, RadixSpline < 1 % against a B-tree's 16 %). We cite the adversarial study for the
**tone of the accounting**, never for a verdict against learned structures. **(b) Partial-input baselines.**
Poliak et al. (*SEM 2018, anthology S18-2023, pp. 180–191) *"significantly outperform"* the majority-class
baseline on **six of ten** NLI datasets; Feng, Wallace & Boyd-Graber (ACL 2019, pp. 5533–5538) supply the
converse caveat — this draft **paraphrases** rather than quotes, and any later revision that quotes must use
the published wording, which differs from the preprint's. **(c) Family sources.** TTT's `b = 16` *"for all
experiments in this paper"* and its Theorem 2; Gated DeltaNet's own presentation of Mamba-2 *"up to specific
parameterization"* (⚠ hedge carried, §5); Mamba-2 (Dao & Gu, ICML 2024, PMLR 235:10041–10071). **(d) Deletion
neighbour.** arXiv:2603.15033 — a 2026 ICLR *workshop* paper; ⚠ we name neither the workshop nor the
presentation type, because the venue listing and the authors' page disagree and the arXiv record carries no
venue. **(e) Streaming-venue sources** (Appendix R.2): the INSECTS benchmark and its published anchors; `river`
0.25.0 cited both as the artifact run (software `@misc`, version-pinned) and as its paper of record; the UCI
Metro dataset record; Webb et al. (2016) for the drift-magnitude framework — ⚠ **attribution split: the
framework is theirs, the total-variation instantiation is ours** (they deliberately leave the distribution
distance unspecified and use Hellinger in their own case study). ⛔ Two never-copy warnings recorded here
because they have bitten: `river` ships **no** SAM-kNN (our port; every *"one-line baseline"* cost estimate is
void), and the SAM-kNN STM cap is a reference-implementation default, not a published number.

## Appendix R — Protocol evidence folded from outside this audit's rig

> **What this appendix is.** Three measurements made on other rigs of this research program, folded in because
> each is evidence about this paper's protocol or thesis rather than a new cell of the audit; each carries its
> own flag-provenance table (A.5–A.6). ⛔ **Standing scope for the whole appendix:** nothing here is a benchmark
> result for any memory — the two streaming-venue runs are **baseline-only** (no CLU cell and no rival-memory
> cell of any kind was run on either venue) and the third-substrate row is a component-build measurement, not a
> verdict. No number from this appendix enters any audit column, and **no external benchmark is claimed as won
> on its own headline metric — here or anywhere in this paper.**

**R.1 The no-daylight row at a third substrate.** The registered form is quoted in §4.8. **Per-arm numbers.**
Cue-side margin A3a (store − table, McNemar SE), mean ± SE per arm: fitted contrastive encoder
**−0.2526 ± 0.0456** · unfitted random-convolution control **−0.1432 ± 0.0409** · PCA reference
**−0.1224 ± 0.0365** — on the cue side the table is *ahead* on every arm, the metric-native ceiling doing what
the theorem says. Stream-side margin A3b (pooled binomial SE): **−0.1042 ± 0.0497 · +0.0156 ± 0.0239 ·
−0.0260 ± 0.0104** — statistically indistinguishable from zero (|A3b| ≤ 0.047 on 7 of 9 cells; 8 of 9 inside
2 SE ≈ 0.175; the exception is the fitted arm's seed 2 at −0.2031 ± 0.1729, on the *table's* side). Under the
registered rule the count is **0 of 9 cells, 0/3 seeds on every arm, both legs**. Visible in the raw stream
reads: the control arm's seed 0 gives store **[0.6875, 0.5000, 0.2500, 0.1875]** against table
**[0.6875, 0.5000, 0.3125, 0.2500]** — identical on the first two read events. **Attribution, verified in
advance rather than reconstructed:** well-depth medians **0.94 / 0.89 / 0.73** (fitted), **0.97 / 0.77 / 0.86**
(control), **0.75 / 0.47 / 0.60** (PCA) — 9/9 cells not inert; the address geometry cleared its registered GO
rule before the run (σ_q/spacing **0.334** at the PCA reference, **0.210** at the fitted encoder, `d = 12`, 3/3
seeds — queries' own jitter over stored-key spacing, so smaller is safer and both sit far below 1); and the
table is audited **in code** (it re-derives its keys from the projected encoder output and raises unless
bit-identical, 9/9 cells). **Riders** (each also in §4.8): matched-items, not matched-bytes, with the byte
asymmetry **2,364×** in the store's favour; the metric-native ceiling was expected in advance (hence the 0.70
prior) and the measured content is the attributable **absence of daylight**; a component-build measurement — no
tier-level, performance, accuracy or benchmark claim, and no adjudication between encoder arms. ⚠ Instrument
label: the census rig's pass-3 instrument, before that rig's later hardening; numbers stay labelled with the
instrument that produced them.

**R.2 The two real-streaming-venue admissibility firings.** **The rule being applied:** the same discipline as
§2.4, transported to an external venue — before any store build, ask whether a **trivial, byte-matched exemplar
store** (a kNN window over raw stored pairs, the streaming analogue of this paper's table launder) already sits
at the venue's strong-baseline frontier. If it does, the venue cannot separate a memory's dynamics from its
stored content at that budget, and no value claim for *any* memory is hosted there. The approved form of the
finding, and the ruled framing that travels with it verbatim — **"a fallback being retired, not a venue
crisis"** — are quoted in §2.4.

**R.2.1 INSECTS** (`incremental-reoccurring` primary; `incremental-abrupt-reoccurring` second condition). The
harness reproduces the venue's published anchors before measuring anything: No-Change **40.4526** vs published
**40.46** (Δ −0.0074, a pure function of the label sequence) and ARF-100 **78.8139 ± 0.0526 SD** (3 seeds) vs
published **77.13** — ⚠ two different implementations (`river` vs **MOA**; ensemble size never stated by the
benchmark's authors, so "ARF-100" is our inference), an anchor check across implementations rather than a
like-for-like replication, and nothing in this section rests on the two being the same object. The best
registered exemplar arm (SAM-kNN at its published 0.634 MiB budget: **76.9157 %** at 665,000 B, causally
standardised) lands **1.90 points** below ARF-100 where the pre-registered rule required more — the margin is
**0.10 points inside the threshold**, so the venue fails; it fails against **every** ARF reference computed
(−1.898 3-seed mean · −1.937 best seed · −1.838 worst seed · −0.497 our ARF-10 · −0.214 the published 77.13),
and the second condition fails at −1.845. An unregistered *stronger* exemplar arm (SAM-kNN, `L_max = 1000`,
**77.0632 %** at 133,000 B) sits within **0.07 points** of the venue's **best published method** (⚠ best of the
six methods the benchmark's own authors ran — not a literature-wide state-of-the-art claim). ⚠ The honest
sentence is ***"the byte-matched exemplar store is at ARF's shoulder"*, never *"it beats ARF"***, and ⚠ ARF is
**byte-matched to nothing** (**9,542,925 B = 14.35×** SAM-kNN's 665,000 B). **Mechanism, measured:** exemplar
accuracy is *monotone decreasing* in store size above `L ≈ 500` (**75.56 / 76.03 / 75.36 / 73.39 / 68.15 /
59.75 %** at `L = 250 / 500 / 1,000 / 2,000 / 5,000 / 14,782`), so the store byte-matched to our own planned
build (14,782 exemplars = 1,966,006 B) is the **worst** exemplar arm — 19 points below a store 30× smaller.
Recency is the hidden regime variable (a 500-example window spans ≈ 0.13 °C of the venue's 20 °C sweep);
SAM-kNN discovers this itself (its short-term memory averages 945 of a **3,000-exemplar STM cap** — ⚠ a
reference-implementation default, not a published number — and its long-term memory is selected on 17.7 % of
instances), and the strong reference is a recency mechanism too (78 of ARF's 100 trees hold exactly one node at
the end of the pass). On INSECTS the winning strategy is *forget fast* — the exact inverse of a persistence
venue. **The rule that decided it:** the frozen CSV is unnormalised, so every exemplar arm ran raw **and**
causally standardised with the tripwire consuming the max (a declared anti-hobbling rule); standardisation is
worth **+4.0 to +6.3** points (SAM-kNN 5000: 71.52 raw → 76.92 std). ⛔ Had the exemplar arms been run raw only,
the venue would have **passed** — an audit that hobbles its trivial baseline admits its own venue; this is
§2.3's lesson reproduced on a real venue. **Caveats carried:** our SAM-kNN is our own port (validated against
the authors' published Weather row and by exact brute-force agreement); the exemplar arms are single
deterministic runs; our ARF is stronger than the published one (+1.68), the direction that makes firing harder;
the standardiser is the analyst's declared, pre-registered choice. ⛔ **The venue's terminal persistence band is
at ceiling and persistence-trivial, and no per-band retention or acquisition number is quoted from it here or
anywhere.** ⛔ The venue's own drift-free-null stream has no data source and is a declared NOT-RUN (Appendix J).

**R.2.2 Metro Interstate Traffic Volume** (hourly, 24-h horizon, hidden clock, label embargo on). Not a
threshold call — the venue fails **three** independent checks. **(1) The exemplar store beats every strong
reference outright.** The best exemplar at SAM-kNN's published budget (k-NN over past windows, `L = 5,037` =
664,884 B; the declared anti-hobbling k-ladder arm at `k = 10`, raw) reads **MAE 314.575** against the best
strong reference `gbdt_tuned` at **335.203** — a −6.15 % relative margin — and the **registered `k = 5`** arm at
the same budget (**320.982**) fires at −4.24 % on its own. It fires against all **nine** strong references by
4–25 % (`gbdt_tuned` −0.0615 · `gbdt` −0.0710 · `gbdt_cat` −0.0721 · `gbdt_recent` −0.0916 · `rls` −0.1604 ·
`gru_big` −0.1605 · `ridge_batch` −0.2119 · `mlp` −0.2419 · `gru` −0.2969). Even a 133 kB store fires
(−2.18 %), and the parity reproduces outside the streaming harness entirely: on a single 70/30 chronological
static holdout, k-NN (`k = 25`) at 265.47 edges tuned GBDT at 266.33. ⚠ The tuned GBDT's own measured state is
3,618,071 B = **5.44×** the exemplar budget it loses to; ⚠ the strong baselines are single-seeded and the
exemplar arms deterministic — no variance estimate is claimed, and none is needed at 4–25 % margins. **(2) The
strong frontier's headroom over one line of code is 2.17 %.** The headroom rule passes against persistence
(574.135 MAE, headroom 41.6 %), but the non-degenerate trivial rule is the weekly seasonal-naive
`ŷ_j = y_{j−168}` at **342.6513**, and the best strong baseline beats *it* by only **2.17 %** relative MAE.
Both readings are in the gate artifact; neither is hidden. **(3) Destroying the stream's temporal order helps
the store.** A fixed-seed uniform permutation of the pair sequence (`P(X, y)`, `P(y|X)` preserved exactly;
positive control exact) *improves* the exemplar store at and above the audited budgets: `L = 5,037`
**320.982 → 311.748** (−2.88 %); `L = 14,894` (the byte budget a store build on this rig would occupy;
**registered `k = 5`**) **306.762 → 301.068** (−1.86 %); `L = 34,847` 304.016 → 297.390 (−2.18 %). It hurts only
*below* the budget (`L = 1,007` +4.71 %, `L = 250` +2.69 %; crossover at `L ≈ 2,000–5,000`). ⇒ *at the byte
budgets this program would use, Metro's temporal ordering carries no exploitable information — there is nothing
there for a memory to be good at.* ⚠ The shuffle is a shuffle, **not a drift-free data source**.
**Mechanism — the exact inverse of INSECTS':** the exemplar ladder is monotone *improving* across its whole
range (MAE **407.91 / 347.89 / 327.79 / 325.22 / 320.98 / 307.08 / 306.76 / 304.02** at
`L = 250 / 500 / 1,000 / 2,000 / 5,037 / 10,000 / 14,894 / 34,847`), because the regime clock is fully encoded
inside each pair's own 24-lag feature window, so a stored exemplar never goes stale; and raw beats
causally-standardised everywhere here (320.98 vs 342.75 at `L = 5,037`) — the anti-hobbling max doing its job in
both directions across the two venues. A drift map built for the venue (drift magnitude in the sense of Webb et
al. (2016, Eq. 6) with the distribution distance instantiated as **total variation** — ⚠ their framework, our
instantiation; 1,101 revisit rows; the purely data-driven day map recovers the weekday/weekend split with no
calendar input) confirms the firing is not a favourable slice: the exemplar-vs-strong margin is negative in
**every band of every map** (−4.7 % … −17.9 %). **Caveats carried:** the 32-feature vector (132 B/exemplar,
identical for every arm) is the analyst's declared choice and the k-NN attack is defined by it — mitigated by
giving the strong baselines exactly the same features and by the static-holdout reproduction; feature gap-fill
≤ 3 h, targets never imputed; a 7,386-hour sensor hole makes the stream two eras; our `knnsam` regression
adaptation is ours, not a published algorithm, and it loses to the plain window anyway (325.71 vs 320.98).

**What follows, and what does not.** ⚠ **INSECTS remains fully admissible as a *mechanics* substrate — that half
is a registered ruling of ours** (criterion 4 gates the *value* venue, not a mechanics build). **We make the
same reading of Metro, and that half is our own inference rather than a registered ruling**: nothing measured
here disqualifies it as a mechanics substrate, and nothing here certifies it as one either. What neither venue
can host at these budgets is a value claim that any memory's dynamics beat its stored content — which is why
every real-data value leg of this program is a **declared NOT-RUN for want of an admissible venue, never a
null** (Appendix J). The framing is the ruled one: **a fallback being retired, not a venue crisis.** And the
standing practice the two firings earn is the cheap one: **run the venue tripwire first** — discovering
inadmissibility after a build is the expensive ordering, and it is free to avoid. (⚠ A proposed companion
practice — publishing the 20-second order-destroying shuffle on every candidate value venue before any build —
is a recommendation on our records, not an adopted rule.)

**R.3 The 24-hour label embargo — a benchmarking-methodology finding.** The approved form is quoted in §2.4.
**Mechanism, stated exactly.** Our own pre-registration asserted the ordinary test-then-train protocol was
leak-free at a 24-hour horizon, *"because pair `j`'s features stop at `j − 24` and its label is revealed only
after prediction."* That assertion is wrong by mechanism: pair `t − 1`'s **label** is the traffic volume at
target time `j − 1` — **23 hours after pair `t`'s forecast origin `j − 24`** — so plain test-then-train hands a
continuously-updated learner up to 23 hours of future labels; and the leak is asymmetric — a k-NN store that
admits every new pair immediately gains from it, a GBDT refit every 720 pairs essentially does not. **The fix,
implemented and used for every primary Metro number:** `A(t)` = the index of the last pair whose *target* time
≤ pair `t`'s *origin* time (`searchsorted(tgt, tgt − 24, 'right') − 1`; median `t − A(t)` = 24, mean 22.86, max
24); every store admission, fit and online update is restricted to indices ≤ `A(t)`. **Measured deltas (leaky →
embargoed, from the 45-arm delta table).** ⚠ **Convention:** each percentage is `(embargoed − leaky)/embargoed`,
taken **relative to the embargoed (conservative) value**, so a **positive** number means plain prequential
evaluation flattered that arm by that much and the embargo took it back. k-NN `L = 250` std **398.17 → 446.64 =
+10.85 %** · k-NN `L = 250` raw 373.00 → 407.90 = +8.56 % · k-NN `L = 1,007` std 344.28 → 363.59 = +5.31 % ·
k-NN `L = 5,037` raw 312.06 → 320.98 = +2.78 % · k-NN `L = 14,894` raw 300.42 → 306.76 = +2.07 % · **`gbdt`
339.80 → 338.65 = −0.34 %** · `mlp` 420.27 → 414.97 = −1.28 %. ⭐ The venue verdict is unchanged by the fix
(within-protocol margin −0.0946 leaky, **−0.0615** embargoed) — the analyst closed a leak biased **toward**
firing the analyst's own tripwire, and R.2.2's numbers are the embargoed, conservative ones. **The companion
defect in the same registration:** seasonal-naive(`t − 24 h`) is degenerate at a 24-hour horizon — bit-identical
to persistence, measured max |diff| = **0.0** (predicted in advance and confirmed exactly); the non-degenerate
trivial rule is the weekly `t − 168 h` naive (MAE 342.6513), which is the one that decides R.2.2's second check.
**Scope, verbatim from the registration:** measured on one stream (Metro, 34,848 hourly pairs, `h` = 24) with
one feature construction; **the mechanism is protocol-general** for any `h`-step-ahead stream evaluated
test-then-train with a continuously-updated learner, **the magnitudes are not**. Two further harness defects
were declared beside it (the ±10 SD standardisation clip; the diverging `rls_ff` arm, excluded and declared
NOT-RUN). **Why it is in this paper:** it is a third measured finding about evaluation protocol in the same
species as §2.1's and §2.5's — a control everyone runs is not neutral, and its bias points in the flattering
direction for exactly the memory-like arm. Anyone auditing a bounded-state memory on an `h`-step-ahead stream
should embargo labels to the forecast origin and check their seasonal-naive is not persistence in disguise.

---

## ⛔ Open editorial items (delete before circulation)

1. **Title** is `[WORKING TITLE: …]` and **authorship** is `[AUTHORS PLACEHOLDER]`; both workshopped at the end,
   both blank in an anonymized build. Metadata scrub, no acknowledgments/funding section, third-person
   self-citation throughout.
2. **Length.** Main text (title → §7) is **≈ 11.0 k words including table cells**; see the CHANGELOG entry for
   the page-count estimate, the assumption behind it and the ranked next-cut ledger. ⚠ **A further trim pass is
   owed before submission** and its targets are named there rather than taken unilaterally here.
3. **Figures 1–5 are reused renders** (Appendix K); Figures 3 and 5 need their captions re-checked against the
   specs, and Figure 2's target render includes the SSD bar. ⛔ No figure was re-rendered for this condensation.
4. **Naming continuity** — this draft uses "CLU" with the continuity sentence *"the CLU, introduced as CHLU in
   Jawahar & Pierini (2026)"* in the abstract and §1. Which paper carries the name's debut is a Hub/Head call.


## Appendix S — Main-text overflow (full statements, moved verbatim at condensation)

> Nothing was deleted in the condensation to this venue's page limit; every passage moved out of the main text is reproduced here verbatim, in main-text order, so that a reviewer who follows a pointer finds the full statement rather than a summary of it.

### S.1 — §3 verification prose and the byte-floor erratum, in full

**Verification:** in exact integer/rational arithmetic over 28 recorded ledger cells the byte decomposition is
exact **28/28** and the identity reproduces the measured ledger ratio **28/28 at 0 ulp**; a shell-atom basis
*raises* the floor by exactly `1/(D+2)`, so a basis change is not a route to matched bytes. ⚠ **Erratum, ours:**
the closed form in our own pre-registration (`1.4A + 0.8`, *"verified to 1e-9 in all 28 cells"*) is **wrong in 4
of 28 cells** — every cell with a spectator dimension (52.00 measured against a published 43.33; printed floor
2.00× where the true floor is 2.40×). The corrected law is exact in all 28, and the error was **conservative**:
it understated ratio and floor alike, so no claim built on it was inflated. **Domain:** exactly this store
family, not an MLP or Hopfield store and not a shared or factored substrate. **The trade, quantified:** with `S`
items sharing each non-private atom, `ratio(S) = A_tot(D+2)/[S(d+m)] + d/(d+m)`, so matched bytes needs
`S* = A_tot(D+2)/m` (**2387** at the shipped anchor `A_tot = 341`), and at `r = 1` **at most `p ≤ 4.19e-4`** —
0.042 % of an item's parameter mass — could remain byte-exactly deletable, at exchange rate `dp/dr = 2.10e-3`.
This bounds **byte** exactness only, not behavioural unlearning metrics. A third verification result — a
settled-point read is **untrainable end-to-end in both directions**, since `Fix(T_θ) = {(q,0) : ∇V_θ(q) = 0}`
contains no transient parameter and the finite-budget remnant is `‖∂z_N/∂ζ‖ ≍ K_ζ e^{−C}` — is in Appendix D.

### S.2 — §4.3 falsifier adjudication and ledger prose, in full

**"The finding inverts" — does NOT fire in the strong form; DOES fire in the weak form, which we state plainly
rather than re-frame.** Against the **arg-min** control read through each memory's own projections the dividend
at nine seeds is **DeltaNet +0.1515 ± 0.0600 · GDN +0.5960 ± 0.0933 · GDN-2 +0.6824 ± 0.0756 · Mamba-2
+0.3575 ± 0.1451** — four positive by more than 2 SE — while both TTT arms are negative (−0.1840 ± 0.1069,
−0.1794 ± 0.0748) and **the CLU is −0.0561 ± 0.0315** (1.78 SE, a sign statement). In that reading **test-time
dynamics pays for the delta-rule and SSD arms and does not pay for ours**: true as measured, unsoftened.
**Strong form: 0 of 6 rival arms beat the raw-metric +0 B table at the same bytes, and neither does the CLU** —
and the audit is not a different paper **only because the distinction that decides it was registered before
measurement**; we regard the ordering, not the outcome, as the credible part. **"Not apples-to-apples" — does
NOT fire**, with a split we never blur: `n_rows = ⌊state_floats/(d_k+d_v)⌋` is *forced* for the six arms
adjudicated **by measurement** across three state types, while ⚠ **Sparse Delta Memory and Titans are
adjudicated from published equations only, never measured**. ⭐ **Metric-nativeness** (per-arm equations,
verdicts and losses: Appendix O.3): all three delta-rule arms are metric-native, TTT-Linear is Nadaraya–Watson
by its own paper's Theorem 2 (⚠ INIT-UNSTABLE ⇒ no comparative margin over it is quoted), Mamba-2 is
metric-native **unnormalised** (it does not L2-normalise its `B`/`C` paths, so `argmin‖q−k‖` and `argmax q·k` do
not coincide), and **TTT-MLP is only *weakly* metric-native** — the one arm where the equation-level argument
does not close, because its GELU means the read is not a kernel average (⛔ NOT RESCUED ⇒ no comparative margin
over it is quoted). ⭐ **Every rival family we measured is metric-native or weakly so** — the matched-bytes
ceiling is a property of the family, not our idiosyncratic problem, and that line belongs to the field rather
than to us.

**The two-sided ledger, and the asymmetry that is itself a finding** (full table: Appendix B.2). Every rival's
declared state sits at a state/table ratio inside **[1.0000, 1.0278]** on every seed and every selected
configuration — ⭐ **every rival's state *can* be byte-matched to its own table; ours provably cannot.** T1's
floor makes matched bytes unreachable under a per-item group-masked write and the audited cell sits at
**54.56×** (⚠ **modal, 8 of 9 seeds**; **45.60×** on the seed where a sixth item is admitted). This is the
sharpest single statement the ledger produces, it runs *against* our own system, and it is why every byte or
dividend claim here carries the **≥ 2.20× (≥ 2.40× with a spectator dimension)** caveat. ⚠ **Params are not
matched** and no arm here is param-matched (the SSD arm's F1, 8380 B, is *lower* than the delta arms' 9956 B —
an asymmetry in the rival's favour). The ledger is enforced structurally — `full == 4[N_at(D+2) + K·d]`,
`launder == 4K(d+m)`, asserted as integers per cell, a drifted store raises. ⛔ **No cell measured in this audit
is a byte-matched dividend; the minimum ratio measured anywhere in this work is 17.11×.**

### S.3 — §4.4 deletion section, in full

## 4.4 Deletion, in the "and also" position

⚠ **Said first, by us: a table deletes exactly by construction.** Byte-exact deletion is a result only for a
*learned or superposed* store, where the item's contribution is not a row one can drop — the entire reason this
column is reported and the entire reason it is not a headline. **Frozen result, with instrument and conditions
in the same sentence:** deleting an item leaves the store **byte-equal to the never-written counterfactual on
3072 of 3072 compared bytes**, and a membership-inference attack against the deleted item reads **AUROC
0.5000 ± 0.0000**, at every tested load from 0.29× to 1.71× of capacity, under three explicit conditions
(`budget ≥ n_cells`, zero leak, depth-ordered eviction; an LRU policy is a hard error). ⛔ We do not call this
*certified*, *unlearning*, or *exact deletion* without those qualifiers: it is **verified byte-exactness under
the stated conditions, with the stated instrument** — locked into §3's trade (at matched bytes **at most
0.042 %** of an item's parameter mass could remain byte-exactly deletable) and **materially narrowed** by
neighbouring work reaching MIA-AUROC ≈ 0.5 by design in a different setting (Appendix Q), so the claim is
phrased on verified byte-exactness rather than priority. ⛔ **No rival family in this audit has a deletion verb
at all** — the SSD arm included — so this column has no cross-family row and we do not manufacture one.

### S.4 — §4.5 coupling measurement prose, in full

Measured on the learned store (delete the query's *second*-nearest stored key, divide by deleting its nearest;
3 seeds per radius), the coupling falls from **0.814 ± 0.31** at `d/s_fit = 1.10` to **0.01534 ± 0.006** at the
audited cell (`d/s_fit = 3.59`), while the per-slot table's third-party Δ is **`0.0` at every slot × every
dropped row × every cell — float equality, not a tolerance** (by construction, never a win). Fitting
`ln κ − ln(d/σ_q)` linearly in `d²` gives implied `s` = **0.3979** at **R² = 0.9953** over **2.72 decades**, and
two further estimates agree to **0.7 %**. ⚠ **This corrects one of our own earlier statements by a large
factor:** an earlier estimate placed this store at `d/s ≈ 1.9` with an `O(1)` coupling by reading the *admission
gate's refusal radius* as an achieved spacing; the achieved separation is `sep = 1.346`, so the audited
configuration runs at **`d/s = 4.34` (atom-width ruler)** / **`d/s_fit = 3.59` (fitted-width ruler)** with
coupling **1.53e-2** — a **45–52×** correction, in the direction that makes the table *harder* to escape, over a
swept span of **525× (2.72 decades)**. **Every `d/s` statement in this paper names its ruler.** ⚠ **Convention,
and the check we owe:** the effective-`s` estimator subtracts the store's own confining term `α‖q‖²` before
fitting, and a `d/s` computed without that subtraction is a different quantity; ⛔ on that convention **`s = 0.40`
is flagged for a check, not discharged** (the re-measurement is a declared NOT-RUN, Appendix J) — ⭐ with the
correction's direction known rather than guessed and running against us (smaller `s`, larger `d/s`, *more*
suppression), so every suppression number printed here is the **conservative** one.

### S.5 — §5 family survey, in full

**The family being audited** (per-arm detail: Appendix O; verified citations and quote scopes: Appendix Q). TTT
makes the memory an inner learner with a sequence-shared `W₀` and mini-batch `b = 16` *"for all experiments in
this paper"*, its Theorem 2 identifying the nonparametric TTT learner with a Nadaraya–Watson estimator; Titans
adds momentum and a forget gate; delta-rule linear attention runs DeltaNet → Gated DeltaNet → **Gated
DeltaNet-2**, our reference arm because it supersedes GDN — ⭐ and concurrently **Erase-then-Delta** makes the
same erase/write-decoupling move at larger scale, so the frontier *moved* rather than being conveniently chosen;
Sparse Delta Memory routes writes into explicit slots with a learned `M₀`; **Mamba-2** carries a matrix-valued
state under a scalar decay, and ⭐ the delta-rule line's own authors place it inside their family — ⚠ with their
hedge *"up to specific parameterization"*, which we carry, since Mamba-2's shipped parameterisation is a
per-head structured state and our arm's identity is stated and tested (Appendix O.2b) rather than inherited from
that sentence. **Evaluation conventions:** Based owns the field's only explicit *state-bytes-during-generation*
axis, populated by six neural sequence mixers and nothing else; MAD normalises to *"an iso-state and
iso-parameter setting"* across **neural** architectures only; Zoology varies state by hyperparameter; Sparse
Delta Memory reports a state-to-parameter ratio; HOLA adds a bounded exact KV cache to a linear-attention state
and compares against a *matched* recency cache — cited favourably: the field is already conceding that part of
the payload belongs in an exact store.

### S.6 — §5 ancestry paragraph, in full

⛔ We do **not** claim the unscoped version (*"no published rival paper runs a non-parametric matched-byte
control"*): it quantifies over all papers and is false. ⭐ One survey clause is no longer only a survey clause —
the SSM family is now represented by a **measured** arm at byte-identical state; ⚠ by **one** member (Mamba-1 and
Mamba-3 carry different state types, both declared NOT-RUN). **The ancestry we concede, in our own voice**
(quote scopes: Appendix Q): audit-at-equal-bits is standard outside this family; the substitute audit is the
partial-input tradition (Poliak et al. **significantly outperform** the majority-class baseline on **six of
ten** NLI datasets), with Feng et al.'s converse caveat carried; a **token-matched** trivial control was
published days before this work was filed, at a read-token rather than a state-byte budget; and kNN-LM /
MassiveDS price a datastore against training compute and parameters, never against a learned memory's state
bytes. ⭐ **What survives is stronger than a monopoly claim:** seven of the fourteen candidates build the
*adjacent* instrument and stop one step short of it. **A conceded ancestor is worth more than a contested
monopoly.** Our byte-floor result is an accounting identity about a specific store family, not a general
capacity bound.

### S.7 — §6 limitations L3–L10, in full

**L3 — The launder's scope.** It tests whether *inference-time* dynamics beat a table **given the
organisation**; both arms inherit the same placement of the same content. That is what this paper measures and
all it measures, in either direction. **L4 — The tuning standard was met; here is what it does not cover.**
Tuning bias is the attack this audit is most exposed to and it runs *toward* our headline; we show the grid is
not the binding constraint for the arms it was drawn around (§2.5), but **not** that no tuning protocol would
rescue an arm — optimiser family, schedule, architecture-side hyperparameters and the head-width allocation
forced by the iso-state rule were all held. **L4a:** our own selection rule is weaker than it looks — fit-split
best-of-grid cannot prefer a regulariser (12 of 45, fourth-decimal tie-breaks) and never prefers a smaller
learning rate (0 of 45); the held-out secondary gives the same headline and a *smaller* rescued set, making the
audit thinner rather than stronger, and we quote the intersection. **L5 — Measured versus reasoned families,
never blurred:** three of the five named rival families are adjudicated by measurement, **Sparse Delta Memory
and Titans from published equations alone**, and the SSM family is represented by one member.

**L6 — The byte-ratio caveat travels with every dividend or byte claim.** `ratio ≥ 2.20×` (`≥ 2.40×` with a
spectator dimension); the audited cell sits at 54.56× (**modal, 8 of 9 seeds**; 45.60× on the sixth-item seed);
**no cell measured in this work is a byte-matched dividend — the minimum ratio measured anywhere is 17.11×.**
**L7 — A table deletes exactly by construction** (§4.4), and the deletion column has no cross-family row.
**L8 — The conceded ancestry** (§5): we claim the uniform state-byte protocol, the learned-initial-state rule
and the finding — not the instrument's invention — and we carry Feng et al.'s converse caveat. **L9 — What the
theory does not derive** (Appendices D–E): the prefactor of the harness's own particle gradients is open (toy
law `N e^{−C}`, measured at `e^{−C}`, ≈ 1200 apart; the structural claim unaffected); ⚠ **naming `s` for a
learned multi-atom well gates the transfer of every geometric domain statement**, and §4.5's value holds **for
this store only** under a declared subtraction convention, ⛔ flagged for a check rather than discharged, with
the correction's known direction running *against* us; T5.4's coupling list is not proven exhaustive; and a live
launch-momentum probe is NOT-RUN, the suppression it would have to beat quantified at a factor **65**.
**L10 — Protocol caveats with domains** (Appendix E): below `s/sep ≈ 0.15` the basin boundary is *inertial* and
no static proxy is valid; **`λ_min > 0` does not certify a nonempty basin** (measured capture radius 0.000 at
`λ_min = +0.910`); `sep/2` is not a certified inradius.

