# [WORKING TITLE: When Does Test-Time Dynamics Buy Anything Over a Table at Matched Bytes?]

**[AUTHORS PLACEHOLDER]**

---

> ## ⛔⛔ DRAFT-STATE BANNER — READ BEFORE QUOTING ANY NUMBER (delete before any external circulation)
>
> **This is `draft-v1`, produced under an open NUMBER-FREEZE GATE.**
>
> Every number in this draft is in one of exactly two states, and the state is marked at the number:
>
> - **FROZEN** — traced to a landed artifact, quotable. This is *all* CLU-column numbers, *all* theorem
>   numbers, *all* instrument-validation (protocol-gate) numbers, *all* byte-ledger numbers, and *all*
>   third-party-attribution numbers.
> - **⟦F3⟧ PROVISIONAL** — a **rival-arm** number from the audit's **first tuning pass** (a reduced
>   learning-rate grid; §2.6). By standing decision of this project's own tuning rule, **no rival number
>   is final until the full-grid tuning pass reports its before/after table**, and **if tuning changes an
>   outcome the paper's claim changes with it.** Every such number is rendered `x ⟦F3⟧`.
>
> **The gate is OPEN as of this draft.** The full-grid pass is in flight; its before/after table has not
> landed. ⛔ **No `⟦F3⟧` number may be quoted, cited, screenshotted, or carried into a talk.** When the
> pass lands, `draft-v2` replaces every `⟦F3⟧` value with the full-grid value, re-states §2.6 as *"the
> full tuning standard was met, and the reduced-grid first pass is disclosed"*, and re-runs the §4.3 and
> §6 adjudications against the new numbers.
>
> Placeholders per project drafting policy: title is `[WORKING TITLE: …]`, authorship is
> `[AUTHORS PLACEHOLDER]`; both are workshopped at the end and both are blank in any anonymized build.

---

## Abstract

Modern sequence memories — test-time-trained memories, delta-rule linear attention, and learned
associative stores — all justify themselves by what their *dynamics* compute at inference. We ask a
question the family does not currently ask of itself: **at a matched state-byte budget, does a memory's
learned test-time dynamics beat a non-parametric table holding the same bytes?** We define a single
protocol that makes the question answerable and applies uniformly across memory types: a matched-byte
table launder, a two-sided byte ledger under an explicit **learned-initial-state rule** (an
initialisation is *parameters*; only the per-sequence deviation is *state*), a **+0 B substitute audit**
over the memory's own stored bytes, a same-keys null, a blank-store control, and a **rescue gate** that
disqualifies any arm sitting within 2 SE of its own blank store. We apply it to two rival test-time-memory
families and to a learned continuous-latent store (the CLU), on a synthetic memory task at CPU scale
(`d_in = 5`, 5–6 stored items, ~10-token streams, 3 seeds), and we report the result for every arm
including our own.

Three findings. **(i)** At matched state bytes, on the one task family that survives our own protocol
validation, **no arm in the audit — neither the rival families ⟦F3⟧ nor the CLU (+0 B margin
−0.3180 ± 0.0804)** — beats a zero-extra-byte reader of a *raw* table holding the same bytes.
**(ii)** The control the field would naturally write is the wrong one: reading a weight-valued memory's
byte-matched table *through the memory's own projections* costs that table up to 1.208 neg-MAE ⟦F3⟧
against a raw-metric table at identical bytes, which manufactures large apparent dividends
(+0.88…+1.02 ⟦F3⟧) that vanish under the raw control. Both controls were pre-registered before
measurement; that ordering is the only reason the second is a finding rather than a re-frame.
**(iii)** For a store with one private parameter group per item, matched bytes is **unreachable by
construction**: an accounting identity gives `ratio = [A(D+2)+d]/(d+m) ≥ 2.20×`, and that floor is
exactly the byte price of one privately-deletable parameter group per item — so byte-exact deletion and
compression are **the same trade**, with a computable exchange rate.

We also measure the single coupling a row-selecting table provably cannot express (a read's dependence
on a stored item the query did *not* select) and find it obeys `exp(−½(d/s)²)` on our learned store
(R² = 0.995) — **exponentially suppressed by the very admission gate that keeps the store writable**.
We conclude with the audit's own conclusion and no more: *a store organised well enough to be safe is
organised well enough to be a table.*

---

# 1. Introduction

A memory module in a modern sequence model earns its place by what it does at inference. Test-time-trained
memories (TTT; Sun et al., 2024) run an inner optimiser over the stream. Delta-rule linear attentions
(DeltaNet, Yang et al., 2024; Gated DeltaNet, Yang et al., 2025; Gated DeltaNet-2, 2026) run a
key-conditioned erase–write recurrence. Titans (Behrouz et al., NeurIPS 2025) runs a momentum-accelerated
associative write. Sparse Delta Memory (Cabannes et al., 2026) routes writes into explicit slots. A
continuous-latent store — the CLU, introduced as CHLU in Jawahar & Pierini (2026) — integrates a damped
particle through a learned potential and reads where it settles. All of these are *dynamics at test time
over a bounded state*, and all of them are evaluated against **other neural architectures**.

The comparison none of them runs is the cheapest one available: **put a non-parametric store on the same
byte budget and see who wins.** That comparison is not exotic. It is standard practice one field over —
learned Bloom filters, learned indexes and their benchmarks compare a learned structure against a classical
one *at matched space* as a matter of course (Mitzenmacher, 2018; Kipf et al., 2019) — and it is a
recognised discipline in evaluation methodology under the name *partial-input baselines* (Poliak et al.,
2018; Feng, Wallace & Boyd-Graber, 2019). We are importing an established discipline into a family that
has not adopted it. We are not inventing it, and §5 says so before a reviewer does.

This paper is an audit. It applies one protocol uniformly to rival memories and to our own, and it reports
the same columns for everybody. Its headline finding is negative for every arm in it, including ours.

## 1.1 Contributions

1. **A uniform matched-byte audit protocol for bounded-state memories** (§2), with five mandatory columns
   — matched-byte table launder · two-sided byte ledger · **+0 B** substitute audit · same-keys null ·
   blank-store control — plus a **rescue gate** and an identical-encoder invariant enforced in code.
2. **The learned-initial-state rule** (§2.3), which we believe is the protocol's load-bearing convention:
   *for any memory with a learned initial state (TTT's `W₀`, an explicit-slot memory's `M₀`, our own
   `V_θ` initialisation), the initialisation is **parameters** and only the per-sequence deviation is
   **state**; both are declared.* Counting the init as state inflates a memory's apparent budget;
   counting the deviation as parameters launders it. **We apply the rule to our own store in the same
   table**, and it costs us: our measured per-stream deviation is 5200 B against 5376 B of
   initialisation, a state/parameter ratio of 0.967 (§4.4).
3. **A methodological finding on the rival side of the audit** (§2.4, §4.3): the byte-matched table one
   would naturally build for a weight-valued memory — the table of the memory's own projected
   `(θ_K x, θ_V x)` pairs — is **not a neutral control**. It costs the table up to 1.208 ⟦F3⟧ against a
   raw-metric table at identical bytes, because those projections are trained for a recurrence rather
   than for a table, and because a single stored value is out of distribution for an output head trained
   on kernel-averaged reads. Running only that control publishes dividends that do not exist.
4. **Three results about what matched bytes can even mean for a per-item store** (§3): the byte-floor
   identity `ratio = [A(D+2)+d]/(d+m) ≥ 2.20×`; its corollary that **compression and byte-exact deletion
   are the same trade** with exchange rate `dp/dr = (d+m)/[(D+2)A_tot]`; and a dichotomy theorem showing
   that a **settled-point read is untrainable end-to-end in both directions** because the fixed-point
   equation contains no transient parameter.
5. **The measurements** (§4): the rival rows ⟦F3⟧, the CLU column, the byte-frontier curve, and the one
   coupling a row-selecting table structurally cannot express, measured on a learned store.
6. **The negative, stated as the finding it is** (§4.2, §7), and a limitations section (§6) that states
   the audit's own thinness before a referee has to.

**Headline figure — Figure 1.** *Signed +0 B margin against a raw-metric byte-matched table, one bar per
audited arm, with the zero line drawn and rescue-gate-failing arms hatched.* Every bar is below zero.
This is the paper. (Specification in Appendix K; the rival bars are ⟦F3⟧ and the figure is not rendered
in this draft.)

## 1.2 What this paper measures, and what it does not — stated in our own voice, early

The control at the centre of this protocol is a **settle-deleted / matched-bytes launder**: the memory's
own stored content, re-served as a table at the same byte budget, read by the cheapest reader that works.
It is a strong control, and it has a precise scope which we state before we use it:

> **The matched-bytes launder tests whether a memory's *inference-time* dynamics beat a table **given the
> organisation** — both arms inherit the same placement of the same content. It does not test how that
> content came to be organised, and it is not evidence about any other stage of the system.**

Stating that scope precisely is this paper's integrity, not its weakness. Everything we report is a
statement about inference-time reads at matched bytes on the tasks named, at the scale named. Nothing in
this paper is an argument about any other regime, and we do not make one.

Two further boundaries, both hard:

- **The task families here are designed synthetics at CPU scale.** They were built to isolate specific
  memory pressures, and our own protocol validation (§2.5) shows that three of four of them measure the
  construction rather than the memory. We report the audit on the one that survives, and §6 says plainly
  that a two-rival-family audit against **one** surviving synthetic family is a thin cross-family audit.
- **Nothing here transfers to a language-model claim.** Every measured cell in this paper runs at
  `d_in = 5`, 5–6 stored items, ~10-token streams, on CPU. We did not size, run, or approximate a
  language-modelling experiment, and no number here should be read as bearing on one.

## 1.3 Verification versus evidence

We distinguish two grades of result throughout, and we label them at the table:

- **Verification** — results on *designed* testbeds: architecturally-specified potentials, two-well toys,
  analytic geometries, and exact arithmetic over recorded ledgers. These confirm that a stated identity
  or law is exact where it is claimed to be exact. They are not discoveries and we do not present them
  as such. §3.1–§3.4 and Appendices C–E are verification.
- **Evidence** — results on *learned* systems: a trained `V_θ` store, trained rival memories, and the
  audit columns computed on them. §4 is evidence. Where a verification result and an evidence result
  meet (e.g. the `exp(−½(d/s)²)` law verified on a designed toy and then measured on a learned
  multi-atom store, §4.6), we say which is which and report the discrepancy rather than the agreement.

---

# 2. The protocol

## 2.1 The object under audit

A **bounded-state memory** is a module that maintains a state `S_t` of fixed byte size across a stream,
writes items into it, and answers queries from it. We audit a memory at a fixed **iso-state budget**:
all arms in a cell are sized so that their declared state occupies the same number of bytes (here
1364 float32 = 5456 B; head widths derived from that budget and registered before any run — Appendix A).

Every arm in a cell shares one read-in encoder `φ`, whose bytes are ledgered on **every** arm including
the launder. This is enforced in code: each cell asserts a content hash of `(q₀, keys)` across all arms
and raises on mismatch. A 1e-9 perturbation raises (tested).

## 2.2 The five mandatory columns

| column | definition | what it rules out |
|---|---|---|
| **matched-byte table launder** | the memory's own stored content re-served as a non-parametric table of `n_rows = ⌊state_floats/(d_k+d_v)⌋` rows at the same byte budget, read by a fixed reader | "the dynamics are doing the work" when the content alone suffices |
| **two-sided byte ledger** | a declared split into **F1 parameters** (shared across sequences), **F2 state** (per-sequence, the audited budget), and **F4 per-read transients**; the breakdown must sum to the total or the run raises | budget laundering in either direction |
| **+0 B substitute audit** | the strongest reader we can construct that uses **zero extra bytes** beyond the launder's own table (arg-min, 2-NN mean, 2-NN inverse-distance, echo, insertion-order, order-aware) | "our memory beat its launder" when neither is the best reader of its own bytes |
| **same-keys null** | the same read run against a store written with the same keys but a permuted/independent payload assignment | key-side leakage through the encoder |
| **blank-store control** | the identical read path against a store with nothing written in it | reads that succeed on an empty store (a measured failure mode: address leaks can make a blank store classify perfectly) |

**The rescue gate.** An arm whose full read sits **within 2 SE of its own blank-store control** is
**NOT RESCUED**, and **no margin against it is quotable**. This is applied per cell, reported first-class,
and it is not a formality: it disqualified two of five arms on the dividend family and **all** arms on the
byte-frontier column in the first pass (§4.5).

**Admissible-cell coverage is reported first-class**, per family per seed, so that a verdict can never be
read without knowing how much of the cell it rests on.

## 2.3 The learned-initial-state rule (contribution 2)

Several memories in this family carry a **learned initial state**: TTT's `W₀` is explicitly *"shared
between all sequences"*; explicit-slot memories learn an `M₀`; our own store's `V_θ` has a learned
initialisation. There is no published convention for how to count it, and the choice moves the number a
lot in both directions.

> **Rule.** The initialisation is **parameters** (F1). Only the **per-sequence deviation** from it is
> **state** (F2). Both are declared, in the same table, for every arm.

Counting the init as state *inflates* a memory's budget (and so makes its matched-byte table larger and
easier to beat). Counting the deviation as parameters *launders* it (and so hides state in the parameter
column). Neither direction is safe, so the rule fixes both and publishes both.

⭐ **We apply the rule to ourselves in the same table, and it is not free.** For our store we do not
*assume* the deviation — we measure it, by diffing `V_θ` before and after the stream. On the audited cell
the write moves **192/192 atom centres (960 floats) but only 160/192 widths and amplitudes**, plus 20
codebook floats, for a measured deviation of **1300 floats = 5200 B** against an initialisation of
**1344 floats = 5376 B**: a state/parameter ratio of **0.967**. (The 32 unmoved widths/amplitudes are the
one free slot, whose centres are re-drawn by the allocator while its widths/amplitudes are re-set to
their initial constants. Benign, fully explained, and — as far as we know — the first time a store's
write footprint has been measured rather than inferred from its write mask. Appendix G.)

## 2.4 The +0 B substitute audit, and the control it must include

The substitute audit asks: *is the memory the best reader of its own bytes?* We give a zero-extra-byte
reader the launder's table and let it answer. If it matches or beats the memory, the memory's advantage
over its own launder is not evidence about the memory.

This idea is not ours in general form and we say so in §5: it is the partial-input / trivial-baseline
tradition (Poliak et al., 2018; Feng, Wallace & Boyd-Graber, 2019), applied here to a *memory's own
stored bytes* at a *state-byte* convention. We also carry Feng et al.'s converse caveat explicitly,
because it bounds what a *passed* audit can license: **a substitute audit that a memory passes does not
show the memory is doing real work.** Only a *failed* one is informative, and in this paper the audit
fails for every arm.

⭐ **The control the audit must include, and why (this is a contribution, not a caveat).** For a
weight-valued memory the natural byte-matched table is the table of the memory's own projected
`(θ_K x, θ_V x)` pairs — that is the construction we pre-registered. We implemented exactly it, and
**it is not a neutral control.** Read through the memory's own projections, a table at the same bytes is
handicapped by 0.203–1.208 neg-MAE ⟦F3⟧ relative to a **raw-metric** table holding the same bytes, for two
reasons visible at equation level: (i) `θ_K, θ_V` are trained for the *recurrence*, not for a table, so
arg-min in the projected space is a worse metric than the raw address space; and (ii) a single stored
value decoded by the memory's own output head `θ_O` is **out of distribution** for `θ_O`, which was
trained on kernel-averaged reads `o = Σ_s v_s(k_s·q)`, not on isolated stored values.

A paper running only the projected control would report large positive dividends
(+0.88…+1.02 ⟦F3⟧ on the delta-rule arms) that **do not survive** a raw table at the same bytes. Both
controls are therefore mandatory in this protocol, and both are reported for every arm; neither is ever
substituted for the other.

⚠ **The pre-registration ordering matters and we state it in the methods rather than in a rebuttal.**
The projected control (predicted to show a positive dividend) and the raw control (predicted to erase it)
were **both registered before any measurement**. Had the raw control been added *after* seeing the
projected result, it would have been indistinguishable from a re-frame, and we would not be entitled to
present it as a finding. The registration order is what makes §4.3 admissible.

## 2.5 Validating the instrument before spending it

Before running the protocol on multiple memory families, we ran it against a **full-attention table
reader** on every candidate task family and asked a falsification question about the protocol itself:
*if a ≤4 B substitute is at the metric's ceiling for every family, including for full attention, the
protocol is measuring the task and not the memory.*

**It did not fire — but it came within one family of firing, and the result reshaped the paper.** Define
the saturation score `S(f) = (sub − blank)/(M − blank)`, with `M` the metric's maximum. Measured
(3 seeds, evidence-grade on the learned store, designed task families):

| family | metric | `M` | blank | +0 B substitute | full attention | **`S(f)`** | verdict |
|---|---|---|---|---|---|---|---|
| `overload` (byte ratio 478.2×) | decode | 1.0 | 0.1667 | **1.0000** (settle-deleted) | 1.0000 | **1.0000** | ⛔ saturated |
| `aggregate` (54.56×) | neg-MAE | 0.0 | −0.4221 | **−0.2081** (2-NN) | −0.2493 | **0.5068** | ✅ **survives** |
| `recency` (54.56×) | acc | 1.0 | 0.5463 | **1.0000** (order-aware) | 0.4755 | **1.0000** | ⛔ saturated |
| `manifold` (52.0×) | R² | 1.0 | −0.0001 | **1.0000** (echo) | 0.0000 | **1.0000** | ⛔ saturated |

**Three of four designed families are struck as protocol-invalid.** On three of the four, something
costing ≤4 B sits at the metric's *exact maximum* — and **it is never the memory**. On `overload` our
own store reads **0.9722 ± 0.0139**, below three different readers of a table costing **1/478th** of its
bytes. On `recency` and `manifold` the ≤4 B substitute is at ceiling while full attention reads
0.4755 / 0.0000; those families measured the construction, not the memory, and we removed them from the
audit rather than reporting the memory's numbers on them.

⚠ **`overload`'s verdict turns entirely on one definitional choice, which was pre-registered before the
run:** with the arg-min launder excluded from the +0 B reader set, `S_excl(overload) = 0.6500` and the
family survives. We report both readings, we selected the strict one, and we carry `overload` **only** as
an explicitly labelled **byte-frontier column** (§4.5) — never as a dividend family and never as reader
discrimination.

⭐ **The one generalisable design rule we can extract, and it is the only one two waves of auditing
support:** `aggregate` survives for exactly one reason — **its target is constructed to be absent from
the table** (a query whose answer lands within tolerance of a stored payload is dropped at construction).
*"The answer is provably not in the table"* is **the only family property that has survived a +0 B
substitute audit** in two rounds of this audit (0-of-4, then 1-of-4). Any future synthetic family for
this protocol should be built to that rule.

## 2.6 ⚠ Tuning the rivals — the audit's own first pass, disclosed

> ⛔ **This subsection is written against the OPEN gate and is re-written in `draft-v2`.**

The audit's finding is *"rivals lose to their own byte-matched tables"*, and **under-tuning a rival
produces exactly that finding.** The bias runs *toward* our headline, which is the dangerous direction,
and rescued baselines are a standing requirement of this project.

**First pass (reported here, ⟦F3⟧):** the rival outer loop was Adam, 400 steps,
`lr ∈ {1e-3, 3.16e-3, 1e-2}` (TTT arms additionally mini-batch `b ∈ {1, 16}`), best-of-grid selected
**on the fit split only**, with the outer parameters never seeing the eval stream. This is a **reduced**
grid relative to the tuning standard this protocol specifies (6 learning rates × 2 weight decays), and
it was declared at the time as a budget choice, not as compliance.

**Full pass (pending; the gate this draft waits on):** the full `6 lr × 2 wd` grid on the rescued arms
(mandatory) and on the two non-rescued arms (rider — their non-rescue is itself a tuning-sensitive
outcome, and *"you hobbled the competition"* bites hardest on an arm declared dead on three learning
rates), plus a 2000-step re-check of the grid's best configuration so that *"more steps would have
rescued it"* is closed by measurement rather than by assertion. Outcome-change thresholds are registered
before the grid runs.

⭐ **The pre-commitment, made before the pass ran and binding on this paper: if proper tuning changes any
outcome, the paper's claim changes with it.** `draft-v2` reports the full-grid numbers as the tuning
standard met, with this reduced-grid run disclosed as the audit's first pass and its before/after table
printed in an appendix.

Independently of the grid, one under-training check is already closed by measurement: at **5×** the outer
budget (2000 steps vs 400) the frontier-column rival arms get *worse*, not better, while a rival's
fit-split loss reaches MAE 0.024 against an eval MAE of 0.75 — a **31× fit-to-eval generalisation gap**
across item geometries, forced by the guard that outer parameters never see the eval stream. That is a
generalisation failure, not a budget artefact (Appendix H).

---

# 3. What matched bytes can mean: three results

This section is **verification** in the sense of §1.3: each result is an identity or a theorem, checked
in exact arithmetic or on designed geometries. None of it is a performance claim, and none of it licenses
one.

## 3.1 The byte-floor identity

Let the store be an atom dictionary
`V_θ(q) = α‖q‖² − Σ_{j=1..N_at} A_j exp(−‖q − c_j‖²/2s_j²)` with learnable `(c_j ∈ R^D, log s_j, amp_j)`,
partitioned into **one atom group per item slot** (a masked, item-local write), and let the matched-bytes
launder be a table of `K` live rows `(key ∈ R^d, payload ∈ R^m)`. Write `A ≡ N_at/K` for atoms per live
item and `D = d + m + n_spectator` for the store dimension.

> **Theorem T1 (per-item atom-group byte floor).**
> ### `ratio ≡ full_bytes / launder_bytes = [A·(D+2) + d] / (d+m)`
> exactly and independently of `K`; and since one atom group per item forces `A ≥ 1`,
> ### `ratio ≥ [(D+2)+d]/(d+m)` = **2.20×** at `(d, m, n_spec) = (4, 1, 0)` and **2.40×** at `(4, 1, 1)`.

Four load-bearing assumptions: (A1) every learnable leaf is one of `centers / log_width / amp`, i.e.
`4(D+2)` bytes per atom in float32; (A2) the live-address codebook `4Kd` is counted on the store side and
the launder's key column is the same `4Kd`, so the address block cancels to a constant `d/(d+m)` rather
than to zero; (A3) the launder row is `(d+m)` floats — address and payload, not spectator coordinates;
(A4) atoms are **private**, no atom's parameters entering two items' reads.

**Verification.** In exact integer/rational arithmetic over 28 recorded ledger cells: the byte
decomposition is exact **28/28**; the identity above reproduces the measured ledger ratio **28/28 at
0 ulp**; the floors are `2.20 / 2.40` (Gaussian, `n_spec = 0/1`) and `2.40 / 2.60` for a shell-atom basis
(a basis change **raises** the floor by exactly `1/(D+2)`, so it is not a route to matched bytes).

⚠ **Erratum, ours, printed here rather than buried.** A previously published closed form for this
ratio (`1.4·A + 0.8`) and the statement *"verified to 1e-9 in all 28 cells"* are **wrong in 4 of the 28
cells** — every cell with a spectator dimension, where the formula divides by the store dimension `D`
where the launder row is `(d+m)` floats. Measured 52.00× against a published 43.33× (+20 %), with the
printed floor 2.00× where the true floor is 2.40×. **The corrected law above is exact in all 28.** The
error is **conservative** — it understated both the ratio and the floor — so no claim built on it was
inflated. The bug was invisible to the test suite because no test exercised a spectator dimension; that
is the coverage lesson, not merely a formula fix.

**Domain.** Exactly this store family (atom dictionary, group-masked write, one group per slot). It does
**not** apply to an MLP or Hopfield store, to a shared/factored substrate, or to any store whose write is
not group-masked.

## 3.2 Compression and byte-exact deletion are the same trade

Let `A_tot` be an item's atom budget and `S` the number of items sharing each non-private atom, so
`N_at = K·A_tot/S`. Then

### `ratio(S) = A_tot(D+2)/[S(d+m)] + d/(d+m)`,  and `ratio = 1 ⟺ S* = A_tot(D+2)/m`.

| quantity | value at `d=4, m=1, n_spec=0` |
|---|---|
| items per shared atom needed for matched bytes at `A_tot = 1` | **`S* = (D+2)/m = 7`** |
| … at the shipped anchor `A_tot = 341` | **`S* = 2387`** |
| private-parameter fraction attainable at ratio `r` | **`p ≤ [(d+m)r − d]/[(D+2)A_tot]`** |
| … at `r = 1`, `A_tot = 341` | **`p ≤ 4.19e-4`** — 0.042 % of an item's parameter mass |
| exchange rate | **`dp/dr = (d+m)/[(D+2)A_tot] = 2.10e-3`** per unit of byte ratio |

> **Prop T1.4.** Byte-exact deletion in `O(1)` is available **exactly on an item's private-atom fraction
> `p`**, because byte exactness *is* the statement that item `i`'s parameters form a block disjoint from
> every other item's — which is the same property (A4) that forces `A ≥ 1` and hence `ratio ≥ 2.20×`. On
> the shared fraction `1−p` there are two options and no third: leave the shared atoms (deletion is not
> byte-exact and carries a residual), or re-fit them (every co-tenant item's bytes change, and the cost
> is a *write*, not a delete — which is precisely the retraining baseline exact unlearning is defined
> against).
>
> **Corollary.** Since a private atom is indivisible, `p ≥ 1/A_tot` forces `r ≥ [(D+2)+d]/(d+m)`.
> ⭐ **The 2.20× floor is exactly the byte price of one privately-deletable atom per item.** It is not a
> property of a budget choice; it is the cost of the smallest unit of byte-exact deletion.

This proposition bounds **byte** exactness only. It says nothing about behavioural unlearning metrics,
which is a different measurement with a different literature (§4.7, §5).

## 3.3 When a settle is arg-min (Prop D2a)

> **Prop D2a.** Under (H1) separable wells — enforced spacing, every stored site a strict local minimum
> with `λ_min > 0`; (H2) equal well depth at the stored centres; (H3) a **settled-point-only** read
> `ŷ = ψ(q*)`; and (H4) a query law supported inside the certified basins — the settle map is **exactly**
> arg-min over the stored centres, and the dividend against a matched-key arg-min launder is **exactly 0**,
> not "small".

The audited anchor cell realises this: dividend **0.0000** with disagreement mass `D = 0`, reproduced
three times independently.

**Which hypothesis, dropped, breaks the conclusion** (verification, designed two-well geometries):

| dropped | mechanism | measured | consequence |
|---|---|---|---|
| (H2) equal depth | the axial separatrix shifts by `δ = ln(A_i/A_j)/(d_ij/s² − 4/d_ij)` | boundary at +0.0163…+0.0791 vs predicted +0.0156…+0.0767 over five depth ratios — **3.1–4.8 % relative** | `D > 0` with a *derived* value; measured/predicted **1.141** over 7 cells |
| (H1) separability | wells merge; sites stop being minima | `D = 0.0000` down to `d/s = 2.86`; **0.0550** at 2.29; **1.0000** at 1.71 | the conclusion fails **and the store fails with it** |
| (H4) query law | ⛔ **refuted as a hypothesis in the equal-depth case** | `D = 0.00000` at every `sep/σ_q ∈ [2,10]`, n = 4000/cell | with equal depths the settle boundary *is* the Voronoi boundary by symmetry, so `D = 0` independently of the query law |
| (H3) settled-point-only read | the **only** drop that opens a channel without degrading the store | within-basin sd of `q*` = **6.6e-10** (piecewise constant); within-basin sd of `q_t/σ_q` = 0.99 (t=1) · 0.43 (t=10) · 0.042 (t=100) · <1e-3 (t=240) · 4.7e-9 (t=1200); `p_t` peaks at **1.88 σ_q at t≈10** | a table returns one value per basin; the trajectory returns a continuum — but only for `t ≲ 240` steps |

⚠ **The disagreement mass `D` is the dividend's *variance*, not its magnitude.** `D` is the query mass
between the arg-min boundary and the settle's true basin boundary. It bounds *where* a dividend could
live and says nothing about its sign: the cell with the largest `D` (0.931) has the **worst** dividend
(−0.875). `D` is never a progress signal.

**Status.** Proven in the symmetric case (an exact reflection-symmetry argument; measured offset 2.92e-8
against a 2e-3 bar); evidenced at 3–5 % in the asymmetric case on a 2-D toy, valid for
`|δ| ≤ 0.25 d_ij` and `s/sep ∈ [0.15, 0.30]`; **invalid below `s/sep = 0.15`**, where the basin boundary
is inertial rather than static (§6, T4.1).

## 3.4 A settled-point read is untrainable end-to-end, in both directions

> **Theorem T3.** For the dissipative velocity-Verlet map `T_θ` with separable `H = T(p) + V_θ(q)`, for
> every `γ ∈ (0,2)`, `dt > 0`, inertial mass `M ≻ 0`:
> **`Fix(T_θ) = {(q, 0) : ∇V_θ(q) = 0}`**. The defining equation contains `θ` **only**. Hence the read's
> parameters split into
> - **fixed-point parameters** `θ` (the store), with `∂q*/∂θ = −(Hess V_θ(q*))⁻¹ ∂_θ∇V_θ(q*)`, exact and
>   with no `(γ, dt, M)` correction; and
> - **transient parameters** `ζ ∈ {q₀, p₀, M, γ, dt, integrator}`, which appear in the *approach* and
>   nowhere else, so **`∂z*/∂ζ ≡ 0` exactly** in the fixed-point limit;
>
> and at finite budget the surviving sensitivity is the un-decayed remnant of the contraction,
> ### `‖∂z_N/∂ζ‖ ≍ K_ζ · exp(−C)`,  `C ≡ Σ_p N_p ln(1/ρ_p)`,
> with `ρ_p` the spectral radius of the exact 2×2 damped-Verlet propagator, `K_ζ = O(1)` for `ζ = q₀`
> (a pure Jacobian product) and `K_ζ = O(N)` for `ζ ∈ {M, γ}` (injected at every step).

**Both directions are the same statement.** `q₀` is the read-in encoder's output; `M` and `γ` are the
particle's attributes. Measured on the learned system: `‖∂L/∂φ‖` = **0.0** (implicit) / **2.654e-9**
(unrolled) / **6.421e-3** (trajectory read) — a ratio of **2.42e6**; and the inertial-mass gradient is
**exactly 0.0 bitwise on 3/3 seeds**. Verified on designed geometries: `Fix(T)` is `(γ, M)`-independent
to **1.67e-15**; the `e^{−C}` slope is **−0.9941 over 143.9 decades** (per-γ −0.981 / −1.007); after
dividing out the derived `N` prefactor the `{M, γ}` law is exact to **±1 %**. For the read schedule used
here, `C = 18.34` ⇒ `e^{−C} = 1.084e-8`, which brackets both measured harness gradients.

⭐ **Zero fitted parameters, and it retro-explains a prior negative result.** An earlier attempt in this
line to *learn* an address by gradient descent through a settled-point read failed at chance
(4.2 % on one implementation; 0–2 of 18 on another; the loss frozen to 7 significant digits over 4000
steps). Because every mode is underdamped at those friction values, `ρ = √(1−γ)` exactly and
landscape-independently, so `‖∇_address‖ = 3.3e-1·(1−γ)^{600}` has **no free parameter but the `γ = 0`
anchor**. Measured versus predicted over five decades: ratios **1.00 / 0.92 / 1.17 / 2.14 / 8.9**, with
the last two measured points non-monotone in `γ` — that instrument's numerical floor, not physics. At a
3000-step probe the predicted address gradient is `10^{−33.4}`, **26 orders below float32 epsilon**:
gradient descent was optimising round-off.

⛔ **Consequence for this protocol, stated as a scope limit rather than a proposal.** The
retrieval-robustness / learnability trade here is definitional, not tunable: retrieval robustness *is*
`∂(final)/∂q₀ → 0`, and the convergence budget `C` sets both. Any claim about training *through* a
settled-point read is bounded by `e^{−C}`; we make none.

**Not derived, and never quoted as settled:** the *prefactor* of the harness's own particle gradients.
The toy law predicts `‖∂q_N/∂{M,γ}‖ ≍ N e^{−C}`, while the measured harness mass gradient (8.73e-9) sits
at `≈ e^{−C}`, i.e. a factor `N ≈ 1200` below. The structural claim — exactly zero implicit,
exponentially small unrolled — is unaffected; the prefactor is open.

---

# 4. Results

**Grade: evidence** (learned store, trained rival arms), except where labelled otherwise. All cells:
3 seeds (0, 1, 2), sample sd with `ddof = 1`, `SE = sd/√3`, identical `φ` asserted in code, byte ledger
identity asserted as integers. Flag-provenance tables: Appendix A.

## 4.1 The audit table

**Column status per `PREREG` §2.** Every closed cell is marked `have`; every open one is a **declared
NOT-RUN with its reason** and is never reported as a null.

| family / arm | matched-byte table launder | **+0 B** substitute (signed margin) | two-sided byte ledger | same-keys null | metric-native verdict | deletion probe | anytime / frontier |
|---|---|---|---|---|---|---|---|
| **CLU** (banked, reproduced digit-for-digit) | have **−0.4472** | have, **−0.3180 ± 0.0804** | have **5456 B / 100 B, 54.56×** | have **−0.8175** | have | have (MIA-AUROC **0.5000 ± 0.0000**, byte-equal **3072/3072**) | have (banked curve) |
| **TTT-Linear** | have −0.4245 ⟦F3⟧ | have, **−0.0523 ⟦F3⟧** | have F1 5592 B / F2 5220 B | have −0.4577 ⟦F3⟧ | have, metric-native | ⛔ NOT-RUN — no deletion verb exists in the family | frontier, **non-informative** |
| **TTT-MLP** | have −0.4108 ⟦F3⟧ | have, **−0.2284 ⟦F3⟧** | have F1 5736 B / F2 5376 B | have −0.4285 ⟦F3⟧ | have, *weakly* metric-native | ⛔ NOT-RUN — as above | — |
| **DeltaNet** | have −0.6658 ⟦F3⟧ | have, **−0.0047 ⟦F3⟧** | have F1 9956 B / F2 5184 B | have −0.6480 ⟦F3⟧ | have, metric-native | ⛔ NOT-RUN | — |
| **Gated DeltaNet** (ablation) | have −1.4158 ⟦F3⟧ | have, **+0.0448 ⟦F3⟧** | have F1 9956 B / F2 5184 B | have −1.2202 ⟦F3⟧ | have, metric-native | ⛔ NOT-RUN | — |
| **Gated DeltaNet-2** (reference arm) | have −1.2735 ⟦F3⟧ | have, **+0.0445 ⟦F3⟧** | have F1 9956 B / F2 5184 B | have −1.1341 ⟦F3⟧ | have, metric-native | ⛔ NOT-RUN | frontier, **non-informative** |
| Titans (MAC) | ⛔ NOT-RUN | ⛔ NOT-RUN | ⚠ **UNPINNED** — `2·|M_θ|` is our reconstruction; the paper states no convention | ⛔ NOT-RUN | positioning only | ⛔ NOT-RUN | ⛔ NOT-RUN |
| Sparse Delta Memory | ⛔ NOT-RUN | ⛔ NOT-RUN | Eq. 6, positioning only | ⛔ NOT-RUN | positioning only | ⛔ NOT-RUN | ⛔ NOT-RUN |
| Mamba-2 / GRU / sliding-window attention | ⛔ NOT-RUN — outside the ruled arm set | | | | | | |

**NOT-RUN reasons, stated once.** *Titans:* peer-reviewed at NeurIPS 2025 (never "a preprint"), but no
official code, the chunk size is never given a numeric value, and no seeds are reported — an arm would be
our reconstruction audited against our reconstruction's table. *Sparse Delta Memory:* its official
implementation requires Torch ≥2.8 / Triton ≥3.4 / SM 80+ hardware and cannot run at this weight class.
*Deletion column:* **no rival family has a deletion verb at all**, which is precisely why the deletion
result sits in the "and also" position (§4.7).

### 4.1.1 The measured audit, in full

| arm | `d_head` | F1 param B | F2 state B | own table B | **full** | own arg-min table | **dividend** | **+0 B margin** | ⭐ **raw-metric +0 B margin** | blank | lift over own blank | **RESCUED?** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ttt_linear | 29 | 5592 | 5220 | 5104 | −0.4546 ± 0.0312 ⟦F3⟧ | −0.4245 ⟦F3⟧ | −0.0302 ⟦F3⟧ | **−0.0523 ⟦F3⟧** | **−0.2465 ⟦F3⟧** | −0.8426 ⟦F3⟧ | +0.3879 ± 0.0869 ⟦F3⟧ | ✅ ⟦F3⟧ |
| ttt_mlp | 12 | 5736 | 5376 | 5376 | −0.6324 ± 0.2036 ⟦F3⟧ | −0.4108 ⟦F3⟧ | −0.2216 ⟦F3⟧ | **−0.2284 ⟦F3⟧** | **−0.4242 ⟦F3⟧** | −0.6031 ⟦F3⟧ | −0.0293 ± 0.1090 ⟦F3⟧ | ⛔ **NO** ⟦F3⟧ |
| deltanet | 36 | 9956 | 5184 | 5184 | −0.4652 ± 0.0402 ⟦F3⟧ | −0.6658 ⟦F3⟧ | +0.2006 ⟦F3⟧ | **−0.0047 ⟦F3⟧** | **−0.2571 ⟦F3⟧** | −0.5657 ⟦F3⟧ | +0.1004 ± 0.1296 ⟦F3⟧ | ⛔ **NO** ⟦F3⟧ |
| gdn | 36 | 9956 | 5184 | 5184 | −0.3961 ± 0.0208 ⟦F3⟧ | −1.4158 ⟦F3⟧ | +1.0197 ⟦F3⟧ | **+0.0448 ⟦F3⟧** | **−0.1880 ⟦F3⟧** | −1.3220 ⟦F3⟧ | +0.9259 ± 0.2387 ⟦F3⟧ | ✅ ⟦F3⟧ |
| **gdn2** | 36 | 9956 | 5184 | 5184 | −0.3964 ± 0.0220 ⟦F3⟧ | −1.2735 ⟦F3⟧ | +0.8771 ⟦F3⟧ | **+0.0445 ⟦F3⟧** | **−0.1883 ⟦F3⟧** | −1.6618 ⟦F3⟧ | +1.2654 ± 0.4968 ⟦F3⟧ | ✅ ⟦F3⟧ |
| **CLU** | — | 5376 | 5200 | 100 | **−0.5261 ± 0.0863** | **−0.4472** | **−0.0789** | **−0.3180 ± 0.0804** | (its own table is already raw) | −0.4221 | — | ✅ |

**Rescue-gate verdicts (first pass, ⟦F3⟧).** Rescued on the dividend family: `ttt_linear`, `gdn`, `gdn2`.
**Not rescued: `ttt_mlp`, `deltanet`** — within 2 SE of their own blank stores, so **no margin against
either is quotable**, here or anywhere. Only rescued arms enter the falsifier adjudications of §4.3.

**Admissible-cell coverage, first-class.** `aggregate`, seeds 0/1/2: **58/72 · 66/80 · 55/80** admissible
queries (0.806 · 0.825 · 0.688) with **5/8** offered items admitted on all three seeds. The drops are the
family's own construction rule (a query whose target lands within payload tolerance of a stored payload is
dropped) — which is exactly what stops the arg-min launder from being accidentally right.

## 4.2 The headline

> ⭐ **At byte-matched state, on the one designed family that survives protocol validation, at `d_in = 5`
> with 5–6 stored items and ~10-token streams over 3 seeds at CPU scale, no memory in this audit — neither
> the two rival test-time-dynamics families nor the CLU — beats a zero-extra-byte reader of a *raw* table
> holding the same bytes: 0 of 5 rival arms (margins −0.188 to −0.424) ⟦F3⟧ and the CLU (−0.3180 ± 0.0804).**

The scale qualifiers in that sentence are not decoration. They are the claim's actual extent.

Two supporting facts, both frozen:

- The CLU's dividend over its own matched-byte launder is **−0.0789** — i.e. the launder is *better*. At
  the separable anchor configuration the dividend is **exactly 0.0000** with disagreement mass `D = 0`,
  which §3.3 shows is structural rather than a null.
- An earlier round of the same substitute audit on this store went **0-for-4**: a +0 B substitute of the
  launder's own table matched or beat the memory on every family (insertion order 0.776 vs 0.302; echo
  1.0000 vs −0.180). *"Beat your own launder"* and *"be the best reader of your own bytes"* are two
  different bars, and only the second is a result anyone should build on.

## 4.3 Where test-time dynamics *does* pay, and against what

We pre-registered two falsification conditions on this paper's own thesis and we adjudicate both here.

**"The finding inverts" — does NOT fire in the strong form; DOES fire in the weak form, and we state the
weak form plainly rather than re-framing.**

- **Weak form, measured:** against the **arg-min** control read through each memory's own projections,
  **`gdn` (+1.02 ⟦F3⟧) and `gdn2` (+0.88 ⟦F3⟧) show large positive dividends while the CLU shows
  −0.0789.** In that reading, **test-time dynamics pays for the delta-rule family and does not pay for
  ours.** That sentence is true as measured, it is in this paper, and we do not soften it.
- **Strong form, measured:** **0 of 5 rival arms beat the raw-metric +0 B table at the same bytes**
  ⟦F3⟧, and neither does the CLU (−0.3180 ± 0.0804). So the audit is not a different paper — **and it is
  not a different paper only because the distinction that decides it was registered before measurement**
  (§2.4). Had we added the raw control after seeing the projected result, it would have been
  indistinguishable from a re-frame. We regard the ordering, not the outcome, as the credible part.

**"Not apples-to-apples" — does NOT fire, with a split we never blur.** A byte-matched table is definable
without an arbitrary modelling choice for the arms we adjudicated **by measurement**: `ttt_linear`,
`ttt_mlp`, `deltanet`, `gdn`, `gdn2` — five state types, each with an explicit float state and an explicit
`(θ_K x, θ_V x)` stream, so `n_rows = ⌊state_floats/(d_k+d_v)⌋` is *forced*, not chosen.
⚠ **Three of the families named in our protocol table — Mamba-2, Sparse Delta Memory, and Titans — were
adjudicated from their published equations only, never measured**, and we never present the two on the
same footing. A verdict over all five *named families* requires Mamba-2 and SDM to be run, and this paper
does not run them (§6).

**Metric-nativeness, argued at equation level and then measured** (evidence; margins ⟦F3⟧):

| arm | verdict | equation-level argument | measured vs the raw table |
|---|---|---|---|
| DeltaNet | metric-native | `o = Sᵀq` with `S` a sum of outer products ⇒ `o = Σ_s z_s(k_s·q)`, a linear kernel smoother; `q, k` are L2-normalised, so `argmin‖q−k‖ ≡ argmax q·k` **exactly**. The only non-metric ingredient is the scalar `β_t` | loses by **0.257** ⟦F3⟧ |
| Gated DeltaNet | metric-native | adds a scalar decay `α_t` — a scalar reweighting | loses by **0.188** ⟦F3⟧ |
| Gated DeltaNet-2 | metric-native | erase `b_t` and write `w_t` become channel-wise, so the effective metric is a learned diagonal, token-dependent Mahalanobis shape rather than the identity. It is still a metric — and the table it is audited against is entitled to the same shape, which is why the +0 B readers run on the same projected keys | loses by **0.188** ⟦F3⟧ |
| TTT-Linear | metric-native | with gradients taken at `W₀` the read is `W₀q − 2η Σ_s(W₀k_s − v_s)(k_s·q)`; the paper's own Theorem 2 makes this general — the nonparametric TTT learner *is* a Nadaraya–Watson estimator with kernel `exp((θ_K x)ᵀθ_Q x')` | loses by **0.247** ⟦F3⟧ |
| TTT-MLP | **weakly** metric-native | the GELU nonlinearity means the read is *not* a kernel average of stored values, so metric-nativeness does **not** close at equation level — the only arm here for which it does not | loses by **0.424** ⟦F3⟧, and is **not rescued** |

⭐ This is the favourable line, and it belongs to the field rather than to us: **every rival family we
surveyed is metric-native or weakly so.** The matched-bytes ceiling is not our idiosyncratic problem; it
is a property of the family. We report it with numbers attached rather than as an assertion.

## 4.4 The two-sided ledger, and the asymmetry that is itself a finding

| arm | F1 parameters | F2 state | state/param | own table bytes | state/table |
|---|---|---|---|---|---|
| ttt_linear (`d=29`) | 5592 B (incl. `W₀` = 870 floats) | **5220 B** = `d² + 16d` | 0.933 | 5104 B | **1.023** |
| ttt_mlp (`d=12`) | 5736 B (incl. `W₀` = 1164 floats) | **5376 B** = `8d² + 16d` | 0.937 | 5376 B | **1.000** |
| deltanet / gdn / gdn2 (`d=36`) | 9956 B (incl. `S₀` = 1296 floats) | **5184 B** = `n_head·d_k·d_v` | 0.521 | 5184 B | **1.000** |
| **CLU** | **5376 B** = `V_θ` init, 1344 floats | **5200 B** = 1300 floats, **measured** | **0.967** | **100 B** | ⛔ **52.0×** |

⭐ **Every rival's state *can* be byte-matched to its own table (1.000–1.023). Ours provably cannot.**
Theorem T1's floor makes matched bytes unreachable under a per-item group-masked write, and the audited
cell sits at **54.56×** (full-to-launder). This is the sharpest single statement the ledger produces, it
runs *against* our own system, and it is the reason every byte or dividend claim in this paper carries the
**≥2.20× (≥2.40× with a spectator dimension)** ratio caveat.

The ledger is enforced structurally, not by convention: each cell asserts, **as integers**,
`full == 4[N_at(D+2) + K·d]` and `launder == 4K(d+m)`. At `N_at = 192, D = 5, K = 5, d = 4, m = 1`
(7 floats/atom, `A = 38.4`): `full = 4(192·7 + 5·4) = 5456 B`, `launder = 4·5·5 = 100 B`,
`ratio = 54.56` — digit-for-digit the recorded ledger. A drifted store raises (tested).

⛔ **No cell measured in this audit is a byte-matched dividend.** The minimum byte ratio measured anywhere
in this work is **17.11×**.

## 4.5 The byte-frontier column (labelled at every appearance)

> ⛔ **`overload@load1x_shipped` is a BYTE-FRONTIER COLUMN, never a dividend family and never reader
> discrimination.** Its table launder sits at the metric's exact maximum (1.0000, 3/3 seeds). Its
> defensibility rests entirely on the declared secondary reading `S_excl = 0.6500` (§2.5).

**Frozen, the CLU's banked accuracy-versus-bytes curve, reused and not re-measured:** decode
**0.972 → 0.097** as the store-to-table byte ratio falls **478× → 2.28×**. This is the curve, and we
quote the curve rather than either endpoint.

**Rivals measured beside it — and the column is NON-INFORMATIVE for them.** ⟦F3⟧ Every rival point is
within noise of its own blank-store control: **0 of 5 rescued**. Under the rescue gate **no margin
against them here is quotable**, and we explicitly decline to draw the comparison the raw numbers invite.
⚠ **This is a NOT-RESCUED verdict, not a result about the rivals.** The full table is in Appendix H,
labelled, with the 5× under-training check that shows the arms get worse rather than better at a larger
budget.

## 4.6 What a table structurally cannot do, measured

*(Grade: the law is **verified** on designed geometries and then **measured as evidence** on the learned
store. The two are reported separately and their discrepancy is a finding.)*

A per-slot matched-bytes table computes `ŷ = ψ(x, r_{i(x),1..S})` for one query-dependent row selection
`i(x)`. Its dependence on the store therefore factors through finitely many selected rows:

> **Prop T5.4 (the shared-index bottleneck).** For fixed `x`, a row-selecting table's
> `∂ŷ/∂(any non-selected row) = 0` **exactly**. A continuous-latent store has no such factorisation: its
> acceleration is `−M⁻¹∇V` and `∇V` sums over **every** well, so deleting a stored item the query did not
> select moves every point of its trajectory.

We measured that coupling, and we measured what it costs to have it. Deleting the query's *second-nearest*
stored key (the row a table provably never reads) and dividing by deleting its *nearest* (the row it does
read), on the learned store, 3 seeds per radius:

| `ball_radius` | coverage | `sep` | fitted `s` | **`d/s`** (fitted ruler) | **measured coupling ± 2 SE** | ⛔ per-slot **table** | `λ_min` |
|---|---|---|---|---|---|---|---|
| 0.42 | **3/6** | 0.5481 | 0.482 | 1.10 | **0.814 ± 0.40** | **0 exactly** | 1.26 |
| 0.55 | 3/3 | 0.7402 | 0.412 | 1.71 | **0.344 ± 0.18** | **0 exactly** | 2.64 |
| 0.64 | 3/3 | 0.8614 | 0.400 | 2.05 | **0.226 ± 0.04** | **0 exactly** | 2.84 |
| 0.80 | 3/3 | 1.0767 | 0.385 | 2.68 | **0.0970 ± 0.02** | **0 exactly** | 3.03 |
| **1.00 (the audited cell)** | 3/3 | **1.3459** | 0.362 | **3.72** | **0.01534 ± 0.006** | **0 exactly** | 3.24 |
| 1.20 | 3/3 | 1.6211 | 0.362 | 4.41 | **1.55e-3 ± 2e-3** | **0 exactly** | 3.16 |

The table's third-party Δ is **`0.0` at every slot × every dropped row × every cell — float equality, not
a tolerance**, and it is *by construction* (Prop T5.4), never a win. The non-vacuous half is asserted
beside it: deleting the row the query *did* select moves the same table by a whole payload level.

**The law transfers to a learned multi-atom store, and it hands back the well width.** Fitting
`ln κ − ln(d/σ_q)` linearly in `d²` (slope `−1/2s²`):

| fit | slope | implied `s` | prefactor | **R²** | decades over the swept span |
|---|---|---|---|---|---|
| static ∇V ratio | −3.158 | **0.3979** | 0.379 | **0.9953** | **2.72** |
| dynamical slot coupling at `t = 1` | −3.155 | **0.3981** | 0.378 | 0.9952 | 2.72 |
| independent well-fit of the learned store | — | **0.4006** | — | — | — |

⭐ Three independent estimates of the learned well width agree to **0.7 %**. ⚠ **And this corrects one of
our own earlier statements by a large factor:** a prior estimate placed this store at `d/s ≈ 1.9` with an
`O(1)` coupling by taking the *admission gate's refusal radius* as if it were the achieved spacing. The
achieved separation is `sep = 1.346`, so the audited configuration runs at `d/s = 4.34` (atom-width ruler)
or `3.72` (fitted-width ruler) with a measured coupling of **1.53e-2** — a **45–52×** correction, in the
direction that makes the table *harder* to escape. The span across the whole admissibly-writable sweep is
**525× (2.72 decades)**, not the 1089× previously stated. Every `d/s` statement in this paper names its
ruler.

**The honest cost, stated where it hurts.** The coupling is `O(1)` **only where the store cannot reliably
be written**: at `d/s = 1.10` (fitted ruler) the write is admissible in **3 of 6 seeds** — the other three
have merged wells (`λ_min` = −0.53 / −2.13 / −1.11) — while every other radius is 3/3. And at the audited
configuration, the coupling clears a **matched blank-store delete control** by 2 SE on three seeds at most
slots (8/12 position slots, 11/12 momentum slots) but is **0/12 against the query's own launch-noise
floor**, two orders of magnitude below it. It is the store's, it is real, and it is unusable as a
read-out.

### ⭐ 4.6.1 The paragraph this section exists to support

> A per-slot matched-bytes table reproduces our slotted read at 37 of 38 slots, so on the answer channel
> the dynamics buy nothing a table cannot. There is exactly one thing such a table structurally **cannot**
> do: its output depends on the store only through the finitely many rows a query selects, so deleting a
> stored item the query did *not* select changes its answer by **exactly zero**. A CLU has no such
> factorisation — its acceleration is `−M⁻¹∇V` and `∇V` sums over every well — so the same deletion moves
> every slot of its trajectory. We measured that coupling, and we measured what it costs to have it. It
> obeys `κ(d) = (d/σ_q)·exp(−(d²−σ_q²)/2s²)` on our learned store with **R² = 0.995** across a 525×
> range, where `s = 0.40` is the store's own fitted well width (two independent estimates, 0.7 % apart).
> It is `O(1)` — 0.81 of the query's own item — only when neighbouring items sit **1.8 well-widths** apart,
> and at that spacing our admission machinery refuses the write in **half** of all seeds because the wells
> have merged (`λ_min < 0`). At the spacing our shipped configuration actually achieves (**4.3
> well-widths**) the coupling is **1.5e-2**: still the store's — it clears a matched blank-store delete
> control by 2 SE on three seeds at most slots — but **two orders of magnitude below the query's own
> launch noise**, i.e. unusable as a read-out. ⭐ **The one capability a table cannot imitate is
> exponentially suppressed by the very gate that makes the store safe: a store organised well enough to
> be safe is organised well enough to be a table.** That is not a defect of this implementation; it is a
> design identity.

⚠ **Bounding this section.** The set of couplings a row-selecting table cannot express is **not proven
exhaustive** — we proved that third-party attribution is inexpressible and that richer per-item slot
content is *not* a route (the slot vector is the image of a `(3d+1)`-dimensional launch map; measured
rank **13 = 3d+1** at `d = 4` for every `S ≥ 2`, and **4 = d** under the audited read, so extra slots add
zero per-item degrees of freedom and any such claim could only ever be cross-item). We did not prove
that no other coupling class exists.

## 4.7 Deletion, in the "and also" position

⚠ **Said first, by us: a table deletes exactly by construction.** Byte-exact deletion is a result only
for a *learned or superposed* store, where the item's contribution is not a row one can drop. That is the
entire reason the column below is reported and it is the entire reason it is not a headline.

**Frozen result, with its instrument and its conditions attached in the same sentence:** on the audited
store, deleting an item leaves the store **byte-equal to the never-written counterfactual on 3072 of 3072
compared bytes**, and a membership-inference attack against the deleted item reads **AUROC
0.5000 ± 0.0000** — at every tested load from 0.29× to 1.71× of capacity, under three explicit
conditions (`budget ≥ n_cells`, zero leak, depth-ordered eviction; an LRU eviction policy is a hard
error). ⛔ We do not describe this as *certified*, as *unlearning*, or as *exact deletion* without those
qualifiers: it is **verified byte-exactness under the stated conditions**, measured with the stated
instrument.

**And the trade it is locked into** (§3.2): the property that makes deletion byte-exact — private,
disjoint per-item parameters — is the same property that makes matched bytes unreachable. At matched
bytes, **at most 0.042 % of an item's parameter mass could remain byte-exactly deletable**.

**Neighbouring work, narrowing this claim.** A 2026 ICLR *workshop* paper (oral;
arXiv:2603.15033) publishes deletion-by-design on the same membership-inference instrument, reaching
MIA-AUROC ≈ 0.5 **by design** — but not exactly, with an average gap to retraining of **0.56 ± 0.21** —
in a memory-augmented transformer for image classification rather than a sequence memory. Our claim
survives that comparison **materially narrowed**, and it must be phrased on *verified byte-exactness*
rather than on priority. (We do not name the workshop: two sources disagree on its identity and the
arXiv record carries no venue.) Separately, recent work on unlearning as *counterfactual state
alignment* — the target being the optimiser state that would have arisen had the deleted samples never
been processed (arXiv:2605.17590) — is the same instrument in someone else's words, on a different
object, and we cite it as convergent framing.

⛔ **No rival family in this audit has a deletion verb at all**, so this column has no cross-family row
and we do not manufacture one.

---

# 5. Related work and positioning

## 5.1 The family being audited

**Test-time-trained memories.** TTT (Sun et al., 2024) makes the memory an inner learner with a learned,
sequence-shared initialisation `W₀` and a mini-batch size `b = 16` used *"for all experiments in this
paper"*; its Theorem 2 identifies the nonparametric TTT learner with a Nadaraya–Watson estimator, which is
why TTT-Linear closes metric-nativeness at equation level and TTT-MLP does not.
**Titans** (Behrouz, Zhong & Mirrokni, NeurIPS 2025) adds momentum and a forget gate to the associative
write; **ATLAS** (arXiv:2505.23735) and the Miras line (arXiv:2504.13173) continue it.
**Delta-rule linear attention:** DeltaNet (Yang et al., NeurIPS 2024), Gated DeltaNet
(Yang et al., ICLR 2025), and **Gated DeltaNet-2** (arXiv:2605.22791), which decouples a channel-wise
erase gate `b_t` from a channel-wise write gate `w_t`; we use GDN-2 as the delta-rule reference arm
because it supersedes GDN. **Sparse Delta Memory** (arXiv:2607.07386) routes writes into explicit slots
with a learned `M₀`. **Theory:** Wang, Shi & Fox (arXiv:2501.12352) unify linear attention, SSMs,
fast-weight programmers, online learners and softmax attention as test-time regression — analytically,
with no experiments and no baselines. They unify these mechanisms; this paper prices them.

**Evaluation conventions in the family.** Based (Arora et al., ICML 2024) owns the field's only explicit
*state-bytes-during-generation* axis — populated by six neural sequence mixers and by nothing else. MAD
(Poli et al., 2024) normalises to *"an iso-state and iso-parameter setting … a common total state
dimension of 4,096"* — across **neural** architectures only. Zoology (Arora et al., ICLR 2024) varies
state by hyperparameter. Sparse Delta Memory reports isoFLOP and isoParameter with a state-to-parameter
ratio column. HOLA (arXiv:2607.02303) is the nearest in-family relative: it adds a bounded exact KV cache
to a linear-attention state — a *semiparametric* test-time memory — and compares against a *matched*
recency-cache variant. We cite HOLA favourably: the field is already conceding that part of the payload
belongs in an exact store.

## 5.2 The positioning claim, and the exact sentence we are entitled to

We surveyed 14 candidate papers against a four-part definition of the control (non-parametric store ·
sized to the learned memory's **declared state bytes** · run on the same task · verdict reported) and
graded **0 HIT · 2 PARTIAL (both out-of-family) · 7 NEAR-MISS · 5 NO**. The sentence that survives is
scoped, dated, and checkable:

> **Across the modern neural sequence-memory family surveyed here — delta-rule and linear-attention
> models (DeltaNet, Gated DeltaNet, Gated DeltaNet-2), SSMs (Mamba-1/2/3), test-time-trained memories
> (TTT, Titans, ATLAS), explicit-slot memories (Sparse Delta Memory), semiparametric hybrids (HOLA), and
> the family's own recall and architecture-search benchmarks (Zoology, Based, MAD, RULER) — we find no
> paper, as of 31 July 2026, that sizes a *non-parametric* store (a table, kNN index, count-based model,
> or explicit (k, v) rows) to a learned memory's **declared state-byte budget**, runs it as a control on
> the same task, and reports the comparison. The nearest existing conventions are iso-state normalisation
> *across neural architectures only* (MAD §3.2), a state-bytes axis populated exclusively by neural
> sequence mixers (Based, App. E / Fig. 2), and isoFLOP/isoParameter reporting with a state-to-parameter
> ratio (Sparse Delta Memory). Budget-matched controls against non-learned alternatives are, by contrast,
> routine **outside** this family: at matched space in learned data structures (learned Bloom filters,
> learned indexes, learned sketches), and — concurrently with this work — as a token-matched recency
> window in LLM-agent memory evaluation. We therefore position this audit as **importing an established
> discipline into a family that has not adopted it**, not as inventing it.**

⛔ We do **not** claim the unscoped version (*"no published rival paper runs a non-parametric matched-byte
control"*). It quantifies over all papers and it is false: two families outside the survey run
recognisable versions of the control.

## 5.3 The ancestry we concede, in our own voice

- **Audit-at-equal-bits is standard outside this family.** Learned Bloom filters (Mitzenmacher, NeurIPS
  2018) and learned indexes with their benchmark suite (Kipf et al., 2019; and the accompanying
  PVLDB study) compare a learned structure against a classical one at matched space, adversarially and
  maturely — the SOSD verdict that a learned index is *"30–80× larger than B-trees"* on real data with
  *"4 orders-of-magnitude more time to build"* is the tone this audit is trying to import. What is absent
  there is any sequence, any test-time dynamics, and any state.
- **The substitute audit is not ours in general form.** It is the partial-input / trivial-baseline audit
  tradition: Poliak et al. (\*SEM 2018) train on hypotheses alone and beat the majority baseline on 6 of
  10 NLI datasets; Feng, Wallace & Boyd-Graber (ACL 2019) supply the caveat we carry with it. What we
  have found no publication of is the specific instantiation — a **+0 B reader over a learned memory's
  own stored bytes**, applied to a memory architecture, with the *omission* of that column named as a
  methodological failure.
- **A token-matched trivial control was published days before this work was filed** in LLM-agent memory
  evaluation (arXiv:2607.21962), reaching the same kind of verdict (*"small-history evaluations like this
  one may fail to separate the tested memory systems from a trivial strategy"*) at a **read-token**
  budget rather than a state-byte one, on agent-memory pipelines rather than test-time-dynamics memories.
- **Retrieval-augmented and datastore-priced work is adjacent but differently priced.** kNN-LM
  (Khandelwal et al., ICLR 2020) and its analysis (Xu, Alon & Neubig, ICML 2023) build a genuinely
  non-parametric datastore, but as an *augmentation* interpolated into an LM rather than as a byte-matched
  control — and its pure form is degenerate (λ = 1 gives unbounded perplexity because unretrieved targets
  receive zero mass). MassiveDS (Shao et al., NeurIPS 2024) prices a datastore against **training compute
  and parameters**, never against a learned memory's state bytes.

⭐ **What survives, and it is stronger than a monopoly claim.** Seven independent groups built the
*adjacent* instrument — a byte axis (Based), an iso-state normaliser (MAD), a state/parameter ratio
column (SDM), a matched trivial-policy control (HOLA), a compute-priced datastore (MassiveDS) — and none
of them closed the loop by putting a non-parametric store on the learned memory's own byte budget. **A
conceded ancestor is worth more than a contested monopoly.**

## 5.4 The theorem side

Our byte-floor result is an accounting identity about a specific store family, not a general capacity
bound; it belongs in the same room as, but does not compete with, capacity results for recall in
bounded-state sequence models (e.g. the Ω(N)-bit state lower bound for multi-query associative recall,
Arora et al., ICML 2024), which is one reason we treat metric-native synthetic recall tasks as
inadmissible primaries. For unlearning vocabulary we use ε-certified removal as stated inline in §3
Eq. (1) of Guo et al.; we note explicitly that that work has no numbered "Definition 1 / Definition 2" to
cite, a mis-citation we found in our own earlier notes.

---

# 6. Limitations

Every item here is load-bearing and none of it is buried.

**L1 — One-family thinness, verbatim.** *Two rival families audited against **one** surviving synthetic
family is a thin cross-family audit, and the rival rows cannot carry more weight than that.* Our own
protocol validation struck three of the four designed families as measuring the construction rather than
the memory, which is the right outcome for the instrument and a real cost to the paper's coverage. A
second independent dividend family, built to the rule in §2.5 (*the answer is provably not in the table*),
is the cheapest thing that would strengthen this work.

**L2 — A second, independent thinness.** On the dividend family, **two of five rival arms are NOT
RESCUED** ⟦F3⟧, and on the byte-frontier column **none of five is rescued**. A cross-family audit whose
rivals sit at their own blank-store floor on one of its two columns is thinner than the arm count
suggests, and no margin against a non-rescued arm appears anywhere in this paper.

**L3 — The launder's scope.** The matched-bytes launder tests whether *inference-time* dynamics beat a
table **given the organisation**: both arms inherit the same placement of the same content. This paper
measures that and only that. It is not evidence about any other stage of any of these systems, in either
direction.

**L4 — The tuning grid, and the direction of its bias.** ⟦F3⟧ The rival arms in this draft were tuned on
a **reduced** grid (Adam, 400 steps, three learning rates; TTT additionally two mini-batch sizes) rather
than the full `6 lr × 2 wd` standard. **The audit's finding is that rivals lose to their own byte-matched
tables, and under-tuning a rival produces exactly that finding — the bias runs toward our headline, which
is the dangerous direction.** A full-grid pass is registered and in flight, with the pre-commitment that
if tuning changes any outcome the paper's claim changes with it; `draft-v2` reports it. One
under-training explanation is already closed by measurement: a 5× outer budget makes the frontier arms
worse, not better, while the fit-to-eval gap is 31× (Appendix H).

**L5 — Measured versus reasoned families, never blurred.** Of the five rival families named in the
protocol table, **three were adjudicated by measurement** (`ttt_linear`, `ttt_mlp`, and the delta arms
sharing one state type) and **Mamba-2, Sparse Delta Memory and Titans were adjudicated from their
published equations alone.** Statements about "the family" in this paper mean the measured arms unless
they say otherwise. Mamba-2 and a GRU / sliding-window-attention arm are both cheap and both would convert
the apples-to-apples adjudication into a genuine five-of-five verdict.

**L6 — The byte-ratio caveat travels with every dividend or byte claim.** Under a per-item group-masked
write, matched bytes is unreachable: `ratio ≥ 2.20×` (`≥2.40×` with a spectator dimension), and the
audited cell sits at 54.56×. **No cell measured in this work is a byte-matched dividend; the minimum
ratio measured anywhere is 17.11×.** This caveat is not discharged by anything in this paper and it
attaches to every quantitative statement about bytes here.

**L7 — A table deletes exactly by construction.** Said before a referee says it (§4.7). Exact deletion is
a result only for a learned or superposed store, and the deletion column has no cross-family row because
no rival family here has a deletion verb.

**L8 — The conceded ancestry.** The audit-at-equal-bits discipline and the substitute audit are both
imported (§5.3). We claim the uniform state-byte protocol, the learned-initial-state rule, and the
finding — not the instrument's invention. And we carry Feng et al.'s converse caveat: a substitute audit
that a memory *passes* does not show the memory is doing real work; only a failed one is informative.

**L9 — What the theory does not derive.** Carried from the theorem set and never quoted as settled:
(i) the **prefactor** of the harness's own particle gradients is open — the toy law predicts `N e^{−C}`
where the measured harness value sits at `e^{−C}`, a factor ≈1200 apart; the structural claim is
unaffected. (ii) ⚠ **Naming `s` for a learned multi-atom well was an unsolved modelling question that
gates the transfer of every geometric domain statement** — `s/sep`, `d/s`, `s_max/σ_q`. §4.6 discharges it
*for this store only* (`s = 0.40`, two independent estimates 0.7 % apart); it is not discharged in general,
and every `d/s` statement here names its ruler. (iii) T5.4's coupling list is **not proven exhaustive**.
(iv) the third-party scaling law was verified on a two-well toy at zero launch momentum; with a live
launch-momentum head the path could in principle be steered toward non-selected wells and the suppression
could be weaker — untested here, and the suppression it would have to beat at the audited geometry is a
factor **65**, i.e. an effective halving of the distance to a non-selected well. (v) the soft-certificate
budget's outer edge, previously quoted as located, is **not located at all** under the corrected ruler
(`B ≥ 0.542` is unrefuted), and the corrected inradius proxy does not transfer to an anisotropic store.

**L10 — Protocol caveats with domains** (full statements in Appendix E): below `s/sep ≈ 0.15` the basin
boundary is **inertial** and no static proxy is valid (measured capture radius 1.306 against a midpoint of
1.000, a 21× miss for the static correction; the asymmetry is destroyed by damping, so it is
momentum-carried); **`λ_min > 0` does not certify a nonempty basin** (measured capture radius **0.000** at
`λ_min = +0.910`); `sep/2` is **not** a certified inradius and the corrected proxy is valid only inside
its four-condition domain; a truncated backpropagation depth governs `∂q_N/∂θ` **only where fixed-point
sensitivity dominates the transient** — in a `K`-item store the far-well parameters are exactly the
interference gradients, so truncation preserves the on-well gradient and destroys the crowding gradient.

**L11 — Weight class.** Every measured cell here runs at `d_in = 5`, 5–6 stored items, ~10-token streams,
float32, on CPU, 3 seeds. **Nothing here transfers to a language-model claim**, and no language-modelling
run was sized or attempted.

**L12 — Reproducibility incidents, disclosed.** One is on record: a per-arm fit key used a
process-salted hash, so two runs at identical seeds differed. It was found and fixed **before any number
in this work was recorded**; the reported run is post-fix and reproduces bit-identically. We report it
because it briefly produced numbers.

---

# 7. Conclusion

We asked whether a bounded-state memory's learned test-time dynamics buy anything over a non-parametric
table holding the same bytes, we built one protocol that makes the question answerable across memory
types, and we ran it on rival families and on our own system with the same columns for everybody.

At matched state bytes, on the one designed family that survives our own protocol validation, at CPU
scale with `d_in = 5`, 5–6 stored items and ~10-token streams over 3 seeds: **nothing in the audit beats
a zero-extra-byte reader of a raw table holding the same bytes** — not the rival test-time-dynamics
families ⟦F3⟧, and not ours (−0.3180 ± 0.0804). Against a weaker but natural control read through each
memory's own projections, **test-time dynamics does pay for the delta-rule family and does not pay for
ours** ⟦F3⟧ — and the only reason we are entitled to report the stronger control beside it is that both
were registered before measurement.

Two structural results bound what the answer could have been. For a store with one private parameter
group per item, matched bytes is unreachable by an accounting identity, and the byte floor is exactly the
price of one privately-deletable group per item — so **compression and byte-exact deletion are the same
trade**, at a computable exchange rate. And the one coupling a row-selecting table provably cannot
express — a read's dependence on a stored item the query did not select — obeys `exp(−½(d/s)²)` on our
learned store, which means it is exponentially suppressed by the same admission gate that keeps the store
writable: at the spacing our configuration achieves it is two orders of magnitude below the query's own
launch noise.

⭐ **A store organised well enough to be safe is organised well enough to be a table.** That is the
audit's conclusion, it is a design identity rather than a defect of one implementation, and it is where
this paper stops.

---
---

# Appendices

> **Appendix policy.** Main text carries main results only; every corollary, negative result, robustness
> check, full cell table and extra figure lives here, fully written. Nothing is omitted at drafting time.

## Appendix A — Flag provenance

Per project policy, every quantitative result travels with the configuration that produced it. Cells in
different sections must not be reproducible into an apparent contradiction.

### A.1 The audit run (§4.1, §4.3, §4.4, §4.5) — ⟦F3⟧ for all rival rows

| item | value |
|---|---|
| environment | JAX 0.9.0 / Equinox 0.13.4 / Optax 0.2.6; float32 throughout, on both sides of the ledger; no dependency re-resolution |
| seeds | **0, 1, 2** on every cell. Sample sd (`ddof = 1`), `SE = sd/√3` |
| fit-stream seeds (train/eval separation guard) | `seed + 101`, `seed + 102` — different sites, different payloads, **never the eval stream** |
| families run | `aggregate@base` (dividend) · `overload@load1x_shipped` (**frontier column only**). ⛔ `recency`, `manifold` NOT RUN (struck by protocol validation, §2.5) |
| task-harness non-default flags | `family=aggregate`, `capacity=6`, `consolidate_every=2`, staged admission on |
| store non-default flags | `capacity=6`, `budget=6`, `min_atoms=192`, `min_atoms_base=192`, `min_atoms_c=1.0`, staged admission on (the audited cell, unmodified) |
| rival outer loop ⟦F3⟧ | Adam, **400 steps**, `lr ∈ {1e-3, 3.16e-3, 1e-2}`, best-of-grid **on the fit split**; TTT arms additionally `b ∈ {1, 16}`. ⚠ **Reduced grid** vs the 6×2 standard — declared as a budget choice, **not** presented as compliance (§2.6, §6 L4) |
| chosen configs ⟦F3⟧ | seed 0: ttt_linear (d29, lr 1e-2, b16) · ttt_mlp (d12, lr 1e-2, b16) · delta arms (d36, lr 3.16e-3). Seeds 1/2: ttt_linear flips to (d36, lr 1e-2, **b1**). ⚠ `b` changes the head width because the mini-batch buffer is inside the state budget — declared, and required by the iso-state rule |
| iso-state budget | **1364 float32 = 5456 B**; head widths **29 / 12 / 36**, registered before any run and asserted in the test suite |
| byte law used | corrected `ratio = [A(D+2)+d]/(d+m)`; floors **2.20×** (`n_spec=0`) / **2.40×** (`n_spec=1`) |
| reproducibility | verified: re-running one cell at the recorded commit reproduces **every rival arm bit-identically**, same encoder hash |
| wall clock | 6 audit cells, 260 s total + frontier; whole run < 8 min |
| ⚠ disclosed incident | a per-rival fit key used a process-salted `hash()`; fixed to a stable index **before any reported number was recorded** (§6 L12) |

### A.2 The protocol-validation run (§2.5)

| item | value |
|---|---|
| seeds | **0, 1, 2** on all 12 cells; 0 degenerate, 0 errors; 2.2 min |
| statistics | mean ± sample sd (`ddof = 1`), `SE = sd/√3`; the attention leg uses the **paired** SE of `sub_s − attn_s` |
| anchors | `overload @ load1x_shipped` (478.2×, `atoms_per_item=341` ⇒ 2046 atoms, `n_offer=capacity=budget=6`); `aggregate`/`recency`/`manifold` @ base (192 atoms, `capacity=budget=6`, `consolidate_every=2`) |
| recency flags | pair-restricted index, staged lifetimes, `leak=0.06`; both coverages emitted |
| manifold flags | one spectator dimension (⇒ `dim=6`), deletion off, revisit off |
| store / write / read | masked local write, 300 steps Adam(3e-3, wd 1e-4), `σ_addr 0.25`, `σ_pay 0.6`, margin 0.15, barrier 0.2; read `dt 0.05`, `γ_address 0.05`, `γ_read 0.02`, 400 + 800 steps, learned-Newtonian kinetic mode |
| query law | `σ_q = 0.15` isotropic, payload tolerance 0.1; `sep/σ_q` **9.06** (overload, manifold) / **7.59** (aggregate, recency) |
| temperature / noise | none — deterministic read, `T = 0`, zero launch momentum, no Langevin step |
| attention arm | `softmax(q·kᵀ/(τ√d))`, value-weighted; `τ` grid-fitted on an **independent** draw of the same query law; ledgered at table bytes + 4 B. ⚠ It is a **table reader**: it consumes the launder's own `(key, payload)` table and never sees a trajectory |
| byte ledger | 57384/120 B (478.2×) · 5456/100 B (54.56×) ×2 · 6240/120 B (52.0×). ⛔ **No cell here is a byte-matched dividend**; min ratio measured anywhere **17.11×** |
| encoder ledger | identity launch, **0 B**, content-hash identical on all five arms of every cell, asserted in code |
| fidelity | per-seed diff against the recorded reference artifact: **exact on every shared arm** |

### A.3 The third-party-attribution sweep (§4.6)

| item | value |
|---|---|
| seeds | **{0, 1, 2}** at every radius, plus **{3, 4, 5}** at `ball_radius = 0.42` only — a bounded top-up because that point's coverage fell to 2/3; every seed reported with its admissibility |
| statistics | sample sd (`ddof=1`), `SE = sd/√n`, "clears" ⇔ `mean − 2·SE > 0`; coupling and gradient-ratio estimators are **medians** over queries (the denominator is a per-query gradient magnitude), means reported beside them |
| family / arm | `overload/load1x_shipped` — `atoms_per_item=341`, `min_atoms=2046`, `n_offer=capacity=budget=6`, staged admission on |
| swept axis | `ball_radius ∈ {0.42, 0.55, 0.64, 0.80, 1.00, 1.20}` with the admission override scaling with `R` (gate-to-geometry ratio invariant at 0.4266; `R = 1.00` is the audited cell). Nothing else changed |
| store / read | `addr_dim=4`, `payload_dim=1`, `dim=6`, `atom_width=0.3`, `confine=0.05`; `dt=0.05`, `γ_address=0.05`, `γ_read=0.02`, 400 + 800 steps, stride 8, `σ_q=0.15`, learned-Newtonian mode (measured `M = 0.99999905`), payload tolerance 0.1 |
| probe | retry disabled on every arm; slot grid `{0,1,2,3,4,6,8,12,16,20,24,29}` = steps `{1,9,17,25,33,49,65,97,129,161,193,233}`; **deletion = amplitude-zeroing of the item's atom group** (exact removal, nothing else moved) — ⛔ *not* the eviction path, which re-draws the freed group; selection = nearest / second-nearest stored key (the table's own row selection) |
| controls | blank-store delete control (the same zeroing on an unwritten group of the harness's own blank store) · launch-noise floor (independent `N(0, σ_q)` re-draw on the address block) · ⛔ per-slot matched-bytes table (**0 by construction**) |
| soft certificate | **off** in every cell |
| byte ledger | ⛔ **no byte-matched claim is made in this section.** The store's ratio at this arm is 478.20× |
| encoder | identity/embedded, **0 B**, identical on every arm |
| coverage | `d/s = 1.10` (fitted ruler): **3/6** (excluded: `λ_min` = −0.5272 / −2.1265 / −1.1137, all write-side, none silently filtered). All other radii **3/3**. Mean endpoint write loss on admissible cells 0.0021–0.0073 against a tolerance of 0.05 |

### A.4 The theorem checks (§3)

| item | value |
|---|---|
| implementation | pure numpy 2.4.1 / scipy 1.17.0, **float64** throughout; the damped velocity-Verlet integrator reproduced line-for-line (3 substeps then `p ← (1−γ)p`); no repository module imported |
| kinetic mode | Newtonian, `M = I` unless a mass sweep is named; `T = 0`, zero launch momentum, no Langevin, no training anywhere |
| constants | `dt = 0.05`; two-phase read `γ₁ = 0.05, N₁ = 400` then `γ₂ = 0.02, N₂ = 800`; `V = α‖q‖² − Σ A_i exp(−‖q−c_i‖²/2s²)`, `α = 0.05`, `s ∈ {0.20…0.90}`, `A ∈ {0.7…6}` |
| ⚠ instrument floor | 12 of 25 `{M, γ}` sensitivity cells sit on a float64 round-off floor at ~1e-16 (accumulated injections) and were **excluded before fitting**. Reporting them as physics would have refuted a correct law; this is the same failure mode a prior finite-difference check hit at 2.2e-12 |
| byte-law arithmetic | exact integers / `fractions.Fraction`, 0 ulp, over 28 recorded ledger cells |

## Appendix B — The protocol, in operational detail

**B.1 Byte classes.** `F1` = parameters shared across sequences (including any learned initial state).
`F2` = per-sequence state — **the audited budget**. `F4` = per-read transients (buffers, index sets,
top-k selections), which are not state but must be declared and must be granted to the launder on equal
terms. Every ledger's breakdown must sum to its total or the run raises.

**B.2 Building a byte-matched table for a memory.** For a memory with an explicit float state and an
explicit `(θ_K x, θ_V x)` stream, `n_rows = ⌊state_floats/(d_k + d_v)⌋` is **forced by the ledger, not
chosen**. Losslessness is checked per cell (`table_is_lossless`).

**B.3 The two mandatory table variants.** *Projected*: rows are the memory's own `(θ_K x, θ_V x)` pairs,
read through the memory's own output head. *Raw-metric*: rows hold the same bytes in the raw address /
payload space, read by the best +0 B reader. **Both are reported for every arm.** (§2.4)

**B.4 The +0 B reader set.** arg-min over keys · 2-NN mean · 2-NN inverse-distance-weighted · echo of the
query · insertion order · order-aware pair reader. "Zero extra bytes" means: no parameter, no threshold,
and no stored quantity beyond the launder's own table. A fitted scalar temperature costs 4 B and is
declared as such.

**B.5 The rescue gate, formally.** Arm `a` on cell `c` is RESCUED iff
`full(a,c) − blank(a,c) > 2·SE_paired`. A non-rescued arm's row is printed, and every margin against it
is suppressed. In the first pass this disqualified `ttt_mlp` and `deltanet` on the dividend family and all
five arms on the byte-frontier column ⟦F3⟧.

**B.6 The identical-encoder invariant.** All arms of a cell share `φ` and its byte cost; a content hash of
`(q₀, keys)` is asserted across all arms; a 1e-9 perturbation raises (tested).

**B.7 The ledger identity, as a blocking check.** `full == 4[N_at(D+2) + K·d]` and
`launder == 4K(d+m)`, asserted as **integers**, on every cell. A drifted store raises.

## Appendix C — Byte-floor theorem: full statement, assumptions, verification

Statement and assumptions as §3.1. Verification detail:

| check | result |
|---|---|
| byte decomposition `V/4 = N_at(D+2)`, `code/4 = Kd`, `launder/4 = K(d+m)` | ✅ **28/28 exact (integers)** |
| corrected law reproduces the recorded ledger ratio | ✅ **28/28 exact (rationals, 0 ulp)** |
| the previously published closed form reproduces it | ⛔ **24/28**; the four spectator-dimension cells miss by **+8.6667** (52.00 measured vs 43.33 published) |
| shell-atom surcharge | ✅ `52.00 → 58.40` exactly; `+1/(D+2) = +12.5 %` on the atom term |
| floors | ✅ 2.20 / 2.40 (Gaussian, `n_spec = 0/1`); 2.40 / 2.60 (shell) |

**The erratum's re-score, cell by cell.** Re-scored offline from the recorded artifact (the harness was
not re-run): **24 of 28 cells bitwise unchanged**; the four spectator-dimension cells change
43.3333 → 52.0000 (+8.6667, +20 %) with their printed floor 2.00× → 2.40×; the **measured minimum ratio
2.2824× is unchanged**. ⭐ The bug was invisible to the test suite because **no test exercised a spectator
dimension** — both byte tests passed `n_spectator = 0` literally and the end-to-end test was parametrised
over two families that both have `n_spec = 0`. A regression test now covers it.

## Appendix D — The dichotomy theorem: verification detail

| check | result |
|---|---|
| `Fix(T)` is `(γ, M)`-independent | settled `q*` spread over `γ ∈ {0.02,0.05,0.1,0.3} × M ∈ {0.5,1,2}` = **1.67e-15**; `max‖p*‖ = 1.5e-15`; `max‖∇V(q*)‖ = 1.3e-14` |
| `‖∂q_N/∂q₀‖ ∝ e^{−C}` | slope of `log₁₀|dq|` vs `C/ln10` = **−0.9941 over 143.9 decades** (25 cells); per-γ **−0.981 / −1.007**; prefactor `|dq|/e^{−C} ∈ [0.33, 8.3]` |
| `‖∂q_N/∂{log M, γ}‖ ∝ N e^{−C}` | raw per-γ slopes −0.90 / −0.91; after dividing by the derived `N` prefactor **−0.989 / −1.002 / −0.996 / −0.996** ⇒ exact to ±1 % |
| the audited read's own budget | `ρ(γ=0.05) = 0.974679`, `ρ(γ=0.02) = 0.989949` ⇒ **`C = 18.34`, `e^{−C} = 1.084e-8`** |
| toy at the audited two-phase schedule vs the learned harness | toy `‖∂q_N/∂q₀‖ = 4.19e-9 … 1.79e-8` (harness `‖∂L/∂φ‖ = 2.654e-9`); toy ratios `logM/q₀ = 10.6–35.7` (harness **3.29**), `γ/q₀ = 8.0–128.6` (harness **24.6**) |

**Sharp form of "almost everywhere".** `q*(q₀)` is *piecewise constant* — constant on each basin, with
jump discontinuities on the codimension-1 separatrices. The gradient is 0 a.e.; on the null set it is a
distribution, not a usable descent direction. This is the measured "staircase with a 1.7e7 cliff ratio",
derived.

**Scope of the inertial-mass half.** The dissolution of `M` is a statement about the **read**, not about
`V`. The settled endpoint is `M`-independent; a trajectory is not, because the kinetic gradient sets the
time parametrisation. Separately, the gauge orbit `(M, V, p₀) → (λM, λV, λp₀)` is exact **only** under a
learned-Newtonian kinetic mode (trajectory residual 2.52e-7); under an identity-mass mode it is **not a
gauge orbit at all** (residual 0.2505), and under a relativistic kinetic term it is broken at `O(1/c²)`
(residual 0.0274 at `c = 1`). The test must compare the **whole trajectory**, never the endpoint —
endpoint comparison passes vacuously once both runs settle (9.1e-2 → 3.6e-3 by doubling the step count
alone).

## Appendix E — Protocol caveats, as numbered propositions with domains

**E.1 Below `s/sep ≈ 0.15` the basin boundary is INERTIAL; no static proxy is valid there.** *Domain:* the
underdamped read used here (`γ ≲ 0.1`). At `s/sep = 0.10` with depth ratio 2, the deeper well's measured
capture radius is **1.306** against a midpoint of 1.000 (+30.6 %) while the static correction predicts
**+0.0144** — a 21× miss. ⭐ The mechanism is confirmed by its friction dependence: the same store gives
`r_deep = 1.308 / 1.306 / 1.002 / 1.000 / 1.000` at `γ = 0.02 / 0.05 / 0.20 / 0.50 / 0.90`. **The
asymmetry is destroyed by damping, so it is momentum-carried, not a property of `V`.** A static watershed
is the `γ → ∞` object.

**E.2 `λ_min > 0` does NOT certify a nonempty basin.** At `s/sep = 0.375` a shallow well is a genuine
minimum with **`λ_min = +0.910`** and a measured capture radius of **0.000** — every trajectory escapes
over the low barrier (also 0.000 at `λ_min = +0.388`). An independent reproduction on a task-like store
found two of six sites with a measured capture radius of **exactly 0.000 while `λ_min = +1.43 > 0`**. Any
site-level certificate needs a **measured capture-radius leg**; a spectral one is insufficient.

**E.3 `sep/2` is NOT a certified inradius, and a prior "violation" claim is RETIRED.** A bound of the form
`D ≤ U` is a theorem **under a certified ball**; the earlier computation took `U` from `sep/2` on stores
whose sites *were not minima* — all seven cells with `D/U > 4` have `λ_min ∈ [−1.199, −0.372]`, so their
certified radius was 0 and `U` was measured against a certificate that did not exist. A corrected proxy
`r_i ≈ min_j[d_ij/2 + ln(A_i/A_j)/(d_ij/s² − 4/d_ij)]` is 14.55× more accurate **inside its
four-condition domain only** (`λ_min > 0` ∧ `s/sep ∈ [0.15, 0.30]` ∧ `|δ_ij| ≤ 0.25 d_ij` ∧ a nonempty
basin under the operating friction); outside it, ≥29 % error below `s/sep = 0.15` and 100 % above 0.30.
⚠ **And it does not transfer to an anisotropic store**: on an axis-anisotropic store it is no better than
`sep/2` at three of four configurations (max |err| 0.098 vs 0.097 · 0.260 vs 0.248 · 0.280 vs 0.256) and
better only at the widest (0.354 vs 0.650).

**E.4 Truncated backpropagation depth governs `∂q_N/∂θ` only where fixed-point sensitivity dominates the
transient.** With `k*(ε) = ln(1/ε)/ln(1/ρ)`: for the settled well's own depth the transient/fixed-point
sensitivity ratio is 64× and truncation error is 2.5e-5 at `k = 270` (holds); for a far well the ratio is
27 396× and the error is **0.448, flat in `k`** — wrong by 456×. ⚠ In a `K`-item store the far-well
parameters are exactly the **interference** gradients, so truncation preserves the on-well gradient and
destroys the crowding gradient.

**E.5 A soft-mode lifetime is floored by the confinement, not by a tilt.** The confinement floors the soft
mode at `2α`, so a payload lifetime is capped at `τ_max = Γ/2α = 4.0` rather than growing as `1/ε`; the
damped-mode floor `τ = 2/Γ = 5.0` time units is confirmed (measured 4.23–6.58) and the above-knee slope is
**0**. The `1/ε` branch is unreachable because `2α` floors the soft eigenvalue 2.5× above the knee, and
lowering `α` breaks the write. ⚠ Every lifetime statement carries the `2α` coercivity coupling. And a
designed near-degeneracy does **not** survive superposition on a learned store: a group's atom-centre
spread is 1.19–1.95× the designed shell radius, the written site's vacuum residual is 0.140–0.343 against
a random-orientation baseline of 0.167 (i.e. at or worse than random), and a tilt **monotonically
reduces** `λ_min` (+0.0994 → −8.28), refuted in sign on two independent implementations and every family.
The geometric ruling survives — *be exactly flat architecturally or comfortably massive* — for **one**
designed atom, and only there.

**E.6 The soft-certificate budget's outer edge is not located.** Reproducing the grid that set it: under
the **measured capture radius**, `D ≤ U` holds at **all four** configurations including the one that set
the edge (`ρ_ex` 7.937 → **0.794**), so **`B ≥ 0.542` is unrefuted**; and the corrected proxy is **outside
its own validity domain at 3 of 4 configurations, including both that located the edge**.

## Appendix F — Prop D2a: the full drop map and the disagreement-mass instrument

Full table in §3.3. Two additional notes. (i) The equal-depth case is *exact* — the boundary is the
fixed-point set of the reflection exchanging the two wells; measured offset **2.92e-8** against a 2e-3
bar. (ii) The registered form of the disagreement-mass law was **refuted as registered** — the registered
formula computed the bisector-*crossing* rate, which is not a disagreement (both labels flip together);
`D = 0.00000` at every `sep/σ_q ∈ [2, 10]` with n = 4000 per cell. The corrected law — `D` is the query
mass **between the two boundaries** — was then verified at measured/predicted **1.141** over 7 cells with
predicted `D > 0.005`. We report the refutation because the corrected mechanism is what the instrument
actually measures.

## Appendix G — The measured write footprint (contribution 2, detail)

Diffing `V_θ` before and after the stream on the audited cell: **192/192 atom centres moved** (960
floats); **160/192 widths and amplitudes moved** (160 + 160); plus 20 codebook floats — **1300 floats =
5200 B**. Fully explained: 5 written slots × 32 atoms = 160 atoms move all three leaves, while the sixth
(free) slot is re-drawn by the allocator, so its 32 centres change while its widths and amplitudes are
re-set to their initial constants and register no change. Benign — but it is a measurement rather than an
inference, and **anyone reasoning about write locality from the write mask alone will get it wrong**.
This is also why our registered prediction for the state/parameter ratio (0.848) had the right direction
and the wrong magnitude (measured 0.967).

## Appendix H — The byte-frontier column, in full, and the under-training check

> ⛔ **BYTE-FRONTIER COLUMN — not a dividend family, never a headline.** Its defensibility is the declared
> secondary reading `S_excl = 0.6500`. ⚠ Every rival row here is **NOT RESCUED** (0 of 5), so **no margin
> against any of them is quotable**, and we do not draw a curve from it.

⟦F3⟧ for all rival rows. 3 seeds, 24 queries, 7 stream tokens, 6 live items, chance = 0.1667.

| rival | `d_head` | state B | table rows | lossless | **full** | own table | blank |
|---|---|---|---|---|---|---|---|
| gdn2 | 2 | 16 | 1 | no | 0.1389 ± 0.1002 | 0.1667 | 0.1806 |
| gdn2 | 4 | 64 | 2 | no | 0.0694 ± 0.0367 | 0.1944 | 0.0972 |
| gdn2 | 8 | 256 | 4 | no | 0.1944 ± 0.0845 | 0.1806 | 0.1528 |
| gdn2 | 16 | 1024 | 8 | **yes** | 0.2222 ± 0.0556 | 0.1111 | 0.1944 |
| gdn2 | 36 | 5184 | 18 | yes | 0.1667 ± 0.0241 | 0.2083 | 0.2083 |
| ttt_linear | 2 | 144 | 9 | yes | 0.0833 ± 0.0481 | 0.2222 | 0.0833 |
| ttt_linear | 8 | 768 | 12 | yes | 0.3333 ± 0.0722 | 0.1667 | 0.1667 |
| ttt_linear | 36 | 7488 | 26 | yes | 0.0694 ± 0.0501 | 0.1111 | 0.1111 |
| **CLU** (banked, frozen) | — | 57384 | — | — | **0.9722** | **1.0000** | 0.1667 |

**Under-training check, run before declaring the column non-informative.** At **5×** the outer budget
(2000 steps vs 400), `gdn2` goes 0.0417 → 0.0000 and `ttt_linear` 0.2083 → 0.1250 — *worse* — while
`ttt_linear`'s **fit-split** loss reaches MAE **0.024** against an eval MAE of **0.75** (a **31×** gap).
It is a generalisation failure across item geometries, forced by the guard that outer parameters never
see the eval stream's items. The payload alphabet is spaced 0.4 apart, so an eval MAE of 0.58–0.75 decodes
at chance **by arithmetic**.

**The CLU's banked frontier curve** (frozen, reused not re-measured): decode **0.972 → 0.097** as the
store-to-table byte ratio falls **478× → 2.28×**. We quote the curve, not the endpoint.

## Appendix I — Pre-registration scorecards

Predictions were registered before the corresponding measurement in each case. We print the misses.

### I.1 The audit's rival predictions ⟦F3⟧

| # | registered | measured | verdict |
|---|---|---|---|
| P2 (measured half) | ≥2 of the 3 measured (k,v)-state families lose to their own byte-matched table's strongest +0 B reader | **1 of 3** (deltanet only) | ⛔ **refuted as registered.** Second reading, pre-committed and printed beside it: **3 of 3** lose to the **raw-metric** +0 B table at the same bytes. ⚠ First half only — the real-data half is untested here. ⚠ Mamba-2 and SDM were adjudicated from equations only |
| P3 | the two function-valued memories show the largest positive dividend | — | ⛔ **NOT-RUN** (no Titans arm ⇒ the pair cannot be formed). **NOT-RUN is not refuted.** TTT-MLP alone is a single-arm datum and it is not rescued |
| P5 | the launder transfers to all five rival state types; 0 of 5 failures | 5 of 5 carry a byte-matched table; 0 failures | ✅ supported |
| R1 | rival arg-min launder ≈ −0.42, band [−0.55, −0.25] | −0.4245 · −0.4108 (TTT) · −0.6658 · −1.4158 · −1.2735 (delta) | ◐ 2 of 5 in band; the 3 delta arms fall far below it — the finding of §2.4 |
| R2/R3 | rival `full` ≈ −0.15, band [−0.30, −0.05] | −0.40 to −0.63 | ⛔ **out of band, all 5.** We over-predicted the rivals: a byte-matched linear memory at `d_in = 5` does not interpolate as well as its own 2-NN reader does |
| R4 | dividend vs own **arg-min** table = +0.27, band [+0.05, +0.45] | mean **+0.3691** | ✅ in band |
| R5 | signed **+0 B** margin = −0.02, band [−0.15, +0.08], **≥3 of 5 ≤ 0** | mean **−0.0392; 3 of 5 ≤ 0** | ✅✅ in band and the count exact |
| R5-raw | *(not banded; added after the band was fixed; reported as a second reading, never substituted)* | −0.1880 … −0.4242 | **5 of 5 ≤ 0** |
| R6 | rival blank ≈ −0.75, band [−1.2, −0.45] | −0.57 … −1.66 | ◐ 4 of 5 in band |
| R7 | same-keys null ≈ −0.45, band [−0.60, −0.30] | −0.43 … −1.22 | ◐ 2 of 5 in band; the delta arms' null tracks their launder, as their equations imply |
| R8 | rival state / own-table bytes = 1.00, band [1.00, 1.06] | 1.000 · 1.000 · 1.000 · 1.000 · 1.023 | ✅ 5 of 5 |
| R9 | `table_is_lossless` true 5 of 5 | true 5 of 5 | ✅ |
| R10 | state bytes < parameter bytes under the learned-init rule; ratio 0.848 | state 5200 B < param 5376 B, **ratio 0.967** | ◐ direction confirmed, **magnitude wrong** — the free slot's re-draw was not anticipated (Appendix G) |
| head widths | 29 / 12 / 36 from the 1364-float budget | 29 / 12 / 36 | ✅ exact, asserted in the test suite |
| frontier knee | at `d_head ≈ 8–10` (19 stream tokens) | knee at `d = 16` for gdn2; ttt_linear lossless from `d = 2` | ⛔ **wrong, and the error is ours:** derived from the wrong offer count (18 vs the 6 the frontier arm sets ⇒ 7 stream tokens) |

**Score: 6 confirmed (2 exact) · 4 partial · 3 wrong-direction · 2 NOT-RUN.** The two sharpest predictions
(R4's band; R5's band **and** its count) both survived; the two we got most wrong are reported as
findings.

### I.2 The theorem set

**Score: 8 ✅ (4 exact) · 2 ◐ · 3 ⛔.** All three refutations produced a corrected law that was then
verified: the byte law's published closed form (Appendix C), the disagreement-mass mechanism
(Appendix F), and the `{M, γ}` sensitivity prefactor (Appendix D). One correction changed an instrument's
specification rather than a number.

### I.3 The third-party-attribution sweep

**Score: 10 ✅ (4 sharper than registered) · 2 ◐ · 3 ⛔.** All three refutations produced a corrected law
that was then verified. The three misses, printed:
(i) the registered span of **5.2 ± 1.5 decades** measured **2.72** — the closed form at the *measured*
`s = 0.398` predicts 2.72 to two decimals, and the earlier `atom_init_width` ruler overstated the decades
by **1.74×** (a 1.33× error in `s` is a 1.74× error in decades, because the exponent is `d²/2s²`);
(ii) the registered ≥3× decay from the first to the last slot at small `d/s` measured **1.2×** — at small
`d/s` **both** deletions saturate at `O(sep)`, so the ratio is flat, and the decay is a large-`d/s`
phenomenon (1.2 / 2.7 / 11.8 / 26.0 / 33.1 / 34.6 across the sweep);
(iii) the registered ballistic-with-free-fall prediction missed **high** at both checkpoints (measured
0.853–0.883 at step 9 versus a registered [0.35, 0.85]). ⭐ The *damping* half is confirmed to **3 %**;
the free-fall factor imported from a two-well toy **does not transfer** — the measured free-fall residual
is 0.97–1.01 at step 9 and only bites past step 25. Corrected law: `Δq(n)` is **damped ballistic to
within 3 % for `n ≤ 17`**.
Confirmed sharper than registered: the momentum/position ratio (**3.78–4.03** at step 9 against a
registered [1.5, 6]; **38.0** at step 1 against a mechanically forced 40; momentum peaks earlier than
position at **6 of 6** radii), and the table's third-party Δ being **exactly 0.0** at every slot × row ×
cell.

## Appendix J — Declared NOT-RUNs (never reported as nulls)

- **A Titans arm.** No official code; the chunk size is never given a numeric value; no seeds reported.
  An arm would be our reconstruction audited against our reconstruction's table. Its `2·|M_θ|` state
  convention stays ⚠ **UNPINNED — our reconstruction, captioned every time.** ⇒ **P3 is NOT-RUN, not
  refuted.**
- **A Sparse Delta Memory arm.** Its official implementation requires Torch ≥2.8 / Triton ≥3.4 / SM 80+
  hardware. Positioning only. ⛔ **Its Table 1 state/parameter ratios are quarantined** — two independent
  extractions disagree — and none is quoted anywhere in this paper.
- **Mamba-2, GRU, and sliding-window-attention arms.** Outside the ruled arm set for this pass. Both
  Mamba-2 and a GRU/SWA arm are cheap and would convert §4.3's apples-to-apples adjudication from
  "3 measured, 3 reasoned" into a genuine five-of-five verdict.
- **A deletion column for any rival.** No rival family has a deletion verb.
- **The `recency` and `manifold` families.** Struck by protocol validation (§2.5) as protocol-invalid;
  their numbers are not reported as memory results in either direction. ⛔ In particular, **no `recency`
  dividend — pre-fix or post-fix — is a null**: a +0 B reader of the table's own row order answers the
  restricted question at 1.0000 on 3/3 seeds, by construction.
- **The `overload` family at the base atom budget** (0/18 admissible, including the control arm).
- **A *trained* attention reader.** The attention arm here is a table reader with a grid-fitted scalar
  temperature.
- **A soft-certificate sweep over the violation budget** — one demonstration cell only.
- **A live-launch-momentum third-party probe** (§6 L9-iv): the one mechanism that could in principle beat
  the `exp(−½(d/s)²)` suppression. Not run, and its target is quantified rather than guessed — the
  suppression it would have to beat at the audited geometry is a factor **65**, i.e. `Δ(d²)/2s² = 4.17`
  in the exponent, an effective halving of the distance to a non-selected well.
- **A `d/s` sweep by varying the atom width** instead of the geometry (moving the atom width would move
  the write's expressivity at the same time).
- **The eviction-path deletion as a robustness arm** (eviction *re-draws* the freed group, so its Δ would
  not be the item's).
- **Any real-data or language-modelling leg.** Not sized, not attempted (§6 L11).
- **Any change to a shipped default** anywhere in this work.

## Appendix K — Figure specifications

**Figure 1 (headline).** Signed **+0 B raw-metric margin** per audited arm, horizontal bars, zero line
drawn, rescue-gate-failing arms hatched and labelled NOT RESCUED, the CLU bar in a distinct fill. Error
bars = ±1 SE over 3 seeds. Caption must carry: family, byte budget, seeds, and the scale qualifier.
⟦F3⟧ rival bars — not rendered in this draft.

**Figure 2.** Two-sided byte ledger, stacked bars per arm: F1 parameters (with the learned-initial-state
component hatched) and F2 state, with each arm's own-table byte count marked as a tick. The CLU's
54.56× excursion is drawn on a broken axis and labelled as unreachable-by-construction (Theorem T1).

**Figure 3.** Third-party attribution: `κ` versus `d/s` on log-linear axes, both rulers plotted, the
fitted `exp(−½(d/s)²)` line with R² annotated, the per-slot table's exactly-zero drawn on the axis, the
audited configuration marked, and the region where the write is inadmissible (`λ_min < 0` on ≥1 seed)
shaded and labelled.

**Figure 4 (appendix).** Protocol validation: `S(f)` per family as a bar chart against the saturation
threshold, with the substitute's byte cost annotated on each bar and the full-attention reader overlaid.

**Figure 5 (appendix).** The CLU's accuracy-versus-bytes frontier curve (decode 0.972 → 0.097 as the ratio
falls 478× → 2.28×), with the rival points **omitted** and a caption stating why (not rescued).

## Appendix L — Negative results and refutations recorded by this work

Per project policy, negatives are documented and never dropped.

1. **The audit's own headline is negative for our system**: dividend −0.0789 against its own launder,
   +0 B margin −0.3180 ± 0.0804, and an earlier round of the same audit went 0-for-4.
2. **Three of four designed task families are struck as protocol-invalid** (§2.5) — including two where
   the memory reads *below* a ≤4 B substitute, and one where it reads below its own blank store.
3. **The registered byte-matched control for weight-valued memories is a weak control** (§2.4) — a
   refutation of our own pre-registered protocol text, costing the table up to 1.208 ⟦F3⟧.
4. **The published closed form of the byte law is wrong in 4 of 28 cells** (Appendix C), conservatively.
5. **The registered disagreement-mass formula was refuted** and replaced (Appendix F).
6. **The registered `{M, γ}` sensitivity ratio band was missed by 4.8×** — the excess is exactly the
   `O(N)` injection prefactor (Appendix D).
7. **The registered third-party decade span, slot-decay ratio, and free-fall prefactor were all missed**
   (Appendix I.3); the ruler, not the law, was wrong.
8. **The registered frontier knee was wrong**, from our own miscount of stream tokens (Appendix I.1).
9. **Our own placement of this store on the `d/s` axis was wrong by 45–52×** (§4.6) — an admission gate's
   refusal radius was read as an achieved spacing. The correction moves *against* the interesting
   direction.
10. **A designed near-degeneracy does not survive superposition on a learned store** — a tilt reduces
    `λ_min` monotonically, refuted in sign on two independent implementations (Appendix E.5).
11. **A soft-certificate budget previously quoted as located is not located at all** under the corrected
    ruler (Appendix E.6).
12. **`λ_min > 0` does not certify a nonempty basin** (Appendix E.2) — measured capture radius 0.000 at
    `λ_min = +0.910`, reproduced independently on a task-like store at `λ_min = +1.43`.
13. **Slot count buys no per-item capacity** — the slot vector's measured rank is `3d+1` for every
    `S ≥ 2` and `d` under the audited read; any such claim could only ever be cross-item (§4.6).
14. **A prior attempt to learn an address by gradient descent through a settled-point read died at
    chance**, and §3.4 explains it with zero fitted parameters.

## Appendix M — Reproducibility and artifact notes

All measured cells are 3-seeded. Statistics conventions are stated per section (Appendix A). The audit
run reproduces bit-identically at its recorded commit, including the encoder hash. The
protocol-validation run reproduces the reference artifact exactly on every shared arm, per seed. The
attribution sweep's per-radius aggregates recompute from the merged artifact to the last digit on every
3/3 point. One reproducibility incident is disclosed in §6 L12.

---

## ⛔ Open editorial items (delete before circulation)

1. **Number-freeze gate is OPEN** — every `⟦F3⟧` value is provisional (banner, §2.6, §6 L4).
2. **Title** is `[WORKING TITLE: …]` and **authorship** is `[AUTHORS PLACEHOLDER]`; both workshopped at
   the end, both blank in an anonymized build.
3. **Naming continuity** — this draft uses "CLU" with the continuity sentence *"the CLU, introduced as
   CHLU in Jawahar & Pierini (2026)"* in §1. Which paper carries the name's debut is a Hub/Head call.
4. **Figures are specified, not rendered** (Appendix K).
5. **One citation is single-sourced** and must be re-verified against the published text before printing:
   the Feng, Wallace & Boyd-Graber (2019) converse-caveat quotation (§2.4, §6 L8).
