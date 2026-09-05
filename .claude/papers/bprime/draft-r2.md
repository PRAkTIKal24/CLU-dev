# [WORKING TITLE: When Does Test-Time Dynamics Buy Anything Over a Table at Matched Bytes?]

**[AUTHORS PLACEHOLDER]**

---

> *Placeholders per project drafting policy: title is `[WORKING TITLE: …]`, authorship is
> `[AUTHORS PLACEHOLDER]`; both are workshopped at the end and both are blank in any anonymized build.*

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
(`d_in = 5`, 5–6 stored items, ~10-token streams; **nine seeds on every rival arm**, three on our own
store), and we report the result for every arm including our own.

Three findings. **(i)** At matched state bytes, on the one task family that survives our own protocol
validation, **no arm in the audit beats a zero-extra-byte reader of a *raw* table holding the same
bytes**: 0 of 5 rival arms (margins **−0.2592 … −0.4602**, every one at least 4.4 SE below zero over nine
seeds, under the full tuning grid our protocol specifies) and not the CLU either
(**−0.3180 ± 0.0804**). **(ii)** The control the field would naturally write is the wrong one: reading a
weight-valued memory's byte-matched table *through the memory's own projections* costs that table
**0.263 … 0.942 neg-MAE** against a raw-metric table at identical bytes (all five arms, all > 2 SE),
which manufactures apparent dividends of that same size for the delta-rule arms — dividends that vanish
under the raw control. Both controls were pre-registered before measurement; that ordering is the only
reason the second is a finding rather than a re-frame.
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
   ⭐ We also report two measured findings about the protocol *itself*, both of which cost us work and
   both of which any user of it needs (§2.6): **the rescue gate is underpowered below nine seeds** — its
   control's seed-to-seed spread exceeds the lift it gates, and at three seeds three legitimate
   configurations return three different rescued sets — and **best-of-grid tuning selected on the fit
   split makes a regulariser axis unselectable**, so a nominally 6 × 2 tuning grid is operationally 6 × 1
   unless the selection split is fixed too.
2. **The learned-initial-state rule** (§2.3), which we believe is the protocol's load-bearing convention:
   *for any memory with a learned initial state (TTT's `W₀`, an explicit-slot memory's `M₀`, our own
   `V_θ` initialisation), the initialisation is **parameters** and only the per-sequence deviation is
   **state**; both are declared.* Counting the init as state inflates a memory's apparent budget;
   counting the deviation as parameters launders it. **We apply the rule to our own store in the same
   table**, and it costs us: our measured per-stream deviation is 5200 B against 5376 B of
   initialisation, a state/parameter ratio of 0.967 (§4.4).
3. **A methodological finding on the rival side of the audit** (§2.4, §4.3): the byte-matched table one
   would naturally build for a weight-valued memory — the table of the memory's own projected
   `(θ_K x, θ_V x)` pairs — is **not a neutral control**. It costs the table **0.263 … 0.942** (nine
   seeds, all five arms, all > 2 SE) against a raw-metric table at identical bytes, because those
   projections are trained for a recurrence rather than for a table, and because a single stored value is
   out of distribution for an output head trained on kernel-averaged reads. Running only that control
   publishes dividends that do not exist.
4. **Three results about what matched bytes can even mean for a per-item store** (§3): the byte-floor
   identity `ratio = [A(D+2)+d]/(d+m) ≥ 2.20×`; its corollary that **compression and byte-exact deletion
   are the same trade** with exchange rate `dp/dr = (d+m)/[(D+2)A_tot]`; and a dichotomy theorem showing
   that a **settled-point read is untrainable end-to-end in both directions** because the fixed-point
   equation contains no transient parameter.
5. **The measurements** (§4): the rival rows at nine seeds under the full tuning grid, the CLU column,
   the byte-frontier curve, and the one coupling a row-selecting table structurally cannot express,
   measured on a learned store.
6. **The negative, stated as the finding it is** (§4.2, §7), and a limitations section (§6) that states
   the audit's own thinness before a referee has to.

**Headline figure — Figure 1.** *Signed +0 B margin against a raw-metric byte-matched table, one bar per
audited arm, with the zero line drawn and arms that do not clear their own blank-store control hatched.*
Every bar is below zero. This is the paper. (Specification in Appendix K; the figure is specified but not
rendered in this draft.)

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
**NOT RESCUED**, and **no margin against it is quotable**. This is applied per cell and reported
first-class, and it is not a formality: on the dividend family it disqualifies one of five arms outright
and leaves a second unresolved (§4.1.1), and on the byte-frontier column no arm clears it at all (§4.5).

⚠ **The gate has a power requirement, and we found it the hard way — it is stated here as part of the
protocol rather than as an afterthought.** The gate's control is the arm's *blank*-store read, which for
a memory with a learned initial state is that initialisation read through fitted projections; its
seed-to-seed spread is comparable to the lift being gated (one arm's blank reads over three seeds:
−0.962 / −2.634 / −1.390). **At three seeds the gate is a coin flip** — three legitimate configurations
of our own harness return three different rescued sets — and **at nine seeds it agrees across two
independent code paths**. ⇒ **Every rescue verdict in this paper is reported at nine seeds, and we
report no three-seed rescue verdict, including our own first pass's** (§2.6, Appendix I.1). Anyone
adopting this gate should run ≥ 9 seeds, pair the control per seed, or average the control over several
initialisations.

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
handicapped by **0.276 / 0.263 / 0.425 / 0.856 / 0.942 neg-MAE** (TTT-Linear · TTT-MLP · DeltaNet · GDN ·
GDN-2, nine seeds, every one more than 2 SE from zero) relative to a **raw-metric** table holding the same
bytes, for two reasons visible at equation level: (i) `θ_K, θ_V` are trained for the *recurrence*, not for
a table, so arg-min in the projected space is a worse metric than the raw address space; and (ii) a single
stored value decoded by the memory's own output head `θ_O` is **out of distribution** for `θ_O`, which was
trained on kernel-averaged reads `o = Σ_s v_s(k_s·q)`, not on isolated stored values.

A paper running only the projected control would report a positive dividend for each gated delta-rule arm
of the same order as that handicap — on those two arms the handicap (0.856 ± 0.091 and 0.942 ± 0.091)
exceeds the margin by which the arm loses to the raw table (0.2600 ± 0.0278 and 0.2592 ± 0.0292), so the
**sign** of the reported result is decided by which control is run. Both controls are therefore mandatory in this
protocol, and both are reported for every arm; neither is ever substituted for the other.

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

## 2.6 Tuning the rivals, and what tuning does not change

The audit's finding is *"rivals lose to their own byte-matched tables"*, and **under-tuning a rival
produces exactly that finding.** The bias runs *toward* our headline, which is the dangerous direction.
The obvious referee attack — *"you hobbled the competition on three learning rates"* — is therefore one
we close by measurement rather than by assertion, and the measurement is the reason this subsection
exists.

**What was run.** Every rival number in this paper comes from the **full tuning grid** the protocol
specifies: `lr ∈ {1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2} × wd ∈ {0, 0.1}` (TTT arms additionally
mini-batch `b ∈ {1, 16}`), i.e. **24 configurations per TTT arm and 12 per delta-rule arm**, best-of-grid
selected on the fit split, with the outer parameters never seeing the eval stream; Adam at `wd = 0`,
decoupled AdamW at `wd = 0.1`; 400 outer steps, plus a **2000-step (5×) re-check** on the sub-grid
containing every 400-step winner. Nine seeds per arm. ⚠ Two sub-clauses of the standard were **not**
adopted and are declared as deviations: its `β = (0.9, 0.98)` and its cosine decay, held fixed so that
exactly one variable moves between the two passes. **An earlier, reduced-grid first pass** (Adam, 400
steps, `lr ∈ {1e-3, 3.16e-3, 1e-2}`, three seeds) was run and declared at the time as a budget choice
rather than as compliance; it is disclosed in full, with a before/after table, in Appendix I.1. The
pre-commitment made before the full pass ran was that **if proper tuning changed any outcome, this
paper's claim would change with it**; what changed and what did not is recorded below and in Appendix I.1.

**What tuning changed: nothing in the headline, and the audit is stronger for having tried.** The
raw-table margin — the paper's load-bearing quantity — is negative on all five arms in every column we
ran: under the full grid, at 5× the outer budget, under a held-out selection rule, and at nine seeds. The
three measurements that close the attack:

- ⭐ **The grid's added points are never selected.** Under the incumbent (fit-split) selection rule,
  **0 of 45** (arm, seed) cells choose any learning rate below `1e-3`; the fit-loss surface is monotone
  improving in `lr` across the whole grid, so every point the wider grid adds is on the worse side of the
  optimum. The delta-rule arms' entire fit surface spans **0.0011–0.0155** — they are not
  learning-rate-limited at all.
- ⭐ **The tuning effect on the arms' scores is smaller than the reporting precision.** Holding the
  initialisation scheme fixed, widening the grid moves each arm's `full` read by
  **−0.0303 / +0.0018 / −0.0009 / +0.0006 / +0.0034** (TTT-Linear · TTT-MLP · DeltaNet · GDN · GDN-2):
  **≤ 0.031 on every arm and ≤ 0.004 on four of five.**
- ⭐ **More steps do not rescue an arm either, and one arm makes that unusually clean.** At 5× the outer
  budget TTT-MLP's fit-split loss falls by **64.1 %** (0.0839 → 0.0301) and its eval metric moves by
  **0.036 — less than one SE (± 0.0891)** — leaving it still short of clearing its own blank-store
  control and still losing to the raw table by **0.2609 ± 0.0903**. The delta-rule arms' fit loss moves by ≤ 0.1 % at 5× budget. The
  binding constraint is not budget: it is the **fit-to-eval generalisation gap across item geometries**
  forced by the guard that outer parameters never see the eval stream (§2.2, Appendix H).

⚠ **A disclosure that matters more than the tuning did.** Widening the grid required changing how each
configuration's initialisation is drawn: the first pass split one key sequentially across grid points, so
the initialisation of *every* configuration depended on the grid's length and order, and no grid could be
widened without re-drawing all of it. We changed to **one initialisation per (arm, seed, mini-batch),
shared across all `(lr, wd)`**, and priced the change by re-selecting the first pass's own sub-grid from
the new fits (the `lite control` column of Appendix I.1a). **That re-draw moved the arms by
−0.148 / +0.125 / +0.018 / −0.015 / −0.042 (same arm order) — between 4× and 35× more than the tuning
did.** Without the control column, the two effects would be inseparable and the tuning effect unreadable;
with it, the tuning effect is the one reported above. We report this because the larger of the two effects is the one
nobody would have asked about.

⭐ **A finding about the tuning standard itself, pre-registered as a finding-in-waiting.** Best-of-grid
selected on the *fit split* selects on the very objective being optimised, so a regulariser can
essentially never win: under that rule `wd = 0.1` is chosen only by fourth-decimal tie-breaks
(**12 of 45** cells) and a lower learning rate is chosen **never** (0 of 45). Under a **held-out**
selection stream the added points genuinely are chosen — **26 of 45** cells pick `lr < 1e-3` and
**24 of 45** pick `wd = 0.1`, and TTT-Linear's read improves from −0.6075 to −0.4461. ⇒ **On this
harness a nominally 6 × 2 grid is operationally 6 × 1 unless the selection split is fixed too.** We
report the held-out selection as a declared secondary (it moves the rescue verdict on two arms and no
raw-table margin: **−0.24 … −0.49**, still 5 of 5 losing), and we recommend that anyone specifying a
best-of-grid tuning standard specify the selection split with it.

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

**Grade: evidence** (learned store, trained rival arms), except where labelled otherwise. Sample sd with
`ddof = 1`, `SE = sd/√n`, identical `φ` asserted in code, byte ledger identity asserted as integers.
**Seed counts, stated once and never mixed silently:** every rival arm on the dividend family is
**n = 9** (seeds 0–8) under the full tuning grid; the CLU column is **n = 3** (seeds 0, 1, 2, banked and
reproduced digit-for-digit rather than re-measured); the byte-frontier column is **n = 3** for the rival
arms and was **not** re-run under the full grid (§4.5, Appendix J). Flag-provenance tables, one per run:
Appendix A.

## 4.1 The audit table

**Column status per `PREREG` §2.** Every closed cell is marked `have`; every open one is a **declared
NOT-RUN with its reason** and is never reported as a null.

**⚠ Seed-count discipline in this table.** The `+0 B` column is the paper's load-bearing column and is
quoted at **n = 9** for every rival arm. The projected-launder and same-keys-null columns exist for every
arm but their **nine-seed aggregate was not computed** — those two columns were aggregated at the first
pass's three seeds only, we mark them `have (n = 3)`, **no claim in this paper rests on them**, and their
three-seed values are printed in Appendix I.1c rather than here. The byte-ledger columns are exact
integers and are seed-independent.

| family / arm | matched-byte table launder | **+0 B** substitute (signed margin) | two-sided byte ledger | same-keys null | metric-native verdict | deletion probe | anytime / frontier |
|---|---|---|---|---|---|---|---|
| **CLU** (banked, reproduced digit-for-digit, `n = 3`) | have **−0.4472** | have, **−0.3180 ± 0.0804** | have **5456 B / 100 B, 54.56×** | have **−0.8175** | have | have (MIA-AUROC **0.5000 ± 0.0000**, byte-equal **3072/3072**) | have (banked curve) |
| **TTT-Linear** | have (`n = 3`, App. I.1c) | have, **−0.2213 ± 0.1062** (`n = 9`) | have F1 5592 B / F2 5220 B | have (`n = 3`) | have, metric-native | ⛔ NOT-RUN — no deletion verb exists in the family | frontier, **non-informative** |
| **TTT-MLP** | have (`n = 3`, App. I.1c) | have, **−0.2095 ± 0.0683** (`n = 9`) | have F1 5736 B / F2 5376 B | have (`n = 3`) | have, *weakly* metric-native | ⛔ NOT-RUN — as above | — |
| **DeltaNet** | have (`n = 3`, App. I.1c) | have, **−0.0172 ± 0.0263** (`n = 9`) | have F1 9956 B / F2 5184 B | have (`n = 3`) | have, metric-native | ⛔ NOT-RUN | — |
| **Gated DeltaNet** (ablation) | have (`n = 3`, App. I.1c) | have, **−0.0102 ± 0.0229** (`n = 9`) | have F1 9956 B / F2 5184 B | have (`n = 3`) | have, metric-native | ⛔ NOT-RUN | — |
| **Gated DeltaNet-2** (reference arm) | have (`n = 3`, App. I.1c) | have, **+0.0473 ± 0.0277** (`n = 9`) | have F1 9956 B / F2 5184 B | have (`n = 3`) | have, metric-native | ⛔ NOT-RUN | frontier, **non-informative** |
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

**Rival arms: nine seeds (0–8), full tuning grid. CLU: three seeds, banked.** The two right-hand columns
report the rescue gate under **both** code paths we have at nine seeds — the full-grid path and the first
pass's own path, re-run at nine seeds as a control — because their agreement (or disagreement) is what
makes a verdict quotable.

| arm | `d_head` | F1 param B | F2 state B | own table B | **full** | **+0 B margin** | ⭐ **raw-metric +0 B margin** | lift over own blank, full grid | lift over own blank, first-pass path | **RESCUED?** |
|---|---|---|---|---|---|---|---|---|---|---|
| ttt_linear | 29 | 5592 | 5220 | 5104 | −0.6075 ± 0.1096 | **−0.2213 ± 0.1062** | **−0.4602 ± 0.1038** | +0.093 ± 0.134 | +0.320 ± 0.083 | ⚠ **UNSTABLE** — clears under one path, not the other |
| ttt_mlp | 12 | 5736 | 5376 | 5376 | −0.5898 ± 0.0731 | **−0.2095 ± 0.0683** | **−0.4425 ± 0.0869** | −0.071 ± 0.090 | +0.093 ± 0.107 | ⛔ **NOT RESCUED** (no configuration) |
| deltanet | 36 | 9956 | 5184 | 5184 | −0.4205 ± 0.0299 | **−0.0172 ± 0.0263** | **−0.2732 ± 0.0395** | +0.294 ± 0.077 | +0.141 ± 0.046 | ✅ **RESCUED** |
| gdn | 36 | 9956 | 5184 | 5184 | −0.4073 ± 0.0120 | **−0.0102 ± 0.0229** | **−0.2600 ± 0.0278** | +0.880 ± 0.227 | +0.947 ± 0.149 | ✅ **RESCUED** |
| **gdn2** | 36 | 9956 | 5184 | 5184 | −0.4065 ± 0.0178 | **+0.0473 ± 0.0277** | **−0.2592 ± 0.0292** | +1.025 ± 0.329 | +1.384 ± 0.276 | ✅ **RESCUED** |
| **CLU** (`n = 3`) | — | 5376 | 5200 | 100 | **−0.5261 ± 0.0863** | **−0.3180 ± 0.0804** | (its own table is already raw) | — | — | ✅ (against blank −0.4221) |

⚠ **Read the two `n`s honestly.** The rival rows are nine-seed means and the CLU row is a three-seed
mean; the two are not pooled and no test is run across them. The CLU's own `+0 B` margin is
**−0.3180 ± 0.0804**, i.e. about four SE below zero on its own three seeds, so the statement *"the CLU
does not beat a raw table at its own bytes"* stands on the CLU column alone and does not borrow power
from the rival column. The projected-versus-raw distinction does not arise for the CLU — its launder is
already a raw table of `(key, payload)` rows — and its dividend over that launder is **−0.0789**, i.e.
the launder reads better than the store does.

**Rescue-gate verdicts, at nine seeds.** **RESCUED: `deltanet`, `gdn`, `gdn2`** — under both code paths.
⛔ **NOT RESCUED: `ttt_mlp`**, in every configuration we ran, so **no margin against TTT-MLP is quotable**
anywhere in this paper. ⚠ **UNSTABLE: `ttt_linear`** — it clears the gate under the first pass's
initialisation scheme (+0.320 ± 0.083) and does not under the full grid's (+0.093 ± 0.134); we print both
readings, treat the verdict as unresolved, and **quote no margin against TTT-Linear** either. Only the
three rescued arms enter the falsifier adjudications of §4.3 as arms we compare *against*.

⚠ **Two columns of the first pass are not re-aggregated at nine seeds** — the projected (arg-min) launder
and the same-keys null. Their three-seed values are in Appendix I.1c, labelled, and no claim in this paper
rests on them. The projected-versus-raw *gap*, which is load-bearing for §2.4, **is** available at nine
seeds and is reported there and in §4.3.

**Admissible-cell coverage, first-class** (recorded per seed on seeds 0–2; the six added seeds were run
for statistical power and their per-seed coverage is not separately tabulated). `aggregate`, seeds 0/1/2:
**58/72 · 66/80 · 55/80** admissible
queries (0.806 · 0.825 · 0.688) with **5/8** offered items admitted on all three seeds. The drops are the
family's own construction rule (a query whose target lands within payload tolerance of a stored payload is
dropped) — which is exactly what stops the arg-min launder from being accidentally right.

## 4.2 The headline

> ⭐ **At byte-matched state, on the one designed family that survives protocol validation, at `d_in = 5`
> with 5–6 stored items and ~10-token streams at CPU scale, no memory in this audit — neither the two
> rival test-time-dynamics families nor the CLU — beats a zero-extra-byte reader of a *raw* table holding
> the same bytes: 0 of 5 rival arms over nine seeds under the full tuning grid (margins
> **−0.2592 ± 0.0292 … −0.4602 ± 0.1038**, every one at least 4.4 SE below zero), and the CLU over three
> seeds (**−0.3180 ± 0.0804**).**

The scale qualifiers in that sentence are not decoration. They are the claim's actual extent.

**The headline is robust to every stress we put on it**, and the stresses were registered before they
were run (§2.6, Appendix I.1): the raw-table margin is negative on all five arms under the full
`6 lr × 2 wd` grid, at **5×** the outer budget (−0.2184 … −0.2630), under a **held-out** selection rule
that actually picks the widened grid's points (−0.24 … −0.49), and at nine seeds (−0.2592 … −0.4602).

⚠ **And the honest qualifier on the arm count.** Two of the five arms do not settle the rescue gate at
nine seeds — TTT-MLP clears its own blank-store control in no configuration we ran, and TTT-Linear clears
it under one initialisation scheme and not the other (§4.1.1) — so for those two the statement *"it loses
to a raw table at its own bytes"* is partly a statement about an arm that may not be reading its store at
all. **On the
three arms that do clear the gate the headline holds with more room: −0.2592 ± 0.0292, −0.2600 ± 0.0278
and −0.2732 ± 0.0395**, i.e. ≈ 8.9, ≈ 9.4 and ≈ 6.9 SE below zero. That restricted form is the one we
would defend if only one form could be.

Two supporting facts, both from the CLU column and both banked before this pass:

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
  **the two gated delta-rule arms show large positive dividends while the CLU shows −0.0789.** The size
  is set by how badly that control handicaps the table: **0.856 ± 0.091 (GDN) and 0.942 ± 0.091 (GDN-2)**
  at nine seeds, against raw-table margins of only 0.2600 ± 0.0278 and 0.2592 ± 0.0292 — so against the
  projected control both arms are comfortably positive, and the CLU is not. In that reading, **test-time
  dynamics pays for the delta-rule family and does not pay for ours.** That sentence is true as measured,
  it is in this paper, and we do not soften it. *(We quote the two components rather than the dividend
  itself because the dividend column was aggregated at the first pass's three seeds and not re-aggregated
  at nine; its sign at nine seeds is forced by the two numbers above.)*
- **Strong form, measured:** **0 of 5 rival arms beat the raw-metric +0 B table at the same bytes**
  (nine seeds, full grid), and neither does the CLU (−0.3180 ± 0.0804). So the audit is not a different
  paper — **and it is not a different paper only because the distinction that decides it was registered
  before measurement** (§2.4). Had we added the raw control after seeing the projected result, it would
  have been indistinguishable from a re-frame. We regard the ordering, not the outcome, as the credible
  part.

**"Not apples-to-apples" — does NOT fire, with a split we never blur.** A byte-matched table is definable
without an arbitrary modelling choice for the arms we adjudicated **by measurement**: `ttt_linear`,
`ttt_mlp`, `deltanet`, `gdn`, `gdn2` — five state types, each with an explicit float state and an explicit
`(θ_K x, θ_V x)` stream, so `n_rows = ⌊state_floats/(d_k+d_v)⌋` is *forced*, not chosen.
⚠ **Three of the families named in our protocol table — Mamba-2, Sparse Delta Memory, and Titans — were
adjudicated from their published equations only, never measured**, and we never present the two on the
same footing. A verdict over all five *named families* requires Mamba-2 and SDM to be run, and this paper
does not run them (§6).

**Metric-nativeness, argued at equation level and then measured** (evidence; nine seeds, full grid). ⚠ The
right-hand column is filled **only for arms that clear the rescue gate**; for the other two the row states
the gate verdict instead, and no margin is printed.

| arm | verdict | equation-level argument | measured vs the raw table |
|---|---|---|---|
| DeltaNet | metric-native | `o = Sᵀq` with `S` a sum of outer products ⇒ `o = Σ_s z_s(k_s·q)`, a linear kernel smoother; `q, k` are L2-normalised, so `argmin‖q−k‖ ≡ argmax q·k` **exactly**. The only non-metric ingredient is the scalar `β_t` | loses by **0.2732 ± 0.0395** |
| Gated DeltaNet | metric-native | adds a scalar decay `α_t` — a scalar reweighting | loses by **0.2600 ± 0.0278** |
| Gated DeltaNet-2 | metric-native | erase `b_t` and write `w_t` become channel-wise, so the effective metric is a learned diagonal, token-dependent Mahalanobis shape rather than the identity. It is still a metric — and the table it is audited against is entitled to the same shape, which is why the +0 B readers run on the same projected keys | loses by **0.2592 ± 0.0292** |
| TTT-Linear | metric-native | with gradients taken at `W₀` the read is `W₀q − 2η Σ_s(W₀k_s − v_s)(k_s·q)`; the paper's own Theorem 2 makes this general — the nonparametric TTT learner *is* a Nadaraya–Watson estimator with kernel `exp((θ_K x)ᵀθ_Q x')` | ⚠ rescue verdict **UNSTABLE** — no margin quoted |
| TTT-MLP | **weakly** metric-native | the GELU nonlinearity means the read is *not* a kernel average of stored values, so metric-nativeness does **not** close at equation level — the only arm here for which it does not | ⛔ **NOT RESCUED** — no margin quoted |

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

**The CLU's banked accuracy-versus-bytes curve, reused and not re-measured:** decode
**0.972 → 0.097** as the store-to-table byte ratio falls **478× → 2.28×**. This is the curve, and we
quote the curve rather than either endpoint.

**Rivals measured beside it — and the column is NON-INFORMATIVE for them.** No rival point on this column
separates from its own blank-store control at the **three** seeds run, and this column was **not** re-run
at nine seeds or under the full tuning grid (Appendix J). Since our own measurement of the rescue gate
shows it is underpowered at three seeds (§2.2), we do not report *"0 of 5 not rescued"* as a verdict
either: what we report is that **no margin on this column is quotable in either direction**, and we
explicitly decline to draw the comparison the raw numbers invite — including the flattering one. The full
table is in Appendix H, labelled, with the 5× under-training check that shows the arms get worse rather
than better at a larger budget.

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

**L2 — A second, independent thinness.** On the dividend family at nine seeds, **one of five rival arms
does not clear its own blank-store control in any configuration we ran (TTT-MLP) and a second clears it
under one initialisation scheme and not the other (TTT-Linear, printed UNSTABLE)**; on the byte-frontier
column no arm separates from its control at the three seeds run. **No margin against TTT-MLP or
TTT-Linear appears anywhere in this paper**, so the cross-family audit is carried by three arms
(DeltaNet, GDN, GDN-2) that share **one** state type. A cross-family audit whose arms sit at their own
blank-store floor on part of its grid is thinner than the arm count suggests, and we would rather say so
than count to five.

**L2a — The rescue gate is underpowered below nine seeds, and we found that out about our own first
pass.** The gate's control is a single initialisation draw read through fitted projections, and its
seed-to-seed spread exceeds the lift it gates (one arm's blank reads across three seeds:
−0.962 / −2.634 / −1.390). At three seeds, three legitimate configurations of our own harness return
three *different* rescued sets; at nine seeds two independent code paths agree on three of five arms and
disagree only on TTT-Linear. Consequences we accept and state: **every rescue verdict in this paper is a
nine-seed verdict, the three-seed verdicts of our first pass are withdrawn rather than quoted, and one
arm's verdict remains unresolved.** A pre-registered prediction of ours — that the rescue statuses would
be stable under re-tuning — was **refuted**, and this is the mechanism (Appendix I.1b, P4).

**L3 — The launder's scope.** The matched-bytes launder tests whether *inference-time* dynamics beat a
table **given the organisation**: both arms inherit the same placement of the same content. This paper
measures that and only that. It is not evidence about any other stage of any of these systems, in either
direction.

**L4 — The tuning standard was met, and here is exactly what it does not cover.** Every rival number
here comes from the full `6 lr × 2 wd` grid at nine seeds, with a 5× budget re-check (§2.6) — but
**tuning bias is the attack this audit is most exposed to**, because *"rivals lose to their own
byte-matched tables"* is precisely what an under-tuned rival produces, and that bias runs *toward* our
headline. What we can show is that on this harness the grid is not the binding constraint: the widened
grid's points are never selected (0 of 45 cells), the tuning effect on an arm's read is ≤ 0.031, and a
64 % cut in fit loss moves the eval metric by less than one SE. What we have **not** shown is that no
tuning protocol whatsoever would rescue an arm: we did not change the optimiser family, the schedule
(the standard's `β = (0.9, 0.98)` and cosine decay are declared non-adopted), the architecture-side
hyperparameters, or the head-width allocation forced by the iso-state rule. Two further disclosures
belong here rather than in a footnote: the initialisation-key scheme had to change to widen the grid, and
that re-draw moved arms **4–35× more than the tuning did** (priced with a control column, §2.6); and the
first pass's reduced grid is reported in full, before/after, in Appendix I.1.

**L4a — Our tuning standard's own selection rule is weaker than it looks, by our measurement.**
Best-of-grid selected on the fit split cannot prefer a regulariser (`wd = 0.1` is chosen only by
fourth-decimal tie-breaks, 12 of 45 cells) and never prefers a smaller learning rate (0 of 45), so the
6 × 2 grid is operationally 6 × 1. We therefore also ran a **held-out** selection stream, under which the
added points *are* chosen (26 of 45 and 24 of 45) and TTT-Linear's read improves from −0.6075 to
−0.4461 — and under which the raw-table margins remain negative on all five arms (−0.24 … −0.49). We
report held-out selection as a **declared secondary**, not as the primary, because the primary rule was
registered first; a reader who prefers it gets the same headline and a smaller rescued set (`{GDN}` at
nine seeds, against `{DeltaNet, GDN, GDN-2}` under the primary rule) — which would make the audit thinner,
not stronger.

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
float32, on CPU — nine seeds on the rival dividend arms, three on the CLU column, the protocol-validation
run, the frontier column and the attribution sweep. **Nothing here transfers to a language-model claim**,
and no language-modelling run was sized or attempted.

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
scale with `d_in = 5`, 5–6 stored items and ~10-token streams: **nothing in the audit beats a
zero-extra-byte reader of a raw table holding the same bytes** — not the rival test-time-dynamics
families (0 of 5 arms, nine seeds, full tuning grid, −0.2592 … −0.4602, and unchanged at 5× budget and
under a held-out selection rule), and not ours (−0.3180 ± 0.0804 over three seeds). Against a weaker but
natural control read through each memory's own projections, **test-time dynamics does pay for the
delta-rule family and does not pay for ours** — and the only reason we are entitled to report the
stronger control beside it is that both were registered before measurement.

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

### A.1 The audit run, full tuning grid (§4.1, §4.2, §4.3, §4.4) — the source of every rival number in the main text

| item | value |
|---|---|
| environment | JAX **0.9.0** / Equinox **0.13.4** / Optax **0.2.6**, resolved and printed in-session and identical to the first pass's; float32 throughout, on both sides of the ledger; no dependency re-resolution (the main environment was reused rather than re-synced) |
| seeds | **0, 1, 2** (the registered primary) **+ 3–8** (a declared power addition) ⇒ **n = 9** pooled on every rival cell reported in the main text. Sample sd (`ddof = 1`), `SE = sd/√n` |
| fit-stream seeds (train/eval separation guard) | `seed + 101`, `seed + 102` — different sites, different payloads, **never the eval stream**; byte-identical to the first pass's fit streams (asserted in a test). `seed + 103` is a **held-out** stream used **only** for the declared secondary selection rule (§2.6, §6 L4a) and never differentiated through, never evaluated on |
| families run | `aggregate@base` (the dividend family) only. ⛔ The `overload@load1x_shipped` **frontier column was NOT re-run** under the full grid — declared, with its reason, in Appendix J. ⛔ `recency`, `manifold` NOT RUN (struck by protocol validation, §2.5) |
| task-harness non-default flags | `family=aggregate`, `capacity=6`, `consolidate_every=2`, staged admission on |
| store non-default flags | `capacity=6`, `budget=6`, `min_atoms=192`, `min_atoms_base=192`, `min_atoms_c=1.0`, staged admission on (the audited cell, unmodified) |
| **rival tuning grid** | `lr ∈ {1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2} × wd ∈ {0, 0.1}`; TTT arms additionally `b ∈ {1, 16}` ⇒ **24 configurations per TTT arm, 12 per delta arm**; best-of-grid on the fit split (primary) or on the held-out stream (declared secondary) |
| optimiser | `adam` at `wd = 0`, decoupled `adamw` at `wd = 0.1` — **no optimiser change** relative to the first pass. ⚠ **Declared deviations from the tuning standard:** its `β = (0.9, 0.98)` and its cosine decay were **not** adopted, so that exactly one variable moves between passes and the control column stays meaningful |
| outer steps | **400** (primary) · **2000** re-check on the sub-grid containing every 400-step winner (`lr ∈ {3.16e-3, 1e-2} × wd ∈ {0, 0.1}`) |
| ⚠ initialisation-key scheme | **CHANGED and priced**: one initialisation per (arm, seed, `b`), shared across all `(lr, wd)`. The first pass split one key sequentially across grid points, which makes every configuration's initialisation depend on the grid's length and order. The change is priced by the `lite control` column (§2.6, Appendix I.1a) |
| chosen configs (primary rule, 400 steps) | ttt_linear: `3.16e-3/0/b1` · `1e-2/0/b16` · `1e-2/0.1/b16` (seeds 0/1/2) — ttt_mlp: `3.16e-3/0/b16` · `1e-2/0.1/b16` · `1e-2/0.1/b16` — deltanet: `3.16e-3/0/b16` · `3.16e-3/0.1/b16` · `3.16e-3/0/b16` — gdn and gdn2: `3.16e-3/0/b16` · `3.16e-3/0/b16` · `3.16e-3/0.1/b16`. ⚠ `b` changes the head width because the mini-batch buffer is inside the state budget — declared, and required by the iso-state rule |
| iso-state budget | **1364 float32 = 5456 B**; head widths **29 / 12 / 36**, registered before any run and asserted in the test suite |
| identical-encoder invariant | enforced in code on every cell; the encoder content hash on `aggregate@base` is `phi_id = 09dc0ee5…`, asserted identical across all arms |
| byte law used | corrected `ratio = [A(D+2)+d]/(d+m)`; ledger identity green on every cell (`5456 B / 100 B / 54.56×`); floors **2.20×** (`n_spec=0`) / **2.40×** (`n_spec=1`). ⛔ never *"verified to 1e-9 in all 28 cells"* (§3.1) |
| CLU column | **banked, not re-derived** in this run; its fidelity check reproduces the recorded per-seed values (`−0.682608 / −0.496261 / −0.438906` at seed 0) digit-for-digit |
| reproducibility | ⭐ the **first pass reproduces digit-for-digit at the base code** from this branch on all five arms (`full −0.4546 / −0.6324 / −0.4652 / −0.3961 / −0.3964`; raw-table margins `−0.2465 / −0.4242 / −0.2571 / −0.1880 / −0.1883`), so every difference between the two passes is attributable to a declared change |
| test suite at the recorded commit | **1143 passed, 0 failed**; lint clean; 17 targeted tests on the grid, held-out selection and before/after thresholds (7 of them new) |
| wall clock | 400-step grid 346 s (3 cells) · seeds 3–8 781 s · 2000-step re-check 317 s · first-pass reproduction 128 s · first-pass code at seeds 3–8 263 s ⇒ **≈ 31 min** of compute in total |
| ⚠ disclosed incident (first pass, inherited) | a per-rival fit key used a process-salted `hash()`; fixed to a stable index **before any reported number was recorded** (§6 L12) |

### A.1b The audit's first pass (reduced grid) — provenance for Appendix I.1 only

| item | value |
|---|---|
| environment | as A.1 (JAX 0.9.0 / Equinox 0.13.4 / Optax 0.2.6) |
| seeds | **0, 1, 2**; `SE = sd/√3`. ⛔ No verdict from this pass is quoted in the main text |
| rival outer loop | Adam, **400 steps**, `lr ∈ {1e-3, 3.16e-3, 1e-2}`, best-of-grid **on the fit split**; TTT arms additionally `b ∈ {1, 16}`; initialisation key split **sequentially** across grid points |
| chosen configs | seed 0: ttt_linear (d29, lr 1e-2, b16) · ttt_mlp (d12, lr 1e-2, b16) · delta arms (d36, lr 3.16e-3). Seeds 1/2: ttt_linear flips to (d36, lr 1e-2, **b1**) |
| families run | `aggregate@base` · `overload@load1x_shipped` (**frontier column**; this pass is the only source for the frontier rival rows, Appendix H) |
| status | ⚠ **Reduced grid** vs the 6 × 2 standard — declared at the time as a budget choice, **not** as compliance. Superseded for every main-text number by A.1 |
| wall clock | 6 audit cells, 260 s + frontier; whole run < 8 min |

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

**B.5 The rescue gate, formally, and its power requirement.** Arm `a` on cell `c` is RESCUED iff its full
read exceeds its own blank-store control by more than **2 SE of that lift**. A non-rescued arm's row is
printed, and every margin against it is suppressed.

⚠ **The gate needs seeds, and this is the operational form we recommend.** Its control — for a memory
with a learned initial state, that initialisation read through fitted projections — is a **single
initialisation draw per seed**, and its spread across seeds can exceed the lift being gated
(observed blank reads on one arm across three seeds: −0.962 / −2.634 / −1.390). Measured consequence on
this harness: at **n = 3** three legitimate configurations return three different rescued sets
(`{ttt_linear, gdn, gdn2}`, `{}`, `{ttt_linear}`); at **n = 9** two independent code paths agree on four
of five arms. ⇒ **report rescue verdicts at n ≥ 9**, and prefer, where affordable, a control that is
paired per seed or averaged over several initialisation draws. We did neither of the latter two and
report the nine-seed verdict instead; both remain open improvements to the protocol.

Verdicts as applied in this paper: **RESCUED** `deltanet`, `gdn`, `gdn2` (dividend family, n = 9, both
code paths); ⛔ **NOT RESCUED** `ttt_mlp` (every configuration run); ⚠ **UNSTABLE** `ttt_linear` (clears
under one initialisation scheme, not the other). On the byte-frontier column no arm separated from its
control at the three seeds run, and that column was not re-run at nine seeds — so it carries **no
quotable margin in either direction** (§4.5).

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
> secondary reading `S_excl = 0.6500`. ⚠ **No rival row here separates from its own blank-store control**,
> so **no margin against any of them is quotable in either direction**, and we do not draw a curve from
> it.

⚠ **Provenance and status of this table:** these rows come from the audit's **first pass** (reduced
tuning grid, three seeds, Appendix A.1b) and were **not** re-run under the full grid or at nine seeds —
a declared NOT-RUN with its reason in Appendix J. Because our own measurement shows the rescue gate is
underpowered at three seeds (§2.2, B.5), we report neither a rescue verdict nor a margin from this table;
it is printed for completeness. 3 seeds, 24 queries, 7 stream tokens, 6 live items, chance = 0.1667.

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
| **CLU** (banked, reused not re-measured) | — | 57384 | — | — | **0.9722** | **1.0000** | 0.1667 |

**Under-training check, run before declaring the column non-informative.** At **5×** the outer budget
(2000 steps vs 400), `gdn2` goes 0.0417 → 0.0000 and `ttt_linear` 0.2083 → 0.1250 — *worse* — while
`ttt_linear`'s **fit-split** loss reaches MAE **0.024** against an eval MAE of **0.75** (a **31×** gap).
It is a generalisation failure across item geometries, forced by the guard that outer parameters never
see the eval stream's items. The payload alphabet is spaced 0.4 apart, so an eval MAE of 0.58–0.75 decodes
at chance **by arithmetic**.

**The CLU's banked frontier curve** (reused, not re-measured): decode **0.972 → 0.097** as the
store-to-table byte ratio falls **478× → 2.28×**. We quote the curve, not the endpoint.

## Appendix I — Pre-registration scorecards

Predictions were registered before the corresponding measurement in each case. We print the misses.

### I.1 The audit's rival predictions (registered before the **first pass**)

⚠ **Every "measured" entry in this table is the first pass at three seeds** (reduced grid, Appendix A.1b)
— that is what the predictions were scored against, and re-scoring a prediction against a later run would
not be a pre-registration any more. Where the full grid at nine seeds moves an entry, the move is stated
in the row and the nine-seed value governs the paper.

| # | registered | measured (first pass, `n = 3`) | verdict |
|---|---|---|---|
| P2 (measured half) | ≥2 of the 3 measured (k,v)-state families lose to their own byte-matched table's strongest +0 B reader | **1 of 3** (deltanet only) | ⛔ **refuted as registered.** Second reading, pre-committed and printed beside it: **3 of 3** lose to the **raw-metric** +0 B table at the same bytes. ⚠ First half only — the real-data half is untested here. ⚠ Mamba-2 and SDM were adjudicated from equations only |
| P3 | the two function-valued memories show the largest positive dividend | — | ⛔ **NOT-RUN** (no Titans arm ⇒ the pair cannot be formed). **NOT-RUN is not refuted.** TTT-MLP alone is a single-arm datum and it is not rescued |
| P5 | the launder transfers to all five rival state types; 0 of 5 failures | 5 of 5 carry a byte-matched table; 0 failures | ✅ supported |
| R1 | rival arg-min launder ≈ −0.42, band [−0.55, −0.25] | −0.4245 · −0.4108 (TTT) · −0.6658 · −1.4158 · −1.2735 (delta) | ◐ 2 of 5 in band; the 3 delta arms fall far below it — the finding of §2.4 |
| R2/R3 | rival `full` ≈ −0.15, band [−0.30, −0.05] | −0.40 to −0.63 | ⛔ **out of band, all 5.** We over-predicted the rivals: a byte-matched linear memory at `d_in = 5` does not interpolate as well as its own 2-NN reader does |
| R4 | dividend vs own **arg-min** table = +0.27, band [+0.05, +0.45] | mean **+0.3691** | ✅ in band |
| R5 | signed **+0 B** margin = −0.02, band [−0.15, +0.08], **≥3 of 5 ≤ 0** | mean **−0.0392; 3 of 5 ≤ 0** | ✅✅ in band and the count exact at three seeds. ⭐ **At nine seeds the count becomes 4 of 5 ≤ 0** (GDN crosses to −0.0102 ± 0.0229; only GDN-2 stays positive at +0.0473 ± 0.0277, **within 2 SE of zero ⇒ a tie, not a win**). ⚠ The band was registered on the cross-arm mean, which we did **not** re-aggregate at nine seeds; the per-arm nine-seed margins are in §4.1.1 |
| R5-raw | *(not banded; added after the band was fixed; reported as a second reading, never substituted)* | −0.1880 … −0.4242 | **5 of 5 ≤ 0**; at nine seeds **5 of 5 ≤ 0** again, at −0.2592 … −0.4602, every one ≥ 4.4 SE below zero |
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

### I.1a The full tuning grid: before / after, arm by arm

`aggregate@base`, mean ± SE. **`full grid` = the 6 lr × 2 wd grid at 400 steps, seeds 0/1/2 (the
registered primary of the second pass); `pooled n = 9` adds seeds 3–8 and is what the paper quotes;
`lite control` re-selects the first pass's own 3-lr sub-grid from the *same* fits, so that
`full grid − lite control` isolates the tuning effect and `lite control − first pass` isolates the
initialisation re-draw.**

| arm | quantity | first pass (`n=3`) | full grid (`n=3`) | lite control (`n=3`) | **pooled (`n=9`)** | outcome |
|---|---|---|---|---|---|---|
| **ttt_linear** | rescued? | ✅ | ⛔ | ⛔ | ⛔ (✅ under the first pass's own code at `n=9`) | **verdict UNSTABLE** |
| | `full` | −0.4546 ± 0.0312 | −0.6332 ± 0.1181 | −0.6029 | **−0.6075 ± 0.1096** | |
| | +0 B margin | −0.0523 | −0.2132 ± 0.1041 | −0.1869 | **−0.2213 ± 0.1062** | sign unchanged |
| | **raw-table margin** | −0.2465 | −0.4251 ± 0.1147 | −0.3948 | **−0.4602 ± 0.1038** | unchanged (negative) |
| **ttt_mlp** | rescued? | ⛔ | ⛔ | ⛔ | ⛔ | **unchanged** |
| | `full` | −0.6324 ± 0.2036 | −0.5052 ± 0.1473 | −0.5070 | **−0.5898 ± 0.0731** | |
| | +0 B margin | −0.2284 | −0.1135 ± 0.1408 | −0.1003 | **−0.2095 ± 0.0683** | sign unchanged |
| | **raw-table margin** | −0.4242 | −0.2971 ± 0.1438 | −0.2988 | **−0.4425 ± 0.0869** | unchanged |
| **deltanet** | rescued? | ⛔ | ⛔ | ⛔ | **✅ at `n=9`** | **changed on power** |
| | `full` | −0.4652 ± 0.0402 | −0.4478 ± 0.0590 | −0.4469 | **−0.4205 ± 0.0299** | |
| | +0 B margin | −0.0047 | −0.0162 ± 0.0772 | −0.0149 | **−0.0172 ± 0.0263** | sign unchanged |
| | **raw-table margin** | −0.2571 | −0.2396 ± 0.0664 | −0.2387 | **−0.2732 ± 0.0395** | unchanged |
| **gdn** | rescued? | ✅ | ⛔ | ⛔ | ✅ | **n=3 verdicts unstable** |
| | `full` | −0.3961 ± 0.0208 | −0.4104 ± 0.0289 | −0.4110 | **−0.4073 ± 0.0120** | |
| | +0 B margin | **+0.0448** | +0.0181 ± 0.0588 | +0.0168 | **−0.0102 ± 0.0229** | **crosses ≤ 0 at `n=9`** |
| | **raw-table margin** | −0.1880 | −0.2022 ± 0.0354 | −0.2028 | **−0.2600 ± 0.0278** | unchanged |
| **gdn2** | rescued? | ✅ | ⛔ | ⛔ | ✅ | **n=3 verdicts unstable** |
| | `full` | −0.3964 ± 0.0220 | −0.4350 ± 0.0394 | −0.4384 | **−0.4065 ± 0.0178** | |
| | +0 B margin | +0.0445 | +0.0305 ± 0.0574 | +0.0352 | **+0.0473 ± 0.0277** | positive but **< 2 SE ⇒ a tie** |
| | **raw-table margin** | −0.1883 | −0.2269 ± 0.0434 | −0.2303 | **−0.2592 ± 0.0292** | unchanged |

**Derived outcomes.** Count of `+0 B` margins ≤ 0: first pass **3 of 5** → full grid at three seeds
**3 of 5** → **at nine seeds 4 of 5**. Rescued set: first pass `{ttt_linear, gdn, gdn2}` → full grid at
three seeds `{}` → **at nine seeds `{deltanet, gdn, gdn2}`** (and `{ttt_linear, deltanet, gdn, gdn2}`
under the first pass's own code at nine seeds — the one disagreement, and the reason TTT-Linear is
printed UNSTABLE). ⭐ **The raw-table margin — the paper's load-bearing quantity — never changes sign in
any column, at any budget, under either selection rule, at either seed count.**

**The tuning effect and the initialisation re-draw, separated** (change in `full`; TTT-Linear · TTT-MLP ·
DeltaNet · GDN · GDN-2): tuning `−0.0303 / +0.0018 / −0.0009 / +0.0006 / +0.0034`; re-draw
`−0.148 / +0.125 / +0.018 / −0.015 / −0.042`. **The re-draw is 4–35× the larger of the two.**

### I.1b The full-grid pass's own pre-registration scorecard

Registered before any grid point was run. **Score: 3 exact/confirmed · 4 partial · 2 refuted.**

| # | registered | measured | verdict |
|---|---|---|---|
| P1 | the widened `lr` axis selects nothing new: **0 of 15** cells pick a new lr | **0 of 15** (and 0 of 45 pooled) | ✅✅ **exact** |
| P2 | `wd = 0.1` selected in ≤ 2 of 15; where selected, Δfit < 0.005 | **6 of 15** (12 of 45); Δfit < 0.005 in every case | ⛔ **refuted on the count**, ✅ on the magnitude — the fit surface is *flat* in `wd`, so selection is a tie-break rather than a preference |
| P2 corollary | held-out selection picks `wd = 0.1` in ≥ 3 of 15 and changes no threshold | **24 of 45**; changes no threshold | ✅ supported |
| P3 | tuning effect on `full`: delta arms < 0.010, ttt_linear < 0.050, ttt_mlp < 0.250 | **−0.0303 / +0.0018 / −0.0009 / +0.0006 / +0.0034** (TTT-Linear · TTT-MLP · DeltaNet · GDN · GDN-2) | ✅ **5 of 5 in band** |
| P3′ | *(the same bands read against the first pass, i.e. tuning **plus** re-draw)* | **−0.179 / +0.127 / +0.017 / −0.014 / −0.039** (same order) | ◐ **4 of 5** — TTT-Linear misses, entirely on the re-draw term |
| P4 | all five rescue statuses **unchanged** | **3 flip at n = 3**; at n = 9 DeltaNet flips the other way | ⛔ **refuted — and it is the pass's main finding** (§6 L2a). The registered *mechanism* (blank-control variance) was right; the arms named as fragile were not |
| P5 | 2000 steps rescues nothing: TTT fit loss ↓ ≥ 20 %, `full` moves < 1 SE, raw margin negative 5 of 5 | ttt_mlp fit **−64.1 %**, ttt_linear −4.5 %; `full` moves < 1 SE on 4 of 5 (ttt_linear ≈ 1.4 SE); raw margin negative **5 of 5** | ◐ supported in substance, one sub-clause missed |
| P6 | the projected-vs-raw gap survives: ≥ 0.15 on all five, ≥ 0.9 on gdn/gdn2 | n = 9: **0.276 / 0.263 / 0.425 / 0.856 / 0.942**, all > 2 SE | ◐ the gap survives **5 of 5**; the ≥ 0.9 sub-clause misses on GDN (0.856) |
| alt. | *"the grid rescues an arm"*, registered with prior ≤ 15 % | did not occur in any column | ✅ the registered primary held |

### I.1c First-pass columns not re-aggregated at nine seeds

⚠ **These are three-seed, reduced-grid values. No claim in this paper rests on them**; they are printed
because the audit's column set is part of the protocol and appendix material is not pruned. The
projected-versus-raw *gap* derived from these columns **is** available at nine seeds (I.1b, P6) and is
what §2.4 and §4.3 quote.

| arm | projected (arg-min) launder | dividend vs that launder | blank store | same-keys null |
|---|---|---|---|---|
| ttt_linear | −0.4245 | −0.0302 | −0.8426 | −0.4577 |
| ttt_mlp | −0.4108 | −0.2216 | −0.6031 | −0.4285 |
| deltanet | −0.6658 | +0.2006 | −0.5657 | −0.6480 |
| gdn | −1.4158 | +1.0197 | −1.3220 | −1.2202 |
| gdn2 | −1.2735 | +0.8771 | −1.6618 | −1.1341 |
| **CLU** (banked, `n = 3`) | **−0.4472** | **−0.0789** | **−0.4221** | **−0.8175** |

### I.1d The tuning grid, audited

**The fit-split loss surface** (three-seed means; `*` marks the arg-min, i.e. the configuration the
primary rule selects). Every point the widened grid adds is to the left of the arg-min:

```
arm          b   wd    1e-04    3e-04    5e-04    1e-03    3e-03    1e-02
ttt_linear   1  0.0   0.3132   0.2555   0.2474   0.2333   0.1919  *0.1865
ttt_linear   1  0.1   0.3130   0.2549   0.2470   0.2332   0.1920  *0.1849
ttt_linear  16  0.0   0.2794   0.2401   0.2308   0.2147   0.1835  *0.1773
ttt_linear  16  0.1   0.2795   0.2397   0.2322   0.2178   0.1822  *0.1788
ttt_mlp      1  0.0   0.3898   0.3170   0.2954   0.2758   0.1814  *0.1241
ttt_mlp      1  0.1   0.3884   0.3246   0.2851   0.2604   0.1917  *0.1329
ttt_mlp     16  0.0   0.2279   0.1954   0.1699   0.1318   0.1021  *0.0930
ttt_mlp     16  0.1   0.2283   0.1965   0.1707   0.1327   0.1062  *0.0847
deltanet    16  0.0   0.3291   0.2795   0.2730   0.2631  *0.2616   0.2623
deltanet    16  0.1   0.3291   0.2794   0.2729   0.2627  *0.2621   0.2630
gdn         16  0.0   0.3708   0.2820   0.2747   0.2645  *0.2615   0.2618
gdn         16  0.1   0.3709   0.2821   0.2747   0.2646  *0.2615   0.2620
gdn2        16  0.0   0.3766   0.2840   0.2739   0.2628  *0.2617   0.2619
gdn2        16  0.1   0.3767   0.2840   0.2738   0.2627  *0.2615   0.2622
```

Monotone improving in `lr` in every row; `wd` moves the loss in the fourth decimal; the delta arms' whole
surface spans **0.0011–0.0155**.

**Three selections from one set of fits** (no extra training cost):

| label | grid | selected on | rescued set (`n=3` / `n=9`) |
|---|---|---|---|
| **full grid** (primary) | 6 lr × 2 wd | the fit split's own loss | `{}` / **`{deltanet, gdn, gdn2}`** |
| lite control | the first pass's 3 lr, `wd = 0` | the same | `{}` / `{deltanet, gdn, gdn2}` |
| held-out (declared secondary) | 6 lr × 2 wd | a held-out stream (`seed + 103`) | `{ttt_linear}` / `{gdn}` |

**The 5× budget re-check** (`lr ∈ {3.16e-3, 1e-2} × wd ∈ {0, 0.1}`, 2000 steps, three seeds):

| arm | fit loss 400 → 2000 | `full` 400 → 2000 | **raw-table margin @ 2000** | +0 B margin |
|---|---|---|---|---|
| ttt_linear | 0.1697 → 0.1620 (−4.5 %) | −0.6332 → −0.4711 ± 0.0488 | **−0.2630 ± 0.0556** | −0.0507 |
| ttt_mlp | 0.0839 → 0.0301 (**−64.1 %**) | −0.5052 → −0.4691 ± 0.0891 | **−0.2609 ± 0.0903** | −0.0717 |
| deltanet | 0.2616 → 0.2615 (−0.0 %) | −0.4478 → −0.4468 ± 0.0797 | **−0.2387 ± 0.0875** | −0.0072 |
| gdn | 0.2614 → 0.2611 (−0.1 %) | −0.4104 → −0.4265 ± 0.0334 | **−0.2184 ± 0.0409** | +0.0025 |
| gdn2 | 0.2615 → 0.2613 (−0.1 %) | −0.4350 → −0.4431 ± 0.0348 | **−0.2349 ± 0.0402** | +0.0547 |

**The rescue gate across every configuration we ran** (lift over the arm's own blank store, ± SE):

| configuration | `n` | ttt_linear | ttt_mlp | deltanet | gdn | gdn2 |
|---|---|---|---|---|---|---|
| first pass (reproduced digit-for-digit) | 3 | ✅ 0.388 ± 0.087 | ⛔ −0.029 ± 0.109 | ⛔ 0.100 ± 0.130 | ✅ 0.926 ± 0.239 | ✅ 1.265 ± 0.497 |
| full grid (primary) | 3 | ⛔ 0.128 ± 0.186 | ⛔ 0.019 ± 0.133 | ⛔ 0.235 ± 0.157 | ⛔ 0.461 ± 0.360 | ⛔ 0.180 ± 0.169 |
| lite control | 3 | ⛔ 0.191 ± 0.184 | ⛔ −0.023 ± 0.067 | ⛔ 0.249 ± 0.153 | ⛔ 0.472 ± 0.354 | ⛔ 0.252 ± 0.131 |
| full grid @ 2000 steps | 3 | ✅ 0.465 ± 0.044 | ⛔ −0.006 ± 0.123 | ⛔ 0.211 ± 0.132 | ✅ 0.623 ± 0.256 | ⛔ 0.779 ± 0.445 |
| held-out selection | 3 | ✅ 0.351 ± 0.148 | ⛔ −0.045 ± 0.327 | ⛔ 0.175 ± 0.123 | ⛔ −0.018 ± 0.038 | ⛔ 0.278 ± 0.170 |
| **full grid, pooled** | **9** | ⛔ 0.093 ± 0.134 | ⛔ −0.071 ± 0.090 | **✅ 0.294 ± 0.077** | ✅ 0.880 ± 0.227 | ✅ 1.025 ± 0.329 |
| **first-pass code, pooled** | **9** | ✅ 0.320 ± 0.083 | ⛔ 0.093 ± 0.107 | **✅ 0.141 ± 0.046** | ✅ 0.947 ± 0.149 | ✅ 1.384 ± 0.276 |

⭐ **Read this table as the evidence for §2.2's power requirement**: the same arm can pass or fail at
three seeds depending on which legitimate configuration is run, while the two nine-seed rows agree on
four of five arms.

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
- **The byte-frontier column under the full tuning grid, and at nine seeds.** Its rival rows are the
  first pass's (Appendix A.1b, Appendix H). The trigger we registered for re-running it — *"the wider grid
  rescues an arm on the dividend family"* — did not occur: the two rescue-verdict changes at nine seeds
  are **power** effects, not tuning effects. One labelled row would close it; we did not run it and we
  therefore quote nothing from that column.
- **The nine-seed aggregate of the projected-launder, dividend, blank-store and same-keys-null columns.**
  The per-seed values exist for all nine seeds; the pooled aggregate was computed only for `full`, the
  `+0 B` margin, the raw-table margin and the rescue lift. The four remaining columns are reported at
  three seeds in Appendix I.1c and carry no claim.
- **Two sub-clauses of the tuning standard**: `β = (0.9, 0.98)` and cosine decay were not adopted
  (declared deviation, §2.6, Appendix A.1), so that exactly one variable moves between the two passes.
- **A re-run of the audit under the held-out selection rule as the primary.** It exists as a declared
  secondary (§6 L4a); adopting it as primary would be a post-hoc change of a registered rule.
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
drawn, the arm that fails the rescue gate hatched and labelled NOT RESCUED, the arm whose verdict is
unstable hatched differently and labelled UNSTABLE, the CLU bar in a distinct fill. Error bars = ±1 SE,
**n = 9 on the rival bars and n = 3 on the CLU bar — the caption must say so**, along with family, byte
budget, tuning grid, and the scale qualifier. Not rendered in this draft.

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
falls 478× → 2.28×), with the rival points **omitted** and a caption stating why (no arm on that column
separates from its own blank-store control at the seeds run, and the column was not re-run under the full
tuning grid).

## Appendix L — Negative results and refutations recorded by this work

Per project policy, negatives are documented and never dropped.

1. **The audit's own headline is negative for our system**: dividend −0.0789 against its own launder,
   +0 B margin −0.3180 ± 0.0804, and an earlier round of the same audit went 0-for-4.
2. **Three of four designed task families are struck as protocol-invalid** (§2.5) — including two where
   the memory reads *below* a ≤4 B substitute, and one where it reads below its own blank store.
3. **The registered byte-matched control for weight-valued memories is a weak control** (§2.4) — a
   refutation of our own pre-registered protocol text, costing the table 0.263–0.942 at nine seeds.
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
15. **Our own rescue-gate verdicts were not reproducible at three seeds** (§6 L2a, Appendix I.1b P4): a
    pre-registered prediction that the five rescue statuses would be stable under re-tuning was refuted,
    three of them flipped, and the gate's own power — not the tuning — was the cause. Every three-seed
    rescue verdict we had previously recorded is withdrawn rather than quoted.
16. **A registered prediction about our own tuning standard was refuted on its count**: we predicted
    `wd = 0.1` would be selected in at most 2 of 15 cells and it was selected in 6 of 15 — but always by
    a fourth-decimal tie-break, which is how we discovered that the standard's selection rule makes its
    own regulariser axis unselectable (§6 L4a).
17. **The initialisation-key scheme, not the tuning, dominated the difference between our two passes**
    (4–35×) — an effect nobody had asked about, and one that would have been invisible without a control
    column we had to think to run (§2.6).

## Appendix M — Reproducibility and artifact notes

Seed counts are stated per section and never mixed silently: the rival dividend arms are 9-seeded; the
CLU column, the protocol-validation run, the frontier column and the attribution sweep are 3-seeded
(Appendix A). Statistics conventions are stated per section. ⭐ **The audit's first pass reproduces
digit-for-digit from the second pass's branch at the base code, on all five arms and on both the `full`
and raw-margin columns**, so every difference between the two passes is attributable to a change declared
in §2.6 — this is the strongest reproducibility statement in the paper and it is the one that licenses
the before/after table. The protocol-validation run reproduces its reference artifact exactly on every
shared arm, per seed; the attribution sweep's per-radius aggregates recompute from the merged artifact to
the last digit on every 3/3 point; the full test suite is green at the recorded commit (1143 passed, 0
failed) with the byte-ledger identity and the identical-encoder invariant asserted per cell. One
reproducibility incident is disclosed in §6 L12.

---

## ⛔ Open editorial items (delete before circulation)

1. **Number-freeze gate is CLOSED.** Every rival number in this draft is the full-grid, nine-seed value;
   the provisional markers of the previous revision are gone, and the first pass survives only as
   labelled appendix material (A.1b, H, I.1–I.1d).
2. **Title** is `[WORKING TITLE: …]` and **authorship** is `[AUTHORS PLACEHOLDER]`; both workshopped at
   the end, both blank in an anonymized build.
3. **Naming continuity** — this draft uses "CLU" with the continuity sentence *"the CLU, introduced as
   CHLU in Jawahar & Pierini (2026)"* in §1. Which paper carries the name's debut is a Hub/Head call.
4. **Figures are specified, not rendered** (Appendix K).
5. **One citation is single-sourced** and must be re-verified against the published text before printing:
   the Feng, Wallace & Boyd-Graber (2019) converse-caveat quotation (§2.4, §6 L8).
