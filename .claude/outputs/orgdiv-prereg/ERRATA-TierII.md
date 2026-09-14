# ERRATA — `PREREG-TierII.md`

**Filed 2026-08-01 (C2W5) by `doc-curator-c2w5-fold`, per the task file's item 4 and the `ERRATA-Bprime.md` precedent.**

> ⛔⛔ **`PREREG-TierII.md` IS NOT EDITED, AND MUST NOT BE.** A pre-registration whose text is revised after the fact stops being one. This file sits **beside** the prereg, points **at** it, and names the wrong sentence, the corrected statement, the affected cells and the **direction** of each error. Every downstream site (draft, registry, report, primer) takes its wording from **here**, not from the prereg's original sentence.
>
> **Scope: five errata, E-T1…E-T5.** All five were found by our own spokes, against our own registered statements, in the wave the prereg was written for. ⭐ **Three of them (E-T1, E-T2, E-T4) are the pre-registration WORKING** — a registered prediction that fails is a finding, and two of these changed the experiment before it produced a number.
>
> ⛔⛔ **STANDING RE-LABELLING THAT GOVERNS ANY QUOTATION OF THIS PREREG (charter §A20.2, binding):** the refuted object is **"the `P`-particle occupancy read protocol at `P = 4`"** — ⛔ **never "the compositional family"** — and **0.272 is a *reference ceiling from an out-of-class decoder*, never an arm's bar** (the arm bar is chance + 0.05 = 0.05039, reader class ≤ `N_a·m = 256`).
>
> **Sources:** `orgdiv-cat-test` (reconciliations 1, 2, 6; §1; §3) · `orgdiv-null-arms` (§1.1, §4, §5) · both `[C2W5]` §10 review entries · charter **ADDENDUM 5 §A20**. Registry companions: **N212 · N214 · N215 · N217 · N224**.

---

## E-T1 — §2.3 rule 4's **second** (payload) assertion is unsatisfiable at its own registered `m = 1`

**Affected statement.** `PREREG-TierII.md` **§2.3, rule 4**, asserts that a held-out query is valid only if its payload half is recoverable — at the registered **`m = 1`**.

| item | registered | **measured** | direction |
|---|---|---|---|
| fraction of held-out queries passing rule 4's payload half at **`m = 1`** | assumed ~all | ⛔ **0.5 %** | **fatal to the metric as registered** |
| the same at **`m = 8`** | — | ✅ **100 %** | the registered deviation that was taken |

**Why it was missed:** ⛔ **§2.4's feasibility check verified only the *set* half of rule 4 and never the payload half.**

**What is corrected.** The cell that ran uses **`m = 8`** (registered deviation **D1**), applied identically to every arm. ⚠ **Any quotation of rule 4 must state `m = 8`**; the registered `m = 1` form is void. ⭐ **`m ∈ (1, 8)` is BRACKETED, NOT LOCATED** — no intermediate value was tested. (**N212**.)

---

## E-T2 — §3.4's registered K1 prediction ("passes at `a ≥ 12`") is **REFUTED**

**Affected statement.** `PREREG-TierII.md` **§3.4** predicts that write admissibility (K1) passes at **`a ≥ 12`** atoms per item.

| `a` | loss ≤ 0.05 | `λ_min > 0` ≥ 90 % | capture ≥ σ_q at ≥ 90 % | K1 |
|---|---|---|---|---|
| 4 | 0.153 | 1.00 | 0.354 | ⛔ FAIL |
| **12 (registered)** | 0.049 | 1.00 | **0.812** | ⛔ **FAIL** (SC-6 capture leg, 3/3 seeds) |
| **32** | 0.0093 | 1.00 | 0.958 | ✅ **PASS** |

**What is corrected.** The cell that ran uses **`a = 32`** (registered deviation **D2**) — **2.7× the registered atom budget.** ⚠ **`a ∈ (12, 32)` is BRACKETED, NOT LOCATED.** ⭐ **Consequence for the byte ledger: see E-T3.** (**N212**.)

---

## E-T3 — §5.1's `ratio = 1.4·A + 0.8` spelling is valid **only at `d = 4, m = 1`**, and the design point's `5.00×` is not the ratio that ran

**Affected statement.** `PREREG-TierII.md` **§5.1** gives the matched-capacity byte ratio as `ratio = 1.4·A + 0.8`, and **§5.2**'s design point reports **5.00×**.

| item | published | **corrected** |
|---|---|---|
| the closed form | `1.4·A + 0.8` | **`[A(D + 2) + d] / (d + m)`** — the `ERRATA-Bprime.md` **E1** form, and the two agree **exactly** at `d = 4, m = 1` (both 5.00×) |
| the same at the cell that RAN (`d = 4, m = 8`) | — | ⛔ **they disagree: 12.0 vs 9.67** |
| the byte ratio of the cell that ran (`a = 32`) | stated **5.00×** (design point) / **3.20×** (`FROZEN-interfaces.md`, `a = 12`) | ⛔ **9.67×** (`n_atoms = 1024`, **57 344 B**) |

**Direction and status.** ⛔ **The ratio is REPORTED, NEVER CLAIMED** — there is no byte-matched tier-ii promise, and reaching one needs `K ≈ 3.6k` (a wave of its own, `PREREG-TierII.md` §5.2, unchanged). ✅ **`FROZEN-interfaces.md`'s two stale rows were corrected in place by the engineer on 2026-08-01 and carry a dated Hub erratum banner (C-3 precedent) — verified in place by this pass**: the matched-capacity ledger row (`a = 32` ⇒ 57 344 B, ratio 9.67×) and the reader parameter counts (`well_table` **72**, not 16; `mlp_small` **92**, not 88; `sum_linear` 104 and `knn` 0 agree). ⭐ **All four readers remain below the `N_a·m = 256` bound, so no verdict moves.** ⚠ **Capacity turned out not to be binding at all: the grid-max is identical at 21 504 / 57 344 / 114 688 B.** (**N212 / N215**.)

---

## E-T4 — §4.3's `null*` was a **placeholder**; it is now **COMPUTED**, over the entire registered grid

**Affected statement.** `PREREG-TierII.md` **§4.3** defines `null*` (the best matched-capacity organizer's score) as the quantity the physics arm must beat, without a value.

| item | registered | **measured** |
|---|---|---|
| `null*` | to be estimated | ⭐ **0.00117** — grid-max over **584 configurations × 5 score seeds**, argmax **N5** (`lr 3e-3, h=64, η=0.9, α=0.01, gate=none, chunk=1`) |
| the bar it is measured against | chance + 0.05 | **0.05039** (chance **3.906e-4**) ⇒ **43× short** |
| per-arm grid-max | — | N1 0.00039 (180 cfg) · N2 0.00039 (84) · N3 0.00078 (60) · N4 0.00078 (20) · N5 **0.00117** (240) |

**Two scope clauses that must travel with the number.** ⛔ **It is an ORACLE-SELECTED UPPER BOUND** — reported so the verdict reads *"no configuration clears"* rather than *"the one we picked didn't"*; **it is never any arm's score.** ⛔ **Readers were not fitted on the full grid** (`stage_gridmax` scores each arm's own native read; reader/native agreement was spot-checked on the selected configs and agrees to the last digit — **an assumption on the grid, declared**). ⚠ **And the companion `OD`/`OD_min` do NOT exist this wave** — there is no physics arm to swap against (it reads at chance, **N212**), so the tier-ii dividend statistic was not computed. ⛔ **Do not substitute the first-review `OD_min = −0.0016 ± 0.0015`, which is the in-house N3/N4 diagnostic.** (**N215**.)

---

## E-T5 — §3.5's F5 falsifier needs an **imitability caveat**: "does not fire" is an optimisation gap, not a channel result

**Affected statement.** `PREREG-TierII.md` **§3.5** registers F5 (*a static power diagram reproduces the physics organizer's assignment at ≥ 0.99*) as the test of whether the organizer is "really" VQ, with a registered read-objective agreement of **0.22**, band [0.15, 0.30].

| fitting route | registered | **measured (3 seeds)** | verdict |
|---|---|---|---|
| diagram fitted on the **read objective** (F5's registered null) | 0.22, band [0.15, 0.30] | **0.2576** (0.273 / 0.229 / 0.271) | ✅ **in band** — and it independently reproduces the cat test's 0.211–0.233 |
| diagram fitted **directly on the arm's own assignments** (oracle imitation, T5.2 rider) | 0.45, band [0.25, 0.70] | ⛔ **0.8888** (0.884 / 0.898 / 0.884); **0.9512** on SEEN | ⛔ **REFUTED — far higher** |
| **F5 fires (≥ 0.99)?** | NO | **NO**, at either fitting | ✅ |

**What is corrected.** ⚠ **STANDING CAVEAT, mandatory beside any quotation of cat-test F5:** *"does not fire" means **the read objective is a bad way to recover the assignment**, NOT **the physics organizer is non-VQ**.* ⭐ **The 0.89-vs-0.26 gap is an optimisation gap in fitting the diagram.** ⛔ **Anyone using F5 to argue the store's partition is not a power diagram is over-reading it.** ⚠ **Scope: the imitation target is the collapsed, untuned arm of N212, whose own unseen score is 0.0000–0.00195 — the imitation is of an assignment map, not of a working memory.** (**N217**.)

---

## Appendix — one instrument correction that is NOT a prereg statement, recorded here because every `d/s` in the prereg rides on it

⚠ **The effective-`s` estimator must subtract the confinement term `α‖q‖²`.** Measured on the cat-test store family: **0.438** unsubtracted vs **0.304** subtracted — a **1.44× inflation**; the cat test's own operating point uses **`s = 0.318`**, giving its `d/s = 2.70` its meaning. ⛔⛔ **`bprime-c6`'s `s = 0.40` is FLAGGED FOR A CHECK, NOT REFUTED**, and re-measuring that rig with the corrected estimator is a **declared NOT-RUN**. ⭐ **Direction: subtracting the confinement makes `s` smaller and `d/s` LARGER** — the uncorrected reading was the optimistic one for anything wanting a small `d/s`. **Owner assignment: charter §A20.5.** (**N224**; handover §7.28.)

---

## AMENDMENT-C2W7 — the reader class is RE-REGISTERED for the multiplicity read (2026-08-05)

**Filed 2026-08-05 by `experiment-engineer` (`c2w7-read-cardinality`), BEFORE any arm of the
cardinality iteration ran, per charter §A21's C2W7 row (deliverable 6, BLOCKING).**
⛔ **`PREREG-TierII.md` IS NOT EDITED.** This block sits beside it and names what changed.

**Why an amendment is required.** C2W7 replaces the read's *answer object*: it is no longer a set
(binary occupancy aggregated by `noisy_or`) but a **weighted counting code** `m ∈ R^{N_a}`,
`m_j = F_hat · cnt_j / Σ_l cnt_l`, produced by a query-driven `F`-commitment and a learned
multiplicity. A reader class is defined by *what a reader consumes*, so the class must be
re-registered and frozen before the first arm — it cannot be widened afterwards.

| reader | consumes | fitted params (`d = 8, m = 8`) | status |
|---|---|---|---|
| `sum_linear` | `z` | 136 | carried |
| `well_table` | `argmax(z)` | 72 | carried (the QUANTISING member, kept deliberately) |
| `knn` | canonicalised `z` | 0 | carried |
| `mlp` | `z` | 108 | carried |
| `soft_well_table` | `π` (noisy-or soft occupancy) | 72 | carried (iteration 1's D8 non-quantising twin) |
| ⭐ **`count_table`** | **`m`** (weighted counting code) | **72** | **NEW (D13)** |
| ⭐ **`count_identity`** | **`m`** | **0** | **NEW (D13)** |

⛔ **The capacity bound is unchanged and unbroken:** all seven are `< N_a·m = 256` (SP-1). The class
keeps ≥ 4 architectures and ≥ 2 non-quantising twins, and it was frozen before the first arm ran.
⛔ **Both new members are applied identically to the physics arm, the null `N1′` and the live
launder** — nothing is added to one arm only.

⭐ **The measurement that forced `count_identity`, filed before the arms ran** (SEEN split, 3 seeds,
launch counting code): the asserted set is exactly `A(x)` on **18.0 %** of queries and on those the
identity residual is **0.006** against `tol = 0.234` — but least squares is dominated by the other
82 %, shrinks the gain to `diag(W) ≈ 0.40`, and pushes the residual on the **good** queries to
**0.537 > tol**. Unseen scores on the *same* latent: **`count_table` (72 params) 0.000 / 0.000 /
0.004** vs **`count_identity` (0 params) 0.172 / 0.227 / 0.168**; a 2-parameter gain+bias reader is
shrunk just as hard (`a ≈ 0.5`) and also scores 0.000.
⛔ **Direction and scope:** this says the registered metric (a *thresholded* exact-set accuracy) and
the registered fitting rule (*least squares*) are mismatched, and that the mismatch — not the
capacity cap — is what can zero an informative latent. It **re-scopes** `tierii-read-fix` §13.3's
open question and it applies retroactively as a *caveat* on iteration 1's zeros, **not** as a
retraction of them (iteration 1's latent was uninformative under the identity too: its own gated
exact-set was 0.0023).

**Registry companion:** to be assigned by the Hub at the C2W7 review.
