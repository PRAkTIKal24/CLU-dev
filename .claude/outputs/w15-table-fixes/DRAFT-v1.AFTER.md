# DRAFT-v1 — the ICLR-long SKELETON for the Naive-CLU paper, revised against REFEREE-v0

**w13 `spoke/draft-revise` · agent `paper-writer` · 2026-09-22.**
**Scored runs spent: ⛔ ZERO. Compute: ⛔ ZERO.** This spoke produced no number. Every figure below
is quoted with the path that holds it.
**Evidence base:** `/c/chlu/evidence/outputs/` (w9-, w10- and w12-advisor-report.md read).
⛔ **`$CHLU_CLAUDE_ROOT` was NOT read at any point.**
⚠ **One source is quoted from OUTSIDE the evidence fold and is flagged at every use:**
`/c/chlu/outputs/w12b-nr6-and-kstar/` — see §0.1.

> ⛔ **THE HEAD WRITES THE PROSE.** This file is structure + claim placement + evidence pointers.
> A claim with no pointer was **deleted, not flagged** (§D).

> ⭐⭐ **AMENDED 2026-09-22 by w15 `spoke/table-fixes`, branch `w15/table-fixes`** — six
> bookkeeping items returned by `w13/referee-2` (`/c/chlu/outputs/w13-referee-2/REFEREE-v1.md`):
> **MF2** (the statistic's own byte cost, with its convention named, in the lead sentence — §I),
> **MF14 part (i)** (the BLIND-SEQUENTIAL and BLIND-FACTOR columns moved INTO the maze main-text
> table — §6.1), and **N5 / N6 / N7 / N8** (§0.2, §3.3, §4.4, §5.1, §5.2, §11.19, §A).
> ⛔ **Zero scored runs, zero compute, no artefact regenerated, `verify_tables.py` NOT touched, and
> `w10/draft-v0` and `w10/referee` NOT merged.** The edit is reproducible from
> `w15_table_fixes/apply_table_fixes.py` and itemised before→after in
> `/c/chlu/outputs/w15-table-fixes/TABLE-FIXES.md`. ⚠ One substantive consequence, reported and not
> softened: with the BLIND columns in §6.1's table, **both request-ignoring references outscore both
> the CLU and the best learned baseline on rows 2 and 3** — see §6.1.

---

## 0. ⭐⭐ WHAT CHANGED SINCE v0 — READ THIS BEFORE ANYTHING ELSE

v0 was **REJECTED** with sixteen must-fixes (`/c/chlu/evidence/outputs/w10-referee/REFEREE-v0.md`
§2). ⭐ **All sixteen are now CLOSED** — the disposition, item by item with how, is §A. None
required a scored run.

But the headline has moved **twice** since v0, and in **opposite directions**. The Head must not
inherit either intermediate state.

| | the composition claim | the artefact that says so |
|---|---|---|
| **v0 (w10)** | led with composition at "an asserted byte pin" | `w10-draft-v0/DRAFT-v0.md` §4 |
| **w12** | ⛔ **does not survive an honest byte match at K = 5** — the store loses to a learner matched to its own bytes at 4 of 5 rungs and, under the pre-declared every-seed bar, clears at **no** rung | `w13-seeds-and-tables/TABLES.md` § TABLE 8; `w12-composition-ladder/LADDER.json` `/rungs/*/arms/single_design_K5/per_seed/*/paired_vs_best_learned` |
| **w12b, 2026-09-22T06:07:28Z** | ⭐⭐ **the sign REVERSES at K\* = 32** — at the K the Head's one-K ruling puts in main text, the store clears the **same** bar at **all five rungs on every seed**, smallest margin **+3.117** | `/c/chlu/outputs/w12b-nr6-and-kstar/KSTAR-LADDER.json` `/verdict/kstar_K32`, `/legs/*/kstar_K32/*/margin` |

### 0.1 ⚠ THE PROVENANCE OF THE REVERSAL, DECLARED

`w12b hub/nr6-and-kstar` wrote its `END_UTC.txt` at **2026-09-22T06:07:28Z**, **two minutes before
this spoke's `START_UTC.txt` (06:09:50Z)**. This spoke's task file clause — *"`w12b` is running the
remaining seeds; if its numbers land before you finish, use them and say so"* — fired. This spoke:

- read that hub's artefacts **read-only**; ⛔ its session, worktree, output directory and processes
  were never touched and no box-wide kill was issued or was possible;
- **re-derived the two load-bearing results itself** rather than quoting the hub's prose: all 15
  per-seed K\* margins from `KSTAR-LADDER.json` `/legs/{n}/kstar_K32/{seed}/margin` (all positive,
  minimum **+3.117**, n = 2 seed 1) and the three-seed T7 rows from `NR6-T7.json` `/arms/cov_tensor`;
- ⛔ **declares that `w12b` is NOT in `/c/chlu/evidence/outputs/`.** It is cited at its live path
  `/c/chlu/outputs/w12b-nr6-and-kstar/` throughout. **`w13-seeds-and-tables/TABLES.md` T7 and T8
  pre-date it and are SUPERSEDED in two rows.** That is **OPEN-1** in §H — the tables must be rebuilt
  and `verify_tables.py` re-run before the paper freezes, or the paper will cite a table whose
  headline row its own source has overturned.

### 0.2 ⭐⭐ THE FOUR SENTENCES THE EVIDENCE NOW FORCES

1. ⭐ **LEAD WITH EXACT DELETION**, store-scoped and defined (MF4), with the four-part byte row
   printed on the same page (MF3). The object that forgets exactly is **24.5466× / 23.8422× /
   23.6697×** the 524,288 B pin on the w8 leg (`TABLES.md` T4 byte ledger — ⛔ of which the
   **statistic alone** is **23.5121× / 22.8077× / 22.6351×**; the two conventions are named wherever
   either is quoted) and **4.0897× / 4.0768× / 4.0847×** per TEST seed on the K\* object
   (`/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T7 byte ledger is **one seed**, id 0 — the other
   two totals are `w12b .../FINDINGS.md` §4; T12's **4.0877×** is the **val** pin leg, ⚠ a fourth
   number and not a fourth seed). ⛔ Quoted with its overrun attached, its convention named and its
   seed count printed, or not at all.
2. ⛔ **The composition claim is K-CONDITIONAL, and the paper must say which K.** At **K = 5** the
   store loses to a 1×-matched learner at 4 of 5 rungs and clears at none; at **K\* = 32** it clears
   at all five on every seed. ⭐ **The `bridge` arm attributes the flip: 62–79 % of every swing is
   the K step, not the construction** (`w12b .../KSTAR-LADDER.json` `/margin_swing_decomposed`) —
   and the bridge, at K = 5 with the K\* construction, **still loses at 4 of 5 rungs**.
3. ⭐⭐ **At K = 5 (MI partition, full-square), `pca_tensor` at 1.358×–1.372× the reference scores
   IDENTICALLY — to four decimals at every rung and every seed — to the single design at
   23.886×–24.875×** (⛔ **a different object from the K = 5 (ward, triangle) of §5.1**, N6) (`TABLES.md` T8 byte
   ledger; `w12-composition-ladder/BYTES.json` `/summary`). ⇒ ⭐ **at K = 5 the four-part ledger buys
   DELETION, not composition.** ⚠ **Whether that equality survives at K\* is NOT MEASURED** —
   `pca_tensor` was not run at K\*; the K\* legs carry `single_design_K32`, `learned_1x` and three
   controls only (`w12b .../kstar_ladder/*/`). **OPEN-2.**
4. ⭐ **The controls collapse, and the two that do not are printed.** At K = 5, `landscape_shuffled`
   and `random_atoms` are beaten by the arm's entire score at every rung (`TABLES.md` T8 control
   columns). At K\*, **13 of 15 legs read exactly 0.0000**; ⚠ **n = 3 seeds 0 and 1 read 0.0083** —
   one surviving novel tuple over 120 corners, judge accuracy 0.0001, against arms of 60.150 and
   62.783 (`w12b .../FINDINGS.md` §1). ⛔ **The brief's "0.000 throughout" is true at K = 5 and is
   not true at K\*. Print the two.** ⇒ the store genuinely carries its score. ⛔ **This is not wave
   9's decorative failure.**

### 0.3 ⛔ WHAT IS **NOT** RESCUED BY THE REVERSAL

- The **1× win at K\* is on a REPLICATION leg, not a scored row.** `KSTAR-LADDER.json`'s first key
  is `REPLICATION_NOT_SCORED: true`. Held out (`kstar_ladder/*/LEG.json` `/corner_check/all_rows_held_out`
  = true; 288,000 train / 192,000 test), 3 seeds, but **0 scored runs drawn**. The scored T7 rows
  beat only the **524,288 B reference** comparators; **the 1× comparator on T7 is still NR8.**
- **`pca_tensor` has no 1×-matched opponent anywhere.** `learned_1x` is matched to the **single
  design's** own total — **3,131,564 – 3,260,030 params** measured across the 15 K = 5 legs
  (`w12-composition-ladder/runs/<rung>_seed<s>/rows_learned_1x.json` `/summary/param_count`) — i.e.
  ≈18× `pca_tensor`'s deployed bytes. ⚠ **The one honest byte match `pca_tensor` does have is w5's**
  (§4.1), and it **wins** it: total deployed **713,632 / 719,352 / 719,352 B** against the rival's
  **715,808 B**.
- **The deletion legs are still one seed** (T9, T10-ours). w12b bought T7's seeds, not T9's.

---

## 1. THE FRAMING

| | |
|---|---|
| ⭐ **LEAD CLAIM** | **Exact, store-scoped deletion of a compositional generative store.** Two implementations, two environments, three seeds each, held out; unclaimed in ≈4,050 screened works. |
| ⭐ **THE SENTENCE THE PAPER CARRIES** | **"Downdate the statistic, not the truncated factors."** |
| ⛔ **NEVER CLAIM** | *"multilinear/bilinear factored generation with a request-indexed tensor head"* — Tenenbaum & Freeman (2000); Vasilescu & Terzopoulos (2002). ⛔⛔ **STOP #1.** |
| ⛔ **NEVER CLAIM** | *"tensor-product label conditioning for unseen attribute combinations"* — MLC-VAE, ICML 2020. |
| ⛔ **NEVER CLAIM** | the downdate **primitive** — Cao & Yang (2015) owns it from the summation form. |
| ⭐ **DO NOT CONCEDE** | exactness to the decremental-PCA line. Hall–Marshall–Martin subtract *deflated* eigenspaces and say so: *"yields a result **approximating**…"*, *"tends to instability"*, *"no rigorous error analysis"*. `NOVELTY.md` §D + `LOG.md` 16:53Z. ⚠ Reading `NOVELTY.md` without its `LOG.md` gives this ground away. |

**Novelty status:** ⚠ **ADJACENT**, on a systematic negative — ≈4,050 unique works across three
independent directions; deletion language in the generation-side set **10 of 3,038**, seven false
positives; frozen-decoder + tensor-head + label-index together **0 of 3,038**; highest atom count
**4 of 6** (CausalX). *Source:* `NOVELTY.md` §§A–E. ⚠ Say **"six of the sweep's 21 queries return
zero"**, not "six orthogonal sweeps" (NTH5).

### §1 Introduction

| slot | what goes here | pointer |
|---|---|---|
| 1.1 | The problem sentence: exact unlearning exists for k-means, bilinear matrix completion and ridge heads on frozen backbones; for *generative* models the literature calls retrain-equivalence out of reach and substitutes approximations. | `NOVELTY.md` §3 (arXiv:2206.14439; arXiv:2603.12977 — *"ridge regression only — no PCA, eigendecomposition, or generative models"*) |
| 1.2 | ⭐ **The contribution sentence.** ⛔⛔ **MF1 — DELETED from v0: *"the survivable form is verbatim there"*.** `NOVELTY.md` §3's final paragraph contains **STOP #1 word for word** (*"a generative model with a request-indexed tensor head"*) and **must be paraphrased, never lifted**. The survivable paragraph is **§I of this file**, written to avoid the phrase by construction. | `NOVELTY.md` §3 final ¶ (cited only as the thing NOT to lift); §I below |
| 1.3 | ⭐ **The mechanism sentence.** *"Downdate the statistic, not the truncated factors."* | `NOVELTY.md` §D; `w13-registry-fold/FINDINGS.md` §1; registry `N-I8-1`(F) |
| 1.4 | Contributions, four, **in this order**: **(a)** exact store-scoped deletion, two implementations, 3 seeds each, held out, **with its byte cost in the same bullet**; **(b)** the interface split (labels touch the model only at the address book); **(c)** composition at a **stated K** against a **1×-matched** learner, with the K = 5 → K\* sign flip reported as the finding it is; **(d)** the route-task negative and its isolated mechanism. ⛔ **The latent-dynamics material is NOT a contribution bullet** — it is rung 3, a *use*. | §§3–6, §8; `w12-positioning-scout/POSITIONING.md` §5.4 |
| 1.5 | ⭐ **The honest-thickness sentence, in the introduction and not buried.** One environment with three scored seeds and a clean instrument returns a **negative**; the positive claim is replicated across two implementations and three seeds. | `w10-seeds-and-tables/FINDINGS.md` §6 final ¶ |
| 1.6 | Headline figure = **`figures/fig4_exact_deletion.png`** (both deletion implementations, log scale, corrected field per MF8). | `w13-seeds-and-tables/figures/fig4_exact_deletion.png` |

⛔ **The introduction must not open on composition.** `NOVELTY.md` §5: composition *"cannot be won on
its own"*. ⭐ **And it must not open on the reversal either** — the reversal is a correction to a
wave-12 reading, not the paper's claim.

### §2 The object

| slot | content | pointer |
|---|---|---|
| 2.1 | The store: per-well leading **principal subspace of the fit pool**; pool mean as the first atom. ⭐ **Unsupervised; no label enters the content.** | `w5-compose-field/LOG.md` §2 |
| 2.2 | The read: a tensor-product address book, **request-only** — `feature_argument_present: false`, `bitwise_stable_under_feature_perturbation: true`, asserted on the deployed object per leg (8 checks/seed on CMAZE-2, 64 on the K\* object). ⛔ **Address bytes declared BESIDE the store and never netted in.** | `TABLES.md` T1 §1e; T7 pins |
| 2.3 | The byte pin. ⛔⛔ **MF13 — v0's *"at 524,288 B"* is deleted.** The supported form: *the **store** is pinned at 524,288 B at tolerance 0; the address book (189,344–195,064 B) is declared beside it; **total deployed bytes are 713,632 / 719,352 / 719,352 against the rival's 715,808 — matched to within ±0.5 %***. ⛔⛔ **MF12 — the parameter match is NOT −0.01 %.** It is **178,408 / 179,838 / 179,838 vs 178,952 = −0.304 % / +0.495 % / +0.495 %**, and **on two of three seeds the arm carries MORE parameters than the bar.** State it that way round. | `TABLES.md` § TABLE 6; `w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json`; the correction bullet at `w5-compose-field/LOG.md` **01:58 UTC** (⛔ **not** the 02:45Z entry v0 cited) |
| 2.4 | The renderer: a **frozen** nonlinear decoder; labels touch the model only at the address book. ⭐ **This is the interface split, and it is contribution (b).** | `NOVELTY.md` §4 claim 2 |
| 2.5 | ⭐⭐ **The frozen encoder and decoder are DECLARED OUTSIDE THE DELETABLE OBJECT** (Head ruling, 2026-09-21, carried in `/c/chlu/tasks/w13-spoke-draft-revise.md`). This is what makes §3's definition honest rather than defensive: the paper states the perimeter **before** it states the claim. | §3.0; `TABLES.json` `/exactness_definition` |
| 2.6 | ⚠ Naming: the paper object is the **Naive-CLU** (Head decision). | `advisor-wave-plan-w8-w10.md` §0 |
| 2.7 | ⛔⛔ **NOTATION, fixed.** **n** = jointly composed factors (the ladder rungs) · **K** = wells. ⛔ Never both written `k`. Every ladder row states its K; every wells row states its n. | `TABLES.json` `/notation`; correction **C19** |

⚠ Write §2 so §2.1 + §2.4 are visibly the **join** of two literatures: the multilinear line (A3+A4)
builds its bases *out of the labels* and renders linearly; the frozen-generator line (A2+A5) has no
label request and no product table. ⛔ Do not present the join as the headline — `NOVELTY.md` §4
makes it claim 2 of 3.

---

## §3 ⭐⭐ EXACT DELETION — the lead result

### 3.0 ⭐⭐ THE DEFINITION COMES BEFORE THE TABLE (MF4)

> **EXACT is STORE-SCOPED.** After deleting a set of rows, the **store content** equals, to float
> roundoff, the content fitted from the retained rows alone, **holding the frozen encoder and
> decoder fixed**. Neither is refit at any point. ⛔ **It is NOT retrain-equivalence of the
> generative model and it is NOT a claim that the renderer forgets.**

*Verbatim from* `w13-seeds-and-tables/TABLES.json` `/exactness_definition`, a citable field.
**Why only this scope is supported:** the frozen autoencoder's own training set contains eval-bank
items — `ae_train_items.eval_bank_train` = **543 / 529 / 552** on seeds 0/1/2
(`w9-maze/runs/scored{1_seed0,2_seed1,3_seed2}_test.json` `/ae_train_items`) — so the weights fitted
with the deleted rows present are untouched. `refit` = **False** on every seed
(`/display/delete_green_exactness/refit`).

⭐ **Two senses of "exact" are kept apart and the paper must keep them apart too:** an exact **JUDGE**
is integer arithmetic with no tolerance (`w9-harness/harness_seed0.json` `/judge_audit/exact` = True);
exact **DELETION** is the store-scoped definition above.

⛔ **Everywhere v0 wrote "of a generative model" or "of a generator", write "of a generative model's
STORE".**

### 3.1 TABLE 4 — both legs side by side, never merged

*Source:* `w13-seeds-and-tables/TABLES.md` § TABLE 4. The deviations are in different units (a
relative residual on the statistic vs an absolute max deviation per well).

**Leg 1 — w9/maze `CluRead.delete_rows_exact`, CMAZE-2, TEST · class RESULT · 3 of 3 scored runs**

| property | seed 0 / 1 / 2 |
|---|---|
| statistic max **relative** residual | **7.522e-16 / 8.826e-16 / 7.818e-16** |
| subspace max **projector** residual | **0.000e+00 / 0.000e+00 / 0.000e+00** |
| rows removed / kept | 216/992 · 226/975 · 193/1084 |
| refit · passes over retained data · items kept · carry bytes deployed | **False · 0 · False · 0** |

**Leg 2 — w8/downdate `exp_w8_downdate`, Shapes3D, TEST · class ⛔ ASSERTION BATTERY (object
property) · 3 seeds · 0 SCORED RUNS** (MF7 — the artefact's own `/note` reads *"No scored run and no
claim"*; the hub commissioned to score it returned 0 of 3, **NR3**. ⛔ The artefact was not relabelled
to match the table; the table was relabelled to match the artefact.)

| property | seed 0 / 1 / 2 |
|---|---|
| ⭐ **MF8 — worst max abs deviation, over the pinned 11,024-row deletion AND the random-64 deletion (the spoke's own row)** | **4.5475e-13 / 1.1255e-09 / 2.8422e-14** |
| that leg's own reproducibility floor | 2.1268e-13 / 4.2817e-09 / 5.6843e-14 |
| ⭐ **MF9 — resolvable at all?** | **True / False / False — TWO of three legs, not one** |
| the narrower field v0 printed, labelled as such | 2.365e-14 / 1.126e-09 / 7.105e-15 |
| negative control fired | True / True / True |
| artefact's own bar | `worst_in_float32_ulp_at_unit_scale` = **0.009442** |

⭐ **MF9 stated the strong way round:** on two of three TEST legs the residual sits **below that leg's
own reproducibility floor**, so it **is not resolvable at all**. An unresolvable residual is the best
available outcome for an exactness claim, and v0 hid two thirds of it.

### 3.2 ⛔ THE CAVEATS THAT LIVE INSIDE THE CITED GATE FILE (MF6) — in the caption, not a footnote

Both are leaf fields of `w8-downdate/DOWNDATE-READY-test.json`, the file the lead result cites:

- `cut_is_marginal` = **True on all three seeds** — a direction sits near the admission threshold, so
  a small perturbation could change which directions are kept: the same instability class as the
  original failure.
- `a1_subspace.passed_every_seed` = **False**, `seeds_failing` = **["2"]**, at **60.36282** float32
  ulp against its own **66.11072** ulp floor — ⛔ **below its own floor**, so that failure is a
  property of the **content**, not of the downdate, which is what the artefact's own
  `reproducibility_floor.what` field states. ⛔ The top-level `all_three_assertions_passed = true`
  **EXCLUDES** `a1_subspace`. A referee opening the lead result's own JSON reads `passed: false`.

### 3.3 ⛔⛔ THE BYTE ROW, ON THE SAME PAGE AS THE CLAIM (MF3)

⛔ **Four parts, printed separately, NEVER netted into one number.**

| leg | atoms | statistic | address book | coupling/holes | honest total | × the 524,288 B pin |
|---|---|---|---|---|---|---|
| **w9/maze (CMAZE-2)** | 31,744 | **35,904** float64 — **2.19 %** of the 1,642,752 B deployed read | 1,611,008 | ⛔ **DECLARED NOT PRESENT** — no field matching `hole` or `coupl` exists in any w9 artefact | 1,678,656 | ⛔ **n/a — this leg is matched on PARAMETERS** (mlp 410,920 / lstm 409,846 / transformer 405,776 against a read of 410,688 floats) |
| **w8/downdate (Shapes3D)** | 524,288 | 12,327,096 / 11,957,784 / 11,867,336 — ⛔ **the statistic ALONE is 23.5121× / 22.8077× / 22.6351× the pin** (statistic ÷ pin) | 11,488 (request-count table) | 5,288 + 1,322 | **12,869,482 / 12,500,170 / 12,409,722** | ⛔⛔ **24.5466× / 23.8422× / 23.6697×** — four-part honest total ÷ pin |
| **the K\* = 32 object (T7/T9), TEST, 3 seeds** | 516,096 | 560,608 | 1,060,864 / 1,054,104 / 1,058,264 | 6,610 | 2,144,178 / 2,137,418 / 2,141,578 | **4.0897× / 4.0768× / 4.0847×** |
| **the K\* = 32 object, ⚠ the VAL pin leg (T12's K = 32 row)** | 516,096 | 560,608 | 1,059,824 | 6,610 | 2,143,138 | **4.0877× — ⚠ val, and labelled val wherever it appears** |

*Sources:* `TABLES.md` T1 §1c-quater, T4 byte ledger, T7 byte ledger;
`w8-downdate/DOWNDATE-READY-test.json` `/honest_total_bytes`, `/times_over_budget`;
`/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T7 byte ledger and T12 K = 32 row;
`/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §4 (the three-seed K\* ledger — ⭐ the task file's
535,784 1× param pin holds across all three seeds to within **0.32 %**, and only part 3 varies with
the seed).
⛔⛔ **N5 — ONE RATIO MAY NOT STAND FOR THREE TOTALS, AND ONE OBJECT MAY NOT CARRY TWO RATIOS
UNLABELLED.** `TABLES.md` **T7 and T9 are ONE-SEED tables** (`seeds (n) | 1`, seed id 0) and their
**4.0897×** is `2,144,178 / 524,288` for **seed 0 alone**; the three-seed totals come from
`w12b .../FINDINGS.md` §4, which prints **three totals and no ratio**. This spoke divided them:
**4.0897× / 4.0768× / 4.0847×**, printed above per seed. ⭐ **Parts 1, 2 and 4 are constant across
seeds — they are arithmetic on the partition shape — and only part 3, the address book, varies**
(the source's own sentence). ⚠ **And the 4.0877× that §5.2 and T12 print for the same object is the
VAL pin leg, a fourth number and not a fourth seed**; it now carries the word **val** in both
places.
⚠ **A DECLARED RESIDUAL, not netted away:** `object_bytes_full − honest_total_bytes` = **exactly
16,384 B on every seed** on the w8 leg; w12r could not attribute it to a named field and reports it
rather than absorbing it (`TABLES.md` T4 byte ledger).

⭐ **The sentence this converts into the paper's cleanest framing:** *the object that forgets exactly
is itself 24.55× the pin the paper measures everything else against — which is why §5 exists.*
⛔ v0 penalised §5's object at 7.48× and exempted its own lead result at 24.5×, one section later.

### 3.4 ⭐ THE PARAGRAPH THE HEAD SHOULD NOT CUT — the failure agreement

⛔ **NEVER "9/9 seeds".** The nine legs are **3 configurations × 3 seeds inside ONE implementation**
(`hub/single-design`, `ward_balanced` K = 40/44/48). There are **three** implementations and their
bands are **not shared**: 7.5e-16…8.8e-16 (w9/maze) · 2.842e-14…1.126e-09 (w8 downdate-build) ·
5.551e-16…1.490e-13 (w8 single-design). *Source:* `TABLES.md` § TABLE 5, which prints them as three
rows with seed count and configuration count as **separate fields**; correction **C12**.

⭐⭐ **The strongest form of the two-implementation claim is the FAILURE AGREEMENT, not the
successes.** The exactness assertion **failed first**, on TEST seed 2, on the incumbent's coarse
imbalanced `factor K = 5, A = 64` partition:

| leg | float32 ulp at unit scale | its own floor |
|---|---|---|
| w8 spoke/downdate-build, atom level | **32.90** (and 35.46) | — |
| w8 spoke/downdate-build, subspace level | **60.36282** | **66.11072** |
| w8 hub/single-design, atom level | **14.377** | — |
| w8 hub/single-design, subspace level | **14.4518** | **36.9305** |

⭐ **Two implementations, different code, different authors, different sign conventions, reproduced
the same failure on the same seed in the same well — and both implementations' own reproducibility
floors EXCEED their observed deviations there**, so both independently conclude that **seed 2's
content is not a function of its data by any method.** ⭐ That makes this a statement about **content
well-posedness**, which is a better finding than a downdate failure.

**How it was closed:** a variance-share admission cut with **no free parameter**,
`tau = 4.941056e-16 / 1.1920929e-07 = 4.144858e-09`, declared before the re-run and not moved.
⚠ **SF3 — say where the numerator comes from:** `delta_rel_worst = 4.9411e-16` is **test seed 2's own
leg worst** — the failing leg — and the spoke declined to round it up *because that would have made
seed 2 more likely to pass* (`w8-downdate/LOG.md:840-855`).

*Sources:* `TABLES.md` § TABLE 5 failure-agreement block; `w8-downdate/LOG.md:618-640,728,794,905`;
`w8-single-design/LOG.md:479,481,501,506`.

### 3.5 ⭐ EXACT IS NOT FREE (MF5) — one sentence, at the end of §3, not four sections away

> *Exact does not mean free. On TEST the same deletion takes surviving-target composition from
> **0.1024 (0.1569 / 0.0702 / 0.0802) to 0.0000** (§11.4); on the item store, evicting `red` refuses
> perfectly (**1.0000** on all three seeds) and costs **−2.0899 / −0.9101 / −6.7416** composition
> against a declared ±1.0 bar, while punching a **hole** refuses at 1.0000 and costs **exactly
> 0.0000**. The claim is exactness, never cost-free erasure.*

*Sources:* `TABLES.md` T1 §1d post-deletion block; `TABLES.md` T11 (`w5-delete/run1/report.json`
`/across_seeds/evict_composition_mean_delta`, `/hole_composition_mean_delta`);
`w5-advisor-report.md` §6.3 — the programme's own ruling: *"'the hole is exact and free' is
supported; 'deletion is exact and free' is NOT."*
⚠ `row_removal_s` reads **0.0000 on all three seeds** — a measured zero below that harness's timer
resolution, printed with its pointer, **not a missing value**.

### 3.6 ⛔⛔ THE PROBE THAT BOUNDS THE WHOLE CLAIM — main text, not an appendix

At K\* = 32, **deleting all 11,044 red rows leaves the 20 red-requesting corners at 32.2500 →
32.2500** and judge accuracy at **0.8958 → 0.8958**. The store-scoped exactness is real — the
**uncarried** read moves **0.610083** relative RMS on red-requesting corners — but the basis change
is **generic, not targeted**: **0.614277** on the corners that do *not* request red. Carried, the read
moves **6.194e-08**.

⇒ ⛔⛔ ***"The store has forgotten red"* and *"the object no longer answers red"* are different
statements, and at K\* only the first is demonstrated.** At K\* the value identity lives in the
**address book**, which the deletion **carries and does not refit** (**1,015,808 of 1,060,864 B**
changed, **0 data rows touched**).

⭐ **And the trade-off did not vanish — it MOVED.** At K = 5, rank/width 0.156, the same deletion
**was** concentrated: `cos_min` **0.1999**, 93.8 % of directions above 0.99, against **0.9976** for a
size-matched random deletion of the same rows. ⇒ **the targeting of a deletion is a function of
rank/width, and it disappears exactly where the byte ledger is cheapest.**

*Sources:* `TABLES.md` T9 probe block;
`w12-single-design-crossover/deletion/probe_test_seed0_K32.json` `/carry/*`;
`w8-single-design/LOG.md` §4.2; registry **`N335`** / matrix **`CM-W12-10`**.

### 3.7 ⛔ THE LANGUAGE RULE ON POST-DELETION COST

⛔ **NEVER "no changes after deletion".** The sentence is:

> *Post-deletion adaptation is **LIGHT and DATA-FREE** — the covariance and count are downdated by
> subtraction, the atoms are re-eigendecomposed from the downdated statistic, and the address book is
> carried across the basis change, with **no row of data read** — against a reference that can delete
> only by refitting from scratch without the deleted rows, reading **120,028–131,072 rows** to do it
> (**0.8998 s vs 9.259 s** on the value scope).*

*Source:* `TABLES.md` T9 language-rule block and cost columns.

### 3.8 The deletion comparator rows the paper owes (the Head's naive-vs-naive amendment)

⭐ Paper rows carry **(i)** retrain-from-scratch-without — the naive rival's only deletion route, cost
printed — **(ii)** the rival as-trained, **(iii)** the refuse-token transformer.
⛔ **DROPPED from paper rows by ruling:** approximate unlearning on the rival's head; sharded
ensembles (SISA) and the voting CLU ensemble are **UNBUILT and stay so**.
⚠ **The extra abstain output for MLP / cont-MLP / LSTM is NOT BUILT — DECLARED NOT-RUN** (OPEN-6);
the spoke ran before the amendment.
⚠⭐ **A correction the programme owes itself:** the two unlearners are **not "unrun"** — they were
**measured on the fit pool** (0 scored runs, no held-out corner) and the numbers are real: **NegGrad+
over-forgets 9.4×–20.8×; fine-tune under-forgets on three arms of four**, at **296,960 matched
gradient row-visits asserted equal across all eight arm × method combinations on all three seeds**
⇒ ⛔ **not a starved-comparator artefact.** What is unrun is **scoring** them.
⭐ **HELD for post-submission by Head ruling — availability is not authorisation.**
*Sources:* `w12-deletion-baselines/FINDINGS.md` §§0–2; registry **`N334`** / **`CM-W12-9`**.

### 3.9 A second, independent deletion property, from the physics block

**Decay commutes with deletion**: `evict ∘ decay^n == decay^n ∘ evict`, **0 bytes differing in a
2,054,866 B image, 12/12 cells per seed, all three seeds, zero tolerance** — and on a **learned**
store, which V5 does not claim. *Source:* `w10-physics-section-evidence/EVIDENCE.md` §3.5 (D32a).
⛔ Its companion **D32b FAILS as pre-registered** on all three seeds and **must travel with it**
(§11.5).

**Figure slot:** `figures/fig4_exact_deletion.png`.

---

## §4 ⭐ COMPOSITION — and the K it is conditional on

> ⛔⛔ **This section may not be written without naming its K.** Same object, same rungs, same
> pre-declared bar: **K = 5 clears at no rung; K\* = 32 clears at all five.**

### 4.1 ⭐ The one composition row matched to its own bytes on a scored TEST leg — w5

**The claim, in the only form the artefacts support:** `pca_tensor` beats `rival_mlp_180k` on **both**
target rows on **all three seeds**, beyond a margin declared before the first held-out number, at
**total deployed bytes matched to ±0.5 %**.

| | row 4 (49 n-way corners) | row 3 (60 pair sub-corners) |
|---|---|---|
| arm `pca_tensor` | 63.6122 / 64.3265 / 67.6531 | 29.7167 / 31.6167 / 31.4500 |
| `rival_mlp_180k` | 53.4694 / 55.6531 / 59.8571 | 25.6833 / 29.1333 / 29.0333 |
| mean paired Δ | **+10.143 / +8.673 / +7.796** | **+4.033 / +2.483 / +2.417** |
| bootstrap 2.5 % quantile | +6.632 / +5.816 / +5.082 | +2.117 / **+1.033** / +1.200 |
| bar (Δ ≥ +1.0 **and** CI95-low > +1.0, every seed) | ⭐ **PASS 3/3** | ⭐ **PASS 3/3** |
| **total deployed bytes** | **713,632 / 719,352 / 719,352** vs the rival's **715,808** = **99.70 % / 100.50 % / 100.50 %** | same |
| **parameters** | **178,408 / 179,838 / 179,838** vs **178,952** = **−0.304 % / +0.495 % / +0.495 %** | same |

*Sources:* `TABLES.md` § TABLE 6 and § TABLE 11;
`w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json`; `w5-compose-field/LOG.md:196,247` and the
**01:58 UTC** correction bullet; `w5-advisor-report.md` §1. Condition `test`, cache generation 2,
seeds 0/1/2, 49 + 60 paired corners, 20 arms scored per seed, zero failures.
⚠ **Print the tight leg as tight.** Row 3 seed 1 clears +1.0 by **0.033**. The hub refused to call it
comfortable; the paper does not get to either.

⭐ **The floor that counts, declared before the numbers.** The two matched-bytes lookups score 0.000
and the hub **declined to claim them as a §C.1 pass** (a single principal *direction* is not a
plausible answer, so 0.000 is nearly automatic). The informative floor is `oracle_content_items` — 64
whole written rows at the same 524,288 B, read by an **oracle** — at **32.3878 / 39.1224 / 41.7755**
on row 4. The arm beats it by **+31.224 / +25.204 / +25.878 while the oracle is handed the true
feature and the arm is not.**

⭐ **And this prices contribution (a) against (c):** the item store, which *can* forget by
construction, sits **−21.0816 / −16.5306 / −18.0816** below the learned reference on row 4 — a 3-seed
mean of **−18.56 tuples/corner**, ⚠ and a **lower bound**, because the item store is a **ceiling
handed the true feature, not an arm**. *Source:* `TABLES.md` T11.

⚠ **SF1 — use `oracle_span_field`, not `oracle_span`.** v0's numbers (−12.531 / −5.224 / −5.653) are
`oracle_span_field`'s; `oracle_span` reads −9.286 / −4.755 / −6.857. **Two different ceilings.**
⚠ **SF2 — print `address_bytes` per rung of the ceiling ladder.** `oracle_span` and `span_head` carry
**1,766,400 B** — **9.3×** the arm's own book and **3.4×** the store pin — while every other rung
carries 0. A byte-ledger paper cannot ship a ladder whose rungs differ 9× in book size silently.
*Source:* `TABLES.md` T6 ceiling-ladder block.

### 4.2 ⛔⛔ THE LADDER, AND THE SIGN FLIP — the section's hardest paragraph

**The pre-declared bar** (`w12-composition-ladder/AMENDMENT-2` §3, frozen 11:46:30Z before any arm
number existed): the arm clears only if it beats the opponent by **1.0 novel tuple per corner on every
declared seed** — not on the seed mean. ⛔ **The bar is not moved anywhere below.**

| rung | ceiling-normalised Δ vs `learned_1x`, **K = 5** | clears at K = 5? | per-seed margin, **K\* = 32** | clears at K\*? |
|---|---|---|---|---|
| n = 2 | **−0.0773** | ⛔ no (−4.567 / −2.217 / −1.567) | **+6.667 / +3.117 / +3.283** | ⭐ **YES** |
| n = 3 ⚠ | **+0.0589** | ⛔ **no — 2 of 3 seeds** (+4.817 / **−0.033** / +7.942) | **+17.875 / +12.808 / +11.533** | ⭐ **YES** |
| n = 4 ⚠ | **−0.0734** | ⛔ no (−6.242 / −6.350 / −3.258) | **+12.700 / +8.742 / +8.625** | ⭐ **YES** |
| n = 5 ⚠ | **−0.0542** | ⛔ no (−4.808 / −4.250 / −2.642) | **+10.350 / +6.517 / +8.200** | ⭐ **YES** |
| n = 6 ⚠ | **−0.0564** | ⛔ no (−5.775 / −4.567 / −1.842) | **+9.375 / +6.108 / +8.450** | ⭐ **YES** |

*K = 5:* `TABLES.md` § TABLE 8; `w12-composition-ladder/LADDER.json`
`/rungs/{n}/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned/mean_paired_delta`.
*K\*:* `/c/chlu/outputs/w12b-nr6-and-kstar/KSTAR-LADDER.json` `/legs/{n}/kstar_K32/{s}/margin` —
**re-derived independently by this spoke; all 15 positive, minimum +3.117.**

⭐ **Report n = 3 at K = 5 as UNEXPLAINED, not as a win.** The store sits at **its own lowest
ceiling-normalised rung (0.8299)** while the learner drops to **0.7709** from a tight 0.9256 / 0.9443
/ 0.9530 / 0.9530 elsewhere. **The gap opens because the opponent fell off, not because the arm
improved**, ⛔ **no wave-12 artefact records a mechanism and the registry does not invent one** — and
it **fails the every-seed bar anyway**. *Source:* registry **`N326`** / **`CM-W12-1`**.

⭐⭐ **The bridge arm attributes the flip, and nobody asked for it.** K = 5 → K\* changes **two** things
at once — the well count **and** the construction (a `DowndatableStore` on the label-MI `factor`
partition vs a `CovContentStore` on the frozen label-free `ward_balanced` cut, with a different
covariance convention). w12b held the construction fixed and set K back to 5:

| rung (seed 0) | ladder K = 5 | **bridge K = 5** | K\* | construction step | **K step** | K's share |
|---|---|---|---|---|---|---|
| n = 2 | −4.567 | **−0.250** | +6.667 | +4.317 | **+6.917** | 62 % |
| n = 3 | +4.817 | **+8.425** | +17.875 | +3.608 | **+9.450** | 72 % |
| n = 4 | −6.242 | **−1.017** | +12.700 | +5.225 | **+13.717** | 72 % |
| n = 5 | −4.808 | **−0.517** | +10.350 | +4.292 | **+10.867** | 72 % |
| n = 6 | −5.775 | **−2.550** | +9.375 | +3.225 | **+11.925** | 79 % |

⭐⭐ **The bridge STILL LOSES at 4 of 5 rungs — the identical pattern to the ladder's own K = 5
verdict.** ⇒ **Changing the construction alone flips nothing. The flip is the K step.** ⛔ Without the
bridge the whole swing would have been charged to K and the claim would have been unattributable.
⚠ **The bridge is ONE SEED (seed 0) on all five rungs** — a single-seed attribution, labelled as one.
⚠ **The K\* column itself carries all three seeds.**
*Source:* `/c/chlu/outputs/w12b-nr6-and-kstar/KSTAR-LADDER.json` `/margin_swing_decomposed`;
`FINDINGS.md` §2.

⭐ **Both halves of the flip move, and it matters which.** The store's own score **rises at every
rung** (+1.86 to +3.38 novel tuples/corner) — the finer partition genuinely reads better, and that is
not a baseline artefact — while the opponent **falls 5.92–5.99×**, because it is byte-matched to an
object whose four-part total is that much smaller (`w12b .../FINDINGS.md` §1).

### 4.3 ⚠⚠ THE AXIS MUST BE LABELLED HONESTLY — for n ≥ 3 this is a TRAINING-DATA axis

⛔ **Rungs n = 3, 4, 5, 6 probe the IDENTICAL 120 corners.** What varies is how coarsely the hold-out
is carved, and with it the held-out row count (**300,000 / 235,200 / 213,600 / 200,640**). Every arm
fitted on the rung's own pool tracks that count at **r = −0.96…−0.98**; every arm not fitted on it is
**exactly constant** across those four rungs. ⭐ **Only n = 2 vs n = 6 is a genuine depth contrast.**
*Sources:* `TABLES.md` T8 instrument block; `KSTAR-LADDER.json` `/axis_label`,
`/identical_corner_rungs` = ["3","4","5","6"]; registry `N327` / `CM-W12-2`.

⭐ **And the flip is present at both ends of the only genuine contrast** — n = 2 (−2.783 → +4.356) and
n = 6 (−4.061 → +7.978) — ⛔ **so it does not rest on the confounded middle rungs.**

⭐ **The store does not deteriorate with depth — it improves**, ceiling-normalised 0.8483 → 0.8966 at
K = 5. ⛔ **The wave's own question presumed deterioration; the premise is false for this arm**, and
⚠ **for n ≥ 3 the rise may be a held-out-fraction trend rather than a depth trend.**

### 4.4 ⭐⭐ WHAT THE BYTES BUY — the cleanest statement in the paper, and its scope

⛔⛔ **NAME THE OBJECT: this is K = 5 (MI partition, full-square)** — the w12 ladder's
`DowndatableStore` on the label-MI `factor` partition, well widths 777/936/216/60/59, statistic
priced as the FULL SQUARE. ⛔ It is **NOT** the K = 5 of §5.1/§5.2, which is **K = 5 (ward,
triangle)** — the crossover/bridge `CovContentStore` on the label-free `ward_balanced` cut, five
≈410-wide wells, statistic priced as the SYMMETRIC TRIANGLE. The two four-part totals differ by
**3.1×** (12,523,008–13,041,864 B here vs 4,118,034 B there) because they differ in **both**
convention **and** partition — ⛔ roughly **2× from the convention alone**, the rest from the
partition. ⚠ Correction **C20** in §G compares 12,284,920 B against 3,391,552 B and, because it moves
both at once, **reads as a 3.6× convention effect when the convention alone is ≈2×**. ⭐ **Use the two
names everywhere; a referee who meets two "K = 5" byte totals 80 lines apart with no cross-reference
reads a contradiction, and on the page that is what it is.** See §5.1 for the other object.

At **K = 5 (MI partition, full-square)**, over 18 fitted legs: the single design costs
**12,523,008 – 13,041,864 B = 23.886× – 24.875×** the 524,288 B reference; `pca_tensor` costs **712,072 – 719,352 B = 1.358× – 1.372×** —
**≈18× less** — **for composition scores identical to four decimals at every rung and every seed.**
⇒ ⭐⭐ **At K = 5 the four-part ledger buys EXACT DELETION, not composition.**
*Source:* `TABLES.md` T8 byte ledger; `w12-composition-ladder/BYTES.json` `/summary`; registry
**`N328`** / **`CM-W12-3`**. ⛔ **Part 4 is DECLARED ABSENT from `BYTES.json`, not printed as a zero.**

⚠ **OPEN-2, stated in the paper and not styled around:** whether that equality survives at K\* is
**NOT MEASURED**. `pca_tensor` was not run at K\*. If it survives, the deletion-vs-composition split
holds at K\* too; if it does not, K\* is the first configuration where the statistic's bytes buy
composition as well. ⛔ **The paper may not assert either.**

### 4.5 ⭐ THE SCORED ROW — T7, now three seeds

⛔⛔ **SUPERSEDING the brief: T7 is no longer a one-seed row.** w12b spent `single-design` runs **2/3
and 3/3**.

| K | n seeds | row 3 (60 pair corners) | row 4 (49 n-way corners) | judge accuracy |
|---|---|---|---|---|
| **32 = K\*** | **3** (0, 1, 2) | **32.350 ± 0.815** (31.600 / 32.233 / 33.217) | **67.177 ± 1.320** (66.959 / 65.980 / 68.592) | 0.9012 / 0.9048 / 0.9362 |
| **5** | ⚠ **1** (seed 0) | 28.817 | 62.122 | 0.8285 |

⭐ **The seed-0 headline sits INSIDE the three-seed spread on both rows — T7 is not a one-seed
artefact.** ⭐ **The control gate was MEASURED at seeds 1 and 2, not assumed:** all four controls read
**exactly 0.0000** on all four legs. Every comparator is beaten beyond the 1.0 margin on **109 of 109
corners** on every leg; the smallest paired delta against any comparator is **+35.560** (the `table`
FLOOR, K = 5 seed 0).

⚠ **Seeds 1 and 2 carry rank-deficient wells the one-seed row never showed:** `rank/width` at K\* is
**0.9688–0.9844** (seed 0), **0.5938–0.9844** (seed 1), **0.4375–0.9844** (seed 2). The saturation
check still passes at the maximum and the controls still collapse — but **the one-seed row understated
the spread of the object's geometry.**

⛔ **The 1× comparator column on T7 remains DECLARED NOT-RUN (NR8)**, and the error runs **against**
the arm: a 1×-matched rival would carry ≈4.09× these parameters and be **stronger**, so the printed
comparison **flatters** the arm and no row above is a 1× result.

⭐ **The val/TEST gap, the third pre-declared falsification, is now testable and DOES NOT FIRE.**
The frozen rule is `K* = max{K : v(5) − mean_seeds v(K) ≤ 0.02}`. The drop is **−0.03376** on val
(3 seeds) and **−0.07276** on TEST (paired, ⚠ seed 0) — **negative on both**, i.e. K\* is *better*
than K = 5 on the held-out condition and by a **wider** margin there than on the selection set.
⚠ The corner sets differ (24 val vs 60 + 49 test), so levels are never subtracted across conditions;
the **drop within a condition** is the rule's own quantity.

*Sources:* `/c/chlu/outputs/w12b-nr6-and-kstar/NR6-T7.json` `/arms/cov_tensor`, `/controls`,
`/verdict`, `/val_test_gap`; `FINDINGS.md` §§4–5; `TABLES.md` T7 (⚠ **superseded in its seed
column** — OPEN-1).

### 4.6 ⭐ THE SATURATION TRAP DOES NOT FIRE, AND THE SPHERE HAZARD WAS MEASURED

- **rank/width 0.9844 on the val sweep at K\*** — above wave 9's decorative threshold of 0.938 — with
  `landscape_removed`, `landscape_shuffled`, `landscape_shuffled_across_wells` and `random_atoms` all
  reading **exactly 0.000** (`TABLES.md` T12 control gate; T7). ⇒ ⛔ **wave 9's decorative failure is
  NOT a property of high rank/width per se.**
- **The sphere hazard was MEASURED and does not reproduce.** A degraded render here is judged
  **cube** — the shape held out of the split entirely — so judged-sphere excess is **negative** for
  both maximally degraded arms: **−0.3291** (`landscape_removed`) and **−0.3078** (`random_atoms`),
  against **+0.0074** after the red-sphere deletion. *Source:* `TABLES.md` T9 sphere-caveat block.
- ⭐ **The conjunction pin is `object_hue = 0.0 ∧ shape = 2.0`, which IS "red ∧ sphere"** (Head ruling:
  name it *red sphere* everywhere). ⛔ **"red ∧ cube" deletes ZERO rows and is VACUOUS** — cube is the
  condition's own held-out shape. `partition_fit_rows` refuses any no-op deletion, so a zero-row
  deletion can never be printed as a deletion result. *Source:* `TABLES.md` T9 rows (b)/(b′);
  registry `N332` / `CM-W12-8`.

**Figure slots:** `fig6_composition_ladder_n.png` (⚠ K = 5 only — no K\* panel exists, **F10**);
`fig7_k_curve_wells.png`; `fig8_four_part_ledger_across_K.png`.

---

## §5 ⭐⭐ THE JOIN — compose XOR forget, and what the nine-K sweep did to it

### 5.1 The chain, with MF16's qualifier and the measurement that supersedes it

| link | number | source |
|---|---|---|
| exact per-item deletion works on the covariance store | downdate == refit, 0 rows refitted, 0 passes, items never kept | `w8-single-design/LOG.md:553-620` §2(1) |
| ⚠ **SF14 — the covariance costs the symmetric TRIANGLE at float64**, `Σ_k d_k(d_k+1)/2 × 8 B + D×8 + K×8` — ⛔ **not `Σ_k d_k²`.** v0's form overstates the count 2× and understates the dtype 2×, and **K ≥ 36 is entirely a float64 fact** | *"Below K = 36 **no atoms fit at all** — the **float64** statistic alone exceeds 524,288 B"* | `w8-single-design/LOG.md:55,89,121,148` |
| ⛔⛔ **MF16 — the K ≥ 36 collapse is conditional and must be printed conditionally** | *"at **K = 40, A = 7** (span 280, **0.991×** the pin — **the one configuration at K ≥ 36 whose composition was measured**) composition collapses to **0.0648**"*, footnoting the **four** unmeasured pin-fitting configurations (K = 36/38/40/42/48, spans 144/266/360/504/864 — three of them carrying MORE deployed span than the object that composes at 0.8918) and the excluded **float32** point (K = 21, A = 94, 517,952 B = 0.988×, span 336) | `w8-single-design/LOG.md` 18:32Z sweep and `LOG.md:99` |
| at K = 5 composition holds | **0.8918** vs the incumbent's **0.8941** — a tie — at an honest four-part ledger of **4,118,034 B = 7.8545×** the pin. ⛔⛔ **NAME THE OBJECT: this is K = 5 (ward, triangle)** — the crossover/bridge `CovContentStore` on the label-free `ward_balanced` cut, five ≈410-wide wells, SYMMETRIC-TRIANGLE statistic. ⛔ It is **NOT** §4.4's **K = 5 (MI partition, full-square)**, whose four-part total is **12,523,008 – 13,041,864 B = 23.886× – 24.875× — 3.1× larger**, on a different partition **and** a different convention. See §4.4 and correction **C20** (⚠ which moves both at once and so reads as a 3.6× convention effect; the convention alone is ≈2×). | `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T12 K = 5 row; `/c/chlu/outputs/w12b-nr6-and-kstar/KSTAR-LADDER.json` `/legs/*/bridge_K5/0/ledger_four_part` |

⭐⭐ **AND THE OPEN PROBLEM IS NOW MEASURED ACROSS NINE K, NOT ARGUED FROM ONE.** The crossover hub
swept **K ∈ {5, 8, 10, 12, 16, 20, 24, 28, 32}** and the honest four-part total **never reaches the
524,288 B pin at rank 64 under any of them**: the minimum over the whole sweep is **3.9942×
(2,094,130 B) at K = 24**. ⇒ the paper's central open problem becomes *"**no object with a float64
sufficient statistic composes and forgets inside 524,288 B — on nine swept partitions, and on the one
K ≥ 36 configuration whose composition was measured**"* — still the wave's answer, now the answer the
evidence supports, and **much harder for a referee to break than v0's unconditional form.**
*Source:* `TABLES.md` T12; registry `N331` / `CM-W12-6`.

### 5.2 ⛔⛔ THE BYTE CURVE IS TWO CURVES, AND v0 QUOTED THE WRONG ONE (C16)

| | K = 5 | K = 24 | K\* = 32 | shape |
|---|---|---|---|---|
| parts **1+2+4 only** (⛔ **no address book**) | 7.4815× | 2.4151× | **2.0663×** | falls monotonically |
| ⭐ **the HONEST four-part total** | **7.8545×** | **3.9942× ← minimum** | **4.0877× ⚠ val** | ⛔ **U-SHAPED** |
| address book | 195,584 B | 827,904 B | **1,059,824 B** | **grows 5.4×** |
| covariance + count | 3,391,552 B | 735,328 B | 560,608 B | falls 6.0× |

⚠⚠ **N5 — every cell of this table is the VAL sweep leg, and its K\* column is NOT the TEST row of
§3.3.** T12's K = 32 honest total is **2,143,138 B = 4.0877×** (address book 1,059,824 B, the val
pin); §3.3's three **TEST** seeds are 2,144,178 / 2,137,418 / 2,141,578 B = **4.0897× / 4.0768× /
4.0847×**. ⛔ The four numbers are one object measured on two conditions and must never be quoted as
a seed spread. *Source:* `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T12 row `K = 32`;
`/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §4 row `(val pin)`.

⛔ **Any sentence quoting "7.48× → 2.07×" is quoting the three-part curve and has dropped part 3** —
and it is read from a file whose **first key is `SMOKE_NOT_A_CLAIM: true`**
(`w12-single-design-crossover/LEDGER-ACROSS-K.json`). ⛔ **The mechanism sentence *"the covariance
shrinks faster than the address book grows"* is TRUE from K = 5 to K = 24 and FALSE from K = 24 to
K = 32.** ⭐ **Quote the direction, not the mechanism unconditionally — and K\* is NOT the cheapest
point.** *Source:* `TABLES.md` T12 caption; correction **C16**; registry `N331`.

### 5.3 The repair that makes post-deletion composition survivable, and its price

Carrying the **book** rather than the atoms — `W_k ← W_k M_k`, `M_k = U_kᵀ U'_k`, formed at delete
time, **no data row read, zero deployed bytes** — recovers judge-space composition **0.0584 → 0.8964
against 0.8941 before deletion = 100.3 % recovery with no refit of anything**.
⚠ **SF5 — the carry rewrites 125,952 B of 158,664 B, NOT 127,488 B.** 127,488 was the **pre-cut**
figure, superseded at `w8-downdate/LOG.md:972` and its FINAL REPORT `:1067`;
`w8-single-design/LOG.md:631` still carries the stale number, which is how v0 inherited it.
⛔ **Never quote the carry rewrite beside the coupling hole's 1 byte.** ⚠ **`val`, not TEST.**
⭐ **The control that says the carry is targeted at K = 5:** `M_k`'s principal-angle spectrum under the
red deletion is **cos_min 0.1999, median 0.9999, 93.8 % above 0.99**, against **cos_min 0.9976** under
a size-matched random deletion of the same 11,044 rows. ⛔ A property of the store, **not** a
composition number. ⚠ **And §3.6 shows it does not hold at K\*.**

### 5.4 ⛔ THE DECLARED NOT-RUNS THAT TRAVEL WITH THIS SECTION, IN MAIN TEXT

**NR3** — partially discharged by w12/w12b: T7 is now the K\* object's scored composition at **3
seeds**; T9/T10-ours are its deletion and refusal legs at **1 seed**. ⛔ **The remaining two seeds of
T9/T10 are OPEN (OPEN-3).** **NR6** (K = 5 on TEST at seeds 1 and 2; the bridge at seeds 1 and 2) ·
**NR7** (any 5-seed row) · **NR8** (the 1× comparator column on T7) · **NR9** (the refusal
head-to-head) · **NR10** (`generation-table` run 3 — ⭐ **unspent**). *Sources:* `TABLES.md`
§DECLARED NOT-RUN; `/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §9.

⛔⛔ **NO ROW IN THIS PROGRAMME CAN EXCEED 3 SEEDS, and that is a MEASUREMENT, not an omission.**
Seeds 3 **and** 4 pass the generation-2 cache rule (`array_shape [480000,128,4,4]`, 59 block hashes,
`status READY`) and **neither can be judged**: `decoder_v1_s{3,4}.pkl` and
`gen-external-classifier/seed_{3,4}` are **both ABSENT**, and `winner_hashes.json`'s keys are exactly
`['seed_0','seed_1','seed_2']`. ⛔ **Never pad to 5.**
*Source:* `/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §10 — ⚠ which **corrects** the earlier
reading that only seed 4 lacked a classifier (**C21**); `TABLES.md` §NR7.

---

## §6 ⛔ IT DOES NOT TRANSFER TO A ROUTE TASK — main text, not an appendix

### 6.0 ⛔⛔ THE ENVIRONMENT IS **CMAZE-2** (MF10)

> **CMAZE-2** is *a REQUEST LAYER over CMAZE-1: the engine, the solver and the exact integer judge are
> `w8_envscout.maze_colour`, imported unmodified; the layer adds named agents and named targets inside
> one maze, placed across two door-free components, and seals one target behind walls per maze.*
> **CMAZE-1** is that generator called with its own defaults, and it is **TABLE 3**, the harness tier.

⛔⛔ **T1 (CMAZE-2) and T3 (CMAZE-1) MUST NEVER SHARE A TABLE. A DIFFERENT ENVIRONMENT, not merely a
different budget.** Without the relabel, correction C4 contradicts itself on the page and the triple
**0.4980** (published CMAZE-1) / **0.5317** (CMAZE-1 at the harness tier) / **0.9037** (CMAZE-2) is
unreadable. *Sources:* `TABLES.md` § DEFINITION — CMAZE-1 and CMAZE-2; `w9-maze/LOG.md:366-372`;
corrections **C4**, **C9**; registry `N-I8-17`.

### 6.1 TABLE 1 — CMAZE-2, held-out TEST, 3 of 3 scored seeds

Pins: K = 8, n_atoms = 31, ridge 0.01, AE 12,000 steps, read = 410,688 floats (**1,642,752 B
deployed**), float64 statistic **35,904 B = 2.19 %** declared beside it and never netted in; baselines
matched at mlp 410,920 / lstm 409,846 / transformer 405,776 params. ⛔ **Not under the 524,288 B pin —
matched on PARAMETERS.** *Source:* `TABLES.md` § TABLE 1.

| row | CLU | best learned baseline | lookup (matched bytes) | `random_atoms` (C.2) | **BLIND-SEQUENTIAL** (ref.) | **BLIND-FACTOR** (ref.) | pipeline ceiling |
|---|---|---|---|---|---|---|---|
| 1 seen | **1.0000** | 0.9865 (transformer) | **1.0000** | 0.9601 | ⛔ **n/a — not a composition row; no blind block** | ⛔ **n/a — not a composition row; no blind block** | 1.0000 |
| 2 unseen pair | 0.1279 (0.1590 / 0.0597 / 0.1651) | **0.1840** (transformer) | 0.1565 | 0.1414 | **0.9037** (0.9077 / 0.8905 / 0.9128) | **0.4762** (0.4564 / 0.4677 / 0.5046) | 0.6319 |
| 3 n-way, CONJUNCTION | 0.0284 (0.0383 / 0.0469 / **0.0000**) | **0.0493** (lstm) | 0.0159 | 0.0195 | **0.9001** (0.9229 / 0.8933 / 0.8841) | **0.4720** (0.4447 / 0.4665 / 0.5048) | 0.5255 (per route) |
| 3 n-way, PER ROUTE | 0.0852 | **0.1246** (transformer) | 0.0946 | 0.0693 | **0.9001** (0.9229 / 0.8933 / 0.8841) | **0.4720** (0.4447 / 0.4665 / 0.5048) | 0.5255 |
| 4 budget interpolation | 0.0996 | **0.1430** (transformer) | 0.1122 | 0.0905 | ⛔ **DECLARED NOT MEASURED — not a blank, not a zero** | ⛔ **DECLARED NOT MEASURED — not a blank, not a zero** | 0.5573 |

⛔⛔ **MF14(i) — READ THE TWO BLIND COLUMNS BESIDE THE CLAIM, WHICH IS WHY THEY ARE HERE.** They are
**REFERENCE** columns in the artefact's own class field, not arms: **BLIND-SEQUENTIAL** = *key order
from A*, **BLIND-FACTOR** = *A × B local only* — request-ignoring policies scored by the **same exact
integer judge, on the same held-out TEST rows, on the same three seeds**. On rows 2 and 3 **both of
them beat the CLU and both beat the best learned baseline**: BLIND-SEQUENTIAL by **7.1×** and
**31.7×** the CLU's own score, BLIND-FACTOR by **3.7×** and **16.6×**.
⛔ **This makes the maze section's negative HARDER, not softer, and it is printed that way on
purpose:** on this split a policy that never reads the request scores 0.90, so ⛔ **the split does
not test composition at all** — which is exactly what §6.3's measured cause says (1,242 of 1,449
optimal routes pick up exactly one key, so rule B has no key ORDER to trap). ⭐ The statement the
paper can make is therefore **stronger and narrower** than *"the CLU fails here"*: **the
environment's own tripwire fires before the arm is reached.** ⛔ Neither column may be quoted as a
baseline the CLU lost a fair comparison to, and ⛔ neither may be dropped to make the row read
better. *Per-seed triples printed on every measured cell (SF12).*

⚠ **SF12 — print the per-seed triples in the main table, as `TABLES.md` does.** Row 3's conjunction
mean **conceals a 0.0000 on one of three seeds**; row 2's CLU spans 0.0597–0.1651, a 2.8× spread. The
charter rule is *"all three seeds or none — no row reports a best-of"*; a main-text mean that hides a
zero inverts its purpose.

- ⛔ **BAR 1 fails on all three composition rows:** −0.0560 (row 2) / −0.0209 (row 3 conjunction) /
  −0.0434 (row 4).
- ⛔ **BAR 3 fails, and that is the mechanism of the table.** `landscape_shuffled` **0.9509** and
  `random_atoms` **0.9601** against 1.0000 on row 1; on row 2 `random_atoms` **0.1414** sits **inside**
  the CLU's own seed spread. Only `landscape_removed` collapses. ⇒ **Under the hub's own
  pre-registration NO RESULT ROW IS CLAIMED for the CLU arm here.** The paper prints the table and
  claims nothing from it.
- ⭐ Row 1's 1.0000 is real but **uninformative**: the matched-bytes lookup also reads 1.0000.
- ⭐ **The store span tracks the pipeline ceiling to within 0.0138 on every row and seed, exactly equal
  on 7 of 12 cells** ⇒ the atoms are not the binding constraint; **the address book is.**
  ⚠ **NTH1 — pre-empt the obvious question:** on seed 2 `unseen_pair` the store-span ceiling (0.7018)
  **exceeds** the pipeline ceiling (0.6881), and that cell *is* the 0.0138 maximum. One sentence.
- ⚠ **C3 — row 3 is printed TWICE.** The scored row is a **CONJUNCTION** (n = 183/192/191); its
  ceilings are **PER ROUTE** (n = 389/403/414). `0.0284 vs 0.5255` is **not** a fraction of ceiling.

⭐⭐ **MF14 is now CLOSED IN BOTH PARTS.** Part (i): BLIND-SEQUENTIAL and BLIND-FACTOR for rows 2
and 3 are **columns of the table above**, not a later sub-section — a reader meets the caveat and the
claim in one glance, which is the whole point of a blind column. Part (ii): rows 1 and 4 carry a
**DECLARED GAP, not a blank:**

> ⛔ **Rows 1 and 4 have no BLIND column on any seed.** `/env_columns` carries exactly four keys —
> `unseen_pair`, `nway`, `fragment`, `fragment_cross_maze` — verified directly on all three scored
> JSONs by w15/table-fixes. The charter rule is **unsatisfiable** for row 4 (a composition row) and
> **inapplicable** to row 1 (a seen row, not a composition row). Row 4 is therefore **DECLARED NOT
> MEASURED, never a blank and never a zero**; row 1 is marked n/a with its reason.

*Sources:* `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T1 §1c and §1c-ter (and the in-tree
`w12r_artefact_mustfix/TABLES.md` §1c, identical cells); primaries
`/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`, `.../scored2_seed1_test.json`,
`.../scored3_seed2_test.json` at
`/env_columns/{unseen_pair,nway}/blind/BLIND-SEQUENTIAL(key order from A)/validity` and
`.../BLIND-FACTOR    (A x B local only)/validity`; correction **C13**.

**Figure slots:** `fig1_cmaze2_main_scoreboard.png`, `fig2_nway_conjunction_vs_route.png`.

### 6.2 ⭐⭐ THE ISOLATED MECHANISM — plan space **1.0000** vs latent space **0.13**

⭐ **The referee called this "the best thing in the paper" and verified every cell. Do not touch it.**
Three analyses of already-reported scored legs; **no new arm, no new held-out number, no bar moved.**

1. **Every piece of every answer is already written.** PLAN COVERAGE of a held-out unseen-pair route
   by the union of the TRAINING routes of the *same* maze: **mean 1.000, median 1.000; 194 of 195
   pairs at ≥ 0.95** (seed 0).
   ⭐⭐ **MF11 / SF18 — and the source's own clinching exhibit, which v0 left on the floor:** a
   **same-maze** fragment table reaches **r = 2 at 0.5114** (unweighted seed mean; **0.5105** pooled —
   correction **C11**) against row 2's pipeline ceiling of **0.6319** — **81 % of the pipeline
   ceiling, where the CLU reaches 20 % and the best learned baseline 29 %.** ⛔ **It is NOT a
   like-for-like arm: FRAG reads the true maze at every step, which charter B.3 forbids the CLU.**
2. **The factorisation the CLU assumes is CORRECT.** Coverage by same-agent plans **0.838**, by
   same-target plans **0.885**, **UNION 0.997**, complete for **99.1 %** of pairs — precisely the pair
   blocks of the address book carry the whole answer.
3. ⭐⭐ **The composition is destroyed by WHERE it is performed.** Same two halves, two spaces, one
   exact judge:

| composition performed in… | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **PLAN space** (overlay the two half-plans, target's plan wins conflicts) | **1.0000** | **1.0000** | **1.0000** |
| **LATENT space** (average the two halves' latents, decode) | **0.1649** | **0.0950** | **0.1449** |
| the same-agent half alone | 0.2835 | 0.1900 | 0.2523 |
| the same-target half alone | 0.1701 | 0.1750 | 0.2523 |

Controls (both halves load-bearing; not a tie-break artefact): agent's-plan-wins **0.5722 / 0.4250 /
0.4252**; target's plan + a RANDOM training plan instead of the agent's half **0.3557 / 0.3700 /
0.4065**. In plan space the overlay is not merely valid, it is **OPTIMAL** (score 1.0000 = opt/len on
every pair).

⭐⭐ **THE FINDING, as the paper should state it:** *the read is linear in the request design, so
composing two addresses composes their **latents**; an exact-judge route task composes in the
**plan**. The information is present, the factorisation is right, and the operator is available and
perfect — what fails is that a point-in-latent-space read cannot express "overlay these two plans, the
second winning".*

⛔ **Two caveats, neither optional.** (a) The plan-space overlay is a **REFERENCE, not an arm** — it
keeps whole written items and is not byte-matched. ⛔ Reported to locate the failure, **never as a
win.** (b) That the overlay is perfect is itself a statement that these routes decompose additively —
the same property that leaves BLIND-SEQUENTIAL at 0.9037 here. In an environment whose routes need two
or more keys the overlay would not be perfect.

⭐ A second, subsuming reading the Head may keep or cut: rank decides whether a single plan can be
**rendered at all** (0.937 explained variance renders 98 % of training routes; 0.544 renders 2.6 %),
and even with rendering perfect, latent superposition still caps the row at ~0.15. ⭐ The sentence that
explains why the same content transferred to Shapes3D and not here: *on Shapes3D the read is graded by
a classifier over an image, which tolerates blur; here one wrong action is a zero, and a
variance-optimal compressor's discarded residual is exactly what decides legality.*

*Source:* `w9-maze/LOG.md:288-346` and `:297`, `08:05Z`; registry `N-I8-3`, matrix `CM-I8-3`.
**Figure slot:** ⛔ **does not exist** (F4).

### 6.3 ⛔ WHY the caveat columns read as they do — and MF11

⭐ **The numbers themselves now live in §6.1's main table (MF14(i)); this section carries only their
mechanism and the CMAZE-1 contrast.** BLIND-SEQUENTIAL on CMAZE-2 is **0.9077 / 0.8905 / 0.9128
(mean 0.9037)** on row 2 and **0.9229 / 0.8933 / 0.8841 (mean 0.9001)** on row 3 — ⛔ **not** the
published CMAZE-1 **0.4980**. ⭐ **And BLIND-FACTOR moves the same way and must be quoted with it:**
**0.4762** (row 2) / **0.4720** (row 3) on CMAZE-2 against **0.2350 (0.1850 / 0.2500 / 0.2700)** at
the CMAZE-1 harness tier (`/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T3) — ⛔ **both blind
references roughly double under the request layer, so the rise is a property of the LAYER and not of
one column's rule.**
⚠ **C1 — the band "0.9037–0.9130" is not a band in the artefact.** 0.9037 is the 3-seed **mean**;
0.9128 (not 0.9130) is seed 2; the per-seed **minimum is 0.8905**.
⭐ **The published tripwire reproduces intact at the CMAZE-1 harness tier: 0.5317 (0.5550 / 0.5300 /
0.5100).** ⇒ ⛔ a property of the **REQUEST LAYER**, not of CMAZE-1: the hub's *unseen pair on a SEEN
maze* layer puts **1,242 of 1,449** optimal routes on exactly one key, so there is no key ORDER for
rule B to trap. **Correction C4.**

⛔⛔ **MF11 — FRAGMENT-LOOKUP: two different quantities, both printed, never interchanged.**

| column | r = 1 | r = 2 |
|---|---|---|
| **CROSS-MAZE** (the published tripwire; the split does NOT leak) | **0.0006** (0.0017 / 0.0000 / 0.0000) | **0.0000** (0.0000 / 0.0000 / 0.0000) |
| **SAME-MAZE** (a DIFFERENT quantity, ⛔ not a like-for-like arm) | **0.0741** (0.0599 / 0.0960 / 0.0665) | **0.5114** (0.5342 / 0.5364 / 0.4636) |

⛔ *"FRAGMENT-LOOKUP r = 1/r = 2 at 0.0000 everywhere"* is **TRUE of the cross-maze quantity and FALSE
of the same-maze quantity.** *Source:* `TABLES.md` T1 §1c-bis; corrections **C10**, **C11**.
**Figure slot:** `fig5_blind_sequential_on_this_split.png`.

---

## §7 ⛔ WHAT THE STORE DOES NOT DO — it does not shape the representation

**TABLE 2** (Shapes3D generation cell, w8 `hub/encoder-loop`, `w8_v1_short` decoder,
`joint_steps = 400`, `n_train = 8192`, 3 seeds). ⛔ **Separate table by charter** — the shortened
decoder is a different *instrument*, not a different setting, and the metric is a **count**
(`distinct_valid_novel_tuples`), not a rate.

| bar | row 3 | row 4 |
|---|---|---|
| BAR A passed (mean paired Δ > +1.0 **and** CI95-low > +1.0, all three seeds) | **False** | **False** |
| BAR B falsifier fired (Δ ≤ 0 on two or more seeds) | ⛔ **True** | False |

⭐ **The decomposition is the finding, and the net is a wash because two real effects cancel:**
unfreezing the encoder with the store on the gradient path **HURTS** (`A′` negative on **5 of 6**
seed-rows, two at p ≤ 0.0013), and the swap loss **REPAIRS** that damage (`A″` positive on **4 of 6**;
⚠ **SF6 — THREE at p ≤ 0.0003**, not two: the four p's are 0.0001751 / 0.000198 / 0.000346 /
0.0001564). Net `A` = **−1.772** (row 3) and **+1.361** (row 4), CI95-low **−3.456** and **−1.592**.
⭐ **And the ablation that reads it correctly:** the CLU's margin over the matched-parameter learner
**does not grow** (+12.078 → +10.311 on row 3; +24.687 → +26.048 on row 4). ⚠ **SF7 — the learner is
"at or within 0.02 of 0.000"**: `rival_mlp_180k @ alone` on row 3 is **0.006** (0.000 / 0.017 /
0.000). ⇒ **This is a finding about the ENCODER, not about the STORE.**
⛔ **"The store shapes the representation" is NOT SUPPORTED and must not be printed.** Registry
`N-I8-4`, matrix `CM-I8-4`. ⛔ **NR4 travels here:** `hub/maze-encoder-loop` was contingent on a
positive w8 result and the contingency did not fire; zero budget drawn.
*Sources:* `TABLES.md` § TABLE 2; `w8-encoder-loop/CROSS-ARM.json` `/VERDICT`.
**Figure slot:** `fig3_encoder_loop_decomposition.png`.

---

## §8 THE EXTENDED PHYSICS SECTION — positioned as RUNG 3, scope limit first

> ⛔⛔ **THE POSITIONING RULING, and it is not the Head's to vary silently.** Position this material as
> **a MEASURED LAW about our own object**, in the Edge-of-Stability grammar, **under the claim ladder's
> rung 3 ("wells as a use")**. ⛔ **Never as a primitive, never as a fifth model class, never as a
> scoreboard row, never a contribution bullet co-equal with rungs 1 and 2.**
> ⭐ **And keep rung 1's sentence out of this section** — mixing *"downdate the statistic"* into rung
> 3's paragraph invites a reviewer to audit rung 1's evidence against rung 3's prose.
> *Source:* `w12-positioning-scout/POSITIONING.md` §§4, 5.0, 5.4.

**Three reasons the positioning is forced, each with a file:** (1) the charter ruled it out of the
scoreboard **before any number was read** — *"there is no lookup comparator for 'does a closed form
predict a rollout', and this hub claims no row against any rival"* (`BENCHMARK-STATEMENT.md` §2 N4);
(2) **the same dynamics measurably COST composition on the same object and the same seeds** (8.1);
(3) `CM-21` already retired the four positioning claims this section would otherwise drift into, and
its approved replacement wording claims **controllability, not capability**.

| slot | content | pointer |
|---|---|---|
| 8.1 | ⛔ **The scope limit, and the same object refuting the scoreboard reading on the same seeds:** `pca_settle` costs row 4 **−39.959 / −46.918 / −39.061** and row 3 −15.800 / −25.533 / −20.350 against `pca_tensor`, while paying **81,312 B more read-side**; `pca_settle_zerostep` is **bit-identical** to the incumbent (`np.array_equal` asserted) at Δ **0.000** on every seed. ⭐ **This is the matched ablation run backwards and it must be printed, not hidden.** | `EVIDENCE.md` §1; `POSITIONING.md` §5.0(2) |
| 8.2 | ⭐⭐ **THE RESULT.** The CRR predicts the field store's settle half-life **parameter-free** to a factor **1.105** at the median held-out corner (B1 = 0.0433 / 0.0435 / 0.0431, bar 0.10, seed spread 0.0004) and beats a one-parameter rival **fitted with sight of the held-out measurements** on **49 of 49 corners on every one of three seeds** (B2 mean paired Δ **−0.5968 / −0.5796 / −0.5982**). ⚠ **SF8 — 192,975 rows is the TOTAL across 56 corners**, of which **124,256** (41,493 / 41,405 / 41,358) sit in the 49 **scored** corners. ⚠ **SF10 — print the rival's own level beside the delta:** FIT1 median error **0.640132 / 0.623155 / 0.641281** against the CRR's **0.0433**, 15× worse, so the reader can judge whether it is a strawman rather than being asked to assume it is not. | `EVIDENCE.md` §2.3 |
| 8.3 | ⭐ The instrument floor, measured **before** the bar was fixed: median **0.0349** on an exact quadratic ⇒ B1's 0.10 bar is ≈ 3× the instrument's own floor. ⚠ **SF9 — print all three:** median 0.0349 / **p90 0.0749** / **max 0.1601** over 17 cells — ⛔ **the max exceeds B1's own 0.10 bar**, at `h/h* = 1.42`, the exceptional point. The bar is safe on the **median** reading, which is what B1 scores. | `EVIDENCE.md` §2.2 |
| 8.4 | ⭐ **Where the law stops, resolved by `h/h*`:** median error **0.0008 (0.2 %)** at `h/h* < 0.1`. ⚠ **SF11 — NOT "all the error in one bucket".** The five buckets are 0.0008 / 0.0264 / **0.1482** / 0.0454 / 0.0262: *the error is dominated by the single bucket straddling the exceptional point (0.1482 against 0.0008–0.0454 elsewhere)*, where D10 says the 2×2 map is **defective**. **The theory predicts where its own simplest form stops.** | `EVIDENCE.md` §2.5 |
| 8.5 | ⛔ **B4 is a MISS on all three seeds** (0.3362 / 0.3301 / 0.3520 against a factor-2 bar; 0.3782 / 0.3784 / 0.3834 against the exact inversion) — **carried, not footnoted**, and pre-registered with its cause in a dated amendment before the run. The predicted curve's own argmin over `γ*_exact` is **1.0000** on every seed, so the grid resolves the critical point and the shortfall is in the rollout, not in D16. ⛔⛔ **AND B3's PRINTED MAX MUST BE CARRIED TOO** — the referee promoted this out of should-fix: B3 passes on the median reading (0.0357 / 0.0282 / 0.0343) and its **max over γ is printed and not moved: 59.1158 / 0.1612 / 1883.2934** against a ≤ 0.10 bar. **A section whose stated discipline is "carry the miss" cannot carry one and drop a larger one.** | `EVIDENCE.md` §2.3, §2.6 |
| 8.6 | ⭐⭐ The instrument was **named in advance** and was worth an order of magnitude: envelope vs first-crossing differ by **9.24×** deep underdamped; first-crossing's error is 1.0088–1.0180 against the envelope's 0.0284–0.0296. ⛔ Scoring first-crossing would have reported the law as broken when the instrument was wrong. | `EVIDENCE.md` §2.7 |
| 8.7 | ⭐ A second, independent reproduction of the same law (`w6 hub/retention`, different hub, target and code path): floor **27.026815** vs the papers' 27.03; simulated envelope vs closed form **4e-13 – 8e-11 over ~7 orders of magnitude** of `n_half`; stability limit `1 + tr + det = 0` to **3.3e-16**; latch `ρ = 1.0` exact. | `EVIDENCE.md` §2.11 |
| 8.8 | ⭐ Bit-exact replication of seed 0 from a clean checkout cloned from the bundle; **runs 2/3 and 3/3 RETURNED UNSPENT**; 1 h 55 m local CPU, no GPU, no cloud, no cluster. | `EVIDENCE.md` §2.10 |
| 8.9 | The mechanism connecting 8.1 and 8.2 — ⛔ **not scored, not a row.** The settle does **not** leave the principal subspace (off-span ≤ **2e−7**; orthogonal fraction predicts the loss at chance, AUC 0.457–0.559). What breaks the read is motion **inside** it: the answers form a dense lattice with one-factor spacing **ρ₁ = 1.71/1.89/2.00** against a state norm of 31.3/32.3/32.8 (**5.5 %**), the judge's decision cells **are** that lattice's cells, and the settle places **0.78–0.86** of its step in the 51 answer-changing directions against the isotropic 0.407. Displacement in lattice units predicts the loss at **AUC 0.904**. | `EVIDENCE.md` §3.10 |
| 8.10 | ⭐ Deletion in the physics block, and it splits — D32a **CONFIRMED [CHLU]** (§3.9); D32b **REFUTED, failing step named** (§11.5). Two consequences derived **before** measuring and then exact: `n_refuse = 19.9316 × n_half` for every curvature, measured **539 = 19.9432** half-lives on all three seeds; and **`μ²` cancels** — two clocks a decade apart give **16/16 horizons byte-identical, worst lifetime gap 0.000e+00** ⇒ ⛔ **no curvature can trade refusal against composition.** | `EVIDENCE.md` §3.5 |
| 8.11 | ⛔ **Status discipline is part of the section, not an appendix.** Every physical claim carries CONFIRMED [CHLU] / CONFIRMED [shorts] / REFUTED / UNTESTED / DEFINITION, and **[shorts] means provenance is a paper, not an artefact in this tree** — the programme never re-ran those measurements. | `EVIDENCE.md` §0 |
| 8.12 | ⭐ The controls, stated because §C.2 is satisfied in the **spectrum**, not in the error: the CRR error is **identical across arms** (0.0428–0.0435 on `field`/`removed`/`shuffled`/`pca_tensor`) *because the law is about the dynamics — stated in advance*; the arms separate in the curvature spectrum (`field` [0.2727, 1.0000] vs `removed` [1.0000, 1.0000], no non-trivial curvature at all). ⛔ **§C.1 DOES NOT APPLY and was not given a manufactured pass.** ⚠ **And say what that costs:** **no arm of ours fails the law, so the law distinguishes nothing** — which is exactly why it is rung 3 and not a claim. ⭐ The one experiment that would upgrade it is named and **NOT RUN, NOT SCOPED, NOT COSTED**: a comparator predicted **in advance** to be mispredicted by the closed form, in the `h/h*` ∈ [0.5, 2] bucket. | `EVIDENCE.md` §2.8; `POSITIONING.md` §§4, 5.5 |

⛔ **Quote the exact-root form of the half-life, never the asymptotic one** — the asymptotic
`2γ ln2/[(2−γ)(εμ)²]` is **2.4 % off already at `h = 0.30 h*`** and was **DECLINED as the operator** by
the hub that used it.
⛔ **Three reporting defects the Head must not walk into:** the **+1.1385** constant (**C6-phys**), the
D18 "thin margin" (**C7-phys**), and the tautological `d14_confinement` field (**C8-phys**) — §G.

---

## §9 WHAT A BENCHMARK WOULD HAVE TO CONTAIN — a requirement list, not a plea

⭐ Drafted from the w5/w6 **negatives**; ⛔ **not** an argument that this programme deserves a
benchmark. Ten requirements, each followed by the measurement that forces it. Full standalone text to
lift: `BENCHMARK-STATEMENT.md`; summary `EVIDENCE.md` §5.3. The five load-bearing ones for main text
(the other five to App. J):

1. **At least one rule must be able to render a candidate ILLEGAL, not merely more expensive.**
   *Forced by:* GYM-2's composition-blind oracle **O3 = 1.0000** (0 faults) while obstacle-blind scores
   0.0387 and speed-blind 0.4213 — a speed limit is an upper bound, so ⛔ *"no metric and no number of
   dimensions makes it composable."* ⚠ **NTH3 — carry the source's own qualifier:** *"Under an
   optimality-gap metric O3 only falls to **0.9367**, still above the 0.90 pass line."*
2. **Score with an optimality gap, not feasibility-or-goal-reaching.** The gap drops TW-B from
   **0.9825 to 0.3900**, below the 0.60 line (stitched plans are valid but sloppy: 1.25 extra steps,
   only 39 % exactly optimal). ⚠ Strictly better **and** it does not rescue requirement 1.
3. **Run a matched-bytes STITCHING tripwire with a `min_frag` sweep before adoption.** ⛔ GYM-1's **T1
   exact-key lookup alone scored 0.0000 and would have certified a dead gym as sound**; in GYM-2
   whole-trajectory retrieval died (T1 0.0000, T3 0.0003) while coarse recombination still fired at
   **0.9825**. ⚠ **NTH4 — sharpen with GYM-1's matched-bytes stitching 0.8520 and GYM-2's T3 0.0003
   against GYM-1's 0.7027.** **Exact-key lookup is not a tripwire.**
4. **It must contain a deletion request, scored on the SAME object and the SAME metric as
   composition.** ⚠ **NTH2 — cite both §5 of this paper AND R2** (the w5 ≈18.6-tuples/corner price and
   the coupling hole), which is what `EVIDENCE.md` §5.3 item 8 actually cites. A benchmark that scores
   only composition cannot see the programme's central open problem, which is **measured, not argued**.
5. ⛔ **It must not be able to score a dynamics prediction as a taxonomy row.** *Forced by:* §8.1.

⛔ **What this list is NOT, and the sentence must appear:** it is **not** a claim that the field lacks
such a benchmark. **No scout in this evidence base surveyed the field for one at this wave level**;
the Campaign-3 benchmark scouts are declared **NOT USED**. The Head's ruling stands: *"benchmark for
the physics CLU declared future work."*

---

## §10 RELATED WORK — positioned so a referee cannot spring it

⛔ **The two claim-STOPs are enforced in this section by construction, by citing the owners first.**

| block | what to say | pointer |
|---|---|---|
| 10.1 the multilinear half | Tenenbaum & Freeman (2000); **TensorFaces** (Vasilescu & Terzopoulos, ECCV 2002) — which itself states it subsumes PCA as the 1-factor case and bilinear style–content as the 2-factor case; Vlasic et al. (2005); **CausalX / M-mode Block SVD** (the one genuine block-multilinear ancestor, and the highest atom count in the whole crawl at **4 of 6**); **MLC-VAE** (ICML 2020), the closest modern statement of the composition claim. ⇒ they own **A3+A4**. ⛔ What they cannot shed: a **complete labelled Cartesian-product tensor**, bases **defined by the label modes**, and a **linear** renderer. | `NOVELTY.md` §§1–2 |
| 10.2 the frozen-generator half | GANSpace (NeurIPS 2020), TCA (BMVC 2021), **PandA** (ICLR 2023) and its **TPAMI 2024** version — still *"entirely unsupervised"*, **no labels and no address book**, deletion-atom grep **0 across the entire paper**. ⇒ they own **A2+A5**. | `NOVELTY.md` §2, §C.4 |
| 10.3 exact unlearning | Cao & Yang (S&P 2015) — the summation form, **which owns the primitive**; Ginart et al. (NeurIPS 2019) k-means; Xu et al. (2023) bilinear matrix completion; the ridge-head-on-frozen-backbone result (arXiv:2603.12977) that **explicitly excludes "PCA, eigendecomposition, or generative models"**. | `NOVELTY.md` §3 |
| 10.4 ⭐ the one sentence that is ours | **Hall–Marshall–Martin and the decremental-PCA line are explicitly approximate — they subtract *deflated* eigenspaces, so truncation error compounds under subtraction.** Our route is exact for the opposite reason: it downdates the **full, untruncated second-moment statistic** and re-eigendecomposes afterwards. **"Downdate the statistic, not the truncated factors."** | `NOVELTY.md` §D; `LOG.md` 16:53Z |
| 10.5 pre-empt, do not be shown | **Feature Unlearning for Pre-trained GANs and VAEs** (AAAI 2024) — the nearest thing in existence, and it **unlearns a feature not rows**, is approximate, and **fine-tunes the generator** (so the decoder is not frozen); **Data Redaction from Pre-trained GANs** (SaTML 2023), useful because it states outright *"redaction is a fundamentally different task from data deletion"*; GANs Unlearning (TDSC 2025), approximate by construction; **Incremental N-Mode SVD** (TIP 2014) — the only work in ≈4,050 bridging eigenspace-update numerics and multilinear generative models, and it is **incremental only, no removal operator**. | `NOVELTY.md` §C |
| 10.6 the theory ancestor for §6.2 | *Compositional Generalization via Forced Rendering of Disentangled Latents* (arXiv:2501.18797) + Compositional Risk Minimization — the modern statement that **latent factorisation can fail to survive the decoder**. Direct prior motivation for the plan-vs-latent finding and for keeping the read linear up to a frozen renderer. | `NOVELTY.md` §2 |

⚠ **Residual novelty risk, in the paper's own limitations and not hidden:** OpenAlex coverage of
sub-3-month preprints and non-indexed workshop papers is imperfect, and abstract-level screening
cannot catch a match buried in a method section. The scout's own judgement: **low, not zero.**
⚠ **SF16 — F9:** `NOVELTY.md` §E names the one outstanding pre-submission action and v0 did not:
*"It is no longer the cheapest outstanding insurance — that is now **a careful read of the two 2024
works in §C.1 and §C.4 in full**."*

---

## §11 LIMITATIONS, NEGATIVES AND THE DECLARED NOT-RUNS

⛔ **A list of measured facts, not a hedge paragraph.** Every entry is already placed above; this is
where a referee can count them.

1. **No RESULT row for the CLU on CMAZE-2** — BAR 3 fails; the hub's own pre-registration forbids it.
2. **"The store shapes the representation" is falsified** — BAR B fired on row 3.
3. **No object with a float64 sufficient statistic composes and forgets inside 524,288 B** — on nine
   swept K the honest four-part total never falls below **3.9942×**, and on the one K ≥ 36
   configuration whose composition was measured it collapses to **0.0648**.
4. **Post-deletion composition on TEST fails as pre-registered:** surviving targets **0.1024 → 0.0000**
   (0.1569 / 0.0702 / 0.0802 → 0.0000). The cause is diagnosed — the w8 carry is stated for the
   orthonormal basis `U_k` while this read is expressed in the deployed modes `G_k` — and the repaired
   mechanism gives **−0.0120 on VAL seed 0**, inside BAR 6's ±0.05. ⛔ **One VAL measurement is not
   three TEST measurements: NR5, not a row.**
5. **D32b REFUTED** — the post-delete store is **not** byte-identical to the never-inserted store:
   **1,339,083 / 1,338,974 / 1,343,641 B differ = 65.17 / 65.16 / 65.39 %** of the image. ⭐ The
   break-out is worth more than the bar: `coupling_v` **798,926** · `term_mu` **267,684** · `atoms`
   **221,590** · `coupling_u` 33,595 · fitted widths/gaps/counts 16,769 · live-set 519 ⇒ **placement is
   only 16.5 % of the obstruction; 83.5 % is that the address book and coupling table are FITTED ON THE
   POOL.** ⚠ This falsifies the **paper's precondition on this store**, not the paper's law. ⚠
   **NTH6 — name the failing precondition: canonical Blelloch–Golovin placement.**
6. **B4 misses on all three seeds; and B3's max over γ is 59.1158 / 0.1612 / 1883.2934** (§8.5).
7. **A lifetime is strictly worse than instant eviction** — composition cost differs from instant
   eviction's by **+0.0562 / +0.0337 / +0.0112** against a declared margin of 1.0, and is paid strictly
   *before* any refusal arrives (`t_pay(50 %)` = 7.992 / 5.994 / 7.992 half-lives vs `t_refuse` =
   19.943). ⭐ **96.30–99.83 %** of the cost is paid by a **continuous** weight while the **discrete**
   liveness bit has not yet flipped.
8. **Symbolic refusal: no claim on rows 5 and 6.** An **8-byte rule** ties the field store exactly at
   (1.0000, 1.0000) on both rows, and `table_absence` **is** `bitmap8` at a byte multiple of
   **59,392×**. ⛔ Pre-registered as the expected outcome and reported first. ⭐ The surviving claim is
   refuse-after-conjunction-deletion, **expressed by a flag (`coupling_hole`), not by geometry**. ⚠
   **NTH8 — carry the premise correction:** *"96 unseen conjunctions" is a **val** figure — **the
   scored split has 3***.
9. **Row 2 (unseen-valid) belongs to a generic learner** — a parameter-matched continuous-input MLP
   beats the arm by **−0.0192 ± 0.0028** (hub-built, 573,440 params = 100.1 % of the arm) and
   **−0.0170 ± 0.0024** (built independently by `spoke-a`), **3/3 seeds with CI95 < 0** on both.
10. **Placement is exhausted** — row 4 vs the label-MI partition **+1.388 / −12.755 / +2.163**, mean
    **−3.068**, seed spread **14.918**; bar not met, runs 2 and 3 returned unspent. ⭐ **What is closed
    is the DIAL, not the criterion.**
11. **Two purpose-built gyms failed their own adoption gates** on two different grounds (§9).
12. ⛔⛔ **A DECLARED ABSENCE:** *"Mexican-hat width / separation sets the interpolation floor and the
    refusal gap"* **does not exist in the programme's own papers** — "interpolation" **0** occurrences,
    "Mexican hat" **1** (a float64 round-off testbed with no width and no separation parameter),
    "centre-surround" / "DoG" / "RBF" / "lateral inhibition" **0**. ⭐ Established independently twice.
    ⭐⭐ And the nearest genuinely derived object **INVERTS** it — the retention curve's floor is
    curvature-**IN**dependent.
13. **A1–A8 are asserted without derivation and are UNTESTED as laws**, including the **Lifetime
    Trilemma** (A1), whose *consequence* is measured but whose statement has **no proof and no
    impossibility argument in either paper**. ⭐ V2 itself says of A6: *"This is a direction on which
    this paper offers no evidence."*
14. **The V1 certificate family (D34–D45) is UNTESTED on the CHLU field store** — [shorts] throughout,
    including V1's own measured limit that `det J = 1` plus zero energy change is **not sufficient** for
    BIBO stability (at exit `b = 5.0`: ΔH = 0.0 exactly, the sub-level test admits, and the trajectory
    escapes — max excursion 0.09 certified vs **104.83** blind).
15. **Ten DECLARED NOT-RUNs, each with a launch line and a cost** — NR1…NR10, verbatim in App. K.
    ⛔ Never a blank, never a zero.
16. **Residual novelty risk** (§10 final ¶).
17. ⛔⛔ **MF15 — REFUSAL IS NOVELTY DETECTION, NOT WALL DETECTION.** The store refuses the
    **reachable-but-unwritten** pair (**0.9781** = 0.9721 / 0.9794 / 0.9829) **as often as the walled
    one** (**0.9801** = 0.9606 / 0.9846 / 0.9952), at a false-refusal rate on grounded-possible requests
    of **0.8182 against a pre-registered 0.20 — a 4× miss.** The supervised refuser, which HAS seen
    impossible examples, **separates better. No refusal win is claimed.** *Source:* registry
    **`N-I8-19`** (tier A); `TABLES.md` T1 §1d. ⛔ v0 would have printed those three DISPLAY rows
    **without the finding that reads them**, and v0's never-quote #14 restates the line `N-I8-19` says
    is not a win.
18. ⭐ **NEW — the value identity does not live in the store at K\*.** Deleting all 11,044 red rows
    leaves the red-requesting corners at **32.25 → 32.25** and judge accuracy at **0.8958 → 0.8958**
    (§3.6). ⛔ The exactness claim is store-scoped in a way that is **measurable, not merely
    definitional**, and the paper states it. *Source:* registry **`N335`** / **`CM-W12-10`**.
19. ⭐ **NEW — the composition verdict is K-conditional**, with its sign stated as a property of
    **K = 5, not of the store** (§4.2). ⛔⛔ **N8 — CORRECTED HERE, AND THE CORRECTION IS THE POINT.**
    v1 of this draft said the K = 5 reading is *"withdrawn from main text"* and carried in the
    appendix. It is **not**, and no ruling says it is: that sentence was lifted from
    `/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §13 item 1, whose own heading is **"WHAT THE HUB
    IS ASKED TO RULE ON"** — ⛔ a recommendation, converted into a statement, contradicting this
    draft's own §C. ⭐ **The governing reading:** the Head's one-K ruling governs **which K the
    composition CLAIM is stated at**, not **which K's NUMBERS are printed**. The **K = 5 margins are
    printed in MAIN TEXT beside K\*'s** — §4.2's ladder table and §C **Table 5 (main)** — and T8's
    *"Placement: APPENDIX"* applies to **the full 13-arm table**, not to the verdict.
    ⛔ **Appendixing the programme's largest negative would be overclaiming by omission; this draft
    does not authorise it.** *Sources:* `/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §13 item 1
    (read as the open question it is); `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T8 placement
    header; §4.2 and §C of this file.
20. ⭐ **NEW — no row in this programme can exceed 3 seeds.** Seeds 3 and 4 have verified generation-2
    caches and **no judge** (§5.4). A ceiling on everything the paper can ever say.

---

## §A ⭐⭐ THE SIXTEEN MUST-FIXES — DISPOSITION, ITEM BY ITEM

⭐ **All sixteen CLOSED — fourteen by w13/draft-revise, and MF2 and MF14 by w15/table-fixes after
`w13/referee-2` returned both as ⚠ PARTIAL (REFEREE-v1 §2).** ⛔ **Zero scored runs, zero fitted
models, zero new measurements** in either pass. The *how* column names the change and the artefact
that licenses it; the two rows re-closed in w15 say so, and say what was still open.
⚠ **REFEREE-v1's SF-a is discharged with them:** its *"fourteen CLOSED, two PARTIAL"* was correct
against v1 as written, and is superseded by the edits recorded in
`/c/chlu/outputs/w15-table-fixes/TABLE-FIXES.md`.

| MF | what it was | status | how it is closed |
|---|---|---|---|
| **MF1** | §1.2 pointed the Head at a **STOP-#1-violating sentence** (*"the survivable form is verbatim there"*) | ⭐ **CLOSED** | The pointer is **deleted** (§1 slot 1.2); §1.2 repoints at **§I of this file**, written to avoid the phrase by construction; the STOP warning is printed at the pointer. `NOVELTY.md` §3 final ¶ is now cited only as *the thing not to lift*. |
| **MF2** | *"whose whole content is a sufficient statistic"* is false of **both** implementations | ⭐ **CLOSED** | The contribution noun phrase is rewritten to *"content **derived from** a retained sufficient statistic, so downdating the statistic **induces** the refit of the content"*, with the statistic's byte cost **in the same sentence** (§I). ⚠ **RE-CLOSED by w15/table-fixes: as written in v1 the lead sentence printed `24.55×`, which is the FOUR-PART HONEST TOTAL and not the statistic** — the second half of MF2, caught by REFEREE-v1 §2 and N7. §I's lead sentence now prints the statistic's own cost **with its convention named in the same clause**: **23.5121× the 524,288 B store pin = 12,327,096 B ÷ 524,288 B, statistic ÷ pin**, and carries the four-part honest total to **24.5466×** separately. Evidence: statistic/content = **2.19 %** (w9, `/ledger/statistic_bytes` 35,904 of 1,642,752 deployed) and **2,261–2,348 %** (w8, `/cov_bytes_per_seed` 12,310,672 / 11,941,360 / 11,850,912 against a 524,288 B store). ⛔ The statistic's **full** ledger line (covariance + running means + per-well counts) is 12,327,096 / 11,957,784 / 11,867,336 B = **23.5121× / 22.8077× / 22.6351×** the pin — `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T4 byte ledger, part 2 TOTAL. The w9 ledger's own note says it: the statistic is *"declared beside them, never netted in"*. |
| **MF3** | the lead table printed **no byte row**; the object is **24.5×** the pin | ⭐ **CLOSED** | Four-part ledger printed on every store table (§3.3), never netted: **24.5466× / 23.8422× / 23.6697×** four-part ÷ pin (w8, of which the statistic alone is **23.5121× / 22.8077× / 22.6351×**), **2.19 %** statistic ÷ deployed read (w9), **4.0897× / 4.0768× / 4.0847×** four-part ÷ pin on the three TEST seeds of K\* (⚠ T7 itself is **one seed**; T12's **4.0877×** is the **val** pin leg), plus the **16,384 B declared residual**. Artefact level closed by w12r/w13 — `TABLES.md` T1 §1c-quater, T4, T7, T9, T9b, T12; correction **C5**. Plus the §3.3 sentence that turns the paper's worst omission into its framing. |
| **MF4** | *"exact"* was never defined, and only a store-scoped definition is supported | ⭐ **CLOSED** | §3.0 prints the **STORE-SCOPED** definition verbatim from `TABLES.json` `/exactness_definition`, before the table; the frozen encoder and decoder are **declared outside the deletable object** (Head ruling 2026-09-21, §2.5); *"of a generative model"* → *"of a generative model's **STORE**"* everywhere; the exact-JUDGE and exact-DELETION senses are separated. Correction **C6**. |
| **MF5** | exactness printed with no cost, against a ruling the programme wrote itself | ⭐ **CLOSED** | §3.5, at the end of §3 and not four sections away: 0.1024 → 0.0000; eviction **−2.0899 / −0.9101 / −6.7416** against the hole's **0.0000**. `w5-advisor-report.md` §6.3. (**SF17** closed with it — the cross-reference is in place.) |
| **MF6** | two caveats inside the cited gate file, never quoted | ⭐ **CLOSED** | §3.2, in T4's caption and not a footnote: `cut_is_marginal` **True on all three seeds**; `a1_subspace.passed_every_seed` **False**, seed 2, **60.36 ulp below its own 66.11 ulp floor** ⇒ a property of the **content**, not the downdate; `all_three_assertions_passed` **excludes** it. Correction **C14**. |
| **MF7** | the w8 leg labelled **RESULT**; its artefact says *"No scored run and no claim"* | ⭐ **CLOSED** | Relabelled **ASSERTION BATTERY (object property) · 3 seeds · TEST · 0 SCORED RUNS** (§3.1), with the sentence that the hub commissioned to score it returned its full budget (**NR3**). ⛔ The artefact was not relabelled to match the table; the table was relabelled to match the artefact. Correction **C7**. |
| **MF8** | T4 printed the narrower of two per-seed fields | ⭐ **CLOSED** | §3.1 prints the source spoke's own row — **4.5475e-13 / 1.1255e-09 / 2.8422e-14** — with the header naming **both deletions** (pinned 11,024 rows + random 64), the narrower field beside it and labelled, and the ulp figure kept. Correction **C8**; `fig4` redrawn from the corrected field. |
| **MF9** | only one of **two** unresolvable legs was declared | ⭐ **CLOSED** | §3.1 prints both against their own floors (seed 1: 1.126e-09 vs 4.282e-09; seed 2: 2.842e-14 vs 5.684e-14) and states it the **strong** way: on two of three legs the residual **is not resolvable at all**. |
| **MF10** | the scored maze rows are **CMAZE-2**; v0 called the whole section CMAZE-1 | ⭐ **CLOSED** | §6.0 defines CMAZE-2 in one line and relabels §6, §6.2 and §6.3; **T3 stays CMAZE-1 proper**; ⛔ **they never share a table**; never-quote #3's parenthesis is fixed to *"different environment **and** different budget"* (**SF19**). Corrections **C4**, **C9**. |
| **MF11** | *"FRAGMENT-LOOKUP 0.0000 everywhere"* is false on the hub's own split | ⭐ **CLOSED** | §6.3 prints both quantities, labelled: **cross-maze 0.0006 / 0.0000**; **same-maze 0.0741 / 0.5114** (pooled **0.5105**, correction **C11**) with the charter-B.3 not-like-for-like caveat. ⭐ And **0.5114 is promoted into §6.2(i)** as the strongest exhibit for *"every piece of every answer is already written"* (**SF18**). Correction **C10**. |
| **MF12** | the parameter match quoted from a **val** figure the source corrected away | ⭐ **CLOSED** | §2.3 prints the three scored TEST legs — **178,408 / 179,838 / 179,838 vs 178,952 = −0.304 % / +0.495 % / +0.495 %** — with the spoke's own sentence that **on two of three seeds the arm carries MORE parameters than the bar**, repointed to `w5-compose-field/LOG.md` **01:58 UTC**. `TABLES.md` T6. |
| **MF13** | *"at 524,288 B"* invites a byte comparison the arm does not win | ⭐ **CLOSED** | §2.3 and §4.1 state the supported form: store pinned at 524,288 B tol 0; book declared beside it; **total deployed 713,632 / 719,352 / 719,352 against 715,808 — matched to ±0.5 %.** ⭐ A **stronger** claim than v0's, and the one the artefacts prove. `TABLES.md` T6. |
| **MF14** | row 4 is a main-text composition row with **no BLIND column in existence**, and rows 2–3's BLIND columns sat outside the main table | ⭐ **CLOSED — both parts** | ⚠ **Part (ii) was closed in v1; part (i) was NOT, and is closed by w15/table-fixes.** In v1, §6.1's table had five columns and no BLIND column; BLIND-SEQUENTIAL lived in §6.3 and **BLIND-FACTOR appeared nowhere as a number** (REFEREE-v1 §2). §6.1's table now carries **BLIND-SEQUENTIAL** and **BLIND-FACTOR** as columns with per-seed triples, rows 1 and 4 marked n/a and **DECLARED NOT MEASURED** respectively. ⛔ **The move makes the maze negative read HARDER — both blind references outscore the CLU and the best learned baseline on rows 2 and 3 — and it is reported, not softened.** `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T1 §1c, §1c-ter; primaries `/c/chlu/outputs/w9-maze/runs/scored{1,2,3}_seed{0,1,2}_test.json` `/env_columns/*/blind/*/validity`; correction **C13**. |
| **MF15** | a tier-A negative with a **4× pre-registration miss** was missing entirely | ⭐ **CLOSED** | §11 item **17**, in main text, with **0.9781 / 0.9801 / 0.8182 against a pre-registered 0.20** and the supervised refuser separating better. Registry **`N-I8-19`**; `TABLES.md` T1 §1d reading block. |
| **MF16** | the paper's own open problem rested on **one of five** pin-fitting configurations | ⭐ **CLOSED, and SUPERSEDED in substance** | §5.1 prints the qualifier — *"K = 40, A = 7, span 280, 0.991×, **the one configuration at K ≥ 36 whose composition was measured**"* — footnotes the four unmeasured pin-fitting configurations with their spans (144/266/360/504/864) and the excluded **float32** point (K = 21, A = 94, 0.988×, span 336), and rewrites §11.3 as *"no object **with a float64 sufficient statistic**…"*. ⭐ **And the nine-K sweep replaces the argument with a measurement:** the honest four-part total never falls below **3.9942×** at rank 64 under any swept K. `w8-single-design/LOG.md` 18:32Z, `:99`, `:148`; `TABLES.md` T12. |

### The twenty should-fixes, the eight nice-to-haves and the one promoted item

⭐ **SF1 · SF2 · SF3 · SF5 · SF6 · SF7 · SF8 · SF9 · SF10 · SF11 · SF12 · SF14 · SF16 · SF17 · SF18 ·
SF19** are carried in the sections named above, each printed at its own number so the Head can find it.
⭐ **SF4** — the admission cut's invoice (directions dropped **3 / 1 / 4 / 5** on val0/test0/test1/test2,
worst variance share lost **8.94e-09**, and `rank/d` in well 3 falling **0.31 → 0.21** on test seed 2, so
the deployed span in that well **shrinks by a fifth**) → **App. D**. It is the only place the exactness
claim costs content. `w8-downdate/LOG.md:910-922`.
⭐ **SF13** — one line under the compose-XOR-forget table: *"a different metric (valid mean) on a
different split (val, 24 corners); not comparable with Table 3's tuple counts."*
⭐ **SF15** — **all six** registry-fold open items are carried in §G, not two. ⛔ And **§5.5, the
spot-check the registry asked for, is the one that produced MF3, MF6 and MF7** — it is now done.
⭐ **SF20** — flagged to the Hub, no change to the paper: the registry's §2 row prints CMAZE seed 2 as
**8.2e-16** where the artefact says **7.818e-16**. The draft uses the artefact and is right.
⭐ **NTH1–NTH8** are carried at §6.1, §9(4), §9(1), §9(3), §1 (novelty phrasing), §11.5, §8.5 and §11.8.
⭐ **The referee's promoted item — B3's max over γ (59.1158 / 0.1612 / 1883.2934)** — is carried at §8.5.

### ⭐ What the referee said to KEEP — and this spoke did not touch

The failure-agreement paragraph (§3.4, ⭐ **with the floor clause the referee asked for now added**) ·
§6.2 plan-vs-latent, every cell verified · the `oracle_content_items` floor choice (§4.1) · App. M and
`verify_tables.py` · §8.1 opening the physics section on the scope limit · C6/C7/C8 · §D's
delete-rather-than-flag discipline and the Campaign-2 skeleton exclusion.

---

## §B — THE CLAIM REGISTER

⭐ **Every claim carries a pointer. ⛔ A claim with no pointer was DELETED (§D).**

| # | claim as the paper may state it | class | n seeds / condition | pointer |
|---|---|---|---|---|
| **K1** | Deleting rows from the store is exactly the store refit without them, **store-scoped**: statistic max relative residual **7.5e-16…8.8e-16**, subspace projector **exactly 0.000e+00**, 0 refits, 0 passes, items never kept | **RESULT** | 3 / TEST | `TABLES.md` T4 (w9 leg); `w9-maze/runs/scored{1_seed0,2_seed1,3_seed2}_test.json` |
| **K2** | Independently reproduced by different code and a different author: worst **4.5475e-13 / 1.1255e-09 / 2.8422e-14**, **two of three legs below their own reproducibility floor**, negative control firing every seed | ⛔ **ASSERTION BATTERY (object property), 0 scored runs** | 3 / TEST | `TABLES.md` T4, T5 row 2; `DOWNDATE-READY-test.json` `/per_seed/*/downdate_equals_refit/worst_max_abs_dev` |
| **K3** | The two implementations agree on **where the mechanism breaks** — the same seed-2 failure at 32.90/35.46 ulp and 14.377 ulp — **and both observed deviations sit below each implementation's own floor**, so both conclude seed 2's content is not a function of its data | **RESULT (supporting)** | 3 / TEST | `TABLES.md` T5 failure-agreement block; `w8-downdate/LOG.md:618-640`; `w8-single-design/LOG.md:479,481,501,506` |
| **K4** | The object that forgets exactly is **24.5466× / 23.8422× / 23.6697×** the 524,288 B pin, four-part ledger, never netted, with a **16,384 B declared residual** | **measured property** | 3 / TEST | `TABLES.md` T4 byte ledger; `DOWNDATE-READY-test.json` `/times_over_budget` |
| **K5** | `pca_tensor` beats `rival_mlp_180k` on rows 3 and 4, all three seeds, CI95-low above the declared +1.0 margin, at **total deployed bytes matched to ±0.5 %** and parameters within **−0.304 %…+0.495 %** | **RESULT** | 3 / TEST | `TABLES.md` T6, T11; `w5-compose-field/LOG.md:196,247`; `READ-TABLE.json` |
| **K6** | It also beats every other rival that can enter those rows (`lstm`, `mlp`, `transformer`, `cont_mlp`, `table`) | **RESULT** | 3 / TEST | `w5-compose-field/LOG.md` §1 |
| **K7** | The win is a property of the **content**, not the budget: `oracle_span_field` − bar = −12.531 / −5.224 / −5.653 while `oracle_content_pool` − bar = +10.980 / +8.224 / +8.551 at the same 524,288 B | **RESULT (supporting)** | 3 / TEST | `w5-compose-field/LOG.md` §7(3) |
| **K8** | The composition-winning content forfeits exact per-item deletion by construction; the item store pays **−18.56 tuples/corner** (3-seed mean) on row 4, ⚠ a **lower bound** (a ceiling handed the true feature, not an arm) | **measured cost** | 3 / TEST | `TABLES.md` T11 |
| **K9** | ⛔ **At K = 5 the store does NOT clear a 1×-matched learner at any rung** under the pre-declared every-seed bar; ceiling-normalised **−0.077 / +0.059 / −0.073 / −0.054 / −0.056** at n = 2…6, and the +0.059 rung fails on **2 of 3 seeds** | **RESULT (negative), 0 scored** | 3 / held-out | `TABLES.md` T8; `LADDER.json` `/rungs/*/arms/single_design_K5/per_seed/*/paired_vs_best_learned`; registry `N326` |
| **K10** | ⭐⭐ **At K\* = 32 it clears the SAME bar at all five rungs on every seed** — all 15 margins positive, minimum **+3.117** | **REPLICATION, ⛔ 0 scored runs** | 3 / held-out | ⚠ `/c/chlu/outputs/w12b-nr6-and-kstar/KSTAR-LADDER.json` `/verdict/kstar_K32`, `/legs/*/kstar_K32/*/margin` |
| **K11** | ⭐ The flip is **the K step, not the construction** — a bridge arm at K = 5 with the K\* construction **still loses at 4 of 5 rungs**; K's share of every swing is **62–79 %** | **attribution, ⚠ ONE seed** | 1 / held-out | ⚠ `KSTAR-LADDER.json` `/margin_swing_decomposed` |
| **K12** | ⭐⭐ At **K = 5 (MI partition, full-square)** the single design at **23.886×–24.875×** and `pca_tensor` at **1.358×–1.372×** score **identically to four decimals at every rung and every seed** ⇒ the four-part ledger buys **deletion, not composition** ⚠ **at K = 5, and ⛔ NOT the K = 5 (ward, triangle) object of §5.1, whose four-part total is 7.8545×** | **RESULT (supporting)** | 3 / held-out | `TABLES.md` T8 byte ledger; `BYTES.json` `/summary`; registry `N328` |
| **K13** | The scored K\* row: **32.350 ± 0.815** (row 3) and **67.177 ± 1.320** (row 4), every 524,288 B comparator beaten on **109/109 corners**, smallest paired delta **+35.560**, four controls **exactly 0.0000** on all legs | **RESULT** | **3** / TEST | ⚠ `/c/chlu/outputs/w12b-nr6-and-kstar/NR6-T7.json` `/arms/cov_tensor`, `/controls`, `/verdict`; `TABLES.md` T7 (⚠ superseded seed column) |
| **K14** | The saturation trap does not fire: rank/width **0.9844** with all four controls at **0.000**; the sphere hazard was measured and is **negative** (−0.3291 / −0.3078) | **RESULT (control)** | 3 / val · 3 / TEST · 1 / TEST | `TABLES.md` T12 control gate, T9 sphere caveat; `NR6-T7.json` `/controls` |
| **K15** | ⛔ **Exact deletion at K\* does not stop the object answering the deleted value**: red-requesting corners **32.25 → 32.25**, judge accuracy **0.8958 → 0.8958**; the uncarried basis change is **generic** (0.610 vs 0.614) | **RESULT (negative, bounding)** | 1 / TEST | `TABLES.md` T9 probe; `deletion/probe_test_seed0_K32.json`; registry `N335` |
| **K16** | The targeting of a deletion is a function of rank/width: concentrated at K = 5 (`cos_min` **0.1999** vs **0.9976** random) and generic at K\* | **measured** | 1 / TEST + val | `w8-single-design/LOG.md` §4.2; `TABLES.md` T9 |
| **K17** | Refusal is a **separate mechanism from forgetting**: 2 holes, **2 bytes**, **0 data rows read**, no refit, layout asserted equal to `gen_field`'s over 1,322 pair slots; one hole is refusable **although no red cube was ever written** | **DISPLAY, never a RESULT** | 1 / TEST | `TABLES.md` T10 |
| **K18** | No object with a float64 statistic composes and forgets inside 524,288 B: minimum honest four-part total **3.9942× at K = 24** over nine swept K; at K = 40 / A = 7 (**0.991×**) composition is **0.0648** | **val falsification + exact arithmetic** | 3/1 val, 0 scored | `TABLES.md` T12; `w8-single-design/LOG.md` §2(2)(3) |
| **K19** | ⛔ The honest four-part byte curve is **U-SHAPED** (7.8545× → **3.9942× at K = 24** → 4.0877× — ⚠ **the VAL sweep leg throughout; K\*'s three TEST seeds are 4.0897× / 4.0768× / 4.0847×**); the monotone 7.4815× → 2.0663× curve **excludes the address book** | **exact arithmetic** | — | `TABLES.md` T12; correction **C16**; registry `N331` |
| **K20** | On CMAZE-2 a matched-parameter transformer beats the CLU on every composition row (BAR 1: −0.0560 / −0.0209 / −0.0434) | **RESULT (negative)** | 3 / TEST | `TABLES.md` T1 §1a |
| **K21** | The C.2 controls do **not** collapse on CMAZE-2 (`landscape_shuffled` 0.9509, `random_atoms` 0.9601 on row 1; 0.1414 inside the CLU's spread on row 2) ⇒ **no RESULT row is claimed** | **RESULT (negative)** | 3 / TEST | `TABLES.md` T1 §1a |
| **K22** | The mechanism is isolated: the same two halves compose to **1.0000 / 1.0000 / 1.0000** in plan space and **0.1649 / 0.0950 / 0.1449** in latent space | **analysis of scored legs** | 3 / TEST | `w9-maze/LOG.md:288-346`; `N-I8-3` |
| **K23** | The information is present (plan coverage 1.000) and the factorisation right (union 0.997, complete for 99.1 %); a **same-maze** fragment table reaches **0.5114** = 81 % of the pipeline ceiling, ⛔ not a like-for-like arm | **analysis** | 3 / TEST | `w9-maze/LOG.md:288-346,297`; `TABLES.md` §1c-bis |
| **K24** | The store span tracks the pipeline ceiling to within **0.0138**, exactly equal on **7 of 12** cells ⇒ the address book is the binding constraint | **RESULT (supporting)** | 3 / TEST | `TABLES.md` T1 §1b |
| **K25** | BLIND-SEQUENTIAL is a **request-layer** property: **0.9037** on CMAZE-2, **0.5317** at the CMAZE-1 harness tier against the published **0.4980**; FRAGMENT-LOOKUP **cross-maze** 0.0006 / 0.0000 | **RESULT (instrument)** | 3 / both tiers | `TABLES.md` T1 §1c, §1c-bis, T3; corrections **C4**, **C10** |
| **K26** | Unfreezing the encoder hurts (`A′` 5/6 negative, p ≤ 0.0013); the swap loss repairs it (`A″` 4/6 positive, **three** at p ≤ 0.0003); the net is a wash and **BAR B fired on row 3** | **RESULT (negative)** | 3 / test | `TABLES.md` T2; `CROSS-ARM.json` `/VERDICT` |
| **K27** | Refusal is **novelty detection, not wall detection**: reachable-but-unwritten **0.9781** ≈ walled **0.9801**, false-refusal **0.8182 against a pre-registered 0.20** | **DISPLAY + registry finding** | 3 / TEST | `TABLES.md` T1 §1d; `N-I8-19` |
| **K28** | The CRR predicts the settle half-life parameter-free to **1.105** at the median held-out corner and beats a sighted fitted rival **49/49 on every seed** — ⛔ **rung 3, a measured law, never a scoreboard row** | **RESULT, rung 3** | 3 / held-out | `EVIDENCE.md` §2.3; `POSITIONING.md` §5.0 |
| **K29** | ⛔ The same dynamics **cost** composition at every strength at which they move the state (row 4 −39.959 / −46.918 / −39.061); the zero-step arm is **bit-identical** to the incumbent | **RESULT (negative)** | 3 | `EVIDENCE.md` §1 |
| **K30** | Decay commutes with deletion exactly: **0 bytes differing in a 2,054,866 B image**, 12/12 cells per seed, 3 seeds, zero tolerance, on a **learned** store. ⛔ Its companion **D32b is REFUTED** (65.17–65.39 % of the image differs; **83.5 %** of the obstruction is the fitted book and coupling table, not placement) | **RESULT + RESULT (negative)** | 3 | `EVIDENCE.md` §3.5 |
| **K31** | Novelty position: **0 of 3,038** citing works combine frozen-decoder + tensor-head + label-index; highest atom count **4 of 6**; ≈4,050 works screened across three directions; **six of the sweep's 21 orthogonal queries** return zero | **scout result** | — | `NOVELTY.md` §§A–B; `N-I8-18` |
| **K32** | The eigenspace-arithmetic line is **explicitly approximate by its own authors' words**, so exactness is not conceded to it | **scout result** | — | `NOVELTY.md` §D; `LOG.md` 16:53Z |

---

## §C — TABLE AND FIGURE SLOTTING

| paper slot | table | pin summary | source |
|---|---|---|---|
| **Table 1 (main)** | TABLE 4 **+ its four-part byte ledger** — exact deletion, two implementations | 3 seeds each, TEST, units never merged, the w8 leg labelled ASSERTION BATTERY | `TABLES.md` T4 |
| **Table 2 (main)** | TABLE 9 **+ the probe** — deletion on the K\* object | ⚠ **1 seed**, TEST, K = 32; the probe is the bounding row | `TABLES.md` T9 |
| **Table 3 (main)** | TABLE 6 + TABLE 11 — the w5 composition RESULT at matched total bytes | 3 seeds, TEST, ±0.5 % on bytes and parameters | `TABLES.md` T6, T11 |
| **Table 4 (main)** | **T7 at three seeds** — the scored K\* rows | ⭐ **3 seeds**, TEST, 524,288 B reference column; 1× is **NR8** | ⚠ `w12b .../NR6-T7.json` |
| **Table 5 (main)** | the **K = 5 vs K\* ladder verdict with the bridge column** | K = 5: 3 seeds · K\*: 3 seeds · bridge: ⚠ 1 seed; all held-out, ⛔ 0 scored | ⚠ `w12b .../KSTAR-LADDER.json` |
| **Table 6 (main)** | TABLE 1 §1a — CMAZE-2 RESULT rows + the C.2 controls + **the 1c caveat columns for rows 2 and 3** + row 4's declared gap | 3/3 scored seeds, TEST, per-seed triples printed, row 3 printed twice | `TABLES.md` T1 |
| **Table 7 (main)** | the CRR bar table B1–B6, **with B3's max over γ** | 3 seeds, held-out | `EVIDENCE.md` §2.3 |
| App. A | TABLE 1 §§1b–1e in full | | `TABLES.md` |
| App. B | TABLE 2 in full | | `TABLES.md` |
| App. C | TABLE 3 (CMAZE-1 **proper**) in full | | `TABLES.md` |
| App. D | TABLE 5 + the admission cut and **its invoice** (SF4) | | `TABLES.md` T5; `w8-downdate/LOG.md:910-922` |
| App. E | the w5 ceiling ladder **with `address_bytes` per rung** (SF2), all controls, and the declared NOT-RUN rivals (`lace` 95.1×, `poe_ebm` 27.1× — *a dash, never a win for us*) | | `TABLES.md` T6; `w5-compose-field/LOG.md` §§1–7 |
| App. F | the byte-ledger discipline appendix; T12's **two curves** | | `TABLES.md` T12 |
| App. G | instrument audits (T1 §1e) | | `TABLES.md` |
| App. H–J, L, N | physics ledger D1–D50 + A1–A8 and the 34/34 reproduction battery; V1 certificates; the other five benchmark requirements; corrections; physics negatives | | `EVIDENCE.md`; `BENCHMARK-STATEMENT.md` |
| App. K | ⛔ the **ten** DECLARED NOT-RUNs verbatim, with launch lines and costs | | `TABLES.md` §DECLARED NOT-RUN |
| App. M | provenance: the **52-entry** source register; `verify_tables.py` re-resolves **52 sources / 872 cells / 1,786 per-seed values / 8 figure hashes**, **PASS** | ⚠ **and it must be re-run after OPEN-1** | `TABLES.md` §SOURCE REGISTER |

### Figures

| slot | file | status |
|---|---|---|
| **Figure 1 (headline)** | `fig4_exact_deletion.png` | ✅ EXISTS, hash-verified, corrected field (MF8) |
| Figure 2 | `fig1_cmaze2_main_scoreboard.png` | ✅ EXISTS (relabelled CMAZE-2) |
| Figure 3 | `fig2_nway_conjunction_vs_route.png` | ✅ EXISTS |
| Figure 4 | `fig3_encoder_loop_decomposition.png` | ✅ EXISTS |
| Figure 5 | `fig5_blind_sequential_on_this_split.png` | ✅ EXISTS |
| Figure 6 | `fig6_composition_ladder_n.png` | ⚠ **EXISTS but is K = 5 ONLY** — no K\* panel (**F10**) |
| Figure 7 | `fig7_k_curve_wells.png` | ✅ EXISTS — ⛔ its caption must name **which** curve |
| Figure 8 | `fig8_four_part_ledger_across_K.png` | ✅ EXISTS |
| Figure 9 — the w5 composition RESULT | — | ⛔ **DOES NOT EXIST** (F1) |
| Figure 10 — plan vs latent | — | ⛔ **DOES NOT EXIST** (F4) |
| Figure 11 — the CRR | — | ⛔ **DOES NOT EXIST** (F3) |
| Figure 12 — the K = 5 → K\* sign flip with the bridge | — | ⛔ **DOES NOT EXIST** (F10) |

⛔ **None may be drawn from a number this spoke invented; every one has its source JSON named.**

---

## §D — CLAIMS THIS SPOKE DELETED OR WITHDREW

| deleted / withdrawn | why |
|---|---|
| *"the survivable form is verbatim there"* (v0 §1.2 pointer) | ⛔ **MF1.** The sentence it points at contains **STOP #1** word for word |
| *"a compositional generative store whose **whole content is a sufficient statistic**"* | ⛔ **MF2.** False of both implementations: the content is the leading eigenvectors **of** a statistic kept separately (2.19 % / 2,261–2,348 % of the content) |
| *"at 524,288 B"* as the composition comparison | ⛔ **MF13.** Total deployed bytes are **713,632–719,352**; the honest form is *matched to ±0.5 %* |
| *"parameter-matched to −0.01 %"* | ⛔ **MF12.** A **val** figure; the scored legs are −0.304 % / +0.495 % / +0.495 % |
| *"the covariance costs Σ_k d_k²"* | ⛔ **SF14.** It is the symmetric triangle at float64, and K ≥ 36 is a float64 fact |
| *"the carry rewrites 127,488 B"* | ⛔ **SF5.** Superseded at `w8-downdate/LOG.md:972`; the figure is **125,952 B** |
| *"CMAZE-1"* for the scored maze rows | ⛔ **MF10.** CMAZE-2, a request layer over CMAZE-1 |
| *"FRAGMENT-LOOKUP r=1/r=2 at 0.0000 everywhere"* | ⛔ **MF11.** True cross-maze, false same-maze (0.0741 / 0.5114) |
| *"at K ≥ 36 composition collapses"* unconditionally | ⛔ **MF16.** One of five pin-fitting configurations, and dtype-conditional |
| *"the honest total falls 7.48× → 2.07×, monotonically"* | ⛔ **C16.** That column **excludes the address book**; the honest total is **U-shaped**, minimum 3.9942× at K = 24, and it is read from a `SMOKE_NOT_A_CLAIM: true` file |
| ⛔ **WITHDRAWN FROM MAIN TEXT: the K = 5 ladder verdict as *the* composition verdict** | ⛔ At K\*, the K the Head's one-K ruling puts in main text, the same bar flips to clearing at all five rungs. The K = 5 reading goes to the appendix **with its sign stated as a property of K = 5, not of the store** |
| ⛔ **WITHDRAWN: "T7 is a one-seed row"** | ⭐ w12b spent runs 2/3 and 3/3; T7 is **3 seeds** and the seed-0 headline sits inside the spread |
| ⛔ **WITHDRAWN: "the controls read 0.000 throughout"** | ⚠ True at K = 5; at K\*, **13 of 15** read exactly 0.0000 and **n = 3 seeds 0 and 1 read 0.0083.** Print the two |
| ⛔ **WITHDRAWN: "the ladder's opponent carried 1,029,508 params"** | ⛔ **C20.** Measured, it carried **3,131,564 – 3,260,030**. 1,029,508 is a different construction's figure, from the `SMOKE_NOT_A_CLAIM: true` file |
| *"exact deletion holds to 1e-16…1e-14, 9/9 seeds, two independent implementations"* | ⛔ Wrong three ways (**C12**) |
| *"the store shapes the representation"* | ⛔ BAR B fired on row 3 |
| *"the CLU composes on a route task"* / any RESULT row for the CLU on CMAZE-2 | ⛔ BAR 1 and BAR 3 both fail; the hub's pre-registration forbids the row |
| *"post-deletion composition holds"* | ⛔ The only held-out number is the failing one (0.1024 → 0.0000); the repair is VAL seed 0 (NR5) |
| *"no changes after deletion"* | ⛔ Head language rule. Post-deletion adaptation is **LIGHT and DATA-FREE** |
| the three novelty-STOP sentences | ⛔ TensorFaces 2002 / Tenenbaum & Freeman 2000 / MLC-VAE 2020 / Cao & Yang 2015 |
| *"Mexican-hat width / separation sets the interpolation floor and the refusal gap"* | ⛔ The law does not exist in the corpus, and the nearest derived object **inverts** it |
| *"wave 9 inherits the 0.4980 blind-sequential column"* | ⛔ **C4** — a request-layer property |
| *"the CRR result supports a scoreboard claim"* | ⛔ Refuted on the same object and the same seeds; Advisor-ruled |
| *"the Lifetime Trilemma"* as a result of the theory | ⛔ No proof and no impossibility argument in either paper |
| *"throat latch impact μ² ∝ A/l²"* (A2) and *"the minimum-separation certificate"* (A5) as derived laws | ⛔ Both refused by `hub/potentials`; `A` and `l` are never defined anywhere in the paper |
| *"the programme's claims need a benchmark that does not exist"* / *"a benchmark is needed"* | ⛔ No scout surveyed the field at this wave level; `EVIDENCE.md` §5.3 forbids it in terms |
| the Campaign-2 ICLR-long thesis | ⛔ Different object, different campaign; a C2-era planning document whose spine is PENDING |

---

## §E — THE NEVER-QUOTE LIST (one line each, for the Head's desk)

1. ⛔ Never `1e-14` for the deletion band. Quote **4.5475e-13 / 1.1255e-09 / 2.8422e-14** with the
   floors, or **0.009442 float32 ulp**.
2. ⛔ Never **"9/9 seeds"**. Three implementations, three rows; the nine legs are 3 configurations ×
   3 seeds inside **one** of them.
3. ⛔ Never merge T1 with T3 — **a different ENVIRONMENT and a different budget** (CMAZE-2 vs
   CMAZE-1). Never merge T2 with the incumbent's rows, the two deletion legs, or **T7 (K = 32) with
   T8 (K = 5)**.
4. ⛔ Never quote the n-way conjunction against the per-route ceiling as a fraction.
5. ⛔ Never print **0.4980** where **0.9037** belongs, or the reverse.
6. ⛔ Never quote the carry rewrite (**125,952 B**) beside the coupling hole's **1 byte**.
7. ⛔ Never quote a store's composition without its **four-part byte ledger** and its **K**.
8. ⛔ Never quote **"7.48× → 2.07×"** — that curve has no address book in it.
9. ⛔ Never present the plan-space overlay as a win — a reference, unmatched in bytes.
10. ⛔ Never quote `d14_confinement.max_abs_residual` (**tautological, exactly 0.0 on 60 cells**);
    quote `hessian_fd_check` (3.519e-11 / 1.202e-10 / 2.699e-11).
11. ⛔ Never quote the asymptotic half-life form; quote the exact root.
12. ⛔ Never write D18 up as a confirmation (§G, **C7-phys**).
13. ⛔ Never claim the three novelty-STOP sentences.
14. ⛔ Never state that the field lacks a composition-and-deletion benchmark.
15. ⛔ Never let a RESULT row and a DISPLAY row share a comparator.
16. ⛔ **NEW — never "no changes after deletion".** Post-deletion adaptation is **LIGHT and
    DATA-FREE**, against a reference that reads 120,028–131,072 rows.
17. ⛔ **NEW — never a composition sentence without its K.** The verdict flips between K = 5 and K\*.
18. ⛔ **NEW — never "the controls are 0.000 throughout"** without the two K\* legs at **0.0083**.
19. ⛔ **NEW — never call the K\* 1× win a scored result.** `REPLICATION_NOT_SCORED: true`.
20. ⛔ **NEW — never say the store has forgotten the deleted value.** At K\* the object still answers
    it identically (32.25 → 32.25). The store-scoped claim is the only one demonstrated.
21. ⛔ **NEW — never quote 1,029,508 as the ladder opponent's size** (C20), and never pad any row
    beyond **3 seeds** (C21).

---

## §F — WHAT THIS DRAFT NEEDS THAT DOES NOT EXIST YET

⛔ Each is a **declared gap with a named source**, never a blank.

| # | gap | what would close it | cost |
|---|---|---|---|
| F1 | **Figure 9** — the w5 composition RESULT has no figure | plot rows 3/4 from `w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json` | analyst-minutes, **0 scored runs** |
| F2 | the compose-XOR-forget figure needs the **nine-K** version | `fig7`/`fig8` exist; what is missing is one panel joining composition to the honest ledger with the pin as a line | analyst-minutes, 0 scored runs |
| F3 | **Figure 11** — the CRR has no figure in any artefact | build from `w6-potentials/test/s{0,1,2}/results.json` | analyst-minutes, 0 scored runs |
| F4 | **Figure 10** — plan vs latent has no figure | the table is already at `w9-maze/LOG.md:288-346` | analyst-minutes, 0 scored runs |
| F5 | **Table 3 of the paper still has no `TABLES.json` provenance row for the w5 RESULT** | extend the generator's register with `READ-TABLE.json` and re-run `verify_tables.py` | analyst-minutes, 0 scored runs |
| F6 | the per-cell **D17/D18** `(γ_window, slope_measured, slope_derived)` rows are in no machine-readable artefact | whoever owns the w6 branch emits them to JSON | ⛔ **CANNOT be run from the artefacts**; declared reporting gap |
| F7 | ⭐ **CLOSED** — the referee spoke's output was read in full by this spoke | — | — |
| F8 | **No abstract, no title.** ⛔ Deliberately not written — the Head writes prose | | — |
| **F9** | ⭐ **NEW (SF16)** — the one outstanding pre-submission action: *"a careful read of the two 2024 works in §C.1 and §C.4 in full"* | reading time | 0 scored runs |
| **F10** | ⛔⛔ **NEW — there is no figure and no folded table for the K = 5 → K\* sign flip**, the paper's most consequential wave-12/12b result | fold `KSTAR-LADDER.json` into the tables generator and draw the bridge decomposition | analyst-minutes, **0 scored runs** |
| **F11** | ⛔ **NEW — `pca_tensor` was never run at K\***, so §4.4's *"the ledger buys deletion, not composition"* is a **K = 5** statement only | one replication leg per rung at K\* with the `pca_tensor` arm; the harness exists (`w12b_ladder_kstar.py`) | ⚠ compute, **0 scored runs** (a replication) |
| **F12** | ⛔ **NEW — T9/T10 (deletion and refusal on the K\* object) are still ONE seed** | `single-design` is now **3/3 spent**; a further seed needs a new ledger row | ⚠ **scored budget — a Hub decision** |

---

## §G — CORRECTIONS THE HEAD MUST ABSORB BEFORE WRITING

⭐ **Nineteen, C1–C19, all in `TABLES.md` §CORRECTIONS**, plus two new ones this spoke carries from
w12b and the three physics ones. **None changes a verdict. All change a number the paper would
otherwise print.**

| # | as narrated | as the artefact says |
|---|---|---|
| **C1** | BLIND-SEQ band "0.9037–0.9130" | 0.9077 / 0.8905 / 0.9128, mean 0.9037; the **minimum is 0.8905** |
| **C2** | deletion "1e-16…1e-14" | worst 1.1255e-09 = **0.009442 float32 ulp** |
| **C3** | the n-way row against its ceiling | conjunction n = 183/192/191 vs per-route n = 389/403/414 |
| **C4** | "wave 9 inherits the 0.4980 column" | 0.5317 at the harness tier, 0.9037 on CMAZE-2 — a **request-layer** property |
| **C5** | no byte row on the lead table | `times_over_budget` **24.5466 / 23.8422 / 23.6697** |
| **C6** | `exact` undefined | **STORE-SCOPED**, frozen encoder/decoder never refit |
| **C7** | the w8 leg labelled RESULT | its own `/note`: *"No scored run and no claim"* |
| **C8** | T4's per-seed absolutes | the spoke's own field **4.547e-13 / 1.126e-09 / 2.842e-14**, and **two** unresolvable legs |
| **C9** | TABLE 1 labelled CMAZE-1 | **CMAZE-2** |
| **C10** | "FRAGMENT-LOOKUP 0.0000 everywhere" | true cross-maze, false same-maze |
| **C11** | same-maze r = 2 = 0.5105 | **0.5114** unweighted seed mean; **0.51044** pooled. ⛔ Do not use interchangeably |
| **C12** | "9/9 seeds, two implementations" | 3 configurations × 3 seeds inside ONE of **three** implementations |
| **C13** | "every composition row carries BLIND" | ⛔ **unsatisfiable for row 4** — declared, not blank |
| **C14** | T4 carried no caption caveats | `cut_is_marginal` 3/3; `a1_subspace` fails seed 2 at 60.36 vs a 66.11 floor |
| **C15** | the scored crossover row "at 4 seeds" | **ONE** seed on disk ⚠ **and now THREE, via w12b** |
| **C16** | "7.48× → 2.07×, monotonically" | that is parts 1+2+4; the honest total is **7.8545× → 4.0877×, U-shaped, minimum 3.9942× at K = 24** — ⚠ **all VAL** (N5); the K\* **TEST** totals are **4.0897× / 4.0768× / 4.0847×** |
| **C17** | the crossover FINDINGS §7 VAL cost line | the artefact reads 2.4543 / 0.1874 / 0.6205 / 0.2554 s and 10.843–23.486 s |
| **C18** | the crossover FINDINGS §5 val sweep, "1 seed at every K, K\* at 0.9016" | **3 seeds at K = 5 and K = 32**; K\* at **0.929012**, drop **−0.033758** |
| **C19** | `k` written for both rungs and wells | **n** = factors, **K** = wells |
| **C20 ⭐ NEW** | `w12-advisor-report.md:146` — *"the ladder's verdict stands against a **1,029,508-param** opponent"* | ⛔ measured, the ladder's `learned_1x` carried **3,131,564 – 3,260,030 params** on all 15 legs. 1,029,508 is the **crossover construction's** K = 5 figure, from the `SMOKE_NOT_A_CLAIM: true` file. The two conventions price the same statistic differently — full-square **12,284,920 B** vs symmetric-triangle **3,391,552 B** — on different partitions. *"The opponent is roughly halved"* is true of the **bridge → K\*** step (1,029,776 → 536,270–536,972 = 1.92×) and **is not a statement about the ladder's opponent**, against which K\* is **5.92–5.99×** smaller. *Source:* `w12b .../FINDINGS.md` §3; `w12-composition-ladder/runs/*/rows_learned_1x.json` `/summary/param_count` |
| **C21 ⭐ NEW** | *"seed 4 has no external classifier"* | ⛔ **it holds for seed 3 too.** Both pass the cache rule and **neither can be judged**: `decoder_v1_s{3,4}.pkl` and `gen-external-classifier/seed_{3,4}` are **both ABSENT**. *Source:* `w12b .../FINDINGS.md` §10 |
| **C6-phys** | the D17 headline pair "+1.1385 derived vs +1.1368 measured" | ⚠ **the paper's sharpest claim has its weakest artefact trail.** The **law** reproduces exactly and `> +1` is verified over the whole band, but **+1.1385 is not reproducible from any recorded machine-readable artefact** — `1 + γ/(2−γ) = 1.1385` at γ = 0.243303, **not a grid point**, and the per-cell fit windows were never written to JSON. A **reporting** gap, not evidence the number is wrong; nothing contradicts it. Fix = **F6** |
| **C7-phys** | D18 "underdamped slope confirmed" | derived −1.0050, measured −1.1090, **margin thin on every seed** (0.1368 vs 0.1417); the shorts' own values are **−1.006** and **−1.0020 ± 0.0003**, so **w6's −1.1090 is the outlier, not the derivation**. ⛔ **Do not write D18 up as a confirmation.** Write: *confirmed on the shorts' own designed and emergent testbeds, inconclusive on this store, by a margin the hub itself declared thin* |
| **C8-phys** | `d14_confinement.max_abs_residual = 0.0` | it compares `hess_eig(α)` to `hess_eig(0) + 2α` — **the same algebra on both sides**, so it can only ever pass. The hub caught this itself. Quote `hessian_fd_check` |

⚠ **SF15 — all SIX open items the registry fold raised, not two:** (1) the wave-12 tiers are
**CURATOR-PROPOSED** and the curator never self-confirms; (2) `claims_matrix.md` **v2.20 is PROPOSED**,
and v2.19 and below are each their own Hub's; (3) `CM-W12-6` supersedes the **framing** of `CM-R12-4`
without editing it — whether they merge is the Hub's call; (4) `N335` / `N336` are **CURATOR-ADDED
beyond the task's nine**; (5) `§I8.3` 3–6 and `§I8.5` 4–6 remain open — the `N-I8-1` band attribution,
the "9/9 seeds" phrasing, the unspent budgets, the pre-ratification spot-checks, `N318`/`N319`'s
artefact trails, `N324`'s three red tests, the alias-vs-crosswalk question; (6) whether the composition
rows should quote `pca_tensor` and let the single design carry deletion alone.
*Source:* `w13-registry-fold/FINDINGS.md` §5.

⭐⭐ **And one standing-rule question the registry escalated, for the Head alone:** `N-I8-21`(b) —
*instruments that manufactured their own verdicts* — is now at **six instances across five waves and at
least four codebases**, two of them from wave 12 (the ladder aggregator comparing an arm **with itself**
and scoring the tie as a failure; a `SMOKE_NOT_A_CLAIM: true` file supplying the wave's headline byte
ratio). ⚠ Limb (a), the duplicate-number allocation, is **CLOSED** and re-verified — 349 entry headings,
**zero duplicate N numbers**. The curator drafted a rule and **declined to adopt it**: ***before a
number leaves an instrument, state what the instrument would print if the effect were absent. If that
answer is the same number, the instrument is not measuring the effect.*** ⛔ Standing-or-wave-scoped,
where it is filed, and whether `N-I8-21` should be split are the **Hub's three decisions**, not this
spoke's. *Source:* `w13-registry-fold/FINDINGS.md` §3.

---

## §H ⛔ DECLARED OPEN ITEMS — what this spoke did not close, and why

| # | open item | why it is open, not closed |
|---|---|---|
| **OPEN-1** | ⛔⛔ **`TABLES.md` T7 and T8 are SUPERSEDED by `w12b` and the manifest has not been rebuilt.** T7 prints `seeds (n) = 1` where the artefact now has **3**; T8's verdict (*"clears at NO rung"*) is a **K = 5** statement the paper must not carry as *the* verdict. | ⛔ Rebuilding `TABLES.json` / `TABLES.md` / the figures and re-running `verify_tables.py` is the **tables spoke's** job and needs the generator, not the draft. A spoke does not edit another spoke's artefact. **The paper cannot freeze before this is done.** |
| **OPEN-2** | ⚠ **`pca_tensor` was never run at K\***, so *"the four-part ledger buys deletion, not composition"* (§4.4) is **a K = 5 statement** and the paper may not generalise it. | ⛔ Requires one replication leg per rung; **not run, not costed by this spoke** (**F11**). |
| **OPEN-3** | ⚠ **T9 and T10-ours (deletion and refusal on the K\* object) are ONE seed.** | ⛔ `single-design` is now **3/3 spent** (w12b). A further seed needs a new scored-ledger row, which is a **Hub decision** (**F12**). |
| **OPEN-4** | ⚠ **The K\* 1× composition win is a REPLICATION, not a scored row**, and **the bridge attribution is one seed.** | ⛔ `KSTAR-LADDER.json` `/REPLICATION_NOT_SCORED` = true. Stated at every mention; not upgradable without budget. |
| **OPEN-5** | ⚠ **Four figures do not exist** (F1, F3, F4, F10) and one exists only at K = 5 (`fig6`). | Analyst-minutes, **0 scored runs** — but not this spoke's to draw, and ⛔ **none may be drawn from a number this spoke invented.** |
| **OPEN-6** | ⚠ **The extra abstain output for the MLP / cont-MLP / LSTM deletion comparators is NOT BUILT.** | The spoke ran before the Head's naive-vs-naive amendment. **DECLARED NOT-RUN**, carried from w12. |
| **OPEN-7** | ⚠ **The +1.1385 constant has no machine-readable trail** (C6-phys), and **F6 cannot be run from the artefacts.** | A reporting gap in the w6 branch, not a defect in the number. ⛔ **Do not write D18 up as a confirmation.** |
| **OPEN-8** | ⚠ **`N-I8-21`(b) promotion, the six registry-fold flags, and the `pca_tensor`-vs-single-design column choice are Hub rulings**, reported here and not taken. | ⛔ A spoke does not promote a standing rule or confirm a claims-matrix version unilaterally. |
| **OPEN-9** | ⚠ **`w12b` is not in `/c/chlu/evidence/outputs/`.** Every number from it is cited at `/c/chlu/outputs/w12b-nr6-and-kstar/` and flagged. | ⛔ Folding the evidence base is the Hub's. This spoke read it because its own task file instructed it to, and flagged every citation. |
| **OPEN-10** | ⚠ **The `w12-advisor-report.md` itself now carries two numbers its own successor artefacts correct** (C20, C21). | ⛔ Editing another wave's report is not this spoke's. Both are printed as corrections. |

---

## §I ⛔ THE FALSIFIER, CHECKED AGAINST THE RESULT — and the contribution paragraph

> **The task's falsifier:** *"The evidence cannot support a contribution paragraph that survives the
> novelty STOP ⇒ say so plainly. That is a finding about the programme, not a writing problem to style
> around."*

**Verdict: it does NOT fire. A contribution paragraph survives — and it is stronger than v0's, because
every clause the referee broke has been replaced by the number the artefact holds.** Offered as a diff
target, not as prose the Head must accept; nothing in it is asserted that is not pointed at above.

> We show that a generative store whose content is **derived from a retained sufficient statistic**
> admits deletion of the **store** that is exactly the content refitted from the retained rows,
> **holding the frozen encoder and decoder fixed** — verified by two implementations on two
> environments, three seeds each, on held-out test data, with no refit performed, no pass over
> retained data and items never kept **[K1, K2]** — at the cost of a retained sufficient statistic
> whose own bytes are **2.19 %** of the deployed read in one environment (**35,904 B of 1,642,752 B
> deployed**; convention: statistic ÷ deployed read) and **23.5121× the 524,288 B store pin** in the
> other (**12,327,096 B ÷ 524,288 B**; convention: statistic ÷ pin, the statistic alone), which takes
> the never-netted four-part honest total — atoms + statistic + request-count table + coupling/holes
> — to **24.5466× the pin** **[K4]**; that the two
> implementations agree not only on where the mechanism works but on where it breaks, and that where
> it breaks the residual sits **below each implementation's own reproducibility floor**, so both
> conclude the failure is a property of the content **[K3]**; that the same store class composes,
> beating a parameter-matched trained learner on both composition rows on all three seeds at **matched
> total deployed bytes (±0.5 %)** **[K5, K6, K7]**; that whether it also beats a learner matched to its
> **own** bytes is **a property of the partition** — at K = 5 it does not, at **K\* = 32 it does at all
> five rungs on every seed**, and a bridge arm attributes **62–79 %** of the swing to the K step rather
> than to the construction **[K9, K10, K11]**; and that at the 524,288 B pin the two properties remain
> incompatible — **no object with a float64 sufficient statistic reaches that pin at rank 64 under any
> of nine swept partitions** **[K8, K18, K19]**. The composition result does not transfer to an
> exact-judge route task on **CMAZE-2**, and we isolate why: the read composes latents where the task
> composes plans — the same two halves compose to **1.0000** in plan space and **0.13** in latent space
> **[K20–K25]**.

**Why it survives the novelty STOP, clause by clause:**
- ⭐ It does **not** claim multilinear factored generation, a request-indexed tensor head, or
  tensor-product label conditioning. Those words are absent **by construction**.
- ⭐ It does **not** claim the downdate primitive. It claims the **object**: exact deletion of a
  generative model's **store**, which the crawl finds in **0 of ≈4,050** screened works **[K31]**.
- ⭐ Its mechanism sentence — *downdate the statistic, not the truncated factors* — is a distinction the
  eigenspace-arithmetic line **does not have, by its own authors' words** **[K32]**.
- ⭐ The composition clause is a **measured comparison at stated bytes and a stated K**, not a novelty
  claim in the crowded compositional-generation literature.

⚠ **Five honest weaknesses in that paragraph, stated rather than styled around:**
1. The deletion result and the w5 composition result are **on different objects**. The object that does
   both is now **scored at 3 seeds on composition (T7) but at 1 seed on deletion (T9)**.
2. ⛔ **At K\* its exact deletion does not change what it answers** — 32.25 → 32.25 **[K15]**. The
   store-scoped claim is the only one demonstrated there.
3. The K\* 1× composition win is a **replication, not a scored row**, and the bridge attribution is
   **one seed**.
4. **No row in this programme can exceed 3 seeds** — seeds 3 and 4 have verified caches and no judge.
   A ceiling on everything the paper can ever say.
5. The one environment with three scored seeds and a clean instrument returns a **negative**.

⛔ **None of the five is a reason to restyle the paragraph.** All five are in §11, §H and App. K and
the Head should keep them there.

---

## §J WHAT THIS SPOKE DID NOT DO

- ⛔ **No scored run, no compute, no model fitted, no number produced.** Every figure above is quoted
  with the path that holds it. The only arithmetic performed is the re-derivation of the 15 K\*
  per-seed margins and the T7 three-seed rows **from their own JSON fields**, to check the w12b hub's
  prose rather than take it.
- ⛔ **`$CHLU_CLAUDE_ROOT` was NOT read at any point.**
- ⛔ **No prose.** The Head writes it. Where a sentence is given verbatim above it is because the
  evidence fixes its content, and it is marked as such.
- ⛔ **Nothing written outside `/c/chlu/outputs/w13-draft-revise/`** and this spoke's own worktree
  (branch `w13/draft-revise`). ⛔ **`/c/chlu/evidence/` was not written.** ⛔ **No `.claude/`
  directory was created.**
- ⛔ **Another hub was running on this box.** Its session, worktree, output directory and processes were
  never touched; `/c/chlu/outputs/w12b-nr6-and-kstar/` was read **read-only** and only after its own
  `END_UTC.txt` was on disk. ⛔ **No background job was launched, none was left running, and NO
  BOX-WIDE KILL** was issued or was possible from anything this spoke ran. **1 thread.**
- ⛔ **No pruning.** Appendix-maximalism is in force. §D lists only sentences with **no artefact**, or
  sentences a later artefact **withdrew**.
- ⛔ **No table, figure, generator or registry belonging to another spoke was edited.** OPEN-1 is
  declared, not silently fixed.
