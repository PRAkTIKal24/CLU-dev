# w20 hub/shapes-data-matched — FINDINGS

**Branch** `w20/shapes-data-matched` · **base** `base/iter8 = 148f3b7` · worktree
`C:/chlu/worktrees/CHLU-w20-shapes-data-matched` · outputs `C:/chlu/outputs/w20-shapes-data-matched/`.
`chlu.__file__` asserted inside the worktree in every process; cache **generation 2** asserted per seed
(`assert_generation2`); seeds **0, 1, 2** on every row (seeds 3–4 have no judge). jax 0.9.0 / eqx 0.13.4.
**Acceptance artefacts:** this file, `SHAPES-MATCHED.json` (every number, with its per-corner source files),
`SHAPES-MATCHED.md` (the same rendered as tables), `PREREG-w20-shapes-data-matched-1.md` (+ Amendments 1–5,
sha256/timestamps in `PREREG-…timestamp.txt`), `LOG.md`.
⛔ Every number below is copied from `SHAPES-MATCHED.json` / `.md`; the JSON key is named beside it.

---

## 0. THE ANSWER, PLAINLY

**The data-matched margin is +7.03 novel tuples per corner (seeds +7.49 / +8.60 / +5.02), and +6.54 against
the rival at its best over every fit that ran (10× and 30× included). The draft's current margin, recomputed
from the same score files, is +39.7 … +49.1. The margin SHRINKS about 6×. It does not collapse:** on every
seed the bootstrap CI clears the pre-declared +1.0 (lowest lower bound +3.73), and the store beats the
strongest data-matched rival on 67–84 of 109 corners per seed.
(`rows34.P3_margin.store_K32_n131072`, `rows34.P3_vs_best_over_all_fits_incl_30x`,
`rows34.draft_margin_recomputed_vs_16k_comparators`.)

**What M1 predicted is what happened, but the cause is SCHEDULE, not DATA.** The draft's rivals scored
5.5 (MLP) because they were trained for 1,280 steps, untuned. An MLP given the rows-3–4 selected schedule
(20,480 steps) scores **38.6 on the same 16,384 rows**, 39.1 on 32,768 rows and 39.2 on all 131,072 rows.
At matched steps, 8× more data buys the MLP ~0.5 novel tuple per corner. The store is just as insensitive
to data (K\* = 32: 47.28 at 16,384 rows → 48.01 at 131,072). So the draft's +39…+49 was carried by an
**under-trained** comparator; data matching is not what closes it, the training budget is.
(`rows34.arms`: `mlp__declared@test16k_stepmatched`, `mlp__declared@test32k_stepmatched`,
`mlp__declared@test`, `w12comp_mlp__as_trained`, `store_K32_n16384`, `store_K32_n131072`.)

**The deletion result under the Head's new definition does not flatter this hub:** refitting the address
book after deleting red ∧ sphere removes **red composition with the other shapes** (−32.1 per corner on
the 20 red corners, seed 0), −6.13 / −6.43 / −6.30 on all 109 corners, on 3/3 seeds. The old "unchanged"
row existed only because the book was **carried**, not refit — referee M3's objection, now measured. §5.

---

## 1. What ran, and the budget

| scored run | ledger row (`/c/chlu/SCORED-LEDGER.md`, written BEFORE) | what |
|---|---|---|
| **1/3** | 2026-09-23T19:55:47Z | Table M-131k, rows 3–4 TEST, seeds 0/1/2 (replications), incl. LACE/PoE@131k and the diagnostics |
| **2/3** | 2026-09-23T21:50:00Z | Table M-16k, rows 3–4 TEST, seeds 0/1/2 |
| **3/3** (LAST) | 2026-09-23T23:36:23Z | matched deletion — every rival's retrain-without on the SAME deleted rows |

**Budget: 3 of 3 spent; target `shapes-matched` exhausted.** The ladder, the store-side deletion
measurements (a)–(d) and every val number are free (w12b precedent; Head AMENDMENT-2). VAL ran first for
every new config (LOG.md: val stores 16,384/32,768/131,072 K=32/5; val rivals mlp/transformer/cont_mlp
DECLARED+TENX; val retrain-without; LACE/PoE@131k val).

## 2. The rivals — ONE set, data-matched, trained to convergence (M1, M2)

`mlp`, `lstm`, `transformer`, `cont_mlp` at the **131,072-parameter read-path pin** (measured 130,514 /
130,721 / 130,433 / 131,891), fitted on the **store's EXACT 131,072-row pool** (`hub-factor-wells/run`
pool; measured identical to `fw.fit_pool(train_rows, 131072, 20260919)`, so the draft's 16,384 comparator
rows are its first 16,384). Spoke-A protocol (6 short lr×wd trials + epochs {20, 40}, selected on a fixed
10 % held-out split); DECLARED = the selection refit on all 131,072 rows; **TENX = 10× the optimiser steps**;
**loss curves printed** (`rows34.convergence_evidence.<family>.<seed>`: per-epoch train MSE, the selection
stage's per-epoch held-out MSE). **Every family selected lr = 3e-3 and 40 epochs — the edge of both spoke-A
grids**; the grids were NOT widened after the fact (PREREG A1.2). The TENX column is the instrument for that.

**P1 (convergence), per seed** (`rows34.P1_convergence`):

| family | DECLARED | TENX | TENX − DECLARED | P1 | 30× (diagnostic, A5) |
|---|---|---|---|---|---|
| mlp | 36.71 / 37.57 / 43.17 | 40.01 / 38.81 / 44.10 | +3.30 / +1.24 / +0.93 | FAIL / FAIL / PASS → TENX carries s0, s1 | 41.48 / 36.54 / 43.50 (s0 still +1.47 over TENX → flagged NOT CONVERGED at 10×) |
| lstm | 33.97 / 32.36 / 40.70 | 33.69 / 32.54 / 39.85 | −0.28 / +0.18 / −0.84 | PASS ×3 | — |
| transformer | 25.56 / 27.79 / 38.65 | 25.30 / 27.24 / 37.84 | −0.26 / −0.55 / −0.81 | PASS ×3 | — |
| cont_mlp | 17.92 / 16.54 / 26.84 | 28.53 / 32.94 / 42.60 | +10.6 / +16.4 / +15.8 | FAIL ×3 → TENX carries | 27.28 / 34.01 / 38.71 (s1 +1.07 over TENX → flagged) |

⛔ Defect 1 of the brief is closed: **no margin below is carried by a rival that failed its budget**. Where
P1 failed, the TENX fit carries it, and the headline is also printed against the best fit of all (30×).
The one residual: at seed 0 the MLP is still gaining +1.47 from 10× to 30× (36.71 → 40.01 → 41.48). The
increments shrink, but it is not proven converged. The best-over-all-fits row (+6.02 at seed 0) is the
honest number for that seed.

## 3. Table M-131k — rows 3–4 TEST (60 pair + 49 n-way corners); fit pools in the caption

**Caption:** store fit on **131,072** rows; data-matched rivals fit on the **same 131,072** rows; the
draft's comparators on **16,384**; `learned_ref_ownpool` on **32,768**; seeds column = 3 on every row.
Ceiling = 52.18 distinct novel requested tuples per corner (row 3: 36; row 4: 72). Source `rows34.arms`.

| arm (fit rows) | all 109 (s0 / s1 / s2) | mean | row 3 | row 4 | n |
|---|---|---|---|---|---|
| **store K\*=32 (131,072)** — reused T7 objects, per-corner reproduction IDENTICAL on 3/3 seeds | 47.50 / 47.40 / 49.12 | **48.01** | 32.35 | 67.18 | 3 |
| store K=5 (131,072) — s0 reused, s1/s2 fitted | 43.79 / 45.68 / 47.80 | 45.76 | 30.62 | 64.29 | 3 |
| mlp DECLARED (131,072) | 36.71 / 37.57 / 43.17 | 39.15 | 25.39 | 55.99 | 3 |
| mlp TENX (131,072) | 40.01 / 38.81 / 44.10 | 40.97 | 26.42 | 58.80 | 3 |
| lstm DECLARED (131,072) | 33.97 / 32.36 / 40.70 | 35.68 | 23.47 | 50.63 | 3 |
| transformer DECLARED (131,072) | 25.56 / 27.79 / 38.65 | 30.67 | 20.05 | 43.67 | 3 |
| cont_mlp TENX (131,072; carried) | 28.53 / 32.94 / 42.60 | 34.69 | 22.59 | 49.50 | 3 |
| `learned_ref_ownpool` (32,768; 2,560 steps; EXISTING) | 9.18 / 11.12 / 15.13 | 11.81 | 7.49 | 17.10 | 3 |
| draft's 16k mlp / lstm / transformer / cont_mlp | 5.54 / 6.27 / 1.65 / 0.04 (means) | | | | 3 |
| LACE @131,072 (own budget) | 0.018 / 0.046 / 0.073 | | | | 3 |
| PoE @131,072 (own budget) | 0.000 / 0.000 / 0.000 | | | | 3 |
| LACE @32,256 (w15, beside) | 0.165 / 0.073 / 0.229 | | | | 3 |
| PoE @32,256 (w15, beside) | 0.009 / 0.000 / 0.000 | | | | 3 |
| table — **FLOOR, never a rival** | 8.23 / 5.07 / 9.75 | 7.69 | 5.17 | 10.77 | 3 |

**Controls (P2), every store row, every seed** (`rows34.P2_controls`): `landscape_removed`,
`landscape_shuffled`, `landscape_shuffled_across_wells`, `random_atoms` all read **0.000** on all six store
variants (K\* and K = 5 at 131,072 / 32,768 / 16,384 rows) × 3 seeds. ⭐ **Defect 2 of the brief is
closed:** `landscape_shuffled_across_wells` is a **derangement**, and its permutation is printed on every
row (`SHAPES-MATCHED.md`). ⚠ **Measured:** the historical T7 permutation at **seed 2 had 2 fixed points**
(N350 realised at K = 32); it still read 0.000, as does the derangement.

**LACE/PoE scope, verbatim:** *LACE/PoE as instantiated in this feature space at their own declared
budgets — not a refutation of LACE in general.* At 131,072 rows (130,560 fit + 512 val, their own 36-trial
selections, LACE classifier 4,073,663 params, PoE 3,556,665 params) they score **no better** than at
32,256 (`LACE_PoE_at_131072.legs.shipped`). Data matching does not rescue them.

**P3 — the margin** (paired per corner, bootstrap CI; `rows34.P3_margin`, `…P3_vs_best_over_all_fits_incl_30x`):

| store | vs | s0 | s1 | s2 | mean | reading |
|---|---|---|---|---|---|---|
| K\*=32 @131,072 | strongest P1-carried rival (mlp TENX) | +7.49 [5.47, 9.65] | +8.60 [6.80, 10.46] | +5.02 [3.73, 6.41] | **+7.03** | **SHRINKS** |
| K\*=32 @131,072 | best of every fit incl. 30× | +6.02 [3.83, 8.33] | +8.60 | +5.02 | **+6.54** | **SHRINKS** |
| K=5 @131,072 | strongest P1-carried | +3.78 [2.15, 5.48] | +6.87 | +3.70 [2.51, 4.96] | +4.78 | SHRINKS |
| K\*=32 @131,072 | the draft's 16k comparators | +40.9 … +47.5 | +42.8 … +47.4 | +39.7 … +49.1 | — | the draft's number |

Per row: K\* − mlp TENX on row 3 (pair) ≈ +5.9 and on row 4 (n-way) ≈ +8.4 (from the means above).

## 4. Table M-16k — the store re-fitted at the rivals' data sizes (rows 3–4 TEST)

**Caption:** store fit on the **first 16,384** rows (= the draft comparators' exact rows) and on the
**first 32,768** rows (= `learned_ref_ownpool`'s exact rows; PREREG A2.2 — spoke A's literal 32,256-row
Montero pool holds both held-out values and cannot be used); rivals as printed. `rows34.arms`, `…P3_margin`.

| store | all 109 (s0 / s1 / s2) | mean | vs draft 16k mlp | vs `learned_ref_ownpool` (32,768) | vs 131k data-matched mlp (TENX) |
|---|---|---|---|---|---|
| K\*=32 @16,384 | 46.15 / 47.09 / 48.59 | 47.28 | +40.95 / +43.85 / +40.41 (mean +41.74) | +36.96 / +35.97 / +33.46 (+35.46) | **+6.30** mean |
| K\*=32 @32,768 | 47.28 / 47.28 / 48.94 | 47.83 | +42.08 / +44.04 / +40.76 (+42.29) | +38.09 / +36.16 / +33.81 (+36.02) | **+6.86** |
| K=5 @16,384 | 43.18 / 44.65 / 47.99 | 45.28 | +37.99 / +41.41 / +39.82 (+39.74) | +34.00 / +33.53 / +32.86 (+33.46) | +4.30 |
| K=5 @32,768 | 43.33 / 45.76 / 48.04 | 45.71 | +38.14 / +42.52 / +39.86 (+40.17) | +34.15 / +34.64 / +32.91 (+33.90) | +4.74 |
| K\*=32 @131,072 | 47.50 / 47.40 / 49.12 | 48.01 | +42.30 / +44.17 / +40.94 (+42.47) | +38.31 / +36.28 / +33.99 (+36.20) | +7.03 |

**P4:** the store at 16,384 rows keeps essentially its whole margin over the 16k comparators (47.28 vs
48.01 at 131k). So "the store only won on data" is refuted for the store side. The rival side is §0: its
deficit was training steps.

## 5. Matched deletion (run 3/3 + the Head's four measurements) — rows 3–4, pool 131,072

**Definition (Head AMENDMENT-2):** forget c = (1) downdate the store, (2) downdate the **address book's
statistics and refit the book from them**, (3) punch the hole. ⛔ The row "c still composes after forgetting
c" is **retired** — not computed, not printed (the w15 evidence stays in its artefacts).
Source: `deletion_matched.per_scope.<scope>.store.<seed>` (files `deletion_store/seed*/…json`).

| scope (rows deleted) | (a) store == refit-without, max\|mode dev\| | (b1) book statistic dev | (b2) rendered-feature rel dev | (c) DISPLAY | (d) survivors before → after (s0 / s1 / s2) | (d) downdate − refit-without |
|---|---|---|---|---|---|---|
| value red (11,044) | 5.96e-08 / 4.82e-08 / 2.01e-08 | **0** | 1.2e-09 – 1.9e-09 | red requests refused 1.000, false refusal 0.000 (value bit, 1 bit, 0 rows) | 50.92→50.87 / 50.53→50.85 / 52.61→52.00 | **+0.000** ×3 |
| red ∧ sphere (5,511) | 5.38e-07 / 1.07e-08 / 6.25e-08 | **0** | 2.0e-09 – 7.8e-08 | no rows 3–4 corner requests red∧sphere (refusal undefined); false refusal 0.000 | **47.50→41.37 / 47.40→40.97 / 49.12→42.82** | **+0.000** ×3 |
| 1 % items (1,311) | 2.34e-07 / 1.49e-08 / 3.16e-08 | **0** | 5.8e-10 – 9.7e-10 | N/A — no hole exists (0 certifiable requests) | 47.50→47.48 / 47.40→47.29 / 49.12→49.11 | **+0.000** ×3 |

- **(a), (b): exact.** The book refit from the downdated statistic equals the refit-without book at 1e-8 or better
  in the rendered features, and composition is **identical** (Δ +0.000 on every corner, 9/9 legs).
- ⚠ **(b) has a cost that must be said:** the ALS book has **no compact exact statistic**. It sees data only
  through per-tuple aggregates, and every Shapes3D tuple is one row, so its exact statistic is the per-row
  first moment of its 117,964 fit rows: **0.97 GB**, i.e. the fit data itself. "Adaptation reads 0 pool
  rows" is true; it is **not** a size claim. The book refit costs 2,367–5,913 s (≈ the refit-without book).
- ⚠ **(d) FAILS its bar for red ∧ sphere on 3/3 seeds** (Δ −6.13 / −6.43 / −6.30 against the pre-declared
  |Δ| ≤ 1.0). Seed 0 breakdown: the 20 corners requesting **red with other shapes** drop **−32.1** per
  corner; the 36 non-red sphere corners −1.47; the 53 others +0.51. Item deletion (a random 1 %) moves
  nothing (−0.02 / −0.11 / −0.01), so this is the deleted **content**, not refit instability. **Under the
  Head's definition, deleting a conjunction removes the store's ability to compose one of its values with
  other values.** The w15 "composition unchanged" row held only because the book was carried.
  (Value-scope (d) passes: −0.06 / +0.33 / −0.61.)

**Rivals, retrain-without on the SAME rows from the SAME pool** (`deletion_matched.per_scope.<scope>.rivals`):
ours-after − rival-retrain-without on the survivor corners:

| scope | mlp | lstm | transformer | cont_mlp |
|---|---|---|---|---|
| value | +11.74 / +11.21 / +7.47 | +13.67 / +15.54 / +9.26 | +23.07 / +21.08 / +11.84 | +30.93 / +24.70 / +32.61 |
| red ∧ sphere | +5.26 [2.09, 8.49] / **+2.49 [−0.94, 6.02]** / **+2.60 [−0.50, 5.67]** | +8.30 / +7.24 / **+2.48 [−0.87, 5.82]** | +16.64 / +13.62 / +4.93 | +23.67 / +23.67 / +25.54 |
| item | +11.60 / +9.79 / +6.95 | +14.23 / +14.71 / +8.23 | +22.42 / +19.21 / +11.34 | +28.34 / +28.84 / +20.60 |

**Cost line, every rival:** retrain-without reads **120,028 / 125,561 / 129,761** rows and rewrites
521,732–527,564 B in **548–2,065 s**. Our store downdate reads **0** pool rows (seconds per scope in
`…a_store_equals_refit_without.downdate_seconds`); the book refit reads 0 pool rows but needs the 0.97 GB
statistic. ⚠ **On red ∧ sphere the store is NOT significantly ahead of the MLP at 2 of 3 seeds, nor of the
LSTM at seed 2**: the rivals lose < 1 novel/corner when retrained without red∧sphere, while the store loses
6. Rival retrains run at the DECLARED schedule (PREREG A3.2), so the mlp (s0, s1) and cont_mlp P1 flags
travel with these rows. ⭐ w15's 16k-rival deletion columns → **appendix only**.

## 6. The ladder (free replication; rung pools 131,072 rows; SEE `SHAPES-MATCHED.md` § The ladder)

**Caption:** every rung's store AND data-matched rivals fit on the rung's own **131,072**-row pool (M-16k stores on its first **16,384 / 32,768** rows); the draft's ladder rivals = spoke A `@ref` (118k params, **32,256** Montero rows); `learned_ref_ownpool` = **32,768** rung rows, 2,560 steps; LACE/PoE = w15 at **32,256** rung rows (131k NOT RUN, §7). Seeds = 3 on every cell. Rungs n = 3–6 probe the IDENTICAL 120 corners (a training-data axis, not depth); ceiling 36 (n=2) / 72 (n≥3). mlp and cont_mlp carried at TENX on every rung/seed (PREREG §4: they failed P1 on rows 3–4). Source `ladder.<n>` in `SHAPES-MATCHED.json`.

| rung | store K\*=32 (131k) | K\*=32 @16,384 | K\*=32 @32,768 | strongest data-matched rival (carried; mean over seeds) | **K\* − strongest, per seed [mean]** | draft-style: K\* − strongest spoke-A `@ref` | `learned_ref_ownpool` | LACE / PoE (w15, 32k) | table FLOOR | n |
|---|---|---|---|---|---|---|---|---|---|---|
| n=2 | 32.64 | 32.08 | 32.15 | mlp tenx (27.51) | +4.63 / +6.92 / +3.85 [**+5.13**] | +12.38 / +10.50 / +9.38 | 7.48 | 0.061 / 0.006 | 5.17 | 3 |
| n=3 | 61.61 | 62.15 | 61.97 | mlp tenx (50.43) | +12.89 / +15.95 / +4.71 [**+11.18**] | +15.72 / +21.22 / +9.92 | 11.79 | 0.156 / 0.000 | 9.53 | 3 |
| n=4 | 66.09 | 65.64 | 65.50 | mlp tenx (53.54) | +14.72 / +15.80 / +7.12 [**+12.55**] | +20.81 / +24.25 / +15.21 | 13.28 | 0.172 / 0.006 | 9.53 | 3 |
| n=5 | 66.58 | 66.26 | 66.57 | mlp tenx (55.89) | +12.00 / +13.54 / +6.52 [**+10.69**] | +21.21 / +25.25 / +15.29 | 14.96 | 0.136 / 0.003 | 9.53 | 3 |
| n=6 | 66.60 | 67.01 | 67.05 | mlp tenx (56.30) | +11.72 / +11.56 / +7.62 [**+10.30**] | +21.48 / +24.68 / +15.66 | 15.61 | 0.133 / 0.003 | 9.53 | 3 |

All K\* legs: CI95 of K\* − strongest data-matched rival lies above 0 on every seed and rung (lowest lower bound +2.46, at n=3 seed 2); the smallest seed-level margin is +3.85.

**Reading.** Against rivals fitted on the rung's own 131,072 rows (MLP carried at its 10× fit), the ladder margin is
**+5.1 (n=2) and +10.3 … +12.6 (n=3–6)**. Against the draft's ladder rivals (spoke A `@ref`) it was +9.4 … +25.3.
The shrink is smaller than on rows 3–4, and **no rung collapses**. The store refit on 16,384 or 32,768 rung rows
matches the 131,072-row store within about 0.6 per corner on every rung, and the ladder repeats the rows 3–4 picture:
the store barely uses the extra data.
⚠ The ladder rival spot checks also fail P1 in places: at n=2 seed 0, lstm moves 20.55 → 21.85 at 10× (+1.30). Per
PREREG §4 only mlp/cont_mlp are carried at 10× on every leg; the lstm flag is printed, not netted
(`ladder.2.P1_convergence`).
⚠ **The EXISTING ladder K=5 arm** (w12's `DowndatableStore`, "as already fitted") carries only 2 of the 4 controls.
Its strong-form `random_atoms` (book refit against random atoms) reads **1.24 at n=4 seed 2 and 2.03 at n=5 seed 2**,
over the 1.0 bar. **Its n=4 and n=5 rows are therefore not claimable under P2.** `landscape_removed` and
`landscape_shuffled_across_wells` were never run on that existing arm (DECLARED). Every w20 store on the ladder (K\*
rebuilt, and K\*/K=5 at 16,384/32,768) collapses on all four controls, with the derangement printed
(`ladder.<n>.P2_controls`).
⭐ Rebuilt K\*=32 ladder stores reproduce w12b's per-corner rows on **15/15** legs (`ladder.<n>.integrity_rebuilt_vs_w12b`).


## 7. What did NOT run — DECLARED, with reasons

1. **LACE/PoE @131,072 on the ladder rungs n = 2…6 × seeds 0–2 (15 legs).** Measured cost at 131k: a
   36-trial selection of 5,955–11,034 s plus a 50-epoch classifier that took ~7.5 h on the val leg, ~3.5 GB
   each, dispatch-bound at ~0.3 core. Two started rung legs (2:0, 2:1) were stopped before producing a number.
   The w15 32,256-row rung rows remain the only LACE/PoE rung rows. (Rows 3–4 at 131k DID run, 3 seeds.)
2. **VAL `lstm`** (fit stopped for memory): the val stage covered the shared protocol on mlp, transformer and
   cont_mlp; lstm is measured on test with its own TENX.
3. **TENX on the ladder** runs only as the pre-declared spot check (rungs 2 and 6, seed 0, all four families).
4. **30× diagnostics** only for mlp and cont_mlp, the two families that failed P1.
5. **cont_mlp / mlp retrain-without at TENX or 30×** (9 fits × ~1–5 h): deletion rivals run at DECLARED; the
   P1 flags travel with the rows.
6. **`learned_1x`** (536k / ~3.2M params, matched to the store's MEMORY ledger) is **not printed**. COMMON:
   rivals are matched on read-path parameters; the inflated-rival row does not exist.
7. **`landscape_removed` and `landscape_shuffled_across_wells` on the EXISTING ladder K=5 arm** (w12's
   `DowndatableStore`, reused "as already fitted"): never run on that arm. Its two existing controls are
   printed, and its n=4/n=5 rows fail P2 on seed 2 (§6).
8. **Spoke A's literal 32,256-row Montero pool as a store fit pool:** it holds both value-holdout held-out
   values (3,157 rows at object_hue 0.5, 7,972 at shape 0.0). The store was fitted on the 32,768-row nested
   prefix of its own pool instead (PREREG §2.3, A2.2).

## 8. Defects found and fixed during the run (each in LOG.md with its timestamp)

- DEV smoke on generation-1 features (CHLU_GEN_FEATURES_ROOT unset). Caught by `assert_generation2`; the output
  tree `rivals/val_SMOKE` is dev-only and unused. env.sh fixed; every fit/store script now asserts generation 2.
- The val `seen_mask` marked every val tuple seen (val corners live in `train_rows`) → first val pass VOID
  (`scores/val_VOID_seenmask_all_seen`); fixed to train_rows − val corner rows.
- **Rung dead-channel masks are per seed**; `load_pool('rung:k')` read seed 0's for all seeds. The seed-1/2 rung
  stores and their scores were VOIDED (`VOID_rung_deadmask_seed12_*`) and rebuilt. Rows 3–4 are unaffected: T7
  used one partition file for all seeds, and so did we.
- Ladder K\*=32 rebuilds with w12b's cached address books: a threshold-width check. Rebuilt rows reproduce w12b's
  per corner (`ladder.<k>.integrity_rebuilt_vs_w12b`).
- One orphan slow-path scorer ran beside the fast path (identical per corner, verified at seed 0; stopped).

## 9. What would have proved this wrong, and what did

- **The margin collapsing (≤ +1.0):** did NOT happen: +7.03 (+6.54 against the best of all fits).
- **The margin holding (≥ +39):** did NOT happen. **The draft's +39…+49 is not a data-matched number; it must be
  replaced by ≈ +7 (rows 3–4, K\* = 32), with the P1 table beside it.**
- **Rivals still improving at TENX:** partly. cont_mlp (all seeds) and mlp (s0, s1) needed TENX. The 30×
  diagnostic shows cont_mlp saturated (s0, s2) and mlp s0 still creeping (+1.47). Printed, not netted.
- **T7 store not reproducing:** it reproduced per corner on 3/3 seeds.
- **The across-wells derangement not collapsing:** it collapsed (0.000) everywhere.
- **(d) composition on the other values unchanged after deletion:** FAILED for red ∧ sphere on 3/3 seeds (§5).
  This is the result that most needs to reach the paper.
