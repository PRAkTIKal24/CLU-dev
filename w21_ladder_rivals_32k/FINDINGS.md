# w21 spoke/ladder-rivals-32k — FINDINGS

**Branch** `w21/ladder-rivals-32k` · **base** `base/iter8 = 148f3b7` · worktree
`C:/chlu/worktrees/CHLU-w21-ladder-rivals-32k` · outputs `C:/chlu/outputs/w21-ladder-rivals-32k/`.
**FREE REPLICATION: no ledger row written, 0 scored runs spent.** The scorer refuses any corner set other than the
ladder rungs (`scripts/w21_score.py`). `chlu.__file__` was asserted inside the worktree in every process. Gen-2 feature
cache was asserted per seed. Seeds 0, 1 and 2 are in every cell. jax 0.9.0 / eqx 0.13.4 / numpy 2.4.1.
Nothing that `hub/shapes-data-matched` (w20) produced was refitted, re-scored or written. Its files were only read.

**Artefacts:** `LADDER-RIVALS-32K.json` (every number, each with its per-corner score file or fit artefact),
`TABLES.md` (rendered from that JSON; the key path is in each section header), `CURVES.md` (the loss curves),
`CAPTION.md` + `caption.json` (the caption for `figures-redraw`, naming all pool sizes and step budgets),
`checks/pool_identity.json`, `checks/x1_encoding_identity.json`, `LOG.md`.
⛔ Every number below is copied from `LADDER-RIVALS-32K.json` / `TABLES.md`, and the key is named beside it.

---

## 0. The answer, plainly

**Fitted on the store's own 32,768 content rows, the rivals leave the ladder margin where the 131,072-row rivals left
it.** K\*=32 minus the strongest rival per seed:

| rung | **A** spoke-A `@ref` · 32,256 rows · 5,040 steps | **B** w20 · 131,072 rows · 20,480 / 204,800 steps | **C** w21 · 32,768 rows · 5,120 / 51,200 steps | **C-sm** w21 · 32,768 rows · step-matched to B |
|---|---|---|---|---|
| n=2 | +10.76 | +5.13 | **+6.08** (+6.65 / +7.37 / +4.22) | +5.87 |
| n=3 | +15.62 | +11.18 | **+11.48** (+13.01 / +15.37 / +6.07) | +13.40 |
| n=4 | +20.09 | +12.55 | **+14.00** (+16.14 / +16.73 / +9.13) | +14.54 |
| n=5 | +20.58 | +10.69 | **+9.47** (+11.12 / +10.60 / +6.69) | +10.62 |
| n=6 | +20.61 | +10.30 | **+10.36** (+11.30 / +12.49 / +7.29) | +9.30 |

(`rungs.<n>.margins_K32_store_minus_rival.{setA_vs_strongest, setB_vs_strongest_declared_or_tenx_RECOMPUTED,
setC_vs_strongest_declared_or_tenx, setC_vs_strongest_stepmatched}.mean_delta`.) Averaged over rungs: A +17.53,
B +9.97, C +10.28, C-sm +10.75. Taking the **rival's best case** instead (the per-seed max over *every* set-C fit)
gives +5.57 / +11.48 / +13.20 / +9.01 / +9.30 (`setC_vs_best_of_every_fit`). Every bootstrap CI95 in sets B, C and
C-sm has a lower bound above 0. The lowest is +1.98, at n=2 seed 2 against the step-matched mlp (per-seed Δ +3.25).

**What this says.** MF-2 is real on paper: set B gave the rivals 4× the store's content rows. Measured, it does not
matter. Rivals fitted on exactly the store's 32,768 rows land within about ±1.5 novel/corner of the 131k set at every
rung, and the difference goes in both directions. This matches w20's rows 3–4 reconciliation: **at matched steps the
pool size barely moves these rivals; the step budget is what moves them** (§3). The ladder shrink from set A to set
B/C is real and survives the data match.

## 1. What ran

- **Pool, read from the artefacts** (`checks/pool_identity.json`, `w21_lib.store_pool_n`): on all 15 rung×seed K32
  legs, w12b `kstar_ladder/<n>_seed<s>_K32/LEG.json` has `n_fit_rows = 32768` and `partitions.json` has
  `n_store_train = 32768`. The code asserts that the two agree. The store's content is `pool[:n_store_train]`
  (`exp_gen_factor_wells.py:820`). The rung pool is identical, on 15/15 legs, to w12-composition-ladder
  `runs/<n>_seed<s>/partitions.npz`, whose first 32,768 rows `learned_ref_ownpool` was fitted on
  (`w12_ladder_1x.py:91`). Every set-C fit therefore read **pool[:32768]: the store's content rows, which are also
  learned_ref_ownpool's rows**. Every fit JSON records `n_fit_rows = 32768` and the artefact paths it read the
  size from.
- **Set C: 4 families × 5 rungs × 3 seeds = 60 legs, all rc = 0.** The fits are mlp, lstm, transformer and
  cont_mlp at the same read-path float match as set B (130,514 / 130,721 / 130,433 / 131,891 params). Each leg ran
  the spoke-A protocol **re-run on the 32,768 rows**: 6 short lr×wd trials, then epochs {20, 40}, selected on a
  fixed 10% held-out split. **Every one of the 60 legs selected lr = 3e-3 and 40 epochs, the edge of both grids,
  as at 131k in w20.** The grids were not widened; the TENX and step-matched columns test that edge. Stages per leg:
  - DECLARED: 40 epochs = 5,120 steps.
  - **TENX: 400 epochs = 51,200 steps (the 10×-steps column).**
  - **STEP-MATCHED** to set B's carried budget: mlp and cont_mlp 204,800 steps; lstm and transformer 20,480 steps.
  - mlp only: `ownpoolsched`, learned_ref_ownpool's schedule (20 epochs, lr 1e-3, wd 0 = 2,560 steps).

  The fits and 195 score files ran as 150 queued jobs, all rc = 0.
- **Loss curves:** every stage's per-epoch train MSE and every selection's per-epoch held-out MSE are in the fit
  JSONs. `CURVES.md` prints them subsampled for all 60 legs (`rungs.<n>.convergence_evidence`).
- **Scoring:** the ladder's own path, `w12b_ladder_leg.score_arm` with FastJudge. The FastJudge positive control
  was asserted identical in every scoring process.
- **Integrity of the comparison:** set B was recomputed from w20's own score files and **equals w20's printed ladder
  P3 on 15/15 legs** (`setB_integrity_vs_w20_printed.identical = true`). Set A reproduces w20's "draft-style"
  column. The margin convention is w20's, applied the same way to A, B and C: per seed, the rival is the max over
  families of max(DECLARED, TENX), paired per corner, `fw.paired_estimand` bootstrap.

## 2. THE CROSS-CHECK: `learned_ref_ownpool`

**X1: same rows, same steps. This is the check that could falsify the spoke, and it passes.** Both arms are mlp,
width 61, 130,514 params, the rung's first 32,768 rows, 20 epochs, lr 1e-3, wd 0, batch 256, 2,560 steps. The bar
was pre-declared in LOG at 13:58Z, before any score existed: |Δ of the 3-seed means| ≤ the larger across-seed SD.
Source: `crosscheck.X1_same_rows_same_steps`.

| rung | `learned_ref_ownpool` (s0/s1/s2) | mine `mlp__ownpoolsched` (s0/s1/s2) | Δ mine − ref | bar | verdict |
|---|---|---|---|---|---|
| n=2 | 5.57 / 7.58 / 9.30 = 7.48 ± 1.87 | 7.37 / 5.43 / 10.83 = 7.88 ± 2.74 | +0.39 | 2.74 | REPRODUCES |
| n=3 | 8.22 / 10.91 / 16.26 = 11.79 ± 4.09 | 8.62 / 9.53 / 14.72 = 10.96 ± 3.29 | −0.84 | 4.09 | REPRODUCES |
| n=4 | 8.54 / 12.53 / 18.77 = 13.28 ± 5.15 | 11.88 / 11.88 / 22.29 = 15.35 ± 6.01 | +2.07 | 6.01 | REPRODUCES |
| n=5 | 9.87 / 15.31 / 19.71 = 14.96 ± 4.93 | 13.85 / 13.38 / 23.84 = 17.03 ± 5.91 | +2.06 | 5.91 | REPRODUCES |
| n=6 | 10.67 / 15.12 / 21.03 = 15.61 ± 5.20 | 14.53 / 13.04 / 23.88 = 17.15 ± 5.87 | +1.54 | 5.87 | REPRODUCES |

⚠ **Stated plainly: per seed, the two are not the same number.** Per-seed paired Δ ranges from −2.15 to +4.13, and the
per-corner CI95 excludes 0 on **12 of 15** legs (`per_seed.<s>.paired_mine_minus_ref`). Mine is higher at seed 0 on
5/5 rungs, lower at seed 1 on 5/5, and higher at seed 2 on 4/5. Mine is also higher on 4 of 5 rung means, by up
to +2.07. This is reported, not netted.

**What I checked, so the per-seed gap has a named cause:**
1. The rows are identical (`checks/pool_identity.json`). The input encoding is identical element for element: a
   32,768 × 57 one-hot, checked at rungs 2 and 6 (`checks/x1_encoding_identity.json`). Width and params are
   identical (61 / 130,514).
2. The trainer is the same optimiser, schedule and per-epoch jax-key permutation. w20's `train_curve` is a
   curve-recording copy of `w4_learners.train`; I read both.
3. **Diagnostic replay, which does not change the verdict:** `scripts/w21_x1_replay.py` refits learned_ref_ownpool's
   own recipe (`w12_ladder_1x.fit_one`) **with its own PRNG keys**. It reproduces the EXISTING rows **identically,
   per corner, on 15/15 legs** (0 corners differing of 60/120; `crosscheck.X1_diagnostic_ref_recipe_replay`).

⇒ The only thing that differs between the two X1 arms is the PRNG draw (init key and shuffle key). The per-seed gap
is the **RNG-draw variance of a single 2,560-step fit, up to about ±4 novel/corner at a fixed seed index**. That
spread is the same size as the step-matched-vs-TENX differences for the converged families in §3. Family
differences smaller than about 4 on a single seed should therefore not be read as real.

**X2: same rows, different steps. The difference is expected and reconciled, as the Hub amendment directs.** Source:
`crosscheck.X2_same_rows_different_steps`. All columns are mlp on the same 32,768 rows except the last, which is
w20's 131,072-row mlp:

| rung | ref · 2,560 steps | mine · 2,560 | mine DECLARED · 5,120 | mine TENX · 51,200 | mine · 204,800 | w20 mlp TENX · 131,072 rows · 204,800 |
|---|---|---|---|---|---|---|
| n=2 | 7.48 | 7.88 | 24.37 | 26.46 | 26.77 | 27.51 |
| n=3 | 11.79 | 10.96 | 41.65 | 50.13 | 48.21 | 50.43 |
| n=4 | 13.28 | 15.35 | 44.55 | 52.09 | 51.55 | 53.54 |
| n=5 | 14.96 | 17.03 | 48.40 | 57.11 | 55.96 | 55.89 |
| n=6 | 15.61 | 17.15 | 49.60 | 56.24 | 57.30 | 55.47 |

learned_ref_ownpool is **not contradicted**. It is the same rows at 1/2 the steps of DECLARED, 1/20 of TENX and 1/80
of the step-matched fit. Doubling its steps (2,560 → 5,120) more than triples its score (×3.2 – ×3.5 by rung). Past about 51,200 steps the mlp is flat. **At
204,800 steps, 32,768 rows and 131,072 rows give the same mlp to within −2.2 … +1.8.**

## 3. Convergence: the 10×-steps column and the step-matched column

Source: `TABLES.md` Table P1 and `rungs.<n>.setC_w21_32768.<fam>.P1`. The bar is TENX − DECLARED > 1.0, the same as
w20.

| family | P1 FAIL (TENX − DECLARED > 1) | TENX − DECLARED range | step-matched − TENX range (legs > +1) | reading |
|---|---|---|---|---|
| mlp | 14/15 | −0.32 … +11.35 | −4.76 … +2.02 (3/15) | Under-trained at 5,120 steps. **Converged by 51,200**: 4× more steps (204,800) moves it within ±RNG noise, in both directions. |
| lstm | 10/15 | −0.15 … +5.88 | −2.22 … +3.31 (4/15) *(20,480 < 51,200 steps)* | Mostly flat past 5,120 steps. |
| transformer | 5/15 | −2.98 … +5.72 | −0.90 … +2.54 (5/15) *(20,480 < 51,200)* | Flat. |
| cont_mlp | 15/15 | +11.15 … +28.93 | −1.25 … **+19.33** (12/15) | **Still gaining at 51,200 steps and not proven converged at 204,800.** It never carries the margin: its best fit (≤ 48.81 mean at n=5) stays below the mlp on every rung. |

The mlp is the strongest set-C rival on every rung and seed, in both C and C-sm (`TABLES.md` "Strongest rival per
seed"). So the ladder margin is carried by a family that the step-matched column shows to be converged.

**Data at matched steps** (`TABLES.md` Table F; the C step-matched column against B's carried column, 3-seed means):

| family | budget | n=2 (32k / 131k) | n=3 | n=4 | n=5 | n=6 |
|---|---|---|---|---|---|---|
| mlp | 204,800 | 26.77 / 27.51 | 48.21 / 50.43 | 51.55 / 53.54 | 55.96 / 55.89 | 57.30 / 55.47 |
| lstm | 20,480 | 23.54 / 23.36 | 40.48 / 38.18 | 45.87 / 45.32 | 49.39 / 49.88 | 50.38 / 49.82 |
| transformer | 20,480 | 19.70 / 19.78 | 30.73 / 27.28 | 37.35 / 38.10 | 41.93 / 42.36 | 42.23 / 42.28 |
| cont_mlp | 204,800 | 21.99 / 22.72 | 38.93 / 35.08 | 36.43 / 39.97 | 48.81 / 48.10 | 47.65 / 49.19 |

(Set B's lstm and transformer are at their DECLARED 20,480 steps; its mlp and cont_mlp at TENX, 204,800.)
**Four times the rows at the same steps moves every family by −3.5 … +3.9, with no consistent sign.** This is the
ladder replication of w20's rows 3–4 finding (16k ≈ 32k ≈ 131k at 20,480 steps).

## 4. Caption (drop-in; `CAPTION.md`, machine form `caption.json`)

See `CAPTION.md`. It names **all three rival pools and their step budgets**: A 32,256 Montero combination-split rows
(⚠ it holds the held-out values) at 5,040 steps; B 131,072 rows at 20,480 / 204,800; C 32,768 store-content rows at
5,120 / 51,200, plus step-matched at 20,480 / 204,800. It also gives the store's content pool (32,768 of a
131,072-row rung pool) and learned_ref_ownpool (32,768 rows, 2,560 steps). Every column header in `TABLES.md` gives
pool AND steps.

## 5. What did NOT run (DECLARED)

1. **Anything on w20's outputs.** By rule, set B was not refitted or re-scored. Its lstm/transformer TENX exists only
   as w20's spot check (rungs 2 and 6, seed 0), and is printed with its seed count.
2. **No store was fitted or scored here.** The K\*=32 store rows are w12b's (`kstar_ladder/*/rows_single_design_K32.json`),
   as in w20. Their Charter-C controls are w12b/w20's: w20 `ladder.<n>.P2_controls` has all four collapsing on every
   K\*=32 ladder store. I did not re-run them.
3. **10× beyond the step-matched budget** (i.e. cont_mlp past 204,800 steps). cont_mlp is not proven converged
   (§3). It is flagged, not chased, because it never becomes the strongest rival.
4. **Wider spoke-A grids.** Every leg selected the grid edge (lr 3e-3, 40 epochs). Per w20's discipline the grid was
   not widened after the fact. TENX and step-matched are the instruments for that edge.
5. **LACE / PoE at 32,768 rows.** This was not in the brief and was not run. w15's 32,256-row rung rows remain the only
   LACE/PoE ladder rows.
6. **Rows 3–4 (TEST) of any kind.** This spoke is ladder-only and free; the scorer refuses other corner sets.

## 6. What would have proved this wrong, and what happened

- **X1 failing (same rows, same steps, a gap beyond the seed spread).** It did not fail at any of the 5 rungs. The
  per-seed gap exists (12/15 CIs exclude 0). It is traced to the PRNG draw alone by an identical-per-corner replay
  of the reference, and it is printed.
- **Set C beating the store, or collapsing the margin (≤ +1.0).** Did not happen. The smallest per-seed margin is
  +3.25, and every CI lower bound is ≥ +1.98.
- **Set C clearly weaker than set B, which would make MF-2 a data defect that inflated the store's margin.** Did not
  happen: C is within −1.2 … +1.5 of B per rung, with mixed sign.
- **The margin carried by an under-trained rival.** The carrying family (mlp) is flat from 51,200 to 204,800 steps.
  The one family that is still gaining (cont_mlp) stays below it.
- **Pool mismatch.** Ruled out mechanically, on 15/15 legs, from the artefacts.
