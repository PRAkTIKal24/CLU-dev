# PREREG — w20 hub/shapes-data-matched — target `shapes-matched` (budget 3)

**Written before any generation-2 number from this hub.** Amendments are appended and dated,
never edited. Base `base/iter8 = 148f3b7`; worktree `C:/chlu/worktrees/CHLU-w20-shapes-data-matched`;
`chlu.__file__` asserted inside it in every process. Cache **generation 2**, asserted per seed
(`exp_gen_delete_row7.assert_generation2`). Seeds **0, 1, 2** (the seeds with a gen-2 cache AND a
judge; seeds 3–4 lack `classifier_dir`, w12 AMENDMENT 1 §A1.1).

## 0. Why, in one paragraph

The draft's headline rivals (T7: MLP / LSTM / transformer / continuous MLP at the 131,072-param
read-path pin) were fitted on **16,384 rows** for **20 epochs, untuned** — `w12-deletion-baselines`'
declared BUILD/SMOKE scale. The ladder's `@ref` rivals (spoke A) were fitted on **32,256** rows of a
different population. The store is fitted on **131,072** rows. The draft's margin, store K\* = 32
minus each rival, per seed, on rows 3–4, is **+39.7 … +49.1** novel tuples/corner
(`w15-single-design-seeds/T7-3SEEDS.json`; "+39 to +49"). Referee M1 (fairness is parameter-only)
and M2 (two rival sets an order of magnitude apart) both trace to the fit pools.
**Declared before any number (LOG.md 17:26Z):** hub/read's `mlp_180k` (fit on a larger pool) scored
53.5 novel/corner on the same 49 n-way corners where T7's 16k MLP scores 7.6. **A collapse of the
margin is a live, stated possibility, and if it happens it is the result.**

## 1. Table M-131k — every rival data-matched to the store

**Rivals.** `mlp`, `lstm`, `transformer`, `cont_mlp` at the **131,072-parameter read-path pin**
(`w12_deletion_baselines.matched_arm` → `w4_learners.match_width`, measured params printed), the
full-request (non-dropout) variant — rows 3–4 and the rungs ask full six-factor requests.
**Fit pool = the store's EXACT 131,072-row pool** (`hub-factor-wells/run/partitions.npz:pool` for
rows 3–4 — measured identical to `fw.fit_pool(train_rows, 131072, 20260919)`; the rung's own
`partitions.npz:pool` on the ladder).

**Protocol = spoke A's (wave-4) `w4_learners.fit_family`, unchanged in its grids:**
stage 1 = 6 short trials, `LR_GRID (3e-4, 1e-3, 3e-3) × WD_GRID (0, 1e-4)` at `TUNE_EPOCHS = 8`;
stage 2 = `EPOCH_GRID (20, 40)` at full length at the stage-1 winner; selection by held-out MSE to
the cached feature on a fixed 10 % of the pool (`default_rng(20260923 + seed)`), the same 90/10
convention the store's own address book uses. Then:
* **DECLARED** — the selected (lr, wd, epochs) refitted on **all 131,072 rows**. This is the rival
  that carries the margin **if it passes the convergence gate (§3 P1)**.
* **TENX** — the same (lr, wd) at **10× the epochs = 10× the optimiser steps**, all 131,072 rows.
  The convergence **diagnostic column**.
* **The loss curve is printed** for both (per-epoch mean train MSE), and the held-out curve of the
  selection stage per epoch.
* **Ladder rungs:** the (lr, wd, epochs) selected on the rows 3–4 pool for the same seed and family
  are **transferred** (same pool size, same feature space); DECLARED is fitted on every rung and
  seed. Whether TENX runs on every rung is decided by §3 P1's outcome on rows 3–4 — see §4.

**Store.** K\* = 32 on rows 3–4: **the T7 objects are REUSED, not refitted** (`arm.pkl` of
w12 `run1_seed0_K32`, w12b `run3_seed{1,2}_K32`); integrity check = re-scoring must reproduce
T7's per-corner rows exactly. K = 5 on rows 3–4: seed 0 reused (w12b `run2_seed0_K5`); seeds 1, 2
**fitted here** on the same pool (new TEST numbers — part of run 1/3). Ladder: K\* = 32 per
rung/seed reused from `w12b-nr6-and-kstar/kstar_ladder` (per-corner rows on disk), K = 5 = the
ladder's own existing K = 5 arms "as already fitted".

## 2. Table M-16k — the store re-fitted at the rivals' data size

**The store** (the crossover object unchanged: `CovContentStore`, `ward_balanced`, A = min(64,
min d_k − 1), `TensorAddress` at `READ_PINS`) **at K\* = 32 and K = 5**, re-fitted on the
**first 16,384 rows** of its pool — which ARE the w12 comparators' fit rows (measured) — and, as
the third column, on the **first 32,256 rows**.
**2.3 Why the 32,256 column is a nested prefix, not spoke A's literal pool:** spoke A's pool is the
`montero_combination` population and contains BOTH value-holdout held-out values
(`RIVALS-READY.json`: 3,157 rows at object_hue 0.5, 7,972 at shape 0.0). A store fitted on it
and scored on rows 3–4 would be scored on content it held. The size-matched column on the store's
own population is the comparison that isolates **data size**; w15-lace-poe-rows used the same
"32,256 rows of the rung's own pool" convention. Recorded; not a choice made after a number.
**Rivals in M-16k:** the EXISTING 16,384-row comparators (T7's, per-corner rows on disk) on rows
3–4; the EXISTING ladder rivals (spoke A `@ref`, 32,256 Montero rows; `learned_1x`, 32,768 rung
rows) on the rungs. ⛔ Nothing refitted on the rival side of M-16k.

## 3. Pre-declared gates and bars (margin = `DECORATION_MARGIN_NOVELTY` = 1.0 novel tuple/corner)

* **P1 — convergence gate (per family, per seed, per corner set).** DECLARED carries a margin only
  if |TENX − DECLARED| ≤ 1.0 novel/corner (mean over the corner set). If TENX exceeds DECLARED by
  more than 1.0 the DECLARED rival is **UNDER-TRAINED**, is flagged, and **the TENX rival carries
  the margin instead**. In every case the margin is also printed against **max(DECLARED, TENX)** —
  the rival at its best, the conservative direction.
* **P2 — control gate.** All four controls (`landscape_removed`, `landscape_shuffled`,
  `landscape_shuffled_across_wells`, `random_atoms`) must score ≤ 1.0 novel/corner on every seed
  for any store row to be claimed. `landscape_shuffled_across_wells` is a **DERANGEMENT**
  (`w20_lib.derangement`: rejection-sampled, 0 fixed points asserted) and its permutation is
  **printed on the row**, beside the historical T7 permutation's fixed-point count.
* **P3 — THE MARGIN.** Data-matched margin = store K\* = 32 minus the **strongest** data-matched
  rival (by P1's carried score), paired over corners (`fw.paired_estimand`, bootstrap CI), per
  seed and mean over seeds 0/1/2. Pre-declared reading against the draft's +39 … +49:
  * **HOLDS** — margin ≥ +39.0 on the seed mean;
  * **SHRINKS** — +1.0 < margin < +39.0: the win survives data matching at a smaller size, and the
    size is the number the paper must carry;
  * **COLLAPSES** — margin ≤ +1.0 (or negative): **this is the result**, reported plainly — it is
    what M1 predicted. ⛔ No re-tuning, no re-selection, no change of rival, pool, K or metric to
    rescue it.
  Also printed per family, per row (3 and 4 separately), and at K = 5.
* **P4 — M-16k reading.** Store@16,384 and store@32,256 minus the existing rivals, same estimand.
  If the store's margin over the 16k rivals at 16,384 rows is ≥ its margin at 131,072, "data size
  explains the gap" is refuted for the store side; otherwise the drop is printed.
* Every table: **both fit-pool sizes in the caption, a seeds column, the four controls, the table
  FLOOR (prototype table; never a rival) and the per-corner CEILING** (distinct requested tuples
  the split never holds). LACE/PoE rows from `w15-lace-poe-rows/GENTABLE-KSTAR.json` added where
  they exist, with their scoping verbatim. ⛔ No inflated-rival row.

## 4. Val-first, and what the val stage decides

VAL is free and precedes every scored run: on the val condition (`run-val` pool, 24 val corners),
seed 0, all four families (protocol + DECLARED + TENX) and the store at 16,384 / 32,256 rows,
K = 32 and 5. The val stage checks that the pipeline runs end to end, that P2 collapses and that
the derangement prints; it **selects nothing** that the test tables use (the rivals' selection is
made on each test pool's own 10 % split by the fixed protocol). Ladder TENX scope: if on rows 3–4
P1 passes for a family at every seed, TENX on the rungs runs at seed 0 only for that family (a
spot check), declared here; if P1 fails anywhere on rows 3–4, the rungs' carried rival for that
family is the TENX fit on every seed.

## 5. Spending

* **run 1/3** — Table M-131k on rows 3–4 TEST: the four data-matched rivals (DECLARED + TENX),
  the four controls (derangement) re-scored on the reused T7 stores, K = 5 seeds 1–2 fitted and
  scored, seeds 0/1/2 (replications of this PREREG, free).
* **run 2/3** — Table M-16k on rows 3–4 TEST: store at 16,384 and 32,256 rows, K\* = 32 and 5,
  seeds 0/1/2.
* run 3/3 — **banked** unless an amendment dated before its use says otherwise.
* The ladder is a replication on the rung conditions (w12b precedent: free, no ledger row).
* ⛔ A ledger row is written in `/c/chlu/SCORED-LEDGER.md` BEFORE each scored run.

## 6. What would prove this hub wrong

* The data-matched rivals fail P1 even at TENX (still improving by > 1.0 between DECLARED and
  TENX) — then no rival here is at budget and **no margin is claimed**; reported as such.
* A re-scored T7 store does not reproduce T7's per-corner rows — every reuse is void.
* The across-wells derangement does not collapse where the historical permutation did — the T7
  control was weaker than printed and the store row is re-read under P2.

---
# AMENDMENT 1 — appended 2026-09-23 (before any TEST number), never an edit

**A1.1 A data-vs-steps diagnostic column (MLP only), added to run 1/3.** Moving from 16,384 rows ×
20 epochs (1,280 steps, T7's comparators) to 131,072 rows × 40 epochs (20,480 steps) changes data
AND optimiser steps together. To say which one the margin change is owed to, run 1/3 also scores
`mlp` fitted on the **first 16,384 rows** (the T7 comparators' exact rows) with the rows-3–4
selected (lr, wd) transferred and **8× the selected epochs = the same 20,480 steps** as the 131k
DECLARED fit (`w20_fit_rival.py --npool 16384 --hp-from test --epochs-mult 8 --suffix 16k_stepmatched`).
It is a DIAGNOSTIC column, never a rival that carries the margin, and it does not enter P3.
**A1.2 Val observation that motivates nothing in the tables:** all val and test selections so far
land on lr = 3e-3 and epochs = 40 — the edge of both spoke-A grids. The grids are NOT widened (that
would be re-tuning the rival after a number); P1's TENX column is the pre-declared instrument for
exactly this question.

---
# AMENDMENT 2 — appended 2026-09-23 ~19:20Z (before any TEST number), never an edit

**A2.1 An existing data-matched, read-path-matched rival is REUSED as a column, not refitted.**
`learned_ref_ownpool` (w12 hub/composition-ladder AMENDMENT-1 l.29; printed in
w13-seeds-and-tables/TABLES.md l.703–710): `mlp` at **130,514** params (the 131,072 read-path pin),
fitted on the **first 32,768 rows** of each leg's own pool — the rows the ladder's K = 5 store content
is fitted from (`n_store_train`) — `w4_learners.train`, 20 epochs, lr 1e-3, no tuning = **2,560
steps**; 3 seeds on every rung AND on the `shipped` leg. Measured here: the `shipped` leg's pool IS
the rows 3–4 test pool (`np.array_equal` → True), and the rung pools are identical to the K\* ladder
pools. So `learned_ref_ownpool` enters **both tables** as a column: rows 3–4 (from
`w12-composition-ladder/runs/shipped_seed{s}/rows_learned_ref_ownpool.json`) and every rung. It is
data-matched to **32,768** rows, NOT to 131,072, and every caption says so. `learned_1x` (the ladder's
arm-total-matched MLP, same 32,768 rows) is carried the same way on the rungs.
**A2.2 The M-16k third column moves from 32,256 to 32,768 rows** (the nested prefix of the store's
own pool), so that it is one-to-one with `learned_ref_ownpool` and `learned_1x` — the programme's
existing data-matched rivals — rather than with spoke A's Montero pool (which cannot be used literally,
§2.3). The 32,256-row val stores and one 32,256 test store fitted before this amendment are recorded
and NOT used in any table.
**A2.3 A reconciliation diagnostic (MLP only), added to run 1/3 beside A1.1:** `mlp` on the SAME
32,768 rows as `learned_ref_ownpool`, with the rows 3–4 selected (lr, wd) and **4× the selected epochs
= 20,480 steps**, the same as the 131k DECLARED fit (`--npool 32768 --hp-from test --epochs-mult 4
--suffix 32k_stepmatched`). With A1.1 (16,384 rows, 20,480 steps) it splits any difference between
`learned_ref_ownpool` and the 131k DECLARED rival into a DATA part and a SCHEDULE part. DIAGNOSTIC:
never enters P3.
**A2.4 No contradiction is permitted silently:** if the 131k data-matched rival scores far above
`learned_ref_ownpool`, the tables print both, with rows and steps side by side, and FINDINGS says
which of data or schedule accounts for it (A1.1/A2.3).

---
# AMENDMENT 3 — appended 2026-09-23 (before any TEST number): the Head's AMENDMENT-1, folded in

**A3.1 LACE and PoE re-fit on the store's pool.** `scripts/w20_lace_poe.py` = w15's
`w15_lace_poe_ladder.py` verbatim except the fit rows: the first **131,072** rows of the same pool,
stratified into **130,560 fit + 512 val** (w15: 32,256 + 512). Their OWN budgets unchanged (LACE:
36-trial selection, 50-epoch classifier, `ode` sampler; PoE: 6 experts, 20 epochs, 36-trial SGLD
grid; selection by the judge's val aggregate, the never-weaken direction). Rows 3–4 (`shipped`, TEST
→ run 1/3) and every rung (free), seeds 0/1/2, printed BESIDE w15's 32k rows, never instead.
Scope VERBATIM: **LACE/PoE as instantiated in this feature space at their own declared budgets —
not a refutation of LACE in general** (w15 FINDINGS, N360). Val first: `--rung val`, seed 0.
**A3.2 Matched deletion = run 3/3.** For each scope — `value` (red, 11,044 rows), `conjunction_live`
(red ∧ sphere, 5,511), `item` (1 %, 1,311, `default_rng(20260921 + seed)`) — every rival family is
RETRAINED WITHOUT exactly the rows the store downdates, from the same 131,072-row pool
(`w20_fit_rival.py --delete-scope`), at the rows 3–4 selected (lr, wd, epochs) — the DECLARED
schedule, or the TENX schedule for a family P1 flags under-trained (then declared here: the TENX
retrain is run for that family if wall clock allows, else the flag travels with the row). Cost line
per arm: seconds, data rows read, bytes rewritten, gradient row-visits. The store's side is REUSED:
T7's deletion artefacts (w12 seed 0, w15 seeds 1–2) — same pool, same masks, same RNG. Scored on
each scope's survivor corners, paired per corner. w15's 16k-rival deletion rows → APPENDIX only.
**A3.3 Spend:** run 1/3 = M-131k rows 3–4 (+ LACE/PoE@131k rows 3–4); run 2/3 = M-16k; run 3/3 =
matched deletion. Anything not covered is DECLARED NOT-RUN with its exact rows named.

---
# AMENDMENT 4 — appended 2026-09-23 (before any deletion number; run 3/3 NOT spent): the Head's AMENDMENT-2

**A4.1 Deletion = three operations** (store downdate; ADDRESS-BOOK statistics downdate + book refit
from them; hole). **Four measurements per row** (value red / red ∧ sphere / 1 % items; 131,072 pool;
seeds 0/1/2), all FREE replications on the T7 objects (`scripts/w20_delete_store.py`):
(a) store == refit-without (RESULT); (b) book statistics == refit-without, b1 at the statistic, b2 at
the book and at the rendered features (RESULT); (c) hole refusal on the deleted corners' requests +
false refusal on the survivors' (DISPLAY, never a claim; item scope = DECLARED N/A, no hole exists);
(d) survivor-corner composition after vs the T7 pre-deletion rows, and vs the refit-without pipeline
(RESULT). Bars: (a) max|mode dev| ≤ 1e-5 and ranks agree; (b1) exactly 0; (b2) rendered-feature
relative deviation ≤ 1e-3; (d) |paired after − before| ≤ 1.0 novel/corner (the programme's margin)
and |downdate − refit-without| ≤ 1.0. A failed bar is reported as the result.
**A4.2 Declared before any number:** the ALS address book has no compact exact statistic on this
data — its statistic is the per-tuple first moment of its 117,964 fit rows (one row per tuple), i.e.
the fit data itself. (b) therefore demonstrates exactness at the cost of a statistic the size of the
data; the bytes are printed on the row and FINDINGS says so. "0 data rows read" is then true of the
pool but NOT a size claim.
**A4.3 RETIRED:** "c still composes after forgetting c" — not computed, not printed.
**A4.4 Run 3/3** remains the matched RIVAL deletion (retrain-without on the same rows, scored on the
survivor corners, cost printed), now read beside (a)–(d).

---
# AMENDMENT 5 — appended 2026-09-24 (after cont_mlp's seed-0 P1 reading; no margin number changes because of it)

**A5.1** cont_mlp FAILED P1 at seed 0 (DECLARED 17.917 → TENX 28.532 novel/corner). Per P1 the TENX fit
carries cont_mlp. Because §6 names "rivals still improving at TENX" as a falsifier, a **30× diagnostic**
(`--epochs-mult 30`, 1,200 epochs, same lr/wd) is fitted for cont_mlp at seed 0 and scored on rows 3–4
(run 1/3 replication). Rule, fixed now: if 30× exceeds 10× by > 1.0 novel/corner, cont_mlp is reported as
**NOT CONVERGED even at 10×** and the margin is ALSO printed against its 30× score (the rival at its best).
It never lowers a rival: the carried score is max over the fits run.
**A5.2** cont_mlp's matched-deletion retrain-without runs at the DECLARED schedule (10× × 9 fits does not fit
the wall clock); its deletion rows carry the P1 flag.
**A5.3 (appended 2026-09-24, before any 30× number exists):** cont_mlp failed P1 on ALL THREE seeds, so the
30× diagnostic runs at seeds 1 and 2 as well (same rule as A5.1). The P3 headline is printed BOTH against the
pre-declared carried rival (P1: max over DECLARED/TENX) AND against max over every fit run incl. 30×.
**A5.4 (appended 2026-09-24 ~01:25Z, before any MLP 30× number):** the MLP's TENX scores are arriving above its
DECLARED scores by > 1.0 (P1 failure for the strongest family). The 30× diagnostic (A5.1 rule) therefore runs for
`mlp` at seeds 0/1/2 too. The headline margin is printed against max over every MLP/cont_mlp fit that ran.
