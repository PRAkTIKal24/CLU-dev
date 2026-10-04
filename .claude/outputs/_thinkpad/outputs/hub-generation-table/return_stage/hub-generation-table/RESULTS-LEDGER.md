# RESULTS LEDGER — `hub/generation-table`

## SCORED RUNS (any number on the held-out sets A / B / C)

| # | date | what | status |
|---|---|---|---|
| — | — | — | ⛔ **NONE. Scored runs spent: 0 of 3.** |

**Why zero, and both reasons are hard:**
1. ⛔ **COMMON §1c** — *"Do not perform your first scored run until a human tells you how many of the three
   remain for your target."* The question was raised in `LOG.md` at 19:10Z, the first log entry of the run,
   and repeated in the final report. **No human answered.** ⛔ I did not proceed on an assumption.
2. ⛔ **The pinned re-encode gate fails on this machine on all three seeds** (`LOG.md` 23:44Z;
   `dev/exp26_reencode.json`). `exp_gen_cell.run_seed:1780` calls
   `_assert_cache_reencodes(...)` with the default `will_extend=True`, **before any arm and before the
   decoder ceiling**, so the scored cell **aborts on this box on every seed having computed no held-out
   number at all**. Seeds 1 and 2 would additionally require 32,199 / 32,241 rows to be encoded here, which
   the gate re-checks at full strength at the point of extension.

⛔ **Nothing was worked around.** No pinned file was edited; `will_extend=False` was never passed from my
code; `SKIP_LOAD` / `SKIP_TEST` were never used. COMMON §0's override covers exactly one named test and
explicitly is not a licence for any other red check.

## THE ONE NUMBER THAT WAS COMPUTED ON `eval_b`, AND WHY IT IS NOT A SCORED RUN
`LOG.md` 19:25Z, **FINDING 1**: the **best-constant bound of 0.5278**. It is arithmetic on the factor labels
of `split.test_rows` — `mean(1/2, 1/2, 1/2, 1/3, 1, 1/3)` over the six factors' realised value counts in the
Montero cell. ⭐ **No image was rendered, no classifier was called, no arm was run, no model touched the
rows.** It is a property of the metric and the split definition, not a measurement through them.

## UNSCORED WORK (all of it on DEV corners cut from inside `split.train_rows`)

| id | what | outcome |
|---|---|---|
| DEV-01 | the shipped read on a DEV corner + controls | `clu` 0.4722 (9 tuples) — **below** the 0.5278 constant floor |
| DEV-02/03 | K-sweep at constant store bytes, 5 → 128 | codebook capacity 0.4884 → 0.7963; the ARM peaks at K=32 |
| DEV-04 | decoder-**diagonal**-weighted codebook | ⛔ null |
| DEV-05 | rivals on DEV; low-rank LOGIT head | `knn_proto` 0.7801; ⛔ low-rank logit head dead |
| DEV-06 | launch-then-snap read | ⛔ `arm ≤ head_alone` at every K and rank |
| DEV-07 | ⭐ isotropic-noise control at matched MSE | ⭐⭐ the quantisation loss is **structure, not magnitude** |
| DEV-08 | ⭐⭐ the shipped store read as vertex / hull / **span** | ⭐⭐ 0.4884 / 0.5509 / **0.8264** |
| DEV-11 | the store as a dictionary, (K, M) grid | beats the atom floor; ⛔ capped by `knn_proto` |
| DEV-12/13/18 | ⭐⭐ is `knn_proto` beatable at all? | ⭐⭐ **yes: +0.1285, 6/6 corners — it is merely ADDITIVE** |
| DEV-14/16/17/20/23 | interaction heads, five parameterisations | best = one-hot additive + 2 one-hot pairs, rank 16 |
| DEV-15 | low-rank of the WHOLE head | ⛔ dead |
| DEV-21 | K = 256/512 full-span | ⛔ **NOT RUN** — disqualified by the §C.2 filter before running |
| DEV-24 | ⭐ the arm through the PRODUCTION objects, **49 corners** | ⭐ **the headline table** |
| DEV-26 | ⛔ the re-encode gate on all three seeds | ⛔ **FAIL / FAIL / FAIL — the blocker** |
| DEV-27 | ⭐ the §C.1 floor against three atom selectors | ⭐ +0.085 vs medoids; ⚠ +0.022 vs k-means++ atoms |

## PRE-REGISTRATION
`PREREG-hub-generation-table-1.md` — written **before** any held-out number, bar and prediction committed,
**not executed**, with the DEV outcome and the blocker appended as dated appendices (⛔ the body untouched).
