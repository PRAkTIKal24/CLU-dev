# gen-val-split-fix — PREREG (filed BEFORE the fixed split was built and BEFORE any re-score)

Filed: 2026-09-14, by `experiment-engineer`, on branch `agent/experiment-engineer/gen-val-split-fix`
(worktree `/Users/user/Desktop/CHLU-gen-valfix`), base `main @ 5a3c32c`.

⛔ Nothing below was measured before it was written. The only facts used are the *inherited*
ones from `gen-atom-mean-control.md` §7 (broken-split realised values, 5-of-6 half-disagreement)
and from reading `chlu/data/shapes3d.py` (factor order + cardinalities).

---

## 1. DIAL DECLARATION (echoed)
- **Dial:** none — REPAIR + RE-SCORE. No new arm, no mechanism, no verdict.
- **Laundering control:** every rival (`knn`/`knn_proto`, LACE incl. the whitening-exactness
  fallback) is re-selected by the *same* procedure on the *same* fixed split as ours.
- **Falsifies:** nothing of ours. It may move published numbers, including refusal's.

## 2. The defect (inherited, not introduced)
`select_baseline_rows` (and its two verbatim copies in `exp_gen_cell.py` and
`exp_gen_atom_mean.py`) take `fit = pool[:n_fit]`, `val = pool[n_fit:n_fit+n_val]` where
`pool = np.sort(rng.permutation(train_rows)[:32768])`. Shapes3D's row index is the
lexicographic index over `(floor_hue, wall_hue, object_hue, scale, shape, orientation)` with
cardinalities `(10, 10, 10, 8, 4, 15)` and `orientation` fastest, so a *sorted tail* of 512
rows is a contiguous corner of the factor grid.

## 3. THE STRATIFICATION RULE (declared before it was coded)
**Proportional marginal stratification with a randomised greedy quota filter**, deterministic
given `(seed, n_val)`:

1. `pool = select_store_rows(...)["store_train"]` — **unchanged**, so the store, the atoms and
   the held-out disjointness are untouched.
2. Factor value-indices `L[i, f]` for every pool row (via the grid bijection, no new data read).
3. Per-factor **proportional quota** `q[f][v] = ceil(n_val * count_pool[f][v] / n_pool)`
   (+ `quota_slack`). Target: `val`'s marginals equal the *pool's* marginals, which is what
   "exchangeable with `fit`" means here.
4. Walk a seeded random permutation of the pool; accept a row into `val` iff for **every** one
   of the six factors its value bucket is below quota. Relax all quotas by +1 and re-walk the
   unaccepted rows if `val` is not full (record `n_relaxations`).
5. `val = sort(accepted)`, `fit = sort(pool \ val)[:n_fit]`.
6. Key derivation: `np.random.default_rng` seeded from `(seed, salt)` — ⛔ not a lengthened
   `jax.random.split` (prefix-stability trap).

Backward compatibility: `mode="head_tail"` reproduces the broken split bit-for-bit; it must be
selected explicitly and it shouts in the log.

## 4. PREDICTED BALANCE (acceptance item 2) — production scale, `n_fit=32256`, `n_val=512`
Derived arithmetically from the lexicographic index, **before running**:

| factor | card | broken `val` `n_values_realised` (predicted) | fixed `val` (predicted) | fixed per-value count (predicted) |
|---|---|---|---|---|
| floor_hue | 10 | **1** | **10** | 51–52 |
| wall_hue | 10 | **2** | **10** | 51–52 |
| object_hue | 10 | 10 | **10** | 51–52 |
| scale | 8 | 8 | **8** | 64 |
| shape | 4 | 4 | **4** | 128 |
| orientation | 15 | 15 | **15** | 34–35 |

Derivation of the broken column: the top 512 of 32,768 sorted rows span ≈ `512/32768 × 479,928
≈ 7,500` consecutive row indices. Block sizes are floor 48,000 / wall 4,800 / object 480 /
scale 60 / shape 15 / orientation 1 ⇒ floor spans 1 block, wall spans 2, everything else ≥ 10
blocks. **This matches `gen-atom-mean-control` §7's measured 1 / 10 / 15 / 8 / 4 / 2 exactly,
which is the positive control on the derivation.**

Predicted `val` feature variance: **0.20 – 0.23** (fit's is 0.2229; broken `val`'s is 0.0924).
I predict the fixed `val` variance lands within **±10 %** of `fit`'s 0.2229.

⭐ **Explicitly checked, not assumed: `scale` and `orientation`.** They were *already* fully
realised in the broken split, so the falsifier here is not "n_values_realised < card" but
**per-value count imbalance**: I predict max/min per-value count ratio ≤ **1.15** on every one
of the six factors in the fixed `val` (broken `val`'s `orientation` and `scale` are predicted
to be roughly flat too, so this is not where the repair shows).

## 5. PREDICTED MOVEMENT (acceptance item 4) — the substantive prereg
`fit` also changes (by exactly the 512 rows swapped with `val`, i.e. 1.56 % of the pool), so
*everything* fitted on `fit` (kNN k-means store, z-score normaliser, LACE whitening + LACE
classifier, `FactorPrototypes`) changes. Therefore:

- **P1. Nothing is bit-identical.** I predict **0** of the published rival/CLU numbers reproduce
  to machine precision, and that the `knn_proto` bit-identity guard
  (`require_knn_proto_bit_identical=True`, refs `0.8762626…/0.8919598…/0.8969849…`) **FIRES**
  and must be re-based onto the fixed split. ⛔ If it does *not* fire, my split did not take.
- **P2. The moves are small relative to seed spread.** `gen-atom-mean-control` §7 measured
  best-to-second trial gaps of 0.0007–0.0221 against a 0.09–0.15 trial spread ⇒ the selection
  sits on a plateau. I predict the **mean absolute move of any published headline is < 0.03**,
  and **< 1 seed-sd** for at least 2/3 of the affected headlines.
- **P3. The selected configuration moves in roughly half the cells.** Given the 5-of-6
  half-disagreement, I predict the argmax changes in **2–5 of the 6** (arm, seed) kNN/LACE
  selection cells and in **2–5 of 6** atom-mean (seed, budget) cells. ⛔ I predict it does
  **not** change in all 6 and does **not** change in 0 of 6.
- **P4. `val`-measured diagnostics move a lot, in the direction of getting HARDER.** The broken
  `val` is a low-diversity corner; on a balanced `val` I predict every `val_aggregate` drops
  (predicted drop **0.02 – 0.15**) and the atom-mean relative read error measured on `val`
  rises from ≈0.94-on-a-2.4×-deflated-variance toward the `A` value ≈0.41 — i.e. the `val`
  read-error column and the `A` column **converge to within ±0.10** of each other.
- **P5. No rival gets weaker beyond noise.** I predict `knn_proto`'s in-corner balanced
  accuracy stays within **±0.02** of 0.8763 / 0.8920 / 0.8970 and that its 3-seed mean does not
  fall by more than **0.01**. ⛔ If a rival's 3-seed mean falls by more than one seed-sd, this
  spoke **STOPS and reports** rather than proceeding (acceptance item 5).
- **P6. LACE's R27.3 whitening-exactness gate keeps firing.** The gate is a property of the
  *whitening* of `fit` (71 dead post-ReLU coordinates ⇒ irreducible 0.0347 floor), which a
  1.56 % row swap cannot change. I predict the per-shrinkage `mean_abs_diag_minus_one` values
  move by **< 0.01** and the `prior_identity_near_exact` booleans are **unchanged for all three
  shrinkages**, so whether the fallback fires is decided by *which shrinkage is selected*, not
  by the whitening itself.
- **P7. The structural guarantees are untouched.** `no_leak` = 0 leaked of 72; `store_train`,
  the store plan, the well sizes, `eval_a`, `eval_b`, `MS`, the decoder/encoder/probe/external
  classifier hashes, and the byte ledgers are **bit-identical**. ⛔ If any of these move, the
  fix reached somewhere it had no business reaching.

## 6. What would prove this spoke wrong
1. Our split fixed, the rivals' not. (Mitigation: all three families share the one code path;
   the balance assertion runs inside every one of them.)
2. Balanced on the hues but not on `scale`/`orientation` (⇒ §4's max/min ratio check, on all
   six factors, is an acceptance item, not a diagnostic).
3. Movement reported without a matching UNCHANGED column.
