# SELECTION RULE — w12 hub/single-design-crossover
**Written 2026-09-21T11:10:24Z, BEFORE the val sweep was executed.**
Charter §A.2 (selection on `val` only) and §A.3 (amendments appended, never edited).

Timestamp of this file's creation (UTC): see `SELECTION-RULE.timestamp.txt` beside it.
No val composition number for any K existed on disk in
`/c/chlu/outputs/w12-single-design-crossover/` at the moment this file was written.

---

## THE RULE (single, binding, not movable)

> **K\* = the finest partition (largest K) whose val composition is within 0.02 of the
> val composition at K = 5.**

Operationally, with `v(K)` = the mean over the 24 val corners of the per-corner `valid`
rate produced by `scripts/w8_score_corners.py`'s code path (the existing 24-corner val
read), averaged over the seeds actually run:

```
K* = max { K in SWEEP : v(5) - mean_seeds v(K) <= 0.02 }
```

`SWEEP = {5, 8, 10, 12, 16, 20, 24, 28, 32}`.
The comparison is one-sided: a K that scores ABOVE K=5 satisfies the rule trivially.
Ties in K cannot occur (K is unique per row).

## THE CROSSOVER (reported beside, not a selection quantity)

> **K_cross = the smallest K in SWEEP for which `v(5) - mean_seeds v(K) > 0.10`.**

If no K in SWEEP exceeds 0.10, that is reported as "no crossover inside the swept range"
and ⛔ the range is NOT extended to manufacture one.

## WHAT HAPPENS IF THE RULE SELECTS NOTHING

If no K > 5 satisfies the rule, then **K\* = 5** and the wave reports that the partition
cannot be made finer without losing composition. ⛔ The rule is NOT relaxed, the
tolerance is NOT moved, and the sweep is NOT re-run with a different metric.
(Charter A.3 — "no K satisfies the declared rule ⇒ report that; do not move the rule.")

## THE GATE AFTER SELECTION — the controls in §2 of the task file

Even once K\* is selected, ⛔ **no scored row may be claimed at K\*** unless, at K\*,
all three controls collapse:
`landscape_shuffled`, `random_atoms`, `landscape_removed`.
If `landscape_shuffled` or `random_atoms` does not collapse at K\*, the store is
**DECORATIVE** there, and that is the wave's reported result.

## FIXED BEFORE THE SWEEP — the sweep's own configuration

- Partition: `ward_balanced_labels` (label-free correlation-dendrogram cut,
  `scripts/w8_matched_bytes_recut.py:ward_balanced_labels`, verbatim from
  `exp_gen_factor_wells.py:594`). Balanced by construction.
- Rank: **A = min(64, min_k d_k - 1)** — "rank 64 per well, capped STRICTLY below well
  width". rank/width is printed on EVERY row.
- Read: `TensorAddress` product address, `R.READ_PINS` rank/ridge/sweeps, 90/10 split,
  `pinned_basis` (the store's own eigen-ordered direction rows), exactly as
  `scripts/w8_read_matched_bytes.py`.
- Metric: the 24-corner val read, `valid` mean, `exp_w8_downdate.stage_smoke`'s path.
- Seeds: 0, 1, 2 (the three with a feature cache on this box). If
  `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json` appears before the SCORED
  leg starts, the SCORED leg uses 5 seeds and says so; the val sweep stays at 3 and says so.
- Exactness: `downdate(delete red) == refit-without-red`, max abs mode deviation and
  max subspace-projector distance, per K per seed.

## THE BYTE PIN

⛔ The 524,288 B pin is NOT primary this wave. The single design is **1x its own total
four-part ledger**; learned baselines are matched to that total. The 524,288 B matching
is printed as a **reference column only**. The four-part ledger
(atoms · covariance/count · address book · coupling/hole table) is printed on every row
and ⛔ never netted.
