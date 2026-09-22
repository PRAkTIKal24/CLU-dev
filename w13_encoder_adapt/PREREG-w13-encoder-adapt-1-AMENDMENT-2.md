# AMENDMENT 2 to PREREG-w13-encoder-adapt-1.md — the metric, and the degeneracy rule

**Appended 2026-09-21, after the FREE `val` floor check on seed 0 / arm `frozen` and
BEFORE any `test`-condition number existed and before any row was written to
`/c/chlu/SCORED-LEDGER.md`.** Exact time in
`PREREG-w13-encoder-adapt-1-AMENDMENT-2.timestamp.txt`.
⛔ **No bar is moved.** §4's bar — *the margin must GROW from `frozen` to an adapted arm
on ≥ 2 of 3 seeds* — is unchanged word for word, as are §5, §7 and §8.

---

## What the free val leg measured

`valscore/val_seed0_frozen_K32/score.json`, seed 0, arm `frozen`, condition **val**
(the free selection condition, charter A.2), K\* = 32, A = 63, wells 64…64,
rank/width 0.9375 … 0.9844, four-part ledger **2,148,858 B = 4.0986×** the 524,288 B
reference, address book 1,065,544 B fitted in 2,292 s. 24 val n-way corners, 1,728 requests.

**`distinct_valid_novel_tuples` came back EXACTLY 0.000 for every row** — the single
design, all four charter-C controls, and all five learned baselines alike.

⭐ **That is NOT a floor, and the same artefact proves it.** On the identical rows:

| row | distinct_valid_tuples / corner | judge accuracy | valid fraction |
|---|---|---|---|
| **`cov_tensor` (the single design)** | **38.500** (sd 8.193) | **0.5347** | 0.5347 |
| `mlp_180k` (179k) | 36.583 | 0.5081 | 0.5081 |
| `transformer` | 30.000 | 0.4167 | 0.4167 |
| `lstm` | 27.958 | 0.3883 | 0.3883 |
| `mlp` (119k) | 25.625 | 0.3559 | 0.3559 |
| `cont_mlp` | 24.083 | 0.3345 | 0.3345 |
| ⭐ `landscape_removed` | **0.000** | **0.0000** | 0.0000 |
| ⭐ `landscape_shuffled` | **0.000** | **0.0000** | 0.0000 |
| ⭐ `landscape_shuffled_across_wells` | **0.000** | **0.0000** | 0.0000 |
| ⭐ `random_atoms` | **0.000** | **0.0000** | 0.0000 |

The renderer reads, the judge reads, and **all four charter-C controls collapse to exactly
0.0000** — the gate the whole design hangs on. The zero in the novel column is a property
of the **val split**: a val corner's factor tuples are inside `train_rows`, so
`seen_tuple_mask` marks every one of them seen and nothing can be novel, for us or for a
rival. On `test` the n-way and pair corners are held out and novelty is meaningful.

⭐ **Therefore the comparison CAN be made at this compute**, and this hub does not report
"the comparison cannot be made". The renderer's step count is not cut further.

## The rule, declared now and fixed

* **PRIMARY metric: `distinct_valid_novel_tuples` per corner** — the programme's own
  rows-3/4 estimand, paired by corner, scored through `fw.paired_estimand`. Unchanged.
* **SECONDARY metric: `distinct_valid_tuples` per corner**, carried on every row.
* ⛔ **DEGENERACY RULE.** The primary is declared DEGENERATE on a leg **iff it reads
  exactly 0 for the single design AND for every learned baseline AND for every control**
  — i.e. it separates nothing whatsoever. Only then does the secondary decide, and the
  artefact must carry `primary_degenerate: true` and say so on the row.
  ⛔ The primary is **never** swapped for whichever number happens to be larger, and a
  leg where the primary separates the arms is decided by the primary, full stop.
* Every row also carries the judge accuracy, the valid fraction and the four control
  values, so a floor can never be reported as a result in either direction.

## Free val numbers are NOT a claim

⛔ The margins in the table above are `val`, seed 0, one arm. They are recorded because
they are what licensed the test leg, not as a result. The reported margin is the
test-condition one, on 3 seeds, all or none.

⚠ For the record, since it will be asked: on `val`, seed 0, `frozen`, the paired
`distinct_valid_tuples` margin of the single design over its best learned baseline
(`mlp_180k`) is **+1.917 per corner** (n = 24, sd 4.452) — ⭐ **POSITIVE**, where wave 12's
byte-matched comparison at K = 5 was negative. ⛔ Not a result: different condition,
different metric, one seed, one arm, and the baselines here are matched to
118,784 / 179,968 parameters, **not** to this object's own 2,148,858 B four-part total.
A 1×-matched rival would be STRONGER, so this comparison **flatters** the single design,
and the scored rows say so.
