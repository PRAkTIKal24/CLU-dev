# AMENDMENT 4 — a corner-budget cut that was CONSIDERED and then NOT MADE

**Stamped 2026-09-21T11:54:52Z** (`AMENDMENT-4.timestamp.txt`). This file exists because that
timestamp exists: the stamp was taken while an amendment was being drafted, and a dangling
stamp with no amendment beside it would be a hole in this hub's record. The amendment was
**not made**, and the reason is on the record here rather than left to inference.

## What was about to be amended

At 11:54Z the first arm timings looked like this — `single_design_K5`, the same arm, the same
code:

| rung | requests | seconds |
|---|---|---|
| k = 2 | 2 160 | 762.6 |
| k = 3 | 8 640 | 918.4 |

Extrapolated, the 18-leg matrix was ~20 hours, which would not fit. The draft amendment would
have cut the scored corner budget at k ≥ 3 from the harness's full 120 corners to the
harness's **pre-declared bar of 49** (`LADDER-READY.json:bar.corners_per_k = 49`), which has
the incidental virtue that k = 6's first 49 corners **are** the shipped row 4, corner for
corner.

## Why it was not made

⛔ The cut was never needed, because the timings were not what they looked like. Measured
before cutting anything (`logs/fastjudge_check.log`, `logs/batch_check.log`):

- `exp_gen_read.arm_features` — the `TensorAddress` span read — costs **< 0.05 s per 720
  rows**. It is not the bottleneck.
- the shipped `Judge.read_back` costs **16.1 s / 720 rows** → ~194 s for an 8 640-row arm.
  Not 918 s either.
- hoisting the per-call `eqx.filter_jit` out of `chlu/eval/gen_probe.py`'s function bodies and
  answering all corners in one batched call — **both proven bit-identical** — brought an arm
  that reuses its address book down to **19–21 s** (`landscape_shuffled`, `logs/2_seed*.log`).

⇒ The 762.6 s / 784.9 s / 918.4 s figures were dominated by **CPU contention** — 3–4 of this
hub's processes (one at `OMP_NUM_THREADS=3`) against `nproc = 3`, beside three other agents'
jobs. Once that was fixed the full 120-corner budget fits, so **every rung is scored on all of
the harness's corners** and no corner budget was ever chosen after seeing a score.

⭐ **Nothing was cut on the strength of a guess about where the time went.** The first
diagnosis in `LOG.md` (that the jit rebuild explained the arm timings) was wrong, is corrected
there, and the correction is what made this amendment unnecessary.
