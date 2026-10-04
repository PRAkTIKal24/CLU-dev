# PREREG — `plan-pass-vectorise` (the lane-parallel controller)

**Filed 2026-08-01, BEFORE running `bench_lane_parallel.py`.** Protocol §5 pre-registration rule:
the acceptance criterion is a **measured ratio** ("≥ 4× cut on the Python term at batch 8"), so the
number is committed to here first, with its derivation.

Code state at filing: worktree `CHLU-lane-parallel-controller`, branch `lane-parallel-controller`,
lane-parallel path implemented and **already asserted decision-identical** by
`tests/test_lane_parallel_controller.py` (3 seeds, batch 3, 7/7 plan fields exact, 49 tests green).
No timing has been taken yet on any configuration.

## Machine
Apple M1 Pro, `os.cpu_count() = 8` (8 physical = **6 performance + 2 efficiency**, logical 8).
⚠ This matters: an 8-way split has **stragglers on the efficiency cores**, so ideal 8× is not
reachable here even with zero IPC. CSF3's GPU nodes are homogeneous-core and should do better.

## The model I am predicting from
Serial Python term per optimisation step:

    T_ser = n_layers x batch x n_chunks x t_lane_chunk           (probe §8.1)
    t_lane_chunk = 1.7–2.4 ms, ~independent of capacity (K = 8/32/64 all in band)
    pilot: 12 x 8 x 16 = 1536 lane-chunks  ⇒  T_ser = 2.6–3.7 s/step

Pooled over the batch axis with W workers (layers stay sequential):

    T_par = n_layers x [ ceil(batch/W) x n_chunks x t_lane_chunk x s  +  c ]

with `s` = straggler/contention inflation (E-cores + the parent process competing) and `c` = the
per-layer round-trip cost (pickle 8 x (z_lane, scfg), unpickle 8 x plan-dict; the biggest payload is
`sites`, `n_chunks x K x dim` float32 = 16 x 32 x 12 x 4 B = 24 KB per lane).

Assumptions I am committing to: `s ≈ 1.5` at W = 8 on this heterogeneous machine (`≈ 1.15` at W = 4,
which fits on performance cores alone), `c ≈ 3 ms`.

## Predictions (point, band)

| # | quantity | predicted | band | how derived |
|---|---|---|---|---|
| **P1** | ⭐ **speedup of the Python controller term, batch 8, W = 8**, pilot store geometry | **5.0×** | **[4.0, 7.0]** | 8 / s with s ≈ 1.5, minus ~5 % for `c` |
| **P2** | speedup at batch 8, **W = 4** | **3.2×** | [2.5, 4.0] | 4 / 1.15 with a larger relative `c` |
| **P3** | speedup at batch 8, **W = 2** | **1.8×** | [1.5, 2.0] | 2 / 1.05 |
| **P4** | per-layer pool round-trip overhead `c` (8 lanes) | **3 ms** | [1, 10] ms | 8 x (24 KB + config) each way through a pipe, ~1 GB/s effective |
| **P5** | decision identity at batch 8, 3 seeds, all 7 plan fields | **exact equality** | — | same code, another process; only a backend/rounding difference could break it, and the workers are pinned to CPU as the parent is here |
| **P6** | pilot Python term after the fix | **0.52–0.74 s/step** | [0.4, 0.95] | (2.6–3.7 s) / P1 |
| **P7** | GPU-idle fraction, **1 s A100 bracket** | **34–43 %** | [29, 49] % | P6 / (P6 + 1 s) |
| **P8** | GPU-idle fraction, **4 s A100 bracket** | **12–16 %** | [9, 19] % | P6 / (P6 + 4 s) |
| **P9** | one-off pool creation cost (spawn, 8 workers, one JAX import each) | **20 s** | [10, 60] s | measured cold `import chlu.training.train_cluformer` = 14.2 s, workers in parallel |

## What would falsify the deliverable
- **P1 < 4.0×** ⇒ the acceptance criterion fails and I report an **honest null with the profile**
  (per-lane time, `c`, straggler spread), not a re-tuned criterion.
- **P5 not exact** ⇒ the change is not admissible at all (decision-replay is the spec); I would ship
  the test as a red tripwire and the feature OFF.

## ⚠ Registered in advance: the probe's "< 30 % GPU idle" claim is at risk
The probe's §8.1 sentence *"the GPU-idle fraction to **< 30 %** on any of the above brackets"*
assumes the full 8× (its "~3 s to ~0.4 s"). My own model says **5×**, which puts the 1 s bracket at
**34–43 %** — i.e. **I predict the probe's < 30 % claim FAILS on the 1 s bracket and holds on the
4 s bracket.** If the measurement lands under 30 % on both, that is a finding against my own model
and I will say so. Either way the number that gets propagated is the measured one.

## Declared NOT measured (so it is never reported as a null)
- Anything on an actual A100 / CSF3 (no cluster route from this machine — the A100 numbers are
  **brackets computed from the probe's two assumed JAX-side times**, exactly as the probe did, and
  are labelled as such).
- Pilot scale end-to-end (26–47 M params). The Python term is measured at **pilot store geometry and
  pilot lane-chunk counts** with a small model, because the controller term is a function of
  `n_layers x batch x n_chunks` and the store config *only* — never of `d_model`. The extrapolation
  to 12 layers is arithmetic and is labelled as arithmetic.
- `fork` as a start method (JAX is not fork-safe; not benchmarked, not offered as a default).
