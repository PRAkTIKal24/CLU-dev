# w12 hub/composition-ladder — THE LADDER TABLE

> ⛔ **PLACEMENT — APPENDIX -- the K-sweep's ENDPOINT at K = 5. NOT main text: the main text carries ONE K, K*, and never both (Head ruling, 2026-09-21T12:38:07Z, HEAD-RULING-K.md).**

> The store arms run at **K = 5, DECLARED** (`PREREG.md` §3): K\* was unpublished when this hub started and is unpublished still — `CROSSOVER-val.json: K_star = null`. The Head re-runs the store arm at K\* as a free replication; ⛔ this hub does not wait for it and ⛔ no row here is a K\* row.

Every number below is `distinct_valid_novel_tuples` per corner, the metric and the margin the harness pre-declared (`LADDER-READY.json:estimand`). Unit = **corner**. Cells read `mean over seeds (seed0 / seed1 / seed2)`.

⛔ **Seed count and configuration count are separate numbers and are never merged.** `n_seeds = 3` (seeds [0, 1, 2]). Configurations per rung = the rung's corner count, printed in the header of each column.

## 1. The ladder — score per corner

| arm | k=2<br><sub>60 corners · ceiling 36</sub> | k=3<br><sub>120 corners · ceiling 72</sub> | k=4<br><sub>— corners · ceiling —</sub> | k=5<br><sub>— corners · ceiling —</sub> | k=6<br><sub>— corners · ceiling —</sub> | shipped<br><sub>— corners (row3+row4)</sub> |
|---|---|---|---|---|---|---|
| **single design** (K=5, DECLARED) | 30.539 (28.633 / 31.650 / 31.333) | 56.950 (56.950) | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm |
| **pca_tensor** (wave-5 subspace store) | 30.539 (28.633 / 31.650 / 31.333) | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm |
| &nbsp;&nbsp;↳ single design · control `landscape_shuffled` | 0.000 (0.000 / 0.000 / 0.000) | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm |
| &nbsp;&nbsp;↳ single design · control `random_atoms` | 0.167 (0.250 / 0.083) | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm |
| &nbsp;&nbsp;↳ pca_tensor · control `landscape_shuffled` | 0.000 (0.000 / 0.000) | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm |
| &nbsp;&nbsp;↳ pca_tensor · control `random_atoms` | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm |
| MLP @ 524,288 B ref | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm |
| LSTM @ 524,288 B ref | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm |
| transformer @ 524,288 B ref | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm |
| continuous MLP @ 524,288 B ref | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm |
| table FLOOR @ 524,288 B ref | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm |
| MLP @ 524,288 B ref, **own pool** (AMENDMENT-1) | **DECLARED NOT-RUN** — no seed of this rung scored this arm | 8.217 (8.217) | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm |
| MLP @ **1x** the single design's own total (AMENDMENT-1) | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm | **DECLARED NOT-RUN** — no seed of this rung scored this arm |

⚠ **The rungs are NOT on one absolute scale.** The metric's ceiling is the corner's own row count: 36 at k=2 and 72 at k≥3. A k=2 number and a k=3 number are not comparable as absolutes; the ceiling is printed on every column for that reason.

## 2. rank / width on every store row (wave 9's saturation trap)

| arm | k=2 | k=3 | k=4 | k=5 | k=6 | shipped |
|---|---|---|---|---|---|---|
| **single design** (K=5, DECLARED) | 0.695 / 0.889 / 0.865 | 0.689 | — | — | — | — |
| **pca_tensor** (wave-5 subspace store) | 1.000 / 0.889 / 0.926 | — | — | — | — | — |
| &nbsp;&nbsp;↳ single design · control `landscape_shuffled` | 1.000 / 0.889 / 0.941 | — | — | — | — | — |
| &nbsp;&nbsp;↳ single design · control `random_atoms` | 1.000 / 0.889 | — | — | — | — | — |
| &nbsp;&nbsp;↳ pca_tensor · control `landscape_shuffled` | 1.000 / 0.889 | — | — | — | — | — |
| &nbsp;&nbsp;↳ pca_tensor · control `random_atoms` | — | — | — | — | — | — |

> wave 9 measured its maze arm DECORATIVE at `rank/d_k = 0.938`. Any cell at or above that is in the saturation regime and is flagged in FINDINGS.

## 3. ⛔ THE CONTROLS — the store is DECORATIVE at any rung where they do not collapse

| store arm | control | k=2 | k=3 | k=4 | k=5 | k=6 | shipped |
|---|---|---|---|---|---|---|---|
| **single design** (K=5, DECLARED) | `landscape_shuffled` | collapses<br><sub>Δ +28.63 / +31.65 / +31.33</sub> | **NOT-RUN** | **NOT-RUN** | **NOT-RUN** | **NOT-RUN** | **NOT-RUN** |
| **single design** (K=5, DECLARED) | `random_atoms` | collapses<br><sub>Δ +28.38 / +31.57</sub> | **NOT-RUN** | **NOT-RUN** | **NOT-RUN** | **NOT-RUN** | **NOT-RUN** |
| **pca_tensor** (wave-5 subspace store) | `landscape_shuffled` | collapses<br><sub>Δ +28.63 / +31.65</sub> | **NOT-RUN** | **NOT-RUN** | **NOT-RUN** | **NOT-RUN** | **NOT-RUN** |
| **pca_tensor** (wave-5 subspace store) | `random_atoms` | **NOT-RUN** | **NOT-RUN** | **NOT-RUN** | **NOT-RUN** | **NOT-RUN** | **NOT-RUN** |

| store arm | k=2 | k=3 | k=4 | k=5 | k=6 | shipped |
|---|---|---|---|---|---|---|
| **single design** (K=5, DECLARED) | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung |
| **pca_tensor** (wave-5 subspace store) | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung |

## 4. Paired margin against the BEST learned baseline

The rule was fixed at 2026-09-21T11:38:41Z (PREREG §5) and **widened** at 2026-09-21T11:49:46Z (AMENDMENT-3) so the opponent can only get STRONGER: per (rung, seed), the learned arm with the **largest** mean over that rung's corners, among all seven learned arms. Both readings are printed — the narrow five-rival reading is in the column marked `§5`.

| rung | best learned (all 7) — s0 / s1 / s2 | best learned (§5, 5 rivals) | Δ single design (K=5, DECLARED) | Δ pca_tensor (wave-5 subspace store) |
|---|---|---|---|---|
| k=2 | — | — | — | — |
| k=3 | learned_ref_ownpool (8.217) | — | +48.73 [+46.04, +51.33] | — |
| k=4 | — | — | — | — |
| k=5 | — | — | — | — |
| k=6 | — | — | — | — |
| shipped | — | — | — | — |

## 4b. ⭐ THE ANSWER — deterioration and the breaking point, per arm

Both words were defined in `AMENDMENT-2.md` at 2026-09-21T11:46:30Z, before any arm number existed. `deteriorates` compares the **ceiling-normalised** score with the arm's own k=2 value and requires the same sign on every seed. `clears the bar` is the harness's own `beats_beyond_margin` against the best learned baseline, on every seed. A DECORATIVE rung claims no row and is excluded from the breaking point.

| arm | deteriorates vs its own k=2 (k=3/4/5/6) | clears the bar (k=2..6) | **breaking point** |
|---|---|---|---|
| **single design** (K=5, DECLARED) | yes / — / — / — | ✗ / ✓ / — / — / — | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| **pca_tensor** (wave-5 subspace store) | — / — / — / — | ✗ / — / — / — / — | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| &nbsp;&nbsp;↳ single design · control `landscape_shuffled` | — / — / — / — | ✗ / — / — / — / — | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| &nbsp;&nbsp;↳ single design · control `random_atoms` | — / — / — / — | ✗ / — / — / — / — | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| &nbsp;&nbsp;↳ pca_tensor · control `landscape_shuffled` | — / — / — / — | ✗ / — / — / — / — | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| &nbsp;&nbsp;↳ pca_tensor · control `random_atoms` | — / — / — / — | — / — / — / — / — | DECLARED NOT-RUN -- no rung of this arm was scored |
| MLP @ 524,288 B ref | — / — / — / — | — / — / — / — / — | DECLARED NOT-RUN -- no rung of this arm was scored |
| LSTM @ 524,288 B ref | — / — / — / — | — / — / — / — / — | DECLARED NOT-RUN -- no rung of this arm was scored |
| transformer @ 524,288 B ref | — / — / — / — | — / — / — / — / — | DECLARED NOT-RUN -- no rung of this arm was scored |
| continuous MLP @ 524,288 B ref | — / — / — / — | — / — / — / — / — | DECLARED NOT-RUN -- no rung of this arm was scored |
| table FLOOR @ 524,288 B ref | — / — / — / — | — / — / — / — / — | DECLARED NOT-RUN -- no rung of this arm was scored |
| MLP @ 524,288 B ref, **own pool** (AMENDMENT-1) | — / — / — / — | — / ✗ / — / — / — | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=3, so there is nothing to break |
| MLP @ **1x** the single design's own total (AMENDMENT-1) | — / — / — / — | — / — / — / — / — | DECLARED NOT-RUN -- no rung of this arm was scored |

> **no arm has a breaking point** — arms with a breaking point: `{}`.

## 5. ⛔ END REPRODUCTION — the non-negotiable

### k = 2 vs the shipped ROW 3 — **NOT TESTED -- the reference leg is absent**

| arm | rung mean (s0/s1/s2) | shipped mean (s0/s1/s2) | |Δ| | within the shipped corner sd on every seed |
|---|---|---|---|---|
| **single design** (K=5, DECLARED) | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| **pca_tensor** (wave-5 subspace store) | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| &nbsp;&nbsp;↳ single design · control `landscape_shuffled` | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| &nbsp;&nbsp;↳ single design · control `random_atoms` | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| &nbsp;&nbsp;↳ pca_tensor · control `landscape_shuffled` | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| &nbsp;&nbsp;↳ pca_tensor · control `random_atoms` | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| MLP @ 524,288 B ref | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| LSTM @ 524,288 B ref | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| transformer @ 524,288 B ref | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| continuous MLP @ 524,288 B ref | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| table FLOOR @ 524,288 B ref | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| MLP @ 524,288 B ref, **own pool** (AMENDMENT-1) | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| MLP @ **1x** the single design's own total (AMENDMENT-1) | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |

### k = 6 (first 49 corners) vs the shipped ROW 4 — **NOT TESTED -- the reference leg is absent**

| arm | rung mean (s0/s1/s2) | shipped mean (s0/s1/s2) | |Δ| | within the shipped corner sd on every seed |
|---|---|---|---|---|
| **single design** (K=5, DECLARED) | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| **pca_tensor** (wave-5 subspace store) | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| &nbsp;&nbsp;↳ single design · control `landscape_shuffled` | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| &nbsp;&nbsp;↳ single design · control `random_atoms` | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| &nbsp;&nbsp;↳ pca_tensor · control `landscape_shuffled` | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| &nbsp;&nbsp;↳ pca_tensor · control `random_atoms` | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| MLP @ 524,288 B ref | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| LSTM @ 524,288 B ref | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| transformer @ 524,288 B ref | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| continuous MLP @ 524,288 B ref | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| table FLOOR @ 524,288 B ref | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| MLP @ 524,288 B ref, **own pool** (AMENDMENT-1) | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |
| MLP @ **1x** the single design's own total (AMENDMENT-1) | — | — | — | **NOT-RUN**: rung or shipped leg absent for this arm |

## 7. ⛔ DECLARED NOT-RUN

| arm | reason | cells |
|---|---|---|
| `learned_1x` | arm not scored in this leg | 8 — 2:s0, 2:s1, 2:s2, 3:s0, 6:s0, 6:s1, shipped:s0, shipped:s1 |
| `learned_1x` | leg directory absent | 10 — 3:s1, 3:s2, 4:s0, 4:s1, 4:s2, 5:s0, 5:s1, 5:s2, 6:s2, shipped:s2 |
| `learned_ref_ownpool` | arm not scored in this leg | 7 — 2:s0, 2:s1, 2:s2, 6:s0, 6:s1, shipped:s0, shipped:s1 |
| `learned_ref_ownpool` | leg directory absent | 10 — 3:s1, 3:s2, 4:s0, 4:s1, 4:s2, 5:s0, 5:s1, 5:s2, 6:s2, shipped:s2 |
| `pca_tensor` | arm not scored in this leg | 5 — 3:s0, 6:s0, 6:s1, shipped:s0, shipped:s1 |
| `pca_tensor` | leg directory absent | 10 — 3:s1, 3:s2, 4:s0, 4:s1, 4:s2, 5:s0, 5:s1, 5:s2, 6:s2, shipped:s2 |
| `pca_tensor__landscape_shuffled` | arm not scored in this leg | 6 — 2:s2, 3:s0, 6:s0, 6:s1, shipped:s0, shipped:s1 |
| `pca_tensor__landscape_shuffled` | leg directory absent | 10 — 3:s1, 3:s2, 4:s0, 4:s1, 4:s2, 5:s0, 5:s1, 5:s2, 6:s2, shipped:s2 |
| `pca_tensor__random_atoms` | arm not scored in this leg | 8 — 2:s0, 2:s1, 2:s2, 3:s0, 6:s0, 6:s1, shipped:s0, shipped:s1 |
| `pca_tensor__random_atoms` | leg directory absent | 10 — 3:s1, 3:s2, 4:s0, 4:s1, 4:s2, 5:s0, 5:s1, 5:s2, 6:s2, shipped:s2 |
| `rival_cont_mlp` | arm not scored in this leg | 8 — 2:s0, 2:s1, 2:s2, 3:s0, 6:s0, 6:s1, shipped:s0, shipped:s1 |
| `rival_cont_mlp` | leg directory absent | 10 — 3:s1, 3:s2, 4:s0, 4:s1, 4:s2, 5:s0, 5:s1, 5:s2, 6:s2, shipped:s2 |
| `rival_lstm` | arm not scored in this leg | 8 — 2:s0, 2:s1, 2:s2, 3:s0, 6:s0, 6:s1, shipped:s0, shipped:s1 |
| `rival_lstm` | leg directory absent | 10 — 3:s1, 3:s2, 4:s0, 4:s1, 4:s2, 5:s0, 5:s1, 5:s2, 6:s2, shipped:s2 |
| `rival_mlp` | arm not scored in this leg | 8 — 2:s0, 2:s1, 2:s2, 3:s0, 6:s0, 6:s1, shipped:s0, shipped:s1 |
| `rival_mlp` | leg directory absent | 10 — 3:s1, 3:s2, 4:s0, 4:s1, 4:s2, 5:s0, 5:s1, 5:s2, 6:s2, shipped:s2 |
| `rival_table` | arm not scored in this leg | 8 — 2:s0, 2:s1, 2:s2, 3:s0, 6:s0, 6:s1, shipped:s0, shipped:s1 |
| `rival_table` | leg directory absent | 10 — 3:s1, 3:s2, 4:s0, 4:s1, 4:s2, 5:s0, 5:s1, 5:s2, 6:s2, shipped:s2 |
| `rival_transformer` | arm not scored in this leg | 8 — 2:s0, 2:s1, 2:s2, 3:s0, 6:s0, 6:s1, shipped:s0, shipped:s1 |
| `rival_transformer` | leg directory absent | 10 — 3:s1, 3:s2, 4:s0, 4:s1, 4:s2, 5:s0, 5:s1, 5:s2, 6:s2, shipped:s2 |
| `single_design_K5` | arm not scored in this leg | 4 — 6:s0, 6:s1, shipped:s0, shipped:s1 |
| `single_design_K5` | leg directory absent | 10 — 3:s1, 3:s2, 4:s0, 4:s1, 4:s2, 5:s0, 5:s1, 5:s2, 6:s2, shipped:s2 |
| `single_design_K5__landscape_shuffled` | arm not scored in this leg | 5 — 3:s0, 6:s0, 6:s1, shipped:s0, shipped:s1 |
| `single_design_K5__landscape_shuffled` | leg directory absent | 10 — 3:s1, 3:s2, 4:s0, 4:s1, 4:s2, 5:s0, 5:s1, 5:s2, 6:s2, shipped:s2 |
| `single_design_K5__random_atoms` | arm not scored in this leg | 6 — 2:s2, 3:s0, 6:s0, 6:s1, shipped:s0, shipped:s1 |
| `single_design_K5__random_atoms` | leg directory absent | 10 — 3:s1, 3:s2, 4:s0, 4:s1, 4:s2, 5:s0, 5:s1, 5:s2, 6:s2, shipped:s2 |
