# PREREG — gen-cell-harness (filed BEFORE the harness existed)

**Filed:** 2026-09-13, before a line of `chlu/experiments/exp_gen_cell.py` was written
(mtime-verifiable against the module and against every run artefact in this directory).

**What is being measured.** `request → launch read → codes → ONE frozen decoder → image →
ONE external classifier`, on the 72 held-out (B) requests against the matched-seen (MS)
negatives, for five arms — `clu` · `store_removed` · `knn_proto` · `lace` · `ebm_compose` —
on seeds 0, 1, 2. Statistic: **in-corner `shape` balanced accuracy** (R20/R23), chance 0.5,
AUROC a companion and never the gate. Every accuracy carries its **distinct-output count**.

⛔ No kill-rule verdict is pre-registered or rendered. Composition was RELEASED as a claim
in R29 (kill rule (ii) already fired). These predictions are about **plumbing fidelity and
instrument behaviour**, not about winning.

---

## Predictions, with their derivations

| # | prediction | derived from | band |
|---|---|---|---|
| **P0** | The `clu` arm's **probe-space** in-corner balacc reproduces `gen-read-compose`'s `read` row **bit-identically** (0.6610 / 0.6809 / 0.6858), because the same store checkpoint, the same grouping, the same decoder and the same probe are loaded. | `gen-read-compose` §4 + the store `.pkl` on disk | max\|Δ\| = 0.0 |
| **P1** | `clu` **external-classifier** in-corner balacc ≈ **0.67**, i.e. within ±0.06 of its probe number. | `gen-external-classifier` measured external−probe = **−0.0057 ± 0.0116** on the same decoder's renders, sign-inconsistent | [0.61, 0.73] |
| **P2** | `store_removed` external in-corner balacc = **0.5000 exactly**, TPR 0 / TNR 1, and **1** distinct image of 72. | `gen-read-compose` §4/§6b: the `mean`-deleted read emits one constant vector | exact |
| **P3** | `knn_proto` external in-corner balacc ≈ **0.88**. | probe 0.8884 ± 0.0088 (`gen-arms-a7-wiring`, byte-invariant, P5 Δ=0.000000) + P1's instrument gap | [0.83, 0.93] |
| **P4** | `lace` (`ode`) external in-corner balacc ≈ **0.68**, band widened for sampler noise. | published probe `ode` **0.6797 ± 0.0207** | [0.56, 0.80] |
| **P5** | Distinct **feature vectors** on B: `clu` = **9 / 8 / 10** exactly (seeds 0/1/2); `knn_proto` = 72/72; `store_removed` = 1; `lace` and `ebm_compose` = 72/72 **trivially** (continuous samplers). | `gen-read-compose` §6b + R30/R31's per-seed counts | exact for the three deterministic arms |
| **P6** | ⚠ **The distinct-*image* count is NOT the informative column for stochastic arms.** The comparable column is the number of distinct **classifier-read factor tuples**; on it `clu` ≤ 12 of 72 and `knn_proto` ≥ 55 of 72. | a continuous sampler cannot emit a duplicate float vector; content diversity must be read from the instrument | — |
| **P7** | `ebm_compose` (independent per-factor experts + Gaussian prior, SGLD in feature space — Du et al. 2020's composition, ⛔ not LACE's) lands between `lace` and `knn_proto`: **0.70** | it has the same dense-label supervision as LACE and a simpler, better-conditioned sampling space, but no whitening-exactness guarantee | [0.50, 0.90] |
| **P8** | **Sampler noise on a 72-item set is smaller than the arm differences it is used to read.** Re-running `lace/ode` on ≥3 independent sampler keys at a fixed (seed, config) moves in-corner balacc by **≤ 0.05 sd**. | 72 positives × ~199 negatives; the ODE is near-deterministic given its init | sd ≤ 0.05 |
| **P9** | **Swapping the instrument does not re-order the arms.** The image-space ranking equals the probe-space ranking: `knn_proto` > {`clu`, `lace`} > `store_removed`. | `gen-external-classifier` leg 3: |Δ| ≤ 0.014 per seed, far below the 0.21 `knn_proto`−`clu` gap | ordinal |

## What would make me report a FAILURE rather than a result
- P0 fails ⇒ my harness is not the compose spoke's path and **no number in the table is comparable
  to anything published**. That is a stop-and-report, not a tune.
- A rival lands **below** its published probe number by more than P8's sampler-noise sd ⇒ I weakened
  a competitor; the table is not reported until it is explained.
- An arm's number moves and I cannot name which of {decoder, classifier, arm} moved it.
