# PREREG — multi-seed-w23 (results-analyst)

Written **2026-07-24, before any full-scale multi-seed harness was launched**. Only the
two `--quick` smoke runs (seed 99, ~15–18 s each, tiny grids) had been executed at the
time of writing; no full-grid seed ≥1 number existed.

Rationale for pre-registering: the acceptance criterion is a **three-way triage
(SURVIVES / SOFTENS / FLIPS)** of w23's headline numbers. A triage verdict is a
prediction about a measured quantity's stability, so protocol §5's pre-registration rule
applies in spirit. I commit here to (a) the predicted verdict, (b) the predicted seed
spread, (c) the falsifier for each.

Method for every prediction: **5 seeds (0,1,2,3,4)** unless stated; report mean ± seed
std and a **bootstrap 95% CI over seeds** (10 000 resamples, percentile). "Flat" claims
are additionally checked **per seed**, not only on the mean, as the task requires.

---

## Item 1 — `phi-read-in`

**P1.1 (⭐ tier-A, the laundering verdict).** `LAUNDERED` on all 4 (dataset × arm) cells
at **every one of 5 seeds**; per-seed `max_clu_margin ≤ 0.03` (the harness tie band) and
the pooled mean of `max_clu_margin` ≤ 0.01.
*Derivation:* the seed-0 margin is exactly 0.000 with kNN at 0.996–1.000 across the
whole load grid; kNN-in-φ is at ceiling, so for CLU to win a cell it would have to exceed
a near-perfect baseline, which no per-seed resample of the store pool can produce.
**Falsifier:** any seed in which `max_clu_margin > 0.03` in any cell ⇒ SOFTENS; ≥2 seeds
with a cell flipping to a CLU win ⇒ ⛔ FLIPS.
**Predicted verdict: SURVIVES.**

**P1.2 (CIFAR CLU-vs-Hopfield-in-φ margin, CM-23 amendment 1).** At CIFAR M=256, AE arm,
CLU-in-φ − Hopfield-softmax-in-φ ≥ **+0.50** at every seed (seed-0 value +0.965).
**Predicted verdict: SURVIVES.**

**P1.3 (⚠ the MNIST high-load cell where Hopfield is ahead).** This is the fragile one.
At MNIST M=256, PCA arm, seed 0 gives Hopfield 0.957 vs CLU 0.871 (Hopfield +0.086).
I predict the **sign is seed-stable** (Hopfield ahead at ≥4/5 seeds) but the **magnitude
is not** — predicted mean margin in favour of Hopfield **+0.04 to +0.10** with a seed std
≥ 0.03, i.e. a CI that may include 0.
**Predicted verdict: SOFTENS** (the binding CIFAR scoping of the wording stands, but the
"0.957 vs 0.871" pair should be replaced by a mean ± CI).

**P1.4 (retry-confidence AUROC).** All 4 cells ≥ **0.80** at every seed; pooled mean over
the 4 cells ≥ 0.90. The CIFAR-AE cell (seed-0 0.845, the lowest) is predicted to have the
widest spread (std ≥ 0.05). **Predicted verdict: SURVIVES** (the pre-registered bar in the
original study was ≥0.65, which I expect to clear at every seed).

## Item 2 — `retry-compute-study`

**P2.1 (⭐ tier-A, mechanism attribution).** `random_kick` and `ensemble` remain **dead
flat in all 8 cells at all 5 seeds**, judged per-seed against the pre-registered ±3 pp
falsification bar: per seed and per cell, `best_acc(control) − first_pass_acc ≤ +3 pp`.
*Derivation:* seed-0 shows an exactly-zero move for `ensemble` in 8/8 cells and ≤0.4 pp
(negative) for `kick`; both controls are structurally unable to move — the ensemble
re-settles from random momenta into the same basin, and the kick is equal-energy and
undirected. The mechanism (a directed re-launch) is a property of the update rule, not of
the data draw.
**Falsifier:** any (seed, cell) with a control lift > +3 pp ⇒ SOFTENS; >2/40 (seed,cell)
pairs over the bar, or any cell where the mean control lift exceeds +3 pp ⇒ ⛔ FLIPS.
**Predicted verdict: SURVIVES.**

**P2.2 (gated lift).** Direction survives (lift > 0 at every seed in all 4 mask cells and
in the 3 non-ceiling noise cells). Magnitude range predicted to **widen** relative to
seed 0's +6.6…+76.2 pp: I register a predicted pooled range of **+0 … +80 pp** with the
largest per-cell seed std at the hard cells (mask p=0.7: std ≥ 5 pp).
**Predicted verdict: SOFTENS** (range restated with CIs).

**P2.3 (saturation multiplier).** Mean compute-at-best in **×1.1 … ×1.9**, i.e. the
"×1.2–1.8" claim survives as a range but should be quoted as ×1.1–1.9.
**Predicted verdict: SOFTENS** (marginally).

**P2.4 (⚠ the NN gap — headed for an appendix).** Predicted: the NN gap is **negative in
every one of the 40 (seed, cell) pairs** (the NN floor dominates in every cell at every
seed). Predicted pooled range of per-cell means **−3 pp … −45 pp**, i.e. the direction
doc's *"within 3–13 pp of the NN ceiling"* is **wrong** and the Hub's corrected
"3.5–17.6 pp on mask, widening to 42 pp under noise" is **approximately right but will
need its endpoints moved** once seed spread is included.
**Predicted verdict: SOFTENS** — I pre-register that the corrected mask range will
*not* be contained in [3.5, 17.6] pp at 5 seeds (I expect the upper mask endpoint to
exceed 17.6 pp).

## Item 3 — `dimension-aware-budget` frontier (≥3 seeds at 2× atoms)

**P3.1 (d=4, K=16, 2× atoms = 4096).** 3-seed mean strict ≥ **0.90** ⇒ the wall d=4 = 16
holds. Seed std predicted ≥ 0.03 (the write is seed-fragile at this rung: 3-seed 0.876 at
1× vs 2-seed 0.93–0.98 at 2×). **Predicted verdict: SURVIVES, but marginally** — I assign
this the highest FLIP risk of Item 3 and register the falsifier: 3-seed mean < 0.90 ⇒
the "`2^d`, base 2" half of N92 SOFTENS to `K_learned(4) = 8` and the exponent is no
longer exactly 2 in the geometric regime.

**P3.2 (d=5, K=32, 2× atoms = 5792).** 3-seed mean strict ≥ **0.90** (2-seed value 0.906
— on the pass line). Predicted **the closest call in the whole task.** I register a
straight 50/50 and predict the 95% CI will straddle 0.90.
**Predicted verdict: SOFTENS** (expect "≥32" to become "≥16, consistent with 32").

**P3.3 (ceiling cells).** d=6 K=64 and d=8 K=64 at 2× atoms both remain **FAIL**
(3-seed mean strict < 0.90; seed-0-era values 0.809 and 0.894). d=8 K=64 at 0.894 is
within one seed std of the 0.9 line, so I predict it stays below but with a CI that may
touch 0.90. **Predicted verdict: SURVIVES** for the *existence* of a d-independent write
ceiling ≈32 (both cells fail), **SOFTENS** for "≈32" as a sharp number.

## Item 4 — `controller-mvp` (verify, do not re-run)

**P4.1.** The reported ±values are seed std over the 5 paired seeds, and the three
headline cells carry a spread: per-admitted 1.000 with **std exactly 0.000** at every K
(the gate is deterministic given the admitted set); per-offered fixed 0.081 with std
≤ 0.01 (it is `admitted/K` with admitted = 5.2 ± 0.4 ⇒ 0.081 ± 0.006); per-offered sized
0.669 with std ≥ 0.02.
**Predicted verdict: multi-seed-clean, SURVIVES.**

## Global

**P0.** Across all four items I predict **zero ⛔ FLIPS on tier-A entries** (N89 and N90
both hold) and **2–4 SOFTENS**, concentrated in Item 2's quoted ranges and Item 3's
frontier walls. If this global prediction is wrong in the FLIP direction it is a finding
and gets escalated in the first 10 lines of the report.

**Compute priority if the budget binds (declared in advance, per task):**
Item 1 laundering (P1.1) and Item 2 mechanism controls (P2.1) run first and complete at
5 seeds no matter what; Item 3 next; anything dropped is named explicitly.
