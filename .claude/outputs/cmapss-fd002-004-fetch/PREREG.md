# PREREG — cmapss-fd002-004-fetch

**Written before running any harness that measures the quantities below.**
Base commit: `d805cd4` (local `main`). Branch `agent/experiment-engineer/cmapss-fd002-004-fetch`.
Author: experiment-engineer. Date: 2026-07-20.

Prior measured facts this reasons from (from `clu-cafe-integration`, FD001, 3 seeds):
- CLU `basin_coords`-only @ γT=1.6 = **0.7230**; CLU ALL @ γT=1.6 = **0.7168 ± 0.0015**; CLU ALL default γT=0.16 = **0.6554 ± 0.0017**.
- raw_stats 56-d through the identical CoxPH probe = **0.7486** (beats every CLU arm).
- CLU physics scalars alone = **0.5887** (near chance).
- Relaxation-budget curve: 0.16→0.6540, **1.60→0.7158 (opt)**, 12.8→0.7075, 64.0→ `q*` spread **exactly 0.000**, CoxPH singular.
- Epochs 150→600 is a null lever (0.7158→0.7147).

Theorist's collapse taxonomy: degradation phase is a **wake-invisible order parameter** for the HCD objective (T1); the learned potential is **effectively single-basin** (measured).

---

## P1 — CLU will do BADLY on FD002/FD004, and for the predicted structural reason

**Claim.** FD002/FD004 are the 6-operating-condition sets. Multi-regime data requires multi-basin structure to represent; CLU's learned potential is measured single-basin. So CLU should *not* benefit from the extra regime structure — it should average over it.

Committed numbers (best CLU arm = `basin_coords`-only @ γT=1.6, 3 seeds):

| dataset | predicted CLU h-AUROC | predicted sign of (CLU − raw_stats) |
|---|---|---|
| FD001 (already measured) | 0.7230 | **negative** (−0.026, measured) |
| FD002 | **0.60 – 0.70** | **negative**, and *more* negative than FD001's −0.026 |
| FD003 | 0.68 – 0.76 (1 op condition, like FD001) | negative |
| FD004 | **0.58 – 0.68** | **negative**, more negative than FD001 |

**Falsifier.** If CLU ≥ raw_stats on FD002 or FD004, or if FD002/FD004 CLU exceeds FD001 CLU, P1 is **wrong** and the single-basin story does not explain the FD002 deficit.

**Secondary (ordering).** Predicted difficulty ordering for CLU: FD003 ≳ FD001 > FD002 > FD004 (op-conditions hurt more than fault modes).

## P2 — the relaxation-budget optimum will NOT move on FD002 (single-basin persists)

Two competing hypotheses, both pre-registered (v5-gate precedent):

- **H_single (predicted, p≈0.8):** the optimum stays in **γT ∈ [0.8, 3.2]**, i.e. statistically indistinguishable from FD001's 1.6, and at **γT = 64 the cross-sample spread of `q*` again falls below 0.01** (probe singular or near-singular). The potential learns one basin regardless of how many regimes the data has.
- **H_multi (the alternative, p≈0.2):** FD002's 6 regimes induce ≥2 genuine basins ⇒ the optimum moves **up** (≥ 12.8, because fuller relaxation now *resolves basin identity* instead of destroying information) **and** spread at γT=64 stays > 0.01, tracking regime label.

**These make opposite predictions at γT=64**, which is the cheap discriminator.

## P3 — bounded-vs-informative rollout diagnostic: CLU collapses early, and FD002 does not rescue it

Definition (committed here, before measurement) — see report for the final wording.
Let `S(n) = mean_c std_i(q_i(n)[c])` = cross-sample std of the rolled-out state at step `n`, and `S_rel(n) = S(n)/S(0)`.
**Collapse length `n*` := the smallest n with `S_rel(n) < 0.01`** (spread has fallen to 1% of the encoder's input spread).

- Predicted: `S_rel(n)` decays **monotonically and approximately exponentially** in the damping budget γ·n·dt, not in `n` alone.
- Predicted collapse budget **γT* ∈ [3, 30]** on FD001 (bracketing: γT=1.6 still informative, γT=64 already exactly collapsed).
- Predicted: **FD002 does not meaningfully delay collapse** — `γT*(FD002) < 2 × γT*(FD001)`.
- Predicted: the state stays **bounded** throughout (confinement term + dissipation) ⇒ **boundedness will NOT discriminate**; a naive "is the rollout stable" check will report success exactly where the representation is dead. This is the trap the diagnostic exists to catch.

**Falsifier.** If FD002's collapse budget is ≥ 2× FD001's, multi-regime data *does* build extra basin structure and P1/P2 are undermined.

## P4 — the labelling bug: large, systematic, but ranking-preserving

**Derivation (not yet verified numerically).** CAFE sets `t = max_cycle − i` for a window ending at cycle `i`. For a *test* unit the recording stops `RUL_unit` cycles before failure, so the true time-to-event is `t_true = RUL_unit + (max_cycle − i)`. Therefore:

> **CAFE's test label is under-estimated by exactly `RUL_unit`, a constant offset per unit.**
> Train labels are *correct* (train recordings run to failure, RUL_unit = 0).

Committed consequences:
- Mean absolute label error per dataset ≈ **mean of the RUL_FD00x file** (predicted ~75–80 cycles for FD001).
- The error is **not noise** — it is a per-unit constant, so it *systematically re-labels whole units*, and the fraction of test windows whose binary label `t ≤ h` flips at some horizon h ≤ 125 will be **large (predicted > 40%)**.
- **Ranking prediction (the decision-relevant one): the CLU-vs-raw_stats ranking will NOT flip.** Relabelling is a label-side change applied identically to every encoder, so predicted **raw_stats > CLU under true-RUL labels too**.
- Predicted h-AUROC *level* change under true RUL: **substantial, |Δ| > 0.02**, and I predict it goes **up** for both encoders (true RUL is a smoother, more physically-coherent target than a recording-truncation artifact).

**Falsifier.** If the ranking flips, then *no* CAFE C-MAPSS number is externally meaningful and the whole comparison must be re-based — a much stronger finding.

---

## What would make me change my mind about CLU (stated up front)
A confirmed P1 is a **negative result that is still a result**: it converts "CLU underperforms" into "CLU underperforms *because its potential is single-basin and the data is multi-regime*", which is a mechanism, and it makes multi-basin/horizon-conditioning a diagnosis rather than a fishing expedition. A **refuted** P1 (CLU doing well on FD002/FD004) would mean the single-basin measurement does not control benchmark performance, and the anti-collapse thread would lose its main empirical motivation.
