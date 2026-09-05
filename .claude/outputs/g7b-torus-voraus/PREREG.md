# PREREG — g7b-torus-voraus (literal joint-angle→so2-coset torus map on voraus-AD)

**Written before the CSF3 harness run that measures the CLU episode AUROC.**
Commit at prereg time: branch `agent/experiment-engineer/g7b-torus-voraus` off
`main` `e3c8931`. Protocol: voraus-AD, **episode AUC-ROC primary** (episode
label, `episode_reduce="mean"`), IDENTICAL to `voraus-baseline-floors`
(window=100, StandardScaler on flattened windows, PRE_A→normal train / all-other
test = 1174 episodes, 130 machine signals). **CM-3 binding: no energy-superiority
claim — the honest CLU-vs-floor comparison on the identical protocol is the
result, whatever it says.**

## The hypothesis (falsifiable)
The theory predicts that **matching the CLU's coset topology to the data's own
topology** buys `n` independent dissipation-proof `U(1)` registers. voraus joint
space is `T^6 = U(1)^6` (6 serial robot axes). The literal map embeds each joint
angle `θ_j` as `(cos θ_j, sin θ_j)` → one `so2_invariant` unit whose `T^1` coset
*is* that joint's `U(1)`, and couples the six cosets on the arm's kinematic chain
(`ring` = 1-D torus) with the U(1)-preserving `channel_spring`.

## Reference floors (from voraus-baseline-floors)
- **Laptop pre-smoke (24-ch subset, INDICATIVE, not the headline floor):** episode
  AUROC knn 0.772 / lof 0.749 / iforest 0.628 / pca 0.528; shuffled-label
  control ≈0.50.
- **Full 130-ch CSF3 floor: pending** (same run will produce both). MVT-Flow
  (paper headline) ≈0.9 mean AUROC — CLU/baselines expected below the flow,
  above chance.

## Pre-registered predictions (state a number, then measure)
**P1 — order of magnitude.** Both CLU arms (energy/residual + predict) land in
**episode AUROC ∈ [0.50, 0.85]** on the full 130-ch protocol — above chance
(0.50), below the MVT-Flow headline (~0.90). A value ≈1.0 or <0.45 for the best
CLU arm ⇒ a loader/label/protocol bug, investigate not report.

**P2 — CLU vs floor (the honest comparison; CM-3, no superiority claim).** Most
likely outcome: **CLU ties-or-loses to the best statistical baseline (knn/lof)**
overall, because knn/lof are strong on voraus's geometric excursions and CLU is
un-tuned. Predicted best-CLU-arm overall AUROC **0.60 ± 0.12**. Where CLU could
be *competitive* per-category: the hard, subtle control-fault categories where
baselines are weakest in the pre-smoke — **LOSE_CAN (knn 0.56), COLLISION_CARTON
(0.58), WOBBLING_STATION (0.62)** — if the Hamiltonian dynamics prior helps at
all it should show here, not on AXIS_FRICTION (already knn 0.99). This is a
*direction-of-effect* prediction, not a superiority claim.

**P3 — TOPOLOGY-MATCH CONTROL (the core falsifier).** The `--lattice-shuffle-angles`
control permutes which coset each coupling bond connects, so the SAME number of
bonds now couple physically NON-adjacent joints (destroys the kinematic-chain
topology) while keeping the exact same units, channels, kappa, and training. The
prereg'd falsifiable claim:

  - **If the topology match matters:** `AUROC(ordered ring) − AUROC(shuffled) > 0`
    (predicted gap **+0.02 to +0.10**, ≥3 seeds, CI excludes 0).
  - **If the topology match does NOT matter** (null): the gap is within seed
    noise (|Δ| ≲ 0.02, CIs overlap). **This null does NOT falsify CLU-as-detector
    — it falsifies the specific "topology-match buys registers" mechanism claim**,
    which is the honest scientific content of G7b. We commit to reporting the gap
    with its sign and CI either way.

**P4 — arm ranking.** No strong prior on energy/residual vs predict (the "score
mode IS the experiment", Head). Weakly predict `predict ≥ energy ≈ residual` on a
dynamics-labelled benchmark (predict is the conventional TSAD framing, fairest vs
PCA-recon). Recorded so a post-hoc ranking is not dressed as a prediction.

**P5 — Born-Oppenheimer / reduction-regime sanity (CM-9/CM-10).** The coupling is
`channel_spring` (U(1)-preserving; the random-`W` spring would break it) at
`κ_c = 0.05` — the SAME κ at which the priced-channel law was validated (CM-10,
N≤16). We report `κ_c` and the measured `J₂/J₁` (first vs second angular harmonic
of the reduced bond potential) from a trained CSF checkpoint; predicted
`J₂/J₁ ≈ 0` (channel_spring gives a pure first harmonic `V = 2κr*²(1−cos Δθ)`,
verified to 2e-16 in `xy-lattice-theory`), confirming we are in the valid
KT-reduction regime and NOT in a p=2-anisotropy-dominated regime.

## What would falsify what
- P1 out of band ⇒ pipeline bug (not a finding).
- P3 gap significantly **negative** (shuffled BEATS ordered by >0.02) ⇒ the
  topology-match hypothesis is not merely null but *wrong-signed* — a strong
  negative result worth reporting prominently.
- P5 `J₂/J₁` not ≈0 ⇒ the coupling silently broke U(1) (wrong coupling wired) —
  a bug, fix before interpreting P3.
