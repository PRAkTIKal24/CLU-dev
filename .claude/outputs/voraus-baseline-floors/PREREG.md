# PREREG — voraus-AD & TEP-Rieth baseline floors (w15, results-analyst)

**Written BEFORE running the harness on voraus/TEP.** Commit-to-a-number, then measure.
Priors: SKAB floors (VUS-PR: knn 0.873 / lof 0.868 / pca 0.580 / iforest 0.522, n=34, point-labelled),
SMD negative control (VUS-PR: pca 0.465 / knn 0.459 / lof 0.428 / iforest 0.273, n=22), both window=100 seed=42.
Binding harness rules: VUS-PR primary for point data, **AUC-ROC primary for episode data**, point-adjust forbidden,
unit-level splits, the 4 statistical baselines mandatory.

Data facts already measured (JAX-free, direct parquet/RData reads — see report §Setup):
- voraus-AD 100 Hz: 2122 episodes, 137 cols = 7 meta + **130 machine signals**; train (setting=PRE_A=72) **948 normal** episodes;
  test **1174** = **755 anomalous + 419 normal (cat 12)**; per-category test counts cat0..11 =
  {0:144,1:156,2:72,3:48,4:22,5:11,6:74,7:80,8:10,9:12,10:89,11:37}; episode length min/median/max = 986/1096/1164 (~11 s @100 Hz).
- TEP-Rieth: fault-free-training 500 runs×500 samples (all normal → the train set); fault-free-testing 500 runs×960 (all normal → degenerate/NaN);
  faulty-testing (not-yet-local) = 20 faults × 500 runs × 960 samples = **10 000 faulty test units**; 52 channels; test onset sample 160 ⇒ anomalous fraction 800/960 = **0.83** per faulty run.

## Metric correction (flag for Hub)
Task item 2 says "per-category **VUS-PR**" for voraus. voraus is **episode-labelled** (no per-timestep GT) ⇒ VUS is undefined;
the harness (`PRIMARY_METRIC["episode"]="AUC-ROC"`) and the voraus-AD paper both use **episode AUC-ROC** (+ AUC-PR). Per-category numbers below are **AUROC**, not VUS-PR.

---

## P1 — voraus-AD (episode AUC-ROC primary; window=100, episode_reduce=mean, seed=42)

**Overall AUROC band (all 4 baselines, all 1174 test episodes):** predict **0.60–0.85**.
Reasoning: well-separated robot-dynamics benchmark, high anomaly base rate (755/1174 = 0.64) but statistical baselines on
100×130 = 13 000-dim flattened windows are curse-of-dimensionality-limited. The voraus-AD paper's own MVT-Flow reports
high detection AUROC (~0.9); classical/simple baselines sit lower — our floor should be *below* the flow, *above* chance.

**Baseline ranking:** predict **knn ≈ lof ≥ pca_recon > iforest** (the SKAB ordering).
*Caveat / competing hypothesis:* at 13 000-dim the distance-based methods (knn/lof) may degrade (curse of dimensionality) and
**pca_recon could lead** — subspace reconstruction is dimension-robust. If pca_recon > knn/lof on voraus, that is a *finding*, not a bug.

**Per-category AUROC map (best baseline; "geometric vs control" structure — the flagship's target):**
| cat | name | predicted AUROC band | rationale |
|---|---|---|---|
| 2,3,4 | COLLISION_{FOAM,CABLE,CARTON} | **0.80–0.98** | large impulsive dynamics/energy excursions — easy |
| 10 | MOTOR_COMMUTATION | 0.80–0.95 | strong periodic current signature |
| 1 | AXIS_WEIGHT | 0.70–0.90 | sustained load change on torque channels |
| 0 | AXIS_FRICTION | 0.65–0.85 | gradual, moderate |
| 7 | CAN_WEIGHT | 0.60–0.85 | payload change, moderate |
| 8 | ENTANGLED | 0.60–0.85 | mechanical binding |
| 11 | WOBBLING_STATION | 0.55–0.80 | subtle vibration |
| 6 | LOSE_CAN | 0.55–0.78 | transient, subtle |
| 5 | MISS_CAN | 0.55–0.78 | brief/absent event, few episodes (n=11) |
| 9 | INVALID_POSITION | 0.55–0.80 | kinematic but small (n=12) |

Predicted ordering by detectability: **collisions ≈ commutation > weight/friction > can-handling/position/wobble**.

**What would surprise me (→ investigate loader/labels, do NOT paper over):**
- Overall AUROC **< 0.60** for the best baseline, or **collisions (2/3/4) AUROC < 0.70** → suspect split/label/scaling bug.
- Any baseline pinned at **~0.50 across all categories** → dead scoring wiring.
- A category AUROC **< 0.45** (systematically *worse* than chance) → sign flip / label inversion.
- Small-n categories (5,8,9 with n=11,10,12) will have wide CIs — a wild value there is noise, not signal.

## P1-neg — internal negative control (voraus)
Not a designated negative-control dataset (SMD was). Internal control: a **label-permuted** episode-score AUROC must be **≈0.50 ± 0.15** (n≈1174).
If the shuffled control departs from 0.5, the metric wiring is biased. Predict: passes (≈0.50).

---

## P2 — TEP-Rieth (point VUS-PR primary; window=100, seed=42; test = 10 000 faulty runs, fault-free-testing degenerate→NaN)

**Overall VUS-PR band (nanmean over valid faulty runs):** predict **0.55–0.85**.
Reasoning: very high per-run anomaly base rate (0.83) lifts the PR floor; TEP normal is tightly clustered (simulated) so
KNN/LOF/PCA separate most faults well — but the aggregate is dragged down by the **notoriously undetectable faults 3, 9, 15**
(TEP literature: near-zero controllable signature). Bimodal per-fault distribution expected.

**Baseline ranking:** predict **knn ≈ lof ≥ pca_recon > iforest** (52 ch × 100 = 5200-dim; less severe than voraus).

**Per-fault VUS-PR structure:**
- **Easy (VUS-PR 0.85–0.99):** faults 1,2,4,5,6,7,8,12,13,14,17,18 (step/large deviations).
- **Hard (VUS-PR near the 0.83 base rate ≈ 0.83, i.e. ~chance-PR, AUROC≈0.5):** **faults 3, 9, 15** (classic TEP "unobservable" faults).
- **Intermediate:** 10,11,16,19,20,21.

**What would surprise me:**
- Easy faults (1,2,6) VUS-PR **< 0.70** → suspect onset-index (160) or channel-selection bug in the loader.
- Faults 3/9/15 scoring **high** (VUS-PR > 0.9) → suspect label leakage (train/test contamination) — investigate.
- Overall wall-clock **≪ 30 min** on CSF3 for 10 000 test units → suspect units silently skipped.

## P2-neg — negative control (TEP)
The 500 **fault-free-testing** runs are all-normal ⇒ degenerate single-class ⇒ harness must emit **NaN rows** (not crash, not 0/1).
Confirm the aggregation counts only valid (faulty) units. This is the "does-the-degenerate-path-hold" control, not a low-floor dataset.

---

## Compute/memory envelope — PRE-registered predictions (to be checked against measured)
- **voraus train-window peak (train_stride=1):** 948 ep × ~995 win = ~943 k windows × 13 000 float32 = **~49 GB** concat peak (before the 100 k subsample cap) → **OOM risk**. Mitigation: `train_stride≥10` ⇒ ~95 k windows ⇒ ~5 GB. Predict CSF3 needs `train_stride=10`.
- **voraus PCA-recon full-SVD** on ~95 k × 13 000: economy-U ≈ 95 k×13 000×8 = **~10 GB** + workspace → PCA is the memory bottleneck; predict total peak **~20 GB**, fits one A100 host slice (≥4 cores × 10.4 GB).
- **voraus wall time (CSF3, CPU baselines):** scaling from SKAB (34 units/9306 win/500-dim → 78 s): ~10× train, ~26× dim, ~34× test units ⇒ predict **20 min – 2 h**; KNN/LOF scoring the dominant term.
- **TEP:** 100 k train windows × 5200-dim; **10 000 test units** scored ⇒ the CSF3-budget stressor. pyreadr RAM: faulty-testing RData ~4 GB + fault-free ~1 GB ⇒ predict peak **~8–12 GB** RAM. Wall time predict **1–6 h** (KNN over 10 000 units dominant). Faulty-training (471 MB) **not needed** (canonical train = fault-free-training only).

*If measurements land outside these bands, the miss is itself a reportable finding (per §5 pre-reg rule).*
