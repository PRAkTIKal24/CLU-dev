# PREREG — retry-compute-study (written BEFORE the harness ran)

Registered by `experiment-engineer`, w23, on branch `agent/experiment-engineer/retry-compute-study`.
Acceptance criterion is a **measured curve dominance** (accuracy vs compute, six lines per (M,σ)
cell), so this file commits to predictions and their derivation before any number is measured.

## Setup being predicted
- Query = Gaussian-noise retrieval query `clamp(|x+N(0,σ)|,0,1)` (UHop `memory_retrieval_noise`).
- CLU register = `GaussianMemoryPotential` (centers = stored patterns, `s=0.3·median-NN`), damped
  velocity-Verlet settle, `clu_steps` relaxation steps per query = 1 compute unit.
- Compute axis = total relaxation steps / first-pass budget (Nq·clu_steps). Feedforward-NN and
  Hopfield-k-steps are placed at their matched budget multiplier (k+1); this mapping is **generous
  to the baselines** (one NN read ≪ one CLU settle) and is stated as such.
- Ladder k ∈ {0,1,2,4,8}. Six lines: CLU-gated retry, ungated-retry-all, ensemble-of-k-reads,
  random-kick retry, feedforward-NN matched, Hopfield-k-steps.

## Predictions (win / tie / lose called per line, relative to CLU-gated)

| # | line | predicted shape | derivation |
|---|---|---|---|
| P1 | **CLU-gated retry** | **rising, monotone non-decreasing** in compute; the *dominant* line at equal compute | w22 measured +46.9pp from one gated boosted re-relaxation; a k-ladder of re-gated boosts should keep recovering the still-low-confidence tail with sub-linear compute (adaptive spend). |
| P2 | **ungated retry-all** | rises but **strictly below** gated at equal compute; may *dip* at small k (blank-guard −38pp signature — boosting already-correct reads corrupts them) | w22 blank guard: retrying high-confidence reads cost −38.3pp. Ungated pays full compute AND corrupts the confident majority. |
| P3 | **ensemble-of-k-reads** | rises modestly, **below gated**; the honest "is it just k tries?" rival — if it *matches* gated the mechanism claim dies | k independent random starts sample basins but do not *aim* at the query; no directed escape from a wrong-but-near well. |
| P4 | **random-kick retry** | rises **less** than gated (equal compute, equal energy); if it *matches* gated, the Lorentz-boost attribution dies (report plainly) | N1 precedent: equal-energy random perturbation is the fair mechanism control. A random kick escapes wrong wells only by luck; the boost aims down (query − settled). |
| P5 | **feedforward-NN matched** | **flat** (or noise-limited wiggle), NOT a rising curve | A feedforward memory has no "try again with more energy from here" knob; TTA-vote over k augmentations averages out but cannot climb. The CM-23 claim "a curve feedforward memories cannot draw" survives ONLY if this is flat-or-worse. |
| P6 | **Hopfield-k-steps** | **flat after step 1** (converged) or mildly worse if iterated | w22 measured: iterating the modern-Hopfield update does not beat 1 step on this protocol (fixed point reached immediately); no compute-accuracy curve. |

## Quantitative anchors
- P1 magnitude: gated k=8 accuracy − k=0 accuracy predicted **+10…+45 pp** in a cell with headroom
  (first-pass acc in [0.3, 0.8]). (w22 single-rung was +47pp at ×1.5; a re-gated ladder should be
  in the same order, saturating.)
- P4 falsification bar: if random-kick k=8 is within **±3pp** of gated k=8 at equal compute in every
  cell, the boost-mechanism attribution is **falsified** (registry entry, keep the curve).
- P3 falsification bar: same ±3pp rule for ensemble.

## What would kill the novelty claim (pre-committed)
If **ensemble-of-k** OR **random-kick** matches the boosted gated ladder (within ±3pp at equal
compute across cells), the *mechanism attribution* is falsified and reported as such — the curve
(a capability: accuracy rises with test-time compute) still stands as a benchmarkable property.
If **feedforward-NN** draws an equally-rising curve, the CM-23 "feedforward cannot draw this"
claim dies and is retracted here.
