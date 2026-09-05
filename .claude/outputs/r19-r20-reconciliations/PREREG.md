# PREREG — r19-r20-reconciliations (results-analyst, w23)

Written **before** running the adjudication harness. Commit base: `main` @ `7ff0651`.
Shipped defaults in effect: `dt=0.125, data_dt=1.0, substeps=8, epochs=150, γ=0.1, relax_steps=32, seed=42`.
Legacy = conflated `dt=data_dt=0.05`. Reinstating config (R19-2) = conflated `dt=data_dt=1.0`.

## R19-1 — which protocol delta carries the n=1 persistence sign
Enumerated deltas between `clu-latent-io-audit` (CLU WINS n=1: 0.4545 vs 0.5673) and
`cmapss-fd002-004-fetch` (CLU LOSES n=1: 0.825 vs 0.600):
(a) launch frame **L-2** (audit; = the shipped `q*` encode launch, `_window_features:120` `q[-1]=w[L-2]`) vs **L-1** (fetch, literal last frame);
(b) target alignment: audit compares traj-step-k against cycle E+1+k (one-cycle **lag**); fetch aligns 1 step ↔ 1 cycle ahead;
(c) γ = 0.0 (audit) vs 0.5 (fetch) — but at n=1 both reports are γ-independent;
(d) anchors 3000 same-engine-guarded (audit) vs 1500 (fetch); (e) dt era (both legacy 0.05).

**Prediction:** the sign is carried by **(a)+(b) jointly = the launch/alignment convention**, NOT by γ, dt, or anchor count.
- Holding launch=L-1 aligned, CLU LOSES at n=1 at every γ and at both dt eras.
- Holding launch=L-2 lagged, CLU WINS at n=1.
- Flipping γ (0→0.5) with launch fixed will NOT flip the n=1 sign.

**Defensible protocol:** the **fetch** one (launch from last observed frame E, aligned 1-step-ahead target,
persistence = hold E). The audit's L-2 launch lags the CLU prediction one cycle behind a smoothly-drifting
target, manufacturing a tracking edge. Predicted verdict: **CLU does NOT beat persistence as a forecaster**;
the win is a launch-convention artifact.

## R20-1 — the ballistic fraction (98.3% vs 79.7%)
Both measured at legacy dt=0.05, γ=0, 16 steps. Definitional knobs:
(i) training **epochs 40** (audit `ballistic.py`) vs **150** (dt-units `item3.py`);
(ii) input **globally z-scored** (audit; = shipped `_prepare`) vs **raw X_train** (dt-units);
(iii) windows 2000 vs 4000; (iv) aggregation global-norm vs per-sample-mean-norm.

**Prediction:** the gap is carried primarily by **(i) epochs** (40-ep under-trained potential → weaker force → more ballistic),
secondarily by **(ii) standardization**. At one canonical definition (**150 ep, standardized input, legacy dt=0.05, γ=0/16, per-sample**)
the number is **≈79–81%** (dt-units is right; 98.3% was an under-trained fit). Under shipped defaults (dt=0.125): **≈50%** (dt-units 50.6%).
Deliverable statement: free-streaming-dominated; **~80% at legacy dt, ~50% at shipped dt**; the 98.3% is retired as a 40-epoch artifact.

## R19-2 — single-basin collapse at correct units
Reinstating config: conflated `dt=1.0, γ=0.5, 64 steps`, q* spread reported 0.0000 with 100% finite.
**Prediction:** the collapse is **REAL, not overflow** (100% finite reproduces), BUT the "0.0000" is display-rounding of a
small finite spread (~1e-6); at float64 the spread is finite and non-zero with a genuine finite fixed point (|q*|~0.2).
It **reproduces across seeds {42,43,44}** and strengthens with γ (γ=1.0 ≥ γ=0.5 collapse). At the **shipped encode budget**
(dt=0.125, γ=0.1, relax_steps=32 ⇒ budget 0.40, not 3.2) it does **NOT** fully collapse (spread stays O(1)). 
Predicted verdict: **REAL strong-global-attractor at high damping budget** (promote as N-candidate, scoped to budget≳1),
**not a property of the shipped encode path**. The floating-point-exact 0.0000 is a rounding/overflow-adjacent display artifact
even though the underlying physics is a real contraction.
