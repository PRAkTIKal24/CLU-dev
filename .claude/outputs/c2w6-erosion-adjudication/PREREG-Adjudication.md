# PREREG — C2W6 erosion adjudication (analyst-side estimators + my own I2 prediction)

**Filed 2026-08-05 by results-analyst, BEFORE any C2W6 erosion cell has run** (verified: no
`erosion_partition` symbol in `chlu/`, no branch `c2w6-anti-erosion`, no raw artifacts — see report
§1). Protocol §5 pre-registration rule applies: my §1 acceptance criterion is a measured
correlation, so the estimator and my predicted value are committed here first.

Binding scope: adjudication of `PREREG-AntiErosion.md` §3/§4. This file does not amend that prereg
(Hub-owned); it fixes MY analysis choices so they cannot be selected after seeing the data.

## 1. Estimator for I2 — ρ(usefulness, erosion rate), partition-OFF arm

- **Erosion rate, per well.** Post-write window only: monitor samples strictly after the well's
  `last_write_step`. Fit OLS slope of `ln(depth)` vs outer step over that window; net the designed
  decay by subtracting the analytic decay law's own `d ln(depth)/dt` over the same window.
  **`erosion_rate ≡ −(residual slope)`** so that *larger = eroding faster*, matching the registered
  direction of `PREREG-AntiErosion` §3 I2 ("most-useful wells erode fastest" ⇒ ρ ≥ +0.5).
- **Usefulness, two measures, both reported, neither promoted after the fact.** Primary =
  read-selection frequency (plan-lane selection count over the same window, first-named in the
  registered text). Confirmatory = leave-one-well-out probe-batch bpc contribution. The verdict is
  taken on the **primary**; disagreement between the two is itself reported as a finding.
- **Correlation.** Spearman ρ per seed across wells (rank-based: robust to the decades-wide depth
  spread documented in report §3-F3). Pooled via Fisher-z: mean z across the 3 seeds,
  back-transformed; SE = sd(z, ddof=1)/√3. Per-seed ρ always shown alongside the pooled value.
- **Floor censoring (committed in advance, because the collapse is known to reach 1e-177).** Wells
  whose depth is < 1e-30 at the window start are rank-degenerate. If > 50 % of a seed's wells are
  floor-censored, that seed's ρ is reported as **"floor-censored — uninformative"** and is NOT
  scored as no-structure or as refutation. Minimum for a scored seed: ≥ 8 non-censored wells and
  ≥ 3 post-write samples per well.
- **Verdict mapping (from the wave prereg, unchanged):** ρ ≥ +0.5 confirms · ρ ≤ −0.3 refutes ·
  |ρ| < 0.3 no-structure. The band (−0.3, −0.5) ∪ (0.3, 0.5) is *unregistered*: I will report it as
  **inconclusive**, not round it into an adjacent verdict.

## 2. My own registered prediction for I2 (mechanistic, derived — not a hedge)

**Predicted: |ρ| < 0.3 — "no usage structure" — on the primary measure, 2/3 seeds.**

*Derivation.* The N223 mechanism is a gradient path from the outer LM loss into the **writer's
shared parameters** (φ and the write-path leaves), not a per-well credit-assignment channel. If the
optimizer's instruction is essentially "stop writing," it is applied once, globally, to machinery
shared by all wells — so post-write erosion should be approximately well-independent to first
order, giving ρ ≈ 0. The banked evidence is consistent with this: the probe's trained baseline
drives depth to 1.4e-87 / 1.5e-62 / 2.5e-177, i.e. *uniform annihilation across the store*, with no
surviving subset (report §2).

*What would have to be true for the Head's ρ ≥ +0.5 to hold:* erosion would have to be carried by
the **per-well initial-atom leaves** rather than shared φ. That makes a falsifiable mechanism
check: **confirmation requires the gradient-magnitude proxy into a well's own atoms to itself
correlate with usefulness at ρ ≥ +0.5.** A confirmed ρ on the depth series *without* a corresponding
gradient-proxy correlation is a **spurious confirmation** (most likely route: usefulness and depth
are both driven by write recency) and I will report it as such rather than as a confirmation.

*Second-most-likely branch:* ρ ≤ −0.3 (optimizer prunes useless wells preferentially). Least
likely: ρ ≥ +0.5.

Note this pre-registration is **against the Head's registered direction**. If ρ ≥ +0.5 measures
out, my prediction is refuted and that is the finding; it does not license re-reading the band.

## 3. Estimators for the gate re-derivation (§2 of my task)

- Depth aggregates on the **log scale** (paired log-ratio, geometric mean), with the linear paired
  Δ ± SE reported alongside for continuity with the banked tables. Rationale and the worked
  disagreement (1.60 SE linear vs 4.33 SE log on identical banked data) are in report §3-F3.
- All seed statistics **paired, ddof=1, SE = sd/√3**; n = 3 stated on every row.
- E1/E2 are per-seed ratio tests as registered — evaluated per seed, counted (≥2/3, 3/3), never
  replaced by a pooled mean.
- E3/K3 (bpc) paired Δ on the linear scale (bpc is already logarithmic).
- Every re-derived value compared **digit-for-digit** against the engineer's `aggregate()`; any
  mismatch is reported as a finding and never silently reconciled.

## 4. Designed-decay separation audit (§3 of my task)

One seed, end-to-end from the per-well raw series: recompute
`depth_netted(t) = depth_raw(t) / decay_law(t − last_write_step)` and compare to the harness's
netted series. **Pass = agreement to < 1 % relative on every sample.** A mis-netted decay fakes
both flattening (E2) and erosion (E1), so this audit gates my acceptance of *both*.

## 5. Non-negotiables

No new training runs without a dated addendum to this file (≤ 30 min diagnostic budget, declared).
No verdict on any axis the wave prereg's declaration excluded. Every caveat rides: monitor #13/N94
on all w4 cells; toy scale is never a pilot number; "depth is not feature importance" stays live
until my §1 verdict is written.
