# PREREG — fdt-relativistic-expc (written before running the harness)

Author: results-analyst · date 2026-07-19 · code HEAD `e3c8931` (`fdt_relativistic` +
`gibbs_defect_parameter` live, from fix-pack-7 merged into wave-15).

Two measured predictions, committed before any Exp-C `fdt_relativistic` run. The
harness reuses the `relativistic-gibbs-expc` instrument (same checkpoints, same MJ
quadrature reference `mj_reference.py`, same classifier, same paired dream inits/keys),
adding one new sampler arm: `langevin_noise="fdt_relativistic"` (F2, InvGauss scale
mixture). This is the first time F2 runs on the real d=784 Exp-C path.

Control parameter reported per cell: `d·Θ = gibbs_defect_parameter(T) = dim·T/(m₀c²)`.
- mnistFFF (c=1, m₀=1): d·Θ = **784** at T=1, **392** at T=0.5.
- c=5 arm (c=5, m₀=1): d·Θ = **31.36** at T=1.
- native-c1 (c=1, m₀=1): d·Θ = **784** at T=1.
All are `d·Θ ≫ 1` ⇒ deep in the regime where the coded `fdt`/`legacy` samplers are
maximally non-Gibbs (Gaussian, radius 28–90× too small — measured in `relativistic-gibbs-expc`).

MJ reference values (exact quadrature, from `relativistic-gibbs-expc` §1, reproduced):
`Var_MJ/(M_eff·T)` = **785.0** (c1,T1), **392.5** (c1,T0.5), **31.43** (c5,T1);
`r_MJ` (RMS of `‖M^{-1/2}p‖`) ≈ **784.5** (c1,T1), **157.0** (c5,T1).

---

## Q1 — Does F2 sample Gibbs (Maxwell–Jüttner) on real Exp-C? PREDICT: YES.

Mechanism: F2's O-step targets `N(0, M/(2s))` with `s|p ~ InvGauss(...)` per relativistic
unit; MJ = Gaussian scale mixture ⇒ the momentum marginal is MJ **exactly** at the
measure level. The `relativistic-gibbs-expc` result — that `fdt`/`legacy` give a **Gaussian**
marginal (|excess kurt| ≤ 0.0065; `Var/(M_eff T)` = 1.00 (fdt) / 0.095 (legacy); radius
28×/90× too small) — must **invert** under F2.

**Registered predictions (mnistFFF, c=1, T=1, unless noted):**

| observable | `legacy` (measured) | `fdt` (measured) | **F2 PREDICTION** | acceptance band |
|---|---|---|---|---|
| `Var(p)/(M_eff·T)` | 0.095 | 1.00 | **→ MJ = 785** | ≥ 400 (≥ ½·MJ, and ≥100× the fdt value) |
| radius `r_MJ/r̄_obs` | 90.3× | 28.0× | **→ 1.0** (radius restored) | ∈ [0.7, 1.5] |
| KL(empirical radial ‖ MJ) vs ‖ coded-Gaussian | ≫ / ≈0 | ≫ / ≈0 | **KL‖MJ ≪ KL‖Gauss** | KL‖MJ < ½·KL‖Gauss |
| per-component excess kurtosis | −0.001 | −0.004 | **≈ 0** (BLIND stat) | not a discriminator (report only) |

- c=5, T=1: predict `Var/(M_eff·T)` → 31.4 (band ≥ 16); radius restored to `r_MJ/r̄ ∈ [0.7,1.5]`.
- native-c1 checkpoint (2nd checkpoint, item 2): same prediction, `Var/(M_eff·T)` → ~785.
- **The discriminating statistic is the radius / total variance, NOT kurtosis** (concentration
  of measure: at d=784 even MJ's own per-component excess kurtosis is 0.0076). Kurtosis reported
  for completeness only; the loop turns on radius + variance.

**Honest caveats registered up front:** (i) MJ variance is enormous (785× Gaussian), so the
chain needs a long burn-in to inflate `p`; I report split-half stationarity and will down-grade
to "partial / not-converged" if split-half `Var` ratio is outside [0.8,1.25]. (ii) F2 carries an
`O(ε²)` discretization shadow (fix-pack-7 saw a positive residual growing with T); the momentum
*marginal* is exact at the measure level but the *chain* extrapolation is evidenced, not proven.

**Failure = finding:** if F2 does NOT restore the MJ radius (stays Gaussian), the F2 fix does
not work on real data and that is the headline — dig into burn-in / convergence / a code bug.

---

## Q2 — Does the correct sampler change the MNIST imbalance? PREDICT: NO (N10 closure).

N10 is UPHELD: `relativistic-gibbs-expc` showed the imbalance survives c=5, m₀=25, and an
exactly-Gibbs MJ momentum refresh (`mj_refresh`; it made the imbalance *worse*, not better).
The mechanism is **quench depth in V_θ**, and — decisively — in relativistic mode velocity
saturation maps a 90× momentum-law error onto a **0.66 %** error in the velocity the dynamics
feel (`relativistic-gibbs-expc` §2b). F2 changes the momentum *law* (28×→785× radius) but the
dream drift `q̇ = c·M⁻¹p/√(...)` saturates at `c`, so the dream trajectory — hence the digit
histogram — should be essentially unchanged.

F2 is the **clean sampler-only swap** the prior report lacked: unlike `mj_refresh` (which also
ran γ=0 and fully decorrelated momentum every step — §5.2 confound), F2 keeps the same γ=0.3
friction and momentum persistence, only correcting the noise. So it is the *cleanest* test of
"correct sampler ⇒ imbalance?" and N10 predicts a null.

**Registered predictions (mnistFFF, c1_fdtrel vs the prior paired arms):**

- **Primary:** `c1_fdtrel` is statistically indistinguishable from its Newtonian-Gibbs sibling
  `c1_fdt` (same γ, same persistence, only the noise law differs): predict **χ²(c1_fdt, c1_fdtrel)
  NON-significant or small** (p > 0.01), `f(3,5,8,9)` within **±0.12** of fdt's 0.745, and
  **argmax digit = 8** (matching c1_fdt), NOT the deterministic-collapse mode 5.
- **The imbalance is NOT eliminated:** `f(3,5,8,9)` under F2 stays well above uniform's 0.4
  (predict > 0.5); TV-from-uniform stays > 0.3. F2 does **not** move the histogram toward uniform.
- **No collapse:** `c1_fdtrel` does NOT collapse onto a single mode like `c5_legacy`/`det`
  (predict entropy > 1.0 bit; argmax ≠ 5-with->0.9-mass).
- Consistency with the `mj_refresh` result: both are MJ-correct in momentum; `mj_refresh`
  worsened balance via a *dynamics* confound (γ=0 + full refresh). F2 (clean) should sit closer
  to `c1_fdt` than `mj_refresh` did — if instead F2 reproduces `mj_refresh`'s 8-collapse, that
  reinforces "MJ momentum ⇏ better balance" (still N10-consistent).

**Acceptance for the N10 closure:** F2 samples Gibbs (Q1 YES) AND the imbalance is not
eliminated / not driven to uniform (Q2 f(3589) > 0.5). A confirmed null on Q2 with a confirmed
YES on Q1 = the loop closes: "we built the exact fix, proved it samples Gibbs on real data, and
confirmed it does not move the imbalance — the sampler was never the cause; the landscape is."

**Three-way consistency to check (item 4):** {F2 samples Gibbs} + {N10: imbalance = landscape}
⇒ {F2 leaves imbalance put}. If Q1 says F2 is Gibbs and Q2 says the imbalance vanished, N10 is
wrong and that is the interesting result — I will dig in, not paper over it (CM-3).

**Failure = finding:** if F2 *eliminates* the imbalance (histogram → uniform), N10 is falsified
and the sampler WAS the cause — report it loudly with the χ² and the mechanism.
