# PREREG — v1-derivation-appendix (physics-theorist)

Written BEFORE any check script was created or executed and before any build.
All predictions below are derived **by hand, symbolically**, from the paper's own stated
map and configuration (pj_sub.tex @ md5 727ebee2b8498b4095f8bb7159258f90, §2 governed map +
flag table A.1). Every numerical check will be run against the **composed map**
(kick–drift–kick–damp built from elementary pieces; squeeze built as weight∘boost∘unweight;
Jacobians by finite differences of the map function), never against the closed form itself.

## Derivation basis (stated here so the predictions are auditable)

- Relativistic kinetic: T(p) = sqrt(c² pᵀM⁻¹p + m₀²c⁴), M = diag(M_i) > 0, m₀ > 0.
- Drift: q ← q + ε ∇_p T(p);  ∂T/∂p_i = c²p_i/(M_i T);  T > c|p_i|/√M_i strictly (m₀>0)
  ⇒ |Δq_i| < εc/√M_i per drift, independent of p and of V_θ. Kicks/damping move only p.
- Mass-weighted Lorentz squeeze (the form that reproduces the paper's bracket):
  u = √M_i q_i, v = p_i/√M_i;  u' = u coshζ + v sinhζ, v' = v coshζ + u sinhζ
  ⇒ q' = q coshζ + (p/M) sinhζ,  p' = p coshζ + M q sinhζ;  det = cosh²ζ − sinh²ζ = 1.
- Matched quadratic H on the active pair: H_q = ½κ(u²+v²)
  ⇒ H'/H = cosh2ζ + [2uv/(u²+v²)]·sinh2ζ ≤ cosh2ζ + |sinh2ζ| = e^{2|ζ|},
  equality iff u = ±v (sign matched to ζ). At launch state q=0 (u=0): ratio = cosh2ζ exactly.
  Untouched coordinates only pull the full-H ratio toward 1, never above the bound.
- Gated jump q ← q + g(q)Δ, p ← p: Jacobian position block I + Δ∇gᵀ (rank-one);
  matrix determinant lemma ⇒ det J = 1 + ∇g·Δ.

## Pre-registered check values

### P1 — causal box (config A.1: ε=0.05, T=100, c=1, m₀=1, M=diag(4.0, 0.25), γ=0)
1. Per-coordinate single-drift displacement bound: |Δq_0| < εc/√M_0 = **0.025** and
   |Δq_1| < εc/√M_1 = **0.1**, strictly, for EVERY tested momentum including |p| = 1e6,
   and for the full composed Verlet step on the double well (kicks move no position).
2. 100-step displacement of coord 0: ≤ L = **2.5** strictly, approached from below as p₀→∞.
3. At p₀ = (1e6, 0): relative deficit 1 − Δq_tot/L ≈ m₀²c²M_0/(2p₀²) = **2.0e-12**.
   (Prediction: this is the origin of the paper's App. E "relative error 2.0×10⁻¹²".)
4. V-independence: same bound with the double-well kick active and with V ≡ 0.

### P2 — squeeze certificate
5. Numerical Jacobian det of the composed squeeze (finite differences): 1 within ~1e-6
   (consistent with the paper's printed det S = 1.000 ± 4e-6); analytic det exactly 1.
6. Energy ratio at launch state u=0 on matched quadratic H: **cosh 2ζ** =
   1.127626 / 1.543081 / 3.762196 / 27.308233 at ζ = 0.25 / 0.5 / 1.0 / 2.0.
7. Bound e^{2ζ} = **1.648721 / 2.718282 / 7.389056 / 54.598150** — must match the paper's
   printed bounds 1.65 / 2.72 / 7.39 / 54.6 to printed precision.
8. Ordering test on the paper's printed measured ratios: cosh2ζ ≤ printed ≤ e^{2ζ} row-wise:
   1.1276 ≤ 1.13 ≤ 1.6487; 1.5431 ≤ 1.55 ≤ 2.7183; 3.7622 ≤ 3.79 ≤ 7.3891;
   27.3082 ≤ 27.5 ≤ 54.5982. **All four rows must pass.**
9. One-parameter consistency: x := 2uv/(u²+v²) fitted from the ζ=2.0 row alone,
   x = (27.5 − cosh4)/sinh4 ≈ **0.00703**, must reproduce the other three printed ratios to
   their printed rounding: predict 1.131 → "1.13", 1.551 → "1.55", 3.788 → "3.79".
   (This is a consistency demonstration, not a derivation — the paper does not state the
   exact phase at which the ratio was measured; q=0 exactly would give cosh2ζ.)
10. Random-state sweep (1000 states, ζ grid): H(S_ζ z) ≤ e^{2|ζ|}H(z) with **zero violations**
    on matched quadratic H (incl. inactive spectator coordinates); equality at u=v to ~1e-12.
11. Quartic well: raw ratio CAN exceed e^{2|ζ|} (existence of ≥1 violating state) — matching
    the paper's stated scope caveat.

### Bracket (reach thresholds; L=2.5, p₀=1.2, M₀=4.0, landing tolerance 0.4)
12. ζ(d=4.0) = arcsinh((4.0−0.4−2.5)·M₀/p₀) = arcsinh(11/3) = **2.010530** (rounds to the
    paper's 2.0105 / "≈2.01").
13. ζ(d=5.0) = arcsinh(7) = **2.644121** (rounds to the paper's "≈2.64" and the task's 2.6441).
14. Composed rollout (squeeze at t=0, then 100 governed steps on a representative double well
    with wells at 0 and d, barrier ΔV_b = 1): **no landing for any ζ < 2.0105 at d=4.0 and
    any ζ < 2.6441 at d=5.0** (theorem direction — must hold for ANY potential).
    Observed landing threshold at d=4.0 expected in **(2.0105, ~2.5)** — the excess over the
    kinematic bound is V-dependent (flow advance < L at finite momentum) and is NOT a paper
    constant; I pre-register only the one-sided bound sharply.
15. At the paper's swept budget ζ ≤ 2.0: max reach ≤ L + 0.3·sinh(2.0) = 2.5 + 1.0876 =
    **3.5876 < 3.6** ⇒ landing at d=4.0, 5.0 IMPOSSIBLE at tolerance 0.4, while
    d ∈ {0.8, 1.6, 2.4, 3.2} remain coverable — exactly the paper's C.1 squeeze row
    (1, 1, 1, 1, 0, 0).

### Hard gate
16. Finite-difference Jacobian det of composed gated jump vs 1 + ∇g·Δ: agreement ≤ ~1e-7
    for arbitrary smooth g and Δ; constant (frozen) g gives det = 1 exactly.
17. The paper's 2.05 implies ∇g·Δ = 1.05 at the unit-test config; that config (g, Δ) is not
    printed, so 2.05 itself is NOT reproducible from the paper — registered as such in advance.

## STOP clause (echoed)
If any derivation above disagrees with a number the paper prints, that is a finding about
the paper; I will stop, report, and fix neither side.
