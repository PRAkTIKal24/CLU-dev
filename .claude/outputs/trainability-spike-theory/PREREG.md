# PREREG — trainability-spike-theory (physics-theorist, C2W1)

Written **before** any script in `.claude/scratch/trainability-spike-theory/` was run.
Repo read-only at `main @ 082d095`, clean tree. numpy 2.4.1 / scipy 1.17.0 (main venv); **no sympy in
this venv**, so the symbolic steps are done by hand and verified numerically on random
non-commuting `(M, Hess V)` pairs (a stronger check than a diagonal symbolic one).

## 0. Pre-registered answer to the task's falsifier (the required first item)

**Falsifier:** "the implicit-function hypotheses fail *generically* on the operating set — `λ_min` of the
relevant Jacobian is not bounded away from zero over the configurations the harness visits."

**My pre-registered answer: the falsifier will NOT fire, but only with a named modification, and with
one honest caveat that is itself the interesting result.** Specifically, before deriving:

- **(A)** I expect `I − ∂T_θ/∂z` to be invertible **iff** `Hess V(q*)` is invertible, for every
  `γ ∈ (0,2)` and every `dt > 0` — i.e. the discrete dissipative map adds **no new** degeneracy on
  top of the continuum condition. Reason (derived before writing this file, one line): the fixed-point
  equations of the shipped map reduce to `∇V(q)=0, p=0` with a `(2−γ)` prefactor that vanishes only at
  `γ=2`, which the code already documents as the divergence boundary.
- **(B)** `λ_min(Hess V)` is **bounded away from zero on an open dense set of θ but NOT uniformly along a
  training path**: the degenerate set is **codimension 1** (saddle-node), so a `θ(t)` trajectory does
  not "avoid a measure-zero set", it **crosses it transversally at isolated events**. I pre-register
  three named event classes where it crosses: (i) **well merger** under permitted basin interaction,
  (ii) **item death** under amplitude decay/lifetimes, (iii) **symmetry restoration** (an N5-type
  vacuum collapse). Therefore: implicit gradients are usable **with a ridge/pseudoinverse and a
  `λ_min` monitor**; the task's declaration says needing a ridge is not a falsification.
- **(C)** I expect the *headline tension* (pillar 1's flat directions vs `O(1/λ_min)` conditioning) to
  be **resolvable**, and I pre-register the resolution I expect to find: **exactly flat directions are
  the easy case, approximately flat directions are the hard case.** At exact symmetry the RHS of the
  implicit linear system is orthogonal to `ker H` (a Ward identity), so the system is consistent and
  the pseudoinverse gives the transverse answer with **transverse** conditioning; the flat coordinate
  itself is set by a damped-transport law I expect to be **exactly summable** (geometric series in
  `1−γ`), hence differentiable in `O(1)` with no conditioning cost. I expect the pseudo-flat
  (small-but-nonzero mass) band to be the only genuinely ill-conditioned regime **and to store no more
  than the massive case**, making "avoid the intermediate band" the recommendation.
- **(D)** I expect to **REFUTE** "the loss is invariant along the designed flat direction" for the
  manifold-storage use case — if it were invariant the flat direction would store nothing. So
  resolution-by-quotient is *sufficient for a nuisance gauge mode* and *insufficient for pillar (d)*.

Registered risk: (C) is the part most likely to be wrong. If the flat coordinate's settled value turns
out to depend strongly on the potential's shape parameters, then the flat direction *is* learned
through `V_θ` and my "no conditioning cost" claim fails.

## 1. Quantitative pre-registrations (measured by the four scripts)

| # | prediction | value predicted | how derived |
|---|---|---|---|
| P1 | Fixed-point set of the shipped map | exactly `{(q,0) : ∇V(q)=0}` for all `γ∈(0,2)`, all `dt>0`, all `M≻0`, Newtonian **and** relativistic `T` | hand algebra on the 3-substep + `(1−γ)` map |
| P2 | `det(I − ∂T/∂z)` at `z*` | `= ((2−γ)dt²/2)^d · det(M⁻¹ Hess V)` **exactly** (rel. err < 1e-9 on random non-commuting `M,S`, d=1..6) | Schur complement; `D − CA⁻¹B = (2−γ)I` |
| P3 | discrete-map implicit gradient | `∂q*/∂θ = −(Hess V)⁻¹ ∂_θ∇V` **exactly**, `∂p*/∂θ = 0`; **no** `(γ,dt,M)` correction — FD agreement 1e-6…1e-9 and spread across `γ∈{0.02,0.05,0.1,0.3}`,`dt∈{0.02,0.05,0.1}` **< 1e-6** | the fixed-point *set* is `(γ,dt,M)`-independent |
| P4 | conditioning penalty of solving against `I−J` instead of `H` | `κ(I−J)/κ(H) ≈ 2γ/((2−γ)dt²)`; at shipped `dt=0.05, γ=0.05` → **≈ 20.5**, within ±30% | per-mode `1−λ_slow ≈ (2−γ)u/(2γ)` |
| P5 | `γ=0` | fixed-point set **unchanged**, but `|λ|=1` for underdamped modes (to 1e-12) ⇒ marginally stable, **no settle** | `det J = (1−γ)^d` per mode |
| P6 | exactly-SO(2) `V` (shipped `SO2InvariantPotential` form, reimplemented in numpy) | exactly one zero Hessian eigenvalue at a vacuum point, `|λ_0|/λ_max < 1e-12`; eigenvector ∥ generator `ξ`, `|cos| > 1 − 1e-10` | Goldstone |
| P7 | Ward identity | `⟨∂_θ∇V(q*), ξ(q*)⟩ = 0` to `< 1e-13` relative **for every** radial-MLP parameter ⇒ RHS ∈ range(H) ⇒ `H⁺` solution exists | differentiate invariance twice |
| P8 | angular momentum under the shipped map | `L_n = (1−γ)^n L_0` **exactly** (rel. err < 1e-12) for isotropic `M` + exactly invariant `V` | `q×∇V=0`; the Verlet substeps each preserve `L` |
| P9 | closed-form settled coset coordinate | `θ_∞ ≈ θ_0 + dt·L_0/(m f² γ)`, accurate to **≤ 5%** when launched near the vacuum radius | sum the geometric series |
| P10 | is `θ_∞` learnable through `V_θ`? | **weakly**: `|∂θ_∞/∂θ_radial|` relative to the `p₀` channel `< 1e-2` (nonzero, entering only through the radial transient `r_n`) | symmetry forbids the leading term |
| P11 | reach boundary (`A = L/s ↓ κ_stat`) | `λ_min` at the **spurious** minimum → 0 with saddle-node exponent **0.5 ± 0.05**; `λ_min` at the **item's own** well stays `≥ 0.5·D/s²` across the boundary | saddle-node normal form |
| P12 | two-well merger (permitted basin interaction) | `λ_min ∝ (d − d_crit)^{0.5±0.05}`, `d_crit/s ∈ [1.5, 2.5]`; at the designed gate `d = 4.4s`, `λ_min ≥ 0.8 ×` the isolated value | same |
| P13 | per-step gradient multiplier | `|λ| = √(1−γ)` **exactly** for underdamped modes; overdamped iff `dt²λ/m ≲ γ²/4`, then `λ_slow = 1 − (2−γ)dt²λ/(2γm)` | quadratic char. poly, `det = 1−γ` |
| P14 | the C1 `∂R_γ/∂q₀ = 2.2e-12` number | it is a **finite-difference floor, not the contraction**: exact-Jacobian value at `γ=0.05, N=3000` should be `≈ 0.95^1500 ≈ 3.5e-34` (predict `< 1e-30`) | P13 |
| P15 | defensible truncation depth | `k*(γ,ε) = 2 ln(1/ε)/ln(1/(1−γ))` for the underdamped-dominated case; at `γ=0.05, ε=1e-3` → **269 ± 5 steps**; `∝ 1/λ_min` instead when a soft mode is overdamped | P13 |

## 2. What would count as a failure I must report as a finding
- P3 failing (a `(γ,dt)`-dependent correction appears) ⇒ the C1 identity does **not** transfer to the
  shipped discrete map and the engineer must differentiate the map, not `∇V=0`.
- P7 failing ⇒ the pseudoinverse resolution of Q2 is dead and pillar 1 costs a ridge with bias.
- P10 coming out **large** ⇒ my Q2 resolution is wrong (see registered risk above).
- P11's "item well stays conditioned" failing ⇒ reach failure and ill-conditioning are the *same*
  object with no caveat (a cleaner unification than I expect, and I will say so).
