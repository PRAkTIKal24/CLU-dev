# trainability-spike-theory — physics-theorist report

Task + acceptance criterion: state and prove the implicit-function conditions for the **shipped dissipative Verlet map** at a settled point, resolve the flat-direction-vs-conditioning tension (Q2, the headline), bound `λ_min` on the operating set incl. the reach boundary, and give a **number** for the defensible truncated-BPTT depth.
Status: **done** — 15 pre-registered predictions run and scored (12 confirmed, **3 falsified as registered and replaced by corrected laws that were then verified**); PREREG written before any script ran.

> ### ⚠ RECONCILIATION LIST (first-10-lines rule — these need an owner at the review that accepts this report)
> **R-1 (my own C1 number, must be corrected wherever quoted).** `clu-controller-spec` §0.2/§2-C1 reports `∂R_γ/∂q₀ = 2.2e-12` after 3000 damped steps and frames the architecture's key fact as **"11.3 orders of magnitude."** That 2.2e-12 is a **finite-difference floor, not the contraction.** The exact Jacobian product over the same 3000 steps gives `10^-32.91` at γ=0.05 and `10^-68.15` at γ=0.1 (s4). The correct statement is **"≥32 orders of magnitude at γ=0.05, and it is a `(1−γ)^{N/2}` law, not a constant."** Conclusion unchanged and **strengthened by ~21 orders**; the measurement was never wrong, its *interpretation* as the contraction was. Owner: curator/Hub — every site quoting "11.3 orders" or "2.2e-12 as the contraction" (`clu-controller-spec`, any handover §1 addendum that inherited it).
> **R-2 (a pillar-(d) framing correction).** The **shipped designed register** (`RingRegisterPotential`, `b=0.05, K=8`) has tangential curvature `λ_tan = bK²/r*² = 3.310` against `λ_max = 7.691` — **2.32× softer than the stiffest mode**, i.e. squarely inside **N46's measured 1.7–4.9× band for the *emergent* arm.** Spectrally, the shipped designed register is **not** a flat direction and does **not** instantiate the manifold pillar; only exact architectural invariance (`SO2InvariantPotential`, or `b→0`) does. Anyone writing pillar (d) must not cite the `b>0` register as evidence for it. Owner: Hub → pillar-(d) author.
> **R-3 (my own falsified preregistrations, never published — recorded for completeness).** P4 (conditioning penalty `2γ/((2−γ)dt²) ≈ 20.5×`) **falsified**, replaced by an exactly-validated `σ_min` formula; P12 (merger exponent 0.5) **falsified** → **1.03**, because a symmetric two-well merger is a **pitchfork**, not a saddle-node; P10's *number* falsified (0.207, not <0.01) while its *structural* claim survived and sharpened. Owner: this report only.

## ⭐ DIAL DECLARATION (echoed per protocol §7)
- **Dial:** none — theory. Enables C2 trainability; makes no performance claim, measures no benchmark.
- **Laundering control:** n/a. The engineer half's gradcheck-vs-truncated-unroll is the substitute-control analogue; its tolerance is specified in §7 (**register 1e-3 at k=270, not 1e-5 — and 1e-5 costs k=449**).
- **Falsifies:** implicit-function hypotheses failing generically on the operating set. **Verdict: did NOT fire** — but with a named modification (ridge + a three-part monitor) and one honest caveat (§Q1.4: the degenerate set is **codimension 1**, therefore *crossed*, not avoided).
- **Does NOT falsify:** needing a ridge (standard DEQ practice) · ill-conditioning at deliberately degenerate points · a loose-but-correct bound.

**What I did:** derived the fixed-point set, Jacobian, invertibility condition and implicit gradient of the *shipped* damped velocity-Verlet map in closed form with `(γ, dt, M)` explicit (§Q1); resolved Q2 with a hybrid **project-and-transport** scheme whose flat-direction half is an *exact geometric sum* rather than a truncated unroll, and proved the Ward identity that makes it work (§Q2); measured `λ_min` across the reach boundary, well merger, and item death and **refuted the proposed reach↔conditioning identification while salvaging the useful half** (§Q3); derived the per-step gradient multiplier and gave the truncation depth with its γ-dependence and a two-sided constraint (§Q4); answered consolidation and showed **N5 is predicted by the same theorem that makes Q2 work** (§Q5).

**How I verified:** 7 self-contained numpy scripts in `.claude/scratch/trainability-spike-theory/` (`common.py`, `s1_fixedpoint.py`, `s1b_conditioning.py`, `s2_flat.py`, `s2b_flat_exact.py`, `s3_reach_lambdamin.py`, `s3b_exponents.py`, `s4_truncation.py`, `s5_ridge_budget.py`, `s6_relativistic_L.py`) + results JSONs alongside. The integrator is `chlu/core/integrators.py::velocity_verlet_step` reproduced line-for-line (3 substeps, then `p ← (1−γ)p`), float64, Newtonian **and** relativistic `T`.

**Flag-provenance (all results in this report).** Repo **read-only** at `main @ 082d095`, clean tree; **no repo code imported, no JAX** (so the w6 worktree-JAX hazard does not apply). numpy **2.4.1**, scipy **1.17.0** (main venv `/Users/user/Desktop/CHLU/.venv`). Seeds: `default_rng(0)` (s1), `default_rng(100+d)` (s1 det-identity), `default_rng(7)` (s2/s2b/s6 SO(2) MLP weights). Shipped constants used, with their source lines: `dt=0.05` (`config.py:179,222,312,374`), `settle_gamma=0.1 / settle_steps=2000` (`config.py:320-321`), two-phase read `gamma_address=0.05, address_steps=400, gamma_read=0.0, read_steps=800` (`config.py` `ExperimentLearnedMemoryConfig`), `clu_dt=0.1, clu_gamma=0.05, clu_steps=1` (primitive harness), register `lam=1.0, f=1.0, b=0.05, K=8, kappa=1.0` (`memory_potentials.py::RingRegisterPotential`), `SO2InvariantPotential(confinement=0.05)`. Landscapes: `V = α‖q‖² − Σ A_i exp(−‖q−c_i‖²/2s²)` with `s∈{0.30,0.35}, α∈{0.05}` (matching C1/S1-S3 and `readout-channel-theory`), and `V = g(r²) + ½k q₂²`, `g(u)=λ(u−f²)² + αu + ε·MLP(u)`, `ε=0.05`, MLP `1→8→8→1` tanh (the shipped `SO2InvariantPotential` structure). No training was run; no config flag of the repo was exercised.

---

# 0. Headline verdicts

1. **The C1 identity transfers to the shipped discrete map EXACTLY, with no `(γ, dt, M)` correction:** `∂q*/∂θ = −(Hess V)⁻¹ ∂_θ∇V` and `∂p*/∂θ = 0`. Verified against re-settled finite differences to **1.3e-8**, with the prediction's spread across `γ∈{0.02,0.05,0.1,0.3,0.9} × dt∈{0.02,0.05,0.1}` equal to **5.5e-14** — i.e. the answer is `(γ,dt)`-independent to 14 digits, as the proof requires. **The dissipation and the discretization enter the *conditioning*, never the *answer*.**
2. **The exact invertibility condition, with the constants in it:** `det(I − ∂T_θ/∂z)|_{z*} = ((2−γ)dt²/2)^d · det(M⁻¹ Hess V(q*))` — verified to **2.07e-11** relative on random *non-commuting* `(M, S)`, `d=1…6`, `γ` up to 1.7. So `I − ∂T/∂z` is invertible **iff `Hess V(q*)` is nonsingular**, for every `γ ≠ 2`: **the discrete dissipative map adds no degeneracy of its own.** Separately, stability requires `0 < dt²λ_i/m < 4` for every Hessian eigenvalue — **γ-independent**, and the shipped `dt=0.05` sits **13.9× below** that limit.
3. **⭐ Q2 (the headline): the pillar does NOT structurally break the trainer — but for a different reason than "quotient it out", and the naive quotient answer is *insufficient*.** At exact symmetry the implicit system is *consistent* (a Ward identity, `⟨∂_θ∇V, ξ⟩ = 0` to **1.35e-16**), so `H⁺` gives the transverse answer with **transverse** conditioning. The flat coordinate itself is then set by an **exactly summable damped transport law** — angular momentum obeys `L_n = (1−γ)^n L_0` to **6.1e-14** (and to **9.1e-14** under the *relativistic* kinetic at `v/c = 0.51`) so `θ_∞ = θ_0 + dt·L_0/(m r*² γ)`, accurate to **0.02–0.75%**. **Recommendation: project-and-transport — implicit gradients on the normal bundle + the closed-form geometric sum on the orbit. Cost: O(1), no unrolling, no `1/λ_min`.**
4. **⭐ And the loss is NOT invariant along the designed flat direction — necessarily so.** If it were, the flat direction would store nothing and pillar (d) would be vacuous. So the pure-quotient resolution applies only to *nuisance gauge* modes. The measured consequence: the stored coset coordinate is carried by `M⁻¹p₀` (`∂θ_∞/∂p₀ = 1.008` vs closed form 1.017), and `V_θ` reaches it **only through the orbit's radius**, obeying `∂θ_∞/∂θ_V = −(θ_∞−θ_0)/r* · ∂r*/∂θ_V` at ratio **0.987–0.990** for 4 of 5 parameters and **exactly 0** for a pure `V`-value shift. Manifold-valued content is written by the momentum channel, read through geometry — and the gradient path to it is the *well-conditioned* transverse solve.
5. **The genuinely dangerous regime is the PSEUDO-flat middle, and it buys nothing.** `cond(H)` at the register goes **7.69 → 11.6 → 116 → 1162 → 9.6e7** as `b = 0.05 → 0.01 → 1e-3 → 1e-4 → 0`, while the *storage* is `K` discrete slots for any `b > 0` and a continuum only at `b = 0`. **Be exactly flat (architecturally) or comfortably massive; never in between.** Corollary R-2: the shipped `b=0.05` register is 2.32× soft — N46's *emergent* band.
6. **⭐ Q3: "reach failure = implicit-gradient ill-conditioning" is REFUTED as stated, and half of it is true.** Across the reach boundary (`L_crit = κ_stat·s = 1.2323`, `κ_stat = 4.1077`) the **item's own** `λ_min` is **11.184–11.199** — constant to 0.13% — and at the reached fixed point it is **11.1862 to six significant figures for every launch point and every γ**. The *address/separatrix* sub-mode (the one that binds on the trained shipped `V`, 31/32) is **invisible to `λ_min`**; it is visible in the **settle residual**, which moves over **14 orders** (8.9e-15 → 2.0) across the same sweep. The **well-loss** sub-mode *is* the same object: merger `λ_min` 8.244→1.26 with the implicit gradient up **147×**; decay-to-death `λ_min` 8.244→0.299 with the implicit gradient up **465×** and then a discontinuous jump of the settled point to a neighbour's well. ⇒ **The trainer's health check is a TRIPLE — (residual, λ_min, basin identity) — and monitor #11 covers only one leg.**
7. **⭐ Q4, the number: `k* = ln(1/ε)/ln(1/ρ)` with `ρ = max(√(1−γ), 1 − (2−γ)dt²λ_min/(2γm))`. At the shipped address phase (γ=0.05, dt=0.05, well curvature 8.24) `ρ = 0.97479` (predicted `√0.95 = 0.97468`) and `k* = 269 steps` at ε=1e-3, `180` at ε=1e-2, `449` at ε=1e-5.** Beyond ~270 steps additional unroll is worthless. Two structural riders: **(a)** the shipped phase 2 runs at `γ_read = 0`, where `ρ = 1.0` exactly — *no* geometric death, so truncation there costs `O(k/N)`, not `O(ε)`; the natural truncation boundary is therefore **the phase-1/phase-2 junction**. **(b)** through a saddle the multiplier *exceeds* 1 (`ρ_+ = 1.153` at a well-scale unstable mode ⇒ 16 steps per decade of gradient **growth**), and the two-sided constraint has **no solution** there (`k_max = 38 < k_needed = 269`) — an argument for the implicit path wherever a fixed point exists.
8. **One number controls four things.** `λ_min(Hess V(q*))` sets (i) implicit-gradient conditioning `O(1/λ_min)`, (ii) truncation depth `k* ∝ 1/λ_min` in the overdamped regime, (iii) proximity to the well-loss bifurcation, (iv) whether the mode is resolvable in the read budget at all. **This licenses one cheap monitor to serve the trainer, the controller and the collapse checklist simultaneously.**
9. **The principled ridge, with zero free parameters: `λ_ridge = 2γ m ln(1/tol) / (N_settle (2−γ) dt²)` — ridge at exactly the softness the read budget cannot resolve.** Shipped (`γ=0.05, N=400, tol=1e-3`): **λ_ridge = 0.354**, verified self-consistent (residual factor 9.4e-4 after 400 steps), costing **4.1% bias** on a healthy well mode and capping gradient amplification at **2.82×**.
10. **⭐ Q5: N5 is not a coincidence — it is a corollary of the theorem that makes Q2 work.** The Ward identity says a settle-based loss has **exactly zero** gradient along the coset direction. That is *why* the implicit system is consistent, and *why* the flat direction is **undefended**: anything else in the objective (CD's negative phase) moves it unopposed. One theorem, two consequences — one good, one N5. The strongest guard is architectural (parametrize by invariants so the lift is **inexpressible**), which the shipped `SO2InvariantPotential` already does. And **"consolidate less often" is not a guard**: N5's own horizons, converted to sleep *updates*, are **116 / 88 / 48** at f = 1/5/20 — rarer sleep inverts in *fewer* updates.

---

# Q1 — Existence: the fixed point of the shipped dissipative Verlet map

**Notation.** Shipped step `T_θ: (q,p) ↦ (q',p')` for separable `H = T(p) + V_θ(q)`:
```
p_half = p − (dt/2)∇V(q)
q'     = q + dt ∇T(p_half)                       [∇T = M⁻¹p Newtonian; c²M⁻¹p/√(c²pᵀM⁻¹p+m₀²c⁴) relativistic]
p'     = (1−γ)[ p_half − (dt/2)∇V(q') ]
```
`S ≡ Hess V_θ(q*)`, `d = dim q`, `u_i ≡ dt²λ_i/m` for a Hessian eigenvalue `λ_i` (isotropic-mass shorthand).

## Prop Q1.1 (the fixed-point set) — **PROVEN + verified**
> For every `γ ∈ (0,2)`, every `dt > 0`, every `M ≻ 0`, and for **both** the Newtonian and the relativistic kinetic,
> **`Fix(T_θ) = { (q, 0) : ∇V_θ(q) = 0 }`** — the critical points of `V_θ`, with momentum **exactly** zero.

*Proof.* `q' = q` ⟹ `∇T(p_half) = 0`. For both kinetics `∇T(p) = 0 ⟺ p = 0` (the relativistic prefactor `c²/√(·)` is strictly positive), so `p_half = 0`, i.e. `p = (dt/2)∇V(q)`. Then `p' = (1−γ)[0 − (dt/2)∇V(q)]`, and `p' = p` gives `(2−γ)·(dt/2)∇V(q) = 0`. Since `γ ≠ 2` and `dt ≠ 0`, `∇V(q) = 0`, hence `p = 0`. ∎

*Verification (s1).* Settling from `c₀+(0.2,−0.15)` on the 4-well ring (`s=0.35, α=0.05`), 20000 steps, `dt=0.05`: `‖p*‖ ≤ 4.4e-15` and `‖∇V(q*)‖ ≤ 2.7e-13` at `γ = 0.01, 0.05, 0.1, 0.3, 0.9, 1.5`; identical to 16 digits under the relativistic kinetic (`m₀=1, c=5`) at `γ = 0.05, 0.1, 0.3`. The off-critical branch is confirmed to machine precision: setting `p = (dt/2)∇V(q)` at a non-critical `q` gives `‖q'−q‖ = 0` **exactly** and `‖p'−p‖ = (2−γ)(dt/2)‖∇V(q)‖` to 16 digits at `γ = 0.05, 0.5, 1.0, 1.9`. **The `(2−γ)` factor is the whole content of the γ-dependence, and it is why `γ=2` is the divergence boundary the code already documents (`eval/config.py`: `relax_gamma > 2` DIVERGES).**

### Q1.1b — what happens at `γ = 0` (why the two-phase read is forced, not conventional)
The fixed-point **set is unchanged** at `γ=0` (the proof only needs `γ≠2`). What changes is attraction:
- `det(∂T/∂z) = (1−γ)^d` exactly (per-mode `det = 1−γ`, s1). At `γ=0` the map is **symplectic**, phase volume is exactly preserved, and **Liouville forbids any attractor.** No amount of `dt` or landscape design can produce a settle at `γ=0`.
- Measured: per-mode multiplier `|λ| = 1.0` exactly; after 20000 steps at `γ=0` the state still has `|p| = 0.231`, `‖∇V‖ = 1.515`, orbiting at 0.293 from the well centre.
⇒ **A settle *is* a dissipation budget.** The shipped two-phase read (`γ_address = 0.05` then `γ_read = 0.0`) is exactly the statement "use dissipation to acquire the address, then switch it off to read the conservative dynamics" — and the γ=0 phase provably has no fixed point to be implicit about. *(Practical rider: at `γ = 1.99` the settle failed to converge in 20000 steps — `1−γ = −0.99` gives sign-flipping, slowly-decaying oscillation. The usable band is `γ ∈ (0, ~1.5]`.)*

## Prop Q1.2 (invertibility, with the `(γ, dt, M)` constants explicit) — **PROVEN + verified**
The exact step Jacobian at `z* = (q*,0)`, Newtonian `T` (`I` = `d×d` identity):
```
∂T/∂z |_{z*}  =  ⎡ I − (dt²/2)M⁻¹S                         dt·M⁻¹                    ⎤
                 ⎣ −(1−γ)·dt·S(I − (dt²/4)M⁻¹S)     (1−γ)(I − (dt²/2)SM⁻¹)          ⎦
```
Its Schur complement against the top-left block collapses **exactly**: `D − CA⁻¹B = (2−γ)I`. Hence
> ### `det( I − ∂T_θ/∂z )|_{z*} = ((2−γ)·dt²/2)^d · det(M⁻¹ Hess V_θ(q*))`
> ### ⟹ `I − ∂T_θ/∂z` is invertible **⟺** `Hess V_θ(q*)` is nonsingular, for every `γ ≠ 2`, `dt ≠ 0`, `M ≻ 0`.

*Verification (s1).* Max relative error **2.07e-11** across `d ∈ {1,2,3,5,6} × γ ∈ {0, 0.05, 0.5, 1.7} × dt ∈ {0.02, 0.05, 0.3}` with `S` symmetric-indefinite and `M` SPD **not commuting with `S`** (so this is not a diagonal identity in disguise). The analytic Jacobian agrees with a central-difference Jacobian of the actual map to **1.05e-10**. On the real ring fixed point: `det(I−J) = 3.8150729753161583e-4` measured vs `3.8150729753161545e-4` from the formula.

**Conditioning (this is where `(γ,dt)` DO enter) — P4 falsified, corrected law verified.** Per-mode characteristic polynomial `λ² − (2−γ)(1−u/2)λ + (1−γ) = 0`, from which:
| regime | condition | multiplier `ρ` | `σ_min(I−J)` |
|---|---|---|---|
| **underdamped** | `λ_i > λ_crit ≡ γ²m/(2(2−γ)dt²)` | `√(1−γ)` **exactly** (dt- and λ-free) | `≈ dt√(λ_i/m)` |
| **overdamped** | `λ_i < λ_crit` | `1 − (2−γ)dt²λ_i/(2γm)` | `(2−γ)dt²λ_i / (2√(m²γ² + dt²))` |

At shipped `dt=0.05, γ=0.05, m=1`: **`λ_crit = 0.256`** — so a shipped *well* mode (`D/s² ≈ 8–11`) is underdamped while the *confinement* mode (`2α = 0.1`) is **overdamped**. My registered P4 (`κ(I−J)/κ(H) ≈ 2γ/((2−γ)dt²) ≈ 20.5`) is **FALSIFIED**: measured ratio **7.69** at those settings, and nearly γ,dt-flat (1.06–7.4 across the grid). Diagnosis: I had written down the overdamped `σ_min` but forgotten that `σ_max` shrinks too. The corrected `σ_min` formula is validated to 3 digits — the measured/predicted ratio equals `γ/√(γ²+dt²)` at **0.371 / 0.707 / 0.894 / 0.986** for `γ = 0.02 / 0.05 / 0.1 / 0.3` (predicted 0.371 / 0.707 / 0.894 / 0.986). **Corrected engineering statement: `κ(I−J) ≈ O(1)·κ(H)`, both `∝ 1/λ_min`. There is no `dt⁻²` penalty. The reason to solve the `d×d` Hessian system rather than the `2d×2d` map system is *size and simplicity*, not conditioning.**

**Stability (a free, actionable by-product).** Both multipliers lie in the open unit disc `⟺ |tr| < 1 + det ⟺ |1 − u_i/2| < 1 ⟺ 0 < u_i < 4` for every `i`. **This is γ-independent.** Verified: `ρ = 1` exactly at `u = 4` for `γ = 0, 0.05, 0.3`, and `ρ > 1` at `u = 4.01`. At the measured shipped curvature `λ_max = 8.24`, the limit is `dt < 2√(m/λ_max) = 0.697` while the shipped `dt = 0.05` gives `u = 0.0206` — **`dt` is 13.9× below the stability limit.** ⚠ **But raising `dt` does not buy settle speed** (see §Q4: `ρ = √(1−γ)` is `dt`-free); it buys ballistic reach per step only.

## Prop Q1.3 (the C1 identity transfers exactly — **the constant does NOT change**) — **PROVEN + verified**
> For the shipped discrete map, at a fixed point with `Hess V_θ(q*)` nonsingular:
> ### `∂q*/∂θ = −(Hess V_θ(q*))⁻¹ ∂_θ∇V_θ(q*)`,  `∂p*/∂θ = 0`
> **exactly** — no `γ`-, `dt`-, or `M`-dependent correction, and no `O(dt²)` remainder.

*Proof 1 (one line).* By Prop Q1.1 the fixed-point set is `{∇V_θ(q)=0, p=0}`, which does not contain `γ`, `dt` or `M` **as a set**. Implicit differentiation of `∇V_θ(q*(θ)) = 0` gives the result; `p* ≡ 0` gives the second half. ∎
*Proof 2 (the explicit solve, for the engineer who wants to see it in the map).* `∂_θT|_{z*} = ( −(dt²/2)M⁻¹g ; −(1−γ)dt(I − (dt²/4)SM⁻¹) g )` with `g ≡ ∂_θ∇V(q*)`. Substituting `(δq, δp) = (−S⁻¹g, 0)` into `(I − ∂T/∂z)(δq,δp)` reproduces **both** blocks identically (row 1: `(dt²/2)M⁻¹S·(−S⁻¹g)`; row 2: `S(I−(dt²/4)M⁻¹S)S⁻¹ = I − (dt²/4)SM⁻¹`). ∎

*Verification (s1).* `θ` = amplitude of well 0 on the 4-well ring; `∂q*/∂θ` by re-settling at `θ ± 1e-4` vs `−H⁻¹∂_θ∇V`: **max relative error 1.3e-8** (finite-difference-limited) over `γ ∈ {0.02, 0.05, 0.1, 0.3, 0.9} × dt ∈ {0.02, 0.05, 0.1}`, and the *prediction's* spread across that whole grid is **5.5e-14**. Also verified `‖p*‖ ≤ 1e-15` at every cell, so `∂p*/∂θ = 0` is not an approximation.

**⇒ C1's result stands verbatim for the shipped map. The dissipation and the discretization are invisible to the implicit gradient.** (What they *do* affect: whether you have arrived at the fixed point — the residual — and the conditioning of the solve.)

## Q1.4 Genericity under a learned `V_θ` with permitted basin interaction — **partially proven, partially measured; the honest answer is "codimension 1, therefore crossed"**
C1 §C1.3's statement (Morse-ness is open-dense in `C²`) survives: for **almost every** `θ`, `V_θ` is Morse and Prop Q1.2/Q1.3 apply. But that is the wrong question for a *trainer*, and the task is right that C1 never covered this case. Three sharpenings:

1. **The degenerate set is codimension 1 in `θ`-space, so a training path crosses it transversally at isolated times** — it is not "avoided with probability 1". Measure zero is the wrong reassurance; codimension is the right accounting. My merger sweep **is** such a crossing (a 1-parameter family through `d_crit`).
2. **Two crossing types, with different measured signatures** (s3b) — this is a usable diagnostic:
   | event | bifurcation | `λ_min` scaling | measured exponent |
   |---|---|---|---|
   | symmetric two-well **merger** (permitted basin interaction) | **pitchfork** | `∝ (d − d_crit)^1` | **1.03** (49 pts) |
   | asymmetric **well loss** (decay to death; reach's spurious-minimum annihilation) | **saddle-node** | `∝ Δ^{1/2}` | **0.49–0.50** (local, reach) / **0.573** (decay, grid-limited) / root-gap **0.505** |
   My registered P12 (0.5 for merger) is **FALSIFIED** — because a *symmetric* merger is a pitchfork. The corrected classification is more useful than the prediction: **the exponent tells you which collapse you are in.**
3. **A learned `V_θ` under capacity pressure is *rewarded* for approaching the degenerate set** (superposing items is how a learned landscape beats a table — charter §2.1(a)), so crossings are a routine event, not a pathology. ⚠ **Consequence for the loss:** `θ ↦ q*(θ)` is **piecewise smooth**, smooth on each basin-identity cell, with **jump discontinuities on the codim-1 cell boundaries.** Measured jump: at item death the settled point moves from 0.18 to **1.53** away from its own centre (into a neighbour's well) between `A₁ = 0.05` and `0.03`. The gradient is correct a.e. (SGD is fine); the *loss* is not continuous, so **loss-decrease-based line searches and trust-region acceptance tests are unsound here** (§7 request 6).

---

# ⭐ Q2 — THE CENTRAL TENSION, resolved: project-and-transport

**The tension as posed.** Pillar 1 says flat directions store a manifold, i.e. `λ_min(Hess V) → 0` along the flat direction; the implicit gradient is `−H⁻¹∂_θ∇V` with conditioning `O(1/λ_min)`. Does the highest-novelty pillar break its own trainer?

**Answer: no — and the reason is sharper than "quotient it out".** Decompose the tangent space at `q*` as `T_orbit ⊕ N` (orbit tangent = `span ξ(q*)`, `ξ` the symmetry generator). The two halves need **different machinery**, and each has an exact result.

## Prop Q2.1 (Ward identity ⇒ the implicit system is CONSISTENT) — **PROVEN + verified to 1.35e-16**
Let `V_θ` be exactly `G`-invariant for all `θ` (`G = SO(2)` on a channel, as in the shipped `SO2InvariantPotential`). Differentiating `V_θ(g_s q) = V_θ(q)` in `s` gives `⟨∇V_θ(q), ξ(q)⟩ ≡ 0` for all `q, θ`; differentiating **that** in `θ` gives
> ### `⟨ ∂_θ∇V_θ(q), ξ(q) ⟩ = 0` for all `q, θ`  — in particular at `q*`.
Since `H = Hess V` is symmetric with `ker H ⊇ span ξ(q*)`, `range(H) = (ker H)^⊥ ∋ ∂_θ∇V`. **The singular linear system `Hx = −∂_θ∇V` is therefore consistent**, and its solution set is `−H⁺∂_θ∇V + ker H`. The arbitrary `ker H` component is exactly "where on the orbit you sit" — a gauge choice, *not* a numerical instability.

*Verification (s2b, analytic derivatives, machine precision).* For the shipped structure `V = g(r²) + ½k q₂²` the identity is one line — `∇V = 2g'(r²)(q₀,q₁,0) + k q₂ e₂` is **exactly radial in the channel**, hence so is `∂_θ∇V = 2 ∂_θ g'(r²)(q₀,q₁,0)`, for **any** parametrization of `g`. Measured `|⟨∂_θ∇V, ξ⟩| / ‖∂_θ∇V‖ ≤ 1.35e-16` across 10 parameters including every MLP weight/bias tested. *(s2's earlier 1.3e-4 was a nested-finite-difference floor, not physics — recorded honestly; my registered P7 bar of 1e-13 was met only after the instrument was fixed.)*
- **Goldstone mode exact:** eigenvalues at the vacuum `(4.93e-16, 1.0, 7.6909)`; `|λ_0|/λ_max = 6.4e-17`; the null eigenvector is the generator to `|cos| = 1.0`; and the transverse radial eigenvalue equals its closed form `4g''(r*²)r*²` to **15 digits**.
- **`H⁺` is the transverse implicit gradient, verified against ground truth:** `−H⁺∂_θ∇V` reproduces the measured *vacuum-manifold* displacement `dr*/dθ · r̂` to **1e-11…2.7e-8** for 8 parameters, with the `ξ`-component **exactly 0**. (Rider for the engineer: `rcond` matters — my first attempt used `rcond=1e-8` against a `1.02e-8`-relative Goldstone eigenvalue and failed by a factor 110. **Project explicitly onto `(ker H)^⊥` using the known generator; do not rely on an SVD threshold.**)

## Prop Q2.2 (the flat coordinate is an EXACT geometric sum — no unrolling) — **PROVEN + verified to 6.1e-14**
For exactly channel-invariant `V` and **isotropic** `M` on that channel, each Verlet substep preserves the angular momentum `L = q×p` (`q×∇V = 0` by invariance; `q' × p_half = (q + dt·∇T(p_half)) × p_half = q × p_half` because `∇T ∥ p_half`), and the friction multiplies it by exactly `(1−γ)`:
> ### `L_n = (1−γ)^n L_0` **exactly**, and `Δθ_n = arcsin(dt·L_n/(m r_n r_{n+1}))`, so
> ### `θ_∞ = θ_0 + dt·L_0 / (m r*² γ) + O((δr/r*)²)`   [`Σ_{n≥0}(1−γ)^n = 1/γ`]

*Verification (s2b, s6).* `L_n` vs `(1−γ)^n L_0`: max relative error **6.1e-14** over 300 steps (5.8e-16 at n=100). **Also exact under the relativistic kinetic** — max rel err **9.1e-14 / 4.6e-14 / 5.5e-14** at `c = 5 / 1 / 0.5` i.e. up to `v/c = 0.51` (because `∇T ∥ p` for any `T(pᵀM⁻¹p)`). **Breaks under anisotropic `M`:** 2.5% error by n=150 at a 2.5× mass anisotropy ⇒ §7 request 10. Closed-form `θ_∞`: relative error on the *displacement* **2.2e-5 … 7.5e-3** over `p₀ ∈ {0.05,0.1,0.2,0.3} × γ ∈ {0.02,0.05,0.1,0.3}` (best at large γ / small kick), i.e. **P9 confirmed far inside its registered 5% bar.**

## Prop Q2.3 (is the loss invariant along the designed flat direction?) — **REFUTED, and necessarily so**
The task asks me to prove or refute invariance. **Refuted — by construction, for the use case the pillar is about.** `∂L/∂θ = (∂L/∂q*)ᵀ ∂q*/∂θ`; the gauge (`ker H`) part of `∂q*/∂θ` contributes `(∂L/∂q*)ᵀξ`, which vanishes **iff** `L` is invariant along the orbit. But a *manifold-valued memory* is precisely a loss that reads the orbit coordinate: if `L` were invariant, the flat direction would store nothing and pillar (d) would be vacuous. Hence:
- **nuisance/gauge flat mode** (`L` invariant): pure quotient works; conditioning = transverse `λ_min`; nothing else needed.
- **storage flat mode** (`L` not invariant — pillar (d)): the fixed-point equation is **silent** in that direction and IFT genuinely does not apply there. The information does not come from `∇V = 0`; it comes from **transport** (Prop Q2.2), which is closed-form. ⇒ **The two candidate resolutions in the task are answers to two different objects, and the pillar needs the hybrid.**

**Where the learning signal for stored coset content actually lives (measured).**
| channel | `∂θ_∞/∂·` measured | closed form | note |
|---|---|---|---|
| `p₀` scale (i.e. `M⁻¹p₀`) | **1.0083** | `dt/(m r* γ) = 1.0171` | the **primary** channel (0.9% agreement) |
| `f` (vacuum radius) | −0.2088 | `−(θ_∞−θ_0)/r* · ∂r*/∂f`, ratio **0.9896** | `V` acts **only** via the orbit radius |
| `α` (confinement) | +0.0522 | ratio **0.9896** | ″ |
| `ε` (MLP residual weight) | +0.01735 | ratio **0.9867** | ″ |
| `W3[0,0]` (an MLP weight) | +0.001397 | ratio **0.9868** | ″ |
| `λ` (quartic stiffness) | −0.003013 | ratio **0.8568** | largest curvature change ⇒ largest transient error |
| `b3` (MLP output bias = pure `V`-value shift) | **0.0 exactly** | 0 | cannot move `∇V` |
| `k_spec` (spectator stiffness) | **0.0 exactly** | 0 | decoupled |

⇒ `∂θ_∞/∂θ_V = −((θ_∞−θ_0)/r*)·∂r*/∂θ_V`, with `∂r*/∂θ_V` the **well-conditioned transverse implicit gradient you are already computing.** My registered P10 predicted the `V`-channel would be `<1e-2` of the momentum channel; measured **0.207** ⇒ **P10's number is FALSIFIED**, while its structural claim (closed form, `O(1)`, no `1/λ_min`) is *confirmed and now quantitative*. This is the better outcome: `V_θ` **does** get a gradient for coset content, and it is exactly the transverse one.

## Prop Q2.4 (the pseudo-flat band is the only ill-conditioned regime — and it stores nothing extra)
Shipped `RingRegisterPotential` tilts the ring by `b(1−cos Kθ)`, giving tangential curvature `λ_tan = bK²/r*²` (verified: **3.31035** measured vs **3.31035** formula at `b=0.05, K=8`; and 0.0066208 vs 0.0066207 at `b=1e-4`):
| `b` | `λ_tan` | `cond(H)` | what it stores along the orbit |
|---|---|---|---|
| 0.05 (**shipped**) | 3.310 | **7.69** | `K = 8` discrete slots · **2.32× softer than λ_max ⇒ N46's *emergent* band (R-2)** |
| 0.01 | 0.662 | 11.6 | `K` slots |
| 1e-3 | 0.0662 | 116 | `K` slots |
| 1e-4 | 0.00662 | 1162 | `K` slots |
| **0 (exact)** | 4.9e-16 | 9.6e7 (naively) → **7.69 transverse** | a **continuum**, and the conditioning is cured by Prop Q2.1 |
> **Recommendation (and this is the Q2 deliverable): be exactly flat, architecturally, or comfortably massive. The intermediate band buys `1/b` conditioning and no extra storage.**

**Cost of the recommendation (honest accounting).**
1. **The symmetry must be designed, not learned** — `ker H` must be *known* (the generator), because Prop Q2.1's projection uses it and an SVD threshold is unreliable (the `rcond` failure above). Shipped `SO2InvariantPotential` already gives this for free (parametrized by `r²`, so non-invariant terms are **inexpressible**). Consistent with N46 (designed-only) and CM-16(a); **not** a new claim about emergent symmetry.
2. **`M` must be isotropic on the symmetry channel** (else Prop Q2.2's transport law loses 2.5% at 2.5× anisotropy). One constraint on the learned mass, at zero expressivity cost on that channel.
3. **Per-read extra cost:** one `d×d` symmetric solve on the projected Hessian (already needed) + one scalar per orbit dimension (`L₀`, `r*`). `O(1)`. **No unrolling. No ridge for the flat direction.**
4. **What you must still ridge:** *approximately* flat modes that are **not** protected by an exact symmetry — the generic learned case, and the **emergent** case (N46: mid-spectrum massive, `1.7–4.9×`). Use §Q4's `λ_ridge = 0.354`.
5. **What you give up:** the coset coordinate cannot be learned by shaping the radial profile at fixed radius (the `b3`/`k_spec` zeros above). If the writer needs to *place* content on the manifold, it must do so through `M⁻¹p₀` (the read-in/momentum channel) — which is a *design* consequence, and it reads on collapse mode #7: `M` and `p₀` separately are an exact gauge, but their **combination `M⁻¹p₀` is the channel**, and here it is the *only* channel. Prop F1 and pillar (d) are consistent; the channel is the velocity, not the mass.

**Verdict.** Proven: Q2.1, Q2.2 (both with machine-precision verification), Q2.3's refutation. Evidenced (toy scale, 3-D, one symmetry group): the quantitative composite law of Q2.3's table and Q2.4's band. **Conjectured:** that a *learned* `V_θ` with permitted basin interaction develops flat directions that are *approximately* symmetric enough for the projection to work without knowing the generator — I did **not** test this and would not rely on it.

---

# Q3 — Conditioning on the operating set, and the reach ↔ conditioning question

Geometry: `readout-channel-theory`'s family, `V = −D e^{−‖q−z‖²/2s²} + α‖q‖²`, `D=1, s=0.30, α=0.05` ⇒ `β = D/(2αs²) = 111.1`, `κ_stat = 4.1077`, `L_crit = κ_stat·s = 1.2323` (my own reach criterion (U), 31/32 on the trained shipped `V`).

## Q3.1 The item's own well is **well-conditioned across the reach boundary** (P11 confirmed)
`λ_min` at the item's own minimum, sweeping `L = 0.9 → 2.0` straight through `L_crit`: **11.199, 11.196, 11.193, 11.191, 11.190, 11.190, 11.189, 11.189, 11.189, 11.188, 11.186, 11.178, 11.152** — i.e. **constant to 0.13%**, and equal to `D/s² + 2α = 11.21` as expected. The spurious minimum and saddle are born at `L_crit` exactly as (★) predicts, and the *spurious* minimum's along-ray eigenvalue → 0 with the **saddle-node** exponent: `0.002537, 0.004602, 0.007889` at `ΔL = 3e-5, 1e-4, 3e-4` ⇒ local exponent **0.495, 0.490** (root gap exponent **0.505**). *(The full-range fit gives 0.446 — contaminated by crossover into the transverse `2α` mode; quoting the local exponent, per the standing "quote the curve" rule.)*

## Q3.2 The address/separatrix sub-mode is **invisible to `λ_min`** and **visible in the residual**
Sweeping the launch point across the payload-zero manifold (`a = 0.8 … 1.4` at `L = 1.3`, whose `(★)` saddle sits at `R₂ = 1.0531`) at `γ = 0.05, 0.3, 0.9`:
- `λ_min` at the reached fixed point: **11.1862 at every single cell** (six significant figures, 75 cells).
- **residual `‖∇V‖` after the shipped 400-step phase-1 budget: 8.9e-15 → 2.0**, i.e. **14 orders of magnitude** across the same sweep (at `γ=0.3`: 8.9e-15 at `a=0.8`, 5.6e-5 at `a=1.05`, **1.99 at `a=1.125`**).
⚠ **Honest scope:** in this 2-D geometry with an unlimited post-budget convergence the deep well captured *every* launch, so **this probe did not exhibit a basin flip** — the flip evidence comes from Q3.4 (item death: settled point jumps 0.18 → 1.53) and from the program's own 31/32 flow test, not from here. What the probe does establish cleanly is the **decoupling**: the address-side failure has no `λ_min` signature and a huge residual signature.

## Q3.3 Interacting basins: the merger band **is** the ill-conditioned region, and `d_safe` protects it
Two equal wells (`s = 0.35, α = 0.05`), spacing `d` swept (s3, s3b):
| `d/s` | `λ_min` | `‖H⁻¹∂_A∇V‖` | note |
|---|---|---|---|
| 4.63 | 8.249 | 0.00974 | effectively isolated |
| **4.40 (= `d_safe`)** | **8.244** | **0.00933** | **99.9% of the isolated `λ_min`** |
| 3.49 | 8.025 | 0.0106 | |
| 2.57 | 5.922 | 0.0681 | |
| 1.89 | **1.262** | **1.368** | **147× amplification of the implicit gradient** |
| 1.91 (`d_crit = 0.67`) | → 0 | → ∞ | **pitchfork**, exponent **1.03** |
| < 1.91 | 8.2 (of the *merged* well) | small | the two items are now one |
> **⭐ Proposition Q3.3.** The designed admission gate `d ≥ d_safe = 4.4s` (C1 §C5-A1) is **simultaneously a conditioning guarantee**: it holds `λ_min` at 99.9% of its isolated value. Conversely, charter §8.2's *permitted basin interaction* is precisely the move that spends this margin — and the price is explicit and monotone: `λ_min` falls ~6.5× and the implicit gradient rises ~147× before the merger.

## Q3.4 Lifetimes meet trainability: **decaying items are gradient bombs** (new, and actionable)
Decaying one item's amplitude at fixed `d = d_safe` (s3, s3b):
`A₁ = 1.0 → λ_min 8.244, ‖H⁻¹g‖ 0.0093` · `0.5 → 4.153, 0.0367` · `0.2 → 1.676, 0.220` · `0.1 → 0.815, 0.863` · **`0.05 → 0.299, 4.33` (465× amplification)** · `0.04 = A_crit` (measured; formula `2α|c|s√e = 0.0444`, 10% agreement) · `0.03 → the item is GONE`: the settle lands 1.53 away, in the neighbour's well, and `λ_min` reads the *neighbour's* 8.25 (saddle-node exponent measured **0.573**).
⇒ **An item approaching the end of its designed lifetime is the worst-conditioned object in the store, and then its contribution to the loss jumps discontinuously.** Nobody has stated this; it is a direct interaction between the lifetimes dial and the trainer. §7 request 9.

## ⭐ Q3.5 Verdict on the task's unification question
> **"Is reach failure (mode #11) the same object as implicit-gradient ill-conditioning?"** — **Refuted as stated; half of it is a theorem.**
> - **Same object:** the **well-loss** half of #11 — a well annihilated by decay (Q3.4), by merger (Q3.3), or the spurious-minimum annihilation at `L_crit` (Q3.1). There `λ_min → 0` with a bifurcation exponent, and the implicit gradient diverges as `1/λ_min`. One monitor serves both.
> - **NOT the same object:** the **address/separatrix** half of #11 — which `readout-channel-theory` showed is *the binding mechanism on the trained shipped `V`* (31/32, the (R/U) row). There `λ_min` is constant to six figures. It is a **wrong-basin / loss-discontinuity** event, detected by the **residual** and by **basin identity**.
> ### ⇒ The trainer's health check is a TRIPLE, computed from objects the read already produces:
> **(i) `‖∇V(q_N)‖ < tol`** — *am I at a fixed point at all?* (14 orders of dynamic range across the reach boundary) · **(ii) `λ_min(H) > λ_floor`** — *is it a nondegenerate minimum, and is the solve conditioned?* (`λ_floor = λ_ridge = 0.354` shipped) · **(iii) basin identity** — *is it the right one?* (the C1 nearest-entry check, `d* < d_capture`).
> **Monitor #11 does not double as the trainer's health check; the union does, and #11 supplies one leg.** And C1.4/C4.3's `λ_min > 0` guard **is** needed on the gradient path — upgraded from `> 0` to `> λ_floor`, because between 0 and 0.354 the mode is not resolvable in the read budget anyway.

---

# Q4 — Truncated backprop over trajectory reads: the number

## Prop Q4.1 (the per-step multiplier) — **PROVEN + verified**
Per Hessian eigenvalue `λ` with `u = dt²λ/m`, the two multipliers solve `λ² − (2−γ)(1−u/2)λ + (1−γ) = 0`, whose **product is exactly `1−γ`**. Hence
> - **underdamped** (`λ > λ_crit = γ²m/(2(2−γ)dt²)`): complex conjugate pair, `|λ| = √(1−γ)` **exactly — independent of `dt`, `m` and `λ`.**
> - **overdamped** (`λ < λ_crit`): `ρ ≈ 1 − (2−γ)dt²λ/(2γm)` → **1** as `λ → 0`.
> - **near a saddle** (`λ < 0`): `ρ_+ > 1` — gradients **grow** backwards in time.

*Verification (s4, s5).* `√(1−γ)` holds to **4.4e-16** across `γ ∈ {0.02,0.05,0.1,0.3} × dt ∈ {0.02,0.05,0.2,0.5} × λ ∈ {1, 8.24, 50}` for every underdamped cell. On the real shipped ring geometry (`λ = 8.20/8.24`), the *measured* per-step decay of `‖∂z_N/∂z_{N−k}‖` is **0.99008 / 0.97479 / 0.94890** at `γ = 0.02/0.05/0.1` vs `√(1−γ) = 0.98995 / 0.97468 / 0.94868`; at `γ = 0.3` the well mode is **overdamped** (`λ_crit = 10.59 > 8.24`) and the measured 0.92935 matches the exact quadratic root 0.9298, not `√0.7 = 0.8367`.

## ⭐ Prop Q4.2 (the truncation depth) — **the number the engineer will use**
> ### `k*(γ, ε) = ln(1/ε) / ln(1/ρ)`,  `ρ = max( √(1−γ), 1 − (2−γ)dt²λ_min/(2γm) )`
> ### Underdamped-dominated form: `k* = 2 ln(1/ε) / ln(1/(1−γ)) ≈ 2 ln(1/ε)/γ`.
| `γ` | `ρ` (shipped λ) | **`k*` at ε=1e-3** | ε=1e-2 | ε=1e-5 | contribution kept at k=270 |
|---|---|---|---|---|---|
| 0.02 | 0.9900 | **684** | 456 | 1140 | 93% |
| **0.05 (shipped address phase)** | **0.9748** | **269** | **180** | **449** | 99.9% |
| 0.10 | 0.9489 | **131** | 87 | 219 | ≈1 |
| 0.30 | 0.9290 (overdamped) | **94** | 63 | 156 | ≈1 |
| soft mode `λ=0.354, γ=0.10` | 0.9908 | **750** | 500 | 1250 | 8% |
| soft mode `λ=0.01, γ=0.02` | 0.99868 | **5216** | 3477 | 8693 | ≈0 |
> ### ⇒ **At the shipped address phase (`γ=0.05`, `dt=0.05`), unroll depth beyond ≈270 steps is numerically worthless (<0.1% of the gradient); the shipped `address_steps = 400` is already 1.5× deeper than useful.** γ-dependence: `k* ∝ 1/γ` while the well modes stay underdamped, and `k* ∝ 1/λ_min` once a mode goes overdamped.

## Prop Q4.3 (two structural riders that change the recommendation)
1. **The shipped phase 2 runs at `γ_read = 0.0`, where `ρ = 1.0` exactly** (measured). There is **no** geometric death, so truncating inside the read window loses `O(k/N)` of the signal, not `O(ε)`. ⇒ **Put the truncation boundary at the phase-1/phase-2 junction:** backprop the **entire** `γ=0` read window (800 steps as shipped), plus **≤270 steps** of the `γ>0` address phase, and use the **implicit/analytic** gradient for everything earlier. *The theory's natural truncation point is the architecture's existing seam.*
2. **Saddle passage inverts the sign of the problem.** `ρ_+ = 1.0045 / 1.0310 / 1.1526` for `λ_neg = −0.1 / −1 / −8.24` at `γ=0.05` ⇒ **514 / 75 / 16 steps per decade of gradient growth**. The two-sided constraint (keep ≥ (1−ε) of the settle gradient **and** cap amplification at 100×) has **no solution** for a well-scale unstable mode: `k_max = 38 < k_needed = 269` (conflict in 15 of 18 grid cells; only `γ≥0.1` with a *stiff* `λ_min` and a *weak* saddle is conflict-free). ⇒ **Prefer the implicit path wherever a fixed point exists; if you must unroll through a saddle, clip in time** (§7 request 8). This also re-explains N62's "a conservative read has no vanishing-gradient problem": at `γ=0`, `ρ=1` — the positive half of N62 is exactly the `ρ=1` statement, and it is *unchanged* by this analysis.

## Prop Q4.4 (settle-budget law — a free by-product that prices the C2 compute constraint)
Since `ρ = √(1−γ)` is `dt`- and landscape-free for underdamped modes,
> ### `N_settle ≈ 2 ln(1/tol) / γ`, independent of `dt`, of `M`, and of the landscape.
`γ=0.05`: **553** steps for `tol=1e-6`, **276** for `1e-3`. `γ=0.02`: 1382. `γ=0.1`: 276. `γ=0.3`: 92.
⇒ **The observed "150–1200 Verlet steps per read" band is not a tuning artifact; it is `2ln(1/tol)/γ` at `γ ≈ 0.05` for `tol = 1e-2 … 1e-9`.** Two consequences for charter §2.2's compute constraint: **(a)** you cannot buy settle speed with `dt` (`u` is 194× below the stability limit but `ρ` is `dt`-free) — only with `γ`, and **collapse mode #1 is the price** (the CAFE note's measured `corr(q*, q_last) 0.56 → 0.92` as `γ: 0.05 → 0.5` is this trade); **(b)** the **anytime accuracy-vs-steps curve is the `tol` ladder**, with a knee at `N ≈ 2/γ` — i.e. the signature figure has a *derived* shape, which is a stronger claim than an empirical curve.

## Prop Q4.5 (P14 settled: the C1 `2.2e-12` is an instrument floor) — **see R-1**
Exact Jacobian product along the same trajectory, same `N = 3000`, float64, log-accumulated: `log10‖J‖ = **−32.91**` at `γ=0.05` (`(1−γ)^{N/2}` predicts −33.41) and `**−68.15**` at `γ=0.1` (predicts −68.64). A central-difference probe at `h=1e-5` has a floor near `2e-11`. **⇒ C1's `2.2e-12` was the FD floor; the true state-space contraction is ≥21 orders smaller, and it is a law (`(1−γ)^{N/2}`), not a constant.** The C1 conclusion ("gradient search for an address is dead"; N61) is *strengthened*, not weakened.

---

# Q5 — Consolidation (repositioned wake–sleep)

## Q5(i) Does consolidation need gradients? — **No, as specified. It needs one linear solve.**
| consolidation op (charter §2.4) | gradient needed? | what it needs instead | authority |
|---|---|---|---|
| **re-packing** (relocate items) | **no** | a placement solver + the gauge theorem: assignment is *exactly* free among assignments satisfying N1–N4 (C1: 10 random permutations, retrieval spread **0.0**) | C1 §C2 (verified) |
| **decay enforcement** | **no** | the closed-form decay law (banked, 6.8e-8) + an `A_crit` floor (Q3.4: `2α|c|s√e`, measured 0.040 vs 0.0444) | banked + this report |
| **gate re-calibration** | **no** | a 1-D `σ` estimator + the erf margin law `acc ≈ erf(margin/√2σ)` (C1: within 0.021) | C1 §C2-N2 |
| **admissibility of every move** | **no** | `‖H_i⁻¹∇δV(q*_i)‖ ≤ δ_budget` — a **linear solve**, the same factorization the implicit gradient uses | C1 §C3 (median ratio 1.0002) |
⇒ **C1's verdict stands: the consolidator is a certifier + allocator, not an optimizer.** The economical consequence: **build the `H`-solve once** — the trainer's implicit gradient, the controller's C3 admissibility check, and the `λ_min` monitor are three uses of one Cholesky/eigendecomposition of a `d×d` matrix (`d = 4…8`, so this is free). **Boundary named honestly:** if "re-packing" is ever extended to *re-fitting `φ`/`ψ` on replayed items*, that **does** need gradients — ordinary supervised ones plus the implicit gradient — and then everything in §Q1–Q4 applies to it.

## ⭐ Q5(ii) Does consolidation re-expose N5? — **Not as specified; and the Ward identity explains *why* N5 happened**
> ### Prop Q5.1 (N5 is a corollary of Prop Q2.1). A settle-based loss has **exactly zero** gradient along the coset direction — that is the same orthogonality `⟨∂_θ∇V, ξ⟩ = 0` that makes the implicit system consistent. **The flat direction is therefore undefended by construction:** any *other* term in the objective (contrastive divergence's negative phase, a value-sensitive regularizer) has an **unopposed** gradient there. One theorem, two consequences — one good (trainability), one N5.
This is exactly N5's recorded mechanism ("*CD has no anchor on `V`'s value; the wake trajectory-MSE cannot see a flat coset direction, so sleep fills it in*"), now **derived** rather than observed. It also predicts N5's **demarcation**: the non-degenerate Exp-B sine vacuum is immune because there is no `ξ` with `⟨∂_θ∇V, ξ⟩ = 0`.

**Is consolidation-as-maintenance exposed?** As specified in charter §2.4 (re-pack / decay / gate re-cal) there is **no negative phase**, so N5 is **not** re-exposed by construction. But two of the three sub-ops *can* move `V` in the vacuum band, so the exposure is conditional and the guards are specific:
1. **⭐ Architectural invariance (the strongest guard, already shipped).** Parametrize the flat block by invariants (`SO2InvariantPotential`: a function of `r²`). Then a symmetry-breaking term is **inexpressible**, and N5 becomes impossible *for the flat direction* — not merely unlikely. (What remains possible is **vacuum collapse**, `r* → 0`: N5's measured signature was `ring depth +0.079 → −0.126`, `r* → 0`, which in this language is the **transverse radial eigenvalue crossing zero** — a symmetry-restoring transition, and therefore visible in the *transverse* `λ_min` the trainer already monitors.)
2. **Anchor, if any generative/negative-phase term exists in a consolidation pass:** carry N5's measured cure, the `V(data)` energy anchor `λ=10` (ring depth flat `+0.068…+0.072` over 1000 ep, `r* ≈ 0.96`, noise_gap `+0.244 > +0.199` wake-only). Energy-gating on `H` is on record as **failing** (kinetic energy defeats the gate) — do not substitute it.
3. **Per-pass monitor with hysteresis:** `r*` (orbit radius) and `λ_transverse`. Abort/roll back a consolidation pass on a `r*` drop beyond the deadband — the C4 deadband transplanted to the structural verb.
4. **Invariant-only decay on symmetry blocks.** Decay is *by design* a change to `V`'s value; if applied non-invariantly to a flat block it **is** a spurion (`μ² > 0`), the manifold collapses to `K` slots, and you land in the pseudo-flat band Q2.4 says to avoid. Decay on such a block must be a function of `r²` only.
5. ⚠ **"Consolidate less often" is NOT a guard — N5's own numbers say the opposite.** Inversion epochs 116 / 442 / 959 at CD frequency f = 1 / 5 / 20 convert to **116 / 88 / 48 sleep *updates***. Rarer consolidation inverts the vacuum in **fewer** updates, i.e. per-update damage *grows* with rarity. ⚠ **This is a re-analysis of recorded numbers, not a new measurement, and it has a confound** (the f=20 arm accumulates more wake-only epochs between sleeps, so the two arms are not the same model at the moment of the update). Flagged as **evidenced-by-reanalysis**; worth one cheap re-run before it is quoted as a law.

---

# 6. Verdict table

| # | claim | status |
|---|---|---|
| Q1.1 | `Fix(T_θ) = {(q,0): ∇V=0}`, all `γ∈(0,2)`, both kinetics | **proven** + verified 4.4e-15 / 2.7e-13 |
| Q1.1b | `γ=0`: same set, `det ∂T/∂z = 1`, Liouville ⇒ no attractor ⇒ no settle | **proven** + verified (`ρ=1.0`) |
| Q1.2 | `det(I−∂T/∂z) = ((2−γ)dt²/2)^d det(M⁻¹HessV)`; invertible ⟺ `HessV` nonsingular | **proven** + verified 2.07e-11 |
| Q1.2b | `σ_min(I−J) = (2−γ)dt²λ_min/(2√(m²γ²+dt²))` (soft mode); `κ(I−J) ≈ O(1)κ(H)` | **corrected law, verified to 3 digits** (P4 falsified) |
| Q1.2c | stability ⟺ `0 < dt²λ_i/m < 4`, γ-independent | **proven** + verified at the boundary |
| Q1.3 | `∂q*/∂θ = −H⁻¹∂_θ∇V`, `∂p*/∂θ=0`, **exact for the discrete map**, no `(γ,dt,M)` correction | **proven (2 proofs)** + verified 1.3e-8, spread 5.5e-14 |
| Q1.4 | degenerate set is codim-1 ⇒ crossed; pitchfork (exp 1.03) vs saddle-node (exp 0.49–0.50); loss piecewise smooth with jumps | **proven** (codim) + **measured** (exponents, jump 0.18→1.53) |
| Q2.1 | Ward identity ⇒ implicit system consistent ⇒ `H⁺` valid, transverse conditioning | **proven** + verified 1.35e-16 |
| Q2.2 | `L_n = (1−γ)^nL_0` exact; `θ_∞` closed form | **proven** + verified 6.1e-14 (9.1e-14 relativistic) |
| Q2.3 | loss is **not** invariant along a *storage* flat direction; content rides `M⁻¹p₀`; `V` acts only via `r*` | **proven** (necessity) + **measured** (ratios 0.987–0.990) |
| Q2.4 | pseudo-flat band is the only ill-conditioned regime and stores nothing extra | **evidenced** (one group, toy scale) |
| Q3.5 | reach↔conditioning: same object for well-loss, **not** for the address sub-mode; health check is a triple | **refuted-as-stated + salvaged**, measured |
| Q3.3 | `d_safe = 4.4s` preserves 99.9% of isolated `λ_min` | **measured** |
| Q3.4 | decaying items are gradient bombs (465× at `A=0.05`), then discontinuity | **measured** |
| Q4.2 | `k* = ln(1/ε)/ln(1/ρ)`; **269 at shipped γ=0.05, ε=1e-3** | **proven** + verified (rate 0.97479 vs 0.97468) |
| Q4.4 | `N_settle ≈ 2ln(1/tol)/γ`, `dt`-free; anytime curve knee at `2/γ` | **proven** + verified |
| Q4.5 | C1's `2.2e-12` is an FD floor; true value `10^-32.9` at γ=0.05, N=3000 | **measured** (R-1) |
| Q5.1 | N5 is a corollary of the Ward identity | **proven** (mechanism); the cure is N5's measured anchor |
| — | Q5(ii)-5 "rarer consolidation is worse per update" (116/88/48) | **evidenced by re-analysis, confounded — do not quote as a law yet** |
| — | that a *learned* `V_θ` grows flat directions symmetric enough to project without a known generator | **conjectured, untested** |

---

# 7. Implementation requests (for `experiment-engineer`, one line each)

1. **Solve the `d×d` Hessian system `H x = −∂_θ∇V`, not the `2d×2d` `(I − ∂T/∂z)` system** — provably the identical answer (Prop Q1.3), half the dimension, and simpler; skip `∂p*/∂θ` entirely (it is exactly 0).
2. **Ridge with zero free parameters: `λ_ridge = 2γ m ln(1/tol) / (N_settle (2−γ) dt²)`** = **0.354** at shipped (`γ=0.05, dt=0.05, N=400, tol=1e-3`); costs 4.1% bias on a healthy well mode, caps amplification at 2.82×. **Alarm (do not silently proceed) if `λ_ridge > 0.1 × median(λ)`** — that means the read budget is too small for the landscape (e.g. `γ=0.1, N=150` gives `λ_ridge = 1.94`).
3. **⭐ Register this gradcheck tolerance (implicit vs truncated unroll)** — the disagreement has **two** predictable terms, `ρ^k` (truncation) and `‖∇V(q_N)‖/λ_min` (non-convergence), so the honest bar is depth-dependent:
   | unroll depth `k` (γ=0.05) | settle residual required | **expected agreement** | register |
   |---|---|---|---|
   | 180 | ≤1e-4 | 1e-2 | pass ≤ 3e-2 |
   | **270** | **≤1e-5** | **1e-3** | **pass ≤ 3e-3 ← recommended default** |
   | 449 | ≤1e-7 | 1e-5 | pass ≤ 3e-5 |
   ⚠ **Do not register 1e-5 at `k=270`** — the theory says that cannot be met. **Expect and report (not "fail") disagreement when `‖∇V(q_N)‖ > 1e-3` or `λ_min < λ_ridge`**: those are the two diagnosed regimes, and a disagreement there is a *measurement* of §Q3.5's triple, not a bug. My own analytic-vs-FD verification of the identity reached **1.3e-8**, so the theory is not the floor.
4. **Compute the §Q3.5 monitor triple at every read, from the factorization you already have:** `‖∇V(q_N)‖` (residual), `λ_min(H)` (and its sign), nearest-codebook-entry distance (basin identity). Log all three per read; they cost one `d×d` eigendecomposition.
5. **Project explicitly onto `(ker H)^⊥` using the known symmetry generator on designed flat blocks — never rely on an SVD `rcond`.** (My first attempt with `rcond=1e-8` against a `1.02e-8`-relative Goldstone eigenvalue was wrong by 110×.)
6. **Report basin-flip counts as a first-class training diagnostic, and do not use loss-decrease line searches / trust-region acceptance tests** — `L(θ)` is piecewise smooth with codim-1 jumps (§Q1.4), so the loss can legitimately jump upward after a good step.
7. **Truncate at the phase seam:** full backprop through the entire `γ_read = 0` window (`ρ = 1`, nothing decays) + **≤270** steps of the `γ = 0.05` address phase + implicit/analytic before that.
8. **Time-clip the retained unroll:** if any Hessian eigenvalue along the retained window is negative, clip the per-step gradient norm (`ρ_+ = 1.153` at a well-scale unstable mode ⇒ 16 steps per decade of growth, and no `k` satisfies both constraints).
9. **Lifetimes × trainability guard:** freeze/exclude an item's `θ`-gradient contribution once `λ_min,i < λ_ridge`, and **retire its codebook entry at the crossing** (measured: 465× gradient amplification at `A=0.05`, then a 1.53 jump into a neighbour's well at `A=0.03`; `A_crit ≈ 2α|c|s√e`).
10. **Lock `M` isotropic on any designed symmetry channel** (`m₀ = m₁` on the SO(2) pair) — a 2.5× anisotropy breaks the exact `L_n = (1−γ)^n L_0` transport law by 2.5% within 150 steps. The fixed-point *manifold* is unaffected (it is a `V`-property); only the closed-form coset transport is.
11. **Implement the coset read as project-and-transport:** transverse `∂r*/∂θ` from the (projected) Hessian solve, orbit coordinate from `θ_∞ = θ_0 + dt L_0/(m r*² γ)` with `L_0 = (q₀ × p₀)`; accumulate `Σ dt L_n/(m r_n r_{n+1})` during the settle if you want the exact value instead of the 0.02–0.75% closed form.
12. **Keep any manifold store out of the pseudo-flat band:** use the exactly-invariant potential class (`b = 0`) or a comfortably massive register; **do not** operate at `λ_tan/λ_max ∈ [0.01, 0.3]` (`cond` 116–1162 for no extra storage).
13. **Consolidation guards (§Q5):** invariant-only (`r²`-only) decay on symmetry blocks · `V(data)` energy anchor `λ=10` if any negative-phase term is ever added · per-pass `r*` + `λ_transverse` monitor with hysteresis · do **not** substitute "consolidate less often".
14. **Cheap opportunity, flagged not requested:** `dt` is 13.9× below the stability limit (`dt_max = 2√(m/λ_max) = 0.697`). This buys *ballistic reach per step*, **not** settle speed (`ρ` is `dt`-free). If reach ever needs a lever, `dt` is an unexploited one — but it changes the trajectory, so it is a design change, not a free win.

---

# 8. Open questions / risks (honest)

- **OQ-1 (the one I could not test).** Does a *learned* `V_θ` under capacity pressure develop flat directions that are approximately symmetric enough to project **without a known generator**? Everything in Q2 assumes the generator is designed and known. If the answer is no, pillar (d) is a **designed-only** capability (N46's precedent), and I would say so in the paper rather than discover it in a reviewer's question. Cheapest test: train the shipped store with permitted basin interaction, eigendecompose `H` at each settled item, and check whether the softest eigenvector is (a) stable across `θ` updates and (b) close to a Killing field of the fitted landscape.
- **OQ-2.** All Q2/Q3 verification is 2–3-D, one symmetry group, single-well or two-well. The *structure* of every proposition is dimension-free (all proofs are basis-free), but the **constants** (`d_crit/s = 1.914`, `A_crit`, `λ_crit = 0.256`) are geometry-specific. Do not quote them at `d = 6, 8`.
- **OQ-3.** `λ_ridge` is derived from a *linear* convergence criterion; near a bifurcation the dynamics is not linear (critical slowing is algebraic, not geometric), so the true resolvable-`λ` floor is *higher* than 0.354 exactly where it matters most. A cheap empirical calibration (residual vs `λ_min` on the trained store) would tighten it.
- **OQ-4.** The Q5(ii)-5 re-analysis (116/88/48 sleep updates) is confounded; a 2-arm re-run at matched *update count* would settle it, and it changes a guard, so it is worth one cheap job.
- **Risk.** Everything here is deterministic (`langevin_noise` off). A stochastic consolidation/sleep phase adds a noise floor to the residual monitor, which would mask leg (i) of the Q3.5 triple; the threshold must then be set relative to the injected `σ`, not absolutely.

---

# 9. Git footprint
**None.** No tracked file was read-modified; the repo stayed clean at `main @ 082d095`. All artifacts under `.claude/outputs/trainability-spike-theory/` (PREREG) and `.claude/scratch/trainability-spike-theory/` (7 scripts + 7 result JSONs). No worktree created. I did not touch `clu-controller-spec.md` or `readout-channel-theory.md` (both are superseded **by reference only**: R-1 amends the *interpretation* of one C1 number; C1's §C1/§C2/§C3/§C5 content is otherwise reaffirmed and extended).

---

# 10. Proposed handover updates (for the Hub)

**§1 (the physics / memory formalism) — add:**
- **The shipped-map implicit-function package (proven + verified):** `Fix(T_θ) = {(q,0): ∇V=0}` for `γ∈(0,2)`, both kinetics · `det(I − ∂T/∂z) = ((2−γ)dt²/2)^d det(M⁻¹HessV)` ⟹ invertible ⟺ `HessV` nonsingular · **`∂q*/∂θ = −H⁻¹∂_θ∇V` is EXACT for the shipped discrete map with no `(γ,dt,M)` correction** (verified 1.3e-8; prediction spread across `(γ,dt)` = 5.5e-14) · stability ⟺ `0 < dt²λ_i/m < 4` (γ-independent; shipped `dt` is 13.9× below it) · `γ=0` has the same fixed-point set but `det ∂T/∂z = 1` so **Liouville forbids a settle — the two-phase read is forced, not conventional**.
- **The settle-budget law:** `N_settle ≈ 2 ln(1/tol)/γ`, `dt`- and landscape-free (`ρ = √(1−γ)` exactly for underdamped modes, verified 4.4e-16). **This derives the observed 150–1200-step read cost and gives the anytime curve a knee at `N ≈ 2/γ`.** Settle speed is buyable only with `γ`, and collapse mode #1 is the price.
- **Project-and-transport (the Q2 resolution):** Ward identity `⟨∂_θ∇V, ξ⟩ = 0` (1.35e-16) ⟹ the singular implicit system is *consistent*, `H⁺` gives the transverse answer with transverse conditioning; the coset coordinate obeys the **exact** law `L_n = (1−γ)^n L_0` (6.1e-14; also exact relativistically) ⟹ `θ_∞ = θ_0 + dt L_0/(m r*²γ)` (0.02–0.75%). **Manifold-valued content rides `M⁻¹p₀`; `V_θ` reaches it only through the orbit radius `r*`.**
- **`λ_min` is the single number controlling implicit conditioning, truncation depth, bifurcation proximity, and budget-resolvability.** Ridge `λ_ridge = 2γm ln(1/tol)/(N(2−γ)dt²) = 0.354` shipped.
- **Truncation:** `k* = ln(1/ε)/ln(1/ρ)`; **269 steps at `γ=0.05, ε=1e-3`**; `ρ=1` at `γ_read=0` (so truncate at the phase seam); `ρ_+>1` at saddles (16 steps/decade of growth at well scale).

**§7 (discrepancies) — add:**
- **R-1: `∂R_γ/∂q₀ = 2.2e-12` / "11.3 orders of magnitude" is an instrument floor.** True value `10^-32.9` at `γ=0.05, N=3000` (`(1−γ)^{N/2}`). Conclusion (N61: address search is dead) **strengthened by ~21 orders**. Every quoting site needs the correction; the measurement itself is not retracted.
- **R-2: the shipped `b=0.05, K=8` register is 2.32× soft — inside N46's *emergent* 1.7–4.9× band.** It is not a flat direction and must not be cited as evidence for pillar (d); only exact architectural invariance (or `b=0`) is.
- **Reach ↔ conditioning is a HALF-identity:** same object for well-loss (merger/decay/spurious-min annihilation), *not* for the address/separatrix sub-mode that actually binds on the trained `V` (31/32) — there `λ_min` is constant to 6 s.f. while the residual moves 14 orders. Monitor #11 supplies one leg of a three-leg trainer health check.

**§8 (open questions) — add:**
- **OQ-1 (the gating one for pillar (d)):** does a *learned* `V_θ` produce flat directions projectable **without a known generator**? If no, pillar (d) is designed-only (N46 precedent). Test recipe in §8 above.
- **OQ-4:** N5's frequency-horizon converted to sleep *updates* (116/88/48) suggests **rarer consolidation is worse per update** — confounded; one cheap 2-arm re-run at matched update count settles a guard.
- **New cross-cutting item:** *lifetimes × trainability* — an item at the end of its designed life is the worst-conditioned object in the store (465× implicit-gradient amplification at `A=0.05`) and then contributes a **loss discontinuity**. The decay schedule and the trainer must share the `λ_min` floor.
