# PREREG — `doctrine-repairs` (physics-theorist, C2W2)

**Written 2026-07-30 BEFORE any script in `.claude/scratch/doctrine-repairs/` existed or was run.**
Protocol §5 pre-registration rule: the acceptance criterion of this task is a *prediction* (charter
§A2.5) plus several derived constants (`eps` for monitor #6, a corrected inradius proxy, a truncation
depth). Each is committed to here with the derivation that produced it.

Provenance of everything below: pure-numpy toy physics in `.claude/scratch/doctrine-repairs/`
(integrator = the shipped damped velocity-Verlet, line-for-line, copied from
`.claude/scratch/controller-doctrine/common.py`), plus **re-scoring of two already-published
artifacts** (`.claude/scratch/controller-doctrine/s5b_results.json`, 54 configs;
`.claude/outputs/memory-gym-v0/exp_memory_gym_metrics.json`, 28 cells). No repo module is imported.
**Every number is a THEORY check; none is a paper number.** Single seed unless stated.

---

## P-A ⭐ The headline: charter §A2.5, the P1 / merge-certificate adjudication

> **§A2.5 (charter, pre-registered by the Advisor):** *all 7 feasible P1 configs satisfy the merge
> certificate — the 13-band intersection exists only in the separable (provably-zero-dividend) regime.*

**P-A1 (scored on the ORIGINAL 54-point grid). CONFIRM.** All 7 zero-trip configs satisfy
`2·s_max + κ′·σ_q ≤ sep` with `κ′ = 2.576` (the 99 % erf constant; R1 of C2W1).
Derivation from `s5b_search.py`'s own constants, done by hand before running:
`s_max = W_ADDR · max_i ‖naxis_i‖_1st = 0.20 · 5.0/√13 = 0.27735`; `σ_q = 0.24` at all 7 clear cells;
`sep = 2R sin(π/6) = R` for the K=6 ring. So `LHS = 0.5547 + 0.6182 = 1.1729`, `sep ∈ {1.3, 1.5}`.
**Predicted margins: `+0.1271` at R=1.3 and `+0.3271` at R=1.5. Predicted score 7/7.**

**P-A2 (is it a theorem or a property of the grid?). It is CONDITIONAL, and the grid satisfies the
condition.** Monitor #8-N2 requires `sep ≥ 5.15 σ_q`. N2 therefore *implies* the merge certificate iff
`2 s_max + 2.576 σ_q ≤ 5.15 σ_q ⟺ **s_max ≤ 1.287 σ_q**`.
On the grid `s_max/σ_q = 0.27735/0.24 = 1.156 ≤ 1.287` at the tightest σ_q and smaller elsewhere ⇒
**predict: the implication N2 ⇒ certificate holds at ALL 54 grid points (54/54), so the 7/7 in P-A1 is
forced by N2 and is NOT independent evidence for §A2.5 as a general claim.** The grid clears the
condition by only **11 %**.

**P-A3 (the extension — the falsifier). PREDICT §A2.5 IS REFUTED OUTSIDE THE GRID.** The merge
certificate is *sufficient* for two wells to stay two wells; the *exact* condition is `λ_min > 0`
(monitor #8-N3), and for equal Gaussian wells the barrier survives to `sep ≈ 2 s` (midpoint curvature
`2A e^{−u²/2}(1−u²)/s² + 2α` with `u = sep/2s` changes sign at `u = 1`). With `σ_q` at its N2 maximum
`sep/5.15`, the certificate demands `s_max ≤ 0.25 sep` while non-merger only demands `s_max ≲ 0.5 sep`.
⇒ **I predict a non-empty region `s_max/sep ∈ (0.25, ~0.45)` in which all 13 bands hold and the merge
certificate is violated.** Point estimate: **3–15 % of an extended grid with `s_max/σ_q ∈ [1.5, 3.0]`
clears all 13 bands while violating the certificate; ≥ 1 witness exists.** If ≥1 witness exists,
**§A2.5 is refuted as a general statement and confirmed only as a property of the C2W1 grid** — and
per the task that is the bigger result (the intersection reaches into the non-separable regime).
**Falsifier of P-A3:** 0 witnesses in the extended grid *and* an identified band that forces the
certificate (then §A2.5 is a theorem on the wider region too).

## P-B ⭐ R2(gym): the `eps` dead-band for monitor #6 (BLOCKING a code fix this wave)

**Derivation.** `slope = polyfit(x, y, 1)[0]` over `n = window+1 = 4` points. Two floors:
1. **Roundoff floor** — for an exactly-constant series the true slope is 0 and the computed value is
   `O(c·u·Y)`, `u = 2^-53 = 1.11e-16`, `Y = max|y|` in the window. **Predict the measured envelope
   constant `c ≈ 2.3` (bound `c ≤ 8`)**, from the gym's own artefacts (`slope_loss = −5.193e-17` at
   `Y ≈ 0.20`; `slope_acq = −5.934e-17` at `Y ≈ 0.235`).
2. **Resolution floor** — a slope whose extrapolated change over the window is below the quantity's own
   resolution is not measurable. `acq` is a proportion over `n_probed` items ⇒ quantum `1/n_probed` ⇒
   `eps_acq = 1/(n_probed·(n−1))`. For the loss use a relative resolution `f_rel = 1e-3` of the window's
   own scale ⇒ `eps_loss = f_rel·Y/(n−1)`.

**Predicted spec:** `eps_loss = max(8·u·Y, 1e-3·Y/(n−1))`, `eps_acq = max(8·u·Y_acq, 1/(n_probed·(n−1)))`,
and the predicate becomes **`slope_loss < −eps_loss AND slope_acq ≤ +eps_acq`** (the acq leg's dead-band
is *permissive*: "not rising" must tolerate `+1e-17`, or the same defect produces false NEGATIVES).
**Predicted numbers:** at the gym's scale (`Y ≈ 0.18`, `n−1 = 3`) `eps_loss ≈ 6.0e-5`; the artefact
population's largest |slope| is `5.2e-17` and the genuine population's smallest is `3.7e-4`
⇒ **a ≈ 12.9-order-wide admissible window; the trip table is invariant for any eps in
[1e-15, 3.7e-4]**.
**Predicted effect on the 28 published gym final readings: exactly 6 of the 13 current #6 trips are
removed** (`load1x_ref8@s0`, `load1x_ref3@s0`, `load1x_shipped@s0`, `load1x_shipped@s2`,
`manifold/base@s0`, `manifold/base@s2`) **and exactly 2 new trips appear** (`overload/base@s0`,
`overload/reach_free@s0` — false NEGATIVES today, `slope_loss = −4.0e-2` with `slope_acq = +7.8e-4`).
**0 genuine trips lost.**

## P-C ⭐ R3(gym): monitor #3's validity leg (BLOCKING a code fix this wave)

**Predicted diagnosis (mechanism, not statistics).** The recorded `gate_margin` is a *distance*; the
harm it must predict is the first-order crowding drift
`Δq_i ≈ ‖∇δV_j(q_i*)‖ / λ_min,i` with `‖∇δV_j(c_i)‖ = A_j (d_ij/s_j²) e^{−d_ij²/2s_j²}`.
Three named reasons a distance margin cannot rank that: **(i)** the map `d ↦ d·e^{−d²/2s²}` is
**non-monotone** (peak at `d = s`), so the sign flips whenever some spacings are below `s_j`;
**(ii)** `A_j, s_j` vary per learned write (N120: `corr(s_fit,|a|) = +0.821`) and do not enter the
margin; **(iii)** `λ_min,i` varies by orders and **is negative at most gym cells** (measured
−1.376 … +3.411) — where `λ_min ≤ 0` the site is not a minimum and "drift" has no first-order theory
at all, so the leg is *inapplicable*, not failing.
**Predicted from the published artifact (re-scoring, no new run):** the three cells with a converged
write and genuine minima (`overload/load1x_shipped@s{0,1,2}`, `λ_min = 3.30/3.01/3.41`) have
**validity_corr > 0 on all three** and **≥ 0.30 on at least 2 of 3**; the sign-unstable population is
concentrated at `λ_min ≤ 0`. Predicted point-biserial/rank association between `λ_min > 1` and
`validity_corr`: **positive, with the λ_min>1 group mean ≥ +0.35 vs the λ_min≤0 group mean ≤ +0.05.**
**Predicted replacement predicate (the deliverable):** the leg is replaced by the **C3 first-order
calibration test** — for each admitted write `j` and each of the `k` nearest incumbents `i`,
`B_ij = ‖∇δV_j(q_i*)‖ / max(λ_min,i, λ_floor)` vs measured `Δ_ij = ‖q_i*(after) − q_i*(before)‖`
**at the relaxed fixed point** — with **soundness** (`P[Δ > κB] ≤ η`, `κ = 2` from N74's measured
0.73–1.63 spread over 4.6 decades, `η = 0.05`) and **tightness** (`median Δ/B ≥ 1/10`) legs, and
**INAPPLICABLE whenever `λ_min,i ≤ λ_floor`** (that failure belongs to #8-N3 and must not be
double-counted).
**Predicted verification numbers (toy, heterogeneous designed store with λ_min > 0):**
`corr(margin, drift)` unstable with |corr| ≤ 0.6 and a sign flip across ≥1 of 8 seeds;
**`corr(B, Δ) ≥ 0.90` on every seed**, soundness violation rate `≤ 0.05` at `κ = 2`, and the toy that
puts some spacings below `s` (the non-monotone branch) produces `corr(margin, drift)` **> 0 → < 0**.

## P-D R1: the `sep/2` inradius proxy

**Predicted corrected proxy** (equal-width two-well saddle shift, leading order in `ln(A_i/A_j)`):
`r_i ≈ min_j [ d_ij/2 + (s²/d_ij)·ln(A_i/A_j) ]`, valid domain `λ_min,i > 0` **and** `d_ij ≥ 2 s`
**and** `|(s²/d)ln(A_i/A_j)| ≤ 0.25 d` (small-shift regime). Outside it: **`r_i := 0` when
`λ_min,i ≤ 0`** (no minimum ⇒ no certified ball) and measure by bisection otherwise.
**Predicted numbers on `s2b`'s published store** (K=6 ring, R=2 ⇒ `sep = 2.0`, `s = 0.35`,
`A ∈ [0.62,1.34]`, measured `r = [0.9539, 0.9930, 1.0047, 0.9539, 0.9516, 1.0445]`): the `sep/2 = 1.0`
proxy's max error is **0.0484**; **predict the corrected proxy's max error < 0.015 (≥ 3.2× better)**.
**Predicted structural statement:** monitor #2's `U` is computed with the *over-claimed* radius, so `U`
is biased **low**; the D1 "violation" is an estimator artefact and is largest when `U` is small —
predicted `D/U` blow-up correlates with `λ_min ≤ 0` in the gym (predict: **all 7 cells with `D/U > 4`
have `λ_min < 0`**).

## P-E R-1 / R-2: truncation for a whole-window read-out

**Predicted law:** `k*_traj(ε) = W_ψ(ε) + ln(1/ε)/ln(1/ρ)`, where `W_ψ(ε)` = the smallest tail window
carrying `1−ε` of ψ's pooling mass. Endpoint ψ ⇒ `W_ψ = 1` ⇒ Q4.2 recovered. Uniform whole-window ψ ⇒
`W_ψ(ε) = (1−ε)N` ⇒ **no useful truncation exists; the loss is `O(1−k/N)`, linear, not geometric.**
**Predicted toy numbers** (linear contraction model, `ρ = 0.97468`, `N = 400`, uniform pooling,
`τ = 1/(1−ρ) = 39.5`): relative θ-gradient error `≈ 1 − (k−τ)/(N−τ)`; **at k = 270 predict 0.36 ± 0.05**
(vs the geometric prediction `ρ^270 = 9.8e-4` — 2.6 orders wrong, same *sign* of failure as the
engineer's measured 0.680 on the two-phase harness).
**Predicted direction result (R-2):** `∂L/∂φ` flows only through `q₀`, at the **head** of the window,
with weight `Σ_n w_n ρ^n`; **tail truncation gives exactly 0.0** (reproduced exactly, not
approximately), and **head truncation at `h` gives relative error exactly `ρ^h`** ⇒ `h*(1e-3) = 269` —
the same number, the other end. **Predicted recipe: a two-sided (head h* + tail k*) retained window**,
and for θ the implicit path is *better* justified under a whole-window ψ than under an endpoint ψ,
because `∂q_n/∂θ = (1−ρ^n)∂q*/∂θ` ⇒ the pooled θ-gradient is `(1 − Σw_nρ^n)·∂q*/∂θ`, a **scalar
multiple** of the implicit gradient (predict the scalar `= 0.967 ± 0.01` in the toy).

## P-F R2(doctrine): the γ band's harness-invariant form

**Predicted invariant:** the band is not in γ but in the **convergence budget**
`C ≡ N·ln(1/ρ)`, `ρ = max(√(1−γ), |1 − (2−γ)dt²λ̄/(2γm)|)`; in band iff `C ≥ ln(1/tol)`.
Edges: `γ_min = 1 − exp(−2ln(1/tol)/N)`; `γ_max = 2κ/(2ln(1/tol) + κ)`, `κ = N dt² λ̄`.
**Predicted scores against the published `s2b` γ×N table** (`s = 0.35, α = 0.05, A ∈ [0.62,1.34]` ⇒
`λ̄ = 2α + Ā/s² ≈ 7.9`, `dt = 0.05`, `tol = 1e-6`): `γ_min(400) = 0.0667`, `γ_max(400) = 0.445`,
`γ_min(1500) = 0.0183`, `γ_max(1500) > 0.9` ⇒ **all four measured band edges predicted at grid
resolution (4/4)**, i.e. in-band at N=400 is `{0.1, 0.2}` and at N=1500 is `{0.02 … 0.9}`.
**Predicted self-correction:** my C2W1 row-1 band `γ ∈ [0.05, 0.5] at N = 400` is **scored at
`tol = 1e-3`, not the row's own `1e-6`** (at `γ=0.05, N=400` the published table reads `3.62e-5`).
I predict the published pair of bands mixes two tolerances and must be re-stated.

## P-G R5(gym): the reach probe that also clears `d_safe`

**Predicted construction & feasibility condition.** Reach is a *single-site* constraint on
`L = √(‖c‖² + ‖a‖²)`; `d_safe` is a *pairwise* constraint. They are separable, so a probe exists iff
the address ball still admits one `d_safe`-separated site: **`N_pack(R,d) > K`**. Construction: put the
probe at the **farthest-point** address of maximal norm (`argmax_c min_j ‖c − c_j‖` on the ball
*boundary* — the boundary both maximises separation from an interior-packed store and **minimises
`a_U`**, since `a_U` decreases in `‖c‖`), then set `‖a‖ = 1.15·a_U(‖c‖, s, D, α)`.
**Predicted numbers (gym-like: `d = 4`, `R = 1.0`, `s = 0.30`, `D = 0.5`, `α = 0.05`, `d_safe = 0.58`):
the construction yields min-distance ≥ `d_safe` with ≥ 20 % margin and reach margin `a_U − ‖a‖ < 0`
by ≥ 10 %, i.e. #11 trips and #3 does not refuse.**

## P-H Falsifiers registered for the whole task (task §4)

- **§A2.5 refuted** (a 13-band config violating the certificate) ⇒ headline, re-prices the campaign
  upward. *(I predict this fires in P-A3.)*
- **A mode with no runtime-computable invariant after the repairs** ⇒ C2W1 falsifier (a) fires late.
  *(I predict it does NOT fire.)*
- **Two productive bands provably disjoint after the soft-certificate relaxation** ⇒ staged activation
  impossible ⇒ escalate to the Head immediately. *(I predict it does NOT fire.)*
- **Does NOT falsify:** a widened band; a proxy retired without replacement; a reconciliation whose
  only honest disposition is "scope it, don't fix it"; a repair that makes a monitor trip *more* often.
