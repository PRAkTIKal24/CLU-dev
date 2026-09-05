# Task: causal-memory-floor — F-10 + F-11: the relativistic governor's *memory-side* payoff (w14, analyst)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/causal-memory-floor.md`
- **Read first:** protocol (§5 flag-provenance mandatory) · **`.claude/outputs/v2-symmetry-deepdive.md` §7bis R3 / R6 / R7 + falsifiables F-10, F-11** · `.claude/claims_matrix.md` **CM-16** (the forgetting laws this extends) · `.claude/outputs/t-lever-forgetting.md` (the harness — `common.retie`, the Langevin rollout, the coset half-life instrument).
- **Repo:** read-only.
- **Scope discipline (BINDING):** **G7 is a LONGS mandate — this does NOT widen any short.** V2/V1/V3 are frozen in shape. This work feeds the **Nature-MI physics flagship** and the ICLR long. Do not propose draft edits to the shorts.

## Why this matters
The published paper sells the relativistic kinetic governor as **velocity safety**. R7 shows it is also a **memory-robustness guarantee**, and this is the first ML-measurable benefit that earns the relativistic term a place in the *memory* story (principle P1: physics must buy a measurable ML benefit).

Because `|q̇| ≤ v_max` at every step, trivially and **without any assumption on the noise**:
```
|Δθ_n| ≤ n·ε·θ̇_max     ⇒     erasing a register to tolerance Δ needs  n ≥ Δ/(ε·θ̇_max) steps
```
**for any noise, any temperature, any adversary.** Newtonian mode admits no such bound. Pair this with CM-16 (*friction cannot erase a flat direction; only temperature can*) and the statement becomes: **temperature is the only eraser, and causality bounds how fast temperature can erase.**

## Items

### Part 1 — F-10: the causal memory-lifetime floor
1. Latch a designed SO(2) register (reuse the `t-lever-forgetting` designed150 checkpoints + `common.retie`). Run two arms: `kinetic_mode ∈ {newtonian_learned, relativistic}`, matched `M_eff` at rest (`M_eff = m₀M` — match it, don't assume it).
2. Inject **escalating noise**: sweep `T` over ≥3 decades, and separately an **adversarial impulse train** (worst-case-aligned kicks along the coset tangent). Measure steps-to-erasure to tolerance `Δ`.
3. **Predictions:** relativistic — `n ≥ Δ/(ε·θ̇_max)` holds at *every* noise amplitude, and `D` **saturates** (deep-dive measured `D_rel: 0.0093 → 0.671` over `T ∈ [0.01,1000]`, vs `D_newt` growing linearly; `D_rel/D_newt = 0.971 → 7.0e-4`). Newtonian — erasure time `∝ 1/T`, unbounded.
4. **The floor is the headline.** A bound that holds against an *adversary* is worth more than one that holds in expectation. Try to break it: search for a noise process that erases faster than the floor. Report the attack and its failure (or its success — that would be the more interesting result).
5. ⚠ **`langevin_noise` interaction (read this).** R8 proves the coded `fdt` noise has **no Gibbs invariant in relativistic mode** (the O-step is linear OU ⇒ Gaussian; Gibbs demands Maxwell–Jüttner). This task's relativistic arm therefore **cannot claim to sample Gibbs.** That is *not* fatal here — the causal floor is a **pathwise, assumption-free** bound (`|q̇| ≤ v_max` every step), so it holds under *any* noise, correct or not, which is precisely its strength. **State this explicitly.** Report `T/(m₀c²)` for every relativistic cell. Do not report any relativistic `T` as an equilibrium temperature.

### Part 2 — F-11: the rapidity register (write companding)
6. Sweep the write impulse `p₀` over 3–4 decades on an SO(2) channel, relativistic vs Newtonian. **Prediction (exact):** `Δθ = ε·θ̇_max·Σₙ tanh ζₙ` with `sinh ζₙ = (1−γ)ⁿ sinh ζ₀`; transport is **logarithmic in `p₀`** — at `M=0.8, ε=γ=0.05`, an impulse `p₀=500` stores `Δθ = 7.678` where the Newtonian latch stores `625.0`.
7. **Aliasing / graceful saturation (R6, and note the theorist self-corrected here).** The protection factor is `p_crit^rel/p_crit^Newt = sinh(ζ*)/ζ*`, `ζ* = 2πγ_c/θ̇_max` — **`48.9×` at γ=0.05 but only `1.29×` at γ=0.01, and → 1 as γ→0.** Verify the `γ`-dependence, including a cell where the protection is **absent**. A conditional benefit reported as unconditional is exactly the kind of overclaim the register catches.
8. Scope honestly: the rapidity law is exact on a **strictly flat** direction. On the **curved** ring a hard write leaves the vacuum manifold (centrifugal excursion `∝ (θ̇r*)²`; theorist measured `0.2%` at `ζ ≤ 3` and left the large-`ζ` correction **open, O7**). Report where the closed form starts to break, and whether a hard write can **dislodge the latch off the orbit** — that is the companding register's natural failure mode and nobody has looked.

**Acceptance:** the causal floor measured (and attacked) in both kinetic modes; `D`-saturation vs linear growth reproduced on the coded path; the companding law verified across decades with the `γ`-conditional aliasing protection stated as conditional; O7's large-`ζ` failure mode probed. Every relativistic cell carries its `T/(m₀c²)` and the R8 non-Gibbs caveat. **Deliver a one-paragraph statement of the guarantee in ML terms** — bounded forgetting under unbounded noise injection — suitable for a longs abstract.
