# Task: xy-1d-control — the go/no-go for the entire Thread-10 thermodynamic framing (**w15**, analyst)

> **✅ SPAWN GATE CLEARED (Hub, 2026-07-10).** Both prerequisites merged to `main` at **`df5e44d`** (suite 278 passed / 1 skipped). The Hub verified the NaN blocker end-to-end: `train_chlu` + `langevin_noise="fdt"` at the exact default trigger now yields finite losses `[1880.4, 843.4, 1.545]` and finite `log_mass`. `coupling_type="channel_spring"`, the conformal-init option, the free-energy `GatedCoupling`, and `torus_edges(L)` are all live. **This task is GO for wave 15. Record `df5e44d` (or later) as your commit.**
>
> ⚠ **Protocol §5 now mandates PRE-REGISTRATION** for measured ratios/exponents/laws. This task's `ξ` is parameter-free, so write `PREREG.md` with the predicted `ξ(T)` values computed from `J = 2κr*²` **before** you run the chain.

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/xy-1d-control.md`
- **Read first:** protocol (§5 flag-provenance mandatory) · **`.claude/outputs/xy-lattice-theory.md`** — §2 (the reduction), §3.2 (the corrected dictionary), §4.3 + §4.6 (the 1D solution and the honest null), §5 (blocking prerequisites), §7.7 (the kill criterion) · `.claude/claims_matrix.md` **CM-16, CM-17**.
- **Repo:** read-only.
- **⛔ SPAWN GATE — do not start until BOTH have merged to `main`:** `fix-pack-6` item 1 (the `sqrt(0)` FDT NaN-gradient fix) and `lattice-xy-prereqs` items 1–2 (free-energy gate + `coupling_type="channel_spring"`). Nothing FDT-correct can be trained before the first; the reduction is void without the second. **Verify both at HEAD and record the commit.**
- **Scope discipline (BINDING):** **G7 is a LONGS mandate.** This does not widen any short. It exists to make a future CSF3 run *deserve its queue time*.

## Why this task exists
The theorist proved the CLU register lattice reduces **exactly** to the classical XY model — but only for the *designed* `channel_spring` coupling, and only on a toy lattice with `N=2`. The whole Thread-10 program (the KT memory phase; the leading collective-memory thesis) rests on that reduction holding **on the real code path, at `N>2`, on a trained-or-designed lattice.**

The Head's standing instruction is that **CSF3 runs are expensive and queue-bound, and must not be spent discovering a local mistake.** The 2-D KT experiment needs `L ≥ 8` (`N ≥ 64`); `L=32` is the program's first genuinely A100-scale run. **Our largest lattice ever is `N=16`, and at `L=4` the naive `T_KT` crossing is 13.5% off — it cannot see the jump.**

So: before we buy any of that, the **1D chain must reproduce a parameter-free exact prediction on the real code path.** This is the cheapest decisive test the program has, it runs on a laptop in days, and it is the theorist's own nominated go/no-go.

## The falsifiable (parameter-free — nothing to fit)
A chain of `N = 16` designed SO(2) CLU units with `channel_spring_coupling(κ)` at temperature `T`, sampled correctly, must have correlation function `C(r) = ⟨cos(θ₀−θ_r)⟩` decaying exponentially with **exactly**
```
ξ = −1 / ln( I₁(J/T) / I₀(J/T) ) ,        J = 2κr*²
```
`I₀,I₁` = modified Bessel. **No free parameters.** `J` is *predicted* from `κ` and the measured `r*`, not fitted.

## Items
1. **Establish the dictionary on the real path.** Measure `r*` per unit, compute `J = 2κr*²`, and verify the reduction directly: on the vacuum torus, the θ-dependence of the joint `V` should be pure first harmonic (theorist's 2-D FFT test: residual power outside the quadratic band ≤ `4.2e-32`). Do this **before** any sampling — if the reduction fails, stop and report.
2. **Sample correctly, and prove you did.** Prerequisites are **physics, not hygiene** (theorist §5 vii):
   - `langevin_noise="fdt"` — under `legacy` the XY temperature is **47.5× off** at `dt=0.02, γ=0.1` (Prop-9's `T_eff` predicts the legacy behaviour to 0.05–1.2%; the dictionary is simply void there).
   - `use_governor=False` — a governed array is **always more ordered than its temperature says** (`⟨cosΔθ⟩/Gibbs = 1.10 / 1.36 / 1.93` at `T = 0.02/0.05/0.10`) and is **still drifting** after 30k steps. It is annealing, not sampling: **no temperature, no phase diagram.**
   - `kinetic_mode="newtonian_learned"` — **not relativistic** (CM-17: the code damps `p`, but Gibbs-preserving Langevin damps the velocity `∇_pT`; measured `⟨cosΔθ⟩/Gibbs = 1.006/1.040/1.134`, drift ≈ 0 ⇒ a genuinely *different* stationary measure). Note the subtlety worth stating in the report: **the Gibbs measure itself is relativity-insensitive** (the momentum integral factorizes) — the failure is in the sampler alone.
   - `κ` small enough for Born–Oppenheimer: `κ ≤ k_r/40` ⇒ `J₂/J₁ ≤ 2.5%`, and `κ < k_r/8` or **the ring collapses**. ⚠ **Trained designed checkpoints (`k_r ∈ [0.458, 0.927]`) at the shipped default `κ_c = 0.05` sit at `κ/k_r ∈ [0.054, 0.109]` — within 1.15–2.3× of ring collapse.** Choose `κ` deliberately and report `κ/k_r` and `J₂/J₁`.
   - Run the **equilibrium-start stationarity protocol** (theorist's `s4b`: start from the exact Gibbs quadrature, check first-half vs second-half drift) extended to `N=16` **before** trusting any number. The theorist's own first sampler run (`s4_langevin_gibbs`) was superseded because its large-γ arms were not equilibrated — the relative mode relaxes in `n_relax ≈ γ/(ε²μ_rel²)` steps.
3. **The three 1D predictions (all sharp, all cheap):**
   - `ξ = −1/ln(I₁(J/T)/I₀(J/T))`, parameter-free, over ≥4 temperatures.
   - `ρ_s → 0` with `N` at every `T > 0` (no stiffness in 1D — the honest null).
   - A written winding survives `τ ∝ 1/N` — **1D memory *degrades* with size** (per-site slip rate constant; theorist measured `0.00656 ± 11%` at `T = 0.5J`). This is the null against which the 2-D claim ("memory *improves* with size") will later be judged, so measure it now, on the real path.
4. **The `γ`-independence check (a headline in itself).** The stationary measure must be **γ-independent** — friction changes the *approach*, never the *destination* (`⟨cosΔθ⟩/Gibbs = 1.00028 ± 0.00218` over `γ ∈ {0.02,0.1,0.4}`, theorist, `N=2`). Reproduce at `N=16`. This is the precise, measurable form of Thread-10's sharpening of CM-16: **a lone register at `T>0` always forgets; friction buys time (∝ γF²/T), not permanence.**
5. **Broken-symmetry control (cheap and devastating).** Re-run one temperature with the **default random-`W` `spring_coupling`**. Predict: the dictionary fails — `ξ` does not match, because the `p=2` anisotropy (`h₂/|J| ≈ 0.6–2.1` on trained models) is a **relevant** perturbation. This simultaneously justifies `lattice-xy-prereqs`' default change and shows *why* the prerequisite is physics.
6. **Kill criterion (state the verdict plainly).** If the CLU-Langevin `C(r)` does not match the parameter-free `ξ` at `J = 2κr*²`, **the reduction has failed on the real code path and the entire thermodynamic framing is void for that lattice.** Report it as a negative (C-9), loudly, and recommend against the CSF3 spend. A clean kill here is worth more than the 2-D run it prevents.

**Acceptance:** the reduction verified on the real path (FFT harmonic test); equilibration proven, not assumed; `ξ` matched parameter-free over ≥4 temperatures **or** a clear kill; the three 1D nulls measured; γ-independence at `N=16`; broken-symmetry control run. Every `n₁/₂`-type number carries its `Δ` and `ℓ_θ/Δ`. **Close with an explicit GO / NO-GO recommendation on the 2-D CSF3 experiment** — that recommendation is this task's deliverable.
