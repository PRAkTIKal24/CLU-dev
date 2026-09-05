# Task: v5-gate — the two cheap experiments that decide whether "Forgetting" is a V5 short (w13)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/v5-gate.md` (+ figures)
- **Read first:** protocol · **`.claude/outputs/t-lever-forgetting.md`** (the whole report — esp. §10 decision rule, §11 R1/R3, §8 the T_φ spec, §9 confound C1) · claims matrix **CM-16** · its scratch harnesses `.claude/scratch/t-lever-forgetting/*` (reuse verbatim).
- **Why:** `t-lever-forgetting` returned **"V5-worthy CONDITIONAL"**. Its own decision rule names two cheap, decisive gates. Run them BEFORE any `T_φ(q)` build is funded. Repo read-only; `langevin_noise="fdt"` + `common.retie` throughout (CM-16's mandatory flags).

## The two gates
1. **R1 — the generalization gate (kills or confirms confound C1, the main V5 risk).** Replicate the §4.2 sign-flip on the **emergent (MLP) checkpoints** `emergent150_s{42,43,44}` — where CHLU's flat directions actually *come from*, and which are only *approximately* flat (μ²_ang ≈ 5.2e-3). **Prediction to test:** the T=0 latch now decays (pseudo-Goldstone, n₁/₂ = ln2/gap(μ²_ang,ε,γ)), so at small T the ∂n₁/₂/∂γ sign flips **back to negative** below a crossover **T\***; above T\* the diffusive (positive-slope) branch takes over. **Measuring T\* is a new falsifiable** and would be the emergent-arm result V5 needs. If instead the sign flip is simply absent/muddy on emergent models, that is the honest kill-shot for a standalone short (→ V2 appendix).
2. **R3 — the free falsifiable (no new code; `FrictionField` already exists).** T5/T6 from the spec: place a **γ_φ(q) friction hole** over a latched coset register at T>0 and measure its half-life vs the un-holed control. **Prediction: the hole is a ≈13.9× MEMORY VAULT** (γ_eff 0.05→0.525 ⇒ n₁/₂ ∝ γ_eff). Converts §7's retro-explanation of the γ_φ −24% negative from *post-hoc* into a *pre-registered prediction*. **Register the prediction in the report before measuring.**

## Deliverable — the V5 call, with evidence
- Plain verdict: **V5 GO** (R1 gives an emergent-arm result *and* R3 lands the vault ⇒ the "friction hole preserves, temperature hole shreds" figure is a short) **or V2-APPENDIX** (fold Fig 2 + Fig 4 as "the T=0 face of the budget cube").
- If GO: confirm the `T_φ(q)` engineer spec (§8) is still exactly right given what R1/R3 showed, and name the one figure the short is built around.
- Negatives fully written (C-9). Flag-provenance per §5 (checkpoints, fdt flag, retie, T/γ grids, Δ vs ℓ_θ — quote no n₁/₂ without its ℓ_θ/Δ).

**Do NOT build `T_φ(q)` in this task.** The build is R4, funded only on a GO.
