# Task: gamma-field-build — the learned trash region: contrastive friction field γ_φ(q)

- **Agent:** `experiment-engineer` · **Base:** `main` · **Branch:** `agent/experiment-engineer/gamma-field-build` · **Output:** `.claude/outputs/gamma-field-build.md`
- **Read first:** protocol · brainstorm **Thread 1 (full: mechanism + Head decisions + S1/S2 studies) + wave-2 update (C1 coupling)** · F5 §7.3 (Def-5, Prop-11 — the horizon theorem) · mo-deep-read §5 C1 (**derived spec: optimal forgetting friction γ_φ(q) ≈ 2εμ(q) — critical damping**). Def-2 nomenclature.

**Goal:** implement the learned friction field and run the S1 pilot — the "horizon forgets garbage completely, memories not at all" Pareto experiment. This is Thread-1's mechanism made real; feeds a V2-adjacent short section + ICLR.

## Build
1. **`FrictionField(eqx.Module)`** (`chlu/core/friction_field.py`): K learnable holes — centers c_k, radii r_k (softplus), strengths γ_k ∈ [0, γ_max] (sigmoid-scaled) — `γ_φ(q) = γ_max·σ(Σ_k …)` or additive-saturating; K + γ_max in config. Plus a `"fixed"` variant (hand-placed, for controls) and `"none"`.
2. **Integrator wiring:** damping step becomes `p ← (1−γ_φ(q_{n+1}))·p` when a field is present (F5 Def-5 — evaluate at q_{n+1}, post-Verlet; Prop-11's det J = (1−γ_φ(q'))^d is the correctness test). Behind config `training.friction_field ∈ {"none","fixed","learned"}`, default `"none"` (zero behavior change).
3. **Contrastive training of γ_φ (Thread-1 round-2 mechanism):** within the wake–sleep loop — **wake: protection term pushing γ_φ(q_data)↓** (penalty on friction at data states); **sleep: γ_φ(q_hallucination)↑** at persistent negatives. One CD signal, two fields (V_θ learns what things look like; γ_φ learns what deserves to die). λ weights in config.
4. **C1 targeting (the derived spec):** report, for each learned hole, the local spectral masses μ(q) near the hole (harness `spectrum_probe`) and compare γ_k to the critical-damping optimum 2εμ — do learned holes find the fast-forgetting regime? (Measure, don't force; add an optional regularizer nudging γ_k → 2εμ(q_k) behind a flag for the ablation.)

## S1 pilot (the Pareto experiment — laptop scale)
Noisy lemniscate or noisy sine (Exp-A/B machinery): signal attractor + injected structured noise. Arms: (i) global γ, (ii) governor, (iii) **learned γ_φ** (K=1 and K=4), (iv) fixed hole at a known noise locus (oracle control). Metric: **signal-retention vs noise-rejection Pareto curve** (retention = latch/trajectory fidelity on the clean attractor; rejection = decay rate of injected perturbations). Prediction on file: (iii) Pareto-dominates (i)/(ii). ≥3 seeds. Report the learned hole placements visually (2D — plot γ_φ(q) heatmap over the potential landscape; `plot_potential_landscape_2d` is the template).

## Tests
det J = (1−γ_φ(q'))^d numerically (Prop-11) · default-"none" bit-compatibility with current integrator · protection term actually lowers γ at data states in a smoke train · config roundtrip.

**Scope guards:** S2 (Hawking re-emission arms) is NOT this task — design the field API so a re-emission hook can attach later (note where). Keep the field module independent of the lattice (composable later).
