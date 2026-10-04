# Task: v2-full-runs — the V2 short's results battery (TOP PRIORITY — Mo is one step behind us)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/v2-full-runs.md` + figures/npz in `.claude/outputs/v2-full-runs/`
- **Read first:** protocol · `.claude/outputs/v2-so2-build.md` (the apparatus + its quick-run baseline) · `.claude/outputs/mo-deep-read.md` §3 (protocol to adopt) + §5 (corollaries C1–C3) · F5 §3–§4. Def-2 nomenclature. Base: current `main` (wave-2 merged; note defaults changed: `lyapunov_penalty="max"` now live — RECORD which flags every run uses).
- **Apparatus:** `chlu exp-d` / `run_experiment_d` / `chlu/experiments/goldstone_harness.py` — build nothing new unless a measurement gap forces a tiny script.

**Goal:** paper-grade measured results for the V2 short. Laptop/CPU-scale (small dims); use CSF3 only if seed-sweeps drag.

## The battery (each item = a figure or table candidate; report actual numbers + error bars)
1. **GMOR δ-sweep (headline law):** train once (defaults), wrap the checkpoint with `TiltedPotential` across δ ∈ {1e-4 … 6e-2} (Mo's grid + extensions through the crossover), measure **both** retention metrics (C2: first-crossing AND envelope half-life). Expect: n₁/₂ ∝ δ⁻¹ overdamped → saturation at 2ln2/(−ln(1−γ)); metric bifurcation past h*.
2. **Mo head-to-head table (the generosity+separator figure, per mo-deep-read §3/Check-6):** run Mo's exact code-level lifetime protocol (φ₀=0.35, threshold 0.2, censoring) on our trained model across the δ-sweep; report measured/Mo-predicted ratio vs regime. This is a headline figure candidate for the short.
3. **γ-sweep through γ* (C1):** fixed trained model, sweep γ across ≈2εμ for a chosen massive mode; show the **retention minimum at critical damping**. Also demonstrate the flat mode's immunity (latch at every γ>0).
4. **Emergent-symmetry variant:** `--potential-type mlp` (same data): does a near-flat direction emerge without the architectural guarantee? Measure the spectrum + retention; compare gap sizes designed-vs-emergent.
5. **Isotropization falsifiable (F5 §4.1):** `--broken-isotropy` training on symmetric data: does learned M isotropize within the channel (‖M₁−M₂‖ trajectory over epochs if cheap via checkpoints, else final)? Report the induced pseudo-Goldstone gap and its measured lifetime; also report **E_eq split** (E^V vs E^T attribution — our refinement of Mo's Eq. 4; harness has the pieces: equivariance of ∇V vs ‖[M,X]‖).
6. **Seeds & rigor:** ≥5 seeds for the headline items (1–3); report mean±std; single-seed OK for 4–5 with a note. Frozen manifest discipline (mo-deep-read §3.5): every figure ↔ config/seed/commit.
7. **(Stretch) EP signatures (C3):** frequency onset ∝ √(h−h*) near the crossover on the learned model.

## Rules
Repo read-only except: you may add small analysis scripts under `.claude/scratch/v2-full-runs/`. If the harness lacks a measurement, specify the gap for the engineer rather than patching `chlu/`. Record runtime per item (informs CSF3 need). Honesty: if the learned model's anharmonicity breaks a law's constants, show it — deviations are results.
