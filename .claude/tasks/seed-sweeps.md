# Task: seed-sweeps — power the w3 single-seed results (laptop-scale)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/seed-sweeps.md` + figures/npz in `.claude/outputs/seed-sweeps/`
- **Read first:** protocol · `.claude/outputs/v3-lattice-build.md` (finding 5 + follow-ups 2–3) · `.claude/outputs/gamma-field-build.md` (findings A + follow-ups 4–5) · relevant configs. Base: current `main` (w3 merged; record flags per run).
- Repo read-only; scripts in `.claude/scratch/seed-sweeps/`. All laptop-CPU (CSF3 only if an item exceeds ~1h — report and defer instead).

## Items
1. **Banded-vs-uniform lattice training (Thread-5 falsifiable iii):** `exp-lattice` training smoke × **5 seeds** × both mass configs (identical params/data per pair); report wake-loss curves (crossover epoch), eval rollout MSE mean±std, and the verdict: does banding reliably beat uniform at matched params? Add a 60-epoch and 300-epoch budget point (learnability-prior signature = early advantage).
2. **Trained-coupling pricing (v3 follow-up 2):** on one trained 2-unit lattice, measure κ_eff from the learned V_c curvature and check the pricing law still holds (sync ∝ κ_eff^−½, n₁/₂ ∝ κ_eff^−1) — the harness needs zero changes.
3. **S1 extras (gamma-field follow-ups 4–5):** add the **governor+field composed arm** (they compose multiplicatively) to the S1 Pareto at 3 seeds; quick λ_protect/λ_hallu re-sweep post-fix-3 (coarse 3×3 grid, quick epochs) — does the learned-K=4 point move toward the oracle?
4. **(cheap, if time) Emergent-bias decomposition (v2-full-runs follow-up 3):** kick-size sweep on one emergent checkpoint to split the +13% retention bias into anharmonicity vs settle residual.

**Report:** per-item numbers with error bars + the one-line verdicts the shorts can quote. Frozen-manifest discipline (config/seed/commit per figure).
