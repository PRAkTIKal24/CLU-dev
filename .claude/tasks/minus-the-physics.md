# Task: minus-the-physics — the missing "CLU minus the physics" controls (critiques P6/G2 + P7/V1.1)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/minus-the-physics.md` (+ figures/npz in `.claude/outputs/minus-the-physics/`)
- **Read first:** protocol · `.claude/critique_register.md` (G2 + V1.1) · `.claude/outputs/v2-so2-build.md` (measurement harness) · `.claude/outputs/v1-pivot.md` (calibration stack + its numbers — the comparison target) · `.claude/outputs/v3-lattice-build.md` (duck-typing precedent: CLULattice runs the harness verbatim) · `chlu/training/calibration.py`.
- **Why:** no functional claim in the program (retention, gating, allocation, routing) currently has an identical-capacity non-symplectic control, and V1's gate stack may be memory-agnostic (energy ≈ readout margin). These two ablations decide whether the papers are about CLUs or about calibration engineering — and what symplecticity *functionally* buys. **Honest results either way; a negative here changes framing, not the program.**
- **Git:** branch `agent/experiment-engineer/minus-the-physics` — **worktree MANDATORY** (§3.2; `mass-lr-doctrine-test` runs concurrently).

## Part A — non-symplectic twin (G2)
1. Implement two controls, both **duck-typing CHLU** so the existing harness + train_chlu run verbatim (CLULattice precedent):
   - **`UnconstrainedTwin`:** same (q,p) dims, matched param count (report the count match), free update `z_{t+1} = z_t + f_θ(z_t)` — no Hamiltonian, no volume constraint.
   - **Broken-volume arm:** the same leapfrog pipeline but with a learned per-step state scaling that breaks det J = 1 (keeps everything else — architecture, potential, dt — identical). This isolates symplecticity specifically, vs the twin which removes the whole physics.
2. Run BOTH through the same measurement protocol as the V2 battery, 3–5 seeds: retention-vs-perturbation curve, latch test (designed flat direction), n₁/₂-vs-μ observables, sleep-erosion susceptibility (150-ep battery settings), plus training quality (eval MSE) on exp-d data.
3. Deliverable: the **"which component buys what" table** — for each functional metric: CLU vs broken-volume vs twin, with the delta attributed to (integrator structure | volume conservation | nothing).

## Part B — gate stack on non-CLU memories (V1.1)
4. Run the **identical** calibration/allocation/LTT stack (`calibration.py`, exp_v1_calibration protocol, same seeds/levels as v1-pivot) on the **Hopfield baseline**, using its natural scalars (Hopfield energy; readout margin). If cheap, add the Part-A twin as a third memory.
5. Report the same metrics as v1-pivot (calibration-transfer AUROC, allocation compute-savings curve at matched accuracy, LTT validity count) side-by-side with the CLU numbers.
6. Deliverable: the V1-identity verdict — **"the gate stack is memory-agnostic (numbers)"** or **"CLU-specific advantage exists at [conditions]."** Either way state what V1 may then claim (mechanism+certificates vs signal superiority — see register P7).

Flag-provenance tables on every result (protocol §5). Do NOT touch V2/V1 production code paths beyond adding the twin module + minimal registration; strict scope.
