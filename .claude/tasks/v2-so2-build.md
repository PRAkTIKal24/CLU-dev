# Task: v2-so2-build — the SO(2) Goldstone-memory experiment (V2's core empirical result)

- **Agent:** `experiment-engineer` · **Base:** `main` · **Branch:** `agent/experiment-engineer/v2-so2-build` · **Output:** `.claude/outputs/v2-so2-build.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md`, **`.claude/outputs/formalism-note.md` §3–§4 (the predictions this experiment tests — F5 is authoritative)**, handover §7, brainstorm Thread 2 + Wave-1 updates. Use F5 Def-2 nomenclature (inertial M vs spectral μ) everywhere.
- **Coordinates with:** `mo-deep-read` (protocol comparability — read its output if it lands first; don't block on it).

**Goal:** build the apparatus that measures the F5 §3.3–§3.4 predictions on a trained CLU with a designed SO(2) symmetry — the latch, the half-life law, the crossover/saturation, pseudo-Goldstone lifting, and kinetic isotropy. This is V2's decisive experiment; the ML4PS short stands on it.

## Build (new module, e.g. `chlu/experiments/exp_d_goldstone.py` + core pieces where they belong)
1. **SO(2)-equivariant potential option** (`potential_type="so2_invariant"` or a config-selected wrapper): V_θ(q) = f_θ(invariants) over designated channel pairs — for a channel (q₁,q₂): V = f_θ(r) with r=√(q₁²+q₂²) (learned radial profile, e.g. small MLP on r), optionally + non-symmetric extra dims governed by a standard MLP. **Kinetic isotropy enforced per F5 §4.1:** equal inertial masses within the channel (config flag: tie the channel's log_mass entries; keep a "broken-isotropy" switch for the falsifiable).
2. **Controlled explicit breaking:** additive tilt δ·cos(n·θ_channel) with configurable δ, n (the GMOR probe from F5 §3.3c).
3. **Measurement harness** (reusable; results-analyst will drive it later):
   - Spectrum probe: at a settled state q*, compute W = M_eff^{-1/2}·∇²V·M_eff^{-1/2}, its eigenvalues μ² and eigenvectors (small dims — exact jax.hessian is fine).
   - Perturb-and-track: kick along chosen eigendirections (position and/or momentum impulses), roll out at configurable γ, record per-mode displacement/retention vs steps.
   - Extract: measured half-life per mode; Noether charge Q = q₁p₂−q₂p₁ trajectory; coset angle θ(t).
4. **Training setup:** small CLU (e.g. dim 4–8, one SO(2) channel + curved spectator dims), trained with the standard wake–sleep on a task whose data is SO(2)-degenerate along the channel (e.g. trajectories on a circle of attractors / rotated copies of a pattern — design the simplest dataset that makes the vacuum a circle; document the choice). Use the **new Lyapunov penalty and FDT noise if fix-pack-2 has merged; otherwise flags off** — note which you used.

## Smoke validation (part of THIS task — laptop scale)
Verify the harness reproduces F5's exact quadratic predictions before any learned-V run: on a hand-built quadratic V with one flat + one curved direction, measure (a) the latch (q∞ = q0+εp0/(Mγ), γ>0), (b) half-life ratio 4.0 for μ²-ratio 4 (overdamped), (c) underdamped saturation at 2ln2/γ, (d) Q decay = (1−γ)ⁿ, (e) with unequal channel masses: O(1) charge drift (isotropy falsifiable). These must match F5 Appendix N numbers — if they don't, your harness is wrong, not F5.
Then one small learned run (quick epochs) end-to-end: train → spectrum → perturb → report the measured spectrum and whether a near-flat direction emerged along the designed orbit.

**Deliverables:** module + tests (harness-vs-F5 smoke checks as pytest), a `projects/`-style runnable entry (CLI hook optional — a documented script is fine at this stage), and the output report with observed numbers. Full-scale runs + figures are a follow-up analyst task — do not gold-plate.
