# Task: mass-spectrum-peek — do trained (CH)LUs already learn non-trivial mass?

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/mass-spectrum-peek.md` (+ figures in `.claude/outputs/mass-spectrum-peek/`)
- **Read first:** `.claude/AGENT_PROTOCOL.md`, `.claude/handover_context.md` (§2, §5 provenance), `.claude/brainstorm_log.md` (Thread 5 — the hypothesis this tests).

## Why
Thread 5 elevates the learned mass matrix M to the program's central object ("budget allocator"). Before we build the formalism and V3's mass-hierarchy on it, check the cheapest possible evidence: **we already have trained checkpoints — what did M actually learn?** Falsifiable #(i): trained models develop non-trivial, interpretable mass spectra. If M turns out ≈ uniform everywhere, Thread 5 needs rethinking — that's a valuable result too.

## Inputs (existing checkpoints — do NOT retrain)
- `projects/finalA/models/exp_a_chlu.pkl` (Exp A: figure-8, dim=2, kinetic **newtonian_identity** — note: M is *unused* in the identity-mode Hamiltonian but still initialized/possibly drifted; report it as the control case).
- `projects/finalA/models/exp_b_chlu.pkl` (Exp B: sine, dim=1, **relativistic** per finalA config — the interesting one; confirm mode from the checkpoint's stored config).
- `projects/mnist*/models/exp_c_chlu.pkl` (×4 variants; Exp C generative, dim=784 conv/relativistic — the high-dim case).
- Loading pattern: `chlu.utils.checkpoints.load_checkpoint(path, template)` — build the template CHLU with the right (dim, hidden, kinetic_mode, potential_type) from each checkpoint's stored config/metadata; `M = jax.nn.softplus(model.log_mass)`. Init distribution for comparison: `log_mass ~ 0.1·N(0,1)` ⇒ softplus ≈ 0.69–0.76 band.

## Questions to answer (with numbers + figures)
1. **Is M non-trivial?** Per checkpoint: M's spectrum (sorted plot, histogram), spread (max/min ratio, CV), and statistical distance from the *initialization* distribution (has training moved it at all? in identity-mode (exp_a) it should be untouched by gradients — verify: does it exactly match init statistics? any drift = optimizer artifacts worth knowing about).
2. **Is M interpretable?** For exp_c (784-dim, pixel space): reshape M to 28×28 and render — is there spatial structure (center vs border pixels heavier/lighter)? Compare across the 4 mnist* variants (different friction/temperature configs — is the M pattern stable across them?). For exp_b (1-dim): just report the scalar trajectory context.
3. **Mass vs curvature (bonus, if time):** at a settled/attractor state (e.g., relax a test input briefly, or use the dataset centroid for exp_c), estimate diag of `Hess V_θ` (finite differences or `jax.hessian` on a subsample of dims for 784) and scatter mass vs curvature per dimension — any correlation? This previews the mode-frequency object `M⁻¹·Hess V`.
4. **Rest-mass & c context:** report the stored hyperparams (rest_mass, c) per checkpoint so the spectra are interpretable in the relativistic formula.

## Practicalities
- Laptop-scale; **JAX cold-start may take ~20 min** — one warm session, all checkpoints. `uv run python` scripts kept in `.claude/scratch/mass-spectrum-peek/`. Seeds irrelevant (pure analysis) but record package versions.
- An `experiment-engineer` may be working concurrently on a git branch — you are read-only on the repo; if `uv run` briefly rebuilds the package mid-session, just re-run the cell.
- If a checkpoint fails to load (template mismatch — the exact config schema may have drifted since finalA was created), document the failure precisely (this is itself useful — it's a reproducibility finding for handover §7) and move on to the loadable ones.

## Output format
(1) Per-checkpoint table (dims, kinetic mode, M stats, distance-from-init verdict); (2) figures (spectra; 28×28 mass maps for exp_c; mass-vs-curvature scatter if done); (3) interpretation vs Thread-5 falsifiable (i) — supported / refuted / mixed; (4) `## Proposed handover updates` including any checkpoint-loading reproducibility issues.
