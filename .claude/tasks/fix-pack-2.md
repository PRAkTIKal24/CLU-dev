# Task: fix-pack-2 — proven-defect fixes from F5 + wave-1 discoveries

- **Agent:** `experiment-engineer` · **Base:** `main` (post first-fixes merge) · **Branch:** `agent/experiment-engineer/fix-pack-2` · **Output:** `.claude/outputs/fix-pack-2.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md`, handover §7 (items 6, 9, 10, 11, 12), and the fix specs in `.claude/outputs/formalism-note.md` (F5 Prop-5, Prop-9 — authoritative).
- **Nomenclature:** use F5 Def-2 language in all new code/docstrings — *inertial mass M* vs *spectral mass μ*; never "mass" unqualified.

**Acceptance:** all five fixes implemented + tested, `uv run pytest -q` green, atomic tagged commits, behavior-changing fixes behind flags with defaults preserving current behavior (checkpoint compatibility matters — mid-sprint retraining is not acceptable collateral).

## Fixes (one commit each)
1. **Lyapunov regularizer replacement (§7.6, F5 Prop-5).** `compute_lyapunov_loss` currently returns mean-log-singular-value ≡ ½ln(1−γ) ≡ 0 at γ=0 — provably θ-independent. Replace with a config-selectable spec: `training.lyapunov_penalty ∈ {"none", "max" (max_i log σ_i), "sq" (Σ(log σ_i)²), "pos" (Σ max(0, log σ_i))}` — **default `"max"`** (the chaos-relevant one), keep the old function available as `"legacy_degenerate"` for reproduction. Add a test asserting the new penalties have nonzero θ-gradients on a random potential and the legacy one has ~zero.
2. **Langevin FDT fix (§7.9, F5 Prop-9).** Add `training.langevin_noise ∈ {"legacy", "fdt"}` (**default `"legacy"`** for checkpoint compat). `"fdt"`: per-mode σ*_i = √(M_eff,i·T·γ(2−γ)), with M_eff per kinetic mode (I / M / m₀M — F5 §2.1 table). Wire through `langevin_step`, `stochastic_step/rollout`, both trainers. Test: with `"fdt"` on a quadratic toy, stationary Var(p_i) ≈ M_eff,i·T (tolerance ~5%, short chain); with `"legacy"`, reproduce the mismatch.
3. **Seed the MNIST subsample (§7.11).** `chlu/data/mnist.py:37` → `np.random.default_rng(seed)` with seed threaded from config (`project.seed`). Test: two loads, same seed ⇒ identical selection.
4. **Wire `--quick`/`train_epochs` (§7.10).** `run_experiment_a/b` currently set `config.experiment_X.train_epochs` but `train_chlu` reads `config.training.epochs`. Make the experiment override reach `train_chlu` (pass `epochs=` explicitly). Test: quick-mode call trains the requested small number (assert via loss-history length).
5. **Durable env fix for the UF_HIDDEN editable-install bug (§7.12).** Investigate the cleanest durable option: recreate the venv, pin a uv setting, switch to non-editable local install, or a documented `chflags` step in a `Makefile`/`justfile` target. Implement the best one; verify `uv run chlu --version` works from a fresh shell **twice in a row** (the flag re-sets each run — that's the trap). Document the fix + root cause in the output file.

## Scope guards
- **No rename work** (CLU fork happens at publication cleanup — decision on file). No experiment logic changes beyond the wiring in (4). S^(M) squeeze and γ_φ(q) live in OTHER tasks — don't implement here.
- Concurrent spokes may run analysis in this repo (read-only) — normal protocol §3 discipline.
