# fix-pack-2 — experiment-engineer report

Task + acceptance criterion: implement + test the 5 proven-defect fixes (Lyapunov-penalty replacement, FDT Langevin flag, seeded MNIST subsample, --quick wiring, durable UF_HIDDEN env fix); `uv run pytest -q` green; atomic tagged commits; behavior-changing fixes behind flags with checkpoint-compatible defaults.

Status: **done** (all 5 fixes implemented, tested, committed; full suite 32 passed; `uv run chlu --version` works twice in a row from fresh shells)

## What I did (one commit per fix)

### 1. Lyapunov regularizer replacement (§7.6, F5 Prop-5) — `bffceac`
- `compute_lyapunov_loss` now takes `penalty ∈ {"none","max","sq","pos","legacy_degenerate"}`; new config `training.lyapunov_penalty` **default `"max"`** (as the task mandated), wired through `train_chlu`'s wake phase. `"legacy_degenerate"` = the exact old computation (mean_i log σ_i), kept for reproduction. `"none"` short-circuits without computing a Jacobian.
- ⚠ **Deliberate behavior change (task-mandated):** default `"max"` means new dynamics-training runs get a *live* regularizer (the old one contributed ~0 loss / ~0 grad). Reproduce historical runs with `lyapunov_penalty: legacy_degenerate` (or `lyapunov_lambda: 0`). No checkpoint-format impact (training-time loss only).

### 2. Discrete-FDT Langevin noise (§7.9, F5 Prop-9) — `85f13d6`
- New config `training.langevin_noise ∈ {"legacy","fdt"}`, **default `"legacy"`** (checkpoint/annealing-schedule compatibility — behavior unchanged unless opted in).
- `"fdt"`: per-mode σ*_i = √(M_eff,i · T · γ(2−γ)). New `CHLU.effective_mass()` returns the p≈0 inertial mass per kinetic mode: `newtonian_identity`→1, `newtonian_learned`→M, `relativistic`→m₀M (F5 §2.1 table; docstrings use Def-2 inertial-vs-spectral nomenclature throughout).
- Wired through `langevin_step` (new `noise_mode`, `m_eff` args, defaults preserve old signature/behavior), `stochastic_step`/`stochastic_rollout` (new `noise_mode` kwarg, default `"legacy"`), and **both trainers** (`train_chlu`, `train_generative`; also exposed as a function-level override param in house style). `langevin_step` raises ValueError for `fdt` without `m_eff` and for unknown modes (trace-time checks; strings static under jit).

### 3. Seeded MNIST subsample (§7.11) — `7a6e437`
- `load_mnist_pca(..., seed=None)` subsamples via `np.random.default_rng(seed)`; `seed=None` keeps non-deterministic behavior; `exp_c_dreaming` threads `config.project.seed`.
- Not touched (no `n_samples` at those call sites, subsample never triggers): `cli/train_cmd.py:127`, `cli/data_cmd.py:153`.

### 4. `--quick`/`train_epochs` wiring (§7.10) — `58e7d3b`
- `run_experiment_a` and `run_experiment_b` now pass `epochs=train_epochs` explicitly to **all three** trainers (`train_chlu`, `train_neural_ode`, `train_lstm` — baselines had the identical default-to-`config.training.epochs` bug; fixing only CHLU would have left `--quick` still training baselines 1000 epochs). Exp C already passed `epochs=` (no change).
- `9fa1c35` = ruff-format of my three new test files only (cosmetic; shared `chlu/` files already fail `ruff format --check` on main and were deliberately left unformatted).

### 5. Durable UF_HIDDEN env fix (§7.12) — `3a407c3`
- **Root cause pinned (mechanism):** `_editable_impl_chlu.pth` gets macOS `UF_HIDDEN`; Python ≥3.11 `site.addpackage` **silently skips hidden .pth files** (confirmed at `site.py:176` of the uv-managed cpython-3.11.13) → `import chlu` fails → CLI dead. Reproduced: hidden pth + `uv run --no-sync chlu --version` → `ModuleNotFoundError: No module named 'chlu'`.
- **Root cause narrowed (who sets the flag):** it is applied to **freshly-written** files at editable-(re)install time, and it is **session-dependent**:
  - `CHLU-v1-l0-gate` worktree venv (built today 00:25 by uv 0.7.15 from that spoke's session): freshly-written `.pth`s **hidden**; cache-cloned files (old mtimes) **clean**. This repo's venv (uv 0.7.3, Jul 4 session): same hidden pattern.
  - The *same* uv 0.7.15 driven from *my* session: every write **clean** — including forced reinstalls with the `.venv` dir flagged hidden and with the old pth flagged hidden (reinstall actually *clears* the flag). So it is not uv-version-deterministic, not dir-flag inheritance, not flag-preservation-on-replace, and not the uv cache (cache copies clean). Remaining suspect: something in the *other agent sessions'* environment (Zed-ACP harness/sandbox?) flags files their processes write. Could not introspect other harnesses from here — but the fix below is robust regardless of who applies the flag.
  - Ruled out per task's option list: venv recreation (today's fresh worktree venv is affected); uv settings (none control BSD flags; link-mode irrelevant — clones were always clean); non-editable install (stale-code footgun for a live-edited research repo).
- **Fix: `make fix-env`** (new top-level `Makefile`, fully documented in-file): (1) `chflags -R nohidden .venv` (heals current state), (2) writes an **unmanaged** path shim `zzz_chlu_dev.pth` (content = project root) into site-packages. uv never rewrites unmanaged files ⇒ the shim never gets re-flagged ⇒ imports survive even when the managed pth is re-hidden by an affected session. Must be re-run after venv recreation and once per new worktree.

## How I verified (real outputs)
- **Full suite:** `uv run --no-sync pytest -q` → **32 passed** (18 pre-existing + 14 new) in 26.8s. `ruff check chlu/ tests/` → "All checks passed!".
- **Fix 1:** θ-grad norms on a random potential (CHLU dim=2, hidden=16, dt=0.05): `max 9.06e-03`, `sq 9.14e-04`, `pos 1.00e-02`, `legacy_degenerate 1.03e-09` — degeneracy confirmed in-code, 6–7 orders below the replacements. Legacy loss value |·| < 1e-4 at γ=0 (= ½ln1). Tests: `tests/test_regularization.py` (5).
- **Fix 2:** harmonic toy (M=[0.5,2.0], k=1, γ=0.3, T=0.5, dt=0.1; 256 chains × 600 steps, 200 burn-in): `fdt` Var(p)=[0.2509, 1.0011] vs Maxwell–Boltzmann [0.25, 1.0] (≤0.4% off, 5% tolerance); `legacy` Var(p)=[0.0590, 0.0589] vs predicted 2Tdt/(2−γ)=0.0588 — mass-independent mismatch reproduced exactly. Tests: `tests/test_langevin_fdt.py` (5, incl. `effective_mass` table + model-level wiring + arg validation).
- **Fix 3:** network-free test (monkeypatched `fetch_openml`, distinct rows): same seed ⇒ identical train/test arrays; different seed ⇒ different. Test in `tests/test_data.py` (+1).
- **Fix 4:** `train_chlu(epochs=3)` with `config.training.epochs=1000` ⇒ loss history length 3; spy trainers assert exp A and exp B forward `epochs=7` to both `train_chlu` and `train_neural_ode`. Tests: `tests/test_experiment_epoch_wiring.py` (3).
- **Fix 5:** `make fix-env` → "OK: chlu importable"; shim survives plain `uv sync`; `uv run chlu --version` → `CHLU 0.2.4`, exit 0, **twice in a row from separate fresh shells**; stronger check: with the managed pth deliberately `chflags hidden`, both runs still succeed via the shim.
- **Config round-trip:** `save_config`/`load_config` YAML round-trips `lyapunov_penalty: pos` + `langevin_noise: fdt` correctly.

## Git footprint
- Branch: `agent/experiment-engineer/fix-pack-2` off local `main` (`d2d2401`, post first-fixes merge). Rebased: already up to date (main unmoved). Working tree clean. Not pushed, no PR (per task).
- Commits: `bffceac` (Lyapunov penalty) · `85f13d6` (FDT flag) · `7a6e437` (MNIST seed) · `58e7d3b` (epochs wiring) · `9fa1c35` (format new tests) · `3a407c3` (Makefile fix-env).
- Files touched: `chlu/core/regularization.py`, `chlu/core/integrators.py`, `chlu/core/chlu_unit.py`, `chlu/config.py`, `chlu/training/train.py`, `chlu/training/train_generative.py`, `chlu/data/mnist.py`, `chlu/experiments/exp_a_stability.py`, `chlu/experiments/exp_b_noise.py`, `chlu/experiments/exp_c_dreaming.py` (one call site, fix 3), `Makefile` (new), `tests/test_regularization.py` (new), `tests/test_langevin_fdt.py` (new), `tests/test_experiment_epoch_wiring.py` (new), `tests/test_data.py` (+1 test).
- Untracked side effects in the local env (not committed): `zzz_chlu_dev.pth` shim installed in this repo's `.venv`; `.venv` recursively un-hidden. **Other agents' worktrees untouched.**

## Open questions / follow-ups / risks
1. **Exp-C dream-phase rollouts are NOT wired to `langevin_noise`** (scope guard: "no experiment logic changes beyond (4)"). If someone trains Exp C with `fdt`, the dreaming `stochastic_rollout` calls still default to `"legacy"` ⇒ train/sample temperature-scale mismatch. One-line-per-call fix in `exp_c_dreaming.run_dream_batch`; recommend bundling into the §7.9 imbalance-test task for results-analyst.
2. **Default flip on fix 1 is a training behavior change** for any new A/B run (mandated by the task): λ_lyap=0.01 × a live `"max"` penalty. Whether it helps/hurts MSE convergence is unmeasured — candidate ablation (`max` vs `none` vs `legacy_degenerate`) for the analyst before the next paper-grade A/B retrain.
3. **UF_HIDDEN flag-applier not fully identified** — mechanism + durable immunity are proven, but *which* session component sets the flag (Zed-ACP harness? sandbox profile?) remains open. If the Head can run `ls -lO .venv/lib/python3.11/site-packages/*.pth` from a plain Terminal after a fresh `uv sync --reinstall-package chlu`, that would bisect human-shell vs agent-harness.
4. **4 concurrent worktrees observed** (`CHLU-csf3-runbook`, `-f2-eval-harness`, `-v1-l0-gate`, `-v2-so2-build`). `CHLU-v1-l0-gate/.venv` currently HAS the hidden editable pth ⇒ that spoke will hit §7.12 on `uv run chlu`/plain `uv run python -c "import chlu"`. I did not touch their trees; they should run `make fix-env` from their worktree root (target exists once they rebase onto this branch/main, or apply the two commands manually).
5. FDT `"fdt"` mode samples the Gibbs measure of the *shadow* Hamiltonian (O(ε²)-biased in H) — fine for our use; Metropolis-adjust or BAOAB only if exactness ever matters (F5 Prop-9 note).
6. Two uv installs coexist (`~/.local/bin/uv` = 0.7.3, `miniconda3/bin/uv` = 0.7.15); this repo's venv was created by 0.7.3. Harmless now, but PATH-dependent uv version is a reproducibility wrinkle worth a line in any runbook.

## Proposed handover updates (for the Hub)
- **§7.6 → RESOLVED on `agent/experiment-engineer/fix-pack-2`** (`bffceac`): `training.lyapunov_penalty` default `"max"`; legacy kept as `"legacy_degenerate"`. Note the intentional new-run behavior change + reproduction recipe.
- **§7.9 → FIX IMPLEMENTED behind flag** (`85f13d6`): `training.langevin_noise` default `"legacy"`; `"fdt"` = exact σ*_i with per-kinetic-mode M_eff via new `CHLU.effective_mass()`. Verified Var(p)=[0.2509,1.0011] vs [0.25,1.0]. **Gap:** Exp-C dream rollouts still legacy-only (see follow-up 1). The §7.9 MNIST-imbalance conjecture test is now unblocked.
- **§7.10 → RESOLVED** (`58e7d3b`): `--quick` now reaches CHLU *and* baseline trainers in exp A/B; exp C was already correct.
- **§7.11 → RESOLVED** (`7a6e437`): `load_mnist_pca(seed=...)`, exp C threads `project.seed`. Historical mnist* subsets remain unreconstructable (flag stays on §7.13 provenance).
- **§7.12 → RESOLVED (workaround-grade durable)** (`3a407c3`): root cause = UF_HIDDEN on fresh venv writes from *some* sessions + Python≥3.11 hidden-pth skip; fix = `make fix-env` (chflags heal + unmanaged pth shim). Update the §6 operational caveat: `uv run chlu` works after `make fix-env`; each new venv/worktree needs one `make fix-env`. Flag-applier identity still open (follow-up 3).
- **§3 config table:** `training` row gains `lyapunov_penalty="max"`, `langevin_noise="legacy"`.
- **New tests:** 32 total now (was 18): +5 regularization, +5 Langevin-FDT, +3 epoch-wiring, +1 MNIST-seeding.
