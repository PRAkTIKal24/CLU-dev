# first-fixes — experiment-engineer report
Task + acceptance criterion: Clear §7 bugs (broken `data figure8/sine`, 3-way version mismatch), add `results/<exp>_metrics.npz` output, make sleep-buffer persistence configurable — repo builds, `uv run pytest -q` green (incl. new smoke tests), both CLI commands work, `--version` consistent, persistence switch works; atomic tagged commits.
Status: **done** (Sections A, B, C implemented + verified; D flagged below). One **pre-existing environment bug** discovered & worked around (see Findings §Env).

## What I did
- **A1** `chlu/cli/data_cmd.py`: replaced nonexistent `generate_figure8_data`/`generate_sine_data` imports with the real `generate_figure8(key, n_cycles, dt, scale)` and `generate_sine_waves(key, n_waves, steps, dt)`. Not a pure rename — reconciled to real outputs: figure8 works in whole cycles, so I generate `ceil(steps/steps_per_cycle)` cycles then truncate to `--steps`, split `[x,y,vx,vy]→q=[x,y],p=[vx,vy]`; sine splits `[x,dx/dt]→q,p`. Both save `q`, `p`, raw array, `dt`. Seed/dt/scale from default config. Added `tests/test_cli_data.py` (calls the cmd fns to a `tmp_path`, asserts shapes). Fixed 2 pre-existing **red** tests in `tests/test_data.py` that used the stale `steps=` kwarg → `n_cycles`.
- **A2** version single-sourced: `chlu/__init__.__version__` now reads `importlib.metadata.version("chlu")` (→ 0.2.4 from pyproject; fallback string if uninstalled); `chlu.py --version` prints `f"CHLU {__version__}"`. Removed hardcoded `0.1.0`/`0.2.3`.
- **B1** added a lightweight, uniform metrics-save to each `run_experiment_*`, written to `save_dir/../results/<exp>_metrics.npz`:
  - exp_a: `chlu_energy` (free-run H series), `energy_drift`, per-model trajectory MSE vs ground truth (`mse_chlu/node/lstm`), `dt`, `chlu_loss_history` (when trained).
  - exp_b: `sigmas`, `mse_chlu/node/lstm`, `target_energy`.
  - exp_c: `wake_loss`, `sleep_loss`, `total_loss`, `target_energy` (only when trained, not on pretrained load).
- **C** `training.persistent_sleep_buffer: bool = False` added to `config.py` (default preserves current Exp A/B CD-with-fresh-negatives behavior). Refactored `train.py` sleep phase: buffer sampling moved **outside** the jit (so persistence doesn't retrigger compilation — the buffer is a non-pytree static arg to `filter_jit`), `sleep_step` returns evolved states via `has_aux=True`, and when the flag is on the caller does `buffer.update((q_evolved,p_evolved), indices)` (mirrors `train_generative`). Added `tests/test_train_persistence.py` (spies on the internal `ReplayBuffer`; asserts contents change across epochs **iff** the flag is on). **Did not** run the A/B comparison (later results-analyst task).

## How I verified (real output)
- `uv run pytest -q` → **18 passed in 13.35s** (baseline was 2 failed / 13 passed; +3 new tests, +2 previously-red figure8 tests fixed). `train.py` coverage 10%→90%.
- `uv run ruff check <all changed files>` → **All checks passed!**
- CLI (after clearing the hidden flag, see Env note):
  - `uv run chlu --version` → `CHLU 0.2.4`
  - `uv run chlu data figure8 --steps 50 --output /tmp/f8.npz` → saved `q=(50,2) p=(50,2) trajectory=(50,4) dt=0.01`
  - `uv run chlu data sine --n-waves 4 --steps 50 --output /tmp/sine.npz` → saved `q=(4,50,1) p=(4,50,1) waves=(4,50,2) dt=0.01`
- **B1 end-to-end:** ran `run_experiment_a(save_dir=/tmp/expa_verify/plots, n_train_cycles=1, n_test_cycles=1)` to completion (exit 0, all plots too). Loaded `exp_a_metrics.npz`: `chlu_energy(125,)`, `energy_drift=2.06e-4`, `mse_chlu=0.0924 mse_node=0.0352 mse_lstm=0.8453`, `dt=0.05`, `chlu_loss_history(1000,)`.
- **C:** `test_sleep_buffer_persists_only_when_enabled` passes → buffer unchanged with flag off, changed with flag on.

## Findings/results
- The physics contrast is already visible in the exp_a metrics smoke run: CHLU energy drift ~2e-4 over the free run, LSTM trajectory MSE ~9× CHLU's — consistent with the paper's stability story (not a real run; train_epochs override didn't propagate, see below).
- **§Env — pre-existing broken editable install (macOS + uv + Python 3.11).** `uv run chlu <anything>` failed with `ModuleNotFoundError: No module named 'chlu'` **on my very first call, before any edit**. Root cause: the uv/setuptools editable marker `.venv/.../site-packages/_editable_impl_chlu.pth` (contents = project root path) has the macOS **`UF_HIDDEN`** BSD flag set (`st_flags=32832`); CPython 3.11 `site.addpackage` **silently skips hidden `.pth` files**, so the project root is never added to `sys.path` and the console script can't import `chlu`. `uv run python`/`pytest` are unaffected (they import via cwd on `sys.path[0]`). Workaround used for verification: `chflags nohidden <that .pth>` — but the editable finder **re-sets the flag during each run**, so it must be cleared immediately before each `uv run chlu` invocation (`PYTHONPATH=<repo> uv run chlu …` also works and is stable). This is an **environment/build issue, not code**, and it breaks **all** `chlu` CLI usage on this machine, not just the two commands I fixed. Also cleaned a stale `chlu.egg-info/PKG-INFO` (Version 0.1.8, untracked/gitignored) that was shadowing the site-packages metadata and making `importlib.metadata` return 0.1.8 instead of 0.2.4.
- **Out-of-scope quirk noticed:** `run_experiment_a/b`'s `train_epochs` override sets `config.experiment_a.train_epochs`, but `train_chlu` reads `config.training.epochs` (=1000), so the override (and `--quick`'s 50) does **not** actually shorten CHLU dynamics training. Pre-existing; not touched.

## Git footprint
- Branch: `agent/experiment-engineer/first-fixes` (off `main`; rebased on `origin/main` — up to date, no conflicts). Not pushed.
- Commits (oldest→newest):
  - `4942765` [experiment-engineer] fix broken chlu data figure8/sine CLI commands
  - `d94a5c5` [experiment-engineer] unify version strings to pyproject via importlib.metadata
  - `0d5d391` [experiment-engineer] make sleep-buffer persistence configurable (PCD switch)
  - `d2d2401` [experiment-engineer] write results/<exp>_metrics.npz from each experiment
- Files (11; +278/−42): `chlu/__init__.py`, `chlu/chlu.py`, `chlu/cli/data_cmd.py`, `chlu/config.py`, `chlu/training/train.py`, `chlu/experiments/exp_a_stability.py`, `chlu/experiments/exp_b_noise.py`, `chlu/experiments/exp_c_dreaming.py`, `tests/test_cli_data.py`, `tests/test_data.py`, `tests/test_train_persistence.py`.
- Deliberately **not** run: blanket `ruff format` — the repo is not ruff-format-clean at baseline (data_cmd.py/test_data.py/train.py had pre-existing trailing-whitespace/multiline style), and reformatting would balloon the diff into unrelated lines (§3.3). `ruff check` (lint) passes. Only my two brand-new test files were formatted.

## D — flagged, not changed (per task)
- **D1** Relativistic mode is not default for Exp A (`newtonian_identity`) / B (`newtonian_learned`); Exp II "velocity saturates at c" needs `relativistic` (finalA override). Unchanged.
- **D2** Confinement α: `PotentialMLP` adds `0.05‖q‖²`; `DeepPotentialMLP`/`ConvPotential` omit it. Unchanged.
- **D3** Lyapunov reg may be near-degenerate on a symplectic Jacobian ((σ,1/σ) ⇒ mean(log σ)≈0). For physics-theorist numerical probe — not touched.

## Open questions / follow-ups / risks
- The `results/` metrics location = `save_dir/../results`. For a project run this is `projects/<name>/results/`; for a no-project run save_dir='results' so it resolves to `./results/`. If results-analyst expects a different path, say so and I'll parameterize (`run_experiment_*` could take a `results_dir` arg, mirroring `models_dir`).
- The `UF_HIDDEN` editable-install bug (Env above) needs an **environment fix** to make `uv run chlu` reliable (candidate: recreate venv, or a uv setting / non-editable install, or a repo `chflags`/postinstall hook). Flagging for Head — it will bite any spoke that drives the CLI.
- Persistence flag builds the switch only; the A/B "does PCD change the energy landscape" comparison is a results-analyst task.

## Proposed handover updates (for the Hub)
- **§7.1 RESOLVED** — `chlu data figure8`/`sine` fixed (real generator APIs + q/p split + smoke tests). Commit `4942765`.
- **§7.2 RESOLVED** — version unified to 0.2.4 via `importlib.metadata`; `--version` prints `chlu.__version__`. Commit `d94a5c5`. (Also removed stale untracked `chlu.egg-info` that pinned metadata to 0.1.8.)
- **§7.3 RESOLVED** — each experiment now writes `results/<exp>_metrics.npz`. Commit `d2d2401`.
- **§7.4 ADDRESSED (switch built)** — `training.persistent_sleep_buffer` (default False). Commit `0d5d391`. Not yet studied (results-analyst).
- **NEW §6/§7 candidate — environment gotcha:** `uv run chlu` is broken on this machine because the editable `.pth` carries the macOS `UF_HIDDEN` flag (Python 3.11 `site.py` skips hidden `.pth`); the finder re-hides it each run. Non-CLI paths (`uv run python`, `pytest`) are fine. Every CLI-driving spoke should either `chflags nohidden .venv/lib/python3.11/site-packages/_editable_impl_chlu.pth` right before each call, or use `PYTHONPATH=<repo> uv run chlu …`. Needs a durable env fix.
- **Doc-drift note:** `run_experiment_*`'s `train_epochs` / `--quick` does not actually shorten CHLU dynamics training (train_chlu reads `config.training.epochs`). Not a first-fixes item; candidate for a follow-up wiring fix.
