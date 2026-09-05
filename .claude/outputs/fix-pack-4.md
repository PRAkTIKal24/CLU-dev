# fix-pack-4 — experiment-engineer report
Task + acceptance criterion: ship the V(data)-energy anchor as first-class `training.anchor_data_energy_lambda` + 4 small accumulated fixes; **acceptance = full suite green (main venv), config round-trip green, each flag default = bit-compatible legacy.**
Status: **done** — 189 passed, 1 skipped (full suite, main venv); all 4 new flags/guards default to bit-compatible legacy behavior with tests pinning that.

## What I did
1. **`training.anchor_data_energy_lambda` (default 0.0 = off, bit-compatible)** — the V(data)-energy anchor wake term `λ·(mean_i V(anchor_i) − target)²`, ported faithfully from the validated anchor-robustness driver (`.claude/scratch/anchor-robustness/run_anchor.py` → `sleep-erosion-study/driver.py`):
   - `config.py`: new `TrainingConfig` field + full docstring (envelope, orthogonality-to-volume-conservation caveat).
   - `train.py` `train_chlu`: `anchor_target` captured ONCE from the **initial (epoch-0) model** as `mean V(data[:,0,:dim])`; wake `loss_fn` adds `anchor_λ·(mean V(anchor_data) − anchor_target)²` using the live (differentiable) `model.potential_net`. Guarded by a **Python-float static branch** (`if anchor_lambda > 0.0`) so λ=0 is a never-entered path → bitwise-identical to the legacy loop.
   - `exp_d_goldstone.run_experiment_d`: new `anchor_lambda` kwarg → `config.training.anchor_data_energy_lambda`.
   - CLI: `chlu exp-d --anchor-lambda F`.
2. **`chlu exp-v1-regime --train-epochs N`** — overrides `experiment_v1_gate.train_epochs` (the compute-parity knob; anchor-robustness P14: 500→2000 closes the Hopfield gap). One CLI flag makes the regime-remap reproducible.
3. **`experiment_v1_wormhole.impostor_policy ∈ {all_others (default), archive_only, neighbors_only}`** — new `_impostor_dicts(dicts, archive, policy)` helper replaces the hardcoded deployment impostor list at the calibrated-head fit site. Default `all_others` reproduces the legacy list **bit-for-bit** (archive-first ordering preserved); `archive_only` = the measured over-routing fix (local FP 53%→7%, v1-router-baseline finding 4).
4. **exp-d erosion guard** — `run_experiment_d` emits a loud `RuntimeWarning` (citing §7.14 / anchor-robustness + the cure) iff `sleep_mode!='off'` AND `train_epochs>300` AND `potential_type=='so2_invariant'` (designed vacuum) AND `anchor_data_energy_lambda==0.0`. UX-only; fires **before** `train_chlu`, no behavior change.
5. **`make fix-env`** — verified on current `main`: exits 0, `import chlu` OK, unmanaged shim `zzz_chlu_dev.pth` in place and un-hidden. **Already durable** (unmanaged `.pth` that uv never rewrites); no hardening needed.

Wiring note: the anchor CLI hook lives on `chlu exp-d` (where `--sleep-mode` is), per the task. The bare module entry `python -m chlu.experiments.exp_d_goldstone` was left unchanged for scope-minimality (it exposes neither `--sleep-mode` nor `--anchor-lambda`); anchor there is reachable via project `config.yaml`.

## How I verified (real output)
- `ruff check` on all 8 touched files → **All checks passed!**
- `pytest tests/test_anchor.py tests/test_exp_d_guard.py tests/test_config.py` → **15 passed** (20.4s).
- `pytest tests/test_wormhole.py -k impostor` → **5 passed** (4 impostor + 1), 14.9s.
- **Full suite** `pytest -q --no-cov` (main venv, warm) → **189 passed, 1 skipped, 14 warnings** (178.5s). The 14 warnings are all pre-existing (`\i` DeprecationWarnings in vendored TSB-AD; one unrelated `Mean of empty slice` in `exp_v1_hopfield_gate`) — none from my code.
- CLI parse smoke: `exp-d --anchor-lambda 25 --sleep-mode on` → `anchor_lambda=25.0`; `exp-v1-regime --train-epochs 2000` → `train_epochs=2000`.
- `make fix-env` (main) → `OK: chlu importable; shim at .venv/.../zzz_chlu_dev.pth`.

## Findings/results
- **Anchor is genuinely wired, off-by-default is inert** (test_anchor, flag-provenance below): λ=0.0 run is bitwise-identical run-to-run across ALL PyTree leaves; λ=100 changes the trained parameters vs λ=0 (the anchor gradient reaches V's params). Config round-trip preserves λ.
- **impostor_policy default = legacy**: `_impostor_dicts(["u0".."u4"], 4, "all_others") == ["u4","u1","u2","u3"]` (the exact pre-fix deployment list); `archive_only == ["u4"]`; `neighbors_only == ["u1","u2","u3"]`; N=2 `neighbors_only` falls back to `["u1"]` (no empty impostor set).
- **Guard**: warns in the erosion regime (epochs=400, sleep on, so2, no anchor); silent with anchor≥1, with sleep off, and at ≤300 epochs.

**Flag-provenance — behavioral anchor tests (`test_anchor.py`):**
| item | value |
|---|---|
| commit | b1f9524 |
| model | CHLU dim=4, hidden=8, potential=so2_invariant, kinetic=newtonian_learned |
| seeds | model PRNGKey(1), train PRNGKey(0), data PRNGKey(0) |
| training | epochs=8, **pure wake** (sleep_frequency=1000 > epochs), dt=0.05, batch=4, buffer=8, window=16 |
| lyapunov | penalty="max", λ_lyap=0.01 | langevin_noise legacy | persistent_sleep_buffer False |
| anchor λ | {0.0, 100.0}; anchor_target = epoch-0 mean V(data[:,0,:4]) |
| data | circle_vacuum n_points=8, seq_len=17, R=1.0 |

## Git footprint
- Branch: `agent/experiment-engineer/fix-pack-4` (off `main`@63fea62; rebase onto main = no-op; **not pushed**, left for review).
- Worked in isolated worktree `../CHLU-fix-pack-4` (main venv reused via `PYTHONPATH` per §4; removed after verifying the branch ref from the main repo shows all 3 commits). No file overlap with the concurrent `agent/experiment-engineer/paid-access-experiments` worktree (they touch core/potentials; I touched config/train/CLI/exp).
- Commits (atomic, tagged):
  - `7f4763b` fix-pack-4: add default-off config flags + CLI knobs — `chlu/config.py`, `chlu/cli/experiment_cmd.py`
  - `b1f9524` fix-pack-4: V(data)-energy anchor + exp-d erosion guard — `chlu/training/train.py`, `chlu/experiments/exp_d_goldstone.py`, `tests/test_anchor.py`, `tests/test_exp_d_guard.py`
  - `d9cce2d` fix-pack-4: wormhole impostor_policy on the deployed head — `chlu/experiments/exp_v1_wormhole.py`, `tests/test_wormhole.py`
- No unresolved conflicts.

## Open questions / follow-ups / risks
- The `router_mlp` wormhole arm reuses `other_dicts` (now policy-driven). With the default `all_others` it is bit-unchanged; under `archive_only`/`neighbors_only` the router MLP sees the same (policy-selected) impostor probe set as the calibrated head — intentional fairness, but worth confirming with the analyst before publishing a policy-sweep.
- `--train-epochs` combined with `--quick` on exp-v1-regime is capped by the in-experiment `min(train_epochs, 120)`; the flag is intended for non-quick reproducible runs (documented in help text).
- Anchor+guard are validated on the exp-d SO(2) path only (unit tests + the analyst's 3000-ep/5-seed study). No new long training run was performed this wave.

## Proposed handover updates (for the Hub)
- **§7 / new resolved items:** `training.anchor_data_energy_lambda` (default 0.0=off, bit-compatible) now ships in `train_chlu` — resolves sleep-erosion-study follow-up #1 and anchor-robustness recommended-next #4a. `experiment_v1_gate` `train_epochs` is now a CLI flag on `exp-v1-regime` (anchor-robustness #4b). `experiment_v1_wormhole.impostor_policy` ships (v1-router-baseline finding 4; `archive_only` = the local-FP 53%→7% fix). All three default to legacy behavior.
- **§7.14 / exp-d:** `run_experiment_d` now emits a loud erosion-regime warning (sleep on, >300 ep, designed so2 vacuum, no anchor) — the guard the erosion law implies. Default exp-d run (150 ep) is unaffected.
- **§7.12 / env:** `make fix-env` confirmed durable on `main`@63fea62 (unmanaged `zzz_chlu_dev.pth` shim intact; `import chlu` OK). No decay observed; no hardening required this wave.
- Full suite on `main`@63fea62 + this branch: **189 passed, 1 skipped**.
