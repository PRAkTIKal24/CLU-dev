# Task: fix-pack-4 — ship the anchor as a first-class flag + small accumulated fixes (w7, engineer, small)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/fix-pack-4.md`
- **Read first:** protocol (§4 worktree-venv rule) · `sleep-erosion-study.md` follow-up 1 + `anchor-robustness.md` (the anchor implementation lives in the analyst's validated scratch `driver.py` — port it faithfully) · `v1-router-baseline.md` finding 4 (impostor composition) · `anchor-robustness.md` recommended-next 4.
- **Git:** branch `agent/experiment-engineer/fix-pack-4` — worktree (concurrent with `paid-access-experiments`; you touch config/train/CLI, they touch core/potentials — minimal overlap, still isolate).

## Items
1. **`training.anchor_data_energy_lambda`** (default 0.0 = bit-compatible): the V(data)-energy-anchor wake term `λ·(mean_i V(anchor_i) − target)²`, target = epoch-0 mean V(data). Port from the anchor-robustness scratch driver (validated 3000-ep, 5 seeds). Round-trip + bit-compat + behavior tests. Wire into `exp-d --sleep-mode` paths.
2. **`experiment_v1_gate.train_epochs` exposed on the regime CLI** (`chlu exp-v1-regime --train-epochs N`) so compute-parity mapping is one flag (regime-remap-2000ep runs at scratch this wave; this makes it reproducible via CLI after).
3. **Archive-only impostor probe policy** as a config option on the calibrated wormhole head (`experiment_v1_wormhole.impostor_policy ∈ {all_others (default, legacy), archive_only, neighbors_only}`) — the measured fix for the over-routing collapse (local FP 53%→7%).
4. **exp-d default-epochs guard** (still open from v2-full-runs Finding 0 fixes? verify): if `sleep_mode != off` and epochs > ~300 with a designed vacuum and NO anchor, emit a loud warning citing the erosion law. One-line UX guard, not a behavior change.
5. Confirm `make fix-env` still holds on current main (fit-gap-anatomy had to re-apply `chflags nohidden` per call — if the shim decayed, harden it).

**Acceptance:** full suite green (main venv), config round-trip green, each flag default = bit-compatible legacy. Flag-provenance in tests where behavioral.
