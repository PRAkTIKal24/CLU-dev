# Task: fix-pack-3 — w3-discovered fixes (approved 2026-07-06)

- **Agent:** `experiment-engineer` · **Base:** `main` (post w3 merge, 136 tests) · **Branch:** `agent/experiment-engineer/fix-pack-3` · **Output:** `.claude/outputs/fix-pack-3.md`
- **Read first:** protocol · handover §10 (w3 entries) · the flagged items in `.claude/outputs/{v2-full-runs,gamma-field-build,generative-studies}.md`. Def-2 nomenclature.

**Acceptance:** all items implemented + tested, full suite green (≥136), atomic tagged commits, behavior changes flagged.

## Items (one commit each)
1. **Exp-D sleep-erosion guard (v2-full-runs Finding 0):** default `experiment_d.train_epochs` → **150**; add `experiment_d.sleep_mode ∈ {"on","off"}` (off = sleep_frequency→∞ wake-only path, the data-pinned regime) so the erosion study can switch cleanly. Document the erosion finding in the config docstring.
2. **Exp-D Noether guard:** `exp_d_goldstone.py:261`-area divides by r*→0 → return NaN + loud warning instead of silent NaN propagation.
3. **Harness `mu_floor` param:** `mode_amplitude(mu_floor=...)` with guidance ≈10×√(baseline residual μ²) in the docstring (v2-full-runs item 3 caveat).
4. **Adaptive-K hole spawning (gamma-field follow-up 1):** spawn rule = allocate a new hole where persistent-hallucination density (energy-gated, same gate as training) accumulates beyond threshold; prune holes whose γ_k decays below floor. Config-gated (`friction_field_adaptive_k: bool = False`). Smoke test: a noise locus missed by K=1 init gets a spawned hole within N epochs.
5. **Compact-support horizon gate (gamma-field follow-up 2):** optional gate variant with exact zero outside radius (e.g. smoothstep with hard cutoff) to close the tail-leakage retention gap; flag-selected, default sigmoid unchanged. Test: γ_φ exactly 0.0 beyond cutoff.
6. **Exp-C dream-rollout FDT wiring (fix-pack-2 follow-up 1):** `run_dream_batch` passes `noise_mode=config.training.langevin_noise` through `stochastic_rollout`. Default legacy = zero behavior change.
7. **Wang arXiv-id reconciliation (di-bernardo intel):** check 2606.24945 vs 2606.24946 (WebFetch the abstracts if reachable via Bash curl; else flag) and correct the handover/roadmap reference in your output's proposed-updates section.
8. **Config round-trip smoke as a permanent test:** `tests/test_config.py` gains an assertion that every `CHLUConfig` field group `is_dataclass` and full YAML round-trip equality (the w2/w3 merge-artifact killer — parts exist from v1-pivot; make it comprehensive).
