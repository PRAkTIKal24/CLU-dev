# v1-l0-gate — experiment-engineer report

Task + acceptance criterion: build S^(M) squeeze op + MQAR generator + CLU associative-memory cascade (F5 Def-7, single shell) with mandatory baselines, and deliver honest measured Q1 (residual-energy calibration) / Q2 (boost-retry recovery vs matched compute) numbers for the ≈Aug-7 V1 gate.
Status: **done** (code + tests + full-scale run + numbers; gate read below)

## What I did
- **`chlu/core/transforms.py`**: raw squeeze `S_ζ`, **mass-weighted `S^(M)_ζ = N⁻¹S_ζN`** (F5 §5.4), `effective_mass(model)` mirroring coded kinematics of all 3 kinetic modes (incl. the `+1e-6` guard in `CHLU.H`, so `∂q/∂ζ = p/M_eff` matches the unit's true velocity response), dense `squeeze_matrix` + `symplectic_form` for verification.
- **`chlu/data/mqar.py`**: MQAR generator per the Zoology spec (scout §3): vocab (default 8192), N, #KV, uniform/power-law gap distributions, published task semantics (target = value after key's first occurrence, `-100` elsewhere). Documented deviations: key/value half-vocab split with token 0 = PAD; per-sequence **injective** dictionary (values w/o replacement) to keep correctness decoding unambiguous. + `make_token_embeddings`.
- **`chlu/experiments/exp_v1_gate.py`** (+ config group `experiment_v1_gate`, CLI `chlu exp-v1-gate [--quick]`, 3 plot helpers appended to `utils/plotting.py`): per-episode pipeline — MQAR dictionary → fixed random embeddings → CHLU (relativistic, `mlp` potential, d=32=2×16) trained as EBM associative memory via **`train_generative` (PCD)** → retrieval by governed relaxation → **F5 Def-7 cascade**: line-searched (grid ζ∈±{0.15,0.3,0.6}, ×1.5^b per retry) S^(M) boosts, B=3 retries, ungated with per-stage records so **every τ policy is simulated post-hoc** (full calibration + compute curves from one run).
- **Arms**: `mass` (S^(M), energy-selected) · `raw` (mass-blind squeeze flag) · `kick` (**kinetically-matched random momentum kick** — same `pᵀM⁻¹p` injection as the paired squeeze candidate, no structured q-reframe; added beyond spec as the "any perturbation?" control) · `margin` (entropy/confidence-gated: same actions, readout-margin selection) · `relax-longer` (matched total Verlet steps) · **modern Hopfield** (softmax attention over stored patterns, matched content).
- **Design choices (assumptions, stated):** (1) **cue-conditioned retrieval** `clamp_key=True` — key half frozen, `p_k=0`; this is the legitimate sub-system Hamiltonian dynamics for all 3 kinetic modes and Hopfield's own setting; without it retrieval provably collapsed (base acc == 1/kv exactly). (2) `embed_scale=2.0` so data lives at the scale `train_generative`'s negative chains explore (buffer N(0,1), re-init U(−1,1), clamp [−1,1]) — at scale 1.0 the stored patterns were **not minima** (|∇V|≈0.5–0.8, states slid to an off-manifold pocket at ‖q‖≈2.5). (3) vocab **8192→256** for d=32 (documented). (4) Difficulty = (N, kv) grid; for a weight-based memory N only shapes the episode/query structure — the operative axis is kv (stated in code docs). (5) floor = `target_energy` (1st-pct stored energy) from `train_generative`; τ swept, not learned (per task).
- Compute accounting: settle(n) = exactly n Verlet steps; stage b costs G·150; boosts free; Hopfield = 1 matvec (reported as reference line, incommensurable).

## How I verified
- `uv run pytest -q` (worktree): **31 passed** (18 pre-existing + 13 new) in ~28 s. New: `test_transforms.py` — **‖SᵀΩS−Ω‖ ≤ 1e-12, |det S−1| ≤ 1e-12** (jax.enable_x64 ctx), exact `∂q_i/∂ζ|₀ = p_i/M_eff,i` (≤1e-10), matrix↔functional agreement, mass-blindness of raw S; `test_mqar.py` — semantics/shape/determinism/validation; `test_v1_gate.py` — cascade record shapes, exact cost bookkeeping, monotone best-R, clamped-relax freezes key half exactly.
- `uv run ruff check chlu/ tests/` → All checks passed; new files `ruff format`-clean.
- CLI parser: `python -m chlu exp-v1-gate --help` OK. (Ran experiments via `python -m`/`-c` — §7.12 UF_HIDDEN bug avoided; worktree venv synced fresh, jax 0.10.2 cpu, imports warm ~2 s.)
- Quick smoke (`quick=True`, ~2 min) then **full run** (seed 42, defaults): 5 levels (N,kv) ∈ {(64,4),(64,8),(128,16),(128,32),(256,64)}, 23 episodes/models, 288 queries, ~10 min wall on CPU. Artifacts: `.claude/outputs/v1-l0-gate/full/{plots,models,results}` (`exp_v1_gate_metrics.npz`, `exp_v1_gate_summary.json`, 3 PNGs, 23 ckpts).
- Repro: `uv run python -c "from chlu.experiments.exp_v1_gate import run_experiment_v1_gate as r; r(save_dir=..., models_dir=..., seed=42)"` on branch `agent/experiment-engineer/v1-l0-gate`.

## Findings/results (full run, seed 42)

| N | kv | trials | base acc | fidelity | Hopfield | AUROC R→wrong | AUROC margin→correct | n wrong |
|---|---|---|---|---|---|---|---|---|
| 64 | 4 | 32 | 1.000 | 1.000 | 1.000 | – (no wrongs) | – | 0 |
| 64 | 8 | 64 | 1.000 | 1.000 | 1.000 | – | – | 0 |
| 128 | 16 | 64 | 0.875 | 0.984 | 0.969 | **0.650** | **0.949** | 8 |
| 128 | 32 | 64 | 0.281 | 0.891 | 0.984 | **0.789** | 0.810 | 46 |
| 256 | 64 | 64 | 0.062 | **0.531** | 0.969 | **0.708** | 0.679 | 60 |

**Q1 — residual-energy calibration: qualified PASS.**
- Within-model, R ranks incorrectness meaningfully >0.5 at every level that has failures (0.650/0.789/0.708); reliability bins at kv=32 are textbook-monotone (acc 0.61→0.38→0.33→0.00 across R quantiles; plot `exp_v1_gate_calibration.png`).
- **Raw R is NOT comparable across models**: pooled raw AUROC = **0.330** (<0.5!); per-episode z-scored pooled = **0.567**. Each trained model has its own energy scale ⇒ a single global τ is meaningless — **empirically confirms the Thread-3 decision that τ must be a learned, per-instance (training-time) object.**
- **Honest caveat:** the trivial readout margin is as good or better at kv≤32 (0.949/0.810 vs 0.650/0.789; standardized pooled 0.641 vs 0.567); energy only edges margin at kv=64 (0.708 vs 0.679). Energy's *uniqueness* as the confidence signal is not established at this scale — it is *a* good signal, not *the* best one.

**Q2 — boost-retry recovery: FAIL at this scale.** Recovery on initially-wrong queries at full budget (B=3, matched compute):

| arm | kv=16 (n=8) | kv=32 (n=46) | kv=64 (n=60) | pooled (n=114) |
|---|---|---|---|---|
| **mass S^(M)** | 0.625 | 0.217 | 0.017 | **0.140** |
| raw squeeze | 0.750 | 0.217 | 0.033 | 0.158 |
| random kick (T-matched) | 0.875 | 0.174 | 0.050 | 0.158 |
| margin-selected | 0.750 | 0.174 | 0.050 | 0.149 |
| **relax-longer control** | 0.750 | 0.196 | 0.033 | **0.149** |

Mass-weighted squeeze retries do **not** beat matched-compute relax-longer (0.140 vs 0.149), nor mass-blind squeezes, nor kinetically-matched random kicks — the Sp(2d) structure of the retry buys no measurable retrieval recovery in a 32-dim single unit. Un-gated retries also **hurt** at easy levels (7 flips at kv=16: final 0.844 < base 0.875); τ-gating repairs this (below).
- Per-mode displacement scatter (Thread-5 falsifiable (ii)): **untestable on trained models** — learned M is near-uniform (log-std ≈ 0.08 across all 23 models, consistent with mass-spectrum-peek), so 1/M_eff has no dynamic range; Spearman(log|Δq_i|, log 1/M_i) = 0.03 (n=4608). Not a refutation — no x-axis. The exact operator response `∂q_i/∂ζ = p_i/M_eff,i` is unit-test-verified instead; mass-weighted ≈ raw in practice here for the same reason.

**Secondary positive — τ-gating is a real compute allocator (CALM/EBT-adjacent payoff, needs only the gate, not the boost):** post-hoc τ policy per level, min mean-cost achieving ≥ full-budget accuracy:
- kv=4: 1.000 @ **384** steps vs 3000 always-full (7.8×); kv=8: 1.000 @ 356 (8.4×); kv=16: **0.875 @ 342** — *above* the un-gated full-budget 0.844 and ≈ the 300-step control, at 8.8× less compute; kv=32: 0.391 @ 1931 (vs 0.406 @ 3000); kv=64: 0.078 @ 1847 (vs 0.078 @ 3000). Calibrated residual energy correctly *routes compute to hard queries and away from easy ones* even where retries can't fix the hard ones.

**Context/caveats:** kv=64 is **storage-limited** (fidelity 0.531 — half the pairs aren't stored; no inference policy can recover unstored answers), so the informative retry band is kv=16–32. Modern Hopfield ≥0.969 everywhere at ~1 matvec — dominates all dynamics arms at all budgets, exactly as the scout predicted must be reported; the CLU-EBM's failures are largely PCD-training-quality artifacts, not fundamental to the gate questions. Deterministic pipeline (no Langevin at inference), single seed (42); n=8 wrongs at kv=16 → that row is noise-level.

**Honest gate read (mine; Hub decides):** Per the task's guide — Q1 "meaningfully >0.5 across difficulties" **holds within-model** (0.65–0.79) with the per-model-τ qualification; Q2 "recovery > matched-compute control" **does not hold**. Net: **the gate half of V1 (trained per-instance residual calibration + escalation policy) is alive and quantitatively supported; the L0 squeeze-retry half shows no advantage over trivial perturbations or plain patience at single-unit scale.** If V1 proceeds, its defensible headline shifts to "calibrated energy-gated compute allocation on a conservative memory" (+ certificates as safety, not as recovery power); squeeze-retries would need either mass-differentiated models (which don't emerge — §7/Thread-5) or multi-shell escalation (V3 lattice) to earn their place. Ambiguity flag: results are one seed at small scale; kv=16 recovery cells have n=8.

## Git footprint
- Branch **`agent/experiment-engineer/v1-l0-gate`** (worktree `../CHLU-v1-l0-gate`, since removed; base `main` @ `d2d2401`, unchanged — rebase no-op). **Not pushed** per protocol.
- Commits: `9f5265f` S^(M) transforms + tests · `9885c72` MQAR generator + tests · `b5b7b49` V1 gate experiment + config + CLI + plots + smoke tests · `796f109` cue-conditioned retrieval + data-scale embeddings (quick-run pathology fix, probe-verified) · `cbf82bd` format + docstring fix.
- Files: **new** `chlu/core/transforms.py`, `chlu/data/mqar.py`, `chlu/experiments/exp_v1_gate.py`, `tests/test_transforms.py`, `tests/test_mqar.py`, `tests/test_v1_gate.py`; **edited (additive hunks only)** `chlu/config.py` (new dataclass + wiring), `chlu/cli/experiment_cmd.py` (new parser+cmd; did NOT touch `_get_config_and_paths` to avoid fix-pack-2 collision), `chlu/utils/plotting.py` (3 fns appended at EOF), `chlu/data/__init__.py`, `chlu/experiments/__init__.py`.
- Commands run: `uv sync` (worktree), `uv run pytest -q` (31 passed), `uv run ruff check/format`, quick + full experiment runs (outputs under `.claude/outputs/v1-l0-gate/`). No conflicts encountered.

## Open questions / follow-ups / risks
1. **Statistical power:** rerun kv∈{16,24,32} with ≥5 seeds / more episodes before any paper claim (recovery deltas are within noise; ~30 min CPU per seed).
2. **Learned τ task** (explicitly out of scope here) is now *empirically motivated*: per-model τ is mandatory (pooled raw AUROC 0.33). Margin-vs-energy as gate signal should be compared head-to-head inside that task — or combined (2-feature gate).
3. Retries were value-subspace-only under `clamp_key` — full-state boosts on an unclamped, better-trained memory remain unexplored (would need PCD quality fixes first; see §7.9 FDT fix in fix-pack-2 — retraining with corrected noise may change basin quality).
4. Hopfield dominance: any V1 short must show something Hopfield can't do (calibrated abstention/escalation is the candidate — Hopfield always answers); worth one experiment: Hopfield + naive confidence vs our calibrated gate on abstention metrics.
5. kv=64 storage ceiling suggests measuring capacity-vs-(d, hidden, epochs) before choosing the short's difficulty band.
6. Untested edge: `use_pretrained=True` reload path exercised only in quick iterations, not in the reported run.

## Proposed handover updates (for the Hub)
- **§8/roadmap V1 (gate evidence, empirical half):** L0 gate run complete (this file). Q1 qualified-pass (within-model AUROC 0.65–0.79; raw R not cross-model comparable → learned per-instance τ empirically mandatory; readout margin ≥ energy at kv≤32). Q2 fail at single-unit scale (S^(M) retries ≈ raw ≈ random kicks ≈ relax-longer; pooled 0.140 vs 0.149). Strong secondary: τ-gated cascade = compute allocator (≈8× steps saved at ≥ full-budget accuracy on easy/moderate levels). Recommended V1 pivot: headline = trained residual calibration + escalation *policy*; squeeze-retry demoted to certified-safety mechanism unless V3-scale shells revive it.
- **Thread-5:** falsifiable (ii) untestable on trained models — M stays near-uniform under generative PCD at this scale too (log-std ≈0.08, 23 fresh models; corroborates mass-spectrum-peek "latent tendency, not hierarchy"). Mass-weighted vs raw squeeze indistinguishable in effect for the same reason.
- **§2/§3 (code):** new module `chlu/core/transforms.py` (squeezes), `chlu/data/mqar.py`, experiment `exp_v1_gate` + config group `experiment_v1_gate` (defaults in dataclass; notable: `clamp_key=True`, `embed_scale=2.0`, vocab 256) + CLI `chlu exp-v1-gate [--quick]` (quick mode wired *inside* the experiment, works unlike §7.10's A/B path). Tests 18→31.
- **New known-issue candidate:** `train_generative`'s negative-chain scales are hardcoded for [−1,1] image data (buffer `initialize_random(scale=1.0)`, re-init `U(−1,1)`); EBM training silently fails to sculpt minima for data at other scales (bit this task; fixed at the data side via `embed_scale`). Consider config-exposing those scales.
