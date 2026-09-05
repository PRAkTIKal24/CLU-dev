# v1-router-baseline — experiment-engineer report

**Task + acceptance criterion:** add the boring physics-free baseline the wormhole must beat (learned-router-MLP arm, ≥5 seeds, non-50/50 workloads, FLOPs cost) + the impostor-composition study + small fixes; deliver the CM-7 verdict sentence *"the energy-gated wormhole beats/matches/loses-to a parameter-matched learned router at [conditions], in FLOPs, ≥5 seeds."*

**Status: done.** All 5 items delivered; 15 tests pass; full 5-seed run completed (N∈{4,8}, 3 workload mixes). **Verdict: the router WINS** (both outcomes were pre-declared publishable; this triggers the CM-2 reframe — see below).

---

## CM-7 VERDICT SENTENCE (acceptance deliverable)
> **The energy-gated wormhole LOSES to a parameter-matched (449-param, 2-layer, physics-free) learned router MLP — at both N=4 and N=8, across all workload mixes {50/50, 80/20, 95/5}, in FLOPs AND in accuracy, over 5 seeds.** Router 8.81e7 vs gated 1.18e8 FLOP/query; accuracy N=4 router **1.000±0.000** vs gated 0.887±0.139, N=8 router **0.948±0.070** vs gated 0.715±0.172. Router ranking AUROC(→distant)=**1.000** at both N vs energy-residual AUROC 0.865–0.963.

Per the task's own instruction ("if the router wins, V1's routing pillar reframes to certificates/interpretability per CM-2 logic — say so explicitly"): **the routing-decision pillar of V1 is a CM-2/CM-3 confirmation, not a physics win.** The CLU energy residual is *not* a better routing signal than a cheap learned classifier on the query embedding. What survives as a genuine mechanism claim (independent of the router) is the **1-hop-edge-vs-N-hop-chain** result, in FLOPs (below).

---

## What I did
1. **Learned-router-MLP arm (`router_mlp`)** in `exp_v1_wormhole.py`: a 2-layer MLP (`e→32→1`, 449 params) on the **raw query cue** — no energy, no relaxation. Fit write-time on the *same* own-key/impostor probe set as the calibrated energy head (`_router_probes`, AdamW logistic). Routes via the *same* direct wormhole edge; because its decision needs no settle, a routed query **skips phase-1** (reads the archive directly) — its honest FLOPs edge. Added to the arm battery run at N∈{4,8}.
2. **FLOPs cost model (§3)** — `_potential_grad_flops`/`_verlet_flops`/`_router_mlp_flops` replace unit-steps as the headline cost: `flops_grad_factor·(dim·h+h²+h)` MACs per potential value-and-grad, `flops_verlet_grads` grad evals/step, ×active-units. Wormhole routed leg = **2 units (flat in N)**, chain = **N units (scales)**. Per-arm FLOPs + a FLOPs-vs-accuracy plot.
3. **Workload realism (§2)** — 5 seeds (was 2); mixes {50/50, 80/20, 95/5} reported per-mix by **exact reweighting** of balanced per-query outcomes (route decisions are mix-independent because z-normalization uses a fixed pool — documented in code).
4. **Impostor-composition study (§4)** at N=8 — refits the calibrated head under {all_others, archive_only, neighbors_only} impostor sets, scores local-over-route / distant-recall / AUROC on deployment.
5. **Small fixes:** (a) `gate=tcfg.friction_field_gate` forwarded into both `FrictionField(...)` in `exp_s1_gamma_field.py`; (b) deduped the **F811** double `class ExperimentV1WormholeConfig` (kept the def the code reads — `gate_route_threshold`/`route_steps`/`kappa=2.0`); (c) added a wormhole config round-trip test.

## How I verified (real output)
- `ruff check` clean on all touched files; F811 confirmed gone (`ruff check chlu/config.py` → "All checks passed"). **plotting.py had pre-existing format drift** — I reverted a whole-file `ruff format` and re-applied ONLY my 2 hunks (48 insertions) to stay in scope (§3.3).
- **`pytest tests/test_wormhole.py tests/test_config.py` → 15 passed** (29 s, warm). New tests: FLOPs-model scaling/ordering (`chain>wormhole`, `N=8 chain == 2×N=4 chain`), router-MLP ranks impostor>own key, smoke asserts the new arm/mixes/router-AUROC present; wormhole config round-trip (nested `workload_mixes` + router knobs).
- **Full 5-seed run** (N∈{4,8}, laptop CPU, ~4 min warm): artifacts in `.claude/outputs/v1-router-baseline/full/` — `plots/{cost_accuracy,flops_accuracy,selectivity}.png`, `results/exp_v1_wormhole_{metrics.npz,summary.json}`, `full_run.log`. Repro: `PYTHONPATH=<repo> python .claude/scratch/v1-router-baseline/run_full.py`.

## Findings/results (5 seeds; mean±std)

### 1. Router beats every energy arm, cheaper (the headline)
| arm | N=4 acc (L/D) | N=4 FLOP/q | N=8 acc (L/D) | N=8 FLOP/q |
|---|---|---|---|---|
| local_only | 0.500 (1.00/0.00) | 5.88e7 | 0.500 (1.00/0.00) | 5.88e7 |
| gated (energy) | 0.887±0.139 (0.90/0.88) | 1.18e8 | 0.715±0.172 (0.82/0.61) | 1.18e8 |
| dense | 0.500 (0.00/1.00) | 1.18e8 | 0.448±0.070 (0.00/0.90) | 1.18e8 |
| chain | 0.652±0.159 (0.90/0.41) | 1.76e8 | 0.548±0.138 (0.82/0.28) | **2.94e8** |
| calibrated (energy head) | 0.860±0.122 (0.72/1.00) | 1.34e8 | 0.677±0.174 (0.46/0.90) | 1.49e8 |
| **router_mlp (no physics)** | **1.000±0.000 (1.00/1.00)** | **8.81e7** | **0.948±0.070 (1.00/0.90)** | **8.81e7** |

- Router AUROC(→distant)=**1.000** at both N; energy-residual R0 AUROC 0.963 (N=4) / 0.865 (N=8); calibrated head AUROC 0.999 / 0.907. **The embedding is (near-)linearly separable by which unit's key-cluster the cue sits in** (keys drawn disjointly per unit), so the router learns local-vs-distant trivially; the energy gate must *discover the same partition via relaxation* and does so more noisily. Physics buys nothing over the cheap classifier on this task → clean **CM-2/CM-3** ("energy ≈/< a learned signal; the mechanism is memory-agnostic").
- Router is also **cheapest of all routing arms** (8.81e7): it pays neither phase-1-on-route (unlike the residual gates) nor the extra units of the chain.

### 2. What still survives: 1-hop edge ≫ N-hop chain, in FLOPs (§3)
The direct wormhole routed leg is **flat in N (1.18e8)**; the chain scales — **1.76e8 (N=4) → 2.94e8 (N=8)** — and is *less accurate distant* (0.41→0.28). This "direct non-local edge beats hop-by-hop diffusion" claim is real and independent of the router comparison (the router uses the same direct edge). This is the defensible core of the wormhole idea; the *energy gating of it* is what the router dominates.

### 3. Per-workload-mix (§2): router dominates at every mix
| mix (L/D) | N=4 gated acc / FLOP | N=4 router acc / FLOP | N=8 gated | N=8 router |
|---|---|---|---|---|
| 50/50 | 0.887 / 1.18e8 | **1.000 / 8.81e7** | 0.715 / 1.18e8 | **0.948 / 8.81e7** |
| 80/20 | 0.893 / 8.96e7 | **1.000 / 7.05e7** | 0.778 / 9.49e7 | **0.979 / 7.05e7** |
| 95/5 | 0.895 / 7.56e7 | **1.000 / 6.17e7** | 0.810 / 8.36e7 | **0.995 / 6.17e7** |
Local-heavy mixes are cheaper for both (fewer routes); the router leads on accuracy AND FLOPs throughout.

### 4. Impostor-composition study (N=8) — why the calibrated head over-routes local
| impostor set | # units | local over-route | distant recall | AUROC |
|---|---|---|---|---|
| all_others (shipped default) | 7 | **0.533±0.333** | 1.000 | 0.907 |
| archive_only | 1 | **0.071±0.142** | 0.821±0.160 | 0.920 |
| neighbors_only | 6 | 0.458±0.251 | 0.858 | 0.900 |

**Cause identified:** training the head with impostors spanning the *whole* non-local pool (7 units) makes it flag **53%** of local queries as "route"; restricting impostors to the actual deployment distant source (archive_only) cuts local FP to **7%** at a modest distant-recall cost (1.00→0.82). This is the exact mechanism behind the calibrated head's N=8 collapse flagged in `v1-wormhole-routing.md` (over-routing local as N grows). It is a **probe-composition/deployment-mix mismatch**, not a physics defect — and it's *why the calibrated energy head (0.677) even loses to the plain smooth gate (0.715) at N=8*.

### 5. Bounded gate energy (unchanged, re-confirmed): mean 1.09 / max 5.49 (N=4), mean 0.96 / max 2.78 (N=8) — finite (F5 §7.4).

## Flag-provenance table
| field | value |
|---|---|
| commit (feature) | `52330f8` on `9339a13` (fixes), off `main` `9a13455` |
| seeds | 0,1,2,3,4 (base_seed=0, `n_seeds=5`) |
| **only non-default flag** | `experiment_v1_wormhole.n_seeds=5` (default 2) — everything else default |
| lattice/task | N=[4,8], embed_dim=12, embed_scale=2.0, vocab=128, kv_per_unit=3, trials_per_type=48 (Q=96), query_cue_noise=0.05 |
| memory write | kinetic=**relativistic**, potential=**mlp** (coercive), hidden=128, epochs=400, lr=1e-3, batch=16, k_steps=50, buffer=128, friction=0.3, temperature=0.3, input_noise_σ=0.05 |
| retrieval/routing | dt=0.05, relax_steps=250, route_steps=250, governor_sensitivity=0.95, kappa_wormhole=2.0, kappa_chain=2.0, gate_z_threshold=0.0, gate_z_width=0.7, gate_route_threshold=0.5 |
| calibrated head | probes_per_key=8, cue_noise_scales=[.05,.15,.3], features=r_margin, l2=1.0, p_route=0.5 |
| router MLP | hidden=32 (**449 params**), epochs=300, lr=3e-3, l2=1e-3, p_route=0.5 |
| workload_mixes | [[.5,.5],[.8,.2],[.95,.05]] |
| FLOPs model | flops_grad_factor=6.0, flops_verlet_grads=2.0 |
| model | rest_mass=1.0, speed_of_causality=1.0 |
| lyapunov / langevin_noise / anchor | N/A (no dynamics training; PCD write only, no MSE/Lyapunov) |

## Git footprint
- **Branch `agent/experiment-engineer/v1-router-baseline`** off local `main` `9a13455` (worktree per §3.2; removed after verifying refs from main repo). **Not pushed.**
- Commits: `9339a13` (small fixes: F811 dedup + friction_field_gate + config test), `52330f8` (router arm + FLOPs + mixes + impostor study + tests + plot).
- Files (6): `chlu/config.py`, `chlu/experiments/exp_s1_gamma_field.py`, `chlu/experiments/exp_v1_wormhole.py`, `chlu/utils/plotting.py`, `tests/test_config.py`, `tests/test_wormhole.py`.
- **Note (commit labeling):** the new wormhole config knobs (router/workload/flops fields) landed in the *fixes* commit `9339a13` (they were added to `config.py` in the same staging); logically they belong to the feature but are harmless there. No amend done to avoid rebase churn.
- **`git rebase main` → up to date** (no new commits on local main since branching; no conflicts). `origin/main` is a *divergent older ref* (`40c2f31 "rm docs/"`, lacks the wave-5 integration merge) — I based on **local main** `9a13455` per the task default, which is the active dev head. Flagging in case the Hub reconciles origin.
- Commands run: `ruff check/format`, `pytest tests/test_wormhole.py tests/test_config.py` (15 passed), quick smoke (1 passed), full 5-seed run (exit 0).

## Open questions / follow-ups / risks
1. **Task-difficulty ceiling drives the router's dominance.** Disjoint per-unit keys make the cue linearly separable, so the physics-free router is near-perfect. A **harder band** (overlapping key clusters / higher kv-per-unit / smaller embed_dim so local-only local-accuracy < 1.0) is the fair stress test of whether the energy signal *ever* adds value. Currently the honest read is "on the laptop testbed, it does not." (Ties to CM-8 / v1-hopfield-stress difficulty framing.)
2. **The chain-vs-1-hop FLOPs claim is the salvageable wormhole story** — recommend the V1 short lead the routing section with it (mechanism, physics-independent) and present the router result as the CM-2 boundary ("energy is not the routing signal; the direct edge is the mechanism").
3. **Impostor-composition is a deployable fix, not just a diagnosis** — archive-only (source-matched) impostors would materially improve the calibrated head; worth wiring as the default probe policy if the head is ever headlined.
4. `origin/main` divergence (above) — needs Hub/Head reconciliation before any push/merge.

## Proposed handover updates (for the Hub)
- **CM-7 (claims_matrix) — REWRITE required.** Current wording ("0.875/0.812 vs 0.50; gate AUROC 0.954–0.960; router-MLP baseline missing") is now: **energy-gated wormhole LOSES to a 449-param physics-free learned router in FLOPs+accuracy, 5 seeds, all workload mixes** (router 1.000/0.948 & 8.81e7 FLOP vs gated 0.887/0.715 & 1.18e8). N=8 gated is **0.715±0.172** (the old 2-seed 0.812 was optimistic). **Salvage = the physics-independent "1-hop edge (flat 1.18e8) ≫ N-hop chain (1.76e8→2.94e8, distant 0.41→0.28)" claim.** Reframe V1 routing pillar to CM-2 (memory-agnostic gate / mechanism = direct edge, not energy-as-signal) per the task's pre-declared branch. Blocker cleared: the router baseline now exists.
- **§7 Known Issues — two resolved:** (i) F811 duplicate `ExperimentV1WormholeConfig` deduped (kept the code-consistent def); (ii) `friction_field_gate` now actually forwarded into `exp_s1_gamma_field.py`'s FrictionField constructions (was silently ignored — any prior S1 run with a non-"sigmoid" gate did NOT use it; only "sigmoid" default was in effect, so past S1 results are unaffected).
- **§2/§3 (code):** `exp_v1_wormhole` gains a `router_mlp` arm, a FLOPs cost model, per-workload-mix reporting, and an impostor-composition study; new config knobs `router_*`, `workload_mixes`, `flops_*`; new plot `plot_v1_wormhole_flops`.
- **Negative-results registry (Charter C-9):** log "energy routing loses to a learned router on the laptop MQAR-style testbed" as a written negative feeding the CM-2 story.
