# v1-wormhole-routing — experiment-engineer report

Task + acceptance criterion: build `exp-v1-wormhole` — energy-gated sparse non-local routing on a chain lattice of CHLU associative memories (5 arms, routing-selectivity + cost-vs-accuracy measures, tests: gate monotonicity + closed-gate bit-equal reduction + routing smoke). Acceptance: runs `--quick`, tests pass, honest reporting of whether the gate routes selectively.

Status: **done** (code + 6 tests + full 2-seed run at N∈{4,8}; numbers below).

**One-line verdict:** the mechanism works and the headline holds — **energy-gated wormhole routing combines local-only's local accuracy with dense's distant accuracy at equal-to-dense cost and ≪ chain cost, with a highly selective gate (AUROC R0→distant 0.95–0.96)**; the honest caveat is that the *routing mechanism* (when it routes) delivers distant answers perfectly, but the smooth z-gate mis-routes ~10–12% (learned calibration head is a better router — re-confirms the v1-pivot "raw R needs calibration" finding).

## What I did
- **`chlu/experiments/exp_v1_wormhole.py`** (new): N-unit chain lattice, each unit a CHLU EBM written by generative PCD (reuses `train_generative`) over its own disjoint `[key‖value]` dictionary. Query at unit 0; answer local (unit 0) or distant (archive = unit N−1).
  - **Routing signal** = local residual `R0 = H0(settled) − floor0` (phase-1 governed relaxation of unit 0, reusing `exp_v1_gate._settle_batch`/`_decode_values`).
  - **Smooth gate** `g = sigmoid((z − t)/w)`, `z` = label-free median/IQR-normalized R0 (addresses §7 v1-l0-gate finding: raw R not cross-model comparable). Route iff `g > 0.5`.
  - **Wormhole** = a **gated KEY-channel spring** (`lattice.channel_spring_coupling`) that transports the clamped query key to the archive's free key half → archive relaxes to the matching pattern → **read the terminal (archive) value**. (Diagnostic-driven design decision — see Findings §2; a weak position coupling cannot drag a unit's *value* out of its own attractor basin, so "deliver into unit 0" is not physical; reading the retrieving unit is.)
  - **Joint relaxation** `_joint_settle`: `@eqx.filter_jit` governed Verlet on the concatenated lattice state, per-query gate weight held fixed during the rollout (H stays C¹ / conformally symplectic — F5 §7.4 no-ledger regime), position-only couplings, unit-0 key clamped. Common-dtype cast so it runs in both x32 (experiment) and x64 (tests).
  - **5 arms**: (a) local-only, (b) gated wormhole, (c) dense always-open (g≡1), (d) chain multi-hop (same gate, route through chain edges — key must diffuse N−1 hops), (e) calibrated τ-gate (write-time self-test fits `calibration.fit_calibration_head` on own-key vs impostor-key probes; route iff `p_route > 0.5`).
  - **Measures**: accuracy split local/distant, cost (unit-steps = Verlet steps × active units, a FLOP proxy; raw Verlet steps also logged), gate-selectivity confusion matrix + precision/recall, AUROC(R0→distant) and AUROC(learned-head→distant), bounded gate energy injected `g·V_wh`. Saves `.npz` + `summary.json`.
- **`chlu/config.py`**: new `ExperimentV1WormholeConfig` (+ registration in `CHLUConfig`/`load_config`/`save_config`).
- **`chlu/utils/plotting.py`**: `plot_v1_wormhole_cost_accuracy` (headline: cost-vs-accuracy per N, overall + distant-only markers) and `plot_v1_wormhole_selectivity` (confusion matrix + R0 histogram by type + gate-vs-residual scatter).
- **CLI**: `chlu exp-v1-wormhole [--project|--seed|--quick]` (`experiment_cmd.py`, `experiments/__init__.py`).
- **`tests/test_wormhole.py`** (6 tests).

## How I verified
- `uv run ruff check chlu/ tests/test_wormhole.py`: clean; new files `ruff format`-clean.
- **`uv run pytest tests/test_wormhole.py`: 6 passed** (18–22 s). With `test_config.py` + `test_lattice.py`: **23 passed**. Post-rebase re-verify: `test_wormhole.py`+`test_config.py` = **9 passed**.
  - gate monotonicity (smooth gate ↑ in drive; lattice `GatedCoupling` ↓ in base value, transmitted energy bounded); closed-gate bit-level reduction to independent units (`sensitivity=0 ⇒ γ=0`, `array_equal`); gate-weight-0 == edge-removed under the governor; open-gate-couples sanity; end-to-end routing smoke.
- CLI parser: `uv run python -m chlu exp-v1-wormhole --help` OK (§7.12 avoided via `python -m`).
- **Diagnostics run** (standalone, `uv run python -c …`): decode fidelity from exact stored pattern = 1.000; clean-cue retrieval 0.25→0.75 as epochs 60→400 (⇒ under-training, not a decode bug); **gated key-channel transport delivers the correct distant value to the archive at κ∈{0.5,2,5}** (archive decode `[80 81 82]` = truth) — the mechanism check that drove the read-terminal design.
- **Full run** (default config, N∈{4,8}, 2 seeds, 400 epochs/unit, laptop CPU): artifacts in `.claude/outputs/v1-wormhole-routing/full/` (`plots/*.png`, `results/exp_v1_wormhole_{metrics.npz,summary.json}`, `full_run.log`). Repro: `uv run python -c "from chlu.experiments.exp_v1_wormhole import run_experiment_v1_wormhole as r; r(seed=0)"`.

## Findings/results (full run; mean±std over 2 seeds)

**1. Cost-vs-accuracy — the headline (acc: overall / local / distant; cost in unit-steps):**

| arm | N=4 acc (L / D) | cost | N=8 acc (L / D) | cost |
|---|---|---|---|---|
| local-only | 0.500 (1.000 / 0.000) | 250 | 0.500 (1.000 / 0.000) | 250 |
| **gated wormhole** | **0.875±0.125 (0.875 / 0.875)** | **500** | **0.812±0.188 (0.896 / 0.729)** | **500** |
| dense always-open | 0.500 (0.000 / 1.000) | 500 | 0.500 (0.000 / 1.000) | 500 |
| chain multi-hop | 0.620±0.245 (0.875 / 0.365) | **750** | 0.609±0.026 (0.896 / 0.323) | **1250** |
| calibrated τ-gate | 0.906±0.094 (0.812 / 1.000) | 547 | 0.750±0.062 (0.500 / 1.000) | 625 |

- **local-only and dense each ceiling one query type and floor the other** (0.50 overall). **Gated routing gets both** (0.875 / 0.812 overall) — attention-like long-range access priced in energy. It dominates dense on accuracy at equal cost (500) and beats local-only massively on distant.
- **Chain multi-hop is strictly dominated**: worse distant (0.365 / 0.323 — lossy hop-by-hop diffusion through intermediate memories' potentials) at higher cost that **scales with N** (750→1250) while the direct wormhole edge stays flat at 500. This is the wormhole's core value proposition: **a direct 1-hop non-local edge beats N−1-hop diffusion**.
- **Cost nuance (honest):** at this 50/50 local:distant workload gated ties dense on mean unit-steps (both 500) but wins accuracy; on local-heavy workloads gated approaches local-only cost (250) because only distant queries pay the route. The "fraction of dense cost" claim is workload-dependent — reported as such.

**2. Gate selectivity — the routing-selectivity result:** the smooth energy gate opens selectively for distant-answer queries.
- N=4: **AUROC(R0→distant) = 0.954**, confusion `[[open·distant 84, open·local 12],[closed·distant 12, closed·local 84]]`, precision/recall = **0.88 / 0.88**.
- N=8: **AUROC = 0.960**, precision/recall = **0.90 / 0.90**.
- The **learned calibration head** ranks even better: AUROC(p_route→distant) = **1.000** at N=4 ⇒ calibrated arm distant accuracy = 1.000 at both N. This re-confirms the v1-pivot finding (raw/normalized R is a good-but-imperfect gate; a learned per-model head is the deployable router).

**3. Bounded gate energy injection (F5 §7.4 smooth-gate claim):** energy injected through the open gate `g·V_wh(q0_key, q_arch_key)` at the settled routed state: N=4 mean 1.69 / max 5.49; N=8 mean 1.20 / max 2.77 — finite/bounded, no runaway.

**4. Honest caveats (reported as demanded):**
- **The smooth z-gate mis-routes ~10–12%** (12/96 each way at N=4) ⇒ gated *distant* accuracy < 1.0 (0.875 / 0.729) even though the *mechanism* delivers distant perfectly when it does route (calibrated distant = 1.000). The short should claim "energy is a strong routing signal that a learned head sharpens", not "raw R routes perfectly".
- **Calibrated arm's local precision degrades at N=8** (local 0.500, acc_std up): the write-time head (impostor probes drawn from the other N−1 units) over-routes some local queries as N grows ⇒ at N=8 the balanced smooth gate (0.812) actually beats the learned head overall (0.750). Head training deserves an impostor-composition study before scaling N.
- **Retrieval needs adequate training**: quick mode (150 epochs, hidden 64) under-trains ⇒ low absolute accuracy (smoke only). Decode itself is exact (fidelity 1.0); the full-run numbers use 400 epochs / hidden 128 / embed_dim 12 / kv_per_unit 3. Laptop-scale, 2 seeds — indicative, not a powered claim.
- **Design deviation from the task's literal wording** (flagged): the task envisioned the gate "delivering the archived value INTO unit 0". The diagnostic showed a weak position coupling cannot overwrite a unit's own attractor (unit 0 keeps its local value), so I read the *terminal* (archive) unit the route reaches and let the wormhole transport the *query key* (not the answer). This preserves the physics (conformally symplectic, bounded energy) and the routing story; it's the honest realizable mechanism. Gate is residual-driven (not the `lattice.GatedCoupling` state-through-potential gate, which opens on endpoint *alignment* — a different trigger); I build on `channel_spring_coupling` as the base and test `GatedCoupling`'s bounded-energy monotonicity separately.

## Git footprint
- Branch **`agent/experiment-engineer/v1-wormhole-routing`** off `main @ b1782b0` (NOT pushed).
- Commits: `d566014` (feature: experiment + config + plotting + CLI + `__init__`), `4d428f4` (6 tests).
- **Rebase note (important):** the branch was initially created on top of 4 unrelated in-flight `fix-pack-3` commits (adaptive-K friction, exp-c langevin, config round-trip, ruff-format) that were concurrently added under me. I `git rebase --onto main`'d my two commits back onto clean `main`, dropping those 4 (one config.py conflict resolved by keeping my dataclass; the `CHLUConfig`/`load_config`/`save_config` registration hunks did not re-apply through the rebase and were re-added by hand — verified present + round-trip-tested post-rebase). Branch now = exactly my 2 commits on `main`; `git merge-base HEAD main == b1782b0`.
- Files touched (6): `chlu/experiments/exp_v1_wormhole.py` (new), `tests/test_wormhole.py` (new), `chlu/config.py`, `chlu/utils/plotting.py`, `chlu/cli/experiment_cmd.py`, `chlu/experiments/__init__.py` (additive hunks only).
- Commands: `uv run ruff check/format`, `uv run pytest` (6/9/23 passed as above), quick + full runs, standalone diagnostics. No unresolved conflicts.

## Open questions / follow-ups / risks
1. **Workload-mix sweep**: report cost-vs-accuracy as a function of the distant-query fraction (the "≪ dense cost" claim is strongest on local-heavy workloads — the realistic archive-is-rare regime). One-parameter extension.
2. **Learned-head impostor composition** (calibrated N=8 local drop): the write-time router should perhaps draw impostors only from the archive (the actual distant source) or calibrate its route threshold per N. Study needed before the head is the headline router.
3. **Capacity / difficulty band**: kv_per_unit=3, embed_dim=12 is easy; push kv/vocab to where local-only local accuracy < 1.0 and Hopfield errs, to stress the routing signal and add the Hopfield head-to-head (currently a union-memory reference only, not wired into arms — kept out of scope).
4. **Multi-archive routing / >1 wormhole edge**: this build has one archive + one gated edge (per task scope). Sparse top-k routing to several archives is the natural next step but needs the energy-ledger for hard selection (deliberately out of scope — smooth gates only here).
5. **3rd seed**: ran 2 for laptop budget; a 3rd would tighten the gated-distant std (0.125–0.188).

## Proposed handover updates (for the Hub)
- **§8/roadmap V1 (pillar 3 = wormholes) — evidence in:** energy-gated sparse non-local routing works at laptop scale (N∈{4,8}, 2 seeds). Gated routing **0.875/0.812 overall vs 0.50 for both local-only and dense** (combines their strengths at equal-to-dense cost); **chain multi-hop strictly dominated** (distant 0.33–0.37 at cost scaling 750→1250 vs the direct edge's flat 500) — *direct non-local edge ≫ multi-hop diffusion*, attention-like long-range access priced in energy. **Gate selectivity AUROC(R0→distant) 0.95–0.96**, precision/recall 0.88–0.90; **learned head AUROC 1.00** (re-confirms v1-pivot: raw/normalized R is a strong signal a calibration head sharpens). Gate energy injection bounded (F5 §7.4). Caveats: smooth gate mis-routes ~10–12%; learned head over-routes local at N=8; results indicative (2 seeds, easy difficulty band).
- **§2/§3 (code):** new experiment `exp_v1_wormhole` + `ExperimentV1WormholeConfig` group + CLI `chlu exp-v1-wormhole [--quick]`; 2 plot fns; reuses `lattice.channel_spring_coupling`, `exp_v1_gate._settle_batch/_decode_values/_auroc`, `training.calibration.fit_calibration_head`, `train_generative`. Tests 6.
- **Design note for the V1 short:** the honest realizable wormhole reads the *terminal* unit (archive) and transports the *query key* through a gated key-channel spring — it does NOT force-write a foreign value into the query unit (a weak position coupling can't leave a unit's attractor basin; diagnostic-confirmed). Frame the gate as residual-driven (route when local relaxation fails), distinct from `lattice.GatedCoupling`'s alignment-driven gate.
- **§3 config defaults (new group):** `experiment_v1_wormhole` — key knobs `n_units_values=[4,8]`, `embed_dim=12`, `kv_per_unit=3`, `n_seeds=2`, `train_epochs=400`, `hidden_dim=128`, `kappa_wormhole=2.0`, `gate_z_width=0.7`, `gate_route_threshold=0.5`, `relax_steps=route_steps=250`. Quick mode: N=4, 1 seed, 150 epochs.
- **Ops flag (git hygiene, for the Hub):** spawning this task from a thread sitting on `fix-pack-3` caused my branch to initially carry 4 unrelated in-flight commits; I rebased them off. If multiple engineer tasks run concurrently, spawn each from a clean `main` checkout (or use worktrees per protocol §3.2) to avoid this.
