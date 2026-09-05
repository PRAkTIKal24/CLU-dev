# Task: g7b-torus-voraus — the literal joint-angle→torus-coset mapping + the two CSF blockers (w16, engineer)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/g7b-torus-voraus.md`
- **Read first:** protocol (§3.5; §5 flag-provenance) · **`.claude/outputs/clu-anomaly-scorer.md`** (the scorer bridge + `CLULatticeConfig` hook it built + Open-Q4 "literal mapping unbuilt") · **`.claude/outputs/voraus-baseline-floors.md`** (the two blockers below + the CSF3 sizing envelope + the episode-mode path it de-risked) · `chlu/eval/clu_scorer.py`, `chlu/eval/config.py` (`CLULatticeConfig`), `chlu/data/industrial/voraus_ad.py`, `chlu/core/lattice.py` (`build_lattice`, `torus_edges`, `channel_spring`) · `scripts/csf3/setup_env_job.sh` · `.claude/claims_matrix.md` (CM-3 forbidden; CM-9/10 coupling).
- **Git:** branch off local `main` (post-w15). Worktree per §3.2.
- **This is the flagship's last-mile.** The scorer bridge exists; this makes the *literal* G7b experiment runnable on CSF3. It gates the first real ICLR-systems result.

## Blocker 1 (from voraus-baseline-floors) — the CSF3 env is missing the eval extra
`scripts/csf3/setup_env_job.sh` runs `uv sync --frozen --extra cuda` only. **The industrial loaders need `--extra eval`** (pandas/pyarrow/pyreadr) or voraus/TEP will not load on a compute node. Fix: `uv sync --frozen --extra cuda --extra eval`. Verify the resolved env still pins jax 0.9.0 (the eval extra must not bump jax). **Without this, no CLU-on-voraus run can even read the data.**

## Blocker 2 (from voraus-baseline-floors) — voraus is episode-labelled → the primary metric is AUC-ROC, not VUS-PR
The harness already knows this (`PRIMARY_METRIC["episode"]="AUC-ROC"`), but the task/handover framing and any report text that says "voraus VUS-PR" is **wrong**. Ensure `chlu eval --dataset voraus` reports **episode AUC-ROC as primary** (per-episode score = a reduction over the episode's windows), with VUS-PR/AUPR as secondary. Confirm the reduction matches what `voraus-baseline-floors` used for the baseline floors so CLU and baselines are compared on the identical protocol. **Do not compare CLU-episode-AUROC against a baseline-window-VUS-PR** — that is exactly the cross-protocol trap the eval charter forbids.

## The science — literal T^n mapping (LITERAL first; learned is the fallback, per Head 2026-07-19)
voraus joint space is `T^n = U(1)^n`. The theory's falsifiable prediction: **match the CLU's coset to the data's own topology → n independent dissipation-proof registers.** Build the literal map:
1. **Identify the joint-angle channels** in voraus (the 6 robot-axis positions; the loader exposes the machine-signal columns). Map **each joint angle θ_j to one `so2_invariant` unit's coset** — i.e. the unit's `(q0,q1)` embeds the angle as `(cos θ_j, sin θ_j)` (or the raw angle on the ring), so the unit's `T^1` coset *is* that joint's `U(1)`. Non-angle channels (velocities, torques, temperatures) either feed additional non-`so2` units or condition the potential — **state your choice and why**; the clean first pass is angles-on-the-torus, everything else as auxiliary units.
2. **Couple the units on `torus_edges`** (the robot's kinematic-chain adjacency if defensible, else nearest-neighbour) with `channel_spring` (U(1)-preserving — the random-`W` `spring_coupling` breaks the symmetry, CM-9/lattice-xy-prereqs). Report `κ/k_r` and the `J₂/J₁` Born-Oppenheimer ratio so we know we're in the valid reduction regime.
3. **Wire it through `CLULatticeConfig` → `make_clu_scorers` → `chlu eval --lattice`** so the flagship runs as `chlu eval --dataset voraus --lattice torus --score-mode default`. The scorer's lattice hook currently **raises on non-exact tilings** — handle voraus's real channel count (it will not be a clean multiple; the loader's column set is fixed), don't just pad.
4. **Both score arms** (energy/residual + predict) as the scorer already supports — the anomaly-score choice is the experiment (Head), so run both, report AUC-ROC + ROC per arm.

## CSF3 run
- Size from `voraus-baseline-floors`'s envelope (voraus ~2.5 GB RAM loaded, 2122 episodes). Use `job_gpu_eval.sh` (built in w15) with `DATASET=voraus`, the literal-lattice config, seeds ≥3. **Smoke locally first** (`--limit`, `--quick`) to catch logic errors before the A100 — the Head's standing rule.
- **Pre-register** (PREREG.md) what "the literal torus map beats/ties/loses to the baseline floors" would look like, and what would falsify the topology-match hypothesis (e.g. a random-permutation-of-angles control that destroys the topology should do *worse* if the match matters).

## Acceptance
Both CSF blockers fixed (env extra + episode-AUROC protocol) and tested; the literal joint-angle→`so2` torus lattice built and runnable via `chlu eval --lattice`; a local smoke on real voraus with both score arms reporting episode AUC-ROC + ROC; the topology-match control pre-registered; CSF3 job sized and launched (or handed to the Head to launch); defaults unchanged; suite green. **CM-3: no energy-superiority claim — the honest CLU-vs-floor comparison on the identical protocol is the result, whatever it says.**
