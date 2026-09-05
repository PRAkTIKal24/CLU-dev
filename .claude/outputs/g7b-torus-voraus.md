# g7b-torus-voraus — experiment-engineer report
Task + acceptance criterion: fix the 2 CSF blockers (eval extra + episode-AUROC protocol) and build the LITERAL joint-angle→so2-coset torus map, runnable via `chlu eval --lattice`, smoked on real voraus with both arms reporting episode AUC-ROC, topology-match control pre-registered, CSF3 job sized; defaults unchanged; suite green.

Status: **done** (CSF3 job sized + launch recipe committed; **handed to the Head to launch** — I have no CSF3 access from here).

**First-10-lines flags (protocol §5 corollary):**
- **No downstream reconciliation list.** One design finding for the Hub/Head (below, "Findings 3"): at `κ_c=0.05` the topology-match control is **rank-preserving on the energy arm** ⇒ episode-AUROC-insensitive in the smoke; the flagship run should sweep `κ_c` and lean on the **predict** arm to give P3 a chance to resolve. Not a contradiction, a run-design input.
- **CM-3 honored throughout:** no energy-superiority claim. Smoke numbers are diagnostics with a flag-provenance table, not results.
- **PREREG.md written before any measurement** (`.claude/outputs/g7b-torus-voraus/PREREG.md`) — the CSF AUROC is not yet measured; the balanced smoke below is a *pipeline check*, explicitly not the floor.

## What I did
### Blocker 1 — CSF3 env missing the eval extra (`scripts/csf3/setup_env_job.sh`)
- `uv sync --frozen --extra cuda` → `--extra cuda --extra eval` (both frozen and fresh paths). Added a **jax-pin assertion** (`jax.__version__` must stay `0.9.x`) + prints `pandas/pyarrow/pyreadr` versions in the CPU sanity check, so a silent jax bump or a missing loader dep fails the setup job loudly instead of on the A100.
- **Verified locally:** the main venv already resolves `--extra eval` with **jax 0.9.0** unchanged, pandas 3.0.3 / pyarrow 24.0.0 / pyreadr 0.5.6 present. The extra is pure-Python/already-pinned wheels; it does not touch the jax pin.

### Blocker 2 — episode-AUROC protocol (verify + lock in)
- Confirmed **already structurally correct**: voraus `label_kind="episode"` → `PRIMARY_METRIC["episode"]="AUC-ROC"`; `EPISODE_METRICS=("AUC-ROC","AUC-PR")` — **VUS-PR is not even computed** for episode mode. `to_markdown` puts the primary (AUC-ROC) first+bold; the CLI summary prints `result.primary_metric`. `episode_reduce="mean"` (EvalConfig default) — **identical to what `voraus-baseline-floors` used**, so CLU and baselines are compared on the same per-episode mean reduction. No cross-protocol trap.
- Added a **regression test** (`test_episode_dataset_reports_aucroc_primary_not_vuspr`) asserting AUC-ROC primary, VUS-PR absent, markdown bolds AUC-ROC, and `episode_reduce=="mean"`.

### Science — the literal T⁶=U(1)⁶ map (LITERAL, per Head 2026-07-19)
1. **Joint-angle channels identified** (JAX-free parquet schema read): voraus 100 Hz = 7 meta + **130 machine signals** = 4 robot-level + 21×6 per-axis. The 6 physical joint angles are **`joint_position_1..6`** (not gear-reduced `motor_position`, not commanded `target_position`).
2. **`VorausTorusAD` + `embed_joint_angles`** (`chlu/data/industrial/voraus_ad.py`): cos/sin-embeds each `joint_position_j` on its ring, lays the six `(cosθ_j, sinθ_j)` pairs **FIRST**, appends every other machine signal raw as auxiliary channels. Exposes `n_so2_units` (=6). **Choice + why (stated):** angles-on-the-torus, everything-else-auxiliary — the clean first pass; velocities/torques/currents carry no `U(1)`, so forcing them onto cosets would be a false topology. Angles assumed radians; `(cos,sin)` is on the circle for any real θ, so limited-range pick-and-place sweeps still embed fine.
3. **`CLULatticeConfig.layout="literal"`** + **`_build_literal_lattice`** (`chlu/eval/clu_scorer.py`): builds a mixed-potential `CLULattice` directly — `n_so2` dim-2 `so2_invariant` coset units (each pair's `T¹` coset *is* that joint's `U(1)`) + `mlp` aux units tiling the remaining `C−2·n_so2` channels, the **LAST aux unit absorbing the non-divisible remainder** (voraus's fixed column set is not a clean multiple — **no padding**, per task). Only the angle units sit on the topology; aux units isolated.
4. **Coupling = `channel_spring` on coords (0,1)** (U(1)-preserving; the random-`W` `spring` breaks it — CM-9), `κ_c=0.05`. **Topology** = `chain` | `ring` (= the 6-axis serial arm's kinematic chain **closed into a 1-D torus** — the honest "torus" for 6 nodes; a 2-D torus needs a perfect square and correctly raises for n=6, guiding to `ring`) | `torus` (2-D).
5. **Topology-match control** (`shuffle_angles`): permutes which coset each bond connects → same units/channels/κ/training, physically **non-adjacent** joints coupled. Pre-registered P3 falsifier.
6. **Wired through the CLI**: `chlu eval --dataset voraus --lattice --lattice-layout literal --lattice-topology ring --score-mode default` (both arms) auto-swaps in `VorausTorusAD` and derives `n_so2_units` from the loader. New flags `--lattice-layout/-aux-dim/-shuffle-angles/-shuffle-seed`, `--lattice-topology` adds `ring`.

## How I verified (real output)
- **ruff** on all 4 touched .py files → All checks passed.
- **Targeted tests** `tests/test_eval_clu_scorer.py` → **24 passed** (14 orig + 10 new) in 172 s (CPU).
- **Full suite** `pytest -q tests/` → **326 passed, 14 warnings in 421 s.** (green; defaults unchanged.)
- **Literal-lattice build check** (C=27, n_so2=6): units `(2,2,2,2,2,2,4,4,7)` — 6 cosets + aux `[4,4,7]` remainder absorbed, `dim==27`; ring adds the `(5,0)` closing bond; all 3 arms finite. chain/ring × shuffle on/off all build+score.
- **Shuffle is a real control (not a no-op):** forward `H(q,p)` differs ordered 19.99 vs shuffled 20.80 (Δ=0.81, = the coupling reconnection in `V`: 3.17→3.97); edges `((0,1)…(5,0))` → `((3,2),(2,5),(5,4),(4,1),(1,0),(0,3))`. No threading bug.
- **REAL-DATA smoke on voraus** (local 1.04 GiB parquet, `--dataset voraus --lattice --lattice-layout literal --lattice-topology ring --score-mode default`): ran end-to-end, wrote npz+md+raw+roc. `--limit 8` gave a single-class slice → NaN (a `--limit` artifact, not a bug), so I ran a **balanced-subset** driver (15 anom + 15 norm episodes) — finite in-range AUC-ROC for every arm (below).

### Balanced-subset REAL-voraus smoke (PIPELINE CHECK, **not** the floor — CM-3)
Episode AUC-ROC primary; 30 test episodes, 40 train episodes, window 64, 40 epochs, literal ring, both arms:
| method | AUC-ROC (ordered) | AUC-ROC (shuffled control) |
|---|---|---|
| pca_recon | 0.7200 | 0.7200 |
| iforest | 0.3778 | 0.3778 |
| lof | 0.6178 | 0.6178 |
| knn | 0.5778 | 0.5778 |
| **clu_energy** | 0.6800 | 0.6800 |
| **clu_residual** | 0.5956 | 0.5956 |
| **clu_predict** | 0.6089 | 0.6089 |

All CLU literal-torus arms are **finite, in-range, above chance** on this tiny balanced slice — the acceptance evidence that the literal map is runnable and separating. Rankings here are meaningless at n=30/40-epochs (baselines undertrained too — cf. the 24-ch floors where knn led). **NOT a claim.**

## Findings / results
1. **Both CSF blockers fixed and tested.** Env extra + jax-pin guard; episode-AUROC protocol verified + regression-locked; reduction matches the floors (`episode_reduce=mean`).
2. **The literal joint-angle→so2 torus lattice is built, wired, and runs on real voraus** with both score arms reporting episode AUC-ROC. Non-divisible channel count handled by a remainder-absorbing final aux unit (no pad), as required.
3. **⚠ P3-control design finding (honest, for the flagship run):** at `κ_c=0.05` the ordered-vs-shuffled episode-AUROC is **numerically identical** on all three CLU arms in the smoke — *not* a bug (H demonstrably changes, Δ=0.81), but because a weak `channel_spring` reconnection shifts each episode's energy by a **near-constant offset** and **episode-AUROC is rank-based** (rank-preserving offset ⇒ same AUROC). Implication for the CSF flagship: (a) the **predict** arm (dynamics-sensitive, coupling forces enter the rollout) is the more sensitive P3 probe than **energy**; (b) consider a small `κ_c` sweep (e.g. 0.05, 0.2, 0.5) so the topology has a chance to move the ranking, else P3's honest verdict will likely be "topology-insensitive / null" and should be reported as such (a finding, per PREREG P3-null branch).
4. **Regime sanity (CM-9/CM-10, PREREG P5):** coupling is `channel_spring` ⇒ the reduced bond potential is a **pure first harmonic** `V=2κr*²(1−cosΔθ)`, so **J₂/J₁ ≈ 0 by construction** (verified to 2e-16 in `xy-lattice-theory`) — U(1) intact, not in the p=2-anisotropy regime. `κ_c=0.05` is exactly the value at which the priced-channel law was validated (CM-10, N≤16). The definitive trained-vacuum `κ/k_r` needs the CSF checkpoint's `r*` and radial curvature (probe post-run).

## CSF3 run (sized; handed to the Head to launch)
- No new job script needed — `scripts/csf3/job_gpu_eval.sh` already threads `EXTRA_ARGS`. Added the **flagship literal-torus recipe** to its header (3 seeds, ring, both arms) + the **shuffle control** + envelope. Flagship launch (per seed S∈{42,43,44}):
  `sbatch --export=ALL,DATASET=voraus,SCORE_MODE=default,SEED=$S,OUT=$HOME/scratch/clu_eval/voraus_torus_s$S,EXTRA_ARGS='--download --lattice --lattice-layout literal --lattice-topology ring --window 100 --train-stride 10 --stride 5 --metrics-mode fast --max-train-windows 100000' -t 12:00:00 scripts/csf3/job_gpu_eval.sh`
  Control: same + `--lattice-shuffle-angles --lattice-shuffle-seed $S`, `OUT=..._shuf_s$S`.
- **Envelope** (from `voraus-baseline-floors`): voraus ~2.5 GB loaded (torus-embed nets +6 ch), `train_stride=10` avoids the ~49 GB concat OOM; KNN/LOF scoring is the wall driver (`test_stride=5`, near-lossless for episode mean-reduce). gpuA `-n8` ≈ 83 GB RAM — ample. **Must** run the CSF env-setup job (Blocker-1 fix) first so pandas/pyarrow are present.
- **Pre-req to trust P3:** per Finding 3, add a `κ_c` sweep or read P3 primarily off the predict arm.

## Flag-provenance (balanced REAL-voraus smoke)
- **Commit:** `d49ed68` (branch tip) · **base** local `main` `e3c8931` (post-w15).
- **Env (reused main venv, §4):** jax **0.9.0**, equinox 0.13.4, optax 0.2.6; CPU (`JAX_PLATFORMS=cpu`), macOS laptop.
- **Seed:** 42 (harness + CLU init/train/subsample + balanced-subset choice).
- **Data:** voraus-AD 100 Hz, local parquet sha256 `c90ab1c7…`; 40 train episodes (PRE_A prefix), 30 test (15 anom + 15 norm, seeded balanced subset). `VorausTorusAD` → 6 cos/sin joint pairs + 124 aux = **136 channels**.
- **Harness flags:** window=64, stride=10, train_stride=30, metrics_mode=fast, max_train_windows=8000 (1381 used), episode_reduce=mean, primary=**AUC-ROC**.
- **CLU flags:** kinetic_mode=newtonian_learned, **lattice layout=literal, n_so2_units=6, aux_unit_dim=4, topology=ring, kappa_c=0.05, coupling=channel_spring, tie_channel_mass(so2)=True**, epochs=40, lr=1e-3, batch=64, max_fit_windows=4000, predict_horizon=16, relax_steps=32, residual_anchors=8, gamma=0.1, dt=0.05, momentum_init=finite_diff. shuffle_angles ∈ {False, True(seed=42)}.
- Artifacts (scratch, uncommitted): `.claude/scratch/g7b-torus-voraus/{smoke_ordered/*, balanced_smoke.py}`.

## Git footprint
- **Branch:** `agent/experiment-engineer/g7b-torus-voraus` off local `main` `e3c8931`; clean descendant (no rebase needed); **not pushed, left for review**. Worked in dedicated **worktree** `../CHLU-g7b-torus` per §3.2 (no concurrent collision; main checkout clean at start). Branch ref verified from the main repo before finishing.
- **Commits (4, atomic):**
  - `03f336a` csf3 env: add `--extra eval` + jax-pin assert (`scripts/csf3/setup_env_job.sh`)
  - `5ab92a5` literal joint-angle→so2-coset torus map (`chlu/eval/config.py`, `chlu/eval/clu_scorer.py`, `chlu/data/industrial/voraus_ad.py`)
  - `b9c1009` wire through `chlu eval` + CSF3 recipe (`chlu/cli/eval_cmd.py`, `scripts/csf3/job_gpu_eval.sh`)
  - `d49ed68` tests (`tests/test_eval_clu_scorer.py`)
- **Files:** 7 changed, +472/−19. All in-scope (`chlu/eval/**`, `chlu/data/industrial/voraus_ad.py`, `chlu/cli/eval_cmd.py`, `scripts/csf3/**`, `tests/**`). **`chlu/core/**` and `chlu/config.py` untouched.** Defaults preserved (`layout` default `"tile"`, `lattice=None` ⇒ single CHLU).

## Open questions / follow-ups / risks
1. **P3 falsifiability at κ=0.05 (Finding 3)** — the flagship should sweep `κ_c` and/or read the topology-match control off the **predict** arm; otherwise P3 will likely land in its pre-registered "null (topology-insensitive)" branch. This is the single most important input for the CSF run design.
2. **StandardScaler ring distortion (documented in `VorausTorusAD`):** the harness scales the `(cos,sin)` channels per-channel (fair protocol — CLU sees the same scaled data as baselines), affine-mapping each unit circle to an off-origin ellipse ⇒ per-unit SO(2) invariance only *approximate*. The lattice U(1) coupling structure and the shuffle control are scaling-invariant, so the falsification lever survives; but if we want an exact per-unit ring we'd need a scaler that leaves embedded angle-pairs un-centered (a harness change, deliberately NOT made — out of scope, would risk cross-protocol unfairness).
3. **Residual arm cost on full voraus:** 8 anchors × 32 relax steps × per-anchor `jax.grad(V_joint)` per window — heaviest arm; fine on A100/GPU, capped by `residual_anchors`.
4. **`κ/k_r` and `J₂/J₁` on a trained checkpoint** — J₂/J₁≈0 is by-construction; the quantitative `κ/k_r` Born-Oppenheimer ratio needs the CSF `r*`; probe post-run (cheap).

## Proposed handover updates (for the Hub)
- **§2/§3 (architecture/CLI):** new `chlu eval --lattice-layout literal` path; `VorausTorusAD` + `embed_joint_angles` (`chlu/data/industrial/voraus_ad.py`) — the literal joint-angle→so2-coset map; `CLULatticeConfig` gains `layout/n_so2_units/aux_unit_dim/shuffle_angles/shuffle_seed` and topology `ring`; `_build_literal_lattice` in `clu_scorer.py`. Defaults unchanged.
- **§8 / CSF3 runbook — Blocker 1 CLOSED:** `setup_env_job.sh` now syncs `--extra cuda --extra eval` (jax stays 0.9.0, asserted). voraus/TEP loaders will now find pandas/pyarrow/pyreadr on a compute node.
- **§1.6 / §5 — Blocker 2 CLOSED + regression-locked:** voraus is episode-labelled ⇒ **episode AUC-ROC is primary** (VUS-PR not computed); `episode_reduce=mean` matches the baseline floors. Any lingering "voraus VUS-PR" wording is wrong.
- **Flagship is runnable:** `chlu eval --dataset voraus --lattice --lattice-layout literal --lattice-topology ring --score-mode default` on CSF3 (recipe in `job_gpu_eval.sh`); PREREG committed. **Design caveat for the run (Finding 3):** sweep `κ_c` / read P3 off predict, else the topology-match control is rank-invariant on episode-AUROC at κ=0.05.
- **Suite:** now **326 passed** (10 new tests in `test_eval_clu_scorer.py`).
