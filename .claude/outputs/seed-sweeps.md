# seed-sweeps — results-analyst report
Task + acceptance criterion: power the w3 single-seed lattice/γ-field results at laptop scale — (1) banded-vs-uniform lattice training × 5 seeds × 2 budgets with verdict; (2) trained-coupling pricing on a 2-unit lattice; (3) S1 governor+field composed arm (3 seeds) + λ_protect/λ_hallu re-sweep; (4) cheap emergent-bias kick decomposition.
Status: **done** for items 1–3 (each re-run against a frozen commit for clean provenance); **item 4 deferred** (no emergent checkpoint on disk; needs a retrain — recipe below).

## Provenance / frozen manifest (READ THIS FIRST)
- **Frozen base commit: `b1782b0`** (main, w3 merged — the task's stated base). All headline numbers below come from a re-run inside a **detached git worktree** `../CHLU-seed-sweeps @ b1782b0` (protocol §3.2 isolation), driven with `PYTHONPATH=../CHLU-seed-sweeps uv run --no-sync python …` so `chlu` resolves to the frozen tree.
- **Why the re-run:** ⚠ a parallel `agent/experiment-engineer/v1-wormhole-routing` (carrying the fix-pack-3 commits) was **editing the shared working tree while my first pass ran** (reflog: `friction_field.py`, `train.py`, `goldstone_harness.py`, `config.py` committed 23:05–23:17, overlapping my runs). The first-pass code was therefore not attributable to a single commit. I re-ran everything against frozen `b1782b0`. **Items 1 and 3 frozen numbers are identical to the first pass** (Item 1 bit-for-bit; Item 3a/3b aggregates identical to 3 dp — the mid-edit friction_field/train deltas did not change any conclusion); **Item 2 frozen differs only in the 3rd–4th significant figure** (μ_rel²=4κ_eff/M rel-err ≤1.2% either way, same verdict). All tables below are the frozen `b1782b0` run.
- Repo read-only (analyst). No tracked-code changes. Scratch scripts: `.claude/scratch/seed-sweeps/{item1_banded_vs_uniform,item2_trained_pricing,item3_s1_extras,analyze_*}.py`. Artifacts (JSON + PNG): `.claude/outputs/seed-sweeps/`.
- Env: JAX warm-session, f32 for training (x64 for Item 2 spectra). All laptop-CPU; longest single run ≈ 3 min → no CSF3 needed.

---

## Item 1 — Banded vs uniform lattice training (Thread-5 falsifiable iii)

**Setup.** `chlu.experiments.exp_lattice` training smoke building blocks (`_build_training_lattice`, `train_chlu`, `_eval_rollout_mse`, `generate_two_timescale_orbits`), 2-unit CLU lattice, **2382 params**, spring coupling κ_c=0.01, hidden=32, dt=0.05, lr=1e-3, two-timescale data (ω=[0.5,2.0], data masses M=[4.0,0.25]). Banded vs uniform differ **only in the log_mass init** (banded = exact softplus-space ×[4.0,0.25] of the same random init ≈ (3.0, 0.19); uniform ≈ (0.7, 0.7)); architecture/params/key/data/budget bit-identical per seed. **5 seeds {0,1,2,3,4} × budgets {60, 300} epochs.**
Command: `python item1_banded_vs_uniform.py --seeds 0 1 2 3 4 --budgets 60 300`.

**Results (frozen `b1782b0`; == first pass bit-for-bit).**

| budget | eval rollout MSE (255 steps) banded | uniform | banded/uniform | banded wins | paired (uniform−banded) |
|---|---|---|---|---|---|
| 60 ep | **2.812 ± 0.327** | 3.751 ± 0.336 | 0.748 | **5/5** | +0.939 ± 0.101 |
| 300 ep | **1.180 ± 0.216** | 2.416 ± 0.345 | 0.490 | **5/5** | +1.236 ± 0.281 |

Final wake loss: 60 ep banded 1.68±0.49 vs uniform 2.45±0.36; 300 ep banded **0.395±0.085** vs uniform **1.696±0.131**.
**Crossover epoch = 1 for all 5 seeds at both budgets** — banded's wake loss is below uniform's from the first epoch and never crosses back.
Figure: `item1_banded_vs_uniform.png` (5-seed wake-loss curves + eval-MSE bars).

**Learned masses barely move from init** (300 ep, e.g. seed 0): banded (3.05, 2.60)/(0.185, 0.185) — stays banded; uniform (0.79, 0.67)/(0.73, 0.73) — **stays uniform**. The optimizer does not discover the 16× mass ratio on its own.

**Verdict (quotable):** *Banding reliably beats uniform at matched params — 5/5 seeds, both budgets (eval-MSE ratio 0.75 @60 ep, 0.49 @300 ep; paired gap +0.9…+1.2, ≈3–9σ). The advantage is present from epoch 1 (crossover=1) — the learnability-prior signature — and it widens with training. Uniform never recovers the mass hierarchy (learned masses stay ≈0.7); the hierarchy must be designed in.*
**Confound (state explicitly):** the banded init is matched to the ground-truth data timescales ([4.0,0.25]); the experiment shows *a correct designed prior wins and the optimizer cannot reach it from a uniform start*, not that arbitrary banding helps.

---

## Item 2 — Trained-coupling pricing (v3 follow-up 2)

**Setup.** Does a *learned* channel coupling still price communication by its curvature (F5 §7.2)? Designed 2-unit SO(2) pair (Mexican-hat channels, f=1, M=1, λ=1); the coupling is a **learnable** channel-aligned spring (static κ knob + learnable W=a·I; κ_eff = κ·a²). Data = γ=0 conservative rollouts of the fixed-W reference at κ_target (excites the relative mode); **wake-only training** (sleep disabled → no SO(2) vacuum erosion, per v2 Finding 0), window 256, 250 epochs, x64. κ_eff read from the mass-weighted Hessian of the *learned* V_c along the relative mode; μ_rel²/sync/n½ measured with the goldstone harness verbatim. κ_target ∈ {0.03, 0.1, 0.3, 1.0}.
Command: `python item2_trained_pricing.py --kappas 0.03 0.1 0.3 1.0 --epochs 250`.

**Results (frozen `b1782b0`).** Calibration: κ_eff extractor returns κ **exactly** on the fixed reference (0.05→0.050000, 0.2→0.200000).

| κ_target | κ_eff (learned) | μ_rel² meas | 4κ_eff/M (pred) | rel err | sync meas/pred | n½ meas/pred |
|---|---|---|---|---|---|---|
| 0.03 | 0.0440 | 0.1743 | 0.1759 | 0.9% | 77 / 75 | 313 / 347 |
| 0.1 | 0.0538 | 0.2150 | 0.2150 | 0.02% | 69 / 68 | 312 / 283 |
| 0.3 | 0.0525 | 0.2098 | 0.2100 | 0.1% | 71 / 69 | 404 / 290 |
| 1.0 | 0.0520 | 0.2054 | 0.2078 | 1.2% | 71 / 69 | 292 / 293 |

**Verdict (quotable):** *For a learned coupling, the curvature read from V_c predicts the communication observables pointwise — μ_rel² = 4κ_eff/M to ≤1.2% and sync = π/2μ_rel·dt to ≤3.6% across four trained lattices. The pricing-by-curvature map survives learning.* n½ carries the expected up-to-≈40% first-crossing/kick-phase-ripple scatter (F5 App-N).
**Limitation (state explicitly):** the κ_eff-**scaling** exponents are **inconclusive** here — supervised wake training does not move the (weakly gradient-coupled, slow) relative coupling far from init, so all four trained κ_eff cluster in 0.044–0.054 (1.22× range); fitted slopes (sync ^ −0.508, n½ ^ +0.32) span too small a range to read as laws (the near-−0.5 sync exponent is coincidental over 1.2×). The designed-lattice exponents (−0.499 / −0.986, v3-lattice-build) remain the authority; Item 2 confirms the *map*, not the exponent. The weak identifiability of the coupling strength from short-horizon supervision is itself consistent with the program's "hierarchy must be designed-in" theme. Figure: `item2_parity.png`.

---

## Item 3 — S1 extras (γ-field follow-ups 4 & 5)

**Common setup.** `chlu.experiments.exp_s1_gamma_field` machinery: Figure-8 attractor + structured off-attractor noise cluster at [1.5,1.5]; retention = clean-free-run curve coverage; rejection = position-return of 16 paired injections. FrictionField K=4 (learned), governor sensitivity 0.95. Base and learned models trained with buffer 30%-seeded at the noise locus (paired). Reimplemented only the ~10-line eval closure; reused the module's metric functions verbatim.

### 3a — Governor + field composed arm (3 seeds, 500 epochs)
Command: `python item3_s1_extras.py --mode compose --seeds 0 1 2 --epochs 500`. Frozen `b1782b0`.

| arm | retention (coverage) | ke_ratio | rejection (pos) | rejection (energy) |
|---|---|---|---|---|
| reference γ=0 | 0.712 ± 0.047 | 0.970 | 0.385 ± 0.055 | 0.000 |
| governor | 0.713 ± 0.046 | 0.969 | 0.737 ± 0.069 | 0.877 |
| oracle hole | 0.712 ± 0.046 | 0.968 | 0.839 ± 0.030 | 0.765 |
| learned K=4 (field only) | 0.506 ± 0.060 | 0.207 | **0.842 ± 0.018** | 0.832 |
| **governor + field K=4 (composed)** | 0.506 ± 0.060 | 0.206 | **0.480 ± 0.092** | 0.628 |

**Verdict (quotable):** *The governor and the learned field do NOT compose constructively. The composed arm keeps field-only retention (0.506) but its rejection collapses to 0.48 — worse than field-only (0.84) and worse than the governor alone (0.74), i.e. Pareto-dominated by both components.* Mechanism: multiplicative over-damping — the governor strips the kinetic energy that would carry an injected state back to the attractor, so combined with hole friction the garbage freezes off-manifold (position never returns → rejection_pos drops). **Recommendation: deploy the field alone; do not stack the governor on top.** (Reproduces the pilot's oracle≻governor and learned-K4-high-rejection/low-retention picture at 3 seeds on current code.)

### 3b — λ_protect / λ_hallu re-sweep (2 seeds, 150 epochs, 3×3 grid)
Command: `python item3_s1_extras.py --mode lambdasweep --seeds 0 1 --epochs 150 --grid 0.3 1.0 3.0`. Frozen `b1782b0`.

| seed | oracle (cov, rej) | K=4 dist-to-oracle (min…max) | γ_on_noise range | K=4 retention range |
|---|---|---|---|---|
| 0 | (0.685, 0.733) | 0.371 … 0.401 | 0.024 … 0.244 | 0.296 … 0.315 |
| 1 | (0.652, 0.709) | 0.358 … 0.359 | 0.0001 … 0.0002 | 0.322 … 0.328 |

**Verdict (quotable):** *No — within the coarse 3×3 grid the learned-K=4 point does not move toward the oracle. The λ_hallu/λ_protect ratio does drive friction onto the locus (γ_on_noise 0.02→0.24 as the ratio rises, seed 0), but retention stays pinned ≈0.30 vs oracle ≈0.67; the Pareto gap barely changes (dist 0.36–0.40).* The retention bottleneck is structural — long-horizon tail over-damping of the signal orbit (ke_ratio collapses) and seed-dependent locus discovery (seed 1's field never found the locus, γ_on_noise≈1e-4) — not λ-balance. This corroborates the γ-field report's own prescription: the fixes are **compact-support horizon gates + adaptive-K spawning**, not re-weighting. (Note: diagonal grid points λ_p=λ_h are identical because Adam is invariant to a global loss scale — only the ratio matters.) Figure: `item3_pareto.png`.
**Caveat:** 150-epoch models undertrain retention (0.30 here vs 0.51 at 500 ep in 3a); the λ-*independence* is the robust finding, the absolute retention is budget-limited.

---

## Item 4 — Emergent-bias decomposition (deferred)
**Not run.** The +13% retention bias lives on the v2-full-runs *emergent* SO(2) checkpoints, but only `emergent_summary.json`/`emergent_ring_profiles.npz` were saved — **no model `.pkl` exists on disk**, so there is nothing to load. Doing it properly needs retraining an emergent (self-breaking) SO(2) EBM and a kick-size sweep; given items 1–3 consumed the laptop budget, this "cheap-if-time" item is deferred rather than rushed.
**Recipe for the next analyst pass (harness-ready):** train one emergent model via the exp-d/goldstone machinery (self-broken tilt, δ≈0.01–0.06); sweep kick amplitude ε∈{0.01…0.3} on the emergent near-flat mode with `perturb_and_track`; fit retention-bias(ε) = a·ε² + b. The ε→0 intercept **b** = the metric/settle residual (first-crossing kick-phase ripple, F5 App-N); the ε² slope **a** = the anharmonicity contribution (softening restoring force). This cleanly splits the +12–15% into settle-residual vs anharmonicity.

---

## How I verified (commands + real output)
- Item 1 frozen == first pass **bit-for-bit** (e.g. seed0/300/banded 0.4160/1.2773 in both) → collision did not affect it.
- Item 2 calibration exact (κ_eff(fixed)=κ to 6 dp); μ_rel²=4κ_eff/M to ≤1.4% on 4 trained lattices.
- Item 3a/3b run at frozen `b1782b0`; per-seed numbers in `item3_compose_frozen.json` / `item3_lambdasweep_frozen.json`.
- Git footprint: **no tracked-code changes, no commits, no branch.** Created and then **removed** a detached worktree `../CHLU-seed-sweeps @ b1782b0` for the isolated re-run. Main working tree left untouched (it is on the parallel agent's `v1-wormhole-routing` branch, which advanced 608cf43→db35494 under its own agent while I worked — I did not commit, checkout, or reset anything there).

## Open questions / follow-ups / risks
1. **Parallel-agent write collision (ops risk):** two agents shared one working tree; my first pass imported code mid-edit. Mitigated by the frozen-worktree re-run, but the Hub should enforce **worktree isolation for analyst runs whenever an engineer is live on `chlu/`** (protocol §3.2), even for read-only analysts.
2. **Item 3 field retention gap** is now triple-confirmed structural (not λ): recommend the engineer's compact-support-gate + adaptive-K work be evaluated by re-running 3a/3b once fix-pack-3 lands on main.
3. **Item 2 exponent** needs either (a) longer-horizon / relative-mode-dominant supervision, or (b) directly initializing a κ_eff sweep, to move learned couplings across a decade of κ_eff and fit the −½/−1 laws on *trained* models.
4. Item 4 recipe above is ready for a short follow-up.

## Proposed handover updates (for the Hub)
- **§1.6 / §8 V3 (Thread-5 falsifiable iii — PROMOTE from single-seed to powered):** banded ≻ uniform at matched params (2382), **5/5 seeds, both budgets**: eval-rollout-MSE ratio 0.748 @60 ep / 0.490 @300 ep, paired gap +0.94±0.10 / +1.24±0.28; **crossover epoch = 1 (early-advantage learnability signature)**; learned masses stay at init (uniform never reaches the hierarchy) — **4th corroboration of "mass hierarchy must be designed-in, not awaited."** Confound: banded init = ground-truth timescales.
- **§8 V3 (follow-up 2):** trained-coupling pricing — **learned V_c prices communication by its curvature pointwise** (μ_rel²=4κ_eff/M to ≈1%, sync to ≤6% on 4 trained lattices); κ_eff-scaling exponent inconclusive because learned couplings cluster near init (coupling strength weakly identifiable from short-horizon supervision).
- **§8 Thread-1 / S1:** (a) **governor+field composition is a NEGATIVE result** — Pareto-dominated by both components (composed rej 0.48 vs field-only 0.84 / governor 0.74; retention unchanged 0.506), 3 seeds; over-damping freezes garbage off-manifold. Use the field alone. (b) **λ re-sweep: learned-K=4 does NOT move toward the oracle** (retention pinned ≈0.30 vs oracle ≈0.67; λ ratio only moves γ_on_locus); bottleneck is tail over-damping + locus discovery → validates the compact-support-gate/adaptive-K roadmap, not λ tuning.
- **§7 (ops):** log the analyst-vs-engineer shared-worktree collision (2026-07-06 ~23:05–23:17) and the frozen-worktree mitigation; recommend mandatory worktree isolation for analyst runs during live `chlu/` engineering.
- **Flag for experiment-engineer:** none code-breaking; my scripts hit no bugs. Item-3 numbers should be re-confirmed after fix-pack-3 (compact-support gates + adaptive-K) merges — the retention gap is the thing those fixes target.
