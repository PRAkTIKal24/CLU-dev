# carried-remeasurements — results-analyst report

Task + acceptance criterion: run the three carried w25 items properly — (1) multi-seed the "τ is not binding in φ-space" observation, (2) occupancy-sweep the post-eviction allocator-trace leak, (3) the d=8 K=256 8×32 sharding probe if compute allows.
Status: **done for items 1 and 2 (both resolved, with a correction to the standing wording of BOTH w24 and w25); item 3 = probe launched, see §4 — reported as NOT RUN / partial with a measured cost, never as a null.**

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). THREE items.**
> 1. ⛔ **`cl-entry-build` §Item-3's line "⭐ τ is not the binding element of the gate in φ-space … the gate's threshold clause should not be quoted as load-bearing in the φ-space regime" is TOO GENERAL and must be narrowed.** It replicates *exactly* (3/3 seeds, max|Δ| = 0) — **but only at the hardest corruption level (p = 0.8)**. At p = 0.5, in the *same* φ store, the four τ curves differ and **τ = 1.0 costs −0.485 ± 0.064 accuracy (mean over 6 cells; −0.546 ± 0.023 end-of-stream) while spending 1.80× instead of 1.12–1.30× compute.** w24's clause is right there. **Owner: `doc-curator` (the §Item-3 wording + wherever the retry gate is described) + Hub (N-registry).**
> 2. ⚠ **`mia-decay-measurement` §2's headline `AUC(z_hole) = 0.99985` must be quoted with a load factor.** It is 0.99985 at 8 offers, **0.9961 at 6, 0.9165 at 4 and 0.6715 at 2** (capacity 8). The leak is real at every load but is *not* a general number. **Owner: `doc-curator` (§1.6/N99 wording) + `placement-landing` (its acceptance test must be stated at 8/8, else it can be passed by choosing a load).**
> 3. ⚠ **The task file's suggested command for item 1 (`chlu exp-cl-entry --items retry --baselines none`) does not run anything** — `retry_native` is nested inside `if "entry" in items` (`exp_cl_entry.py` L1053–1073). The working command is `--items entry,retry`. **Owner: `experiment-engineer`** (either de-nest the retry block or document the dependency).

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial:** compute-adaptive reads (item 1) + lifetimes/isolation (item 2) + capacity (item 3). This task **strengthens or retracts existing claims**; it opens none.
- **Laundering control:** item 1 carries the **kNN-in-φ floor at matched compute** (`feedforward_knn_phi`, reported below at every cell); item 2's control is the **paired-placement column**, which must be exactly 0.5000; item 3 carries the monolithic line at matched `K_total`.
- **Falsifies:** item 1 — the τ-identity fails to replicate at ≥3 seeds. Item 2 — the leak is **not** occupancy-driven (stays ≈1.0 at low load).
- **Does NOT falsify:** item 3 failing at d=8 K=256 (a pre-registered expected-FAIL; a FAIL is the confirmation).
- **Verdict on the falsifiers:** item 1's falsifier **does not fire at p = 0.8** (identity exact, 3/3 seeds, 6/6 cells) but **DOES fire at p = 0.5** (identity false, 6/6 cells) — so the observation is *scoped*, not retracted. Item 2's falsifier **does not fire**: `AUC(2/8) = 0.6715`, far from 1.0 ⇒ the leak **is** occupancy-scaled, which is the *better* of the two pre-registered outcomes for `placement-landing`.

---

## Flag-provenance table

| item | value |
|---|---|
| base commit | local `main` @ **`ff85573`** (`ff85573f0bc6dc2240297dcfbf8deaecae51ca45`) — **`git status --short` empty before AND after; no tracked file created, modified or deleted** |
| PREREG | `.claude/outputs/carried-remeasurements/PREREG.md` — main body written **before any harness ran**; **ADDENDUM 1** written after the seed-0 p=0.8 run and **before** any p=0.5 τ-sweep run (§1.3) |
| JAX / venv | **0.9.0**, main venv (`/Users/user/Desktop/CHLU/.venv`), no worktree, no `uv sync` (protocol §4) |
| **item 1** commands | `PYTHONPATH=. .venv/bin/python -m chlu exp-cl-entry --project w26tau{0,1,2} --items entry,retry --baselines none --seed {0,1,2}` (default `retry_mask_levels = [0.5, 0.8]`) · **and** `--project w26tauE{0,1,2} … --seed {0,1,2}` with a project `config.yaml` whose only edit is `retry_mask_levels: [0.5]` (⇒ the τ-sweep runs at p = 0.5; **configuration only, no code change**) |
| item 1 seeds | **0, 1, 2** on both arms; 2 store snapshots per seed (`mid_stream` = after task 3, `end_of_stream`) ⇒ **12 τ-sweep cells** |
| item 1 non-default flags | none in the `w26tau*` arm. In the `w26tauE*` arm: **`retry_mask_levels = [0.5]`** (only change). All else default: `dataset=mnist`, `n_tasks=5`, `classes_per_task=2`, `phi_regimes=[task1_only, generic_frozen]`, `phi_arm=pca`, `phi_dim=32`, `n_train_per_task=2000`, `memory_items=200`, `clu_s_frac=0.2`, `d_safe_mult=4.4`, `s_policy=refit`, `store_alpha=1e-3`, `clu_gamma=0.1`, `clu_steps=150`, `clu_dt=0` (auto), `clu_tail_frac=0.1`, `clu_kinetic_mode=newtonian_identity`, `retry_ladder=[0,1,2,4,8]`, `retry_tau=0.99`, `retry_tau_grid=[0.99,0.999,0.9999,1.0]`, `retry_boost=1.5`, `retry_step_frac=0.1`, `ff_aug_sigma=0.1`, `retry_mid_task=2`, `--baselines none` (baseline table skipped; the retry cell does not use it) |
| item 1 metrics | `projects/w26tau{0,1,2}/results/exp_cl_entry_mnist_metrics.json`, `projects/w26tauE{0,1,2}/results/…`; consolidated → **`.claude/outputs/carried-remeasurements/item1_tau_metrics.json`** (every number in §1 re-derived from these, none transcribed from stdout) |
| item 1 runtime | 3 seeds × 2 arms ≈ **11 min + 10 min** wall (≈3.3 min/seed), plus a 30 s `--quick` smoke |
| **item 2** command | `PYTHONPATH=. .venv/bin/python .claude/scratch/carried-remeasurements/occupancy_harness.py` |
| item 2 harness provenance | **copy-derived** from `.claude/outputs/mia-decay-measurement/mia_harness.py` (sha256 `075cc7847893c6dd8bd8f0c3dd716c917ae3f32ee6fc1f6da20cce471d70c292`, git base `ff85573`). **The original was NOT edited** (verbatim copy kept at `.claude/scratch/carried-remeasurements/mia_harness_SOURCE_COPY.py`; `placement-landing` depends on the original this wave). Only change: `N_BG` is a parameter and only the post-evict block runs |
| item 2 seeds / n | **0, 1, 2**; 8 targets × 3 seeds = **24 per-example values**; **128 paired IN/OUT worlds** per target; 16 queries per world; **4 load factors** |
| item 2 store (all shipped defaults, `experiment_controller_mvp`) | `AtomStorePotential(dim=3, capacity=8, α=0.02, s=0.35, s_pay=s, κ=1.0)`, `Controller(d_safe = 4.4·0.35 = 1.540, budget = 8, n_relocation_candidates = 400)`, proposal disk `R = radius_for_capacity(8, 1.54) = 2.2869`, packing bound **8.00**, read = shipped `two_phase` (`dt 0.05`, `γ_address 0.05×400 → γ_read 0.0×800`, tail 0.25, 8 subsamples), `payload_tol = 0.1000`, `amp_floor = 0.05` |
| item 2 metrics | **`.claude/outputs/carried-remeasurements/occupancy_metrics.json`** (all 24 per-example values per statistic per load); analysis `item2_analysis.txt`; figure `fig_occupancy.png` |
| item 2 runtime | **209.7 s**, exit 0, no NaN / divergence / OOM |
| **item 3** command | `PYTHONPATH=. .venv/bin/python -m chlu exp-sharded-store --project w26p4 --cells 8:256:8 --arms monolithic sharded_matched --items 1 --seed 0` — see §4 |
| N94 | **no training anywhere in item 2** (designed store, no epoch count applies). Item 1 trains only the frozen task-1 PCA φ and the baseline-free store build; no gradient training of any CLU potential |

---

## 0. Headline (five numbers)

1. **The τ-identity replicates exactly — at the hardest level only.** At p = 0.8, all four thresholds {0.99, 0.999, 0.9999, 1.0} give **bit-identical** ladders at every k ∈ {0,1,2,4,8}: `max|Δacc| = 0.000e+00`, `max|Δcompute| = 0.000e+00`, at **3/3 seeds × 2 store snapshots = 6/6 cells**.
2. **At p = 0.5, in the same φ store, τ is decisively binding and τ = 1.0 is catastrophic.** End-of-stream, k = 8: **0.9950 / 0.9900 / 0.9950** at τ = 0.99 vs **0.4171 / 0.4600 / 0.4650** at τ = 1.0 (seeds 0/1/2) — **−0.546 ± 0.023** accuracy for **+0.66×** compute. Over all 6 p = 0.5 cells: **−0.485 ± 0.064**.
3. **The discriminating variable is not φ-space vs pixel-space — it is the eligible-pool size against the retry budget.** Measured exactly from the compute multiplier: `#{cos₀ < 0.99}` = **24 / 28 / 31** (end-of-stream) and **47 / 52 / 60** (mid-stream) of ~200 at p = 0.5, versus **≥160 = the full 8·step_n budget** at p = 0.8. w24 and w25 measured the two ends of one axis; neither is wrong.
4. **The allocator-trace leak is strongly occupancy-scaled, and the w25 headline is the top of the curve.** `AUC(z_hole)` post-evict = **0.6715 ± 0.0405 → 0.9165 ± 0.0265 → 0.9961 ± 0.0040 → 0.99985 ± 0.00070** at 2 / 4 / 6 / 8 offers; `TPR@FPR 1 %` = **0.029 → 0.118 → 0.924 → 1.000**. The 8/8 cell reproduces `mia-decay-measurement` **element-for-element** (identical arrays, all six statistics).
5. **The paired-placement control is exactly 0.5000 ± 0.0000 at every load, on every statistic** (24 examples × 4 loads × 6 statistics). No harness bug. And `evict` still leaves the row verbatim at every load (`centers` max err **5.622e−8**, `payloads` **0.0**) — D1 unchanged.

---

## 1. Item 1 ⭐ — the τ observation at 3 seeds, and the regime it belongs to

### 1.1 The mechanism (pre-registered, and it is exact arithmetic, not a fit)

`exp_retry_compute._retry_ladder` gates on `eligible = (~locked) & (cos < τ)`, takes the `step_n = round(0.1·N)` lowest-cosine eligible reads per round, and **locks** them. Crucially it only ever *updates* the reads it retries — so an **unlocked read's cosine is frozen at its first-pass value**. Therefore, exactly:

> `total reads retried by round k = min(k · step_n, #{cos₀ < τ})`, and the compute multiplier is `1 + (retried)/N`.

Two consequences, both measured below:
- If `#{cos₀ < 0.99} ≥ k·step_n`, **every τ in the grid selects the same reads in the same order** ⇒ the ladders are identical. The identity is **structural, not stochastic** — which is why it replicates perfectly.
- If `#{cos₀ < 0.99} < k·step_n`, τ = 0.99 **auto-stops** while τ = 1.0 keeps spending — and what it spends the extra budget on is *high-confidence, mostly-correct* reads, which the boost corrupts.

### 1.2 p = 0.8 (the hardest level, the one the shipped code sweeps) — identity CONFIRMED 6/6

| seed | snapshot | N | `#{cos₀<0.99}` (from compute) | budget `8·step_n` | mean cos₀ | first-pass | **acc@k=8, τ = 0.99 / 0.999 / 0.9999 / 1.0** | compute@8 | identical? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | mid_stream | 200 | ≥160 | 160 | 0.71873 | 0.8150 | **0.8350 / 0.8350 / 0.8350 / 0.8350** | 1.8000 | ✅ |
| 0 | end_of_stream | 199 | ≥160 | 160 | 0.73470 | 0.8141 | **0.8643 / 0.8643 / 0.8643 / 0.8643** | 1.8040 | ✅ |
| 1 | mid_stream | 200 | ≥160 | 160 | 0.72862 | 0.8250 | **0.8500 / 0.8500 / 0.8500 / 0.8500** | 1.8000 | ✅ |
| 1 | end_of_stream | 200 | ≥160 | 160 | 0.73849 | 0.8500 | **0.8450 / 0.8450 / 0.8450 / 0.8450** | 1.8000 | ✅ |
| 2 | mid_stream | 200 | ≥160 | 160 | 0.73182 | 0.8450 | **0.8700 / 0.8700 / 0.8700 / 0.8700** | 1.8000 | ✅ |
| 2 | end_of_stream | 200 | ≥160 | 160 | 0.73345 | 0.8450 | **0.8800 / 0.8800 / 0.8800 / 0.8800** | 1.8000 | ✅ |

`max|Δacc| = 0.000e+00` and `max|Δcompute| = 0.000e+00` in **every** cell — not "within noise", **identical floats**. Between-seed spread of the end-of-stream k = 8 accuracy: mean 0.8631, **sd 0.0143, range 0.0350** (T4 band was ≤ 0.10).

**⇒ `cl-entry-build`'s single-seed observation is REPLICATED, not an artefact.**

### 1.3 p = 0.5 (same stores, same code, τ-sweep enabled by a *config-only* change) — identity FALSE 6/6

The shipped code sweeps τ only on the **last** level (`if li == len(cfg.retry_mask_levels) - 1`, `exp_cl_entry.py` L645). Setting `retry_mask_levels: [0.5]` in the project `config.yaml` makes p = 0.5 the last level. **A4 cross-check passed: the p = 0.5 cell is bit-identical between the two arms at all 3 seeds** (same `first_pass_acc`, `mean_confidence`, `n_items`, `knn_phi_floor` and gated ladder), so the two arms are directly comparable.

| seed | snapshot | N | `#{cos₀<0.99}` **exact** | mean cos₀ | first-pass | acc@8 τ=0.99 | τ=0.999 | τ=0.9999 | **τ=1.0** | compute@8 (0.99 → 1.0) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | mid_stream | 200 | **47** | 0.97281 | 1.0000 | 0.9900 | 0.9850 | 0.9700 | **0.5600** | 1.235 → 1.800 |
| 0 | end_of_stream | 199 | **24** | 0.98137 | 0.9899 | 0.9950 | 0.9849 | 0.9598 | **0.4171** | 1.121 → 1.804 |
| 1 | mid_stream | 200 | **52** | 0.96918 | 0.9850 | 0.9800 | 0.9750 | 0.9700 | **0.5450** | 1.260 → 1.800 |
| 1 | end_of_stream | 200 | **28** | 0.98154 | 0.9950 | 0.9900 | 0.9850 | 0.9650 | **0.4600** | 1.140 → 1.800 |
| 2 | mid_stream | 200 | **60** | 0.96615 | 0.9900 | 0.9900 | 0.9900 | 0.9700 | **0.5850** | 1.300 → 1.800 |
| 2 | end_of_stream | 200 | **31** | 0.97931 | 0.9950 | 0.9950 | 0.9950 | 0.9800 | **0.4650** | 1.155 → 1.800 |

**`acc(τ=1.0) − acc(τ=0.99)` at k = 8: mean −0.4846, sd 0.0640, range [−0.578, −0.405] over 6 cells; −0.5460 ± 0.0226 on the three end-of-stream cells.** Accuracy is **monotonically non-increasing in τ** in all 6 cells, and compute is monotonically non-decreasing.

**Figure:** `.claude/outputs/carried-remeasurements/fig_tau_regimes.png` — left panel (p = 0.5) shows four separated curves collapsing to 0.44 at 1.8×; right panel (p = 0.8) shows four curves **exactly overplotted**.

### 1.4 The regime statement (this is the answer the task asked for)

> **τ is binding exactly when the low-confidence pool is smaller than the retry budget.** Formally, the gate's threshold is load-bearing iff `#{cos₀ < τ} < k · step_n`. It is **not** a φ-space-vs-pixel-space distinction: both behaviours occur **inside φ-space**, in the *same store*, ~3 minutes apart, distinguished only by how hard the query is.
> - **w25's finding (`cl-entry-build` §Item-3) belongs to:** crowded store + **hard** corruption (p = 0.8), where 80 %+ of reads settle below cos 0.99 and the ranking + 10 %-per-round budget + lock do all the work.
> - **w24's finding (`exp_retry_compute`, τ = 1.0 over-retries and costs accuracy) belongs to:** any regime with a confident majority — which includes **φ-space at p = 0.5** (measured here, −0.485) and the `--quick` config (23-item store, mean cos₀ 0.999: τ = 0.99 never retries at all, 1.000× compute, acc 1.000; τ = 1.0 retries and drops to **0.826**; `.claude/scratch/carried-remeasurements/item1_smoke.log` — quick config, reported as *suggestive*, not as a scientific cell).
>
> **Neither of the two quoted clauses is wrong; the w25 one is stated too broadly.** The defensible sentence is: *"the threshold is what makes the ladder auto-stop; where the low-confidence tail is larger than the compute budget the threshold is inert and the ranking + budget + lock carry the curve, but where a confident majority exists the threshold is load-bearing and removing it (τ = 1.0) costs up to −0.58 accuracy."*

### 1.5 The laundering control (dial declaration), reported at every cell

The matched-compute **kNN-in-φ floor** (`feedforward_knn_phi`) at p = 0.8, k = 8: **0.8643 / 0.8550 / 0.8900** (end-of-stream, seeds 0/1/2, mean 0.8698) vs gated retry **0.8643 / 0.8450 / 0.8800** (mean 0.8631). **The gate does not beat the floor here (−0.007 ± 0.010) — it ties it, which is exactly the w25 "R3-native tie" and is unchanged by this remeasurement.** At p = 0.5 the floor is 0.9950 / 0.9950 / 1.0000 vs gated 0.9950 / 0.9900 / 0.9950 — also a tie. ⛔ Nothing here upgrades the retry claim; it only fixes *which* regime the threshold clause applies to.

### 1.6 PREREG scorecard — item 1

| # | registered | measured | verdict |
|---|---|---|---|
| T1 | four τ ladders identical at every k, all 3 seeds, p = 0.8 | identical in **6/6 cells**, `max|Δ| = 0` exactly | ✅ **confirmed** |
| T2 | fraction with `cos₀ < 0.99` ≥ 0.85 at p = 0.8; `mean_confidence` ≈ 0.90–0.97 | fraction **≥ 0.800/0.804** (a *lower bound* — the pool saturates the budget, so the exact count is not recoverable from the saved data); `mean_confidence` = **0.7187–0.7385** | ◐ **partial / sub-prediction falsified.** Direction right (pool ≫ budget); the `mean_confidence` band is wrong by ~0.2 (reads are *far* less confident than I predicted, which strengthens the mechanism); the ≥0.85 figure is **not verifiable** from what the harness saves |
| T3 | compute@8 = 1.800 ± 0.001 for all τ, all seeds | **1.8000 / 1.8040** exactly, all τ, all 6 cells | ✅ confirmed |
| T4 | between-seed spread of gated k = 8 acc ≤ 0.10 | **0.0350** (sd 0.0143) | ✅ confirmed |
| T5 | the identity is a **φ-space** property; pixel space differs | ❌ **FALSIFIED — and this is the finding.** Both behaviours occur inside φ-space; the discriminator is the confidence distribution, not the space | ❌ **falsified**, replaced by §1.4 |
| A1 | p = 0.5 τ-ladders NOT identical | identity **False in 6/6 cells** | ✅ confirmed |
| A2 | compute@8: 0.99 → ≈1.121, 1.0 → 1.804, monotone in τ | **1.1206 → 1.8040** (seed 0 end-of-stream); monotone non-decreasing in τ in 6/6 | ✅ confirmed to 4 dp |
| A3 | acc monotone non-increasing in τ; **acc(1.0) < 0.95** end-of-stream | monotone in 6/6; acc(1.0) = **0.4171 / 0.4600 / 0.4650** | ✅ confirmed, far more strongly than registered |
| A4 | the p = 0.5 cell is bit-identical across the two arms | identical in **6/6** (`first_pass`, `mean_conf`, `n_items`, `knn_floor`, full gated ladder) | ✅ exact |

**Score: 6 confirmed, 1 partial, 1 falsified (T5 — the falsification IS the result).**

---

## 2. Item 2 ⭐ — the occupancy sweep on the allocator trace

Nominal load = items offered / capacity 8. **Achieved** occupancy is lower at high load because the shipped admission gate refuses some offers — reported alongside.

### 2.1 The curve

| offers | nominal load | **achieved `n_live` IN / OUT** | post-evict retention | **`AUC(z_hole)` history** | TPR@5 % | **TPR@1 %** | **paired-placement control** |
|---|---|---|---|---|---|---|---|
| 2 (1 bg) | 0.250 | 2.000 / 1.000 | 0.0000 | **0.67148 ± 0.04050** | 0.092 | 0.029 | **0.5000 ± 0.0000** |
| 4 (3 bg) | 0.500 | 4.000 / 3.000 | 0.0000 | **0.91646 ± 0.02649** | 0.361 | 0.118 | **0.5000 ± 0.0000** |
| 6 (5 bg) | 0.750 | 5.922 / 5.000 | 0.0000 | **0.99611 ± 0.00404** | 0.999 | 0.924 | **0.5000 ± 0.0000** |
| 8 (7 bg) | 1.000 | 6.654 / 6.534 | 0.0000 | **0.99985 ± 0.00070** | 1.000 | **1.000** | **0.5000 ± 0.0000** |

**Figure:** `.claude/outputs/carried-remeasurements/fig_occupancy.png` (3 panels: AUC vs load for all six statistics, TPR@1 % vs load, and the underlying `z_hole` distributions).

### 2.2 All six residual channels, at every load (direction-calibrated AUC, mean ± sd over 24 examples)

| offers | `s1` value | `s2` address | `s4` white-box addr depth | `s5` white-box full `V` | **`z_hole`** | `n_live` |
|---|---|---|---|---|---|---|
| 2 | 0.6513 ± 0.0388 | 0.5596 ± 0.0448 | 0.6220 ± 0.0323 | 0.6099 ± 0.0355 | **0.6715 ± 0.0405** | 0.5000 ± 0.0000 |
| 4 | 0.6482 ± 0.1119 | 0.6262 ± 0.0867 | 0.7659 ± 0.0409 | 0.7398 ± 0.0481 | **0.9165 ± 0.0265** | 0.5000 ± 0.0000 |
| 6 | 0.6350 ± 0.1025 | 0.7186 ± 0.1317 | 0.7811 ± 0.0533 | 0.7409 ± 0.0667 | **0.9961 ± 0.0040** | 0.5389 ± 0.0205 |
| 8 | 0.6015 ± 0.0915 | 0.7828 ± 0.1435 | 0.7599 ± 0.0644 | 0.7106 ± 0.0796 | **0.99985 ± 0.00070** | 0.8114 ± 0.0349 |

`TPR@FPR 1 %` for `s4`/`s5` jumps **0.023 → 0.098 → 0.919/0.932 → 1.000** across the same loads: the *entire* residual-leak surface, not just `z_hole`, is occupancy-scaled.

### 2.3 Mechanism — the exclusion disk, measured

`z_hole` = distance from `c_i` to the nearest **live** site, after the target is evicted.

| offers | IN (target written, then evicted) | OUT (never written) | `d_safe` |
|---|---|---|---|
| 2 | 2.4578 ± 0.5504 (min **1.5403**) | 1.9575 ± 0.8479 (min 0.0266) | 1.540 |
| 4 | 1.9065 ± 0.2903 (min **1.5400**) | 1.0654 ± 0.5107 (min 0.0266) | 1.540 |
| 6 | 1.7025 ± 0.1408 (min **1.5400**) | 0.7925 ± 0.3157 (min 0.0197) | 1.540 |
| 8 | 1.6636 ± 0.1192 (min **1.5400**) | 0.7128 ± 0.2589 (min 0.0197) | 1.540 |

**The IN minimum is `d_safe` to 4 decimal places at every load** — the write reserves a hard 1.54 exclusion disk and eviction does not give it back. What changes with load is the *OUT* distribution: with one background item it is often outside 1.54 anyway (overlap ⇒ AUC 0.67); with seven, mutual exclusion packs the sites into the remaining free area and the OUT mean falls to 0.71, i.e. **the attacker's separation comes from the background getting closer, not from the hole getting deeper.**

### 2.4 Replication of the w25 cell (O3) — element-for-element

The 8-offer cell is the exact `mia-decay-measurement` configuration. Comparing `occupancy_metrics.json` against `mia_metrics.json` per-example arrays:

| statistic | w25 (`mia_metrics.json`) | w26 (`occupancy_metrics.json`) | arrays identical? |
|---|---|---|---|
| `s1` | 0.601489 ± 0.091525, TPR@1 % 0.1540 | 0.601489 ± 0.091525, 0.1540 | **True** |
| `s2` | 0.782772 ± 0.143513, 0.3766 | 0.782772 ± 0.143513, 0.3766 | **True** |
| `s4` | 0.759902 ± 0.064424, 1.0000 | idem | **True** |
| `s5` | 0.710636 ± 0.079628, 1.0000 | idem | **True** |
| **`z_hole`** | **0.999854 ± 0.000701, 1.0000** | idem | **True** |
| `n_live` | 0.811375 ± 0.034892, 0.1576 | idem | **True** |

⚠ **This is a determinism/faithfulness check, not an independent replication** — same code path, same seeds, same PRNG. Its value is that it proves my derived harness is the w25 harness, so the other three loads are comparable to the published number.

### 2.5 PREREG scorecard — item 2

| # | registered | measured | verdict |
|---|---|---|---|
| O1 | `AUC(z_hole)` strictly monotone increasing in load | 0.6715 → 0.9165 → 0.9961 → 0.99985 | ✅ confirmed |
| O2 | 0.68 ± 0.10 / 0.87 ± 0.08 / 0.95 ± 0.05 / [0.995, 1.000] | **0.6715 / 0.9165 / 0.9961 / 0.99985** | ✅ **all four inside band** (deltas −0.009 / +0.047 / +0.046 / +0.020 vs the model) |
| O3 | 8/8 reproduces 0.99985 ± 0.00070 to ±0.002 | reproduced **exactly**, arrays element-identical on all 6 statistics | ✅ exact |
| O4 | leak does not vanish at low load: `AUC(2/8) ≥ 0.60` | **0.6715** | ✅ confirmed |
| O5 | the falsifier `AUC(2/8) ≥ 0.99` does NOT fire | 0.6715 | ✅ does not fire |
| O6 | TPR@1 %: 1.000 at 8/8, < 0.50 at 2/8 | **1.000** and **0.029** | ✅ confirmed |
| O7 | paired column = 0.5000 ± 0.0000 exactly, all statistics, all loads | **0.5000 ± 0.0000** in 24 of 24 (6 statistics × 4 loads) | ✅ exact — **no harness bug** |
| O8 | `AUC(n_live)` 0.500 ± 0.02 at 2/8 → ≈0.81 ± 0.05 at 8/8 | **0.5000 ± 0.0000** at 2 *and* 4 offers (fully tied — admission refuses nothing), 0.5389 at 6, **0.8114** at 8 | ✅ confirmed |
| O9 | post-evict retention = 0 at every load | **0.0000 ± 0.0000**, all 4 loads | ✅ exact |
| O10 | `evict` leaves the row verbatim (D1) | `centers` max err **5.622e−8**, `payloads` **0.0**, `amps` 0.0, `active` 0.0 — at every load | ✅ confirmed, D1 unchanged |
| model | `AUC ≈ 1 − ½(1−p)^n`, `p ≈ 0.36`, a lower bound loosening with load | model 0.680 / 0.869 / 0.946 / 0.978; **implied `p` = 0.343 / 0.449 / 0.621 / 0.687** | ✅ the registered structure holds: `p` at n = 1 is 0.343 vs the derived 0.36 (−5 %), and rises with load exactly as the "mutual exclusion compresses OUT" correction predicted |

**Score: 11 confirmed (10 registered + the model), 0 partial, 0 falsified.**

### 2.6 What this means for `placement-landing`

- Its acceptance test (**drive post-evict `AUC(z_hole)` to 0.5**) is **load-dependent** and must be stated at a load. At 8 offers the target is 0.99985 → 0.5; at 2 offers it is only 0.6715 → 0.5. **A placement rule evaluated at low load can look ~2× closer to done than it is.** Recommendation: the acceptance test is *the full four-load curve*, with 8/8 as the headline cell.
- The mechanism the fix must kill is explicit: **the IN-world minimum `z_hole` is exactly `d_safe` (1.5400) at every load.** Any order-independent placement must destroy that hard floor, not merely blur it — a rule that merely jitters positions leaves `min z_hole = d_safe` and the LiRA attack at TPR 1.000 @ FPR 1 % survives.
- The paired-placement column being **0.5000 ± 0.0000 everywhere** confirms once more that the amplitude/value channel is not the problem: **placement is the entire post-evict leak.**

---

## 3. Cross-item note (unsolicited but load-bearing)

Items 1 and 2 have the same shape, and it is the shape the program keeps finding: **a designed knob is inert in the regime where a cheaper mechanism saturates, and load-bearing in the regime where it does not.** τ is inert when the low-confidence pool exceeds the budget and decisive when it does not; the allocator trace is a near-perfect oracle when the store is full and a weak signal when it is not. Both are *scoping* results, and in both cases the previously-quoted number was the extreme of a curve. Recommend the Hub adopt "quote the curve, not the endpoint" as a standing rule for these two claims.

---

## 4. Item 3 — the d=8 K=256 8×32 sharding probe: **NOT COMPLETED**

**Status: launched as a cost probe, then DELIBERATELY KILLED for machine-load reasons before any arm completed. Reported as NOT RUN — explicitly NOT as a null, per the task file and PREREG §Item-3. No P4 number exists and none is quoted.**

- **Command actually issued** (single seed, two arms, discriminator item only, to bound the cost):
  `PYTHONPATH=. .venv/bin/python -m chlu exp-sharded-store --project w26p4 --cells 8:256:8 --arms monolithic sharded_matched --items 1 --seed 0`
  Log: `.claude/scratch/carried-remeasurements/item3_probe.log`.
- **Observed:** started 10:36:01, killed 10:43. In **7.5 minutes it printed `[item 1] the 2x2 discriminator` / `[cell] 8:256:8` and nothing further** — the *first* arm (monolithic, K = 256) had not completed a single write. `projects/w26p4/results/` is **empty**; no metrics file was produced.
- **Why I killed it:** it spawned 4 worker processes and the machine's **1-minute load average went from ~15 to 139.76 on 8 cores** with four `experiment-engineer` worktrees running concurrently. Per the task file's compute note ("items 1 and 2 are cheap and are the priority… item 3 is explicitly droppable"), continuing would have degraded four other agents' runs for a probe whose result is a *pre-registered expected FAIL* and which is **budget-confounded anyway** (next bullet). Kill was `pkill -f exp-sharded-store`; exit code 144. **Nothing else on the machine was touched.**
- **Why this was foreseeable and is not a surprise.** `lattice-sharded-store` §8.3 measured ~700 s of *write* per cell at d = 6 K = 64 and ~2× at d = 8, per seed, on 8 cores with each worker saturating ≈1 core. The P4 cell is **K = 256**, i.e. ~4× the items of the d = 8 K = 64 cell for the monolithic laundering arm ⇒ an estimated **~1.5 h per seed for the monolithic arm alone**, before the sharded arm, before the 3 seeds and before the blank control. The machine was concurrently running four engineer worktrees (`uptime` load average peaked at **30.9** on 8 cores during this session), so the effective throughput was well below the §8.3 baseline.
- **The budget confound stands and must be quoted with any future P4 result.** N107: the w23 atom floor `512·√2^d` is **per-store, not per-item**, so a parameter-matched 8-way split at d = 8 is **~8× starved** by construction — exactly the confound that made w25 report the 4×16 collapse (0.6559) as *confounded rather than clean*. **A parameter-matched P4 cell cannot discriminate starvation from geometry**; the informative version needs `sharded_per_shard` (8 × 8192 = 65 536 atoms), which §8.3 already declared unaffordable at 2 shards.
- ⛔ Routing discipline observed: only **pre-settle energy (R2)** and the **classical registry (RG, declared classical indexing per N89)** would have been read; post-settle energy (N97) and settling displacement (N104-proposed) are broken and were not used.

**Recommendation to the Hub:** do **not** re-attempt P4 as a parameter-matched cell. Either (a) fund the budget-adequate `sharded_per_shard` version on real hardware (it is the only version that answers the question), or (b) leave P4 registered-and-unrun, which — as `lattice-sharded-store` correctly said — removes a confirmation, not a claim.

---

## How I verified

- `--quick` smoke first on **both** harnesses: `exp-cl-entry --quick` (30 s, exit 0) and `occupancy_harness.py --quick` (3 s, exit 0), before any full run. No NaN, no divergence, no OOM anywhere; every run exited 0.
- **Every number in §1 and §2 is re-derived from a saved metrics JSON**, never transcribed from stdout: §1 from `item1_tau_metrics.json` (itself built from the six `exp_cl_entry_mnist_metrics.json` run files) via `consolidate_item1.py`; §2 from `occupancy_metrics.json` via `analyze_item2.py`. Both analysis outputs are saved (`item1_summary.txt`, `item1_analysis_hard.txt`, `item2_analysis.txt`).
- **Independent internal checks that had to come out exactly right, and did:** (i) the p = 0.5 cell is bit-identical between the `[0.5, 0.8]` and `[0.5]` arms at all 3 seeds; (ii) the 8-offer occupancy cell reproduces `mia_metrics.json` element-for-element on six statistics; (iii) the paired-placement control is 0.5000 ± 0.0000 in 24 of 24 (statistic × load) combinations; (iv) the compute-multiplier arithmetic `1 + retried/N` recovers integer retry counts exactly (24, 26, 28, 31, 47, 51, 52, 60, 160 …), which is the consistency check on the `#{cos₀ < τ}` derivation.
- The τ-identity is asserted with `==` on floats, not a tolerance: `max|Δacc| = 0.000e+00`.

## Git footprint
**None.** `git status --short` was **empty before and after** (verified twice; base `ff85573`, no branch created, no commit, no tracked file created/modified/deleted). All artefacts live under `.claude/outputs/carried-remeasurements/`, `.claude/scratch/carried-remeasurements/` and `projects/w26tau*`, `projects/w26tauE*`, `projects/w26p4` — all gitignored. **`.claude/outputs/mia-decay-measurement/mia_harness.py` was NOT edited** (`placement-landing` depends on it this wave); I worked on a copy and kept a verbatim `mia_harness_SOURCE_COPY.py` next to it.

## Open questions / follow-ups / risks

1. **`#{cos₀ < 0.99}` at p = 0.8 is only a lower bound (≥160/200).** The harness does not save the first-pass cosine distribution, so the mechanism at the hard level is confirmed *structurally* (the pool saturates the budget) but the exact fraction is unrecoverable. **One-line fix for `experiment-engineer`: record `float(np.mean(cos0 < cfg.retry_tau))` (or a small histogram of `cos0`) in the retry cell.** That would make the τ-binding condition directly readable instead of inferred.
2. **The τ-sweep only ever runs on the last corruption level.** That is why this contradiction survived a wave. Either sweep every level or record the eligible-pool fraction per level (item 1 above). Cheap either way.
3. **All 12 τ cells are one dataset (Split-MNIST), one φ (task1_only PCA-32), one store (200 items).** The mechanism is arithmetic and should transfer, but the *location* of the crossover (which corruption level flips the regime) is not measured — it lies between p = 0.5 and p = 0.8 and nobody has swept it.
4. **The occupancy sweep varies the number of background *offers*, holding the proposal disk fixed.** That is the right control for "does the hole statistic depend on how full the store is", but it conflates *count* with *density*: an alternative sweep (fixed count, varying disk radius) would separate them. Not done.
5. **Achieved occupancy saturates at 6.65/8**, not 8/8, because admission refuses offers — so "8/8" in this report means *8 offers*, not 8 live items. The genuinely-full case (raise `budget`, force 8 live) is untested and would, if anything, push `AUC(z_hole)` even closer to 1.
6. **Item 3 remains open** and is now, in my judgement, not worth funding in its parameter-matched form (§4).

---

## Proposed handover updates (for the Hub)

1. **§1.6 / R3 — NARROW the w25 τ line and RESTORE w24's, with numbers.** Replace *"τ is not the binding element of the gate in φ-space"* with:
   > *"Whether the retry threshold τ is load-bearing is decided by one measurable quantity: the eligible pool `#{cos₀ < τ}` against the compute budget `k · step_n`. In the crowded φ store at the hardest corruption (p = 0.8) the pool saturates the budget (≥160 of 200) and **all four thresholds {0.99, 0.999, 0.9999, 1.0} give bit-identical ladders — `max|Δ| = 0` at 3/3 seeds × 2 store snapshots**; the curve is produced by the ranking + the 10 %-per-round budget + the lock. At p = 0.5 in the same store the pool is only 24–60 of 200 and **τ is decisive: τ = 1.0 costs −0.485 ± 0.064 accuracy (−0.546 ± 0.023 end-of-stream, e.g. 0.9950 → 0.4171) while spending 1.80× instead of 1.12–1.30×.** w24 and w25 measured the two ends of the same axis."*
   **This retracts nothing and un-contradicts the two quoted places.** ⚠ Do **not** keep the "φ-space vs pixel-space" framing — it was my own pre-registered hypothesis and it is **falsified** (both behaviours occur inside φ-space).
2. **§8 / negative registry — candidate new N (recommend registering, tier B).** *"The retry gate's threshold is inert exactly when the low-confidence pool exceeds the compute budget. It is not a property of the address space: in Split-MNIST φ-space, p = 0.8 gives a bit-identical τ-sweep (pool ≥ 80 % of N) and p = 0.5 gives a −0.485 accuracy penalty at τ = 1.0 (pool 12–30 % of N). Any statement about the threshold must name the corruption level."* (Pre-registered as A1–A3 before measurement; T5, the competing hypothesis, was pre-registered and rejected.)
3. **§1.6 / N99 upgrade — the allocator-trace number becomes a CURVE.** *"Post-eviction membership from the allocator trace alone (`z_hole` = distance to nearest live site) is **occupancy-scaled**: AUC **0.6715 ± 0.0405 / 0.9165 ± 0.0265 / 0.9961 ± 0.0040 / 0.99985 ± 0.00070** and TPR@FPR 1 % **0.029 / 0.118 / 0.924 / 1.000** at 2 / 4 / 6 / 8 offers into a capacity-8 store (24 per-example values, 128 paired worlds, seeds 0/1/2). The mechanism is a hard `d_safe = 1.5400` exclusion disk that survives eviction — **the IN-world minimum `z_hole` equals `d_safe` to 4 dp at every load**; what changes is how close the background packs in the OUT world (mean 1.96 → 0.71). The paired-placement control is **0.5000 ± 0.0000** at every load, on every statistic."* ⚠ The bare "0.99985" must never again be quoted without the load.
4. **`placement-landing`'s acceptance test needs a load qualifier (act on this at review).** The target "post-evict `AUC(z_hole)` → 0.5" is **~2× easier at 2 offers than at 8**. Recommend the acceptance test be *the four-load curve*, headline at 8 offers, and that the fix be judged on whether it destroys `min z_hole = d_safe`, not on whether it lowers the mean. This harness is the test: `.claude/scratch/carried-remeasurements/occupancy_harness.py`.
5. **§7-CURRENT — two code issues for `experiment-engineer` (neither moves a published number).**
   (E1) `exp_cl_entry.run_experiment_cl_entry` nests the whole retry block inside `if "entry" in items` (L1053–1073), so **`--items retry` silently produces an empty `retry_native`** — the task file's own suggested command. Either de-nest or document.
   (E2) The retry cell saves `mean_confidence` but not `#{cos₀ < τ}`; because the τ-sweep also only runs on the last corruption level, **the quantity that decides the whole τ question is not recorded**. Add `frac_cos_below_tau` (one line) and/or sweep τ at every level.
6. **§5 provenance — new artefact set.** `.claude/outputs/carried-remeasurements/`: `PREREG.md` (+ ADDENDUM 1), `occupancy_metrics.json`, `item1_tau_metrics.json`, `item2_analysis.txt`, `item1_summary.txt`, `item1_analysis_hard.txt`, `fig_occupancy.png`, `fig_tau_regimes.png`. Harnesses/scripts under `.claude/scratch/carried-remeasurements/` (`occupancy_harness.py`, `mia_harness_SOURCE_COPY.py`, `analyze_item2.py`, `analyze_item1.py`, `consolidate_item1.py`, `fig_item2.py`, run logs). Runs under `projects/w26tau{0,1,2}`, `projects/w26tauE{0,1,2}`, `projects/w26p4`. Base `ff85573`, JAX 0.9.0, seeds 0/1/2, **no tracked code touched**.
7. **§1.6 / item 3 — record P4 as still NOT RUN, and recommend closing it.** A parameter-matched d = 8 K = 256 8×32 cell is **~8× atom-starved by construction** (N107: the `512·√2^d` floor is per-store), so it cannot discriminate starvation from geometry; the budget-adequate `sharded_per_shard` version was already declared unaffordable at 2 shards in w25. Recommend either funding it on real hardware or leaving it registered-and-unrun. It removes a confirmation, not a claim.
8. **Standing wording rule (recommend adopting).** Both carried items turned out to be **endpoints of curves quoted as constants**. Suggest a program-level rule: *any headline number that depends on a load/difficulty/crowding parameter is quoted with that parameter, or as the curve.*
