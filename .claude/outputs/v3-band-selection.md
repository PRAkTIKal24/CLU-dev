# v3-band-selection — experiment-engineer report
Task + acceptance criterion: answer V3.2's "we told the model the answer" attack end-to-end — (1) mis-banded confound-killer, (2) a from-data band-selection recipe, (3) characterize the mass-lr inducer's sweet-spot (CM-5). Accept iff confound killed + selection story exists + inducer characterized, each with flag-provenance.

Status: **done** — all items complete (item 1: 40 runs; item 2: 20 runs; item 3: 48 cells at 1500 ep; item 4 written up). Acceptance met: confound killed + selection recipe exists + inducer characterized, all with flag-provenance.

---

## Flag-provenance (applies to ALL runs unless noted)
| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/v3-band-selection` off `main` @ `9a13455` (worktree `../CHLU-v3-band-selection`) |
| code commit | `c4bc004` (band_selection module + tests) — additive only, **zero existing code touched** (`git diff --stat main..` = 2 new files, +251) |
| testbed | N-unit `CLULattice`, `kinetic_mode=newtonian_learned`, hidden=32, spring κ=`train_kappa_c`=0.01, dt=0.05, lr=1e-3, window=64 — identical to seed-sweeps item 1 / mass-lr-doctrine-test |
| data | `generate_two_timescale_orbits`, shared stiffness k=1 ⇒ ω_i=1/√M_i, geomean-1 masses spanning `ratio`, radius=1, n_traj=64, seq_len=256 (N=2)/512 (N=4) |
| init | **uniform** log_mass except where an arm designs a band (matched/anti/orthogonal/selector); all banded arms use exact softplus-space scaling |
| non-default flags | `training.mass_lr_mult ∈ {1,3,10,30,100}` (item 2/3); everything else `get_default_config()` defaults (langevin_noise=legacy, lyapunov=max, persistent_sleep_buffer=False, sleep_freq=5, sleep_steps=500, sleep_temperature=0.5) |
| seeds | item 1/2: {0,1,2,3,4}; item 3: {0,1,2} |
| metrics | `eval_mse` = held-out 255-step rollout MSE (8 traj); `align` = Spearman(per-unit-mean learned M, true band); `log_spread` = std(log M-spectrum) |
| ⚠ env caveat | worktree venv freshly `uv sync`'d to **JAX 0.10.2** (main venv = 0.9.0). One pre-existing lattice test (`test_kappa_zero_reduction_bitlevel`) fails ONLY under 0.10.2 (bit-level reduction change) — unrelated to my code (see §How-verified). My headline numbers are **version-stable**: item-1 matched@300 reproduces seed-sweeps bit-for-bit. |

## Code deliverable (item 2's "method") — `chlu/data/band_selection.py` (+ `tests/test_band_selection.py`, 6 tests)
- `estimate_unit_frequencies(data, n_units, dt, method={fft,autocorr})` — per-unit dominant angular frequency ω_i from a trajectory-averaged periodogram (or autocorr first-zero-crossing) of that unit's position channels.
- `select_mass_bands(...)` — maps ω_i → **M_i ∝ 1/ω_i²** (shared-stiffness, F5 §5.3), normalized to unit geomean (F5 gauge). Drop-in for `banded_mass_scales`.
- `mismatch_band(matched, kind={anti,shuffle_common})` — same-log-spread mis-banded controls.
- Verified: recovers the oracle band **[4.0, 0.25]** from default data (both FFT & autocorr), and a monotone 16× band at N=4 (selected [3.87,1.72,0.62,0.24] vs true [4.0,1.59,0.63,0.25]).

---

## Item 1 — mis-banded control (THE CONFOUND-KILLER) ✅
4 arms, all spending the **identical log-scale-spread budget** (multiset {heavy,light}), differing only in the assignment axis:
- **matched** = oracle band on the between-unit (data) axis; **anti** = inverted ordering; **orthogonal** = same spread spent on the within-unit channel axis (data units are isotropic ⇒ that axis carries no signal); **uniform** = no band (reference).

5 seeds. eval-MSE mean ± std:

| budget | matched | uniform | orthogonal | anti |
|---|---|---|---|---|
| 60 ep | **2.812 ± 0.327** | 3.751 ± 0.336 | 8.874 ± 2.198 | 14.905 ± 0.686 |
| 300 ep | **1.180 ± 0.216** | 2.416 ± 0.345 | 6.924 ± 1.883 | 12.791 ± 0.698 |

matched beats every other arm **5/5 seeds at both budgets**. MSE ratios (300 ep): matched/uniform = **0.488**, matched/orthogonal = **0.170**, matched/anti = **0.092**.

**Degradation curve (300 ep): matched (1.18) < uniform (2.42) < orthogonal (6.92) < anti (12.79).**

**Verdict (quotable):** *A correct inertial-mass prior helps (0.49× the uniform MSE, 5/5 seeds); a wrong prior of the same magnitude COSTS — an inverted (anti) band is 5.3× worse than uniform and 10.8× worse than matched; an orthogonally-placed band of identical spread is 2.9× worse than uniform. The banding win is therefore not "any structure helps" and not "we told the model the answer for free" — it is a correct-prior effect with a measured, steep price for guessing wrong.* The confound V3.2 raised is killed: matched@300 (1.180±0.216) and uniform (2.416±0.345) reproduce seed-sweeps bit-for-bit, and the two new mis-banded arms bracket the cost of a wrong guess.

---

## Item 2 — a band-selection recipe (THE METHOD) ✅
5 seeds, 2-unit, 16× ratio, 300 ep. Arms: **uniform** (floor), **oracle** (matched designed band), **selector** (spectral FFT → M∝1/ω²), **masslr_init** (induce ordering with mass_lr_mult=10 for 150 ep → snap masses to oracle magnitudes in the induced order → freeze mass → retrain).

| arm | eval-MSE (mean ± std) | gap vs oracle |
|---|---|---|
| uniform | 2.416 ± 0.345 | +1.236 |
| **oracle** | **1.180 ± 0.216** | 0 |
| **selector** | **1.180 ± 0.216** | **+0.000** |
| masslr_init | 3.527 ± 4.493 | +2.347 |

- **The spectral selector recovers the oracle band [4.0, 0.25] on 5/5 seeds → selector-oracle MSE gap = 0.000 ± 0.000.** The FFT timescale estimate is exact enough here that the data-driven band IS the oracle band, at negligible cost (one FFT over the training set).
- **masslr_init is fragile:** induced the correct ordering on **4/5 seeds** (those → MSE 1.09–1.52 ≈ oracle); seed 2 inverted the ordering → snapped to the anti-band → **MSE 12.5** (the full mis-banded penalty), driving the mean/variance up.

**Verdict / the one honest sentence for the V3 short:** *"Bands can be chosen from data by a per-unit spectral (FFT) timescale estimate at negligible cost, recovering the oracle band to FFT resolution (selector–oracle MSE gap 0.000, 5/5 seeds); a mass-lr induction-then-snap recipe is a plausible optimizer-native alternative but fragile — correct ordering on 4/5 seeds, and the single inversion pays the full mis-banded penalty."* **Caveat (state in the short):** the selector's zero gap relies on the task's clean spectral signature (well-separated per-unit ω); on data without cleanly separated timescales the FFT estimate degrades and the gap would open — the recipe is "cheap and exact WHEN timescales are spectrally separable."

---

## Item 3 — sweet-spot generalization (mass-lr follow-up 1) ✅
48 cells complete: mult∈{3,10,30,100} × N∈{2,4} × ratio∈{4×,16×}, 3 seeds, 1500 ep, **uniform init** (banding NOT designed in). `align`=Spearman(learned M ordering, true band), +1=correct; `spread`=std(log M).

| N | ratio | mult | align | log-spread | eval-MSE |
|---|---|---|---|---|---|
| 2 | 4× | 3 | +1.00±0.00 | 0.263 | 0.274 |
| 2 | 4× | **10** | **+1.00±0.00** | 0.553 | 0.033 |
| 2 | 4× | 30 | +1.00±0.00 | 0.767 | 0.022 |
| 2 | 4× | 100 | +1.00±0.00 | 1.171 | 0.373 |
| 2 | 16× | 3 | +1.00±0.00 | 0.386 | 1.372 |
| 2 | 16× | **10** | **+1.00±0.00** | 0.488 | 0.633 |
| 2 | 16× | 30 | +0.33±0.94 | 0.071 | 0.366 |
| 2 | 16× | 100 | **−1.00±0.00** | 0.176 | 0.332 |
| 4 | 4× | 3 | +1.00±0.00 | 0.222 | 0.149 |
| 4 | 4× | **10** | **+1.00±0.00** | 0.442 | 0.028 |
| 4 | 4× | 30 | +1.00±0.00 | 0.646 | 0.067 |
| 4 | 4× | 100 | +1.00±0.00 | 1.718 | **35.27±47.26** |
| 4 | 16× | 3 | +0.93±0.09 | 0.347 | 0.680 |
| 4 | 16× | **10** | **+0.80±0.00** | 0.611 | 0.296 |
| 4 | 16× | 30 | +0.20±0.28 | 0.678 | 0.197 |
| 4 | 16× | 100 | **−0.20±0.00** | 1.459 | 0.800 |

**Verdict (quotable) — the CM-5 sweet-spot GENERALIZES, and the "100× inverts" clause is sharpened to be ratio-dependent:**
- **mult≈10 is a reliable inducer of the correct ordering across the whole grid** — align **+0.80…+1.00 (3 seeds)** at N∈{2,4} × ratio∈{4×,16×}, with a healthy log-spread (0.44–0.61). This confirms CM-5's "≈10× induces ordering" as a **usable default**, not a single-cell artifact.
- **High mult (≥30) is harmful, but the failure mode depends on the data-mass ratio.** At **16×** ratio it drives an **ordering inversion** (N=2 mult=100 → align −1.00; N=4 mult=100 → −0.20; onset already at mult=30). At **4×** ratio the ordering does **not** invert even at 100× (align stays +1.00); instead over-lr triggers a **global-mass runaway that explodes the fit** (N=4/4×/mult=100 → eval-MSE **35.3 ± 47.3**). So "≈100× inverts" holds *for large ratio*; for small ratio the same over-lr instead blows up the magnitude without flipping the order. Either way mult≈10 is the safe operating point.
- **The MSE-as-fit trap (CM-5 follow-up 4) reproduces:** at N=2/16×, mult=100 has the *lowest* MSE (0.332) yet align=**−1.00** (inverted) — a global mass rescale fits better independent of hierarchy. **Read align/spread, never MSE, as the hierarchy signal.**
- **N=4 induces slightly less cleanly than N=2 at high ratio** (align +0.80 vs +1.00 at mult=10, 16×): bigger lattices need the inducer more but the correct ordering is a touch noisier.

---

## Item 4 — negatives (charter C-9, appendix material)
- **Orthogonal/anti mis-banding costs** (item 1): a wrong-axis or inverted prior is *worse than no prior* — appendix "price of guessing" panel.
- **masslr_init inversion failure** (item 2, seed 2): the induce-then-snap recipe inherits the induction's ordering errors catastrophically (snapping to a wrong ordering = designing-in the anti-band). Documents *why the spectral selector is preferred over the optimizer-native one*.
- **Over-lr failure modes (item 3):** naive mult≥30 is a documented failure — at large data-mass ratio it *inverts* the induced ordering (align −1.00), at small ratio it *runs the global mass away* and explodes the fit (MSE 35×). Both are appendix "why ≈10× not 100×" evidence; neither is a bug.

---

## How I verified (commands + real output)
- Selector/oracle recovery, arm construction, log-spread equality across arms — printed numerically (see §Item 1/2 tables; scratch `common.py`, `run_item{1,2,3}.py`, `analyze.py`).
- Item 1 matched@300 = **1.1801 ± 0.2162** and uniform = **2.4159 ± 0.3452** — **bit-for-bit** equal to seed-sweeps `b1782b0` (banded 1.180±0.216 / uniform 2.416±0.345), confirming version-independence of the numerics despite the worktree's JAX 0.10.2.
- Tests: `pytest tests/test_band_selection.py` → **6 passed**. Broader run `test_band_selection + test_data + test_lattice + test_mass_lr` → **24 passed, 1 failed**; the 1 failure = `test_lattice.py::test_kappa_zero_reduction_bitlevel`, which **passes on main's venv (JAX 0.9.0)** and **fails only in the worktree venv (JAX 0.10.2)** — a JAX bit-level reduction change, NOT my code (`git diff main..` = 2 new files, no existing-code edits). Flagged for the Hub; not a regression from this branch.
- Lint: `ruff check` clean on both new files; `ruff format` applied.

## Git footprint
- Branch `agent/experiment-engineer/v3-band-selection` (worktree, §3.2 mandatory — concurrent `v1-router-baseline`), base `main` @ `9a13455`.
- Commit `c4bc004`: `chlu/data/band_selection.py` (+165), `tests/test_band_selection.py` (+86). No existing code modified.
- All experiment scripts (`common.py`, `run_item{1,2,3}.py`, `analyze.py`) + per-cell results (`item{1,2,3}_results.jsonl`) under `.claude/scratch/v3-band-selection/` (untracked). Not pushed, no PR.
- **Branch ref verified from the MAIN repo** (`git log --oneline main..agent/experiment-engineer/v3-band-selection` shows `c4bc004`, §3.2 anti-loss check). Worktree `../CHLU-v3-band-selection` **left in place** for Hub review (clean tree); safe to `git worktree remove` — the commit is on the shared ref. Base `main` @ `9a13455` unmoved ⇒ no rebase needed.
- Experiment runs used the worktree venv (JAX 0.10.2); code is version-stable (item-1 reproduces seed-sweeps bit-for-bit).

## Open questions / follow-ups / risks
1. **Worktree JAX-version drift (ops):** fresh `uv sync` in a worktree pulled JAX 0.10.2 vs main's 0.9.0 → one bit-level lattice test flips. Recommend the Hub pin JAX in `uv.lock` (or spokes `uv sync` from a shared lock) so worktree venvs match main. My results are unaffected (bit-reproduce seed-sweeps).
2. **Selector generality:** zero oracle-gap is task-specific (clean spectral separation). A follow-up on overlapping/noisy timescales would size the gap honestly.
3. **masslr_init** is dominated by the spectral selector here; keep it as the "optimizer-native but fragile" foil, not the recommended recipe.

## Proposed handover updates (for the Hub)
- **CM-5 / V3.2 (P10) — ANSWERED end-to-end.** (1) Confound killed: mis-banded controls give the degradation curve **matched (1.18) < uniform (2.42) < orthogonal (6.92) < anti (12.79)** @300ep, 5/5 seeds — a correct prior is 0.49× uniform, a wrong prior 2.9–5.3× worse than uniform. (2) Selection recipe exists: a spectral FFT selector (`chlu/data/band_selection.py`) **recovers the oracle band with 0.000 MSE gap, 5/5 seeds**; the mass-lr induce-then-snap alternative is fragile (4/5 correct, 1 inversion → full penalty). (3) **Inducer characterized:** mult≈10 reliably induces correct ordering across N∈{2,4}×ratio∈{4×,16×} (align +0.80…+1.00, 3 seeds) — CM-5 sweet-spot generalizes as a usable default; high mult (≥30) harmful with a **ratio-dependent** failure mode — *inverts* ordering at 16×, *runs mass away / explodes MSE* at 4×. **Suggested CM-5 addendum:** "the ~100× inversion is specific to large data-mass ratio; at small ratio over-lr instead causes a global-mass runaway. mult≈10 is the safe default across N and ratio."
- **§3 config / code:** new module `chlu.data.band_selection` (spectral band selector + mismatch controls), 6 tests. No config schema change.
- **§7 ops:** log the worktree JAX 0.9.0→0.10.2 drift + the one version-sensitive lattice bit-test.
