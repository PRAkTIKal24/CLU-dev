# mass-lr-doctrine-test — experiment-engineer report

Task + acceptance criterion: give `log_mass` its own Adam lr (new flag `training.mass_lr_mult`, default 1.0 = bit-compatible) and test on the banded-lattice testbed (uniform-init M) whether a mass-specific lr lets the optimizer discover the mass hierarchy the "designed-in or induced" doctrine (Hyp-3) says it cannot. Accept iff a quotable verdict with error bars: *"hierarchy IS learnable given a mass-specific lr"* or *"doctrine fortified — even at 100× lr / 5× epochs / curriculum, M stays uniform"*, each number carrying a flag-provenance table.

Status: **done.** All 50 cells complete (sweep 30 + banded 10 + curriculum 10, 5 seeds each). Quotable verdict with error bars below.

---

## Flag-provenance (applies to ALL runs below unless noted)
| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/mass-lr-doctrine-test` off `main` @ `db3369b` |
| feature commits | `be12b73` (flag + train wiring), `df11483` (tests) |
| testbed | 2-unit `CLULattice`, `kinetic_mode=newtonian_learned`, hidden=32, spring κ=0.01, dt=0.05, lr=1e-3 (base), ~2382 params — same as seed-sweeps item 1 |
| data | `generate_two_timescale_orbits`, ω=[0.5,2.0], data masses **[4.0, 0.25]** (unit0 slow/heavy, unit1 fast/light), radius=1, n_traj=64, seq_len=256, window=64 |
| init | **uniform** log_mass (banding NOT designed in) except the `banded` reference arm |
| non-default flags | `training.mass_lr_mult ∈ {1,10,100}`; everything else at `get_default_config()` defaults (langevin_noise=legacy, lyapunov_penalty=max, persistent_sleep_buffer=False, sleep_freq=5, sleep_steps=500, sleep_temperature=0.5, sleep_friction=0) |
| seeds | 0,1,2,3,4 (each seed fixes data/init/train keys; only mass_lr_mult & banded differ within a seed) |
| target band | learned M vs `[4,4,0.25,0.25]`; a discovered hierarchy → unit0 heavy, unit1 light |

## Code change (item 1) — minimal diff, train.py + config only
- **`config.py`:** new `TrainingConfig.mass_lr_mult: float = 1.0`. Runs every unit's `log_mass` on its own Adam slot at `learning_rate * mass_lr_mult`. Default 1.0 = bit-compatible.
- **`training/train.py`:** unified the optimizer builder. When `mass_lr_mult != 1.0` OR a friction field is active, build one `optax.multi_transform` over groups `{main, mass, field}`; a leaf is labelled `mass` if any attribute key on its path is `log_mass` (path-based ⇒ selects a lone CHLU's mass AND every unit of a `CLULattice`), `field` if under `friction_field`, else `main`. When neither is active the optimizer is exactly `optax.adam(lr)` — **historical path bit-for-bit preserved**. Reuses the existing two-timescale label-FUNCTION pattern (a CHLU-shaped labels pytree is itself callable ⇒ must pass a fn, gamma-field-build lesson).
- **Tests:** `tests/test_config.py::test_mass_lr_mult_default_and_round_trip` (default 1.0 + YAML round-trip at 100.0); `tests/test_mass_lr.py` (mult>1 moves log_mass >3× further/epoch; mult=1.0 bit-identical run-to-run).

## How I verified (real output)
- **Mechanism check** (`optax.multi_transform` + path labels): CHLU → 1 `log_mass` leaf labelled `mass`; 2-unit lattice → **2** mass leaves (one per unit). With identical grads, `mult=100` produces exactly 100× the `log_mass` update magnitude (0.001732 → 0.173204) while `main` leaves are untouched.
- **Tests:** `pytest tests/test_config.py tests/test_mass_lr.py tests/test_train_persistence.py tests/test_friction_field.py` → **30 passed** (49 s), warm venv.
- **Bit-compat / inertness:** `mass_lr_mult` default 1.0 ⇒ plain-adam path. In `newtonian_identity` mode (exp-a default) `mass_lr_mult=100` leaves `log_mass` **bit-identical** (max|Δ|=0.00e+00) — H never reads log_mass ⇒ zero grad ⇒ zero update at any lr. (Item 4, exp-a spot-check.)
- **Lint:** `ruff check` clean on all 4 changed files (pre-existing `ExperimentV1WormholeConfig` F811 in config.py is on main, out of scope); my added lines are `ruff format`-clean (train.py's remaining format drift is pre-existing on main, verified identical baseline).

---

## Doctrine test (items 2–3) — RESULTS
Mean ± std over 5 seeds {0..4}. `log-spread` = std(log M) over the 4 mass components (uniform-init sits at ~0.06; the ~0.08 "ceiling" is the quoted PCD-narrowness figure). `align` = Spearman(learned M, target `[4,4,0.25,0.25]`) ∈ [−1,1] (+1 = correct band ordering unit0>unit1). `MSE` = 255-step held-out rollout. `unitM` = (mean M unit0 slow [target 4.0], mean M unit1 fast [target 0.25]). Figure: `mass-lr-doctrine-test-summary.png`.

| arm | ep | mult | log-spread | align (Spearman) | eval-MSE | unitM (slow, fast) |
|---|---|---|---|---|---|---|
| sweep (uniform) | 300 | 1 | 0.064 ± 0.027 | −0.09 ± 0.44 | 2.416 ± 0.345 | (0.74, 0.73) |
| sweep | 300 | 10 | 0.081 ± 0.038 | +0.45 ± 0.49 | 1.771 ± 0.254 | (1.09, 1.00) |
| sweep | 300 | 100 | 0.089 ± 0.056 | +0.09 ± 0.52 | 1.264 ± 0.190 | (2.75, 2.67) |
| curriculum | 300 | 100 | 0.063 ± 0.028 | +0.09 ± 0.18 | 1.067 ± 0.173 | (4.00, 3.99) |
| **banded (ref)** | 300 | 1 | **1.370 ± 0.025** | **+0.89 ± 0.00** | 1.180 ± 0.216 | (2.86, 0.19) |
| sweep (uniform) | 1500 | 1 | 0.168 ± 0.040 | **+0.89 ± 0.00** | 1.846 ± 0.086 | (0.82, 0.61) |
| **sweep** | 1500 | **10** | **0.524 ± 0.288** | **+0.89 ± 0.00** | 0.635 ± 0.070 | (1.75, 0.98) |
| sweep | 1500 | 100 | 0.168 ± 0.031 | **−0.80 ± 0.18** | 0.324 ± 0.012 | (3.90, 5.34) |
| curriculum | 1500 | 100 | 0.496 ± 0.366 | −0.54 ± 0.72 | 0.327 ± 0.074 | (3.81, 6.06) |
| **banded (ref)** | 1500 | 1 | **1.395 ± 0.020** | **+0.89 ± 0.00** | **0.046 ± 0.005** | (3.11, 0.19) |

### VERDICT (quotable) — the doctrine is *sharpened*, not simply retired or fortified
**Mass hierarchy is PARTIALLY learnable given a mass-specific lr + a longer horizon — the correct *ordering* is reliably induced, the designed *band* is not.** Concretely:
- **The "optimizer never finds the hierarchy / M stays at the ~0.08 ceiling" claim is a TRAINING-BUDGET artifact, not a hard limit — and the control now exists.** At 1500 ep the uniform lattice orders its masses in the **correct direction on 5/5 seeds** (align +0.89 ± 0.00, i.e. unit0 heavy > unit1 light — *bit-consistent across seeds*), even at mult=1. A mass-specific lr of **10× amplifies the correct-direction spread to 0.524 ± 0.288 — 6.5× the ~0.08 ceiling** — and cuts eval-MSE to 0.635 (vs 1.846 at mult=1). So the earlier "~0.08 / never differentiates" reading (v1-l0-gate, mass-spectrum-peek, seed-sweeps) reflects the **300-epoch budget**: at 300 ep NO mult differentiates (spread pinned 0.06–0.09, masses move only *globally*: unitM slow≈fast at every mult).
- **BUT the designed-in prior remains strictly superior, so the doctrine's operative clause survives.** No induced run reaches the designed band: best learned (mult=10/1500) = (1.75, 0.98) & spread 0.52 vs designed (3.11, 0.19) & spread 1.395; the **designed lattice fits 14× better** (eval-MSE 0.046 vs 0.635 at mult=10, and still 7× better than the lowest induced MSE 0.324). The optimizer induces the *qualitative* hierarchy, never the *quantitative* one.
- **Naive over-lr is actively harmful (a genuine failure mode).** mult=100 **overshoots into a global-mass runaway that INVERTS the ordering** (align −0.80 ± 0.18 at 1500 ep; unitM (3.90, **5.34**) — the fast unit ends up *heaviest*). Its low MSE (0.324) is a trap: it comes from a global mass rescale that happens to fit, not from the band. So there is a **sweet spot at mult≈10**; more is worse.
- **Curriculum (slow-components-first) did NOT help — arguably hurt** (item 3, at best-mult=100). 300 ep: drove *both* units to ~4.0 (locked in the slow scale globally, align +0.09). 1500 ep: stayed anti-aligned (align −0.54 ± 0.72, high variance) like plain mult=100. Freezing the fast band first taught both masses toward the slow scale; curriculum + mass-lr does **not** beat mass-lr alone.

**One-line for the ledger:** *the "mass hierarchy must be designed-in or induced, not awaited" doctrine (Hyp-3) holds with the missing control now in hand — a mass-specific lr (≈10×) + 5× epochs is a genuine **inducer of the correct mass ordering** (Spearman +0.89, 5/5 seeds, spread 6.5× the ceiling) but not of the designed **magnitude** (designed prior fits 7–14× better; naive 100× lr overshoots into an ordering-inverting runaway; slow-first curriculum does not help). The ~0.08 "narrowness ceiling" is a training-budget artifact, not a wall.*

### Item 4 (optional spot-check)
- **exp-a M spread under mass-lr:** in `newtonian_identity` (exp-a default) `mass_lr_mult=100` leaves `log_mass` **bit-identical** to init (max|Δ|=0.00e+00) — H never reads log_mass ⇒ zero grad. Mass-lr provably cannot change the exp-a M spread; it is a safe no-op there (consistent with mass-spectrum-peek's identity control).
- **V2 isotropization verdict:** NOT re-run (deferred — needs an exp-d/goldstone SO(2) train with `tie_channel_mass=False` + mass-lr and a commutator/charge-drift measurement; heavier than the "cheap" budget and not load-bearing for the doctrine verdict). Flagged for a follow-up if the Hub wants the within-channel-isotropy angle.

## Git footprint
- Branch `agent/experiment-engineer/mass-lr-doctrine-test` (worktree `../CHLU-mass-lr-doctrine-test`), base `main` @ `db3369b`.
- Commits: `be12b73` (config.py, training/train.py), `df11483` (tests/test_config.py, tests/test_mass_lr.py). No rebase needed (main unmoved). Not pushed, no PR.
- Files touched: `chlu/config.py` (+12), `chlu/training/train.py` (+64/−22 incl. block restructure), `tests/test_config.py` (+12), `tests/test_mass_lr.py` (+75 new). Experiment scripts/artifacts under `.claude/scratch/mass-lr-doctrine-test/` (untracked).
- Worktree isolation per §3.2 (concurrent `minus-the-physics` also touches `chlu/`); branch ref verified from main repo before any removal (see below).
- Suite: 30 relevant tests green pre-format-edit; 8 (config+mass_lr) re-confirmed green post-format-edit. Pre-existing `ExperimentV1WormholeConfig` F811 in config.py left untouched (out of scope, on main).

## Open questions / follow-ups / risks
1. **The sweet spot is data-specific.** mult≈10 is the inducer here on a 2-unit / 16× data-mass-ratio testbed; the optimal multiplier and whether the overshoot-inversion at 100× generalizes to N>2 / deeper ratios is untested. A short mult∈{3,10,30} × N∈{2,4} follow-up would firm up "≈10× is the inducer, 100× overshoots."
2. **Curriculum realization is one of several.** I implemented slow-first as "freeze the fast unit's motion for the first 30% of epochs." It underperformed (locked the global scale). A gentler curriculum (amplitude ramp of the fast band, or sampling-frequency schedule) might behave differently; the current negative is for *this* realization. Cheap to retry if the Hub cares.
3. **Interaction with V3.2 (P10):** per the interaction map, P5's outcome shapes the banding story. This result says: banding is **not** purely "a prior the optimizer cannot reach" (it reaches the *ordering*), so V3.2's framing should become **"designed-in gives the magnitude the optimizer only approaches; a mass-specific lr is a partial inducer (direction yes, band no) + a selection recipe still pays."** Not "optimizer never finds it."
4. **MSE-as-fit is a confound for the doctrine metric.** mult=100/1500 has the *lowest* sweep MSE (0.324) yet the *worst* alignment (−0.80): global mass rescaling improves fit independent of hierarchy. Any claim must read alignment/spread, not MSE, as the hierarchy signal (MSE rewards the runaway).
5. Item-4 V2-isotropization spot-check deferred (see above) — the only task sub-item not executed.

## Proposed handover updates (for the Hub)
- **§8 / ledger "mass-narrowness pivot" + Hyp-3 doctrine — UPDATE (critique P5/G4 ANSWERED, control now exists):** the mass-specific lr was run (branch `agent/experiment-engineer/mass-lr-doctrine-test`). Verdict: **the ~0.08 narrowness ceiling is a 300-epoch training-budget artifact, not a wall** — at 1500 ep the uniform lattice orders masses correctly on 5/5 seeds (Spearman +0.89±0.00) and a **10× mass-lr amplifies the correct-direction spread to 0.52 (6.5× the ceiling)** and eval-MSE 1.85→0.64; **but** no induced run reaches the designed band (designed fits 7–14× better) and **naive 100× lr overshoots into an ordering-INVERTING global-mass runaway** (align −0.80). Curriculum (slow-first) did not help. Net: **Hyp-3 "designed-in or induced, not awaited" HOLDS with the missing control** — mass-lr is a real *inducer of ordering*, not of magnitude. The three prior corroborations (v1-l0-gate log-std 0.08, mass-spectrum-peek σ_struct, seed-sweeps "optimizer never finds it") remain correct **at 300-ep scale** but should be re-scoped as budget-limited, not fundamental.
- **§3 config:** new `training.mass_lr_mult` (default 1.0 = bit-compatible; own Adam slot at `lr*mult` for every unit's `log_mass` via the unified multi_transform, shared with the friction-field two-timescale path). Round-trip + behavioral tests added.
- **§7 (if tracking): NOTE for V3.2 (P10) framing** — banding is not "unreachable by the optimizer"; the optimizer reaches the ordering, not the magnitude. Reframe accordingly (see follow-up 3).
- **Provenance:** figure `mass-lr-doctrine-test-summary.png`; per-cell rows `.claude/scratch/mass-lr-doctrine-test/results.jsonl` (50 cells, seeds 0–4).
