# fix-pack-7 — experiment-engineer report
Task + acceptance criterion: expose `d·Θ` (+ report it in the CM-17 warning), stop `training.sleep_friction` being a silent no-op on the generative path (default path bit-identical), implement the exact relativistic latent-mass thermostat behind a flag (verified vs pre-registered bias targets), route it on the lattice; defaults byte-for-byte unchanged; suite ≥278 green.
Status: **done.** All four items landed and verified; 302 tests pass; pre-registered bias targets reproduced.

Reconciliation-list owner note: this report contains a **§7 Known-Issues reconciliation list** (Proposed handover updates) — resolves the two w14-created defects and the "4th silent knob", adds the `fdt_relativistic` capability. Hub should action at review.

## What I did
- **Item 1 (`d·Θ`, not `Θ`).** Added `CHLU.gibbs_defect_parameter(T) → d·Θ = dim·thermal_causal_ratio` (the quantity that governs the defect); kept `thermal_causal_ratio()`=`Θ` but rewrote its docstring to say plainly that `Θ` alone does **not** govern it and the `d=1` table is `d=1`-only. `RelativisticGibbsWarning` now reports `d·Θ` (retains the old `Θ = T/(m0*c^2) = …` substring for back-compat) and states the free mitigation needs `c ≳ √(dT/m₀) ≈ 28` (not `c=5`). De-refuted the shipped "raise c is free/benign" text at 4 sites (`thermal_causal_ratio`, `integrators.langevin_step` caveat, `config.py` comment, `train_generative` docstring) — these were arXiv-bound-in-code refuted claims flagged by `fix-pack-6` §6b.
- **Item 2 (`sleep_friction` silent no-op).** `sleep_friction` is **not** dead (live in `train.py`/dynamics), so I did not delete it. Added `_resolve_generative_sleep_friction(config)`: honours `training.sleep_friction` **only when moved off its 0.0 dataclass default**, else falls back to `experiment_c.friction`. Every historical config sets `training.sleep_friction: 0.0` (== default) → all resolve to `experiment_c.friction` byte-for-byte. Resolved value + source now printed (no longer silent).
- **Item 3 (exact relativistic thermostat).** `langevin_step` gains `noise_mode="fdt_relativistic"`: MJ as a Gaussian scale mixture — one closed-form inverse-Gaussian draw `s|p` per relativistic unit (`_sample_inverse_gaussian`, Michael–Schucany–Haas), then the *same* linear-Gaussian O-step with variance `M/(2s)`. Same double-where safe-sqrt as `fdt` (0 value AND 0 grad at γ=0). `CHLU.stochastic_step` routes it (passes `m_eff, rest_mass, c`); the CM-17 warning does **not** fire for it. No default changed.
- **Item 4 (lattice).** `CLULattice.stochastic_step` routes `fdt_relativistic` with per-unit `rest_mass`/`c` arrays and `group_sizes=unit_dims`, so each relativistic unit gets its own latent scale `s` (H is kinetic-separable ⇒ MJ factorizes per unit — exact for heterogeneous lattices).

## How I verified
- **Full suite (worktree code via `PYTHONPATH`):** `302 passed, 14 warnings in 239s` (14 = by-design `RelativisticGibbsWarning` on `fdt`+relativistic). +18 new tests in `tests/test_langevin_fdt.py`.
- **Pre-registered Item-3 bias** (`.claude/outputs/fix-pack-7/PREREG.md`, written before running the harness). 1-D harmonic well, k=1, m₀=c=M=1, γ=0.1, dt=0.05, 8000 walkers × 12000 steps, burn 4000, seed=1, `Var(q)/(T/k)−1`:

  | Θ(=T) | `fdt` biased (pred / **meas**) | `fdt_relativistic` (pred / **meas**) | newt fdt ctrl | collapse |
  |---|---|---|---|---|
  | 0.5 | −0.31 / **−0.31028** | +0.0006 / **+0.00079** | +0.00038 | 388× |
  | 2 | −0.54 / **−0.53535** | +0.0011 / **+0.00205** | +0.00038 | 261× |
  | 8 | −0.73 / **−0.72710** | +0.0011 / **+0.00460** | +0.00038 | 158× |

  **All three registered criteria met:** biased `fdt` reproduces the theorist to <0.01 (crit. 2, ±0.03); `fdt_relativistic` collapses |B| by 158–388× (≥30×) to a **positive** floor |B|<0.01 (crit. 1); Newtonian control flat in T at +0.00038 (crit. 3). *Honest caveat:* the Θ=8 residual (+0.0046) exceeds the theorist's +0.0011 — this is the relativistic `O(ε²)` shadow bias (their honesty note 3), which grows with T; still within the registered |B|<0.01 and ≥30× collapse. Momentum-marginal test (in-suite) confirms `fdt_relativistic` Var(p) is heavy-tailed (>3× M_eff·T at Θ=8, true 16.28×) vs `fdt` ≈ M_eff·T.
- **Defaults unchanged (proven):** `test_generative_sleep_friction_default_is_bit_identical` asserts default path → `experiment_c.friction` (0.3); `langevin_noise` default still `"legacy"`; `TrainingConfig.sleep_friction` default still 0.0; `thermal_causal_ratio` return unchanged; `fdt_relativistic` is opt-in.
- **Ruff:** clean on all 6 files.

## Flag provenance (numeric results above)
| field | value |
|---|---|
| branch tip | `c873d1b` (core code `6061ef8`) |
| harness | 1-D harmonic well, custom `H_fn` (no CHLU config / no trained model — a property of the map/measure) |
| kinetic | relativistic `T=c√(p²/M+(m₀c)²)`, m₀=1, c=1, M=1 |
| integrator | `chlu.core.integrators.langevin_step`, dt=0.05, γ=0.1 |
| noise modes | `legacy` default unused; measured `fdt`, `fdt_relativistic`, and `fdt`(newtonian control) |
| MC | 8000 walkers × 12000 steps, burn 4000, seed `PRNGKey(1)`, float32 (JAX default, no x64) |
| suite run | `pytest /worktree/tests -o addopts=""`, `PYTHONPATH=<worktree>`, main venv (JAX unchanged) |

## Findings/results
- The exact relativistic thermostat works and is cheap (one InvGauss draw/step, closed-form, jax-able), preserves MJ, and keeps momentum persistence (dominates an MJ refresh). It is the correct paper fix that "raise c" is not.
- The two w14-created defects are closed: the helper/warning now key on `d·Θ`; `training.sleep_friction` is honoured on the generative path without changing any checkpoint's history.

## Git footprint
- **Branch:** `agent/experiment-engineer/fix-pack-7` off local `main` (`df5e44d`); rebased onto local `main` = no-op (base unmoved); **not pushed** (per anonymity strategy).
- **Commits (3):** `6061ef8` items 1/3/4 core (`chlu/core/integrators.py`, `chlu/core/chlu_unit.py`, `chlu/core/lattice.py`, `chlu/config.py`) · `7a381e3` item 2 (`chlu/training/train_generative.py`) · `c873d1b` tests (`tests/test_langevin_fdt.py`, +18).
- **Worktree** `../CHLU-fix-pack-7` used for isolation; verified the 3 commits from the MAIN repo (`git log main..agent/…` shows them) **before** `git worktree remove`. Main HEAD unchanged at `df5e44d`, working tree clean.
- ⚠ **Process incident (self-caught, fully recovered).** The bash tool's `cd <worktree>` was silently ignored (persistent cwd stayed at main), so a `cat >>` heredoc appended my new tests to the **main checkout's** `tests/test_langevin_fdt.py` instead of the worktree's. Code edits (Edit tool, absolute paths) were unaffected. I detected it via `git status` (worktree test file unmodified), confirmed main showed **only** that one file changed (no foreign work), verified `head -763 main == worktree` byte-identical, copied main's file into the worktree, and `git checkout -- tests/test_langevin_fdt.py` in main to restore it. Main is now clean; all test edits live only on the branch. The earlier "73 passed" and bias runs were still valid — they executed main's (my-edited) test file against the **worktree** code injected via `PYTHONPATH` (the §4-recommended pattern), which is exactly why the new tests ran and passed there.

## Open questions / follow-ups / risks
- The relativistic `O(ε²)` shadow bias (Item-3 Θ=8 residual +0.0046 vs Newtonian +0.0004) is unquantified — matches the theorist's open item (their §Open-1 / deep-dive O5). Not a correctness issue for the momentum marginal (which is exact); a discretization shadow. Cheap follow-up if wanted.
- `fdt_relativistic` is verified on the map/measure at d≤5 (suite) and d=1 (bias harness). The chain at Exp-C's d=784 with a trained ConvNet `V_θ` is untested (cost); the measure-level exactness is d-agnostic, the chain-level extrapolation is strongly evidenced not proven (same caveat the theorist carries).
- Whether wiring `fdt_relativistic` into Exp-C dream/train fixes the 3/5/8/9 imbalance remains the conjectured empirical question (N10/O8) — out of scope here; the flag now makes that experiment runnable.

## Proposed handover updates (for the Hub)
**§7 Known Issues — resolve the two w14 follow-ups + the 4th silent knob, add the new capability:**
- **[RESOLVED] The two w14-created defects** (handover "TWO FOLLOW-UPS THIS WAVE CREATED"): (1) `thermal_causal_ratio` returned `Θ` while the operative parameter is `d·Θ` → **fixed**: `CHLU.gibbs_defect_parameter(T) → d·Θ` added, `RelativisticGibbsWarning` now reports `d·Θ` and names `c ≳ √(dT/m₀) ≈ 28` (not c=5). (2) `training.sleep_friction` silent no-op on `train_generative` → **fixed** without changing history (honour-when-non-default; default resolves to `experiment_c.friction` byte-for-byte). Branch `agent/experiment-engineer/fix-pack-7`.
- **[RESOLVED, docs] The refuted "raise c is free/benign" text in tracked code** (`fix-pack-6` §6b flag): corrected at all 4 sites; the `d=1` defect table is now labelled `d=1`-only everywhere it appears.
- **[N19 scope] "`sleep_temperature` is a no-op whenever `sleep_friction=0`" — still true for the DYNAMICS path (`train_chlu`) only.** On the generative path the resolved gamma is `experiment_c.friction` (0.3) by default, so sleep is stochastic there; N19's scope line should say "dynamics path". (Behaviour unchanged by this fix; the plumbing is just no longer silent.)
- **[NEW CAPABILITY] `langevin_noise="fdt_relativistic"`** — the exact latent-mass thermostat (CM-17 F2 / f5-corrigendum-2 §3), behind a flag, default unchanged. Preserves the Maxwell–Jüttner momentum marginal exactly (verified: bias `−0.31/−0.54/−0.73 → +0.0008/+0.0021/+0.0046`, ≥158× collapse). Routed on `CHLU` and `CLULattice` (per-unit `s`). Does not fire the CM-17 warning. This is the runnable "correct sampler" arm for a re-spec'd F-9 / N10 test.
- **CM-17 / F5-note:** the code now implements F2; the `d·Θ` control parameter is exposed as `gibbs_defect_parameter`. No matrix contradiction introduced (warning + docstrings match CM-17 v1.9).
