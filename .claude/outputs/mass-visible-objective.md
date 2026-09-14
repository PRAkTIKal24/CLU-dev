# mass-visible-objective — experiment-engineer report

Task + acceptance criterion: make the CLU objective able to *see* the mass spectrum — reproduce the w19 blindness diagnosis post-`dt`-fix, add and ablate mass-visible terms, produce a timescale-vs-mass curve, and implement per-address masses or deliver a change-list.
Status: **done** (Items 1–3 complete with ablations; Item 4 — the bounded core prerequisite **implemented**, the full codebook delivered as a change-list). Tests green: **431 passed, 0 failed**.

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5, first 10 lines).**
> 1. **The w19 "softplus trap" is largely a DEAD ISSUE post-`dt`-fix, and should stop being quoted as a live blocker.** It was a consequence of the `E_reg` common-mode runaway; with `E_reg` at 0.16% of the mass gradient, `log_mass` never reaches softplus's linear regime. Reparameterizing to `exp` now buys only **1.089×** at matched `log_mass` (w19 implied a much larger prize).
> 2. **My PREREG P4 (timescale exponent +0.5) FAILED as registered** — measured **0.78–0.80** (R²>0.99). The *mechanism* is confirmed far more strongly than predicted; I registered the wrong damping limit (underdamped 0.5 when the probe is heavily damped, where the limit is 1.0). Anyone quoting "τ ∝ √M" for this codebase should quote **τ ∝ M^0.79** instead.
> 3. **`neg_momentum_scale` is a new shipped knob that is OFF by default.** No existing default changed. Nothing in the negatives registry moves without someone turning it on.
> 4. **This task did NOT measure FD001 h-AUROC.** Every number here is a mass/timescale diagnostic. Whether any of this helps the metric is **untested** — see risk 1. Do not let "the mass spectrum is fixed" propagate as "CLU improved".

---

## Answer first

1. **The defect reproduces EXACTLY post-`dt`-fix, and it is structural, not a units artifact.** `∂E_contrast/∂log_mass` is **bit-zero** (`np.all(g == 0.0)` is `True`, not "small"). Analytically: `H = K(p) + V(q)`, negatives are `(q+δq, p)`, so the kinetic terms cancel identically in `⟨H_data⟩ − ⟨H_neg⟩`. My baseline arm reproduces `dt-units-split` to **4 d.p. on every quantity** (`M_max/M_min` 1.3489, common:diff 3.28:1, mass-gradient shares 99.84/0.00/0.16%).
2. **Mass-perturbed negatives fix it at the source.** `E_contrast`'s share of the mass gradient goes **0.00% → 42.70%**, and `M_max/M_min` **1.3489 → 2.4581**. The ablation is decisive: keep the momentum perturbation but zero the contrastive weight and the gain **vanishes to 1.3005** — *below* baseline. The gain runs through the repaired term, not through a side effect.
3. **All three candidates work, and they compose.** Combined (`exp_zeromean` + `neg_momentum_scale=0.5`) reaches **`M_max/M_min` = 4.1051** vs a 1.2656 init and a 1.3489 baseline — a **3.0× improvement over baseline**, and P5's pre-registered ">2.0" holds.
4. **The spectrum is FUNCTIONAL — this is the acceptance criterion and it passes.** Using the new per-launch `mass_override` to sweep mass on a **fixed landscape** (so every curvature `k_i` is bit-identical), **τ ∝ M^0.779 (R²=0.9968)** baseline and **M^0.796 (R²=0.9910)** combo. Mass genuinely sets the timescale. The mass-*attributable* timescale spread rises **1.27× → 3.05×**.
5. **Per-launch masses are implemented** (`T`/`H`/`step`/`__call__` take `mass_override`, default `None` = bit-identical). The full per-address codebook is a change-list (§Item 4).

---

## Item 1 — the diagnosis, reproduced post-`dt`-fix

### The contrastive term's mass gradient is identically zero — analytically and numerically

`H(q,p) = K(p) + V(q)` with the mass entering only `K`. Negatives are built as `(q + δq, p)` — **same `p`**. Hence

```
E_contrast = ⟨H(q,p)⟩ − ⟨H(q+δq,p)⟩ = [⟨K(p)⟩ − ⟨K(p)⟩] + [⟨V(q)⟩ − ⟨V(q+δq)⟩]
           = ⟨V(q)⟩ − ⟨V(q+δq)⟩                       (mass-free, identically)
```

Measured (`zero_grad.py`, dim 14, 512 samples), RMS `|∂/∂log_mass|`:

| `neg_momentum_scale` | rms `\|g_logmass\|` | max `\|g\|` | **exactly zero?** | rms `\|g_V\|` (reference) |
|---|---|---|---|---|
| **0.00 (shipped)** | **0.000000e+00** | **0.000000e+00** | **True** | 1.4050e-03 |
| 0.10 | 5.302598e-03 | 8.720160e-03 | False | 1.0155e-03 |
| 0.50 | 1.306335e-01 | 1.678259e-01 | False | 1.0155e-03 |
| 1.00 | 5.385490e-01 | 6.705958e-01 | False | 1.0155e-03 |

Not a small number — **bit-zero**, as the identity requires.

### Reproduction of the four w19 claims at the corrected units

| # | w19 claim (at `dt=0.05`) | post-`dt`-fix (my baseline arm) | verdict |
|---|---|---|---|
| 1 | `E_contrast` has exactly zero mass gradient | **0.00% of the mass gradient; bit-zero** | ⭐ **SURVIVES — structural** |
| 2 | 99.8% of the mass gradient from `E_reg` | **0.16%** (predict_mse now 99.84%) | **fixed by the `dt` split** |
| 3 | differential:common 1:39; ratio 1.153 **below** init | common:diff **3.28:1**; ratio **1.3489 above** init 1.2656 | **fixed by the `dt` split** |
| 4 | softplus trap (log spread stops buying range) | present but **minor**: `exp` buys only **1.089×** at matched `log_mass` | **largely defused** |

**The `dt` fix already resolved (2) and (3) — that is `dt-units-split`'s result and my baseline reproduces it to 4 d.p.** What remains, and what this task repairs, is **(1)**: the objective's representational half is blind to the mass. **(4) is now a minor effect**, because it was downstream of the `E_reg` runaway that the `dt` fix removed.

---

## Item 2 — candidates and ablations

FD001, seed 42, 150 epochs, 4000 windows, `dt=0.125`/`data_dt=1.0`. `common:diff` is the **gauge-invariant** version (drift of `log M_effective`, i.e. after the parameterization) so the `_zeromean` arms are comparable.

| arm | `M_max/M_min` | vs init 1.2656 | common:diff | `E_contrast` share of mass-grad | `E_reg` share |
|---|---|---|---|---|---|
| **baseline** | 1.3489 | +6.6% | 3.43 | **0.00%** | 0.16% |
| (a) `neg_momentum_scale=0.25` | 1.8243 | +44% | 0.51 | 0.73% | 0.58% |
| **(a) `neg_momentum_scale=0.5`** | **2.4581** | **+94%** | 1.29 | **42.70%** | 1.44% |
| (a) `neg_momentum_scale=1.0` | 2.8346 | +124% | 3.88 | 19.75% | 6.32% |
| (b) `softplus_zeromean` | 1.7087 | +35% | **0.02** | 0.00% | 0.43% |
| (c) `exp` | 1.6640 | +32% | 2.33 | 0.00% | 21.16% |
| (b+c) `exp_zeromean` | 1.8636 | +47% | **0.00** | 0.00% | 6.43% |
| **(a+b+c) combo** | **4.1051** | **+224%** | **0.00** | 27.79% | 2.13% |
| *ablation:* `energy_reg=0` | 1.4235 | +12% | 4.82 | 0.00% | — |
| *ablation:* `zeromean`+`energy_reg=0` | 1.6253 | +28% | 0.02 | 0.00% | — |
| ⭐ *ablation:* `negp=0.5`+`energy_weight=0` | **1.3005** | **+2.8%** | 14.09 | (70.77%)† | 1.69% |

† **Read with care:** my partition measures `∂(term)/∂log_mass` for each term *independently of its loss weight*. At `energy_weight=0` that gradient exists but is multiplied by zero and **never enters an update**. The arm is the ablation, not a counterexample.

**Every candidate raises the ratio, and all three ablations behave as a real fix should:**

- **(a) is confirmed at its source.** `negp=0.5` gives 2.4581. Remove the contrastive term's weight while *keeping* the momentum perturbation → **1.3005**, i.e. the entire gain disappears (and lands below baseline). The gain is caused by the repaired contrastive term, not by momentum noise acting as a generic regularizer.
- **(b) is not re-routed `E_reg` pressure** — the failure mode the task warned about. With `energy_reg=0` entirely, `zeromean` still gives **1.6253** against a `baseline_noreg` of **1.4235**. The gauge fix contributes on its own.
- **(c) is real but small.** At *identical* `log_mass` (same PRNG key), `exp` gives 1.3776 vs softplus 1.2656 = **1.089×**. The heuristic `ratio ≈ exp(3.4·std(log M))` tracks all 11 arms within ~15% (exp arm: predicted 1.681, measured 1.664).
- **Non-monotone in σ:** `negp=1.0` raises the ratio further (2.8346) but pushes common:diff back up to 3.88 and `E_reg` to 6.32% — the perturbation starts inflating `H` again. **σ=0.5 is the better operating point** despite the lower raw ratio.

---

## Item 3 — is the spectrum *functional*? (the real acceptance criterion)

### The confound, and why the pre-registered univariate test could not settle it

Each channel has its own **learned** curvature `k_i`, and `τ ∝ √(M/k)` (underdamped) or `τ ∝ M/k` (overdamped). Regressing `log τ` on `log M` alone is therefore confounded, exactly as PREREG P4 warned in advance. Measured `corr(log k, log M)` ranges from **+0.45 to −0.93** across arms, and the univariate slope scatters from **−0.86 to +2.83** with R² mostly <0.5. **The pre-registered univariate test is uninformative here** — not because the effect is absent, but because the regressor is confounded.

The declared-secondary two-variable fit `log τ = a·log M + b·log k + c` is much better behaved, and **recovers the predicted curvature exponent**:

| arm | a (log M) | b (log k) | R² |
|---|---|---|---|
| baseline | 1.517 | **−0.503** | 0.928 |
| negp0.5 | 0.753 | **−0.566** | 0.967 |
| negp1.0 | 0.669 | **−0.727** | 0.939 |
| baseline_noreg | 0.540 | **−0.487** | 0.950 |
| exp | 0.731 | **−0.559** | 0.949 |
| exp_zeromean | 1.701 | **−0.470** | 0.961 |

`b ≈ −0.5` in every well-fit arm, as `τ ∝ k^-1/2` predicts.

### ⭐ The decisive measurement — mass swept on a FIXED landscape

Rather than regress the confound away, the new per-launch `mass_override` **removes it by construction**: sweep a global multiplier `α` on one trained model and every `k_i` is bit-identical across the sweep, so any change in `τ` is caused by mass alone.

| α | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 | 8.0 | 16.0 |
|---|---|---|---|---|---|---|---|
| `dt·ω` (∝ α^−1/2, baseline 1.68) | 3.36 ❌ | 2.38 ❌ | 1.68 | 1.19 | 0.84 | 0.59 | 0.42 |
| baseline mean τ | *11.05* | *0.66* | 0.795 | 1.223 | 2.134 | 3.705 | 6.786 |
| combo mean τ | *7.65* | *0.92* | 1.348 | 2.054 | 3.313 | 6.464 | 12.000 |

**α ≤ 0.5 is excluded on an a-priori stability criterion, not by curve-fitting:** `ω ∝ 1/√M`, so lightening the mass drives `dt·ω` past Verlet's limit of 2 (3.36 and 2.38). Those two points are integration artifacts — the same trap `dt-units-split` documented, now reached from the mass side. On the stable branch:

| arm | fit range | **τ ~ M^slope** | R² |
|---|---|---|---|
| baseline | α ≥ 1 | **M^0.7787** | **0.9968** |
| combo | α ≥ 1 | **M^0.7962** | **0.9910** |
| baseline | α ≥ 0.5 | M^0.6934 | 0.9780 |
| combo | α ≥ 0.5 | M^0.7429 | 0.9873 |

**Verdict: the timescale hierarchy is real and cleanly mass-addressed.** The exponent ≈**0.79** sits between the underdamped (0.5) and overdamped (1.0) harmonic limits, consistent with the probe's damping budget (γ·n·dt = 5.0). Both limits agree that mass sets the timescale, so the conclusion does not depend on which limit one assumes.

**Mass-attributable timescale spread** (`(M_max/M_min)^0.79`, curvature held fixed):

| arm | mass ratio | **τ spread from mass alone** |
|---|---|---|
| baseline | 1.3489 | **1.27×** |
| negp0.5 | 2.4581 | 2.04× |
| **combo** | 4.1051 | **3.05×** |
| *ablation* negp0.5_noecon | 1.3005 | 1.23× |

**The objective fix takes the mass-addressed timescale hierarchy from 1.27× to 3.05×.** ⚠ Caveat: baseline's raw `τ_max/τ_min` across channels is already 4.5, but that is driven by **curvature**, not mass — a hierarchy the address selector cannot reach. The point of this work is that the *mass-addressable* fraction grew 2.4×.

---

## Item 4 — per-address masses

### Implemented (bounded): the per-launch mass override

The theorist's OQ-B spec, `2286a23`. `mass_override` threads through `T → H → step → __call__`; `None` (default) keeps the global trainable mass.

Verified: default path **bit-identical**; override with the global mass reproduces it to **0.0**; a 4× launch mass gives **0.277×** displacement against the 1/4 free-streaming prediction; the override **carries gradient** (finite, non-zero), so the key is learnable. This is the change w19 called the one "the architecture cannot proceed without" — as shipped, `m` was a global parameter shared by every rollout and therefore not an address component at all.

### Change-list for the full per-address codebook (NOT implemented — task-sized)

The remaining work is architectural, not a fold-in. Precise list:

1. **`chlu/eval/config.py`** — new `AddressCodebookConfig`: `n_addresses: int`, `address_source: "free" | "derived"`, `mass_init_spread: float`. Default `None` ⇒ current single-global-mass behaviour.
2. **New `chlu/core/codebook.py`** — `AddressCodebook(eqx.Module)` holding `log_mass_k: (K, dim)` (+ optionally `q0_k`, `p0_k`). Per Prop 6 only the **overall scale** is gauge, so pin it (`exp_zeromean` per entry, already implemented) and let the K−1 ratios train.
3. **Selection, not search.** The theorist measured **cross-basin gradient address search dead under 4 protocols** (best 3/18 ≈ chance). So the codebook must use **derived addresses** — `a_i = derive(V_θ, i)` via relaxation to the basin minimum — *not* a learned soft-attention over `k`. This is the main design decision and the reason this is a separate task.
4. **`chlu/eval/clu_scorer.py`** — 2 call sites only (`:247` `rollout_on_data_grid`, and the wake roll in `loss_fn`): thread `mass_override=codebook.mass(i)`. `chlu/eval/cafe_model.py:120` likewise for the encode path.
5. **Loss** — per-item, with the K-dependent D1 partition **`η_m/(η_m + K·η_k)`** (⚠ *not* the old `1/11`; K=2 ⇒ 1/21).
6. **Tests** — ratios learnable while the scale stays frozen (Toy E's 2.2e-14 is the target precision).

Effort: ~1 task. Items 1–2 and 4 are mechanical; item 3 carries the research risk.

---

## How I verified

- **Tests:** `uv`-equivalent `pytest tests/ -q` on the main venv → **431 passed, 0 failed** (`fullsuite3.txt`); baseline before my work was 429+1F. 22 new tests in `tests/test_mass_visible_objective.py`. Verified green under **both** float32 and `JAX_ENABLE_X64=1`.
- `ruff check` clean on all 4 touched files.
- **Backward compatibility:** all new knobs default-off; the default arm reproduces `dt-units-split` to 4 d.p. on every mass quantity. `neg_momentum_scale == 0` deliberately does **not** split the PRNG key, so the default RNG stream is unchanged.
- **Two of my own tests failed and were fixed** — both were my errors, not the code's; see Honest failures below.
- Harnesses in `.claude/scratch/mass-visible-objective/`: `zero_grad.py`, `arms.py` (11 arms), `analyze.py`, `timescale_override.py`. Raw: `arms.json`, `timescale_override.json`, `arms.log`, `tso.log`, `fullsuite3.txt`.

### PREREG scoring (`.claude/outputs/mass-visible-objective/PREREG.md`, written before `arms.py` ran)

| # | prediction | measured | verdict |
|---|---|---|---|
| P0 | baseline reproduces `dt-units-split` exactly | 1.3489 / 3.28:1 / 99.84-0.00-0.16% | ✅ **exact** |
| P1 | `E_contrast` share >10%; common:diff <3.28; ratio 1.4–2.5 | 42.70%; 1.29; **2.4581** | ✅ **holds** |
| P2 | zeromean ratio >1.3489; survives `energy_reg=0` at >1.30 | 1.7087; **1.6253** | ✅ **holds** |
| P3 | `exp` > softplus at matched `log_mass`; `≈exp(3.4s)` | 1.3776 vs 1.2656; 1.664 vs 1.681 predicted | ✅ **holds** |
| P4 | **τ slope +0.5, in [0.35,0.65], R²>0.5** | **0.779 / 0.796, R²>0.99** | ❌ **FAILS as registered** |
| P5 | combo `M_max/M_min` > 2.0 | **4.1051** | ✅ **holds** |

**P4 is the informative failure.** I registered the *underdamped* limit (τ∝√M) but the probe runs at damping budget 5.0, where the correct limit is τ∝M. Measured 0.79 sits between. **The mechanism is confirmed far more strongly than registered (R²>0.99); the exponent I committed to was derived from the wrong regime.** Reported per §5: a pre-registered prediction that fails is a finding, and the correction is that this codebase's law is **τ ∝ M^0.79**, not τ ∝ √M.

### Honest failures (mine)
1. `test_contrastive_mass_gradient_scales_as_sigma_squared` failed in the full suite (3.55 vs 4.0±15%). **The code was right, the test wrong**: the σ² law governs the *expectation*; a single draw carries an O(σ) cross-term `−Σpᵢδᵢ/Mᵢ`. Averaging 64 keys recovers the law to **0.5%** (3.979/3.998/4.000). Fixed in `14334e5`.
2. The companion crossover test then passed standalone but **failed in the full suite** — it keyed off a single noise realization, so another module enabling x64 changed the number. Replaced with a key-averaged magnitude statistic; verified under both precisions. Fixed in `186e0aa`.
3. My first `arms.py` run **crashed after ~6 min of training** on a vmap shape bug (`n_probe=128` vs a 64-window batch). Re-run from scratch; no results were salvaged from it.

### Flag provenance
| item | value |
|---|---|
| commits | `b296023` (knobs), `2286a23` (mass_override), `14334e5`, `186e0aa` (test fixes) |
| **base** | **`agent/experiment-engineer/dt-units-split` @ `0eec592`, NOT `main`** — see Git footprint |
| seeds | **42 (single seed, all 11 arms)**; known seed spread on this path ≈0.002 h-AUROC, **unmeasured for mass-ratio** |
| dataset | `cmapss_fd001`, cached `X_train` (17731, 30, 14), reused from `dt-units-split` scratch |
| model | `CHLU`, `kinetic_mode=newtonian_learned`, `potential_type=mlp`, `hidden=64` |
| CLU config | `dt=0.125`, `data_dt=1.0` (substeps 8), `gamma=0.1`, `epochs=150`, `lr=1e-3`, `batch=64`, `max_fit_windows=4000`, `predict_horizon=16`, `relax_steps=32`, `neg_noise_scale=0.5`, `energy_reg=0.005`, `momentum_init=finite_diff`, `mass_lr_mult=1.0`, `mass_spread_lambda=0.0`, no lattice |
| **varied** | `neg_momentum_scale ∈ {0, 0.25, 0.5, 1.0}`; `mass_parameterization ∈ {softplus, exp, softplus_zeromean, exp_zeromean}`; `energy_reg ∈ {0.005, 0}`; `energy_weight ∈ {1, 0}` |
| diagnostics | gradient partition on the **trained** model, batch 64 seed 0; timescale probe 128 windows (override sweep: 64), 400 steps, γ=0.1, unit kick, 1/e decay |
| **not measured** | **FD001 h-AUROC — no CAFE probe was run in this task** |
| env | main `.venv` reused from the worktree (`PYTHONPATH`, no `uv sync`), JAX **0.9.0**, equinox 0.13.4, CPU |
| prereg | `.claude/outputs/mass-visible-objective/PREREG.md`, written before `arms.py` |

---

## Git footprint
- Branch **`agent/experiment-engineer/mass-visible-objective`**, **based on `agent/experiment-engineer/dt-units-split` @ `0eec592`** (the task declares a hard dependency on that work, which is **not yet merged to `main`**). Rebase onto that base: up to date, no-op. **Not pushed.**
- Commits: `b296023`, `2286a23`, `14334e5`, `186e0aa`. Branch ref **verified from the main repo** before worktree teardown (protocol §3.2, w4 lesson).
- Files: `chlu/core/chlu_unit.py`, `chlu/eval/config.py`, `chlu/eval/clu_scorer.py`, `tests/test_mass_visible_objective.py` (new, 22 tests).
- **Worked in worktree `../CHLU-massvis`** — required, because `agent/experiment-engineer/dt-units-split` was checked out in the main repo and three other agent worktrees were live. No collision; main checkout untouched.
- ⚠ **Merge note for the Hub:** this branch must be merged **after** `dt-units-split`, or it will carry those 3 commits in with it.

## Open questions / follow-ups / risks
1. ⭐ **No task metric was measured.** I fixed a *representational* defect and verified it produces a *dynamical* hierarchy, but **nobody has checked whether any of this moves FD001 h-AUROC** (baseline 0.6569, raw-stats reference **0.7486**). It could easily be neutral — `dt-units-split` corrected the physics at zero metric cost, and this may too. **Cheapest high-value follow-up: run the combo arm through CAFE.** Until then, no claim about task performance is licensed.
2. **The task's suppression warning — my read.** The `negp` gain is *not* obviously suppression: unlike `mass_lr_mult`, it works by adding a gradient path that was structurally absent, and the ablation localizes the gain to that path. But I did **not** measure the force/free-streaming share, which is the actual suppression test w19 used. **That check is outstanding**, and until it is run "not suppression" is an argument, not a measurement.
3. **Single seed on all 11 arms.** The mass-ratio seed spread is **unmeasured** on this path. The combo-vs-baseline gap (4.11 vs 1.35) is large, but the smaller contrasts (e.g. `zeromean` 1.709 vs `zeromean_noreg` 1.625) are **not seed-resolved** and should not be over-read.
4. **A new stability hazard, from the mass side.** `ω ∝ 1/√M`, so *lightening* masses breaks Verlet just as sharpening curvature does — measured directly here (α=0.5 ⇒ `dt·ω`=2.38). Any mass-spreading term can walk `M_min` into instability. This compounds `dt-units-split`'s §7 hazard and strengthens the case for a **mass floor**, which `exp_zeromean` only partly supplies (it pins the geometric mean, not the minimum).
5. **`mass_spread_lambda` (R-1) was not combined with these candidates.** It is now unblocked and attacks the same target more directly; whether it is additive with (a) is untested.
6. **`τ` was probed at γ=0.1 with 400 steps (budget 5.0), which is far more damping than the encode path's 0.40.** The exponent 0.79 is a property of that probe. A lighter-damped probe should sit nearer 0.5 — worth one cheap re-measure before the τ∝M^0.79 law is quoted as general.
7. I did **not** re-run **voraus**, which shares `_SharedCLUFit`. All my changes are default-no-op, so it should be unaffected — but unlike `dt-units-split` I also did not verify it.

## Proposed handover updates (for the Hub)
1. **§7 — the "EBM contrastive term has zero mass gradient" issue is RESOLVED, conditionally** (`b296023`): a fix exists and is verified, but ships **OFF** (`neg_momentum_scale=0.0`). Record as "fix available, not enabled by default".
2. **§3 config table:** add `CLUScorerConfig.neg_momentum_scale` (default **0.0**), `CLUScorerConfig.mass_parameterization` (default **"softplus"**), `CHLU.mass_parameterization` (default **"softplus"**), and the `CHLU.T/H/step/__call__` **`mass_override`** parameter (default `None`). **No existing default changed.**
3. **§7 — DOWNGRADE the w19 "softplus trap".** It was downstream of the `E_reg` runaway; post-`dt`-fix `exp` buys only 1.089× at matched `log_mass`. It is no longer a blocker for mass-hierarchy work.
4. **§7 — NEW hazard: mass-side integrator instability.** `ω ∝ 1/√M` ⇒ mass-spreading terms can drive `dt·ω` past 2 via `M_min` (measured 2.38 at α=0.5). Compounds the existing curvature-side hazard; argues for a **mass floor**.
5. **Record the mechanism:** τ ∝ **M^0.79** (R²>0.99, fixed landscape) — **not** τ ∝ √M. Mass-attributable timescale spread **1.27× → 3.05×** with the combined fix.
6. **Prop 6 / OQ-B prerequisite is DONE** — per-launch `mass_override` implemented and tested. The remaining per-address codebook work is scoped as a change-list (§Item 4) and needs a task; the key design constraint is that **addresses must be derived from `V_θ`, not searched** (theorist: cross-basin search dead under 4 protocols).
7. **Reconciliation owner needed** for the two corrections in my first 10 lines (softplus-trap downgrade; τ∝M^0.79 replacing τ∝√M).
8. **Flag for the running log:** this task measured **no task metric**. Guard against "mass spectrum fixed" being read as "CLU improved".
