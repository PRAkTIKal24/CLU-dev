# fix-pack-6 — experiment-engineer report

**Task + acceptance criterion:** clear the ⛔ blocking `fdt` NaN-gradient at γ=0 (proved by an end-to-end `train_chlu`+`fdt` smoke), delegate `transforms.effective_mass`, make the `fdt` docs true per kinetic mode + add a relativistic-Gibbs guard-rail, record the absorb-only design decision; defaults unchanged, suite green.

**Status: done.** All four items landed. Blocker cleared and demonstrated end-to-end in all three kinetic modes. Full suite **268 passed, 1 skipped** (baseline 217+1; +51 tests). Defaults byte-for-byte unchanged. `lattice.py` never touched.

⚠ **I found and fixed a second-order bug in my own item-3 guard-rail before committing it** (§4). The first implementation was *silently dead inside `jit`* — i.e. dead on the exact `train_chlu`/`train_generative` paths it exists to protect. It is now tested under `filter_jit`. Reporting because a "warning added ✅" claim would otherwise have been false.

---

## 1. Flag-provenance table (§5, mandatory)

| item | value |
|---|---|
| branch | `agent/experiment-engineer/fix-pack-6`, 6 commits, tip **`beae8f5`** |
| base | local `main` @ **`d6f8bac`** (unmoved throughout; `git rebase main` → "up to date", a no-op) |
| `origin/main` | `40c2f31` — **stale, deliberately NOT rebased onto** (protocol §3.5 / §7.21) |
| env | **jax 0.9.0**, equinox 0.13.4 — **main venv reused, no `uv sync`** (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python`) ⇒ no w6 version drift |
| worktree | `../CHLU-fix-pack-6` (sibling `lattice-xy-prereqs` held `../CHLU-lattice-xy-prereqs`); verified branch ref from main repo, then removed |
| precision | float32 (JAX default; `jax_enable_x64` NOT set) |
| seeds | model `PRNGKey(0)`/`PRNGKey(42)`; noise `PRNGKey(1)`; smoke `PRNGKey(42)`→split |
| **smoke run config** | `train_chlu`, `epochs∈{3,4,5}`, `sleep_steps=5`, `window_size=16`, `batch_size=8`, `dim=2`, `hidden=16`, data = 80-step analytic figure-8; **`langevin_noise="fdt"` is the ONLY non-default** |
| defaults in effect during smoke | `sleep_temperature=0.5`, `sleep_friction=0.0`, `sleep_frequency=5`, `dt=0.05`, `lyapunov_penalty="max"`, `lyapunov_lambda=0.01`, `clamp_strength=1000`, `use_governor` n/a |
| kinetic modes exercised | all three: `newtonian_identity`, `newtonian_learned`, `relativistic` |
| **config defaults changed** | **NONE.** `config.py` diff is *comment-only* (verified: zero non-comment changed lines). `langevin_noise='legacy'`, `sleep_friction=0.0`, `sleep_temperature=0.5` all unchanged |

---

## 2. Item 1 — ⛔ the blocker. **Cleared.**

### 2.1 Reproduced first, at `d6f8bac`, before touching anything
`jax.grad` of one `stochastic_step` w.r.t. the learnable `log_mass`:

| kinetic_mode | noise | γ=0 | γ=1e-12 | γ=0.05 |
|---|---|---|---|---|
| newtonian_identity | legacy / fdt | finite | finite | finite |
| newtonian_learned | legacy | finite | finite | finite |
| **newtonian_learned** | **fdt** | **NaN** | finite | finite |
| relativistic | legacy | finite | finite | finite |
| **relativistic** | **fdt** | **NaN** | finite | finite |

Exactly the theorist's table (`xy-lattice-theory` §5 i-b). End-to-end, pre-fix, at repo defaults + `fdt`:
```
kinetic=newtonian_learned losses=[254.19472  nan  nan  nan]
finite losses: False | NaN params: 338 | target_energy=nan
```

### 2.2 The fix (`chlu/core/integrators.py`, 1 hunk)
Theorist-prescribed double-`where` safe sqrt, verbatim:
```python
arg  = m_eff * temperature * gamma * (2.0 - gamma)
safe = jnp.where(arg > 0.0, arg, 1.0)
noise_scale = jnp.where(arg > 0.0, jnp.sqrt(safe), 0.0)
```

**Bit-identity of the value** (`m_eff=0.7321, T=0.5`), old vs new, `jnp.array_equal`:

| γ | 0 | 1e-12 | 1e-8 | 0.05 | 0.2 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|
| old | 0 | 8.556284001e-07 | 8.556283865e-05 | 0.1889176369 | 0.3630124032 | 0.6050206423 | 0 |
| new | *identical* | *identical* | *identical* | *identical* | *identical* | *identical* | *identical* |
| bit-identical | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Gradient at γ=0:** `d(noise_scale)/d(m_eff)` = **`nan` (old)** → **`0.0` (new)**. (γ=2.0 also covered: `arg=0` there too, both give 0.)

### 2.3 ⭐ THE ACCEPTANCE EVIDENCE — end-to-end `train_chlu` + `fdt` at repo defaults
Verbatim output (`langevin_noise="fdt"` the only non-default; the sleep step fires at epoch 0):

```
repo defaults in effect: sleep_temperature=0.5 sleep_friction=0.0 sleep_frequency=5 dt=0.05 lyapunov_penalty=max

kinetic=newtonian_learned  losses=[2.54194717e+02 1.41735031e+02 1.00719025e+02 1.61575079e-01 ...]
finite losses: True | NaN params: 0 | target_energy=0.9978944659233093

kinetic=relativistic       losses=[2.4966888e+02 1.4605109e+02 8.8963097e+01 1.5563729e-01 1.3207516e-01]
finite losses: True | NaN params: 0 | target_energy=1.8034588098526

kinetic=newtonian_identity losses=[2.50626099e+02 1.48366638e+02 8.96052933e+01 1.57426417e-01 ...]
finite losses: True | NaN params: 0 | target_energy=0.7306750416755676
```
5 epochs, **finite losses, 0 NaN parameters, loss decreasing**, all three kinetic modes. Grad through a jitted γ=0 fdt scan: `log_mass` grad `[7.42e-07, 1.03e-06]`, finite.

> **An FDT-correct model can now be trained at repo defaults.** Thread-10 / KT, the `T_φ(q)` build, and the Gibbs-sampling claims are unblocked — *subject to* the kinetic-mode restriction of §4 (Newtonian only, per CM-17, which `xy-lattice-theory` P4 already mandates).

---

## 3. Item 2 — `transforms.effective_mass` delegated

`chlu/core/transforms.py:35` now `return model.effective_inertia()` (and the now-unused `import jax` dropped).

| kinetic_mode | tied | bitwise == old inline spelling | == `effective_inertia()` |
|---|---|---|---|
| newtonian_identity | F / T | **True** | True |
| newtonian_learned | **F** | **True** | True |
| newtonian_learned | T | False *(the fix: tie now applied)* | True |
| relativistic | **F** | **True** | True |
| relativistic | T | False *(the fix)* | True |

**Untied is strictly bit-identical, 3/3 modes** — as the task predicted (both spellings already carried `+1e-6`, unlike the `chlu_unit` case). Asserted in `test_transforms_effective_mass_bit_identical_for_untied`. Tied models now see equal channel inertias (`m_eff[0]==m_eff[1]`), so V1's `mass_weighted_squeeze` is safe on a tied checkpoint.

---

## 4. Item 3 — `fdt` docs now true; guard-rail live (**and a bug I caught in it**)

**Scoped the claim at all six sites** that repeated *"exact discrete FDT / temperatures in energy units"*: `config.py:92-97`, `integrators.langevin_step` docstring, `train.py:78`, `train_generative.py:73`, `CHLU.stochastic_step`, `CHLU.stochastic_rollout`. Each now states: Gibbs holds **only in the Newtonian kinetic modes**; root cause (*Gibbs-preserving underdamped Langevin damps the velocity `∇_pT`, the code damps `p`*); control parameter `T/(m₀c²)`; free mitigation (raise `c` or `m₀`). The `legacy` caveat (T not in energy units, no Gibbs in any mode) is stated too.

**`CHLU.thermal_causal_ratio(temperature)` → `T/(m₀c²)`** exposed. Verified against the theorist's degeneracy: `(c=1,T=8)` and `(c=0.5,T=2)` both → **8.0**. Exp-C default cell → **1.0**; `finalA` (`c=5`) → **0.04**.

**Guard-rail** — new `RelativisticGibbsWarning(UserWarning)`; **warns, never raises**. Fires iff `noise_mode="fdt"` ∧ `kinetic_mode="relativistic"` ∧ `T>0`:

| kinetic_mode | noise | T | warned |
|---|---|---|---|
| newtonian_identity | legacy / fdt | 0.5 | 0 / 0 |
| newtonian_learned | legacy / fdt | 0.5 | 0 / **0** ← fdt IS exactly Gibbs here |
| relativistic | legacy | 0.5 | 0 |
| **relativistic** | **fdt** | **0.5** | **1** |
| relativistic | fdt | 0.0 | 0 ← no noise ⇒ no sampler ⇒ no claim |

Message names the call's ratio, e.g. `… This call has T/(m0*c^2) = 0.5 (T=0.5, rest_mass=1, c=1); … Free mitigation: raise c or rest_mass until T << m0*c^2.`
**Suite-wide no-cry-wolf check:** `pytest -W error::…RelativisticGibbsWarning` → **268 passed, 1 skipped**. It fires nowhere in the existing suite. `relativistic-gibbs-expc` gets its cell, warned but running.

### ⚠ 4.1 The bug inside my own guard-rail (found by verification, fixed in `f5b6bef`)
My first probe was `float(jnp.max(jnp.asarray(temperature)))` inside `try/except`. **Inside a `jit`/`filter_jit` trace this raises `ConcretizationTypeError` even for a genuine Python `float`** (confirmed: `type(T).__name__ == 'float'`, `repr(T) == '0.5'`, yet `float(jnp.max(...))` raised). The `except` swallowed it ⇒ **zero warnings on the `train_chlu`/`train_generative` paths**, which close over a concrete `sleep_temperature` and call `stochastic_step` under `eqx.filter_jit`. The guard-rail was dead exactly where it mattered; the unit test (eager) passed regardless.

Fixed by probing through **numpy**: `np.asarray` keeps concrete values concrete and raises `TracerArrayConversionError` on true tracers (the scanned per-step temperature) — precisely the intended semantics. Now verified firing in a **real `train_chlu`** run:
```
Training CHLU: 0%| | 0/4 …/chlu/core/chlu_unit.py:484: RelativisticGibbsWarning: noise_m…
kinetic=relativistic losses=[2.4966888e+02 1.2174979e+02 4.4266897e-01 1.5552568e-01]
finite losses: True | NaN params: 0
```
and under `filter_jit` + `lax.scan` (`warnings=1`). `stochastic_rollout` warns **exactly once**, up front, on the hottest T of an annealing schedule (no per-step duplicates). Both pinned by tests.

---

## 5. Item 4 + the DO-NOT-FIX comment

- **Propagation pinned by tests** (assertions in the *test file*, not `lattice.py` — sibling's file): `CLULattice.effective_mass() == effective_inertia()` bitwise, tied **and** untied; same for `BrokenVolumeCHLU` (delegates to the wrapped `CHLU`), incl. the tie surviving the wrapper.
  **Correction to the task's item 4:** only *one* `twins.py` wrapper delegates. `UnconstrainedTwin` wraps **no** CHLU — its `effective_inertia`/`effective_mass` both return `jnp.ones(dim)` — so it was never exposed to the tie bug. Comment in `twins.py` records this accurately.
- **Absorb-only design note landed** at the noise-scale site in `integrators.py`, citing `v5-gate` §R3 with its numbers (107.77 ± 4.78× vault; coupled-bath 13.28 ± 0.12×, rejected by 8.11 ± 0.37×; `T_local=1.26e-4` vs `1e-3`), and pointing the coupled-bath idea at the S2 study behind its own flag. `FrictionField` behavior **unchanged**.

---

## 6. How I verified (commands + real output)

```
# repro (pre-fix, at d6f8bac):  NaN table above; train_chlu losses=[254.19, nan, nan, nan], 338 NaN params
python .claude/scratch/fix-pack-6/repro_nan.py            → 2/6 cells NaN  (post-fix: 0/6)
python .claude/scratch/fix-pack-6/smoke_train_fdt.py <km> → finite losses, 0 NaN params, 3 modes
python .claude/scratch/fix-pack-6/verify_fix.py           → bit-identity + ratio + warn matrix
python .claude/scratch/fix-pack-6/verify_jit_warn.py      → jit+scan: warnings=1, grads finite
pytest tests/test_langevin_fdt.py -q                      → 73 passed          (was 22 on main)
pytest -q                                                 → 268 passed, 1 skipped, 7 warnings (225 s)
pytest -q -W error::chlu.core.chlu_unit.RelativisticGibbsWarning → 268 passed, 1 skipped
ruff check <all 8 touched files>                          → All checks passed!
```
- The **1 skip is pre-existing & unrelated**: `test_industrial_loaders.py` (needs the `eval` extra).
- `pytest -W error::UserWarning` yields 3 failures in `tests/test_eval_metrics.py` — **pre-existing on `main`** (sklearn `UndefinedMetricWarning`; reproduced against the main checkout). Not mine, not touched.
- **Formatting discipline:** `ruff format --check` shows `integrators.py`, `chlu_unit.py`, `twins.py`, `train.py` were **already format-unclean on `main`** — left alone (running `ruff format` would sweep foreign reformatting into my commits; w4 precedent, §3.3). `transforms.py`, `config.py`, `train_generative.py` are clean. `tests/test_langevin_fdt.py` *was* clean on main, so I formatted it (my additions only; diff is `+375/−1`, the `−1` being the import line).

---

## 7. Git footprint

- **Branch:** `agent/experiment-engineer/fix-pack-6` off local `main` @ `d6f8bac`. **Not pushed, no PR. Left for review.**
- **Commits (6, tip `beae8f5`):**
  | hash | subject |
  |---|---|
  | `2a6cbec` | fix NaN gradient in fdt noise scale at gamma=0 *(+ absorb-only design note)* |
  | `d778efc` | delegate transforms.effective_mass to effective_inertia() |
  | `22ced20` | make the fdt flag tell the truth (CM-17) + guard-rail |
  | `f4455b5` | pin the fdt fixes with regression tests (22 → 72) |
  | `f5b6bef` | fix guard-rail: probe temperature concreteness via numpy |
  | `beae8f5` | test the guard-rail under filter_jit + no duplicates |
- **Files touched (8, +604/−30):** `chlu/core/integrators.py`, `chlu/core/transforms.py`, `chlu/core/chlu_unit.py`, `chlu/core/twins.py` (6-line comment), `chlu/config.py` (comment-only), `chlu/training/train.py` (docstring), `chlu/training/train_generative.py` (docstring), `tests/test_langevin_fdt.py`.
- **`chlu/core/lattice.py`: NOT touched** (sibling `lattice-xy-prereqs` owns it). Worked in worktree `../CHLU-fix-pack-6`; verified `main..agent/experiment-engineer/fix-pack-6` shows all 6 commits **from the main repo** before removing the worktree (protocol §3.2, wave-4 lesson). Worktree removed; `main` still `d6f8bac`, tree clean.
- **Rebase:** `git rebase main` → "up to date" (no-op; base never moved). `origin/main` correctly avoided. **No conflicts.**
- **Scratch (gitignored):** `.claude/scratch/fix-pack-6/{repro_nan,smoke_train_fdt,verify_fix,verify_jit_warn,debug_warn}.py`

---

## 8. Open questions / follow-ups / risks

1. **The guard-rail is a `warnings.warn` at trace time** ⇒ once per compilation, and invisible under `-W ignore` or when a caller has already triggered Python's "once per location" registry. It is a signpost, not an enforcement mechanism. If the Hub wants a hard stop for Thread-10, a config-level assertion (`require_gibbs: bool = False`) would be the durable form — **not** built, since the task said warn-not-raise.
2. **`stochastic_rollout` warns on the hottest T of the schedule.** For Exp-C's `1.0 → 0.01` anneal that's ratio `1.0`, correctly the worst case; the *late* chain is far more benign (`0.01`). Analysts quoting a single ratio for an annealed run should say which.
3. **The γ=0 + `fdt` combination is now numerically safe but physically degenerate:** at γ=0 the FDT noise vanishes identically, so sleep at repo defaults (`sleep_friction=0.0`) with `fdt` is *microcanonical, not canonical* — the dynamics is deterministic despite `sleep_temperature=0.5`. The blocker fix makes it **run**; it does not make it sample. `xy-lattice-theory` §5(iii) says the same. **Anyone switching to `fdt` for a thermal claim must also set `sleep_friction > 0`.** This is arguably a second, quieter trap in the defaults — flagged, not changed (out of scope, and changing it would alter behavior).
4. **Not tested: trained checkpoints.** All evidence here is synthetic/smoke-scale. The `fdt`-trained-model behavior (loss landscape, energy floor) is unexplored — `train_chlu`+`fdt` had literally never completed an epoch before this branch.
5. **`newtonian_identity` + `fdt`** is immune to *both* fixed bugs (`M_eff ≡ 1`): no tie, no learnable mass in the sqrt. Exp-A's default.

---

## Proposed handover updates (for the Hub)

### §7-CURRENT — **resolve 7.18's successor; log the blocker as FIXED**

> **7.23 [RESOLVED 2026-07-10 by `fix-pack-6`, branch `agent/experiment-engineer/fix-pack-6` @ `beae8f5`, NOT merged] The `fdt` noise had a NaN gradient at γ=0 — no FDT-correct model could be trained at repo defaults.** `integrators.py` built the fdt scale as `sqrt(max(0, m_eff·T·γ·(2−γ)))`; at `γ=0` (the **default** `sleep_friction`) that is `sqrt(0)`, whose derivative w.r.t. the learnable `log_mass` inside `m_eff` is `∞·0 = NaN`. Since `sleep_temperature=0.5` (default ⇒ sleep is stochastic) and `epoch % sleep_frequency == 0` fires at epoch 0, **`train_chlu(langevin_noise="fdt")` NaN'd every parameter on the first sleep step** (measured pre-fix: `losses[0]=254.19`, `losses[1:]=NaN`, 338 NaN params). `legacy` was immune (its sqrt argument carries no parameter). **Fixed** with the double-`where` safe sqrt: bit-identical for `arg>0` at γ ∈ {0,1e-12,1e-8,0.05,0.2,1,2}, gradient exactly `0.0` at `arg==0`. **Verified end-to-end:** finite losses + 0 NaN params over 5 epochs in all three kinetic modes at repo defaults. **Thread-10 / KT, the `T_φ(q)` build and the Gibbs claims are unblocked.** Regression: `tests/test_langevin_fdt.py::test_train_chlu_fdt_at_repo_defaults_stays_finite` (+3 more).

> **7.24 [RESOLVED 2026-07-10 by `fix-pack-6`] `transforms.effective_mass` tie bug** (was logged as the "NEW latent issue" from `fix-pack-5` §7). Now delegates to `model.effective_inertia()`. **Bit-identical for untied models** (verified, 3/3 kinetic modes); tied models now get the channel tie, so V1's `mass_weighted_squeeze` is safe on tied checkpoints. No shipped result was contaminated.

> **7.25 [NEW, OPEN — a quieter trap in the same defaults]** `fdt` + the default `sleep_friction=0.0` is **microcanonical, not canonical**: at γ=0 the FDT noise vanishes identically, so `sleep_temperature=0.5` buys *nothing* — the sleep phase is deterministic. The `fix-pack-6` fix makes this cell **run**; it does not make it **sample**. Any thermal/Gibbs claim under `fdt` must also set `sleep_friction > 0` (cf. `xy-lattice-theory` §5(iii)). Consider making this an assertion or a default change (behavior-changing — Head call).

### §7 / CM-17 — **the `fdt` flag now tells the truth in code**

> All six sites that promised *"exact discrete fluctuation-dissipation; temperatures in energy units"* (`config.py`, `langevin_step`, `train_chlu`, `train_generative`, `CHLU.stochastic_step`, `CHLU.stochastic_rollout`) are now **kinetic-mode-qualified** per CM-17, give the control parameter `T/(m₀c²)`, and name the free mitigation (raise `c` or `m₀`). New **`CHLU.thermal_causal_ratio(T)`** returns that number (verified `(c=1,T=8) ≡ (c=0.5,T=2) = 8.0`; Exp-C default = `1.0`; `finalA` = `0.04`). New **`RelativisticGibbsWarning`** fires — **warn, never raise** — iff `fdt` ∧ `relativistic` ∧ `T>0`, naming the call's ratio; silent on Newtonian+fdt, every `legacy` path, and `T=0`. Suite-wide `-W error::RelativisticGibbsWarning` is green (268 passed) ⇒ it never cries wolf. **`relativistic-gibbs-expc` will see exactly one warning per compilation and keep running.**

### §7 — **`FrictionField` absorb-only is now documented in code as load-bearing**

> A comment at the noise-scale site (`integrators.py`) records that the noise scale uses the **scalar γ only**, never `γ_φ(q)`, citing `v5-gate` §R3: absorb-only makes a friction hole a **107.77 ± 4.78× memory vault** (brake **and** refrigerator, `T_local=1.26e-4` vs `1e-3`), where the coupled-bath form gives only **13.28 ± 0.12×** — **rejected by 8.11 ± 0.37×** against its own control. Future agents are told not to "correct" it; the coupled bath belongs to the S2 study behind its own flag.

### §9 / protocol — **a testing lesson worth propagating**

> A `warnings.warn` guard-rail that is unit-tested only in **eager** mode can be **silently dead inside `jit`**: `float(jnp.max(jnp.asarray(x)))` raises `ConcretizationTypeError` inside a trace *even for a concrete Python float*, so a `try/except` concreteness probe swallows the warning on precisely the `eqx.filter_jit` training paths it protects. Probe with **numpy** (`np.asarray`), and **test the guard-rail under `filter_jit`**. Caught in `fix-pack-6` only because the end-to-end `train_chlu` smoke was inspected for the warning, not just for NaNs.

### §5 — **task-file correction (minor)**

> `fix-pack-6`'s item 4 says "both `twins.py` wrappers delegate to `CHLU.effective_mass()`". Only **`BrokenVolumeCHLU`** does. **`UnconstrainedTwin` wraps no CHLU** (`effective_inertia`/`effective_mass` ≡ `jnp.ones(dim)`), so it was never exposed to the tie bug.
