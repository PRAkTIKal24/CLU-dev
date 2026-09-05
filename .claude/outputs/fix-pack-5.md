# fix-pack-5 — experiment-engineer report

**Task + acceptance criterion:** fix the `effective_mass()`/`tie_channel_mass` FDT Gibbs bug so `fdt` noise uses the inertia the dynamics invert; keep `legacy`/untied bit-identical; pin Gibbs with a regression test; audit every shipped result for `langevin_noise="fdt"` ∧ `tie_channel_mass=True`.

**Status: done.** Bug fixed, regression test pins Gibbs on tied+fdt, suite green (217 passed, 1 pre-existing unrelated skip). **Audit answer: NO shipped/reported result is contaminated** — stated with evidence in §3.

⚠ **One acceptance clause needed correcting (§2.2):** "untied models must remain bit-identical" is **not achievable** and was based on a false premise. `effective_inertia()` includes the `+1e-6` that `effective_mass()` omitted, so delegating necessarily shifts untied `newtonian_learned`/`relativistic` fdt noise by `3.2e-06` relative. That epsilon **is the second half of the same bug** (the analyst flagged it: "omits the `+1e-6` that `H` inverts"). The exact, provable bit-compat guarantee is stated and tested below. Bit-identity and exact Gibbs are mutually exclusive here; I chose Gibbs, as the task's own item-1 fix prescription requires.

---

## 1. Flag-provenance table (§5, mandatory)

| item | value |
|---|---|
| branch / commit | `agent/experiment-engineer/fix-pack-5` @ **`d6f8bac`** (single commit) |
| base | local `main` @ `9bc2cf7` (unmoved during session; rebase was a no-op) |
| `origin/main` | `40c2f31` — **stale, deliberately not rebased onto** (protocol §3.5) |
| env | jax **0.9.0**, equinox 0.13.4, main venv `/Users/user/Desktop/CHLU/.venv`, `uv run --no-sync` (no worktree, no re-sync ⇒ no w6 version drift) |
| precision | float32 (JAX default; `jax_enable_x64` NOT set) |
| seeds | model `PRNGKey(0)`; noise `PRNGKey(0..3)`; MC seed `0` |
| model under test | `CHLU(dim=3, hidden=8, kinetic_mode="newtonian_learned", tie_channel_mass=True)`, `log_mass` overridden to `[-1.0, 1.5, 0.2]` via `eqx.tree_at` to exaggerate channel spread |
| MC config (§2.3 numbers) | `dt=0.02`, `gamma=0.3`, `T=0.5`, 2048 chains × 3000 steps, burn 1200 |
| MC config (test) | same, 512 chains × 1500 steps, burn 500, `rtol=0.05` |
| `langevin_noise` | both arms exercised explicitly (`legacy`, `fdt`); **repo default remains `"legacy"`** — unchanged |
| `tie_channel_mass` | both arms (`False`, `True`); **repo defaults unchanged** (`ExperimentDConfig` = `True`, everything else `False`) |
| config defaults changed | **none** |

---

## 2. What I did & how I verified

### 2.1 The fix (one method, `chlu/core/chlu_unit.py:296`)
`CHLU.effective_mass()` now **delegates to `effective_inertia()`** (the inertia `H` actually inverts: `mass_vector() + 1e-6`, with the channel tie applied). Docstring records the bug history.

I did **not** touch `integrators.py` — `langevin_step` was already correct; it faithfully consumed a wrong `m_eff`. Minimal diff: **1 hunk**, `@@ -297,34 +297,31 @@`.

**Propagation (verified, `v3_lattice.py`):** `CLULattice.effective_mass()` concatenates `u.effective_mass()`, so the lattice's `fdt` path — a **second call site the task did not name** (`lattice.py:429`, same `m_eff = self.effective_mass() if noise_mode == "fdt"` pattern) — inherits the fix for free. Confirmed `lattice.effective_mass()` is now bitwise `== effective_inertia()`, tie applied per unit. `CHLU.stochastic_rollout` and `twins` delegate too.

### 2.2 Exact blast radius (`v1_scope.py` — old path emulated bit-exactly)
Since `langevin_step` is untouched, the pre-fix `stochastic_step` is reproduced *exactly* by feeding it the old raw-softplus `m_eff`. All 12 cells (3 kinetic modes × tied/untied × legacy/fdt):

| kinetic_mode | tied | noise | `p` bitwise == ? | max\|Δp\| | max rel ΔM |
|---|---|---|---|---|---|
| newtonian_identity | F/T | legacy | **True** | 0 | 0 |
| newtonian_identity | F/T | **fdt** | **True** | 0 | 0 |
| newtonian_learned | F | legacy | **True** | 0 | 3.235e-06 |
| newtonian_learned | F | **fdt** | False | 1.371e-06 | **3.235e-06** ← epsilon only |
| newtonian_learned | T | legacy | **True** | 0 | 1.637 |
| newtonian_learned | T | **fdt** | False | 5.501e-01 | **1.637** ← THE BUG |
| relativistic | F | legacy | **True** | 0 | 3.235e-06 |
| relativistic | F | **fdt** | False | 2.027e-06 | **3.235e-06** ← epsilon only |
| relativistic | T | legacy | **True** | 0 | 1.637 |
| relativistic | T | **fdt** | False | 7.780e-01 | **1.637** ← THE BUG |

**Provable guarantees (all tested):**
- `noise_mode="legacy"` — **bit-identical in 6/6 cells**, every kinetic mode, tied and untied. Exact, because `m_eff` is never read on that branch.
- `fdt` + `newtonian_identity` — **bit-identical 2/2** (`M_eff = 1` either way).
- `fdt` + **untied** learned/relativistic — **not** bit-identical; shifts by exactly the `+1e-6` epsilon (`3.2e-06` relative). Pinned by `test_fdt_untied_learned_changes_only_by_epsilon` (`rel < 1e-5`).
- `fdt` + **tied** — changes by `1.64` relative. This is the fix.

### 2.3 Gibbs restored — the momentum-variance instrument (`v2_gibbs.py`)
Used the analyst's instrument (`s5b_fdt_bug_direct`), **not** `D_θ(θ₀)` (their documented negative: the coset angle wanders within a block and washes out the anisotropy).

For separable `H` the momentum marginal of `exp(−H/T)` is exactly Gaussian with `Var(p_i) = effective_inertia()_i · T`, **independent of V** — so the channel temperature ratio must be exactly 1.

Tied model, `M_dyn = [0.82594, 0.82594, 0.79814]`, `M_noise_old = [0.31326, 1.70141, 0.79814]`:

| arm | `Var(p)` | `Var(p)/(M_dyn·T)` | `T_eff,0/T_eff,1` | dev. from Gibbs |
|---|---|---|---|---|
| **shipped (pre-fix)** | `[0.1569, 0.8515, 0.3994]` | `[0.380, 2.062, 1.001]` | **0.18426** | **81.6 %** |
| *predicted* `M_noise,0/M_noise,1` | — | — | *0.18412* | (obs matches to **0.08 %**) |
| **fixed** | `[0.4136, 0.4133, 0.3994]` | `[1.0016, 1.0009, 1.0009]` | **1.00074** | **0.07 %** |

The **untied coordinate 2 is unaffected in both arms** (`1.0009`) — independent confirmation of the blast radius. My anisotropy (81.6 %) is far larger than the analyst's 8.4 % because I deliberately exaggerated `log_mass` spread; trained checkpoints have a narrow spread. Same mechanism, same predicted ratio.

### 2.4 Tests (`tests/test_langevin_fdt.py`, 5 → 22 tests)
- **Updated** `test_effective_mass_per_kinetic_mode` — it *pinned the buggy semantics* (`effective_mass() == softplus(log_mass)`, no epsilon). Now asserts the H-consistent `M + 1e-6`.
- **Added** `test_effective_mass_is_exactly_effective_inertia` (6 cells, `jnp.array_equal`)
- **Added** `test_effective_mass_applies_channel_tie`
- **Added** `test_legacy_noise_bit_identical_after_fix` (6 cells, bit-compat oracle vs. verbatim pre-fix impl)
- **Added** `test_fdt_untied_identity_bit_identical_after_fix` (2 cells)
- **Added** `test_fdt_untied_learned_changes_only_by_epsilon` (pins the epsilon)
- **Added** `test_fdt_tied_channel_restores_gibbs` — **the regression**: asserts `Var(p_i) ≈ M_dyn,i·T` (rtol 5 %) and `|T_eff,0/T_eff,1 − 1| < 0.02`; *and* asserts the pre-fix path both breaks equipartition (`>10 %`) and tracks `M_noise,0/M_noise,1` to 5 %.

**Commands & real output:**
```
uv run --no-sync pytest tests/test_langevin_fdt.py -q   → 22 passed in 11.89s
uv run --no-sync pytest -q            (at clean d6f8bac) → 217 passed, 1 skipped in 235.28s
uv run --no-sync ruff check chlu/core/chlu_unit.py tests/test_langevin_fdt.py → All checks passed!
```
The 1 skip is **pre-existing and unrelated**: `tests/test_industrial_loaders.py:10: loaders need the eval extra`. Suite on `main` = 200 tests (that file had 5); now 217. ✅ ≥200.

**Formatting note (deliberate):** `ruff format` wants to reformat `chlu/core/chlu_unit.py`, but I verified **`main`'s version is already format-unclean** (`ruff format --check` on `git show main:…` → "Would reformat") and every proposed hunk lands on **pre-existing** code (`potential_type` field, `step()`, the `langevin_step` call, `stochastic_rollout`'s ValueError) — none on my edit. Running it would sweep unrelated reformatting into my commit (§3.3 violation, w4 precedent). Left alone; `ruff check` passes. `tests/test_langevin_fdt.py` *was* clean on main, so I formatted it (my additions only).

---

## 3. AUDIT (task item 3) — **the important part**

**Question:** does any shipped/reported result co-occur `langevin_noise="fdt"` **AND** `tie_channel_mass=True`?

**Answer: NO. Zero contaminated results. No published/claimed number needs an asterisk.** Four independent lines of evidence:

| # | check | command | result |
|---|---|---|---|
| A | repo default | `grep -n langevin_noise chlu/config.py` | `TrainingConfig.langevin_noise: str = "legacy"` — the **only** definition |
| B | project overrides | `grep -rn langevin_noise projects/` | **none** (0 hits across all 14 project dirs) |
| C | project overrides | `grep -rn tie_channel_mass projects/` | **none** (0 hits) |
| D | CLI surface | `grep -rn langevin_noise chlu/cli/` | **none** — no CLI flag exposes it; only settable via `config.yaml` (see B) or a direct Python kwarg to `train_chlu`/`train_generative` |

**The decisive structural argument.** `tie_channel_mass=True` appears as a default in **exactly one** dataclass — `ExperimentDConfig` (`config.py:280`). And:

```
grep -rn "langevin|noise_mode|fdt|stochastic" chlu/experiments/exp_d_goldstone.py
  → (no matches)
```

**Exp-D never invokes Langevin/`stochastic_step` at all.** Conversely, every consumer of `fdt`-capable noise (`exp_c_dreaming.py:341`, `train.py:356`, `train_generative.py:193`) reads `config.training.langevin_noise`, which is `"legacy"` everywhere, and none of them constructs a tied model. The two conditions are **structurally disjoint in the shipped code.**

The one place both conditions *were* met is the analyst's own `t-lever-forgetting` study (`designed150_s{42..46}`, `fdt` + tied) — and they neutralized it with the pytree-level `common.retie(model)` workaround, verified `H`-bit-identical. **Their quantitative results stand unmodified** and are, in fact, now reproducible without `retie` on this branch. Also checked: `exp_v1_gate` / `exp_paid_access` (the `transforms.effective_mass` consumers) build **untied** models and use no Langevin.

---

## 4. Sibling bug found — NOT fixed (out of scope), one-line patch supplied

`chlu/core/transforms.py:35` defines a **separate free function** `effective_mass(model)` used by the mass-weighted squeeze (`exp_v1_gate.py:520`, `exp_paid_access.py`, `tests/test_transforms.py`, `tests/test_v1_gate.py`). It has the **same tie bug** (raw `jax.nn.softplus(model.log_mass)`, no `tie_channel_mass`), though it *does* include the `+1e-6`.

Its docstring promises it "matches the unit's true velocity response `dq/dt = grad_p T` at small p" — **false on a tied model.**

- **Live contamination: none.** Both consumers construct untied CHLUs and never use Langevin (verified by grep, §3).
- **Latent trap:** the moment anyone squeezes a tied checkpoint, `mass_weighted_squeeze` reframes with the wrong inertia.
- I did **not** edit it: `transforms.py` is not named in my task, and §3.3 forbids opportunistic edits to shared files a concurrent agent may hold.

**Recommended patch (assign to a follow-up):**
```python
def effective_mass(model) -> jnp.ndarray:
    return model.effective_inertia()   # tie-aware, epsilon-consistent
```
This is bit-identical for untied models (both already include `+1e-6`) — a *strictly* safe change, unlike the `chlu_unit` one.

---

## 5. Git footprint

- **Branch:** `agent/experiment-engineer/fix-pack-5` (off local `main` @ `9bc2cf7`). **Not pushed. No PR. Left for review.**
- **Commit:** `d6f8bac` — `[experiment-engineer] fix FDT noise inertia: effective_mass() -> effective_inertia()`
- **Files touched (2, both named in the task):**
  - `chlu/core/chlu_unit.py` — 1 hunk, `effective_mass()` body + docstring (+15 / −24)
  - `tests/test_langevin_fdt.py` — +237 / −5 (5 → 22 tests)
- **Rebase:** `git rebase main` → "up to date" (base unmoved). `origin/main` (`40c2f31`) correctly avoided.
- **Conflicts:** none. **Working tree clean** at commit time; no foreign changes present; no worktree needed (only `main` checked out, `git worktree list` = 1 entry).
- **Scratch (gitignored):** `.claude/scratch/fix-pack-5/{v1_scope,v2_gibbs,v3_lattice}.py`

---

## 6. Open questions / follow-ups / risks

1. **The `+1e-6` on untied `fdt` is a real (if microscopic) behavior change.** Any *existing* `fdt`+untied result changes by `~3e-06` relative in the noise scale — far below MC error, but it means "untied is bit-identical" is false. Since no shipped result uses `fdt` at all (§3), this is moot in practice. Flagging because the task asserted the opposite.
2. **`transforms.effective_mass` sibling bug** (§4) — needs a task; trivial and strictly safe.
3. **The `T_φ` build (analyst §8) is now unblocked.** Its "blocking prerequisite: fix §6 first" is satisfied on this branch; `T1` (`T_φ ≡ 0` bit-identical) should be checked against **this** branch, not `main`.
4. **Not tested: trained tied checkpoints.** My regression uses a synthetic tied CHLU with exaggerated `log_mass`. The analyst's 5-seed trained-checkpoint evidence (8.4 %) is the real-world number; I reproduced the *mechanism*, at larger amplitude, and the predicted-ratio agreement (0.08 %) confirms it's the same effect. A cheap confirmation would be re-running `s5b_fdt_bug_direct.py` **without** `retie` on this branch — expect the ratio to move from 0.916…1.019 to ≈1.
5. **`kinetic_mode="newtonian_identity"` is immune** to this bug entirely (`M_eff ≡ 1`). Exp A's default. Worth remembering.

---

## Proposed handover updates (for the Hub)

### §7 Known Issues — **resolve item 9's successor / log the new item as FIXED**

> **BUG (found by `t-lever-forgetting` 2026-07-09, FIXED by `fix-pack-5` 2026-07-10, branch `agent/experiment-engineer/fix-pack-5` @ `d6f8bac`, NOT yet merged).** `CHLU.effective_mass()` returned raw `softplus(log_mass)` — ignoring `tie_channel_mass` (applied by `mass_vector`, hence by `H`/`T`) and omitting the `+1e-6` that `H` inverts. `stochastic_step` built the `noise_mode="fdt"` scale from it, so on a tied checkpoint the noise used a different inertia than the dynamics ⇒ per-channel temperatures ⇒ **no Gibbs invariant**. **Fixed:** `effective_mass()` now delegates to `effective_inertia()`. `CLULattice.effective_mass()` (a *second*, undocumented call site at `lattice.py:429`) inherits the fix.
> **Verified:** tied+fdt `T_eff,0/T_eff,1` `0.18426 → 1.00074`; `Var(p)/(M_dyn·T) = [1.0016, 1.0009, 1.0009]`. **Bit-compat:** `legacy` bit-identical 6/6 (all kinetic modes × tied/untied); `fdt`+`newtonian_identity` bit-identical 2/2; `fdt`+untied learned/relativistic shift by the `+1e-6` epsilon only (`3.2e-06` rel — the *second half of the same bug*, so "untied bit-identical" as originally scoped was unachievable **and** undesirable). Suite 217 passed / 1 pre-existing skip. Regression: `tests/test_langevin_fdt.py::test_fdt_tied_channel_restores_gibbs` (+6 more; file 5 → 22 tests).

### §5 Provenance / §7 — **AUDIT ANSWERED: no result contaminated**

> **No shipped, published, or claimed CHLU number is affected by the FDT/tie bug.** `TrainingConfig.langevin_noise` defaults to `"legacy"` and is overridden **nowhere** — 0 hits in `projects/**`, no CLI flag exists. `tie_channel_mass=True` is a default in **exactly one** dataclass, `ExperimentDConfig`, and **Exp-D never calls `stochastic_step`/Langevin** (0 grep hits for `langevin|noise_mode|fdt|stochastic` in `exp_d_goldstone.py`). The two conditions are structurally disjoint in shipped code. The only `fdt`+tied usage was `t-lever-forgetting`'s own analysis, which pre-empted the bug with the `common.retie` pytree workaround — **its results stand**, and on this branch they reproduce without `retie`.

### §7 — **NEW latent issue to log (unfixed, out of fix-pack-5's scope)**

> **`chlu/core/transforms.py:35` `effective_mass(model)`** (a *separate free function* from the `CHLU` method) has the **same tie bug**: raw `softplus(log_mass)`, no `tie_channel_mass`. It feeds `mass_weighted_squeeze` (`exp_v1_gate.py:520`, `exp_paid_access.py`). **No live contamination** (both consumers build untied models, no Langevin), but it is a trap for any tied-checkpoint boost/squeeze work — i.e. **V1 (Lorentz-boost attention) is exposed the moment it touches a tied model.** Fix is one line and *strictly* bit-identical for untied models: `return model.effective_inertia()`. Assign: `experiment-engineer`.

### §8 / analyst §8.2 — **`T_φ` prerequisite cleared**

> The analyst's "blocking prerequisite: fix §6 first" for the localized temperature field `T_φ(q)` is **satisfied** on `agent/experiment-engineer/fix-pack-5`. Any `T_φ` build should branch off that (or off `main` post-merge), and its acceptance test **T1** (`T_φ ≡ 0` ⇒ bit-identical to current `langevin_step`) must be evaluated **against the fixed baseline**, not `main` — on `main` the tied+fdt baseline is itself non-Gibbs.
