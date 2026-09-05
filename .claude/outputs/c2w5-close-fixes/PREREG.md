# PREREG — c2w5-close-fixes item 4: the `s`-ruler re-measurement of `bprime-c6`'s rig

**Written and committed BEFORE the measuring harness was run** (protocol §5 pre-registration rule).
Only item 4 measures a number; items 1–3 and 5 are re-aggregations / renders / plumbing of already
banked values and carry no prediction.

## 0. What is being measured, and why

Curator G-5 / N224 / charter §A20.5 / handover §7.28: *"the effective-`s` estimator must subtract
`α‖q‖²` or a pure confinement bowl log-fits as a well — measured 1.44× inflation on the cat-test
store; flags `bprime-c6`'s `s = 0.40` (their rig NOT re-measured)."* Every `d/s` in the program rides
on that `s = 0.40`.

**Rig:** `exp_route3_attribution._write_and_query("overload", "load1x_shipped", seed, clu_extra=
_sweep_overrides(1.0))` — the shipped cell (`ball_radius = 1.0`, `d_safe_override = 0.58`), seeds
0/1/2, i.e. bit-for-bit the cell whose row in `c6_summary.json` reads
`sep = 1.3459`, `d = 1.3006`, `s_fit = 0.3625`, `d/s_fit = 3.5902`, `d/s_proxy = 4.3353`.

**Instrument:** `chlu.core.factored_store.effective_s(V, z, confine=α)` — the corrected
(confinement-subtracted) estimator, the exemplar being `tierii-read-fix`'s §7.28 datum
(`s = 0.2879`, `R² = 0.9986`). Control arm: the identical call with `confine = 0.0` on the identical
store. Comparison arm: `CluSystem.well_fits()` (the route c6 published).

`α = CluSystemConfig.confine = 0.05`, and `DesignFreedomPotential.__call__` is
`v_learned(q) + confine·‖q‖²` **exactly** (`chlu/core/memory_potentials.py:639`), so the subtraction
is analytic, not approximate.

## 1. The two competing hypotheses, and how they were derived

**H-A (my primary prediction): `s = 0.40` is CONFIRMED — c6's rig was never contaminated.**
Derivation, from the shipped code rather than from the report:

1. c6's **fitted-well** ruler is `CluSystem.well_fits() -> _well_fit`, and `_well_fit`
   (`chlu/core/clu_system.py:1200`, present since the function's first commit `4cd1a9a`, verified by
   `git log -L`) forms `y = V(z + r u) − α(‖z+ru‖² − ‖z‖²) − V(z)` — i.e. it **already subtracts the
   confinement analytically**. §7.28's flag is therefore about the *estimator family*, and c6 happens
   to sit on the corrected side of it.
2. c6's **law** ruler (`s_implied = 0.3979`, `R² = 0.9953`) is fitted to `κ = ‖∇(V_full − V_{−k})‖ /
   ‖∇(V_full − V_{−sel})‖`. Both numerator and denominator are **differences of two potentials that
   share the identical `α‖q‖²` term**, so the confinement cancels *exactly* in each gradient before
   the ratio is taken. A confinement bowl cannot enter this estimator at all.

⇒ **Predicted:** the corrected `effective_s` at the shipped cell lands in **[0.33, 0.45]** (i.e.
within ±15 % of the cell's own `s_fit = 0.3625` and of the across-radius `0.3979–0.4006`), the
implied `d/s` stays in **[2.9, 3.9]** on the fitted ruler, and the verdict is **CONFIRMED**.

**H-B (the §7.28 alternative, stated so it can win): `s = 0.40` is CORRECTED downward.**
If c6's ruler were contaminated in the way the cat-test store was, the inflation factor measured
there (**1.44×**) transfers: corrected `s ≈ 0.3625 / 1.44 = 0.252` at the shipped cell (or
`0.40 / 1.44 = 0.278` on the across-radius number), and `d/s = 1.3006 / 0.252 = 5.17` — **larger**,
i.e. further inside the designed-gate regime, exactly the direction the task puts on record.

**H-C (mechanism check, must hold under BOTH H-A and H-B).** On the *same* store, dropping the
subtraction (`confine = 0.0`) must inflate the fitted width: predicted ratio
`s(confine=0) / s(confine=α) ≥ 1.10`. If this comes back ≈ 1.00, the store's wells are deep enough
that the bowl is negligible **on this rig**, which is itself the answer to G-5 and is reported as
such rather than as a null.

## 2. Decision rule (fixed in advance)

| outcome | verdict |
|---|---|
| corrected `s ∈ [0.33, 0.45]` at the shipped cell, `R² ≥ 0.95` | **CONFIRMED** (H-A) |
| corrected `s < 0.33` with `R² ≥ 0.95` | **CORRECTED** (H-B direction: `s` smaller, `d/s` larger) |
| corrected `s > 0.45` with `R² ≥ 0.95` | **CORRECTED**, against the recorded direction — flag loudly |
| `R² < 0.95` or `s = nan` on a majority of items | **INSTRUMENT FAILS ON THIS RIG** — report as such |

Everything is reported per seed and per live item; the headline is the **median over items, mean over
seeds**, with the item spread printed, because `well_fits()`'s own published number is a median.

## 3. Declared in advance

- ⛔ No registry, `negative_results.md`, N224 or charter edit — the number is reported only.
- The two estimators (`effective_s`'s log-linear fit of the profile relative to the outermost radius,
  vs `_well_fit`'s non-linear `D(1 − e^{−r²/2s²})` least squares) are **not** algebraically identical;
  a disagreement of a few percent between them is an estimator difference, not a contamination, and
  will be labelled that way. The confinement question is settled by the `confine = 0` control on the
  same store, which is why that arm exists.
- `n = 3` seeds (c6's own count at this radius). Under-powered by design; the deliverable is a ruler,
  not a significance test.
