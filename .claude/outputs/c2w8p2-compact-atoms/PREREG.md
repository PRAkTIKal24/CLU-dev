# PREREG — `c2w8p2-compact-atoms` (ARM A, C2W8 pass 2)

**Filed 2026-08-06 by experiment-engineer (wt1, branch `agent/experiment-engineer/c2w8p2-compact-atoms`,
worktree `../CHLU-c2w8a`, base `main @ 80d7d4b`) — BEFORE any harness or pytest cell that measures a
number below was executed.** Protocol §5 pre-registration rule.

Binding above this file: `ERRATA-C2W8-PASS2.md` §1 (four Head rulings) and `PREREG-C2W8-PASS2.md`
(the Hub's §5 P1–P6). ⛔ **I do not re-derive, re-tune or replace P1–P5.** Everything below is either
(a) a restatement of the Hub's registered prior that I am measured against, or (b) a NEW prediction
about the *mechanism* that the Hub's prereg does not cover, registered here before it is measured.

---

## 0. What the arm is (declared before it runs)

**Mechanism:** the atom influence profile of `AtomDictionaryPotential` becomes selectable and, in the
arm, **compact** — exactly zero beyond a support radius `R = cutoff · s` — with `s` **co-scaled to the
MEASURED key spacing** `geometry.median_nn_task1` of that seed's own run (never hardcoded; recovered
inside the cell as `d_safe / d_safe_frac`).

Primary kernel: **Wendland C²**, `φ(t) = (1−t)⁴(1+4t)` for `t = r/R ≤ 1`, `0` for `t > 1`.
`φ(0) = 1`, `φ'(0) = 0`, `φ(1) = φ'(1) = φ''(1) = 0` ⇒ the potential is **C² everywhere**, the write
gradient is **continuous** and **exactly zero beyond R** (no sigmoid tail — the K2 lesson applied one
level down). Secondary kernel shipped for contrast: **truncated Gaussian**, C⁰ compact (value
continuous, gradient *discontinuous* at R).

## 1. K7 — the capture instrument, two-sided, pytest-asserted (measured FIRST, before any arm number)

`chlu/core/well_lifecycle.py` and `chlu/core/soft_certificate.py` are **read-only**; K7 asserts, it
does not repair.

| leg | construction | **predicted, registered here** | declared tolerance |
|---|---|---|---|
| **K7-1 synthetic** | analytic `relax_fn`: returns the site for `‖x−z‖ ≤ R_true = 0.37`, else pushes 10× away. `r_hi = 1`, `steps = 12` | `capture_radius = 0.37` | within one bisection cell, `r_hi/2^steps = 2.44e-4`, i.e. `[0.3695, 0.3700]` |
| **K7-2 synthetic negative** | `relax_fn(x) = z + 5(x−z)` (escapes everywhere) | `capture_radius = 0.0` | **exactly** 0.0 |
| **K7-3 planted store, positive** | real `CluSystem` (`addr_dim=2, payload_dim=1, capacity=2`, shipped Gaussian atoms), two planted wells at `(±0.30, 0)`, `depth=1.0`, `width=0.15`, unused groups flattened; capture measured at site 1 with `n_dirs=64`, `steps=10`, `tol=σ_q=0.15` | **`capture_radius = 0.30`** — half the site separation. *Derivation:* the two wells are identical and symmetric about the origin, where the confinement bowl is also centred, so the separatrix is **exactly** the plane `x = 0`, at distance `a/2 = 0.30` from the site. `a/(2s) = 2 > 1` ⇒ two distinct minima (not a merged one). Perpendicular/outward directions have a strictly larger edge (force balance `D(r/s²)e^{−r²/2s²} = 2α(|z|+r)` gives ≈0.75), so the min over directions is set by the separatrix; finite direction sampling can only push the reading **up** (`0.30/max|u_x|`), inertial overshoot under `γ_read` only **down**. | **two-sided `[0.20, 0.40]`** (0.30 × [0.67, 1.33]) |
| **K7-4 planted flat, negative** | same rig, one planted item at `‖z‖ = 0.6` with `depth = 1e-9`, unused groups flattened ⇒ `V = α‖q‖²` only | `capture_radius = 0.0` **exactly** (every relaxation runs to the bowl minimum at the origin, 0.6 away > `tol`) | **exactly** 0.0 |
| **K7-5 declared instrument caveat** | same flat store but planted **at the origin** | `capture_radius ≥ 0.9 · r_hi` — a **FALSE POSITIVE**: on a flat landscape a site that coincides with the confinement minimum captures everything. Registered as a **known instrument limitation**, asserted so it cannot be discovered late; benign in the census only because real `φ`-sites sit at ‖z‖ ≈ 0.5–1.0. | ≥ 0.9 |

⛔ Until K7-1…K7-4 are green, no arm number is reported.

## 2. K6 — OFF is bit-identical AND parameter-count-identical

`atom_kernel = "gaussian"` (default) must reproduce `main @ 80d7d4b` **bitwise**: asserted by
comparing `AtomDictionaryPotential.__call__` against the literal pre-change expression
(`-Σ A_j exp(−d²/2s²) + α‖q‖²`) with `np.array_equal` on the raw values, and by asserting the
inexact-array leaf count/size is identical across all three kernels and across
`atom_site_local_init` on/off (the new fields are `eqx.field(static=True)` / plain floats and carry
**zero** parameters). Predicted: **exact equality, 0 parameter delta.**

## 3. NEW mechanism prediction (not covered by the Hub's P1–P6) — registered before measuring

**M1 — compact atoms on the SHIPPED scattered init are exactly dead.**
`ERRATA-C2W8.md` §3 measured the distance from a unit-norm site to the *nearest of all* 8 192 atoms at
`addr_dim = 8`: **0.738**. The arm's support radius is `R = cutoff·s = 2.5 × (0.5 × 0.1407) = 0.176`
(seed 0), i.e. **4.2× smaller than the nearest atom**. Every atom's profile — and therefore every
partial derivative w.r.t. `amp`, `centers`, `log_width` — is **identically zero** over the whole
write-objective sample cloud (`σ_addr = 0.25` around the site).
⇒ **Predicted: fitted own-atom depth exactly 0.000 on every well, `G-CAP = 0/16`, self-probe
`decode = chance`, on every seed. P(dead as described) = 0.95.**
⇒ **Consequence registered in advance:** the co-scaled compact kernel is *only* runnable together
with a **site-local atom initialisation at admission time** (the group's atoms re-drawn in a ball of
radius `≈ s` around the item's own address the moment the slot is allocated). This is the N98
localized init (`atom_local_radius`, already shipped for `AtomDictionaryPotential`) moved from *build
time* to *admission time*, which is the only time a `φ`-addressed stream knows its address
(`ERRATA-C2W8.md` §3 records precisely this as the blocker). It is an **initialisation of atom
parameters**, not a constraint on the attractor: the write objective is already handed the target site
`c_i` (`DesignFreedomPotential` docstring, honesty note 2), the settled point stays free, basins stay
free to interact, and **nothing pins, snaps or regularizes the attractor toward `φ(item)`.** It is
flagged separately (`atom_site_local_init`) so the Hub can score the two levers apart, and it is
raised to the Hub in the report's first 10 lines as a compliance question.

**M2 — the arm's own/foreign diagnostic is measured by the FROZEN census, which hard-codes a Gaussian
kernel** (`well_lifecycle.own_foreign_site_depth`, lines 114–121). Under a compact kernel that
estimator is *kernel-mismatched* (it over-reads both legs by the Gaussian tail). Predicted direction:
the frozen estimator reads **foreign ≥ the true compact foreign** on every well. ⛔ I do not modify the
frozen file; this is a reconciliation item for the Hub, and own/foreign remains a **diagnostic, never
a target, never a gate leg**.

## 4. What I am measured against (the Hub's numbers, restated, NOT re-derived)

| # | quantity | arm A registered prior (Hub) |
|---|---|---|
| P1 | `capture_radius > 0` fraction | 0.35–0.75; P(majority) = 0.45 |
| P2 | self-probe `decode` (chance 0.0625) | 0.08–0.20; P(> chance @2 SE) = 0.50 |
| P3 | median `site_drift` vs key spacing ≈ 0.14 | 0.05–0.15; P(< spacing) = 0.55 |
| P4 | all three legs, same arm, ≥3 seeds | 0.35 |
| P5 | own/foreign (diagnostic) | foreign > own on < 24/48 |

## 5. Config selection protocol (declared before the census runs)

Kernel form and `cutoff = 2.5` are **fixed by theory, not swept**. The two co-scaling numbers
(`atom_width_frac_spacing ∈ {0.5, 1.0}`, `site_local_radius_frac ∈ {1.0}` in units of the resulting
`s`) are chosen on **pilot seeds 7 and 8 — disjoint from the census seeds 0/1/2** — on G-CAP and
G-DEC, and the choice is declared in the report. ⛔ No selection on own/foreign, and the census seeds
are run once at the selected configuration.

## 6. Declared NOT-RUNs (never reported as nulls)

merge · prune · depth restoration · any §2.7 claim cell · the factored store · I2 · cross-stream ·
wormholes/learned p₀ · CSF3 · any tier-ii / full-CLU / I2 verdict · any performance claim
(the pass-2 gate is retrievability and is **byte-blind**, ERRATA §1 Q3).
