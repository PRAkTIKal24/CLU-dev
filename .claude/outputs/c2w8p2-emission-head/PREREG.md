# PREREG — `c2w8p2-emission-head` (C2W8 pass 2, ARM B)

**Filed 2026-08-06 by `experiment-engineer` (wt2), BEFORE any harness cell — K7 included — was run.**
Base: `main @ 80d7d4b`, branch `c2w8p2-emission-head`, worktree `../CHLU-c2w8b`.
Binding: `PREREG-C2W8-PASS2.md` §3/§4/§5 and `ERRATA-C2W8-PASS2.md` §1 (implemented, **not**
re-derived). This file registers only what *this spoke* must predict in advance: **K7's numbers**,
which are a measured radius and a measured ratio, i.e. exactly the objects protocol §5's
pre-registration rule governs.

---

## 1. K7 — the capture instrument must be proven able to report a POSITIVE

`chlu.core.well_lifecycle.capture_radii` → `chlu.core.soft_certificate.capture_radius`: along each
of `n_dirs` seeded random unit directions, bisect for the largest displacement `r` from the site
whose relaxation returns to within `tol` of the site; report the **minimum over directions**.
Bisection: `lo=0`, `hi=r_hi`, `steps` halvings ⇒ resolution `r_hi · 2^-steps`, and `lo` is a
**lower** bound on the true crossing (never an over-estimate).

All three cells are run on a hand-built store (`well_lifecycle.plant_item` +
`flatten_unused_groups`, both read-only this wave), `addr_dim = 2`, `payload_dim = 1` (`dim = 3`),
`r_hi = ball_radius = 1.0`, `steps = 8`, `n_dirs = 16`, `seed = 0`, `tol = query_sigma`.

### K7-a — POSITIVE, saturating (an isolated planted well)

One planted well of depth `D = 1.2`, width `s = 0.25` at `z0 = (0.5, 0.0, 0.1)`; every other atom
flattened. The landscape is then `V = 0.05|q|² − D·exp(−|q−z0|²/2s²)` exactly. Its only stationary
point inside the ball is the well's minimum, displaced from `z0` by
`|δ| ≈ 2·α·s²·|z0| / D = 2(0.05)(0.0625)(0.51)/1.2 = 2.7e-3`, i.e. **three orders below `tol`**.
Every launch inside `r_hi` therefore returns.

> **PREDICTION K7-a: `capture_radius = r_hi·(1 − 2^-steps) = 0.99609375`, exactly (|Δ| < 1e-9).**

### K7-b — NEGATIVE, two-sided (a planted FLAT site)

Same store with the well's depth planted at `0.0` (a flat site at the same location), everything
else flattened ⇒ `V = 0.05|q|²`, whose unique minimum is the origin, `|z0| = 0.5099 > tol = 0.15`.
No launch returns to within `tol` of `z0`, so `lo` never advances.

> **PREDICTION K7-b: `capture_radius = 0.0`, exactly.**

### K7-c — POSITIVE, FINITE and analytically located (two identical planted wells)

Two identical wells (`D = 1.2`, `s = 0.25`) planted at `±R·ê₀` (payload channel 0 for both, so the
configuration is exactly reflection-symmetric in coordinate 0). By that exact symmetry the
separatrix between the two basins **is the hyperplane `q₀ = 0`**, so from the site at `+R·ê₀` the
crossing distance along unit direction `u` is `R / (−u₀)` when `u₀ < 0` and unreachable otherwise.
The directions are deterministic (`np.random.default_rng(0)`, 16 rows, dim 3, row-normalised);
computed **before running**: `max_k(−u_{k,0}) = 0.9677334093782249`.

> **PREDICTIONS K7-c:**
> * `R = 0.30` ⇒ `capture_radius = 0.30 / 0.96773 = 0.31000` (point prediction)
> * `R = 0.60` ⇒ `capture_radius = 0.60 / 0.96773 = 0.62001` (point prediction)
> * **ratio = exactly 2.000** (the instrument recovers the *planted geometry*, not a constant)
>
> **Registered tolerances** (chosen before measuring, and stated as the assertions that ship):
> point predictions to **±25 %** — the slack is the bisection resolution (`3.9e-3`, one-sided
> downward) plus finite-horizon damped relaxation (a launch just inside the separatrix is at rest,
> rolls in, and may take longer than `read_steps` to arrive within `tol`, which biases the measured
> radius **down**, never up); ratio in **[1.6, 2.4]**; and strictly `0 < r < 0.99609375` on both
> (i.e. finite — not the saturating K7-a value).

**What each outcome means.** K7-a green + K7-b green = the instrument is two-sided. K7-c green =
it recovers a *planted geometry*, not merely a sign. ⛔ Until all three are green, a
majority-positive G-CAP from either arm is *"an untested instrument agreeing with us"* — the exact
way pass 1 went wrong — and no arm number counts.

**Falsifier.** If K7-a returns 0, or K7-b returns > 0, or K7-c's ratio falls outside [1.6, 2.4],
the instrument is not fit for a positive verdict and **both arms' G-CAP legs are void**, which is a
finding about pass 2's gate, not about arm B.

---

## 2. The arm's predictions — the Hub's, echoed, NOT re-derived

`PREREG-C2W8-PASS2.md` §5 registers these for arm B and this spoke implements them as filed:

| # | quantity | arm B registered prior |
|---|---|---|
| P1 | G-CAP: fraction of wells with `capture_radius > 0` | **0.60 – 0.95**; P(clears majority) = **0.70** |
| P2 | G-DEC: self-probe `decode` (chance 0.0625) | **0.15 – 0.50**; P(> chance, 2 SE) = **0.70** |
| P3 | G-DRIFT: median `site_drift` vs key spacing ≈ 0.14 | **≈ 0**; P(< spacing) = **0.85** |
| P4 | all three legs, same arm, ≥ 3 seeds | **0.60** |
| P6 | arm B lands in the **private-well** configuration ⇒ `NO_TIER_II_CLAIM` | **0.85** |

⚠ **P6 is settled by construction, not by measurement, and is declared so here rather than claimed
as a confirmed prediction:** the shipped head emits **one private well per item**
(`wells_per_item = 1`, `vocabulary_shared = False`), so the arm is `NO_TIER_II_CLAIM` before a
single number exists. That is the honest reading of P6 — the outcome was chosen by the build, and
the build chose it because the alternative (a shared vocabulary) is a **declared NOT-RUN** this wave.

---

## 3. This spoke's own registered expectations (arm-specific, added here so they can fail)

| # | quantity | prediction | why |
|---|---|---|---|
| **E1** | `emission_head_bytes` (head parameter bytes, float32) | **≈ 4.5e4 – 5.5e4 B** at `addr_dim=8`, `hidden=64`, `layers=2` | MLP `9→64→64→11`: `(9·64+64) + (64·64+64) + (64·11+11) = 640+4160+715 = 5515` params ⇒ **22 060 B**. *Registered point prediction: 22 060 B.* (The band above was my first, sloppier estimate; the exact count is the prediction that ships.) |
| **E2** | arm-B `clu_total_bytes` vs pass 1's **360 960** at the SAME atom budget | **> 1.0×** (strictly worse) | the store is unchanged (same `n_atoms`) and the head is *added*. ⭐ A head whose parameters exceed the store it replaces would be a finding; at the pass-1 budget the head is ~6 % of the store, so the structural claim must be made at the **min-store** cell instead. |
| **E3** | min-store cell (`atoms_per_item = 1`) `clu_total_bytes` | **< 3 000 B store + ~22 kB head** ⇒ total **≈ 2.2e4 – 2.6e4 B**, i.e. **≈ 14–16× BELOW** pass 1's 360 960 | 16 atoms × (dim 9 + 1 + 1) × 4 B = 704 B + codebook 512 B. |
| **E4** | median `|c_emitted − φ|` (the anti-pin diagnostic) | **> 0.05**, i.e. **> ~35 % of the key spacing (0.14)** and **not ≈ 0** | nothing pins the center; the only φ-coupling is a hinge that is exactly zero once the launch is inside `ρ·s` with `ρ = 2`, `s ∈ [0.15, 0.80]` ⇒ a slack of up to 1.6 in address units. ⛔ If this came out ≈ 0 the arm would have drifted into the prohibition and would not ship. |

---

*Filed before `pytest tests/test_emission_head.py` and before any arm-B census cell was executed.
Corrections, if any, go in a dated block appended to `ERRATA-C2W8-PASS2.md`; this file is not edited.*
