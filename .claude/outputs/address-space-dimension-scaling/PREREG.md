# PREREG — address-space-dimension-scaling (w20)

**Written BEFORE any harness was written or run.** Protocol §5 pre-registration rule:
the acceptance criterion is a measured growth law, so the predicted values and their
derivation are committed here first.

Author: experiment-engineer. Base: `main @ 089cc6e`.

---

## 0. The design being measured (committed before running)

Generalizing the w19 2-D ring to `d` dimensions. Address plane `x ∈ R^d`, payload
channel `y = q[d]` (always launched at `y(0) = 0` — the w19 anti-decoration guard).

```
V(x, y) = c_conf * relu(||x||^2 - R^2)^2                     # flat inside the ball of radius R
        - b * sum_k exp(-||x - c_k||^2 / (2 w^2))            # K item wells
        + 0.5 * kappa * (y - s(x))^2,  s(x) = sum_k a_k exp(-||x - c_k||^2/(2 w^2))
```

Sites `c_k` placed by **farthest-point sampling** in the ball of radius `R` (deterministic
given a seed) — this maximizes the achieved minimum separation, i.e. it is the *best
possible* packing at that K, so the capacity measured is an upper envelope of the design.

Queries: `x_0 = c_k + sigma_q * N(0, I_d)`, `p_0 ~ sigma_p * N(0, I_d)`, `y_0 = p_y(0) = 0`.
Read: tail-25% of `y`, linear codebook read (w19 verbatim). Blank control on every cell.

---

## 1. Derivation of the predicted law

**(a) Achieved packing.** K points farthest-point-sampled in a `d`-ball of radius `R`
achieve minimum pairwise separation, by the volume argument
`K * (Delta/2)^d ~ R^d`, of

> `Delta(d, K) ≈ 2 R K^(-1/d)`.

**(b) The resolution floor.** Two independent requirements for a site to be addressable:

1. **Query-noise resolution (w19-measured).** w19 found the break at
   `sigma_theta / spacing ≈ 0.2`, i.e. sites must be `≳ c * sigma_q` apart with
   **c ≈ 5**. This is the criterion that actually produced the 8-item ceiling.
2. **Basin distinctness.** Two Gaussian wells of width `w` merge into one basin unless
   separated by `≳ 2 * r_cap`, where `r_cap` is the measured **capture radius** (the
   largest launch offset from a site that still settles back at that site).

⇒ resolution floor `Delta_req = max(c * sigma_q, 2 * r_cap)`.

**(c) The predicted capacity law.** Setting `Delta(d, K_max) = Delta_req`:

> **`K_max(d) = A^d`  with  `A = 2R / max(c * sigma_q, 2 * r_cap)`**

**Exponential in `d`.** With the planned landscape (`R = 1`, `sigma_q = 0.15`, `c = 5`)
the query-noise branch gives `A = 2 / 0.75 = 2.67`.

## 2. Point predictions (primary hypothesis H1)

| d | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| **predicted `K_max` (A = 2.67)** | **7.1** | 19 | 51 | 360 | 2.6e3 | 1.3e5 | 6.5e6 |

Registered tolerance: `A` measured by log-log fit of `K_max` vs `d` lands in
**`A ∈ [2.0, 3.5]`**, and `log K_max` is **linear in `d`** with R² ≥ 0.95.

**Consistency check I commit to in advance:** the d=2 prediction (7.1, ball geometry)
must reproduce the w19 ring ceiling of **8** (`2*pi*R/(c*sigma) = 8.4`) to within a
factor 1.5. If the d=2 cell of the new harness does not land at 6–12, the harness
disagrees with w19 and *the harness is wrong*, not the physics.

## 3. Competing hypotheses, registered so they can win

- **H0 (the fear): `K_max` is flat at ≈ 8 regardless of `d`.** CLU stores a handful of
  items; the associative-memory framing dies; fallback ICLR positioning.
  **Registered as REJECTED by H1.** If H0 survives, it is the wave's headline and I
  report it as such without redesigning the landscape to escape it.
- **H2 (the task's stated packing bound `(1+2R/w)^d`) OVERESTIMATES the measurement.**
  It counts geometric room and ignores query noise. Registered prediction: measured
  `K_max` is **strictly below** `(1+2R/w)^d` at every `d`, with the gap factor
  **growing with `d`** (both are exponential, but the bases differ:
  `2R/Delta_req < 1 + 2R/w` whenever `Delta_req > w`, and the ratio of bases is
  raised to the `d`). Under my planned constants I predict a gap of ≥ 5x already at
  d=4 and ≥ 100x at d=8.
  ⇒ **The packing bound is an upper bound, not the law.** If the measurement instead
  matches `(1+2R/w)^d` tightly, my resolution-floor derivation is wrong.

## 4. Item 2 — basin-width sweep at fixed `d` (registered)

Sweeping `w` at fixed `d` and fixed `sigma_q`, I predict `K_max` does **NOT** follow
`(1+2R/w)^d` over the whole range. Registered form:

> `K_max(w) = (2R / max(2w, c*sigma_q))^d`

i.e. **a plateau for `w < c*sigma_q/2 = 0.375`** (query-noise-limited: `K_max`
independent of `w`), crossing over to a power-law fall `∝ w^(-d)` for larger `w`.
Registered discriminator: at fixed d, `K_max(w=0.15) / K_max(w=0.30)` is
**≈ 1.0 (plateau)**, not the `(1+2R/w)^d` prediction of ≈ 3.6 at d=2 / ≈ 47 at d=4.

## 5. Item 3 — regime assignment (registered)

The three regimes (theorist): 1 barrier-protected (selectivity ~1.00) · 2 washboard
death zone (selectivity ~0.49) · 3 continuum register (selectivity recovers ~0.96).

Registered: **designed ball-packing shows NO death zone at any `d`.** Regime 2 is a
property of the 1-D ring washboard (a single periodic azimuthal coordinate with residual
barriers); an isotropic `d`-ball of Gaussian wells has no such coherent residual force.
Prediction: selectivity is **monotone non-increasing** in `K` at every `d` — regime 1 up
to `K_max`, then a smooth decay, **no non-monotone recovery**. If a death zone appears
at higher `d` for designed structure, the three-regime picture is falsified and that is
the headline (task Item 3 says so explicitly).

## 6. Item 4 — dissipation (registered)

w19: durability 1.000 at `gamma > 0`, 0.813 at `gamma = 0`.

Registered: (i) the `gamma = 0` degradation **persists at every `d`**; (ii) it gets
**worse with `d`** — at `gamma = 0` a conservative particle explores an energy shell
whose escape-direction count grows with `d`, so the fraction settling in the correct
well should fall roughly monotonically in `d` (predict `< 0.6` at `d = 8`, vs 0.813 at
d=2); (iii) the **required `gamma` is approximately `d`-independent** (`gamma_min ≈
0.005–0.01`) because it is set by the settling time `~1/gamma` against the rollout
length, which has no `d` in it. A measured `gamma_min` that grows with `d` falsifies (iii).

## 7. What would make me distrust my own numbers

- A `K_max` that keeps rising simply because the blank control also rises (a
  full-state-read-style leak). **Mitigation: blank control on every cell; any cell
  whose blank exceeds `chance + 0.15` is discarded, not reported.**
- Farthest-point sampling degenerating at large K (candidate pool too small) —
  I will report the *achieved* `Delta_min` per cell and check it tracks `2R K^(-1/d)`.
- Compute truncation masquerading as a ceiling: if a cell fails only because I capped
  K, that is reported as **censored**, not as `K_max`.
