# PREREG — `bprime-c6` (third-party store attribution)

**Written and committed BEFORE any measured run of the probe** (protocol §5 pre-registration rule).
Agent: experiment-engineer. Branch `agent/experiment-engineer/bprime-c6`, base local `main @ d4f56c8`.

⚠ **What was computed before this document, and why it is not a measurement of a registered
quantity.** `chlu.core.memory_potentials.designed_sites` is deterministic and depends on no store,
no write and no read. I evaluated it (pure geometry, no CLU, no coupling) to size the sweep:

| `ball_radius` R | 0.42 | 0.55 | 0.64 | 0.80 | 1.00 (**shipped**) | 1.20 | 1.40 |
|---|---|---|---|---|---|---|---|
| achieved `sep` of the 6 designed sites in the 4-ball (`seed=0`) | 0.571 | 0.748 | 0.870 | 1.088 | **1.3595** | 1.631 | 1.903 |

⭐ **This calibration already contains the task's first reconciliation and it is registered here, not
discovered later:** the shipped `overload/load1x_shipped` cell has an achieved geometric separation of
**`sep = 1.3595`**, not `0.58`. `d_safe_override = 0.58` is an **admission gate** (a refusal radius),
not the achieved spacing. So under the geometric ruler with `s = atom_width = 0.30` the rig C2W3
actually runs sits at **`d/s ≈ 4.53`**, i.e. *at* T5.5's "designed admission gate" row (4.4), **not** at
its `1.9–2.0` "where the gym actually runs" row. **I register that prediction now and will report the
measurement against it.**

---

## 0. What is being measured (the registered quantity)

For a live store of `K` items and a query set `Q`:
* `sel(i)` = the query's **nearest live address** (this is exactly the row a per-slot table selects,
  which is what Prop T5.4 quantifies over); `third(i)` = the **second-nearest** live address.
* Delete item `k` = set that item's own atom group's amplitude to **0** (`LearnedVStore.
  scale_group_amplitude(slot, 0.0)`; `A = amp²`, so the group's contribution to `V_θ` becomes exactly
  0 and nothing else in `V_θ` moves). Re-read the **same** launch points.
* `Δ_third(i, t, ch)` = `‖s_t^{(delete third(i))}(i) − s_t^{full}(i)‖` over the full latent, per channel
  `ch ∈ {q, p}`; `Δ_own` likewise with `sel(i)`.
* **coupling** `κ(t) = mean_i Δ_third(i,t) / Δ_own(i,t)` — dimensionless, and the direct analogue of
  T5.5's `‖∇V_j‖ / own`.
* **gradient ratio** `κ_∇ = mean_i ‖∇V_{third(i)}(q₀_i)‖ / ‖∇V_{sel(i)}(q₀_i)‖`, where `∇V_k` is the
  gradient of the **deleted difference** `V_full − V_{−k}` — the static, dynamics-free form of the same
  quantity, and the one T5.5's table actually predicts.

**Controls (all native to the inherited rig):** blank-store delete control (the same zeroing applied to
an unwritten group of the harness's own blank store) · launch-noise floor (`‖s_t(q₀+δ) − s_t(q₀)‖`,
`δ ~ N(0, σ_q)` on the address block) · the **per-slot matched-bytes table** (§A9.5's object) with the
same row deleted.

**Slot grid (C7):** buffer slots `{0,1,2,3,4,6,8,12,16,20,24,29}` ⇒ integrator steps
`{1,9,17,25,33,49,65,97,129,161,193,233}` ⇒ `t ∈ [0.05, 11.65]` time units at `dt = 0.05`. All inside
C7's `t ∈ [1, 240]` steps.

**Sweep:** `ball_radius R ∈ {0.42, 0.55, 0.64, 0.80, 1.00, 1.20}` on `overload/load1x_shipped`, with
`d_safe_override = 0.58·R` so the **gate-to-geometry ratio is invariant** (0.4266 at every R) and
`R = 1.00` reproduces the shipped cell exactly. `atom_width = 0.30` and `query_sigma = 0.15` fixed.
Seeds `{0, 1, 2}` on every point. **Nothing else is changed.**

---

## 1. Registered predictions

### P1 — the coupling at each declared `d/s` (**how derived**)
T5.5's table is reproducible in closed form, and I reproduce it here **before** using it, so the model
being tested is explicit. For two isotropic Gaussian wells of width `s`, the own-well gradient is
evaluated at `r = σ_q` and the third-party gradient at `d`:

> **`κ_∇(d) = (d/σ_q) · exp(−(d² − σ_q²)/(2s²))`**

Check against T5.5 at its own `σ_q = 0.15, s = 0.35`: `d/s = 1.9 → 0.799` (T5.5: **0.80**) ·
`2.9 → 0.111` (T5.5: **0.111**) · `4.4 → 7.06e-4` (T5.5: **7.0e-4**). **Exact — the formula IS T5.5.**

At **our** rig's `s = atom_width = 0.30`, `σ_q = 0.15`:

| R | `sep` | `d/s` (geometric `d`, `s = 0.30`) | **registered `κ_∇`** |
|---|---|---|---|
| 0.42 | 0.571 | **1.90** | **0.708** |
| 0.55 | 0.748 | 2.49 | 0.242 |
| 0.64 | 0.870 | **2.90** | **0.0981** |
| 0.80 | 1.088 | 3.63 | 1.10e-2 |
| 1.00 | 1.3595 | **4.53** (the shipped rig) | **3.56e-4** |
| 1.20 | 1.631 | 5.44 | 4.7e-6 |

**Do I expect the learned multi-atom store to agree?** *Partially, and I register where I expect it to
fail.* The model is a two-well toy at `p₀ = 0` (theorist §9 item 4), our store has 341 atoms per item
whose written well is **wider and shallower** than a single atom (`_well_fit` exists precisely because
reading `s` off the atom parameters underestimates it). ⇒ I predict the measured `κ_∇` **exceeds** the
registered value at every point (a wider effective `s` ⇒ weaker suppression), by **0.5–2 decades** at
the large-`d/s` end and by **< 0.3 decades** at `d/s = 1.9`. Registered tolerance for "the law holds":
the *shape* is scored, not the offset — see P2.

### P2 — the `exp(−½(d/s)²)` law and its decades
Taking logs of P1's formula, `ln κ_∇ − ln(d/σ_q) = −(d² − σ_q²)/(2s²)` is **linear in `d²`** with slope
`−1/(2s²)`. Registered:
* **the regression is linear in `d²` with R² ≥ 0.95** over the six sweep points;
* the fitted width **`s_fit = (−2·slope)^{−1/2}` lies in `[0.25, 0.60]`** (i.e. between the atom-init
  width and 2× it) — this is an *independent third estimate of the learned well width*, and it is
  registered to be compared with `CluSystem.well_fits()`;
* **decades over the swept span** (`d/s = 1.90 → 5.44`, geometric ruler): registered **5.2 decades** for
  `κ_∇` from the closed form; **measured tolerance ±1.5 decades**, because a wider fitted `s` compresses
  the span. ⛔ **If the measured span disagrees by more than 1.5 decades the exchange rate that prices
  Route 3 is wrong** and that goes in my first 10 lines (task §4 falsifier 2).

### P3 — the dynamical coupling vs the static one
`κ(t)` (slot-Δ ratio) at the **smallest** slots is registered to agree with `κ_∇` **within a factor 3**
(both are the same ballistic quantity at `t → 0`), and to **decrease monotonically in `t`** thereafter
(the own-item Δ saturates at `O(sep)` once the query has fallen into its own well while the third-party
Δ does not) — registered decrease **≥ 3×** from slot 0 to slot 29 at `d/s ≈ 1.9`.

### P4 — the prefactor, WITH the free-fall correction (task §3, and the theorist missed his own bar)
Bare ballistic: `Δq(τ) = (τ²/2M)‖∇V_j(q₀)‖`. Two corrections **both** apply on our rig and I register
their product:
1. **Damping.** The shipped read is `p ← (1−γ)p_s` per step at `γ_address = 0.05`. With `p₀ = 0` and a
   constant force, `Δq(n) = (dt²F/Mγ)[n − (1−γ)(1−(1−γ)ⁿ)/γ]`, so the damped/bare ratio is
   `R_γ(n) = [n − (1−γ)(1−(1−γ)ⁿ)/γ] / (γ n(n+1)/2)` = **0.864 at n = 10**, **0.744 at n = 20**.
2. **Free-fall.** The theorist's own measurement of the residual: **0.61–0.73× at t = 10**, **0.17–0.34×
   at t = 20** (the particle falls toward its own well, so `‖∇V_j‖` along the path is below its value at
   `q₀`).

⇒ **registered measured/bare-ballistic = `0.864 × [0.61, 0.73]` = [0.53, 0.63] at step 10** and
`0.744 × [0.17, 0.34]` = **[0.13, 0.25] at step 20**. Registered **acceptance band, widened for the
learned multi-atom store: [0.35, 0.85] at step 10 and [0.08, 0.40] at step 20.** Scored honestly, and a
miss is reported as a miss (the theorist's bare version missed by 0.61–0.73×).

### P5 — q vs p (C7)
C7: momentum peaks at **1.88 σ_q at t ≈ 10**, ≈ **4×** the position channel. Mechanically, with `p₀ = 0`
and `Δq(n) ≈ n·dt·Δp(n)/2M`, the ratio `Δp/Δq` at step `n` is `≈ 2M/(n·dt)` = **4.0 at n = 10**
(`dt = 0.05`, `M = 1`). ⇒ registered **`Δp/Δq ∈ [1.5, 6]` at step 9 (slot 1)**, and the p-channel Δ
peaks at a **smaller `t`** than the q-channel Δ.

### P6 — the table's Δ (the contrast's definition)
Deleting a **non-selected** row of the §A9.5 per-slot matched-bytes table changes its prediction for
every query that did not select that row by **exactly 0.0** (float equality, `max |Δ| == 0.0`, not
`< tol`). Registered as **exactly 0 by construction (Prop T5.4)**, computed and not assumed. ⛔ This is
not a win and will not be reported as one.

### P7 — the falsifier bar (task §4 item 1)
At `d/s ≈ 1.9` the third-party Δ is registered to **clear the blank-store delete control beyond 2 SE on
3 seeds at ≥ 1 slot** (predicted: at every slot). At the shipped rig (`d/s ≈ 4.53`) I register that it
**does NOT clear the launch-noise floor** (predicted `κ_∇ ≈ 3.6e-4`), and that this is the finding, not
a failure.

### P8 — the fitted well width (the bonus, `doctrine-repairs` OQ-C)
`CluSystem.well_fits()` on the shipped learned `V_θ` is registered to return `s_fit ∈ [0.30, 0.80]`
(≥ the atom-init width; the module's own docstring says reading `s` off the atoms underestimates it).
⛔ Registered as a **bracket-breaking measurement**: if `s_fit` differs from `atom_width` by more than
1.5×, then **T5.5's `d/s` x-axis is mis-scaled** and every `d/s` statement built on `atom_init_width`
moves. Both x-axis conventions are reported side by side either way.

---

## 2. D3 (OQ-A) — re-locating `B = 0.33`

**How `B` was located** (`doctrine-repairs` §4.3): on the grid `R = 1.3, σ_q = 0.24, pay = 0.24`,
`w_addr ∈ {0.20, 0.30, 0.40, 0.50}` the relative certificate deficits are `0 / 11.6 / 32.9 / 54.2 %` and
`ρ_ex = D/U` is `0.127 / 0.183 / 0.294 / 7.94`; the last is `D > U`, the **R1 estimator breaking down**,
so the budget was set at the last surviving point, `B = 0.33`. `U` there is
`P[‖q_a − c_i‖ > sep/2]` — the **broken ruler**.

**Registered predictions.**
* **PD3a.** With `s_max = w_addr · 1.3868` (the harness's own max anisotropy factor) and `sep = 1.3`,
  the four configs sit at `s_max/sep = 0.213 / 0.320 / 0.427 / 0.533`. R1's corrected proxy is valid
  only in `s/sep ∈ [0.15, 0.30]` ⇒ **registered: 3 of the 4 configs, including BOTH configs that
  located the edge, are OUTSIDE the corrected proxy's own validity domain.** ⇒ `B` **cannot be
  re-located with the corrected proxy**; it must be re-located with the **measured** (32-direction
  bisection) capture radius, which is SC-6's leg and is valid everywhere.
* **PD3b.** With the **measured** radius, `U` rises (the true basin is smaller than `sep/2` for wide
  wells) and the `w = 0.50` breakdown **disappears**: registered `D ≤ U` at all four configs, i.e.
  `ρ_ex(measured) < 1` at `w = 0.50` (vs 7.94 with `sep/2`).
* **PD3c.** ⇒ **registered corrected `B`: the edge is NOT located by this grid under the correct ruler.**
  The measured statement will be "`B ≥ 0.542` is unrefuted on this grid" rather than "`B = 0.33`".
  ⛔ I do **not** change the shipped default (`SC3_BUDGET_B = 0.33`); this is a measurement and a Hub
  item.
* **PD3d.** The `λ_min` price is unchanged by the ruler (it is not computed from a radius): registered
  `λ_min = 12.45 / 5.59 / 3.19 / 2.08` reproduced to 3 significant figures.

---

## 3. ⛔ Declared NOT-RUN before the fact

* **OQ-2** (a learned `p₀` steering the path toward non-selected wells) — the **only** pre-registered
  revival trigger for inference-read claims, and **wave-boundary only** (charter §A14.1). Running it
  inside a wave would launder a revival past its own gate. **NOT RUN.**
* No stage-2 object of any kind (no slotted write objective, no `allocate`, no γ-as-selector).
* No re-measurement of the stage-1 attribution curve, the §A9.5 table launder verdict, the 2×2, or the
  byte-floor theorem.
* No shipped default is changed anywhere (`SC3_BUDGET_B`, `FAMILY_DEFAULTS`, `CluSystemConfig`).

*Committed before the first measured run.*
