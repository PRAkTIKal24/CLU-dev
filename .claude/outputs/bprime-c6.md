# bprime-c6 — experiment-engineer report

**Task + acceptance criterion:** measure the one coupling a per-slot matched-bytes table gives exactly 0
for by construction (third-party store attribution, Prop T5.4) — per-slot Δ vs `t`, swept across `d/s`,
≥3 seeds, both x-axis conventions — plus D3 (re-locate `B = 0.33` with a corrected ruler) and D4 (land
the `exp-route3-attribution` CLI hook).
**Status: done** (D1–D4 all run; every number multi-seed; one point needed a bounded seed top-up and
says so).

## ⛔ RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary, in the first 10 lines)
1. ⭐⭐ **T5.5's placement of our own rig on the `d/s` axis is wrong, and the error is 45–52×.** *"The rig
   C2W3 actually runs: `d/s ≈ 1.9–2.0`, coupling `0.69–0.80`, O(1)"* takes `d = d_safe_override = 0.58`
   — an **admission gate** (a refusal radius) — as if it were the achieved spacing. The shipped
   `overload/load1x_shipped` cell's achieved separation is **`sep = 1.346`**, so it runs at **`d/s = 4.34`**
   (atom-width ruler) / **3.72** (fitted-width ruler) with a **measured coupling of `1.53e-2`**. **The rig
   we run is already inside the designed-gate regime.** Sites: charter **§A12** (the C6 bullet) and
   **§A14.1**'s pricing · `bprime-theory` **§T5.5** table + caption and **§8**'s fourth falsifier ·
   handover **§10 C2W3 Decision-Point-2** (*"0.69–0.80 at the rig actually run"*, *"1089× span"*, *"O(1),
   measurable on the existing rig today"*).
2. **The "1089× span" is a span between a label and a gate, not between two geometries we can occupy.**
   Measured across the whole admissibly-writable sweep (`d/s` 1.77 → 5.32) the coupling spans
   **10^2.72 = 525×**; between where we actually run and the designed gate it is **≈10×**.
3. **`atom_init_width` is a 1.33×-low proxy for `s`.** Two *independent* measurements of the learned
   well width agree to **0.7 %**: the coupling law's own slope gives **0.398**, `CluSystem.well_fits`
   gives **0.4006**. Using 0.30 overstates the law's decades by **1.74×**. (This is `doctrine-repairs`
   **OQ-C** / theorist **§9 item 2**, and it is now measured on the shipped learned `V_θ`.)
4. **`B = 0.33`'s outer edge is not merely located by a broken ruler — under the correct ruler it is not
   located at all.** With the measured capture radius the `D ≤ U` estimator **does not break anywhere**
   on the grid that set `B` (`ρ_ex` 7.937 → **0.794** at the config that broke it) ⇒ **`B ≥ 0.542` is
   unrefuted**; and R1's *corrected proxy* is **outside its own validity domain at 3 of the 4 configs**,
   including both that located the edge. ⛔ I changed **no** shipped default (`SC3_BUDGET_B` is still
   0.33). Sites: `soft_certificate.py` `SC3_BUDGET_B`/`BUDGET_DOMAIN` docstrings · `doctrine-repairs`
   §4.3/§4.4 · charter §A9.8.
5. **R1's corrected inradius does not transfer to the anisotropic store the certificate work uses.** Its
   published **14.55×** improvement was measured on an *isotropic* store; on the `naxis`-anisotropic
   gym-like store it is **no better than `sep/2`** at 3 of 4 configs (max |err| 0.098 vs 0.097 · 0.260 vs
   0.248 · 0.280 vs 0.256) and better only at the widest (0.354 vs 0.650).

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echoed before the first result
- **Dial / pillar: none — protocol evidence for TIER i (the audit paper).** ⛔ No dividend, no win, no
  inference-read claim, no revival. **§A9.5's kill stands** and is scoped by §A14.1; `route3-stage2`
  stays parked; nothing here unlocks it.
- **Laundering control:** native to the inherited rig — the **blank-store delete control** (the identical
  zeroing applied to an unwritten group of the harness's own blank store), the **launch-noise floor**,
  and the **per-slot matched-bytes table**, whose third-party Δ is **exactly 0 by construction (Prop
  T5.4)** — computed, reported as the measured zero it is, and **never** as a win.
- **Falsifies:** §4 below. **Does NOT falsify:** a small coupling at the designed gate (predicted) · the
  CLU still losing on the answer channel (already measured; not this task's question).

---

## 1. ⭐ THE `d/s` CURVE — the first screen (D1 + D2)

`overload/load1x_shipped`, seeds {0,1,2} (+ {3,4,5} at the one point whose coverage fell below 3), the
shipped write path verbatim. **Coupling = the third-party Δ ÷ the query's own item's Δ**, i.e. delete the
query's **second-nearest** stored key (the row a per-slot table provably never reads) and divide by
deleting its **nearest** (the row it does read).

| `ball_radius` | **coverage** | `sep` | fitted `s` | **`d/s` (atom_width = 0.30)** | **`d/s` (fitted `s`)** | `d/s` if you use `d_safe` | **measured coupling ± 2 SE** | T5.5 closed form at `s = 0.30` | ⛔ **per-slot TABLE** | `λ_min` |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.42 | **3 / 6** | 0.5481 | 0.482 | **1.77** | 1.10 | 0.81 | **0.814 ± 0.40** | 0.841 | **0 exactly** | 1.26 |
| 0.55 | 3 / 3 | 0.7402 | 0.412 | 2.34 | 1.71 | 1.06 | **0.344 ± 0.18** | 0.345 | **0 exactly** | 2.64 |
| 0.64 | 3 / 3 | 0.8614 | 0.400 | **2.73** | 2.05 | 1.24 | **0.226 ± 0.04** | 0.155 | **0 exactly** | 2.84 |
| 0.80 | 3 / 3 | 1.0767 | 0.385 | 3.43 | 2.68 | 1.55 | **0.0970 ± 0.02** | 0.0233 | **0 exactly** | 3.03 |
| **1.00 (the SHIPPED cell)** | 3 / 3 | **1.3459** | 0.362 | **4.34** | 3.72 | **1.93** | **0.01534 ± 0.006** | 9.83e-4 | **0 exactly** | 3.24 |
| 1.20 | 3 / 3 | 1.6211 | 0.362 | 5.32 | 4.41 | 2.32 | **1.55e-3 ± 2e-3** | 1.55e-5 | **0 exactly** | 3.16 |

*(figure: `results/c6_curve.png`, three panels — the `d/s` curve with both x-axis conventions, the three
declared regimes marked and the table's 0 on it; the per-slot Δ vs `t` curves for q **and** p at every
radius; and the shipped rig against both of its floors. `results/exp_route3_thirdparty.{json,png}` +
`results/thirdparty_topup_R042.json` + `c6_summary.json` are the raw artifacts.)*

### 1.1 ⭐ The law holds on a learned multi-atom store — and it hands back the well width
`ln κ − ln(d/σ_q)` is linear in `d²` with slope `−1/2s²` (this **is** T5.5, see §3 P1):

| fit | slope | **implied `s`** | prefactor | **R²** | decades over the swept span |
|---|---|---|---|---|---|
| static ∇V ratio | −3.158 | **0.3979** | **0.379** | **0.9953** | **2.72** |
| dynamical slot coupling at `t = 1` step | −3.155 | **0.3981** | 0.378 | 0.9952 | 2.72 |
| `CluSystem.well_fits()` (independent) | — | **0.4006** | — | — | — |

⭐ **The exponent transfers exactly; the prefactor is 0.379×; and the width the law implies agrees with
the directly fitted well width to 0.7 %.** T5.5's caveat that `s` is *"a bracket, not a measurement"* is
discharged for this store: **`s = 0.40`, measured two independent ways.**

### 1.2 The two x-axis conventions, and what changes between them
The conclusion's **sign and shape do not move** — the same six measurements, replotted — but the
**span does**: 2.72 decades on the fitted ruler vs **4.73** predicted on the atom-width ruler (1.74×).
And a third "convention", `d_safe/atom_width` (the one T5.5's table caption uses for our rig), is not a
geometry at all: it puts the shipped cell at 1.93 where the store's own achieved spacing puts it at 4.34.
⇒ **falsifier 3 (§4) fires in magnitude, not in sign.**

### 1.3 The curve vs `t`, both channels (D1's actual deliverable)
`|Δ slot| / σ_q`, mean over admissible seeds. **Steps 1…233, inside C7's `t ∈ [1, 240]`.**

| step `t` | 1 | 9 | 17 | 25 | 33 | 49 | 65 | 97 | 129 | 161 | 193 | 233 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **d/s = 1.77, q** | 4.97e-3 | 3.51e-1 | 1.06 | 1.77 | 2.27 | **2.42** | 2.19 | 2.09 | 2.06 | 2.06 | 2.05 | 2.05 |
| **d/s = 1.77, p** | 1.89e-1 | 1.41 | 1.98 | 2.36 | **2.40** | 1.96 | 1.28 | 7.17e-1 | 3.03e-1 | 1.31e-1 | 6.19e-2 | 2.40e-2 |
| **d/s = 4.34, q** (shipped) | 1.04e-4 | 6.99e-3 | 1.95e-2 | 2.99e-2 | 3.56e-2 | 4.45e-2 | 6.89e-2 | 1.19e-1 | 1.79e-1 | 1.40e-1 | 1.47e-1 | **1.95e-1** |
| **d/s = 4.34, p** (shipped) | 3.93e-3 | 2.68e-2 | 3.08e-2 | 3.34e-2 | 3.86e-2 | 4.80e-2 | 4.59e-2 | 9.87e-2 | **1.21e-1** | 7.50e-2 | 9.48e-2 | 4.86e-2 |
| **blank-store delete control** (shipped) | 1.28e-6 | 8.78e-5 | 2.76e-4 | 5.31e-4 | 8.26e-4 | 1.47e-3 | 2.13e-3 | 3.31e-3 | 4.20e-3 | 4.89e-3 | 5.47e-3 | 6.11e-3 |
| ⛔ **per-slot table** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

- **The momentum channel leads and the position channel trails, by exactly the factor the integrator
  forces:** `Δp/Δq = 37.9–38.0` at step 1 (mechanically `2M/(n·dt) = 40`) and **3.78–4.03** at step 9
  (C7's *"≈4× at t ≈ 10"* — **confirmed at its own `t`**). The p-channel peaks **earlier** than the
  q-channel at **6 of 6** radii (33 vs 49 · 49 vs 65 · 65 vs 97 · 97 vs 129 · 129 vs 233 · 233 vs 193).
- **Where the coupling clears its controls** (12 slots per channel, `mean − 2 SE > 0`, ≥3 seeds):

| `d/s` | 1.77 | 2.34 | 2.73 | 3.43 | **4.34 (shipped)** | 5.32 |
|---|---|---|---|---|---|---|
| clears the **blank-store delete control** (q / p) | 12/12 · 12/12 | 12/12 · 12/12 | 12/12 · 12/12 | 12/12 · 12/12 | **8/12 · 11/12** | 3/12 · 4/12 |
| clears the **launch-noise floor** (q / p) | 9/12 · 6/12 | 7/12 · 1/12 | 4/12 · 0/12 | 4/12 · 0/12 | **0/12 · 0/12** | 0/12 · 0/12 |

### 1.4 ⛔ The table's Δ, computed
At **every** slot × **every** dropped row × **every** cell in the sweep: `max |Δ| = 0.0` — **float
equality, not a tolerance**. **0 by construction (Prop T5.4).** The non-vacuous half is asserted beside
it in the test suite: deleting the row the query *did* select moves the same table by a whole payload
level. ⛔ **This is the definition of the contrast, not a win.**

### 1.5 ⛔ The honest cost, stated where it hurts
The coupling is O(1) **only where the store cannot be reliably written**: at `d/s = 1.77` the write is
admissible in **3 of 6 seeds** (`λ_min` = −0.53 / −2.13 / −1.11 on the other three — merged wells), while
every other point is 3/3. **The regime in which a per-slot table is structurally inadequate is the regime
in which our own admission machinery says the store is unsafe.**

---

## 2. D3 (OQ-A) — re-locating `B = 0.33`, with the ruler named

Reproduction first: on `doctrine-repairs` §4.3's **own** grid (`R = 1.3, σ_q = 0.24, pay = 0.24`,
`w_addr ∈ {0.20, 0.30, 0.40, 0.50}`) I reproduce its deficits `0 / 11.6 / 32.9 / 54.2 %` and its
`ρ_ex = 0.127 / 0.183 / 0.294 / 7.937` **digit for digit** (published `7.94`).

| `w_addr` | deficit | `s_max/sep` | `s_eff/sep` | `λ_min` | `D` | `U`(**`sep/2`**) → `ρ_ex` | `U`(**corrected proxy**) → `ρ_ex` | `U`(**measured radius**) → `ρ_ex` | proxy in domain? |
|---|---|---|---|---|---|---|---|---|---|
| 0.20 | 0.0 % | 0.213 | 0.112 | +12.45 | 0.0053 | 0.0420 → 0.127 ✅ | 0.0427 → 0.125 ✅ | 0.0660 → 0.081 ✅ | **yes** |
| 0.30 | 11.6 % | 0.320 | 0.169 | +5.59 | 0.0077 | 0.0420 → 0.183 ✅ | 0.0457 → 0.168 ✅ | 0.1163 → 0.066 ✅ | ⛔ no |
| 0.40 | **32.9 %** | 0.427 | 0.225 | +3.19 | 0.0123 | 0.0420 → 0.294 ✅ | 0.0583 → 0.211 ✅ | 0.1533 → 0.080 ✅ | ⛔ no |
| 0.50 | **54.2 %** | 0.533 | 0.281 | +1.43 | 0.3333 | 0.0420 → **7.937 ⛔ `D>U`** | 0.2127 → **1.567 ⛔** | **0.4197 → 0.794 ✅** | ⛔ no |

⭐ **The measurement, stated as a measurement with its ruler named:**
> **`B = 0.33` was located by the breakdown of an estimator, not by a property of the store.** Under the
> **measured capture radius** (SC-6's own leg, valid everywhere) `D ≤ U` holds at **all four** configs
> including the one that set the edge, so **the edge is not located by this grid at all: `B ≥ 0.542` is
> unrefuted.** Under **R1's corrected proxy** the edge cannot be re-located either — the proxy is
> **outside its own validity domain (`s/sep ∈ [0.15, 0.30]`) at 3 of 4 configs, including both that
> located the edge**, and where it is applied out of domain it still reports `D > U` (1.567).

⛔ **I changed no shipped default.** `SC3_BUDGET_B = 0.33` and `BUDGET_DOMAIN` are untouched;
`chlu/core/monitors.py` was not opened. **This is a Hub item + a `doc-curator` erratum** (reconciliation 4).

**Does it move D2's soft-certificate row?** The soft-certificate row of §1 is defined by `d/s ≈ 2.9`,
which is a *geometry*, and my sweep measures the coupling **at that geometry directly** (`d/s = 2.73`,
coupling **0.226**) rather than inferring it from `B`. So **no, the D2 row does not move** — but the
*mapping* "deficit `B` → `d/s`" that produced the 2.9 label does, and it should be re-derived by whoever
owns `B` before the number is printed as "the soft-certificate coupling".

**Two riders found on the way** (both in the reconciliation list): the corrected proxy is not better than
`sep/2` on the anisotropic store (item 5); and at `w = 0.50` **two of six sites have a measured capture
radius of exactly 0.000 while `λ_min = +1.43 > 0`** — an independent reproduction of **T4.2** (`λ_min > 0`
does not certify a nonempty basin) on the gym-like store.

---

## 3. PREREG SCORECARD (`.claude/outputs/bprime-c6/PREREG.md`, written and timestamped before any measured run)

| # | registered | measured | verdict |
|---|---|---|---|
| **P1** | coupling `0.708 / 0.0981 / 3.56e-4` at `d/s = 1.90/2.90/4.53` (s = 0.30), **and measured > registered at every point, by < 0.3 decades at `d/s ≈ 1.9` and 0.5–2.0 decades at the large end** | ratio measured/registered = **0.97 / 1.46 / 4.2 / 15.6 / 100** across `d/s = 1.77 → 5.32`, i.e. **−0.013 decades at 1.77** and **+2.00 decades at 5.32** | ✅ **CONFIRMED**, at both edges of the registered band |
| **P2a** | the law is linear in `d²`, **R² ≥ 0.95** | **R² = 0.9953** (n = 6) | ✅ |
| **P2b** | implied `s ∈ [0.25, 0.60]` | **0.3979**, and `well_fits` independently **0.4006** (0.7 % apart) | ✅✅ |
| **P2c** | **5.2 ± 1.5 decades** over the swept span | **2.72** | ⛔ **REFUTED**, and the corrected law is verified: the closed form at the *measured* `s = 0.398` predicts **2.72** (to 2 dp). The decades are set by the fitted width; the `atom_init_width` ruler overstates them by **1.74×** |
| **P3a** | dynamical slot coupling at the smallest slot ≈ static ∇V ratio **within 3×** | ratio = **1.0000 / 0.9999 / 1.0000 / 0.9999 / 0.9995 / 1.0052** at the six radii | ✅✅ **within 0.5 %**, not 3× |
| **P3b** | coupling falls **≥ 3×** from slot 0 to slot 29 at `d/s ≈ 1.9` | **1.2×** at `d/s = 1.77`; the decay is monotone in `d/s` (1.2 / 2.7 / 11.8 / 26.0 / **33.1** / 34.6) and exceeds 3× only from `d/s ≥ 2.34` | ⛔ **REFUTED as registered.** Corrected: at small `d/s` **both** deletions saturate at `O(sep)`, so the ratio is flat; the decay is a large-`d/s` phenomenon |
| **P4** | measured / bare-ballistic **[0.35, 0.85] at step ≈10** and **[0.08, 0.40] at step ≈20** (damping 0.864/0.744 × the theorist's free-fall 0.61–0.73 / 0.17–0.34) | **0.853–0.883 at step 9** and **0.713–0.766 at step 17** | ⛔ **MISSED, high, at both.** ⭐ The *damping* half is confirmed to **3 %** (predicted 0.878 at step 9); **the free-fall factor imported from the theorist's two-well toy does not transfer** — the measured free-fall residual is **0.97–1.01 at step 9**, 0.92–0.99 at step 17, and only bites past step 25 (0.80–0.94 → 0.16–0.34 by step 49). Corrected law: **`Δq(n)` = damped ballistic to within 3 % for `n ≤ 17`** |
| **P5** | `Δp/Δq ∈ [1.5, 6]` at step 9; p peaks at a smaller `t` than q | **3.78–4.03** at step 9 (**38.0** at step 1 vs a mechanical 40); p peaks earlier at **6/6** radii | ✅✅ |
| **P6** | the table's third-party Δ is **exactly 0.0** (float equality) | **0.0** at every slot × row × cell; `exactly_zero = True` everywhere | ✅ (**by construction**) |
| **P7a** | at `d/s ≈ 1.9` the Δ clears the blank-store control by 2 SE on ≥3 seeds at **≥1 slot** | **12/12 slots, both channels** | ✅ |
| **P7b** | at the shipped rig the Δ does **not** clear the **launch-noise** floor | **0/12** both channels | ✅ — but it **does** clear the matched blank-store control at **8/12 q, 11/12 p** ⇒ sharper than registered |
| **P8** | `well_fits ∈ [0.30, 0.80]`; **bracket breaks iff `s_fit/atom_width > 1.5`** | 0.362–0.482, mean **0.4006** ⇒ ratio **1.335 < 1.5** | ◐ **the threshold did not fire but its consequence did** — a 1.33× error in `s` is a **1.74×** error in decades, because the law's exponent is `d²/2s²`. My registered threshold was the wrong instrument; the right statement is the sensitivity, not a ratio bar |
| **PD3a** | ≥3 of 4 configs outside the corrected proxy's validity domain, **including both edge-locating ones** | **3 of 4** out of domain; both edge configs out | ✅ |
| **PD3b** | with the measured radius the `w = 0.50` breakdown disappears (`ρ_ex < 1`) | **0.794** (from 7.937) | ✅ |
| **PD3c** | the edge is **not located** by this grid under the correct ruler ⇒ `B ≥ 0.542` unrefuted | exactly that | ✅ |
| **PD3d** | `λ_min = 12.45 / 5.59 / 3.19 / 2.08` reproduced to 3 s.f. | **12.45 / 5.59 / 3.19 / 1.43** | ◐ **my error, not the harness's**: the `2.08` I registered belongs to a *different* config (`R = 1.6, σ = 0.20, w = 0.50`). The three that belong to this grid reproduce exactly |

**Score: 10 ✅ (4 of them sharper than registered) · 2 ◐ · 3 ⛔.** ⭐ **All three refutations produced a
corrected law that was then verified** (the decades from the fitted width; the flat-ratio mechanism at
small `d/s`; the damped-ballistic law without an imported free-fall factor).

---

## 4. FALSIFIERS, ADJUDICATED (task §4)

- ⛔ **"The coupling is not measurable on the shipped rig."** ⭐ **Split verdict, and both halves matter.**
  Against the **matched** control — the identical delete applied to the harness's own blank store — it
  **IS measurable at the shipped rig**: `8/12` q-slots and `11/12` p-slots clear by 2 SE on 3 seeds, and
  at the regime T5.5 *labels* as ours (`d/s ≈ 1.9`) it is `12/12` on both channels. Against the
  **launch-noise floor** named in the task it is **not measurable at the shipped rig** (`0/12`; the Δ is
  10⁻¹–10⁻² of that floor) and is measurable only at `d/s ≤ 2.34` (`9/12` q at 1.77). ⚠ **The two floors
  are different questions** and I report both rather than choosing: the blank-delete control asks *"is
  this Δ the store's?"*, the launch-noise floor asks *"is this Δ bigger than the query's own jitter?"* —
  and the second is a *comparison with a different intervention*, not a null for this one. **The task's
  finding-either-way clause therefore lands on the launch-floor reading: at the weight class we run, the
  escape a table cannot express exists, is real, is the store's, and is two orders below the query
  noise.**
- ⛔ **"The `exp(−½(d/s)²)` law does not hold on a learned store."** **Did not fire — the law holds
  (R² = 0.9953) with the exponent exactly as stated and a 0.379 prefactor.** But **the exchange-rate
  TABLE is wrong at the large-`d/s` end** (up to **100×** low at `d/s = 5.32`) and **its placement of our
  rig is wrong by 45–52×**, both because it uses `atom_init_width` for `s` and `d_safe_override` for `d`.
  ⇒ **reconciliation entries 1–3, in the first 10 lines, as required.**
- ⛔ **"The `s` convention decides the answer."** **Fires in magnitude, not in sign.** Both conventions
  give the same monotone curve and the same qualitative conclusion; the **span** moves by 1.74× and the
  **rig's position on the axis** moves by 2.2× (4.34 vs 1.93). ⇒ every `d/s` statement must name its
  ruler; T5.5's table is a bracket, and now a **measured** one (`s = 0.40 ± 0.7 %`).
- **Did NOT fire / declared non-falsifying:** a small coupling at the designed gate (predicted, and it is
  where we already are) · noisy late-`t` slots (C7 predicted it; the late-slot SEs are the largest in the
  set) · disagreement with a two-well toy at `p₀ = 0` (out of domain by declaration — and the one place
  it bit is P4's free-fall factor, reported as a finding, not a bug).

---

## 5. ⭐ ONE PARAGRAPH FOR THE AUDIT PAPER (`bprime-draft` may lift this verbatim)

> A per-slot matched-bytes table reproduces our slotted read at 37 of 38 slots, so on the answer channel
> the dynamics buy nothing a table cannot. There is exactly one thing such a table structurally **cannot**
> do: its output depends on the store only through the finitely many rows a query selects, so deleting a
> stored item the query did *not* select changes its answer by **exactly zero**. A CLU has no such
> factorisation — its acceleration is `−M⁻¹∇V` and `∇V` sums over every well — so the same deletion moves
> every slot of its trajectory. We measured that coupling, and we measured what it costs to have it. It
> obeys `κ(d) = (d/σ_q)·exp(−(d²−σ_q²)/2s²)` on our learned store with **R² = 0.995** across a 525×
> range, where `s = 0.40` is the store's own fitted well width (two independent estimates, 0.7 % apart).
> It is `O(1)` — 0.81 of the query's own item — only when neighbouring items sit **1.8 well-widths** apart,
> and at that spacing our admission machinery refuses the write in **half** of all seeds because the wells
> have merged (`λ_min < 0`). At the spacing our shipped configuration actually achieves (**4.3 well-widths**)
> the coupling is **1.5e-2**: still the store's — it clears a matched blank-store delete control by 2 SE on
> three seeds at most slots — but **two orders of magnitude below the query's own launch noise**, i.e.
> unusable as a read-out. ⭐ **The one capability a table cannot imitate is exponentially suppressed by the
> very gate that makes the store safe: a store organised well enough to be safe is organised well enough
> to be a table.** That is not a defect of this implementation; it is a design identity, and it is the
> sharpest reason to move the claim from inference-time reads to training-time organisation.

---

## 6. HOW I VERIFIED (commands + observed output)

All as `PYTHONPATH=/Users/user/Desktop/CHLU-c6 /Users/user/Desktop/CHLU/.venv/bin/python …`, cwd in the
worktree (main venv reused, protocol §4 — **JAX 0.9.0**, no worktree `uv sync`).

```
-m pytest tests/test_attribution.py -q --no-cov            -> 20 passed in 4.62s   (was 14)
-m pytest tests/test_route3_attribution.py -q --no-cov     -> 13 passed in 22.34s  (was 7)
ruff check chlu/ tests/                                    -> All checks passed!
-m chlu.experiments.exp_route3_attribution --part thirdparty --seeds 0 1 2
                                                           -> 18 cells, 6.4 min, 17/18 admissible
-m chlu.experiments.exp_route3_attribution --part thirdparty --seeds 3 4 5 --radii 0.42
                                                           -> 3 cells, 1/3 admissible (2 x lambda_min<0)
python d3_budget.py                                        -> reproduces rho_ex 0.127/0.183/0.294/7.937
-m pytest -q --no-cov                                      -> 1073 passed, 0 failed, 1000.21s
```

**⭐ Refactor bit-identity gate (the `_write_and_query` extraction).** A full `--quick` attribution cell
was run on **`main @ d4f56c8`** and on this branch and the two JSON dumps compared:
**every row, the §A8.2 Jacobian and every flag are bit-identical**; the only differing field is
`read_wall_s` (wall-clock). ⇒ the C2W3 stage-1 numbers are unaffected by this branch.

**Determinism.** Seeds {0,1,2} at `R = 0.42` appear in **both** the main sweep and (implicitly, same code
path) the analysis merge; the per-radius aggregates recomputed from the merged artifact reproduce the
runner's own aggregates to the last digit for every 3/3 point.

### 6.1 Full suite
✅ **`1073 passed, 0 failed` in 1000.21 s (16 m 40 s)** on the branch, `ruff check chlu/ tests/` clean.
The arithmetic closes with no unexplained tests: **+12** from my two files (`test_attribution` 14 → 20,
`test_route3_attribution` 7 → 13) ⇒ the base at `d4f56c8` is **1061**. ⚠ That base number is *inferred by
subtraction*, not measured — I did not spend a second 16-minute suite re-running `main` while
`bprime-rivals` holds the box (w26 thermal-cap discipline).

---

## 7. FLAG PROVENANCE (mandatory — every quantitative result above)

| item | value |
|---|---|
| commits | `e38f9d2` (instrument) · `f2f45d8` (probe + sweep) · `be995ca` (CLI hook) |
| base / branch / worktree | local `main` **`d4f56c8`** · `agent/experiment-engineer/bprime-c6` · `../CHLU-c6` |
| env | **main venv reused**, **JAX 0.9.0**, no worktree `uv sync`; `chlu.__file__` verified inside the worktree |
| seeds | **{0, 1, 2}** at every radius; **+{3, 4, 5} at `ball_radius = 0.42` only** — a bounded top-up because that point's coverage was 2/3 (below the 3-seed bar); every seed is reported with its admissibility |
| sd convention | sample sd `ddof=1`; `SE = sd/√n`; "clears" ⇔ `mean − 2·SE > 0`; **coupling and ∇V-ratio estimators are MEDIANS over queries** (the denominator is a per-query gradient magnitude; fixed on `--quick` plumbing runs **before** the sweep), means reported beside them |
| family / arm | `overload/load1x_shipped` — `atoms_per_item=341`, `min_atoms=2046`, `n_offer=capacity=budget=6`, `reference_capacity=6`, `stage_admission=True` |
| **the swept axis** | `ball_radius ∈ {0.42, 0.55, 0.64, 0.80, 1.00, 1.20}` with **`d_safe_override = 0.58·R`** (gate-to-geometry ratio invariant at 0.4266; `R = 1.00` **is** the shipped cell). ⛔ **Nothing else changed.** |
| store / read | `addr_dim=4`, `payload_dim=1`, `dim=6`, `atom_width=0.3`, `atom_depth_init=1e-4`, `atom_init_scale=1.0`, `confine=0.05`; `dt=0.05`, `gamma_address=0.05`, `gamma_read=0.02`, `address_steps=400`, `read_steps=800`, `traj_stride=8`, `query_sigma=0.15`, `kinetic_mode=newtonian_learned` (measured `M = 0.99999905`), `payload_tol=0.1` |
| write | `write_steps=300`, `lr=3e-3`, `weight_decay=1e-4`, `sigma_addr=0.25`, `sigma_pay=0.6`, `margin=0.15`, `barrier=0.2`, `masked_write=True`; **no escalation was needed or spent** |
| **probe-specific** | **`allow_retry=False` on every arm** (so all arms share one buffer length; the shipped rig reports `retries=0` anyway) · slot grid `{0,1,2,3,4,6,8,12,16,20,24,29}` = steps `{1,9,17,25,33,49,65,97,129,161,193,233}` · **deletion = `scale_group_amplitude(slot, 0.0)`** (`A = amp²`, exact removal, nothing else moved) — ⛔ **not** the shipped `evict` path, which *re-draws* the freed group · selection = nearest / second-nearest stored key (**the table's own row selection**, Prop T5.4); agreement with the query's own item = **1.000** at every cell |
| controls | blank-store delete control (`seed+991`, the gym's own blank system, same zeroing on an unwritten group) · launch-noise floor (independent `N(0, σ_q)` re-draw on the address block, `seed+20260731`) · ⛔ per-slot matched-bytes table (**0 by construction**) |
| soft certificate | **OFF** (`soft_certificate=False`, the shipped default) in every cell of the sweep |
| D3 grid | `doctrine-repairs` §4.3's own toy, `R=1.3, σ_q=0.24, pay_scale=0.24, w_addr ∈ {0.20,0.30,0.40,0.50}`, `K=6` ring, `NQ=3000`, `α=0.05`, `γ=0.2`, `dt=0.05`, `NST=700`; measured radius = 32-direction bisection, `tol=2e-3`, launched with payload channels at 0 (the query's own launch) |
| byte ledger | ⛔ **no byte-matched claim is made anywhere in this report.** The store's ratio at this arm is **478.20×** (never-quote as a byte-matched dividend; min ratio anywhere 17.11×). The table launder's Δ is a *structural* zero, not a byte claim |
| φ | identity/embedded, **0 B, identical on every arm** (the gym embeds the address directly) |
| wall-clock | sweep 6.4 min (18 cells) + 1.1 min (3 top-up cells) + D3 15 s |

**Admissible-cell coverage, first-class:** `d/s = 1.77`: **3/6** (excluded: seed 1 `λ_min = −0.5272`,
seed 3 `−2.1265`, seed 4 `−1.1137` — all write-side, none silently filtered). All other radii **3/3**.
Mean endpoint write loss on admissible cells **0.0021–0.0073** against a tolerance of 0.05.

---

## 8. ⛔ DECLARED NOT-RUNs (never to be reported as nulls)

- ⛔⛔ **OQ-2 — the third-party probe with a live particle head (a learned `p₀` steering the path toward
  non-selected wells). NOT RUN.** Reason, verbatim from the task and charter §A14.1: it is the **only**
  pre-registered revival trigger for inference-read claims and it is **wave-boundary only**; running it
  inside a wave would launder a revival past its own gate. ⚠ What my curve *implies* about it, offered as
  evidence for a future Head decision and **not** as a result: the suppression it would have to beat at
  our shipped geometry is a factor **65** (1.53e-2 → O(1)), which in the exponent is `Δ(d²)/2s² = 4.17`,
  i.e. a steering that effectively **halves** the distance to a non-selected well (`d` 1.35 → 0.93). That
  is a large ask for a `p₀` bounded by the query law, and it is cheaper to learn from this number than
  from a build.
- **No stage-2 object of any kind** — no slotted write objective, no `allocate`, no γ-as-selector.
  `route3-stage2.md` stays parked; **§A9.5's kill is not reopened by anything here.**
- **No re-measurement** of the stage-1 attribution curve, the §A9.5 table-launder verdict, the 2×2, the
  byte-floor theorem, or `PREREG-Bprime.md` §7.
- **The `aggregate` / `manifold` families** — not probed. They were 0/3 admissible in C2W3 after their one
  bounded escalation; a third-party coupling measured on an unwritten store measures the write's failure.
- **A `d/s` sweep by varying `atom_width` instead of the geometry** — declared unrun; `ball_radius` is the
  axis T5.5 is a function of at fixed `(s, σ_q)`, and moving `atom_width` would move the write's own
  expressivity at the same time.
- **The shipped `evict` (re-draw) deletion path as a robustness arm** — specified in the code's docstring
  and **not run**: a re-draw substitutes a random row rather than deleting one, so its Δ would not be the
  item's. Declared, cheap (~7 min) if the Hub wants it.
- **Any change to a shipped default**, `chlu/config.py`, `chlu/core/monitors.py`, or `memory_gym.py`.

---

## 9. GIT FOOTPRINT

Branch **`agent/experiment-engineer/bprime-c6`**, worktree `../CHLU-c6`, base local `main` **`d4f56c8`**.
**Not pushed, no PR, left for Hub review.** Rebase onto local `main`: **already up to date** (base
unmoved), no conflicts.

| commit | what | files |
|---|---|---|
| `e38f9d2` | C6 instrument: `table_third_party_delta` (the computed structural zero, fixed row content), `slot_deltas`, `third_party_curve`, `coupling_law_fit`, `THIRDPARTY_SLOT_GRID` | `chlu/eval/attribution.py` · `tests/test_attribution.py` (+6 tests) |
| `f2f45d8` | the probe + the `d/s` sweep + the derived damping correction + `_write_and_query` (bit-identical extraction) | `chlu/experiments/exp_route3_attribution.py` · `tests/test_route3_attribution.py` (+5 tests) |
| `be995ca` | ⭐ **C2W3 reconciliation 6**: the `exp-route3-attribution` CLI hook, both parts | `chlu/cli/experiment_cmd.py` · `tests/test_route3_attribution.py` (+1 test) |

**Files touched: 5, all inside my §6 ownership list.** `chlu/config.py`, `chlu/core/monitors.py`,
`chlu/core/soft_certificate.py`, `chlu/experiments/memory_gym.py`, `chlu/eval/race.py`,
`chlu/eval/rivals/`, `chlu/eval/dividend.py`, `chlu/core/clu_system.py` were **imported/read/measured,
never edited** (`soft_certificate.py` was read for D3 and left byte-identical — D3 is a measurement).
**Worktree slot 2/3 RELEASED.** Verified from the MAIN repo **before and after** removal
(`git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/bprime-c6` shows all
three commits — the w4 lesson), then `git worktree remove ../CHLU-c6`; `git worktree list` now shows main
+ `../CHLU-rivals` only, and the main checkout is **clean at `d4f56c8`, untouched**. ⛔ **`origin`
untouched; nothing pushed anywhere.**
Artifacts (all under `.claude/`): `outputs/bprime-c6/PREREG.md` · `outputs/bprime-c6/c6_summary.json` ·
`outputs/bprime-c6/results/{exp_route3_thirdparty.json, thirdparty_topup_R042.json, c6_curve.png,
exp_route3_thirdparty.png}` · `outputs/bprime-c6/{sweep,topup}.log` ·
`scratch/bprime-c6/{d3_budget.py, d3_budget.json, analyse.py, plot_merged.py}`.

---

## 10. OPEN QUESTIONS / FOLLOW-UPS / RISKS

1. ⭐ **The prefactor 0.379 is unexplained.** The exponent is T5.5's exactly; the amplitude is not. Two
   candidates, both cheap: the own-well gradient is evaluated at the *mean* jitter radius rather than at
   `σ_q` (a `E‖r‖/σ_q ≈ 1.88` effect in the denominator, which would move the prefactor the right way by
   about that factor), or the multi-atom well's gradient profile is not a single Gaussian's near its
   centre. **Until it is settled, quote the law with its measured prefactor, not as T5.5's table.**
2. **`s = 0.40` is measured for THIS store at THIS write budget.** It varies systematically with the
   geometry (0.482 at `R = 0.42` → 0.362 at `R ≥ 1.0`), i.e. **the fitted well width is not a constant of
   the architecture**, so `d/s` is a ruler that moves under the thing it measures. The two conventions in
   §1 bracket it; a third wave-level fix would be to report `d` and `s` separately and never their ratio.
3. **The launch-noise floor is the wrong null for a Δ probe and the right one for a read-out claim.** I
   report both; whoever writes the audit paragraph must not silently pick the flattering one. (My §5
   paragraph deliberately quotes both.)
4. **Risk — one family, one arm.** Everything here is `overload/load1x_shipped`. The two other families
   cannot be written admissibly at this atom budget (C2W3, independently reproduced in C2W2), so the
   coupling curve rests on the same single family every Route-3 verdict has rested on.
5. **D3's grid is the theorist's 2-D toy, not the shipped 4-D store.** The `B` statement is about that
   grid, exactly as the original `B = 0.33` was. Re-locating `B` **on the shipped store** would need the
   capture-radius leg run there, which is `monitors.py`/`clu_system.py` territory and not mine this wave.

---

## Proposed handover updates (for the Hub)

- **§2 architecture:** `chlu/eval/attribution.py` gains the **C6 third-party instrument**
  (`table_third_party_delta` — the computed structural zero; `slot_deltas`; `third_party_curve`;
  `coupling_law_fit`, which returns an *independent estimate of the learned well width*;
  `THIRDPARTY_SLOT_GRID`). `chlu/experiments/exp_route3_attribution.py` gains **part 2**
  (`run_thirdparty_cell`, `run_experiment_thirdparty`, `BALL_RADIUS_GRID`) and a shared
  `_write_and_query` (verified bit-identical to `d4f56c8`).
- **§3 CLI/config:** ⭐ **`chlu exp-route3-attribution` now exists** (`--part {curve,thirdparty}`,
  `--families`, `--seeds`, `--radii`, `--quick`, `--no-escalate`) — **C2W3 reconciliation 6 is CLOSED**.
  **No config default changed anywhere; `chlu/config.py` untouched.**
- **§7 Known Issues — ADD (OPEN, needs an owner):** (a) *"T5.5's `d/s` placement of the shipped rig uses
  `d_safe_override` as `d` and `atom_init_width` as `s`; the achieved geometry is `d/s = 4.34`, and the
  measured coupling there is `1.53e-2`, 45–52× below the charter's own pricing"*; (b) *"`B = 0.33`'s
  outer edge is unlocated under the corrected/measured ruler — `B ≥ 0.542` is unrefuted; the shipped
  default is unchanged and this is a measurement"*; (c) *"R1's corrected inradius gives no improvement
  over `sep/2` on the anisotropic (`naxis`) store the certificate work uses; its 14.55× was an isotropic
  result"*.
- **§7 — ADD (RESOLVED):** *"`exp-route3-attribution` has no CLI hook"* — landed with a test.
- **Registry / OQ ledger:** `doctrine-repairs` **OQ-A is CLOSED** (D3, this report) and
  **OQ-C is materially advanced** — the learned multi-atom well width is now **measured**: `s = 0.40`,
  two independent ways, 0.7 % apart. `bprime-theory` **§9 item 2**'s *"unsolved modelling question"*
  should be annotated with that measurement and its domain (this store, this write budget, and it drifts
  0.48 → 0.36 with the geometry).
- **NOT-RUN list:** **OQ-2 stays declared NOT-RUN** (wave-boundary only). New declared NOT-RUNs: the
  `evict`-path deletion robustness arm · an `atom_width` sweep · `aggregate`/`manifold` third-party
  curves.
- **Test count:** `test_attribution` 14 → **20**, `test_route3_attribution` 7 → **13**; full suite on the
  branch **1073 passed, 0 failed** (16 m 40 s) ⇒ base `d4f56c8` = **1061** (inferred by subtraction, not
  separately measured).
- ⭐ **One sentence for the Advisor, if only one survives:** *the only coupling a per-slot table cannot
  express is real, is the store's, obeys `exp(−½(d/s)²)` on the learned store to R² = 0.995 — and our own
  admission gate has already put it two orders of magnitude below the query's launch noise, so "a store
  organised well enough to be safe is organised well enough to be a table" is TRUE, quantitatively, at
  the geometry we ship.*
