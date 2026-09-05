# PREREG — `r2-d-sweep-close` (w27, campaign [C1W27])

**Written before any harness run that measures a registered quantity.** What preceded it:
(i) a pure JAX/import warm-up, (ii) the Stage-1 config edit + its unit test (a *criterion*
change, whose only measured content — a value-blank at m=4 scoring `strict_tol = 1.0000` and
`strict_decode = 0.03125 = 1/32` on a **designed, untrained** store — is the thing the Head
already ruled on and is asserted in the test), (iii) pure-numpy codebook / atom-count /
site-separation arithmetic (no dynamics, no training). No learned write and no re-render of any
w26 table has been executed at the time of writing.

Base: local `main @ 082d095`. Branch `agent/experiment-engineer/r2-d-sweep-close`, worktree
`../CHLU-r2dsweep`, **main venv reused** (JAX **0.9.0**, equinox **0.13.4**).

---

## ⭐ DIAL DECLARATION (protocol §7, carried from the task file verbatim)
- **Dial:** capacity (the R2 law). A law about the primitive — exempt from the masked-recall
  demotion; its figure is **never** framed as beating anything (CM-23(m)).
- **Laundering control:** the **designed** write (`BallRegisterPotential`) at matched geometry, on
  the **same payload format** and the **same read**, at **every** `d`, `K`, `m` and `σ_obs` I
  report. Value-blank control on every reported PASS. If the learned arm only "works" by becoming
  more designed → **N46 scope collapse**, not a win.
- **Falsifies the claim:** at d=6 and/or d=8, m=4 (± the annealed read) does **not** move the wall
  above the w23-class ceiling (`K_learned(6) = K_learned(8) = 32`, the "`min(2^d, ~32)`" law) at
  **≥3 seeds with payload read-noise ON** ⇒ the w26 unclamping is a **d=4 artefact** and R2 closes
  with the exponent unchanged. That outcome is reported as *the result*.
- **Does NOT falsify:** failing to reach exactly `4·2^d` (the prefactor gap is expected and known);
  any comparison to kNN / Hopfield / any external method; losing on a metric-native protocol
  (standing metric-native-ceiling theorem); a smaller wall movement than hoped.

---

## 1. The five binding fairness conditions, restated as checkable items (Head ruling B1.3)

| # | condition | the check I will publish |
|---|---|---|
| **1** | **Bits-per-item constant** across every arm, every `d`, and the baseline | for every reported cell I print `code_minsep`; it must equal `Δ = 2/(K−1)` at **every** `m` to 1e−5 (asserted in `test_payload_codebook_holds_min_separation_and_cuts_excursion`). Same `K` + same `Δ` + same per-axis noise ⇒ same `log₂K` bits. Only `max‖a‖` (the reach demand) may fall. |
| **2** | **Byte accounting pinned explicitly** | `n_learned_params` printed per cell; the m-channel arm's extra floats are exactly `n_atoms·(m−1)` (atom centers). Published as a table with items/param and bits/param. The **`spectator`** control (m dims + the same params, code left at excursion 1.0) separates "extra bytes" from "the code". |
| **3** | **Payload read-noise ON** | `payload_launch_sigma = 0.05` on **every** Stage-2/3 number, `payload_obs_sigma ∈ {0, 0.005, 0.010}` swept, scored by nearest-codeword **decode**. No `K_learned` is quoted at `pscale ≠ 1`; **every arm here runs at `pscale = 1`.** |
| **4** | **Baselines given the same format** | the designed arm is built from the identical `(K,m)` codebook and read through the identical (possibly annealed) read at the identical `σ_obs`. A learned PASS is only claimable **where the designed arm at the same `σ_obs` still passes** (w26's rule, carried). |
| **5** | **The laundering control travels** | every table carries its designed column; if any format/lever change makes the *designed* arm weaker, the comparison is void and I say so. |

**Metric.** `pass_metric = "decode"` (the Stage-1 default) everywhere. Cell **PASS** ⇔ mean
`strict ≥ pass_strict = 0.9` **and** the value-blank control passes. `strict = basin_ok ∧ decode_ok`.
**sd convention: population sd (`np.std`, ddof=0)** — matching w26's r2 tables. Stated in every table.

---

## 2. Compute priority order (declared; followed in order; anything unreached = **NOT RUN**, never a null)

1. **Stage 1** — `pass_metric` default → `decode`, regression test, re-render of the w26 d=4 tables.
2. **Stage 2, d=6** — `K_learned(6)` at m=4 vs the designed arm (the headline).
3. **Stage 3** — the arm(a)×arm(b) 2×2 at d=4 **and** d=6.
4. **Stage 2, d=8**.
5. **Stage 4** — items/bits per parameter, *only if* the wall moves at d ≥ 6.

**≤ 4 concurrent background jobs** (w26 thermal incident: load 575 on 8 cores). `PPID=1` jobs are
harness-detached, not orphaned.

**Budget declaration (N92).** d=6 primary = the **shipped** floor `min_atoms_base = 512` ⇒ **4096
atoms** — deliberately the *same* budget that produced the w23-class `K_learned(6) = 32` I am
testing against, so the comparison is apples-to-apples. Every **first-fail** cell is re-checked at
**2×** and (compute permitting) **4×** atoms; a cell that passes at 2× is reported as
**budget-limited (a lower bound)**, never as a wall. d=8 primary = shipped floor ⇒ **8192 atoms**.
w26's d=4 numbers used the coverage-raised 4× floor (8192 atoms); the d=4 cells I re-run for
Stage 3 keep that budget so they stay comparable to w26, and **the budget is printed in every table**.

---

## 3. Registered predictions

### Stage 1 — what the `decode` default changes in the w26 d=4 tables
Derivation: `decode` is stricter than `tol` iff half the codeword min-separation is below
`payload_tol`, i.e. `1/(K−1) < 0.1` ⇔ `K > 11`. Every w26 cell has `K ≥ 16`, so **every** number
previously quoted on `tol` can only fall or stay. How far it falls is set by the read's own
settling error vs `Δ/2`.

| # | registered | P |
|---|---|---|
| **S1.1** | **Stage A (d=6 K=64 m=1 σ=0)**: all four factorial cells fall by **≤ 0.03** absolute; the interaction stays **positive** and ≥ +0.35; the simple effect of `atom_init_local` at width 0.30 stays a **null** (|effect| < 0.05). *The Stage-A conclusion is unchanged.* | 0.75 |
| **S1.2** | **arm (b) d=4 K=16/K=32, σ=0 column**: changes by **≤ 0.01** (at `σ_launch = 0` the read error is ~1e−4 ≪ `Δ/2 = 0.0323` at K=32). Base stays 0.82±0.01, κ₀=3 stays ≥ 0.99. | 0.85 |
| **S1.3** | **the designed laundering table (§5) changes at large K**: `K_designed(4)` at σ=0 **stays 128** (`Δ/2 = 0.0079` at K=128 and the designed read's settling error is ~1e−3). | 0.50 |
| **S1.3′** | alternative: `K_designed(4)` at σ=0 drops **128 → 64** under decode | 0.40 |
| **S1.4** | **no arm-(a) number changes** — §4/§6/§7's arm-(a) rows were already decode | 0.95 |
| **S1.5** | the *count* of w26 numbers that move is **> 0** (i.e. the reconciliation list is non-empty) | 0.90 |

### Stage 2 — the wall at d=6 and d=8 (the headline)
Derivation. The w23-class law is `K_learned(d) = min(2^d, ~32)`: **32 at d=6 and at d=8**, against
`K_designed(d) = 4·2^d` (256 and 1024). w26 showed the d=4 wall is set by **read-out reach**, a
constraint on `max‖a‖ = 1` that is *independent of d*, while address packing gets **easier** with d
at fixed K (site separation at K=64: 0.549 at d=4, **0.795** at d=6, **0.908** at d=8). If reach is
the binding constraint at every d, removing it (m=4 cuts `max‖a‖` to 0.055 at K=64) should unclamp
at d=6/d=8 too, and the new binding constraint becomes whichever of {address packing, atom budget,
codebook precision vs `σ_obs`} bites first. ⚠ The **codebook precision** ceiling is a **K-property,
not a d-property** (`Δ = 2/(K−1)`), so it caps *every* arm identically — which is why every wall
below is quoted **against the designed arm at the same `σ_obs`**.

| # | registered | P |
|---|---|---|
| **S2.0** | **the falsifier does NOT fire**: at d=6, m=4 passes some `K > 32` at ≥3 seeds with σ_launch = 0.05 on | 0.70 |
| **S2.1** | **`K_learned(6) = 64` at `σ_obs = 0.005`** (3 seeds, decode, budget-adequate) — i.e. **equal to the designed wall at that noise**, tax **1/1**, vs the w23-class 32 | 0.50 |
| **S2.2** | **`K_learned(6) = 128` at `σ_obs = 0`** (σ_launch = 0.05 still on); range 64–256 | 0.40 |
| **S2.3** | **the tax at d=6**, `K_learned(6)/K_designed(6)` at `σ_obs = 0`: **1/2** (128 vs 256) | 0.40 |
| **S2.3′** | alternative taxes at d=6: **1/1** P=0.30 · **≤1/4** P=0.30 | — |
| **S2.4** | **d=8** (if reached): `K_learned(8) ≥ 64`, i.e. above the w23-class 32 | 0.50 |
| **S2.5** | the **designed** arm at d=6 does **not** reach `4·2^6 = 256` under decode at `σ_obs = 0` — the codebook precision (`Δ/2 = 0.0039` at K=256) binds before the geometry does, so `K_designed(6)` measures **128**, not 256 | 0.55 |
| **S2.6** | **exponent**: I will have at most **2–3 d-points on one metric** (d=4 from the Stage-1 re-render, d=6, maybe d=8). ⛔ **I pre-commit to NOT quoting any exponent (base-√2, `d^1.62`, or a new one) unless I have ≥3 d-points on the decode metric with ≥3 seeds each**, and to stating explicitly what I can and cannot re-measure. | — |

### Stage 3 — the arm(a) × arm(b) interaction sign ⭐ (the registered call)
Derivation. `readout-channel-theory` §4.2 states the two arms are **not additive** — "both act on
the same inequality `r ≤ a_U(s_read)`": arm (a) lowers the demand `r`, arm (b) raises the supply
`a_U`. Once either has cleared the inequality the other buys nothing. w26's Stage A found the same
structure for the init×width pair (**substitutes**, interaction +0.459 — note its interaction was
*positive* because the levers were scored on a *damage-repair* axis; here both levers are scored on
a *gain* axis, so substitution shows as a **negative** interaction). I therefore register:

Design: 2×2 at the **first-fail K of m=4 alone** (so nothing is ceiling-clipped),
factors `m ∈ {1, 4}` × `read ∈ {base, aniso κ₀ = 3, L = 4}`, ≥3 seeds, at d=4 and d=6.
Interaction `I = (m4,aniso) − (m4,base) − (m1,aniso) + (m1,base)`.

| # | registered | P |
|---|---|---|
| **S3.1** ⭐ | **`I` is NEGATIVE (sub-additive / substitutes), `I ≤ −0.05`** | **0.70** |
| **S3.2** | additive, `|I| < 0.05` | 0.20 |
| **S3.3** | super-additive, `I ≥ +0.05` | 0.10 |
| **S3.4** | at a cell where m=4 alone already **passes** (strict ≥ 0.99), adding arm (b) buys **< +0.01** (a ceiling, and therefore not evidence either way — I will not score S3.1 on such a cell) | 0.85 |
| **S3.5** | the **combination does not move the wall beyond arm (a) alone** at d=4 (i.e. `K_learned(4)` is the same with and without arm (b)) | 0.60 |

### Stage 4 — capacity per byte (runs **only if** the wall moves at d ≥ 6)
w21's ~1.3-bits-per-param is **stale** (measured *under* the reach ceiling). Byte accounting:
learned floats `P = n_atoms · (dim + 2)` with `dim = d + m` (centers `n_atoms·dim`, `log_width`,
`amp`). Stored information `B = K·log₂K` bits (K items, each one of K codewords).

| # | registered | P |
|---|---|---|
| **S4.1** | at the best d=6 cell, **items per 10⁵ learned floats rises ≥ 4×** over the w23-class (K=32) cell at the same budget | 0.60 |
| **S4.2** | **bits per learned parameter stays ≪ 1** (order 1e−2 or below) at every cell I measure — i.e. the re-measure does **not** rescue w21's comparison, it only shows the numerator moved. ⛔ This is a **measurement, not a claim**, and is **not** framed as beating anything. | 0.85 |

---

## 4. What would make me report a NEGATIVE close (and I will, without hedging)
If at d=6, m=4 (± aniso) fails every `K > 32` at ≥3 seeds with the budget verified adequate at 2×
atoms, I report: **the w26 unclamping is a d=4 artefact; `K_learned(d) = min(2^d, ~32)` stands; the
learned/designed tax is unchanged at d ≥ 6.** That is the falsifier firing and it is the wave's
result.

## 5. Defaults
⛔ **No default other than `pass_metric` changes in this task** (B1.4). `atom_init_local`,
`atom_init_width`, `n_payload_channels`, `read_anneal_*` and `m` stay at their shipped values
except inside a declared arm.
