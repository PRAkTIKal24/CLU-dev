# PREREG — c2w11-substrate-and-kills (spoke A)

**Filed 2026-08-10 by `experiment-engineer` BEFORE any cell has run and before the harness code
existed.** Binding above this file: `.claude/outputs/c2w11/PREREG-C2W11.md` (§4 K0–K8, §6 M1–M8, §7
the coverage trigger), `PREREG-TierII.md`, charter ADDENDUM 12 §A33–§A34.

⛔ **Nothing in this document is a result.** Every number is a prediction, a threshold, a selection
protocol registered before the sweep it governs, or a quotation of a banked measurement with its
provenance.

⛔ **Wells are never named semantically** (`PREREG-TierII.md` §2.6, copied verbatim into my report).

---

## 0. The one-line scope

Build the repaired substrate (placing write · RE-SELECTED co-scaled widths · feature-factored
launches) on the factored store, and run **K0 → K7-CAP/K6 → K1 → K2 → K3 → K4 → K5 → K8** plus
**M1/M2/M4/M5/M6** and the coverage half of the C2W9 trigger. ⛔ Every leg is **MECHANICS**. No VALUE
leg, no `OD`, no organizer swap, no verdict.

---

## 1. ⭐ THE WIDTH-SELECTION PROTOCOL — registered BEFORE the sweep (repair (b))

⛔ `atom_width_frac_spacing = 1.5` is **NOT inherited**. The store-population spacing it co-scales
against has changed, so the banked value is no longer a *selected* value.

**The store population, defined.** In the factored store the store population **is** the `N_a`
well anchors (there is no separate sizing set — every placed well is a member of the store). So
`store_population_spacing ≡ median_j min_{i≠j} ||u_i − u_j||` over the `N_a` anchors, measured per
seed, reported per seed.

**The circularity, and how it is broken.** The anchor spacing `sep` is set from the *measured* `s`
(`sep = target_ds · s`), and `s` is set by the atom width, which is a fraction of `sep`. The protocol
below breaks it in one refinement pass and is fixed here before it runs:

1. Place wells at a provisional `sep₀ = 0.859` (the banked `2.7 × 0.318`, `orgdiv-cat-test` §4).
2. For each `w_frac ∈ {0.20, 0.25, 0.30, 0.37, 0.50, 0.75, 1.00, 1.50}`: place the store with atom
   width `s_atom = w_frac × store_population_spacing`, run the **placing write**, then **MEASURE**
   `s` with the `α‖q‖²`-subtracted radial estimator and record `d/s = sep_achieved / s_measured`,
   the fit `R²`, and K1's three legs.
3. **Selection rule (mechanical, no discretion):** among the `w_frac` whose **measured** `d/s` lies
   inside the registered band `[2.5, 2.9]` **and** which pass all three K1 legs, take the one whose
   measured `d/s` is closest to `2.7`. If none satisfies both, take the one closest to `2.7` on
   measured `d/s` alone and **declare K1 FAILED at the selected width** (a failure is reported, never
   selected around).
4. One refinement pass: re-place at `sep = 2.7 × s_measured(selected)` and re-measure; the reported
   operating point is the refined one.
5. **Selection seeds are 100/101/102 — DISJOINT from the claim seeds 0/1/2(/3/4).**
6. The harness then **REFUSES** to run at any other width (`UnselectedAtomWidth`, the repair-(d)
   pattern), pytest-asserted.

⚠ Both the sweep and the selection run on the **STORE population** spacing. `d_safe_population =
"sizing"` has no analogue here and is not used.

---

## 2. ⭐ THE FEATURE-FACTORED LAUNCH — the construction, registered before it is measured (§A34.1)

**The channel decomposition of φ.** φ's own decomposition is its frozen code dictionary `{e_j}`. The
launch head extracts `k` channels from the *set-code alone* by **greedy matched-filter deflation**:

```
r_0 = phi(x)                                    # the lossy set-code, R^d
for c = 1..k:  j_c = argmax_j <r_{c-1}, e_j>    # channel c's code direction
               r_c = r_{c-1} - <r_{c-1}, e_{j_c}> e_{j_c}
launch point of channel c:  q_c = (R * e_{j_c} + sigma_q * xi_c , 0_m)
```

- **One particle per semantic feature channel of φ**, `k` structured by the encoder's decomposition
  rather than free. `k = F = 4` is the registered headline (`n_channels = None ⇒ F`); `k = 8` runs as
  a **declared out-of-protocol diagnostic** (it re-draws the launch protocol, so it is never a score
  — the `orgdiv-null-arms` §3.1 `P`-sweep precedent).
- ⛔ **No binding structure is built.** Channels are independent particles; binding is the READ + ψ's
  job (spoke B).
- **Legitimacy:** the head reads only `phi(x)` and φ's own frozen parameters `{e_j}` (ledgered in
  `phi_bytes`). It never reads `A(x)`, the payloads, or the store. It is the **address head**, its
  own head, per §A31.4.
- The deflation makes the `k` selected code directions **distinct with probability 1** (after full
  deflation `<r_c, e_{j_c}> = 0` exactly), which is the structural difference from C2W5's `P = 4`
  designed offsets from ONE set-code.
- **Launch keys are frozen and emitted**, so the null arms are bit-identical on launches.

**The launch diamond** (the coverage instrument, §7 of the wave prereg): channel `c`'s diamond is the
ball of radius `reach = reach_radius_frac_s × s_measured` around `q_c`, `reach_radius_frac_s = 2.0`
registered here. A needed well `j ∈ A(x)` is **covered** iff `min_c ||u_j − q_c|| ≤ reach`.

**Coverage-failure threshold, registered before the run:** the trigger **FIRES** iff the mean over
unseen queries of the fraction of `A(x)` wells that are **not** covered exceeds **0.20**.

---

## 3. ⭐⭐ THE PREDICTIONS — committed numbers (w14 rule), scored in my report

Statistics: 3 seeds on instrument cells, 5 on family/no-store cells, `ddof = 1`, `SE = sd/√n`.

### 3.1 The kill-conditions

| id | quantity | **point prediction** | band | P(the leg passes its bar) | reasoning |
|---|---|---|---|---|---|
| ⭐ **K0** | distinct-`F` fraction, feature-factored, `k=F=4`, unseen, **no store** | **0.97** | [0.85, 1.00] | **0.90** | deflation guarantees `k` distinct **code directions**; anchors are the relaxed codes at min separation, so channel `c`'s launch point's nearest anchor should be its own well. Residual risk = **feature-channel collinearity at `d = 4`** (32 codes in `R^4`), which can put a launch point nearer a third anchor. That risk is exactly what K0 exists to measure and is the whole reason my band reaches down to 0.85 |
| **K0** | mean distinct wells reachable | **3.95** | [3.70, 4.00] | bar `≥ F − 0.5 = 3.5` | as above |
| **K0-baseline** | the SAME statistic on C2W5's `P = 4` designed offsets, re-run here | **0.050 / 2.20** | ±0.02 | — | a **reproduction** of `orgdiv-null-arms` §3; if I do not reproduce it my instrument is wrong, not the banked number |
| **K1** | placing write at the selected width, `a = 12` | endpoint loss **0.030** · `λ_min>0` **1.00** · capture ≥ σ_q **0.95** | loss [0.005, 0.20]; capture [0.60, 1.00] | **0.60** | the placing write sets depth/width by construction instead of hoping 300 adamw steps find them; the banked gradient write needed `a = 32` to clear the capture leg at `a = 12` (0.69–0.88). ⚠ The **stationarity** term is the risk: an atom cloud placed exactly at the target does not cancel `2αq` (magnitude `2·0.05·2.24 = 0.224 ⇒ l_grad ≈ 0.05` on its own), which is why the placing write carries a **rigid cloud shift** solving `∇V(target) = 0` |
| **K2** | rule 4, set half / payload half at `m = 8` | **1.000 / 1.000** | — | **0.95** | re-verification of C2W5 D1, not a new claim |
| **K2** | payload half at the registered `m = 1` | **0.005** | [0.00, 0.02] | ⛔ FAILS | D1's banked sweep 1/2/4/6/8/12 → 0.005/0.119/0.802/0.987/1.000/1.000 |
| **K3** | nearest-item table · strongest +0 B substitute | **0.000 / 0.001** | ≤0.01 | **0.95** PASS | banked 0.0000 / 0.0008. ⚠ **I predict it passes VACUOUSLY** (P = 0.80) |
| **K4** | four leak controls, **store-only form** | all **0.000** | ≤0.005 | **0.90** PASS | banked all 0.0000. ⚠ **I predict it passes VACUOUSLY** (P = 0.75); the non-vacuous form needs full ψ and is spoke B's, emitted as a frozen obligation |
| **K5** | per-item table launder margin | **0.00** | [0.00, 0.10] | **0.20** | banked: FAILED vacuously (read 0.0000, table 0.0000). The substrate repairs are addressability repairs; K5 needs the *payload channel* to arrive |
| ⭐ **K6** | fraction of queries whose **asserted set is already exactly right**, before any reader | **0.010** | [0.002, 0.08] | reported, not a bar | banked matched-filter EXACT-set at `d = 4` = **0.006** (`orgdiv-cat-test` §7.1). ⭐ **The launch head raises `K0` without raising `K6`** is the registered separation of *addressability* from *precision*; if `K6 > 0.15` the fitted-reader scores are not interpretable and I say so |
| **K7-CAP** | reader params `< N_a·m = 256` | 104 / 72 / 0 / 92 **+ 0** (the zero-parameter member) | exact | **0.98** | re-measured from the code that computes them, never from a doc |
| **K7-CAP** | the SP-1 out-of-class probe (`N_a·m` dof, OLS on the true indicator, blank store) | exact-set **1.0000**, `‖v̂−v‖∞ < 1e-10` | — | **0.95** | banked 1.0000 / 4.25e-15 |
| ⭐ **K8** | `K < N_a` cell `(N_a=32, F=4, K=24)`: rule-4 split exists | `n_valid ≥ 30 000` | — | **0.95** | one stored item blocks `F(N_a−F)+1 = 113` combinations; `24·113 = 2712` of 35 960 |
| ⭐ **K8** | SP-1 rank-deficiency ⇒ probe **cannot** recover `v` | design-matrix rank **24** (`< 32`); probe exact-set **< 0.05**, `‖v̂−v‖∞ > 0.1` | — | **0.90** | the `1_A ↦ y` design matrix has `K = 24` rows and `N_a = 32` columns; verified at C2W5's `K=12 < N_a=16` fixture |

### 3.2 The MECHANICS legs I own

| id | quantity | **point prediction** | band | P |
|---|---|---|---|---|
| **M1** (=K0) | distinct-`F` fraction ≥ 0.80 | 0.97 | [0.85, 1.00] | 0.90 |
| **M1** designed negative | a launch set **collapsed to one channel** scores ≈ chance | distinct-`F` fraction **0.000**, mean distinct **1.00** | exact | 0.99 |
| **M2** (=K1) | admissible at the selected width | see K1 | | 0.60 |
| **M2** designed negative | the harness **REFUSES** at a non-selected width | `UnselectedAtomWidth` raised | exact | 0.99 |
| **M4** sharing/refresh | a re-encountered feature **deepens the existing well**; fraction of rewrite events with non-decreasing depth | **1.000** | [0.90, 1.00] | 0.85 |
| **M4** designed negative | a store that spawns a **private well per item** FAILS the leg | fraction **0.000** | exact | 0.95 |
| **M5** anti-collapse | wells-visited `W/N_a` on the raw feature-factored launch geometry (unseen) | **1.00** | [0.90, 1.00] | — |
| **M5** | the same **after the settle** | **0.60** | [0.30, 1.00] | — |
| **M5** declared band | not-collapsed iff `W/N_a ≥ 0.75`; below ⇒ reported **COLLAPSED**, never null | — | — | P(settled side COLLAPSED) = **0.55** |
| **M5** designed negatives | a one-well launch ⇒ `W/N_a = 1/32` ⇒ COLLAPSED; a uniform launch ⇒ `W/N_a = 1.00` ⇒ pass | exact | exact | 0.99 |
| ⭐ **M6** ⛔DIAGNOSTIC | occupancy precision of the **raw launch geometry** | **0.62** | [0.45, 0.80] | — |
| **M6** | occupancy precision **after the settle** | **0.58** | [0.30, 0.80] | — |
| ⭐ **M6** | ⛔ **the dividend (settle − launch)** | **−0.040** | [−0.20, +0.06] | P(dividend ≥ 0) = **0.35** |
| **M6** | distinct wells occupied, launch → settle | **3.95 → 3.20** | — | — |

> ⭐ **I register a DISAGREEMENT with the Hub's Q3.** The Hub moved `P(dividend ≥ 0)` to **0.50** on
> the capture-repair mechanism. I register **0.35**. My reason: the capture repair fixes whether a
> particle *reaches a basin*, but the feature-factored launch head is a **matched filter on φ's own
> codes**, so the launch geometry it hands the settle is *already much better than C2W5's random
> offsets* (predicted precision 0.62 vs banked 0.406). A settle that merely preserves a good launch
> scores a dividend of 0; a settle that merges neighbouring particles scores a negative one. ⛔ **The
> better the launch head, the harder the settle has to work to earn a non-negative dividend** — this
> is §A25.2's by-construction launder ceiling operating on a *stronger* launder, exactly the
> reservation the Hub itself registered. Both predictions are on the record; the measurement decides.

### 3.3 Geometry and the coverage trigger

| id | quantity | **point prediction** | band |
|---|---|---|---|
| **W1** | selected `atom_width_frac_spacing` | **0.37** | [0.30, 0.45] |
| ⭐ **W2** | measured `d/s` at the **banked, non-inherited** `w_frac = 1.5` | **0.67** | [0.45, 0.95] |
| ⭐ **W2** | ⇒ `w_frac = 1.5` is **below the 2.01 merger floor** and K1 FAILS there | FAIL | P = **0.85** |
| **W3** | `store_population_spacing` (median NN of the 32 anchors) at the refined operating point | **0.86** | [0.78, 0.96] |
| **W4** | `σ_q / store_population_spacing` | **0.175** | [0.15, 0.21] |
| **W5** | effective-`s` estimator fit `R²` at the selected width | **0.995** | [0.95, 1.000] |
| ⭐ **C1** | mean fraction of needed wells **NOT covered** by the union of the `k` launch diamonds | **0.45** | [0.20, 0.70] |
| ⭐ **C2** | ⇒ the **C2W9 coverage trigger FIRES** (threshold 0.20) | **FIRES** | P = **0.75** |

*Rationale for C1: with `k = F = 4` channels and a predicted occupancy precision of 0.62, roughly
1.5 of the 4 needed wells are not the channels' own targets; a diamond of radius `2 s ≈ 0.64` against
a spacing of `0.86` covers a needed well only if a channel happened to select it.*

### 3.4 The registered null outcome, stated in advance so it cannot be spun

⭐ **If K0 clears 0.80 but K5/K6/M6 do not move**, that is the **FIFTH convergent datum on write-side
organization with the substrate repairs CONTROLLED FOR**, and I will write it up as the finding it is
— not as a wasted wave. ⛔ **If K0 does NOT clear, the wave stops at this spoke** and I report it as a
**structural cap**, not a physics null.

---

## 4. Declared NOT-RUNs (⛔ never reported as nulls)

1. **ψ, the novelty head, the organization loss, any null arm, any organizer swap** — not mine.
2. **K4 at full ψ capacity** — I do not own ψ. I ship the store-only form (blocking) **and** a frozen
   K4 harness the organizer spoke must re-run, named in `FROZEN-INTERFACES-C2W11.json` as
   `k4_full_psi_obligation`.
3. **M3 (per-feature G-ADDR)** — `chlu/core/well_lifecycle.py` and `tests/test_gate_addr.py` are
   C2W8-close's territory and READ-ONLY all wave. Declared NOT-RUN with its owner named.
4. **M7 / M8 (the curvature-shape term and its spectrum)** — loss term (c) is the loss-package
   spoke's; not built here.
5. **V1 / V2 / V3 scores, `OD`, `OD_min`** — VALUE, wave level, not mine. I only **freeze** the V3
   budget grid so the two scoring spokes share an axis.
6. **`k = 8` channels** — a declared out-of-protocol diagnostic, never a score.
7. **Attention-ψ, `d = 16`, wormholes, `lambda_traj > 0`** — inherited NOT-RUNs.

---

## 5. Provenance of this file

Filed before `chlu/core/feature_launch.py`, `chlu/experiments/exp_c2w11_substrate.py` and
`tests/test_c2w11_substrate.py` existed, on branch `c2w11-substrate-and-kills` off `main @ 2e1cdb2`,
worktree `../CHLU-c2w11a`.

---

# ⭐ ADDENDUM (filed 2026-08-11, **BEFORE the payload-radius sweep ran**) — the PAYLOAD-REACH REPAIR

Task: `.claude/tasks/c2w11-payload-reach-repair.md`. Branch `c2w11-substrate-and-kills` @ `5db2496`,
worktree `../CHLU-c2w11a`. ⛔ Filed before `stage_payload_reach` existed and before any cell of it ran.
Everything in §§1–5 above is **untouched**; this addendum registers only the repair.

## A1. The registered target (restated verbatim from the task, and it is an ARITHMETIC condition)

> Sweep `payload_radius` (with `atom_payload_init_radius` co-scaled, per C2W5 deviation D4) to the
> **largest** value satisfying `‖v_j‖ / measured_capture_radius ≤ 0.75`.

⛔ **The selection is on the RATIO, never on K5, `OD` or any score.** K5 is scored **once**, at the
selected operating point. If K5 still fails at a ratio ≤ 0.75 with the controls green, that is the
finding and it is reported as the finding.

**Mechanical selection rule, registered now:**
- **Grid** (registered, closed): `payload_radius ∈ {1.00, 0.75, 0.60, 0.50, 0.40, 0.30, 0.20}`,
  with `atom_payload_init_radius = payload_radius` at every point (D4 co-scaling).
- **Selection seeds 100/101/102** — disjoint from the claim seeds 0/1/2 (same discipline as the
  width selection, §1 above).
- **Instrument:** the SC-6 capture radius **as K1 already measures it** (`n_dirs = 8`, `r_hi = 1.0`,
  `steps = 8`, `tol = 0.15`, min over directions), so the number is comparable to the banked
  **0.8535** and the repair cannot be bought by changing the ruler. `r_hi = 1.5` is measured beside
  it as a **censoring diagnostic only** and is NOT the selection instrument.
- **Rule:** `ratio(r, seed) = r / median_j capture_j(r, seed)`. **Select the largest `r` on the grid
  with `max_seed ratio(r, seed) ≤ 0.75`** (holds on every selection seed, not on a pooled median).
  If no grid point qualifies, take the smallest `r` and **declare the target UNMET** — a miss is
  reported, never selected around.
- **Reported beside it (not selection inputs):** the **per-well** capture-radius distribution
  (all `N_a = 32` wells, min/p10/median/max), the per-well ratio distribution and the fraction of
  wells with `r ≤ capture_j`; and the **payload-direction reach** (the same bisection along the one
  direction the read actually crosses, `−v_j/‖v_j‖` in the payload block).

## A2. Numeric predictions (⭐ committed BEFORE the sweep)

| # | quantity | **predicted** | band / probability | derivation |
|---|---|---|---|---|
| **R1** | `capture_radius(r)` over the grid | **≈ flat at 0.85** | [0.75, 0.92] for `r ≥ 0.4` | the binding competitor is the **address** spacing 0.8586, which does not depend on `r`; full-space well separation `√(0.859² + 2r²)` only *shrinks toward* 0.859 as `r → 0`, so capture cannot rise much and should not fall until the payload block stops contributing |
| **R2** | **selected `payload_radius`** | **0.60** | [0.40, 0.75] | largest grid point below `0.75 × 0.85 = 0.638` |
| **R3** | achieved ratio at the selection | **0.70** | [0.55, 0.75] | `0.60 / 0.85` |
| **R4** | fraction of wells with `r ≤ capture_j` at the selection | **0.95** | [0.80, 1.00] | a 25 % median margin should absorb the per-well spread |
| ⛔ **K5** best margin at the repaired point | **0.000** | [0.000, 0.03]; **P(K5 PASS) = 0.10** | both `y` and `tol` scale with `r`, so the family's difficulty is scale-**invariant**; the binding statistic is `correct-and-distinct = 0.922 of 4`, which is an **address-side** quantity the launch head fixes before any payload exists. Closing reach lets a particle that launched near a needed well *settle into* it; it cannot make the head *select* it. |
| | cells remain mechanically **vacuous** | **YES** | P = **0.85** | same argument; chance ≈ 0.0007 |
| ⭐ **M6** launch precision | **0.2308** (unchanged, to 4 dp) | P(unchanged) = **0.95** | occupancy is computed on the **address block** only and the launch head never reads a payload ⇒ payload-independent **by construction** |
| ⭐ **M6** settle precision | **0.19** | [0.10, 0.2308] | particles now start inside the basin of the well they launched at, so the settle should approximately *preserve* launch occupancy instead of losing 68 % of it |
| ⭐⭐ **M6** dividend | **−0.04** | [−0.12, +0.01]; **P(dividend ≥ 0) = 0.20** | the settle can only *lose* wells the launch already occupied; it has no mechanism to acquire a well the launch missed. ⛔ **I predict the sign does NOT flip** — it moves most of the way to zero. |
| **M6** distinct wells (launch → settle) | 3.998 → **3.95** | [3.6, 4.00] | same mechanism |
| **C1b** address-space uncovered fraction | **0.7546** (unchanged) | P = **0.95** | the address-space coverage statistic never touches a payload |
| **C1c** full-space uncovered fraction | **0.93** | [0.80, 1.00] | the coverage **diamond** radius is `2s = 0.64`, *not* the capture radius 0.85; at `r = 0.60` a needed full target is covered only if the address error is `≤ √(0.64² − 0.60²) = 0.22` |
| **C2b** the coverage mode | **PERSISTS** | P = **0.95** | address-space 0.75 ≫ 0.20 |
| **K0** | **bit-identical** to banked (0.9967 / 3.9967) | P = **0.99** | K0 runs with **no store** and no payload |
| **K2** payload half | **exactly invariant**: 1.000 at `m = 8`, 0.0052 at `m = 1` | P(|Δ| < 1e-6) = **0.90** | ⚠ **VERIFIED, NOT ASSUMED** (task): `tol = tol_frac · RMS‖y − ȳ‖` and `min_B‖y(A) − y(B)‖` are both **homogeneous of degree 1** in `payload_radius`, so their ratio is invariant |
| **K1 / K3 / K4 / K6 / K7-CAP** | all unchanged and **green** | P = **0.85** each | K6/K7-CAP are launch- and reader-class-side; K1's legs are set by the placed depth and width |
| ⭐ **`kills_all_passed`** | **false** (K5) | P = **0.88** | above |

## A3. What would make me wrong (the falsifiers, in advance)

- **R1/R2 wrong** if capture collapses with `r` (basins merging in the payload block) ⇒ the ratio is
  **unclosable** and the family-construction law `‖v_j‖ < capture ≲ spacing` has **no admissible
  interior** at this design point. That is a *stronger* structural finding than the current one.
- **K5 prediction wrong** (margin > 0.10) ⇒ reach, not organization, was the binding constraint all
  along, and four waves of write-side-organization conclusions need re-reading.
- **M6 sign flips positive** ⇒ the settle *adds* correct-well information once it can reach; the
  dividend's sign was a reach artefact, not a physics one.

## A4. Registered non-negotiables for this addendum

⛔ No re-opening of the width selection (`w_frac = 0.37`) or of `a`. ⛔ No ψ / organizer / null arm /
`OD` / VALUE leg. ⛔ `v3_budget_grid`, `k8_structural_split` and the reader class stay **unchanged**
unless the sweep forces it, and a forced change is reported loudly.
⛔ `TRAVERSAL-FAILURE-SIGNATURE.md` **§1 is not edited**; a dated **§1b** is appended.
