# c2w11-payload-reach-repair — experiment-engineer report

⛔ **THIS REPORT CONTAINS A DOWNSTREAM RECONCILIATION LIST — 7 items, immediately below, and it needs
a Hub owner** (protocol §5 corollary). Item 1 is blocking for spokes B and C: **the operating point
moved**.

**Task + acceptance criterion (one line):** close the ONE measured arithmetic blocker
(`‖v_j‖ / measured capture = 1.172 > 1`) by sweeping `payload_radius` to the **pre-registered** ratio
`≤ 0.75`, then re-run the kill set with the controls **blocking** and score **K5 once** at the
selected operating point.
**Status: done.** Ratio **closed** (1.172 → **0.692**, target met on every seed). ⛔ **K5 still FAILS,
still vacuously; `kills_all_passed = false`.** Every control (K0 K1 K2 K3 K4 K6 K7-CAP, M4, M5)
re-run and **green**, K2's scale-invariance **measured, not assumed**. **M6's dividend moved
−0.1567 ± 0.0052 → −0.0015 ± 0.0026** (2 SE spans zero; 1/3 seeds positive).

## ⛔ RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary, first 10 lines)
1. ⭐⭐ **THE OPERATING POINT MOVED.** `payload_radius = atom_payload_init_radius = **0.60**` (was 1.0)
   ⇒ `tol` **0.4783/0.4714/0.4660 → 0.2870/0.2828/0.2796** (exactly ×0.6). **Spokes B and C must
   re-read `FROZEN-INTERFACES-C2W11.json`** — it is re-emitted with `payload_reach_ratio`. Everything
   they are gated on (`v3_budget_grid`, `k8_structural_split`, reader class, φ, launch protocol,
   selected width) is **byte-identical** — verified by diffing the two JSONs, not asserted.
2. ⭐⭐ **The M6/C2W5 "the settle destroys occupancy information" reading is a REACH ARTEFACT.** At
   `‖v‖/capture = 0.692` the dividend is **−0.0015 ± 0.0026**. Every site quoting −0.1567 (or C2W5's
   −0.1094) must be **scoped to `‖v‖/capture > 1`**. ⚠ **The Hub's Q3** (dividend non-negative,
   P = 0.50), which my spoke-A report scored **REFUTED**, is at the repaired point **neither refuted
   nor confirmed** (2 SE spans 0). Needs an owner and a re-score.
3. ⭐ **C2W5 deviation D4 (`atom_payload_init_radius`) is INERT under the placing write** — measured:
   the placed centers are **bit-identical** across `atom_payload_init_radius ∈ {1.0, 0.0, 0.3}` (they
   differ under the gradient write). It is co-scaled as the task requires and emitted in the JSON, but
   it changes nothing at this write mode. Anywhere D4 is quoted as a live mechanism in a placing-write
   cell needs a note.
4. ⭐ **The family-construction law has a measured FLOOR too:** at `‖v‖ ≤ 0.20` the **placing write
   itself fails K1** (loss 0.058–0.062 > 0.05). Registry form should be
   **`k1_write_floor ≲ ‖v_j‖ < capture_radius ≲ min-well-spacing`** — an *interval*, measured
   `≈[0.25, 0.64]` here. C2W5's D3 picked **1.0**, i.e. **above** the interval.
5. **Spoke A's reconciliation 1 (G-DRIFT) stands but its NUMBER moved 200×:** site drift/spacing
   `2.0e-6 → 4.3e-4` at `r = 0.60` (and `1.0e-2`, i.e. *at* the floor, at `r = 0.20`). Still
   `fails_low_D2a_table_expressible`. The escalated ruling should use the new number.
6. **§13 item 2 of spoke A's report ("Nothing in this wave's budget can close [reach]") is
   SUPERSEDED** — it was closed inside the wave, and closing it did **not** move K5.
7. ⚠ **Numbering:** the task asked for a "§12 addendum" to `c2w11-substrate-and-kills.md`; §12/§13
   already exist and the body was to stay untouched, so it is filed as **§14** (C2W8-close vi.3
   collision precedent). No line of §§1–13 was edited.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echoed before the first result
- **Dial / pillar:** **none — instrument repair.** This is a bounded repair of one measured number on
  the TIER ii substrate, plus the re-run of the kill set. ⛔ No VALUE leg, no `OD`, no organizer, no
  ψ, no tier-ii verdict, no paper number.
- **Laundering control:** every launder margin here is **DIAGNOSTIC** and can never fail a leg
  (§A33.1). The launch-only launder, the settle-deleted launder, the address-leak launder and the byte
  ledger are reported beside every reading. The **blocking** controls are K0/K1/K2/K3/K4/K6/K7-CAP,
  and a K5 pass bought by a degraded control would have been declared **not a pass**.
- **Falsifies the repair:** a ratio that cannot be closed at any grid point (it was closed);
  **or** a control degrading to buy the ratio (none did — K2's payload half is bit-identical, K1
  passes at `a` = 4/12/32, K0 is bit-identical).
- **Does NOT falsify:** K5 failing at a closed ratio. That was **pre-registered** as the likely
  outcome (P(pass) = 0.10) and it is the wave's finding, not a defeat.
- ⛔ Wells are never named semantically (`PREREG-TierII.md` §2.6, quoted verbatim in the spoke-A report).

---

## §1 — ⭐ PREREG ADDENDUM FILED **BEFORE** THE SWEEP

`.claude/outputs/c2w11-substrate-and-kills/PREREG.md`, section **“ADDENDUM (filed 2026-08-11, BEFORE
the payload-radius sweep ran)”** — written before `stage_payload_reach` existed. It registers: the
closed grid `{1.00, 0.75, 0.60, 0.50, 0.40, 0.30, 0.20}`; selection seeds **100/101/102** (disjoint
from the claim seeds); the **instrument** (SC-6 capture exactly as K1 measures it: `n_dirs = 8`,
`r_hi = 1.0`, `steps = 8`, `tol = 0.15`, min over directions — so the number is comparable to the
banked 0.8535 and the repair cannot be bought by changing the ruler); the rule (**largest** `r` with
`max_seed r/median_j capture_j ≤ 0.75`, else the smallest `r` with `target_met = false`); and numeric
predictions for K5, M6, coverage, K0, K1, K2. **Scorecard in §7.**

⛔ **The selection saw only ratios.** `select_payload_radius()` takes a `{radius: ratio}` map and
nothing else — it cannot see K5, `OD` or any score, and that is pytest-asserted by its signature and
its designed negative.

---

## §2 — ⭐⭐ THE SWEEP, AND THE PER-WELL DISTRIBUTION THE MEDIAN WAS HIDING

**Selection seeds 100/101/102, all `N_a = 32` wells measured per cell, `a = 12`, `w_frac = 0.37`.**

| `payload_radius` = `‖v_j‖` | capture median, per seed | ⛔ **ratio, per seed** | max | **frac wells inside their own basin** | payload-direction reach | K1 3/3 |
|---|---|---|---|---|---|---|
| **1.00** *(the banked, broken point)* | 0.8633 / 0.8809 / 0.8945 | **1.158 / 1.135 / 1.118** | **1.158** | ⛔ **0.000** | 0.855 | ✅ |
| 0.75 | 0.8828 / 0.9023 / 0.9238 | 0.850 / 0.831 / 0.812 | 0.850 | 0.802 | 0.899 | ✅ |
| ⭐ **0.60 (SELECTED)** | 0.8867 / 0.8965 / 0.9414 | **0.677 / 0.669 / 0.637** | **0.677** ✅ | **0.979** | 0.932 | ✅ |
| 0.50 | 0.8750 / 0.8594 / 0.9531 | 0.571 / 0.582 / 0.525 | 0.582 | 0.990 | 0.955 | ✅ |
| 0.40 | 0.8477 / 0.8535 / 0.9160 | 0.472 / 0.469 / 0.437 | 0.472 | 0.990 | 0.979 | ✅ |
| 0.30 | 0.8105 / 0.8418 / 0.9141 | 0.370 / 0.356 / 0.328 | 0.370 | 0.979 | 1.005 | ✅ |
| ⛔ **0.20** | 0.8086 / 0.8457 / 0.9141 | 0.247 / 0.236 / 0.219 | 0.247 | 0.875 | 1.031 | ⛔ **FAIL** |

> ⭐ **SELECTED `payload_radius = 0.60`** (`atom_payload_init_radius` co-scaled), the **largest** grid
> point meeting the registered `≤ 0.75` on **every** selection seed. `qualified_radii = [0.2, 0.3,
> 0.4, 0.5, 0.6]`, `target_met = true`.

**⭐ Capture is FLAT in the payload radius, as registered (R1).** It varies over `0.809–0.953` with no
trend — because the binding competitor is the **address** spacing (0.8586), which does not depend on
`r`. ⇒ **the ratio is closed entirely by the numerator**, exactly as the derivation said it had to be.

### 2.1 ⭐ The per-well distribution at the SELECTED point, on the CLAIM seeds (`a = 12`, 16 wells/seed)

| seed | ratio (median) | capture: min / p10 / **median** / max | reach ratio per well: min / median / **max** | wells inside own basin | censored at `r_hi = 1.0` |
|---|---|---|---|---|---|
| 0 | **0.6919** | 0.656 / 0.699 / **0.867** / 0.996 | 0.602 / 0.764 / **0.914** | **16/16 = 1.000** | 5 |
| 1 | **0.6919** | 0.566 / 0.709 / **0.867** / 0.996 | 0.602 / 0.746 / **1.059** | 15/16 = 0.938 | 3 |
| 2 | **0.6812** | 0.574 / 0.738 / **0.881** / 0.996 | 0.602 / 0.742 / **1.045** | 15/16 = 0.938 | 6 |

⚠ **Reported because it is the honest reading:** the 25 % median margin puts **45 of 48** wells inside
their own basin, **not 48**. Two wells (seeds 1, 2) still have `‖v_j‖ > capture_j` (ratios 1.059,
1.045) — the per-well spread (`sd ≈ 0.135`) is wide enough that a *median* margin is not a *per-well*
guarantee. ⚠ **And the ratio is conservative where it is censored:** 3–6 wells per seed hit the
bisection's `r_hi = 1.0` ceiling, so their capture radii are **lower bounds** and their true ratios
are **smaller** than reported.

### 2.2 ⭐ The direction the read actually crosses (diagnostic, never the selection instrument)
SC-6's capture radius is a **min over 8 random directions** — a worst case. The read crosses exactly
one direction: `−v_j/‖v_j‖` in the payload block. Measured with the same bisection:
**0.855 at `r = 1.0` → 0.932 at `r = 0.60`.** ⭐ **Even on its own favourable direction the read was
outside the basin at `r = 1.0` (0.855 < 1.0)** — the wall was not an artefact of the worst-case metric.

### 2.3 ⭐ The FLOOR, discovered by the grid (not registered, and stated as unregistered)
At `‖v‖ = 0.20` **K1 fails 3/3** (endpoint write loss **0.0576 / 0.0616 / 0.0586 > 0.05**; capture
`≥ σ_q` on only **0.875 / 0.844 / 0.906** of wells). The endpoint loss and the site drift rise
monotonically as `‖v‖` falls (loss `0.0005 → 0.0077 → 0.041 → 0.058` at `r = 1.0 / 0.6 / 0.3 / 0.2`;
drift/spacing `2.4e-6 → 4.3e-4 → 5.6e-3 → 9.5e-3`) — neighbouring wells' placed atom clouds crowd in
the payload block as the payload shell shrinks. ⇒ **the law is an interval, not a ceiling**:

> ### ⭐⭐ `k1_write_floor ≲ ‖v_j‖ < capture_radius ≲ min-well-spacing`
> measured here as `‖v_j‖ ∈ [≈0.25, ≈0.64]` at (`N_a = 32, F = 4, m = 8, d = 4, w_frac = 0.37`).
> **C2W5's D3 chose 1.0 — above the interval. The repair chooses 0.60 — inside it.**

---

## §3 — ⛔ K5, SCORED **ONCE**, AT THE SELECTED OPERATING POINT

**Seeds 0/1/2 · 512 unseen rule-4-valid queries · `payload_radius = 0.60` · everything else unchanged.**

| seed | chance | `tol` | best physics reader | K5 table | ⛔ **best margin** (bar > 0.10) | vacuous |
|---|---|---|---|---|---|---|
| 0 | 0.00000 | 0.2870 | 0.00195 (zero-parameter identity) | 0.00000 | **0.00000** | ✅ **True** |
| 1 | 0.00195 | 0.2828 | 0.00195 (`sum_linear`, `mlp`, identity) | 0.00195 | **0.00195** | ✅ **True** |
| 2 | 0.00000 | 0.2796 | 0.00000 | 0.00000 | **0.00000** | ✅ **True** |

> ⛔⛔ **K5 FAILS, 3/3, and it fails VACUOUSLY** (`top_physics_score ≤ chance + 0.01`, computed
> mechanically). ⛔ **This is a "not expressible at all" finding, NOT a "table-expressible" one** — the
> table scores exactly what the read scores, and both score chance.
> ⭐ **It was pre-registered**: margin **0.000**, band [0.000, 0.03], **P(pass) = 0.10** — and the
> mechanism was pre-registered with it (the binding statistic is the **address-side**
> `correct-and-distinct = 0.92 of 4`, which the launch head fixes before any payload exists).

**`kills_all_passed = FALSE`** (mechanical AND over K0…K7-CAP; K5 is the only false).

---

## §4 — ⛔ THE BLOCKING CONTROLS, ALL RE-RUN AT THE REPAIRED POINT

| leg | bar | **measured at `‖v‖ = 0.60`** | banked at `‖v‖ = 1.0` | verdict |
|---|---|---|---|---|
| **K0** | ≥0.80 distinct-`F`; mean distinct ≥ 3.5 | **0.9967 ± 0.0026 · 3.9967 ± 0.0026** | 0.9967 · 3.9967 | ✅ **PASS — BIT-IDENTICAL** (`stage_k0.json` summaries compare `==`) |
| **K1** | loss ≤0.05 · λ_min>0 ≥90 % · capture ≥σ_q ≥90 % | `a=4`: 0.0074–0.0116 · `a=12`: 0.0077–0.0121 · `a=32`: 0.0078–0.0121; λ **1.00**; capture **1.00** | 0.0005–0.0027 | ✅ **PASS at 4, 12 AND 32** (⚠ loss ×10, still 4× under the bar) |
| **K2** set half | 100 %, max overlap ≤ `F−2` | **100 %**, max overlap **2** | same | ✅ PASS |
| ⭐ **K2** payload half | 100 % of held-out, at `m = 8` | **1.000 / 1.000 / 1.000**; `m=1` **0.0052** | identical | ✅ **PASS — scale-invariance MEASURED (§5)** |
| **K3** | ≤0.60 | table **0.0000** · strongest +0 B substitute **0.0013** | 0.0000 / 0.0013 | ✅ PASS — ⚠ VACUOUS |
| **K4** (store-only) | all legs ≤ chance+0.05 | blank ≤0.00195 · query-only ≤0.00195 · permuted **0.0000** · ⭐ address-leak dividend **−0.0028** (was −0.1623) | — | ✅ PASS — ⚠ VACUOUS |
| **K6** | reported, never a bar | **0.00065 ± 0.0013** (0/512 · 0/512 · 1/512) | identical | ✅ reported, unmoved |
| **K7-CAP** | every reader < `N_a·m = 256` | `sum_linear` **104** · `well_table` **72** · `knn` **0** · `mlp` **92** · zero-parameter **0** | identical | ✅ PASS |
| **K7-CAP** SP-1 probe (out-of-class DIAGNOSTIC) | — | exact-set **1.0000**, `‖v̂−v‖∞ = 1.22e-15` | 1.4988e-15 | reproduces |
| **K8** | rank-deficient at `K<N_a` + rule 4 | rank **24 < 32**, `sp1_exact_set_unseen` **0.0469/0.0684/0.0723** (identical), `‖v̂−v‖∞` **0.4149** = 0.6 × banked 0.6916 | — | ✅ PASS, **unchanged** |
| **M4** sharing/refresh | ≥90 % non-decreasing depth | **1.000 3/3** (private-well negative **0.000**) | 1.000 | ✅ PASS |
| **M5** anti-collapse | `W/N_a ≥ 0.75`, two-sided | launch **1.000** → settle **1.000**, **OK** 3/3 | 1.000 → 0.885 | ✅ PASS (**improved**) |

⛔ **No control was degraded to buy the ratio.** The only control number that moved materially is K1's
endpoint write loss (0.0007 → 0.0077, bar 0.05) and K1 still passes at every `a` — reported, not
hidden. ⛔ **And K5 did not pass anyway**, so the "a pass bought by a degraded control is not a pass"
clause never had to be invoked.

---

## §5 — ⭐ K2's SCALE-INVARIANCE: **VERIFIED, NOT ASSUMED** (the task's explicit demand)

C2W5's D3 *registers* the K2 metric as scale-invariant in the payload radius. Two independent checks:

1. **Measured, run-to-run.** The whole `m`-sweep at `r = 0.60` is **bit-identical** to the banked
   sweep at `r = 1.0`:
   `m = 1/2/4/6/8/12 → 0.00521 / 0.11979 / 0.80859 / 0.98828 / 1.0 / 1.0` on both, per seed and per
   value (`==` on the raw lists), while `tol` scaled by **exactly 0.6** (`0.47827→0.28696`,
   `0.47141→0.28285`, `0.46601→0.27960`; ratios `0.600000`).
2. **Pytest-pinned** (`test_K2s_payload_half_is_SCALE_INVARIANT_in_the_payload_radius`): at
   `r ∈ {0.6, 0.4, 0.2}`, `payloads = r × payloads(1.0)`, `‖v_j‖ = r` exactly, `tol = r × tol(1.0)`
   (rtol 1e-9) and every K2 field (`frac_payload_sep_ok`, `payload_sep_ok`, `overlap_ok`) is equal.

⇒ **The payload half did not degrade, so the family does NOT need `m > 8`.** (Independent
confirmation from K8: `‖v̂−v‖∞` scaled by exactly 0.6, `0.6916 → 0.4149`, with `exact_set` unchanged.)

---

## §6 — ⭐⭐ M6: THE DIAGNOSTIC THAT MATTERS MOST (⛔ cannot fail a gate, §A33.1)

**3 seeds × 512 unseen, written store, `payload_radius = 0.60`. ⛔ Scored against the BLANK STORE /
raw launch geometry, never against `F/N_a`.**

| statistic | **launch (raw geometry)** | **after the settle** | **dividend** | *banked at broken reach* |
|---|---|---|---|---|
| occupancy precision | **0.2303 ± 0.0090** | **0.2288 ± 0.0065** | ⭐ **−0.0015 ± 0.0026** | *0.2303 → 0.0736, **−0.1567 ± 0.0052*** |
| per seed | 0.2378 / 0.2310 / 0.2222 | 0.2344 / 0.2290 / 0.2231 | **−0.0034 / −0.0020 / +0.0010** | −0.1606 / −0.1577 / −0.1519 |
| distinct wells occupied | 3.998 ± 0.0023 | **3.990 ± 0.0069** | −0.008 | *3.998 → 3.807* |
| exact-set occupancy | 0.00065 | **0.00065** | 0 | *0.0007 → 0.0000* |
| M5 wells-visited `W/N_a` | 1.000 | **1.000** | 0 | *1.000 → 0.885* |
| *(C2W5 reference)* | *0.4061* | *0.2967* | *−0.1094* | |

> ### ⭐⭐ THE SIGN: it did **not** flip at the mean — and that is no longer the interesting statement
> The dividend is **−0.0015 ± 0.0026**: **2 SE spans zero**, and **seed 2 is positive (+0.0010)**.
> The banked −0.1567 (2 SE **0.0052**, no seed near zero) and the repaired value are **>100 SE apart**.
> ⇒ **the settle is no longer destructive: it is neutral.** With reach closed it *preserves* the
> launch geometry (precision 0.2303 → 0.2288, distinct 3.998 → 3.990) instead of losing 68 % of it.
> ⭐ **The −0.1567/−0.1094 “the settle destroys occupancy information” reading was a REACH ARTEFACT**
> — reconciliation item 2. ⛔ **What it is NOT:** evidence the settle *adds* information. It adds
> none (dividend ≈ 0, exact-set occupancy unmoved at 0.00065, K5 at chance). ⛔ M6 fails no gate.

The same movement appears independently in **K4's address-leak dividend: −0.1623 → −0.0028**, measured
by a different code path in a different stage.

---

## §7 — ⭐ PREREG SCORECARD (addendum §A2, filed before the sweep)

| # | registered | measured | verdict |
|---|---|---|---|
| **R1** capture flat ≈0.85, [0.75, 0.92] for `r ≥ 0.4` | flat | **0.848–0.953, no trend** | ◐ **flat CONFIRMED**; 3 of 15 cells above my band's top (0.94–0.95 on seed 102) |
| **R2** selected `payload_radius` **0.60** [0.40, 0.75] | 0.60 | **0.60** | ✅✅ **HIT exactly** |
| **R3** achieved ratio **0.70** [0.55, 0.75] | | **0.677** (sel.) / **0.692, 0.692, 0.681** (claim) | ✅ HIT |
| **R4** frac wells inside own basin **0.95** [0.80, 1.00] | | **0.979** (sel.) / **1.000, 0.938, 0.938** (claim) | ✅ HIT |
| ⛔ **K5** margin **0.000** [0.000, 0.03]; P(pass) = 0.10 | | **0.00000 / 0.00195 / 0.00000**, FAIL | ✅ **HIT** (the registered branch) |
| **K5** cells remain vacuous (P = 0.85) | | **vacuous 3/3** | ✅ HIT |
| **M6** launch precision unchanged (P = 0.95) | 0.2303 | **0.2303 ± 0.0090** | ✅✅ HIT (identical) |
| **M6** settle precision **0.19** [0.10, 0.2308] | | **0.2288 ± 0.0065** | ✅ HIT (top of band) |
| ⭐ **M6** dividend **−0.04** [−0.12, +0.01]; P(≥0) = 0.20 | | **−0.0015 ± 0.0026** | ✅ HIT (in band, at the upper edge) |
| **M6** sign does **not** flip | negative | **negative at the mean; 2 SE spans 0; 1/3 seeds positive** | ◐ **sign held, significance did not** |
| **M6** distinct 3.998 → **3.95** [3.6, 4.00] | | **3.990** | ✅ HIT |
| **C1b** address-space uncovered unchanged 0.7546 (P = 0.95) | | **0.7546 ± 0.0116** (identical per seed) | ✅✅ HIT |
| **C1c** full-space uncovered **0.93** [0.80, 1.00] | | **0.9299 ± 0.0038** | ✅✅ **HIT to 3 dp** |
| **C2b** coverage mode PERSISTS (P = 0.95) | | **PERSISTS, fires 3/3** | ✅ HIT |
| **K0** bit-identical (P = 0.99) | | **bit-identical** | ✅✅ HIT |
| **K2** payload half exactly invariant (P = 0.90) | | **bit-identical sweep, `tol` ×0.600000** | ✅✅ HIT |
| **K1/K3/K4/K6/K7-CAP** unchanged & green (P = 0.85) | | **all green**; K1's loss ×10 (0.0007→0.0077) | ◐ green, K1's loss **moved** |
| ⭐ **`kills_all_passed` false** (P = 0.88) | | **false** | ✅ HIT |
| — **UNREGISTERED** — | | **K1 FAILS at `‖v‖ = 0.20`** ⇒ the law has a floor | ⚠ a **finding**, declared unregistered |

**Score: 13 hits (5 of them exact), 3 partial, 0 misses, 1 unregistered finding.**

---

## §8 — FLAG PROVENANCE (governs every number in this report)

Commit **`9627876`** (the harness tree of record; **the run executed exactly this file content** — no
`chlu/**` edit was made after the run started, only `tests/`, committed as `4324002`) · branch
**`c2w11-substrate-and-kills`** · base **`main @ 2e1cdb2`** · worktree **`../CHLU-c2w11a`** · **main
venv** (protocol §4; `jax 0.9.0`, float32) · seeds **0/1/2** (claim) and **100/101/102** (payload-radius
selection; disjoint).

| flag | value | note |
|---|---|---|
| ⭐ `payload_radius` / `atom_payload_init_radius` | **0.60 / 0.60** | ⭐ **THE REPAIR**, selected on the registered ratio; D4 co-scaled (⚠ and D4 is *inert* under the placing write — recon 3) |
| `write_mode` | `placing` | unchanged |
| `launch_mode` / `n_channels` `k` / `n_particles` | `feature_factored` / `None ⇒ F` / **4** | unchanged |
| `atom_width_frac_spacing` / `atom_width_selected_frac` | **0.37 / 0.37** | ⛔ **NOT re-opened** (loaded from `run1/stage_width_selection.json`) |
| resolved atom width | 0.3178 | `= 0.37 × 0.8590` |
| `place_depth` / `place_jitter_frac_s` / `place_stationarity_shift` | 0.30 / 0.5 / True (40 iters) | unchanged |
| `n_wells / f_subset / n_items / n_unseen` | 32 / 4 / 128 / 512 | unchanged |
| `atoms_per_well a` | **12** (K1 re-run at 4, 12, 32) | unchanged |
| `payload_dim m` / `addr_dim d` | **8** / **4** | ⛔ `m` NOT raised — K2's payload half did not degrade (§5) |
| `s_measured` / `d/s` | **0.3204** / **2.680** | (was 0.3199 / 2.684 — the radial fit sees a different store) |
| `tol` per seed | **0.28696 / 0.28285 / 0.27960** | ⭐ **= 0.6 × the banked values** |
| `chance` per seed | **0.0 / 0.00195 / 0.0** | unchanged (scale-invariant) |
| `store_population_spacing` | 0.8586 (median NN over 32 anchors) | address-space; payload-independent |
| `query_sigma σ_q` / `confine α` | 0.15 / 0.05 | unchanged |
| `gamma_address` / `gamma_read` / budget | 0.05 / 0.02 / **400 + 800** @ `dt = 0.05` | unchanged; every γ statement is read-budget-scoped |
| `kinetic_mode` / `p₀` / `lambda_traj` | `newtonian_learned` / 0 / 0.0 | unchanged |
| `reach_radius_frac_s` / `coverage_trigger_threshold` | 2.0 / 0.20 | unchanged, both pre-registered |
| capture instrument | `n_dirs = 8, r_hi = 1.0, steps = 8, tol = 0.15`, **min over directions** | ⛔ identical to K1's, so the banked 0.8535 is comparable |
| organizer / ψ / null arms | ⛔ **NOT RUN** | not this task's |
| bytes | store 21 504 B (`a=12`), φ 576 B, head **0 B** | unchanged (the repair costs **0 bytes** — it is a family constant) |

---

## §9 — HOW I VERIFIED (commands + observed output)

```bash
# 1. PREREG addendum filed BEFORE any sweep cell (appended to spoke A's PREREG.md)

# 2. the new stage + the repaired kill set, main venv, cwd = the worktree
cp run1/stage_width_selection.json run2/          # ⛔ the width is NOT re-opened
PYTHONPATH=/Users/user/Desktop/CHLU-c2w11a /Users/user/Desktop/CHLU/.venv/bin/python \
  -m chlu exp-c2w11-substrate --seeds 0 1 2 \
  --stages k0 reach m6 k6 k7cap k1 k2 k3 k4 k5 k8 m4 m5 coverage freeze \
  --out-dir .../c2w11-substrate-and-kills/run2
# -> K0 0.9941/0.9980/0.9980 (bit-identical to run1)
# -> [reach] r=1.00 ratio 1.158/1.135/1.118 inside 0.00 ; r=0.60 ratio 0.677/0.669/0.637 inside 0.98
# -> [reach] SELECTED payload_radius = 0.6 (ratio 0.677 <= 0.75: True)
# -> [M6] launch 0.2378/0.2310/0.2222 settle 0.2344/0.2290/0.2231 dividend -0.0034/-0.0020/+0.0010
# -> [K1] PASS at a=4,12,32 (loss 0.0074-0.0121, lam+ 1.00, cap 1.00)
# -> [K3/K4/K5] K3=True K4=True K5=False vacuous=True on 3/3
# -> [coverage] uncovered 0.7461/0.7520/0.7656 fired=True   (full-space 0.9331/0.9268/0.9297)
# -> FROZEN-INTERFACES-C2W11.json (kills_all_passed = False)

# 3. quick smoke of the whole wiring first (16 wells, 2 seeds): PASSED end-to-end

# 4. targeted tests (the 5 new ones)
python -m pytest -q --no-cov tests/test_c2w11_substrate.py -k "reach or SCALE or payload_radius_grid"
# -> 6 passed, 27 deselected in 23.31s

# 5. ruff
python -m ruff check chlu/experiments/exp_c2w11_substrate.py chlu/cli/experiment_cmd.py \
       tests/test_c2w11_substrate.py    # -> All checks passed!

# 6. the re-loadable-stage path: re-emit the freeze from banked stages ALONE
python -m chlu exp-c2w11-substrate --stages freeze --out-dir <copy of run2>
# -> the selected payload radius is picked up from stage_payload_reach.json and
#    the emitted JSON compares == to the shipped one (checked in python)

# 7. the FULL suite (⚠ never the file alone — the x64 hazard is a KNOWN defect class here)
PYTHONPATH=... python -m pytest -q --no-cov tests/
# -> 1612 passed, 29 warnings in 2633.92s (0:43:53)   ⛔ 0 failed
```

**Independent numeric check of reconciliation 3** (D4 inert under the placing write):
```
placing write: centers identical across atom_payload_init_radius 1.0/0.0/0.3 -> True True
gradient write: identical -> False
```

**⭐ The re-emitted frozen file was re-CHECKED for self-sufficiency**, not assumed — spoke A's
spoke-B/C simulation (`.claude/scratch/c2w11-substrate-and-kills/verify_frozen.py`), which reads
**only the JSON**, re-run against the new file:
```
phi:  bytes + byte-hash MATCH -> a2713a0fb155e09f965cb6808720dbb1
launches: BIT-IDENTICAL across an independent rebuild -> (512, 4, 12)
tol from JSON 0.286960 vs rebuilt 0.286960          <- the MOVED tol re-derives
v3 grid: [50,100,200,400,800,1200]  particle-steps: [200,400,800,1600,3200,4800]
kills_all_passed: False | coverage fired: True
ALL RE-DERIVATION CHECKS PASSED
```

---

## §10 — SUITE ARITHMETIC

| checkout | commit | collected | result |
|---|---|---|---|
| base (spoke A's own measurement) | `5db2496` in `../CHLU-c2w11a` | **1 607** | 1 607 passed / 0 failed |
| **this branch** | `4324002` in `../CHLU-c2w11a` | **1 612** | ✅ **1 612 passed, 0 failed in 2 633.92 s (43 m 54 s)** |

**Count arithmetic:** `1 612 − 1 607 = +5`, and **5 is exactly the number of tests I added** to
`tests/test_c2w11_substrate.py` (the reach-selection rule, its designed negative, K2's scale-invariance,
the per-well reach legs, the registered grid). I added no tests elsewhere and removed none.
`--collect-only` on this branch: **1 612 tests collected**. ⚠ Counts are comparable only within one
checkout; both numbers were taken with the main venv in `../CHLU-c2w11a`.
⚠ **HEAD honesty:** the suite was launched at `HEAD = 5db2496` **with my changes present in the
working tree**, and those exact changes were committed as `9627876` + `4324002` while it ran. Checked
after the fact: `git status` is **clean** and `git diff HEAD --stat` is **empty** at `4324002` ⇒ the
tree the suite measured **is** the tree of `4324002`. No other agent works in `../CHLU-c2w11a`.
⚠ Per the standing hazard, this is the **FULL** suite, never the file alone: the x64 defect class
(`jax_enable_x64` flipped globally by an earlier module) is invisible to a per-file run.

---

## §11 — DECLARED NOT-RUNs (⛔ never reported as nulls)

1. ⛔ **ψ, the novelty head, the organization loss, every null arm, the organizer swap, `OD`, any
   VALUE leg** — out of scope by the task, unchanged.
2. ⛔ **The width selection and the `a` choice** — explicitly NOT re-opened; `run1`'s
   `stage_width_selection.json` was **loaded**, not re-run (the harness's re-loadable-stage path).
3. ⛔ **The two-sided drift leg / G-DRIFT adjudication** — escalated to Head/Advisor, explicitly not
   mine; I report the moved number (recon 5) and adjudicate nothing.
4. ⛔ **M3, M7, M8, K4-at-full-ψ** — unchanged NOT-RUNs with their owners named in spoke A's §11.
5. **Traversal** — spoke B's; `TRAVERSAL-FAILURE-SIGNATURE.md` §2 is still absent and its absence is
   still a declared NOT-RUN, not a measurement.
6. **5-seed cells** — every cell here is a 3-seed **instrument** cell; ⛔ none may be quoted as a claim
   cell.
7. **`d`-sweep, `k = 8` channels** — not re-run; spoke A's §9 diagnostic stands (K6 = 0.53 at `d = 16`).

---

## §12 — Git footprint

**Branch `c2w11-substrate-and-kills`** (continued, as the task requires — **no new worktree**),
worktree `../CHLU-c2w11a`, base `main @ 2e1cdb2`, **not pushed** — left for Hub review.
⭐ **Branch ref verified FROM THE MAIN REPO** (protocol §3.2, the wave-4 lost-commits precedent):
`git -C /Users/user/Desktop/CHLU log --oneline main..c2w11-substrate-and-kills` shows `4324002`,
`9627876` on top of spoke A's five. `git rebase main` → *"already up to date"* (base has not moved).

| commit | subject | files |
|---|---|---|
| `9627876` | close the reach gap: sweep `‖v_j‖` to a REGISTERED ratio | `chlu/experiments/exp_c2w11_substrate.py` (+309/−6), `chlu/cli/experiment_cmd.py` (+5/−3) |
| `4324002` | pin the reach repair: the selection rule, its designed negative, and K2's scale-invariance VERIFIED rather than assumed | `tests/test_c2w11_substrate.py` (+105) |

⛔ **Not touched:** `chlu/config.py` · `chlu/core/monitors.py` (import-read-only, C2W10's) ·
`chlu/core/factored_store.py` · `chlu/core/feature_launch.py` (neither needed a change — the repair is
a *config point*, not new physics) · `tests/test_factored_store.py` · all C2W8-close, C2W10, CSF3 and
live-pilot territory. **No conflicts. Rebase onto `main`: no-op (base has not moved).**

**Artifacts (all under `.claude/`):**
`.claude/outputs/c2w11/FROZEN-INTERFACES-C2W11.json` (**re-emitted**, with `payload_reach_ratio`) ·
`.claude/outputs/c2w11/TRAVERSAL-FAILURE-SIGNATURE.md` (**§1b appended, §1 untouched**) ·
`.claude/outputs/c2w11-substrate-and-kills.md` (**§14 appended, §§1–13 untouched**) ·
`.claude/outputs/c2w11-substrate-and-kills/PREREG.md` (**addendum, filed before the sweep**) ·
`.claude/outputs/c2w11-substrate-and-kills/run2/{stage_payload_reach,stage_k0,stage_m6,stage_k6_k7cap,stage_k1,stage_k2,stage_k3_k4_k5,stage_k8,stage_m4,stage_coverage}.json`
+ `FROZEN-INTERFACES-C2W11.json` + `c2w11_substrate_summary.json`.

---

## §13 — Open questions / follow-ups / risks

1. ⭐⭐ **The wave's question is now sharp and it is not reach.** With reach closed and the settle
   neutral, the only thing between this substrate and a non-vacuous K5 is the **launch head's
   precision** (0.2303, `correct-and-distinct 0.92 of 4`, address-space coverage 0.7546 uncovered) —
   an address-side quantity. Spoke A's §13.1 lever (a **different code geometry at `d = 4`**:
   near-orthogonal / equiangular codes) is now the *only* live lever, and it is a φ change.
2. ⚠ **Should spoke B run?** My reading (⛔ not an adjudication): unchanged from spoke A's — every arm
   still reads chance, so a swap-differenced V3 curve is still a difference of two zeros. What the
   repair *does* buy spoke B is that the settle no longer destroys the launch geometry, so a
   `psi`-pooled read now starts from `0.2288`, not `0.0736`.
3. ⚠ **45/48 wells, not 48.** If a future cell needs a per-well guarantee, the ratio target must be
   set on the **p90 of the per-well distribution** (≈0.86 at `r = 0.60`), not the median — i.e.
   `r ≈ 0.5`. I did not do that because the task registered the median form.
4. ⚠ **The interval is only ~2.5× wide** (`0.25 ≲ ‖v‖ ≲ 0.64`) at this design point and both ends are
   measured. Any change to `N_a`, `w_frac`, `place_depth` or `d` moves both ends and the law must be
   **re-measured**, never inherited — that is exactly the mistake D3 made.
5. **Risk in how §6 gets quoted.** "The dividend went from −0.157 to −0.002" is true and is the
   headline of the repair — ⛔ **it must never be quoted without "and K5 is still at chance, and the
   settle adds nothing; it merely stopped subtracting"** in the same sentence.

## Proposed handover updates (for the Hub)

- **§3 config/CLI — NEW:** `chlu exp-c2w11-substrate --stages … reach …` (new stage, runs after
  `width`, before every scored stage). New module-level constants `PAYLOAD_RADIUS_GRID`,
  `REACH_RATIO_TARGET = 0.75`; new functions `stage_payload_reach`, `select_payload_radius`,
  `_directional_reach`. ⛔ **No `chlu/config.py` change** — `payload_radius` and
  `atom_payload_init_radius` already existed in `CatTestConfig`; the repair is a *selected operating
  point*, not a new knob.
- **§7 Known Issues — NEW (closed by measurement):** *the C2W11 read launches outside every basin
  (`‖v_j‖/capture = 1.172`)* → **CLOSED at `payload_radius = 0.60` (ratio 0.692)**. ⛔ The Known Issue
  it exposed should be **rewritten, not deleted**: the read is still not expressible, now provably for
  an **address-side** reason.
- **§7 Known Issues — NEW (open):** *C2W5 deviation D4 (`atom_payload_init_radius`) is **inert** under
  the placing write* (measured; live only under the gradient write).
- **§7 Known Issues — REWRITE:** *"the settle destroys 68 % of the launch's correct-well information"*
  must be **scoped to `‖v‖/capture > 1`**. At a closed ratio the dividend is **−0.0015 ± 0.0026**.
- **Registry/doctrine candidate (⭐⭐ the durable one):**
  **`k1_write_floor ≲ ‖v_j‖ < capture_radius ≲ min-well-spacing`** — a compositional family whose
  payload targets sit outside the basins that must capture them is **unreadable by construction**, and
  one whose targets crowd too close is **unwritable**. Both ends are measured
  (`[≈0.25, ≈0.64]` here). ⛔ Re-measure per design point; C2W5's D3 chose 1.0 *for reach reasons* and
  never measured that 1.0 was outside.
- **Doctrine candidate:** ⭐ *a median margin is not a per-well guarantee* — the reach wall lived for a
  wave inside a median, and even at a 25 % median margin 3 of 48 wells remain outside their basin.
  Every per-well condition ships with its per-well distribution.
- **`PREREG-C2W11.md` scoring input:** **Q3 (M6 dividend non-negative, 0.50)** — my spoke-A report
  scored it **REFUTED at −0.1567**; at the repaired operating point it is **−0.0015 ± 0.0026, 2 SE
  spanning zero**. ⇒ **Q3's spoke-A score is reach-scoped and needs re-adjudication.**
