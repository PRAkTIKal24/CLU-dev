# c2w8p3-gate-addr — experiment-engineer report

**Task + acceptance criterion (one line):** build **G-ADDR**, the addressability leg the C2W8 gate was
blind to, with designed negatives that prove it can fail, ship
`GATE-ADDR-VALIDATED.json` with a mechanically computed boolean, and run Ruling 3's compliance
counterfactual. **Status: DONE.** ⭐ **`gate_addr_validated = true`. Arm B FAILS G-ADDR 3/3 (designed
negative 1). Arm A FAILS G-ADDR 3/3. The attractor CAN move off the stored key — Ruling 3's compliance
ruling STANDS, no escalation.**

## ⚠ DOWNSTREAM RECONCILIATION LIST (protocol §5 corollary — needs an owner at the review that accepts this report)

1. ⛔⛔ **At `main @ 1eda6a0` ARM A COULD NOT RUN AT ALL** — `TypeError: unexpected keyword argument
   'overrides'`. Arm B's merge added the `overrides=` seam to `run_census_cell`; arm A's substituted
   store-config factory never accepted it, and neither merge re-ran the other. Fixed here
   (`cb13740`, behaviour-preserving) + regression test. **Every pass-3 spoke that touches arm A needs
   this commit.**
2. ⛔⛔ **The banked arm-A configuration is NOT the config default.** `atom_width_frac_spacing`
   defaults to **0.5** (the pilot cell); the banked census ran at **1.5**, passed on the CLI. I
   measured the default: it **does not clear the pass-2 gate** (G-DEC exactly at chance 0.0625 3/3,
   G-DRIFT ratio 0.61–0.71, depth 0.20 vs 0.61). **The spine will silently score the wrong store
   unless it sets 1.5.** Either change the default or make the arm's census refuse to run at a
   non-selected width.
3. ⭐⭐ **"Reads landing in no basin" (58/62/62 of 64, digit-identical pass 1 → pass 2) is a
   LAUNCH-POINT statistic, not a settle statistic.** `covered = min_j‖q0 − c_j‖ ≤ ½·min-sep` is
   computed on `q0` in `clu_system._read_diagnostics`. Same φ + same admitted codebook ⇒ same number,
   whatever the store does. **That is the mechanical explanation of the digit-identity the Advisor
   read as decisive**, and `n_never_read` inherits it. I did **not** change `covered` (out of my
   declared ownership, and it would move banked numbers silently). **The Hub owns the fix.**
4. ⚠ **The §4 scale guard, applied for the first time, fails on the RIG, not on G-ADDR — and the
   PASS-2 legs fail it harder.** Address-only rescaling ×0.8 moves arm A's A1 by −0.125, but also
   self-probe `acq` 0.4844→0.3203, G-DEC 0.1484→0.1094 and **G-DRIFT ratio ×3.8**. Cause measured, not
   argued: the **payload channel is absolute**. Co-scale it and A1 returns to **0.5000** and A3a to
   **−0.3984**, both to 4 dp. ⛔ **A strict reading of §4 unships G-DEC and G-DRIFT too. Hub call.**
5. ⭐ **`theta_att = 0.0000` degeneracy is live in the banked data:** arm B **seed 1** has
   `n_non_capturing = 0` ⇒ `theta_att = 0.0000` ⇒ `P = 0.8750` against 0.1250 / 0.0000 on its sibling
   seeds. **`P` is not comparable across seeds, let alone arms, without `n_non_capturing` beside it.**
   Now documented in the census's own docstring; **the scorecard sites need the qualifier.**
6. ⚠ **The Hub's registered Q2 is falsified, and so is my own competing band.** Q2 predicted arm A's
   A1 at **0.03–0.12**; measured **0.5000 / 0.4219 / 0.5156**. My band (0.10–0.45) caught 1 of 3.

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial:** **none — instrument construction + one compliance counterfactual.** ⛔ No claim cell, no
  performance number, no tier-ii / full-CLU / I2 verdict, **and no arm-race adjudication** (§A30.1:
  the race is VOID as a comparison and stays unadjudicated — nothing below ranks arm A against arm B).
- **Laundering control:** G-ADDR's **A3 leg IS a launder margin**. Every quotation below is
  **matched-ITEMS** (same keys, same queries, same φ); **matched-bytes is NOT met** —
  `clu_total_bytes / knn_launder_bytes` = **627×** (arm A) / **665×** (arm B) on the G-ADDR cue
  launder, and pass 1's **1 253×** stream-launder ratio travels unchanged.
- **Falsifies:** a G-ADDR that cannot fail its designed negatives does not ship. Ruling 3's
  counterfactual failing REVERSES the `atom_site_local_init` compliance ruling ⇒ escalate.
- **Does NOT falsify:** losing to the kNN launder **on the metric-native cue protocol** — 1-NN is the
  Bayes rule there; that is the metric-native-ceiling theorem, not news (this is why A3's criterion is
  "does not lose beyond 2 SE", ERRATA §1).
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ N94 discipline on every reading.

---

## 1. Acceptance checklist (task §Acceptance, mechanical)

| # | requirement | where | status |
|---|---|---|---|
| 1 | `GATE-ADDR-VALIDATED.json` with a **mechanically computed** `gate_addr_validated` | §3 | ✅ **true** |
| 2 | both designed negatives measured and green (arm B FAILS; planted-permutation ≈ 0) | §4 | ✅ arm B fails 3/3; permutation **A1 = 0.0000** |
| 3 | scale-only control reported; G-ADDR moves ≈ 0 under it | §6 | ✅ **for the leg** (0.0000 on every scale-covariant rescaling); ⛔ **the address-only rescaling moves the RIG** — reported, attributed, Hub call |
| 4 | Ruling-3 counterfactual reported with its margin, or escalated | §7 | ✅ `follow = 0.910` / **1.008** at arm A's kernel — **no escalation** |
| 5 | housekeeping 1–3 landed, item 1 with its cross-kernel test | §8 | ✅ |
| 6 | full suite green, count arithmetic stated with the checkout | §9 | ✅ (see §9) |
| 7 | reconciliation list in the first 10 lines; NOT-RUNs declared as NOT-RUNs | top / §10 | ✅ |

**Registration trail (both filed before the cells they govern):**
`.claude/outputs/c2w8p3-gate-addr/PREREG.md` (leg definitions, thresholds, predictions P1–P8) and
`…/ERRATA.md` §1 (A3's threshold, filed after the *positive* control and before any arm), §2 (A2's
definition), **§3 (⚠ explicitly POST-HOC: the scale movement and the two attribution cells, with their
predictions registered before they ran)**.

---

## 2. What G-ADDR is (definitions, all registered before measurement)

**Cue set — ground truth by construction, which is what the pass-2 instrument never had.** For each
live item `i`, `n_q` queries `q = c_i + κ_q · spacing_ref · ε`, `ε ~ N(0, I_d)` per coordinate, read
through the shipped `CluSystem.read`; the settled point is `q*` (`state.q_star`).
⭐ **`κ_q` is dimensionless** (registered 1.0) — §4's guard: `σ_q = 0.15` is *absolute* while addresses
are normalised to unit radius, so any leg in absolute units is movable by rescaling φ for free.
`spacing_ref` = the cell's own **measured** `median_nn_task1` (0.1407 / 0.1375 / 0.1468) — the same
number G-DRIFT is scored against ⇒ the cue reproduces the shipped σ_q to within 7 %.

| leg | definition | threshold |
|---|---|---|
| **A1 correct-basin** | `q*` resolves to **the QUERIED item** (address argmin, the shipped `_assign` rule) **AND** `‖q* − z_i‖ ≤ ρ_i`, `ρ_i` = that item's **measured** SC-6 capture radius (`ρ_i ≤ 0` ⇒ no basin ⇒ never correct) | `≥ max(4·chance, chance + 2 SE)` = **0.25** at n=16 |
| **A2 never-addressed** | live items with **zero** correct cue reads | `≤ 0.5` |
| **A3a cue margin** | store vs **kNN-in-φ launder**, same queries, **same decision rule** | `≥ −2 SE` (paired/McNemar) |
| **A3b stream margin** | `mean(read_acc − knn_acc)` over the census's held-out read events | `≥ −2 SE` (pooled binomial); **NOT-APPLICABLE (declared)** where no stream exists |

⛔ **"Some basin" is reported (`any_basin_rate`) and is NOT the leg.** That is precisely how 58/64
unassigned coexisted with a passing gate. The permutation control below shows the difference: `A1`
goes 1.0000 → 0.0000 while `any_basin_rate` stays **exactly 1.0000**.

⚠ **A3's criterion is "does not LOSE beyond 2 SE", not "beats"** (ERRATA §1, filed after the positive
control and before any arm). The cue is metric-native, so 1-NN over the stored keys is the **Bayes
rule**; `A3a > 0` measured **exactly 0.000** on a store that addresses perfectly ⇒ it is a leg that
cannot pass, the mirror of the defect this spoke exists to close. *Daylight above the launder is the
spine's question (§6 of the Hub's prereg), not this gate's.* Both margins are reported two-sided.

---

## 3. ⭐ THE DELIVERABLE — `GATE-ADDR-VALIDATED.json`

`.claude/outputs/c2w8p3-gate-addr/GATE-ADDR-VALIDATED.json`, **`gate_addr_validated = true`**,
computed as the AND of nine checks (verbatim from the file):

```
C_plus_positive_passes                                    true
N1_armB_banked_config_fails                               true
N1prime_narrow_wells_fail                                 true
N2_planted_permutation_scores_zero_and_fails              true
S_planted_scale_only_moves_A1_by_leq_0p05                 true
S_real_rig_scale_covariant_rescaling_moves_A1_by_leq_0p05 true
S_leg_machinery_is_exactly_scale_covariant                true
S_has_at_least_one_NON_SATURATED_real_rig_covariant_pair  true
R3_attractor_can_move_off_the_key                         true
```

The arm re-scores are **measurements** and enter the boolean only through N1 (arm B must fail).
⛔ The address-only rescaling result is carried in the same file under
`scale_control.rig_scale_noninvariance` + `scale_control.hub_decision_owed` — **not** hidden and
**not** silently absorbed (§6).

---

## 4. ⭐ THE DESIGNED CONTROLS — it can fail, and it can pass

`designed_controls.json`; every row pytest-asserted in `tests/test_gate_addr.py` (16 + 1 tests).
Planted rigs: `addr_dim = 8`, `n = 6` wells on orthogonal axes at radius 0.7, depth 1.0, unused groups
flattened, `spacing_ref = 0.10` declared, `κ_q = 1.0`, seed 0.

| control | A1 | A2 | A3a | `any_basin` | verdict |
|---|---|---|---|---|---|
| **C+ positive** (planted, addressable) | **1.0000** | 0.0000 | +0.0000 ± 0.0000 | 1.0000 | ✅ **PASSES** |
| **N2 planted permutation** (same store, same queries, wrong declared targets) | **0.0000** | 1.0000 | 0.0000 | **1.0000 — identical** | ✅ **FAILS** |
| **N1′ narrow wells** (retrievable at their own sites, unreachable from a cue) | **0.0000** | 1.0000 | 0.0000 | **0.0000** | ✅ **FAILS** |

⭐ **N2 is the defect class, isolated.** The two runs differ **only** in the declared target. The
pass-2-style "did the read land in *a* basin?" question returns **1.0000 for both**; A1 separates them
1.0000 vs 0.0000. **A gate reading `any_basin` cannot fail on the thing that matters; A1 can.**
⭐ **N1′ is arm B's blind spot in miniature**: `A1_voronoi = 1.0000` (the read resolves to the right
item) with `any_basin = 0.0000` (it lands in no basin at all).

### 4.1 ⭐⭐ DESIGNED NEGATIVE 1 — arm B's banked configuration, re-scored LIVE, 3 seeds

⛔ **It FAILS G-ADDR on every seed** (`gate_addr_pass = false` 3/3), which is the single most important
assertion in this task. Banked pass-2 legs **reproduced exactly** (G-CAP/G-DEC/G-DRIFT 3/3 pass,
`decode = 0.25` 3/3, self-probe `acq` 0.9844/0.9609/0.9375 — digit-identical to `census_armB.json`).

| seed | A1 | A2 | **A3a** | **A3b** | G-ADDR |
|---|---|---|---|---|---|
| 0 | 0.9297 ✅ | 0.0625 ✅ | **−0.0156** (2 SE 0.0221) ✅ | **−0.6094** (2 SE 0.1355) ❌ | ❌ **FAIL** |
| 1 | 0.9844 ✅ | 0.0000 ✅ | **−0.0078** (2 SE 0.0156) ✅ | **−0.6094** (2 SE 0.1388) ❌ | ❌ **FAIL** |
| 2 | 0.8750 ✅ | 0.1250 ✅ | **−0.0625** (2 SE 0.0442) ❌ | **−0.5625** (2 SE 0.1452) ❌ | ❌ **FAIL** |

⭐ **And the failure is informative, not incidental.** Arm B **passes A1 and A2 handsomely** — its
wells *are* reachable from a cue near their own site. What kills it is **A3b: on real held-out data it
loses to its own kNN-in-φ launder by 0.56–0.61, 4.1–4.5× the pooled 2 SE.** The store is addressable
by a query it already knows and not by a query from the world. ⛔ This is a measurement of a banked
arm, **not** an adjudication of the race.

### 4.2 Arm A's banked configuration, re-scored LIVE, 3 seeds (the Hub's Q2 cell)

Banked pass-2 legs **reproduced to the digit** (G-CAP 1.000 / 0.938 / 0.938, `decode`
0.1484 / 0.1641 / 0.1406, G-DRIFT ratio 0.0071 / 0.0474 / 0.0102, self-probe `acq`
0.4844 / 0.4141 / 0.4922 — every one matching `capture_armA.json`; the pass-2 gate passes 3/3 here
too), and
**A3b reproduces to the digit: −0.359375 / −0.500000 / −0.203125, mean −0.354** = the Advisor's −0.354.
✅ **My registered P5 (−0.354 ± 0.05) holds exactly.**

| seed | A1 | A2 | **A3a** | **A3b** | G-ADDR |
|---|---|---|---|---|---|
| 0 | **0.5000** ✅ | 0.2500 ✅ | **−0.3984** (2 SE 0.1220) ❌ | **−0.3594** (2 SE 0.1648) ❌ | ❌ **FAIL** |
| 1 | **0.4219** ✅ | 0.3750 ✅ | **−0.4375** (2 SE 0.1250) ❌ | **−0.5000** (2 SE 0.1531) ❌ | ❌ **FAIL** |
| 2 | **0.5156** ✅ | 0.2500 ✅ | **−0.2734** (2 SE 0.1094) ❌ | **−0.2031** (2 SE 0.1666) ❌ | ❌ **FAIL** |

⛔ **Q2 (Hub, 0.03–0.12) is FALSIFIED by a factor 4–5**, and my own competing band (0.10–0.45) caught
only seed 1. **Arm A's wells ARE reachable from a cue about half the time** — the store is not
address-blind at the sites; it is beaten by a 576-byte table on both query distributions
(cue: 0.50 vs launder **0.8984**; stream: 0.34 vs 0.70). ⭐ **The wave's framing "queries do not reach
the wells" is too strong for arm A once the settle is measured instead of the launch point:** the
correct statement is **"the reads reach the wells at ~half rate and lose to nearest-key indexing on
every protocol tested."**

⚠ **Matching, on every number above: matched-ITEMS.** Byte ratio **627×** (arm A) / **665×** (arm B)
against the G-ADDR cue launder; pass 1's **1 253×** stream-launder ratio unchanged. **No performance
claim is made at any of these ratios.**

---

## 5. The A2 finding — the banked never-read counter is a launch-point statistic

| arm / seed | **A2** (settle-side, the leg) | banked `frac_never_read` | banked `n_unassigned` |
|---|---|---|---|
| armA 0 / 1 / 2 | **0.2500 / 0.3750 / 0.2500** | 0.9375 / 0.9375 / 0.8750 | **58 / 62 / 62** |
| armB 0 / 1 / 2 | **0.0625 / 0.0000 / 0.1250** | 1.0000 / 1.0000 / 1.0000 | **64 / 64 / 64** |

`n_unassigned` reproduces pass 1's and pass 2's numbers **to the digit** — because
`covered = min_j‖q0 − c_j‖ ≤ ½·min-sep(centers)` is evaluated at the **launch point**, so at fixed φ
and a fixed admitted codebook it is a constant of the query distribution. **The digit-identity between
pass 1 and pass 2 arm A is explained mechanically and is not evidence that the physics did not move.**
The settle-side A2 tells a different story on the same cells (0.25–0.375, not 0.94). Reconciliation
item 3.

---

## 6. ⭐ THE SCALE-INVARIANCE GUARD (§4) — the leg is exactly covariant; **the RIG is not**

Registered: `|ΔA1| ≤ 0.05` under "identical φ, address scale × a declared constant" (Hub **Q8**, 0.90).

**(a) The leg's own machinery is exactly scale-covariant** — measured, on arm A seed 0, whose codebook
*is* the scaled φ keys:

| quantity | `a = 1.0` | `a = 0.8` | `a = 1.25` |
|---|---|---|---|
| A3a **launder** rate (the leg's own comparator) | **0.8984375** | **0.8984375** | **0.8984375** |
| `cue_sigma / codebook_spacing` | 0.3163310 | 0.3163310 | 0.3163310 |

**(b) On every rescaling that is genuinely scale-covariant, ΔA1 = 0.0000:**

| rig | pair | ΔA1 |
|---|---|---|
| planted C+ | ×0.8 and ×1.25 | **0.0000** (1.0000 → 1.0000) |
| **arm A real rig**, ×0.8 **with the payload co-scaled** (`payload_scale 9 → 11.25`) | non-saturated | **0.0000** (0.5000 → **0.5000**), and **ΔA3a = 0.0000** (−0.3984 → −0.3984) |
| arm B real rig, ×0.8 | non-saturated | **0.0000** (0.9297 → 0.9297) |

**(c) ⛔ Under an ADDRESS-ONLY rescaling the arm-A rig moves, and it takes the pass-2 legs with it:**

| quantity (arm A seed 0) | `a = 0.8` | `a = 1.0` | `a = 1.25` |
|---|---|---|---|
| **A1** | 0.3750 | **0.5000** | 0.6328 |
| self-probe `acq` | 0.3203 | **0.4844** | (n/a) |
| **G-DEC** `decode` (pass-2 leg) | 0.1094 | **0.1484** | — |
| **G-DRIFT** ratio (pass-2 leg) | 0.0273 | **0.0071** | — (**×3.8**) |
| G-CAP median radius | 0.3359 | 0.4297 | — (co-scales ≈ ×0.8 ✓) |

**Cause, measured not argued.** The rig's **payload channel is absolute**: sites are `(c_i·a | a_i)`
with `|a_i| ≤ 0.5` fixed while the compact support `R = 2.5 s` co-scales. At `a = 1.0`,
`R = 0.528 > 0.5`; at `a = 0.8`, `R = 0.422 < 0.5` — the rescaling walks arm A back across **its own
payload wall** (arm A report §5). ⭐ **The attribution cell registered in ERRATA §3 before it ran
predicted "A1 returns to 0.5000 ± 0.08" and measured 0.5000 — exact to 4 dp.**

⛔ **Hub decision owed (reconciliation item 4).** A strict reading of §4 — *"if rescaling moves
G-ADDR, the leg measures the scale and it does not ship"* — unships G-ADDR. **On identical evidence it
unships G-DEC and G-DRIFT, which move more.** My reading, applied in the JSON and flagged as
reversible: the guard has done its job by **finding a non-covariance in the substrate**, and the leg
itself is clean. The spine should either co-scale the payload channel or declare the address:payload
aspect ratio as a fixed rig constant.

---

## 7. ⭐ RULING 3's COUNTERFACTUAL — the attractor CAN move off the stored key. **NO ESCALATION.**

Construction (`well_lifecycle.displaced_write_counterfactual`), deliberately the harshest honest one:
admit at `c`; run **`atom_site_local_init` at `c`** (the lever under scrutiny, at full strength, atoms
placed exactly on the stored key); then run the **shipped** learned write with its target displaced to
`c + δ`; then relax **from `c`** and see where the attractor is.
Registered statistic (PREREG §3 P6): `follow = ‖q* − c‖ / ‖δ‖`, **PASS iff ≥ 0.5**, point prediction
≥ 0.80. `addr_dim = 8`, `‖δ‖` orthogonal to the key, `write_steps = 200`, seed 0.

| cell | `‖δ‖` | site-local init | **follow** | moved off key | residual to the displaced target |
|---|---|---|---|---|---|
| ⭐ **arm A's kernel** (`wendland`, cutoff 2.5, `s = 0.20`) | 0.30 | **ON** | **1.0079** | 0.3024 | **0.0064** |
| gaussian | 0.30 | **ON** | **0.9104** | 0.2731 | 0.0269 |
| gaussian | 0.60 | ON | 0.6860 | 0.4116 | 0.7276 |
| gaussian | 0.30 | OFF | 1.3720 | 0.4116 | 0.5093 |
| gaussian | 0.60 | OFF | 0.6860 | 0.4116 | 0.7276 |

⭐ **At arm A's own kernel with the lever ON, the settled point lands 0.0064 from the displaced
minimum and 0.3024 from the stored key.** The near-zero site drift arm A measured is an **outcome of
the write objective**, not an algebraic identity. ✅ **Hub Q7 (prior 0.85) and my P6 both hold.**
✅ **`atom_site_local_init` remains COMPLIANT; §A30.3's reversal condition is NOT met; I am not
escalating.** ⚠ Honest limit: at `‖δ‖ = 0.60` the move saturates at 0.4116 (`follow = 0.686`) — the
write can drag the attractor a bounded distance, which is a *capacity* statement, not a pinning one.

---

## 8. Census housekeeping (items 1–3), now owned by this module

1. ⭐ **`own_foreign_site_depth` no longer hard-codes the Gaussian kernel.** It reads the store's own
   `kernel`/`kernel_cutoff` (and `axis_width_scale`) through a float64 mirror of `atom_profile`, pinned
   against the shipped profile for **all three kernels** (`test_numpy_atom_profile_mirrors_the_shipped_one_for_every_kernel`
   — adding a kernel without mirroring it fails the test). **Cross-kernel test**
   (`test_own_foreign_reads_the_compact_kernel_and_the_gaussian_form_over_reads`): at a wendland store
   with `R = 0.75 < 1.0` site separation the repaired foreign leg is **exactly 0.0** while the legacy
   Gaussian form reports a fictitious tail. Continuity pinned: under `gaussian` the change is 3e-8
   relative (the store's own `1e-9` epsilon vs the old `1e-12`) and **no reported digit moves**.
   **Measured over-read on the real arm A** (P8 predicted ≥ 2×, massively confirmed):

   | seed | foreign median, legacy (banked) | foreign median, **repaired** | over-read |
   |---|---|---|---|
   | 0 | 0.1233 | **0.00091** | **136×** |
   | 1 | 0.3532 | **0.02359** | **15×** |
   | 2 | 0.3407 | **0.00464** | **73×** |

   own median moves 0.770/0.798/0.731 → **0.584/0.633/0.555** (1.3×). ⛔ Direction as predicted (up);
   the gate never read this column, so **no pass-2 verdict changes** — but every quoted own/foreign
   number for arm A should be re-quoted from the repaired estimator.
2. **K7's two instrument properties are now in the census's own docs** (`capture_radii` docstring): the
   **`tol / expansion_rate` positive floor** (a barely-moving site reads positive with no basin;
   K7-2 measured 0.001953 where 0.0 was predicted) and the **confinement-minimum false positive**
   (≈`r_hi`; K7-5 measured 0.99902), with the operating-point caveat that `tol = σ_q` here.
3. **`theta_att`'s arm-dependent dynamic range is documented** (`measure_theta_att` docstring) with the
   binding consequence: `P` is not cross-arm comparable without `n_non_capturing`. **Live example from
   this run:** arm B seed 1 has `n_non_capturing = 0` ⇒ `theta_att = 0.0000` ⇒ **`P = 0.8750`** vs
   0.1250 / 0.0000 on seeds 0 / 2.

---

## 9. How I verified (commands + observed output)

Main venv (`/Users/user/Desktop/CHLU/.venv`, **JAX 0.9.0**, no worktree `uv sync`) with `PYTHONPATH`
set to the worktree `/Users/user/Desktop/CHLU-c2w8p3a` (protocol §4's preferred recipe).

```
python -m pytest tests/test_gate_addr.py -q -p no:randomly --no-cov   -> 17 passed
python .claude/scratch/c2w8p3-gate-addr/controls.py <out>/designed_controls.json
python -m chlu.experiments.exp_capture_armA --quick                   -> smoke, 47 s
python .claude/scratch/c2w8p3-gate-addr/rescore.py armA {0,1,2} 1.0 <out> 1.5
python .claude/scratch/c2w8p3-gate-addr/rescore.py armB {0,1,2} 1.0 <out>
python .claude/scratch/c2w8p3-gate-addr/rescore.py armA 0 {0.8,1.25} <out> 1.5
python .claude/scratch/c2w8p3-gate-addr/rescore.py armA 0 0.8 <out> 1.5 11.25   (S-pay)
python .claude/scratch/c2w8p3-gate-addr/make_validated.py <out>/GATE-ADDR-VALIDATED.json
python -m ruff check chlu/ tests/test_gate_addr.py                    -> All checks passed!
python -m pytest -q -p no:randomly --no-cov                           -> 1521 passed, 0 failed (2721.88 s)
```
Cells were run as parallel processes (independent, each carrying its own seed); per-cell wall
**1075–1281 s** (arm A) / **966–1281 s** (arm B).

**Full-suite arithmetic (⚠ checkout-dependent — stated with the checkout).** On the **worktree
`/Users/user/Desktop/CHLU-c2w8p3a`** (fresh worktree, MAIN venv reused, JAX 0.9.0):

| | collected | result |
|---|---|---|
| branch `c2w8p3-gate-addr` | **1521** | **1521 passed, 0 failed, 0 skipped** (2721.88 s) |
| same checkout, `--ignore=tests/test_gate_addr.py` (= the base) | **1504** | — |
| **`tests/test_gate_addr.py` (new)** | **17** | 17 passed |

**1504 + 17 = 1521 ✓.** ⚠ The base here is **1504**, not the directive's 1445 / the pass-2 branch's
1458: this is a **fresh worktree** (three modules `importorskip`/`skipif` on cached datasets) *and*
both pass-2 arms' test files have landed on `main` since. **No pre-existing test was modified,
skipped or deleted** — the delta is exactly my new file.

### Flag-provenance table (every non-default flag in effect)

| flag | arm A cells | arm B cells |
|---|---|---|
| commit | `8146779` (branch `c2w8p3-gate-addr`, base `main @ 1eda6a0`) | same |
| seeds | 0, 1, 2 (+ seed 0 for the scale/attribution cells) | 0, 1, 2 (+ seed 0 scale) |
| `atom_kernel` / `atom_kernel_cutoff` | **`wendland`** / 2.5 | gaussian (default) |
| `atom_width` | **0.21107 / 0.20631 / 0.22025** = `1.5 × median_nn_task1` ⚠ **set explicitly; the config default 0.5 is NOT the banked arm** | store default |
| `atom_site_local_init` / `radius` | **True** / = `atom_width` | False |
| `emission_head` | False | **True** (`emission_head_bytes` 22 316) |
| `addr_dim` / `payload_dim` / `n_atoms` / `capacity` / `well_budget` | 8 / 1 / 8192 / 16 / 8 | same |
| `d_safe_override` | 0.12383 / 0.12103 / 0.12921 | same |
| `leak` / `stage_lifetimes` / `permanent_per_task` | 0.02 / True / 1 | same |
| `write_steps` / `read_steps` / `address_steps` | 300 / 800 / 400 | same |
| `query_sigma` (σ_q) / `payload_tol` / `n_query_per_item` | 0.15 / 0.1 / 8 | same |
| **`gaddr_kappa_q` / `gaddr_n_query_per_item` / `run_gate_addr`** | **1.0 / 8 / True** (new, additive) | same |
| **`addr_scale_mult`** | **1.0** (0.8 / 1.25 on the declared scale cells) | 1.0 (0.8 on one cell) |
| `payload_scale` | 9.0 (**11.25** on the S-pay attribution cell only) | 9.0 |
| `kinetic_mode` / `dt` / `gamma_address` / `gamma_read` | `newtonian_learned` / 0.05 / 0.05 / 0.02 | same |
| φ | `pca`, regime `task1_only`, dataset **mnist**, scale fixed on the fit pool | same |
| capture instrument | `capture_dirs` 16, `capture_bisect_steps` 8 (pass-1 values, untouched) | same |
| `promotable` | **False** (inherited: `phi_dim = addr_dim = 8` below the CL entry's binding 16) | same |

Planted-control provenance: `addr_dim = 8`, `n = 6`, depth 1.0, `atom_width` 0.30 (0.03 for N1′),
radius 0.7, `spacing_ref = 0.10` **declared** (no φ pool exists on a planted rig), `κ_q = 1.0`,
`n_q = 4`/item, `capture_dirs = 8`, `bisect_steps = 6`, seed 0, unused groups flattened.

---

## 10. Declared NOT-RUNs (⛔ never nulls)

- ⛔ **The arm-A-vs-arm-B RACE ADJUDICATION — VOID as a comparison (§A30.1) and NOT made here.** Both
  arms fail G-ADDR; no ranking of one against the other appears anywhere above.
- **merge / prune / depth restoration / every §2.7 claim cell** — not built, not in scope.
- **monitor #3's never-refusing admission gate** — still open, **NOT fixed** (refusal rate 0.000
  reproduced on every cell); it is pass 1's defect and not mine to fix silently.
- **`covered`'s launch-point definition** — diagnosed, **NOT changed** (outside my declared ownership;
  changing it moves banked numbers).
- The **φ_dim → addr_dim projection** (wt2), the **geometry precondition** (wt2), the **spine** (wt3).
- Any **tier-ii / full-CLU / I2 verdict**; any **performance claim**; any **paper number**.
- **G-ADDR on a strong-φ / CIFAR rig** — pass 3's spine, not this spoke; every number above is the
  **MNIST PCA d=8** census rig and ⛔ **is not a baseline for any pass-3 CIFAR number** (ERRATA-PASS3 §1 R1).
- **Seed cut:** the scale-only and attribution cells ran on **seed 0 only** (declared), 3 seeds on the
  main re-scores.

---

## 11. Git footprint

- **Branch** `c2w8p3-gate-addr`, **worktree** `../CHLU-c2w8p3a`, **base `main @ 1eda6a0`** (named
  explicitly; the shared checkout is occupied by the live `pilot-ttt-nan-and-d5-wiring` spoke and was
  never touched).
- **Commits** (both `Co-Authored-By: Claude`):
  - `cb13740` `[experiment-engineer] fix: arm A could not run AT ALL at main @ 1eda6a0` —
    `chlu/experiments/exp_capture_armA.py` (3 lines).
  - `8146779` `[experiment-engineer] build G-ADDR: the addressability leg the C2W8 gate was blind to`
    — `chlu/core/well_lifecycle.py`, `chlu/experiments/exp_well_lifecycle.py`, `chlu/config.py`
    (additive only), `tests/test_gate_addr.py` (new).
- **Files touched:** exactly the five above. ⛔ Untouched as declared: `chlu/core/emission_head.py`,
  `chlu/core/memory_potentials.py`, `chlu/experiments/exp_capture_armB.py`, `scripts/csf3/`,
  `chlu/training/train_cluformer.py`, `chlu/core/blocks.py`, `chlu/experiments/exp_cluformer_pilot.py`,
  `chlu/experiments/{exp_cl_entry,phi_encoders,exp_phi_read_in}.py`, every C2W6/C2W7 file.
- **Rebase onto the named base `main`:** `git rebase main` → *"Current branch c2w8p3-gate-addr is up to date"* (no-op; the base has not moved). Branch ref verified **from the main repo**: `git -C /Users/user/Desktop/CHLU log --oneline main..c2w8p3-gate-addr` shows both commits. **Not pushed; no PR.** Worktree `../CHLU-c2w8p3a` **removed after** the §3.2 verification (clean tree, both commits visible from the main repo), freeing a slot under the ≤3 cap for the gated spine spoke; the branch `c2w8p3-gate-addr` is intact in the main repo and can be checked out or re-worktree'd at will.
- **Artifacts** (all under `.claude/`): `outputs/c2w8p3-gate-addr/{PREREG.md, ERRATA.md,
  GATE-ADDR-VALIDATED.json, designed_controls.json, rescore/*.json, attribution/*.json,
  rescore_v0/*.json (the config-default mis-run, kept as evidence for reconciliation item 2),
  full_suite.log, controls.py, rescore.py, make_validated.py}`.

---

## 12. Open questions / risks

1. **⛔ Hub call owed on §4 (reconciliation item 4).** Strict reading unships G-ADDR *and* G-DEC/G-DRIFT.
   I shipped on the leg-level reading and flagged it as reversible; the numbers for either decision are
   in `GATE-ADDR-VALIDATED.json`.
2. **A3b is the leg doing the work on both arms** (arm A −0.20…−0.50, arm B −0.56…−0.61), and it is the
   only leg that uses real held-out data. On the spine's CIFAR rig its launder must be re-declared and
   re-ledgered (`ERRATA-PASS3 §1 R2`: the launder reads the PROJECTED φ).
3. **A1's basin test uses full store space** (address + payload) against a full-space capture radius.
   That is consistent with the SC-6 instrument, but it makes A1 sensitive to the payload channel —
   which is exactly the sensitivity §6 exposed. An address-space-only variant is a one-line change if
   the Hub prefers it; I did **not** make that choice unilaterally after seeing the scale result.
4. **`κ_q = 1.0` on `spacing_ref = median_nn_task1`** puts the cue at 0.32 (arm A) / 0.12 (arm B) of the
   *codebook* spacing. Both ratios are reported per cell. A spine at a different `d` will land at a
   different ratio and must quote it.
5. The **positive control saturates** (A1 = 1.0000), so it proves the gate *can* pass but does not
   calibrate its dynamic range. Arm A's 0.42–0.52 supplies the mid-range evidence.

---

## Proposed handover updates (for the Hub)

- **§7-CURRENT, new item — `main @ 1eda6a0` ships an arm A that cannot run.** `exp_capture_armA`
  raises `TypeError: unexpected keyword argument 'overrides'` against `run_census_cell`'s pass-2 seam.
  **RESOLVED on `c2w8p3-gate-addr` (`cb13740`).** Root cause pattern worth recording: two arms merged
  independently, neither merge re-ran the other.
- **§7-CURRENT, new item — `ExperimentCaptureArmAConfig.atom_width_frac_spacing` defaults to 0.5, but
  the banked pass-2 census is 1.5.** Measured at the default: G-DEC exactly at chance 3/3, G-DRIFT
  0.61–0.71, depth 0.20. A caller using defaults silently scores a store that does not clear pass 2.
  Evidence: `.claude/outputs/c2w8p3-gate-addr/rescore_v0/`.
- **§7-CURRENT, new item — `covered` (hence `n_unassigned`, hence `n_never_read`) is a LAUNCH-POINT
  test.** Explains the pass-1/pass-2 digit-identity (58/62/62) that §A29.2 reads as decisive. OPEN, no
  owner; `clu_system._read_diagnostics`.
- **§3 config table:** `ExperimentWellLifecycleConfig` gains `run_gate_addr = True`,
  `gaddr_kappa_q = 1.0`, `gaddr_n_query_per_item = 8`, `addr_scale_mult = 1.0` (all additive;
  `addr_scale_mult = 1.0` and `run_gate_addr = False` reproduce the pass-1/2 cell exactly).
- **Instrument doctrine:** `chlu/core/well_lifecycle.py` is no longer "the census" — it carries
  **G-ADDR** and Ruling 3's counterfactual. Test file `tests/test_gate_addr.py` (17 tests).
- **§A29.7 line "the `own_foreign_site_depth` instrument hard-codes the Gaussian kernel … needs an
  owner" is DISCHARGED** (repaired + cross-kernel test), and arm A's own/foreign numbers should be
  re-quoted: the legacy estimator **over-read the foreign leg by 15–136×**.
- **§A29.2's quotable form needs softening for arm A:** "reads landing in no basin 58/62/62" is a
  launch-point statistic; the settle-side measurement on the same cells is **A2 = 0.25–0.375** and
  **A1 = 0.42–0.52**. The defensible claim is *"arm A's reads reach the correct well about half the
  time and lose to nearest-key indexing on every protocol tested"*, not *"queries do not reach them"*.
- **Registered-prediction ledger:** Q1 ✅ (arm B fails 3/3) · **Q2 ❌ FALSIFIED** (0.03–0.12 predicted,
  0.42–0.52 measured; my competing 0.10–0.45 also mostly missed) · Q7 ✅ (`follow` 0.91–1.01) ·
  Q8 ✅ **for the leg**, ⛔ **not for the rig under address-only rescaling** (Hub call owed).
