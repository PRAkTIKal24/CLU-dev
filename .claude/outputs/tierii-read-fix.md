# tierii-read-fix — experiment-engineer report

**Task + acceptance criterion (one line):** build the §A20.3(b)–(f) read-protocol iteration —
K0 adjudicated first, the multi-well read, the k-particle head with its four guards each shown
firing on a designed negative, consolidate-to-budget + trash-region pruning, learned `p₀` — and
**re-run the organizer swap against the new read**, `OD_min` per reader × seed, `S_eff` in band or
labelled COLLAPSED, tests green.
**Status: done** (all deliverables built, measured and committed; **two of the registered
falsifiers do not clear and are reported as negatives, not as verdicts** — see the one-line verdict).

> ## ⛔ THE ONE-LINE VERDICT
> **The read protocol's ADDRESSING is repaired and the addressing failure that killed C2W5 is gone:
> `K0` goes `0.0477 → 1.0000`, distinct wells occupied `2.20 → 11.82` launched / `11.27` settled,
> `S_eff` goes `34–51` (COLLAPSED) → `16.00` (exactly `S`, in band). ⛔ **The read's EXPRESSIVITY is
> not repaired**: exact-set occupancy is `0.0023 ± 0.0047` (R1's clear bar 0.02), every arm still
> scores `≈ 0` (`OD_min = −0.0008 ± 0.0016`, a **vacuous TIE**), and **guard 1 FIRES —
> `read − live launder = −0.0016 ± 0.0019`: the store adds nothing over its own launches.**
> **The named, measured next blocker is CARDINALITY, not addressability:** the read visits ~11.3
> wells and has no mechanism that commits to exactly `F = 4`, and the launch's own top-`F` ranking is
> a *better* set estimator than the settled occupancy's.

## ⛔ RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary, in the first 10 lines)
1. **The `0.272` launch ceiling is `P=4`-mean-specific and must be scoped wherever it is quoted.**
   Measured here under a **one-draw** query-noise model: `d=4` → **0.072**, `d=8` → **0.695**,
   `d=16` → 0.982, `d=24` → 0.992. ⭐ At `d = 4` the ceiling is **1.4× the arm bar (0.0507)** — the
   registered cell was *information-starved*, which is a second, independent reason C2W5's arms read 0.
2. **`orgdiv-cat-test` §1.2's "the settle destroys 27 % of the set information" is read-protocol
   specific.** Under this read the raw destruction is **4.6 %** and the descent gate turns it into a
   net **+0.076** over the launch. The blanket sentence must be scoped to the `P=4` read.
3. **`orgdiv-cat-test` §5.2's COLLAPSED verdict is read-protocol specific.** `S_eff` = **16.00 ±
   0.00** here (band [8, 16]) vs 34.1/51.2/36.6 there. Allocation collapse was a property of the
   refuted read, not of the physics arm.
4. **A new program-wide hazard (see §9):** reverse-mode AD through an un-`checkpoint`-ed settle
   **OOM-kills the process silently — exit code 0, no traceback, no partial output.** One full run
   was lost to it. Any future differentiate-through-the-settle work must `jax.checkpoint` the body.
5. **`FROZEN-interfaces.md` reader-param row needs a `(d, m)` column.** Measured at `d=8, m=8`:
   `sum_linear` **136**, `well_table` **72**, `mlp` **108**, `knn` **0**, `soft_well_table` **72**.
   `orgdiv-null-arms` recon-2's 72/92 were `d=4` numbers; only `well_table` is `d`-independent.
6. **§7.28 ruler, new datum:** `s = 0.2879` at (`d=8, m=8, a=32`, payload radius 0.5),
   confinement-subtracted, `R² = 0.9986` — and it is a **fixed point** of the placement loop
   (`sep = 2.7 s` re-measures the same `s` at two different `sep`). `d/s = 2.700–2.886` over 5 seeds.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echoed before the first result
- **Dial:** TIER ii — the organization dividend, **read-protocol iteration**.
- **Control:** the **ORGANIZER SWAP**, re-run against the new read under the same launch/read budget
  and the same `k` (on every arm's byte ledger, guard 4). Null = `null_arms.n1_gradient_placed`
  (`N1′`): the *same store parameterisation*, the *same objective*, the *bit-identical launch
  points*, a **static** softmax read, **no dynamics anywhere**.
- **Falsifies:** the re-registered falsifiers of `PREREG.md` §3 (K0, R1, R2, F1, F2, G1–G4, `S_eff`),
  each with a sign/threshold/seed count filed before the harness ran.
- **Does NOT falsify:** N1-still-composing-zero (the nulls' failure is not my claim); anything about
  tier i/iii.

---

# 1. ⭐ K0 — THE PRE-CONDITION, ADJUDICATED FIRST (store-free, before the build)

`P(≥ F distinct wells reachable)` from the launch geometry alone. **Bar 0.90.** 5 seeds, 2 560
unseen queries, `.claude/scratch/tierii-read-fix/k0_design{,2,3}.py` → `k0_design*.json`, and
re-run inside the harness (`stage k0`).

| launch protocol | `d` | `k` | distinct wells | **K0 = P(≥F)** | precision | K0b = P(A ⊆ occ) |
|---|---|---|---|---|---|---|
| **shipped, REFUTED** (`P=4` frozen offsets) — at its own registered `d` | 4 | 4 | **2.2023** | **0.0477** | **0.4053** | — |
| the same, at the new `d` | 8 | 4 | 2.0695 | 0.0383 | 0.6242 | — |
| ⭐ **the new head, designed init (REGISTERED)** | **8** | **12** | ⭐ **11.8156** | ⭐ **1.0000** | 0.2742 | **0.3844** |
| the new head (declared NOT-RUN as an arm) | 8 | 16 | 15.91 | 1.000 | 0.221 | 0.581 |

⭐ **K0: PASS (1.0000 vs bar 0.90).** The instrument reproduces the cell it refutes to 3 digits
(`orgdiv-null-arms` §3 published 2.202 / 0.050 / 0.4106 / exact-set 0.0000; measured here 2.2023 /
0.0477 / 0.4053 / 0.0000) — asserted in `tests/test_multiwell_read.py`.

⛔ **K0 forced the one registered deviation that matters (D6, `d = 4 → 8`),** and the argument is a
measurement, not a preference: the out-of-class combinatorial ceiling on the *same noisy code* is
**0.072 at `d = 4`** against an arm bar of `chance + 0.05 = 0.0507`. ⭐ **No read, however good, has
room to clear the bar by `> 0.05` at `d = 4`.** At `d = 8` the ceiling is **0.695** while the
rule-2/K4 leak stays at `R² = 0.159` (`d = 16` → 0.325, `d = 24` → 0.654). `d` is SP-2's own swept
axis, so this is a move *along* a registered axis; every arm gets the same `d`.

---

# 2. Flag provenance (mandatory — every quantitative result in this report)

Commits `201d250` · `ea46ce5` · `bcefed7` on
`agent/experiment-engineer/tierii-read-fix`, base local `main @ 9b2d4db`, worktree
`../CHLU-tierii-read-fix`, **main venv reused** (protocol §4 w6 lesson): **jax 0.9.0, equinox
0.13.4, numpy 2.4.1, float32**. **Seeds 0–4** everywhere except `levers` (0–2, declared).

| flag | value | note |
|---|---|---|
| `n_wells / f_subset / n_items` | 32 / 4 / 128 | registered design point, unchanged |
| **`n_unseen`** | **256** | ⚠ reduced from 512 (compute); 1 280 scored queries per arm |
| `atoms_per_well a` / `payload_dim m` | **32** / **8** | carried as MEASURED constraints (K1/K2b) |
| **`addr_dim d`** | **8** | ⚠ **deviation D6**, argued §1 |
| **`payload_radius` / `atom_payload_init_radius`** | **0.5** / **0.5** | ⚠ **deviation D10**, argued §5 (basin reach; the metric is scale-invariant in this radius) |
| **`s_measured`** | **0.2879** | ⭐ measured on the written store, confinement-subtracted (§7.28); a fixed point of the placement loop |
| `target_ds` → measured `d/s` | 2.7 → **2.700 / 2.784 / 2.786 / 2.886 / 2.747** | ✅ all 5 seeds inside the registered soft-certificate band [2.5, 2.9] |
| `min_sep` / `sep` | 0.7773 | |
| `depth_ratio` | 3.0 (alternate wells) | ablated in `levers` |
| **`k_particles`** | **12** | ⭐ CAPACITY — on every arm's ledger; `k = 16/24` declared |
| head `tau / kappa / rho / gain` | 0.002 / 2.0 / 0.25 / 1.0 | designed init (K0 measured at exactly these) |
| head `conf_b / conf_w` | 0.25 / **0.05** | overlap→gate; ⛔ tuned on the **SEEN** split only |
| head `log_mass_conf/unconf` · `gamma_mult_conf/unconf` · `p0_gain` | 0.0/1.5 · 1.0/4.0 · 1.0 | |
| `occ_tau` / `dedupe` / `payload_ref` / `descent_gate` | 0.25 / **noisy_or** / 0.5 / True | |
| **query noise** | `σ_q = 0.15`, **ONE draw per query** | ⚠ **deviation D7** — strictly LESS launch information than the refuted `P` i.i.d. draws |
| read budget | **400 + 800** Verlet steps, `dt = 0.05` | every γ/read statement is budget-scoped |
| `gamma_address / gamma_read` | 0.05 / 0.02 | one claim cell (γ axis is a declared NOT-RUN) |
| `kinetic_mode` | `newtonian_learned`, per-particle `mass_override` | shipped Prop-6 per-address mass |
| **organizer (physics)** | 60 Adam steps @ 3e-3, batch 16, **through the settle**, training budget **60 + 120** | ⚠ read-budget-scoped; loss 2.53–2.90 → 1.35–1.73 |
| **organizer (null N1′)** | **400** Adam steps @ 1e-3, `τ=1.0`, `init="written"`, static softmax | ⚠ the null gets **6.7× more optimiser steps** — conservative for a physics claim, declared |
| `tol` | `0.25 × RMS‖y − ȳ_seen‖` = **0.2338** | **chance = 0.0000** (constant predictor, `m=8`) |
| ψ | ⛔ **SHIPPED ψ on every arm**; `psi-payload-residual` had not landed at PREREG filing and landed mid-run (see §12) | Advisor amendment A2 honoured; **no mid-task hot-swap** |
| bytes | store 73 728 B · head **556 B** · φ 1 152 B · **total 75 436 B**, identical on both arms | `read_flops/query` **2.507e8**, identical on both arms |

---

# 3. ⭐⭐ THE ORGANIZER SWAP, RE-RUN AGAINST THE NEW READ (5 seeds, `Q_unseen`, exact-set)

`chance = 0.0000` · bar `chance + 0.05 = 0.05` · clears iff `mean − 2 SE > bar`.

| reader | params | **physics** | **null `N1′`** | **live launder `L_a`** | **`OD(R)`** |
|---|---|---|---|---|---|
| `sum_linear` | 136 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `well_table` (hard) | 72 | 0.0008 ± 0.0016 | 0.0000 ± 0.0000 | 0.0008 ± 0.0016 | +0.0008 ± 0.0016 |
| `knn` | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0008 ± 0.0016 | 0.0000 ± 0.0000 |
| `mlp` | 108 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| ⭐ `soft_well_table` (D8, the non-quantising twin, same 72 params) | 72 | 0.0000 ± 0.0000 | 0.0008 ± 0.0016 | 0.0008 ± 0.0016 | **−0.0008 ± 0.0016** |
| — | | | | | **`OD_min` = −0.0008 ± 0.0016** |

- **F1: TIE** (registered ambiguous band `|OD_min| ≤ 0.05`) — ⚠ and **vacuous, exactly as in C2W5**:
  every arm, every reader and the launder are all within one query (1/1280) of zero. ⛔ **This is not
  a tier-ii datum.** Per `PREREG.md` §4's honesty clause, R1 firing first means the swap number
  carries no information about physics-vs-not.
- **In-sample liveness (the L1-style anchor):** on SEEN the physics arm reads `sum_linear` 0.008,
  `soft_well_table` 0.000, **`knn` 1.000**. The memorising reader gets 100 % in-sample and 0.000 on
  `Q_unseen` — rule 4 doing its job, and the cleanest evidence that the split is sound while the
  vehicle is not.
- **F2 (the margin is the reader's): does not fire** — all five readers within ±0.0008 of 0.

## 3.1 ⛔ GUARD 1 FIRES — and it is the report's most decision-relevant number
**`read − live-recomputed launch-only launder` = `−0.0016 ± 0.0019`** (worst reader; per seed
`0.0000 / 0.0000 / −0.0039 / 0.0000 / −0.0039`). The launder is **this task's own launches with the
landscape deleted and the written payload table retained**, scored through the **same reader class at
the same `k`** — the strong form of Advisor amendment A1. ⛔ **The store does not add value over its
own launches at this vehicle.** (0.272 appears nowhere as a bar; see reconciliation 1.)

---

# 4. ⭐⭐ R1 / R2 — WHAT THE REFUTATION'S OWN STATISTICS SAY NOW

| statistic (unseen) | refuted `P = 4` (published, `orgdiv-null-arms` §3 / `orgdiv-cat-test` §1.2) | **this read (physics, 5 seeds)** | verdict |
|---|---|---|---|
| distinct wells, **launch** | 2.202 | **11.8156** | ⭐ 5.4× |
| distinct wells, **settled** | **1.70** | **11.2688 ± 0.5189** | ⭐ 6.6× |
| the settle's Δ(distinct) | **−0.50 (−23 %)** | **−0.547 (−4.6 %)** | ⭐ destruction cut 5× … |
| **R2** (fires iff settled `< ` launched `− 2 SE`) | — | 11.7877 vs 11.8156 | ⛔ **FIRES, marginally** (margin 0.028 = 0.24 % of `k`) |
| `P(≥ F distinct)`, settled | 0.050 | **1.0000 ± 0.0000** | ⭐ |
| occupancy precision, **launch** | 0.4061 | 0.2742 | (lower: `d=8` spreads the code) |
| occupancy precision, **settled, raw** | 0.2967 | 0.2222 ± 0.0102 | |
| the settle's dividend, raw | ⛔ **−0.1094** | **−0.0520** | halved |
| ⭐ occupancy precision, **settled + descent-GATED** | *(no gate existed)* | ⭐ **0.3502 ± 0.0193** | ⭐ **+0.0760 over its own launch** — the sign of cat-test §1.2's headline is **reversed** |
| **exact-set occupancy** (R1's statistic, gated) | **0.0000 / 2 560** | **0.0023 ± 0.0047** | ⛔ **R1 does NOT clear** (bar 0.02); does not literally fire either (fire ≤ 0.001) — **INDETERMINATE**, driven by one seed (0.0117; the other four 0.0000) |
| exact-set occupancy, raw (`k > F` ⇒ 0 by construction) | 0.0000 | 0.0000 | reported for continuity only |
| **`S_eff`** (band [8, 16]) | **34.1 / 51.2 / 36.6 → COLLAPSED** | ⭐ **16.0000 ± 0.0000 → IN BAND** | ⭐ the collapse is fixed; all 32 wells are visited on every seed |
| `S_eff` on the *gated* set | — | 21.06 ± 1.24 | ⛔ out of band on that (different) object; reported |

⭐ **The physics's measurable contribution, isolated:** the settled+gated occupancy precision is
**+0.0543** over the null's static read (0.3502 ± 0.0193 vs 0.2959 ± 0.0118 — 2 SEs disjoint) and
**+0.0760** over its own launch geometry (0.2742), because particles that land in a well of `A(x)`
descend **further** than those that do not (`w|A = 0.853` vs `w|¬A = 0.513`, §6). ⛔ **That is a
real, measured, physics-only signal — and it is not enough to move an exact-set metric by one query.**

## 4.1 ⛔ THE NAMED NEXT BLOCKER — cardinality, not addressability
The read visits ~11.3 distinct wells and has **no mechanism that commits to exactly `F = 4`**.
Three measurements pin it:
1. the gated set has **5.79 ± 0.90** members (physics) / **9.11 ± 0.91** (null), never 4;
2. `coverage_gated` (`A(x) ⊆` gated set) is **0.0531 ± 0.0186** — the gate throws away *correct*
   wells while keeping wrong ones;
3. ⭐ a **π-sharpening sweep** (`β ∈ {1,2,4,8,16}` with per-query mass normalised to `F`, offline on a
   settled latent, SEEN only) leaves `top-F(π) == A(x)` at **0.000 for the settled read and 0.016 for
   the launder** at every `β` — i.e. **the launch's own ranking is a better set estimator than the
   settled occupancy's.** The mechanism: `depth_ratio = 3` makes π's ranking track *which wells are
   deep*, not *which wells the query selected* (see the `depth_ratio = 1` lever, §6).
⇒ Any next iteration must add an **`F`-cardinality commitment** that is query-driven, not
depth-driven. Sharpening π cannot supply it; the information is not in the settled occupancy.

---

# 5. Registered deviations (argued, never silent)

| # | prereg says | I ran | the measurement that forced it |
|---|---|---|---|
| **D6** | `d = 4` | **`d = 8`** | the 0.072 launch ceiling vs the 0.0507 bar (§1). |
| **D7** | `σ_q` i.i.d. **per particle** | **ONE draw per query** | a `k`-particle head fed `k` i.i.d. views averages the query noise to `σ/√k` and buys its score from the compute dial. The new protocol therefore sees **strictly less** launch information than the refuted one. |
| **D8** | reader class = 4 | **5** (+ `soft_well_table`) | the shipped `well_table` hard-assigns — it *is* the quantisation §A20.3(b) forbids. The soft twin has the **identical 72 fitted parameters**; both stay in the class and `OD_min` is the min over both. |
| **D9** | `k = P = 4` | **`k = 12`** | K0. `k` is ledgered and matched (guard 4). |
| **D10** | `payload_radius = 1.0` (cat-test D3) | **0.5** | ⭐ basin reach, measured: at radius 1.0 the launch sits at `1.0/0.36 = 2.8 s` from the well in the payload block, the well's pull there is `exp(−3.9) = 2 %` of depth, and the confinement wins — settled precision collapses to **0.117 ≈ chance (0.125)** and distinct wells to 9.1. At 0.5: precision **0.253**, distinct **11.7**. ⚠ The metric is scale-invariant in this radius (`tol` scales with it), so K2/chance are unaffected — this is exactly cat-test D3's own argument applied one step further. |
| **D11** | 5 seeds, `n_unseen = 512` | 5 seeds, **`n_unseen = 256`** (`levers`: 3 seeds) | compute. 1 280 scored queries/arm; measurement grain 1/1280 = 7.8e-4. |

---

# 6. (e) THE LEVERS — and the one that is LOAD-BEARING

⛔ 3 seeds (0–2), `Q_unseen`, registered cell. `w|A` = mean descent weight of particles that land in
a well of `A(x)`; `w|¬A` the same for particles that do not.

| lever | `w\|A` | `w\|¬A` | distinct | prec (raw) | prec (gated) | **cov (gated)** | best reader |
|---|---|---|---|---|---|---|---|
| **`p₀` ON** (registered) | **0.853** | 0.513 | 11.64 | 0.239 | 0.350 | **0.112** | 0.0013 |
| **`p₀` OFF** | **0.691** | **0.240** | 10.77 | 0.169 | 0.378 | **0.009** | 0.0000 |
| `depth_ratio = 1` (no heterogeneity) | 0.887 | 0.631 | 11.77 | 0.252 | 0.326 | **0.161** | 0.0013 |
| `dedupe = "sum"` (the refuted multiset aggregation) | 0.853 | 0.513 | 11.64 | 0.239 | 0.350 | 0.112 | 0.0013 |
| `descent_gate` OFF | 1.000 | 1.000 | 11.64 | 0.239 | 0.302 | 0.152 | 0.0013 |

⭐ **(e) learned `p₀` is LOAD-BEARING, and the statistic it moves is coverage, by 12×**
(0.009 → 0.112). Without the confidence-gated ballistic kick the particles reach 19 % less far and
`A(x) ⊆ occupied` almost never happens. ⚠ At a *harder* cell (`payload_radius = 1.0`, i.e. the
launch 2.8 `s` from the well) the same lever moved `w|A` from **0.022 to 0.352 — a factor 16**: the
lever's value is a monotone function of how informationally dead the inter-well gap is, exactly as
§A20.3(e)'s "ballistic reach across dead gaps" framing predicts. ⛔ It does **not** buy gated
precision (0.350 vs 0.378) — reach and selectivity are different quantities and this lever buys only
the first.

⭐ **The descent gate is worth +0.048 of gated precision** (0.350 vs 0.302) and is what reverses
cat-test §1.2's sign — but it costs coverage (0.112 vs 0.152), which is §4.1's cardinality problem
in miniature: the gate discards correct wells along with wrong ones.

⛔ **The dedupe verb is currently INERT and this is a measured finding, not an omission.**
`dedupe="sum"` (the refuted multiset aggregation) is **identical to `noisy_or` in every digit** at
this cell, because the successive-suppression head already sends 11.64 of `k = 12` particles to
*distinct* wells — there is nothing left to deduplicate at the read. The verb only becomes live in
the regime §A20.3(c) describes as *confident ⇒ the `k` collapse onto `F` unique wells*, which this
head never enters. ⚠ Anyone quoting "dedupe/evolve-unique as a controller verb" must carry this.

⭐ **`depth_ratio = 1` is the §4.1 suspect, confirmed in the direction predicted:** removing depth
heterogeneity *raises* both reach (0.887) and coverage (0.161) and *lowers* gated precision (0.326).
The registered ≥3× heterogeneity — which exists so that F5 does not fire — is buying structure at the
read's expense. See follow-up 2.

# 7. (d) CONSOLIDATE-TO-BUDGET + THE TRASH REGION'S FIRST USE

512 probes descended for 300 steps; minima merged single-linkage at `merge_radius = 0.5 sep =
0.389`; ranked by **confinement-subtracted** depth; truncated to the designed budget (32); the rest
routed to `γ_φ(q)` (`chlu.core.friction_field`, **built in C1 and never used until here**), compact
gate ⇒ `γ_φ ≡ 0` **exactly** outside the horizons (F5 Prop-11).

| quantity (5 seeds) | measured |
|---|---|
| candidate minima found (over-dig) | **508.4** of 512 probes |
| merged + kept to budget | **32.0** |
| trashed — over budget | **476.4** |
| trashed — below the depth threshold (the controller decision) | **0.0** |
| kept depth (mean) / trashed depth (mean) | **0.0097** / **0.0000** |
| ⭐ mean distance, kept centre → nearest *designed* anchor | **1.0871** (`sep = 0.777`) |
| trash horizons instantiated | 478 (seed 0) |
| effect on raw occupancy precision (off → on) | 0.2435 → **0.2734** (+0.030) |
| effect on **gated** precision (off → on) | 0.3444 → **0.1426** (−0.202) |
| `S_eff` (off / on) | 16.00 / 16.00 |

⭐ **Two findings.** (i) **The store's actual attractors are not at its designed anchors** — the
kept centres sit **1.09** away, i.e. 1.4 `sep`, and 476 of the 508 "minima" have depth **0.0000**
(they are confinement-bowl resting points, not wells). The over-dig is real and mechanical to
detect. (ii) **The trash region works as specified and is not adopted at this cell**: it improves the
raw nearest-well assignment (+0.030) and destroys the gated one (−0.202), because damping a particle
is exactly what zeroes the descent weight the gate reads. Reported, not used in the claim cell.

---

# 8. ⭐ THE FOUR §A20.3(c) GUARDS — each shown FIRING on a designed negative

⛔ A guard that cannot fire is N74's vacuous gate. Every row below has a *demonstrated* negative,
asserted in `tests/test_multiwell_read.py`.

| guard | statistic | **measured** | designed negative | fires? |
|---|---|---|---|---|
| **G1** launch-only launder, **recomputed LIVE** (A1) | `read − launder`, worst reader | **−0.0016 ± 0.0019** | a 0-step read equals the launder **bit-for-bit** (`test_guard1_…`) | ⛔ **FIRES** |
| **G2** soft-occupancy training signal | `‖∇_head‖` soft vs hard | soft **2 612**, hard **0.0 EXACTLY**, ratio **0.0** | the `argmax` one-hot assignment, magnitude channel detached | ✅ fires on the negative |
| **G3** staged store-then-launch | `‖∇‖` blank / written | head **1.29e-3**; store **1.06e-2** (*designed* init) · head **2.64e-5**; **store 1.08e-13** (*historical* init) | two blanks: the designed-init store and the historical-scatter store | ✅ fires (11 orders) |
| **G4** `k` on the byte ledger | `k=12` vs `k=24` at matched bytes | coverage **0.2266 → 0.4648** (+0.2383); read flops **2.507e8 → 5.014e8** | `assert_k_matched` **raises** on a mismatched ledger | ✅ fires |

⭐ **G3 is a finding in its own right and it discharges a §A17.1 open item.** The *historical*
scatter init reproduces the pilot's dead store exactly (`‖∇_store‖ = 1.08e-13`); the **designed**
init (localized atoms `atom_local_radius` + the payload-shell init) revives it by **eleven orders of
magnitude** (to 4.4e-2). ⛔ **The store gradient is no longer dead at init — but the head's is still
1 300× weaker on a blank store than on a written one, so the store-then-launch ordering remains
mandatory.** The mechanism cat-test §5.1 measured is now attributed to the *initialisation*, not to
the write.

---

# 9. How I verified (commands + observed output)

```
git worktree add ../CHLU-tierii-read-fix -b agent/experiment-engineer/tierii-read-fix main
# main venv reused (protocol §4 w6 lesson); cwd = the worktree
PYTHONPATH=$PWD .venv/bin/python .claude/scratch/tierii-read-fix/k0_design{,2,3}.py   # K0, store-free
PYTHONPATH=$PWD .venv/bin/python .claude/scratch/tierii-read-fix/tune{2,3,4}.py       # ⛔ SEEN-split tuning only
PYTHONPATH=$PWD .venv/bin/python -m ruff check chlu/ tests/                            # All checks passed!
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-tierii-read --quick --out-dir …/quick     # 5 stages, ~2 min
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-tierii-read --stages k0 arms guards consolidate \
    --seeds 0 1 2 3 4 --out-dir …/main            # 1 516 s
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-tierii-read --stages levers --seeds 0 1 2 --out-dir …/levers
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_multiwell_read.py -q             # 17 passed in 64.01s
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/ -q
#   ⭐ 1306 passed, 0 failed (31 warnings, 1463.84 s) — baseline 1289 on main @ 9b2d4db + my 17
```
Artifacts under `.claude/outputs/tierii-read-fix/`: `PREREG.md` · `k0_design{,2,3}*.json` ·
`main/tierii_read_summary.json` (every seed × reader × arm) · `levers/` · `quick/` ·
`main_offband_ds3.39/` + `run_main_offband.log` (**a labelled off-band diagnostic, not a claim
cell** — see §11) · `run_main.log`, `run_levers.log`, `pytest_full.log`.

**⚠ TWO FAILURES HAPPENED AND BOTH ARE REPORTED.**
1. ⛔ **The first full run was killed by the OS with exit code 0, no traceback and no partial
   output**, at `[stage] arms`. Cause: reverse-mode AD through a 300-step `lax.scan` settle of
   `B·k = 192` particles against `1024 × 16` atoms tapes ~25 MB of per-atom residuals **per step**
   (~7 GB). Fix: `@jax.checkpoint` on the settle body (`multiwell_read.settle_particles`) — memory
   drops to the `(q, p)` carries, the organizer step falls to **1.5 s**, and the forward path is
   unchanged (nothing is recomputed when there is no backward pass). ⚠ **This is a program-wide
   hazard** (reconciliation 4): the failure mode is silent.
2. ⛔ **The first completed run sat OUT of the registered soft-certificate band** at `d/s = 3.386`,
   because `s_measured = 0.3611` had been measured at `payload_radius = 1.0` and the registered cell
   uses 0.5. Re-measured (`s = 0.2879`, a fixed point of `sep = 2.7 s`), the whole matrix was
   **re-run** at `d/s = 2.700–2.886`. The off-band run is retained, labelled, and **is not quoted as
   a claim cell**; its conclusions are identical in sign and magnitude (`OD_min 0.0000 ± 0.0000`,
   `G1_min −0.0008 ± 0.0016`, `S_eff 16.0`, exact-occ `0.0016 ± 0.0019`).

---

# 10. PREREG SCORECARD (`.claude/outputs/tierii-read-fix/PREREG.md`, filed before the module existed)

| # | registered | measured | verdict |
|---|---|---|---|
| **K0** | ≥ 0.90 store-free, else redesign | **1.0000** | ✅✅ **PASS, adjudicated first** |
| **R1** exact-set occupancy | fires ≤ 0.001, clears > 0.02 | **0.0023 ± 0.0047** | ⛔ **does not clear** — INDETERMINATE (1 of 5 seeds) |
| **R2** settle destroys addressability | fires iff settled < launched − 2 SE | 11.7877 vs **11.8156** | ⛔ **FIRES, marginally** (0.24 % of `k`; the refuted read lost 23 %) |
| **F1** organizer swap | fires `< +0.05`, TIE if \|·\| ≤ 0.05 | `OD_min` **−0.0008 ± 0.0016** | **TIE (vacuous)** |
| **F2** the margin is the reader's | — | all readers within ±0.0008 | ✅ does not fire |
| **G1** store adds nothing over its launches | fires iff `≤ 0` | **−0.0016 ± 0.0019** | ⛔ **FIRES** |
| **G2** hard assignments don't backprop | ratio `< 1e-3` | **0.0 exactly** | ✅✅ |
| **G3** staging | blank `< 1e-6` while written `> 1e-2` | store **1.08e-13** / 4.17 (historical init) | ✅ (⚠ the *designed*-init blank is 1.06e-2, not 1e-6 — the init already revived it, §8) |
| **G4** `k` is capacity | doubling `k` raises the score | coverage +0.2383 at 2× flops; ledger raises | ✅ |
| **`S_eff`** | in [8, 16] or COLLAPSED | **16.0000 ± 0.0000** | ✅ **in band** |
| **P1** physics best reader | 0.12, band [0.02, 0.35] | **0.0008** | ⛔ over-predicted **150×** |
| **P2** null best reader | 0.10, [0.02, 0.35] | **0.0008** | ⛔ |
| **P3** `OD_min` | −0.01, [−0.10, +0.05] | **−0.0008** | ✅ in band |
| **P4** live launder | 0.09, [0.02, 0.25] | **0.0008** | ⛔ |
| **P5** read − launder | +0.02, [−0.05, +0.15] | **−0.0016** | ✅ in band |
| **P6** exact-set occupancy | 0.06, [0.01, 0.25] | **0.0023** | ⛔ below band |
| **P7** settled distinct wells | 10.0, [6, 12] | **11.27** | ✅ |
| **P8** settled occupancy precision | 0.28, [0.15, 0.45] | 0.222 raw / **0.350** gated | ✅ |
| **P9** `S_eff` | 12, [8, 22] | **16.0** | ✅ |
| **P10** `‖∇_hard‖/‖∇_soft‖` | **0 exactly**, [0, 1e-8] | **0.0** | ✅✅ |
| **P11** learned-`p₀` Δ score | +0.01, [−0.03, +0.06] | +0.0013 (score) · **+0.162 `w\|A`** | ✅ in band (score); the reach effect is the real one |

**Score: 11 ✅ · 5 ⛔ · 2 marginal/indeterminate.** ⭐ **The pattern in the five ⛔ is one fact: I
predicted absolute scores 100× too high on EVERY arm including the launder and the null.** The
prediction that survived is the *relative* one (P3/P5) and the mechanism ones (P7–P10). ⛔ **The
absolute level of this family under any reader in the registered class is `≈ 0`, and that is now
measured three times (C2W5 cat-test, C2W5 null-arms, here) across three different read protocols.**

# 11. ⛔ DECLARED NOT-RUNs (never to be reported as nulls)
1. **`k = 16` / `k = 24` as scored arms** (`k=24` appears only as guard 4's capacity probe).
2. **A ψ A/B.** Shipped ψ everywhere; `psi-payload-residual` had not landed (A2 honoured).
3. **N2 / N4 / N5** as swap arms (the swap runs against `N1′` — the strongest organizer in
   `orgdiv-null-arms`, which fits 100 % of its own training items — and `N3′` is **also not run**,
   dropped for compute; declared).
4. **The γ axis** — one claim cell at `γ_address = 0.05`.
5. **`d ∈ {16, 24}`** as scored cells (K0 geometry only; the query-only leak grows to `R² = 0.33/0.65`).
6. **A learned (trained) launch head as a scored arm.** `train_launch_head` is built, tested and
   used only by guard 3's probe; the scored cells use the **designed** head, which is the tighter
   swap (bit-identical launches on both arms) and the cell K0 was adjudicated at.
7. **The off-band `d/s = 3.386` matrix** — retained as a labelled diagnostic (§9.2).

# 12. Git footprint
- **Branch:** `agent/experiment-engineer/tierii-read-fix` (off local `main @ 9b2d4db`), worktree
  `../CHLU-tierii-read-fix`. ⛔ Not pushed, no PR, no merge. `origin` untouched, `clu-dev` untouched.
- **Commits** (verified from the MAIN repo, protocol §3.2 — the wave-4 lesson):
  - `201d250` `[experiment-engineer] the multi-well read protocol (charter §A20.3(b)-(e))`
  - `ea46ce5` `[experiment-engineer] exp_tierii_read + the chlu exp-tierii-read CLI hook`
  - `bcefed7` `[experiment-engineer] tests: K0, the four guards, and the non-quantising latent`
  `git diff --stat main..HEAD` = **4 files changed, 1 906 insertions(+), 0 deletions**. The s-band
  correction and the K0 `d=4` control (§9.2) were folded into `ea46ce5` before it was committed —
  **every number in this report comes from the committed tree.**
- **Files touched (the declared ownership list, exactly):** `chlu/core/multiwell_read.py` (**new**) ·
  `chlu/experiments/exp_tierii_read.py` (**new**) · `tests/test_multiwell_read.py` (**new**) ·
  `chlu/cli/experiment_cmd.py` (**+1 subcommand block, +1 handler, no existing line altered**).
  ⛔ **Not touched:** `chlu/core/factored_store.py` (no additive hunk turned out to be needed — the
  new module composes with its public API), `chlu/core/null_arms.py` (used through its public API:
  `n1_gradient_placed` / `n1_apply`), `chlu/core/psi_readout.py`, `chlu/config.py`,
  `chlu/experiments/exp_cat_test.py`, `chlu/experiments/exp_null_arms.py`.
- **Rebase:** onto local `main` (⚠ **not** `origin/main`) — no-op, base unmoved.
- **Concurrent work:** no worktrees were registered at spawn; the shared checkout was clean and was
  never edited by me. ⚠ **`agent/experiment-engineer/psi-payload-residual` appeared mid-task**
  (worktree `../CHLU-psires`, report filed 22:35). **Zero file overlap** — their branch touches
  `chlu/core/blocks.py`, `chlu/experiments/exp_psi_residual.py`, `tests/test_psi_residual.py`; mine
  touches none of them (verified by diffing both branches against `main`). ⛔ **Advisor amendment A2
  honoured to the letter: their report landed AFTER my PREREG was filed and DURING my science runs,
  so per the no-mid-task-hot-swap clause every cell here uses the SHIPPED ψ, uniformly on all arms.**
  ψ's marginal value is not measured here and belongs to its own pre-registered ablation.

# 13. Open questions / follow-ups / risks
1. ⭐ **Cardinality is the next named blocker and it is not a physics question** (§4.1). The read has
   no `F`-commitment; sharpening `π` provably cannot supply one (measured 0.000 at every `β`), and
   the *launch's* ranking already beats the *settled* occupancy's. A next iteration needs a
   query-driven cardinality mechanism (a learned budget/stopping verb), or the family needs a metric
   that is not all-or-nothing at `m = 8`.
2. ⭐ **`depth_ratio` is in direct conflict with the read.** The registered depth heterogeneity
   (≥ 3×, there so F5 does not fire) is exactly what makes the occupancy ranking depth-driven rather
   than query-driven. **The prereg's structural falsifier and its read protocol are pulling in
   opposite directions**, and someone has to adjudicate which one moves.
3. **The family's absolute level is `≈ 0` under three read protocols now.** Before a fourth read is
   built, the cheapest decisive instrument is the one this task already has: the **out-of-class
   ceiling per `(d, k, σ_q)`** (60 s, no store). At `d = 8` it is 0.695 — so the information *is*
   there and the gap is entirely decoder-side, inside a reader class capped at `< N_a·m` parameters.
   ⚠ It is worth asking whether that cap, not the store, is what makes every arm read 0.
4. **G1 firing is the strongest single argument against the current vehicle** and it is
   arm-independent: it says the *launch head* carries whatever signal exists. Any tier-ii claim built
   on this vehicle must clear G1 first, and G1 is cheap (no settle).
5. **Risk to how this gets quoted.** "K0 repaired, `S_eff` in band, the settle's destruction reversed"
   are all true and all *addressing* statements. ⛔ They must never be quoted without R1 and G1 in the
   same paragraph.

---

## Proposed handover updates (for the Hub)

- **§3 CLI/config — NEW:** `chlu exp-tierii-read [--stages k0 arms guards consolidate levers]
  [--seeds …] [--organize-steps N] [--k-particles K] [--quick] [--out-dir D]`; new module
  `chlu/core/multiwell_read.py` with `MultiWellReadConfig` (lives next to its code, the
  `CatTestConfig` / `NullArmGrid` precedent — **not** in `chlu/config.py`).
- **§7 Known Issues — NEW (open, program-wide, HIGH):** *reverse-mode AD through an
  un-`jax.checkpoint`-ed `lax.scan` settle OOM-kills the process silently* — **exit code 0, no
  traceback, no partial output**. Measured: a 300-step settle of 192 particles against 1024×16 atoms
  tapes ~7 GB. Remedy: `@jax.checkpoint` on the scan body (`multiwell_read.settle_particles` is the
  exemplar); the organizer step drops to 1.5 s and the forward path is unchanged.
- **§7 Known Issues — NEW (open):** *`payload_radius` is a basin-reach constraint, not a free
  scale.* At `payload_radius / s = 2.8` the launch cannot reach the well (well pull = 2 % of depth,
  confinement wins) and settled occupancy precision collapses to chance (0.117 vs 0.125). Measured
  fix: `payload_radius = 0.5` at `s = 0.288`. The metric is scale-invariant in this radius, so it is
  a free design knob — but only if someone checks the ratio.
- **§7 Known Issues — RESOLVE/RE-SCOPE (three C2W5 entries are read-protocol-specific, not
  properties of the physics):** the `P`-particle occupancy K0 entry (0.050 → **1.000** at `k=12`,
  `d=8`); cat-test §1.2's settle-destroys-27 % (→ **4.6 %** raw, **+0.076** gated); cat-test §5.2's
  `S_eff` COLLAPSED (→ **16.00**, in band). Reconciliations 2–3.
- **§7.28 ruler — NEW DATUM:** `s = 0.2879` (`d=8, m=8, a=32`, payload radius 0.5,
  confinement-subtracted, `R² = 0.9986`), and it is a **fixed point** of `sep = target_ds · s`.
- **`FROZEN-interfaces.md` erratum (rides with `orgdiv-null-arms` recon 2):** the reader-param row
  needs a `(d, m)` column — at `d=8, m=8` the counts are 136 / 72 / 108 / 0 / 72, all `< N_a·m = 256`.
- **Registry/doctrine candidates:** (i) ⭐ *`K0` should be standing*: `P(≥F distinct wells reachable)`
  is computable store-free in seconds and would have killed the C2W5 cell before a single store was
  written. (ii) ⭐ *a launch-information ceiling must be quoted with its noise model* — the same
  family reads 0.072 / 0.272 / 0.695 depending on `(d, draws)`. (iii) *a guard's designed negative
  belongs in the test suite, not the report* — all four here are `pytest`-asserted, which is what
  makes "the guard fires" checkable by the next agent.

---

## ⛔⛔ DATED CURATOR ERRATUM BANNER (2026-08-06, `doc-curator-c2w7-fold`, [C2W7] — body above UNTOUCHED, C-3 precedent)

**Authority:** charter **ADDENDUM 8 §A24/§A26.4/§A26.5** + **ADDENDUM 9 §A27.2**, Head-ratified 2026-08-06.
**Sources:** `.claude/outputs/c2w7-read-cardinality.md` §3/§4/§5 · `.claude/outputs/reader-fitting-audit.md`
§4/§6/§9 · the 2026-08-05 and 2026-08-06 `[C2W7]` §10 entries. ⛔ **No number in this report is
retracted; four quotation forms move, and one of this report's own hopes is measured and refuted.**

**1. ⛔ THE `S_eff` RE-LABEL IS MANDATORY (§A26.4).** The `[8, 16]` band is **RETIRED**: `S_eff = K·F/W
= 512/W` with `W ≤ N_a = 32` ⇒ **`S_eff ≥ 16` always**, so the band's lower half is **unreachable by
construction** and **16 was the ONLY attainable in-band value**. ⭐ **This report's `16.0000 ± 0.0000
→ IN BAND` is re-labelled *"all 32 wells visited"*** — still a real 100 %-utilisation result, ⛔ **but
it must never read as *"comfortably inside a band"***, and §1's *"`S_eff` goes 34–51 (COLLAPSED) →
16.00 (exactly `S`, in band)"* must carry that reading. ⚠ **§4's own note (*"all 32 wells are visited
on every seed"*) is the correct form and is unaffected.** ⚠ **C2W5's 34–51 stays COLLAPSED — it is
CONCENTRATION, which is what the word is now reserved for.** (**N239**.)

**2. ⛔ THE `k`-CAPACITY CLAIM IS REFUTED AT A MULTIPLICITY HEAD.** This report's registered *"doubling
`k` raises the score"* (**+0.2383** coverage) does **not** hold once the head commits to a cardinality:
at C2W7's guard 4, `k = 12 → 24` at **exactly 2× ledgered read flops** moves best-reader
**0.04296875 → 0.04296875** and coverage **0.234375 → 0.234375** (identical to 4 dp). ⭐ **`k` is a
RESOLUTION dial once the head commits.** (**N234**.)

**3. ⛔ `K0 = 1.0000` IS A REACHABILITY STATEMENT ONLY.** Measured at C2W7: a **collapsed** head
(`F_hat` saturated at `f_max`) scores **`K0 = 1.000`** while **every reader reads 0.0000** and
occupancy precision falls to chance (0.124 = 4/32). ⛔ **Never quote `K0` without `R1` and `M15` in the
same paragraph.** ⚠ **And §5's risk item is now the standing never-quote it predicted it would be:
"`K0` repaired, `S_eff` in band, the settle's destruction reversed" is not quotable without `G1`.**
(**N235**.)

**4. ⛔⛔ THIS REPORT'S OWN `G1` GETS WORSE UNDER A ZERO-PARAMETER READER — the hope is REFUTED.** The
C2W7 reader-fitting pathology raised the possibility that this report's arms were scored through a
reader that erased their signal. The audit re-scored **this cell** (physics arm + live launder,
9 readers × 3 arms × 5 seeds) with zero-parameter identity readers **added**: ⭐ **`gated_well_identity`
scores 0.00234375, per seed `[0, 0, 0, 0.01171875, 0]` — BIT-IDENTICAL to this report's published
`exact_set_occupancy_gated`** (the largest identity-minus-fitted gain anywhere, **+0.00156**), ✅
**`OD_min` is UNCHANGED at −0.00078125 (identical per-seed)**, `S_eff = 16.00` and
`exact_set_occupancy_raw = 0.0000` reproduce — ⛔⛔ **and `G1_min` moves from −0.0015625 to
−0.00234375, i.e. WORSE**, because on seed 0 the *launder's* `sum_identity` scores where the physics
arm's does not. ⇒ ***the fitting artifact was not hiding a physics dividend here.*** (**N237**;
verdict **SURVIVES**, §A27.2.)

**5. ⚠ AND THE C2W7 RESULT THIS REPORT HANDED FORWARD, for the record:** its named blocker
(**CARDINALITY**) was **solved** — `F_hat = 4.542 ± 0.122`, exactly `F` on 39.4 %, and **`R1` goes
0.0023 → 0.065625 ± 0.023876, clearing its bar for the first time in the programme (28×)** — ⛔ **and
`G1` fired harder: −0.007031 ± 0.001562, 5/5 seeds negative, with the designed/untrained head beating
the trained one (0.0695 vs 0.0539).** ⭐ **The mechanism is this report's own §7 finding, promoted:
the store's attractors sit ~1.4 `sep` from its designed anchors, so the settle SCATTERS a committed
allocation (4.834 → 5.668 distinct wells)** — ⛔ **and `R2`, as registered here, cannot see that
failure (it fires only if settled < launched; this failure has the opposite sign).** ⛔ **The read
track is now PAUSED (§A26.6): there is no iteration 3, and the next tier-ii iteration is WRITE-SIDE.**
(**N232 / N233 / N236**.)

---

## ✅ DATED REGISTRATION NOTICE (2026-08-06, `doc-curator-tierii-read-fix-catchup` — body and banner above UNTOUCHED)

⛔⛔ **Until this date this report had NO registry entry**, while the corrections in the banner above
were already filed against it. **That is now closed.** This wave is registered at
`negative_results.md` **N241–N249**, each entry written **in its already-corrected form**
(N241 addressing · N242 expressivity + the audit's `G1`-gets-worse finding · N243 the `k`-capacity
claim with its refutation · **N244 the 1.4-`sep` attractor offset, cross-linked to N233 both ways** ·
N245 learned `p₀` · N246 consolidate/trash as prior art for C2W8 · N247 the dedupe verb ·
N248 this report's reconciliations 1 and 5 · N249 the silent-OOM hazard), plus a
`DECLARED NOT-RUNs — READ ITERATION 1` block transcribed from §11.
**Also filed:** `claims_matrix.md` **v2.14 (PROPOSED)** — a dated catch-up block in §0.11 + **CM-36/CM-37** ·
`philosophy-synthesis.md` **⟲ READ ITERATION 1 addendum (catch-up)** · `HEP_primers.md` **§11.24, Record 27** ·
`future_work.md` **read-iteration-1 catch-up block** · a `(d, m)` addendum on
`orgdiv-cat-test/FROZEN-interfaces.md` (this report's **reconciliation 5**, discharged).
⚠ **Reconciliations 2/3 were discharged by the C2W7 fold; reconciliation 1 (the `0.272` scoping) and
the §7.28 ruler datum are registered WITH OWNERS NAMED and are the Hub's, not the curator's.**
⛔ **No number in this report is retracted by this pass, and nothing above is re-scored.**
