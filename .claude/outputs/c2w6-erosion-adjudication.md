# c2w6-erosion-adjudication — results-analyst report (RE-SPAWN, gate satisfied)

Task + acceptance criterion: adjudicate I2 (ρ(usefulness, erosion rate)) and re-derive the C2W6
run-3 gate from the raw artifacts, independently of the engineer's tables.
**Status: done.** Supersedes the 11:39 `BLOCKED` version of this file (its §2–§5 pre-cell audit is
retained below as §0.1 and is unchanged).

> ## ⚠ RECONCILIATION LIST — needs a Hub-assigned owner THIS review (protocol §5, first-10-lines)
> **A1 — the erosion curve is NOT netted of the designed decay, and the decay's exponent DRIFTS.**
> `last_write_chunk` moves 0→12 within one slot across a run (6–10 distinct values/slot), so each
> reading carries a different designed-decay factor `exp(−0.04·(15−c))`. Netting it changes E1 at
> p1_off seed 0 from **9.78× → 6.47×** (−34 %). ⛔ **No C2W6 verdict flips** (E1 0/3 raw *and*
> netted). *(Owner: engineer — the netting belongs in the C2W8/C2W10 harness before any
> flattening/erosion-rate claim; C2W6's record needs only the caveat.)*
> **A2 — "0.708 ± 0.57× — it does not decay, it recovers" is not a defensible claim form** (Hub Q1,
> adjudicated §3): the direction is **not significant under either estimator** (arith 0.708 ± 0.570
> = 0.51 SE from 1; geo **0.327**, log-mean/SE **−1.29**). ⭐ **And §3.1 supplies the replacement:
> the curve has a large TRANSIENT trough (to 0.9 % / 5.3 % of untrained at steps 150 / 275 on 2/3
> seeds) and the banked 200-step anchor sits INSIDE it — N223's mechanism fires, it is just not
> terminal at this config.** *(Owner: Hub/curator — the engineer's R1 + §3, the §7.27 re-scope
> wording, and the in-flight monitor's trigger shape.)*
> **A3 — the engineer's §6 "pooled (n wells)" ρ column is a NON-registered estimator and it flips
> sign** on the mechanism row: ρ(‖∂L/∂atoms‖, erosion) pooled **+0.2198** vs the registered
> per-seed-then-mean **−0.1619 ± 0.2806**. The sentence "the gradient magnitude … is weakly
> predictive … the mechanism the hypothesis assumed" does not survive the registered estimator.
> *(Owner: Hub — strike or relabel; it is not in the N-registry candidate list, so containment is
> cheap.)*
> **A4 — the LOO usefulness proxy has ≤ 0 reliability at the primary arm** (ICC(1,1) =
> −0.205/−0.218/−0.248, 3/3 seeds) ⇒ **ρ(LOO) = +0.067 is not a null, it is undefined.** It must
> never be quoted as a measured correlation. *(Owner: Hub + engineer's I2 table.)*

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)

- **Dial:** lifetimes/isolation (training-time protection of written content) — **adjudication /
  instrument form.** ⛔ No paper number is produced here.
- **Laundering control:** live vs blank vs memory-deleted, paired seeds, both arms; **K4**
  re-derived per cell-seed below (§4) and it does not fire on any of the 21 cell-seeds.
- **Falsifies / does not falsify:** per `PREREG-AntiErosion.md` §3 + my `PREREG-Adjudication.md`.
  Losing to a classical/oracle method on a metric-native protocol is not in scope here.
- ⚠ Monitor #13 / N94 rides every w4 number. ⛔ TOY (0.16 M). ⛔ **"Depth is NOT quotable as
  feature importance" — my §5 verdict KEEPS the caveat active** (branch = `NO_USAGE_STRUCTURE`;
  only the refutation branch would lift it, and it was not reached). **The Hub lifts, not me.**

## 0. FLAG PROVENANCE (every number in this report)

| item | value |
|---|---|
| repo state | **`main @ d1149a4`** (the Hub's C2W6 merge). ⛔ **No code touched, no branch, no commit, nothing pushed.** |
| artifacts read | `.claude/outputs/c2w6-anti-erosion/erosion_{p1_off,p1_on,p1_on_i1_on,w40_p1_off,w40_p1_on,resoff_p1_off,resoff_p1_on}_records.json` + `erosion_aggregate.json`; banked `psi-payload-residual/psires_trained_records.json`, `pilot-placement-probe/probe_trained_records.json` |
| smoke exclusion | the three throwaway smoke runs are under `smoke/` and are excluded by my loader; **n_records = 21** (7 cells × 3 seeds) — independently recounted |
| JAX | **0.9.0** (recorded in every record file); analysis ran on the main venv, numpy only — **no model was run, nothing was trained** |
| seeds | **0, 1, 2** on every cell, paired; SE = sample sd (ddof=1)/√3, n stated on every row |
| scale / rig | TOY 0.16 M (`d_model 64`, `n_layers 2`, `seq_len 512`, `batch 4`, `chunk 32`), enwik8 bytes; **CSF3 run-2 config**: `atom_place_radius=0.3`, `write_margin=0.6`, `psi_payload_residual=True`, `psi_residual_source=q_star`, `psi_residual_gain=1.0`, all stage flags TRUE, `leak 0.02`, `capacity 8`, `budget 6`, `n_atoms 1024`, `dim 3` |
| levers | `erosion_partition` ∈ {False, True}; `refresh_monotonic` False except `p1_on_i1_on`; `w40_*` = `write_inner_steps=40` (else 4) |
| horizon | 1000 outer steps (w4) / 400 (w40), `monitor_every=25` ⇒ 41 / 17 readings |
| my scripts (reproducible, read-only) | `.claude/outputs/c2w6-erosion-adjudication/{rederive_gate.py, rederive_i2.py, rederive_proxies.py, power_pricing.py}`; figure `adjudication_figs.png` |
| pre-registration | `PREREG-Adjudication.md` mtime **11:37:17**, i.e. **before the first smoke run (11:58:32) and 100 min before the first science cell (13:17:55)** — my *estimators* (§1 netted rate, Fisher-z pooling, floor censoring) are pre-registered w.r.t. every cell. ⛔ Everything labelled **POST-HOC** below (F1/F2/F3 band repairs per the Hub's ruling; the §6 power/reliability analysis) is applied only as a reporting-side estimator, never as a re-scoring of a registered verdict. |

### 0.1 Carried forward from the pre-cell audit (unchanged, still standing)
All nine banked anchors of `PREREG-AntiErosion.md` §3 were independently re-derived and verified
(untrained 0.07265/0.14895/0.22502 · residual-off 0.001112/0.058154/0.037318 · residual-on
0.002132/0.273180/0.120959 · paired Δdepth +0.0999 ± 0.0623 · Δbpc +0.0053 ± 0.0030 · probe
0.02882 → 1.4e-87/1.5e-62/2.5e-177, arith 4.954e-63, **geo 3.76e-109**, 114.8 decades).
F1 (the 4.95e-63 censoring form), F2 (P-residual row unscoreable as written), F3 (depth on the log
scale), F4 (w4 ≠ "run-2 config" lineage), F5 (E2 is near-tautological given a green K1) stand as
written. **F5 is now confirmed empirically:** ρ(grad, ·) is `nan` on every P1-ON arm because
‖∂L/∂atoms‖ ≡ 0 at all 41 readings — E2 tested a curve whose gradient driver was severed.

---

## 1. VERIFICATION: the engineer's numbers reproduce, digit-for-digit, from raw

Every scalar recomputed from `records[*].curve` (the per-reading series) and compared to the
record's own stored fields and to `aggregate()`:

| check | result |
|---|---|
| 21 records × 5 curve scalars (`depth_untrained/at_200/final`, both ratios) recomputed from `curve` | **worst relative deviation 0.00e+00** |
| median over live wells of `telemetry[*].wells[*].depth` vs `curve[*].depth_median`, p1_off s0, 41 readings | **max abs diff 0.000e+00** |
| I2 ρ (read-sel, LOO, grad) recomputed from `telemetry[*].wells` for 12 cell-seeds | **MATCH** (< 1e-12) on all |
| gate legs E1/E2/K3/K4 (both budgets), prereg scorecard | reproduce exactly; `n_records: 21`, no smoke contamination |
| Hub's own re-derivation (E1 9.782/0.9035/0.5305 · K3 −0.004853 ± 0.00078, 6.23 SE · w40 ON/OFF 21.53/1.516/1.464 geo 3.628 · ρ_sel −0.2571 ± 0.1512 · ρ_LOO +0.0667 ± 0.1627 · `FAILS_FLATTEN`) | **confirmed, all of it** |

⇒ **No mismatch anywhere between raw artifacts, the engineer's tables, and the Hub's pass.** The
findings below are about *estimators and instrument validity*, never about transcription.

## 2. THE GATE, RE-DERIVED (prereg §4 vocabulary, from the raw curves)

| leg | rule (registered) | w4 (`p1_on` vs `p1_off`) | w40 |
|---|---|---|---|
| **E2** ON flattens | final/200 ≥ 0.5, **3/3** | 7.0680 / 0.5392 / 2.9360 → **3/3 ✅** (inside the registered [0.5,1.05] band on **1/3**) | 0.1541 / 1.4013 / 1.1833 → **2/3 ❌** |
| **E1** OFF decays | final/200 ≤ 0.5 on ≥2/3 (or collapsed by 200) | 9.7815 / 0.9035 / 0.5305 → **0/3 ❌**; `collapsed_by_200` false ×3 | 0.1599 / 1.8772 / 0.9691 → **1/3 ❌** |
| **K3** bpc not worse | paired Δ ≤ 2 SE and ≤ 0.01 | **−0.004853 ± 0.000780 (6.23 SE, 3/3 better) ✅** | +0.005676 ± 0.005613 (1.01 SE) ✅ |
| **K4** not relocated | fires iff \|live−blank\|≈0 **and** md-margin ≤ 0 | \|live−blank\| 3.14e-04…1.95e-03; md margin +0.1048/+0.1243/+0.2259 → **did not fire ✅** | \|live−blank\| 5.3e-05…1.70e-03; md margin +0.0035/+0.0125/+0.0233 → **did not fire ✅** |
| **VERDICT** | | **`FAILS_FLATTEN`** | **`FAILS_FLATTEN`** |

**Run-3 verdict in the prereg's own vocabulary: `FAILS_FLATTEN` at both budgets, and it matches
`aggregate()` digit-for-digit.** The failing leg is **E1, the baseline** — this re-derivation
confirms the engineer's R2 and the Head's §A23.2 re-scope premise, independently.

Two things the gate table should not be allowed to imply:
- **The depth-protection leg is a SIGN statement, not an effect size.** w40 paired ON/OFF final
  depth 21.527 / 1.516 / 1.464 → paired log-ratio **geo 3.628× [1.49, 8.84] = 1.45 SE**; sign test
  3/3 one-sided p = 0.125. The Advisor's §A22 "w40-scoped, 3/3" wording is exactly right; it must
  not be upgraded to a resolved magnitude. (w4: geo 1.421× [0.70, 2.89], 2/3.)
- **K4 "did not fire" is not "the memory is useful."** The memory-deleted margin is **positive on
  6/6 gate cell-seeds** — the block is *better* with the memory deleted. Re-derived retrained-`none`
  agreement: +0.0719/+0.0911/+0.1551 vs eval-swap +0.0778/+0.1248/+0.1966 (p1_off only —
  `bpc_none_retrained` is NaN on 18/21 records, the engineer's declared cut; flagged for anyone
  recomputing K4 from raw).

## 3. ⭐⭐ Q1 — THE HEADLINE ESTIMATOR, ADJUDICATED (the Hub's provisional finding, ruled on)

**The Hub's arithmetic is correct and I confirm it**: `final/untrained` = 0.1351 / 1.8471 / 0.1404,
arith **0.708 ± 0.570**, **geo 0.327**. Two seeds lost ~7.4× and 7.1× depth; one gained 1.85×.

Adding the statistic the Hub did not compute — the significance under each estimator:

| arm | per-seed final/untrained | arith ± SE | SE from 1 | **geo [±1 SE]** | **log-mean / SE** |
|---|---|---|---|---|---|
| `p1_off` (w4, primary) | 0.1351 / 1.8471 / 0.1404 | 0.708 ± 0.570 | **0.51 (n.s.)** | **0.327** [0.138, 0.778] | **−1.29 (n.s.)** |
| `p1_on` (w4) | 0.1463 / 0.8989 / 0.7646 | 0.603 ± 0.232 | 1.71 (n.s.) | 0.465 [0.260, 0.831] | −1.32 (n.s.) |
| `w40_p1_off` | 0.0029 / 2.7253 / 0.8896 | 1.206 ± 0.802 | 0.26 (n.s.) | 0.192 [0.023, 1.596] | −0.78 (n.s.) |
| `resoff_p1_off` ⛔DIAG | 0.0188 / 0.0751 / 0.3291 | 0.141 ± 0.095 | 9.0 | **0.0775** [0.034, 0.177] | **−3.10 (significant)** |

**Ruling (the defensible claim form).** The E1 refutation is robust — as the Hub says, and it also
survives my designed-decay audit (§4). But **"it does not decay, it recovers" is not supportable**:

- Under the **arithmetic** estimator the mean is 0.708 ± 0.570 — **0.51 SE below 1**, i.e. no
  evidence of *either* decay or recovery, and the mean is carried by the single seed that gained.
- Under the **geometric** estimator (the right one for a positive quantity spanning decades —
  within-arm seed spread here is **13.7×**) the point estimate is **0.327×, a 3.1× LOSS**, and it is
  **also not significant** (log-mean/SE −1.29).
- The two estimators **disagree in direction** on the same three seeds. Any sentence that asserts a
  direction is estimator-dependent and therefore not a finding.

⭐ **Defensible claim (recommended wording):** *"Over 1000 outer steps at the run-2 config the
median fitted well depth ends statistically indistinguishable from its untrained value on 3 paired
seeds (geometric mean 0.33×, log-mean −1.29 SE; arithmetic 0.708 ± 0.570). The monotone collapse
banked at the shipped config (→ 1.4e-87 / 1.5e-62 / 2.5e-177 after 200 steps) does not occur on any
seed: the smallest final/untrained ratio observed is 0.135."*

⛔ **NEVER-QUOTE wordings:** "0.708 ± 0.57× — the store does not decay, it recovers"; "there is no
erosion at the run-2 config"; any bare "0.708×" without its SE and its geometric counterpart.
✅ **What DOES survive, and is the wave's real R1:** the *comparative* statement. The residual-off
corner erodes **significantly** (geo 0.0775×, log-mean/SE −3.10, 3/3 seeds) while the run-2 arm does
not — **erosion tracks uselessness**, and that comparison is estimator-robust (arith 0.141 ± 0.095
vs 0.708 ± 0.570; geo 0.0775 vs 0.327, a 4.2× separation).

### 3.1 ⭐⭐ The shape the ratio statistics hide — N223's erosion DOES happen, transiently
*(POST-HOC — not in any prereg; a description of the registered curve, not a re-scoring.)*

Reading the raw curves rather than their endpoints (figure panel 1): on `p1_off` the depth **falls
hard and then partly recovers**. Per seed, minimum over the 41 readings and where it occurs:

| `p1_off` seed | untrained | **trough (step)** | **min / untrained** | at step 200 | final | final / min |
|---|---|---|---|---|---|---|
| 0 | 4.63e-02 | 4.12e-04 (**step 150**) | **0.0089 — a 112× loss** | 6.39e-04 | 6.25e-03 | 15.2× |
| 1 | 1.62e-01 | 1.62e-01 (step 0 — **no dip**) | 1.000 | 3.31e-01 | 2.99e-01 | 1.85× |
| 2 | 2.49e-01 | 1.32e-02 (**step 275**) | **0.0532 — a 19× loss** | 6.59e-02 | 3.50e-02 | 2.64× |
| | | | **geo 0.0779, log-mean/SE −1.85** | geo 0.196 (−1.13 SE) | geo 0.327 (−1.29 SE) | geo 4.20× |

This is **not** a minimum-of-noise artifact: the reading-to-reading noise over the last 400 steps is
**1.33× / 1.10× / 1.05×** (median \|Δln depth\|), two orders of magnitude smaller than the 112× and
19× excursions, and seed 1's argmin is at step 0 (it simply never dips).

⭐⭐ **The reconciliation with N223 that nobody has stated:** the banked collapse was measured **at
200 outer steps** — which at the run-2 config is **inside the transient trough** (troughs at steps
150 and 275 on the two seeds that dip). So the E1 statistic as registered (`final / step-200`) has
its **denominator sitting in the trough on 2/3 seeds**, which is precisely why it reads 9.78× and
0.53×: **E1 was structurally measuring recovery-from-the-trough, not decay.** The honest description
of the baseline is therefore neither "the store collapses" nor "there is no erosion":

> ⭐ **Recommended claim form (replaces "does not decay, it recovers"):** *"At the run-2 config the
> in-block store undergoes a large transient depth loss over the first 150–275 outer steps (to
> 0.9 % and 5.3 % of untrained depth on 2/3 seeds; the third seed does not dip), followed by partial
> recovery to a final level statistically indistinguishable from untrained (geo 0.33×, −1.29 SE).
> The banked shipped-config number was taken at 200 steps, i.e. at the bottom of this transient. The
> monotone, terminal collapse to ~1e-63 does not reproduce at the run-2 config on any seed."*

Consequences: (i) **N223's watch item is still live** — the mechanism fires, it is just no longer
terminal at this config, so the in-flight monitor's trigger should key on the *transient*, not on a
monotone trend; (ii) any future erosion gate must fix its reference reading **outside** the
transient (e.g. `final / max over the run`, or a registered late-window baseline), because a
step-200 denominator is config-coupled to the trough; (iii) `p1_on` shows the same shape
(min/untrained geo 0.105, troughs at 150/725/375) — **P1 does not prevent the transient**, which is
consistent with §5's finding that the surviving depth loss is not carried by the partitioned
channel alone.

## 4. ⭐ Q3 — THE DESIGNED-DECAY SEPARATION AUDIT (my §3, on the real curves)

**The decay law, read out of the code and confirmed against the data** (not assumed):
`train_cluformer.build_lane_plan` l.804–809 sets `group_scale[c,s] = exp(−leak)` for every live slot
at every chunk tick (`leak = 0.02`); `blocks.py` l.1402 applies it before chunk *c*'s write. Depth
∝ amp² ⇒ per-tick depth factor `exp(−0.04)`, predicted relative drop **0.039211**.

| audit check | result |
|---|---|
| harness-measured `median_rel_drop_own` over **717 readings** (all cells/seeds) | median **0.039211**, min 0.039210, max 0.039211 — the law is exact |
| own-leg residual vs `D·group_scale²` over all readings | max **3.324e-07** (float32 ULP) — the masked write is C3-local, confirming the engineer's A1-4 |
| curve recomputed end-to-end from the per-well series (p1_off s0) | **max diff 0.000e+00** vs `curve.depth_median` — the plotted curve *is* the median over live wells |

⛔ **The finding: the curve is raw, and the decay exponent is not constant.** Depth is read at the
end of the pass, so a slot last written at chunk *c* carries `exp(−0.04·(15−c))`. `last_write_chunk`
**drifts across readings** as φ moves the allocator: p1_off s0 slot 0 goes chunk **0 → 12**; every
slot takes 6–10 distinct values over a run. Netting it out (`depth / exp(−0.04·(15−c))`, per well,
then median):

| cell | E1/E2 (final/200), RAW → NETTED, per seed | geo raw → net | verdict raw → net |
|---|---|---|---|
| `p1_off` | 9.78→**6.47** · 0.904→0.887 · 0.530→0.634 | 1.674 → 1.538 | E1 **0/3 → 0/3** (unchanged) |
| `p1_on` | 7.07→7.07 · 0.539→0.631 · 2.936→2.711 | 2.237 → 2.295 | E2 **3/3 → 3/3** (unchanged) |
| `w40_p1_off` | 0.160→0.159 · 1.877→2.110 · 0.969→0.924 | 0.663 → 0.677 | 1/3 → 1/3 |
| `w40_p1_on` | 0.154→0.154 · 1.401→1.297 · 1.183→1.030 | 0.635 → 0.590 | 1/3 → 1/3 |
| `resoff_p1_off` | 1.348→1.482 · 0.228→0.190 · 3.078→3.063 | 0.982 → 0.952 | 1/3 → 1/3 |

⭐ **Audit verdict: PASS with a caveat.** No mis-netting exists (the law is exact to float32), the
curve is exactly what it claims to be, and **no C2W6 verdict changes under netting** — E1's
refutation and E2's pass are both robust, so §2's gate and §3's ruling stand. **But** up to **34 %
of seed 0's apparent "recovery" (9.78 → 6.47) is allocator drift toward later write chunks, not
depth restoration.** Since `final/untrained` moves 0.1351 → 0.0858 at that seed, netting makes the
run-2 arm look *more* eroded, never less — reinforcing §3's ruling against "it recovers".
Figure: `.claude/outputs/c2w6-erosion-adjudication/adjudication_figs.png` (panels 1–3, raw solid vs
netted dashed).

## 5. ⭐⭐ I2 — THE VERDICT, AND THE QUOTATION CONSEQUENCE

Registered estimator (`PREREG-AntiErosion` A1-5 / my `PREREG-Adjudication` §1): Spearman per seed
over live wells, then mean ± SE across seeds. Primary usefulness measure = **read-selection**
(first-named in the registered text and in my prereg); LOO is confirmatory. Partition-OFF arm.

| arm | ρ(read-selection) per seed | mean ± SE | ρ(LOO) mean ± SE | ρ(grad) mean ± SE |
|---|---|---|---|---|
| **`p1_off` (w4, PRIMARY)** | +0.0286 / −0.4857 / −0.3143 | **−0.2571 ± 0.1512** | +0.0667 ± 0.1627 | −0.1619 ± 0.2806 |
| same, **decay-netted rate** (my pre-registered estimator) | +0.0286 / −0.3143 / −0.3143 | **−0.2000 ± 0.1143** | −0.0286 ± 0.1512 | −0.1238 ± 0.2432 |
| `w40_p1_off` | +0.8721 / +0.1429 / −0.7714 | +0.0812 ± 0.4754 | +0.2190 ± 0.2147 | +0.0905 ± 0.4660 |
| `resoff_p1_off` ⛔DIAG | −0.2571 / +0.2000 / −0.0857 | −0.0476 ± 0.1333 | +0.2000 ± 0.2007 | +0.4667 ± 0.2016 |
| `p1_on` (partition receipt) | −0.0857 / −0.2571 / −0.2571 | −0.2000 ± 0.0571 | +0.2381 ± 0.0687 | **`nan` — ‖∂L/∂atoms‖ ≡ 0 at all 41 readings** |

- **Stratified exact permutation test** (within-seed relabelling, primary proxy, p1_off): observed
  mean ρ = −0.2571, **null sd 0.2580, two-sided p = 0.349**, null 95 % interval **[−0.505, +0.505]**.
- **My pre-registered floor-censoring rule was not triggered**: 0/705 live-well readings below
  1e-30 (min live depth 2.506e-04). All 3 seeds are scored; no seed is "uninformative" on that rule.

### ⛔ VERDICT: `NO_USAGE_STRUCTURE` — the registered branch, on both estimators and both proxies.
ρ ≥ +0.5 (confirm) not reached; ρ ≤ −0.3 (refute) not reached (−0.2571, and −0.2000 netted).

> ### **QUOTATION CONSEQUENCE, stated explicitly as the task requires:**
> **The caveat "depth is NOT quotable as feature importance" (charter §A21) REMAINS ACTIVE.**
> The registered lift condition is the *refutation* branch (ρ ≤ −0.3) and it was not met. My §6
> analysis gives an additional, independent reason not to lift: the confirmatory usefulness proxy
> has zero reliability, so there is no validated well-level "importance" measure in this rig at all.
> **I recommend the Hub does NOT lift it. The Hub lifts, not me.**

**My own pre-registered prediction is partially refuted, and I report it as a finding.** I
registered "|ρ| < 0.3 (no-structure) on **2/3 seeds**". Measured: **1/3** (+0.029 ✓, −0.486 ✗,
−0.314 ✗); netted also 1/3. The *aggregate* form of my prediction lands in the band, the *per-seed*
form does not. The per-seed commitment was the wrong one at n = 6, where the exact null sd of a
single-seed ρ is **0.447** — my own prereg made the same n-blind error I flagged in F2. My
mechanistic derivation (erosion carried by shared φ, not per-well credit) is *consistent* with the
result but is **not** evidenced by it: the mechanism row (ρ_grad −0.1619 ± 0.2806) is itself a null.

## 6. ⭐⭐ Q2 — IS `NO_USAGE_STRUCTURE` A NULL OR AN UNDERPOWERED READING? (POST-HOC, labelled)

⛔ **This whole section is POST-HOC** (not in `PREREG-Adjudication.md`). It changes **no verdict**;
it prices the one in §5. Simulation: `power_pricing.py`, 200 k draws/row, Gaussian copula, seed
20260805.

**(a) Instrument reliability — measured, not assumed.**

| proxy | estimator | p1_off (primary) | reading |
|---|---|---|---|
| read-selection | split-half over readings, Spearman-Brown | ρ_½ 0.486/0.725/0.290 → **rel 0.648** | a real, stable well-level signal |
| erosion rate | odd-vs-even readings, Spearman-Brown | ρ_½ 0.771/0.086/0.771 → **rel 0.633** | usable |
| **leave-one-well-out** | **ICC(1,1) over the 4 checkpoints** | **−0.205 / −0.218 / −0.248 (3/3 seeds)** | ⛔ **between-well variance < within-well noise ⇒ reliability ≤ 0** |

⇒ **A4: ρ(LOO) = +0.0667 is not a null — it is undefined.** With zero reliability the attenuation
ceiling on that proxy is **0.000**: no true correlation of any size could have been observed.
(|loo_delta_bpc| median 1.15e-04, max 8.71e-04 — at the engineer's own reported ~1e-4 cross-process
float noise floor, measured with `--loo-batches 2`.)

**(b) Construct validity — the two registered "usefulness" measures are ANTI-correlated.**
ρ(read-selection, LOO) per seed = **−0.257 / −0.600 / −0.657 → −0.505 ± 0.125, 4.04 SE, 3/3 seeds
negative** (p1_off). They are not two views of one construct. Also: only **10/18 wells have
loo_delta_bpc > 0** — 8/18 wells are *net-harmful*, consistent with the standing memory-is-a-net-cost
finding (K4 margins +0.105/+0.124/+0.226). The hypothesis "most-useful wells erode fastest"
presupposes a usefulness gradient that is, at this rig, half absent and half unmeasurable.

**(c) The registered decision rule is self-capping — more data does NOT fix it.**
P(observed mean ρ ≥ +0.5 \| **true ρ = exactly the registered +0.5**):

| n_wells × seeds, reliability | 6×3, rel 1.0 | 6×3, rel 0.648 | 20×3, rel 1.0 | 100×3, rel 1.0 | 100×10, rel 1.0 | 6×30, rel 1.0 |
|---|---|---|---|---|---|---|
| P(confirm) | 0.424 | **0.282** | 0.439 | 0.468 | 0.432 | 0.171 |

Because the rule thresholds **at** the effect size, P(confirm) is capped near 0.5 by construction at
any n. Symmetrically, P(**false** refute \| true ρ = 0) = **0.123** at the current 6×3.

**(d) But the data are NOT uninformative — they exclude the Head's registered effect size.**
P(mean ρ ≤ the observed −0.2571 \| true ρ = +0.5):

| combined reliability (sel × rate) | 1.000 | 0.648 | **0.410 (measured: 0.648×0.633)** | 0.300 | 0.200 |
|---|---|---|---|---|---|
| one-sided p | 0.0033 | 0.0102 | **0.0221** | 0.0320 | 0.0466 |

⇒ **ρ_true ≥ +0.5 is excluded at p ≈ 0.02** at the empirically-measured reliability, and the
exclusion survives down to a combined reliability of 0.2. Meanwhile the 95 % compatibility ranges
(n=6, 3 seeds, rel 0.648) show what is *not* resolved:

| true ρ_s | 0.0 | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 |
|---|---|---|---|---|---|---|
| 95 % range of the observed mean-of-3 ρ | [−0.505, +0.505] | [−0.371, +0.619] | [−0.295, +0.657] | [−0.219, +0.714] | [−0.143, +0.752] | [+0.029, +0.829] |

The observed −0.2571 is inside the range for **true ρ ∈ [0, 0.4]** and outside it for **ρ ≥ 0.5**.

⭐ **Answer to Q2, precisely:** `NO_USAGE_STRUCTURE` is **the correct registered branch and a
half-informative reading — not a clean null and not simply "unmeasurable."** The rig **can**
exclude the Head's hypothesis at the strength it was registered (ρ ≥ +0.5, p ≈ 0.02 on the primary
proxy) and **cannot** distinguish ρ_true = 0 from |ρ_true| ≤ 0.4. The engineer's closing line "not
confirmed and not refuted — it is unmeasurable at 6 wells/seed" is **too pessimistic on the primary
proxy and too generous on the LOO proxy** (which is genuinely unmeasurable, ρ ceiling 0).

**(e) The price of a rig that could answer the weak version (needs registering before it runs).**
1. **Re-specify the rule first — it is free and it is the binding constraint.** Replace
   "confirm iff mean ρ ≥ +0.5" with a **test against 0** (Fisher-z CI, two-sided 5 %). Under a
   test-against-0 at true ρ_s = 0.5 with 80 % power: need `seeds × (n_wells − 3) ≥ 26` at perfect
   reliability → **3 seeds × 12 wells, or 6 wells × 9 seeds**; at the measured rel 0.648, `≥ 43` →
   **3 seeds × 17 wells, or 6 wells × 14 seeds**. Adding wells to the *current* rule buys nothing.
2. **Fix the LOO proxy before buying wells** (negative ICC): more LOO batches/checkpoints, or drop
   it and register a single primary measure. As shipped it cannot support any correlation.
3. **Establish well identity.** Slot index is only a partial proxy for item identity: same-slot
   site displacement between consecutive readings has median **0.443 / 1.525 / 0.533** (p1_off
   s0/s1/s2) against a between-slot spread of **1.084 / 2.278 / 1.668** — ratios **0.41 / 0.67 /
   0.32**, with `atom_place_radius = 0.30`. A slot's occupant changes materially over a run, so
   both "the well's erosion rate" and "the well's usefulness" are slot-level mixtures. An item-id
   key in the telemetry is a prerequisite, not an upgrade.
4. Only then: raise `capacity`/`budget` or pool lanes to get n_wells ≥ 17 (both change the rig).

## 7. ⭐ Q4 — THE P-RESIDUAL INTERACTION ROW, RE-SCORED UNDER MY F2 REPAIR

Registered (`PREREG-AntiErosion` §3, last item): *"partition-ON final depth ≥ residual-only final
depth (0.132 banked) on ≥2/3 seeds. If partition-ON depth COLLAPSES below the partition-OFF arm, P1
is disproved as specified and P3's coefficient form gets priced."*

| scoring | reference | result | count |
|---|---|---|---|
| (a) engineer's, as registered | per-seed vs the **pooled mean 0.1321** | 0.00677 ✗ / 0.14552 ✓ / 0.19032 ✓ | **2/3 met** (seeds 1, 2) |
| (b) **F2 repair — paired per-seed**, registered (unmatched) horizon | banked residual-on 0.00213/0.27318/0.12096 | 0.00677 ✓ / 0.14552 ✗ / 0.19032 ✓ | **2/3 met** (seeds **0, 2**) |
| (c) F2 repair at a **matched step count** (p1_on @200 vs banked @200) | same | 0.00096 ✗ / 0.26988 ✗ / 0.06482 ✗ | **0/3**; ratios 0.449/0.988/0.536, geo **0.620× (−2.00 SE)** |
| (d) **the disproof clause** (within-wave, paired, clean) | p1_off final | ON/OFF 1.083 / 0.487 / 5.444 | ON below OFF on **1/3**; geo **1.421×** (w4), **3.628×** (w40, 3/3) |

**Ruling.** ⭐ **The "met" verdict SURVIVES my own F2 repair** — 2/3 either way — but **on different
seeds** (0 and 2 rather than 1 and 2), which is the expected signature of a per-seed quantity that
had been scored against a pooled mean. Row (c) is **not** a verdict: matching the step count does
not match the run, because the banked 200-step arm completed a 200-step warmup-cosine schedule while
the C2W6 snapshot at step 200 sits mid-1000-step schedule at high LR. Horizon and LR schedule are
confounded in opposite directions in (b) and (c), so **the cross-wave leg of this row is not
decision-grade in either direction** and I would not quote it.

⛔ **The P3 re-price trigger did NOT fire, and this is the clean leg.** The disproof clause is a
within-wave paired comparison with no cross-wave confound: partition-ON is below partition-OFF on
**1/3 seeds only**, geo **1.421×** deeper at w4 and **3.628× on 3/3 seeds** at w40. **The partition
did not starve the write of its one useful gradient.** No P3 re-price is triggered by this data;
the Head is not being handed a trigger.

## How I verified (commands + observed output)

```
git log --oneline -1                       -> d1149a4  (the Hub's C2W6 merge)
.venv/bin/python .claude/outputs/c2w6-erosion-adjudication/rederive_gate.py
  -> n_records = 21; worst relative deviation over 21 records x 5 scalars: 0.00e+00
  -> p1_off final/200 [9.7815 0.9035 0.5305] arith 3.739 +- 3.023 GEO 1.674
  -> p1_on vs p1_off dbpc -0.004853 +- 0.000780 (6.23 SE, 3/3 better)
  -> w40 depth ON/OFF [21.527 1.516 1.464] geo 3.628x [1.49, 8.84] (3/3 >1)
PYTHONPATH=... .venv/bin/python .../rederive_i2.py
  -> median_rel_drop_own median 0.039211 (predicted 1-exp(-0.04) = 0.039211), 717 readings
  -> max |median(live wells) - curve.depth_median| = 0.000e+00 (41 readings)
  -> all 12 cell-seed rho triples: [vs harness: MATCH]
  -> exact null n=6: sd 0.4472; stratified perm: obs -0.2571, null sd 0.2580, p = 0.3487
PYTHONPATH=... .venv/bin/python .../rederive_proxies.py
  -> rho(sel, loo) p1_off [-0.257 -0.600 -0.657] = -0.5048 +- 0.1249 (4.04 SE)
  -> LOO ICC(1,1) [-0.205 -0.218 -0.248]; read-sel Spearman-Brown +0.648
  -> P-residual (a) 2/3 seeds [1,2] | (b) 2/3 seeds [0,2] | (c) 0/3 | (d) ON<OFF 1/3
.venv/bin/python .../power_pricing.py   -> the §6(c)(d)(e) tables
# §3.1 trough (inline, same loader):
  -> p1_off min/untrained [0.0089 1. 0.0532] geo 0.0779 (-1.85 SE), argmin steps [150, 0, 275]
  -> d200/d_untrained [0.0138 2.0444 0.2648] geo 0.1955 (-1.13 SE)
  -> adjacent-reading noise (steps>=600) 1.33x / 1.10x / 1.05x
```
⛔ **No model was run and nothing was trained** (JAX was never imported); no diagnostic re-run was
needed, so none was declared. Total analyst compute: numpy, < 3 min.

## Git footprint
**None.** No tracked file touched, no branch, no commit, nothing pushed. All artifacts under
`.claude/outputs/c2w6-erosion-adjudication/` (gitignored): `PREREG-Adjudication.md`,
`rederive_banked_anchors.py`, `rederive_gate.py`, `rederive_i2.py`, `rederive_proxies.py`,
`power_pricing.py`, `adjudication_figs.png`, and this report.

## Open questions / follow-ups / risks
1. **No code bug found in this pass.** Two *instrument* items for `experiment-engineer`, neither a
   correctness bug in C2W6: (i) the erosion curve should net the designed decay per well before it
   is used for any rate/flattening claim (§4, A1); (ii) the I2 telemetry needs an **item-id** key —
   slot index is a partial proxy (§6e-3) — and the LOO probe needs more batches/checkpoints or
   removal (§6a). Both belong to C2W8/C2W10, where erosion actually has to be measured.
2. **Risk if the rig is scaled without (e).** Buying wells against the current threshold rule is
   money for nothing (P(confirm) 0.42 → 0.47 going from 6 to 100 wells). The rule must be
   re-registered *first*.
3. **The w40 cell is where the science is** and it is the least-powered: n=5–6 wells, 17 readings,
   per-seed ρ spanning +0.872 to −0.771, and the only cell where erosion appears at all (seed 0,
   ×0.0029). C2W8 should put its instrument budget there, not at w4.
4. **A standing exposure I could not close:** every depth claim in this wave rests on 3 seeds with
   within-arm spreads of 13.7× (untrained depth 0.046/0.162/0.249). No 2× depth effect is resolvable
   at n=3 here; only sign consistency is. This is the wave's dominant methodological risk and it is
   unchanged by anything I did.

---

## Proposed handover updates (for the Hub)

**§1.6 / experiments — C2W6 adjudication (status: complete).** Independent re-derivation from raw:
**every engineer and Hub number reproduces digit-for-digit** (21 records × 5 curve scalars, worst
relative deviation **0.00e+00**; all 12 cell-seed ρ triples MATCH; gate legs and `aggregate()`
identical). Run-3 gate verdict in the prereg's own vocabulary: **`FAILS_FLATTEN` at both budgets, on
the E1 (baseline) leg** — confirmed independently, which is the evidentiary basis for the Head's
§A23.2 re-scope. **I2 = `NO_USAGE_STRUCTURE`; the depth-as-importance caveat STAYS ACTIVE** (the
lift condition is the refutation branch and it was not met; a second, independent reason is that the
LOO usefulness proxy has ≤ 0 reliability).

**§5 / provenance — numbers to fold (all `main @ d1149a4`, JAX 0.9.0, seeds 0/1/2, toy 0.16 M,
run-2 config, w4 unless marked):**
- E1 (`p1_off` final/step-200): **9.7815 / 0.9035 / 0.5305**; arith 3.739 ± 3.023; **geo 1.674**;
  **0/3 met raw AND 0/3 after netting the designed decay**.
- Headline (`p1_off` final/untrained): 0.1351 / 1.8471 / 0.1404 → arith **0.708 ± 0.570 (0.51 SE
  from 1, n.s.)**, **geo 0.327 [0.138, 0.778], log-mean/SE −1.29 (n.s.)** — *no significant change
  in either direction under either estimator*.
- ⭐ **The transient (new, §3.1):** `p1_off` minimum-over-run / untrained = **0.0089 (step 150) /
  1.000 (no dip) / 0.0532 (step 275)**, geo **0.0779**, log-mean/SE −1.85; recovery final/min geo
  **4.20×**; reading-to-reading noise only 1.05–1.33×. At step 200 (a fixed reading) geo **0.196×**.
  **The banked N223 anchor was taken at 200 steps — inside this trough.** `p1_on` shows the same
  shape (geo 0.105×, troughs 150/725/375): the partition does not prevent the transient.
- Erosion-tracks-uselessness (the surviving comparative claim): `resoff_p1_off` geo **0.0775×,
  log-mean/SE −3.10 (significant)** vs `p1_off` geo 0.327× (n.s.) — a **4.2×** separation.
- K3 w4 **−0.004853 ± 0.000780 (6.23 SE, 3/3)**; w40 **+0.005676 ± 0.005613 (1.01 SE)**.
- w40 depth protection: ON/OFF **21.527 / 1.516 / 1.464**, paired log-ratio **geo 3.628× [1.49,
  8.84] = 1.45 SE**, sign test 3/3 (p = 0.125) — **a sign statement, not a resolved magnitude**.
- Designed decay verified exactly: predicted per-tick depth drop **0.039211** vs measured median
  **0.039211** over 717 readings; own-leg residual ≤ **3.324e-07**.
- I2 primary: ρ(read-sel) **−0.2571 ± 0.1512** (netted **−0.2000 ± 0.1143**); stratified exact
  permutation **p = 0.349**, null 95 % [−0.505, +0.505]; ρ(LOO) +0.0667 ± 0.1627 **[DO NOT QUOTE —
  proxy reliability ≤ 0]**; ρ(grad) **−0.1619 ± 0.2806** (registered estimator).
- I2 power (post-hoc): P(confirm | true ρ = +0.5) = **0.424** at 6×3 and **0.468** at 100×3 — the
  rule is self-capping; P(false refute | true 0) = **0.123**. But **ρ_true ≥ +0.5 is excluded at
  p ≈ 0.022** at the measured combined reliability 0.41.

**§8 / caveats + corrections (four, each with a proposed owner — the reconciliation list above):**
1. **A2, highest priority:** replace "0.708 ± 0.57× — does not decay, it **recovers**" with the
   **transient-trough form** everywhere it has propagated (engineer's R1/§3, the §7.27 re-scope
   text, any running-log entry). Recommended sentence in report **§3.1**. This is an *upgrade*, not
   a retraction: it restores N223's mechanism as live-but-non-terminal and explains E1's 0/3
   mechanically (a step-200 denominator sitting in the trough), instead of leaving "the premise the
   wave was built on is retired" as the record.
2. **A4:** mark ρ(LOO) = +0.067 as *not a measured correlation* in the engineer's I2 table and
   anywhere it was carried; the proxy's ICC is negative on 3/3 seeds.
3. **A3:** strike or relabel the engineer's §6 "pooled (n wells)" ρ column — non-registered, and it
   flips sign vs the registered estimator on the mechanism row (+0.2198 vs −0.1619 ± 0.2806).
4. **A1:** carry "the erosion curve is not netted of designed decay; `last_write_chunk` drifts" as a
   known instrument limitation of C2W6 (no verdict moves), and make the netting a build requirement
   for C2W8/C2W10.

**§8 / standing caveat — re-affirmed with a second reason:** "depth is NOT quotable as feature
importance" **stays active**. Registered reason: the refutation branch was not reached. Added
reason: no validated well-level importance measure exists at this rig (the two registered usefulness
proxies are anti-correlated at **−0.505 ± 0.125, 4.04 SE**, and one of them has zero reliability).

**Wave-planning note (C2W8/C2W10 scoping).** Before I2 is re-run at any scale, three things must be
re-registered, in this order: **(i)** replace the "confirm iff ρ ≥ +0.5" threshold with a
test-against-0 (the current rule caps its own confirmation probability near 0.5 at any n);
**(ii)** an item-id key in the per-well telemetry (slot ≠ well: same-slot site drift is 0.32–0.67×
the between-slot spread at a place radius of 0.30); **(iii)** a LOO probe with enough batches to
have positive ICC, or a single registered primary proxy. Only then is buying wells (≥ 17/seed at
3 seeds, or 6 wells × 14 seeds) worth the compute.
