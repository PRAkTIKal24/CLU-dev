# c2w10-i2-usage-erosion — results-analyst report

**Task + acceptance criterion:** re-measure I2 (usage vs erosion) at the only live-well count that can carry it, compute the registered LIFT-RULE booleans, emit `I2-VERDICT.json`. **Status: done.**
**⚖ BRANCH = `INDETERMINATE` · `lift_rule_satisfied = false` · `i2a_pass = true` · `i2b_pass = true` · CONFIRM fires 0/3.** The §A23.5 caveat therefore stays exactly where it is.
**One line, as required: the Hub proposes; the Advisor amends charter §A23.5; this spoke does neither.**

## ⚠ DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary; detail in §7)
- **(R1)** ⛔ **I2-c's MECHANISM is absent by construction at this rig** — `exp_persistent_store.py` @`6e0c325` runs **no outer objective and no optimizer step on store parameters**, and L5 records the cell never rewrites a live well ⇒ the Head's channel (gradient magnitude ∝ contribution) cannot act. Add.7 ruling 5 deferred I2 here for **live-well count**; the count is fixed (57–60 scored), the **channel** was never checked at deferral. I2-c's registered arithmetic ran and is reported; **the mechanism is a declared NOT-RUN at this rig, not a null.**
- **(R2)** Netting makes depth **time-invariant** here (median \|last/first − 1\| = **4.8e-7 / 4.0e-7 / 4.5e-7**) ⇒ **I2-d as registered tests WRITE-TIME depth**, not live depth. Any re-registration must name which depth it means.
- **(R3)** The "≈ 0.25 detectable \|ρ\| at n = 64" figure in PREREG-C2W10 §5 / the task file should be quoted as **0.272 / 0.265 / 0.272**: `n_live = 64` is transient (`n_live_end = 63` on 3/3) and the ≥4-readings rule scores **57 / 60 / 57**.
- **(R4)** ⛔ Nobody may quote `ρ(U, depth_raw) = +0.18 / +0.22 / +0.19` as positive evidence: it is below the detection floor **and** confounded (ρ(E_raw, age) = **+0.74 / +0.94 / +0.85**).

## ⭐ DIAL DECLARATION (echoed before the first result)
- **Dial:** none — **instrument re-measurement.** No performance claim, no CLU verdict, no cell run.
- **Laundering control:** N/A; the controls are statistical (ICC, the power precondition, the two-sided lift rule).
- **Falsifies:** I2-a or I2-b failing ⇒ UNDERPOWERED/UNDEFINED, a declared NOT-RUN. *(Neither failed.)*
- **Does NOT falsify:** an INDETERMINATE result — the registered modal outcome (Hub prior 0.60). **That is what was measured.**
- ⛔ Depth is not feature importance (§A23.5 ACTIVE).

---

## 0. ⛔ MECHANICAL PRECONDITION — checked FIRST, measured values quoted

`.claude/outputs/c2w10-lifecycle/USAGE-TELEMETRY.json` **exists** (416 418 B, mtime 2026-08-10 19:09).

| condition | required | **measured** | met |
|---|---|---|---|
| `n_live_max` | ≥ 64 | **64** (`n_live_max_per_seed = {0: 64, 1: 64, 2: 64}`) | ✅ |
| `n_seeds` | ≥ 3 | **3** (`seeds = [0,1,2]`, `n_seeds_meeting_64 = 3`) | ✅ |

⇒ **not BLOCKED**; the spoke proceeds. (Had either failed, the report would be `BLOCKED` per the task.)

## 1. Flag provenance (every number in this report)

| item | value |
|---|---|
| artifact | `.claude/outputs/c2w10-lifecycle/USAGE-TELEMETRY.json`, **read-only**, produced by `exp_persistent_store (MECHANICS)` at commit **`6e0c325`** (engineer branch `agent/experiment-engineer/c2w10-lifecycle-mechanics`) |
| seeds | **0, 1, 2** (3 seeds; every headline number is per-seed, never single-seed) |
| rig | `CluSystem`, learned `V_θ` (`DesignFreedomPotential`, rung `free_mlp`, family `atoms`), `atom_site_local_init=True`, `atom_kernel=wendland` cutoff 2.5, `atom_width = 1.5 × spacing` = 0.6391, **`addr_dim = 12`**, `n_atoms = 32768`, `dim = 13` |
| store / lifecycle | `capacity 72`, `budget = well_budget = 64`, `leak 0.02`, `stage_lifetimes=True`, **`gamma_phi` ON**, `h_hi 2 · h_lo 1 · window 2 · d_dwell 3 · d_demote 2 · k_streams 3 · trash_criterion last_k_streams · censoring_guard True · f_max 0.25 · refresh_monotonic False · refresh_max_gain 4.0` |
| admission | `d_safe_frac = **0.60**` (`d_safe_override 0.25566`; the 0.88 companion is in `run1/`) |
| read/write | chunk `C = 8`, `offers_per_chunk 3`, **`write_steps = 40` (N94 floor)**, `read_steps 200`, `address_steps 100`, `read_batch 128` |
| address block | cheap **unfitted** random projection (§A31.4), 0 fit steps |
| stream | synthetic regime-switcher, schedule **(0,1,2,0,1,2)** = 6 streams / 5 change points / 3 revisits, `n_anchors 96`, `jitter 0.02`, `n_per_stream 64`, 384 instances, 48 chunks, `decimation m = 1`, `persistent_store = 1` |
| ⛔ venue | **synthetic = MECHANICS instrument, NEVER a claim venue (§A14.8)** |
| my analysis env | `/Users/user/Desktop/CHLU/.venv/bin/python` **3.11.13**, numpy **2.4.1**, matplotlib **3.10.8** (main venv reused, no `uv sync`); **no JAX, no model run, no tracked code touched**; repo HEAD during analysis `7fcef50` |
| my scripts / seeds | `.claude/outputs/c2w10-i2/i2_analysis.py` (permutation seed `20260811 + seed`, `n_perm = 20000`), `make_figs.py` |
| commands | `./.venv/bin/python .claude/outputs/c2w10-i2/i2_analysis.py` → `I2-VERDICT.json`, `per_well_table.csv`, `run_log.txt`; `./.venv/bin/python .claude/outputs/c2w10-i2/make_figs.py` → `i2_figs.png` |

**PREREG:** `.claude/outputs/c2w10-i2/PREREG.md`, filed **before** any ρ or ICC was computed, with its pre-filing disclosure (schema + the netted-curve flatness + the source read were already seen; **no correlation was**). Scorecard in §6.

## 2. Estimators (closed form, as registered in PREREG §2)

* **Population P1 (primary)** = live wells at the final depth-recording point = the ids in `usage_summary.hits_by_item` (= end-of-run codebook), n = 63/63/63; **scored** after the ≥4-readings rule: **57 / 60 / 57** (dropped 6 / 3 / 6 short-curve items).
* **`U_i` = item-id-keyed `read_hits`** (registered primary proxy). ⛔ Depth never enters `U`.
* **`E_i` = −β_i**, `β_i = Σ(c_k−c̄)(ln D_k − ln D‾)/Σ(c_k−c̄)²` on the **NETTED** curve `D = depth_netted`, **log scale**, nats per chunk, positive = eroding. The raw-curve slope is computed as a **diagnostic only**.
* **ρ** = Spearman per seed; 2-SE bounds `tanh(atanh ρ ± 2/√(n−3))`; **detectable \|ρ\| = 2/√(n−3)** quoted beside every ρ. Pooled-across-seed forms are **non-registered** and labelled (rider 1).
* **ICC(1,1)** of `U`: one-way random effects, items × the 6-stream schedule, item *i* eligible on streams `s ≥ first_seen_stream[i]` (missing ⇒ 0 hits), unbalanced `k0`.

---

## 3. RESULTS

### 3.1 I2-a POWER — **PASS**

| seed | `n_live_max` | `n_live_end` | **n scored** | **detectable \|ρ\| = 2/√(n−3)** | read-coverage ceiling on `U` | never-read (scored) |
|---|---|---|---|---|---|---|
| 0 | 64 | 63 | **57** | **0.272** | 0.4639 | 20 |
| 1 | 64 | 63 | **60** | **0.265** | 0.4520 | 44 |
| 2 | 64 | 63 | **57** | **0.272** | 0.5136 | 24 |

`i2a_pass = true`. ⚠ The `read_coverage` rate (0.45–0.51) is the launch-point ceiling on the proxy's resolution and travels with every `read_hits` statement; 3225/3297/2926 of 6016 read events were **unassigned** (credited to nobody, by design). Seed 1 has **44/60 never-read wells**, i.e. a large tie block at `U = 0` that costs effective power beyond the nominal n.

### 3.2 I2-b ICC — **PASS on 3/3** (and this is an instrument advance over C2W6)

| seed | **ICC(1,1) of `U`** | k0 | N | status | split-half (odd/even stream, Spearman–Brown) |
|---|---|---|---|---|---|
| 0 | **+0.2797** | 3.90 | 55 | **usable** | 0.395 |
| 1 | **+0.4466** | 4.18 | 59 | **usable** | 0.638 |
| 2 | **+0.4665** | 4.12 | 57 | **usable** | 0.573 |

`i2b_pass = true`. ⛔ **No LOO number is quoted anywhere** — `exp_persistent_store` emits **no leave-one-out telemetry**; the LOO leg is **NOT PRESENT / declared NOT-RUN**, never a null (C2W6's `ρ(LOO)=+0.067` remains UNDEFINED per Add.9 §A27.1). The ICC above is of the **registered primary proxy across streams**, which is the reliability the ρ's actually need.
**Attenuation ceiling** `√(rel_U · rel_E)` = **0.592 / 0.732 / 0.668** — no ρ below could exceed these.

### 3.3 The whole curve, not endpoints (the C2W6 E1 erratum guard) — **no transient exists here**

| seed | items with **any** raw increase | max raw `final/min` | median raw total log-drop | median **netted** total log-drop | netted span in **float32 ULP** (median / max) |
|---|---|---|---|---|---|
| 0 | **0 / 57** | **1.000** | 0.580 nats | **4.81e-07** | 4.12 / 6.79 |
| 1 | **0 / 60** | **1.000** | 0.590 nats | **3.98e-07** | 3.65 / 6.74 |
| 2 | **0 / 57** | **1.000** | 0.620 nats | **4.49e-07** | 3.77 / 6.53 |

The raw curves are **monotone non-increasing on every well of every seed** and the final reading *is* the minimum — so no 112×-style trough can be mistaken for a decay here (an `argmin`-based statistic would have mis-fired on plateaus; ties were handled explicitly). Figure: `.claude/outputs/c2w10-i2/i2_figs.png` (top row: all 57–60 curves per seed, raw blue vs netted red).

⭐ **The substantive structural result: after netting there is nothing left.** The undesigned erosion channel measures a **median 4e-7 nats of total log-drop over a whole 48-chunk run against 0.58–0.62 nats raw** — a ratio of `median |E_net| / |E_raw|` = **6.95e-7 / 6.77e-7 / 6.77e-7**, i.e. **~4 float32 ULP**. 54/58/57 wells show netted *increases* as well as decreases — the signature of round-off, not of physics.

### 3.4 I2-c EROSION — the registered numbers, and why they are not quotable as physics

| seed | **ρ(U, E_netted)** | 2-SE lower | 2-SE upper | detectable \|ρ\| | perm p (2-sided, 20 k) | **CONFIRM (upper < −0.20)?** |
|---|---|---|---|---|---|---|
| 0 | **−0.2396** | −0.4750 | **+0.0278** | 0.272 | 0.0735 | ❌ |
| 1 | **−0.2834** | −0.5052 | **−0.0265** | 0.265 | 0.0286 | ❌ |
| 2 | **−0.3246** | −0.5434 | **−0.0646** | 0.272 | 0.0143 | ❌ |

**`n_seeds_confirm = 0/3` ⇒ the CONFIRM branch does NOT fire** (it needs the upper 2-SE bound below −0.20; the observed uppers are −0.026 to +0.028). The point estimates are negative on 3/3 and are in the same place as C2W6's (`−0.257` primary, `−0.200` netted) — **sign concordance across two rigs, at magnitudes at/below the detection floor.**

⛔ **Do not read these as support for the Head's hypothesis at this rig.** Three measured reasons:
1. **Magnitude/validity:** `E_netted` is 7e-7 of the designed channel (§3.3). A correlation with a 1.5e-8 nats/chunk quantity is a correlation with a rounding signature. (Reliability does **not** rescue it — see §3.6.)
2. **The correlation is largely inherited numerics:** ρ(E_net, E_raw) = **+0.278 / +0.598 / +0.461**, and the partial **ρ(U, E_net | E_raw)** collapses to **−0.152 / −0.139 / −0.221**.
3. **A complete alternative chain, all of it measured:** `U` ↑ with *current* raw depth (**+0.182 / +0.224 / +0.190** — deeper-right-now wells capture more of the read batch), current raw depth ↓ with `E_raw` (**−0.765 / −0.954 / −0.693**), `E_raw` ↑ with age (**+0.736 / +0.942 / +0.853**), and round-off tracks the number of decay multiplies. That chain **predicts a negative ρ(U, E) with no erosion mechanism at all** — i.e. depth→reads, not reads→erosion.

**Censoring (rider 2):** `frac_at_floor(1e-30)` = **0.000 / 0.000 / 0.000**; also **0.000** at the float32 netting floor (5.3e-8, R5). Minimum netted depth = **0.2935 / 0.7377 / 0.3212**. No censored reading enters any ρ, and no censored reading is reported as a point estimate.

### 3.5 I2-d DEPTH–USAGE — a clean null, well inside the detection floor

| seed | **ρ(U, depth_netted)** | 2-SE lower | 2-SE upper | detectable \|ρ\| | perm p | **leg 2 (lower ≥ +0.30)?** |
|---|---|---|---|---|---|---|
| 0 | **−0.0361** | −0.2988 | +0.2318 | 0.272 | 0.793 | ❌ |
| 1 | **+0.1033** | −0.1598 | +0.3528 | 0.265 | 0.427 | ❌ |
| 2 | **+0.0201** | −0.2469 | +0.2842 | 0.272 | 0.884 | ❌ |

Because netting is exact here, **netted depth is time-invariant** (§R2) ⇒ this leg is testing *"were the wells written deep the ones that later get read?"*. **No.** The raw (current) depth version is +0.182/+0.224/+0.190 — **below** the 0.27 detection floor and age-confounded (R4).

### 3.6 ⭐ A methodological finding the Hub should bank (my own prereg refuted here)

I pre-registered that `E_netted`'s split-half reliability would be **≤ 0.20** (it is float noise, so it should not reproduce). **Measured: 0.885 / 0.839 / 0.778 — highly reliable.** The ULP residual is **deterministic and item-specific**, so it split-halves beautifully while being physically meaningless.
⇒ **Reliability (ICC/split-half) is NECESSARY BUT NOT SUFFICIENT to validate an erosion proxy.** The C2W6 lesson ("ICC ≤ 0 ⇒ UNDEFINED") has a twin that this measurement discovered: **a proxy can have ICC ≈ 0.85 and still be undefined as physics.** The second, mandatory check is **magnitude against the designed channel** (here 7e-7 ⇒ fail). I recommend this pairing be carried as an instrument rule.

### 3.7 Riders

* **Rider 1 (estimator hygiene).** ⛔ **NON-REGISTERED pooled-across-seeds estimator, never evidence:** pooled n = 174, `ρ(U,E) = −0.184`, **`ρ(U,depth) = −0.216`**. The pooled depth number **flips sign** against the registered per-seed form (−0.036 / **+0.103** / **+0.020**) — C2W6's warning reproduces exactly. Registered per-seed values in §3.4–3.5 are the only quotable ones.
* **Rider 3 (CONFIRM implications).** The branch is **not** CONFIRM, so the rider's vindication statement is **not** claimed. Neutral, measured note for L3 instead: `U` is **not** independent of depth (ρ(U, depth_raw) ≈ +0.19, ICC(U) ≈ 0.28–0.47), so a `read_hits`-keyed trash criterion carries a **mild depth endogeneity** — worth a watch item, not a defect.
* **P2 robustness** (`P2_robustness.json`; population truncated at the last even chunk with `n_live = 64`: c* = 44/42/46, n = 54/55/57): ρ(U,E) = −0.215/−0.155/−0.325, ρ(U,depth) = −0.075/+0.069/+0.020. **Every registered boolean is unchanged.**

---

## 4. ⚖ THE REGISTERED LIFT RULE — booleans only (I compute; I do not interpret)

| leg | rule | seed 0 | seed 1 | seed 2 | 3/3? |
|---|---|---|---|---|---|
| **I2-a** | `n_live_max ≥ 64`, ≥3 seeds | ✅ 64 | ✅ 64 | ✅ 64 | **PASS** |
| **I2-b** | `ICC(1,1) > 0` | ✅ +0.280 | ✅ +0.447 | ✅ +0.467 | **PASS** |
| **leg 1** | 2-SE **lower** on ρ(U,E) **> −0.10** | ❌ −0.475 | ❌ −0.505 | ❌ −0.543 | **FAIL 0/3** |
| **leg 2** | 2-SE **lower** on ρ(U,depth) **≥ +0.30** | ❌ −0.299 | ❌ −0.160 | ❌ −0.247 | **FAIL 0/3** |
| **CONFIRM** | 2-SE **upper** on ρ(U,E) **< −0.20**, ≥2/3 | ❌ +0.028 | ❌ −0.026 | ❌ −0.065 | **0/3** |

```
branch                = "INDETERMINATE"
lift_rule_satisfied   = false
i2a_pass = true   i2b_pass = true   n_seeds_confirm = 0
```

**Neither leg fires. INDETERMINATE ⇒ the caveat stays ACTIVE — reported as a RESULT, not a shortfall** (it is the registered modal outcome, Hub prior 0.60). And per C2W6's carried condition: **only a positive-structure finding lifts anything; a second no-structure reading leaves the caveat exactly where it is.**

> **Authority (Head ruling 3, 2026-08-10): the Hub proposes; the ADVISOR amends charter §A23.5; this spoke does neither.** `lift_rule_satisfied` is a **measurement**, not a lift — and here it is `false` anyway.

**Artifact:** `.claude/outputs/c2w10-i2/I2-VERDICT.json` — every field computed arithmetically, incl. `n_live_by_seed`, `icc_by_seed`, `rho_U_E_by_seed(+2SE)`, `rho_U_depth_by_seed(+2SE)`, `detectable_rho`, `i2a_pass`, `i2b_pass`, `branch`, `lift_rule_satisfied`, plus `validity_E_netted_by_seed`, `confound_diagnostics_by_seed`, `censoring_by_seed`, `loo_leg`, `mechanism_channel_status`, `pooled_non_registered`.

## 5. How I verified

```
ls -la .claude/outputs/c2w10-lifecycle/            # precondition file exists, 416418 B
python -c "json.load(...)['n_live_max'] ..."       # 64, 64/64/64, n_seeds 3  -> precondition MET
git show 6e0c325:chlu/experiments/exp_persistent_store.py   # read-only: no outer loss, no optimizer
./.venv/bin/python .claude/outputs/c2w10-i2/i2_analysis.py  # exit 0 -> I2-VERDICT.json + per_well_table.csv
./.venv/bin/python .claude/outputs/c2w10-i2/make_figs.py    # exit 0 -> i2_figs.png
```
Two failures I hit and fixed, on the record: (i) my first curve-shape statistic used `np.argmin` and reported "argmin not at the end for 100 % of items", which was **plateau ties**, not troughs — replaced with an explicit monotonicity count (`n_items_with_any_raw_increase = 0/0/0`); (ii) one run exited 1 (`UnboundLocalError`, partial-correlation block ordered after its use) — fixed, rerun clean, exit 0. Everything reported comes from the clean run.

## 6. Pre-registration scorecard (PREREG.md §4 — filed before the measurements)

| # | prediction | outcome | measured |
|---|---|---|---|
| P-1 | censoring 0.000 3/3 | ✅ | 0.000 / 0.000 / 0.000 |
| P-2 | detectable \|ρ\| = 0.258 at n = 63 | ⚠ partial | arithmetic right, **n was 57/60/57** ⇒ 0.272/0.265/0.272 (R3) |
| P-3 | median \|E_net\| < 1e-6 nats/chunk 3/3 | ✅ | 1.85e-8 / 1.62e-8 / 1.65e-8 |
| P-4 | split-half reliability of E ≤ 0.20 | ❌ **REFUTED** | **0.885 / 0.839 / 0.778** → the §3.6 instrument finding |
| P-5 | \|ρ(U,E)\| ≤ 0.25 3/3; band contains 0 3/3 | ❌ **REFUTED** | 1/3 and 1/3 (−0.240/−0.283/−0.325) |
| P-6 | leg 1 fails on ≥2/3 | ✅ | fails **3/3** |
| P-7 | ICC positive 3/3, in [0.10, 0.60] | ✅ | +0.280 / +0.447 / +0.467 |
| P-8 | ρ(U,depth) in [+0.05,+0.45] on ≥2/3 | ❌ **REFUTED** | −0.036 / +0.103 / +0.020 (1/3 in range) |
| P-9 | leg 2 fails 3/3 | ✅ | fails 3/3 |
| P-10 | branch = INDETERMINATE, lift false | ✅ | INDETERMINATE, false |
| P-11 | the Head's mechanism is structurally absent | ✅ (source-verified) | see R1 |

7 confirmed / 3 refuted / 1 partial. **The refutations are the useful part** — P-4 in particular would have licensed a bogus "reliable erosion signal" reading had I checked reliability alone.

## 7. Limitations & confounds (the honest scope of this verdict)

1. **⛔ The strongest limitation, R1:** this rig has **no optimizer channel into store depth at all**. The verdict "INDETERMINATE" is therefore about *this* configuration's measurable channel; **it neither confirms nor refutes the Head's optimizer-erosion hypothesis, which was not exercised.** Anyone reading "I2 re-measured at n ≥ 64 ⇒ INDETERMINATE" as "the hypothesis was tested with power and found wanting" is over-reading it: the **power** problem is fixed, the **channel** problem is now the binding one.
2. **Configuration scope:** persistent-store rig at `d = 12`, `d_safe_frac 0.60`, `write_steps 40`, `read_batch 128`, synthetic regime-switcher (**mechanics instrument, never a claim venue, §A14.8**), 3 seeds.
3. **Proxy ceiling:** launch-point read coverage 0.45–0.51; 20/44/24 never-read wells create large `U = 0` tie blocks; the effective power is below the nominal n, so the honest detection floor is ≥ 0.27, before the 0.59–0.73 attenuation ceiling.
4. **`E` on the raw curve is age-confounded** (ρ(E_raw, age) up to +0.94), which is exactly why Add.9 §A27.1 requires netting — and netting here removes 100 % of the signal, leaving ULP.
5. **The ≥4-readings exclusion** drops 6/3/6 wells (short curves, admitted late). They are the youngest wells; excluding them removes some of the age gradient rather than adding one.
6. Three seeds is the registered n; the per-seed spread on ρ(U,E) (−0.24 → −0.32) is comfortably within one 2-SE half-width, so no seed is an outlier.

## 8. Recommended next experiments (in priority order)

1. **⭐ If I2 is to be answered at all, it needs a rig with the channel.** The minimum viable venue is one where the **outer loss can reach store depth** (partition OFF) *and* live wells are rewritten — i.e. C2W6's `p1_off` arm at C2W10's live-well count. That is a re-price of Add.7 ruling 5: "defer until `n_live ≥ 64`" should become **"defer until `n_live ≥ 64` AND an erosion channel is live"**, and it needs an Advisor ruling before anyone funds it.
2. **Cheap and decisive first:** re-run the C2W6 `p1_off` cell with only the live-well count raised, and compute the identical estimators from this report (they are in `i2_analysis.py`, artifact-driven). If ρ(U,E) stays at ≈ −0.25 there **with a physical E**, that is the first real evidence for the Head's hypothesis; if it moves to 0, C2W6's sign was noise.
3. **Instrument:** if a future wave wants to measure undesigned erosion at all, the store amplitudes must be **float64** (at float32 the netted channel saturates at ~4 ULP, i.e. the instrument's noise floor is above any effect it could see), and the cell should emit **LOO telemetry** so the ICC/LOO pairing from C2W8 §3.4 can be reproduced rather than substituted.
4. **Depth endogeneity of `U`** (rider 3 note): a 1-line check in the lifecycle harness — `ρ(read_hits, depth_raw)` per seed — should be added to the census output so L3's criterion carries its own endogeneity number.

## Git footprint
**None.** No tracked code touched, no branch, no commit, no worktree (task: ZERO worktrees). All artifacts under `.claude/outputs/c2w10-i2/`: `PREREG.md`, `i2_analysis.py`, `make_figs.py`, `I2-VERDICT.json`, `P2_robustness.json`, `per_well_table.csv`, `run_log.txt`, `i2_figs.png`. Repo HEAD during analysis: `7fcef50` (unchanged; `git status` untouched by me).

---

## Proposed handover updates (for the Hub)

**§1.6 (experiments) — add the C2W10 I2 row:**
> **I2 re-measurement (C2W10, `6e0c325`, seeds 0/1/2, persistent-store rig d=12, d_safe_frac 0.60):** power precondition **MET** (`n_live_max = 64` on 3/3; scored n = 57/60/57; detectable \|ρ\| **0.272/0.265/0.272**). **`ICC(1,1)` of `read_hits` = +0.280 / +0.447 / +0.467 (positive 3/3 — the proxy is reliable at this rig, unlike C2W6's LOO).** `ρ(U, E_netted)` = **−0.240 / −0.283 / −0.325** (2-SE uppers +0.028/−0.026/−0.065) ⇒ **CONFIRM fires 0/3**. `ρ(U, depth_netted)` = **−0.036 / +0.103 / +0.020** ⇒ **leg 2 fails 3/3**. **Branch = `INDETERMINATE`, `lift_rule_satisfied = false`** ⇒ **§A23.5 caveat stays ACTIVE**. Censoring 0.000 on 3/3 (min netted depth 0.294/0.738/0.321).

**§5 (provenance) — three things to carry verbatim:**
1. ⛔ **The mechanism was not exercised:** `exp_persistent_store` runs no outer objective and no optimizer step on store parameters; netted depth is flat to **~4 float32 ULP** (median netted total log-drop **4e-7 nats** vs **0.58–0.62 nats** raw; ratio **6.8e-7**). I2-c's arithmetic is reported; **its mechanism is a declared NOT-RUN at this rig, not a null.**
2. ⛔ **Never quote** `ρ(U, depth_raw) = +0.18/+0.22/+0.19` as positive evidence (below the 0.27 floor, age-confounded: ρ(E_raw, age) = +0.74/+0.94/+0.85). ⛔ **Never quote** the pooled estimator — pooled `ρ(U,depth) = −0.216` **flips sign** against the per-seed registered form.
3. ⛔ **No `ρ(LOO)` exists in this artifact** — the LOO leg is NOT PRESENT / declared NOT-RUN here; C2W6's `+0.067` remains **UNDEFINED** (Add.9 §A27.1) and this measurement does not disturb that.

**§8 (instrument rules) — propose a new standing rule (the §3.6 finding):**
> **Reliability is necessary but not sufficient.** C2W6 gave "ICC ≤ 0 ⇒ UNDEFINED". C2W10 gives its twin: `E_netted`'s split-half reliability was **0.78–0.89** while the quantity itself was **4 ULP of float32 round-off**. Every erosion/usage proxy must therefore carry **two** checks: (a) reliability (ICC/split-half), and (b) **magnitude against the designed channel** — a proxy at ≤1e-6 of the designed channel is UNDEFINED regardless of its reliability.

**For the Advisor (the Hub proposes, the Advisor amends §A23.5):** the measured branch is `INDETERMINATE`; **no amendment is proposed** — the caveat stands. What *is* proposed is a **re-scope of Add.7 ruling 5's deferral condition**: the count precondition is discharged, and the binding precondition going forward is **"an erosion channel is live"** (partition OFF and/or live-well rewrites), without which I2 cannot be answered at any n.

**For `experiment-engineer` (no bug found; two instrument requests):** (i) `exp_persistent_store` emits no LOO telemetry — the C2W8 §3.4 ICC/LOO pairing cannot be reproduced from it, only substituted (I used across-stream ICC and said so); (ii) float32 store amplitudes put the netted-channel noise floor at ~4 ULP — any future undesigned-erosion measurement needs float64 amplitudes or it is measuring round-off. Also note for the census: `argmin`-based curve-shape statistics are fooled by depth plateaus (repeat readings between decay ticks) — use an explicit monotonicity count.
