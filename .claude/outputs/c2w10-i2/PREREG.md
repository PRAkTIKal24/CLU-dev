# PREREG — c2w10-i2-usage-erosion (results-analyst, filed 2026-08-11 BEFORE any ρ/ICC was computed)

Protocol §5 pre-registration rule. The acceptance criterion of this task is a set of **measured
correlations and a rule keyed on them** ⇒ I commit to numbers and to the estimators, in closed form,
before running the harness that measures them.

## 0. Disclosure — what I had ALREADY looked at when filing this (honesty, not a loophole)

Filed after the mechanical precondition check and a **schema/mechanism** inspection, before any
correlation. Already seen at filing time:

* `USAGE-TELEMETRY.json` top-level: `n_seeds = 3`, `seeds = [0,1,2]`, `n_live_max = 64`,
  `n_live_max_per_seed = {0:64, 1:64, 2:64}`, `n_seeds_meeting_64 = 3`, `commit 6e0c325`.
* Per seed: `n_live_end = 63/63/63`, `usage_summary.n_items = 63/63/63`,
  `n_never_read = 20/44/24`, `read_coverage.rate = 0.4639/0.4520/0.5136`, curve counts 77/69/75.
* **The structural fact that drives my predictions:** the *netted* depth curves are flat to float32
  ULP — per-item relative range `max/min − 1` has median **5.33e-7 / 4.61e-7 / 5.24e-7** and max
  **8.10e-7 / 8.45e-7 / 8.25e-7**; **0 items** on any seed show a >0.01 % increase; min netted depth
  over all items = 0.2691 / 0.7377 / 0.2917 (no censoring anywhere near a floor).
* The rig source (`exp_persistent_store.py` @ `6e0c325`, read-only): **there is no outer objective and
  no optimizer step on store parameters anywhere in the cell.** Depth moves only through (i) the
  designed decay law (which netting divides out exactly) and (ii) interference from neighbouring
  writes.
* **NOT looked at:** any correlation, any ICC, any per-item erosion slope, any rank statistic.

## 1. Population, and the measurement point (registered)

* **Primary population P1** = the live wells at the **final depth-recording point** = exactly the item
  ids in `usage_summary.hits_by_item` (= the end-of-run codebook), **n = 63 per seed**. This is the
  population the banked usage proxy is item-id keyed to (C2W8 §3.4).
* `n_live_max = 64` (3/3 seeds) is the **I2-a precondition value**; the scored `n` is quoted
  separately and every ρ carries the detectable |ρ| computed at **its own n**.
* **Robustness population P2** = wells live at the chunk where `n_live` first reaches 64
  (index 36/38/37 of 48). Reported as a labelled robustness column only.

## 2. Estimators, in closed form

* **`U_i`** = `usage_summary.hits_by_item[i]` — item-id-keyed cumulative `read_hits`. ⛔ Depth never
  enters `U`.
* **`E_i`** (erosion rate, **netted, log scale**): over item *i*'s readings `(c_k, D_k)` with
  `D_k = depth_netted`,
  `β_i = Σ_k (c_k − c̄)(ln D_k − ln D‾) / Σ_k (c_k − c̄)²`, and **`E_i = −β_i`** (nats of log-depth
  lost per chunk; positive = eroding). Items with fewer than **4** readings are excluded from the
  I2-c leg and the exclusion count is reported. The **raw**-curve slope is computed too but is
  **diagnostic only** and never the basis of a verdict (Add.9 §A27.1).
* **Whole-curve reporting (C2W6 E1 erratum guard):** per item I report `max/min` on the netted and
  raw curves and the argmin location, so a transient trough cannot be mistaken for a decay.
* **Censoring fraction** = fraction of scored wells whose netted depth reaches the floor
  (registered floor `1e-30`, C2W6's rule; the float32 netting floor ≈ `5.3e-8` from R5 is quoted
  beside it).
* **`depth_i`** (I2-d) = netted depth at the final reading. Spearman is rank-based ⇒ invariant to the
  log transform, so "log-scale depth" and depth give the identical ρ; stated rather than assumed.
* **ρ** = Spearman (the registered C2W6 estimator), per seed; **2-SE bounds** by Fisher-z:
  `tanh(atanh(ρ) ± 2/√(n−3))`; **detectable |ρ| = 2/√(n−3)** quoted beside every ρ.
* **`ICC(1,1)`** of the usage proxy: one-way random effects over items (targets) × streams (repeated
  measures). Item *i*'s eligible streams are `s ≥ first_seen_stream[i]` over the 6-stream schedule
  (missing ⇒ 0 hits), `k_i = 6 − first_seen_stream[i]`;
  `ICC = (MSB − MSW)/(MSB + (k0−1)·MSW)`, `k0 = (Σk_i − Σk_i²/Σk_i)/(N−1)`.
  Reported beside a split-half (odd vs even stream) Spearman–Brown reliability, the C2W6 form.
* **Reliability of `E`** (needed because a ρ against an unreliable variable is UNDEFINED, not a null —
  Add.9 §A27.1): split-half odd-vs-even readings → `E_odd`, `E_even` per item, Spearman, then
  Spearman–Brown. Also the between-item SD of `E` against the within-item SE of `β_i`.
* **Pooled-across-seed correlations are a NON-REGISTERED estimator**; if reported at all they are
  labelled as such beside the per-seed registered form (C2W6's pooled ρ flipped sign).

## 3. The registered booleans (arithmetic, no interpretation)

* `i2a_pass` = `n_live_max ≥ 64` on ≥3 seeds.
* `i2b_pass` = `ICC(1,1) > 0` on 3/3 seeds.
* **leg 1** (refutation): `lower_2SE(ρ(U,E)) > −0.10`.
* **leg 2** (positive): `lower_2SE(ρ(U,depth)) ≥ +0.30`.
* `lift_rule_satisfied` = `i2a_pass ∧ i2b_pass ∧ leg1(3/3) ∧ leg2(3/3)` → `branch = REFUTE_BOTH_LEGS`.
* `CONFIRM` = `upper_2SE(ρ(U,E)) < −0.20` on ≥2/3 seeds.
* `NOT_RUN` if `i2a_pass` or `i2b_pass` is false. Otherwise `INDETERMINATE`.

## 4. PREDICTIONS (committed; each with its derivation)

| # | quantity | prediction | derivation |
|---|---|---|---|
| P-1 | censoring fraction | **0.000 on 3/3** | min netted depth 0.269/0.738/0.292 ≫ 1e-30 and ≫ 5.3e-8 |
| P-2 | detectable \|ρ\| at n = 63 | **0.258** (= 2/√60); 0.254 at n = 64 | arithmetic |
| P-3 | `E_i` (netted) | between-item spread at the **float-noise floor**: median \|E\| < 1e-6 nats/chunk on 3/3 | the rig has **no outer-loss gradient channel into store depths** (§0), so netted depth is constant up to float32 ULP; the flat curves are already seen |
| P-4 | split-half reliability of `E` (netted) | **≤ 0.20**, plausibly ≤ 0 | E is float noise ⇒ between-item variance ≈ within-item noise |
| P-5 | `ρ(U, E)` per seed | \|ρ\| ≤ 0.25 on 3/3; the 2-SE band contains 0 on 3/3 | correlating a real variable against noise |
| P-6 | **leg 1** | **FAILS on ≥ 2/3 seeds** | leg 1 needs `lower_2SE > −0.10` ⇒ at n = 63 needs ρ ≳ **+0.15**; a noise ρ centred on 0 gives a lower bound ≈ −0.26 |
| P-7 | `ICC(1,1)` of `U` | **positive on 3/3**, in **[0.10, 0.60]** | usage is item-specific and regimes are revisited (schedule 0,1,2,0,1,2), so between-item variance should exceed within-item; but 20/44/24 never-read items compress it |
| P-8 | `ρ(U, depth_netted)` | **positive**, in **[+0.05, +0.45]** on ≥2/3 seeds | deeper/wider wells capture a larger share of the read batch ⇒ mild positive; the ceiling is the 46–51 % launch-point coverage rate |
| P-9 | **leg 2** | **FAILS on 3/3** | leg 2 needs `lower_2SE ≥ +0.30` ⇒ ρ ≳ **+0.52** at n = 63; P-8's band tops out at 0.45 |
| P-10 | **branch** | **`INDETERMINATE`**, `lift_rule_satisfied = false`, `CONFIRM` does not fire | P-6 ∧ P-9 |
| P-11 | the mechanism statement | the Head's I2 mechanism (**gradient magnitude ∝ contribution**) is **structurally absent** in this rig — no optimizer touches store depth — so what I2-c measures here is the *interference* channel, not the optimizer channel | source read, §0 |

**If P-10 is wrong** — i.e. if the branch is `CONFIRM` or `REFUTE_BOTH_LEGS` — that is a finding and I
report it as one, against this filed prediction.

## 5. What this prereg does NOT authorise

⛔ No lift. `lift_rule_satisfied = true` would be a **measurement**, not a lift: the Hub proposes, the
Advisor amends charter §A23.5, this spoke does neither. ⛔ No performance claim, no CLU verdict, no
cell run — instrument re-measurement only.
