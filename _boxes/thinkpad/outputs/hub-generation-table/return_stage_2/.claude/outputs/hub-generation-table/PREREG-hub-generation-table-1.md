# PREREG-hub-generation-table-1 — the FACTORED DICTIONARY READ on the generation table

**Hub:** `hub/generation-table` · **Branch:** `hub/generation-table` · **Base:** `8e36bc9` (tree
`345244bb…` == the bundle's `pin_tree` for pin `4125fce7…`)
**Written:** 2026-09-16, **before any number on `eval_a` / `eval_b` / `C` has been computed by this hub.**
**Scored runs spent by this hub at the time of writing: 0.**

> ⛔⛔ **THIS RUN HAS NOT BEEN EXECUTED.** COMMON §1c: *"Do not perform your first scored run until a human
> tells you how many of the three remain for your target."* No human has answered. This file is the
> pre-registration, complete and executable; ⛔ the run is **held** at the budget question.

---

## 0. DIAL DECLARATION (AGENT_PROTOCOL §7)
- **Dial:** compute-adaptive reads — *"can a request-addressed read over a byte-matched store compose an
  unseen factor combination?"*, scored end to end on IMAGES.
- **Laundering control:** ⭐ **three**, all at identical bytes and identical configuration —
  (i) the **atom basis** (M real train rows per well instead of learned modes) = charter §C.1;
  (ii) the **PCA basis** (the sharpest trivial substitute for a dictionary);
  (iii) the **random Gaussian basis** (the D-matched null).
  Plus charter §C.2, **store-removed** (every mode replaced by its well mean).
- **Falsifies the claim:** the arm failing to beat the atom basis beyond seed spread; or the store-removed
  control failing to collapse below the constant-output floor.
- **Does NOT falsify the claim:** losing to `knn_proto`. `knn_proto` decodes an **unbounded real-valued
  channel of D = 2048 floats**; this arm's channel is **K·M = 832 reals**. Losing to a wider channel is the
  metric-native-ceiling theorem, ⛔ not news — ⚠ **but the task's falsifier 1 requires WIN-or-TIE against it,
  so a loss there is reported as a FAILED falsifier regardless of this clause.**

---

## 1. THE ARM, EXACTLY

```
request (6 factor values)
  → Phi_add   : bias + one-hot per factor                                  58 columns
  + Phi_pair  : one-hot per cell of  floor_hue×scale  and  object_hue×scale  160 columns
  → a(request) = Phi_add · C_add  +  (Phi_pair · E) · U        (N, K, M) real coefficients
  → x = Σ_k Σ_m a_{k,m} · mode_{k,m}     ⭐ the SHIPPED gen_compose.clu_wide_vector(store, a)
  → THE frozen v1_capacity_schedule decoder  →  image
  → THE hash-asserted external classifier    →  factor tuple
```

| quantity | value |
|---|---|
| store | `GenStore`, **K = 32** equal wells over a balanced correlation partition, **M = 26** modes |
| partition | ward linkage on \|corr\| of train activations, **dendrogram leaf order cut into 32 equal blocks**, dead channels spread evenly (`chlu/core/gen_partition.py`) |
| codebook | k-means++ init + 25 Lloyd iterations per well on train activations — ⛔ label-free |
| head | `chlu/core/gen_factored_read.fit_factored_head`, **rank = 16**, **ridge = 10.0**, additive block at FULL rank |
| **store bytes** | `M·D·4` = **212,992** |
| **head bytes** | `(58·K·M + 160·r + r·K·M)·4` = `(48,256 + 2,560 + 13,312)·4` = **256,512** |
| **TOTAL bytes** | **469,504** = ⭐ **0.988 × `knn_proto`'s 475,136 B** — ⭐ measured on the LIVE objects by `FactoredRequestHead.byte_ledger()` + `gen_store.store_bytes()`, ⛔ not from the formula alone |
| channel | **832 real coefficients** (⚠ WIDE — ⛔ never quoted in the 30-bit column; `knn_proto`'s is 2048 reals) |
| seeds | 0, 1, 2 — the cell's own |
| code | `chlu/core/gen_factored_read.py`, `chlu/core/gen_partition.py`, `chlu/core/gen_address.py` (new files on this branch) |

⛔ **`chlu/experiments/exp_gen_cell.py` and every other frozen-harness file are UNTOUCHED** (charter §A.4).
The arm is *wired*, not rebuilt: it calls the harness's own `render`, `score_images`, `distinct_counts`,
`load_external_classifier`, `resolve_decoder`, `select_store_rows`, `gather_features` and `split_fit_val`.

**Everything selected on DEV, and nothing else:** `K`, `M`, `rank`, `ridge`, and the **two** pair terms.
Selected on Montero-shaped DEV corners cut from inside `split.train_rows` and held out of the DEV store —
⛔ **never on `eval_a`, `eval_b` or C.** The pair ranking came from DEV corner 0; corners c3–c5 were held out
of every selection decision and are the basis of the prediction below.

---

## 2. WHAT WILL BE RUN

1. `python -m chlu.experiments.exp_gen_cell --dataset shapes3d --pretext recon --stage cell --seed {0,1,2}
   --out $CHLU_HUB_OUTPUTS/hub-generation-table/scored/cell` — ⭐ **the pinned harness, unedited**, producing
   `clu`, `store_removed`, `knn_proto`, `lace`, `ebm_compose` (and `x0` if the budget allows the extra time).
2. `$CHLU_HUB_OUTPUTS/hub-generation-table/scored/run_arm.py` — my arm and its four controls, in the same
   process family, on the same rows, through the same decoder and classifier objects.
3. `--stage aggregate`.

⭐ **Both legs are ONE scored run**, because both are computed on the same `eval_b` rows in the same sitting.
⭐ **The rivals are re-measured HERE, on this machine**, and the published Mac numbers are quoted beside them
via the harness's own `regression_check`. COMMON §4 forbids mixing Mac- and Windows-produced artefacts inside
one comparison, so **the like-for-like claim is against my own run's rival rows**, ⛔ not against the
published table.

---

## 3. THE BAR — ⛔ committed before the run, and ⛔ never moved afterwards

**Headline metric:** external-classifier composition-B aggregate, 3 seeds, mean ± sd, ⭐ **with the distinct
factor-tuple count beside every single value.**

| row | must be | rule |
|---|---|---|
| `store_removed` | **beaten outright** | beyond seed spread |
| `ebm_compose` | **beaten outright** | beyond seed spread |
| `lace` (CELL instrument) | **beaten outright** | beyond seed spread |
| `clu` (the shipped 30-bit read) | **beaten outright** | beyond seed spread |
| matched-bytes **atom-mean lookup** | **beaten outright** | charter §C.1 |
| ⭐ **my own atom-basis control at the same configuration** | **beaten outright** | charter §C.1, the sharper form |
| ⭐ **my own PCA-basis control** | **beaten** | the laundering control; a loss here means the store is "any subspace" |
| §C.2 **store-removed at the same coefficients** | **must COLLAPSE below 0.5278** | charter §C.2 |
| ⭐⭐ `knn_proto` **0.8650 ± 0.0122** | **WIN if Δ > 0.020**; **TIE (favourable) if `0 < Δ ≤ 0.020` with 3/3 seeds same sign**; ⛔ otherwise **LOSS** | the task's ceiling rule, `m = 0.020` |
| ⭐ **constant-output floor 0.5278** | must be cleared with **distinct tuples ≫ 2.67** | the task's falsifier 2 |

⛔ **A number quoted without its distinct-tuple count is void.** ⛔ `x0`'s 0.6852 is never quoted without
"**at 8× bytes (4,194,304 B)**". ⛔ `lace` is 0.3117 on this instrument; 0.3241 is the BASELINES instrument at
688,128 B and is not comparable.

---

## 4. THE PREDICTION — ⭐ committed now, derived from DEV, ⛔ not adjusted afterwards

DEV is optimistic in absolute level (the frozen encoder/decoder were fitted on the train split that *contains*
the DEV corners) but it is calibrated: `knn_proto` reads **0.8302 ± 0.0770** across six DEV corners against
its published B value of **0.8650 ± 0.0122**, i.e. DEV runs ≈ **0.035 low** for that arm; the shipped `clu`
argmax reads **0.4722** on DEV against **0.5949** published, i.e. ≈ 0.12 low (B's `shape` column is free — one
class — while a DEV corner's is not). Applying the *smaller*, arm-appropriate offset:

> ### ⭐ PREDICTION, pre-registered
> 1. **The arm scores `0.82 ± 0.05` on composition-B, with `45 – 60` distinct factor tuples of 72.**
> 2. **`Δ` against `knn_proto` is NEGATIVE and small: my point estimate is `−0.03`, 90 % interval
>    `[−0.10, +0.02]`.** ⇒ **I predict falsifier 1 FAILS at the `knn_proto` clause.**
> 3. **The arm beats the shipped `clu` row (0.5949) by more than +0.15**, and beats the matched-bytes
>    atom-mean lookup (0.3927) by more than +0.35.
> 4. **The §C.2 store-removed control lands below 0.35** (DEV: 0.20–0.30).
> 5. **The arm beats its own atom-basis §C.1 control by `+0.02 … +0.07`** (DEV held-out: +0.049).
> 6. **`store_removed` on the shipped `clu` path reproduces ≈ 0.12** and `knn_proto` reproduces within the
>    harness's own `rival_regression_tolerance` of 0.05 of its published value.

⭐ **A prediction that survives is evidence; one that fails is a finding.** Both are reported.

---

## 5. WHAT WOULD MAKE THIS RUN WORTHLESS — named in advance
- ⛔ Moving `m`, the byte budget, the pair set, `K`, `M`, `rank` or `ridge` **after** seeing a B number.
- ⛔ Quoting the arm against the **published** Mac rival rows instead of this run's own rival rows.
- ⛔ Reporting the arm's aggregate without its distinct-tuple count, or without its **total** byte ledger
  (store + head) beside `knn_proto`'s 475,136 B.
- ⛔ Reporting a §C.1 or §C.2 control that was computed at a *different* configuration from the arm.
- ⛔ Presenting a mid-0.6 as progress: the constant-output floor on this metric is **0.5278**.

---

## 6. STATUS

| | |
|---|---|
| scored runs spent by this hub | **0** |
| this pre-registration executed | ⛔ **NO — held at COMMON §1c** |
| blocking question to the operator | **How many of the 3 scored runs for target `generation-table` remain unspent across all machines?** |
| what is ready | the arm, its four controls, the assertions, the byte ledger, the runner script, this bar |


---

## APPENDIX A — DEV OUTCOME, appended 2026-09-17T00:15Z (⛔ the body above is NOT edited)

Charter §A.3: *"A hub may not change a bar it has already seen a result against. Amendments are appended,
dated, never edited."* Nothing above this line has been touched. ⛔ Still **0 scored runs**; nothing below is
a number on `eval_a` / `eval_b` / `C`.

**The pre-registered configuration, measured on 49 Montero-shaped DEV corners (seed 0, production objects):**

| row | aggregate | sd | distinct tuples / 72 | bytes |
|---|---|---|---|---|
| §C.2 store-removed | 0.2860 | 0.0981 | 8.7 | 469,504 |
| best constant | 0.5278 | — | 1 | 0 |
| the shipped ARGMAX read, same store | 0.6314 | 0.0719 | 18.9 | 469,504 |
| random basis | 0.6623 | 0.0792 | 33.4 | 469,504 |
| PCA basis | 0.7055 | 0.1149 | 48.7 | 469,504 |
| atom basis (§C.1) | 0.7954 | 0.0806 | 51.9 | 469,504 |
| **THE ARM** | **0.8135** | 0.0647 | **49.9** | **469,504** |
| `knn_proto` | 0.8307 | 0.0621 | 51.8 | 475,136 |

**Against prediction 2** — pre-registered as *"Δ against `knn_proto` is NEGATIVE and small: point estimate
−0.03, 90 % interval [−0.10, +0.02]"*:

> ⭐ **realised DEV `Δ = −0.0172`, SE 0.0089, 95 % CI [−0.0347, +0.0003], 16/49 paired wins, t = −1.93.**
> **The prediction SURVIVES** — the realised value sits inside the interval and within 0.013 of the point
> estimate — and it predicted the right conclusion: **falsifier 1 fails at the `knn_proto` clause.**

**Against prediction 4** (§C.2 below 0.35): realised **0.2860**. ✅
**Against prediction 5** (§C.1 margin +0.02…+0.07): realised **+0.0180 [+0.0005, +0.0356]** — ⚠ at the very
bottom of the predicted band and **marginal**, not comfortable. Reported as PASSED-BUT-MARGINAL.

⛔ **The bar in §3 is unchanged and remains binding if this run is ever executed.** ⛔ `K`, `M`, `rank`,
`ridge`, the pair set, `m` and the byte budget are frozen as written above.

## APPENDIX B — a SECOND, INDEPENDENT blocker, found 2026-09-16T23:44Z

`exp_gen_store._assert_cache_reencodes` — the pin's own re-encode gate — **FAILS on this machine on all three
seeds** (51 / 81 / 59 float16 ULP against a bound of 1; `max |Δ| = 2⁻⁹`; encoder parameter hashes all match
their cache manifests). `exp_gen_cell.run_seed:1780` calls it with the default `will_extend=True`, before any
arm and before the decoder ceiling, so **the scored cell aborts on this box on every seed having computed no
held-out number**. Seeds 1 and 2 would additionally need **32,199 / 32,241** rows encoded here, which the gate
re-checks at full strength at the point of extension.

⇒ ⛔ **This pre-registration is blocked twice over:** by COMMON §1c (unknown scored-run budget) **and** by a
hard, pinned, measured gate. ⛔ Neither was worked around. See `LOG.md` 23:44Z for the numbers and the
one-line remedy that belongs to the harness's owner, not to a hub.
