# PREREG-hub-generation-table-2 — the SHARED-BASIS factored read on the generation table

**Hub:** `hub/generation-table-2` (RELAUNCH) · **Branch:** `hub/generation-table-2`
**Base:** `945a3e9` = `base/iter2` (the pin tree `345244bb…` + the five reference-read files + the
Head-authorised §2.2(b) gate patch in `exp_gen_cell.py` only)
**Machine:** FORGIS_14, Windows 11 Pro AMD64, CPU only (R3: every number carries its machine)
**Written:** 2026-09-17, **before any number on `eval_a` / `eval_b` / C has been computed by this hub.**
**Scored runs spent by this hub at the time of writing: 0.**
**Budget (LAUNCH-COMMON §1c): AT MOST ONE scored run this iteration, SEED 0 ONLY.** It will be reported
as *"scored 1/3 — seed 0 only, 3-seed pending full caches"*.

> ⛔ **Body frozen at 13:15Z.** Anything learned afterwards is appended as a dated appendix.
> ⛔ **No bar, bound, criterion or control in §3 is changed after a held-out number exists.**

---

## 0. DIAL DECLARATION (AGENT_PROTOCOL §7)
- **Dial:** compute-adaptive reads — *can a request-addressed read over a byte-matched store compose an
  unseen factor combination?*, scored end to end on IMAGES through the frozen decoder and the gated
  external classifier.
- **Laundering controls** (all at the identical configuration, the identical atom count and a byte ledger
  measured from the LIVE objects):
  1. **§C.1** a dictionary of **real stored rows** in three forms — `atoms_medoid` (the real row nearest
     each learned centre: differs from the arm in **exactly one thing**), `atoms_kpp` (maximally spread
     real rows, the adversarial form), `atoms_random`;
  2. **§C.2** `store_removed` — every atom replaced by its well mean, **the same request coefficients**;
  3. the **D-matched Gaussian null** (`random`);
  4. **R4 `pca_affine`** — mean + M−1 principal directions. ⛔ **A CONTROL ONLY, never an arm** (R4:
     a principal direction cannot be decoded or deleted, and the deletion target needs atoms);
  5. **`ADD_ONLY`** — the identical arm at the identical atom count with the pair block removed, so the
     interaction's contribution is a paired difference rather than an assumption.
- **Falsifies the claim:** the arm failing to beat the §C.1 medoid dictionary beyond seed spread; or the
  §C.2 control failing to collapse below the constant-output floor of **0.5278**; or §C.4 showing the read
  sits on its nearest atom.
- **Does NOT falsify the claim:** losing to `knn_proto`. `knn_proto` decodes an **unbounded real-valued
  channel of D = 2048 floats**; this arm's channel is **1,216 reals**. Losing to a wider channel is the
  metric-native-ceiling theorem, ⛔ not news — ⚠ **but the task's falsifier 1 requires WIN-or-TIE against
  it under R1, so a loss there is reported as a FAILED falsifier regardless of this clause.**
- **Also does NOT falsify it:** the R4 `pca_affine` control scoring above the arm. It is measured, it is
  reported on the same line, and it is not adoptable.

---

## 1. THE ARM, EXACTLY — frozen 13:10Z on the SELECT corners, before any CONFIRM corner was read

```
request (6 factor values)
  -> Phi_add   : bias + one-hot per factor, one reference level dropped     52 columns
  -> Phi_pair  : one-hot per cell of  floor_hue x scale,  object_hue x scale,
                 shape x orientation                                       220 columns
  -> a(request) = (Phi_add  +  Phi_pair . E) . C_add        (N, 1216) real coefficients
                    ^ SHARED BASIS: U := C_add, stored ONCE.  The interaction costs
                      n_pair*n_add*4 = 45,760 B and does NOT scale with the atom count.
  -> x = sum_k sum_m a_{k,m} . atom_{k,m}   <- the SHIPPED gen_compose.clu_wide_vector
  -> THE frozen v1_capacity_schedule decoder  -> image
  -> THE hash-asserted external classifier    -> factor tuple
```

| quantity | value |
|---|---|
| store | `chlu.core.gen_store.GenStore`, **K = 64** equal wells of 32 channels |
| partition | ward linkage on \|corr\| of train activations, dendrogram leaf order cut into 64 equal blocks, dead channels spread evenly — **label-free** |
| dictionary | **NMF** (multiplicative updates, k-means++ init), **A = 1,216 atoms** (M = 19 per well) — the span objective with atoms in the data's own non-negative orthant ⇒ every atom is a valid, decodable, deletable feature vector. ⛔ **Label-free: it sees train activations and nothing else (§B.6).** |
| head | `chlu.core.gen_shared_basis_head.fit_shared_basis_head`, `ridge_add = 10`, `ridge_pair = 100`, additive block at FULL rank, interaction fitted by least squares **in the feature-space metric** |
| **store bytes** | `A · (D/K) · 4` = `1216 · 32 · 4` = **155,648** |
| **head bytes** | `(52·1216 + 220·52)·4` = `(63,232 + 11,440)·4` = **298,688** |
| **TOTAL bytes** | **454,336 = 0.956 × `knn_proto`'s 475,136 B** — ⭐ measured on the LIVE objects by `gen_store.store_bytes()` + `SharedBasisHead.byte_ledger()`, ⛔ not from the formula alone |
| channel | **1,216 real coefficients** (⚠ WIDE — ⛔ never quoted in the 30-bit column; `knn_proto`'s is 2,048 reals) |
| seed | **0 only** (LAUNCH-COMMON §1c; seeds 1–2 have `scope="used"` caches and cannot be covered here without an encode, which is forbidden) |
| code | `chlu/core/gen_shared_basis_head.py`, `chlu/core/gen_wide_read.py` (new, additive, on this branch); `chlu/core/gen_dict_fit.py` (inherited from the stopped iteration-2 predecessor, adopted 11:35Z) |

⛔ **`chlu/experiments/exp_gen_cell.py` and every other frozen-harness file are UNTOUCHED** beyond the
Head-authorised gate patch that already ships in `base/iter2` (charter §A.4). Every primitive is the
harness's own: `render`, `score_images`, `distinct_counts`, `load_external_classifier`, `resolve_decoder`,
`select_store_rows`, `split_fit_val`, `gather_features`, `FactorPrototypes`, `clu_wide_vector`.

### Everything selected on DEV, and nowhere else
`K`, `A`, the head form, the dictionary method, the ridges and the three pair terms were selected on **6
SELECT corners** cut from inside `split.train_rows` and held out of the DEV store — ⛔ **never on `eval_a`,
`eval_b` or C.** **43 CONFIRM corners** were held out of every selection decision and are the basis of §4.

---

## 2. WHAT WILL BE RUN — ONE scored run, two legs, same rows, same sitting

1. **Leg 1 — the pinned harness, UNEDITED:**
   `python -m chlu.experiments.exp_gen_cell --dataset shapes3d --pretext recon --stage cell --seed 0
   --out $CHLU_OUT3/scored/cell`, default arms (`clu`, `store_removed`, `knn_proto`, `lace`,
   `ebm_compose`). ⛔ No `--quick`: a quick number is never reported.
2. **Leg 2 — `relaunch/scored/run_arm.py`:** the arm and its controls, in the same process family, on the
   same `eval_b` rows, through the **same** decoder and classifier objects, plus:
   - ⭐ `clu_argmax_shipped` — the pinned `clu` read recomputed independently, as a cross-check on leg 1;
   - ⭐⭐ **`clu_linear`** — **the SHIPPED store, the SHIPPED bytes, only the READ VERB changed** from
     `read_from_codes(argmax)` to `clu_wide_vector(real-valued request-addressed weights)`. The head is
     additive-only at `n_add · K·M · 4 = 58 · 320 · 4 = 74,240 B` = the shipped addressing head **to the
     byte**;
   - `knn_proto_recheck` — the ceiling recomputed here against leg 1's own row.
3. `--stage aggregate`.

⭐ **Both legs are ONE scored run**, because both are computed on the same `eval_b` rows in the same
sitting. ⭐ **Every rival is re-measured HERE, on this machine** (R3: headline comparisons are same-device
by construction); Mac-published rows are cited as prior rows with provenance, ⛔ never mixed inside a
paired comparison.

### ⛔⛔ THE NO-ENCODE GUARD, in code, before anything is gathered
`run_arm.py` computes `n_shortfall` over `pool ∪ eval_b ∪ MS ∪ fit ∪ val` and **raises `SystemExit` unless
it is 0**. If one row this run needs is not already in the Mac-written cache, the script **STOPS and
reports the shortfall**. ⛔ It never encodes, and it never passes `will_extend=False` with a shortfall > 0
(LAUNCH-COMMON §4). The §5.1 falsifier verdict (`REENCODE-VERDICT-iter2.md`, CLEAR) licenses a
**no-extension** run on this box and nothing else.

---

## 3. THE BAR — ⛔ committed before the run, and ⛔ never moved afterwards

**Headline metric:** external-classifier **composition-B aggregate**, seed 0, ⭐ **with the distinct
factor-tuple count beside every single value**, and the total byte ledger beside every single value.

| row | must be | rule |
|---|---|---|
| ⭐⭐ `knn_proto` | **WIN or TIE under R1** | R1: *win* = Δ > 0 with the paired 95 % CI excluding 0; *tie* = CI ∋ 0 **or** \|Δ\| ≤ m = 0.020; *loss* = Δ < −0.020 with CI excluding 0 |
| `clu` (the shipped 30-bit argmax read) | **beaten outright** | the harness's own leg-1 row |
| `store_removed` | **beaten outright** | the harness's own leg-1 row |
| `lace` (CELL instrument) | **beaten outright** | ⛔ 0.3241 is the BASELINES instrument at 688,128 B and is not comparable |
| `ebm_compose` | **beaten outright** | the harness's own leg-1 row |
| ⭐ **§C.1 `atoms_medoid`** at the identical configuration | **beaten** | charter §C.1, the sharp form |
| §C.1 `atoms_kpp` / `atoms_random` | reported whatever they say | the adversarial and easy forms |
| ⭐ **§C.2 `store_removed` at the same coefficients** | **must COLLAPSE below 0.5278** | charter §C.2, ⭐ **reported with its `K`** |
| the D-matched `random` null | **beaten** | the laundering control |
| ⛔ R4 `pca_affine` | **reported beside the arm whatever it says** | ⛔ a control; not adoptable as a store |
| ⭐ **§C.4** distance-to-nearest-atom / atom spread | reported | charter §C.4: a read sitting ON its nearest atom IS a lookup table and is reported as one |
| ⭐ **constant-output floor 0.5278** | cleared, with distinct tuples **≫ 2.67/72** | the task's falsifier 2 |

⛔ **A number quoted without its distinct-tuple count is void.**
⛔ **`x0`'s 0.6852 is never quoted without "at 8× bytes (4,194,304 B)".**

### The three falsifiers, in the task file's own words
1. Beats `store_removed`, `ebm_compose`, `lace`, `clu` **and** the matched-bytes atom-mean lookup
   outright, beyond seed spread — and **WIN or TIE against `knn_proto`** under R1.
2. Clears the constant-output floor: distinct factor tuples **well above 2.67/72**, reported beside the
   aggregate.
3. The store is load-bearing: `store_removed` collapses the arm.

---

## 4. THE PREDICTION — ⭐ committed 2026-09-17 15:45Z, BEFORE leg 1 or leg 2 was started
⛔ **No number on `eval_a`, `eval_b` or C existed when this section was written.** It is derived from the
42 CONFIRM corners (DEV, inside `split.train_rows`) and from iteration 1's measured DEV→B calibration.

**The calibration I am using, stated so the prediction can be judged.** Iteration 1 measured its DEV
instrument running LOW against the published B rows: `knn_proto` DEV 0.8302 vs published 0.8650
(−0.035); the shipped `clu` argmax read DEV 0.5490 vs published 0.5949 (−0.046). My own CONFIRM
`knn_proto` reads **0.8250**, so I expect leg 1's `knn_proto` on `eval_b` near **0.86**. ⚠ A DEV corner
sits inside the split the frozen encoder/decoder were fitted on, and `eval_b`'s `shape` column is free
(one class) while a DEV corner's is not — the two effects push in opposite directions, which is why the
offset is small and why my intervals are wide.

> ### ⭐ PREDICTION, pre-registered
> 1. **The arm scores `0.79 ± 0.07` on composition-B, with `42 – 62` distinct factor tuples of 72.**
> 2. ⭐ **Δ against `knn_proto` is NEGATIVE and inside the tie band: point estimate `−0.013`, 90 %
>    interval `[−0.07, +0.03]`.** ⇒ **I predict falsifier 1 resolves as a TIE under R1, not a WIN.**
>    ⛔ I am committing to "not a win" in writing, before measuring it.
> 3. **The arm beats the shipped `clu` argmax row by more than +0.15**, and beats `lace`, `ebm_compose`
>    and `store_removed` outright.
> 4. **The §C.2 `store_removed` control lands below 0.50** (CONFIRM: 0.4119) with **≤ 15 distinct tuples**,
>    i.e. below the 0.5278 constant floor.
> 5. **The arm beats its §C.1 `atoms_medoid` control by `+0.01 … +0.05`** (CONFIRM: +0.0261), and
>    ⚠ **ties its `atoms_kpp` control** (CONFIRM: +0.0033).
> 6. ⛔ **The R4 `pca_affine` control scores at or slightly above the arm** (CONFIRM: +0.0111).
> 7. ⭐⭐ **`clu_linear` beats `clu_argmax_shipped` at a byte-identical 598,528 B ledger by `+0.08 … +0.20`,
>    and takes the distinct tuples from ≈ 9 to ≥ 25.**
> 8. ⭐ **THE PARITY ROW.** Leg 1's `clu` composition-B aggregate on this Windows box lands **within the
>    published 3-seed sd of 0.5949 ± 0.0082**, and its distinct-tuple count **within ±1 of 9.00**.
>    ⛔ This is a labelled ledger row, not a spend.
> 9. Leg 1's `knn_proto` reproduces its published value within the harness's own
>    `rival_regression_tolerance` of 0.05.
> 10. **§C.4** distance-to-nearest-atom / atom-spread ratio **> 1.5** ⇒ not a lookup table.

⭐ **A prediction that survives is evidence; one that fails is a finding. Both are reported.**

---

## 5. WHAT WOULD MAKE THIS RUN WORTHLESS — named in advance
- ⛔ Moving `m`, the byte budget, `K`, `A`, the pair set, the ridges or the dictionary method **after**
  seeing a B number.
- ⛔ Quoting the arm against the **published Mac** rival rows instead of this run's own rival rows (R3).
- ⛔ Reporting the arm's aggregate without its distinct-tuple count, or without its **total** byte ledger
  (store + head) beside `knn_proto`'s 475,136 B.
- ⛔ Reporting a §C.1 or §C.2 control computed at a *different* configuration or a different atom count
  from the arm.
- ⛔ Presenting a mid-0.6 as progress: the constant-output floor on this metric is **0.5278**.
- ⛔ Promoting the R4 `pca_affine` control to an arm because it scores well.
- ⛔ Taking the `K = 128` configuration, which scores higher and is **disqualified by §C.2**.
