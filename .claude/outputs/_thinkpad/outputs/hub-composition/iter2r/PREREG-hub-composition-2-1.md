# PREREG — `hub/composition-2`, scored run 1 of 3
**Filed 2026-09-17T13:30Z, BEFORE the run.** ⛔ Nothing below is edited after a result is
seen; amendments are appended, dated, never edited (charter §A.3).
Branch `hub/composition-2` · base `945a3e9` (`base/iter2`) · machine **Windows AMD64
(FORGIS_14)** · seed **0 only** · scored-run cost **1 of 3**.

---

## 1. THE OBJECT — fixed, and I state why it is this one
**The reference object on `base/iter2`, unchanged:**

| | |
|---|---|
| store | **K = 32** correlation-partitioned wells (shipped ward grouping), **M = 26** k-means modes/well |
| read | request-addressed **LINEAR** read `chlu.core.gen_compose.clu_wide_vector` |
| head | `chlu.core.gen_factored_read.fit_factored_head`, **rank 16**, ridge **10.0**, pairs `floor_hue×scale` + `object_hue×scale` |
| bytes | store 212,992 + head 256,512 = **469,504 B = 0.9881 × `knn_proto`'s 475,136 B** |
| legality | `assert_request_only_factored` and `assert_no_bypass` pass; addressed not indexed; atoms in wells; request-only; matched bytes |

⭐ **Why no departure.** Two matched-bytes searches, ~400 scored DEV renders:
**C04** — 45 head designs × ranks × ridges, store fixed → best beats the reference by **+0.375
tuples** (22.88 vs 22.50), inside a corner sd of ~14.
**C07** — 5 store geometries (K ∈ {8,16,32,64,128}, each at its largest affordable M) → the
reference **K=32/M=26 is the peak** (8.25 / 11.12 / **22.50** / 21.62 / 20.12).
⇒ the reference object is at the optimum of its own family. ⛔ Departing would fragment
charter §C.3's "same object" clause for nothing.

## 2. THE BAR — R2's, as the Head set it. ⛔ I did not write it and I do not move it.
> **Distinct valid novel tuples on B at matched bytes, WIN-or-TIE vs `knn_proto` under R1**,
> with the capacity law `rel-err = e^1.346 · bytes^−0.1651` (R² 0.9960) and the decoder
> ceiling **67.67 ± 2.49** reported beside it.

**R1** (Head, 2026-09-17): *tie* := the paired 95 % CI on Δ(arm − rival) includes 0, **or**
|Δ| ≤ m = 0.020; *win* := Δ > 0 with the CI excluding 0; *loss* := Δ < −m with the CI
excluding 0. ⛔ Set before this run, and before the DEV numbers in §4 existed in their final
form.

**Definitions, ⛔ not negotiable and ⛔ not re-defined here:**
- **VALID(i)** := the external classifier reads render *i* as the tuple that was REQUESTED,
  on **all six** factors (identical to `gen_probe`'s per-item `exact_match`).
- **NOVEL(t)** := *t* is not the factor tuple of **any** row the store was fitted on.
- **HEADLINE** := |{ distinct *t* : produced, VALID **and** NOVEL }|.

## 3. THE STATISTIC — declared before the run
B is **one cell of 72 items**, so the pairing unit is the **ITEM**:
```
d_i = valid_arm(i) − valid_rival(i) ∈ {−1, 0, +1},   i = 1..72
Δ   = mean(d) = exact_match_arm − exact_match_rival        (aggregate units)
CI95 = paired t on d;  McNemar's exact test reported beside it
```
By the inherited identity `distinct_valid_novel = 72 · exact_match` on B — all 72 targets
distinct, none in any train row, **asserted per render** by `comp.headline`, which raises if
it ever fails — the same Δ in **tuples** is 72 · Δ. R1 is applied in aggregate units with
m = 0.020 and printed in both.
⚠ **Stated in advance:** the paired per-corner difference has **sd ≈ 13.4 tuples** across 49
DEV corners. B is ONE corner. ⇒ the single-cell B read carries a predictive sd of ≈ ±13
tuples and ⛔ cannot by itself separate a real effect from corner identity. The 49-corner DEV
estimate is the precise one; B is the pre-registered held-out readout. Both are reported.

## 4. THE EVIDENCE THAT EARNED THE RUN — DEV, 49 corners, ⛔ B never touched
`analysis_ref_all_tuples_valid_novel.json`. Montero-SHAPED corners cut from inside the train
split, each held out of its own store and address book; 8 + 7 corners **refused** by the
held-out guard for touching `eval_a`, ⛔ never trimmed.

| arm | bytes | mean /72 | Δ vs `knn_proto` | CI95 | R1 |
|---|---|---|---|---|---|
| ⛔ `ceiling` (NOT an arm) | — | **54.96 ± 16.07** | +41.45 | [+36.06, +46.84] | — |
| `knn_proto` | 475,136 | 13.51 ± 14.19 | — | — | — |
| ⭐ **`clu_linear`** | **469,504** | **14.47 ± 13.30** | **+0.96** | **[−2.88, +4.80]** | ⭐ **TIE** |
| `head_additive_only` | 406,016 | 6.96 | −6.55 | [−9.23, −3.87] | loss |
| §C.2 `store_removed_zero` | 469,504 | **0.00** | −13.51 | [−17.59, −9.43] | loss |
| §C.2 `store_removed_shuffle` | 469,504 | **0.00** | −13.51 | [−17.59, −9.43] | loss |
| §C.1 `lookup_feat` | 466,944 | **0.00** | −13.51 | [−17.59, −9.43] | loss |
| §C.1 `lookup_tuple` | 466,944 | **0.00** | −13.51 | [−17.59, −9.43] | loss |
| constant-output floor | 8,192 | **0.00** | −13.51 | [−17.59, −9.43] | loss |

**The gate the addendum sets is met:** the bar is *win-or-tie*; DEV reads **tie**; every §C
control passes on all 49 corners.

## 5. ⭐ THE PREDICTIONS — committed to NUMBERS, before the run (protocol §5)
Calibrated by regressing each arm's DEV count on that corner's **measured decoder ceiling**
across all 49 corners, evaluated at B's inherited ceiling of **67.67**
(`iter2r/dev/predict_B.json`). ⛔ Written down before `c06_score_B.py` was executed.

| # | quantity | ⭐ **predicted** | 95 % prediction interval |
|---|---|---|---|
| P1 | `clu_linear` tuples_valid_novel on B | **19** | [0, 44] (regression PI [−5.3, 43.8], truncated at 0) |
| P2 | `knn_proto` tuples_valid_novel on B | **16** | [0, 45] |
| P3 | **Δ (`clu_linear` − `knn_proto`) in tuples** | **+3** | [−24, +30] |
| P4 | **the R1 verdict on B** | ⭐ **TIE** | — |
| P5 | `head_additive_only` on B | **7** | a LOSS to `knn_proto` |
| P6 | §C.2 `store_removed_zero` / `store_removed_shuffle` | ⭐ **0 and 0** | exact |
| P7 | §C.1 `lookup_feat` / `lookup_tuple` | ⭐ **0 and 0** | exact |
| P8 | constant-output floor | ⭐ **0** | exact |
| P9 | my measured decoder ceiling on B | **67.67** | within ±5 of the inherited 67.67 ± 2.49 |
| P10 | the asserted identity `distinct_valid_novel = 72 × exact_match` on B | **holds** | `comp.headline` raises if not |

⭐ **I predict a TIE, and I say so before looking.** A prediction that survives is evidence;
one that fails is a finding. ⛔ P6–P8 are the ones that would destroy the result if wrong: a
non-zero lookup or a non-zero ablated store means the column is not measuring what I claim.

## 6. WHAT THE RUN DOES, AND WHAT IT MAY NOT DO
`iter2r/dev/c06_score_B.py`, which is the **only** file in this hub that reads `eval_b`. It
refuses to start without `SCORED_RUN_PREREG` pointing at this file, and refuses to overwrite
an existing result — a scored run is spent once.
- ⭐ **R3 same-device by construction:** every arm, rivals included, is built and scored HERE
  from the Mac-written feature cache. ⛔ No Mac-published row enters any paired comparison;
  Mac rows are cited beside, as priors with provenance.
- ⭐ **The gate:** measured `run_seed` shortfall for seed 0 is **0 of 41,487 needed rows**
  (`coverage_seed0.json`; cache 480,000 rows, `scope="full"`). ⇒ nothing can be extended and
  the re-encode gate cannot bind. ⛔ `ULP_BOUND` / `FRAC_BOUND` untouched; `exp_gen_store.py`
  has a zero-line diff from the pin.
- ⛔ **Seeds 1 and 2 are NOT run**: their caches are `scope="used"` and covering the cell
  would mean encoding ~32,200 new rows, which is forbidden. This is reported as
  **"scored 1/3 — seed 0 only, 3-seed pending full caches"**.
- ⛔ The frozen scoring harness is **not edited**. `exp_gen_cell.ARMS` does not contain this
  object, so the stock cell cannot produce this column; the run uses the same frozen
  instruments through my own runner instead (§7).

## 7. WHY THIS COSTS 1 OF 3 — the ledger entry, written BEFORE the run
The addendum says the readout may cost **0** if my object is the same as
`hub/generation-table-2`'s and the readout is taken from ITS scored cell.
⛔ I cannot read a concurrent peer's outputs (LAUNCH-COMMON §3), and
`exp_gen_cell.ARMS = (clu, store_removed, knn_proto, lace, ebm_compose, dynread)` — ⛔ the
request-addressed **linear factored read is not among them**, and `gen_factored_read.py`
arrived on `base/iter2` as a library that nothing in `exp_gen_cell` calls. ⇒ on the evidence
available to me the stock cell cannot carry this column, so I spend **1 of 3**.
⚠ **If the coordinator knows the peer cell does carry this exact object, this spend was
avoidable and I say so here rather than after the fact.**

## 8. FALSIFIERS FOR THIS RUN
1. ⛔ Any §C.1 lookup or §C.2 ablated store scoring **> 0** on B ⇒ the column is not measuring
   valid-novel composition and the result is void.
2. ⛔ The identity `distinct_valid_novel = 72 × exact_match` failing on B ⇒ B is not the clean
   cell the bar assumes; **report it, do not paper over it**.
3. ⛔ `assert_request_only_factored` or `assert_no_bypass` failing ⇒ the arm is not
   charter-legal and no number from it may be spoken.
4. ⛔ A cache shortfall > 0 at run time ⇒ **STOP and report the shortfall.** ⛔ Never encode.

---

# ⚠ AMENDMENT 1 — 2026-09-17T16:20Z. **APPENDED, ⛔ nothing above is edited** (charter §A.3).
**Filed AFTER the run and AFTER the §A.5 replication.** ⛔ It changes no bar, no definition
and no reported number. It records a defect in §1 of this file and the fix.

## A1.1 THE DEFECT — §1's object specification is UNDER-DETERMINED
The §A.5 replication spoke could not reproduce the object's headline (**25** where this hub
measured **42**), while reproducing `knn_proto` (12), the landscape-removed control (0), the
decoder ceiling (67), both byte ledgers and the `72 × exact_match` identity **exactly**.
Its diagnosis is correct and I accept it:

> *"A PREREG that pins ridge to one decimal place but leaves the store's fit unpinned has
> pinned the wrong things."*

Three things §1 failed to name:
1. ⛔ **The equal-well partition.** §1 says *"K = 32 correlation-partitioned wells"*, which
   **cannot be executed literally**: `gen_factored_read.mode_basis` **raises** on the shipped
   `gen_store.correlation_grouping` at K = 32 (well sizes 36 … 156 — ragged). An equal-well
   substitution is required and §1 does not say which. This hub used
   **`fast.balanced_labels`** — ward leaf order with the **61 dead channels SPREAD at evenly
   spaced slots** ("so no single well is spent on constants"), inherited verbatim from
   `hub/generation-table`. The spoke used the pinned library's
   **`gen_partition.balanced_correlation_partition`**, which puts **every dead channel into
   the smallest well**. Same |corr|, same ward linkage, same leaf order — **different stores**.
2. ⛔ **The k-means++ RNG stream** (one stream consumed well-by-well vs per-well independent).
3. ⛔ **The Lloyd iteration budget.** This hub used **25** (`fast.kmeans_pq`'s default).

## A1.2 THE FIX, for every future PREREG in this programme
⭐ **Ship the fitted store's `param_hash` in `PREREG-object.json`**, exactly as
`resolve_decoder` and `load_external_classifier` already do for the decoder and the external
classifier. ⭐ **Both of those instruments replicated bit-exactly. That is not a coincidence.**
A configuration is not pinned until the artefact it produces is hashed.

## A1.3 ⭐ WHAT THE DEFECT DID AND DID NOT COST — measured on 49 corners, ⛔ not argued
`c09_construction.py`, `c08_store_seed.py`, `compare_ctor.py`, `compare_raw0.py`,
`seed_spread.py`. Every variant below is the same object at the same 469,504 B, scored by
the same instrument against the same rival, on the same 49 DEV corners.

| store-fit variant | DEV mean /72 | Δ vs `knn_proto` on DEV | verdict | on the single B cell |
|---|---|---|---|---|
| **this hub** — `balanced_labels`, Lloyd 25 | 14.47 | +0.96 [−2.88, +4.80] | **TIE** | **42** |
| **the spoke** — `gen_partition`, Lloyd 25 | 15.00 | +1.49 [−2.29, +5.27] | **TIE** | **25** |
| `balanced_labels`, **no Lloyd at all** | 15.51 | +2.00 [−1.73, +5.73] | **TIE** | (spoke: 36) |
| `knn_proto` | 13.51 | — | — | **12** |

- ⭐ **The construction gap is not systematic:** mine − the spoke's = **−0.53, CI
  [−1.94, +0.88], 19/10/20 — a TIE.** ⛔ My build is not a favourable choice; it is very
  slightly *worse* on average.
- ⭐ **Lloyd convergence does not carry the headline:** no-Lloyd − Lloyd-25 = **+1.04, CI
  [−1.87, +3.95], 18/11/20 — a TIE.** ⛔ The spoke's *"R1 is substantially measuring
  unconverged codebook noise"* is **not supported** over 49 corners.
- ⭐⭐ **Where the 17-tuple disagreement actually lives:** three store seeds × 49 corners give
  a between-corner sd of **12.92 tuples** against a store-seed sd of **4.09** ⇒ **the corner
  is 90.9 % of the variance in Δ.** Two builds that agree to ±1 tuple over 49 cells land 17
  apart on one cell.

## A1.4 ⭐ WHAT STANDS, AND WHAT I WITHDRAW
- ⭐ **STANDS — the verdict.** R2 asks for *win-or-tie vs `knn_proto` on B under R1*. Both
  independent builds give a **WIN**: this hub Δ = **+30.00** [+20.27, +39.73], the spoke
  Δ = **+13.00** [+3.88, +22.12]. ⭐ The bar is met under either reading of the unpinned spec.
- ⭐ **STANDS — every control.** §C.1 and §C.2 are **0** on B and **0.00 on all 49 corners**,
  in both builds, and the spoke independently confirmed the zeroed read is provably the
  all-zeros feature.
- ⛔ **WITHDRAWN — the level "42" as a reproducible number.** The honest statement is
  **"25–42 on B, depending on a store-fit choice this pre-registration failed to pin."**
- ⭐ **ADDED — the generalising estimate**, which this file should have led with:
  **Δ = +1.40 tuples, CI95 [−2.31, +5.11], over 49 corners × 3 store seeds ⇒ a TIE.**

## A1.5 ⛔ WHAT WAS **NOT** DONE
⛔ No bar was moved. ⛔ No definition of VALID or NOVEL was touched. ⛔ No threshold was
adjusted. ⛔ No scored run was repeated to get a better number — `eval_b` was read **once**,
by `c06_score_B.py`, and every measurement in A1.3 is on DEV corners that never touch it.
⛔ Nothing in §1–§8 above was edited; this amendment is appended, as §A.3 requires.
