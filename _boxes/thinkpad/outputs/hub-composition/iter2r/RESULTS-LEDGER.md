# `hub/composition-2` (relaunch) — RESULTS LEDGER
Charter §A.6: *"a results ledger (every scored run)"*. Branch `hub/composition-2`, base
`945a3e9`. Machine **Windows AMD64 (FORGIS_14)** — R3 provenance on every row.

---

## 1. SCORED RUNS AGAINST THE HELD-OUT SETS A / B / C

| # | date (UTC) | PREREG | set | seed | configuration | result | artefact |
|---|---|---|---|---|---|---|---|
| **1 of 3** | 2026-09-17 13:31Z | `PREREG-hub-composition-2-1.md` (filed 13:30Z, **before** the run) | **B** (`eval_b`, 72 rows) | **0** | the reference object: K=32, M=26, LINEAR read `clu_wide_vector`, rank-16 pair head on `floor_hue×scale` + `object_hue×scale`, ridge 10, **469,504 B = 0.9881 × `knn_proto`** | ⭐ **`clu_linear` 42 / 72 distinct valid novel tuples vs `knn_proto` 12 / 72. Δ = +30.00 tuples, CI95 [+20.27, +39.73], McNemar exact p = 2.27 × 10⁻⁷ ⇒ WIN under R1.** Every §C.1 and §C.2 control **exactly 0**. Decoder ceiling **67**. | `iter2r/dev/SCORED_B_seed0.json` |
| 2 of 3 | — | — | — | — | ⛔ **NOT SPENT** | — | — |
| 3 of 3 | — | — | — | — | ⛔ **NOT SPENT** | — | — |

**Quote every number from run 1 as: "scored 1/3 — seed 0 only, 3-seed pending full caches."**

### ⛔ Why seed 0 only
Seeds 1 and 2 have `scope="used"` caches (9,288 rows); covering the cell would mean encoding
~32,200 new rows. ⛔ That is a re-encode and it is forbidden (LAUNCH-COMMON §4). Seed 0's
cache is `scope="full"` (480,000 rows) with a measured `run_seed` shortfall of **0 of 41,487
needed rows** (`iter2r/dev/coverage_seed0.json`), so ⛔ nothing could be extended and the
re-encode gate could not bind. `exp_gen_store.py` has a **zero-line diff from the pin**;
`ULP_BOUND` and `FRAC_BOUND` were never touched.

### ⚠ WHY THIS SPEND HAPPENED — written in the PREREG **before** the run (§7 there)
The addendum says the readout may cost **0** if my object is the same as
`hub/generation-table-2`'s and is taken from ITS scored cell. ⛔ I cannot read a concurrent
peer's outputs (LAUNCH-COMMON §3), and
`exp_gen_cell.ARMS = (clu, store_removed, knn_proto, lace, ebm_compose, dynread)` — ⛔ the
request-addressed linear factored read is **not among them**, and `gen_factored_read.py`
arrived on `base/iter2` as a library that nothing in `exp_gen_cell` calls. On the evidence
available to me the stock cell cannot carry this column. ⚠ **If the coordinator knows the
peer cell does carry this exact object, this spend was avoidable — flagged here, not after
the fact.**

### ⚠ ACCOUNTING QUESTION FOR THE COORDINATOR (⛔ not mine to settle)
Charter §A.5 requires an **independent replication of the winning configuration** before any
number is spoken, and the winning configuration is the B cell. The replication spoke
(`iter2r/replication/`) therefore re-runs **the same pre-registered configuration at the same
seed on the same cell with independent code**. ⭐ I account that as a **reproduction of run 1,
not a second spend**: it selects nothing, tests no new hypothesis, and changes no
configuration. ⚠ If the coordinator counts it as a spend, the correct total is **2 of 3**, and
I would rather it be corrected here than discovered later.

---

## 2. PRE-REGISTRATIONS FILED

| file | filed | status |
|---|---|---|
| `PREREG-hub-composition-2-1.md` | 2026-09-17T13:30Z | ⭐ **executed 13:31Z.** Predictions P1–P10 scored in it and in `LOG.md`; ⛔ **P4 (the R1 verdict) FAILED** — I predicted a TIE and measured a WIN. Reported as a finding. |
| `PREREG-object.json` | 2026-09-17T13:30Z | the object, machine-readable; `c06_score_B.py` reads it so the PREREG and the run cannot drift apart |

---

## 3. NON-SCORED WORK — all on DEV corners cut from inside the train split. ⛔ B never entered.
Every DEV corner is Montero-SHAPED, held out of its own store and address book, and **refused
outright** (⛔ never trimmed) if it intersects `eval_a`/`eval_b` — 8 refused in the selection
half, 7 in the validation half.

| # | unit | what | corners | artefact | touches B? |
|---|---|---|---|---|---|
| C01 | `c01_wiring.py` | wiring: the reference object reproduces (38 / 35); fast head == pinned head to 4.8e-07 with an identical headline; the identity asserted | 2 | `dev/c01_seed0.json` | no |
| C02 | `c02_rivals.py` + `rivals.py` | ⭐ the rival and every §C control on the SAME corners, on THIS box | **49** (24 sel + 25 val) | `dev/rivals_ledger.jsonl`, `dev/analysis_ref_*.json` | no |
| — | `coverage_check.py` | cache coverage — a set-membership count on ROW INDICES only | — | `dev/coverage_seed0.json` | ⛔ no (indices only; nothing read, rendered or scored) |
| C04 | `c04_head.py` | 45 head designs × ranks × ridges at matched bytes, store fixed | 8 | `dev/rivals_ledger.jsonl` (tag `head1`) | no |
| C05 | `c05_dream.py` | dreaming, STATE space, pre-registered T grid, β ∈ {0,1} | 3 | `dev/c05_dream_dream_seed0.json` | no |
| C05a | `c05a_readout_gap.py` | ⭐ where the object lives: coefficient geometry + a τ sweep over 3 decades | 3 | `dev/c05a_readout_gap_seed0.json` | no |
| C05b | `c05b_coef_dream.py` | dreaming, COEFFICIENT space + the corrected T = 0 state legs | 3 | `dev/c05b_coef_dream_seed0.json` | no |
| C07 | `c07_store_km.py` | 5 store geometries (K, M) at matched bytes | 8 | `dev/rivals_ledger.jsonl` (tag `km`) | no |
| — | `predict_B.py` | the calibrated pre-run prediction (read-only over the DEV ledger) | 49 | `dev/predict_B.json` | ⛔ no |

⭐ Every one of these is selection / mechanism work, which charter §A.1–2 place under
*"iterate freely on MECHANISM"* and *"selection on `val` only"*. ⛔ Not one is a scored run and
⛔ not one reads `eval_b`. **`c06_score_B.py` is the only file in this hub that opens B**, and
it refuses to start without a PREREG and refuses to overwrite an existing result.

---

## 4. §C FLOOR CHECKS — on the scored cell, and on 49 DEV corners

| check | on B (scored 1/3) | on 49 DEV corners | verdict |
|---|---|---|---|
| **§C.1** matched-bytes lookup (57 real pool rows; feature medoids **and** factor-space medoids) | **0** and **0** | **0.00** and **0.00**, all 49 | ⭐ **PASSES** |
| **§C.1** the rival `knn_proto` itself | 12 vs the object's 42 | 13.51 vs 14.47 | ⭐ B: beaten. DEV: tie |
| **§C.2** landscape-removed (atoms zeroed) | **0** | **0.00**, all 49 | ⭐ **PASSES** |
| **§C.2** landscape-removed (atoms shuffled, bytes identical) | **0** (while producing 65 distinct tuples) | **0.00**, all 49 | ⭐ **PASSES** |
| **§C.3** "a tie is the win narrative" clause | ⛔ **NOT CLAIMED** — dreaming is DEAD on this object (C05/C05a/C05b). The clause must be carried by the deletion or refusal hubs on the same object, or not at all. | | ⛔ **not claimed** |
| **§C.4** settled state on its nearest atom ⇒ a lookup | ⭐ the object's read is a **signed basis expansion**: 47 % negative coefficients, L1 ≈ 8.5 per well, **0.00 %** of wells a probability vector ⇒ its output lies far **outside** the convex hull of its own atoms | same on 3 corners measured | ⭐ ⛔ **not a lookup** |
| constant-output floor | **0** | **0.00**, all 49 | ⭐ **PASSES** |
| byte ledger | 469,504 = **0.9881 ×** `knn_proto` | same | ⭐ **matched and under** |
| `assert_request_only_factored` | ⭐ pass, n = 72, bit-identical under feature replacement | pass | ⭐ **PASSES** |
| `assert_no_bypass` | ⭐ pass, n = 64, max delta 0.0 | pass | ⭐ **PASSES** |

---

## 5. ADDITIONAL NON-SCORED UNITS — after the scored run, all on DEV corners. ⛔ B never re-read.
Every unit below exists because the **§A.5 replication refuted the headline level**. ⛔ None
touches `eval_b`; `c06_score_B.py` opened it once and its result file cannot be overwritten.

| # | unit | what | corners | artefact |
|---|---|---|---|---|
| — | `predict_B.py`, `ceiling_strata.py`, `corner_structure.py` | the pre-run prediction, and three tests of *why B read +30* | 49 | `predict_B.json`, `ceiling_strata.json`, `corner_structure.json` |
| **REP** | independent spoke (Opus), own code, barred from this hub's scripts | re-ran the pre-registered configuration on B | — | `iter2r/replication/REPLICATION.md` |
| C09 | `c09_construction.py genpart_25` | the **spoke's** equal-well construction | **49** | ledger tag `ctor_genpart_25`, `construction_gap.json` |
| C09 | `c09_construction.py mine_raw0` | **no Lloyd at all** — tests the spoke's "unconverged noise" criticism | **49** | ledger tag `ctor_mine_raw0`, `raw0_test.json` |
| C08 | `c08_store_seed.py 1`, `… 2` | **store-init seed spread** (bridge seed fixed at 0) | **49 × 2** | ledger tags `storeseed1`, `storeseed2`, `store_seed_spread.json` |

## 6. ⭐ THE REPLICATION (charter §A.5) — *"unreplicated = not a result"*
An independent spoke, barred from every file under `iter2r/dev/`, `iter2/dev/`, `instrument/`
and `mech/` and from all of this hub's result files, wrote its own runner against the pinned
library and re-ran the pre-registered configuration on `eval_b`, seed 0.

| quantity | hub | replication | verdict |
|---|---|---|---|
| `knn_proto` | 12 | **12** | ⭐ **CONFIRMED exact** |
| landscape-removed (zeroed) | 0 | **0** | ⭐ **CONFIRMED exact** |
| decoder ceiling | 67 | **67** | ⭐ **CONFIRMED exact** |
| object bytes | 469,504 | **469,504** | ⭐ **CONFIRMED exact** |
| `knn_proto` bytes | 475,136 | **475,136** | ⭐ **CONFIRMED exact** |
| identity `= 72 × exact_match` | asserted | **holds, all 4 arms** | ⭐ **CONFIRMED** |
| `assert_request_only_factored` | pass | **pass, 72/72** | ⭐ **CONFIRMED** |
| ⭐ **`clu_linear` — THE HEADLINE LEVEL** | **42** | ⛔ **25** | ⛔⛔ **REFUTED as a level** |
| ⭐ **the R1 VERDICT on B** | **WIN** (+30.00 [+20.27, +39.73]) | ⭐ **WIN** (+13.00 [+3.88, +22.12]) | ⭐ **CONFIRMED as a verdict** |

⭐ **The refutation was accepted, its cause was found, and it was then run down on 49 corners**
(Amendment 1 to the PREREG; `LOG.md` 13:53Z / 14:39Z / 14:59Z / 16:17Z):
- the two constructions **tie** over 49 corners (Δ = −0.53 [−1.94, +0.88]);
- removing k-means entirely also **ties** (Δ = +1.04 [−1.87, +3.95]);
- three store seeds give a within-corner sd of **4.09** tuples against a between-corner sd of
  **12.92** ⇒ ⭐ **the corner is 90.9 % of the variance in Δ**.
⇒ ⭐ the level is not reproducible and is **withdrawn as a number**; ⭐ the verdict is
reproducible and stands; ⭐ **the generalising estimate is Δ = +1.40, CI95 [−2.31, +5.11],
over 49 corners × 3 store seeds — a TIE.**

## 7. ⚠ THE DEFECT THIS HUB OWNS
**The PREREG under-specified the object** — it pinned ridge to one decimal place and left the
equal-well partition, the k-means++ RNG stream and the Lloyd iteration budget free.
⭐ **Fix, for every future PREREG in this programme:** ship the fitted store's `param_hash` in
`PREREG-object.json`, exactly as `resolve_decoder` and `load_external_classifier` already do.
⭐ **Both of those instruments replicated bit-exactly. That is not a coincidence.**

---

## 8. C12 — the shipped `train_store` column (added after the replication's second report)
| # | unit | what | corners | artefact |
|---|---|---|---|---|
| C12 | `c12_trainstore.py 1200` | ⭐ the shipped `init_store` + `train_store` (Adam) as the MODE FITTER, vs plain Lloyd. ⛔ Only the fitter differs; byte ledger asserted unchanged at 469,504 B | **2 of 6 attempted** | ledger tag `trained1200`, `dev/trainstore_vs_lloyd.json` |

| corner | Lloyd | shipped `train_store` | Δ | `knn_proto` |
|---|---|---|---|---|
| c0 | 38 | **33** | **−5** | 5 |
| c2 | 35 | **28** | **−7** | 24 |

⭐ **The trainer moves the WRONG way and beat Lloyd on neither corner** (mean −6.0). ⇒ the
replication spoke's hypothesis that this variant would "close the gap to 42" is contradicted
in direction, and the last alternative to the corner-variance account of 42-vs-25 is closed.
⛔ **n = 2**: this does NOT establish that Lloyd beats the shipped trainer.
⚠ **Stopped at 2 of 6 for a resource reason, stated plainly:** Adam training reached **50.3 GB
resident** and drove this shared four-hub box to **0.4 GB free of 63.4 GB**, with the third
corner thrashing for 52 min. Stopped; free RAM returned to 50.8 GB; no peer process touched.
⛔ **My defect:** launched without a memory or per-corner wall-clock cap despite the spoke's
report that the same variant had already blown a 2,400 s cap. ⭐ Resumability meant the kill
cost one corner, not the run.
⛔ **Verified by grep, not asserted:** every store build in this hub — including the scored run
— goes through `fast.kmeans_pq` (plain Lloyd); `train_store`/`init_store` appear nowhere in the
`ref` code path. ⇒ the spoke's "PREREG says Lloyd but ran `train_store`" hypothesis is **false**.
