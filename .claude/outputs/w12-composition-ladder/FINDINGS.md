# w12 hub/composition-ladder — FINDINGS

**Report to the Hub.** Branch `w12/composition-ladder`, base `base/iter8 = aa91785`.
Box FORGIS_14, local CPU, 3 lanes × 1 thread (`nproc = 3`, shared with three other agents).
`chlu.__file__ = C:\chlu\worktrees\CHLU-w12-composition-ladder\chlu\__init__.py`, asserted at
the top of every leg and written into every `LEG.json`.

> ⛔ **PLACEMENT (Head ruling 2026-09-21T12:38:07Z, `HEAD-RULING-K.md`): APPENDIX — the
> K-sweep's ENDPOINT at K = 5. NOT main text. The main text carries ONE K, K\*, and never
> both.** The label is carried by the artifacts themselves: `LADDER.json:placement`, the first
> line of `LADDER.md`, and the figure's own suptitle and caption.

Every number below names the file it came from. ⛔ Anything not run is a DECLARED NOT-RUN with
a reason — never a zero, never a blank.

---

## 0. THE ANSWER

**Scope: 18 legs (5 rungs + `shipped`) × 3 seeds × 13 arms = 234 scored arm-runs, `failed=[]`
on every leg, zero `FAILED_*.json` anywhere.**

### The ladder (store arm; `pca_tensor` is identical at every rung — see §6)

| rung | held-out rows | store | `learned_1x` | normalised | clears the bar? | deteriorates vs its own k=2? |
|---|---|---|---|---|---|---|
| k=2 | 192,000 | 30.5389 | **33.3222** | 0.8483 | **no** | (base) |
| k=3 | **300,000** | 59.7500 | 55.5083 | 0.8299 | **no** (2 of 3 seeds) | no consistent direction |
| k=4 | 235,200 | 62.7083 | **67.9917** | 0.8709 | **no** | no consistent direction |
| k=5 | 213,600 | 64.7194 | **68.6194** | 0.8989 | **no** | **False — it goes UP** |
| k=6 | 200,640 | 64.5556 | **68.6167** | 0.8966 | **no** | **False — it goes UP** |

### Four findings

1. ⛔ **There is NO BREAKING POINT, because the store NEVER CLEARED THE BAR.** Against a learner
   byte-matched to the store's own measured total it is behind at k=2, 4, 5 and 6 on **every**
   seed, and clears only 2 of 3 seeds at k=3. Under the pre-declared rule (`AMENDMENT-2` §3:
   every seed) it clears at **no rung**. `same_k_for_every_arm`: *"no arm has a breaking point."*
   ⛔ There was never a row to lose, so the question "where does it break" has no answer here —
   and that, not a curve, is what this hub reports.
2. ⭐ **Nothing deteriorates as k grows — the opposite.** Ceiling-normalised the store runs
   0.8483 → 0.8299 → 0.8709 → 0.8989 → 0.8966, higher at the far end than at k=2 on every seed
   at k=5 and k=6. ⛔ **The ONE QUESTION's premise is false on Shapes3D for these arms**, and
   under the task's own third falsifier this is reported as "the ladder does not discriminate
   by depth", not as a flat line dressed up as a result.
3. ⭐⭐ **`pca_tensor` is BIT-IDENTICAL to the single design at EVERY rung and EVERY seed, at
   ~1/18th of the bytes** (≈1.37× the 524,288 B pin against **23.89×–24.88×**). The single
   design's byte cost buys **exact deletion, not composition.** Task falsifier #2 fires, in its
   strongest available form.
4. ⭐⭐ **For k ≥ 3 this ladder is a TRAINING-DATA axis, not a composition-depth axis** (§5b).
   Those four rungs probe the *identical* 120 corners; every arm fitted on the rung's pool
   tracks the held-out row count at **r = −0.96 … −0.98**, and every arm not fitted on it is
   **exactly constant**.

### And the one that decides how to write it up

⭐ **The store's apparent advantage is an artefact of the baseline's byte pin.** Against spoke
A's 524,288 B rivals it wins by **+7.08…+9.65 at k=2, rising to +14.99…+22.43 at k=6**. Against
a learner matched to its own 24.4× total it **loses** at four rungs of five. ⛔ The byte pin does
not shift the level — it flips the sign, and therefore flips how the k-trend reads.

---

## 1. Gates, seeds and K — all DECLARED

| gate | state at 11:31:33Z (`START_UTC.txt`) | consequence |
|---|---|---|
| `w12-kway-ladder-harness/LADDER-READY.json` | **EXISTS** (22,591 B) | gate OPEN; the ladder is the harness's, ⛔ this hub built no family |
| `w12-cache-gen2-seeds-3-4/CACHE-READY.json` | **ABSENT** (produced 12:27:01Z, 56 min later) | 3-seed branch |
| `w12-single-design-crossover/CROSSOVER-val.json` | `"K_star": null` | **K = 5, DECLARED**, carried on every row |

**Seeds — the count used is 3 (seeds 0, 1, 2), DECLARED** (`AMENDMENT-5.md`). Re-evaluated per
seed when `CACHE-READY.json` appeared:

- **seed 3 PASSES the cache gate** — `array_shape [480000,128,4,4]`, `n_blocks 59`,
  `len(block_sha256) 59`, `scope full`, `generation 2`, `shape_dtype_matches_seeds_0_2: true`.
- **seed 4 FAILS** — `"status": "NOT_RUN"`, no `array_shape`, 0 blocks.
- ⛔ **But seed 3 cannot be scored and the cache is not the reason.** `decoder_v1_s3.pkl` does
  not exist (`winner_hashes.json` keys are exactly `['seed_0','seed_1','seed_2']`), nor does
  `gen-external-classifier/seed_3`, nor `spoke-a-rivals/rivals/seed_3`. Measured:
  `Judge(3) REFUSES: FileNotFoundError ... decoder_v1_s3.pkl`. `CACHE-READY.json` says the same
  of itself — seed 3 is `status: CACHE_READY_BRIDGE_PENDING`, `bridge.NOT_RUN`.
- ⇒ **seed 3 and seed 4 are both DECLARED NOT-RUN**, for different reasons, both stated.
⛔ No 3-seed row is mixed with any other count; `n` is printed on every table and the figure.

**Feature cache.** `CHLU_GEN_FEATURES_ROOT` is **not** in this box's ambient environment, so an
unguarded run reads cache GENERATION 1, which w5-common §3.8 forbids. Every leg pins it to
spoke A's generation-2 tree and calls `exp_gen_delete_row7.assert_generation2(seed)` before any
number; each `LEG.json` carries `generation 2 / scope full / n_rows 480000`.

## 2. Pre-registration — five documents, every one timestamped before the numbers it governs

| file | written | verified empty at that instant | what it fixes |
|---|---|---|---|
| `PREREG.md` | 11:38:41Z | `runs/*/rows_*.json` matched nothing | 3 seeds · K=5 · arms · the two charter-C.2 controls · **the best-learned-baseline rule** |
| `AMENDMENT-1.md` | 11:42:09Z | same | the 1× byte-matched column moves from NOT-RUN **into** the run |
| `AMENDMENT-2.md` | 11:46:30Z | same | **"deteriorates"** and **"breaking point"**, and the ceiling problem |
| `AMENDMENT-3.md` | 11:49:46Z | only one learned arm on disk, no arm of this hub's | the opponent set widens — it can only get **stronger** |
| `AMENDMENT-4.md` | 11:54:52Z stamp | — | an amendment CONSIDERED and **NOT MADE**, explained rather than left dangling |
| `AMENDMENT-5.md` | 12:28:44Z | — | the seed gate, re-evaluated per seed |
| `AMENDMENT-3-CORRECTION.md` | 13:05:53Z | — | ⛔ **a claim of this hub's own, withdrawn** — see §7 |

## 3. ⛔ THE NON-NEGOTIABLE — both ends reproduce

Criterion fixed in `PREREG.md` §6 **before any number**: the end reproduces if the rung mean and
the shipped mean agree within that arm's own across-corner sd, per seed. A sixth condition,
`shipped`, was run with the identical arms and seeds on the value-hold-out condition as it
exists on disk.

**k = 2 vs the SHIPPED row 3** (60 corners vs 60) — **reproduces on every arm and every seed**

| arm | max abs diff | per seed (rung / shipped) |
|---|---|---|
| `single_design_K5` = `pca_tensor` | **1.0833** | 28.6333/29.7167 · 31.6500/31.6167 · 31.3333/31.4500 |
| all five rivals | **0.0000** | identical on every seed |

**k = 6, first 49 corners, vs the SHIPPED row 4** (49 vs 49) — **reproduces on every arm and
every seed**

| arm | max abs diff | per seed (rung / shipped) |
|---|---|---|
| `single_design_K5` = `pca_tensor` | **0.5306** | 63.1837/63.6122 · 63.7959/64.3265 · 67.9184/67.6531 |
| all five rivals | **0.0000** | identical on every seed |

⭐ The **structure** of that result is the evidence, not just the verdict: a rival's features are
a function of the REQUEST alone and not of the rung's fit pool, so only the novelty mask could
have moved them — and it did not. The store arms move by 0.03–1.08, which is exactly the
quantity the `shipped` leg exists to measure, since a different split draws a different
131,072-row fit pool and therefore a different store. ⇒ **the split change does not carry the
ladder's signal. The ladder is ON-INSTRUMENT and may be plotted.**

## 4. ⛔ THE CONTROLS — and the store is NOT decorative

Run on both store arms at every rung, on every seed, as wave 9 requires.

- `landscape_shuffled` — atoms permuted, the arm's own fitted address book kept.
- `random_atoms` — random atoms at the same Frobenius norm, address book **REFIT** against
  them (the strong form; `exp_w5_compose_field`'s `pca_random_refit`, `w9_maze`'s `random_atoms`).

**Both collapse at every rung on every seed.** At k=2: `landscape_shuffled` paired delta
+28.633 / +31.650 / +31.333 and `random_atoms` +28.383 / +31.567 / +30.583, every CI95 lower
bound above +26.4, every sign test p = 1.73e-18. At k=6 `landscape_shuffled` is 0.0000 on all
three seeds and `random_atoms` 0.2583 / 0.1583 / 0.4917 against a store at 62.7–67.0.

⇒ ⭐ **no rung is DECORATIVE, and a row is claimable at every rung.** Verified exhaustively:
`collapse_both_every_seed = True` for **both store arms at all five rungs and on the `shipped`
condition — 12 of 12 (store arm × condition) cells**. Contrast wave 9, whose maze arm read
0.9396/0.9222 against 1.0000 and claimed nothing.

⛔ **This matters for reading finding #1.** The store is not beaten because it is decorative —
its content is doing real work at every rung. It is beaten because a learner of the same true
size does the same job better.

⚠ **rank/width is a warning, not a verdict — measured.** At k=2 seed 0 `pca_tensor` runs
rank/width = [0.082, 0.068, 0.296, **1.000**, **1.000**] — two wells fully saturated, above
wave 9's decorative threshold of 0.938 — and its controls still collapse to 0.0000. ⛔ Only the
control decides. rank/width is printed on every store row of `LADDER.md` §2 regardless.

## 5. ⛔⛔ AN INSTRUMENT FINDING THE HUB MUST CARRY — for k ≥ 3 the rungs share ONE corner set

`instrument/corner_set_equality.txt`, built from the harness's own `build_kway_split`:

| rung | cells | corners | rows/corner | held-out rows |
|---|---|---|---|---|
| k=2 | 5 | 60 | 36 | 192,000 |
| k=3 | 17 | 120 | 72 | **300,000** |
| k=4 | 29 | 120 | 72 | 235,200 |
| k=5 | 65 | 120 | 72 | 213,600 |
| k=6 | 125 | 120 | 72 | 200,640 |

⭐ **k = 3, 4, 5 and 6 probe the IDENTICAL 120 boxes** (the same 8,640 rows); only the order and
the corner labels differ. k=2's 60 boxes are disjoint from all of them.

This is the harness's own **cell ≠ corner** design, not a defect: a k=3 cell pins
(object_hue, shape, scale) and is TILED over floorwall and orientation, and each tile pins all
six coordinates — which is what a k=6 cell is. What had not been drawn is the consequence.

⛔ **Three consequences:**
1. For k ≥ 3 the x-axis is **not** "how many factors the probe composes" — every probe pins all
   six. It is **how coarsely the hold-out is carved**, from 12 large `kway3-*` cells to 120
   singleton `kway6-*` cells.
2. ⇒ the five rival columns are **CONSTANT by construction** across k = 3…6. ⭐ Measured, and a
   free positive control that came out right: `rival_mlp` 35.2083 and `rival_lstm` 44.4333 at
   k=3 seed 0 are equal to four decimals to their k=6 seed 0 values; the per-corner arrays
   differ only by permutation (max abs diff 56).
3. ⇒ across k = 3…6 **only the store arms can move**, and only through their fit pool.

## 5b. ⭐⭐ WHAT THE LADDER ACTUALLY MEASURES FOR k ≥ 3 — the declared confound IS the signal

Because k = 3…6 probe the identical corners, the only thing that varies across those four rungs
is how many rows the cell removed from the fit pool. Correlating each arm's across-seed mean
against the rung's held-out row count:

| arm | k=3 | k=4 | k=5 | k=6 | Pearson r vs held-out rows |
|---|---|---|---|---|---|
| **held-out rows** | 300,000 | 235,200 | 213,600 | 200,640 | — |
| `single_design_K5` | 59.750 | 62.708 | 64.719 | 64.556 | **−0.9841** |
| `pca_tensor` | 59.750 | 62.708 | 64.719 | 64.556 | **−0.9841** |
| `learned_1x` | 55.508 | 67.992 | 68.619 | 68.617 | **−0.9591** |
| `learned_ref_ownpool` | 11.794 | 13.281 | 14.961 | 15.606 | **−0.9605** |
| `rival_mlp` | 42.175 | 42.175 | 42.175 | 42.175 | **exactly constant** |
| `rival_lstm` | 43.503 | 43.503 | 43.503 | 43.503 | **exactly constant** |
| `rival_table` | 9.528 | 9.528 | 9.528 | 9.528 | **exactly constant** |

⭐⭐ **Every arm fitted on the rung's own pool — store AND learner alike — tracks the held-out row
count at r = −0.96 … −0.98. Every arm not fitted on it is exactly constant** (their r is
undefined for zero variance, which is the finding rather than a gap).

⇒ ⛔ **For k ≥ 3 the x-axis is training data, not composition depth.** The harness DECLARED this
confound in advance and proved it cannot be removed by a row budget
(`row_matching_feasibility.feasible = false`; required floor 45,000 rows against an available
ceiling of 8,640). What this hub adds is the measurement that the confound accounts for
essentially all the variation — r² = 0.97 on the store — which follows once the corners are held
identical, because there is nothing else left for it to be.

⭐ The **k=2 → k≥3 step is a genuine change of probe** (60 corners of 36 rows → 120 of 72,
disjoint sets), and it is the only step on this ladder that is.

## 6. ⭐⭐ THE BYTE LEDGERS — measured on the objects this hub fitted, never netted

`BYTES.json`, from `downdate_store.byte_ledger` + `covariance_ledger` + the fitted address book.

Over all **18** fitted legs:

| arm | atoms | sufficient statistics | address book | TOTAL | × 524,288 B |
|---|---|---|---|---|---|
| `single_design_K5` | 524,288 | 11.9–12.3 M | 158–175 k | **12,523,008 – 13,041,864 B** | **23.886× – 24.875×** |
| `pca_tensor` | 524,288 | **0** | 188–195 k | **712,072 – 719,352 B** | **1.358× – 1.372×** |

⇒ ⭐⭐ **the single design costs ≈18× what `pca_tensor` costs and returns bit-identical
composition scores at every rung.** `pca_tensor` keeps no sufficient statistics — that zero is
the price it does not pay, and equally the reason it has no exact deletion.

⭐ This confirms the w10 referee's **MF3** ("the sufficient statistic that buys the exactness is
**24.5×** the byte budget and appears in no table") on this hub's own fitted object — and on the
real factor-MI partition, whose wells are UNEQUAL (≈777/936/216/60/59), not the equal-width
409–410 arithmetic in `w12-single-design-crossover/LEDGER-ACROSS-K.json`.

⛔ The 1× target is matched against the **honest full total** (float64 covariance), not the
ledger's own cheaper `if_symmetric_triangle_float32` variant (which would give ≈7.2×). The
honest number makes the opponent **bigger and stronger** — the direction that runs against this
hub's interest.

## 7. ⛔ CORRECTIONS THIS HUB ISSUED AGAINST ITSELF

1. **`AMENDMENT-3`'s stated motivation is WITHDRAWN** (`AMENDMENT-3-CORRECTION.md`). It claimed
   own-pool fitting beats every architecture in spoke A's set, citing `learned_ref_ownpool`
   8.2167 at k=3 against a published 0.006. ⛔ That was a **cross-instrument** comparison — the
   0.006 is the shortened `w8_v1_short` decoder, and that table says so itself. Same-instrument
   at k=2, own-pool is ~13 points **weaker** (7.48 vs the rivals' 17.59–20.83). ⭐ The **rule**
   stands unchanged and was right for a reason not yet measured at the time: a maximum over a
   superset can only raise the opponent — and it matters through `learned_1x`.
2. **A defect in this hub's own aggregator**, found and fixed: the bar test compared an arm with
   itself when that arm WAS the best learned baseline and scored the tie as a failure, so
   `learned_1x` read "NEVER CLEARED THE BAR". A self-comparison is now `NOT APPLICABLE`. ⛔ No
   store-arm number was affected.
3. **A wrong first diagnosis of the wall-clock problem**, corrected in `LOG.md` before it
   changed anything: the per-call `eqx.filter_jit` rebuild in `chlu/eval/gen_probe.py` is real
   and worth 2.8×, but it does **not** explain the 918.4 s arm timings — `arm_features` costs
   < 0.05 s per 720 rows and `read_back` ≈194 s per arm. The timings were **CPU contention**.
   ⛔ Nothing was cut on the strength of the wrong story; `AMENDMENT-4` records the corner-budget
   cut that was therefore never made.

## 8. Two verified no-ops adopted in the scoring path

Neither changes a pinned value; both are gated by a positive control that runs in every leg.

1. **`FastJudge`** — `Judge.read_back` with the jit hoisted out of the per-call body. Same
   frozen decoder, same hash-asserted classifier, same `model_dtype`, same `jax.vmap`, same
   `eval_batch = 256`, same argmax. `assert_identical` on real fit-pool features:
   **True, 1,536 elements**, in every `LEG.json` as `fastjudge_positive_control`. Measured
   16.1 s → 5.6 s over 720 rows.
2. **Corner batching** — all corners answered in one `arm_features` call and one `read_back`.
   Measured `IDENTICAL: True, max|diff| 0.0` and judge tuples identical.

## 9. ⛔ DECLARED NOT-RUN

- **seed 3** — gen-2 cache passes the gate; the frozen decoder, external classifier and fitted
  rivals do not exist for it; `Judge(3)` raises. **seed 4** — fails the cache gate.
- **A 1×-matched LSTM / transformer / continuous MLP / table.** One family (`mlp`) is fitted at
  the 1× pin, DECLARED; the other three are a NOT-RUN with wall clock as the reason.
- **dSprites** — the ladder is defined on Shapes3D's `object_hue`/`shape`; no number is produced
  for any other dataset and none is implied.
- **The coverage-matched control the harness proposed** (k=6 refitted on k=3's train index) —
  NOT-RUN; it was the harness's suggestion and costs a further leg.
- ⚠ **Pre-existing and NOT this hub's:** 3 tests in `tests/test_gen_value_holdout.py` fail at
  `aa91785` on a clean tree. Flagged by `w12/kway-ladder-harness`; ⛔ not touched here.

---

## 10. Artifacts

| file | what |
|---|---|
| `LADDER.json` | every rung × seed × arm, both best-learned readings, control verdicts, breaking-point verdicts, end reproduction |
| `LADDER.csv` | one flat row per (rung, seed, arm) |
| `LADDER.md` | the rendered tables, placement label on line 1 |
| `BYTES.json` | the four-part byte ledgers, measured on all 18 legs, never netted |
| `fig_composition_ladder.png` | the two-panel figure (appendix-labelled in its own suptitle) |
| `instrument/INSTRUMENT.json` | corner counts, disjointness, both ends vs the shipped condition |
| `instrument/corner_set_equality.txt` | the k≥3 shared-corner-set proof |
| `PREREG.md`, `AMENDMENT-{1,2,3,4,5}.md`, `AMENDMENT-3-CORRECTION.md`, `HEAD-RULING-K.md` | the declarations, each with its timestamp file |
| `LOG.md` | the full working record, including every correction issued against this hub |
| `runs/<rung>_seed<n>/` | per-leg `LEG.json`, `rows_<arm>.json` (per-corner), stores, address books |

**Scale:** 18 legs × 13 arms = **234 scored arm-runs**, 3 seeds, `failed=[]` on every leg,
**0 NOT-RUN cells** in the matrix.

## 11. What would have proved this hub wrong, and what actually happened

| the task's falsifier | outcome |
|---|---|
| every arm collapses at the same k ⇒ a property of Shapes3D, not of any arm | **did not happen** — no arm has a breaking point at all, because none cleared the bar |
| `pca_tensor` matches or beats the single design at every k ⇒ the single design's justification is deletion, not composition | ⭐ **HAPPENED, in the strongest form: bit-identical at every rung and seed, at 1/18th the bytes.** Reported plainly, as required |
| no arm deteriorates ⇒ the ladder does not discriminate; say so | ⭐ **HAPPENED** — and worse for the ladder: for k ≥ 3 it does not vary the probe at all, only the training data |

## 12. For the Hub to rule on

1. ⛔ **The composition claim does not survive an honest byte match.** Against the 524,288 B
   rivals the store wins everywhere; against a learner matched to its own 24.4× total it loses
   at four rungs of five. **Which baseline does the paper report?** ⭐ This hub reports both on
   every row and recommends the 1× column lead, because the 524,288 B column is measured to
   flatter the arm.
2. ⭐ **`pca_tensor` at 1.37× is bit-identical to the single design at 24.4×.** The single
   design's case is **exact deletion**, and this hub has measured that it is *only* that on
   these rows. Should the composition rows quote `pca_tensor` and let the single design carry
   the deletion rows alone?
3. ⛔ **For k ≥ 3 the ladder is a training-data axis.** Any prose that calls it a
   composition-depth ladder for those rungs is wrong. **Does the paper keep the k ≥ 3 rungs at
   all, or report k=2 vs k=6 as the only genuine depth contrast?**
4. ⚠ **The K\* re-run should report ‖P_singledesign − P_pca‖ per well**, exactly as this hub
   reports it at K = 5 (≤1.8e-08 on the three wide wells). If the span does not move at K\*,
   the two arms will be identical there too and the re-run will not change finding #3.
