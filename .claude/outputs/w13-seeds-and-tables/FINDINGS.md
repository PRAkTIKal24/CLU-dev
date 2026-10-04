# w13 spoke/seeds-and-tables — FINDINGS

**Report to the Hub.** Branch `w13/seeds-and-tables`, worktree
`C:\chlu\worktrees\CHLU-w13-seeds-and-tables`, base `base/iter8 = aa91785`. Box FORGIS_14, 2 threads.
`chlu.__file__ = C:\chlu\worktrees\CHLU-w13-seeds-and-tables\chlu\__init__.py` — asserted before any
number, though this spoke produces no model number.

> ⛔ **SCORED RUNS SPENT: 0.** Every number below is re-read from an artefact at a path and a JSON
> pointer that `verify_tables.py` re-resolves from disk.
> ⛔ **Nothing outside the 52-entry source register was read**, and `$CHLU_CLAUDE_ROOT` was not read.
> ⛔ **Anything not run is a DECLARED NOT-RUN with a reason** — §6, NR1…NR10. Never a zero, never a blank.

---

## 0. ⭐⭐ THE THREE THINGS TO READ FIRST

1. ⛔ **The brief's seed count for the main scored row is wrong by 4×.** The task file says the
   crossover's scored row 1/3 "was spent at **4 seeds**, declared". On disk it is **ONE seed**:
   `/c/chlu/outputs/w12-single-design-crossover/scored/` holds exactly one directory,
   `run1_seed0_K32`, with `/seed = 0` and `/run_n = 1`. What *is* at 3 seeds in that hub is the
   **val** sweep at K = 5 and K = 32 — and that file labels itself `SMOKE_NOT_A_CLAIM: true`
   (`CROSSOVER-val.json` `/SMOKE_NOT_A_CLAIM`). **T7 prints `seeds (n) = 1` as a column and says so
   in its title.** Correction **C15**.
2. ⛔ **"5 seeds where possible" is possible NOWHERE, and the caches are not the reason.** The
   generation-2 caches for seeds 3 and 4 are real, complete and independently re-hashed
   (`CACHE-READY.json` produced 2026-09-21T15:27:15Z, both `status: READY`, 59 block hashes each,
   `shape_dtype_matches_seeds_0_2: true`; `VERIFY.json` `/all_pass = true`; the shipped re-encode gate
   passes at `frac_differing 0.0`). **They still buy no row**: `decoder_v1_s3.pkl`,
   `fw.classifier_dir(3)`/`(4)` and the fitted rivals do not exist —
   `/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json` `/rivals_seeds_present` is `[0, 1, 2]` for all
   seven families — so `fw.Judge(3)` raises and no corner of seeds 3 or 4 can be judged. **NR7.**
3. ⛔ **The brief's K-curve caption quotes the wrong curve and inverts its mechanism.** "7.48× → 2.07×"
   is `total_bytes_parts_1_2_4_only` — the total **with the address book left out**. The honest
   four-part total runs **7.8545× → 4.0877×** and is **U-SHAPED**, minimum **3.9942× at K = 24**,
   because the address book **grows 5.4×** (195,584 → 1,059,824 B) while the covariance falls 6.0×
   (3,391,552 → 560,608 B). T12 prints both totals as separate columns; fig7 panel B draws both.
   Correction **C16**.

---

## 1. ACCEPTANCE — `verify_tables.py` PASSES, and the counts before and after

`verify_tables.py` is **byte-identical** to `scripts/w10_verify_tables.py` (`diff` = empty). It is the
acceptance check, so it was **not edited**. Every table below was produced by editing the
**GENERATOR** — seven exact-string patch scripts in `patch/`, each asserting its anchor occurs exactly
once (w12r's method).

| | BEFORE (w12r, the build this spoke starts from) | AFTER (w13) |
|---|---|---|
| verdict | PASS | **PASS** |
| sources registered / re-hashed | 32 / 32 | **52 / 52** |
| cells in `TABLES.json` | 240 (w10) → 286 (w12r) | **872** |
| per-seed values re-resolved | 854 (3 reductions) | **1,786** (3 reductions) |
| figures (hashes re-verified) | 5 (5) | **8 (8)** |
| declared NOT-RUN rows | 5 | **10** |
| corrections | 14 | **19** |

Command, and where the BEFORE figure comes from:
`python verify_tables.py /c/chlu/outputs/w12r-artefact-mustfix/tables-r1` → PASS, 32 / 286 / 854 / 5 / 5 / 14.
`python verify_tables.py /c/chlu/outputs/w13-seeds-and-tables` → PASS, 52 / 872 / 1,786 / 8 / 10 / 19.

**Rebuilding is deterministic.** Two consecutive `python build_tables.py` runs give: all eight figure
sha256 **identical**; `TABLES.md` byte-identical **except its `**Built**` timestamp line**;
`TABLES.json` **equal ex-`generated_utc`** (checked by loading both and popping that one key).

---

## 2. ⭐⭐ THE TWO NOTATIONS, SEPARATED IN THE MANIFEST

`TABLES.json` `/notation` now defines both symbols and `TABLES.md` prints the definition table before
any wave-12 row.

| symbol | means | measured range | tables |
|---|---|---|---|
| **n** | jointly composed factors — the LADDER RUNGS | 2, 3, 4, 5, 6 | T8, fig6 |
| **K** | WELLS — the partition | 5, 8, 10, 12, 16, 20, 24, 28, 32; **K\* = 32** | T7, T9, T9b, T12, fig7, fig8 |

⛔ **Every ladder row states its K (5). Every wells row states its n.** The collision is real in the
sources: `LADDER.json` `/rungs/5` is a rung of five jointly composed factors while
`LADDER.json` `/K_of_the_store_arms` is 5 wells. `LADDER.json` also still carries `/K_star: null` and
`/K_star_source` = *"CROSSOVER-val.json: K_star=null, 'K=5 reference row absent; rule not applicable'"*
— a stale read of a file that now says `/K_star = 32`. Correction **C19**.

⛔ **T7 (K = 32) and T8 (K = 5) never share a table or a figure axis**, and `LADDER.json` `/placement`
carries the Head's ruling verbatim: *APPENDIX — the K-sweep's ENDPOINT at K = 5. NOT main text.*

---

## 3. THE MAIN TABLE, AS BUILT — seven tables, every row labelled, every cell traced

| table | what | class | K | n | seeds | placement |
|---|---|---|---|---|---|---|
| **T7** | scored rows 3 and 4 on the single design, TEST | RESULT | 32 | one n-set | **1** (id 0) | MAIN |
| **T8** | the n-way ladder | RESULT | 5 | **2…6** | **3** | APPENDIX (Head-ruled) |
| **T9** | deletion — value / conjunction / conjunction-live / item, TEST | RESULT | 32 | — | **1** | MAIN |
| **T9b** | the same four scopes, VAL | **SMOKE** (the artefact says so) | 5 | — | **1** | APPENDIX |
| **T10** | refusal DISPLAY rows | **DISPLAY, never a RESULT** | 32 / 5 / n-a | — | **1 / 3** | MAIN, labelled |
| **T11** | the two ablations | ablation | 5 | — | **3** | MAIN |
| **T12** | the K-curve | SMOKE + exact arithmetic | **5…32** | not varied | **3 / 1** | APPENDIX |

T1…T6 are carried unchanged from w12r. **The seed count is a COLUMN of every one of T7…T12**, and
`TABLES.json` `/seed_ledger` carries the same table machine-readably.

### T7 — rows 3 and 4 at K\* = 32, TEST, 1 seed
`/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json`.
Arm `cov_tensor`: **31.600** on row 3 (60 pair corners), **66.959** on row 4 (49 n-way corners),
**47.495** over all 109, judge accuracy **0.9012**. All four controls read **exactly 0.000** on both
rows at rank/width **0.9688** — above wave 9's decorative threshold of 0.938 — so the control gate
passes and the row may be claimed. Paired Δ against the prototype **table floor +39.2661**, CI95
lower bound **+35.8624** against a **+1.0** margin declared in advance. Comparators (all at the
**524,288 B reference**, 130,433–131,891 params): `mlp` +42.3028, `lstm` +42.7339,
`transformer` +46.2661, `cont_mlp` +47.4679 — **109/109 corners** each.

### T8 — the ladder, n = 2…6, K = 5, 3 seeds
`/c/chlu/outputs/w12-composition-ladder/LADDER.json`. Store arm per-seed means at n = 2:
28.6333 / 31.6500 / 31.3333; at n = 6: 62.6917 / 63.9917 / 66.9833. `pca_tensor` is **identical to
four decimals at every rung and every seed**. No arm has a breaking point
(`/same_k_for_every_arm/statement`), `not_run` is `[]`, 18 legs × 13 arms.

---

## 4. ⛔ BOTH BYTE MATCHINGS ARE PRINTED ON EVERY ROW, AND THE FLIP IS CAPTIONED

| | 1× = the object's own measured total | 524,288 B reference |
|---|---|---|
| K = 5 | **4,118,034 B = 1,029,508** float32 params | 524,288 B = **131,072** params |
| K\* = 32 | **2,143,138 B = 535,784** params (scored object: 2,144,178 B, 4.0897×) | same 131,072 |
| status | ⛔ **DECLARED NOT-RUN (NR8)** — no rival fitted at this size on T7; ONE family (`learned_1x`, an MLP) on T8 | MEASURED; rivals within ±0.63 % of 131,072 |

**Against the 524,288 B rivals the margin GROWS with n** — +9.2167 / +9.6500 / +7.0833 at n = 2 to
+18.2583 / +22.4250 / +14.9917 at n = 6 (`/rungs/{2,6}/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned_5/mean_paired_delta`).
**Against `learned_1x` it is NEGATIVE** at n = 2, 4, 5 and 6 — ceiling-normalised
**−0.0773 / +0.0589 / −0.0734 / −0.0542 / −0.0564** for n = 2…6 (derived; the 20 pointers are in
`TABLES.json` `/tables/T8/byte_matching_flip/...from`). ⛔ **The byte pin does not shift the level, it
flips the sign, and therefore flips how the n-trend reads.** Both columns are on every row of T8 and
both lines are on fig6 panel B.

⛔ **And the 1× column is the WEAKEST available 1× opponent.** Only an MLP was fitted at 1×; a
1×-matched LSTM, transformer, continuous MLP and table are NR8, so the true 1× margin can only be
more negative.

---

## 5. ⭐ THE FOUR-PART LEDGER IS A PRINTED ROW, NEVER NETTED — and the K-curve is TWO curves

The referee rejected the draft partly because Table 1 printed no byte row while the object is ~24.5×
the pin. Every table that carries a store now prints **atoms · covariance/count ledger · address
book · coupling/hole table** separately, plus the summed honest total.

| K | 1 atoms | 2 cov+count | 3 address book | 4 coupling/hole | parts 1+2+4 only | **honest four-part** | × ref (1+2+4) | **× ref (four-part)** |
|---|---|---|---|---|---|---|---|---|
| 5 | 524,288 | 3,391,552 | 195,584 | 6,610 | 3,922,450 | **4,118,034** | 7.4815 | **7.8545** |
| 24 | 524,288 | 735,328 | 827,904 | 6,610 | 1,266,226 | **2,094,130** | 2.4151 | **3.9942 ← minimum** |
| 32 | 516,096 | 560,608 | 1,059,824 | 6,610 | 1,083,314 | **2,143,138** | 2.0663 | **4.0877** |

Sources: `LEDGER-ACROSS-K.json` `/rows/{i}/...` and `CROSSOVER-val.json` `/per_K/{K}/ledger_four_part/...`.

⛔ **The object never reaches the retired pin at rank 64**: the minimum honest total over the whole
sweep is **3.9942×**. "Matched bytes at 524,288 B" is not available to this object under any K, which
is why 1× is the object's own total and 524,288 B is a reference column.

⭐ **On the ladder's own fitted legs** (`BYTES.json` `/summary`): single design **12,523,008 –
13,041,864 B = 23.8857× – 24.8754×**; `pca_tensor` **712,072 – 719,352 B = 1.3582× – 1.3721×** —
≈18× less for identical composition. **The single design's bytes buy exact deletion, not composition.**
⛔ **Part 4 is DECLARED ABSENT from `BYTES.json`, not printed as a zero** — the ladder's ledger file
carries three parts per leg.

---

## 6. DELETION, REFUSAL AND THE ABLATIONS — the rows the main table needed

### T9, deletion at K\* = 32 on TEST, seed 0 (`deletion/test_seed0_K32/deletion.json`)

| scope | rows deleted | our s | our bytes | our rows read | refit-without s / rows | store-scoped exactness | Δ on survivors |
|---|---|---|---|---|---|---|---|
| (a) VALUE red | 11,044 | **0.8998** | 2,081,024 | **0** | 9.259 / 120,028 | 5.960e−08 | +0.0000 (89 corners) |
| (b) CONJ red ∧ cube | **0 — VACUOUS** | 0.0518 | 2,081,024 | **0** | 9.368 / 131,072 | 0.000e+00 | +0.0000 (109) |
| (b′) RED SPHERE | 5,511 | 0.3916 | 2,081,024 | **0** | 9.262 / 125,561 | 5.383e−07 | +0.0000 (109) |
| (c) ITEM 1 % | 1,311 | 0.1055 | 2,081,024 | **0** | 9.788 / 129,761 | 2.338e−07 | +0.0000 (109) |

⛔ **The probe that bounds the whole claim** (`deletion/probe_test_seed0_K32.json`): deleting all
11,044 red rows leaves the 20 red-requesting corners at **32.25 → 32.25** and judge accuracy
**0.8958 → 0.8958**. Uncarried, the read moves **0.610083** on red-requesting corners and **0.614277**
on the others — **the basis change is generic, not targeted**. Carried, it moves 6.19e−08. At K\* = 32
the value identity lives in the **address book**, which the deletion carries and does not refit.
⛔ *"The store has forgotten red"* and *"the object no longer answers red"* are different statements
and **only the first is demonstrated.**

**T9b (VAL, K = 5) is a separate table and is labelled SMOKE**, because
`deletion/val_seed0_K5/deletion.json` `/SMOKE_NOT_A_CLAIM` is `true` while the TEST file's is `false`.
Different condition, different K, different claim status ⇒ never one table.

### T10, refusal DISPLAY — two sides, printed, never differenced
OURS (TEST, K\* = 32, seed 0): **2 holes, 2 bytes, 0 data rows read, no refit**, layout asserted equal
to `gen_field`'s over **1,322** pair slots, 60 indices checked. One hole (red ∧ cube) is refusable
**although no red cube was ever written** — refusal is a separate mechanism from forgetting.
COMPARATOR (`BASELINES-READY.json`, 3 seeds): refuse-token transformer, AUROC **0.8692 / 0.8747 /
0.8755** on the value deletion and **0.8269 / 0.8340 / 0.8133** on the conjunction, with a **FALSE
refusal rate on possible requests of 0.0947 / 0.1270 / 0.1094**. ⛔ These are **not the same quantity**
— ours is a cost line, theirs is a feature-level separation diagnostic with no judge and no decoder.
The head-to-head is **NR9**.
⛔ **The item deletion has no refusal row at all, and that is MEASURED, not a zero**: 0 of 164
deleted-row requests are certifiably impossible, on all three seeds.

### T11, the two ablations (w5 compose-field, TEST, 3 seeds, one 524,288 B pin)
- **Composition-only store** (`pca_tensor`, ARM): row 3 **29.7167 / 31.6167 / 31.4500**, row 4
  **63.6122 / 64.3265 / 67.6531**. Composes; **cannot forget** — it keeps no sufficient statistics.
- **Item store** (`oracle_content_items`, and the artefact labels it **"NOT AN ARM (ceiling, 64 REAL
  ITEMS @524,288 B)"**): row 3 **18.7500 / 18.2333 / 20.4667**, row 4 **32.3878 / 39.1224 / 41.7755**.
  Against `rival_mlp_180k` that is **−21.0816 / −16.5306 / −18.0816** on row 4, 3-seed mean
  **−18.5646** — the programme's "≈18.6 tuples/corner price of keeping items", now traced to its six
  pointers rather than to a prose line.
- The other half, from `w5-delete/run1/report.json`: **eviction refuses at 1.0000 on all three seeds
  and costs −2.0899 / −0.9101 / −6.7416 composition** against a declared ±1.0 bar, while **the hole
  refuses at 1.0000 and costs exactly 0.0000**. ⚠ `row_removal_s` reads 0.0 on all three seeds — a
  measured zero below that harness's timer resolution, printed with its pointer, not a missing value.

---

## 7. DECLARED NOT-RUN — five new ones, each with a reason and a launch line

| id | row | reason in one line |
|---|---|---|
| **NR6** | single-design rows 3/4 at seeds 1 and 2 | wall clock; 2 of 3 scored runs remain. Run 2/3 should first buy a **TEST** number at K = 5 — the third pre-declared falsification is untestable while only K\* has a TEST row |
| **NR7** | **any 5-seed row** | the caches exist and pass every gate; the **judge** does not exist for seeds 3/4 (no decoder, no external classifier, no fitted rivals), and the caches landed after both hubs closed their seed gates |
| **NR8** | the **1× comparator column** | `BASELINES-READY.json` pins `param_target = 131,072` and names no 1× entry point. ⛔ The error runs AGAINST the programme |
| **NR9** | the refusal DISPLAY **head-to-head** | wall clock; both sides exist separately and are printed in T10 |
| **NR10** | `generation-table` run 3 on the K\* object | `exp_gen_cell.py`'s arm registry would have to be wired; the run remains UNSPENT |

NR1…NR5 are carried unchanged from w12r. ⛔ **NR3** (the w8 single-design object's rows 3/4 and
post-deletion composition/refusal at 3 seeds) is the one wave 12 **partially discharged** — T7, T9 and
T10 are its 1-seed instance at K\* = 32; NR6 carries the remaining two seeds.

## 8. CORRECTIONS ISSUED — five new, C15…C19

**C15** the brief's "4 seeds" for the scored row (disk: 1). **C16** the "7.48× → 2.07×" caption quotes
the three-part total and inverts the mechanism. **C17** the crossover FINDINGS §7 VAL cost line
(1.45 / 0.32 / 0.72 / 0.28 s and 13.2–14.7 s) disagrees with its own artefact (2.4543 / 0.1874 /
0.6205 / 0.2554 s and 10.843–23.486 s); every other column in that section matches exactly.
**C18** the crossover FINDINGS §5 prints the val sweep as 1 seed at every K and K\* at 0.9016; the
final `CROSSOVER-val.json` carries **3 seeds at K = 5 and K = 32** and K\* at **0.929012**, drop
**−0.033758** — the prose was written against an earlier state of the same file. **C19** the n/K
collision and `LADDER.json`'s stale `K_star: null`.

⚠ **C17 and C18 are both instances of the same hazard the standing rules name: a published file is
not a finished file — here in the growing direction.** `TABLES.json` `/gate_checks` records the
completeness check run on every wave-12 artefact before a cell was emitted from it (END_UTC present;
`LADDER.json` 6 rungs × 13 arms × 3 seeds with `not_run = []`; `BYTES.json` 18/18 legs;
`scored.json` `SCORED = true` and 60 + 49 = 109 corners; `CROSSOVER-val.json` 9 K all non-null;
`CACHE-READY.json` both seeds READY with 59/59 block hashes; `BASELINES-READY.json` 150 records,
seeds [0,1,2]; `RIVALS-READY.json` seeds [0,1,2]).

---

## 9. ⛔ WHAT WOULD PROVE THIS SPOKE WRONG — the two declared checks, answered

1. **"A cell you cannot trace ⇒ the evidence base is thinner than the claims; say so."**
   — 872 of 872 cells re-resolve from disk. ⚠ **But the traceable ≠ claimable gap is real and is
   printed:** T9b and T12 are built entirely from artefacts whose own `SMOKE_NOT_A_CLAIM` field is
   `true`, and T7/T9/T10-ours rest on **one** seed. **58 entries in T7…T12 (78 across the whole
   manifest) are `derived` carriers, not cells** — each carries
   `machine_checked_by_verify_tables: false`, its formula and every pointer behind it, because the
   verifier asserts a cell equals a raw field on disk and a derived number is not one. Most of the 58
   are declared labels ("this scope deletes no corner", "— (this IS the arm)") that exist so no blank
   is ever printed; the genuinely **computed** ones are **three**: T8's ceiling-normalised 1× deltas
   (20 pointers) and T11's two row-4 differences (6 pointers each), and all three were recomputed
   independently from their pointers before being written. ⭐ The evidence base is **thinner in seeds**
   than the claims the draft wants, and **exactly as thick in provenance** as the table says.
2. **"Rows at different pins, seeds or decoder recipes sharing one table ⇒ the table is void; split it."**
   — split, three times: T9 (TEST, K = 32) from T9b (VAL, K = 5); T7 (K = 32) from T8 (K = 5);
   T10's 1-seed our-rows from its 3-seed comparator rows, with the seed count as a column and the
   explicit statement that the two sides are not the same quantity and are not differenced.

## 10. FOR THE HUB TO RULE ON

1. ⛔ **The composition claim does not survive an honest byte match, and the ladder hub said so first.**
   The arm loses to a learner matched to its own bytes at four rungs of five. **Which column leads in
   the paper?** This spoke prints both on every row and recommends the 1× column lead, with NR8 stated
   beside it.
2. ⭐ **`pca_tensor` at 1.37× is identical to the single design at 24.4×.** Should the composition rows
   quote `pca_tensor` and let the single design carry the deletion rows alone? T11 is built so either
   choice is one edit to the generator.
3. ⛔ **T7 is a 1-seed row in the MAIN table.** Either run 2/3 and 3/3 (NR6) before the draft freezes,
   or the main table's headline row carries `n = 1` in print. There is no third option that is honest.
4. ⛔ **For n ≥ 3 the ladder is a training-data axis, not a composition-depth axis** — the four rungs
   probe the identical 120 corners. Does the paper keep those rungs, or report n = 2 vs n = 6 as the
   only genuine depth contrast?

## 11. ARTEFACTS

| file | what |
|---|---|
| `TABLES.md` | 1,506 lines; T1…T6 carried, **T7…T12 added**, notation block, seed ledger, completeness gates |
| `TABLES.json` | the manifest; 52 sources, 872 cells, `/notation`, `/seed_ledger`, `/gate_checks`, `/w13_fold` |
| `figures/` | fig1–fig5 carried (hashes unchanged), **fig6** the ladder in n, **fig7** the K-curve, **fig8** the four-part ledger |
| `build_tables.py` | the generator, 4,545 lines. Also committed as `scripts/w13_build_tables.py` |
| `verify_tables.py` | ⛔ **byte-identical** to `scripts/w10_verify_tables.py` — the acceptance check, not edited |
| `patch/p1..p7*.py` | the seven exact-string patches from w12r's generator to this one; each asserts its anchor occurs exactly once. Also committed as `scripts/w13_patch/` |
| `LOG.md`, `START_UTC.txt`, `END_UTC.txt` | the record |
