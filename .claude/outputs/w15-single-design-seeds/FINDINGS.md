# w15 `hub/single-design-seeds` — FINDINGS

**Branch** `w15/single-design-seeds` · **base** `base/iter8 = 148f3b7` ·
**worktree** `C:/chlu/worktrees/CHLU-w15-single-design-seeds` ·
**output dir** `C:/chlu/outputs/w15-single-design-seeds/`
**Env asserted:** `chlu.__file__ = C:\chlu\worktrees\CHLU-w15-single-design-seeds\chlu\__init__.py`,
jax 0.9.0 / numpy 2.4.1 / equinox 0.13.4, feature cache **generation 2**
(`C:/chlu/outputs/spoke-a-rivals/cache_generation2`, 480,000 rows, scope `full`),
asserted per seed by `exp_gen_delete_row7.assert_generation2`.

**Acceptance artefacts:** this file, `T7-3SEEDS.json`, and `T7-3SEEDS.md` (the same
numbers rendered as tables). ⛔ No number below lives only in prose; each carries the
artefact path it was read from.

---

## 0. THE HEADLINE, IN FOUR SENTENCES

1. **The two scored runs I was offered were already spent, on exactly this content.**
   `single-design` is **EXHAUSTED at 3/3**; seeds 1 and 2 at K\* = 32 on TEST exist as
   scored artefacts. ⭐ **I spent ZERO scored runs.**
2. **T7 now prints at n = 3, seeds [0, 1, 2]**, and **all four charter-C controls
   collapse to exactly 0.0000 on every one of the three seeds** — measured per seed, not
   assumed from seed 0. The store row is claimable.
3. **The deletion rows exist at all three seeds for the first time** (they existed at
   seed 0 only), with **store-scoped exactness at 2 × 10⁻⁸ – 5 × 10⁻⁷**, adaptation in
   **0.04 – 3.8 s reading 0 data rows**, against a refit-without that reads
   **120,028 – 131,072 rows** in **9.3 – 18.6 s**.
4. ⚠ **And the row that does not flatter this hub:** deleting every red row leaves the
   red corners **still able to compose red**, at all three seeds. **Data forgetting and
   capability removal are two different things on this object**, and the deletion claim
   here is store-scoped exactness plus a data-free adaptation cost — **not capability
   erasure**. §6.

---

## 1. THE SCORED-RUN ACCOUNTING — 0 SPENT, AND WHY

My task file offered `single-design` runs **2/3** and **3/3** and instructed me to check
first whether w12b had already produced these seeds. It had.

| what | where | what it says |
|---|---|---|
| ledger row 43 | `/c/chlu/SCORED-LEDGER.md` | `2026-09-21T21:33:51Z \| w12b hub/nr6-and-kstar \| single-design \| 2/3` |
| ledger row 44 | `/c/chlu/SCORED-LEDGER.md` | `... \| 3/3 \| ... "adds seeds 1 and 2 at K* = 32 on TEST, taking T7 to 3 seeds with the count printed on the row"` |
| seed 0 artefact | `C:\chlu\outputs\w12-single-design-crossover\scored\run1_seed0_K32\scored.json` | `SCORED: true`, `condition: test`, `K_star: 32` |
| seed 1 artefact | `C:\chlu\outputs\w12b-nr6-and-kstar\scored\run3_seed1_K32\scored.json` | `SCORED: true`, `condition: test`, `K_star: 32`, `run_n: 3` |
| seed 2 artefact | `C:\chlu\outputs\w12b-nr6-and-kstar\scored\run3_seed2_K32\scored.json` | same |
| the table already built | `C:\chlu\outputs\w12b-nr6-and-kstar\NR6-T7.md` | `K32: n = 3 seeds [0, 1, 2]` |

⇒ **Runs 2/3 and 3/3: DECLARED NOT-NEEDED and NOT AVAILABLE.** The target is exhausted
at 3/3 and this hub spent **0**. A ledger **NOTE** — not a run row — was appended at
**2026-09-22T21:31:47Z**, before any number in this file, saying exactly that.

**What was left unproduced by those charged runs**, and is what I did instead: the
PREREG's **RESULT 3 — DELETION, three scopes, same seeds**, which existed at **seed 0
only** (`w12-single-design-crossover/deletion/test_seed0_K32/deletion.json`; directory
listing shows no seed-1 or seed-2 deletion artefact anywhere). I completed RESULT 3 at
seeds 1 and 2 **on the same already-fitted arms** (`arm.pkl` from those same scored
directories), **same K\* = 32, same TEST condition, same PREREG + AMENDMENT 1**. ⛔ No
bar moved, no K selected, no design change. That is the unproduced remainder of runs
already charged, not a fourth run.

⭐ **The integrity check that makes this legitimate, and it passed exactly at both
seeds.** The before-deletion arm is re-scored here from the same `arm.pkl` and must
reproduce the already-charged T7 per-corner row bit for bit:

```
seed 1   BEFORE reproduces the scored row: max |per-corner deviation| = 0.0   → identical
seed 2   BEFORE reproduces the scored row: max |per-corner deviation| = 0.0   → identical
```
(`deletion/test_seed{1,2}_K32/deletion.json` → `before_reproduces_the_scored_row`.)

---

## 2. T7 — ROWS 3 AND 4 AT K\* = 32 ON TEST, **n = 3, SEEDS [0, 1, 2]**

Metric `distinct_valid_novel_tuples` **per corner**; the corner is the unit.
Row 3 = **60** tiled `pair-*` sub-corners, row 4 = **49** Montero `nway-*` corners.
Source: `T7-3SEEDS.json` → `T7_rows_3_and_4`, read from the three `scored.json` above.

| arm | role | row 3 (pair) mean ± sd | row 4 (n-way) mean ± sd | per-seed row 3 | per-seed row 4 | **n seeds** |
|---|---|---|---|---|---|---|
| `cov_tensor` | **THE ARM** | **32.350 ± 0.815** | **67.177 ± 1.320** | 31.600 / 32.233 / 33.217 | 66.959 / 65.980 / 68.592 | **3** |
| `landscape_removed` | control | 0.000 ± 0.000 | 0.000 ± 0.000 | 0 / 0 / 0 | 0 / 0 / 0 | 3 |
| `landscape_shuffled` | control | 0.000 ± 0.000 | 0.000 ± 0.000 | 0 / 0 / 0 | 0 / 0 / 0 | 3 |
| `landscape_shuffled_across_wells` | control | 0.000 ± 0.000 | 0.000 ± 0.000 | 0 / 0 / 0 | 0 / 0 / 0 | 3 |
| `random_atoms` | control | 0.000 ± 0.000 | 0.000 ± 0.000 | 0 / 0 / 0 | 0 / 0 / 0 | 3 |
| `mlp__as_trained` | 524,288 B REFERENCE comparator | 3.878 ± 1.229 | 7.565 ± 4.381 | 2.933 / 3.433 / 5.267 | 7.959 / 3.000 / 11.735 | 3 |
| `lstm__as_trained` | 524,288 B REFERENCE comparator | 3.372 ± 1.341 | 9.810 ± 4.415 | 2.500 / 2.700 / 4.917 | 7.531 / 7.000 / 14.898 | 3 |
| `transformer__as_trained` | 524,288 B REFERENCE comparator | 1.017 ± 0.606 | 2.429 ± 2.098 | 0.667 / 0.667 / 1.717 | 1.918 / 0.633 / 4.735 | 3 |
| `cont_mlp__as_trained` | 524,288 B REFERENCE comparator | 0.006 ± 0.010 | 0.075 ± 0.024 | 0.000 / 0.000 / 0.017 | 0.061 / 0.061 / 0.102 | 3 |
| `table` | **PRINTED FLOOR, never a rival** | 5.167 ± 1.468 | 10.769 ± 3.567 | 5.833 / 3.483 / 6.183 | 11.163 / 7.020 / 14.122 | 3 |

⛔ **The comparator column is the 524,288 B REFERENCE column, NOT a 1× column.**
`BASELINES-READY.json` pins `param_target = 131,072` and names no 1×-matched entry
point. **The 1× column is a DECLARED NOT-RUN** (AMENDMENT 1 §A1.2), and the direction of
that error runs **against** this hub: a 1×-matched rival would carry several times these
parameters and would be **stronger**, so the available comparison **flatters the arm**.
⛔ No row here is reported as a 1× result.

⚠ **A second asymmetry, measured this wave and also against this hub.** The comparators
were fitted at `BASELINES-READY.json → per_seed.*.build_scale.n_fit_rows_used = 16,384`
while the store is fitted on the full **131,072**-row pool. A scale-matched rival would
be stronger still. Printed on every rival row in `T7-3SEEDS.json`; ⛔ never netted
against the 1× caveat, both are stated.

---

## 3. THE CHARTER-C CONTROL GATE — MEASURED AT **EVERY** SEED

⛔ **No store row is claimed unless all four controls collapse on every seed. A control
that collapsed at seed 0 is not assumed to collapse at seeds 1 and 2.**
Source: `T7-3SEEDS.json` → `control_gate`, read per seed from the three `scored.json`.

| seed | arm (all 109 corners) | `landscape_removed` | `landscape_shuffled` | `landscape_shuffled_across_wells` | `random_atoms` | all four collapse? |
|---|---|---|---|---|---|---|
| 0 | 47.4954 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **YES** |
| 1 | 47.4037 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **YES** |
| 2 | 49.1193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **YES** |

**ALL FOUR COLLAPSE ON EVERY SEED: YES ⇒ the store rows are claimable at n = 3.**

Paired estimand (`fw.paired_estimand`, margin 1.0, the corner is the unit), every
control, every seed — **109 / 109 corners won, 0 lost, on all twelve legs**:

| seed | mean paired delta vs every control | CI95 | won / tied / lost | beats the 1.0 margin? |
|---|---|---|---|---|
| 0 | 47.495 | [44.10, 51.03] | 109 / 0 / 0 | **YES** |
| 1 | 47.404 | [44.06, 50.79] | 109 / 0 / 0 | **YES** |
| 2 | 49.119 | [45.77, 52.53] | 109 / 0 / 0 | **YES** |

(All four controls read identically at each seed because each scores exactly 0 valid
novel tuples; the twelve individual rows are in `T7-3SEEDS.md`.)

⭐ **The mechanism for the collapse is measured, not asserted**, and it is *not* the same
at every seed — which is why the caveat column is measured per seed:

| seed | `landscape_removed` judged-sphere excess | `random_atoms` judged-sphere excess | arm's own excess (before) |
|---|---|---|---|
| 0 | **−0.3291** | −0.3078 | +0.0074 |
| 1 | **+0.6709** | −0.2328 | +0.0063 |
| 2 | **+0.6709** | −0.1338 | +0.0030 |

At seed 0 a zeroed store degrades towards **cube** (the held-out shape). At seeds 1 and 2
it degrades towards **sphere** — exactly the hazard `chlu/data/gen_value_holdout.py:299`
named. ⛔ It does **not** touch the control gate (a degraded arm still scores 0 valid
novel tuples, because `judge_valid` requires the *whole* requested tuple), but it does
mean the red-sphere conjunction row must be read with its caveat column, and the caveat
column is now **measured at each seed** rather than carried from seed 0. Source:
`deletion/test_seed{1,2}_K32/deletion.json` → `sphere_caveat.measured_upper_bound_degraded_arms`
and `w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` for seed 0.

---

## 4. THE FOUR-PART BYTE LEDGER, PER SEED — ⛔ NEVER NETTED

**Convention, named on the table as the charter requires:** four-part total, **summed**;
part 1 atoms = `sum_k A × d_k` float32 at 4 B/float; part 2 = the packed float64
second-moment triangle + first moment + count that make the deletion exact; part 3 = the
fitted `TensorAddress` parameters; part 4 = the coupling / hole table that carries the
refusal holes. The **524,288 B** figure is a **REFERENCE COLUMN only** this wave — it is
**not a budget**, and the object is **1× its own total**. ⛔ The memory ledger is printed
separately and is **never matched** between arms.

Source: `T7-3SEEDS.json` → `byte_ledger_per_seed`.

| seed | 1. atoms | 2. covariance + count | 3. address book | 4. coupling / hole | **four-part total** | × the 524,288 B REFERENCE | n seeds |
|---|---|---|---|---|---|---|---|
| 0 | 516,096 | 560,608 | 1,060,864 | 6,610 | **2,144,178 B** | 4.0897× | 1 |
| 1 | 516,096 | 560,608 | 1,054,104 | 6,610 | **2,137,418 B** | 4.0768× | 1 |
| 2 | 516,096 | 560,608 | 1,058,264 | 6,610 | **2,141,578 B** | 4.0847× | 1 |

Geometry, identical at all three seeds (`scored.json` → `rank_over_width_max`,
`A_requested`, `well_sizes`): K = **32** wells of width **64**, **A = 63**,
**rank/width = 0.9844**.

---

## 5. THE DELETION ROWS — THREE SCOPES, ONE OBJECT, EVERY COST LINE PRINTED

⭐ **"EXACT" means STORE-SCOPED exactness**, everywhere the word appears:
`downdate(delete rows)` equals the store fitted from scratch without those rows, at the
same partition and the same rank. ⭐ **The frozen encoder and the frozen decoder are
declared OUTSIDE the deletable object and are NOT refit.**

⭐ **Post-deletion adaptation is LIGHT and DATA-FREE** — the covariance and count are
downdated by subtraction, the atoms are re-eigendecomposed from the downdated statistic,
the address book is carried across the basis change, and **no row of data is read**.
⛔ It is never "no changes".

⛔ **The conjunction pin.** `red ∧ cube` deletes **0 rows** — `shape = cube` is held out
of the value-holdout split entirely — and is reported **with its count of 0 and its
reason**, never as a passing row. The live pin is `object_hue = 0.0 ∧ shape = 2.0`, named
**"red sphere"** everywhere, adopted from `BASELINES-READY.json → pins.conjunction_deletion`
so that arm and comparator delete the same rows.

Source: `T7-3SEEDS.json` → `deletion_rows`; seed 0 from
`w12-single-design-crossover/deletion/test_seed0_K32/deletion.json`, seeds 1 and 2 from
`w15-single-design-seeds/deletion/test_seed{1,2}_K32/deletion.json`.

| scope | seed | rows deleted | survivors | novel/corner before → after | paired after − before | exactness max &#124;mode dev&#124; | ranks agree | **OUR** s | **OUR** B rewritten | **OUR** data rows read |
|---|---|---|---|---|---|---|---|---|---|---|
| `value` (red) | 0 | 11,044 | 89 | 50.921 → 50.921 | **+0.0000** | 5.960e−08 | YES | 0.900 | 2,081,024 | **0** |
| `value` (red) | 1 | 11,044 | 89 | 50.528 → 50.528 | **+0.0000** | 4.821e−08 | YES | 3.750 | 2,074,368 | **0** |
| `value` (red) | 2 | 11,044 | 89 | 52.607 → 52.607 | **+0.0000** | 2.011e−08 | YES | 1.309 | 2,078,464 | **0** |
| `conjunction` (red ∧ cube) | 0 | **0** | 109 | 47.495 → 47.495 | +0.0000 | 0.000e+00 | YES | 0.052 | 2,081,024 | **0** |
| `conjunction` (red ∧ cube) | 1 | **0** | 109 | 47.404 → 47.404 | +0.0000 | 0.000e+00 | YES | 0.051 | 2,074,368 | **0** |
| `conjunction` (red ∧ cube) | 2 | **0** | 109 | 49.119 → 49.119 | +0.0000 | 0.000e+00 | YES | 0.043 | 2,078,464 | **0** |
| **`red sphere`** | 0 | 5,511 | 109 | 47.495 → 47.495 | **+0.0000** | 5.383e−07 | YES | 0.392 | 2,081,024 | **0** |
| **`red sphere`** | 1 | 5,511 | 109 | 47.404 → 47.404 | **+0.0000** | 1.074e−08 | YES | 0.768 | 2,074,368 | **0** |
| **`red sphere`** | 2 | 5,511 | 109 | 49.119 → 49.119 | **+0.0000** | 6.253e−08 | YES | 0.831 | 2,078,464 | **0** |
| `item` (1 %) | 0 | 1,311 | 109 | 47.495 → 47.495 | **+0.0000** | 2.338e−07 | YES | 0.106 | 2,081,024 | **0** |
| `item` (1 %) | 1 | 1,311 | 109 | 47.404 → 47.404 | **+0.0000** | 1.486e−08 | YES | 0.418 | 2,074,368 | **0** |
| `item` (1 %) | 2 | 1,311 | 109 | 49.119 → 49.119 | **+0.0000** | 3.162e−08 | YES | 0.384 | 2,078,464 | **0** |

⛔ The `conjunction` (red ∧ cube) rows are printed **with their count of 0** and are
**not** a passing result; they are the literal specification, reported as vacuous with
its reason, exactly as the task file requires.

### 5.1 Adaptation cost — ours against the refit-without reference, per seed

| scope | seed | arm | seconds | **data rows read** | bytes rewritten | refit? |
|---|---|---|---|---|---|---|
| `value` | 0 | **ours** | 0.900 | **0** | 2,081,024 | no |
| `value` | 0 | refit-without | 9.259 | 120,028 | 1,065,216 | **yes** |
| `value` | 1 | **ours** | 3.750 | **0** | 2,074,368 | no |
| `value` | 1 | refit-without | 18.595 | 120,028 | 1,065,216 | **yes** |
| `value` | 2 | **ours** | 1.309 | **0** | 2,078,464 | no |
| `value` | 2 | refit-without | 14.430 | 120,028 | 1,065,216 | **yes** |
| `red sphere` | 0 | **ours** | 0.392 | **0** | 2,081,024 | no |
| `red sphere` | 0 | refit-without | 9.262 | 125,561 | 1,065,216 | **yes** |
| `red sphere` | 1 | **ours** | 0.768 | **0** | 2,074,368 | no |
| `red sphere` | 1 | refit-without | 17.402 | 125,561 | 1,065,216 | **yes** |
| `red sphere` | 2 | **ours** | 0.831 | **0** | 2,078,464 | no |
| `red sphere` | 2 | refit-without | 11.539 | 125,561 | 1,065,216 | **yes** |
| `item` | 0 | **ours** | 0.106 | **0** | 2,081,024 | no |
| `item` | 0 | refit-without | 9.788 | 129,761 | 1,065,216 | **yes** |
| `item` | 1 | **ours** | 0.418 | **0** | 2,074,368 | no |
| `item` | 1 | refit-without | 17.624 | 129,761 | 1,065,216 | **yes** |
| `item` | 2 | **ours** | 0.384 | **0** | 2,078,464 | no |
| `item` | 2 | refit-without | 18.649 | 129,761 | 1,065,216 | **yes** |

⚠ **Read the bytes column honestly.** Our "bytes rewritten" is **larger** than the
refit-without reference's (≈ 2.08 MB against 1.07 MB) because ours counts the atoms
re-derived **plus** the covariance/count downdated **plus** the address book carried,
while the reference rewrites only the atoms and the statistic. The quantity that
separates the two arms is **data rows read: 0 against 120,028–131,072**, and the seconds.
⛔ Nothing here is netted.

### 5.2 A defect of w12's harness, found at seed 1 and fixed before any number

`w12_deletion_rows.py` carries the address book onto the **width-matched** post-deletion
basis and then reads the store through the **raw** one. Where the widths agree those are
the same array, and **at seed 0 they always agreed** (`basis_width_adjustments` is `[]`
on every scope), which is why w12 never saw it. At **seed 1, value scope, well 14**, the
mean's perpendicular part falls below `pinned_basis`'s 1e-4 admission threshold after the
11,044 red rows go, so the raw post basis is **43** columns wide against the book's
**44**, and `chlu/core/gen_read.py:133` dies in `C @ Uk.T`.

⭐ **Fix:** the book is read through the basis it was **carried onto**. The padding
column is exactly zero, so the book's coefficient for a direction the store no longer
holds is projected onto nothing — which is what the deletion *means* and what
`match_basis_width`'s own docstring says the padding is for. **Recorded on the row**
(`deletion/test_seed1_K32/deletion.json` → `scopes.value.basis_width_adjustments` =
`[{"well": 14, "rank_before": 44, "rank_after": 43, "action": "zero-padded to the book's width"}]`),
never done silently. ⛔ At seed 0 the two arrays are identical, so **this changes no
published number**. Same class of defect as AMENDMENT 1 §A1.4, handled the same way.

### 5.3 The refusal DISPLAY row — ⛔ reported, never claimed

Mechanism (ii), a **different array** from the statistic. Two holes punched
(`red ∧ cube` and `red sphere`) in a **1,322 B** coupling table, cost **2 bytes
rewritten, 0.000 s, 0 data rows read, no refit**, at all three seeds
(`deletion/test_seed{1,2}_K32/deletion.json → display_refusal`; seed 0 the same).
⛔ **The comparator is a DECLARED NOT-RUN**: the only permitted one is the refuse-token
transformer, and wave 9 measured a refuser of that quality at **+0.047** separation with
a ridge probe at **AUC 0.48–0.55, i.e. chance**, so the comparison is not yet
informative and the row says so.

---

## 6. ⚠ THE RESULT THAT DOES NOT FLATTER THIS HUB, AND IT REPLICATES AT ALL THREE SEEDS

Deleting **every** red row leaves the **20 deleted (red-requesting) corners** still able
to compose red:

| seed | value scope, the 20 DELETED corners | novel/corner before → after | judge accuracy before → after |
|---|---|---|---|
| 0 | 20 | 32.250 → 32.250 | 0.8958 → 0.8958 |
| 1 | 20 | 33.500 → 33.500 | 0.9306 → 0.9306 |
| 2 | 20 | 33.600 → **33.550** | 0.9333 → **0.9319** |

(`deletion/test_seed{1,2}_K32/deletion.json → scopes.value.after.deleted` and
`.before_same_corners.deleted`; seed 0 from w12's artefact.)

⭐ This is the PREREG's §5 point **measured at n = 3 rather than asserted**: a
compositional store can still compose "red sphere" from "red" and "sphere" after the
red-sphere rows are gone — **that is composition working, not deletion failing** — and
the hole is the separate mechanism that makes the request refusable. ⛔ Both are reported,
separately, on the same object.

⛔ **But stated plainly rather than buried:** if a reader's question is *"after deletion,
can the system still produce the deleted content?"*, the answer on this object is **yes**,
at all three seeds. **The deletion claim here is store-scoped exactness plus a data-free
adaptation cost, not capability erasure.** Capability removal is the **hole's** job, and
the hole is a **DISPLAY** row with a **DECLARED NOT-RUN comparator** (§5.3) — so it is
**not claimed**. ⚠ **An ICLR reader will ask this question, and the honest answer is the
one above.**

---

## 7. THE RIVAL DELETION COLUMN — GENUINELY NEW, AT ALL THREE SEEDS

⛔ **A rival deletes ONLY by retrain-without, and its cost is printed. Rival as-trained is
printed beside it and is NOT a deletion mechanism.** This column was **absent at every
seed including 0** — seed 0's `deletion.json` carries
`display_refusal.comparator_status = "DECLARED NOT-RUN"` and no rival arm at all.

Every figure is measured on the **same survivor corners** as the arm, so the rival's
as-trained → retrain-without step is a before/after on exactly the corners ours uses.
Source: `T7-3SEEDS.json` → `rival_deletion_column`;
`deletion/test_seed0_K32/rivals.json` for seed 0 and the two `deletion.json` for seeds
1 and 2. Checkpoints:
`C:\chlu\outputs\w12-deletion-baselines\comparators\seed_{0,1,2}\<fam>__retrain_without__<spec>.rival.eqx`,
loaded through `chlu.eval.rivals.w12_deletion_baselines.load_comparator` as
`BASELINES-READY.json` names it.

### 7.1 `value` scope — delete red (89 survivor corners)

| seed | family | rival as-trained | rival retrain-without | **rival after − before** | **ours after deletion** | paired ours − rival | CI95 | won/tied/lost | beats 1.0? | rival retrain s | rival rows read |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `mlp` | 5.551 | 4.022 | −1.528 | **50.921** | 46.899 | [43.15, 50.58] | 89/0/0 | **YES** | 8.86 | 15,032 |
| 0 | `lstm` | 5.247 | 3.708 | −1.539 | **50.921** | 47.213 | [43.57, 50.85] | 89/0/0 | **YES** | 15.99 | 15,032 |
| 0 | `transformer` | 1.494 | 0.865 | −0.629 | **50.921** | 50.056 | [46.21, 53.85] | 89/0/0 | **YES** | 34.62 | 15,032 |
| 0 | `cont_mlp` | 0.034 | 0.067 | +0.034 | **50.921** | 50.854 | [46.99, 54.69] | 89/0/0 | **YES** | 8.87 | 15,032 |
| 1 | `mlp` | 2.865 | 2.169 | −0.697 | **50.528** | 48.360 | [44.49, 52.20] | 89/0/0 | **YES** | 16.74 | 15,032 |
| 1 | `lstm` | 4.652 | 3.921 | −0.730 | **50.528** | 46.607 | [43.04, 50.11] | 89/0/0 | **YES** | 19.91 | 15,032 |
| 1 | `transformer` | 0.596 | 0.112 | −0.483 | **50.528** | 50.416 | [46.55, 54.17] | 89/0/0 | **YES** | 37.14 | 15,032 |
| 1 | `cont_mlp` | 0.034 | 0.056 | +0.022 | **50.528** | 50.472 | [46.60, 54.25] | 89/0/0 | **YES** | 13.59 | 15,032 |
| 2 | `mlp` | 8.697 | 6.697 | −2.000 | **52.607** | 45.910 | [42.33, 49.55] | 89/0/0 | **YES** | 6.35 | 15,032 |
| 2 | `lstm` | 10.663 | 8.449 | −2.213 | **52.607** | 44.157 | [40.60, 47.74] | 89/0/0 | **YES** | 11.57 | 15,032 |
| 2 | `transformer` | 3.360 | 2.281 | −1.079 | **52.607** | 50.326 | [46.64, 54.02] | 89/0/0 | **YES** | 23.24 | 15,032 |
| 2 | `cont_mlp` | 0.067 | 0.112 | +0.045 | **52.607** | 52.494 | [48.72, 56.24] | 89/0/0 | **YES** | 5.85 | 15,032 |

### 7.2 `red sphere` scope (109 survivor corners)

| seed | family | rival as-trained | rival retrain-without | rival after − before | ours after deletion | paired ours − rival | CI95 | won/tied/lost | beats 1.0? | rival retrain s | rival rows read |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `mlp` | 5.193 | 4.413 | −0.780 | **47.495** | 43.083 | [39.94, 46.31] | 109/0/0 | **YES** | 9.22 | 15,717 |
| 0 | `lstm` | 4.761 | 4.055 | −0.706 | **47.495** | 43.440 | [40.33, 46.64] | 109/0/0 | **YES** | 15.66 | 15,717 |
| 0 | `transformer` | 1.229 | 1.000 | −0.229 | **47.495** | 46.495 | [43.16, 49.95] | 109/0/0 | **YES** | 31.80 | 15,717 |
| 0 | `cont_mlp` | 0.028 | 0.046 | +0.018 | **47.495** | 47.450 | [44.06, 50.97] | 109/0/0 | **YES** | 8.51 | 15,717 |
| 1 | `mlp` | 3.239 | 2.890 | −0.349 | **47.404** | 44.514 | [41.01, 48.03] | 109/0/0 | **YES** | 15.02 | 15,717 |
| 1 | `lstm` | 4.633 | 4.826 | +0.193 | **47.404** | 42.578 | [39.50, 45.70] | 109/0/0 | **YES** | 14.85 | 15,717 |
| 1 | `transformer` | 0.651 | 0.284 | −0.367 | **47.404** | 47.119 | [43.79, 50.50] | 109/0/0 | **YES** | 40.95 | 15,717 |
| 1 | `cont_mlp` | 0.028 | 0.037 | +0.009 | **47.404** | 47.367 | [44.02, 50.75] | 109/0/0 | **YES** | 13.72 | 15,717 |
| 2 | `mlp` | 8.174 | 7.284 | −0.890 | **49.119** | 41.835 | [38.71, 44.96] | 109/0/0 | **YES** | 5.38 | 15,717 |
| 2 | `lstm` | 9.404 | 7.982 | −1.422 | **49.119** | 41.138 | [38.16, 44.13] | 109/0/0 | **YES** | 11.11 | 15,717 |
| 2 | `transformer` | 3.073 | 2.211 | −0.862 | **49.119** | 46.908 | [43.64, 50.19] | 109/0/0 | **YES** | 23.09 | 15,717 |
| 2 | `cont_mlp` | 0.055 | 0.055 | 0.000 | **49.119** | 49.064 | [45.72, 52.48] | 109/0/0 | **YES** | 5.26 | 15,717 |

### 7.3 `item` scope — a random 1 % (109 survivor corners)

| seed | family | rival as-trained | rival retrain-without | rival after − before | ours after deletion | paired ours − rival | CI95 | won/tied/lost | beats 1.0? | rival retrain s | rival rows read |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `mlp` | 5.193 | 5.092 | −0.101 | **47.495** | 42.404 | [39.25, 45.63] | 109/0/0 | **YES** | 10.28 | 16,220 |
| 0 | `lstm` | 4.761 | 4.147 | −0.615 | **47.495** | 43.349 | [40.21, 46.55] | 109/0/0 | **YES** | 22.25 | 16,220 |
| 0 | `transformer` | 1.229 | 1.128 | −0.101 | **47.495** | 46.367 | [43.00, 49.83] | 109/0/0 | **YES** | 39.66 | 16,220 |
| 0 | `cont_mlp` | 0.028 | 0.028 | 0.000 | **47.495** | 47.468 | [44.07, 50.99] | 109/0/0 | **YES** | 10.62 | 16,220 |
| 1 | `mlp` | 3.239 | 3.211 | −0.028 | **47.404** | 44.193 | [40.64, 47.74] | 109/0/0 | **YES** | 6.75 | 16,220 |
| 1 | `lstm` | 4.633 | 4.560 | −0.073 | **47.404** | 42.844 | [39.79, 45.93] | 109/0/0 | **YES** | 12.42 | 16,220 |
| 1 | `transformer` | 0.651 | 0.560 | −0.092 | **47.404** | 46.844 | [43.49, 50.24] | 109/0/0 | **YES** | 24.00 | 16,220 |
| 1 | `cont_mlp` | 0.028 | 0.028 | 0.000 | **47.404** | 47.376 | [44.03, 50.76] | 109/0/0 | **YES** | 6.45 | 16,220 |
| 2 | `mlp` | 8.174 | 7.789 | −0.385 | **49.119** | 41.330 | [38.15, 44.54] | 109/0/0 | **YES** | 6.06 | 16,220 |
| 2 | `lstm` | 9.404 | 9.312 | −0.092 | **49.119** | 39.807 | [36.84, 42.74] | 109/0/0 | **YES** | 10.77 | 16,220 |
| 2 | `transformer` | 3.073 | 3.037 | −0.037 | **49.119** | 46.083 | [42.83, 49.33] | 109/0/0 | **YES** | 21.97 | 16,220 |
| 2 | `cont_mlp` | 0.055 | 0.055 | 0.000 | **49.119** | 49.064 | [45.72, 52.48] | 109/0/0 | **YES** | 5.76 | 16,220 |

### 7.4 Reading the rival column honestly

- **The rival's own deletion cost is small in seconds** (5.3 – 41.0 s) **but it reads
  15,032 – 16,220 data rows** on every scope — and that is the whole retain set at the
  comparator's own **16,384-row** build scale. **Ours reads 0.** That is the comparison.
- ⚠ **The rivals are so weak in absolute terms that their deletion delta is barely
  informative.** They score **0.03 – 10.7** novel tuples per corner against the arm's
  **47.4 – 52.6**; a −1.5 change on a base of 5.5 cannot be read as "retrain-without
  damages composition" in any useful sense. The row is printed because the task file asks
  for it and because a rival that deletes only by retraining **pays 15k row-reads to do
  it**, which is the point — ⛔ not because the rival's composition number is a
  meaningful contrast at this strength.
- ⚠ **`cont_mlp` "improves" after retrain-without** (+0.009 to +0.045). At 0.03–0.11
  novel tuples per corner it is at the floor; this is noise, not a finding, and it is
  printed rather than dropped.
- ⛔ **The comparison is NOT 1×, and it is NOT scale-matched.** Both asymmetries run
  against this hub (§2). No row here is a 1× result.
- ⚠ **Item-scope caveat, declared before the run.** The arm's 1 % mask indexes the
  131,072-row partition pool (`w12_deletion_rows.py`, `default_rng(20_260_921 + seed)`);
  the comparator's indexes its own 16,384-row `fit_rows`
  (`w12_deletion_baselines.partition_fit_rows`, `default_rng(20260921)`). **Two different
  index spaces, so the two delete different random 1 % subsets at every seed including
  0.** w12's mask is kept byte-for-byte because this leg is a pure replication; the
  mismatch is printed rather than silently re-pinned.

---

## 8. WHAT DID **NOT** RUN — DECLARED, NEVER A ZERO AND NEVER A FAILURE

| item | status | reason |
|---|---|---|
| `single-design` runs **2/3 and 3/3** | **DECLARED NOT-NEEDED and NOT AVAILABLE** | Already spent by w12b hub/nr6-and-kstar at 2026-09-21T21:33:51Z on exactly this content; the target is exhausted at 3/3. This hub spent **0**. §1 |
| The **1× comparator column** | **DECLARED NOT-RUN** (carried, AMENDMENT 1 §A1.2) | `BASELINES-READY.json` pins `param_target = 131,072` (the 524,288 B REFERENCE column) and names no 1×-matched entry point. The error runs **against** this hub. |
| **Scale-matched** comparators | **DECLARED NOT-RUN** | The checkpoints on disk were fitted at `n_fit_rows_used = 16,384` against the store's 131,072. A scale-matched rival would be **stronger**. Printed on every rival row. |
| Rival retrain-without on the **`red ∧ cube`** scope | **DECLARED NOT-RUN, and cannot exist** | The scope deletes 0 rows; `w12_deletion_baselines.partition_fit_rows` refuses a no-op spec by construction ("a deletion that removes zero rows is not a deletion"). |
| The **refusal DISPLAY comparator** | **DECLARED NOT-RUN** (carried) | The only permitted comparator is the refuse-token transformer; wave 9 measured a refuser of that quality at +0.047 separation, ridge probe AUC 0.48–0.55 (chance). Not yet informative. |
| `unlearn_neggrad` / `unlearn_finetune` comparators | **DECLARED NOT-RUN** | They exist on disk for all three seeds but the task file's rule is *"rival deletes only by retrain-without"*. Scoring approximate-unlearning arms would have added 24 more 109-corner passes and is not what the row asks for. Available at `w12-deletion-baselines/comparators/seed_*/` for whoever wants them. |
| **Seeds 3 and 4** | **DECLARED NOT-RUN** (carried, AMENDMENT 1 §A1.1) | They pass the generation-2 cache rule but have **no decoder and no external classifier**, so `fw.Judge(seed)` cannot be built. ⛔ Never padded to 4 or 5 seeds. |
| A **re-scored** T7 row at seeds 1 and 2 | **NOT RE-RUN, BY DESIGN** | It already exists and is charged. It is **read**, and the reused arm is proved to reproduce it at `max per-corner deviation = 0.0`. §1 |

⛔ **Never "9/9 seeds."** The seed count is **3** and the configuration count is reported
separately: **1** configuration (`CovContentStore`, `ward_balanced` K\* = 32, A = 63) ×
**4** deletion scopes × **3** seeds.

---

## 9. WHAT WOULD HAVE PROVED THIS WRONG

1. **A control failing to collapse at seed 1 or 2** ⇒ no store row is claimed and that is
   the reported result. ⛔ **It did not happen:** all four read exactly **0.0000** at all
   three seeds, 109/109 corners, §3. ⚠ Note this is a *strong* form of collapse — the
   controls score literally zero, which is what a `judge_valid` that requires the whole
   six-factor tuple does to a degraded render. A reader should treat "0.0000" as "the
   control produces nothing the judge accepts", not as a graded margin.
2. **Store-scoped exactness failing** — `downdate` diverging from the refit-without store,
   or the ranks disagreeing. ⛔ Did not happen: max |mode deviation| **2.0e−08 – 5.4e−07**
   and `ranks_agree = True` on all 12 scope×seed legs, §5.
3. **The reused arm failing to reproduce the charged T7 row** ⇒ the deletion legs would
   not be measured against the published row. ⛔ Did not happen: **0.0** deviation at both
   seeds, §1.
4. **The deleted corners degrading after the value deletion** would have been evidence
   that deletion removes capability. ⚠ **The opposite happened, at all three seeds** — and
   that is reported as the finding, §6, not buried. It is the single most important
   limitation in this file.
5. **A seed where `landscape_removed` degrades towards the held-out shape at one seed and
   away from it at another** would make the sphere caveat un-boundable from seed 0 alone.
   ⚠ **This did happen** (−0.3291 at seed 0, +0.6709 at seeds 1 and 2), which is precisely
   why the caveat column is now measured per seed, §3.

---

## 10. ARTEFACT INDEX — every number above lives in one of these

| path | what |
|---|---|
| `C:\chlu\outputs\w15-single-design-seeds\T7-3SEEDS.json` | **the acceptance artefact**: T7 at n = 3, the byte ledger per seed, the control gate per seed, the deletion rows and the rival deletion column |
| `C:\chlu\outputs\w15-single-design-seeds\T7-3SEEDS.md` | the same numbers as rendered tables |
| `C:\chlu\outputs\w15-single-design-seeds\deletion\test_seed1_K32\deletion.json` | seed 1: arm deletion legs + rival column (**new**) |
| `C:\chlu\outputs\w15-single-design-seeds\deletion\test_seed2_K32\deletion.json` | seed 2: arm deletion legs + rival column (**new**) |
| `C:\chlu\outputs\w15-single-design-seeds\deletion\test_seed0_K32\rivals.json` | seed 0: rival column (**new**; arm legs carried from w12) |
| `C:\chlu\outputs\w15-single-design-seeds\LOG.md` | the running record, including every choice made where the brief left one open |
| `C:\chlu\outputs\w15-single-design-seeds\PIDS.txt` | my own PIDs, the only processes this hub stopped |
| `C:\chlu\outputs\w15-single-design-seeds\env.sh` | the env shim (path-format repair + the generation-2 cache root) |
| `C:\chlu\outputs\w15-single-design-seeds\logs\` | `del_s1.log`, `del_s2.log`, `riv_s0.log` |
| `C:\chlu\outputs\w15-single-design-seeds\w15-single-design-seeds.bundle` | the branch, whole |
| `C:\chlu\outputs\w12-single-design-crossover\scored\run1_seed0_K32\scored.json` | seed 0 T7 (**read, not re-run**) |
| `C:\chlu\outputs\w12b-nr6-and-kstar\scored\run3_seed{1,2}_K32\scored.json` | seeds 1 and 2 T7 (**read, not re-run**) |
| `C:\chlu\outputs\w12-single-design-crossover\deletion\test_seed0_K32\deletion.json` | seed 0 arm deletion legs (**read, not re-run**) |
| `C:\chlu\outputs\w12-deletion-baselines\BASELINES-READY.json` | every comparator's cost line, pins and build scale |
| `/c/chlu/SCORED-LEDGER.md` | the NOTE appended 2026-09-22T21:31:47Z: **0 runs spent** |
