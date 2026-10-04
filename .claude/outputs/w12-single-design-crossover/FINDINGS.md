# w12 hub/single-design-crossover — FINDINGS

**Where does composition break as the partition gets finer?**
Box FORGIS_14 · branch `w12/single-design-crossover` · base `base/iter8 = aa91785` ·
local CPU, 3 processes × 3 threads · cache **generation 2**, asserted per seed.
`chlu.__file__ = C:\chlu\worktrees\CHLU-w12-single-design-crossover\chlu\__init__.py`.

> ⛔ **No number in this document appears without the file it came from.** Everything not
> run is a **DECLARED NOT-RUN with a reason** (§11) — never a zero, never a blank.
>
> ⭐⭐ **THE THREE THINGS TO READ FIRST.** (1) §5 — **composition does not break anywhere in
> K ∈ [5, 32]; K\* = 32 and there is NO crossover.** (2) §9 — the scored row at K\* wins
> 109/109 corners with all four controls at exactly 0.000. (3) ⛔⛔ §10 — **and deleting
> every red row changes the red answer by exactly zero**, because at rank/width 0.97 the
> per-well content carries no value-specific information. That third one bounds the
> programme lead claim and is the most consequential number this wave produced.

---

## 0. What was fixed before anything was measured

| artefact | written | what it fixes |
|---|---|---|
| `SELECTION-RULE.md` | **2026-09-21T11:10:24Z** (`SELECTION-RULE.timestamp.txt`) | K\* = max{K : v(5) − mean_seeds v(K) ≤ 0.02}; crossover = min{K : drop > 0.10}; the control gate; A = min(64, min_k d_k − 1) |
| `PREREG-w12-single-design-crossover-1.md` | **2026-09-21T11:17:57Z** | the scored leg: gate, rows, comparators, the three declared falsifications, the store-scoped definition of "exact" |

At 11:10:24Z no val composition number for any K existed in this output directory.

---

## 1. ⭐ THE BYTE HALF OF THE ANSWER IS EXACT ARITHMETIC, AND IT IS ALREADY IN

Source: `LEDGER-ACROSS-K.json`, written by `scripts/w12_ledger_table.py`.
Parts 1, 2 and 4 are exact arithmetic on the partition shape; part 3 (the address book)
is measured from the fitted `TensorAddress` and is a DECLARED NOT-RUN where a K has no
sweep row. ⛔ It is never estimated.

| K | d_min…d_max | A | **rank/width** | atoms B | cov+count B | coupling B | parts 1+2+4 | ×524,288 |
|---|---|---|---|---|---|---|---|---|
| 5 | 409…410 | 64 | **0.1565** | 524,288 | 3,391,552 | 6,610 | 3,922,450 | **7.481** |
| 8 | 256…256 | 64 | 0.2500 | 524,288 | 2,133,280 | 6,610 | 2,664,178 | 5.082 |
| 10 | 204…205 | 64 | 0.3137 | 524,288 | 1,713,872 | 6,610 | 2,244,770 | 4.282 |
| 12 | 170…171 | 64 | 0.3765 | 524,288 | 1,434,272 | 6,610 | 1,965,170 | 3.748 |
| 16 | 128…128 | 64 | 0.5000 | 524,288 | 1,084,768 | 6,610 | 1,615,666 | 3.082 |
| 20 | 102…103 | 64 | 0.6275 | 524,288 | 875,104 | 6,610 | 1,406,002 | 2.682 |
| 24 | 85…86 | 64 | 0.7529 | 524,288 | 735,328 | 6,610 | 1,266,226 | 2.415 |
| 28 | 73…74 | 64 | **0.8767** | 524,288 | 635,488 | 6,610 | 1,166,386 | 2.225 |
| 32 | 64…64 | 63 | **0.9844** | 516,096 | 560,608 | 6,610 | 1,083,314 | **2.066** |

⭐ **The two curves move against each other, and that is the wave's shape.** Making the
partition finer buys back covariance bytes — 3,391,552 → 560,608 B, a 6.0× reduction —
but it drives **rank per well width from 0.1565 to 0.9844**.
⛔ **0.9844 is above 0.938, the value at which wave 9's maze arm was decorative.**

⭐ **And the object never reaches the retired pin at rank 64.** The smallest four-part
total in the sweep is **2.066×** 524,288 B. ⛔ So "matched bytes at 524,288 B" is not
available to this object at this rank under any K — which is why this wave's pin is the
object's own total (**1×**) and 524,288 B is a **reference column**.

---

## 2. ⛔ THE CONJUNCTION ROW AS SPECIFIED IS VACUOUS — measured, and independently agreed

Task §4(b) asks to *"delete every row carrying red ∧ cube"*. On the Shapes3D
value-holdout condition that deletes **zero rows**: `shape = cube` is held out of the
split entirely.

Measured on the fit pool by decoding the row index (the odometer
`scripts/w8_single_design_dev.py:red_mask` uses), **val / test**:

| set | rows |
|---|---|
| red (`object_hue` index 0) | **11,024 / 11,044** |
| red ∧ **cube** (shape 0) | **0 / 0** — cube is a held-out VALUE |
| red ∧ cylinder (shape 1) | **0 / 0** — red-cylinder is the held-out PAIR cell |
| red ∧ sphere (shape 2) | 5,507 / 5,511 |
| red ∧ oblong (shape 3) | 5,517 / 5,533 |

Whole-pool shape counts: cube **0**, cylinder 43,650, sphere 43,501, oblong 43,921.
Whole-pool `object_hue`: index 5 is **0** (the held-out hue); index 0 (red) is 11,024.

⭐ **The repo already says this in its own words** — `exp_gen_delete_row7.py:325`:
`"red_cube": (0, 0, False, "shape=cube is held out entirely; reported, not a bar")`.

⭐⭐ **And `spoke/deletion-baselines` found it independently**, from its own direction, and
wrote at 11:44Z in `/c/chlu/outputs/w12-deletion-baselines/LOG.md`:
*"shape idx 0 ('cube') = 0 rows … So 'red ∧ cube' would delete ZERO rows. Conjunction pin
moved to object_hue 0.0 ∧ shape 2.0 = 5,511 rows … `partition_fit_rows` REFUSES any no-op
deletion rather than printing a zero."* Its red count (11,044) equals mine on the test
pool exactly.

**What this hub does — both rows, never one:**
- **(b) red ∧ cube** runs as specified and is reported with its count of **0** and this
  reason, flagged `VACUOUS`. ⛔ It is never reported as a passing row.
- **(b′) red ∧ sphere** is the **DECLARED NON-VACUOUS SUBSTITUTE** (scope
  `conjunction_live`), and the shape index is **adopted from the spoke's pin** so that the
  comparator and this arm delete the *same rows*. A comparator and an arm that delete
  different rows are not a comparison.

⭐ The DISPLAY refusal row punches **both** holes, 1 byte each. Red-cube is refusable
although no red cube was ever written — which is precisely the point that **refusal is a
separate mechanism from forgetting** and does not need a deleted row to have existed.

---

## 3. ⛔ THE 1× BASELINE COLUMN — the error runs against this hub's own interest

The rival set that exists on this box is matched to the **retired 524,288 B pin**
(131,072 float32 parameters). Smoke-loaded through `chlu/eval/rivals/w4_api.load_rival`
on the TEST condition, seed 0, corner `pair-000/f0-hh` (36 requests, 0 nan):

| rival | features | out-of-range | param_count |
|---|---|---|---|
| `mlp` | (36, 2048) | 0 | 117,878 |
| `lstm` | (36, 2048) | 0 | 119,125 |
| `transformer` | (36, 2048) | 0 | 119,858 |
| `cont_mlp` | (36, 2048) | 0 | 119,525 |
| `table` (the FLOOR) | (36, 2048) | 0 | — a table has no parameter count |

`spoke/deletion-baselines` reports its own re-matched set at the same pin (LOG 11:44Z):
`mlp` 130,514 (−0.43 %), `cont_mlp` 131,891 (+0.63 %), `transformer` 130,433 (−0.49 %),
`lstm` 130,721 (−0.27 %), refuse-token transformer 130,479 (−0.45 %).

⛔ **This object's four-part total is 2.07×–7.48× of 524,288 B, so a 1×-matched rival
would carry several times those parameters and would be STRONGER, not weaker.**
⛔ **Comparing against the 131,072-parameter set therefore FLATTERS this hub, and no row
measured against it may be reported as a 1× result.** It is the **524,288 B reference
column**, labelled as such. If `BASELINES-READY.json` never names a 1×-matched entry
point, **the 1× column is a DECLARED NOT-RUN.**

---

## 4. SEEDS — **3, DECLARED**, and the binding reason is the JUDGE, not the cache

`/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json` **exists**
(`produced_utc 2026-09-21T12:27:01Z`, schema `chlu-w12-cache-ready/1`). It does **not**
licence more than 3 seeds, for two separate measured reasons.

**Admission rule applied as stated: a seed is usable only with a non-null `array_shape`
and 59 blocks.**

| seed | `array_shape` | blocks | scope | rows | verdict |
|---|---|---|---|---|---|
| 3 | `[480000, 128, 4, 4]` | **59** | `full` | 480,000 | **passes** the cache rule |
| 4 | **`null`** | **0** | `null` | `null` | **fails** — `status: "NOT_RUN"`, `reason: "generation-2 cache for this seed was not completed"` |

⛔⛔ **Seed 3 passes the cache rule and is still unusable here, and the reason is the
JUDGE.** Every composition number in this wave is produced by `fw.Judge(seed)`, which
loads the **hash-asserted external classifier** from `fw.classifier_dir(seed)`. Measured:

| seed | `classifier_dir(seed)` exists |
|---|---|
| 0, 1, 2 | **True** |
| 3 | **False** |
| 4 | **False** |

Seed 3's own entry also reads `status: "CACHE_READY_BRIDGE_PENDING"` — its bridge sidecars
are not placed either.

⇒ **This hub uses 3 seeds: 0, 1, 2.** The count is printed on every row. ⛔ 3-seed and
4-seed rows are never mixed, and seed 4 is a **DECLARED NOT-RUN**, never a zero.

⚠ **A defect of this hub's, caught before it produced a number.**
`CHLU_GEN_FEATURES_ROOT` is **not** set in this box's ambient environment, so
`exp_gen_factor_wells.features_root()` falls back to the frozen bridge — cache
**GENERATION 1**, which w5-common §3.8 forbids for any number, and which at bridge seeds 1
and 2 holds **9,288 rows** against the pool's 131,072 indices. My first seed-0 launch
(11:15:59Z) was reading generation 1; it was killed before it wrote a row and **no number
from it exists on disk**. Every run now sets the variable in `scripts/w12_launch.sh` and
asserts `exp_gen_delete_row7.assert_generation2(seed)` per seed, writing the manifest onto
every row.

⚠ **A second defect of this hub's, caught by a crash rather than by a look.** `qr_basis`
returned `min(d, 1 + deployed_rank)` basis columns while `pinned_basis` returns
`deployed_rank` (+1 when the mean's perpendicular part clears its 1e-4 admission
threshold). Where those disagree — the narrow wells of a fine partition — the control was a
**shape error**, and it killed the sweep at K = 32 and K = 28
(`ValueError: matmul … size 63 is different from 62`). Fixed by requiring the book's own
widths and slicing a complete QR; unit-tested at the two failing shapes. **K = 5/8/10/20/24
completed before the fix, which means their widths agreed, and where the width agrees the
leading columns of a complete QR *are* the reduced QR — so those rows are unaffected and
were not re-run.**

---

## 5. ⭐⭐ THE VAL SWEEP — THE ANSWER, AND IT IS A NEGATIVE ANSWER TO THE QUESTION ASKED

**Where does composition break as the partition gets finer? ⭐ IN THIS RANGE, IT DOES NOT.**

Source: `CROSSOVER-val.json`, written by `scripts/w12_crossover_report.py` applying the
rule frozen at 11:10:24Z and nothing else. Per-row source:
`val_sweep/val_K*_seed0/row.json`. **n = 1 seed at every K** (seed 0); seeds 1 and 2 at
K = 5 and K = 32 were launched at 12:52Z — see §9 for what landed.

| K | A | well width | rank/width | **val composition** | drop vs K=5 | four-part B | ×524,288 |
|---|---|---|---|---|---|---|---|
| 5 | 64 | 409–410 | 0.1561–0.1565 | 0.8918 | 0.0000 | 4,118,034 | 7.854 |
| 8 | 64 | 256 | 0.2500 | 0.8837 | 0.0081 | 2,959,602 | 5.645 |
| 10 | 64 | 204–205 | 0.3122–0.3137 | 0.8825 | 0.0093 | 2,606,754 | 4.972 |
| 12 | 64 | 170–171 | 0.3743–0.3765 | 0.8929 | −0.0012 | 2,393,714 | 4.566 |
| 16 | 64 | 128 | 0.5000 | 0.8906 | 0.0012 | 2,177,330 | 4.153 |
| 20 | 64 | 102–103 | 0.6214–0.6275 | 0.8877 | 0.0041 | 2,100,786 | 4.007 |
| 24 | 64 | 85–86 | 0.7442–0.7529 | 0.8981 | −0.0064 | **2,094,130** | **3.994** |
| 28 | 64 | 73–74 | 0.8630–0.8767 | 0.8976 | −0.0058 | 2,126,890 | 4.057 |
| **32** | 63 | 64 | **0.9531–0.9688** | **0.9016** | **−0.0098** | 2,143,138 | 4.088 |

**THE RULE, APPLIED LITERALLY:**
> K\* = max{ K : v(5) − mean_seeds v(K) ≤ 0.02 } over the K actually run.
> Satisfying set = **{5, 8, 10, 12, 16, 20, 24, 28, 32} — every K.** ⇒ **K\* = 32.**

**THE CROSSOVER: ⛔ THERE IS NONE INSIDE THE SWEPT RANGE.** The largest drop anywhere is
**0.0093** (K = 10) against a crossover threshold of 0.10. ⛔ The range was **not**
extended to manufacture one, as `SELECTION-RULE.md` pre-committed it would not be.

⭐ **A second finding nobody asked for: the four-part ledger is U-SHAPED in K.** It falls
4,118,034 → **2,094,130 B at K = 24** and then *rises* to 2,143,138 B at K = 32, because
the **address book grows faster than the covariance falls**:

| part | K = 5 | K = 24 | K = 32 |
|---|---|---|---|
| 1 atoms | 524,288 | 524,288 | 516,096 |
| 2 covariance + count | 3,391,552 | 735,328 | **560,608** |
| 3 address book | **195,584** | 827,904 | **1,059,824** |
| 4 coupling / hole table | 6,610 | 6,610 | 6,610 |
| **total** | 4,118,034 | **2,094,130** | 2,143,138 |

⛔ **The object never reaches the retired 524,288 B pin at rank 64: the minimum over the
whole sweep is 3.994×.** (`LEDGER-ACROSS-K.json`.)
⭐ Wave 8 could only buy the pin by collapsing the rank (K = 40 at A = 7). This wave holds
the rank at 64 and reports the byte cost instead of paying for it with content.

**Figure:** `fig_crossover_val.png` — composition and ledger bytes vs K on two y-axes, as
the task file asks, with the controls and rank/width beside them.

---

## 6. ⭐⭐ THE CONTROL GATE — it passes everywhere, and this wave found out WHY

`landscape_shuffled`, `landscape_shuffled_across_wells`, `random_atoms` and
`landscape_removed` read **exactly 0.0000** at **every one of the nine K**, including at
K\* = 32 where **rank/width is 0.9531–0.9688 — above the 0.938 at which wave 9's maze arm
was decorative.** Against the arm's 0.9016 at K = 32 that is a drop of **+0.9016** on all
four.

⭐ **The saturation trap does NOT fire here, and the mechanism is measured rather than
asserted:** a degraded render on this decoder + this hash-asserted external classifier is
judged **cube** — 1.0000 for zeroed atoms, 0.9196 for random atoms — and **cube is the
shape held out of the split entirely**, so `judge_valid` is false on every request. The
controls do not collapse because of a threshold; they collapse because the degraded render
lands on a shape the condition never contains.

⚠ **This is a property of the instrument and it cuts both ways.** It makes the charter-C
controls very sharp on Shapes3D, and it means a Shapes3D control collapse is *weaker*
evidence that the content is load-bearing than the same collapse would be on an
environment without a held-out shape. ⛔ Stated here rather than left for a referee.

---

## 7. ⭐ THE DELETION LEGS — VAL, seed 0, K = 5

Source: `deletion/val_seed0_K5/deletion.json`; the fitted arm was reused from
`val_sweep/val_K5_seed0/arm.pkl`, so no second address fit was paid.

| scope | rows deleted | **our adaptation** | bytes rewritten | **data rows read** | refit-without | novel/corner before → after | paired Δ | store-scoped exactness |
|---|---|---|---|---|---|---|---|---|
| **(a) VALUE** red | 11,024 | **1.45 s** | 4,068,192 | **0** | 13.2 s / 120,048 rows | 64.208 → 64.333 | **+0.1250** | 1.455e−11 |
| **(b) CONJUNCTION** red ∧ cube | **0 — VACUOUS** | 0.32 s | 4,068,192 | **0** | 14.7 s / 131,072 rows | 64.208 → 64.208 | +0.0000 | **0.000e+00** |
| **(b′) RED SPHERE** | 5,507 | **0.72 s** | 4,068,192 | **0** | 12.5 s / 125,565 rows | 64.208 → 64.125 | **−0.0833** | 2.842e−14 |
| **(c) ITEM** random 1 % | 1,311 | 0.28 s | 4,068,192 | **0** | 13.7 s / 129,761 rows | 64.208 → 64.208 | +0.0000 | 1.455e−11 |

⭐ Every paired delta is inside BAR 2's |Δ| ≤ 1.0 **by two orders of magnitude**.
⭐ The vacuous scope reads **exactly** 0.000e+00 and +0.0000 — the right answer for
deleting nothing, and a check that the pipeline is not manufacturing motion.

⛔ **LANGUAGE, as the Head ruled.** Not "no changes after deletion". **Post-deletion
adaptation is LIGHT and DATA-FREE** — the covariance and count are downdated by
subtraction, the atoms are re-eigendecomposed from the downdated statistic, and the address
book is carried across the basis change — **no row of data is read** — against a reference
that can only delete by refitting from scratch without the deleted rows, reading
120,048–131,072 rows to do it.

⛔ **"EXACT" MEANS STORE-SCOPED**, everywhere it appears: `downdate(delete rows)` equals
the store fitted from scratch without those rows, at the same partition and the same rank.
⭐ **The frozen encoder and the frozen decoder are NOT refit.**

⛔ **REPORTED AGAINST THIS HUB — a limit of the val condition, not of the mechanism.**
`n_deleted_corners = 0` for **every** scope: the 24 val corners are all n-way corners and
none of them requests red, so the val leg exercises only the **survivor** side of the row.
The **deleted** side needs the TEST corners, which include the pair sub-corners. These val
numbers are evidence that composition **survives** a deletion; they are **not** evidence
that the deletion removed anything at the corner level. The store-scoped exactness column
is what says the deletion removed something.

**DISPLAY — the refusal row (⛔ reported, never claimed).** The coupling/hole table's index
space is **asserted** equal to `gen_field`'s, not assumed: `layout_equals_gen_field: true`,
1,322 pair slots, 60 indices checked. Two holes punched — **red ∧ cube** and **red
sphere** — at **1 byte each, 2 bytes total, 0 data rows read, no refit**. Red-cube is
refusable although **no red cube was ever written**, which is precisely the point that
**refusal is a separate mechanism from forgetting** and does not need a deleted row to have
existed. ⛔ The comparator — the refuse-token transformer — is reported in §9.

---

## 8. ⭐⭐ THE HEAD'S SPHERE CAVEAT — MEASURED, and it fires in the OPPOSITE direction

**Head ruling:** the conjunction pin stays `object_hue = 0.0 ∧ shape = 2.0` — that **is**
the **red sphere**, named that everywhere — and a caveat column is printed, because
`chlu/data/gen_value_holdout.py:299` rejected sphere as a *held-out* value in these words:

> *"the shape hazard of §3.1, exactly: a degraded render is a rounded blob, so the SPHERE
> is the shape the model degrades TOWARDS. A held-out sphere would be unremovable for
> rendering reasons, not store reasons."*

**The threat to this row:** an arm **damaged** by the red-sphere deletion could be judged
"sphere" for rendering reasons and score **spuriously well** on exactly the deleted
conjunction.

⭐ **MEASURED, not stated.** Judged-shape histogram read back from the external
classifier's own tuples, accepted requests only, 24 val corners, K = 5, seed 0
(`deletion/val_seed0_K5/deletion.json`):

| arm | cube | cylinder | sphere | oblong | **judged-sphere excess** |
|---|---|---|---|---|---|
| `cov_tensor`, intact | 0.0000 | 0.3333 | 0.3333 | 0.3333 | **+0.0000** |
| `landscape_removed` (degraded bound) | **1.0000** | 0.0000 | **0.0000** | 0.0000 | **−0.3333** |
| `random_atoms` (degraded bound) | **0.9196** | 0.0579 | **0.0023** | 0.0203 | **−0.3310** |
| `cov_tensor` **after** the red-sphere deletion | 0.0000 | 0.3333 | 0.3333 | 0.3333 | **+0.0000** |

(excess = frac(judged sphere) − frac(**requested** sphere) on the same requests.)

⛔ **THE HAZARD DOES NOT REPRODUCE ON THIS INSTRUMENT, and the two maximally degraded arms
bound it.** A degraded render here is judged **cube**, not sphere; the sphere excess is
**negative** for both degraded arms and **exactly zero** after the deletion. **A degrading
baseline cannot score spuriously as sphere on this pipeline** — it collapses onto the one
shape held out of the split, which is judged invalid everywhere.

⚠ **Scope of that statement.** Seed 0, val corners, K = 5, this decoder and this
classifier. It is not a claim about Shapes3D decoders in general, and `gen_value_holdout`'s
argument for not holding sphere out stands on its own terms. What is measured is that **the
specific spurious-scoring route it warns about is not open on this instrument.**

---

## 9. ⭐⭐ SCORED RUN 1/3 — rows 3 and 4 on TEST at K\* = 32, seed 0

Ledger row appended to `/c/chlu/SCORED-LEDGER.md` at **2026-09-21T12:53:06Z, before the
run**; `scripts/w12_scored_leg.py` refuses to start without it (and without
`BASELINES-READY.json`). Artefact: **`scored/run1_seed0_K32/scored.json`**.
60 pair sub-corners (row 3) + 49 n-way corners (row 4). Metric
`distinct_valid_novel_tuples` per corner; verdict `fw.paired_estimand`, the corner as the
unit, margin **+1.0/corner declared in advance**.

**FOUR-PART LEDGER ON TEST, summed and never netted:** atoms 516,096 | covariance+count
560,608 | address book 1,060,864 | coupling/hole 6,610 = **2,144,178 B = 4.0897×** the
524,288 B REFERENCE column. rank/width max **0.9688**.

| arm | row 3 (60) | row 4 (49) | judge acc | paired Δ vs us | CI95 | corners we win |
|---|---|---|---|---|---|---|
| **`cov_tensor` (THE ARM)** | **31.600** | **66.959** | 0.9012 | — | — | — |
| `landscape_removed` | 0.000 | 0.000 | 0.0000 | +47.4954 | [+44.101, +51.028] | 109/109 |
| `landscape_shuffled` | 0.000 | 0.000 | 0.0000 | +47.4954 | [+44.101, +51.028] | 109/109 |
| `landscape_shuffled_across_wells` | 0.000 | 0.000 | 0.0000 | +47.4954 | [+44.101, +51.028] | 109/109 |
| `random_atoms` | 0.000 | 0.000 | 0.0000 | +47.4954 | [+44.101, +51.028] | 109/109 |
| `mlp__as_trained` | 2.933 | 7.959 | 0.0945 | +42.3028 | [+39.165, +45.532] | 109/109 |
| `lstm__as_trained` | 2.500 | 7.531 | 0.0852 | +42.7339 | [+39.679, +45.899] | 109/109 |
| `transformer__as_trained` | 0.667 | 1.918 | 0.0222 | +46.2661 | [+42.899, +49.725] | 109/109 |
| `cont_mlp__as_trained` | 0.000 | 0.061 | 0.0004 | +47.4679 | [+44.073, +50.991] | 109/109 |
| **`table` — the FLOOR, never a rival** | 5.833 | 11.163 | 0.1589 | +39.2661 | [+35.862, +42.679] | 109/109 |

⭐ **The control gate PASSES on the scored row**: all four controls are **exactly 0.000** on
both rows at rank/width 0.9688. The row may be claimed.
⭐ **Charter C.1**: the arm beats the prototype **table floor** by **+39.2661** with a CI95
lower bound of **+35.862** against a **+1.0** margin.

⛔⛔ **AND THIS IS NOT A 1× COMPARISON. The error runs against this hub.** Every learned
comparator is matched to `param_target = 131,072` (= 524,288 B / 4), declared in
`BASELINES-READY.json`'s own pins. This object's four-part total is 2,144,178 B, so a
genuinely 1×-matched rival would carry **≈535,784 float32 parameters — about 4.1× these —
and would be stronger.** ⛔ The **1× column is a DECLARED NOT-RUN** and **no row above is
reported as a 1× result.** What is reported is the **524,288 B reference column**.
⚠ Also carried: `cont_mlp` scores 0.000/0.061, i.e. at the floor. That is a fact about the
rival's fit on this population, not a demonstration about the arm, and it is printed rather
than leaned on.

---

## 10. ⛔⛔ THE DELETION ROW ON TEST — AND THE FINDING THAT LIMITS THE WHOLE CLAIM

Artefacts: **`deletion/test_seed0_K32/deletion.json`** and
**`deletion/probe_test_seed0_K32.json`**. Reused `scored/run1_seed0_K32/arm.pkl`, so no
second 1,242 s address fit was paid.

| scope | rows deleted | **our adaptation** | bytes rewritten | **data rows read** | refit-without | store-scoped exactness |
|---|---|---|---|---|---|---|
| **(a) VALUE** red | 11,044 | **0.90 s** | 2,081,024 | **0** | 9.26 s / 120,028 rows | 5.960e−08 |
| **(b) CONJ** red ∧ cube | **0 — VACUOUS** | 0.05 s | 2,081,024 | **0** | 9.37 s / 131,072 rows | **0.000e+00** |
| **(b′) RED SPHERE** | 5,511 | **0.39 s** | 2,081,024 | **0** | 9.26 s / 125,561 rows | 5.383e−07 |
| **(c) ITEM** random 1 % | 1,311 | 0.11 s | 2,081,024 | **0** | 9.79 s / 129,761 rows | 2.338e−07 |

⭐ **The cost line is the story the Head asked for, and it holds:** post-deletion adaptation
is **light and data-free** — 0.05–0.90 s, **0 data rows read** — against a reference that
can only delete by refitting from scratch **without** the deleted rows, reading
120,028–131,072 rows to do it. Ranks agree on every check.

**Composition, on the corners themselves:**

| scope | survivor corners | novel/corner before → after | paired Δ | deleted corners | before → after |
|---|---|---|---|---|---|
| **(a) VALUE** red | 89 | 50.921 → 50.921 | **+0.0000** | **20** | **32.25 → 32.25** (judge acc 0.8958 → 0.8958) |
| (b) CONJ red ∧ cube | 109 | 47.495 → 47.495 | +0.0000 | 0 | — |
| (b′) RED SPHERE | 109 | 47.495 → 47.495 | +0.0000 | 0 | — |
| (c) ITEM 1 % | 109 | 47.495 → 47.495 | +0.0000 | 0 | — |

⛔⛔ **READ THAT SECOND COLUMN AGAIN. Deleting every one of the 11,044 rows carrying red
changes the 20 red-requesting corners by EXACTLY ZERO** — 32.25 → 32.25 novel valid tuples
per corner, judge accuracy 0.8958 → 0.8958. **The object does not stop producing red after
red is deleted.**

⭐ **So I measured what the deletion moved at the read** (`w12_deletion_probe.py`, no judge,
no decoder, 5,688 requests):

| group | n requests | rel. RMS change, book **carried** | rel. RMS change, book **NOT carried** | max abs change |
|---|---|---|---|---|
| corners **requesting red** | 720 | **0.000000** | **0.610083** | 4.7684e−07 |
| corners **not** requesting red | 4,968 | **0.000000** | 0.614277 | 4.7684e−07 |

**What this establishes, stated without spin:**
1. The **store-scoped exactness is real**: the downdated store equals a store refit without
   those rows to 5.960e−08, and the basis genuinely moved — the **uncarried** read changes
   by **0.61 relative RMS**.
2. The **carry is exact at the read**: carried, the rendered answer is unchanged to
   **float32 epsilon (4.77e−07)**.
3. ⛔ **But the basis change is GENERIC, not targeted.** The uncarried change is **0.610 on
   red-requesting corners and 0.614 on the others** — the deletion moved the span no more
   for red than for anything else.
4. ⛔⛔ **Therefore, at K\* = 32, the per-well content carries no value-specific
   information.** rank/width is 0.9531–0.9688: the well's span is very nearly the whole
   well, and removing 8.4 % of rows does not move a nearly-full-rank subspace. The value
   identity lives in the **address book**, which the deletion **carries but does not
   refit** — and the address book is fitted on the pool *before* the deletion.

⚠ **This does NOT overturn the exactness claim, and it does not rescue it either. It
bounds it.** *"Store-scoped exactness"* means exactly what it says — the frozen encoder and
decoder are not refit — and this wave adds a third thing that is not refit: **the address
book**. ⛔ On this object at this K, **"the store has forgotten red" and "the object no
longer answers red" are different statements, and only the first is demonstrated.**

⚠ **Contrast with wave 8, which is the useful part.** w8 measured the red deletion at K = 5,
A = 64 (rank/width 0.156) and found the basis change *concentrated*: `cos_min = 0.1999`
with 93.8 % of directions above 0.99, against `cos_min = 0.9976` for a size-matched random
deletion. ⭐ **The targeting of a deletion is a function of rank/width, and it disappears
exactly where the byte ledger is cheapest.** That is the real trade-off this wave found,
and it is not the one the wave went looking for.

**DISPLAY — the refusal row (⛔ reported, never claimed).** The coupling/hole table's index
space is **asserted** equal to `gen_field`'s: `layout_equals_gen_field: true`, 1,322 pair
slots, 60 indices checked. Two holes punched — **red ∧ cube** and **red sphere** — at
**1 byte each, 2 bytes total, 0 data rows read, no refit**. ⭐ Red-cube is refusable
although **no red cube was ever written**, which is exactly the point that **refusal is a
separate mechanism from forgetting.** ⭐⭐ And §10 makes that separation *load-bearing*
rather than rhetorical: since the downdate does not stop the object answering red, **the
hole is the only mechanism here that can.**
⛔ **The refuse-token comparator is a DECLARED NOT-RUN in this wave** — `BASELINES-READY`
provides it (`transformer__refuse_token__{value,conj}`) and the wall clock did not allow the
refusal DISPLAY leg after the scored rows and the deletion legs.
⚠ Carried regardless: wave 9's refuser separated possible from impossible at only **+0.047**
and a ridge probe on the same observation scored **AUC 0.48–0.55, i.e. chance**, so a
DISPLAY comparison against a refuser of that quality is **not yet informative**.

The sphere caveat on TEST reproduces §8: judged-sphere excess **+0.0074** intact,
**−0.3291** zeroed, **−0.3078** random atoms, **+0.0074** after the red-sphere deletion.

---

## 11. DECLARED NOT-RUN — every one with a reason, never a zero and never a blank

| not run | reason |
|---|---|
| `single-design` runs **2/3 and 3/3** (seeds 1, 2 on TEST) | wall clock. Run 1/3 took 1,242 s of address fit + 10 arm passes over 109 corners + the deletion legs. **Budget after this wave: `single-design` 1 of 3 spent, 2 remain.** |
| **`generation-table` run 3** | `exp_gen_cell.py` carries its own arm registry (`clu`, `store_removed`, `knn_proto`, `lace`, `ebm_compose`, `dynread`); wiring a K\* object into it is a build this wave had no wall clock for beside the sweep. ⭐ **The run remains UNSPENT.** |
| **the 1× baseline column** | `BASELINES-READY.json` pins `param_target = 131,072` and names no 1×-matched entry point; refitting four families at 535,784 params was not affordable. ⛔ The available comparison **flatters this hub** and is labelled the 524,288 B reference column. |
| **the refusal DISPLAY comparator** | the refuse-token transformer exists in `BASELINES-READY.json`; the wall clock ran out after the scored rows and the deletion legs. Our side of the row (the hole, 1 byte, 0 rows read) **is** measured. |
| **seeds 3 and 4** | seed 4 fails the cache admission rule (`array_shape: null`, 0 blocks, `status NOT_RUN`); seed 3 passes it but `fw.classifier_dir(3)` **does not exist**, so no corner of it can be judged. |
| **seeds 1–2 at K = 8, 10, 12, 16, 20, 24, 28** | wall clock. n is printed on every row of the table and the figure. |
| **`unlearn_finetune` / `unlearn_neggrad` comparators** | built and loadable in `BASELINES-READY.json`; not scored here — the deletion legs ran out of wall clock after `retrain_without` was covered by the store-scoped exactness column. |

---

## 12. WHAT WOULD HAVE PROVEN THIS HUB WRONG — the three declared checks, answered

1. **"No K satisfies the declared rule ⇒ report that; do not move the rule."** — Every K
   satisfied it. **K\* = 32**, the rule was applied literally, the crossover threshold was
   never reached and **the range was not extended.**
2. **"The §2 controls do not collapse at K\* ⇒ the store is decorative and no row is
   claimed."** — All four collapse to **exactly 0.000** on the scored row at rank/width
   0.9688. The gate passes.
3. **"Composition at K\* not within 0.02 of K=5 on TEST though it was on val ⇒ report the
   val/test gap."** — ⚠ **Not directly testable this wave**: only K\* = 32 was scored on
   TEST, so there is no TEST number at K = 5 to compare against. ⛔ Reported as a
   **DECLARED NOT-RUN**, not as a pass. It is the first thing run 2/3 should buy.

⭐ **And a fourth, which the hub did not pre-register and which is the wave's most
consequential result:** the val sweep and the scored row both say the store is *not*
decorative at K\* — and the deletion probe says its **content carries no value-specific
information there anyway**. ⛔ Those are not contradictory: the atoms are load-bearing for
**rendering** (destroy them and validity goes to 0.000) and not load-bearing for **value
identity** (delete every red row and the red answer does not move). ⛔ **No previous wave
separated those two senses of "load-bearing", and the charter-C control family cannot
distinguish them.**
