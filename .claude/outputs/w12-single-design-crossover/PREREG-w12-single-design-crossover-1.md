# PREREG — w12 hub/single-design-crossover — SCORED RUN 1 of 3 (`single-design`)

**Written 2026-09-21T11:17:57Z, BEFORE any held-out (test-condition) number this wave.**
Charter §A.1. Amendments are appended and dated, never edited (§A.3).
Base `base/iter8 = aa91785`. Worktree `C:/chlu/worktrees/CHLU-w12-single-design-crossover`,
branch `w12/single-design-crossover`. `chlu.__file__` asserted inside the worktree.
Local CPU, 3 threads, 3 processes. Cache **generation 2**, asserted per seed by
`exp_gen_delete_row7.assert_generation2`.

---

## 0. THE GATE — mechanical, and this run does not start without it

⛔ This run does not start until `/c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json`
exists and names each comparator's entry point, matched-parameter count and cost fields.
⛔ If the gate never opens, **the scored leg is a DECLARED NOT-RUN with that reason** and no
`single-design` run is spent. A declared NOT-RUN is never a null and never a zero.

**Seeds.** 3 seeds (0, 1, 2) — the seeds with a generation-2 **full** 480,000-row feature
cache on this box. ⭐ 5 seeds **IF** `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json`
exists **at the moment this scored leg starts**; otherwise **3 seeds, DECLARED**.
⛔ 3-seed and 5-seed rows are never mixed in one plot without a printed `n`.

## 1. ⭐⭐ THE PIN CHANGED. THE SINGLE DESIGN IS 1x.

⛔ **This is NOT wave 8's experiment and it must not be read as one.** w8 answered its own
question at the 524,288 B pin: there the covariance forces K ≥ 36, and composition dies above
K ≈ 5, so at **that pin** the two properties are incompatible. That result stands and is not
re-run.

⭐ **This wave changes the pin, not the object.** The object is `CovContentStore`
(`chlu/experiments/exp_w8_single_design.py:293`) at **rank 64 per well, capped STRICTLY below
the well width** — `A = min(64, min_k d_k - 1)`. Its **own four-part total** is the byte
budget, and the learned baselines are matched to **that** total (**1x**). The **524,288 B**
figure is printed as a **REFERENCE COLUMN** on every row and is **not a budget this wave**.

⛔ **THE FOUR-PART LEDGER IS PRINTED ON EVERY ROW AND IS NEVER NETTED:**

| part | what it is |
|---|---|
| 1. atoms | `sum_k A x d_k` float32, the deployed content |
| 2. covariance + count | the packed float64 second-moment triangle, first moment and count that make deletion exact |
| 3. address book | the fitted `TensorAddress` parameters |
| 4. coupling / hole table | the request counts that carry the refusal holes |

## 2. K\* — SELECTED ON `val` ONLY, BY A RULE FROZEN BEFORE THE SWEEP RAN

`SELECTION-RULE.md`, written **2026-09-21T11:10:24Z** (timestamp in
`SELECTION-RULE.timestamp.txt`), before any val composition number for any K existed on disk:

> **K\* = max { K in {5,8,10,12,16,20,24,28,32} : v(5) − mean_seeds v(K) ≤ 0.02 }**

`v(K)` is the mean over the 24 val corners of the per-corner `valid` rate, produced by the
`scripts/w8_score_corners.py` code path. K\* is read from `CROSSOVER-val.json`, which
`scripts/w12_crossover_report.py` writes by applying that rule and nothing else.

⛔ **If no K > 5 satisfies the rule, K\* = 5 and this prereg says so** — the rule is not
relaxed, the tolerance is not moved, the sweep is not re-run with a different metric
(charter A.3).

## 3. ⛔⛔ THE CONTROL GATE — no row may be claimed until it passes

On **every scored row**, and printed on every scored row:

| control | what it does | required |
|---|---|---|
| `landscape_shuffled` | atom VALUES permuted inside each well (the shipped C.2 definition, `exp_gen_read.perturb_store`) | **MUST collapse** |
| `landscape_shuffled_across_wells` | mode blocks permuted ACROSS wells (the task file's literal wording; blocks cropped/zero-padded to the destination width, recorded on the row) | **MUST collapse** |
| `random_atoms` | atoms replaced by a random Gaussian block of the same shape and Frobenius norm, address book NOT refit | **MUST collapse** |
| `landscape_removed` | atoms zeroed | **MUST collapse to floor** |

⛔ **If `landscape_shuffled` or `random_atoms` does not collapse at K\*, the store is
DECORATIVE at that K and NO ROW MAY BE CLAIMED for it.** That is then the wave's reported
result, stated in those words.

⚠ **This is wave 9's saturation trap and it is why `rank/width` is printed everywhere.** At
`rank/d_k = 0.938` on CMAZE-2, any full-rank atom block spanned the same subspace, so the
content carried nothing and the address book did all the work; `landscape_shuffled` read
0.9396/0.9725/0.9405 against the CLU's 1.0000 and the arm was caught **only** because those
two controls were run (`w9-advisor-report.md` §1.3).
⚠ A control must be read through a basis of **its own** atoms
(`w12_crossover_lib.qr_basis`) — `pinned_basis` assumes orthonormal direction rows, which a
shuffled or random block is not, and reusing it would silently discard the perturbation and
make the control prove nothing.

## 4. THE ROWS, each labelled RESULT or DISPLAY

### RESULT 1 — rows 3 and 4, no deletion, at K\*

Instrument: `exp_gen_factor_wells` paired-corner estimand, test condition. Row 3 = 60 tiled
`pair-*` sub-corners, row 4 = 49 Montero `nway-*` corners. Metric = the same per-corner
`valid` rate the val read uses, reported with its per-corner n.

Comparators, **all matched to the single design's own four-part total (1x)**, with the
524,288 B matching printed as a reference column: **MLP · LSTM · transformer · continuous
MLP**, via the entry points named in `BASELINES-READY.json`. **The prototype table is printed
as a FLOOR.**

**ABLATIONS, printed as ablations and never as rivals:**
- **composition-only store** — `pca_tensor` / `PCASpanStore`: *composes, cannot forget.*
- **the wave-4 item store** — *forgets, loses composition.*

⚠ **Declared falsification:** if `cov_tensor` at K\* loses rows 3–4 to the composition-only
store beyond seed spread, **the two properties remain incompatible at this pin too, and that
is this hub's answer.** It is reported as the result, not buried.

⚠ **Declared val/test check:** if composition at K\* is **not** within 0.02 of K=5 on TEST
though it was on val, **the val/test gap IS the finding** and is reported as such.

### RESULT 2 — `generation-table` run 3 (the last one on that target)

Instrument: `exp_gen_cell.py`, the same object, against the full rival set including the
table. Metrics: `external composition_B aggregate` (headline) and `in_corner` balanced
accuracy with its mandatory AUROC companion, each with its `distinct factor_tuples` count —
⛔ a rate is never reported without its count. ⛔ This is the cell instrument, a single
72-request corner; it is **not** quoted against a row-3/4 verdict.

### RESULT 3 — DELETION, three scopes, same seeds

**(a) VALUE.** Delete every fit row carrying **red** (`object_hue` index 0,
`exp_gen_factor_wells.py:1348`) ⇒ score green/blue composition on rows 3–4 over the
survivor corners. Comparators, **always both**: baselines **retrained-without-red** (cost
printed) **and** baselines **as-trained**.

**(b) CONJUNCTION.** Delete every fit row carrying **red ∧ cube** ⇒ score green/blue cubes
and red spheres/cylinders on rows 3–4. Same two comparators.

**(c) ITEM.** Delete a random **1 %** of fit rows ⇒ composition unchanged vs
**refit-without**, with exactness printed.

⭐ **EVERY row prints adaptation cost for EVERY arm: seconds · bytes rewritten · data rows
read (0 for us).**

⛔ **LANGUAGE, the Head's ruling, binding on every sentence this hub writes:** never *"no
changes after deletion"*. The sentence is: **post-deletion adaptation is LIGHT and DATA-FREE**
— re-eigendecompose the ledger, carry the address book, **no row of data read** — against
baselines that can only delete by **retraining from scratch without the deleted rows**.

⛔ **"EXACT" IS DEFINED, everywhere the word appears (MF4):** **STORE-SCOPED exactness** —
`downdate(delete rows)` equals the store fitted from scratch without those rows, at the same
partition and the same rank; ⭐ **the frozen encoder and the frozen decoder are NOT refit.**

⛔ **Never "9/9 seeds."** The **seed count** and the **configuration count** are reported
separately, always.

### DISPLAY A — post-deletion refusal (⛔ reported, never claimed)

Red requests after (a); red-cube requests after (b). Our refusal via **the hole** vs every
baseline's **render rate**. ⛔ The only permitted comparator is the **refuse-token
transformer** from `spoke/deletion-baselines`.
⚠ Carried honestly: wave 9's refuser separated at **+0.047** and a ridge probe on the same
observation scored **AUC 0.48–0.55, i.e. chance**, so a DISPLAY comparison against a refuser
of that quality is **not yet informative** and the row says so.

## 5. ⭐ TWO EXACT MECHANISMS, ON ONE OBJECT, REPORTED SEPARATELY

(i) **data forgetting = covariance/count DOWNDATE.**
(ii) **refusal after deletion = the request count / coupling table WITH HOLES.**

⭐⭐ **A compositional store can still compose "red cube" from "red" and "cube" after the
red-cube rows are gone. THAT IS COMPOSITION WORKING, NOT DELETION FAILING** — the refusal row
is what the hole is for. ⛔ Both are reported, separately, on the same object.

## 6. WHAT WOULD PROVE THIS HUB WRONG — declared here, in advance

1. **No K satisfies the declared rule** ⇒ report that; ⛔ do not move the rule (charter A.3).
2. **The §3 controls do not collapse at K\*** ⇒ the store is **decorative** there and **no row
   is claimed** — that is the wave's result.
3. **Composition at K\* is not within 0.02 of K=5 on TEST** though it was on val ⇒ the
   **val/test gap is the finding**.

## 7. BUDGET AND LEDGER

- `single-design`: **3 runs**. `generation-table`: **1 run** (run 3, the last).
- ⛔ **The ledger row in `/c/chlu/SCORED-LEDGER.md` is written BEFORE each scored run** — this
  hub's OWN row. Another hub's row does not authorise this one.
- All runs spent are reported. ⛔ The best is never reported alone. Every quoted number
  carries "scored n/3" and the file path it came from.

---

# AMENDMENT 1 — 2026-09-21T12:29Z, appended, never an edit (charter §A.3)

⛔ Written BEFORE any test-condition number. **Scored 0/3 at the time of writing.**

## A1.1 THE SEED COUNT IS **3**, and the binding reason is the JUDGE, not the cache

§0 said "5 seeds IF `CACHE-READY.json` exists at the moment this scored leg starts".
That file **now exists** (`produced_utc 2026-09-21T12:27:01Z`). It does **not** licence 5
seeds, and the two reasons are separate measured facts:

1. **Seed 4 fails the admission rule.** Its entry has `array_shape: null`, `n_blocks: 0`,
   an empty `block_sha256`, `status: "NOT_RUN"` and
   `reason: "generation-2 cache for this seed was not completed"`. A seed is admitted only
   with a **non-null `array_shape` and 59 blocks**. ⛔ Seed 4 is a DECLARED NOT-RUN, not a zero.
2. **Seed 3 passes the admission rule and is still unusable here.** Its entry is
   `array_shape [480000, 128, 4, 4]`, `n_blocks 59`, 59 block hashes, `scope "full"`,
   `generation 2`, `shape_dtype_matches_seeds_0_2 true`, same `rows_sha256` as seeds 0–2.
   ⛔ **But every composition number in this wave is produced by `fw.Judge(seed)`, which
   loads the hash-asserted external classifier from `fw.classifier_dir(seed)` — and that
   directory does not exist for seed 3.** Measured: `classifier_dir(0|1|2)` exist,
   `classifier_dir(3)` and `classifier_dir(4)` do **not**. Seed 3's own entry also reads
   `status: "CACHE_READY_BRIDGE_PENDING"`.

⇒ **This hub uses 3 seeds: 0, 1, 2.** The count is printed on every row.
⛔ 3-seed and 4-seed rows are never mixed in one plot or one table.

## A1.2 The 1× baseline column is a DECLARED NOT-RUN

`BASELINES-READY.json` (`generated_utc 2026-09-21T12:02:45Z`) pins
`param_target: 131072` — i.e. **524,288 B / 4**. It names **no** 1×-matched entry point.
Refitting four families at this object's own four-part total was not affordable beside the
sweep. ⇒ The available comparator set is the **524,288 B REFERENCE column**.
⛔ **The direction of this error runs against this hub:** a 1×-matched rival would carry
several times these parameters and would be **stronger**, so the available comparison
**flatters** the arm. No row measured against it is reported as a 1× result.

## A1.3 The conjunction pin is adopted from the spoke, and the specified one is vacuous

`red ∧ cube` deletes **0** rows (`shape = cube` is held out of the split entirely). It is
run and reported with its count of 0. The non-vacuous substitute is **`red ∧ sphere`**,
`shape` index 2 — the index **adopted from** `BASELINES-READY.json`'s
`pins.conjunction_deletion = [{object_hue, 0}, {shape, 2}]`, so that the comparator and
this arm delete the same rows.

## A1.4 A defect of this hub's own harness, fixed before any scored number

`qr_basis` returned `min(d, 1 + deployed_rank)` basis columns while `pinned_basis` returns
`deployed_rank` (+1 when the mean's perpendicular part clears its 1e-4 admission
threshold). Where those disagree — the narrow wells of a fine partition — the control was
a **shape error**, and it crashed the val sweep at K=32 and K=28. `qr_basis` now takes the
book's own widths and slices a complete QR. K=5/8/10/20/24 completed before the fix, which
means their widths agreed, and where the width agrees the leading columns of a complete QR
**are** the reduced QR — so those rows are unaffected and were not re-run.
