# PREREG — w8 hub/single-design — SCORED RUN 1 of 3

**Written 2026-09-20, BEFORE any held-out number.** Charter §A.1. Amendments are appended and dated, never edited (§A.3).
Base `base/iter7 = ec2ef7c`. Frozen harness untouched (§A.4). Generation **2** only. Local CPU, 3 threads.

## 0. The gate

⛔ This run does not start until `/c/chlu/outputs/w8-downdate/DOWNDATE-READY.json` exists and asserts its three
properties. If the gate never opens, this run is declared **NOT RUN** with its reason. ⛔ A declared NOT-RUN is never a null.

## 1. The object under test

**`cov_tensor`** — the single design. Identical to the incumbent `pca_tensor` in every respect except that the
per-well content is derived from a retained **sufficient statistic** instead of a one-shot SVD:

- per well `k`: `S_k = sum_i x_i x_i^T` (symmetric, triangle stored), `t_k = sum_i x_i`, `n_k`.
- deployed atoms = `[mu_k ; leading A-1 eigenvectors of (S_k - t_k t_k^T / n_k)/(n_k-1)]`, i.e. **the same content
  the SVD route produces**, to float precision.
- **delete item i** = `S_k -= x_i x_i^T`, `t_k -= x_i`, `n_k -= 1`, then re-derive. ⭐ No pass over the data, no refit,
  items never kept. That is the property `pca_tensor` declares it gives up (`exp_w5_compose_field.py:485`).
- partition: inherited label-MI criterion, re-cut to **K = 22** wells; atoms **A = 15**. Labels enter the partition and
  the address book ONCE at fit time and at the interface; ⭐ the CONTENT is unsupervised, exactly as the incumbent's is.
- read: unchanged — `TensorAddress` product address, R = 128, `assert_read_request_only` must pass.

**Declared ledger (charter §B.5), summed and never netted:**

| item | bytes | params |
|---|---|---|
| atoms `A x D` | 122,880 | 30,720 |
| covariance triangle `sum_k d_k(d_k+1)/2` | 385,400 | 96,350 |
| means `t_k` + counts `n_k` | 8,280 | 2,070 |
| **store, honest total** | **516,560** | **129,140** |
| address book (reported beside, as for every arm) | 200,784 | 50,196 |
| **arm total** | **717,344** | **179,336** |

Against the 524,288 B store pin: **0.985x — under budget.** Against `mlp_180k` and against `pca_tensor`'s own arm total
of 179,968 params: **−0.35%**, inside the declared `PARAM_TOLERANCE = 0.10`.

⚠ If the built object's realised ledger differs from this table, the realised table is printed and this one is marked superseded.

## 2. Selection — on `val` only (§A.2)

K and A are chosen on the **`val` corners** (`exp_gen_factor_wells.val_corners`, the separate selection condition carved
from unused n-way slots) and on nothing else. ⛔ The scored corners are not looked at before the run.

Candidate grid, all matched-bytes: `K in {20, 22, 24, 28, 32}` with `A` the largest that fits the budget.
⛔ `K >= 52` is excluded a priori, not by selection: there `A >= d_k`, the projection is the identity and the store is
decorative under charter C.1/C.4.

## 3. The rows, each labelled RESULT or DISPLAY

### RESULT 1 — rows 3 and 4, no deletion

Instrument: `exp_gen_factor_wells` paired-corner estimand. Unit = the corner. Metric = **distinct valid novel tuples**.
Row 3 = 60 tiled `pair-*` sub-corners; row 4 = 49 Montero `nway-*` corners. 3 seeds.

Comparators, all at matched parameters: **`mlp_180k`** (179,968 p), **`transformer`**, **`lstm`**, **`cont_mlp`**
(each bisected to its budget by `w4_learners.match_width`, ±10% asserted), and the **prototype table printed as a FLOOR**.
Incumbent control: **`pca_tensor`**.

**Bar:** mean paired per-corner delta over `mlp_180k` **> +1.0 novel tuple/corner** with the 2.5% bootstrap quantile also
above +1.0 (`DECORATION_MARGIN_NOVELTY`, the programme's standing margin, not re-derived here).

**Expected direction:** `cov_tensor` ≈ `pca_tensor` (same content to float precision, finer partition) and both > `mlp_180k`.

⚠ **Falsification, declared:** if `cov_tensor` loses rows 3-4 to `pca_tensor` beyond seed spread, **the two properties
remain incompatible and that is this hub's answer.** It is reported as the result, not buried.

### RESULT 2 — ⭐⭐ POST-DELETION COMPOSITION. The row this wave exists for.

Delete **red = `object_hue` index 0**, the value pinned in code at `exp_gen_factor_wells.py:1348`
(`DELETED_FACTOR, DELETED_VALUE_INDEX = 2, 0`) — pair-000's hue, so the survivors are the other two pair cells,
i.e. **green and blue**. Survivor set = 89 corners (49 n-way + pair-001 + pair-002); `exp_gen_delete_row7.py:266`
asserts >= 49 and aborts otherwise.

Score green/blue rows 3-4 on the survivors, paired, after the deletion, against **BOTH** comparators, **always both**:

- **(a) baselines RETRAINED WITHOUT RED**, at their own cost line — the honest comparator.
  `w4_deletion.run_deletion_comparator(ctx, factor="object_hue", value=0.0, families=[...])`, refit from scratch on the
  reduced pool, param-matched, cost line printed: wall-clock seconds, rows dropped, rows refitted.
- **(b) baselines LEFT AS TRAINED** — the same checkpoints, unchanged.

**The cost the CLU does not pay is printed on the row**, not in a footnote: the CLU's deletion is a rank-1 subtract on
`S_k`; the baselines' is a full refit (the existing instance of this comparator refits 29,099 rows at ~75 s per learner
per seed).

**Bar:** `abs(mean paired delta, after − before) <= 1.0` on the 89 survivors — the same BAR 2 the w5 eviction route
FAILED at −2.0899 / −0.9101 / −6.7416 (`exp_w6_retention.py:150`).

**Control, mandatory (the one that makes the row mean anything):** `evict_size_control` — the same number of items
removed at random from items NOT holding red, 8 draws. w5 measured **−2.7584** there and concluded "the cost is the
mechanism, not the target." ⭐ `cov_tensor` must beat its own size-matched control, or it has not deleted, it has been damaged.

**Expected direction:** delta ≈ 0 for `cov_tensor` (it downdates a statistic; it does not lose span), clearly negative for
an item-eviction route.

⚠ **Declared in advance:** if **(a) matches us post-deletion, our advantage is COST, not capability.** I will say exactly
that, in those words, and print both cost lines.

### RESULT 3 — generation-table run 3 (the last one)

Instrument: `exp_gen_cell.py`, the same object, against the full rival set **including the table**.
Metrics: `external composition_B aggregate` (headline) and `in_corner` balanced accuracy with its mandatory AUROC
companion, each with its `distinct factor_tuples` count (a rate is never reported without it).

⛔ This is the cell instrument, a single 72-request corner — it is **not** quoted against a row-3/4 verdict.

### DISPLAY A — post-deletion refusal (⛔ never quoted as a RESULT)

Our refusal rate on red requests after deletion vs the baselines' **red-render rate**.
⭐ The only comparator is a **transformer with a refuse token trained on impossible examples**; our line is
**"no impossible example was ever seen."** Scored as the refusal **pair** `(reject_invalid, accept_unseen_valid)`,
gmean, per `gen_scoreboard.score_refusal_row` — never as one number.

⚠ Carried honestly: an **8-byte presence+bracket rule ties the field store at (1.0000, 1.0000)** on rows 5 and 6
(`exp_gen_factor_wells.py:1035`, `:1049`). Symbolic refusal is closed; this row shows inductive bias, nothing more.

### DISPLAY B — the conjunction hole (⛔ never quoted as a RESULT)

w5 measured 1 byte, 0 outside, no refit, **0.0000 composition cost**, 101 other never-held conjunctions untouched.
Reproduced, not re-litigated, via `gen_field.punch_hole` + `assert_changed_bytes_within` + `assert_no_refit`.

## 4. Exactness of the deletion — asserted, with a negative control

1. `downdate == refit-without-those-rows` to float precision; **max abs deviation printed per well**.
2. **No pass over the data** during deletion, asserted mechanically.
3. **Items never kept** — the store holds `S_k, t_k, n_k` and atoms, never the rows.
4. `assert_changed_bytes_within` -> `bytes_changed_outside_permitted == 0`; `assert_no_refit` on the frozen segments.
5. ⭐ **NEGATIVE CONTROL, or the exactness claim is uninstrumented:** perturb one retained entry by 1e-3 and
   `assert_no_refit` MUST fire (the idiom at `exp_gen_delete_row7.py:993`). A check that cannot fail is not a check.

## 5. Charter §C floor checks, one line each, on every row

- C.1 matched-bytes lookup (`pca_lookup_argmax`, `pca_lookup_convex` on the same atoms) — must be beaten beyond seed spread.
- C.2 landscape-removed (atoms zeroed) and landscape-shuffled (values permuted) — must collapse.
  ⚠ Also `pca_random_refit`: at a saturated partition it would NOT collapse. `rank/d_k` is printed on every row so
  saturation is visible rather than hidden.
- C.3 tie clause — a tie with the lookup counts only with refusal / exact deletion / novel dreaming on the same artefacts.
- C.4 distance-to-nearest-atom printed; if it goes to 0 the arm is reported as a lookup table.

## 6. Reporting rules I bind myself to now

- All 3 scored runs reported; the best is never reported alone; every quoted number carries "scored n/3".
- ⚠ `lace/ld_corrected` moves more than its seed spread between generations (0.0177 vs 0.0151, 1 of 2 seeds; seed 1
  NOT RUN) — that caveat travels with any table quoting lace.
- Rivals are fitted on the `montero_combination` pool, which contains every factor VALUE; only COMBINATIONS are held
  out. Rows 3-4 are the rows they are built for (`row_eligibility.json`), so they are like-for-like there. On the
  post-deletion rows the population footnote is printed.
- A rival that cannot enter a row gets a dash and a footnote. ⛔ A dash is never a win for us.

---

# AMENDMENT 1 — 2026-09-20, appended, not edited (charter §A.3)

⛔ Written BEFORE any held-out number. No scored run has been spent. The gate
(`DOWNDATE-READY.json`) is `status: partial`, `condition: val`, so nothing above has been scored.

## A1.1 The object gains a declared step: the BASIS-CHANGE CARRY

**Why it is needed, measured.** The deletion changes the store's atoms — that is what makes it a
charter B.4 operation on atoms rather than an interface edit. But `gen_read.span_basis` re-derives the
per-well basis `U_k` from the atom block on load (it costs zero bytes), so an address book fitted before
the deletion is expressed in a basis that no longer exists. Measured, downdated store vs refit-without
store, identical coefficients through the repo's own `span_basis` + `render_span`:

| K | A | max abs rendered dev | projector dev |
|---|---|---|---|
| 38 | 7 | 2.137 | 1.52e-07 |
| 42 | 12 | 5.449 | 2.10e-07 |
| 48 | 18 | 3.854 | 2.13e-07 |

The subspace is identical and the read is not. ⛔ Scoring post-deletion composition without addressing
this would produce a collapse that has nothing to do with the deletion mechanism and everything to do
with a stale coordinate system — and it would look exactly like the w5 eviction failure.

**What is added.** At delete time, `R_k = U_k^T U'_k` is formed from the two bases already in hand and
the address book is carried across: `W_k <- W_k R_k`.

- `R_k` is `r x r`, touches **no pool row**, and is **transient** — applied and discarded.
- **Deployed byte cost: 0.** The ledger in §1 is unchanged.
- Measured effect: rendered downdate-vs-refit agreement goes from O(1) to **3.6e-07 / 9.5e-07 / 9.5e-07**
  against a signal scale of 3.2 / 3.8 / 4.3 — float32 precision.

## A1.2 A failed attempt, recorded rather than dropped

⛔ The first fix tried was orthogonal Procrustes on the ATOMS, rotating them to preserve the old basis.
It **failed**: rendered deviation 1.739 / 2.778 / 4.462, worse at K=48. Making the direction rows exactly
orthonormal makes the atom block's singular values degenerate, so `span_basis`'s SVD is arbitrary again.
It is recorded because it is the natural first idea and the reason it fails is a property of the interface.

## A1.3 A control that FAILED, and the claim it corrected

The audit first written for `R_k` asserted `R^T R = I`. It **failed**: 0.912 at K=42, 0.621 at K=48.
⭐ `R_k` is not a rotation and must not be: it holds the principal-angle cosines between the pre- and
post-deletion spans, and is an isometry only if the deletion changed nothing. "Gauge rotation" was the
wrong name; the step is a **basis-change carry** and the audit now REPORTS the spectrum instead of
asserting it away.

**The spectrum becomes a pre-registered DISPLAY row**, because it is a property of the store and not a
composition number:

| deletion | cos min | cos median | frac > 0.99 | frac < 0.5 |
|---|---|---|---|---|
| red (`object_hue = 0.0`, 11,044 rows) | 0.1999 | 0.9999 | 0.938 | 0.002 |
| size-matched RANDOM (11,044 non-red rows) | 0.9976 | 1.0000 | — | 0.000 |

**Expected direction, declared:** the red deletion moves a small specific set of directions and leaves the
rest alone; a size-matched random deletion moves nothing. ⛔ This is the store-level analogue of w5's
`evict_size_control` separation and it is **NOT** evidence about composition. It may not be quoted as a
RESULT, and it does not become a composition claim until the corners are scored.

## A1.4 Partition, corrected

⛔ §1's `K = 22, A = 15` is **WITHDRAWN**. It rested on well sizes `[303, 305, 573, 501, 366]` taken from
`tests/test_gen_dynamical_read.py:152`, a **test fixture**, not a deployed partition. The realised scored-leg
partition is `[777, 939, 216, 58, 58]`. Corrected consequences:

- honest ledger at K=5 with the required float64 statistic: **6,704,240 B = 12.787x** the pin (not 4.43x).
- `factor_aligned_balanced_labels` **refuses any K > 6**, so the inherited label-MI criterion cannot reach a
  matched-bytes partition at all.
- The criterion that can is the shipped **label-free** `ward_balanced` cut (`exp_gen_factor_wells.py:594`),
  under which the matched-bytes window is `K = 36..48`. ⭐ Being label-free makes the object *strictly more*
  label-free than the incumbent, not less.
- ⚠ A `ward_balanced` arm differs from the incumbent in **two** ways (criterion and granularity). The control
  that separates them is pre-registered here: **`ward_balanced` at K=5, A=64** — my criterion, the incumbent's
  granularity. ⛔ Without it the comparison is confounded and I will not report rows 3-4 from it.
- Candidate set for selection **on `val` only**: `K in {38, 42, 48}` with `A` the largest that fits and
  saturates no well. ⛔ `K < 36` is excluded arithmetically (the statistic alone exceeds the pin).
- ⛔ The param triangle in §1 (179,336 params, −0.35% vs `mlp_180k`) rested on the withdrawn partition and is
  **withdrawn with it**. It will be recomputed from the realised object and not re-quoted until then.

## A1.5 Also measured, and it is about the incumbent

`pca_tensor` is **already saturated in 2 of its 5 wells** (`d_k = 58 < A = 64`), and their effective ranks are
12 and 40 once `partitions.npz['dead']` is taken into account. So its realised deployed span is **308**, not
320, and `pca_random_refit` cannot collapse in those two wells. It is 5.7% of the atom bytes and does not
overturn the incumbent, but `rank/d_k` is printed on every row from here on.

---

# AMENDMENT 2 — 2026-09-20T20:30Z, appended, not edited (charter §A.3)

⛔ Written BEFORE any test-condition number. **Scored 0/3 at the time of writing.**

## A2.1 The object that would actually be scored — corrected, and it is NOT the matched-bytes one

⛔ Amendment 1's `cov_tensor` at `K=38 / A=7` was never built. The object that exists, is asserted, and
could be scored today is the **spoke's `DowndatableStore` at the incumbent's own partition**, fitted by me
on the **test** condition:

- `K = 5`, wells `[777, 939, 216, 58, 58]`, `A = 64`, ranks `[63, 63, 63, 13, 40]`, rank/d
  `[0.0811, 0.0671, 0.2917, 0.2241, 0.6897]`. ⚠ Wells 4 and 5 are `d_k = 58 < A = 64` — **saturated**,
  the incumbent's own property.
- **Ledger, summed and never netted:** store **524,288 B** (pin matches, tolerance 0) · covariance
  **12,310,672 B** (float64 `d_k^2`) · counts 11,488 B · coupling 5,288 + 1,322 B.
  **HONEST SUM 12,869,482 B = 24.5466x the pin.**
- ⛔ **Therefore rows 3-4 against `pca_tensor` are NOT matched-bytes, and will be reported as such.**
  My target file's declared outcome for this case is to print it rather than net it. That is what will happen.
  ⛔ It is NOT the claim Amendment 1 hoped for and it must not be presented as one.

## A2.2 The deletion procedure, fixed now so it cannot be chosen after seeing a number

1. Delete red = `object_hue` index 0 by rank-1 subtract on `S_k, t_k, n_k`. No data pass, no refit, items never kept.
2. Derive the basis with the spoke's **`pinned_span_basis`** (zero bytes), not `gen_read.span_basis`.
3. Carry the address book: **`W_k <- W_k M_k`, `M_k = U_k^T U'_k`**, computed at delete time. No data row read.
4. ⛔ **Print the carry's cost on the row: it rewrites 127,488 B of the address book's 158,664 B.**
   ⛔ It is **not** the coupling hole's 1 byte and is never quoted beside it as if it were.

## A2.3 What is already known on `val` and therefore is NOT what the scored run tests

⛔ These are the spoke's val numbers, not mine, not scored, and quoted here only so the scored run cannot
be presented as having discovered them: pre-deletion `downdate_store` = `pca_tensor` at **0.8941**, paired
**+0.0000** over 24 corners; post-deletion naive **0.0584**; re-indexed **0.0451**; **carried 0.8964**;
refit 0.9005.

**So the scored run's job is narrow and I state it in advance:** does the post-deletion, carried object hold
its composition on the **test** condition, on the **>=49 paired-corner estimand**, against **learned
baselines at matched parameters**, in **3 seeds** — none of which the val smoke establishes.

## A2.4 Expected directions, declared

- Rows 3-4 pre-deletion: `cov_tensor` ≈ `pca_tensor` within seed spread (val showed +0.0000).
- **Post-deletion, carried:** `|mean paired delta| <= 1.0` on the 89 survivors (BAR 2), i.e. composition intact.
- **If the carry is omitted, the row collapses** — and that collapse is interface staleness, not forgetting.
  ⛔ A collapsed row will not be reported as a deletion cost without the carried leg beside it.
- **Both baseline comparators always:** retrained-without-red at their own cost line, and left-as-trained.
- ⚠ **If the retrained-without baseline matches the carried object, our advantage is COST, not capability.**
  I will say exactly that and print both cost lines: ours = 0 data rows + 127,488 B; theirs = a full refit.

## A2.5 Charter §C, restated for THIS object

- C.1/C.4 floors and C.2 controls transfer unchanged (same atom block, same interface).
- ⚠ `pca_random_refit` **cannot collapse** in the two saturated wells; `rank/d_k` printed on every row.
- ⛔ The 24.5x ledger means this object cannot claim a matched-bytes win on rows 3-4 under any outcome.
