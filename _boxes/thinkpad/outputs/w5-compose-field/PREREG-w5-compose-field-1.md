# PREREG — w5 hub/compose-field — scored run 1 of 3 — target `composition`

**Written and committed BEFORE the first held-out number of this hub exists.**
Charter §A.1/§A.2. Base `base/iter5 = 6675904`. Pin `4125fce`. Worktree
`/c/chlu/worktrees/CHLU-w5-compose-field`. Log `/c/chlu/outputs/w5-compose-field/LOG.md`.

---

## 1. What is being decided

**Rows 3 and 4 of the generation table, against the FULL RIVAL SET.**

The dial is **WHAT IS STORED — the atoms themselves**. Not the read (wave 4 retired it against
the wave-3 store's ceiling), not the partition (w5-common §3.5 declares it exhausted; it is
inherited here unchanged).

### The arm

| | |
|---|---|
| name | `pca_tensor` |
| CONTENT | the per-well leading **principal subspace of the fit pool** — 64 atoms × 2048 float32, the pool mean as the first atom. **Unsupervised: no label enters the content.** |
| STORE BYTES | **524,288 B**, asserted by the store's own ledger (`matches_budget: true`) before the arm is scored. 131,072 parameters. |
| PARTITION | inherited, unchanged: the label-MI `factor` wells, K=5, sizes [771, 940, 217, 60, 60] |
| READ | the tensor-product address book, R=128, ridge 1e-2, 40 sweeps, fitted on the pool's span coefficients — **w4's own mechanism, unchanged** |
| ADDRESS BYTES | **191,424 B**, declared BESIDE the store and **never netted into it** (47,856 parameters) |
| REQUEST-ONLY | charter B.3, asserted on the deployed object per leg: `feature_argument_present: false`, `bitwise_stable_under_feature_perturbation: true` |

**Both budgets, both printed (w5-common §1).**
*Byte budget — PRIMARY:* store 524,288 B, matched to the pin exactly, tolerance 0.
*Parameter budget — SECONDARY:* 131,072 (store) + 47,856 (address book) = **178,928** against
`mlp_180k`'s **178,952**, i.e. **−0.013 %**. ⛔ Neither budget is ever quoted without naming which.

### The bar

⛔ **`mlp_180k` (178,952 parameters) is THE BAR.** A win over it is a RESULT.
⛔ The wave-3 aligned store (`span_head`) is an **ABLATION**; beating it is **not** a result and
will be labelled so on the table.
⛔ Wave 3's +11.612 was over `basis_head`. It is not cited here and nothing is "kept" from it.

---

## 2. The estimand, fixed here

* **Row 4** = the **49** n-way Montero corners. **Row 3** = the **60** pair-tiled sub-corners.
  Both ≥ 49 as w5-common §2 requires; neither row is decided by a single cell.
* **Unit** = the corner. **Statistic** = the **mean paired difference** between the arm and the
  comparator on the same corners. **Metric** = `distinct_valid_novel_tuples` per corner.
* ⚠ Row 3's per-corner tuple ceiling is **36** and row 4's is **72**. The two rows are **not** on
  one absolute scale and are never averaged together.
* **Seeds 0, 1, 2**, bridge seeds on **cache GENERATION 2**
  (`C:/chlu/outputs/spoke-a-rivals/cache_generation2`, `features_sha256` recorded in each leg's
  `pins.json`). ⛔ The launcher refuses `--condition test` without the generation-2 override.
  ⛔ No generation-1 number appears on this table in either direction.
* **Rivals are run through MY corners by MY judge**, via spoke A's own callable
  (`chlu.eval.rivals.w4_api.load_rival`), never as an imported aggregate. A rival that declines a
  request is a **DASH with its own words in the footnote** — ⛔ never a win for us.
* Labels are used ONCE at fit time to align the wells and at the address book, and nowhere else
  (charter B.6, w5-common §4). ⭐ The arm's CONTENT sees no label at all.

## 3. THE MARGINS, declared before the first held-out number

**BAR 1 — row 4, the RESULT.** `pca_tensor` − `rival_mlp_180k`, paired over the 49 n-way corners:
mean paired delta **≥ +1.0** novel tuple/corner **AND** the bootstrap 2.5 % quantile also **> +1.0**,
**on every one of seeds 0, 1, 2.**

**BAR 2 — row 3, the RESULT.** The same comparator, the same margin, the same three-seed
requirement, over the 60 pair sub-corners.

**BAR 3 — the rest of the rival set.** `pca_tensor` ≥ **+1.0** over each of `table`, `mlp`,
`cont_mlp`, `transformer`, `lstm` on rows 3 and 4, on every seed. (`lace` and `poe_ebm` are
expected to REFUSE; that is a dash and a footnote, not a win.)

**BAR 4 — §C.1/C.4, the floor.** `pca_tensor` beats **both** matched-bytes lookups over its OWN
atoms driven by its OWN address (`pca_lookup_argmax`, `pca_lookup_convex`) beyond seed spread.
⛔ If it ties either, the store is **decorative** and is reported as such.

**BAR 5 — §C.2, the controls.** `pca_landscape_removed` (atoms zeroed) and
`pca_landscape_shuffled` (atom VALUES permuted) must **collapse**. `pca_random_refit` (random
atoms at the same Frobenius norm, address book **REFIT**) must also collapse — ⛔ if it does not,
the answer lives in the address book and not in the store, and this hub reports that.

⛔ Charter §A.3: **no bar above is moved once a result has been seen against it.** Amendments are
appended, dated, never edited.

## 4. What I expect, including against myself

Declared in advance from `val` (seed 0, 24 val n-way corners, row 4 only, unscored, cache gen 2):

| leg | val r4 | class |
|---|---|---|
| `oracle_true_feature` | 71.250 | ⛔ NOT AN ARM |
| `oracle_content_fine` (K=32, same 524,288 B) | 71.250 | ⛔ NOT AN ARM |
| `oracle_content_oracle` | 71.042 | ⛔ NOT AN ARM |
| `oracle_content_pool` (the arm's own ceiling) | 68.083 | ⛔ NOT AN ARM |
| **`pca_tensor` — THE ARM** | **64.375** | ARM |
| `rival_mlp_180k` — THE BAR | 47.500 | RIVAL |
| `oracle_content_items` (64 real items) | 47.250 | ⛔ NOT AN ARM |
| `oracle_span_field` (the field store's atoms) | 39.125 | ⛔ NOT AN ARM |
| `oracle_span` (wave-3's fitted modes) | 36.375 | ⛔ NOT AN ARM |

* I expect BAR 1 and BAR 2 to **PASS**, by a margin far larger than +1.0. The val row-4 margin is
  +16.875 and the arm is already within 3.708 of its own content ceiling.
* ⚠ **Row 3 is the leg I am least sure of and I say so before I see it.** The selection condition
  emits **no** row-3 corners, so nothing about row 3 has been selected, tuned or even looked at.
  Its per-corner ceiling is half row 4's. If BAR 2 fails while BAR 1 passes, that is a real
  partial result and it will be reported as one, not smoothed over.
* ⚠ The val margin will **shrink** on the scored corners. `hub/placement` measured a val margin
  shrink of 8.6× going to the scored set, and `exp_gen_field_store.content_ceiling` records the
  val ceiling misleading badly on `orientation` (0.550 val vs 0.8804 scored). An 8.6× shrink of
  +16.875 is +1.96, which is **still above +1.0 but not comfortably**. ⛔ So BAR 1 is a genuine
  bar and not a formality.
* I expect `pca_random_refit` to **collapse**. ⚠ If it does not, this hub's own claim is dead and
  I will report the arm as an address-book effect.
* I expect `oracle_span_field` to remain **below** `mlp_180k` — i.e. the task file's literal gate,
  measured on the field store as inherited, reads CLOSED — while `oracle_content_pool` reads
  **OPEN**. ⭐ The interesting result is that both are true at once: the dial is open and the
  field store's atom selection is what was closing it.

## 5. What would falsify this hub

* `pca_tensor` loses `mlp_180k` on rows 3/4 ⇒ the content dial is closed after all. Report and stop.
* `pca_tensor` beats the wave-3 store but loses `mlp_180k` ⇒ an **ABLATION** win. Label it, claim nothing.
* A matched-bytes lookup over the same atoms matches the arm ⇒ §C.1, the store is a **lookup**.
* `pca_random_refit` does not collapse ⇒ the answer is the address book, not the content.
* The gain needs more than 524,288 B ⇒ no claim; the byte pin is primary and is asserted per leg.

## 6. Declared NOT-RUN in advance (⛔ never a null)

* `lace` and `poe_ebm` are sampler draws, not functions of a request; spoke A's callable refuses
  them. Dash + their own reason. ⚠ Byte multiples printed beside them where they appear: LACE
  **95.1×** the store budget, PoE-EBM **27.1×** (w5-common §3.9).
* `pca_span_rotated` (the predicted no-op) is run on `val` only and reported from there. It is a
  diagnostic of the read's invariance, not a bar.
* Every cluster (CSF3) leg: **NOT RUN**, 2FA-unreachable by any hub.
* GPU: **NOT USED**. This hub runs on the local CPU only.

## 7. Replication (charter §A.5)

The winning configuration is re-run from this file on a clean checkout of the branch. Seeds 1 and
2 are part of this same PREREG and are **free replications**, not additional scored runs
(w5-common §2). The branch is bundled whole so a replicator has the code and the pins.

---

## AMENDMENT 1 — 2026-09-20T00:22Z — appended, nothing above is edited

⛔ **No bar is moved.** BARS 1–5 and §2's estimand stand exactly as written. This amendment
corrects a **mechanism sentence** in §4 and adds one declaration.

1. **§4's clause "the field store's atom selection is what was closing it" is withdrawn as
   stated.** Measured after §4 was written and before any held-out number exists: 64 **random**
   pool rows have the **identical** span ranks to the field store's atoms — `[64, 64, 64, 13, 41]`
   — so the rank deficiency in the two 60-channel wells is a property of **real feature rows**
   (they are strongly collinear there), not of the field store's selection.
   ⭐ The corrected mechanism, and the one this hub's arm actually exploits: **items cannot cover
   a well; directions can.** Wells 3 and 4 admit 60 dimensions; any 64 real rows reach 13 and 41
   of them, while the pool's principal **directions** reach all 60 (variance explained 1.000).
   An orthonormal direction costs the same bytes as an item and can point along a low-variance
   axis no single item isolates.
   ⚠ Still unexplained and **not claimed**: the field store's atoms score 39.125 on val against
   random items' 47.250 **at equal rank**, so its selection is worse within an equal span budget
   for a reason this hub has not established and does not need.

2. **Declared in advance, so it cannot be read as a post-hoc excuse:** the two matched-bytes
   lookup floors (`pca_lookup_argmax`, `pca_lookup_convex`) select **one atom**, or a simplex
   mixture of atoms, and the arm's atoms are principal **directions** rather than items. A
   direction is not a plausible answer on its own, so these floors are expected to score very
   low. ⛔ That makes BAR 4 **weak evidence in this hub's favour and it is reported as weak**:
   the informative floor here is `oracle_content_items` (64 whole rows, 524,288 B, an oracle
   read) at val 47.250, which is the honest "could a lookup-shaped store have done this" figure,
   and `pca_random_refit` (random atoms, book refit), which is the control that can actually
   kill the arm. ⛔ A win over `pca_lookup_argmax` is **not** quoted as passing charter §C.1 on
   its own.
