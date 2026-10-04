# PREREG — hub/read — scored run 1 of 3
**Target:** `read` · **Hub:** `hub/read` · **Branch/worktree:** `/c/chlu/worktrees/CHLU-w4-read`
**Base:** `base/iter4 = 9dfed5e` · **Pin:** charter §A.4 `4125fce`
**Written:** **2026-09-19T18:11:01Z** — ⛔ **BEFORE the first number on the 49 n-way corners, the 60 pair
sub-corners or any of the four query sets.**
**Status at writing:** 0/3 scored runs spent. Selection complete on `val` and on nothing else.

---

## 0. What moves, and what does not

**DIAL: the READ** — the map from a request to the coefficients that address the atoms.

⛔ Held fixed and not touched by this hub: the encoder · the frozen feature cache · the fit pool
(`pool_rng_seed = 20260919`, `n_pool = 131072`) · the partition (`factor_aligned_balanced`, wave 3's
arm) · the mode count (64) · the **524,288 B store ledger** · the frozen decoder · the
hash-asserted external classifier · every corner, the acceptance rule, the validity rule, the
per-corner scores and the paired estimand.

⭐ The row machinery is not reimplemented. `exp_gen_read` **imports** `exp_gen_factor_wells` and
replaces exactly two functions inside it — `build_arm` and `arm_features` — under a context manager
that restores them and raises if anything else moved. Every row on this hub is therefore computed
by the code that produced wave 3's numbers.

---

## 1. THE ARM

`tensor_addr` — **the tensor-product address book**. The address of a request is the elementwise
**product** of one per-factor address vector per factor:

```
a(request) = prod_f ( w_f(v_f)^T E_f )        in R^R
c_k        = a(request) @ W_k                 in R^{r_k}
x_hat[w_k] = c_k @ U_k^T
```

* `w_f(v)` is `gen_request_head.one_hot_interpolated` — an exact one-hot on a grid value and the
  interpolation of the two bracketing one-hots between them, **the same interface rule the
  incumbent uses**. The address is therefore continuous in every factor: this is the "continuous
  per-well coordinate" the target file asks for, and it is what row 2 would need.
* `U_k` is an orthonormal basis of `span(G_k)` from the SVD of the **deployed atoms**. It carries
  **zero bytes** (recomputed from the store on load) and `span(U_k) == span(G_k)`, so this read and
  the incumbent address exactly the same content.
* Fitted by closed-form alternating least squares. No optimiser, no learning rate; one declared
  seed for the initialisation only. Labels enter ONCE, as the factor indices of the fit rows — the
  address book's own words (charter §B.6, w4-common §5). ⭐ **Declared in every table caption.**

**Pinned configuration (`READ_PINS`), fixed by selection on `val` and not moved again:**

| pin | value |
|---|---|
| `rank` R | **128** — pinned by a rule written before the number existed: R=256 must beat R=128 by > +1.0 novel tuples/corner on `val` or the smaller book wins. It did not (31.167 vs 32.083). |
| `ridge` | **1e-2** |
| `sweeps` | **40** — ⚠ a declared BUDGET, not convergence: val mse is still falling ~0.4 %/sweep at the cap, identically for every arm and seed, so every number for this arm is a LOWER BOUND on its mechanism |
| `address_seed` | 0 |
| `n_addr_train` | 131,072 — **the same pool and the same row count as the incumbent head's gram** |
| `standardise_targets` | `False` (measured: per-coordinate standardisation costs the hues) |
| `partition` | `balanced` (`factor_aligned_balanced`) |

**Byte ledger, printed on every row:** store `524,288 B` (unchanged, asserted) · address book
**195,584 B** (48,896 parameters) · incumbent head `1,766,400 B`. ⭐ The address book is reported BESIDE the store's
bytes and never netted into them — the rule `gen_request_head.head_bytes` states for the incumbent.

---

## 2. THE COMPARISON

### 2.1 Rivals (w4-common §1) — a win here is a RESULT
`table` (prototype table, 118,784 params) · `mlp` · **`mlp_180k`** (178,952 params — spoke A's
rival at THIS hub's own total, added because no arm here can be parameter-matched to the
118,784 target: the pinned store alone is 131,072 parameters) · `transformer` · `lstm` ·
`cont_mlp` · `poe_ebm` · `lace_repaired`, all from spoke A at matched parameters ±10 % on **cache generation 2**,
seeds 0/1/2, entered through `chlu.eval.rivals.w4_api.load_rival(seed, name)` and run through **my**
corners by **my** judge, so every row is the same path.
⛔ A rival that spoke A declares `RivalRefusal` gets a **dash and its footnote**, never a null and
never a win for us.

### 2.2 Ablations — a win here is an ABLATION and is labelled so in every table
`span_head` (the SAME store read by the incumbent ridge head — the like-for-like for this dial) ·
`corr_linear` (the correlation-split store + linear head, w4-common §1's like-for-like on every row) ·
`tensor_addr_sum` (identical rank, fit, ridge, seed and bytes, the address SUMMED instead of
multiplied — isolates the product) · `pvm_span` and `pvm_table` (the per-value-mean read the task
file names first, projected into the span and as the plain full-D table).

### 2.3 Controls, on every arm and every row
`landscape_removed` (atoms zeroed, ledger unchanged — must collapse) · `landscape_shuffled` (the
atom VALUES permuted inside each well's mode matrix after the address book was fitted — must
collapse) · `random_span_refit` (atoms replaced by a random subspace of the same shape and
Frobenius norm, address book REFIT — **stronger than charter §C.2 asks**: if the arm survives this,
the answer was never in the store) · `span_rotated` (a random orthogonal rotation inside each well;
the span is mathematically unchanged, so this is a **PREDICTED NO-OP** and is reported as the
statement of what this read uses) · `lookup_argmax` / `lookup_convex` (charter §C.1/§C.4, the
matched-bytes lookup over the same atoms driven by the same address).

### 2.4 The ceiling — ⛔ DECLARED NOT AN ARM
`oracle_span`: the true feature of the requested row projected onto `span(G_k)`. It reads a feature,
so charter §B.3 disqualifies it as a mechanism. It is printed on every row because it is the exact
ceiling of this dial and it says when the next gain must come from a different dial.

---

## 3. THE ESTIMAND — declared before the first held-out number

* **Unit:** the corner, paired. ⛔ Never a single cell.
* **Row 4:** the **49** `nway-*` corners (72 requests each). **Row 3:** the **60** `pair-*`
  sub-corners (36 each). Both are 1.0000 retained-values-only (measured), i.e. every value is one
  the store holds and only the conjunction is new.
* **Metric:** `distinct_valid_novel_tuples` per corner, with `judge_accuracy` and
  `distinct_valid_tuples` reported beside it.
* **Statistic:** the mean paired difference, with a two-sided sign test and a 10,000-draw paired
  bootstrap 95 % interval.
* **Seeds:** 0, 1 and 2 on **cache generation 2**, reported per seed and pooled. ⛔ No
  generation-1 number is quoted beside any of them.

### 3.1 THE MARGINS — declared in advance, never moved after a result is seen
| claim | comparison | declared margin |
|---|---|---|
| **row 4, primary** | `tensor_addr` − each rival, mean paired Δ novel tuples/corner | **> +1.0**, and the bootstrap 2.5 % quantile also **> +1.0** |
| **row 3, non-inferiority** | `tensor_addr` − each rival, same metric | **> −1.0** (a loss larger than this is a LOSS and is reported as one) |
| row 4, ablation leg | `tensor_addr` − `span_head` | > +1.0 — ⚠ an ABLATION, not a result |
| row 1 | reported, not claimed | — |
| floor, charter §C.1 | `tensor_addr` − `lookup_argmax` / `lookup_convex` / `table` | > +1.0, else the store is decorative and is reported as such |
| control, charter §C.2 | `landscape_removed`, `landscape_shuffled` | must fall below `tensor_addr` by > +1.0 |

### 3.2 EXPECTED DIRECTION, stated before the run
`tensor_addr` > `span_head` on rows 3 and 4; `tensor_addr` > `pvm_span` and > `pvm_table` on every
row; `landscape_removed` ≈ 0; `span_rotated` = `tensor_addr` to numerical noise.

---

## 4. ⛔ WHAT WOULD FALSIFY THIS HUB — from the target file, restated as a decision rule
1. `tensor_addr` **ties the full rival set on row 4** ⇒ the read is decoration; reported in those words.
2. `tensor_addr` beats `span_head` but **loses to the MLP or the transformer** ⇒ ⛔ NOT A RESULT;
   labelled an ablation.
3. **The per-value-mean table alone matches the arm** ⇒ the store is not earning its bytes; said so.
4. **The gain vanishes on generation-2 seeds 1–2** ⇒ it was a seed-0 artefact.
5. **Row 3 regresses while row 4 improves** ⇒ the trade is reported; ⛔ row 4 is never quoted alone.
6. `random_span_refit` ties `tensor_addr` ⇒ the answer is in the address book, not the store.

---

## 5. Selection evidence (on `val` only, 24 corners, before this file was written)
24 val corners (retained values, held-out conjunction, rows removed from the fit pool),
1,728 requests, THE frozen decoder + THE hash-asserted external classifier, seed 0,
generation-1 cache (DEV). Metric: `distinct_valid_novel_tuples` per corner.

| row | class | novel/corner ± sd | judge acc | scale | orientation | address B |
|---|---|---|---|---|---|---|
| `oracle_span` | ⛔ NOT AN ARM (the dial's ceiling) | 34.542 ± 11.275 | 0.4797 | 0.844 | 0.550 | — |
| **`tensor_addr` R=128** | **ARM** | **32.083 ± 11.717** | **0.4456** | **0.751** | 0.573 | **195,584** |
| `tensor_addr` R=256 | capacity variant, NOT pinned | 31.167 ± 11.032 | 0.4329 | 0.795 | 0.532 | 388,608 |
| `span_head` | ABLATION (the deployed read) | 25.458 ± 14.987 | 0.3536 | 0.576 | 0.578 | 1,766,400 |
| `tensor_addr` R=64 | capacity variant, NOT pinned | 22.375 ± 8.622 | 0.3108 | 0.674 | 0.463 | 99,072 |
| `corr_linear` | ABLATION | 20.167 ± 14.141 | 0.2801 | 0.532 | 0.519 | 1,766,400 |
| RIVAL `mlp` | RIVAL | 19.792 ± 15.472 | 0.2749 | 0.453 | 0.650 | — |
| RIVAL `table` | RIVAL | 6.000 ± 8.424 | 0.0833 | 0.105 | **0.909** | — |
| RIVAL `cont_mlp` | RIVAL | 3.875 ± 6.174 | 0.0538 | 0.255 | 0.594 | — |

Paired estimands on those 24 corners (margin +1.0, sign test, 10,000-draw bootstrap):
`tensor_addr` − `cont_mlp` **+28.208** [+21.792, +33.875] p=3.6e-05 · − `table` **+26.083**
[+22.042, +30.125] p=1.2e-07 · − `mlp` **+12.292** [+6.875, +17.708] p=0.0106 ·
− `corr_linear` +11.917 · − `span_head` **+6.625** [+2.292, +11.167] p=0.0931 ·
− `oracle_span` −2.458 (below its own ceiling, as it must be).

**What the selection says, and it is the claim this run tests:** four of the six factors are
already at 1.000 for every arm on this population; ⛔ `orientation` has NO headroom in this
dial (the oracle reads it at 0.550, the deployed read already at 0.578 — the atoms do not
carry it, and the per-value-mean table reads it at 0.909 without them); ⭐ **the entire
recoverable headroom of the read is `scale`, 0.576 → 0.844, and the product address
recovers 65 % of it.**

⚠ Carried with every rival number above: the val corners sit in the value-holdout TRAIN
index and spoke A's rivals are fitted from that same index, so a rival may have seen a val
corner's rows. That is the generous direction for the rivals and the conservative one for
this arm.

---

## 6. Provenance
* Cache: **generation 2**, `cache_gen2_root` from `RIVALS-READY.json`, sha256s recorded in the leg's
  `pins.json`.
* Stores refit per seed by `exp_gen_factor_wells.build_arm` at the pinned recipe; the
  `store_bytes == 524_288` assertion must pass or the leg aborts.
* `assert_read_request_only` and `assert_head_request_only` run on every arm of every row.
* Command: `scripts/hub_read_leg.sh test <seed> /c/chlu/outputs/hub-read/run` with
  `CHLU_GEN_BRIDGE_ROOT_OVERRIDE` set to the generation-2 root; the launcher REFUSES a `test` leg
  without it.

---

## 7. AMENDMENT 1 — appended 2026-09-19T18:16Z, ⛔ before any held-out number exists

⚠ **The rank comparison that pinned R=128 is UNDERPOWERED, and that is stated here rather
than after a result.** Raised by spoke A: R=256 lost to R=128 by **0.916** novel
tuples/corner (31.167 vs 32.083) against a +1.0 threshold, on **24** val corners of 72
requests each. That difference is inside the spread one would expect from seed noise alone
— spoke A measures their own rivals moving ±0.03 balanced accuracy across seeds on 72-item
sets.

**What does NOT change:** the pin stays R=128. The rule was written before the number
existed, the number did not clear it, and the smaller address book wins ties by that rule.
⛔ The pin is not revisited on this run.

**What is declared now, so it cannot be claimed later:** if R=256 should later look better
on the 49 scored corners, ⛔ **that is not evidence the rule failed and it is not a result
for R=256.** It is evidence that a 24-corner val comparison could not separate two ranks
0.9 apart. Any future rank claim needs its own PREREG and its own scored budget.

## 8. AMENDMENT 2 — appended 2026-09-19T18:16Z, same timestamp, same reason

⚠ **A rival ranking must be read per factor, not on an aggregate.** Spoke A measured, on
their own corner over three seeds, that the prototype table reads `orientation` at 0.995
and `scale` at 0.204 while the matched MLP reads `orientation` at 0.644 and `scale` at
0.741 — the MLP wins the aggregate (0.894 vs 0.865) purely by winning the factor the other
loses. My own selection numbers have the same shape on a different population with a
different corner set (table `orientation` 0.909 / `scale` 0.105; my arm `scale` 0.751).
⛔ The two measurements are NOT combined. But the agreement means this report will print
**per-factor accuracy beside every aggregate on every row**, and will say in the caption
that an aggregate over six factors is a mixture and not a ranking.
