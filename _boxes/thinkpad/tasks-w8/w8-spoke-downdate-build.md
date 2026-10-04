# w8 spoke/downdate-build — THE SINGLE DESIGN. Build only. ⛔ No scored run.

- **Agent:** `experiment-engineer`. **Base `base/iter7 = ec2ef7c`.**
- **Worktree:** `/c/chlu/worktrees/CHLU-w8-downdate` · **Output:** `/c/chlu/outputs/w8-downdate/`
- ⭐⭐ **You are the critical path: `hub/single-design` cannot score until you publish §5.**
  Publish `DOWNDATE-READY.json` **partial** the moment the store fits and reads; do not wait.

## DIAL DECLARATION
**Dial: the object.** You build the store that is meant to compose *and* forget. **Falsified if** the
downdate does not equal the without-rows fit to float precision — then the design's whole premise fails
and the Head needs that **immediately**, not at the end.

## 1. Why this object exists — the problem it is built to solve
⭐ Composition is won by **`pca_tensor`**: per-well leading principal subspaces, unsupervised, 524,288 B.
⛔ But a principal direction **is not a written item**, so exact per-item deletion does not hold of it.
⛔ Meanwhile the object that *does* forget exactly — the coupling hole on the item store — **loses
composition when it evicts atoms.** ⭐⭐ **The single design is the attempt to have both.**

## 2. What to build
**The single-design store, one object, four parts:**
- **per-well principal subspaces** — fixed **64 directions**, **524,288 B** (the primary pin, tolerance 0);
- **per-well running covariance** — fixed `d_k²` per well, ⭐ **declared in the ledger BESIDE the store,
  never netted into it**;
- **a request count table**;
- **a coupling table with holes** (w5's mechanism: 1 byte, 0 outside, 0.0000 composition cost).

**`delete(rows)` = covariance downdate + re-eigendecomposition + count downdate.**
⭐⭐ **Three assertions, and they are the deliverable:**
1. **the downdated store EQUALS the store fitted WITHOUT those rows, to float precision** — assert it,
   print the max absolute deviation per well;
2. ⛔ **no pass over the data** — assert it mechanically, not in prose;
3. ⛔ **items are never kept** — no row is retained anywhere in the object.

## 3. Ledger discipline
Print the store bytes and the covariance bytes **separately**, both declared. ⛔ The covariance is not
free and must not be hidden inside the store's 524,288 B, nor netted against the address book.

## 4. Smoke, not claims
Show the object fits, reads, and renders through the shared frozen decoder and judge; rows 3–4 within
reach of `pca_tensor`'s spread on `val`. ⛔ **You claim nothing. `hub/single-design` scores it.**

## 5. The readiness file — a hub is blocked on this
`/c/chlu/outputs/w8-downdate/DOWNDATE-READY.json`:
```json
{"status":"ready|partial","store_bytes":524288,"cov_bytes":0,
 "downdate_equals_refit":{"max_abs_dev_per_well":[],"passed":false},
 "no_pass_over_data":false,"items_kept":false,"seeds":[0,1,2],"utc":""}
```

## 6. ⛔ What would prove this spoke wrong
- The downdate does **not** equal the without-rows fit ⇒ ⛔ **stop and report immediately.** The design
  has no claim without it; an approximate downdate is a different (and much weaker) object.
- Exact downdate requires keeping rows ⇒ the design collapses into the item store. Say so.
- The covariance cost pushes the honest ledger past the budget ⇒ **print it and say so**; ⛔ do not
  quietly net it against anything.
- Re-eigendecomposition after downdate is unaffordable at the shipped cell scale ⇒ report the cost.
