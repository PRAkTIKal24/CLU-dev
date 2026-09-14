# gen-metric-invariance — METHOD (fixed before the numbers; the report quotes this)

## What was held fixed and what was varied
One `(configuration, seed)` **cell** = one frozen encoder's feature cache. Inside a cell:

1. `X_store` (32,768 × 2,048) is gathered **once** from that cache and never rebound.
2. The channel-correlation matrix `corr`, the dead-channel mask and the clustered reference grouping are
   computed **once** from `X_store[:16,384]`.
3. `K = 5` is loaded **verbatim** from `gen-store-grouping`'s `store_plan.json` (⛔ never re-derived).
4. Six partitions of the 2,048 coordinates into those 5 wells are built from that one `corr`:
   `correlation` · `random` · `random_alt` · `well_permuted` · `contiguous` · `balanced_random`.
5. For each partition: `init_store` (k-means++ on `X_store`) → `train_store` (1,200 Adam steps, store-only
   objective) → `address_codes(store, X_store)` → the axes.

⭐ **The features do not move. Only the partition does.** Measured, not asserted: `features_fixed.json`.

## The three axes, scored through one code path
| tag | definition |
|---|---|
| `dci_c` | `dci_scores(codes, labels, cards)["completeness"]` |
| `leak_raw` | `1 − BA(numerator)/BA(denominator)`, per well per factor, pooled over wells then factors (⛔ the shipped R23.2 spelling, copied from `gen-r0-wide-refit`'s harness) |
| `leak_cc` | identical, with `BA` replaced by `(BA − 1/c)/(1 − 1/c)` at each well × factor × set, `c` = classes **realised in that set** |

Numerator / denominator are each configuration's **own shipped sets**:
- **Montero:** numerator `eval_b (72 held-out) ∪ matched_seen (198)`, denominator `eval_a (1,024)`.
- **wide:** numerator `∪ over the 13 cells of (pos ∪ neg)` (≈3,100 rows), denominator `eval_a (1,024)`.

## The legs (tolerances pre-registered in `PREREG.md`, written 00:40:44 before any measurement)
- **1a relabelling (exact, `tol = 1e-12`).** The fitted store is held; the *codes matrix* is relabelled — well
  indices permuted (4 draws) and mode indices permuted within each well. For the ratio axes the per-well list is
  reversed, which is the same operation on a pooled mean.
- **1b pipeline floor (reported, not gating).** `|axis(well_permuted) − axis(correlation)|`. Same set partition,
  different well *order*, so `_kmeanspp_init` consumes its RNG in a different order ⇒ a genuinely different fit.
  This is the empirical noise scale of "the same value".
- **2 class count (`sign` + `S ≤ 0.50`).** Same store, same partition; only the realised class count of the
  evaluation changes. The restriction keeps exactly the `scale`/`orientation` values the **numerator realises**,
  applied to each axis's own evaluation set(s) — for the ratio axes to numerator **and** denominator together.
- **2b mismatched (ratio axes only).** Numerator on its own corner, denominator on the **unrestricted** grid —
  ⛔ the shipped construction, and the one diagnosed in `gen-r0-wide-refit` §4.2.
- **3 configuration (`6/6` signs + magnitude ratio in `[1/3, 3]`).** Montero vs wide, 3 seeds each.

**Verdict = AND over the legs an axis has.** ⛔ The negative control `leak_raw` is scored first.
