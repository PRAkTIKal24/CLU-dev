# PREREG ADDENDUM 1 — gen-codebook-allocation

**A NEW FILE, never an edit of `PREREG.md` (R34's seventh strike).** Filed after the
3-seed production run completed and **before** `summary.json` was regenerated with
the corrected verdict wording. ⛔ It changes **no pre-registered answer**.

## 1. ⛔ A DEFECT IN MY OWN VERDICT LOGIC, declared rather than quietly patched
`PREREG.md` §4 names three readings — **PASS**, **TRADE**, **NULL** — and the harness
implemented them with an `else: LOSS` fallback. The run produced a **fourth**
outcome that §4 did not enumerate and the fallback therefore mislabelled:

> `mse_optimal`: TOTAL-RETAINED **flat** (Δ = −0.012 against a seed spread of 0.075)
> **but** distinct-of-72 **rose beyond spread** (9.00 ± 0.82 → 14.33 ± 3.30).

The harness printed **"⛔ LOSS"** for that. ⛔ **That label is wrong** — nothing fell.
The fix adds the missing branch:

> **SPLIT (⛔ NOT A PASS):** TOTAL-RETAINED is within seed spread of `uniform` **and**
> distinct-of-72 rises beyond seed spread. The pre-registered pass requires TOTAL to
> **rise**; it did not, so the answer to §4 is **still NO**.

**LOSS** is narrowed to what it always meant: `Δ TOTAL < −seed spread`.

⭐ **Nothing about the pre-registered pass/fail answer moves.** `equal_distortion` was
a LOSS before this addendum and is a LOSS after it; `mse_optimal` was NOT-A-PASS
before and is NOT-A-PASS after. Only the *name* of `mse_optimal`'s non-pass changes,
from a label that was factually false to one that is not.

⚠ **The distinct-of-72 rise is therefore a POST-HOC observation, not a pre-registered
result.** It is the same shape as `gen-variance-equalisation`'s `diag_std` finding
(9 → 15) and carries the same caveat: **a 72-row statistic with a 3.30 seed sd, on
3 seeds.** ⛔ It must be replicated before anything is built on it, and ⛔ it must not
be promoted to primary.

## 2. ⚠ ONE POST-HOC ANALYSIS, declared as post-hoc
`PREREG.md` §3 pre-registered the law's fit as a **pooled** log-log regression of
`ε_measured` on `ε_predicted` (L5–L8). Having run it, the more direct question is
visible and was **not** pre-registered:

> for each (seed, well), fit `log ε_measured` against **`log M`** across the three
> arms, and compare the fitted exponent to the law's own `−2/PR`.

This is a **within-well `M` sweep** the three allocations happen to provide for free.
⛔ It was computed **after** the artefacts were complete, has **no pre-registered
interval**, and is reported as an observation with its limitations (for several wells
two of the three `M` values are nearly equal, so the slope is effectively a two-point
contrast over `M ∈ [4, 186]`). ⛔ It may not be extrapolated to `M` outside that range
without saying so.

⛔ No per-seed artefact is recomputed by either item: `summary.json` is regenerated
from the unchanged `alloc_s{0,1,2}.json` (`skip_if_exists` HIT, 3/3).
