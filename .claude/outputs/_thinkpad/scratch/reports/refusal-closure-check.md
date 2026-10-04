# REFUSAL CLOSURE CHECK — DIRECTIVE 2026-09-17 rev.2 §2.3
**Run by the coordinator, seed 0, 1 thread, read-only. ⛔ No scored row. Script:
`_hub_scratch/_closure_check.py` · raw output `/tmp/closure.json`.**

## (a) Does the §C.2 atom shuffle move the address-book posterior? — ⭐ **NO, exactly zero**

| rung | atoms/well | shuffles | `max_abs(P_full − P_shuffled)` |
|---|---|---|---|
| **A** | 64 × 5 wells | 16 | ⭐ **0.0** |
| **B** | 512 × 5 wells | 16 | ⭐ **0.0** |

Max over `log_prior` **and** every factor's `log_cond`, over all 16 shuffles, both rungs.
⭐ **§1a's premise holds and no escalation is required** (falsifier §5.1 does not fire).

This is structural, not incidental: `refusal_lib.shuffled_store`'s own docstring states the shuffle
permutes each channel column among the `N` atoms and that *"the address book, the bandwidths and the
coupling are untouched: exactly ONE declared quantity differs from the full arm."* The measurement
confirms the docstring at machine precision. ⇒ A symbolic request's posterior is identical under the
control, so **no function of the posterior can be certified by the §C.2 shuffle** — which is exactly
the ceiling §1a asserts.

## (b) Well-matrix rank, real vs shuffled — ⚠ full at rung A, **rank-deficient at rung B**

| rung | atoms/well | rank, real (5 wells) | rank, shuffled | full rank? |
|---|---|---|---|---|
| **A** | 64 | 64, 64, 64, 64, 64 | 64, 64, 64, 64, 64 | ⭐ **yes, both** |
| **B** | 512 | **236, 305, 512, 479, 366** | **236, 305, 512, 479, 366** | ⛔ **no, both** |

⭐ **Rank is preserved exactly by the shuffle** (a per-column permutation is rank-preserving), real and
shuffled identical well by well, at both rungs — as expected, and it means the control is not
degenerating the store's dimensionality.

⚠ **New, not previously reported: at rung B four of five wells are rank-deficient** — 236/512,
305/512, 479/512, 366/512. The 512 atoms do not span 512 dimensions in those wells; only well 3 does.
Consequences worth carrying forward:
- The rung-B "capacity control" (8× atoms, 8× bytes) is **not** 8× independent directions — a chunk of
  the extra atoms is linearly dependent on the rest. That is a *more* charitable reading of rung B's
  flat result than "more store does not help": part of the added capacity was never there.
- ⛔ It does **not** rescue any refusal claim: (a) is exactly zero regardless of rank, and the
  `hub/refusal-2` family bound was measured at both rungs.
- ⭐ It is relevant to `hub/admission` and to `hub/generation-table-2`'s partition-`K` lever: atoms per
  well beyond the well's channel dimension buy no new directions. **Reported, not acted on.**

## Provenance
Built through `refusal_lib.build_context(0)` → `build_land(ctx, budget=…)`, i.e. the shipped path, not a
re-implementation. Grouping re-derivation reproduced the plan **bit-identically** on its own seed.
⚠ The run logged the expected `⛔⛔ THE RE-ENCODE CHECK WOULD HAVE ABORTED THIS RUN (ulp)` notice with
`enforced=False` — 51.0 ULP, 0.977 % — consistent with the standing ruling: it gathers only cached
`store_train` rows, extends nothing, and the bound is re-applied at full strength by `gather_features`
if any uncached row is ever requested. ⛔ Nothing was encoded. ⛔ Nothing pinned was edited or deleted.
