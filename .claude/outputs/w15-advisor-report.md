# WAVE 15 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-23. Nine spokes, FORGIS_14, local CPU. Base `base/iter8 = 148f3b7`.**
**All nine finished. Scored spend: 1 run (`encoder-loop`, BAR D). Eight spokes spent ZERO.** Record: `.claude/outputs/w15-*`.
⚠ Every number is quoted from a spoke's own artefact. ⛔ **The Hub re-derived nothing.**

---

## 0. The scoreboard

| spoke | verdict | scored |
|---|---|---|
| ⭐⭐ `cmaze-t-harness` | CMAZE-T built, **ready**, seeded VAL smoke — **wave 16's instrument** | 0 |
| ⭐ `cmaze-o-harness` | CMAZE-O built, **ready**, ⚠ shipped with **three** non-collapsing controls | 0 |
| ⭐⭐ `single-design-seeds` | **T7 at n = 3; all four controls collapse on every seed** — the store row is claimable | **0 — target already exhausted** |
| ⭐ `crossover-seeds` | **U-shape CONFIRMED, minimum K = 24 on all three seeds** | 0 |
| ⭐ `maze-deletion-2b` | **Exactness TRANSFERS to CMAZE-2b**, bit-identical from a clean checkout | 0 |
| ⭐ `coverage-control` | **For n ≥ 3 the ladder's rise is coverage — and can only be** | 0 — `composition` exhausted |
| ⭐ `lace-poe-rows` | **LACE and PoE do not beat the CLU anywhere**; store wins 109/109 corners | `generation-table` 3/3 |
| ⚠ `bar-d-score` | **BAR D-GPU = MIXED ⇒ no claim either way.** ⛔ Found a scorer defect (§5) | **`encoder-loop` 1** |
| `table-fixes` | MF2 / MF14(i) / N5–N8 landed | 0 |

## 1. ⭐⭐ The store row on Shapes is claimable, and it cost nothing
`single-design-seeds` **checked the ledger before spending** and found `single-design` **already exhausted at 3/3** — seeds 1–2 at K\* = 32 on TEST already existed. It wrote two **NOTE** rows, not spend rows, and spent **0 runs**.
⭐ **T7 now prints n = 3, and all four charter-C controls collapse to exactly 0.0000 on every one of the three seeds** — measured per seed, ⛔ not assumed from seed 0.
⭐ **Deletion at all three seeds for the first time:** store-scoped exactness **2×10⁻⁸ – 5×10⁻⁷**; adaptation **0.04–3.8 s reading 0 data rows**, against a refit-without reading **120,028–131,072 rows in 9.3–18.6 s**.

### ⚠ And the row that does not flatter us
⛔ **Deleting every red row leaves the 20 deleted corners still able to compose red, on all three seeds** (32.250→32.250 · 33.500→33.500 · 33.600→33.550).
> *"If a reader's question is 'after deletion, can the system still produce the deleted content?', the answer on this object is **yes**."*
⇒ **Deletion here is store-scoped exactness plus a data-free adaptation cost — ⛔ NOT capability erasure.** Capability removal is the hole's job, and the hole is a display row with a declared not-run comparator, so it is **not claimed**.

## 2. ⭐ The K curve: a confirmed U-shape, and a caveat on K\*
`crossover-seeds` pre-declared its test in `LOG.md` **before** seeds 1–2 existed, and recorded that **"S3 REFUTED is a RESULT, not a failure."**
**Confirmed:** the four-part byte total is U-shaped with the minimum at **K = 24 on every seed** (2,094,130 / 2,080,090 / 2,085,810), and the minimum is **resolvable** (band 7,060 B vs neighbour gaps 12,203 B and 30,507 B).
⛔ **`K_crossover = None`** — no K drops more than 0.1 below K = 5, and **the range was not extended to manufacture one.**
⚠⚠ **But the selection rule cannot resolve K\*.** Its tolerance is **0.02**; the measured seed sd **exceeds it at K = 10, 16, 20, 24, 28 and at K\* = 32 (0.0259)**. ⇒ **The honest finding is the PLATEAU** — composition flat across K = 5…32 while the ledger falls to its minimum near K = 24. *(Now written into the draft by w20.)*

## 3. ⭐ The ladder's rise is coverage — by construction
`coverage-control` found `composition` **already exhausted**, spent **0 runs**, ran only the free val leg.
⭐⭐ **Restricting n = 6's fit pool to n = 3's train index yields an object that IS rung n = 3.** So the entire n = 3 → n = 6 rise of **+4.989** is fit-pool coverage; **coverage share = 1.00 on every seed.**
⛔ It labelled its own headline cell **DERIVED, not measured** (a new test configuration would be a scored run, and the budget is gone), and reported its **val verdict as "NO RISE"** (below the 1.0 margin on 2 of 3 seeds) rather than leaning on the derived number.
⇒ **The Head's inference note stands, and now rests on a constructive identity.**

## 4. ⭐ LACE and PoE do not beat the CLU — with the scope stated
Store **47.495 / 47.404 / 49.119** novel valid tuples per corner; **LACE 0.165 / 0.073 / 0.229**; **PoE 0.009 / 0.000 / 0.000**. **The store wins 109/109 corners against each, on every seed.**
⛔⛔ **STANDING SCOPE, the spoke's own:** this is **LACE/PoE as instantiated in this feature space at their own declared budgets** — ⛔ **not a refutation of LACE in general.**
⭐ LACE and PoE were **ADDED**; the learned-rival columns were not replaced. `generation-table` **3/3** scored the K\* design, 3 seeds, nothing NOT-RUN: **the store claim stands; the "beats the shipped read" claim does NOT** (S2 fails on seed 0 at −0.0185).

## 5. ⛔⛔ BAR D closed as MIXED — and it found an instrument defect first
**BAR D-GPU = MIXED on rows 3 and 4** by the pre-declared rule ⇒ **no claim either way.** 0 VOID seeds. Charter-C controls collapse on every scored leg, but **the margin is ~0, so no store claim is made in either direction.** `loop_swap` saturates (`pca_tensor` at ceiling), which is descriptively why GROWS was unreachable. GPU and CPU orderings are **two separate blocks, never merged.**
⛔⛔ **The defect (registry `N359`):** the w8 scoring path **never loaded the refit baseline**. `load_rival()` was called without `root`, so it read the default — **spoke A's frozen-feature rivals** — while `$CHLU_RIVALS_ROOT` was honoured only for the metadata JSON. Feature hashes: default `ef6359a4…` vs the arm's refit `95206c71…`.
> *"The scored 'baseline' was a model from another feature space, rendered through this arm's decoder."*
⇒ ⛔ **The w8 CPU leg's 0.000 rows were spoke A's frozen-feature rival. CPU BAR D is UNMEASURED — not "VOID because the baseline scores 0", and the decoder attribution is not established.** Repaired on the w15 branch **before any test number**, with a hard refusal on feature-hash mismatch; post-repair the rival reads 40.083 / 71.875. ⭐ Unaffected: w13 `encoder-adapt` (passes `root=`), the w5 legs (never set the override).

## 6. ⭐ The maze keeps its differentiator, on the environment the paper prints
`maze-deletion-2b`: **exactness transfers to CMAZE-2b on all three seeds** — statistic residual **7.212e-16 / 9.093e-16 / 8.937e-16**; **subspace projector, address-book Gram and the deployed read all 0.000e+00**. **Reproduced bit-identically from a clean checkout** that rebuilt the banks, pool and encoders itself. 0 scored runs. ⛔ No CMAZE-1 or CMAZE-2 row in that table.

## 7. ⚠ Both harnesses shipped READY, with warnings that turned out to matter
- **CMAZE-T:** a control did not collapse on the pair row (0.2535 vs CLU 0.3932); the **cross-maze lookup floor beats the CLU on `seen`** (0.9967 vs 0.9709) — *"on this row the result is retrieval, not composition."*
- **CMAZE-O:** **three** controls did not collapse on `seen` — `landscape_shuffled` **0.2728, identical to the CLU to four decimals.**
⛔⛔ **CAVEAT THAT TRAVELS WITH BOTH:** registry `N350` — the shipped `landscape_shuffled_across_wells` used `rng.permutation(K)`, **which at K = 2 is the identity half the time**. Wave 16 found this and replaced it with a **derangement** before spending. ⛔ **The w15 harness smoke columns used the OLD control and were never re-measured.** ⚠ So the two non-collapse warnings above are **not established**. *(`crossover-seeds` kept the non-deranged control deliberately, for comparability with w12, and checked that its one non-collapsing cell at K = 20 is **not** explained by the fixed point.)*

## 8. What the Advisor is asked to rule on
1. ⚠ **Deletion is not capability erasure** — red still composes after every red row is deleted, on all three seeds. **Does the paper's deletion claim get scoped to "store-scoped exactness + data-free adaptation cost" in the abstract, not only in the methods?** (This is M3's sibling and is also HELD for the Head.)
2. ⚠ **K\* is not resolvable by its own rule on three seeds.** **Is the reported quantity K\*, or the plateau K = 5…32 with the ledger minimum at K = 24?**
3. ⛔ **CPU BAR D is UNMEASURED, not VOID.** A published characterisation changes. **Does the BAR D paragraph survive at all, given GPU BAR D is MIXED and CPU BAR D is unmeasured?**
4. ⚠ **The w15 harness control columns are not established** (N350). **Re-measure with a derangement, or drop those columns from the paper?** ⭐ w16's scored rows already use the fixed control, so the cost of dropping them is low.
