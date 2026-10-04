# PREREG ADDENDUM 3 — `gen-variance-equalisation`
⛔ **A NEW FILE.** `PREREG.md` and addenda 1–2 are **not edited**. Filed **BEFORE** seed 2 is re-run.

## What happened: the harness applied the stop rule PER SEED; the prereg's rule is ≥ 2 OF 3
`PREREG.md` §3 reads, verbatim: *"**PREMISE HOLDS** iff, on **≥ 2 of 3 seeds**, for **BOTH** `scale` and
`orientation` … **PREMISE DEAD** otherwise ⇒ ⛔ **STOP after Leg 1. Do not whiten.**"* — an **aggregate**, spoke-level
rule.
⛔ **`exp_gen_variance_eq` implemented it per seed** (`stop_if_premise_dead`), which is **stricter** than the rule I
filed. The production run therefore returned:

| seed | Leg-1 premise | direction test | what the harness did |
|---|---|---|---|
| 0 | **HOLDS** | pass | ran Legs 2–4 |
| 1 | **HOLDS** | pass | ran Legs 2–4 |
| **2** | ⛔ **DEAD** | **fail**, on `orientation` only (`rank_frac_median` 0.0051 vs the `2 × 0.0042 = 0.0084` bar) | ⛔ **stopped at Leg 1 — no arms** |

⇒ **the spoke-level verdict under the filed rule is HOLDS (2 of 3)**, but seed 2 has no Leg 2–4 data, so the
3-seed falsifier statistics the acceptance criteria require cannot be formed.

## The correction, and it is a correction TOWARD the prereg, not away from it
1. `premise_gate_scope` is added (`"spoke"` default, `"seed"` available). Under `"spoke"` the **per-seed verdict is
   recorded and shipped on its own row**, and the stop is adjudicated **once, across seeds, by the ≥ 2-of-3 rule**,
   in `summarize` — which **withholds every Leg 2–4 conclusion** if the aggregate is DEAD.
2. **Seed 2 is re-run through Legs 2–4.** ⛔ This is *not* "whitening anyway to see what happens": under the rule I
   filed, the premise is **alive at the spoke level**, and seed 2 is one of the three seeds that rule is computed
   over. ⭐ Seed 2's **DEAD** verdict ships in the headline table and in the report's first ten lines regardless of
   what its arms do.
3. ⭐ **Nothing about seeds 0 and 1 is recomputed** — their artefacts are complete and are reused by hash of path
   (`skip_if_exists`).

## ⭐ Committed BEFORE the re-run
- **A1.** Seed 2's `pca_whiten` arm will behave like seeds 0–1: distinct codes on 4,096 SPREAD rows **< 1,500**
  (seeds 0–1 measured 803 / 516 against an unwhitened 3,598 / 3,594), and TOTAL-RETAINED **below** its `identity`
  control by **> 1.0**.
- **A2.** Seed 2's `identity` arm will reproduce the shipped store **exactly**, i.e. distinct codes on SPREAD
  = **3,723** (`gen-loss-attribution`'s seed-2 figure) and its code-probe row identical to the `shipped` row, as it
  did on seeds 0 and 1.
- **A3.** The spoke-level falsifier answers will not change sign from the two-seed values already on disk.
