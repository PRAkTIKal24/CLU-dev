# PREREG — gen-c-set-a7 (written BEFORE the detector was run)

Written: 2026-09-13, before any detector AUROC was measured on any artifact.
Branch `agent/experiment-engineer/gen-c-set-a7`, base `main @ 1bcfcb9`.

## What is being measured
A **no-model structural detector**: a function that sees a query record's *request term list with
every factor VALUE masked* (plus, as a reported diagnostic only, the record's id and key set) and
scores it. For each structural feature f we report the **orientation-free AUROC**
`max(AUROC(f), 1 - AUROC(f))` of invalid-vs-valid, so a detector that separates by firing *low*
counts exactly as much as one that fires high.

## Feature list, fixed in advance (11 assertion features)
`n_terms`, `n_distinct_factors`, `max_factor_multiplicity`, `n_redundant_terms`,
`is_factor_order_sorted`, `n_order_inversions`, `first_duplicate_position`, `sum_factor_indices`,
`first_factor_index`, `last_factor_index`, `masked_request_json_len`.
Diagnostic-only (NOT asserted, because they are set *labels* by construction, not features an arm
scores): `id_numeric_suffix`, `id_token_count`, `record_key_count`.

## Predictions (derived, not guessed)

**P1 — BEFORE A7, C scored against the base 6-term valid pool (A∪B).**
`build_c_invalid` appends a 7th term to every `duplicate_factor` record and leaves `out_of_range`
records at 6 terms, one per factor, in factor order. Therefore, *derived from the builder's code*:
- `duplicate_factor` vs base: `n_terms`, `n_distinct_factors`, `max_factor_multiplicity`,
  `n_redundant_terms`, `masked_request_json_len` all **exactly 1.0000** (complete separation, no ties).
- `out_of_range` vs base: **exactly 0.5000** on every assertion feature (both are 6 distinct terms in
  factor order; the only difference is a value, which is masked).
- pooled (stratified over the two kinds, each against its pre-A7 control = base): **0.7500**.
  This reproduces `gen-certificate` §3's measured `count` arm: 1.0000 / 0.5000 / 0.7500.

**P2 — AFTER A7, each kind scored against its declared structural control**
(`duplicate_factor` → `a7_redundant`, `out_of_range` → `base`).
A7 appends one term repeating a factor **at the value it already has**, at the **end** of the term
list, cycling the repeated factor index as `i % n_factors` — the same cycling rule `build_c_invalid`
uses for its duplicated factor. Hence, for `|A7|` and `|C_dup|` both multiples of `n_factors`, every
assertion feature has an **identical distribution** in the two sets. Prediction:
- every assertion feature, `duplicate_factor` vs `a7_redundant`: **exactly 0.5000**;
- every assertion feature, `out_of_range` vs `base`: **exactly 0.5000**;
- **max over features, per kind and pooled: exactly 0.5000.**

**P3 — the tolerance.** Declared in the test as `DETECTOR_TOLERANCE = 0.05`. P2 predicts the realised
max is 0.5000, i.e. **0.05 of headroom is 100 % unused**. If the realised max lands anywhere in
(0.50, 0.55] that is *not* a pass to be celebrated — it means a structural asymmetry exists that my
construction did not intend, and I must name it in the report.

**P4 — the falsifier (from the task's DIAL DECLARATION).** If any assertion feature exceeds
0.5 + 0.05 against the matched control after A7, another structural giveaway remains and this spoke
has NOT closed the leak. I commit to reporting that as a failure rather than widening the tolerance.

**P5 — bit-identity.** Re-running the emitter reproduces `c_invalid.json` byte-for-byte:
predicted sha256 unchanged from the shipped file (recorded before the re-run:
`shapes3d` file at `.claude/data/gen/shapes3d/c_invalid.json`).
