# PREREG-addendum-0 — Advisor countersignature conditions (2026-09-15, BEFORE the scored run)

Both amendments HARDEN revision 3; neither moves a bar in the kinder direction. They bind the scored run and every rung.

## A. §2b tie rule — a WIN must clear the margin
- **WIN** = `Δ > m` (m = 0.020) **and** the per-seed paired difference has the same sign on 3/3 seeds.
- `0 < Δ ≤ m` with 3/3 same sign is reported as **TIE (favourable)**, never as WIN. A +0.001 with three agreeing signs is a tie.
- TIE / LOSS / INDETERMINATE unchanged. `m` unchanged.

## B. §9 refusal — the threshold is set from VALID data only, before scoring
- The single threshold on `(final energy, still-descending)` is chosen on the repaired stratified `val` split using **valid requests only** (e.g. a declared quantile of settled energy on seen-valid rows), and written into the run's `PREREG-addendum-1.md` **before** any invalid (C-set) request is scored.
- ⛔ Choosing or adjusting the threshold using any invalid example is invalid-example supervision and voids the zero-supervision claim for that run.
- The (reject-invalid, accept-unseen-valid) pair and AUROC are then reported at that one threshold, 3 seeds.

## C. §5b.0 identification gate — accepted, capped
- One L4 calibration job, run once. Its `n_val` is recorded and used for every rung. ⛔ No further iteration on the gate itself; it is a precondition, not a study.

— Advisor, 2026-09-15
