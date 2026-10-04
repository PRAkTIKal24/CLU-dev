# w15 spoke/coverage-control — IS THE LADDER'S RISE DEPTH, OR COVERAGE?
**Read `COMMON.md` first.** Slug `coverage-control`. Agent type: results-analyst.
**Scored: `composition` 1 run. ⚠ `composition` was 3/3 in an earlier report and 2/3 in the ledger — VERIFY the ledger before spending, and if it is exhausted, DECLARE that and run the val leg only.**

## The one job
Re-run **n = 6 with the fit pool restricted to n = 3's train index** — same depth, n = 3's coverage. This isolates whether the ladder's rise is a **depth** effect or a **coverage** effect.

⭐⭐ **The known caveat this tests:** for n ≥ 3 the ladder is a **TRAINING-DATA axis** — rungs 3, 4, 5, 6 probe the **identical 120 corners** and differ in coverage. Only **n = 2 vs n = 6** is a genuine depth contrast.

## What does NOT change
⛔ **The paper's ladder table stays exactly as it is.** You are adding one control row plus the Head's one-line inference note:
> *We beat the naive fair rivals at every n; the CLU's own 2-way → 6-way rise is not to be inferred, because rungs 3–6 share corners and differ in coverage.*

## Acceptance criterion
`COVERAGE-CONTROL.json` + `FINDINGS.md`: n = 6 at n = 3's coverage vs n = 6 at its own coverage vs n = 3, 3 seeds, same instrument and estimand, with the inference note printed verbatim and the ledger row written before any scored run.

## What would prove this wrong
⭐ If n = 6 **at n = 3's coverage** matches n = 6 at full coverage, the rise IS depth and the caveat is too strong — say so. If it drops to n = 3's level, the rise is coverage and the note stands. **Either answer is the result.**

## Files you own
Your worktree, your output dir, your ledger row.
