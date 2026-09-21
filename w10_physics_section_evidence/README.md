# w10 spoke/physics-section-evidence — the reproducible half

This directory carries the ONE code artefact this spoke produced. ⛔ It touches no `chlu/`
production code, imports nothing from `chlu`, and spends no scored run.

- `derivation_checks.py` — numerical reproduction of every derivation the paper's extended physics
  section quotes (D3–D50 where a check is possible), plus a recomputation of w6 `hub/potentials`'
  B1/B2/audit bars straight from `test/s{0,1,2}/results.json`. Pure stdlib. Runs in ~4 s:
  `python derivation_checks.py`
- `derivation_checks.out` — its output as committed: **34 / 34 checks reproduce, 0 FAIL.**

It reads (read-only) from `/c/chlu/evidence/outputs/_thinkpad/w6/w6-potentials/test/s{0,1,2}/results.json`.
Change `BASE` at the bottom of the script if that tree moves.

## The prose deliverables are NOT in this repo, deliberately

`EVIDENCE.md`, `BENCHMARK-STATEMENT.md` and `LOG.md` live in
`/c/chlu/outputs/w10-physics-section-evidence/`. They are not committed here because this repo's
own `.gitignore` excludes `projects/*/` and `docs/**` — prose is kept out of the tree by design,
and this spoke did not override that convention.

## Two findings recorded here as well as in EVIDENCE.md

1. The headline constant **+1.1385** (D17's derived overdamped branch slope) is `1 + gamma/(2-gamma)`
   at **gamma = 0.243303**, which is NOT a point of the recorded 9-point gamma grid. The per-cell
   fit windows that average to it are in no machine-readable artefact, only in `LOG.md` /
   `REPORT.txt` / `RESULTS-LEDGER.md`. The LAW reproduces exactly; the CONSTANT does not reproduce
   from the artefacts. A reporting gap, not a wrong number.
2. `test/s*/results.json -> d14_confinement.max_abs_residual` is exactly 0.0 on 60 cells: it compares
   the same algebra to itself. The SCORED B6 bar is `hessian_fd_check` (3.5e-11 / 1.2e-10 / 2.7e-11),
   a finite difference that can fail. Quote that one.
