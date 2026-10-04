# PREREG — w15 spoke/coverage-control (VAL LEG ONLY)

Written BEFORE any number of this spoke exists (`val/` absent at write time; see PREREG.timestamp.txt).

## Budget
`composition` is DECLARED EXHAUSTED (ledger: 5 composition spends since 2026-09-16 vs a cap of 3 — L8, L11/12, L15, L27, L30/32). **0 scored runs are spent. No test-split number of a new configuration is produced.** A NOTE row (0 runs) is appended to `/c/chlu/SCORED-LEDGER.md`.

## Question
Is the ladder's n=3 → n=6 rise (test, K*=32: 61.614 → 66.603, `w12b-nr6-and-kstar/KSTAR-LADDER.md` seed-matched table) DEPTH or COVERAGE? For n ≥ 3 the rungs probe identical corners, so the only things that differ are (i) the store's fit-pool coverage and (ii) the estimand's train-derived masks.

## Instrument (unchanged except the val geometry)
- Store: K*=32 `CovContentStore`, `ward_balanced` label-free cut, A=min(64, min d_k − 1), address book `R.READ_PINS` (rank 128, 40 sweeps) — `w12b_ladder_kstar.build_cov_store` / `address_book`, unchanged.
- Estimand: `distinct_valid_novel_tuples` per corner, `fw.per_corner_scores`, FastJudge asserted identical to `fw.Judge`. Margin `DECORATION_MARGIN_NOVELTY = 1.0`.
- Pool: `fw.fit_pool(train, 131072, pool_rng_seed=20260919)` — positive control: recomputation reproduces the w12 ladder's test pools for rung 3 and 6 (checked at seed 0 before this PREREG: True/True; re-asserted per seed in the leg).
- VAL geometry: the ladder's own builder with `scale` = the middle band (0.964, 1.036), the only scale values no test cell uses. val k=3: 6 cells / 36,000 rows; val k=6: 60 cells / 2,880 rows; both probe the identical 60 corners × 48 rows; every val row is a TRAIN row of both test rungs (asserted). train_3val = 144,000 (≥ 131,072), train_6val = 276,480.
- Seeds 0, 1, 2 — all or none. Seed count is a column on every table.

## Cells (per seed)
| | estimand n=3 masks | estimand n=6 masks |
|---|---|---|
| store on pool(train_3val) | **A** = n=3 | **C** = n=6 at n=3's coverage |
| store on pool(train_6val) | D (diagnostic) | **B** = n=6 |
Plus charter-C controls (landscape_removed / landscape_shuffled / random_atoms, post-fit form) on every cell, and `learned_1x` (1× the K* four-part total, fitted on the same pool) on every cell.

## Bars (fixed now)
- Control gate: every store cell's controls must be < store − 1.0 on every seed, else that cell claims nothing.
- RISE exists on val iff B − A > 1.0 on every seed. If not: "no rise to decompose on val" — reported as that.
- coverage share s = (B − C)/(B − A), per seed.
- **COVERAGE** (the Head's note stands): s ≥ 0.5 on every seed **and** B − C > 1.0 on every seed.
- **DEPTH** (the caveat is too strong): |B − C| ≤ 1.0 on every seed (C matches B).
- Otherwise **MIXED**, reported with per-seed s.
- ⛔ The paper's ladder table is not touched. The test-split C cell is DECLARED NOT-RUN (budget).
