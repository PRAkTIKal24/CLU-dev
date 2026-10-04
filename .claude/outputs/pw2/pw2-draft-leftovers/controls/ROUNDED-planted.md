| # | phrase in the text | where | record file(s) | verbatim in record | rule holds | recorded values / rule |
|---|---|---|---|---|---|---|
| 1 | about six tuples per held-out combination (red sphere, survivors) | Limitation 2 | `w20-shapes-data-matched/FINDINGS.md` | yes (6.13, 6.43, 6.30) | PASS | each seed 6.13 / 6.43 / 6.30 rounds to 6 and is >= 6 |
| 2 | about thirty on the combinations that request red with another shape at seed 0 | Limitation 2 | `w20-shapes-data-matched/FINDINGS.md` | yes (32.1) | PASS | -32.1 per combination (20 combinations, seed 0); 30 <= 32.1 |
| 3 | at most about a tenth of a tuple (random 1% deletion) | Limitation 2 | `w20-shapes-data-matched/FINDINGS.md` | yes (-0.02 / -0.11 / -0.01) | PASS | item deletion moves -0.02 / -0.11 / -0.01; largest 0.11 rounds to 0.1 (same phrase and rule as Sec. 4.3, pw2-draft-numbers ROUNDED row 26) |
| 4 | nearly a gigabyte (the book's retained statistic) | Limitation 3 | `w20-shapes-data-matched/SHAPES-MATCHED.json`; `w20-shapes-data-matched/FINDINGS.md` | yes (967304800, 0.97 GB) | PASS | 967,304,800 B = 0.967 GB (decimal): below 1 and within 10% |
| 5 | its refit takes longer than a baseline's retrain-without | Limitation 3 | `w20-shapes-data-matched/FINDINGS.md`; `w20-shapes-data-matched/SHAPES-MATCHED.md` | yes (2,367, 2064.5) | PASS | fastest book refit 2,367 s (2,367.3 in Tab. del) > slowest baseline retrain 2,064.5 s (Tab. rivals-del) |
| 6 | about four times the reference (the deletable object at K*) | Limitation 4 | `w13-referee-2/REFEREE-v1.md` | yes (4.0897, 4.0768, 4.0847) | PASS | per seed 4.0897 / 4.0768 / 4.0847 (Tab. ledger-kstar, cited to REFEREE-v1.md; TABLES.md prints the seed-0 4.0897 only) all round to 4 and are >= 4 |
| 7 | U-shaped, with its minimum at K=24 | Limitation 4 | `w13-seeds-and-tables/TABLES.md` | yes (3.9942, 4.0877) | PASS | four-part x column of Tab. kcurve: 7.8545 … 4.0069 (K=20), 3.9942 (K=24), 4.0567 (K=28), 4.0877 (K=32); minimum at K=24 |
| 8 | on the one configuration at K>=36 whose composition was measured, composition collapses | Limitation 4 | `w13-seeds-and-tables/TABLES.md`; `w13-referee-2/REFEREE-v1.md` | yes (0.991x, 0.0648, 0.8918, 0.991) | PASS | K=40 / A=7 at 0.991x the pin: valid mean 0.0648, against 0.8918 at K=5 (TABLES.md, the w12r qualifier kept verbatim in sense) |
| 9 | the MLP at seed 0 still gains more than a tuple from ten to thirty times its steps | Limitation 5 | `w20-shapes-data-matched/FINDINGS.md` | yes (+1.47) | PASS | +1.47 (36.71 -> 40.01 -> 41.48 at seed 0) |
| 10 | PLANTED: about four times the reference (record 3.9942) | - | `w13-seeds-and-tables/TABLES.md` | yes (3.9942) | FAIL | planted: 4 > 3.9942, rounds up past the record |

10 rows, 1 FAIL
