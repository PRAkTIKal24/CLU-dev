| # | phrase in the text | where | record file | verbatim in record | rule holds | recorded values / rule |
|---|---|---|---|---|---|---|
| 1 | about seven (more tuples than the strongest, an MLP) | Abstract; Contribution 1; Summary (pre-existing) | `w20-shapes-data-matched/SHAPES-MATCHED.md` | yes (+7.034) | PASS | mean paired margin +7.034 rounds to 7; 7 <= 7.034 |
| 2 | by more (over the transformer) | Contribution 1 | `w20-shapes-data-matched/FINDINGS.md` | yes (+7.03) | PASS | transformer paired margins 21.936 / 19.615 / 10.468 (Tab. t7) all exceed 7.034 |
| 3 | about six (survivors fall, red sphere) | Abstract; Sec. 4.3 | `w20-shapes-data-matched/FINDINGS.md` | yes (6.13, 6.43, 6.30) | PASS | each seed 6.13 / 6.43 / 6.30 rounds to 6 and is >= 6 |
| 4 | far more than after a random item deletion | Abstract | `w20-shapes-data-matched/FINDINGS.md` | yes (6.13, 0.11) | PASS | smallest conjunction loss 6.13 against largest item move 0.11 |
| 5 | as large as its fit data / the size of its fit data | Abstract (Contribution 2 pre-existing) | `w20-shapes-data-matched/FINDINGS.md` | yes (0.97 GB, the fit data itself) | PASS | record: "first moment of its 117,964 fit rows: 0.97 GB, i.e. the fit data itself" |
| 6 | nearly a gigabyte | Contribution 2; Sec. 4.3 | `w20-shapes-data-matched/SHAPES-MATCHED.json` | yes (967304800) | PASS | 967,304,800 B = 0.967 GB (decimal), below 1 and within 10% |
| 7 | refit slower than / longer than retraining a baseline | Abstract; Contribution 2; Sec. 4.4 | `w20-shapes-data-matched/FINDINGS.md` | yes (2,367) | PASS | fastest book refit 2,367 s (FINDINGS) > slowest baseline retrain 2,064.5 s (SHAPES-MATCHED.md, Tab. rivals-del) |
| 8 | (slowest retrain) 2,064.5 s | Sec. 4.4 comparison | `w20-shapes-data-matched/SHAPES-MATCHED.md` | yes (2064.5) | PASS | printed in the record |
| 9 | several times the byte reference | Abstract | `w13-seeds-and-tables/TABLES.md` | yes (4.0897) | PASS | 4.0768 to 4.0897 times |
| 10 | about four thousand unique works | Sec. 1.1 | `_thinkpad/w8/w8-novelty/NOVELTY.md` | yes (4,050) | PASS | approximately 4,050 rounds to 4,000 and 4,000 <= 4,050 |
| 11 | ten generation-side works; most of them false positives | Sec. 1.1 | `_thinkpad/w8/w8-novelty/NOVELTY.md` | yes (10) | PASS | 10 works, 7 false positives (as printed in the draft before; record NOVELTY.md) |
| 12 | four (claim atoms) of the six | Sec. 1.1 | `_thinkpad/w8/w8-novelty/NOVELTY.md` | yes (4 of 6) | PASS | exact count |
| 13 | about nine in ten (n-CLU valid, seed 0) | Sec. 4.1 | `../../outputs/pw1-render-shapes/REQUESTS-pw1.json` | yes (5177, 5688) | PASS | 5,177 / 5,688 = 0.910 |
| 14 | more often than any baseline; lookup table least often | Sec. 4.1 | `../../outputs/pw1-render-shapes/REQUESTS-pw1.json` | yes (4361, 3703, 2786, 3110, 897) | PASS | n-CLU 5,177 > MLP 4,361 > …; table 897 lowest |
| 15 | almost always wrong in one factor only | Sec. 4.1 (pre-existing words) | `../../outputs/pw1-render-shapes/factor-miss-seed0.txt` | yes (497, 511) | PASS | 497 / 511 = 0.973 |
| 16 | not confined to the capsule | Sec. 4.1 | `../../outputs/pw1-render-shapes/REQUESTS-pw1.json` | yes (170, 310) | PASS | 170 of 310 are capsules |
| 17 | tracks that count closely, inversely | Sec. 4.2 | `w12-composition-ladder/FINDINGS.md` | yes (0.96, 0.98) | PASS | r = -0.96 to -0.98 (record prints both bounds) |
| 18 | about one tuple (LSTM spot check) | Sec. 4.2 | `w20-shapes-data-matched/FINDINGS.md` | yes (20.55, 21.85) | PASS | 21.85 - 20.55 = 1.30 |
| 19 | four times the n-CLU's content rows | Sec. 4.2 (pre-existing words) | `w21-ladder-rivals-32k/CAPTION.md` | yes (32,768) | PASS | 131,072 / 32,768 = 4 exactly |
| 20 | up to about four novel tuples; most rung-seed legs | Sec. 4.2 | `w21-ladder-rivals-32k/FINDINGS.md` | yes (4.13, 12 of 15, 12/15) | PASS | largest |difference| 4.13 (range -2.15 to +4.13); 12 of 15 |
| 21 | nearly five tuples (TEST rise n=3 -> 6) | Sec. 4.2 | `w15-coverage-control/FINDINGS.md` | yes (4.989) | PASS | +4.989 is below 5, within 10% |
| 22 | coverage share complete; mask channel contributes nothing | Sec. 4.2 | `w15-coverage-control/FINDINGS.md` | yes (1.00, 0.000) | PASS | coverage share 1.00 on every seed; mask channel 0.000 |
| 23 | about thirty each (red with another shape, seed 0) | Sec. 4.3 | `w20-shapes-data-matched/FINDINGS.md` | yes (32.1) | PASS | -32.1 per combination; 30 <= 32.1 |
| 24 | about one (sphere in another colour, seed 0) | Sec. 4.3 | `w20-shapes-data-matched/FINDINGS.md` | yes (1.47) | PASS | -1.47 rounds to 1; 1 <= 1.47 |
| 25 | the others barely move | Sec. 4.3 | `w20-shapes-data-matched/FINDINGS.md` | yes (+0.51) | PASS | +0.51, inside the one-tuple margin |
| 26 | at most about a tenth of a tuple (random 1% deletion) | Sec. 4.3 | `w20-shapes-data-matched/FINDINGS.md` | yes (0.11) | PASS | largest item move 0.11 (0.110) rounds to 0.1 (stated as "about") |
| 27 | less than the declared one-tuple margin (value deletion) | Sec. 4.3 | `w20-shapes-data-matched/FINDINGS.md` | yes (0.61) | PASS | -0.06 / +0.33 / -0.61, all |.| < 1.0 |
| 28 | about 0.6 relative RMS; as much on the others | Sec. 4.3 | `w13-seeds-and-tables/TABLES.md` | yes (0.610083, 0.614277) | PASS | 0.610083 (red) and 0.614277 (others) |
| 29 | only at roundoff (with the carry) | Sec. 4.3 | `w13-seeds-and-tables/TABLES.md` | yes (6.194, 6.163) | PASS | 6.194e-08 and 6.163e-08 relative |
| 30 | nearly all of the address book's bytes | Sec. 4.3 | `w13-seeds-and-tables/TABLES.md` | yes (1015808, 1060864) | PASS | 1,015,808 / 1,060,864 = 0.958 |
| 31 | far lower minimum cosine | Sec. 4.3 | `w13-seeds-and-tables/TABLES.md` | yes (0.1999, 0.9976) | PASS | cos_min 0.1999 against 0.9976 |
| 32 | a byte each; no measurable time | Sec. 4.5 | `w13-seeds-and-tables/TABLES.md` | yes (0.0000) | PASS | two holes cost 2 bytes and take 0.0000 s |
| 33 | a free value-presence bitmap of a few bytes; perfectly | Sec. 4.5 | `w12-deletion-baselines/FINDINGS.md` | yes (1.0000) | PASS | 8-byte bitmap; AUROC / refusal 1.0000 |
| 34 | beyond a declared one-tuple bar on two of them | Sec. 4.5 | `w13-seeds-and-tables/TABLES.md` | yes (2.0899, 0.9101, 6.7416) | PASS | |-2.0899| and |-6.7416| exceed 1.0; |-0.9101| does not |
| 35 | just under 1.4 times the reference | Sec. 4.6 | `w13-seeds-and-tables/TABLES.md` | yes (1.372, 1.358) | PASS | 1.358 to 1.372 times |
| 36 | within half a percent (bytes and parameters) | Sec. 4.6 | `w13-seeds-and-tables/TABLES.md` | yes (715,808, 713,632, 719,352) | PASS | bytes 99.70% / 100.50%; parameters -0.304% / +0.495% (178,408 / 179,838 against 178,952) |
| 37 | more than eighteen tuples (item memory, 3-seed mean) | Sec. 4.6 | `w13-seeds-and-tables/TABLES.md` | yes (18.56) | PASS | 3-seed mean 18.56 |
| 38 | far below one distinct valid novel tuple (LACE, PoE) | Sec. 4.7 | `w15-lace-poe-rows/FINDINGS.md` | yes (0.267) | PASS | at most 0.267 and 0.020 |
| 39 | though its mean is positive (S2) | Sec. 4.7 | `w15-lace-poe-rows/FINDINGS.md` | yes (0.1551) | PASS | mean +0.1551 |
| 40 | about four times the reference bytes | Sec. 4.7 | `w15-lace-poe-rows/FINDINGS.md` | yes (4.09) | PASS | 4.09--4.11 times |
| 41 | about a third of the fit rows (seed-5 code) | Sec. 4.8 | `w23-seed5-diagnosis/FINDINGS.md` | yes (317, 211) | PASS | 106 / 317 = 0.334 (106 = 317 - 211, as the existing SRC note states) |
| 42 | small margin | Sec. 4.8 | `w23-seed5-diagnosis/FINDINGS.md` | yes (0.04) | PASS | margin 0.04 |
| 43 | small or absent on every seed but one; sizeable only on seed 2 | Sec. 4.8 | `w21-rel-newseeds/FINDINGS.md` | yes (0.165) | PASS | leads -0.007 / +0.024 / +0.165 / +0.006 / 0 / 0 |
| 44 | all or nearly all of the margin (seeds 3 and 4) | Sec. 4.8 | `w21-rel-newseeds/FINDINGS.md` | yes (0.97) | PASS | 0.97 and 1.00 (printed in the draft before, cited to w21-rel-newseeds/FINDINGS.md) |
| 45 | well behind (value row, n-CLU-REL) | Sec. 4.8 | `w21-rel-newseeds/TABLES.md` | yes (0.554, 0.797) | PASS | 0.554 against 0.797 |
| 46 | no valid rollout; about nine in ten per step; a minority (n-CLU-REL) | Sec. 4.8 | `w21-maze-t-seed4/FINDINGS.md` | yes (0.900, 0.930) | PASS | base 0.000; per step 0.900 to 0.930; REL at most 0.289 |
| 47 | under a second; a minute or more; thousands of gradient steps | Sec. 4.8 | `w21-maze-t-seed4/FINDINGS.md` | yes (0.88, 71) | PASS | n-CLU 0.09 to 0.88 s; retrain 71 to 140 s and 8,000 steps |
| 48 | less than one six-factor tuple (gentlest) | Sec. 5.3 | `w12-clue-theorist/CLUE.md` | yes (0.531) | PASS | -0.531 / -0.510 / -0.388 |
| 49 | about forty on the seed mean (full strength) | Sec. 5.3 | `w12-clue-theorist/CLUE.md` | yes (39.959, 46.918, 39.061) | PASS | mean 41.98; per seed 39.959 / 46.918 / 39.061 |
| 50 | well within its declared bar | Sec. 5.4 | `w10-physics-section-evidence/EVIDENCE.md` | yes (0.0433, 0.10) | PASS | median log error 0.0433 / 0.0435 / 0.0431 against bar 0.10 |
| 51 | nearly two thirds of the image; most of it | Sec. 5.4 | `w10-physics-section-evidence/EVIDENCE.md` | yes (65.17, 65.39, 83.5) | PASS | 65.17% to 65.39%; 83.5% |

51 rows, 0 FAIL
