# w9 hub/maze — RESULTS LEDGER

Every run against a held-out set is a row here, written BEFORE it executes.
Budget: **`maze`, 3 scored runs.** DEV legs are on VAL and are not scored runs.

| # | kind | leg | seed | config | PREREG | gate state at launch | status |
|---|---|---|---|---|---|---|---|
| D1 | DEV | val | 0 | K=8 A=8 ridge=1.0, AE 6000, no pool | — | absent at launch | done 06:10Z — encoder memorised (row ceiling 0.41), latents unstandardised |
| D2 | DEV | val | 0 | K=8 A=8 ridge=1.0, AE 12000, pool 270 sites | — | partial | done 07:05Z — ceilings 0.62/0.47/0.52; CLU crippled by store span 0.034 |
| D3 | DEV | val | 0 | store sweep: K in {8,16,32} x atoms x ridge in {0.01,1,100}, 45 configs | — | partial | done 08:05Z — selected K=8, n_atoms=31, ridge=0.01 |

Scored rows are appended below as they are pre-registered.

| # | kind | leg | seed | config | PREREG | gate state at launch | status |
|---|---|---|---|---|---|---|---|
| **S1** | **SCORED 1/3** | **test** | 0 | K=8, n_atoms=31, ridge=0.01 (rank/d_k 0.938) + `clu_lowrank` n_atoms=8; AE 12000 steps, pool 270 sites; baselines matched to 410,688 floats | PREREG-w9-maze-1.md | partial (seeds [0], mlp only) | **DONE 08:26Z** — BAR 1/2/3 FAIL, deletion exactness PASS |
| **S2** | **SCORED 2/3** | **test** | 1 | identical | PREREG-w9-maze-2.md | partial | **DONE 09:12Z** — same verdicts, CLU weakest seed |
| **S3** | **SCORED 3/3** | **test** | 2 | identical | PREREG-w9-maze-3.md | partial | **DONE 09:40Z** — same verdicts |

| A1 | AMENDMENT (DEV) | val | 0 | as S1 but `--exact-head-delete` | Amendment 1 in LOG | partial | done 09:55Z — post-deletion composition −0.0120 vs −0.1569 |
| D4 | DEV | val | 0 | per-well explained-variance spectrum | — | partial | done 10:20Z — effective dimension 46/256 |
| R1 | REPLICATION | test | 0 | clean checkout of 7db71f6, encoder retrained from scratch | PREREG-w9-maze-1.md | partial | done 10:05Z — **zero numeric differences** vs S1 |

**SCORED BUDGET USED: 3 of 3.** No held-out number was taken outside S1/S2/S3.
Every other leg above is VAL (DEV) or a replication of an already-pre-registered scored leg.
