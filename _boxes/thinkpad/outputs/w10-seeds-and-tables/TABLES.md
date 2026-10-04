# TABLES.md — the paper's scoreboard, every row at 3 seeds on generation 2

**Built** 2026-09-21T08:34:28Z by `build_tables.py` in this directory. **Scored runs spent by this spoke: ZERO** — it re-reads artefacts only.

**How to trace any cell.** Every number below is written `mean (seed0 / seed1 / seed2)`. `TABLES.json` carries the same cell with a `provenance` list — one `{source, path, pointer}` entry per seed — and `sources` maps every source id to its absolute path, byte count and sha256. `python build_tables.py` regenerates this file, `TABLES.json` and every figure from those paths alone.

**The rules this file is built under** (from the w10 charter):
- All three seeds or none - no row reports a best-of.
- Runs at different compute budgets do not share a table - T1/T2/T3/T4 are four pins.
- w8/encoder-loop absolutes (shortened `w8_v1_short` decoder) do not share a table with the incumbent's.
- CMAZE-1's n-way row is reported as a CONJUNCTION and PER ROUTE, because its ceilings are per route.
- Every CMAZE-1 composition row carries BLIND-SEQUENTIAL measured on its own split, not the published 0.4980.
- A row that was not run is a DECLARED NOT-RUN with a reason and a launch line, never a blank or a zero.

---

## TABLE 1 (MAIN) — MAIN SCOREBOARD - CMAZE-1 (w9 hub/maze), held-out TEST, 3 of 3 scored seeds

**Pins.** `cell` = CMAZE-1 (env-scout adopted generator), hub/maze · `condition` = test (held out) · `generation` = 2 · `seeds` = [0, 1, 2] · `scored_runs` = 3 of 3 (S1/S2/S3, pre-registered before execution) · `K` = 8 · `n_atoms` = 31 · `ridge` = 0.01 · `ae_steps` = 12,000 · `baseline_steps` = 4,000 · `latent_d` = 256 · `read_floats` = 410,688 · `deployed_bytes` = 1,642,752 · `statistic_bytes_float64_declared_beside` = 35,904 · `rank_over_width` = 0.9375 · `matched_params` = mlp 410,920, lstm 409,846, transformer 405,776 · `n_mazes` = 96

*Every cell is the mean of scored seeds 0/1/2 with the three per-seed values in brackets. Pins: K=8, n_atoms=31, ridge=0.01, AE 12,000 steps, baselines 4,000 steps, read = 410,688 floats (1,642,752 B deployed) with the float64 sufficient statistic (35,904 B) declared beside it and never netted in; matched parameters mlp 410,920 / lstm 409,846 / transformer 405,776. Row 3 is reported BOTH as the scored CONJUNCTION and PER ROUTE, because the ceilings are per route.*

### 1a. RESULT rows — validity under the exact judge

| row | class | n (s0/s1/s2) | CLU (the arm) | MLP (matched params) | LSTM (matched params) | Transformer (matched params) | lookup, matched bytes (C.1 floor) | clu_lowrank (n_atoms=8) | clu_nopair (no pair term) | landscape_shuffled (C.2) | random_atoms (C.2) | landscape_removed (C.2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| row 1 - seen request | **RESULT** | 1208 / 1201 / 1277 | 1.0000 (1.0000 / 1.0000 / 1.0000) | 0.9118 (0.9048 / 0.9192 / 0.9115) | 0.6930 (0.6507 / 0.7344 / 0.6938) | 0.9865 (0.9892 / 0.9867 / 0.9836) | 1.0000 (1.0000 / 1.0000 / 1.0000) | 0.0157 (0.0240 / 0.0067 / 0.0164) | 0.0140 (0.0182 / 0.0158 / 0.0078) | 0.9509 (0.9396 / 0.9725 / 0.9405) | 0.9601 (0.9222 / 0.9825 / 0.9757) | 0.0000 (0.0000 / 0.0000 / 0.0000) |
| row 2 - unseen agent-target pair, seen maze | **RESULT** | 195 / 201 / 218 | 0.1279 (0.1590 / 0.0597 / 0.1651) | 0.1300 (0.1385 / 0.1095 / 0.1422) | 0.1450 (0.1897 / 0.0846 / 0.1606) | 0.1840 (0.1897 / 0.1741 / 0.1881) | 0.1565 (0.1590 / 0.1592 / 0.1514) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0198 (0.0256 / 0.0199 / 0.0138) | 0.1383 (0.1590 / 0.1045 / 0.1514) | 0.1414 (0.1692 / 0.0945 / 0.1606) | 0.0000 (0.0000 / 0.0000 / 0.0000) |
| row 3 - n-way (CONJUNCTION) | **RESULT** | 183 / 192 / 191 | 0.0284 (0.0383 / 0.0469 / 0.0000) | 0.0196 (0.0273 / 0.0156 / 0.0157) | 0.0493 (0.0383 / 0.0677 / 0.0419) | 0.0442 (0.0492 / 0.0417 / 0.0419) | 0.0159 (0.0164 / 0.0052 / 0.0262) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0055 (0.0164 / 0.0000 / 0.0000) | 0.0319 (0.0383 / 0.0469 / 0.0105) | 0.0195 (0.0273 / 0.0312 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) |
| row 3 - n-way (PER ROUTE) | **RESULT** | 389 / 403 / 414 | 0.0852 (0.1183 / 0.0769 / 0.0604) | 0.1120 (0.1131 / 0.1141 / 0.1087) | 0.1140 (0.1311 / 0.1141 / 0.0966) | 0.1246 (0.1285 / 0.1390 / 0.1063) | 0.0946 (0.1028 / 0.0893 / 0.0918) | 0.0026 (0.0077 / 0.0000 / 0.0000) | 0.0126 (0.0154 / 0.0223 / 0.0000) | 0.0908 (0.1080 / 0.1017 / 0.0628) | 0.0693 (0.0874 / 0.0769 / 0.0435) | 0.0000 (0.0000 / 0.0000 / 0.0000) |
| row 4 - budget interpolation | **RESULT** | 434 / 443 / 452 | 0.0996 (0.1313 / 0.0722 / 0.0951) | 0.1183 (0.1290 / 0.1151 / 0.1106) | 0.1229 (0.1498 / 0.1084 / 0.1106) | 0.1430 (0.1452 / 0.1445 / 0.1394) | 0.1122 (0.1198 / 0.1129 / 0.1040) | 0.0015 (0.0046 / 0.0000 / 0.0000) | 0.0136 (0.0184 / 0.0181 / 0.0044) | 0.1071 (0.1267 / 0.1016 / 0.0929) | 0.0905 (0.1106 / 0.0813 / 0.0796) | 0.0000 (0.0000 / 0.0000 / 0.0000) |

**Column classes.** Every column above is one of three things, and the charter requires the label to be printed, not inferred:

| column | class | what it is |
|---|---|---|
| CLU (the arm) | **arm** | the CLU read: atoms + request-addressed address book |
| MLP (matched params) | **RESULT baseline** | learned baseline, 410,920 params |
| LSTM (matched params) | **RESULT baseline** | learned baseline, 409,846 params |
| Transformer (matched params) | **RESULT baseline** | learned baseline, 405,776 params |
| lookup, matched bytes (C.1 floor) | **ablation (charter C.1 floor)** | a printed floor: the best a table of stored routes can do at the arm's byte budget |
| clu_lowrank (n_atoms=8) | **ablation** | the same read with 8 atoms per well instead of 31 - the saturation-ratio lever |
| clu_nopair (no pair term) | **ablation** | the read with the pair interaction removed |
| landscape_shuffled (C.2) | **ablation (charter C.2 control)** | atoms permuted across wells - if the row survives, the store is decorative |
| random_atoms (C.2) | **ablation (charter C.2 control)** | atoms replaced by random directions at the same rank |
| landscape_removed (C.2) | **ablation (charter C.2 control)** | atoms zeroed - the collapse control |

> ⛔ **BAR 3 (the C.2 controls) FAILS and that is the mechanism of this table.** On row 1 `landscape_shuffled` reads 0.9509 and `random_atoms` 0.9601 against the CLU's 1.0000; on row 2 `random_atoms` (0.1414) is inside the seed spread of the CLU (0.1279). Only `landscape_removed` collapses to 0.0000. ⛔ **Under this hub's own pre-registration no RESULT row is claimed for the CLU arm on CMAZE-1.** BAR 1 also fails on all three composition rows: against the best learned baseline the CLU is behind by -0.0560 (row 2), -0.0209 (row 3 conjunction) and -0.0434 (row 4). Traced to `/c/chlu/outputs/w9-maze/LOG.md` §5.

### 1b. Ceilings printed beside every RESULT row

| row | representation | pipeline (encode->decode->execute->judge) | store span | ceiling n (s0/s1/s2) | row n (s0/s1/s2) |
|---|---|---|---|---|---|
| row 1 - seen request | 1.0000 (1.0000 / 1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000 / 1.0000) | 1208 / 1201 / 1277 | 1208 / 1201 / 1277 |
| row 2 - unseen agent-target pair, seen maze | 1.0000 (1.0000 / 1.0000 / 1.0000) | 0.6319 (0.6205 / 0.5871 / 0.6881) | 0.6365 (0.6205 / 0.5871 / 0.7018) | 195 / 201 / 218 | 195 / 201 / 218 |
| row 3 - n-way (CONJUNCTION) | 1.0000 (1.0000 / 1.0000 / 1.0000) | 0.5255 (0.5090 / 0.5409 / 0.5266) | 0.5230 (0.5090 / 0.5360 / 0.5242) | 389 / 403 / 414 | 183 / 192 / 191 |
| row 3 - n-way (PER ROUTE) | 1.0000 (1.0000 / 1.0000 / 1.0000) | 0.5255 (0.5090 / 0.5409 / 0.5266) | 0.5230 (0.5090 / 0.5360 / 0.5242) | 389 / 403 / 414 | 389 / 403 / 414 |
| row 4 - budget interpolation | 1.0000 (1.0000 / 1.0000 / 1.0000) | 0.5573 (0.5438 / 0.5485 / 0.5796) | 0.5566 (0.5461 / 0.5440 / 0.5796) | 434 / 443 / 452 | 434 / 443 / 452 |

> ⚠ **Row 3's denominators differ.** The scored n-way row is a CONJUNCTION over the routes of one request (n = 183 / 192 / 191); its ceilings are computed PER ROUTE (n = 389 / 403 / 414). The conjunction reading and the per-route reading are both printed above, and only the per-route reading is comparable with the printed ceiling. See figure 2.

### 1c. Caveat columns, measured on THIS split — they travel with every composition row

| row | BLIND-SEQUENTIAL | BLIND-FACTOR | BLIND-A | BLIND-COLOUR | BLIND-B | ORACLE | RANDOM | rule B bends the route |
|---|---|---|---|---|---|---|---|---|
| row 2 - unseen agent-target pair, seen maze | 0.9037 (0.9077 / 0.8905 / 0.9128) | 0.4762 (0.4564 / 0.4677 / 0.5046) | 0.4160 (0.3590 / 0.4577 / 0.4312) | 0.2037 (0.2308 / 0.2289 / 0.1514) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 1.0000 (1.0000 / 1.0000 / 1.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.484 (0.513 / 0.473 / 0.468) |
| row 3 - n-way (CONJUNCTION) | 0.9001 (0.9229 / 0.8933 / 0.8841) | 0.4720 (0.4447 / 0.4665 / 0.5048) | 0.4199 (0.3779 / 0.4566 / 0.4251) | 0.2176 (0.2622 / 0.2432 / 0.1473) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 1.0000 (1.0000 / 1.0000 / 1.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.487 (0.506 / 0.474 / 0.481) |
| row 3 - n-way (PER ROUTE) | 0.9001 (0.9229 / 0.8933 / 0.8841) | 0.4720 (0.4447 / 0.4665 / 0.5048) | 0.4199 (0.3779 / 0.4566 / 0.4251) | 0.2176 (0.2622 / 0.2432 / 0.1473) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 1.0000 (1.0000 / 1.0000 / 1.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.487 (0.506 / 0.474 / 0.481) |

> ⛔ **BLIND-SEQUENTIAL on this split is 0.9077 / 0.8905 / 0.9128 (mean 0.9037) on row 2 and 0.9229 / 0.8933 / 0.8841 (mean 0.9001) on row 3 — NOT the published CMAZE-1 0.4980.** Measured cause (w9/maze LOG §3): 1,242 of 1,449 optimal routes pick up exactly one key, so there is no key ORDER for rule B to trap. This is a property of the REQUEST LAYER, not of CMAZE-1. Published-value provenance: `/c/chlu/outputs/w8-env-scout/screen_results_DEFAULTS.txt`. See figure 5.

### 1d. DISPLAY rows — inductive bias; the only comparator is the supervised refuser

**Refusal.** The CLU has never seen an impossible example; the supervised refuser is trained on 552 / 562 / 522 of them (seeds 0/1/2) and is evaluated on held-out mazes 48..95 only, so the like-for-like CLU column is the held-out one. Lower is better on the grounded-possible row (those are FALSE refusals).

| row | class | n (s0/s1/s2) | CLU refusal rate (all mazes) | CLU refusal rate (held-out mazes 48..95) | supervised refuser (the only comparator) | CLU validity when it acts (all mazes) |
|---|---|---|---|---|---|---|
| refusal - budget below the shortest legal route | **DISPLAY** | 280 / 260 / 232 | 0.9601 (0.9714 / 0.9692 / 0.9397) | 0.9537 (0.9669 / 0.9669 / 0.9274) | 0.4972 (0.5099 / 0.5702 / 0.4113) | 0.0000 (0.0000 / 0.0000 / 0.0000) |
| refusal - target walled off | **DISPLAY** | 837 / 846 / 837 | 0.9801 (0.9606 / 0.9846 / 0.9952) | 0.9688 (0.9348 / 0.9740 / 0.9976) | 0.2995 (0.3357 / 0.3050 / 0.2577) | 0.0000 (0.0000 / 0.0000 / 0.0000) |
| refusal - unseen (agent,target) novelty control | **DISPLAY** | 573 / 534 / 644 | 0.9781 (0.9721 / 0.9794 / 0.9829) | 0.9735 (0.9559 / 0.9891 / 0.9754) | 0.2008 (0.1140 / 0.2545 / 0.2338) | 0.0012 (0.0035 / 0.0000 / 0.0000) |
| refusal - grounded-possible  (FALSE refusals - lower is better) | **DISPLAY** | 584 / 604 / 632 | 0.8182 (0.7979 / 0.8560 / 0.8006) | 0.7986 (0.7544 / 0.8296 / 0.8117) | 0.2248 (0.2384 / 0.2412 / 0.1948) | 0.0560 (0.0616 / 0.0414 / 0.0649) |

**Post-deletion refusal.** Two different operations, kept apart: the CONTENT deletion removes every item written under the word from the sufficient statistic (this is the one that carries the exactness claim, Table 4); the INTERFACE deletion removes the action word from the emission interface and touches no atom.

*post-deletion refusal - delete the green target (CONTENT deletion)*  ·  class **DISPLAY**  ·  n = 106 / 134 / 108

| column | mean (seed0 / seed1 / seed2) |
|---|---|
| refusal on green requests BEFORE | 0.7604 (0.7642 / 0.8134 / 0.7037) |
| refusal on green requests AFTER | 0.9950 (1.0000 / 0.9851 / 1.0000) |
| validity on green requests AFTER | 0.0000 (0.0000 / 0.0000 / 0.0000) |

*post-deletion refusal - delete the move-left action (INTERFACE deletion, no atom touched)*  ·  class **DISPLAY**  ·  n = 538 / 577 / 611

| column | mean (seed0 / seed1 / seed2) |
|---|---|
| refusal BEFORE (grounded-possible) | 0.8182 (0.7979 / 0.8560 / 0.8006) |
| refusal AFTER, routes that NEED move-left | 0.9850 (0.9740 / 0.9844 / 0.9967) |
| validity AFTER, routes that NEED move-left | 0.0000 (0.0000 / 0.0000 / 0.0000) |
| refusal AFTER, routes that do NOT need it | 0.7766 (0.8696 / 0.8889 / 0.5714) |

**Post-deletion composition without retraining.**

*post-deletion composition WITHOUT retraining - surviving targets (as pre-registered)*  ·  class **DISPLAY**  ·  n = 478 / 470 / 524

| column | mean (seed0 / seed1 / seed2) |
|---|---|
| validity on surviving targets BEFORE | 0.1024 (0.1569 / 0.0702 / 0.0802) |
| validity on surviving targets AFTER | 0.0000 (0.0000 / 0.0000 / 0.0000) |
| delta | -0.1024 (-0.1569 / -0.0702 / -0.0802) |

> ⚠ **This row is the AS-PRE-REGISTERED number and it FAILS.** The w8 basis-change carry `W_k <- W_k R_k` is stated for a read expressed in the orthonormal basis `U_k`, while this hub's read is expressed in the deployed modes `G_k`, so the carry is the wrong change of coordinates here. Re-measured on VAL seed 0 with the repaired mechanism (`arms.CluRead.delete_rows_exact`, which downdates the head's own sufficient statistics `A = Phi^T Phi` and `Mx = Phi^T X`): surviving targets **0.0944 -> 0.0823, delta -0.0120**, inside BAR 6's +-0.05. ⛔ That is VAL, seed 0, ONE measurement — the TEST version at 3 seeds is **DECLARED NOT-RUN NR5**, not a row. Provenance: `/c/chlu/outputs/w9-maze/runs/amend_exactdelete_seed0_val.json` `/display/delete_green/composition_without_retraining/validity_delta_on_other_targets`.

### 1e. Instrument audits that make the rows readable at all

| audit | result | traced to |
|---|---|---|
| judge exact in all four parts (legal moves / reaches target / within budget / pickup rules) | **True** | `/c/chlu/outputs/w9-harness/harness_seed0.json` `/judge_audit/exact` |
| tolerance used anywhere in scoring | **None** | `/c/chlu/outputs/w9-harness/RESULTS.md` audit table |
| stepper disagreements with the judge | **0 / 750 routes** | `/c/chlu/outputs/w9-harness/harness_seed{0,1,2}.json` `/stepper_audit/disagreements` |
| impossible examples seen by a non-refuser arm | **0** | `/c/chlu/outputs/w9-harness/RESULTS.md` audit table |
| read is request-only (no feature argument; bitwise stable under feature perturbation) | **True, 8 checked per seed** | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/request_only_assertion` |
| plan-map collisions (representation ceiling) | **0 on every row, every seed** | `/c/chlu/outputs/w9-maze/runs/scored{1_seed0,2_seed1,3_seed2}_test.json` `/representation_ceiling/*/collisions` |
| independent replication of scored run 1 (clean checkout, encoder retrained from scratch) | **zero numeric differences** | `/c/chlu/outputs/w9-maze/replication/REPLICATION.md`; `/c/chlu/outputs/w9-maze/LEDGER.md` row R1 |

---

## TABLE 2 — Shapes3D generation cell - w8 hub/encoder-loop instrument, 3 seeds

**Pins.** `cell` = Shapes3D generation cell, w8 hub/encoder-loop · `condition` = test · `generation` = 2 · `seeds` = [0, 1, 2] · `scored_runs` = 1 of 3 (2 returned unspent) · `decoder_recipe` = w8_v1_short · `joint_steps` = 400 · `n_train` = 8,192 · `dec_refit_steps` = 400 · `warm_steps` = 1,500 · `k_wells` = 5 · `n_modes` = 64 · `metric` = distinct_valid_novel_tuples · `margin_declared_in_advance` = 1.0

*SEPARATE TABLE BY CHARTER. These absolutes were produced by the shortened `w8_v1_short` decoder at joint_steps=400 / n_train=8192 and are NOT comparable with the incumbent's row-3/row-4 values (a shortened decoder recipe is a different instrument, not a different setting). The metric is a COUNT of distinct valid novel factor tuples per corner, not a validity rate: row 3's ceiling is 36 tuples/corner and row 4's is 72, so the two rows are not on one absolute scale either.*

> ⛔ **This table does not share a table with Table 1, nor with the incumbent's row-3/row-4 values.** Different cell, different metric, different compute pin, and a shortened decoder recipe (`w8_v1_short`) — a different instrument, not a different setting.

**row 3 - 2-way composition**  ·  class **RESULT**

| column | mean (seed0 / seed1 / seed2) |
|---|---|
| pca_tensor @ encoder ALONE | 12.083 (17.450 / 8.067 / 10.733) |
| pca_tensor @ loop_nosw (unfrozen, NO swap loss) | 8.461 (11.483 / 5.450 / 8.450) |
| pca_tensor @ loop_swap (THE ARM) | 10.311 (10.683 / 7.817 / 12.433) |
| A' = loop_nosw - alone | -3.622 (-5.967 / -2.617 / -2.283) |
| A'' = loop_swap - loop_nosw (what the SWAP LOSS bought) | +1.850 (-0.800 / +2.367 / +3.983) |
| A = loop_swap - alone (the NET) | -1.772 (-6.767 / -0.250 / +1.700) |
| A: bootstrap CI95 low | -3.456 (-8.717 / -1.600 / -0.050) |
| A: n corners (paired) | 60 (60 / 60 / 60) |
| A'': sign-test p (two-sided) | 0.2614 (0.7838 / 0.0001751 / 0.000198) |
| A': sign-test p (two-sided) | 0.1364 (7.513e-07 / 0.05946 / 0.3497) |

**row 3 - 2-way composition - ablation: matched-parameter learner inside each encoder arm**  ·  class **ablation**

| column | mean (seed0 / seed1 / seed2) |
|---|---|
| rival_mlp_180k @ alone | 0.006 (0.000 / 0.017 / 0.000) |
| rival_mlp_180k @ loop_swap | 0.000 (0.000 / 0.000 / 0.000) |
| CLU margin over rival_mlp_180k @ alone | +12.078 (+17.450 / +8.050 / +10.733) |
| CLU margin over rival_mlp_180k @ loop_swap | +10.311 (+10.683 / +7.817 / +12.433) |

> the CLU's margin over the matched-parameter learner DOES NOT GROW. The features are better (or worse) FOR EVERYONE: this is a finding ABOUT THE ENCODER, NOT ABOUT THE STORE.

**row 4 - n-way composition**  ·  class **RESULT**

| column | mean (seed0 / seed1 / seed2) |
|---|---|
| pca_tensor @ encoder ALONE | 24.687 (33.510 / 18.102 / 22.449) |
| pca_tensor @ loop_nosw (unfrozen, NO swap loss) | 22.143 (34.163 / 13.265 / 19.000) |
| pca_tensor @ loop_swap (THE ARM) | 26.048 (32.796 / 18.959 / 26.388) |
| A' = loop_nosw - alone | -2.544 (+0.653 / -4.837 / -3.449) |
| A'' = loop_swap - loop_nosw (what the SWAP LOSS bought) | +3.905 (-1.367 / +5.694 / +7.388) |
| A = loop_swap - alone (the NET) | +1.361 (-0.714 / +0.857 / +3.939) |
| A: bootstrap CI95 low | -1.592 (-3.694 / -1.796 / +0.714) |
| A: n corners (paired) | 49 (49 / 49 / 49) |
| A'': sign-test p (two-sided) | 0.2555 (0.766 / 0.000346 / 0.0001564) |
| A': sign-test p (two-sided) | 0.4525 (0.8854 / 0.00126 / 0.4709) |

**row 4 - n-way composition - ablation: matched-parameter learner inside each encoder arm**  ·  class **ablation**

| column | mean (seed0 / seed1 / seed2) |
|---|---|
| rival_mlp_180k @ alone | 0.000 (0.000 / 0.000 / 0.000) |
| rival_mlp_180k @ loop_swap | 0.000 (0.000 / 0.000 / 0.000) |
| CLU margin over rival_mlp_180k @ alone | +24.687 (+33.510 / +18.102 / +22.449) |
| CLU margin over rival_mlp_180k @ loop_swap | +26.048 (+32.796 / +18.959 / +26.388) |

> the CLU's margin over the matched-parameter learner DOES NOT GROW. The features are better (or worse) FOR EVERYONE: this is a finding ABOUT THE ENCODER, NOT ABOUT THE STORE.

| bar | row 3 | row 4 |
|---|---|---|
| BAR A passed (mean paired delta > +1.0 and CI95 low > +1.0, all three seeds) | False | False |
| BAR B falsifier fired (delta <= 0 on two or more seeds) | True | False |

> **The reading.** The net is a wash because two real effects cancel: unfreezing the encoder with the store on the gradient path HURTS (A′ negative on 5 of 6 seed-rows, two at p ≤ 0.0013) and the swap loss REPAIRS that damage (A″ positive on 4 of 6 seed-rows, two at p ≤ 0.0003). See figure 3.

---

## TABLE 3 — CMAZE-1 at the w9/harness tier - learned baselines and the environment's own columns, 3 seeds

**Pins.** `cell` = CMAZE-1 (env-scout adopted defaults), w9 harness tier `cmaze1` · `condition` = eval split, 200 instances/seed · `generation` = 2 · `seeds` = [0, 1, 2] · `n_train` = 500 · `n_eval` = 200 · `enc_steps` = 1,200 · `bc_steps` = 2,000 · `target_params` = 60,000 · `matched_params` = mlp 60,047, lstm 59,965, transformer 59,819 · `judge_exact` = True

*SEPARATE TABLE BY CHARTER: this tier trains ~60k-parameter arms on 500 instances and evaluates 200, while T1's arms are matched at ~410k parameters over 96 mazes / 1,208 requests. The two are different compute budgets on the same environment and do not share a table.*

| row | class | n (s0/s1/s2) | validity (exact judge) | score | goal rate, budget lifted | legal prefix / optimal |
|---|---|---|---|---|---|---|
| MLP (matched ~60k) | **baseline** | 200 / 200 / 200 | 0.0050 (0.0050 / 0.0000 / 0.0100) | 0.0048 (0.0045 / 0.0000 / 0.0100) | 0.0050 (0.0050 / 0.0000 / 0.0100) | 0.4226 (0.6270 / 0.1554 / 0.4853) |
| LSTM (matched ~60k) | **baseline** | 200 / 200 / 200 | 0.0017 (0.0000 / 0.0000 / 0.0050) | 0.0017 (0.0000 / 0.0000 / 0.0050) | 0.0017 (0.0000 / 0.0000 / 0.0050) | 0.2319 (0.2738 / 0.1083 / 0.3136) |
| Transformer (matched ~60k) | **baseline** | 200 / 200 / 200 | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.1997 (0.2479 / 0.0960 / 0.2551) |
| Table floor (raw state) | **ablation** | 200 / 200 / 200 | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.1664 (0.1974 / 0.1646 / 0.1372) |
| Table-enc (encoder bits) | **ablation** | 200 / 200 / 200 | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0719 (0.1129 / 0.0484 / 0.0543) |
| ORACLE (full composition) | **reference** | 200 / 200 / 200 | 1.0000 (1.0000 / 1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000 / 1.0000) |
| BLIND-A | **reference** | 200 / 200 / 200 | 0.1400 (0.1050 / 0.1400 / 0.1750) | 0.1400 (0.1050 / 0.1400 / 0.1750) | 0.1400 (0.1050 / 0.1400 / 0.1750) | 0.3706 (0.3503 / 0.3780 / 0.3834) |
| BLIND-B | **reference** | 200 / 200 / 200 | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.2225 (0.2234 / 0.2416 / 0.2024) |
| BLIND-COLOUR | **reference** | 200 / 200 / 200 | 0.0933 (0.0850 / 0.0900 / 0.1050) | 0.0931 (0.0850 / 0.0894 / 0.1050) | 0.0983 (0.0900 / 0.0950 / 0.1100) | 0.3070 (0.3082 / 0.3076 / 0.3052) |
| BLIND-FACTOR | **reference** | 200 / 200 / 200 | 0.2350 (0.1850 / 0.2500 / 0.2700) | 0.2350 (0.1850 / 0.2500 / 0.2700) | 0.2350 (0.1850 / 0.2500 / 0.2700) | 0.4609 (0.4335 / 0.4742 / 0.4748) |
| BLIND-SEQUENTIAL | **reference** | 200 / 200 / 200 | 0.5317 (0.5550 / 0.5300 / 0.5100) | 0.5314 (0.5547 / 0.5296 / 0.5100) | 0.5433 (0.5650 / 0.5450 / 0.5200) | 0.5433 (0.5650 / 0.5450 / 0.5200) |
| FRAGMENT-LOOKUP r=1 | **reference** | 200 / 200 / 200 | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.3231 (0.2946 / 0.3475 / 0.3272) |
| FRAGMENT-LOOKUP r=2 | **reference** | 200 / 200 / 200 | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0392 (0.0375 / 0.0369 / 0.0432) |

> ⭐ **The published tripwire reproduces HERE.** BLIND-SEQUENTIAL is **0.5317 (0.5550 / 0.5300 / 0.5100)** at this tier against the published CMAZE-1 **0.4980**, and FRAGMENT-LOOKUP r=1 and r=2 are both **0.0000** on every seed. Compare Table 1, where the same column reads **0.9037** on the hub's unseen-pair split. ⛔ The difference is the REQUEST LAYER, not the environment — see correction C4.

**DISPLAY comparator at this tier**

| row | class | refusal on IMPOSSIBLE (correct) | refusal on POSSIBLE (false) |
|---|---|---|---|
| supervised refuser (the only DISPLAY comparator), decision at t=0 | **DISPLAY** | 0.8317 (0.8200 / 1.0000 / 0.6750) | 0.7850 (0.7950 / 1.0000 / 0.5600) |

---

## TABLE 4 — Exact deletion - two independent implementations, 3 seeds each, TEST condition

*Both legs are 3 seeds on generation 2 on their own TEST condition. They are reported side by side as two implementations of one claim, not as one instrument: the deviations are in different units (w9/maze reports a RELATIVE residual on the statistic and a projector residual; w8/downdate reports an ABSOLUTE max deviation per well).*

**w9/maze `CluRead.delete_rows_exact` - delete the green target, TEST leg**  ·  class **RESULT**  ·  implementation: w9 hub/maze

| property | seed 0 / seed 1 / seed 2 |
|---|---|
| statistic max relative residual (downdate vs refit) | 7.522e-16 / 8.826e-16 / 7.818e-16 |
| subspace max projector residual | 0.000e+00 / 0.000e+00 / 0.000e+00 |
| rows removed | 216 / 226 / 193 |
| rows kept | 992 / 975 / 1084 |
| refit performed | False / False / False |
| passes over retained data | 0 / 0 / 0 |
| items kept | False / False / False |
| address-book bytes deployed by the basis-change carry | 0 / 0 / 0 |

**w8/downdate `exp_w8_downdate` - covariance downdate == refit-without-those-rows, TEST leg**  ·  class **RESULT**  ·  implementation: w8 spoke/downdate-build (different code, different author)

| property | seed 0 / seed 1 / seed 2 |
|---|---|
| max abs deviation, worst well | 2.365e-14 / 1.126e-09 / 7.105e-15 |
| negative control fired on every seed | True / True / True |
| cov bytes (statistic, declared beside the pin) | 12,310,672 / 11,941,360 / 11,850,912 |
| no pass over data | True / True / True |
| items kept | False / False / False |

> Artefact's own summary: `passed_every_seed` = True · `worst_max_abs_dev_over_seeds` = 1.1255e-09 · `worst_in_float32_ulp_at_unit_scale` = 0.009442

See figure 4.

---

## DECLARED NOT-RUN register — never a blank, never a zero

### NR1 — Shapes3D generation cell (incumbent instrument: K=64, A=1,216 NMF atoms, shared-basis head, eval_b, composition-B validity) at 3 seeds

**Status:** NOT RUN at 3 seeds - SEED 0 ONLY EXISTS

**What does exist:** one scored cell, seed 0: arm 0.8495 / 51 tuples of 72 at 454,336 B against knn_proto 0.8588 / 57 at 475,136 B; delta -0.0093, TIE under R1

**Why:** hub/generation-table-2 spent scored run 1 of 3 on seed 0; run 3 is still unspent. A 1-seed row cannot enter a 3-seed table and is not averaged into one.

**Evidence:**
- `/c/chlu/outputs/hub-generation-table/LOG-iter2.md` — 16:12Z scored-run table
- `/c/chlu/outputs/hub-generation-table/relaunch/scored/scored_spec.json` — /seed
- `/c/chlu/evidence/outputs/w8-advisor-report.md` — Budget after wave 8

**Launch line:**

```bash
# for SEED in 1 2 ; two legs per seed, from $CHLU_ROOT
python -m chlu.experiments.exp_gen_cell --stage cell --seed $SEED
jq '.seed=$SEED' /c/chlu/outputs/hub-generation-table/relaunch/scored/scored_spec.json > /tmp/scored_spec_s$SEED.json
python /c/chlu/outputs/hub-generation-table/relaunch/scored/run_arm.py <out_dir>/seed$SEED /tmp/scored_spec_s$SEED.json
```

**Cost:** leg 1 814 s + leg 2 277 s on FORGIS_14 CPU for seed 0; two more seeds ~ 40 min CPU total, but each consumes one of the remaining `generation-table` scored runs (1 left).

### NR2 — Shapes3D generation cell at the GPU compute pin (joint_steps=4000, n_train=32768, dec_steps=1800)

**Status:** NOT RUN - deferred, spec ready, blocked on the Docker daemon on the Head's Mac

**Why:** the CPU leg was forced to joint_steps=400 and the shortened `w8_v1_short` decoder; at that pin BAR D is VOID (the 400-step decoder pins the MSE-regression baseline to exactly 0.000 where the incumbent's 1800-step decoder gives it 53-60) and C.1 is weak evidence only. This is a SECOND TABLE at a different budget, not extra rows for T2.

**Evidence:**
- `/c/chlu/outputs/w8-encoder-loop/cloud/LAUNCH-SPEC.md` — SS1 launch block, SS5 cost table
- `/c/chlu/outputs/w8-encoder-loop/LOG.md` — FINAL REPORT / DECLARED NOT-RUN

**Launch line:**

```bash
export CHLU_W8_CKPT_REMOTE=gs://chlu-cloud-forgisprova/jobs
for SEED in 0 1 2; do
  scripts/cloud/run_until_done.sh --job "w8enc-s${SEED}" --card a100-40 --spot \
      --exp w8_encoder_loop --units "${SEED}" \
      --artefacts gs://chlu-cloud-forgisprova/artefacts \
      --exp-args "--stage cloud --ckpt-every-s 300 --joint-steps 4000 --dec-steps 1800 --warm-steps 4000 --n-train 32768" \
      --max-hours 6 --max-attempts 8 --max-usd 12.00
done
```

**Cost:** $26.46 for 3 seeds at A100-40 SPOT $1.47/h; A100 concurrency is ONE, so the seeds are sequential. `kill -9` resumability proved (resumed leg reached bit-identical parameters).

### NR3 — w8 single-design object (covariance store that composes AND forgets) - RESULT rows 3/4 and post-deletion composition/refusal at 3 seeds on TEST

**Status:** NOT RUN - 0 of 3 scored, full budget intact

**Why:** the object was falsified on `val` before a scored run was justified: the covariance costs sum_k d_k^2, fitting it inside the 524,288 B pin forces K >= 36, and at K=40/A=7 (519,794 B = 0.991x the pin) composition collapses to valid mean 0.0648. At K=5 composition holds at 0.8918 (incumbent 0.8941) but the honest four-part ledger is 3,922,450 B = 7.48x the pin. Byte pin is primary at tolerance 0.

**Evidence:**
- `/c/chlu/outputs/w8-single-design/LOG.md` — FINAL REPORT (REVISED) SS1-SS3

**Launch line:**

```bash
# would require a content whose exact deletion does NOT need a dxd statistic; no such
# object exists on disk. The nearest runnable leg is the ward_balanced K=5 object at
# 7.48x the byte pin, which cannot carry a RESULT row under the pin-primary rule:
python -m chlu.experiments.exp_w8_single_design --stage score --partition ward_balanced --k 5 --condition test --seed {0,1,2}
```

**Cost:** declared NOT-RUN is not a null: the val falsification is the wave-8 answer.

### NR4 — hub/maze-encoder-loop - the w8 encoder-loop comparison repeated on CMAZE-1

**Status:** NOT RUN - contingent, and the contingency did not fire

**Why:** the wave plan commissions it only `if w8's result was positive; otherwise a declared NOT-RUN`. w8/encoder-loop's BAR B falsifier FIRED on row 3 and the net deltas straddle zero on row 4, so the contingency is unmet.

**Evidence:**
- `/c/chlu/evidence/outputs/advisor-wave-plan-w8-w10.md` — SS2 hub/maze-encoder-loop
- `/c/chlu/outputs/w8-encoder-loop/CROSS-ARM.json` — /VERDICT

**Launch line:**

```bash
# only if a positive encoder-loop result appears at the GPU pin (NR2):
python -m chlu.experiments.exp_w9_maze --stage score --encoder-arm {alone,loop_swap} --condition test --seed {0,1,2}
```

**Cost:** no maze-encoder-loop budget was drawn.

### NR5 — w9/maze post-deletion composition with the REPAIRED deletion, on TEST at 3 seeds

**Status:** NOT RUN on TEST - measured on VAL, seed 0 only (Amendment 1)

**Why:** the scored budget was 3/3 spent before the repair existed. The as-pre-registered scored-leg DISPLAY row stands (surviving targets fall to 0.0000 on all three seeds); the repaired mechanism was re-measured on VAL seed 0 only, at -0.0120, inside BAR 6's +-0.05.

**Evidence:**
- `/c/chlu/outputs/w9-maze/runs/amend_exactdelete_seed0_val.json` — /display/delete_green/composition_without_retraining/validity_delta_on_other_targets
- `/c/chlu/outputs/w9-maze/LOG.md` — 09:55Z AMENDMENT 1
- `/c/chlu/outputs/w9-maze/LEDGER.md` — row A1

**Launch line:**

```bash
python -m chlu.experiments.exp_w9_maze --stage score --exact-head-delete --condition test --seed {0,1,2}   # needs a 4th scored run; budget is 3/3 spent
```

**Cost:** ~240 s/seed on FORGIS_14 CPU, but it is a HELD-OUT number and the maze budget is exhausted.

---

## CORRECTIONS found while tracing every cell

Each is a place where a narrative number in a LOG, a report or the task file does not match the artefact it is drawn from. None of them changes a verdict; all of them change a number the paper would otherwise print.

### C1

- **As written:** BLIND-SEQUENTIAL on this hub's split is `0.9037-0.9130` (w10 task file; w9-maze LOG.md 09:40Z bullet quotes 0.9130 for seed 2)
- **Artefact says:** unseen-pair BLIND-SEQUENTIAL = 0.9077 / 0.8905 / 0.9128 for seeds 0/1/2, mean 0.9037; n-way BLIND-SEQUENTIAL = 0.9229 / 0.8933 / 0.8841, mean 0.9001
- **Effect:** the quoted band's upper end is 0.9128 not 0.9130, and its lower end is the SEED MINIMUM 0.8905, not the 3-seed mean 0.9037. The band as written mixes a mean with a per-seed value.
- **What this file does:** T1 prints the three per-seed values; the mean is labelled as a mean.
- **Traced to:** `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity`; `/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json` `/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity`; `/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` `/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity`

### C2

- **As written:** w8 advisor report: exact deletion `downdate == refit ... to 1e-16...1e-14`, 9/9 seeds
- **Artefact says:** DOWNDATE-READY-test.json worst_max_abs_dev_over_seeds = 1.1255e-09 (seed 1, well 3); DOWNDATE-READY.json (val) worst_max_abs_dev = 1.1642e-10
- **Effect:** the stated range understates the worst observed deviation by ~5 orders of magnitude. The claim still holds in the units the artefact itself uses for the bar (worst_in_float32_ulp_at_unit_scale = 0.0094, i.e. under one float32 ulp), but the paper must quote the ulp figure or the true absolute, not `1e-14`.
- **What this file does:** T4 prints per-seed absolutes and the ulp figure side by side.
- **Traced to:** `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/downdate_equals_refit`; `/c/chlu/outputs/w8-downdate/DOWNDATE-READY.json` `/downdate_equals_refit`

### C3

- **As written:** the n-way row and its ceiling (w9 charter warning)
- **Artefact says:** the scored n-way row has n = 183 / 192 / 191 (a CONJUNCTION over the routes of one request), while pipeline_ceiling_rows.nway has n = 389 / 403 / 414 (PER ROUTE)
- **Effect:** the row and its printed ceiling are computed over different denominators, so 0.0284 vs 0.5255 is not a like-for-like fraction-of-ceiling.
- **What this file does:** T1 carries row 3 TWICE - conjunction and per route - and prints n on both.
- **Traced to:** `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/table/nway/clu/n`; `/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json` `/table/nway/clu/n`; `/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` `/table/nway/clu/n` …

### C4

- **As written:** w8 advisor report SS5.5: `Wave 9 inherits the 0.4980 blind-sequential column as a caveat`
- **Artefact says:** BLIND-SEQUENTIAL is 0.5317 (0.5550 / 0.5300 / 0.5100) at the w9/harness tier - which DOES reproduce the published 0.4980 - and 0.9037 (0.9077 / 0.8905 / 0.9128) on the w9/maze hub's unseen-pair split
- **Effect:** the caveat is not a property of CMAZE-1 that wave 9 `inherits`; it is a property of the REQUEST LAYER. The published tripwire survives intact where the harness draws fresh instances, and is destroyed only by the hub's `unseen pair on a SEEN maze` request layer, where 1,242 of 1,449 optimal routes pick up exactly one key and there is no key ORDER for rule B to trap. Two different numbers, two different splits, and the paper must not print one where the other belongs.
- **What this file does:** T1 and T3 each print the value measured on their own split; figure 5 draws both against the published 0.4980.
- **Traced to:** `/c/chlu/outputs/w9-harness/harness_seed0.json` `/columns/blind-sequential (key order from A)/validity`; `/c/chlu/outputs/w9-harness/harness_seed1.json` `/columns/blind-sequential (key order from A)/validity`; `/c/chlu/outputs/w9-harness/harness_seed2.json` `/columns/blind-sequential (key order from A)/validity` …

---

## FIGURES — all regenerated from the manifest by `python build_tables.py`

| figure | file | what it shows | sources |
|---|---|---|---|
| fig1 | `figures/fig1_cmaze1_main_scoreboard.png` | T1 RESULT rows 1-4: arm, learned baselines and the matched-bytes floor, 3 seeds as dots, pipeline ceiling per row | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` |
| fig2 | `figures/fig2_nway_conjunction_vs_route.png` | Row 3 under both readings against the per-route ceiling | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` |
| fig3 | `figures/fig3_encoder_loop_decomposition.png` | T2: A' / A'' / A per seed on rows 3 and 4 | `/c/chlu/outputs/w8-encoder-loop/CROSS-ARM.json` |
| fig4 | `figures/fig4_exact_deletion.png` | T4: both deletion implementations, log scale | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json`<br>`/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` |
| fig5 | `figures/fig5_blind_sequential_on_this_split.png` | T1 caveat column: BLIND-SEQUENTIAL on this split vs the published 0.4980 | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json`<br>`/c/chlu/outputs/w8-env-scout/screen_results_DEFAULTS.txt` |

---

## SOURCE REGISTER — every file this spoke read

| id | path | bytes | sha256 (first 16) |
|---|---|---|---|
| maze_s0 | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` | 37,201 | 5dc435ead5df60e6 |
| maze_s1 | `/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json` | 37,239 | 0cdaa0501d9e2c3d |
| maze_s2 | `/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` | 37,230 | d9aab888a467391f |
| maze_amend_val_s0 | `/c/chlu/outputs/w9-maze/runs/amend_exactdelete_seed0_val.json` | 30,534 | b2eace4499302abc |
| maze_replica_s0 | `/c/chlu/outputs/w9-maze/replication/replica_seed0_test.json` | 38,358 | 66befda2ec12e64c |
| maze_replication_md | `/c/chlu/outputs/w9-maze/replication/REPLICATION.md` | 13,142 | 7f4ceab322779b12 |
| maze_ledger | `/c/chlu/outputs/w9-maze/LEDGER.md` | 2,197 | 4f04f52f118f7f10 |
| maze_log | `/c/chlu/outputs/w9-maze/LOG.md` | 44,541 | 84251f7b9e8c4e01 |
| maze_prereg1 | `/c/chlu/outputs/w9-maze/PREREG-w9-maze-1.md` | 7,083 | e4a9bd0127b17adc |
| enc_crossarm | `/c/chlu/outputs/w8-encoder-loop/CROSS-ARM.json` | 48,622 | 299b5cbe6ec49ea0 |
| enc_report | `/c/chlu/outputs/w8-encoder-loop/run/encoder_loop_report.json` | 19,334 | 26f2cfd2e9db8723 |
| enc_log | `/c/chlu/outputs/w8-encoder-loop/LOG.md` | 74,147 | a7e266a96764928d |
| enc_launchspec | `/c/chlu/outputs/w8-encoder-loop/cloud/LAUNCH-SPEC.md` | 18,007 | e0495e8976739799 |
| harness_s0 | `/c/chlu/outputs/w9-harness/harness_seed0.json` | 24,811 | 7dfaa35d7099fe74 |
| harness_s1 | `/c/chlu/outputs/w9-harness/harness_seed1.json` | 23,645 | a10ccef81267a88f |
| harness_s2 | `/c/chlu/outputs/w9-harness/harness_seed2.json` | 24,855 | e6f98bab023ee2f7 |
| harness_results | `/c/chlu/outputs/w9-harness/RESULTS.md` | 6,609 | 40374ab9a0dc3c2c |
| downdate_test | `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` | 29,375 | c4dae614c88dc650 |
| downdate_val | `/c/chlu/outputs/w8-downdate/DOWNDATE-READY.json` | 8,823 | ac6eabcf19bbe3fa |
| single_design_log | `/c/chlu/outputs/w8-single-design/LOG.md` | 126,149 | 83740a7ae62ef691 |
| gentable_spec | `/c/chlu/outputs/hub-generation-table/relaunch/scored/scored_spec.json` | 745 | 7990c75559ff9393 |
| gentable_log | `/c/chlu/outputs/hub-generation-table/LOG-iter2.md` | 100,346 | e68f74abec35b63b |
| gentable_runarm | `/c/chlu/outputs/hub-generation-table/relaunch/scored/run_arm.py` | 22,263 | 065a0ecddaa55511 |
| envscout_env | `/c/chlu/outputs/w8-env-scout/ENV.md` | 30,128 | 400af82465c7b375 |
| envscout_defaults | `/c/chlu/outputs/w8-env-scout/screen_results_DEFAULTS.txt` | 4,521 | 600583bbcb2e61e9 |
| wave_plan | `/c/chlu/evidence/outputs/advisor-wave-plan-w8-w10.md` | 6,408 | c6c08a4237f985d3 |
| w8_advisor | `/c/chlu/evidence/outputs/w8-advisor-report.md` | 8,698 | 166df363cdeb5b0a |

⛔ **Nothing outside this register was read**, and in particular `$CHLU_CLAUDE_ROOT` was not read at any point — it is a frozen older snapshot.

