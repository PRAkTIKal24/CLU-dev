# TABLES.md — the paper's scoreboard, with the seed count printed on every row

**Built** 2026-09-21T17:15:06Z by `build_tables.py` in this directory. **Scored runs spent by this spoke: ZERO** — it re-reads artefacts only.

**How to trace any cell.** Every number below is written `mean (seed0 / seed1 / seed2)`. `TABLES.json` carries the same cell with a `provenance` list — one `{source, path, pointer}` entry per seed — and `sources` maps every source id to its absolute path, byte count and sha256. `python build_tables.py` regenerates this file, `TABLES.json` and every figure from those paths alone.

**The rules this file is built under** (from the w10 charter):
- All three seeds or none - no row reports a best-of.
- Runs at different compute budgets do not share a table - T1/T2/T3/T4 are four pins.
- w8/encoder-loop absolutes (shortened `w8_v1_short` decoder) do not share a table with the incumbent's.
- CMAZE-2's n-way row is reported as a CONJUNCTION and PER ROUTE, because its ceilings are per route.
- Every CMAZE-2 composition row carries BLIND-SEQUENTIAL measured on its own split, not the published 0.4980; where no BLIND block exists (row 4) that is DECLARED, never blank.
- w12r/MF10: T1 is CMAZE-2 (a request layer over CMAZE-1) and T3 is CMAZE-1 proper. They are a DIFFERENT ENVIRONMENT, not merely a different budget, and they never share a table.
- w12r/MF3: every table that carries a store prints its FOUR-PART byte ledger - atoms, covariance/count ledger, address book, coupling/hole table - separately, NEVER netted into one number.
- w12r/MF4: `exact` is STORE-SCOPED and the frozen encoder and decoder are never refit; the definition is a citable field at TABLES.json /exactness_definition.
- w12r: never `9/9 seeds`. Seed count and configuration count are separate fields, and the three implementations are three rows (T5).
- A row that was not run is a DECLARED NOT-RUN with a reason and a launch line, never a blank or a zero.
- w13: n (jointly composed factors, the ladder rungs) and K (wells, the partition) are NEVER both written k. Every ladder row states its K; every wells row states its n. See /notation.
- w13: the seed count is a COLUMN of every table, never a footnote, and rows at different seed counts are never plotted together without it. See /seed_ledger.
- w13: BOTH byte matchings are printed on every row that has them - the object's own measured total (1x) and the 524,288 B reference - because they tell opposite stories about the n-trend.
- w13: T7 (K = 32, 1 seed, TEST) and T8 (K = 5, 3 seeds) are different objects at different pins and never share a table or a figure axis.
- w13: an artefact that labels itself SMOKE_NOT_A_CLAIM is rendered as SMOKE (T9b, T12), never as a RESULT.

---

## ⭐ DEFINITION — what `exact` means in this paper (`TABLES.json` `/exactness_definition`)

> **STORE-SCOPED.** EXACT means: after deleting a set of rows, the STORE CONTENT equals, to float roundoff, the content fitted from the retained rows alone, HOLDING THE FROZEN ENCODER AND DECODER FIXED. Neither encoder nor decoder is refit at any point. It is NOT a claim that the renderer forgets, and it is NOT retrain-equivalence of the generative model.

**What it is NOT:**
- ⛔ NOT retrain-equivalence of the generative model - the renderer is never refit
- ⛔ NOT cost-free - on TEST the same deletion takes surviving-target composition from 0.1024 to 0.0000
- ⛔ NOT a byte-level claim about the address book - the basis-change carry rewrites the book and is declared separately

**Why only this scope is supported.** The frozen autoencoder's own training set contains eval-bank items, so the encoder and decoder weights fitted with the deleted rows present are untouched by the deletion.

| evidence | path | pointer / locator |
|---|---|---|
| frozen autoencoder training inventory, seed 0: eval_bank_train 543 / pool 4860 / pool_sites 270 | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` | `/ae_train_items` |
| seed 1: eval_bank_train 529 | `/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json` | `/ae_train_items` |
| seed 2: eval_bank_train 552 | `/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` | `/ae_train_items` |
| every arm is rendered by the same FROZEN DECODER through the same latent standardisation | `/c/chlu/outputs/w9-maze/LOG.md` | `LOG.md:376` |
| refit = False - the deletion refits nothing | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` | `/display/delete_green_exactness/refit` |

⛔ **Every cell whose label carries the word `exact` is scoped by this definition.** In this file that is: T1's RESULT rows and every ceiling row (`validity under the exact judge` — a different sense: the JUDGE is exact, see 1e); T1 §1d's post-deletion rows; all of TABLE 4; all of TABLE 5. The two senses are kept apart on purpose: **an exact JUDGE** is integer arithmetic with no tolerance (`/judge_audit/exact` = True), while **exact DELETION** is the store-scoped definition above.

## ⭐ DEFINITION — CMAZE-1 and CMAZE-2 are two environments, not one

| environment | what it is | tables |
|---|---|---|
| **CMAZE-1** | the env-scout adopted colour-maze generator `w8_envscout.maze_colour`, called with its own defaults | T3 |
| **CMAZE-2** | a REQUEST LAYER over CMAZE-1: the engine, the solver and the exact integer judge are `w8_envscout.maze_colour`, IMPORTED UNMODIFIED; the layer adds named agents and named targets inside one maze, placed across two door-free components, and seals one target behind walls per maze | T1, T4 (the w9/maze leg), T5 (implementation 1) |

> ⛔⛔ **T1 (CMAZE-2) and T3 (CMAZE-1) MUST NEVER SHARE A TABLE. They are a DIFFERENT ENVIRONMENT, not merely a different budget.**

**The three numbers only read coherently once they are separated:**

| BLIND-SEQUENTIAL | value | where |
|---|---|---|
| published CMAZE-1 | **0.4980** | `/c/chlu/outputs/w8-env-scout/ENV.md` lines 42, 49, 358 |
| CMAZE-1 at the w9/harness tier | **0.5317**  (0.5550 / 0.5300 / 0.5100) | `/c/chlu/outputs/w9-harness/harness_seed0.json` `/columns/blind-sequential (key order from A)/validity` — TABLE 3 |
| CMAZE-2 | **0.9037**  (0.9077 / 0.8905 / 0.9128) | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity` — TABLE 1 |

## ⛔ NEVER `9/9 SEEDS` — seed count and configuration count are separate fields

> The phrase '9/9 seeds' is wrong as written and must never be printed. The nine legs are 3 CONFIGURATIONS x 3 SEEDS inside ONE implementation. Seed count and configuration count are recorded here as SEPARATE fields so no sentence can multiply them into a seed count.

| implementation | seeds | configurations | legs | scored runs | configuration |
|---|---|---|---|---|---|
| w9/maze CluRead.delete_rows_exact | 3 | 1 | 3 | 3 | K=8, n_atoms=31 - the one pinned CMAZE-2 configuration |
| w8 spoke/downdate-build exp_w8_downdate | 3 | 1 | 3 | 0 | factor partition K=5, A=64 (the incumbent's grain) on Shapes3D features |
| w8 hub/single-design matched-bytes candidates | 3 | 3 | 9 | 0 | ward_balanced K=40 / K=44 / K=48 - THIS is where the phrase '9/9' comes from |

**Correct phrasings:** 3 seeds x 3 configurations inside ONE implementation (hub/single-design) = 9 legs · 3 seeds at ONE configuration in each of the two other implementations · THREE implementations in total, printed as three rows - see T5

---

## TABLE 1 (MAIN) — MAIN SCOREBOARD - CMAZE-2 (w9 hub/maze), held-out TEST, 3 of 3 scored seeds

**Pins.** `cell` = CMAZE-2 (a REQUEST LAYER over CMAZE-1: engine, solver and exact integer judge imported unmodified from `w8_envscout.maze_colour`), hub/maze · `environment` = CMAZE-2 · `environment_is_NOT` = CMAZE-1 - see TABLES.json /environments; T3 is CMAZE-1 proper · `condition` = test (held out) · `generation` = 2 · `seeds` = [0, 1, 2] · `scored_runs` = 3 of 3 (S1/S2/S3, pre-registered before execution) · `K` = 8 · `n_atoms` = 31 · `ridge` = 0.01 · `ae_steps` = 12,000 · `baseline_steps` = 4,000 · `latent_d` = 256 · `read_floats` = 410,688 · `deployed_bytes` = 1,642,752 · `statistic_bytes_float64_declared_beside` = 35,904 · `rank_over_width` = 0.9375 · `matched_params` = mlp 410,920, lstm 409,846, transformer 405,776 · `n_mazes` = 96

*THE ENVIRONMENT IS CMAZE-2: a REQUEST LAYER over CMAZE-1, with the engine, the solver and the exact integer judge imported unmodified from `w8_envscout.maze_colour`. It is NOT CMAZE-1, and it must never share a table with T3, which is CMAZE-1 proper - a different environment, not merely a different budget. Every cell is the mean of scored seeds 0/1/2 with the three per-seed values in brackets. Pins: K=8, n_atoms=31, ridge=0.01, AE 12,000 steps, baselines 4,000 steps, read = 410,688 floats (1,642,752 B deployed) with the float64 sufficient statistic (35,904 B) declared beside it and never netted in; matched parameters mlp 410,920 / lstm 409,846 / transformer 405,776. The four-part byte ledger is printed as its own block below and is never netted into one number. Row 3 is reported BOTH as the scored CONJUNCTION and PER ROUTE, because the ceilings are per route.*

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

> ⛔ **BAR 3 (the C.2 controls) FAILS and that is the mechanism of this table.** On row 1 `landscape_shuffled` reads 0.9509 and `random_atoms` 0.9601 against the CLU's 1.0000; on row 2 `random_atoms` (0.1414) is inside the seed spread of the CLU (0.1279). Only `landscape_removed` collapses to 0.0000. ⛔ **Under this hub's own pre-registration no RESULT row is claimed for the CLU arm on CMAZE-2.** BAR 1 also fails on all three composition rows: against the best learned baseline the CLU is behind by -0.0560 (row 2), -0.0209 (row 3 conjunction) and -0.0434 (row 4). Traced to `/c/chlu/outputs/w9-maze/LOG.md` §5.

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

### 1c-bis. ⛔ FRAGMENT-LOOKUP — same-maze and cross-maze are DIFFERENT QUANTITIES

> 'FRAGMENT-LOOKUP r=1/r=2 at 0.0000 everywhere' is TRUE of the CROSS-MAZE quantity and FALSE of the SAME-MAZE quantity on this hub's own split. Both are printed here, labelled, so no sentence can quote one as the other.

| column | r=1 | r=2 |
|---|---|---|
| FRAGMENT-LOOKUP, CROSS-MAZE (the published CMAZE-1 tripwire; the split does NOT leak) | 0.0006 (0.0017 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) |
| FRAGMENT-LOOKUP, SAME-MAZE (a DIFFERENT quantity, and NOT a like-for-like arm) | 0.0741 (0.0599 / 0.0960 / 0.0665) | 0.5114 (0.5342 / 0.5364 / 0.4636) |

> ⛔ **Not a like-for-like arm.** FRAG reads the TRUE MAZE at every step, which charter B.3 forbids the CLU. The same-maze table is therefore a floor on what a stored-route table achieves when it may look, not a comparable arm.

> **Why the same-maze number is high.** this hub's rows are unseen PAIRS on SEEN mazes, so a fragment table contains routes through the very same maze, and a 5x5 colour-canonicalised patch is close to a position index in a 15x15 grid

> ⭐ **same-maze r=2 reaches 0.5105 against row 2's pipeline ceiling of 0.6319 - 81% of the pipeline ceiling, where the CLU reaches 20% and the best learned baseline 29%. It is the strongest single exhibit for 'every piece of every answer is already written'.**

> ⚠ **Which mean?** the three per-seed r=2 validities are 0.5342 / 0.5364 / 0.4636 at n = 584 / 604 / 632. This file prints the UNWEIGHTED seed mean, 0.5114. The 0.5105 quoted by `w9-maze/LOG.md` and by REFEREE-v0.md is the n-WEIGHTED (pooled) mean, 0.51044. See correction C11.

### 1c-ter. ⛔ DECLARED NOT MEASURED — the charter rule that is unsatisfiable for row 4

| row | column | status | why |
|---|---|---|---|
| row 4 - budget interpolation | BLIND-SEQUENTIAL (and every other BLIND column) | **DECLARED NOT MEASURED - not a blank, not a zero** | `/env_columns` carries exactly four keys - `unseen_pair`, `nway`, `fragment`, `fragment_cross_maze` - on every scored seed. There is no blind block for the budget-interpolation row on any seed, so the charter rule is unsatisfiable for row 4 and is declared rather than silently skipped. |

### 1c-quater. ⛔ THE BYTE LEDGER — four parts, printed, never netted

**Byte ledger — FOUR PARTS, PRINTED SEPARATELY, NEVER NETTED INTO ONE NUMBER.**

| part (seed 0 / seed 1 / seed 2) | bytes | how it is obtained |
|---|---|---|
| 1. atoms (the store) | 31,744 / 31,744 / 31,744 | store_atom_floats x 4 (float32) |
| 2. sufficient statistic (covariance / count ledger), float64 | 35,904 (35,904 / 35,904 / 35,904) | stored field |
| 3. address book (the head) | 1,611,008 / 1,611,008 / 1,611,008 | head_floats x 4 (float32) |
| 4. coupling / hole table | **DECLARED NOT PRESENT - not a zero and not a blank** | no coupling table and no hole table exists in the w9/maze artefacts: the three scored run files carry no field matching 'hole' or 'coupl'. The w9 leg's deletion is a downdate of (S, t, n) and of the head statistics (A, Mx); it has no interface hole table of the kind the w8 leg carries. |
| **deployed (atoms + address book) - what produces a read** | **1,642,752 (1,642,752 / 1,642,752 / 1,642,752)** | stored field |
| **honest total (deployed + statistic)** | **1,678,656 / 1,678,656 / 1,678,656** | deployed_bytes + statistic_bytes |
| **statistic as a share of deployed** | **2.19% / 2.19% / 2.19%** | statistic_bytes / deployed_bytes |

> ⛔ **pin_status:** T1 IS NOT UNDER THE 524,288 B PIN. That pin belongs to the Shapes3D arms (T2, T4's w8 leg). T1's budget is the MATCHED-PARAMETER budget: mlp 410,920 / lstm 409,846 / transformer 405,776 against the CLU's read of 410,688 floats. Do not print a `x the pin` figure for this table.

> The artefact's own words: *"deployed bytes are the atoms plus the address book, which is what produces a read; the float64 sufficient statistic is what buys EXACT deletion and is declared beside them, never netted in"*


### 1d. DISPLAY rows — inductive bias; the only comparator is the supervised refuser

**Refusal.** The CLU has never seen an impossible example; the supervised refuser is trained on 552 / 562 / 522 of them (seeds 0/1/2) and is evaluated on held-out mazes 48..95 only, so the like-for-like CLU column is the held-out one. Lower is better on the grounded-possible row (those are FALSE refusals).

| row | class | n (s0/s1/s2) | CLU refusal rate (all mazes) | CLU refusal rate (held-out mazes 48..95) | supervised refuser (the only comparator) | CLU validity when it acts (all mazes) |
|---|---|---|---|---|---|---|
| refusal - budget below the shortest legal route | **DISPLAY** | 280 / 260 / 232 | 0.9601 (0.9714 / 0.9692 / 0.9397) | 0.9537 (0.9669 / 0.9669 / 0.9274) | 0.4972 (0.5099 / 0.5702 / 0.4113) | 0.0000 (0.0000 / 0.0000 / 0.0000) |
| refusal - target walled off | **DISPLAY** | 837 / 846 / 837 | 0.9801 (0.9606 / 0.9846 / 0.9952) | 0.9688 (0.9348 / 0.9740 / 0.9976) | 0.2995 (0.3357 / 0.3050 / 0.2577) | 0.0000 (0.0000 / 0.0000 / 0.0000) |
| refusal - unseen (agent,target) novelty control | **DISPLAY** | 573 / 534 / 644 | 0.9781 (0.9721 / 0.9794 / 0.9829) | 0.9735 (0.9559 / 0.9891 / 0.9754) | 0.2008 (0.1140 / 0.2545 / 0.2338) | 0.0012 (0.0035 / 0.0000 / 0.0000) |
| refusal - grounded-possible  (FALSE refusals - lower is better) | **DISPLAY** | 584 / 604 / 632 | 0.8182 (0.7979 / 0.8560 / 0.8006) | 0.7986 (0.7544 / 0.8296 / 0.8117) | 0.2248 (0.2384 / 0.2412 / 0.1948) | 0.0560 (0.0616 / 0.0414 / 0.0649) |

> ⛔ **THE READING OF THESE THREE ROWS, which the registry filed and the draft dropped (w12r/MF15, registry `N-I8-19`, tier A).** *Refusal is NOVELTY DETECTION, not wall detection.* The store refuses the **reachable-but-unwritten** combination (`mt_unseen`, **0.9781** = 0.9721 / 0.9794 / 0.9829) as often as the **walled** one (`walled`, **0.9801** = 0.9606 / 0.9846 / 0.9952) — both rows are in the table above — and its false-refusal rate on grounded-possible requests is **0.8182 against a pre-registered 0.20**, a 4x miss. The supervised refuser, which HAS seen impossible examples, separates better. ⛔ **No refusal win is claimed.** Printing the three DISPLAY rows without this reading is what the registry asked the paper not to do. *Traced to:* `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/display/refusal/*/refusal` and `/display/clu_refusal_rate_heldout_mazes/*/refusal`.

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

**Byte ledger — FOUR PARTS, PRINTED SEPARATELY, NEVER NETTED INTO ONE NUMBER.**

| part (seed 0 / seed 1 / seed 2) | bytes | how it is obtained |
|---|---|---|
| 1. atoms (the store - 'the modes ARE the store') | 524,288 / 524,288 / 524,288 | stored field |
| 2. sufficient statistic (covariance / count ledger) | **DECLARED NOT PRESENT - not a zero and not a blank** | this instrument retains no per-well covariance and makes NO deletion claim. The exact-deletion claim lives in T4 and T5, not here. |
| 3. address book | 195,064 / 195,064 / 195,064 | byte-ledger assertion, per arm per seed: store 524,288 B exact, address book 195,064 B declared BESIDE it and never netted in |
| 4. coupling / hole table | **DECLARED NOT PRESENT** | no coupling or hole table exists on this instrument; there is no deletion interface here. |
| **honest total (atoms + address book)** | **719,352 / 719,352 / 719,352** | store_bytes + address_book_bytes(195,064, markdown locator) |
| **x the 524,288 B pin, store only** | **1.0000x / 1.0000x / 1.0000x** | store_bytes / 524288 |

> The artefact's own words: *"The modes ARE the store. The addressing head is reported separately and is not part of this count; a byte-matched baseline matches store_bytes."*


> **The reading.** The net is a wash because two real effects cancel: unfreezing the encoder with the store on the gradient path HURTS (A′ negative on 5 of 6 seed-rows, two at p ≤ 0.0013) and the swap loss REPAIRS that damage (A″ positive on 4 of 6 seed-rows, two at p ≤ 0.0003). See figure 3.

---

## TABLE 3 — CMAZE-1 PROPER at the w9/harness tier - learned baselines and the environment's own columns, 3 seeds

**Pins.** `cell` = CMAZE-1 PROPER (env-scout adopted defaults), w9 harness tier `cmaze1` · `environment` = CMAZE-1 · `condition` = eval split, 200 instances/seed · `generation` = 2 · `seeds` = [0, 1, 2] · `n_train` = 500 · `n_eval` = 200 · `enc_steps` = 1,200 · `bc_steps` = 2,000 · `target_params` = 60,000 · `matched_params` = mlp 60,047, lstm 59,965, transformer 59,819 · `judge_exact` = True

*SEPARATE TABLE BY CHARTER, AND ON TWO GROUNDS, NOT ONE. (i) ENVIRONMENT: this table is CMAZE-1 PROPER; T1 is CMAZE-2, a REQUEST LAYER over it. (ii) BUDGET: this tier trains ~60k-parameter arms on 500 instances and evaluates 200, while T1's arms are matched at ~410k parameters over 96 mazes / 1,208 requests. w12r/MF10: the earlier wording of this caption said `the same environment`, which was wrong - T1 and T3 are a DIFFERENT ENVIRONMENT, not merely a different budget, and they must never share a table.*

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
| FRAGMENT-LOOKUP r=1 (CMAZE-1 tier column) | **reference** | 200 / 200 / 200 | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.3231 (0.2946 / 0.3475 / 0.3272) |
| FRAGMENT-LOOKUP r=2 (CMAZE-1 tier column) | **reference** | 200 / 200 / 200 | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 0.0392 (0.0375 / 0.0369 / 0.0432) |

> ⭐ **The published tripwire reproduces HERE.** BLIND-SEQUENTIAL is **0.5317 (0.5550 / 0.5300 / 0.5100)** at this tier against the published CMAZE-1 **0.4980**, and FRAGMENT-LOOKUP r=1 and r=2 are both **0.0000** on every seed. Compare Table 1, where the same column reads **0.9037** on the hub's unseen-pair split (CMAZE-2). ⛔ The difference is the REQUEST LAYER — which IS the difference between the two environments, CMAZE-1 here and CMAZE-2 there. See correction C4 and C9, and ⛔ never merge these two tables.

> **BYTE LEDGER: NOT APPLICABLE - THIS TABLE CARRIES NO STORE. Declared, not omitted.**

> every arm at this tier is a learned parametric baseline matched at ~60,000 parameters (mlp 60,047 / lstm 59,965 / transformer 59,819) or one of the environment's own reference columns. There is no atom block, no sufficient statistic, no address book and no coupling table, so there is nothing to print a four-part ledger for.

**DISPLAY comparator at this tier**

| row | class | refusal on IMPOSSIBLE (correct) | refusal on POSSIBLE (false) |
|---|---|---|---|
| supervised refuser (the only DISPLAY comparator), decision at t=0 | **DISPLAY** | 0.8317 (0.8200 / 1.0000 / 0.6750) | 0.7850 (0.7950 / 1.0000 / 0.5600) |

---

## TABLE 4 — Exact deletion (STORE-SCOPED) - two implementations side by side, 3 seeds each, TEST condition. THE THREE implementations, with their bands kept apart, are TABLE 5.

*EXACT HERE MEANS STORE-SCOPED (TABLES.json /exactness_definition): after deleting a set of rows the STORE CONTENT equals, to float roundoff, the content fitted from the retained rows alone, HOLDING THE FROZEN ENCODER AND DECODER FIXED. Neither is refit. It is not a claim that the renderer forgets. Both legs are 3 seeds on generation 2 on their own TEST condition. They are reported side by side as two implementations of one claim, not as one instrument: the deviations are in different units (w9/maze reports a RELATIVE residual on the statistic and a projector residual; w8/downdate reports an ABSOLUTE max deviation per well). CAVEATS THAT TRAVEL WITH THE w8 LEG (w12r/MF6): the store is built under a variance-share admission cut that is MARGINAL on every seed (`cut_is_marginal_any_seed = true`); the byte-level assertion `downdate_equals_refit` passes on all three seeds, while the separate subspace-level assertion `a1_subspace` on the PRE-CUT store FAILS on seed 2 at 60.36 ulp - below its own 66.11 ulp floor - so that failure is a property of the content, not of the downdate; and the top-level `all_three_assertions_passed = true` EXCLUDES `a1_subspace`. THE w8 LEG IS NOT A SCORED ROW (w12r/MF7): it is an assertion battery on the object at 0 scored runs; the hub commissioned to score it returned its full budget (DECLARED NOT-RUN NR3). THE BYTE LEDGER IS PRINTED BELOW AND IS NEVER NETTED (w12r/MF3): the w8 object's honest total is 24.55x / 23.84x / 23.67x the 524,288 B pin.*

**w9/maze `CluRead.delete_rows_exact` - delete the green target, TEST leg (CMAZE-2)**  ·  class **RESULT**  ·  implementation: w9 hub/maze

| property | seed 0 / seed 1 / seed 2 |
|---|---|
| statistic max relative residual (downdate vs refit) | 8.055e-16 (7.522e-16 / 8.826e-16 / 7.818e-16) |
| subspace max projector residual | 0.000e+00 (0.000e+00 / 0.000e+00 / 0.000e+00) |
| rows removed | 212 (216 / 226 / 193) |
| rows kept | 1017 (992 / 975 / 1084) |
| refit performed | False / False / False |
| passes over retained data | 0 (0 / 0 / 0) |
| items kept | False / False / False |
| address-book bytes deployed by the basis-change carry | 0 (0 / 0 / 0) |

> ⛔ **Row class.** w12r/MF7: this leg's RESULT label is CORRECT - it sits inside the pre-registered scored runs S1/S2/S3, 3 of 3 spent.

**w8/downdate `exp_w8_downdate` - covariance downdate == refit-without-those-rows, TEST leg**  ·  class **ASSERTION BATTERY (object property) - 3 seeds - TEST condition - 0 SCORED RUNS**  ·  implementation: w8 spoke/downdate-build (different code, different author)

| property | seed 0 / seed 1 / seed 2 |
|---|---|
| max abs deviation, worst well AND worst deletion (pinned 11,024 rows + random 64) - THE SPOKE'S OWN ROW | 4.5475e-13 / 1.1255e-09 / 2.8422e-14 |
| reproducibility floor for that leg (same rows downdated in two orders) | 2.1268e-13 / 4.2817e-09 / 5.6843e-14 |
| deviation ABOVE its own floor, i.e. resolvable at all? | True / False / False |
| max abs deviation over the PINNED deletion only (w12r/MF8: the narrower field the draft printed) | 3.752e-10 (2.365e-14 / 1.126e-09 / 7.105e-15) |
| negative control fired on every seed | True / True / True |
| cov bytes (statistic, declared beside the pin) | 12,310,672 / 11,941,360 / 11,850,912 |
| no pass over data | True / True / True |
| items kept | False / False / False |

> ⛔ **Row class.** w12r/MF7. This row was labelled RESULT. Its own artefact says otherwise, verbatim: `/note` = 'No scored run and no claim: these are properties of the object (deviations, byte counts, row counts), not outcomes. hub/single-design scores the object.' The hub commissioned to score it - `hub/single-design` - scored 0 of 3 and is DECLARED NOT-RUN NR3, with its full budget intact. Nobody scored this leg. The artefact was NOT relabelled to match the table; the table was relabelled to match the artefact.

> ⛔ **Caveats that live inside the cited gate file.** MF6. Both are leaf fields of the very file this row cites, and neither was carried.

| caveat | value | what it means |
|---|---|---|
| `cut_is_marginal` per seed | True / True / True | a direction sits near the threshold, so a small perturbation could change WHICH directions are kept - the same instability class as the original seed-2 failure. It does not block the gate; it belongs in the caption of anything scored on this object. |

| field | value |
|---|---|
| `a1_subspace.passed_every_seed` | **False** |
| `a1_subspace.seeds_failing` | 2 |
| `a1_subspace.worst_in_float32_ulp` | **60.36282** |
| `a1_subspace.worst_floor_subspace_in_float32_ulp` | **66.11072** |

> the byte-level assertion `downdate_equals_refit` passes on all three seeds; the separate subspace-level assertion `a1_subspace` on the PRE-CUT store fails on seed 2 at 60.36 ulp - BELOW its own 66.11 ulp floor - so that failure is a property of the CONTENT and not of the downdate, which is what the artefact's own `reproducibility_floor.what` field states.

> ⛔ the top-level `all_three_assertions_passed = true` EXCLUDES `a1_subspace`. A referee opening the cited JSON reads `passed: false` inside the lead result's own artefact.

> Artefact's own summary: `passed_every_seed` = True · `worst_max_abs_dev_over_seeds` = 1.1255e-09 · `worst_in_float32_ulp_at_unit_scale` = 0.009442

### TABLE 4 — THE BYTE LEDGER, four parts, printed, never netted (w12r/MF3)

**w9/maze leg (CMAZE-2)**

| part (seed 0 / seed 1 / seed 2) | bytes | how it is obtained |
|---|---|---|
| 1. atoms (the store) | 31,744 / 31,744 / 31,744 | store_atom_floats x 4 (float32) |
| 2. sufficient statistic, float64 | 35,904 (35,904 / 35,904 / 35,904) | stored field |
| 3. address book (the head) | 1,611,008 / 1,611,008 / 1,611,008 | head_floats x 4 (float32) |
| 4. coupling / hole table | **DECLARED NOT PRESENT in any w9/maze artefact** | the three scored run files carry no field matching 'hole' or 'coupl'. This leg's deletion downdates (S, t, n) and the head statistics (A, Mx); it has no interface hole table of the kind the w8 leg carries. |
| deployed (atoms + address book) | 1,642,752 (1,642,752 / 1,642,752 / 1,642,752) | stored field |
| honest total (deployed + statistic) | 1,678,656 / 1,678,656 / 1,678,656 | deployed_bytes + statistic_bytes |
| statistic as a share of deployed | 2.19% / 2.19% / 2.19% | statistic_bytes / deployed_bytes |
| bytes the deletion's basis-change carry deploys | 0 (0 / 0 / 0) | stored field |

> NOT under the 524,288 B pin - this leg is matched on PARAMETERS, not on store bytes.

**w8/downdate leg (Shapes3D, the 524,288 B pin at tolerance 0)**

| part (seed 0 / seed 1 / seed 2) | bytes | how it is obtained |
|---|---|---|
| 1. atoms - principal subspaces (THE STORE, and it matches the pin exactly) | 524,288 / 524,288 / 524,288 | stored field |
| 2a. running covariance, float64 sum_k d_k^2 | 12,310,672 / 11,941,360 / 11,850,912 | stored field |
| 2b. running means + per-well counts (the residual the source table does not name) | 16,424 / 16,424 / 16,424 | sum_k d_k x 8 + n_wells x 8 (float64 means, float64 counts) |
| 2. covariance / count ledger, TOTAL | 12,327,096 / 11,957,784 / 11,867,336 | cov_bytes + means_and_counts |
| 3. request count table | 11,488 / 11,488 / 11,488 | part 3 of the source spoke's own four-part ledger table |
| 4. coupling + holes | 5,288 + 1,322 / 5,288 + 1,322 / 5,288 + 1,322 | part 4 of the source spoke's own four-part ledger table: coupling table + one byte per hole |
| honest total (all four parts) | 12,869,482 / 12,500,170 / 12,409,722 | stored field |
| x THE 524,288 B PIN | 24.5466x / 23.8422x / 23.6697x | stored field |
| ledger closes to the byte? | True / True / True | store + cov + means_and_counts + 11,488 + 5,288 + 1,322 == honest_total_bytes |
| serialised object bytes, for comparison | 12,885,866 / 12,516,554 / 12,426,106 | stored field |
| DECLARED RESIDUAL, unexplained | 16,384 / 16,384 / 16,384 | object_bytes_full - honest_total_bytes |

> ⛔⛔ **the object that forgets exactly is itself 24.55x / 23.84x / 23.67x the 524,288 B pin. The paper penalises the compose-and-forget object of section 5 at 7.48x; its own lead result is three times worse and printed no byte row at all.**

> **The residual, declared rather than netted away.** The source spoke's own four-part table (`/c/chlu/outputs/w8-downdate/LOG.md` lines 414-422 and 1036-1044) names store, covariance, count table and coupling+holes. Those five numbers leave **exactly 16,424 B** unaccounted on every one of the three TEST seeds. w12r identifies that residual as the running MEANS (`sum_k d_k x 8 = 16,384 B`, and `sum_k d_k = 2,048 = D` on every seed) plus the per-well COUNTS (`5 x 8 = 40 B`); with it the ledger closes to the byte against `honest_total_bytes`. ⚠ The identification is w12r's inference from `d_k`; the residual itself is arithmetic. Separately, `items_kept_detail/object_bytes_full` exceeds `honest_total_bytes` by **exactly 16,384 B on every seed** — w12r could not attribute that to a named field and reports it as a DECLARED RESIDUAL rather than netting it away.

> ⭐ **Exact is not free.** On TEST the same deletion takes surviving-target composition from 0.1024 (0.1569 / 0.0702 / 0.0802) to 0.0000 (0.0000 / 0.0000 / 0.0000). The claim is exactness, never cost-free erasure.

See figure 4.

---

## TABLE 5 — Exact deletion - THE THREE IMPLEMENTATIONS AS THREE ROWS, and the failure they agree on

*THE BANDS ARE NOT SHARED AND ARE NOT MERGED. Implementation 1 (w9/maze, CMAZE-2) is at 7.5e-16..8.8e-16 with a projector residual of exactly 0.000e+00 and is the only one inside scored runs. Implementation 2 (w8 spoke/downdate-build) runs 2.842e-14..1.126e-09 - five orders coarser - with TWO of its three TEST legs below their own reproducibility floors. Implementation 3 (w8 hub/single-design) is 3 CONFIGURATIONS x 3 SEEDS = 9 legs, worst 0.094 atom ulp / 0.0251 subspace ulp. Nine legs are NOT nine seeds.*

| implementation | class | seeds | configs | legs | scored runs | configuration | band |
|---|---|---|---|---|---|---|---|
| 1. w9 hub/maze - `CluRead.delete_rows_exact` (CMAZE-2, TEST) | **RESULT - 3 of 3 scored runs (S1/S2/S3)** | 3 | 1 | 3 | 3 | K=8, n_atoms=31 - one pinned configuration | **7.522e-16 .. 8.826e-16 on the statistic; EXACTLY 0.000e+00 on the subspace projector** |
| 2. w8 spoke/downdate-build - `exp_w8_downdate` (Shapes3D, TEST) | **ASSERTION BATTERY (object property) - 0 SCORED RUNS** | 3 | 1 | 3 | 0 | factor partition K=5, A=64 - the incumbent's grain | **2.842e-14 .. 1.126e-09 - FIVE ORDERS COARSER than implementation 1** |
| 3. w8 hub/single-design - matched-bytes candidates (Shapes3D, TEST) | **ASSERTION BATTERY (object property) - 0 SCORED RUNS** | 3 | 3 | 9 | 0 | ward_balanced K=40 / K=44 / K=48 - THREE configurations, three seeds each | **5.551e-16 .. 1.490e-13 on seed 0; worst ulp over all nine legs 0.094 (atom) / 0.0251 (subspace)** |

**1. w9 hub/maze - `CluRead.delete_rows_exact` (CMAZE-2, TEST)**  ·  metric: max RELATIVE residual on the sufficient statistic; max projector residual on the subspace

| quantity | value (seed 0 / seed 1 / seed 2) | traced to |
|---|---|---|
| statistic max relative residual | 8.0552e-16 (7.5217e-16 / 8.8265e-16 / 7.8176e-16) | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`; `/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json`; `/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` `/display/delete_green_exactness/statistic_max_rel_residual` |
| subspace max projector residual | 0.0000e+00 (0.0000e+00 / 0.0000e+00 / 0.0000e+00) | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`; `/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json`; `/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` `/display/delete_green_exactness/subspace_max_projector_residual` |
| refit performed | False / False / False | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`; `/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json`; `/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` `/display/delete_green_exactness/refit` |
| passes over retained data | 0 (0 / 0 / 0) | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`; `/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json`; `/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` `/display/delete_green_exactness/passes_over_retained_data` |

> this is the leg that is unambiguously in the 1e-16 band, it is the only one inside scored runs, and it replicates from a clean checkout with ZERO numeric differences (`/c/chlu/outputs/w9-maze/replication/REPLICATION.md`)

**2. w8 spoke/downdate-build - `exp_w8_downdate` (Shapes3D, TEST)**  ·  metric: max ABSOLUTE deviation per well, downdate vs refit-without-those-rows

| quantity | value (seed 0 / seed 1 / seed 2) | traced to |
|---|---|---|
| worst max abs deviation (pinned 11,024 rows + random 64) | 4.5475e-13 / 1.1255e-09 / 2.8422e-14 | `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/per_seed/0/downdate_equals_refit/worst_max_abs_dev … one entry per seed` |
| its own reproducibility floor | 2.1268e-13 / 4.2817e-09 / 5.6843e-14 | `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/per_seed/0/reproducibility_floor/floor_max_abs_dev … one entry per seed` |

> the worst leg (TEST seed 1, 1.126e-09) sits BELOW its own reproducibility floor of 4.282e-09, and so does seed 2 (2.842e-14 vs 5.684e-14): on TWO of three legs the residual is not resolvable at all. Printed that way it is STRONGER, not weaker - an unresolvable residual is the best available outcome for an exactness claim. The two additional val legs the advisor quotes are 4.547e-13 and 1.164e-10.

**3. w8 hub/single-design - matched-bytes candidates (Shapes3D, TEST)**  ·  metric: max atom deviation in float32 ulp at unit scale; and the same at subspace level

| quantity | value (seed 0 / seed 1 / seed 2) | traced to |
|---|---|---|
| atom-level ulp, worst over the 9 legs | 0.0940 | `/c/chlu/outputs/w8-single-design/LOG.md` LOG.md:476-481 |
| subspace-level ulp, worst over the 9 legs | 0.0251 | `/c/chlu/outputs/w8-single-design/LOG.md` LOG.md:495-503 |
| absolute deviation, seed 0, K=40 / K=44 / K=48 | 5.5510e-16 / 1.4210e-14 / 1.4900e-13 | `/c/chlu/outputs/w8-single-design/LOG.md` LOG.md:472 |

> THIS is the implementation the phrase '9/9' came from, and the nine are 3 CONFIGURATIONS x 3 SEEDS, not nine seeds. It passes under BOTH the atom definition and the subspace definition, so it does not need the definition change to survive. Every number in this row lives in a MARKDOWN artefact and is carried as a locator: verify_tables.py re-hashes the file but cannot re-resolve these values.

### ⭐⭐ THE STRONGEST FORM OF THE TWO-IMPLEMENTATION CLAIM IS THE FAILURE AGREEMENT, NOT THE SUCCESSES

> Two implementations, different code, different authors, different sign conventions, reproduced THE SAME failure on THE SAME seed in THE SAME well: the incumbent's coarse, imbalanced `factor K=5, A=64` partition on TEST seed 2.

| leg | float32 ulp at unit scale | traced to |
|---|---|---|
| w8 spoke/downdate-build, atom level | 32.90 | `/c/chlu/outputs/w8-downdate/LOG.md` LOG.md:625-640 (3.921e-06 = 32.90 ulp) and LOG.md:728,794,905 (4.227e-06 = 35.46 ulp) |
| w8 spoke/downdate-build, subspace level (a stored field) | 60.36282 | `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/a1_subspace/worst_in_float32_ulp` |
| w8 spoke/downdate-build, its own subspace floor (a stored field) | 66.11072 | `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/a1_subspace/worst_floor_subspace_in_float32_ulp` |
| w8 hub/single-design, atom level | 14.377 | `/c/chlu/outputs/w8-single-design/LOG.md` LOG.md:479,481 |
| w8 hub/single-design, subspace level | 14.4518 | `/c/chlu/outputs/w8-single-design/LOG.md` LOG.md:501 |
| w8 hub/single-design, its own two-order floor | 36.9305 | `/c/chlu/outputs/w8-single-design/LOG.md` LOG.md:501,506 |

> ⭐⭐ **The citable sentence.** The second implementation reproduced the same seed-2 failure with different code and its own sign convention - 14.377 ulp against 32.90 - and both implementations' own reproducibility floors EXCEED their observed deviations there, so both independently conclude that seed 2's content is not a function of its data by any method. That is a stronger corroboration than either implementation's successes.

> **How it was closed.** a variance-share admission cut with NO FREE PARAMETER, declared before the re-run and not moved, `tau` = 4.144858e-09 (4.941056e-16 / 1.1920929e-07 = 4.144858e-09).

> ⛔⛔ **What must never be printed.** '1e-16..1e-14 on Shapes3D, 9/9 seeds, two independent implementations'. Wrong three ways: the nine legs are 3 configurations x 3 seeds inside ONE implementation; the band is not shared (the peer runs 2.842e-14 .. 1.126e-09, five orders coarser, its worst leg below its own 4.282e-09 reproducibility floor); and it was not unconditional - it FAILED first at 32.90 ulp.

---

## TABLE 6 — w5 compose-field - the BYTE and PARAMETER ledger of `pca_tensor` against `rival_mlp_180k`, TEST, 3 seeds

**Pins.** `condition` = test · `seeds` = [0, 1, 2] · `store_pin_B` = 524,288 · `tolerance` = 0

*THE STORE IS PINNED AT 524,288 B AT TOLERANCE 0; THE ADDRESS BOOK IS DECLARED BESIDE IT AND NEVER NETTED. Total deployed bytes are 713,632 / 719,352 / 719,352 against the rival's 715,808 - matched to within +-0.5%, which is what the artefacts prove and is a stronger statement than `at 524,288 B`. On parameters the arm carries 178,408 / 179,838 / 179,838 against 178,952: -0.304% / +0.495% / +0.495%, so ON TWO OF THREE SEEDS THE ARM CARRIES SLIGHTLY MORE PARAMETERS THAN THE BAR. It is stated that way round rather than quoting the seed that flatters the arm.*

**`pca_tensor` (THE ARM)**  ·  class **arm**

| quantity | seed 0 / seed 1 / seed 2 | traced to |
|---|---|---|
| store bytes (the pin, tolerance 0) | 524,288 (524,288 / 524,288 / 524,288) | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` `/arms/pca_tensor/store_bytes` |
| address book bytes (declared beside, never netted) | 193,157 (189,344 / 195,064 / 195,064) | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` `/arms/pca_tensor/address_bytes` |
| store parameters | 131,072 (131,072 / 131,072 / 131,072) | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` `/arms/pca_tensor/param_count` |
| TOTAL deployed bytes (store + address book) | 713,632 / 719,352 / 719,352 | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` `/arms/pca_tensor/store_bytes … one entry per seed` |
| TOTAL parameters (store + address book / 4) | 178,408 / 179,838 / 179,838 | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` `/arms/pca_tensor/param_count … one entry per seed` |

**`rival_mlp_180k` (the parameter bar)**  ·  class **RESULT baseline**

| quantity | seed 0 / seed 1 / seed 2 | traced to |
|---|---|---|
| store bytes | 0 (0 / 0 / 0) | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` `/arms/rival_mlp_180k/store_bytes` |
| address book bytes | 0 (0 / 0 / 0) | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` `/arms/rival_mlp_180k/address_bytes` |
| parameters | 178,952 (178,952 / 178,952 / 178,952) | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` `/arms/rival_mlp_180k/param_count` |
| TOTAL deployed bytes | 715,808 / 715,808 / 715,808 | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` `/arms/rival_mlp_180k/param_count` |

**THE COMPARISON, as the artefacts support it**  ·  class **derived**

| quantity | seed 0 / seed 1 / seed 2 | traced to |
|---|---|---|
| arm bytes as a share of the rival's | 99.70% / 100.50% / 100.50% | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` `/arms/pca_tensor/address_bytes` |
| arm parameters against the bar | -0.304% / +0.495% / +0.495% | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` `/arms/pca_tensor/param_count … one entry per seed` |

> ⛔ **MF12.** The draft printed *178,928 vs 178,952 = -0.01%, pointed at `w5-compose-field/LOG.md` 02:45Z*. 178,928 is the VAL leg's figure. The source spoke wrote a correction bullet at 01:58 UTC headed 'CORRECTION to a number I put in the PREREG' saying so, and the 02:45Z entry the draft points at is the val arm-clears-the-bar bullet (+16.875), not a byte or parameter entry. The scored TEST legs say **178,408 / 179,838 / 179,838 = -0.304% / +0.495% / +0.495%**.

> The source spoke's own sentence: *"the secondary parameter match is within +-0.5% rather than the +-0.013% I wrote, and on two of three seeds the arm carries slightly MORE parameters than the bar. I state it that way round rather than quoting the seed that flatters me."*

**Ceiling ladder — address-book bytes per rung (SF2)**

| rung | class | address book bytes | store bytes |
|---|---|---|---|
| oracle_content_fine | NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, fine K) | 0 (0 / 0 / 0) | 524,288 (524,288 / 524,288 / 524,288) |
| oracle_content_items | NOT AN ARM (ceiling, 64 REAL ITEMS @524,288 B) | 0 (0 / 0 / 0) | 524,288 (524,288 / 524,288 / 524,288) |
| oracle_content_pool | NOT AN ARM (ceiling, BEST REALISABLE content @524,288 B) | 0 (0 / 0 / 0) | 524,288 (524,288 / 524,288 / 524,288) |
| oracle_span | NOT AN ARM (ceiling) | 1,766,400 (1,766,400 / 1,766,400 / 1,766,400) | 524,288 (524,288 / 524,288 / 524,288) |
| oracle_span_field | NOT AN ARM (ceiling, FIELD store) | 0 (0 / 0 / 0) | 524,288 (524,288 / 524,288 / 524,288) |
| span_head | ABLATION (the deployed read) | 1,766,400 (1,766,400 / 1,766,400 / 1,766,400) | 524,288 (524,288 / 524,288 / 524,288) |

> ⚠ SF2. A byte-ledger paper cannot ship a ceiling ladder whose rungs differ 9x in address-book size silently. `oracle_span` and `span_head` carry an address book of 1,766,400 B - 9.3x the arm's own book and 3.4x the 524,288 B store pin - while every other rung carries 0.

---

# ⭐ WAVE 12's ROWS, FOLDED IN — TABLES 7 to 12

**Added by `w13 spoke/seeds-and-tables`. ZERO scored runs: every number below is re-read from a wave-12 artefact at the path and pointer printed beside it.**

## ⛔⛔ THE TWO NOTATIONS THAT MUST NEVER COLLIDE

| symbol | means | range measured | where it lives |
|---|---|---|---|
| **n** | the number of jointly composed factors -- the LADDER RUNGS | n = 2, 3, 4, 5, 6 | T8 (and fig6) |
| **K** | the number of WELLS -- the partition of the feature space | K = 5, 8, 10, 12, 16, 20, 24, 28, 32; K* = 32 | T7, T9, T9b, T12 (and fig7, fig8) |

> ⛔ **EVERY LADDER ROW STATES ITS K. EVERY WELLS ROW STATES ITS n. A row that states neither is not a row.**

> T8's five rows are n = 2..6 at ONE K (K = 5). T7's one row is ONE n-set (60 pair corners + 49 n-way corners) at ONE K (K* = 32). They are not the same axis and never share a table.

## ⭐ SEED COUNTS ARE A COLUMN, NOT A FOOTNOTE

| table | what it is | seeds (n) | seed ids | why not more |
|---|---|---|---|---|
| T1 / T2 / T3 / T4 / T5 / T6 | waves 5-9, carried unchanged from w12r | 3 | 0, 1, 2 | wave-5..9 scored legs; not re-run |
| **T7** | scored rows 3 and 4 at K* = 32, TEST | 1 | 0 | NR6 (2 of 3 scored runs unspent) and NR7 (seeds 3-4 unjudgeable) |
| **T8** | the n-way ladder, n = 2..6, K = 5 | 3 | 0, 1, 2 | NR7 |
| **T9** | deletion, TEST, K* = 32 | 1 | 0 | NR6, NR7 |
| **T9b** | deletion, VAL, K = 5 (SMOKE) | 1 | 0 | NR6, NR7 |
| **T10** | refusal DISPLAY - ours / the comparator | 1 / 3 | 0  /  0, 1, 2 | our rows NR6; the comparator is already at 3 seeds. The head-to-head is NR9 |
| **T11** | the two ablations, w5 TEST | 3 | 0, 1, 2 | wave-5 scored legs; never re-run at 5 |
| **T12** | the K-curve, VAL (SMOKE) | 3 / 1 | 0, 1, 2 at K = 5 and K = 32;  0 only at K = 8..28 | wall clock at the seven middle K; NR7 beyond 3 |

> ⛔ **3-, 4- and 5-seed rows are NEVER plotted together without a printed n per row. In this file no row is at 4 or 5 seeds; the counts actually present are 1 and 3, and they are printed as a column on every table and annotated on every figure that mixes them.**

> ⭐ **The generation-2 caches for seeds 3 and 4 DO exist and are verified** — both READY, generation 2, scope full, 480,000 rows, 59 block hashes each, shape and dtype matching seeds 0-2, bridge complete and its gate PROCEED; independently re-hashed from the bytes on disk with `all_pass: true`; the SHIPPED re-encode gate passes on both at `frac_differing 0.0` and `max_ulp_delta 0.0`.

> ⛔ **And they still buy no row.** no frozen decoder (`decoder_v1_s3.pkl` absent), no external classifier (`fw.classifier_dir(3)` and `(4)` absent) and no fitted rivals (`/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json` records `rivals_seeds_present` = [0, 1, 2] for all seven families) exist for either seed, so `fw.Judge(3)` raises and no corner of seeds 3 or 4 can be judged. The caches also landed AFTER both scoring hubs closed their seed gates. See NR7 for the launch line.

---

## TABLE 7 (MAIN) - Shapes3D value-holdout, the SINGLE DESIGN at K* = 32, TEST: rows 3 and 4. 1 scored seed, DECLARED.

**Class:** RESULT · **Placement:** MAIN TEXT

> **Notation on this table.** **K_wells**: 32 · **n_factors**: row 3 = pair corners (2 factors pinned); row 4 = n-way corners (all six pinned). This table varies neither: it is ONE n-set at ONE K. · **rule**: T7 states K = 32 on every row; the ladder that varies n is T8 and every one of its rows sits at K = 5.

**Pins.** `condition` = test · `K (wells)` = 32 · `A_requested` = 63 · `seed` = 0 · `metric` = distinct_valid_novel_tuples per corner · `estimand` = fw.paired_estimand, the CORNER as the unit · `margin declared in advance` = 1.0 · `feature cache` = generation 2, scope full, 480,000 rows · `address fit seconds` = 1242.3163497447968

*Pins traced to:* `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` `/K_star`; `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` `/A_requested`; `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` `/seed`; `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` `/condition` …

**Seed count**

| field | value | traced to |
|---|---|---|
| scored seed id | 0 | `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` `/seed` |
| PRINTED AS A COLUMN, NOT A FOOTNOTE | n = 1 (seed 0). Every row of T7 is a 1-seed row. |  |
| why not 3 | the crossover hub spent run 1 of its 3 `single-design` scored runs on seed 0 at K* = 32; runs 2/3 and 3/3 (seeds 1 and 2) are NR6, wall clock. |  |
| why not 5 | seeds 3 and 4 have generation-2 caches (NR7) but no frozen decoder, no external classifier and no fitted rivals, so no corner of them can be judged. |  |
| NEVER MIX | T7 (1 seed) must not share a plot with T8 (3 seeds) or T1/T2/T3 (3 seeds) unless the seed count is printed per row. |  |

| row | class | seeds (n) | row 3 - 60 pair corners | row 4 - 49 n-way corners | all 109 corners | judge accuracy | paired delta vs the arm | CI95 of that delta | corners the arm wins | params @ the 524,288 B pin |
|---|---|---|---|---|---|---|---|---|---|---|
| **`cov_tensor` -- THE ARM** (single design, K = 32) | RESULT | 1 | 31.600 | 66.959 | 47.495 | 0.9012 | -- (this IS the arm) | -- | -- | NOT A PARAMETER MODEL - see the byte ledger |
| `landscape_removed` | control | 1 | 0.000 | 0.000 | 0.000 | 0.0000 | +47.4954 | +44.101 / +51.028 | 109 | -- |
| `landscape_shuffled` | control | 1 | 0.000 | 0.000 | 0.000 | 0.0000 | +47.4954 | +44.101 / +51.028 | 109 | -- |
| `landscape_shuffled_across_wells` | control | 1 | 0.000 | 0.000 | 0.000 | 0.0000 | +47.4954 | +44.101 / +51.028 | 109 | -- |
| `random_atoms` | control | 1 | 0.000 | 0.000 | 0.000 | 0.0000 | +47.4954 | +44.101 / +51.028 | 109 | -- |
| `mlp` (524,288 B reference) | comparator @524,288 B | 1 | 2.933 | 7.959 | 5.193 | 0.0945 | +42.3028 | +39.165 / +45.532 | 109 | 130,514 |
| `lstm` (524,288 B reference) | comparator @524,288 B | 1 | 2.500 | 7.531 | 4.761 | 0.0852 | +42.7339 | +39.679 / +45.899 | 109 | 130,721 |
| `transformer` (524,288 B reference) | comparator @524,288 B | 1 | 0.667 | 1.918 | 1.229 | 0.0222 | +46.2661 | +42.899 / +49.725 | 109 | 130,433 |
| `cont_mlp` (524,288 B reference) | comparator @524,288 B | 1 | 0.000 | 0.061 | 0.028 | 0.0004 | +47.4679 | +44.073 / +50.991 | 109 | 131,891 |
| `table` -- the FLOOR, never a rival | floor | 1 | 5.833 | 11.163 | 8.229 | 0.1589 | +39.2661 | +35.862 / +42.679 | 109 | -- |

**rank / width — the decorative warning**

| field | value | traced to |
|---|---|---|
| rank/width max on the scored object | 0.9688 | `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` `/rank_over_width_max` |
| wave 9's decorative threshold | 0.938 |  |
| reading | 0.9688 is ABOVE the 0.938 at which wave 9's maze arm was decorative, and the four controls still read exactly 0.000. Only the control decides; rank/width is printed because it is the warning it is. |  |

**⛔ BOTH BYTE MATCHINGS, PRINTED**

| field | value | traced to |
|---|---|---|
| WHY BOTH COLUMNS ARE PRINTED | the two matchings tell OPPOSITE stories and a reader who sees one draws the opposite conclusion. See T8's caption for the sign flip on the ladder. |  |
| 1x = this object's own measured total, bytes | 2,144,178 | `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` `/ledger_four_part/honest_total_bytes` |
| 1x = float32 parameters a 1x-matched rival would carry | 535,784 | `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/per_K/32/baseline_match_1x/target_params_float32` |
| 1x column status | DECLARED NOT-RUN - no rival was fitted at this size (NR8). |  |
| 524,288 B reference, bytes | 524,288 | `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` `/ledger_four_part/reference_pin_bytes` |
| 524,288 B reference, float32 parameters | 131,072 | `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/per_K/32/baseline_match_1x/reference_pin_params_float32` |
| rivals actually fitted, params | 130,433 - 131,891, i.e. within +/-0.63 % of 131,072 |  |
| 524,288 B column status | MEASURED - every comparator row above is against this column. |  |
| DIRECTION OF THE ERROR | AGAINST the arm. A 1x-matched rival would carry about 4.09x these parameters and would be STRONGER, so the printed comparison FLATTERS the arm and no row above is a 1x result. |  |

**Byte ledger — FOUR PARTS, PRINTED, NEVER NETTED (w12r/MF3). Seed 0, K* = 32, TEST.**

| part (ONE seed, id 0) | bytes | how it is obtained |
|---|---|---|
| 1. atoms | 516,096 | stored field |
| 2. covariance + count ledger | 560,608 | stored field |
| 2a. of which covariance | 549,120 | stored field |
| 2b. of which count | 11,488 | stored field |
| 3. address book | 1,060,864 | stored field |
| 4. coupling / hole table | 6,610 | stored field |
| **honest total (the sum of the four)** | **2,144,178** | stored field |
| **x the 524,288 B reference** | **4.0897** | stored field |

> ⛔ **pin_status:** the object is 4.0897x the 524,288 B figure. That figure is a REFERENCE COLUMN in this table, never a budget the object met.


> ⭐ **VERDICT.** the control gate PASSES - all four controls read exactly 0.000 on both rows at rank/width 0.9688 - so the row may be claimed, at 1 seed, against the 524,288 B reference column and NOT at 1x.

**verdict, traced**

| field | value | traced to |
|---|---|---|
| charter C.1 vs the prototype table floor | +39.2661 | `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` `/paired_vs_cov_tensor/table/mean_paired_delta` |
| its CI95 lower bound | +35.8624 | `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` `/paired_vs_cov_tensor/table/bootstrap_ci95/0` |
| margin declared in advance | +1.0 | `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` `/paired_vs_cov_tensor/table/margin_declared_in_advance` |

> ⚠ `cont_mlp` scores 0.000 / 0.061 - at the floor. That is a fact about that rival's fit on this population, not a demonstration about the arm, and it is printed rather than leaned on.

> ⚠ The deletion legs of this same object are T9 and the refusal DISPLAY is T10. Same object, same seed, still separate tables, because a DISPLAY is not a RESULT.

> ⚠ The request-only property is asserted in the artefact: no feature tensor enters launch, settle or read (charter B.3), bitwise stable under feature perturbation over 64 checks.

---

## TABLE 8 - the n-WAY LADDER on Shapes3D: n = 2..6 jointly composed factors, K = 5 WELLS on every row, 3 seeds on every row.

**Class:** RESULT · **Placement:** APPENDIX -- the K-sweep's ENDPOINT at K = 5. NOT main text: the main text carries ONE K, K*, and never both (Head ruling, 2026-09-21T12:38:07Z, HEAD-RULING-K.md).

> **Notation on this table.** **n**: the ladder rung = the number of jointly composed factors. n = 2, 3, 4, 5, 6. · **K**: 5 on EVERY row of this table. The source writes the rung as `k`; this table writes the rung as n and prints K separately, because K is the WELL COUNT and the two were collided once. · **RULE**: T8 (K = 5) and T7 (K = 32) are two different objects and never share a table or an axis.

**Pins.** `condition` = Shapes3D value-holdout, k-way families from the w12 harness · `K (wells)` = 5 · `metric` = distinct_valid_novel_tuples per corner · `estimand` = paired, the corner as the unit · `margin` = 1.0 · `arms per leg` = 13 · `legs` = 18 = (5 rungs + `shipped`) x 3 seeds · `scored arm-runs` = 234

*Pins traced to:* `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/K_of_the_store_arms`; `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/metric`; `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/margin`; `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/n_seeds_declared` …

**Seed count**

| field | value | traced to |
|---|---|---|
| seeds on every row | 3 | `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/n_seeds_declared` |
| PRINTED AS A COLUMN | `seeds (n)` is a column of this table, not a footnote. |  |
| why not 5 | seed 3 passes the generation-2 cache gate and still cannot be scored - `Judge(3)` raises FileNotFoundError on `decoder_v1_s3.pkl`; seed 4's cache landed at 2026-09-21T15:27:15Z, after this hub had closed its seed gate. Both are NR7. |  |

| row | class | seeds (n) | K (wells) | seeds (n), traced | corners | ceiling / corner | single design (the arm), K = 5 | pca_tensor - composition-only store (ABLATION) | learned_1x - MLP matched to the arm's OWN total | rival_mlp @524,288 B | rival_lstm @524,288 B | rival_transformer @524,288 B | rival_cont_mlp @524,288 B | rival_table - the FLOOR | learned_ref_ownpool | delta vs 1x (learned_1x) | delta vs best @524,288 B | rank/width max (arm) | control landscape_shuffled, paired delta | control random_atoms, paired delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **n = 2** | RESULT (APPENDIX placement) | 3 | 5 | 3 | 60 / 60 / 60 | 36 | 28.6333 / 31.6500 / 31.3333 | 28.6333 / 31.6500 / 31.3333 | 33.2000 / 33.8667 / 32.9000 | 17.0167 / 22.0000 / 23.3333 | 19.4167 / 19.3333 / 23.7500 | 12.7500 / 15.7833 / 24.2500 | 3.8833 / 4.5667 / 6.5333 | 5.8333 / 3.4833 / 6.1833 | 5.5667 / 7.5833 / 9.3000 | -4.5667 / -2.2167 / -1.5667 | +9.2167 / +9.6500 / +7.0833 | 0.6949 / 0.8889 / 0.8649 | +28.6333 / +31.6500 / +31.3333 | +28.3833 / +31.5667 / +30.5833 |
| **n = 3** | RESULT (APPENDIX placement) | 3 | 5 | 3 | 120 / 120 / 120 | 72 | 56.9500 / 59.4250 / 62.8750 | 56.9500 / 59.4250 / 62.8750 | 52.1333 / 59.4583 / 54.9333 | 35.2083 / 41.5667 / 49.7500 | 44.4333 / 39.4000 / 46.6750 | 28.8000 / 33.9000 / 51.9917 | 6.3333 / 9.1167 / 12.6667 | 8.4333 / 7.9167 / 12.2333 | 8.2167 / 10.9083 / 16.2583 | +4.8167 / -0.0333 / +7.9417 | +12.5167 / +17.8583 / +10.8833 | 0.6885 / 0.8767 / 0.8649 | +56.9500 / +59.4250 / +62.8750 | +56.3833 / +59.2500 / +62.2250 |
| **n = 4** | RESULT (APPENDIX placement) | 3 | 5 | 3 | 120 / 120 / 120 | 72 | 60.9333 / 62.2333 / 64.9583 | 60.9333 / 62.2333 / 64.9583 | 67.1750 / 68.5833 / 68.2167 | 35.2083 / 41.5667 / 49.7500 | 44.4333 / 39.4000 / 46.6750 | 28.8000 / 33.9000 / 51.9917 | 6.3333 / 9.1167 / 12.6667 | 8.4333 / 7.9167 / 12.2333 | 8.5417 / 12.5333 / 18.7667 | -6.2417 / -6.3500 / -3.2583 | +16.5000 / +20.6667 / +12.9667 | 0.6964 / 0.8788 / 0.9219 | +60.9333 / +62.2333 / +64.9500 | +60.4500 / +62.0500 / +63.7167 |
| **n = 5** | RESULT (APPENDIX placement) | 3 | 5 | 3 | 120 / 120 / 120 | 72 | 63.2417 / 64.5083 / 66.4083 | 63.2417 / 64.5083 / 66.4083 | 68.0500 / 68.7583 / 69.0500 | 35.2083 / 41.5667 / 49.7500 | 44.4333 / 39.4000 / 46.6750 | 28.8000 / 33.9000 / 51.9917 | 6.3333 / 9.1167 / 12.6667 | 8.4333 / 7.9167 / 12.2333 | 9.8667 / 15.3083 / 19.7083 | -4.8083 / -4.2500 / -2.6417 | +18.8083 / +22.9417 / +14.4167 | 0.6949 / 0.9014 / 0.8649 | +63.2417 / +64.5083 / +66.4083 | +62.5583 / +63.9250 / +64.3750 |
| **n = 6** | RESULT (APPENDIX placement) | 3 | 5 | 3 | 120 / 120 / 120 | 72 | 62.6917 / 63.9917 / 66.9833 | 62.6917 / 63.9917 / 66.9833 | 68.4667 / 68.5583 / 68.8250 | 35.2083 / 41.5667 / 49.7500 | 44.4333 / 39.4000 / 46.6750 | 28.8000 / 33.9000 / 51.9917 | 6.3333 / 9.1167 / 12.6667 | 8.4333 / 7.9167 / 12.2333 | 10.6667 / 15.1167 / 21.0333 | -5.7750 / -4.5667 / -1.8417 | +18.2583 / +22.4250 / +14.9917 | 0.6949 / 0.8889 / 0.8889 | +62.6917 / +63.9917 / +66.9833 | +62.4333 / +63.8333 / +66.4917 |
| `shipped` - the value-holdout condition as it exists on disk (reproduction leg) | instrument check, NOT a rung | 3 | 5 | 3 | 109 / 109 / 109 | 72 | 44.9541 / 46.3211 / 47.7248 | 44.9541 / 46.3211 / 47.7248 | 48.8991 / 49.5413 / 49.4404 | 26.1101 / 28.5780 / 36.5596 | 30.0275 / 28.9174 / 36.2294 | 20.7156 / 24.9817 / 37.5138 | 6.0092 / 6.1927 / 11.8349 | 8.2294 / 5.0734 / 9.7523 | 9.1835 / 11.1193 / 15.1284 | -3.9450 / -3.2202 / -1.7156 | +14.9266 / +17.4037 / +10.2110 | 0.6897 / 0.8889 / 0.8889 | — | — |

**⛔ BOTH BYTE MATCHINGS, PRINTED — AND THE FLIP**

| field | value | traced to |
|---|---|---|
| THE CAPTION THAT MUST TRAVEL WITH THIS TABLE | THE TWO BYTE MATCHINGS TELL OPPOSITE STORIES ABOUT THE n-TREND. Against the 524,288 B rivals the arm's margin GROWS with n: +7.083..+9.650 at n = 2 rising to +14.992..+22.425 at n = 6. Against a learner matched to the arm's OWN total (1x) the margin is NEGATIVE at n = 2, 4, 5 and 6. A reader shown one column alone draws the opposite conclusion, so both columns are printed on every row of this table. |  |
| delta vs best @524,288 B - n = 2, per seed | +9.2167 / +9.6500 / +7.0833 | `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/rungs/2/arms/single_design_K5/per_seed/0/paired_vs_best_learned_5/mean_paired_delta … one entry per seed` |
| delta vs best @524,288 B - n = 6, per seed | +18.2583 / +22.4250 / +14.9917 | `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/rungs/6/arms/single_design_K5/per_seed/0/paired_vs_best_learned_5/mean_paired_delta … one entry per seed` |
| delta vs 1x - n = 2, per seed | -4.5667 / -2.2167 / -1.5667 | `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/rungs/2/arms/single_design_K5/per_seed/0/paired_vs_best_learned/mean_paired_delta … one entry per seed` |
| delta vs 1x - n = 6, per seed | -5.7750 / -4.5667 / -1.8417 | `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/rungs/6/arms/single_design_K5/per_seed/0/paired_vs_best_learned/mean_paired_delta … one entry per seed` |
| delta vs 1x, ceiling-normalised, n = 2/3/4/5/6 | -0.077315 / +0.058912 / -0.073380 / -0.054167 / -0.056404 | `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/rungs/2/arms/single_design_K5/per_seed/0/paired_vs_best_learned/mean_paired_delta … one entry per seed` |
| the_1x_opponent | `learned_1x` is ONE family (an MLP) fitted at the arm's own total. A 1x-matched LSTM, transformer, continuous MLP and table are a DECLARED NOT-RUN (NR8), so the 1x column is the WEAKEST available 1x opponent and the true 1x margin can only be more negative. |  |

**Instrument — what this ladder actually varies**

| field | value | traced to |
|---|---|---|
| FOR n >= 3 THIS LADDER IS A TRAINING-DATA AXIS, NOT A COMPOSITION-DEPTH AXIS | n = 3, 4, 5 and 6 probe the IDENTICAL 120 corners; only how coarsely the hold-out is carved changes, and with it the held-out row count (300,000 / 235,200 / 213,600 / 200,640). Every arm fitted on the rung's own pool tracks that count at r = -0.96 .. -0.98; every arm not fitted on it is EXACTLY constant across those four rungs. The n = 2 -> n >= 3 step is the only genuine change of probe on this ladder. |  |
| corners at n = 2 (disjoint from n >= 3) | 60 / 60 / 60 | `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/rungs/2/n_corners_per_seed/0 … one entry per seed` |
| corners at n = 6 | 120 / 120 / 120 | `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/rungs/6/n_corners_per_seed/0 … one entry per seed` |
| no arm has a breaking point | no arm has a breaking point | `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/same_k_for_every_arm/statement` |
| legs that failed | 0 - the source's own `not_run` field is the empty list | `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/not_run` |
| ceiling-normalised arm, n = 2 -> n = 6 | 0.8483 / 0.8299 / 0.8709 / 0.8989 / 0.8966 | `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/ladder_verdicts/single_design_K5/per_rung/2/normalised_mean … one entry per seed` |

**Byte ledger — FOUR PARTS, NEVER NETTED. The ladder's ledger file carries parts 1-3 measured on all 18 fitted legs; part 4 (coupling / hole table) is NOT in that file and is a DECLARED ABSENCE below, never a zero.**

| part (seed 0 / seed 1 / seed 2 unless the row says otherwise) | bytes | how it is obtained |
|---|---|---|
| 1. atoms (both store arms, every leg) | 524,288 | stored field |
| 2. sufficient statistics - single design, n = 2, seeds 0/1/2 | 12,019,517 (12,284,920 / 11,948,136 / 11,825,496) | stored field |
| 2. sufficient statistics - pca_tensor (the ablation) | 0 (0 / 0 / 0) | stored field |
| 3. address book - single design, n = 2, seeds 0/1/2 | 168,891 (158,144 / 175,304 / 173,224) | stored field |
| 4. coupling / hole table | **DECLARED ABSENT FROM THE SOURCE** | `BYTES.json` carries three parts per leg. The coupling/hole table is measured on the K* object in T7 (6,610 B) and across the K-sweep in T12; it is NOT measured on the ladder's legs and is therefore not printed as a zero here. |
| **single design total - min over 18 legs** | **12,523,008** | stored field |
| **single design total - max over 18 legs** | **13,041,864** | stored field |
| **x the 524,288 B reference - min** | **23.8857** | stored field |
| **x the 524,288 B reference - max** | **24.8754** | stored field |
| **pca_tensor total - min / max** | **712,072 / 719,352** | stored field |
| **pca_tensor x reference - min / max** | **1.3582 / 1.3721** | stored field |

> ⛔ **THE HEADLINE THE DRAFT OMITTED:** the single design costs 23.886x-24.875x the 524,288 B reference and `pca_tensor` costs 1.358x-1.372x - about 18x less - for composition scores that are IDENTICAL at every rung and every seed. The single design's bytes buy EXACT DELETION, not composition.


> ⭐ **VERDICT.** the store NEVER CLEARS THE BAR against a learner matched to its own bytes - at n = 2, 4, 5 and 6 on every seed, and at 2 of 3 seeds at n = 3, so under the pre-declared every-seed rule it clears at NO rung. Nothing deteriorates as n grows; ceiling-normalised the arm goes UP, 0.8483 -> 0.8966. Both statements are the ladder's result and neither of them is a curve.

---

## TABLE 9 - EXACT DELETION (STORE-SCOPED) on the single design, TEST condition, K = 32, 1 seed (id 0).

**Class:** RESULT · **Placement:** MAIN TEXT

> **Notation on this table.** **K_wells**: 32 · **n_factors**: not varied in this table; the n-ladder is T8 · **rule**: every row of T9 states K = 32 and its seed count.

**Pins.** `condition` = test · `K (wells)` = 32 · `seed` = 0 · `definition of exact` = STORE-SCOPED - see TABLES.json /exactness_definition · `feature cache` = generation 2, scope full, 480,000 rows

*Pins traced to:* `/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` `/K`; `/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` `/seed`; `/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` `/condition`; `/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` `/A_requested` …

> ⚠ **The artefact's own `SMOKE_NOT_A_CLAIM` field reads false.**

**Seed count**

| field | value | traced to |
|---|---|---|
| seeds (n) | 1 |  |
| PRINTED AS A COLUMN | `seeds (n)` is a column of this table. |  |
| why not 3 or 5 | NR6 (wall clock, seeds 1-2) and NR7 (seeds 3-4 unjudgeable). |  |

| row | class | seeds (n) | K (wells) | seeds (n), traced | rows deleted | fraction of the fit pool | OUR adaptation, seconds | OUR bytes rewritten | OUR data rows read | OUR refit? | REFERENCE refit-without, seconds | REFERENCE refit-without, data rows read | store-scoped exactness, max abs mode dev | ranks agree | survivor corners | paired delta on survivors | DELETED corners | principal-angle cos_min (the basis DID move) | DELETED corners, novel/corner BEFORE | DELETED corners, novel/corner AFTER | DELETED corners, judge acc BEFORE | DELETED corners, judge acc AFTER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) VALUE - every fit row carrying red (`object_hue` index 0) | RESULT | 1 | 32 | 1 | 11,044 | 0.084259 | 0.8998 | 2,081,024 | 0 | false | 9.259 | 120,028 | 5.960e-08 | true | 89 | +0.0000 | 20 | 0.006538 | 32.2500 | 32.2500 | 0.8958 | 0.8958 |
| (b) CONJUNCTION - red AND cube, AS THE TASK SPECIFIES IT -- **VACUOUS** | RESULT | 1 | 32 | 1 | 0 | 0.000000 | 0.0518 | 2,081,024 | 0 | false | 9.368 | 131,072 | 0.000e+00 | true | 109 | +0.0000 | 0 | 1.000000 | DECLARED: this scope deletes no corner | DECLARED: this scope deletes no corner | — | — |
| (b') RED SPHERE - the DECLARED non-vacuous substitute | RESULT | 1 | 32 | 1 | 5,511 | 0.042046 | 0.3916 | 2,081,024 | 0 | false | 9.262 | 125,561 | 5.383e-07 | true | 109 | +0.0000 | 0 | 0.066688 | DECLARED: this scope deletes no corner | DECLARED: this scope deletes no corner | — | — |
| (c) ITEM - a random 1 % of fit rows | RESULT | 1 | 32 | 1 | 1,311 | 0.010002 | 0.1055 | 2,081,024 | 0 | false | 9.788 | 129,761 | 2.338e-07 | true | 109 | +0.0000 | 0 | 0.079602 | DECLARED: this scope deletes no corner | DECLARED: this scope deletes no corner | — | — |

**The sphere caveat, MEASURED**

| field | value | traced to |
|---|---|---|
| HEAD_RULING | the conjunction pin STAYS object_hue = 0.0 AND shape = 2.0 -- the RED SPHERE -- and this caveat column is printed with it | `/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` `/sphere_caveat/HEAD_RULING` |
| source | chlu/data/gen_value_holdout.py:299 | `/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` `/sphere_caveat/source` |
| what_it_threatens | an arm DAMAGED by the red-sphere deletion could be judged sphere for rendering reasons and score spuriously well on exactly the deleted conjunction | `/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` `/sphere_caveat/what_it_threatens` |
| judged-sphere excess, `landscape_removed` (degraded bound) | -0.3291 | `/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` `/sphere_caveat/measured_upper_bound_degraded_arms/landscape_removed/judged_sphere_excess` |
| judged-sphere excess, `random_atoms` (degraded bound) | -0.3078 | `/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` `/sphere_caveat/measured_upper_bound_degraded_arms/random_atoms/judged_sphere_excess` |
| judged-sphere excess AFTER the red-sphere deletion | +0.0074 | `/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` `/scopes/conjunction_live/shape_report_after/judged_sphere_excess` |
| verdict | the hazard does NOT reproduce on this instrument: a degraded render here is judged CUBE, the shape held out of the split entirely, so the sphere excess is NEGATIVE for both maximally degraded arms. A degrading baseline cannot score spuriously as sphere here. Scope: seed 0, TEST corners, K* = 32, this decoder and this classifier. |  |

**Byte ledger — FOUR PARTS, NEVER NETTED. Measured on the object this deletion ran on.**

| part (ONE seed, id 0) | bytes | how it is obtained |
|---|---|---|
| 1. atoms | 516,096 | stored field |
| 2. covariance + count ledger | 560,608 | stored field |
| 3. address book | 1,060,864 | stored field |
| 4. coupling / hole table | 6,610 | stored field |
| **honest total** | **2,144,178** | stored field |
| **x the 524,288 B reference** | **4.0897** | stored field |


**⛔⛔ THE PROBE THAT BOUNDS THE WHOLE DELETION CLAIM**

> how far the RENDERED answer moved on exactly the requests whose evidence was deleted, in the store's own units. No judge, no decoder. 5,688 requests.

| row | class | seeds (n) | requests | rel. RMS change, address book CARRIED | rel. RMS change, address book NOT carried | max abs change, carried |
|---|---|---|---|---|---|---|
| corners REQUESTING red | probe | 1 | 720 | 6.194e-08 | 0.610083 | 4.768e-07 |
| corners NOT requesting red | probe | 1 | 4,968 | 6.163e-08 | 0.614277 | 4.768e-07 |

**the carry**

| field | value | traced to |
|---|---|---|
| address book bytes changed by the carry | 1,015,808 | `/c/chlu/outputs/w12-single-design-crossover/deletion/probe_test_seed0_K32.json` `/carry/address_book_bytes_changed` |
| address book bytes total | 1,060,864 | `/c/chlu/outputs/w12-single-design-crossover/deletion/probe_test_seed0_K32.json` `/carry/address_book_bytes_total` |
| data rows touched by the carry | 0 | `/c/chlu/outputs/w12-single-design-crossover/deletion/probe_test_seed0_K32.json` `/carry/data_rows_touched` |

> ⛔⛔ **deleting all 11,044 red rows changes the 20 red-requesting corners by EXACTLY ZERO (32.25 -> 32.25 novel valid tuples per corner, judge accuracy 0.8958 -> 0.8958). The store-scoped exactness is real - the uncarried read moves by 0.61 relative RMS - but the basis change is GENERIC, not targeted: 0.6101 on red-requesting corners against 0.6143 on the others. At K* = 32 (rank/width 0.9531-0.9688) the per-well content carries no value-specific information; the value identity lives in the address book, which the deletion CARRIES and does not refit. 'The store has forgotten red' and 'the object no longer answers red' are different statements and only the first is demonstrated.**

> ⭐ at K = 5, A = 64 (rank/width 0.156) wave 8 found the basis change CONCENTRATED: cos_min 0.1999 against 0.9976 for a size-matched random deletion. The targeting of a deletion is a function of rank/width and it disappears exactly where the byte ledger is cheapest.

> ⛔ **language rule the head set:** NEVER 'no changes after deletion'. The sentence is: post-deletion adaptation is LIGHT and DATA-FREE - the covariance and count are downdated by subtraction, the atoms are re-eigendecomposed from the downdated statistic and the address book is carried across the basis change, with NO row of data read - against a reference that can only delete by refitting from scratch without the deleted rows, reading 120,028-131,072 rows to do it.

---

## TABLE 9b - EXACT DELETION (STORE-SCOPED) on the single design, VAL condition, K = 5, 1 seed (id 0).  SMOKE - NOT A CLAIM.

**Class:** SMOKE (the artefact labels itself so) · **Placement:** APPENDIX

> **Notation on this table.** **K_wells**: 5 · **n_factors**: not varied in this table; the n-ladder is T8 · **rule**: every row of T9b states K = 5 and its seed count.

**Pins.** `condition` = val · `K (wells)` = 5 · `seed` = 0 · `definition of exact` = STORE-SCOPED - see TABLES.json /exactness_definition · `feature cache` = generation 2, scope full, 480,000 rows

*Pins traced to:* `/c/chlu/outputs/w12-single-design-crossover/deletion/val_seed0_K5/deletion.json` `/K`; `/c/chlu/outputs/w12-single-design-crossover/deletion/val_seed0_K5/deletion.json` `/seed`; `/c/chlu/outputs/w12-single-design-crossover/deletion/val_seed0_K5/deletion.json` `/condition`; `/c/chlu/outputs/w12-single-design-crossover/deletion/val_seed0_K5/deletion.json` `/A_requested` …

> ⚠ **The artefact's own `SMOKE_NOT_A_CLAIM` field reads true.**

**Seed count**

| field | value | traced to |
|---|---|---|
| seeds (n) | 1 |  |
| PRINTED AS A COLUMN | `seeds (n)` is a column of this table. |  |
| why not 3 or 5 | NR6 (wall clock, seeds 1-2) and NR7 (seeds 3-4 unjudgeable). |  |

| row | class | seeds (n) | K (wells) | seeds (n), traced | rows deleted | fraction of the fit pool | OUR adaptation, seconds | OUR bytes rewritten | OUR data rows read | OUR refit? | REFERENCE refit-without, seconds | REFERENCE refit-without, data rows read | store-scoped exactness, max abs mode dev | ranks agree | survivor corners | paired delta on survivors | DELETED corners | principal-angle cos_min (the basis DID move) | DELETED corners, novel/corner BEFORE | DELETED corners, novel/corner AFTER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) VALUE - every fit row carrying red (`object_hue` index 0) | SMOKE - NOT A CLAIM | 1 | 5 | 1 | 11,024 | 0.084106 | 2.4543 | 4,068,192 | 0 | false | 23.486 | 120,048 | 1.455e-11 | true | 24 | +0.1250 | 0 | 0.183704 | DECLARED: this scope deletes no corner | DECLARED: this scope deletes no corner |
| (b) CONJUNCTION - red AND cube, AS THE TASK SPECIFIES IT -- **VACUOUS** | SMOKE - NOT A CLAIM | 1 | 5 | 1 | 0 | 0.000000 | 0.1874 | 4,068,192 | 0 | false | 11.752 | 131,072 | 0.000e+00 | true | 24 | +0.0000 | 0 | 1.000000 | DECLARED: this scope deletes no corner | DECLARED: this scope deletes no corner |
| (b') RED SPHERE - the DECLARED non-vacuous substitute | SMOKE - NOT A CLAIM | 1 | 5 | 1 | 5,507 | 0.042015 | 0.6205 | 4,068,192 | 0 | false | 10.843 | 125,565 | 2.842e-14 | true | 24 | -0.0833 | 0 | 0.881141 | DECLARED: this scope deletes no corner | DECLARED: this scope deletes no corner |
| (c) ITEM - a random 1 % of fit rows | SMOKE - NOT A CLAIM | 1 | 5 | 1 | 1,311 | 0.010002 | 0.2554 | 4,068,192 | 0 | false | 11.446 | 129,761 | 1.455e-11 | true | 24 | +0.0000 | 0 | 0.999843 | DECLARED: this scope deletes no corner | DECLARED: this scope deletes no corner |

**Byte ledger — FOUR PARTS, NEVER NETTED. Measured on the object this deletion ran on.**

| part (ONE seed, id 0) | bytes | how it is obtained |
|---|---|---|
| 1. atoms | 524,288 | stored field |
| 2. covariance + count ledger | 3,391,552 | stored field |
| 3. address book | 195,584 | stored field |
| 4. coupling / hole table | 6,610 | stored field |
| **honest total** | **4,118,034** | stored field |
| **x the 524,288 B reference** | **7.8545** | stored field |


> ⛔ **language rule the head set:** NEVER 'no changes after deletion'. The sentence is: post-deletion adaptation is LIGHT and DATA-FREE - the covariance and count are downdated by subtraction, the atoms are re-eigendecomposed from the downdated statistic and the address book is carried across the basis change, with NO row of data read - against a reference that can only delete by refitting from scratch without the deleted rows, reading 120,028-131,072 rows to do it.

---

## TABLE 10 - REFUSAL DISPLAY ROWS.  A DISPLAY IS NEVER A RESULT.

**Class:** DISPLAY · **Placement:** MAIN TEXT, labelled DISPLAY

> **Notation on this table.** **K_wells**: 32 on our TEST row, 5 on our VAL row; n/a for a byte-matched rival · **rule**: our two rows are at different K and are labelled as such.

**Pins.** `our object` = single design, Shapes3D value-holdout · `comparator` = refuse-token transformer, the ONLY permitted comparator for a refusal DISPLAY row · `comparator byte pin` = 524,288

*Pins traced to:* `/c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json` `/pins/byte_pin`; `/c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json` `/pins/refuse_threshold`; `/c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json` `/seeds_present`; `/c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json` `/NOT_A_SCORED_RUN` …

**Seed count**

| field | value | traced to |
|---|---|---|
| our rows | 1 |  |
| the comparator's rows | 3 |  |
| NEVER MIX | our 1-seed rows and the comparator's 3-seed rows are printed in one table with the seed count AS A COLUMN, and are NOT plotted together without it. |  |

| row | class | seeds (n) | K (wells) | seeds (n), traced | holes punched | bytes rewritten | data rows read | seconds | refit? | coupling table bytes | layout asserted equal to `gen_field`'s | pair slots / indices checked | requests certified impossible | AUROC, impossible vs possible | separation (mean score gap) | refuse rate on impossible | FALSE refusal on possible | params | requests built from deleted rows, tested | of those, CERTIFIED impossible | a refusal row applies? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **OURS - the coupling/hole table**, TEST, K* = 32, seed 0 | DISPLAY - never a result | 1 | 32 | 1 | 2 | 2 | 0 | 0.0000 | false | 1,322 | true | 1,322 / 60 | — | — | — | — | — | — | — | — | — |
| OURS - the coupling/hole table, VAL, K = 5, seed 0 (SMOKE) | DISPLAY - SMOKE, never a result | 1 | 5 | 1 | 2 | 2 | 0 | 0.0000 | false | 1,322 | true | — | — | — | — | — | — | — | — | — | — |
| COMPARATOR - refuse-token transformer, VALUE deletion (red) | DISPLAY comparator - 3 seeds, FEATURE-LEVEL, off the hub's instrument | 3 | n/a - a learned rival has no wells | 3 | — | 521,916 / 521,916 / 521,916 | 4,096 / 4,096 / 4,096 | 10.5220 / 12.1854 / 7.4278 | — | — | — | — | 1,352 / 1,352 / 1,352 | 0.8692 / 0.8747 / 0.8755 | 0.4549 / 0.4717 / 0.4673 | 0.6709 / 0.7109 / 0.6846 | 0.0947 / 0.1270 / 0.1094 | 130,479 / 130,479 / 130,479 | — | — | — |
| COMPARATOR - refuse-token transformer, CONJUNCTION deletion (red AND sphere) | DISPLAY comparator - 3 seeds, FEATURE-LEVEL, off the hub's instrument | 3 | n/a - a learned rival has no wells | 3 | — | 521,916 / 521,916 / 521,916 | 4,096 / 4,096 / 4,096 | 10.4826 / 7.2859 / 6.9145 | — | — | — | — | 667 / 667 / 667 | 0.8269 / 0.8340 / 0.8133 | 0.3671 / 0.3747 / 0.3455 | 0.5591 / 0.5709 / 0.5556 | 0.0781 / 0.0859 / 0.1055 | 130,479 / 130,479 / 130,479 | — | — | — |
| ITEM deletion (random 1 %) - THERE IS NO REFUSAL ROW AT ALL | DECLARED NOT-APPLICABLE, measured - never a zero | 3 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 164 / 164 / 164 | 0 / 0 / 0 | false / false / false |

> ⛔ **WHAT THE TWO SIDES OF THIS TABLE ARE, AND ARE NOT:** OUR side and the COMPARATOR side are NOT the same quantity and are not differenced. Ours is a cost line: 2 bytes, 0 data rows, no refit, on the object T9 deleted from. The comparator's is a SEPARATION diagnostic measured at the FEATURE level by spoke/deletion-baselines on its own 16,384-row fit pool, with no judge and no decoder. The head-to-head DISPLAY leg - our refusal rate against the refuse-token transformer's render rate on the same post-deletion requests - is NR9, a DECLARED NOT-RUN: the crossover hub's wall clock ran out after the scored rows and the deletion legs.

> ⛔ **THE POINT THE HOLES MAKE:** one hole (index 920, red AND cube) is refusable although NO RED CUBE WAS EVER WRITTEN - cube is held out of the split entirely. Refusal is a SEPARATE mechanism from forgetting and does not need a deleted row to have existed. T9's probe makes that separation load-bearing rather than rhetorical: since the downdate does not stop the object answering red, the hole is the only mechanism here that can.

> ⭐ **VERDICT.** our side of every refusal row is MEASURED (2 bytes, 0 rows read, no refit, layout asserted equal to the field store's over 1,322 pair slots). The head-to-head against the refuse-token transformer is NR9. The comparator itself is measured at 3 seeds and separates at AUROC 0.8133-0.8755 with a 7.8-12.7 % FALSE refusal rate on possible requests, which is why a DISPLAY comparison against it would have to print that false-refusal column too.

---

## TABLE 11 - THE TWO ABLATIONS: the composition-only store (composes, cannot forget) and the ITEM store (forgets, loses composition). w5 compose-field, TEST, 3 seeds, one pin.

**Class:** ablation · **Placement:** MAIN TEXT

> **Notation on this table.** **K_wells**: 5 · **n_factors**: row 3 = 2-way, row 4 = n-way; this table does not vary n · **rule**: one pin (524,288 B store bytes), one condition (TEST), 3 seeds on every row.

**Pins.** `condition` = test · `store bytes` = 524,288 · `K (wells)` = 5 · `metric` = distinct novel valid tuples per corner · `corners` = row 3: 60, ceiling 36 / row 4: 49, ceiling 72

*Pins traced to:* `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json` `/arms/pca_tensor/store_bytes`; `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json` `/corner_counts/r3_comp_2way`; `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json` `/corner_counts/r4_comp_nway`; `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json` `/corner_ceilings/r3_comp_2way` …

**Seed count**

| field | value | traced to |
|---|---|---|
| seeds on every row | 3 |  |
| PRINTED AS A COLUMN | `seeds (n)` = 3 on every row of T11. |  |
| note | these are wave-5 scored TEST legs and were never re-run at 5 seeds. |  |

| row | class | seeds (n) | pin (store bytes) | address-book bytes | params @ the pin | row 3 (2-way), 60 corners, ceiling 36 | row 3, fraction of ceiling | row 4 (n-way), 49 corners, ceiling 72 | row 4, fraction of ceiling | class as the artefact records it |
|---|---|---|---|---|---|---|---|---|---|---|
| **`pca_tensor` - the COMPOSITION-ONLY store** (no sufficient statistics, so no exact deletion) | ABLATION - composes, cannot forget | 3 | 524,288 / 524,288 / 524,288 | 189,344 / 195,064 / 195,064 | 131072 / 131072 / 131072 | 29.7167 / 31.6167 / 31.4500 | 0.8255 / 0.8782 / 0.8736 | 63.6122 / 64.3265 / 67.6531 | 0.8835 / 0.8934 / 0.9396 | ARM / ARM / ARM |
| **`oracle_content_items` - the ITEM store**, 64 whole written rows read by an ORACLE | ABLATION - forgets, loses composition (and it is a CEILING, not an arm) | 3 | 524,288 / 524,288 / 524,288 | 0 / 0 / 0 | 131072 / 131072 / 131072 | 18.7500 / 18.2333 / 20.4667 | 0.5208 / 0.5065 / 0.5685 | 32.3878 / 39.1224 / 41.7755 | 0.4498 / 0.5434 / 0.5802 | NOT AN ARM (ceiling, 64 REAL ITEMS @524,288 B) / NOT AN ARM (ceiling, 64 REAL ITEMS @524,288 B) / NOT AN ARM (ceiling, 64 REAL ITEMS @524,288 B) |
| `rival_mlp_180k` - the learned reference these two are read against | comparator | 3 | 0 / 0 / 0 | 0 / 0 / 0 | 178952 / 178952 / 178952 | 25.6833 / 29.1333 / 29.0333 | 0.7134 / 0.8093 / 0.8065 | 53.4694 / 55.6531 / 59.8571 | 0.7426 / 0.7730 / 0.8313 | RIVAL / RIVAL / RIVAL |
| `oracle_content_fine` - the ABSOLUTE content ceiling at the same pin | ceiling, NOT AN ARM | 3 | 524,288 / 524,288 / 524,288 | 0 / 0 / 0 | 131072 / 131072 / 131072 | 33.4667 / 34.3667 / 34.9667 | 0.9296 / 0.9546 / 0.9713 | 69.0816 / 68.1633 / 70.7347 | 0.9595 / 0.9467 / 0.9824 | NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, fine K) / NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, fine K) / NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, fine K) |

**The price of keeping items**

| field | value | traced to |
|---|---|---|
| item store minus the learned reference, row 4, per seed | -21.0816 / -16.5306 / -18.0816 | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` `/arms/oracle_content_items/r4_comp_nway/novel_tuples_per_corner … one entry per seed` |
| the composition-only store minus the item store, row 4, per seed | +31.2245 / +25.2041 / +25.8776 | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json`; `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` `/arms/pca_tensor/r4_comp_nway/novel_tuples_per_corner … one entry per seed` |
| reading | the content that WINS composition forfeits exact per-item deletion by construction, and that forfeit is priced here at 18.56 tuples/corner on row 4 (3-seed mean). The item store is a CEILING handed the true feature, not an arm, so the price is a lower bound on what an item store would actually cost. |  |

**The other half of the item-store ablation — it forgets, and loses composition**

| field | value | traced to |
|---|---|---|
| what | the other half of the item-store ablation, measured by w5 hub/delete on its own object: EVICTION refuses perfectly on the deleted set and FAILS the composition bar. |  |
| eviction: refusal on the deleted set | 1.0000 / 1.0000 / 1.0000 | `/c/chlu/outputs/w5-delete/run1/report.json` `/across_seeds/evict_refuse_on_deleted_set/per_seed/0 … one entry per seed` |
| eviction: composition mean delta (bar is +/-1.0) | -2.0899 / -0.9101 / -6.7416 | `/c/chlu/outputs/w5-delete/run1/report.json` `/across_seeds/evict_composition_mean_delta/per_seed/0 … one entry per seed` |
| the HOLE instead: flag on the deleted set | 1.0000 / 1.0000 / 1.0000 | `/c/chlu/outputs/w5-delete/run1/report.json` `/across_seeds/hole_flag_on_deleted_set/per_seed/0 … one entry per seed` |
| the HOLE instead: composition mean delta | +0.0000 / +0.0000 / +0.0000 | `/c/chlu/outputs/w5-delete/run1/report.json` `/across_seeds/hole_composition_mean_delta/per_seed/0 … one entry per seed` |
| row-removal seconds | 0.0000 / 0.0000 / 0.0000 | `/c/chlu/outputs/w5-delete/run1/report.json` `/across_seeds/row_removal_s/per_seed/0 … one entry per seed` |
| retrain-from-scratch seconds | 3.7900 / 8.4766 / 6.6760 | `/c/chlu/outputs/w5-delete/run1/report.json` `/across_seeds/retrain_from_scratch_s/per_seed/0 … one entry per seed` |
| verdict | the operation that is exact on bytes AND free on behaviour is the HOLE, not deletion in general: evicting `red` refuses at 1.0000 and costs -2.0899 / -0.9101 / -6.7416 composition against a declared +/-1.0 bar, while the hole refuses at 1.0000 and costs exactly 0.0000. |  |

> ⛔ **cross reference:** T8's byte ledger prices the same composition-only ablation on the LADDER's own fitted objects: `pca_tensor` at 1.358x-1.372x the 524,288 B reference against the single design's 23.886x-24.875x, for composition scores identical at every rung and every seed.

---

## TABLE 12 - THE K-CURVE: the four-part byte ledger and val composition against K = 5..32 WELLS. n is NOT varied in this table.

**Class:** SMOKE (the val sweep labels itself so) + exact arithmetic on the partition shape · **Placement:** APPENDIX, and the figure is a main-text candidate

> **Notation on this table.** **K**: the number of WELLS - this is the axis of T12 · **n**: NOT VARIED. Every row of T12 reads the same 24 val corners. The n-ladder is T8 and it sits at one K. · **rule**: T12 is a K-axis; T8 is an n-axis; they are never the same plot.

**Pins.** `condition` = val · `A_requested` = 64 (63 at K = 32, where A < well width binds) · `rank per well` = capped strictly below the well width · `selection rule` = K* = max{K : v(5) - mean_seeds v(K) <= 0.02}, frozen 2026-09-21T11:10:24Z · `crossover rule` = min{K : drop > 0.10}

*Pins traced to:* `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/rule/within`; `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/rule/crossover`; `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/rule/declared_at`; `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/K_star` …

> ⚠ **The two source artefacts' own `SMOKE_NOT_A_CLAIM` fields read true / true (CROSSOVER-val.json / LEDGER-ACROSS-K.json).**

**Seed count**

| field | value | traced to |
|---|---|---|
| PRINTED AS A COLUMN | `seeds (n) at this K` is a column: 3 at K = 5 and K = 32, 1 at every other K. |  |
| NEVER MIX | a plot of this table MUST print n per point - the two ends carry 3 seeds and the seven middle points carry 1. A line drawn through them without n is a lie about the evidence. |  |

| row | class | seeds (n) | K (wells) | seeds (n) at this K | well width min / max | rank/width max | val composition, mean over the seeds run at this K | drop vs K = 5 (the frozen rule) | 1. atoms B | 2. covariance + count B | 3. address book B | 3. address book status | 4. coupling / hole table B | TOTAL, parts 1+2+4 ONLY (no address book) | x reference, parts 1+2+4 | TOTAL, the HONEST FOUR-PART sum | x reference, four-part | 1x params a matched rival would carry |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **K = 5** | SMOKE (val sweep) + exact byte arithmetic | 3 | 5 | 3 | 409 / 410 | 0.1565 | 0.895255 | +0.000000 | 524,288 | 3,391,552 | 195,584 | measured | 6,610 | 3,922,450 | 7.4815 | 4,118,034 | 7.8545 | 1,029,508 |
| **K = 8** | SMOKE (val sweep) + exact byte arithmetic | 1 | 8 | 1 | 256 / 256 | 0.2500 | 0.883681 | +0.011574 | 524,288 | 2,133,280 | 295,424 | measured | 6,610 | 2,664,178 | 5.0815 | 2,959,602 | 5.6450 | 739,900 |
| **K = 10** | SMOKE (val sweep) + exact byte arithmetic | 1 | 10 | 1 | 204 / 205 | 0.3137 | 0.882523 | +0.012731 | 524,288 | 1,713,872 | 361,984 | measured | 6,610 | 2,244,770 | 4.2816 | 2,606,754 | 4.9720 | 651,688 |
| **K = 12** | SMOKE (val sweep) + exact byte arithmetic | 1 | 12 | 1 | 170 / 171 | 0.3765 | 0.892940 | +0.002315 | 524,288 | 1,434,272 | 428,544 | measured | 6,610 | 1,965,170 | 3.7483 | 2,393,714 | 4.5656 | 598,428 |
| **K = 16** | SMOKE (val sweep) + exact byte arithmetic | 1 | 16 | 1 | 128 / 128 | 0.5000 | 0.890625 | +0.004630 | 524,288 | 1,084,768 | 561,664 | measured | 6,610 | 1,615,666 | 3.0816 | 2,177,330 | 4.1529 | 544,332 |
| **K = 20** | SMOKE (val sweep) + exact byte arithmetic | 1 | 20 | 1 | 102 / 103 | 0.6275 | 0.887731 | +0.007523 | 524,288 | 875,104 | 694,784 | measured | 6,610 | 1,406,002 | 2.6817 | 2,100,786 | 4.0069 | 525,196 |
| **K = 24** | SMOKE (val sweep) + exact byte arithmetic | 1 | 24 | 1 | 85 / 86 | 0.7529 | 0.898148 | -0.002894 | 524,288 | 735,328 | 827,904 | measured | 6,610 | 1,266,226 | 2.4151 | 2,094,130 | 3.9942 | 523,532 |
| **K = 28** | SMOKE (val sweep) + exact byte arithmetic | 1 | 28 | 1 | 73 / 74 | 0.8767 | 0.897569 | -0.002315 | 524,288 | 635,488 | 960,504 | measured | 6,610 | 1,166,386 | 2.2247 | 2,126,890 | 4.0567 | 531,722 |
| **K = 32**  (= K\*) | SMOKE (val sweep) + exact byte arithmetic | 3 | 32 | 3 | 64 / 64 | 0.9844 | 0.929012 | -0.033758 | 516,096 | 560,608 | 1,059,824 | measured | 6,610 | 1,083,314 | 2.0663 | 2,143,138 | 4.0877 | 535,784 |

**The control gate at K\***

| field | value | traced to |
|---|---|---|
| landscape_removed, per seed | 0.0000 / 0.0000 / 0.0000 | `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/control_gate_at_K_star/landscape_removed/per_seed/0 … one entry per seed` |
| landscape_shuffled, per seed | 0.0000 / 0.0000 / 0.0000 | `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/control_gate_at_K_star/landscape_shuffled/per_seed/0 … one entry per seed` |
| landscape_shuffled_across_wells, per seed | 0.0000 / 0.0000 / 0.0000 | `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/control_gate_at_K_star/landscape_shuffled_across_wells/per_seed/0 … one entry per seed` |
| random_atoms, per seed | 0.0000 / 0.0000 / 0.0000 | `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/control_gate_at_K_star/random_atoms/per_seed/0 … one entry per seed` |

> ⛔ **THE CAPTION THAT MUST TRAVEL WITH THIS TABLE - TWO CURVES, NOT ONE:** TWO DIFFERENT TOTALS ARE PRINTED AND THEY MOVE IN DIFFERENT DIRECTIONS. (1) PARTS 1+2+4 ONLY - atoms, covariance/count and the coupling table, with NO address book - FALLS MONOTONICALLY, 7.4815x -> 2.0663x from K = 5 to K = 32, because the covariance shrinks 6.0x (3,391,552 -> 560,608 B). (2) THE HONEST FOUR-PART TOTAL - the same three parts PLUS the measured address book - is U-SHAPED: 7.8545x at K = 5, a minimum of 3.9942x at K = 24, and 4.0877x at K* = 32, because the ADDRESS BOOK GROWS 5.4x (195,584 -> 1,059,824 B) and at the fine end it grows FASTER than the covariance falls. ANY SENTENCE THAT QUOTES '7.48x -> 2.07x' IS QUOTING CURVE (1) AND HAS DROPPED PART 3. The mechanism for curve (1) is not 'the covariance shrinks faster than the address book grows' - there is no address book in it. Curve (2), which does contain one, says the opposite at the fine end.

> ⛔ **the object never reaches the retired pin:** the minimum honest four-part total over the whole sweep is 3.9942x 524,288 B, at K = 24. 'Matched bytes at 524,288 B' is not available to this object at rank 64 under any K, which is why this wave's 1x pin is the object's own total and 524,288 B is a REFERENCE column.

> ⭐ **VERDICT.** every K in {5, 8, 10, 12, 16, 20, 24, 28, 32} satisfies the frozen rule, so K* = 32 - the finest partition swept. The crossover threshold of 0.10 is never reached; the largest drop anywhere is +0.012731 at K = 10. The range was NOT extended to manufacture a crossover.

---

## ⭐ COMPLETENESS GATES — a published file is not a finished file

> a published file is not a finished file. Three times this programme a READY json appeared while its author was still writing. Every wave-12 artefact this file gates on was checked COMPLETE - expected counts and non-null fields, not merely present - before a cell was emitted from it.

| artefact | check | expected | found | verdict |
|---|---|---|---|---|
| `w12-single-design-crossover` | END_UTC.txt present | a UTC stamp | 2026-09-21T14:28:13Z | CLOSED |
| `w12-composition-ladder` | END_UTC.txt present | a UTC stamp | 2026-09-21T16:54:33Z | CLOSED |
| `w12-deletion-baselines` | END_UTC.txt present | a UTC stamp | 2026-09-21T12:16:00Z | CLOSED |
| `w12-cache-gen2-seeds-3-4` | END_UTC.txt present | a UTC stamp | 2026-09-21T15:28:41Z | CLOSED |
| `LADDER.json` | rungs present and no leg failed | rungs {2,3,4,5,6,shipped}, `not_run` = [] | 6 rungs, not_run = [] | COMPLETE |
| `LADDER.json` | arms per rung x seeds per arm | 13 arms x 3 seeds at every rung | 13 arms, 3 seeds at n = 2 | COMPLETE |
| `BYTES.json` | fitted legs with a measured ledger | 18 legs per store arm | 18 / 18 | COMPLETE |
| `scored/run1_seed0_K32/scored.json` | it is a scored run and its corners add up | `SCORED` true, 60 + 49 = 109 corners | SCORED = True, 60 + 49 = 109 | COMPLETE |
| `CROSSOVER-val.json` | every swept K carries a ledger and a mean | 9 K, each with `ledger_four_part` and `val_composition_mean` | 9 K, all non-null | COMPLETE |
| `CACHE-READY.json` | seeds 3 and 4 READY with 59 block hashes each | status READY, n_blocks 59, len(block_sha256) 59, shape match true | seed 3: READY / 59 / 59 / True ; seed 4: READY / 59 / 59 / True | COMPLETE - and it STILL buys no row; see NR7 |
| `RIVALS-READY.json` | which seeds have FITTED rivals | status ready, a seed list per family | status 'ready', seeds_complete [0, 1, 2], rivals_seeds_present[mlp] [0, 1, 2] | COMPLETE - and it names 3 seeds, not 5 |
| `BASELINES-READY.json` | records present on every seed | 150 records, seeds [0, 1, 2] | 150 records, seeds [0, 1, 2] | COMPLETE |

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

> ⛔ **w12r qualifier.** w12r/MF16. `at K >= 36 composition collapses` is TRUE OF THE ONE CONFIGURATION AT K >= 36 WHOSE COMPOSITION WAS MEASURED, and must be stated that way. The same hub's 18:32Z `ward_balanced` sweep lists FIVE configurations that fit the 524,288 B pin at float64 - K=36 (span 144, 523,680 B, 0.999x), K=38 (span 266, 523,744 B, 0.999x), K=40 (span 360, 518,080 B, 0.988x), K=42 (span 504, 522,704 B, 0.997x), K=48 (span 864, 521,984 B, 0.996x) - and the composition of FOUR of them was never measured. The one that was measured is K=40 with A=7, span 280, 519,794 B = 0.991x (the 23:22Z entry), which sits BELOW the sweep's own K=40 row and below the composing object's realised span of 308: three of the five that fit the pin carry MORE deployed span than the object that composes at 0.8918. The threshold is also DTYPE-CONDITIONAL: the same hub's earlier sweep prints a float32 matched-bytes point at K=21, A=94, 517,952 B = 0.988x the pin, span 336, excluded by ARGUMENT (the peer's measured 1e-10..5e-10 downdate residuals are held not to survive float32) and NEVER MEASURED. The supported form is: NO OBJECT WITH A FLOAT64 SUFFICIENT STATISTIC composes and forgets inside 524,288 B, on the one such configuration measured.

> *Traced to:* `/c/chlu/outputs/w8-single-design/LOG.md` LOG.md, the 18:32Z ward_balanced sweep; the 23:22Z K=40/A=7 entry; LOG.md:99 for the float32 matched-bytes point

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

### NR6 — single-design rows 3 and 4 at K* = 32 on TEST at SEEDS 1 AND 2 (T7, T9, T10 our rows)

**Status:** NOT RUN - 1 of 3 scored runs spent, 2 remain

**What does exist:** run 1/3, seed 0, at `scored/run1_seed0_K32/scored.json`: 60 pair corners + 49 n-way corners, 10 arms, control gate passed, and the deletion legs beside it.

**Why:** wall clock. Run 1/3 cost 1,242.3 s of address fit plus 10 arm passes over 109 corners plus four deletion legs. Budget after wave 12: `single-design` 1 of 3 spent, 2 remain. The FIRST thing run 2/3 should buy is a TEST number at K = 5, because the third pre-declared falsification - 'composition at K* not within 0.02 of K = 5 on TEST though it was on val' - is NOT directly testable while only K* has a TEST row.

**Evidence:**
- `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` — /run_n
- `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` — /seed
- `/c/chlu/outputs/w12-single-design-crossover/FINDINGS.md` — FINDINGS.md section 11

**Launch line:**

```bash
# from the crossover hub's worktree, one scored run per seed
for SEED in 1 2 ; do
  python scripts/w12_scored_leg.py --seed $SEED --K 32 --condition test \
    --out /c/chlu/outputs/w12-single-design-crossover/scored/run${SEED}_seed${SEED}_K32
done
```

**Cost:** about 1,242 s of address fit + about 10 arm passes over 109 corners per seed on FORGIS_14 CPU, and each consumes one of the two remaining `single-design` scored runs.

### NR7 — ANY row at 5 seeds - seeds 3 and 4 on the ladder (T8) and on the deletion rows (T9)

**Status:** NOT RUN. ⭐ The generation-2 CACHES for seeds 3 and 4 DO exist, are complete and are independently re-hashed - and they still buy no row.

**What does exist:** `CACHE-READY.json` (produced 2026-09-21T15:27:15Z) carries BOTH seeds at `status: READY`, `generation: 2`, `scope: full`, 480,000 rows, `array_shape [480000, 128, 4, 4]`, 59 block hashes each, `shape_dtype_matches_seeds_0_2: true`, bridge complete, bridge gate PROCEED. `VERIFY.json` re-hashes every block from the bytes on disk from a process that does not import chlu: `all_pass: true`. The SHIPPED re-encode gate passes on both seeds at `frac_differing 0.0`, `max_ulp_delta 0.0`.

**Why:** the cache is not the binding constraint - the JUDGE is. Every composition number in wave 12 is produced by `fw.Judge(seed)`, which loads a frozen decoder and a hash-asserted external classifier. Measured by both wave-12 hubs independently: `fw.classifier_dir(3)` and `fw.classifier_dir(4)` do NOT exist, `decoder_v1_s3.pkl` does not exist, and spoke A's own readiness manifest records `rivals_seeds_present` = [0, 1, 2] for every one of its seven rival families, so the rivals are missing for seeds 3 and 4 too. `Judge(3)` raises FileNotFoundError. SEPARATELY: the caches landed AFTER both scoring hubs closed their seed gates - the ladder gated at 11:31:33Z when the file was absent, and the crossover read a 12:27:01Z version in which seed 4 still read `status: NOT_RUN`, 0 blocks. Neither hub could have used them even if the judge had existed.

**Evidence:**
- `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json` — /seeds/3/status
- `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json` — /seeds/4/status
- `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json` — /produced_utc
- `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/VERIFY.json` — /all_pass
- `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/reencode_gate_seeds_3_4.json` — /verdict/passes
- `/c/chlu/outputs/w12-composition-ladder/FINDINGS.md` — FINDINGS.md section 1
- `/c/chlu/outputs/w12-single-design-crossover/FINDINGS.md` — FINDINGS.md section 4
- `/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json` — /rivals_seeds_present/mlp
- `/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json` — /seeds_complete

**Launch line:**

```bash
# the missing pieces are the JUDGE's, not the cache's: for SEED in 3 4
python -m chlu.experiments.exp_gen_bridge --stage all --seed $SEED     # decoder_v1_s$SEED.pkl
python -m chlu.experiments.exp_gen_external_classifier --seed $SEED    # classifier_dir($SEED)
python -m chlu.eval.rivals.w4_api --fit --seed $SEED                   # spoke-a-rivals/seed_$SEED
# only then re-run the ladder and the deletion legs at 5 seeds
```

**Cost:** a decoder + external classifier + five fitted rival families per seed, then the 18-leg ladder and the deletion legs again at 5 seeds. The caches themselves cost 396.6 s (seed 3) and 340.9 s (seed 4) and are already paid.

### NR8 — the 1x BYTE-MATCHED COMPARATOR COLUMN on T7, and three of the four 1x families on T8

**Status:** NOT RUN on T7 entirely; on T8 ONE family (an MLP, `learned_1x`) exists and three do not

**What does exist:** T8's `learned_1x` is an MLP fitted at the arm's own measured total. That is the whole of the 1x opponent set anywhere in wave 12.

**Why:** `BASELINES-READY.json` pins `param_target = 131,072` (= 524,288 B / 4) and names no 1x-matched entry point; refitting four families at 535,784 parameters was not affordable beside the sweep. ⛔ The direction of the error runs AGAINST the programme: a 1x-matched rival would carry about 4.09x these parameters and would be STRONGER, so the available comparison FLATTERS the arm and no row is reported as a 1x result. On T8 the single measured 1x family is therefore the WEAKEST available 1x opponent, and the arm is already NEGATIVE against it at n = 2, 4, 5 and 6.

**Evidence:**
- `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` — /baseline_matching/1x_column
- `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` — /baseline_matching/1x_reason
- `/c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json` — /pins/param_target
- `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` — /per_K/32/baseline_match_1x/target_params_float32
- `/c/chlu/outputs/w12-composition-ladder/FINDINGS.md` — FINDINGS.md section 9

**Launch line:**

```bash
# fit the four families at THIS object's own total, not at the retired pin
python -m chlu.eval.rivals.w12_deletion_baselines --fit \
  --arms mlp,lstm,transformer,cont_mlp --param-target 535784 --seed 0 \
  --out /c/chlu/outputs/w12-deletion-baselines/comparators_1x
# then re-score T7's rows 3 and 4 against them
```

**Cost:** four families at about 4.1x the parameter count of the fitted set, plus one re-score of 109 corners per family. No new scored run is consumed if it rides on the same run as NR6.

### NR9 — the REFUSAL DISPLAY HEAD-TO-HEAD (T10) - our refusal rate against the refuse-token transformer's render rate on the SAME post-deletion requests

**Status:** NOT RUN

**What does exist:** BOTH sides exist separately and are printed in T10. Ours: 2 holes, 1 byte each, 0 data rows read, no refit, layout asserted equal to `gen_field`'s over 1,322 pair slots. Theirs: the refuse-token transformer at 3 seeds with a measured separation (AUROC 0.8133-0.8755, false refusal on possible 7.8-12.7 %). ⛔ They are NOT the same quantity and T10 does not difference them.

**Why:** the crossover hub's wall clock ran out after the scored rows and the deletion legs. ⚠ Carried regardless: wave 9's refuser separated possible from impossible at only +0.047 and a ridge probe on the same observation scored AUC 0.48-0.55, i.e. chance, so a DISPLAY comparison against a refuser of that quality is not yet informative - which is an argument for printing the comparator's FALSE-REFUSAL column beside any such row, not for skipping the row.

**Evidence:**
- `/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` — /display_refusal/comparator_status
- `/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` — /display_refusal/carried_caveat
- `/c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json` — /comparators/refuse_token/entry_point

**Launch line:**

```bash
# both sides are already on disk; this is one scoring pass, not a fit
python scripts/w12_refusal_display.py --seed 0 --K 32 --condition test \
  --ours /c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32 \
  --comparator transformer__refuse_token__value-object_hue-0 \
  --out /c/chlu/outputs/w12-single-design-crossover/display/refusal_test_seed0_K32
```

**Cost:** one pass over the post-deletion request set; no fit and no scored run is consumed.

### NR10 — `generation-table` run 3 on the K* object against the full rival set

**Status:** NOT RUN - ⭐ the run remains UNSPENT

**What does exist:** nothing on this object; `exp_gen_cell.py` carries its own arm registry.

**Why:** `exp_gen_cell.py`'s registry (`clu`, `store_removed`, `knn_proto`, `lace`, `ebm_compose`, `dynread`) would have to be wired to accept a K* single-design object. That is a build, and wave 12 had no wall clock for it beside the sweep. It is carried here because the wave plan commissions it in the same clause as T7's rows.

**Evidence:**
- `/c/chlu/outputs/w12-single-design-crossover/FINDINGS.md` — FINDINGS.md section 11
- `/c/chlu/evidence/outputs/advisor-wave-plan-w12-w14.md` — section 1, hub/single-design-crossover item 2

**Launch line:**

```bash
# wire the K* object into the cell's arm registry, then
python -m chlu.experiments.exp_gen_cell --stage cell --seed 0 --arm single_design_K32
python /c/chlu/outputs/hub-generation-table/relaunch/scored/run_arm.py <out_dir>/seed0 <spec>.json
```

**Cost:** a build plus one leg; consumes the last `generation-table` scored run.

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

### C5 (w12r/MF3)

- **As written:** TABLE 4 (the lead result) printed EIGHT property rows on the w9 leg and FIVE on the w8 leg, and NO BYTE ROW on either.
- **Artefact says:** the w8 gate file records the overrun as named fields: `honest_total_bytes` = 12,869,482 / 12,500,170 / 12,409,722 B and `times_over_budget` = 24.5466 / 23.8422 / 23.6697 against the 524,288 B pin. The w9 ledger records `deployed_bytes` = 1,642,752 with `statistic_bytes` = 35,904 (2.19%) declared beside it.
- **Effect:** the paper penalised the compose-and-forget object of section 5 at 7.48x the pin and exempted its own lead result at 24.55x - three times worse - one section later. A referee who opens the cited JSON reads `times_over_budget: 24.5466` in the file cited for the headline.
- **What this file does:** every table that carries a store now prints its FOUR-PART byte ledger as its own block, never netted: T1 section 1c-quater, T2, T4. T3 carries no store and says so.
- **Traced to:** `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/per_seed/0/times_over_budget`; `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/per_seed/1/times_over_budget`; `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/per_seed/2/times_over_budget` …

### C6 (w12r/MF4)

- **As written:** `exact` was used without a definition anywhere in the tables.
- **Artefact says:** the frozen autoencoder's training inventory includes eval-bank items (`ae_train_items`: eval_bank_train 543 / 529 / 552), and `refit` = False: the encoder and decoder fitted with the deleted rows present are untouched.
- **Effect:** without the definition the claim reads as retrain-equivalence of the generative model, which is not what was measured, and is the standard unlearning objection.
- **What this file does:** `/exactness_definition` is a top-level citable field (STORE-SCOPED, frozen encoder and decoder never refit); T4's caption carries it; every place the word appears is flagged, and the two senses (exact JUDGE vs exact DELETION) are separated.
- **Traced to:** `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/ae_train_items`; `/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json` `/ae_train_items`; `/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` `/ae_train_items` …

### C7 (w12r/MF7)

- **As written:** TABLE 4's w8/downdate leg was `class RESULT`.
- **Artefact says:** `/note`: "No scored run and no claim: these are properties of the object (deviations, byte counts, row counts), not outcomes. hub/single-design scores the object." The hub commissioned to score it scored 0 of 3 (DECLARED NOT-RUN NR3, full budget intact).
- **Effect:** a RESULT label on a leg nobody scored is the one thing in this table a referee can call inflation. It costs the paper nothing to drop: two independent implementations is still the claim.
- **What this file does:** relabelled `ASSERTION BATTERY (object property) - 3 seeds - TEST condition - 0 SCORED RUNS`. The artefact was NOT relabelled to match the table.
- **Traced to:** `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/note`

### C8 (w12r/MF8 + MF9)

- **As written:** T4 printed `/downdate_equals_refit/max_abs_dev_per_well_by_seed` (2.365e-14 / 1.126e-09 / 7.105e-15) and declared ONE unresolvable leg.
- **Artefact says:** `/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev` - the field the source spoke's own FINAL REPORT prints, worst over the pinned 11,024-row deletion AND the random-64 deletion - is 4.547e-13 / 1.126e-09 / 2.842e-14: 19x larger on seed 0 and 4x larger on seed 2. And TWO legs sit below their own reproducibility floor (seed 1: 1.126e-09 vs 4.282e-09; seed 2: 2.842e-14 vs 5.684e-14), not one.
- **Effect:** a correction whose whole point was to stop an understated figure being printed was printing a different understated figure; and the paper was hiding two thirds of the strongest fact it has, since an unresolvable residual is the best available outcome for an exactness claim.
- **What this file does:** T4 and T5 print the spoke's own row first, the narrower field beside it and labelled, each leg's own floor, and a resolvable? row. Figure 4 plots the corrected field and the floors.
- **Traced to:** `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/per_seed/0/downdate_equals_refit/worst_max_abs_dev`; `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/per_seed/1/downdate_equals_refit/worst_max_abs_dev`; `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/per_seed/2/downdate_equals_refit/worst_max_abs_dev` …

### C9 (w12r/MF10)

- **As written:** TABLE 1 was labelled CMAZE-1, the same label as TABLE 3.
- **Artefact says:** the w9/maze FINAL REPORT defines the scored environment as CMAZE-2, "a REQUEST layer over CMAZE-1. The engine, the solver and the exact integer judge are `w8_envscout.maze_colour`, imported unmodified". TABLE 3 is CMAZE-1 proper.
- **Effect:** under one label the triple 0.4980 (published CMAZE-1) / 0.5317 (CMAZE-1 at the harness tier) / 0.9037 (CMAZE-2) is unreadable, and correction C4 - which says 0.4980 is a property of the REQUEST LAYER, not of CMAZE-1 - contradicts itself on the page.
- **What this file does:** T1, its caption, its pins, figures 1 and 5 and every w9 maze row are relabelled CMAZE-2; T3 is labelled CMAZE-1 PROPER; `/environments` defines both in one line each; T3's caption no longer says `the same environment`; the two never share a table.
- **Traced to:** `/c/chlu/outputs/w9-maze/LOG.md` `LOG.md:366-372`; `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity`; `/c/chlu/outputs/w9-harness/harness_seed0.json` `/columns/blind-sequential (key order from A)/validity`

### C10 (w12r/MF11)

- **As written:** "FRAGMENT-LOOKUP r=1/r=2 at 0.0000 everywhere".
- **Artefact says:** true of the CROSS-MAZE quantity (0.0006 / 0.0000) and false of the SAME-MAZE quantity on w9's own split (0.0741 / 0.5105 pooled, 0.5114 as the unweighted seed mean).
- **Effect:** the false half is also the paper's strongest exhibit: same-maze r=2 reaches 81% of row 2's pipeline ceiling of 0.6319 where the CLU reaches 20%.
- **What this file does:** T1 section 1c-bis prints both, labelled, with the charter-B.3 not-like-for-like caveat. T3's fragment rows are labelled CMAZE-1 tier columns.
- **Traced to:** `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/env_columns/fragment/FRAG-r2/validity`; `/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json` `/env_columns/fragment/FRAG-r2/validity`; `/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` `/env_columns/fragment/FRAG-r2/validity` …

### C11 (w12r) - which mean is 0.5105?

- **As written:** the same-maze FRAGMENT-LOOKUP r=2 figure is quoted as 0.5105 by `w9-maze/LOG.md` and by REFEREE-v0.md MF11.
- **Artefact says:** the three per-seed validities are 0.5342 / 0.5364 / 0.4636 at n = 584 / 604 / 632. The UNWEIGHTED seed mean is 0.51143; the n-WEIGHTED (pooled) mean is 0.51044, which rounds to the quoted 0.5105.
- **Effect:** 0.0010 absolute. It changes nothing in the reading, but this file's convention everywhere else is the unweighted per-seed mean, so the two numbers must not be used interchangeably without saying which is which.
- **What this file does:** T1 section 1c-bis prints the three per-seed values and the unweighted mean 0.5114; the pooled 0.5105 is named here so the prose can cite either knowingly.
- **Traced to:** `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/env_columns/fragment/FRAG-r2/validity`; `/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json` `/env_columns/fragment/FRAG-r2/validity`; `/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` `/env_columns/fragment/FRAG-r2/validity` …

### C12 (w12r) - the '9/9 seeds' phrasing

- **As written:** "1e-16...1e-14 on Shapes3D, 9/9 seeds, two independent implementations".
- **Artefact says:** the nine legs are 3 CONFIGURATIONS (ward_balanced K=40/44/48) x 3 SEEDS inside ONE implementation. There are THREE implementations, and their bands are not shared: 7.5e-16..8.8e-16 (w9/maze), 2.842e-14..1.126e-09 (w8 downdate-build), 5.551e-16..1.490e-13 (w8 single-design). And it was not unconditional: it FAILED first on TEST seed 2 at 32.90 float32 ulp.
- **Effect:** three defects in one sentence. The strongest form of the two-implementation claim is the FAILURE AGREEMENT - the second implementation reproduced the same seed-2 failure with different code at 14.377 ulp against 32.90.
- **What this file does:** TABLE 5 prints the three implementations as three rows with seed count and configuration count as SEPARATE fields, and makes the failure agreement a citable row. `/seed_and_configuration_counts` is a top-level field. The phrase is on the never-print list.
- **Traced to:** `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/a1_subspace/worst_in_float32_ulp`; `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/a1_subspace/worst_floor_subspace_in_float32_ulp`; `/c/chlu/outputs/w8-single-design/LOG.md` `LOG.md:472,479,481,495-503` …

### C13 (w12r/MF14)

- **As written:** the charter rule "every composition row carries BLIND-SEQUENTIAL measured on its own split", applied to all four rows.
- **Artefact says:** `/env_columns` carries exactly four keys - `unseen_pair`, `nway`, `fragment`, `fragment_cross_maze` - on every scored seed. There is no blind block for row 4 (budget interpolation) on any seed.
- **Effect:** the rule is unsatisfiable for row 4, and neither the tables nor the draft said so.
- **What this file does:** T1 section 1c-ter declares it NOT MEASURED - not a blank and not a zero - and the charter rule in `rules_applied` now carries the exception.
- **Traced to:** `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/env_columns`; `/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json` `/env_columns`; `/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` `/env_columns`

### C14 (w12r/MF6)

- **As written:** TABLE 4 carried no caption caveats.
- **Artefact says:** two leaf fields of the cited gate file: `cut_is_marginal` = true on ALL THREE seeds, and `a1_subspace.passed_every_seed` = false with `seeds_failing` = ["2"] at 60.36282 float32 ulp against its own 66.11072 ulp floor. The top-level `all_three_assertions_passed` = true EXCLUDES `a1_subspace`.
- **Effect:** a referee opening the paper's lead-result artefact reads `passed: false` inside it and the paper says nothing. Stated with the floor beside it, the failure is a property of the CONTENT and not of the downdate - which is what the artefact's own `reproducibility_floor.what` field says.
- **What this file does:** both are printed under T4's w8 row and both are in T4's caption.
- **Traced to:** `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/a1_subspace`; `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/cut_is_marginal_any_seed`; `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` `/reproducibility_floor/what`

### C15 (w13)

- **As written:** the wave-13 brief states that the crossover's scored row 1/3 was spent at 4 SEEDS, declared, and that the ladder ran 3 seeds.
- **Artefact says:** the ladder's 3 seeds are right. The scored crossover row is at ONE seed. `scored/` holds exactly one directory, `run1_seed0_K32`, its `/seed` is 0 and its `/run_n` is 1. The crossover's own FINDINGS section 9 is headed 'SCORED RUN 1/3 - rows 3 and 4 on TEST at K* = 32, seed 0'. What IS at 3 seeds in that hub is the VAL sweep at K = 5 and K = 32 (`/per_K/5/n_seeds` = 3, `/per_K/32/n_seeds` = 3) - and the val sweep labels itself `SMOKE_NOT_A_CLAIM: true`.
- **Effect:** a four-fold overstatement of the evidence behind the main scored row. T7 prints `seeds (n)` = 1 in a COLUMN and says so in its title.
- **What this file does:** T7's seed column and seed_count block; NR6 carries the two unspent runs.
- **Traced to:** `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` `/seed`; `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` `/run_n`; `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/per_K/32/n_seeds` …

### C16 (w13)

- **As written:** 'the honest total FALLS 7.48x -> 2.07x from K = 5 to K = 32, because the covariance shrinks faster than the address book grows.'
- **Artefact says:** 7.4815x -> 2.0663x is `total_bytes_parts_1_2_4_only` / `times_reference_pin_parts_1_2_4` - the total with the ADDRESS BOOK LEFT OUT. The HONEST four-part total, `honest_total_bytes`, runs 4,118,034 -> 2,143,138 B, i.e. 7.8545x -> 4.0877x, and it is U-SHAPED with a minimum of 3.9942x at K = 24. The address book GROWS 195,584 -> 1,059,824 B (5.4x) while the covariance falls 3,391,552 -> 560,608 B (6.0x), and past K = 24 the book grows FASTER than the covariance falls - the crossover hub's own FINDINGS section 5 says exactly that.
- **Effect:** the stated mechanism is inverted and the quoted curve is a three-part total, not the four-part one the referee asked for. A reader would conclude the object gets monotonically cheaper as K rises; it does not.
- **What this file does:** T12 prints BOTH totals as separate columns with the address-book column between them, and the caption names which curve each figure is quoting. fig7 panel B draws both; fig8 stacks the four parts.
- **Traced to:** `/c/chlu/outputs/w12-single-design-crossover/LEDGER-ACROSS-K.json` `/rows/0/times_reference_pin_parts_1_2_4`; `/c/chlu/outputs/w12-single-design-crossover/LEDGER-ACROSS-K.json` `/rows/8/times_reference_pin_parts_1_2_4`; `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/per_K/5/ledger_four_part/times_reference_pin` …

### C17 (w13)

- **As written:** the crossover FINDINGS section 7 prints the VAL deletion cost line as 1.45 / 0.32 / 0.72 / 0.28 s for our adaptation and 13.2 / 14.7 / 12.5 / 13.7 s for the refit-without reference.
- **Artefact says:** `deletion/val_seed0_K5/deletion.json` reads 2.4543 / 0.1874 / 0.6205 / 0.2554 s for our adaptation and 23.486 / 11.752 / 10.843 / 11.446 s for refit-without. Every other column in that section - rows deleted, bytes rewritten, data rows read, paired delta, store-scoped exactness - matches the artefact exactly.
- **Effect:** the prose understates our own adaptation time and understates the reference's, in the same direction, so the RATIO the sentence rests on is roughly preserved - but the printed seconds are not the measured seconds. T9b prints the artefact's.
- **What this file does:** T9b reads every cost figure straight from the artefact; the FINDINGS table is not used.
- **Traced to:** `/c/chlu/outputs/w12-single-design-crossover/deletion/val_seed0_K5/deletion.json` `/scopes/value/our_cost_line/seconds`; `/c/chlu/outputs/w12-single-design-crossover/deletion/val_seed0_K5/deletion.json` `/scopes/value/exactness_store_scoped/refit_without_cost_line/seconds`; `/c/chlu/outputs/w12-single-design-crossover/FINDINGS.md` `FINDINGS.md section 7`

### C18 (w13)

- **As written:** the crossover FINDINGS section 5 prints the val sweep at K = 5 as 0.8918 and at K = 32 as 0.9016, with 'n = 1 seed at every K'.
- **Artefact says:** the final `CROSSOVER-val.json` carries THREE seeds at both ends: `/per_K/5/val_composition_mean` = 0.895255 over seeds [0, 1, 2] and `/per_K/32/val_composition_mean` = 0.929012 over seeds [0, 1, 2]. The seven middle K still carry one seed each. The prose was written against an earlier state of the same file.
- **Effect:** the K* row is understated by 0.0274 and the drop from K = 5 is -0.0338, not -0.0098. More importantly the seed count differs BETWEEN ROWS OF ONE TABLE, which is exactly the hazard this spoke exists to print.
- **What this file does:** T12 prints `seeds (n) at this K` as a COLUMN - 3 at K = 5 and K = 32, 1 at the seven middle points - and fig7 annotates n at every point.
- **Traced to:** `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/per_K/5/n_seeds`; `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/per_K/8/n_seeds`; `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` `/per_K/32/val_composition_mean` …

### C19 (w13)

- **As written:** the ladder artefact writes its rung as `k` (`/rungs/{k}`) and the crossover writes its well count as `K` (`/per_K/{K}`). Read together, the two files put a 'k=5' and a 'K=5' on the same page meaning different things - and `LADDER.json` also carries `K_star: null`, written when the crossover's `K_star` had not yet been decided.
- **Artefact says:** `LADDER.json` `/K_of_the_store_arms` = 5 (WELLS) while `/rungs/5` is a rung of FIVE JOINTLY COMPOSED FACTORS at that same K = 5. `CROSSOVER-val.json` `/K_star` now reads 32. `LADDER.json` `/K_star_source` records the stale read in its own words: 'K_star=null, K=5 reference row absent; rule not applicable'.
- **Effect:** a collision the Hub has made once already. Anyone merging the two tables on 'k' produces a table in which one axis is two different quantities.
- **What this file does:** `TABLES.json` `/notation` defines both symbols; T8 relabels the rung as **n** and prints K = 5 as a COLUMN on every row; T7/T9/T9b/T12 print K and state that n is not varied. No figure mixes the two axes.
- **Traced to:** `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/K_of_the_store_arms`; `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/K_star`; `/c/chlu/outputs/w12-composition-ladder/LADDER.json` `/K_star_source` …

---

## FIGURES — all regenerated from the manifest by `python build_tables.py`

| figure | file | what it shows | sources |
|---|---|---|---|
| fig1 | `figures/fig1_cmaze2_main_scoreboard.png` | T1 RESULT rows 1-4: arm, learned baselines and the matched-bytes floor, 3 seeds as dots, pipeline ceiling per row | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` |
| fig2 | `figures/fig2_nway_conjunction_vs_route.png` | Row 3 under both readings against the per-route ceiling | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json` |
| fig3 | `figures/fig3_encoder_loop_decomposition.png` | T2: A' / A'' / A per seed on rows 3 and 4 | `/c/chlu/outputs/w8-encoder-loop/CROSS-ARM.json` |
| fig4 | `figures/fig4_exact_deletion.png` | T4: both deletion implementations, log scale | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json`<br>`/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` |
| fig5 | `figures/fig5_blind_sequential_on_this_split.png` | T1 caveat column: BLIND-SEQUENTIAL on this split vs the published 0.4980 | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json`<br>`/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json`<br>`/c/chlu/outputs/w8-env-scout/screen_results_DEFAULTS.txt` |
| fig6 | `figures/fig6_composition_ladder_n.png` | T8: the n-way ladder (n = 2..6) at K = 5 wells, 3 seeds; panel B carries BOTH byte matchings and is where the sign flips | `/c/chlu/outputs/w12-composition-ladder/LADDER.json` |
| fig7 | `figures/fig7_k_curve_wells.png` | T12: val composition and BOTH byte totals against K = 5..32 wells, with n_seeds printed at every point and rank/width beside them | `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json`<br>`/c/chlu/outputs/w12-single-design-crossover/LEDGER-ACROSS-K.json` |
| fig8 | `figures/fig8_four_part_ledger_across_K.png` | T12: the FOUR-PART byte ledger stacked across K, never netted, against the 524,288 B reference | `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` |

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
| downdate_log | `/c/chlu/outputs/w8-downdate/LOG.md` | 81,305 | d5ae978a87bd0357 |
| single_design_log | `/c/chlu/outputs/w8-single-design/LOG.md` | 126,149 | 83740a7ae62ef691 |
| w5_read_s0 | `/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json` | 59,053 | 795fca8cfa78c9d3 |
| w5_read_s1 | `/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json` | 59,083 | 435361b0c25c8ed2 |
| w5_read_s2 | `/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json` | 58,726 | a6946a7536eea3be |
| w5_log | `/c/chlu/outputs/w5-compose-field/LOG.md` | 64,498 | e64d858814d5fc90 |
| gentable_spec | `/c/chlu/outputs/hub-generation-table/relaunch/scored/scored_spec.json` | 745 | 7990c75559ff9393 |
| gentable_log | `/c/chlu/outputs/hub-generation-table/LOG-iter2.md` | 100,346 | e68f74abec35b63b |
| gentable_runarm | `/c/chlu/outputs/hub-generation-table/relaunch/scored/run_arm.py` | 22,263 | 065a0ecddaa55511 |
| envscout_env | `/c/chlu/outputs/w8-env-scout/ENV.md` | 30,128 | 400af82465c7b375 |
| envscout_defaults | `/c/chlu/outputs/w8-env-scout/screen_results_DEFAULTS.txt` | 4,521 | 600583bbcb2e61e9 |
| wave_plan | `/c/chlu/evidence/outputs/advisor-wave-plan-w8-w10.md` | 6,408 | c6c08a4237f985d3 |
| w8_advisor | `/c/chlu/evidence/outputs/w8-advisor-report.md` | 8,698 | 166df363cdeb5b0a |
| x_cross_val | `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` | 71,664 | 7eb12295f989fe76 |
| x_ledger_K | `/c/chlu/outputs/w12-single-design-crossover/LEDGER-ACROSS-K.json` | 5,521 | 5b6fc67d225c4c0a |
| x_scored1 | `/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json` | 740,898 | 07f71bd6f04cb256 |
| x_del_test | `/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json` | 226,176 | 90197a10ce4ecd1b |
| x_del_val | `/c/chlu/outputs/w12-single-design-crossover/deletion/val_seed0_K5/deletion.json` | 77,082 | 9509ecbf8ed7ba1e |
| x_probe | `/c/chlu/outputs/w12-single-design-crossover/deletion/probe_test_seed0_K32.json` | 1,991 | 8bf7ce1aa4ac8477 |
| x_findings | `/c/chlu/outputs/w12-single-design-crossover/FINDINGS.md` | 32,206 | f5f51ca62de0361b |
| x_selection_rule | `/c/chlu/outputs/w12-single-design-crossover/SELECTION-RULE.md` | 3,581 | fcb200622436acc6 |
| lad_json | `/c/chlu/outputs/w12-composition-ladder/LADDER.json` | 833,221 | f6db3aad719c939b |
| lad_bytes | `/c/chlu/outputs/w12-composition-ladder/BYTES.json` | 64,414 | 51da86c60ea552fb |
| lad_findings | `/c/chlu/outputs/w12-composition-ladder/FINDINGS.md` | 21,205 | 3507de39260bb123 |
| lad_ruling | `/c/chlu/outputs/w12-composition-ladder/HEAD-RULING-K.md` | 3,421 | 5a9835756cf1685d |
| harness_ready | `/c/chlu/outputs/w12-kway-ladder-harness/LADDER-READY.json` | 22,591 | 4255bc63c9ecdfdb |
| base_ready | `/c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json` | 379,454 | c35e475f49d5f1e9 |
| cache_ready | `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json` | 25,595 | 203f97be67aa8560 |
| cache_verify | `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/VERIFY.json` | 1,740 | 59c6f3203f2e11fc |
| cache_gate | `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/reencode_gate_seeds_3_4.json` | 2,006 | ccf07bca6c56e31c |
| w5_delete_report | `/c/chlu/outputs/w5-delete/run1/report.json` | 227,128 | e0b6fb56785177d8 |
| rivals_ready | `/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json` | 8,120 | a38efb81cd86c1b1 |
| wave_plan_12 | `/c/chlu/evidence/outputs/advisor-wave-plan-w12-w14.md` | 11,875 | 9e28ce07b34a35b1 |

⛔ **Nothing outside this register was read**, and in particular `$CHLU_CLAUDE_ROOT` was not read at any point — it is a frozen older snapshot.

