# w15 spoke/lace-poe-rows — FINDINGS

Branch `w15/lace-poe-rows` off `base/iter8 = 148f3b7` · box FORGIS_14, CPU only ·
`chlu.__file__ = C:\chlu\worktrees\CHLU-w15-lace-poe-rows\chlu\__init__.py` (asserted before every number).
Acceptance artefact: **`GENTABLE-KSTAR.json`** (+ human-readable `GENTABLE-KSTAR.md`), built by
`scripts/w15_lace_poe_table.py` from the per-seed / per-leg artefacts it names.
⛔ Every number below carries the artefact it came from. Seed count is a column wherever a table appears.

---

## 0. The three things to read first

1. **The K\* single design is in the generation-table registry and it scored.** `generation-table` run
   **3/3** (the last one) was spent on it — ledger row written first at **2026-09-23T09:14:17Z**
   (`/c/chlu/SCORED-LEDGER.md`), PREREG + AMENDMENT 1 before any number
   (`PREREG-w15-lace-poe-rows-1.md`). **3 seeds, nothing NOT-RUN.** The K\* arm **beats the shipped
   CLU read on the seed mean (0.7500 vs 0.5949) but NOT on every seed** — seed 0 is −0.0185 — so the
   pre-declared bar S2 **FAILS** and the PREREG's own falsification clause fires. It **loses to
   `knn_proto`** on the per-factor aggregate (−0.1150, S3 LOSE). All four charter-C controls collapse on
   every seed (S1 PASS): **the store claim stands, the "beats the shipped read" claim does not.**
2. **LACE and PoE do NOT beat the CLU on Shapes rows 3–4 — nor anywhere on the ladder (§3).** On rows 3–4
   the K\* store scores 47.50 / 47.40 / 49.12 novel valid tuples per corner; LACE 0.165 / 0.073 / 0.229;
   PoE 0.009 / 0.000 / 0.000; the store wins **109/109 corners against each, on every seed.** This is not a
   headline negative. ⚠ It is also not flattering evidence: both rivals render **zero exact-match**
   requests on the shipped cell too (§1c) — they fail this metric's "all six factors valid" requirement
   in this feature space at their own budgets, and a reader should weigh the result as "LACE/PoE, as
   re-implemented at these budgets, do not produce valid full tuples here", not as "the CLU out-composes
   LACE in general".
3. ⛔ **LACE and PoE were ADDED; nothing was replaced.** MLP / LSTM / transformer / continuous MLP (and
   `learned_1x` on the rungs) and the table floor are printed beside them in every block of
   `GENTABLE-KSTAR.md`, read from their original artefacts.

---

## 1. PART A — the K\* single design on the generation table (scored run 3/3)

### 1a. What was wired
* **Registry entry:** `kstar` in `chlu/experiments/exp_gen_cell.py` — two **add-only** `elif` branches
  (resume check + dispatch); `git diff 148f3b7 -- chlu/experiments/exp_gen_cell.py` is **+46 / −0**.
  ⛔ Not appended to `ARMS` (the pin's rival list), exactly as `x0` / `dynread` are kept off it.
  Construction lives in the new module `chlu/experiments/exp_gen_cell_kstar.py`, which imports the
  crossover hub's own library (`scripts/w12_crossover_lib.py`, `scripts/w8_matched_bytes_recut.py`) —
  one implementation, not two.
* **The object:** `CovContentStore`, label-free `ward_balanced` cut at **K\* = 32** (not re-selected),
  A = 63, wells 64 wide, rank/width max **0.9844**, TensorAddress at `READ_PINS` (rank 128, ridge 1e-2,
  40 sweeps), fitted on the cell's own 32,768-row store pool. Source: `scored/cell/seed_*/cell_manifest.json:kstar_val.info`.
* **Four-part byte ledger (crossover convention, summed, never netted), per seed**
  (`scored/cell/seed_*/cell_manifest.json:kstar_val.ledger`): atoms 516,096 | covariance+count 560,608 |
  address book 1,062,424 / 1,069,704 / 1,072,304 | coupling/hole 6,610 ⇒ **2,145,738 / 2,153,018 /
  2,155,618 B = 4.09–4.11× the 524,288 B reference column.** ⛔ No K\* row is matched-bytes.

### 1b. Val first (free), then scored
VAL bars (PREREG §1a), cell's 512 stratified val rows, gen-2 (`dev/kstar_val_gen2/seed_*/cell_manifest.json:kstar_val.gate`):

| seed | K\* val aggregate | removed | shuffled | across wells | random atoms | VB1 ≥ 0.50 | VB2 ≥ 0.10 below |
|---|---|---|---|---|---|---|---|
| 0 | 0.9212 | 0.1234 | 0.1549 | 0.1217 | 0.1084 | PASS | PASS |
| 1 | 0.9720 | 0.1234 | 0.1074 | 0.1042 | 0.1172 | PASS | PASS |
| 2 | 0.9186 | 0.1230 | 0.1237 | 0.1117 | 0.1217 | PASS | PASS |

### 1c. The scored table — `scored/cell/cell_results.json` (per-seed: `scored/cell/seed_*/cell_manifest.json`)
External classifier (hash-asserted), composition-B aggregate over 72 held-out requests; **n seeds = 3** on every row.

| arm | n | comp-B aggregate s0 / s1 / s2 | mean ± sd | exact-match s0/s1/s2 | in-corner balacc mean | distinct tuples on B (s0/s1/s2) | deployed B (convention) |
|---|---|---|---|---|---|---|---|
| **`kstar`** | 3 | 0.5718 / 0.9977 / 0.6806 | **0.7500 ± 0.1807** | 0.500 / 0.986 / 0.500 | 0.8174 | 47 / 71 / 48 | 2.15 MB four-part (4.09–4.11× ref) |
| `kstar/landscape_removed` | 3 | 0.0556 / 0.0556 / 0.2778 | 0.1296 | — | 0.5000 | 1 / 1 / 1 | same object, atoms zeroed |
| `kstar/landscape_shuffled` | 3 | 0.1481 / 0.1389 / 0.0602 | 0.1157 | — | 0.4941 | 29 / 7 / 25 | same bytes |
| `kstar/landscape_shuffled_across_wells` | 3 | 0.1065 / 0.0764 / 0.1435 | 0.1088 | — | 0.5417 | 14 / 19 / 7 | same bytes |
| `kstar/random_atoms` | 3 | 0.1458 / 0.2523 / 0.1597 | 0.1860 | — | 0.4657 | 18 / 31 / 25 | same bytes |
| `clu` (shipped read) | 3 | 0.5903 / 0.6065 / 0.5880 | 0.5949 ± 0.0082 | 0 / 0 / 0 | 0.6568 | 9 / 8 / 10 | 524,288 (store modes) |
| `store_removed` | 3 | 0.1111 / 0.1389 / 0.1111 | 0.1204 | — | 0.5000 | 1 / 1 / 1 | 524,288 |
| `knn_proto` (ceiling) | 3 | 0.8588 / 0.8542 / 0.8819 | 0.8650 ± 0.0122 | 0.167 / 0.139 / 0.292 | 0.9165 | 57 / 51 / 60 | 475,136 (retired-budget selection, declared) |
| `lace` (ode) | 3 | 0.2940 / 0.3241 / 0.2917 | 0.3032 ± 0.0148 | 0 / 0 / 0 | 0.6280 | 72 / 71 / 72 | not byte-matched (R9.3) |
| `ebm_compose` (PoE) | 3 | 0.1759 / 0.1875 / 0.2269 | 0.1968 ± 0.0218 | 0 / 0 / 0 | 0.5500 | 72 / 72 / 72 | 14,226,660 (27.1× ref) |

Exact-match figures: `scored/cell/seed_*/cell_manifest.json:arms.<arm>.external.composition_B.exact_match`.
`lace/ld_corrected` and `lace/ld_published` are also on disk (0.2832 / 0.2577 means) and in `GENTABLE-KSTAR.md`.
✔ Reproduction: `clu`, `store_removed`, `knn_proto` reproduce the frozen Mac cell's in-corner numbers on
every seed (e.g. seed 0: 0.6540 / 0.5000 / 0.9198 — `gen-read-cell-arm/runs/cell/seed_0/seed_0/cell_manifest.json`).

### 1d. The pre-declared bars (PREREG §1b) — `GENTABLE-KSTAR.json:PART_A_generation_table.bars`

| bar | per-seed Δ (K\* − other) | verdict |
|---|---|---|
| **S1** four controls collapse (Δ ≥ 0.10 every seed) | min over controls: +0.4236 / +0.7454 / +0.4028 | **PASS** — the store is load-bearing on this table |
| **S2** vs `clu`, ≥ +0.020 on EVERY seed | −0.0185 / +0.3912 / +0.0926 (mean +0.1551) | **FAIL** (seed 0) — ⛔ PREREG falsification clause fires |
| **S3** vs `knn_proto`, mean, margin 0.020 | −0.2870 / +0.1435 / −0.2014 (mean −0.1150) | **LOSE** |
| **S4** vs `lace`, mean | +0.2778 / +0.6736 / +0.3889 (mean +0.4468) | WIN |
| **S4** vs `ebm_compose`, mean | +0.3958 / +0.8102 / +0.4537 (mean +0.5532) | WIN |

**Reading.** The K\* arm is the first store read on this table whose controls all collapse *and* which
clears the shipped read on the seed mean and on in-corner balanced accuracy (0.8174 vs 0.6568), with
~5× the distinct tuples (55.3 vs 9.0 mean). ⛔ But the pre-declared every-seed bar against `clu` fails on
seed 0, the seed spread (sd 0.18) is 22× `clu`'s, and it loses to the prototype ceiling on the
per-factor aggregate at 4.1× the reference bytes. ⚠ Not pre-declared, printed as secondary: on
**exact match** (all six factors right) K\* beats `knn_proto` on every seed (0.50 / 0.99 / 0.50 vs
0.17 / 0.14 / 0.29) — `knn_proto` gets `scale` wrong (0.17 / 0.14 / 0.31) while K\* gets whole requests
right-or-wrong in halves. That is an observation, not a bar, and it is not claimed.

---

## 2. PART B — rows 3–4: LACE and PoE added beside the learned rivals

`GENTABLE-KSTAR.json:PART_B_rows_3_4`. Metric `distinct_valid_novel_tuples` per corner (60 pair + 49 n-way
corners, unchanged `fw.per_corner_scores`), mean over the 109 corners; **n seeds = 3** on every row.

| arm | n | s0 | s1 | s2 | row 3 (pair) s0/s1/s2 | row 4 (n-way) s0/s1/s2 | source |
|---|---|---|---|---|---|---|---|
| **K\* store** | 3 | 47.495 | 47.404 | 49.119 | 31.600 / 32.233 / 33.217 | 66.959 / 65.980 / 68.592 | w12-single-design-crossover `scored/run1_seed0_K32`, w12b `scored/run3_seed{1,2}_K32` |
| K\* controls ×4 | 3 | 0.000 | 0.000 | 0.000 | 0 | 0 | same files |
| MLP @ 524,288 B ref | 3 | 5.193 | 3.239 | 8.174 | 2.933 / 3.433 / 5.267 | 7.959 / 3.000 / 11.735 | same files `mlp__as_trained` |
| LSTM @ ref | 3 | 4.761 | 4.633 | 9.404 | 2.500 / 2.700 / 4.917 | 7.531 / 7.000 / 14.898 | `lstm__as_trained` |
| transformer @ ref | 3 | 1.229 | 0.651 | 3.073 | 0.667 / 0.667 / 1.717 | 1.918 / 0.633 / 4.735 | `transformer__as_trained` |
| continuous MLP @ ref | 3 | 0.028 | 0.028 | 0.055 | 0.000 / 0.000 / 0.017 | 0.061 / 0.061 / 0.102 | `cont_mlp__as_trained` |
| table — FLOOR, never a rival | 3 | 8.229 | 5.073 | 9.752 | 5.833 / 3.483 / 6.183 | 11.163 / 7.020 / 14.122 | `table` |
| **LACE** (own budget) | 3 | 0.165 | 0.073 | 0.229 | 0.117 / 0.033 / 0.267 | 0.224 / 0.122 / 0.184 | `ladder/shipped_seed{0,1,2}/rows_lace.json` |
| **PoE** (own budget) | 3 | 0.009 | 0.000 | 0.000 | 0 / 0 / 0 | 0.020 / 0 / 0 | `ladder/shipped_seed{0,1,2}/rows_poe.json` |

**Their own budgets (printed in every row of `GENTABLE-KSTAR.md`; per leg in `rows_*.json:summary.budget`):**
* **LACE:** 4,073,663 classifier parameters = **31.08× the 524,288 B reference** (63.10× counting the
  2048×2048 whitening + mean); 32,256 fit / 512 val rows from the rung's own pool; 36-trial selection;
  50 epochs, 12,600 steps; primary `ode` sampler.
* **PoE:** 6 independent experts, 3,556,665 parameters = **27.13× the reference**; same rows; 20 epochs;
  36-trial SGLD grid.

**Verdict (PREREG §2a):** paired Δ (rival − K\*) per seed — LACE **−47.33 / −47.33 / −48.89**
(bootstrap CI95 e.g. seed 0 [−50.85, −43.92]), PoE **−47.49 / −47.40 / −49.12**; K\* wins
**109/109 corners, 0 ties, on every seed, against both.** ⇒ **CLU ahead.** (The two identical LACE Δs at
seeds 0 and 1 are arithmetic coincidence: 0.1651 − 47.4954 = 0.0734 − 47.4037.)

---

## 3. PART B — every ladder rung

`GENTABLE-KSTAR.json:PART_B_ladder.k=<k>` (full rows incl. every control and the table floor in `GENTABLE-KSTAR.md`). Metric `distinct_valid_novel_tuples` per corner; k=2: 60 corners × 36 rows; k=3–6: 120 corners × 72 rows. **n seeds = 3 on every cell.** Cells are seed 0 / 1 / 2 (mean).

| rung | K\* store | learned_1x @ K\* total | MLP @ ref | LSTM @ ref | transformer @ ref | cont. MLP @ ref | **LACE** | **PoE** | K\* wins (LACE / PoE) every seed |
|---|---|---|---|---|---|---|---|---|---|
| k=2 | 31.800 / 32.500 / 33.633 (32.644) | 25.133 / 29.383 / 30.350 (28.289) | 17.017 / 22.000 / 23.333 (20.783) | 19.417 / 19.333 / 23.750 (20.833) | 12.750 / 15.783 / 24.250 (17.594) | 3.883 / 4.567 / 6.533 (4.994) | 0.067 / 0.050 / 0.067 (0.061) | 0.017 / 0.000 / 0.000 (0.006) | 60/60/60 ; 60/60/60 of 60 |
| k=3 | 60.150 / 62.783 / 61.908 (61.614) | 42.275 / 49.975 / 50.375 (47.542) | 35.208 / 41.567 / 49.750 (42.175) | 44.433 / 39.400 / 46.675 (43.503) | 28.800 / 33.900 / 51.992 (38.231) | 6.333 / 9.117 / 12.667 (9.372) | 0.133 / 0.125 / 0.208 (0.156) | 0.000 / 0.000 / 0.000 (0.000) | 120/120/120 ; 120/120/120 of 120 |
| k=4 | 65.242 / 65.817 / 67.200 (66.086) | 52.542 / 57.075 / 58.575 (56.064) | 35.208 / 41.567 / 49.750 (42.175) | 44.433 / 39.400 / 46.675 (43.503) | 28.800 / 33.900 / 51.992 (38.231) | 6.333 / 9.117 / 12.667 (9.372) | 0.117 / 0.217 / 0.183 (0.172) | 0.008 / 0.000 / 0.008 (0.006) | 120/120/120 ; 120/120/120 of 120 |
| k=5 | 65.642 / 66.817 / 67.283 (66.581) | 55.292 / 60.300 / 59.083 (58.225) | 35.208 / 41.567 / 49.750 (42.175) | 44.433 / 39.400 / 46.675 (43.503) | 28.800 / 33.900 / 51.992 (38.231) | 6.333 / 9.117 / 12.667 (9.372) | 0.083 / 0.142 / 0.183 (0.136) | 0.000 / 0.008 / 0.000 (0.003) | 120/120/120 ; 120/120/120 of 120 |
| k=6 | 65.908 / 66.250 / 67.650 (66.603) | 56.533 / 60.142 / 59.200 (58.625) | 35.208 / 41.567 / 49.750 (42.175) | 44.433 / 39.400 / 46.675 (43.503) | 28.800 / 33.900 / 51.992 (38.231) | 6.333 / 9.117 / 12.667 (9.372) | 0.125 / 0.100 / 0.175 (0.133) | 0.000 / 0.008 / 0.000 (0.003) | 120/120/120 ; 120/120/120 of 120 |

**Paired Δ (rival − K\* store), per seed (PREREG §2a):**

| rung | LACE Δ s0 / s1 / s2 | PoE Δ s0 / s1 / s2 | verdict |
|---|---|---|---|
| k=2 | -31.733 / -32.450 / -33.567 | -31.783 / -32.500 / -33.633 | LACE: CLU ahead; PoE: CLU ahead |
| k=3 | -60.017 / -62.658 / -61.700 | -60.150 / -62.783 / -61.908 | LACE: CLU ahead; PoE: CLU ahead |
| k=4 | -65.125 / -65.600 / -67.017 | -65.233 / -65.817 / -67.192 | LACE: CLU ahead; PoE: CLU ahead |
| k=5 | -65.558 / -66.675 / -67.100 | -65.642 / -66.808 / -67.283 | LACE: CLU ahead; PoE: CLU ahead |
| k=6 | -65.783 / -66.150 / -67.475 | -65.908 / -66.242 / -67.650 | LACE: CLU ahead; PoE: CLU ahead |

K\* controls on the rungs (`landscape_removed`, `landscape_shuffled`, `random_atoms`, from w12b) read 0.000–0.008 on every rung and seed (`w12b-nr6-and-kstar/kstar_ladder/<k>_seed<s>_K32/rows_single_design_K32__*.json`).

**Own budgets on every rung** are identical in form to rows 3–4 (per leg: `ladder/<k>_seed<s>/rows_{lace,poe}.json:summary.budget`): LACE 4,073,663 classifier params (31.08× ref; 63.10× with whitening), PoE 3,556,665 params (27.13× ref), 32,256 fit / 512 val rows from the rung's own pool, 36 selection trials each.

⚠ **Axis caveats carried from the instrument, not introduced here.** (i) Rungs k = 3–6 probe the IDENTICAL 120 corners (w12b `KSTAR-LADDER.md`), so for k ≥ 3 the rung axis is a training-data axis; only k=2 vs k=6 is a composition-depth contrast. (ii) The `@ ref` learned columns are spoke A's rung-independent rivals (fitted once per seed on the Montero pool), which is why their k=3–6 cells are identical; `learned_1x` is fitted per rung. (iii) Ceilings differ: 36 at k=2, 72 at k ≥ 3.

---

## 4. What did NOT run — declared, never a zero

* **`dynread` / `x0` were not re-run** on the generation table in scored run 3/3 (not this dial; their
  Mac rows stand in `gen-read-cell-arm/runs/cell_results.json`).
* **The 1×-matched learned rival on rows 3–4 is not produced here**; on the rungs `learned_1x` (1× the K\*
  four-part total) is read from `w12b-nr6-and-kstar/kstar_ladder/*/rows_learned_1x.json`.
* **LACE's `ld_published` / `ld_corrected` samplers were not run on rows 3–4 / the rungs** (primary `ode`
  only; both are on the generation table).
* **PoE ran once per (rung, seed)**, not twice; the three seeds carry the spread.
* **The ladder's `landscape_shuffled_across_wells` control** exists only on rows 3–4; on the rungs the K\*
  controls are the three w12b ran (`landscape_removed`, `landscape_shuffled`, `random_atoms`).
* **Generation-1 cell seeds 1 and 2**: STOPPED by the no-extension launcher (shortfall 32,395 rows;
  `logs/dev_kstar_val_s12.log`) — nothing encoded; the scored run used generation 2 (AMENDMENT 1).

## 5. Deviations and decisions (all in `LOG.md`)

* D2 — the enforced re-encode gate aborts on this box (known: `hub-generation-table/REENCODE-VERDICT-iter2.md`);
  the launcher applies the no-extension ruling (measured `n_shortfall = 0` per seed, else STOP); shared code untouched.
* D6 / AMENDMENT 1 — generation-2 cache for the cell (byte-identical encoders, full scope); written before
  the scored run.
* LACE/PoE selection reads val renders through the scoring judge (the never-weaken direction), declared in the PREREG.
* The PREREG §2a beat rule (mean Δ ≥ +1.0 on every seed) is weaker than `fw.paired_estimand`'s
  `beats_beyond_margin` (CI lower bound > 1.0); both are in the JSON.

## 6. What would have proved it wrong

* **Part A:** any K\* control failing to sit 0.10 below the arm (S1) — did not happen; K\* ≤ `clu` + 0.02 on
  any seed (S2) — **happened on seed 0**, and is reported as the falsification it was declared to be.
* **Part B:** LACE or PoE with mean paired Δ ≥ +1.0 novel tuple/corner over the K\* store on every seed at
  any rung, rows 3–4 above all — see §2–§3; did not happen.

## 7. Artefacts (verified by directory listing)

* `GENTABLE-KSTAR.json`, `GENTABLE-KSTAR.md` — the acceptance artefact.
* `PREREG-w15-lace-poe-rows-1.md` (+ `.timestamp.txt`, AMENDMENT 1) · ledger row `/c/chlu/SCORED-LEDGER.md` @ 2026-09-23T09:14:17Z.
* `scored/cell/{cell_results.json, cell_table.png, cache_coverage.json, seed_{0,1,2}/cell_manifest.json}` — scored run 3/3.
* `dev/kstar_val_gen2/seed_{0,1,2}/cell_manifest.json` — the val gates (DEV); `dev/cell_smoke/` — the gen-1 reproduction smoke (DEV).
* `ladder/<rung>_seed<s>/{rows_lace.json, rows_poe.json, LEG.json}` for rung ∈ {shipped, 2, 3, 4, 5, 6}, s ∈ {0, 1, 2} — 18 legs, 36 row files.
* `logs/` — every run's stdout; `LOG.md`; `PIDS.txt`.
* `w15-lace-poe-rows.bundle` — branch `w15/lace-poe-rows` @ `70885c0`, `git bundle verify` OK.
