# PREREG — w15 spoke/lace-poe-rows — `generation-table` SCORED RUN 3/3

**Written 2026-09-23, BEFORE any K\* generation-table number, any K\* val-gate number, and any
LACE / PoE number on a test corner existed.** Timestamp: `PREREG-w15-lace-poe-rows-1.timestamp.txt`.
Amendments are appended below with a UTC stamp; nothing above an amendment is edited.

Box FORGIS_14, CPU only. Branch `w15/lace-poe-rows` off `base/iter8 = 148f3b7`.
Target: **`generation-table`, run 3 of 3 — the LAST.** Runs 1/3 (wave 2, eval_b seed 0) and 2/3
(hub/store-content) are spent (`/c/chlu/SCORED-LEDGER.md`).

## 0. What is already on disk before this was written (DEV, not results)
* `dev/cell_smoke/seed_0/cell_manifest.json`: `knn_proto` on the generation-table cell, seed 0, through
  this branch's launcher → composition-B aggregate 0.8588, in-corner balacc 0.9198, 57 tuples — the
  frozen Mac cell's published row to the last digit. A reproduction check of the instrument; no K\*,
  LACE or PoE number was produced by it.

## 1. PART A — the scored run: the K\* single design on the generation table

**Instrument (unchanged):** `chlu.experiments.exp_gen_cell` — `request → read → codes → THE frozen
decoder (v1_capacity_schedule, hash-asserted) → image → THE hash-asserted external classifier`, on the
`recomb2element` cell: B = 72 held-out requests, MS = its matched-seen negatives, store pool 32,768
rows, stratified fit 32,256 / val 512. Seeds **0, 1, 2 — all three, or none.** Launcher
`scripts/w15_lace_poe_gencell.py` (no-extension ruling: measured `n_shortfall == 0` per seed, else STOP).

**The arm `kstar` (registered in the `exp_gen_cell` registry; module `exp_gen_cell_kstar`):**
`CovContentStore` on the label-free `ward_balanced` cut at **K\* = 32 (NOT re-selected;
`w12-single-design-crossover/CROSSOVER-val.json`)**, A = min(64, min d_k − 1), `pinned_basis`, a
`TensorAddress` at `exp_gen_read.READ_PINS` (rank 128, ridge 1e-2, 40 sweeps), fitted on the cell's own
store pool. Byte convention: the crossover hub's **FOUR-PART** ledger (atoms | covariance+count |
address book | coupling/hole table), summed, never netted; the table's 524,288 B is a REFERENCE column
and the multiple is printed on every K\* row. ⛔ No K\* row is called matched-bytes.

**Its four charter-C controls, on every seed:** `kstar/landscape_removed` (atoms zeroed),
`kstar/landscape_shuffled` (atom values permuted within each well), `kstar/landscape_shuffled_across_wells`
(blocks deployed into other wells), `kstar/random_atoms` (Gaussian block, same Frobenius norm) — all
applied AFTER the address book is fitted, read through `qr_basis` at the book's own widths.

**Every other arm on the table is run in the same invocation, same seeds:** `clu`, `store_removed`,
`knn_proto`, `lace`, `ebm_compose` (the pinned `ARMS`, untouched). `dynread` / `x0` are not re-run
(DECLARED NOT-RUN: not part of this dial; their Mac rows stand).

### 1a. VAL BARS (val-first; evaluated IN THE SAME PROCESS BEFORE any B/MS number is recorded)
On the cell's 512 stratified val rows, external-classifier aggregate (mean per-factor accuracy):
* **VB1:** `kstar` val aggregate **≥ 0.50**.
* **VB2:** each of the four controls sits **≥ 0.10 below** `kstar` on val.
A seed failing VB1 or VB2 records `kstar` as NOT RUN with the gate on disk; ⛔ no B number is produced
for it, and the run is reported as a val failure. A val-only DEV pass (`CHLU_KSTAR_VAL_ONLY=1`) is
run before the scored run and its numbers are recorded in `LOG.md`.

### 1b. SCORED BARS (the verdict vocabulary, declared now)
Primary statistic: **external composition-B aggregate**, per seed, 3 seeds; distinct factor-tuples on B
printed beside every accuracy.
* **S1 — store claim gate:** all four K\* controls **collapse** on every seed: aggregate ≤ `kstar` − 0.10.
  If any does not, NO store claim — the row is an interface result or nothing, and we say which.
* **S2 — vs the shipped CLU read (`clu`, 524,288 B):** `kstar` − `clu` ≥ **+0.020** on every seed ⇒
  "K\* beats the shipped read"; |Δ| < 0.020 ⇒ tie; otherwise loses. (Byte multiple printed.)
* **S3 — vs the ceiling `knn_proto`:** the table's own `DYNREAD_MARGIN = 0.020`: mean Δ ≥ +0.020 win;
  |Δ| ≤ 0.020 tie; else lose.
* **S4 — vs LACE and PoE (`lace`, `ebm_compose`) on this cell:** reported with the same 0.020 margin.
**Expected direction (stated so a miss is visible):** `kstar` > `clu` (the shipped 30-bit read renders 9±1
distinct vectors of 72); `kstar` < `knn_proto` is plausible and would be printed as a loss.
**Falsified if:** `kstar` ≤ `clu` + 0.020 on any seed (S2 fails), or S1 fails.

## 2. PART B — LACE and PoE as ADDED rows: Shapes rows 3–4 and every ladder rung

Rows 3–4 = the `shipped` rung (60 pair sub-corners + 49 n-way corners); ladder rungs k = 2, 3, 4, 5, 6
(`w12/kway-ladder-harness`). **3 seeds (0, 1, 2).** Metric `distinct_valid_novel_tuples` per corner,
`fw.per_corner_scores`, unchanged; judge `fw.Judge`/`FastJudge` (identity asserted per leg). Driver
`scripts/w15_lace_poe_ladder.py`.

**Their own budgets, declared (printed in every row):** 32,768 rows = the first 32,768 of the rung's
own ladder pool (= `learned_1x`'s `N_FIT_ROWS`), stratified 32,256 fit / 512 val. LACE: 36-trial
selection (σ × form × shrinkage; `GenBaselinesConfig` schedule) → 50-epoch classifier → primary `ode`
sampler; bytes printed as classifier-only AND classifier + whitening. PoE: 6 independent experts
(hidden 256, depth 2, 20 epochs), 36-trial SGLD grid; bytes = expert params × 4. Selection = val
aggregate through the scoring judge (the never-weaken direction). ⛔ No corner row enters selection.

⛔⛔ The learned-rival columns stay: MLP / LSTM / transformer / continuous MLP @ 524,288 B ref (and
`learned_1x` where the ladder has it) are READ from `w12-composition-ladder/runs/*` and
`w12b-nr6-and-kstar/{kstar_ladder,scored}`, printed beside LACE and PoE. The table floor is printed as a
floor, never a rival.

**The CLU on these rows** = the K\* single design (the main text's object): `w12-single-design-crossover/
scored/run1_seed0_K32` + `w12b-nr6-and-kstar/scored/run3_seed{1,2}_K32` for rows 3–4 and
`w12b-nr6-and-kstar/kstar_ladder/<k>_seed<s>_K32` for the rungs.

### 2a. What would prove the CLU wrong — the HEADLINE NEGATIVE rule, declared now
For each (rung, rival ∈ {LACE, PoE}) and each seed: paired per-corner Δ = rival − K\* store on the SAME
corners. **The rival BEATS the CLU at a rung iff mean Δ ≥ +1.0 novel tuple/corner
(`DECORATION_MARGIN_NOVELTY`) on EVERY one of the 3 seeds.** Mean Δ > 0 on the seed mean but short of
that ⇒ "rival ahead, within margin" — printed as such. ⛔ If LACE or PoE beats the CLU on rows 3–4 it is a
headline negative, it goes in the main text, and neither rival is re-tuned down.

## 3. What does NOT run (declared now)
* The 1×-matched rival column is not produced here (it exists in `w12b` as `learned_1x` on the rungs).
* LACE's `ld_published` / `ld_corrected` samplers are not run on the ladder (primary `ode` only).
* PoE is run once per (rung, seed), not twice; the 3 seeds carry the spread.
* `dynread` / `x0` are not re-run on the generation table.

---
## AMENDMENT 1 — 2026-09-23T07:36:03Z — the cache generation (before any scored number)
Measured: the generation-1 bridge caches for seeds 1 and 2 are scope `used` (9,288 rows); the cell needs
33,039 rows per seed, so a seed-1/2 cell on generation 1 would EXTEND the cache, which the no-extension
ruling forbids (launcher STOPPED, `logs/dev_kstar_val_s12.log`). The generation-2 tree
`C:/chlu/outputs/spoke-a-rivals/cache_generation2` holds, per seed, the byte-identical `encoder.pkl`
(same sha256, same param hash) at scope `full` (480,000 rows). ⇒ **PART A runs on generation 2 for all
three seeds**, every arm re-run in the same invocation, so the table never mixes generations. The
seed-0 generation-1 val-only DEV numbers in `LOG.md` stay DEV. Bars, arms, controls and seeds unchanged.
