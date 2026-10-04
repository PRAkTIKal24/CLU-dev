# w12 spoke/deletion-baselines — LOG

START_UTC 2026-09-21T11:09:27Z · box FORGIS_14 · branch `w12/deletion-baselines`
worktree `C:/chlu/worktrees/CHLU-w12-deletion-baselines` · 3 threads · SCORED RUNS: 0 (none permitted).

## 11:09Z — orientation
- Read `/c/chlu/tasks/w12-spoke-deletion-baselines.md` (standing rules + the one job).
- Read `/c/chlu/evidence/outputs/advisor-wave-plan-w12-w14.md`, `w10-advisor-report.md`,
  `w9-advisor-report.md` (§ refuser: separation **+0.047** cmaze1, +0.020 easy; probe at chance).
- `chlu.__file__` = `C:\chlu\worktrees\CHLU-w12-deletion-baselines\chlu\__init__.py` — inside MY
  worktree. python 3.11.9 · numpy 2.4.1 · jax 0.9.0.

## Assets found (not built by me)
- `chlu/eval/rivals/w4_api.py` — `load_rival(seed, name)`; families
  `mlp, mlp_180k, cont_mlp, cont_mlp_notune, transformer, lstm` (+ `_dropout` variants), `table`,
  `lace`, `poe_ebm`. Fitted checkpoints at `/c/chlu/outputs/spoke-a-rivals/rivals/seed_{0,1,2}/`.
- `chlu/eval/rivals/w4_deletion.py` — existing VALUE-scoped retrain-without comparator (w5).
- `chlu/eval/rivals/w4_learners.py` — `builders/match_width/train/predict/encode_inputs`.
- gen-2 feature cache, 480,000 rows × seeds 0/1/2, at
  `/c/chlu/outputs/spoke-a-rivals/cache_generation2/shapes3d_recon` (`RIVALS-READY.json`).

## 11:40Z — design decisions, declared before any number
1. **Feature source outside the frozen snapshot.** The rows-3/4 harness resolves features
   through `CHLU_GEN_FEATURES_ROOT` (`exp_gen_factor_wells.features_root`, unset ⇒ the frozen
   bridge). Set to `C:/chlu/outputs/spoke-a-rivals/cache_generation2`, so the whole build reads
   **generation-2 features from outside `$CHLU_CLAUDE_ROOT`**. `CHLU_GEN_DATA_ROOT=C:/chlu/data/gen`
   is also outside it. ⭐ Nothing in this spoke reads the frozen snapshot.
2. **Comparators are `RivalRead`s.** Every comparator is returned and SAVED as the same
   `chlu.eval.rivals.w4_api.RivalRead` the rows-3/4 harness already consumes
   (`values (N,F) -> (feats (N,D) float32, oor (N,))`), so the crossover hub calls them by name
   with `load_rival(seed, name, root=<comparators dir>)` and **never reimplements one**.
3. **Matched compute = matched UPWARDS to the exact reference.** The unlearner gets the SAME
   gradient row-visits (`steps × batch`) the retrain-from-scratch of the SAME arm got. Matching it
   to our own data-free adaptation would starve it. Compute matched; DATA ACCESS not — and both
   are printed. `assert_compute_matched` checks equality across arms rather than claiming it.
4. **No tuning loop anywhere.** Epochs/lr/wd are INHERITED from the shipped `w4_learners` grids
   (20 = lower end of `EPOCH_GRID`, 1e-3 = middle of `LR_GRID`), not selected here.

## 11:40Z — two defects found in shipped code, fixed in my worktree
- ⛔ `chlu/eval/rivals/w4_learners.py::train` cast **all** inputs to float32. `tokens` inputs are
  vocabulary indices; under jax 0.9.0 `eqx.nn.Embedding` then raises
  `TypeError: Indexer must have integer or boolean type, got indexer with type float32`, so
  **`transformer` and `lstm` could not be trained through that function at all**. Fixed by
  `_as_model_input` (dtype preserved for integers; the float path is byte-identical).
- `chlu/eval/rivals/w4_api.py`: `load_rival` refused any name not on `LEARNER_FAMILIES`. Extended
  so a checkpoint that DECLARES its `base_family` loads — the module's own stated principle
  ("loadability is decided by what the checkpoint SAYS IT IS"). `save_learner` gained `extra=`.

## 11:44Z — measured, before the build
- Matched to the 524,288 B pin (131,072 params), MEASURED by `match_width` (builds the model):
  `mlp` 130,514 (−0.43%) · `cont_mlp` 131,891 (+0.63%) · `transformer` 130,433 (−0.49%) ·
  `lstm` 130,721 (−0.27%). Refuse-token transformer at `out_dim = D+1`: 130,479 (−0.45%).
- Fit-pool composition (`fit_pool(train_rows, n_pool=131072, rng_seed=20260919)`, 131,072 rows):
  object_hue idx 0 ("red") = **11,044 rows**; shape idx 0 ("cube") = **0 rows** — the cube class is
  the value-holdout condition's own held-out shape. ⛔ So "red ∧ cube" would delete ZERO rows.
  Conjunction pin moved to **object_hue 0.0 ∧ shape 2.0 = 5,511 rows**, declared in `PINS_W12`;
  `partition_fit_rows` REFUSES any no-op deletion rather than printing a zero.

## 11:52Z — build launched
`scripts/w12_deletion_baselines.py --seed 0 --n-fit 16384 --epochs 20`, PID 113577, 3 threads.

## 11:58Z — property checks: 15/15 PASS
`python test_w12_deletion_baselines.py` (hermetic: its own 4-factor grid, its own random
targets, no feature cache, no decoder, no judge). Full list in `test_w12_deletion_baselines.py`;
the load-by-name check confirms a saved comparator round-trips through
`w4_api.load_rival` **with no w12 import at all**.

⛔ A THIRD shipped defect surfaced while writing the checks and is fixed:
`w4_learners.train` built `optax.warmup_cosine_decay_schedule` with
`decay_steps=total`; at `total == 1` optax computes `decay_steps - warmup_steps == 0` and
raises. Guard added (`decay_steps=max(total, warm + 1)`), a no-op for every `total >= 2`.

## 12:14Z — the first seed-0 build STOPPED ITSELF on the item leg, and it was right to
`build_seed0.log` (first pass) ended:
`AssertionError: 164 of 164 IMPOSSIBLE requests were NOT certified impossible`.
⭐ That is my own certifier refusing to train a refuser on a label it cannot support, and the
underlying fact is a FINDING, not a bug:
**an ITEM deletion (random 1% of rows) makes NOTHING impossible.** It clears no
(factor, value) bit and punches no conjunction hole; every deleted row's factor tuple is still
composable from values the pool still holds. So ⛔ **the item-deletion row has no refusal
DISPLAY row at all** — refusing a deleted ITEM would need a per-tuple deletion table, which is
a store bolted to a generator (the same objection `exp_gen_refuse` already makes one level
coarser about the pair-deletion table).
Added `refusal_row_applies(grid, part, bitmap)` — MEASURED, returns the certified count — and
the runner now prints a **DECLARED NOT-RUN with that measurement** instead of a zero.
Rebuild launched 12:14Z, PID 115314.

## 13:48Z — seed 0 rebuilt clean: 50 comparator records, 884.9 s
`BASELINES-READY.seed_0.json`. Item leg's refuse-token now a DECLARED NOT-RUN carrying the
measurement (0 of 164 certified impossible).

## 13:49–14:12Z — seeds 1 and 2
Ran sequentially (PID 117042), 834.1 s and 566.3 s. Runner changed to write
`BASELINES-READY.seed_S.json` and MERGE into `BASELINES-READY.json`, so the gate file carries
every seed that exists rather than the last one that ran.
⭐ `BASELINES-READY.json` → `seeds_present = [0, 1, 2]`, **150 comparator records**.
⭐ Three DIFFERENT feature caches (`ef6359a4cd54b082` / `86b44d0f1e6470e7` / `3ab14e583c3f500d`),
so the spread is a real instrument spread, not three RNG draws on one.
⭐ Every conclusion in FINDINGS.md holds on **3 of 3 seeds**.

## 14:20Z — FINDINGS.md written with 3-seed tables
All three deletion tables carry the 3-seed mean plus the three per-seed values.
Totals: 0 scored runs · 0 GPU · 3 threads · 2,285 s of compute across the three seeds.

## 14:26Z — independent verification of the gate file
`scripts/w12_verify_baselines_ready.py --load-every` — imports `w4_api` ONLY, never the builder,
so a defect in the builder cannot hide itself. Result: **BASELINES-READY VERIFIED**.
- 5/5 entry points resolve to real callables
- 150/150 records carry a complete cost line
- every arm inside ±10% of the 131,072-param pin (worst +0.625%)
- matched compute equal across 8 arms × 3 legs × 3 seeds
- every measured NOT-RUN carries a reason
- ⭐ all 150 records LOAD BY NAME through `w4_api.load_rival` and return finite features
  (126 distinct checkpoints, 42 per seed; the 4 `as_trained` arms are shared across the three legs)

## 14:15Z — commit + bundle
`1635e0a [w12] spoke/deletion-baselines: the three deletion comparators, wired by name`
(6 files, +2190 / −11).
`git bundle create .../w12-deletion-baselines.bundle w12/deletion-baselines` — **that branch
only, never --all**. `git bundle verify`: *"is okay … records a complete history"*.
`git bundle list-heads`: **exactly one ref** —
`1635e0a3e95c92a7dbbfd4052940f688425aacf1 refs/heads/w12/deletion-baselines` (6,028,480 B).

## 14:30Z — process hygiene
All of my own PIDs (113577, 115314, 117042) have exited; nothing of mine is detached.
⛔ **No box-wide kill was run.** The python processes still alive on the box
(`w12_ladder_leg.py` rungs at seeds 0/1/2, and `exp_gen_bridge`) belong to
`spoke/kway-ladder-harness` and the gen-2 cache spoke and were left untouched.

## TOTALS
- **Scored runs spent: 0.** GPU: **0**. Threads: 3. Compute: 2,285 s across three seeds,
  plus the dry runs and the property checks.
- Deliverables in `/c/chlu/outputs/w12-deletion-baselines/`: `BASELINES-READY.json`
  (seeds 0/1/2, 150 records) · `BASELINES-READY.seed_{0,1,2}.json` · `FINDINGS.md` · `LOG.md` ·
  `build_seed{0,1,2}.log` · `comparators/seed_{0,1,2}/` (42 checkpoints each, 126 total) · `smoke/seed_{0,1,2}/` ·
  `w12-deletion-baselines.bundle`.
