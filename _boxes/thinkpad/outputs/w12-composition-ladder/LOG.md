# w12 hub/composition-ladder — LOG

## 2026-09-21T11:31:33Z — START
- START_UTC.txt written.
- Task file read: /c/chlu/tasks/w12-hub-composition-ladder.md

## Gate checks (at start)
- GATE (mechanical): `/c/chlu/outputs/w12-kway-ladder-harness/LADDER-READY.json` — **EXISTS**
  (22591 B, mtime 2026-09-21 13:17 local). Gate OPEN.
- SEED GATE: `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json` — **ABSENT** at start
  (dir contains only START_UTC.txt, LOG.md, PIDS.txt, bridge/, bridge_seed_3.log).
  => **DECLARED: 3 seeds** for every row in this hub. Not 5.

## Instrument acquisition
- Fetched `/c/chlu/outputs/w12-kway-ladder-harness/w12-kway-ladder-harness.bundle`
  (list-heads: exactly one ref `refs/heads/w12/kway-ladder-harness` @ `04a3d07`; `git bundle verify`
  = "okay ... complete history").
- **Fast-forwarded** `w12/composition-ladder` onto `04a3d07`. `git diff --stat aa91785 04a3d07` =
  3 new files, 1493 insertions, **zero tracked files modified** — the harness is purely additive.
  => ⛔ I build NO k-way family of my own. `chlu/data/gen_kway_ladder.build_kway_split` is the sole source.
- `chlu.__file__ = C:\chlu\worktrees\CHLU-w12-composition-ladder\chlu\__init__.py` (asserted).

## K* gate
- `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json`: `"K_star": null`,
  `"K_star_reason": "K=5 reference row absent; rule not applicable"`; all 9 K are DECLARED NOT-RUN there.
  => **K* is NOT published. DECLARED: single design runs at K = 5.** Carried on every row.

## Feature cache
- `CHLU_GEN_FEATURES_ROOT` is **unset** in the ambient env (confirmed: `env | grep -i chlu`).
  Unset => `exp_gen_factor_wells.features_root()` falls back to `CHLU_GEN_BRIDGE_ROOT` = cache
  GENERATION 1, which w5-common 3.8 forbids. Every run of mine sets
  `CHLU_GEN_FEATURES_ROOT=C:/chlu/outputs/spoke-a-rivals/cache_generation2` and calls
  `exp_gen_delete_row7.assert_generation2(seed)` before any number.

## Instrument check (index arithmetic only, no features) — instrument/INSTRUMENT.json
Run 11:33Z, BEFORE any arm.
- Shipped value-hold-out condition on disk: 54 cells (49 nway + 3 pair + 2 value),
  train 284,472 / test 195,528. Row 3 = 60 corners x 36 rows; row 4 = 49 corners x 72 rows.
- Ladder rungs from `gen_kway_ladder.build_kway_split` (the HARNESS's, not mine):
  | k | cells | corners | rows/corner | train | test | disjoint | all rows in hold-out |
  |---|---|---|---|---|---|---|---|
  | 2 | 5   | 60  | 36 | 288,000 | 192,000 | True | True |
  | 3 | 17  | 120 | 72 | 180,000 | 300,000 | True | True |
  | 4 | 29  | 120 | 72 | 244,800 | 235,200 | True | True |
  | 5 | 65  | 120 | 72 | 266,400 | 213,600 | True | True |
  | 6 | 125 | 120 | 72 | 279,360 | 200,640 | True | True |
- ⭐ MEASURED, ends: the k=2 corner list equals the SHIPPED `pair_subcorners` row set for row
  set, IN ORDER (60/60). k=6's FIRST 49 corners equal the SHIPPED `nway_corners` row set for
  row set, IN ORDER (49/49); all 49 shipped row-4 corners are k=6 corners.
- ⚠ INSTRUMENT DEFECT FOUND IN THE HARNESS (reported, not routed around):
  `gen_kway_ladder.assert_k2_matches_row3` returns `corner_ids_identical: true` and names its
  reference as `exp_gen_factor_wells.pair_subcorners`, but the code compares against
  `gen_kway_ladder.row3_subcorners`, a longhand re-implementation in the same module. The two
  disagree on the corner ID: `pair_subcorners` labels the floor window by its SEQUENTIAL index
  (`f{fi}` from `enumerate` -> f0..f4) and `row3_subcorners` by its START index
  (`f{int(win[0])}` -> f0,f2,f4,f6,f8). Against the real `pair_subcorners` the ids do NOT match.
  ⭐ The ROW SETS do match exactly, so the defect is in the assertion's LABEL, not in the ladder.
- ⛔ The rungs sit on DIFFERENT SPLITS from the shipped condition (k=6 holds out 120 kway cells
  = 200,640 rows against the shipped 49 nway cells = 195,528). So "k=2 reproduces row 3" is a
  claim about NUMBERS and is tested as one: a 6th condition `shipped` is run with the identical
  arms and seeds. Declared in PREREG section 6 before any arm number.

## PREREG
- `PREREG.md` + `PREREG.timestamp.txt` written **2026-09-21T11:38:41Z**, with
  `runs/3_seed0/` holding no `rows_*.json` and no other leg directory existing.
- Fixes: 3 seeds; K=5 DECLARED (K* null); the arm list; the two charter-C.2 controls; and
  ⭐ the BEST-LEARNED-BASELINE rule (per (k, seed), the learned arm with the LARGEST mean) —
  the strongest opponent, chosen before any number, running against this hub's own interest.

## Wall-clock measurements (probe leg, runs/3_seed0, logs/probe_k3_s0.log)
- corners assertion 1.6 s; fit pool draw 0.7 s; channel-factor relevance (2048x6) 45.3 s;
  partitions written at 67.9 s; judge load 0.3 s (decoder 01ef91f75eea + classifier c8741ac417d8).
- `single_design_K5` store fit: 17.8 s, 64 atoms x 2048 f32 = 524,288 B on 131,072 rows,
  ranks [63,63,63,12,41], **rank/width [0.0814, 0.0675, 0.2877, 0.1967, 0.6721]**.
- span targets 12.9 s; TensorAddress ALS ~10.7 s/sweep x 40 sweeps ~= 430 s per address book.
  4 books per leg (pca, pca+random_atoms, sd, sd+random_atoms; `landscape_shuffled` REUSES the
  arm's own book by construction) => ~30 min/leg before scoring.
- Box: `nproc` = 3, shared with other agents' runs. 3 lanes x 1 thread.

## Launch
- PIDs captured in `PIDS.txt` (⛔ no box-wide kill; only these).
- lane A (pid 115952): 2:0 6:0 shipped:0 3:1 4:0 5:0
- lane B (pid 115964): 2:1 6:1 shipped:1 3:2 4:1 5:1
- probe (pid 115696): 3:0
- lane C queued after the probe: 2:2 6:2 shipped:2 4:2 5:2

## AMENDMENT 1 — 11:42:09Z, before any arm number existed on disk
`runs/*/rows_*.json` matched NOTHING at that instant (verified by glob).
`AMENDMENT-1.md` + `AMENDMENT-1.timestamp.txt`. It moves the **1x byte-matched learned
column** from a declared NOT-RUN into the run, because the ground for the NOT-RUN was wrong:
`/c/chlu/outputs/w12-deletion-baselines/comparators/seed_0/cont_mlp__as_trained.rival.json`
`cost_line.seconds = 11.2617` (and 8.5121 / 8.8698 / 7.2493 / 9.7728 / 9.9941 on its five
siblings) — a learner at this byte pin costs SECONDS. Two arms are added, both fitted by this
hub on **each rung's own fit pool** (`scripts/w12_ladder_1x.py`):
- `learned_ref_ownpool` @ 131,072 params — the 524,288 B reference pin, own pool: this also
  removes the `montero_combination` population caveat.
- `learned_1x` @ the MEASURED total bytes of `single_design_K5` on that leg / 4.
⛔ The best-learned-baseline rule of PREREG section 5 is UNCHANGED and still runs over the
five pre-existing learned arms, so panel B does not move.

## Lane C queued
- waiter pid 116290 polls the probe pid 115696 and then runs lane C: 2:2 6:2 shipped:2 4:2 5:2.
- Steady-state ALS: ~13.5 s/sweep with 3 concurrent 1-thread processes
  => ~540 s per address book, 4 books/leg, ~45 min/leg, 3 legs at a time.

## AMENDMENT 2 — 11:46:30Z, before any arm number existed on disk
`runs/*/rows_*.json` still matched NOTHING (verified). `AMENDMENT-2.md` fixes the two words
the task asks the answer to be reported in, so neither is chosen after the curve is visible:
- the metric's CEILING is the corner's own row count (36 at k=2, 72 at k>=3, measured in
  `instrument/INSTRUMENT.json` and already stated in
  `/c/chlu/evidence/outputs/w10-seeds-and-tables/TABLES.md`), so any cross-k statement uses the
  ceiling-normalised score and every table prints BOTH the raw count and the ceiling;
- **deteriorates at k** = normalised score below its own k=2 value on EVERY seed;
- **breaking point** = the smallest k>=3 that clears the bar at every lower rung and fails at
  k, where "clears the bar" is the harness's own `beats_beyond_margin` against the PREREG
  section-5 best learned baseline on every seed;
- the control gate comes FIRST: a DECORATIVE rung claims no row and is excluded from the
  breaking-point determination, with the exclusion printed beside it.

## Smoke of the AMENDMENT-1 runner
`scripts/w12_ladder_1x.py --rung 3 --seed 0 --which ref` launched 11:45Z against the probe
leg's own `partitions.npz` (task by7jelnyg). Its purpose is to prove the fit+score path before
15 legs depend on it.

## AMENDMENT 3 — 11:49:46Z, with ONE learned number on disk and ZERO arm numbers
At that instant the only `rows_*.json` anywhere in `runs/` was
`runs/3_seed0/rows_learned_ref_ownpool.json`. No `single_design_K5`, no `pca_tensor`, no
control, at any rung or seed.
- ⭐ MEASURED: `learned_ref_ownpool` (130,514 params, `match_width` -0.43% of the 131,072
  target, width 61, fitted on 32,768 rows of the k=3 pool, final train MSE 0.0325) scores
  **8.2167 +- 9.3864** distinct valid novel tuples over k=3's 120 corners, judge accuracy
  0.1141. The published rival set on the shipped rows is four orders below that —
  `w10-seeds-and-tables/TABLES.md`: `rival_mlp_180k @ alone` = 0.006 (row 3), 0.000 (row 4).
  ⇒ **fitting the learner on the pool the arm is fitted on beats every architecture in spoke
  A's set.** Keeping PREREG section 5 as written would have this hub quote a soft bar under a
  hard bar's name.
- `AMENDMENT-3.md` widens the best-learned-baseline candidate set to all SEVEN learned arms.
  ⛔ The change can only RAISE the opponent, so every margin this hub reports can only get
  smaller. Both readings are computed and printed: `best_learned_5` (the narrow rule as
  written) and `best_learned_all` (the amended rule, which the figure and every verdict use).

## Figure colour — the dataviz procedure, run rather than eyeballed
The programme's house palette IS the skill's reference instance (`w10_build_tables.py:CAT`
== `references/palette.md`'s 8 slots). It is validated for ADJACENT pairs; a line chart admits
ANY pair, so it was re-validated at `--pairs all`:
- 8 slots: **FAIL** — normal-vision worst `#e34948`<->`#eb6834` dE 7.1; CVD worst
  `#008300`<->`#eb6834` dE 3.2 (protan).
- 7/6/5 slots: still FAIL (`#eda100`<->`#eb6834` dE 13.7 at 5 slots).
- ⭐ `#2a78d6,#eb6834,#1baf7a,#4a3aa7` (4 slots): **ALL CHECKS PASS** — CVD worst dE 9.2
  (deutan), normal-vision worst dE 16.3; one contrast WARN (`#1baf7a` 2.74:1), discharged by
  direct labels **and** the table view (LADDER.md / LADDER.csv).
⇒ per the skill's own remedy (cut series or facet, never generate a hue), hue carries the
BYTE PIN + POPULATION group (4 entities) and line style carries the arm inside the group.
⛔ Every arm still gets its own line, and its own row in the tables.

## First arm numbers, and a MEASURED DEFECT in the shipped read-back path
| leg | arm | corners | ceiling | distinct valid novel / corner | judge acc | scoring seconds |
|---|---|---|---|---|---|---|
| k=3 seed0 | single_design_K5 | 120 | 72 | **56.9500 +- 11.5423** | 0.7910 | 918.4 |
| k=2 seed0 | single_design_K5 | 60 | 36 | **28.6333 +- 7.6623** | 0.7954 | 762.6 |
| k=2 seed1 | single_design_K5 | 60 | 36 | **31.6500 +- 5.9797** | 0.8792 | 784.9 |
| k=3 seed0 | learned_ref_ownpool | 120 | 72 | **8.2167 +- 9.3864** | 0.1141 | 302.4 (incl. fit) |
Sources: `runs/{3_seed0,2_seed0,2_seed1}/rows_*.json`, `logs/*.log`.

⚠ **Four times the rows for 1.2x the time.** k=2 is 2,160 requests and k=3 is 8,640, and the
SAME arm costs 762.6 s and 918.4 s. A cost that barely moves with the row count is not
arithmetic. Cause, found in shipped code: `chlu/eval/gen_probe.py:44` and `:59` build
`eqx.filter_jit(jax.vmap(model))` INSIDE the function body, so each call gets a fresh wrapper
and a fresh JAX compilation cache — the decoder and the classifier are **recompiled on every
call**. `exp_gen_factor_wells.Judge.read_back` calls them once per batch and this hub calls
`read_back` once per corner, so a 120-corner arm pays 240 compilations.
⛔ Reported, not silently worked around. `scripts/w12_fastjudge.py` hoists the jit and changes
NOTHING arithmetic (same frozen decoder, same hash-asserted classifier, same `model_dtype`,
same `jax.vmap`, same batch 256, same argmax, same int64). It is gated by a POSITIVE CONTROL,
`assert_identical`, on real fit-pool features: the leg refuses to produce a number unless the
two paths agree element for element, and the check is written into every `LEG.json`.

## The bottleneck, MEASURED rather than assumed — and my first diagnosis was incomplete
`logs/fastjudge_check.log` (1 thread, real fit-pool features, 720 rows in the leg's own
10 per-corner calls):
- ⭐ **EQUAL** — `FastJudge.read_back` is element-for-element identical to the shipped
  `exp_gen_factor_wells.Judge.read_back`. The positive control passes.
- shipped `Judge.read_back`: **16.1 s** / 720 rows -> ~194 s extrapolated to 8,640 rows.
- `FastJudge.read_back`: **5.6 s** / 720 rows; **69.8 s** measured over 8,640 rows in 120 calls.
- **speedup on read-back: 2.8x.**
⚠ But 194 s is not 918 s, so the per-call jit rebuild is NOT the whole story and I said so
before acting. `logs/batch_check.log`, on the real k=3 store arm and its fitted address book:
- `exp_gen_read.arm_features` batched vs per-corner: **IDENTICAL: True, max|diff| 0.0**;
  cost **< 0.05 s for 720 rows** — ⛔ the TensorAddress span read is NOT the bottleneck either.
- `read_back` per-corner(72) 7.7 s vs batched(256) 8.2 s over 720 rows, tuples **IDENTICAL: True**.
⇒ the 918.4 s / 762.6 s / 784.9 s arm timings are dominated by **CPU contention**, not by any
one call: those legs ran 3-4 processes (one at OMP=3) against `nproc = 3`, alongside three
other agents' jobs. ⛔ No number is discarded on the strength of a guess; the four scores
already on disk stand and the batched path is proven identical to the path that produced them.

## Restart, 11:59Z — two verified no-ops adopted, nothing pinned changed
1. `FastJudge` (jit hoisted out of the per-call body), gated per leg by `assert_identical` on
   real features; the check is written into every `LEG.json` as `fastjudge_positive_control`.
2. `score_arm` answers all corners in ONE `arm_features` call and ONE `read_back` call at the
   pinned `eval_batch = 256`, then slices by offsets. Both proven IDENTICAL above.
⛔ Metric, margin, estimand, `eval_batch`, `READ_PINS`, seeds and K are untouched.
⛔ Process hygiene: only MY processes were stopped — the three leg pythons (Windows PIDs
5176 / 28560 / 31104), then the lane shells, then two legs the dying lanes had spawned.
Verified afterwards: `lane shells left: 0`, `my leg pythons left: 0`, and the three OTHER
agents' jobs still running and untouched (`exp_gen_bridge --seed 3`, `w12_val_sweep`,
`w12_deletion_baselines --seed 1`). ⛔ No box-wide kill at any point.
- new lanes (seed-major, rung order 2, 6, shipped, 3, 4, 5 so both ends and the reproduction
  reference land before the middle): A=118123 (seed 0), B=118130 (seed 1), C=118140 (seed 2).
- surviving cached artifacts reused, not recomputed: `partitions.npz`, `single_design_store.pkl`,
  `address_single_design_K5_*.pkl`, `pca_content.pkl` in 2_seed0 / 2_seed1 / 3_seed0.

## ⭐⭐ k=2: the single design and `pca_tensor` are the SAME ARM for composition — measured
`runs/2_seed0/rows_single_design_K5.json` vs `runs/2_seed0/rows_pca_tensor.json`:
- same 60 corners in the same order; **per-corner `distinct_valid_novel_tuples` IDENTICAL,
  max|diff| = 0**; `n_valid` identical on every corner. Both read **28.6333 +- 7.6623**,
  judge accuracy 0.7954. Seed 1: both **31.6500 +- 5.9797**, judge accuracy 0.8792.
- ⛔ This is NOT a wiring accident. The two atom blocks are DIFFERENT
  (`max |mode diff|` per well = 0.730 / 0.559 / 0.621 / 2.000 / 1.435) and the two address
  books are separate files of different size (383,876 B vs 318,345 B).
- ⭐ THE MECHANISM, measured: the read is a SPAN projection, so only the subspace matters.
  Well widths [777, 936, 216, 60, 59].
  | well | d | rank sd | rank pca | ‖P_sd − P_pca‖_max |
  |---|---|---|---|---|
  | 0 | 777 | 64 | 64 | 1.824e-08 |
  | 1 | 936 | 64 | 64 | 8.219e-09 |
  | 2 | 216 | 64 | 64 | **0.000e+00** |
  | 3 | 60 | **15** | **60** | (different rank) |
  | 4 | 59 | **41** | **59** | (different rank) |
  On the three WIDE wells (1,929 of 2,048 channels) the two stores span the SAME subspace to
  1e-8. On the two NARROW wells the single design's variance-share truncation keeps 15 of 60
  and 41 of 59 while `pca_tensor` keeps all of them — and that difference changes **not one of
  the 2,160 judged requests**. Those 119 channels carry nothing the judge can see.
⇒ ⭐ **The task's falsifier #2 fires, in its strongest form: `pca_tensor` does not merely match
the single design at k=2, it is bit-identical to it.** The single design's measured 12,472,424
– 12,898,080 B (23.79x – 24.60x the 524,288 B pin, `_smoke/BYTES.json`) buys **exact deletion**,
not composition. Reported plainly, as the task requires.

## ⭐ The first charter-C.2 control, and a nuance about rank/width
`single_design_K5__landscape_shuffled` = **0.0000 +- 0.0000**, judge accuracy **0.0000**, on
BOTH seeds (`logs/2_seed{0,1}.log`, 21.2 s / 19.3 s). ⇒ total collapse; the store is NOT
decorative at k=2 on this control.
⚠ And rank/width alone would have said otherwise. At k=2 seed 0, `pca_tensor` runs
rank/width = [0.082, 0.068, 0.296, **1.000**, **1.000**] — two wells FULLY saturated, above
wave 9's decorative threshold of 0.938 — while its atoms still carry everything. ⛔ rank/width
is a WARNING, not a verdict; only the control decides. Wave 9's own lesson, sharpened.

## SEED GATE RE-EVALUATED — 12:28:44Z (AMENDMENT-5)
`/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json` now exists,
`"produced_utc": "2026-09-21T12:27:01Z"` — 56 min AFTER this hub started (11:31:33Z), so the
task's start-time clause had already fixed the 3-seed branch. Re-checked per seed anyway.

Gate = non-null `array_shape` AND 59 blocks:
| seed | array_shape | n_blocks | len(block_sha256) | scope | gate |
|---|---|---|---|---|---|
| 3 | [480000,128,4,4] | 59 | 59 | full | **PASSES** |
| 4 | absent | 0 | — | — | **FAILS** (`status: NOT_RUN`, "generation-2 cache for this seed was not completed") |

⛔ But seed 3 still cannot be scored, and the reason is NOT the cache. Measured:
- `$CHLU_GEN_DECLIFT_DIR` holds `decoder_v1_s{0,1,2}.pkl` only; `winner_hashes.json` keys are
  exactly `['seed_0','seed_1','seed_2']`.
- `gen-external-classifier/` holds `seed_0 seed_1 seed_2` only.
- `spoke-a-rivals/rivals/` holds `seed_0 seed_1 seed_2` only.
- `fw.Judge(3)` -> `FileNotFoundError ... decoder_v1_s3.pkl`. The judge IS the frozen decoder
  plus the hash-asserted classifier, so no arm can be scored at seed 3 at all.
- CACHE-READY itself says so: seed 3 `status = CACHE_READY_BRIDGE_PENDING`, `bridge.NOT_RUN`.

⭐ **COUNT USED, DECLARED: n_seeds = 3 (seeds 0, 1, 2)** on every row, table and figure panel.
- seed 3 = DECLARED NOT-RUN (cache passes the gate; decoder/classifier/rivals absent).
- seed 4 = DECLARED NOT-RUN (fails the gate: no array_shape, 0 blocks).
⛔ Never a zero, never a blank. No 3-seed and 4-seed row is mixed; `n` is printed everywhere.

## AMENDMENT-4.md written — an amendment CONSIDERED and NOT MADE
`AMENDMENT-4.timestamp.txt` (11:54:52Z) had no document beside it. It was stamped while
drafting a cut of the scored corner budget from the harness's 120 corners to its pre-declared
bar of 49, when the matrix looked like ~20 h. The cut was **not made**: the FastJudge +
batching measurements showed the timings were CPU contention, not corner count, so the full
120-corner budget fits and every rung is scored on all of the harness's corners. The dangling
stamp is now explained rather than left to inference.

## HEAD RULING ON K — recorded 12:38:07Z, applied immediately (HEAD-RULING-K.md)
> "Finish the ladder at K=5 as declared. Main text carries ONE K only, K-star; the K=5 ladder
> goes to the appendix labelled as the sweep's endpoint. Never both in main text. I will re-run
> the store arm at K-star as a free replication once the crossover hub publishes it — do not
> wait for it."
Applied:
- ⛔ the run does NOT change and does NOT wait. Lanes continue exactly as launched; K* is still
  `null` in `CROSSOVER-val.json` and ⛔ no row of this hub will be relabelled a K* row.
- ⭐ PLACEMENT is now carried BY THE ARTIFACTS, not by a covering note:
  `LADDER.json` gains `placement`, `K_of_the_store_arms = 5`, `K_star = null` +
  `K_star_source`; `LADDER.md` prints the placement as its first line; the figure carries
  "APPENDIX FIGURE · the K-sweep's ENDPOINT at K = 5 (not main text)" in its **suptitle** and
  repeats the ruling in its caption. A number cannot be lifted out of this hub without the
  placement travelling with it. Verified by regenerating all three from the partial data.
- ⚠ Flagged FOR the K* re-run, with a checkable prediction rather than a hope: this hub has
  already measured that at K=5 `pca_tensor` and the single design are BIT-IDENTICAL on all
  three seeds at k=2, because they span the same per-well subspace (‖P_sd − P_pca‖ ≤ 1.8e-08 on
  the three wide wells, 1,929 of 2,048 channels). ⇒ if the K* re-run is to say anything new
  about COMPOSITION, the SPAN must move at K*, not the ledger — so the re-run should report
  ‖P_sd − P_pca‖ per well at K*, exactly as this hub reports it at K=5.

## ⭐ RUNG k=2 COMPLETE — all 3 seeds, 11/11 arms each, `failed=[]` on every leg
Source: `LADDER.json:rungs.2`, `runs/2_seed{0,1,2}/`. 60 corners/seed, 36 rows/corner,
ceiling 36, `pairwise_disjoint: True`, `all_rows_held_out: True`, cache generation 2 scope
full, `fastjudge_positive_control: True` (1,536 elements) on every leg.

| arm | across-seed mean | [min .. max] |
|---|---|---|
| `single_design_K5` | **30.5389** | 28.6333 .. 31.6500 |
| `pca_tensor` | **30.5389** | 28.6333 .. 31.6500 |
| `rival_lstm` | 20.8333 | 19.3333 .. 23.7500 |
| `rival_mlp` | 20.7833 | 17.0167 .. 23.3333 |
| `rival_transformer` | 17.5944 | 12.7500 .. 24.2500 |
| `rival_table` (FLOOR) | 5.1667 | 3.4833 .. 6.1833 |
| `rival_cont_mlp` | 4.9944 | 3.8833 .. 6.5333 |
| `single_design_K5__landscape_shuffled` | **0.0000** | 0.0000 .. 0.0000 |
| `pca_tensor__landscape_shuffled` | **0.0000** | 0.0000 .. 0.0000 |
| `single_design_K5__random_atoms` | **0.3611** | 0.0833 .. 0.7500 |
| `pca_tensor__random_atoms` | **0.3611** | 0.0833 .. 0.7500 |
| `learned_ref_ownpool`, `learned_1x` | DECLARED NOT-RUN at this rung so far — the AMENDMENT-1 pass runs after the ladder lanes free the cores |

⭐ **CONTROLS: both stores survive both charter-C.2 controls at k=2, on every seed.**
`landscape_shuffled` delta +28.633 / +31.650 / +31.333, `random_atoms` delta +28.383 /
+31.567 / +30.583; every CI95 lower bound is above +26.4 and every sign test p = 1.73e-18.
⛔ Nothing here is decorative. Contrast wave 9, where the maze arm's controls read 0.94/0.92
against 1.0000 and no row was claimed.

⭐ **MARGIN vs the BEST LEARNED BASELINE (AMENDMENT-3 rule; both readings AGREE at k=2):**
| seed | opponent | delta | CI95 | sign p | beats +1.0 |
|---|---|---|---|---|---|
| 0 | `rival_lstm` 19.4167 | **+9.2167** | [+6.7333, +11.8333] | 1.52e-07 | ✓ |
| 1 | `rival_mlp` 22.0000 | **+9.6500** | [+7.3500, +12.0667] | 6.4e-11 | ✓ |
| 2 | `rival_transformer` 24.2500 | **+7.0833** | [+5.2167, +9.0171] | 1.38e-07 | ✓ |
⇒ both store arms CLEAR THE BAR at k=2 on every seed. Identical numbers for the two arms,
for the span reason measured earlier.
⚠ **The opponent's identity CHANGES with the seed** — LSTM, then MLP, then transformer. That is
exactly why PREREG section 5 fixed the rule per (rung, seed) rather than naming an architecture.
⚠ And the learned set is NOT a floor here: 17.6–20.8 against the store's 30.5 on a ceiling of
36. ⛔ The `w10-seeds-and-tables/TABLES.md` figure of 0.006 for `rival_mlp_180k` is a DIFFERENT
INSTRUMENT (the shortened `w8_v1_short` decoder) and its own table says so; these runs use the
incumbent frozen decoder `01ef91f75eea (v1_capacity_schedule)`.

## ⭐⭐ THE k=2 HEADLINE — the store's margin is an artefact of the BASELINE'S BYTE PIN
`LADDER.json:rungs.2`, all 3 seeds, 60 corners/seed, ceiling 36.
`learned_1x` = an MLP matched to the single design's OWN MEASURED total, fitted on the rung's
own pool (`runs/2_seed*/ONEX_TARGET.json`: 524,288 B atoms + 12,284,920 B statistics +
158,144 B address book = **12,967,352 B = 24.7333x** the 524,288 B reference pin =>
**3,241,838 float32 params**; `match_width` landed at 3,241,076 / 3,161,048 / 3,131,564,
within -0.03%..+0.03%).

| seed | store arms | `learned_1x` | paired delta (store − 1x) | CI95 | 5-rival reading |
|---|---|---|---|---|---|
| 0 | 28.6333 | **33.2000** (acc 0.9222) | **−4.5667** | [−6.383, −2.883] | +9.2167 vs lstm |
| 1 | 31.6500 | **33.8667** (acc 0.9407) | **−2.2167** | [−3.367, −1.200] | +9.6500 vs mlp |
| 2 | 31.3333 | **32.9000** (acc 0.9139) | **−1.5667** | [−2.717, −0.500] | +7.0833 vs transformer |

⭐ Every CI95 upper bound is BELOW zero: the store is behind on every seed.
⇒ under AMENDMENT-2 section 3, both store arms **NEVER CLEARED THE BAR at k=2**, so they have
**no breaking point** — there was never a row to lose. ⛔ Reported as the result.
⚠ At the 524,288 B REFERENCE pin the same arms win by +7.08..+9.65. The entire apparent
composition advantage is the baseline's byte pin. This is the w10 referee's MF3 ("the
sufficient statistic that buys the exactness is 24.5x the byte budget and appears in no table")
made quantitative ON THE COMPOSITION ROW.

## ⛔ CORRECTION issued against this hub's own earlier claim (AMENDMENT-3-CORRECTION.md, 13:05:53Z)
`AMENDMENT-3.md` justified widening the opponent set by asserting "fitting the learner on the
pool the arm is fitted on is worth more than any architecture in spoke A's set", citing
`learned_ref_ownpool` 8.2167 at k=3 against a published 0.006. ⛔ That was a CROSS-INSTRUMENT
comparison — the 0.006 is the shortened `w8_v1_short` decoder, and that table says so itself.
Same-instrument at k=2: own-pool 7.4833 mean (5.5667/7.5833/9.3000) vs montero-pool rivals
17.59–20.83. **Own-pool fitting at the reference pin is ~13 points WEAKER.** The sentence is
withdrawn. ⭐ The RULE is unchanged and still right — a maximum over a superset can only raise
the opponent — and it matters through `learned_1x`, not `learned_ref_ownpool`.

## Aggregator defect found and fixed (my own)
The bar test compared an arm against itself when that arm WAS the best learned baseline, and
scored the tie as a failure — `learned_1x` was being reported as "NEVER CLEARED THE BAR".
Fixed: a self-comparison is `bar_test: NOT APPLICABLE`, excluded from the breaking-point
sequence, and the arm is reported as "it IS the bar". ⛔ No store-arm number is affected.

## ⭐ BOTH ENDS COMPLETE — k=2 and k=6, 3 seeds each, 13 arms each, `failed=[]` everywhere
`LADDER.json:rungs.{2,6}`. k=2: 60 corners x 36 rows, ceiling 36. k=6: 120 corners x 72 rows,
ceiling 72. Controls collapse at BOTH ends on every seed
(`landscape_shuffled` 0.0000 everywhere; `random_atoms` 0.2583 / 0.1583 / 0.4917 at k=6).
⇒ the store is NOT decorative at either end and a row is claimable at both.

| rung | seed | store | `learned_1x` | delta vs ALL7 | CI95 | delta vs 5-rival |
|---|---|---|---|---|---|---|
| k=2 | 0 | 28.6333 | 33.2000 | **−4.5667** | [−6.383, −2.883] | +9.2167 |
| k=2 | 1 | 31.6500 | 33.8667 | **−2.2167** | [−3.367, −1.200] | +9.6500 |
| k=2 | 2 | 31.3333 | 32.9000 | **−1.5667** | [−2.717, −0.500] | +7.0833 |
| k=6 | 0 | 62.6917 | 68.4667 | **−5.7750** | [−7.517, −4.092] | +18.2583 |
| k=6 | 1 | 63.9917 | 68.5583 | **−4.5667** | [−5.758, −3.450] | +22.4250 |
| k=6 | 2 | 66.9833 | 68.8250 | **−1.8417** | [−2.933, −0.742] | +14.9917 |

⭐⭐ **TWO FINDINGS THAT POINT IN OPPOSITE DIRECTIONS, BOTH REAL:**
1. ⛔ **The store is behind a 1x-byte-matched learner at BOTH ENDS, on EVERY seed** — all six
   CI95 upper bounds are below zero. Under AMENDMENT-2 section 3 it **NEVER CLEARED THE BAR**,
   so it has **NO BREAKING POINT**: there was never a row to lose. `learned_1x` is the best
   learned baseline at every rung and every seed (6/6).
2. ⭐ **The store does NOT deteriorate as k grows — it IMPROVES.** Ceiling-normalised it reads
   0.7954 / 0.8792 / 0.8704 at k=2 and 0.8707 / 0.8888 / 0.9303 at k=6;
   `deteriorates_vs_k2 = False` at k=6 on all three seeds. ⛔ The ONE QUESTION's premise
   ("do all arms deteriorate from their 2-way result as k grows") is FALSE on Shapes3D for
   this arm.
⚠ And the margin over the 524,288 B REFERENCE rivals **GROWS** with k: +7.08..+9.65 at k=2 to
+14.99..+22.43 at k=6. ⛔ So the choice of baseline byte pin does not merely shift the level —
it flips the sign of the k-trend's interpretation. Both columns are printed on every row.

## ⭐⭐ THE NON-NEGOTIABLE PASSES — both ends reproduce, every arm, every seed
`LADDER.json:end_reproduction`. Criterion fixed in PREREG section 6 BEFORE any number: the end
reproduces if the rung mean and the shipped mean agree within that arm's own across-corner sd,
per seed.

**k=2 vs the SHIPPED row 3** (60 corners vs 60) — VERDICT: *reproduces on every arm and every seed*
| arm | max abs diff | per seed (rung vs shipped) |
|---|---|---|
| `single_design_K5` / `pca_tensor` | **1.0833** | 28.6333/29.7167 · 31.6500/31.6167 · 31.3333/31.4500 |
| `rival_{mlp,lstm,transformer,cont_mlp,table}` | **0.0000** | identical on every seed |

**k=6 first 49 corners vs the SHIPPED row 4** (49 vs 49) — VERDICT: *reproduces on every arm and every seed*
| arm | max abs diff | per seed (rung vs shipped) |
|---|---|---|
| `single_design_K5` / `pca_tensor` | **0.5306** | 63.1837/63.6122 · 63.7959/64.3265 · 67.9184/67.6531 |
| `rival_*` | **0.0000** | identical on every seed |

⭐ The STRUCTURE of the result is itself the evidence: the five rivals reproduce EXACTLY because
their features are a function of the REQUEST alone and not of the rung's fit pool, so only the
novelty mask could have moved them and it did not. The store arms move by 0.03..1.08 — exactly
the quantity the `shipped` leg exists to measure, since a different split draws a different
131,072-row fit pool and therefore a different store. ⇒ **the split change does not carry the
ladder's signal, and the ladder is ON-INSTRUMENT and may be plotted.**
⛔ Note the shipped leg's 109 corners pool two ceilings (row 3's 60 at 36, row 4's 49 at 72);
the test slices them [0:60] and [60:109] rather than comparing pooled means.

## ⛔⛔ INSTRUMENT FINDING — for k >= 3 the RUNGS SHARE ONE CORNER SET
`instrument/corner_set_equality.txt`, built from the harness's own
`gen_kway_ladder.build_kway_split`:

| rung | cells | corners | rows/corner | held-out rows |
|---|---|---|---|---|
| k=2 | 5 | 60 | 36 | 192,000 |
| k=3 | 17 | 120 | 72 | **300,000** |
| k=4 | 29 | 120 | 72 | 235,200 |
| k=5 | 65 | 120 | 72 | 213,600 |
| k=6 | 125 | 120 | 72 | 200,640 |

Corner-set equality, comparing the corners' ROW SETS as sets:
```
      k=2 k=3 k=4 k=5 k=6
 k=2    =   .   .   .   .
 k=3    .   =   =   =   =
 k=4    .   =   =   =   =
 k=5    .   =   =   =   =
 k=6    .   =   =   =   =
```
⭐ **k = 3, 4, 5 and 6 probe the IDENTICAL 120 boxes** (same 8,640 rows); only the ORDER and the
corner LABELS differ. k=2's 60 boxes are disjoint from all of them.

**Why, and it is the harness's own design, not a defect:** a k=3 cell pins
(object_hue, shape, scale) and is TILED over floorwall and orientation; each tile pins all six
coordinates — which is exactly what a k=6 cell is. The harness's PREREG separates **cell** (what
is removed from the fit pool) from **corner** (the probe box, the estimand's unit) and says so;
the consequence for the LADDER had not been drawn.

⛔ **THREE CONSEQUENCES THE FIGURE AND THE PROSE MUST CARRY:**
1. For k >= 3 the x-axis is **NOT** "how many factors the probe composes" — every probe pins all
   six at every one of those rungs. It is **how coarsely the hold-out is carved**: 12 large
   `kway3-*` cells (300,000 rows) down to 120 singleton `kway6-*` cells (200,640 rows).
2. ⇒ **the five rival columns are CONSTANT by construction across k = 3..6** and cannot
   discriminate between those rungs. ⭐ MEASURED, and it is a free positive control that came
   out right: `rival_mlp` 35.2083 and `rival_lstm` 44.4333 at k=3 seed 0 are EQUAL to four
   decimals to their k=6 seed 0 values, because a rival's features are a function of the
   REQUEST alone and the corner multiset is the same. Per-corner arrays differ (max|diff| 56) —
   they are a permutation.
3. ⇒ across k = 3..6 **only the store arms can move**, and only through their fit pool. The
   k=2 -> k>=3 step is the only one that changes the probe itself.
⛔ Reported, not routed around. The harness is used exactly as shipped.

## RUNG k=3 COMPLETE — the ONE rung where the store nearly clears the bar
`LADDER.json:rungs.3`. 120 corners/seed, ceiling 72, all 3 seeds, `failed=[]`.
store **59.7500** [56.9500 .. 62.8750]  ·  `learned_1x` **55.5083**

| seed | paired delta (store − learned_1x) | CI95 | beats +1.0 |
|---|---|---|---|
| 0 | **+4.8167** | [+2.367, +7.317] | **YES** |
| 1 | **−0.0333** | [−1.967, +1.933] | no (a tie) |
| 2 | **+7.9417** | [+5.658, +10.342] | **YES** |

⇒ the store clears the bar on **2 of 3 seeds** at k=3 — its best showing anywhere — but the
pre-declared rule is EVERY seed, so k=3 does not clear either. ⛔ Reported as 2-of-3, never as
"the store wins at k=3".

⚠ **k=3 is the rung with the LARGEST hold-out: 300,000 rows (62.5 %)** against 200,640 at k=6
(`instrument/corner_set_equality.txt`). Since k=3..6 probe the SAME corners, the only thing that
differs is how much the fit pool lost — and the store does RELATIVELY best exactly where the
pool is most depleted. That is a real, mechanistic reading and it is the one place the store
design looks like it is buying something.

⚠⚠ **A data asymmetry that runs AGAINST the learner and therefore against this finding's
convenience, declared:** `learned_1x` is matched on BYTES and fitted on
`N_FIT_ROWS = 32,768` rows (AMENDMENT-1: deliberately "the same rows the store's content is
fitted from"), while the store's ADDRESS BOOK is fitted on the full `n_addr_train = 131,072`.
⇒ the 1x learner sees **4x fewer rows** than the store's read does, and still beats it at k=2
and k=6. ⛔ The asymmetry is in the store's favour and is printed rather than corrected.

## Ladder shape so far, ceiling-normalised (store arms)
| rung | seed0 | seed1 | seed2 | mean | deteriorates vs its own k=2? |
|---|---|---|---|---|---|
| k=2 | 0.7954 | 0.8792 | 0.8704 | 0.8483 | (base) |
| k=3 | 0.7910 | 0.8253 | 0.8733 | 0.8299 | **no consistent direction across seeds** (2 down, 1 up) |
| k=6 | 0.8707 | 0.8888 | 0.9303 | 0.8966 | **False** — it goes UP on all three seeds |
⇒ non-monotone, and ⛔ never a clean decline. k=4 and k=5 pending.

## ⭐⭐ THE BYTE LEDGERS, measured on all 15 fitted legs (`BYTES.json`, never netted)
| arm | atoms | sufficient statistics | address book | TOTAL | x 524,288 B |
|---|---|---|---|---|---|
| `single_design_K5` | 524,288 | 11.9–12.3 M | 158 k–175 k | **12,523,008 – 13,041,864 B** | **23.89x – 24.88x** |
| `pca_tensor` | 524,288 | **0** | 191 k–195 k | **715,192 – 719,352 B** | **~1.36x – 1.37x** |

(the 1.00x floor in the summary is leg `4_seed2`, still mid-flight, whose pca address book is
correctly flagged `p3_status: DECLARED NOT-RUN` — it is not a measured 1.00x.)

⭐⭐ **The two arms produce BIT-IDENTICAL composition scores at every rung measured so far, and
the single design costs ~18x more bytes to do it.** `pca_tensor` keeps NO sufficient statistics
— that zero is the price it does not pay, and it is also why it has no exact deletion. ⇒ the
single design's ~24.4x byte total buys **exact deletion and nothing else on the composition
rows**. This is the task's falsifier #2 in its strongest form and it is reported plainly.

## ⭐⭐⭐ WHAT THE LADDER ACTUALLY MEASURES FOR k >= 3 — the confound is the WHOLE signal
k=3,4,5,6 probe the IDENTICAL 120 corners (§ corner_set_equality). The ONLY thing that differs
across those four rungs is how many rows the cell removed from the fit pool. Correlating each
arm's across-seed mean against the rung's held-out row count, over those four rungs:

| arm | k=3 | k=4 | k=5 | k=6 | Pearson r vs held-out rows |
|---|---|---|---|---|---|
| held-out rows | 300,000 | 235,200 | 213,600 | 200,640 | — |
| `single_design_K5` | 59.750 | 62.708 | 64.719 | 64.556 | **−0.9841** |
| `pca_tensor` | 59.750 | 62.708 | (pending) | 64.556 | **−0.9992** |
| `learned_1x` | 55.508 | 67.992 | (pending) | 68.617 | **−0.9531** |
| `learned_ref_ownpool` | 11.794 | 13.281 | (pending) | 15.606 | **−0.9552** |
| `rival_mlp` | 42.175 | 42.175 | (pending) | 42.175 | **exactly constant** |
| `rival_lstm` | 43.503 | 43.503 | (pending) | 43.503 | **exactly constant** |
| `rival_table` | 9.528 | 9.528 | (pending) | 9.528 | **exactly constant** |
(the rivals' r is undefined — zero variance — which is the point, not a missing number.)

⭐⭐ **EVERY arm fitted on the rung's own pool — store AND learner alike — tracks the held-out
row count at r = −0.95 .. −0.999. EVERY arm not fitted on it is exactly constant.**
⇒ ⛔ **For k >= 3 this ladder is not a composition-depth axis at all: it is a training-data
axis.** The harness DECLARED this confound in advance ("the ladder confounds composition depth
with held-out fraction") and proved it could not be removed by a row budget
(`row_matching_feasibility.feasible = false`, required_floor 45,000 > ceiling 8,640). ⭐ This hub
now MEASURES that the confound accounts for essentially all of the variation (r^2 = 0.97 on the
store) — because with the corners held identical there is nothing else left for it to be.
⭐ The k=2 -> k>=3 step IS a real change of probe (36-row corners -> 72-row, disjoint sets).

## Bar-test guard added (my own gap, found on partial data)
`clears_bar` was computed over the seeds PRESENT, so a rung with 1 of 3 seeds scored read
`True`. AMENDMENT-2 section 3 says "on EVERY seed". Now a rung with fewer than the 3 declared
seeds returns `clears_bar: None` with `bar_test: INCOMPLETE -- n of 3 declared seeds ...`.
⛔ A partial pass can no longer read as a pass. (Caught at k=5 with one seed in.)

## DONE — 2026-09-21T16:54:33Z (`END_UTC.txt`)
**18 legs × 13 arms = 234 scored arm-runs. `failed=[]` on every leg. 0 `FAILED_*.json`.
0 NOT-RUN cells in the matrix.** Controls collapse on 12 of 12 (store arm × condition) cells.
Both ends reproduce on every arm and every seed.

Final artifacts regenerated from the complete data: `LADDER.json`, `LADDER.csv`, `LADDER.md`,
`BYTES.json` (18 legs), `fig_composition_ladder.png`, `FINDINGS.md`.
Figure re-rendered and EYEBALLED (dataviz step 7) three times: the caption was cut and pushed
clear of the legend, the figure narrowed, and the end-of-line labels staggered because
`single_design_K5` and `pca_tensor` coincide EXACTLY and were overprinting. A ⭐ glyph missing
from DejaVu Sans was replaced rather than left as a tofu box.

Bundle: `w12-composition-ladder.bundle` — `git bundle verify` = "okay ... complete history",
`list-heads` = **exactly one ref**, `c55e00a refs/heads/w12/composition-ladder`.
⛔ `--all` was never used.

Commits on the branch (7): 1eb82b1, b9b6f18, ea56c05, 56e46b2, 95305eb, cd94569, c55e00a.

### Process hygiene
All PIDs captured in `PIDS.txt`. The one restart (11:59Z) stopped ONLY this hub's processes,
identified by command line: three leg pythons (Windows PIDs 5176/28560/31104), then my lane
shells, then two legs the dying lanes had spawned. Verified afterwards: `lane shells left: 0`,
`my leg pythons left: 0`, and the three OTHER agents' jobs still running and untouched
(`exp_gen_bridge --seed 3`, `w12_val_sweep`, `w12_deletion_baselines --seed 1`).
⛔ No box-wide kill at any point.
