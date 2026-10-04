# w12 hub/single-design-crossover — LOG

## 2026-09-21T11:09:27Z START
- Wrote START_UTC.txt.
- Read task file /c/chlu/tasks/w12-hub-single-design-crossover.md.
- Gate check at 11:09Z:
  - /c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json : ABSENT (only START_UTC.txt present) -> SCORED leg gated.
  - /c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json   : ABSENT -> seed count decision deferred to scored-leg start.
- Val sweep is free; starting reconnaissance for it now.

## 11:10Z SELECTION RULE DECLARED BEFORE THE SWEEP
- /c/chlu/outputs/w12-single-design-crossover/SELECTION-RULE.md written at 11:10:24Z
  (timestamp in SELECTION-RULE.timestamp.txt). At that moment no val composition number
  for any K existed in this output dir.
- Rule: K* = max{ K in {5,8,10,12,16,20,24,28,32} : v(5) - mean_seeds v(K) <= 0.02 }.
- Crossover: K_cross = min{ K : v(5) - mean_seeds v(K) > 0.10 }.
- Env asserted: chlu.__file__ = C:\chlu\worktrees\CHLU-w12-single-design-crossover\chlu\__init__.py
  (INSIDE this worktree). numpy 2.4.1, scipy 1.17.0, jax 0.9.0, Python 3.11.9.
- Feature caches present for seeds 0,1,2 only (fw.features_seed_dir(3|4)/features.npy ABSENT)
  -> val sweep is 3 seeds, DECLARED.

## 11:11Z-11:16Z VAL SWEEP -- setup, two defects caught and fixed before any number
- Pipeline factored from the w8 single-design legs so the number is produced by the SAME
  code path: scripts/w8_build_matched_bytes.py + w8_read_matched_bytes.py + w8_score_corners.py
  -> scripts/w12_crossover_lib.py + scripts/w12_val_sweep.py (this worktree).
- TIMING REFERENCE, from w8's own logs (not re-derived):
  /c/chlu/outputs/w8-single-design/read_val_wardbalK5.log:45 "address book fitted in 867.2s"
  /c/chlu/outputs/w8-single-design/read_val_K40.log:45       "address book fitted in 945.7s"
  /c/chlu/outputs/w8-single-design/corners_val_wardbalK5.log "20/24 corners, 57s" -> ~70s per
  24-corner pass. So one (K, seed) row is ~15 min address fit + ~70s per scored pass.
- DEFECT 1, CAUGHT BY A CRASH, mine: seeds 1 and 2 under the AMBIENT features root
  (CHLU_GEN_BRIDGE_ROOT) hold 9,288 rows, not 480,000 -> IndexError at pool index 414,408.
  logs/sweep_s1.log, logs/sweep_s2.log (first attempt, 11:15:42Z and 11:15:45Z).
- DEFECT 2, THE SERIOUS ONE, caught while diagnosing defect 1: CHLU_GEN_FEATURES_ROOT is NOT
  set in this box's ambient environment. exp_gen_factor_wells.features_root() then falls back
  to CHLU_GEN_BRIDGE_ROOT = cache GENERATION 1, which w5-common 3.8 forbids for any number.
  My first seed-0 launch (11:15:59Z, bashpid 112879) was reading generation 1. It was KILLED
  before it produced a row and NO number from it exists on disk.
  FIX: scripts/w12_launch.sh exports CHLU_GEN_FEATURES_ROOT=C:/chlu/outputs/spoke-a-rivals/
  cache_generation2, and scripts/w12_val_sweep.py now calls
  exp_gen_delete_row7.assert_generation2(seed) per seed and writes the manifest onto every row.
  Verified: seeds 0/1/2 = generation 2, scope full, 480,000 rows. Seeds 3/4 = ABSENT.
- PROCESS DISCIPLINE: every launch appends to logs/PIDS.txt. Two of my own processes were
  killed by PID (112632 probe, 112879 generation-1 launch). NO box-wide kill was issued; the
  peer processes on this box (w12-cache-gen2-seeds-3-4's bridge for seed 3, w12-kway-ladder's
  pytest) were left running and are visible in the PowerShell process listing at 11:16Z.
- LAUNCHED 11:16:47Z, three processes, 3 threads each (the w8/w9 convention):
  sweep_s0 (seed 0, all 9 K, WITH the four charter-C controls)
  sweep_s1 (seed 1, all 9 K, --no-controls)
  sweep_s2 (seed 2, all 9 K, --no-controls)
  Controls are run at every K on seed 0 to draw the saturation panel; the SCORED gate re-runs
  them on the test condition at K*.
- 11:18:48Z RE-ORDERED, my own two secondary processes only: sweep_s1 and sweep_s2 were killed by
  PID after ~8 min and relaunched sweeping K DESCENDING (32 -> 5) while sweep_s0 keeps ascending
  (5 -> 32). Reason: the address-book fit is ~15 min per (K, seed) and its cost rises with K
  (span = sum_k r_k goes 320 at K=5 to ~2016 at K=32), so a single ascending order would leave the
  HIGH-K end -- where rank/width approaches 1 and the w9 saturation trap lives -- unmeasured if the
  wall clock ran out. Bracketing from both ends means any K left unrun is in the MIDDLE, and its
  seed count n is printed on every row.
  No other agent's process was touched; the peer bridge job for seed 3 was running throughout.

## 11:17Z-11:21Z SCORED-LEG AND DELETION HARNESSES BUILT WHILE THE SWEEP RUNS
- PREREG-w12-single-design-crossover-1.md written 11:17:57Z, BEFORE any test-condition
  number. It states the gate, the frozen K* rule, the control gate, the four-part ledger,
  the three deletion scopes, the STORE-SCOPED definition of "exact", and the three
  declared falsifications.
- scripts/w12_scored_leg.py: rows 3 (60 pair sub-corners) + 4 (49 n-way corners) on TEST,
  metric distinct_valid_novel_tuples per corner via fw._per_corner, verdict via
  fw.paired_estimand (the corner is the unit; never a single cell). It REFUSES to run:
  tested at 11:20Z, output
  "GATE CLOSED: C:\chlu\outputs\w12-deletion-baselines\BASELINES-READY.json does not exist.
   The scored leg is a DECLARED NOT-RUN with that reason. No single-design run has been spent."
  It also refuses if this hub's own row is absent from /c/chlu/SCORED-LEDGER.md.
- scripts/w12_deletion_rows.py: the three scopes (VALUE red; CONJUNCTION red AND cube;
  ITEM random 1 %), each with downdate -> re-eigendecompose -> carry_address_book
  (chlu/core/downdate_store.py:1297), STORE-SCOPED exactness against refit-without, and a
  cost line per arm (seconds | bytes rewritten | data rows read = 0 for us) beside the
  refit-without reference's own cost line. Plus the DISPLAY refusal row built on the
  coupling/hole table, whose index space is ASSERTED equal to gen_field's.
- Peer processes observed at 11:21Z and left alone: the cache spoke's bridge for seed 3
  (pid 21316, 875 CPU s) and the deletion-baselines spoke's fit (pid 3216). My three
  sweeps are pids 20864 / 31280 / 2780; memory 3.4 / 8.0 / 7.9 GB of 63 GB, 38 GB free.
- 11:21:46Z RE-PLANNED (my three processes only, killed by PID; peers untouched).
  The address fit is ~15 min per (K, seed) and its cost grows with span = sum_k r_k, which
  grows with K (320 floats at K=5, ~1,970 at K=32), so 27 (K, seed) cells is 4+ hours of
  wall clock on 3 processes. A COMPLETE 9-point curve at ONE seed WITH the charter-C
  controls is worth more than 2 K values at 3 seeds, so the K axis is now SPLIT across my
  three processes, all on seed 0, balanced by expected cost:
     A: K = 5, 32, 12   B: K = 8, 28, 16   C: K = 10, 24, 20
  K=5 (the rule's reference) and K=32 (rank/width 0.9844 -- the w9 saturation regime) are
  first in the queue. Seeds 1 and 2 are QUEUED BEHIND this and their coverage will be
  reported with its realised n per K; any K left unrun on a seed is a DECLARED NOT-RUN.
- MEASURED AT LAUNCH, and it is already the wave's central tell-tale -- rank/width at the
  two ends of the sweep, from the launch logs:
     K=5  wells 409..410  A=64  rank/width 0.1561..0.1565   (logs/sweep_s0_A.log)
     K=32 wells 64..64    A=63  rank/width 0.2656..0.9844   (logs/sweep_s1.log, seed 1)
                                             0.5781..0.9844 (logs/sweep_s2.log, seed 2)
  w9's maze arm was decorative at rank/d_k = 0.938. K=32 sits ABOVE that.

## 11:25Z ⛔ THE CONJUNCTION ROW AS SPECIFIED IS VACUOUS -- MEASURED, NOT ASSUMED
Task file section 4(b) asks: "delete every row carrying red AND cube". On this condition
that deletes ZERO rows, because `shape = cube` is HELD OUT of the value-holdout split
entirely -- the fit pool contains no cube at all.

MEASURED on the fit pool (decoded from the row index by the same odometer
scripts/w8_single_design_dev.py:red_mask uses), val / test:
    red total          11,024 / 11,044
    red AND cube            0 / 0        (cube is a held-out VALUE)
    red AND cylinder        0 / 0        (red-cylinder is the held-out PAIR cell)
    red AND sphere      5,507 / 5,511
    red AND oblong      5,517 / 5,533
Whole-pool shape counts: cube 0, cylinder 43,650, sphere 43,501, oblong 43,921.
Whole-pool object_hue counts: index 5 is 0 (the held-out hue); index 0 (red) is 11,024.

THE REPO ALREADY SAYS THIS in its own words, and I found it after measuring:
  chlu/experiments/exp_gen_delete_row7.py:325
    "red_cube": (0, 0, False, "shape=cube is held out entirely; reported, not a bar")
  chlu/experiments/exp_gen_delete_row7.py:26
    "scale, orientation and shape = cube are NOT deletion targets here"

WHAT I DO ABOUT IT -- both, never one:
  (b)  red AND cube        run as specified and reported with its count of 0 and this
                           reason. It is NEVER reported as a passing row.
  (b') red AND oblong      the DECLARED NON-VACUOUS SUBSTITUTE, scope `conjunction_live`.
                           It is the conjunction whose two marginals BOTH survive the
                           deletion (red survives via red-sphere, oblong survives via
                           green/blue-oblong), which is the only version of this row that
                           can test anything. exp_gen_delete_row7.py:320 names it
                           "red's other shape, SEEN -- the hole must leave it allowed".
The DISPLAY refusal row punches BOTH holes, 1 byte each, and says why each is there:
red-cube is refusable although no red cube was ever written, which is precisely the point
that refusal is a SEPARATE mechanism from forgetting.

CODE-PATH SMOKE (NOT A NUMBER: 8,192-row pool, rank 8, 2 sweeps) at 11:24Z confirmed the
downdate -> pinned_basis -> match_basis_width -> carry_address_book -> read path returns a
(32, 2048) feature block with 0 out-of-range, at K=5 and K=12, with
carry correspondence min singular value 0.6511 (K=5) and 0.2979 (K=12) on the red deletion
and 1.0000 on the vacuous one -- i.e. the carry correctly does nothing when nothing is
deleted. No basis-width adjustment was needed at either K.

## 11:25Z PLAN FOR THE REMAINING WALL CLOCK (written now so it is not chosen after seeing a number)
Round 1 (in flight, 11:21:46Z ->): seed 0, all 9 K, WITH the four charter-C controls.
  Expected ~66 min per process: 3 K x (~15 min address fit + ~6 min of 5 corner passes
  + ~1 min exactness), from the w8 timing reference.
Round 2: seeds 1 and 2 at the DECISION-RELEVANT K only -- K=5 (the rule's reference), K*,
  and the crossover K if it differs -- split across the three processes, so the selected
  point carries a real seed spread. Every other (K, seed) cell is a DECLARED NOT-RUN with
  "wall clock" as its reason and its n printed on the row.
Round 3: the deletion rows on VAL at K*, reusing the fitted arm.pkl from round 1 so no
  second 15-minute address fit is paid; and the SCORED leg only if
  /c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json has appeared by then.
The figure code path was exercised at 11:24Z against a SYNTHETIC CROSSOVER json and both
synthetic artefacts were deleted from this directory immediately afterwards.

## 11:26Z THE RIVAL SET THAT EXISTS TODAY IS THE 524,288 B REFERENCE COLUMN, NOT THE 1x COLUMN
Smoke-loaded every rival the task file names, through chlu/eval/rivals/w4_api.load_rival,
on the TEST condition, seed 0, corner pair-000/f0-hh (36 requests, 0 nan):
    mlp          OK  feats (36, 2048)  oor 0  param_count 117,878
    lstm         OK  feats (36, 2048)  oor 0  param_count 119,125
    transformer  OK  feats (36, 2048)  oor 0  param_count 119,858
    cont_mlp     OK  feats (36, 2048)  oor 0  param_count 119,525
    table        OK  feats (36, 2048)  oor 0  param_count None (the FLOOR; a table has no
                                              parameter count and is printed as a floor)
  (checkpoints at /c/chlu/outputs/spoke-a-rivals/rivals/seed_{0,1,2}/)
These are matched to the RETIRED 524,288 B pin: 524,288 / 4 = 131,072 float32 parameters.
⛔ They are therefore the REFERENCE column, not the 1x column. The 1x column is matched to
THIS object's own four-part total, which is several times larger, and it requires a REFIT
at that larger parameter count. That refit is spoke/deletion-baselines' deliverable and is
named in BASELINES-READY.json. If that file never names 1x-matched entry points, the 1x
column is a DECLARED NOT-RUN and only the 524,288 B reference column can be printed --
which is a comparison in the rivals' FAVOUR and is stated as such.

## 11:27Z A PROPERTY OF MY OWN CONTROL, MEASURED AND DECLARED RATHER THAN QUIETLY FIXED
Unit-tested the four charter-C perturbations at UNEQUAL well widths (410/410/409/409/410,
the K=5 ward_balanced shape) before any control number existed:
    zero            shapes preserved, Frobenius 0 on every well
    value_permute   shapes preserved, Frobenius EXACTLY preserved per well
    well_permute    shapes preserved, Frobenius preserved (block moved between wells)
    random_span     shapes preserved, Frobenius EXACTLY preserved per well
  (original per-well Frobenius 161.401 / 160.679 / 161.969 / 161.207 / 161.969)
FOUND, AGAINST MYSELF: `well_permute` draws a uniform permutation with only the IDENTITY
rejected -- not a derangement -- so a well can keep its own block. At K=5 seed 0 one of
the five wells did. A uniform permutation has ONE fixed point in expectation at any K.
⛔ I did NOT change it mid-sweep. The three seed-0 processes had already imported the
module, so a fix would have applied to some rows and not others and the wave would carry
two different controls under one name. The definition is held fixed and the property is
printed instead. It makes that control slightly WEAKER than a derangement -- the
CONSERVATIVE direction for a control that is required to collapse -- and the PRIMARY C.2
control is `value_permute`, which permutes atom VALUES inside each well and has no such
property. The verdict is taken from `value_permute` and `random_atoms`.

## 11:27Z TWO THINGS READ OFF THE GATE SPOKES' OWN LOGS (they are my gate, so their interface is operationally necessary)
1. ⭐ INDEPENDENT AGREEMENT ON THE VACUOUS CONJUNCTION. spoke/deletion-baselines reached the
   same finding from its own direction and wrote it at 11:44Z in
   /c/chlu/outputs/w12-deletion-baselines/LOG.md:
     "object_hue idx 0 ('red') = 11,044 rows; shape idx 0 ('cube') = 0 rows -- the cube
      class is the value-holdout condition's own held-out shape. So 'red AND cube' would
      delete ZERO rows. Conjunction pin moved to object_hue 0.0 AND shape 2.0 = 5,511 rows,
      declared in PINS_W12; partition_fit_rows REFUSES any no-op deletion rather than
      printing a zero."
   Its red count (11,044) equals mine on the test pool exactly.
   ⇒ I ADOPT ITS PIN: my `conjunction_live` scope was red AND oblong (shape 3) and is now
   red AND SPHERE (shape 2), so the comparator and my arm delete the SAME rows. A
   comparator and an arm that delete different rows are not a comparison.
2. ⛔ THE 1x COLUMN IS IN DOUBT, AND THE DIRECTION OF THE ERROR IS AGAINST MY OWN INTEREST.
   The spoke's comparators are matched to the 524,288 B pin -- its 11:44Z entry gives the
   measured counts: mlp 130,514 (-0.43%), cont_mlp 131,891 (+0.63%), transformer 130,433
   (-0.49%), lstm 130,721 (-0.27%), refuse-token transformer 130,479 (-0.45%), against
   131,072 params. My object's four-part total is SEVERAL TIMES 524,288 B, so a 1x-matched
   rival would have SEVERAL TIMES those parameters and would be STRONGER, not weaker.
   ⇒ Comparing against the 131,072-param set FLATTERS ME. ⛔ It therefore cannot be
   reported as a 1x result under any outcome. It is the 524,288 B REFERENCE column, it is
   labelled as such, and if no 1x-matched entry point is named in BASELINES-READY.json the
   1x column is a DECLARED NOT-RUN.
3. 5 SEEDS: the cache spoke's own projection is "~1 h 50 m per seed" for seeds 3 and 4
   (/c/chlu/outputs/w12-cache-gen2-seeds-3-4/LOG.md), so CACHE-READY.json will not exist
   when my scored leg would start. ⇒ 3 SEEDS, DECLARED, per the task file's clause.

## 11:28Z INDEPENDENT SYNTHETIC CROSS-CHECK OF THE DOWNDATE (single-threaded, no feature cache)
WellStats.add / remove / atoms exercised in isolation on synthetic Gaussian data at the
EXACT well widths the sweep uses, 4,000 rows, 400 deleted, A = min(64, d-1):
    d=409  max|atom dev| 0.000e+00   subspace projector dist 0.000e+00   ranks 63/63
    d=256  0.000e+00  0.000e+00  63/63
    d=170  0.000e+00  0.000e+00  63/63
    d=128  7.451e-09  5.094e-09  63/63     <- the only non-zero cell, at float32 atom scale
    d=102  0.000e+00  0.000e+00  63/63
    d= 85  0.000e+00  0.000e+00  63/63
    d= 73  0.000e+00  0.000e+00  63/63
    d= 64  0.000e+00  0.000e+00  62/62     <- A capped at 63 by the strict-below-width rule
⭐ This is STORE-SCOPED exactness measured without the encoder, the decoder, the judge or
the feature cache anywhere in the path, so a failure here would have been a defect in the
OBJECT rather than in my harness. It is a cross-check, NOT the wave's exactness row -- the
sweep measures that on the real features and reports it per K per seed.

## 11:29Z SECOND SYNTHETIC CROSS-CHECK: DOES THE ADDRESS-BOOK CARRY ACTUALLY DO ANYTHING?
Synthetic factored data (D=240, K=6, N=6,000, 3 factors of cardinality 4/5/3), no feature
cache, no decoder, no judge. Delete every row whose factor-0 value is 0, then compare the
rendered answer against a book REFIT from scratch on the post-deletion store:
    signal scale (mean |refit render|)        1.4442
    NAIVE   (book NOT carried) vs refit       1.5084   = 104 % of signal
    CARRIED (book carried)     vs refit       0.5221   =  36 % of signal
    carry min singular value 0.0583 · address bytes changed 3,072 · DATA ROWS TOUCHED 0
⚠ REPORTED HONESTLY ABOUT WHAT THIS IS AND IS NOT: the comparator here is an
INDEPENDENTLY REFITTED ALS book, which is a different solution of a non-convex fit, so the
residual 0.5221 is dominated by ALS solution difference and is NOT an exactness number.
It shows the DIRECTION only -- the carry removes about two thirds of the discrepancy that
ignoring the basis change produces, at 0 data rows read. w8's own carry measurement
(rendered downdate-vs-refit agreement 3.6e-07 / 9.5e-07 / 9.5e-07 against a signal scale
of 3.2 / 3.8 / 4.3, PREREG-w8-single-design-1.md Amendment 1 A1.1) is the exactness
statement and is NOT restated as mine.

## 11:29Z FIRST REAL LEDGER ROW LANDS -- K=5, seed 0, val
From logs/sweep_s0_A.log:
    TOTAL 4,118,034 B = 7.8545x the 524,288 B REFERENCE
  parts 1+2+4 from LEDGER-ACROSS-K.json are 3,922,450 B, so the fitted address book is
  4,118,034 - 3,922,450 = 195,584 B.
⭐ That is EXACTLY w8's measured address-book size at ward_balanced K=5, A=64:
  /c/chlu/outputs/w8-single-design/read_val_wardbalK5.log:45
  "address book fitted in 867.2s; bytes 195,584".
  Two waves, two harnesses, the same object, the same byte count to the byte.

## 11:29Z FIRST COMPOSITION NUMBER, AND IT REPRODUCES WAVE 8
logs/sweep_s0_A.log:
    address book fitted 378.4s  195,584 B  (90/10 split, 117,964 train)
    TOTAL 4,118,034 B = 7.8545x the 524,288 B REFERENCE
    cov_tensor K=5   valid 0.8918 +- 0.1007  [0.7222, 1.0000]  over 24 val corners  (35s)
CROSS-CHECK AGAINST WAVE 8, same object, same partition criterion, same granularity,
different harness and a different wave:
    /c/chlu/outputs/w8-single-design/read/val_K5_A64_seed0/corners_val.json
        valid_mean 0.8935185185185185, sd 0.0973, min 0.7361, max 1.0000, 24 corners
    /c/chlu/outputs/w8-single-design/read_val_wardbalK5.log:45
        "address book fitted in 867.2s; bytes 195,584"
⭐ Address book bytes agree TO THE BYTE (195,584). Composition agrees to 0.0017
(0.8918 vs 0.8935); w8's 0.8935 is its 100%-pool fit and mine is the 90/10 protocol, which
is the difference the w8 hub itself flagged at scripts/w8_read_matched_bytes.py (its
"split9010" leg exists for exactly this reason).
⚠ The address fit took 378.4s here against w8's 867.2s. Same pins (rank 128, ridge 1e-2,
40 sweeps, 131,072-row pool). The difference is machine load, not protocol; both are on
this box and neither number is a claim about the method.

## 11:33Z-11:39Z THE FIRST THREE SWEEP ROWS, AND THE DELETION LEGS ON VAL AT K=5
All from logs/sweep_s0_{A,B,C}.log and val_sweep/val_K*_seed0/row.json.

| K | four-part TOTAL B | xRef | val comp (24 corners) | removed | shuffled | shuffled_across | random_atoms | exactness max abs dev | downdate s | refit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 4,118,034 | 7.8545 | 0.8918 +- 0.1007 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.455e-11 | 1.55 | 12.5 |
| 8 | 2,959,602 | 5.6450 | 0.8837 +- 0.0949 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.684e-14 | 1.04 |  9.6 |
| 10| 2,606,754 | 4.9720 | 0.8825 +- 0.1021 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.910e-11 | 0.79 |  8.3 |
Address books: 195,584 B (K=5) · 295,424 B (K=8) · 361,984 B (K=10) -- part 3 GROWS with K
while part 2 falls, which is why the total falls more slowly than the covariance does.
⭐ ALL FOUR CONTROLS COLLAPSE TO EXACTLY 0.0000 at K=5, 8 and 10. Ranks agree on every
exactness check. drop from K=5: K=8 = 0.0081, K=10 = 0.0093 -- both inside the declared
0.02 band, so K* >= 10 on the evidence so far.

DELETION LEGS, VAL, seed 0, K=5, reusing val_sweep/val_K5_seed0/arm.pkl (no second address
fit paid). deletion/val_seed0_K5/deletion.json:

| scope | rows deleted | OUR adaptation | bytes rewritten | data rows read | refit-without | novel/corner before -> after | paired | store-scoped exactness |
|---|---|---|---|---|---|---|---|---|
| VALUE red | 11,024 | 1.4525 s | 4,068,192 | **0** | 13.2 s, 120,048 rows | 64.208 -> 64.333 | **+0.1250** | 1.455e-11 |
| CONJUNCTION red AND cube | **0 (VACUOUS)** | 0.3225 s | 4,068,192 | **0** | 14.7 s, 131,072 rows | 64.208 -> 64.208 | +0.0000 | 0.000e+00 |
| CONJUNCTION red AND sphere (substitute) | 5,507 | 0.7211 s | 4,068,192 | **0** | 12.5 s, 125,565 rows | 64.208 -> 64.125 | **-0.0833** | 2.842e-14 |
| ITEM random 1 % | 1,311 | 0.2781 s | 4,068,192 | **0** | 13.7 s, 129,761 rows | 64.208 -> 64.208 | +0.0000 | 1.455e-11 |

⭐ Every paired delta is inside BAR 2's |delta| <= 1.0, by two orders of magnitude.
⭐ The vacuous scope reads EXACTLY 0.000e+00 and +0.0000 -- the right answer for deleting
nothing, and a check that the pipeline is not manufacturing motion.
⛔ REPORTED AGAINST MYSELF -- A LIMITATION OF THE VAL CONDITION, not of the mechanism:
n_deleted_corners = 0 for EVERY scope. The 24 val corners are all n-way corners and none
of them requests red, so the val leg exercises only the SURVIVOR side of the row. The
DELETED side (does the deleted content actually stop being produced?) needs the TEST
corners, which include the pair sub-corners -- and that is a SCORED leg. The val numbers
above are therefore evidence that composition SURVIVES a deletion, and are NOT evidence
that the deletion removed anything at the corner level. The store-scoped exactness numbers
are what say the deletion removed something.
DISPLAY REFUSAL (reported, never claimed): coupling/hole table, layout ASSERTED equal to
gen_field's (layout_equals_gen_field True, 1,322 pair slots, 60 indices checked). Two holes
punched -- red AND cube, and red AND sphere -- at 1 byte each, 2 bytes total, 0 data rows
read, no refit. The comparator (the refuse-token transformer) is a DECLARED NOT-RUN here:
its gate file does not exist yet.

## 12:16Z ⛔⛔ A BUG OF MINE, CAUGHT BY A CRASH, NOT BY A LOOK
Processes sweep_s0_A (K=32) and sweep_s0_B (K=28) DIED at the `landscape_shuffled`
control with
  ValueError: matmul: ... (size 63 is different from 62)   logs/sweep_s0_A.log
  ValueError: matmul: ... (size 64 is different from 63)   logs/sweep_s0_B.log
inside chlu/experiments/exp_gen_read.py:734 -> addr.read(store, values, grid.values, basis).

CAUSE, diagnosed: my `qr_basis` returned `min(d, 1 + deployed_rank)` columns, while
`pinned_basis` returns `deployed_rank` columns -- or `deployed_rank + 1` when the mean's
perpendicular component clears its 1e-4 relative admission threshold. Those two counts
agree for most wells and DISAGREE exactly where the mean lies numerically inside the
span, which is common in the NARROW wells of a fine partition. The address book emits one
coefficient per column of the basis it was fitted against, so a control basis of a
different width is not a weak control -- it is a shape error.

FIX: `qr_basis` now REQUIRES `widths=[b.shape[1] for b in pinned_basis(store)]` and
returns exactly those widths, slicing a COMPLETE QR so extra orthonormal columns exist
when the perturbed block spans fewer directions than the book expects. Those extra
columns carry no content, which is the correct thing for a control whose content is meant
to be destroyed. Unit-tested at the two failing shapes and two more:
  A=63 r=62 d=64 w=62/63 and A=64 r=63 d=74 w=64/63, all three perturbations,
  columns exactly as asked, max |Q^T Q - I| 8.88e-16 .. 1.22e-15.

⭐ THE ALREADY-WRITTEN ROWS ARE NOT AFFECTED AND I CHECKED RATHER THAN ASSUMED.
K=5, 8, 10, 20 and 24 completed, which means their widths AGREED; and where the width
agrees, the leading columns of a complete QR ARE the reduced QR, so the basis those
controls used is bit-for-bit the basis the fixed code produces. They are not re-run.
⚠ LOST TO THE CRASH and being re-run now: K=32 and K=28 (their arm numbers were printed
before the crash but no row.json was written), and K=12 and K=16, which were queued behind
them and never started.
RELAUNCHED 12:16:51Z: A2 = K 32 then 12 · B2 = K 28 · C2 = K 16. Three processes, my own
PIDs in logs/PIDS.txt, no peer process touched.

## 12:28Z SEED COUNT, RE-CHECKED AGAINST CACHE-READY.json UNDER THE STATED ADMISSION RULE
CACHE-READY.json appeared at 12:27:01Z (produced_utc), 13,666 B, schema
chlu-w12-cache-ready/1, generation2_root
C:\chlu\outputs\w12-cache-gen2-seeds-3-4\cache_generation2\shapes3d_recon.
Admission rule applied as stated: a seed is usable only if its entry has a NON-NULL
array_shape and 59 blocks.
  seed 3: array_shape [480000, 128, 4, 4] · n_blocks 59 · 59 entries in block_sha256 ·
          scope "full" · n_rows 480000 · generation 2 · shape_dtype_matches_seeds_0_2 True
          · rows_sha256 dbddda1cf6d5a73f... (the same row vector as seeds 0-2)
          ⇒ PASSES the admission rule.
  seed 4: array_shape NULL · n_blocks 0 · block_sha256 {} · scope null · n_rows null ·
          status "NOT_RUN" · reason "generation-2 cache for this seed was not completed"
          ⇒ FAILS the admission rule. ⛔ Not used, and not counted as a zero.

⛔⛔ BUT SEED 3 STILL CANNOT ENTER THIS HUB'S INSTRUMENT, AND THE REASON IS THE JUDGE, NOT
THE CACHE. Every composition number here is produced by fw.Judge(seed), which loads the
HASH-ASSERTED external classifier from fw.classifier_dir(seed). Measured just now:
    classifier_dir(0) exists True   .../gen-external-classifier/seed_0
    classifier_dir(1) exists True   .../gen-external-classifier/seed_1
    classifier_dir(2) exists True   .../gen-external-classifier/seed_2
    classifier_dir(3) exists FALSE  .../gen-external-classifier/seed_3
    classifier_dir(4) exists FALSE  .../gen-external-classifier/seed_4
Seed 3's own entry also reads status "CACHE_READY_BRIDGE_PENDING" -- its bridge sidecars
are not placed yet either.
⇒ THE SEED COUNT USED BY THIS HUB IS 3 (seeds 0, 1, 2), DECLARED, for two separate
measured reasons: seed 4 fails the cache admission rule, and seed 3 passes it but has no
external classifier, so no corner of it can be judged on the instrument every other number
in this wave was measured on. ⛔ 3-seed and 4-seed rows are never mixed, and the count is
printed on every row.

## 12:42Z ⭐⭐ THE HEAD'S SPHERE CAVEAT, MEASURED -- AND IT FIRES IN THE OPPOSITE DIRECTION
HEAD RULING: the conjunction pin STAYS object_hue = 0.0 AND shape = 2.0 -- that IS the
RED SPHERE, named "red sphere" everywhere from here -- and a caveat column is printed:
chlu/data/gen_value_holdout.py:299 rejected sphere as a HELD-OUT value in these words:
  "the shape hazard of section 3.1, exactly: a degraded render is a rounded blob, so the
   SPHERE is the shape the model degrades TOWARDS. A held-out sphere would be unremovable
   for rendering reasons, not store reasons."
THE THREAT TO THIS ROW: an arm DAMAGED by the red-sphere deletion could be judged "sphere"
for rendering reasons and score spuriously well on exactly the deleted conjunction.

⭐ MEASURED, not stated. logs/deletion_val_K5_s0_v2.log, 24 val corners, judged-shape
histogram read back from the hash-asserted external classifier's own tuples, accepted
requests only:
    arm                     cube     cylinder   sphere    oblong    sphere excess
    cov_tensor (intact)     0.0000   0.3333     0.3333    0.3333    +0.0000
    landscape_removed       1.0000   0.0000     0.0000    0.0000    -0.3333
    random_atoms            0.9196   0.0579     0.0023    0.0203    -0.3310
  (sphere excess = frac(judged sphere) - frac(REQUESTED sphere) on the same requests)

⛔ THE HAZARD DOES NOT REPRODUCE ON THIS INSTRUMENT, AND THE TWO MAXIMALLY DEGRADED ARMS
BOUND IT: a degraded render on this decoder + this external classifier is judged CUBE,
not sphere -- 1.0000 and 0.9196 cube against 0.0000 and 0.0023 sphere. Sphere excess is
NEGATIVE for both. So a degrading baseline CANNOT score spuriously as sphere after the
red-sphere deletion on this pipeline; it collapses onto the one shape that is held out of
the split entirely and is therefore judged invalid everywhere.
⭐ THIS ALSO EXPLAINS THE CONTROL COLLAPSE MECHANISTICALLY, which no previous wave had:
every charter-C control reads EXACTLY 0.0000 because its render is judged `cube`, and cube
is the held-out shape, so judge_valid is false on every request. The controls do not
collapse because of a threshold; they collapse because the degraded render lands on a
shape the condition never contains.
⚠ SCOPE OF THIS MEASUREMENT: seed 0, val corners, K=5, this decoder and this classifier.
It is not a claim about Shapes3D decoders in general, and gen_value_holdout's argument for
NOT holding sphere out stands on its own terms -- what is measured here is that the
specific spurious-scoring route it warns about is not open on this instrument.

## 13:35Z THE SWEEP CLOSES: K* = 32, NO CROSSOVER, AND THE SEED SPREAD AGREES PER SEED
CROSSOVER-val.json, rule frozen 11:10:24Z, applied literally over the K actually run.
    K       5      8      10     12     16     20     24     28     32
    n       3      1      1      1      1      1      1      1      3
    comp  .8953  .8837  .8825  .8929  .8906  .8877  .8981  .8976  .9290
    sd    .0065    -      -      -      -      -      -      -    .0259
    drop  .0000  .0116  .0127  .0023  .0046  .0075 -.0029 -.0023 -.0338
  satisfying set = every K ⇒ K* = 32. Largest drop anywhere 0.0127, against a 0.10
  crossover threshold ⇒ NO CROSSOVER inside the swept range, and the range is NOT extended.
PER SEED, and this is the stronger statement because it is PAIRED:
    K=5   seed0 0.8918  seed1 0.8912  seed2 0.9028
    K=32  seed0 0.9016  seed1 0.9323  seed2 0.9531
  EVERY seed scores HIGHER at K=32 than at K=5. The common-seed robustness check (printed
  beside the rule, never instead of it) gives drop_on_common_seeds = -0.0338 at K=32 over
  seeds {0,1,2} -- the same conclusion as the rule, computed within seed.
CONTROL GATE at K*=32, val, 3 seeds: landscape_removed, landscape_shuffled,
landscape_shuffled_across_wells and random_atoms ALL exactly 0.0000 against the arm's
0.9290 -- a drop of +0.9290 on all four, at rank/width up to 0.9844.
⚠ n is 3 only at K=5 and K=32; the seven K in between are n=1 and the table prints n on
every row. Those cells are a DECLARED NOT-RUN at seeds 1-2 with "wall clock" as the reason.
Figure regenerated: fig_crossover_val.png.

## 14:00Z ⭐⭐ SCORED RUN 1/3 SPENT AND REPORTED -- single-design, seed 0, TEST, K* = 32
Ledger row appended to /c/chlu/SCORED-LEDGER.md at 2026-09-21T12:53:06Z, BEFORE the run;
scripts/w12_scored_leg.py refuses to start without it. Artefact:
scored/run1_seed0_K32/scored.json. 60 pair sub-corners (row 3) + 49 n-way corners (row 4).
FOUR-PART LEDGER ON TEST, summed and never netted:
  atoms 516,096 | covariance+count 560,608 | address book 1,060,864 | coupling/hole 6,610
  = 2,144,178 B = 4.0897x the 524,288 B REFERENCE column. rank/width max 0.9688.

| arm | row 3 (60 pair) | row 4 (49 n-way) | judge acc | paired delta vs us | CI95 | corners we win |
|---|---|---|---|---|---|---|
| cov_tensor (THE ARM) | 31.600 | 66.959 | 0.9012 | -- | -- | -- |
| landscape_removed | 0.000 | 0.000 | 0.0000 | +47.4954 | [+44.101, +51.028] | 109/109 |
| landscape_shuffled | 0.000 | 0.000 | 0.0000 | +47.4954 | [+44.101, +51.028] | 109/109 |
| landscape_shuffled_across_wells | 0.000 | 0.000 | 0.0000 | +47.4954 | [+44.101, +51.028] | 109/109 |
| random_atoms | 0.000 | 0.000 | 0.0000 | +47.4954 | [+44.101, +51.028] | 109/109 |
| mlp__as_trained | 2.933 | 7.959 | 0.0945 | +42.3028 | [+39.165, +45.532] | 109/109 |
| lstm__as_trained | 2.500 | 7.531 | 0.0852 | +42.7339 | [+39.679, +45.899] | 109/109 |
| transformer__as_trained | 0.667 | 1.918 | 0.0222 | +46.2661 | [+42.899, +49.725] | 109/109 |
| cont_mlp__as_trained | 0.000 | 0.061 | 0.0004 | +47.4679 | [+44.073, +50.991] | 109/109 |
| table (FLOOR, never a rival) | 5.833 | 11.163 | 0.1589 | +39.2661 | [+35.862, +42.679] | 109/109 |

⭐ THE CONTROL GATE PASSES ON THE SCORED ROW: all four controls are EXACTLY 0.000 on both
rows at rank/width 0.9688. The row may be claimed.
⭐ Charter C.1: the arm beats the prototype TABLE FLOOR by +39.2661 novel tuples per corner
with a CI95 lower bound of +35.862 against a declared margin of +1.0.
⛔⛔ AND THE COMPARISON IS NOT A 1x COMPARISON, AND THE ERROR RUNS AGAINST ME. Every
learned comparator here is matched to param_target 131,072 (= 524,288 B / 4), declared in
BASELINES-READY.json's own pins. My object's four-part total is 2,144,178 B, so a genuinely
1x-matched rival would carry ~535,784 float32 parameters -- about 4.1x these -- and would
be STRONGER. ⛔ The 1x column is a DECLARED NOT-RUN and NO row above is reported as a 1x
result. What is reported is the 524,288 B REFERENCE column, labelled as such.
⚠ Also carried: cont_mlp scores 0.000/0.061, i.e. at the floor. That is a number about the
rival's fit on this population, not a demonstration about the arm, and it is printed rather
than leaned on.
14:00Z LAUNCHED the deletion rows on TEST at K*=32, reusing scored/run1_seed0_K32/arm.pkl
so no second 1,242 s address fit is paid. Scope row counts on the test pool: value 11,044
(8.43 %), conjunction red-AND-cube 0 (0.00 %, VACUOUS), conjunction_live RED SPHERE 5,511
(4.20 %), item 1,311 (1.00 %).

## 14:27Z ⛔⛔ THE DELETION READ-PROBE -- THE WAVE'S SHARPEST AND LEAST COMFORTABLE NUMBER
The TEST deletion row exposed something the val leg could not: on TEST the VALUE scope has
20 corners that REQUEST red, so the DELETED side is exercised for the first time.
  deletion/test_seed0_K32/deletion.json
    survivors (89 corners) 50.921 -> 50.921   paired +0.0000
    DELETED   (20 corners) 32.25  -> 32.25    judge accuracy 0.8958 -> 0.8958
⇒ Deleting all 11,044 rows carrying red changes the red-requesting corners by EXACTLY ZERO.

So I measured what the deletion moved AT THE READ rather than guessing.
scripts/w12_deletion_probe.py (no judge, no decoder), deletion/probe_test_seed0_K32.json:
  group                        n      rel RMS change CARRIED   UNCARRIED   max|delta|
  corners REQUESTING red       720    0.000000                 0.610083    4.7684e-07
  corners NOT requesting red   4968   0.000000                 0.614277    4.7684e-07

WHAT THIS ESTABLISHES, WITHOUT SPIN:
 1. Store-scoped exactness is REAL -- downdate == refit-without to 5.960e-08, and the basis
    genuinely moved: the UNCARRIED read changes by 0.61 relative RMS.
 2. The carry is EXACT at the read -- carried, the answer is unchanged to float32 epsilon.
 3. ⛔ The basis change is GENERIC, not targeted: 0.610 on red corners against 0.614 on the
    others. The deletion moved the span no more for red than for anything else.
 4. ⛔⛔ At K* = 32 the per-well CONTENT carries no VALUE-SPECIFIC information. rank/width
    is 0.9531-0.9688, so the well's span is nearly the whole well and removing 8.4 % of rows
    does not move it. Value identity lives in the ADDRESS BOOK, which the deletion carries
    but does NOT refit.
⚠ This does not overturn the exactness claim and does not rescue it. It BOUNDS it.
"Store-scoped" now has a third thing that is not refit -- the address book -- and on this
object at this K, "the store has forgotten red" and "the object no longer answers red" are
DIFFERENT STATEMENTS, of which only the first is demonstrated.
⚠ CONTRAST WITH WAVE 8, which is the useful part: w8 measured the same red deletion at K=5,
A=64 (rank/width 0.156) and found the basis change CONCENTRATED -- cos_min 0.1999 with
93.8 % of directions above 0.99, against cos_min 0.9976 for a size-matched random deletion
(PREREG-w8-single-design-1.md Amendment 1 A1.3). ⭐ THE TARGETING OF A DELETION IS A
FUNCTION OF rank/width, AND IT DISAPPEARS EXACTLY WHERE THE BYTE LEDGER IS CHEAPEST.
That is the real trade-off this wave found, and it is not the one it went looking for.
⭐⭐ It also makes the refusal row load-bearing rather than rhetorical: since the downdate
does not stop the object answering red, THE HOLE IS THE ONLY MECHANISM HERE THAT CAN.

## 14:30Z CLOSE-OUT
- No process of mine is running. PowerShell listing at 14:27Z shows five python processes,
  all belonging to OTHER agents (w12-cache-gen2-seeds-3-4's bridge for seed 4; four
  w12-composition-ladder legs). ⛔ No box-wide kill was ever issued in this wave; every
  process I killed was one of my own, by PID, and each is recorded in logs/PIDS.txt with
  the reason.
- BUDGET AFTER THIS WAVE: `single-design` 1 of 3 SPENT (run 1/3, ledger row 12:53:06Z),
  2 REMAIN. `generation-table` run 3 UNSPENT -- a DECLARED NOT-RUN, reason in FINDINGS.md
  section 11.
