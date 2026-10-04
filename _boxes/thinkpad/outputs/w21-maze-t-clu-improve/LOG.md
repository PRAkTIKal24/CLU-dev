# w21 hub/maze-t-clu-improve — LOG

START_UTC 2026-09-23T22:43:05Z. Box FORGIS_14 (shared). Branch w21/maze-t-clu-improve @ 0455961 (w19 tip).

## Setup
- env: `source env.sh` (sources /c/chlu/env_w3.sh, re-exports PYTHONPATH with ';' Windows paths, asserts chlu.__file__ inside this worktree).
  Asserted: chlu.__file__ = C:\chlu\worktrees\CHLU-w21-maze-t-clu-improve\chlu\__init__.py.
- Free memory at start 14.5 GB of 63.4 GB (peers running). Our PIDs appended to PIDS.txt.
- D1: VAL search runs on a CACHE of the frozen instrument (scripts/w21_cache.py): W16.banks / make_holdout / split_rows / cap_split / frozen codec recipe (latent 128, 2,500 steps), exactly the w19 path. Plus the w9 dyn scalars (w9feat.dyn, 31 wide) per row. CLU candidates are closed form -> scripts/w21_val_sweep.py fits and scores each on the SAME train rows/codec, judge W16.sc 'exact'. Rivals are NOT involved in the VAL gate (the gate is CLU-vs-A1).
- ⭐ Instrument check: A1-repro on the cache = 0.8019 / 0.5689 / 0.8034 = w19 VAL A1 to every printed digit.

## Gate (strict, never relaxed)
KEPT iff VAL n-way mean > 0.725 AND every seed >= A1 (0.8019 / 0.5689 / 0.8034 exact).

## Round 1 (val/sweep_r1.json, logs/sweep_r1.log) — Advisor's ranked starting points
| candidate | what changed | VAL n-way (s0/s1/s2) | mean | pair | floats | gate |
|---|---|---|---|---|---|---|
| A1-repro | nothing | 0.8019/0.5689/0.8034 | 0.7247 | 0.4056 | 4584 | = A1 (reference) |
| pairscale-0.5 | pair block x0.5 (= ridge 4 on pairs) | 0.8019/0.5689/0.8034 | 0.7247 | 0.4329 | 4584 | FAIL (not > mean) |
| pairscale-0.25 | pair x0.25 (= ridge 16 on pairs) | 0.8019/0.5689/0.8034 | 0.7247 | 0.4391 | 4584 | FAIL |
| pairscale-2 | pair x2 (lighter) | 0.8019/0.5689/0.8034 | 0.7247 | 0.3911 | 4584 | FAIL |
| ridge-0.1 | ridge 0.1 | 0.8019/0.5689/0.8034 | 0.7247 | 0.4048 | 4584 | FAIL |
| ridge-10 | ridge 10 | 0.8019/0.5689/0.8034 | 0.7247 | 0.4393 | 4584 | FAIL |
| tri-all | + all 10 triple blocks | 0.8019/0.4785/0.8034 | 0.6946 | 0.3546 | 23184 | FAIL |
| tri-PDT | + pocket x door x tint triple | 0.8019/0.5689/0.8034 | 0.7247 | 0.4447 | 6684 | FAIL |
| dyn-main | + 31 dyn scalars (standardised) | 0.8019/0.5689/0.8034 | 0.7247 | 0.4457 | 5018 | FAIL |

Finding: shrinkage DOES help the unseen 2-way row (0.406 -> 0.439, Advisor's point 2 confirmed on 'pair'), but n-way is invariant to all of it.

## Diagnostic (scripts/w21_diag.py, logs/diag_nway.log)
n-way errors are all-or-nothing per held-out (pocket, door, tint) tuple:
- RULE-B tuples (c,0,c) / (c,open,c): true = illegal; CLU says legal or wrong displacement. The (pocket=c, tint=c) pair is absent from train for that colour (train_triple_rows = 0), so a colour-IDENTITY design (one-hots, pairs, triples) has nothing to read; the rule 'pocket == tint' is only learnable colour-generically.
- INTERACTION tuples (c,c,c): true = legal + consume; CLU says illegal/wrong.
- Everything decided by colour identity already seen is 1.000.
=> Shrinkage/ridge/triples/dyn cannot move n-way. The OPEN item that can is COLOUR CODING.

## D2: colour-relational coding (OPEN list: "colour coding")
The four colour-bearing words (pocket, key, door [closed colour 1..k], tint) share ONE colour alphabet 1..k. rel block = eq(x,y) for the 6 pairs of colour words (both name a colour AND the same colour), + eq3(pocket,door,tint) for 'eq3'; rel_x = crossed with the named words' one-hots. Pure function of the words (the request); rivals' input is unchanged (frozen). Same store, same closed-form head, same head_read.

## Round 2 (val/sweep_r2.json, logs/sweep_r2.log)
| candidate | VAL n-way (s0/s1/s2) | mean | pair | floats | gate |
|---|---|---|---|---|---|
| rel-eq (6 eq, no cross) | 0.8019/0.5689/0.8034 | 0.7247 | 0.5906 | 4656 | FAIL |
| rel-eq3 (7, no cross) | 0.8019/0.5689/0.8034 | 0.7247 | 0.6201 | 4668 | FAIL |
| rel-eq3-xdir | 0.9029/0.5689/0.6747 | 0.7155 | 0.6271 | 5088 | FAIL (s2 < A1) |
| rel-eq3-xdoor-dir | 0.9029/0.8000/0.9236 | 0.8755 | 0.7863 | 5676 | PASS |
| rel-eq3-xall | 0.9029/0.6933/0.9392 | 0.8451 | 0.7752 | 6936 | PASS |
| rel-eq3-xall-s3 (rel x3) | 0.9029/0.8104/0.9392 | 0.8842 | 0.8515 | 6936 | PASS |
| rel-eq3-xall-pair0.5 | 0.9029/0.8015/0.9236 | 0.8760 | 0.8077 | 6936 | PASS |

## D3: selection rule, declared BEFORE round 3
Among gate-passers: highest VAL n-way mean; ties within 0.005 broken by FEWER read floats (a smaller CLU means smaller matched rivals, the harder comparison for... no: the rule is fixed by the standing float match either way; fewer floats = simpler object). Then round 4 re-sweeps ridge / K / atoms around the winner (Advisor point 3). Every candidate logged.

## Round 3 (val/sweep_r3.json, logs/sweep_r3.log) — grid rel {eq,eq3} x rel_x {door+dir, pocket+door+tint+dir, all} x rel_scale {1,3,10} x pair_scale {1,0.5}
All 36 PASS the gate (every seed >= A1). Verbatim:
```
g-eq-xDdir-r1-p1                   nway 0.8755 (0.9029 / 0.8000 / 0.9236)  pair 0.7818 seen 0.9707 value 0.4865  floats   5520  GATE-PASS [54s]
g-eq-xDdir-r1-p0.5                 nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8593 seen 0.9707 value 0.4890  floats   5520  GATE-PASS [59s]
g-eq-xDdir-r3-p1                   nway 0.9076 (0.9029 / 0.8963 / 0.9236)  pair 0.8497 seen 0.9707 value 0.4879  floats   5520  GATE-PASS [52s]
g-eq-xDdir-r3-p0.5                 nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8593 seen 0.9707 value 0.4876  floats   5520  GATE-PASS [48s]
g-eq-xDdir-r10-p1                  nway 0.9066 (0.9029 / 0.8933 / 0.9236)  pair 0.8256 seen 0.9707 value 0.4871  floats   5520  GATE-PASS [39s]
g-eq-xDdir-r10-p0.5                nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8593 seen 0.9707 value 0.4879  floats   5520  GATE-PASS [43s]
g-eq-xPDTdir-r1-p1                 nway 0.7990 (0.9029 / 0.5704 / 0.9236)  pair 0.7244 seen 0.9707 value 0.4828  floats   6240  GATE-PASS [44s]
g-eq-xPDTdir-r1-p0.5               nway 0.8760 (0.9029 / 0.8015 / 0.9236)  pair 0.7827 seen 0.9707 value 0.4843  floats   6240  GATE-PASS [41s]
g-eq-xPDTdir-r3-p1                 nway 0.8760 (0.9029 / 0.8015 / 0.9236)  pair 0.7811 seen 0.9707 value 0.4812  floats   6240  GATE-PASS [38s]
g-eq-xPDTdir-r3-p0.5               nway 0.8760 (0.9029 / 0.8015 / 0.9236)  pair 0.7884 seen 0.9707 value 0.4835  floats   6240  GATE-PASS [32s]
g-eq-xPDTdir-r10-p1                nway 0.8800 (0.9029 / 0.8133 / 0.9236)  pair 0.7743 seen 0.9707 value 0.4817  floats   6240  GATE-PASS [41s]
g-eq-xPDTdir-r10-p0.5              nway 0.8790 (0.9029 / 0.8104 / 0.9236)  pair 0.7884 seen 0.9707 value 0.4839  floats   6240  GATE-PASS [42s]
g-eq-xall-r1-p1                    nway 0.8400 (0.9029 / 0.6933 / 0.9236)  pair 0.7765 seen 0.9707 value 0.4822  floats   6600  GATE-PASS [42s]
g-eq-xall-r1-p0.5                  nway 0.8760 (0.9029 / 0.8015 / 0.9236)  pair 0.8099 seen 0.9707 value 0.4843  floats   6600  GATE-PASS [40s]
g-eq-xall-r3-p1                    nway 0.8790 (0.9029 / 0.8104 / 0.9236)  pair 0.8458 seen 0.9707 value 0.4788  floats   6600  GATE-PASS [47s]
g-eq-xall-r3-p0.5                  nway 0.8790 (0.9029 / 0.8104 / 0.9236)  pair 0.8572 seen 0.9707 value 0.4824  floats   6600  GATE-PASS [35s]
g-eq-xall-r10-p1                   nway 0.8800 (0.9029 / 0.8133 / 0.9236)  pair 0.8482 seen 0.9707 value 0.4777  floats   6600  GATE-PASS [41s]
g-eq-xall-r10-p0.5                 nway 0.8790 (0.9029 / 0.8104 / 0.9236)  pair 0.8572 seen 0.9707 value 0.4822  floats   6600  GATE-PASS [43s]
g-eq3-xDdir-r1-p1                  nway 0.8755 (0.9029 / 0.8000 / 0.9236)  pair 0.7863 seen 0.9707 value 0.4875  floats   5676  GATE-PASS [38s]
g-eq3-xDdir-r1-p0.5                nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8614 seen 0.9703 value 0.4893  floats   5676  GATE-PASS [38s]
g-eq3-xDdir-r3-p1                  nway 0.9076 (0.9029 / 0.8963 / 0.9236)  pair 0.8536 seen 0.9707 value 0.4878  floats   5676  GATE-PASS [35s]
g-eq3-xDdir-r3-p0.5                nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8614 seen 0.9707 value 0.4895  floats   5676  GATE-PASS [32s]
g-eq3-xDdir-r10-p1                 nway 0.9066 (0.9029 / 0.8933 / 0.9236)  pair 0.8296 seen 0.9707 value 0.4870  floats   5676  GATE-PASS [33s]
g-eq3-xDdir-r10-p0.5               nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8614 seen 0.9707 value 0.4872  floats   5676  GATE-PASS [43s]
g-eq3-xPDTdir-r1-p1                nway 0.7990 (0.9029 / 0.5704 / 0.9236)  pair 0.7193 seen 0.9707 value 0.4827  floats   6516  GATE-PASS [44s]
g-eq3-xPDTdir-r1-p0.5              nway 0.8760 (0.9029 / 0.8015 / 0.9236)  pair 0.7815 seen 0.9706 value 0.4861  floats   6516  GATE-PASS [41s]
g-eq3-xPDTdir-r3-p1                nway 0.8812 (0.9029 / 0.8015 / 0.9392)  pair 0.7869 seen 0.9707 value 0.4821  floats   6516  GATE-PASS [43s]
g-eq3-xPDTdir-r3-p0.5              nway 0.8760 (0.9029 / 0.8015 / 0.9236)  pair 0.7884 seen 0.9707 value 0.4863  floats   6516  GATE-PASS [43s]
g-eq3-xPDTdir-r10-p1               nway 0.8851 (0.9029 / 0.8133 / 0.9392)  pair 0.7870 seen 0.9707 value 0.4817  floats   6516  GATE-PASS [40s]
g-eq3-xPDTdir-r10-p0.5             nway 0.8790 (0.9029 / 0.8104 / 0.9236)  pair 0.7884 seen 0.9707 value 0.4853  floats   6516  GATE-PASS [38s]
g-eq3-xall-r1-p1                   nway 0.8451 (0.9029 / 0.6933 / 0.9392)  pair 0.7752 seen 0.9706 value 0.4820  floats   6936  GATE-PASS [36s]
g-eq3-xall-r1-p0.5                 nway 0.8760 (0.9029 / 0.8015 / 0.9236)  pair 0.8077 seen 0.9706 value 0.4854  floats   6936  GATE-PASS [41s]
g-eq3-xall-r3-p1                   nway 0.8842 (0.9029 / 0.8104 / 0.9392)  pair 0.8515 seen 0.9706 value 0.4796  floats   6936  GATE-PASS [42s]
g-eq3-xall-r3-p0.5                 nway 0.8842 (0.9029 / 0.8104 / 0.9392)  pair 0.8572 seen 0.9706 value 0.4824  floats   6936  GATE-PASS [47s]
g-eq3-xall-r10-p1                  nway 0.8851 (0.9029 / 0.8133 / 0.9392)  pair 0.8541 seen 0.9707 value 0.4773  floats   6936  GATE-PASS [47s]
g-eq3-xall-r10-p0.5                nway 0.8842 (0.9029 / 0.8104 / 0.9392)  pair 0.8572 seen 0.9707 value 0.4815  floats   6936  GATE-PASS [44s]
```
Leader: eq x (door, dir) with pair_scale 0.5 -> 0.9135 (0.9029/0.9141/0.9236), 5,520 floats, flat in rel_scale. eq3 ties it at 5,676 floats -> eq (fewer floats) by D3.

## Round 4 (val/sweep_r4.json, logs/sweep_r4.log) — around eq x (door,dir), pair 0.5 (Advisor point 3: K/atoms/ridge)
```
r4-pair0.25                        nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8614 seen 0.9700 value 0.4868  floats   5520  GATE-PASS [50s]
r4-pair0.1                         nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8614 seen 0.9700 value 0.4877  floats   5520  GATE-PASS [56s]
r4-ridge0.3                        nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8569 seen 0.9707 value 0.4893  floats   5520  GATE-PASS [62s]
r4-ridge3                          nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8614 seen 0.9700 value 0.4888  floats   5520  GATE-PASS [38s]
r4-atoms4                          nway 0.8781 (0.9029 / 0.9067 / 0.8246)  pair 0.7975 seen 0.9574 value 0.4808  floats   3680  GATE-PASS [60s]
r4-atoms8                          nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8615 seen 0.9762 value 0.4917  floats   7360  GATE-PASS [65s]
r4-atoms12                         nway 0.9371 (0.9029 / 0.9141 / 0.9943)  pair 0.8713 seen 1.0000 value 0.4963  floats  11040  GATE-PASS [69s]
r4-K3                              nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8593 seen 0.9708 value 0.4901  floats   7896  GATE-PASS [77s]
r4-K4                              nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8622 seen 0.9717 value 0.4903  floats  10272  GATE-PASS [75s]
r4-K1                              nway 0.9371 (0.9029 / 0.9141 / 0.9943)  pair 0.8713 seen 1.0000 value 0.4963  floats   6288  GATE-PASS [48s]
r4-xDdir+pocket                    nway 0.8800 (0.9029 / 0.8133 / 0.9236)  pair 0.8355 seen 0.9707 value 0.4851  floats   5880  GATE-PASS [53s]
r4-mainscale2                      nway 0.9135 (0.9029 / 0.9141 / 0.9236)  pair 0.8602 seen 0.9707 value 0.4827  floats   5520  GATE-PASS [54s]
```
## Round 5 (val/sweep_r5.json, logs/sweep_r5.log)
```
r5-atoms10                         nway 0.9371 (0.9029 / 0.9141 / 0.9943)  pair 0.8713 seen 1.0000 value 0.4957  floats   9200  GATE-PASS [40s]
r5-atoms16                         nway 0.9371 (0.9029 / 0.9141 / 0.9943)  pair 0.8722 seen 1.0000 value 0.4964  floats  14720  GATE-PASS [49s]
r5-atoms12-eq3                     nway 0.9371 (0.9029 / 0.9141 / 0.9943)  pair 0.8722 seen 1.0000 value 0.4970  floats  11352  GATE-PASS [44s]
r5-atoms12-pair1                   nway 0.9035 (0.9029 / 0.8133 / 0.9943)  pair 0.8668 seen 1.0000 value 0.4890  floats  11040  GATE-PASS [33s]
r5-atoms9                          nway 0.9258 (0.9029 / 0.9141 / 0.9604)  pair 0.8713 seen 1.0000 value 0.4943  floats   8280  GATE-PASS [51s]
```
- D4: K = 1 (r4-K1, 0.9371, 6,288 floats) is INADMISSIBLE: with one well there is no derangement, so the frozen charter-C control landscape_shuffled_across_wells cannot exist (the runner asserts no fixed point). Excluded, not selected.
- D5 SELECTION (rule D3): max VAL n-way mean = 0.9371 (0.9029 / 0.9141 / 0.9943), reached by atoms 10 / 12 / 16 (K 2) and atoms-12-eq3; tie -> fewest read floats = **atoms 10, 9,200 floats**. atoms 9 = 0.9258 (below).
- FROZEN: prereg/CLU-REL-A10.json = K 2, n_atoms 10, ridge 1.0, latent 128 (codec unchanged), design [bias | main | 0.5 x pair | eq(6 colour-word pairs) | eq x door one-hot | eq x dir one-hot]; width 318 + 6 + 6*7 + 6*5 = 396... (ledger prints the exact width).
- Gate vs A1: mean 0.9371 > 0.725; per seed 0.9029 >= 0.8019, 0.9141 >= 0.5689, 0.9943 >= 0.8034 -> KEPT.
- Total candidates tried on VAL: 9 (r1) + 7 (r2) + 36 (r3) + 12 (r4) + 5 (r5) = 69, all logged above with numbers.
- PREREG written + hashed 2026-09-24T00:27:19Z (before any TEST number).

## Diagnostic on the frozen config (logs/diag_nway_frozen.log)
Every RULE-B tuple (c,0,c), (c,open,c) that A1 missed is now 1.000 (seed 2 (3,0,3): 0.353 -> 0.660). Remaining n-way failures = the INTERACTION family (c,c,c) only, 0.000 on every seed (A1: also 0.000) -- the irreducible 3-way (door consumes key, THEN tint) is NOT solved by the relational coding; eq3 (the 3-way same-colour indicator) did not fix it either (round 3), because (c,c,c) has no train support in most seeds.

## Launch queue (launch_queue.sh): BOTH val 0-2 -> BOTH TEST 0-2 (runs 7-9, ledger row >> before each) -> WORDS-FULL val 0-2; <=3 of ours alive, >=5 GB free per launch.
2026-09-24T01:01:15Z launched BOTH:val:0 msys_pid=222478 winpid=16548 free_gb=10
2026-09-24T01:04:01Z launched BOTH:val:1 msys_pid=222589 winpid=28836 free_gb=7
2026-09-24T01:05:42Z launched BOTH:val:2 msys_pid=222675 winpid=2552 free_gb=10
- NOTE: in every w21 runner JSON, 'config' = the runner's A1 cfg (K2/6 atoms/ridge 1), which ONLY the frozen learned blind arms use; the CLU + its 4 controls + deletion objects are built from 'w21_clu_cfg' (CLU-REL-A10, 10 atoms) -- see 'clu_ledger' (read_floats 9200, design_width 396).
2026-09-24T05:20:26Z launched BOTH:test:0 msys_pid=231275 winpid=29232 free_gb=15

## BOTH VAL seed 0 (full runner, rivals re-matched to 9,200) — logs/val_BOTH_seed0.log
CLU 0.9029 | best matched 8k transformer 0.7926, 80k transformer 0.8112 | controls: removed 0.000, shuffled 0.447, across-wells (deranged [1,0]) **0.782**, random 0.371 | blind 0.242/0.000 | floor 0.000 | '+rel' diag: mlp-80k 0.912, transformer 0.903.
⚠ on this seed the across-wells control is above CLU-0.15.
## POST-HOC (after freezing; never changes the config) — posthoc/controls_val.json, logs/posthoc_controls.log
Controls on the VAL cache (means over 3 seeds, n-way): A1 max 0.250 (= w19 VAL, instrument check); rel-atoms4 max 0.260; rel-atoms6 0.296; rel-atoms8 0.361; CLU-REL-A10 **0.615** (across-wells), bar 0.787 -> gate holds on the mean, but the controls climb with atom count: more atoms per 64-wide well = a less binding store.
2026-09-24T05:29:07Z launched BOTH:test:1 msys_pid=231549 winpid=36956 free_gb=14

## BOTH VAL, 3 seeds (val_full/LABEL-BOTH-val.json, logs/label_BOTH_val.log) — free, does NOT gate the scored runs (PREREG §3)
```
BOTH val nway: CLU 0.9371 [0.9029, 0.9141, 0.9943] | R* rival@10x-steps: transformer 0.6849 [0.8112, 0.5689, 0.6747] | best/seed [0.8112, 0.5689, 0.6747] | ahead 3/3 s 0.1395 | ctl max 0.6147 (bar 0.7871) | blind 0.3344 | floor 0.0000 | +rel best diagnostic+rel@80k: transformer +rel 0.8656 | -> WIN
```
- Mechanism read: matched rivals on the frozen input cap at 0.5689 on seed 1 (the RULE-B tuples); the same rivals handed the CLU's relational columns ('+rel') reach 0.866 mean -> most of the lift is the colour-relational CODING; the CLU keeps +0.071 over the best '+rel' rival on the mean (VAL).
2026-09-24T05:37:46Z launched BOTH:test:2 msys_pid=231888 winpid=32988 free_gb=17
2026-09-24T08:35:54Z launched WORDS-FULL:val:0 msys_pid=238092 winpid=17696 free_gb=17
2026-09-24T08:45:32Z launched WORDS-FULL:val:1 msys_pid=238660 winpid=30208 free_gb=22

## SCORED TEST, runs 7-9 (test/LABEL-BOTH-test.json, logs/label_BOTH_test.log)
```
BOTH test nway: CLU 0.9285 [0.908, 0.8935, 0.984] | R* rival@10x-steps: transformer 0.6851 [0.8029, 0.5674, 0.6851] | best/seed [0.8029, 0.5674, 0.6851] | ahead 3/3 s 0.1205 | ctl max 0.6006 (bar 0.7785) | blind 0.3418 | floor 0.0000 | +rel best diagnostic+rel@80k: transformer +rel 0.8621 | -> WIN
```
- per-seed controls n-way: across-wells 0.778/0.563/0.461 (seed 0 above its own CLU-0.15 = 0.758; mean gate holds 0.601 <= 0.779); shuffled 0.456/0.452/0.477; random 0.385/0.241/0.373; removed 0/0/0.
- deletion: store-scoped exactness rel dS 7.7e-16..1.2e-15 (bar 1e-5), downdate == fitted-without on 100% of surviving n-way rows, all 6 (seed x mode); CLU adapt 0.36-2.78 s; rival retrain-without 112-169 s.
2026-09-24T08:49:13Z launched WORDS-FULL:val:2 msys_pid=238896 winpid=27084 free_gb=21
- CORRECTION: candidate count is 68 (+ A1-repro), 57 gate-passers; PREREG §1 / ledger rows say 69 (included A1-repro). Not edited there (hashed / append-only).

## WORDS-FULL VAL, 3 seeds (val_full/LABEL-WORDS-FULL-val.json, logs/label_WORDS-FULL_val.log)
```
WORDS-FULL val nway: CLU 0.9371 [0.9029, 0.9141, 0.9943] | R* rival@10x-steps: transformer 0.7529 [0.8497, 0.5689, 0.8402] | best/seed [0.8497, 0.5689, 0.8586] | ahead 3/3 s 0.1505 | ctl max 0.6147 (bar 0.7871) | blind 0.3344 | floor 0.0000 | +rel best diagnostic+rel@8k: transformer +rel 0.9414 | -> WIN
```
- '+rel' transformer 8k 0.941 (0.923/0.914/0.987) >= CLU 0.937: on WORDS-FULL the relational coding is the whole margin (COMPARABLE with a rival handed the same columns).
2026-09-24T10:39:46Z queue finished

## Close-out
- All our processes exited (checked by command line: PID 16548 had been reused by a w20 peer process and was NOT touched). No box-wide kill was ever run.
- Scored runs spent: 7, 8, 9 of 9 (BOTH TEST), ledger rows before each + outcome row. Result: WIN (Head's rule) vs frozen-input matched rivals; mechanism = colour-relational request coding (the '+rel' diagnostic closes most of the gap on BOTH, all of it on WORDS-FULL VAL).
- DID NOT RUN: WORDS-FULL TEST (not this hub's), OBS / WORDS-MAIN with the new CLU, any other config on the full runner or TEST, dyn-interactions / two-stage fit / item-store / read-verb variants (declared, FINDINGS §6).
