# w9 maze harness - results

Scored by `w9_harness.judge_maze.JUDGE` - env-scout's exact integer judge, re-exported unwrapped. No tolerance anywhere.

## Audits

| check | result |
|---|---|
| judge exact (all four parts) | **True** |
| tolerance used anywhere | **None** |
| A source: float machinery in judge | none |
| B types: float contact under tracing ints | none |
| C redundant: disagreements with independent replay | 0 / 750 routes |
| D oracle: optimum / truncation / budget / mutants | 0 failures |
| stepper disagreements with the judge | 0 / 750 routes |
| encoder loss sees actions or judge | True |
| action head identical shape across arms | True |
| impossible-set certification failures | 0 |
| impossible examples seen by a non-refuser arm | 0 |

## CMAZE-1, env-scout adopted defaults

### Main tier - 3 seed(s): 0, 1, 2

tier `cmaze1`, generator (env-scout adopted defaults), 500 train / 200 eval instances per seed

| arm / column | params | validity (exact judge) | score | goal, budget lifted | legal prefix / opt | per-step acc |
|---|---|---|---|---|---|---|
| MLP | 60047 | 0.0050 [0.0000, 0.0100] | 0.0048 [0.0000, 0.0100] | 0.0050 [0.0000, 0.0100] | 0.4226 [0.1554, 0.6270] | 0.5203 [0.4257, 0.5779] |
| LSTM | 59965 | 0.0017 [0.0000, 0.0050] | 0.0017 [0.0000, 0.0050] | 0.0017 [0.0000, 0.0050] | 0.2319 [0.1083, 0.3136] | 0.7515 [0.7221, 0.7684] |
| Transformer | 59819 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.1997 [0.0960, 0.2551] | 0.7366 [0.6664, 0.7772] |
| Table floor (raw state) | - | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.1664 [0.1372, 0.1974] | n/a |
| Table-enc (encoder bits) | - | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0719 [0.0484, 0.1129] | n/a |
| | | | | | | |
| oracle (full composition)  | - | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | - |
| blind-A (rule B dropped) **reported column** | - | 0.1400 [0.1050, 0.1750] | 0.1400 [0.1050, 0.1750] | 0.1400 [0.1050, 0.1750] | 0.3706 [0.3503, 0.3834] | - |
| blind-B (rule A dropped) **reported column** | - | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.2225 [0.2024, 0.2416] | - |
| blind-colour (colours collapsed) **reported column** | - | 0.0933 [0.0850, 0.1050] | 0.0931 [0.0850, 0.1050] | 0.0983 [0.0900, 0.1100] | 0.3070 [0.3052, 0.3082] | - |
| blind-factor (A x B local only) **reported column** | - | 0.2350 [0.1850, 0.2700] | 0.2350 [0.1850, 0.2700] | 0.2350 [0.1850, 0.2700] | 0.4609 [0.4335, 0.4748] | - |
| blind-sequential (key order from A) **reported column** | - | 0.5317 [0.5100, 0.5550] | 0.5314 [0.5100, 0.5547] | 0.5433 [0.5200, 0.5650] | 0.5433 [0.5200, 0.5650] | - |
| fragment-lookup r=1 **reported column** | - | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.3231 [0.2946, 0.3475] | - |
| fragment-lookup r=2 **reported column** | - | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0392 [0.0369, 0.0432] | - |

**Supervised refuser** (the only comparator for DISPLAY rows; refusal read as the decision at t=0):

| | refusal rate | validity when it acts |
|---|---|---|
| on IMPOSSIBLE (correct refusals) | 0.8317 [0.6750, 1.0000] | 0.0000 [0.0000, 0.0000] |
| on POSSIBLE (false refusals) | 0.7850 [0.5600, 1.0000] | 0.0000 [0.0000, 0.0000] |
| **separation** | 0.0467 [0.0000, 0.1150] | |

Alternative reading, refusal allowed at any step: impossible 0.8700 [0.7350, 1.0000], possible 0.8167 [0.6300, 1.0000].


## CMAZE-1-EASY (declared tier: h=w=4, k=2, same generator, same judge)

### Easy tier - 3 seed(s): 0, 1, 2

tier `cmaze1-easy`, generator {'h': 4, 'w': 4, 'n_colours': 2}, 500 train / 200 eval instances per seed

| arm / column | params | validity (exact judge) | score | goal, budget lifted | legal prefix / opt | per-step acc |
|---|---|---|---|---|---|---|
| MLP | 60331 | 0.0150 [0.0000, 0.0350] | 0.0150 [0.0000, 0.0350] | 0.0150 [0.0000, 0.0350] | 0.4160 [0.1485, 0.5503] | 0.5865 [0.4860, 0.6408] |
| LSTM | 59345 | 0.0183 [0.0000, 0.0400] | 0.0179 [0.0000, 0.0388] | 0.0183 [0.0000, 0.0400] | 0.3255 [0.1917, 0.4152] | 0.7195 [0.6984, 0.7465] |
| Transformer | 60237 | 0.0150 [0.0100, 0.0200] | 0.0147 [0.0100, 0.0191] | 0.0150 [0.0100, 0.0200] | 0.3136 [0.2035, 0.3770] | 0.7373 [0.7220, 0.7491] |
| Table floor (raw state) | - | 0.0017 [0.0000, 0.0050] | 0.0017 [0.0000, 0.0050] | 0.0017 [0.0000, 0.0050] | 0.2463 [0.2195, 0.2757] | n/a |
| Table-enc (encoder bits) | - | 0.0017 [0.0000, 0.0050] | 0.0017 [0.0000, 0.0050] | 0.0017 [0.0000, 0.0050] | 0.0922 [0.0790, 0.1183] | n/a |
| | | | | | | |
| oracle (full composition)  | - | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | - |
| blind-A (rule B dropped) **reported column** | - | 0.4983 [0.4900, 0.5100] | 0.4983 [0.4900, 0.5100] | 0.4983 [0.4900, 0.5100] | 0.6505 [0.6441, 0.6623] | - |
| blind-B (rule A dropped) **reported column** | - | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.2655 [0.2585, 0.2708] | - |
| blind-colour (colours collapsed) **reported column** | - | 0.3733 [0.3600, 0.3850] | 0.3733 [0.3600, 0.3850] | 0.3733 [0.3600, 0.3850] | 0.5397 [0.5262, 0.5522] | - |
| blind-factor (A x B local only) **reported column** | - | 0.6467 [0.6200, 0.6650] | 0.6467 [0.6200, 0.6650] | 0.6467 [0.6200, 0.6650] | 0.7560 [0.7369, 0.7672] | - |
| blind-sequential (key order from A) **reported column** | - | 0.7983 [0.7750, 0.8200] | 0.7981 [0.7748, 0.8200] | 0.7983 [0.7750, 0.8200] | 0.7983 [0.7750, 0.8200] | - |
| fragment-lookup r=1 **reported column** | - | 0.0033 [0.0000, 0.0050] | 0.0033 [0.0000, 0.0050] | 0.0033 [0.0000, 0.0050] | 0.2937 [0.2831, 0.2992] | - |
| fragment-lookup r=2 **reported column** | - | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0680 [0.0596, 0.0748] | - |

**Supervised refuser** (the only comparator for DISPLAY rows; refusal read as the decision at t=0):

| | refusal rate | validity when it acts |
|---|---|---|
| on IMPOSSIBLE (correct refusals) | 0.6217 [0.3500, 0.9950] | 0.0000 [0.0000, 0.0000] |
| on POSSIBLE (false refusals) | 0.6017 [0.3350, 1.0000] | 0.0050 [0.0000, 0.0150] |
| **separation** | 0.0200 [-0.0050, 0.0500] | |

Alternative reading, refusal allowed at any step: impossible 0.6900 [0.4350, 1.0000], possible 0.6700 [0.4200, 1.0000].

