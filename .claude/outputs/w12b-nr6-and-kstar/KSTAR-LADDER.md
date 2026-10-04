# The K* replication of the composition ladder

⛔ **A REPLICATION. FREE. No ledger row, no scored budget.**

⛔ **AXIS LABEL.** ⛔ rungs n = 3, 4, 5 and 6 probe the IDENTICAL 120 corners (w12 spoke/seeds-and-tables). For n >= 3 this is a TRAINING-DATA axis, not a composition-depth axis; only n = 2 vs n = 6 is a genuine depth contrast. The rungs are reported and the axis is named for what it is.

## The three columns

| rung | n seeds | `ladder K=5` store − `learned_1x` | `bridge K=5` store − `learned_1x` | `K* = 32` store − `learned_1x` |
|---|---|---|---|---|
| n=2 | 3/1/3 | -2.7833 | -0.2500 | 4.3556 |
| n=3 ⚠same corners | 3/1/3 | 4.2417 | 8.4250 | 14.0722 |
| n=4 ⚠same corners | 3/1/3 | -5.2833 | -1.0167 | 10.0222 |
| n=5 ⚠same corners | 3/1/3 | -3.9000 | -0.5167 | 8.3556 |
| n=6 ⚠same corners | 3/1/3 | -4.0611 | -2.5500 | 7.9778 |

## The opponent's size — the thing that changed

| rung | `ladder K=5` params | `bridge K=5` params | `K* = 32` params |
|---|---|---|---|
| n=2 | 3,177,896 | 1,029,776 | 536,972 |
| n=3 | 3,174,386 | 1,029,776 | 536,270 |
| n=4 | 3,214,400 | 1,029,776 | 536,270 |
| n=5 | 3,177,194 | 1,029,776 | 536,972 |
| n=6 | 3,180,704 | 1,029,776 | 536,972 |

## The control gate at K* — checked BEFORE any margin

| rung | seed | arm | `landscape_removed` | `landscape_shuffled` | `random_atoms` | exactly 0? | collapses beyond the declared 1.0 margin? |
|---|---|---|---|---|---|---|---|
| n=2 | 0 | 31.800 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=2 | 1 | 32.500 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=2 | 2 | 33.633 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=3 | 0 | 60.150 | 0.0000 | 0.0083 | 0.0000 | no | **YES** |
| n=3 | 1 | 62.783 | 0.0000 | 0.0083 | 0.0000 | no | **YES** |
| n=3 | 2 | 61.908 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=4 | 0 | 65.242 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=4 | 1 | 65.817 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=4 | 2 | 67.200 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=5 | 0 | 65.642 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=5 | 1 | 66.817 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=5 | 2 | 67.283 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=6 | 0 | 65.908 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=6 | 1 | 66.250 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=6 | 2 | 67.650 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |

⚠ `exactly 0` is the crossover hub's OBSERVATION at K* on val, not a bar. The PRE-DECLARED bar is `LADDER-READY.json`'s `DECORATION_MARGIN_NOVELTY = 1.0` novel tuple per corner. ⛔ Neither is moved to accommodate the other, and a non-zero control reading is printed rather than rounded away.


## SEED-MATCHED K = 5 vs K\* — the only honest comparison

| rung | seeds | store K=5 | store K\* | opp K=5 params | opp K\* params | opp shrank | margin K=5 | margin K\* | sign flipped? |
|---|---|---|---|---|---|---|---|---|---|
| n=2 | [0, 1, 2] | 30.539 | 32.644 | 3,177,896 | 536,972 | 5.92× | -2.783 | 4.356 | **YES** |
| n=3 | [0, 1, 2] | 59.750 | 61.614 | 3,174,386 | 536,270 | 5.92× | 4.242 | 14.072 | **no** |
| n=4 | [0, 1, 2] | 62.708 | 66.086 | 3,214,400 | 536,270 | 5.99× | -5.283 | 10.022 | **YES** |
| n=5 | [0, 1, 2] | 64.719 | 66.581 | 3,177,194 | 536,972 | 5.92× | -3.900 | 8.356 | **YES** |
| n=6 | [0, 1, 2] | 64.556 | 66.603 | 3,180,704 | 536,972 | 5.92× | -4.061 | 7.978 | **YES** |

### The ladder's OWN pre-declared bar, evaluated at both K

⛔ `AMENDMENT-2 §3` of `w12 hub/composition-ladder`: the arm clears only if it beats the opponent by `DECORATION_MARGIN_NOVELTY = 1.0` novel tuple per corner on **every** declared seed — not on the seed mean. The bar is not moved here; it is evaluated at the other K as well.

| rung | clears at K=5? | worst seed K=5 | clears at K\*? | worst seed K\* |
|---|---|---|---|---|
| n=2 | **no** | -4.567 | **YES** | **3.117** |
| n=3 | **no** | -0.033 | **YES** | **11.533** |
| n=4 | **no** | -6.350 | **YES** | **8.625** |
| n=5 | **no** | -4.808 | **YES** | **6.517** |
| n=6 | **no** | -5.775 | **YES** | **6.108** |

## ⭐ THE SWING, DECOMPOSED — what the bridge column buys

| rung | seeds | margin: ladder K=5 → bridge K=5 → K\* | CONSTRUCTION step | K step | total | K's share |
|---|---|---|---|---|---|---|
| n=2 | [0] | -4.567 → -0.250 → 6.667 | **+4.317** | **+6.917** | +11.233 | 62% |
| n=3 | [0] | 4.817 → 8.425 → 17.875 | **+3.608** | **+9.450** | +13.058 | 72% |
| n=4 | [0] | -6.242 → -1.017 → 12.700 | **+5.225** | **+13.717** | +18.942 | 72% |
| n=5 | [0] | -4.808 → -0.517 → 10.350 | **+4.292** | **+10.867** | +15.158 | 72% |
| n=6 | [0] | -5.775 → -2.550 → 9.375 | **+3.225** | **+11.925** | +15.150 | 79% |

⛔ Without the bridge the WHOLE swing would have been charged to K. The construction step is the partition rule, the store class and the covariance convention moving together; only the K step is what the Head's ruling is about.


## Verdict
```
{
 "SIGN_FLIPPED_AT_RUNGS": [
  "2",
  "4",
  "5",
  "6"
 ],
 "reading": "\u2b50\u2b50 THE K* REPLICATION FLIPS THE SIGN at rungs ['2', '4', '5', '6']. Under the Head's ruling the main text carries K*, so this is a CORRECTION to the K = 5 verdict and must be reported as one, not buried.",
 "verdict": {
  "ladder_K5": {
   "store_wins_at_rungs": [
    "3"
   ],
   "store_loses_at_rungs": [
    "2",
    "4",
    "5",
    "6"
   ],
   "DECLARED_NOT_RUN_rungs": [],
   "margins_per_rung": {
    "2": -2.783333333333335,
    "3": 4.241666666666667,
    "4": -5.283333333333334,
    "5": -3.8999999999999986,
    "6": -4.061111111111113
   },
   "seeds_per_rung": {
    "2": 3,
    "3": 3,
    "4": 3,
    "5": 3,
    "6": 3
   }
  },
  "bridge_K5": {
   "store_wins_at_rungs": [
    "3"
   ],
   "store_loses_at_rungs": [
    "2",
    "4",
    "5",
    "6"
   ],
   "DECLARED_NOT_RUN_rungs": [],
   "margins_per_rung": {
    "2": -0.25,
    "3": 8.425000000000004,
    "4": -1.0166666666666728,
    "5": -0.5166666666666657,
    "6": -2.549999999999997
   },
   "seeds_per_rung": {
    "2": 1,
    "3": 1,
    "4": 1,
    "5": 1,
    "6": 1
   }
  },
  "kstar_K32": {
   "store_wins_at_rungs": [
    "2",
    "3",
    "4",
    "5",
    "6"
   ],
   "store_loses_at_rungs": [],
   "DECLARED_NOT_RUN_rungs": [],
   "margins_per_rung": {
    "2": 4.355555555555555,
    "3": 14.072222222222221,
    "4": 10.022222222222219,
    "5": 8.355555555555554,
    "6": 7.977777777777779
   },
   "seeds_per_rung": {
    "2": 3,
    "3": 3,
    "4": 3,
    "5": 3,
    "6": 3
   }
  }
 }
}
```
