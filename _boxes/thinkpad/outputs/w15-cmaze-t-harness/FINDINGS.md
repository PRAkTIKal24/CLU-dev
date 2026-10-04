# w15/cmaze-t-harness — FINDINGS

**CMAZE-T, the compositional transition instrument.** Built, audited, and shipped with a
seeded VAL smoke so wave 16 opens with numbers.

* Acceptance artefact: **`HARNESS-READY.json`**, `"status": "ready"`.
* Smoke artefact: **`VAL-SMOKE.json`** + per-seed `val_seed{0,1,2}.json`.
* Pre-registration: **`PREREG.md`** (+ amendments 1–3, dated, with the pre-amendment
  runs preserved under `preamendment2/` and `amendment23/`).
* Narrative and every design decision: **`LOG.md`**.
* **Scored runs spent: ZERO.** No row of `/c/chlu/SCORED-LEDGER.md` is claimed. The
  task file budgets zero for this spoke and the val smoke is free under `COMMON.md`.
* **Tolerance used: none.** Every judgement in this harness is an integer comparison.
* **Seed count is a column on every table below: n_seeds = 3** (seeds 0, 1, 2, all or
  none). Every byte figure travels with its convention named.

Branch `w15/cmaze-t-harness` on base `base/iter8 = 148f3b7`; bundle
`w15-cmaze-t-harness.bundle`.

---

## 1. What CMAZE-T is

Given an instance, a grid state `s_t = (cell, keystate)` and an action `a_t`, an arm
**generates the next state**. The **engine judges the transition exactly** — legality
under RULES A+B, displacement, keystate and pocket — by integer comparison against
`w9_harness.dynamics.step`, which is itself already audited route-for-route against
`w8_envscout.maze_colour.judge`. A **rollout** chains *generated* states: the state fed
to step `t+1` is the arm's own `ŝ_t`, never the engine's.

⛔ **Engine, solver and exact judge are imported unmodified.** Their file SHA-256 values
are in `HARNESS-READY.json → provenance`. `w8_envscout/maze_colour.py` hashes to
`9eacdda17d87cac0ad5ced6ccbb6fa43ebc1508c0460d0cc183e545d066beaac` — **byte-identical to
the value w14/request-layer-fix recorded** in its own `REQUEST-LAYER-READY.json`, so
"the engine did not move" is checkable rather than asserted. Source: `chlu/exp_w15_cmaze_t.py`,
`chlu/exp_w15_cmaze_t_arms.py`, `scripts/w15_cmaze_t_run.py`,
`scripts/w15_cmaze_t_ready.py` — the only files this spoke owns.

**The five request words** (`LOG.md` D1) — an agent in this environment has a colour
exactly when it is carrying a key, so *agent colour* **is** the pocket and *carry state*
is its coarsening, which is why the task file puts it in parentheses:

| word | alphabet (k=4) | what it is |
|---|---|---|
| `w_agent` | 5 | pocket colour before the step (0 = empty) |
| `w_key` | 5 | colour of an **untaken** key home on the current cell |
| `w_door` | 7 | target cell: 0 none, 1..k CLOSED door colour c, k+1 OPEN, k+2 WALL |
| `w_tint` | 5 | target cell tint colour |
| `w_dir` | 5 | N E S W P |

**The emission**: `(legal, displacement, key_event, pocket_next)` — one-hot blocks,
`2 + 5 + 9 + 5 = 21` dims at k=4. The harness applies `(legal, displacement, key_event)`
to the real `s_t` and requires the generated `(legal, pos′, ks′)` to equal the engine's
`step()` output **exactly**, with `pocket_next` agreeing.

---

## 2. THE ONE DISTRIBUTIONAL CHANGE — tinted doors (and why it was necessary)

`maze_colour.generate` zeroes the tint on start, goal, **door** and key cells.
**Measured: 160 door cells over 40 generator seeds, 0 of them tinted**
(`LOG.md` §"the structural finding"). So in CMAZE-1 as shipped, RULES A and B can never
interact *within one transition*, the 3-way configuration `(door=c, pocket=c, tint=c)`
cannot occur, and BLIND-SEQUENTIAL is **vacuous** at the transition level — it would
print 1.0000 identically to ORACLE and measure nothing.

CMAZE-T's request layer therefore tints door cells and hands the result to the
**unchanged** engine, solver and judge. The rules already specify the case — `judge`,
`step`, `judge_ref` and `solve` all resolve the door first and then test the tint against
the **updated** pocket — only the generator declines to emit it. Same class of move as
`w9_maze.task._poison_pairs` and `_seal`. Every tinted instance is re-solved with the
unmodified solver and a tint that kills the oracle is reverted.

**Receipts** (`val_seed{0,1,2}.json → banks.train.door_tint`), 90 train mazes per seed:

| seed | door cells | tinted | self-tinted | reverted to preserve the oracle | mean opt_len before → after |
|---|---|---|---|---|---|
| 0 | 360 | 356 | 222 | 4 | 45.33 → 45.33 |
| 1 | 360 | 354 | 202 | 6 | 42.97 → 42.97 |
| 2 | 360 | 354 | 215 | 6 | 40.04 → 40.10 |

**What it creates**: entering a **CLOSED** door `c` tinted `c` while carrying `c` is
LEGAL (the key is spent, the pocket empties, the tint no longer bites); entering the
**same cell once the door is OPEN** while carrying `c` is ILLEGAL. Same three colours,
opposite verdicts, flipped by the door's open/closed state. That configuration occurs
**265 / 234 / 226** times in the three val banks
(`HARNESS-READY.json → ACCEPTANCE.4 → three_way_configuration_count_per_seed_val_bank`).

---

## 3. ACCEPTANCE — all six conditions, each measured, not asserted

Source: `HARNESS-READY.json → ACCEPTANCE`. Bars from `PREREG.md` §4.1.

### 3.1 Independent replay — **0 disagreements over 164,767 transitions** ✅
`HARNESS-READY.json → ACCEPTANCE.1`. Bar: 0 disagreements over ≥ 50,000.

| seed | transitions replayed | disagreements |
|---|---|---|
| 0 | 55,285 | **0** |
| 1 | 54,710 | **0** |
| 2 | 54,772 | **0** |
| **total** | **164,767** | **0** |

**Two** independent drivers, both compared against the incremental stepper the harness
used to build each row:
1. **`step_ref`** — written from the RULES, carrying state as a scalar and two **sets**
   instead of the array-of-trits the stepper uses, so a shared bug cannot hide in a
   shared data structure;
2. **`judge_ref`** — the wave-9 second replay, driven from `inst.start` over the whole
   **prefix + action**, which never sees `s_t` at all. Every bank row carries a legal
   action prefix (states come from a BFS over `(cell, keystate)`), which is what makes
   this driver cover 100 % of rows rather than a reachable fraction.

### 3.2 No impossible example shown to a non-refuser arm — **0 of 164,767** ✅
`HARNESS-READY.json → ACCEPTANCE.2`. An example is IMPOSSIBLE iff **no** emission exists
that the exact transition judge accepts. The audit does not assert there is one — it
**constructs** the accepting emission for every row and checks the judge takes it.
**164,767 rows checked, 0 with no accepting emission.**
**Refusal displays are retired**: the value hold-out is **scored**, and no arm in this
harness carries a refuse token.

### 3.3 Hold-outs disjoint from train by construction — ✅
`HARNESS-READY.json → ACCEPTANCE.3`.

| seed | train rows | rows mentioning the held-out colour | held-out n-way combos in train | held-out 2-way combos in train | distinct word tuples in train |
|---|---|---|---|---|---|
| 0 | 20,218 | **0** | **0** / 20 | **0** / 8 | 381 |
| 1 | 19,447 | **0** | **0** / 20 | **0** / 8 | 363 |
| 2 | 17,618 | **0** | **0** / 20 | **0** / 8 | 342 |

**3b — the five words determine the transition** (which is what makes a hold-out over
word combinations a statement about the *transition* and not about a featurisation):

| seed | rows | distinct word tuples | tuples with two outcomes |
|---|---|---|---|
| 0 | 55,285 | 1,081 | **0** |
| 1 | 54,710 | 1,143 | **0** |
| 2 | 54,772 | 1,117 | **0** |

**Mazes**: train and val mazes come from **disjoint generator seed blocks**, so every
scored maze is a random maze never seen at train and a within-maze lookup cannot
transfer. The **test** block (`[2.30M, 2.40M]`, `[2.70M, 2.80M]`, `[3.10M, 3.20M]`) is
recorded and **never built or evaluated** — it is wave 16's.

### 3.4 Rule B bites — **REPORTED COLUMN, never an abort condition**
`HARNESS-READY.json → ACCEPTANCE.4`. Fraction of transitions the engine makes ILLEGAL
under RULES A+B that would be LEGAL under RULE A alone.

| distribution | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| stratified bank, train | 0.0719 | 0.0732 | 0.0716 |
| stratified bank, val | 0.0727 | 0.0728 | 0.0704 |
| natural distribution, train | 0.0128 | 0.0120 | 0.0121 |
| natural distribution, val | 0.0144 | 0.0161 | 0.0104 |

The **stratified bank** caps rows per WORD TUPLE, which is what gives the rare
configurations enough rows to score; the **natural distribution** is a uniform sample of
`(reachable state, action)` pairs. Both are printed, each named, neither is a gate.
For reference the overall illegal fraction on the natural val distribution is ≈ 0.566.

### 3.5 Seeded VAL smoke — 3 seeds, all rows, every rival ✅ → §4
### 3.6 PREREG before any number ✅ — `PREREG.md`, sha256 in `HARNESS-READY.json`

---

## 4. THE VAL SMOKE — Config-A vs every rival, 3 seeds

Source: `VAL-SMOKE.json → rows`. Scored column: **`exact`**, the fraction of transitions
whose generated next state the exact transition judge accepts. Values are
**mean ± sd over n_seeds = 3**, with the per-seed values beside them.

**Row sizes per seed** (a declared, uniform, seeded cap of 4,000 rows per class keeps
the Python-side exact judge inside the wall clock; every arm on a row sees the same rows):
`seen` 4000/4000/4000 · `unseen 2-way` 812/1121/2129 · `unseen n-way` 752/675/707 ·
`value hold-out` 4000/4000/4000.

### 4.1 `seen`

| arm | mean ± sd | per seed |
|---|---|---|
| **CLU (single design)** | **0.9709 ± 0.0078** | 0.975 / 0.976 / 0.962 |
| *diagnostic: transformer @ 10× read floats* | *0.9028 ± 0.0146* | *0.886 / 0.913 / 0.909* |
| *diagnostic: lstm @ 10×* | *0.8672 ± 0.0151* | *0.850 / 0.879 / 0.872* |
| *diagnostic: mlp @ 10×* | *0.8408 ± 0.0321* | *0.826 / 0.819 / 0.878* |
| rival: transformer (matched) | 0.7253 ± 0.0974 | 0.666 / 0.672 / 0.838 |
| rival: mlp (matched) | 0.4395 ± 0.1372 | 0.284 / 0.543 / 0.491 |
| rival: lstm (matched) | 0.3718 ± 0.3483 | 0.082 / 0.275 / 0.758 |
| arm: BLIND-SEQUENTIAL (learned) | 0.8483 ± 0.1207 | 0.709 / 0.908 / 0.927 |
| arm: BLIND-COLOUR (learned) | 0.5137 ± 0.0503 | 0.463 / 0.514 / 0.564 |
| floor: lookup **cross-maze WORDS** | 0.9967 ± 0.0018 | 0.998 / 0.995 / 0.997 |
| floor: lookup cross-maze raw r1 | 0.3802 ± 0.0155 | 0.364 / 0.381 / 0.396 |
| floor: fragment cross-maze r1 | 0.2918 ± 0.0122 | 0.278 / 0.295 / 0.302 |
| floor: lookup WITHIN-maze raw r1 *(not transfer)* | 0.2903 ± 0.0230 | 0.271 / 0.284 / 0.316 |
| control: landscape_removed | 0.0203 ± 0.0351 | 0.000 / 0.061 / 0.000 |
| control: landscape_shuffled | 0.4678 ± 0.1623 | 0.282 / 0.537 / 0.584 |
| control: landscape_shuffled_across_wells | 0.5351 ± 0.1779 | 0.538 / 0.356 / 0.712 |
| control: random_atoms | 0.4996 ± 0.2036 | 0.266 / 0.596 / 0.637 |
| column: ORACLE (planner) | 1.0000 ± 0.0000 | — |
| column: BLIND-SEQUENTIAL (planner) | 0.9969 ± 0.0037 | 0.998 / 1.000 / 0.993 |
| column: BLIND-COLOUR (planner) | 0.8325 ± 0.0405 | 0.818 / 0.801 / 0.878 |
| column: CODEC CEILING (pipeline ceiling) | 1.0000 ± 0.0000 | — |

### 4.2 `unseen 2-way`

| arm | mean ± sd | per seed |
|---|---|---|
| **CLU (single design)** | **0.3932 ± 0.2033** | 0.167 / 0.450 / 0.562 |
| *diagnostic: mlp @ 10×* | *0.3977 ± 0.1070* | *0.372 / 0.306 / 0.515* |
| *diagnostic: transformer @ 10×* | *0.3969 ± 0.1006* | *0.399 / 0.295 / 0.496* |
| *diagnostic: lstm @ 10×* | *0.3677 ± 0.1143* | *0.321 / 0.284 / 0.498* |
| rival: transformer (matched) | 0.3143 ± 0.1620 | 0.249 / 0.195 / 0.499 |
| rival: mlp (matched) | 0.2219 ± 0.0555 | 0.212 / 0.172 / 0.282 |
| rival: lstm (matched) | 0.1867 ± 0.2617 | 0.016 / 0.056 / 0.488 |
| **arm: BLIND-SEQUENTIAL (learned)** | **0.5059 ± 0.0766** | 0.501 / 0.432 / 0.585 |
| arm: BLIND-COLOUR (learned) | 0.0937 ± 0.0657 | 0.076 / 0.038 / 0.166 |
| floor: lookup **cross-maze WORDS** | **0.0000 ± 0.0000** | 0.000 / 0.000 / 0.000 |
| floor: lookup cross-maze raw r1 | 0.0183 ± 0.0081 | 0.016 / 0.012 / 0.027 |
| floor: fragment cross-maze r1 | 0.0449 ± 0.0237 | 0.055 / 0.018 / 0.062 |
| floor: lookup WITHIN-maze raw r1 *(not transfer)* | 0.0836 ± 0.0117 | 0.070 / 0.090 / 0.091 |
| control: landscape_removed | 0.0000 ± 0.0000 | — |
| control: landscape_shuffled | 0.1779 ± 0.1736 | 0.149 / 0.021 / 0.364 |
| control: landscape_shuffled_across_wells | 0.2535 ± 0.2602 | 0.021 / 0.205 / 0.535 |
| control: random_atoms | 0.2309 ± 0.1854 | 0.021 / 0.300 / 0.372 |
| column: BLIND-SEQUENTIAL (planner) | 0.9128 ± 0.0578 | 0.872 / 0.888 / 0.979 |
| column: BLIND-COLOUR (planner) | 0.3477 ± 0.1895 | 0.174 / 0.550 / 0.320 |
| column: CODEC CEILING | 1.0000 ± 0.0000 | — |

### 4.3 `unseen n-way` — **the row the instrument exists for**

| arm | mean ± sd | per seed |
|---|---|---|
| **CLU (single design)** | **0.6911 ± 0.1931** | 0.802 / 0.468 / 0.803 |
| *diagnostic: transformer @ 10×* | *0.4137 ± 0.0659* | *0.338 / 0.456 / 0.447* |
| *diagnostic: lstm @ 10×* | *0.4122 ± 0.0495* | *0.359 / 0.421 / 0.457* |
| *diagnostic: mlp @ 10×* | *0.3884 ± 0.0507* | *0.332 / 0.401 / 0.431* |
| rival: transformer (matched) | 0.3304 ± 0.1197 | 0.366 / 0.197 / 0.429 |
| rival: mlp (matched) | 0.1929 ± 0.1342 | 0.324 / 0.056 / 0.198 |
| rival: lstm (matched) | 0.1742 ± 0.2139 | 0.000 / 0.110 / 0.413 |
| **arm: BLIND-SEQUENTIAL (learned)** | **0.1485 ± 0.2572** | 0.000 / 0.000 / 0.446 |
| **arm: BLIND-COLOUR (learned)** | **0.0000 ± 0.0000** | 0.000 / 0.000 / 0.000 |
| floor: lookup **cross-maze WORDS** | **0.0000 ± 0.0000** | — |
| floor: lookup cross-maze raw r1 | 0.0005 ± 0.0008 | 0.000 / 0.000 / 0.001 |
| floor: fragment cross-maze r1 | 0.0206 ± 0.0175 | 0.040 / 0.016 / 0.006 |
| floor: lookup WITHIN-maze raw r1 *(not transfer)* | 0.1046 ± 0.0032 | 0.104 / 0.108 / 0.102 |
| control: landscape_removed | 0.0000 ± 0.0000 | — |
| control: landscape_shuffled | 0.2945 ± 0.2629 | 0.367 / 0.003 / 0.513 |
| control: landscape_shuffled_across_wells | 0.1887 ± 0.1758 | 0.218 / 0.000 / 0.348 |
| control: random_atoms | 0.2738 ± 0.2481 | 0.338 / 0.000 / 0.484 |
| column: BLIND-SEQUENTIAL (planner) | 0.9135 ± 0.0104 | 0.903 / 0.914 / 0.924 |
| column: BLIND-COLOUR (planner) | 0.5597 ± 0.0080 | 0.556 / 0.569 / 0.554 |
| column: BLIND-A / BLIND-FACTOR (planner) | 0.5267 / 0.5781 | — |
| column: CODEC CEILING | 1.0000 ± 0.0000 | — |

### 4.4 `value hold-out` (never-seen colour — **scored, no refusal display**)

| arm | mean ± sd | per seed |
|---|---|---|
| **CLU (single design)** | **0.5386 ± 0.1559** | 0.418 / 0.715 / 0.482 |
| *diagnostic: mlp @ 10×* | *0.5551 ± 0.0454* | *0.541 / 0.518 / 0.606* |
| *diagnostic: transformer @ 10×* | *0.4672 ± 0.0566* | *0.453 / 0.419 / 0.529* |
| *diagnostic: lstm @ 10×* | *0.3999 ± 0.0251* | *0.381 / 0.390 / 0.428* |
| rival: transformer (matched) | 0.4403 ± 0.1600 | 0.427 / 0.287 / 0.607 |
| rival: mlp (matched) | 0.1961 ± 0.0680 | 0.120 / 0.218 / 0.250 |
| rival: lstm (matched) | 0.1723 ± 0.1947 | 0.032 / 0.091 / 0.395 |
| arm: BLIND-SEQUENTIAL (learned) | 0.5260 ± 0.1236 | 0.415 / 0.659 / 0.503 |
| arm: BLIND-COLOUR (learned) | 0.2693 ± 0.0618 | 0.210 / 0.334 / 0.264 |
| floor: lookup **cross-maze WORDS** | 0.0543 ± 0.0044 | 0.056 / 0.050 / 0.058 |
| floor: lookup cross-maze raw r1 | 0.0711 ± 0.0053 | 0.074 / 0.065 / 0.074 |
| floor: fragment cross-maze r1 | 0.1143 ± 0.0003 | 0.114 / 0.115 / 0.114 |
| floor: lookup WITHIN-maze raw r1 *(not transfer)* | 0.1511 ± 0.0129 | 0.150 / 0.139 / 0.164 |
| control: landscape_removed | 0.0079 ± 0.0137 | 0.000 / 0.024 / 0.000 |
| control: landscape_shuffled | 0.1950 ± 0.0640 | 0.134 / 0.190 / 0.262 |
| control: landscape_shuffled_across_wells | 0.2552 ± 0.1161 | 0.231 / 0.153 / 0.382 |
| control: random_atoms | 0.2525 ± 0.1277 | 0.110 / 0.293 / 0.355 |
| column: BLIND-SEQUENTIAL (planner) | 0.9912 ± 0.0025 | — |
| column: BLIND-COLOUR (planner) | 0.5533 ± 0.0069 | — |
| column: CODEC CEILING | 1.0000 ± 0.0000 | — |

### 4.5 `rollout validity` — full oracle routes, chained GENERATED states
Source: `VAL-SMOKE.json → rollout`. 45 rollouts per seed, **mean route length 43.4
steps**, actions taken from the engine's own optimal route so nothing about the action
sequence is chosen by an arm. A rollout is **valid** iff every step is exactly right and
the goal is entered within budget; the state fed to step `t+1` is the arm's own generated
state, and the moment a step is wrong the rollout is dead.

| arm | validity | all-steps-exact | per-step exactness |
|---|---|---|---|
| column: ORACLE (planner) | 1.0000 | 1.0000 | 1.0000 |
| column: BLIND-A / BLIND-FACTOR (planner) | 1.0000 | 1.0000 | 1.0000 |
| column: BLIND-SEQUENTIAL (planner) | 0.2815 | 0.2815 | 0.9785 |
| column: BLIND-COLOUR (planner) | 0.1630 | 0.1630 | 0.9538 |
| *diagnostic: transformer @ 10×* | *0.0148* | *0.0148* | *0.9158* |
| floor: lookup cross-maze WORDS | 0.0148 | 0.0148 | 0.8986 |
| **CLU (single design)** | **0.0000** | **0.0000** | **0.9147** |
| arm: BLIND-COLOUR (learned) | 0.0000 | 0.0000 | 0.9147 |
| arm: BLIND-SEQUENTIAL (learned) | 0.0000 | 0.0000 | 0.8421 |
| rival: transformer (matched) | 0.0000 | 0.0000 | 0.7838 |
| rival: lstm / mlp (matched) | 0.0000 | 0.0000 | 0.4516 / 0.4025 |
| floor: lookup cross-maze raw r1 | 0.0000 | 0.0000 | 0.4347 |
| floor: fragment cross-maze r1 | 0.0000 | 0.0000 | 0.4067 |
| control: landscape_removed | 0.0000 | 0.0000 | 0.1481 |
| control: random_atoms | 0.0000 | 0.0000 | 0.4687 |

⚠ **Read the rollout row honestly.** At 43.4 chained steps a per-step exactness of
0.9147 gives `0.9147^43.4 ≈ 0.02`, so **validity near zero is arithmetic, not a
surprise**, and the row's dynamic range lives in `per_step_exactness`. The rollout row
also mixes hold-out classes — an oracle route walks through `seen`, `2-way`, `n-way` and
`value` configurations alike — so it is not comparable with the per-class rows.
An `L = 8` block is reported in each seed JSON as a declared diagnostic
(`ROLLOUT_L8_DIAGNOSTIC`).

---

## 5. WHAT THE SMOKE SAYS

### 5.1 The falsifier does NOT fire for a learned composition-blind policy ⭐
`PREREG.md` §4.4: the warning fires iff
`max(BLIND-COLOUR, BLIND-SEQUENTIAL) ≥ CLU − 0.05` on `unseen n-way`.

    CLU                              0.6911
    bar                              0.6411
    BLIND-SEQUENTIAL (learned)       0.1485    far below
    BLIND-COLOUR    (learned)        0.0000    far below

**The instrument measures composition.** `BLIND-COLOUR (learned)` is the same store, the
same head machinery, the same rows and the same codec, with **colour identity removed
from the request** — it still sees that there is a door, that the door is closed, that
the floor is tinted and that the pocket is full, and it scores **exactly 0.0000** on
`unseen n-way` on all three seeds. Since RULE A is `pocket == door colour` and RULE B is
`pocket == tint colour`, a request that cannot express a colour *match* cannot express
either rule, and the row says so.

### 5.2 ⚠ The instrument warning that DOES fire, and how to read it
`HARNESS-READY.json → instrument_warnings`:

> `composition_blind_PLANNER_scores_near_the_clu_on_unseen_nway` —
> `column: BLIND-SEQUENTIAL (planner)` 0.9135 vs CLU 0.6911.

**This is shipped with the harness, as the task file requires — and it is the weaker of
the two readings.** The `BLIND-*` **planner** columns are rule-**knowing** relaxed
oracles: they read the grid, they are never trained, and so they never have to generalise
to a held-out combination at all. They are an **upper bound** on what a composition-blind
policy could do, not a policy. The falsifier the task file names is the **learned**
arm, §5.1, and it does not fire. Both families are printed and each is labelled.

That said, the planner columns are still informative and they are still **discriminating**
on `unseen n-way`: `BLIND-COLOUR (planner)` 0.5597, `BLIND-A (planner)` 0.5267,
`BLIND-FACTOR (planner)` 0.5781 against `ORACLE` 1.0000 — i.e. a relaxed oracle that
drops one rule loses ~45 points on that row. `BLIND-SEQUENTIAL (planner)` stays at 0.9135
because the pre-vs-post-pocket distinction bites only on the tinted-door configuration,
which is 226–265 of the ~700 n-way rows.

### 5.3 The cross-maze lookup floor is a printed floor, and it is the right one ⭐⭐
Every scored maze is a random maze never seen at train. The cross-maze floor over the
five words is:

| row | cross-maze WORDS floor | CLU |
|---|---|---|
| `seen` | **0.9967** | 0.9709 |
| `unseen 2-way` | **0.0000** | 0.3932 |
| `unseen n-way` | **0.0000** | 0.6911 |
| `value hold-out` | **0.0543** | 0.5386 |

`HARNESS-READY.json` correctly fires `cross_maze_lookup_floor_scores_near_the_clu` on
`seen` — **on that row the result is retrieval, not composition, and `seen` is a sanity
row, not a claim.** On every held-out row the floor is at or near zero while the CLU is
not, which is the whole point of the instrument. The **within-maze** column
(0.2903 / 0.0836 / 0.1046 / 0.1511) is built from the val mazes' *natural-distribution*
rows and is **DECLARED NOT transfer** — it is printed only as the ceiling a lookup
reaches when the maze repeats. ⛔ The lookup table is a printed floor, never a rival.

### 5.4 Naive vs naive, and the capacity diagnostic
Rivals are matched on the CLU's **read-path float count** (store atoms + address book).
The **memory ledger is printed separately and matched by nothing.**

| seed | CLU read floats | achieved mlp / lstm / transformer | residual |
|---|---|---|---|
| 0 | 4,200 | 4,100 / 4,290 / 3,753 | −2.4 % / +2.1 % / **−10.6 %** |
| 1 | 4,008 | 4,100 / 4,290 / 3,753 | +2.3 % / +7.0 % / **−6.4 %** |
| 2 | 8,016 | 8,216 / 8,255 / 8,385 | +2.5 % / +3.0 % / +4.6 % |

⚠ The transformer's residual is **−10.6 %** on seed 0 and **−6.4 %** on seed 1: at a
4 k target the smallest transformer step is coarse, so it is matched *to within 11 %*,
which is not matched. **Stated, not glossed.** It is the smaller side, so the transformer
is if anything under-provisioned relative to the CLU on those seeds.

Because Config-A's read path is small, the matched rivals are width-4 to width-11 trunks
on a 636-dim observation. The obvious objection is starvation, so the harness also prints
a **capacity diagnostic**: the same three trunks at **10× the read-path budget**
(39 k–80 k params), same observation, same frozen encoder, same rows, same 8,000
gradient steps. On `unseen n-way` the best 10× trunk reaches **0.4137** against the CLU's
**0.6911** — the margin is **not** a starvation artefact. ⛔ This is not COMMON.md's
forbidden "inflated-rival row": that prohibition is against matching a rival to the CLU's
**memory ledger**, which is never done here.

### 5.5 Charter-C — say which claim is available, per row ⭐
`PREREG.md` §4.3: all four controls must satisfy `control ≤ CLU − 0.15`.

| row | CLU | bar | removed | shuffled | shuffled-across-wells | random_atoms | verdict |
|---|---|---|---|---|---|---|---|
| `seen` | 0.9709 | 0.8209 | 0.0203 | 0.4678 | 0.5351 | 0.4996 | **STORE claim available** |
| `unseen 2-way` | 0.3932 | 0.2432 | 0.0000 | 0.1779 | **0.2535** ✗ | 0.2309 | **INTERFACE claim only** |
| `unseen n-way` | 0.6911 | 0.5411 | 0.0000 | 0.2945 | 0.1887 | 0.2738 | **STORE claim available** |
| `value hold-out` | 0.5386 | 0.3886 | 0.0079 | 0.1950 | 0.2552 | 0.2525 | **STORE claim available** |

On `unseen 2-way` `landscape_shuffled_across_wells` sits at 0.2535 against a bar of
0.2432 — it misses by 0.0103. **On that row this is an INTERFACE claim, not a store
claim**, and it is said plainly rather than averaged away.

**⭐⭐ The saturation ratio is the knob, and it is the instrument's own limit.** The
next-state code has only **23–31 distinct values in train** and **450 in the enumerated
alphabet**. A random projection of a few dozen well-separated points into a handful of
dimensions preserves their separation (Johnson–Lindenstrauss), so a random-atom store can
stay decodable although its content was never fitted. Measured across a latent sweep
(`logs/probe_latent.log`, `seen` row, worst of the four controls):

| latent | K | atoms | span = atoms / d_k | CLU seen | worst control | store claim? |
|---|---|---|---|---|---|---|
| 16 | 4 | 3 | 0.750 | 0.9813 | 0.9394 | no |
| 24 | 4 | 3 | 0.500 | 0.9989 | 0.9256 | no |
| 32 | 8 | 3 | 0.750 | 0.9901 | 0.9868 | no |
| 32 | 4 | 3 | 0.375 | 0.9829 | 0.8545 | no |
| 64 | 8 | 3 | 0.375 | 0.9752 | 0.8848 | no |
| **64** | **8** | **2** | **0.250** | **0.9741** | **0.6926** | **yes** |
| **64** | **4** | **3** | **0.188** | **0.9725** | **0.7697** | **yes** |
| **64** | **4** | **2** | **0.125** | **0.9047** | **0.5052** | **yes** |

A store claim on CMAZE-T is available **only below about span 0.25**. That is why the
four charter-C controls were made part of the Config-A selection bar (PREREG A1.2) and
why the config grid spans the saturation ratio.

### 5.6 The structural result the rows show
`unseen 2-way` (0.3932) is **harder** for the CLU than `unseen n-way` (0.6911), and the
reason is mechanical: the request design is `[bias | main | pair]`. Holding out a
`(pocket=a, door=b)` pair leaves that **pair cell untrained**, so the read falls back to
main effects that cannot express a colour match. Holding out an n-way 4-tuple leaves its
constituent **pairs** in train, so the pair block carries it. The same pattern shows up
in the learned blind arms: `BLIND-SEQUENTIAL (learned)` — main effects only, no
interaction terms — **beats** the full CLU on `unseen 2-way` (0.5059 vs 0.3932, it has no
untrained pair cell to be dragged by) and **loses badly** on `unseen n-way`
(0.1485 vs 0.6911, where the interaction terms are exactly what transfers).

**That is the compositional claim in one line: the pair block generalises to unseen
n-way combinations and not to unseen 2-way combinations, and CMAZE-T separates the two.**

---

## 6. DELETION ROWS

Source: `val_seed{0,1,2}.json → DELETION`. Two deletions, both measured against the
**surviving** colours. The deleted colour is **colour 1**, an ordinary TRAIN colour —
the value hold-out colour 4 is already absent from train and could not be deleted from it.

* **VALUE**: every transition carrying colour 1 in **any** rule-bearing role.
* **CONJUNCTION**: colour 1 **∧** tinted floor — rows whose TINT word is 1.

### 6.1 Store-scoped exactness — ✅ measured, not asserted
> *the store after deletion equals the store **fitted without** the rows — per-well
> sufficient statistic `S_k`, and the deployed atoms derived from it — at the **same**
> well partition.* Bar: `max relative delta ≤ 1e-5`.

| seed | deletion | rows deleted / kept | `S_k` max rel. delta | deployed atoms max rel. delta | pass |
|---|---|---|---|---|---|
| 0 | value | 8,598 / 11,620 | 7.799e-16 | **0.000e+00** | ✅ |
| 0 | conjunction | 3,361 / 16,857 | 7.727e-16 | **0.000e+00** | ✅ |
| 1 | value | 8,185 / 11,262 | 8.783e-16 | **0.000e+00** | ✅ |
| 1 | conjunction | 2,169 / 17,278 | 7.165e-16 | **0.000e+00** | ✅ |
| 2 | value | 6,937 / 10,681 | 1.093e-15 | **0.000e+00** | ✅ |
| 2 | conjunction | 2,238 / 15,380 | 8.451e-16 | **0.000e+00** | ✅ |

**Deletion language, stated exactly.** The exactness is **store-scoped**. The frozen
codec and the frozen state encoder are declared **OUTSIDE** the deletable object and are
not touched by either deletion. The post-deletion adaptation is **light and data-free**:
the head's retained statistics are downdated (`A -= Φ_dᵀΦ_d`, `Mx -= Φ_dᵀX_d`) alongside
the store's (`S_k -= X_dᵀX_d`, `t_k -= ΣX_d`, `n_k -= m`) and the **same closed form** is
re-solved — no pass over the retained pool, no item kept, nothing read that the caller
did not hand in. ⛔ It is **not** "no changes".

### 6.2 Composition on the SURVIVING colours, and the adaptation cost printed

Each cell is `(after deletion by downdate, fitted WITHOUT the rows)` for the CLU, and
`(retrain-without, as-trained)` for the rival:

| seed / deletion | row | CLU: downdate / fitted-without | rival: retrain-without / as-trained |
|---|---|---|---|
| 0 value | seen | 0.9530 / 0.9530 | 0.5163 / 0.2846 |
| 0 value | 2-way | 0.0451 / 0.0451 | 0.2029 / 0.1332 |
| 0 value | n-way | 0.4159 / 0.4159 | 0.3972 / 0.2243 |
| 0 conj. | seen | 0.9688 / 0.9688 | 0.4182 / 0.2846 |
| 1 value | seen | 0.9470 / 0.9470 | 0.5605 / 0.5505 |
| 1 value | n-way | 0.2514 / 0.2514 | 0.0276 / 0.0304 |
| 1 conj. | n-way | 0.4475 / 0.4475 | 0.0525 / 0.0304 |
| 2 value | seen | 0.9553 / 0.9553 | 0.4496 / 0.4923 |
| 2 value | n-way | 0.6561 / 0.6561 | 0.2566 / 0.1693 |
| 2 conj. | 2-way | 0.3978 / 0.3978 | 0.4165 / 0.2930 |

⭐ **The CLU's downdated read equals its fitted-without read to four decimals on every
row, every deletion and every seed** — the functional face of §6.1.

**Adaptation cost, printed for both sides:**

| | CLU (downdate) | rival (retrain-without) |
|---|---|---|
| wall seconds | **0.023 – 0.119 s** | **8.9 – 18.0 s** |
| gradient steps | **0** | **8,000** |
| examples seen | **0** | **1,920,000** |
| passes over retained data | **0** | all of them — the whole surviving pool is re-read every batch |
| items kept | **no** | yes |

The rival **as-trained** row is printed beside them, per the charter: the rival has no
other way to delete than to retrain without.

---

## 7. LEDGERS — every byte figure with its convention named

Source: `val_seed{0,1,2}.json → clu_ledger`. **n_seeds = 3.**

| seed | Config-A (latent, K, atoms, ridge) | span | read floats = atoms + head | deployed bytes (float32) | sufficient statistic (float64) |
|---|---|---|---|---|---|
| 0 | 128, 4, 3, 1.0 | 0.094 | 4,200 = 384 + 3,816 | **16,800 B** | 2,244 floats = **17,952 B** |
| 1 | 64, 4, 3, 1.0 | 0.188 | 4,008 = 192 + 3,816 | **16,032 B** | 612 floats = **4,896 B** |
| 2 | 128, 8, 3, 1.0 | 0.188 | 8,016 = 384 + 7,632 | **32,064 B** | 1,224 floats = **9,792 B** |

**Convention, named**: *deployed bytes* = 4 × (atom floats + address-book floats) — the
object that produces a read. The *float64 sufficient statistic* is what buys **exact**
deletion and is declared beside it, **never netted in and never matched by a rival**.
Request design width 318 (`[bias | 27 main | 290 pair]`).
`assert_head_request_only` passes on every seed: `head_read(heads, store, Φ)` has **no
feature argument**, so charter B.3 holds by signature.

**The two frozen objects, both outside the deletable object:**

| seed | state encoder (conv AE on maze images, reconstruction only) | codec (next-state code ↔ latent) |
|---|---|---|
| 0 | BCE 0.00012, pixel acc 1.00000, on-pixel recall 1.0000 | latent 128, loss 0.000228 |
| 1 | BCE 0.00003, pixel acc 1.00000, on-pixel recall 1.0000 | latent 64, loss 0.000198 |
| 2 | BCE 0.00008, pixel acc 1.00000, on-pixel recall 1.0000 | latent 128, loss 0.000080 |

**Pipeline ceiling** (encode → decode → exact judge): **1.0000 on every row and every
seed** after AMENDMENT 2. Before it, the value row's ceiling was 0.4963 ± 0.0368 — see §9.

---

## 8. CONFIG-A — how it was chosen

`PREREG.md` §4.2 as amended: **val `exact` on the `seen` row only**, with the bar
`seen ≥ 0.90` **and** `max(all four charter-C controls) ≤ CLU − 0.15` on `seen`; ties by
fewest read floats. ⛔ The held-out rows and the rollout row were **not** consulted.
The full 16-config sweep per seed is in `val_seed*.json → clu_val_sweep`.

| seed | Config-A | val `seen` | bar met |
|---|---|---|---|
| 0 | latent 128, K 4, atoms 3, ridge 1.0 | 0.9745 | ✅ |
| 1 | latent 64, K 4, atoms 3, ridge 1.0 | 0.9762 | ✅ |
| 2 | latent 128, K 8, atoms 3, ridge 1.0 | 0.9620 | ✅ |

⚠ **Config-A came out different on each seed, and that is a large part of the
seed-to-seed spread** on the held-out rows (`unseen n-way` 0.802 / 0.468 / 0.803). ⭐ The
recommendation to wave 16 is to **pin one Config-A across seeds**. It is left unpinned
here because pinning it would mean choosing from these seeds' numbers, and this spoke
reports rather than tunes.

---

## 9. WHAT WENT WRONG AND WAS FIXED — with the numbers that caught it

All of these are in `LOG.md` (D8–D17) and `PREREG.md` amendments 1–3. Every superseded
run is preserved: `preamendment2/` (smoke 1) and `amendment23/` (smoke 2).

| # | defect | the number that caught it | fix |
|---|---|---|---|
| D8 | `landscape_shuffled` as a **row** permutation is a no-op by construction — the head is fitted after the control and absorbs it exactly | control 0.8195 vs CLU 0.8195 on `unseen n-way`, identical to every digit | permute atom **channels** instead |
| D9 | the `BLIND-*` planner columns are rule-knowing oracles, not policies | planner 0.9098 vs CLU 0.8195 on `unseen n-way`, which says nothing about composition | added two **learned** blind arms (§5.1) |
| D10 | rollout at fixed `L = 8` could not reach the goal of a ~45-step route | validity 0.0000 for every arm by construction | full oracle routes; `L=8` kept as a declared diagnostic; lockstep batching took the row from ~700 s to 0.3 s |
| D11 | the within-maze lookup column read its own answer | 1.0000 on `unseen n-way` | table built from the val mazes' natural-distribution rows |
| D12 | charter-C controls do not collapse above span ≈ 0.25 (Johnson–Lindenstrauss on a 26-point code cloud) | worst control 0.9394 at span 0.75 vs 0.5052 at span 0.125 | controls added to the selection bar; grid spans the saturation ratio |
| D13 | rivals at 200 steps were emitting the majority class | per-step exactness 0.0000 on an all-legal rollout | 8,000 gradient steps for every rival |
| **D14** | **the frozen codec capped the value row** | codec ceiling **0.4963 ± 0.0368** on `value`; train rows realise 23–31 of the 45 codes the val rows need | codec fitted on train codes **+ the enumerated 450-code emission alphabet** (a statement about the format, not the data); ceiling now **1.0000** on every row |
| **D15** | **store-scoped exactness compared two different objects** — the reference recomputed the ward partition on the surviving rows | `max_relative_delta = 1.26` while the per-well counts matched exactly (11620 = 11620) | `well_channels=` override; now 7.8e-16 … 1.1e-15, atoms exactly 0 |
| D16 | the displacement block admitted "displaced by P", which is not a displacement | `KeyError: 'P'` once the codec covered the full alphabet | block narrowed to `{stay, N, E, S, W}`; `code_dim` 22 → 21 |
| D17 | matched rivals are tiny because the CLU's read path is tiny | width-4 to width-11 trunks on a 636-dim observation | **capacity diagnostic** at 10× the read-path budget (§5.4) |

⚠ **D14, D15 and D17 were found *after* the first smoke's numbers existed.** They are
recorded as dated amendments, the superseded numbers are preserved and quoted above, and
nothing was replaced silently.

---

## 10. WHAT DID NOT RUN — declared, not reported as zeros

1. **Any SCORED run.** ZERO by the task file. No ledger row is claimed, no RESULT row.
2. **The TEST maze block.** Built nowhere, evaluated nowhere. Generator seed ranges
   `[2.30M, 2.40M]`, `[2.70M, 2.80M]`, `[3.10M, 3.20M]` are recorded in each seed JSON
   under `banks.test_RESERVED_NOT_EVALUATED`. It is wave 16's to spend.
3. **A refusal display.** Refusal displays are retired. The value hold-out is **scored**
   and no arm carries a refuse token. The `REFUSE` token exists in `w9_harness` and is
   deliberately unused here.
4. **Pinning Config-A across seeds.** DECLARED NOT RUN (§8): it would require choosing
   from these seeds' numbers. Recommended to wave 16.
5. **`k ≠ 4`, `h/w ≠ 7`, any generator parameter other than the defaults.** The generator
   is called at ITS OWN DEFAULTS. Only the door-tint layer (§2) changes the distribution.
6. **A CMAZE-T variant without tinted doors.** Not run as a scored arm; the reason it is
   not the shipped instrument is measured in §2 and `LOG.md` D7 (the 160-door-cell count
   and the vacuity of BLIND-SEQUENTIAL without it).
7. **Any change to `w8_envscout`, `w9_harness` or `w9_maze`.** Imported only; SHA-256 in
   `HARNESS-READY.json → provenance`.
8. **GPU and cluster.** Local CPU only, three concurrent seed processes, OMP/MKL/OpenBLAS
   threads pinned to 4. Wall clock: 660–705 s per seed.
9. **A per-class breakdown of the rollout row.** The rollout mixes hold-out classes
   (§4.5); it is reported as a mixture and named as one.

---

## 11. WHAT WOULD HAVE PROVED THE IDEA WRONG

1. **A learned composition-blind policy scoring near the CLU on `unseen n-way`.** It
   does not: 0.0000 and 0.1485 against 0.6911, bar 0.6411 (§5.1). *This was the
   falsifier and it was survived.*
2. **The cross-maze lookup floor scoring near the CLU on a held-out row.** It does not:
   0.0000 / 0.0000 / 0.0543 against 0.3932 / 0.6911 / 0.5386 (§5.3). It **does** on
   `seen`, and `HARNESS-READY.json` says so — that row is retrieval and is a sanity row,
   not a claim.
3. **Charter-C controls failing to collapse.** They collapse on `seen`, `unseen n-way`
   and `value`, and **fail by 0.0103 on `unseen 2-way`**, where the result is therefore
   an **INTERFACE** claim (§5.5). Said plainly.
4. **The five words failing to determine the transition.** 0 clashes over 164,767 rows
   and 1,081–1,143 distinct word tuples (§3.3b). Had this failed, hold-outs over word
   combinations would not have been statements about the transition.
5. **Rule B never rendering an action illegal.** It does: 0.070–0.073 of the stratified
   bank, 0.010–0.016 of the natural distribution (§3.4). Reported, never aborted on.
6. **The rivals being starved.** At 10× the read-path budget the best rival still reaches
   only 0.4137 on `unseen n-way` against the CLU's 0.6911 (§5.4).

**The single clearest way to break this instrument** would be to show that the five
request words are not the transition's factors — e.g. by finding a `(state, action)` pair
whose word tuple appears in train with a different outcome. The audit for exactly that
runs on every seed and returns 0.

---

## 12. WHAT WAVE 16 SHOULD DO FIRST

1. **Pin one Config-A across seeds** (§8). The per-seed selection is the dominant source
   of spread on the held-out rows.
2. **Keep the store below span ≈ 0.25** or no store claim is available (§5.5).
3. **Use the enumerated-alphabet codec** (already the default) or the value row measures
   the codec (§9 D14).
4. **Spend the test block**, which this spoke never touched.
5. Treat `seen` as a sanity row — the cross-maze WORDS floor is 0.9967 there.

---

## 13. ARTEFACT INDEX

| artefact | what it is |
|---|---|
| `HARNESS-READY.json` | ⭐ the acceptance artefact, `"status": "ready"` |
| `VAL-SMOKE.json` | the 3-seed smoke, all rows, every arm |
| `val_seed{0,1,2}.json` | per-seed raw measurements |
| `PREREG.md` | pre-registration + amendments 1–3 |
| `LOG.md` | narrative, every design decision D1–D17 |
| `FINDINGS.md` | this file |
| `logs/seed{0,1,2}.log` | the final run's console output |
| `logs/dry*.log` | the dry runs that surfaced D8–D13 |
| `logs/probe_latent.log` | the saturation-ratio sweep behind §5.5 |
| `logs/probe_delete.log` | the store-scoped exactness re-measurement behind D15 |
| `preamendment2/` | smoke 1, preserved whole |
| `amendment23/` | smoke 2, preserved whole |
| `PIDS.txt` | every PID this spoke launched |
| `START_UTC.txt`, `END_UTC.txt` | wall clock |
| `w15-cmaze-t-harness.bundle` | the branch, whole, `git bundle verify`-ed |
