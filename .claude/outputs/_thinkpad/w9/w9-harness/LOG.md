# w9 spoke/maze-harness — LOG

Branch `w9/harness`, base `base/iter8 = fa60b15`. Worktree `/c/chlu/worktrees/CHLU-w9-harness`.
Output `/c/chlu/outputs/w9-harness/`. Start UTC in `START_UTC.txt` (2026-09-21T03:57:43Z).

**Dial: the instrument. Falsified if the judge needs a tolerance anywhere.**
It does not. See 2026-09-21 below.

---

## 2026-09-21

- Read the task file and env-scout's shipped CMAZE-1 (`w8_envscout/maze_colour.py`,
  `screen.py`, `sensitivity.py`, already folded into `base/iter8`). **The environment was
  not rebuilt.** `generate`, `solve`, `judge` and the blind-planner flags are used as
  shipped.
- **The judge is exact, with no tolerance.** `w9_harness/judge_maze.py` re-exports
  `w8_envscout.maze_colour.judge` **unwrapped** as `JUDGE` — every arm in this harness is
  scored by literally that function object — and adds a four-part audit:
  - **A source.** AST scan of `judge` and `_pocket`: no float literal, no true division,
    no `isclose`/`allclose`/`atol`/`rtol`/`eps`/`tol`/`round`/`float`, no float attribute.
    PASS.
  - **B types.** Replay with an `int` subclass that raises the moment it is compared with,
    or coerced to, a float. PASS. The tracer is shown non-vacuous by a test that makes it
    fire.
  - **C redundant.** A second replay (`judge_ref`) written from the RULES in the docstring
    rather than from the judge's code; the two must agree on verdict, reason *including the
    colour suffix*, and step index. PASS, 0 disagreements.
  - **D oracle.** On every instance: the exact BFS optimum is accepted at exactly its own
    length; the optimum truncated by one step is rejected; a route one over budget is
    rejected *as over-budget*; every single-action mutation is decided identically by both
    judges. PASS, 0 failures.
  - First run of C failed 9/250 — my reference judge reported `locked-door` where the
    shipped judge reports `locked-door-c1`. That was my reason vocabulary, not the judge;
    fixed by carrying the colour, which makes the agreement check *stricter* (it now also
    compares which colour was blamed).
  - **`judge_exact: true`, `tolerance_used: null`.** Nothing downstream inherits an
    approximation.
- **The stepper is the judge's twin, and is audited as such.** A rollout needs a transition
  function, which is a second piece of code that knows the rules. `w9_harness/dynamics.py`
  keeps that from becoming a second scorer: the stepper never produces a score, and
  `audit_stepper` checks that its accept/reject and its stop index match `JUDGE` on every
  random route. PASS, 0 disagreements.
- **One shared encoder, reconstruction only.** `render.py` turns an instance into a
  `3+3k`-plane binary image of the **initial** maze (floor, start, goal, door colour, tint
  colour, key home). `encoder.py` trains a 3-layer conv autoencoder on those images with a
  per-pixel Bernoulli loss whose only arguments are `(model, images)` — no actions, no
  routes, no judge — then **freezes** it before any arm exists. `independence_report()`
  prints the loss source and asserts it sees no labels and no judge. Every arm reads the
  same frozen features.
- **One shared observation.** `features.obs` = 5×5 agent-centred patch of the encoder
  feature map ++ the same map average-pooled to 5×5 (a global view; a local patch alone
  cannot tell an arm where the far half of the maze is) ++ 29 numbers the image cannot
  carry (keystate, pocket, goal delta, position, budget, key-still-there). 431 dims at
  `feat_dim=8`, `k=4` after the budget fix below (429 before it). No arm gets a private
  channel.
- **One shared action head.** `d_model` is **fixed at 32 for every arm** and the head is
  built from a single module-level PRNG key, so before training all heads are
  byte-identical arrays of identical shape (test asserts both). The matched-parameter
  search moves the *trunk* width, never the head. The heads are then trained per arm —
  they have to be, since a head frozen on one trunk is a head tuned for that trunk — and
  that part is stated as not-shared rather than glossed. An early version let `d_model`
  follow the trunk width; that made the heads different shapes and the report said
  `identical_shape_across_arms = False`. Fixed by pinning `d_model`.
- **Matched parameters.** `arms.match_widths` searches trunk width (and, for the
  transformer, depth ∈ {1,2,3} and heads ∈ {2,4}, because one width knob alone leaves
  several percent on the table and "matched to within 7%" is not matched). It reports the
  counts it **achieved** and the residual.
- **Reported columns are wired as columns.** `columns.py` runs env-scout's blind planners
  (rule-B dropped, rule-A dropped, colours collapsed, A×B local only, key order from A)
  unchanged, and `tabular.fragment_column` runs env-scout's fragment lookup unchanged.
  Both are printed with validity and score from the exact judge. Nothing aborts on them and
  no instance is filtered because of them; a test greps the driver for an abort keyed on a
  column value.
- **Table floor**, plus an honest extra. `table` is the memorisation floor: a lookup from a
  raw, uncanonicalised local code patch + pocket + goal-sign to the most frequent oracle
  action — deliberately the one arm that does *not* consume the encoder. `table-enc` is the
  same lookup policy keyed on the **shared encoder's** sign-quantised features, reported
  beside it, so "can a lookup policy use the encoder at all" is answered rather than
  assumed. Both print coverage.
- **The supervised refuser.** A transformer with a REFUSE token, matched to the same
  parameter target, trained on possible instances (oracle actions) plus **impossible**
  instances (REFUSE). Impossible instances are *exactly certified*, not sampled:
  - `budget` — budget set one below the exact BFS optimum, so no route can fit;
  - `legality` — tints added until BFS over (cell, keystate) finds nothing at all.
  `certify_impossible` re-derives both with the solver; a test also brute-forces 200 random
  routes per instance and finds none accepted. `assert_unseen` checks the impossible seeds
  are disjoint from every other arm's data — **our line, "no impossible example was ever
  seen", is a checked property of the code, not a claim in prose.**
- **PARTIAL `HARNESS-READY.json` published at 2026-09-21T04:36:37Z** — judge exact, encoder
  trained, MLP + table floor + refuser wired, both reported columns present, seed 0 only.
  hub/maze is unblocked from that moment.
- **Box contention.** `scripts/w9_maze_bank.py` (hub/maze) is running three processes on
  this 3-thread box. Sizing was cut accordingly (`std` scale: 500 train / 200 eval
  instances, 1200 encoder steps, 2000 BC steps) and one performance bug was fixed: the
  rollout built its `eqx.filter_jit` from a fresh lambda each call, so every rollout paid a
  full recompile. Hoisted to a module-level jitted `_forward(policy, X)`.


### Later on 2026-09-21 — what the three-seed sweep found

- **Two declared tiers, never averaged together.** `cmaze1` is env-scout's adopted
  configuration unchanged. `cmaze1-easy` is the *same generator* with `h=w=4, k=2`
  (~18-step optima instead of ~43) and the *same exact judge*. The easy tier exists
  because with zero tolerance a 43-step route needs 43 consecutive correct actions, and
  every learned arm reads 0.0000 — an instrument with no dynamic range tells the hub
  nothing.
- **Three defects found in my own instrument, all fixed, one not re-run:**
  1. *The action head was not one head.* `d_model` followed each trunk's matched width,
     so the heads had different shapes. Pinned `d_model=32`; all heads are now 165
     parameters from one PRNG key, byte-identical at init. Checked by test.
  2. *The observation hid the budget.* It carried only `t/budget` and `(budget-t)/budget`,
     which are 0 and 1 at t=0 for **every** instance — so a maze whose only defect is a
     budget below its optimum was indistinguishable from a solvable one at the moment a
     refusal is decided. The absolute budget is now in the observation.
  3. *The encoder is not reliably trained in one run.* At 1200 steps seed 1 stalled at
     BCE 0.0796 (linear probe AUC 0.61 — near chance; sign-quantised features collapsed
     to 3–5 distinct patterns for a whole grid) where seeds 0 and 2 reached 0.0050 and
     0.0038 (AUC 0.92, 0.80). Everything downstream reads those features. `encoder.train`
     now takes `restarts` / `accept_bce`, **selected on reconstruction loss alone** so the
     encoder stays arm-agnostic. Verified across all six seed × tier combinations:
     worst-case BCE **0.101 → 0.010**, worst-case mean probe AUC **0.441 → 0.754**, and at
     no extra cost (the acceptance threshold is met on the first restart every time).
     ⛔ **DECLARED NOT-RUN:** re-running the arm sweep with the fixed encoder. Wall clock;
     the box was shared with hub/maze throughout. Seed 1's arm numbers therefore inherit a
     degraded encoder, and the tables say so.
- **A fourth bug was in the measuring instrument, not the thing measured:** the first
  encoder probe thresholded a ridge fit at 0.5 on planes that are 0.4% positive and read
  0.000 recall for facts that are perfectly present. Replaced with AUC and balanced
  accuracy at the best threshold.
- Two process mistakes worth recording: two sweep loops from a killed launch survived and
  wrote seed reports with the *old* observation width alongside the new ones (caught by
  checking `obs_dim` in every report before aggregating, and re-run); and a rollout built
  its jitted forward pass from a fresh lambda each call, paying a full recompile per
  rollout.

## FINAL REPORT

**Dial: the instrument. Falsified if the judge needs a tolerance anywhere. It does not.**

### 1. The judge is exact — this is the load-bearing result

`w9_harness.judge_maze.JUDGE` **is** `w8_envscout.maze_colour.judge`, re-exported
unwrapped, so every arm is scored by literally the same function object. Over 3 seeds ×
250 fuzzed routes = **750 judged routes** per tier:

| check | result |
|---|---|
| A source — float literal, true division, `isclose`/`atol`/`rtol`/`eps`/`tol`/`round`/`float` in `judge` or `_pocket` | **none** |
| B types — float contact while replaying under a tracing `int` (tracer shown non-vacuous by test) | **none** |
| C redundant — disagreements with an independently written replay, on verdict, reason *including the colour blamed*, and step index | **0 / 750** |
| D oracle — optimum accepted at its exact length, truncation rejected, one-over-budget rejected *as over-budget*, every single-action mutation decided identically by both judges | **0 failures** |
| stepper — the rollout's transition function vs. the judge | **0 / 750 disagreements** |

`judge_exact: true`, `tolerance_used: null`. **Nothing downstream inherits an
approximation.** The budget is an exact integer boundary: length L accepted, L−1 rejected,
on every instance tested.

### 2. Shared means shared, and the part that is not shared is named

- **Encoder — fully shared.** One conv autoencoder per seed; the loss takes `(ae, X)` and
  its identifiers are `{X, ae, dec, enc, jax, logits, mean, optax,
  sigmoid_binary_cross_entropy, vmap, x}` — no action, no route, no label, no judge
  (AST-checked, docstring excluded). Frozen before any arm is built; every arm reads the
  same arrays.
- **The encoder does not advantage an arm, and it carries what it should.** A linear probe
  from a cell's 8-dim feature back to that cell's image planes: mean AUC 0.92 / 0.61 /
  0.80 for seeds 0/1/2 before the fix, 0.92 / 0.84 / 0.82 after. Walls, doors, tints, key
  homes and the goal are all linearly readable. So `into-wall` and `pickup-on-empty-cell`
  failures are arms **failing to use** the features, not failing to have them — and
  whatever is present is present for all of them equally.
- **Action head — shared shape and shared initialisation, trained per arm.** `d_model=32`
  for every arm, one module-level PRNG key: 165 parameters, byte-identical at init,
  verified by test. Trained separately, because a head frozen on one trunk is a head tuned
  for that trunk. Stated, not glossed.
- **Matched parameters, achieved rather than claimed:** MLP 60047, LSTM 59965,
  transformer 59949, refuser 59982 against a 60000 target — within 0.09%. The transformer
  search covers depth and head count as well as width, because width alone leaves ~7% on
  the table and "matched to within 7%" is not matched.

### 3. The arms, scored by the exact judge (3 seeds, mean [min, max])

| | validity, cmaze1 | validity, cmaze1-easy | per-step acc (cmaze1 / easy) |
|---|---|---|---|
| MLP | 0.0050 [0.0000, 0.0100] | 0.0150 [0.0000, 0.0350] | 0.52 / 0.59 |
| LSTM | 0.0017 [0.0000, 0.0050] | 0.0183 [0.0000, 0.0400] | 0.75 / 0.72 |
| Transformer | 0.0000 | 0.0150 [0.0100, 0.0200] | 0.74 / 0.74 |
| **Table floor** (raw state) | 0.0000 | 0.0017 [0.0000, 0.0050] | — |
| Table-enc (encoder bits) | 0.0000 | 0.0017 [0.0000, 0.0050] | — |
| oracle | **1.0000** | **1.0000** | — |

**The learned arms sit at the floor under zero tolerance.** That is a real property of the
task at this training budget, not a plumbing failure: teacher-forced per-step accuracy is
0.52–0.75, and 0.75⁴³ ≈ 4 × 10⁻⁶. The dynamic range lives in the per-step accuracy and in
the exact diagnostics (`goal with the budget lifted`, `mean legal prefix / optimum`),
which are in `RESULTS.md`, and in the easy tier, where the ordering LSTM > MLP ≈
transformer > table floor is visible and every learned arm beats the table floor. On
`cmaze1` the learned arms are not distinguishable from the table floor on validity alone.

### 4. The reported columns (never abort conditions)

| column | cmaze1 | cmaze1-easy |
|---|---|---|
| blind-A (rule B dropped) | 0.1400 | 0.4983 |
| blind-B (rule A dropped) | 0.0000 | 0.0000 |
| blind-colour (colours collapsed) | 0.0933 | 0.3733 |
| blind-factor (A × B local only) | 0.2350 | 0.6467 |
| **blind-sequential** (strongest) | **0.5317** | **0.7983** |
| fragment-lookup r=1 / r=2 | 0.0000 / 0.0000 | 0.0033 / 0.0000 |

Fragment lookup reproduces env-scout's 0.0000 exactly, at 100% train coverage — it
retrieves the training distribution perfectly and still cannot compose. The headline
composition-blind number is the **strongest** blind planner on the run, not a relaxation
fixed after the fact. Nothing aborts or filters on either column, and a test greps the
driver for an abort keyed on a column value.

### 5. ⛔ The refuser does not work, and I can say why

**Read this before using a DISPLAY row.** The supervised refuser is the only comparator
for DISPLAY rows, and it does **not** separate possible from impossible:

| | refusal on impossible | refusal on possible | separation |
|---|---|---|---|
| cmaze1 | 0.8317 [0.6750, 1.0000] | 0.7850 [0.5600, 1.0000] | **+0.047 [0.000, 0.115]** |
| cmaze1-easy | 0.6217 [0.3500, 0.9950] | 0.6017 [0.3350, 1.0000] | **+0.020 [−0.005, 0.050]** |

Class weighting only moves the operating point (seed 1 refused 100% of everything), never
the separation. The cause is established rather than guessed: a ridge probe on the **same
shared observation at t=0**, using the restart-fixed encoder, gets

- possible vs **all** impossible: AUC **0.48–0.55** across all six seed × tier runs — chance;
- vs impossible-by-**budget**: 0.44–0.63;
- vs impossible-by-**legality**: 0.54–0.85 (extra tints change the image, so there is some
  signal);
- the budget scalar alone: 0.38–0.46 — **no giveaway**; the impossible set cannot be picked
  off by a shortcut.

So the failure is the **problem**, not the wiring: deciding "is this maze solvable within
B steps" is close to solving it, and a 60k-parameter supervised model at this budget does
not. **DISPLAY-row comparisons against this refuser are not yet informative.** What would
change it: a much larger refuser training budget, or an impossible family that is
detectable perceptually — and the second would be choosing the answer, so anyone who takes
it should declare it loudly.

Our line survives intact, and as a checked property rather than prose: `assert_unseen`
reports **0** impossible instances in any non-refuser arm's data across all seeds and both
tiers, and `certify_impossible` reports **0** certification failures over 2700 impossible
instances (exact: budget below the exact BFS optimum, or no BFS route at all; a test also
brute-forces 200 random routes per instance and finds none accepted). MLP, LSTM and
transformer have no refuse token by construction and score 0 on the impossible set —
reported, not hidden.

### 6. What the hub gets

`HARNESS-READY.json` — `status: ready`, `judge_exact: true`, `tolerance_used: null`, seeds
[0,1,2], all five baselines, both columns, both tiers, and three `CAVEAT_*` notes carrying
§5 and the encoder instability. `RESULTS.md` has the full tables including the diagnostics
and validity split by optimal length. `HARNESS.md` is the contract for scoring through the
instrument. `code/` is the instrument. `tests.txt` is 26/26 passing.

### 7. ⛔ Declared NOT-RUN

- Re-running the arm sweep with the **restart-fixed encoder**. The fix is implemented and
  verified at the encoder level across all six seed × tier combinations; the arm numbers
  above were produced with the single-run encoder, and seed 1's inherit a degraded one.
- A refuser at a training budget large enough to test whether the refusal task is learnable
  at all, rather than merely unlearnable at 60k parameters and 2000 steps.
- Any RL arm. Every arm here is behaviour cloning on exact oracle routes.
- Scales above `std` (500 train / 200 eval instances per seed). The box was shared with
  hub/maze's three-process bank build for the whole session.
