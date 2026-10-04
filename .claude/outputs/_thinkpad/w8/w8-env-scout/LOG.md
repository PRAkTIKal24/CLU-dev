# w8-env-scout — LOG

Spoke: `web-scout` (+ engineer read of candidate code). Task: `/c/chlu/tasks-w8/w8-spoke-env-scout.md`.
Worktree: `C:/chlu/worktrees/CHLU-w8-env-scout`. Outputs: `/c/chlu/outputs/w8-env-scout/`.
Dial: **the instrument**. Falsified if an adopted environment cannot give an **exact** validity check.

---

## 2026-09-20T16:36Z — start
- `START_UTC.txt` written `2026-09-20T16:36:40Z`. No pre-existing live START for this spoke (checked
  `/c/chlu/outputs/*/START_UTC.txt` — 20 other spokes/hubs have one, none is `w8-env-scout`).
- Verified instrument resolves inside my worktree:
  `chlu.__file__ = C:\chlu\worktrees\CHLU-w8-env-scout\chlu\__init__.py`. Python 3.11.9,
  numpy 2.4.1 / jax 0.9.0 / equinox 0.13.4 from the frozen venv.
- Probed for candidate packages **already installed**: `minigrid` NO, `gymnasium` NO, `gym` NO,
  `babyai` NO. So MiniGrid/BabyAI cannot be run locally and I am forbidden to `pip install`.
  ⇒ The MiniGrid/BabyAI assessment will be an **engineer read of the published source**, quoted,
  not a local execution. Recorded here so no downstream reader mistakes a read for a run.
- Screening law I am applying, from the task file (established by GYM-1 and GYM-2):
  1. **Legality**: at least one rule must render a route **ILLEGAL**, not merely more expensive.
  2. **Non-sufficiency of legality**: the A×B interaction must be worth enough that a
     **composition-blind** policy *cannot* score well (GYM-2 had infeasibility and still died at
     0.9367–1.0000 blind).
  3. Adopt on (a) random mazes/episode, (b) legality composing across agent × key/colour × door,
     (d) **exact** validity check; (c) budget desirable.

## 2026-09-20T16:45Z — MiniGrid / BabyAI: engineer read of the actual source
- Could not `pip install`; fetched the upstream source with `curl` instead and read it locally
  (`Farama-Foundation/Minigrid@master`): `core/world_object.py`, `minigrid_env.py`,
  `core/roomgrid.py`, `core/actions.py`, `envs/doorkey.py`, `envs/obstructedmaze.py`,
  `envs/babyai/unlock.py`, `envs/keycorridor.py`, `envs/blockedunlockpickup.py`.
- **LEGALITY — PASS, and it is real.** `Door.can_overlap()` returns `self.is_open`; a closed or
  locked door is *impassable*, not expensive. `Door.toggle()`:
  `if self.is_locked: if isinstance(env.carrying, Key) and env.carrying.color == self.color: ...
   return False`. So a route through a locked red door while not carrying a red key is **ILLEGAL**.
  This is categorically unlike GYM-1/GYM-2's speed limits. ✅ The *mechanism* is the right one.
- **SINGLE-CARRY — the thing that would make colour compose.** `step()` pickup branch:
  `if fwd_cell and fwd_cell.can_pickup(): if self.carrying is None: ...`. One slot. So holding key A
  forbids holding key B; colour × door × agent-state is a genuine 3-way coupling *in the engine*.
- ⚠ **BUT: in every shipped level, colour is REDUNDANT WITH REACHABILITY.** This is the trap.
  - `envs/doorkey.py::_gen_grid` hard-codes ONE colour:
    `self.put_obj(Door("yellow", is_locked=True), splitIdx, doorIdx)` and
    `self.place_obj(obj=Key("yellow"), ...)`. One door, one key, one colour ⇒ a colour-blind policy
    is *bit-identical* to the optimal policy. Composition-blind score on `MiniGrid-DoorKey-*` is
    **1.0000 by construction** — no measurement needed, the generator proves it.
  - `babyai/unlock.py::UnlockPickup`, `BlockedUnlockPickup`, `UnlockLocal`, `KeyInBox`, `Unlock`:
    all do `add_door(..., locked=True)` then `add_object(..., "key", door.color)` — exactly one key,
    whose colour is *defined as* the door's. Blind "pick up the key you can reach, toggle the locked
    door you can reach" wins. Colour never discriminates.
  - `babyai/unlock.py::UnlockToUnlock` is the only shipped *chain* (key B → door B → key A → door A,
    2 colours, single carry forces a drop). Still blind-solvable: the generator makes exactly one key
    reachable at each stage, so "grab the reachable key, try the reachable locked door" is optimal
    without ever reading a colour. And its map is a **fixed 1×3 room corridor** — fails (a).
  - `envs/obstructedmaze.py::ObstructedMaze_Full` is the *only* place where two differently coloured
    keys sit in one room (`for k in [-1,1]: add_door(*side_room, locked=True,
    color=self.door_colors[(i+k) % len(self.door_colors)], ...)` with `obj = Key(door.color)`
    placed by `self.place_in_room(i, j, obj)`). There colour does discriminate — but a wrong pick is
    **cheaply recoverable** (drop, pick the other), so blindness costs steps, not validity. That is
    the GYM-2 failure mode exactly: reward is `1 - 0.9 * (step_count / max_steps)`, so a blind
    policy still lands high. And `key_in_box=True` hides the colour until toggled, so even the
    *optimal* policy must sample.
- **(a) RANDOM MAZES — FAIL for the composing levels.** `RoomGrid` is a fixed `num_rows × num_cols`
  lattice of rooms; randomness is door position within a wall and object position within a room.
  `DoorKey` randomises only `splitIdx`/`doorIdx` on a 2-region split. Neither is a random maze.
- **(d) EXACT VALIDITY — PASS, if validity is defined as replay.** Dynamics are deterministic given
  the seed; a proposed action sequence is valid iff replaying it terminates with `reward > 0`.
  No tolerance anywhere. But note the *score* is step-scaled, which is what let GYM-2's blind
  policy sit at 0.9367.
- **Dependency cost, recorded because it is an adoption fact:** `minigrid`/`gymnasium` are NOT in the
  frozen bundle (`import minigrid` → ModuleNotFoundError). Adopting MiniGrid means changing a frozen
  instrument's dependency set for wave 9.
- ⇒ **Preliminary verdict: MiniGrid/BabyAI fit on paper and their door–key legality DOES compose
  across colour in the engine, but NO SHIPPED LEVEL REQUIRES IT.** The generators make the correct
  key the only reachable key. This is precisely the failure the task told me to look for.
  Next: procedural-maze landscape, then (if nothing fits) build + measure.

## 2026-09-20T16:58Z — procedural-maze landscape, screened against the same law
Read source, not just abstracts, wherever the candidate had source I could fetch.
- **Procgen `heist`** (`procgen/src/games/heist.cpp`). Random maze per level
  (`maze_gen->generate_maze_with_doors(num_keys)`), 3 colours, and **real legality**:
  `is_blocked_ents: if (target->type == LOCKED_DOOR) return !has_keys[target->image_theme];`
  ⚠ **But `has_keys` is a permanent per-colour inventory flag** (`has_keys[obj->image_theme] = true;`
  on pickup, never cleared). No capacity limit ⇒ "pick up every key you walk past, then walk to the
  gem" is optimal and never reads a colour. Composition-blind ≈ **1.0**. Also: 64×64 RGB only, C++
  build, no route-level judge. ✖
- **XLand-MiniGrid** (`src/xminigrid/core/{actions,grid,constants}.py`). Same good mechanism as
  MiniGrid and in JAX: `Tiles.DOOR_LOCKED`, single-slot pocket (`is_empty_pocket`), and
  `toggle`: `jnp.equal(next_tile[0], Tiles.DOOR_LOCKED) & equal(agent.pocket,
  TILES_REGISTRY[Tiles.KEY, next_tile[1]])` — colour-matched unlock, enforced. ✅ mechanism.
  ✖ but its *benchmark* is meta-RL over production rulesets (`AgentNearRule`, `TileNearRule` …),
  its layouts are `room/two_rooms/four_rooms/nine_rooms` — **not random mazes** — and no shipped
  ruleset makes door colour load-bearing. Needs `flax` + package, absent from the frozen bundle.
- **`maze-dataset`** (understanding-search). Random mazes (DFS / Wilson / percolation) + exact A*
  solver ⇒ gives (a) and (d) cleanly, ✖ but has **no keys, doors, colours or legality at all**.
  It is a maze *generator*, not a composable environment. Useful as a component, not a candidate.
- **MiniGrid `KeyCorridor`** — checked because a search result claimed it forces colour reasoning.
  `_gen_grid`: `door, _ = self.add_door(2, room_idx, 2, locked=True)` then
  `self.add_object(0, self._rand_int(0, self.num_rows), "key", door.color)`. **One key, colour
  defined as the door's.** Same blindness. ✖
- **seqBench** (2025, tunable LLM sequential-reasoning benchmark with key/door edges on a random
  tree): exact ground truth, but it is a *question-answering* benchmark over a described graph, not
  an environment emitting routes, and its keys are one-per-door on a tree. ✖ for (b).
- ⭐ **Generalisation, stated with the code behind it:** *every* off-the-shelf door–key environment I
  read places **exactly one key per locked door** and then either (i) makes that key the only
  reachable one at that stage (MiniGrid/BabyAI/KeyCorridor/UnlockToUnlock) or (ii) lets the agent
  hold every key at once (Procgen heist). Under (i) or (ii) a **colour-blind policy is optimal or
  near-optimal**. The colour match is *enforced* by every engine and *load-bearing* in none.
  ⇒ **NOTHING FITS.** Proceeding to the build under the task's fallback clause.

## 2026-09-20T17:20Z — build: CMAZE-1, and the harness that screens it
Nothing fit, so I built the fallback the task specifies (colour-keyed forbidden tiles, random
mazes, a budget, an exact judge). Code committed in this worktree under `w8_envscout/`:
`maze_colour.py` (env + generator + exact judge + exact solver), `screen.py` (screening harness),
`sensitivity.py` (knob sweep). numpy only — nothing outside the frozen bundle.
- **RULE A** a door cell of colour c is IMPASSABLE unless the pocket holds key c.
- **RULE B** a floor cell tinted colour c is IMPASSABLE while the pocket holds key c.
- The pocket holds **one** key and a key is **consumed** by the door it opens. That is the hinge:
  you cannot put a key down, so committing to colour c locks you out of every c-tinted cell until
  you spend it, and the *order* you commit to colours changes which routes exist at all.
  Both rules make routes ILLEGAL, never merely longer — no metric, no upper bound, nothing that
  GYM-1/GYM-2's speed limit would satisfy.
- State is `(cell, keystate)` with `keystate[c] ∈ {UNTAKEN, CARRIED, USED}`, ≤1 CARRIED. A door is
  open iff its key is USED, so the door mask is not separate state. ≈20 keystates × ~100 floor
  cells ⇒ exact BFS is instant, which is what makes an exact optimum and an exact budget possible.
- **Judge is integer replay.** `judge()` contains no float and no comparison that is not between
  small ints; its failure reasons are `into-wall / locked-door-cN / forbidden-tint-cN /
  over-budget / no-goal`. The dial ("the instrument") is satisfiable: nothing here needs a tolerance.
- **How a "blind" planner is defined.** Not by hand-writing a weak heuristic — by running the SAME
  exact BFS against a RELAXED legality relation and grading with the UNRELAXED judge. Each blind
  planner is therefore *optimal within the world it is given*, which is the strongest honest form of
  the baseline. Family: BLIND-A (rule B dropped), BLIND-B (rule A dropped), BLIND-COLOUR (colours
  collapsed: any key opens any door), BLIND-FACTOR (A×B treated as interacting only near doors),
  BLIND-SEQUENTIAL (fully rule-aware routing, but commits to the key order that rule A alone
  prefers — blind only to the fact that rule B changes which orders survive).
- Two score columns: **validity** (exact judge accepts) and **score** = `valid ? opt_len/len : 0`,
  in [0,1], 1.0 for an optimal valid route — the column to compare against GYM-2's 0.9367.
- Dry run (n_eval=30) already showed the shape; full sweeps (n_eval=500, n_train=2000) launched.

## 2026-09-20T17:44Z — REDIRECTION received mid-turn (logged verbatim in substance)
- Instruction from the Head: **there is no reason to stop.** The law the two failed gyms
  established — (i) at least one rule must render a route **ILLEGAL**, not merely slower, and
  (ii) the two rules' interaction must be worth enough that a **composition-blind policy cannot
  score well** — still guides the design, **but it is now a DESIGN CONSTRAINT TO SATISFY, not a
  gate that rejects.**
- What this changes for me: nothing about the screening work already done, and it confirms the
  branch I am on. I did not use the law to reject and stop — I used it to screen the off-the-shelf
  candidates (all of which fail (ii), and Procgen/MiniGrid fail it in the engine-enforces-but-never-
  requires way), and then **built** CMAZE-1 so that both clauses hold by construction. The
  screening numbers below are therefore REPORTED COLUMNS, exactly as the task file demanded, and
  the Head weighs them. No candidate is being rejected on a number.
- Concrete consequence: I will report the **unfiltered** generator's numbers too, even though they
  are bad (a composition-blind planner scores 0.9240 there — a near-exact reproduction of GYM-2's
  0.9367). That is a design finding about which knob carries the composition, not a reason to abort.

## 2026-09-20T17:52Z — screening numbers, n_eval=500 per sweep, exact judge
Raw output kept at `screen_results.txt` (not rewritten). `score = valid ? opt_len/len(route) : 0`,
so 1.0000 is an optimal valid route and the column is directly comparable to GYM-2's 0.9367.

**A. ADOPTION CONFIGURATION** (7×7 cells → 15×15 grid, 3 colours, p_tint 0.22, braid 0.12,
budget = opt +10%, acceptance filter ON). Seed acceptance 0.952. Optimal 52.9 steps mean;
rule B forces a **+12.9 step detour and bends the route in 100.0%** of accepted episodes.

| planner | validity | score | top failure |
|---|---|---|---|
| ORACLE (full composition)            | 1.0000 | 1.0000 | — |
| BLIND-SEQUENTIAL (key order from A)  | **0.4440** | **0.4435** | `no-route` ×276 |
| BLIND-FACTOR (A×B local only)        | 0.1180 | 0.1180 | `forbidden-tint` ×155 |
| BLIND-COLOUR (colours collapsed)     | 0.0820 | 0.0819 | `no-route` ×251 |
| BLIND-A (rule B dropped)             | 0.0000 | 0.0000 | `forbidden-tint` ×173 |
| BLIND-B (rule A dropped)             | 0.0000 | 0.0000 | `locked-door` ×181 |
| RANDOM                               | 0.0000 | 0.0000 | — |
| FRAGMENT-LOOKUP r=1                  | **0.0000** | **0.0000** | 18 236 sigs / 2 000 routes, coverage 0.9013 |
| FRAGMENT-LOOKUP r=2                  | 0.0000 | 0.0000 | 100 625 sigs, coverage 0.0657 |

⇒ **COMPOSITION-BLIND SCORE = 0.4440** (the max over the blind family, which is the only honest
way to read it — a benchmark is only as good as its best blind attacker).
⇒ **FRAGMENT-LOOKUP SCORE = 0.0000**, and that is *not* a coverage artefact: at r=1 the table
already covers **90.1%** of the states on held-out optimal routes and still never completes a
route, because a 53-step route has to be right at every step in a maze it has never seen.

**B. UNFILTERED GENERATOR** (identical except the "rule B bends the route" acceptance filter is
OFF). Rule B bends the route in only **12.2%** of episodes; detour +1.2 steps.
ORACLE 1.0000 · BLIND-SEQUENTIAL **0.9240** · BLIND-FACTOR 0.8640 · BLIND-A 0.8560 ·
BLIND-COLOUR 0.2360 · FRAG r=1 0.0040.
⚠ ⭐ **This is GYM-2 reproduced.** 0.9240 against GYM-2's 0.9367. Random colour-keyed tints in a
random maze are *almost never* load-bearing on their own. **The acceptance filter is the
composition.** Reported as a design finding, not as a reason to reject anything.

**C. LOOSE BUDGET** (filter ON, budget = opt +50% instead of +10%).
BLIND-SEQUENTIAL 0.4640 (vs 0.4440), BLIND-FACTOR 0.0840, BLIND-COLOUR 0.0680, BLIND-A 0.0000.
⭐ Quintupling the slack moves the blind score by **+0.02**. The blind planners are not failing
because they are slow — they are failing because their routes are **ILLEGAL**. The budget is a
nice-to-have here, exactly as the task's "(c) is desirable" ranking implies.

**Exactness tests** — `python -m w8_envscout.test_maze_colour`, 6/6 PASS:
- `judge()`'s executable body (docstring stripped via `ast`) contains no float literal, no
  division, no `abs`, no `isclose`, no tolerance;
- every exact optimum replays to `goal` at exactly `opt_len` steps (200 seeds);
- `judge` is a pure function of `(instance, route)`;
- RULE A, RULE B and the budget were each *observed* refusing routes;
- every accepted instance satisfies the acceptance contract (200 seeds, 190 accepted);
- ⭐ **`blind failures: 190 by ILLEGALITY, 0 by budget`** — the single line that says CMAZE-1 is not
  GYM-1 or GYM-2.

## 2026-09-20T18:12Z — sensitivity: a stuck row that turned into a design finding
- The first sensitivity run hung on the `braid=0.00` row. Cause, measured directly:
  `braid=0.00 → 0/60 seeds accepted; braid=0.04 → 10/60; braid=0.12 → 23/60`.
- ⭐ **Why, and it matters:** a *perfect* maze has exactly **one** route between any two cells. Rule B
  therefore cannot **bend** a route there — it can only make the instance **infeasible**, which the
  generator's condition 2 then rejects. **Braiding is not cosmetic: it is what gives route choice,
  and without route choice "compose A with B to find a DIFFERENT route" is vacuous.** Any wave-9
  maze source that emits perfect mazes (e.g. `maze-dataset`'s DFS/Wilson defaults) must be braided
  before use.
- Fixed by bounding the search (`make_set_bounded`) so a structurally impossible setting reports an
  acceptance rate of 0 instead of looping. Stopped the stuck job with `TaskStop` on its own task id
  (`bjn32rbc5`) — no box-wide kill, no `taskkill`. Relaunched.

## 2026-09-20T18:30Z — sensitivity complete, code committed and bundled
- Full sensitivity table in `sensitivity_results.txt` (16 settings, n=150 each, 278s). The
  composition-blind score sits in **0.27–0.58 across every setting that produces instances at all** —
  never near GYM-2's 0.9367. Monotone in `n_colours` (0.5200 → 0.4123 → 0.3467 for k=2→3→4), flat in
  `slack` (0.3867 → 0.4184 across 0%→50%), rising with maze size (0.2733 at 5×5 → 0.5798 at 9×9).
  ⚠ The `accept` column in that file is NOT comparable with sweep A's 0.952: the sensitivity harness
  caps `max_tries=60` per seed where `screen.py` uses 400. Noted in ENV.md so nobody reads it as a
  regression.
- Commit `0b7d4e8` on `w8/env-scout`; `git status` clean; bundle at
  `/c/chlu/outputs/w8-env-scout/w8-env-scout.bundle`, `git bundle verify` → *"is okay … records a
  complete history"*. Source also copied to `outputs/w8-env-scout/code/` so the Head needs no worktree.

---

# FINAL REPORT — w8/env-scout

**Deliverable:** `ENV.md` (full report) · `screen_results.txt`, `sensitivity_results.txt` (raw) ·
`code/` + `w8-env-scout.bundle` (commit `0b7d4e8`) · this `LOG.md`.

## The answer
**Nothing off the shelf fits. Adopt `CMAZE-1`, built and measured here.** It gives (a) random braided
mazes per episode, (b) two *legality* rules that compose across agent × colour × door, (c) an integer
budget, and (d) an exact judge that is pure integer replay.

## The two required screening numbers, n = 500, exact judge
| | CMAZE-1 (adopted config) | unfiltered generator | GYM-2 |
|---|---:|---:|---:|
| **composition-blind score** | **0.4440** | 0.9240 | 0.9367–1.0000 |
| **fragment-lookup score** | **0.0000** | 0.0040 | — |
| oracle | 1.0000 | 1.0000 | |
| random | 0.0000 | 0.0000 | |

Composition-blind = the **max over five blind planners**, each of which is the *same exact BFS* run
against a relaxed legality relation and then graded by the unrelaxed judge — so each is optimal
within the world it is given, and none is a straw man.

## The three things the Head should take from this
1. ⭐ **MiniGrid/BabyAI fit on paper and fail in fact, and I can point at the line.** `Door.toggle`
   really does require `env.carrying.color == self.color`, and `step()` really does allow one carried
   object. But `doorkey.py` hard-codes `Door("yellow")` + `Key("yellow")`, and every BabyAI unlock
   level does `add_object(..., "key", door.color)` — **one key per door, defined as the door's
   colour**. Procgen `heist` has random mazes and real colour locks but an *unbounded* key inventory.
   ⇒ **The colour match is enforced by every engine and load-bearing in none.** Composition-blind on
   `MiniGrid-DoorKey-*` is **1.0000 by construction** — proved by the generator, not measured.
2. ⭐ **What makes colour load-bearing is contention, not the door.** Several keys reachable at once,
   a capacity of one, and no way to undo the commitment. CMAZE-1 gets it by consuming the key and
   forbidding same-colour tiles while carrying.
3. ⚠ **The honest weakness, reported not hidden:** CMAZE-1's composition lives in the generator's
   acceptance filter (`opt(A+B) > opt(A alone)`), not in the raw random distribution. With the filter
   off, a composition-blind planner scores **0.9240** — GYM-2 reproduced inside my own environment.
   The filter is a statement about the *instance*, not about any baseline, which is why I consider it
   legitimate; the Head should weigh that. Two un-implemented strengtheners: tint **adversarially**
   along the rule-A-optimal route instead of uniformly, or raise `n_colours` (blind score falls
   monotonically 0.52 → 0.41 → 0.35 for k = 2 → 3 → 4).

## Dial verdict — "the instrument"
**Not falsified.** `judge()` is integer replay; `test_maze_colour.py` strips the docstring with `ast`
and asserts the executable body contains no float literal, no division, no `abs`, no `isclose`, no
tolerance. 6/6 tests pass, including ⭐ **`blind failures: 190 by ILLEGALITY, 0 by budget`** — the one
line that says CMAZE-1 is not GYM-1 and not GYM-2. No downstream refusal or deletion claim inherits
an epsilon from this environment.

## Declared NOT-RUN (never a null)
MiniGrid, BabyAI, Procgen and XLand-MiniGrid were **read, not executed** — `pip install` is forbidden
and none is in the frozen bundle. Their §2 assessments are structural arguments from source, plus one
number (`DoorKey` = 1.0000) that the generator code proves outright. They are **not** measurements.
Also not done, and not needed for the recommendation: an RL-trained (rather than search-based) blind
baseline; an adversarial tint placer; a scored run of any kind (the task forbade one).

## 2026-09-20T19:10Z — Head directive: raise n_colours to 4 and rerun the screen
- Instruction: *"raise n_colours to 4 and rerun the screen. Report the number, do not gate on it."*
- Acting on it directly. Same harness, same seeds, same exact judge, only `n_colours=3 → 4`:
  `python -m w8_envscout.screen n_eval=500 n_train=2000 n_colours=4`. All three sweeps
  (A adoption / B unfiltered / C loose budget) rerun so the k=4 column is like-for-like with k=3.
- Output to a NEW file `screen_results_k4.txt`; `screen_results.txt` (k=3) is NOT overwritten, so
  both columns stay on the record.
- ⚠ Prior expectation on the record before the run, so it can be checked against the result: the
  n=150 sensitivity row gave best-blind 0.3467 at k=4 vs 0.4123 at k=3. I expect roughly that, but
  the sensitivity harness used `max_tries=60` and a different seed block, so this is not a prediction
  to the third decimal. **Reported as a column either way — not a gate.**

## 2026-09-20T19:36Z — n_colours=4 screen complete (raw: `screen_results_k4.txt`)
Like-for-like with the k=3 run: same harness, same n_eval=500 / n_train=2000, same exact judge,
same three sweeps. **k=3 numbers are NOT overwritten** — `screen_results.txt` still holds them.

**A. ADOPTION CONFIGURATION, k=4.** Seed acceptance **0.992** (k=3: 0.952). Optimal **59.7** steps
mean (k=3: 52.9). Rule B forces **+16.0** steps (k=3: +12.9) and bends the route in **100.0%**.

| planner | k=4 validity | k=4 score | k=3 score | top failure (k=4) |
|---|---:|---:|---:|---|
| ORACLE | 1.0000 | 1.0000 | 1.0000 | — |
| **BLIND-SEQUENTIAL** | **0.3360** | **0.3359** | 0.4435 | `no-route` ×328 |
| BLIND-FACTOR | 0.0860 | 0.0860 | 0.1180 | `forbidden-tint` ×118 |
| BLIND-COLOUR | 0.0360 | 0.0360 | 0.0819 | `no-route` ×215 |
| BLIND-A | 0.0000 | 0.0000 | 0.0000 | `forbidden-tint` ×127 |
| BLIND-B | 0.0000 | 0.0000 | 0.0000 | `locked-door` ×135 |
| RANDOM | 0.0000 | 0.0000 | 0.0000 | — |
| **FRAGMENT-LOOKUP r=1** | **0.0000** | **0.0000** | 0.0000 | 21 151 sigs, coverage 0.9004 |
| FRAGMENT-LOOKUP r=2 | 0.0000 | 0.0000 | 0.0000 | 113 184 sigs, coverage 0.0585 |

⇒ ⭐ **COMPOSITION-BLIND SCORE = 0.3360 at k=4** (was 0.4440 at k=3). **FRAGMENT-LOOKUP = 0.0000.**
Every blind planner moved *down*, not just the max: FACTOR 0.1180→0.0860, COLOUR 0.0819→0.0360.
The n=150 sensitivity row had predicted ≈0.3467 at k=4; the n=500 run gives 0.3360 — consistent.

**B. UNFILTERED, k=4.** Rule B bends the route in only 12.4% of episodes (k=3: 12.2%).
BLIND-SEQUENTIAL **0.9100** (k=3: 0.9240) · BLIND-FACTOR 0.8680 · BLIND-A 0.8560 ·
BLIND-COLOUR 0.2200 · FRAG r=1 0.0040.
⚠ ⭐ **Raising k does NOT fix the unfiltered distribution.** 0.9100 is still GYM-2 (0.9367). This is
the important negative result of this rerun: **k is a margin lever, not the mechanism.** The
acceptance filter remains the thing that carries the composition, and the Head should keep weighing
it on its own merits rather than treating k=4 as having retired the question.

**C. LOOSE BUDGET (+50%), k=4.** BLIND-SEQUENTIAL **0.4020** (k=3: 0.4640), FACTOR 0.0800,
COLOUR 0.0180, FRAG 0.0000. Quintupling the slack moves the blind score 0.3360 → 0.4020 (+0.066,
k=3 was +0.020). Still legality, not budget — but the budget matters slightly *more* at k=4, because
a wrong key order at k=4 more often leaves a legal-but-long completion rather than none at all.

**Exactness at k=4 — 6/6 PASS** (`test_maze_colour.py` re-run with `n_colours=4`):
199/200 seeds accepted and all satisfy the acceptance contract; every exact optimum replays to
`goal` at exactly `opt_len`; judge body still float-free; ⭐ **`blind failures: 199 by ILLEGALITY,
0 by budget`.** The dial is unaffected by k.

**Cost, so the Head can weigh it:** sweep A took 295s at k=4 vs 185s at k=3 (~1.6×), from the larger
keystate space (≈65 vs ≈20 reachable keystates). Still trivial in absolute terms.

**Recommendation updated: adopt k=4.** It is strictly better on every axis measured — lower blind
score (0.3360 vs 0.4440), higher seed acceptance (0.992 vs 0.952), longer routes (59.7 vs 52.9),
larger forced detour (+16.0 vs +12.9) — at 1.6× solver cost and with no change to exactness.
⛔ Reported as a column, not a gate: k=3 remains fully viable and its numbers stay on the record.

## 2026-09-20T19:48Z — k=4 made the default; code recommitted and re-bundled
- `generate(..., n_colours=4)` is now the default in `maze_colour.py`, so the adopted configuration
  and the shipped default agree and wave 9 cannot pick up k=3 by accident. Tests re-run against the
  new default: **6/6 PASS**, `blind failures: 199 by ILLEGALITY, 0 by budget`.
- Commit `f968886` on `w8/env-scout`; `git status` clean; bundle rebuilt at
  `/c/chlu/outputs/w8-env-scout/w8-env-scout.bundle` and `git bundle verify` → *"records a complete
  history"*. `code/` refreshed. `screen_results.txt` (k=3) kept untouched alongside
  `screen_results_k4.txt` (k=4) — both columns stay on the record.
- `ENV.md` updated: k=4 is the adopted configuration in §1 and §4, with the k=3 column printed
  beside every row; §5's falsifier table and the "one honest weakness" paragraph now record that
  raising k was tried in full and did **not** retire the filter question.

---

# FINAL REPORT (REVISED after the k=4 rerun) — w8/env-scout

Supersedes the 18:30Z final report above on the numbers only; the screening verdict on the
off-the-shelf candidates is unchanged. Earlier text left in place, not rewritten.

**Deliverable:** `ENV.md` · `screen_results_k4.txt` (adopted) + `screen_results.txt` (k=3) +
`sensitivity_results.txt` · `code/` + `w8-env-scout.bundle` (commit `f968886`) · this `LOG.md`.

## The answer
**Nothing off the shelf fits. Adopt `CMAZE-1` at `n_colours=4`.** (a) random braided mazes per
episode, (b) two *legality* rules composing across agent × colour × door, (c) an integer budget,
(d) an exact judge that is pure integer replay.

## The two required screening numbers, n = 500, exact judge
| | **k=4 (adopted)** | k=3 | unfiltered k=4 | unfiltered k=3 | GYM-2 |
|---|---:|---:|---:|---:|---:|
| **composition-blind** | **0.3360** | 0.4440 | 0.9100 | 0.9240 | 0.9367–1.0000 |
| **fragment-lookup** | **0.0000** | 0.0000 | 0.0040 | 0.0040 | — |
| oracle | 1.0000 | 1.0000 | 1.0000 | 1.0000 | |
| random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | |
| seed acceptance | 0.992 | 0.952 | 1.000 | 1.000 | |

Composition-blind = **max over five blind planners**, each the *same exact BFS* on a relaxed legality
relation graded by the unrelaxed judge — each optimal within the world it is given, none a straw man.

## What the k=4 rerun bought, and what it did not
- **Bought:** composition-blind 0.4440 → **0.3360**, and *every* blind planner fell
  (FACTOR 0.1180 → 0.0860, COLOUR 0.0819 → 0.0360). Seed acceptance 0.952 → 0.992. Mean optimum
  52.9 → 59.7 steps. Forced detour +12.9 → +16.0. `no-route` failures 276 → 328 of 500. Cost 1.6×.
  Exactness unchanged.
- ⚠ **Did not buy:** the unfiltered distribution. 0.9240 → **0.9100** — still GYM-2. ⭐ **`n_colours`
  is a margin lever, not the mechanism.** The generator's acceptance filter (`opt(A+B) > opt(A)`) is
  still the thing carrying the composition, and it is still the number a sceptic should attack first.
  The un-implemented strengthener I would reach for is **adversarial** tint placement along the
  rule-A-optimal route rather than uniform tinting.

## The three things the Head should take from this
1. ⭐ **MiniGrid/BabyAI fit on paper and fail in fact, with the line to point at.** `Door.toggle`
   really requires `env.carrying.color == self.color` and `step()` really allows one carried object —
   but `doorkey.py` hard-codes `Door("yellow")` + `Key("yellow")`, and every BabyAI unlock level does
   `add_object(..., "key", door.color)`. Procgen `heist` has random mazes and real colour locks but
   an *unbounded* key inventory. ⇒ **enforced by every engine, load-bearing in none.**
   Composition-blind on `MiniGrid-DoorKey-*` is **1.0000 by construction**, proved by the generator.
2. ⭐ **What makes colour load-bearing is contention, not the door:** several keys reachable at once,
   capacity one, no way to undo a commitment. More colours sharpen that; they do not create it.
3. ⚠ **The filter is the open question**, and raising k has now been shown not to close it. Reported
   as a column for the Head to weigh — ⛔ not used to gate, and k=3 remains fully viable on the record.

## Dial verdict — "the instrument"
**Not falsified, and not affected by k.** `judge()` is integer replay; `test_maze_colour.py` strips
the docstring with `ast` and asserts the body has no float literal, no division, no `abs`, no
`isclose`, no tolerance. 6/6 PASS at k=4, including ⭐ **`blind failures: 199 by ILLEGALITY,
0 by budget`.** No downstream refusal or deletion claim inherits an epsilon from this environment.

## Declared NOT-RUN (never a null)
MiniGrid, BabyAI, Procgen and XLand-MiniGrid were **read, not executed** — `pip install` is forbidden
and none is in the frozen bundle; their §2 assessments are structural arguments from source plus one
number (`DoorKey` = 1.0000) the generator code proves outright, and are **not** measurements.
Also not done: an RL-trained (rather than search-based) blind baseline; adversarial tint placement;
a k=4 sensitivity sweep (only the n=500 screen was rerun at k=4 — the 16-row sensitivity table in
`sensitivity_results.txt` remains a k=3-centred sweep, with k=4 present only as one row); and any
scored run, which the task forbade.

## 2026-09-20T20:05Z — Head directive: run adversarial tint placement, report the number
- Instruction: *"run the adversarial tint placement and report the number. Do not gate on it. Then
  finish: the 0.9100 stays as a printed column wave 9 inherits."* Acting on it directly.
- **Implemented** `_poison()` in `maze_colour.py` + `generate(..., adversarial=R)`. It runs the
  targeted blind planner, replays its route, and for every cell that route enters **while carrying
  key c** sets that cell's tint to **c** — precisely the condition RULE B refuses. Each addition is
  kept only if the instance stays feasible under A+B, so the oracle is never destroyed. Round 0
  targets the rule-A-only planner; round 1 targets the factored planner.
  ⭐ The point: it moves the interaction out of a **rejection filter** and into the **distribution** —
  nothing is discarded, every instance is constructed to interact.
- ⚠ **First attempt largely failed, and the reason is the finding.** At the adopted `braid=0.12`,
  ~8 of 9 candidate cells per route reverted: the key→door corridor is often the **only** path, so
  tinting any cell on it does not bend the route, it destroys feasibility. Poisoning needs **route
  multiplicity** to bite. Measured directly:

  | braid | cells poisoned/round | mean optimum | BLIND-A valid | BLIND-FACTOR valid |
  |---:|---:|---:|---:|---:|
  | 0.12 | 1.0 | 39.7 | 0.600 | 0.750 |
  | 0.25 | 2.0 | 40.2 | 0.325 | 0.400 |
  | **0.40** | **4.0** | 43.6 | **0.150** | **0.275** |
  | 0.55 | 3.6 | 35.0 | 0.158 | 0.237 |

  (n=40 each, no acceptance filter.) Chose **braid=0.40** for the full run — past it the maze opens
  into rooms and the optimum falls without buying anything.
- ⭐ **Running a braid-matched CONTROL as well** (`braid=0.40`, uniform tinting, no filter), because
  otherwise I could not tell whether a drop came from adversarial *placement* or merely from more
  *braiding*. Two full screens launched, n_eval=500 / n_train=2000 each:
  `screen_results_adversarial.txt` and `screen_results_braid40_control.txt`.

## 2026-09-20T20:44Z — adversarial tint placement: the number, with a braid-matched control
Raw: `screen_results_adversarial.txt` (adversarial, braid 0.40) and
`screen_results_braid40_control.txt` (uniform tinting, braid 0.40). n_eval=500, n_train=2000, k=4,
exact judge, both runs identical apart from the tint **placement**.

### ⭐ THE NUMBER — unfiltered generator (acceptance filter OFF), composition-blind score
| generator | braid | tint placement | **composition-blind** | fragment-lookup |
|---|---:|---|---:|---:|
| as inherited by wave 9 | 0.12 | uniform | **0.9100** | 0.0040 |
| braid-matched CONTROL | 0.40 | uniform | **0.8940** | 0.0020 |
| **ADVERSARIAL** | 0.40 | **poisoned** | **0.4980** | **0.0000** |

⭐ **Attribution is clean: braid alone buys −0.016, adversarial placement buys −0.396.** The drop is
the placement, not the extra route choice that enables it. Full unfiltered adversarial row:
BLIND-SEQUENTIAL 0.4980 · BLIND-FACTOR 0.2340 · BLIND-A 0.1420 · BLIND-COLOUR 0.0840 ·
RANDOM 0.0000 · FRAG r=1 0.0000 · ORACLE 1.0000. Rule B bends the route in **78.4%** of episodes
with no filter at all (uniform control: 26.2%; uniform braid-0.12: 12.4%).

### Filtered sweeps, for completeness
| config | filter | **composition-blind** |
|---|---|---:|
| k=4, braid 0.12, uniform *(current adopted)* | ON | **0.3360** |
| k=4, braid 0.40, adversarial | ON | 0.4020 |
| k=4, braid 0.40, uniform | ON | 0.5800 |
| k=4, braid 0.40, adversarial, +50% slack | ON | 0.4440 |

⇒ The **filtered** adopted config is still the lowest blind score (0.3360). Adversarial placement's
value is not a lower number — it is that it reaches **0.4980 with no filter at all**, which is the
thing the filter dependency was costing us.

### ⚠ A correction to how I reported "seed acceptance" earlier
`generate()` retries **internally** up to `max_tries` within a single seed, so "seed acceptance"
counts seeds that *eventually* produced an instance, **not** the filter's rejection rate. The
adversarial sweep A shows acceptance 1.000 *and* a filter that is still doing work. So the 0.952 /
0.992 figures I reported earlier **understate** how much the filter discards. Stated here rather
than quietly corrected; it does not change any score, only the reading of that one column.

### ⭐ What this establishes for wave 9
The interaction can be put in the **distribution** instead of a **rejection filter**, and it roughly
halves the composition-blind score when you do (0.8940 → 0.4980 like-for-like). It needs route
multiplicity to work: at braid 0.12 the key→door corridor is usually unique, so ~8 of 9 poison
candidates revert to preserve feasibility. **Adversarial placement and braiding are a pair.**
⛔ Reported as columns. Not a gate. **The 0.9100 stays as a printed column wave 9 inherits.**

---

# FINAL REPORT (FINAL) — w8/env-scout

Supersedes the 18:30Z and 19:48Z reports on the numbers only; the screening verdict on the
off-the-shelf candidates is unchanged throughout. Earlier text left in place, never rewritten.

**Deliverable:** `ENV.md` · raw: `screen_results_k4.txt` (adopted) · `screen_results.txt` (k=3) ·
`screen_results_adversarial.txt` + `screen_results_braid40_control.txt` (adversarial + its control) ·
`sensitivity_results.txt` · `code/` + `w8-env-scout.bundle` (commit `55e507d`) · this `LOG.md`.

## The answer
**Nothing off the shelf fits. Adopt `CMAZE-1`.** (a) random braided mazes per episode, (b) two
*legality* rules composing across agent × colour × door, (c) an integer budget, (d) an exact judge
that is pure integer replay. Default config `n_colours=4`, `braid=0.12`, `p_tint=0.22`, `slack=0.10`.

## The required screening numbers — every column, n = 500, k = 4, exact judge
| configuration | filter | **composition-blind** | **fragment-lookup** | oracle | random |
|---|---|---:|---:|---:|---:|
| braid 0.12, uniform *(adopted)* | ON | **0.3360** | **0.0000** | 1.0000 | 0.0000 |
| braid 0.12, uniform | OFF | **0.9100** ⬅ *inherited by wave 9* | 0.0040 | 1.0000 | 0.0000 |
| braid 0.40, uniform *(control)* | OFF | **0.8940** | 0.0020 | 1.0000 | 0.0000 |
| **braid 0.40, adversarial** | **OFF** | **0.4980** | **0.0000** | 1.0000 | 0.0000 |
| braid 0.40, adversarial | ON | 0.4020 | 0.0000 | 1.0000 | 0.0000 |
| braid 0.12, uniform, k=3 | ON | 0.4440 | 0.0000 | 1.0000 | 0.0000 |
| braid 0.12, uniform, k=3 | OFF | 0.9240 | 0.0040 | 1.0000 | 0.0000 |

Composition-blind = **max over five blind planners**, each the *same exact BFS* run against a relaxed
legality relation and graded by the unrelaxed judge — each optimal within the world it is given.
GYM-2 died at 0.9367–1.0000.

## The four things wave 9 should carry forward
1. ⭐ **MiniGrid/BabyAI fit on paper and fail in fact, with the line to point at.** `Door.toggle`
   really requires `env.carrying.color == self.color`, `step()` really allows one carried object —
   but `doorkey.py` hard-codes `Door("yellow")`+`Key("yellow")` and every BabyAI unlock level does
   `add_object(..., "key", door.color)`. Procgen `heist` has random mazes and real colour locks but
   an **unbounded** key inventory. ⇒ **enforced by every engine, load-bearing in none.**
   `MiniGrid-DoorKey-*` composition-blind = **1.0000 by construction**, proved by the generator.
2. ⭐ **What makes colour load-bearing is contention, not the door:** several keys reachable at once,
   capacity one, no way to undo a commitment.
3. ⭐ **Where the interaction lives is a design choice, and both options are now measured.** A
   rejection **filter** gives 0.3360 but discards instances; **adversarial placement** gives 0.4980
   with **no filter at all**, discarding nothing — roughly halving the unfiltered blind score against
   a braid-matched control (0.8940). Attribution is clean: braid alone buys −0.016, placement −0.396.
   It needs `braid≈0.40`: at braid 0.12 the key→door corridor is usually unique, so poisoning it
   destroys feasibility instead of bending the route. **Placement and braiding are a pair.**
4. ⛔ **The 0.9100 stays printed.** It is the number the uniform, unfiltered generator hands wave 9,
   and it is a live warning, not a retired one: `n_colours` did not move it (0.9240 → 0.9100) and
   braiding did not move it (→ 0.8940). Only the placement did.

## Dial verdict — "the instrument"
**Not falsified, under every configuration run.** `judge()` is integer replay; `test_maze_colour.py`
strips the docstring with `ast` and asserts the body has no float literal, no division, no `abs`, no
`isclose`, no tolerance. 6/6 PASS at k=4 uniform (`blind failures: 199 by ILLEGALITY, 0 by budget`)
and 6/6 on the adversarial generator (`167 by ILLEGALITY, 0 by budget`, 200/200 instances feasible
with load-bearing doors). No downstream refusal or deletion claim inherits an epsilon from this
environment.

## Declared NOT-RUN (never a null)
- MiniGrid, BabyAI, Procgen, XLand-MiniGrid: **read, not executed** — `pip install` forbidden, none
  in the frozen bundle. §2 is structural argument from source plus one number (`DoorKey` = 1.0000)
  the generator code proves outright. **Not measurements.**
- An RL-trained (rather than search-based) blind baseline.
- A sensitivity sweep at k=4 or under adversarial placement — `sensitivity_results.txt` is a
  k=3-centred, uniform-tinting sweep; k=4 and braid appear there only as single rows.
- Any scored run, which the task forbade.

## Process record
`START_UTC.txt` 2026-09-20T16:36:40Z, no live one pre-existing. No `taskkill` and no box-wide kill at
any point; the one stuck job was stopped with `TaskStop` on its own task id. No detached work was
left running at any turn boundary — every background job was polled to completion inside its turn.
`git status` clean at `55e507d`; `git bundle verify` → "records a complete history".

## 2026-09-20T21:05Z — Head directive: make braid-0.40 adversarial the DEFAULT, rerun sensitivity at k=4
- Instruction: *"make braid 0.40 adversarial the default and rerun the sensitivity sweep at k=4.
  That closes your last declared gap and it is what wave 9 will inherit."* Acting on it directly.
- **`generate()` defaults changed** to the adopted configuration:
  `n_colours=4, braid=0.40, adversarial=2, require_bend=False, p_tint=0.22, slack=0.10`.
  ⭐ `require_bend` is now **OFF by default**: the interaction lives in the **distribution**
  (adversarial placement), not in a rejection filter, so no instance is discarded for failing to
  interact. The docstring now carries the measured numbers so a reader of the code sees them.
- **Test updated to match, honestly rather than quietly.** `test_generator_contract` used to assert
  `opt_len > opt_len_a_only` per instance; with the filter off that clause is exactly what the new
  default does *not* guarantee. It now asserts the two unconditional conditions per instance (doors
  load-bearing, feasible, budget ≥ opt) and measures the bend as a **rate**, asserting only that it
  stays above 0.5 — so a collapse would be caught with a number rather than silently allowed.
  Result at the new defaults: **200/200 seeds accepted, rule B bends the route in 0.725.**
  6/6 tests pass; `blind failures: 167 by ILLEGALITY, 0 by budget`.
- **`screen.py` sweep labels/flags corrected** so a future run matches the new default: sweep A is now
  *"ADOPTION CONFIGURATION (adversarial placement, NO acceptance filter)"* with `require_bend=False`,
  sweep B is the filter-ON comparison, sweep C the loose budget with the filter off. ⚠ The already-
  published raw files were produced under the old labelling; `screen_results_adversarial.txt` sweep B
  (filter OFF) is the run that corresponds to today's sweep A. Recorded so nobody misreads them.
- **Sensitivity rewritten and rerun at the new defaults** — this closes the gap I declared in the
  previous final report ("no sensitivity sweep at k=4 or under adversarial placement"). 20 settings,
  n=150 each, now including `adversarial` rounds 0–3, `max_poison` 2–16, `n_colours` up to 5, and an
  explicit `require_bend=True` row. A `bends` column was added so the table shows how often rule B
  actually changes the route without any filter.

## 2026-09-20T21:22Z — sensitivity at the adopted defaults: the gap is closed
Raw: `sensitivity_results_k4_adversarial.txt` (20 settings × n=150, 572s). `bends` = fraction of
instances where rule B changes the optimal route, measured with **no filter**, so it is a property
of the distribution. One knob varies per row; everything else sits at the shipped defaults.

| lever | setting → blind score | reading |
|---|---|---|
| ⭐ **adversarial rounds** | 0 → **0.8590** (bends 0.260) · 1 → 0.5261 · **2 → 0.5333** · 3 → 0.4929 | **the placer is the mechanism**; one round captures nearly all of it |
| ⭐ **braid** | 0.12 → 0.7600 · 0.25 → 0.6467 · **0.40 → 0.4597** · 0.55 → 0.4325 | braid now matters *because* the placer needs routes to divert onto; 0.40 is the knee (0.55 costs acceptance 0.758 → 0.418) |
| `n_colours` | 2 → 0.7600 · 3 → 0.6600 · **4 → 0.5133** · 5 → **0.4333** | monotone, and acceptance *rises* with k (0.725 → 0.882 for 4 → 5) |
| `max_poison` | 2 → 0.5395 · 4 → 0.4927 · **8 → 0.5193** · 16 → 0.5265 | flat — the placer must be *targeted*, not greedy |
| `slack` | 0.00 → 0.4667 · **0.10 → 0.4400** · 0.25 → 0.4658 · 0.50 → 0.5067 | budget still not the lever |
| `p_tint` | 0.12 → 0.5200 · **0.22 → 0.4796** · 0.32 → 0.5591 | flat within noise |
| maze size | 5×5 → 0.5329 · **7×7 → 0.5867** · 9×9 → 0.5593 | flat within noise |
| `require_bend=True` | → **0.3865** (bends 1.000, acceptance 0.628) | the filter still gives a lower number, at the cost of discarding ~37% of seeds — the trade the default now declines |

- ⭐ **The single most useful row is `adversarial=0`: 0.8590.** Holding everything else at the
  defaults and turning only the placer off sends the blind score back to GYM-2 territory. That is the
  cleanest isolation of the mechanism in the whole spoke.
- ⚠ **Row-to-row scatter is ≈ ±0.05 at n=150.** Three rows *are* the default configuration under
  different labels and give 0.5333 / 0.4597 / 0.4796; the n=500 screen gives **0.4980**. Treat
  0.4980 as the number and the table as the shape. Stated so nobody quotes a single row as the score.
- ⛔ **`n_colours=5` looks strictly better** (0.4333 at acceptance 0.882) and is **reported, not
  acted on** — k=4 is what was directed; k=5 is a free margin lever available to wave 9.
- Commit `587d580`; `git status` clean; bundle rebuilt and verified ("records a complete history");
  `code/` refreshed; `sensitivity_results.txt` (k=3, uniform, filter ON) kept alongside the new file.

---

# FINAL REPORT (FINAL — supersedes all earlier reports on numbers) — w8/env-scout

Earlier reports left in place, never rewritten. The screening verdict on the off-the-shelf
candidates is unchanged throughout; only the adopted configuration and its numbers moved.

**Deliverable:** `ENV.md` · raw: `screen_results_adversarial.txt` (adopted) ·
`screen_results_braid40_control.txt` (its control) · `screen_results_k4.txt` · `screen_results.txt` ·
`sensitivity_results_k4_adversarial.txt` (adopted) · `sensitivity_results.txt` ·
`code/` + `w8-env-scout.bundle` (commit `587d580`) · this `LOG.md`.

## The answer
**Nothing off the shelf fits. Adopt `CMAZE-1` at the shipped defaults** —
`n_colours=4, braid=0.40, adversarial=2, require_bend=False, p_tint=0.22, slack=0.10`.
(a) random braided mazes per episode · (b) two *legality* rules composing across agent × colour ×
door · (c) an integer budget · (d) an exact judge that is pure integer replay.
⭐ The interaction lives in the **distribution**, not in a rejection filter: nothing is discarded.

## The required screening numbers — n = 500, k = 4, exact judge
| configuration | filter | **composition-blind** | **fragment-lookup** | oracle | random |
|---|---|---:|---:|---:|---:|
| **adversarial, braid 0.40 — SHIPPED DEFAULT** | **OFF** | **0.4980** | **0.0000** | 1.0000 | 0.0000 |
| uniform, braid 0.40 (control) | OFF | 0.8940 | 0.0020 | 1.0000 | 0.0000 |
| uniform, braid 0.12 | OFF | **0.9100** ⬅ *inherited if the placer is reverted* | 0.0040 | 1.0000 | 0.0000 |
| uniform, braid 0.12 | ON | 0.3360 | 0.0000 | 1.0000 | 0.0000 |
| adversarial, braid 0.40 | ON | 0.4020 | 0.0000 | 1.0000 | 0.0000 |
| uniform, braid 0.12, k=3 | ON / OFF | 0.4440 / 0.9240 | 0.0000 / 0.0040 | 1.0000 | 0.0000 |

Composition-blind = **max over five blind planners**, each the *same exact BFS* run against a relaxed
legality relation and graded by the unrelaxed judge. GYM-2 died at 0.9367–1.0000.

## The four things wave 9 inherits
1. ⭐ **MiniGrid/BabyAI fit on paper and fail in fact, with the line to point at.** `Door.toggle`
   requires `env.carrying.color == self.color` and `step()` allows one carried object — but
   `doorkey.py` hard-codes `Door("yellow")`+`Key("yellow")`, every BabyAI unlock level does
   `add_object(..., "key", door.color)`, and Procgen `heist` keeps an **unbounded** key inventory.
   ⇒ **enforced by every engine, load-bearing in none.** `MiniGrid-DoorKey-*` composition-blind =
   **1.0000 by construction**, proved by the generator.
2. ⭐ **Colour becomes load-bearing through contention, not through the door:** several keys reachable
   at once, capacity one, no way to undo a commitment.
3. ⭐ **Adversarial tint placement is the mechanism, and the sensitivity sweep isolates it.** Turning
   only the placer off, at the defaults: **0.5333 → 0.8590**, bend rate 0.793 → 0.260. It needs
   braid ≈ 0.40 to have routes to divert onto; `max_poison` is flat, so it must be targeted, not
   greedy. Budget, tint density and maze size are all flat.
4. ⛔ **The 0.9100 stays printed.** It is what the uniform, unfiltered generator yields, and it is a
   live warning: `n_colours` did not move it (0.9240 → 0.9100) and braiding did not move it
   (→ 0.8940). Only the placement did.

## Dial verdict — "the instrument"
**Not falsified, under every configuration run.** `judge()` is integer replay; the test strips the
docstring with `ast` and asserts the body has no float literal, no division, no `abs`, no `isclose`,
no tolerance. 6/6 PASS at the shipped defaults: **200/200 seeds accepted, bend rate 0.725,
`blind failures: 167 by ILLEGALITY, 0 by budget`.** No downstream refusal or deletion claim inherits
an epsilon from this environment.

## Declared NOT-RUN (never a null)
- MiniGrid, BabyAI, Procgen, XLand-MiniGrid: **read, not executed** — `pip install` forbidden, none in
  the frozen bundle. §2 of ENV.md is structural argument from source plus one number
  (`DoorKey` = 1.0000) the generator code proves outright. **Not measurements.**
- An RL-trained (rather than search-based) blind baseline.
- A fresh n=500 screen under the *new sweep labelling*: the adopted 0.4980 comes from
  `screen_results_adversarial.txt` sweep B, which is exactly this configuration under the previous
  labelling. The n=150 sensitivity rows agree with it inside their ±0.05 scatter.
- Any scored run, which the task forbade.

## Process record
`START_UTC.txt` 2026-09-20T16:36:40Z; no live one pre-existing. No `taskkill`, no box-wide kill at any
point — the one stuck job was stopped with `TaskStop` on its own task id. No detached work left
running across a turn boundary; every background job was polled to completion inside its turn.
`git status` clean at `587d580`; `git bundle verify` → "records a complete history"; `code/` byte-
identical to the worktree.

## 2026-09-20T21:40Z — Head directive: k=5 declined; close the provenance gap instead
- Instruction: *"No to k=5 — colours are measured not to be the lever (0.9240 to 0.9100), placement
  is (0.4980). Instead close your own provenance gap: rerun the clean n=500 screen under the CURRENT
  labelling so the headline 0.4980 is not inherited from old labels. Then finish."*
- ⭐ **The k=5 reasoning is correct and I am recording why, so it does not get re-proposed.** Colours
  move the unfiltered composition-blind score by 0.0140 (0.9240 → 0.9100). Placement moves it by
  0.3960 (0.8940 → 0.4980). The k=5 sensitivity row (0.4333 vs 0.5133) is a *within-adversarial*
  margin at n=150, which is inside that table's ±0.05 scatter — it is not evidence that colours are
  a lever, and quoting it as one would have muddled the finding the spoke exists to deliver.
  ⛔ k stays at 4. No further knob changes proposed.
- **Provenance gap, stated plainly:** the headline **0.4980** came from
  `screen_results_adversarial.txt` **sweep B**, produced under the OLD sweep labelling
  (A = filter ON, B = filter OFF) and with braid/adversarial passed as explicit CLI arguments rather
  than read from the defaults. It is the right configuration, but a reader has to know two facts
  about a superseded labelling to see that. That is a provenance defect even though the number is
  sound. Rerunning clean: `python -m w8_envscout.screen n_eval=500 n_train=2000` with **no arguments
  beyond the sizes**, so every generator knob comes from the shipped defaults and sweep A is the
  adopted configuration under the CURRENT labelling → `screen_results_DEFAULTS.txt`.

## 2026-09-20T21:57Z — clean n=500 screen at the shipped defaults: provenance gap CLOSED
`screen_results_DEFAULTS.txt` — `python -m w8_envscout.screen n_eval=500 n_train=2000`, **no
arguments beyond the sizes**, so every generator knob is read from the shipped defaults and sweep A
is the adopted configuration under the CURRENT labelling. Nothing is inherited from old labels.

| sweep | filter | SEQUENTIAL | FACTOR | COLOUR | BLIND-A | BLIND-B | RANDOM | FRAG | bends | opt |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A — adoption configuration** | **OFF** | **0.4980** | 0.2600 | 0.0840 | 0.1280 | 0.0000 | 0.0000 | **0.0000** | 81.6% | 42.4 |
| B — filter also ON (comparison) | ON | 0.3720 | 0.1160 | 0.0200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0% | 47.3 |
| C — loose budget (+50%) | OFF | 0.5420 | 0.2440 | 0.1060 | 0.1480 | 0.0000 | 0.0000 | 0.0000 | 79.4% | 41.6 |

- ⭐ **COMPOSITION-BLIND = 0.4980. FRAGMENT-LOOKUP = 0.0000. ORACLE = 1.0000.** Seed acceptance 1.000
  — nothing discarded. Mean optimum 42.4 steps, mean detour forced by rule B +14.8.
- ⭐ **It reproduces.** The earlier measurement (`screen_results_adversarial.txt` sweep B, old
  labelling, different seed block) also gave **0.4980** — identical to four decimals with a
  *different* failure-mode breakdown (`no-route` ×247 here vs ×250 there; FACTOR 0.2600 vs 0.2340).
  Two independent samples, same number. The headline is not an artefact of one seed block.
- The filter-ON comparison now sits at **0.3720** at the defaults. It is lower, but it discards
  instances to buy 0.126 — the trade the default declines, and now measured on like-for-like
  generator settings rather than across two configurations.
- Loose budget 0.4980 → 0.5420: the blind planners are still failing on **legality**, not cost.
- ⛔ **k stays at 4.** Recording the Head's reasoning so it is not re-proposed: colours move the
  unfiltered blind score by 0.0140 (0.9240 → 0.9100); placement moves it by 0.3960 (0.8940 → 0.4980).
  The k=5 sensitivity row (0.4333 vs 0.5133) is a within-adversarial margin at n=150, inside that
  table's ±0.05 scatter, and is **not** evidence that colours are a lever. Quoting it as one would
  have muddled the finding this spoke exists to deliver.
- No code changed this round, so the commit and bundle stand at `587d580`; `ENV.md` repointed so the
  headline cites `screen_results_DEFAULTS.txt` directly, and the NOT-RUN item for this screen is
  struck through and closed.

---

# FINAL REPORT (CLOSING) — w8/env-scout

Supersedes all earlier reports on numbers; the screening verdict on the off-the-shelf candidates is
unchanged throughout. Earlier text left in place, never rewritten.

**Deliverable:** `ENV.md` · **`screen_results_DEFAULTS.txt`** (headline, from the shipped defaults) ·
`screen_results_adversarial.txt` + `screen_results_braid40_control.txt` (the placer and its control) ·
`screen_results_k4.txt` · `screen_results.txt` · `sensitivity_results_k4_adversarial.txt` ·
`sensitivity_results.txt` · `code/` + `w8-env-scout.bundle` (commit `587d580`) · this `LOG.md`.

## The answer
**Nothing off the shelf fits. Adopt `CMAZE-1` at the shipped defaults** —
`n_colours=4, braid=0.40, adversarial=2, require_bend=False, p_tint=0.22, slack=0.10`.
(a) random braided mazes per episode · (b) two *legality* rules composing across agent × colour ×
door · (c) an integer budget · (d) an exact judge that is pure integer replay.
The interaction lives in the **distribution**, not in a rejection filter: **nothing is discarded**.

## The two required numbers — n = 500, exact judge, straight from the defaults
| | **CMAZE-1 (shipped defaults)** | uniform tinting, no filter | GYM-2 |
|---|---:|---:|---:|
| **composition-blind** | **0.4980** | 0.9100 | 0.9367–1.0000 |
| **fragment-lookup** | **0.0000** | 0.0040 | — |
| oracle / random | 1.0000 / 0.0000 | 1.0000 / 0.0000 | |
| rule B bends the route | 81.6% | 12.4% | |

Composition-blind = **max over five blind planners**, each the *same exact BFS* run against a relaxed
legality relation and graded by the unrelaxed judge — each optimal within the world it is given.

## The four things wave 9 inherits
1. ⭐ **MiniGrid/BabyAI fit on paper and fail in fact, with the line to point at.** `Door.toggle`
   requires `env.carrying.color == self.color` and `step()` allows one carried object — but
   `doorkey.py` hard-codes `Door("yellow")`+`Key("yellow")`, every BabyAI unlock level does
   `add_object(..., "key", door.color)`, and Procgen `heist` keeps an **unbounded** key inventory.
   ⇒ **enforced by every engine, load-bearing in none.** `MiniGrid-DoorKey-*` composition-blind =
   **1.0000 by construction**, proved by the generator, not measured.
2. ⭐ **Colour becomes load-bearing through contention, not through the door:** several keys reachable
   at once, capacity one, no way to undo a commitment.
3. ⭐ **Placement is the lever; colours are not.** Turning only the placer off, defaults otherwise:
   0.5333 → **0.8590** (bend rate 0.793 → 0.260). Raising colours k=3 → 4 unfiltered: 0.9240 →
   0.9100. The placer needs `braid≈0.40` to have routes to divert onto, and `max_poison` is flat, so
   it must be **targeted, not greedy**. Budget, tint density and maze size are flat.
4. ⛔ **The 0.9100 stays printed.** It is what the uniform, unfiltered generator yields and it is a
   live warning, not a retired one.

## Dial verdict — "the instrument"
**Not falsified, under every configuration run.** `judge()` is integer replay; the test strips the
docstring with `ast` and asserts the body has no float literal, no division, no `abs`, no `isclose`,
no tolerance. 6/6 PASS at the shipped defaults: **200/200 seeds accepted, bend rate 0.725,
`blind failures: 167 by ILLEGALITY, 0 by budget`.** No downstream refusal or deletion claim inherits
an epsilon from this environment.

## Declared NOT-RUN (never a null)
- MiniGrid, BabyAI, Procgen, XLand-MiniGrid: **read, not executed** — `pip install` forbidden, none in
  the frozen bundle. ENV.md §2 is structural argument from source plus one number
  (`DoorKey` = 1.0000) the generator code proves outright. **Not measurements.**
- An RL-trained (rather than search-based) blind baseline.
- Any scored run, which the task forbade.
- *(Closed this round: the k=4 / adversarial sensitivity sweep, and the clean n=500 screen at the
  shipped defaults. No open gaps remain that I am aware of.)*

## Process record
`START_UTC.txt` 2026-09-20T16:36:40Z; no live one pre-existing. No `taskkill`, no box-wide kill at any
point — the one stuck job was stopped with `TaskStop` on its own task id. No detached work left
running across a turn boundary; every background job was polled to completion inside its turn.
`git status` clean at `587d580`; `git bundle verify` → "records a complete history"; `code/`
byte-identical to the worktree. LOG.md written in dated bullets throughout and never rewritten.
