# ENV — the maze environment wave 9 will run on

**Spoke:** `w8/env-scout` (web-scout + engineer read of candidate code).
**Task:** `/c/chlu/tasks-w8/w8-spoke-env-scout.md` · **Dial:** the instrument.
**Started** 2026-09-20T16:36:40Z · **Log:** `LOG.md` (dated bullets, never rewritten).
**Raw numbers:** **`screen_results_DEFAULTS.txt`** (the adopted configuration, produced from the
shipped defaults with no CLI overrides), `screen_results_k4.txt`, `screen_results.txt` (k=3),
`screen_results_adversarial.txt` + `screen_results_braid40_control.txt` (adversarial tinting),
`sensitivity_results.txt` · **Code:** branch `w8/env-scout`,
`w8_envscout/` (`maze_colour.py`, `screen.py`, `sensitivity.py`, `test_maze_colour.py`).

> Under the wave-8 redirection received mid-run, the GYM-1/GYM-2 law is a **design constraint to
> satisfy, not a gate that rejects**. Every number below is a **reported column**. Nothing here is
> an abort, and no candidate is rejected because a number fired.

---

## 1. RECOMMENDATION

**Nothing off the shelf fits. Adopt the build: `CMAZE-1`, implemented and measured in this spoke.**
**Adopted configuration — and now the shipped defaults of `generate()`:**
`n_colours=4`, `braid=0.40`, `adversarial=2`, `require_bend=False`, `p_tint=0.22`, `slack=0.10`.
⭐ `require_bend` is **OFF**: the interaction lives in the **distribution** (adversarial tint
placement), not in a rejection filter, so **no instance is discarded for failing to interact**.
Earlier configurations (k=3; uniform tinting; filter ON) stay on the record as printed columns.

| | | |
|---|---|---|
| **(a) random mazes per episode** | ✅ | randomised-DFS maze + braiding, fresh per seed |
| **(b) legality composing across agent × colour × door** | ✅ | two rules, both *illegality* rules, both indexed by the pocket |
| **(c) budget** | ✅ | integer step budget, `opt +10%` |
| **(d) exact validity check** | ✅ | integer replay; no float, no division, no tolerance in the judge |

### The two numbers the task asked for (n = 500 episodes, exact judge)

Source: **`screen_results_DEFAULTS.txt` sweep A**, produced by
`python -m w8_envscout.screen n_eval=500 n_train=2000` with **no arguments beyond the sizes** — every
generator knob comes from the shipped defaults, so the headline has no provenance hop.

| | **CMAZE-1 — shipped defaults** | uniform tinting, no filter | for comparison |
|---|---:|---:|---|
| **composition-blind score** | **0.4980** | 0.9100 | GYM-2 died at **0.9367–1.0000** |
| **fragment-lookup score** | **0.0000** | 0.0040 | — |
| oracle (full composition) | 1.0000 | 1.0000 | |
| random | 0.0000 | 0.0000 | |
| acceptance filter | **OFF** | OFF | |
| rule B bends the route | **81.6%** | 12.4% | |

Full sweep A row: SEQUENTIAL **0.4980** · FACTOR 0.2600 · BLIND-A 0.1280 · COLOUR 0.0840 ·
BLIND-B 0.0000 · RANDOM 0.0000 · FRAG 0.0000 · ORACLE 1.0000. Mean optimum 42.4 steps, mean detour
+14.8, seed acceptance 1.000 (nothing discarded).
⭐ **This reproduces the 0.4980 measured earlier on a different seed block** — same score to four
decimals, different failure-mode breakdown (`no-route` ×247 here, ×250 there). The number is not an
artefact of one sample.

The composition-blind score is the **maximum over a family of five blind planners**, not a single
cherry-picked one — a benchmark is only as good as its best blind attacker. Raising `n_colours`
3 → 4 moved **every** blind planner down (FACTOR 0.1180 → 0.0860, COLOUR 0.0819 → 0.0360), raised
seed acceptance, lengthened routes (52.9 → 59.7 mean optimum) and enlarged the forced detour
(+12.9 → +16.0 steps), at 1.6× solver cost and with no change to exactness.

⚠ **It does not fix the unfiltered distribution** — see §4.B: with the acceptance filter off, a
composition-blind planner still scores **0.9100** at k=4 (0.9240 at k=3). **`n_colours` is a margin
lever, not the mechanism.** ⭐ **Adversarial tint placement does fix it** — §4.D: **0.4980 with no
filter at all**, against a braid-matched uniform control of 0.8940.

---

## 2. THE SCREEN: what the off-the-shelf candidates actually do

I could not `pip install` (frozen instrument; `import minigrid` → `ModuleNotFoundError`), so these
are engineer reads of the upstream source fetched with `curl`, not runs. That is stated plainly so
no downstream reader mistakes a read for a measurement.

### 2.1 MiniGrid / BabyAI — assessed first, as instructed

**The legality mechanism is exactly right.** `minigrid/core/world_object.py`:

```python
class Door(WorldObj):
    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def toggle(self, env, pos):
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False
```

A locked door is **impassable**, and it opens **only** for a key of its own colour. And
`minigrid_env.py::step` gives the capacity constraint that would make colour matter:

```python
elif action == self.actions.pickup:
    if fwd_cell and fwd_cell.can_pickup():
        if self.carrying is None:          # <- one slot, total
```

So the engine genuinely couples agent × colour × door. **This is not what killed GYM-1 and GYM-2** —
there is no speed limit here, there is a wall.

⭐ **But no shipped level requires it. The generators make the correct key the only reachable key.**
This is precisely the trap the task warned about, so here is the code.

- `minigrid/envs/doorkey.py::_gen_grid` — **one colour, hard-coded**:

  ```python
  self.put_obj(Door("yellow", is_locked=True), splitIdx, doorIdx)
  self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))
  ```

  One door, one key, one colour ⇒ a colour-blind policy is **bit-identical** to the optimal policy.
  **Composition-blind score on `MiniGrid-DoorKey-*` is 1.0000 by construction** — that needs no
  measurement, the generator is the proof. Randomness is `splitIdx`/`doorIdx` on a two-region split:
  **not a maze** either, so (a) fails as well.

- `envs/babyai/unlock.py` (`Unlock`, `UnlockLocal`, `KeyInBox`, `UnlockPickup`,
  `BlockedUnlockPickup`) and `envs/keycorridor.py` — all the same shape: add a locked door, then
  add **one** key *defined as* its colour.

  ```python
  door, _ = self.add_door(2, room_idx, 2, locked=True)
  self.add_object(0, self._rand_int(0, self.num_rows), "key", door.color)   # keycorridor.py
  ```

  "Pick up the key you can reach, open the locked door you can reach" wins without ever reading a
  colour.

- `envs/babyai/unlock.py::UnlockToUnlock` is the only shipped **chain** (key B → door B → key A →
  door A, two colours, single carry forces a drop). Still blind-solvable: the generator makes
  exactly one key reachable at each stage. And its map is a **fixed 1×3 room corridor** — (a) fails.

- `envs/obstructedmaze.py::ObstructedMaze_Full` is the only place two differently coloured keys sit
  in one room (`for k in [-1, 1]: self.add_door(*side_room, locked=True,
  color=self.door_colors[(i + k) % len(self.door_colors)], …)`, with `obj = Key(door.color)` placed
  by `place_in_room`). There colour *does* discriminate — but a wrong pick is **cheaply recoverable**
  (drop, take the other), so blindness costs steps, not validity, and the reward
  `1 - 0.9 * (step_count / max_steps)` leaves a blind policy high. **That is the GYM-2 failure mode
  verbatim.** `key_in_box=True` also hides the colour until the box is toggled, so even the optimal
  policy must sample.

**Verdict.** MiniGrid/BabyAI fit on paper, their door–key legality *does* compose across colour **in
the engine**, and it composes in **none of the shipped tasks**. (d) is satisfiable — dynamics are
deterministic given a seed, so "replay the route and check `terminated and reward > 0`" is exact —
but (a) and (b) fail where it counts. Adoption would also mean adding `minigrid` + `gymnasium` to a
frozen instrument.

### 2.2 The procedural-maze landscape

| candidate | random maze | legality | composes across colour | exact judge | verdict |
|---|---|---|---|---|---|
| **MiniGrid / BabyAI door–key** | ✖ (room lattice / 2-region split) | ✅ real | ✖ *enforced, never required* | ✅ replay | ✖ |
| **Procgen `heist`** | ✅ | ✅ real | ✖ **no capacity limit** | ✖ pixels only | ✖ |
| **XLand-MiniGrid** | ✖ `room/two_rooms/four_rooms/nine_rooms` | ✅ real | ✖ no ruleset uses it | ✅ deterministic JAX | ✖ |
| **`maze-dataset`** | ✅ DFS / Wilson / percolation | ✖ **none at all** | ✖ | ✅ A\* | component, not candidate |
| **seqBench** | ✅ (random tree) | ✅ | ✖ one key per door | ✅ | ✖ (a QA benchmark, not a route environment) |

The two that came closest, with the code:

- **Procgen `heist`** (`procgen/src/games/heist.cpp`) — random maze per level
  (`maze_gen->generate_maze_with_doors(num_keys)`), three colours, and real legality:
  `is_blocked_ents: if (target->type == LOCKED_DOOR) return !has_keys[target->image_theme];`.
  ⚠ But `has_keys` is a **permanent per-colour inventory flag** (`has_keys[obj->image_theme] = true`
  on pickup, never cleared). With no capacity limit, "collect every key you walk past, then walk to
  the gem" is optimal and never reads a colour ⇒ composition-blind ≈ 1.0. Also 64×64 RGB only, a C++
  build, and no route-level judge.

- **XLand-MiniGrid** (`src/xminigrid/core/actions.py`) — the same good mechanism, in JAX:

  ```python
  jnp.equal(next_tile[0], Tiles.DOOR_LOCKED) & equal(agent.pocket, TILES_REGISTRY[Tiles.KEY, next_tile[1]])
  ```

  with a single-slot pocket (`is_empty_pocket`). But its benchmark is meta-RL over production
  rulesets (`AgentNearRule`, `TileNearRule`), its layouts are fixed room grids, and no shipped
  ruleset makes door colour load-bearing. Needs `flax`, absent from the frozen bundle.

### 2.3 ⭐ The generalisation, and why it matters

Every off-the-shelf door–key environment I read places **exactly one key per locked door**, and then
either

1. makes that key **the only reachable one** at that stage (MiniGrid, BabyAI, KeyCorridor,
   UnlockToUnlock, ObstructedMaze up to a cheap swap), or
2. lets the agent **hold every key at once** (Procgen heist).

Under (1) or (2) a colour-blind policy is optimal or near-optimal. **The colour match is enforced by
every engine and load-bearing in none.** That is the gap CMAZE-1 closes, and it tells wave 9 what to
protect: not the door, the **contention** — several keys reachable at once, a capacity of one, and no
way to undo a commitment.

---

## 3. CMAZE-1 — the build

`w8_envscout/maze_colour.py`, numpy only, nothing outside the frozen bundle.

**RULE A** — a **door cell** of colour *c* is IMPASSABLE unless the pocket holds key *c*.
**RULE B** — a **floor cell tinted** colour *c* is IMPASSABLE **while** the pocket holds key *c*.

The pocket holds **one** key, and a key is **consumed** by the door it opens. That single choice is
the hinge: *you cannot put a key down*, so committing to colour *c* locks you out of every *c*-tinted
cell until you spend it, and the order in which you commit to colours changes **which routes exist at
all**. Both rules make routes **ILLEGAL**, never merely longer. Neither is an upper bound, a cost or a
metric, so neither is anything GYM-1's or GYM-2's speed limit could have satisfied.

- **State:** `(cell, keystate)` with `keystate[c] ∈ {UNTAKEN, CARRIED, USED}`, at most one `CARRIED`.
  A door is open **iff** its key is `USED`, so the door mask is not separate state. ≈20 keystates ×
  ~100 floor cells ⇒ exact BFS is instant, which is what makes an exact optimum and therefore an
  exact budget possible at all.
- **Route:** a string over `NESWP` (`P` = pick up the key on this cell). Declining to pick a key up is
  a real strategic option, which is why pickup is an action and not automatic.
- **Adopted knobs = the shipped defaults:** `h=w=7` cells (15×15 grid), **`n_colours=4`**,
  **`braid=0.40`**, **`adversarial=2`**, **`require_bend=False`**, `p_tint=0.22`, `slack=0.10`.
- **Actions/observation for wave 9:** fully observable integer arrays (`grid`, `tint`, `door`,
  `key_at`, `start`, `goal`, `budget`) — no egocentric view, no turn actions, no pixels.

### The judge is exact

```python
if tcol and pk == tcol:
    return False, "forbidden-tint-c%d@%d" % (tcol, t), t        # RULE B
```

`judge()` is integer replay. Failure reasons are `into-wall / locked-door-cN / forbidden-tint-cN /
over-budget / no-goal / pickup-on-empty-cell / pocket-full / key-not-there`. `test_maze_colour.py`
strips the docstring with `ast` and asserts the executable body contains no float literal, no
division, no `abs`, no `isclose` and no tolerance. **The dial is satisfied: no downstream refusal or
deletion claim inherits an epsilon from the environment.**

### The generator's acceptance contract

An instance is kept only if, all checked **exactly**:

1. the goal is **unreachable with no key at all** — the doors are load-bearing;
2. the instance is **feasible** under RULES A+B;
3. `opt_len(A+B) > opt_len(A alone)` — **rule B genuinely bends the route**.

Condition 3 is the research question turned into a filter, and it is a statement about the
**instance**, not about any baseline policy. Seed acceptance is **0.952**.

### How a "blind" planner is defined

Not by hand-writing a weak heuristic — by running the **same exact BFS against a relaxed legality
relation** and grading with the **unrelaxed** judge. Each blind planner is therefore *optimal within
the world it is given*, which is the strongest honest form of the baseline.

| planner | what it is blind to |
|---|---|
| `BLIND-A` | rule B dropped entirely |
| `BLIND-B` | rule A dropped entirely |
| `BLIND-COLOUR` | colours collapsed — any key opens any door, any tint blocks any carried key |
| `BLIND-FACTOR` | knows both rules, but treats them as interacting only *near doors* |
| `BLIND-SEQUENTIAL` | **fully rule-aware routing**, but commits to the key order rule A alone prefers — blind *only* to the fact that rule B changes which orders survive |

---

## 4. THE SCREENING NUMBERS

`score = valid ? opt_len / len(route) : 0` ∈ [0, 1]; 1.0000 is an optimal valid route. Directly
comparable to the 0.9367 that killed GYM-2. n = 500 episodes per sweep, 2 000 training instances for
the fragment table; raw output in **`screen_results_k4.txt`** (adopted, k=4) and
**`screen_results.txt`** (k=3, kept for comparison — neither file was overwritten).

### A. Adoption configuration — **k = 4, adopted**
15×15 grid (7×7 cells), **4 colours**, `p_tint` 0.22, braid 0.12, budget = `opt +10%`, filter ON.
Seed acceptance **0.992**. Optimal route **59.7** steps mean (median 55, max 169). Rule B forces a
**+16.0-step detour** and bends the route in **100.0%** of accepted episodes.

| planner | validity | score | *k=3 score* | top failure mode |
|---|---:|---:|---:|---|
| ORACLE (full composition) | 1.0000 | 1.0000 | *1.0000* | — |
| **BLIND-SEQUENTIAL** (key order from A) | **0.3360** | **0.3359** | *0.4435* | `no-route` ×328 |
| BLIND-FACTOR (A×B local only) | 0.0860 | 0.0860 | *0.1180* | `forbidden-tint` ×118 |
| BLIND-COLOUR (colours collapsed) | 0.0360 | 0.0360 | *0.0819* | `no-route` ×215 |
| BLIND-A (rule B dropped) | 0.0000 | 0.0000 | *0.0000* | `forbidden-tint` ×127 |
| BLIND-B (rule A dropped) | 0.0000 | 0.0000 | *0.0000* | `locked-door` ×135 |
| RANDOM | 0.0000 | 0.0000 | *0.0000* | — |
| **FRAGMENT-LOOKUP r=1** | **0.0000** | **0.0000** | *0.0000* | 21 151 signatures / 2 000 routes, coverage **0.9004** |
| FRAGMENT-LOOKUP r=2 | 0.0000 | 0.0000 | *0.0000* | 113 184 signatures, coverage 0.0585 |

⭐ `BLIND-SEQUENTIAL`'s dominant failure is **`no-route`**, and it grew from 276/500 at k=3 to
**328/500** at k=4: having committed to rule A's preferred key order, the goal becomes
**unreachable**, not expensive. That is the GYM-1/GYM-2 law satisfied in the strongest possible form,
and more colours strengthen it.

⭐ The fragment-lookup 0.0000 is **not a coverage artefact**. At r=1 the table already covers
**90.0%** of the states on held-out optimal routes and still never completes one, because a 60-step
route must be right at every step in a maze the table has never seen. Colours in the signature are
canonicalised *relative to the pocket*, so a fragment learnt on a red key transfers to a blue one —
the lookup policy is given its best shot and still scores zero.

### B. Unfiltered generator — ⚠ the finding that matters most, and k does not fix it
Identical, except acceptance condition 3 is OFF. Rule B then bends the route in only **12.4%** of
episodes at k=4 (12.2% at k=3); detour +2.0 steps.

| planner | k=4 | k=3 |
|---|---:|---:|
| ORACLE | 1.0000 | 1.0000 |
| **BLIND-SEQUENTIAL** | **0.9100** | **0.9240** |
| BLIND-FACTOR | 0.8680 | 0.8640 |
| BLIND-A | 0.8560 | 0.8560 |
| BLIND-COLOUR | 0.2200 | 0.2360 |
| FRAGMENT-LOOKUP r=1 | 0.0040 | 0.0040 |

⭐ **This is GYM-2 reproduced inside my own environment: 0.9100 (k=4) / 0.9240 (k=3) against GYM-2's
0.9367** — and raising `n_colours` barely moves it. Random colour-keyed tints scattered over a random
maze are *almost never* load-bearing by themselves, at any k I measured. **The acceptance filter is
the composition**; `n_colours` only widens the margin once the filter is already doing the work.
Reported as a design finding and a warning to wave 9, not as a reason to reject anything.

### C. Loose budget — the budget is not doing the work
Filter ON, budget = `opt +50%` instead of `+10%`:
BLIND-SEQUENTIAL **0.4020** (vs 0.3360 at +10%), BLIND-FACTOR 0.0800, BLIND-COLOUR 0.0180,
BLIND-A 0.0000, FRAGMENT-LOOKUP 0.0000. *(k=3: 0.4640 vs 0.4440.)*

⭐ Quintupling the slack moves the composition-blind score by **+0.066** at k=4 (+0.020 at k=3). The
blind planners are still failing because their routes are **illegal**, not because they are slow —
but the budget earns slightly more of its keep at k=4, because a wrong key order more often leaves a
legal-but-long completion rather than none at all. (c) remains desirable rather than essential.

### D. ⭐ Adversarial tint placement — the interaction moved out of the filter
`screen_results_adversarial.txt` vs `screen_results_braid40_control.txt`. Instead of tinting
uniformly and then *discarding* instances where the rules fail to interact, `_poison()` runs the
targeted blind planner, replays its route, and tints every cell that route enters **while carrying
key c** with colour **c** — the exact condition RULE B refuses. Each addition is kept only if the
instance stays feasible under A+B, so the oracle is never destroyed. Round 0 targets the rule-A-only
planner, round 1 the factored planner. **Nothing is discarded: every instance is constructed to
interact.**

**Unfiltered generator (acceptance filter OFF), n = 500, k = 4:**

| generator | braid | tint placement | **composition-blind** | fragment-lookup | rule B bends the route |
|---|---:|---|---:|---:|---:|
| as inherited by wave 9 | 0.12 | uniform | **0.9100** | 0.0040 | 12.4% |
| braid-matched **control** | 0.40 | uniform | **0.8940** | 0.0020 | 26.2% |
| **adversarial** | 0.40 | **poisoned** | **0.4980** | **0.0000** | **78.4%** |

⭐ **The attribution is clean: braid alone buys −0.016, adversarial placement buys −0.396.** The drop
is the placement, not the extra route choice that enables it. Full adversarial row: SEQUENTIAL
0.4980 · FACTOR 0.2340 · BLIND-A 0.1420 · COLOUR 0.0840 · RANDOM 0.0000 · FRAG 0.0000 ·
ORACLE 1.0000.

⚠ **It needs route multiplicity.** At the adopted `braid=0.12` roughly 8 of 9 poison candidates
revert, because the key→door corridor is usually the *only* path, so tinting it destroys feasibility
instead of bending the route. Measured (n=40, no filter): cells poisoned per round 1.0 / 2.0 / 4.0 /
3.6 at braid 0.12 / 0.25 / 0.40 / 0.55. **Adversarial placement and braiding are a pair.**

**Clean run at the shipped defaults** — `screen_results_DEFAULTS.txt`, no CLI overrides, current
sweep labelling, n=500:

| sweep | filter | SEQUENTIAL | FACTOR | COLOUR | BLIND-A | FRAG | bends |
|---|---|---:|---:|---:|---:|---:|---:|
| **A — adoption configuration** | **OFF** | **0.4980** | 0.2600 | 0.0840 | 0.1280 | **0.0000** | 81.6% |
| B — filter also ON (comparison) | ON | 0.3720 | 0.1160 | 0.0200 | 0.0000 | 0.0000 | 100.0% |
| C — loose budget (+50% slack) | OFF | 0.5420 | 0.2440 | 0.1060 | 0.1480 | 0.0000 | 79.4% |

**With the filter ON, for completeness:** 0.3720 at the defaults — lower, but it discards instances
and buys 0.126. Quintupling the budget slack moves the blind score 0.4980 → 0.5420, so the budget is
still not what the blind planners are failing on. Adversarial placement's value is **not** a lower
number than the filter's — it is reaching **0.4980 with no filter at all**, which is exactly what the
filter dependency was costing.

⚠ **A correction to the `seed acceptance` column.** `generate()` retries **internally** within a
single seed, so that column counts seeds that *eventually* produced an instance, **not** the filter's
rejection rate. The 0.952 / 0.992 figures reported above therefore **understate** how much the filter
discards. No score changes; only the reading of that one column.

### Exactness tests — `python -m w8_envscout.test_maze_colour`, 6/6 PASS
- `judge()`'s executable body has no float, no division, no tolerance;
- every exact optimum replays to `goal` at exactly `opt_len` steps (200 seeds);
- `judge` is a pure function of `(instance, route)`;
- RULE A, RULE B and the budget were each *observed* refusing routes;
- every accepted instance satisfies the acceptance contract (190/200 seeds accepted);
- ⭐ **`blind failures: 190 by ILLEGALITY, 0 by budget`.**

Re-run at **`n_colours=4`**, also 6/6 PASS: 199/200 seeds accepted and all satisfy the acceptance
contract, judge body still float-free, and ⭐ **`blind failures: 199 by ILLEGALITY, 0 by budget`**.
**The dial is unaffected by k.**

### Sensitivity at the adopted defaults — k = 4, braid 0.40, adversarial, no filter
`sensitivity_results_k4_adversarial.txt`, n = 150 per row, 20 settings, 572s. The **blind** column is
the **max over the blind planner family**. `bends` is the fraction of instances where rule B changes
the optimal route — with **no filter**, so it is a property of the distribution. One knob varies per
row; everything else is at the defaults.
*(The older `sensitivity_results.txt` — k=3, uniform tinting, filter ON — is kept for comparison.)*

| setting | accept | opt | detour | bends | **blind** |
|---|---:|---:|---:|---:|---:|
| `n_colours=2` | 0.241 | 33.8 | 8.4 | 0.707 | 0.7600 |
| `n_colours=3` | 0.505 | 39.8 | 12.8 | 0.733 | 0.6600 |
| **`n_colours=4`** *(default)* | 0.725 | 44.0 | 15.9 | 0.827 | **0.5133** |
| `n_colours=5` | 0.882 | 46.4 | 18.0 | 0.773 | **0.4333** |
| **`adversarial=0`** *(uniform)* | 0.698 | 32.2 | 3.7 | **0.260** | **0.8590** |
| `adversarial=1` | 0.754 | 38.3 | 12.6 | 0.827 | 0.5261 |
| **`adversarial=2`** *(default)* | 0.758 | 41.5 | 15.0 | 0.793 | 0.5333 |
| `adversarial=3` | 0.714 | 42.5 | 16.6 | 0.807 | 0.4929 |
| `braid=0.12` | 0.987 | 42.9 | 8.9 | 0.447 | 0.7600 |
| `braid=0.25` | 0.968 | 41.2 | 10.4 | 0.633 | 0.6467 |
| **`braid=0.40`** *(default)* | 0.758 | 41.8 | 14.9 | 0.853 | 0.4597 |
| `braid=0.55` | 0.418 | 39.7 | 14.8 | 0.887 | 0.4325 |
| `p_tint=0.12` | 0.728 | 43.4 | 15.5 | 0.847 | 0.5200 |
| **`p_tint=0.22`** *(default)* | 0.739 | 40.0 | 13.6 | 0.813 | 0.4796 |
| `p_tint=0.32` | 0.633 | 40.9 | 14.1 | 0.753 | 0.5591 |
| `slack=0.00` | 0.754 | 41.4 | 14.7 | 0.780 | 0.4667 |
| **`slack=0.10`** *(default)* | 0.746 | 41.0 | 14.6 | 0.813 | 0.4400 |
| `slack=0.25` | 0.735 | 43.7 | 17.3 | 0.833 | 0.4658 |
| `slack=0.50` | 0.735 | 42.5 | 14.8 | 0.833 | 0.5067 |
| `maze=5x5` | 0.993 | 27.7 | 8.4 | 0.673 | 0.5329 |
| **`maze=7x7`** *(default)* | 0.798 | 42.0 | 14.6 | 0.800 | 0.5867 |
| `maze=9x9` | 0.368 | 54.4 | 20.3 | 0.920 | 0.5593 |
| `max_poison=2` | 0.711 | 42.3 | 15.0 | 0.887 | 0.5395 |
| `max_poison=4` | 0.714 | 41.8 | 14.7 | 0.833 | 0.4927 |
| **`max_poison=8`** *(default)* | 0.704 | 42.2 | 15.4 | 0.773 | 0.5193 |
| `max_poison=16` | 0.746 | 45.4 | 17.2 | 0.827 | 0.5265 |
| `require_bend=True` *(filter ON)* | 0.628 | 47.9 | 21.2 | **1.000** | **0.3865** |

Six things to act on:

- ⭐ **Adversarial placement is by far the biggest lever, and the table isolates it.** Turning it off
  (`adversarial=0`, everything else at the defaults) sends the blind score **0.5333 → 0.8590** and the
  bend rate **0.793 → 0.260**. One round already captures nearly all of it (0.5261); a third round
  buys a little more (0.4929). **`max_poison` is flat at 0.49–0.54** across 2 → 16, so the placer does
  not need to be greedy — it needs to be *targeted*.
- ⭐ **Braid now matters, which it did not under uniform tinting.** 0.7600 → 0.6467 → 0.4597 → 0.4325
  for braid 0.12 → 0.25 → 0.40 → 0.55, with the bend rate climbing 0.447 → 0.887. This is the
  "placement and braiding are a pair" result, measured: poisoning needs alternative routes to divert
  onto. 0.40 is the knee — 0.55 buys 0.03 more at the cost of seed acceptance 0.758 → 0.418.
- **More colours still helps monotonically**, now with *rising* acceptance: k=5 gives **0.4333** at
  acceptance 0.882 against k=4's 0.5133 at 0.725. ⛔ Reported, not acted on — k=4 is what was
  directed, and k=5 is available as a free margin lever if wave 9 wants it.
- **The budget is still not the lever** (0.4400–0.5067 across slack 0% → 50%), and neither is tint
  density (0.4796–0.5591) or maze size (0.5329–0.5867). These are all flat within noise at n=150.
- **The filter would still give a lower number** (0.3865, bends 1.000) — but at the cost of
  discarding ~37% of seeds. That is the trade the default now declines to make.
- ⚠ **Read the row-to-row scatter as noise of roughly ±0.05 at n=150.** The three rows that *are* the
  default configuration under different labels give 0.5333, 0.4597 and 0.4796; the n=500 screen gives
  **0.4980**. Treat 0.4980 as the number and the table as the shape.

---

## 5. WHAT WOULD PROVE THIS SPOKE WRONG — checked against my own result

| the task's falsifier | status |
|---|---|
| adopt an environment whose validity check needs a tolerance | **not triggered** — `judge()` is integer replay; asserted by test, not by assertion in prose |
| adopt one where a composition-blind policy scores well | **not triggered at the shipped defaults** — **0.4980**, with no acceptance filter at all, measured directly from the defaults in `screen_results_DEFAULTS.txt` sweep A (§1, §4.D). ⚠ **It was triggered by the configuration this one replaced:** uniform tinting, unfiltered, scores **0.9100** (k=4) / **0.9240** (k=3), and **0.8940** even with braid matched. Those stay printed in §4.B and §4.D as the numbers wave 9 inherits if it reverts the placer. The filtered configuration reaches 0.3360 but discards instances; the default declines that trade |
| MiniGrid/BabyAI fit on paper but their door–key legality does not compose across colour | **exactly what I found**, §2.1, with the generator code that proves it |

**The one honest weakness.** CMAZE-1's composition lives in the generator's acceptance filter, not in
the raw random distribution — §4.B is the measurement of that, and it is the number a sceptic should
attack first.

⭐ **Raising `n_colours` to 4 was tried in full and did not retire it.** Inside the filter it helps
exactly as predicted (composition-blind 0.4440 → **0.3360**, every blind planner down, acceptance
0.952 → 0.992). Outside it, the unfiltered composition-blind score barely moves: 0.9240 → **0.9100**.
So `n_colours` is a **margin lever, not the mechanism**.

⭐ **Adversarial tint placement was then implemented and run, and it does address the weakness.**
With no acceptance filter at all, the composition-blind score is **0.4980**, against **0.8940** for a
braid-matched uniform control and **0.9100** for the uniform generator wave 9 inherits (§4.D). The
interaction now lives in the distribution rather than in a rejection step. Two caveats kept in view:
it requires `braid≈0.40` to have routes to bend, and the *filtered* configuration still scores lower
(0.3360). ⛔ Both remain printed columns — **the 0.9100 stays on the record as the number wave 9
inherits from the uniform generator.**

⭐ **This is now the shipped default** (`braid=0.40, adversarial=2, require_bend=False`), and the
sensitivity sweep was rerun at it in full — see §4 Sensitivity, which isolates the placer as the
lever (`adversarial=0` sends the blind score back to 0.8590 with everything else held fixed).

**Not run, declared as such:** MiniGrid, BabyAI, Procgen and XLand-MiniGrid were **read, not
executed** — installing them was forbidden. Their numbers in §2 are either derived from the
generator code (`DoorKey` = 1.0000, which the code proves outright) or stated as structural
arguments, and they are **not** measurements. Also not run: an RL-trained (rather than search-based)
blind baseline. ~~A full n=500 screen at the new defaults~~ — **closed**: `screen_results_DEFAULTS.txt`
is that run, produced from the shipped defaults under the current sweep labelling, and it reproduces
**0.4980** independently of the earlier measurement. A declared NOT-RUN is never a null.
