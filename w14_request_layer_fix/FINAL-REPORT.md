# w14 `spoke/request-layer-fix` — FINAL REPORT

**Agent:** `experiment-engineer`. **Scored runs spent: 0 of 0.** Local CPU only, 3 concurrent
processes, no GPU, no cluster. Branch `w14/request-layer-fix` off `6699665`
(`[hub] base/iter8: fold w13/encoder-adapt`). Worktree
`C:\chlu\worktrees\CHLU-w14-request-layer-fix`; `chlu.__file__` asserted inside it before any
number was taken. Output dir `/c/chlu/outputs/w14-request-layer-fix/`, mirrored in the repo at
`w14_request_layer_fix/`; code at `w14_request_layer/`.

> ## VERDICT
> **`REQUEST-LAYER-READY.json` — `"status": "ready"`, all four acceptance conditions PASS,
> `tolerance_used: null`.** The repair is **one acceptance rule** in the request layer:
> *a pair is kept iff its RULES A+B optimal route picks up at least two keys.*
> **Key-count histogram 1242 / 198 / 9 (of 1449) → 3825 / 751 / 35 (of 4611): ZERO one-key
> routes, `fraction_multi_key` 0.1428 → 1.0000.** **BLIND-SEQUENTIAL on held-out rows falls
> 0.9077 → 0.1202–0.2344** (0.8295 → 0.1068–0.1810 at CMAZE-1's own slack-10 % budget).
> **ORACLE stays 1.0000 on all 51 row × budget-condition × split blocks of the three
> fixed banks, and on every block of the BEFORE bank.** `w8_envscout.maze_colour` — the
> engine, the solver and the **exact integer judge** — is imported unmodified; its file
> sha256 `9eacdda17d87cac0…` is recorded and re-checked.
> **28 of 28 independent verification checks pass** (`artefacts/VERIFY.txt`).

---

## 1. ⛔⛔ The diagnosis w9 handed down is right in direction and INCOMPLETE in mechanism

w9 wrote (`/c/chlu/evidence/outputs/_thinkpad/w9/w9-maze/LOG.md:92,413`): *"1242 of 1449
optimal routes pick up exactly ONE key … with one key there is no key ORDER for rule B to
trap"*, and prescribed *"a request layer must force pairs whose routes cross two or more
doors."*

**Before building anything I measured CMAZE-1 proper** — `generate()` at its own defaults,
n = 500, seeds 0–499, zero rejections (`artefacts/anchor_cmaze1.json`):

| planner | validity |
|---|---|
| ORACLE | **1.0000** |
| BLIND-A (rule B dropped) | 0.1540 |
| BLIND-B (rule A dropped) | 0.0000 |
| BLIND-COLOUR | 0.1000 |
| BLIND-FACTOR | 0.2620 |
| **BLIND-SEQUENTIAL** | **0.5200** |

⭐ **0.5200 against the published 0.4980 and w9-harness's 0.5317 — the environment reproduces
a third time, in a third worktree.** Bend fraction 0.778 against the published 0.816.

⛔⛔ **And CMAZE-1's OWN key-count histogram is `{1: 368, 2: 95, 3: 34, 4: 3}` of 500 —
73.6 % single-key, against CMAZE-2's 85.7 %.** A single-key majority is therefore **not by
itself** what separates 0.4980 from 0.9037. The mechanism had to be measured.

### The mechanism, measured (`artefacts/anchor_breakdown.txt`, CMAZE-1 proper, n = 400)

| | n | BLIND-SEQUENTIAL valid |
|---|---|---|
| rule-A key order **==** A+B key order | 201 | **201 (1.0000)** |
| rule-A key order **!=** A+B key order | 199 | **3 (0.0151)** |

**By key count of the A+B optimum: 1 key 0.6532 (194/297) · 2 keys 0.1351 (10/74) ·
3 keys 0.0000 (0/26) · 4 keys 0.0000 (0/3).**

⭐ So ≥ 2 keys **is** the right lever, for a reason that is now explicit: with ONE key the
constraint `key_order=(c,)` admits exactly one pickup of exactly colour c — which is what the
oracle does too — so **BLIND-SEQUENTIAL *is* the oracle on a one-key route.** CMAZE-1 still
fails on 34.7 % of its one-key routes only because its single key can be the *wrong* one
(several doors joining the two components). CMAZE-2's cut is a single door, so even that
escape is closed and blind-sequential goes to ~1.

---

## 2. ⛔⛔ THE ROOT CAUSE, and it is sharper than "the layer under-exercises rule B"

Two probes, run before the fix was written:

- `artefacts/probe_density.txt` — 24 random cells per maze, all ordered pairs, 12 mazes. The
  ">= 2 keys" column is **0.00 or 1.00 and nothing between**. The key count is a property of
  the **door CUT**, hence of the **maze**, not of the pair.
- `artefacts/probe_direction.txt` — 6 agents in one door-free component, 6 targets in the
  other, 18 cross pairs, both orientations, 60 seeds. Where the multi-key direction exists it
  is **18 / 18**; and in **every** such maze the **REVERSE direction is 0 / 18 FEASIBLE** —
  the key that opens the cut door lives on one side of the cut, so from the far side the
  target cannot be reached at all. **13 of 40** mazes with two usable components have such a
  direction.

⛔⛔ **w9's layer required `min_pairs = 12` cross pairs drawn SYMMETRICALLY in both
directions. A maze whose cut needs two keys cannot supply the reverse direction at all.
The symmetry requirement was itself the filter that selected single-key cuts.** The one-key
routes were not an accident of the generator; they were forced by the request layer's own
acceptance test.

---

## 3. The repair — `w14_request_layer/task2.py`

```
a pair is kept iff its RULES A+B optimal route picks up at least MIN_KEYS = 2 keys
```

The count is read off the solver's own route by `w8_envscout.screen.key_order_of`,
unmodified. **It is an integer. No tolerance enters anywhere.** Everything else is w9's layer
line for line: `generate()` at its own defaults, `PER_SIDE = 3`, `MIN_PAIRS = 12`, cross pairs
only, two rounds of generalised adversarial tinting, the walled target, the budget words.

Two consequences, both **declared**:

1. ⚠ **All six agent words sit in the SOURCE component and all six target words in the
   DESTINATION one**, because the reverse direction is infeasible. The kept set is therefore
   **no longer symmetric in direction**; direction is a CONSTANT of the split, not a balanced
   factor. It cannot confound a held-out comparison because it does not vary — but it is a
   difference from CMAZE-2 and it is stated rather than hidden.
2. ⚠ **The ≥ 2-key screen runs on the UNSEALED maze**, before the walled target is chosen, so
   the walls that create the refusal condition cannot be what creates the multi-key structure.

**Cost of the repair: generator seeds, not code.** Acceptance falls from **0.2000** (before)
to **0.1649 / 0.1661 / 0.1375** (after, seeds 0/1/2).

---

## 4. ⭐⭐ THE CONTROL — the BEFORE bank reproduces wave 9 EXACTLY

96 sites at w9's own seeds with `w9_maze.task.build_site` untouched
(`artefacts/measure_before_0.json`):

| quantity | wave 9 (`runs/scored1_seed0_test.json`, `LOG.md`) | this worktree |
|---|---|---|
| key-count histogram | **{1: 1242, 2: 198, 3: 9}, n = 1449** | **{1: 1242, 2: 198, 3: 9}, n = 1449** |
| `n_train_requests` | 1208 | **1208** |
| rows seen / unseen_pair / nway / budget_interp | 1208 / 195 / 389 / 434 | **1208 / 195 / 389 / 434** |
| unseen_pair BLIND-SEQUENTIAL | 0.907692 | **0.9077** |
| unseen_pair BLIND-A | 0.358974 | **0.3590** |
| unseen_pair BLIND-COLOUR | 0.230769 | **0.2308** |
| unseen_pair BLIND-FACTOR | 0.456410 | **0.4564** |
| unseen_pair bend fraction | 0.512821 | **0.5128** |
| nway BLIND-SEQUENTIAL | 0.922879 | **0.9229** |
| FRAG-r1 same-maze | 0.059932 | **0.0599** |
| VAL unseen_pair BLIND-SEQUENTIAL | 0.8750 (`LOG.md`) | **0.8750** |

⭐⭐ **Independent re-derivation: different worktree, different process, nothing read from w9's
artefacts.** The BEFORE/AFTER comparison below is therefore against wave 9 itself.

⚠ **One number does not reproduce and it is reported:** cross-maze fragment lookup reads
**0.0000 / 0.0000** here where w9 reported **0.0006 / 0.0000**. 0.0006 is one valid rollout in
~1700; my pool bank is 48 sites, so the single lucky rollout did not recur. Both are
effectively zero and both are printed.

---

## 5. ⭐⭐ ACCEPTANCE CRITERION 1 — the key-count histogram

| | 1 key | 2 | 3 | 4 | n routes | fraction ≥ 2 keys |
|---|---|---|---|---|---|---|
| **BEFORE** (w9 layer, seed 0) | **1242** | 198 | 9 | — | 1449 | **0.1428** |
| **AFTER** (w14 layer, seeds 0+1+2) | **0** | 3825 | 751 | 35 | **4611** | **1.0000** |
| *per seed, AFTER* | 0 / 0 / 0 | 1228 / 1271 / 1326 | 284 / 274 / 193 | 18 / 0 / 17 | 1530 / 1545 / 1536 | 1.0000 |
| *CMAZE-1 proper, for reference* | 368 | 95 | 34 | 3 | 500 | 0.2640 |

⭐⭐ **Not "a large majority" — ALL of them.** Re-derived from the bank pickles by an
independent checker that does not import the reporting code: `artefacts/VERIFY.txt`,
*"no route uses fewer than 2 keys — min keys = 2"* on all three AFTER banks.

---

## 6. ⭐⭐ ACCEPTANCE CRITERION 2 — BLIND-SEQUENTIAL falls back

Validity under the **exact integer judge**. Three budget conditions, three seeds. `SLACK10` is
every kept pair at CMAZE-1's own rule `opt + ceil(0.10·opt)` — the column directly comparable
with the published 0.4980 and with w9's own sentence that this measurement left CMAZE-2 at
1.0000.

| condition · row | BEFORE (w9 layer) | AFTER s0 | AFTER s1 | AFTER s2 |
|---|---|---|---|---|
| **SLACK10, all pairs** | **0.8295** | **0.1810** | **0.1553** | **0.1068** |
| GRID_W9 test · unseen_pair | **0.9077** | 0.2344 | 0.2000 | 0.1818 |
| GRID_W9 test · nway | 0.9229 | 0.2689 | 0.2451 | 0.1520 |
| GRID_W9 test · budget_interp | 0.9217 | 0.2437 | 0.2381 | 0.1746 |
| GRID_W9 val · unseen_pair | 0.8750 | 0.3188 | 0.2982 | 0.1803 |
| GRID_REBASED test · unseen_pair | — | **0.1759** | **0.1566** | **0.1202** |
| GRID_REBASED test · nway | — | 0.2145 | 0.1602 | 0.1197 |
| GRID_REBASED val · unseen_pair | — | 0.2451 | 0.1900 | 0.1111 |

**Published CMAZE-1: 0.4980. w9's CMAZE-2 scored seeds: 0.9077 / 0.8905 / 0.9130.**
The worst held-out cell of each budget condition is **GRID_W9/val 0.3188 · GRID_W9/test
0.2689 · GRID_REBASED/val 0.2451 · GRID_REBASED/test 0.2145 · SLACK10 0.1810** — every one of
them **≤ 0.4980**. ✔

### ⛔ It lands BELOW 0.4980, not at it — and that is the expected place, not an overshoot

0.4980 is a **MIXTURE** over CMAZE-1's own key counts: 73.6 % of CMAZE-1's routes are
single-key and score **0.6532**, the rest score **0.1351 / 0.0000**. CMAZE-2b is **100 %
multi-key**, so it must sit at CMAZE-1's **multi-key band**, not at CMAZE-1's marginal.
**0.1068–0.1810 at SLACK10 against CMAZE-1's own 2-key 0.1351 and 3-key 0.0000** is exactly
that band. ⭐ The fixed layer reproduces CMAZE-1's *conditional* number; it does not
manufacture a new one.

### The other blind arms, and the bend fraction

| | BEFORE (SLACK10) | AFTER s0 (SLACK10) |
|---|---|---|
| ORACLE | 1.0000 | **1.0000** |
| BLIND-A (rule B dropped) | 0.2843 | 0.0209 |
| BLIND-B (rule A dropped) | 0.0000 | 0.0000 |
| BLIND-COLOUR | 0.1415 | 0.0000 |
| BLIND-FACTOR (A×B local only) | 0.3782 | 0.0373 |
| **rule-B bend fraction** | **0.6184** | **0.9536** |

⭐ **The rule-B bend fraction rises from w9's CMAZE-2 0.484 to 0.93–0.98, above CMAZE-1's
published 0.816.** Mean detour 29.1 steps (opt 65.8 against rule-A-only 36.7).

---

## 7. FRAGMENT-LOOKUP — ⛔ two different quantities, both labelled

⛔ **The published CMAZE-1 figure (0.0006 / 0.0000) is the CROSS-MAZE one.** The same-maze
figure (w9's 0.0741 / 0.5105) is built from routes through the very maze being evaluated and
is a far stronger floor. They are not comparable and both are printed on every row.

| | r = 1 | r = 2 |
|---|---|---|
| **CROSS-MAZE** (published quantity), AFTER, all 3 seeds, both grids, both splits | **0.0000** | **0.0000** |
| CROSS-MAZE, BEFORE, this worktree | 0.0000 | 0.0000 |
| *published CMAZE-1* | *0.0006* | *0.0000* |
| **SAME-MAZE** (a DIFFERENT quantity), AFTER s0 test, GRID_W9 | 0.0820 | 0.5191 |
| SAME-MAZE, AFTER s0 test, GRID_REBASED | 0.0051 | 0.4829 |
| SAME-MAZE, AFTER s1 / s2 test, GRID_REBASED | 0.0000 / 0.0148 | 0.5692 / 0.5041 |
| SAME-MAZE, BEFORE test | 0.0599 | 0.5342 |
| *w9's CMAZE-2* | *0.0741* | *0.5105* |

⭐ **The split does not leak: the cross-maze tripwire reads exactly 0.0000 on every AFTER
bank.** The same-maze r = 2 floor stays where w9 found it (~0.48–0.57), so the repair has not
made the rows easier to memorise from fragments either.

---

## 8. ⛔ A SECOND request-layer defect the repair exposes — the ABSOLUTE budget grid

A two-key route is longer. Mean optimum **37.4 → 65.8** steps. w9's absolute budget words
30/45/60 then write almost nothing:

| | writable at the three words | train requests | unseen_pair n |
|---|---|---|---|
| BEFORE at 30/45/60 | 0.3389 / 0.7833 / 0.9241 | 1208 | 195 |
| AFTER at 30/45/60 | **0.0268 / 0.2007 / 0.4425** | **454** | **64** |
| AFTER at 55/85/105 | **0.3660 / 0.8059 / 0.9366** | **1442** | **199** |

⭐ **The rule for the rebased grid was fixed in `LOG.md` BEFORE it was applied and not moved:**
the three words sit at the same percentiles of the AFTER bank's route-length distribution that
w9's words occupy in the BEFORE bank's, rounded to the nearest 5; the interpolation words are
the midpoints. **The rule mentions route lengths and nothing else — no planner, no validity,
no blind column enters it.**

⭐ **Self-check:** applied with BEFORE as both source and target it returns **[30, 45, 60] —
w9's own grid, exactly.** ⚠ The interpolation words come back **[38, 52]** against w9's
**[38, 53]** because Python rounds 52.5 to even. **Declared, not silently fixed.**

⚠ **The rule is per-bank, so it gives three grids: [55, 85, 105] · [60, 85, 100] ·
[55, 75, 90].** BLIND-SEQUENTIAL is insensitive to the difference (0.1111–0.2451 across all of
them). ⛔ **The Hub must pick ONE grid before any scored run**; this spoke did not, because
freezing a grid is a pre-registration decision and not the spoke's to take.

### The refusal material and the novelty control (`artefacts/refusal_material.json`)

| | BEFORE | AFTER (w9 grid) | AFTER (rebased) |
|---|---|---|---|
| sites with a WALLED target | 93 / 96 | **56 / 96** | 56 / 96 |
| walled (IMPOSSIBLE) requests | 837 | **504** | 504 |
| budget-infeasible requests | 280 | 660 | 257 |
| `mt_unseen` control requests | 573 | **191** | **602** |
| mazes with `nway` / `unseen_pair` | 96 / 96 | 93 / 95 | 93 / 95 |

⚠ **DECLARED: the walled condition is 40 % scarcer.** All six target words now sit in one
component, so sealing one without breaking the other pairs' ≥ 2-key routes succeeds on 56 of
96 sites rather than 93. 504 impossible requests remain — enough for the DISPLAY refusal row,
but the Hub should know.
⭐ `mt_unseen` — the reachable-but-ungrounded control that stopped w9's refusal from being read
as wall detection — is **thin at w9's grid (191) and healthy at the rebased grid (602)**. That
is a second, independent reason to rebase.

---

## 9. ⛔ The exact judge did not move, and that is checked rather than asserted

- `w8_envscout.maze_colour.judge` is **imported**, not wrapped and not reimplemented.
  `provenance.judge` records module, qualname, path and **sha256
  `9eacdda17d87cac0…`**; `verify.py` re-hashes the file on disk and compares.
- `tolerance_used` is `null` at every occurrence in `REQUEST-LAYER-READY.json`.
- **Every stored route of every bank — 6060 routes — is re-judged at `budget = opt`, with no
  slack, and accepted: 1530/1530 · 1545/1545 · 1536/1536 · 1449/1449.** Route length equals
  the stored optimum on all of them, **0 mismatches**. ⭐ So ORACLE = 1.0000 is a property of
  the artefacts, not of the reporting.
- **ORACLE = 1.0000 on all 51 row × budget-condition × split blocks of the three AFTER
  banks** (`REQUEST-LAYER-READY.json -> ACCEPTANCE.3`, `distinct_values_seen: [1.0]`), and on
  every block of the BEFORE bank.
- ⛔ **`w8_envscout/maze_colour.py` and `w9_harness/judge_maze.py` were not edited.**
  `git show --stat` on this branch touches only `w14_request_layer/` and
  `w14_request_layer_fix/`.

---

## 10. ⛔ What would have proved this spoke wrong, and did not

| the task file's kill condition | what happened |
|---|---|
| multi-key routes exist but blind-sequential stays high ⇒ rule B does not compose here, report it and do not tune | **Did not occur.** 1.0000 multi-key **and** blind-sequential 0.1068–0.3188 on every held-out row of every seed. |
| forcing ≥ 2 keys makes ORACLE < 1.0000 ⇒ the generator produces unsolvable instances; stop | **Did not occur.** ORACLE 1.0000 on 64/64 blocks; 6141/6141 stored routes re-judged valid at zero slack. |
| the fix requires changing the JUDGE ⇒ STOP | **Did not occur.** The judge is imported; its file hash is recorded and re-checked; `tolerance_used` is null. |

---

## 11. ⛔ DECLARED NOT-RUN

1. **Every scored run.** 0 of 0 by the task file. No arm is fitted, no store exists, no RESULT
   row is claimed. The learned arms, the CLU, the encoder and the pipeline ceiling on the fixed
   split are `hub/read-verb`'s work; this file is its precondition, not its result.
2. **BEFORE banks at seed indices 1 and 2.** Only `before 0` (96 sites) and a 48-site
   `before 1` pool were built. Reason: the BEFORE column reproduces w9's published numbers
   exactly on one bank, so a second and third add nothing the spoke needs; the wall-clock went
   to three AFTER seeds instead.
3. **A frozen budget grid.** The rebasing rule is published and self-checked, but the three
   per-bank grids are not collapsed to one. Reason: freezing a budget grid is a
   pre-registration decision and belongs to the Hub.
4. **Any re-run of w9's arm sweep, refuser or DISPLAY rows on the fixed split.** Reason: the
   spoke owns the request layer, not the arms.
5. **GPU and cluster.** Local CPU only. CSF3 was not attempted.
6. **`cmaze1-easy` tier, RL arms, any scale above `std`.** Out of scope.

## 12. Limits

The AFTER column is **three banks of 96 sites, one split seed each** (val and test both
reported). The BEFORE column is **one bank** — chosen deliberately, see NOT-RUN 2. The
mechanism probes (`breakdown`, `density`, `direction`) are **n = 400 / 12 mazes / 60 seeds**
respectively, and the density probe's all-or-nothing reading rests on 12 mazes. The
`rule-B bend fraction` and the blind columns are **planner measurements on the environment**,
not arm results: nothing here says a learned object will do better or worse. The per-bank
rebased grids differ and the spoke did not choose between them. The walled-refusal condition is
40 % scarcer than w9's and that is a real reduction in DISPLAY material.

## 13. What the Hub is asked to decide

1. ⭐⭐ **Adopt CMAZE-2b as the request layer for `hub/read-verb`?** The precondition is met on
   every published criterion, and `REQUEST-LAYER-READY.json` says `ready`.
2. ⛔ **Freeze ONE budget grid.** The rule gives [55, 85, 105] · [60, 85, 100] · [55, 75, 90]
   on the three banks. The blind columns do not distinguish them. The spoke recommends
   **seed 0's [55, 85, 105], interp [70, 95]**, stated as a recommendation and not taken.
3. ⚠ **Accept the loss of direction symmetry?** It is forced: the reverse direction across a
   two-key cut is infeasible, so a direction-symmetric request layer *cannot* exercise rule B
   on this generator. The alternative is a different generator, which is out of this spoke's
   scope and would break the "environment adopted unmodified" line.
4. ⚠ **Accept 56/96 walled sites (504 impossible requests) for the DISPLAY refusal row**, or
   commission a separate walled-site construction.
5. ⭐ **A correction the programme should carry:** w9's stated cause — *"1242 of 1449 routes use
   exactly one key"* — is a symptom. **The cause is that w9's request layer demanded cross
   pairs in BOTH directions, and a two-key door cut is one-directional by construction.** That
   sentence should replace the symptom wherever the programme repeats it.
