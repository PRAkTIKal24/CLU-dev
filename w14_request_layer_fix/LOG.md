# w14 spoke/request-layer-fix — LOG

All times UTC. The box's local clock is UTC+2; every stamp below is converted.
Worktree: `C:\chlu\worktrees\CHLU-w14-request-layer-fix`, branch `w14/request-layer-fix`,
base `6699665` (`[hub] base/iter8: fold w13/encoder-adapt`).

## 06:14Z START

- `START_UTC.txt` written: `2026-09-22T06:14:36Z`.
- Import assertion: `chlu.__file__ = C:\chlu\worktrees\CHLU-w14-request-layer-fix\chlu\__init__.py`
  — inside THIS worktree. Python 3.11.9.
- Task read: `/c/chlu/tasks/w14-spoke-request-layer-fix.md`. Evidence read:
  `w9-`, `w10-`, `w12-advisor-report.md` under `/c/chlu/evidence/outputs/`.

## 06:20Z The defect, located in the code

`w9_maze/task.py::build_site` places `per_side` agents and `per_side` targets in **each of the
two LARGEST door-free components** and keeps only the CROSS pairs. `w8_envscout.maze_colour`
generates **4 doors** (one cell each, one per colour) in a 15x15 braided maze; removing 4 cells
from a braid=0.40 maze almost always leaves exactly **two** components, joined by **one** door.
A cross pair therefore needs **exactly one** door, hence **exactly one key** — and with one key
there is no key ORDER, so RULE B has nothing to trap. That is w9's measured
**1242 / 1449 routes use exactly one key** (198 two, 9 three), LOG line 92 of
`/c/chlu/evidence/outputs/_thinkpad/w9/w9-maze/LOG.md`.

w9's own prescription, same file line 415: *"a request layer must force pairs whose routes cross
two or more doors."* That is the fix this spoke implements — **in the request layer only**.
⛔ `w8_envscout/maze_colour.py` (engine, solver, exact judge) is NOT touched.

## 06:20Z The anchor — CMAZE-1 proper, so the target is a measured number and not a memory

`artefacts/anchor_cmaze1.json`, n=500 instances from `generate()` at ITS OWN DEFAULTS,
seeds 0..499, zero rejections:

| planner | validity | score |
|---|---|---|
| ORACLE | **1.0000** | 1.0000 |
| BLIND-A (rule B dropped) | 0.1540 | 0.1540 |
| BLIND-B (rule A dropped) | 0.0000 | 0.0000 |
| BLIND-COLOUR | 0.1000 | 0.1000 |
| BLIND-FACTOR | 0.2620 | 0.2620 |
| **BLIND-SEQUENTIAL** | **0.5200** | 0.5197 |

⭐ **0.5200 against the published 0.4980 and w9-harness's 0.5317 — the environment
reproduces a third time, in a third worktree.** Rule-B bend fraction 0.778 (published 0.816).

⛔⛔ **AND THE PUBLISHED DIAGNOSIS IS INCOMPLETE.** CMAZE-1's own key-count histogram is
**{1: 368, 2: 95, 3: 34, 4: 3} of 500 — 73.6 % of CMAZE-1's OWN routes use exactly one key**,
against CMAZE-2's 85.7 %. A one-key majority is therefore NOT by itself what separates
0.4980 from 0.9037. The mechanism had to be measured, not assumed.

## 06:22Z The mechanism, measured — `artefacts/anchor_breakdown.txt`

n=400 CMAZE-1 instances. BLIND-SEQUENTIAL commits to the key order the RULE-A-only
planner chose, then re-plans under A+B with that order fixed.

| | count | blind-sequential valid |
|---|---|---|
| rule-A key order **==** A+B key order | 201 | **201 (1.0000)** |
| rule-A key order **!=** A+B key order | 199 | **3 (0.0151)** |

**Blind-sequential validity by key count of the A+B optimum:**
**1 key 0.6532 (194/297) · 2 keys 0.1351 (10/74) · 3 keys 0.0000 (0/26) · 4 keys 0.0000 (0/3)**

⭐ So ≥ 2 keys IS the right lever, for a reason that is now explicit: with ONE key
`key_order=(c,)` admits exactly one pickup of exactly colour c — which is what the oracle
does too — so **BLIND-SEQUENTIAL *is* the oracle on a one-key route**. It is only because
CMAZE-1's single key can be the WRONG one (two components joined by several doors) that it
still fails on 34.7 % of one-key routes. CMAZE-2's cut is a single door, so even that
escape is closed and blind-sequential goes to ~1.

## 06:24Z The repair — `w14_request_layer/task2.py`, and one consequence that was measured, not chosen

**The rule added, and it is one line:** a pair is kept iff its RULES A+B optimal route
picks up **>= 2 keys** (`MIN_KEYS = 2`). The count is read off the solver's own route by
`w8_envscout.screen.key_order_of`, unmodified. It is an integer; **no tolerance enters.**

Two probes then decided the SHAPE of the fix:

- `artefacts/probe_density.txt` — 24 random cells per maze, all ordered pairs. The
  ">= 2 keys" column is **0.00 or 1.00 and nothing between** on all 12 mazes. The key
  count is a property of the DOOR CUT, hence of the MAZE, not of the pair.
- `artefacts/probe_direction.txt` — 6 agents in one door-free component, 6 targets in the
  other, 18 cross pairs, both orientations. Where the multi-key direction exists it is
  **18/18**, and the REVERSE direction is **0/18 FEASIBLE** — the key that opens the cut
  door lives on one side of it, so from the far side the target cannot be reached at all.
  **13 of 40** mazes with two usable components have such a direction.

⛔⛔ **THAT IS THE ROOT CAUSE, and it is sharper than w9's statement.** w9's layer demanded
`min_pairs = 12` cross pairs drawn SYMMETRICALLY in both directions. A maze whose cut needs
two keys cannot supply the reverse direction at all, so **the symmetry requirement itself
selected for single-key cuts.** The single-key routes were not an accident of the generator;
they were forced by the request layer's own acceptance test.

**So CMAZE-2b places all 6 agent words in the SOURCE component and all 6 target words in the
DESTINATION component**, and keeps the same 18 cross pairs, the same two rounds of
generalised adversarial tinting, the same walled target, the same budget grid, the same
`generate()` defaults.
⚠ **DECLARED:** the kept set is no longer symmetric in direction. Direction is now a
CONSTANT of the split rather than a balanced factor. It cannot confound a held-out
comparison because it does not vary, but it is a difference from CMAZE-2 and it is stated.
⚠ **DECLARED:** the `>= 2 keys` screen is applied to the UNSEALED maze, before the walled
target is chosen, so the walls that create the refusal condition cannot be what creates the
multi-key structure.

**Smoke, 6 sites, 16 generator seeds, 7 s:** key-count histogram **{2: 75, 3: 18}** —
93 routes, **0 at one key**. Pairs per site 15/15/15/18/15/15.

## 06:28Z Banks, and a process-discipline note

Three builds launched, 3 threads: `before 0`, `after 0`, `after 1`, each 96 sites, seeds
`1_000_000 + 100_000 * index` — w9's own seeds, `PER_SIDE=3`, `MIN_PAIRS=12`, walled target
cycling, exactly `scripts/w9_maze_bank.py`'s configuration.

⚠ **A launch bug of mine, recorded:** a first `before 0` launch whose PID capture failed had
in fact started a python; a second launch started a duplicate. **PID 33752 — my own, verified
by its full command line — was stopped**, leaving 15764 to finish. ⛔ No box-wide kill was
issued and no process belonging to another agent was touched; one other agent's python
(`hub-unseen-valid/w4`) was seen in the process list and left alone. PIDs in `logs/PIDS.txt`.

## 06:35Z Banks built, and a SECOND request-layer defect the repair exposes — declared before it is measured

Banks (96 sites each, w9's seeds and w9's `PER_SIDE=3 / MIN_PAIRS=12 /` walled-target cycle):

| bank | sites | generator seeds tried | acceptance | wall |
|---|---|---|---|---|
| `before` seed 0 | 96 | 480 | 0.2000 | 576 s |
| `after` seed 0 | 96 | 582 | 0.1649 | 342 s |
| `after` seed 1 | 96 | 578 | 0.1661 | 308 s |

⛔ **THE SECOND DEFECT.** A two-key route is LONGER than a one-key route, and w9's budget
words are ABSOLUTE integers. On the fixed bank the fraction of kept pairs WRITABLE at each
of w9's words collapses: **30 → 0.0268 · 45 → 0.2007 · 60 → 0.4425** (interp 38 → 0.0895,
53 → 0.3307). The split therefore carries **454 training requests against w9's 1208**, and
`unseen_pair` is **n=64 against w9's 195**. ⛔ A request layer that writes a third of the
items is not a fixed request layer.

⭐ **THE RULE FOR THE REBASED GRID, FIXED HERE BEFORE IT IS APPLIED and NOT MOVED:**
the three budget words are the **quantiles of the AFTER bank's optimum distribution at the
same percentiles w9's 30 / 45 / 60 occupy in the BEFORE bank's optimum distribution**,
rounded to the nearest multiple of 5; the two interpolation words are the **midpoints of
consecutive words**, as w9's 38 and 53 are the midpoints of 30–45 and 45–60. ⛔ The rule
refers to route lengths only. **No blind-planner number enters it.**

⚠ Both grids are measured and both are published. w9's UNCHANGED grid is the primary
column, because it is the one that changes nothing but the acceptance rule; the rebased
grid is published beside it so the Hub can see the cost of each and choose.

## 06:36Z The decisive number, on 8-site banks, at CMAZE-1's own slack-10 % budget

| | BEFORE (w9 layer) | AFTER (w14 layer) |
|---|---|---|
| ORACLE | **1.0000** | **1.0000** |
| BLIND-A | 0.1083 | 0.0000 |
| BLIND-B | 0.0000 | 0.0000 |
| BLIND-COLOUR | 0.0750 | 0.0000 |
| BLIND-FACTOR | 0.1500 | 0.0000 |
| **BLIND-SEQUENTIAL** | **0.8333** | **0.1707** |
| rule-B bend fraction | 0.8417 | 0.9512 |

On the full 96-site `after` bank, all pairs at slack 10 %: **BLIND-SEQUENTIAL 0.1810**.

## 06:50Z The rebased grid, computed by the declared rule

`artefacts/measure_after_0.json -> rebase`. The BEFORE bank has **1449 routes — w9's own
1449, to the route** — and its optima put w9's words 30/45/60 at percentiles
**33.89 / 78.33 / 92.41**. Reading the AFTER bank's optima at those percentiles gives
**54.0 / 82.66 / 103.0**, rounded to **55 / 85 / 105**, interp **70 / 95**.

⭐ **Self-check of the rule: applied with BEFORE as both source and target it returns
[30, 45, 60] — w9's own grid, exactly.** The interpolation words come back **[38, 52]**
against w9's **[38, 53]**, because Python rounds 52.5 to even. ⚠ Declared, not silently fixed.

| | opt mean | opt median | opt max | writable at the three words |
|---|---|---|---|---|
| BEFORE, w9 grid 30/45/60 | 37.38 | 36 | 111 | 0.3389 / 0.7833 / 0.9241 |
| AFTER, w9 grid 30/45/60 | 65.82 | 64 | 128 | 0.0268 / 0.2007 / 0.4425 |
| AFTER, rebased 55/85/105 | 65.82 | 64 | 128 | **0.3660 / 0.8059 / 0.9366** |

## 06:52Z RESULT — the fixed split, seed 0 (`artefacts/measure_after_0.json`)

**Key-count histogram, AFTER, 96 sites, 1530 routes: {2: 1228, 3: 284, 4: 18}.**
⭐⭐ **ZERO one-key routes. `fraction_multi_key = 1.0000` against w9's 0.1428.**

Validity under the EXACT judge. ⛔ `tolerance_used` is `null` in every block.

| condition | row | n | ORACLE | BLIND-A | BLIND-B | BLIND-COLOUR | BLIND-FACTOR | **BLIND-SEQ** | bend |
|---|---|---|---|---|---|---|---|---|---|
| SLACK10 (all pairs) | — | 1530 | **1.0000** | 0.0209 | 0.0000 | 0.0000 | 0.0373 | **0.1810** | 0.9536 |
| GRID_W9 test | seen | 454 | **1.0000** | 0.0881 | 0.0000 | 0.0000 | 0.1366 | **0.2996** | 0.8722 |
| GRID_W9 test | unseen_pair | 64 | **1.0000** | 0.0781 | 0.0000 | 0.0000 | 0.1406 | **0.2344** | 0.8594 |
| GRID_W9 test | nway | 119 | **1.0000** | 0.0420 | 0.0000 | 0.0000 | 0.1176 | **0.2689** | 0.9160 |
| GRID_W9 test | budget_interp | 119 | **1.0000** | 0.0588 | 0.0000 | 0.0000 | 0.1345 | **0.2437** | 0.8992 |
| GRID_REBASED test | seen | 1442 | **1.0000** | 0.0333 | 0.0000 | 0.0000 | 0.0541 | **0.2101** | 0.9397 |
| GRID_REBASED test | unseen_pair | 199 | **1.0000** | 0.0302 | 0.0000 | 0.0000 | 0.0603 | **0.1759** | 0.9296 |
| GRID_REBASED test | nway | 387 | **1.0000** | 0.0155 | 0.0000 | 0.0000 | 0.0465 | **0.2145** | 0.9587 |
| GRID_REBASED test | budget_interp | 424 | **1.0000** | 0.0189 | 0.0000 | 0.0000 | 0.0472 | **0.2075** | 0.9481 |

VAL legs are in the JSON and agree: unseen_pair SEQ 0.3188 (w9 grid) / 0.2451 (rebased).

**FRAGMENT-LOOKUP.** ⛔ Two DIFFERENT quantities, both printed:
**CROSS-MAZE (the published quantity): r=1 0.0000, r=2 0.0000** on both grids, both splits
— against CMAZE-1's published **0.0006 / 0.0000**.
**SAME-MAZE (a different quantity): r=1 0.0820 / r=2 0.5191** (w9 grid, test) and
**0.0051 / 0.4829** (rebased, test) — against w9's CMAZE-2 **0.0741 / 0.5105**.

⭐ **Rule-B bend fraction rises 0.484 (w9's CMAZE-2) -> 0.86–0.96**, above CMAZE-1's
published 0.816.

## 06:58Z ⭐⭐ THE BEFORE BANK REPRODUCES WAVE 9 EXACTLY — this is the control on the whole spoke

`artefacts/measure_before_0.json`, built from a clean worktree at `6699665` with
`w9_maze.task.build_site` untouched, 96 sites from seeds `1_000_000..`:

| quantity | wave 9 (`runs/scored1_seed0_test.json`, LOG.md) | this worktree |
|---|---|---|
| key-count histogram | **{1: 1242, 2: 198, 3: 9}, n = 1449** | **{1: 1242, 2: 198, 3: 9}, n = 1449** |
| n_train_requests | 1208 | **1208** |
| rows seen / unseen_pair / nway / budget_interp | 1208 / 195 / 389 / 434 | **1208 / 195 / 389 / 434** |
| unseen_pair BLIND-SEQUENTIAL | 0.907692 | **0.9077** |
| unseen_pair BLIND-A | 0.358974 | **0.3590** |
| unseen_pair BLIND-COLOUR | 0.230769 | **0.2308** |
| unseen_pair BLIND-FACTOR | 0.456410 | **0.4564** |
| unseen_pair bend fraction | 0.512821 | **0.5128** |
| nway BLIND-SEQUENTIAL | 0.922879 | **0.9229** |
| FRAG-r1 same-maze | 0.059932 | **0.0599** |
| VAL unseen_pair BLIND-SEQUENTIAL | 0.8750 (LOG.md) | **0.8750** |

⭐⭐ **Every one of these is an independent re-derivation: different worktree, different
process, nothing read from w9's artefacts.** The BEFORE/AFTER comparison below is therefore
a comparison against wave 9 itself, not against a re-implementation of it.

⚠ One number does NOT reproduce and it is reported: **cross-maze fragment lookup reads
0.0000 / 0.0000 here where w9 reported 0.0006 / 0.0000.** 0.0006 is 1 valid route in ~1700;
my pool bank is 48 sites against w9's, so the single lucky rollout did not recur. Both are
effectively zero and both are printed.

## 06:59Z The refusal material and the novelty control survive — `artefacts/refusal_material.json`

| | BEFORE | AFTER (w9 grid) | AFTER (rebased grid) |
|---|---|---|---|
| sites with a WALLED target | 93 / 96 | **56 / 96** | 56 / 96 |
| walled (IMPOSSIBLE) requests | 837 | **504** | 504 |
| budget-infeasible requests | 280 | 660 | 257 |
| `mt_unseen` control requests | 573 | 191 | **602** |
| mazes with `nway` / `unseen_pair` | 96 / 96 | 93 / 95 | 93 / 95 |

⚠ **DECLARED:** the walled condition is harder to satisfy in CMAZE-2b — all six target words
sit in one component, so sealing one without breaking the other pairs' >= 2-key routes
succeeds on 56 of 96 sites rather than 93. 504 impossible requests remain, which is enough
for the DISPLAY refusal row but it is a 40 % reduction and the Hub should know.
⭐ The `mt_unseen` control (a reachable but ungrounded target word — the control that stopped
w9's refusal being read as wall detection) is **thin at w9's grid (191) and healthy at the
rebased grid (602)**. That is a second, independent reason the rebased grid is the one to use.

## 07:08Z All three AFTER seeds measured

| bank | sites | seeds tried | acceptance | routes | key histogram | rebased grid |
|---|---|---|---|---|---|---|
| after 0 | 96 | 582 | 0.1649 | 1530 | {2: 1228, 3: 284, 4: 18} | [55, 85, 105] / [70, 95] |
| after 1 | 96 | 578 | 0.1661 | 1545 | {2: 1271, 3: 274} | [60, 85, 100] / [72, 92] |
| after 2 | 96 | 698 | 0.1375 | 1536 | {2: 1326, 3: 193, 4: 17} | [55, 75, 90] / [65, 82] |
| **pooled** | 288 | 1858 | — | **4611** | **{2: 3825, 3: 751, 4: 35}** | — |

⭐⭐ **ZERO one-key routes on all three banks.** BLIND-SEQUENTIAL, all pairs at slack 10 %:
**0.1810 / 0.1553 / 0.1068** against the BEFORE bank's **0.8295**. ORACLE **1.0000** everywhere.
⚠ The rebasing rule is per bank and gives three grids; the blind columns do not distinguish
them (0.1111–0.2451 across all of them). ⛔ **This spoke did NOT collapse them to one — freezing
a budget grid is a pre-registration decision and belongs to the Hub.**

## 07:12Z `REQUEST-LAYER-READY.json` — status `ready`

| acceptance condition | verdict |
|---|---|
| 1 a large majority of routes need >= 2 keys | **PASS** — `fraction_multi_key` 0.1428 -> **1.0000** |
| 2 BLIND-SEQUENTIAL falls back toward 0.4980 | **PASS** — worst held-out cell per condition 0.3188 / 0.2689 / 0.2451 / 0.2145 / 0.1810, all <= 0.4980 |
| 3 ORACLE stays 1.0000 | **PASS** — `distinct_values_seen: [1.0]` over 51 blocks |
| 4 the exact integer judge does not move | **PASS** — `tolerance_used: null`, judge sha256 recorded |

## 07:14Z ⭐ Independent verification — `artefacts/VERIFY.txt`, 28/28, exit 0

`w14_request_layer/verify.py` does **not** import the reporting code and does **not** read the
measure JSONs. It goes back to the bank pickles and to the engine:

- judge file sha256 on disk **unchanged** (`9eacdda17d87cac0…`), judge object is
  `w8_envscout.maze_colour.judge` itself;
- `tolerance_used` null at every occurrence;
- **6060 of 6060 stored routes re-judged VALID at `budget = opt`, zero slack, 0 length
  mismatches** — so ORACLE = 1.0000 is a property of the artefacts, not of the reporting;
- **min key count = 2** on all three AFTER banks;
- the doors are load-bearing for every kept pair, and agent and target never share a door-free
  component;
- both key-count histograms recomputed independently and equal to the published ones.

⚠ **The verifier caught a defect of its own first, and it is recorded rather than quietly
fixed.** Its first version recomputed the door-free components from `site.grid` and asserted
"all agents in one component, all targets in another"; it FAILED on 56 / 55 / 58 sites — exactly
the sites with a WALLED target, whose `grid` carries the seal and is therefore not the grid the
placement was made on. The check was replaced by the two properties that are actually
load-bearing (doors load-bearing per pair; agent and target in different components of the
site's OWN grid), which pass 0-violation. ⛔ The data were never the problem; the check was.

## 07:18Z Close

`FINAL-REPORT.md` written. Outputs mirrored into the worktree at `w14_request_layer_fix/`,
committed, bundled to `w14-request-layer-fix.bundle` (this branch only, never `--all`).
**Scored runs spent: 0 of 0.** No GPU, no cluster. ⛔ No box-wide kill was ever issued; the
only process stopped was my own duplicate `before 0` build (PID 33752), verified by its full
command line. `logs/PIDS.txt` carries every PID I launched.

## 07:22Z Bundle

`git bundle create /c/chlu/outputs/w14-request-layer-fix/w14-request-layer-fix.bundle
w14/request-layer-fix` — ⛔ **that branch only, never `--all`.** `git bundle verify` reports
*"is okay"* and *"records a complete history"*; `git bundle list-heads` prints **exactly one
ref**, `refs/heads/w14/request-layer-fix`. The bundle is regenerated after this commit, so the
ref it carries is this file's own commit; ⚠ the copy of this LOG inside the repo is therefore
one commit behind the bundle's tip by construction, and that is stated rather than papered over.
