# w14 hub/read-verb — LOG

⚠ **All times UTC** (the box's local clock is UTC+2).
Branch `w14/read-verb` · worktree `C:/chlu/worktrees/CHLU-w14-read-verb` · output `/c/chlu/outputs/w14-read-verb/`.

## 07:15:51Z — START
`START_UTC.txt` written = `2026-09-22T07:15:51Z`.

## 07:16Z — GATE CHECK (mechanical, BOTH required before ANY maze run)

| gate | file | verdict |
|---|---|---|
| request layer | `/c/chlu/outputs/w14-request-layer-fix/REQUEST-LAYER-READY.json` | ⭐ **COMPLETE** |
| refusal closeout | `/c/chlu/outputs/w14-refusal-displays-closeout/CLOSEOUT.md` | ⭐ **COMPLETE** |

**Gate 1 — COMPLETE, not merely present.** `status: "ready"`, `stamp_utc 2026-09-22T06:41:38Z`,
`tolerance_used: null`, `scored_runs_spent: 0`. All four ACCEPTANCE items `pass: true`:
(1) `fraction_multi_key_AFTER = 1.0` vs `BEFORE = 0.142857` (threshold 0.9);
(2) blind-sequential worst held-out row per budget condition
    `GRID_REBASED/test 0.21447`, `GRID_REBASED/val 0.24510`, `GRID_W9/test 0.26891`,
    `GRID_W9/val 0.31884`, `SLACK10_all_pairs 0.18105` — all BELOW the published CMAZE-1 0.4980;
(3) ORACLE `distinct_values_seen [1.0]` over 51 row×condition;
(4) exact integer judge unmoved, `w8_envscout.maze_colour.judge` sha256
    `9eacdda17d87cac0ad5ced6ccbb6fa43ebc1508c0460d0cc183e545d066beaac`.
Spoke closed: its own `END_UTC.txt`, `FINAL-REPORT.md` (19,940 B) and
`w14-request-layer-fix.bundle` (7,357,130 B) are all present.
New request layer = **CMAZE-2b** = `w14_request_layer.task2.build_site_v2`; banks at
`/c/chlu/outputs/w14-request-layer-fix/banks/bank_after_seed{0,1,2}.pkl`
(96 sites; 1530 / 1545 / 1536 routes).

**Gate 2 — COMPLETE.** `CLOSEOUT.md` (19,532 B) rules the maze refusal DISPLAY rows RETIRED
(Head 2026-09-21, `outputs/advisor-wave-plan-w12-w14.md:56`), names the revival bar, and closes
with its own `END_UTC.txt`, `FINDINGS.md` and `w14-refusal-displays-closeout.bundle`.
⇒ **no refusal display row appears anywhere in this hub's tables.**
⚠ Carried forward from CLOSEOUT §2: the CLU display means `0.9601 / 0.9801 / 0.9781 / 0.8182`
are **3-seed means**, and `0.8182` is a mean, not seed 2 (seed 2 = `0.800633`).

⇒ **UNGATED.** Scored runs are authorised subject to the SCORED-LEDGER row being written first.

## 07:17Z — the object, and where every piece comes from

`chlu.__file__ = C:\chlu\worktrees\CHLU-w14-read-verb\chlu\__init__.py`; `w9_maze`,
`w8_envscout` and `w14_request_layer` all assert inside the same root. `jax 0.9.0`,
`[CpuDevice(id=0)]` — **local CPU only, zero GPU.**
`w14/request-layer-fix` was merged into `w14/read-verb` (its branch is in the shared
repo; ⛔ its worktree, output dir and processes were never touched).

| held fixed | imported from |
|---|---|
| the store + charter-C controls | `w9_maze.arms.CluRead` (unmodified) |
| the request-only read | `chlu.core.gen_request_head.head_read` (unmodified) |
| the frozen encoder + shared decoder | `w9_maze.encoder` (unmodified) |
| the parameterless executor | `w9_maze.render.execute` (unmodified) |
| the exact integer judge | `w8_envscout.maze_colour.judge` (imported, **not wrapped**) |
| the request layer | `w14_request_layer.task2.build_site_v2` = **CMAZE-2b** (the gate) |
| the split, the item, the learned baselines | `w9_maze.{splits,render,baselines}` |

**THE DIAL — and it is one line of arithmetic.** The head's own design encodes `NaN` as
*this factor is not requested* (`one_hot_interpolated`: "the head's version of `-1`").
So for a request `(m, a, t, b)` the two half-addresses are

    half A (the AGENT half)  = (m, a, NaN, b)
    half B (the TARGET half) = (m, NaN, t, b)

⭐ **Mechanically verified before any arm was built** (design width 1624, `pair_block=True`):
the FULL design has **11** non-zero columns — bias + 4 mains + 6 pair blocks; **half A has 7**
(it drops the target main and the three pair blocks `(maze,target)`, `(agent,target)`,
`(target,budget)`) and **half B has 7** (the mirror). ⭐ Their union is the FULL design minus
**exactly one** block — `(agent, target)` — which is the one block a held-out pair cannot
ground in any case. The halves are the address book's own factorisation, not a new mechanism.

| | |
|---|---|
| **THE ARM** `MERGE` | `overlay( decode(Z_A), decode(Z_B) )` — each half decoded SEPARATELY through the shared decoder, then a fixed plan-space overlay, target's plan winning conflicts |
| **THE ABLATION** `AVG` | `decode( (Z_A + Z_B)/2 )` — w9's read, on the IDENTICAL two latents |
| **reference** `FULL` | `decode( Z_(m,a,t,b) )` — w9's arm, unchanged |
| **merge controls** | `agent_wins`, `agent_only`, `target_only` on the identical two plans |

⛔ **Every arm gets both verbs and the identical decoder** — the CLU, `clu_lowrank`,
`clu_nopair`, all three charter-C controls, both lookup floors and the MLP / LSTM /
transformer at matched parameters are read at the same two half-addresses and composed
by the same operator. ⛔ Nothing in the verb is fitted: no threshold, no per-instance
tuning, `tolerance_used` stays **null**.

## 07:19Z — ⭐⭐ THE GO/NO-GO, run BEFORE any arm and with ZERO scored budget

w9 shipped its 1.0000 overlay with a caveat that is this wave's precondition
(`outputs/_thinkpad/w9/w9-maze/LOG.md:339-341`):
> *"that the overlay is perfect is itself a statement that CMAZE-2's routes decompose
> additively … **In an environment whose routes need two or more keys the overlay would
> not be perfect**"*

CMAZE-2b is that environment (**100 %** of routes take ≥ 2 keys, gate file
`KEY_COUNT_HISTOGRAM.AFTER__w14_request_layer`). So the operator's reach was re-measured
on the repaired layer, on VAL, on all three banks, with whole written TRAINING plans
(⛔ a REFERENCE, not an arm — it keeps whole items and is matched to no byte count).
`w14_read_verb/overlay_ref.py` → `artefacts/overlay_ref_*.json`.

**GRID_W9 (30/45/60), VAL, three seeds:**

| row | seed | n | **target_wins** | agent_wins | agent_only | target_only | random agent half | plan coverage |
|---|---|---|---|---|---|---|---|---|
| unseen_pair | 0 | 69 | **0.9710** | 0.5797 | 0.4058 | 0.1014 | 0.3043 | 1.0000 |
| unseen_pair | 1 | 57 | **1.0000** | 0.5965 | 0.3509 | 0.2807 | 0.4561 | 1.0000 |
| unseen_pair | 2 | 61 | **1.0000** | 0.6557 | 0.2951 | 0.2295 | 0.3607 | 0.9975 |
| nway | 0 | 128 | **0.9844** | 0.6797 | 0.3672 | 0.1797 | 0.3203 | 1.0000 |
| nway | 1 | 103 | **1.0000** | 0.5825 | 0.3495 | 0.1748 | 0.3204 | 1.0000 |
| nway | 2 | 138 | **0.9855** | 0.6232 | 0.4275 | 0.1957 | 0.3841 | 0.9997 |

**GRID_REBASED (55/85/105), VAL, seed 0:** unseen_pair n=204 → **0.9853** / 0.7206 /
0.3922 / 0.1225 / random 0.2647; nway n=392 → **0.9898** / 0.7321 / 0.3673 / 0.1531 /
random 0.3138. Coverage 1.0000 / union 1.0000 on both rows.

⭐ **In every cell `validity == score` to four decimals** ⇒ where the overlay is valid it
is **OPTIMAL** (`score = opt/len = 1`), exactly as on CMAZE-2.

⇒ ⭐⭐ **GO, with the caveat fired and quantified.** The fixed operator still expresses the
task on a 100 %-multi-key layer, but ⛔ it is **no longer perfect**: 0.9710–1.0000 against
w9's flat 1.0000. ⭐ w9's prediction was right in direction, and the size of the effect is
now on the record. ⛔ **The reference is therefore a CEILING of 0.971–1.000, not 1.0000**,
and every arm number below is read against it.
⭐ The rule is load-bearing on the new layer too: `agent_wins` 0.58–0.73, one half alone
0.10–0.43, the random-agent-half control 0.26–0.46 — all far below `target_wins`.
⛔ **No per-instance tuning was needed, so the task file's stopping rule does not fire.**

## 07:21Z — the encoder pool, and the PRE-REGISTERED budget grid

**Pool:** 270 sites (3 shards × 90), `w14_read_verb/pool.py`, seed range **7,000,000+**
— disjoint from w9's pool (5,000,000+) and from every CMAZE-2b evaluation bank
(1,000,000 / 1,100,000 / 1,200,000 + ≤ a few thousand). Built by `build_site_v2` with
`adversarial=0`, no walling. Acceptance 0.1387 / 0.1538 / 0.1505; 228 / 221 / 225 s.
⭐ 270 sites is w9's own pool size. ⚠ **DECLARED difference from w9's pool:** it is built by
`build_site_v2`, not `build_site`, so the shared encoder's reconstruction material is
≥ 2-key routes like the evaluation bank's. A single-key pool would train the one shared
encoder on a different route distribution from the one every arm is graded on — a
confound carried by the encoder, which is shared, so it would hit every arm.

### ⛔ PRE-REGISTERED, BEFORE ANY ARM NUMBER EXISTS — the scored budget condition
**`GRID_REBASED`.** The rule, stated so it can be checked: *choose the budget condition
whose held-out row sizes are closest to wave 9's, so this wave's table can be read
against the wave-9 table it must be compared with.* It mentions **row sizes only** — no
score, no arm, no verb.

| condition | unseen_pair n | nway n | train requests | w9's own |
|---|---|---|---|---|
| GRID_W9 (30/45/60) | 69 / 57 / 61 | 128 / 103 / 138 | 454 | 195 / 389 / 1208 |
| ⭐ **GRID_REBASED** | **204** (s0) | **392** (s0) | **~1442** | **195 / 389 / 1208** |

⚠ **DECLARED, and it is a departure from the gate file.** `REQUEST-LAYER-READY.json`
names `GRID_W9` its *primary* — correctly, for the ENVIRONMENT columns, where the point
is that nothing but the acceptance rule differs from w9. For an ARM comparison the
binding quantities are held-out row size and training volume, and the gate file published
`GRID_REBASED` for exactly that reason (`DECLARED_DIFFERENCES_FROM_CMAZE2`: *"Routes are
longer … which is why GRID_REBASED exists. At w9's unchanged words the split carries 454
training requests against w9's 1208; at the rebased words it carries about 1442."*).
⚠ **The cost, stated:** `GRID_REBASED` words are re-derived **per seed** (s0 55/85/105,
s1 60/85/100, s2 55/75/90), so the budget words are a per-seed constant like the bank
itself, not one grid across seeds. `GRID_W9` is reported beside it on VAL.
⭐ The words are **quoted from the gate file's `BUDGET_GRID` block**, never recomputed here.

**Store configuration:** w9's own scored configuration, carried unchanged —
`n_atoms=31, K=8, ridge=0.01, ae_steps=12000, bl_steps=4000, LATENT_D=256`
(`outputs/_thinkpad/w9/w9-maze/runs/scored1_seed0_test.json`). `clu_lowrank` at
`n_atoms=7`. ⛔ Nothing about the store is a dial this wave.
