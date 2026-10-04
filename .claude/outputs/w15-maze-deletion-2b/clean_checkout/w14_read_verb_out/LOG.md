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

## 08:02Z — THE VAL LEGS (⛔ selection only; no scored budget spent)

Three legs, three concurrent processes on a **3-core box**, PIDs in `logs/PIDS.txt`.
⛔ No box-wide kill; only this hub's own PIDs were ever handled.
`dev1 seed 0 rebased` 2420 s · `dev1 seed 1 rebased` 2391 s · `dev1w9 seed 0 GRID_W9` 2353 s.

**The shared encoder trained cleanly on all three legs** — final reconstruction loss
**3e-5 / 2e-5 / 1e-5**, well under the 0.005 wave 9's healthy seeds reached and far from
the 0.080 its stalled seed 1 reached. **Pipeline ceiling `train` = 1.0000 on all three.**

| leg | grid | train req | unseen_pair n | nway n | ceiling unseen_pair | ceiling nway |
|---|---|---|---|---|---|---|
| s0 | REBASED 55/85/105 | 1442 | 204 | 392 | 0.6912 | 0.6046 |
| s1 | REBASED 60/85/100 | 1430 | 200 | 385 | 0.6850 | 0.5429 |
| s0 | GRID_W9 30/45/60 | 454 | 69 | 128 | 0.5797 | 0.6094 |

⭐ **BLIND-SEQUENTIAL, printed on THIS HUB'S OWN SPLIT** (⛔ never the published 0.4980,
never another split's number), `w9_maze.envcols.blind_columns`, CMAZE-2b:

| leg | row | BLIND-A | BLIND-B | BLIND-COLOUR | BLIND-FACTOR | **BLIND-SEQ** | ORACLE | rule-B bend fraction |
|---|---|---|---|---|---|---|---|---|
| s0 REBASED | unseen_pair | 0.0294 | 0.0000 | 0.0000 | 0.0588 | **0.2451** | 1.0000 | 0.9314 |
| s0 REBASED | nway | 0.0306 | 0.0000 | 0.0000 | 0.0612 | **0.2143** | 1.0000 | 0.9362 |
| s1 REBASED | unseen_pair | 0.0450 | 0.0000 | 0.0150 | 0.0450 | **0.1900** | 1.0000 | 0.9550 |
| s1 REBASED | nway | 0.0545 | 0.0000 | 0.0156 | 0.0545 | **0.1558** | 1.0000 | 0.9455 |
| s0 GRID_W9 | unseen_pair | 0.0725 | 0.0000 | 0.0000 | 0.1449 | **0.3188** | 1.0000 | 0.8696 |
| s0 GRID_W9 | nway | 0.0625 | 0.0000 | 0.0000 | 0.1406 | **0.2578** | 1.0000 | 0.2578→bend 0.8828 |

⭐⭐ **The request-layer repair holds on this hub's own split.** Wave 9's CMAZE-2 read
**0.9077** (unseen_pair) and **0.9229** (nway); CMAZE-2b reads **0.1900–0.3188**.
⭐ **An exact cross-check, not a coincidence:** the GRID_W9/val figure **0.3188** is
byte-for-byte the gate file's own `GRID_W9/val` entry (0.3188405797101449) — this hub's
independently-run column reproduces the gate spoke's to every printed digit.
⭐ **Rule B now bends 87–96 % of routes**, so the row is not being won on the easy half.
⛔ ORACLE stays **1.0000** everywhere: the environment still admits a perfect answer.

### ⭐⭐ THE VAL RESULT — and it is the same on both grids and both seeds

**Row `unseen_pair`, CLU, validity:**

| leg | n | `FULL` (w9's read) | `AVG` (**the ablation**) | **`MERGE` (the arm)** | **MERGE − AVG** | ceiling | plan-space reference |
|---|---|---|---|---|---|---|---|
| s0 REBASED | 204 | 0.2402 | 0.1127 | **0.5931** | **+0.4804** | 0.6912 | 0.9853 |
| s1 REBASED | 200 | 0.1950 | 0.0550 | **0.5000** | **+0.4450** | 0.6850 | — |
| s0 GRID_W9 | 69 | 0.1594 | 0.0435 | **0.4058** | **+0.3623** | 0.5797 | 0.9710 |

⭐ **Wave 9's number reproduces on the repaired layer:** `FULL` on GRID_W9 seed 0 is
**0.1594** against wave 9's **0.1590** — the CMAZE-2 arm's own band, unchanged by the
≥ 2-key repair.
⭐⭐ **Moving the composition from the latent to the plan multiplies the CLU's held-out
validity by 3.7×–9.1×** (0.1127→0.5931, 0.0550→0.5000, 0.0435→0.4058) on **identical
latents**, and takes it from below w9's own `FULL` read to **86 % of the pipeline ceiling**.

**And the merge rule is load-bearing, not "more slots":** with the identical two decoded
plans, `agent_wins` reads 0.4412 / 0.3550 / 0.2754 and one half alone reads 0.0392–0.2941.
The gain is the specific overlay, not the fact that two decodes happened.

### ⛔ BAR 3 FAILS, AS PRE-REGISTERED — the store is DECORATIVE here

`MERGE`, row `unseen_pair`, seed 0 REBASED: clu **0.5931**, `landscape_shuffled`
**0.4412**, `random_atoms` **0.4020**; on `seen`, clu 0.7212 against 0.5111 / 0.5395.
⛔ **The controls do not collapse** (the bar was ≤ 0.5 × clu). **rank/width = 0.9375 on
every store row**, and `clu_lowrank` at **rank/width 0.1875** scores **0.0000 on every
held-out row** — wave 8's saturation trap, exactly as wave 9 recorded it.
⇒ ⛔ **NO ROW IS CLAIMED FOR THE STORE.** ⭐ The verb contrast survives, because `MERGE`
and `AVG` consume **identical latents** and the gap is present on *every* arm — which is
itself the point of §BAR 2.

### ⛔⛔ BAR 5 FAILS, AND IT IS THE MOST INFORMATIVE FAILURE IN THE WAVE

`MERGE`, row `unseen_pair`: **`lookup_matched` reaches 0.9755 (s0) / 0.9800 (s1) /
0.8986 (GRID_W9 s0)** — against the CLU's 0.5931 / 0.5000 / 0.4058, and **essentially at
the plan-space reference's own 0.9853 / 0.9710.**
⚠ **Declared:** at this configuration the matched-bytes budget is **not binding** — the
store's 410,688 read floats buy 1,604 latents and there are only 1,442 training items, so
`lookup_matched` == `lookup_full`. Wave 9 had the same (1208 == 1208). The floor is
therefore *the whole training set*, and it must be read that way.
⭐ **Under `FULL` the lookup LOSES** (0.1127 vs the CLU's 0.2402). ⛔ **Under `MERGE` it
wins by 0.38.** The verb changes which arm is better — which is the sharpest possible
statement that the operator, not the store, is carrying the result.

### ⭐⭐ THE MERGE HELPS EVERY ARM — `MERGE − AVG`, row `unseen_pair`, seed 0 REBASED

| clu | landscape_shuffled | random_atoms | lookup_matched | mlp | lstm | transformer | clu_lowrank | clu_nopair | landscape_removed |
|---|---|---|---|---|---|---|---|---|---|
| **+0.4804** | +0.3725 | +0.3186 | **+0.8333** | +0.1471 | +0.0784 | +0.2402 | 0.0000 | +0.0049 | 0.0000 |

⛔ **The CLU's delta is NOT the largest** — the matched-bytes lookup's is, by 0.35.
⇒ ⭐⭐ **The composition operator is a property of the TASK INTERFACE, not of the store.**
(The only arms it does not help are the ones that read nothing: `landscape_removed`, whose
atoms are zeroed, and `clu_lowrank`/`clu_nopair`, which cannot render a route at all.)

⇒ **Proceeding to the three scored TEST legs**, ledger rows 1/3, 2/3, 3/3 appended to
`/c/chlu/SCORED-LEDGER.md` at 08:05Z **before** the runs, naming
`PREREG-w14-read-verb-1.md`; append-only, mechanically checked additive against a
pre-append copy.

## 08:05Z — THE LEDGER, THEN THE THREE SCORED RUNS

⭐ **Three rows appended to `/c/chlu/SCORED-LEDGER.md` BEFORE the runs**, target
`read-verb`, this hub's OWN rows (⛔ no other hub's row authorises anything), each naming
`outputs/w14-read-verb/PREREG-w14-read-verb-1.md`. Append-only; a pre-append copy was kept
and `diff <(head -n -3 SCORED-LEDGER.md) backup` returned empty ⇒ **nothing before the new
rows changed.** ⭐ `leg.py` **refuses** `--which test` unless both gates are ready AND a
ledger row already names that leg's tag — the standing rule is enforced by code.

`scored1 seed 0` **260 s** · `scored2 seed 1` **259 s** (both reused the encoder their own
VAL leg cached — same `split.train_pairs`, so it is literally the same object) ·
`scored3 seed 2` **1228 s** (trained its own encoder to loss **1e-5**).

## 08:23Z — THE SCORED RESULT

Generated by `w14_read_verb/tables.py` → `TABLES.md`; ⛔ no cell hand-copied.

**Row `unseen_pair`, CLU, TEST:**

| verb | seed 0 (n=199) | seed 1 (n=198) | seed 2 (n=208) |
|---|---|---|---|
| `FULL` | 0.2261 | 0.2929 | 0.3173 |
| `AVG` (**ablation**) | 0.0704 | 0.1313 | 0.1298 |
| ⭐ **`MERGE`** | **0.5126** | **0.6667** | **0.5769** |
| **MERGE − AVG** | **+0.4422** | **+0.5354** | **+0.4471** |

`nway` (conjunction): `AVG` **0.0000 / 0.0160 / 0.0000** → `MERGE` **0.4891 / 0.4652 /
0.3915**. `budget_interp`: 0.0802/0.0875/0.0924 → 0.5142/0.5650/0.4988.

⭐⭐ **BAR 1 PASSES ON ALL THREE SEEDS** at 4–5× its declared margin.

**BLIND-SEQUENTIAL on this hub's OWN split, TEST:** `unseen_pair` 0.1759 / 0.1566 / 0.1202,
`nway` 0.2145 / 0.1602 / 0.1197; ORACLE 1.0000 everywhere; rule-B bend fraction
0.9296 / 0.9545 / 0.9712. ⭐ Wave 9's CMAZE-2 read 0.9077 / 0.9229 — **the repair holds.**

### ⛔⛔ AND THE OPERATOR BELONGS TO THE INTERFACE, NOT THE STORE

`MERGE − AVG`, row `unseen_pair`, **3-seed mean, every arm**:

| lookup_matched | **CLU** | random_atoms | landscape_shuffled | transformer | mlp | lstm | clu_nopair | clu_lowrank | landscape_removed |
|---|---|---|---|---|---|---|---|---|---|
| **+0.8299** | **+0.4749** | +0.3557 | +0.3541 | +0.2757 | +0.1829 | +0.1380 | +0.0032 | 0.0000 | 0.0000 |

⛔ **The CLU's delta is not the largest.** BAR 2 passes against its one named comparator
(the transformer: +0.2111 / +0.2424 / +0.1442) and **fails the claim it was written to
protect**. ⇒ ⭐⭐ **the composition operator is a property of the TASK INTERFACE, not of the
store** — in those words.

### ⛔ BAR 3 FAILS — the store is decorative; NO ROW IS CLAIMED FOR IT
`landscape_shuffled` 0.5111/0.5552/0.5727 and `random_atoms` 0.5395/0.5650/0.5858 on
`seen`, against 0.5 × CLU = 0.3606/0.3455/0.3508. **rank/width 0.9375 on every store row**;
`clu_lowrank` at 0.1875 scores 0.0000 on every held-out row. Wave 8's saturation trap,
exactly as w9 recorded it. ⚠ ⛔ **Not a law about rank/width** — w12 saw the same controls
collapse to 0.000 at 0.9844 on Shapes3D.

### ⛔⛔ BAR 5 FAILS — and it is the wave's most informative failure
`lookup_matched` under `MERGE`: **0.9648 / 0.9646 / 0.9808** (`unseen_pair`), **0.9674 /
0.9626 / 0.9683** (`nway`) — at the plan-space reference's own 0.9853. Under `FULL` the
same floor **loses** to the CLU (0.1407 vs 0.2261). ⭐ **The verb reverses the ranking.**
⚠ Declared: the byte match is **not binding** — 410,688 read floats buy 1,604 latents and
the split has 1,442; `lookup_matched` == `lookup_full`. Wave 9 had the same.

### ⭐⭐ THE MERGE BREAKS THE SINGLE-DECODE CEILING
`lookup_matched` under `MERGE` exceeds the `encode(true item)→decode→execute` ceiling by
**+0.3216 / +0.2778 / +0.2740** (`unseen_pair`) and **+0.4118 / +0.3915 / +0.4022**
(`nway`). ⚠ Not a contradiction: that ceiling bounds a **one-decode** read, and `MERGE`
decodes twice and unions. ⭐ **The union of two decodes repairs what one decode loses.**
⛔ No leak: both halves read TRAINING latents only; the judge runs on the TRUE maze.

### ⭐⭐ THE MECHANISM, MEASURED
`cos(Z_A, Z_B)` = **+0.9668 / +0.9589 / +0.9630** (CLU) and `cos(Z_AVG, Z_FULL)` =
**+0.9971 / +0.9964 / +0.9966`. The halves are **0.96-parallel**; their average is a
shrunk copy of the full read. The same halves **decoded** are complementary — 0.1960 and
0.0704 alone, **0.5126** overlaid. ⭐ Wave 9's sentence, now a number.

## 08:26Z — INDEPENDENT VERIFICATION

`w14_read_verb/verify.py` → `artefacts/VERIFY.txt`. It imports **none** of
`pipeline/report/tables/leg` and re-derives every verdict from the result JSONs, the gate
files and the ledger. **PASS** on: gate ready + 4/4 ACCEPTANCE + gate closed; judge sha256
equal to the gate's; `tolerance_used` null in **all six** legs; **exactly three** scored
TEST legs at **three distinct seeds**; a ledger row naming **each** tag; **11 arms carry
every verb on every row, missing `[]`**, per seed; half-addresses 7 + 7 of 11; **no
CMAZE-1 token in any result file**.
⚠ **Six FAILs, and they are the pre-registered ones:** the BAR-3 control collapse, on both
controls, on all three seeds — **computed by the verifier, not quoted from the report.**

## 08:28Z — CLOSE
Artefacts: `FINAL-REPORT.md`, `TABLES.md`, `PREREG-w14-read-verb-1.md`, `LOG.md`,
`runs/` (3 scored + 3 val JSONs + 4 cached encoders), `artefacts/` (4 overlay-reference
JSONs + `VERIFY.txt`), `pool/` (3 shards, 270 sites), `logs/` (incl. `PIDS.txt`).

## 08:25Z — THE BUNDLE, AND A CLEAN-CLONE REPLICATION

```
git bundle create /c/chlu/outputs/w14-read-verb/w14-read-verb.bundle w14/read-verb
```
⛔ **That branch only — never `--all`.**

```
$ git bundle verify  w14-read-verb.bundle
C:/chlu/outputs/w14-read-verb/w14-read-verb.bundle is okay
The bundle contains this ref:
8437142526cd3f92284974581a2382282420be45 refs/heads/w14/read-verb
The bundle records a complete history.
$ git bundle list-heads w14-read-verb.bundle | wc -l
1
```
⭐ **Exactly ONE ref**, complete history, 7,486,263 B.

⭐⭐ **Replicated from a clean clone of the bundle itself** (not from the worktree):
`git clone -b w14/read-verb <bundle> replica` → `python -m w14_read_verb.tables` on the
replica's own artefacts regenerates **`TABLES.md` BYTE-IDENTICAL** to the shipped one
(`diff` empty), and `python -m w14_read_verb.verify` from the replica returns the
**identical** verdicts — every PASS, and the same **six** pre-registered BAR-3 control
failures. ⛔ **The negative is reproduced, not just the positive.**

## 08:25Z — END
`END_UTC.txt` = **2026-09-22T08:24:58Z**. Window **07:15:51Z → 08:24:58Z ≈ 1 h 09 m**.
⭐ **Scored spend 3/3 on `read-verb`.** ⛔ ZERO GPU; local CPU only, 3 cores, at most 3
concurrent processes throughout. ⛔ No other agent's session, worktree, output dir or
process touched; ⛔ no box-wide kill — this hub's own PIDs are in `logs/PIDS.txt`.
⛔ `/c/chlu/evidence/` was read-only throughout. `/c/chlu/SCORED-LEDGER.md` was appended
to, additively, and nothing above the three new rows changed.
