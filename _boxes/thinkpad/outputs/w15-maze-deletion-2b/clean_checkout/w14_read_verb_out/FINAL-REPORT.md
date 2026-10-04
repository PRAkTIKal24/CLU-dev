# w14 `hub/read-verb` — FINAL REPORT

**Agent:** autonomous hub, FORGIS_14, local CPU only (**3 cores**), **ZERO GPU**.
**Branch** `w14/read-verb` · **worktree** `C:/chlu/worktrees/CHLU-w14-read-verb` ·
**output** `/c/chlu/outputs/w14-read-verb/` · base `base/iter8 = 6699665` with
`w14/request-layer-fix` merged in.
**Window:** `START_UTC.txt` **2026-09-22T07:15:51Z** → `END_UTC.txt`.
⚠ **All times UTC** (the box's local clock is UTC+2).
**Scored spend: 3/3 on target `read-verb`** — ledger rows 1/3, 2/3, 3/3 in
`/c/chlu/SCORED-LEDGER.md`, each written **before** its run, each naming
`outputs/w14-read-verb/PREREG-w14-read-verb-1.md`.
⭐ `chlu.__file__ = C:\chlu\worktrees\CHLU-w14-read-verb\chlu\__init__.py`, asserted with
`w9_maze`, `w8_envscout` and `w14_request_layer` before any number was produced.
⛔ No other agent's session, worktree, output dir or process was touched; ⛔ no box-wide
kill; this hub's own PIDs are in `logs/PIDS.txt`.

---

## 0. THE WAVE ANSWER, IN ONE PARAGRAPH

> ⭐⭐ **THE COMPOSITION OPERATOR IS PART OF THE INTERFACE AND IT IS TASK-SHAPED —
> CONFIRMED on a held-out estimand, three seeds, all or none.** Moving the composition of
> two half-addresses from LATENT space to PLAN space, on **identical latents** and through
> the **identical frozen decoder**, lifts the CLU's held-out `unseen_pair` validity from
> **0.0704 / 0.1313 / 0.1298** to **0.5126 / 0.6667 / 0.5769** — **+0.4422 / +0.5354 /
> +0.4471**, a 4.4× / 5.1× / 4.4× multiplication, under the unchanged exact integer judge.
> ⛔⛔ **AND IT BELONGS TO THE INTERFACE, NOT TO THE STORE.** The same fixed merge lifts
> *every* arm, and it lifts the **matched-bytes lookup floor more than twice as much**
> (mean **+0.8299** against the CLU's **+0.4749**), carrying that floor to **0.9648 /
> 0.9646 / 0.9808** — past the CLU by ~0.4 and past the single-decode pipeline ceiling
> itself. ⛔ The charter-C controls do **not** collapse, so **NO ROW IS CLAIMED FOR THE
> STORE**. ⭐ The verb claim survives that, because `MERGE` and `AVG` consume the same two
> latents; **a verb claim is not a store claim, and only the verb claim is made.**

---

## 1. THE GATE — mechanical, BOTH required, BOTH COMPLETE

| gate | file | verdict |
|---|---|---|
| request layer | `/c/chlu/outputs/w14-request-layer-fix/REQUEST-LAYER-READY.json` | ⭐ **COMPLETE** |
| refusal closeout | `/c/chlu/outputs/w14-refusal-displays-closeout/CLOSEOUT.md` | ⭐ **COMPLETE** |

`status: "ready"`, `scored_runs_spent: 0`, `tolerance_used: null`, all four ACCEPTANCE
items `pass: true`; the spoke is closed (`END_UTC.txt`, `FINAL-REPORT.md`,
`w14-request-layer-fix.bundle`). The closeout likewise closed.
⭐ **The judge is provably unmoved:** `w8_envscout/maze_colour.py` hashes to
`9eacdda17d87cac0ad5ced6ccbb6fa43ebc1508c0460d0cc183e545d066beaac`, **byte-for-byte the
sha256 the gate recorded** (`artefacts/VERIFY.txt` check 2a).
⇒ **No refusal DISPLAY row appears anywhere in this report.** RESULT rows never refuse.
⛔ **The harness tier is CMAZE-1; this hub's maze is CMAZE-2b. They share no table here**,
and `artefacts/VERIFY.txt` check 8 confirms no CMAZE-1 token occurs in any result file.

---

## 2. THE DIAL — the READ VERB, and nothing else

**Held fixed and imported, not re-implemented:** the store and its charter-C controls
(`w9_maze.arms.CluRead`), the request-only read (`chlu.core.gen_request_head.head_read` —
**no feature argument, by signature**), the frozen encoder and shared decoder
(`w9_maze.encoder`), the parameterless executor (`w9_maze.render.execute`), the exact
integer judge, the split (`w9_maze.splits`), the item (`w9_maze.render.item_tensor`) and
the learned baselines at matched parameters (`w9_maze.baselines`).

**The two half-addresses are the head's OWN `NaN` = "not requested" encoding**
(`one_hot_interpolated`: *"the head's version of `-1`"*):

    half A (the AGENT half)  = (m, a, NaN, b)
    half B (the TARGET half) = (m, NaN, t, b)

⭐ **Mechanically verified, and re-verified by the independent verifier** (check 7): the
FULL design has **11** non-zero columns (bias + 4 mains + 6 pair blocks); **half A has 7**
and **half B has 7**. Their union is FULL minus **exactly one** block — `(agent, target)`
— which is precisely the block a held-out pair cannot ground. The halves are the address
book's own factorisation, not a new mechanism.

| | |
|---|---|
| **THE ARM** `MERGE` | `overlay( decode(Z_A), decode(Z_B) )`, target's plan wins conflicts |
| **THE ABLATION** `AVG` | `decode( (Z_A + Z_B)/2 )` — w9's read, **identical two latents** |
| reference `FULL` | `decode( Z_(m,a,t,b) )` — w9's arm, unchanged |
| merge controls | `agent_wins`, `agent_only`, `target_only` on the identical two plans |

⛔ **Every arm gets both verbs and the identical decoder.** `artefacts/VERIFY.txt` check 4
confirms, per seed: **11 arms carry every verb on every row, missing `[]`.**
⛔ Nothing in the verb is fitted — no threshold, no per-instance tuning; the head is
**never refit** for the halves. `tolerance_used` is `null` in **all six** legs (check 2b).

---

## 3. ⭐⭐ THE GO/NO-GO — wave 9's own caveat, fired and quantified, at ZERO scored cost

Wave 9 shipped its 1.0000 plan-space overlay with a caveat
(`outputs/_thinkpad/w9/w9-maze/LOG.md:339-341`):
> *"that the overlay is perfect is itself a statement that CMAZE-2's routes decompose
> additively … **In an environment whose routes need two or more keys the overlay would not
> be perfect**"*

CMAZE-2b is that environment — **100 %** of its routes take ≥ 2 keys. Re-measured on VAL,
all three banks, with whole written TRAINING plans (⛔ a REFERENCE, not an arm: it keeps
whole items and is matched to no byte count). `w14_read_verb/overlay_ref.py` →
`artefacts/overlay_ref_*.json`:

| row | seed | n | **target_wins** | agent_wins | agent_only | target_only | random agent half | plan coverage |
|---|---|---|---|---|---|---|---|---|
| unseen_pair | 0 | 69 | **0.9710** | 0.5797 | 0.4058 | 0.1014 | 0.3043 | 1.0000 |
| unseen_pair | 1 | 57 | **1.0000** | 0.5965 | 0.3509 | 0.2807 | 0.4561 | 1.0000 |
| unseen_pair | 2 | 61 | **1.0000** | 0.6557 | 0.2951 | 0.2295 | 0.3607 | 0.9975 |
| nway | 0 | 128 | **0.9844** | 0.6797 | 0.3672 | 0.1797 | 0.3203 | 1.0000 |
| nway | 1 | 103 | **1.0000** | 0.5825 | 0.3495 | 0.1748 | 0.3204 | 1.0000 |
| nway | 2 | 138 | **0.9855** | 0.6232 | 0.4275 | 0.1957 | 0.3841 | 0.9997 |

(GRID_REBASED seed 0: unseen_pair n=204 → **0.9853**, nway n=392 → **0.9898**.)
⭐ `validity == score` in **every** cell ⇒ where the overlay is valid it is **OPTIMAL**.

⇒ ⭐⭐ **GO — and wave 9's prediction was right in direction.** The fixed operator still
expresses the task on a 100 %-multi-key layer, but ⛔ **it is no longer perfect**
(0.9710–1.0000 against w9's flat 1.0000). ⛔ **The plan-space reference is therefore a
ceiling of 0.971–1.000, not 1.0000**, and every arm number is read against it.
⛔ **No per-instance tuning was needed, so the task file's stopping rule did not fire.**

---

## 4. ⭐⭐ THE SCORED RESULT — 3 seeds, TEST, all or none

All tables are **generated** by `w14_read_verb/tables.py` into `TABLES.md`; ⛔ no cell in
this report is hand-copied. Sources: `runs/scored1_seed0_test.json`,
`runs/scored2_seed1_test.json`, `runs/scored3_seed2_test.json`.

### 4.1 The leg

| quantity | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| budget words (`GRID_REBASED`) | 55/85/105 | 60/85/100 | 55/75/90 |
| training requests | 1442 | 1430 | 1451 |
| n `unseen_pair` / `nway` | 199 / 387 | 198 / 387 | 208 / 401 |
| **single-decode pipeline ceiling** `unseen_pair` | 0.6432 | 0.6869 | 0.7067 |
| store-span ceiling `unseen_pair` | 0.6382 | 0.6869 | 0.7067 |
| ⭐ **rank/width** (printed on every store row) | **0.9375** | **0.9375** | **0.9375** |
| read floats (atoms + address book) | 410,688 (7,936 + 402,752) | same | same |
| `tolerance_used` | **null** | **null** | **null** |

⭐ Shared encoder final reconstruction loss **1e-5 – 3e-5**; **pipeline ceiling `train` =
1.0000 on all three seeds.** Baselines matched to the store's 410,688 read floats:
mlp 410,920 · lstm 409,846 · transformer 405,776, all `matched: true`.

### 4.2 ⭐ BLIND-SEQUENTIAL, printed on THIS HUB'S OWN SPLIT

⛔ Never the published 0.4980; ⛔ never another split's number.
`w9_maze.envcols.blind_columns`, CMAZE-2b, TEST:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **BLIND-SEQUENTIAL** `unseen_pair` | **0.1759** | **0.1566** | **0.1202** |
| **BLIND-SEQUENTIAL** `nway` | **0.2145** | **0.1602** | **0.1197** |
| ORACLE (both rows) | 1.0000 | 1.0000 | 1.0000 |
| rule-B bend fraction `unseen_pair` | 0.9296 | 0.9545 | 0.9712 |

⭐⭐ **The request-layer repair holds on this hub's own split.** Wave 9's CMAZE-2 read
**0.9077** / **0.9229**; CMAZE-2b reads **0.12–0.21**. ⭐ Rule B now bends **93–97 %** of
held-out routes, so no row here is won on the easy half of the environment.
⭐ **An exact cross-check:** this hub's independently-run GRID_W9/val column read
**0.3188**, byte-for-byte the gate file's own `GRID_W9/val` entry
(0.3188405797101449).

### 4.3 ⭐⭐ THE PRIMARY ROW — `unseen_pair`, CLU, validity

| verb | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `FULL` (w9's read) | 0.2261 | 0.2929 | 0.3173 |
| `AVG` (**the ablation** — w9's read on the two halves) | 0.0704 | 0.1313 | 0.1298 |
| ⭐ **`MERGE` (the arm)** | **0.5126** | **0.6667** | **0.5769** |
| **MERGE − AVG** | **+0.4422** | **+0.5354** | **+0.4471** |

`nway` (the CONJUNCTION — two agents, one target, both routes valid):
`AVG` **0.0000 / 0.0160 / 0.0000** → `MERGE` **0.4891 / 0.4652 / 0.3915**
(**+0.4891 / +0.4492 / +0.3915**).
`budget_interp` (untrained budget words): `AVG` 0.0802 / 0.0875 / 0.0924 → `MERGE`
0.5142 / 0.5650 / 0.4988.

⭐ **The merge RULE is load-bearing — it is not "two decodes give more slots".** With the
identical two decoded plans (`TABLES.md`, merge-controls tables): `agent_wins` reads
0.2714 / 0.4848 / 0.4423, the agent half alone 0.1960 / 0.2424 / 0.2404 and the target
half alone 0.0704 / 0.1010 / 0.1202 — all far below `target_wins`.

### 4.4 ⭐⭐ WHY LATENT AVERAGING CANNOT WORK — the geometry, measured

| arm | quantity | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| `clu` | **cos(Z_A, Z_B)** | **+0.9668** | **+0.9589** | **+0.9630** |
| `clu` | cos(Z_AVG, Z_FULL) | +0.9971 | +0.9964 | +0.9966 |
| `lookup_matched` | cos(Z_A, Z_B) | +0.9214 | +0.9079 | +0.9148 |
| `transformer` | cos(Z_A, Z_B) | +0.9483 | +0.9496 | +0.9174 |

⭐⭐ **The two half-latents are 0.96-PARALLEL, and their average is 0.997-aligned with the
full read.** Averaging two nearly-parallel vectors is very nearly a no-op: it returns a
slightly shrunk copy of what the linear read already gives. ⛔ **The latent space has no
room for a union.** The *same two halves decoded* are complementary in PLAN space — 0.1960
and 0.0704 alone, **0.5126 overlaid**. ⭐ **That is wave 9's diagnosis turned into a
measured quantity: a read linear in the request design composes by averaging, and an
exact-judge route task composes by union.**

---

## 5. ⛔ THE BARS — every verdict recomputed from the JSONs by an independent verifier

| bar | seed 0 | seed 1 | seed 2 | verdict |
|---|---|---|---|---|
| **BAR 1** CLU `MERGE`−`AVG` ≥ +0.10 | +0.4422 | +0.5354 | +0.4471 | ⭐ **PASS** |
| **BAR 2** CLU delta − transformer delta ≥ +0.05 | +0.2111 | +0.2424 | +0.1442 | **PASS — ⛔ and see §5.1** |
| **BAR 3** `landscape_shuffled` ≤ 0.5 × CLU on `seen` | 0.5111 vs 0.3606 | 0.5552 vs 0.3455 | 0.5727 vs 0.3508 | ⛔ **FAIL** |
| **BAR 3** `random_atoms` ≤ 0.5 × CLU on `seen` | 0.5395 vs 0.3606 | 0.5650 vs 0.3455 | 0.5858 vs 0.3508 | ⛔ **FAIL** |
| **BAR 5** CLU `MERGE` > `lookup_matched` `MERGE` | 0.5126 vs 0.9648 | 0.6667 vs 0.9646 | 0.5769 vs 0.9808 | ⛔ **FAIL** |

### 5.1 ⛔⛔ BAR 2 PASSES ITS ARITHMETIC AND FAILS THE CLAIM IT WAS WRITTEN TO PROTECT

BAR 2 named one comparator, the transformer, and against that one comparator it passes.
⛔ **That is not the honest reading, and this hub will not report it as one.**
`MERGE − AVG`, row `unseen_pair`, every arm, TEST, 3-seed mean:

| lookup_matched | **CLU** | random_atoms | landscape_shuffled | transformer | mlp | lstm | clu_nopair | clu_lowrank | landscape_removed |
|---|---|---|---|---|---|---|---|---|---|
| **+0.8299** | **+0.4749** | +0.3557 | +0.3541 | +0.2757 | +0.1829 | +0.1380 | +0.0032 | 0.0000 | 0.0000 |

⛔ **The CLU's delta is not the largest — the matched-bytes lookup's is, by 0.355**, i.e.
the merge helps the floor **1.75× more** than it helps the store. The only arms it does
not help are the ones that read nothing (`landscape_removed`, atoms zeroed) or cannot
render a route at all (`clu_lowrank` at rank/width 0.1875, `clu_nopair`).
⇒ ⭐⭐ **THE MERGE HELPS EVERY ARM. THE COMPOSITION OPERATOR IS A PROPERTY OF THE TASK
INTERFACE, NOT OF THE STORE** — said in exactly the words the task file required.

### 5.2 ⛔ BAR 3 FAILS — the store is DECORATIVE here, and NO ROW IS CLAIMED FOR IT

`rank/width = 0.9375` on every store row. `landscape_shuffled` (same bytes, wrong basins)
and `random_atoms` (Gaussian atoms, same shape, same head fit) sit at **0.5111–0.5858** on
`seen` against the CLU's **0.6909–0.7212**, and at 0.3065–0.4747 on `unseen_pair` against
0.5126–0.6667. ⛔ **They do not collapse.** Only `landscape_removed` (atoms zeroed)
collapses to 0.0000, and `clu_lowrank` at **rank/width 0.1875** — the non-saturated regime
the Shapes3D incumbent lives in — scores **0.0000 on every held-out row on every seed**.
⭐ This is **wave 8's saturation trap, reproducing exactly as wave 9 recorded it**
(w9: 0.9396/0.9222 against 1.0000 at rank/d_k 0.938).
⚠ **And it is not a law about high rank/width:** wave 12 measured the same controls
collapsing to **0.000** at rank/width **0.9844** on Shapes3D
(`outputs/w12-advisor-report.md` §2). ⛔ **The decorative failure is a property of this
environment, not of the ratio.**
⇒ ⛔⛔ **NO ROW IS CLAIMED FOR THE STORE, on either verb.**
⭐ **What survives, and why:** `MERGE` and `AVG` are two verbs applied to the **same two
latents**, so the contrast between them does not depend on the store carrying information
— which is confirmed by the fact that the contrast is present on every arm. **A verb claim
is not a store claim. Only the verb claim is made.**

### 5.3 ⛔⛔ BAR 5 FAILS — and it is the most informative failure in the wave

⭐⭐ **Under `MERGE`, the matched-bytes lookup floor reaches 0.9648 / 0.9646 / 0.9808 on
`unseen_pair` and 0.9674 / 0.9626 / 0.9683 on `nway`** — essentially the plan-space
reference's own 0.9853 — against the CLU's 0.5126 / 0.6667 / 0.5769.
⭐ **And the verb reverses the ranking:** under `FULL` the lookup **loses** to the CLU
(0.1407 vs 0.2261, seed 0); under `MERGE` it **wins by 0.45**. ⛔ Charter C.1: an arm that
does not beat the matched-bytes lookup is **reported as decorative**, and this one does
not.

⚠ **DECLARED — the byte match is not binding at this configuration.** The store's 410,688
read floats buy 1,604 latents and the split has only 1,442 / 1,430 / 1,451 training items,
so `lookup_matched` (369,152 / 366,080 / 371,456 floats) **is** `lookup_full`: the floor is
*the whole training set*, inside budget. ⚠ Wave 9 had exactly the same (1208 == 1208).
⛔ **The floor must be read as "keep everything", not as "a compressed rival".**

### 5.4 ⭐⭐ THE MERGE BREAKS THE SINGLE-DECODE CEILING — a structural finding

| row | | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| `unseen_pair` | single-decode pipeline ceiling | 0.6432 | 0.6869 | 0.7067 |
| `unseen_pair` | `lookup_matched` under `MERGE` | **0.9648** | **0.9646** | **0.9808** |
| `unseen_pair` | **excess over the ceiling** | **+0.3216** | **+0.2778** | **+0.2740** |
| `nway` | **excess over the ceiling** | **+0.4118** | **+0.3915** | **+0.4022** |

⚠ **This is not a contradiction and must not be printed as one.** The "pipeline ceiling" is
`encode(true item) → decode → execute → judge`: it bounds an arm that emits **one** latent
and decodes **once**. `MERGE` decodes **twice** and takes the union, so it is **not bounded
by that ceiling**, and the lookup demonstrates it empirically on every seed and both
held-out rows. ⭐⭐ **The union of two decodes repairs what a single decode loses.**
⛔ There is no leak: both halves read **training latents only**, and the judge runs on the
**true** maze (`artefacts/VERIFY.txt`; `w14_read_verb/pipeline.py:read_values`).

---

## 6. ⭐ THE OUTCOME THE TASK FILE ASKED TO BE DECLARED IN ADVANCE — BAR 4

The task file required this be stated in exactly these words if a transformer plus the
same merge reached the ceiling. ⚠ **It did not**: the transformer under `MERGE` reads
**0.2965 / 0.3990 / 0.4279** on `unseen_pair`, below the CLU's 0.5126 / 0.6667 / 0.5769 and
well below the 0.6432–0.7067 ceiling. ⛔ **But the matched-bytes lookup floor did**, by
+0.27 to +0.41 (§5.4) — which is the same conclusion reached through a floor rather than a
learner, and it is therefore reported in exactly those words:

> ⭐⭐ **The CLU's differentiator in the maze is DELETION, which already transferred**
> (**7.5e-16 / 8.8e-16 / 8.2e-16**, projector **0.000e+00**, replicated from a clean
> checkout with **zero numeric differences**). ⛔ **That is a clean answer, not a failure.**

⭐ Wave 12 reached the same place from Shapes3D and said it in its own words: *"the
four-part ledger buys DELETION, not composition"* (`outputs/w12-advisor-report.md` §1).

---

## 7. ⚠ DISCLOSURES THIS HUB MAKES AGAINST ITSELF

1. ⛔ **Charter C.4 fires on the `seen` row.** `nearest_atom_distance / typical_inter_item`
   = **0.0445 / 0.0412 / 0.0420** (< 0.05) ⇒ *"LOOKUP TABLE: the read settles on a written
   item"*. On `unseen_pair` it does not (0.1349 / 0.1442 / 0.1438). ⛔ **The `seen` row's
   1.0000 must be read as a lookup verdict, not as a generalisation result.**
2. ⚠ **The scored budget condition departs from the gate file's declared primary.** The
   gate names `GRID_W9` primary for the ENVIRONMENT columns; this hub pre-registered
   `GRID_REBASED` by a rule mentioning **row sizes only** (match wave 9's 195/389 so the
   two waves' tables are comparable: 199/387, 198/387, 208/401 against `GRID_W9`'s
   69/128). ⚠ **The cost:** `GRID_REBASED` words are re-derived **per seed**, so the budget
   words are a per-seed constant like the bank, not one grid across seeds. Both conditions
   were run on VAL and both show the same effect (GRID_W9 seed 0: +0.3623).
3. ⚠ **The encoder pool is built by `build_site_v2`, not w9's `build_site`** — ≥ 2-key
   routes, to match the evaluation distribution. It is one shared encoder, so this choice
   hits every arm identically, but it is a difference from wave 9's pool and it is stated.
4. ⚠ **On CMAZE-2b the split's two blocks are AGENT-INDEX halves, not direction halves.**
   The gate declared direction is now a constant of the layer; `w9_maze.splits`' docstring
   wants "neither held-out set is a single direction", and that property no longer holds.
   It cannot confound a held-out comparison because it does not vary, but it is a
   difference from wave 9 and it is stated rather than hidden.
5. ⚠ **`FULL` reproduces wave 9 but is not identical to it.** On GRID_W9 seed 0 VAL,
   `FULL` = **0.1594** against wave 9's **0.1590** — the same band, different bank, split
   and layer. ⛔ It is a reproduction of the *band*, not a bit-level replication.
6. ⚠ **The `seen` row is identical between VAL and TEST by construction** (val and test
   share one training set). It is printed as a consistency check, ⛔ never as a second
   measurement.

---

## 8. ⛔ DECLARED NOT-RUN

| not run | reason |
|---|---|
| ⛔ **Any refusal DISPLAY row, in any form** | Gate 2 retires them (`CLOSEOUT.md`); RESULT rows never refuse. ⛔ Not a wall-clock casualty — a ruling. |
| ⛔ **Any post-deletion leg under the new verb** | Wall clock. Deletion is not this wave's dial; wave 9 already measured it. ⛔ **Never "no changes after deletion"** — post-deletion adaptation is **LIGHT and DATA-FREE**, and w9's AMENDMENT 1 (basis-change carry −0.1569 vs the corrected −0.0120 on VAL) is the record to read, not this file. |
| ⛔ **A scored `GRID_W9` leg** | One budget condition per pre-registration; `GRID_W9` is VAL-only here (seed 0, all four rows, `runs/dev1w9_seed0_val.json`). Seeds 1 and 2 on `GRID_W9` were **not** run. |
| ⛔ **A seed-2 VAL leg** | Selection was closed on seeds 0 and 1 before the ledger rows were written; a third VAL leg would have changed nothing and cost 40 min of a 3-core box. Seed 2's encoder was trained inside its scored leg, on TRAIN items plus the disjoint pool only. |
| ⛔ **Re-fitting the head on half-addresses** | That changes the FIT, not the VERB, and would void the dial. The head is fitted on FULL requests only and read at the halves. |
| ⛔ **Any second merge rule as a candidate** | A tuned merge is not a fixed operator. `agent_wins` / `agent_only` / `target_only` are reported as CONTROLS on the deployed rule, never as alternatives selected between. |
| ⛔ **GPU, cluster, any scale above `std`** | Local CPU only, 3 cores, 3 concurrent processes. |
| ⛔ **Any change to `w8_envscout.maze_colour` or `w9_maze/`** | Imported unmodified; the judge's sha256 is checked against the gate's. |
| ⛔ **A CMAZE-1 column anywhere** | Different environment (w9 AMENDMENT 2). Machine-checked: `VERIFY.txt` check 8. |

---

## 9. ⛔ WHAT WOULD HAVE PROVEN THIS HUB WRONG — and what actually happened

| the task file's falsifier | outcome |
|---|---|
| **The plan-space merge does not beat latent averaging** ⇒ the read verb is not the lever and w9's diagnosis was wrong | ⛔ **Did not fire.** +0.4422 / +0.5354 / +0.4471 on the primary held-out row, three seeds, same direction. |
| **The merge helps every arm equally** ⇒ a property of the task interface, not of the store | ⭐⭐ **FIRED.** It helps every arm, and helps the matched-bytes floor **1.75× more** than the store. **Reported as an interface property, in those words.** |
| **The controls fail to collapse** ⇒ decorative; claim nothing | ⛔ **FIRED.** `landscape_shuffled` and `random_atoms` do not collapse on any seed. **No row is claimed for the store.** |
| **A fixed merge cannot express the task without per-instance tuning** ⇒ stop | ⛔ **Did not fire** (§3). The operator reaches 0.971–1.000 with no tuning. |

---

## 10. WHAT THE HUB ASKS THE ADVISOR TO RULE ON

1. ⭐⭐ **The plan-vs-latent result is now a held-out, three-seed, exact-judge number, not a
   post-hoc analysis.** The w10 referee called wave 9's §6.1 *"the best thing in the paper,
   and exact"* (`outputs/w10-advisor-report.md` §2). ⛔ **Does it now lead — as an
   INTERFACE result, with the store explicitly not claimed?**
2. ⛔ **The matched-bytes lookup beats the store under the new verb and loses under the old
   one.** ⛔ **Is the composition story now a statement about operators rather than about
   stores** — and does that change which arm the composition rows quote (cf. w12's open
   question 1, `pca_tensor` at 1.37× equalling the single design at 24.4×)?
3. ⚠ **The merge exceeds the single-decode pipeline ceiling by +0.27 to +0.41.** ⛔ **Does
   the paper's "ceiling" column need re-defining per VERB**, since a two-decode read is not
   bounded by a one-decode ceiling?
4. ⚠ **The maze's decorative failure is environment-specific**, not a law about rank/width
   (w12 saw the controls collapse at 0.9844 on Shapes3D). ⛔ **Should the saturation caveat
   be re-scoped to the maze in the registry?**
5. ⚠ **Charter C.4 calls the `seen` row a lookup table on all three seeds.** ⛔ Should the
   `seen` row be dropped from RESULT tables entirely and kept only as an instrument check?

---

## 11. REPRODUCING IT

```
python -m w14_read_verb.pool <shard> 90 <pool_dir>            # 3 shards, seeds 7,000,000+
python -m w14_read_verb.overlay_ref <bank.pkl> <seed> <out.json> [grid]   # the GO/NO-GO
python -m w14_read_verb.leg --seed S --which val  --tag dev1    --grid rebased
python -m w14_read_verb.leg --seed S --which test --tag scoredN --grid rebased
python -m w14_read_verb.tables C:/chlu/outputs/w14-read-verb > TABLES.md
python -m w14_read_verb.verify C:/chlu/outputs/w14-read-verb   # exit 0 only if all pass
```

⭐ `leg.py` **refuses** a `--which test` run unless both gate files are present and ready
**and** a `/c/chlu/SCORED-LEDGER.md` row for target `read-verb` already names that leg's
tag — so the standing rule *"the row is written BEFORE the run"* is enforced by the code,
not by discipline.
⭐ `verify.py` does **not** import `pipeline.py`, `report.py`, `tables.py` or `leg.py`: it
re-derives every verdict from the result JSONs, the gate files and the ledger.
⚠ Its six FAILs are the **pre-registered** BAR-3 control failures, computed by the verifier
itself rather than quoted from this report — which is how the negative is established
rather than asserted.
