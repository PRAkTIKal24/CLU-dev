# PREREG — `w14 hub/read-verb`, scored runs 1–3 of 3 on target `read-verb`

**Written 2026-09-22, BEFORE any TEST leg and before any ledger row.** ⚠ All times UTC
(the box's local clock is UTC+2). Branch `w14/read-verb`, worktree
`C:/chlu/worktrees/CHLU-w14-read-verb`, output `/c/chlu/outputs/w14-read-verb/`.
Base `base/iter8 = 6699665`, with `w14/request-layer-fix` merged in.

---

## 0. THE QUESTION

> **Is the COMPOSITION OPERATOR part of the interface, and is it TASK-SHAPED?**

Wave 9 measured, on CMAZE-2, that the store holds every piece of every held-out answer
(plan coverage **1.000**) and that the factorisation the address book assumes is correct
(same-agent ∪ same-target training plans cover **0.997** of a held-out route) — and yet:

> **Overlaying the two halves IN PLAN SPACE scores 1.0000 — optimal on every pair, all
> three seeds. Averaging the identical halves IN LATENT SPACE scores 0.13.**
> — `outputs/w9-advisor-report.md` §1

So the read is LINEAR in the request design; composing two addresses composes their
LATENTS, while an exact-judge route task composes in the PLAN. This wave puts that
diagnosis on a held-out estimand by making the READ VERB — and nothing else — the dial.

## 1. THE DIAL, AND WHAT IS HELD FIXED

**Held fixed:** the same store (`w9_maze.arms.CluRead`), the same request-only read to
per-address latents (`chlu.core.gen_request_head.head_read` — no feature argument, by
signature), the frozen encoder and the shared decoder (`w9_maze.encoder`), the
parameterless executor (`w9_maze.render.execute`), the exact integer judge
(`w8_envscout.maze_colour.judge`, imported, not wrapped, not reimplemented), the split,
the item and the learned baselines at matched parameters.

**The two half-addresses** are the head's own `NaN` = "not requested" encoding:

    half A (AGENT half)  = (m, a, NaN, b)      7 of the 11 non-zero design columns
    half B (TARGET half) = (m, NaN, t, b)      7 of the 11 non-zero design columns

Their union is the FULL design minus exactly one block, `(agent, target)` — the one
block a held-out pair cannot ground in any case.

| | |
|---|---|
| **THE ARM** `MERGE` | `overlay( decode(Z_A), decode(Z_B) )`, target's plan wins conflicts |
| **THE ABLATION** `AVG` | `decode( (Z_A + Z_B)/2 )` — wave 9's read, identical two latents |
| **reference** `FULL` | `decode( Z_(m,a,t,b) )` — wave 9's arm, unchanged |
| **merge controls** | `agent_wins`, `agent_only`, `target_only` |

⛔ **Every arm gets both verbs and the identical decoder**: `clu`, `clu_lowrank`,
`clu_nopair`, `landscape_removed`, `landscape_shuffled`, `random_atoms`,
`lookup_matched`, `lookup_full`, `mlp`, `lstm`, `transformer`. A merge only the CLU is
given is not a result.
⛔ Nothing in the verb is fitted: no threshold, no per-instance tuning, **`tolerance_used`
stays `null`** on every leg.

## 2. THE ENVIRONMENT, THE SPLIT AND THE GRID — all fixed before this file

* **Request layer: CMAZE-2b** = `w14_request_layer.task2.build_site_v2`, this hub's gate
  (`REQUEST-LAYER-READY.json`, `status: ready`, all four ACCEPTANCE items pass,
  `scored_runs_spent: 0`). **100 % of routes take ≥ 2 keys.**
  ⛔ **The harness tier is CMAZE-1; this hub's maze is CMAZE-2b. They NEVER appear in one
  table** — different environment, not merely different budget (w9 AMENDMENT 2).
* **Banks:** `/c/chlu/outputs/w14-request-layer-fix/banks/bank_after_seed{0,1,2}.pkl`,
  96 sites each, built by the gate spoke. **Encoder pool:** 270 sites this hub built at
  seed range 7,000,000+ (`w14_read_verb/pool.py`), disjoint from every evaluation bank.
* **Budget condition: `GRID_REBASED`**, pre-registered in `LOG.md` §07:21Z by a rule that
  mentions **row sizes only** — it is the condition whose held-out row sizes match wave
  9's (204/392 against w9's 195/389) so the two waves' tables can be read against each
  other. Words quoted from the gate file: s0 55/85/105 (interp 70/95), s1 60/85/100
  (72/92), s2 55/75/90 (65/82). `GRID_W9` reported beside it **on VAL only**.
* **Store configuration:** wave 9's own scored configuration, carried unchanged —
  `n_atoms=31, K=8, ridge=0.01, ae_steps=12000, bl_steps=4000, LATENT_D=256`;
  `clu_lowrank` at `n_atoms=7`. ⛔ The store is not a dial this wave.
* **Rows:** `unseen_pair` (**PRIMARY** — the composition row), `nway`, `seen`;
  `budget_interp` reported beside them. **Selection is on VAL only**; the scored legs are
  TEST, one per seed, seeds 0/1/2, **all or none**.
* **BLIND-SEQUENTIAL is printed on this hub's OWN split**, from
  `w9_maze.envcols.blind_columns`. ⛔ Never the published 0.4980 and never another
  split's number.

## 3. ⭐ THE GO/NO-GO IS ALREADY ANSWERED, ON VAL, AT ZERO SCORED COST

The task file's stopping rule is *"a fixed merge cannot express the task without
per-instance tuning ⇒ stop"*. Measured on VAL, all three banks, with whole written
TRAINING plans (a REFERENCE, not an arm): the fixed `target_wins` overlay reaches
**0.9710 / 1.0000 / 1.0000** (unseen_pair) and **0.9844 / 1.0000 / 0.9855** (nway) on
GRID_W9, and **0.9853 / 0.9898** on GRID_REBASED seed 0, with `validity == score` in
every cell (⇒ where valid it is OPTIMAL). **No per-instance tuning was used.**
⛔ **The operator is therefore no longer perfect** — wave 9 predicted exactly this for a
≥ 2-key environment — so **the plan-space reference is a ceiling of 0.971–1.000, not
1.0000**, and every arm number is read against it. **The stopping rule does not fire.**

## 4. ⛔ THE BARS — declared here, before any TEST number exists

**BAR 1 — THE WAVE'S QUESTION.** On the CLU, on the PRIMARY row (`unseen_pair`, TEST),
`MERGE(target_wins) − AVG ≥ +0.10` validity **on all three seeds, same direction**.
⛔ **If BAR 1 fails, the read verb is NOT the lever and wave 9's diagnosis was wrong, and
this hub reports that** — the task file names it the most valuable outcome this wave can
have, and it will be reported as such, not buried.

**BAR 2 — WHOSE PROPERTY IS IT?** The CLU's `MERGE − AVG` delta must exceed the
matched-parameter transformer's by `≥ +0.05` on the primary row **on all three seeds**.
⛔ **If BAR 2 fails — if the merge helps every arm equally — then the composition operator
is a property of the TASK INTERFACE, not of the store, and this hub says exactly that,
in those words.**

**BAR 3 — CHARTER C, THE CONTROLS.** On every scored row, under the DEPLOYED verb,
`landscape_shuffled` and `random_atoms` must collapse: control validity ≤ **0.5 ×** the
CLU's on the `seen` row. ⛔ **If they do not collapse, the store is DECORATIVE there and
NO ROW IS CLAIMED FOR THE STORE.**
⚠ **Declared in advance:** wave 9's controls did **not** collapse at this configuration
(0.9396 / 0.9222 against the CLU's 1.0000 on `seen`, at `rank/d_k = 0.938`), and wave 8's
saturation trap is the stated reason. This bar is expected to fail again. ⭐ **It does not
void the wave**, because `MERGE` vs `AVG` is a contrast of two VERBS on **identical
latents** and does not depend on the store carrying information — but the two claims are
kept apart: **a verb claim is not a store claim**, and only the verb claim survives a
BAR-3 failure. **rank/width is printed on every store row.**

**BAR 4 — THE ANSWER THAT IS STILL A RESULT, declared now.** ⭐⭐ **If a transformer plus
the SAME merge also reaches the ceiling, then the CLU's differentiator in the maze is
DELETION, which already transferred** (7.5e-16 / 8.8e-16 / 8.2e-16, projector 0.000e+00,
replicated from a clean checkout with zero numeric differences). ⛔ **That is a clean
answer, not a failure**, and it will be reported in exactly those words.

**BAR 5 — THE FLOOR.** `MERGE` on the CLU must beat `MERGE` on `lookup_matched`
(charter C.1: an arm that does not beat the matched-bytes lookup is reported as
decorative) and must beat `landscape_removed` under the same verb.

## 5. ⛔ WHAT IS NOT BEING CLAIMED, AND WHAT IS NOT BEING RUN

* ⛔ **No refusal DISPLAY row, in any form.** Gate 2
  (`w14-refusal-displays-closeout/CLOSEOUT.md`) retires them; RESULT rows never refuse.
* ⛔ **No CMAZE-1 number in any table with a CMAZE-2b number.**
* ⛔ **No post-deletion leg is claimed here.** If it is run it is VAL only, and
  ⛔ **never "no changes after deletion"** — post-deletion adaptation is **LIGHT and
  DATA-FREE**.
* ⛔ **The best seed is never reported alone; there is no "9/9 seeds".** Three seeds, all
  or none, each printed.
* ⛔ **The exact integer judge does not move.** `tolerance_used: null` is asserted in
  every result file.

## 6. THE LEDGER

Three rows will be appended to `/c/chlu/SCORED-LEDGER.md`, target `read-verb`, **one
BEFORE each scored TEST leg**, naming this file. `w14_read_verb/leg.py` refuses a
`--which test` run unless both gate files are present and ready AND a ledger row for
`read-verb` carrying that leg's tag already exists.
