# w16 hub/maze-wm — THE MAZE, ALL OUT
**Agent type: autonomous-hub. Target `maze-wm` = 18 scored runs, hub-owned.**
⭐⭐ **This is the wave the paper's maze section depends on. It is PRE-AUTHORISED IN FULL: you never wait for the Hub, the Head or the Advisor.**

## Your base and paths
Worktree `/c/chlu/worktrees/CHLU-w16-maze-wm`, branch `w16/maze-wm`, base `base/iter8 = 148f3b7`.
Output dir `/c/chlu/outputs/w16-maze-wm/` — the ONLY place you write outside your worktree.
Before any number: `source /c/chlu/env_w3.sh` and assert `chlu.__file__` resolves inside your worktree. ⛔ If it does not, STOP.

## Your instruments (built in wave 15 — read their HARNESS-READY.json first)
- **CMAZE-T** (transition): `/c/chlu/outputs/w15-cmaze-t-harness/` — engine/solver/judge from CMAZE-1, unmodified.
- **CMAZE-O** (observation generation): `/c/chlu/outputs/w15-cmaze-o-harness/`.
⚠ **Read each harness's `instrument_warnings` before you spend anything.** If a harness warns that a blind arm or the lookup floor reaches the CLU, that is information about what a win there would mean — factor it into which task you push, and record the reasoning in `LOG.md`.

## ⭐⭐ THE SPENDING RULE — this is the whole method
- **Every config is measured on VAL first. VAL IS FREE.**
- Scored runs (**3 seeds = 3 runs per config**) go ONLY to a config that **passes its val bars**.
- A config that **fails val** is **DECLARED** in `LOG.md` with its numbers, and **the next config starts**. ⛔ It does not consume a scored run.
- A config that **passes val but misses the scored bar** is **DECLARED**, and **new configs keep coming until the 18 runs are spent.**
- ⛔ **Do not stop early. Do not bank runs.** Keep trying configs until the budget is spent.
- ⭐ Write the ledger row in `/c/chlu/SCORED-LEDGER.md` **BEFORE** each scored run, naming its PREREG.

## Task order — Head-ruled, follow it
### 1. Task 1 — CMAZE-T (transition). Budget cap: 6 scored runs.
**Config A = the Shapes recipe UNCHANGED:** label-free wells; K by the crossover rule on val; **rank/width ≤ 0.3 printed**; tensor head over the transition factors; shared frozen encoder/decoder; latent-mixing read.
Best val config → 3 scored seeds vs rivals.
If A fails, **up to TWO amended iterations**, each seeded only when val passes:
- **Config B** = A + the task-shaped interface for rollouts — ⛔ **every rival gets the IDENTICAL interface.** (w14 proved the merge lifts the lookup floor MORE than the CLU; an interface given only to the CLU proves nothing.)
- **Config C** = K/rank re-sweep on the maze latent + item/field store ablation + address-book rank as a dial.

### 2. Task 2 — CMAZE-O (observation generation). Budget cap: 6 scored runs.
Same config ladder, same rule.

### 3. Task 3 — fallback, ONLY if Tasks 1–2 yield no win. Budget: the remaining runs (≥ 6).
The route task on **CMAZE-2b**, **CLU vs NN rivals ONLY** (MLP / LSTM / transformer at matched params), the **lookup printed as the floor with the cross-maze fragment-lookup column beside it**, the **w14 merge as the SHARED interface**, seeded. Configs continue on val until the remaining runs are spent.

## On the winning task/config, additionally produce
Deletion rows (**value**, **conjunction**) · post-deletion composition vs rival **retrain-without** · **adaptation cost printed** · **rollout validity** where the task has rollouts.

## ⛔ The claim gate
A **store** row is claimed only if the **charter-C controls collapse** (`landscape_removed`, `landscape_shuffled`, `landscape_shuffled_across_wells`, `random_atoms`). If they do not, it is an **INTERFACE row or NOTHING** — say which, explicitly, in the row itself.

## If no task yields a win after 18 runs
⭐ **That is a complete and publishable outcome — write it properly.** For each negative, name **which link fails**: content · factorisation · read · interface. The paper's maze section then = deletion transfer (CMAZE-2b) + the verb finding (w14) + these negatives. **Still an honest main-text section.** ⛔ Do not pad, do not re-tune to manufacture a win, do not report the best seed alone.

## Standing rules that bind every row
- **Naive vs naive:** rivals matched on read-path params (matched params on the maze). Memory ledger printed separately, never matched. ⛔ The inflated-rival row does not exist.
- **3 seeds on every number. ⛔ No 5-seed rows anywhere. ⛔ Never "9/9 seeds".**
- ⛔ **CMAZE-1 / CMAZE-2 / CMAZE-2b / CMAZE-T never share a table.**
- ⛔ **Every byte figure travels with its convention named.** Seed count is a column.
- **Deletion language:** store-scoped exactness; frozen encoder/decoder outside the deletable object; adaptation light and data-free — ⛔ never "no changes".
- ⛔ **Never run a box-wide kill.** Capture your PIDs into `PIDS.txt`; kill only those.
- ⛔ **Poll background jobs to completion INSIDE a turn.** Never end a turn with detached work running.
- ⛔ **No sub-agents.**
- ⭐ Where a choice is open, take the option closest to an **ICLR main-track result**, record it in `LOG.md`, and proceed.

## Acceptance criterion
`FINDINGS.md` + `MAZE-WM.json` recording: every config tried with its val numbers and verdict, every scored run with its ledger row and PREREG, the 18-run budget fully accounted (spent or explicitly declared), the control gate per claimed row, and for each negative the named failing link. Plus `LOG.md` written continuously, `START_UTC.txt` at the start, `END_UTC.txt` only when genuinely finished, and the branch committed whole and bundled (`git bundle create /c/chlu/outputs/w16-maze-wm/w16-maze-wm.bundle w16/maze-wm`, then `git bundle verify`).
