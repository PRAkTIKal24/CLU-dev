# w19 hub/maze-t-symmetric — DOES THE CMAZE-T WIN SURVIVE SYMMETRIC INPUTS?
**Agent type: autonomous-hub. Target `maze-t-symmetric`. Budget 9 scored runs (3 conditions × 3 seeds).**
⭐⭐ **PRE-AUTHORISED IN FULL. No decision from the Head or the Advisor is needed anywhere in this wave. Never wait for either.**

## Base and paths
Base = **`w16/maze-wm` tip (`9f80f59`)**, which descends from `b1ca534` (the merge of both w15 harness branches) — i.e. **the branch wave 16 actually scored on**, with w16's own fixes included.
Worktree `/c/chlu/worktrees/CHLU-w19-maze-t-symmetric`, branch `w19/maze-t-symmetric`, outputs `/c/chlu/outputs/w19-maze-t-symmetric/`.
Before any number: `source /c/chlu/env_w3.sh`, assert `chlu.__file__` is inside your worktree. ⛔ If not, STOP.
⚠ **This box's python is Windows-native and CANNOT open an msys `/c/...` path.** Convert first: `W=$(cygpath -m "$F")`. Do not rediscover this.

## The one job
⛔ **NOTHING about the instrument changes.** The CMAZE-T harness, banks, hold-outs, bars, controls, blind arms, lookup floors and the winning store config (**Config A1 = latent 128, K 2, 6 atoms**; **C-512** as replication) are **UNCHANGED**.
⭐ **The only thing that changes is the INPUT — and within a condition EVERY arm gets the identical input.**

### The conditions
1. **WORDS-MAIN** — the 27 main-effect one-hots of the five words, for the CLU and every rival. ⭐ The CLU's design is then main effects only, so **this is also the learned blind-sequential reading — print it as such.**
2. **WORDS-FULL** — the full **318-wide** design (bias + 27 main + 290 pair indicators) for the CLU and every rival (MLP / LSTM / transformer / cont-MLP at **matched read-path floats**). ⭐ **Decides whether the pair block or the store carries the margin.**
3. **OBS** — conv-AE features ⊕ the **29 dynamics scalars**, for every arm. ⭐ The CLU's read stays **request-only**: a **DECLARED word extractor** (small classifier obs → five words, trained on **TRAIN rows only**, its **accuracy printed on every row including held-out combos**) feeds the request head. ⛔ **Rivals get the raw observation AND a second rival row with the extractor's output appended**, so the extractor is **not a private channel**.
4. **BOTH** (words + observation, every arm) — run on **VAL for all arms**; ⭐ scored **only if runs remain**.

### Rows, per condition
`unseen n-way` (**primary**) · `seen` · `unseen 2-way` · `value hold-out` · `per-step exactness`. **Rollout validity printed ONCE as a limitation.**
Rivals trained **8,000 steps** + the **10× steps diagnostic column**. **Charter-C controls per condition. 3 seeds.**
**Deletion rows re-run on the winning condition.**

## ⛔⛔ THE REPORTING RULE — this is the point of the wave
**Whatever the outcome — a win in all conditions, in some, or in none — the paper prints the condition table IN FULL.**
⛔ **A loss in a condition is a PRINTED ROW, not a reason to drop the task.** ⛔ Do not re-tune to rescue a condition. Report every condition whether it wins or loses.

## Table discipline
⛔ **Every table names its condition.** ⛔ **Conditions never share a table with wave 16's rows** — w16's rows stay as the draft's placeholder until your report exists.
⛔ CMAZE-1 / CMAZE-2 / CMAZE-2b / CMAZE-T / CMAZE-O never share a table. Seed count is a column. Every byte figure names its convention.

## ⚠ Two defects from earlier waves you must not re-introduce
- ⛔ **`landscape_shuffled_across_wells` must be a DERANGEMENT, printed.** `rng.permutation(K)` at K=2 is the identity half the time; on w16's Config-A VAL it WAS the identity on seeds 0 and 1 (registry `N350`). **A permutation that can fix a well is not a control.**
- ⛔ **Rivals must be trained to the declared budget before any margin is claimed.** In w16 a recorded CMAZE-O win was withdrawn purely because rivals were under-trained (referee M1). Print the 8,000-step row AND the 10× diagnostic column.

## Spending rule
**Val-first. VAL IS FREE.** Scored runs (3 seeds) go ONLY to a condition that passes its val bars. A condition failing val is **DECLARED** and the next starts — ⛔ it does not consume a scored run. ⭐ **Write the ledger row in `/c/chlu/SCORED-LEDGER.md` BEFORE each scored run**, naming its PREREG.

## Acceptance criterion
`FINDINGS.md` + `MAZE-T-SYMMETRIC.json`: every condition with its val numbers and verdict, every scored run with its ledger row and PREREG, the 9-run budget fully accounted (spent or explicitly declared), charter-C controls per condition with the derangement stated, the word-extractor accuracy on every OBS row, and the condition table printed in full including losses.

## Operating rules
⛔ Never run a box-wide kill — capture your PIDs into `PIDS.txt`. ⛔ No sub-agents. ⛔ Poll background jobs to completion INSIDE a turn. ⛔ Avoid `rm <glob>` — the dangerous-rm guard is not covered by bypassPermissions and will stall you; use a fresh timestamped subdir. ⭐ `LOG.md` continuously; `START_UTC.txt` now; `END_UTC.txt` only when genuinely finished; commit the branch whole and `git bundle create /c/chlu/outputs/w19-maze-t-symmetric/w19-maze-t-symmetric.bundle w19/maze-t-symmetric` then verify it. ⭐ **Say what did NOT run.**
