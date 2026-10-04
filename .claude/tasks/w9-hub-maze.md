# w9 hub/maze — TARGET: the world-model audience

**Boot:** charter → `w9-hub-common.md` → this file. **Base `base/iter8 = fa60b15`.**
**Worktree:** `/c/chlu/worktrees/CHLU-w9-maze` · **LOG:** `/c/chlu/outputs/w9-maze/LOG.md`
**Budget: `maze`, 3 runs.** ⛔ Gated on `HARNESS-READY.json`. ⭐ DEV starts now.

## DIAL DECLARATION
**Dial: does the object transfer off Shapes3D.** Controls every row: landscape-removed,
landscape-shuffled, the table floor, and the fragment-lookup and composition-blind columns printed.
**Falsified if** the CLU loses the RESULT rows to a matched-parameter learned baseline — the same
bar `pca_tensor` cleared on Shapes3D.

## RESULT rows (vs learned baselines at matched parameters; table is a printed floor)
1. **seen** — a trained agent–target pair on a seen maze.
2. **unseen agent–target pair on a seen maze** — a new legal route.
3. **n-way** — two agents to one target; agent × target × maze.
4. **budget interpolation** — a budget between trained values.

## DISPLAY rows (inductive bias; ⛔ never quoted as RESULTs)
- **refusal** when the budget is below the shortest legal route, or the target is walled.
- **post-deletion refusal** — delete `move-left`; delete the `green` target.
- **post-deletion composition without retraining.**
⭐ The **only** comparator is the supervised refuser; our line is **"no impossible example ever seen."**

## ⚠ The caveat that travels with every composition claim
**BLIND-SEQUENTIAL scores 0.4980** on this environment. ⭐ Print it beside every RESULT row.
⭐ In your favour, and print this too: **FRAGMENT-LOOKUP is 0.0000 at both radii** — the tripwire
that killed GYM-1 and GYM-2 is dead here — and **rule B bends 81.6 % of routes**.

## ⛔ What would prove this hub wrong
- A matched-parameter LSTM or transformer beats the CLU on the RESULT rows ⇒ the Shapes3D result
  does not transfer, and that is the wave's answer. **Report it.**
- The CLU wins only where BLIND-SEQUENTIAL also wins ⇒ you measured the easy half of the environment.
  ⭐ **Break the rows out by whether rule B bends the route.**
- Post-deletion composition needs retraining ⇒ the deletion story does not transfer off Shapes3D.
- Fragment-lookup rises above 0.0000 on YOUR split ⇒ ⛔ your split leaks; the environment's does not.
