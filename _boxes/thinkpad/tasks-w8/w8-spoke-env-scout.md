# w8 spoke/env-scout — find the maze environment wave 9 will run on, or spec the build

- **Agent:** `web-scout` (with an engineer read of any candidate's code).
- **Output:** `/c/chlu/outputs/w8-env-scout/ENV.md` · ⛔ No scored run. **Build only if nothing fits.**

## DIAL DECLARATION
**Dial: the instrument.** **Falsified if** an adopted environment cannot give an **exact** validity
check — an approximate judge makes every downstream refusal and deletion claim approximate.

## ⛔ Why this matters — two environments have already failed
GYM-1 (1-D) and GYM-2 (2-D) both failed their adoption gates. ⭐⭐ **The law those two spokes
established, and the thing you must screen for:** *to test "compose rule A with rule B to find a
different route", at least one rule must be able to render a route **ILLEGAL**, not merely slower.*
A speed limit is an upper bound and only changes a route's cost — no metric and no number of
dimensions makes it composable.
⚠ **And legality alone is not sufficient:** GYM-2 had infeasibility and still failed, because a
**composition-blind policy scored 0.9367–1.0000**. ⭐ The interaction between the two rules must be
worth enough that a policy blind to it **cannot** score well.

## The job
Assess **MiniGrid / BabyAI door–key first**, then procedurally generated maze tasks and maze datasets.
**Adopt if it gives (a), (b) and (d); (c) is desirable:**
- **(a)** random mazes per episode;
- **(b)** a legality rule that **composes across agent × key/colour × door**;
- **(c)** a budget;
- **(d)** an **exact** validity check.
⭐ **Otherwise spec the build:** colour-keyed forbidden tiles, random mazes, a budget, an exact judge.

⭐ **Report the screening numbers for any candidate you recommend**, not just the feature list: the
composition-blind score and the fragment-lookup score. ⚠ **Under wave 8's rules these are REPORTED
COLUMNS, not abort conditions** — ⛔ do not reject a candidate because a number fires; report it and
let the Head weigh it.

## ⛔ What would prove this spoke wrong
- You adopt an environment whose validity check needs a tolerance ⇒ every downstream claim inherits it.
- You adopt one where a composition-blind policy scores well ⇒ the third failed gym in a row.
- MiniGrid/BabyAI fit on paper but their door–key legality does not actually compose across colour ⇒
  ⭐ say so with the code you read; that is exactly the trap the first two gyms fell into.
