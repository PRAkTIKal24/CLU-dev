> ## ⛔ STANDING RULES — evidence base `/c/chlu/evidence/outputs/` (w9-, w10-, w12-advisor-report.md); ⛔ NOT `$CHLU_CLAUDE_ROOT`, a stale frozen snapshot. Write ONLY to your own output dir + worktree. START_UTC.txt first, LOG.md throughout, END_UTC.txt last. Assert `chlu.__file__` is inside YOUR worktree before any number. Poll background jobs to completion INSIDE a turn. ⛔ NEVER a box-wide kill — other agents share this box; capture your own PIDs. Bundle YOUR branch only, never `--all`; one ref. A published file is not a finished file. Anything not run is a DECLARED NOT-RUN with a reason.

# w14 spoke/request-layer-fix — make CMAZE-2's rule B actually bite

- **Agent:** `experiment-engineer`. ⭐ **ZERO scored runs.** ⭐⭐ **PRECONDITION for `hub/read-verb`
  (Head ruling) — no maze run happens until this lands.**
- **Output:** `/c/chlu/outputs/w14-request-layer-fix/` incl. **`REQUEST-LAYER-READY.json`**.
- **Owns:** its output dir + branch `w14/request-layer-fix`.

## THE ONE JOB — the defect, in one measurement
⛔⛔ **CMAZE-2's rule B is untested on most routes: 1,242 of 1,449 routes use EXACTLY ONE KEY, so
there is no key ORDER for rule B to trap.** ⚠ That is why blind-sequential reads **0.9037–0.9130** on
w9's split against **0.5317** on CMAZE-1 proper at the harness tier, and **0.4980** as published.
⭐ **The environment is fine as adopted. The REQUEST LAYER is what broke it.** Fix the request layer.

## ACCEPTANCE CRITERION — a number, not a feature list
⭐ **A large majority of routes must require ≥ 2 keys**, and ⭐⭐ **BLIND-SEQUENTIAL on the fixed
split must fall back toward the published 0.4980.** Publish in `REQUEST-LAYER-READY.json`:
- the **key-count histogram** over routes (before and after);
- **BLIND-SEQUENTIAL, BLIND-A, BLIND-B, BLIND-COLOUR, BLIND-FACTOR** on the fixed split;
- **FRAGMENT-LOOKUP r=1 and r=2, CROSS-MAZE** ⭐ (the cross-maze quantity is the published one —
  0.0006 / 0.0000; ⛔ the same-maze 0.0741 / 0.5105 is a DIFFERENT quantity, label both);
- **ORACLE** (must stay 1.0000) and the **rule-B bend fraction**.
⛔ **The exact integer judge does not move.** `tolerance_used` must stay **null**.

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
- Multi-key routes exist but **blind-sequential stays high** ⇒ ⭐ rule B does not compose in this
  environment at all, and **that is a finding that kills the maze track** — report it, do not tune.
- Forcing ≥ 2 keys makes **ORACLE < 1.0000** ⇒ the generator produces unsolvable instances; stop.
- ⚠ The fix requires changing the **judge** ⇒ ⛔ **STOP.** Every downstream deletion and refusal claim
  inherits an exact judge; a tolerance anywhere voids them.
