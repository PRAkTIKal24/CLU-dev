> ## ⛔ STANDING RULES FOR THIS BOX — read before the job below
> - ⭐ **The evidence base is `/c/chlu/evidence/outputs/`** (live, shipped 2026-09-21).
>   ⛔⛔ **DO NOT read `$CHLU_CLAUDE_ROOT`** — it points at a FROZEN, OLDER snapshot of the same
>   tree, and every number in it is out of date. Reading it silently produces a stale paper.
> - ⭐ **You write ONLY to `/c/chlu/outputs/w10-referee/`.** ⛔ No other spoke's tree. ⛔ Never
>   into the Head's profile. ⛔ Never create a `.claude/` directory in your worktree.
> - ⭐ **Poll any background job to completion INSIDE a turn** (loop on its output file with sleep).
>   ⛔ Do not end a turn while detached work is running — nothing will re-invoke you.
> - ⛔⛔ **Never run a box-wide kill** (`taskkill /F /IM python.exe` or equivalent). Kill only PIDs
>   you captured at launch. Another agent's training run has been destroyed this way before.
> - ⭐ **Commit and bundle your branch WHOLE before you finish** — `git bundle create <f> <your
>   branch>`, ⛔ not `--all`, which mixes other agents' branches into your file.
> - ⭐ Write a `LOG.md` in your output dir as you go. It is the record, and the Hub reads it.
> - ⭐ If something is not run, it is a **DECLARED NOT-RUN with a reason**, ⛔ never a zero, a blank
>   or a silent omission.

# w10 spoke/referee — adversarial review of DRAFT-v0 before it reaches the Head

- **Agent:** `paper-referee`.
- **Output:** `/c/chlu/outputs/w10-referee/REFEREE-v0.md`.
- **Owns:** `/c/chlu/outputs/w10-referee/**`. ⛔ Read-only on the draft and the repo.
- **GATED — mechanical precondition:** ⛔ do not start until
  `/c/chlu/outputs/w10-draft-v0/DRAFT-v0.md` exists.

## THE ONE JOB
Referee the skeleton as the hostile-but-fair composite of the reviewers we actually draw at ICLR.
Itemised report, **must-fix / should-fix / nice-to-have**, plus a simulated accept/reject verdict.

## ⭐ WHERE TO PUSH HARDEST — this programme's actual failure modes, not generic ones
1. ⛔ **Cross-check every number against `/c/chlu/evidence/outputs/**`.** A number that traces to no file is
   a must-fix, however plausible it reads.
2. ⛔ **Hunt for tables that merge different compute budgets or different decoder recipes.** w8's CPU
   and GPU legs and the `w8_v1_short` recipe are all separate instruments. A merged table is fatal.
3. ⛔ **Test the novelty STOP.** If any contribution sentence reduces to request-indexed factored
   generation, it is TensorFaces (2002) and must be cut.
4. ⭐ **Check the negatives are in the MAIN TEXT.** The route-task failure and the
   compose/forget incompatibility are results. A draft that appendixes them is overclaiming by
   omission — treat it as such.
5. ⛔ **Check every CMAZE-1 composition claim carries its BLIND-SEQUENTIAL column measured on its
   own split** (0.9037–0.9130), not the published 0.4980. A missing caveat column is a must-fix.
6. ⚠ **Attack the deletion claim properly** — it is the paper's lead. Is "exact" defined? Is the
   downdate-vs-refit comparison the right one? Does the byte accounting net the address book in
   anywhere? ⭐ If it survives you, it will survive a reviewer.

## ACCEPTANCE CRITERION
Every must-fix names **the claim, the evidence it fails against, and the smallest change that
fixes it**. ⛔ "Weak" is not a finding; the number that makes it weak is.

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
You return no must-fix items. ⭐ That is not a pass — re-read §1 and §2 and check the arithmetic in
the byte ledgers, because a skeleton assembled this fast has them.
