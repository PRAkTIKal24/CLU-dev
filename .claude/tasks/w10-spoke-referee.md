# w10 spoke/referee — adversarial review of DRAFT-v0 before it reaches the Head

- **Agent:** `paper-referee`.
- **Output:** `.claude/outputs/w10-referee/REFEREE-v0.md`.
- **Owns:** `.claude/outputs/w10-referee/**`. ⛔ Read-only on the draft and the repo.
- **GATED — mechanical precondition:** ⛔ do not start until
  `.claude/outputs/w10-draft-v0/DRAFT-v0.md` exists.

## THE ONE JOB
Referee the skeleton as the hostile-but-fair composite of the reviewers we actually draw at ICLR.
Itemised report, **must-fix / should-fix / nice-to-have**, plus a simulated accept/reject verdict.

## ⭐ WHERE TO PUSH HARDEST — this programme's actual failure modes, not generic ones
1. ⛔ **Cross-check every number against `.claude/outputs/**`.** A number that traces to no file is
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
