> ## ⛔ STANDING RULES FOR THIS BOX — read before the job below
> - ⭐ **The evidence base is `/c/chlu/evidence/outputs/`** (live, shipped 2026-09-21).
>   ⛔⛔ **DO NOT read `$CHLU_CLAUDE_ROOT`** — it points at a FROZEN, OLDER snapshot of the same
>   tree, and every number in it is out of date. Reading it silently produces a stale paper.
> - ⭐ **You write ONLY to `/c/chlu/outputs/w10-physics-section-evidence/`.** ⛔ No other spoke's tree. ⛔ Never
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

# w10 spoke/physics-section-evidence — assemble the extended section's evidence base

- **Agent:** `physics-theorist`.
- **Output:** `/c/chlu/outputs/w10-physics-section/EVIDENCE.md`.
- **Owns:** `/c/chlu/outputs/w10-physics-section/**`. ⛔ No `chlu/` production code.
- **Scored runs:** ⛔ none. Small numerical sanity checks only.

## THE ONE JOB
Assemble the **evidence base** for the paper's extended physics section: the shorts' validations
plus **w6's CRR result**, and draft the **"a benchmark is needed"** statement from the w5/w6
negatives.

## THE SPINE — verified present before this spoke was scoped
⭐ **The Curvature–Retention Relationship predicts the field store's settle half-life,
parameter-free, to a factor 1.10 at the median held-out corner**, and beats a one-parameter rival.
⭐ **The sharpest number (D17):** derived overdamped branch slope **+1.1385**, measured **+1.1368**.
⭐ This is the programme's cleanest theory-meets-measurement result.

## ⛔ THE SCOPE LIMIT THAT MUST SURVIVE INTO THE PROSE
⚠ **CRR predicts a settle half-life, NOT a taxonomy row.** It supports **no scoreboard claim**.
⛔ A section that lets a dynamics result imply a benchmark result is the failure this spoke exists
to prevent. Say what it predicts and stop.
⚠ **B4 missed on all three seeds** — carry that, do not quietly drop it.

## THE BENCHMARK STATEMENT
Draft from the **w5/w6 negatives**: what the programme measured, what no existing benchmark scores,
and what a benchmark would have to contain to make the claim falsifiable. ⭐ Write it as a
**requirement list**, not as an argument that we deserve one.

## ACCEPTANCE CRITERION
Every physical claim carries **its derivation and the measurement that confirms or refutes it**,
and each is marked **CONFIRMED / REFUTED / UNTESTED**. ⭐ An untested claim is welcome and labelled.

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
- A derivation you cannot reproduce numerically ⇒ mark it REFUTED and say which step fails.
- The half-life law failing to reproduce from the recorded artefacts ⇒ that is the finding, and it
  outranks the section.
