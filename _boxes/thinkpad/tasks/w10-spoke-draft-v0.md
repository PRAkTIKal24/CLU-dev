> ## ⛔ STANDING RULES FOR THIS BOX — read before the job below
> - ⭐ **The evidence base is `/c/chlu/evidence/outputs/`** (live, shipped 2026-09-21).
>   ⛔⛔ **DO NOT read `$CHLU_CLAUDE_ROOT`** — it points at a FROZEN, OLDER snapshot of the same
>   tree, and every number in it is out of date. Reading it silently produces a stale paper.
> - ⭐ **You write ONLY to `/c/chlu/outputs/w10-draft-v0/`.** ⛔ No other spoke's tree. ⛔ Never
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

# w10 spoke/draft-v0 — the ICLR long skeleton, from the evidence base only

- **Agent:** `paper-writer`.
- **Output:** `/c/chlu/outputs/w10-draft-v0/DRAFT-v0.md`.
- **Owns:** `/c/chlu/outputs/w10-draft-v0/**`. ⛔ Writes no other spoke's tree, ⛔ no `chlu/` code.
- **GATED — mechanical precondition, not a sentence.** ⛔ Do not start until **all three** exist:
  `/c/chlu/outputs/w10-seeds-and-tables/TABLES.json`,
  `/c/chlu/outputs/w10-physics-section/EVIDENCE.md`, and the updated negatives registry.
  ⭐ If one is missing, **say so and stop** — do not substitute your own numbers.

## THE ONE JOB
A **skeleton** of the ICLR long: section structure, every claim placed, every table and figure
slotted with its source. ⛔ **The Head writes the prose.** Draft enough that the Head is editing
structure and evidence placement, not inventing it.

## ⭐ THE FRAMING — decided, and not this spoke's to relitigate
- ⭐ **LEAD WITH: exact deletion of a compositional generative model.** It is the strongest claim
  the programme owns, it is confirmed twice over on two environments, and it is unclaimed.
- ⭐ The sentence the paper should carry: **"Downdate the statistic, not the truncated factors."**
- ⛔ **NEVER claim** *"multilinear/bilinear factored generation with a request-indexed tensor head"*
  — Tenenbaum & Freeman (2000) and TensorFaces (2002) own it.
- ⭐ Read **both** `/c/chlu/evidence/outputs/_thinkpad/w8/w8-novelty/NOVELTY.md` **and its `LOG.md`**; the LOG
  carries corrections that give ground back and the file alone reads too pessimistic.

## ⛔ THE NEGATIVES ARE MAIN TEXT, NOT AN APPENDIX TO HIDE IN
The composition result **does not transfer to a route task**, and the isolated reason is that the
read composes latents where the task composes plans. ⭐ That belongs in the paper as a result, with
the plan-space **1.0000** against the latent-space **0.13**. ⛔ A draft that buries it is wrong.

## APPENDIX-MAXIMALISM
Main results in main text; corollaries, negatives, extra plots in appendices. ⛔ **Nothing is
omitted before the dedicated pruning passes.**

## ACCEPTANCE CRITERION
Every claim in the skeleton carries **a pointer to the table row or artefact that supports it**.
⭐ A claim with no pointer is deleted by this spoke, not flagged for later.

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
The evidence base does not support a contribution paragraph that survives the novelty STOP ⇒
report that plainly; it is a finding about the programme, not a writing problem to style around.
