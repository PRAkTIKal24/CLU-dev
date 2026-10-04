# w4 hub/read — TARGET: rows 1, 3, 4 on factor-aligned wells, with a read that uses the content

**Boot:** `autonomous-hub-charter.md` → `w4-hub-common.md` → this file. **Base: `base/iter4 = 9dfed5e`.**
**Worktree:** `/c/chlu/worktrees/CHLU-w4-read` · **LOG:** `/c/chlu/outputs/hub-read/LOG.md`
**Scored budget: target `read`, 3 runs.** ⛔ Gated on `RIVALS-READY.json` (w4-common §2). DEV starts now.

## DIAL DECLARATION
**Dial: the READ** — how a well's content becomes an answer. ⛔ Not the partition (that is `hub/placement`),
⛔ not what is stored (retired). Controls on every row: landscape-removed, landscape-shuffled, and the
correlation-split + linear-head ablation. **Falsified if** the new read ties the full rival set on row 4.

## The target
**Rows 1, 3 and 4 on factor-aligned wells, beating the FULL RIVAL SET.**
**Bar: beats the MLP and the table on row 4, and does not lose row 3, in expectation over ≥ 49 corners.**
⚠ Wave 3 beat `basis_head` — an **ablation** under w4-common §1. ⭐ **Nobody has beaten a trained
generic learner. That is the bar now.**

## ⭐⭐ The mechanism to try first, and why it is not a guess
**Address inside each well by the per-value average of that well's content.** This is the mechanism that
already reads `orientation` at **0.9320** where the current address-book read gets **0.1828** — from
**identical content**. The information is present and the read is discarding it. ⭐ **Then extend to a
continuous per-well coordinate**, which is what row 2 will need and what `hub/unseen-valid` is building
in parallel.

## Report both, always
The per-value-mean read is a **specialist's** mechanism; folding it into an addressed store is the
contribution. ⭐ **Report the plain per-value-mean table beside your arm on every row** — if your arm
does not beat it, the store is adding nothing and you say so in those words.

## ⛔ What would prove this hub wrong
- Your read ties the full rival set on row 4 ⇒ decoration; report it as such.
- Your read beats `basis_head` but loses to the MLP or the transformer ⇒ ⛔ **that is not a result**,
  it is an ablation, and it is labelled so.
- The per-value-mean table alone matches your arm ⇒ the store is not earning its bytes.
- The gain vanishes on generation-2 seeds 1–2 ⇒ it was a seed-0 artefact.
- Row 3 regresses while row 4 improves ⇒ report the trade; ⛔ do not quote row 4 alone.
