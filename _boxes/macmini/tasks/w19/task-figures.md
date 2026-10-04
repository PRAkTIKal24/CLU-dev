# w19 spoke/task-figures — WHAT THE TASKS ARE, DRAWN
**Agent type: results-analyst. ⭐⭐ LOAD THE `dataviz` SKILL FIRST, before any chart code.** Zero compute, zero scored runs.
⭐ PRE-AUTHORISED. ⛔ Never wait for the Hub, the Head or the Advisor.

## The one job
**Two task-definition figures, placed BEFORE any result in the draft.** A reader must understand what is being asked before they see a number.

### (a) Shapes
The **six factors with example values** · **one example request and its rendered answer** · **what a 2-way and an n-way hold-out means** — i.e. **which cells are missing from the fit pool**, drawn so the omission is visible.

### (b) CMAZE-T — one annotated grid
Agent · pocket · keys · **closed/open doors by colour** · tinted floor · goal · budget.
Plus **the two rules and the door-then-tint interaction drawn as three tiny before/after panels**: **legal** / **illegal** / **legal-because-the-key-was-spent**.
Plus **the five request words and the four-fact answer**.

## Requirements
- **To-the-point titles.** ⛔ **No in-figure prose.**
- ⭐ **Consistent names with every other figure in the paper** — an arm, a factor or a word has one name and one colour everywhere. Check against `w17-paper/figures/` and `w18-seeds-and-tables/TABLES.md` before inventing a label.
- Vector **PDF + PNG**.
- ⭐ **Byte-identical on rebuild** — pin seeds, sort inputs, pin the matplotlib backend and font. Build twice, `shasum` both, show the match.
- Colour-blind-safe palette, consistent with the existing figures.

## Acceptance criterion
Both figures as PDF+PNG in `~/clu/outputs/w17-paper/figures/`, a `TASK-FIGURES.md` naming every element and the artefact or engine source it came from, and the rebuild-determinism proof.

## What would prove this wrong
⭐ If the CMAZE-T rules cannot be drawn so that **illegal** is visibly different from **legal-but-slower**, say so — that is the two-gym law failing to be legible, and it is information about the task, not a drawing problem. ⛔ Do not fudge the panel to look clean.

## Files you own
`figures/fig_task_shapes.*`, `figures/fig_task_cmazet.*`, `TASK-FIGURES.md`. ⛔ Nothing else — `success-failure-panels` owns `figures/rendered/`, `draft-figure-slots` owns the `.tex` files.
