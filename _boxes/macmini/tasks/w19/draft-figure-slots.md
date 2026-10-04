# w19 spoke/draft-figure-slots — PUT THE FIGURES IN THE PAPER
**Agent type: paper-writer.** Zero compute, zero scored runs. ⭐ PRE-AUTHORISED.
⛔⛔ **GATE: do NOT start until BOTH `task-figures` and `success-failure-panels` have written `END_UTC.txt`.** You slot figures that must already exist.
⛔⛔ **You and `w18/placeholder-fill` both edit the two `.tex` files. You must NEVER run at the same time.** The Hub sequences you; if a `placeholder-fill` session is live, stop and say so.

## The one job
1. **Insert the two task figures BEFORE the results sections** of **BOTH** `DRAFT-v2.tex` and `DRAFT-v2-style.tex`. A reader meets the task before the first number.
2. **Slot the panel figures beside their tables.**
3. **Placeholders where wave 19's rows have not landed** — `\PLACEHOLDER{}`, named, and ⛔ **never presuming a win** (each must say it may land as a negative).
4. **Re-run the style diff-check** and show all four diffs EMPTY: numbers, placeholders, artefact comments, section order. Checker: `~/clu/outputs/w17-style-paraphrase/style_diff_check.py`.

## Hard constraints
- ⛔ **Do not touch the four `\PLACEHOLDER{}` owned by `bar-d-score` and `lace-poe-rows`** — `placeholder-fill`'s second pass owns those.
- ⛔ **Do not change one word of prose.** You insert float environments, captions and labels.
- ⭐ **Both drafts must still compile.** Run `pdflatex`, save the logs. ⛔ Never claim a compile you did not perform.
- Every figure caption names its source artefact in a LaTeX comment.
- ⛔ Wave 19's conditions never share a table or a figure panel with wave 16's rows.

## Acceptance criterion
Both `.tex` updated, both compile (logs saved), four diffs EMPTY, a `FIGURE-SLOTS.md` listing every figure inserted with its location and source, and every wave-19 placeholder named.

## Files you own
`DRAFT-v2.tex`, `DRAFT-v2-style.tex`, `FIGURE-SLOTS.md` — ⛔ and only while no `placeholder-fill` session is running.
