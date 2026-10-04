# w18 spoke/referee-fixes — APPLY THE MECHANICAL MUST-FIXES FROM REFEREE-v2
**Read `COMMON.md` first.** Slug `referee-fixes`. Agent type: paper-writer. **ZERO compute, ZERO scored runs.**
⭐ **Scoped by the Hub 2026-09-23.** `REFEREE-v2.md` returned **3/10 reject** with 7 must-fix items, and the wave-18 plan assigns nobody to apply them — `placeholder-fill` fills placeholders and `referee-4` re-reviews. This spoke closes that gap.

## The one job
Apply **M2, M4, M5, M6 and M7** from `~/clu/outputs/w17-paper/REFEREE-v2.md` to **BOTH** `DRAFT-v2.tex` and `DRAFT-v2-style.tex`, using each item's own "Minimal fix" text.

- **M2** — two rival sets sit in main-text tables an order of magnitude apart on identical corners, unexplained. Add the §3 paragraph naming both sets, their fit pools, schedules and sources (`w12-deletion-baselines` vs `w12-composition-ladder`/spoke-A); state which is the stronger set on the shared corners; report the store's margin against the **strongest** available rival. ⛔ If the gap cannot be explained from artefacts, write a `\TODO` saying so — ⛔ do **not** lead with Table 1's +39 to +49.
- **M4** — the positioning sentence is contradicted by work neither cited nor distinguished. Cite and distinguish, or weaken the sentence. ⭐ Prefer cite-and-distinguish; `CITATIONS.md` and `references.bib` already hold 96 entries.
- **M5** — environments share tables, against the paper's own rule. Split them. ⛔ CMAZE-1 / CMAZE-2 / CMAZE-2b / CMAZE-T never appear in one table.
- **M6** — tables missing the seed-count column. Add it everywhere the paper claims it exists.
- **M7** — venue compliance per `VENUE.md` §§0, 1, 4, 7. ⚠ Each item alone risks desk rejection.

## ⛔⛔ WHAT YOU MUST NOT TOUCH
- ⛔ **M1 and M3 are HELD for the Head and the Advisor.** They change what the paper CLAIMS — M1 is whether "naive vs naive" survives being parameter-only, M3 is whether exactness is rescoped off the address book. ⛔ **Do not apply them, do not pre-empt them, do not soften them.** Leave them exactly as they are and list them as HELD in your `FINDINGS.md`.
- ⛔ **No w15 or w16 number enters either draft.** The Head's Amendment 1 still binds. Every w15/w16 row stays a `\PLACEHOLDER{}`.
- ⛔ **Do not touch any `\PLACEHOLDER{}`** — `placeholder-fill` owns those.

## The invariants you must preserve (they were proven; do not break them)
⭐ After your edits, re-run the style diff-check between the two drafts and show **all four diffs still EMPTY**: numbers, placeholders, artefact comments (265, in identical order), section order. The checker exists at `~/clu/outputs/w17-style-paraphrase/style_diff_check.py`.
⭐ **Both drafts must still compile.** Run `pdflatex` and save the logs. ⛔ Do not claim a compile you did not perform.
⭐ Any number you add carries a LaTeX comment naming its artefact under `~/clu/evidence/outputs/`.

## Acceptance criterion
`REFEREE-FIXES.md`: M2, M4, M5, M6, M7 each with **before → after**, file and location; the four empty diffs shown; both compile logs; and M1/M3 listed as **HELD, not applied**, with one line each on why. Any item you could not land is **DECLARED NOT-DONE** with its reason.

## What would prove this wrong
⭐ If applying M5 or M6 makes a claim look weaker, ⛔ **do not soften it and do not revert** — that is the honest number becoming visible, which is the entire point of the referee pass. Report it.

## Files you own
`DRAFT-v2.tex`, `DRAFT-v2-style.tex`, `REFEREE-FIXES.md`. ⛔ Sequential with `placeholder-fill` — you finish before it starts; you must never both be running.
