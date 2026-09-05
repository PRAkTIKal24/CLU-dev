# pj-fidelity-v2-r2 — doc-curator — ROUND 2 fidelity audit of the Head's rewritten V2 `pj_sub.tex`

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 57, 2026-08-22).** Read `.claude/AGENT_PROTOCOL.md`, then this file. Output: `.claude/outputs/pj-fidelity-v2-r2.md`.

**Context:** the Head has completely rewritten `pj_sub.tex` since the round-1 audit (`outputs/pj-fidelity-v2.md`, 2026-08-22). The file grew from 1,263 words to a full paper (compiles clean; the Advisor rendered `pj_sub.pdf`, **14 pages**, zero repairs). Round 1 found the numbers clean and the claims drifted; this round verifies the rewrite against the same bar AND re-adjudicates every round-1 finding.

**DIAL DECLARATION: none — read-and-report only; zero file edits anywhere.**

## Absolute constraints
- ⛔ **`pj_sub.tex` is EDIT-BARRED (Head ruling, Add.53, still in force). You issue ZERO writes against any file in `.claude/NIPSsubmission/` — your only write is your report.** Advisor-pinned md5 at task issue: `d15de78712d90eb94d2495d4bd9ad948`; the Advisor re-verifies after your pass.
- ⚠ You have NO shell tool (Read/Grep/Glob/Write only). There is no render step in this task — the PDF already exists. If any check seems to need a shell, report the limitation honestly; never fake it.
- Object: `.claude/NIPSsubmission/v2-neurreps/pj_sub.tex`. Source of truth: `.claude/NIPSsubmission/v2-neurreps/submission.tex` (the accepted clean base, Add.52) — and, where the base inherits an approved wording, the registries as cited below.

## Part A — numeric fidelity (the Head's standing question)
Every numeric token in `pj_sub.tex` matched against `submission.tex` for **value, precision, units, ±/CI, seed counts, and scope** (what arm/condition the number belongs to). ⛔ **Any number lacking an ancestor in the source is the most serious finding class available** — quote it with its context and say what the nearest source number is.
⭐ **New in round 2 — the citation ancestry check:** the rewrite adds prose author-year citations (~55 instances) and a manual `\section*{References}` list. For every cited work: does it appear in the clean base's bibliography (ancestor = verified at Add.23/Add.52) or is it **NEW**? List every NEW citation (author, year, claimed venue/ID) — new citations are UNVERIFIED records and the list feeds a cite-check spoke. Do not attempt to verify them yourself (no web access); ancestry only. Also list any base bibliography entry that was DROPPED while its in-text claim survives.

## Part B — claims fidelity + the round-1 re-adjudication
1. **Re-adjudicate every round-1 V2 finding by direct quotation (new text beside base text): FIXED / PARTIALLY FIXED / UNFIXED**, specifically: (a) the four claim-strength widenings (incl. the anchored-survival sentence, round 1's "most quantitatively wrong sentence"); (b) the missing prior-art disclaimer — does the anchor still read as ours, or is the credit restored? (c) zero citations (closed in form — confirm no claim-bearing assertion remains uncited that the base cites); (d) zero figures (5 `\includegraphics` now present — confirm each figure's caption matches the base's claim for that figure, and note which base figures remain unused).
2. **The do-not-cut walk (same list as round 1):** approved wordings verbatim where present · the N46 designed-only rider · C-5 scale qualifiers · the CM-16a/b split (friction preserves / temperature erases — never merged) · the CM-21 retirements · kinetic-mode fine print · the §A20.5 substrate-scope sentence · the N1 novelty scoping (⛔ no sentence may claim the zero-mode ⇒ pseudo-gap ⇒ lifetime chain or use "pseudo-gap" as ours; our claim is the two-branch closed-form law + crossover + floor on a trained potential, per Add.49 Ruling 1) · the no-biological-claim sentence (exactly once). For each: **present / absent — and where absent, which claim now stands unqualified**, ranked by consequence.
3. **Claims table:** every surviving substantive claim quoted side by side with its base form, ruled **IDENTICAL / NARROWER (safe) / WIDER (⛔) / CHANGED IN KIND**. Drift modes to watch (round-1 pattern): a tie becoming a win · a bound becoming a point estimate · a designed-arm result reading as general · a correlation reading as a law.
4. **Mechanical inventory:** approximate main-text/references/appendix page split (from the PDF's section boundaries), figure count vs base, References entry count vs base bibliography count.

## Acceptance criteria
- Every Part-A mismatch and Part-B absence quoted with line context, ⛔ flagged never fixed.
- The round-1 re-adjudication table complete (every round-1 finding has a verdict).
- The NEW-citation list complete (or explicitly empty).
- Zero writes outside `.claude/outputs/pj-fidelity-v2-r2.md`.
