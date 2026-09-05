# v2-cite-pass — paper-writer — wire `refs.bib` into V2 and fill every open citation site

**Commissioned by the Shorts Advisor at the Head's direction, 2026-08-24 (charter Addendum 69).** Read `.claude/AGENT_PROTOCOL.md`, then this file. Output report: `.claude/outputs/v2-cite-pass.md`; build note `BUILD-NOTE.md` **inside the submission folder** = deliverable #1.

**Working directory — a NEW clean submission folder, outside the repo:** `~/Desktop/V2_NeurReps_Submission/`
- `paper.tex` — the object (a copy of the Head's `pj_sub.tex`; ⛔ **the original in `.claude/NIPSsubmission/v2-neurreps/` is READ-ONLY for you and must stay byte-identical**)
- `refs.bib` — 53 entries, Advisor-verified, compile-clean · `figs/` — 5 PNGs · `jmlr.cls`, `jmlrutils.sty` — the NeurReps 2026 template · `pmlr-sample.tex`, `INSTRUCTIONS.txt` — venue reference

**DIAL DECLARATION: none — citation wiring. ⛔ Zero new claims, zero new numbers, zero changes to any scientific sentence beyond replacing a manual author-year string with its `\cite` command.**

## Part A — the citation conversion (the Head's instruction: *"tag the appropriate papers at the correct locations using `~\cite{}`"*)
1. **Convert every manual author-year citation in the prose to a natbib command.** The paper currently names works in text (e.g. *"(Dinc et al.\ 2025)"*, *"Renart, Song \& Wang 2003"*). Each becomes `~\citep{key}` (parenthetical) or `\citet{key}` (textual, when the name is the sentence's subject). ⭐ **This is what makes the stale metadata self-correct** — years, titles and venues then come from `refs.bib`, so e.g. the literal *"Dinc et al.\ 2025"* becomes the bib's 2026 PRX record automatically. **Every manual author-year string must be gone; a full inventory is an acceptance criterion.**
2. **Delete the hand-built `\section*{References}` block** (l.118–222 at last reading) and replace it with `\bibliography{refs}` + the class's bibliography style. ⛔ Verify no entry's content is lost that the `.bib` does not carry.
3. **Fill the seven open sites** (Head-ruled; keys verified present in `refs.bib`):

| site | ruling | key(s) |
|---|---|---|
| l.33 `\cite{}` + `\TODO{cite welling and another SSB paper}` | the Welling SSB/Goldstone work | `iqbal_spontaneous_2026` |
| l.33 `\TODO{cite numbu-goldstone…}` | ⭐ **both, comma-separated** | `watanabe_counting_2020,minami_spontaneous_2018` |
| l.33 `\cite{}` + `\TODO{relevant citations}` | ⛔ **no new work** — the orbit-neutrality trio already used at l.55, plus the anchor (per Add.49 this existence claim **is** its theorem, and attributing it only in an appendix reads as under-attribution) | `golubitsky_singularities_1988,krupa_bifurcations_1990,rumberger_lyapunov_2001,mo_symmetry-protected_2026` |
| l.35 `\TODO{cite CHLU}` | the self-citation, third person | `jawahar_chlu_2026` |
| l.48 `\TODO{cite}` · l.86 `\cite{}` · l.96 `\cite{}` | all three are the same work | `mo_symmetry-protected_2026` |

4. **Attach the citations that currently sit in the bibliography unused**, at the sentence homes identified in `.claude/outputs/v2-bib-doi-list.md` §Deliverable 4(a) — ⭐ two are **restorations** of citations `submission.tex` carried and the condensation dropped (`hairer_geometric_2006` at the leapfrog/`h<2` stability band; `huang_measuring_2025` at the Discussion's own *"solution degeneracy between the designed and emergent arms"* sentence). Also cite **`gell-mann_behavior_1968`** at the first GMOR use in Appendix G — the relation is currently named six times with no citation at all (Head-ruled: add it). ⚠ **Report every attachment with its site in a table so the Head can strike any one.** ⛔ If a work has no honest home, leave it uncited — under BibTeX an uncited entry simply does not print, so nothing decorative survives.
5. **Delete the `\TODO` tags** as they are discharged. ⚠ **The two figure TODOs (l.81, l.95) are ALREADY DONE** by `v2-figure-text-pass` — delete both tags, and ⛔ **do not act on the literal text of the l.95 tag: its "change to the citation number" instruction was overruled by the Head in favour of the descriptive label the figure now carries.**

## Part B — the template port ⚠ **STRIKE THIS PART IF THE HEAD WANTS TO DO IT PERSONALLY**
Port `paper.tex` onto the venue class so the folder builds as the venue receives it:
- `\documentclass[mlabstract,onecolumn]{jmlr}` — ⭐ **`mlabstract` is the NON-ARCHIVAL extended-abstract track**, which is the Head's ruled venue; ⛔ **never `mlmain`, which is the archival proceedings track and is barred while the ICLR long is under review.** (Both options are documented in `INSTRUCTIONS.txt`.)
- Follow `pmlr-sample.tex` for the class's title/author/abstract macros; the author block stays **anonymous** (double-blind).
- ⛔ **Do not re-add packages the class already loads** (amsmath, amssymb, natbib, graphicx, url, algorithm2e) — `INSTRUCTIONS.txt` warns this breaks camera-ready generation.
- ⚠ **Citation style is the class's own natbib author-year**; that is the venue's house style and is correct here. ⛔ Do not force a numeric style.
- ⚠ **A `\cite` inside the l.86 subsection heading is poor practice** (it propagates into the TOC, bookmarks and running heads). Flag it to the Head with a suggested rephrasing that keeps the citation in the body; ⛔ **do not rewrite the heading's claim yourself.**
- Report the page split; ⛔ **page limits are deferred by standing Head ruling — report, never optimize, and never cut content to fit.**

## Part C — ⛔⛔ THE DIFF CONTRACT (Head requirement, 2026-08-24: *"there will be no edits to the text itself apart from the citations"*)

**Baseline:** `~/Desktop/pj_sub_BASELINE_19h01_07acd432.tex` — md5 `07acd432d56fdc87575516ee9f074772`, byte-identical to `paper.tex` as you receive it. ⛔ **Never modify the baseline.** (The Head's own `~/Desktop/pj_sub.tex copy` is an EARLIER 17:50 state and is **not** the reference; leave it untouched.)

**Deliverable: a line-by-line classified diff** (`diff -u` baseline vs final `paper.tex`), in `BUILD-NOTE.md`, where **every single changed hunk carries exactly one label**:
- **(1) CITATION** — a manual author-year string replaced by `\citep`/`\citet`, or an open site filled.
- **(2) TODO-DELETION** — a discharged `\TODO{...}` tag removed.
- **(3) BIBLIOGRAPHY** — the hand-built `\section*{References}` block replaced by `\bibliography{refs}`.
- **(4) TEMPLATE** — preamble / `\documentclass` / title-block changes, **Part B only**.
- **(5) ⛔ OTHER — MUST BE ZERO.** Any hunk that is not (1)–(4) is a violation of the Head's instruction. If you believe one is unavoidable, ⛔ **do not make it** — report it as a blocked item with the reason, and leave the text as the Head wrote it.

⚠ **Specifically forbidden, however tempting:** fixing a typo, correcting grammar, adjusting capitalisation, rewording for clarity, re-ordering a sentence, changing whitespace or line-wrapping, "harmonising" terminology, or touching any number, caption, label or heading. ⭐ **The Head's text is the Head's.** A defect you notice goes in a findings list at the end of the report, never into the file.

⚠ **Word-level check, mandatory:** for every (1)-labelled hunk, print the before/after with the citation portion marked, and confirm that **the surrounding words are byte-identical**. A citation conversion that also reflows a sentence is class (5).

## ⛔ THE THEORY NOTE IS OUT (Head ruling, 2026-08-24)
**No supplementary companion is submitted with V2.** The Head: *the short is self-consistent and already carries enough for a workshop paper; if referees ask for specific proofs, those individual proofs move into the appendix later.* Advisor-verified on disk: the Head's own last edit removed the only in-text reference, and **nothing dangles** — the surviving `Sylvester` mention is Sylvester's law of inertia (a classical result named in passing), and every `thm` hit is the `amsthm` package or a substring, not a reference to an external document.
⇒ **`anonymous_theory_2026` has been REMOVED from `refs.bib` (now 52 entries).** The hand-built list's line-126 "Anonymous (2026)" entry disappears with the References block in Part A step 2 — ⛔ **do not re-create it, do not add a supplementary reference anywhere, and do not add a "provided in the supplementary material" note to any entry.**

## Constraints
- ⛔ **No scientific sentence is reworded** except where a manual author-year string is replaced by its citation command. Approved wordings and mandatory riders stay **verbatim** (Add.30 boundary).
- ⛔ The author token appears in **no** body text, caption, label or filename — only as a citation and in the bibliography (Add.45/51). ⚠ Under author-year rendering a `\citep` legitimately prints the name: that **is** the sanctioned citation form (Add.45's own example), not a violation.
- The `.claude/NIPSsubmission/v2-neurreps/` originals stay byte-identical (md5 them before and after).

## After editing
Build with `/Library/TeX/texbin/pdflatex → bibtex → pdflatex ×2` **inside the submission folder**; report **0 errors, 0 undefined citations, 0 undefined references**; print the count of `\citep`/`\citet` commands, the count of bibliography entries actually typeset, and the remaining-manual-author-year count (**must be 0**). `BUILD-NOTE.md` carries: the per-site citation table, the attachment table, the TODO-deletion list, the page split, and anything you could not do.

## Acceptance criteria
Every open site filled with its ruled key; zero manual author-year strings left; zero `\TODO` tags left; the bibliography prints from `refs.bib`; build clean with zero undefined citations; originals byte-identical; every attachment reported for Head strike-out.
