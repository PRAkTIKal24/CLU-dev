# `v5-cite-pass` — wire the generated `refs.bib` into V5's PALM submission

**Agent:** `paper-writer`
**Commissioned by the Shorts Advisor at the Head's direction, 2026-08-26 (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 85).**
**Object:** `~/Desktop/V5_PALM_Submission/paper.tex`
**Report:** `.claude/outputs/v5-cite-pass.md`

---

## ⛔ MECHANICAL PRECONDITION — check this on disk FIRST, before reading anything else

`~/Desktop/V5_PALM_Submission/refs.bib` **must exist.**

✅ **At scoping (2026-08-26) it EXISTS and is verified** — 30 entries, Advisor-repaired and compile-tested against the paper's own preamble: **0 BibTeX warnings · 0 LaTeX errors · 0 undefined citations · 30/30 bibitems.** Re-check it on disk anyway; if it has been replaced since, the checks below still bind.

If it is absent: ⛔ **STOP. Write the report saying the precondition is unmet, edit nothing, and exit.** Do not guess entries, do not hand-build a `.bib`, do not proceed on any other part of the task. (A prior spoke in this program did exactly this and its refusal was the correct behaviour — the cost of an early launch is one wasted pass, not a corrupted artifact.)

---

## 1. What this pass is

`paper.tex` currently carries a **hand-built `\item` bibliography** (31 entries) and cites works in prose as **manual author-year strings** — e.g. `Blelloch \& Golovin's (2007)`, `Snyder (1977)`, `introduced as CHLU by Jawahar \& Pierini (2026)`. There is not one `\cite` command in the file.

**Your job: replace the manual apparatus with real BibTeX citations, and change nothing else.**

⭐ **The reason this matters beyond tidiness:** once every mention is a `\cite`, the printed author, year, title and venue come from `refs.bib` rather than from hand-typed prose — so any stale year or truncated title in the current text **self-corrects from the verified records**. That only works if the conversion is complete. A single surviving manual string is a reference that silently disagrees with the bibliography.

---

## 2. Part A — the conversion

1. **Add the bibliography machinery** at the point where the hand-built block currently sits:
   `\bibliographystyle{plainnat}` + `\bibliography{refs}`.
   **Style: author-year (natbib), which the template explicitly permits** (`neurips_2026.tex` §"Citations within the text": *"Citations may be author/year or numeric, as long as you maintain internal consistency"*). Author-year is chosen deliberately because the current prose is already author-year, so the rendered text barely moves — which is what makes Part C checkable.
2. **Delete the hand-built `\item` reference list in full** — it is superseded. ⛔ Delete the *list*, not the section heading if the heading is what `\bibliography` replaces; make the result build with exactly one References section.
3. **Convert every manual author-year mention to `\citet`/`\citep`/`\citealp`**, choosing by grammar:
   - the citation is the sentence's subject → `\citet{key}` (*"\citet{blelloch2007} establish…"*)
   - the citation is parenthetical support → `\citep{key}`
   - inside an existing parenthesis → `\citealp{key}`
   ⛔ **The surrounding words do not change.** You are swapping a hand-typed name-and-year for the command that produces it, not rewriting the sentence.
4. **Report any bibliography entry that is cited nowhere** and any prose mention with no matching `.bib` key. ⛔ Fix neither — list both.

## 2b. ⛔ FOUR FACTS ABOUT `refs.bib` — read before you match a single key

1. ⛔⛔ **TWO KEYS ARE NAMED AFTER PEOPLE WHO DID NOT WRITE THE PAPER.** Zotero generated them from the LNCS *series-editor* field:
   - **`goos_lower_2003`** = **Buchbinder & Petrank**, *Lower and Upper Bounds on Obtaining History Independence*, CRYPTO 2003.
   - **`hutchison_uniquely_2008`** = **Blelloch, Golovin & Vassilevska**, *Uniquely Represented Data Structures for Computational Geometry*, SWAT 2008.
   The `author` fields inside both entries are **correct**; only the keys mislead. ⚠ A writer grepping the `.bib` for "Buchbinder" or "Vassilevska" **will not find the key** and may wrongly conclude the entry is missing. ⛔ Do not rename the keys (that is churn); just wire them correctly.
2. **Three entries carry an Advisor-corrected publication year** because Zotero imported the arXiv posting year for works this paper cites by venue: `guo_certified_2023` → **2020** (ICML) · `behrouz_titans_2024` → **2025** (NeurIPS) · `zhong_memorybank_2023` → **2024** (AAAI). ⛔ The key strings still contain the old year — **keys are source-only and never printed; do not "fix" them to match.**
3. **Two entries were added from the arXiv API** because the exporter silently dropped them: `bourtoule_machine_2019` (Machine Unlearning) and `sekhari_remember_2021` (Remember What You Want to Forget).
4. ⛔ **ONE PROSE CITATION HAS NO `.bib` ENTRY: Jude, Perich, Miller & Hennig (2023)** — no DOI exists and its arXiv record could not be retrieved. ⛔ **Do not invent a key and do not delete the mention.** Leave that one mention as the manual author-year string it already is, and **report it as the single known exception** to the zero-manual-strings criterion in §7.2.

## 3. Part B — three items that are decided, so you do not have to decide them

1. ⛔ **The `Anonymous (2026)` theory-note entry is DROPPED** unless `refs.bib` contains it (it should not). V5 is self-contained — it ships no theory note and no supplementary companion. ⛔ Do not re-create the entry, do not add a supplementary reference anywhere, and do not add a *"provided in the supplementary material"* note to any entry.
2. ⛔ **There is no `\TODO` tag any more** — the Head had it removed on 2026-08-25 and its work is now a separate commissioned pass (`tasks/v5-derivation-appendix.md`, a physics-theorist writing a lean proofs appendix). ⛔ Do not re-create the tag, and ⛔ do not add, reference or anticipate that appendix here — if it lands first you will see it in the file and cite it only if a `\label` already exists.
3. **The CLU continuity sentence** — *"the Causal Learning Unit (CLU), introduced as CHLU by Jawahar \& Pierini (2026)"* — is **charter-mandated**. Convert its parenthetical to a `\cite`, keep the sentence. ⛔ Never anonymize or remove it: third-person self-citation is the sanctioned double-blind mechanism.

⚠ **One rendering consequence, stated so it is not mistaken for a defect:** under author-year, `\citep` on the anchor preprint renders its author's surname in body text. **That is compliant** — the standing rule governs *phrasing* (we never write the name as our own prose), not the rendering of a citation, and the bibliography legitimately names its authors. ⛔ Do not strip it, and ⚠ note that "Morse"/"Moser" are different words that must survive any sweep.

## 4. Part C — ⛔ THE DIFF CONTRACT (the Head's requirement, and the acceptance test)

**There are to be no edits to the text itself apart from the citations.**

Every changed hunk carries exactly one label:
| label | what it covers |
|---|---|
| `CITATION` | a manual author-year string becomes a `\cite*` command |
| `BIBLIOGRAPHY` | the `\item` list out, `\bibliographystyle`+`\bibliography` in |
| ⛔ `OTHER` | **must be ZERO** |

A hunk that is none of the first two is a violation — ⛔ **do not make it; report it blocked.** Specifically forbidden however tempting: typo fixes, grammar, capitalisation, rewording, re-ordering, whitespace or re-wrapping, terminology harmonisation, and any touch to a number, caption, label, section heading or table.

⭐ *The Head's text is the Head's. A defect you notice goes in the findings list, never into the file.*

**A word-level check is mandatory:** for each `CITATION` hunk print before/after with the citation portion marked, and confirm **the surrounding words are byte-identical**. A conversion that also reflows a sentence is a violation.

## 5. Deliverables

1. `BUILD-NOTE-CITE.md` in the submission folder — **deliverable #1**, written before the PDF is shipped.
2. The labelled diff (hunk counts per class; ⛔ `OTHER` = 0) and the word-level table.
3. The citation map: every `\cite` added, its key, and the sentence it attaches to — as a table, for one-word strike-out by the Head.
4. The uncited-entries list and the unmatched-mentions list.
5. Build evidence: `0 errors · 0 undefined citations · 0 undefined references`, run as `pdflatex → bibtex → pdflatex → pdflatex`, with the true page split reported (main text / references / appendices).
6. Final `md5` of `paper.tex`, and confirmation that `figs/`, `neurips_2026.sty` and `refs.bib` are byte-untouched.

## 6. Boundaries

⭐ **File-chain rule, read this first.** `.claude/NIPSsubmission/v5-palm/pj_sub.tex` is the **canonical file the Head edits**; `~/Desktop/V5_PALM_Submission/paper.tex` is a **build copy** refreshed from it. This pass is a mechanical transformation, not authoring, so it runs on the **build copy** — that is where `refs.bib`, `figs/` and the venue `.sty` live. **The Advisor copies your result back into `pj_sub.tex` on acceptance; ⛔ you never write to `pj_sub.tex` yourself.**
⚠ **Hazard:** if the Head edits `pj_sub.tex` while this pass runs, the copy-back would clobber their work. **Record `pj_sub.tex`'s md5 and mtime at boot and again at the end, and report both** — if it moved, say so loudly in the report's first ten lines and the Advisor will reconcile by hand rather than copy.

- ⛔ **Build only inside `~/Desktop/V5_PALM_Submission/`.** Never build inside `.claude/NIPSsubmission/v5-palm/` or any lineage folder.
- ⛔ `.claude/NIPSsubmission/v5-palm/**` is **byte-untouched** by this pass. `paper.tex` in the Desktop folder is the only file you may write.
- `pdflatex`/`bibtex` are **not on `PATH`** on this machine: use `/Library/TeX/texbin/pdflatex` and `/Library/TeX/texbin/bibtex`.
- ⛔ **Never write to `paper.tex` while the Head is editing it.** Check its mtime at boot and at the end; if it moved mid-pass, stop and report rather than clobbering.
- ⛔ **Zero new numbers, zero new claims.** This pass adds citation commands and a bibliography. Nothing else.
- ⛔ **Page limits are out of scope.** Report the measured split; never cut anything to fit.
- ⚠ **Grep hazard:** directory-level grep over `.claude/` silently returns nothing (gitignored). Sweep per-file, and positive-control every negative before reporting "zero occurrences."

## 7. Acceptance criteria

1. `refs.bib` precondition checked on disk and reported.
2. Manual author-year strings remaining in prose = **0** (printed, with the sweep's positive control).
3. `OTHER` hunks = **0**, with the word-level byte-identity check printed.
4. Build clean: 0 errors, 0 undefined citations, 0 undefined references.
5. The three Part-B items honoured exactly.
6. `BUILD-NOTE-CITE.md` shipped as deliverable #1.

## DIAL DECLARATION
**Dials touched: NONE.** No experiment, no config, no registry, no charter. This pass edits one `.tex` file's citation apparatus and writes one build note and one report.
