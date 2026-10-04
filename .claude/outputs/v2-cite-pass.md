# v2-cite-pass — paper-writer report

Task + acceptance criterion: wire `refs.bib` into V2, fill every open citation site, zero manual author-year strings, zero `\TODO`, bibliography prints from the `.bib`, clean build, originals byte-identical, every attachment reported for strike-out.
Status: **done** — with **one build error I was contractually forbidden to fix** (§B1 below), and **three orphan citations deliberately NOT attached** for a substantive reason (§4).

**DIAL DECLARATION (echoed): none — citation wiring.** No dial, no laundering control, no falsification axis. ⛔ Zero new claims, zero new numbers, zero reworded scientific sentences.

> ## ⚠ DOWNSTREAM RECONCILIATION LIST (needs an owner — first-10-lines rule, AGENT_PROTOCOL §5)
> 1. ⛔⛔ **ONE BUILD ERROR, ONE DELETION, HEAD-OWNED.** App. G's first negative-results table has a **4-cell row in a 3-column tabular** (`… \sqrt{M_iT\gamma(2-\gamma)}$. & --- \\[2pt]`), introduced by the Head's own 19:01 edit (the 08-22 accepted file is correct). LaTeX raises `! Extra alignment tab has been changed to \cr` and drops the `---`. ⛔ Part C class (5) forbade me to touch it. **Proof it is the only error: an identical scratch build with that one `&` removed gives 0 errors / 0 undefined / 23 pp.** Owner: Head, one word.
> 2. ⛔ **Three orphans NOT attached, and the reason must not be lost.** `gardner_toroidal_2022` / `khona_attractor_2022` / `kim_ring_2017` were orphaned **by the Head's 19:01 edit**, which deleted the §2 clause *"and measurable toroidal codes in biological systems (Kim…; Gardner…; Khona & Fiete…)"* **together with its no-biological-claims disclaimer**. Re-attaching them would re-import a biological citation into a paper that just dropped that disclaimer — a positioning change wearing a citation's clothes. Left uncited (they do not print). Owner: Head; two concrete restore options in `BUILD-NOTE.md` §3.
> 3. ⚠ **`refs.bib` changed underneath this pass, at 19:17, mid-build**: 53 → **52** entries (md5 `01740ec3…` → `01d586ec…`); the removed entry is `@misc{anonymous_theory_2026}`. Consistent with the Head's 19:01 deletion of the prose that referenced it. My build used the 52-entry file. The task file's "53 entries" is therefore stale, not a discrepancy in my counts. Owner: Hub, one-line ledger note.
> 4. ⚠ **Five citation years now render as the arXiv posting year rather than the proceedings year** (Ságodi 2024→2025, Xu 2023→2022, Vastola 2024→2025, Wang & Ponce 2023→2022, Behrouz 2025→2024) because those `.bib` records are `@misc` arXiv entries. ⭐ The Dinc self-correction the pass was built for **did fire**: 2025 preprint → **2026 PRX 16(2):021058**. Owner: whoever owns `refs.bib` — one `year = {…}` line each if proceedings years are wanted. ⛔ I made zero writes to `refs.bib`.
> 5. ⚠ **`bernardo_shaping_2025` prints "Bernardo et al."**, not "Di Bernardo et al." — the `.bib` author field is `{Bernardo, Arianna Di}`. One-line fix, not made by me.

---

## What I did
- **Filled all seven Head-ruled open sites** with the ruled keys (Iqbal · Watanabe+Minami · the orbit-neutrality trio + Mo · Jawahar & Pierini · Mo ×3).
- **Converted all 33 manual author-year prose strings** to `\citep`/`\citet`/`\citealp`. **Zero remain.**
- **Attached 5 previously-unused entries** at the scout's identified homes (`hairer_geometric_2006`, `huang_measuring_2025` — the two restorations; `gell-mann_behavior_1968` — Head-ruled; `jelassi_repeat_2024`; `karuvally_exponential_2025`+`ramsauer_hopfield_2020`). All five tabulated with a strike-risk rating in `BUILD-NOTE.md` §3.
- **Left 4 entries uncited** (`rusch_unicornn_2021` + the three biology orphans) — they simply do not print.
- **Deleted all 7 `\TODO` tags.** ⛔ Did **not** act on the l.95 tag's literal "citation number" text (Head-overruled; the class is author–year, there is no number).
- **Deleted the hand-built `\section*{References}` block** (baseline l.118–222, 50 entries, archived at `.claude/scratch/v2-cite-pass/refblock_removed.txt`) → `\bibliography{refs}` with the class's own `plainnat`.
- **Part B: ported to the venue class** — `\documentclass[mlabstract,onecolumn]{jmlr}` (⛔ never `mlmain`), jmlr title/boilerplate macros, author block omitted for double-blind, and the six class-loaded packages (amsmath, amssymb, natbib, graphicx, url, algorithm2e) removed from the preamble per `INSTRUCTIONS.txt`. Kept only `fontenc`, `booktabs`, `bm`, `microtype`.
- Wrote `BUILD-NOTE.md` **inside the submission folder** — per-site citation table, attachment table with strike risks, TODO-deletion list, classified diff, page split, and everything I could not do.

## How I verified
```
md5 pj_sub.tex (read-only original)  07acd432d56fdc87575516ee9f074772  BEFORE and AFTER  → byte-identical
md5 baseline  pj_sub_BASELINE_19h01  07acd432d56fdc87575516ee9f074772  unmodified
md5 paper.tex (final)                e0e8fe7495bfdd8b9e73829db43abf65
pdflatex → bibtex → pdflatex ×2 (+1) inside ~/Desktop/V2_NeurReps_Submission/
  errors:               1   (the Head's extra alignment tab, paper.log l.809 — §B1)
  undefined citations:  0
  undefined references: 0
  bibtex warnings:      0   (paper.blg: "warning$ -- 0")
  hyperref warnings:    9   (all from the \citet in the §4.2 subsection heading)
  pages:               23
scratch diagnostic build, identical except the one stray "&" removed:
  errors: 0 · undefined: 0 · pages: 23   (.claude/scratch/v2-cite-pass/diagbuild/)
\citep 37 · \citet 6 · \citealp 1 · bare \cite{} 0 · \TODO 0
bibliography entries typeset: 48 of 52 in refs.bib
manual author-year strings remaining: 0
  - grep 'et al'                          → 0 hits
  - grep -E '[A-Z][a-z]+ \& [A-Z][a-z]+'  → 0 hits
  - grep -E '(19|20)[0-9]{2}'             → every hit is inside a \cite key or \jmlryear{2026}
  - surname sweep over all 40+ cited authors, excluding lines with 'cite' → 0 hits
  - standalone author-token sweep (Add.45/51), "Morse"/"Moser" excluded  → 0 hits
```
Rendered forms confirmed from `pdftotext` of the built PDF, not from the source: *"(Iqbal et al., 2026)"*, *"(Watanabe, 2020; Minami and Hidaka, 2018)"*, *"(Golubitsky et al., 1988; Krupa, 1990; Rumberger, 2001; Mo, 2026)"*, *"similar to Jawahar and Pierini (2026)"*, *"in Mo (2026)"*, *"(Dinc et al., 2026)"*, *"(Hairer et al., 2006)"*, *"(Huang et al., 2025)"*, *"(Gell-Mann et al., 1968)"*.

## Findings/results

**Page split** (⛔ reported, never optimised): main text **pp. 1–8** · References **pp. 8–12** · Appendices A–I **pp. 12–23** · total **23 pp**.
⚠ **15 pp → 23 pp for identical text is the venue class, not content**: `jmlr` sets 10.95 pt on a 6.0 in measure where `neurips_2025_ml4ps` set 10 pt on a wider block. Nothing was added, removed or compressed.

**Diff contract (Part C): 8 hunks, and class (5) OTHER = ZERO.** Every changed line is (1) CITATION, (2) TODO-DELETION, (3) BIBLIOGRAPHY or (4) TEMPLATE. The mandatory word-level check ran as `git diff --word-diff` over baseline→final: in all 44 citation sites the surrounding words are byte-identical; the only non-citation additions are the preamble block and the `\bibliography{refs}` line. Artifacts: `.claude/scratch/v2-cite-pass/full.diff`, `word.diff`.

**⛔ Four defects in the Head's text found and NOT fixed** (class 5 forbids it): (B1) the extra alignment tab — the one build error; (B2) `Fig.` + a stray backtick, and the reference points at `fig:gmor` while the figure below it is `fig:pricelist`; (B3) `isrelationship` in Contribution 1; (B4) `magnitutde` ×6. All four listed in `BUILD-NOTE.md` §7-B.

**The `\cite`-in-a-heading residual is real and measured.** 9 × `Token not allowed in a PDF string` from the `\citet` in the §4.2 subsection heading; it also propagates to the running head and PDF outline. ⛔ I did not rewrite the heading. Two options given to the Head in `BUILD-NOTE.md` §7-A1 — a **zero-text-change** fix (a TOC-only short title, printed heading byte-identical, kills all 9 warnings) and a rephrasing that moves the citation into the body.

**Anonymity holds.** No `\author` block; standalone author-token sweep clean with "Morse"/"Moser" correctly excluded; all names appear only as rendered citations or in the bibliography (Add.45/51). ⚠ The venue class restores the banner *"Under Review - Extended Abstract Track"* + *"Symmetry and Geometry in Neural Representations"* — the old hand-written venue suppression is gone by design; if a venue-neutral artifact is still wanted for another purpose it is now a separate build.

**Charter compliance.** C-1 (no audit-confession paragraph — none added, none present); C-2/C-5/C-6 untouched (no claim sentence was edited); C-8 hermetic (every key cited is a published/citable work in `refs.bib`; no program short referenced); the CHLU self-citation is third person and non-possessive. ⚠ C-7 note only: the R4 continuity sentence *"the CLU, introduced as CHLU in Jawahar & Pierini (2026)"* was removed by the Head's 19:01 edit; the current §1 carries the equivalent *"…similar to Jawahar and Pierini (2026), presented here as the Causal Learning Unit (CLU)"*. Flagged, not touched.

**Attachment table (for one-word strike-out)** — full version with rationale in `BUILD-NOTE.md` §3:

| key | site | strike risk |
|---|---|---|
| `hairer_geometric_2006` | §3, *"leapfrog stability parameter h = εμ"* | low — restoration, trap-safe use |
| `huang_measuring_2025` | Discussion, *"solution degeneracy between the designed and emergent arms"* | very low — restoration, sentence names the title object |
| `gell-mann_behavior_1968` | App. I.1, first GMOR use | very low — Head-ruled |
| `jelassi_repeat_2024` | App. A item (1), beside HiPPO-LegS | ⚠ **highest — strike first if any** (sits under an "e.g." for retention *guarantees*; Jelassi is a *limits* result) |
| `karuvally_exponential_2025`, `ramsauer_hopfield_2020` | App. A item (4), trailing *"…for associative memory"* | medium — could be mis-read as attaching to Titans' mechanism |

Related-work positioning prose was **not** rewritten in this pass; the attachment homes are taken verbatim from `.claude/outputs/v2-bib-doi-list.md` §Deliverable 4(a) (web-scout report), which is the report I lifted them from.

## Git footprint
**None.** No tracked file touched, no branch, no commit. All writes are outside the repo (`~/Desktop/V2_NeurReps_Submission/`) or under `.claude/` (`outputs/v2-cite-pass.md`, `scratch/v2-cite-pass/`). ⛔ `.claude/NIPSsubmission/v2-neurreps/` was never opened for writing; md5 re-verified after the pass.

## Open questions / follow-ups / risks
1. **The extra alignment tab (recon item 1)** — one deletion, Head's word. Until then the shipped folder builds with 1 error.
2. **The three biology orphans (recon item 2)** — attach with option (a), restore the deleted clause with its disclaimer via option (b), or leave uncited. Head's call.
3. **`\citet` vs `\citep` at three sites** (the CHLU self-cite, the §4.2 heading, the Fig. 2 caption) — I chose `\citet` for grammar (*"similar to (Jawahar and Pierini, 2026)"* is a grammar error class). One-token reverts if the Head prefers strict parentheticals.
4. **`\cite` in the subsection heading** — the zero-text-change TOC-short-title fix is available and removes all 9 hyperref warnings; needs a yes.
5. **`refs.bib` year policy** (recon item 4) and the **Di Bernardo surname** (recon item 5) — both one-line `.bib` edits, neither made by me.
6. **The three bare `arXiv:2605.03338` literals** were left as literals (out of the seven enumerated sites; converting them would be class 5). Each is a one-token swap to `~\citep{mo_symmetry-protected_2026}` if the Head wants them as citations.
7. **Build products** now sit in the submission folder (`paper.pdf/.aux/.log/.bbl/.blg/.out`). Add.69 specified a folder with no build artefacts; `BUILD-NOTE.md` §8 gives the one-line cleanup that keeps `paper.pdf`.

## Proposed handover updates (for the Hub)
- **V2's citation layer is complete and built.** `~/Desktop/V2_NeurReps_Submission/` now holds `paper.tex` on `\documentclass[mlabstract,onecolumn]{jmlr}` with `\bibliography{refs}`: **44 citation commands** (37 `\citep` / 6 `\citet` / 1 `\citealp`), **48 of 52** `.bib` entries typeset, **0 manual author-year strings**, **0 `\TODO`**, **0 undefined citations/references**, **23 pp** (main 1–8 · refs 8–12 · appendices 12–23). `BUILD-NOTE.md` is in the folder.
- ⛔ **One blocking defect is Head-owned:** the extra alignment tab in App. G's first negative-results table, introduced at 19:01, is the sole build error; a scratch build with it removed is clean at 0 errors.
- ⛔ **Ledger correction:** `refs.bib` is now **52 entries, not 53** — `anonymous_theory_2026` was removed at 19:17, consistent with the Head's deletion of its citing sentence. V2 no longer references the supplementary theory note in prose or bibliography.
- ⭐ **The self-correcting-metadata mechanism worked as designed:** Dinc et al. now prints **2026, Phys. Rev. X 16(2):021058**; the truncated Vafidis title prints in full; Nijkamp carries the AAAI DOI instead of the wrong `arXiv:1903.12370`. ⚠ Side-effect to rule on: five other citations now print their **arXiv posting year** instead of their proceedings year.
- ⛔ **Standing note for any future V2 spoke:** three bibliography orphans (Gardner, Khona, Kim) exist *because the Head deleted their sentence and its no-biological-claims disclaimer*. They must not be silently re-attached — doing so re-imports a biological claim without its qualifier.
- **Add.69's `\cite`-in-heading residual is confirmed with evidence** (9 hyperref bookmark warnings) and has a zero-text-change remedy on offer.
