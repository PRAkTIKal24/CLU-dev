# v2-submission-build — paper-writer report

Task + acceptance criterion: build `papers/v2-short/submission/` (submission.tex + submission.pdf + build note + CHANGELOG entry) from the v0.8 canonical — internal apparatus stripped, appendix triage to the Head's visual/measured-results criterion, parentheses purged with the C-5/C-6 distinction preserved, full Add.2 + Add.28-Option-B anonymization, canonical byte-untouched.
**Status: done — with ONE acceptance-adjacent criterion NOT met and reported as a finding (the 4-pp page target; see §1).**
**DIAL DECLARATION (echoed): none — editorial build; zero content/number changes; no performance claim, so no laundering control applies.**

> ## ⚠ DOWNSTREAM RECONCILIATION LIST (needs an owner — Hub/Head decision)
> 1. **The 4-pp limit is still not reachable from the sanctioned menu.** Main text measures **5.42 pp** in-template. The Add.25 lean was applied and **measured to buy exactly one typeset line (≈0.02 pp)**. Residual ≈**1.42 pp ≈ 940 words ≈ 29 % of the main text**. Costed menu in `submission/BUILD-NOTE.md` §3. ⛔ Nothing else moved (C-10 / task rule). **Head-level call.**
> 2. **No NeurReps or NeurIPS-2026 template exists on this machine** (filesystem search negative; the official NeurIPS style URL returns HTTP 403). Built on the only genuine NeurIPS-family style file present, `neurips_2025_ml4ps.sty` (NeurIPS-2025 ML4PS workshop) — correct NeurIPS geometry (5.5in × 9in, 10pt Times, `Anonymous Author(s)`), **its workshop notice box suppressed** so no wrong venue string appears. **The page count must be re-measured in the real template before submission.**
> 3. **C-7 (flag-provenance tables are part of the paper) is now deliberately suspended for submission artifacts**, per the Head's instruction 1 and the Add.29 doctrine. The canonical keeps Appendix B; the submission has none. If the Charter text is to record this, that is a curator edit — **it currently reads as an unqualified rule.**
> 4. **The title placeholder was resolved, not workshopped.** `[WORKING TITLE: …]` brackets are internal apparatus a reviewer would see, so the brackets were removed and the working title printed as the title. **The Head owns titles** — one word changes it. Same for the supplementary note.
> 5. **The primer's attribution sentence was removed** ("adapted from a co-author's tutorial note; the acknowledgement is placed in the camera-ready"). An anonymized build carries no acknowledgment, so this is correct for review — but **the camera-ready acknowledgment obligation is now recorded only in the canonical and here.**
> 6. **Registry tokens (N4/N5/N46/N51/N149/N150/N6/N12–N15/N19/N22, CM-4, SF-1…3) are stripped from the artifact.** Every *finding* survives (the negatives table carries all of them, keyed by claim), but a reviewer's copy can no longer be cross-walked to the registry by number. If any future referee pass wants to check "is N51 still in V2", it must be checked by content.

## What I did

**Deliverables** (all under `.claude/papers/v2-short/submission/`; zero repo/code edits, zero git footprint):
`submission.tex` · `submission.pdf` · `supplementary-theory-note.tex`/`.pdf` · `BUILD-NOTE.md` · `figs/` (3 figures, neutral filenames) · `neurips_2025_ml4ps.sty`. Plus one prepended entry in `../CHANGELOG.md`.

### Instruction 1 — internal apparatus
Stripped: venue-class header, draft-status block, reading-order note, cross-reference-convention note, the A.5 inline HTML sign-off comment (**the comment only — the corrected A.5 sentence is byte-identical**), every commit hash, every source-report and checkpoint name, every registry/claims-matrix/task token, the whole flag-provenance appendix, the whole retained long-form.

### Instruction 2 — appendix triage (11 appendices kept, 2 removed, 3 exceptions honoured)
- **REMOVED:** flag-provenance appendix (was B) · **Appendix M in full** (the retained long-form, ≈13 pp) · Appendix L.6 · the placement-pending notes.
- **KEPT** (figures/tables/measured results, each trimmed to result + figure/table + mandatory fine print): erosion study · isotropization/charge oscillation · kick-amplitude probe · exceptional point + damping corollary · loan curve + recovery ladder · per-step compute · GMOR proper · T>0 coset diffusion.
- **Exception 1 — the primer: KEPT** (Add.11). Boxed scope note compacted to one paragraph, **all four binding clauses intact**; corrected A.5 sentence verbatim.
- **Exception 2 — negatives: KEPT AS A COMPACT TABLE.** 11 rows, columns *claim tested · result · number*; every row carries a measurement (two carry "—" where the canonical has no number: the task-RMSE axis and the mean-spectrum regularizer). Two prose riders retained beneath it — the learned-store reading rules (a)/(b) and the sampler kinetic-mode scope — because they are mandatory claim-scope, not commentary. Rendered form verified visually (p. 13).
- **Exception 3 — retirements + prior art: KEPT COMPRESSED** to ≈half a page (four retirements + one bounding-prior-art paragraph).
- Appendices renumbered C→B … L→K, **every cross-reference converted to `\ref`** — no skipped letter, no pointer into a removed appendix, 0 undefined references.

### Instruction 3 — the parentheses purge
| | count |
|---|---|
| prose-bearing parentheticals, canonical kept-region | 392 |
| prose-bearing parentheticals, submission | 210 |
| **net removed** | **182** |
| **stripped** (provenance 33 · pointer chains 32 · editorial asides 4) | **69** |
| **converted to prose, retained in the sentence** (C-5 scale 26 · statistical/± 28 · C-2 labels 3) | **57** |
| technical glosses folded into the sentence | 131 |
| survivors carried verbatim | 136 |

⛔ **Nothing in a mandatory class disappeared** — proven two ways in "How I verified".

### Writing style (Add.30)
Applied to connective tissue only: an ABT opening added to §1 (setup → *however, no closed-form account* → *this paper therefore asks*), section signposting, subsections given real `\subsection` headings, parenthetical qualifiers rewritten into sentences, magnitude language left as the data supports it. **Weasel-word sweep: 0 hits** (the 32 "very" matches are all substrings of "every"/"discovery"). ⛔ **Matrix-approved wordings, mandatory riders and scope qualifiers were repositioned, never paraphrased** — the narrow-claim positioning sentence, the CM-1 non-comparability caveat, the FDT flag box, the GMOR precision fine print and the learned-store fence are carried verbatim.

### Anonymization (Add.2 + Add.28 Option B)
`Anonymous Author(s)` block · no acknowledgments/funding · metadata Author/Title/Subject/Keywords **all empty** · **no absolute path, username or project string anywhere in either PDF** · third-person self-citation intact ("the CLU (Causal Learning Unit), **introduced as CHLU in Jawahar & Pierini (2026)**") · theory note cited as **"Anonymous (2026), provided in the supplementary material"**.
**Supplementary produced:** `supplementary-theory-note.pdf`, 12 pp, from the f5-note source — `\author{Anonymous}`, date and arXiv preprint-class line cleared, **the internal provenance appendix removed in full** (it was self-marked "strip on arXiv" and carried `.claude/scratch` paths and the retracted-number corrigenda), internal preamble comments removed, the practitioner-note footnote **kept**. ⛔ `papers/f5-note/f5-note.tex` **byte-untouched** (mtime Jul 20 00:33, unchanged).

## How I verified (commands + real numbers)

- **Build:** `pdflatex` ×3, TeX Live 2026. **0 errors · 0 undefined references · 0 overfull/underfull-critical boxes · 20 pp · 900,533 bytes.** Two overfull boxes found on the first build (11.30 pt in the notation table, 2.32 pt in the negatives table) were cleared by narrowing column specs; `\raggedbottom` set for the appendix block only.
- **Page count:** `pdftotext -layout` per page. A full body page is **53 lines**; "References" is the **23rd non-blank line of p. 6** ⇒ main text = 5 pp + 22 lines = **5.42 pp**. Word counts from the source: **3,281 words** main text + 64-word caption + Figure 1 (≈0.38 pp).
- **What the Add.25 lean bought — measured, not asserted:** built the identical document with the long GMOR-proper sentence restored. References moved from line 244 to line 245, i.e. the lean saved **exactly one typeset line ≈ 0.02 pp**. The paragraph reflows, so the ≈45-word saving does not convert to page saving at that position.
- **No number changed (forward audit):** every numeric token in `submission.tex` — **446 distinct** — occurs in the canonical `draft.tex`+`draft.md`. **Zero exceptions.** (The only later additions are the LaTeX column widths `1.52`/`2.06` in.)
- **No mandatory qualifier lost (reverse audit):** every numeric token of the canonical **main text** (113 distinct) occurs in the submission main text. The only absences are `3.3`/`3.4` (now rendered by `\ref`) and `149` (a registry token, deliberately stripped).
- **Apparatus sweep, per-file, positive-controlled.** **ZERO HITS:** `commit` · `agent/` · `chlu/` · `.claude` · `tectonic` · `draft.md` · `draft.tex` · `Registry`/`registry` · `provenance` · `Appendix M` · `N<digits>` · `CM-<n>` · `SF-<n>` · `MF-<n>` · `[WORKING TITLE` · `[AUTHORS PLACEHOLDER]` · `<!--` · every source-report name · every checkpoint name.
- **Never-quote sweep. ZERO HITS:** CLU-former · certified · unlearning · exact deletion · "the item is gone" · "exact discrete FDT" · "samples Gibbs" · 0.384 · 16.28 · CAFE · C-MAPSS · N-CMAPSS · HEPA · CAMELS · bpc · S_eff · z_hole · 0.99985 · 54.56 · 306.76 · 300.09 · deltanet · ttt_mlp · MUNKEY · 0.4545 · 13.9 · memory vault · 107.77 · compositional · unaskable · Guo · Ginart · Sekhari · Track A · waitlist · paid-access · **companion · sibling · "our other" · "this program" · "the program"** · experiment-engineer · "per the Head" · **wormhole (now 0 — rephrased to "inter-unit coupling")**.
- **HITS, context-checked, compliant:** `CHLU` ×2 (continuity sentence + reference entry — the sanctioned instances) · "energy units" ×1 (the refusal statement in the FDT box) · `2.6` ×4 (all grid/probe values 2.6e−2 / 2.6e−6 / 2.69e−14 — **never the retracted number**) · "critical damping" ×1, "V-curve"/"V-shape" ×4, **all after `\appendix`; 0 in abstract/§1/§3 — Q11 intact** · `branch` ×3 (overdamped branch) · `head` (head-to-head/headline/single head) · `hub` (Schmidhuber).
- **Positive controls fired:** GMOR 15 · "introduced as CHLU" 1 · Rusch 7 · verification 11 · evidence 20 · Anonymous 3 · **33 reference entries** (the full v0.8 bibliography).
- **PDF metadata**, both files: Title/Subject/Keywords/Author empty; Creator `LaTeX with hyperref`; Producer `pdfTeX-1.40.29`. Binary scan for `/Users/`, the username, `Desktop`, `CHLU`: **0 hits in both PDFs**.
- **Canonical untouched:** `draft.md`/`draft.tex` mtime Aug 18 23:39 (unchanged); `f5-note.tex`/`.pdf` mtime Jul 20 00:33 (unchanged). Only `CHANGELOG.md` was written, as commissioned.
- **Visual check:** rendered p. 1 (clean anonymized title block, no venue string, no apparatus) and p. 13 (the negatives compact table).

## Findings

1. **The page target is a demotion decision, not a drafting problem — and the sanctioned lean is now empirically exhausted.** The v0.8 measurement (5.1 pp) was taken in a 10pt/0.9in `article` approximation; the real NeurIPS text block is ≈20 % smaller in area, so the same text is 5.42 pp. Removing all apparatus and purging 182 parentheticals did **not** move the needle materially, because the removed matter was mostly *inside* paragraphs that reflow. Recommended order if the Head wants 4 pp: demote §3.1 (verification-grade under C-2, ≈261 w) → 2 sentences + pointer; then §4 ¶1 (≈120 w); then §5's limitations ¶ → a scope box (≈150 w); then, only if still needed, move Figure 1 to the appendix (≈0.45 pp) — **which costs the C-3 headline figure and I do not recommend it.**
2. **The appendix triage cost less than expected.** The submission is 20 pp against the canonical's 32 — the cut is almost entirely Appendix M and the provenance tables. Every kept appendix is now table/figure-first.
3. **The negatives-as-a-table form is a genuine improvement, not just a compression.** Forcing "claim tested · result · number" exposed that two entries carry no number in the canonical either (the task-RMSE axis; the mean-spectrum regularizer degeneracy). Both are real nulls with no measurement attached; they are printed with an em-dash rather than being given one.
4. **One factual-error near-miss caught during the style pass:** my first subsection title read "a *published* lifetime law" for Mo (2026), which is a **preprint** — the exact error v2-revision-7 fixed at 8 sites. Corrected to "recently posted" before the build. Flagging it because it shows the relabel is easy to undo by accident in any future rewrite.

## Open editorial questions for the Hub / Head

1. **The 4-pp residual** — which demotion, if any? (menu above and in the build note).
2. **Title:** print the working title as-is (current state), or does the Head want it workshopped now?
3. **Keep the primer?** Add.11 says yes and I kept it; it costs ≈1.5 appendix pages and is the one appendix that is *not* figures/tables, so it is the visible tension with the Head's own criterion. One word strikes it.
4. **The template must be swapped before submission.** Confirm the venue and re-measure; the geometry should be near-identical but the notice box, line numbers and any venue-mandated section (e.g. an impact statement) are venue-specific.
5. **Should the derived `supplementary-theory-note.tex` stay in the submission directory?** It documents how the anonymized PDF was made; if the directory is meant to be exactly the upload set, it should move to scratch.

## Proposed handover updates (for the Hub)

- `papers/v2-short/submission/` exists as the first artifact under the Add.29 submission-build doctrine; `BUILD-NOTE.md` maps every removal to its canonical home and is the template for the next short's build.
- **Standing datum for the other shorts:** in the real NeurIPS text block, **≈664 words of running text = 1 page**, and a `\linewidth` figure ≈ 0.38 pp. V5's 4-pp PALM target should be planned against those numbers, not against the `article` approximation.
- **Standing lesson to record:** apparatus stripping and parenthesis purging buy **almost no pages** — they buy compliance and readability. Page targets are met only by demoting results or figures. Any future task that pairs "clean it up" with "and it will then fit" is mis-scoped.
- C-7's suspension for derived submission artifacts (reconciliation item 3) needs either a curator edit to the Charter or an explicit Advisor note, so the next paper-writer does not read C-7 and re-insert the provenance appendix.
