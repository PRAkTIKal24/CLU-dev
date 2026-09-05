# pj-minimal-v2 — paper-writer — the HEAD-APPROVED minimal pass on V2 `pj_sub.tex` (exactly 12 main-text items + §-symbol rule + appendix-only bank restorations)

**Commissioned by the Shorts Advisor (charter Addendum 61, 2026-08-22). ⛔ THE PREVIOUS RESTORATION (R3) WAS REJECTED BY THE HEAD AS OVER-SCOPED AND REVERTED — this pass exists because the Head approved a line-item list. You execute EXACTLY that list. Anything not enumerated here is FORBIDDEN, however beneficial it looks.** Read `.claude/AGENT_PROTOCOL.md`, then this file. Object: `.claude/NIPSsubmission/v2-neurreps/pj_sub.tex` (md5 at issue: `d15de78712d90eb94d2495d4bd9ad948`). Sources: `submission.tex` (same folder, primary ancestor) and `.claude/scratch/pj-restore-R3-preserved/pj_sub_v2_R3.tex` (the R3 bank — already-fitted blocks; cite which source each edit used). Report → `.claude/outputs/pj-minimal-v2.md`; `BUILD-NOTE-R4.md` in the folder = deliverable #1.

**DIAL DECLARATION: none — editorial; zero new numbers without ancestors.**

## Head style rule (global, this file)
⛔ **The `§` symbol is never used for section references.** Write `Sec.~\ref{...}` (sections) / `Appendix~\ref{...}` (appendices) instead. Convert the file's existing 9 occurrences (add a `\label` where a referenced section lacks one); every sentence you add uses the same form. Report the conversion table.

## Part 1 — the 12 approved main-text items (word/clause/sentence level; NOTHING else in main text)
1. Abstract: *"is required to protect"* → *"protects"*.
2. *"guarantees $\dim(G/\mathcal H)$"* → *"guarantees at least $\dim(G/\mathcal H)$"* (l.60).
3. l.44: *"This chapter"* → *"This paper"*.
4. Abstract: add *"most"* before *"infinitesimal perturbations"*.
5. *"recently published"* → *"recently posted"* at l.47 and l.104.
6. l.226: add *"(overdamped-only)"* to the 0.9987 correlation.
7. l.268: add *"except the $\gamma_\varphi$ rung"* to the config line.
8. l.381: un-fuse the two facts (base 382's two separate sentences: the √ onset *"down to"* + the f=2,4 quality-factor condition + the dynamically-silent clause) — verbatim from base.
9. Add the §A20.5 substrate-scope sentence (base 145(iii), verbatim) as a Discussion bullet.
10. Add the score sentence in its measured form (base 145(ii) verbatim: *"No external benchmark is won on its own headline metric anywhere in this paper"*) beside the Head's existing benchmark sentence — the Head's sentence stays.
11. Fig-3 caption: append the single-representative-checkpoint clause (base 267, verbatim).
12. Add the fit-spec label to each slope site (−0.956 = per-point fit over all overdamped rows, at l.130; −0.961 = seed-mean OLS over the 7 overdamped δ, in the Fig-3 caption/legend text) — two short clauses.

⚠ Item 13 from the approved list (the unverified authorless poster): soften l.106–108 to uncited prose (*"concurrent workshop work has explored soft symmetry regularization for continuous attractors"*) and delete its References entry.

## Part 2 — banked information → APPENDICES ONLY (the Head's amendment; nothing from this part touches main text)
Restore, at their base appendix homes, from the R3 bank / base:
(a) the 6 missing negatives rows + their within-row numbers → Appendix E (base App E, verbatim);
(b) the head-to-head finite-horizon robustness numbers (corr = 0.9995 · 0.86–1.03 · 0.30 vs 0.31) → Appendix F (base 376) — ⛔ appendix only, the main-text §4.2 sentence stays OUT;
(c) the width-match confound + scan-amortized timing protocol → Appendix D (base 273, 299);
(d) the Appendix-G fences (demotion label · the "do not quote 2.2×10⁻¹⁶" precision fine print · the δ=0.3 no-NLO clause · G.6 honest scope · the definition of $x$) → Appendix G (base App F.5/F.6 forms per the R3 bank);
(e) the CM-17 sampler fence (*"the failure is in the sampler, not the thermodynamics…"* + *"we never assert a relativistic unit 'has no equilibrium'"* + the `newtonian_learned` scope) → the Appendix-E FAQ row at l.358 (its site IS appendix content).
⛔ Banked items whose base home is MAIN TEXT stay out entirely (N46 rider at the negative's site · anchor non-novelty clause · legacy-default warning · sleep flags · orphan-citation sentences · 4.5/4.6-decade clauses · seed attributions · ≈35× · floor ripple · coRNN footnote · "generally" (SF-7, unruled) · all intensifier/garble edits). List them as not-restored in the build note.

## Part 3 — figures
V2 already uses all 5 available PNGs. ⛔ No figure work; confirm 5/5 in the build note.

## After editing (mandatory)
Rebuild ×2 (`/Library/TeX/texbin/pdflatex`, in-folder; 0 errors / 0 undefined refs) · two-way numcheck (every added token → ancestor line) · sweeps positive-controlled (never-quote list · author-token rule · `pseudo-gap` = 0 · hermeticity) · **⭐ THE HEAD'S REPORT ITEM: a table of which appendices were ADDED or CHANGED relative to the Head's rewrite** (expected: E, F, D, G changed; none newly created — say so explicitly either way) · final md5 printed · every other folder file byte-untouched · page split reported, not optimized.

## Acceptance criteria
Exactly the enumerated edits + Part-2 appendix restorations + § conversion — zero other diffs (the build note proves it with a classified edit list); riders verbatim; orphan list empty; `submission.tex` untouched; the appendices-changed table present.
