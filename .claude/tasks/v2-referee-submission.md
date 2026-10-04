# v2-referee-submission — paper-referee — the LAST look at V2 before it is submitted

**Commissioned by the Shorts Advisor at the Head's direction, 2026-08-24 (charter Addendum 79).** Read `.claude/AGENT_PROTOCOL.md`, then this file. Output: `.claude/outputs/v2-referee-submission.md`.

**⭐ This is the final pass. The next thing that happens to this paper is submission.** Frame every finding accordingly: the only useful output is *what would change the outcome*, not what would improve a draft with time to spare.

**DIAL DECLARATION: none — read-and-report. ⛔ ZERO edits to any file anywhere; your only write is your report.**

## Independence bar
⛔ **Do not read any prior referee, fidelity, curator or theorist report** — `v2-referee-final`, `v2-referee-v07`, `pj-referee-v2*`, `pj-fidelity-v2*`, `v2-condensation-equivalence`, `v2-derivation-appendix`, `v2-cite-pass`, `v2-bib-doi-list`, or any `BUILD-NOTE*`. This design has produced independent convergence on real defects three times; it is worth nothing if you are handed a predecessor's conclusions.

## Object
- **`.claude/scratch/v2-final-build/condensed_paper.pdf`** — 25 pp, built by the Advisor from the live source at md5 `4610001977d1aeed8c306d97f68473c4` (source, `refs.bib`, `figs/` and the class files sit beside it). ⚠ The Head maintains the canonical build in Overleaf; this build exists so you review pixels, not LaTeX.
- **Venue: NeurReps 2026 Extended Abstract track** — non-archival, double-blind, references and appendices **excluded from the page limit and of any length**. Track purpose, verbatim: *"Early-stage results, negative findings, opinion pieces, or novel datasets."*

## ⚠ What changed since this paper was last reviewed — where to spend effort
The paper has been **condensed**: main text roughly halved, with the displaced material relocated into appendices (deliberate, Head-ruled, **and NOT a finding**). A **new derivation appendix** was added. Several typo-class defects were fixed. So:
1. ⭐⭐ **Does the MAIN TEXT still stand on its own?** This is the pass's central question. A reviewer of an EA track reads the main text and is **not required** to open an appendix. Read the main text **alone, first, before you look at any appendix**, and answer explicitly: *does every claim it makes carry the qualification it needs, in the main text?* Name any claim that is stated in main text but qualified only in an appendix — ⛔ **that is the failure mode this condensation risks, and a diff cannot see it.**
2. ⭐ **The new derivation appendix.** Judge it as a reviewer would: are its assumptions stated, is it followable, and does it actually support the closed forms the main text asserts? ⛔ Do not re-derive it line by line — judge sufficiency and honesty, and flag anything that looks asserted rather than shown.
3. **Page compliance — MEASURE IT, and this is now live rather than deferred.** Report the exact main-text extent (to the nearest tenth of a page) and where the bibliography begins. ⚠ **A note you must resolve rather than assume: the program's banked venue fact (Head-verified from the venue site, 2026-08-18) is a 4-page main-text limit; the Head has been working to ~4.5.** State the measured number plainly and what a chair does with it at both readings. ⛔ Do not recommend content cuts — say what the number is.
4. **A final defect sweep of the reviewed surface:** typos, broken references, figure/text mismatches, undefined symbols, internal number contradictions, anything that reads as unproofread. ⚠ Weight main text far above appendices — a typo on page 2 costs more than three in an appendix.
5. **Anonymity / desk-reject surface** as shipped: metadata, author block, identifying phrasing, self-citation handling, class option (must be the non-archival `mlabstract`).

## Deliverables
- **Simulated verdict**, stated separately for (a) content and (b) the artifact as shipped, with the **desk-reject question answered explicitly**: would this be rejected before review, and on what?
- **MUST / SHOULD / NICE**, each anchored to a quoted sentence or named object. ⭐ For each MUST, state **the cost of NOT fixing it** in reviewer terms — the Head is deciding what is worth touching hours before submission.
- **The three sentences a hostile reviewer would quote.**
- ⭐ **A single explicit judgement: SUBMIT AS IS / SUBMIT AFTER THE MUSTS / DO NOT SUBMIT THIS CYCLE**, with your reason in one paragraph.
- **One standing question the Head has never ruled on, which you should judge independently:** the paper states *"Our results hold generally for the class of damped symplectic recurrences"* while every measurement is at latent dimension 4, on $S^1$, ≤5 seeds, one architecture, on a laptop CPU. **Is that sentence defensible as written?** Give a verdict and, if not, the minimal wording that would be.

## Acceptance criteria
Zero writes outside `.claude/outputs/v2-referee-submission.md`; the independence bar stated at the report head; the main-text-alone reading (item 1) done **before** any appendix is opened and its answer stated explicitly; the page number measured rather than estimated; the submit/don't-submit judgement given without hedging.
