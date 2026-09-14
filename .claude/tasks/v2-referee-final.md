# v2-referee-final — paper-referee — the submission-state referee pass on V2, as the venue receives it

**Commissioned by the Shorts Advisor at the Head's direction, 2026-08-24 (charter Addendum 73).** Read `.claude/AGENT_PROTOCOL.md`, then this file. Output: `.claude/outputs/v2-referee-final.md`.

**DIAL DECLARATION: none — read-and-report. ⛔ ZERO edits to any file in `~/Desktop/V2_NeurReps_Submission/` or anywhere else; your only write is your report.**

## Independence bar (the design that has paid off twice — keep it)
⛔ **Do NOT read any prior referee or fidelity report** (`pj-referee-v2*`, `pj-fidelity-v2*`, `v2-referee-v07`, `v2-cite-pass`, `v2-bib-doi-list`, `v2-figure-text-pass`, or the submission folder's `BUILD-NOTE.md`). A defect you find unaided is far stronger evidence than one you were handed, and agreement between independent passes is what the Head can act on. You review the artifact as a reviewer receives it.

## Object and venue
- **`~/Desktop/V2_NeurReps_Submission/paper.pdf`** — 23 pp, built in the venue class. Source `paper.tex`, bibliography `refs.bib`, figures in `figs/`. ⛔ All read-only.
- **Venue: NeurReps 2026, Extended Abstract track** — **4 pp main text** (references and appendices excluded and unlimited), **non-archival**, **double-blind**. Track purpose, verbatim: *"Early-stage results, negative findings, opinion pieces, or novel datasets."*
- ⚠ **Page limits are DEFERRED by standing Head ruling.** Report the true split and state plainly what a chair does with it, but **review the content on its merits** and ⛔ never recommend a cut merely to fit — a dedicated compression pass comes later.
- Calibrate on `.claude/outputs/audience-refresh-2025-2026.md` (the 2025–26 room: *symmetry breaking* is titular vocabulary; the dynamics/attractors/RNN bucket is ~15% of the poster list; *canonicalization* replaced the 2022 term) and `.claude/outputs/neurreps-audience-scout.md` (the continuous-attractor bridge literature).

## What is NEW in this state — weight your effort here
A prior independent pass traced every headline number to its source and found **zero fabricated or mis-transcribed values**, so ⛔ **do not re-trace the numeric spine exhaustively.** Spot-check, and flag only a number that looks **new, changed, or internally inconsistent**. Spend your effort on:
1. ⭐ **The bibliography, now real.** ~48 citations were wired in this pass. **Is the right work cited in the right place?** Check for: a citation that does not support the sentence it is attached to; a claim that still needs a citation and has none; over-citation used as decoration; and the `\citet`/`\citep` distinction being used sensibly.
2. ⭐⭐ **The paper is now fully self-contained — no supplementary companion is submitted.** A theory note that earlier versions cited has been removed entirely by Head ruling. **Test that independently: does any claim, derivation, constant or "it can be shown" now rest on something the reader cannot see?** ⛔ This is the single highest-value question in this pass. If the paper asserts a result whose proof exists nowhere in the submission, say so and name the sentence.
3. **The novelty boundary against the head-to-head anchor** (`arXiv:2605.03338`, cited throughout). The defensible line: **theirs** = the zero-Lyapunov-exponent theorem (a *lower bound*, and *sufficiency* not necessity) and the qualitative breaking⇒finite-lifetime prediction; **ours** = the two-branch closed-form law, the exceptional-point crossover, the floor, measurement on a *trained* potential, and the demonstration that their estimator is the overdamped face only. ⚠ Units differ: their λ is a Lyapunov exponent (1/time), our μ² a curvature (1/time²). **Flag any sentence that crosses the line in either direction — over-claiming ours or under-crediting theirs.**
4. **Figures.** Five, freshly relabelled. Do they carry the claims they are cited for? Legibility at printed size? Captions self-contained? Any claim whose only evidence is appendix-only?
5. **Anonymity / desk-reject surface.** The author block is intentionally empty; check the built PDF for identifying material (metadata, acknowledgements, self-referential phrasing, figure filenames, any URL). ⚠ The class option must be **`mlabstract`** (non-archival), never `mlmain`.

## Deliverables
- **MUST / SHOULD / NICE triage**, each item anchored to a quoted sentence or a named object.
- **Simulated verdict** at the EA track, stated separately for (a) the content and (b) the artifact as mechanically shipped.
- **The three sentences a hostile reviewer would quote.**
- **The true page split** (main / references / appendices) and the chair's likely mechanical response.
- ⭐ **A section on what this paper's ONE contribution is, in your own words after reading it** — the Head de-scoped it deliberately to a single claim, and whether that lands is the pass's central judgement. If you cannot state the contribution in one sentence, that is itself the finding.
- **Missing-experiment list**, split into "a reviewer will demand this" vs "nice to have" — under the standing quality-first posture, *"this claim needs a run"* is a welcome finding, not an inconvenience.

## Acceptance criteria
Zero writes outside `.claude/outputs/v2-referee-final.md`; the submission folder byte-untouched; every MUST anchored to a quotation; the independence bar stated at the report head; the self-containment question (item 2) answered explicitly either way.
