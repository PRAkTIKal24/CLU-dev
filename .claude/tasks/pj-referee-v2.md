# pj-referee-v2 — paper-referee (review the Head's V2 submission version as a NeurReps EA reviewer receives it)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 53; Head directive 2026-08-22).** ⛔ **Mechanical precondition: `.claude/NIPSsubmission/v2-neurreps/pj_sub.pdf` exists on disk.** Read `.claude/AGENT_PROTOCOL.md`, then this file. One report: `.claude/outputs/pj-referee-v2.md`.

## ⛔ Two constraints
1. **Edit nothing.** `pj_sub.tex` is the Head's own file and is not to be touched by any pass. You review; you do not fix.
2. ⛔ **Do NOT read `outputs/pj-fidelity-v2.md`** (a fidelity audit running in parallel). **Your value is independence** — a defect you find on your own is far stronger evidence than one you were handed, and where the two passes agree the Head learns something real. Read the paper as a reviewer receives it.

**DIAL DECLARATION: none — adversarial review; no performance claim; no laundering control applies.**

## What you are reviewing
`.claude/NIPSsubmission/v2-neurreps/pj_sub.pdf` — the Head's own condensation (1,263 words of source text, ~12 % of the clean base) for the **NeurReps 2026 Extended Abstract track**: 4 pp main text excluding references and appendices, non-archival, double-blind. Read the PDF as the artifact; consult `pj_sub.tex` only for what the PDF renders ambiguously.

## The reviewer you are
The composite NeurReps 2026 EA reviewer, calibrated on **current** venue data (`outputs/audience-refresh-2025-2026.md`): a room whose own award categories are Neuroscience & Interpretability · Topological & Geometric ML · Symmetry & Equivariance; where **dynamics/attractors/RNNs is ~15 % of the accepted set**; where *symmetry breaking* and *canonicalization* are current vocabulary and *flow* is ambiguous between Ricci flow and one-parameter Lie time-symmetries. The track's stated purpose is *"early-stage results, negative findings, opinion pieces, or novel datasets."* Reflex objections in this room: physics-analogy overreach, toy scale, and "what does the geometry buy that an architecture does not."

## Review it on
1. **Does it stand alone?** At this compression, the first question is whether a reader who has never seen the long version can follow the claim, the evidence and the scope. Name every place where a step is missing, a term is used before it is defined, or a number arrives without its setup.
2. **Is the contribution clear and correctly bounded?** The paper should claim the two-branch closed-form retention law with its crossover and floor, measured on a trained potential, and the demonstration that the cited single-exponential relation is the overdamped face only. ⛔ **Flag any sentence that reads as claiming the cited work's theorem, its qualitative breaking⇒lifetime prediction, or its terminology.**
3. **Positioning against the current room** — is it placed against 2025–26 neighbours, and does it say what it adds to them? Flag stale framing.
4. **Evidence sufficiency at EA scale** — for each claim, is the evidence in the paper (not in an appendix the reviewer need not read, not in a companion)? Note where a claim rests on something not shown.
5. **Figures and captions** — do they show the claim; are they legible; do captions carry scope (seeds, dimension, budget)?
6. **Scope honesty** — the negatives and limitations this track explicitly welcomes: are they present and prominent, or buried?
7. **Fit for the track** — is this an extended abstract or a compressed full paper? Say which it reads as, and why.

## Deliverable
An itemized referee report: **MUST-FIX / SHOULD-FIX / NICE**, each with location and either a citation or a concrete reviewer-failure scenario; a **simulated accept/reject verdict for the EA track**; and **the three sentences a hostile reviewer would quote back**. Then a short section: *what this version gained and lost against a longer treatment* — the Head condensed hard and deserves to know what the compression cost in a reviewer's eyes. Standard `## Proposed handover updates` and `## Flags`.

## Acceptance criteria
1. Every MUST-FIX carries a location and a concrete failure scenario.
2. The verdict is stated plainly, with the track's stated purpose weighed explicitly.
3. Zero edits to any file except your report; state that `pj_sub.tex` was not touched.
