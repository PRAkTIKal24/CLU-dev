# v2-neurreps-descope — paper-writer (V2's NeurReps EA: ONE contribution, current audience, corrected novelty scope — FINISH IT)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addenda 41–42; Head rulings 2026-08-21).** Read `.claude/AGENT_PROTOCOL.md`, then this file. **No gate — start now.**

**Output: a NEW, distinct folder — `papers/v2-neurreps-descoped/`** (create it): `submission.tex` · `submission.pdf` · `BUILD-NOTE.md` · `figs/` (+ the supplementary theory-note PDF copied across). ⛔ **Both existing V2 artifacts stay byte-untouched: `papers/v2-short/**` (the live build) and `papers/neurreps-variants/v2/**` (the first reframe).** Three distinct V2 artifacts now coexist by the Head's design; keep them apart.

**Source:** `papers/neurreps-variants/v2/submission.tex` (the reframe).

**DIAL DECLARATION: none — scoping/editorial pass; zero number changes, zero retractions.**

## ⛔ The Head's standing instruction on length, and it overrides the earlier stop-rule
**Finish the paper.** Aim at 4 pp, but **do not stop, do not truncate, and do not trade a C-6 fine-print block for space if you overshoot** — the Head will do a condensation pass personally. A finished paper at 4.4 pp is the deliverable; an unfinished paper at 4.0 pp is not. Report the true split and hand the Head the condensation aid below.

## 1 — The de-scope (the ruling)
**Main text carries ONE contribution: the quantitative price list — the closed-form `μ⁻²` retention law with its crossover and floor, measured on a *trained* potential — with the published-law head-to-head as its evidence and the designed-vs-emergent boundary as its honest negative.** That is what survived the scout's novelty retraction, and it is a complete paper.

**Demoted out of main text (⛔ demoted, never retracted):** GMOR-proper · the realization taxonomy · the price-of-the-prior.
- Each keeps an appendix home **if it has a plot or a results table** (GMOR has both). Prose-only demotions go canonical-only and are listed in the build note.
- ⛔ **Abstract and contributions list claim only the retained contribution.** Demoted results may support it in a clause, never appear as contributions.
- ⛔ Nothing is retracted; no number changes; no finding leaves the record.

## 2 — The audience corrections
1. **Re-aim §2 at the CURRENT audience.** The Head's census is NeurReps **2022** (PMLR v197) — four years stale. Write §2 to the **2026 CFP** (its bullets *"Dynamics of neural representations"* and *"Symmetries, dynamical systems, and learning"* are this paper's subject line) and cite the current verified neighbours from `outputs/neurreps-audience-scout.md`, including the **v228 (2024)** items where relevant (Vastola on optimal packing of attractor states; Dönmez). Keep v197 works only where the scout marks them the genuine nearest neighbour — **Xu et al. (conformal isometry of a Lie-group representation in a recurrent grid-cell network) is the most important one**: the audience's own instance of this paper's object, and the natural place to say what we add.
2. ⛔ **Apply the scout's novelty retraction wherever the anchor/erosion result is framed.** The destroy-and-restore pattern is established prior art here (**Renart, Song & Wang 2003**; **Vafidis et al. 2022**). Print the scout's §2.3 scoped form. ⛔ No sentence may imply the phenomenon or its cure is first-reported by us. **What is ours is the price list on a trained potential** — say it plainly.
3. **Foreground the negative findings** — the designed-vs-emergent gap with N46's rider verbatim, the retirements, the instrument caveats — in the abstract's THEREFORE clause and the contributions list (the EA track's stated purpose is *"early-stage results, negative findings"*).
4. **CM-21 retirements:** stated, compressed, wherever the claim they bound is still made; those bounding a demoted contribution may move to the appendix. ⛔ None may be re-asserted anywhere.
5. **Bridge vocabulary** exactly as the scout mapped it, used only where marked exact; **the no-biological-claim sentence stays, exactly once**.

## 3 — Sweep-up
- **References:** drop entries left uncited by the de-scope; every remaining entry must be cited from the surviving body.
- **Figures:** Figure 1 (head-to-head) is the headline and stays in main text; figures whose contribution left main text travel with it to their appendix. ⚠ **A parallel `figure-render-pass` is regenerating all variant figures at final printed size and will list caption edits owed.** Use the figures as they stand now; the Advisor will hand the Head a one-step follow-up (copy the regenerated PNGs in, apply the listed caption edits, rebuild). Do not wait for it.
- **Style:** `PJ_Writing_Style_Context.md` strictly and more directly than the source — ABT openings, plain terms, one idea per sentence, no bold outside structural headers.

## Boundaries (absolute)
1. ⛔ Approved wordings, mandatory riders, scope qualifiers and fine print stay **VERBATIM**, beside the claims they qualify (C-6). ⛔ **Never traded for page space** — overshoot instead and report.
2. ⛔ Zero number changes; zero findings dropped from the record (demotion ≠ deletion). Two-way numeric check against the source variant, printed, plus the submission-absent/canonical-present list.
3. ⛔ All sweeps (never-quote · internal-apparatus · semantic hermeticity), per-file, positive-controlled, printed.

## 4 — ⭐ The condensation aid (a required deliverable — the Head condenses from this)
In `BUILD-NOTE.md`, a single table over every main-text block: **block · words · measured pp · PROTECTED or FREE · one line on what it does**. "Protected" = approved wording, mandatory rider, fine print, or the retained contribution's own evidence; "free" = connective prose, framing, motivation, anything the Head may cut without a claims consequence. Then: the measured cost of the three largest free blocks, and any block whose removal would change a claim (flagged ⛔, with which claim). This table is what makes the Head's manual pass fast — write it for a reader who did not write the paper.

## Acceptance criteria
1. The paper is **finished** and builds clean; main-text page count reported honestly against 4 pp (overshoot permitted and expected); total split reported.
2. Abstract and contributions claim only the retained contribution; novelty scoping printed; negatives visible in both.
3. §2 written to the 2026 CFP; every citation scout-verified.
4. Numeric check, sweeps, demotion list and **the §4 condensation aid** all printed; `papers/v2-short/**` and `papers/neurreps-variants/v2/**` byte-untouched (state both checks).
