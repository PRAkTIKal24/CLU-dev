# nips-v2-clean — paper-writer (V2's clean iteration base: `.claude/NIPSsubmission/v2-neurreps/`)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 48; Head directive 2026-08-21).** Read `.claude/AGENT_PROTOCOL.md`, then this file.

**Output — a NEW folder `.claude/NIPSsubmission/v2-neurreps/`:** `submission.tex` · `submission.pdf` · `figs/` · `BUILD-NOTE.md`. ⛔ **`BUILD-NOTE.md` is deliverable #1, not an afterthought** (the previous pass shipped its PDFs before its note). ⛔ Every other paper folder is READ-ONLY; never run `pdflatex` outside your own output folder. `pdflatex` is at `/Library/TeX/texbin/pdflatex`.

**Source:** `.claude/papers/plain/v2/submission.tex` (the plain build: page-fitting devices stripped, the author token removed, banked figures restored).

**This folder is the base the Head will ITERATE on.** Get the content and the framing right; the page count is explicitly not a target in this pass.

**DIAL DECLARATION: none — framing/editorial pass; zero number changes.**

## What carries over unchanged from the source (verify, do not redo)
No page-fitting typography · no bold outside structural headers · the author token absent from prose, captions, labels and filenames (⚠ "Morse"/"Moser" must survive) · banked figures restored with provenance-bearing captions · single-seed figures labelled as such. **Verify each and state the check; fix anything that slipped.**

## 1 — Audience scoping on 2025–2026 data (the new work; source: `outputs/audience-refresh-2025-2026.md`)
The draft's framing was written against a 2022 census. Re-aim it at the room as it is now:
- ⭐ **Use the room's current words.** *"Symmetry breaking"* is now titular here (a 2025 spotlight uses it) — state our spontaneous-symmetry-breaking framing **directly, with no apology and no translation**. Use **"canonicalization"**, never *"fundamental-domain projection"* (four years stale). ⚠ **Disambiguate "flow" on first use** — the 2025 poster list contains both Ricci flow and one-parameter Lie time-symmetry senses.
- ⭐ **Position against the 2025–26 neighbours, not the 2022 ones.** The refresh's set, with what each does to us: **N5/N5b flow-equivariant RNNs / world models** — same object (a one-parameter Lie flow carried in a recurrent state, long horizon), **orthogonal contribution: they generalize, we price** — this is the best positioning anchor in the room; **N2 symmetry-regularized continuous attractors** — the emergent-arm competitor (they obtain the flat direction by soft regularization, we by construction); **N4 solution degeneracy across task-trained RNNs** — the instrument a referee will ask us to apply to the designed-vs-emergent gap, so name it before they do; **N6 geometry of memory organization in RNNs** — the organizers' own live line, and the strongest "cite your reviewers' current work" candidate; **N7** (Noether-in-noise, symplectic integrators, 2024) — precedent that our formal register is accepted here, so introduce nothing apologetically.
- **Fit, stated plainly in the framing:** dynamics/attractors/RNNs is ~15 % of the 2025 accepted set and a standing CFP topic; this is a dynamics paper and should read like one.

## 2 — ⛔ The N1 novelty scoping (binding, conservative)
**arXiv:2605.03338** already publishes: ≥ dim(G/H) zero Lyapunov exponents tangent to the group orbit; that broken protection yields a **pseudo-gap**; and that the pseudo-gap **predicts finite memory lifetime** — on S¹, against matched GRU/LSTM/orthogonal-RNN baselines.
⛔ **No sentence may present the zero-mode ⇒ pseudo-gap ⇒ finite-lifetime chain, or the term *pseudo-gap*, as ours.** Cite N1 for it, in the citation-only form (item: the author token appears nowhere).
✅ **Claim only what survives:** the **closed-form price list on a trained potential** — the μ⁻² law with its measured slope, the mass-independent floor, the exceptional-point crossover, and the invariance of those laws under the corrective anchor. Say plainly that the existence result and the qualitative lifetime prediction are established, and that our contribution is the **exchange rate**.
⚠ **Units, load-bearing:** their gap is a **Lyapunov exponent** (1/time); our μ² is a **curvature of a trained potential** (1/time²). Any comparison states the conversion.
⚠ **A full-text read of N1 is in flight** (`tasks/n1-fulltext-and-track-check.md`). Write the conservative version now; if that report finds a lifetime *law* rather than a prediction, one further narrowing pass follows. **Do not wait for it.**

## 3 — Wording: simple, direct, strictly PJ
`.claude/PJ_Writing_Style_Context.md` applied strictly — ABT openings (abstract, §1, each results subsection) · macro-to-micro · **short declarative sentences, one idea each** · plain technical terms in place of program-internal formulations · zero weasel words · signposting · "we" for our actions, passive for established facts · `\texttt{}` for software/flags/files · italics sparingly. ⛔ **Simplify the prose, never the claim** — approved wordings, riders, scope qualifiers and fine print stay **verbatim** beside their claims.

## Boundaries
1. ⛔ Zero number changes; zero findings added or dropped; two-way numeric-token check against the source, printed.
2. ⛔ All sweeps (never-quote · internal-apparatus · semantic hermeticity), per-file, positive-controlled, printed; anonymization posture unchanged.
3. ⛔ Page count reported but **not optimised** — no typography, no cutting for space. If something is long, leave it long and say so.

## Acceptance criteria
1. `BUILD-NOTE.md` present and complete: page split (marked not-optimised) · the audience-scoping changes made · the N1 scoping statement · numeric check · sweeps · the carried-over verifications · figure inventory.
2. Every 2025–26 neighbour used is from the refresh's verified set; no 2022-era term survives as the audience's current term.
3. No sentence claims the N1 chain; the price-list framing is explicit.
4. All source folders byte-untouched (state the check).
