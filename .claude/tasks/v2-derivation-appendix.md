# v2-derivation-appendix — physics-theorist — ONE appendix section deriving the paper's central closed forms

**Commissioned by the Shorts Advisor at the Head's direction, 2026-08-24 (charter Addendum 75).** Read `.claude/AGENT_PROTOCOL.md`, then this file. Output report: `.claude/outputs/v2-derivation-appendix.md`.

**Why this exists:** an independent referee pass found that V2's central closed forms are **asserted with no derivation anywhere in the submission** — the paper carries two instances of derivation/proof vocabulary in total, and the theory note that once held the algebra has been withdrawn by Head ruling. The Head has authorised **one page of the most essential proofs, and nothing else.**

**DIAL DECLARATION: none — an analytical write-up. ⛔ ZERO new measurements and ⛔ ZERO new numbers: every formula you derive ALREADY APPEARS in the paper. You are supplying the algebra between them, not new results.**

## Object and the absolute constraint
- **File: `.claude/NIPSsubmission/v2-neurreps/condensed_paper.tex`** (the Head's ~4.5-pp condensation; `jmlr`/`mlabstract` class).
- ⛔⛔ **You ADD exactly one `\section{}` inside the existing `\appendix` block, and change NOTHING else. Not one character outside your inserted section** — not the main text, not another appendix, not the preamble, not a caption, not a typo you notice. **Head ruling: *"apart from the additional appendix section nothing else should be touched."*** Prove it: md5 the file before, and in your report show a `diff` whose only hunk is your insertion.
- ⚠ **Placement:** immediately after the definitions/setup appendix, so it reads before the results appendices. **Give it a `\label{}`** — see the Pointer note below.

## What to derive — the essential set, in this order
All from the **damped velocity-Verlet (leapfrog) map with per-step damping `p → (1−γ)p`** on a locally quadratic potential of transverse curvature `μ²`, reduced to its 2×2 normal-mode block. Take the paper's own symbols and conventions (`ε` step, `γ` damping, `μ` transverse curvature, `h ≡ εμ`) — read them off the paper, do not invent notation.
1. **The 2×2 block and its eigenvalues.** State the one-step propagator for a normal mode and solve the quadratic. This is the object everything else follows from, and the paper currently states it exists without solving it.
2. **The three bands.** Show where the discriminant changes sign, giving the overdamped / underdamped split, and the stability limit the paper quotes.
3. ⭐ **The exceptional point at `h* ≈ γ/2`:** show the block becomes **defective** (repeated eigenvalue, non-diagonalisable) there. The paper asserts defectiveness; derive it.
4. ⭐ **The overdamped half-life:** from the slow eigenvalue, derive `n₁/₂ ≈ 2γ ln2 / [(2−γ)(εμ)²]` — including the `2γ/(2−γ)` coefficient, which is exactly what a sceptical reviewer would try to reproduce.
5. ⭐ **The curvature-independent floor:** derive `n₁/₂ = 2ln2 / (−ln(1−γ))` and show it is μ-independent — i.e. why the law saturates rather than continuing.
6. **If it fits in the page:** the coset diffusion coefficient `D_θ = εT(2−γ)/(2F²γ)` from the FDT-consistent noise the paper specifies. ⚠ If space is tight, **items 1–5 outrank it**; say so rather than compressing 1–5.

## ⭐ Numerical self-check (your comparative advantage — do this)
Before writing, verify each derived expression reproduces the paper's own published constants — e.g. the floor at `γ = 0.05` must give **27.03 steps**, and the overdamped branch must reproduce the quoted slope of `−1` in `log n₁/₂` vs `log μ`. **Run a small numerical check** (numpy/sympy: build the 2×2 map, iterate, measure the half-life, compare to the closed form). ⛔ **If any derivation disagrees with a published number, STOP and report it** — a mismatch is a finding about the paper, not something to paper over by adjusting the algebra.

## Register and length
- **One page.** Terse, standard, checkable — the register of a methods appendix, not a tutorial. Numbered equations, minimal prose between them, no motivation paragraphs.
- ⛔ **No intensifiers, no new claims, no interpretation, no "this proves our contribution".** State the algebra and stop.
- ⛔ **The author token appears nowhere**; the theory note is withdrawn and ⛔ **must not be cited or alluded to** (Add.71).
- Use only packages the class already loads (`amsmath`, `amssymb`); ⛔ add no `\usepackage` line.

## ⚠ Pointer question — flagged, NOT executed
The appendix's value is highest if the main text's *"guided by an exactly-solvable underlying theory"* and the verification/evidence sentence point at it. **That is main-text editing, which the Head has forbidden in this pass.** ⇒ **Give your section a clean `\label{}`, state in your report the exact one-line `\ref` edits that would wire it in, and ⛔ do not make them.** The Head rules separately.

## After writing
Build to confirm it compiles — ⛔ **but build a COPY in `.claude/scratch/v2-derivation-appendix/`, never in `NIPSsubmission/`** (the folder holds the Head's live artifacts). Report: 0 errors; the added page count; the numerical self-check results; the `diff` proving a single insertion hunk; the before/after md5.

## Acceptance criteria
One new appendix section, one page, items 1–5 derived and numerically confirmed against the paper's own constants; **the file byte-identical everywhere else, proven by diff**; zero new numbers; the `\ref` wiring proposed but not applied.
