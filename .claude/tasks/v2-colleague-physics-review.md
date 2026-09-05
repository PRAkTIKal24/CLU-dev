# v2-colleague-physics-review — physics-theorist

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 7) at the Head's direction, 2026-08-18.** Read `.claude/AGENT_PROTOCOL.md`, then this file. Repo is read-only for you; you write exactly one report.

## Why this exists
The Head's colleague (a co-author; the Head owns all authorship decisions) has contributed a LaTeX physics primer for the V2 short: `.claude/colleague/main.tex` (25 lines, "Memory ML CLU - physics content only") — a pedagogical SO(2) spontaneous-symmetry-breaking derivation (Wigner-Weyl vs broken realization; the Goldstone mode as the zero-eigenvalue Hessian direction `∇²V·Jq* = 0`), ending in one claims-bearing sentence: the flat direction is *"a neutral direction in which information can be stored without being pulled back."* Before the Head decides how (and whether) it enters V2, the program needs: a rigor check, a reconciliation against the theory estate's conventions, and a complete rider list. ⛔ You do NOT rewrite or edit the colleague's prose, the V2 draft, or any file other than your report — integration is an authorship decision above your pay grade.

## Inputs (read in this order)
1. `.claude/colleague/main.tex` — the object under review.
2. `.claude/papers/v2-short/draft.md` §1–§3 + Appendix J, and its CHANGELOG (v0.5 adopted a three-realization symmetry taxonomy in §2.1 from `v2-symmetry-deepdive`; v0.6 applied the CM-16 split in App J).
3. `.claude/outputs/v2-symmetry-deepdive` (§1–§4.1 minimum) and `.claude/outputs/f1-gmor-condensate` — the program's existing SSB/GMOR formalism and notation.
4. `.claude/papers/f5-note/draft.md` (notation/conventions section) — the canonical-constants source.
5. Registry context (read the cited entries only): `negative_results.md` N46, N149/N150; `claims_matrix.md` CM-15, CM-16a/CM-16b (⛔ never cite "CM-16" whole), CM-17.

## Deliverable — `.claude/outputs/v2-colleague-physics-review.md`
Four sections:

1. **Correctness audit.** Verify every equation and every mathematical statement in `main.tex` line by line. Include a small numerical sanity check (jax/numpy/sympy) on a concrete SO(2)-symmetric potential (e.g. Mexican hat): confirm the Hessian at a chosen vacuum has the zero eigenvalue along `Jq*` and a positive radial eigenvalue; include the check code inline in the report. List technical/notation defects separately from substantive errors (candidates already spotted by the Advisor, verify and extend: `H ∈ G` vs `H ⊂ G`; "coset space corresponds to the number of broken generators" conflating space with dimension; "Lie algebra corresponds to rotations" phrasing).
2. **Convention reconciliation.** Symbol-by-symbol table: the primer's notation (q, α, J, R(α), r*, ω, vacuum/VEV language) vs the F5 note's, `v2-symmetry-deepdive`'s, and V2 draft §2.1/§3's. Flag every collision or divergence (e.g. α used elsewhere in the program for other quantities; mass/Hessian-eigenvalue conventions feeding CM-15 GMOR). State explicitly whether the primer's development is forward-compatible with the pseudo-Goldstone/explicit-breaking case (the third realization in §2.1's taxonomy and the home of V2's mode-mass budget) — the primer stops at the massless case.
3. **Claims-collision table.** Every claims-bearing sentence in the primer (at minimum the final "stored without being pulled back" sentence and the "spectral mass of zero" framing) × the binding registry objects: N46 (designed-only scope for any coset-register/latch claim), N149/N150 (pseudo-Goldstone tilt refuted in sign on a learned store — blast radius for downstream lifetime wording), CM-16a/CM-16b (friction preserves / temperature erases — the flat direction is exactly where T>0 diffusion erases; the sentence needs a T=0 scope or the caveat beside it), CM-17 (novelty scope: cite, don't claim). For each: the mandatory rider verbatim, or "no collision" with a one-line reason. ⛔ Flag, never fix.
4. **Integration map (options, not prose).** Where the content could live relative to the existing draft: replace part of §2.1 / feed §2.1 as a pedagogical lead-in / appendix primer / not-in-this-short. For each option: what riders travel, what duplicates the existing taxonomy, and what would have to be reconciled. No recommendation ranking — the Head and the colleague decide; your job is that each option arrives with its full cost printed.

## Acceptance criteria
1. Every flag in §3 carries a registry citation (N-number or CM row); every convention claim in §2 carries a file + section pointer.
2. The numeric sanity check runs (code + output in the report); if any equation fails verification, that is the report's headline.
3. Zero edits outside `.claude/outputs/v2-colleague-physics-review.md`. The colleague's file, the drafts, and the registries are untouched.
4. Substantive issues, mandatory riders, and editorial nits are three separate lists — never mixed.
5. Standard `## Proposed handover updates` and `## Flags` sections at the end.
