# `v5-derivation-appendix` — a LEAN proofs appendix for V5, covering only what the current version asserts

**Agent:** `physics-theorist` (deliberate: this is algebra, and the theorist can numerically self-check it — the choice that made the V2 equivalent referee-proof)
**Commissioned by the Shorts Advisor at the Head's direction, 2026-08-26 (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 86).**
**Object:** `~/Desktop/V5_PALM_Submission/paper.tex`
**Report:** `.claude/outputs/v5-derivation-appendix.md`

---

## 1. Why this exists

The Head's own note on the paper read: *"add relevant proofs to appendix. no theory note, only relevant proofs to be added to appendix and referenced inline in the text."* V5 **ships no theory note and no supplementary companion** — it is self-contained by ruling. So any closed form the paper asserts must be derivable **inside the submission**, or a referee is asked to take it on faith.

⭐ **The precedent, and the bar:** the same job was done for the sibling paper by a theorist who derived every closed form from the underlying map and then **reproduced the paper's own published constants from its own algebra** (the retention floor: derived 27.0268 against the paper's printed 27.03). That converted an unverifiable assertion into a referee-proof one. **Match that bar.**

⛔ **"Lean" is the Head's word and it is binding.** This is **not** a port of the theory note. Derive **only what this version of the paper actually asserts**, in the fewest pages that do it honestly. A correct derivation of something the paper does not claim is scope creep, not generosity.

---

## 2. Boot — establish the worklist from the paper, not from this file

1. Read `~/Desktop/V5_PALM_Submission/paper.tex` in full. **It is the authority for what must be proved.**
2. ⭐ **Build the worklist yourself: sweep the paper for every closed form, exact relation, threshold and "it can be shown"-class assertion, and classify each as DERIVED-IN-PAPER / ASSERTED-WITHOUT-DERIVATION / CITED-TO-A-SOURCE.** The middle class is your worklist. Print the sweep with its positive control.
3. **Expected to be on it** (Advisor's reading — ⚠ verify against the file, do not assume): the V-curve minimum at `γ_crit = 2εμ`; the two log-slopes (`−1` below, `+1` above) and why the branches are asymmetric; the `T=0` latch condition; the coset diffusion coefficient `D_θ = εT(2−γ)/(2F²γ)`; the friction-hole / "refrigerator" absorb-only prediction the vault results are referenced to; and the left-branch complex-eigenpair exclusion `|λ|² = 1−γ ⇒ n₁/₂ = ln2/(−½ln(1−γ))`. ⛔ If the file does not assert one of these, **drop it** — do not derive it anyway.
4. Read `.claude/outputs/v5-vcurve-validation.md` for the measured constants your algebra must reproduce.

## 3. What to write

A single appendix section, `\label{app:derivation}`, placed with the other appendices. For each item: the assumptions stated as a short bullet list, the derivation, and the result **in exactly the form the paper prints it** (same symbols, same normalisation).

⛔ **ZERO new numbers, and zero new claims.** Every formula you derive already appears in the paper; you are supplying the algebra between them. If a derivation would require an assumption the paper never states, ⛔ **stop and report it as a finding about the paper** — that is a more valuable output than a patched proof.

## 4. ⭐ The numerical self-check — this is the deliverable's value, not an extra

**Pre-register your check values before running anything**, then verify each closed form **against the underlying map** (compose the elementary steps numerically and compare), **not against itself**. Reproduce the paper's own printed constants where they exist.

⛔ **THE STOP CLAUSE: if a derivation disagrees with a number the paper prints, STOP and report it as a finding about the paper. ⛔ Never adjust the algebra to match, and never adjust a printed number.** A disagreement found here is worth more than a clean appendix.

## 5. Wiring

The paper must **reference** the appendix, or it buys nothing (the sibling paper shipped a derivation appendix that nothing pointed at — the referee called it unreachable).

- You may add `\label{app:derivation}` and the appendix body.
- ⭐ **You may add the inline `\ref` wiring — at most TWO sites**, placed where the paper first asserts a closed form. **Report each as a separate labelled hunk.** Anything beyond those two `\ref` insertions is main-text editing and is ⛔ forbidden.
- ⛔ **Everything else in the main text is untouched.** No rewording, no renumbering, no re-flowing, no touching a number, caption, table or heading. ⭐ *The Head's text is the Head's; a defect you notice goes in the findings list, never into the file.*

## 6. Boundaries

- ⛔ **Never write to `paper.tex` while the Head is editing it.** Record its `md5` and mtime at boot; if either moved when you finish, **stop and report** rather than clobber. **Prove your footprint by diff:** the only hunks are the appendix insertion and the ≤2 `\ref` sites.
- **Build in a scratch copy**, never in `~/Desktop/V5_PALM_Submission/` — copy the folder to `/tmp`, iterate there, and only then apply the verified insertion to the live file.
- `pdflatex` is **not on `PATH`**: use `/Library/TeX/texbin/pdflatex`.
- ⛔ `.claude/NIPSsubmission/v5-palm/**` and `.claude/papers/**` are byte-untouched.
- ⚠ **Page budget:** main text is at **4.31 pp against a 4-pp venue limit** and the Head has accepted that. **Appendices are excluded from the limit**, so your appendix is free — ⛔ **but it must not push a single line into the main text**, and the two `\ref` insertions must not reflow it. Report the main-text page measurement before and after; if it moved, say so.
- ⚠ **Grep hazard:** directory-level grep over `.claude/` silently returns nothing (gitignored). Sweep per-file and positive-control every negative.

## 7. Deliverables

1. The classification sweep (§2.2) with its positive control — what the paper asserts and what it derives.
2. The appendix itself, in the live file, diff-proven as a single insertion + ≤2 `\ref` hunks.
3. **The numerical check table:** pre-registered values, derived values, the paper's printed values, and the agreement — including any row where the STOP clause fired.
4. Build evidence (0 errors, 0 undefined references) and the main-text page count **before and after**.
5. Anything the paper asserts that you could **not** derive from its stated assumptions, named explicitly. ⛔ This is a required section and "none" is an acceptable answer only if it is true.

## 8. Acceptance criteria

1. Every ASSERTED-WITHOUT-DERIVATION item is either derived or listed in §7.5 with a reason.
2. Zero new numbers; zero new claims; the STOP clause honoured.
3. Numerical check runs against the composed map, with pre-registered values, and reproduces the paper's printed constants.
4. Diff shows only the appendix insertion and ≤2 `\ref` hunks; main-text page count unchanged.
5. The appendix is reachable — at least one `\ref` resolves to it.

## DIAL DECLARATION
**Dials touched: NONE.** No experiment, no config change, no registry, no charter. This pass adds one appendix and at most two cross-references, and writes one report plus a scratch numerical check script.
