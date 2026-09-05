# pj-fidelity-v2 — doc-curator (fidelity audit of the Head's `pj_sub.tex` for V2, + render `pj_sub.pdf`)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 53; Head directive 2026-08-22).** Read `.claude/AGENT_PROTOCOL.md`, then this file. One report: `.claude/outputs/pj-fidelity-v2.md`.

## ⛔⛔ THE ABSOLUTE CONSTRAINT
**`.claude/NIPSsubmission/v2-neurreps/pj_sub.tex` IS THE HEAD'S OWN FILE AND MAY NOT BE EDITED — not one character, not a typo, not a broken reference.** You report; the Head decides. This holds even if it fails to compile.

**DIAL DECLARATION: none — verification/audit pass; no performance claim; no laundering control applies.**

## What this is
The Head hand-edited the clean base into `pj_sub.tex`. **It is a deep condensation, not a light paraphrase: 1,263 words against the source's 10,369 (~12 %).** Two questions follow, and both must be answered separately.

**Source of truth:** `.claude/NIPSsubmission/v2-neurreps/submission.tex` (the clean base, Advisor-accepted at Add.52) and, behind it, the registries (`claims_matrix.md`, `negative_results.md`).

## Part A — the Head's question: is what survived FAITHFUL?
For every factual statement in `pj_sub.tex`:
1. **Numbers, exactly.** Every numeric token — value, precision, units, error bars, seed counts, sample sizes — matched against the source. Report any that differ **in any digit**, are rounded, lose a ±, or appear with a different unit or scope. Also report numbers present in `pj_sub.tex` that appear **nowhere** in the source (a number with no ancestor is the most serious finding available here).
2. **Claim equivalence.** For each surviving claim, quote **both** texts side by side and rule: **IDENTICAL / NARROWER (safe) / WIDER (⛔ a misrepresentation) / CHANGED IN KIND**. Watch specifically for: a tie becoming a win; "no harm" becoming an improvement; a bound becoming a point estimate; a designed-arm result reading as general; a correlation reading as a law; an instrument reading as a measurement of the thing it proxies.
3. **Attribution and priority.** ⭐ The N1 scoping is binding (Add.49): the zero-Lyapunov-exponent theorem, the qualitative *breaking ⇒ finite lifetime* prediction, and the word *pseudo-gap* are **not ours**. Check no sentence now reads as claiming them. Confirm what we do claim is the two-branch law, the crossover, the floor, the trained-potential measurement, and the regime-structure demonstration.
4. **The author-name rule** (Add.51): the token must be absent from body text, captions, labels and filenames; ⚠ "Morse"/"Moser" are different words and must survive. The bibliography entry keeps its authors.

## Part B — the companion question the Head did not ask, and needs: what was LOST?
At 12 % survival, the risk is not misquotation but **omission of content that is mandatory**. ⛔ Flag, never fix — and separate this list from Part A:
- **Approved wordings** (matrix CM rows) that appear in the source and are absent or reworded in `pj_sub.tex` — quote the approved form and the replacement.
- **Mandatory riders and scope qualifiers** now missing from the claims they qualify: the N46 designed-only rider · the C-5 scale qualifiers (dim, seeds, laptop-CPU) · the `fdt`+Newtonian kinetic-mode fine print · the instrument caveats · the CM-16a/b split · the CM-21 retirements · the substrate-scope sentence · the score sentence.
- **Never-quote violations** introduced by the condensation — full per-file sweep against matrix §0.1–§0.14, positive-controlled, zero-list printed.
- Citations: any claim whose supporting citation was dropped; any citation with no bibliography entry.
- Figures: any figure whose caption no longer matches what it shows, or whose scope label (single-seed, n<3) was dropped.
For each: **what the rule is, where it was, and which claim now stands unqualified.** Rank by consequence — a claim standing without its mandatory rider is a claims violation, not a style matter.

## Part C — render (mechanical)
Build **`pj_sub.pdf`** in place with `/Library/TeX/texbin/pdflatex -interaction=nonstopmode pj_sub.tex` ×3. Auxiliary files (`.aux`/`.log`/`.out`) are fine; ⛔ **the `.tex` is not touched.**
⚠ **If it does not compile:** do **not** repair `pj_sub.tex`. Copy it to `pj_sub_buildcopy.tex`, make the **minimum** repairs needed to produce a readable PDF, build `pj_sub.pdf` from the copy, and **list every repair** in your report as an edit the Head owes their own file. State clearly in the report and in the PDF's own build note which file the PDF came from.
Report the page split (main / references / appendices) — reported, not judged.

## Acceptance criteria
1. Part A: every number and every claim adjudicated with both texts quoted; the verdict vocabulary used exactly.
2. Part B: a ranked list of mandatory-content losses, separate from Part A.
3. `pj_sub.pdf` exists and is readable; if built from a copy, every repair listed.
4. ⛔ `pj_sub.tex` byte-identical at the end of the pass — state the check (md5 before and after).
