# pj-fidelity-v5 — doc-curator (fidelity audit of the Head's `pj_sub.tex` for V5, + render `pj_sub.pdf`)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 53; Head directive 2026-08-22).** Read `.claude/AGENT_PROTOCOL.md`, then this file. One report: `.claude/outputs/pj-fidelity-v5.md`.

## ⛔⛔ THE ABSOLUTE CONSTRAINT
**`.claude/NIPSsubmission/v5-palm/pj_sub.tex` IS THE HEAD'S OWN FILE AND MAY NOT BE EDITED — not one character, not a typo, not a broken reference.** You report; the Head decides. This holds even if it fails to compile.

**DIAL DECLARATION: none — verification/audit pass; no performance claim; no laundering control applies.**

## What this is
The Head hand-edited the clean base into `pj_sub.tex`. **It is a deep condensation: 2,684 words against the source's 8,993 (~30 %).** Two questions follow, answered separately.

**Source of truth:** `.claude/NIPSsubmission/v5-palm/submission.tex` (the clean base, Advisor-accepted at Add.52) and, behind it, the registries.

## Part A — the Head's question: is what survived FAITHFUL?
1. **Numbers, exactly.** Every numeric token — value, precision, units, ±, seeds, sample sizes — matched against the source. Flag any digit change, any dropped ±, any unit or scope change, and ⛔ any number with **no ancestor in the source**. V5's number-dense claims deserve individual attention: the V-curve argmin and both slopes, the rollout/Jacobian agreement, the vault ratios (⚠ the law-referenced emergent figure vs the designed one — they are different objects), the AUC pair in the leakage result, the R₅₀ pair, the deletion load range, and the lifecycle leg counts.
2. **Claim equivalence.** Quote **both** texts side by side per claim and rule **IDENTICAL / NARROWER (safe) / WIDER (⛔) / CHANGED IN KIND**. Watch for: the designed-only contrast number reading as general; the vault's *laws* transfer reading as the vault transferring; store-level deletion reading as system-level or as certified unlearning; a mechanics result reading as a value result; the lifecycle reading as evaluated rather than shipped.
3. **The author-name rule** (Add.51): the token absent from body, captions, labels, filenames (⚠ "Morse"/"Moser" survive); bibliography keeps its authors.

## Part B — the companion question the Head did not ask, and needs: what was LOST?
⛔ Flag, never fix; keep separate from Part A. The `v5-referee-v02` **do-not-cut list is the checklist** — for each item, is it present, and does it still sit beside the claim it qualifies?
- **N108's sentence** (*"the store stops answering before it stops leaking"*) · **the exact-deletion form with its three conditions and the recency exclusion** · **the Blelloch–Golovin attribution at EVERY deletion site** (N118) · the lifecycle's two riders (demotion is re-exposure, never the trash region; the trash criterion is keyed on read-hits, never depth) · the substrate-scope sentence · the score sentence · the designed-symmetry precondition · the `fdt`+Newtonian fine print · the emergent-arm caveats (⛔ no σ_θ ratio; the θ=π-is-not-a-vacuum confound; the contrast number designed-only) · the k-regime scope clause on the erosion horizon · the corrected Guo citation form (§2, Eq. (1), ε-only) · CM-25(f) verbatim · the C-5 scale qualifiers.
- ⛔ **"certified"** must appear only in denial or literature-description form; **"unlearning"** never applied to our mechanism.
- Never-quote sweep, full, per-file, positive-controlled, zero-list printed.
- Citations dropped from claims that need them; citations with no bibliography entry; figure captions that no longer match their figure or lost their scope label.
Rank by consequence: **a claim now standing without its mandatory rider is a claims violation, not a style matter.**

## Part C — render (mechanical)
Build **`pj_sub.pdf`** in place with `/Library/TeX/texbin/pdflatex -interaction=nonstopmode pj_sub.tex` ×3. Aux files fine; ⛔ the `.tex` is not touched.
⚠ **If it does not compile:** do not repair the original. Copy to `pj_sub_buildcopy.tex`, make the **minimum** repairs, build the PDF from the copy, and **list every repair** as an edit the Head owes their own file; state in the report which file the PDF came from.
Report the page split (main / references / appendices) — reported, not judged. ⚠ Context, not a target: PALM's limits are 4 pp short / 9 pp full, references and supplementary excluded; the track choice is open before the Head (Add.52).

## Acceptance criteria
1. Part A: every number and every claim adjudicated with both texts quoted; verdict vocabulary used exactly.
2. Part B: the do-not-cut list walked item by item, present/absent stated for each, ranked by consequence.
3. `pj_sub.pdf` exists and is readable; repairs listed if built from a copy.
4. ⛔ `pj_sub.tex` byte-identical at the end — state the md5 check.
