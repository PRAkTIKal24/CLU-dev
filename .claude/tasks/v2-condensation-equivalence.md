# v2-condensation-equivalence — doc-curator — is the condensed V2 the SAME PAPER, only re-flowed?

**Commissioned by the Shorts Advisor at the Head's direction, 2026-08-24 (charter Addendum 75).** Read `.claude/AGENT_PROTOCOL.md`, then this file. Output: `.claude/outputs/v2-condensation-equivalence.md`.

**The Head's question, verbatim substance:** *"make sure content wise everything is the same and just the flow is changed."* They condensed to ~4.5 pp main text **by relocating material into the appendix, which is accepted practice** — so ⛔ **relocation is NOT a finding.** Loss, addition, and separation are.

**DIAL DECLARATION: none — read-and-report. ⛔ ZERO edits to any file. Your only write is your report.**
⚠ **You have no shell** (Read/Grep/Glob/Write only). There is **no build step** in this task; do not attempt one, and declare honestly anything that would need it.

## The two files
- **BASELINE (what it was):** `~/Desktop/V2_NeurReps_Submission/paper.tex` — the cited, built, referee-reviewed state.
- **OBJECT (what it is now):** `.claude/NIPSsubmission/v2-neurreps/condensed_paper.tex` — the Head's condensation, same `jmlr`/`mlabstract` class, citations preserved.
⚠ **Measured facts to start from, and one is counter-intuitive:** the condensed file is **LONGER overall** (6,185 vs 5,992 words) and carries **more distinct citation keys (50 vs 48)**, while its **main text is much shorter** (1,813 vs 2,763 words). ⇒ material moved down **and something was added**. Characterising the additions precisely is a core deliverable.

## Deliverable 1 — the relocation map
Every block that changed position: what it is, where it was, where it now sits. Group by destination appendix. This is the Head's evidence that flow-only is what happened.

## Deliverable 2 — ⛔ LOSS: content in the baseline that is in NEITHER the condensed main text NOR its appendices
Walk the baseline systematically: every claim · every numeric value (with its precision, ±, seed count, units) · every citation key · every figure and table · every caption · every scope qualifier and mandatory rider · every negative-results row. For each, state **present (where) / ABSENT**. ⛔ **An absent number or an absent rider is the most serious finding class available** — quote it and name the claim it used to support.

## ⭐ HEAD PRE-AUTHORISATION (2026-08-24) — read before Deliverables 3 and 4
The Head has stated the growth's cause and ruled it acceptable: *"the file grew because I added some text, but also there is a bit of duplication so that the text in the appendix is still fairly free flowing. That's acceptable."*
⇒ ⛔ **DUPLICATION BETWEEN MAIN TEXT AND APPENDIX IS DELIBERATE AND IS NOT A FINDING.** Do not report a passage as duplicated merely because it appears twice; do not recommend de-duplication; do not treat repeated prose as bloat. Added connective and framing text is likewise authorised.
⭐ **But duplication creates three specific hazards that ARE findings, and they are the only reason it is worth looking at twice:**
1. ⛔ **A number that appears in two places and DISAGREES between them.** This program has already shipped one such pair (an overdamped slope reported as two different values in two sections, both individually correct, from two different fits). Check **every** duplicated numeric value for exact agreement — value, precision, ±, seed count, units.
2. ⛔ **A claim duplicated where only ONE instance carries its mandatory rider.** The unqualified copy is a live claims exposure regardless of the qualified one existing elsewhere.
3. ⛔ **An approved wording or mandatory rider duplicated but PARAPHRASED in one instance.** Approved wordings bind verbatim in **every** place they appear — a "smoothed" second copy is a claims violation, not a stylistic variant.
⇒ Report duplication **only** under one of these three heads, with both instances quoted side by side.

## Deliverable 3 — ⛔ ADDITION: content in the condensed file with no ancestor in the baseline
The word and citation counts prove additions exist. For each: quote it, say whether it is **(a) connective/structural prose** written to make the new flow read (expected and fine), **(b) a restored item** the baseline had lost, or a Head-authorised duplication/expansion for flow, or **(c) ⛔ a NEW claim, number or citation** — which would mean the condensation is not content-preserving. ⛔ **Any new numeric value without an ancestor in the baseline is the top finding of this pass.** Also list the 2 new citation keys and the sentence each supports.

## Deliverable 4 — ⭐⭐ THE RIDER-ADJACENCY CHECK (the specific hazard of relocation, and the reason this pass exists)
When a number moves and its mandatory qualifier does not — or vice versa — the paper acquires an unqualified claim **without a single word being edited**. This is the failure mode a diff cannot see and the one that matters most here. For every claim/rider pair, state whether they are **still in the same section after the move**:
- the CM-16a/b split (friction preserves / temperature erases — ⛔ never cite CM-16 whole)
- CM-17: the relativistic no-go is a failure **of the sampler, not the thermodynamics**; ⛔ never "has no equilibrium"
- FDT-consistent noise **and** Newtonian kinetic mode required for every finite-temperature result
- CM-4: the retention advantage does not survive compute normalisation, retired as a compute claim
- CM-1: the loan called at ≈700 steps; boundedness, not plateau
- N46 designed-only scope on the emergent negative, **with** the counterexample clause
- every lifetime naming its metric; the Δ and ℓ_θ/Δ reporting rule
- the substrate-scope sentence; the score sentence in its measured form
- C-5 scale qualifiers **in-sentence** (dim 4, S¹, ≤5 seeds, laptop CPU)
- C-2 designed-verification vs learned-evidence labelling
⚠ **Report separation even when both halves survive somewhere** — a rider in Appendix G does not qualify a claim on page 2.

## Deliverable 5 — status of five known defects
Report present/absent, no fixes: `isrelationship` (a broken sentence inside Contribution 1) · `magnitutde` (misspelling) · the backtick `Fig.\`` figure reference · the `±0.05%` vs `±0.2%` parameter-match contradiction · whether the headline figure's label is now referenced by the text (`\ref{fig:pricelist}`).

## Method
Per-file greps with **positive controls** (the standing hazard: directory-level grep over `.claude/` silently returns nothing). Compare like with like — a number relocated to an appendix is **present**. ⛔ Flag nothing as lost without confirming it is absent from the *whole* file.

## Acceptance criteria
All five deliverables complete; every loss and every addition quoted with location; the rider-adjacency table covers every listed pair; zero edits to any file; anything needing a shell declared rather than faked.
