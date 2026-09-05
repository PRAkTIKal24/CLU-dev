# pj-referee-v5 — paper-referee (review the Head's V5 submission version as a PALM reviewer receives it)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 53; Head directive 2026-08-22).** ⛔ **Mechanical precondition: `.claude/NIPSsubmission/v5-palm/pj_sub.pdf` exists on disk.** Read `.claude/AGENT_PROTOCOL.md`, then this file. One report: `.claude/outputs/pj-referee-v5.md`.

## ⛔ Two constraints
1. **Edit nothing.** `pj_sub.tex` is the Head's own file.
2. ⛔ **Do NOT read `outputs/pj-fidelity-v5.md`** (a parallel fidelity audit). **Independence is the point** — findings the two passes reach separately are the ones the Head can trust.

**DIAL DECLARATION: none — adversarial review; no performance claim; no laundering control applies.**

## What you are reviewing
`.claude/NIPSsubmission/v5-palm/pj_sub.pdf` — the Head's own condensation (2,684 words, ~30 % of the clean base) for **PALM @ NeurIPS 2026** (*"Personalized, aligned, long-term memory for AI systems"*), non-archival, double-blind. ⚠ **Track is undecided** (short 4 pp vs full 9 pp, both excluding references and supplementary) — so **do not review against a page limit; review the paper, and say which track it reads as.**

## The reviewer you are
The composite PALM reviewer, from the venue's own CFP and its scouted proxy room (`outputs/v5-scope-scout.md`, `outputs/audience-refresh-2025-2026.md`): communities spanning ML, NLP, agents, HCI, cognitive science, neuroscience, privacy, security and safety; CFP topics that explicitly name *"memory update and deletion tests"* and *"right-to-be-forgotten mechanisms"*; a room that **accepts theory papers and negative results**, that now uses physics vocabulary, and whose proxy set contains **almost nothing on deletion** — i.e. this paper enters near-empty ground. ⭐ The pool includes a membership-inference/privacy specialist, the author of the nearest published retention-dial architecture, and a brain–language-model alignment researcher. Reflex objections here: toy scale, "this is not a deployed memory system", and "what does a physical mechanism buy over a TTL policy?"

## Review it on
1. **Does it stand alone at this compression?** Can a reader who has never seen the long version follow claim, evidence and scope? Name every missing step, undefined term, or number without its setup.
2. **The three contributions, correctly bounded:** the retention dial with a computable optimum; scoped retention (the vault, and the confinement result on the emergent arm); a structural deletion guarantee stated as the composition claim. ⛔ Flag any sentence that reads as: a system-level or certified-unlearning guarantee; the designed-only contrast number generalizing; the vault itself (rather than its laws) transferring to the emergent arm; a mechanics result reading as a value result.
3. **The TTL comparison.** This room's baseline *is* expiry policy. Is our own fired laundering control — a boolean TTL flag being near-indistinguishable from physical decay against an exact adversary — present and prominent, with the honest leakage sentence beside it? Its absence or softening is a MUST-FIX; its presence is the paper's credibility anchor with the privacy specialist in the room.
4. **Positioning** — against the agent/LLM long-term-memory literature this audience knows, and the nearest retention-dial neighbour. Flag stale or missing placement.
5. **Evidence sufficiency** — per claim, is the evidence in the paper the reviewer actually reads?
6. **Figures and captions** — do they show the claim, are they legible, do captions carry scope (seeds, dimension, budget, single-seed labels)?
7. **Scope honesty** — the CFP names negative results as an accepted type: are the negatives and limitations present and prominent, or buried?
8. **Track fit** — short or full? Say which it reads as and why; this feeds a live Head decision.

## Deliverable
Itemized **MUST-FIX / SHOULD-FIX / NICE** with locations and concrete failure scenarios; a **simulated accept/reject verdict**; **the three sentences a hostile reviewer would quote back**; and a short section on **what the compression gained and lost** in a reviewer's eyes. Standard `## Proposed handover updates` and `## Flags`.

## Acceptance criteria
1. Every MUST-FIX carries a location and a concrete failure scenario.
2. The verdict and the track-fit reading are both stated plainly.
3. Zero edits to any file except your report; state that `pj_sub.tex` was not touched.
