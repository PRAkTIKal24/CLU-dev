# neurreps-audience-scout — web-scout (the NeurReps audience's own literature: records + what each claims + how ours relates)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 37; Head directive 2026-08-20 — reframe V2 and V5 for the NeurReps audience).** Read `.claude/AGENT_PROTOCOL.md`, then this file. Read-only; one report: `.claude/outputs/neurreps-audience-scout.md`.

**DIAL DECLARATION: none — literature scouting + citation verification; no performance claim; no laundering control applies.**

## Why this exists
⚠ **Scope, per the Head's ruling of 2026-08-20: this scout serves the V2 → NeurReps reframe ONLY.** V5 reframes for PALM instead, and its audience is already scouted (`outputs/v5-scope-scout.md`). The V2 reframe must speak the NeurReps audience's vocabulary with **verified** citations. The Head supplied the audience profile (their own topic census): geometric deep learning / equivariance (~9 works) · neuroscience (~8) · topology and learning theory. Nothing may be cited from that census until its record is verified — and a framing that mis-describes a neighbouring paper is worse than no framing.

## Part 1 — verify the census works (records + BibTeX + one line of what each ACTUALLY claims)
**Geometric DL / equivariance:** Lie-algebraic network representations · disentangling images with Lie group transformations + sparse coding (Olshausen line) · sparse convolutions on Lie groups (van der Ouderaa, van der Wilk) · icosahedral projection for SO(3) reasoning (Klee, Biza, Platt, Walters) · SE(3)-equivariant moving-frame networks · group-invariant learning by fundamental-domain projections.
**Neuroscience:** cross-session neural population variability (Perich, Miller, Hennig) · information geometry of probabilistic population codes (Vastola, Drugowitsch) · level sets and invariance of neural tuning landscapes (Binxu Wang, Ponce) · invariance manifolds of visual sensory neurons (Sinz lab) · inter-areal interactions in mouse visual cortex (Allen Institute) · topological ensemble detection.
**Topology and learning theory:** whatever the census's third cluster resolves to — report what you find and label the resolution as your inference.
Per work: canonical record (venue, year, authors, DOI/arXiv), BibTeX in the house pattern with never-copy traps in `note`, **one sentence of what it claims in its own words (quote where possible)**, and a retrieval date. ⚠ Where the Head's description is a topic rather than a specific paper, name the best-matching primary work(s) and label the match as your inference, not the Head's.

## Part 2 — the bridge literature the reframes actually need (the highest-value part)
Two phenomenon classes are the natural bridge and must be verified before either draft leans on them:
1. **Continuous attractors and drift** — ring/head-direction attractor models; the drift-along-the-manifold result class; any work quantifying diffusion along a continuous attractor. This is the neuroscience-native name for our flat direction and its finite-temperature diffusion.
2. **Representational drift** — the cross-session population-variability literature (Perich/Miller/Hennig sits here) stated as its own phenomenon.
Per item: record + what is claimed + ⚠ **explicitly, what our result is NOT**: we make no claim about biological systems and no claim to model neural data. Give the writers the honest contrast sentence for each.

## Part 3 — vocabulary map (the deliverable the writers use most)
A two-column table: **our internal term → the audience's standard term**, with a one-line note where the mapping is exact vs approximate. Seed set to resolve (extend as the literature dictates): flat/neutral direction · coset register · mode-mass spectrum · spectral mass μ² · latch · erosion · the settle · retention half-life · designed vs emergent symmetry · the budget cube. ⛔ Where no standard term exists, say so — inventing an audience-sounding term that means something else there is the failure mode this map exists to prevent.

## Acceptance criteria
1. Every record primary-verified with a retrieval date; single-sourced items flagged; unresolvable items declared, never guessed.
2. Part 2 delivers the honest no-claim contrast sentence per item.
3. The vocabulary map marks each mapping exact / approximate / no-equivalent.
4. Standard `## Proposed handover updates` and `## Flags`.
