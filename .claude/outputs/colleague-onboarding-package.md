# colleague-onboarding-package — paper-writer report

**Status: DONE.** V2 is reviewable by the HEP-theorist colleague. Two deliverables written: `AI_for_physicists_primer.md` (durable, reusable) + `V2_colleague_cover.md` (this-send reading guide + ask). Package to send = **V2 draft PDF + F5 note PDF + `v2-symmetry-deepdive.md` + AI primer + cover note** (all exist on disk).

**Task + acceptance criterion:** produce (1) an AI-for-physicists primer covering *exactly* the ML concepts the V2 draft uses, physics-native, analogy-driven, no new physics, reusable; (2) a cover note giving reading order + specific asks (real open questions) + settled-vs-open + housekeeping + authorship context; both HEP-legible; flag whether the deep-dive needs de-internalization. **Do not edit the V2 draft or the deep-dive** (companion docs only) — honored, no companion doc touched.

## What I did
- Read (in protocol order): `AGENT_PROTOCOL.md`; the Positioning Charter (C-1…C-10, philosophy-synthesis.md §416+); `claims_matrix.md` (v1.9 — canonical constants §1, CM-1…CM-17, negatives); the task file. Then the sources: `papers/v2-short/draft.md` (full — all 5 §§ + Appendices A–J), `outputs/HEP_primers.md` (the inverse doc, register-matched), `outputs/v2-symmetry-deepdive.md` (full — S1–S13, O1/O2/O6/O7, V4 seeds, §7bis running-decay-constant), `papers/f5-note/f5-note.tex` (structure + proposition list).
- Wrote **Deliverable 1** `outputs/AI_for_physicists_primer.md`: 7 sections, one per ML concept the draft leans on, each with a **Physics analogy / What it actually is / Look-up / → In the V2 draft** block (the inverse of HEP_primers' Philosophy/Look-up/Math/→CLU). Covers exactly: (§1) recurrent nets + exploding/vanishing gradients + LSTM gates; (§2) the CLU as a learned separable Hamiltonian evolved by velocity-Verlet + damping — led with, per task; (§3) "trained"/learned $V_\theta$ + wake–sleep/CD as an EBM/Boltzmann rule; (§4) LSTM/LEM/coRNN baselines + "well-trained baseline" fairness; (§5) RMSE (rad), autonomous-retention protocol, map-step, per-step-compute normalization caveat; (§6) Mo's single-exponential lifetime law, overdamped regime, the containment headline; (§7) ML4PS/NeurReps non-archival frame + the verification-vs-evidence discipline.
- Wrote **Deliverable 2** `outputs/V2_colleague_cover.md`: reading order (primer → draft PDF → deep-dive → F5 note), the specific asks (O1/O2/O6/O7 + non-abelian V4 seeds SO(3)→SO(2)/torus/custodial + running-decay-constant), settled-vs-open (claims-matrix locked claims vs Appendix-F negatives), verification-vs-evidence calibration, deep-dive housekeeping pointer + de-internalization recommendation, authorship/CLU-CHLU/anonymization context.

## How I verified (charter/matrix compliance, self-checked)
- **Naming (rule 7):** CLU continuity sentence present verbatim in both docs ("CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)"). Inertial $M$ vs spectral $\mu$ kept distinct; no bare "mass" (primer §2 states the distinction explicitly).
- **Hermetic citations (rule 5 / C-8):** only J&P 2026 (3rd person), the F5 note as "Anonymous, 2026" (3rd person), Mo 2026, Di Bernardo 2025, and standard published ML/ChPT lit are referenced. **No other program short is mentioned or assumed to exist.**
- **C-2 (verification vs evidence):** the discipline is taught in primer §7 and used to calibrate the cover note's "settled vs open" — designed-testbed = verification (machine-precision), learned-system = evidence (2–15% with predicted deviations).
- **C-5 (scale qualifiers):** every quantitative aside carries dim 4 / ≤5 seeds / $S^1$ testbed / laptop-CPU in-sentence.
- **CM-3 (forbidden):** no "energy is a better confidence signal" claim anywhere (V2 doesn't touch it; confirmed absent). CM-1/CM-4/CM-6/CM-15/CM-16/CM-17 wordings inherited from the draft where referenced, not re-derived.
- **Number provenance (rule 3):** every number in both docs is quoted from `papers/v2-short/draft.md` (which carries the flag-provenance tables in its Appendix A) or the deep-dive; none adjusted, rounded, or invented. Where a number would be needed but isn't settled, I pointed at it as an open question rather than improvising.
- Package file existence confirmed: `papers/v2-short/draft.pdf` and `papers/f5-note/f5-note.pdf` both present.

## Findings / editorial notes for the Hub
- **De-internalization verdict (task item 4):** the deep-dive needs a *light* pass for wider circulation (strip "Proposed handover updates" + "Open questions for the Head" sections, soften wave-number/task-slug refs), **but for this send the cover-note pointer suffices** — internal refs (Head/Hub/wave-N/`.claude/` slugs) are transparent and won't mislead a trusted co-author. I **recommended, did not edit** (per task). One substantive internal item I flagged *to* the colleague as real-not-noise: the deep-dive's S6/R3 identifies a proven error in the F5 note (kinetic-isotropy clause) that is a correction in flight.
- **Open editorial question for Hub/Head:** the cover note tells the colleague his co-authorship is "the Head's call." If the Head wants a firmer or softer posture on co-authorship, that one paragraph (cover §5) is the place to adjust.
- **Second editorial question:** the cover note cites the F5 note as "Anonymous, 2026" per the anonymization posture. If a title/authors get chosen before this package is sent, cover §1/§5 and the reference string need a one-line update.
- The primer is written to be **reusable for any future physicist collaborator** (durable), not V2-specific in its teaching sections — only the "→ In the V2 draft" blocks are V2-specific and are cleanly separable.

## Git footprint
None — research-only task, all artifacts under `.claude/outputs/`. No tracked code touched.

## Proposed handover updates (for the Hub)
- New durable asset: `outputs/AI_for_physicists_primer.md` — the inverse of `HEP_primers.md`, reusable onboarding for any HEP/physics collaborator. Worth indexing alongside HEP_primers in the handover doc's companion-docs list.
- `outputs/V2_colleague_cover.md` is send-specific (V2 + this colleague); regenerate per vertical if other verticals go out to physicist reviewers.
- Deep-dive de-internalization is queued as a *recommended, unowned* light pass — if the Head greenlights wider circulation, convert to an explicit doc-curator task (strip handover/open-questions-for-Head sections; soften wave/slug refs).
