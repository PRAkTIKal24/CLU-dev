# Task: philosophy-synthesis — the idea→theory→experiment→revision ledger (companion to HEP_primers)

- **Agent:** `physics-theorist` · **Output:** `.claude/outputs/philosophy-synthesis.md` (markdown, GitHub-preview-clean math — same audience rules as HEP_primers: AI experts without deep HEP; this goes to EXTERNAL readers for feedback)
- **Read first (this is a synthesis task — the corpus is the input):** `.claude/outputs/HEP_primers.md` (the companion doc — mirror its tone/structure/status-tags; cross-reference its sections by name) · F5 v1.1 (`formalism-note.md`) · `brainstorm_log.md` (the idea provenance — attributions matter) · handover §10 · ALL w1–w3 output reports (mass-spectrum-peek, v2-so2-build, v2-full-runs, v1-l0-gate, v1-pivot, v3-lattice-build, gamma-field-build, generative-studies, mo-deep-read, di-bernardo-skim, scout reports).

**Purpose (Head's words, honor them):** make it possible for an external reader (and us) to see, per idea: what was philosophically planned → what the math/theory said → what results we therefore expected → what experiment we ran → what we measured vs expected → how that validated/changed the idea → what accidental/emergent findings appeared and what they imply for the math AND the philosophy. This document will drive: positioning of each short, what the ICLR paper needs to be a rockstar, and the shortcomings that should shape ICLR eval-task/baseline choices.

## Structure (one chapter per idea-thread; use this exact 7-part rubric in each)
For each of: **(1) the mode-budget/Goldstone memory idea (Thread 2), (2) the mass-as-allocator idea (Thread 5), (3) the trash-region/trinity idea (Thread 1), (4) the boost/cascade/test-time-compute idea (Thread 3, incl. the squeeze-retry failure and the pivot), (5) the lattice/scale-by-mass-and-size idea (Thread 5/V3), (6) the sampler-thermodynamics thread (FDT/Prop-9), (7) the training-dynamics thread (Lyapunov degeneracy, sleep erosion, PCD null):**

1. **Philosophy** — the original intuition, attributed (Head/Hub/theorist), quoting brainstorm_log where it crystallized; link to the matching HEP_primers section.
2. **Theory** — the formal statement (F5 Prop/Cor numbers), with the one-equation essence.
3. **Expected** — what the theory predicted we would measure, stated BEFORE-the-fact where the record shows it (the falsifiables lists).
4. **Experiment** — what was actually run (task, scale, seeds).
5. **Measured vs expected** — the numbers side by side; agreement quality; where reality deviated (e.g., emergent +13% bias, learned holes over-damped 2.6–13×, EP prefactor).
6. **Verdict & revision** — validated / refuted / reframed, and how the idea changed (e.g., "half-life ∝ 1/mass²" → exact-with-scope; "boost retries recover answers" → parked, headline pivoted; "FDT explains imbalance" → refuted, conditional on banding; "M self-organizes" → doctrine: designed-in or induced).
7. **Emergent discoveries** — what we stumbled on (sleep erosion; charge-flow-as-communication; "objectives must not legislate friction"; latch-vs-erase distinction; metric bifurcation; EP spectroscopy) and what each implies mathematically AND philosophically (e.g., for the trinity: destruction has an *optimum*, and friction alone cannot destroy sustained content — Shiva needs Brahma's noise or a tilted landscape).

## Closing chapters
- **Scorecard table:** idea × {validated / validated-with-scope / reframed / refuted / open}, one line of evidence each.
- **Positioning consequences:** 3 short paragraphs — what each short's strongest honest claim is, given the above.
- **ICLR gap analysis:** what a rockstar long paper still lacks — which claims have only synthetic evidence (→ the industrial eval), which baselines are missing (coRNN/LEM, Mo's S¹ task, capacity-stressed Hopfield), which levers are untested at scale (banding on real data, fdt-under-banding, reversible O(1) memory) — mapped to concrete eval-task/baseline recommendations.
- **Shortcomings, plainly:** the honest-limits list (single-seed items, laptop scale, MQAR-only memory results, narrow mass spectra, erosion horizon, no positional equilibrium in generative sampling).

## Rules
Every number traceable to a report (cite the output file + section). Status-tag discipline as in F5/HEP_primers. No new experiments, no code. Attribute ideas honestly (Head's raw intuitions vs Hub formalizations vs agent discoveries — the brainstorm log is the provenance record). Length: as long as it needs to be, but every sentence earns its place; this is the document we hand to a skeptical outsider.
