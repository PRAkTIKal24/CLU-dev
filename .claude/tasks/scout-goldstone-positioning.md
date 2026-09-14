# Task: scout-goldstone-positioning — V2's literature position (Welling deep-extract + SSB-in-DL sweep)

- **Agent:** `web-scout` · **Output:** `.claude/outputs/scout-goldstone-positioning.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md`, `.claude/handover_context.md` (§1, §8), `.claude/brainstorm_log.md` (Threads 2 & 5 — what we intend to claim), and the local PDF `docs/2605.14685v1.pdf` (Iqbal, Keller, Song, Miyato, Welling — "Spontaneous symmetry breaking and Goldstone modes for deep information propagation", 28pp). Extract text via `/Users/user/opt/miniconda3/bin/python` + `pypdf` if needed.

## Why
V2's short paper claims specific novelty beyond Welling et al. Before we sink a month of theory+experiments, verify each claimed differentiator is actually open, and map the surrounding literature so the related-work section writes itself.

## Acceptance criterion
For each of our five claimed differentiators (below): a verdict — **OPEN / PARTIALLY TAKEN (by whom, where) / TAKEN** — with quotes/equations as evidence. Plus a bibtex-ready related-work map.

## Sub-tasks
1. **Deep-extract the Welling paper (local PDF, read ALL of it):** their exact setting (architectures, symmetry groups used, what "SSB" means operationally there), main theorems/claims, experiments, metrics, and — critically — their stated future work (that's where we're most likely to collide). Summarize their formalism precisely (notation, key equations).
2. **Verdict our five differentiators** (from handover §8 / brainstorm Thread 2): (i) Hamiltonian/energy-space account (theirs: equivariant feedforward/RNN layers); (ii) dissipation interplay — "friction kills Goldstone momentum but cannot erase Goldstone displacement"; (iii) multiplicity/pseudo-Goldstone *engineering* — dim(G/H) as designed channel count + graceful degradation with half-life ∝ 1/mass²; (iv) EFT organization of corrections (HEFT/SMEFT-style parameterizations of latent dynamics, coset/nonlinear-sigma-model layers); (v) causal-bound c on Goldstone signal propagation.
3. **Broader sweep** (arXiv/Semantic Scholar/OpenReview): SSB in deep learning; Goldstone/gapless modes in NNs; flat directions & mode connectivity of loss/energy landscapes as memory; nonlinear sigma models / coset manifolds in ML; symmetry-protected information propagation; pseudo-Goldstone analogs; Hamiltonian NN + symmetry work (incl. anything post-dating our CHLU paper). Also: anyone citing the Welling paper already?
4. **Adjacent must-knows:** LyTimeT (already cited in CHLU paper), symplectic RNN line (Chen 2020, Erichson 2021), Noether-in-ML papers (e.g., Noether networks / conservation-law learning).

## Output format
(1) Welling paper précis (their claims, formalism, experiments, future-work list); (2) differentiator verdict table with evidence; (3) related-work map grouped by theme with 1-line relevance each; (4) risks — closest prior art that could sink a review; (5) bibtex block; (6) `## Proposed handover updates`.
