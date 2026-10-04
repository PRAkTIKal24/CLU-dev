# Task: scout-adaptive-compute-prior-art — is V1's escalation cascade actually open territory?

- **Agent:** `web-scout` · **Output:** `.claude/outputs/scout-adaptive-compute-prior-art.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md`, `.claude/handover_context.md` (§8), `.claude/brainstorm_log.md` (Threads 3 & 5 — the mechanism we intend to build).

## What we intend to claim (verify it's open)
V1 = an **energy-gated retrieval/compute cascade** in a stacked CLU: query relaxes in the smallest shell → **residual energy** (height above learned floor) = confidence signal → if high, retry via **boost/squeeze** (symplectic squeeze transformations — non-compact Sp(2d) elements; structure-preserving ⇒ "retries that can't destabilize") with mass-weighted differential response → if still high, **shell-jump** to a deeper/heavier sector. Calibration of the energy signal is a *training-time objective* (EBM margin loss + learned threshold).

## Sub-tasks — four prior-art fronts, verdict each (OPEN / PARTIALLY TAKEN / TAKEN + evidence)
1. **Adaptive computation & early exit:** ACT (Graves), BranchyNet, PABEE, CALM (Schuster et al.), Mixture-of-Depths, speculative decoding, cascade models (model cascades in vision/LLM serving). What confidence signals do they use (softmax entropy, learned halting, verifier)? Has anyone used an **energy/EBM-based confidence signal** for escalation?
2. **Energy-based confidence & calibration:** EBM uncertainty/OOD literature (energy scores for OOD — Liu et al. 2020 line), margin-based EBM training (LeCun tutorial lineage), conformal + energy. Is "learned in-training calibration of energy as answer-quality, generalizing off-distribution" claimed anywhere?
3. **Squeeze/symplectic transformations in ML:** squeezing (quantum-optics-style) in classical ML; symplectic/metaplectic attention or transforms; Sp(2d)-equivariant nets; hyperbolic-rotation reparameterizations; LorentzNet-adjacent Minkowski attention (we know LorentzNet — check who's built on it 2023–2026). Anyone using **inference-time structure-preserving transforms as retrieval retries**?
4. **Hierarchical memory / multi-scale retrieval in dynamical or associative-memory models:** modern Hopfield networks (Ramsauer et al.) & energy-based associative recall, hierarchical Hopfield/attractor models, fast-weight programmers, multi-timescale RNNs (clockwork RNN etc.), anything doing "cheap store → escalate to expensive store" with a principled gate.

## Also
- Grab the standard **associative-recall / selective-copy** synthetic-task definitions used in the SSM/attention literature (H3/Hyena/Mamba lineage) — exact task specs + typical difficulty knobs, so our gate experiment is comparable to known setups.

## Output format
(1) Verdict table per front with closest-neighbor citations + one-line distinction; (2) the 5 most dangerous "reviewer will say you're X" papers, each with our honest differentiation (or a warning that we don't have one); (3) associative-recall task specs; (4) bibtex; (5) `## Proposed handover updates`.
