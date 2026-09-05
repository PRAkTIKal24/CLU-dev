# Task: retry-compute-study — the accuracy-vs-compute curve, done properly (the novelty flagship) (w23)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/retry-compute-study.md` · **Branch:** `agent/experiment-engineer/retry-compute-study`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (§3.2 worktrees — 4 parallel engineer tasks this wave) · `.claude/outputs/hopfield-capacity-benchmark.md` §retry (+46.9pp at ×1.5 compute, −38.3pp blank guard — the numbers this task promotes) · `.claude/research_roadmap.md` v0.6 thread (a) · CM-23 (approved retry wording)
- **⭐ The Head's ruling:** extract performance results any way we can, explained or not. Retry is the program's most promising novel-property-with-teeth and sits in the field's hottest area (test-time compute). This task turns one demo point into a defensible curve + a created-benchmark spec anchored to an existing task.

## Why
w22's retry was a single rung: one boosted re-relaxation on the low-confidence half, one task setting. A referee will ask: (1) is the *curve* real (multiple compute budgets, monotone?), (2) is it the *physics* or would any stochastic-restart heuristic match it, (3) can a feedforward baseline given the SAME extra compute draw the same curve? Nobody has run these controls.

## Item 1 — the retry ladder
On the w22 retrieval protocol (MNIST first; CIFAR + φ-space as stretch if `phi-read-in` lands in time — coordinate via output files, do not share a worktree): retry budgets k = 0,1,2,4,8 boosted re-relaxations, **confidence-gated** (retry only below-threshold reads; threshold swept). Report accuracy vs total compute (relaxation-step count = the honest unit; wall-clock secondary) at ≥2 load levels (M) and ≥2 noise levels (σ).

## Item 2 — the controls (each pre-registered win/tie/lose)
1. **Ungated retry-all** (quantifies the gate's contribution; w22's −38pp says gating is load-bearing — show the curve).
2. **Ensemble-of-k-reads**: k independent starts, best-confidence answer — the fair "is it the boost or just k tries?" rival.
3. **Random-kick retry**: replace the boost with an equal-energy random perturbation — is the *Lorentz boost* doing anything a kick doesn't? (N1 precedent says test this honestly.)
4. **Feedforward matched-compute**: the trivial NN baseline given the same extra budget (k-fold ensemble / k nearest neighbours vote). The claim "an accuracy-vs-compute curve feedforward memories cannot draw" (CM-23) survives ONLY if this line is flat-or-worse — measure it, don't assert it.
5. **Hopfield-with-k-steps**: extra relaxation steps for the closed-form line.

## Item 3 — the created-benchmark spec
One page in the report: task definition (retrieval under load+noise), the metric (**accuracy at compute budget c**, curve dominance), the mandatory baselines (Items 2.2/2.4), and how an external method plugs in. Anchor every choice to the existing protocol (created benchmarks get discounted; anchored ones less so).

## Acceptance
The accuracy-vs-compute figure with all six lines (CLU-gated + 5 controls) at each (M, σ) cell, the threshold sweep, the spec page, pre-registrations stated. ⚠ If ensemble-of-k or the random kick matches the boosted ladder, the novelty claim dies — report it plainly and keep the curve (a tie against controls is still a capability; a falsified mechanism-attribution is a registry entry). Tests green; x64-safe rollout reused.
