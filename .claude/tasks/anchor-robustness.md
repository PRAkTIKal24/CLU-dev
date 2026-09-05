# Task: anchor-robustness — harden the erosion cure + test it as the memory-fidelity intervention (critiques P11/V2.4 + P14/V1.3)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/anchor-robustness.md` (+ figures in `.claude/outputs/anchor-robustness/`)
- **Read first:** protocol · `.claude/critique_register.md` (V2.4: "cure validated at one setting, two seeds"; V1.3: "'fixable' asserted, not demonstrated") · `.claude/outputs/sleep-erosion-study.md` (the law + cure) · `.claude/outputs/v1-hopfield-stress.md` (the regime map + "gap driver = CLU-EBM memory fidelity") · `.claude/outputs/minus-the-physics.md` (CD-vacuum survival cross-link) · claims matrix CM-6/CM-8.
- Repo **read-only**; scratch in `.claude/scratch/anchor-robustness/`; laptop-CPU.

## Items
1. **P11 — anchor robustness sweep:** λ_anchor ∈ {1, 10, 100} × 5 seeds × epochs {300, 1000, 3000} × testbeds {exp-d SO(2) (erosion-prone), Exp-B (immune control), one lattice config}. Metrics: vacuum integrity (μ² spectrum + ring-vs-max), noise rejection, wake MSE. Deliverables: the operating envelope ("anchor holds for λ∈[..] up to N epochs; costs/benefits elsewhere") + confirmation the demarcation law presents as THEORY (wake-invisible flat directions), with the Exp-B control behaving as predicted.
2. **P14 — the memory-fidelity intervention:** take 2–3 losing cells of the v1-hopfield-stress regime map (correlated patterns / capacity stress) and retrain the CLU-EBM memory with the anchor (+ longer/persistent training as a second arm). Does ANY cell move materially (retrieval accuracy, basin integrity, compute-savings recovery)? Deliverable: the honest sentence V1 needs — *"memory fidelity is improvable by [X] at [cost]"* or *"the anchor does not close the Hopfield gap — fidelity remains the named open workstream."* Either is publishable (charter C-9).
3. **(cheap, cross-link)** one broken-volume arm (from `chlu/core/twins.py`) trained WITH the anchor: does the anchor rescue the non-symplectic vacuum too, or is CD-robustness genuinely a volume-conservation payoff (sharpens CM-1/CM-6 wording)?

**Report:** numbers + error bars, verdicts per item, flag-provenance per §5 (anchor flag, epochs, sleep_frequency are exactly the erosion-sensitive knobs — table every run).
