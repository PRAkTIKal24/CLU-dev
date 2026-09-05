# Task: dimension-aware-budget — pin the learned-capacity exponent ("probably geometry" → "geometry, proven") (w23)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/dimension-aware-budget.md` · **Branch:** `agent/experiment-engineer/dimension-aware-budget`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (§3.2 worktrees — 4 parallel engineer tasks this wave) · `.claude/outputs/designed-mechanism-learned-content.md` (the setup you re-run + its KEY METHOD FINDING) · `.claude/outputs/address-space-dimension-scaling.md` (the designed law `4·2^d` + the repaired packing bound) · registry N82

## Why
w22's primitive-decider gave **K_learned = {4, 8, 8, 32, 8} at d = {2,3,4,6,8}** — H-LEARNING rejected (the wall is geometry), but the exponent is **UNPINNED** (5-pt fit base 1.18, R² 0.26; d≤6 sub-curve base 1.64, R² 0.94) because the atom/parameter budget was scaled with K only, **not with d** — the d=8 point stalled on trivially-separated cells (site sep 1.838) with 4× more geometric room than d=6, an optimization failure near-provably (capacity cannot shrink with more room). One clean re-run converts the program's central claim into a referee-proof scaling law. **Cheapest task of the wave; laptop-scale.**

## Item 1 — the dimension-aware re-run
Same designed-atom-dictionary-mechanism / learned-content protocol as w22, d ∈ {2,3,4,6,8} (add d=5 if cheap), with **`min_atoms ∝ c^d`** (choose and justify `c`; verify per-point budget adequacy the way w22 did for d=2's K=8 failure — a failed K must fail with a demonstrably adequate budget, param count reported). Strict-0.9 criterion unchanged. ≥3 seeds at the K-frontier points.

## Item 2 — the fit
Fit `K_learned(d) = A·B^d` (and the repaired `d_eff` form as an alternative). Deliverable: the pinned base with CI + R², the designed ceiling `4·2^d` on the same axes, and the **designed/learned tax curve** (w22: ¼ → ⅛ → ≤1/16, widening — does dimension-aware budgeting arrest it?).

## Item 3 — the d=8 verdict
Either d=8 now clears ≥ its d=6-implied value (monotone restored ⇒ the optimizer-failure diagnosis is CONFIRMED and the law stands), or it stalls again under an adequate dimension-aware budget — in which case run the w22 stall diagnostic (where does the writer stop using the room?) and report the failure mode precisely. **A second stall under adequate budget is a real ceiling candidate and must be reported as such, not excused.**

## Acceptance
The K_learned(d) table + fitted law + tax curve + the d=8 verdict, budget-adequacy evidence per point, ≥3 seeds at frontiers. **Pre-register your expected K per d before running** (w22's corrected reading predicts ≈ 4 → 8 → ~16 → 32 → ≥64). Never quote a clean `A^d` law unless the fit now earns it (CM-22). Tests green.
