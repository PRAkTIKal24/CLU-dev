# PREREG — dimension-aware-budget (w23)

Written BEFORE running the discriminator harness with the dimension-aware atom floor.
Author: experiment-engineer. Base commit: `7ff0651` (local main). Branch:
`agent/experiment-engineer/dimension-aware-budget`.

## What is being measured
`K_learned(d)` = largest item count a LEARNED atom-dictionary CLU clears at strict
≥ 0.9 (value criterion, blank-controlled, ≥3 seeds at frontier) on a d-ball address
space, re-running the w22 designed-mechanism/learned-content protocol but with the
atom budget floor scaled with the address DIMENSION: `n_atoms = max(atoms_per_item·K,
min_atoms, round(base·c^d))`. w22 scaled with K only (fixed floor) and the d=8 point
stalled on a trivially-separated cell (site sep 1.838) — an optimizer/budget artifact,
since capacity cannot shrink with more geometric room.

## Choice of c (pre-committed)
`c = 2.0` — primary. Justification: (i) matches the DESIGNED capacity growth
`K_designed = 4·2^d`, so the atom budget holds atoms-per-designed-cell ≈ constant
across d; (ii) compensates the ~per-dimension geometric thinning of the fraction of
`N(0,init_scale)` atoms that land near a stored site (radius ~R) in the (d+1)-ball.
`base` chosen for per-point adequacy AND laptop-scale compute (set after a timing
probe; anchored so the d=2 floor ≥ the w22-verified-adequate value where feasible).
Adequacy is verified empirically at each reported point (a failing K must fail with a
budget whose further increase does not flip the verdict).

## Pre-registered K_learned(d) predictions

Two competing hypotheses (both registered; the survivor is evidence, the failure is a
finding — protocol §5):

| d | H-WEAK (w22 corrected reading, base≈1.6) | H-STRONG (constant ¼ designed-tax, base=2.0) |
|---|---|---|
| 2 | 4  | 4   |
| 3 | 8  | 8   |
| 4 | 16 | 16  |
| 5 | 16 (boundary; 16 or 32) | 32 |
| 6 | 32 | 64  |
| 8 | 64 (≥ d=6 value → monotone restored) | 128–256 |

- **Primary registered prediction = H-WEAK point sequence: {4, 8, 16, 16/32, 32, 64}.**
  This is the task's stated corrected reading (≈ 4 → 8 → 16 → 32 → ≥64). Fitted base
  **B ∈ [1.55, 1.75]**, exponential **R² ≥ 0.90** (the dimension-aware budget ARRESTS
  the w22 noise: R² rises from 0.26 at the 5-pt fixed-budget fit toward ≥0.90).
- **H-STRONG** (base 2.0, K_learned = 2^d) is the alternative: registered so that if the
  dimension-aware budget lifts d=6→64 / d=8→≥128, that is a measured finding, not a
  post-hoc rescue.

## The d=8 verdict (Item 3, pre-committed decision rule)
- **CONFIRM optimizer-failure diagnosis** iff d=8 clears **≥ 32** (i.e. ≥ the d=6 value
  → monotonicity restored). Then the law stands and the w22 non-monotone dip (d=8→8)
  is confirmed a budget artifact.
- **REAL-CEILING candidate** iff d=8 stalls at ≤ 16 **under a demonstrably adequate
  budget** (increasing atoms further does not lift it, write loss converged). Then run
  the w22 stall diagnostic (where does the writer stop using the room?) and report the
  failure mode — NOT excused as compute.

## Tax curve (designed/learned) prediction
w22 (fixed budget): ¼ → ⅛ → ≤1/16 (WIDENING). Registered prediction under the
dimension-aware budget: the tax **stops widening / narrows** — designed/learned holds
≈ 4 (H-STRONG) or drifts to ≈ 4→8 but does NOT keep doubling (H-WEAK). A tax that keeps
widening to 1/16+ despite adequate budget would falsify "budget-artifact" and support a
learned-content ceiling that grows slower than the designed ceiling.

## Falsifiers
- If the fitted base stays < 1.3 AND K_learned does not grow ≥ +2 rungs over the sweep
  → H-LEARNING (flat wall) resurrected; the primitive claim is in trouble.
- If d=8 stalls at ≤ 16 with adequate budget → real ceiling at d=8; the clean `A^d` law
  is rejected and must be reported as capped.
- If the fit is non-monotone/weak (R² < 0.8) even with the dimension-aware budget → the
  law is NOT pinnable on this harness; report "trend, not law" (do not quote a base).
