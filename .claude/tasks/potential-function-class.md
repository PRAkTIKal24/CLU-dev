# Task D: potential-function-class — is the learned-landscape failure EXPRESSIVITY or SUPPORT STRUCTURE? (w21)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/potential-function-class.md` · **Branch:** `agent/experiment-engineer/potential-function-class`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/learned-landscape-write-read.md` (the design-freedom ladder this re-runs) · `.claude/outputs/clu-controller-spec.md` §C3 + §4 (`AtomDictionaryPotential`) · `.claude/outputs/relaxation-addressing-theory.md` §Item 5 (Ramsauer)

## Why — two hypotheses that make OPPOSITE predictions
w20 measured that a learned landscape loses everything a designed one provides: **no learned rung clears 0.9 at both K=4 and K=8** (5 seeds), write locality goes **0.000 → 2.9e-2…5.0e-1**, and one subsequent write destroys the best rung (**strict 1.000 → 0.000**). Every learned rung used **`PotentialMLP`**. Two explanations are on the table:

| | hypothesis | prediction for a **transformer/attention** potential | prediction for an **atom-dictionary** potential |
|---|---|---|---|
| **H-EXPR** | the MLP is too weak to represent a good landscape (Head's hypothesis) | **fixes it** — more capacity, better landscape | also fixes it |
| **H-SUPP** | the failure is **global support**: any weight update moves *every* stored item (theorist C3; CM-6 erosion) | **fails, possibly worse** — attention is *more* global than an MLP | **fixes it** — atom writes are local by construction |

**These diverge on exactly one arm**, which is what makes the experiment worth running. **The transformer arm is the discriminator.**

## Item 1 — four function classes, matched parameters
Re-run the `exp-learned-memory` design-freedom protocol with the potential's **function class** as the swept variable, everything else fixed:
1. **`PotentialMLP`** — the w20 baseline (global support).
2. **⭐ Transformer/attention potential** — global support, higher capacity. `V: R^d → R`. Two natural constructions; **run at least (b) and say which you used**:
   (a) treat the `d` coordinates as `d` tokens, self-attend, pool to a scalar;
   (b) **attention over a learned memory codebook** — ⭐ note that `V(q) = −logsumexp(β⟨q, k_i⟩)` **is exactly the modern-Hopfield energy**, which makes this arm a direct in-framework test of *attention-as-memory versus atoms-as-memory*, and the concrete form of the Ramsauer question.
3. **`AtomDictionaryPotential`** — local support (theorist §4: `α‖q‖² + Σ_i −A_i exp(−‖q−c_i‖²/2s_i²)`), **learned** amplitudes/centers/widths. This is the MVC-0 substrate.
4. **Designed** — the w20 reference rung, unchanged, as the ceiling.

⚠ **Match parameter count across all four** (tolerance ≤5%, reported). An unmatched comparison settles nothing. Report the match table.

## Item 2 — the discriminating measurements
For each class, at **K ∈ {4, 8}**, **≥5 seeds** (w20 showed single seeds mislead here — its own single-seed answer was overturned by its 5-seed check):
1. **strict retrieval** (the leak-immune value criterion — **not** classification);
2. ⚠ **blank control over the STRONGEST read in use**, mandatory on every cell. w20's method finding: under a learned `V`, nearest-centroid blanks score **0.992–1.000 on landscapes with nothing stored**, because a deterministic payload channel makes a **1e-4** address leak a perfect item code. **A cell without a passing blank is not a measurement.**
3. **⭐ cross-write interference** — write A, then write B, re-read A. **This is the H-EXPR/H-SUPP discriminator**; report corruption per class against designed (0.000) and MLP (2.9e-2…5.0e-1).
4. **support radius, measured not assumed:** perturb θ by a single write and measure `‖δV(q)‖` as a function of distance from the write site. **Report the decay curve per class.** This is the mechanism behind (3) and the most transferable number in the task.

## Item 3 — does the design-freedom curve move?
Re-run w20's rung ladder with the best-performing class. **The w20 headline was "minimum designed structure = essentially all of it."** Does a better function class move that point? **This is the task's headline result either way.**

## Item 4 — cost
Report params, wall-clock and FLOPs per class. The atom dictionary is likely cheapest and the transformer dearest; if a class wins on fidelity but is 10× the cost, that belongs in the same table.

## Acceptance
The matched-parameter table, per-class fidelity + blank + interference + support-decay at ≥5 seeds, the re-run design-freedom point, and cost. Tests green. **State explicitly which of H-EXPR / H-SUPP the data supports, or that it supports neither.**

⚠ **Pre-register the H-EXPR/H-SUPP predictions per arm before running.** ⚠ **A transformer potential that beats the atom dictionary would be a genuinely surprising result and must be checked hard before it is reported** — it would contradict the theorist's C3 argument and CM-6. Conversely, **if all learned classes fail, that is a strong program-level result** (the function class is not the problem; structure is) and should be stated as such rather than buried.
