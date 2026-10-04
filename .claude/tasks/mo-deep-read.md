# Task: mo-deep-read — verify V2's distinctness from Mo 2026 before theory work starts

- **Agent:** `physics-theorist` · **Output:** `.claude/outputs/mo-deep-read.md` (no code; scratch in `.claude/scratch/mo-deep-read/`)
- **Read first:** `.claude/AGENT_PROTOCOL.md`, `.claude/outputs/scout-goldstone-positioning.md` (the threat assessment), `.claude/outputs/formalism-note.md` §3–§4 (our claims), brainstorm log Thread 2 + Wave-1b update.
- **Target paper:** Mo 2026, "Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks," **arXiv:2605.03338** (fetch via ar5iv/arXiv HTML; WebFetch is available to this agent — if the HTML is unavailable, say so rather than working from the scout's summary alone).

**Why:** Mo partially takes V2's differentiator (iii) (≥dim(G/H) zero-Lyapunov modes + pseudo-gap→lifetime). V2's reframe (constructive allocation + Hamiltonian curvature mass-law) is only safe if Mo's framework provably cannot express our claims. This task de-risks a month of theory.

**Acceptance:** a distinctness verdict per claim (below), each PROVABLY-DISTINCT / OVERLAPPING / UNCLEAR with the exact mathematical reason; plus a reusable-protocol extraction.

## Sub-tasks
1. **Precise reconstruction of Mo's framework:** the exact setting of Theorem 1 (flow class, equivariance assumptions, what "Lyapunov neutral" quantifies), the pseudo-gap definition (is it a Lyapunov-spectrum object of a *dissipative/contractive* flow? an empirical gap? of what operator?), and the explicit-breaking experimental protocol (how they break, how they measure lifetime, the corr ≈ 1.0 claim's exact scope).
2. **Distinctness audit vs our F5 claims:** for each of — (a) the **curvature mass-law** (retention ∝ 1/ω², ω² = eig(M_eff⁻¹·∇²V) — a potential-curvature statement), (b) the **Goldstone latch** (γ freezes displacement; q∞ = q0+εp0/(Mγ)), (c) the **overdamped/underdamped crossover + saturation** (half-life saturates at 2ln2/γ — does Mo's lifetime law have any analogous regime structure? if theirs is single-regime, that's a sharp empirical separator!), (d) **kinetic isotropy** (multiplet-mass condition), (e) **constructive allocation** (choosing G/H as a design knob) — state whether Mo's formalism can even *express* the claim, and if partially, exactly where the boundary is.
3. **Protocol mining:** extract their controlled-symmetry-breaking experimental design in enough detail that our V2 experiment can (i) reuse it for comparability and (ii) add the regimes theirs cannot probe (friction sweep γ, curvature sweep, the saturation crossover).
4. **One-page positioning paragraph** (draft prose for the V2 short's related work): how we cite Mo generously and state precisely what is new. Also flag anything in Mo that *contradicts* F5 — if their measured lifetime law conflicts with our exact 2×2 solution in any shared regime, that's a red alert, not a nuance.

**Format:** HEP-colleague-legible markdown (this note will likely be shared with the Manchester collaborator). Honest verdicts — if Mo covers more than the scout believed, say so loudly; better now than at review.
