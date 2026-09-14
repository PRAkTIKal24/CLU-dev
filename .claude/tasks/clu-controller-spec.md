# Task: clu-controller-spec — what conditions must the CLU controller meet, and what is the minimum viable one? (w20)

- **Agent:** `physics-theorist` · **Output:** `.claude/outputs/clu-controller-spec.md` · **Branch:** none (no production code — flag changes for the engineer)
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/handover_context.md` §"THE HEAD'S ARCHITECTURAL VISION" (2026-07-21) and the Running Log entries of 2026-07-21 · `.claude/outputs/clu-memory-architecture.md` (your own w19 work) · `.claude/outputs/clu-retrieval-demo.md` · `.claude/negative_results.md` (N46, N7, N12, N13) · `.claude/claims_matrix.md` (CM-5, CM-16a)

## Why
**Every verb in the program's vision is a controller verb**: *decide* which basin, *triage* what is permanent vs decaying, *add to an existing basin / open a new one / trash an old one*, *select* the mass, *combine* the retrieved pieces. We have built or theorized nearly every **mechanism** (latch, pseudo-Goldstone half-life, SSB vacua, γ_φ trash field, mass-as-budget, boost ladder, wormholes, `CLULattice`) and have **never built a controller**. This is the load-bearing absence in the architecture — larger than capacity or the read-in.

## The hard constraint you must respect
Your own **T3**: permutation-symmetric regularizers restore *diversity* but **can never choose *assignment***. ⇒ **The write-assignment policy cannot emerge from a regularizer.** It must be an explicit mechanism, or an arbitrary-but-fixed gauge (**T2**: a wake-null parameter is pinned at zero cost). The Head's independent formulation: *"allow arbitrary choices for the location of a register or masses if these parameters don't affect performance much, instead of infinitely restructuring."*

Also binding: **D1** (rich-gradient levers absorb the signal; slow levers freeze — and D1 bites even at equal lr, because it is about gradient *richness*), **D3** (exact structure is measure-zero), and the four emergence negatives **N46 / N7 / CM-5 / CM-16a**. The Head's ruling: *"learning yes, but designed where learning is weak or arbitrary."*

## Item 1 — the conditions (the Head asked for this explicitly)
Derive and state the conditions a CLU controller **must** satisfy. At minimum address:
- **Consistency:** the read path must land where the write path wrote. State this as a formal condition on the landscape and the relaxation map, and say what it requires of `V_θ`.
- **Non-degeneracy vs correctness:** per T2/T3, the address assignment need only be non-degenerate, not correct. **Formalize "non-degenerate enough to be functional"** — what is the precise condition distinguishing a usable arbitrary assignment from a broken one?
- **Stability under rewrite:** adding item `K+1` must not corrupt items `1..K`. w19 measured designed locality at 4.17e-7 (exactly zero) — state the condition that guarantees it and whether it survives a learned landscape.
- **Anti-thrashing:** a deadband/hinge (your own earlier proposal) implements the Head's request directly. Specify it — what is the deadband on, and what sets its width?
- **Capacity admissibility:** given your three regimes (barrier-protected / washboard death zone ~K=16 selectivity 0.49 / continuum register K≥24), what must the controller do to keep the system OUT of regime 2?

## Item 2 — the three decisions, mechanised
For each of **write-to-existing basin / open a new basin / trash an existing basin**, specify a concrete decision rule in terms of quantities CLU actually has (energies, barrier heights `h`, the confinement threshold `M* = p₀²/2h`, relaxation residuals, `μ²`, γ_φ). **State the trigger, the action, and the failure mode.** Include: what makes an item "correlated with" an existing one (⇒ write nearby with a half-life) versus "essential but uncorrelated" (⇒ fresh location, new permanent mode)?

⚠ **Trash is the least developed and needs the most care.** The γ_φ friction field exists as a *mechanism*; deciding to trash is a *policy*. N12/N13 are the relevant negatives — check them before proposing anything.

## Item 3 — minimum viable controller
Specify the **simplest controller that makes the verbs exist at all** — hand-coded and dumb is explicitly acceptable for the first paper. Deliver it as a spec the engineer can implement directly: inputs, state, decision rules, outputs.

**Head's ruling on scope, binding:** *differentiability and training speed/efficiency are downstream concerns and NOT first-paper requirements.* Do **not** compromise the design to keep it differentiable. **But do state, explicitly and separately, which parts are non-differentiable and what the eventual differentiable relaxation would have to look like** — so the debt is recorded rather than discovered later.

## Item 4 — what the controller CANNOT fix
Per T3 and D1/D3 there will be things a controller cannot rescue. State them. In particular: is there any controller that makes a *fully unconstrained* `V_θ` self-organize into a usable memory, or is designed structure strictly necessary? **If designed structure is provably necessary, say so as a proposition — that is a program-level result and it should be written as one.**

## Acceptance
The conditions (Item 1) stated formally with proofs or explicit conjecture labels, the three decision rules (Item 2), an implementable minimum-viable spec (Item 3) with its non-differentiability debt itemized, and the impossibility statement (Item 4). Small numerical sanity checks where they settle something.

⚠ **Label every claim as proven / verified-numerically / conjectured.** The program has twice had a Hub claim overturned by an agent who checked; that is the desired behaviour. If you think the Hub's two-phase addressing proposal (γ>0 relax to address, γ=0 roll to read) is wrong, **say so** — `relaxation-addressing-theory.md` is testing it in parallel and disagreement between you is a useful signal, not a problem.
