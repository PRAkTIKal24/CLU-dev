# Task: relaxation-addressing-theory — is "addressing = relaxation, retrieval = rollout" sound? (w20)

- **Agent:** `physics-theorist` · **Output:** `.claude/outputs/relaxation-addressing-theory.md` · **Branch:** none (no production code)
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/clu-memory-architecture.md` (your own w19 work — Props 2/4/5/6, Toy D, the three capacity regimes) · `.claude/outputs/clu-retrieval-demo.md` (engineer's measurements) · `.claude/outputs/scout-dynamical-memory-priorart.md` (Ramsauer / Hopfield-attention equivalence, HiPPO, Kong 2024)

## The proposal under test (Hub's, offered for you to accept or refute)
w19 established: gradient address search is **dead** (4 protocols, 2 agents, independently), your **Prop 5** shows dissipation kills address gradients (loss frozen to 7 significant digits over 4000 steps; ‖∇‖ down 7 orders), and your **Prop 4** shows the γ=0 Verlet tangent map is symplectic ⇒ `‖J_N‖ ≥ 1` always, so a **conservative** read has no vanishing-gradient problem ever.

The Hub proposes these are not a contradiction but a **phase separation**:
```
query   → [γ>0 dissipative relaxation] → address (m*, q₀*)     ← forward only; never differentiated through
address → [γ=0 conservative rollout]   → trajectory → readout   ← gradients flow; Prop 4 guarantees ‖J‖ ≥ 1
```
with the write rule **deriving** the address: write item `c` at the location `c` relaxes to under the γ>0 dynamics, so read = relax the query and land on the address. The learned selector then exists only as a **fast amortization** of the relaxation, trained by plain regression on write-time `(content, address)` pairs — **no backprop through the rollout anywhere.**

Composed with **Prop 6**: put the *learnable* part of the address in the **mass** (ratios are wake-visible and exactly learnable) and the *derived* part in **position** (where gradients die). ⇒ mass = learned coarse selector, `q₀` = relaxation-derived fine address.

## Item 1 — does the phase separation actually evade Prop 5?
The Hub's claim is that Prop 5 is **routed around structurally** rather than defeated, because the dissipative segment is forward-only. **Check this rigorously.** Specifically: the write rule *defines* addresses via the γ>0 relaxation, and `V_θ` is learned — so does a gradient path from the loss back to `V_θ` **through the address definition** re-introduce the differentiate-through-dissipation problem by the back door? If it does, the proposal fails as stated and needs a stop-gradient or an implicit-function treatment — say which.

## Item 2 — the candidate no-go, sharpened or refuted
You proposed: *a dynamically-robust addressable memory has no useful address gradient* — because retrieval robustness **is** `∂(final)/∂q₀ → 0`, exactly the signal address-descent needs. **Either promote this to a stated proposition with a proof and its precise scope, or refute it.**
⚠ **Scope matters enormously.** If it holds only for *gradient search over `q₀`*, the two-phase proposal survives it cleanly. If it holds for *any* learned addressing including regression-amortized, the architecture has a much deeper problem. **Determine which.** Note the Head's weakening: we need directions usefully descending *over epochs*, not one-shot address recovery — re-examine under that weaker requirement too.

## Item 3 — does relaxation land where the writer wrote?
The load-bearing assumption. True by construction in w19 (designed additive separability + exact symmetry). **State the condition on `V_θ` under which the relaxation fixed point of a query coincides with the write location**, and say whether that condition is generic, measure-zero (your D3), or designable. ⚠ The engineer is testing this empirically in `learned-landscape-write-read.md` — **derive the prediction before reading their result if it has landed**, and say what would falsify you.

## Item 4 — capture by annealing, not by descent
w19's failure was **capture** (cross-basin re-localization): strict success is an *energy* criterion requiring GD to shed energy *while positioned over the target*, across fractal separatrices. A **damped particle sheds energy while moving** — the natural solver. The Hub proposes **retry = Lorentz boost + re-relax**, i.e. annealing in the **physical energy**, distinct from w19's "energy-annealed 0/18" which annealed the **loss**.
**Analyze this.** Does boost-and-re-relax have the right acceptance structure to escape a wrong basin without destroying a correct one? Relate to your `M* = p₀²/2h` confinement threshold: **a boost that enables escape necessarily crosses `M*` — so the same threshold that gives Prop 2 its provable read isolation also prices retry.** Is there a usable window, or does retry destroy isolation?

## Item 5 — the Ramsauer objection, which we will face
If addressing reduces to relaxation to a fixed point, it is Hopfield-like — and **Ramsauer et al. showed attention IS Hopfield retrieval**. A referee will ask what the dynamics adds over attention. **The Hub's answer, which needs your rigor or your rejection:** Hopfield returns a *fixed point*; CLU returns the *whole trajectory from it*, with mass setting the timescale and **Prop 2 giving provable read isolation that softmax structurally cannot have** (softmax mixes every key; a sub-barrier particle *cannot* read a foreign item).
**Is that a real distinction or a restatement?** Formalize what information the trajectory carries that its endpoint does not. ⚠ If the answer is "nothing that matters," we need to know now.

## Item 6 — the HiPPO question (the primitive-level competitor)
Under the "CLU is a general primitive" framing, the sharpest competitor is **HiPPO-LegS / S4 / Mamba, not attention**. HiPPO-LegS is **exactly timescale-equivariant, has no `dt` at all**, and carries a provable retention guarantee with `Θ(1/t)` **polynomial** (not exponential) gradient decay. **A referee will ask: why does a learned mass spectrum beat provable timescale invariance? We currently have no answer.**
The Hub's conjecture: **the answer is addressability** — HiPPO compresses history into a single state with no notion of separate, isolated, individually-retrievable items, while Prop 2 says CLU has exactly that. **Make this rigorous or kill it.** If rigorous, it is a central claim of the flagship and needs to be stated at the right strength — including the honest cost, since a fair comparison must say what HiPPO's guarantee buys that ours does not.

## Acceptance
Items 1–6, each labelled **proven / verified-numerically / conjectured**, with small numerical checks where they settle something. A ranked list of which results are load-bearing for the architecture and which are commentary.

⚠ **Refuting the Hub's proposal is a fully acceptable outcome and should not be softened.** The Hub has been wrong twice in this program and was corrected by agents who checked. **Do not quote:** the "8-item ceiling" as CLU's capacity · "attention has no retention analogue" (HiPPO does) · "retrieval-as-rollout" as novel (Kong et al., *Nature Communications* 2024) · continuity as our NTM/DNC escape hatch (canonical NTM **and** DNC were already fully soft and end-to-end differentiable).
