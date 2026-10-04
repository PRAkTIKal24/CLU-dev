# Task: learned-landscape-write-read — does the w19 memory loop survive a LEARNED landscape? (w20)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/learned-landscape-write-read.md` · **Branch:** `agent/experiment-engineer/learned-landscape-write-read`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/clu-retrieval-demo.md` (your own w19 work — this is its direct successor) · `.claude/outputs/clu-memory-architecture.md` (theorist) · `chlu/core/memory_potentials.py`
- **⭐ Why this is the highest-value untested claim in the program:** w19 demonstrated the full write→address→retrieve loop with **zero learning** — the landscape was hand-designed, and **write locality came from designed additive separability + exact symmetry, not from emergence.** The program's vision requires the landscape to be *learned* (designed where design is needed, loss-refined elsewhere). **Nothing tells us the loop survives that transition**, and the proposed addressing repair (below) rests entirely on it. This task can kill or validate the architecture cheaply.

## The claim under test
The Hub's proposed addressing mechanism is: **the address of an item is derived, not chosen** — write item `c` at the location `c` itself relaxes to under the γ>0 dynamics, so that reading = relax the query under the same dynamics and land on the address. Retrieval then splits into two phases with different γ:
```
query   → [γ>0 dissipative relaxation] → address (m*, q₀*)     ← no gradient needed
address → [γ=0 conservative rollout]   → trajectory → readout   ← gradients safe here (Prop 4)
```
**The load-bearing assumption is: relaxation lands where the writer wrote.** That is true by construction in the w19 demo (additive separability). **It is untested under a learned `V_θ`, and if it is false the mechanism does not close.**

## Item 1 — the decisive test
Train `V_θ` (rather than hand-designing it) on a small set of `K` items, then measure, for each item:
1. **Write→relax consistency:** write item `i` at address `a_i`; relax from a *perturbed* query near `c_i`; does it converge to `a_i`? Report **basin-level** and **strict** (address recovered to the w19 payload tolerance) success rates separately — w19 showed these diverge sharply (theorist Toy D: 9/18 basin-level vs 2/18 strict).
2. **Retrieval fidelity** under the learned landscape vs the w19 hand-designed baseline (payload 9.98e-4, codebook read 1.000 @ K=2, 0.992 @ K=8).
3. ⚠ **Blank-landscape control on every cell, mandatory** (w19: 0.469 ≈ chance). A full-state read scores 1.000 on a blank landscape because it reads the *address* back. **Any cell without a passing blank control is not a measurement.**
4. **Durability** — w19 got 1.000 at every read position out to 1200 steps. Does the learned landscape hold that, or does retrieval decay back into a trace?

## Item 2 — how much design does the learned landscape need?
This is the program's central question, restated concretely (Head: *"designed where learning is weak or arbitrary"*). Sweep the **amount of designed structure** imposed on `V_θ`, from fully hand-designed (w19) to fully free, with intermediate rungs (e.g. designed basin skeleton + learned residual; designed separability + learned depths; symmetry imposed vs learned).

**Deliverable: the fidelity-vs-design-freedom curve, and the identification of the minimum designed structure that preserves the loop.** That number is a program-level result whichever way it comes out.

⚠ **Do not assume structure emerges.** The program's strongest negatives say it does not: **N46** (emergent has no coset register), **N7/CM-5** (mass hierarchy must be designed in), **CM-16a** (designed-only split), theorist **D1** (rich-gradient levers absorb the signal; slow levers freeze) and **D3** (exact structure is measure-zero). If a fully-free `V_θ` produces a working loop, that **contradicts four of our own measurements** and must be reported as an extraordinary claim with extraordinary checking, not as a success.

## Item 3 — does additive separability survive learning?
w19's write locality (permanent-item corruption by a decaying write measured at **4.17e-7**, i.e. exactly zero) came from designed additive separability. Under a learned `V_θ`, measure **cross-write interference** directly: write item A, then write item B, then re-read A. Report the corruption. **If interference appears, locality was an artifact of the design and the "write near without disturbing" vision element is not yet real.**

## Item 4 — γ, restated as a two-phase question
w19 found retrieval requires dissipation (0.813 at γ=0 → 1.000 at γ>0) while Prop 5 found dissipation destroys address gradients (‖∇‖ falls 7 orders). The proposed resolution is that these are **different phases wanting different γ**. Test it: measure retrieval fidelity as a function of (γ_address, γ_read) as a **2-D** sweep rather than a single γ. **The prediction is that the good region is off-diagonal — γ_address > 0, γ_read ≈ 0.** Confirm or refute.

## Acceptance
Items 1–4 with blank controls passing throughout, the design-freedom curve with its minimum-viable-design point, the interference number, and the 2-D γ map. Tests green.

⚠ **This task is allowed to return "the mechanism does not survive learning."** That is a valid and valuable result — it redirects the architecture early and cheaply. Do not tune until it works and then report the tuned configuration as the finding; report the honest success rate and what it took to get there.
