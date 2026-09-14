# Task: address-space-dimension-scaling — is the 8-item ceiling a 2-D ring artifact or CLU's real capacity? (w20)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/address-space-dimension-scaling.md` · **Branch:** `agent/experiment-engineer/address-space-dimension-scaling`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/clu-retrieval-demo.md` (your own w19 work — this extends it) · `.claude/outputs/clu-memory-architecture.md` (theorist: packing bound + three capacity regimes) · `chlu/core/memory_potentials.py` · `chlu exp-retrieval`
- **Why:** w19 built a working write→address→retrieve loop with **zero learning** and measured a capacity ceiling of **8 items**. Both the engineer and the theorist independently warned that **8 must NOT be quoted as CLU's capacity** — it is a property of the 2-D ring geometry (`K_max ≈ 0.2·2π/σ_θ`), while the theorist's packing bound is `(1+2R/w)^d`, **exponential in latent dimension `d`**.
- **⭐ Why this is the wave's priority:** this single number decides whether CLU is a memory or a curiosity, and it gates the ICLR framing. If capacity grows exponentially (or even strongly polynomially) in `d`, the associative-memory story is real and we can make a capacity claim against attention. If it stays near 8, "handful of items" is a property of the primitive and we take the fallback positioning. **Run this before anything downstream of it.**

## Item 1 — the scaling measurement (the deliverable)
Generalize the w19 hand-designed retrieval demo from the 2-D ring to **d-dimensional latents**, `d ∈ {2, 3, 4, 6, 8, 12, 16}` (extend upward if cheap; stop when compute bites and say where you stopped).

For each `d`, measure the **maximum number of items K_max** that can be written and retrieved at a fixed fidelity criterion. Use the w19 criteria verbatim so the numbers are comparable:
- payload retrieval error (w19 single-item: 9.98e-4)
- **linear-codebook read** (w19: 1.000 at K=2, 0.992 at K=8)
- ⚠ **the blank-landscape control is mandatory on every cell** (w19: 0.469 ≈ chance). A full-state read scores 1.000 on a blank landscape because it reads the *address* back. **Any cell without a passing blank control is not a measurement.**

**Deliverable: `K_max` vs `d`, with the packing-bound prediction `(1+2R/w)^d` overlaid.** State the fitted growth (exponential? polynomial? flat?) and report `R` and `w` as *measured* from the landscape, not assumed.

## Item 2 — separate geometry from dimension
The ring ceiling is an angular-resolution limit. Confirm the mechanism rather than just the trend: at fixed `d`, vary the basin width `w` and confirm `K_max` tracks `(1+2R/w)^d`. If it does not, the packing bound is wrong and that is a more important result than the scaling curve — report it as such.

## Item 3 — which capacity regime are we in?
The theorist identifies three regimes: barrier-protected (100% selectivity) · **washboard death zone (~K=16, selectivity 0.49)** · continuum register (K≥24, written angle retained exactly, selectivity recovers to 0.96). For each `d`, report **selectivity** alongside `K_max` and locate which regime each cell sits in. ⚠ The standing claim is that an *emergent* CHLU is stuck in regime 2 while *designed* structure reaches regime 3 — this task is designed-only, so it should reach regime 3. **If designed structure also lands in the death zone at higher `d`, that falsifies the three-regime picture and is the headline.**

## Item 4 — does dissipation still gate it?
w19: durability is 1.000 out to 1200 steps at γ>0 but degrades to **0.813 at γ=0** — retrieval *requires* dissipation. Confirm this holds at `d > 2`, and report whether the required γ scales with `d`. (Relevant to the addressing redesign: γ helps addressing and provably kills address gradients, so knowing where γ must sit is load-bearing.)

## Acceptance
The `K_max`-vs-`d` curve with blank controls passing on every cell, the fitted growth law vs the packing-bound prediction, per-`d` selectivity with regime assignment, and the γ dependence. Tests green.

⚠ **Report the honest ceiling.** If capacity does not grow with `d`, say so — that is a program-level result and we need it more than we need a good number. Do not tune the landscape per-`d` to flatter the curve; if you tune, report the tuning and show the untuned curve alongside.
