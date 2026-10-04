# Task: hopfield-capacity-benchmark — CLU vs modern-Hopfield SOTA on the associative-memory PERFORMANCE benchmark (w22)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/hopfield-capacity-benchmark.md` · **Branch:** `agent/experiment-engineer/hopfield-capacity-benchmark`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/scout-benchmark-targeting.md` §Item 3 #1 + §5.1 (the protocol, the SOTA line, the Ramsauer/Hu-Wu-Liu capacity theory) · `.claude/outputs/address-space-dimension-scaling.md` (our measured `4·2^d`) · `.claude/outputs/relaxation-fiber-capacity.md` (the fiber; the parameter ceiling) · `chlu/core/memory_potentials.py`
- **⭐ This is the ONE external benchmark where a *designed* CLU is admissible** — the scout's top-ranked target — because **nothing is learned on either side** (the Hopfield line writes patterns in closed form; we design a landscape). The w20 "learning destroys everything" blocker does not bind here.

## The framing, per the Head (2026-07-23): PERFORMANCE IS THE DELIVERABLE
Lead with **task performance vs published SOTA**, not with our physics. The physics (Prop 2 isolation, fiber payload, retention control, retry) are *candidate explanations* for a performance edge — test them as such, but **a win we cannot cleanly attribute to a named property is still a win.** Conversely, matching the capacity exponent is **not** a result on its own: exponential capacity has been the modern-Hopfield headline since Demircigil 2017, and Hu-Wu-Liu (NeurIPS 2024) proved it is the optimal-spherical-code scaling. **Do not claim the exponential as novel.**

## Item 0 — get the protocol EXACTLY right (the scout's numbers are SECONDARY)
The scout flagged the half-mask/cosine-0.9 constants and the U-Hop margin as **single-sourced/secondary**. You have Bash + clone authorization; the scout did not. **Clone the actual U-Hop (arXiv:2404.03827, ICML 2024) and/or Ramsauer `hopfield-layers` repositories and match their protocol verbatim:** dataset preprocessing, mask fraction, success criterion (cosine > 0.9 AND/OR sum-of-squared-pixel error), the capacity-vs-#stored sweep, and the Gaussian-noise-robustness sweep. **Report the exact protocol you matched, with the repo commit hash.** A subtly-wrong protocol produces a number we later retract.

## Item 1 — the performance comparison (the deliverable)
Datasets: **MNIST and CIFAR-10** (add Tiny-ImageNet only if cheap). Queries = target with **50% of pixels masked** (match the repo). 
Arms:
1. **Modern Hopfield (dense softmax, Ramsauer)** — the base competitor, from the repo.
2. **U-Hop / a sparse-or-entmax Hopfield variant** if the repo provides it — the SOTA line.
3. **CLU designed register** — write the patterns into a landscape (`BallRegisterPotential` / `AtomStorePotential`), retrieve by the two-phase rollout (γ_address > 0 → γ_read = 0), read the settled pattern.
4. **A trivial baseline** (nearest-neighbor in pixel space) — the floor, so "CLU retrieves" is legible.

**Deliverables:** (a) **retrieval accuracy vs number of stored memories** (the capacity-degradation curve), all arms, on the same axis; (b) **retrieval accuracy vs Gaussian noise level** at fixed load; (c) the **cross-over point** where each method's accuracy falls below criterion. **Report where CLU wins, ties, and loses — per dataset, never averaged.**

## Item 2 — the three differentiators, tested AS PERFORMANCE, not as properties
The scout named three things the Hopfield line lacks. Test each as a *measurable performance advantage*, and drop any that does not show one:
1. **Fiber payload** (theorist Prop 11): store a *payload larger than the pattern dimension* per memory (the local jet), and show retrieval of payload bits the endpoint cannot carry. ⚠ Respect `relaxation-fiber-capacity`: quote fiber bits only with `(σ_read, N, launches)`; the parameter ceiling `B_total ≤ P·b_θ` binds for any *learned* landscape (this task is designed, so it does not bind here — but say so).
2. **Per-item retention control** (μ²): show a *demonstrated capability* Hopfield has no analogue for — e.g. some memories permanent, others decaying on a set schedule, in the same store.
3. **Retry** (codebook-gated boost, 0.665 cross-basin at a 1.38·h friction tax): does a second retry pass *recover* queries the first pass misses, lifting accuracy above the single-pass Hopfield? This is the adaptive-compute angle — an accuracy-vs-compute curve Hopfield cannot draw.

## Item 3 — the honest positioning check
State plainly, in the report: is CLU's capacity curve **at, above, or below** the dense-Hopfield and U-Hop curves? Cite Hu-Wu-Liu (optimal capacity = spherical codes) and frame our `Δ_req`/`d_eff` as the **measured constant and mechanism** in a Hamiltonian setting — the novel part — never the exponential itself.

## Acceptance
The matched protocol (with repo hash), the accuracy-vs-load and accuracy-vs-noise curves against real Hopfield/U-Hop baselines per dataset, the three differentiator tests (each kept only if it shows a performance edge), and the honest at/above/below verdict. Tests green.

⚠ **A loss is informative and must be reported plainly** — but this is the target the scout ranked #1 for winnability, so tune the CLU arm honestly hard (report the budget). ⚠ **Do not hand-tune the landscape per-load to flatter the curve**; if you tune, show the untuned curve alongside.
