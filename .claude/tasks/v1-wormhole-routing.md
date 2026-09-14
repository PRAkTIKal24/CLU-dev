# Task: v1-wormhole-routing — energy-gated test-time routing (wormholes join V1's mechanism suite)

- **Agent:** `experiment-engineer` · **Base:** `main` (needs the merged lattice) · **Branch:** `agent/experiment-engineer/v1-wormhole-routing` · **Output:** `.claude/outputs/v1-wormhole-routing.md`
- **Read first:** protocol · F5 §7.4–§7.5 (GatedCoupling, Def-7 cascade, Λ shell-lift = Open-6) · `chlu/core/lattice.py` (GatedCoupling exists — build on it) · `.claude/outputs/v1-pivot.md` (the gate machinery) · brainstorm Thread 3.
- **Head decision (2026-07-06):** V1 is the *inference-time mechanisms* vertical — calibrated gate (proven) + certified retries (parked) + **wormhole routing (this task)**. This experiment is V1's third pillar and the first test of F5 Def-7's escalation beyond a single shell.

**Goal:** demonstrate **energy-gated sparse non-local routing at test time**: when a query relaxes badly in its local unit (high residual R), an energy-gated wormhole edge opens a path to a distant unit that holds the answer — retrieval succeeds at a fraction of the cost of dense coupling or long multi-hop relaxation.

## Build/run
1. **Setup:** a chain lattice of N=4 (then 8) units, each an associative memory over its own key subset (reuse the MQAR/EBM write machinery per unit, small dims); ONE gated wormhole edge between the query unit and the distant "archive" unit (`GatedCoupling`, smooth energy gate — gate opens as a function of the local residual/mismatch energy). Queries arrive at unit 0; some answers live locally, some only in the archive.
2. **Arms:** (a) local-only relaxation (no route — ceiling on local, floor on distant queries); (b) **gated wormhole** (the mechanism); (c) always-open dense edge (cost control — same connectivity, no gate); (d) chain-only multi-hop relaxation (information must diffuse through intermediate units); (e) the calibrated τ-gate composed on top (escalate-to-wormhole only when R > τ — the full Def-7 story minus squeezes).
3. **Measures:** retrieval accuracy split by local-vs-distant queries; steps/FLOPs to answer; **gate-opening statistics** (does the gate open selectively for distant-answer queries? = the routing-selectivity plot); energy injected through the open gate (bounded, per F5 §7.4 smooth-gate claim); calibration of R as the routing signal (AUROC: R → "answer is non-local").
4. **The headline plot:** cost-vs-accuracy for the 5 arms + the gate-selectivity confusion matrix. Claim to test: *gated sparse routing ≈ dense accuracy at ≪ dense cost, ≫ local-only accuracy* — attention-like long-range access priced in energy.
5. **Tests:** gate monotonicity in the driving energy; closed-gate ⇒ bit-equal to uncoupled units (extend the existing κ_c=0 reduction test to the gated edge); routing smoke.

**Scope guards:** smooth gates only (no top-k selection — keeps everything conformally symplectic, no energy ledger needed); squeeze retries stay parked (do NOT bundle); Λ shell-lift (Open-6) stays open — this is routing *within* one lattice, not shell-jumping between models. 2–3 seeds; laptop-scale. Honest reporting if the gate doesn't learn/route selectively — parameterization notes welcome.
