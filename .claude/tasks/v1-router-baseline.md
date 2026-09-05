# Task: v1-router-baseline — the boring baseline wormholes must beat + small fixes (critique P9/V1.2)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/v1-router-baseline.md`
- **Read first:** protocol · `.claude/critique_register.md` (V1.2) · `.claude/claims_matrix.md` **CM-7 (the wormhole headline is BLOCKED until this baseline exists) + CM-2/CM-3** (no energy-superiority claims) · `.claude/outputs/v1-wormhole-routing.md` (apparatus + the queued impostor-composition study).
- **Git:** branch `agent/experiment-engineer/v1-router-baseline` — **worktree MANDATORY** (`v3-band-selection` runs concurrently).

## Items
1. **Learned-router-MLP arm:** a 2-layer MLP on the query embedding deciding when/where to route — same sparsity budget as the gated wormhole, no physics. Run the full w4 routing battery (N=4, 8) against all existing arms.
2. **Power + workload realism:** ≥5 seeds (w4 ran 2); workload mixes {50/50, 80/20, 95/5} local-vs-nonlocal; report per-mix.
3. **Honest cost accounting:** replace unit-steps with a FLOPs model (potential evals × dims + gate/router overhead incl. the router MLP's own cost); re-state the "1-hop flat vs chain-diffusion-scales-with-N" result in FLOPs.
4. **Impostor-composition study at N=8** (queued from w4): why does the calibrated head over-route local? Composition of the calibration probe set vs deployment mix.
5. **Small fixes (fold-in):** (a) forward `gate=tcfg.friction_field_gate` in `exp_s1_gamma_field.py` (~lines 361, 405 — exact edit in `v2-prefreeze-baselines.md` §3); (b) dedupe the F811 duplicate `class ExperimentV1WormholeConfig` in `config.py` (identical definitions, keep one); (c) add the config round-trip test if the dedup touches load/save.

**Acceptance:** the CM-7 verdict sentence: *"the energy-gated wormhole beats/matches/loses-to a parameter-matched learned router at [conditions], in FLOPs, ≥5 seeds."* Either outcome publishable — if the router wins, V1's routing pillar reframes to certificates/interpretability per CM-2 logic (say so explicitly). Charter C-9: negatives fully written. Flag-provenance per §5.
