# Task: v1-l0-gate — the L0 boost-retry gate experiment (V1 kill/continue evidence)

- **Agent:** `experiment-engineer` · **Base:** `main` · **Branch:** `agent/experiment-engineer/v1-l0-gate` · **Output:** `.claude/outputs/v1-l0-gate.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md`, **F5 (`.claude/outputs/formalism-note.md`) §5.4 + §7.5 (S^(M) spec, Def-6/7, Prop-12)**, `.claude/outputs/scout-adaptive-compute-prior-art.md` (MQAR spec §3, baselines, EBT positioning), brainstorm Threads 3+5. Def-2 nomenclature throughout.
- **Runs parallel-safe with:** fix-pack-2 and v2-so2-build (worktree per protocol §3.2 recommended — you share `chlu/core/` risk with v2-so2-build; coordinate by scope: you own the squeeze op, they own potentials).

**Goal:** the empirical half of the ≈Aug-7 V1 gate. Two questions: **(Q1)** does residual energy of a relaxed CLU state calibrate with retrieval correctness? **(Q2)** do mass-weighted squeeze retries recover answers plain relaxation misses, at cost ≪ escalating to a bigger model? Honest negative answers kill V1 cleanly — that is a valid outcome.

## Build
1. **`S^(M)` squeeze operator** (in `chlu/core/`, e.g. `transforms.py`): per-pair squeeze S_ζ with rapidity vector ζ, **mass-weighted** S^(M)_ζ = N⁻¹·S_ζ·N, N = diag(M_eff^{1/2}, M_eff^{−1/2}) — F5 §5.4 is exact; raw squeezes are mass-blind (do not ship the naive version except as a comparison flag). Unit test: symplecticity ‖SᵀΩS−Ω‖ ≤ 1e-12, det=1, and position-response ∂q_i′/∂ζ|₀ = p_i/M_eff,i on random states.
2. **MQAR data generator** per the Zoology spec in the scout report (§3): configurable vocab (default 8192 — scale down if needed for CLU dims, document), N ∈ {64…512}, #KV pairs, gap distribution. Match the published task semantics so results are community-legible.
3. **Associative-retrieval CLU setup:** encode key-value pairs as attractors of V_θ (train a CLU as an energy-based associative memory on the MQAR dictionary — generative-PCD path is the natural trainer; keep dims modest, e.g. embed tokens to 16–64d). Retrieval = initialize q at the query embedding, relax under the governor, read out nearest stored value. *This is the first CLU associative-memory implementation — keep it minimal and document design choices; perfection is not the bar, measurability is.*
4. **The cascade loop (F5 Def-7, single shell):** relax → residual R = H(settled) − floor → if R > τ: apply S^(M) with line-searched scalar ζ (a coarse grid + golden-section is fine), re-relax, up to B retries. τ for THIS experiment: sweep it (calibration curve) — the *learned* τ objective is a later task; do not build it here.
5. **Baselines (mandatory, per scout):** (a) modern-Hopfield retrieval (softmax attention over stored patterns — a few lines, matched stored content); (b) entropy/confidence-gated version of the same cascade (gate on readout-distance margin instead of energy); (c) always-relax-longer (same total compute, no boosts) — the "boosts vs just more steps" control that EBT positioning demands.

## Measure (report actual numbers; small scale is fine — this is a gate, not the short)
- **Q1:** AUROC of residual-energy-vs-correctness, per difficulty level (N, #KV). Include reliability-diagram-style binning.
- **Q2:** recovery rate = fraction of initially-wrong retrievals fixed by ≤B boost retries; vs the always-relax-longer control at matched FLOPs/wall-clock; per-mode displacement vs 1/M_eff,i (Thread-5 falsifiable (ii) — one scatter).
- Compute-matched accuracy curves: cascade vs Hopfield vs entropy-gated.

**Gate reading guide (for the Hub, include your own honest read):** PASS needs Q1 AUROC meaningfully >0.5 across difficulties AND Q2 recovery > the matched-compute control. Hopfield beating everything at all compute budgets = expected at small scale but must be reported with the curve shape. Ambiguous results: report as ambiguous.
