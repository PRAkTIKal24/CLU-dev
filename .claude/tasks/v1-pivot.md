# Task: v1-pivot — V1's new headline: calibrated energy-gated compute allocation

- **Agent:** `experiment-engineer` · **Base:** `main` · **Branch:** `agent/experiment-engineer/v1-pivot` · **Output:** `.claude/outputs/v1-pivot.md` (+ results in `.claude/outputs/v1-pivot/`)
- **Read first:** protocol · `.claude/outputs/v1-l0-gate.md` (the gate evidence + follow-ups 1–5) · scout-adaptive-compute (EBT/CALM positioning) · brainstorm Thread 3.
- **Head decision on file:** V1's headline = **trained per-instance residual calibration + escalation/abstention policy** ("calibrated energy-gated compute allocation on conservative memories"). **Squeeze retries are PARKED** for V3-scale shells (do not develop further here; keep the S^(M) path importable/tested — we own the mechanism).

**Goal:** the pivoted V1 short's core evidence, statistically powered.

## Build + run
1. **Learned τ (the Thread-3 training-time calibration objective, now empirically mandated — pooled raw AUROC was 0.33):** implement a calibration head trained *with* the EBM: per-model margin objective such that residual R = H(settled) − floor ranks correctness, plus a learned threshold τ (start simple: temperature/affine calibration of R fitted on a held-out calibration split during training; note what generalizes). Optional but encouraged: **2-feature gate** g(R, readout-margin) — the gate evidence showed margin ≥ energy at kv≤32; a combined gate is the honest strongest instrument, and "energy adds information over margin" (or doesn't) is a reportable result either way.
2. **Statistical powering:** kv ∈ {16, 24, 32}, **≥5 seeds each**, enough episodes that recovery/abstention deltas get error bars (gate run had n=8 wrongs at kv=16 — fix that). Reuse `exp_v1_gate` machinery; extend, don't fork.
3. **Abstention head-to-head (the anti-Hopfield experiment, gate follow-up 4):** selective-prediction evaluation — **risk–coverage curves** for (a) CLU + learned τ/gate, (b) modern Hopfield + naive confidence (max softmax / logit margin over stored patterns), (c) Hopfield + temperature-scaled calibration (be fair to the baseline), (d) entropy-gated CLU. Metrics: AURC, coverage at fixed risk (e.g., 95% precision), ECE of each confidence signal. **The claim to test: the conservative-dynamics energy gate abstains/allocates better than an always-answering associative memory dressed with confidence.** If it doesn't, report it — that kills the short honestly.
4. **Compute-allocation curve (the 8× result, powered):** τ-gated cascade cost-vs-accuracy across difficulty, mean±std over seeds, vs always-small/always-full/entropy-gated.
5. **CALM wrapper (stretch):** Learn-then-Test-style distribution-free wrapper on the learned τ for a deployment-guarantee paragraph.

## Rules
Def-2 nomenclature. Keep training-path changes behind config (house style). pytest for the calibration head. Runs are CPU-scale (~30 min/seed per the gate report) — laptop fine, note runtimes. Report per protocol §5 with the honest read: does the pivoted headline survive powered statistics?
