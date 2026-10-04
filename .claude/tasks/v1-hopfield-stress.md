# Task: v1-hopfield-stress — map where CLU-vs-Hopfield actually trades (Head decision 1b)

- **Agent:** `experiment-engineer` · **Base:** `main` · **Branch:** `agent/experiment-engineer/v1-hopfield-stress` · **Output:** `.claude/outputs/v1-hopfield-stress.md`
- **Read first:** protocol · `.claude/outputs/v1-pivot.md` (findings 3 + follow-ups 1, 4) · `exp_v1_calibration` machinery (extend, don't fork).
- **Head framing:** the goal is NOT to hide Hopfield's dominance — it's to chart the regime map so a reader knows *when to build with CLU and when to stick with Hopfield*. Honest crossovers are the deliverable; "Hopfield wins everywhere we could reach" is a valid (reportable) outcome.

## Build/run
1. **Stress axes (pick the 2 most informative after a pilot):** (a) **capacity stress**: kv ≫ current (64→512) at fixed pattern dim d (Hopfield's β-limited separation should degrade — track its known capacity scalings); (b) **noisy evaluation cues** (σ on the query embedding — both systems degrade; who degrades *gracefully* and whose confidence signal stays calibrated?); (c) **correlated/clustered keys** (reduced separation — classic Hopfield failure mode); (d) memory-budget parity (Hopfield stores all patterns explicitly = O(kv·d) memory; CLU compresses into fixed θ — compare at matched *parameter/memory* budget, not matched content).
2. **For each stressed regime:** accuracy + AURC + coverage@risk for CLU-with-learned-gate vs Hopfield-with-calibrated-confidence (fair: Platt-fit Hopfield's margin as in v1-pivot); 3–5 seeds.
3. **The deliverable figure: a regime map** — axes = stress dimensions, contours/regions = "Hopfield dominant / comparable / CLU-gate advantage (accuracy, calibration, or abstention)". Plus the compute-allocation curves in the stressed regimes (does the 4.8× hold when the task is hard everywhere?).
4. **Honest framing paragraph** drafted for the short: where the crossover is (if found), what drives it, and the practical guidance sentence.

**Scope guards:** reuse exp_v1_calibration; new stress knobs into the `calib_*`/gate config group; laptop-scale (pilot first, report runtimes; defer to CSF3 only if a cell exceeds ~1h). If no crossover is reachable at laptop scale, say so with the trend lines — that IS the result.
