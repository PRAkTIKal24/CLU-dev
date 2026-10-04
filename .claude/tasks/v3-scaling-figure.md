# Task: v3-scaling-figure — generate V3's headline O(N)-vs-O(1) scaling-curve figure (w9, micro)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/v3-interference-ntk/fig_scaling_curve.png` + a 10-line note in `.claude/outputs/v3-scaling-figure.md` (what's plotted, from which JSONs, error bars).
- **Read first:** protocol · `v3-interference-ntk.md` item 3 (the table: modular S ≈6.8e-5→1.7e-4 vs monolith 0.635→1.384 for N=4→8, 3 seeds) + its data files `.claude/outputs/v3-interference-ntk/{interference_init,through_training}.json`.
- **The figure:** per-unit received interference S_B vs N (log-y), two curves (modular ≈flat O(1) at ~1e-4; monolith ≈linear O(N) crossing S=1 self-signal line at N=8), seed error bars, the S=1 "exceeds own signal" reference line labeled. Style per the existing v3 figure conventions. This is the V3 short's headline Fig 1 (v3-revision-2 embeds it).
- Repo read-only; if the JSONs lack what the table shows, regenerate the two N-points from the existing scratch scripts (cheap) rather than guessing.
