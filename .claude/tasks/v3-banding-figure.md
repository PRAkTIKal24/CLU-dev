# Task: v3-banding-figure — render the missing V3 banding figure (w10, micro)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/v3-band-selection/fig_banding.png` + a 8-line note `.claude/outputs/v3-banding-figure.md`.
- **Read first:** protocol · `v3-band-selection.md` items 1–2 (the data: degradation curve matched 1.180 < uniform 2.416 < orthogonal 6.924 < anti 12.791 @300ep, 5 seeds; FFT selector gap 0.000) + its scratch JSONs.
- **The figure:** V3 §3.4's banding panel — eval-MSE bar/point plot of the four mis-banding arms (matched / uniform / orthogonal / anti) at 300 ep with 5-seed error bars, + the FFT-selector point overlaid on the matched bar (annotate "selector = oracle, gap 0.000"). Log-y if the anti-band (12.8) compresses the others. Style per existing v3 figure conventions. This is V3 Fig 2 (v3-revision-3 embeds it; v3-referee reviews the claim not the asset).
- Repo read-only; regenerate from the band-selection scratch results (`.claude/scratch/v3-band-selection/item{1,2}_results.jsonl`) if a direct plot helper doesn't exist. Numbers must match `v3-band-selection.md` / CM-11 exactly — no new claims.
