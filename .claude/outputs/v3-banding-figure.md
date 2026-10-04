# v3-banding-figure — results-analyst report
Task+accept: render V3 Fig 2 (§3.4 banding panel) from band-selection scratch; numbers must match v3-band-selection.md/CM-11 exactly, no new claims. **Status: done.**
Artifact: `.claude/outputs/v3-band-selection/fig_banding.png` (150 dpi); gen script `.claude/scratch/v3-band-selection/fig_banding.py` (reads item1/item2 JSONLs; repo untouched, read-only).
Figure: log-y bar plot, eval-MSE @300ep, 5 seeds ±std — matched **1.180±0.216** (green) < uniform **2.416±0.345** (grey) < orthogonal **6.924±1.883** (orange) < anti **12.791±0.698** (red); FFT-selector diamond overlaid on matched bar, annotated "selector = oracle, gap 0.000".
Provenance: values recomputed from `item{1,2}_results.jsonl` (budget=300); population std used (matches report ±). Selector–oracle gap = 0.000 (5/5 seeds). Reproduces v3-band-selection.md tables bit-for-bit; underlying runs = commit `c4bc004`, seeds {0..4}, mass_lr defaults, JAX 0.10.2 worktree (numerics version-stable per that report).
Log-y chosen because anti (12.8) compresses the sub-3 arms; style follows v3-interference-ntk figs (log bars, value annotations, error bars, good=green/bad=red).
No new claims; asset only. This is V3 Fig 2 for v3-revision-3 to embed.
## Proposed handover updates (for the Hub)
- V3 Fig 2 asset ready at `.claude/outputs/v3-band-selection/fig_banding.png`; hand to v3-revision-3 for §3.4 embed. No numbers changed vs CM-11.
