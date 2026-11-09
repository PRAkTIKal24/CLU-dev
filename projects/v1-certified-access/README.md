# V1 — Certified Access (LCFM short)

Experiment data and figure-generation scripts for the "Certified Access" workshop short,
accepted at the NeurIPS 2026 Long-Context Foundation Models workshop.

## Reproducing figures

```bash
cd projects/v1-certified-access/plots
python -I new_fig1.py        # fig1_certificate.png
python -I new_fig4.py        # fig4_bibo.png
python -I new_frontier.py    # fig_frontier_clean.png
```

Output goes to `plots/new_renders/` by default; override with `REPLOT_OUT=<dir>`.

Dependencies: numpy, matplotlib (no chlu package required).

## Data

`data/paid_access_metrics.json` — metrics from the certified-access paid-read experiments (3 seeds, 4 K_v, 200 epochs).

`data/frontier_exact.npy` — exact float64 frontier values recovered from the 6-decimal table; tap hashes match the banked figure digest.
